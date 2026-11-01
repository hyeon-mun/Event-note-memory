"""Train a sequence-memory flow policy.

loss modes:
  plain  : flow loss with memory always on
  drop   : memory replaced by null with prob p_drop (enables memory guidance)
  paired : L = mean(w * L_mem) + mean(L_null),  w = 1 + lam * relu(sg(L_null - L_mem)) / mean(.)
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from data import EpisodeDataset, LengthBucketSampler, collate, task_table
from model import SeqPolicy

p = argparse.ArgumentParser()
p.add_argument("--tasks", type=str, required=True, help="comma list of task ids, or split name")
p.add_argument("--out", type=str, required=True)
p.add_argument("--mem", default="gru", choices=["none", "stack", "gru", "tf", "event", "todo"])
p.add_argument("--tokens", type=int, default=5)
p.add_argument("--no_proprio_mem", action="store_true")
p.add_argument("--loss", default="paired", choices=["plain", "drop", "paired"])
p.add_argument("--p_drop", type=float, default=0.2)
p.add_argument("--lam", type=float, default=1.0)
p.add_argument("--lam_write", type=float, default=0.05, help="penalty on soft write rate above --write_target (event/todo)")
p.add_argument("--write_target", type=float, default=0.05, help="allowed write rate before penalty")
p.add_argument("--lam_live", type=float, default=0.05, help="penalty on live-slot fraction (todo)")
p.add_argument("--sparse_warmup", type=int, default=5000, help="steps before sparsity penalties start (then ramp over same)")
p.add_argument("--quant", type=str, default="", help="FSQ levels for slot contents, e.g. 5,5,5")
p.add_argument("--mem_src", default="dino", choices=["dino", "cnn", "oracle"])
p.add_argument("--stack", type=int, default=8)
p.add_argument("--find", action="store_true", help="memory-as-query top-down attention readout")
p.add_argument("--goal", action="store_true", help="memory-grounded spatial goal head (hindsight goal supervision)")
p.add_argument("--goal_match", action="store_true", help="goal head = local-key / memory-query matching")
p.add_argument("--mem_bottleneck", type=str, default="", help="FSQ levels of the memory state, e.g. 3,3")
p.add_argument("--goal_rf", type=int, default=11, choices=[1, 11], help="receptive field of match-head keys")
p.add_argument("--lam_goal", type=float, default=1.0)
p.add_argument("--p_tf", type=float, default=0.5, help="prob. of feeding the demo goal (vs. predicted) to the action head")
p.add_argument("--vis", default="cnn", choices=["cnn", "siglip"], help="reactive visual encoder")
p.add_argument("--horizon", type=int, default=8)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--n_sel", type=int, default=16)
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--feat_noise", type=float, default=0.1)
p.add_argument("--n_train_eps", type=int, default=250)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--workers", type=int, default=12)
p.add_argument("--save_every", type=int, default=10000)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
os.makedirs(args.out, exist_ok=True)
tt = task_table()
if args.tasks in ("short", "medium", "long"):
    tids = [k for k, v in tt.items() if v["split"] == args.tasks]
else:
    tids = [int(x) for x in args.tasks.split(",")]
json.dump(vars(args) | {"tids": tids}, open(f"{args.out}/args.json", "w"), indent=1)

ds = EpisodeDataset(tids, episodes=range(args.n_train_eps), horizon=args.horizon,
                    n_sel=10 ** 6 if args.mem_src == "cnn" else args.n_sel,
                    tokens=args.tokens, seed=args.seed)
# proprio normalisation stats over training tasks
allp = np.concatenate([s.state for s in ds.stores.values()])
p_mean, p_std = allp.mean(0), allp.std(0) + 1e-3
dl = torch.utils.data.DataLoader(ds, batch_sampler=LengthBucketSampler(ds.lengths(), args.bs, args.seed),
                                 collate_fn=collate, num_workers=args.workers, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=4)

dev = "cuda"
if args.mem_src == "oracle":   # privileged cue labels (diagnostic upper bound), RememberColor-type tasks only
    from cue_labels import cluster, cue_colors
    ORACLE = {t: torch.tensor(cluster(cue_colors(t), {13: 3, 14: 5, 15: 9, 50: 3, 51: 5, 52: 9}[t])[0]) for t in tids}
model = SeqPolicy(mem=args.mem, mem_tokens=args.tokens, horizon=args.horizon,
                  use_proprio_mem=not args.no_proprio_mem,
                  quant=tuple(int(x) for x in args.quant.split(",")) if args.quant else None,
                  mem_src=args.mem_src, stack=args.stack, find=args.find, vis=args.vis, goal=args.goal, goal_match=args.goal_match,
                  mem_bottleneck=tuple(int(x) for x in args.mem_bottleneck.split(",")) if args.mem_bottleneck else None,
                  goal_rf=args.goal_rf).to(dev)
model.register_buffer("p_mean", torch.tensor(p_mean, device=dev))
model.register_buffer("p_std", torch.tensor(p_std, device=dev))
for t_, s_ in ds.stores.items():
    n_tr = int(sum(s_.ep_len[:args.n_train_eps]))
    model.a_scale[t_] = torch.tensor(np.maximum(np.quantile(np.abs(s_.action[:n_tr]), 0.99, axis=0), 1e-3), device=dev)
ema = copy.deepcopy(model).eval().requires_grad_(False)
nparam = sum(x.numel() for x in model.parameters())
print(f"params {nparam / 1e6:.1f}M  tasks {tids}  episodes {len(ds)}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, s / 1000) * (0.5 * (1 + np.cos(np.pi * min(s, args.steps) / args.steps))))


def shift_aug(x, pad=6):
    """x uint8 (N,H,W,3) -> randomly shifted uint8, per-sample (DrQ-style)."""
    N, H, W, C = x.shape
    xp = F.pad(x.permute(0, 3, 1, 2).float(), (pad,) * 4, mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (N,), device=x.device); oy = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    ar = torch.arange(H, device=x.device)
    rows = (oy[:, None] + ar[None])[:, :, None].expand(N, H, W)
    cols = (ox[:, None] + ar[None])[:, None, :].expand(N, H, W)
    bidx = torch.arange(N, device=x.device)[:, None, None].expand(N, H, W)
    out = xp.permute(0, 2, 3, 1)[bidx, rows, cols]
    return out.to(torch.uint8)


def flow_loss(model, b):
    B, Tm = b["feat"].shape[:2]
    prop = (b["proprio"] - model.p_mean) / model.p_std
    feat = b["feat"]
    if args.feat_noise > 0:
        feat = feat.float() + args.feat_noise * torch.randn_like(feat, dtype=torch.float32) * feat.float().std()
    smask = b["smask"]
    bi, li = smask.nonzero(as_tuple=True)
    ti = b["sel"][bi, li]
    top, wrist = shift_aug(b["top"][bi, li]), shift_aug(b["wrist"][bi, li])
    react = model.reactive(top, wrist, prop[bi, ti], b["text"][bi])
    if args.mem_src == "oracle":
        lab = torch.stack([ORACLE[int(t)][int(e)] for t, e in zip(b["tid"], b["ep"])]).to(dev)
        mem_x = F.one_hot(lab, 16).float()[:, None].expand(B, Tm, 16)
    elif args.mem_src == "cnn":       # every step is selected: memory runs over the augmented reactive embeddings
        mem_x = torch.zeros(B, Tm, react.shape[-1], device=dev, dtype=react.dtype)
        mem_x[bi, ti] = react
    else:
        mem_x = model.mem_inputs(feat, prop, b["text"])
    mem, _ = model.memory(mem_x)                                         # (B,Tm,d)
    m_sel = mem[bi, ti]                                                  # (N,d)
    logs = {}
    l_goal = 0.0
    if args.goal:                     # ground memory into the hindsight goal pixel (un-shifted frames: pixel targets)
        logits = model.goal(b["top"][bi, li], b["wrist"][bi, li], m_sel)
        gt = b["goal"][bi, li]
        l_goal = F.cross_entropy(logits.float(), model.goal.cell(gt), label_smoothing=0.05)
        pred = model.goal.uv(logits.detach())
        tf = torch.rand(len(gt), device=dev) < args.p_tf
        model._goal_uv = torch.where(tf[:, None], gt, pred)
        logs["l_goal"] = l_goal.item(); logs["goal_err_px"] = (pred - gt).norm(dim=-1).mean().item()
    x1 = b["act"][bi, li] / model.a_scale[b["tid"][bi]][:, None]     # normalised chunk (N,H,7)
    valid = b["act_valid"][bi, li].float()                               # (N,H)
    N = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.sigmoid(torch.randn(N, device=dev))                        # logit-normal flow time
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    def per_sample(cond):
        v = model.head(xt, t, cond)
        return (((v - target) ** 2).mean(-1) * valid).sum(-1) / valid.sum(-1).clamp(min=1)

    ones = torch.ones(N, dtype=torch.bool, device=dev)
    if args.loss == "plain":
        lm = per_sample(model.cond(react, m_sel, ones)); loss = lm.mean()
    elif args.loss == "drop":
        keep = torch.rand(N, device=dev) > args.p_drop
        lm = per_sample(model.cond(react, m_sel, keep)); loss = lm.mean()
    else:
        lm = per_sample(model.cond(react, m_sel, ones))
        ln = per_sample(model.cond(react, m_sel, ~ones))
        gain = (ln - lm).detach()
        g = gain.clamp(min=0)
        w = 1 + args.lam * g / (g.mean() + 1e-6)
        loss = (w * lm).mean() + ln.mean()
        logs["gain"] = gain.mean().item(); logs["gain_pos_frac"] = (gain > 0).float().mean().item()
        logs["l_null"] = ln.mean().item()
    logs["l_mem"] = lm.mean().item()
    loss = loss + args.lam_goal * l_goal
    if args.mem in ("event", "todo"):
        aux = model.memory.net.aux
        ramp = min(1.0, max(0.0, (step - args.sparse_warmup) / args.sparse_warmup))
        over = torch.relu(aux["write_rate"] - args.write_target)
        loss = loss + ramp * (args.lam_write * over / args.write_target + (args.lam_live * aux["live"] if args.mem == "todo" else 0))
        logs["write_rate"] = aux["write_rate"].item(); logs["live"] = aux["live"].item()
        logs["hard_writes"] = model.memory.net.last["write"].mean().item()
    return loss, logs


step, t0, n_bad = 0, time.time(), 0
logf = open(f"{args.out}/train_log.jsonl", "a")
for b in dl:
    b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, logs = flow_loss(model, b)
    opt.zero_grad(set_to_none=True)
    bad = not torch.isfinite(loss); gn = torch.tensor(float("nan"))
    if not bad:
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bad = not torch.isfinite(gn)
        if bad and n_bad < 3:
            print("non-finite grads in:", [n for n, q in model.named_parameters()
                                           if q.grad is not None and not torch.isfinite(q.grad).all()][:8], flush=True)
    if bad:
        n_bad += 1; print(f"non-finite loss/grad at step {step}, skipping", flush=True)
        opt.zero_grad(set_to_none=True)
        if n_bad > 200:
            raise RuntimeError("too many non-finite steps")
    else:
        opt.step()
    sched.step(); step += 1
    with torch.no_grad():
        dec = min(args.ema, (1 + step) / (10 + step))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - dec)
        for be, bm in zip(ema.buffers(), model.buffers()):
            be.copy_(bm)
    if step % 100 == 0:
        logs |= dict(step=step, loss=loss.item(), gn=gn.item(), lr=sched.get_last_lr()[0],
                     sps=100 / (time.time() - t0))
        t0 = time.time()
        logf.write(json.dumps(logs) + "\n"); logf.flush()
        if step % 1000 == 0:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in logs.items()}), flush=True)
    if step % args.save_every == 0 or step == args.steps:
        torch.save({"ema": ema.state_dict(), "args": vars(args), "tids": tids, "step": step},
                   f"{args.out}/ckpt_{step}.pt")
        torch.save({"ema": ema.state_dict(), "args": vars(args), "tids": tids, "step": step}, f"{args.out}/last.pt")
    if step >= args.steps:
        break
print("done", flush=True)
