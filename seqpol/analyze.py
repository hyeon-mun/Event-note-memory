"""Offline memory analysis on dataset episodes.

For each timestep t of an episode, estimates the memory gain
    dL(t) = E_{x0,tau}[ ||v(x_tau|o_t, null) - u||^2 - ||v(x_tau|o_t, m_t) - u||^2 ]
with paired noise (same x0, tau for both terms). At the flow optimum this equals
E||E[u|x_tau,o] - E[u|x_tau,o,m]||^2 >= 0: how much memory changes the optimal velocity field.

Also: memory transplant. Replace m_t of episode A by m_t of episode B (same task) and measure how much
the predicted chunk moves toward B's actions (memory_follow) - a counterfactual test of causal memory use.
"""
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F

from data import EpisodeDataset, collate
from evaluate import PolicyRunner  # noqa: F401  (ensures same model construction)
from model import SeqPolicy


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu"); a = ck["args"]
    m = SeqPolicy(mem=a["mem"], mem_tokens=a["tokens"], horizon=a["horizon"], use_proprio_mem=not a["no_proprio_mem"],
                           quant=tuple(int(x) for x in a["quant"].split(",")) if a.get("quant") else None,
                           mem_src=a.get("mem_src", "dino"), stack=a.get("stack", 8), find=a.get("find", False), vis=a.get("vis", "cnn"),
                           goal=a.get("goal", False), goal_match=a.get("goal_match", False),
                           mem_bottleneck=tuple(int(x) for x in a["mem_bottleneck"].split(",")) if a.get("mem_bottleneck") else None,
                           goal_rf=a.get("goal_rf", 11)).to(dev)
    m.register_buffer("p_mean", torch.zeros(8, device=dev)); m.register_buffer("p_std", torch.ones(8, device=dev))
    m.load_state_dict(ck["ema"], strict=False); m.eval()
    return m, a


_ORACLE = {}


def oracle_label(tid, ep):
    if tid not in _ORACLE:
        from cue_labels import cluster, cue_colors
        _ORACLE[tid] = cluster(cue_colors(tid), {13: 3, 14: 5, 15: 9, 50: 3, 51: 5, 52: 9}[tid])[0]
    return int(_ORACLE[tid][ep])


@torch.no_grad()
def episode_gain(m, b, K=16, dev="cuda"):
    b = {k: v.to(dev) for k, v in b.items()}
    prop = (b["proprio"] - m.p_mean) / m.p_std
    T = int(b["T"][0]); sel = b["sel"][0][: T]
    react = m.reactive(b["top"][0], b["wrist"][0], prop[0, sel], b["text"].expand(len(sel), -1))
    if m.mem_src == "oracle":
        lab = oracle_label(int(b["tid"][0]), int(b["ep"][0]))
        mem, _ = m.memory(F.one_hot(torch.tensor([lab], device=dev), 16).float()[:, None].expand(1, T, 16))
    else:
        mem, _ = m.memory(react[None] if m.mem_src == "cnn" else m.mem_inputs(b["feat"], prop, b["text"]))
    msel = mem[0, sel]
    x1 = b["act"][0] / m.a_scale[int(b["tid"][0])]; valid = b["act_valid"][0].float()
    ones = torch.ones(len(sel), dtype=torch.bool, device=dev)
    c_mem, c_null = m.cond(react, msel, ones), m.cond(react, msel, ~ones)
    gains, lm_all = [], []
    for _ in range(K):
        x0 = torch.randn_like(x1); tau = torch.sigmoid(torch.randn(len(sel), device=dev))
        xt = (1 - tau[:, None, None]) * x0 + tau[:, None, None] * x1; u = x1 - x0
        def L(c):
            return (((m.head(xt, tau, c) - u) ** 2).mean(-1) * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        lm, ln = L(c_mem), L(c_null)
        gains.append((ln - lm).cpu().numpy()); lm_all.append(lm.cpu().numpy())
    return np.mean(gains, 0), np.mean(lm_all, 0), sel.cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--n_eps", type=int, default=10); ap.add_argument("--ep_lo", type=int, default=240); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    m, targs = load(a.ckpt, dev)
    out = {}
    for tid in [int(x) for x in a.tasks.split(",")]:
        ds = EpisodeDataset([tid], episodes=range(a.ep_lo, a.ep_lo + a.n_eps), horizon=targs["horizon"], n_sel=10 ** 6,
                            tokens=targs["tokens"])
        curves = []
        for i in range(len(ds)):
            g, lm, sel = episode_gain(m, collate([ds[i]]))
            curves.append(dict(gain=g.tolist(), l_mem=lm.tolist()))
        out[tid] = curves
        allg = np.concatenate([c["gain"] for c in curves])
        L = max(len(c["l_mem"]) for c in curves)
        prof = [np.mean([c["l_mem"][i] for c in curves if len(c["l_mem"]) > i]) for i in range(min(L, 30))]
        print(f"t{tid:02d} mean l_mem {np.concatenate([c['l_mem'] for c in curves]).mean():.4f}  by step: " + " ".join(f"{x:.2f}" for x in prof), flush=True)
        print(f"t{tid:02d} mean gain {allg.mean():.4f}  p90 {np.percentile(allg, 90):.4f}  max {allg.max():.4f}", flush=True)
    json.dump(out, open(a.out, "w"))
