"""Held-out goal grounding accuracy of a --goal policy: pixel error of the goal head after motion onset
(when the goal is visible) on episodes 225..249, plus the fraction within 4 px (~ one cube)."""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from analyze import load, oracle_label
from data import EpisodeDataset, collate


LAG = None


@torch.no_grad()
def run(m, ds, dev="cuda"):
    errs, early = [], []
    for i in range(len(ds)):
        b = {k: v.to(dev) for k, v in collate([ds[i]]).items()}
        prop = (b["proprio"] - m.p_mean) / m.p_std
        T = int(b["T"][0]); sel = b["sel"][0][:T]
        if m.mem_src == "oracle":
            lab = oracle_label(int(b["tid"][0]), int(b["ep"][0]))
            mem, _ = m.memory(F.one_hot(torch.tensor([lab], device=dev), 16).float()[:, None].expand(1, T, 16))
        else:
            mem, _ = m.memory(m.mem_inputs(b["feat"], prop, b["text"]))
        x = b["proprio"][0, :, :3]
        mv = ((x[1:] - x[:-1]).norm(dim=-1) > 1e-3).nonzero()
        t0 = int(mv[0]) if len(mv) else 0
        ts = torch.arange(t0, T, device=dev)
        mt = mem[0, ts] if LAG is None else mem[0, max(0, t0 - LAG)][None].expand(len(ts), -1)   # memory frozen before the scene
        lg = m.goal(b["top"][0][ts], b["wrist"][0][ts], mt.float())
        e = (m.goal.uv(lg) - b["goal"][0][ts]).norm(dim=-1)
        errs.append(e[0].item()); early.append(e[: max(1, len(ts) // 3)].mean().item())
    errs, early = np.array(errs), np.array(early)
    return dict(onset_err_px=errs.mean(), onset_acc4=(errs < 4).mean(), first_third_err=early.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpts", nargs="+"); ap.add_argument("--task", type=int, default=51); ap.add_argument("--ep_lo", type=int, default=225)
    ap.add_argument("--lag", type=int, default=None)
    a = ap.parse_args(); LAG = a.lag
    for c in a.ckpts:
        m, targs = load(c, "cuda")
        ds = EpisodeDataset([a.task], episodes=range(a.ep_lo, a.ep_lo + 25), horizon=targs["horizon"], n_sel=10 ** 6, tokens=targs["tokens"])
        print(c.split("runs/")[-1], {k: round(float(v), 3) for k, v in run(m, ds).items()}, flush=True)
