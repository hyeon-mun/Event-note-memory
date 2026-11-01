"""Offline direction check on held-out episodes: does the sampled action chunk move the EE the same way (xy)
as the demonstrator's chunk at each step? Reports mean cosine of summed xy deltas per timestep bucket."""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
ONSET = False

from analyze import load, oracle_label
from data import EpisodeDataset, collate


@torch.no_grad()
def run(m, ds, dev="cuda", n_samples=4):
    out = {}
    for i in range(len(ds)):
        b = {k: v.to(dev) for k, v in collate([ds[i]]).items()}
        prop = (b["proprio"] - m.p_mean) / m.p_std
        T = int(b["T"][0]); sel = b["sel"][0][:T]
        react = m.reactive(b["top"][0], b["wrist"][0], prop[0, sel], b["text"].expand(T, -1))
        if m.mem_src == "oracle":
            lab = oracle_label(int(b["tid"][0]), int(b["ep"][0]))
            mem, _ = m.memory(F.one_hot(torch.tensor([lab], device=dev), 16).float()[:, None].expand(1, T, 16))
        else:
            mem, _ = m.memory(react[None] if m.mem_src == "cnn" else m.mem_inputs(b["feat"], prop, b["text"]))
        gt = b["act"][0][:, :4, :2].sum(1)                                   # demo xy motion over 4 steps
        moving = (gt.norm(dim=-1) > 0.05).nonzero()
        t0 = int(moving[0]) if len(moving) else 0                            # demo motion onset
        for _ in range(n_samples):
            m.reactive(b["top"][0], b["wrist"][0], prop[0, sel], b["text"].expand(T, -1))   # refresh maps
            pr = m.sample(react.float(), mem[0].float(), tid=int(b["tid"][0]))[:, :4, :2].sum(1)
            cos = F.cosine_similarity(pr, gt, dim=-1).cpu().numpy()
            mag = gt.norm(dim=-1).cpu().numpy()
            for t in range(T):
                if mag[t] > 0.05:
                    k = min(t // 3, 6) if not ONSET else min(max((t - t0) // 5, -1), 6)
                    out.setdefault(k, []).append(cos[t])
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--task", type=int, default=14); ap.add_argument("--onset", action="store_true")
    a = ap.parse_args()
    ONSET = a.onset
    m, targs = load(a.ckpt, "cuda")
    ds = EpisodeDataset([a.task], episodes=range(225, 250), horizon=targs["horizon"], n_sel=10 ** 6, tokens=targs["tokens"])
    r = run(m, ds)
    print("cos by bucket" + (" (5-step bins from motion onset)" if ONSET else " (3-step bins)") + ":", " ".join(f"{k}:{v:.2f}" for k, v in r.items()))
