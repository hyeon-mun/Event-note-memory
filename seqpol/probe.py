"""Linear probe: is the cue (label) decodable from the memory state at later steps, on held-out episodes?

Labels come from cue_labels.py (RememberColor-type tasks). Memory states m_t are taken at steps t >= t_min
(after the cue has disappeared); a softmax-regression probe is trained on train episodes and tested on val ones.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from analyze import load
from cue_labels import cluster, cue_colors
from data import EpisodeDataset, collate


@torch.no_grad()
def mem_states(m, ds, t_min, t_max, dev="cuda"):
    X, E = [], []
    for i in range(len(ds)):
        b = {k: v.to(dev) for k, v in collate([ds[i]]).items()}
        prop = (b["proprio"] - m.p_mean) / m.p_std
        T = int(b["T"][0]); sel = b["sel"][0][:T]
        react = m.reactive(b["top"][0], b["wrist"][0], prop[0, sel], b["text"].expand(len(sel), -1))
        mem, _ = m.memory(react[None] if m.mem_src == "cnn" else m.mem_inputs(b["feat"], prop, b["text"]))
        for t in range(t_min, min(T, t_max)):
            X.append(torch.cat([mem[0, t], react[t]]).float().cpu()); E.append(i)
    return torch.stack(X), np.array(E)


def fit_probe(X, y, Xv, yv, epochs=300):
    mu, sd = X.mean(0), X.std(0) + 1e-5
    X, Xv = (X - mu) / sd, (Xv - mu) / sd
    W = torch.zeros(X.shape[1], int(y.max()) + 1, requires_grad=True); b = torch.zeros(int(y.max()) + 1, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=1e-3)
    yt = torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(X @ W + b, yt).backward(); opt.step()
    return float(((Xv @ W + b).argmax(1).numpy() == yv).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--task", type=int, default=14)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--t_min", type=int, default=8); ap.add_argument("--t_max", type=int, default=14)
    a = ap.parse_args()
    m, targs = load(a.ckpt, "cuda")
    lab, _ = cluster(cue_colors(a.task), a.k)
    mk = lambda eps: EpisodeDataset([a.task], episodes=eps, horizon=targs["horizon"], n_sel=10 ** 6, tokens=targs["tokens"])
    Xtr, Etr = mem_states(m, mk(range(0, 225)), a.t_min, a.t_max)
    Xva, Eva = mem_states(m, mk(range(225, 250)), a.t_min, a.t_max)
    ytr, yva = lab[Etr], lab[225 + Eva]
    d = m.memory.norm.normalized_shape[0]
    acc_mem = fit_probe(Xtr[:, :d], ytr, Xva[:, :d], yva)
    acc_react = fit_probe(Xtr[:, d:], ytr, Xva[:, d:], yva)
    print(f"probe val acc  memory {acc_mem:.3f}  reactive {acc_react:.3f}  (chance {1 / a.k:.2f}, n_val {len(yva)})")
