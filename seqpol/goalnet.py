"""Goal localisation diagnostic: given the cue (oracle colour label, or the step-0 frame) and the frame at motion
onset, predict the demo's final EE xy (the touched cube). Tests whether cue->target grounding is learnable from
the 250 demos at full image resolution, independent of the action policy.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cue_labels import cluster, cue_colors

ROOT = "/data/workspaces/mikasa/data_mikasa_robo/mm"
K = {13: 3, 14: 5, 15: 9, 50: 3, 51: 5, 52: 9}


def load(tid):
    d = f"{ROOT}/t{tid:02d}"
    top = np.load(f"{d}/top.npy", mmap_mode="r"); wrist = np.load(f"{d}/wrist.npy", mmap_mode="r")
    s = np.load(f"{d}/state.npy"); st = np.load(f"{d}/ep_start.npy"); L = np.load(f"{d}/ep_len.npy")
    X0, Xt, Wt, Y, on = [], [], [], [], []
    for e in range(len(L)):
        x = s[st[e]:st[e] + L[e]]
        mv = np.nonzero(np.linalg.norm(np.diff(x[:, :3], axis=0), axis=1) > 1e-3)[0]
        t = int(mv[0]) if len(mv) else 0
        X0.append(top[st[e]]); Xt.append(top[st[e] + t]); Wt.append(wrist[st[e] + t])
        Y.append(x[-1, :2]); on.append(t)
    lab = cluster(cue_colors(tid), K[tid])[0]
    f = lambda a: torch.from_numpy(np.stack(a))
    return f(X0), f(Xt), f(Wt), torch.from_numpy(np.stack(Y)).float(), torch.from_numpy(lab).long(), np.array(on)


def calib(Xt, Y, lab, C, tr):
    """Pixel<->table-plane affine maps from colour-matched cube centroids on training episodes (stands in for the
    known camera calibration)."""
    P = []
    for i in range(len(Y)):
        im = Xt[i].numpy().astype(float); m = ((im - C[lab[i]]) ** 2).sum(-1) < 60 ** 2; m[:60] = False
        ys, xs = np.nonzero(m); P.append([xs.mean(), ys.mean(), 1] if len(xs) else [np.nan] * 3)
    P = np.array(P); Yn = Y.numpy(); tr = [i for i in tr if not np.isnan(P[i, 0])]
    p2w = np.linalg.lstsq(P[tr], Yn[tr], rcond=None)[0]
    w2p = np.linalg.lstsq(np.c_[Yn[tr], np.ones(len(tr))], P[tr, :2], rcond=None)[0]
    return p2w, w2p


class Net(nn.Module):
    def __init__(self, k, cue="label", d=64):
        super().__init__()
        conv = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(8, o), nn.GELU())
        self.enc = nn.Sequential(conv(6, d, 1), conv(d, d, 2), conv(d, d, 1), conv(d, d, 2), conv(d, d, 1))  # 32x32
        self.cue = cue
        self.q = nn.Embedding(k, d) if cue == "label" else nn.Sequential(conv(3, d, 2), conv(d, d, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(d, d))
        self.film = nn.Linear(d, 2 * d)
        self.head = nn.Sequential(conv(d + 2, d, 1), nn.Conv2d(d, 1, 1))
        self.out = nn.Linear(2, 2)                                                   # pixel -> world xy (affine)
        g = torch.linspace(-1, 1, 32)
        self.register_buffer("grid", torch.stack(torch.meshgrid(g, g, indexing="ij"), 0)[None])  # (1,2,32,32)

    def forward(self, cue, top, wrist):
        img = torch.cat([top, wrist], -1).permute(0, 3, 1, 2).float() / 255.0
        h = self.enc(img)
        q = self.q(cue) if self.cue == "label" else self.q(cue.permute(0, 3, 1, 2).float() / 255.0)
        g, b = self.film(q).chunk(2, -1)
        h = h * (1 + g[..., None, None]) + b[..., None, None]
        return self.head(torch.cat([h, self.grid.expand(len(h), -1, -1, -1)], 1)).flatten(1)  # (B, 32*32) logits


class Siamese(nn.Module):
    """Matching readout: a shared per-pixel encoder embeds the cue frame and the current frame; the memory is a
    saliency-pooled cue embedding, and the goal is the soft-argmax of its correlation with the current map."""

    def __init__(self, d=32, stride=2):
        super().__init__()
        conv = lambda i, o, k, s: nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.GELU())
        self.enc = nn.Sequential(conv(3, d, 3, stride), conv(d, d, 1, 1), nn.Conv2d(d, d, 1))
        self.sal = nn.Conv2d(d, 1, 1)                                                # what to remember from the cue frame
        self.tau = nn.Parameter(torch.tensor(2.0))
        self.out = nn.Linear(2, 2)
        n = 128 // stride; g = torch.linspace(-1, 1, n)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("grid", torch.stack([xx, yy], 0).flatten(1))           # (2, n*n)

    def embed(self, im):
        return F.normalize(self.enc(im.permute(0, 3, 1, 2).float() / 255.0), dim=1)

    def forward(self, cue, top, wrist=None):
        h0 = self.embed(cue); a = self.sal(h0).flatten(1).softmax(-1)               # (B, HW)
        m = F.normalize((h0.flatten(2) * a[:, None]).sum(-1), dim=-1)                # memory vector (B, d)
        ht = self.embed(top).flatten(2)                                              # (B, d, HW)
        return (m[:, :, None] * ht).sum(1) * self.tau.exp() * 5                    # (B, n*n) logits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=51); ap.add_argument("--cue", default="label", choices=["label", "image", "none", "siamese"])
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--no_wrist", action="store_true"); ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    torch.manual_seed(0)
    X0, Xt, Wt, Y, lab, on = load(a.task)
    if a.no_wrist:
        Wt = torch.zeros_like(Wt)
    dev = "cuda"; tr, va = torch.arange(225), torch.arange(225, 250)
    net = (Siamese() if a.cue == "siamese" else Net(K[a.task], "label" if a.cue in ("label", "none") else "image")).to(dev)
    n = 64 if a.cue == "siamese" else 32
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    cue = lambda i: (lab[i] * (a.cue != "none")).to(dev) if a.cue not in ("image", "siamese") else X0[i].to(dev)
    p2w, w2p = calib(Xt, Y, lab, cluster(cue_colors(a.task), K[a.task])[1], tr.tolist())
    pix = torch.from_numpy(np.c_[Y.numpy(), np.ones(len(Y))] @ w2p).float()            # target pixel (u, v)
    cell = lambda uv: (uv[:, 1] * n / 128).long().clamp(0, n - 1) * n + (uv[:, 0] * n / 128).long().clamp(0, n - 1)
    def predict(i):
        k = net(cue(i), Xt[i].to(dev), Wt[i].to(dev)).argmax(-1).cpu()
        uv = torch.stack([(k % n).float() + 0.5, (k // n).float() + 0.5], -1) * 128 / n
        return torch.from_numpy(np.c_[uv.numpy(), np.ones(len(uv))] @ p2w).float()
    for it in range(a.steps + 1):
        i = tr[torch.randint(0, 225, (32,))]
        loss = F.cross_entropy(net(cue(i), Xt[i].to(dev), Wt[i].to(dev)), cell(pix[i]).to(dev), label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0:
            with torch.no_grad():
                err = (predict(va) - Y[va]).norm(dim=-1); errt = (predict(tr[:100]) - Y[tr[:100]]).norm(dim=-1)
            print(f"it {it} loss {loss.item():.3f}  train err {errt.mean()*100:.1f}cm  val err {err.mean()*100:.1f}cm  "
                  f"val <3cm {(err < 0.03).float().mean():.2f}", flush=True)
