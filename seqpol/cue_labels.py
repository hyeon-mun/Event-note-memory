"""Cue-color labels for RememberColor-type tasks: most saturated pixels near the table center at step 0."""
import numpy as np
from data import TaskStore


def cue_colors(tid, box=(64, 88, 52, 76)):
    s = TaskStore(tid); out = []
    for a in s.ep_start:
        img = np.asarray(s.top[a])[box[0]:box[1], box[2]:box[3]].reshape(-1, 3).astype(np.float32)
        sat = img.max(1) - img.min(1)
        out.append(img[np.argsort(-sat)[:4]].mean(0))
    return np.stack(out)


def cue_shape_crop(img, box=(60, 92, 48, 80)):
    """10x10 blue-minus-red crop centred on the cue (RememberShape tasks: all objects are blue)."""
    d = np.asarray(img)[box[0]:box[1], box[2]:box[3]].astype(np.float32)
    d = d[..., 2] - d[..., 0]
    ys, xs = np.nonzero(d > 40); cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    return np.pad(d, 5)[cy:cy + 10, cx:cx + 10].reshape(-1)


def cue_shapes(tid):
    s = TaskStore(tid)
    return np.stack([cue_shape_crop(s.top[a]) for a in s.ep_start])


def cluster(cols, k, iters=100, seed=0):
    best = None
    for r in range(20):
        rng = np.random.default_rng(seed + r); C = cols[rng.choice(len(cols), k, replace=False)]
        for _ in range(iters):
            lab = ((cols[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C = np.stack([cols[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        d = ((cols - C[lab]) ** 2).sum()
        if best is None or d < best[0]:
            best = (d, lab, C)
    return best[1], best[2]


if __name__ == "__main__":
    from collections import Counter
    cols = cue_colors(14); lab, C = cluster(cols, 5)
    print(C.round(0)); print(Counter(lab.tolist()))
    print("within", np.sqrt(((cols - C[lab]) ** 2).sum(-1)).mean().round(1))


def color_clusters(tid, k):
    """(labels, centres) of the cue colours. RememberColor9 short (t15): k-means merges yellow/orange there, so its
    episodes are assigned to the RememberColor9-Long (t52) centres (same 9 colours, clean clusters)."""
    c = cue_colors(tid)
    if tid == 15:
        C = cluster(cue_colors(52), 9)[1]
        return ((c[:, None] - C[None]) ** 2).sum(-1).argmin(1), C
    return cluster(c, k)
