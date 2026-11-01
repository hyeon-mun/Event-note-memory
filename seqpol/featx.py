"""Frozen DINOv2-S/14 features per frame: CLS + 4x4 pooled patch grid (17 tokens) per camera.

Writes <task>/feat.npy  (N, 2, 17, 384) float16.
"""
import argparse, glob, json, os, time
import numpy as np, torch, torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dino(device):
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    return m.eval().to(device).half()


@torch.no_grad()
def encode(model, x_uint8):
    """x_uint8: (B,128,128,3) uint8 tensor on device -> (B,17,384) fp16."""
    x = x_uint8.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
    x = ((x - MEAN.to(x.device)) / STD.to(x.device)).half()
    out = model.forward_features(x)
    cls = out["x_norm_clstoken"]                                   # (B,384)
    p = out["x_norm_patchtokens"].reshape(-1, 16, 16, 384).permute(0, 3, 1, 2)
    p = F.adaptive_avg_pool2d(p.float(), 4).flatten(2).transpose(1, 2).half()  # (B,16,384)
    return torch.cat([cls[:, None], p], 1)


def run_task(model, d, device, bs=1024):
    out = f"{d}/feat.npy"
    if os.path.exists(out + ".done"):
        return "skip"
    n = json.load(open(f"{d}/meta.json"))["n"]
    cams = [np.load(f"{d}/{c}.npy", mmap_mode="r") for c in ("top", "wrist")]
    feat = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n, 2, 17, 384))
    for s in range(0, n, bs):
        for ci, cam in enumerate(cams):
            x = torch.from_numpy(np.ascontiguousarray(cam[s:s + bs])).to(device, non_blocking=True)
            feat[s:s + bs, ci] = encode(model, x).cpu().numpy()
    feat.flush(); open(out + ".done", "w").close()
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/workspaces/mikasa/data_mikasa_robo/mm")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda"
    model = load_dino(dev)
    dirs = sorted(d for d in glob.glob(f"{a.root}/*") if os.path.exists(f"{d}/meta.json"))[a.shard::a.nshard]
    for d in dirs:
        t = time.time(); r = run_task(model, d, dev)
        print(os.path.basename(d), r, f"{time.time() - t:.0f}s", flush=True)
