"""Change-focused note test: crop the region that changed between the cue frame and the empty-table frame, zoom it,
and ask the VLM for the object's colour. usage: zoom_bench.py MODEL_ID PROMPT_ID"""
import sys, collections
import numpy as np
from PIL import Image
from note_bench import VLM, frames, first_color, SYN, Y0
PROMPTS = {"plain": "Describe the object in the image in a few words, including its color.",
           "precise": "What is the color of the object? Use a precise color name such as maroon, red, orange, yellow, "
                      "olive, lime, green, teal, cyan, blue, navy, purple, magenta, pink."}


def zoom(img, mask, m=6, s=24):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    h = max(s, ys.max() - ys.min() + 2 * m, xs.max() - xs.min() + 2 * m) // 2
    y0, x0 = np.clip(cy - h, 0, 128 - 2 * h), np.clip(cx - h, 0, 128 - 2 * h)
    return Image.fromarray(np.asarray(img)[y0:y0 + 2 * h, x0:x0 + 2 * h]).resize((384, 384), Image.NEAREST)


if __name__ == "__main__":
    F = frames(51, 40) + frames(52, 40)
    cue = [f for f in F if f[2] == "cue"]; emp = {(f[0], f[1]): f[3] for f in F if f[2] == "empty"}
    imgs = []
    for tid, e, _, img, gt in cue:
        d = np.abs(img.astype(int) - emp[(tid, e)].astype(int)).max(-1) > 40; d[:Y0] = False
        imgs.append(zoom(img, d))
    imgs[0].save("/tmp/claude-1003/-data-workspaces-mikasa/5f7ed5c4-6cff-4972-865e-1e30f92b1f6b/scratchpad/zoom0.png")
    notes = VLM(sys.argv[1])(imgs, PROMPTS[sys.argv[2]])
    c = collections.defaultdict(collections.Counter)
    for (tid, e, _, _, gt), s in zip(cue, notes):
        c[gt][first_color(s)] += 1
    print(sys.argv[1].split("/")[-1], sys.argv[2], "acc", np.mean([first_color(s) in SYN[f[4]] for f, s in zip(cue, notes)]))
    print({g: dict(c[g]) for g in sorted(c)}); print(notes[:4])
