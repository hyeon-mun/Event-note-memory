"""Before/after change-crop notes on the cue-removal event and the candidates-appearance event (t51, t52 demos).
usage: change_bench.py MODEL_ID"""
import sys, collections
import numpy as np
from note_bench import VLM, frames, first_color, SYN, Y0
from zoom_bench import zoom
P = ("List the objects in the image with precise color names, e.g. 'a cyan cube, a maroon cube'. "
     "The wooden surface is not an object. If there are no objects, answer 'nothing'.")
F = frames(51, 40) + frames(52, 40)
by = collections.defaultdict(dict)
for f in F:
    by[(f[0], f[1])][f[2]] = f
imgs, meta = [], []
for k, d in by.items():
    for a, b in (("cue", "empty"), ("empty", "cand")):
        m = np.abs(d[a][3].astype(int) - d[b][3].astype(int)).max(-1) > 40; m[:Y0] = False
        imgs += [zoom(d[a][3], m), zoom(d[b][3], m)]; meta += [(k, a, b, d["cue"][4])] * 2
n = VLM(sys.argv[1])(imgs, P)
rm = [(n[i], n[i + 1], meta[i][3]) for i in range(0, len(n), 2) if meta[i][1] == "cue"]
ap = [(n[i], n[i + 1]) for i in range(0, len(n), 2) if meta[i][1] == "empty"]
c = collections.defaultdict(collections.Counter)
for b, a, gt in rm:
    c[gt][first_color(b)] += 1
print(sys.argv[1].split("/")[-1])
print(" removal: before colour ok", np.mean([first_color(b) in SYN[g] for b, a, g in rm]), " after says nothing", np.mean(["nothing" in a.lower() for b, a, g in rm]), collections.Counter(a for b, a, g in rm).most_common(3))
print(" appear: before says nothing", np.mean(["nothing" in b.lower() for b, a in ap]), " after", collections.Counter(a for b, a in ap).most_common(6))
print(" ", {g: dict(c[g]) for g in sorted(c)})
print(" ex", rm[:3])
