"""Crop-prompt selection on t52 cue-removal crops (9 colours): discriminability = accuracy of the best map note-word -> colour."""
import collections, sys
import numpy as np
from note_bench import frames, first_color
from notes import ChangeNoteWriter, DPIX
LIST = "maroon, red, orange, yellow, olive, lime, green, teal, cyan, blue, navy, purple, magenta, pink"
PS = {"list": f"List the objects in the image with precise color names, e.g. 'a cyan cube, a maroon cube'. The wooden surface is not an object. If there are no objects, answer 'nothing'.",
      "vocab": f"List the objects in the image, naming each color precisely (for example {LIST}). The wooden surface is not an object. If there are no objects, answer 'nothing'.",
      "name": f"Name the object in the image with a precise color name (such as {LIST}), e.g. 'a cyan cube'. If there is no object, answer 'nothing'."}
F = frames(52, 250); by = collections.defaultdict(dict)
for f in F: by[f[1]][f[2]] = f
w = ChangeNoteWriter(); crops, gts, empt = [], [], []
for e, d in by.items():
    m = np.abs(d["cue"][3].astype(int) - d["empty"][3].astype(int)).max(-1) > DPIX
    crops.append(w.zoom(d["cue"][3], m)); empt.append(w.zoom(d["empty"][3], m)); gts.append(d["cue"][4])
for k, p in PS.items():
    n = w.ask(crops + empt[:40], p); a, b = n[:len(crops)], n[len(crops):]
    c = collections.defaultdict(collections.Counter)
    for s, g in zip(a, gts): c[first_color(s)][g] += 1
    print(k, "discriminability", sum(v.most_common(1)[0][1] for v in c.values()) / len(gts),
          "| empty crop ->", collections.Counter(b).most_common(3), flush=True)
    cc = collections.defaultdict(collections.Counter)
    for s, g in zip(a, gts): cc[g][first_color(s)] += 1
    print("  ", {g: dict(cc[g]) for g in sorted(cc)})
