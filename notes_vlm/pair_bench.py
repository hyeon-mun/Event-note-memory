"""Two-image change note: zoomed before/after crops of the changed region, one sentence about the change.
usage: pair_bench.py MODEL_ID"""
import sys, collections, re
import numpy as np, torch
from note_bench import VLM, frames, first_color, SYN, Y0
from zoom_bench import zoom
P = ("The first image shows a small region of a table before a change, the second image the same region after it. "
     "In one short sentence, say what appeared or disappeared, with a precise color name (such as maroon, red, orange, "
     "yellow, olive, lime, green, teal, cyan, blue, navy, purple, magenta, pink), e.g. 'A cyan cube disappeared.' "
     "The wooden table itself is not an object.")


class Pair(VLM):
    @torch.no_grad()
    def __call__(self, pairs, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        out = []
        for i in range(0, len(pairs), 8):
            b = pairs[i:i + 8]
            inp = self.proc(text=[text] * len(b), images=[list(x) for x in b], return_tensors="pt", padding=True).to("cuda", torch.bfloat16)
            g = self.model.generate(**inp, max_new_tokens=30, do_sample=False)
            out += [s.strip() for s in self.proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out


if __name__ == "__main__":
    F = frames(51, 40) + frames(52, 40)
    by = collections.defaultdict(dict)
    for f in F:
        by[(f[0], f[1])][f[2]] = f
    pairs, meta = [], []
    for k, d in by.items():
        for a, b in (("cue", "empty"), ("empty", "cand")):
            m = np.abs(d[a][3].astype(int) - d[b][3].astype(int)).max(-1) > 40; m[:Y0] = False
            pairs.append((zoom(d[a][3], m), zoom(d[b][3], m))); meta.append((a, d["cue"][4]))
    n = Pair(sys.argv[1])(pairs, P)
    rm = [(s, g) for s, (a, g) in zip(n, meta) if a == "cue"]; ap = [s for s, (a, g) in zip(n, meta) if a == "empty"]
    c = collections.defaultdict(collections.Counter)
    for s, g in rm:
        c[g][first_color(s)] += 1
    print(sys.argv[1].split("/")[-1], "removal colour ok", np.mean([first_color(s) in SYN[g] for s, g in rm]),
          "says disappeared", np.mean([bool(re.search("disappear|remov|gone|missing", s, re.I)) for s, g in rm]),
          "| appear says appeared", np.mean([bool(re.search("appear", s, re.I)) and not re.search("disappear", s, re.I) for s in ap]))
    print(" ", {g: dict(c[g]) for g in sorted(c)}); print(" ex", rm[:3], ap[:3])
