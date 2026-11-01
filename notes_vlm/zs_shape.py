"""Zero-shot check: does a VLM's generic scene note identify the cue shape of t54 (RememberShape5-Long)?
Cue labels: blue-mask (area, height, width) at step 0. Score: accuracy of predicting the label from the
note's shape word with the majority mapping word -> label (an upper bound of what a policy can extract).
usage: zs_shape.py MODEL_ID [N]"""
import collections, re, sys
import numpy as np, torch
from PIL import Image

mid, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120
PROMPT = "Briefly describe the objects on the table."
d = "/data/workspaces/mikasa/data_mikasa_robo/mm/t54/"
T, st = np.load(d + "top.npy", mmap_mode="r"), np.load(d + "ep_start.npy")
frames, lab = [], []
for a in st[:N]:
    im = np.asarray(T[a]); m = im[60:92, 48:80, 2].astype(int) - im[60:92, 48:80, 0] > 40
    ys, xs = np.nonzero(m); lab.append((int(m.sum()), int(np.ptp(ys) + 1), int(np.ptp(xs) + 1))); frames.append(im)

from transformers import AutoProcessor, AutoModelForImageTextToText
proc = AutoProcessor.from_pretrained(mid)
model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.bfloat16).cuda().eval()
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
text = proc.apply_chat_template(msgs, add_generation_prompt=True)
caps = []
for i in range(0, len(frames), 8):
    imgs = [[Image.fromarray(f).resize((384, 384))] for f in frames[i:i + 8]]
    inp = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True).to("cuda", torch.bfloat16)
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=40, do_sample=False)
    caps += [c.strip() for c in proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
words = lambda c: tuple(sorted(set(re.findall(r"[a-z]+", c.lower())) - {"a", "an", "the", "and", "on", "table", "is", "of", "there", "white", "robot", "object", "gray", "black"}))
key = [re.search(r"blue (\w+)", c.lower()).group(1) if re.search(r"blue (\w+)", c.lower()) else words(c) for c in caps]
m = collections.defaultdict(collections.Counter)
for k, l in zip(key, lab): m[k][l] += 1
acc = sum(v.most_common(1)[0][1] for v in m.values()) / len(lab)
base = collections.Counter(lab).most_common(1)[0][1] / len(lab)
print(f"{mid}: word->label acc {acc:.2f} (majority {base:.2f}), distinct keys {len(m)}")
for l in sorted(set(lab)): print(l, collections.Counter(k for k, ll in zip(key, lab) if ll == l).most_common(5))
for c in caps[:5]: print("  ", c)
