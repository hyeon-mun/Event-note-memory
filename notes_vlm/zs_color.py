"""Zero-shot check: can an off-the-shelf VLM name the cue cube colour from the 128px step-0 top frame?"""
import sys, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
mid = sys.argv[1]
proc = AutoProcessor.from_pretrained(mid)
model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()
d = "/data/workspaces/mikasa/data_mikasa_robo/mm/t51/"
top = np.load(d + "top.npy", mmap_mode="r"); st = np.load(d + "ep_start.npy")
sys.path.insert(0, "/data/workspaces/mikasa/research/seqpol")
from cue_labels import cluster, cue_colors
from make_note_names import color_names
lab, C = cluster(cue_colors(51), 5); names = color_names(C)
q = "What color is the small cube on the table? Answer with one word."
hits = []
for e in range(0, 250, 5):
    img = Image.fromarray(np.asarray(top[st[e]])).resize((384, 384))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=text, images=[img], return_tensors="pt").to("cuda", torch.bfloat16)
    out = model.generate(**inp, max_new_tokens=5, do_sample=False)
    ans = proc.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    hits.append(names[lab[e]].lower() in ans or ans.rstrip(".") in names[lab[e]].lower())
    if e < 30: print(e, names[lab[e]], "->", ans, flush=True)
print("acc", np.mean(hits), "n", len(hits))
