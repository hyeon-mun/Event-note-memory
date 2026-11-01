"""Note-quality benchmark on demo event frames of RememberColor-Long (t51, t52).
Frame types from the v2 events: 0 = cue (one cube), 1 = empty table, last = candidates.
Ground truth of the cue colour: pixels that differ from the empty-table frame, nearest env colour.
Scores: cue colour named first (lenient synonyms), empty frame says it is empty, robot/hallucinated-object mentions.
usage: note_bench.py MODEL_ID VARIANT [N_EPS]   VARIANT in full|crop|crop_prompt|full_prompt"""
import json, re, sys
import numpy as np, torch
from PIL import Image

PAL = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 0, 0],
                [255, 128, 0], [0, 128, 128]], float)
SYN = [["red"], ["green", "lime"], ["blue"], ["yellow"], ["magenta", "pink", "purple", "violet"], ["cyan", "turquoise", "light blue", "aqua", "teal"],
       ["maroon", "dark red", "red", "brown", "burgundy"], ["orange", "olive", "amber"], ["teal", "dark green", "green", "turquoise", "cyan", "dark cyan"]]
WORDS = sorted({w for s in SYN for w in s} | {"white", "black", "gray", "grey", "silver"}, key=len, reverse=True)
EXTRA = re.compile(r"robot|arm|gripper|monitor|keyboard|mouse|computer|laptop|remote|cup|book|phone|bird|printer|toy|furniture", re.I)
EMPTY = re.compile(r"empty|nothing|no objects|no object|bare", re.I)
PROMPTS = {"generic": "Briefly describe the objects on the table.",
           "table": "What objects are on the wooden table? Name each object with its color, e.g. 'a red cube'. "
                    "Ignore the robot arm. If there is nothing on the table, answer 'The table is empty.'"}
Y0, Y1, X0, X1 = 44, 112, 16, 112                     # workspace (table) region of the fixed 128px top camera


def view(f, crop):
    f = np.asarray(f)
    if crop:
        f = f[Y0:Y1, X0:X1]
    return Image.fromarray(f).resize((384 * f.shape[1] // max(f.shape[:2]), 384 * f.shape[0] // max(f.shape[:2])), Image.BICUBIC)


def first_color(s):
    s = s.lower(); best = None
    for w in WORDS:
        i = s.find(w)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, w)
    return best[1] if best else None


class VLM:
    def __init__(self, mid):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()

    @torch.no_grad()
    def __call__(self, imgs, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        out = []
        for i in range(0, len(imgs), 8):
            b = imgs[i:i + 8]
            inp = self.proc(text=[text] * len(b), images=[[x] for x in b], return_tensors="pt", padding=True).to("cuda", torch.bfloat16)
            g = self.model.generate(**inp, max_new_tokens=40, do_sample=False)
            out += [s.strip() for s in self.proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out


def frames(tid, n):
    mm = f"/data/workspaces/mikasa/data_mikasa_robo/mm/t{tid:02d}"
    top = np.load(f"{mm}/top.npy", mmap_mode="r"); st = np.load(f"{mm}/ep_start.npy")
    ev = json.load(open(f"/data/workspaces/mikasa/data_mikasa_robo/notes/t{tid:02d}_events_v2.json"))
    out = []
    for e in range(n):
        ts = [t for t, _ in ev[str(e)]]
        if len(ts) < 3:
            continue
        cue, emp, cand = (np.asarray(top[st[e] + t]) for t in (ts[0], ts[1], ts[-1]))
        d = np.abs(cue.astype(int) - emp.astype(int)).max(-1) > 40
        d[:Y0] = False                                   # robot region
        if d.sum() < 3:
            continue
        px = cue[d].astype(float)
        px = px[np.argsort(-px.max(1))[: max(3, len(px) // 2)]]   # lit faces
        gt = int(np.argmin(((PAL - px.mean(0)) ** 2).sum(1)))
        out += [(tid, e, "cue", cue, gt), (tid, e, "empty", emp, None), (tid, e, "cand", cand, None)]
    return out


if __name__ == "__main__":
    mid, var = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    F = frames(51, n) + frames(52, n)
    crop = var.startswith("crop"); prompt = PROMPTS["table" if var.endswith("prompt") else "generic"]
    notes = VLM(mid)([view(f[3], crop) for f in F], prompt)
    r = {"cue": [], "empty": [], "extra": [], "cand_extra": []}
    for (tid, e, kind, _, gt), s in zip(F, notes):
        if kind == "cue":
            r["cue"].append(first_color(s) in SYN[gt]); r.setdefault("cue_empty", []).append(bool(EMPTY.search(s)))
        if kind == "empty":
            r["empty"].append(bool(EMPTY.search(s)))
        r["extra"].append(bool(EXTRA.search(s)))
    print(f"{mid.split('/')[-1]} {var}: cue-colour {np.mean(r['cue']):.2f} (n={len(r['cue'])})  empty-said-empty {np.mean(r['empty']):.2f}"
          f"  cue-said-empty {np.mean(r['cue_empty']):.2f}  robot/hallucinated-object mention {np.mean(r['extra']):.2f}", flush=True)
    gts = [f[4] for f in F if f[2] == "cue"]
    print("gt colour hist", np.bincount(gts, minlength=9).tolist())
    json.dump([dict(tid=f[0], ep=f[1], kind=f[2], gt=f[4], note=s) for f, s in zip(F, notes)],
              open(f"/data/workspaces/mikasa/research/logs/note_bench_{mid.split('/')[-1]}_{var}.json", "w"), indent=0)
