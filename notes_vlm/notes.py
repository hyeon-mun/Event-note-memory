"""Language event notes: a task-agnostic event detector + an off-the-shelf VLM note writer.

Event: a sudden visual change of the top camera that is not caused by the robot's own motion: at least NPIX
pixels change by more than DPIX in some channel, at least RATIO x (1 + the median count of the previous IDLE steps),
while the end-effector has moved less than 1 mm per step during those IDLE steps. A pixel count (not the mean
difference) is invariant to the global re-shading present in the recorded demos; the ratio test rejects the slow
creep of a policy that drifts while waiting (each 0.01 mm of drift flips ~1 pixel at the gripper's edge).
At each event the VLM writes one short note about the scene; the policy is conditioned on the instruction plus the
last K notes. No task labels are used anywhere.
"""
import re
import numpy as np
import torch
from PIL import Image

DPIX, NPIX, RATIO, IDLE, K, BURST = 40, 8, 4, 5, 8, 10
PROMPT = "Briefly describe the objects on the table."
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


class EventDetector:
    def __init__(self):
        self.prev, self.counts, self.moves = None, [], []

    def __call__(self, top, xyz):
        """top uint8 (H,W,3), xyz (3,). Returns True if this frame is a scene event."""
        if self.prev is None:
            self.prev = (top, xyz)
            return True
        c = int((np.abs(top.astype(np.int16) - self.prev[0].astype(np.int16)).max(-1) > DPIX).sum())
        mv = float(np.linalg.norm(xyz - self.prev[1]))
        ev = (len(self.counts) >= IDLE and max(self.moves[-IDLE:]) < 1e-3 and c >= NPIX
              and c >= RATIO * (1 + np.median(self.counts[-IDLE:])))
        self.prev = (top, xyz); self.counts.append(c); self.moves.append(mv)
        return ev


class Scribe:
    """v2 write rule. Any history length counts (cue changes can come every 2-3 steps from t=0), and the note is
    written once the scene has settled: at the first frame after an event whose change is below NPIX (or SETTLE
    frames later). A change spread over several frames, or several changes in a row, gives one note of the end state."""
    SETTLE = 5

    def __init__(self, full=False):
        """full=True (v3): xyz is the whole proprio vector (EE position, orientation, gripper) and the robot counts as
        still only if every component changed by < 1e-3; otherwise the EE position moved < 1 mm."""
        self.full = full
        self.prev, self.counts, self.moves, self.wait, self.t = None, [], [], -1, 0
        self.before, self.before_t, self.mask = None, -1, None   # v3 notes: frame before the change, changed pixels

    def __call__(self, top, xyz):
        """top uint8 (H,W,3), xyz (3,). Returns True if a note should be written from this frame."""
        t, self.t = self.t, self.t + 1
        if self.prev is None:
            self.prev = (top, xyz)
            return True
        d = np.abs(top.astype(np.int16) - self.prev[0].astype(np.int16)).max(-1) > DPIX
        c = int(d.sum())
        mv = float(np.abs(xyz - self.prev[1]).max() if self.full else np.linalg.norm(xyz - self.prev[1]))
        prev_top = self.prev[0]
        h = self.counts[-IDLE:]
        ev = (h and max(self.moves[-IDLE:] + [mv]) < 1e-3 and c >= NPIX and c >= RATIO * (1 + np.median(h)))
        self.prev = (top, xyz); self.counts.append(c); self.moves.append(mv)
        if ev:
            if self.wait < 0:                            # first change of a burst: keep the scene before it
                self.before, self.before_t, self.mask = prev_top, t - 1, d
            else:                                        # only pixels that changed at event frames (EE still):
                self.mask = self.mask | d                # settling frames may already contain robot motion
            self.wait = 0
            return False
        if self.wait >= 0:
            self.wait += 1
            if c < NPIX or self.wait >= self.SETTLE:
                self.wait = -1
                return True
        return False


def episode_events(top, xyz, v=1):
    if v == 3:                                           # (write frame, frame before the change, changed pixels)
        sc, out = Scribe(full=True), []
        for t in range(len(top)):
            if sc(top[t], xyz[t]):
                out.append((t, sc.before_t, sc.mask) if t > 0 else (0, -1, None))
        return out
    if v == 2:
        sc = Scribe()
        return [t for t in range(len(top)) if sc(top[t], xyz[t])]
    det = EventDetector()
    ev = [t for t in range(len(top)) if det(top[t], xyz[t])]
    # an event at t=0 sees the initial scene; a change usually completes over 1-2 frames -> caption t+1 when possible
    return [min(t + 1, len(top) - 1) if t > 0 else 0 for t in ev]


class NoteWriter:
    def __init__(self, dev="cuda"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(VLM_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(VLM_ID, torch_dtype=torch.bfloat16).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, frames):
        """frames: list of uint8 (H,W,3) -> list of short notes."""
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        out = []
        for i in range(0, len(frames), 16):
            imgs = [[Image.fromarray(np.asarray(f)).resize((384, 384))] for f in frames[i:i + 16]]
            inp = self.proc(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(self.dev, torch.bfloat16)
            g = self.model.generate(**inp, max_new_tokens=30, do_sample=False)
            out += [s.strip() for s in self.proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out


class ChangeNoteWriter(NoteWriter):
    """v3 notes. The first frame gets a scene note (objects on the table, the robot ignored, 'empty' allowed).
    Every later event gets a change note: the region whose pixels changed between the frame before the event and
    the settled frame is cropped, zoomed and described before and after, so small objects are seen at a usable
    size and unchanged parts of the scene (the robot, the table) cannot be hallucinated into the note."""
    VLM_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    SCENE = ("What objects are on the wooden table? Name each object with its color, e.g. 'a red cube'. "
             "Ignore the robot arm. If there is nothing on the table, answer 'The table is empty.'")
    CROP = ("List the objects in the image with precise color names, e.g. 'a cyan cube, a maroon cube'. "
            "The wooden surface is not an object. If there are no objects, answer 'nothing'.")

    def __init__(self, dev="cuda"):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(self.VLM_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(self.VLM_ID, torch_dtype=torch.bfloat16).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def ask(self, imgs, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        out = []
        for i in range(0, len(imgs), 16):
            b = imgs[i:i + 16]
            inp = self.proc(text=[text] * len(b), images=[[x] for x in b], return_tensors="pt", padding=True).to(self.dev, torch.bfloat16)
            g = self.model.generate(**inp, max_new_tokens=40, do_sample=False)
            out += [s.strip().rstrip(".") for s in self.proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out

    @staticmethod
    def main_region(mask):
        """largest connected group of changed pixels (8 px dilation joins the faces of one object)"""
        from scipy import ndimage
        lab, n = ndimage.label(ndimage.binary_dilation(mask, iterations=4))
        if n <= 1:
            return mask
        k = np.argmax(ndimage.sum(mask, lab, range(1, n + 1))) + 1
        return mask & (lab == k)

    @staticmethod
    def zoom(img, mask, m=6, s=24):
        H, W = mask.shape
        ys, xs = np.nonzero(mask)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        h = min(max(s, ys.max() - ys.min() + 2 * m, xs.max() - xs.min() + 2 * m) // 2, H // 2, W // 2)
        y0, x0 = int(np.clip(cy - h, 0, H - 2 * h)), int(np.clip(cx - h, 0, W - 2 * h))
        return Image.fromarray(np.asarray(img)[y0:y0 + 2 * h, x0:x0 + 2 * h]).resize((384, 384), Image.NEAREST)

    def __call__(self, items):
        """items: list of (after, before, mask); before=None -> scene note of the first frame.
        Returns notes (None: the change did not persist)."""
        scene = [i for i, (a, b, m) in enumerate(items) if b is None]
        crops, idx = [], []
        for i, (a, b, m) in enumerate(items):
            if b is None:
                continue
            m = self.main_region(m & (np.abs(np.asarray(a).astype(np.int16) - np.asarray(b).astype(np.int16)).max(-1) > DPIX))
            if m.sum() >= NPIX // 2:
                crops += [self.zoom(b, m), self.zoom(a, m)]; idx.append(i)
        out = [None] * len(items)
        for i, s in zip(scene, self.ask([Image.fromarray(np.asarray(items[i][0])).resize((384, 384)) for i in scene], self.SCENE)):
            out[i] = s + "."
        c = self.ask(crops, self.CROP)
        for k, i in enumerate(idx):
            out[i] = f"Change: {c[2 * k]} -> {c[2 * k + 1]}."
        return out


class NoteMemory:
    """Notes written so far. An event within BURST steps of the previous one is the same change still settling,
    so its note replaces the previous note instead of adding a new one. A note identical to the previous one carries
    no new information and is dropped."""

    def __init__(self, burst=BURST):
        self.notes, self.last, self.burst = [], -10 ** 9, burst      # v2 (Scribe) settles changes itself: burst=0

    def add(self, t, note):
        if self.notes and note == self.notes[-1]:
            pass
        elif t - self.last <= self.burst and self.notes:
            self.notes[-1] = note
        else:
            self.notes.append(note)
        self.last = t


NO_OBJECT = re.compile(r"^(nothing|brown|(a |the )?wooden surface|(a |the )?table)$", re.I)


def gone_note(note):
    """v4: keep only what a change removed from view (the 'before' crop); what appeared stays visible anyway."""
    if note is None or not note.startswith("Change: "):
        return note
    b = note[len("Change: "):].rsplit(" -> ", 1)[0].strip().rstrip(".")
    return None if NO_OBJECT.match(b) else f"Gone: {b}."


COLOR_WORDS = None


HTML16 = {"black": "#000000", "silver": "#c0c0c0", "gray": "#808080", "white": "#ffffff", "maroon": "#800000",
          "red": "#ff0000", "purple": "#800080", "fuchsia": "#ff00ff", "green": "#008000", "lime": "#00ff00",
          "olive": "#808000", "yellow": "#ffff00", "navy": "#000080", "blue": "#0000ff", "teal": "#008080",
          "aqua": "#00ffff", "orange": "#ffa500"}            # HTML 4 basic colours + CSS2.1 orange


def measured_color(after, before, mask, vocab="css4"):
    """v5: CSS4 name (148 generic web colour names, not tuned to any task) of the mean of the most saturated quarter
    of the pixels the change removed. The VLM names hue families well but merges close shades (maroon/red,
    orange/yellow, teal/cyan); the pixel measurement does not."""
    m = ChangeNoteWriter.main_region(mask & (np.abs(np.asarray(after).astype(np.int16) - np.asarray(before).astype(np.int16)).max(-1) > DPIX))
    if m.sum() < NPIX // 2:
        return None
    px = np.asarray(before)[m][:, :3].astype(np.float32)
    s = px.max(1) - px.min(1)
    c = px[s >= np.quantile(s, 0.75)].mean(0)
    from css4 import CSS4
    V = HTML16 if vocab == "html" else CSS4                # v6: coarse vocabulary, wide margins between names
    return min(V, key=lambda n: ((np.array([int(V[n][i:i + 2], 16) for i in (1, 3, 5)]) - c) ** 2).sum())


def color_note(note, name, first=False):
    """v5: replace the first colour word of a 'Gone:' note with the measured name (prefix it if there is none)."""
    global COLOR_WORDS
    if note is None or name is None or not note.startswith("Gone: "):
        return note
    if COLOR_WORDS is None:
        from css4 import CSS4
        w = set(CSS4) | {"pink", "light blue", "dark blue", "light green", "dark green", "dark red", "light red"}
        COLOR_WORDS = re.compile(r"\b(" + "|".join(sorted(w, key=len, reverse=True)) + r")\b", re.I)
    b = note[len("Gone: "):]
    if first:                                              # v6: one object per change region (the measured one)
        b = b.rstrip(".").split(",")[0].strip() + "."
    return "Gone: " + (COLOR_WORDS.sub(name, b, count=1) if COLOR_WORDS.search(b) else re.sub(r"^((an?|the) )?", lambda g: f"{g.group(0) or 'a '}{name} ", b, count=1))


def with_notes(instr, notes):
    if not notes:
        return instr
    return instr + " Notes: " + " ".join(f"({i + 1}) {n}" for i, n in enumerate(notes[-K:]))


class VisualMemory:
    """Visual notes: at each change event the zoomed crop of what was there *before* the change (the content the
    current view no longer shows) is written into the next free tile of a GxG grid image (write-once)."""
    G, S = 3, 42

    def __init__(self, full=False):
        self.img = np.zeros((128, 128, 3), np.uint8)
        self.n, self.full = 0, full                      # full: whole before-frame per tile (keyframe ablation)

    def add(self, after, before, mask):
        m = ChangeNoteWriter.main_region(mask & (np.abs(np.asarray(after).astype(np.int16) - np.asarray(before).astype(np.int16)).max(-1) > DPIX))
        if m.sum() < NPIX // 2 or self.n >= self.G * self.G:
            return False
        crop = (Image.fromarray(np.asarray(before)).resize((self.S, self.S), Image.BILINEAR) if self.full
                else ChangeNoteWriter.zoom(before, m).resize((self.S, self.S), Image.NEAREST))
        r, c = divmod(self.n, self.G)
        self.img[1 + r * self.S:1 + (r + 1) * self.S, 1 + c * self.S:1 + (c + 1) * self.S] = np.asarray(crop)[..., :3]
        self.n += 1
        return True
