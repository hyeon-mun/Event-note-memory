"""Policy server (runs in the LeRobot env). Protocol over multiprocessing.connection, numpy-version agnostic:
  ("reset", task_str) -> "ok"
  ("act", {name: (bytes, shape, dtype_str)}, task_str) -> (bytes, shape, "float32")   one 7-d (8-d for the joint-delta -v0 tasks) action
usage: server.py CKPT_DIR PORT
"""
import ctypes, os, signal, sys
from multiprocessing.connection import Listener
import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)                       # exit with the client process
ckpt, port = sys.argv[1], int(sys.argv[2])
if os.environ.get("MEM_GB"):                                              # hard cap when sharing the GPU
    torch.cuda.set_per_process_memory_fraction(float(os.environ["MEM_GB"]) * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.device = "cuda"
policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to("cuda").eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt)
ADIM = cfg.output_features["action"].shape[0]                            # 7 (EEF delta, -VLA tasks) or 8 (joint delta, -v0)

DEADBAND = float(os.environ.get("DEADBAND", 0))                           # zero tiny arm actions (|a| < DEADBAND)
notes_v8 = os.environ.get("NOTES_V") == "8"                                 # v8: periodic baseline, full-frame scene caption every 50 steps
notes_v7 = os.environ.get("NOTES_V") == "7"                                 # v7: v6 without the t=0 scene note
notes_v6 = os.environ.get("NOTES_V") in ("6", "7")                                 # v6: v5, HTML colour names, first object
notes_v5 = os.environ.get("NOTES_V") in ("5", "6", "7")                                 # v5: v4 + measured colour name
notes_v4 = os.environ.get("NOTES_V") in ("4", "5", "6", "7")                                 # v4: v3 notes, only the removed (before) side
notes_v3 = os.environ.get("NOTES_V") == "3" or notes_v4 or notes_v8                               # v3: v2 write rule + zoomed change notes
notes_v2 = os.environ.get("NOTES_V") == "2" or notes_v3                     # v2 write rule (Scribe, settle before writing)
notes_on = os.environ.get("NOTES") == "1"                                 # online VLM event notes (notes_vlm/notes.py)
vmem_on = "observation.images.memory" in cfg.input_features              # visual notes (notes.VisualMemory) as a camera
note_log = open(os.environ["NOTE_LOG"], "a") if os.environ.get("NOTE_LOG") else None
if notes_on or vmem_on:
    sys.path.insert(0, "/data/workspaces/mikasa/research/notes_vlm")
    from notes import color_note, gone_note, measured_color, ChangeNoteWriter, EventDetector, NoteMemory, NoteWriter, Scribe, VisualMemory, with_notes
if notes_on:
    writer = ChangeNoteWriter() if notes_v3 else NoteWriter()


def chroma420(x):
    """Match the training videos: yuv420 storage halves the chroma resolution (small coloured objects become
    2x2-blocky in colour while luminance keeps full resolution). BT.601 full range, 2x2 mean, nearest upsample."""
    x = x.astype(np.float32); R, G, B = x[..., 0], x[..., 1], x[..., 2]
    Y = .299 * R + .587 * G + .114 * B
    U = -.168736 * R - .331264 * G + .5 * B
    V = .5 * R - .418688 * G - .081312 * B
    h, w = Y.shape
    U, V = [c.reshape(h // 2, 2, w // 2, 2).mean((1, 3)).repeat(2, 0).repeat(2, 1) for c in (U, V)]
    out = np.stack([Y + 1.402 * V, Y - .344136 * U - .714136 * V, Y + 1.772 * U], -1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


C420 = os.environ.get("C420") == "1"


def arr(t):
    b, shape, dt = t
    return np.frombuffer(b, dtype=dt).reshape(shape)


with Listener(("127.0.0.1", port), authkey=b"seqpol") as lst:
    print("READY", flush=True)
    conn = lst.accept()                                                     # one client per server
    if True:
        try:
            while True:
                msg = conn.recv()
                if msg[0] == "reset":
                    policy.reset(); conn.send("ok")
                    swap = msg[1] if len(msg) > 1 and msg[1] else None     # control: force this colour into "Gone:" notes
                    if notes_on:
                        det, mem, pending, t, dump = EventDetector(), NoteMemory(), False, 0, []
                        if notes_v2: det, mem = Scribe(full=notes_v3), NoteMemory(0)
                    if vmem_on:
                        vdet, vm, vt = Scribe(full=True), VisualMemory(os.environ.get("VMEM_FULL") == "1"), 0
                    if note_log: print("RESET", file=note_log, flush=True)
                elif msg[0] == "act":
                    o, task = msg[1], msg[2]
                    if C420:
                        rgb_u8 = arr(o["rgb"])
                        o["rgb"] = (np.concatenate([chroma420(rgb_u8[..., :3]), chroma420(rgb_u8[..., 3:])], -1).tobytes(), rgb_u8.shape, "uint8")
                    if notes_on:                                    # caption the frame after an event (t=0: the first frame)
                        top_u8 = arr(o["rgb"])[..., :3]
                        ev = det(np.ascontiguousarray(top_u8), arr(o["proprio"])[:None if notes_v3 else 3].astype(np.float64))
                        if notes_v8:
                            if t % 50 == 0:
                                n = writer([(top_u8, None, None)])[0]
                                if n is not None: mem.add(t, n)
                                if note_log: print(t, n, file=note_log, flush=True)
                        elif (ev and t == 0) or pending or (notes_v2 and ev):
                            if notes_v3:
                                n = writer([(top_u8, det.before, det.mask) if t > 0 else (top_u8, None, None)])[0]
                                if notes_v4: n = gone_note(n)
                                if notes_v5 and t > 0: n = color_note(n, measured_color(top_u8, det.before, det.mask, "html" if notes_v6 else "css4"), notes_v6)
                                if notes_v7 and t == 0: n = None
                                if swap and t > 0 and n is not None: n = color_note(n, swap, True)
                                if n is not None: mem.add(t, n)
                                if note_log: print(t, n, file=note_log, flush=True)
                            else:
                                mem.add(t, writer([top_u8])[0])
                                if note_log: print(t, mem.notes[-1], file=note_log, flush=True)
                        if os.environ.get("NOTE_DUMP"):
                            dump.append((top_u8.copy(), arr(o["proprio"]).copy()))
                            if t % 50 == 0: np.save(os.environ["NOTE_DUMP"], np.array([d[0] for d in dump])); np.save(os.environ["NOTE_DUMP"] + ".prop.npy", np.array([d[1] for d in dump]))
                        pending = ev and t > 0 and not notes_v2
                        t += 1
                        task = with_notes(task, mem.notes)
                    if vmem_on:                                     # same write rule as the demos (build_vis.py)
                        top_u8 = arr(o["rgb"])[..., :3]
                        if vdet(np.ascontiguousarray(top_u8), arr(o["proprio"]).astype(np.float64)) and vt > 0:
                            if vm.add(top_u8, vdet.before, vdet.mask):
                                if note_log: print(vt, "VIS", vm.n, file=note_log, flush=True)
                                if os.environ.get("VMEM_DUMP"): np.save(f"{os.environ['VMEM_DUMP']}_{vt:03d}.npy", np.stack([vm.img, top_u8, vdet.before]))
                        vt += 1
                    rgb = torch.from_numpy(arr(o["rgb"]).copy()).float() / 255.0          # (128,128,6)
                    batch = {"observation.images.top": rgb[..., :3].permute(2, 0, 1)[None],
                             "observation.images.wrist": rgb[..., 3:].permute(2, 0, 1)[None],
                             "observation.state": torch.from_numpy(arr(o["proprio"]).copy()).float()[None],
                             "task": [task]}
                    if vmem_on:
                        batch["observation.images.memory"] = torch.from_numpy(chroma420(vm.img) if C420 else vm.img.copy()).float().permute(2, 0, 1)[None] / 255.0
                    with torch.no_grad():
                        a = post(policy.select_action(pre(batch)))
                    a = a.reshape(-1)[:ADIM].float().cpu().numpy()
                    if DEADBAND and np.abs(a[:-1]).max() < DEADBAND:  # demos hold with exactly zero actions while waiting
                        a[:-1] = 0
                    conn.send((a.tobytes(), a.shape, "float32"))
                elif msg[0] == "close":
                    break
        except EOFError:
            pass
        conn.close()
