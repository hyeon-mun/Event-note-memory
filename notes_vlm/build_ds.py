"""Write VLM event notes for every demo of a task and build a LeRobot-v3 copy whose per-frame task string is
instruction + notes written so far. Videos are symlinked.
usage: build_ds.py TID SRC_DIR DST_DIR [V]   (V=2: Scribe write rule, no burst merging; V=3: + change notes)
"""
import glob, json, os, shutil, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

from notes import color_note, gone_note, measured_color, BURST, ChangeNoteWriter, NoteMemory, NoteWriter, episode_events, with_notes

tid, src, dst = int(sys.argv[1]), sys.argv[2], sys.argv[3]
v = int(sys.argv[4]) if len(sys.argv) > 4 else 1
p8 = v == 8                                                     # v8 = periodic baseline: full-frame scene caption every PERIOD steps
v7 = v == 7                                                     # v7 = v6 without the t=0 scene note
v6 = v in (6, 7)                                                # v6 = v5 with the HTML colour vocabulary, first object only
v5 = v in (5, 6, 7)                                                  # v5 = v4 + measured colour name
v4, v = v in (4, 5, 6, 7), 3 if v in (4, 5, 6, 7, 8) else v          # v4 = v3 captions, only the 'gone' side kept
PERIOD = 50
mm = f"/data/workspaces/mikasa/data_mikasa_robo/mm/t{tid:02d}"
top = np.load(f"{mm}/top.npy", mmap_mode="r"); state = np.load(f"{mm}/state.npy")
st, L = np.load(f"{mm}/ep_start.npy"), np.load(f"{mm}/ep_len.npy")
base = pq.read_table(f"{src}/meta/tasks.parquet").to_pandas().index[0]

ev_json = f"{os.path.dirname(dst.rstrip('/'))}/t{tid:02d}_events{'_v8' if p8 else f'_v{v}' if v > 1 else ''}.json"
cache = {}                                                       # captions are deterministic: reuse earlier ones
if os.path.exists(ev_json):
    cache = {(int(e), t): c for e, v in json.load(open(ev_json)).items() for t, c in v}
ev = {e: episode_events(top[st[e]:st[e] + L[e]], state[st[e]:st[e] + L[e], :None if v == 3 else 3], v) for e in range(len(L))}
before = {}
if p8:
    ev = {e: [(t, None, None) for t in range(0, int(L[e]), PERIOD)] for e in range(len(L))}
if v == 3:                                                       # (t, frame before the change) -> t
    before = {(e, t): (tb, m) for e in ev for t, tb, m in ev[e]}
    ev = {e: [x[0] for x in ev[e]] for e in ev}
flat = [(e, t) for e in ev for t in ev[e]]
todo = [k for k in flat if k not in cache]
print("events/episode", np.mean([len(v) for v in ev.values()]), "total", len(flat), "to caption", len(todo), flush=True)
if todo and v == 3:
    items = [(top[st[e] + t], top[st[e] + before[(e, t)][0]] if t > 0 and not p8 else None, before[(e, t)][1]) for e, t in todo]
    cache.update(zip(todo, ChangeNoteWriter()(items)))
elif todo:
    cache.update(zip(todo, NoteWriter()([top[st[e] + t] for e, t in todo])))
notes = {}
for e, t in flat:
    if cache[(e, t)] is not None:
        notes.setdefault(e, []).append((t, cache[(e, t)]))
json.dump({int(e): v for e, v in notes.items()}, open(ev_json, "w"), indent=0)
if v4:
    notes = {e: [(t, gone_note(n)) for t, n in x if gone_note(n) is not None] for e, x in notes.items()}
if v5:
    notes = {e: [(t, color_note(n, measured_color(top[st[e] + t], top[st[e] + before[(e, t)][0]], before[(e, t)][1], "html" if v6 else "css4") if t > 0 else None, v6))
                 for t, n in x] for e, x in notes.items()}

if v7:
    notes = {e: [(t, n) for t, n in x if t > 0] for e, x in notes.items()}

# per-frame task strings
task_of = {}
def frame_tasks(e, T):
    out, mem = [], NoteMemory(0 if v >= 2 else BURST)
    evs = dict(notes.get(e, []))
    for t in range(T):
        if t in evs:
            mem.add(t, evs[t])
        out.append(task_of.setdefault(with_notes(base, mem.notes), len(task_of)))
    return np.array(out, np.int64)

shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
os.symlink(f"{src}/videos", f"{dst}/videos")
shutil.copytree(f"{src}/meta", f"{dst}/meta")
per_ep = {}
for f in sorted(glob.glob(f"{src}/data/chunk-*/file-*.parquet")):
    t = pq.read_table(f).to_pandas()
    ti = np.zeros(len(t), np.int64)
    for e, idx in t.groupby("episode_index").indices.items():
        fi = t["frame_index"].to_numpy()[idx]
        ft = frame_tasks(int(e), int(fi.max()) + 1)
        ti[idx] = ft[fi]; per_ep[int(e)] = sorted(set(ft.tolist()))
    t["task_index"] = ti
    out = f.replace(src, dst); os.makedirs(os.path.dirname(out), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False, schema=pq.read_schema(f)), out)
tasks = sorted(task_of, key=task_of.get)
pd.DataFrame({"task_index": np.arange(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(f"{dst}/meta/tasks.parquet")
for f in glob.glob(f"{dst}/meta/episodes/chunk-*/file-*.parquet"):
    e = pq.read_table(f).to_pandas()
    e["tasks"] = [[tasks[j] for j in per_ep[int(i)]] for i in e["episode_index"]]
    pq.write_table(pa.Table.from_pandas(e, preserve_index=False, schema=pq.read_schema(f)), f)
info = json.load(open(f"{dst}/meta/info.json")); info["total_tasks"] = len(tasks)
json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
print("tasks", len(tasks), "example:", tasks[-1][:300], flush=True)
