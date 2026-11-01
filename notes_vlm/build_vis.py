"""Build a LeRobot-v3 copy of a task dir with an extra camera 'observation.images.memory' = visual notes
(notes.VisualMemory written at the v3 change events of each demo). Task strings and other videos are kept.
usage: build_vis.py TID SRC_DIR DST_DIR [full]
"""
import glob, json, os, shutil, sys
from fractions import Fraction
import av, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

from notes import VisualMemory, episode_events

tid, src, dst = int(sys.argv[1]), sys.argv[2], sys.argv[3]
FULL = len(sys.argv) > 4 and sys.argv[4] == "full"
KEY, TOP = "observation.images.memory", "observation.images.top"
mm = f"/data/workspaces/mikasa/data_mikasa_robo/mm/t{tid:02d}"
top = np.load(f"{mm}/top.npy", mmap_mode="r"); state = np.load(f"{mm}/state.npy")
st, L = np.load(f"{mm}/ep_start.npy"), np.load(f"{mm}/ep_len.npy")
info = json.load(open(f"{src}/meta/info.json")); fps = info["fps"]


def memory_frames(e):
    T = top[st[e]:st[e] + L[e]]
    ev = {t: (tb, m) for t, tb, m in episode_events(T, state[st[e]:st[e] + L[e]], 3) if t > 0}
    vm, out, n = VisualMemory(FULL), np.zeros((L[e], 128, 128, 3), np.uint8), 0
    for t in range(L[e]):
        if t in ev:
            n += vm.add(T[t], T[ev[t][0]], ev[t][1])
        out[t] = vm.img
    return out, n


shutil.rmtree(dst, ignore_errors=True); os.makedirs(f"{dst}/videos")
for k in os.listdir(f"{src}/videos"):
    os.symlink(f"{src}/videos/{k}", f"{dst}/videos/{k}")
shutil.copytree(f"{src}/meta", f"{dst}/meta")
os.symlink(f"{src}/data", f"{dst}/data")

eps = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{dst}/meta/episodes/chunk-*/file-*.parquet"))])
tiles = []
for (c, fi), g in eps.groupby([f"videos/{TOP}/chunk_index", f"videos/{TOP}/file_index"]):
    g = g.sort_values(f"videos/{TOP}/from_timestamp")
    path = f"{dst}/videos/{KEY}/chunk-{c:03d}/file-{fi:03d}.mp4"; os.makedirs(os.path.dirname(path), exist_ok=True)
    with av.open(path, "w") as out:
        s = out.add_stream("libx264", rate=fps); s.width = s.height = 128; s.pix_fmt = "yuv420p"
        s.time_base = Fraction(1, fps); s.options = {"crf": "18", "g": "10"}
        i = 0
        for e, t0, t1 in zip(g["episode_index"], g[f"videos/{TOP}/from_timestamp"], g[f"videos/{TOP}/to_timestamp"]):
            assert abs(t0 * fps - i) < 1e-3 and round((t1 - t0) * fps) == L[e], (e, t0, t1, i, L[e])
            fr, n = memory_frames(int(e)); tiles.append(n)
            for x in fr:
                f = av.VideoFrame.from_ndarray(x, format="rgb24"); f.pts = i; i += 1
                out.mux(s.encode(f))
        out.mux(s.encode())
print("tiles/episode", np.mean(tiles), "max", max(tiles), flush=True)

for f in glob.glob(f"{dst}/meta/episodes/chunk-*/file-*.parquet"):
    e = pq.read_table(f).to_pandas()
    for col in [c for c in e.columns if f"/{TOP}/" in c]:
        e[col.replace(TOP, KEY)] = e[col]
    e.to_parquet(f, index=False)
info["features"][KEY] = json.loads(json.dumps(info["features"][TOP]))
info["features"][KEY]["info"]["video.codec"] = "h264"
json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
stats = json.load(open(f"{dst}/meta/stats.json")); stats[KEY] = stats[TOP]
json.dump(stats, open(f"{dst}/meta/stats.json", "w"), indent=4)
print("done", dst, flush=True)
