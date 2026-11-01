"""Copy a LeRobot-v3 task dir with per-episode instructions that include a 'note' (e.g. the oracle cue).

Videos are symlinked; data/meta parquet files are rewritten with new task strings / indices.
usage: make_note_ds.py SRC_DIR DST_DIR TID
"""
import json, os, shutil, sys, glob
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cue_labels import cluster, color_clusters, cue_colors, cue_shapes

NAMES = {"red": (220, 10, 10), "green": (10, 225, 20), "blue": (5, 5, 230), "yellow": (238, 220, 14),
         "magenta": (222, 8, 228), "cyan": (10, 220, 220), "orange": (240, 140, 10), "white": (230, 230, 230),
         "purple": (120, 20, 200), "maroon": (160, 10, 1), "teal": (8, 167, 166)}

src, dst, tid = sys.argv[1], sys.argv[2], int(sys.argv[3])
SHAPE = {16: 3, 17: 5, 18: 9, 53: 3, 54: 5, 55: 9}
COUNT = (47, 48, 49)                                # BlinkCount: true count per demo from oracle_labels.py
if tid in COUNT:
    n = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/oracle_t{tid}.npy")[:, 0]
    vals = sorted(set(n.tolist())); lab = np.array([vals.index(x) for x in n]); k = len(vals)
    cname, what = [str(v) for v in vals], "count"
elif tid in SHAPE:                                   # shapes have no names we can read off pixels: use letters
    k = SHAPE[tid]; lab, C = cluster(cue_shapes(tid), k)
    np.save(f"{os.path.dirname(os.path.abspath(__file__))}/cue_shape_centers_t{tid}.npy", C)
    cname, what = [chr(65 + j) for j in range(k)], "shape"
else:
    k = {13: 3, 14: 5, 15: 9, 50: 3, 51: 5, 52: 9}[tid]
    lab, C = color_clusters(tid, k)
    cname, what = [min(NAMES, key=lambda n: np.sum((np.array(NAMES[n]) - c) ** 2)) for c in C], "color"
print("clusters", cname, np.bincount(lab))
shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
os.symlink(f"{src}/videos", f"{dst}/videos")
shutil.copytree(f"{src}/meta", f"{dst}/meta")
base = pq.read_table(f"{src}/meta/tasks.parquet").to_pandas().index[0]
tasks = [f"{base} The blue lamp blinked {cname[j]} times." if what == "count" else f"{base} The cue {what} was {cname[j]}."
         for j in range(k)]
pd.DataFrame({"task_index": np.arange(k)}, index=pd.Index(tasks, name="task")).to_parquet(f"{dst}/meta/tasks.parquet")
for f in glob.glob(f"{src}/data/chunk-*/file-*.parquet"):
    t = pq.read_table(f).to_pandas()
    t["task_index"] = lab[t["episode_index"].to_numpy()].astype(np.int64)
    out = f.replace(src, dst); os.makedirs(os.path.dirname(out), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False, schema=pq.read_schema(f)), out)
for f in glob.glob(f"{dst}/meta/episodes/chunk-*/file-*.parquet"):
    e = pq.read_table(f).to_pandas()
    e["tasks"] = [[tasks[lab[i]]] for i in e["episode_index"]]
    pq.write_table(pa.Table.from_pandas(e, preserve_index=False, schema=pq.read_schema(f)), f)
info = json.load(open(f"{dst}/meta/info.json")); info["total_tasks"] = k; json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
print("done", dst)
