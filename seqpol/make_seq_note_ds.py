"""Oracle-note copy of a Chain/Seq task: instruction + 'The cue colors were A, B, C.' (true sequence, env colour names).
usage: make_seq_note_ds.py SRC_DIR DST_DIR TID   (needs cue_seq_tTID.npy from seq_labels.py)"""
import json, os, shutil, sys, glob
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
NAMES = ["Red", "Lime", "Blue", "Yellow", "Magenta", "Cyan", "Maroon", "Olive", "Teal"]   # COLOR_MAPPING order


def seq_note(seq):
    return " The cue colors were " + ", ".join(NAMES[int(c)] for c in seq) + "."


if __name__ == "__main__":
    src, dst, tid = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seqs = np.load(f"{os.path.dirname(os.path.abspath(__file__))}/cue_seq_t{tid}.npy")
    base = pq.read_table(f"{src}/meta/tasks.parquet").to_pandas().index[0]
    tasks = sorted({base + seq_note(s) for s in seqs}); idx = {t: i for i, t in enumerate(tasks)}
    lab = np.array([idx[base + seq_note(s)] for s in seqs])
    shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst)
    os.symlink(f"{src}/videos", f"{dst}/videos")
    shutil.copytree(f"{src}/meta", f"{dst}/meta")
    pd.DataFrame({"task_index": np.arange(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(f"{dst}/meta/tasks.parquet")
    for f in glob.glob(f"{src}/data/chunk-*/file-*.parquet"):
        t = pq.read_table(f).to_pandas()
        t["task_index"] = lab[t["episode_index"].to_numpy()].astype(np.int64)
        out = f.replace(src, dst); os.makedirs(os.path.dirname(out), exist_ok=True)
        pq.write_table(pa.Table.from_pandas(t, preserve_index=False, schema=pq.read_schema(f)), out)
    for f in glob.glob(f"{dst}/meta/episodes/chunk-*/file-*.parquet"):
        e = pq.read_table(f).to_pandas()
        e["tasks"] = [[tasks[lab[i]]] for i in e["episode_index"]]
        pq.write_table(pa.Table.from_pandas(e, preserve_index=False, schema=pq.read_schema(f)), f)
    info = json.load(open(f"{dst}/meta/info.json")); info["total_tasks"] = len(tasks)
    json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
    print("tasks", len(tasks), tasks[0])
