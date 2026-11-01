"""Decode MIKASA-Robo-VLA LeRobot v3 tasks into flat uint8 memmaps for fast training.

out/<task>/ rgb.npy (N,128,128,6) uint8 [top|wrist], state.npy (N,7), action.npy (N,7),
            ep_start.npy, ep_len.npy (E,), meta.json
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool
import numpy as np, av, pyarrow.parquet as pq

import csv
TASK_ID = {r["Name"]: int(r["id"]) for r in csv.DictReader(open("/data/workspaces/mikasa/MIKASA-Robo/mikasa_robo_vla_envs.csv"), delimiter=";")}

def decode_cam(args):
    task_dir, cam, out_path, n_total = args
    files = sorted(glob.glob(f"{task_dir}/videos/observation.images.{cam}/chunk-*/file-*.mp4"))
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(n_total, 128, 128, 3))
    i = 0
    for f in files:
        c = av.open(f); s = c.streams.video[0]; s.thread_type = "AUTO"
        for fr in c.decode(s):
            if i >= n_total: break
            arr[i] = fr.to_ndarray(format="rgb24"); i += 1
        c.close()
    arr.flush()
    return cam, i

def prep(task_dir, out_root, pool):
    name = os.path.basename(task_dir.rstrip("/"))
    src = json.load(open(f"{task_dir}/source_rlds_metadata.json"))
    out = f"{out_root}/t{TASK_ID[src['env_id']]:02d}"
    if os.path.exists(f"{out}/meta.json"):
        return name, "skip"
    os.makedirs(out, exist_ok=True)
    tabs = [pq.read_table(f) for f in sorted(glob.glob(f"{task_dir}/data/chunk-*/file-*.parquet"))]
    st = np.concatenate([np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)) for t in tabs]).astype(np.float32)
    ac = np.concatenate([np.stack(t.column("action").to_numpy(zero_copy_only=False)) for t in tabs]).astype(np.float32)
    epi = np.concatenate([t.column("episode_index").to_numpy() for t in tabs])
    tki = np.concatenate([t.column("task_index").to_numpy() for t in tabs])
    n = len(st)
    eps = pq.read_table(sorted(glob.glob(f"{task_dir}/meta/episodes/chunk-*/file-*.parquet"))[0]).to_pandas()
    ep_start = eps["dataset_from_index"].to_numpy(); ep_len = eps["length"].to_numpy()
    assert (np.diff(epi) >= 0).all() and ep_start[-1] + ep_len[-1] == n
    res = pool.map(decode_cam, [(task_dir, c, f"{out}/{c}.npy", n) for c in ("top", "wrist")])
    for cam, cnt in res:
        assert cnt == n, (name, cam, cnt, n)
    np.save(f"{out}/state.npy", st); np.save(f"{out}/action.npy", ac)
    np.save(f"{out}/ep_start.npy", ep_start); np.save(f"{out}/ep_len.npy", ep_len)
    instr = eps["tasks"].iloc[0][0]
    tdf = pq.read_table(f"{task_dir}/meta/tasks.parquet").to_pandas()
    tasks = [None] * len(tdf)
    for txt, row in tdf.iterrows(): tasks[int(row["task_index"])] = txt
    np.save(f"{out}/ep_task.npy", tki[ep_start])
    json.dump({"task": name, "env_id": src.get("env_id"), "instruction": instr, "tasks": tasks, "n": int(n), "episodes": int(len(ep_len))},
              open(f"{out}/meta.json", "w"), indent=1)
    return name, n

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/workspaces/mikasa/data_mikasa_robo/data_lerobot")
    ap.add_argument("--out", default="/data/workspaces/mikasa/data_mikasa_robo/mm")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(f"{a.src}/*/") if os.path.exists(f"{d}/meta/info.json"))
    if a.tasks: dirs = [d for d in dirs if os.path.basename(d.rstrip('/')) in a.tasks]
    # parallelise across tasks: outer pool of tasks, each with 2 cam workers
    from concurrent.futures import ThreadPoolExecutor
    with Pool(2 * max(1, a.workers // 2)) as pool, ThreadPoolExecutor(a.workers // 2) as ex:
        for r in ex.map(lambda d: prep(d, a.out, pool), dirs):
            print(r, flush=True)
