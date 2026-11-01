"""Parallel evaluation: shards (task, episode-range) jobs over GPUs/processes, then merges.

python peval.py --ckpt X --tasks 0,3,12 --out DIR [--gpus 6,7 --procs_per_gpu 12 --chunk 10 -- extra evaluate.py args]
"""
import argparse, glob, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import task_table

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--gpus", default="6,7"); ap.add_argument("--procs_per_gpu", type=int, default=12)
ap.add_argument("--episodes", type=int, default=50); ap.add_argument("--chunk", type=int, default=0)
a, extra = ap.parse_known_args()
tt = task_table()
tids = [k for k, v in tt.items() if v["split"] == a.tasks] if a.tasks in ("short", "medium", "long", ) else \
       (list(tt) if a.tasks == "all" else [int(x) for x in a.tasks.split(",")])
os.makedirs(f"{a.out}/parts", exist_ok=True)
jobs = []
for t in tids:
    ch = a.chunk or (25 if tt[t]["max_len"] <= 100 else 10 if tt[t]["max_len"] <= 400 else 5)
    for lo in range(0, a.episodes, ch):
        jobs.append((t, lo, min(ch, a.episodes - lo)))
jobs.sort(key=lambda j: -tt[j[0]]["max_len"] * j[2])     # longest first
gpus = a.gpus.split(","); slots = [g for g in gpus for _ in range(a.procs_per_gpu)]
here = os.path.dirname(os.path.abspath(__file__)); py = sys.executable
running, t0 = {}, time.time()
while jobs or running:
    for s in list(running):
        if running[s][0].poll() is not None:
            running.pop(s)
    free = [i for i in range(len(slots)) if i not in running]
    while free and jobs:
        i = free.pop(); t, lo, n = jobs.pop(0)
        out = f"{a.out}/parts/t{t:02d}_{lo:03d}.json"
        if os.path.exists(out):
            continue
        cmd = [py, "-W", "ignore", f"{here}/evaluate.py", "--ckpt", a.ckpt, "--tasks", str(t), "--episodes", str(n),
               "--ep_lo", str(lo), "--out", out] + extra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=slots[i])
        running[i] = (subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=open(out + ".err", "w")), (t, lo))
    time.sleep(5)
# merge
res = {}
for f in sorted(glob.glob(f"{a.out}/parts/*.json")):
    for r in json.load(open(f))["results"]:
        d = res.setdefault(r["tid"], dict(tid=r["tid"], mem_type=r["mem_type"], split=r["split"], successes=[], seeds=[]))
        d["successes"] += r["successes"]; d["seeds"] += r["seeds"]
for d in res.values():
    d["sr"] = float(np.mean(d["successes"])); d["n"] = len(d["successes"])
json.dump(dict(ckpt=a.ckpt, extra=extra, results=sorted(res.values(), key=lambda d: d["tid"])),
          open(f"{a.out}/results.json", "w"), indent=1)
srs = [d["sr"] for d in res.values()]
print(f"mean SR {np.mean(srs):.3f} over {len(srs)} tasks  ({time.time() - t0:.0f}s)")
for d in sorted(res.values(), key=lambda d: d["tid"]):
    print(f"  t{d['tid']:02d} {d['mem_type']:<12} {d['sr']:.2f} (n={d['n']})")
