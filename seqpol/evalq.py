"""Global evaluation queue.

Drop job files into QDIR/*.json: {"ckpt":..., "tasks":"0,3", "out":DIR, "extra":["--guidance","2"], "episodes":50}
The scheduler expands every job whose ckpt exists into (task, episode-range) parts, runs them over a pool of
GPU slots (QDIR/slots.json: {"6": 8, "7": 8, "2": 4}; re-read every loop), and writes DIR/results.json when a job
is complete. Finished jobs move to QDIR/done/.
"""
import glob, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import task_table

QDIR = sys.argv[1] if len(sys.argv) > 1 else "/data/workspaces/mikasa/research/evalq"
os.makedirs(f"{QDIR}/done", exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
tt = task_table()


def parts_of(job):
    tids = [k for k, v in tt.items() if v["split"] == job["tasks"]] if job["tasks"] in ("short", "medium", "long") \
        else [int(x) for x in str(job["tasks"]).split(",")]
    n = job.get("episodes", 50); out = []
    for t in tids:
        ch = 25 if tt[t]["max_len"] <= 100 else 10 if tt[t]["max_len"] <= 400 else 5
        out += [(t, lo, min(ch, n - lo)) for lo in range(0, n, ch)]
    return out


def merge(job):
    res = {}
    for f in sorted(glob.glob(f"{job['out']}/parts/*.json")):
        for r in json.load(open(f))["results"]:
            d = res.setdefault(r["tid"], dict(tid=r["tid"], mem_type=r["mem_type"], split=r["split"], successes=[], seeds=[]))
            d["successes"] += r["successes"]; d["seeds"] += r["seeds"]
    for d in res.values():
        d["sr"] = float(np.mean(d["successes"])); d["n"] = len(d["successes"])
    rs = sorted(res.values(), key=lambda d: d["tid"])
    json.dump(dict(job=job, mean_sr=float(np.mean([d["sr"] for d in rs])), results=rs),
              open(f"{job['out']}/results.json", "w"), indent=1)


running = {}   # (gpu, idx) -> (proc, partfile)
while True:
    slots = json.load(open(f"{QDIR}/slots.json")) if os.path.exists(f"{QDIR}/slots.json") else {"6": 4, "7": 4}
    for key in list(running):
        if running[key][0].poll() is not None:
            running.pop(key)
    busy = {pf for _, pf in running.values()}
    # parts started by an earlier scheduler instance (restart-safe): count them against their GPU's slots
    mine = {p.pid for p, _ in running.values()}
    orphan_gpu = {}
    for l in subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines():
        if "seqpol/evaluate.py" in l and "--out " in l:
            pid = int(l.split()[0]); busy.add(l.split("--out ")[1].split()[0])
            if pid in mine:
                continue
            try:
                envs = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
                g = next(e.split(b"=")[1].decode() for e in envs if e.startswith(b"CUDA_VISIBLE_DEVICES="))
                orphan_gpu[g] = orphan_gpu.get(g, 0) + 1
            except Exception:
                pass
    pending = []
    for jf in sorted(glob.glob(f"{QDIR}/*.json")):
        if jf.endswith("slots.json"):
            continue
        job = json.load(open(jf))
        if not os.path.exists(job["ckpt"]):
            continue
        os.makedirs(f"{job['out']}/parts", exist_ok=True)
        todo, complete = [], True
        for t, lo, n in parts_of(job):
            pf = f"{job['out']}/parts/t{t:02d}_{lo:03d}.json"
            if not os.path.exists(pf):
                complete = False
                if pf not in busy:
                    todo.append(((job.get("priority", 5), -tt[t]["max_len"] * n), job, t, lo, n, pf))
        if complete and not busy & {f"{job['out']}/parts/t{t:02d}_{lo:03d}.json" for t, lo, _ in parts_of(job)}:
            merge(job); os.rename(jf, f"{QDIR}/done/{os.path.basename(jf)}")
            print(time.strftime("%H:%M"), "done", job["out"], flush=True)
        pending += todo
    pending.sort(key=lambda x: x[0])
    for g, k in slots.items():
        for i in range(max(0, k - orphan_gpu.get(g, 0))):
            if (g, i) in running or not pending:
                continue
            _, job, t, lo, n, pf = pending.pop(0)
            cmd = [PY, "-W", "ignore", f"{HERE}/evaluate.py", "--ckpt", job["ckpt"], "--tasks", str(t),
                   "--episodes", str(n), "--ep_lo", str(lo), "--out", pf] + list(job.get("extra", []))
            running[(g, i)] = (subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=g),
                                                stdout=subprocess.DEVNULL, stderr=open(pf + ".err", "w")), pf)
    time.sleep(10)
