"""Success rate of the RC-Long models binned by the memory delay (empty-phase length) of each episode.
The delay is drawn by the env from the episode seed, so it is recomputed here by resetting the env (no policy needed).
usage (MIKASA venv):  cd seqpol && python ../tools/delay_bins.py   -> logs/rc_long_phases.json + table on stdout
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "seqpol"))
import mikasa_robo_suite.vla.memory_envs  # noqa: F401
from mikasa_robo_suite.vla.benchmarking import BenchmarkConfig, make_benchmark_env
from evaluate import task_table

L = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
S0 = 4242424242
BINS = [(50, 183), (184, 316), (317, 450)]
path = f"{L}/rc_long_phases.json"
if os.path.exists(path):
    D = {tuple(map(int, k.split("_"))): v for k, v in json.load(open(path)).items()}
else:
    D = {}
    for tid in (50, 51, 52):
        env = make_benchmark_env(task_table()[tid]["env_id"], BenchmarkConfig(n_episodes=150, start_seed=S0, sim_backend="cpu"))
        for i in range(150 if tid == 52 else 50):
            env.reset(seed=S0 + i); u = env.unwrapped
            D[(tid, i)] = (int(u.cue_steps_per_env[0]), int(u.empty_steps_per_env[0]))
        env.close()
    json.dump({f"{a}_{b}": v for (a, b), v in D.items()}, open(path, "w"))


def succ(run, tid):
    out = {}
    for f in [f"{L}/eval_db_{run}_020000.json"] + glob.glob(f"{L}/eval_x52_{run}*.json"):
        try:
            R = json.load(open(f))
        except Exception:
            continue
        for r in R["results"]:
            if r["tid"] == tid:
                out.update({s - S0: ok for s, ok in zip(r["seeds"], r["successes"])})
    return out


print("| model | " + " | ".join(f"t{t} delay {lo}-{hi}" for t in (50, 51, 52) for lo, hi in BINS) + " |")
for run in ["pi05_t5x_base", "pi05_t5x_v2", "pi05_t5x_v4", "pi05_t5x_v6", "pi05_t5x_oracle"]:
    cells = []
    for tid in (50, 51, 52):
        s = succ(run, tid)
        for lo, hi in BINS:
            k = [i for i in s if lo <= D[(tid, i)][1] <= hi]
            cells.append(f"{np.mean([s[i] for i in k]):.2f} (n={len(k)})" if k else "-")
    print(f"| {run} | " + " | ".join(cells) + " |")
