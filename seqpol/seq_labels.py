"""True cue sequence of every demo episode of a Chain/Seq task (seed matched via episode length + initial proprio).
usage: seq_labels.py TID   -> cue_seq_tTID.npy (n_eps, seq_len)"""
import glob, json, sys, numpy as np
tid = int(sys.argv[1]); sys.argv = sys.argv[:1]
from evaluate import task_table, BenchmarkConfig, make_benchmark_env
tt = task_table()[tid]
mm = f"/data/workspaces/mikasa/data_mikasa_robo/mm/t{tid:02d}"
state, st, L = np.load(f"{mm}/state.npy", mmap_mode="r"), np.load(f"{mm}/ep_start.npy"), np.load(f"{mm}/ep_len.npy")
meta = [m for m in (json.load(open(f)) for f in glob.glob("/data/workspaces/mikasa/data_mikasa_robo/data_lerobot/*/source_rlds_metadata.json")) if m["env_id"] == tt["env_id"]][0]
env = make_benchmark_env(tt["env_id"], BenchmarkConfig(n_episodes=1, start_seed=0, sim_backend="cpu")); u = env.unwrapped
out, bad = [], 0
for e in range(len(L)):
    best = None
    for i in [i for i, l in enumerate(meta["episode_lengths"]) if l == L[e]]:
        obs, info = env.reset(seed=meta["episode_seeds"][i])
        d = float(np.abs(obs["proprio"][0].cpu().numpy() - state[st[e]]).max())
        if best is None or d < best[0]: best = (d, u.true_color_indices[0].tolist(), info.get("oracle_info"))
    bad += best[0] > 1e-4
    out.append(best[1])
    if e < 3: print(e, best, flush=True)
np.save(f"cue_seq_t{tid}.npy", np.array(out)); print("done, unmatched", bad, np.array(out).shape)
