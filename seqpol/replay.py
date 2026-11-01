"""Replay recorded demo actions in the eval env (seed matched via episode length + initial proprio).
usage: replay.py TID N_EPS BACKEND"""
import json, sys, numpy as np, torch
sys.argv, (tid, n, backend) = sys.argv[:1], (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
from evaluate import task_table, BenchmarkConfig, make_benchmark_env
tt = task_table()[tid]
mm = f"/data/workspaces/mikasa/data_mikasa_robo/mm/t{tid:02d}"
act, state = np.load(f"{mm}/action.npy", mmap_mode="r"), np.load(f"{mm}/state.npy", mmap_mode="r")
st, L = np.load(f"{mm}/ep_start.npy"), np.load(f"{mm}/ep_len.npy")
import glob
meta = None
for f in glob.glob("/data/workspaces/mikasa/data_mikasa_robo/data_lerobot/*/source_rlds_metadata.json"):
    m = json.load(open(f))
    if m["env_id"] == tt["env_id"]: meta = m
lens, seeds = meta["episode_lengths"], meta["episode_seeds"]
env = make_benchmark_env(tt["env_id"], BenchmarkConfig(n_episodes=1, start_seed=0, sim_backend=backend))
u = env.unwrapped
ok = 0
for e in range(n):
    cand = [i for i, l in enumerate(lens) if l == L[e]]
    best = None
    for i in cand:
        obs, _ = env.reset(seed=seeds[i])
        d = float(np.abs(obs["proprio"][0].cpu().numpy() - state[st[e]]).max())
        if best is None or d < best[0]: best = (d, i)
    obs, _ = env.reset(seed=seeds[best[1]])
    succ, err = False, 0.0
    for t in range(L[e]):
        obs, r, term, trunc, info = env.step(torch.as_tensor(np.asarray(act[st[e] + t]))[None].float())
        if t + 1 < L[e]: err = max(err, float(np.abs(obs["proprio"][0, :3].cpu().numpy() - state[st[e] + t + 1, :3]).max()))
        succ |= bool(info["success"].any())
    extra = ""
    if hasattr(u, "touched_cubes_idx"):
        extra = f" touched={u.touched_cubes_idx[0].tolist()} true={u.true_color_indices[0].tolist()}"
    ok += succ
    print(f"ep{e} len{L[e]} match_d={best[0]:.4f} succ={succ} max_xyz_err={err:.4f}{extra}", flush=True)
print("replay success", ok, "/", n)
