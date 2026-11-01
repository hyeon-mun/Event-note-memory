"""Batched development evaluation: N parallel GPU envs per task in one process.

Initial states differ from the canonical num_envs=1 stream (ManiSkill batched seeding), so these numbers are
for development comparisons only; final tables use evaluate.py (canonical protocol).
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, "/data/workspaces/mikasa/MIKASA-Robo")
import mikasa_robo_suite.vla.memory_envs  # noqa: F401
from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

from data import pool_tokens, prop_tf, task_table
from evaluate import PolicyRunner
from featx import encode


class BatchRunner(PolicyRunner):
    def reset_batch(self, instrs, tid=None):
        self.tid = tid
        self.txt = torch.tensor(np.stack([self.text(s) for s in instrs]), device=self.dev)
        self.hist, self.h, self.queue, self.qi = [], None, None, 0

    @torch.no_grad()
    def act_batch(self, obs):
        rgb = obs["rgb"].to(self.dev); B = rgb.shape[0]
        top, wrist = rgb[..., :3], rgb[..., 3:]
        prop = (prop_tf(obs["proprio"].to(self.dev).float()) - self.m.p_mean) / self.m.p_std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = torch.stack([encode(self.dino, top), encode(self.dino, wrist)], 1)
            f = torch.from_numpy(pool_tokens(f.float().cpu().numpy()[:, None], self.a["tokens"])).to(self.dev)
            x = self.m.mem_inputs(f, prop[:, None], self.txt)
            kind = self.a["mem"]
            if kind == "gru":
                mem, self.h = self.m.memory(x, self.h); mem = mem[:, -1]
            else:
                self.hist.append(x)
                if kind == "stack":
                    self.hist = self.hist[-self.m.memory.stack:]
                mem = self.m.memory(torch.cat(self.hist, 1))[0][:, -1]
            if self.queue is None or self.qi >= self.exec_steps:
                react = self.m.reactive(top, wrist, prop, self.txt)
                if self.mem_off:
                    mem = self.m.null_mem.expand_as(mem)
                self.queue = self.m.sample(react.float(), mem.float(), self.flow_steps, self.guidance, tid=self.tid)
                self.qi = 0
        a = self.queue[:, self.qi]; self.qi += 1
        return a.float()


def run_task_batched(runner, tid, n_envs, seed):
    tt = task_table()[tid]
    env = gym.make(tt["env_id"], num_envs=n_envs, obs_mode="rgb", control_mode="pd_ee_delta_pose",
                   reward_mode="normalized_dense", render_mode="all", sim_backend="gpu")
    env = apply_mikasa_vla_wrappers(env, include_overlays=False)
    obs, info = env.reset(seed=[seed + i for i in range(n_envs)])
    instr = info.get("language_instruction", tt["instr"])
    instrs = list(instr) if isinstance(instr, (list, tuple)) else [instr] * n_envs
    runner.reset_batch(instrs, tid)
    ok = torch.zeros(n_envs, dtype=torch.bool); done = torch.zeros(n_envs, dtype=torch.bool)
    for _ in range(int(env.max_episode_steps)):
        a = runner.act_batch(obs).to(env.unwrapped.device)
        obs, r, term, trunc, info = env.step(a)
        ok |= torch.as_tensor(info["success"]).cpu().bool() & ~done
        done |= (torch.as_tensor(term).cpu().bool() | torch.as_tensor(trunc).cpu().bool())
        if done.all():
            break
    env.close()
    return dict(tid=tid, mem_type=tt["mem"], split=tt["split"], sr=float(ok.float().mean()), successes=ok.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--n_envs", type=int, default=50); ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--guidance", type=float, default=1.0); ap.add_argument("--exec", type=int, default=4)
    ap.add_argument("--flow_steps", type=int, default=10); ap.add_argument("--mem_off", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    runner = BatchRunner(a.ckpt, guidance=a.guidance, exec_steps=a.exec, flow_steps=a.flow_steps, mem_off=a.mem_off)
    tids = [int(x) for x in a.tasks.split(",")]
    prev = json.load(open(a.out))["results"] if os.path.exists(a.out) else []
    res = [r for r in prev if r["tid"] in tids]; have = {r["tid"] for r in res}
    for tid in tids:
        if tid in have:
            continue
        t0 = time.time(); r = run_task_batched(runner, tid, a.n_envs, a.seed); r["sec"] = time.time() - t0
        res.append(r); print(f"t{tid:02d} sr={r['sr']:.3f} ({r['sec']:.0f}s)", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(args=vars(a), mean_sr=float(np.mean([x["sr"] for x in res])), results=res), open(a.out, "w"), indent=1)
