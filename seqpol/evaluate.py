"""Closed-loop evaluation following the suite's canonical protocol.

Same env construction (make_benchmark_env), seeds (start_seed + i, num_envs=1), and success_once latch
as the official runner; the only difference is an explicit policy.reset() per episode (the official
runner has no reset hook, which a stateful policy needs). Memory is updated on EVERY env step; a new
action chunk is sampled every --exec steps.
"""
import argparse, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

sys.path.insert(0, "/data/workspaces/mikasa/MIKASA-Robo")
import mikasa_robo_suite.vla.memory_envs  # noqa: F401  (registers envs)
from mikasa_robo_suite.vla.benchmarking import BenchmarkConfig, make_benchmark_env

from data import TextTable, pool_tokens, prop_tf, task_table
from featx import encode, load_dino
from model import SeqPolicy


class Text:
    def __init__(self):
        self.tab = TextTable(); self.clip = None

    def __call__(self, s):
        if s in self.tab.idx:
            return self.tab(s)
        if self.clip is None:
            from transformers import CLIPTokenizer, CLIPTextModelWithProjection
            self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.clip = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").eval()
        with torch.no_grad():
            e = self.clip(**self.tok([s], return_tensors="pt")).text_embeds
        return torch.nn.functional.normalize(e, dim=-1)[0].numpy()


class PolicyRunner:
    def __init__(self, ckpt, dev="cuda", guidance=1.0, exec_steps=4, flow_steps=10, mem_off=False):
        ck = torch.load(ckpt, map_location="cpu")
        a = ck["args"]
        self.m = SeqPolicy(mem=a["mem"], mem_tokens=a["tokens"], horizon=a["horizon"],
                           use_proprio_mem=not a["no_proprio_mem"],
                           quant=tuple(int(x) for x in a["quant"].split(",")) if a.get("quant") else None,
                           mem_src=a.get("mem_src", "dino"), stack=a.get("stack", 8), find=a.get("find", False), vis=a.get("vis", "cnn"),
                           goal=a.get("goal", False), goal_match=a.get("goal_match", False),
                           mem_bottleneck=tuple(int(x) for x in a["mem_bottleneck"].split(",")) if a.get("mem_bottleneck") else None,
                           goal_rf=a.get("goal_rf", 11)).to(dev)
        self.m.register_buffer("p_mean", torch.zeros(8, device=dev)); self.m.register_buffer("p_std", torch.ones(8, device=dev))
        self.m.load_state_dict(ck["ema"], strict=False); self.m.eval()   # old ckpts: no a_scale (=1)
        self.a, self.dev = a, dev
        self.dino = load_dino(dev)
        self.text = Text()
        self.guidance, self.exec_steps, self.flow_steps, self.mem_off = guidance, exec_steps, flow_steps, mem_off
        if a.get("mem_src") == "oracle":
            from cue_labels import cluster, color_clusters, cue_colors
            t0 = ck["tids"][0]; self.centers = cluster(cue_colors(t0), {13: 3, 14: 5, 15: 9, 50: 3, 51: 5, 52: 9}[t0])[1]

    def reset(self, instruction, tid=None):
        self.tid = tid
        self.txt = torch.tensor(self.text(instruction), device=self.dev)[None]
        self.hist, self.h, self.queue, self.t = [], None, [], 0

    @torch.no_grad()
    def act(self, obs):
        rgb = obs["rgb"].to(self.dev)                                    # (1,128,128,6) uint8
        top, wrist = rgb[..., :3], rgb[..., 3:]
        prop = (prop_tf(obs["proprio"].to(self.dev).float()) - self.m.p_mean) / self.m.p_std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            react = self.m.reactive(top, wrist, prop, self.txt)
            if self.m.mem_src == "oracle":           # cue label read from the first frame (diagnostic only)
                if self.t == 0:
                    px = top[0, 64:88, 52:76].reshape(-1, 3).float().cpu().numpy()
                    c = px[np.argsort(-(px.max(1) - px.min(1)))[:4]].mean(0)
                    self.lab = int(((self.centers - c) ** 2).sum(1).argmin())
                x = torch.nn.functional.one_hot(torch.tensor([self.lab], device=self.dev), 16).float()[:, None]
            elif self.m.mem_src == "cnn":
                x = react[:, None]
            else:
                f = torch.stack([encode(self.dino, top), encode(self.dino, wrist)], 1)   # (1,2,17,384)
                f = torch.from_numpy(pool_tokens(f.float().cpu().numpy()[:, None], self.a["tokens"])).to(self.dev)
                x = self.m.mem_inputs(f, prop[:, None], self.txt)                        # (1,1,Din)
            kind = self.a["mem"]
            if kind == "gru":
                mem, self.h = self.m.memory(x, self.h)
                mem = mem[:, -1]
            else:
                self.hist.append(x)
                if kind == "stack":
                    self.hist = self.hist[-self.m.memory.stack:]
                mem = self.m.memory(torch.cat(self.hist, 1))[0][:, -1]
            if not self.queue:
                if self.mem_off:
                    mem = self.m.null_mem.expand_as(mem)
                if self.m.goal is not None:
                    self.m._goal_uv = self.m.goal.uv(self.m.goal(top, wrist, mem.float()))
                chunk = self.m.sample(react.float(), mem.float(), self.flow_steps, self.guidance, tid=self.tid)[0]
                self.queue = list(chunk[: self.exec_steps])
        self.t += 1
        return self.queue.pop(0).float()


class RemoteRunner:
    """Policy served from the LeRobot env (lrbridge/server.py). note='oracle' appends the cue colour read from
    the first frame to the instruction (diagnostic upper bound for language-note memory)."""

    def __init__(self, ckpt_dir, note=None):
        import random, subprocess, time
        from multiprocessing.connection import Client
        self.port = random.randint(20000, 60000)
        self.proc = subprocess.Popen(["/data/workspaces/mikasa/envs/lr/bin/python", "-W", "ignore",
                                      "/data/workspaces/mikasa/research/lrbridge/server.py", ckpt_dir, str(self.port)],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        while "READY" not in self.proc.stdout.readline():
            if self.proc.poll() is not None:
                raise RuntimeError("policy server died")
        self.conn = Client(("127.0.0.1", self.port), authkey=b"seqpol")
        self.note = note

    def reset(self, instruction, tid=None, swap=""):
        self.instr, self.tid, self.t = instruction, tid, 0
        self.conn.send(("reset", swap or "")); self.conn.recv()

    def act(self, obs):
        rgb = obs["rgb"][0].cpu().numpy(); prop = obs["proprio"][0].float().cpu().numpy()
        if self.t == 0 and self.note == "oracle" and self.tid == 54:
            # demo shape clusters (make_note_ds.py) matched to env shapes by their pixel footprint after yuv420:
            # cube->E, sphere->D, cylinder->A, cross->C, torus->B (torus/cross are rendered larger online than in the demos)
            j = {0: 4, 1: 3, 2: 0, 3: 2, 4: 1}[int(np.asarray(self.oracle).reshape(-1)[0])]
            self.instr = f"{self.instr} The cue shape was {chr(65 + j)}."
        elif self.t == 0 and self.note == "oracle" and self.tid in (47, 48, 49):   # true blink count
            self.instr = f"{self.instr} The blue lamp blinked {int(np.asarray(self.oracle).reshape(-1)[0])} times."
        elif self.t == 0 and self.note == "oracle" and self.tid in (28, 29, 30, 65, 66, 67):
            from make_seq_note_ds import seq_note      # oracle_info is the true colour sequence
            self.instr = self.instr + seq_note(np.asarray(self.oracle).reshape(-1))
        elif self.t == 0 and self.note == "oracle":
            from cue_labels import cluster, color_clusters, cue_colors
            from make_note_names import color_names
            k = {13: 3, 14: 5, 15: 9, 50: 3, 51: 5, 52: 9}[self.tid]
            C = color_clusters(self.tid, k)[1]
            px = rgb[64:88, 52:76, :3].reshape(-1, 3).astype(np.float32)
            c = px[np.argsort(-(px.max(1) - px.min(1)))[:4]].mean(0)
            self.instr = f"{self.instr} The cue color was {color_names(C)[int(((C - c) ** 2).sum(1).argmin())]}."
        self.t += 1
        self.conn.send(("act", {"rgb": (rgb.tobytes(), rgb.shape, str(rgb.dtype)),
                                "proprio": (prop.tobytes(), prop.shape, str(prop.dtype))}, self.instr))
        b, shape, dt = self.conn.recv()
        return torch.from_numpy(np.frombuffer(b, dtype=dt).reshape(shape).copy())


SWAP = os.environ.get("NOTE_SWAP", "")                   # "oracle" / "wrong": replace the colour of online "Gone:" notes
HTML_NAMES = ["red", "lime", "blue", "yellow", "fuchsia", "aqua", "maroon", "orange", "teal"]   # env colour index -> v6 vocabulary


def run_task(runner, tid, n_eps, start_seed, log=None, ep_lo=0, sim_backend="gpu", video_dir=None):
    tt = task_table()[tid]
    if sim_backend == "auto":     # physics consistent with how the demos were collected (MP: CPU, PPO: GPU)
        sim_backend = "cpu" if tt["src"] == "MP" else "gpu"
    cfg = BenchmarkConfig(n_episodes=n_eps, start_seed=start_seed, sim_backend=sim_backend)
    env = make_benchmark_env(tt["env_id"], cfg)
    succ, lens, prog = [], [], []
    for i in range(ep_lo, ep_lo + n_eps):
        obs, info = env.reset(seed=start_seed + i)
        instr = info.get("language_instruction", None)
        if instr is None:
            instr = tt["instr"]
        if isinstance(instr, (list, tuple)):
            instr = instr[0]
        runner.oracle = info.get("oracle_info")
        swap = ""
        u = env.unwrapped
        if SWAP and hasattr(u, "cubes") and hasattr(u, "true_color_indices"):   # note-content controls (NOTE_SWAP)
            true = int(u.true_color_indices.reshape(-1)[0])
            k = true if SWAP == "oracle" else np.random.RandomState(i).choice([c for c in sorted(u.cubes) if c != true])
            swap = HTML_NAMES[int(k)]
        if swap: runner.reset(instr, tid, swap)
        else: runner.reset(instr, tid)
        ok, n, frames = False, 0, []
        for _ in range(int(env.max_episode_steps)):
            if video_dir:
                frames.append(np.concatenate([obs["rgb"][0, ..., :3].cpu().numpy(), obs["rgb"][0, ..., 3:].cpu().numpy()], 1))
            a = runner.act(obs).to(env.unwrapped.device)[None]
            obs, r, term, trunc, info = env.step(a)
            n += 1
            ok = ok or bool(torch.as_tensor(info["success"]).reshape(-1)[0])
            u = env.unwrapped
            if hasattr(u, "cubes") and hasattr(u, "true_color_indices") and not hasattr(u, "touched_cubes_idx"):
                tcp = u.agent.tcp.pose.p[0]                  # single-choice tasks: closest approach to each cube
                d = [float(torch.linalg.norm(u.cubes[k].pose.p[0] - tcp)) for k in sorted(u.cubes)]
                mind = d if n == 1 else [min(a, b) for a, b in zip(mind, d)]
            if bool(torch.as_tensor(term).reshape(-1)[0]) or bool(torch.as_tensor(trunc).reshape(-1)[0]):
                break
        succ.append(ok); lens.append(n)
        u = env.unwrapped
        if hasattr(u, "cubes") and hasattr(u, "true_color_indices") and not hasattr(u, "touched_cubes_idx"):
            prog.append(dict(mind=mind, true=int(u.true_color_indices.reshape(-1)[0])))
        if hasattr(u, "touched_cubes_idx") and hasattr(u, "true_color_indices"):   # sequence tasks: partial progress
            prog.append(dict(touched=u.touched_cubes_idx[0].tolist(), true=u.true_color_indices[0].tolist()))
        if video_dir:
            import cv2
            os.makedirs(video_dir, exist_ok=True)
            vw = cv2.VideoWriter(f"{video_dir}/ep{i:03d}_{'ok' if ok else 'fail'}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (512, 256))
            for f in frames:
                vw.write(cv2.resize(f[:, :, ::-1], (512, 256), interpolation=cv2.INTER_NEAREST))
            vw.release()
        if log: print(f"  t{tid:02d} ep{i} ok={ok} n={n}", file=log, flush=True)
    env.close()
    return dict(tid=tid, mem_type=tt["mem"], split=tt["split"], sr=float(np.mean(succ)), successes=succ, lens=lens,
                seeds=[start_seed + i for i in range(ep_lo, ep_lo + n_eps)], progress=prog)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes", type=int, default=50); ap.add_argument("--start_seed", type=int, default=4242424242)
    ap.add_argument("--guidance", type=float, default=1.0); ap.add_argument("--exec", type=int, default=4)
    ap.add_argument("--flow_steps", type=int, default=10); ap.add_argument("--mem_off", action="store_true")
    ap.add_argument("--ep_lo", type=int, default=0)
    ap.add_argument("--sim_backend", default="gpu")
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--note", default=None, help="remote policies: 'oracle' appends the true cue to the instruction")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    if os.path.isdir(a.ckpt):     # LeRobot checkpoint dir (.../checkpoints/NNN/pretrained_model)
        runner = RemoteRunner(a.ckpt, note=a.note)
    else:
        runner = PolicyRunner(a.ckpt, guidance=a.guidance, exec_steps=a.exec, flow_steps=a.flow_steps, mem_off=a.mem_off)
    res = []
    for tid in [int(x) for x in a.tasks.split(",")]:
        t0 = time.time()
        r = run_task(runner, tid, a.episodes, a.start_seed, ep_lo=a.ep_lo, sim_backend=a.sim_backend, video_dir=a.video_dir)
        r["sec"] = time.time() - t0; res.append(r)
        print(f"t{tid:02d} sr={r['sr']:.3f} ({r['sec']:.0f}s)", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(args=vars(a), results=res), open(a.out, "w"), indent=1)
