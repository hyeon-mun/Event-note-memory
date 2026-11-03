"""Episode-level dataset over the decoded memmaps (data_mikasa_robo/mm/tXX)."""
import csv, json, os
import numpy as np
import torch

ROOT = "/data/workspaces/mikasa/data_mikasa_robo/mm"
TASK_CSV = "/data/workspaces/mikasa/MIKASA-Robo/mikasa_robo_vla_envs.csv"
TEXT_FILE = "/data/workspaces/mikasa/data_mikasa_robo/text_emb.npz"


# standard MIKASA-Robo (non-VLA) RememberColor tasks, for comparison with the memory papers that report on them.
# No language in these envs: the -VLA instruction is used. Actions are pd_joint_delta_pos (8-d), see tools/conv_v0.py.
V0_TASKS = {91 + i: dict(env_id=f"RememberColor{k}-v0", max_len=60, mem="Object", split="short", src="PPO",
                         instr="Observe the cube's color, wait, then touch the cube of the same color.")
            for i, k in enumerate((3, 5, 9))}


def task_table():
    rows = list(csv.DictReader(open(TASK_CSV), delimiter=";"))
    t = {int(r["id"]): dict(env_id=r["Name"], max_len=int(r["Max Length"]), mem=r["Memory Type"],
                            split=r["Horizon Split"].lower(), instr=r["language_instruction"],
                            src=r["Data Source"]) for r in rows}
    t.update(V0_TASKS)
    return t


def pool_tokens(f, tokens):
    """f (...,2,17,384) -> (...,2,tokens,384)."""
    if tokens == 17:
        return f
    if tokens == 1:
        return f[..., :1, :]
    g = f[..., 1:, :].reshape(*f.shape[:-2], 2, 2, 2, 2, 384).astype(np.float32).mean((-4, -2))
    return np.concatenate([f[..., :1, :], g.reshape(*f.shape[:-2], 4, 384).astype(f.dtype)], -2)


def prop_tf(s):
    """7-d EEF state (xyz, rpy, grip) -> 8-d with roll as (sin, cos): roll sits at the +-pi wrap."""
    if isinstance(s, np.ndarray):
        return np.concatenate([s[..., :3], np.sin(s[..., 3:4]), np.cos(s[..., 3:4]), s[..., 4:]], -1).astype(np.float32)
    return torch.cat([s[..., :3], torch.sin(s[..., 3:4]), torch.cos(s[..., 3:4]), s[..., 4:]], -1)


PROP_DIM = 8

# fixed top ("base_camera") calibration of the suite's scenes: OpenCV extrinsic (world->cam) and intrinsics
CAM_E = np.array([[0.0, 1.0, 0.0, 0.0], [0.781, 0.0, -0.625, 0.141], [-0.625, 0.0, -0.781, 0.656]], np.float32)
CAM_K = np.array([[64.0, 0.0, 64.0], [0.0, 64.0, 64.0], [0.0, 0.0, 1.0]], np.float32)


def project(xyz):
    """world xyz (...,3) -> top-image pixel (u, v) (...,2)."""
    c = xyz @ CAM_E[:, :3].T + CAM_E[:, 3]
    p = c @ CAM_K.T
    return p[..., :2] / p[..., 2:3]


def hindsight_goals(state, action):
    """Per-step hindsight goal of one episode: the EE position at the next key event (gripper command change or
    episode end), projected into the top camera. state (T,7) raw EEF, action (T,7) -> (T,2) pixels."""
    T = len(state)
    g = np.sign(action[:, 6])
    ev = np.r_[np.nonzero(g[1:] != g[:-1])[0] + 1, T - 1]
    nxt = ev[np.searchsorted(ev, np.arange(T))]
    return project(state[nxt, :3]).astype(np.float32)


class TextTable:
    def __init__(self, path=TEXT_FILE):
        z = np.load(path, allow_pickle=True)
        self.emb = z["emb"].astype(np.float32)
        self.idx = {s: i for i, s in enumerate(z["text"].tolist())}

    def __call__(self, s):
        return self.emb[self.idx[s]]


class TaskStore:
    def __init__(self, tid, root=ROOT):
        d = f"{root}/t{tid:02d}"
        self.tid = tid
        self.meta = json.load(open(f"{d}/meta.json"))
        self.feat = np.load(f"{d}/feat.npy", mmap_mode="r")
        self.top = np.load(f"{d}/top.npy", mmap_mode="r")
        self.wrist = np.load(f"{d}/wrist.npy", mmap_mode="r")
        raw = np.load(f"{d}/state.npy")
        self.state = prop_tf(raw)
        self.action = np.load(f"{d}/action.npy")
        self.ep_start = np.load(f"{d}/ep_start.npy")
        self.ep_len = np.load(f"{d}/ep_len.npy")
        self.goal = np.concatenate([hindsight_goals(raw[a:a + T], self.action[a:a + T])
                                    for a, T in zip(self.ep_start, self.ep_len)])
        self.ep_task = np.load(f"{d}/ep_task.npy")


class EpisodeDataset(torch.utils.data.Dataset):
    """Returns a whole episode's memory inputs plus L sampled supervision timesteps."""

    def __init__(self, tids, episodes=None, horizon=8, n_sel=16, tokens=5, text=None, seed=0):
        self.stores = {t: TaskStore(t) for t in tids}
        self.text = text or TextTable()
        self.items = []
        for t, s in self.stores.items():
            eps = range(len(s.ep_len)) if episodes is None else episodes
            self.items += [(t, e) for e in eps]
        self.h, self.n_sel, self.tokens = horizon, n_sel, tokens
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def lengths(self):
        return np.array([self.stores[t].ep_len[e] for t, e in self.items])

    def __getitem__(self, i):
        t, e = self.items[i]
        s = self.stores[t]
        a, T = int(s.ep_start[e]), int(s.ep_len[e])
        rng = np.random.default_rng()
        sel = np.sort(rng.choice(T, size=min(self.n_sel, T), replace=False))
        # action chunk targets; pad past episode end with "hold": zero deltas, keep gripper command
        idx = sel[:, None] + np.arange(self.h)[None]
        valid = idx < T
        act = s.action[a + np.minimum(idx, T - 1)].copy()
        act[~valid, :6] = 0.0
        instr = s.meta["tasks"][int(s.ep_task[e])]
        return dict(
            feat=pool_tokens(np.asarray(s.feat[a:a + T]), self.tokens),
            proprio=s.state[a:a + T],
            sel=sel,
            top=np.asarray(s.top[a + sel]), wrist=np.asarray(s.wrist[a + sel]),
            act=act, act_valid=valid, goal=s.goal[a + sel], text=self.text(instr), T=T, tid=t, ep=e)


def collate(batch):
    B = len(batch); Tm = max(b["T"] for b in batch); L = max(len(b["sel"]) for b in batch)
    f0 = batch[0]["feat"]
    feat = np.zeros((B, Tm) + f0.shape[1:], f0.dtype); prop = np.zeros((B, Tm, PROP_DIM), np.float32)
    sel = np.zeros((B, L), np.int64); smask = np.zeros((B, L), bool)
    top = np.zeros((B, L, 128, 128, 3), np.uint8); wrist = np.zeros_like(top)
    act = np.zeros((B, L) + batch[0]["act"].shape[1:], np.float32); av = np.zeros((B, L, act.shape[2]), bool)
    goal = np.zeros((B, L, 2), np.float32)
    for i, b in enumerate(batch):
        T, l = b["T"], len(b["sel"])
        feat[i, :T] = b["feat"]; prop[i, :T] = b["proprio"]
        sel[i, :l] = b["sel"]; smask[i, :l] = True
        top[i, :l] = b["top"]; wrist[i, :l] = b["wrist"]; act[i, :l] = b["act"]; av[i, :l] = b["act_valid"]
        goal[i, :l] = b["goal"]
    out = dict(feat=feat, proprio=prop, sel=sel, smask=smask, top=top, wrist=wrist, act=act, act_valid=av, goal=goal,
               text=np.stack([b["text"] for b in batch]), T=np.array([b["T"] for b in batch]),
               tid=np.array([b["tid"] for b in batch]), ep=np.array([b["ep"] for b in batch]))
    return {k: torch.from_numpy(v) for k, v in out.items()}


class LengthBucketSampler(torch.utils.data.Sampler):
    """Batches of similar episode length (multi-task mixes 25..2160-step episodes)."""

    def __init__(self, lengths, batch_size, seed=0):
        self.lengths, self.bs = np.asarray(lengths), batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            order = np.argsort(self.lengths + self.rng.uniform(0, 0.3, len(self.lengths)) * self.lengths)
            batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
            for j in self.rng.permutation(len(batches)):
                yield batches[j].tolist()

    def __len__(self):
        return 10 ** 9
