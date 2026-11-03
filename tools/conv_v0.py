"""Standard MIKASA-Robo RememberColor{3,5,9}-v0 demos (HF avanturist/mikasa-robo, npz, PPO expert) -> LeRobot v3
in the same layout as data_lerobot/remember_color_*_vla_v0, plus source_rlds_metadata.json so prep_data.py can
decode it into mm/tXX.
npz per episode: rgb (60,128,128,6) [base|hand], joints (60,25) = tcp_pose(7: xyz, wxyz) + qpos(9) + qvel(9),
action (60,8) pd_joint_delta_pos (7 arm joints + gripper; NOT the end-effector delta of the -VLA tasks),
success (60,). Episodes always run 60 steps; like the -VLA demos we cut each one at its first success step.
The first N successful episodes are kept (the memory papers train on 250 per task).
usage: conv_v0.py ENV_ID OUT_DIR [N]
"""
import io, json, os, shutil, sys, zipfile
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

env_id, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 250
INSTR = "Observe the cube's color, wait, then touch the cube of the same color."   # wording of RememberColor*-VLA-v0
z = zipfile.ZipFile(f"/data/workspaces/mikasa/data_mikasa_robo/raw_v0/{env_id}.zip")


def rpy(q):
    """wxyz -> roll, pitch, yaw (same formula as ConvertJointsToEEFXyzRpyGripperWrapper)."""
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    w, x, y, zz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.arctan2(2 * (w * x + y * zz), 1 - 2 * (x * x + y * y))
    p = np.arcsin(np.clip(2 * (w * y - zz * x), -1, 1))
    yw = np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz))
    return np.stack([r, p, yw], -1)


img = {"dtype": "video", "shape": [128, 128, 3], "names": ["height", "width", "channel"]}
feats = {"observation.images.top": img, "observation.images.wrist": dict(img),
         "observation.state": {"dtype": "float32", "shape": [7], "names": None},
         "action": {"dtype": "float32", "shape": [8], "names": None}}
shutil.rmtree(out, ignore_errors=True)
ds = LeRobotDataset.create(repo_id=f"local/{os.path.basename(out)}", fps=10, features=feats, root=out, robot_type="mikasa_robo")
lens, used, skipped, i = [], [], 0, 0
while len(lens) < N:
    d = np.load(io.BytesIO(z.read(f"{env_id}/train_data_{i}.npz")))
    s = np.nonzero(d["success"])[0]
    if len(s) == 0:
        skipped += 1; i += 1; continue
    T = int(s[0]) + 1
    j = d["joints"][:T]
    st = np.concatenate([j[:, :3], rpy(j[:, 3:7]), j[:, 14:16].sum(1, keepdims=True)], 1).astype(np.float32)
    for t in range(T):
        ds.add_frame({"observation.images.top": d["rgb"][t, ..., :3], "observation.images.wrist": d["rgb"][t, ..., 3:],
                      "observation.state": st[t], "action": d["action"][t].astype(np.float32), "task": INSTR})
    ds.save_episode()
    lens.append(T); used.append(i); i += 1
ds.finalize()
json.dump({"env_id": env_id, "num_episodes": len(lens), "episode_lengths": lens, "source_files": used,
           "skipped_without_success": skipped, "action_space": "pd_joint_delta_pos (8)", "cut": "first success"},
          open(f"{out}/source_rlds_metadata.json", "w"), indent=1)
print(env_id, "episodes", len(lens), "frames", sum(lens), "mean len", np.mean(lens), "skipped", skipped, flush=True)
