"""Print dataset task dirs whose files are fully downloaded."""
import os, collections
from huggingface_hub import HfApi
root = "/data/workspaces/mikasa/data_mikasa_robo/data_lerobot"
files = HfApi().list_repo_tree("mikasa-robo/mikasa-robo-vla-lerobot", repo_type="dataset", recursive=True)
need = collections.defaultdict(list)
for f in files:
    if hasattr(f, "size") and "/" in f.path:
        need[f.path.split("/")[0]].append((f.path, f.size))
for t, fl in sorted(need.items()):
    if all(os.path.exists(f"{root}/{p}") and os.path.getsize(f"{root}/{p}") == s for p, s in fl):
        print(t)
