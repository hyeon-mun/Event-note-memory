#!/bin/bash
# v6 notes (and their v3 caption cache) for all 26 MIKASA-Robo-VLA -Long tasks (t50-72, t82-84)
cd /data/workspaces/mikasa/research/notes_vlm
M=/data/workspaces/mikasa/data_mikasa_robo
for t in $(seq 50 72) 82 83 84; do
  name=$(python3 -c "import json;print(json.load(open('$M/mm/t$t/meta.json'))['task'])")
  [ -f $M/notes/t${t}_v6/meta/info.json ] && continue
  echo "== t$t $name $(date)"
  CUDA_VISIBLE_DEVICES=7 /data/workspaces/mikasa/envs/lr/bin/python -W ignore build_ds.py $t $M/data_lerobot/$name $M/notes/t${t}_v6 6 2>&1 | grep -v -i "warn\|kwargs\|Loading weights"
done
echo ALLDONE
