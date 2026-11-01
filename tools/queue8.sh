#!/bin/bash
# v4 text notes: v3 captions, only what disappeared ("Gone: X."), joint t50/51/52
cd /data/workspaces/mikasa/research/notes_vlm
for x in "50 3" "51 5" "52 9"; do set -- $x; /data/workspaces/mikasa/envs/lr/bin/python -W ignore build_ds.py $1 /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_$2_long_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/notes/t${1}_v4 4 || exit 1; done > ../logs/build_v4.log 2>&1
cd ..; /data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v4 /data/workspaces/mikasa/data_mikasa_robo/notes/t50_v4 /data/workspaces/mikasa/data_mikasa_robo/notes/t51_v4 /data/workspaces/mikasa/data_mikasa_robo/notes/t52_v4 > logs/merge_t5x_v4.log 2>&1 || exit 1
DEADBAND=0.02 NOTES_V=4 tools/train_eval.sh pi05_t5x_v4 /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v4 20000 50,51,52 1 020000
