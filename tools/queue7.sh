#!/bin/bash
# visual notes (event crops as a memory camera), joint t50/51/52, no text notes
cd /data/workspaces/mikasa/research/notes_vlm
for x in "50 3" "51 5" "52 9"; do set -- $x; /data/workspaces/mikasa/envs/lr/bin/python -W ignore build_vis.py $1 /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_$2_long_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/notes/t${1}_vis || exit 1; done > ../logs/build_vis.log 2>&1
cd ..; /data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_vis /data/workspaces/mikasa/data_mikasa_robo/notes/t50_vis /data/workspaces/mikasa/data_mikasa_robo/notes/t51_vis /data/workspaces/mikasa/data_mikasa_robo/notes/t52_vis > logs/merge_t5x_vis.log 2>&1 || exit 1
DEADBAND=0.02 tools/train_eval.sh pi05_t5x_vis /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_vis 20000 50,51,52 0 010000 020000
