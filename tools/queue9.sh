#!/bin/bash
# keyframe ablation: same events/grid as t5x_vis but each tile is the whole before-frame (no zoomed crop)
cd /data/workspaces/mikasa/research/notes_vlm
for x in "50 3" "51 5" "52 9"; do set -- $x; /data/workspaces/mikasa/envs/lr/bin/python -W ignore build_vis.py $1 /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_$2_long_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/notes/t${1}_visfull full || exit 1; done > ../logs/build_visfull.log 2>&1
cd ..; /data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_visfull /data/workspaces/mikasa/data_mikasa_robo/notes/t50_visfull /data/workspaces/mikasa/data_mikasa_robo/notes/t51_visfull /data/workspaces/mikasa/data_mikasa_robo/notes/t52_visfull > logs/merge_t5x_visfull.log 2>&1 || exit 1
until [ -f runs/lr/pi05_t5x_v4/checkpoints/020000/pretrained_model/model.safetensors ]; do sleep 120; done; sleep 300
VMEM_FULL=1 DEADBAND=0.02 tools/train_eval.sh pi05_t5x_visfull /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_visfull 20000 50,51,52 0 020000
