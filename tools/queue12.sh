#!/bin/bash
# v5 text notes = v4 + measured CSS4 colour name of the removed region; joint t50/51/52. Then the t1x no-note control.
cd /data/workspaces/mikasa/research/notes_vlm
for x in "50 3" "51 5" "52 9"; do set -- $x; /data/workspaces/mikasa/envs/lr/bin/python -W ignore build_ds.py $1 /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_$2_long_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/notes/t${1}_v5 5 || exit 1; done > ../logs/build_v5.log 2>&1
cd ..; rm -rf /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v5
/data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v5 /data/workspaces/mikasa/data_mikasa_robo/notes/t50_v5 /data/workspaces/mikasa/data_mikasa_robo/notes/t51_v5 /data/workspaces/mikasa/data_mikasa_robo/notes/t52_v5 > logs/merge_t5x_v5.log 2>&1 || exit 1
while pgrep -f "[t]rain_eval.sh pi05_t1x_oracle" >/dev/null; do sleep 120; done
DEADBAND=0.02 NOTES_V=5 tools/train_eval.sh pi05_t5x_v5 /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v5 20000 50,51,52 1 020000
SIM=gpu tools/train_eval.sh pi05_t1x_base /data/workspaces/mikasa/data_mikasa_robo/notes/t1x_base 10000 13,14,15 0 010000
