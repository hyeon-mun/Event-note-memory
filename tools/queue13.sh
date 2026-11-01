#!/bin/bash
# v6 text notes = v5 with the HTML basic colour vocabulary (wide name margins) and only the first object; joint t50/51/52.
cd /data/workspaces/mikasa/research/notes_vlm
for x in "50 3" "51 5" "52 9"; do set -- $x; /data/workspaces/mikasa/envs/lr/bin/python -W ignore build_ds.py $1 /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_$2_long_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/notes/t${1}_v6 6 || exit 1; done > ../logs/build_v6.log 2>&1
cd ..; rm -rf /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v6
/data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v6 /data/workspaces/mikasa/data_mikasa_robo/notes/t50_v6 /data/workspaces/mikasa/data_mikasa_robo/notes/t51_v6 /data/workspaces/mikasa/data_mikasa_robo/notes/t52_v6 > logs/merge_t5x_v6.log 2>&1 || exit 1
while pgrep -f "[t]rain_eval.sh pi05_t1x_base" >/dev/null || pgrep -f "[l]erobot_train\|[t]_train.py" >/dev/null; do sleep 60; done
DEADBAND=0.02 NOTES_V=6 tools/train_eval.sh pi05_t5x_v6 /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v6 20000 50,51,52 1 020000
