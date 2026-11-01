#!/bin/bash
# after queue1: RememberColor 3/9-Long (generality of v2 event notes), notes vs no-note control
cd /data/workspaces/mikasa/research
while pgrep -f "[q]ueue1.sh" >/dev/null; do sleep 120; done
until grep -q "events/episode" logs/build_t50_t52.log && [ -f /data/workspaces/mikasa/data_mikasa_robo/notes/t52_v2/meta/info.json ] && ! pgrep -f "[b]uild_ds.py" >/dev/null; do sleep 120; done
tools/train_eval.sh pi05_t52_notes /data/workspaces/mikasa/data_mikasa_robo/notes/t52_v2 15000 52 1 010000 015000
tools/train_eval.sh pi05_t52_base /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_9_long_vla_v0 10000 52 0 010000
tools/train_eval.sh pi05_t50_notes /data/workspaces/mikasa/data_mikasa_robo/notes/t50_v2 15000 50 1 010000 015000
tools/train_eval.sh pi05_t50_base /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_3_long_vla_v0 10000 50 0 010000
