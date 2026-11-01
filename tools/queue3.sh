#!/bin/bash
# v3 notes (zoomed change notes, SmolVLM2-2.2B): t52 and t51 retrain, then t50 v3 + control; one training at a time
cd /data/workspaces/mikasa/research
while pgrep -f "[t]rain_eval.sh pi05_t52_base" >/dev/null; do sleep 120; done
until [ -f /data/workspaces/mikasa/data_mikasa_robo/notes/t51_v3/meta/info.json ]; do sleep 120; done
DEADBAND=0.02 NOTES_V=3 tools/train_eval.sh pi05_t52_v3 /data/workspaces/mikasa/data_mikasa_robo/notes/t52_v3 15000 52 1 010000 015000
DEADBAND=0.02 NOTES_V=3 tools/train_eval.sh pi05_t51_v3 /data/workspaces/mikasa/data_mikasa_robo/notes/t51_v3 15000 51 1 010000 015000
until [ -f /data/workspaces/mikasa/data_mikasa_robo/notes/t50_v3/meta/info.json ]; do sleep 120; done
DEADBAND=0.02 NOTES_V=3 tools/train_eval.sh pi05_t50_v3 /data/workspaces/mikasa/data_mikasa_robo/notes/t50_v3 15000 50 1 010000 015000
tools/train_eval.sh pi05_t50_base /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_3_long_vla_v0 10000 50 0 010000
