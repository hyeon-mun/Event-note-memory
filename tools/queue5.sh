#!/bin/bash
cd /data/workspaces/mikasa/research
/data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v2 /data/workspaces/mikasa/data_mikasa_robo/notes/t50_v2 /data/workspaces/mikasa/data_mikasa_robo/notes/t51_v2 /data/workspaces/mikasa/data_mikasa_robo/notes/t52_v2 > logs/merge_t5x_v2.log 2>&1
DEADBAND=0.02 NOTES_V=2 tools/train_eval.sh pi05_t5x_v2 /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v2 20000 50,51,52 1 020000
