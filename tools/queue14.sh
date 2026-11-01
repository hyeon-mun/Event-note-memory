#!/bin/bash
# no-note pi0.5 on all 26 -Long tasks (after the t5x_v6 training)
cd /data/workspaces/mikasa/research
until [ -f runs/lr/pi05_t5x_v6/checkpoints/020000/pretrained_model/model.safetensors ]; do sleep 120; done; sleep 120
until grep -q "^done" logs/merge_long_base.log; do sleep 60; done
tools/train_eval_long.sh pi05_long_base /data/workspaces/mikasa/data_mikasa_robo/notes/long_base 90000 0
