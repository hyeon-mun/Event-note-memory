#!/bin/bash
# BlinkCount Easy (t47, 1-3 blinks, counting memory): oracle-note feasibility, then no-note control. MP data -> CPU physics.
cd /data/workspaces/mikasa/research
until [ -f runs/lr/pi05_t1x_base/checkpoints/010000/pretrained_model/model.safetensors ]; do sleep 120; done; sleep 300
EVAL_ARGS="--note oracle" DEADBAND=0.02 tools/train_eval.sh pi05_t47_oracle /data/workspaces/mikasa/data_mikasa_robo/notes/t47_oracle 15000 47 0 010000 015000
DEADBAND=0.02 tools/train_eval.sh pi05_t47_base /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/blink_count_button_press_easy_vla_v0 15000 47 0 015000
