#!/bin/bash
# standard (short) RememberColor 3/5/9 = t13/14/15, joint: oracle-note feasibility first, then no-note control.
# PPO-collected demos -> canonical GPU physics for eval.
cd /data/workspaces/mikasa/research
for x in "13 3" "14 5" "15 9"; do set -- $x; /data/workspaces/mikasa/envs/lr/bin/python -W ignore seqpol/make_note_ds.py /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_$2_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/notes/t${1}_oracle9 $1 || exit 1; done > logs/build_t1x.log 2>&1
/data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t1x_oracle /data/workspaces/mikasa/data_mikasa_robo/notes/t13_oracle9 /data/workspaces/mikasa/data_mikasa_robo/notes/t14_oracle9 /data/workspaces/mikasa/data_mikasa_robo/notes/t15_oracle9 >> logs/build_t1x.log 2>&1 || exit 1
/data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t1x_base /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_3_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_5_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_9_vla_v0 >> logs/build_t1x.log 2>&1 || exit 1
until [ -f runs/lr/pi05_t5x_v4/checkpoints/020000/pretrained_model/model.safetensors ]; do sleep 120; done; sleep 300
SIM=gpu EVAL_ARGS="--note oracle" tools/train_eval.sh pi05_t1x_oracle /data/workspaces/mikasa/data_mikasa_robo/notes/t1x_oracle 10000 13,14,15 0 005000 010000
SIM=gpu tools/train_eval.sh pi05_t1x_base /data/workspaces/mikasa/data_mikasa_robo/notes/t1x_base 10000 13,14,15 0 010000
