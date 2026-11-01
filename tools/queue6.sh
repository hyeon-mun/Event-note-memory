#!/bin/bash
# joint oracle-note upper bound for t50/51/52 (exact env colour names incl. maroon/teal)
cd /data/workspaces/mikasa/research
for x in "50 3" "51 5" "52 9"; do set -- $x; /data/workspaces/mikasa/envs/lr/bin/python -W ignore seqpol/make_note_ds.py /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_color_$2_long_vla_v0 /data/workspaces/mikasa/data_mikasa_robo/notes/t${1}_oracle9 $1; done > logs/build_oracle9.log 2>&1
/data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/merge_ds.py /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_oracle /data/workspaces/mikasa/data_mikasa_robo/notes/t50_oracle9 /data/workspaces/mikasa/data_mikasa_robo/notes/t51_oracle9 /data/workspaces/mikasa/data_mikasa_robo/notes/t52_oracle9 > logs/merge_t5x_oracle.log 2>&1
EVAL_ARGS="--note oracle" DEADBAND=0.02 tools/train_eval.sh pi05_t5x_oracle /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_oracle 20000 50,51,52 0 020000
