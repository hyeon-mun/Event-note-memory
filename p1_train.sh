#!/bin/bash
# Phase-1 dev comparison: 10-task multitask models
PY=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
cd /data/workspaces/mikasa/research/seqpol
DEV=0,3,12,14,15,31,35,47,79,85
R=/data/workspaces/mikasa/research/runs/p1
mkdir -p $R
run() { name=$1; gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -W ignore train.py --tasks $DEV --out $R/$name --steps 30000 --workers 10 "$@" > $R/$name.log 2>&1 &
}
run none_plain   6 --mem none  --loss plain
run stack_plain  6 --mem stack --loss plain
run gru_plain    6 --mem gru   --loss plain
run tf_plain     6 --mem tf    --loss plain
run gru_drop     7 --mem gru   --loss drop
run gru_paired   7 --mem gru   --loss paired
run tf_drop      7 --mem tf    --loss drop
run tf_paired    7 --mem tf    --loss paired
wait
