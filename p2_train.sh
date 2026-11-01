#!/bin/bash
# Phase-2: hard-memory dev set (medium horizon), slot memories vs baselines
PY=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
cd /data/workspaces/mikasa/research/seqpol
DEVB=22,25,28,43,49,68,73,86
R=/data/workspaces/mikasa/research/runs/p2
mkdir -p $R
run() { name=$1; gpu=$2; shift 2
  [ -f $R/$name/ckpt_30000.pt ] && return
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -W ignore train.py --tasks $DEVB --out $R/$name --steps 30000 --bs 16 --workers 10 "$@" > $R/$name.log 2>&1 &
}
run none_paired  ${G1:-6} --mem none  --loss plain
run gru_paired   ${G1:-6} --mem gru   --loss paired
run tf_paired    ${G1:-6} --mem tf    --loss paired
run event_paired ${G2:-7} --mem event --loss paired
run todo_paired  ${G2:-7} --mem todo  --loss paired
run todo_plain   ${G2:-7} --mem todo  --loss plain
wait
