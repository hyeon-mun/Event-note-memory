#!/bin/bash
# decode -> features -> text embeddings, for all fully-downloaded tasks
PY=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
cd /data/workspaces/mikasa/research
$PY ready.py > logs/ready.txt
$PY prep_data.py --tasks $(cat logs/ready.txt) --workers 64
cd seqpol
CUDA_VISIBLE_DEVICES=6 $PY -W ignore featx.py --shard 0 --nshard 2 &
CUDA_VISIBLE_DEVICES=7 $PY -W ignore featx.py --shard 1 --nshard 2 &
wait
$PY -W ignore text_emb.py
echo PREP_DONE
