#!/bin/bash
# online-note evals with the action deadband: eval_db.sh run task step...
cd /data/workspaces/mikasa/research/seqpol
R=$1; T=$2; shift 2
for s in "$@"; do
  C=../runs/lr/$R/checkpoints/$s/pretrained_model
  until [ -f $C/model.safetensors ]; do sleep 60; done; sleep 60
  CUDA_VISIBLE_DEVICES=7 DEADBAND=0.001 C420=1 NOTES=1 NOTES_V=2 MEM_GB=15 NOTE_LOG=../logs/notes_db_${R}_$s.txt /data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python -W ignore evaluate.py --ckpt $C --tasks $T --episodes 50 --sim_backend cpu --out ../logs/eval_db_${R}_$s.json > ../logs/eval_db_${R}_$s.log 2>&1
  echo "$R $s $(tail -1 ../logs/eval_db_${R}_$s.log)" >> ../logs/eval_db.log
done
