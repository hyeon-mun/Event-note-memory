#!/bin/bash
# mid-training eval of a -Long notes model on all 26 tasks (4 parallel groups, 15 GB each) while its training continues
# usage: mid_eval_long.sh NAME STEP NOTES NOTES_V
cd /data/workspaces/mikasa/research
N=$1; s=$2; NO=$3; V=$4
C=runs/lr/$N/checkpoints/$s/pretrained_model
until [ -f $C/model.safetensors ]; do sleep 120; done; sleep 120
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
ev() {  # GROUP TAG
  (cd seqpol; CUDA_VISIBLE_DEVICES=7 HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=$NO NOTES_V=$V MEM_GB=15 NOTE_LOG=../logs/notes_long_${N}_${s}_$2.txt $E -W ignore evaluate.py --ckpt ../$C --tasks $1 --episodes 50 --sim_backend cpu --out ../logs/eval_long_${N}_${s}_$2.json > ../logs/eval_long_${N}_${s}_$2.log 2>&1)
  grep "sr=" logs/eval_long_${N}_${s}_$2.log | sed "s/^/$N $s long /" >> logs/eval_db.log
}
ev 50,54,58,62,66,70,84 m0 &
ev 51,55,59,63,67,71 m1 &
ev 52,56,60,64,68,72 m2 &
ev 53,57,61,65,69,82,83 m3 &
wait
