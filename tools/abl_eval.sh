#!/bin/bash
# eval-only ablations on t52 (50 eps, seeds 0-49): swap the note writer at test time. One ablation eval at a time.
cd /data/workspaces/mikasa/research/seqpol
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
run() {  # TAG CKPT_NAME NOTES_V [extra env]
  C=../runs/lr/$2/checkpoints/020000/pretrained_model; O=../logs/abl_$1
  env CUDA_VISIBLE_DEVICES=7 HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=1 NOTES_V=$3 MEM_GB=15 NOTE_LOG=$O.notes.txt $4 $E -W ignore evaluate.py --ckpt $C --tasks 52 --episodes 50 --sim_backend cpu --out $O.json > $O.log 2>&1
  grep "sr=" $O.log | sed "s/^/abl $1 /" >> ../logs/eval_db.log
}
run v4model_v3notes pi05_t5x_v4 3      # appearance notes added back at test time (distractors)
run v3model_v4notes pi05_t5x_v3 4      # distractors removed at test time only
