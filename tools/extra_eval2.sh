#!/bin/bash
# extra t52 episodes (seeds 50..149) for v4
cd /data/workspaces/mikasa/research/seqpol
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
C=../runs/lr/pi05_t5x_v4/checkpoints/020000/pretrained_model; O=../logs/eval_x52_pi05_t5x_v4
CUDA_VISIBLE_DEVICES=7 HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=1 NOTES_V=4 MEM_GB=15 NOTE_LOG=$O.notes.txt $E -W ignore evaluate.py --ckpt $C --tasks 52 --episodes 100 --ep_lo 50 --sim_backend cpu --out $O.json > $O.log 2>&1
grep "sr=" $O.log | sed "s/^/pi05_t5x_v4 020000 x52eps50-149 /" >> ../logs/eval_db.log
