#!/bin/bash
# CPU-backend eval of the VLM-note policy with online notes, at 5k and 10k steps
cd /data/workspaces/mikasa/research/seqpol
for s in ${STEPS:-005000 010000}; do
  C=../runs/lr/pi05_t51_notes/checkpoints/$s/pretrained_model
  until [ -f $C/model.safetensors ]; do sleep 60; done; sleep 60
  CUDA_VISIBLE_DEVICES=7 NOTES=1 MEM_GB=15 NOTE_LOG=../logs/notes_eval_$s.txt /data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python -W ignore evaluate.py --ckpt $C --tasks 51 --episodes 50 --sim_backend cpu --out ../logs/eval_pi05_t51_notes_$s.json > ../logs/eval_pi05_t51_notes_$s.log 2>&1
done
