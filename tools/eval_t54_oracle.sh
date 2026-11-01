#!/bin/bash
cd /data/workspaces/mikasa/research/seqpol
for s in 010000 015000; do
  until [ -f ../runs/lr/pi05_t54_oracle/checkpoints/$s/pretrained_model/model.safetensors ]; do sleep 60; done; sleep 60
  CUDA_VISIBLE_DEVICES=7 MEM_GB=15 /data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python -W ignore evaluate.py --ckpt ../runs/lr/pi05_t54_oracle/checkpoints/$s/pretrained_model --tasks 54 --episodes 50 --sim_backend cpu --note oracle --out ../logs/eval_pi05_t54_oracle_$s.json > ../logs/eval_pi05_t54_oracle_$s.log 2>&1
done
