#!/bin/bash
# re-evaluate with yuv420 chroma degradation of the observations (matches the training videos)
cd /data/workspaces/mikasa/research/seqpol
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
ev() {  # run step task notes note_arg
  C=../runs/lr/$1/checkpoints/$2/pretrained_model
  until [ -f $C/model.safetensors ]; do sleep 60; done
  CUDA_VISIBLE_DEVICES=7 C420=1 NOTES=$4 MEM_GB=15 NOTE_LOG=../logs/notes_c420_$1_$2.txt $E -W ignore evaluate.py --ckpt $C --tasks $3 --episodes 50 --sim_backend cpu $5 --out ../logs/eval_c420_$1_$2.json > ../logs/eval_c420_$1_$2.log 2>&1
  echo "$1 $2 $(tail -1 ../logs/eval_c420_$1_$2.log)"
}
ev pi05_t51_notes 015000 51 1
ev pi05_t51_base 010000 51 0
ev pi05_t54_notes 020000 54 1
ev pi05_t54_oracle 010000 54 0 "--note oracle"
ev pi05_t54_oracle 015000 54 0 "--note oracle"
ev pi05_t54_base 010000 54 0
ev pi05_t51_oracle 010000 51 0 "--note oracle"
