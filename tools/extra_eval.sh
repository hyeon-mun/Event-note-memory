#!/bin/bash
# extra t52 episodes (seeds 50..149) for the joint models; never more than one of my evals at a time
cd /data/workspaces/mikasa/research/seqpol
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
run() {  # NAME NOTES NOTES_V EXTRA
  while pgrep -f "[e]valuate.py --ckpt" -u $USER >/dev/null; do sleep 60; done
  C=../runs/lr/$1/checkpoints/020000/pretrained_model; O=../logs/eval_x52_$1
  CUDA_VISIBLE_DEVICES=7 DEADBAND=0.02 C420=1 NOTES=$2 NOTES_V=$3 MEM_GB=15 NOTE_LOG=$O.notes.txt $E -W ignore evaluate.py --ckpt $C --tasks 52 --episodes 100 --ep_lo 50 --sim_backend cpu $4 --out $O.json > $O.log 2>&1
  grep "sr=" $O.log | sed "s/^/$1 020000 x52eps50-149 /" >> ../logs/eval_db.log
}
run pi05_t5x_oracle 0 2 "--note oracle"
run pi05_t5x_v3 1 3
run pi05_t5x_v2 1 2
run pi05_t5x_base 0 2
