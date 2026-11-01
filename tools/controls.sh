#!/bin/bash
# eval-only note-content controls on the RC-Long v6 model (t5x, 20k), one eval process at a time:
# empty notes (NOTES=0), wrong cue colour in the "Gone:" note, true cue colour in the "Gone:" note
cd /data/workspaces/mikasa/research
while pgrep -f "[m]id_eval_long.sh pi05_long_v6" >/dev/null; do sleep 120; done
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
C=../runs/lr/pi05_t5x_v6/checkpoints/020000/pretrained_model
run() {  # TAG NOTES SWAP
  O=../logs/ctl_$1
  (cd seqpol; env CUDA_VISIBLE_DEVICES=7 HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=$2 NOTES_V=6 NOTE_SWAP=$3 MEM_GB=15 NOTE_LOG=$O.notes.txt $E -W ignore evaluate.py --ckpt $C --tasks 50,51,52 --episodes 50 --sim_backend cpu --out $O.json > $O.log 2>&1)
  grep "sr=" logs/ctl_$1.log | sed "s/^/ctl $1 /" >> logs/eval_db.log
}
run v6_wrong 1 wrong
run v6_empty 0 ""
run v6_oracle 1 oracle
echo CTL_DONE
