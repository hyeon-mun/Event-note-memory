#!/bin/bash
# single 15 GB evals lost in the 11/2 06:04 gpuguard kill, one at a time next to the 4-group evals
cd /data/workspaces/mikasa/research
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 7; }
until [ $(free) -gt 35000 ]; do sleep 120; done
O=../logs/ctl_v6_oracle_r
(cd seqpol; env CUDA_VISIBLE_DEVICES=7 HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=1 NOTES_V=6 NOTE_SWAP=oracle MEM_GB=15 NOTE_LOG=$O.notes.txt $E -W ignore evaluate.py --ckpt ../runs/lr/pi05_t5x_v6/checkpoints/020000/pretrained_model --tasks 51,52 --episodes 50 --sim_backend cpu --out $O.json > $O.log 2>&1)
grep "sr=" logs/ctl_v6_oracle_r.log | sed "s/^/ctl v6_oracle /" >> logs/eval_db.log
T=52 tools/delay_eval.sh pi05_long_v6 045000 1 6
T=52 tools/delay_eval.sh pi05_long_base 045000 0 0
echo SIDEQ_DONE
