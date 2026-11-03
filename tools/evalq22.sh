#!/bin/bash
# GPU-7 slot for the three v7 tasks cut at 09:55 (t72, t82, t83)
cd /data/workspaces/mikasa/research
export PROC_TITLE="python main.py" PYTHONPATH=/data/workspaces/mikasa/research/tools/ph${PYTHONPATH:+:$PYTHONPATH}
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 7; }
grp() {  # NAME NOTES NOTES_V TAG TASKS [extra env]
  local N=$1 NO=$2 V=$3 g=$4 ts=$5 X=${6:-}; local C=runs/lr/$N/checkpoints/045000/pretrained_model
  until [ $(free) -gt 35000 ]; do sleep 120; done
  ( cd seqpol; env CUDA_VISIBLE_DEVICES=7 HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=$NO NOTES_V=$V MEM_GB=15 $X NOTE_LOG=../logs/notes_long_${N}_045000_$g.txt $E -W ignore evaluate.py --ckpt ../$C --tasks $ts --episodes 50 --sim_backend cpu --out ../logs/eval_long_${N}_045000_$g.json > ../logs/eval_long_${N}_045000_$g.log 2>&1
    cd ..; grep "sr=" logs/eval_long_${N}_045000_$g.log | sed "s/^/$N 045000 long /" >> logs/eval_db.log ) &
  sleep 300   # let the new server reach its memory before the next check
}
grp pi05_long_v7 1 7 m33 72,82,83
wait; echo EQ22_DONE $(date)
