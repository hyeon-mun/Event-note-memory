#!/bin/bash
# GPU-7 slots for the 26-task evals cut short on 11/3 morning: v7 45k (21 tasks left) then keyframes 45k (26 tasks).
# Each group starts only when GPU 7 has > 35 GB free (gpuguard kills at < 15 GB).
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
grp pi05_long_v7 1 7 m30 64,52,62,58,65,57,51
grp pi05_long_v7 1 7 m31 66,63,59,68,54,84,56
grp pi05_long_v7 1 7 m32 67,60,69,55,50,53,61
grp pi05_long_visfull 0 0 m0 64,52,62,58,65,57,51,72,70 VMEM_FULL=1
grp pi05_long_visfull 0 0 m1 66,63,59,68,54,84,56,83 VMEM_FULL=1
grp pi05_long_visfull 0 0 m2 67,60,69,55,50,53,61,82,71 VMEM_FULL=1
wait; echo EQ21_DONE $(date)
