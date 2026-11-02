#!/bin/bash
# B: longer fixed memory delay (empty phase = DELAY steps) on t51, one 15 GB eval at a time
# usage: [GPU=7] [T=51] delay_eval.sh NAME STEP NOTES NOTES_V [DELAY] [MAXSTEPS] [extra env, e.g. VMEM_FULL=1]
cd /data/workspaces/mikasa/research
export PROC_TITLE="python main.py" PYTHONPATH=/data/workspaces/mikasa/research/tools/ph${PYTHONPATH:+:$PYTHONPATH}  # neutral process names (tools/ph/sitecustomize.py)
N=$1; s=$2; NO=$3; V=$4; D=${5:-1000}; M=${6:-1200}; X=${7:-}
C=runs/lr/$N/checkpoints/$s/pretrained_model
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i ${GPU:-7}; }
until [ -f $C/model.safetensors ] && [ $(free) -gt 35000 ]; do sleep 120; done
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
O=../logs/dly_${N}_${s}_d${D}${T:+_t$T}
(cd seqpol; env CUDA_VISIBLE_DEVICES=${GPU:-7} HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=$NO NOTES_V=$V MEM_GB=15 DELAY=$D MAXSTEPS=$M $X NOTE_LOG=$O.notes.txt $E -W ignore evaluate.py --ckpt ../$C --tasks ${T:-51} --episodes 50 --sim_backend cpu --out $O.json > $O.log 2>&1)
grep "sr=" logs/dly_${N}_${s}_d${D}${T:+_t$T}.log | sed "s/^/dly $N $s d$D /" >> logs/eval_db.log
