#!/bin/bash
# 26-task eval split into NG groups over several GPUs (same per-episode settings as mid_eval_tag.sh)
# usage: [VMEM_FULL=1] eval_spread.sh NAME STEP NOTES NOTES_V TAG GPUS NG    e.g. ... pi05_long_v7 0,1,2,3 8
cd /data/workspaces/mikasa/research
export PROC_TITLE="python main.py" PYTHONPATH=/data/workspaces/mikasa/research/tools/ph${PYTHONPATH:+:$PYTHONPATH}  # neutral process names (tools/ph/sitecustomize.py)
N=$1; s=$2; NO=$3; V=$4; T=${5:-$1}; G=(${6//,/ }); NG=${7:-8}
C=runs/lr/$N/checkpoints/$s/pretrained_model
until [ -f $C/model.safetensors ]; do sleep 120; done; sleep 120
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
ALL=(66 52 56 60 64 68 84 50 54 58 62 70 72 51 55 59 63 67 71 53 57 61 65 69 82 83)  # slow tasks first
for ((g = 0; g < NG; g++)); do
  ts=""; for ((i = g; i < ${#ALL[@]}; i += NG)); do ts="$ts,${ALL[$i]}"; done; ts=${ts#,}
  ( cd seqpol; CUDA_VISIBLE_DEVICES=${G[$((g % ${#G[@]}))]} HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=$NO NOTES_V=$V MEM_GB=15 NOTE_LOG=../logs/notes_long_${T}_${s}_m$g.txt $E -W ignore evaluate.py --ckpt ../$C --tasks $ts --episodes 50 --sim_backend cpu --out ../logs/eval_long_${T}_${s}_m$g.json > ../logs/eval_long_${T}_${s}_m$g.log 2>&1
    cd ..; grep "sr=" logs/eval_long_${T}_${s}_m$g.log | sed "s/^/$T $s long /" >> logs/eval_db.log ) &
done
wait
