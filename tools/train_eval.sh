#!/bin/bash
# one GPU-7 training (40 GB cap) + CPU-physics evals of its checkpoints (deadband 0.001, C420, v2 online notes if NOTES=1)
# usage: train_eval.sh NAME DATA_ROOT STEPS TASK NOTES(0/1) EVAL_STEP...
cd /data/workspaces/mikasa/research
N=$1; R=$2; S=$3; T=$4; NO=$5; shift 5
while pgrep -f "[l]erobot-train" -u $USER >/dev/null; do sleep 120; done
export HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets
P=/data/workspaces/mikasa/envs/lr/bin/python; E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/$N --dataset.root=$R --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=$S --log_freq=100 --save_freq=2500 --num_workers=8 --output_dir=runs/lr/$N --wandb.enable=false > runs/lr/$N.log 2>&1 &
for s in "$@"; do
  C=runs/lr/$N/checkpoints/$s/pretrained_model
  until [ -f $C/model.safetensors ]; do sleep 60; done; sleep 60
  (cd seqpol; CUDA_VISIBLE_DEVICES=7 DEADBAND=${DEADBAND:-0.001} C420=1 NOTES=$NO NOTES_V=${NOTES_V:-2} MEM_GB=15 NOTE_LOG=../logs/notes_db_${N}_$s.txt $E -W ignore evaluate.py --ckpt ../$C --tasks $T --episodes 50 --sim_backend ${SIM:-cpu} $EVAL_ARGS --out ../logs/eval_db_${N}_$s.json > ../logs/eval_db_${N}_$s.log 2>&1)
  grep "sr=" logs/eval_db_${N}_$s.log | sed "s/^/$N $s /" >> logs/eval_db.log
done
wait
