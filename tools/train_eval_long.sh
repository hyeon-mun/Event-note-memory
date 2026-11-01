#!/bin/bash
# one GPU-7 training (40 GB cap) on the 26 -Long tasks, then the final checkpoint evaluated on every task,
# 3 evaluate.py processes in parallel (15 GB each). CPU physics (MP demos), deadband 0.02, C420.
# usage: train_eval_long.sh NAME DATA_ROOT STEPS NOTES(0/1)   (NOTES_V from the environment)
cd /data/workspaces/mikasa/research
N=$1; R=$2; S=$3; NO=$4
while pgrep -f "[l]erobot-train" -u $USER >/dev/null; do sleep 120; done
export HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets
P=/data/workspaces/mikasa/envs/lr/bin/python; E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
s=$(printf "%06d" $S)
if [ ! -f runs/lr/$N/checkpoints/$s/pretrained_model/model.safetensors ]; then
  CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/$N --dataset.root=$R --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=$S --log_freq=100 --save_freq=5000 --num_workers=8 --output_dir=runs/lr/$N --wandb.enable=false > runs/lr/$N.log 2>&1
fi
C=runs/lr/$N/checkpoints/$s/pretrained_model
ev() {  # GROUP TAG
  (cd seqpol; CUDA_VISIBLE_DEVICES=7 DEADBAND=0.02 C420=1 NOTES=$NO NOTES_V=${NOTES_V:-2} MEM_GB=15 NOTE_LOG=../logs/notes_long_${N}_$2.txt $E -W ignore evaluate.py --ckpt ../$C --tasks $1 --episodes ${EPS:-50} --sim_backend cpu --out ../logs/eval_long_${N}_$2.json > ../logs/eval_long_${N}_$2.log 2>&1)
  grep "sr=" logs/eval_long_${N}_$2.log | sed "s/^/$N $s long /" >> logs/eval_db.log
}
ev 50,53,56,59,62,65,68,71,82 g0 &
ev 51,54,57,60,63,66,69,72,83 g1 &
ev 52,55,58,61,64,67,70,84 g2 &
wait
