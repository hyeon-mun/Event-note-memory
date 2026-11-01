#!/bin/bash
# train on the 26 -Long tasks with the same 90k-step schedule as pi05_long_{base,v6} and stop at 45k
# (same learning-rate schedule as their 45k checkpoints). usage: train45.sh NAME DATA_ROOT
cd /data/workspaces/mikasa/research
N=$1; R=$2
while pgrep -f "[l]erobot-train" -u $USER >/dev/null; do sleep 120; done
until [ -f $R/meta/info.json ]; do sleep 120; done
export HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets
P=/data/workspaces/mikasa/envs/lr/bin/python
C=runs/lr/$N/checkpoints/045000/pretrained_model/model.safetensors
[ -f $C ] && exit 0
CUDA_VISIBLE_DEVICES=7 setsid $P -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/$N --dataset.root=$R --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=90000 --log_freq=100 --save_freq=5000 --num_workers=8 --output_dir=runs/lr/$N --wandb.enable=false > runs/lr/$N.log 2>&1 &
TP=$!
until [ -f $C ] || ! kill -0 $TP 2>/dev/null; do sleep 60; done
sleep 120                                             # let the checkpoint finish writing
kill -- -$TP 2>/dev/null; sleep 30; kill -9 -- -$TP 2>/dev/null
echo "$N stopped at 45k $(date)"
