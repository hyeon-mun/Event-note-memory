#!/bin/bash
# t65 (ChainOfColors3-Long): v2-note policy 20k + online-note evals (C420), then the no-note control 10k + eval
cd /data/workspaces/mikasa/research
P=/data/workspaces/mikasa/envs/lr/bin/python; E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
train() {  # name dataset_root steps
  CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/$1 --dataset.root=$2 --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=$3 --log_freq=100 --save_freq=2500 --num_workers=8 --output_dir=runs/lr/$1 --wandb.enable=false > runs/lr/$1.log 2>&1
}
evalc() {  # name step notes(0/1)
  C=runs/lr/$1/checkpoints/$2/pretrained_model
  until [ -f $C/model.safetensors ]; do sleep 60; done; sleep 60
  (cd seqpol; CUDA_VISIBLE_DEVICES=7 C420=1 NOTES=$3 NOTES_V=2 MEM_GB=15 NOTE_LOG=../logs/notes_c420_$1_$2.txt $E -W ignore evaluate.py --ckpt ../$C --tasks 65 --episodes 50 --sim_backend cpu --out ../logs/eval_c420_$1_$2.json > ../logs/eval_c420_$1_$2.log 2>&1)
  echo "$1 $2 $(tail -1 logs/eval_c420_$1_$2.log)" >> logs/chain_t65.log
}
train pi05_t65_notes /data/workspaces/mikasa/data_mikasa_robo/notes/t65_v2 20000 &
for s in 010000 020000; do evalc pi05_t65_notes $s 1; done
wait
train pi05_t65_base /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/chain_of_colors_3_long_vla_v0 10000
evalc pi05_t65_base 010000 0
