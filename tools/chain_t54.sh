#!/bin/bash
# t54 (RememberShape5-Long): VLM-note policy 20k + online-note evals, then the no-note control 10k + eval
cd /data/workspaces/mikasa/research
P=/data/workspaces/mikasa/envs/lr/bin/python; E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
train() {  # name dataset_root steps
  CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/$1 --dataset.root=$2 --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=$3 --log_freq=100 --save_freq=2500 --num_workers=8 --output_dir=runs/lr/$1 --wandb.enable=false > runs/lr/$1.log 2>&1
}
evalc() {  # name step notes(0/1)
  (cd seqpol; CUDA_VISIBLE_DEVICES=7 NOTES=$3 MEM_GB=15 NOTE_LOG=../logs/notes_$1_$2.txt $E -W ignore evaluate.py --ckpt ../runs/lr/$1/checkpoints/$2/pretrained_model --tasks 54 --episodes 50 --sim_backend cpu --out ../logs/eval_$1_$2.json > ../logs/eval_$1_$2.log 2>&1)
}
train pi05_t54_notes /data/workspaces/mikasa/data_mikasa_robo/notes/t54_vlm 20000 &
for s in 010000 020000; do
  until [ -f runs/lr/pi05_t54_notes/checkpoints/$s/pretrained_model/model.safetensors ]; do sleep 60; done; sleep 60
  evalc pi05_t54_notes $s 1
done
wait
train pi05_t54_base /data/workspaces/mikasa/data_mikasa_robo/data_lerobot/remember_shape_5_long_vla_v0 10000
evalc pi05_t54_base 010000 0
