#!/bin/bash
# after the no-note control finishes: train the VLM-note policy, then evaluate the control on the CPU backend
cd /data/workspaces/mikasa/research
while pgrep -f "[l]erobot-train.*pi05_t51_base" >/dev/null; do sleep 60; done
D=/data/workspaces/mikasa/data_mikasa_robo/notes/t51_vlm
CUDA_VISIBLE_DEVICES=7 /data/workspaces/mikasa/envs/lr/bin/python -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/t51_notes --dataset.root=$D --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=10000 --log_freq=100 --save_freq=2500 --num_workers=8 --output_dir=runs/lr/pi05_t51_notes --wandb.enable=false > runs/lr/pi05_t51_notes.log 2>&1 &
cd seqpol
CUDA_VISIBLE_DEVICES=7 MEM_GB=15 /data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python -W ignore evaluate.py --ckpt ../runs/lr/pi05_t51_base/checkpoints/last/pretrained_model --tasks 51 --episodes 50 --sim_backend cpu --out ../logs/eval_pi05_t51_base_cpu.json > ../logs/eval_pi05_t51_base_cpu.log 2>&1
wait
