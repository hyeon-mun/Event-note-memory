#!/bin/bash
# after chain_t54: pi05 with the oracle shape note (upper bound: can the policy act on a perfect note at 128 px?)
cd /data/workspaces/mikasa/research
while pgrep -f "[l]erobot-train" -u $USER >/dev/null; do sleep 60; done
P=/data/workspaces/mikasa/envs/lr/bin/python; E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/t54_oracle --dataset.root=/data/workspaces/mikasa/data_mikasa_robo/notes/t54_oracle --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=15000 --log_freq=100 --save_freq=2500 --num_workers=8 --output_dir=runs/lr/pi05_t54_oracle --wandb.enable=false > runs/lr/pi05_t54_oracle.log 2>&1 &
for s in 010000 015000; do
  until [ -f runs/lr/pi05_t54_oracle/checkpoints/$s/pretrained_model/model.safetensors ]; do sleep 60; done; sleep 60
  (cd seqpol; CUDA_VISIBLE_DEVICES=7 MEM_GB=15 $E -W ignore evaluate.py --ckpt ../runs/lr/pi05_t54_oracle/checkpoints/$s/pretrained_model --tasks 54 --episodes 50 --sim_backend cpu --note oracle --out ../logs/eval_pi05_t54_oracle_$s.json > ../logs/eval_pi05_t54_oracle_$s.log 2>&1)
done
wait
