#!/bin/bash
# t65 oracle-note upper bound: after chain_t65 (notes + base), train pi05_t65_oracle 20k, eval 10k/20k (deadband, C420)
cd /data/workspaces/mikasa/research
while pgrep -f "[l]erobot-train" -u $USER >/dev/null; do sleep 120; done
P=/data/workspaces/mikasa/envs/lr/bin/python; E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/pi05_t65_oracle --dataset.root=/data/workspaces/mikasa/data_mikasa_robo/notes/t65_oracle --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=20000 --log_freq=100 --save_freq=2500 --num_workers=8 --output_dir=runs/lr/pi05_t65_oracle --wandb.enable=false > runs/lr/pi05_t65_oracle.log 2>&1 &
for s in 010000 020000; do
  C=runs/lr/pi05_t65_oracle/checkpoints/$s/pretrained_model
  until [ -f $C/model.safetensors ]; do sleep 60; done; sleep 60
  (cd seqpol; CUDA_VISIBLE_DEVICES=7 DEADBAND=0.001 C420=1 MEM_GB=15 $E -W ignore evaluate.py --ckpt ../$C --tasks 65 --episodes 50 --sim_backend cpu --note oracle --out ../logs/eval_db_pi05_t65_oracle_$s.json > ../logs/eval_db_pi05_t65_oracle_$s.log 2>&1)
  echo "pi05_t65_oracle $s $(tail -1 logs/eval_db_pi05_t65_oracle_$s.log)" >> logs/eval_db.log
done
wait
