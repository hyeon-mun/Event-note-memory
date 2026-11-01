#!/bin/bash
# after chain_t65: t51 with the v2 write rule (same method as t65), 15k + online-note evals (C420)
cd /data/workspaces/mikasa/research
while pgrep -f "[l]erobot-train" -u $USER >/dev/null; do sleep 120; done
P=/data/workspaces/mikasa/envs/lr/bin/python; E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 /data/workspaces/mikasa/envs/lr/bin/lerobot-train --dataset.repo_id=local/pi05_t51_v2 --dataset.root=/data/workspaces/mikasa/data_mikasa_robo/notes/t51_v2 --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --batch_size=16 --steps=15000 --log_freq=100 --save_freq=2500 --num_workers=8 --output_dir=runs/lr/pi05_t51_v2 --wandb.enable=false > runs/lr/pi05_t51_v2.log 2>&1 &
for s in 010000 015000; do
  C=runs/lr/pi05_t51_v2/checkpoints/$s/pretrained_model
  until [ -f $C/model.safetensors ]; do sleep 60; done; sleep 60
  (cd seqpol; CUDA_VISIBLE_DEVICES=7 DEADBAND=0.001 C420=1 NOTES=1 NOTES_V=2 MEM_GB=15 NOTE_LOG=../logs/notes_c420_pi05_t51_v2_$s.txt $E -W ignore evaluate.py --ckpt ../$C --tasks 51 --episodes 50 --sim_backend cpu --out ../logs/eval_c420_pi05_t51_v2_$s.json > ../logs/eval_c420_pi05_t51_v2_$s.log 2>&1)
  echo "pi05_t51_v2 $s $(tail -1 logs/eval_c420_pi05_t51_v2_$s.log)" >> logs/chain_t51v2.log
done
wait
