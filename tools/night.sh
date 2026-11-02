#!/bin/bash
# multi-GPU batch (see docs/RUN_ELSEWHERE.md): v7 and v8 trainings with data parallelism (global batch 16 = same
# as the single-GPU runs, same 90k schedule, stopped at 45k) + v7 26-task eval, and short RC v6
# (notes -> 10k training -> eval). Optional: v6 45k test-time-v7 eval (TT7=1, needs the v6 45k checkpoint).
# GPU sets: V7_GPUS (default 0,1,2,3), V8_GPUS (4,5), RC_GPU (6)
cd /data/workspaces/mikasa/research
V7_GPUS=${V7_GPUS:-0,1,2,3}; V8_GPUS=${V8_GPUS:-4,5}; RC_GPU=${RC_GPU:-6}
n7=$(echo $V7_GPUS | tr , ' ' | wc -w); n8=$(echo $V8_GPUS | tr , ' ' | wc -w)
for g in $(echo $V7_GPUS,$V8_GPUS,$RC_GPU | tr , ' '); do
  u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
  [ $u -lt 2000 ] || { echo "GPU $g busy ($u MiB), abort"; exit 1; }
done
export HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets
P=/data/workspaces/mikasa/envs/lr/bin/python; LT=/data/workspaces/mikasa/envs/lr/bin/lerobot-train
M=/data/workspaces/mikasa/data_mikasa_robo
COMMON="--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --log_freq=100 --num_workers=6 --wandb.enable=false"
ddp() {  # NAME DATA GPUS NPROC PORT
  local N=$1 R=$2 G=$3 K=$4
  CUDA_VISIBLE_DEVICES=$G setsid $P -m torch.distributed.run --nproc-per-node=$K --master-port=$5 $LT --dataset.repo_id=local/$N --dataset.root=$R $COMMON --steps=90000 --save_freq=5000 --batch_size=$((16 / K)) --parallelism.dp_replicate=$K --output_dir=runs/lr/$N > runs/lr/$N.log 2>&1 &
  local TP=$! C=runs/lr/$N/checkpoints/045000/pretrained_model/model.safetensors
  until [ -f $C ] || ! kill -0 $TP 2>/dev/null; do sleep 60; done
  sleep 120; kill -- -$TP 2>/dev/null; sleep 30; kill -9 -- -$TP 2>/dev/null
  echo "$N done $(date)"
}
( ddp pi05_long_v7 $M/notes/long_v7 $V7_GPUS $n7 29511
  GPU=${V7_GPUS%%,*} tools/mid_eval_gpu.sh pi05_long_v7 045000 1 7 ) > logs/night_v7.log 2>&1 &
( ddp pi05_long_v8 $M/notes/long_v8 $V8_GPUS $n8 29512 ) > logs/night_v8.log 2>&1 &
( cd notes_vlm
  for x in "13 3" "14 5" "15 9"; do set -- $x
    [ -f $M/notes/t${1}_v6/meta/info.json ] || CUDA_VISIBLE_DEVICES=$RC_GPU $P -W ignore build_ds.py $1 $M/data_lerobot/remember_color_$2_vla_v0 $M/notes/t${1}_v6 6 2>&1 | grep -v -i "warn\|kwargs\|Loading weights"
  done
  cd ..
  $P -W ignore tools/merge_ds.py $M/notes/t1x_v6 $M/notes/t13_v6 $M/notes/t14_v6 $M/notes/t15_v6 || exit 1
  N=pi05_t1x_v6
  CUDA_VISIBLE_DEVICES=$RC_GPU $P -W ignore $LT --dataset.repo_id=local/$N --dataset.root=$M/notes/t1x_v6 $COMMON --steps=10000 --save_freq=2500 --batch_size=16 --output_dir=runs/lr/$N > runs/lr/$N.log 2>&1
  C=runs/lr/$N/checkpoints/010000/pretrained_model
  (cd seqpol; CUDA_VISIBLE_DEVICES=$RC_GPU DEADBAND=0.001 C420=1 NOTES=1 NOTES_V=6 MEM_GB=15 NOTE_LOG=../logs/notes_db_${N}_010000.txt /data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python -W ignore evaluate.py --ckpt ../$C --tasks 13,14,15 --episodes 50 --sim_backend gpu --out ../logs/eval_db_${N}_010000.json > ../logs/eval_db_${N}_010000.log 2>&1)
  grep "sr=" logs/eval_db_${N}_010000.log | sed "s/^/$N 010000 /" >> logs/eval_db.log
  echo "short RC done $(date)" ) > logs/night_rc.log 2>&1 &
[ "$TT7" = 1 ] && ( sleep 900; GPU=$RC_GPU tools/mid_eval_gpu.sh pi05_long_v6 045000 1 7 pi05_long_v6tt7 ) > logs/night_tt7.log 2>&1 &
wait; echo NIGHT_DONE $(date)
