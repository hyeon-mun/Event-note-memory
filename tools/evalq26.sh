#!/bin/bash
# Standard MIKASA-Robo RememberColor{3,5,9}-v0 (ids 91-93; data from tools/conv_v0.py, mm/t91-93 via prep_data.py)
# for the comparison with the memory papers that report on these tasks (250 demos/task, 100 eval episodes).
# Split of evalq25 PART=train so that the no-note run trains while pi05_t1y_v6 is still training (GPU 7 has room for both).
#   PART=base : pi05_t1y_base 10k -> eval (no notes)
#   PART=v6ev : wait for the running pi05_t1y_v6 training (pid V6PID) to exit -> eval with v6 notes
# GPU 7 only: every start holds the shared lock and waits for free memory (see slot() in tools/evalq23.sh).
cd /data/workspaces/mikasa/research
export PROC_TITLE="python main.py" PYTHONPATH=/data/workspaces/mikasa/research/tools/ph${PYTHONPATH:+:$PYTHONPATH}
export HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets
P=/data/workspaces/mikasa/envs/lr/bin/python; LT=/data/workspaces/mikasa/envs/lr/bin/lerobot-train
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python; M=/data/workspaces/mikasa/data_mikasa_robo
R=/data/workspaces/mikasa/research; L=$R/logs/.gpu7.lock
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 7; }
slot() {  # NEED_MIB SETTLE_S CMD... : run CMD in the background once NEED_MIB is free; sets $SP
  local n=$1 s=$2; shift 2
  exec 9>$L; flock 9
  until [ $(free) -gt $n ]; do sleep 60; done
  "$@" & SP=$!
  sleep $s; flock -u 9; exec 9>&-
}
train() {  # NAME DATA
  slot 58000 600 env CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 $LT --dataset.repo_id=local/$1 --dataset.root=$2 --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --log_freq=100 --num_workers=6 --wandb.enable=false --steps=10000 --save_freq=2500 --batch_size=16 --output_dir=runs/lr/$1
  wait $SP
}
ev() {  # NAME NOTES NOTES_V : 100 episodes per task, GPU physics, no deadband (the PPO demos never hold still)
  local C=runs/lr/$1/checkpoints/010000/pretrained_model O=logs/eval_db_$1_010000
  slot 35000 300 bash -c "cd seqpol; CUDA_VISIBLE_DEVICES=7 DEADBAND=0 C420=1 NOTES=$2 NOTES_V=$3 MEM_GB=15 NOTE_LOG=../logs/notes_db_$1_010000.txt $E -W ignore evaluate.py --ckpt ../$C --tasks 91,92,93 --episodes 100 --sim_backend gpu --out ../$O.json > ../$O.log 2>&1"
  wait $SP; grep "sr=" $O.log | sed "s/^/$1 010000 /" >> logs/eval_db.log
}

if [ "$PART" = base ]; then
  train pi05_t1y_base $M/notes/t1y_base; ev pi05_t1y_base 0 0
  echo "base done $(date)"
elif [ "$PART" = v6ev ]; then
  while kill -0 $V6PID 2>/dev/null; do sleep 60; done
  [ -f runs/lr/pi05_t1y_v6/checkpoints/010000/pretrained_model/config.json ] || { echo "no v6 10k checkpoint $(date)"; exit 1; }
  ev pi05_t1y_v6 1 6
  echo "v6 done $(date)"
fi
