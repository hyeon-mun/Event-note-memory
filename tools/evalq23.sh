#!/bin/bash
# GPU 7 only (from 11/3 10:00). One queue for everything left, so that no two jobs start at the same time:
#  S1 short RC v6: notes (t13-15) -> 10k training (40 GB cap) -> eval (GPU physics)
#  S2 v8 45k 26-task eval (4 groups)      S3 v7 t72/82/83 + keyframes 45k 26-task eval (3 groups)
#  S4 delay 1000 on t50/t53/t56 for v6 and base 45k (two at a time)
# Every start holds a lock, waits for enough free memory on GPU 7, and lets the new job reach its memory before
# releasing (gpuguard kills all my GPU-7 jobs below 15 GB). S2-S4 wait until the RC training is on the GPU.
cd /data/workspaces/mikasa/research
export PROC_TITLE="python main.py" PYTHONPATH=/data/workspaces/mikasa/research/tools/ph${PYTHONPATH:+:$PYTHONPATH}
export HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets
P=/data/workspaces/mikasa/envs/lr/bin/python; LT=/data/workspaces/mikasa/envs/lr/bin/lerobot-train
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python; M=/data/workspaces/mikasa/data_mikasa_robo
R=/data/workspaces/mikasa/research; L=$R/logs/.gpu7.lock; F=$R/logs/.t1x_v6_started; rm -f $F
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 7; }
slot() {  # NEED_MIB SETTLE_S CMD... : run CMD in the background once NEED_MIB is free; sets $! 
  local n=$1 s=$2; shift 2
  exec 9>$L; flock 9
  until [ $(free) -gt $n ]; do sleep 60; done
  "$@" & SP=$!
  sleep $s; flock -u 9; exec 9>&-
}
ev() {  # NAME STEP NOTES NOTES_V TAG TASKS SIM DEADBAND OUT_PREFIX [extra env]
  (cd seqpol; env CUDA_VISIBLE_DEVICES=7 DEADBAND=$8 C420=1 NOTES=$3 NOTES_V=$4 MEM_GB=15 ${10} NOTE_LOG=../logs/notes_$9_$5.txt $E -W ignore evaluate.py --ckpt ../runs/lr/$1/checkpoints/$2/pretrained_model --tasks $6 --episodes 50 --sim_backend $7 --out ../logs/$9_$5.json > ../logs/$9_$5.log 2>&1)
}
lg() {  # NAME STEP NOTES NOTES_V GROUP TASKS [extra env]: 26-task eval group -> eval_db.log
  ev $1 $2 $3 $4 ${1}_$2_$5 $6 cpu 0.02 long_x "$7"
  mv logs/long_x_${1}_$2_$5.log logs/eval_long_${1}_$2_$5.log; mv logs/long_x_${1}_$2_$5.json logs/eval_long_${1}_$2_$5.json 2>/dev/null
  mv logs/notes_long_x_${1}_$2_$5.txt logs/notes_long_${1}_$2_$5.txt 2>/dev/null
  grep "sr=" logs/eval_long_${1}_$2_$5.log | sed "s/^/$1 $2 long /" >> logs/eval_db.log
}
dl() {  # NAME NOTES NOTES_V TASK: delay 1000 eval -> eval_db.log
  local O=dly_$1_045000_d1000_t$4
  ev $1 045000 $2 $3 $O $4 cpu 0.02 d "DELAY=1000 MAXSTEPS=1200"
  mv logs/d_$O.log logs/$O.log 2>/dev/null; mv logs/d_$O.json logs/$O.json 2>/dev/null; mv logs/notes_d_$O.txt logs/$O.notes.txt 2>/dev/null
  grep "sr=" logs/$O.log | sed "s/^/dly $1 045000 d1000 /" >> logs/eval_db.log
}
waitF() { until [ -f $F ]; do sleep 60; done; }

( N=pi05_t1x_v6; echo "S1 notes $(date)"
  cd notes_vlm
  for x in "13 3" "14 5" "15 9"; do set -- $x
    [ -f $M/notes/t${1}_v6/meta/info.json ] && continue
    slot 35000 120 env CUDA_VISIBLE_DEVICES=7 $P -W ignore build_ds.py $1 $M/data_lerobot/remember_color_$2_vla_v0 $M/notes/t${1}_v6 6; wait $SP
  done; cd ..
  [ -f $M/notes/t1x_v6/meta/info.json ] || $P -W ignore tools/merge_ds.py $M/notes/t1x_v6 $M/notes/t13_v6 $M/notes/t14_v6 $M/notes/t15_v6 || exit 1
  echo "S1 train wait $(date)"
  slot 58000 600 env CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 $LT --dataset.repo_id=local/$N --dataset.root=$M/notes/t1x_v6 --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --log_freq=100 --num_workers=6 --wandb.enable=false --steps=10000 --save_freq=2500 --batch_size=16 --output_dir=runs/lr/$N
  TP=$SP; touch $F; echo "S1 train started $(date)"; wait $TP; echo "S1 train done $(date)"
  slot 35000 300 ev $N 010000 1 6 010000 13,14,15 gpu 0.001 eval_db_$N; wait $SP
  grep "sr=" logs/eval_db_${N}_010000.log | sed "s/^/$N 010000 /" >> logs/eval_db.log; echo "S1 done $(date)" ) >> logs/evalq23.log 2>&1 &
( waitF
  for x in "m0 50,54,58,62,66,70,84" "m1 51,55,59,63,67,71" "m2 52,56,60,64,68,72" "m3 53,57,61,65,69,82,83"; do set -- $x
    slot 35000 300 lg pi05_long_v8 045000 1 8 $1 $2
  done; wait; echo "S2 v8 done $(date)" ) >> logs/evalq23.log 2>&1 &
( waitF
  slot 35000 300 lg pi05_long_v7 045000 1 7 m33 72,82,83
  for x in "m0 64,52,62,58,65,57,51,72,70" "m1 66,63,59,68,54,84,56,83" "m2 67,60,69,55,50,53,61,82,71"; do set -- $x
    slot 35000 300 lg pi05_long_visfull 045000 0 0 $1 $2 VMEM_FULL=1
  done; wait; echo "S3 v7+keyframes done $(date)" ) >> logs/evalq23.log 2>&1 &
( waitF; sleep 3600
  for x in "pi05_long_v6 1 6 50" "pi05_long_base 0 0 50" "pi05_long_v6 1 6 53"; do set -- $x; slot 35000 300 dl $1 $2 $3 $4; wait $SP; done ) >> logs/evalq23.log 2>&1 &
( waitF; sleep 4000
  for x in "pi05_long_base 0 0 53" "pi05_long_v6 1 6 56" "pi05_long_base 0 0 56"; do set -- $x; slot 35000 300 dl $1 $2 $3 $4; wait $SP; done ) >> logs/evalq23.log 2>&1 &
wait; echo EQ23_DONE $(date) >> logs/evalq23.log
