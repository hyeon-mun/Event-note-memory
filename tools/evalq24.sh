#!/bin/bash
# GPU 7: remaining evals after the 11/4 scope cut (see note below); shares the start lock with evalq23 (short RC)
cd /data/workspaces/mikasa/research
export PROC_TITLE="python main.py" PYTHONPATH=/data/workspaces/mikasa/research/tools/ph${PYTHONPATH:+:$PYTHONPATH}
export HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets
P=/data/workspaces/mikasa/envs/lr/bin/python; LT=/data/workspaces/mikasa/envs/lr/bin/lerobot-train
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python; M=/data/workspaces/mikasa/data_mikasa_robo
R=/data/workspaces/mikasa/research; L=$R/logs/.gpu7.lock; F=$R/logs/.t1x_v6_started
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
# 11/4 00:05 (user): compare against existing methods with what we have. v8 (periodic captions) and the extra delay
# evals are dropped; keyframes (RePlanMem-style visual memory) only on the 9 single-cue tasks, where any method
# is above zero (the other 17 tasks are near 0 for base, v6 and v7).
( slot 35000 300 lg pi05_long_v7 045000 1 7 m33 72,82,83
  for x in "m0 54,57,50" "m1 55,58,51" "m2 56,53,52"; do set -- $x
    slot 35000 300 lg pi05_long_visfull 045000 0 0 $1 $2 VMEM_FULL=1
  done; wait; echo "S3 v7+keyframes(9) done $(date)" ) >> logs/evalq24.log 2>&1
