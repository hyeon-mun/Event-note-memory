#!/bin/bash
# demo videos + note logs on non-colour -Long tasks (one episode per task), long_v6 45k then long_base 45k.
# extra job on the shared GPU: stops itself if free memory < 16.5 GB or when the long_v6 training ends (its final eval starts)
cd /data/workspaces/mikasa/research/seqpol
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
JOBS="53:1 56:2 54:3 57:4 59:0 62:0 65:0 68:3 70:0 82:0"
rec() {  # RUN TAG NOTES V
  for j in $JOBS; do
    t=${j%:*}; e=${j#*:}; O=../logs/demo_long_$2_t$t
    [ -f $O.json ] && continue
    env CUDA_VISIBLE_DEVICES=7 HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=$3 NOTES_V=$4 MEM_GB=15 NOTE_LOG=$O.notes.txt \
      $E -W ignore evaluate.py --ckpt ../runs/lr/$1/checkpoints/045000/pretrained_model --tasks $t --episodes 1 --ep_lo $e --sim_backend cpu --video_dir $O --out $O.json > $O.log 2>&1 &
    P=$!
    while kill -0 $P 2>/dev/null; do
      f=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
      if [ "$f" -lt 16500 ] || ! kill -0 416210 2>/dev/null; then echo "$(date) free=$f stop" >> $O.log; pkill -P $P; kill $P; echo STOPPED; exit 1; fi
      sleep 0.5
    done
  done
}
rec pi05_long_v6 v6 1 6
rec pi05_long_base base 0 0
echo DEMO_LONG_DONE
