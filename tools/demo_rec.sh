#!/bin/bash
# record demo videos (+ note log) of the same episodes for notes v6 vs base (t52, t50; episode index 2).
# extra job on a shared GPU: a sidecar kills only this job if free memory drops below 18 GB, before gpuguard (15 GB) fires.
cd /data/workspaces/mikasa/research/seqpol
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
rec() {  # TAG RUN NOTES NOTES_V
  O=../logs/demo_$1
  env CUDA_VISIBLE_DEVICES=7 HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets DEADBAND=0.02 C420=1 NOTES=$3 NOTES_V=$4 MEM_GB=12 NOTE_LOG=$O.notes.txt \
    $E -W ignore evaluate.py --ckpt ../runs/lr/$2/checkpoints/020000/pretrained_model --tasks 52,50 --episodes 1 --ep_lo 2 --sim_backend cpu --video_dir $O --out $O.json > $O.log 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    f=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$f" -lt 18000 ] && { echo "$(date) free=$f sidecar kill $P" >> $O.log; kill $P; }
    sleep 0.5
  done
}
rec v6 pi05_t5x_v6 1 6
rec base pi05_t5x_base 0 0
echo DEMO_DONE
