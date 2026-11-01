#!/bin/bash
# evaluate each phase-1 run when its final checkpoint appears
PY=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
cd /data/workspaces/mikasa/research/seqpol
DEV=0,3,12,14,15,31,35,47,79,85
R=/data/workspaces/mikasa/research/runs/p1
GPUS=${GPUS:-2,3,6,7}
for name in none_plain stack_plain gru_plain tf_plain gru_drop gru_paired tf_drop tf_paired; do
  until [ -f $R/$name/ckpt_30000.pt ]; do sleep 60; done
  $PY peval.py --ckpt $R/$name/ckpt_30000.pt --tasks $DEV --out $R/$name/eval_g1 --gpus $GPUS --procs_per_gpu 4 > $R/$name/eval_g1.txt 2>&1
  case $name in *drop|*paired)
    for g in 1.5 2.0 3.0; do
      $PY peval.py --ckpt $R/$name/ckpt_30000.pt --tasks $DEV --out $R/$name/eval_g$g --gpus $GPUS --procs_per_gpu 4 --guidance $g > $R/$name/eval_g$g.txt 2>&1
    done;;
  esac
done
echo P1_EVAL_DONE
