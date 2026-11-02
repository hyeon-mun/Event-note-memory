#!/bin/bash
# B on more memory types: fixed delay 1000 (outside the trained 50-450 range) for the 26-task models at 45k,
# RC3-L (t50), Shape3-L (t53), ShapeAndColor3x2-L (t56); one eval per GPU. usage: [GPUS=1,2,3,4,5,6] delay_more.sh
cd /data/workspaces/mikasa/research
G=(${GPUS:-1,2,3,4,5,6}); G=(${G[@]//,/ }); i=0
for t in 50 53 56; do
  GPU=${G[$((i % ${#G[@]}))]} T=$t tools/delay_eval.sh pi05_long_v6 045000 1 6 & i=$((i + 1))
  GPU=${G[$((i % ${#G[@]}))]} T=$t tools/delay_eval.sh pi05_long_base 045000 0 0 & i=$((i + 1))
done
wait; echo DELAY_MORE_DONE $(date)
