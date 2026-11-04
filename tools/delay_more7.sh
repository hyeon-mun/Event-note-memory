#!/bin/bash
# delay_more.sh on GPU 7 only, starts staggered by 90 s so each free-memory check sees the previous eval
cd /data/workspaces/mikasa/research
for t in 50 53 56; do
  GPU=7 T=$t tools/delay_eval.sh pi05_long_v6 045000 1 6 & sleep 90
  GPU=7 T=$t tools/delay_eval.sh pi05_long_base 045000 0 0 & sleep 90
done
wait; echo DELAY_MORE_DONE $(date)
