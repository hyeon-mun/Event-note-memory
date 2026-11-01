#!/bin/bash
cd /data/workspaces/mikasa/research
T=52 tools/delay_eval.sh pi05_long_v6 045000 1 6 &
sleep 600; T=52 tools/delay_eval.sh pi05_long_base 045000 0 0 &
wait; echo DLYQ2_DONE
