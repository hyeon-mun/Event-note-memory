#!/bin/bash
# 26-task evals at 45k, one model at a time (4 groups x 15 GB), after the pi05_long_v6 90k final eval
cd /data/workspaces/mikasa/research
while pgrep -f "[q]ueue15.sh" >/dev/null || pgrep -f "[m]id_eval_long.sh pi05_long_v6" >/dev/null; do sleep 120; done
tools/mid_eval_long.sh pi05_long_base 045000 0 0
tools/mid_eval_long.sh pi05_long_v7 045000 1 7
VMEM_FULL=1 tools/mid_eval_long.sh pi05_long_visfull 045000 0 0
tools/mid_eval_long.sh pi05_long_v8 045000 1 8
echo EQ17_DONE
