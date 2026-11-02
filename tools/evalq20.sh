#!/bin/bash
# GPU 7 evals during/after the night run (replaces evalq19): v6 90k (running) -> keyframes 45k -> v8 45k
# (v7 and the test-time v7 variant are evaluated on GPUs 0 and 6 by night.sh)
cd /data/workspaces/mikasa/research
while pgrep -f "[m]id_eval_tag.sh pi05_long_v6 090000" >/dev/null; do sleep 120; done
VMEM_FULL=1 tools/mid_eval_gpu.sh pi05_long_visfull 045000 0 0
tools/mid_eval_gpu.sh pi05_long_v8 045000 1 8
wait; echo EQ20_DONE
