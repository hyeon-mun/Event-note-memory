#!/bin/bash
# replaces evalq18: v6 90k final -> v6 45k with test-time v7 notes -> keyframes 45k -> v7 45k -> periodic 45k
cd /data/workspaces/mikasa/research
while pgrep -f "[m]id_eval_long.sh pi05_long_base" >/dev/null; do sleep 120; done
tools/side_q.sh > logs/side_q.log 2>&1 &
tools/mid_eval_tag.sh pi05_long_v6 090000 1 6
tools/mid_eval_tag.sh pi05_long_v6 045000 1 7 pi05_long_v6tt7
VMEM_FULL=1 tools/mid_eval_tag.sh pi05_long_visfull 045000 0 0
tools/mid_eval_tag.sh pi05_long_v7 045000 1 7
tools/mid_eval_tag.sh pi05_long_v8 045000 1 8
wait; echo EQ19_DONE
