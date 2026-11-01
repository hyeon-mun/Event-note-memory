#!/bin/bash
# comparison baseline 1: event-triggered full-frame keyframe memory (RePlanMem-style visual keyframes) on t5x, 20k
# starts after the Long training (416210) and the Long 45k mid-evals finish (GPU memory budget)
cd /data/workspaces/mikasa/research
while kill -0 416210 2>/dev/null || pgrep -f "[m]id_eval_long.sh" >/dev/null; do sleep 120; done
VMEM_FULL=1 DEADBAND=0.02 tools/train_eval.sh pi05_t5x_visfull /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_visfull 20000 50,51,52 0 020000
echo Q16_DONE
