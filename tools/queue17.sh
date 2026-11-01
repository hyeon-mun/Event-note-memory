#!/bin/bash
# 26-task comparison trainings (45k of the 90k schedule): v7 -> visual keyframes -> periodic captions
cd /data/workspaces/mikasa/research
M=/data/workspaces/mikasa/data_mikasa_robo/notes
while kill -0 416210 2>/dev/null || pgrep -f "[m]id_eval_long.sh pi05_long_v6" >/dev/null; do sleep 120; done   # long_v6 training + its 45k evals (GPU memory)
tools/train45.sh pi05_long_v7 $M/long_v7
tools/train45.sh pi05_long_visfull $M/long_visfull
tools/train45.sh pi05_long_v8 $M/long_v8
echo Q17_DONE
