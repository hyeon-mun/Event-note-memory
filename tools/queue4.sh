#!/bin/bash
# after t52_v3: one policy for RememberColor 3/5/9-Long (750 demos) with v3 notes vs the same without notes
cd /data/workspaces/mikasa/research
while pgrep -f "[t]rain_eval.sh pi05_t52_v3" >/dev/null; do sleep 120; done
until grep -q "done.*t5x_base" logs/merge_t5x.log; do sleep 120; done
DEADBAND=0.02 NOTES_V=3 tools/train_eval.sh pi05_t5x_v3 /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_v3 20000 50,51,52 1 010000 020000
DEADBAND=0.02 tools/train_eval.sh pi05_t5x_base /data/workspaces/mikasa/data_mikasa_robo/notes/t5x_base 20000 50,51,52 0 020000
