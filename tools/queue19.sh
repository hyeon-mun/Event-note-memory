#!/bin/bash
# replaces queue17 after the 11/2 06:04 kill: keyframes (running) -> v7 (retrained) -> periodic captions, all 45k
cd /data/workspaces/mikasa/research
M=/data/workspaces/mikasa/data_mikasa_robo/notes
while pgrep -f "[t]rain45.sh pi05_long_visfull" >/dev/null; do sleep 120; done
tools/train45.sh pi05_long_v7 $M/long_v7
tools/train45.sh pi05_long_v8 $M/long_v8
echo Q19_DONE
