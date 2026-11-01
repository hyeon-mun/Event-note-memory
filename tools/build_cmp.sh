#!/bin/bash
# datasets for the 26-task comparison: v7 notes (cached captions), visual keyframes (visfull), periodic captions (v8)
cd /data/workspaces/mikasa/research/notes_vlm
M=/data/workspaces/mikasa/data_mikasa_robo; P=/data/workspaces/mikasa/envs/lr/bin/python
TASKS="$(seq 50 72) 82 83 84"
merge() {  # NAME SUFFIX
  [ -f $M/notes/$1/meta/info.json ] && return
  (cd ..; $P -W ignore tools/merge_ds.py $M/notes/$1 $(for t in $TASKS; do echo $M/notes/t${t}_$2; done) > logs/merge_$1.log 2>&1) || { echo "merge $1 failed"; exit 1; }
  echo "merged $1 $(date)"
}
for t in $TASKS; do
  name=$(python3 -c "import json;print(json.load(open('$M/mm/t$t/meta.json'))['task'])")
  [ -f $M/notes/t${t}_v7/meta/info.json ] || { echo "== v7 t$t $(date)"; CUDA_VISIBLE_DEVICES=7 $P -W ignore build_ds.py $t $M/data_lerobot/$name $M/notes/t${t}_v7 7 2>&1 | grep -v -i "warn\|kwargs\|Loading weights"; }
done
merge long_v7 v7
for t in $TASKS; do
  name=$(python3 -c "import json;print(json.load(open('$M/mm/t$t/meta.json'))['task'])")
  [ -f $M/notes/t${t}_visfull/meta/info.json ] || { echo "== visfull t$t $(date)"; $P -W ignore build_vis.py $t $M/data_lerobot/$name $M/notes/t${t}_visfull full 2>&1 | tail -2; }
done
merge long_visfull visfull
# periodic captions need the VLM on GPU 7: wait until the Long 45k mid-evals are done (memory budget)
while pgrep -f "[m]id_eval_long.sh pi05_long_v6" >/dev/null; do sleep 120; done
for t in $TASKS; do
  name=$(python3 -c "import json;print(json.load(open('$M/mm/t$t/meta.json'))['task'])")
  [ -f $M/notes/t${t}_v8/meta/info.json ] || { echo "== v8 t$t $(date)"; CUDA_VISIBLE_DEVICES=7 $P -W ignore build_ds.py $t $M/data_lerobot/$name $M/notes/t${t}_v8 8 2>&1 | grep -v -i "warn\|kwargs\|Loading weights"; }
done
merge long_v8 v8
echo BUILD_CMP_DONE
