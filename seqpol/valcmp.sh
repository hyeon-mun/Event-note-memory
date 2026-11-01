#!/bin/bash
# usage: valcmp.sh RUN_DIR TASK [ckpt-name]  -> train/val loss (all steps, and steps 0-8)
PY=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
cd /data/workspaces/mikasa/research/seqpol
ck=${3:-ckpt_8000.pt}
for split in "train 0" "val 225"; do set -- $1 $2 $split
  $PY -W ignore analyze.py --ckpt $1/$ck --tasks $2 --n_eps 25 --ep_lo $4 --out /tmp/claude-1003/v.json 2>/dev/null | grep "l_mem" | \
   python3 -c "import sys; l=sys.stdin.read().split(); i=l.index('step:'); v=list(map(float,l[i+1:])); print('$3', 'all', l[3], 'early(0-8)', round(sum(v[:9])/9,3))"
done
