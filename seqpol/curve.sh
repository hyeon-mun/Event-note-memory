#!/bin/bash
# usage: curve.sh RUN_DIR TASK  -> one line per checkpoint: step train_early val_all val_early
PY=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python
cd /data/workspaces/mikasa/research/seqpol
for ck in $(ls $1/ckpt_*.pt | sort -t_ -k2 -n); do
  s=$(basename $ck .pt | cut -d_ -f2)
  v=$($PY -W ignore analyze.py --ckpt $ck --tasks $2 --n_eps 25 --ep_lo 225 --out /tmp/claude-1003/c$$.json 2>/dev/null | grep l_mem | python3 -c "import sys; l=sys.stdin.read().split(); i=l.index('step:'); v=list(map(float,l[i+1:])); print(l[3], round(sum(v[:9])/9,3))")
  echo "$(basename $1) step $s val(all,early) $v"
done
