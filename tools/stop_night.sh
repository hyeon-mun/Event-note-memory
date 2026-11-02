#!/bin/bash
# end of the night window: stop everything the night batch runs on GPUs 0-6 (GPU-7 queues untouched),
# then copy the per-task results of the cut evals into eval_db.log
cd /data/workspaces/mikasa/research
keep="$(ps -o pgid= -p $(pgrep -f '[e]valq19.sh') $(pgrep -f '[s]ide_q.sh') 2>/dev/null | tr -d ' ' | sort -u)"
for p in $(pgrep -f "[t]ools/night.sh"); do ps -o pgid= -p $p; done | tr -d ' ' | sort -u > /tmp/.ng.$$
for g in 0 1 2 3 4 5 6; do
  for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p $p 2>/dev/null)" = "$USER" ] && ps -o pgid= -p $p
  done
done | tr -d ' ' | sort -u >> /tmp/.ng.$$
for pg in $(sort -u /tmp/.ng.$$); do
  echo "$keep" | grep -qx "$pg" && continue
  echo "$(date) kill group $pg"; kill -- -$pg 2>/dev/null
done
rm -f /tmp/.ng.$$; sleep 30
for f in logs/eval_long_pi05_long_v7_045000_m*.log logs/eval_long_pi05_long_visfull_045000_m*.log; do
  [ -f $f ] || continue; T=$(basename $f | sed -E 's/eval_long_(.+)_045000_m[0-9]+\.log/\1/')
  grep "sr=" $f | sed "s/^/$T 045000 long /" | while read -r l; do grep -qxF "$l" logs/eval_db.log || echo "$l" >> logs/eval_db.log; done
done
echo "STOP_NIGHT_DONE $(date)"
