#!/bin/bash
# Kill my GPU-7 processes if free memory on GPU 7 drops below MIN_FREE MiB (protects a co-tenant's periodic eval).
MIN_FREE=${MIN_FREE:-15000}
while true; do
  free=$(nvidia-smi -i 7 --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free" -lt "$MIN_FREE" ]; then
    for p in $(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader); do
      [ "$(ps -o user= -p $p 2>/dev/null)" = "$USER" ] && { echo "$(date) free=$free kill $p $(ps -o args= -p $p | cut -c1-80)"; kill $p; }
    done
  fi
  sleep 2
done
