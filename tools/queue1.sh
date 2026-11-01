#!/bin/bash
# sequential GPU-7 queue (one training at a time): t65 oracle upper bound, then t51 with v2 notes
cd /data/workspaces/mikasa/research
tools/chain_t65_oracle.sh
tools/chain_t51v2.sh
