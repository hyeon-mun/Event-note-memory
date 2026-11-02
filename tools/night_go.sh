#!/bin/bash
cd /data/workspaces/mikasa/research
pkill -f "[q]ueue19.sh"; pkill -f "[e]valq19.sh"     # parents only; the running v6 90k eval and visfull training continue
setsid nohup tools/evalq20.sh > logs/evalq20.out 2>&1 < /dev/null &
setsid nohup tools/night.sh > logs/night.out 2>&1 < /dev/null &
