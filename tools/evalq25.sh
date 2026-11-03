#!/bin/bash
# Standard MIKASA-Robo RememberColor{3,5,9}-v0 (ids 91-93; data from tools/conv_v0.py, mm/t91-93 via prep_data.py)
# for the comparison with the memory papers that report on these tasks (250 demos/task, 100 eval episodes).
#   PART=prep : v6 notes (VLM, GPU 7) -> notes/t1y_v6, plus a 1-episode env smoke test with zero actions
#   PART=train: pi05_t1y_v6 10k -> eval 100 ep/task, then pi05_t1y_base 10k -> eval (no notes)
# GPU 7 only: every start holds the shared lock and waits for free memory (see slot() in tools/evalq23.sh).
cd /data/workspaces/mikasa/research
export PROC_TITLE="python main.py" PYTHONPATH=/data/workspaces/mikasa/research/tools/ph${PYTHONPATH:+:$PYTHONPATH}
export HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets
P=/data/workspaces/mikasa/envs/lr/bin/python; LT=/data/workspaces/mikasa/envs/lr/bin/lerobot-train
E=/data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python; M=/data/workspaces/mikasa/data_mikasa_robo
R=/data/workspaces/mikasa/research; L=$R/logs/.gpu7.lock
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 7; }
slot() {  # NEED_MIB SETTLE_S CMD... : run CMD in the background once NEED_MIB is free; sets $SP
  local n=$1 s=$2; shift 2
  exec 9>$L; flock 9
  until [ $(free) -gt $n ]; do sleep 60; done
  "$@" & SP=$!
  sleep $s; flock -u 9; exec 9>&-
}
train() {  # NAME DATA
  slot 58000 600 env CUDA_VISIBLE_DEVICES=7 $P -W ignore tools/capgpu.py 40 $LT --dataset.repo_id=local/$1 --dataset.root=$2 --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.chunk_size=10 --policy.n_action_steps=5 --policy.push_to_hub=false --log_freq=100 --num_workers=6 --wandb.enable=false --steps=10000 --save_freq=2500 --batch_size=16 --output_dir=runs/lr/$1
  wait $SP
}
ev() {  # NAME NOTES NOTES_V : 100 episodes per task, GPU physics, no deadband (the PPO demos never hold still)
  local C=runs/lr/$1/checkpoints/010000/pretrained_model O=logs/eval_db_$1_010000
  slot 35000 300 bash -c "cd seqpol; CUDA_VISIBLE_DEVICES=7 DEADBAND=0 C420=1 NOTES=$2 NOTES_V=$3 MEM_GB=15 NOTE_LOG=../logs/notes_db_$1_010000.txt $E -W ignore evaluate.py --ckpt ../$C --tasks 91,92,93 --episodes 100 --sim_backend gpu --out ../$O.json > ../$O.log 2>&1"
  wait $SP; grep "sr=" $O.log | sed "s/^/$1 010000 /" >> logs/eval_db.log
}

if [ "${PART:-prep}" = prep ]; then
  echo "notes $(date)"
  cd notes_vlm
  for x in "91 3" "92 5" "93 9"; do set -- $x
    [ -f $M/notes/t${1}_v6/meta/info.json ] && continue
    slot 35000 120 env CUDA_VISIBLE_DEVICES=7 $P -W ignore build_ds.py $1 $M/data_lerobot/remember_color_$2_v0 $M/notes/t${1}_v6 6; wait $SP
  done; cd ..
  [ -f $M/notes/t1y_v6/meta/info.json ] || $P -W ignore tools/merge_ds.py $M/notes/t1y_v6 $M/notes/t91_v6 $M/notes/t92_v6 $M/notes/t93_v6
  echo "smoke $(date)"
  slot 35000 60 bash -c "cd seqpol; CUDA_VISIBLE_DEVICES=7 $E -W ignore -c \"
import torch, evaluate as V
class Z:
    def reset(self, *a, **k): pass
    def act(self, obs):
        assert obs['rgb'].shape[-1] == 6 and obs['proprio'].shape[-1] == 7, (obs['rgb'].shape, obs['proprio'].shape)
        return torch.zeros(8)
for t in (91, 92, 93):
    r = V.run_task(Z(), t, 1, 4242424242, sim_backend='gpu'); print('smoke', t, 'len', r['lens'], 'sr', r['sr'], flush=True)
\""; wait $SP
  echo "prep done $(date)"
else
  train pi05_t1y_v6 $M/notes/t1y_v6; ev pi05_t1y_v6 1 6
  train pi05_t1y_base $M/notes/t1y_base; ev pi05_t1y_base 0 0
  echo "train+eval done $(date)"
fi
