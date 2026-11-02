# 다른 서버에서 실행하기 (11/2 밤 GPU 배치)

`tools/night.sh` 하나로 아래 세 가지를 돌립니다. 다른 실험은 원래 서버 GPU 7에서 계속 돌아갑니다.

| 작업 | GPU (기본값) | 결과 |
|---|---|---|
| v7 학습 (26개 과제, 45k) → 26개 과제 평가 | `V7_GPUS=0,1,2,3` | `runs/lr/pi05_long_v7/checkpoints/045000`, `logs/eval_long_pi05_long_v7_045000_m*.json` |
| v8 학습 (26개 과제, 45k) | `V8_GPUS=4,5` | `runs/lr/pi05_long_v8/checkpoints/045000` (평가는 원래 서버에서) |
| 짧은 RC 3/5/9 v6: 노트 생성 → 10k 학습 → 평가 | `RC_GPU=6` | `logs/eval_db_pi05_t1x_v6_010000.json` |

- **분산 학습:** v7과 v8은 GPU 여러 장에 나눠 학습합니다(`torch.distributed.run`, `--parallelism.dp_replicate=K`, GPU당 batch `16/K`).
  - 한 step에 쓰는 데이터는 16으로, 원래 서버의 1장 학습(base, v6)과 같습니다. step 수와 학습률 일정도 같습니다(90k 일정을 45k에서 멈춤).
  - GPU 수 K는 1, 2, 4, 8 중 하나여야 합니다(16을 나눌 수 있어야 함).
- **예상 시간 (H100급):**
  - 1장 학습 기준 약 1.3초/step이라 45k step에 약 16시간입니다. 4장이면 약 4–5시간, 2장이면 약 8–9시간입니다.
  - 짧은 RC는 노트 생성 약 30분, 학습 약 4시간, 평가 약 30분입니다.
- **시작 전 확인:** 스크립트는 시작할 때 지정한 GPU가 비어 있는지(사용 메모리 2 GB 미만) 확인하고, 하나라도 사용 중이면 시작하지 않습니다.

## 1. 필요한 것

**디렉토리 구조.** 코드에 `/data/workspaces/mikasa/...` 절대경로가 박혀 있습니다. 같은 구조를 만들거나, 다른 위치라면 2절의 경로 치환을 합니다.

```
/data/workspaces/mikasa/
  research/            # 이 저장소
  MIKASA-Robo/         # git clone https://github.com/CognitiveAISystems/MIKASA-Robo, commit 509b875
  envs/lr/             # 학습/정책 서버 환경 (Python 3.12), env/requirements_train.txt
  MIKASA-Robo/.venv/   # 시뮬레이터 환경 (Python 3.11), env/requirements_sim.txt (MIKASA-Robo를 -e로 설치)
  data_mikasa_robo/
    notes/long_v7/     # 13 GB, 원래 서버에서 복사
    notes/long_v8/     # 13 GB, 원래 서버에서 복사
    data_lerobot/remember_color_{3,5,9}_vla_v0/   # 각 19 MB, 원래 서버에서 복사
    mm/t13 mm/t14 mm/t15/                         # 각 약 450 MB, 노트 생성 입력(top.npy, state.npy 등), 복사
  cache/hf_datasets/   # 비어 있어도 됨
```

**모델 가중치.** Hugging Face 캐시(`~/.cache/huggingface/hub`)에 다음이 있어야 합니다.
- `lerobot/pi05_base` (14 GB)
- `google/paligemma-3b-pt-224`: 토크나이저용. **gated 모델이라 접근 권한이 있는 계정으로 로그인해야 합니다.**
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`: 짧은 RC 노트 생성용

원래 서버의 캐시를 통째로 복사하는 것이 가장 간단합니다. 토큰은 저장소나 로그에 남기지 마세요.

**환경.**
- 학습 환경: `lerobot @ git+https://github.com/huggingface/lerobot.git@146d847`, `torch==2.11.0+cu128`. 전체 목록은 `env/requirements_train.txt`에 있습니다.
- 시뮬레이터 환경: `mani-skill==3.0.0b15`, `torch==2.2.1`, MIKASA-Robo editable 설치. 전체 목록은 `env/requirements_sim.txt`에 있습니다.

## 2. 경로가 다를 때

```bash
BASE=/새/경로    # mikasa 디렉토리에 해당
cd $BASE/research
grep -rl /data/workspaces/mikasa --include=*.py --include=*.sh . | xargs sed -i "s#/data/workspaces/mikasa#$BASE#g"
```

이 치환은 커밋하지 마세요. 원래 서버 경로가 기준입니다.

## 3. 실행

```bash
cd /data/workspaces/mikasa/research
V7_GPUS=0,1,2,3 V8_GPUS=4,5 RC_GPU=6 setsid nohup tools/night.sh > logs/night.out 2>&1 < /dev/null &
```

**시작 후 확인할 것:**
- `runs/lr/pi05_long_v7.log`, `runs/lr/pi05_long_v8.log`: 첫 수백 step의 속도(s/step)와 에러를 봅니다.
- 분산 학습이 실패하면 GPU를 1장만 지정해 다시 실행합니다(`V7_GPUS=0`, 이 경우 K=1, batch 16).
- `logs/night_v7.log`, `logs/night_v8.log`, `logs/night_rc.log`: 각 작업의 진행 기록입니다.

**정해진 시간에 끝내야 할 때:** 끝나지 않은 학습은 마지막 저장본(5k step 간격)에서 재개할 수 있습니다.

```bash
pkill -f night.sh; pkill -f "torch.distributed.run"; pkill -f lerobot-train
```

재개할 때는 `lerobot-train --config_path=runs/lr/NAME/checkpoints/last/pretrained_model/train_config.json --resume=true`를 씁니다.

## 4. 원래 서버로 가져올 것

- `logs/eval_db.log`의 새 줄(`pi05_long_v7 045000 long ...`, `pi05_t1x_v6 010000 ...`)
- `logs/eval_long_pi05_long_v7_045000_m*.{json,log}`, `logs/notes_long_pi05_long_v7_045000_m*.txt`
- `logs/eval_db_pi05_t1x_v6_010000.{json,log}`, `logs/notes_db_pi05_t1x_v6_010000.txt`
- `runs/lr/pi05_long_v8/checkpoints/045000/` (약 23 GB): 원래 서버에서 v8 평가를 하기 위해 필요합니다.
- `runs/lr/pi05_long_v7/checkpoints/045000/`: 데모 영상 녹화용입니다(선택).

결과 json과 로그는 이 저장소에 커밋해도 됩니다(`.gitignore`가 `logs/eval_*.json`을 허용함). 체크포인트는 git에 넣지 말고 직접 복사하세요.
