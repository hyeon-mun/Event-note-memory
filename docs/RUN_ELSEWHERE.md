# 다른 서버에서 실행하기 (11/3 밤)

v7·v8 학습과 keyframe(visfull) 학습은 원래 서버(GPU 0–6)에서 돌립니다(`JOBS="v7 v8 vf" TT7=1 tools/night.sh`). 다른 서버에서는 아래 두 가지를 돌립니다.

| 작업 | GPU | 결과 | 예상 시간 |
|---|---|---|---|
| 짧은 RC 3/5/9 v6: 노트 생성 → 10k 학습 → 평가 (`JOBS=rc tools/night.sh`) | `RC_GPU=0` | `logs/eval_db_pi05_t1x_v6_010000.json` | 약 5시간 |
| 지연 1000 step을 다른 메모리 유형으로 확장 (`tools/delay_more.sh`): v6 45k와 base 45k를 RC3-L(t50), Shape3-L(t53), ShapeAndColor3×2-L(t56)에서 평가 | `GPUS=1,2,3,4,5,6` (평가 6개, GPU당 1개) | `logs/dly_pi05_long_{v6,base}_045000_d1000_t{50,53,56}.{json,log}` | 약 2–3시간 |

- 두 번째 실험은 논문의 "학습 범위 밖 지연" 결과(현재 RC5-L 하나뿐)를 다른 메모리 유형으로 넓힙니다. 평가만 하므로 학습은 없습니다.
- 모든 파이썬 프로세스는 nvidia-smi와 ps에 `python main.py`로 보입니다(`tools/ph/sitecustomize.py`). 이를 위해 두 환경 모두에 `setproctitle`이 필요합니다(아래 1절).

## 1. 필요한 것

**디렉토리 구조.** 코드에 `/data/workspaces/mikasa/...` 절대경로가 박혀 있습니다. 같은 구조를 만들거나, 다른 위치라면 2절의 경로 치환을 합니다.

```
/data/workspaces/mikasa/
  research/            # 이 저장소 (git clone)
  MIKASA-Robo/         # git clone https://github.com/CognitiveAISystems/MIKASA-Robo, commit 509b875
  envs/lr/             # 학습/정책 서버 환경 (Python 3.12), env/requirements_train.txt
  MIKASA-Robo/.venv/   # 시뮬레이터 환경 (Python 3.11), env/requirements_sim.txt (MIKASA-Robo를 -e로 설치)
  data_mikasa_robo/
    data_lerobot/remember_color_{3,5,9}_vla_v0/   # 각 19 MB, 원래 서버에서 복사
    mm/t13 mm/t14 mm/t15/                         # 각 약 450 MB, 노트 생성 입력, 복사
  research/runs/lr/pi05_long_v6/checkpoints/045000/pretrained_model/     # 8.8 GB, 복사
  research/runs/lr/pi05_long_base/checkpoints/045000/pretrained_model/   # 8.8 GB, 복사
  cache/hf_datasets/   # 비어 있어도 됨
```

`notes/long_v7`, `notes/long_v8`은 더 이상 필요하지 않습니다(원래 서버에서 학습).

**모델 가중치.** Hugging Face 캐시(`~/.cache/huggingface/hub`)에 `lerobot/pi05_base`, `google/paligemma-3b-pt-224`(gated, 토크나이저), `HuggingFaceTB/SmolVLM2-2.2B-Instruct`가 있어야 합니다. 원래 서버 캐시는 링크 구조라 `tar -h`나 `rsync -L`로 실제 파일을 복사합니다. 토큰은 저장소나 로그에 남기지 마세요.

**환경.** `env/requirements_train.txt`, `env/requirements_sim.txt`로 설치합니다. 이미 있는 환경이라면 `setproctitle`만 추가합니다.
```bash
uv pip install --python /data/workspaces/mikasa/envs/lr/bin/python setproctitle
uv pip install --python /data/workspaces/mikasa/MIKASA-Robo/.venv/bin/python setproctitle
```

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
JOBS=rc RC_GPU=0 setsid nohup tools/night.sh > logs/night.out 2>&1 < /dev/null &
GPUS=1,2,3,4,5,6 setsid nohup tools/delay_more.sh > logs/delay_more.out 2>&1 < /dev/null &
```

- 시작할 때 지정한 GPU가 비어 있는지(2 GB 미만) 확인합니다. `delay_more.sh`는 GPU별 여유 메모리가 35 GB를 넘을 때까지 기다립니다.
- 진행 확인: `tail logs/night_rc.log runs/lr/pi05_t1x_v6.log`, `grep -h "sr=" logs/dly_*_t5[036].log`
- 이름 확인: `nvidia-smi`에 `python main.py`만 보여야 합니다.

## 4. 원래 서버로 가져올 것

- `logs/eval_db.log`의 새 줄 (`pi05_t1x_v6 010000 ...`, `dly ...`)
- `logs/eval_db_pi05_t1x_v6_010000.{json,log}`, `logs/notes_db_pi05_t1x_v6_010000.txt`
- `logs/dly_pi05_long_{v6,base}_045000_d1000_t{50,53,56}.*`

결과 json과 로그는 이 저장소에 커밋해도 됩니다(`.gitignore`가 `logs/eval_*.json`, `logs/dly_*`를 허용함). 체크포인트는 git에 넣지 마세요.
