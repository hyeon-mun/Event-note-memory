# 실험 설정, 목록, 계획

기준일은 2026-11-01입니다. 결과 숫자는 [RESULTS.md](RESULTS.md)와 자동 생성 표 [RESULTS_AUTO.md](RESULTS_AUTO.md)에 있습니다.

## 1. 공통 설정 (논문 Setup 절과 Table 1에 들어갈 내용)

### 벤치마크
- MIKASA-Robo-VLA (`MIKASA-Robo` 커밋 `509b875`)
- 전체 90개 과제(공식 분류 short 38 / medium 30 / long 22), 과제당 demo 250개

### 우리가 쓰는 과제
이름에 "-Long"이 붙은 변형 전체 **26개**입니다. 공식 분류로는 long 14개 + medium 12개입니다. 공식 long 중 BatteriesChecker(44-46), TraceShapeSeq(76-78), GatherAndRecall(88-89)은 포함하지 않습니다. 논문에는 "사용자 정의 과제 묶음"으로 명시합니다.

| id | env_id | 메모리 유형 | 공식 split | demo 수집 방식 |
|---|---|---|---|---|
| t50 | `RememberColor3-Long-VLA-v0` | Object | medium | MP |
| t51 | `RememberColor5-Long-VLA-v0` | Object | medium | MP |
| t52 | `RememberColor9-Long-VLA-v0` | Object | medium | MP |
| t53 | `RememberShape3-Long-VLA-v0` | Object | medium | MP |
| t54 | `RememberShape5-Long-VLA-v0` | Object | medium | MP |
| t55 | `RememberShape9-Long-VLA-v0` | Object | medium | MP |
| t56 | `RememberShapeAndColor3x2-Long-VLA-v0` | Object | medium | MP |
| t57 | `RememberShapeAndColor3x3-Long-VLA-v0` | Object | medium | MP |
| t58 | `RememberShapeAndColor5x3-Long-VLA-v0` | Object | medium | MP |
| t59 | `BunchOfColors3-Long-VLA-v0` | Capacity | long | MP |
| t60 | `BunchOfColors5-Long-VLA-v0` | Capacity | long | MP |
| t61 | `BunchOfColors7-Long-VLA-v0` | Capacity | long | MP |
| t62 | `SeqOfColors3-Long-VLA-v0` | Capacity | long | MP |
| t63 | `SeqOfColors5-Long-VLA-v0` | Capacity | long | MP |
| t64 | `SeqOfColors7-Long-VLA-v0` | Capacity | long | MP |
| t65 | `ChainOfColors3-Long-VLA-v0` | Sequential | long | MP |
| t66 | `ChainOfColors5-Long-VLA-v0` | Sequential | long | MP |
| t67 | `ChainOfColors7-Long-VLA-v0` | Sequential | long | MP |
| t68 | `ShellGameShuffleTouch-Long-VLA-v0` | Tracking | medium | MP |
| t69 | `ShellGameShuffleColorLampTouch-Long-VLA-v0` | Tracking | medium | MP |
| t70 | `BlinkCountButtonPressEasy-Long-VLA-v0` | Temporal | long | MP |
| t71 | `BlinkCountButtonPressMedium-Long-VLA-v0` | Temporal | long | MP |
| t72 | `BlinkCountButtonPressHard-Long-VLA-v0` | Temporal | long | MP |
| t82 | `TimedTransferEasy-Long-VLA-v0` | Temporal | medium | MP |
| t83 | `TimedTransferMedium-Long-VLA-v0` | Temporal | long | MP |
| t84 | `TimedTransferHard-Long-VLA-v0` | Temporal | long | MP |
| t13-15 | `RememberColor{3,5,9}-VLA-v0` | Object | short | PPO (참고 실험만) |

### 정책과 학습

| 항목 | 값 |
|---|---|
| 정책 | π0.5, LeRobot `lerobot/pi05_base`에서 fine-tune |
| 정밀도 | bf16 |
| 메모리 절약 | gradient checkpointing |
| 입력 | 상단·손목 카메라 128×128, proprio |
| 행동 | chunk 10 예측, 5 step 실행 |
| batch | 16 |
| 학습 방식 | 과제 묶음당 정책 1개를 함께 학습(joint). RC-Long 3과제 묶음은 20k step(약 6.5시간), 26개 과제 묶음은 90k step(약 22시간) |
| 하드웨어 | GPU 1장, 학습 1개당 40GB 상한 |

### 평가 프로토콜

| 항목 | 값 |
|---|---|
| 에피소드 수 | 과제당 50개, seed 4242424242 + i로 고정 |
| RC9-Long 추가 평가 | 주요 모델은 seed 50~149를 더해 150 에피소드 |
| physics | MP로 수집된 과제(Long 전부)는 **CPU physics**. 이유는 아래 |
| 관측 열화 | demo 영상과 같은 yuv420 색 열화(`C420=1`) |
| 정지 판정 | `DEADBAND=0.02` |
| 보고 checkpoint | 마지막 checkpoint만 보고(best checkpoint 선택 없음) |
| 지표 | 성공률(`success_once`), Wilson 95% CI, 단일 선택 과제는 가장 가까이 다가간 큐브(선택 정확도) |

**CPU physics를 쓰는 이유:** MP 수집기는 `sim_backend`를 지정하지 않아 CPU physics로 수집됐습니다. demo 행동을 같은 seed로 재생하면 CPU에서는 8/8 성공하고, 공식 GPU physics에서는 0/8 성공합니다(2.5~4.5cm 궤적 오차, IK 차이). 이 문제는 LOG.md 10/19 항목에 기록돼 있습니다.

### 평가 명령 예시

```bash
cd seqpol
CUDA_VISIBLE_DEVICES=7 DEADBAND=0.02 C420=1 NOTES=1 NOTES_V=6 MEM_GB=15 NOTE_LOG=../logs/x.notes.txt \
  python evaluate.py --ckpt ../runs/lr/pi05_t5x_v6/checkpoints/020000/pretrained_model --tasks 50,51,52 --episodes 50 --sim_backend cpu --out ../logs/x.json
```

평가 변형 옵션(환경 변수):

| 변수 | 효과 |
|---|---|
| `NOTE_SWAP=oracle\|wrong` | 온라인 "Gone:" 노트의 색을 정답 색 또는 다른 후보 색으로 교체 |
| `DELAY=1000 MAXSTEPS=1200` | 단서 후 대기 시간을 고정하고 에피소드 길이를 늘림 |

## 2. 실험 목록

### 2.1 완료

| run 이름 | 데이터 | step | 내용 |
|---|---|---|---|
| `pi05_t5x_{base,v2,v3,v4,v5,v6,oracle}` | RC-Long 3과제(t50-52) | 20k | 노트 버전별 비교, 상한 |
| `pi05_t5x_vis` | RC-Long 3과제 | 20k | crop 시각 메모리를 추가 카메라로 입력 |
| `abl_v4model_v3notes`, `abl_v3model_v4notes` | – | – | 테스트 때 노트 규칙 교체 |
| `pi05_t1x_{base,oracle}` | 표준 RC 3과제(t13-15) | 10k | 짧은 버전 참고 실험(GPU physics) |
| `pi05_long_base` | 26개 과제 | 90k | 26개 과제 base |
| RC-Long 지연 길이별 재분석 | – | – | `tools/delay_bins.py` (결과 R4) |

### 2.2 진행 중 (11/1 저녁 기준, 큐 스크립트)

| # | run / 실험 | 내용 | 스크립트 | 예상 완료 |
|---|---|---|---|---|
| M1 | `pi05_long_v6` 90k | 26개 과제 노트 모델(주 결과) | `queue15.sh` → `train_eval_long.sh` | 11/2 오전 |
| M1' | `pi05_long_v6` 45k | 중간 평가 | `mid_eval_long.sh` | 11/1 밤 |
| C1 | 노트 내용 통제 | t5x_v6 20k 모델에 노트 비우기, 틀린 색, 정답 색 (RC-Long 3과제) | `controls.sh` | 11/2 |
| C2 | `pi05_long_v8` | **주기적 캡션**: 이벤트 감지·선택 없이 50 step마다 전체 화면 캡션 | `build_cmp.sh`, `queue17.sh`, `evalq17.sh` | 11/4 |
| C3 | `pi05_long_visfull` | **시각 keyframe 메모리**(RePlanMem 방식): 이벤트 때 전체 프레임을 추가 카메라로 입력 | 위와 같음 | 11/4 |
| C4 | `pi05_long_v7` | v6에서 t=0 장면 노트 제거(활용) | 위와 같음 | 11/3 |
| C5 | `pi05_long_base` 45k | 같은 step 비교용 | `evalq17.sh` | 11/2 |

C2~C4는 26개 과제로 **90k 스케줄로 학습하다 45k에서 멈춥니다**(`train45.sh`). 그래서 base와 v6의 45k checkpoint와 학습률 스케줄까지 같습니다. 비교표는 5개 방법(base, v6, v7, visfull, v8) × 26개 과제, 모두 45k step입니다.

### 2.3 추가 예정 (11/2~11/4)

| # | 실험 | 방법 | 비고 |
|---|---|---|---|
| B | 학습 범위 밖 지연 | `DELAY=1000`으로 RC5-Long 평가, 26개 모델 45k 중 v6, v7, visfull, v8 | 평가만. 공식 벤치마크 변형으로 명시 |
| C | 무관한 이벤트 간섭 | 단서가 사라진 뒤 대기 중 회색 물체를 0/2/5번 놓았다 치움(전체 대기 시간 고정). v6, v8, visfull에서 성공률, 단서 노트 보존율, 불필요한 노트 수 측정 | 환경 수정 반나절. 안 되면 제외 |
| D | 이벤트 선택 방식 | demo 영상(GPU 불필요)에 고정 간격 / 픽셀 변화만 / 우리 방식을 적용해 단서 사건 검출 recall·precision, 에피소드당 쓰기 수 비교. 평가 노트 로그로 결정 시점 단서 노트 보존율 계산 | 평가 로그는 `logs/notes_*.txt`, `*.notes.txt` |

### 2.4 하지 않기로 한 것과 이유

| 제안 | 결정 | 이유 |
|---|---|---|
| seed 반복 | 하지 않음 | 시간 부족. 핵심 차이(0.08 vs 0.49, CI 겹치지 않음)는 seed로 뒤집히지 않음 |
| 짧은 RC/Shape에서 노트 학습 | 하지 않음 | PPO expert가 단서 전에 정답 쪽으로 움직임(모방학습으로 메모리 효과 측정 불가). 단서가 1~5 step만 보이고 로봇이 계속 움직여 정지 기반 감지가 작동하지 않음 |
| μVLA, MemoryVLA 등 공개 checkpoint 재평가 | 하지 않음 | 구조와 데이터가 달라 같은 조건이 안 되고 설정에 1~2일 필요. 대신 RePlanMem 방식과 주기적 캡션을 같은 조건에서 직접 구현 |
| RePlanMem 행동 불일치 검출 재현 | 시간이 남으면 | demo 전체에 정책을 돌려야 함(GPU) |
| 지연별 재학습 | 하지 않음 | 학습 범위 안은 이미 무작위 지연(50~450)이라 기존 결과로 대체 |

## 3. 일정

| 날짜 | GPU | GPU 없이 하는 작업 |
|---|---|---|
| 11/1 밤 | v6 90k 최종 평가, v7 학습, 통제 실험 C1 | 간섭 환경 구현 |
| 11/2 | 26개 최종 결과, base 45k 평가, visfull 학습 | 이벤트 선택 분석(D), 원고(방법, 설정) |
| 11/3 | v8 학습 종료, v7·visfull 평가, B·C 평가 | 원고, 그림 |
| 11/4 | v8 평가, B·C 마무리 | 결과 표 |
| 11/5 | **실험 마감** | |
| 11/6~11/9 | | 원고 완성, 검토 |
| 11/10 | 제출 | |

## 4. 데이터와 산출물 위치 (서버)

| 종류 | 위치 |
|---|---|
| 원본 demo (LeRobot) | `/data/workspaces/mikasa/data_mikasa_robo/data_lerobot/<task>` |
| 노트 데이터셋 | `/data/workspaces/mikasa/data_mikasa_robo/notes/t{XX}_v{N}`, 병합본 `t5x_*`, `long_*` |
| 캡션 캐시 | `notes/tXX_events_v3.json`, `tXX_events_v8.json` |
| 체크포인트 | `research/runs/lr/<run>/checkpoints/<step>/pretrained_model` (레포 미포함) |
| 평가 결과 | `logs/eval_*.json`, `logs/eval_db.log` (레포 포함) |
| 온라인 노트 로그 | `logs/notes_*.txt` |
