# Event-Driven Language Memory for π0.5 (MIKASA-Robo-VLA long-horizon tasks)

π0.5(LeRobot)의 구조는 그대로 두고, 화면에서 **사라진 것만** 짧은 텍스트 노트로 적어 지시문 뒤에 붙이는 메모리 연구입니다. IJCAS 특별호 제출(2026-11-10)을 목표로 합니다.

> 핵심 결과 (같은 데이터, 같은 학습량의 노트 없는 π0.5와 비교)
> - RememberColor-Long: 0.14 → 0.96 (RC3), 0.14 → 0.94 (RC5), 0.08 → 0.49 (RC9, 150 에피소드)
> - Long 26개 과제 중 평가가 끝난 15개 평균: 0.045 → 0.217 (노트 모델은 아직 45k step)

## 논문 쓰는 사람은 여기서 시작하세요

| 문서 | 내용 |
|---|---|
| [docs/PAPER_GUIDE.md](docs/PAPER_GUIDE.md) | **논문 작성 가이드**: 투고 양식, 논문의 입장(주장할 것과 하지 말 것), 제목, 가설, 기여, 절별 구성과 그림·표의 데이터 출처, 리뷰어 질문 대응 |
| [docs/METHOD.md](docs/METHOD.md) | 방법 상세: 단계별 파라미터, VLM 프롬프트 원문, 노트 버전(v1~v8) 차이, 실제 노트 예시 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 실험 설정(26개 env_id, 학습·평가 프로토콜, CPU physics 이유), 완료·진행 중·예정 실험, 하지 않기로 한 것과 이유, 일정 |
| [docs/RESULTS.md](docs/RESULTS.md) | 주요 결과표와 해석 (R1~R6) |
| [docs/RESULTS_AUTO.md](docs/RESULTS_AUTO.md) | 로그에서 자동 생성한 전체 결과표. `python3 tools/make_tables.py`로 갱신 |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | 관련 연구 비교와 확인한 수치 |
| [docs/report/index.html](docs/report/index.html) | 그림과 시연 영상이 들어간 보고서(브라우저로 열기). 영상 재생에 맞춰 노트가 실시간으로 표시됨 |
| [docs/figures/](docs/figures) | 파이프라인 그림(`pipeline.svg`), 오류 분해 그림, 노트 예시 프레임과 crop |
| [paper/main.tex](paper/main.tex) | IJCAS 양식 원고(Overleaf에 `paper/` 폴더째 업로드) |
| [RESEARCH_SUMMARY_1101.md](RESEARCH_SUMMARY_1101.md) | 연구 전체 요약(시간순 발견 포함) |
| [LOG.md](LOG.md) | 날짜별 실험 기록(실패와 원인 포함) |

## 코드 구조

| 경로 | 내용 |
|---|---|
| `notes_vlm/notes.py` | 이벤트 감지(`Scribe`), 변화 영역 crop, VLM 노트 작성(`ChangeNoteWriter`), `gone_note`, `measured_color`, `NoteMemory`, `VisualMemory` |
| `notes_vlm/build_ds.py` | demo에 노트가 붙은 지시문을 써서 LeRobot 데이터셋 생성 (`V`=2..8) |
| `notes_vlm/build_vis.py` | 시각 메모리(추가 카메라) 데이터셋 생성 |
| `lrbridge/server.py` | LeRobot 정책 서버. 온라인 노트(`NOTES=1 NOTES_V=..`), 시각 메모리, 노트 교체 통제(`NOTE_SWAP`) |
| `seqpol/evaluate.py` | 평가 클라이언트: MIKASA 환경 rollout, CPU/GPU physics, 선택 정확도와 지연 길이 기록, `DELAY` 변형 |
| `tools/train_eval*.sh`, `train45.sh` | 학습과 평가 |
| `tools/queue*.sh`, `evalq17.sh`, `controls.sh`, `build_cmp.sh` | 실험 큐(실행 순서와 조건이 기록돼 있음) |
| `tools/make_tables.py`, `delay_bins.py` | 결과표와 지연 분석 |
| `logs/eval_db.log` | 모든 평가 결과 요약(`모델 step 과제 sr=`) |
| `logs/*.json` | 평가 상세(에피소드별 성공, 길이, 큐브별 최소 거리, 지연 길이) |

## 환경 (서버 기준)

- 학습·정책: `/data/workspaces/mikasa/envs/lr` (LeRobot, `lerobot/pi05_base`)
- 시뮬레이터·평가: `/data/workspaces/mikasa/MIKASA-Robo/.venv` (MIKASA-Robo 커밋 `509b875`)
- 데이터: `/data/workspaces/mikasa/data_mikasa_robo/` (원본 demo와 노트 데이터셋)
- 체크포인트: `runs/lr/<run>/checkpoints/<step>/pretrained_model` (용량 때문에 레포 미포함)

## 재현 예시

```bash
# 1) 노트 데이터셋 (v6)을 과제별로 만들고 병합
cd notes_vlm && python build_ds.py 52 <원본 demo 폴더> <출력 폴더> 6
python tools/merge_ds.py <병합 출력> <t50_v6> <t51_v6> <t52_v6>
# 2) 학습 20k + 평가 (CPU physics, C420, deadband 0.02)
NOTES_V=6 DEADBAND=0.02 tools/train_eval.sh pi05_t5x_v6 <병합 출력> 20000 50,51,52 1 020000
# 3) 결과표 갱신
python3 tools/make_tables.py
```

평가 규약: 과제당 50 에피소드(seed 4242424242부터 고정), MP 수집 과제는 CPU physics, 마지막 checkpoint 보고(best checkpoint 선택 없음).

- 다른 서버에서 v7/v8 학습과 짧은 RC 실험을 돌리는 방법: [docs/RUN_ELSEWHERE.md](docs/RUN_ELSEWHERE.md)
