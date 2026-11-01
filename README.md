# Event-driven text notes as memory for a π0.5 policy (MIKASA-Robo long-horizon tasks)

π0.5(LeRobot) 정책 구조는 그대로 두고, "화면에서 사라진 것"만 짧은 텍스트 노트로 적어 지시문 뒤에 붙이는 메모리 연구입니다.
연구 전체 요약은 [RESEARCH_SUMMARY_1101.md](RESEARCH_SUMMARY_1101.md), 날짜별 실험 기록은 [LOG.md](LOG.md), 문헌 정리는 notes_lit*.md에 있습니다.

## 폴더

| 경로 | 내용 |
|---|---|
| `seqpol/evaluate.py` | 평가 (정책 서버를 띄우고 MIKASA 환경에서 rollout, CPU/GPU physics 선택, 선택 정확도 기록) |
| `lrbridge/server.py` | LeRobot 정책 서버. 온라인 노트 생성(NOTES=1, NOTES_V=2..6), 시각 메모리(VMEM) |
| `notes_vlm/notes.py` | 이벤트 감지(Scribe), 변화 영역 crop, VLM 노트 작성, Gone 노트, 측정 색, NoteMemory, VisualMemory |
| `notes_vlm/build_ds.py` | demo 데이터셋에 노트가 붙은 지시문을 써서 LeRobot 데이터셋 생성 |
| `notes_vlm/build_vis.py` | 시각 메모리(추가 카메라) 데이터셋 생성 |
| `tools/` | 학습·평가 큐 스크립트 (`train_eval.sh`, `train_eval_long.sh`, `mid_eval_long.sh`, `queue*.sh`) |
| `paper/` | IJCAS LaTeX 원고 (`main.tex`, `IJCAS.cls`) |
| `logs/eval_db.log` | 모든 평가 결과 한 줄씩 (`모델 step 과제 sr=`) |
| `logs/*.json` | 평가 상세 (에피소드별 성공, 길이, 다가간 큐브 거리) |

## 환경 (서버 기준 경로)

- 정책/학습: `/data/workspaces/mikasa/envs/lr` (LeRobot, π0.5)
- 시뮬레이터/평가: `/data/workspaces/mikasa/MIKASA-Robo/.venv`
- 데이터: `/data/workspaces/mikasa/data_mikasa_robo/` (원본 LeRobot demo, `notes/`에 노트 데이터셋)
- 체크포인트: `runs/lr/<run>/checkpoints/<step>/pretrained_model` (용량 때문에 레포에서 제외)

## 재현 예시

```bash
# 노트 데이터셋 (v6) 생성 후 3과제 병합
cd notes_vlm && python build_ds.py 52 <src_dir> <dst_dir> 6
python tools/merge_ds.py <t5x_v6> <t50_v6> <t51_v6> <t52_v6>
# 학습 20k + 평가 (CPU physics, C420, deadband 0.02)
NOTES_V=6 DEADBAND=0.02 tools/train_eval.sh pi05_t5x_v6 <t5x_v6> 20000 50,51,52 1 020000
```

평가 규약: 과제당 50 에피소드(seed 4242424242부터 고정), MP 수집 과제는 CPU physics, 마지막 checkpoint 보고(best checkpoint 선택 없음).
