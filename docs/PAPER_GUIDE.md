# 논문 작성 가이드

이 문서만 보고 원고를 쓸 수 있도록 정리했습니다. 숫자는 모두 이 레포의 로그에서 나온 것이고, 표마다 출처 파일을 적었습니다.

- 최신 결과표는 `python3 tools/make_tables.py`를 실행하면 `docs/RESULTS_AUTO.md`로 다시 만들어집니다.
- 기준일은 2026-11-01이며, 진행 중인 실험은 [EXPERIMENTS.md](EXPERIMENTS.md)에 상태를 적었습니다.

---

## 0. 투고 정보

| 항목 | 내용 |
|---|---|
| 학술지 | IJCAS (International Journal of Control, Automation, and Systems), 특별호 |
| 제출 목표 | 2026-11-10 |
| 원고 종류 | Regular Paper, 6~10쪽 (2단 양식) |
| 양식 | `paper/IJCAS.cls` (공식 2026-02-19판). `paper/main.tex`가 뼈대 |
| 초록 | 300단어 이하. 수식, 인용, 각주 금지 |
| 키워드 | 4개 이상, 알파벳순 |
| 참고문헌 | 번호순 [1] 형식. 권·호·쪽·연도 포함 |
| 그림 | 컬러 300dpi 이상. 흑백 인쇄에서도 구분 가능해야 함 |
| 필수 Declarations | Conflict of Interest, Authors' Contributions, Funding, Data Availability |
| AI 도구 사용 선언 | 참고문헌 바로 앞에 필수. 사용한 도구와 역할을 사실대로 적을 것 |

공지 원문: https://www.ijcas.org/content/authors/information_for_authors.html

## 1. 한 문장 요약과 논문의 입장

> π0.5 구조를 바꾸지 않고, 화면에서 **사라진 것만** 짧은 텍스트 노트로 적어 지시문에 붙이면 장기 기억 과제가 풀린다. 남은 실패는 노트 품질이 아니라 정책이 노트를 행동에 연결하는 단계에서 생긴다.

**입장:**
- 이 논문은 새 메모리 아키텍처나 SOTA를 주장하지 않습니다.
- "선택적 언어 메모리가 언제 효과적이고 어디서 실패하는지"를 같은 조건의 비교와 ablation으로 보이는 **실험 중심 논문**입니다.

**주장하지 말 것:**
- 텍스트 메모리가 모든 기억 과제에 최선이라는 주장
- 언어 메모리가 시각 메모리보다 일반적으로 낫다는 주장. 우리 시각 메모리는 구현 하나의 결과입니다.
- MIKASA 전체 SOTA
- 외부 논문 수치와의 직접 우열. 평가 조건이 다릅니다.
- 측정 색이 범용 표현이라는 주장. 측정 색은 색 과제에 특화된 요소입니다.

## 2. 제목 후보

1. **Event-Driven Language Memory for Long-Horizon Vision-Language-Action Policies** (추천)
2. What to Remember and How to Use It: An Empirical Study of Event-Driven Memory for Robotic VLA Policies
3. Selective Semantic Memory for Long-Horizon Robotic Manipulation with Vision-Language-Action Models

## 3. 가설과 근거

| 가설 | 내용 | 근거 | 상태 |
|---|---|---|---|
| H1 선택 | 사라진 정보만 남기는 것이 모든 변화를 적는 것보다 낫다 | v3→v4 오류 분해, 테스트 때 노트 교체(결과 R3), 주기적 캡션 비교 | 일부 확인, 주기적 캡션은 진행 중 |
| H2 표현 | 정규화된 색 이름이 비슷한 색 혼동을 줄인다 | v4→v6에서 쌍 혼동 13→5 (R2) | 확인 |
| H3 활용 | 노트가 정확해도 정책 활용이 별도 병목이다 | v6 실패 17건 모두 노트 색은 맞음, oracle과의 간극, 시각 메모리 무시, v7과 노트 교체 통제 실험 | 일부 확인, v7과 통제는 진행 중 |
| H4 범위 | 단일 단서 기억에는 효과적이고, 다중 단서·추적·횟수·시간에는 한계가 있다 | Long 26개 결과 (R5) | 중간 결과로 경향 확인 |

## 4. 기여 (영문 원고용)

1. **Event-driven semantic memory.** An event-driven pipeline that selectively preserves task-relevant information disappearing from view, without modifying the VLA architecture.
2. **Selection and representation.** Controlled evidence that writing only what disappeared and naming attributes from measured pixels change downstream decisions; redundant or inconsistent notes degrade the policy.
3. **Utilization gap.** Generated notes, oracle notes, content-swapped notes, and visual memory together separate memory correctness from action success.
4. **Long-horizon evaluation.** 26 long-horizon memory tasks with one joint policy per method, characterizing where lightweight language memory helps (single-cue recall of colour and shape) and where it does not (multiple cues, tracking, counting, timing).

## 5. 구성, 그림·표와 데이터 출처

| 절 | 내용 | 그림·표 | 데이터 출처 |
|---|---|---|---|
| I. Introduction | 부분 관측, 사라진 단서, 현재 VLA의 한계(base Long 26개 평균 0.048), 기여 | – | R5 |
| II. Related Work | 메모리 VLA, 이벤트 기반 선택(RePlanMem), 언어 메모리(RoboMME), 벤치마크 | – | [RELATED_WORK.md](RELATED_WORK.md) |
| III. Method | 이벤트 감지 → crop → VLM → Gone 선택 → 측정 색 → 노트 메모리 | **Fig. 1** `docs/figures/pipeline.svg`. **Fig. 2** 노트 예시 `docs/figures/rc9_example_*.png` | [METHOD.md](METHOD.md) |
| IV. Setup | 과제 목록(env_id), CPU physics 이유, 학습·평가 프로토콜, 비교 방법 | **Table 1** 설정표 | [EXPERIMENTS.md](EXPERIMENTS.md) §1 |
| V-A. Main | RC-Long: base / v2~v6 / oracle, 신뢰구간 | **Table 2** | R1 |
| V-B. Selection | v3 vs v4, 노트 교체, 주기적 캡션 | **Table 3**, **Fig. 3** 오류 분해 | R2, R3, C2 |
| V-C. Representation | v4 vs v6 쌍 혼동, v5 어휘 교훈 | Fig. 3 | R2 |
| V-D. Utilization | 노트 내용 통제, v7, 시각 메모리, oracle 간극 | Table 3 | R3, C1, C3 |
| V-E. Memory types | Long 26개의 메모리 유형별 결과, 지연 시간 분석 | **Table 4** | R4, R5 |
| VI. Conclusion | 효과, 선택과 표현의 중요성, 활용 병목, 한계와 향후 연구 | – | – |

R1~R5, C1~C3은 [RESULTS.md](RESULTS.md)의 항목 번호입니다.

## 6. 예상 리뷰어 질문과 대응

| 질문 | 대응 |
|---|---|
| 알고리즘 novelty가 약하다 | 조합의 새로움이 아니라 선택·표현·활용 각각의 효과와, 정확한 메모리가 곧 좋은 행동은 아니라는 발견을 통제된 실험으로 보인다. |
| 정말 메모리 덕분인가 | 같은 모델에 테스트 노트만 바꿔도 성능이 바뀐다(R3). 노트 비우기, 틀린 색, 정답 색 통제(C1). |
| 그냥 과거 프레임을 주면 되지 않나 | 같은 조건의 시각 keyframe 메모리(RePlanMem 방식)와 crop 시각 메모리 비교(C3, R1의 vis). |
| v6가 v2보다 낫지 않다 | 숨기지 않는다. 쌍 혼동은 줄었지만 다른 오류가 남은 것을 표현과 grounding 사이의 간극으로 분석하고, v7로 원인을 검증한다. |
| 일반적인 메모리 과제를 푸는가 | 아니다. 단일 단서에서만 효과가 있고, 다중 단서·추적·횟수·시간은 한계로 명시한다(R5). |
| 비교가 공정한가 | 주 결과는 같은 데이터, 학습 step, LR 스케줄, 평가 seed의 내부 비교다. 외부 수치는 참고 표로 분리한다. |
| 통계적 신뢰도 | 과제당 50 에피소드(RC9는 150)와 Wilson 95% CI. best checkpoint를 고르지 않고 마지막 checkpoint를 보고한다(다른 논문보다 보수적). seed 반복은 하지 않았다. MemoryVLA, GMP 등도 seed 반복 없이 best checkpoint를 보고한다(Chameleon, RoboMME는 3 seed). |
| 선택 정확도와 성공률이 다르다 | "맞는 큐브"(가장 가까이 간 큐브가 정답)와 성공(터치 인정)을 별도 지표로 보고한다. |
| 26개 과제 구성이 공식 split과 다르다 | 이름에 -Long이 붙은 변형 전체다. 공식 split으로는 long 14개 + medium 12개이며, 공식 long 중 44-46, 76-78, 88-89는 제외됐다. env_id를 표로 명시한다. |
| 왜 CPU physics인가 | MP로 수집된 demo는 CPU physics다. demo 행동을 재생하면 CPU에서는 8/8 성공, GPU에서는 0/8 성공(2.5~4.5cm 오차)이다. 벤치마크 관련 발견으로 함께 보고한다. |

## 7. 원고 작성 상태

| 부분 | 상태 |
|---|---|
| 제목, 초록 초안, 키워드, Method | `paper/main.tex`에 초안 있음 |
| 나머지 절 | 주석으로 내용만 잡혀 있음 |
| 수치 | 확정되지 않은 수치는 `\tbd{}`(빨간색)로 표시. 실험 마감(11/5) 후 RESULTS_AUTO.md 기준으로 채움 |
| 참고문헌 | arXiv 번호만 있음. IJCAS 형식으로 완성 필요 |
