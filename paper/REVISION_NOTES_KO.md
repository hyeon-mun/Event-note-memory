# 원고 수정 및 제출 전 점검 — 2026-11-05

## 수정 방향

핵심 기여를 **현재 관측만 쓰는 VLA에 외부 언어 노트를 연결하고, 노트 교체로 실패 양상을 진단하는 실증 연구**로 정리했다. 새 메모리 아키텍처나 전체 메모리 과제의 해결을 주장하지 않는다. 기존 실험 수치를 유지하면서 주장의 범위를 근거에 맞췄다.

`main.tex`는 IJCAS 클래스다. 저장소의 연구 요약에는 IEEE 6쪽 목표도 적혀 있어 최종 투고처와 분량은 아직 일치하지 않는다. 현재 클래스는 보존했으며, 채택 가능성을 판단하려면 최종 투고처와 논문 유형을 확정해야 한다.

## 반영한 주요 수정

- 초록·도입·기여점·결론을 다시 정리하고, RC9에서 v6가 이전 버전보다 성공률을 개선했다는 인상을 제거했다. 오류 구성 변화와 전체 성공률 개선은 다른 결과다.
- `notes_vlm/notes.py`와 대조해 이벤트 감지의 **중앙값 + 1**, 짧은 초기 이력, 현재 proprio 변화 포함 여부, settling 조건, persistent mask, dilation, 최소 영역 크기를 바로잡았다.
- 색의 “채도”를 실제 코드의 RGB 채널 범위로 정의하고, 이름 매칭 거리와 percentile 동률 처리도 명시했다.
- “사라진 물체”가 객체 집합 차분으로 검출되는 것이 아니라 **변화 영역의 before caption을 쓰는 휴리스틱**임을 밝혔다.
- 체크포인트에서 확인한 optimizer/scheduler 값, dead band, YUV420 처리, 평가 seed, 성공 지표를 추가했다.
- 26개 과제의 45k 결과가 **중간 체크포인트 비교**이며 no-memory/v6의 최종 90k 결과는 각각 0.048/0.116임을 분명히 했다. “모든 결과가 마지막 체크포인트”라는 모순을 제거했다.
- 시각 메모리의 실패를 “정책이 입력을 무시한다”로 단정하지 않고, 3×3 타일과 축소된 입력 인터페이스의 한계를 적었다. RePlanMem-VLA 재현이 아니라 영감을 받은 별도 baseline으로 표기했다.
- v4→v6는 측정 색 이름과 첫 객체만 남기는 처리가 함께 바뀌므로 순수한 색 이름 ablation이 아님을 명시했다.
- 틀린 노트 주입 표에서 `pair`와 `injected-note target`이 혼용된 분류를 고쳤다. 본문에 19/14/17의 실제 의미를 유지했다.
- 성공률을 단순한 1/N 선택 확률과 동일시한 chance 표현, oracle을 엄밀한 상한으로 보는 표현, 서로 다른 프로토콜의 외부 연구 순위 주장을 수정했다.
- 참고문헌의 저자·제목을 arXiv 원문 메타데이터와 대조하고, RoboMME의 PMLR 쪽수를 채웠다. 사용자 신상·펀딩·공개 저장소 등 확인할 수 없는 정보는 placeholder로 남겼다. AI 사용 항목은 이번 편집·검산에 실제 사용한 Codex의 역할로 채웠으며 저자 확인이 필요하다.

## 기존 평가 로그에서 추가 계산한 통계

`python3 paper/audit_results.py`를 실행하면 `RESULTS_AUDIT.md`를 재생성한다. 원시 JSON의 성공률을 검산하고, RC9의 50개+100개 확장 평가에 seed 중복이 없는지 확인한다. 같은 seed끼리 대응시킨 20,000회 percentile bootstrap으로 아래 구간을 계산했다.

| 비교 | 성공률 차이 | paired bootstrap 95% CI |
|---|---:|---:|
| RC3-L v6 − no memory, n=50 | +0.820 | [0.700, 0.920] |
| RC5-L v6 − no memory, n=50 | +0.800 | [0.680, 0.920] |
| RC9-L v6 − no memory, n=150 | +0.413 | [0.327, 0.500] |
| 고정 v4 정책에 v3 writer 적용, n=50 | −0.240 | [−0.400, −0.060] |
| 고정 v3 정책에 v4 writer 적용, n=50 | +0.080 | [−0.080, 0.240] |

첫 번째 writer 교체 효과는 구간이 0을 포함하지 않지만, 반대 방향의 +8%p 개선은 포함한다. 두 방향을 똑같이 확정적인 개선·악화로 쓰면 안 된다. 모든 구간은 현재 학습된 정책에 조건부이며 **학습 seed 변동성을 대신하지 않는다**. 여러 비교에 대한 보정은 하지 않았다.

## 제출 전에 필요한 일 — 우선순위

1. **학습 seed 반복.** 최소한 RC-Long의 no-memory/v6/oracle을 반복하고 seed별 성공률과 평균·변동성을 보고해야 한다. 현재 에피소드 수가 많다는 이유로 학습 안정성을 주장할 수 없다. 이번 원고 수정에서는 GPU 학습을 새로 실행하지 않았다.
2. **이벤트 방식 자체의 이득을 분리할 baseline.** Long 과제에서 first-frame caption과 주기적 caption을 같은 예산으로 비교해야 한다. 현재 비교는 노트의 가치와 writer 구성에 대한 근거이며, 이벤트 trigger가 대안보다 우수하다는 검증은 부족하다.
3. **정답 cue가 보존되는 비율과 쓰기 비용.** 이벤트 precision/recall, 결정 시점 cue-note 보존율, 노트 수·토큰 수, captioner 호출 수, 지연시간과 메모리 비용을 보고한다. 정책 구조가 그대로라는 사실은 계산 비용이 같다는 뜻이 아니다.
4. **표현 ablation 분리.** v4 + 첫 객체만 / v4 + 측정 색만 / 둘 다를 비교하면 색 정규화의 독립 효과를 주장할 수 있다. 현재는 결합 효과로만 해석했다.
5. **oracle 통제 명확화.** 별도 oracle 정책은 문장 형식과 cue 공급 시점이 event notes와 다르다. 같은 `Gone:` 형식·시점의 oracle 학습 및 평가가 있으면 utilization 격차의 해석이 더 강해진다. 현재 oracle labeler는 평가 코드에서 고정 cue 영역과 색 cluster를 사용하므로, “완벽한 정답 노트”의 정확성도 원본 cue label과 대조하는 것이 좋다.
6. **시각 baseline 해석.** 추가 카메라의 42px 타일 압축이 성능 병목일 수 있다. 동일한 해상도·보존량의 시각 history와 비교하기 전까지 언어가 시각 메모리보다 본질적으로 낫다고 주장하지 않는다.
7. **짧은 과제 expert의 특권 정보 확인.** 저장소 연구 요약에는 PPO expert가 후보 등장 전부터 정답 쪽으로 움직인다는 관찰이 있다. 시점별 action/target 상관을 확인하고, 확인되면 짧은 과제는 순수 메모리 검증이 아니라 보조 실험임을 명시해야 한다.
8. **재현성 마무리.** LeRobot/benchmark commit, 실제 학습 seed, training config와 scheduler 실행 경로, oracle 생성 방식, camera/proprio/action 정의, 코드·로그 공개 범위 및 외부 표 수치의 원문 위치를 보충한다.
9. **형식·메타데이터.** 목표 학회/저널에 맞춰 템플릿과 분량을 확정한다. 저자·소속·교신저자·펀딩·기여도·데이터 공개·AI 사용 고지를 실제 정보로 채운다. 익명 심사라면 해당 규정도 적용한다.

현재 수정은 원고의 논리·구현 일치·통계 해석을 개선한 것이며, 위 실험의 완료나 채택 수준을 보장하지 않는다. 특히 1~3이 주요 심사 쟁점이다.

## 확인한 1차 출처

- π0.5: https://arxiv.org/abs/2504.16054
- SmolVLM: https://arxiv.org/abs/2504.05299
- MIKASA-Robo: https://proceedings.iclr.cc/paper_files/paper/2026/hash/1088fffae5f2aa8c8324e4f45f62248d-Abstract-Conference.html
- MIKASA-Robo-VLA: https://arxiv.org/abs/2610.00604
- RePlanMem-VLA: https://arxiv.org/abs/2610.22556
- μVLA: https://arxiv.org/abs/2606.12497
- RoboMME: https://proceedings.mlr.press/v306/dai26c.html
- MemoryVLA: https://arxiv.org/abs/2508.19236
- MemoryVLA++: https://arxiv.org/abs/2606.09827
- SimpleMemVLA: https://arxiv.org/abs/2609.05533
- GMP: https://arxiv.org/abs/2604.18933
- Chameleon: https://arxiv.org/abs/2603.24576

외부 방법의 성공률 표는 저장소의 `docs/RELATED_WORK.md`·문헌 조사 기록을 바탕으로 유지했다. 이번 검증의 외부 접근은 주로 서지 메타데이터이며, 모든 논문의 표·구현을 독립 재현한 것은 아니다.

## 빌드 및 검산 결과

- 수정 PDF: `draft_v0.2.pdf`, IJCAS 형식 **8쪽**. 기존 `draft_v0.1.pdf`는 보존했다.
- Tectonic 0.17.0으로 빌드 성공. 미해결 인용·참조와 overfull box는 없다. 일부 underfull spacing 및 클래스의 폰트 대체 경고는 남아 있다.
- 26과제 표를 양단 표로 넓히고 짧은 과제 표의 중복 memory 열을 줄여 가독성을 개선했다. 파이프라인 그림에서 입력→정책 화살표 방향도 수정했다.
- 재검산: `python3 paper/audit_results.py` (research 디렉터리 기준).
- 재빌드: `cd paper && tectonic main.tex`. 출력 `main.pdf`를 확인한 뒤 버전 PDF로 복사하면 된다. 일반 LaTeX 환경에서는 `pdflatex main.tex`를 두 번 실행할 수 있다.
