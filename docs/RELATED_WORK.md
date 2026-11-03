# 관련 연구 정리 (논문 Related Work용)

자세한 검증 기록은 [../notes_lit_3.md](../notes_lit_3.md)에 있습니다. 표·그림 번호와 프로토콜까지 확인한 기록이고, 앞선 조사는 notes_lit.md, notes_lit_2.md에 있습니다. 아래 수치는 원문 표·그림에서 확인한 것만 적었습니다. "확인 필요" 표시가 있는 항목은 인용 전에 원문을 다시 확인하세요.

## 1. 가장 가까운 연구

### RePlanMem-VLA (arXiv 2610.22556)
*When Plans Change, Remember: Replanning-Guided Visual Memory for VLA Models.* MIKASA 그룹의 연구입니다.

- **방식:** 겹치는 행동 chunk 예측이 어긋날 때(계획 불일치) 그 시점의 전체 프레임을 keyframe 메모리로 씁니다.
- **실험:** MIKASA-Robo-VLA, π0.5 기반. 쓰기 시점 검출과 메모리 보존 지표 중심이며, 과제 성공률은 보고하지 않았습니다.

우리와의 비교:

| 구분 | RePlanMem-VLA | 우리 |
|---|---|---|
| 쓰기 신호 | 행동 계획 불일치 | 관측 변화(로봇 정지 중 픽셀 급변) |
| 저장 내용 | 전체 프레임 keyframe | 사라진 물체의 텍스트 노트 |
| 정책 입력 | 시각 context | 지시문 문자열 |
| 평가 | 검출 지표 | 과제 성공률과 오류 분해 |

- 같은 조건 비교: 우리 `visfull`(이벤트 때 전체 프레임 keyframe을 추가 카메라로 입력)이 이 방식에 해당합니다. 쓰기 신호는 우리 감지기를 씁니다.
- 주의: RePlanMem도 메모리 증강 정책을 SFT로 학습합니다. "단순 저장 알고리즘"이라고 쓰면 안 됩니다.

### RoboMME (arXiv 2603.04639, PMLR v306)
*Benchmarking and Understanding Memory for Robotic Generalist Policies.*

- π0.5 위에 메모리 변형 14개를 만들어 비교했습니다(Table 3, AVG).
  - 언어 subgoal 메모리: oracle 84.08% vs VLM 생성 32.70%
  - 최고 비-oracle: FrameSamp+Modul 44.51%(시각)
- 결론: "effectiveness of memory representations is highly task-dependent"
- 프로토콜: 16 tasks, 100 demo/task, 50 eval/task, 마지막 3 checkpoint × 3 seed
- 우리와의 관계:
  - 같은 간극(VLM이 만든 언어 메모리 ≪ oracle)을 보입니다. 우리는 그 간극을 선택·표현·활용으로 나눠 원인을 분해하고 줄입니다.
  - MIKASA RememberColor 수치는 없습니다.

### μVLA (arXiv 2606.12497)
*On Recurrent Memory for Partially Observable Manipulation in VLA Models.*

- OpenVLA-OFT backbone 안에 recurrent memory token을 둡니다.
- MIKASA-Robo-VLA 23개 과제 결과(Table 1, 100 ep): RC3/5/9 = 0.92/0.93/0.41 (m=64, K=2). 학습은 RC5 포함 5과제 혼합, RC3·RC9는 held-out. 같은 표의 π0.5는 0.07/0.12/0.05 (11/4 원문 확인).
- 우리와 달리 메모리가 잠재 공간에 있습니다. 우리 메모리는 사람이 읽을 수 있는 텍스트라서 노트 교체나 oracle 대체로 단계별 오류를 분리할 수 있습니다.

## 2. 메모리 기반 정책 (표준 MIKASA-Robo RememberColor 3/5/9, %)

| 방법 | RC3 / RC5 / RC9 | 출처 | 프로토콜 |
|---|---|---|---|
| MemoryVLA | 44 / 30 / 20 | 2508.19236 Table 4 | 7B, 5과제 joint, 100 ep, best checkpoint |
| MemoryVLA++ | 50 / 19 / 16 | 2606.09827 Table III | 위와 같음 |
| GMP | 80 / 61 / 17 | 2604.18933 Fig. 11 (그림 판독) | 과제별 정책, VLA 아님, best checkpoint |
| Chameleon | 94.4 / 72.2 / 53.3 | 2603.24576 Table S7 | VLA 아님(~60M), 30 ep × 3 seed |
| SimpleMemVLA | 71 / 58 / 59 | 2609.05533 Table 3 | Qwen3.5-4B, 시간 정보가 있는 시각 이력 |

이 수치들은 표준(짧은) `RememberColor{3,5,9}-v0`이고 우리 Long 변형과 다릅니다. 논문에서는 참고 표로만 쓰고 직접 우열은 주장하지 않습니다.

## 3. 벤치마크

**MIKASA-Robo-VLA (arXiv 2610.00604):**
- 90개 언어 조건 조작 과제, 10가지 메모리 유형, 공식 split short 38 / medium 30 / long 22.
- 기준선 π0.5를 14개 과제로 fine-tune해 평균 0.211 ± 0.044(초록 기준).
- 공식 평가는 과제당 50 에피소드, 지정 seed 50개, `pd_ee_delta_pose`, `success_once`.
- 문서: https://mikasarobo.github.io/evaluation_protocol.html

**MIKASA-Robo (ICLR 2026):** 원래의 RL/IL 메모리 벤치마크(표준 RememberColor 등).

## 4. 그 밖의 후보 (인용 전 원문 확인 필요)

- **KEMO** (2606.23589), **EventVLA** (2606.20092): 이벤트/keyframe 기반 쓰기
- **RoboMME-Interference** (2606.22338): 방해 세션이 늘면 지각 메모리는 크게 떨어지고 VLM 텍스트 메모리는 상대적으로 유지된다고 보고. 우리 간섭 실험(C)과 연결됩니다.
- **ReMem-VLA:** MIKASA 결과 없음

## 5. 우리가 메울 빈자리

- 텍스트·이벤트 노트로 **MIKASA Long 과제 성공률**을 보고한 연구는 찾지 못했습니다(2026-10-29 조사).
- 선택·표현·활용을 **같은 정책에서 분리**해 측정한 연구도 찾지 못했습니다.
