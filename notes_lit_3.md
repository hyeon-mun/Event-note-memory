# 문헌 검증 노트 3: MIKASA-Robo RememberColor 3/5/9 수치 확인

작성일: 2026-10-29. 출처는 arXiv abs/HTML 페이지(WebFetch)와 GMP Figure 11 원본 이미지 판독이다. 수치는 paper에 적힌 그대로 옮겼다.

## 0. 요약 표 (RememberColor3 / 5 / 9, success %)

| Method | 리뷰 노트 수치 | 실제 확인 수치 | 원 출처 (1차) | 판정 |
|---|---|---|---|---|
| MemoryVLA | 44/30/20 | 44/30/20 | MemoryVLA (2508.19236) Table 4 | verified |
| MemoryVLA++ | 50/19/16 | 50/19/16 | MemoryVLA++ (2606.09827) Table III | verified |
| GMP | 80/61/17 | 80/61/17 | GMP (2604.18933v2) Figure 11 (막대그래프. 표는 없음) | verified (그림에서 판독) |
| Chameleon | 94.4/72.2/53.3 | 94.4±1.3 / 72.2±1.7 / 53.3±1.6 | Chameleon (2603.24576) Table S7 | verified |
| SimpleMemVLA | 71/58/59 | 71/58/59 | SimpleMemVLA (2609.05533 v1·v2) Table 3 | verified |

**다섯 개 수치는 모두 일치한다.** 다만 프로토콜이 서로 달라 직접 비교할 수 없다는 점이 핵심 주의사항이다(아래 참조).

- 모두 **standard MIKASA-Robo**(`RememberColor{3,5,9}-v0`, 짧은 에피소드, 250 demo/task, 128×128)의 5-task 세트(ShellGameTouch, InterceptMedium, RC3/5/9)를 쓴다. MIKASA-Robo-VLA의 `-VLA-v0`나 `-Long` variant가 아니다.
- **GMP는 per-task 정책**(task마다 따로 학습하고 horizon을 튜닝, checkpoint 중 best를 보고)이고 non-VLA 소형 모델이다. MemoryVLA, MemoryVLA++, Chameleon은 5-task **joint/mixed** 정책 하나를 쓴다. SimpleMemVLA도 "one model per suite"이다. 그래서 SimpleMemVLA 스스로 GMP를 "per-task non-VLA reference"로 따로 분류한다.
- 사실상 모든 논문이 **best-checkpoint 선택** 방식으로 보고한다(MemoryVLA/++: 1k step마다 검증해서 best를 보고, GMP: 20 epoch마다 checkpoint를 저장해서 best를 보고). held-out test 방식이 아니다.

---

## 1. MemoryVLA
- **Title:** MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation (ICLR 2026)
- **arXiv:** 2508.19236
- **수치:** Table 4. Caption은 "Performance comparison on Mikasa-Robo with Franka robot. Success rates (%) are reported. CronusVLA results are reproduced by us."
  - MemoryVLA: ShellGameTouch 88, InterceptMedium 24, **RC3 44, RC5 30, RC9 20**, Avg 41.2 (+11.8)
  - Baselines: CronusVLA 32/5/31/13/9 (avg 18.0, 직접 재현), SpatialVLA 23/27/27/17/11, OpenVLA-OFT 47/14/59/16/6, π0 33/42/35/22/15
- **프로토콜:** "standard Mikasa-Robo protocol using 250 demonstrations per task at 128×128 resolution, 100 evaluation episodes per task and end-effector control". 5개 task를 20k step 동안 joint training했고 1k step마다 검증해서 best를 보고한다.
- **Base:** 7B Prismatic VLM (DINOv2+SigLIP, LLaMA-7B)과 ~300M diffusion action expert(CogACT 계열).
- **주의:** SpatialVLA, OpenVLA-OFT, π0 수치는 MIKASA-Robo 원 논문(2502.10550) Table 4에서 가져온 것이다. 원 논문의 표기는 "OpenVLA (K=8)" = 0.47/0.14/0.59/0.16/0.06이다. MemoryVLA는 이 행을 "OpenVLA-OFT"로 다시 이름 붙였다. 본문의 "highest across all tasks" 주장은 표와 맞지 않는다(RC3에서는 OpenVLA 59 > 44).
- **판정:** verified

## 2. MemoryVLA++
- **Title:** MemoryVLA++: Temporal Modeling via Memory and Imagination in Vision-Language-Action Models
- **arXiv:** 2606.09827 (2026-06)
- **수치:** Table III, "Temporal robotic manipulation performance on Mikasa-Robo".
  - MemoryVLA++: SGT 97, IM 40, **RC3 50, RC5 19, RC9 16**, Avg 44.4 (+15.0)
  - 같은 표에 Octo 46/39/45/17/11 (31.6)이 있고, DP/PTP/MaIL(SGT와 RC3만 보고)도 있다.
- **프로토콜:** 250 demos/task, 100 eval episodes/task, 128×128, end-effector control, 5 task joint 20k step, best validation checkpoint
- **Base:** 7B Prismatic VLM과 latent world model, diffusion action expert
- **주의:** RC5와 RC9는 전작 MemoryVLA(30/20)보다 **낮다**. 평균 상승은 SGT와 IM 덕분이다. baseline이 재현인지 인용인지는 명시되어 있지 않다.
- **판정:** verified

## 3. GMP (Gated Memory Policy)
- **Title:** Gated Memory Policy: In-Context Memorization and Adaptation (Gao, Liu, Li, Song)
- **arXiv:** 2604.18933 (v2 확인)
- **수치:** **표는 없고 Figure 11 막대그래프뿐이다.** 원본 이미지(`figures/jpg/exp10_mikasa.jpg`)의 막대 위 라벨을 직접 판독했다.
  - GMP: SGT 98, IM 83, **RC3 80, RC5 61, RC9 17** (error bar 있음). 평균은 67.8(SimpleMemVLA·Chameleon이 인용한 값과 일치)
  - 그림 속 baseline: Octo-small 46/39/45/17/11, OpenVLA 47/14/59/16/6, SpatialVLA 23/27/27/17/11, π0 33/42/35/22/15, MemoryVLA 88/24/44/30/20
  - Caption: "out-performing prior work MemoryVLA by 26.6% on average. The baseline performance are reported in [MIKASA-Robo] and [MemoryVLA]."
- **프로토콜(App. 8.5):** 250 training episodes/task(oracle state-based PPO), 128×128, 500 epoch, 20 epoch마다 checkpoint를 저장하고 **각 checkpoint를 100 episode로 평가해 best를 보고**한다. absolute joint action으로 학습했고(baseline은 end-effector control) task별 horizon을 튜닝했다. **task별 개별 정책**이며 non-VLA diffusion policy 계열이다.
- **주의:** 저자 스스로 "250 episodes is insufficient for RememberColor, especially RC9, and 1000 episodes would significantly improve"라고 적었다. 그래서 RC9 17은 MemoryVLA(20)보다 낮다. 텍스트에 per-task 수치가 없으므로 인용할 때는 "Fig. 11"로 표기해야 한다.
- **판정:** verified (그림 판독)

## 4. Chameleon
- **Title:** Chameleon: Control-Indexed Prospective Memory for Visuomotor Manipulation
- **arXiv:** 2603.24576
- **수치:** Table S7(App. D), "Per-task results on public benchmarks. Success rate (%)"
  - Chameleon(5-task mixed): IM 58.9±2.5, **RC3 94.4±1.3, RC5 72.2±1.7, RC9 53.3±1.6**, SG 96.7±1.4, Avg 75.1±1.4
  - 비교 열 "VLA baseline"은 24.0/44.0/30.0/20.0/88.0 (avg 41.2)로 MemoryVLA 수치와 같지만, 표에 method 이름이 붙어 있지 않다. "DP-family" 열은 전부 "–"이고 평균만 67.8(GMP 값)이다. 라벨이 어긋나 있다.
  - Table 2(main): MIKASA 5-task mixed 75.1±1.4. 2-task specialist 설정(VPWEM 설정)에서는 95.6±1.0.
- **프로토콜:** 250 demos/task, **30 rollouts/task, 3 seeds**(Table S2), 2 views 128², 25-d proprio, 8-d action, horizon 8, 10 Hz. 모델은 ~60M(trainable ~66.7M)의 non-VLA visuomotor policy다. baseline은 "taken from the cited sources"(재현 없음). MIKASA에 대한 success 정의는 따로 명시되어 있지 않고 ± 값의 정의도 명시되어 있지 않다.
- **주의:** 평가 episode 수가 30×3 seeds로 다른 논문(100 episode)과 다르다. "RC3 94.4"는 30 rollout의 평균이다. 모델 크기 표기도 일관되지 않다(Table S5는 100–500M).
- **판정:** verified

## 5. SimpleMemVLA
- **Title:** SimpleMemVLA: A Simple but Effective Native-Video Memory for Vision-Language-Action Models
- **arXiv:** 2609.05533 (v1: 2026-09-02, v2도 존재. 두 버전의 Table 3은 동일하다)
- **수치:** Table 3, "MIKASA-Robo five-task success rates (%). GMP is a per-task non-VLA reference. Best VLA per column in bold."
  - SimpleMemVLA: ShellGame 99, Intercept 83, **RC-3 71, RC-5 58, RC-9 59**, Avg 74.0
  - 같은 표: MemoryVLA 44/30/20, MemoryVLA++ 50/19/16, GMP 80/61/17(reference), Octo 45/17/11, OpenVLA-OFT 59/16/6, π0 35/22/15 등
  - **Chameleon은 표에 없다**(언급되지 않음).
- **프로토콜:** App. A에 따르면 "MIKASA-Robo ... VLA baselines follow the MemoryVLA++ report", "GMP numbers come from its Figure 11 with per-task training and tuned horizons". backbone은 **Qwen3.5-4B**(native video, plaintext timestamps)이고 suite당 모델 하나를 쓴다. MIKASA 설정(Table 8)은 overhead history cam과 wrist current cam, 3 s window, 20 fps, 60 frames, action horizon 16, 128 H100으로 16 h 학습이다. **평가 episode 수, demo 수, success 정의는 확인한 텍스트 범위에서 찾지 못했다.** "closed-loop evaluation on held-out seeds"라고만 적혀 있다.
- **주의:** 본문의 "margin over the best prior VLA widens from 12 to 28 to 39 points"에서 RC3의 best prior VLA는 MemoryVLA++(50)이 아니라 OpenVLA-OFT(59)이므로 margin 12(71−59)는 맞다. RC5는 58−30=28, RC9는 59−20=39로 일치한다.
- **판정:** verified (프로토콜 일부는 확인하지 못함)

---

## 6. ReMem-VLA
- **Title:** ReMem-VLA: Empowering Vision-Language-Action Model with Memory via Dual-Level Recurrent Queries
- **arXiv:** 2603.12942 (실제로 존재한다)
- **내용:** frame-level과 chunk-level의 learnable recurrent query(단기·장기 memory)에 Past Observation Prediction 보조 손실을 더했다. backbone은 frozen Qwen3-VL-2B이고 diffusion action head를 붙였다.
- **벤치마크:** **MemoryBench**(확장판: Put Block Back, Rearrange Block, Reopen Drawer, Long Horizon Task)와 실제 UR5 4개 task. Table 1에서 ReMem-VLA avg 94.5, π0.5 6, OpenVLA-OFT 0, MemoryVLA 0(모두 동일 프로토콜로 재현, 100 demo/task, 100 rollout/task). 실로봇은 82.5% vs π0.5 11% vs MemoryVLA 8%.
- **MIKASA-Robo / RememberColor:** **보고하지 않는다**(not found).

## 7. RePlanMem-VLA
- **Title:** When Plans Change, Remember: Replanning-Guided Visual Memory for Vision-Language-Action Models (Smelova, …, Cherepanov, Kachaev, Panov, Kovalev)
- **arXiv:** 2610.22556
- **내용:** 겹치는 action chunk 사이의 예측 불일치(replanning disagreement)를 event 신호로 써서 bounded visual memory에 write한다. **event-triggered memory**에 해당한다. 35개 disagreement 척도를 비교했고, from-scratch 정책과 pretrained OpenVLA-OFT/π0.5에서 검증했다.
- **MIKASA 사용:** MIKASA-Robo-**VLA** task를 쓴다. RQ1은 8 task(RememberColor3-VLA-v0 포함, task당 250 demo, from scratch, 50 episode), RQ2는 19 task(OFT/π0.5 fine-tune), RQ3는 RememberColor3-VLA-v0만 사용한다.
- **수치:** **task success rate는 보고하지 않는다.** keyframe 검출 ROC-AUC 0.981–0.997(예: OFT, EXEC4, XGBoost ROC-AUC 0.997), "decision point까지 task-relevant observation 보존 100% of episodes", OFT memory write 336→169, eviction 186→19(50 ep) 정도만 있다.
- **판정:** RememberColor success 수치는 not found. 방법론적으로는 우리 event-triggered memory와 가장 가까운 선행 연구다.

## 8. RoboMME
- **Title:** RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies (Dai, …, Finn, Fazeli, Chai; PMLR v306)
- **arXiv:** 2603.04639
- **MIKASA/RememberColor:** related work와 Table 2(벤치마크 비교: 12 tasks, 1,250 demos, ~72 avg steps)에만 등장한다. **성능 수치는 없다**(본문 범위 확인. appendix 전체는 확인하지 못함).
- **Language memory vs oracle(Table 3, AVG):** π0.5 위에 14개 memory variant를 만들었다.
  - Symbolic(subgoal text): **GroundSG+Oracle 84.08%** vs **GroundSG+QwenVL 32.70%**로 약 51 pt 차이가 난다. 저자는 이를 subgoal 생성 오류와, StopCube·InsertPeg처럼 조작이 많은 task에서 언어 지도가 갖는 한계 탓으로 본다.
  - Best non-oracle: **FrameSamp+Modul 44.51%**(perceptual), TokenDrop+Modul 38.04%, best recurrent TTT+Expert 22.35%
  - 결론: "effectiveness of memory representations is highly task-dependent". 언어 memory는 subgoal이 정확할 때만 강하다. 즉 oracle text는 상한이 매우 높지만 VLM이 생성한 text는 perceptual memory보다 못하다.
- **프로토콜:** 16 tasks, 100 demo/task, 50 eval episodes/task, max 1,300 steps, 마지막 3 checkpoint × 3 seeds = 9 run 평균
- **관련:** RoboMME-Interference(2606.22338)에서는 perceptual memory가 distractor session이 늘면 급격히 떨어지는 반면 subgoal(VLM-written text) memory는 gain을 더 잘 유지한다. 언어 memory의 robustness를 뒷받침하는 근거다.

## 9. 그 외 2026년 MIKASA RememberColor 보고 논문

### μVLA (arXiv 2606.12497): partially verified
- **Title:** μVLA: On Recurrent Memory for Partially Observable Manipulation in VLA Models (Cherepanov et al., CogAI/MIRAI)
- **Base:** OpenVLA-OFT(LoRA r=32)에 in-backbone recurrent memory를 더했다.
- **Table 1:** MIKASA-Robo-VLA 23개 env, 100 deterministic episodes, mean. 학습 task는 RememberColor5(그리고 ShellGamePush, InterceptMedium, TakeItBack, RememberShapeAndColor3x3)이다. RC3와 RC9는 held-out이다.
  - m=64, K=2 설정: **RC5 0.93**(train), **RC3 0.92**, **RC9 0.41**(held-out), oracle(+1st obs) 각각 0.96/0.91/0.47
  - π0.5: RC3/5/9 = 0.07/0.12/0.05. OpenVLA-OFT: 0.00/0.04/0.03. OFT† episodic: 0.19/0.09/0.11
  - 주의: 추출된 HTML의 열 헤더가 깨져 있어서 K=2 열 외의 μVLA variant 라벨은 확정하지 못했다. demo 수와 success 정의는 App. E에 있다고 하나 확인하지 못했다. standard MIKASA-Robo가 아니라 **-VLA variant**라 위 5개 논문과 직접 비교할 수 없다.

### MIKASA-Robo-VLA (arXiv 2610.00604): 벤치마크 자체
- π0.5 reference(memory 없음, 14 task joint fine-tune, task당 20 episode, success_once): **RememberColor5-VLA-v0 0.250**, **RememberColor5-Long-VLA-v0 0.300**. RC3과 RC9는 fine-tune 세트에 없다. 5-cube의 chance level은 0.200이다. 14 task 평균은 0.211±0.044.
- horizon band: Short 25–200, Medium 250–600, Long 700–2,160 step. split은 "-Long" 접미사가 아니라 band 기준이다(RC5-Long은 Medium으로 분류된다).
- language/text memory 실험은 없다.

### MEM (arXiv 2603.03596, Physical Intelligence)
- π0.6에 short-term video encoder와 **language memory**(자연어 요약을 high-level policy가 갱신)를 더했다. 이 방식이 subtask instruction concat보다 낫다고 보고한다. **MIKASA 수치는 없다**(실로봇, 10 rollout). 언어 memory 선행 연구로는 중요하다.

### 찾지 못한 것
- MIKASA-Robo RememberColor에서 **language/text memory**를 쓰는 VLA의 성공률을 보고한 2026년 논문은 찾지 못했다(검색 2회 + 개별 확인).
- event-triggered memory로 RememberColor **success**를 보고한 논문도 찾지 못했다(RePlanMem-VLA는 검출 지표만 보고).

---

## 10. 원 MIKASA-Robo 논문(2502.10550v3) Table 4: baseline 수치의 1차 출처
"Performance of VLA models on selected memory-intensive tasks", 100 eval episodes, mean±sem, 250 demos, 128×128 base+wrist, end-effector control.

| Model | SGT | IM | RC3 | RC5 | RC9 |
|---|---|---|---|---|---|
| Octo-small | 0.46 | 0.39 | 0.45 | 0.17 | 0.11 |
| OpenVLA (K=4) | 0.12 | 0.06 | 0.21 | 0.09 | 0.08 |
| OpenVLA (K=8) | 0.47 | 0.14 | 0.59 | 0.16 | 0.06 |
| SpatialVLA (K=4) | 0.23 | 0.27 | 0.27 | 0.17 | 0.11 |
| π0 (K=4) | 0.33 | 0.42 | 0.35 | 0.22 | 0.15 |

후속 논문들의 "OpenVLA-OFT 47/14/59/16/6"은 이 표의 OpenVLA (K=8) 행이다.

## 11. 우리 논문 인용 시 권고
1. 수치는 모두 맞다. 다만 표에 넣을 때는 "joint VLA(MemoryVLA, MemoryVLA++, SimpleMemVLA)", "per-task non-VLA(GMP)", "mixed non-VLA 60M(Chameleon, 30 ep×3 seeds)"를 구분해서 표기할 것.
2. Chameleon 94.4/72.2/53.3이 standard RC 최고치이지만 평가 episode 수가 다르고 non-VLA다. RC9는 SimpleMemVLA 59가 최고치다.
3. GMP는 Figure 11 출처라고 명시할 것(표 없음).
4. MIKASA-Robo-VLA(`-VLA-v0`/Long) 수치(μVLA, π0.5 reference)는 별도 표로 분리할 것.
5. 대부분 best-checkpoint 방식으로 보고되었다는 점을 caveat으로 각주에 넣을 것.
