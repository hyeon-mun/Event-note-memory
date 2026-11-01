# Literature notes (2026-10-18)

## Suite paper: arXiv 2610.00604 (RoboPAD workshop)
- Only baseline: pi0.5 fine-tuned from pi05_base, current frame only, open-loop 8-step chunks, 14 tasks, 20 episodes each.
  Mean 0.211 (Long tasks 0.017). TakeItBack .50, InterceptMedium .40, GatherAndRecall1 .35, BunchOfColors3-Long 0, ChainOfColors3-Long 0.
- 28 tasks have a cue-absent gap longer than 16 steps. 10 tasks are reactive controls.
- The paper says chunking changes success by up to 0.65.

## Original 32-task suite (2502.10550)
- Offline RGB, 1000 demos, RATE/DT/BC/CQL/DP:
  - ShellGameTouch .92/.53/.28/.16/.18
  - RememberColor3 .65/.01/.27/.29/.32
  - TakeItBack .42/.08/.33/.04/.05
  - Every method scores 0 on Bunch/Seq/Chain.
- VLA Table 4 (columns SGT/IM/RC3/RC5/RC9):
  - Octo .46/.39/.45/.17/.11
  - OpenVLA K=8 .47/.14/.59/.16/.06
  - pi0 .33/.42/.35/.22/.15

## Memory methods on the 5-task subset (SGT/IM/RC3/RC5/RC9, average)
- MemoryVLA 2508.19236: 41.2
- MemoryVLA++ 2606.09827: 44.4. DP/PTP/MaIL with history: SGT 18-28, RC3 3-11.
- GMP 2604.18933 (gated history cross-attention DiT, noise on past actions): 98/83/80/61/17, average 67.8
- SimpleMemVLA 2609.05533 (Qwen3.5-4B, timestamped video, flow head): 99/83/71/58/59, average 74.0
- uVLA 2606.12497 (OpenVLA-OFT, 64 recurrent memory tokens, TBPTT K=2, memory masked from action tokens): 0.42 -> 0.84 on its own 5 tasks.
  - Collapses under open-loop chunks.
- ELMUR 2510.07151, TFP 2607.08283, FibVLA 2607.29596, VPWEM 2603.04910, MemBodied 2609.28256
- Baseline rows are often copied between papers.

## Related ideas
- PTP 2505.09561: diffusion policies UNDER-use history; predicts past tokens.
- Copycat / causal confusion: 1905.11979, 2010.14876.
- BPP 2602.15010: keyframes.
- MEM (PI) 2603.03596
- Streaming Flow Policy 2505.21851
- RTC 2506.07339
- MaIL 2406.08234, Mamba Policy 2409.07163

## Gaps
1. Memory robust to action chunking.
2. Per-memory-type and per-retention-gap analysis on the full 90 tasks.
3. Causal use of memory: counterfactual cue swap, generalization to held-out semantics.

## Expected baselines
- pi0.5 no-history
- frame-stack
- MemoryVLA (or a re-implementation)
- uVLA-style recurrent tokens
- GMP
- PTP
- SimpleMemVLA-style full video
