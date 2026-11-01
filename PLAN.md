# Plan

## Thesis
Imitation-trained memory policies under-use memory. Most timesteps are reactive, so the gradient that would teach
the policy to keep a cue is tiny, and copycat shortcuts are cheaper to learn. We make memory dependence explicit in a
flow-matching policy:
1. **Paired null-memory flow.** Each sample is trained twice, once with memory and once with a learned null memory.
   - The memory gain dL = L_null - L_mem is a per-sample estimate of the action information that memory adds
     (a pointwise conditional MI proxy).
2. **Gain-weighted flow matching.** Up-weight the timesteps where memory matters.
3. **Memory guidance at inference.** v = v_null + w(v_mem - v_null), i.e. CFG on memory.
4. **Streaming memory decoupled from chunked execution.** Memory is updated on every frame while actions run in
   chunks. This addresses the collapse under chunking reported for recurrent memory.
5. **Analysis.**
   - dL localises memory-critical moments.
   - Counterfactual cue swaps check that memory is used causally.
   - Results broken down by delay and by memory type.

## Phases
- **P1 dev (10 tasks, one multitask model per method):** 0,3,12,14,15,31,35,47,79,85.
  - Methods: none / stack8 / gru / tf, each with plain, paired, and guidance sweeps.
- **P2:** multitask per split (Short 38, Medium 30, Long 22), canonical 50 episodes.
- **P3:** baselines (PTP-style aux loss, GMP-like gate) and a pi0.5 fine-tune reference if feasible.
- **P4:** analysis figures, then a real robot.

## Naming
Neutral names only (no "vla"). Tasks are referred to as tXX ids.
