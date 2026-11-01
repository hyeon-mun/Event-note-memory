# Literature survey 2: text notes, event-triggered writes, visual notes (2026-10-26)

Scope: complements notes_lit.md. Papers already covered there (MemoryVLA, MemoryVLA++, GMP, SimpleMemVLA,
uVLA, ELMUR, PTP, BPP, MEM, MIKASA 2502.10550 / 2610.00604) are only mentioned where needed.
Verification legend: [V-abs] = arXiv abstract page fetched; [V-html] = full HTML read (parts);
[V-snip] = seen only in search snippets / secondary pages (title+id confirmed, details less sure); [?] = uncertain.

---------------------------------------------------------------------------------------------------
## 1. Language / text as memory for robot policies and embodied agents

- **RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies** — 2603.04639 [V-html]
  16-task memory benchmark + 14 memory variants on a pi0.5 backbone. "Symbolic memory" = language subgoals
  appended to the task instruction (from Gemini-2.5-Pro, fine-tuned Qwen3-VL-4B, or GT; "SimpleSG" vs "GroundSG"
  with image coords). Perceptual = TokenDrop / FrameSamp; recurrent = TTT / RMT.
  Avg success: symbolic 11.6-32.7% (best GroundSG+QwenVL 32.7), **oracle GT subgoals 84.1%**, perceptual
  30.7-44.5% (best FrameSamp+Modulator 44.5), recurrent 18-22%. Finding: representation is task-dependent; language
  wins on counting / event-salient tasks, perceptual wins on motion/time tasks; VLM grounding errors hurt symbolic.
  -> CLOSEST prior evidence for our story: text-in-instruction memory on pi0.5, large VLM-vs-oracle gap.
  Differences: their text is written every step / per subgoal (not event-triggered), they do not isolate *attribute*
  loss (colour), and they don't test visual notes in the same slot. Not MIKASA.

- **Simple Agentic Memory for Generalist Robot Policies (SimpleARM)** — 2609.36595 [V-html]
  Typed episode state (entity-relation, progress counts, trajectory) written by frozen tools (Grounding DINO,
  DINOv2 identity anchor, optical flow, SAM2) under an evidence gate; recalled entities re-grounded in the live
  view before a frozen pi0.5 acts. RoboMME: 67.2% vs no-mem 32.7, MemER 42.4, FrameSamp 44.5, oracle 84.1.
  -> Strong recent "structured explicit memory" baseline; uses visual identity anchors (DINOv2), i.e. sidesteps
  naming colours. Close to our "visual note" idea in spirit. Not MIKASA.

- **MEM: Multi-Scale Embodied Memory (PI)** — 2603.03596 (known). Running language summary for long-range
  semantic memory + compressed video for short range. Our text notes are a minimal, event-triggered version of
  its language memory; MEM does not study what the summary loses.

- **MemER: Scaling Up Memory for Robot Control via Experience Retrieval** — 2510.20328 (ICLR 2026) [V-abs]
  High-level Qwen2.5-VL-7B selects/tracks past keyframes and emits text subtask instructions to a pi0.5
  low-level policy; ~50 demos with subtask annotations. Real-robot tasks only. On RoboMME 42.4% (per SimpleARM).
  -> Hierarchical VLM->text->pi0.5 interface like ours, but memory is keyframes inside the VLM; the policy
  receives only language (so it also has a language bottleneck at the interface).

- **Explicit Language Memory for Long-Horizon Planning in VLA Models** — 2608.04765 [V-abs] **WITHDRAWN**
  (uploaded without co-authors' awareness). High-level VLM recursively updates a textual memory + subtask
  for a low-level VLA. No numbers. Cite with caution or not at all.

- **Hi Robot** — 2502.19417 [V-abs]. Hierarchical VLM interprets instructions/feedback, emits language
  commands to a pi0-family low-level policy. Background for "language channel into VLA".
- **Statler: State-Maintaining Language Models for Embodied Reasoning** — 2306.17840 [V-abs]. LLM maintains
  a textual world-state estimate updated after each action; planning-level (code-as-policies), no visuomotor policy.
- **REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction** — 2306.15724 [V-abs].
  Event-based hierarchical text summary of multisensory robot history (keyframes chosen at subgoal/audio/state
  changes) for LLM failure reasoning. Early "captions-as-memory + event selection" precedent.
- **Inner Monologue** — 2207.05608 (2022, outside window; well-known, not re-fetched). Textual feedback loop for LLM planners.
- **ReMEmbR** — 2409.13682 [V-abs]. Navigation: VLM captions of video chunks stored in a vector DB for
  retrieval-augmented QA. Captions-as-memory in robotics (QA, not control).
- **EchoVLA** — 2511.18112 [V-snip]. Scene + episodic memory banks for mobile manipulation (latent, not text).

## 2. Event-triggered / keyframe / change-based memory writes

- **When Plans Change, Remember: Replanning-Guided Visual Memory for VLAs (RePlanMem-VLA)** — 2610.22556
  (posted ~Oct 2026, MIKASA authors: Kachaev, Cherepanov, Panov, Kovalev ...) [V-html, partial]
  Write trigger = disagreement between overlapping action chunks (policy "re-plans" -> event); 35 disagreement
  measures compared; bounded FIFO of **full RGB frames** appended as extra images [top, wrist, M1..Mm];
  robot-masked write filter / clustering to drop redundant writes. Evaluated on **MIKASA-Robo-VLA** (detection
  over 19 tasks incl. RememberColor5/9; retention on RememberColor3-VLA-v0) with OpenVLA-OFT and pi0.5.
  Numbers: detector ROC-AUC 0.981-0.997; cue retained until decision in 100% (50/50 OFT episodes); raw surprise
  gating at 95th pct wrote the cue in 0% (OFT) / <20% (pi0.5) of episodes, 48%/85% at 80th pct; writes 336->169,
  evictions 186->19 with masking. **No task success rates found in the text read**, no text-memory baseline,
  no pixel-difference baseline (top-view L2 is only a detector input feature).
  -> MOST DIRECT COMPETITOR for "when to write". Our pixel-change+stillness trigger is simpler and
  policy-agnostic; theirs is policy-internal. We must compare or at least discuss; we can add the "what to write"
  axis (text vs crop vs oracle) they don't study, and report end-task success on RememberColor-Long.
- **KEMO: Event-Driven Keyframe Memory for Long-Horizon Manipulation with VLA Policies** — 2606.23589 [V-abs]
  Events from robot kinematics + visual filtering; keyframes -> memory tokens fused by cross-attention / gated
  residual into pi0.5; event-weighted loss. Real dual-arm tasks: +23.6% task success, +34.1% stage completion vs
  memory-free pi0.5; event-driven > uniform / recent-frame. Not MIKASA. -> Kinematics-gated event trigger
  ~ our "robot stillness" gating; requires architectural change + training, ours writes into existing inputs.
- **EventVLA: Event-Driven Visual Evidence Memory for Long-Horizon VLA Policies** — 2606.20092 [V-abs]
  Learned keyframe-probability head (foresight) stores sparse task-critical frames + initial/short-term anchors.
  New RoboTwin-MeM benchmark (17 sim + 4 real): +40% avg over memory-augmented VLAs. Not MIKASA.
- **AURA: Action-Gated Memory for Robot Policies at Constant VRAM** — 2606.02775 [V-snip]
  Learned surprise (fast-weight prediction error) write gate with write budget ~0.15; LIBERO-Long 0.233 with 7x
  fewer writes (n=60). Single author; weak evidence.
- **EM-LLM: Human-inspired Episodic Memory for Infinite Context LLMs** — 2407.09450 [V-abs]
  Surprise-based event segmentation of token streams; cognitive-science motivation for event-triggered writes.
- **REFLECT** (above) and **MemER** (above) also select keyframes, but via subgoal events / a learned VLM.
- **RoboMemArena / PrediMem** — 2605.10921 [V-abs]. 26-task memory benchmark with keyframe annotations; dual-
  system VLM planner with recent + keyframe buffers. No numbers in abstract.

## 3. Visual memory via selected frames / crops / object-centric memory

- **SAM2Act / SAM2Act+** — 2501.18564 [V-abs]. SAM2-style memory bank of past visual features for 3D
  manipulation; MemoryBench avg 94.3%. Feature-level, not crops.
- **Embodied-SlotSSM: Rethinking Progression of Memory State ... Object-Centric Perspective** — 2511.11478 [V-snip].
  Slot-based (object-centric) SSM memory. Closest "object-centric memory" for manipulation; latent slots.
- **SimpleARM** (sec.1): DINOv2 identity anchors on detected boxes = object-centric visual keys.
- **RePlanMem / EventVLA / KEMO / BPP** store whole frames (or tokens of whole frames).
- **IMPRINT: Memory Without Curation: Multi-Stream Latent Memory for Robot Policies** — 2610.14406 [V-abs].
  Argues *against* curating what to remember; latent memory over obs/action/proprio streams, >=26% over base VLA.
  Useful counter-position for our "explicit curated note" design.
- **Remember Smarter: Visual History Compressor ...** — 2608.15269 [V-snip], **NativeMEM** — 2607.06678 [V-snip],
  **MemBodied** — 2609.28256 [V-html; RMBench 50.0% vs pi0 6.4%, NativeMEM 38.4%; no MIKASA]: latent compression.
- Gap: I found **no paper that stores zoomed crops of the changed region (pre-change content) tiled into an
  extra camera image** for a VLA. Searches for crop/zoom/tiling memory returned nothing; claim novelty cautiously.

## 4. Evidence that language bottlenecks lose fine-grained attributes

- **RoboMME** 2603.04639: VLM-written subgoal memory 32.7% vs GT-oracle subgoals 84.1% on same pi0.5 -> the
  writer (VLM perception/grounding) is the bottleneck, not the language channel per se. Mirrors our 9-colour gap
  (VLM 0.38-0.50 vs oracle 0.80).
- **MemEye: Visual-Centric Evaluation of Multimodal Agent Memory** — 2605.15128 [V-abs; detail V-snip]. 13 memory
  methods x 4 VLMs; caption-based storage loses instance/pixel-level evidence; text-vs-multimodal gap grows with
  evidence granularity.
- **MemLens** — 2605.14906 [V-snip]. Largest gaps on visually grounded questions; captions keep gist, drop
  counts/attributes; "bottleneck is lossy cross-modality storage".
- **RAVEN** — 2606.25206 [V-snip]. Names the "semantic bottleneck" of caption memory in navigation.
- **ColorBench: Can VLMs See and Understand the Colorful World?** — 2504.10514 [V-abs]. 32 VLMs; colour
  understanding "largely neglected", small gaps across models, scale and the LLM part matter; CoT helps.
  Supports "small VLM (SmolVLM2) colour naming is unreliable".
- **Rainbow** (EACL 2024 colour-naming probe, Bexte et al.) [V-snip, no arXiv id confirmed]: notes uncertainty at
  boundaries between neighbouring colour terms — exactly maroon/red, teal/cyan type errors.
- **Human-AI Perceptual Alignment by Playing Hues and Cues** — 2608.07141 [V-snip]. Colour-vocabulary misalignment.
- **Remember the Decision, Not the Description (DeMem)** — 2605.10870 [V-abs]. LLM agents: memory should preserve
  decision-relevant distinctions, not faithful descriptions (rate-distortion). Good framing: "red" vs "maroon" is
  a decision-relevant distinction a generic caption doesn't preserve.

## 5. 2026 work evaluating on MIKASA-Robo (not in the known list)

- **Chameleon: Control-Indexed Prospective Memory for Visuomotor Manipulation** — 2603.24576 [V-html]
  ~60M-param policy, event tokens + slow SSM traces + learned control index, Control-JEPA. MIKASA (mixed 5-task,
  250 demos/task, 30 rollouts x 3 seeds), success %: RememberColor3 **94.4**, RC5 **72.2**, RC9 **53.3**,
  ShellGame 96.7, InterceptMedium 58.9, avg 75.1 (vs MemoryVLA 41.2, GMP 67.8, pi0 29.4, OFT 28.4).
  2-task specialist: 95.6 vs DP-VPWEM 86.5, DP 19.5. NOTE: these are the original (non -Long, non -VLA)
  RememberColor tasks from the MemoryVLA protocol -> not directly comparable to our -Long numbers; still the
  strongest published RC9 (53.3) I found. Our text notes on RC9-Long (0.38-0.50) are below; oracle (0.80) above.
- **VPWEM: Non-Markovian Visuomotor Policy with Working and Episodic Memory** — 2603.04910 [V-html]
  Diffusion policy + sliding window + transformer compressor to episodic memory tokens. MIKASA ShellGameTouch-v0
  and RememberColor3-v0; ">20% over VLA baselines" (Octo, OpenVLA, pi0, SpatialVLA, CronusVLA, MemoryVLA, BPP,
  MemER); per-task numbers only in a figure (not extracted). Per Chameleon: DP-VPWEM 86.5 avg on the 2 tasks.
- **RePlanMem-VLA** — 2610.22556 (sec.2): MIKASA-Robo-VLA, detection/retention metrics only (as read).
- **MIKASA-Robo-VLA** 2610.00604 (known) reference pi0.5, 20 eps: RememberColor5-VLA 0.25, RememberColor5-Long-VLA
  0.30 (chance 0.20); RC3/RC9 not evaluated; no memory baselines; mean 0.211 over 14 tasks.
- [?] "EmbodiedMemory-Bench": secondary summary claims OpenVLA 59% RC3 -> 6% RC9; could not locate on arXiv. Do not cite.
- Checked, NOT on MIKASA: MemBodied 2609.28256, SimpleARM 2609.36595, KEMO, EventVLA, Counterfactual Memory
  Audit 2609.27247 (Mem-0 tasks), FibVLA 2607.29596 (abstract has no benchmark names).

## 6. Hierarchical / IL+RL hybrids for memory tasks (brief)

- Hierarchical VLM->VLA with memory at the high level: MemER 2510.20328, Hi Robot 2502.19417, MEM 2603.03596,
  RoboMemArena/PrediMem 2605.10921, SimpleARM 2609.36595 (tool-written state, frozen VLA). Ours is in this family
  but with a write-only VLM (no planning) and an event trigger.
- IL+RL: original MIKASA 2502.10550 shows PPO-LSTM solves RememberColor3 (100%) while PPO-MLP ~25%, degrading
  with more colours. I found no 2026 paper doing RL fine-tuning of a VLA specifically for memory tasks / MIKASA;
  generic VLA-RL fine-tuning (e.g. HABC 2606.17043 [V-snip]) does not address memory. RATE (recurrent action
  transformer, offline RL, MIKASA authors) exists [not re-verified]. Low priority.

---------------------------------------------------------------------------------------------------
## (a) Positioning

Most 2026 memory-for-VLA work (MemoryVLA, GMP, Chameleon, KEMO, EventVLA, MemBodied, IMPRINT, uVLA) adds a
*learned latent* memory and changes the architecture. A smaller line keeps the policy unchanged and writes
*explicit* memory into its inputs: language subgoals (RoboMME symbolic, MemER, MEM summaries), typed tool-written
state (SimpleARM) or retained full frames (RePlanMem-VLA, BPP). Our pitch: **an architecture-free, event-triggered
external memory for a standard pi0.5 policy, with a controlled study of the note *format* (none / VLM text /
visual crop note / oracle text) holding the write trigger fixed**. The format study is the main novelty: RoboMME
shows VLM-vs-oracle language gaps but attributes them broadly to grounding; MemEye/MemLens show caption loss for
QA, not control. We can show, on a controlled attribute-scaling axis (3/5/9 colours), that (i) the event trigger +
a correct note nearly solves the task (oracle 0.96/0.98/0.80), (ii) the lossy part is the VLM's colour vocabulary
(specific confusions maroon/red, orange/yellow, teal/cyan), and (iii) whether visual crop notes close that gap.
Novel (as far as found): crop-based "visual notes" of the pre-change region injected as an extra camera image;
the text-vs-visual-vs-oracle comparison on the same write events; MIKASA RememberColor-Long end-task numbers for an
explicit-memory method. NOT novel: event/keyframe-triggered memory writes per se (KEMO, EventVLA, RePlanMem-VLA,
REFLECT, AURA); language memory in the instruction (RoboMME, MemER, MEM); "captions lose details" in general
(MemEye, MemLens). Phrase the trigger as a simple, policy-agnostic instance of event-driven writing, and ideally
show it matches policy-internal triggers (RePlanMem) at much lower complexity. Beware comparing our -Long success
to Chameleon/GMP (different task variants, demos, eval protocol) without stating the difference.

## (b) Baselines / related work reviewers will expect

1. **RePlanMem-VLA (2610.22556)** — same benchmark, same group as MIKASA, event-triggered visual (full-frame)
   memory for pi0.5. Reviewers (possibly these authors) will ask "why pixel change vs plan-disagreement trigger,
   and why crops vs full frames?". Ideal: full-frame-memory ablation of our pipeline (store whole pre-change frame
   instead of crop) as a proxy if their code isn't available.
2. **RoboMME (2603.04639)** — the controlled symbolic-vs-perceptual memory study on pi0.5 incl. oracle subgoals;
   our format study must be framed relative to it. Its FrameSamp baseline (uniform frame sampling into context)
   is a cheap must-have comparison ("uniform-sampled memory frame" vs event-triggered).
3. **KEMO (2606.23589) / EventVLA (2606.20092)** — event-driven keyframe memory for VLAs; justify our trigger and
   show event-triggered > uniform/recent baselines as they do.
4. **MemoryVLA (2508.19236) / GMP (2604.18933) / Chameleon (2603.24576)** — the MIKASA RememberColor numbers
   everyone reports; include a table with the protocol caveat (non-Long tasks, 250 demos).
5. **MEM (2603.03596) / MemER (2510.20328)** — language-summary / VLM-keyframe hierarchical memory with pi0.5;
   closest precedents for "VLM writes text that conditions a pi0.5 policy".
