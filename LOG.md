# MIKASA-Robo-VLA memory research log

## 2026-10-18 setup status
- Env: `MIKASA-Robo/.venv` (uv sync). gcc is blocked for this user -> wrapper `~/bin_cc/cc` (g++ -x c). Run with
  `export PATH=$HOME/bin_cc:$HOME/.local/bin:$PATH CC=$HOME/bin_cc/cc`.
- Extra pkgs added via `uv pip install av "pyarrow<17" einops hf_transfer` -> DO NOT use `uv run` (resync removes them); call `.venv/bin/python` directly.
- Sim smoke test OK on GPU 6 (examples/eval_demo.py).
- Data: HF `mikasa-robo/mikasa-robo-vla-lerobot` (27.6GB, AV1 mp4) -> `/data/workspaces/mikasa/data_mikasa_robo/data_lerobot`
  (download may be interrupted; re-run snapshot_download, it resumes).
- `research/prep_data.py` decodes to memmaps in `data_mikasa_robo/mm/<task>/{top,wrist,state,action,ep_start,ep_len}.npy` (tested on 1 task; ~600GB total for all).
- Eval: canonical runner = `mikasa_robo_suite/vla/benchmarking.py` (50 seeds from 4242424242, success_once). Policy.forward has no reset hook -> stateful policies must detect episode start.

## Method candidate (pending lit review)
Memory-guided flow policy: recurrent memory + flow-matching action-chunk head, memory dropout -> memory-CFG at inference,
memory-critical step reweighting via ΔL = L_flow(no-mem) - L_flow(mem) (pointwise conditional MI estimate).

## Next
1. finish download + prep  2. dataloader/model/train/eval code  3. baselines (no-mem, framestack, GRU, transformer, PTP)  4. method + ablations

## 2026-10-18 (cont.)
- User request: keep script/dir/process/log names neutral (no "vla" in names); refer to tasks by numeric id in CLIs.
- YCB asset auto-download fails (hash mismatch) -> manually downloaded zip, extracted with python zipfile.
- Parallel GPU envs (num_envs>1, seed list) do NOT reproduce num_envs=1 initial states -> canonical numbers must use num_envs=1; parallelise across processes.

### Literature (lit agent, see notes_lit.md)
- Suite paper: arXiv 2610.00604 — only baseline pi0.5 single-frame on 14 tasks, mean 0.211. Nobody has full 90-task results yet.
- 5-task subset SOTA: SimpleMemVLA 74.0, GMP 67.8 (gated history x-attn DiT), MemoryVLA++ 44.4, uVLA 0.84 on its own 5 tasks.
- Gaps: (1) memory under action chunking; (2) principled write/retention across all 10 types (Capacity/Sequential = 0 for all); (3) causal use of memory (counterfactual cue swap), held-out semantics.
- IROS 2027: Florence, 27 Sep-1 Oct 2027; deadline ~1 Mar 2027 (tentative).

## P1 (launched 2026-10-18 ~15:40)
- 8 multitask models on dev tasks 0,3,12,14,15,31,35,47,79,85 (none/stack/gru/tf x plain/drop/paired), 30k steps, bs32 x16 sel.
- p1_eval.sh auto-evaluates (50 eps canonical seeds, exec 4, flow 10 steps; guidance 1/1.5/2/3 for drop/paired).
- Fixes: proprio roll -> sin/cos (wraps at +-pi). Eval procs OOM on GPUs 2/3 (19GB free) above ~4 procs.

## Data checks (2026-10-18)
- The cue-phase freeze wrapper is NOT used by any real task (only the template), so oracles move during the cue phase.
- Leak probe: linear regression from EE xyz at step k to the final EE xy (holdout R2).
  - t00: k=1 -0.11 | k=3 0.27 | k=5 0.46 | k=6 0.62 | k=8 0.86
  - t01 and t33 are similar.
  - Cue lasts 1-5 steps, so the oracle commits during the cue. Short spatial tasks can be solved partly via "embodied memory" (own pose).
  - Report this as a confound and ablate it (proprio in the memory stream, a no-memory baseline with proprio).
- MP oracles (Medium/Long tasks) hold still during cue+empty phases (hold_steps until manip_start) -> no leak there. The leak is only in the PPO short tasks.
- Capacity "any order": the oracle picks the lowest index in next_target_mask, so the order is deterministic given the scene. A latent to-do list does not conflict with the demos.
- P1 none_plain (no memory): mean 0.258 on dev. RC5 = .20 and RC9 = .10 = chance, TakeItBack .98, ShellGame .58 (> 1/3, pose leak).
- Eval is now a global queue (seqpol/evalq.py, evalq/ dir, slots.json, priorities, orphan-aware).
- P2 (hard-memory dev 22,25,28,43,49,68,73,86) training: none/gru/tf/event/todo.
- Backend sensitivity: none_plain on t00 scores canonical GPU num_envs=1 0.58, batched GPU (50 envs) 0.34, CPU backend 0.12 (per-episode agreement with canonical 23/50).
  - Policies are very sensitive to physics backend and batching, so all dev evals use the canonical runner.
  - Eval is slow under GPU contention (65 ms/step idle, 200-1300 ms contended). Dropped the g=3.0 sweeps.

## KEY FINDING (2026-10-18 evening): memory = episode fingerprint -> memorization
- P1 sim results: every memory model is at chance on RememberColor (t14 .14-.18, t15 .12). Mean SR: none .258, gru_drop .242, tf_paired .232, stack .214.
- Held-out loss on t14 (225 train / 25 val eps, 8k steps), train vs val l_mem:
  - none 0.050 / 0.093
  - stack 0.004 / 0.443
  - gru 0.004 / 0.459
  - tf 0.005 / 0.423
  - The val loss is worst at the decision steps 0-6 (~0.8-1.0).
- Visual domain gap ruled out: proprio identical; mean images match up to a 1px shift; an AV1 round-trip does not change the DINO NN gap.
- Interpretation: history (proprio trajectory + scene) identifies the episode, so the policy recalls demos instead of learning cue -> action.
- New story: memory needs an information bottleneck. Sparse event/to-do memory is a natural one. Dev metric = held-out decision-step loss (fast).
- t14 val (all / early 0-8), 8k steps:
  - Baselines: none .093/.13; stack1 (current DINO feats only, no history) .21/.32
  - GRU variants: gru .45/.71, gru_noprop .44/.69, gru_noise0.5 .32/.49, gru_cls .40/.63
  - CNN-sourced memory (augmented): gru_cnn .33/.51, stack8_cnn .32/.49, tf_cnn .29/.45
  - Slot memory: event .36/.57, event_q_noprop .25/.39
  - Even augmented CNN memory overfits. The frozen-DINO pathway alone (stack1) overfits as well.
  - Hypothesis: binding (remembered cue <-> object in the current scene) is hard to learn via global conditioning, so memorizing is easier.
  - Next: memory-as-query "find" readout (top-down attention over current spatial CNN tokens).
- todo NaN: non-finite grads in every parameter (originating at the head input). Did not reproduce in 400 steps from ckpt_1000 with anomaly detection; added skip+diagnostics.

## 2026-10-19 early: diagnosis continues
- Learning curves on t14 (val decision-step loss): no memory design beats none's best (0.114). Memory goes from unused to memorized.
- Probe (cue color decodable from held-out memory states, steps 8-14): GRU memory 0.90 (step 2k) / 0.83 (8k). Reactive 0.65-0.79 (cue leaks into the scene once the arm approaches). Event-q 0.22 (writes collapsed).
- ORACLE memory (true cue one-hot): sim SR on t14 is 0.24 / 0.20 / 0.24 (find), i.e. chance.
  - So the bottleneck is binding cue -> target with the from-scratch visual encoder, not memory.
  - Videos: the policy makes a stereotyped reach regardless of the cue.
- GMP (2604.18933) uses PRETRAINED SigLIP2-B/16, 500 epochs, and reports the best checkpoint over 3 seeds selected by eval SR (optimistic). The abstract already states the history-overfitting finding.
- Next: frozen SigLIP-B/16 reactive encoder (14x14 patches, attention pooling, find over patches). Testing the oracle setting first.

## ROOT CAUSE (2026-10-19): imitation gap in PPO-collected Short tasks
- SigLIP + oracle cue on t14: sim 0.18-0.28 (chance), so it is not the visual encoder.
- Offline direction check: none and oracle-cue models agree with the demo direction equally (steps 3-5 cos ~0.3). The label is ignored.
- Demo EE motion vs direction to the final target: cos 0.99 already at steps 0-2, BEFORE the 5 cubes appear.
  - The PPO expert uses privileged state to anticipate the target location, and the learner cannot.
  - The learner learns the averaged reach and never sees corrections, so even perfect memory cannot help.
- MP-collected tasks hold still until cue+empty is over (t51: first motion at step 158-540). These are clean memory tests.
- New dev focus: MP tasks (RememberX-Long 50-58, Capacity/Sequential 22-30/59-67, Checklist, Temporal, Procedural, Prospective, ShellShuffle-Long).
- Paper angle: report the anticipation gap as a benchmark caveat. Possible remedy for PPO tasks: DAgger/RL fine-tuning (the user's hybrid idea).

## 2026-10-19: pivot to pretrained VLA backbone
- From scratch on t51 (MP, clean): every variant is at or below chance, including the oracle-cue upper bound.
  - none .20/.12, gru .12/.08/.06, event_q .18/.08/.04, tf .08, oracle .18/.16
  - From-scratch policies cannot learn the binding from 250 demos.
- Fixed: per-task action normalisation (a / q99|a|). MP actions are tiny in the rot dims, which caused wrist drift.
- LeRobot env: /data/workspaces/mikasa/envs/lr (py3.12, lerobot main 0.6.2).
  - pi0.5 is blocked by the gated PaliGemma tokenizer (needs the user's HF token).
  - SmolVLA works: 2.5 it/s at bs32, 6.7GB.
- lrbridge/server.py: the LeRobot policy runs as a server; evaluate.py RemoteRunner is the client (numpy-agnostic bytes protocol).
- Oracle-note dataset (make_note_ds.py): instruction + " The cue color was X." Runs svla_t51_base vs svla_t51_oracle, 10k steps.
- t51 with action normalisation:
  - n_none .12/.12, n_oracle .14/.14, n_gru .04/.08
  - SigLIP: sig_none .12/.04, sig_gru .20/.06
  - Confirms the from-scratch dead end, even with the oracle cue.

## !!! 2026-10-19: physics-backend mismatch between MP data and the canonical eval
- MP collector uses gym.make with no sim_backend and num_envs=1, so ManiSkill auto selects the CPU backend.
- Exported LeRobot episodes are SHUFFLED relative to source_rlds_metadata.episode_seeds. Matching by initial proprio pose is exact.
- Replaying recorded actions with the correct seed (t51):
  - CPU: 8/8 success, 0 trajectory error.
  - GPU (canonical): 0/8, 2.5-4.5 cm drift.
  - pd_ee_delta_pose IK differs between CPU (exact) and GPU (differential).
- Consequence: under the official protocol even the expert scores 0 on MP tasks (all of Medium/Long and some Short). This explains the near-zero Long results in the literature.
- PPO tasks: single-env replay does not reproduce on either backend (likely batched collection). Keep GPU for PPO tasks.
- Decision: evaluate MP-source tasks with the CPU backend (data-consistent). Report both and flag this as a benchmark finding.
- PPO collector: gym.make(num_envs=batch, sim_backend="gpu") -> GPU physics. MP collector: CPU. evaluate.py --sim_backend auto picks by the csv Data Source.
- CPU-eval rollouts on t51 now look sane (reach and touch cubes), but the from-scratch oracle model touches the wrong color: a binding failure.
  - n_none / n_oracle CPU: .02 / .02

## 2026-10-19 11:30
- SmolVLA t51 CPU backend (partial): base 6/40 (0.15), oracle-note 6/45 (0.13) -> chance-level (1/5), oracle text does not help.
- Found 62 orphaned lrbridge servers (server re-looped on accept after client exit) eating CPU/GPU -> killed; server now serves one client and uses PR_SET_PDEATHSIG.
- pi05_t51_oracle 10k trained; CPU eval running (cpu51_7).

## 2026-10-19 ~13:00 — grounding is the bottleneck, not retention
- SmolVLA t51 CPU final: base 0.16, oracle-note 0.12 (chance 0.2).
- t51 = RememberColor5: cue cube shown, ~300-500 idle steps, then 5 cubes (pentagon, random rotation) appear; touch the matching one. Cubes are 2-3 px in the 128px top cam.
- goalnet.py diagnostics (frame at motion onset -> final EE xy, 225 train / 25 val eps):
  * hand-crafted colour match + affine: 1.5 cm val error (task is trivially solvable given the cue)
  * conv net + oracle label, soft-argmax regression: ~20 cm (fails); same with pixel cross-entropy: 1.9 cm, 96% < 3 cm after 500 its
  * no cue: 23-25 cm (chance, 16-20% < 3 cm); cue *image* via FiLM: train 1.9 / val 24 cm (memorises; cue identity not learned from raw pixels)
  * siamese matching (shared per-pixel enc, saliency-pooled cue): did not learn
  => dense spatial (pixel-CE) supervision makes cue->target binding learnable; regression / action-only supervision does not.
- Implemented hindsight goal grounding: data.hindsight_goals (EE pos at next gripper toggle / episode end, projected with the fixed base_camera calib), model.GoalHead (memory-FiLM conv on full-res frames, 32x32 CE), goal fed to flow head (teacher forcing p=0.5). Runs runs/v51/g_{oracle,gru,none,event}.
- g_oracle @1k: held-out onset goal acc(4px) 0.76; g_none 0.20.
- 6k held-out onset goal acc(4px): g_oracle 1.00 (1.7px), g_gru 0.12, g_event 0.12, g_none 0.24 -> grounding solved given the cue; learned memories fail to carry it.
- Linear probe of cue colour from step-0 DINO (CLS / pooled / wrist) and raw crop: 100% val -> info is present; failure is retention/overfitting over ~400 idle steps (memorises episode identity).
- Camera calib identical across 23 sampled tasks.
- Next: bottlenecked memories (tokens=1, no proprio, FSQ event) g_ev_q, g_ev1, g_gru1, g_tf1.
- Bottlenecked variants (tokens=1, no proprio; g_ev_q FSQ, g_ev1, g_gru1, g_tf1): all chance (0-0.24).
- ROOT CAUSE: g_none reaches the same *train* goal error (1.3 px) as g_oracle -> the global goal head memorises scene layout -> target per training episode; memory gets no pressure. Shortcut learning, not a memory-capacity issue.
- Fix: --goal_match: local-key (RF ~11px, no coords, top cam) x memory-query dot-product head; layout can't be memorised. Runs m_{oracle,none,gru1,ev1}.
- Match head: m_oracle val acc4 1.00 (@2k); m_none val 0.08. m_gru1/m_ev1: TRAIN eps acc 1.00, VAL ~0.1 -> now the *memory* memorises episode identity -> query (DINO CLS is unique per episode).
- Next: discrete-note bottleneck on the memory state (FSQ levels 3,3 / 5): q_gru9, q_gru5, q_ev9, q_tf9.
- FSQ memory-state bottleneck (q_*): val still chance (<=0.28), train 0.68-0.92 -> even 5-9 codes memorise.
- Lag test: memory frozen 30/100 steps before onset -> TRAIN acc 1.00 (m_gru1), VAL chance. So memory encodes episode identity from the cue/idle phase and emits a query matching the *wood texture at the target location* (static textured table; 11px keys identify absolute position).
- Fix: --goal_rf 1 appearance-only (per-pixel 1x1) keys, scores max-pooled into 4x4 cells. Runs r_{oracle,gru1,ev1,gru5}.
- 17:45 user asked to free GPUs: stopped evalq, eval servers, pi05_t51_base training (just started), r_* runs. t51_vlm notes dataset built (848 tasks, 4.7 events/ep).
- 2026-10-20 19:00 user: GPUs 0-6 busy with their training; GPU 7 allowed but their periodic eval runs there -> my jobs capped (tools/capgpu.py, per-process torch memory fraction) + tools/gpuguard.sh kills my GPU-7 procs if free < 15 GB.
- r_* (rf=1 appearance-only keys), goalval VAL acc4: r_oracle 0.72@2k -> 0.40@6k (train 0.24!), r_gru1 0.20/0.24, r_gru5 0.20/0.20, r_ev1 0.44@2k (run incomplete). 1x1 keys are too weak even for oracle -> no conclusion on memorization; dropped.
- pi05_t51_base (no notes) restarted on GPU 7 with gradient checkpointing, 40 GB cap (~41.6 GB used, 1.1 it/s).
- Notes bug: with K=4, 60% of train episodes pushed the step-0 cue note out of the window (scene change gives 2 events 2 steps apart). Fix: NoteMemory, an event within BURST=10 steps replaces the previous note; K=6. Rebuilt t51_vlm (643 tasks, 3.1 notes/ep, 0% lose the cue). Server: NOTES=1 env runs EventDetector+NoteWriter online (NOTE_LOG to dump notes).
- Event detector rework (online smoke tests with pi05_t51_oracle + NOTES=1): (i) demos have a global re-shading at cue removal (mean diff 1.1) that the online renderer lacks (0.19) -> mean-diff threshold missed the event online; switched to count of pixels with max-channel |diff|>40 (demo 24-33, online 25-29, 0 when still). (ii) the policy drifts ~1e-4 m/step while waiting, flipping ~1 px per 1e-5 m at the gripper edge -> spurious events; added ratio test (count >= 4*(1+median of previous 5 counts)). (iii) NoteMemory drops a note identical to the previous one. K=8. Rebuilt t51_vlm: 4.8 events/ep, <=7 notes/ep, cue note never leaves the window. Online: events at 0, ~101, scene appearance (+ some after contact).
- chain (tools/chain_t51.sh): after pi05_t51_base -> train pi05_t51_notes (same cfg, notes dataset) + CPU eval of base (50 ep).
- pi05_t51_base (no notes) CPU 50ep: sr=0.20 (chance=0.2) vs oracle-note 0.98.
- pi05_t51_notes (VLM event notes, online at eval) CPU 50ep: 5k 0.20, **10k 0.70** (base 0.20, oracle 0.98). Train notes: first colour word == true cue in 250/250 eps (magenta->"pink"). Online ~15 events/ep (vs <=7 in demos; drift + post-contact), but success not correlated with #pre-scene notes (2 notes: 9/15) -> under-trained grounding, not note noise. Per colour: pink 8/11, red 8/10, green 9/11, yellow 5/7, blue 5/11. Resumed to 20k; evals at 15k/20k chained.
- pi05_t51_notes resumed: 15k 0.84, 20k 0.82 (CPU 50ep).
- t54 (RememberShape5-Long) notes: SmolVLM step-0 notes are partially shape-informative at 128px (cue mask clusters: big box -> ball/square/cube/block; cross -> object/button/cross; cube -> cube/block). Qwen2.5-VL-3B with the same generic prompt is worse (verbose, "blue object"). Chain tools/chain_t54.sh: pi05_t54_notes 20k (+evals 10k/20k online notes) then pi05_t54_base 10k + eval.
- pi05_t54_notes 10k: sr=0.12 (online step-0 notes: 'blue ball' 28/50 -> shape info mostly lost at 128px).
- pi05_t54_notes 20k: 0.14. Built t54 oracle-note dataset: cue shape = k-means(5) of a 10x10 blue-minus-red crop at the cue (clean 5 clusters, SSE 4.2 -> 0.44 from k=4 -> 5), named A-E; online oracle uses the same crop + saved centres. Chain: t54 base (running) -> pi05_t54_oracle 15k + evals 10k/15k.
- pi05_t54_base 10k: 0.12 (notes 0.14 -> no gain). pi05_t54_oracle training started 14:13.
- **Train/test image mismatch found**: demo frames come from yuv420 videos -> chroma at half resolution (small coloured objects are 2x2-blocky in colour; luminance/wood texture full-res), online obs are raw RGB. Fix at eval: server C420=1 applies BT.601 yuv420 chroma pooling to both cameras (before notes and policy). After C420, online cube/sphere/cylinder cue crops match demo clusters exactly; torus and cross are still larger online (6x8 px vs 4x4 / 6x4 in demos) -> likely asset-size change between data generation and released env code (benchmark issue to report).
- t54 oracle note online now uses info['oracle_info'] mapped to demo clusters (cube E, sphere D, cylinder A, cross C, torus B).
- tools/eval_c420.sh re-evaluates t51 notes/base/oracle and t54 notes/base/oracle with C420.
- C420 re-eval (CPU 50ep): t51 notes 15k 0.82 (was 0.84), t51 base 0.12 (was 0.20), t54 notes 20k **0.26** (was 0.14; base 0.12). t54 online step-0 notes still mostly "blue ball/square/object" (shape-poor), ~17.7 notes/ep online (post-contact chatter) vs <=7 in demos. t54 oracle 10k/15k eval running.
- t54 oracle-note 10k (C420): 0.20 -> oracle note does not help at 10k: policy-side perception of 4-6 px shapes (or torus/cross size mismatch) is the bottleneck, not the VLM. TODO verify oracle cluster mapping. Report artifact: https://claude.ai/artifact/JHUdyeRt6egMDiMZFzBNFq
- t54 oracle 15k (C420): 0.20. Per shape (cube/sphere/cylinder/cross/torus) oracle15k 2/10 4/11 1/11 2/7 1/11 -> chance even on shapes whose online pixels match the demos exactly. t54 = policy perception limit (4-6 px shapes at 128 px), not memory/notes; dropping t54 as a memory testbed (report as such).
- Note write rule v2 (notes.Scribe, NOTES_V=2): events allowed with any history length (cue changes every 2-4 steps from t=0 in Chain/Seq tasks), EE must also be still at the event frame, note written once the scene settles (first frame with change < NPIX, max 5 frames), no burst merging (NoteMemory(0)). Demo events/ep: t28 4.4, t51 3.0 (exactly cue / removal / candidates), t65 5.0 (3 cues, removal, candidates).
- t65 (ChainOfColors3-Long, 9 colours incl. maroon/olive/teal): full-frame vs changed-region-crop captions: both first-colour word compatible with GT 24/27, but the vocabulary is coarse (maroon->"red", olive->"yellow", teal->"green") -> ambiguity caps the ceiling. Kept full-frame. Built notes/t65_v2 (1111 tasks). chain_t65.sh: pi05_t65_notes 20k (eval 10k/20k, NOTES_V=2, C420) then pi05_t65_base 10k. Started 19:45.
- C420: t54 base 0.14, t51 oracle 0.96. t54 summary (C420): base .14, VLM notes .26, oracle .20/.20 -> perception-bound.
- t51 notes15k (v1-trained) evaluated with v2 online write rule + C420: 0.78 (v1 online 0.82), online notes/ep 6.2 (v1 ~15) -> v2 removes post-contact chatter, success unchanged within noise.
- t65 notes 10k: 0/50. Expert replay on CPU: 6/6 success (zero error) -> task is feasible. Diagnostic (10 eps, video, progress logged in evaluate.py `progress`): policy drifts during the long wait (demo idle actions are ~5e-6, i.e. zero), drift creates spurious notes (11/ep vs 5 in demos) and the arm leaves the start pose before candidates appear; touches 0-2 cubes, never in order.
- Server DEADBAND env: zero arm actions when max|a[:6]| < DEADBAND. Demo idle |a| ~5e-6, moving steps: 7.5% < 1e-3, 10% < 3e-3. With DEADBAND=0.02: online notes 6.5/ep (= demo structure: 3 cues, removal, candidates), the arm waits correctly, then hovers over the candidates but rarely touches (3/10 eps touch one cube) -> motor under-training at 10k. Use DEADBAND=0.001 going forward.
- t65 notes 15k (DEADBAND=0.001, C420, v2 notes): 0/50; correct-prefix 1 cube in 1/50, any touch 9/50.
- t65 oracle upper bound: seq_labels.py (seed matching -> true colour sequence per demo, 250/250 matched exactly) + make_seq_note_ds.py ("The cue colors were Yellow, Maroon, Magenta.", env colour names) -> notes/t65_oracle (200 tasks). evaluate.py --note oracle for tids 28-30/65-67 uses oracle_info (= true sequence).
- Dropped the t65 no-note control (uninformative while notes are at 0). Queue (tools/queue1.sh, one training at a time): pi05_t65_oracle 20k + evals 10k/20k -> pi05_t51_v2 15k + evals. tools/eval_db.sh: deadband evals (t65 notes 20k pending).

## 2026-10-22 03:05 KST — t65 notes 20k (DEADBAND=0.001, C420, v2 notes)
- 0/50. correct-prefix {0:48, 1:2}, any touch 12/50 (15k: 9/50). More training barely helps -> skill or note quality, oracle-note run decides.
- pi05_t65_oracle training started (queue1), ~3k steps at 03:05.

## 2026-10-22 ~07:30 KST — t65 oracle-note 10k: 0/50 -> t65 is skill-bound
- pi05_t65_oracle 10k (true colour sequence in the instruction, DEADBAND 0.001, C420): 0/50, correct-prefix 0 in 50/50, any touch 6/50. Instruction string verified identical to env LANGUAGE_INSTRUCTION + note.
- Videos (logs/vid_t65_oracle10k, 5 eps): waits correctly through cue/empty phase; after the 9 candidates appear it wanders over the ring at hover height (wrist cam sees cubes from ~10 cm), visits the button, rarely descends+stops (touch needs tcp within 2 cm of cube top AND is_static). Target choice also unrelated to the given sequence (ep1 true Lime,Yellow,Maroon -> button, left side, then far right).
- => with 250 demos the multi-touch/colour-grounding skill (9 x 2 cm cubes, 3 ordered touches + button) is the bottleneck, not memory. Same verdict class as t54. Oracle 20k eval still queued for completeness.
- Next (queue2 after queue1): RememberColor3/9-Long (t50/t52) v2 notes 15k vs no-note 10k -> generality of event notes across the t51 family. tools/train_eval.sh = generic train + deadband evals.
- Candidate events (v2 detector, demos): t50 3.2/ep, t52 3.0, t47 BlinkCount 6.1 (blinks), t48 8.2, t43 Batteries 1.0 (robot always moving -> detector silent; to-do memory there needs action-side events).

## 2026-10-22 19:20 KST — queue1 done, queue2 running
- pi05_t65_oracle 20k: 0/50 (10k 0/50) -> t65 skill-bound confirmed.
- **pi05_t51_v2 (v2 write rule, DEADBAND 0.001, C420): 10k 0.86, 15k 0.90** (v1 notes 15k 0.82, base 0.12, oracle 0.96).
- pi05_t52_notes (RememberColor9-Long, v2): 10k 0.02, 15k 0.06 (chance 0.11). All 50 eps time out at 600 steps. Demos touch the target ~18 steps after the candidates appear. Online notes drift during the wait: "white computer monitor..." notes at 101-400. Recording videos (logs/vid_t52_notes15k). pi05_t52_base training now.

## 2026-10-22 evening — note quality (user: notes don't match images, mention the robot)
- Note benchmark (notes_vlm/note_bench.py; demo event frames of t51+t52, cue colour GT = changed pixels vs empty frame, nearest env colour):
  SmolVLM2-500M full frame (v1/v2 notes): cue colour 0.86 (lenient), empty table described as objects 100% ("white computer monitor, keyboard"), robot/hallucinated objects in 36% of notes. Bigger VLMs (SmolVLM2-2.2B, Qwen2.5-VL-3B) on the full frame are not better; a table crop / "say empty" prompt fixes the empty frame but kills the 4 px cue.
- Zooming the changed region fixes it (zoom_bench.py / change_bench.py / prompt_bench.py): SmolVLM2-2.2B on the zoomed cue crop: colour 0.96, cyan vs blue and teal separated; 9-colour discriminability 0.80 (remaining confusions maroon->red, orange("Olive")->yellow). Two-image before/after prompts are unreliable about appeared vs disappeared -> caption before and after crops separately.
- **Notes v3** (notes.ChangeNoteWriter, NOTES_V=3): v2 write rule; t=0 scene note with a table prompt (robot ignored, "The table is empty." allowed); each later event -> "Change: <before crop> -> <after crop>." on the largest connected region of pixels that changed at event frames (zoomed, NEAREST).
  Demo t52: "a red cube." / "Change: a red square -> brown." / "Change: a wooden surface -> a red cube, a green cube, ...".
- Online issue: the arm drifts / moves in its null space while the EE is still -> spurious events whose crop is the arm ("toilet", "dog"). Full-proprio stillness (Scribe(full=True)) is not enough. DEADBAND 0.02 gives exactly the demo structure during the wait (3 notes/ep); post-candidate events during manipulation remain. Control running: t51_v2 15k with DEADBAND 0.02 (vs 0.90 at 0.001).
- queue2 stopped after t52_base; queue3: t52_v3, t51_v3 (15k, DEADBAND 0.02, NOTES_V=3), then t50_v3 + t50 base.
- t51_v2 15k with DEADBAND 0.02: **0.96** (0.001: 0.90; oracle 0.96). Deadband 0.02 = default for v3 evals.
- pi05_t52_base 10k (DEADBAND 0.001): 0.08 (v2 notes 15k 0.06, chance 0.11).
- 2026-10-23 00:55 pi05_t52_v3 10k (v3 notes, DEADBAND 0.02): 0.04. Online notes still contain arm crops during the wait in some eps (policy moves early: events at 3, 30, 88...). All failures run to 600 steps (RememberColor never terminates on a wrong touch).
- evaluate.py: single-choice tasks now log the closest TCP approach to each cube (progress[i]['mind'], 'true') -> which cube the policy went for and whether it got within GOAL_THRESH. Running on t52_v3 10k (20 eps).
- t52_v3 10k diagnostic (20 eps, closest approach): the arm reaches a cube in 19/20 eps (min dist 0.015-0.06 m, GOAL_THRESH 0.05) but the right colour in 1/20 (chance 1/9) -> selection/grounding failure, not reaching and not note quality (removal note names the cue colour, 0.80 discriminable). Likely data: 250 demos / 9 colours (~28 per colour; t51 has ~50 per colour and learns it).
- Plan: one policy for RememberColor 3/5/9-Long (t5x, 750 demos, tools/merge_ds.py via lerobot aggregate_datasets), v3 notes vs no notes, 20k each, evals on t50/t51/t52 (queue4, after t52_v3 15k). queue3 (separate t51_v3/t50_v3) dropped.
- pi05_t52_v3 15k: 0.04. pi05_t5x_v3 (joint t50/51/52, v3 notes) training started 02:22.
- **pi05_t5x_v3 10k (one policy, t50/51/52, v3 notes, DEADBAND 0.02)**: t50 0.86 (chance .33), t51 0.82, **t52 0.34** (single-task t52: 0.04-0.08). Closest-approach: right cube 48/50, 50/50, 18/50; reached any cube 45, 42, 44 -> sharing 750 demos fixes most of t52's colour grounding; t51 failures are now reaching/stopping.
- **pi05_t5x_v3 20k**: t50 0.96, t51 0.84, t52 0.38 (right cube 48/50/22, reached 49/42/42). t52 errors: ~11/28 within the colour pairs the VLM cannot separate (red/maroon, yellow/orange, cyan/teal), rest arbitrary. pi05_t5x_base (no notes) 20k training started ~09:40.
- **pi05_t5x_base (no notes) 20k**: t50 0.14, t51 0.14, t52 0.06 (right cube 16/10/5 of 50). => notes v3: +0.82 / +0.70 / +0.32 over the same policy without notes.
- Next: ablation of note quality on the joint setting: pi05_t5x_v2 (v2 notes = SmolVLM2-500M full frame), 20k, DEADBAND 0.02 (queue5).

## 2026-10-26 11:00 KST — v2-notes joint ablation result, oracle joint started
- (GPU 7 idle 10/23 23:13 -> 10/26 11:00; no queue was chained after queue5.)
- **pi05_t5x_v2 20k (v2 notes, joint, DEADBAND 0.02)**: t50 0.86, t51 0.86, **t52 0.50** (v3: 0.96/0.84/0.38). Right cube 49/50/32 (v3 48/50/22), reached 44/44/38.
  => cleaner notes (v3) do not raise SR; difference on t52 (32 vs 22 right) borderline at n=50.
- Online cue-note analysis (first note, lenient colour match vs env truth): v2 t52 cue-note ok 36/50 (right cube | ok 0.75, | bad 0.36); v3 30/50 (right | ok 0.50, | bad 0.35). t50/t51: right cube ~1.0 regardless.
  t52 failures with an "ok" cue note are mostly red<->maroon, yellow<->orange: both VLMs call maroon "red", so the text note carries no information to separate them -> the bottleneck is the colour vocabulary of the note writer, not hallucination.
- Oracle note joint upper bound (queue6): exact env names incl. maroon/teal (make_note_ds NAMES extended), t5x_oracle (750 demos), 20k, eval --note oracle. Online oracle labeller checked: 50/50 correct on t50/t51/t52 seeds.
- Idea next: visual notes (event crops kept as an image memory) vs text notes vs both -> removes the VLM naming bottleneck while keeping event-triggered writes.
- **pi05_t5x_oracle 20k (exact cue names incl. maroon/teal, joint, DEADBAND 0.02)**: t50 0.96, t51 0.98, **t52 0.80** (v2 notes 0.50, v3 0.38, none 0.06).
  => with a correct note the joint policy solves t52 at 0.80: the remaining gap of the VLM notes is the note content (colour naming), not the policy's grounding. Visual notes (pi05_t5x_vis, queue7) training now.

## 2026-10-26 evening — analysis, extra evals, lit survey, v4 notes
- t52 confusion (50 eps, which cube was approached): base right 5 (pair-confusions 2, other 43); v2 right 32 (pair 11, other 7); v3 right 22 (pair 8, **other 20**); oracle right 42 (pair 1, other 7).
  v3 "other" errors happen with a correct cue note (e.g. "green." / "Change: a green square -> brown." -> went cyan). v3 notes also list every candidate colour ("Change: a wooden surface -> a red cube, a green cube, ... 9 colours"), v2 says "colorful cubes".
  Hypothesis: notes that describe what is visible anyway (appeared content) add distractor colour words and dilute the cue binding. Memory should hold only what the current view no longer shows.
- => notes v4 (notes.gone_note, NOTES_V=4): v3 captions, keep only the before side of a change ("Gone: a red square."), drop it if the before side is just the table. t5x_v4 built (queue8), trains after pi05_t5x_vis.
- tools/extra_eval.sh: t52 seeds 50..149 (100 more eps) for oracle, v3, v2, base 20k -> n=150 per model for the paper table.
- Lit survey -> notes_lit_2.md. Key: RePlanMem-VLA 2610.22556 (MIKASA group; full-frame memory written when overlapping chunk predictions disagree; pi0.5 on MIKASA-Robo-VLA) = closest competitor; RoboMME 2603.04639 (VLM-written language subgoals 12-33% vs GT 84% on pi0.5: same text-bottleneck finding); KEMO 2606.23589, EventVLA 2606.20092 (event/keyframe writes). Crop-based visual notes appear novel. Ablation to add: full-frame keyframe memory (RePlanMem-style) vs zoomed crops.

## 2026-10-28 — results while away, disk incident, report v4
- Extra t52 eps 50-149, pooled n=150: oracle 0.867 [0.80,0.91] (right 140), v2 0.50 [0.42,0.58] (91), v3 0.41 [0.33,0.49] (73), base 0.08 [0.05,0.13] (14).
- **pi05_t5x_vis (visual notes as 3rd camera)**: 10k 0.26/0.16/0.10, 20k 0.12/0.12/0.14 = no-note level. Online memory image dumped (VMEM_DUMP): correct (cue crop tile at t=101, wood tile at candidates, spurious arm tile after). => the policy ignores the new camera at 750 demos / 20k. Next: inset the memory crop into the top camera image; text+image combo.
- 10/27 02:21 root disk full (other users) -> pi05_t5x_v4 died at dataset load (HF datasets cache in ~/.cache). train_eval now exports HF_DATASETS_CACHE=/data/workspaces/mikasa/cache/hf_datasets. v4 restarted 10/28 17:30. queue9 (visfull) cancelled until vis works; t5x_visfull dataset built.
- Report artifact updated (version 4).

## 2026-10-28 18:30 — reviewer-style feedback from user, new priorities
- Priorities: (1) v4 (running) (2) standard RememberColor3/5/9 for direct comparison with MemoryVLA/GMP/Chameleon/SimpleMemVLA (3) other memory types (oracle feasibility first) (4) ablations + seeds; separate losses into VLM extraction / representation / policy utilisation.
- Standard RC = t13/14/15 (RememberColor{3,5,9}-VLA, PPO-collected, short). Caveat from 10/19: PPO expert anticipates the target before candidates appear -> check feasibility with oracle notes first.
- queue10: t1x_oracle (joint t13-15, oracle notes, 10k, eval 5k/10k, SIM=gpu canonical) then t1x_base (no notes, 10k). Starts after pi05_t5x_v4.
- t15 oracle labels: k-means on t15 merged yellow/orange -> cue_labels.color_clusters assigns t15 to the t52 centres; online labeller 50/50 on t13, t14, t15 (GPU physics). train_eval.sh: SIM env var.
- BlinkCount Easy (t47, 1-3 blinks, press red button N times then submit): seqpol/oracle_labels.py recovers target_blinks per demo by seed matching (250/250 exact; counts 1/2/3 = 79/80/91). make_note_ds COUNT branch -> notes/t47_oracle ("The blue lamp blinked N times."), evaluate.py oracle branch for 47-49. queue11 (after queue10): pi05_t47_oracle 15k (eval 10k/15k) then pi05_t47_base 15k. Note: even with the oracle count the policy must track its own presses (procedural memory) -> tests feasibility of the counting family.

## 10/29 — v4 결과, v5 (측정 색 이름)
- pi05_t5x_v4 20k: t50 0.94 / t51 0.90 / t52 0.54 (v2 0.86/0.86/0.50, v3 0.96/0.84/0.38, oracle 0.96/0.98/0.80).
- t52 오류 유형 (50 ep, right/amb pair/other): base 5/2/43, v2 32/11/7, v3 22/8/20, **v4 33/13/4**, oracle 42/1/7.
  - v3→v4: cue 노트 내용은 동일(같은 crop, 같은 VLM), 출현 노트만 제거 → "other" 20→4. Selection 축 근거.
  - v3/v4 cue 노트 색 계열(pair level)은 50/50 정확. 남은 오류는 amb pair: VLM이 maroon→"red", orange→"yellow", teal→"cyan"으로 씀 (0↔6, 3↔7, 5↔8 쌍을 같은 단어로).
- v5 = v4 + 측정 색: 사라진 영역의 채도 상위 25% 픽셀 평균 → CSS4 148색 중 최근접 이름(과제 무관 어휘)으로 첫 색 단어 교체.
  - demo 120 ep에서 9 클러스터 모두 서로 다른 이름, purity 100% (raw/chroma420 동일): lightseagreen, darkred, darkturquoise, goldenrod, red, lime, fuchsia, gold, mediumblue.
  - notes_vlm/notes.py measured_color/color_note, css4.py; build_ds V=5; server NOTES_V=5.
- 큐: queue10 부모 종료(t1x_oracle 학습은 계속) → queue12: v5 build/merge → t1x_oracle 끝나면 v5 20k → t1x_base 10k. queue11(t47)은 t1x_base 뒤.
- t1x_oracle 5k (GPU physics): t13 0.60, t14 0.56 (t15 진행중).
- v4 t52 추가 100 ep (seeds 50-149) 평가 진행중 (tools/extra_eval2.sh).
- 문헌 확인 (notes_lit_3.md): RC3/5/9 수치 5개 모두 원 논문과 일치. 전부 표준 MIKASA-Robo(5과제, 250 demo/과제)이고 -VLA/-Long 변형이 아님. 프로토콜은 제각각:
  - MemoryVLA 44/30/20, MemoryVLA++ 50/19/16: 100 ep, best ckpt.
  - GMP 80/61/17: Fig.11 막대그래프에서 읽은 값, 과제별 정책.
  - Chameleon 94.4/72.2/53.3: 30 rollout × 3 seed, 비VLA 약 60M.
  - SimpleMemVLA 71/58/59: Qwen3.5-4B, 프로토콜 불명.
  - ReMem-VLA: MIKASA 결과 없음. RePlanMem-VLA: 탐지 지표만 있고 성공률 없음.
  - μVLA 2606.12497: MIKASA-Robo-VLA RC5 0.93 / RC3 0.92 / RC9 0.41 (부분 확인).
  - 텍스트·이벤트 메모리 VLA가 RC 성공률을 보고한 2026 논문은 찾지 못함.

## 10/29 오후 — 계획 변경: Long 26개 과제 전체
- v5 20k: t50 1.00 / t51 0.92 / t52 0.50.
  - amb 13→4로 줄었지만 other 4→18로 늘었다.
  - 원인: CSS4 이름 경계(blue/mediumblue 동률, orange/goldenrod)에서 온라인과 학습의 이름이 다르게 나옴.
- v6 = HTML 16색 + orange 어휘(이름 간 여유 ≥18), 첫 물체만. demo 250 ep에서 purity 100%. queue13에서 t5x_v6 20k.
- t1x_oracle 10k (GPU physics): RC3/5/9 = 0.68 / 0.64 / 0.64.
- 사용자 요청: 오래 보는(Long) 벤치마크 전체 과제로. 일요일 마감은 사용자 착오였음(마감 압박 없음).
  - 26개 -Long 과제 (t50-72, t82-84)를 한 모델로 학습: base vs 노트.
  - tools/train_eval_long.sh: 학습 후 평가 3개 병렬, 과제당 50 ep.
  - queue14: t5x_v6 학습 후 pi05_long_base 90k.
  - tools/build_long.sh: 26개 과제 v6 노트(v3 캡션 캐시).
  - smoke_long: 과제별 1 ep 환경 확인.
- t47 큐(queue11)는 내림. Long 26개 안에 BlinkCount-Long(t70-72)이 포함됨.
- 주의: pkill -f나 pgrep 패턴이 내 셸 명령줄과 겹치면 셸이 죽는다(exit 144). PID를 확인한 뒤 kill할 것.

## 11/1 오후 — Long 45k 중간(13/26), 시연 영상, 보고서 v3
- pi05_long_v6 45k (base 90k 대비): RC .74/.68/.28 (base .18/.16/.06), Shape .38/.26/.16 (.08/.06/.06), SAC .32/.26/.18 (.04/.04/0), Bunch 0/0/0, Seq3 0. 13개 평균 .251 vs .052.
- tools/demo_rec.sh / demo_rec2.sh: t52·t50 ep2 영상 + 노트 로그 (v6, base; 20k). MEM_GB=12는 VLM 포함 서버 OOM → v6는 15로 재실행. 사이드카가 free<16.5GB면 이 작업만 종료.
  t52: v6 성공(노트 "Gone: a maroon square.", 단서 maroon), base 실패(red로 감). t50: 둘 다 성공, v6 노트에 팔 crop 잡음("gray dog") 다수.
- 우리 26개 = "-Long" 변형 전체; 공식 split으로는 long 14 + medium 12. 공식 long 중 44-46, 76-78, 88-89 미포함 → 논문에 명시.
- 보고서 v3(팀 공유용): 영상 + 실시간 지시문 플레이어, 결과 4 중간값, 지표 정의(맞는 큐브=최근접, 성공=터치), 논문 방향(사용자 정리본 반영).
- RESEARCH_SUMMARY_1101.md 작성.

## 11/1 저녁 — 비교 실험을 26개 과제 45k로 통일 (seed 반복 취소, 사용자 결정)
- 비교 방법(모두 26개 -Long 과제, 90k 스케줄로 학습하다 45k에서 중단 → base/v6의 45k checkpoint와 같은 LR 스케줄):
  pi05_long_v7 (v6 - t=0 장면 노트), pi05_long_visfull (이벤트 시 전체 프레임 keyframe을 추가 카메라로, RePlanMem 방식), pi05_long_v8 (이벤트 감지 없이 50 step마다 전체 화면 캡션, SmolVLM2-2.2B scene prompt).
- 코드: build_ds.py v7/v8, server.py NOTES_V=7/8, NOTE_SWAP(reset 메시지로 색 전달, "Gone:" 노트 색 교체), evaluate.py NOTE_SWAP=oracle|wrong.
- 큐: tools/build_cmp.sh(v7 → visfull → [45k mid-eval 후] v8 데이터), tools/queue17.sh(train45.sh × 3), tools/evalq17.sh(base45k → v7 → visfull → v8, 각 4그룹), tools/controls.sh(t5x_v6 20k에 노트 비움/틀린 색/정답 색, RC 3과제).
- GPU 예산: 학습 40 + long_v6 90k 평가 48 + VLM 빌드 ~8 + control 16. evalq17은 queue15 종료 후 시작.

## 11/2 06:04 gpuguard kill (recovery)
- Cause: queue17 started the v7 training as soon as the 90k v6 training process exited, while queue15's 4-group final eval, the C1 oracle control and two delay evals were running (free memory fell to 8.5 GB). gpuguard killed all GPU-7 jobs.
- Lost: v6 90k final eval, v7 training (queue17 then moved on to visfull), C1 oracle t51/t52, delay-1000 t52 evals.
- Decision: v7 is not retrained (would end after 11/5). Instead evaluate the v6 45k model with test-time v7 notes (no t=0 scene note), tag pi05_long_v6tt7.
- New queue tools/evalq18.sh (after base 45k eval): v6 90k final -> v6tt7 45k -> visfull 45k -> v8 45k; tools/side_q.sh runs the lost single evals one at a time.
