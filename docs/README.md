# Coda Research Docs — Index

Navigation by question. Most docs are dated; check timestamps before acting on their recommendations.

## "Why is Coda v9 the way it is?"

- [`move_ordering_understanding_2026-04-19.md`](move_ordering_understanding_2026-04-19.md) — **Start here** if investigating move-ordering performance. Why v9's same-ordering-code scores 72% first-move-cut vs v5's 82%. Mechanism: flatter eval distribution. Validated on 200 positions across 3 test suites.
- [`ordering_coupled_pruning_2026-04-19.md`](ordering_coupled_pruning_2026-04-19.md) — How Coda's pruning parameters (LMP, SEE, history pruning) are compensated for the ordering weakness. Introduces **pre-search vs post-factum** framing for classifying signals.
- [`v9_nps_findings_2026-04-18.md`](v9_nps_findings_2026-04-18.md) — NPS gap vs Reckless. Historical investigation. Confirmed v9 ~2× slower than Reckless on same hardware; ~+4% banked via 2b rewrite, target_feature, fused PSQ/threat updates.

## "What should we experiment with next?"

- [`signal_context_sweep_2026-04-19.md`](signal_context_sweep_2026-04-19.md) — Systematic matrix of (signal × context) pairs. Tiered by ROI. Multiple items already resolved; check status.
- [`move_ordering_ideas_2026-04-19.md`](move_ordering_ideas_2026-04-19.md) — Move-ordering-specific idea catalogue. Categories: NNUE signals, training regime, architecture. Some superseded by completed experiments.
- [`threat_ideas_plan_2026-04-19.md`](threat_ideas_plan_2026-04-19.md) — Threat-signal experiment plan, tiered. Live results embedded.
- [`shape_experiments_proposal_2026-04-19.md`](shape_experiments_proposal_2026-04-19.md) — Formula-shape experiments for Hercules: history bonus shape, RFP unified shape, futility TT-hit modulated. Drop-in specs.
- [`next_ideas_2026-04-21.md`](next_ideas_2026-04-21.md) — **Data-driven idea shortlist.** Pattern analysis (W1–W3 wins, L1–L4 losses) of 2026-04-19→21 session, novel ideas in 4 tiers, ranked queue for Hercules, explicit stop-trying list.
- [`peripheral_mechanisms_2026-04-22.md`](peripheral_mechanisms_2026-04-22.md) — **Non-pure-search idea catalogue.** Optimism (SF/Reckless/Viri), halfmove-gated TT cutoff, 50mr-threatened mate downgrade, novel twofold-in-history blend, shuffle detector, fortress soft-cap. Blue-ocean class after #619 50mr blending landed +3.3.
- [`correctness_audit_2026-04-22.md`](correctness_audit_2026-04-22.md) — **Deep correctness audit of existing code.** 10 parallel subsystem reviews including Bullet trainer. **9 CRITICAL** (headline: C8 Bullet training/inference semi-exclusion frame mismatch affecting ~50% of training positions — potentially high-impact). Plus is_pseudo_legal EP holes, TB cache halfmove, NMP stale stack, NEON SCReLU shift, TB_WIN ply-adjust, stale TM limits, convert-checkpoint momentum, HIP Adam ABI. ~35 LIKELY, ~30 SPECULATIVE. Ranked fix batches. Resolution log appended.
- [`correctness_audit_2026-04-25.md`](correctness_audit_2026-04-25.md) — **Fresh correctness audit post-v9-merge-into-main.** 6 parallel subsystem reviews of NEW code introduced in ~107 commits since 2026-04-22. **4 CRITICAL** (headline: Bullet C8 fix is INCOMPLETE — same-type x-ray paths still diverge between training and inference; plus `coda fetch-net` rejects every valid net since 2026-04-22 audit fix; UCI `eval` + `eval-dist` silently zero v9 threat features; `--no-xray-trained` flag doesn't exist in clap). ~14 HIGH/LIKELY (incl. N6 promotion-imminent extension is dead code, `SEE_CAP_MULT` dead tunable, `NMP_VERIFY_DEPTH` floor-pinned). ~20 SPECULATIVE. Search/NNUE/TT/SMP cores verified structurally clean.
- [`hobbes_selfplay_case_study_2026-04-30.md`](hobbes_selfplay_case_study_2026-04-30.md) — **Hobbes counter-example to the self-play strategy doc.** 100% self-play, random init, 41 iterations, reached CCRL Blitz #19 at 3715 Elo. Architecture endpoint matches Coda v9. Recipes Coda doesn't use: WDL-ramp schedule, 2-stage training, opening eval-bound filter, DFRC blend, high-node subset, fen-skipping. Validates "multi-iteration progressive bootstrap is viable" while preserving the prior conclusion that single-iteration replacement always loses.
- [`selfplay_data_strategy_2026-04-30.md`](selfplay_data_strategy_2026-04-30.md) — **Self-play vs T80 strategy.** Cross-engine survey of 17 trainers; Coda's −20 Elo single-iteration result is consistent with viri52's −16.4 (literature confirms single-replacement at parity always loses). Four published recipes (random-RL bootstrap, wider-teacher relabel, re-eval-corpus + mix, per-iteration Integral chain). Concrete Elo deltas. Recommends re-eval pilot (E1, +5-25 Elo expected) and same-arch mixed self-play pilot (E2, +4-8 expected) before any replacement strategy.
- [`cross_engine_comparison_2026-04-25.md`](cross_engine_comparison_2026-04-25.md) — **Cross-engine comparison vs top 17 engines + empirical re-prioritisation.** 5 parallel mechanism-class surveys covering Stockfish/Reckless/Obsidian/PlentyChess/Integral/Berserk/Viridithas/Quanticade/Clover/Alexandria/Tarnished/Halogen/Astra/RubiChess/Stormphrax/Horsie/Caissa. **17 Coda outliers vs 5+ engine consensus** + **19 novel mechanisms**. Cross-validated against `reckless_vs_coda_pruning_diff_2026-04-25.md`: empirical 30%→57% NMP cutoff-rate gap promotes `nmp-cut-node-gate-only` to Tier-1 top; `cutoff_count` propagation promotes from Tier 3 to 2; ordering items lead Tier 1 (cause-fixes); over-firing pruning gates (LMP 28×, FP 5×, SEE 3× per Kn vs Reckless) reframed as symptom-fixes. Tier-1 cumulative envelope +25-50 Elo; ranked queue of 25 experiments.
- [`research_threads_2026-04-24.md`](research_threads_2026-04-24.md) — **Data-driven response to Hercules's R1–R5 threads.** R1 L1 sparsity (λ=1e-4 structurally doomed — Bullet proximal coupled to lr; 3 better paths). R2 LMP/NMP (why direct ports H0'd: LMP_BASE=10 vs consensus 3; pure binary NMP TT-capture skip exists in no top engine). R3 NPS levers (Viridithas input-chunk L1 permutation = likely biggest unexploited lever, +8-15% NPS). R4 retune timing (WAIT for factor SB800). R5 audit SPECULATIVE triage + 5-category fuzzer programme.

## "What do other engines do differently?"

- [`tunable_anomalies_2026-04-19.md`](tunable_anomalies_2026-04-19.md) — 5-engine SPSA tunable comparison, 3 passes. Flagged anomalies (contempt, LMP), confirmed vs refuted via SPRT.
- [`capture_ordering_crossengine_2026-04-20.md`](capture_ordering_crossengine_2026-04-20.md) — Capture-scoring deep dive across 9 engines (SF, Alexandria, Caissa, Halogen, Quanticade, Stormphrax, Reckless, RubiChess, Coda). Three SEE-threshold shapes, four captHist dimensions. Notes prior dynamic-SEE attempt and why it failed.
- [`../engine-notes/`](../engine-notes/) — Per-engine technical reviews. Top-4 hidden-layer engines (reckless, obsidian, viridithas, plentychess) have v9-Refresh sections as of 2026-04-19.

## "How should v9 merge to main?"

- [`v9_merge_plan_2026-04-19.md`](v9_merge_plan_2026-04-19.md) — Merge sequencing, tunable handling, open decisions.

## Training / NNUE

- [`training_crelu_hidden_2026-04-18.md`](training_crelu_hidden_2026-04-18.md) — CReLU vs SCReLU at hidden layers.
- [`training_warmup_curve_2026-04-19.md`](training_warmup_curve_2026-04-19.md) — Warmup schedule investigation.
- [`v9_sparsity_investigation_2026-04-19.md`](v9_sparsity_investigation_2026-04-19.md) — Sparsity of active threat features in training data.
- [`v7_training_guide.md`](v7_training_guide.md) — v7 architecture training notes.

## Older but still relevant

- [`threat_features_design.md`](threat_features_design.md) — Original threat-feature design document.
- [`search_review_findings.md`](search_review_findings.md) — Earlier cross-engine search review.
- [`t80_data_analysis.md`](t80_data_analysis.md) — Training data analysis.
- [`coda_vs_reckless_nps.md`](coda_vs_reckless_nps.md) — Historical NPS comparison.
- [`move_ordering_analysis.md`](move_ordering_analysis.md) — Older move-ordering notes (superseded by move_ordering_understanding).
- [`tm_redesign.md`](tm_redesign.md) — Time-management redesign.
- [`se-diagnosis.md`](se-diagnosis.md) — Older singular-extension diagnosis.

## Conventions

- Docs dated **YYYY-MM-DD** in filename are *state-at-that-time*. Numbers and recommendations may be stale; re-check against current trunk.
- Recommended work is handed off via branches (`experiment/*`, `docs/*`, `tune/*`). Docs reference branch names for traceability.
- Companion-doc links at the bottom of each doc cross-reference related work.

## Methodology notes captured elsewhere

Recurring lessons learned (in memory, not docs):
- SPSA drift vs compensation — before "try consensus value", check for coupling (see `ordering_coupled_pruning`).
- Cross-engine comparison needs 5 reference engines minimum (see `tunable_anomalies` pass 2).
- <1000 games SPRT = pure noise — don't trust the number, watch LLR.
- Doc live-tracking is waste; write final results when tests H0/H1.
- v9 trunk SPRT/SPSA needs `--dev-network 6AEA210B --scale-nps 250000` (transitional until v9 merges to main).
