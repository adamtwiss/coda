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

## "What do other engines do differently?"

- [`tunable_anomalies_2026-04-19.md`](tunable_anomalies_2026-04-19.md) — 5-engine SPSA tunable comparison, 3 passes. Flagged anomalies (contempt, LMP), confirmed vs refuted via SPRT.
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
