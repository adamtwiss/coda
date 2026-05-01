# Coda TODO

Shared work queue across Claude instances. Prompt/reminder, not exhaustive.
`experiments.md` is source of truth for completed/in-flight SPRTs.

## Pipeline-stability gate (blocks training-side experiments)

Most training-side items below are blocked on **basin-A landing reliability**.
Until SB200 replicas cluster tightly (gpu4-vs-gpu5 +1.19 ±2.89 / 16K is the
target shape), recipe magnitudes are partly seed luck.

- [ ] **T1 hl-screlu revert SB200 ×3** (in flight, gpu3/4/5, ~6-8h) — does
  reverting `--hidden-activation crelu` restore basin-A clustering? See
  `docs/seed_variance_investigation_2026-05-01.md`.
- [ ] **T2 / T3 conditional follow-ups** if T1 doesn't clear it: revert C8
  fix, then revert both. SB50 fast-screen is viable for T2/T3.
- [ ] **T4 final-LR bump** only if T1-T3 inconclusive (H3 hypothesis: low
  final LR exposes late-stage seed variance).

## Post-T1 training queue (cross-engine survey, ordered by ROI)

Source: `docs/training_methodology_cross_engine_2026-05-01.md`.
**Fire only after pipeline-stability gate clears.**

- [ ] **N3 Arasan recipe stack** — WDL=0.0, factoriser, FT clip ±0.99,
  100× LR decay. 4-line Bullet config edit. Highest-precedent, most
  directly portable. SB200 + 1 replica.
- [ ] **N4 QA=181 re-quantisation** — no-retrain inference test on existing
  prod net. +16.76 Elo fixed-nodes precedent. Today-task; doesn't need GPU.
- [ ] **N5 sibilant L1-on-activations** — orthogonal to our group-lasso
  (which is L1-on-weights). +5.9 Elo LTC viri precedent. SB200 probe.
- [ ] **N1 eleison WDL ramp + EVAL_SCALE recalc** — replaces our planned
  Hobbes-style probe. Bare WDL ramp regressed in 3 viri experiments
  (-3 to -57); only the rescale-paired version delivered (+9.27 VLTC).
- [ ] **WDL pole probe** — once basin-A clean, A/B WDL=0 (Arasan-like) and
  WDL=0.4 (viri) at SB200, 1 replica each, vs current 0.15.
- [ ] **N9 dual-activation FT** — Hobbes h-40 pattern, now confirmed by
  Alexandria production. SB200 probe.
- [ ] **N8 Halogen int8 threat weights** — 49 MB threat matrix → ~25 MB,
  L3-residency win on most fleet hosts. NPS gain via depth lever. Stack
  with group-lasso shrinkage (orthogonal mechanism).
- [ ] **Weight repermutation** — viri co-occurrence-driven FT neuron
  reorder for NNZ locality. Pure inference; relevant when group-lasso
  delivers sparse threat rows.

## Search Improvements — Untested or open

### TT-pollution follow-ups (after #901 / #902 resolve)

- [ ] **Cross-gen TT cutoff gating** (probe-side) — if #901/#902 H0,
  attack pollution from probe side: expose `generation` in TtEntry,
  gate TT_LOWER cutoffs on cross-gen entries with stricter depth.
  Direct attack on the disentangled 79% TT-pollution finding.
- [ ] **Per-search TT probe/hit/cutoff counters** (task #141) — instrumentation
  to measure cross-gen hit ratio, cutoff source breakdown. Should land
  before any further TT-policy experiments.

### Standing search ideas

- [ ] **Store moved piece in search stack** — ply-2 cont-hist currently
  looks up piece via `board.piece_at(move_to(pm2))`, returns stale piece
  when ply-1 captured on that square (~10-20% of probes). Add `moved_piece`
  to per-ply search info before make_move. Enables correct ply-2+ cont-hist
  for both LMR and pruning.
- [ ] **Retry: drop killers / counter-move** — original test was -530 with
  pre-4D history. Now we have 4D threat-aware history (the prerequisite
  SF/Reckless/viri used when dropping killers). #1 retest priority.
- [ ] **Weiss NMP skip after null move** — anti-chain NMP. Untested.
- [ ] **Reverse-direction history in cont-hist** — Arasan-style. Untested.
- [ ] **Passed pawn push pruning exemption** — untested.
- [ ] **Soft-stop SMP thread consensus** — 65% of helper threads must agree
  on best move before stopping. (Reckless review)
- [ ] **Opponent easy-captures gating on RFP/NMP** — Berserk pattern.
- [ ] **Const attack tables + runtime PEXT reconciliation** — earlier probe
  showed +1-2 Elo but conflicted with runtime AMD PEXT detection (+20%
  NPS path). Need both to coexist.

## Lower priority / speculative

- [ ] **SF dual-net (big+small)** — gate by `|simple_eval|`; FT=1024 close,
  FT=128 decided. Significant infra; defer until single-net path saturates.
- [ ] **N6 Halogen TD-bootstrap → supervised** — pre-bootstrap a net via
  Temporal Difference, then Bullet-finetune. New training infra.
- [ ] **N10 PlentyChess self-distillation** — top-3 engine, mechanism
  undisclosed. Research-first.
- [ ] **N2 Tarnished bigger-arch teacher** — train 1.5-2× wider preceptor,
  rescore corpus, retrain prod arch. ~2× current training cost; reserve
  for after same-arch re-eval (cheaper) saturates.
- [ ] **Wider L1 (32 instead of 16)** — current is L1=16, L2=32. Doubles
  hidden capacity at ~20% NPS cost. SparseL1 path already kept for L1≥32.

## Infrastructure

- [ ] **Coda on CCRL** — submit once a stable post-seed-variance net lands.
- [ ] **`check-net` in Coda** — port from GoChess for offline net validation.
