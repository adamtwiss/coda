# Move Ordering Ideas — v9 investigation catalogue (2026-04-19)

## Status as of 2026-04-21

**Landed:**
- **B1 (xray-blockers × MovePick) — #502 +52 Elo merged.** Biggest
  single-feature win in project history. The doc's "specific tactical
  motif" category validated against the "generic scoring nudge"
  category (both hanging-escape and enter-penalty H0'd).
- **Offense bonus (#554 +5.7, #558 LTC +6.3)** — Reckless-style
  quiet-attacks-enemy bonus. Adjacent to this doc's Category B family.

**Priority 4 in current execution plan (caphist-retune blocker doc):**
- **C: threat-delta capture ordering** — zero NPS cost (reuses existing
  `board.threat_deltas`). Re-rank captures by whether the move creates
  a new threat. Specific-mechanism, builds on B1 precedent, no SPSA
  needed. Slated for execution after caphist-retune and B2 skewer land.

**Still untried (lower priority):**
- A (shallow NNUE at ordering time) — high NPS cost, deferred
- B (NNUE history seeding) — training-regime dependency
- D (eval-delta extension) — compounds with threat-mag diagnostic work
- E/F/G (training regime changes: smoothness regularizer, move-rank
  aux loss, etc.) — require Titan-level research planning

**Closed since 2026-04-19:**
- Victim-scaled B1 variant (#533) H0'd — the flat B1 bonus fully
  captures the signal; victim-value-scaling doesn't compound.

---


## Background

v5 first-move-cut rate: **82.3%**. v9 w15: **72.2%**. Gap ~10pp flat
across all depths (7 through 16), which is a uniform architectural
property — not depth-to-depth consistency decay.

Measured eval-surface on 30 WAC positions:
- v5 gap_n (best vs 2nd-best, normalized by stddev): mean **1.82σ**
- v9 gap_n: mean **1.39σ**, median 1.06σ
- v5/v9 top-1 agreement on child evals: **60%**

The best move in v9 is less clearly separated from its peers. Ordering
signals (history, TT, MVV) have to work harder to distinguish.

Ruled out:
- WDL blend (w0 vs w15 produces same ordering regression)
- Threat features (a v7 net with hidden-layers-no-threats scores *worse*
  at 68.7% first-move — threats actually help ordering slightly)

Cause: **hidden-layer non-linearity**. v5 is linear-in-FT (smooth eval
surface); v9's L1+L2 transformation produces a more rugged landscape.
This is the cost of the eval-strength gain. Reckless/Obsidian/PlentyChess
all have the same ~70-75% range.

Goal reframed: **exploit v9's stronger eval while compensating for
reduced ordering clarity.** Not "fix v9 to be like v5".

## Catalogue

### Category 1: Move ordering signals from NNUE

| Tag | Idea | Status | Expected |
|---|---|---|---|
| **A** | Shallow NNUE at movepicker for top-K candidates. PV nodes only. 1-ply NNUE eval of top 5-10 quiet moves by current ordering, re-rank by eval. Cost ~5-10% NPS at PV, <1% overall (PV is rare). | not started | +5-15 Elo; most direct attack on the "peers-are-close" problem |
| **B** | NNUE-based history bonus seeding. At node entry, NNUE-score top-K moves, temporarily bias history. Short-term override of long-term history. | not started | +3-8 Elo, speculative |
| **C** | **Threat-accumulator delta for capture ordering.** Captures with large `\|threat_deltas\|` create tactical complications. Bias MVV+CapHist by magnitude. Zero NPS cost (already computed during make_move). | not started — compelling next candidate | +3-8 Elo |
| **D** | **Extension on big eval-delta.** `\|NNUE(child) - NNUE(parent)\| > threshold` → extend depth. Uses rich eval as a tactical-move oracle. Natural follow-up to #500 if that lands. | not started | +3-8 Elo |
| **C1** | Trained threat-weight-magnitude oracle. Precompute L1-norm of each (attacker, victim) weight row at net load as a "tactical importance" score. Movepicker bonus scaled by norm of threats activated. High variance but uniquely possible with v9's feature space. | not started, high variance | ±? Elo |

### Category 2: Already-live move-ordering-related experiments

| Test | Branch | Idea | Status |
|---|---|---|---|
| #500 | `experiment/threat-mag-lmr` | Threat-accumulator magnitude as LMR reduction modulator (position-level tactical density signal). Closest to idea D but applied to LMR instead of extensions. | +10.2 ± 19.0 @ 376 games, trending H1 |
| #501 | `experiment/stratified-escape-canwin` (post-SPSA) | Stratified escape bonuses + can_win_material RFP, with SPSA #486 tuned values applied. Post-tune retest of H0'd #479. | just submitted |
| — | `experiment/discovered-attack-bonus` | S14 from sweep: xray-blockers movepicker bonus. Hercules just created. | in progress |

### Category 3: Training regime changes (requires retraining cycle)

| Tag | Idea | Hypothesis | Risk |
|---|---|---|---|
| **E** | Smoothness regularizer. Add `λ * \|\|eval(pos) - avg_child_eval\|\|²` to training loss. Penalizes jagged eval landscapes → smoother ordering surface. | Non-linear eval is jagged because training doesn't penalize local roughness. A smoothness term trades absolute accuracy for surface smoothness. | Likely reduces peak eval accuracy → playing strength. Needs careful λ tuning |
| **F** | Move-rank auxiliary loss. Supplementary loss: given position and its k best moves per a strong engine, train network to preserve rank order (not absolute cp). | Direct optimization for ordering quality. Eval-as-regression-target doesn't tell the net which relative-magnitude differences matter for move picking. | Data-intensive; needs pre-generated rank labels |
| **G** | Anti-jaggedness data augmentation. Find positions where small perturbations cause big eval jumps, upweight them during training. | Explicitly train out the roughness. | Requires data preprocessing infrastructure |

### Category 4: Architectural changes

| Tag | Idea | Notes |
|---|---|---|
| **11** | Reduced L2 size (L2=16 or remove L2 layer). | L2=32 adds non-linearity. Smaller L2 → smoother eval → better ordering, possibly at eval-quality cost |
| **12** | Dual-head net: linear v5-style head for ordering + non-linear v9-style head for final eval | Gives search two signals: smooth linear for ordering, rich non-linear for eval. Adds inference cost |

### Category 5: Search parameter tweaks — SHELVED

Every search-parameter-tweak idea (widen RFP margins, wider aspiration,
force-top-N full-depth, loosen history pruning threshold, etc.) is
**shelved** per SPSA-critique discipline. These knobs are already tuned
— moving them without adding a new signal just races SPSA. They come
back onto the table only after a structural change introduces new
gradients.

## Signal → context sweep (reuses for move ordering)

If any Category 1 idea lands, the same signal × context sweep logic from
`signal_context_sweep_2026-04-19.md` applies:

- **Threat-accum magnitude** (#500): landing in LMR → try RFP, futility,
  ProbCut gate, extension, aspiration-width.
- **Threat-delta per move** (C): landing in capture ordering → try
  quiet ordering (via re-rank), LMR gate, futility gate.
- **NNUE eval-delta extension** (D): landing as extension → try RFP
  widening (high delta = noisy eval), LMR reduce-less, quiet move
  ordering bonus.

## Prioritised next-to-test

If Hercules wants a pick list:

1. **C** (threat-accum delta for capture ordering) — smallest code
   change, zero NPS cost, uses the delta Vec already populated by
   make_move. Distinct from #500 which is position-level LMR.
2. **D** (extension on big eval-delta) — probes the "NNUE as tactical
   oracle" hypothesis from the extend-depth side. Compounds with #500
   if both land.
3. **A** (shallow NNUE for movepicker top-K) — the big-lever idea but
   high NPS cost. Only worth it if C and D both land and suggest the
   "eval-as-ordering-signal" direction has legs.

## Not worth retrying

- WDL blend tweaks: measured flat.
- Pure NPS micro-optimizations as an Elo-lever: the move-ordering gap
  exists independent of NPS. NPS work is still valuable for its own
  sake but doesn't fix ordering.
- Search parameter tweaks without new structure: SPSA has already
  explored them.

## Updates

Update this doc when experiments land or close. Bring forward items from
Category 1 to Category 2 as they're branched. Archive Category 2 entries
that H0 or H1 into a "closed" section.
