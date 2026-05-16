# Loose SPSA Knobs — Audit + Plan

**Date:** 2026-05-16
**Owner:** Hercules
**Status:** Phase 1 (audit) done; Phase 3 (first ablation batch) in flight.

## Hypothesis

Coda's full-sweep SPSA tunes have become less effective per fleet-hour than
they were a few months ago. The current `tunables!` macro carries 82
parameters; we suspect a meaningful fraction of those are **loose knobs**:

* SPSA cannot find a directional gradient signal for them (they drift
  randomly across tunes), so they contribute noise instead of optimization.
* Because SPSA's simultaneous-perturbation gradient estimator averages the
  loss change across ALL parameters per iteration, every loose knob is a
  noise injection into the gradient seen by every *useful* knob. With 82
  params, if 30 are loose, the high-leverage params fight through ~30 axes
  of pure noise to find their own gradient.

This explains the empirical pattern we've seen: full-sweep tunes that
should be banking +5 Elo are landing at +1-2 Elo, and many adjacent
parameters disagree on direction across same-trunk tunes (see #1247 vs
#1250 comparison).

## Two failure modes — IMPORTANT distinction

Not every loose knob is pure noise. Two structurally different cases look
identical in SPSA drift patterns:

1. **Pure noise** — the gated feature provides no Elo. The tunable can be
   removed entirely along with the feature code.
2. **Rare-firing but valuable** — the feature fires only in specific
   positions (king-zone gates, undefended-piece checks, narrow tactical
   patterns). The gradient is noisy because the firing rate is low, but
   the feature is real and the right value matters when it does fire.
   Hardcoding to a consensus value preserves the feature without
   contributing SPSA noise.

We can only distinguish these via **ablation SPRT + static-analysis of the
gate structure** — not by SPSA drift alone.

## Audit data

`scripts/loose_knob_audit.py` pulls SPSA digests from OB for 11 recent
full-sweep tunes (#1250, #1247, #1228, #1117, #1070, #1071, #928, #882,
#871, #870, #855) and classifies each parameter by cross-tune behaviour:

| Class | Count | Meaning |
|-------|-------|---------|
| STABLE-CONVERGED | 29 | Values cluster across tunes (low stdev). Well-tuned, keep. |
| CONSISTENT-PULL | 11 | High movement, 80%+ same direction. SPSA still finding gradient. Keep tuning. |
| **DRIFTING-LOOSE** | **24** | Small/medium movement, near-50% sign-split. Loose-knob candidates. |
| DISAGREEING | 6 | High movement, near-50% sign-split. Possibly coupled (basin-swapping) or broken feature. |
| MIXED | 34 | Intermediate. Need more data. |

Top **DRIFTING-LOOSE** candidates (smallest absolute-magnitude movement
with the weakest direction signal — these are the strongest "pure noise"
candidates):

| Param | N tunes | Mean abs % | Sign-consistency |
|-------|---------|-----------|------------------|
| BONUS_BOOST_AT | 5 | 11.0% | 60% |
| ESCAPE_BONUS_MINOR | 7 | 10.6% | 57% |
| QSEE_BONUS | 7 | 9.5% | 57% |
| MVV_CAP_MULT | 7 | 7.2% | 57% |
| ASP_DELTA | 7 | 7.0% | 50% |
| QS_MAX_CAPTURES | 7 | 6.6% | 57% |
| ESCAPE_BONUS_Q | 7 | 6.5% | 57% |
| BATTERY_BONUS | 7 | 6.0% | 57% |
| BAD_NOISY_DEPTH | 7 | 5.0% | 50% |
| ASP_SCORE_DIV | 7 | 3.2% | 50% |

These are predominantly **positional-pattern bonuses** (escape, battery,
qsee) and **search-edge tunables** (aspiration, QS depth, bad-noisy gate)
— features whose contribution is either redundant with NNUE eval or whose
firing window is narrow enough that SPSA can't probe them.

## Plan of action

### Phase 1 — Audit (DONE)
Cross-tune variance dump via `scripts/loose_knob_audit.py`.

### Phase 2 — Static analysis (per-candidate, light-touch)
For each ablation candidate, grep `src/search.rs` for the gate structure:
* Universal gate (always fires) → SPSA drift = pure noise.
* Narrow gate (king-zone, undefended-piece, specific piece-type) → SPSA
  drift = expected even for valuable features.
* Dead path → ablation at zero cost.

### Phase 3 — First ablation batch (IN FLIGHT)
Five `[-3, 3]` SPRTs against current main (UHO book, fleet-default since
2026-05-16):

| ID | Branch | Param ablated |
|----|--------|---------------|
| #1254 | ablate-battery-bonus | BATTERY_BONUS → 0 |
| #1255 | ablate-escape-bonus-minor | ESCAPE_BONUS_MINOR → 0 |
| #1256 | ablate-escape-bonus-q | ESCAPE_BONUS_Q → 0 |
| #1257 | ablate-qsee-bonus | QSEE_BONUS → 0 |
| #1258 | ablate-bonus-boost | BONUS_BOOST_AT depth-boost removed (both sites) |

Per-test action on outcome:
* **H1 (ablation banks Elo)** → feature was hurting; remove feature code,
  remove tunable.
* **H0 in [-1, +1]** → feature is neutral at current scale; remove tunable
  (hardcode to current default if the code path is shared with other
  logic), reclaim SPSA bandwidth.
* **Clear regression (< -2)** → feature is load-bearing; keep code,
  hardcode tunable to consensus value (still reclaims SPSA bandwidth).

### Phase 4 — Second ablation batch
After Phase 3 resolves, the next 5-10 candidates from DRIFTING-LOOSE +
DISAGREEING. Likely candidates: MVV_CAP_MULT, ASP_DELTA, ASP_SCORE_DIV,
QS_MAX_CAPTURES, BAD_NOISY_DEPTH.

### Phase 5 — `--core` flag
Build a curated 30-40-param "core" list (NMP/RFP/Fut/LMR/LMP/SE/DEXT/
HistBonus/CapHist/CorrHist clusters at leadership level). Add
`scripts/ob_tune.py --core` to use it. Default `ob_tune.py` keeps the
full 82 (for major-restructure retunes); `--core` is the routine retune
testbed. Combined with UHO book, the SNR per fleet-hour should improve
markedly.

### Phase 6 — Recurring audit
Run `loose_knob_audit.py` every ~15 merged structural changes. Knobs that
have become loose (because their feature got bypassed) drop out; knobs
that have become active (interact with new features) come back in.

## Compound effect with UHO book switch

Today's UHO book adoption (2026-05-16) and the loose-knob removal stack
multiplicatively:

* **UHO**: ~2.5× SPSA gradient information per iteration (decisive-rate
  advantage)
* **Loose-knob removal**: removing 20-30 noise axes from an 82-param
  tune sharpens the gradient for the remaining params

A full-sweep retune after Phase 3+4 should land **noticeably more Elo per
fleet-hour** than the #1247/#1250 baseline. The combination is the main
driver behind reviving SPSA productivity.

## Open questions

* **How aggressive should ablation removal be?** Conservative answer:
  only remove tunables whose ablation SPRT is H0 in [-1, +1]. The "[-3, 3]
  with mildly negative result" cases are the risky ones — feature might
  be load-bearing in untested positions.
* **How does loose-knob audit interact with retune-on-branch?** A branch
  that changes search shape might activate previously-loose knobs. The
  audit should be re-run on retune-on-branch tunes specifically, not just
  full main-retunes.

## Cross-references

* `scripts/loose_knob_audit.py` — the audit script
* `feedback_uho_book_better_for_sprt_and_spsa.md` — UHO upgrade context
* `feedback_spsa_signposted_audit_workflow.md` — adjacent pattern
  (SPSA outliers as audit signposts; loose-knob audit is the inverse —
  SPSA drift instead of pull)
* `feedback_10x_compresses_magnitude_spread_for_spsa.md` — the _10X
  rename pattern; sister concept (SPSA SNR per-parameter)
