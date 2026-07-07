# _10X migration plan for coarse-grained core tunables (2026-07-07)

**Trigger:** tune #2612 (LTC 2500-iter core, sf/lmr-corr-combined) showed
`RFP_DEEP_QUAD` pinned oscillating 1↔2 wanting 1.4 — the classic
floor-pin/coarse-grain SPSA failure (a 1-unit step in QUAD is a
120–400cp deep-margin jump at tail depths). Review of all core tunables
with |default| < 10 found the same disease class. **Do NOT apply while
#2612 runs** — this executes as part of the tune-application commit
train.

## Selection rule
Any tunable whose semantics are continuous (margins, coefficients,
divisors, depth *gates*) and whose default is small enough that
c_end < ~10% of value forces sub-integer steps → ×10 fixed-point via the
existing `tp10` pattern. Genuinely discrete count semantics
(QS_MAX_CAPTURES-class) stay integer.

## Migration table (core)

| Old | Default | New | Seed default | Notes |
|---|---|---|---|---|
| RFP_DEEP_QUAD | 2 | RFP_DEEP_QUAD_10X | #2612 converged ×10 (~14) | flagship |
| RAZOR_DEPTH | 4 | RAZOR_DEPTH_10X | ×10 from #2612 | depth gate |
| RFP_DEEP_KNEE | 6 | RFP_DEEP_KNEE_10X | ×10 from #2612 | knee position |
| LMR_ROOT_COEF | 8 | LMR_ROOT_COEF_10X | ×10 from #2612 | slope; formula gains /10 |
| LMP_DEPTH | 8 | LMP_DEPTH_10X | ×10 from #2612 | depth gate |
| LMR_CUTNODE_BUMP | 2 | LMR_CUTNODE_BUMP_CENTI | ×100 from #2612 | fractional-LMR branches only: multiplies LMR_SCALE already, so centi (not 10X) — consistent with the T1.1 terms |
| LMP_BASE | 4 | LMP_BASE_10X | ×10 | probable tier |
| PROBCUT_ROOT_FADE | 3 | PROBCUT_ROOT_FADE_10X | ×10 | probable tier |
| SEE_CAP_DEPTH | 8 | SEE_CAP_DEPTH_10X | ×10 | probable tier |
| SEE_CAP_HIST | 9 | (optional) | — | migrate only if #2612 trace shows pinning |
| BAD_NOISY_DEPTH | 9 | (optional) | — | ditto |

Non-core riders (same disease, not in #2612): TT_DAMP_TT_WEIGHT (3, blend
weight — needs formula rework `(W10·tt + 10·β)/(W10+10)`), CORR_ERR_DIV (6).

## Mechanics (per established convention)
1. **Mechanical commit first**: rename + `tp10()` at read sites + default
   ×10 + range ×10 + c_end ×10. Bench MUST be identical when the ×10
   default reproduces the old integer (verify per param).
2. **Applied-values commit second**: seed defaults from #2612's converged
   (now expressible) fractional values. Bench changes here, expectedly.
3. Decide optional-tier entries from #2612 trajectories (pin-and-oscillate
   signature = migrate).
4. SPRT the applied branch per the standing tune-application flow; the
   migration itself adds no behavior at step 1.
