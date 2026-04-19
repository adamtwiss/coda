# Coda v9 Tunable Anomalies — Cross-Engine Comparison

## Methodology

- **Coda source**: `src/search.rs` `tunables!` macro on `feature/threat-inputs` HEAD (post tune #490).
- **Reference engines** (5): Stockfish dev, Reckless, Obsidian, Viridithas, Alexandria. All in `~/chess/engines/`.
- **Scope (this pass)**: 10 core pruning/search constants where formulas are structurally comparable.
- **Status**: second pass. First pass used SF+Reckless only and over-flagged futility/history as outliers; adding Obsidian/Viridithas/Alexandria softens those flags and isolates the real anomalies.

"Outlier" = Coda >1.5× away from median of 5-engine consensus, OR outside the full spread (not just above the median).

## Revised summary — what's actually anomalous

With 5 reference engines, the real outliers shrink to 2-3 items. Most "flagged" parameters in the first pass turned out to have wide consensus spreads that Coda sits within.

| Rank | Parameter | Status | Recommended action |
|---|---|---|---|
| **1** | **Contempt (CONTEMPT_VAL=19)** | **Strongest outlier.** All top engines (SF/Reckless/Obsidian/Viridithas/Alexandria) use 0 or don't have contempt. | Trivial SPRT with `CONTEMPT_VAL=0`. If neutral or positive, remove the tunable |
| **2** | **SEE quiet pruning scale** | Coda's `SEE_QUIET_MULT=45` applied as `-45·d²` gives threshold `-1125` at d=5. Viridithas uses flat `-93`, Obsidian `-21` per depth. Very different formula structures | Structural comparison worth doing — may indicate Coda's SEE pruning is over-aggressive at high depth |
| **3** | **History bonus cap** | Coda `HIST_BONUS_MAX=1584`. Consensus spread 1409 (Obs) — 2910 (Alex). Coda is at the low end. Combined with slope of 300 (high end), shape is steeper-then-clipped-earlier | Joint retune of `HIST_BONUS_MULT` + `HIST_BONUS_MAX` to explore different shapes |

Everything else I flagged in the first pass turns out to be within consensus spread once more engines are included.

## Full parameter comparison (10 params, 5 engines)

### 1. Futility margin per-depth coefficient

First pass flagged this as a 2× outlier (based on SF + Reckless only). That was wrong.

| Engine | Formula (simplified) | Per-depth coefficient |
|---|---|---|
| Coda | `FUT_BASE + lmr_d·FUT_PER_DEPTH + hist_adj + threats_adj` | **163** |
| Obsidian | `FpBase + lmrDepth·FpDepthMul + hist_adj` | **153** |
| Alexandria | `futilityCoeff0 + lmrDepth·futilityCoeff1` | **118** |
| Reckless | `eval + depth·88 + history/1024·63 + 88·(eval≥β) - 114` | **88** |
| Stockfish | `(76 - 21·!ttHit)·d - improving·corrections` | **76** |
| Viridithas | `FUTILITY_COEFF_0 + FUTILITY_COEFF_1·d` | **70** |

**Two clusters visible**: Coda/Obs/Alex at 118-163, Reck/SF/Vir at 70-88. Not a uniform consensus — appears to be a **tuning-family** split. Coda at 163 is at the top of the high cluster but not an outlier of the overall spread (70-163).

**Revised verdict**: not an anomaly. The bimodal spread is probably genuine engine-family-dependent (which LMR formula, which improving-adjustment shape). No action.

### 2. History bonus slope (statBonus linear coefficient)

First pass flagged as 2× outlier. Revised view:

| Engine | Formula | Slope |
|---|---|---|
| Alexandria | `min(historyBonusMax, historyBonusMul·d + historyBonusOffset)` | **333** |
| Coda | `min(HIST_BONUS_MAX, HIST_BONUS_MULT·d)` | **300** |
| Reckless | `min(1806, 185·d - 81)` or `min(1459, 172·d - 78 - 54·cutNode)` | **172-185** |
| Obsidian | `min(StatBonusMax, StatBonusLinear·d + StatBonusBias)` | **175** |
| Stockfish | ~`min(1690, 133·d - 72)` | **~133** |
| Viridithas | Multi-param HistoryConfig (not directly comparable) | — |

**Coda 300 is high** but **Alexandria 333 is higher**. So Coda isn't uniquely outlier — it's in a high cluster with Alexandria. Revised: not an anomaly.

### 3. History bonus cap

This is where Coda IS somewhat anomalous:

| Engine | Cap |
|---|---|
| Coda | **1584** |
| Obsidian | 1409 |
| Stockfish | ~1690 |
| Reckless | 1459-1806 |
| Alexandria | **2910** |

Coda's cap of 1584 is mid-low (between Obs and SF). Combined with the high slope (300), Coda reaches cap faster than everyone except Alexandria. **Shape ends up steeper-clipped-earlier**. At d=5: Coda bonus = 1500, saturated. SF: 593. Alex: 1824.

**Revised verdict**: worth one joint retune of MULT + MAX. Not clearly a bug.

### 4. ProbCut margin

| Engine | Base margin | Improving adjustment |
|---|---|---|
| Viridithas | 176 | - 78 |
| Obsidian | 190 | - (various adjustments) |
| Coda | **193** | (none) |
| Stockfish | 224 | - 61 |
| Reckless | 269 | - 72 |
| Alexandria | 287 | ± 54 |

Three clusters: low (Vir/Obs/Coda 176-193), mid (SF 224), high (Reck/Alex 269-287). Coda in low cluster, not outlier.

**Not an anomaly.**

### 5. NMP base R + depth div

| Engine | Base R | Depth divisor |
|---|---|---|
| Obsidian | 4 | 3 |
| Coda | **5** | **3** |
| Reckless | 5 | 4 |
| Stockfish | 7 | 3 |
| Viridithas | formula different (additive mul) | — |
| Alexandria | formula different (depth margin) | — |

Coda sits in the middle of the spread. **Not anomalous.**

### 6. NMP eval divisor

| Engine | Value |
|---|---|
| Coda | **122** |
| Obsidian | 147 |
| Viridithas (reduction_eval_divisor) | 174 |
| Alexandria (reduction) | 221 |
| Stockfish | not directly comparable |

Coda is at the low end (more aggressive R+ from eval margin) but not far outside spread. **Borderline; not top priority.**

### 7. RFP margin (improving / non-improving)

| Engine | Improving | Not improving |
|---|---|---|
| Viridithas | 76 | 73 |
| Stockfish | ~55-65 | ~76 |
| Coda | **82** | **128** |
| Obsidian | 87 (adjusted) | 87 (adjusted) |
| Alexandria | 75 - 61 = 14 | 75 |
| Reckless | unified futility | unified |

**Coda's not-improving margin (128) is the largest in the spread.** SF and Reckless have softer improving/not-improving differentiation; Coda has stronger split.

Combined with futility observation (Coda in high cluster), the pattern: **Coda prunes harder when not-improving than consensus engines**. Could be appropriate (more caution for harder positions) or could be over-tuned.

**Borderline anomaly.** Revisit jointly with futility (#1).

### 8. Contempt

| Engine | Value |
|---|---|
| Coda | **19** (applied as -19 on self-draw) |
| Stockfish | **0** (removed circa 2022) |
| Reckless | 0 (not implemented) |
| Obsidian | 0 (not implemented) |
| Viridithas | 0 (not implemented) |
| Alexandria | 0 (not implemented) |

**Universally 0 across consensus.** Coda's 19 stands alone.

**Action**: SPRT at `CONTEMPT_VAL=0`. Expected: neutral or tiny positive. If positive, remove the parameter. If strongly negative, Coda has a structural reason the others don't (unlikely; consensus is pretty unanimous post-2022).

### 9. Singular extension — starting depth

| Engine | Starting depth |
|---|---|
| Obsidian | ~5 |
| Coda | **5** |
| Reckless | 5-6 |
| Stockfish | ~5 (recently lowered) |
| Alexandria | ~8 |
| Viridithas | ~6 |

Consensus spread 5-8. Coda at lower bound. **Not anomalous.**

### 10. Singular extension — margin per depth

| Engine | Margin per depth |
|---|---|
| Obsidian SBetaMargin | 64/64 = 1.0·d |
| Reckless | d (with variants) |
| Stockfish | (60 + 66·ttPv)·d / 55 ≈ 1.09-2.29·d |
| Viridithas | varies (double=13 flat, triple=201 flat) |
| Coda DEXT_MARGIN=10 (double-ext triggering threshold), DEXT_CAP=18 | — |

Coda's SE formulation is slightly different (uses DEXT_MARGIN/DEXT_CAP for double-ext gating, not a linear margin). Structure mismatch makes direct comparison hard. Deferred.

### 11. SEE quiet pruning scale (added this pass)

| Engine | Quiet SEE threshold formula |
|---|---|
| Coda | `-SEE_QUIET_MULT·d² = -45·d²` at d=5 → **-1125** |
| Viridithas | `MAIN_SEE_BOUND = -93` (flat?) |
| Obsidian | `PvsQuietSeeMargin = -21` per depth |
| Alexandria | depth-squared with different base |

At d=5: Coda prunes quiet moves that lose >1125 cp. Viridithas's -93 is clearly flat (regardless of depth). Not directly comparable.

**However:** `-1125 cp` is 1.25× queen value. A quiet that loses a queen at d=5 is obviously bad, but `-45·d²` scales aggressively. At d=10: `-4500`. Pruning quiets that "lose 4500 cp" at d=10 means almost nothing gets pruned that high. So the `d²` scaling is normal.

**Not anomalous in practice**, but formula structure (linear vs `d²`) varies across engines.

## Revised prioritisation (replaces first-pass list)

| # | Target | Effort | Expected |
|---|---|---|---|
| 1 | **SPRT `CONTEMPT_VAL = 0`** | Trivial (single param change) | H0/flat most likely. If H1 positive → +1-3 Elo and one fewer parameter to tune |
| 2 | **Joint retune of `HIST_BONUS_MULT` + `HIST_BONUS_MAX`** | Focused SPSA, 1000 iters | Explores different bonus shape; +2-5 Elo if current shape is wrong |
| 3 | **Review RFP improving/not-improving asymmetry** | Code read + maybe SPRT | Check whether Coda's widening is gated by conditions others handle differently |

## What to **drop** from the first-pass priorities

- Futility margin scaling — the 2× outlier framing was wrong once Obs/Alex were added. Coda is at the top of a wide spread, not alone.
- History bonus slope — Alexandria's 333 is higher than Coda's 300. Not anomalous.
- NMP base R — well within consensus spread.
- ProbCut margin — in the low cluster with two other engines.

## What's still unexamined

- **OpenBench sites**: didn't get to mining `chess.swehosting.se`, `furybench.com`, `recklesschess.space` for post-SPSA tuned values. Post-tune values are a stronger consensus signal than source defaults — engines' codebases may ship with stale defaults that SPSA has moved significantly. Deferred for a future pass.
- **LMR formula comparison**: Coda uses separate C_QUIET/C_CAP tables; most others use a unified log-based table with multipliers. Structural mismatch; not straightforward to compare.
- **Corr-hist weights**: Coda has 5 sources (pawn, NP, minor, major, cont); Viridithas has 5 but the weights are on a very different scale (1000-2000 range). Different scale normalization; deferred.
- **LMP thresholds**: Coda LMP_BASE=13 LMP_DEPTH=9 vs Obsidian LmpBase=3. Obvious formula difference; need deeper dive.
- **Aspiration window**: Coda ASP_DELTA=12 vs Obsidian AspWindowStartDelta=6. Coda is 2× — might be real outlier worth testing.

## Calibration update vs first pass

This is the second-pass version. The first pass (`git show HEAD~1 -- docs/tunable_anomalies_2026-04-19.md` for diff) over-flagged futility and history slope as 2× outliers by using only SF+Reckless as consensus. Adding Obs/Vir/Alex changed 3 "outlier" flags to "within spread".

**Lesson**: for cross-engine comparison, 5 engines is the minimum for a trustworthy consensus claim. 2 engines can produce an apparent outlier from which the third engine later shows the 2 were themselves at one end of a wide spread.

---

## Third-pass extension (LMP, aspiration, SEE quiet)

Added after the contempt=0 SPRT landed +2.58 Elo (#508), validating the cross-engine anomaly → SPRT pipeline. Same method applied to 3 unexamined parameter categories from the first-pass "still unexamined" list.

### LMP base — new outlier, worth SPRT

| Engine | Formula | Base |
|---|---|---|
| Stockfish | `moveCount >= (3 + depth·depth) / (2 - improving)` | **3** |
| Obsidian | `seenMoves >= (depth·depth + LmpBase) / (2 - improving)` | **3** |
| Viridithas | table: `2.5 + 2·d²/4.5` (improving=false) | ≈ 2.5 |
| **Coda** | `(LMP_BASE + d²) / (2 - improving)` | **13** |
| Reckless | entirely different (linear with history) | N/A |

Coda uses the **same formula as SF/Obsidian** but with base=13 vs their 3. Effect is strongest at low depth:

| Depth (improving=0) | Coda threshold | SF/Obs threshold | Coda extra leniency |
|---|---|---|---|
| 2 | 8 | 3 | +167% |
| 3 | 11 | 6 | +83% |
| 4 | 14 | 9 | +56% |
| 6 | 24 | 19 | +26% |
| 10 | 56 | 51 | +10% |

**At shallow depths (2-4), Coda keeps searching 1.5-2.5× more moves than SF/Obsidian.** High depths normalise because `d²` dominates. The asymmetry is because base adds a fixed amount while `d²` grows quadratically.

**Hypothesis**: SPSA tuned LMP_BASE up from its original default (likely ~3-4 matching SF/Obs) to 13. Could be genuine (v9 needs more leniency at shallow depth because of noisier eval — later LMP cutoff lets deeper search find corrections) or drift (like contempt was).

**Recommended action**: SPRT with `LMP_BASE=3` (SF/Obs consensus) vs current 13. Bounds `[-5, 5]` since sign unknown. LMP_BASE is a UCI tunable — no code change.

### Aspiration window

| Engine | Starting delta | Eval-scaled adjustment |
|---|---|---|
| Obsidian | `AspWindowStartDelta = 6` | none apparent |
| Coda | `ASP_DELTA = 12` | `+ eval² / ASP_SCORE_DIV (33333)` |
| Reckless | `delta = 13` | `+ avg² / 23660` |
| Viridithas | `delta_initial` + `avg² / ASPIRATION_EVAL_DIVISOR` | yes |
| Stockfish | similar eval-scaled pattern | yes |

Coda's pattern (flat 12 + eval²/33333) matches Reckless (13 + eval²/23660) closely. **Not an anomaly.** Obsidian is the outlier here at 6 flat with no eval-scaling, not Coda.

**No action.**

### SEE quiet pruning scaling

| Engine | Formula (quiet SEE threshold) |
|---|---|
| Coda | `-SEE_QUIET_MULT · d² = -45 · d²` (quadratic) |
| Obsidian | `PvsQuietSeeMargin · d = -21 · d` (linear) |
| Viridithas | `MAIN_SEE_BOUND = -93` (flat per-node) |
| Alexandria | `-score/32 + 236` (inverted, varies by move-picker context) |
| Stockfish | razoring uses `-502 - 306·d²` (quadratic, different magnitude/purpose) |

**Coda uses `d²`**, matching SF's razoring scaling family. Obsidian uses linear, Viridithas flat.

This is a **structural formula difference**, not a tuning anomaly. Engine-family design choice. Coda's current formula came from the +11.4 Elo SEE quiet fix (per experiments.md) that identified and corrected 3 structural bugs in the original implementation — it's at an informed local optimum.

**No action** — different formula families, all functional.

### OpenBench post-SPSA mining — attempted, deferred

Attempted to fetch current SPSA values from `recklesschess.space` via WebFetch. The landing page shows recent test headers (LLR, games, branch names) but not per-tune parameter values. Per-tune pages accessed by test ID would be needed.

**Conclusion**: OB post-SPSA values are accessible in principle but need per-test URL scraping, which doesn't fit WebFetch's summary mode cleanly. Could be revisited later by (a) per-test-ID navigation with targeted URLs, or (b) checking each engine's git history for "Apply tune" commits.

Lower priority than the concrete LMP finding above.

### Updated prioritisation

| Rank | Target | Source | Expected |
|---|---|---|---|
| 1 | **SPRT `LMP_BASE = 3`** (UCI option flip, no code) | Third-pass extension | Neutral most likely; if H1, +1-3 Elo and one fewer drift-tuned parameter |
| 2 | SPSA joint retune of history-shape (incl. LMR_HIST_DIV, HIST_PRUNE_MULT) | Shape-experiments doc | +2-5 Elo if shape change is meaningful |
| 3 | RFP improving/not-improving asymmetry — SF-style unified shape | Shape-experiments doc | +0-3 Elo |

## Companion docs

- `signal_context_sweep_2026-04-19.md` — signal × context experiment matrix
- `move_ordering_ideas_2026-04-19.md` — move-ordering-specific ideas
- `threat_ideas_plan_2026-04-19.md` — threat-signal experiment plan with results
- `shape_experiments_proposal_2026-04-19.md` — formula-shape experiments (Hercules handoff)
- *This doc* — cross-engine tunable anomaly scan (3 passes: initial 2-engine, 5-engine expansion, LMP/aspiration/SEE extension)
