# Coda v9 Tunable Anomalies — First-Pass Cross-Engine Comparison

## Methodology

- **Coda source**: `src/search.rs` `tunables!` macro on `feature/threat-inputs` HEAD (post tune #490).
- **Reference engines**: Stockfish (dev branch in `~/chess/engines/Stockfish`), Reckless (`~/chess/engines/Reckless`).
- **Scope (first pass)**: 8 core pruning/search constants where the formulas are structurally comparable across engines. Left for follow-up: LMR base tables, SE double/triple margins, corr-hist weights, history-table dimensions.

"Outlier" = Coda >1.5× away from median of SF/Reckless on a directly comparable constant, or pinned near a range boundary by SPSA.

## Summary — findings that warrant investigation

| Rank | Parameter | Status | Hypothesis |
|---|---|---|---|
| **1** | Futility margin per-depth | **2.1× SF, 1.9× Reckless** | Implementation-level outlier. Possibly a scaling issue analogous to the history-scaling 27× bug that was worth +31.6 Elo. Top priority to investigate. |
| **2** | History bonus slope (HIST_BONUS_MULT) | **2.3× SF, ~1.7× Reckless** | But max (HIST_BONUS_MAX=1584) is *below* Reckless's 1806 and SF's ~1690. Implies Coda saturates earlier → bonus curve is steeper-then-clipped. Possible calibration quirk. |
| **3** | ProbCut margin | **~25% below consensus** | Coda 193 vs SF 224 vs Reckless 269. ProbCut fires on a tighter margin → more aggressive probcut. May be appropriate if Coda's eval is lower-variance post-tune, or may be over-tuned. |
| **4** | NMP base R | Smaller than SF (5 vs 7) | Coda's reduction depth starts lower. Could be legitimate (different LMR dynamics) or could be a compensating setting for a bug elsewhere. |
| **5** | Contempt | 19 (SF: 0 since 2022) | Coda uses non-zero contempt; consensus top engines have dropped this. Likely vestigial; worth SPRT at 0. |

## Full parameter comparison (8-parameter first pass)

All values are from current HEAD. "Coda formula" and "Engine formula" are simplified transcriptions; see source for full context.

### 1. Futility margin per-depth

**Most suspicious outlier.**

| Engine | Formula | Per-depth component |
|---|---|---|
| Coda | `FUT_BASE + lmr_d * FUT_PER_DEPTH + hist_adj` | **163** |
| Stockfish | `futilityMult * d - (2686·improving + 362·opp_worsening) * futilityMult / 1024`, where `futilityMult = 76 - 21·!ttHit` | **76 (ttHit), 55 (no ttHit)** |
| Reckless | `eval + 88·depth + 63·history/1024 + 88·(eval>=beta) - 114` | **88** |

At depth 5: Coda margin ≈ 885; SF ≈ 380; Reckless ≈ 326. **Coda's margin is ~2.3× SF's, 2.7× Reckless's.**

Caveats:
- Coda uses `lmr_d` (LMR-reduced depth) not raw depth. At move-count 10, depth 10, lmr_d ≈ 6. Coda effective `lmr_d=6` → 163 * 6 = 978; SF `d=10` → 76*10 = 760. Still ~1.3× but comparison is less clean.
- Coda's eval scale (EVAL_SCALE=400) differs from SF's (EVAL_SCALE=328 via `PieceValue` baseline). Real comparable would be margin ÷ eval_scale: Coda 163/400 = 0.407 vs SF 76/328 = 0.232 → **Coda's margin is 1.76× SF's as a fraction of eval scale.** Still an outlier.

**Hypothesis**: the futility margin is over-compensating for v9's higher eval variance, or the `lmr_d` substitution hasn't been recalibrated against the raw-depth convention used in SF/Reckless. If the scaling is off by ~2×, an SPSA re-tune with the starting value halved might find a better basin.

**Suggested investigation** (handoff):
- Run a focused SPSA with `FUT_PER_DEPTH` starting at 82 (half of current 163) and wide range 30-200. If SPSA pulls it back to ~163, the current value is intrinsic to v9's eval distribution. If it converges near 80, the previous SPSA got stuck in a high basin.

### 2. History bonus (stat_bonus) formula

| Engine | Formula | Slope | Cap |
|---|---|---|---|
| Coda | `min(HIST_BONUS_MAX, HIST_BONUS_MULT * d)` | **300** | **1584** |
| Stockfish | `min(1690, 133·d - 72)` (approx) | ~133 | ~1690 |
| Reckless | `min(1806, 185·d - 81)` capture / `min(1459, 172·d - 78 - 54·cut_node)` quiet | 172-185 | 1459-1806 |

Coda slope is **1.8× SF, 1.6× Reckless**. But cap is the lowest of the three. Shape: steeper-then-clipped.

At d=3: Coda = 900, SF = 327, Reckless = 435. Coda's bonus at low depth is 2.1-2.7× the consensus.

**Hypothesis**: history bonus shape mismatch. Coda compensates for an untested/broken other signal (e.g., cont-hist weighting) by over-bonusing early-depth history. Could also just be v9's move-ordering needs stronger history signals because of the less-peaked eval surface (per the move-ordering investigation).

**Suggested investigation**:
- Try `HIST_BONUS_MULT = 150` (halve) and `HIST_BONUS_MAX = 1700` (raise to match consensus). If SPRT lands neutral or positive, the shape was misshapen.

### 3. ProbCut margin

| Engine | Formula |
|---|---|
| Coda | `beta + 193 (PROBCUT_MARGIN)` |
| Stockfish | `beta + 224 - 61·improving` |
| Reckless | `beta + 269 - 72·improving` |

Coda **14% below** SF's non-improving, **28% below** Reckless. Tighter margin → ProbCut fires more aggressively.

**Hypothesis**: likely a consequence of Coda's RFP + ProbCut tune being co-optimized — if ProbCut catches positions RFP would have caught, a tighter margin compensates. Or (less likely) a sign ProbCut is too aggressive on v9's noisier eval.

**Suggested investigation**: low priority — 193 vs 224 is within "legitimate tuning variation". Revisit only if ProbCut cutoff rate deviates from consensus.

### 4. NMP base R

| Engine | Formula |
|---|---|
| Coda | `NMP_BASE_R + d/NMP_DEPTH_DIV = 5 + d/3` |
| Stockfish | `R = 7 + depth / 3` |
| Reckless | `R = 5 + depth / 4` (from code; depth_div=4) |

Coda's base is 2 less than SF's. At d=12: SF R=11, Coda R=9, Reckless R=8. Coda is between SF and Reckless.

**Not a clean outlier.** Reckless is in fact more aggressive (R=8). Coda sits in the middle of the spread. Unless SPSA has aggressively pushed NMP_BASE_R away from consensus, no action needed.

Did recent SPSA move it? NMP_BASE_R=5 currently. Pre-tune value from tune #454 logs was also 5. Stable. **No action.**

### 5. RFP margin (improving case)

| Engine | Formula | Per-depth |
|---|---|---|
| Coda | `RFP_MARGIN_IMP * d` | **82** |
| Stockfish | `76·d - improving_correction` (~60-70 per-depth when improving) | ~65 |
| Reckless | (embedded in unified futility) | ~70-88 |

**Not an outlier.** Coda 82 is close to SF ~65-76 and Reckless 70-88.

### 6. RFP margin (not-improving case)

| Engine | Per-depth |
|---|---|
| Coda | **128** |
| Stockfish | ~76 (formula doesn't split by improving for the multiplier; imp/noimp is a subtraction) |
| Reckless | Per same unified futility |

Coda not-improving is **~1.7× SF per-depth**. Combined with the futility-margin outlier (#1), suggests Coda's "when not improving" case may be consistently widening margins more than consensus. Could be legitimate (v9's eval is noisier) or could be over-correction.

**Flag for investigation** as part of #1.

### 7. Contempt

| Engine | Value | Notes |
|---|---|---|
| Coda | **19** (applied as -19 to self-draw score) | Anti-draw bias |
| Stockfish | **0** | Removed contempt ~2022 |
| Reckless | **0** | Not used |
| Obsidian | **0** | Not used |

**Coda is outlier against current consensus**, but Coda's is a very small effect (10-20cp). Possibly vestigial from v5 era.

**Suggested investigation**: SPRT with CONTEMPT_VAL=0 on feature/threat-inputs. If neutral or positive, remove the tunable.

### 8. SE depth gate

| Engine | Starting depth |
|---|---|
| Coda | **5** |
| Stockfish | (varies; lowered to ~4-5 in recent versions) |
| Reckless | `5 + tt_pv as i32` → 5 or 6 |

**Not an outlier.** Coda sits at consensus lower bound.

## What's left out of this first pass

- LMR coefficients (C_QUIET, C_CAP) — formula differences between Coda's linear and SF's log-based table make direct comparison fiddly; deferred.
- History-table dimensions (from/to threat bits, conthist plies) — compared in existing engine-notes SUMMARY.md, not reproduced here.
- Correction history weights (5 sources in Coda) — different engines have different source counts; hard to normalize.
- Aspiration, SEE pruning multipliers, LMP thresholds — worth a pass later; none are flagged as obvious outliers from skim.
- Obsidian, Viridithas, PlentyChess, Alexandria — not yet consulted for this pass. Adding them would strengthen the "consensus" picture but the SF+Reckless pair is already a reasonable benchmark.

## Suggested prioritisation for handoff

In order of expected yield:

1. **Futility margin scaling** — 2× outlier, precedent (history-scaling bug) says this is the style of finding that pays off large. High-effort investigation (probably needs a bug hunt + retune, not just a parameter tweak).
2. **History bonus slope + cap retune** — if the shape is wrong, a simultaneous retune of MULT/MAX might find a better basin. Medium effort, single SPSA cycle.
3. **Contempt zero test** — trivial SPRT. Expected: neutral. If positive, delete parameter.
4. **RFP not-improving margin** — bundled with #1 investigation.

## Follow-ups (not in this doc, possibly next pass)

- Expand consensus to Obsidian + Viridithas for the same 8 parameters.
- Mine OpenBench sites (recklesschess.space, chess.swehosting.se, furybench.com) for recent SPSA-tuned values — those are the "post-tune" landing values rather than source defaults, which is the better consensus signal.
- Compare LMR formula / tables (Coda's separate C_QUIET/C_CAP tables vs SF's log-based `reductions[d] * reductions[mn]`).
- Compare corr-hist weight distribution (Coda has 5 sources; SF has fewer).

## Companion docs

- `signal_context_sweep_2026-04-19.md` — signal × context experiment matrix
- `move_ordering_ideas_2026-04-19.md` — move-ordering-specific ideas
- `threat_ideas_plan_2026-04-19.md` — threat-signal experiment plan with live results
- *This doc* — tunable-level anomaly scan, first pass
