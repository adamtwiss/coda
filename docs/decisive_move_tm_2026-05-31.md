# Decisive-position / falling-eval time management — survey + proposal (2026-05-31)

## Origin

lichess KsX0b6KG (60+10, ponder): codabot spent ~8.7s/move in a trivially won
position. The mate-in-1 case is fixed (`fix/mate-tm-early-emit` — emit
immediately on forced mate). This doc addresses the *won-but-not-mate* case:
should Coda move faster when the position is decisive, and how?

## What top engines actually do (survey of 10 engines, /home/adam/chess/engines)

**Key finding: NO surveyed engine uses an absolute `|score| > T → move fast`
trigger.** The naive "if winning by >400cp, halve time" design is NOT what
anyone does. Instead they use *comparative* eval signals — score relative to
prior iterations — as a time multiplier.

The dominant pattern is a **falling-eval factor** (SF's `fallingEval`): a
multiplier on optimum time keyed on `(previous_score − current_score)`:
- **Stockfish** `fallingEval` = clamp(11.87 + 2.21·(prevAvg−best) + 1.0·(iter−best))/100, [0.572, 1.708]. Falling eval → as low as 0.57× time; improving → up to 1.71×.
- **Integral** `score_change_factor` ∈ [0.53, 1.70], keys on positive score diff vs depth-3 and prev iter.
- **Obsidian** `scoreLoss` factor, tuned coeffs on (idPrev−score) and (searchPrev−score).
- **PlentyChess** `tmEvalDiff` factor + a `complexity` factor (large score swings → less time).
- **Reckless** `score_trend` = clamp(0.8 + 0.05·(prev−cur), [0.80, 1.45]).
- **Hobbes** score-stability [0.88, 1.2].

Plus universal **best-move stability** (Coda HAS this — STABILITY_TABLE) and
**node/subtree TM** (Coda HAS this — subtree_size_multiplier), and Viridithas's
**forced-move detector** (Coda HAS this — forced_move_multiplier).

## The gap in Coda

Coda's 4 TM factors (search.rs ~2431): stability, failed-low, forced-move,
subtree-size. It is **missing the falling-eval / score-trend factor that 5 of
the surveyed engines have.**

Worse: **Coda already computes the signal and discards it.** search.rs ~2271:
`let _score_drop = ...` — the iteration-to-iteration `drop = tm_prev_score −
prev_score` is computed (with proper mate-score guards) and assigned to a
LEADING-UNDERSCORE variable = unused. So the data plumbing exists; only the
multiplier wiring is missing.

## Why this addresses KsX0b6KG (won-but-not-mate)

In a winning position that's *still improving or stable-high*, the score
isn't "falling" — so a pure falling-eval factor (SF-style) does NOT directly
shorten time in a won-but-quiet position. SF/Integral get the speed-up in won
positions mainly from **best-move stability** (the move is obvious → stable →
0.65×/0.5× time) and **node TM** (best move hogs nodes → move early), NOT from
the eval being large. Coda HAS both of those already.

So the honest read: KsX0b6KG's ~8.7s/move was NOT primarily a missing
falling-eval factor — it was the **post-ponderhit floor** (~inc seconds) which
the mate fix only bypasses for mates. In a won non-mate position the floor
still applies. The real lever for KsX0b6KG-class slowness is the FLOOR, not a
new eval factor.

## Two separable proposals

### Proposal A (the KsX0b6KG lever): scale the post-ponderhit floor by stability
The ~inc-second post-ponderhit floor is uniform regardless of position. When
the best move has been stable for many iterations (obvious move), the floor
should shrink — there's no reason to sit ~inc seconds on an obvious recapture.
This directly targets the observed slowness. Risk: the floor exists to prevent
clock-stockpiling (PZ7pCyrx) — shrinking it on "stable" could re-introduce
that. Needs the anti-stockpile case re-tested.

### Proposal B (the consensus feature): wire the discarded score-trend factor
Add a falling-eval multiplier using the already-computed `drop`, SF/Integral-
shaped, e.g. `score_trend = clamp(BASE + K·drop, [LO, HI])` folded into the
multiplier product. This is a general TM-quality improvement (5 top engines
have it; Coda uniquely discards the signal it already computes), NOT specifically
a KsX0b6KG fix. Retune-on-branch candidate (it shifts the multiplier product,
so the other factors' calibration moves). SPSA the new constants + the cluster.

## Recommendation

These are DIFFERENT changes for DIFFERENT problems:
- **B is the higher-value, lower-risk structural win**: it's a consensus
  feature, the signal is already computed, and it's a clean retune-on-branch
  experiment. Expected small-but-real (+1-4 Elo class, like other TM factors).
- **A is the direct KsX0b6KG cosmetic/clock fix** but touches the load-bearing
  anti-stockpile floor — higher regression risk, needs the PZ7pCyrx case
  re-validated. The mate fix already removed the worst instance (M1 18.8s).

Suggested order: do B first (consensus, low-risk, retune-validated). Revisit A
only if won-position slowness still shows on lichess after B + the mate fix.

Validation for both: TM-class, so ponder-enabled local RR primary + SPRT
non-regression cross-check (per feedback_tm_testing_methodology). B needs the
retune-on-branch cluster (stability/failed-low/forced/subtree + the new
score-trend constants).
