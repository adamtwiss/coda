# LTC Scaling — Technique Review & Experiment Plan (2026-07-07)

## Context

Coda currently performs relatively *worse* at long time control (LTC, deep
searches) than at short (STC). This is a search/eval **scaling** problem: a good
change should hold across time controls, and several known effects only surface —
or only pay off — with depth. This doc records a review of established
computer-chess techniques that are known to affect time-control scaling, assesses
which Coda already has, and lays out a prioritised experiment plan. It is our own
engineering evaluation of well-documented techniques on Coda's architecture — not
a transcription of anyone else's implementation.

A recurring, separately-diagnosed symptom links here: Coda's NNUE **over-values a
material edge in low-material drawn endings** (e.g. opposite-coloured bishops — see
`docs/corrhist_fortress_drift_2026-07-06.md` and the OCB draw-rejection notes).
Material-aware eval scaling (idea #1) addresses *both* the LTC gap and that
over-eval, which is why it leads the plan.

## What Coda already has (not candidates)

- **Fractional (centi-ply) LMR reductions** — full fixed-point accumulator
  already implemented (`search.rs`, centi-ply scale). This is often the biggest
  single LTC-scaling lever; we're not leaving it unused.
- **History aging**, **per-ply cutoff-count tracking**, a **6-source correction
  history**, and a **score-trend time-management term** all exist.

## Techniques to evaluate (mechanism + expected scaling behaviour)

External evidence for this whole class is consistent: these tend to be
STC-neutral (occasionally slightly STC-negative) and **LTC-positive** — so each
must be validated **LTC-first**, or our normal STC gate would wrongly reject it.

1. **Material-aware eval scaling.** Multiply the raw NNUE output by a factor that
   shrinks as (non-pawn) material leaves the board, so the eval is compressed
   toward 0 in reduced-material positions. Coda today scales output only via a
   single `EVAL_SCALE` constant plus the 50-move term — no material dependence.
   Highest leverage: expected to help LTC *and* remove the low-material
   over-eval that causes draw-rejections. Cost: shifts the eval scale, so it
   needs an `EVAL_SCALE`/threshold **retune-on-branch**.

2. **Fail-high blend weight cap.** Coda blends a fail-high score toward beta as
   `(best·depth + beta)/(depth+1)` using **raw, uncapped `depth`**
   (`search.rs:5881`); at LTC depths (25–30) `best` swamps beta. Cap the weight
   (e.g. `min(depth, 8)`) so beta stays meaningful deep. One-liner.

3. **Draw-score jitter.** Coda returns a flat `0` at every draw terminal. Return
   a small zero-mean node-keyed jitter instead (e.g. `(nodes % 5) − 2`) so equal
   drawn lines aren't bit-identical — breaks search out of drawn plateaus. Cheap,
   contempt-free (contempt itself was removed, SPRT #508).

4. **Root history.** A small `[side][from_to]` ordering table updated and read
   *only at the root*, added to root move scores. Stabilises the PV move earlier
   each iteration — compounds over long iterative-deepening runs. Cheap,
   ordering-only, low-risk; Coda has no root-specific history.

5. **Fractional futility margin.** Coda floors the centi-ply LMR value to an
   integer before the futility gate (`search.rs:4829/4846`). Feed the
   *un-rounded* centi-ply value into the futility **margin** for sub-ply
   precision near the leaves. Nearly free — the accumulator already exists.

6. **Low-material / blocked-structure eval contraction.** Additional eval
   down-scaling keyed on blocked-pawn count and/or opposite-coloured-bishop
   material signatures. Complements #1 for locked/fortress positions.

7. **Extension gating at depth.** Lower the singular-extension depth floor and/or
   re-gate double/triple extensions; extensions cost node budget that mostly pays
   back with depth. Strongly LTC-first (can be STC-negative).

8. **Time-management scaling.** Two small, TC-robustness fixes to cross-check
   against our Viridithas-shaped model: gate the "best move dominates, stop early"
   governor on a *fraction of soft time* rather than an absolute node count (an
   absolute node gate fires far too early at LTC); and review whether the
   max-time bound can serve directly as the hard limit.

9. **Expected-all-node reduction term.** Add a small depth-shrinking reduction
   bump at expected all-nodes (`r += r/(depth+1)`); verify Coda has no equivalent.

## Prioritised experiment plan

| # | Experiment | Branch | Effort | Why first / notes | Validate |
|---|-----------|--------|--------|-------------------|----------|
| E1 | Fail-high blend weight cap | `search/fh-blend-weight-cap` | one-liner | Cheapest, high-confidence, LTC-biased; verified applicable | STC + LTC SPRT |
| E2 | Material-aware eval scaling | `eval/material-scaled-output` | med + retune | Flagship: LTC gap **and** OCB over-eval; needs `EVAL_SCALE` retune-on-branch | LTC-first, retune, then SPRT |
| E3 | Draw-score jitter | `search/draw-score-jitter` | one-liner | Cheap, LTC-biased, contempt-free | STC + LTC SPRT |
| E4 | Root history | `search/root-history` | small | Ordering-only, low-risk; new table | STC + LTC SPRT |
| E5 | Fractional futility margin | `search/fractional-futility-margin` | small | Free reuse of existing accumulator | LTC-first |
| E6 | Low-material / blocked eval contraction | `eval/low-material-scaling` | small | Do after E2 (shares eval-scale calibration) | LTC-first |
| E7 | Extension gating at depth | `search/extension-depth-gating` | small | STC-risky; strictly LTC-first + retune | LTC-first |
| E8 | TM early-stop fraction + hard-limit review | `tm/softtime-fraction-earlystop` | small | Robustness; local RR + SPRT | RR + SPRT |

Run order: **E1 first** (fast signal), **E2 in parallel** (flagship, longer
loop), then E3–E5 as fleet allows. Everything LTC-first per the scaling caveat.

## Method notes

- Validate LTC-first (`40.0+0.4`, Hash 256); STC as the non-regression cross-check.
- Eval-path changes (E2, E6) change the natural eval scale → SPSA
  retune-on-branch before a verdict (the standard Coda pattern).
- Log every result (H0 or H1) to `experiments.md`.
