# Eval-overrate investigation — why Coda overscores where SF doesn't (2026-06-30)

Investigation into the eval-overrate population (positions where Coda's static
NNUE eval sits far above LC0 ground truth, while SF static is close). Test set:
`testdata/heldout_overrate_lc0_2023_06.tsv` (7,847 positions / 7,847 distinct
games, held-out June-2023 T80, LC0-MCTS truth, SF-static-arbitrated — see its
README). Current prod net `E6C62000` (v4-swa, FT1024/L1=32).

## TL;DR

- **Both bucket hypotheses are refuted** on the general distribution: error is
  **flat across king buckets** (incl. castled vs uncastled-center) and the
  output-bucket/piece-count gradient is a **difficulty** gradient, not a
  coverage one (the *most-trained* bucket has the *lowest* error).
- The blind spot is **eval quality / training signal, not architecture.**
  **Coda prod and SFNNv13 are the SAME net shape** — FT1024 → 32 → 32 → 8 (see
  `docs/sfnnv13_architecture_review_2026-05-23.md`). So it is **not** capacity
  and **not** bucket layout — the gap is in **training recipe + maturity**.
- The dominant motif is **king-safety / king-danger under-weighting** (~54% of
  the worst-error mass), then material-over-count and under-valued passers.
- Across **every recent recipe** (wdl/wrm/mse/bake-length/blindspot), the
  overscore metric is **~210 ± 10cp — flat.** No recipe tweak fixes it; more
  bake helps only ~7%.

## Methodology caution (a real bug we hit)

Scraping `coda`'s UCI `eval` output by **line order** misaligns — a few
positions emit off-by-one output and everything after shifts, producing
**fabricated** extreme values (we briefly "found" +1500–1800cp overrates and a
"96% black-to-move color bias" — both pure artifacts). **Always key eval↔truth
by FEN**, via `coda eval-dist --csv` (one row per position, fen included), not
by output order. After fixing this, the real picture is undramatic: mean
overrate +40, mean |err| 212 on the set; balanced colors. (If
`eval_compare_nets.py` matches by order, its absolute |err| may be inflated the
same way — the *within-pair* delta survives, which is why the dose-response
still corroborated the SPRT.)

## Bucket hypotheses — refuted (general distribution, n=56,683, mean |err| 98cp)

**King bucket (reckless kb10) — flat:**

| bucket | meaning | mean \|err\| |
|---|---|---|
| 1 | castled g1/b1 | 95 |
| 3 | **center d1/e1 (uncastled)** | **98** |
| 0 | corner | 110 |
| 9 | advanced (rank 4-8) | 90 |
| others | rank 2 / rank 3 | 93–105 |

The center/uncastled-king bucket is exactly average. Uncastled kings are **not**
worse-evaluated. The net already *has* king buckets (capacity to encode
king-position-dependent danger) yet under-weights king-safety **uniformly across
all of them** → it's a training-signal gap, not a bucket-resolution gap.

**Output bucket (piece count) — difficulty gradient, not coverage:**

| pieces | mean \|err\| | train share |
|---|---|---|
| 2–5 | **21** | 21.9% (most-trained) |
| 6–13 | 105–108 | |
| 14–21 | **116–119** (peak) | |
| 26–33 | 63–80 | |

Error peaks at mid-material (14–21 pieces) and is lowest at the *most-trained*
bucket (bare endgame). So it is not under-training; mid-material positions are
genuinely the hardest to evaluate statically (most dynamic factors per piece).

## Motif taxonomy (worst-error tail, chess-knowledge labelling)

| Motif | share pos | share err-mass |
|---|---|---|
| **M1 enemy heavy-piece attack on own (often cornered) king** | 37% | 38% |
| M4 material over-valued vs opponent activity/structure | 20% | 20% |
| M3 opponent's advanced/passed pawns under-weighted | 20% | 18% |
| **M2 own king stuck in center / uncastled** | 17% | 18% |
| M5 drawn endgame despite extra material (fortress) | 7% | 7% |

**M1+M2 (king-safety broadly) ≈ 54%** of the error mass. Cross-cutting cause:
the net **over-weights static material and its own infiltrated-piece "activity,"
under-weights initiative / king-danger / promotion dynamics.**

**Caveat:** simple python-chess heuristics could **not** cheaply reproduce the
king-safety detection (an attacker−defender count fired 0×; nominal defenders
don't stop infiltration mates). A king-safety-targeted harvest needs an
LLM/agent labelling pass or a smarter detector, not a one-line heuristic.

## Model scan — overscore is recipe-invariant

Mean |coda−LC0| on the heldout set, recent models:

```
207  s400-baseline (longer bake)
208  mse24-qa23 / s200-wrm-sf
210  multi-v8-s3 / ww-wdl24-wrm-sf
212  prod E6C62000 / s3-v4
213-214  s200-wdl24 / s200-baseline
217  s2   221  bs-mix(S200)   224  s1
```

Whole spread ~8%. More bake → lower MSE → marginally less overscore (s1→s3
~7%); loss-shapers (wrm/mse) at the good end. But **no recipe meaningfully fixes
it** — it is a persistent eval-quality gap.

## Why we overscore where SF (same data) does not

- **Not buckets** (refuted), **not a recipe we missed** (scan flat).
- The king-safety positions are **quiet, |lc0|≤600, no checks/mates** → they
  survive every filter and are **already in training at normal frequency.** The
  problem is they're a **rare minority**; MSE on the bulk doesn't pressure the
  net to get them right (signal-emphasis, not data-absence).
- Live differences from SF are **training recipe, NOT capacity or maturity** —
  the nets are the same shape, and **training scale/data is NOT a meaningful
  delta either**: our prod schedule (350/900/3000-equivalent stage lengths)
  matches SF's nettest schedule (`max_epochs: 250/950/3000`,
  `vondele/nettest/threats.yaml`), and the data pools overlap heavily
  (leela96, test60/70/78/79/80 — the same `linrock`/`vondele` HF sets we use).
  **CORRECTED 2026-06-30 (Adam, against source)** — the recipe deltas that
  actually hold up:
  - **MSE exponent: SF 2.435, us 3.0** (`pow-exp: 2.435`, threats.yaml line
    ~99 in commit `b5023a3e`) — we are HIGHER than SF, not lower as
    previously stated here (was wrongly written as "SF 2.6 vs us 2.5",
    backwards on both numbers).
  - **WDL: SF 0.26 (single value, not a range)** — derived from
    `end-lambda: 0.74` / `start-lambda: 0.74` in `advanced_stage_options`
    (SF's `lambda` = eval-label weight, so WDL-equivalent = 1 − lambda =
    0.26). Our recent prods are all 0.20. (Previously wrongly stated here as
    "SF ~0.25–0.40" — no source for a range; 0.26 is the actual fixed value
    in the current published recipe.)
  - **Eval-scale 600 vs 400 — RETRACTED, unsourced.** No `eval-scale` /
    output-scale field exists in threats.yaml; the `in-scaling`/`out-scaling:
    300/350` fields there are WRM (win-rate-model) input scaling, not a
    centipawn eval-scale analogue to Coda's `EVAL_SCALE`. Don't restate this
    claim without a real source.
  - **Dual L1 activation — NOT a closed door, just under-tested at current
    recipe.** `project_psqt_dual_regress_s800.md` H0'd dual at S800 on
    2026-06-16, but the recipe (wdl/mse/fenskip/etc.) has moved meaningfully
    since then; that memory's own caution against over-claiming closure
    applies. Worth a fresh probe under the current recipe before ruling it
    out again, not asserting it's dead.
  - **Training maturity/scale — RETRACTED as a delta.** See above; SF's
    schedule and data pool are comparable to ours, not "far more."
  - **MSE exponent — RETESTED 2026-06-30, the SF-matching direction LOSES for
    us, on both axes.** We already have a same-recipe paired comparison
    (`warm10-inter-mse30-s200` = our default 3.0, vs `warm10-inter-mse24-s200`
    = the exact net SPRT-tested in #2032/#2033 at ≈SF's 2.435). On THIS
    overrate set, mse30 ranks #123/213 (\|err\| 218) vs mse24 #191/213
    (\|err\| 228) — mse30 is better here too, not just in general Elo (mse30
    won decisively #2032/#2033: STC +12.2±5.3 H1, LTC +10.2±4.9). So "match
    SF's MSE exponent" is not an untested lever or even a tradeoff — it's
    tested and loses on both the general-Elo and the overrate-specific
    metric for our recipe. Drop it. (A *different* mse24 net, `mse24-qa23-200`,
    ranks #9 on the overscore scan — but that's a different recipe family,
    not an MSE-only ablation, so it doesn't contradict this; don't cite it as
    support for adopting mse24.)
  - **"SF does corrective fine-tuning too (`fine_tune_binpacks`)" — RETRACTED
    as an explanation (Adam, 2026-06-30).** `fine_tune_binpacks` was added to
    `vondele/nettest` only in the last few weeks; the SF build/net we
    benchmark overscore against in this investigation predates that addition.
    So it cannot explain why SF doesn't show this overscoring — SF's
    resistance to the king-safety blind spot must come from something else
    (recipe, data, or simply not being decomposed yet), not from a corrective
    stage it didn't have when measured. The corrective-data lever for Coda
    (below) still stands on its own measured merits (the −20cp blindspot
    result), just not as "parity with what SF does."

## Levers (ranked)

1. **Concentrated, motif-aware corrective data** — oversample king-safety /
   dynamic mid-material positions specifically (blindspot diluted moved only
   −20cp because it spreads across all motifs). Needs agent/LLM labelling since
   cheap heuristics fail. Stands on its own measured result, NOT on an
   (incorrect) claim that SF does the same thing — see retraction above.
2. **WDL → SF's 0.26, but only paired with WRM.** wdl24 alone ranks #1/213 on
   the overscore metric (\|err\| 204 vs 0.20's 208) but H0'd −4.0 Elo alone in
   general SPRT (#2405); only the wdl24+WRM pairing was H1 (+4.6, #2404), and
   WRM carried essentially all of that gain. MSE-exponent matching is NOT a
   lever (see retest above — loses on both axes for us).
3. **Loss-shaping** (WRM specifically) — the one piece of "adopt SF's recipe"
   that has cleanly won alone (+7.0 H1, #2406).
4. **Not a lever**: king-bucket / output-bucket layout changes, a bigger net
   (Coda and SF are the same size, FT1024→32→32→8), training scale/data
   volume (comparable, not a gap), or MSE-exponent matching (tested, loses).

## Open / next

- **"Why doesn't SF have this issue" is still genuinely unanswered.** Neither
  capacity, bucket layout, training scale, MSE exponent, nor (per the
  retraction above) SF's own corrective-data stage explain it — that stage
  postdates the SF build we're comparing against. WDL=0.26+WRM is the one
  lever with positive evidence so far, but it's untested specifically against
  THIS overscore population.
- **Piece-type specificity probe — done, refuted.** Tried to isolate which
  piece type drives the blind spot via synthetic perturbation (remove
  bishops/knights/rooks/queens from worst-overscored positions) and via a
  natural-subpopulation comparison (bin real positions by piece-presence,
  true LC0 labels, no synthetic edits). Synthetic probe gave an unstable,
  confounded signal (sign flipped between samples); the natural-data
  follow-up refutes the one thing that looked consistent (bishop
  specificity) — knight/rook show comparable or larger effects. The blind
  spot is not isolable to one piece type by either method. See
  `docs/overscore_perturbation_probe_2026-06-30.md`.
- Locate + test the SF-vs-Coda corrective model (overscore metric + SPRT) once
  the de-duplicated data is retrained.
- Population-scale motif frequencies on the 300M corpus (agent-assisted
  labelling, run on Hercules/Atlas not GPU4).
- Quantify the capacity hypothesis directly (FT/L1 size vs overscore).
