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
- The blind spot is **eval quality / training signal, not bucket architecture.**
  Net **capacity** (SF's net is much larger) remains a live contributor; the
  bucket *layout* is not.
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
- Live differences from SF: **capacity** (SF's net is much larger → can afford
  to encode the hard dynamic tail) and possibly SF's loss/filtering. The
  bake-helps signal points the same way (learning-the-tail problem).

## Levers (ranked)

1. **Concentrated, motif-aware corrective data** — oversample king-safety /
   dynamic mid-material positions specifically (blindspot diluted moved only
   −20cp because it spreads across all motifs). Needs agent/LLM labelling since
   cheap heuristics fail.
2. **Capacity** — a larger net (toward SF size) to afford the dynamic tail;
   the one architecture axis not yet refuted.
3. **Loss-shaping** (wrm/wdl) — marginal but real (tails weighted more).
4. Not a lever: king-bucket or output-bucket layout changes.

## Open / next

- Locate + test the SF-vs-Coda corrective model (overscore metric + SPRT) once
  the de-duplicated data is retrained.
- Population-scale motif frequencies on the 300M corpus (agent-assisted
  labelling, run on Hercules/Atlas not GPU4).
- Quantify the capacity hypothesis directly (FT/L1 size vs overscore).
