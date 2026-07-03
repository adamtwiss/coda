# Train-vs-inference parity — full-surface verification (2026-07-03)

Follow-up to the overscore investigation (`docs/overrate_eval_investigation_2026-06-30.md`)
and the 2026-07-01 threat-encoder campaign. Adam's preferred hypothesis for the
~5% blindspot population (large |coda−LC0| where SF-static is close) was
**another C8-class training/inference delta** hiding behind the fuzz tester
("not 100% like for like"). This session enumerated every remaining divergence
surface and closed each one empirically.

## Surface map (what was / wasn't verified before today)

| Surface | Status before | Status now |
|---|---|---|
| Threat encoder (enum, semi-excl, x-ray, extra[0]) | verified 2026-07-01 (cross-differ 0/12,000) | unchanged |
| **PSQ/HalfKA half** (king buckets, e-file mirror, NTM frame, piece order) | **never empirically cross-checked** (comments/asserts only) | **0/12,000** (`scripts/psq_cross_differ.py`) |
| Output-bucket selection | code-audit match ((pc−2)/4 both sides) | covered by fp32 harness below |
| **Value path** (fp32 training forward vs int8/int16 inference) | **no tool existed** | fp32 parity harness (below) |
| Label path (WRM cp-vs-/400, WDL blend) | code-audit only | n/a to eval parity (loss-side only) |

## 1. PSQ cross-differ — 0 mismatches / 12,000

`coda dump-threats` now emits a per-POV `PSQ` line (`halfka_index_with`,
reckless kb10 tables); `scripts/psq_cross_differ.py` diffs it against the REAL
Bullet training encoder (`ChessBucketsMirrored` inside
`ChessBucketsWithThreats`, via the `extra_stamp_probe::dump_file` test on
bullet branch `tooling/eval-fens-parity`) over standard + chess960 +
promotion-heavy positions, with **exact per-perspective pairing** (catches
perspective swaps a union diff would hide). Result: **0/12,000.**

With threats already 0/12,000, the **entire v9 input feature path is now
empirically verified train==inference.**

## 2. Mirror-residual scan — the deliberate asymmetry is a minor contributor

`coda eval-fens` (new batch subcommand, FEN-keyed — no UCI scraping) over the
heldout overrate corpus (7,847) and a matched June-2023 control (20,000,
same binpack/filters minus the disagreement gate), prod net E6C62000.
Residual = eval_stm(color-mirror) − eval_stm(orig); a symmetric evaluator
gives 0. This quantifies the *deliberate* physical-frame semi-exclusion
asymmetry (`threat_eval_asymmetry_2026-06-17.md` measured 17–70cp on 3
positions; option 3 of that doc asked for exactly this corpus-scale check).

| set | mean \|residual\| | p90 / p99 | mean \|coda−lc0\| | corr(\|res\|, \|err\|) |
|---|---|---|---|---|
| heldout (overrate) | 12.6 | 40 / 119 | 211.6 | 0.078 |
| control | 5.0 | 16 / 52 | 71.6 | 0.133 |

Elevated ~2.5× on the blindspot population but ~6% of its error magnitude,
and per-position correlation with error is ~zero. **The representation
asymmetry is real but is NOT the blindspot cause.** (Also: mean residual ≈ 0
both sets — no systematic color bias.)

## 3. Threat-weight i16→i8 clip — refuted as class-specific cause

convert-bullet clamps threat weights (trained i16 @ QA=255) to i8 [−128,127].
Prod checkpoint `multi-v8-l132-s3-v4-3000-swa/quantised.bin` (pre-clip i16):
2,441 / 68.5M weights clipped (0.0036%), but excess is large where it happens
(mean 37, max 386 — a weight wanting 513 stored as 127) across 1,602 feature
rows. Hypothesis: king-attack motifs activate clipped rows → inference
under-responds to king danger (the M1/M2 motif, 54% of error mass).

**Refuted empirically:** ~95% of ALL positions (heldout and control alike)
activate ≥1 clipped feature; heldout mean clip-mass (1,617) is LOWER than
control (2,721); corr(clip mass, |err|) is ≈0/negative in both sets. The clip
is ubiquitous background, not blindspot-specific. (An i16 threat-storage
format change remains a possible small generic win, unrelated to the
blindspot.)

## 4. fp32 forward parity harness (the definitive like-for-like)

Bullet branch `tooling/eval-fens-parity` adds `--eval-fens <file>` to
`examples/coda_v9_768_threats.rs`: with `--warmstart-from <checkpoint>` it
runs the REAL fp32 training graph (same `map_features`, same graph as
training, extra[0] stamped from FEN STM) over a FEN list and prints
`fen<TAB>eval_cp`. Diffed against `coda eval-fens` on the converted prod net
(same weights): any structured disagreement = value-path divergence
(quantization, integer arithmetic, output buckets, activation chain).

Also answers the ceiling question directly: if the fp32 training-graph eval
shows the same ~212cp error on the heldout set, the blindspot is in the
TRAINED WEIGHTS, and no inference-side bug can be responsible.

### Results (prod checkpoint `multi-v8-l132-s3-v4-3000-swa` ≙ E6C62000)

delta = coda_inference − bullet_fp32, both STM-POV cp:

| set | mean delta | std | mean \|delta\| | p50/p90/p99 \|delta\| | mean \|coda−lc0\| | mean \|fp32−lc0\| | corr(\|delta\|, \|err\|) |
|---|---|---|---|---|---|---|---|
| heldout (7,845) | −2.9 | 31.6 | 23.3 | 18 / 50 / 98 | 211.6 | **212.0** | 0.137 |
| control (19,164) | +4.9 | 20.3 | 15.4 | 12 / 33 / 65 | 74.0 | 74.4 | 0.207 |

**The decisive line: the fp32 training graph itself scores 212.0 mean error on
the heldout set — identical to inference (211.6).** The blindspot is fully
present in the trained weights, evaluated with the training-side feature code,
in fp32, with no quantization and no Coda inference code anywhere in the loop.
Inference adds only small, near-unbiased noise (~±20–30cp σ, means ±3–5cp,
weak error correlation, worst tails ~150–300cp concentrated in low-piece
endgames) that does not change mean error at all (−0.4cp).

### Verdict

**The train/inference-delta hypothesis is closed, across every surface:**

- Features: threats 0/12,000 + PSQ 0/12,000 vs the real Bullet encoder.
- Value path: fp32-vs-inference identical mean error; quantization noise
  unbiased and blindspot-agnostic.
- Deliberate threat asymmetry: 12.6cp mean on the corpus, corr 0.08 — minor.
- i16→i8 threat clip: ubiquitous background, anti-correlated with error.

The blindspot is **in the trained weights** — a training-signal problem
(rare-motif under-emphasis under bulk MSE, per the 06-30 taxonomy), not a
pipeline bug. This redirects all further effort to data/objective levers
(motif-targeted corrective data, WRM/WDL, emphasis mechanisms).

**Caveat on prod specifically:** prod E6C62000 was trained BEFORE the
2026-07-01 extra[0] loader fix, so prod's weights carry that (known, now-fixed,
Elo-positive-to-fix) training-side frame bug. It is NOT the blindspot cause —
the extra[0]-fixed nets show the same ~210cp overscore metric ("Elo+ but
metric-flat", experiments.md 2026-07-01) — but it is one more reason the
pending prod retrain is unbooked Elo.

**Secondary observation (not the blindspot):** total inference-vs-fp32 noise of
~±20–30cp σ per position is larger than commonly assumed. It is unbiased and
error-neutral here. **Correction (Adam, 2026-07-03): prod multi-v8 AND
multi-v9 were both trained WITH `--qat`** (an earlier draft wrongly said the
prod recipe omitted it), so this σ is the noise remaining *after* QAT models
the FT-act/pairwise/weight quantization — the audit's unmodeled site (the L1
matmul-output `/127` truncation) plus weight-grid rounding is what's left. An
i16-threat-storage or L1-truncation probe stays a possible small generic win,
valued in Elo, explicitly not blindspot-related.

## Addendum: multi-v9-s3-swa (post-extra[0]-fix prod-beater, 2026-07-03)

Same heldout corpus, `nets/multi-v9-s3-swa.nnue`:

| net | mean \|coda−lc0\| | mean \|mirror residual\| |
|---|---|---|
| prod E6C62000 (multi-v8, pre-fix) | 211.6 | 12.6 |
| multi-v9-s3-swa (post-fix) | 208.8 | **5.6** |

The blindspot persists essentially unchanged (208.8, inside the 207–224
recipe-invariance band) — consistent with everything above. But the mirror
residual **halved to control level** (heldout 5.6 vs old-net control 5.0):
the extra[0] fix removed trained-in frame inconsistency on black-STM
same-type pairs, cleaning the learned representation's color symmetry without
moving the blindspot — further confirmation the asymmetry was never the cause.

## Tooling added (permanent gates)

- `coda eval-fens` — batch FEN-keyed static eval (coda branch `tooling/psq-cross-differ`)
- `coda dump-threats` PSQ lines + `scripts/psq_cross_differ.py` (same branch)
- bullet `tooling/eval-fens-parity`: `eval_raw_output_board` (extra[0]-stampable
  fp32 forward), `--eval-fens` mode, PSQ_STM/PSQ_NTM probe lines
