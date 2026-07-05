# SFNNv13 Architecture Review — Findings for Coda

**Date**: 2026-05-23
**Sources**:
- [nnue-pytorch docs/nnue.md](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md) (3112 lines, fetched directly)
- [SFNNv13 architecture SVG](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/img/SFNNv13_architecture_detailed_v2.svg)
- SFNNv13 commit: official-stockfish/Stockfish a6d055d, 2026-02-18
- SFNNv12 commit: official-stockfish/Stockfish 83e4204, 2026-02-12

Companion to `docs/king_threat_exclusion_2026-05-22.md` (also derived
from SF architectural review).

---

## Architecture comparison: SFNNv13 vs Coda v9

| Layer | SFNNv13 | Coda v9 | Coda + L1=32 (in flight) |
|---|---|---|---|
| Features | 82,672 (HalfKAv2_hm + FullThreats) | ~78,128 (HalfKA-style + 66,864 threats) | (unchanged) |
| FT (output per perspective) | **1024** | **768** | (unchanged) |
| Pairwise mult output per persp | 512 | 384 | (unchanged) |
| Post-pairwise concat (input to first dense) | 1024 (i8) | 768 (i8) | (unchanged) |
| **First hidden dense (out)** | **32** + dual-activation → 62 | **16** + CReLU | **32** + CReLU |
| Second hidden dense (out) | 32 | 32 | 32 |
| Output buckets | 8 main + 8 PSQT | 8 main, no PSQT | (unchanged) |
| Hidden activation | SqrCReLU(31) ++ CReLU(31) concat | CReLU | (unchanged) |
| Eval scaling | output × 600 | output × 400 | (unchanged) |
| Sigmoid scaling | 410 | 400 | (unchanged) |
| MSE loss exponent | 2.6 (per docs) | **2.5** (coda_v9 trainer) | (unchanged) |
| QA (accumulator) | 255 | 255 | (unchanged) |
| QB (output) | 64-ish | 64 | (unchanged) |

**Architectural gap is much smaller than the literature naming convention
suggests** — see "Naming convention clarification" below.

---

## Naming convention clarification (critical)

SF docs sometimes refer to "L1=1024" meaning the **FT output width per
perspective**, NOT the first hidden dense layer's output width. This
caused us to misframe the SF-vs-Coda capacity gap multiple times.

**Two layer-naming conventions in NNUE literature**:

| Convention | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| **"L1=FT" naming** (SF docs use this in some places) | (input features) | FT (1024 per perspective) | first hidden dense (32 out) | second hidden dense (32 out) |
| **"L1=first dense" naming** (Coda + Bullet use this) | FT (768 per perspective) | first hidden dense (16 or 32 out) | second hidden dense (32 out) | output |

Our prior "SF L1=1024 vs Coda L1=16" framing was **wrong by an order of
magnitude** — it compared SF's FT output width to Coda's first hidden
dense output width.

**The actual gap at the first hidden dense layer**:
- SF: 32 outputs
- Coda current: 16 outputs
- Coda with L1=32 (in flight): **32 outputs, matches SF**

Implication: the L1=16→32 widening currently in test isn't "catching up
to a much wider model" — it's **landing at SF's actual first-hidden-dense
width**.

---

## Architectural convergence: independent arrival at similar designs

SF's recent architectural changes:
- **SFNNv11 → SFNNv12** (2026-02-12): added FullThreats. Prior to v12, SF had no threat features.
- **SFNNv12 → SFNNv13** (2026-02-18): widened second hidden dense (L2 in their naming) from 16 → 32 outputs. That's it — only one change.

**Timeline reality check**: Coda is ~8 weeks old (born 2026-03-27 as a
Rust rewrite of GoChess). Coda v9 (with threats) merged 2026-04-24.
SFNNv12 (with threats) shipped 2026-02-12 — about 2.5 months **before**
Coda v9.

So the timeline is: **Reckless first** (had threats well before March
2026, was the early mover), **SF next** (Feb 2026), **Coda inherited
the direction from Reckless via the rewrite**. We didn't lead SF on
threats — we followed Reckless, and SF happened to follow around the
same time we were building Coda v9.

The point that holds: SF and Coda **converged on similar architecture
from different starting points**, both pulled by the threats-input
direction Reckless established. SF's recipe (threats.yaml) looks
familiar to us because we're on adjacent architectural paths, not
because either led the other.

---

## Top actionable findings (ranked by Elo-per-effort)

### Tier 1 — high signal, currently miscalibrated vs SF

#### 1. L1 widening 16 → 32 — structurally done, NPS-blocked (update 2026-06-06)

**Validation**: SFNNv13's headline change was exactly this widening at
the corresponding layer. SF shipped this in production in Feb 2026.

**Status update (2026-06-06)**: L1=32 trains cleanly, produces a stronger
*net* by bench-stats, but does NOT yet win net-of-NPS at SPRT. We have
**partial L1=32 SIMD work landed** in Coda's inference path, but the
remaining ~10% NPS gap relative to L1=16 still costs the net more Elo
than the eval improvement gives back. So L1=32 is currently weaker
overall **at our current training scale and SIMD specialization**, not
an unproven direction.

Two independent paths to flip L1=32 net-positive: (a) close the NPS gap
with more L1=32-specific SIMD kernels (AVX2 / AVX-VNNI / AVX-512 VNNI /
NEON); (b) richer training (more data, better recipe) that grows the
L1=32 eval advantage past the NPS tax.

**Magnitude prior** (treat as direction-only, not load-bearing): a few
Elo with full SIMD specialization; uncertain magnitude from training-recipe
improvements.

**Action**: pursue the SIMD path independently of training-recipe work.
A bigger eval-side win from v5/v6 recipe iterations would unlock L1=32
without needing to fully close the SIMD gap.

#### 2. PSQT outputs per bucket

**SF has had this since SFNNv5 (years).** Their FT outputs 1024 + 8 PSQT
outputs per perspective. The 8 PSQT outputs are averaged across
perspectives as `(wpsqt − bpsqt) × (us − 0.5)` and added directly to the
output, bypassing the L1/L2 pipeline.

Per SF docs: *"Normally the nets have a hard time learning high material
imbalance, or even representing high evaluations at all. But we can help
it with that."*

**Gap**: Coda has no separate PSQT outputs. All material signal goes
through the L1/L2 funnel.

**Effort**: small Bullet patch (8 extra outputs from FT, gather by
piece-count bucket, average and add to output). New net file format
section needed.

**Magnitude prior**: +3-8 Elo. This is a logical/structural assist for a
known eval-weak-spot, not a data-scale-dependent feature.

**Action**: fire as Tier-1 architecture experiment in next net rev.
Highest-prior architecture change we haven't tried.

#### 3. Dual SqrCReLU + CReLU activation at first hidden layer — **IN FLIGHT (2026-06-11)**

**Status**: trainer support landed (bullet fork `feature/dual-hl-activation`,
`--hidden-activation dual`); converter/inference already supported it
(legacy `--dual` + `--hl-crelu`, v8-era feature). SB20 smoke passed
end-to-end (sane RMS, zero NPS cost). S200 paired probe
(`warm10-inter-w20-dual` vs crelu control) training on gpu2.

SFNNv13 splits the first dense's 32 outputs into `SqrCReLU(31) ++
CReLU(31)` (drops one element) → concat 62 → second dense input. We use
pure CReLU at the first hidden layer.

**Gap**: Coda uses CReLU only. SqrCReLU adds quadratic curvature
cheaply (square + clamp, no extra matmul).

**Effort**: small Bullet patch (activation change) + small SIMD work
(add `v² → clamp → i8` step alongside CReLU and concat into L2 input).
~10-30 lines per architecture.

**Magnitude prior**: +2-5 Elo. We've tested "SCReLU at hidden" before
and got H0; the **concat variant is different** (uses both activations
in parallel rather than replacing one with the other).

**Action**: fire as Tier-1 training-recipe + architecture experiment.

#### 4. i8 threat weights (ARM-amplified NPS win) — **ALREADY IMPLEMENTED, gap row was wrong (verified 2026-06-11)**

**Correction**: Coda has stored threat weights as i8 in memory since the
ORIGINAL v9 threat accumulator (commit 2d7c562, Phase 1b/2a) —
`threat_weights: Vec<i8>`, i16 accumulator, on-the-fly i8→i16 widening
in both scalar and SIMD paths, NEON parity backfilled 1af1bdc. The
"Coda's threat weights are i16" gap claim below was never true; we
inherited the i8 pattern from Reckless. Nothing to do. Original
(incorrect) analysis kept below for the record.

Per SF docs: *"Threat features have different refresh patterns... 3-4×
as many changing threat features as piece features in midgame... memory
bandwidth becomes a bottleneck for accumulation speed. We thus store
threat features as i8 and convert them to i16 on the fly during
accumulation. This process seems to increase speed much more on ARM
architectures (+10%) compared to x86 (+5%)."*

**Gap**: Coda's threat weights are i16. Same 3-4× change-count
observation applies (we're a threat-heavy v9 net).

**Effort**: medium SIMD kernel work. Need i8 weight storage for threat
FT rows, on-the-fly i8→i16 widening (`_mm256_cvtepi8_epi16` /
`vmovl_s8` for NEON). Training needs weight-clipping to i8 range.

**Magnitude prior**: +5% NPS x86 (~+2-5 Elo at STC), **+10% NPS ARM
(~+5-10 Elo)**. ARM-amplification is rare for SIMD patterns — usually
NPS wins are uniform. Worth disproportionately on aarch64 deployment.

**Action**: Tier-1 NPS experiment, especially for ARM-deployment hosts.

### Tier 2 — meaningful but lower priority

#### 5. MSE loss exponent 2.5 → 2.6 (correction 2026-06-06)

SF docs L887: *"in practice, the exponent can be >2. Higher exponents
give more weight towards precision at a cost of accuracy. Stockfish
networks had good training results with an exponent of 2.6 for example."*

**Correction**: Coda's `coda_v9_768_threats` trainer already uses
`power_error(target, 2.5)` (hardcoded at `examples/coda_v9_768_threats.rs:644`,
not the Bullet default 2.0 the original table suggested). So the actual
delta to SF is 2.5 → 2.6, not 2.0 → 2.6.

**Effort**: trivial — expose as `--mse-power` flag (~5 lines).

**Magnitude prior**: tiny (one-tenth of the exponent we're already at).
The big jump (2.0 → 2.5) is already baked in.

**Action**: add `--mse-power` flag for future experimentation, but
don't allocate dedicated SPRT cycles to a 2.5 → 2.6 probe. Use the
flag if some other recipe direction wants to vary exponent.

#### 6. Blocked-sparse linear layer with input chunk size 4

SF's hot-path kernel for L1=1024 → 32 (i.e., first dense): processes
4 input bytes at a time via `_mm256_set1_epi32` over packed-as-int32,
with weights permuted by `get_weight_index_scrambled`. Heavy use of
`m256_add_dpbusd_epi32`. VNNI variant available.

**When relevant for Coda**: once L1=32 lands and we want to optimize the
post-pairwise dense.

**Effort**: substantial (~2 days). Needs scrambled-weight serialization
in `nnue_export.rs` + matching inference path.

**Magnitude prior**: +3-8% NPS from blocked-sparse + sparse-input on
typical chess positions.

**Action**: park until L1=32 lands as a confirmed direction.

#### 7. Feature factorizer can be dropped late in training

SF docs L818: *"While the factorizer helps the net to generalize, it
seems to only be relevant in the early stages... in later stages of the
training can be removed to gain some training speed."*

**Effort**: small Bullet patch.

**Magnitude prior**: training wall-clock speedup 10-20% in the tail
(not direct Elo). Enables more iteration cycles.

**Action**: park as Bullet optimization.

### Tier 3 — confirmed parity, already in Coda

- **Pairwise multiply (product pooling)**: SF describes as "Introduced
  by Stockfish, not common in machine learning... increases network's
  capacity." Coda already uses this since v9.
- **HalfKA-style features with king buckets**: parity (we use kb10
  Reckless layout; SF uses 32 squares × 11 buckets HalfKAv2_hm).
- **Threats accumulated separately from piece accumulator**: parity
  (our `threat_accum.rs` matches SF's pattern).
- **Lambda interpolation (loss = λ·MSE(eval) + (1-λ)·MSE(result))**:
  parity (we use λ=0.15).
- **i16 accumulator with QA=255**: parity.

---

## Confirms existing parking decisions

### King-as-victim threat exclusion (v10 candidate) — SF tried and didn't gain

Per SF docs (L2902): *"Since evaluation is not called in check, attacks
to a king are also redundant, **though accounting for this has not
gained in practice**."*

This **confirms `docs/king_threat_exclusion_2026-05-22.md`'s parking
decision was correct**. SF tested removing king-involving threats and
got no Elo. Don't pursue this experiment for Coda.

### Quantmoid4 — SF removed it from production

SF docs document Quantmoid4 (piecewise quadratic approximation of
sigmoid(4x)) but **don't deploy it in v13**. Treated as historical.
Don't reinvest.

### FT widening to 1024 — S200 negative, S800 positive (update 2026-06-06)

**Prior test: #1100, H0 at −7.22 Elo at S200** (https://ob.atwiss.com/test/1100/).
Bench-stats showed better eval, but combined with the NPS penalty
(wider FT = slower inference) the net effect was negative at the
under-trained S200 scale.

**S800 retest landed positive: #1693, +6.35 ±3.52 H1 at N=12808**
(https://ob.atwiss.com/test/1693/). Dev branch `experiment/ft1024-s800`,
dev net `ft1024-fs0.5-swa720-800-s800`, base `main` with `gpu4-normal-s800`
(FT=768). Net-of-NPS at the SPRT — the wider FT's eval gain at S800
overcame the NPS tax.

**What we now know:**
- S200 result is exactly what it was — **FT=1024 is data-starved at S200**.
  Not a "FT=1024 is wrong for us" finding.
- S800 with fenskip 0.5 + SWA720 has FT=1024 **already at parity-to-positive**
  vs FT=768 at the same training scale. That's a small but real win
  with the NPS tax already paid.
- The recipe of #1693 was **pre-interleave and pre-SWA-understanding**
  (it used SWA720 for what we now know was a sequential-data wobble
  mechanism). With our current understanding (interleave subsumes most
  of what SWA was fixing, see SWA discussion in `experiments.md`
  2026-06-06), the FT=1024 advantage might transfer differently —
  worth a fresh interleave-based retest before deploying.
- **Pause reason was data-pipeline issues, not architecture**: at the
  time we'd been struggling to consume more data (sequential file-list
  distribution shifts pre-interleave hurt us in many tests). With
  interleave fixing that and SWA's role better understood, the
  data-axis question that was blocking FT=1024 is now resolved.

**Action: pick FT=1024 back up.** A fresh S800 paired probe with
interleave (no SWA, or with-SWA as a separate axis) vs a matched
FT=768 baseline answers whether the +6.35 transfers, regresses, or
grows under the new data regime. Independent of multistage; doesn't
need v5 to land. Could run in parallel on a non-GPU4 host once one
is free.

**Magnitude prior**: not load-bearing. The #1693 number sets a real
floor (+6.35 with old recipe); modern recipe may move it either way.

**SF's FT=1024 works for them with ~200B unique positions (8× ours).**
The data-gap argument hasn't gone away — at our scale FT=1024's
advantage may be smaller than SF gets — but #1693 already shows it's
crossed the net-positive threshold at S800, so the question is
direction-of-improvement, not viability.

---

## SF's path-vs-data scale calibration

> **STALE TABLE — superseded 2026-07-05.** Every "gap" row below has since
> closed. Data: the full SF training set was acquired onto GPU4 (~4× the 47B
> core set; CLAUDE.md §Training Data) and production nets train on it — plus
> SFvsCoda datagen supplements, so Coda's pool is now the SAME data, slightly
> LARGER. Position-views: multi-stage prod runs (e.g. 250/900/3000-SB chains
> ≈ 400B+ views) are within ~2× of SF's. Arch: FT=1024 + L1=32 are in prod
> (E161C665-era), matching SFNNv13's 1024/32/32; PSQT-per-bucket and dual
> activation were TESTED and H0-regressed for us (#2036-#2042) — closed, not
> pending. Ratios below are kept only as the 2026-05-23 historical snapshot.
> Do not cite them as current (this table was mis-cited 2026-07-05 as live
> evidence of an "8× data gap" that no longer exists).

| Dimension | SF | Coda (2026-05-23) | Ratio (then) |
|---|---|---|---|
| Unique filtered training positions | ~200B (stage 1 alone ~80-150B) | ~25B | 8× |
| Total position-views per net (max) | ~800B over 7 stages | 80B (S800) | 10× |
| FT output width per perspective | 1024 | 768 | 1.33× |
| First hidden dense output | 32 | 16 (→ 32 in flight) | matches with L1=32 |
| Output buckets | 8 main + 8 PSQT | 8 main, no PSQT | (gap) |

SF is also at architecture diminishing returns. **Their entire arch
delta over 3 months was the L2 widening in v13.** This is a strong
signal that our scale-appropriate architectural lift list is bounded —
PSQT + dual activation + L1=32 brings us very close to SFNNv13 shape,
and beyond that is data-axis territory.

---

## Roadmap implications for Coda

### What to chase (updated 2026-06-06)

Each of these can be tested as an **independent S200 paired probe**,
NOT bundled into multistage runs. Multistage (v5 in flight on GPU4) is
a separable recipe-iteration thread.

1. **PSQT-per-bucket** — adds capacity for material-imbalance signal.
   Significant Bullet patch + new net format section. Highest doc-prior
   among untested probes, but biggest setup cost. Decouple from multistage.
2. **Dual SqrCReLU+CReLU activation at L1** — effective capacity boost,
   no parameter increase. Small Bullet patch + small Coda SIMD work.
   Validate as S200 paired probe.
3. **L1=32 NPS specialization** — engineering thread (no GPU). L1=32
   already structurally done; closing the SIMD-NPS gap flips it net-positive.
4. **FT=1024 retest with interleave** — pick back up. S800 retest with
   #1693's +6.35 prior + interleave data regime. Independent of multistage.
5. **i8 threat weights** — NPS win, ARM-amplified for Lichess deployment.
   Medium SIMD work, independent of training.
6. **MSE exponent flag** — add `--mse-power` to coda_v9 trainer for future
   experimentation. Not worth dedicated SPRT cycles at 2.5→2.6.

Priors above are direction-only, not load-bearing magnitudes. Historical
record (per Adam, 2026-06-06): priors have been wrong in both directions;
don't size decisions on them.

### What NOT to chase (currently)

- **King-threat exclusion in v10** — SF tested, no gain. Definitive.
- **Quantmoid4** — SF removed it from production.
- **Aggressive weight pruning / structured sparsity** — SF documents
  but doesn't deploy in v13; our group-lasso explorations parallel.

### Queue for later (deferred, not closed)

- **FT widening to 1024** — see "FT widening to 1024" section above.
  No longer "open" — #1693 closed it positive at S800 (+6.35 H1 net of
  NPS). Pick back up under the current interleave regime.
- **Multi-stage training** — multi-v4 chain landed 2026-06-06: cumulative
  1300 SB, ≈ −12.5 vs prod, methodology validated, see experiments.md.
  v5 iteration (1900 SB cumulative) planned on GPU4.

### Future work amplified by data-axis growth

These become MORE attractive as we add data (~50B+ filtered):

- **FT widening to 1024** — capacity makes sense once data supports it.
- **L1 regularization** — only valuable when capacity exceeds data
  ability to constrain.
- **Fen-skip-at-S800** — only works in under-data regime, which
  doubling data restores.
- **Multi-stage training past S800** — the recipe-side scaling SF uses
  requires their data scale.

---

## Notable absences from the SF doc

The SF nnue.md doc explicitly excludes: *"datasets, optimizers,
hyperparameters; a log of experimental results."* So no info on:

- WDL ramp shape and scheduling
- Fen-skipping rate and scheduling
- SWA / stochastic weight averaging
- Jitter / loss target perturbation
- Multi-stage warm restart schedules
- Diverse-data stage-1 bootstrap composition

These are in `threats.yaml`, fishtest experiment logs, and the SF
Discord — outside the doc.

Reference for these recipe knobs:
`docs/sf_recipe_experiments_2026-05-22.md` covers what we've extracted
from `threats.yaml`.

---

## When to update this doc

- If SF ships SFNNv14 or beyond: update the architecture table,
  re-rank findings against the new shape.
- If we land any of the Tier 1 experiments (L1=32 retune, PSQT,
  dual activation, i8 threats): log SPRT result here as a
  "validated/refuted" annotation.
- If data scale changes meaningfully (e.g. 50B+ unique filtered
  positions): revisit the FT widening and L1 regularization
  decisions, both of which are currently gated on data.

Don't replace — extend. This doc is a calibration snapshot for
2026-05-23, and the calibration vs SF will shift as both engines
progress.
