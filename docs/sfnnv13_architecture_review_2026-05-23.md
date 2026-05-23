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
| MSE loss exponent | 2.6 (per docs) | 2.0 (Bullet default) | (unchanged) |
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

#### 1. L1 widening 16 → 32 (in flight as of 2026-05-23)

**Validation**: SFNNv13's headline change was exactly this widening at
the corresponding layer. The fact that SF shipped this in production in
Feb 2026 is strong validation that L1=32 banks Elo at our scale too.

**Effort**: SIMD kernel work (L1=16 has specialized AVX2/AVX-512 VNNI
kernels; L1>16 falls to slower row-major path at ~5-7% NPS cost
documented).

**Magnitude prior**: small (+1 to +3 Elo at our scale) plus the NPS
overhead that recovers when we specialize.

**Status**: untuned −3.79 Elo at S200 paired probe; retune in flight.
The fenskip-S800 precedent for retune-on-branch banking was small (~+3);
this case has less bench delta (33%) so retune may bank more.

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

#### 3. Dual SqrCReLU + CReLU activation at first hidden layer

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

#### 4. i8 threat weights (ARM-amplified NPS win)

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

#### 5. MSE loss exponent 2.6 (instead of 2.0)

SF docs L887: *"in practice, the exponent can be >2. Higher exponents
give more weight towards precision at a cost of accuracy. Stockfish
networks had good training results with an exponent of 2.6 for example."*

**Effort**: trivial Bullet config change (may need small patch if Bullet
hardcodes 2.0).

**Magnitude prior**: +0 to +2 Elo. Cheap probe, near-zero cost.

**Action**: fire as a quick S200 paired probe whenever there's GPU
slack.

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

### FT widening to 1024 — Coda's own prior experiment was net-negative

Per our prior testing: FT=1024 produced better eval (better bench-stats)
but slower NPS, net down a few Elo. This is consistent with the
data-limited-can't-support-wider-FT framing:

- At FT=1024, capacity exceeds what our 25B filtered positions can
  train productively
- Extra weights don't fully calibrate
- NPS cost is real and immediate
- Net: marginal negative

SF's FT=1024 works for them because they have **~200B unique positions
(8× ours)** AND deployment hardware where NPS is less sensitive. Our
architecture should be calibrated to OUR data scale.

**Don't chase FT-width parity with SF** until data scales. The +15 Elo
per data-doubling lever is the prerequisite for FT=1024 paying off.

---

## SF's path-vs-data scale calibration

| Dimension | SF | Coda | Ratio |
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

### What to chase

1. **L1=32** (currently in flight) — matches SFNNv13 at first hidden dense
2. **PSQT-per-bucket** — adds capacity for material-imbalance signal
3. **Dual SqrCReLU+CReLU activation** — effective capacity boost, no
   parameter increase
4. **i8 threat weights** — NPS win, ARM-amplified
5. **MSE exponent 2.6** — cheap recipe-side probe

These five experiments close most of the structural gap to SFNNv13 at
our data scale.

### What NOT to chase

- **FT widening to 1024** — we tested, net-negative
- **King-threat exclusion in v10** — SF tested, no gain
- **Quantmoid4** — SF removed it
- **Aggressive weight pruning / structured sparsity** — SF documents
  but doesn't deploy in v13; our group-lasso explorations parallel

### Data-axis-gated future work

These require more unique data first (~50B+ filtered) before they make
sense:

- **FT widening to 1024** (revisit once data supports the capacity)
- **L1 regularization** (only valuable when capacity exceeds data
  ability to constrain)
- **Multi-stage training past S800 with fenskip** (the under-data
  regime fenskip needs)

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
