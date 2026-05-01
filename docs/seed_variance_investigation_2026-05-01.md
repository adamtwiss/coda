# Seed-Variance Investigation — 2026-05-01

## Why this matters (broader context)

Coda's training pipeline is producing nets that differ by ±13-17 Elo
across replica trains with identical configuration (only seed differs).
This is a **methodology problem before it's a training-recipe problem**:
every recent net-vs-net SPRT magnitude claim — recipe variants, factor vs
non-factor, group-lasso decay levels, low-LR tail experiments — has been a
single-seed sample drawn from this distribution. Direction-of-effect is
usually preserved; magnitude is partly seed luck.

This investigation is trying to answer two questions:

1. **Is this variance new, or has it always been there?** If always-there,
   every prior SPRT magnitude needs derating. If new, *what changed* and
   can we revert?
2. **What's the actual mechanism?** Identifying it lets us pick the right
   deployment net (median-of-replicas vs lucky-pick) and prevents the next
   training-recipe experiment from being uninterpretable.

The downstream stakes: the +3.3 Elo factor edge over non-factor (#782),
the +11.22 Elo group-lasso 1e-2 result, the C8-fix −10.7 Elo (#836), and
a long tail of recipe results all become "direction-only" until we have a
noise floor.

**Target tightness.** gpu4 vs gpu5 (Basin A ↔ Basin A, see below)
finalised at **+1.19 ±2.89 Elo over 16,308 games**. That is the
ideal — every replica pair should look like that. Anything wider is
the variance we're hunting.

## Hypothesis trail

### H1 (initial): Factor architecture is the noise source — RETRACTED

Origin: SB800 factor replica vs prod (#879) showed −22 Elo gap with
near-identical loss (0.5%). Factor's parameterisation redundancy (factor
weight + regular weight produce identical effective input weights) creates
flat directions in loss landscape — random init picks different points
along flat dirs. Plausible mechanism.

Test (this work, 2026-05-01): four non-factor SB200 replicas on
gpu1/3/4/5 (gpu1 different CUDA/GPU class; gpu3/4/5 identical infra),
independent seeds, hlcrelu config. SPRT pairs at bounds [-5, 5]:

| Pair | Elo | ± | Games | Result | Interpretation |
|---|---:|---:|---:|---|---|
| gpu4 vs gpu5 | +1.19 | 2.89 | 16,308 | trending H1 | Both Basin A — **target tightness** |
| gpu3 vs gpu4 | −14.7 | 10.1 | ~2K | H0 | Cross-basin |
| gpu1 vs gpu4 | −13.0 | 9.6 | ~2K | H0 | Cross-basin |

Note: bounds [-5, 5] were too wide for a non-regression SPRT; [-3, 3]
would have given tighter CIs. Future replica SPRTs should use [-3, 3].

**Verdict on H1**: factor is NOT the unique source. Non-factor shows the
same magnitude. Retracted.

### H2: A recent pipeline change introduced bimodal basins

Reference data: the older w0/w05/w10/w15/w20 wdl-sweep
(pre-2026-04-22, **independent seeds** per Adam) clustered tightly in
cross-net SPRT — inconsistent with ±13 Elo variance. Either seed variance
was lower in the older pipeline, or some recent change has created/
amplified bimodality.

Move-ordering stats from the four replicas reveal **bimodal training
basins**:

| Net | First-move-cut % | Bench (nodes) | EBF | Basin |
|---|---:|---:|---:|---|
| gpu4 | ~76% | 1.4-2.3M | 1.7-1.8 | **A** |
| gpu5 | ~75-77% | 1.4-2.3M | 1.7-1.8 | **A** |
| gpu1 | ~71-73% | 3.2-3.9M | 1.82 | **B** (different GPU/CUDA) |
| gpu3 | ~73% | intermediate | — | A/B fence |
| (old) w20 | ~71-73% | 3.2-3.9M | — | **B** (single point in old data) |

Basin A: tight, fast-converging, low EBF. Basin B: wider QS, higher
node count. The single Basin-B point in the old wdl-sweep (w20) is a
hint that bimodality existed pre-C8 but was **less prevalent**.

#### Old-pipeline corroboration: warm-sweep matches wdl-sweep baseline

A second batch of old-pipeline nets — the **warm-sweep** (warm5/30/40/50,
HL-SCReLU + pre-C8, run independently of the wdl-sweep but in the same
era) — provides additional support for the "amplified, not created"
reading. Adam initially flagged it as a possible HL-CReLU set;
clarification: warm-sweep was HL-SCReLU. We were independently testing
HL-CReLU experimentally at the same time.

Bench/move-ordering across warm-sweep:

| Net | First-move-cut % | Bench | Basin |
|---|---:|---:|---|
| warm5 | tight, ~75-77% | low | A |
| warm30 | tight, ~75-77% | low | A |
| warm40 | wider, ~71-73% | high | **B** |
| warm50 | tight, ~75-77% | low | A |

1/4 in Basin B, matching the wdl-sweep's 1/5 (w20) — both old-pipeline
sets give a **~20-25% Basin-B outlier rate**. The current post-C8
hlcrelu era (3/4 = ~75% drift toward Basin B at SB200) is roughly
**2-3× wider** than this baseline. Bimodality did not appear from
nothing; it was amplified.

This also retires my earlier framing that the wdl-sweep WDL=0.15 result
was "measurement-noisy" — Adam's pushback ("the bench values bar w20
were all pretty similar") is correct on the data: 4/5 wdl-sweep nets
clustered in Basin A, w20 was a single outlier. WDL=0.15 was a real
read in the *old* pipeline. What's noisy is the *current* pipeline.

### H3 (Adam, 2026-05-01): Final LR is too low, exposing late-stage seed variance — STANDING

Adam's framing: v9 final LR (~2.4e-6) is much lower than v5's (~5e-6).
Going even lower than that caused material degradation in past
experiments — so we know we're at the bottom of a productive range.
The hypothesis: a too-low final LR provides too little noise for SGD
to average out per-batch variance, so the final 100-200 SBs leave
parameters drifting in whatever narrow basin they happen to be near
at the moment LR slows. Net result: late-stage basin landing becomes
random.

Adam self-flagged this as possibly recency bias from Titan's Hobbes
training-pattern analysis (`docs/training_patterns_2026-04-30.md`),
where Hobbes uses a much higher constant LR for stretches of training.
Worth treating as a real candidate anyway:

**Mechanism (concrete).** SGD noise scale ≈ LR / batch_size. To escape
a narrow minimum and land in a flatter basin, you need enough noise.
If LR drops too low while training continues, the trajectory effectively
freezes at whichever minimum was nearest when LR fell below the escape
threshold. The choice of minimum then depends on per-batch order
(seed-determined), not on training data structure.

**Compatibility with H2.** H3 is **not competing** with H2 — it could
be a *latent baseline*. Per Adam: final LR has not been altered between
the wdl-sweep era and now. So if H3 alone explained the variance,
wdl-sweep would have shown the same magnitude. It didn't (one Basin-B
point in five nets). The cleaner reading: low final LR provides a
**baseline level of basin-landing randomness**, and a recent change
(hlcrelu and/or C8) has **amplified** which seeds end up in which basin.
Combined, you get the current bimodality.

**Cheap diagnostic for H3.** If T1 (hlcrelu revert, see below) clears
the bimodality, low-final-LR is *probably* not the dominant factor —
the recent change owns the variance. If T1 leaves residual variance,
H3 becomes a candidate worth probing via a final-LR bump experiment
(e.g. SB200 ×3 at final_lr 5e-6 instead of 2.4e-6, same basin-landing
metric).

## What changed between the wdl-sweep era and now

Header-bit decode confirms two artifacts differ between the eras:

| Artifact | wdl-sweep (pre-2026-04-22) | current replicas |
|---|---|---|
| Magic / version | v9 | v10 |
| flags byte | 0b11000110 | 0b11100111 |
| `hl_crelu` field | 0 (HL = SCReLU) | 1 (HL = CReLU) |
| FT activation | SCReLU + pairwise | SCReLU + pairwise (bit 1 was repurposed in v10 — no real change) |

**Caveat on bit 1**: I initially read bit 1 differing between v9 and v10
as a real FT-activation change. Adam clarified the bit was repurposed
when we ran out of header space; both eras genuinely run SCReLU +
pairwise pre-multiply on FT. **The only confirmed activation difference
is HL: SCReLU → CReLU.**

Bullet config commit window (changes since wdl-sweep era):

- `b5590de` (2026-04-19): added `--hidden-activation` CLI flag (default
  screlu). Recent trains pass `--hidden-activation crelu`.
- `a8e2c7d` (2026-04-22): C8 fix part 1 — same-type-pair semi-excluded
  threats (R-R, B-B, Q-Q, N-N) row consolidation.
- `281efb3` (2026-04-23): MAX_THREAT_ACTIVE 256→512 (sample-budget bump).
- `12d11c4` (2026-04-24): `--group-l1-decay` CLI flag added (group-lasso
  scaffolding active even when decay=0).
- `62931d1` (2026-04-25): C8 fix part 2 — `phys_flip` x-ray same-type
  re-routing.

Adam confirmed: WDL weight, FinalLR, epoch count unchanged in this
window. The HL activation flip and C8 fix are the structural deltas.

## Empirical: per-feature L1-norm comparison

`scripts/threat_l1_compare.py` parses the .nnue header, locates the
threat int8 weight matrix, and computes per-row L1 norm (sum |w| over
ft_size). For each row, compute mean / std / coefficient-of-variation
across replicas. Cross-replica CV bucketed:

**Pre-C8 wdl-sweep (5 nets, varying WDL ∈ {0, 0.05, 0.10, 0.15, 0.20} +
independent seeds):**
```
CV [0,5%):    91.87%
CV [5,10%):    7.18%
CV [10,20%):   0.69%
CV [20,30%):   0.25%
CV [30,50%):   0.01%
CV ≥ 50%:      0.00%
max CV:        0.664
```

**Post-C8 hlcrelu replicas (4 nets, same WDL=0.15, same epochs, only
seed differs):**
```
CV [0,5%):    86.03%
CV [5,10%):    7.69%
CV [10,20%):   1.84%
CV [20,30%):   1.20%   (5× wider tail than pre-C8)
CV [30,50%):   0.86%   (86× wider)
CV ≥ 50%:      2.38%   (∞× wider)
max CV:        1.732
```

**Reading**: the mass of stable rows shifts only modestly (91.87% →
86.03% under 5% CV), but the **high-variance tail explodes**. 2.38% of
rows have CV ≥ 50% in the post-C8 set — i.e. their L1 weight magnitude
varies by more than half across replicas. In the pre-C8 set this bucket
was empty.

**Confound**: the pre-C8 set varies WDL while the post-C8 set holds WDL
fixed, so the pre-C8 sample is *under-controlled* (more noise sources)
yet still tighter. That makes the post-C8 broadening more striking, not
less.

This is consistent with H2 but doesn't disambiguate: **two confirmed
changes** between eras (hlcrelu, C8 fix). T1 disambiguates.

## Suspect ranking

In rough decreasing order of plausibility:

1. **HL CReLU activation** (most likely — H2 candidate). CReLU has dead
   zones at 0 and at saturation; which neurons die depends on init.
   Different deaths → different effective architectures → different
   basins. Mechanism is concrete and well-known in ReLU networks.
2. **C8 fix** (part 1 + part 2 — H2 candidate). Row consolidation of
   same-type-pair semi-excluded threats reduces parameter count for
   those rows; the `phys_flip` re-routing creates parameter-tying across
   x-ray pairs. Tied parameters can create flat directions much like
   factor's redundancy. Per the C8 review agent: side-effect plausibility
   medium — not a slam-dunk like activation, but real.
3. **Low final LR** (latent baseline — H3 candidate). v9 final LR
   ~2.4e-6 vs v5's ~5e-6. Probably not the *amplifier* (unchanged
   between eras) but plausibly the *baseline* — a higher floor of
   basin-landing randomness that combines with H1/H2 amplifiers.
   Probe only if T1/T2/T3 leave residual variance.
4. **MAX_THREAT_ACTIVE 256→512** — increased per-sample threat-feature
   budget. More features per sample = more parameter-update variance
   per step. Lower priority.
5. **Group-lasso scaffolding** (decay=0). Code path active even when
   decay=0; if any non-no-op behaviour leaks at zero decay, would affect
   every train. Lowest priority — purely defensive.

## Diagnostic plan

### Iteration-speed framing (Adam, 2026-05-01)

SB200 wall-clock is 6-8h. T1 fired at SB200 (already in flight on
gpu3/4/5 as of 2026-05-01) gives a clean A/B against the existing
4× SB200 hl-crelu replicas. For follow-up diagnostics where SB200
turnover would slow iteration (T2/T3), SB50 no-warmup is a viable
fast-screen — H1/H2 mechanisms (CReLU dead-zone, C8 row-tying)
commit to a basin *early*, so SB50 should fingerprint them.

H3 (low-LR-tail freeze) requires the long tail at low LR for the
trajectory to settle into a nearby random minimum. SB50 spends much
less time in the freeze regime — **does not expose H3**. T4 stays
SB200.

### T1 (IN FLIGHT 2026-05-01, SB200): revert hlcrelu

3× SB200 replicas on gpu3/4/5 with `--hidden-activation screlu`,
otherwise identical to current pipeline. Wall-clock 6-8h. Direct A/B
against the existing 4× SB200 hl-crelu replicas (gpu1/3/4/5) at the
same length where bimodality is known to manifest.

**Decision rule:**

- All 3 land in Basin A (bench/EBF/first-move-cut clustered, pairwise
  SPRTs within ~±3 Elo over 16K games — i.e. gpu4-vs-gpu5-tight):
  hlcrelu is the dominant cause. Revert HL to SCReLU. T2/T3 not needed.
- Still bimodal at SB200: hlcrelu cleared. Move to T2.
- Less bimodal but not as tight as wdl-sweep: H2 mechanism partially
  identified, residual variance worth attributing to H3 or another
  suspect — T2 still useful.

### SB50 fast-screen (future option for T2/T3)

If T1 clears hlcrelu and T2/T3 are needed, SB50 (no warmup) is a
viable fast-screen for the C8 / structural mechanisms — H1/H2 commit
to a basin early (CReLU dead-zones / C8 parameter-tying lock in
within a few SBs). Design when needed:

- **Calibration**: 3× SB50, no warmup, current config. Confirms SB50
  is long enough to reproduce bimodality.
- **T2/T3**: 3× SB50, no warmup, with the suspect reverted.

6 trains × ~2h, parallel on 3 GPUs → ~4h wall-clock total. ~3-4×
faster than SB200 sequential. Falls back to SB200 if calibration
shows SB50 too short.

H3 (final-LR tail) remains SB200-only — mechanism requires the
long low-LR tail.

### T2 (conditional, SB50): revert C8 fix only

Train 3 SB50 replicas with hlcrelu kept, C8 fix reverted (both parts).
Same decision rule, same SB50-fallback-to-SB200 logic if SB50
calibration suggests it's too short for the C8 mechanism.

### T3 (conditional, SB50): revert both

Train 3 SB50 replicas with both hlcrelu and C8 reverted — should
reproduce pre-C8 wdl-sweep behaviour. If still bimodal here, H2 is
cleared and the cause is H3 or deeper (MAX_THREAT_ACTIVE, group-lasso,
or something I haven't identified).

### T4 (conditional, SB200 — must be SB200): final LR bump

If T1-T3 leave residual variance, train 3 **SB200** replicas at
final_lr 5e-6 (v5 value, ~2× current), otherwise current pipeline.
SB50 is unsuitable for this test — H3 mechanism specifically requires
the long low-LR tail. If basin landing tightens at SB200, H3 is real
and worth a careful sweep. Expect SPRT to be ambiguous (final-LR
changes move strength too); the metric is replica-cluster tightness
on bench/EBF/first-move-cut, not net Elo.

## Methodology implications

Independent of which suspect wins, several rules tighten:

1. **Single-seed SPRT magnitudes for net-recipe experiments are noisy.**
   Past results to derate (treat direction-only):
   - C8fix-xray −10.7 Elo (#836)
   - Group-lasso 1e-2 +11.22 at SB200
   - Factor +3.3 Elo over non-factor (#782)
   - Any "recipe X = +/− N Elo" claim from one seed pair
2. **For deployment-candidate nets**: train 2 replicas, take median or
   worst, never best. Adds wall-clock cost but eliminates lucky-pick.
3. **For experimental probes**: 1 replica is enough for direction;
   don't quote magnitude.
4. **Loss-as-metric ceiling is much lower than assumed.** Two nets
   with 0.5% loss gap can have 60% gap in basic Q-vs-R discrimination
   (per `feedback_sb800_seed_variance_is_meaningful.md`).
5. **Move-ordering stats + bench as fast pre-screen**: first-move-cut
   % and bench separate Basin A from Basin B without an SPRT. ~10s per
   net to compute. Use this before committing fleet to replica SPRTs.

## Outputs and tooling

- **`scripts/threat_l1_compare.py`** — per-row L1-norm CV analysis
  across replica nets. Reusable for any future replica-set comparison.
- **`scripts/sf_rearbitrate_played_move.py`** — d24 SF re-arbitration
  for played_depth ≥ 18 sudden cases (separate diagnostic, but built
  in same session).
- **Memory**: `feedback_sb800_seed_variance_is_meaningful.md`,
  `project_sudden_bucket_is_bimodal.md` updated with these findings.
- **OB nets uploaded**: gpu1=A6F8418A, gpu3=F5834226, gpu4=551F8480,
  gpu5=6208612C.

## Connections

- `feedback_sb800_seed_variance_is_meaningful.md` — primary memory,
  evidence table.
- `feedback_loss_is_not_strength.md` — loss ≠ Elo at SB800 scale.
- `feedback_bench_stats_dont_predict_net_quality.md` — separate concern:
  bench measures search-given-eval, not eval. But bench *can* fingerprint
  training-basin landing, which is what we use here.
- `feedback_piece_value_test_design.md` — sigmoid-saturation caveat for
  cp-magnitude reads on LC0-trained nets.
- `feedback_sb_lengths_dont_cross_compare.md` — SB200 results don't
  translate to SB800 prod-readiness; combined with this, multi-replica
  SB800 is the only honest deployment test.
- `docs/training_patterns_2026-04-30.md` — Titan's Hobbes synthesis;
  source of LR-schedule perspective that motivated H3.
- `experiments.md` — section "2026-05-01 — non-factor SB200 4-replica
  seed-variance diagnostic (#896 / #897 / #898)".
