# Empirical layer-magnitude signatures from the 6-net SB50 isolation RR

Date: 2026-05-05
Context: post-rebase Bullet rebuilding stable training pipeline.
Probes are SB50 single-axis variations of v4-simple. Anchor is SB200
prod (pre-rebase Bullet, factor + hl-crelu + w15 + warm30 ON).

## Comparison table

| Net | RR Δ vs v4-simple | PSQ \|mean\| | threats \|mean\| | l0b \|mean\| | l1w \|mean\| | l2w (milli) | output_w \|mean\| | L1w 0% | L2w 0% | threats dead-rows | PSQ dead-rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anchor (SB200, factor+hlcrelu+w15+warm30) | +108 | 39.5 | 12.6 | 74.9 | 1.67 | 105.7 | 39.2 | **62.3%** | **62.3%** | **5226** | **600** |
| v4-simple (SB50, baseline, no factor, screlu, w0, warm0) | 0 | 37.9 | 19.5 | 64.1 | 0.85 | 108.0 | 27.5 | 9.0% | 9.0% | 0 | 0 |
| clip4-s50 (SB50, factor+clip4, all 4 axes ON) | -80 | **61.5** | 20.7 | 70.8 | 0.72 | 116.0 | 30.8 | 8.4% | 8.4% | 0 | 0 |
| iso-w15 (SB50, +w15 only) | -6 | 37.4 | 18.8 | 60.5 | 0.77 | 100.0 | **21.4** | 8.8% | 8.8% | 0 | 0 |
| iso-warm30 (SB50, +warm30 only) | -83 | 39.8 | 20.6 | 67.6 | 1.05 | **250.9** | 42.1 | 7.6% | 7.6% | 0 | 0 |
| iso-hlcrelu (SB50, +hl-crelu only) | -278 | 36.9 | 18.5 | 57.0 | 1.03 | 126.6 | 38.0 | 35.1% | 35.1% | 0 | 0 |

## Findings

### A. Factor + clip4 grows PSQ magnitudes 1.6× — and that's the mechanism

clip4-s50's PSQ |mean| is 61.5; every other net (including anchor) is
37-40. The factoriser sums shared l0f weights into per-bucket l0w, and
with the loosened clip ±4.0 (vs default ±0.99) the magnitudes can fully
express.

**Implication:** clip4 isn't a tweak; it's the UNLOCK for factor's
contribution. Without clip4, factor hits the ±0.99 clipping ceiling and
adds little net representational capacity. With clip4, factor's full
"shared structure across king buckets" representation lands. The
+220 Elo gap between iso-hlcrelu (factor off) and clip4-s50
(factor+clip4 on, despite hlcrelu also on) is consistent with this
unlock.

### B. iso-warm30 has anomalously bloated L2 layer (~2.5× others)

L2 |mean| is 250.9 milli for iso-warm30, vs 100-127 for every other
SB50 net. FT-stage layers (PSQ, threats, l0b) are within 5% of others
— this isn't a Signature-A "stuck layer" or Signature-B "wrong
convergence" pattern.

**Hypothesis:** warm30 schedule has 30 SBs of warmup → 20 SBs of cosine
decay. At SB50 total, that's 60% of training in low-LR warmup. The
network has only had 20 SBs of "real" learning. Late layers (L2) are
under-converged and still partly retain initialization-scale magnitudes
that haven't been pruned/refined yet.

**Implication:** the SB50 reading of warm30 isn't a quality signal at
all — it's a "training is incomplete" signal. The proper test is at
SB200+ where warm30 takes a healthy fraction of training (15-20%). At
SB200 with old-Bullet pre-rebase, warm30 was small-positive.

### C. Anchor's L1 is 62.3% zero; SB50 nets are 9-40% zero

The biggest single signal in the table. Anchor (SB200) has gone through
the low-LR tail and undergone heavy structural sparsification. SB50
nets (regardless of axis) are pre-sparsification.

**Implication:** the late-LR tail doesn't just refine weight values —
it **prunes structure**. The 62% sparsity in L1 is the network finding
which neurons are useful and zeroing the rest. This is consistent with
the v9 +88 Elo SB400→SB800 finding (sparse threats want lower final LR);
the lesson generalises.

**Lesson for hyperparameter understanding:**
- "training length" matters for two reasons: weight precision AND
  structural sparsification.
- low-LR tail acts like an automatic pruner.
- this is why SB50 can't tell us the answer for some questions — the
  pruning regime hasn't engaged.

### D. iso-hlcrelu shows mild Signature B (late-layer over-compensation)

vs v4-simple (only difference: hl-crelu vs hl-screlu):
- PSQ -3%, threats -5%, l0b -11% (FT-stage SAME within noise)
- l1w +20%, l2w +17%, **output_w +38%** (LATE layers all bigger)

This matches `feedback_weight_histogram_signatures.md`'s Signature B
("wrong convergence / over-compensation"): FT-stage normal,
late-layer magnitudes 30%+ bigger to compensate for upstream signal
deficit.

**Mechanism inference:** hl-crelu at the hidden layers (without factor)
creates a gradient-flow / capacity bottleneck. The network compensates
by amplifying late-layer magnitudes, but reaches a worse local minimum
overall. This is NOT a clean "frozen layer" or "broken op" signature —
it's "training succeeded but converged worse."

**Implication:** the post-rebase hl-crelu "bug" may not be a trainer
bug at all. With factor + clip4 enabled (clip4-s50), the upstream
capacity is sufficient and the Signature B pattern is mild (+11% on
output_w). hl-crelu may simply require factor to be load-bearing in
the recipe. **Test: factor+clip4+hlcrelu vs factor+clip4+hlscrelu at
SB50.** If close (≤30 Elo gap), hl-crelu is fine WITH factor. If still
huge gap, real trainer bug remains.

## Cross-cutting empirical lessons

1. **Late-LR tail is a pruner, not just a tuner.** Don't expect SB50
   nets to reveal anything about pruning-regime structure.
2. **Output_w |mean|** is the most diagnostic single number for
   detecting "over-compensation" / Signature B. Healthy v4-simple
   baseline: 27.5. Bloated: 38-42.
3. **PSQ |mean| > 50** signals factor is enabled and clipping is loose.
   Anchor's 39.5 hides a sparsified version (factor + clipping +
   pruning all interact at SB200).
4. **Threats dead-rows is purely a training-length artifact** in the
   pre-pruning regime. Anchor's 5226 dead threats are from the low-LR
   tail; SB50 nets all have 0.
5. **L2 bloat without late-layer Signature B** = training truncated /
   under-converged, not a quality issue. This is iso-warm30's pattern.

## Old-bullet SB200 paired data (added)

We have prior trainings for hl-crelu vs hl-screlu, no factor, w15, on
**old bullet at SB200**:

| Net | PSQ \|m\| | l1w \|m\| | l2w milli | output_w \|m\| |
|---|---:|---:|---:|---:|
| hlcrelu gpu4 | 23.0 | 1.18 | 112 | **44.0** |
| hlcrelu gpu5 | 24.5 | 0.93 | 103 | **44.9** |
| hlscrelu gpu3 | 23.7 | 0.89 | 163 | 38.9 |
| hlscrelu gpu4 | 22.6 | 1.58 | 217 | 39.7 |
| hlscrelu gpu5 | 22.0 | 1.95 | 300 | 40.3 |

**Three findings:**

1. **hlcrelu's "bigger output_w" Signature B pattern is INTRINSIC**,
   not a post-rebase trainer bug. Old-bullet SB200 hlcrelu nets show
   output_w +12% vs hlscrelu. New-bullet SB50 iso-hlcrelu shows +38%.
   Same direction, much amplified at SB50.

2. **hlscrelu has 85% L2w spread across replicas** (163-300 milli);
   hlcrelu only ~10% (103-112). SCReLU permits a wider L2 magnitude
   range that all trains OK; CReLU forces a tighter convergence
   target.

3. **Factor's empirical PSQ contribution is +16** (anchor PSQ 39.5,
   nonfactor SB200 PSQ ~23). Quantifiable evidence that factor adds
   ~70% PSQ representation magnitude.

**Conclusion (UPDATED 2026-05-05 night, Adam pushback): both factor
AND hl-crelu are most likely broken on new bullet.**

Earlier draft of this doc speculated hl-crelu was just a "recipe rule"
needing factor + length. Adam corrected: that's an extrapolation we
don't have evidence for. The actual evidence we have:

- iso-hlcrelu SB50 on new bullet: -280 vs v4-simple. **Confirmed bad
  on new bullet.**
- hlcrelu/hlscrelu pair-trainings at SB200 used OLD bullet (May 1-3).
  They show hl-crelu had a mild Signature B pattern (output_w +12%)
  on old bullet, where it was net +5 Elo. They tell us nothing about
  whether SB200 hl-crelu on new bullet would fail the same way, fail
  differently, or fail much worse.
- First SB200 trainings on new bullet were disasters (~800 Elo
  behind). Multiple bugs fixed since (init line drop, others).
  We switched to SB50 to iterate faster, not because SB50 is the
  right scale to test these features.

**Most parsimonious reading: factor and hl-crelu are both broken on
new bullet. Need to root-cause both, not extrapolate.**

The clip4-s50 (-80 vs v4-simple) result is consistent with both being
broken — its all-4-axes-ON state is below v4-simple, not above it. If
factor were healthy on new bullet, clip4-s50 should land at-or-above
v4-simple even with hl-crelu's penalty stacking on top.

**Critical next test (#219): factor + clip4 + screlu (no hl-crelu) at
SB50 on new bullet.** This is the cleanest single read we can get of
factor's contribution on new bullet.
- If +200 vs v4-simple: factor works on new bullet. (Then we still
  need to root-cause hl-crelu — it's broken too.)
- If near v4-simple (or below): factor is broken on new bullet.
  Both axes need separate root-cause and fixes. The "+5 Elo
  hl-crelu on old bullet" expectation can't be assumed to recover
  on new bullet just because we fix factor.

## Open empirical questions to investigate

- **Q1: Is factor's +220 Elo signal real (silver bullet) or
  capacity-coupled to hl-crelu?** Test #219 (factor+clip4+screlu)
  isolates this.
- **Q2: Is hl-crelu itself broken (-280 in iso) or capacity-limited?**
  Test #220 (factor+clip4+hl-crelu vs factor+clip4+hl-screlu) decides.
- **Q3: Does w15's "small positive" prior actually exist on new
  Bullet?** Need 10K-game net-vs-net SPRT (paired SB50 or SB200 nets).
- **Q4: At what SB length does warm30 transition from negative to
  positive?** Paired SB100 + SB200 + SB400 ablation could trace the
  curve.
- **Q5: How does the L1 sparsification fraction grow with SB length?**
  Run dump-net-stats on intermediate snapshots (s50, s100, s200, s400,
  s800) of the same training run to trace structural sparsification.

## Caveats

- Single-replica reads. Seed variance ~10-15 Elo per replica means
  ±20-25 Elo in pairwise comparison.
- iso-w15 / iso-warm30 / iso-hlcrelu are sister probes from the same
  isolation experiment, so seed variance differs from anchor's
  pre-rebase replica.
- The "0% L1w / L2w" columns are reported using zero-count from
  histograms; might be conflated under quantisation rounding.
