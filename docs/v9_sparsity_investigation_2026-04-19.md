# v9 Sparsity Investigation (2026-04-19)

## Context

Observed variance on OpenBench: ionos6 runs v9 ~80% faster than
ionos1-5,7,8 despite showing identical CPU topology (AMD EPYC Milan,
Zen 3, 32 MB L3). On v5 the same machines perform similarly, and PGO
helps v5 but hurts v9.

`perf stat` profiling on Hercules:

| Metric | v5 | v9 | v9/v5 |
|---|---:|---:|---:|
| Wall time | 1.78s | 6.35s | 3.57× |
| Instructions | 15.2B | 51.2B | 3.36× |
| IPC | 1.79 | 1.70 | 0.95× |
| **Branch-miss rate** | 2.27% | **1.95%** (better) | 0.86× |
| **L1d miss rate** | 16.5% | **26.0%** (worse) | 1.58× |
| LLC miss rate | 1.14% | 1.54% | 1.35× |

v9 is memory-bound, not branch-bound. Since both machines report
identical static cache (32 MB L3 on both), the per-machine variance is
runtime-dynamic: noisy-neighbor L3 contention, memory-bandwidth
contention, sustained-clock variability. Cloud VMs with memory-bound
workloads are known to vary dramatically by host load.

Optimization direction: shrink v9's memory footprint so the workload
becomes less memory-bound, reducing sensitivity to host-level
contention.

## Measurement: threat feature activation frequency

Added diagnostic test `threat_accum::incremental_tests::measure_feature_sparsity`.
Samples 11,940 positions from 5 diverse starting FENs × 30 games ×
up to 80 plies of random play. For each position, enumerates threat
features active from both POVs and tallies activation counts across
all 66,864 features.

### Results

```
Total positions sampled:   11,940
Total activations:         647,526
Avg features per POV-position: 27.1

--- Feature activation distribution ---
  0 hits    (never fired):    45,430 features (67.9%)
  1-9 hits  (very rare):      12,074 features (18.1%)
  10-99     (uncommon):        8,272 features (12.4%)
  100-999   (common):          1,012 features (1.5%)
  1000+     (hot):                76 features (0.1%)

--- Coverage ---
  Top 706 features capture 50% of activations
  Top 7,665 features capture 90% of activations
  Top 17,145 features capture 99% of activations
  Max activations on a single feature: 3,266 (0.5% of positions)
```

Distribution is extremely heavy-tailed. Top 1% of features do half the
work; 74% of features do essentially nothing.

### Memory implications

Weight matrix size: 66,864 features × 768 bytes/row (i8 weights) = **49 MB**.

| Action | Saved | Remaining matrix |
|---|---|---|
| Drop dead features (0 hits) | 33.3 MB (67.9%) | **15.7 MB** |
| Drop dead + very rare (<10 hits) | 42.1 MB (86.0%) | **6.9 MB** |
| Keep only 99%-coverage features | ~36 MB (74%) | **12.9 MB** |

Even conservative dead-only pruning drops the matrix under 16 MB —
comfortably fitting any realistic L3 slice. Aggressive pruning to
99%-coverage fits in 13 MB, and dropping all rare features (<10 hits)
drops to 7 MB (fits in L2 on some machines).

### Caveats

- Random self-play ≠ real game distribution. Opening theory and
  endgame positions may activate different feature subsets. Real
  tournament games likely activate MORE unique features, but the
  66.9% "dead" floor is unlikely to shift below 50%.
- 11,940 positions is a modest sample. A "rare" feature (1-9 hits) may
  actually be common in specific material imbalances or specific
  openings we didn't sample.
- A feature is "dead" in the current sample if it's an invalid piece-
  pair interaction at observed piece placements. Some dead features
  may be structurally impossible (e.g., pawn on 1st/8th rank) — those
  are permanently dead and can safely be dropped.

## Interpretation: sparsity is real and actionable

The measurement confirms the hypothesis from the perf profile:
- v9's memory footprint is the bottleneck on memory-constrained hosts
- The matrix is grossly oversized relative to its useful content
- Structured sparsity (whole-row pruning) directly shrinks allocated
  memory and thereby the working set

### Three sparsity types, ordered by cache impact

1. **Structured row sparsity (biggest win)**: whole feature rows that
   never fire or fire trivially. Directly reduces allocated weight
   memory. Training with L1 regularization on per-feature-row norm
   makes cold features collapse to zero; a convert-pass can then drop
   them entirely. **67.9% dead rate today with no pruning effort
   whatsoever.**

2. **Activation sparsity**: post-CReLU hidden-layer activations that
   are zero. Already improving via the creluHL training switch
   (SCReLU → CReLU on L1/L2, landing +4 Elo). Horsie's NNZ-bitmap
   sparse L1 matmul exploits this at inference — queued as a future
   Tier 2 port.

3. **Unstructured weight sparsity (lowest ROI)**: individual zero
   weights within otherwise-active rows. Doesn't help cache much
   because cache lines are 64 bytes wide regardless of internal
   zeros. Would require sparse-matmul kernels; complex for marginal
   win.

## Recommended action sequence

### Near-term (no retrain)

1. **Drop impossible features at converter time.** Some feature indices
   are structurally impossible (e.g., pawn-on-1st-rank as attacker).
   Identify via the piece-interaction map + square validity; zero out
   weights and compact the matrix. May recover ~10-30% of the dead
   67.9% with zero risk.

2. **Drop ALL zero-hit features from current production net.** The
   11,940-position sample shows 45,430 features never fire. Zero out
   their weight rows and compact. Expected +0 Elo (they contribute
   nothing) and -33 MB memory. Worth a confirm-SPRT to catch any
   single-game-phase surprises.

### Medium-term (retrain)

3. **Retrain v9 with per-feature-row L1 regularization.** Bullet
   config change: `loss += λ * Σ_feature ||W_row||_2`. λ chosen so
   training pushes cold-feature rows to exact zero while preserving
   hot features. Post-convert pruning captures the savings.
   Expected: ~70-80% features prunable after training → ~10-15 MB
   matrix → fits L3 on every machine.

4. **Reduce accumulator width from 768 → 512 or 640.** Brute-force
   memory shrink. Needs a full retrain. Strength hit uncertain but
   bounded; combined with feature pruning gives additive memory wins.

### Future (implementation)

5. **Sparse L1 matmul (Horsie pattern).** Measure post-CReLU L1
   activation sparsity on current net (separate measurement). If
   >50% activations are zero, NNZ-bitmap sparse matmul beats dense
   at v9's L1 size.

6. **Prefetch in delta-apply loop.** `_mm_prefetch` the next feature
   weight row while applying current. Helps L1d miss latency;
   smaller win given L3 is the real frontier.

## Diagnostic artifact

`cargo test --release measure_feature_sparsity -- --nocapture --ignored`

Full test at `src/threat_accum.rs:measure_feature_sparsity`. Extendable
for per-game-phase distribution, per-piece-pair, or per-bucket slicing
if we want finer-grained sparsity targeting later.

## Not measured yet (future passes)

- **Activation sparsity at L1 output** on current SCReLU net. Needed to
  size the sparse-L1-matmul opportunity.
- **Per-game-phase feature distribution** — openings vs middlegames vs
  endgames may have very different active-feature sets. If the sets
  are disjoint, one "pruning mask" might not cover all phases.
- **Distribution on real tournament games** (not random play). Random
  play overrepresents unusual positions; tournament games would show a
  tighter but possibly different sparsity profile.

---

## Update: weight-norm measurement reframes the picture

Added second diagnostic: `nnue::tests::measure_threat_weight_norms`
(runs on current production v9 net, `net-v9-768th16x32-w15-e800s800-xray.nnue`).
Computes per-row L∞ (max |weight|) and L1 (sum |weight|) for all
66,864 weight rows.

### Results

```
Net: 66,864 threat features × 768 hidden = 51.4M i8 weights (49.0 MB)

--- L∞ norm (max |weight| per row) ---
  0:        5,604 rows (8.4%)
  1-2:         12 rows (0.0%)
  3-7:        489 rows (0.7%)
  8-15:     1,965 rows (2.9%)
  16-31:    3,924 rows (5.9%)
  32-63:   28,351 rows (42.4%)
  64-127:  26,519 rows (39.7%)

--- L1 norm (sum |weight| per row) ---
  0:          5,604 rows (8.4%)
  1-99:           1 row  (0.0%)
  100-499:      666 rows (1.0%)
  500-999:    2,067 rows (3.1%)
  1k-3k:      1,555 rows (2.3%)
  3k-10k:    53,876 rows (80.6%)
  10k-30k:    3,095 rows (4.6%)

--- Compaction scenarios (drop rows with L∞ ≤ threshold) ---
  ≤ 0: keep 61,260 of 66,864 (91.6%) → 44.9 MB from 49.0 MB
  ≤ 2: keep 61,248 of 66,864 (91.6%) → 44.9 MB
  ≤ 5: keep 60,991 (91.2%) → 44.7 MB
  ≤ 8: keep 60,586 (90.6%) → 44.4 MB
  ≤ 16: keep 58,525 (87.5%) → 42.9 MB
```

### Reconciliation: activation ≠ weight sparsity

The two measurements tell very different stories:

| Metric | Sparsity |
|---|---|
| Features that NEVER activate in 12,000 random-play positions | **67.9%** |
| Features whose weight rows are all zero | **8.4%** |

Training produced **non-zero weights for ~91.6% of features**, even
features that our random-play sample never activated. Three possible
explanations:

1. **Training saw different positions.** Bullet trains on real-game
   data (T80, etc.); our random-play sample is a different distribution.
   Training saw openings and endgames we under-sampled, which activate
   the "never-fired" features.
2. **Gradient leakage via regularization.** Even with no activation
   signal, L2 regularization (standard in Bullet) pulls weights
   toward small non-zero values. This can keep weights at ~0.01 scale
   even for never-seen features.
3. **Structural overlap.** Some feature indices near dead ones share
   gradient updates (e.g., flipping orientation). Training updates
   propagate to neighbors even if the primary feature never fires.

### Implication

**The "simple load-time compact" idea saves only ~4 MB, not ~33 MB.**
Weight-row pruning at ε=0 drops 5,604 rows (8.4%) — the truly-zero
rows, likely structurally-impossible features the training never
produced gradients for. That's 44.9 MB vs 49 MB matrix.

Modest gain for the code effort (~50 lines of inference code +
SPRT). Probably not a strong "step change" at current hardware tier
boundaries — still well above 32 MB L3 fit.

**To get the 67% memory shrink I originally described, we need
training-side changes** — specifically L1 regularization on per-row
norms, which explicitly drives rarely-updated rows to zero. Without
it, L2 regularization alone leaves a long tail of tiny-but-nonzero
weights that can't be auto-pruned at load time.

---

## Cache-tier step-function considerations

(Added 2026-04-19 after discussion: memory-access performance is
NOT linear in matrix size — there are distinct knees at cache-tier
boundaries.)

For random-ish access patterns like threat weight lookup:

- **L1d (32 KB per core)**: unreachable for threats (one weight row
  alone is 768 B, ~42 rows max).
- **L2 (512 KB per core)**: would need to prune to < ~600 features.
  Only reachable via drastic compaction or hot-subset-only approaches.
- **L3 (16-32 MB, shared)**: currently 49 MB matrix spills. Shrinking
  below ~16 MB would cross this tier on most hardware.
- **DRAM (catastrophic)**: anything larger than L3 slice → ~300
  cycle penalty per miss.

Expected curve shape: **step function at tier boundaries, not
linear**. Shrinking 49 → 45 MB (8.4% row pruning) likely gives
little perceptible NPS change because we're still in the same
"partial L3 fit, partial DRAM spill" regime. Shrinking to 32 MB
(fully in L3) would give a visible step. Below that, returns
diminish until we hit L2 (another step, but requires ~100×
compaction — probably infeasible without radical redesign).

### Heavy-tailed access amplifies this

Our access pattern is heavy-tailed: top 706 features (541 KB) do
50% of activations. Those are L2-resident regardless of total
matrix size. **The cold-tail accesses dominate miss cost.**
Shrinking the cold tail (the 74% of features that do 1% of work)
has disproportionate benefit relative to their contribution.

## SPRT fleet-variance caveats

Testing memory-footprint changes via SPRT across OB workers is
hard because:

1. **Machines have different real cache behavior** even when lscpu
   reports identical specs (noisy-neighbor L3 contention on shared
   hosts; actual vs advertised sustained clock varies).
2. **OB already uses `scale_nps=250000` for v9** to normalize TC —
   if a memory shrink bumps NPS, scale_nps should be updated too
   so the SPRT measures Elo effect not "got more time per game".
3. **Elo effect of memory-footprint changes is indirect**: the
   change itself produces identical move choices; it changes speed
   → deeper search per unit time → Elo uplift. The uplift varies
   across machines because the speed uplift varies.

### Recommended validation path

Before committing to a memory-shrink experiment:

1. **Single-machine measurement first.** Run `perf stat` on
   Hercules before and after the proposed change. Measure NPS,
   L1d miss rate, LLC miss rate. Kill OB worker first.
2. **Single-cramped-machine measurement.** Same on a likely-constrained
   ionos host (via OB bench log scraping if we can't SSH). Confirm the
   improvement is bigger on the contended machine.
3. **Only then SPRT.** Use bounds [0, 5] or tighter. Expect the
   SPRT Elo to be the FLOOR of what individual fleet members would
   show — constrained machines gain more than beefy ones.
4. **Update `--scale-nps` after landing.** Post-shrink NPS goes up;
   scale_nps should match or SPRTs measure time-management artifacts.

## Revised action plan (given reconciled findings)

### Tier 1 — actually worth doing (short effort, bounded upside)

- **Retrain with per-row L1 regularization** on threat weights. Bullet
  config change: `loss += λ * Σ_feature ||W_row||_1`. λ chosen so
  training pushes cold-feature rows to exact zero rather than small
  non-zero. **This is the path to 50%+ structured sparsity.** Requires
  one training cycle but no inference-side changes until the new net
  is loaded and compact-at-load is added.

- **Load-time compact for zero-weight rows** (~50 LoC). Works with
  current net or L1-trained net. Gives ~8% on current net, could
  give 50%+ on L1-trained net. Start implementing, value depends on
  whether L1-trained net arrives.

### Tier 2 — bigger lift, longer path

- **Reduce accumulator width 768 → 512 or 640.** 33% memory reduction
  brute-force. Needs full retrain. Combined with L1 reg could shrink
  matrix under 16 MB → crosses L3 boundary.

- **Sparse L1 matmul for hidden layer activations** (Horsie NNZ pattern).
  Measurement still needed — see "Not measured yet" above. Impact is
  on the 16→32 hidden matmul (tiny memory) but could be a compute win.

### Tier 3 — architectural

- **Piece-interaction map redesign** to exclude structurally impossible
  features. Requires new feature index space, new net format, retrain.
  Probably drops 20-30% of features permanently. Bigger refactor.

## Revised expected impact

- **Load-time compact alone** on current net: ~4 MB saved, probably
  < 2% NPS. Not worth the SPRT signal to detect.
- **L1-regularized retrain + compact**: potentially 50%+ saved, crosses
  L3 boundary → could be 10-30% NPS on memory-constrained machines,
  smaller on beefy ones. Worth the training cycle.
- **Width reduction + L1 reg + compact**: potential 70%+ saved. Biggest
  shrink, biggest cache step if it reaches L2 fit or close.
