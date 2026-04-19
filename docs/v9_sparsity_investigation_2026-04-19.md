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
