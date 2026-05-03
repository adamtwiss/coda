# Threat-feature histogram findings — 2026-05-01

Per-feature-index hit counts collected during `./coda bench 13` (~967k
nodes, ~1.2M apply_threat_deltas calls, ~10M total deltas), gated behind
`--features profile-threats`. CSV dump at `/tmp/threat_feature_hits.csv`.

## Summary

| Bucket | Count | % of full space (66864) | % of hits |
|---|---:|---:|---:|
| Dead (0 hits) | 31,108 | 46.5% | 0.0% |
| Active (≥1 hit) | 35,756 | 53.5% | 100% |
| Top 10% of active (3,575) | 3,575 | 5.4% | **84.1%** |
| Top 1% of active (357) | 357 | 0.5% | **39.3%** |

**Heavy Pareto.** ~5% of the threat space carries ~84% of the cache reads.

## Memory footprint by tier

| Slice | Size | Cache-tier on Zen 5 |
|---|---:|---|
| Full 66864-row matrix | 49.0 MB | L3 (96 MB) |
| Active 35756-row matrix | 26.2 MB | L3 |
| Top-10% active (3575 rows, 84% of hits) | 2.6 MB | L3 (above 1 MB L2) |
| Top-1% active (357 rows, 39% of hits) | 268 KB | **L2-resident** |

## Implications for "drop dead features"

**This experiment is the wrong target.** Dead rows aren't read during
inference — they don't enter L1/L2/L3 because `threat_index()` never
emits their indices. Removing them from the matrix shrinks physical
allocation 49 MB → 26 MB but yields **zero L1/L2 cache benefit** since
those rows weren't being fetched in the first place. The cache pressure
comes from the hot 2.6 MB working set being scattered across a 49 MB
address range (TLB / spatial-locality), not from dead-row pollution.

## What WOULD reduce cache pressure

The lever is **clustering the hot 5% to a contiguous low-address region**,
not dropping the dead 47%. Two paths previously explored:

1. **Hot-feature frontload** (feature/threat-feature-frontload) —
   permute the matrix at .nnue load time so the hottest features land at
   indices 0..N. Required a 67k-entry runtime perm table. **Bench-negative
   −2.5% on Zen 5** because the per-delta perm-table lookup added more
   cycles than the cache savings recovered. Dropped 2026-04-30.

2. **Encoding-side clustering** — design a `threat_index()` arithmetic
   that naturally produces low indices for hot threat tuples. Hot-feature
   clustering is *data-dependent* (corpus statistics), not formulaic, so
   any encoding-side fix has to be statistics-aware (build a perm table
   from a training corpus, freeze it at training time, embed it in the
   .nnue file). Equivalent in mechanism to (1) but with the perm-lookup
   amortized at training instead of inference time — same lookup cost
   per delta if implemented naively.

## Genuinely promising directions (training-side)

These cost more wall-clock (need retrains) but attack the right structural
problem:

- **Smaller hidden_size (768 → 512 or 384)** — each row goes from 12 cache
  lines to 8 or 6. Matrix shrinks proportionally. Eval-quality risk;
  needs a full retrain + SPRT.
- **i4 weight quantization** — halves bytes-per-row. Doubles "rows per
  L2 cache line." Matmul kernel changes too.
- **Sparsity-promoting training** (group lasso on FT rows, magnitude
  pruning) — encourages near-zero weights that can be dropped from the
  forward pass via row-sum thresholding. Targets the right working
  set directly.
- **Re-shape encoding** — fewer (attacker, victim) pairs, fewer king
  buckets, mirror more aggressively. Each restructuring trades feature
  resolution for a smaller matrix.

## Status

- Histogram tooling lives behind `--features profile-threats`. Re-run
  any time with `cargo build --release --features 'embedded-net profile-threats'`
  followed by `./target/release/coda bench 13`. Output dumps to
  `/tmp/threat_feature_hits.csv` (or `$CODA_FEATURE_HISTOGRAM`).
- Drop-dead-features experiment: **not pursued** based on this
  finding (cache benefit ≈ 0).
- Next NPS lever in scope: search-side (REGS=24 PSQ + REGS=16 threat,
  SPRT #926). Beyond that, cache improvements likely need training-side
  collaboration.
