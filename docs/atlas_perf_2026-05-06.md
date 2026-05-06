# Atlas (Zen 1, AVX-2) — Fresh NPS Decomposition (2026-05-06)

Re-baseline + drill-down on Atlas after the AccDataStack + MaybeUninit
landings (May 3, +6.6% on Atlas per the `nps_microbench_hostdata.md`
fleet table). Targets the AVX-2-only fleet (Atlas + most OB workers +
lichess host).

Companion to `coda_vs_reckless_nps_2026-04-23.md` (decomposition framing)
and `nps_structural_findings_2026-05-01.md` (Zeus structural findings).

## Atlas spec recap

AMD EPYC 7351P, Zen 1 / Naples (2017). 16C/32T. **AVX-2 only — no AVX-512.**
L1d 32KB/core, L2 512KB/core, **L3 8MB *per chiplet*** (NUMA-4, threats
matrix doesn't fit on any single chiplet). 132 GB RAM. Idle CPU,
`perf_event_paranoid=1`.

## Headline numbers

### Search bench (depth 13, single-thread, both engines on current trunk)

| Engine | Bench NPS | Ratio |
|---|---:|---:|
| Coda (post-AccDataStack) | **322 K** | — |
| Reckless | **484 K** | **R/C 1.50×** |

**Down from R/C 1.81× on Titan April 23** — the AccDataStack +
MaybeUninit + register-tiling + eval-only-TT-writeback wave closed
~17% of the gap on this host class. This contradicts the "we're not
making progress on the Reckless gap" framing — Atlas is materially
better than the April-23 Titan baseline.

### Eval-bench (incremental, 800K reps, AVX-2 path)

```
Coda evals/sec: 711 K
IPC: 2.02 (vs Atlas search-bench 1.54 — pure eval is less branchy)
L1-dcache miss rate: 11.49% (vs Reckless Zeus AVX-2 0.55-1.07%)
Branches: 1.28 B / 800K = 1605/eval (vs Reckless Zeus 137M / 800K = 171/eval = 9.4× ratio)
Branch miss rate: 0.71% (predictor is fine)
```

The 9× branches/eval ratio is consistent with the
`nps_structural_findings_2026-05-01.md` "10-arm dispatch tree +
`#[target_feature]` inline barriers" analysis, not closed by anything
since.

## Hotspot decomposition (perf record, eval-bench --mode incremental)

```
72.75% forward_with_l1_pairwise_inner    ← THE bottleneck on AVX-2
14.93% simd_acc_fused_avx2               ← FT incremental update
 3.17% materialize                        ← Lazy accumulator computation
 1.48% make_move
 0.84% build_dirty_piece
 0.75% unmake_move
```

**On Zeus (Zen 5 AVX-512+VNNI), the same function takes 36.13% of
incremental eval cycles. On Atlas it's 72.75%.** The gap is mostly
because:

1. AVX-2 has no VPDPBUSD (VNNI) — int8 dot product needs
   VPMADDUBSW + VPMADDWD + VPADDD (3 ops vs 1).
2. AVX-2 has 16 YMM registers vs 32 ZMM — register tiling is
   tighter; current `dense_l1_avx2` runs 4-way interleaved
   accumulators where the AVX-512 path runs 24-way.
3. Atlas's 8 MB-per-chiplet L3 makes the dense iteration of
   the 12-24 KB L1 weight matrix per call sit at the edge of L1
   eviction.

This means: **on the AVX-2 fleet, optimising
`forward_with_l1_pairwise_inner` alone is roughly 2× the leverage
it has on Zeus.** A 30% cost reduction here would be ~22% NPS gain
on Atlas vs ~11% on Zeus.

## Rebuild-by-cause split (new instrumentation)

`coda bench` with new `stats_rebuild_*` counters separating cause:

```
NNUE full rebuilds:    1,265,500 (48.59% of evals)
  by cause:  king-bucket=399,714 (31.59%)  root=0 (0.00%)  chain-break=865,786 (68.41%)
NNUE incremental:      1,338,871 (51.41% of evals)
```

**68.41% of rebuilds are chain-breaks** (parent ply not computed —
lazy-accumulator chain failure). Only 31.59% are forced king-bucket
crossings.

**865,786 chain-break rebuilds out of 4,815,017 nodes = 18% of all
nodes** pay a Finny-diff refresh that walkback-style chain-forward
could partially eliminate.

### Walkback distance distribution (where the chain breaks land)

From `--features profile-materialize`:

```
walkback distance: d1=6.3%  d2=27.9%  d3-d8=~30%  d≥9=35.6%  no-anc=0.0%
```

Chain-break is bimodal:
- d=2 (27.9%) → would benefit a LOT from chain-forward (~5 deltas vs
  ~16-20 row Finny diff).
- d≥9 (35.6%) → chain-forward (~24 deltas) loses to Finny diff
  (~16-20 rows) at the long-distance tail.

A bounded walkback (e.g. `WALKBACK_LIMIT=5-8`) covers ~60% of the
chain-break cases and falls back to Finny diff for the long tail.
That's exactly what `c48617d`/`5fa55b5` were exploring on
`tune/psq-refresh-perpov-c8fix` — never SPRT'd, and now too stale to
rebase (1306 inserts diff vs main on `nnue.rs`+`search.rs`).

## What's actionable on Atlas (and the AVX-2 fleet by extension)

Lever ranking specific to the AVX-2 fleet, post-AccDataStack:

### Tier 1 — biggest expected lever

**1. Reduce `forward_with_l1_pairwise_inner` work for AVX-2.** Highest
   share (72.75%) on this host class. Two sub-levers:

   - **a. Drop `#[target_feature]` from the function body.** Single
     biggest unrealised lever per `nps_structural_findings_2026-05-01.md`.
     Per-ISA build (`make avx2`/`make avx512vnni`) would let LLVM
     inline the SIMD primitives flat into the function and across the
     caller. Effort: medium (build matrix). Expected: 5-12% NPS on
     AVX-2 hosts specifically.

   - **b. AVX-2 register-tiling raise.** `dense_l1_avx2` uses 4-way
     interleaved accumulators. AVX-2 has 16 YMM registers; current
     working set is well under that. A REGS=8 or REGS=12 raise (paired
     with the existing FT-side raise from #930) may extract more
     pipelining at the matmul. Effort: small. Expected: 2-5%.

### Tier 2 — chain-break reduction

**2. PSQ walkback reattempt — bounded, on current trunk.** 18% of
   nodes pay rebuild cost. Bounded walkback (`WALKBACK_LIMIT≈6`,
   per-pov) covers the d≤8 majority. The `c48617d`/`5fa55b5`
   approach is sound; a fresh re-implementation on current trunk
   is the action. Effort: 1 day. Expected: small-to-modest (the
   FT side already does Finny diff cheaply; main savings are
   reduced chain breakage in lazy-accumulator-heavy subtrees).

### Tier 3 — cache hygiene next pass

**3. NUMA pin the production binary on multi-NUMA Zen 1 / Zen 2 hosts.**
   Atlas has 4 NUMA nodes. SMP search across nodes would suffer
   cross-node L3 latency. For single-thread bench measurements,
   `numactl --cpunodebind=0 --membind=0` would cut variance and
   may shift NPS up several percent at the bench level. Validate:
   `numactl --cpunodebind=0 --membind=0 ./coda bench`.

### What's NOT actionable

- Threat-side walkback: already in main (`recompute_threats_if_needed`),
  works correctly.
- Eval-only TT writeback: already merged at +14.7 Elo (#713).
- Direct-check LMP carve-out: already merged (+2.5 Elo, #708).
- Sparse-first L1 matmul: tested at -4% bench on current density model
  (`nps_structural_findings_2026-05-01.md` PM update 2026-05-01).
- Manual prefetch in apply loop: H0 (#719/#721) — HW prefetcher
  already optimal on current code.

## Methodology — Atlas run

```bash
# perf permission
sudo sysctl kernel.perf_event_paranoid=1

# Atlas search bench
./coda bench

# Atlas eval-bench (incremental hotspot)
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses \
  ./coda eval-bench --mode incremental --reps 100000 \
  -n net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-factor.nnue

perf record -F 4000 -g -o /tmp/coda_incr.data --quiet -- \
  ./coda eval-bench --mode incremental --reps 200000 \
  -n net-v9-768th16x32-kb10-w15-e800s800-crelu-C8fix-factor.nnue
perf report -i /tmp/coda_incr.data --no-children --stdio --sort=overhead,symbol

# Reckless search bench (no patch needed for headline NPS)
cd ~/chess/engines/Reckless && echo "bench" | ./target/release/reckless

# Cause-split rebuild stats (this commit)
./coda bench   # bench output now prints "by cause: king-bucket=... chain-break=..."
```

## Cross-references

- `docs/coda_vs_reckless_nps_2026-04-23.md` — original decomposition (Zeus).
- `docs/nps_structural_findings_2026-05-01.md` — Zeus L1-matmul-restructure
  findings; concluded the sparse-first reframe was wrong, real lever is
  inline-barrier removal.
- `docs/nps_microbench_hostdata.md` — fleet table; Titan = Atlas spec.
- `docs/reckless_commit_catalog_2026-05-01.md` — 199-commit walk; AVX-2-
  applicable items still untried (e.g. byteboard-splat AVX-2 #45d9cc5a
  +2.27 STC) but smaller share now that AccDataStack is in.

## Open questions

- Is the 9× branches/eval gap quantifiable at the *function* level?
  i.e. how many of the 1605 branches/eval are in `forward_with_l1_pairwise_inner`
  vs elsewhere? Would tell whether dispatch-tree consolidation captures
  the win or whether it's spread across other helpers.
- Does `numactl --cpunodebind=0 --membind=0` shift Atlas bench NPS? Cheap
  to test (one command). If it does meaningfully, fleet-wide NUMA pinning
  could be a free lever without code changes. **Tested 2026-05-06**:
  median 329K pinned vs 318K unpinned across 3 runs each = ~3% gain,
  inside per-run noise (range 312K-350K). Inconclusive at 3 runs each;
  worth re-testing with larger sample if pursued.
- Reckless's incremental eval-bench wasn't measurable here (their patch
  is stale for current Reckless source). For a clean apples-to-apples
  on current trunks, the `scripts/reckless_evalbench.patch` needs
  refreshing against `666cef53` (or wherever Reckless trunk is).
