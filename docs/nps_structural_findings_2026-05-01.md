# NPS Structural Findings — 2026-05-01

Companion to `docs/coda_vs_reckless_nps_2026-04-23.md` (decomposition + lever
ranking) and `docs/reckless_commit_catalog_2026-05-01.md` (199-commit walk).
This doc is the answer to the question we kept circling: **after porting
the obvious wins, why are we still at ~50% of Reckless's NPS?**

The short answer: **the 1.79× per-eval instruction count and the 19× cache-
references gap have the same root cause — sparse-first vs dense-first L1
matmul, made worse by runtime SIMD dispatch with `#[target_feature]` inline
barriers.** Coda's L1 matmul reads all 768 inputs; Reckless's reads only
the ~85 non-zero inputs (89% sparse). Coda has ten dispatch branches that
LLVM cannot eliminate; Reckless has one inlined SIMD path.

Until that is restructured, individual SIMD ports keep delivering +1-3 Elo
each — real, but compounding off the same ceiling.

## Trigger

After ~6 weeks of incremental NPS work (AVX-512 `apply_threat_deltas`,
setwise movegen, pair-unroll, TT prefetch, accumulator reorder), Coda is
still at ~50% of Reckless's NPS on Zen 5. The Reckless commit catalog
walk (199 commits) confirmed there is no single missing port that closes
the gap. So we stepped back: what **fundamental** thing are we missing?

## Phase 1 — Compile-time SIMD dispatch validation

`coda_vs_reckless_nps_2026-04-23.md` flagged runtime dispatch as a
hypothesis (4.7× more branches per incremental eval). Phase 1 was a quick
A/B: replace one `if self.has_avx512` with a `cfg(target_feature = "avx512f")`
in the hot path and remeasure.

**Result: −1% NPS on Zen 5, not the predicted +5-9%.**

The compile-time/runtime distinction at the dispatch *check* is not the
real lever. The real barrier is the `#[target_feature(enable = "avx2")]`
attribute on the SIMD bodies themselves (e.g. `forward_with_l1_pairwise_inner`
at `nnue.rs:2564`). That attribute marks the function as having a
different ABI/feature set than its caller, which **prevents LLVM from
inlining**. Once the SIMD body is non-inlined, no amount of changing the
dispatch check shape recovers the cost.

**Lesson:** the dispatch overhead is not 10 branch predictions per eval.
It's that the entire SIMD body is opaque to the surrounding optimizer,
including const propagation, register allocation, and tail-call
specialisation. A real "compile-time dispatch" port would mean removing
`#[target_feature]` and building separate binaries per ISA the way
Reckless does — much bigger structural change than the original sketch.

## Phase 2 — Fresh perf decomposition on current trunk

After all the recent ports landed, we re-ran `perf record` against
`coda eval-bench --mode incremental` on the production v9 net and
`coda bench 13`.

| Function | % of cycles | Change vs 2026-04-23 |
|---|---:|---|
| `forward_with_l1_pairwise_inner` | **36.13%** | was 3.27% |
| `simd_acc_fused_avx512` | 15.44% | unchanged |
| `materialize` | 6.95% | down (lazy push wins) |
| `simd512_pairwise_pack_fused` | 4.80% | unchanged |
| `make_move` | 4.20% | unchanged |
| (other) | 32.48% | — |

Two readings of the same data:

1. **The L1 matmul is now the dominant single hotspot.** Earlier
   measurement at 3.27% was misleading — that was before the threat
   path got SIMD-fast and before lazy-push reduced `materialize` churn.
   With the threat-side and accumulator-fused-update paths optimised,
   the L1 matmul stands out: > 1/3 of every eval cycle.
2. **The 1.79× instruction-count gap from `coda_vs_reckless_nps_2026-04-23.md`
   §3 is unchanged.** All the SIMD ports landed since 2026-04-23 reduced
   *cycles* (each port made some piece faster) without reducing
   *instructions executed* (the work shape stayed the same). When the
   instruction-count gap doesn't move, you can't be on the right axis.

Coda still does 1.79× more instructions per incremental eval than
Reckless, with cache-references at 19× and L1-dcache-load-misses at
~28× the rate. The eval-side cache gap from the previous doc is fully
present on current trunk.

## Structural diff: how the L1 matmul actually differs

This is the structural finding. Both engines do "pairwise-CReLU pack →
int8 L1 matmul → activation → L2 → output". The diff is in the **shape
of the int8 L1 matmul** in the middle.

### Reckless `propagate_l1` (`Reckless/src/nnue/forward/vectorized.rs:55`)

```rust
pub unsafe fn propagate_l1(
    ft_out: &Aligned<[u8; L1_SIZE]>, nnz: &[u16],  // ← nnz indices precomputed
    bucket: usize, parameters: &Parameters,
) -> Aligned<[f32; L2_SIZE]> {
    const CHUNKS: usize = 4;
    let mut pre_activations = Aligned::new([simd::zeroed(); L2_SIZE / simd::F32_LANES]);
    let packed = std::slice::from_raw_parts(ft_out.as_ptr().cast::<i32>(), L1_SIZE / CHUNKS);

    let mut pairs = nnz.chunks_exact(2);
    for pair in &mut pairs {
        let index1 = *pair.get_unchecked(0) as usize;
        let index2 = *pair.get_unchecked(1) as usize;

        let input1 = simd::splat_i32(*packed.get_unchecked(index1));
        let input2 = simd::splat_i32(*packed.get_unchecked(index2));

        let weights1 = parameters.l1_weights[bucket].as_ptr().add(index1 * L2_SIZE * CHUNKS);
        let weights2 = parameters.l1_weights[bucket].as_ptr().add(index2 * L2_SIZE * CHUNKS);

        for j in (0..L2_SIZE).step_by(simd::F32_LANES) {
            let weights1 = *weights1.add(j * CHUNKS).cast();
            let weights2 = *weights2.add(j * CHUNKS).cast();
            let vector = &mut pre_activations[j / simd::F32_LANES];
            *vector = simd::double_dpbusd(*vector, input1, weights1, input2, weights2);
        }
    }
    // ... tail handler for odd nnz count, then dequant+activation
}
```

Properties:
- **Sparse-first.** Iterates over the precomputed `nnz: &[u16]` list of
  non-zero 4-byte input chunks (~85 of 768 at 89% sparsity). The dense
  (zero) chunks are never touched.
- **Pair-unrolled.** Two NNZ entries per loop body via `chunks_exact(2)`,
  feeding `simd::double_dpbusd` (two VPDPBUSDs back-to-back). Hides the
  4-cycle latency of VPDPBUSD across two independent accumulator chains.
- **Single inlined SIMD path.** `simd::splat_i32`, `simd::double_dpbusd`,
  `simd::F32_LANES` are all `cfg(target_feature)`-dispatched at the
  *primitive* level. The body of `propagate_l1` itself is one path. LLVM
  inlines the primitives flat into the body. No `#[target_feature]`
  attribute; no inline barrier.
- **Compact memory footprint.** Weights for a single bucket at L2_SIZE=16,
  L1_SIZE=1536 are `1536 * 16 * 4 = 96 KB` — fits in L2. Sparse access
  touches `~85 NNZ × 16 outputs × 4 bytes = ~5 KB` of weight data per
  call, all hot in L1.

### Coda `forward_with_l1_pairwise_inner` (`src/nnue.rs:2565`)

```rust
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
unsafe fn forward_with_l1_pairwise_inner(...) -> i32 {
    // ... pairwise pack into stm_pw / ntm_pw (skipped) ...

    #[cfg(target_arch = "x86_64")]
    if self.has_avx512_vnni && !self.l1_weights_sparse.is_empty() && l1 == 16 && pw % 4 == 0 {
        crate::sparse_l1::dense_l1_avx512_vnni(...);
    } else if self.has_avx512_vnni && pw % 64 == 0 && !self.l1_weights_8t.is_empty() {
        // dense, row-major path
    } else if self.has_avx_vnni && !self.l1_weights_sparse.is_empty() && l1 <= 16 && pw % 4 == 0 {
        crate::sparse_l1::dense_l1_avx_vnni(...);
    } else if self.has_avx512 && pw % 64 == 0 && !self.l1_weights_8t.is_empty() {
        // dense AVX-512 row-major
    } else if self.has_avx2 && !self.l1_weights_sparse.is_empty() && l1 <= 16 {
        crate::sparse_l1::dense_l1_avx2(...);
    } else if self.has_avx2 && pw % 32 == 0 && !self.l1_weights_8t.is_empty() {
        // dense AVX-2 multi-neuron x4
    }
    // ... three more cfg-gated `if !(...) { scalar fallback }` blocks ...
}
```

Properties:
- **Dense-first.** The "best" path on Zen 5 is `dense_l1_avx512_vnni`
  (despite the file name `sparse_l1.rs`). It iterates **all** 768 input
  chunks per call, multiplying through zeros. We have `sparse_l1_avx2` /
  `sparse_l1_avx512_vnni` implementations available, but they were
  measured at −6% NPS at H=16 in 2026-04-15 testing and disabled by
  default.
- **Ten dispatch paths.** Six `else if` arms on x86_64 + scalar fallback
  + aarch64 path + scalar non-x86/aarch64 path + the inlined call sites.
  Each `if self.has_avx512_vnni` reads a heap-allocated boolean.
- **`#[target_feature(enable = "avx2")]` on the outer function.** Forces
  the function to be its own ABI island; LLVM cannot inline the SIMD
  primitives into the caller, and cannot inline this function into its
  caller (`forward_with_l1_pairwise`).
- **Larger memory footprint per call.** `l1_weights_8t` for v9 at
  `pw=384, l1=16`: `2 * 384 * 16 = 12,288 bytes ≈ 12 KB` per perspective,
  ~24 KB both sides. Dense iteration touches all of it. Reckless's
  sparse iteration touches ~5 KB on the same op-point.

### Why this produces both gaps simultaneously

The 1.79× instruction count gap and the ~19× cache-references gap are
not two separate problems. They're two symptoms of dense-first iteration:

| Symptom | Mechanism |
|---|---|
| **1.79× instructions** | Dense matmul does ~9× more multiply-accumulate work (768 inputs vs ~85 NNZ). Even highly vectorised, the work is ~9× larger; SIMD width and pair-unrolling close some of that, leaving the observed 1.8×. |
| **19× cache-references / 28× L1 miss rate** | Dense access pattern reads the entire weight matrix per call, exhausting L1 working set. Sparse access pattern touches only the rows of NNZ inputs — a working set ~5× smaller fits in L1. |

This is also why "fix the cache layout" alone wouldn't fully close the
gap. You can flatten `AccEntry`, frontload hot features, prefetch the
next weight row — and you'd still execute 1.79× more instructions on the
dense matmul because the work axis is wrong.

### Why the previous sparse experiment underperformed

2026-04-15 measured `sparse_l1_avx2` at −6% vs dense at H=16 and the
sparse path was disabled by default. Reading that result against
Reckless's success suggests the regression was not from sparse-first
being wrong, but from sparse-first **alongside** Coda's existing dispatch
overhead and `#[target_feature]` barriers. Reckless gets the win because
the entire kernel (NNZ scan + sparse matmul + dequant) is one inlined
fast path; Coda's sparse path was bolted into a function that already
pays the inline-barrier tax, where the marginal sparsity win didn't
exceed the residual overhead.

In other words: **the sparse path needs to land as part of restructuring
the surrounding code, not as an alternative branch inside it.**

## Cache as the same axis, not a separate one

`coda_vs_reckless_nps_2026-04-23.md` §3 framed cache behaviour as a
distinct lever (flatten AccEntry, hot-feature frontload, prefetch). Those
levers are still real, but the dominant cache cost is in the L1 matmul
itself, and the right way to fix it is to **read less**, not to
**arrange the read better**.

Per-call working set on v9 production net (Zen 5, AVX-512+VNNI):

| Operation | Reckless | Coda | Gap |
|---|---:|---:|---:|
| L1 matmul weight bytes touched | ~5 KB | ~24 KB | 4.8× |
| L1 matmul input bytes touched | ~340 B | ~3 KB | 8.8× |
| L1-dcache-load-misses (microbench) | 5.6 M / 800k evals | 304 M / 800k evals | 54× |
| L1 hit rate | 99.45% | 84.28% | — |

The miss-rate gap (54× on the matmul-dominant microbench, 28× on the
broader incremental eval) is tightly coupled to the working-set ratio.
Restructuring the matmul to sparse-first roughly 5× the working set
naturally — without any other change — would restore L1 residency on
the hot path.

## Halogen mixed-precision context

`docs/training_methodology_cross_engine_2026-05-01.md` §N8 surfaced a
related observation from Halogen's `arch.hpp`: **Halogen quantises threat
weights to int8 while keeping king-bucketed PSQ weights at int16.**
Memory and bandwidth halve on the threat embedding without the precision
loss mattering, because threats are sparse and bounded.

Coda already does this for v9. `l1_weights_8t` is int8 in inference; the
threat-weight matrix in `nnue.rs` is also int8 on the active path. So
"adopt mixed-precision" is partially already in place. The Halogen
insight is more useful as a confirmation that **dimension-specific
precision is a valid lever** than as a directly-applicable port.

The next dimension to audit is the threat *accumulator* itself
(`ThreatEntry.values: [[i16; 768]; 2]` at `threat_accum.rs:67`). At 256
plies × 3,072 bytes = 786 KB, this is half of L2 cache on a typical
host, traversed every push. If that can be int8 without precision loss
(threats are bounded) **and** without breaking the L1 matmul's input
chunk format, it would shrink the per-ply state by 2× and the L2
pressure correspondingly. This is a training-side question (does the
loss curve survive int8 threat accumulators) plus an inference
restructure.

It is not a near-term lever — it depends on the L1 matmul restructure
landing first, since the int8/int16 boundary lives at the matmul input.
Mentioned here to thread it into the NPS roadmap rather than discover it
again in 2026-Q3.

## Proposals

### Proposal 1 — Restructure `forward_with_l1_pairwise_inner` along Reckless's pattern

**Effort: 1-2 days. Expected impact: closes most of the 1.79× instruction
gap and the L1-matmul cache miss share — biggest single NPS lever
identified.**

What this means concretely:

1. **Move the SIMD primitives to a `simd` module** with `cfg(target_feature)`
   gating at the *primitive* level (mirror `Reckless/src/nnue/simd/`).
   Functions like `splat_i32`, `dpbusd`, `double_dpbusd`, `clamp_i16`,
   `pack_i16_i8` are the unit of dispatch.
2. **Drop `#[target_feature]` from `forward_with_l1_pairwise_inner`.**
   Build separate binaries per ISA (the OB harness already supports
   this; the `.openbench.yaml` and Makefile would gain an `avx2` /
   `avx512` / `avx512vnni` matrix). Or, if a single binary is non-
   negotiable for distribution, accept the inline barrier on the outer
   function but eliminate it on the SIMD primitives.
3. **Make the L1 matmul sparse-first, single-path.** One body that
   iterates `nnz: &[u16]` (already produced by `find_nnz_chunks4` in
   `sparse_l1.rs`), pair-unrolled like Reckless's `chunks_exact(2)`.
   The dense path becomes the scalar fallback for !x86/!aarch64.
4. **Inline the `find_nnz` step into the same call.** Currently we
   recompute NNZ every call from `stm_pw` / `ntm_pw`. Reckless does the
   same; the benefit is that NNZ generation is then one `find_nnz_chunks4`
   call inlined into the matmul, single fast path.

Order of operations matters. Doing this in the inverse order (start
with sparse-first, leave the dispatch tree) is what we tried in
2026-04-15 and it regressed −6%.

Acceptance criterion: NPS up by ≥10% on Zen 5 microbench, instruction
count gap measured on `eval-bench --mode incremental` shrinks below
1.3× Reckless's, L1 miss rate on the same microbench drops below 5%
(currently 15.7%). Then SPRT bounds [0, 5] for the resulting NPS Elo.

### Proposal 2 — PGO regression investigation (Task #20)

PGO regresses Coda v9 by ~10-12% (`make pgo` is currently shelved on
v9 branches). Working hypothesis after Phase 1: the regression has
the same root cause as the inline barriers. PGO needs full inlining
visibility to specialise hot paths; `#[target_feature]` annotations on
the SIMD bodies prevent the cross-function inlining PGO normally
unlocks.

**Validation step (cheap, 30 min):** build a PGO binary on a branch
with one `#[target_feature]` removed (and the call sites verified
correct on Zen 5 `-Ctarget-cpu=native`), benchmark. If PGO gain
returns to v5-style +3-5%, that confirms the hypothesis and Proposal
1 unblocks PGO as a side-effect.

If Proposal 1 lands, this validation becomes "rebuild PGO and
remeasure". So Proposal 2 is downstream of Proposal 1, not a parallel
work item.

### Proposal 3 — Mixed-precision audit beyond int8 threats

**Effort: research-grade. Defer until Proposal 1 lands.**

Coda already has int8 threat weights in inference, matching Halogen.
The next audit target is the threat *accumulator* (`ThreatEntry.values`
at `threat_accum.rs:67`):

- Current: `[[i16; 768]; 2]` per ply = 3,072 bytes × 256 plies = 786 KB.
- Hypothesis: int8 accumulator survives at v9's range (threats are
  bounded by feature definition, not learned).
- Validation: training-side experiment — train a v9 net with int8
  accumulator clamping and measure loss / inference Elo.
- Wins: 393 KB per-ply state instead of 786 KB; halves L2 pressure on
  the per-ply walk during `materialize`.

This is gated on (a) the L1 matmul restructure (input format becomes
the boundary), and (b) a Bullet training cycle to validate accuracy.
Park for 2026-Q3 unless Proposal 1 surfaces a cheaper path.

## Lever ranking — updated

Replacing the lever ranking table from `coda_vs_reckless_nps_2026-04-23.md`
§"Revised lever ranking (final)". The cache hygiene levers there
(flatten AccEntry, hot-feature frontload, prefetch) remain valid but
move down — the L1 matmul restructure is the upstream lever that
either subsumes them or makes them independently testable on a
healthier base.

| # | Lever | Effort | Expected NPS | Notes |
|---|---|---|---:|---|
| 1 | **L1 matmul restructure (Proposal 1)** | 1-2 days | **+15-25%** | The structural lever |
| 2 | PGO regression (Proposal 2) | Half day after #1 | +3-5% | Likely unblocked by #1 |
| 3 | Cache checking squares + retry LMP direct-check | 1-2 days | +5 LTC Elo (search-side) | `reckless_commit_catalog_2026-05-01.md` #4 |
| 4 | PST feature-index vectorisation (#792) | Half day | +2.7 STC | `reckless_commit_catalog_2026-05-01.md` #3 |
| 5 | Accumulator update reorder (#826) | Half day | +1.8 STC | `reckless_commit_catalog_2026-05-01.md` #6 |
| 6 | Flatten `AccEntry` to inline arrays | Half day | +2-4% | Was lever #1 in 2026-04-23 — now depends on #1 |
| 7 | Hot-feature frontloading | Half day | +1-3% | Same — depends on #1 |
| 8 | Prefetch in `apply_threat_deltas` | Half day | +0.5-1% | Marginal after #1 |
| 9 | Mixed-precision threat accumulator (Proposal 3) | Training cycle + restructure | +2-5% | Gated on #1 |

The big shift: levers 6-8 from the previous doc, while still real, are
each bounded by what the dense-first matmul leaves on the table. Lever
#1 is upstream of all of them.

## Cross-references

- `docs/coda_vs_reckless_nps_2026-04-23.md` — the per-eval / evals-per-
  node decomposition this doc builds on. Headline numbers (1.86× per-
  eval cost, 1.30× more evals per node, 2.42× combined) are still the
  framing; this doc explains the *structural* axis behind the per-eval
  cost.
- `docs/reckless_commit_catalog_2026-05-01.md` — 199-commit walk that
  established no single missing port closes the gap. Phase-3 ports
  remain valid (TT prefetch, accumulator reorder, PST vectorisation),
  but they live downstream of the L1 matmul structure.
- `docs/training_methodology_cross_engine_2026-05-01.md` §N8 — Halogen
  mixed-precision int8 threat / int16 PSQ. Coda already at parity on
  threat weights; accumulator-side audit is Proposal 3.
- `docs/v9_nps_findings_2026-04-18.md` — original v9 NPS investigation;
  some claims revised in `coda_vs_reckless_nps_2026-04-23.md` and
  further here.
- `docs/v9_sparsity_investigation_2026-04-19.md` — the 89%-sparse
  pairwise activation distribution that makes sparse-first viable in
  the first place. The L1-reg / weight-row sparsity work there is
  orthogonal — it shrinks the matrix for cache fit; Proposal 1 reads
  less of the matrix per call. Both compose.

## Methodology

- `perf record -F 999` on `coda eval-bench --mode incremental --reps 100000`,
  Zen 5 idle (OB worker stopped). Functions ranked by self-cycles.
- `perf stat -e instructions,cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses`
  same workload, both engines. Instruction-count gap measured here.
- Reckless source at `~/chess/engines/Reckless/src/nnue/forward/vectorized.rs`
  read in full; `propagate_l1` is the single call site producing the
  L1 matmul output.
- Coda hot-path source at `src/nnue.rs:2565` (`forward_with_l1_pairwise_inner`)
  + `src/sparse_l1.rs` (sparse and dense kernels, both available).

---

*Investigation 2026-05-01. Next action: prototype Proposal 1 on a
feature branch, microbench, then SPRT.*
