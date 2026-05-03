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
real lever. With `target-cpu=native` and `lto=true`, LLVM already folds
those `if self.has_avx512` branches at compile time — they cost almost
nothing to keep. The real barrier is the
`#[target_feature(enable = "avx512f", enable = "avx512bw")]` attribute on
the SIMD function *bodies* (e.g. `forward_with_l1_pairwise_inner` at
`nnue.rs:2564`). That attribute marks the function as having a different
ABI/feature set than its caller, which **prevents LLVM from inlining the
function into any caller without those exact target features set**. Once
the SIMD body is non-inlined, no amount of changing the dispatch check
shape recovers the cost.

To remove the barrier requires either:
- Specialise the entire `apply_threat_deltas` / `forward_with_l1_pairwise_inner`
  body for each ISA with a wrapper-dispatch (multi-level) so the wrapper
  pays one branch and each specialised body inlines its primitives, or
- Multi-binary build with explicit per-ISA target features (Reckless's
  approach — `cfg(target_feature)` cfg-gated SIMD modules + separate
  binaries per ISA).

Both are real-port-effort, not quick validation.

**Lesson:** the dispatch overhead is not 10 branch predictions per eval.
It's that the SIMD body is opaque to the surrounding optimizer —
including const propagation, register allocation, and cross-function
inlining. The compile-time-dispatch hypothesis as I'd framed it
(simple cfg swap = +5-9%) was wrong. The real lever exists but is
hidden behind a non-trivial restructure.

## Phase 2 — Fresh perf decomposition on current trunk

After all the recent ports landed (AVX-512 `apply_threat_deltas`,
setwise movegen, eval-only TT writeback), we re-ran the full perf-stat
+ `perf record` battery on Zen 5 (idle, OB worker stopped) on the
current trunk.

### `coda bench 16` — search-bench cycle decomposition

`perf record --call-graph=fp -F 4000` against `coda bench 16`,
production v9 net, 14 256 samples, 14.3 G cycles total:

| Symbol | % cycles | Change vs 2026-04-23 |
|---|---:|---|
| `ThreatStack::ensure_computed` | 21.49% | apply_threat_deltas inlined here now |
| `forward_with_l1_pairwise_inner` | **10.96%** | was 3.27% — **3.4× share** |
| `push_threats_for_piece` | 5.45% | flat |
| `finny_batch_apply` | 4.04% | flat |
| `simd_acc_fused_avx512` | 2.25% | flat |
| `attackers_to` | 1.80% | flat |
| `ThreatStack::refresh` | 1.74% | flat |
| `forward_with_threats` | 0.57% | flat |

Two reads of the search-bench profile:

1. **AVX-512 `apply_threat_deltas` no longer the top symbol.** Today's
   merge cut its standalone share — it's been folded into
   `ensure_computed` (21.49%, including wrapper machinery). Total
   absolute cycles in the threat-update path dropped meaningfully.
2. **`forward_with_l1_pairwise_inner` jumped 3.27% → 10.96%.** That's
   the L1 matmul forward path — now the second-biggest hotspot.

### `coda eval-bench --mode incremental` — pure forward-pass decomposition

Same `perf record`, isolated to the incremental forward pass (no
movegen, no TT, no search). 1.27 G cycles total:

| Symbol | % cycles | What it does |
|---|---:|---|
| `forward_with_l1_pairwise_inner` | **36.13%** | FT→L1→L2→output matmul pipeline |
| `simd_acc_fused_avx512` | 15.44% | Accumulator update SIMD |
| `materialize` | 6.95% | Lazy accumulator computation |
| `make_move` / `unmake_move` | 5.91% | Board mechanics |
| `simd512_pairwise_pack_fused` | 4.80% | Pairwise activation pack |
| `forward_with_threats` | 1.45% | Threat fold-in |

That's ~70% of the eval cost. The L1 matmul forward dominates pure
incremental eval at 36% of cycles, more than 2× the next symbol.

### `perf stat` — instruction-count + cache gap to Reckless

Same workload (`eval-bench --mode incremental --reps 100000` on Coda;
`evalbench incremental 100000` on Reckless). Both engines on AVX-512+VNNI
Zen 5, idle CPU.

| Metric | Coda | Reckless | Ratio | Δ from 2026-04-23 |
|---|---:|---:|---:|---|
| Instructions per 800k evals | 3.71 B | 2.08 B | **1.79× more** | flat (was 1.86×) |
| Cycles | 1.32 B | 0.77 B | 1.71× | flat |
| Cache-references | 648 M | 34.5 M | **18.8× more** | improved from 31.6× |
| L1-miss rate | 16.88% | 1.07% | 16× worse | flat |
| Branches | 634 M | 137 M | 4.6× more | flat |

**The two structural gaps that have NOT closed:**

1. **1.79× more instructions per eval** — Coda runs nearly twice as
   much code per evaluation. Was 1.86× nine days ago; the AVX-512
   merge improved cycles-per-instruction but didn't reduce instruction
   count (3.71 B now vs 3.75 B then). All the SIMD ports landed since
   2026-04-23 reduced cycles per piece of work without reducing how
   much work each eval does.
2. **18.8× more cache references** — touching way more memory lines.
   The headline ratio dropped from 31.6× to 18.8×, but **that's
   because Reckless's number went UP (19 M → 34.5 M, their recent
   commits added some memory traffic), not because Coda's improved**.
   Coda's cache-references *jumped* 596 M → 648 M (+9%) over the same
   period.

Bench-NPS gap closed from 1.66× to ~1.5× over our recent merges, but
the structural instruction-count gap (1.79×) is essentially unchanged.
We've made the same code faster per cycle without reducing how much
code we run per eval.

| Gap | Was (2026-04-23) | Now (2026-05-01) | Closed? |
|---|---:|---:|---|
| Instruction count | 1.86× | 1.79× | barely |
| Cache references | 31.6× | 18.8× | partly (Reckless got worse too) |
| L1-miss rate | 15.72% | 16.88% | got slightly worse |
| Cycles | 1.83× | 1.71× | a bit |

That's the real "fundamental" answer: we're doing nearly twice as many
instructions per eval as Reckless, and most of those extra instructions
live in `forward_with_l1_pairwise_inner` (36% of eval cycles). Until we
shrink that, the gap is structural.

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

### Quantifying the per-call gap

Per L1 matmul call on v9 production (h16x32 net, 89% input sparsity per
`v9_sparsity_investigation_2026-04-19.md`):

| Per L1 matmul call | Reckless | Coda dense | Ratio |
|---|---:|---:|---:|
| Inputs iterated | ~85 (89% sparse skip) | 768 (all) | 9× more in Coda |
| Weight memory touched | ~5 KB | ~24 KB | 4.8× more |
| Inner-loop iterations | ~43 (pair-unrolled) | 768 | 18× more |

That ~9× per-call work ratio in the matmul is the dominant contributor
to the observed 1.79× total instruction-count gap. Combined with smaller
factors (dispatch overhead, more accumulator fields, branch overhead from
the 10-arm dispatch tree), the structural gap accounts for essentially
all of the 1.79×.

### Why this produces both gaps simultaneously

The 1.79× instruction count gap and the ~19× cache-references gap are
not two separate problems. They're two symptoms of dense-first iteration:

| Symptom | Mechanism |
|---|---|
| **1.79× instructions** | Dense matmul does ~9× more multiply-accumulate work (768 inputs vs ~85 NNZ). Even highly vectorised, the work is ~9× larger; SIMD width and pair-unrolling close some of that, leaving the observed 1.8×. |
| **19× cache-references / 16× L1-miss-rate** | Dense access pattern reads the entire weight matrix per call, exhausting L1 working set. Sparse access pattern touches only the rows of NNZ inputs — a working set ~5× smaller fits in L1. Reckless's `34.5 M cache-refs / 800 k evals = 43 refs/eval`; Coda's `648 M / 800 k = 810 refs/eval`. |

This is also why "fix the cache layout" alone wouldn't fully close the
gap. You can flatten `AccEntry`, frontload hot features, prefetch the
next weight row — and you'd still execute 1.79× more instructions on the
dense matmul because the work axis is wrong. Cache effects pile on the
same axis: less work in matmul = less memory traffic = better cache
behaviour. Restructuring the matmul to sparse-first roughly 5× the
working set naturally — without any other change — would restore L1
residency on the hot path *and* close most of the instruction-count gap.

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
distinct lever (flatten `AccEntry`, hot-feature frontload, prefetch).
Those levers are still real, but as the per-call quantification above
shows, the dominant cache cost is in the L1 matmul itself, and the
right way to fix it is to **read less**, not to **arrange the read
better**. Coda's L1-dcache-load-misses on the eval-bench microbench
sit at ~16.88% (vs Reckless's 1.07%); the working-set ratio (24 KB vs
~5 KB per matmul call) accounts for most of the gap directly.

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

## Update 2026-05-01 PM — Step C-cheap attempted, model corrected

Step A (extract SIMD primitives to `src/nnue_simd.rs`) landed bench-
neutral as planned (merged to main). Step B added a Reckless-shape
sparse-first kernel `propagate_l1_avx512_vnni_v2` + VBMI2 SIMD NNZ
scanner and a standalone microbench — soft pass on the gate:
matmul-only ceiling **5.65× vs dense** with combined-buffer signature,
end-to-end **2.41×** with VBMI2 SIMD scan. Step C-cheap plugged v2
into `forward_with_l1_pairwise_inner` and produced a **−4% bench
regression** (1273 k → 1219 k NPS), opposite the +18-22% the
microbench predicted.

**Root cause: the doc's "89% sparse" framing referred to feature-row
activation, not post-pairwise-CReLU 4-byte chunk density.** Adding a
diagnostic counter at the call site showed real production density
is ~73% non-zero per chunk (140/192 NNZ chunks per call), not 11%
(27/192) as the microbench synthesised. After the pairwise
multiplication `stm_pw[i] = ((a * b) >> FT_SHIFT) as u8`, a chunk is
non-zero whenever ANY pair of accumulator values has both halves
non-zero — much denser than raw feature activation.

At 73% chunk-density, sparse iteration saves only ~1.4× theoretical
work (192 → 140 chunks). Combined with NNZ-scan overhead and the
larger function body (icache pressure), v2 is bench-negative in
production traffic.

### Reckless's `propagate_l1` re-examined

Reading Reckless's `propagate_l1` more carefully against this finding:
their kernel uses **a single 1-ZMM accumulator** at `L2_SIZE=16`, with
4-cycle VPDPBUSD latency unhidden across the inner `nnz.chunks_exact(2)`
loop. At ~140 NNZ chunks × 4 cycles serial latency ≈ 560 cycles
≈ ~150 ns per call. Coda's `dense_l1_avx512_vnni` uses **4-way
interleaved accumulators** to hide that latency, achieving ~50 ns at
192 chunks.

So: **Reckless's L1 matmul is probably slower per-call than Coda's
dense kernel** at this op-point. Their forward pass nevertheless runs
faster overall, but **not because of sparse iteration** — the
sparsity-iteration framing was wrong for both engines. The real lever
seems to be the inlined single-SIMD-path structure (no
`#[target_feature]` inline barriers, no dispatch overhead, no
per-perspective demux), not the dense-vs-sparse iteration choice.

### Implications for the lever ranking

Lever #1 ("Restructure forward_with_l1_pairwise_inner along Reckless's
pattern, sparse-first") was based on a flawed sparsity model and
needs reframing. The correct framing is closer to Step C-FULL:
**eliminate `#[target_feature]` inline barriers and consolidate
dispatch paths**, not "switch to sparse iteration."

Three follow-ups:

1. **Step C-cheap is dead at this op-point.** v2 + VBMI2 SIMD scan
   are kept on `feature/nnue-simd-restructure` (NOT merged to main)
   as infrastructure for future investigations (lower-density nets,
   wider L1, off-architecture ports). **Cleanup tripwire**: if Step
   C-full hasn't validated by 2026-05-15, delete the v2/microbench
   code from that branch and retain only the experiments.md log.
2. **Step C-full — drop `#[target_feature]` from
   `forward_with_l1_pairwise_inner`** becomes the real structural
   lever. Phase 1 found that simple cfg swaps don't unlock anything;
   the real test is removing the attribute and ensuring all callees
   inline. Multi-binary build per ISA may be the path of least
   resistance.
3. **Lever ranking #6-8 (cache hygiene: flatten `AccEntry`,
   frontload, prefetch)** are still valid and now move UP, since the
   sparse-first matmul lever didn't pan out.

The original "+15-25% NPS" estimate for Lever #1 should be discounted
substantially. Step C-cheap result places the realistic upper bound
on the L1-matmul-restructure family at probably **~3-8% NPS** (from
inline-barrier removal + dispatch consolidation, not work reduction).

### Methodology lesson banked

The 2.41× microbench gate was a SOFT pass (target was 6×) and we
proceeded. The right intermediate step would have been **probing
real production NNZ density before integration**. The microbench
synthesised 89% sparsity from the doc's framing without verifying
the framing matched the actual data shape. Adding "probe production
data distribution at the integration point before microbenching" to
the metrics-discipline memory.

---

*Investigation 2026-05-01. Step A merged to main; Step B/C-cheap
explored on `feature/nnue-simd-restructure` (kept for Step C-full
follow-up). Step C-full remains the real candidate but needs
validation that inline-barrier removal alone delivers the predicted
NPS.*

## Update 2026-05-03 — Cache hygiene leg landed: +6.77 Elo H1 (SPRT #921, MERGED)

The Step C-cheap recheck spawned a **per-callsite L1-miss
decomposition** that turned out to be the highest-leverage
diagnostic in the whole leg. The findings + fixes from that
decomposition landed as `feature/acc-data-stack` and SPRT'd at
+6.77 Elo H1 — biggest NPS-class win since AVX-512
`apply_threat_deltas` (+4.4 Elo).

### What the decomposition found

`perf record -e L1-dcache-load-misses --call-graph=fp` against
`coda eval-bench --mode incremental --reps 100000` and Reckless's
equivalent. **Total event counts: Coda 322 M vs Reckless 10.8 M
(~30× aggregate).** Per-callsite breakdown:

| Callsite | Coda absolute | Reckless equivalent | Ratio |
|---|---:|---:|---:|
| L1 matmul / forward inner | 103.6 M | 3.1 M (`activate_ft`) | 33× |
| **Push (`simd_acc_fused_avx512`)** | **87.4 M** | **0.92 M** (`add1_sub2`) | **95×** |
| Inlined threat helpers | ~88 M | ~3.3 M | ~27× |

The push-path 95× vs aggregate 30× was the disproportionate outlier
that motivated the AccEntry layout investigation. Reading the
actual code revealed `AccEntry`'s inline `[i16; MAX_HIDDEN_SIZE =
2048]` arrays were 2.67× over-provisioned vs production
hidden_size = 768 (16.5 KB AccEntry, 4.3 MB stack — vs Reckless's
~3 KB per ply, 1.5 MB stack).

### Three commits that landed

1. **`AccDataStack`** — one contiguous `Box<[i16]>` per stream (psq
   + threat), sized to runtime hidden_size. Mirrors Reckless's
   `Box<[PstAccumulator]>` of inline `[[i16; L1_SIZE]; 2]`. Stack
   memory: 4.3 MB → 1.65 MB.
2. **`MaybeUninit` on `forward_with_l1_pairwise_inner` stack arrays.**
   `perf annotate` showed two `memset@GLIBC` calls at function
   entry as the top L1-miss source — `[0u8; 2048] × 2 + [0i32;
   512]` zeroing ~3.6 GB of memset traffic per bench. Replaced
   with `MaybeUninit::<[u8; 2048]>::uninit()`. **The biggest
   single win in the leg.**
3. **`MaybeUninit` on `apply_threat_deltas` index buffers.** Same
   pattern, smaller absolute win.

### Result trajectory (eval-bench --mode incremental, Zen 5)

| Stage | L1 misses | vs Reckless | Bench NPS |
|---|---:|---:|---:|
| Trunk baseline | 304 M | 38× | 1273k |
| + AccDataStack | 273 M | 34× | 1268k |
| + memset fix #1 (forward) | 157 M | 20× | 1306k |
| + memset fix #2 (threats) | **145 M** | **18×** | **1313k** |

**~Half the L1-miss gap to Reckless closed.** IPC went 2.82 →
2.99 (now exceeds Reckless's 2.78 — Zen 5 OoO is no longer
wait-stalling on memory).

### Fleet-wide validation (5 hosts, OoO-strength gradient)

| Host | uArch | Δ NPS |
|---|---|---:|
| Hercules | Coffee Lake (Intel Xeon, AVX-2) | **+6.7%** |
| Atlas | Zen 1 (EPYC 7351P) | **+6.6%** |
| MacBook M5 | Apple Silicon (NEON) | +3.5% |
| Zeus | Zen 5 (AVX-512+VNNI) | +3.1% |
| ionos1 | Zen 3 (Milan, VM) | +2.4% (noisy) |

Pattern matched the prediction: **weaker-OoO uArchs gained ~2× as
much** (cache misses no longer hidden by OoO get materialised as
cycle savings; on wide-OoO hosts only the eliminated memset
traffic is real CPU-work savings).

### SPRT #921: +6.77 ±3.9 Elo, H1 ✓ at 5234 games

Bigger than fleet-weighted bench gain (~+4-5%) would predict.
Likely from worker mix biased toward older uArchs (where bench
gain was 2× larger) plus reduced run-to-run variance feeding more
deterministic search.

### Lever ranking — re-revised 2026-05-03 PM (after merge)

The "promoted Lever #1" framing was directionally right but the
attribution was wrong:

- The flatten-AccEntry framing assumed accumulator data layout was
  the issue. AccDataStack alone was bench-neutral. The actual
  bottleneck was **memset traffic** from over-provisioned stack
  arrays — invisible without `perf annotate` per-symbol.
- The diagnostic chain that found it: per-callsite L1-miss decomp
  → AccEntry restructure (neutral but informative) →
  re-decompose → annotate `forward_with_l1_pairwise_inner` →
  spot the memsets.

**Updated priorities for further cache work:**

1. **Find more memset/init waste** in less-hot paths.
   `forward_with_l1` (v7 forward) has the same `[0u8; 2048] × 2`
   pattern. Dead code on v9 prod but trivial fix.
2. **L1 weight matrix layout / hot-feature reorder.** Reckless's
   ~5 KB working set per matmul vs Coda's ~24 KB is the dominant
   remaining cache delta. Sparse-first was bench-negative; the
   right approach may be hot-feature frontloading (rank features
   by activation count, permute matrix at load).
3. **Push-path delta-row count.** Instrument to confirm Coda
   applies the same number of weight rows per push as Reckless.
   If we apply more consistently, that's a different fix (better
   diff calculation).
4. **Step C-full** (drop `#[target_feature]` from outer SIMD
   functions, consolidate dispatch tree) remains valid for the
   instruction-count gap (1.79× more in Coda) — the structural
   axis the cache work doesn't touch.

### Methodology durables banked

- **`perf annotate -e L1-dcache-load-misses` per-symbol** is the
  right tool for finding memset/init waste. 5-minute exercise that
  surfaced the +3% bench commit. Should be standard before any
  cache-hygiene investigation declares "fundamental work, hard to
  reduce."
- **Per-callsite L1-miss decomposition** (not aggregate miss
  rate) is what reveals disproportionate hotspots. The push
  path's 95× ratio vs aggregate 30× motivated the right next
  experiment.
- **Probe production data distribution before microbenching** —
  banked from the Step C-cheap revert two days earlier;
  reaffirmed here as the right pre-refactor diagnostic discipline.

---

*Investigation closed 2026-05-03 with #921 H1 merged. AccDataStack
+ memset fixes shipped to main (commit `6d8168b`). The original
"L1 matmul restructure as Lever #1" framing was retired — cache
hygiene matters but the actual mechanism was memset waste, not
matmul shape. Future cache work should follow the per-callsite
L1-miss decomposition methodology that found it.*
