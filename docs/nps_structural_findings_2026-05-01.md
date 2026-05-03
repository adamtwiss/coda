# NPS Structural Findings — 2026-05-01

> **CORRECTIONS 2026-05-03 — two load-bearing claims retracted.**
> The body of this doc presents two hypotheses as the structural
> answer; both have been refuted by subsequent measurement. Read the
> "Update 2026-05-01 PM" section at the bottom and the additional
> "Update 2026-05-03" notes inline before acting on the lever
> ranking. The headline measurements (1.79× more instructions per
> eval, 18.8× more cache-references, 54× worse L1 miss rate) are
> still correct and load-bearing — only the *causes* attributed
> below are wrong.
>
> 1. **Sparse-first vs dense-first L1 matmul** is NOT the structural
>    lever. Production chunk-level density is ~73% non-zero, not 11%
>    as my microbench synthesised. Sparse iteration only saves ~1.4×
>    theoretical work, not 9×. Reckless's `propagate_l1` may itself
>    be slower per-call than Coda's `dense_l1_avx512_vnni`. See
>    "Update 2026-05-01 PM" below.
> 2. **Reckless's "compact memory footprint" claim** (line 210, "96 KB
>    L1 weights … 5 KB working set") was framed as a structural
>    advantage Coda lacked. **It's the same on both engines** at the
>    matmul step. Reckless's full NNUE has the same 49 MB threat
>    matrix as Coda — the cross-engine memory-size advantage cited
>    in `coda_vs_reckless_nps_2026-04-23.md` was wrong (corrected
>    there 2026-05-03).
>
> **What survives:** the AccEntry layout difference (Coda's 7
> heap-Vec scatter vs Reckless's `Box<[PstAccumulator]>` of inline
> `[[i16; L1_SIZE]; 2]` arrays) — verified, plausibly the dominant
> cause of the 54× L1-miss-rate gap, but **per-callsite L1-miss
> decomposition still pending**.

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

> **[CORRECTION 2026-05-03]** — this section frames sparse-first vs
> dense-first as THE structural lever. Subsequent measurement
> (Step C-cheap, see the "Update 2026-05-01 PM" section at the
> bottom) refuted that framing: production chunk density is ~73%
> non-zero, the per-call ratio at that density is ~1.4× not ~9×, and
> in production the integration was bench-NEGATIVE −4%. Read this
> section as historical context for the framing we tried, not as
> active recommendation.

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

> **[CORRECTION 2026-05-03] — sparsity premise is wrong.** The "89%
> input sparsity" in this table is feature-row activation sparsity
> from `v9_sparsity_investigation_2026-04-19.md`. The L1 matmul
> input (post-pairwise-CReLU 4-byte chunks) is **only ~27% sparse**
> in production (140/192 non-zero). Numbers below assume the wrong
> sparsity figure; treat as superseded.

Per L1 matmul call on v9 production (h16x32 net, ~~89% input sparsity per
`v9_sparsity_investigation_2026-04-19.md`~~ ← actual production density
73% non-zero, see "Update 2026-05-01 PM" below):

| Per L1 matmul call | Reckless (assumed at synthetic 89% sparse) | Coda dense | Ratio (assumed) |
|---|---:|---:|---:|
| Inputs iterated | ~85 (89% sparse skip) | 768 (all) | 9× more in Coda |
| Weight memory touched | ~5 KB | ~24 KB | 4.8× more |
| Inner-loop iterations | ~43 (pair-unrolled) | 768 | 18× more |

~~That ~9× per-call work ratio in the matmul is the dominant contributor
to the observed 1.79× total instruction-count gap.~~ **[CORRECTION
2026-05-03]: at production's 73% non-zero density the per-call ratio
is ~1.4×, not 9×. Sparse iteration is not the dominant contributor;
the 1.79× instruction-count gap must come from elsewhere — most
likely AccEntry layout + dispatch-tree code shape.**

### Why this produces both gaps simultaneously

> **[CORRECTION 2026-05-03] — this section's framing is REFUTED.** The
> table below claims dense-first iteration is the unifying mechanism
> for both the 1.79× instruction gap and the 19× cache-refs gap. The
> Step C-cheap experiment showed it's not — in production at ~73%
> chunk density, sparse-first iteration is bench-NEGATIVE −4%.
>
> **Best current understanding** (still requires per-callsite L1-miss
> decomposition to confirm):
>
> | Symptom | Most-likely mechanism |
> |---|---|
> | 1.79× instructions | `#[target_feature]` inline barriers preventing whole-pipeline optimisation + 10-arm dispatch tree in `forward_with_l1_pairwise_inner` body; runtime dispatch overhead. NOT sparse-vs-dense matmul work. |
> | 19× cache-refs / 16× L1-miss-rate | Coda's `AccEntry` layout (7 heap-allocated `Vec<i16>` per ply, ~1800 scattered allocations) vs Reckless's `Box<[PstAccumulator]>` of inline `[[i16; L1_SIZE]; 2]` arrays (one contiguous slice). NOT total NNUE size — both engines have the same 49 MB threat matrix. |
>
> The table immediately below is the original (refuted) framing,
> retained for context.

~~The 1.79× instruction count gap and the ~19× cache-references gap are
not two separate problems. They're two symptoms of dense-first iteration:~~

| Symptom | ~~Mechanism (refuted 2026-05-03)~~ |
|---|---|
| **1.79× instructions** | ~~Dense matmul does ~9× more multiply-accumulate work (768 inputs vs ~85 NNZ). Even highly vectorised, the work is ~9× larger; SIMD width and pair-unrolling close some of that, leaving the observed 1.8×.~~ |
| **19× cache-references / 16× L1-miss-rate** | ~~Dense access pattern reads the entire weight matrix per call, exhausting L1 working set. Sparse access pattern touches only the rows of NNZ inputs — a working set ~5× smaller fits in L1.~~ Raw measurements still valid: Reckless's `34.5 M cache-refs / 800 k evals = 43 refs/eval`; Coda's `648 M / 800 k = 810 refs/eval`. |

~~This is also why "fix the cache layout" alone wouldn't fully close the
gap. You can flatten `AccEntry`, frontload hot features, prefetch the
next weight row — and you'd still execute 1.79× more instructions on the
dense matmul because the work axis is wrong.~~ **[CORRECTION 2026-05-03]:
the inverse looks more likely. Cache hygiene (flatten AccEntry, etc.)
appears to BE the load-bearing lever, since the matmul-restructure
turned out NPS-negative in production.**

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

> **[CORRECTION 2026-05-03]** — this section's "read less, not arrange
> the read better" framing INVERTS the right answer. With the L1
> matmul restructure refuted, **arranging the read better (cache
> hygiene, especially flatten `AccEntry`) is now the primary lever**,
> not the secondary. The 16.88% vs 1.07% L1-miss-rate gap is real,
> but the cause is layout (Coda's heap-Vec scatter vs Reckless's
> contiguous slice), not matmul work amount.

### Per-callsite L1-miss decomposition (2026-05-03 measurement)

`perf record -e L1-dcache-load-misses --call-graph=fp -F 4000` against
`coda eval-bench --mode incremental --reps 100000` and Reckless's
equivalent `evalbench incremental 100000`. Both built with frame
pointers, OB worker stopped, Zen 5 idle.

**Total event count, 800 k evals:**
- Coda: **322 M L1-dcache-load-misses**
- Reckless: **10.8 M L1-dcache-load-misses**
- **Aggregate ratio: ~30× more in Coda** (matches 28× headline from
  `coda_vs_reckless_nps_2026-04-23.md`)

**Top callsites with absolute miss counts and per-callsite ratios:**

| Callsite (Coda → Reckless equivalent) | Coda % | Coda absolute | Reckless % | Reckless absolute | Ratio |
|---|---:|---:|---:|---:|---:|
| L1 matmul / forward inner | 32.17% | 103.6 M | 28.88% (`activate_ft`) | 3.1 M | **33×** |
| **Accumulator push (`simd_acc_fused_avx512` ← `materialize`) vs Reckless `PstAccumulator::add1_sub2`** | **27.13%** | **87.4 M** | **8.52%** | **0.92 M** | **95×** |
| Inlined threat helpers (0x1a1xxx, ← `forward_with_threats`) | ~27.5% | ~88 M | (proportional in `Network::evaluate` 30.2%) | ~3.3 M | ~27× |
| `build_dirty_piece` | 3.66% | 11.8 M | (n/a — Reckless does inline) | — | — |
| `materialize` (own self-cycles) | 3.47% | 11.2 M | — | — | — |
| `simd512_pairwise_pack_fused` (Coda) | 1.82% | 5.9 M | (folded into `activate_ft`) | — | — |

### What the data confirms

**The AccEntry layout hypothesis is confirmed.** Coda's accumulator-
push path has **95× more L1 misses** than Reckless's equivalent — 3×
worse than the average 30× ratio. That makes it the single most
cache-unfriendly part of Coda's hot path.

The mechanism, at `src/nnue.rs:4295`:
```rust
simd_acc_fused_avx512(&mut current.white, &parent.white, w_adds, w_subs, h);
simd_acc_fused_avx512(&mut current.black, &parent.black, b_adds, b_subs, h);
```
`current.{white,black}` and `parent.{white,black}` are 4 separate
`Vec<i16>` heap dereferences per push. Each dereference reaches a
heap allocation potentially-cold in L1, made worse by the fact that
neighbouring stack plies' Vecs are at unrelated heap addresses (no
spatial locality, no HW prefetcher win). Reckless reads
`pst_stack[i].values[pov]` — slice indexing into one contiguous
`Box<[PstAccumulator]>` where neighbouring plies are adjacent in
memory.

**The L1 matmul path's 33× is high but PROPORTIONAL to the aggregate
30× ratio** — so the dispatch-tree / `#[target_feature]` story still
applies (Step C-full could close it), but it's not the disproportionate
outlier the push path is.

### Implication for the lever ranking

Lever #1 (flatten `AccEntry`) was promoted to top of ranking on
2026-05-03 based on a *hypothesis*. After this measurement it's also
the **measured** highest-leverage lever. **Expected gain isn't tiny:**
even closing the push path's 95× → 30× (matching aggregate ratio)
removes ~57 M L1 misses per 800 k evals. At ~12 cycles per L1 miss
on Zen 5, that's ~684 M cycles saved per 800 k evals — roughly **half
of the total cycle budget for the eval-bench microbench** (1.32 B
cycles measured). Realistic NPS gain estimate: **+10-25%** if the
fix lands cleanly.

~~`coda_vs_reckless_nps_2026-04-23.md` §3 framed cache behaviour as a
distinct lever (flatten `AccEntry`, hot-feature frontload, prefetch).
Those levers are still real, but as the per-call quantification above
shows, the dominant cache cost is in the L1 matmul itself, and the
right way to fix it is to **read less**, not to **arrange the read
better**.~~ Coda's L1-dcache-load-misses on the eval-bench microbench
sit at ~16.88% (vs Reckless's 1.07%); the working-set ratio
~~(24 KB vs ~5 KB per matmul call)~~ does NOT account for most of the
gap (per `coda_vs_reckless_nps_2026-04-23.md` correction 2026-05-03,
both engines have the same 49 MB threat matrix and similar total
NNUE footprint). The cause is most likely the AccEntry layout
difference; per-callsite L1-miss decomposition is the next experiment.

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

> **[CORRECTION 2026-05-03] — Proposal 1 has been TRIED AND REVERTED.**
> The "biggest single NPS lever identified" claim is wrong. Step
> C-cheap (the integration of Proposal 1's restructure) regressed
> bench by 4% in production. See "Update 2026-05-01 PM" at the
> bottom and the revised lever ranking. Proposal 1 below is retained
> as the path that was attempted; the realistic upper bound is now
> ~3-8% NPS from inline-barrier removal alone (Proposal 2 / Step
> C-full), and the cache-hygiene levers (flatten AccEntry etc.) move
> to the top of the ranking.

### Proposal 1 — Restructure `forward_with_l1_pairwise_inner` along Reckless's pattern

> **[CORRECTION 2026-05-03] — TRIED AND REVERTED.** Step C-cheap
> integrated this on `feature/nnue-simd-restructure` and produced
> −4% bench regression in production despite a +2.4× microbench win.
> Root cause: chunk-density assumption (89% sparse) was wrong —
> production is 73% non-zero. See "Update 2026-05-01 PM" below.
> Code retained on the branch as infra; not currently a path to
> pursue.

**Original framing (refuted): Effort: 1-2 days. Expected impact:
closes most of the 1.79× instruction gap and the L1-matmul cache
miss share — biggest single NPS lever identified.**

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

> **[CORRECTION 2026-05-03]** — the table below was built around L1
> matmul restructure as Lever #1. With that refuted (Step C-cheap
> regressed −4% in production), the ranking is wrong. **Use the
> revised ranking immediately below; the original is retained
> beneath for diff context.**

### Revised lever ranking (2026-05-03)

| # | Lever | Effort | Expected NPS | Notes |
|---|---|---|---:|---|
| 1 | **Flatten `AccEntry` to inline arrays** (mirror Reckless's `Box<[PstAccumulator]>` of `Aligned<[[i16; L1_SIZE]; 2]>`) | Half day | unknown — best candidate for closing the 54× L1-miss gap | Only verified structural memory-layout difference vs Reckless. Per-callsite L1-miss decomposition pending. |
| 2 | **Per-callsite L1-miss decomposition** (`perf record -e L1-dcache-load-misses` with call-graph) | Half day | diagnostic | Confirms where the misses live before committing to #1's blast radius. Should run BEFORE #1. |
| 3 | Step C-full (drop `#[target_feature]` from `forward_with_l1_pairwise_inner`, multi-binary build) | 1-2 days | **+3-8%** (revised down from +15-25%) | Inline-barrier removal alone, validated by removing the attribute and measuring whether callees inline. |
| 4 | Hot-feature frontloading | Half day | +1-3% | Generic cache-hygiene; valid regardless of Reckless. |
| 5 | Prefetch in `apply_threat_deltas` | Half day | +0.5-1% | Same. |
| 6 | Cache checking squares + retry LMP direct-check | 1-2 days | +5 LTC Elo (search-side) | `reckless_commit_catalog_2026-05-01.md` #4 |
| 7 | PST feature-index vectorisation (#792) | Half day | +2.7 STC | `reckless_commit_catalog_2026-05-01.md` #3 |
| 8 | Accumulator update reorder (#826) | Half day | +1.8 STC | `reckless_commit_catalog_2026-05-01.md` #6 |
| 9 | Mixed-precision threat accumulator (Proposal 3) | Training cycle + restructure | +2-5% | Standalone (no longer "gated on #1") |
| 10 | PGO revalidation (Proposal 2) | Half day after #3 | +3-5% | Likely unblocked by #3, not by old-Proposal-1. |
| 11 | Training-side memory shrink (was old-#1 in 2026-04-23) | Training cycle | uncertain | Both engines have same 49 MB matrix; not a Reckless-gap closer. |

### Original ranking (refuted, retained for context)

~~Replacing the lever ranking table from `coda_vs_reckless_nps_2026-04-23.md`
§"Revised lever ranking (final)". The cache hygiene levers there
(flatten AccEntry, hot-feature frontload, prefetch) remain valid but
move down — the L1 matmul restructure is the upstream lever that
either subsumes them or makes them independently testable on a
healthier base.~~

| # | Lever | Effort | Expected NPS | Notes |
|---|---|---|---:|---|
| ~~1~~ | ~~**L1 matmul restructure (Proposal 1)**~~ | ~~1-2 days~~ | ~~**+15-25%**~~ | ~~The structural lever~~ — refuted, regressed −4% in production |
| ~~2~~ | ~~PGO regression (Proposal 2)~~ | ~~Half day after #1~~ | ~~+3-5%~~ | ~~Likely unblocked by #1~~ — moved to revised #10 |
| ~~3~~ | ~~Cache checking squares + retry LMP direct-check~~ | (moved to revised #6) | | |
| ~~4~~ | ~~PST feature-index vectorisation (#792)~~ | (moved to revised #7) | | |
| ~~5~~ | ~~Accumulator update reorder (#826)~~ | (moved to revised #8) | | |
| ~~6~~ | ~~Flatten `AccEntry` to inline arrays~~ | ~~Half day~~ | ~~+2-4%~~ | ~~Was lever #1 in 2026-04-23 — now depends on #1~~ — promoted to revised #1 |
| ~~7~~ | ~~Hot-feature frontloading~~ | (moved to revised #4) | | |
| ~~8~~ | ~~Prefetch in `apply_threat_deltas`~~ | (moved to revised #5) | | |
| ~~9~~ | ~~Mixed-precision threat accumulator (Proposal 3)~~ | (moved to revised #9, no longer gated) | | |

~~The big shift: levers 6-8 from the previous doc, while still real, are
each bounded by what the dense-first matmul leaves on the table. Lever
#1 is upstream of all of them.~~ **[CORRECTION 2026-05-03]: the inverse
is correct — Lever #1 (matmul) was refuted, so the cache-hygiene
levers (old #6-8) become the leading candidates. They are NOT bounded
by the matmul restructure since the matmul work amount is closer to
optimal than initially modelled.**

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
