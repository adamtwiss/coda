# v9 NPS Findings — 2026-04-18

Consolidated summary of NPS investigation on `feature/threat-inputs`.
Supersedes the hopeful estimates in
`docs/v9_nps_const_generic_test_2026-04-17.md` and
`docs/v9_nps_optimization_plan.md` with empirical measurements.

---

## Starting point

- Coda v9 with e800s800 net: ~548K NPS (10-run mean)
- Coda v5 production: ~980K NPS same hardware
- Reckless: ~989K NPS same hardware
- **Gap to v5/Reckless: ~1.77× (44% slower)**

v5 and Reckless are both AVX2-only on our measurement boxes (Xeon
E-2288G, EPYC 7351P). Neither has AVX-VNNI or AVX-512-VNNI. The
gap is not from missing VNNI instructions.

## Profile decomposition (post-fixes)

NNUE-related total ≈ **61% of CPU time**. Decomposition of a
depth-14 bench, `perf record --call-graph=fp`:

| Function | %CPU |
|---|---|
| `apply_deltas_avx2` (threat delta apply) | 12.7% |
| `forward_with_l1_pairwise_inner` | 12.6% |
| `push_threats_for_piece` | 9.3% |
| `SearchInfo::eval` (wrapper) | 8.0% |
| `dense_l1_avx2` (L1 matmul) | 6.1% |
| `finny_batch_apply_avx2` (PSQ refresh) | 5.1% |
| `simd_acc_fused_avx2` (PSQ incremental) | 3.3% |
| `simd_pairwise_pack_fused` | 2.0% |
| `refresh_accumulator` | 2.3% |

Search code (negamax, movepicker, board ops) total ~15%.

Key data point: Coda's NNUE alone consumes ~3.4 μs/node.
Reckless's **entire** per-node time is ~3.3 μs. The gap is NNUE.

---

## What worked (landed)

### 1. Fused PSQ incremental update (commit 317deab)

**Before**: N passes over the accumulator for an N-change move.
First delta fused copy+delta, remaining deltas each their own
load/compute/store. Capture = 3 passes, castling = 4.

**After**: single pass. `simd_acc_fused_avx2(dst, src, add_rows,
sub_rows, h)` loads src chunk into registers, applies all adds/subs
in-register, stores once. Memory traffic constant in delta count.

**Measured**: ~-1.3% total CPU → +0.4% NPS. Much smaller than the
"3× bandwidth" back-of-envelope because h=768 is L1-resident (3KB
total). Cache hits, not DRAM/L2 traffic.

### 2. Fused threat-delta copy+apply (commit 2218233)

**Before**: `apply_threat_deltas` did `dst.copy_from_slice(src)`
then `apply_deltas_avx2(dst, …)` — two streaming passes.

**After**: single pass. `apply_deltas_avx2` takes `src` parameter,
loads parent chunk directly into registers, applies deltas, stores
to dst. Separate memcpy gone.

**Measured**: +2.6% NPS. The comment at the call site had been
claiming "21× less memory traffic than per-delta streaming"
aspirationally — this commit actually delivers it.

### 3. Column-major L1 matmul via `dense_l1_avx2` (commit afd2fcd)

**Before**: row-major L1 weights. For each output, scan all
pairwise-CReLU inputs (16 cache-line touches per input chunk).

**After**: input-chunk-major layout. For each 4-byte input chunk,
splat_i32 broadcast + maddubs/madd contributes to all 16 L1 outputs
via 2 AVX2 accumulators. Matches Reckless's layout.

The `l1_weights_sparse` field and the code were already in the
repo, gated behind `else if false && …` with a note that the
previous implementation was 6% slower at L1=16. That measurement
compared **sparse-with-zero-check** vs **row-major dense**. The
winner today is **dense column-major** (no zero-check overhead,
same cache-friendly layout) at +1.2% NPS.

**Caveat**: without AVX-VNNI, we emulate `vpdpbusd` with
`maddubs+madd+add`. The Reckless scheme gets its biggest win on
VNNI hardware where the broadcast contributes to all outputs in
one instruction; on AVX2-only this is diluted.

### Cumulative NPS win: ~548K → ~569K (10-run mean, +3.8%)

Correctness verified: bench unchanged (1,798,525 nodes), all
fuzzers pass.

---

## What didn't work

### 1. Whole-function const-generic specialisation

**Attempted**: copy `forward_with_l1_pairwise_inner` to a
const-generic variant `forward_v9_prod_avx2<const H, const L1_PB,
const L2_PB>` for v9-production dimensions (H=768, L1=16, L2=32).
Hardcoded all `self.{hidden_size, l1_size, l1_per_bucket,
l2_per_bucket, use_pairwise, bucketed_hidden, dual_l1, l1_scale}`
as compile-time constants derived from generic parameters. Only
AVX2 path retained.

**Measured** (10-run means, same HEAD, same net, same bench count):
- With fast path: 545K NPS
- Without fast path: 543K NPS
- **Delta: +0.4% — within noise**

**Why it didn't help**: the time is in the SIMD helpers
(`simd_pairwise_pack_fused`, `dense_l1_avx2`) which are
`pub unsafe fn` with `target_feature(enable = "avx2")`. Call
boundary to them forces runtime arguments regardless of const-ness
at the caller. The outer function's const-ness only buys us
elimination of a few dozen arithmetic ops — LLVM was already doing
a reasonable job on them.

**Extension ceiling**: would need to make **every** SIMD helper
const-generic **too**, AND eliminate every `&self.field` read
across the entire forward path. That's a 2-3 day end-to-end
refactor with no strong empirical signal that it'd land the
originally-projected 10-15%. **Shelved unless the training pipeline
stalls and NPS becomes the only lever.**

### 2. PGO (profile-guided optimization)

**Attempted**: `cargo pgo instrument` + bench profile run +
`cargo pgo optimize`. Also tried variants:
- Vanilla
- Without `--features embedded-net` (smaller binary)
- With `panic = "abort"` in Cargo.toml (simpler CFG)

**Measured** (trunk, v9 e800s800 net):
- Non-PGO: ~549K NPS
- PGO (any variant): 445-491K NPS → **-10.4% regression**

**Same hardware comparison**: Reckless with PGO gains +1.5% (974K
→ 989K). Something structural in Coda's hot path that cargo-pgo's
inlining heuristics misjudge. The documented hypothesis ("instrument
build distorts profile via counter overhead, leading to wrong
inlining decisions") is plausible but unverified.

**Status**: documented negative result. Revisit only with AutoFDO
(sampling-based, no instrumentation overhead) tooling when
available.

### 3. Reckless-pattern vectorised threat-delta algorithm (`board_to_rays` + `closest_on_rays`)

**Full port shelved; hybrid path unmeasured.** Three variants:

a) **Full port (emit direct threats only, drop x-rays)** — would
   require retraining to recover x-ray features. Net -57 to -82
   Elo. **Shelved.**

b) **Port primitives + extend with Coda x-rays end-to-end**
   (vectorise both direct and x-ray generation) — multi-day,
   correctness-critical. **Shelved.**

c) **Hybrid: vectorise direct-threat primitives, keep x-rays
   scalar** — NOT yet measured, should not be shelved. The
   `closest_on_rays` bit trick computes closest occupant per
   ray in O(1); x-rays in our scheme are the second occupant
   past each closest blocker, derivable with one additional
   magic-bitboard call per affected slider. The hybrid keeps
   our feature set exactly, uses Reckless's O(1) closest-finder
   for the direct portion, and does existing scalar x-ray on the
   handful of sliders per move that changed blocker state.

**Data we need before deciding on (c)**: direct-vs-xray CPU split
inside `push_threats_for_piece`. If direct is >5% and x-ray is
<3%, hybrid reopens as a 1-2 day experiment with ~+3-4% NPS
potential. **Measurement planned.**

**Reckless's feature set vs Coda's**:

Reckless's `refresh` (canonical feature enumeration) emits
**direct threats only**. Coda's `enumerate_threats` emits
**direct + per-slider x-ray-through-blocker** — more features,
trained into our net weights at +110 Elo gain from yesterday's
x-ray fix.

### 4. Walk-back refresh (previously stated "expected +1-2%, fast refresh anyway")

**Underexamined.** 41% of `materialize` calls fall through to
refresh because the parent's accumulator was never computed (TT
cut or pruned-before-eval at parent). But grandparent may still
be computed — walking back 1-2 ancestors and applying 4-8 deltas
forward from there is almost certainly cheaper than a full
Finny-cached refresh.

**Data we need**: distribution of "distance to nearest computed
ancestor" at the moment refresh fires. If >60% have an accurate
ancestor within 2-3 plies, walk-back delivers >2% NPS (not
1-2%). Matches Reckless's `update_pst_accumulator` pattern
`for i in accurate..self.index`. **Measurement planned.**

### 5. Inline attribute audit vs Reckless — zero-risk experiment

**Not yet attempted.** cargo-pgo regresses Coda -10.4% while
Reckless gains +1.5% on the same hardware. Most-likely cause
is that Reckless's hot functions are already `#[inline]` /
`#[inline(always)]` or small enough for LLVM to inline without
hint, leaving PGO little to confirm. Coda's hot path goes
through `&self.field` loads across function boundaries where
inline decisions depend on heuristics.

**30-minute experiment**: diff inline attributes on hot functions
(forward path, accumulator update path, threat delta path)
between Coda and Reckless. Where Reckless marks `#[inline]` or
`#[inline(always)]` and Coda doesn't, add the attribute. May
recover 1-3% or may be flat. Zero semantic risk, no retraining,
no tooling.

---

## Key lessons

### The "memory bandwidth" framing overstated the PSQ/threat fuse wins

For h=768 i16 (1.5KB per perspective × 2 = 3KB working set),
multi-pass accumulator updates turn into **L1-cache hits**, not
DRAM/L2 traffic. The cost of "3 passes" is register pressure and
loop-overhead, not memory bandwidth. Bandwidth arithmetic
overestimates gains by ~10× for this working-set size. Real
gains were 0.4% (PSQ) and 2.6% (threat) vs theoretical 3×
bandwidth reductions.

### Const-generic at caller doesn't propagate through SIMD helpers

SIMD intrinsic calls in Rust (x86_64 `__m256i` etc behind
`target_feature`) are opaque to LLVM's inlining/specialisation.
Caller-side const arguments to these functions don't help unless
the helpers themselves are `#[inline(always)]` const-generic.
Making them so is possible but requires systemic rewrite.

### Refresh rate is surprisingly high

41% of `materialize` calls fall through to refresh (not
incremental) because the parent wasn't computed (TT cut or
pruned-before-eval). Potential small win from walking ancestors
up the stack to find a computed base, then applying deltas
forward — but refresh via Finny-cache is already fairly fast,
so the expected gain is small (~1-2% NPS).

### The real v5/Reckless gap lives in NNUE, not search

Coda v9 NNUE alone: 3.4 μs/node. Reckless total: 3.3 μs/node.
Coda's search/board/TT work is comparable to Reckless's. To
match Reckless NPS, NNUE needs to drop roughly 2× — which
means structural changes (feature-set change, full
const-generic chain, different quantisation scheme, etc.)
not micro-optimisations.

---

## What's left as plausible NPS levers

Ranked by realistic upside and effort:

| Lever | Upside | Effort | Correctness risk |
|---|---|---|---|
| Wait for Reckless KB net + retune | +30-60 Elo (not NPS) | Low | None |
| Factoriser in training | +10-20 Elo (not NPS) | Training run | None |
| **Inline attribute audit vs Reckless** | **+1-3% NPS** | **30 min** | **None** |
| **Walk-back refresh** (if measurement supports) | **+2-4% NPS** | **Half day** | **Low** |
| **Hybrid threat path** (vectorise direct, scalar x-ray) | **+3-4% NPS** | **1-2 days** | **Medium** |
| Delta-count histogram → cap/batch if long-tailed | +1-2% NPS | Half day | Low |
| Full end-to-end const-generic NNUE | +5-10% NPS | 2-3 days | Medium |
| AutoFDO (sampling PGO) | Unknown, likely +3-5% NPS | 1 day + tooling | Low |
| Reckless-pattern full threat port (drop x-rays + retrain) | +28 Elo NPS **−110 Elo features** | 3-5 days + train | Very high |

**The Elo economics favour training/tuning over NPS micro-optimisation
at this point.** The last tune delivered +38.6 Elo in a day. Three
NPS commits in a day delivered +3.8% NPS ≈ +4 Elo.

**Revised ordering rule**: if NPS work resumes, order is
1) inline audit (30 min, zero risk), 2) walk-back refresh
(half day, measurement-gated), 3) hybrid threat path
(1-2 days, measurement-gated). Full const-generic and full threat
port remain shelved.

---

## Things NOT to re-try (already tested)

- Const-generic-ing individual SIMD kernels in isolation (April 17
  probe: zero gain)
- Whole-outer-function const-generic while leaving helpers runtime
  (2026-04-18 probe: zero gain)
- cargo-pgo in any of its 3 tested variants (regression every time)
- Enabling the existing `sparse_l1_avx2` sparse path (with
  zero-check) at L1=16 (slower than row-major dense by 6%; slower
  than column-major dense by ~10%)

## Measurements planned before the next NPS investigation

Three cheap data-gathers before deciding which remaining lever to
pursue:

1. **Direct-vs-xray CPU split in `push_threats_for_piece`** — cfg
   gate per-section counters inside the function, bench with
   `--features profile-threats`. Decides whether the hybrid threat
   path (vectorise direct only, keep x-ray scalar) is worth 1-2
   days or not.
2. **Walk-back distance histogram at `materialize` refresh
   fallbacks** — extend the existing `profile-materialize` infra
   to track "how many ancestors back is the nearest computed
   frame" when refresh fires. Decides whether walk-back gives
   >2% or <1%.
3. **Delta-count histogram in `apply_threat_deltas`** — bucket
   move types by n_deltas. If there's a long tail of moves
   emitting 20-30+ deltas, capping or batching could help. If
   uniform 8-12, no win.

All three are 1-hour tasks. None require a retrain or an SPRT.

## Why the 1.77× gap is structural, not micro-optimisation

v5 has no threats and no hidden layers. v9 pays:
- apply_threat_deltas: 12.7% CPU (zero on v5)
- push_threats_for_piece: 9.3% CPU (zero on v5)
- threat refresh work: ~2% CPU (zero on v5)
- dense_l1_avx2 + pairwise pack + hidden SCReLU: ~10% CPU (zero on v5)

That's ~34% of v9's CPU on infrastructure v5 doesn't have. Even if
our NNUE implementation were as efficient as Reckless's, we'd still
be ~1.5× slower than v5. Reckless is ~1.8× slower than its v5
equivalent (bench on their repo with a v5-ish testnet), so we're
already in a similar ballpark of "architecture tax" — the remaining
gap to *Reckless* is the recoverable part, and that's 8-12% NPS
per the hotspot decomposition, not 44%.

The framing that survives scrutiny: **Coda v9 is structurally 1.5×
slower than Coda v5 because of architecture, and the remaining
~15-20% gap to Reckless is what NPS work can realistically close.**
Matching v5 NPS requires changing architecture (fewer features,
fewer params, or quantisation scheme), not micro-optimisation.

---

## Reference: Coda vs Reckless algorithmic differences

| Aspect | Coda v9 | Reckless |
|---|---|---|
| L1 weight layout | input-chunk-major **now** (was output-row-major) | input-chunk-major |
| PSQ incremental update | single-pass **now** (was N-pass) | single-pass (`add1_sub1`/`add1_sub2`/`add2_sub2`) |
| Threat delta copy+apply | single-pass **now** (was 2-pass) | single-pass |
| Threat delta algorithm | scalar 8-direction ray walks + magic lookups | `VPSHUFB`-based ray-order shuffle + bit tricks |
| Threat feature set | direct + per-slider x-ray-through-blocker | direct only |
| `PARAMETERS` | runtime-sized `Vec` through `&self` | `static PARAMETERS = include_bytes!(env!("MODEL"))` — fully const dims |
| VNNI | emulated via `maddubs+madd` on AVX2 | uses `vpdpbusd` on hardware that has it |
| Const-generic dims | runtime `self.*` fields | `const L1_SIZE: usize = 768;` etc. |

Three items landed today. Two remain structurally different
(threat algo, const-dim PARAMETERS) — both shelved per the
cost-benefit analysis above.
