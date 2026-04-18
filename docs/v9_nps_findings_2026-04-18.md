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

**Considered, not attempted.** Reckless's vectorised threat-delta
code (`src/nnue/threats/vectorized/avx2.rs`) is genuinely a
different algorithm: `VPSHUFB`-based ray-order mailbox shuffle +
bit tricks to find closest occupant per ray, no scalar ray walks.
Could plausibly drop `push_threats_for_piece` from 9.3% to ~1-2%
CPU — **recovering ~7-8% of CPU = ~9-10% NPS gain**.

**Why not**: **feature-set incompatibility**. Reckless's
`refresh` (canonical feature enumeration) emits **direct threats
only**. Coda's `enumerate_threats` emits **direct + per-slider
x-ray-through-blocker** — more features, trained into our net
weights at +110 Elo gain from yesterday's x-ray fix.

Cost-benefit:
- Straight port (lose x-rays): +28% NPS = +28 Elo, -85 to -110
  Elo from feature removal. **Net: -57 to -82 Elo.**
- Port primitives + extend with Coda x-rays: engineering-heavy,
  correctness-critical, expected +10% NPS = +12.5 Elo for
  multi-day effort.
- Drop: 0 change.

Not worth the risk. **Shelved.**

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
| Full end-to-end const-generic NNUE | +5-10% NPS | 2-3 days | Medium |
| Reckless-pattern threat port + extend x-rays | +10% NPS (~12 Elo) | 2-3 days | High |
| AutoFDO (sampling PGO) | Unknown, likely +3-5% NPS | 1 day + tooling | Low |

**The Elo economics favour training/tuning over NPS micro-optimisation
at this point.** The last tune delivered +38.6 Elo in a day. Three
NPS commits in a day delivered +3.8% NPS ≈ +4 Elo.

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
