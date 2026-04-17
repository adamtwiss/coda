# Coda vs Reckless: NPS Gap Analysis (2026-04-17)

Consolidated findings from profiling both engines in this session.
Bundles `v9_nps_profile_2026-04-17.md` (Coda-only profile),
`v9_nps_const_generic_test_2026-04-17.md` (kernel-level probe), and the
Reckless cross-engine comparison done in the same session.

Use this doc as the starting point when returning to NPS work. The
individual docs it references have more detail.

## Headline

Coda v9 runs at ~485K NPS. Reckless runs at ~969K NPS. **Real 2× gap**,
not a threading artefact (both measured single-threaded) and not a
node-counting artefact (both count per-child-visit equivalently —
Coda at function entry, Reckless inside `make_move` before recursion).

## Where the gap lives

### Function-level differences (~45% of our CPU time)

| Function area | Reckless % | Coda % | Ratio |
|---|---|---|---|
| NNUE forward pass (L1 matmul + eval wrapper) | 20.2% | ~30% (25.3% inner + ~5% wrapper) | **1.5×** |
| PSQ accumulator refresh | 2.27% | 6.96% | **3.1×** |
| Threat delta generation | ~3% | 8.5% (post-2b-cull) | **2.8×** |
| MovePicker, SEE, TT, movegen | similar (~2-3% each) | similar | ≈1× |

If Coda matched Reckless on all the disproportionate areas (~20pp of
our total CPU time saved), we'd run at ~485K / 0.8 = **~606K NPS**.

### Structural baseline gap (~55% of the gap)

Reckless sits at 969K. Our theoretical post-function-parity state is
~606K. Gap remaining: **~60% more throughput** that no function-level
difference explains. This is a uniform, across-the-board speed advantage.

Strong suspect: **const vs runtime dimensions.**

- Reckless: `const L1_SIZE: usize = 768;` at `src/nnue.rs:56`. Every
  loop, allocation, SIMD call uses the literal 768.
- Coda: `pub hidden_size: usize` (runtime field on NNUENet). Every hot
  loop reads `self.hidden_size` as a variable.

This affects basically every hot loop: forward pass, accumulator update,
threat SIMD, L1 matmul. Compiler can't specialise as aggressively with
runtime values — no full unrolling, no bounds-check folding of slice
indexing, no register-pressure reduction from eliminating indirect
struct reads.

**Estimated contribution of the const-generic advantage: 15-25% NPS.**

## What does NOT work as a fix

### PGO (2026-04-17 investigation)

PGO regresses v9 NPS by ~10-12% on `feature/threat-inputs`:

| Config | v5 NPS | v9 NPS |
|---|---|---|
| Non-PGO (`make openbench`) | 1,438K | 484K |
| `make pgo` default (v5 profile, v5 embedded) | 1,364K | — |
| `make pgo` binary, v9 loaded via `-n` | — | 424K |
| `make EVALFILE=<v9> pgo` (v9 profile + v9 embedded) | — | 427K |
| PGO without embedded net | — | 435K |
| PGO with longer profile run (`bench 16`) | — | 436K |

Root cause:
- PGO instrumentation adds entry/branch counters that slow the bench
  profile run by ~20% (487K → 387K NPS instrumented).
- That slow profile captures a counter-burdened hot path where small
  SIMD functions dominate.
- PGO's inlining decisions are made against this degraded view and
  over-inline small functions into `push_threats_for_piece` etc.,
  bloating the delta-generation path.

**Not** caused by binary size (2.5MB non-embedded binary regresses the
same as 72MB embedded). **Not** caused by net-profile mismatch (v9
profile gives same regression as v5 profile, within noise).

Even v5 on this branch regresses by ~5% under PGO — the v9 code
compiled into the binary affects layout decisions for v5 paths too,
even though those paths aren't executed.

**If we want profile-guided optimisation for v9 later, try AutoFDO**
(sampling via `perf record` instead of instrumentation). AutoFDO sees
the uncounted binary's actual behaviour so layout decisions match
reality. Low priority.

### Isolated const-generic on hot SIMD kernels

Experimentally verified: duplicating `simd_l1_int8_dot_x4` with
`const H: usize = 384` (hardcoded loop bound) produced **no measurable
improvement** (~483K vs ~485K baseline, within noise).

The compiler was already generating near-optimal code for the
innermost SIMD kernel with runtime `h` — LLVM treats SIMD intrinsics as
opaque black boxes and schedules the loop with fixed 32-byte stride
regardless of whether `h` is const-known.

See `docs/v9_nps_const_generic_test_2026-04-17.md` for full detail.

## What WOULD work as a fix

### Real const-generic refactor (~4-8h effort)

`pub struct NNUENet<const H: usize>` with H=768 as the primary
monomorphization. At load time, dispatch to `NNUENet<768>` when
header FT size matches; fallback to the runtime-sized variant for
legacy v5/v7 nets.

Expected gain: **10-15% NPS (~10-15 Elo at bullet TC)**. Closes about
half of the "structural" portion of the gap.

Risk: medium. Touches every `NNUENet` method and every call site, but
each change is mechanical.

### PSQ refresh investigation (~4h effort)

We're 3× slower than Reckless here (6.96% vs 2.27%). Possible causes:
- More king bucket transitions (16 vs 10 → Reckless 10-KB work addresses this)
- Finny cache miss rate differences
- Per-refresh cost differences (SIMD patterns, bookkeeping)

Needs targeted profiling inside `refresh_accumulator`. Potentially
worth +4-5% NPS.

### Threat delta path further cull (~2h effort)

Post-2b-cull we're still at 8.5% vs Reckless's ~3%. Remaining sources:
- Section 2 Z-finding has no cull (similar to 2b but different logic)
- X-ray enumeration in section 1b has genuine cost
- Scalar ray walks in section 2b when they do fire

Could replace scalar 8-direction walks with bitboard magic attacks
(Reckless's approach). Potential +2-4% NPS.

### Reckless 10-KB king bucket layout (already queued as T2)

Reduces PSQ refresh frequency (rank 4-8 king shuffles no longer
trigger refresh). Overlaps with "PSQ refresh investigation" above.
~2-3% NPS bonus on top of the Elo benefit from fewer PSQ rows / more
data per row.

## Stacking estimate

If we do const-generic refactor + PSQ investigation + Reckless 10-KB:

| Change | NPS gain | Effort |
|---|---|---|
| Const-generic NNUENet<768> | +10-15% | 4-8h |
| PSQ refresh inner work | +3-5% | 4h |
| Reckless 10-KB (partial overlap with above) | +1-2% | infra done, needs trained net |
| Section 2 / 1b extra culls | +2-4% | 2h |
| **Stacked** | **+17-26%** | **~12-14h total** |

485K × 1.22 ≈ **580-610K NPS**. Still short of 969K, but inside 1.6-1.7×
rather than 2×. At that level, further micro-optimization hits diminishing
returns and the remaining gap would be genuinely hard-to-close things
like LLVM/Rust version deltas, alignment quirks, and data-layout
differences.

## Sanity check: NPS counting is comparable

Validated the two engines count nodes equivalently:

- **Coda** (`src/search.rs:1526, 2740`): `info.nodes += 1` at entry of
  `negamax` and `quiescence`, after stop-flag check.
- **Reckless** (`src/search.rs:1359` inside `make_move`): counter
  increment inside Reckless's own `make_move(td, ply, mv)` helper (not
  `board.make_move`), called just before recursing into child.

Both count "one node per child visit about to start" — same event,
slightly different instants. Both include NMP null moves, TT-cutoff
nodes, QS nodes. Both skip stop-aborted nodes. Root counted by Coda
but not Reckless (1 node out of millions — negligible).

**We're not chasing rainbows.** The ~2× NPS gap is real per-second
work throughput, not a counting artefact.

## Related docs

- `docs/v9_nps_profile_2026-04-17.md` — Coda profile breakdown (post-2b-cull)
- `docs/v9_nps_const_generic_test_2026-04-17.md` — kernel probe result
- `docs/v9_nps_optimization_plan.md` — earlier historical plan
- `docs/v7_training_guide.md` — current NPS state + training queue
- `Makefile` — PGO comment block with full investigation summary
