# Const-Generic Hot-Kernel Probe (2026-04-17)

## Motivation

Profile comparison vs Reckless (~485K NPS vs ~969K NPS) suggested two
distinct sources of the ~2× gap:

1. **Function-level slowdowns** (PSQ refresh 3×, threat deltas 2.5×,
   forward pass ~1.5×). Sum to ~45% of our CPU time — closing to
   Reckless's parity would recover ~20 percentage points, taking us to
   ~600K NPS at best.

2. **Across-the-board baseline speed advantage** Reckless enjoys, ~60%
   of NPS that isn't explained by any single function. Candidates: const
   vs runtime dimensions, compiler inlining decisions, data-structure
   layout, etc.

The big structural suspect: Reckless has `const L1_SIZE: usize = 768`
at the top of `src/nnue.rs`. Every loop, allocation, and SIMD call uses
the literal `768`. Our equivalent is `self.hidden_size: usize` at runtime.

## Experiment

**Hypothesis**: Replacing runtime `h` with a compile-time constant in
the L1 matmul's hottest function (`simd_l1_int8_dot_x4`) will let the
compiler fully unroll the 12-iteration inner loop and generate tighter
code, measurably improving NPS.

**Method**:
- Duplicated `simd_l1_int8_dot_x4(..., h: usize)` as
  `simd_l1_int8_dot_x4_384(...)` with `const H: usize = 384;` as the
  loop bound.
- Dispatched to the const variant from `forward_with_l1_pairwise_inner`
  when `pw == 384` (always true for v9 w15 xray net).
- Bench: `./coda bench` × 5 runs.

## Result

No meaningful improvement.

| Config | Avg NPS (5 runs) |
|---|---|
| Baseline (runtime `h`) | ~485K |
| Const-384 variant in x4 kernel | ~483K |

Difference is within run-to-run noise (~5K spread across runs in both
configs).

## Interpretation

The compiler was already generating near-optimal code for
`simd_l1_int8_dot_x4` with runtime `h`. LLVM sees:

```rust
let mut i = 0;
while i < h {
    // SIMD intrinsics with +32 stride
    i += 32;
}
```

and schedules the loop with fixed 32-byte stride per iteration regardless
of whether `h` is known at compile time. The SIMD intrinsics are opaque
black boxes to LLVM — it doesn't try to reorder or re-schedule across
them, so there's nothing to gain from unrolling.

## What this means for the real const-generic refactor

The ~15-25% across-the-board gap between Coda and Reckless **does not
come from innermost SIMD kernels being poorly compiled**. Those are
already tight. The gap must come from higher-level structure:

- **Fan-in decisions**: whether the compiler inlines helper calls (like
  `simd_pairwise_pack_fused`, `simd_l1_int8_dot_x4`, etc.) into the
  caller when sizes are const-known. Inlining unlocks cross-function
  scheduling.
- **Eliminating indirect struct reads**: `self.hidden_size`,
  `self.l1_size`, `self.l1_per_bucket` — every one of these is a memory
  load through `&self`. If the compiler knew these were constants, they'd
  become immediate operands or folded into addressing.
- **Bounds check elimination**: slice indexing like
  `&self.l1_weights_8t[gi * pw..(gi + 1) * pw]` has runtime bounds
  checks that depend on `pw`. Const-known pw would let these fold.
- **Register pressure**: if the compiler can inline the whole forward
  path with const dimensions, it can keep more state in registers.

## Implication for next steps

**Isolated const-generic-ing of hot kernels is not worth doing.** It
won't recover the NPS gap.

**The real refactor would be**: monomorphize `NNUENet` (or at least its
forward path) with `const H: usize = 768`. That means:

1. `pub struct NNUENet<const H: usize>` where H is the hidden/accumulator
   width (always 768 for v9 production).
2. At load time, dispatch to the correct specialization based on
   file-header's FT size.
3. All methods on `NNUENet<768>` see H=768 as a constant, enabling the
   fan-in + struct-load elimination + bounds-check + register-pressure
   benefits throughout.
4. Fallback to runtime H for non-standard sizes (for test nets, v5/v7
   legacy, etc.).

Effort: ~4-8 hours of focused refactoring. Risk: medium (touches every
function on NNUENet, but each change is mechanical). Expected gain based
on the cross-engine analysis: 10-15% NPS, ~10-15 Elo at bullet TC.

## What NOT to do

- Don't duplicate individual SIMD functions with const sizes (this test
  showed zero gain; maintenance cost + no benefit).
- Don't const-generic-ify the Vec allocations alone (the struct-field
  reads are the bigger cost).
- Don't assume a quick specialization in one hot function will recover
  the gap — the gap is structural.

## Related files

- `docs/v9_nps_profile_2026-04-17.md` — live profile showing hotspots.
- `docs/v9_nps_optimization_plan.md` — earlier plan (P3 const-generic,
  estimated +5-10%, now likely a low-end estimate).
- `src/nnue.rs` (reverted, no changes committed from this probe).
