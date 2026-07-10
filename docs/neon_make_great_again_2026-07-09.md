# Make NEON Great Again — aarch64 SIMD catch-up plan (2026-07-09)

Status tracker for closing the NEON-vs-x86 SIMD gap. aarch64 is a
first-class target (Apple M-series, Graviton) but carries **zero
fleet-Elo weight** — every SPRT/OB/Lichess/CCRL target is x86. So this
work is opportunistic NPS for ARM deployment, not a strength lever.
Correctness parity is cheap and safe (scalar oracle + parity/fuzz
harness already exist); the wins are output-identical, so **no SPRT is
needed or possible** (OB is x86) — acceptance is "parity tests pass +
NPS up".

## How we got behind

Last real NEON kernel work was `1af1bdc` (2026-06-10, fix
`neon_pairwise_dot` + vectorize L2 + backfill parity tests). Everything
below landed on x86 **after** that date with no NEON counterpart:

| date | commit | x86-only work | NEON status |
|------|--------|---------------|-------------|
| 2026-06-15 | `bf695a5` | AVX-512 VNNI column-major L1=32 kernel | — |
| 2026-06-15 | `02d7884` | AVX-VNNI (YMM) column-major L1=32 kernel | — |
| 2026-07-03 | `4a0d525` | AVX2-host tier-1 locality + dispatch batch | partial |
| 2026-07-03 | `1cf4ff7` | maddubs-pair fused AVX2 L1=32 kernel | — |
| 2026-07-04 | `7bc1840` | Splat A: AVX-512 x-ray threat-delta enumerator | — |
| 2026-07-04 | `9e5e7bf` | Splat B: AVX2 fallback for the splat enumerator | — |

## What IS at parity (no action)

FT / accumulator / activation / L2 / threat-apply are fully mirrored and
parity+fuzz tested (17 NEON tests, all green on the RK3588 dev box
2026-07-09):
`simd_acc_fused_neon`, `neon_acc_add/sub`, `finny_batch_apply_neon`,
`neon_screlu_pack`, `neon_pairwise_pack{,_fused}`, `neon_crelu_dot`,
`neon_screlu_dot_i8`, `neon_pairwise_dot`, `l2_fmadd_neon_x32`,
`screlu/crelu_f32_neon_x32`, `dot_fmadd_neon_x32`, `apply_deltas_neon`,
`add_weight_rows_neon`.

## Test hardware & feature matrix

| feature | RK3588 dev box (A76+A55) | MacBook Air M5 | gates |
|---------|--------------------------|----------------|-------|
| `neon`/asimd | ✅ | ✅ | everything today |
| `dotprod` (SDOT/UDOT) | ✅ | ✅ | Gap 1 signed-dotprod path |
| `i8mm` (USDOT) | ❌ | ✅ (M2+) | Gap 1 clean u8×i8 fused dot |

The RK3588 is exactly the machine that proves the **no-i8mm fallback** is
needed. The M5 validates the USDOT/i8mm fast path. Between them, all
Gap-1 tiers get real coverage.

**NPS-measurement caveat:** the RK3588 is big.LITTLE. Raw `coda bench`
NPS is meaningless unless pinned to the A76 cores
(`taskset -c <a76-ids> ./coda bench`) — the scheduler otherwise floats
onto A55 LITTLE cores and NPS drops ~2–3×. Correctness tests don't care.

## High-level plan (3 steps)

### Step 1 — dotprod L1 int8 matmul  ← DOING NOW
**Gap:** `neon_l1_int8_dot{,_x4,_sparse}` (nnue.rs) use `vmlal_s16`
widening — 4 widening MLALs per 16-byte chunk. There is **no ARM dotprod
anywhere in the tree**. x86 has the whole VNNI (`VPDPBUSD`) family; the
ARM analogue is `USDOT`/`SDOT`.
**Fix:** runtime-detect `i8mm`/`dotprod` (today `detect_neon` just
returns `true`, no sub-features) and add a three-tier L1 path:
USDOT (i8mm, clean u8×i8) → SDOT (dotprod, sign-offset for the
u8×i8 mismatch) → current `vmlal` fallback.
**Reward:** hottest inference kernel; ~4 MLALs/chunk → 1 dot instr.
Output is bit-identical (same int32 accumulation), so pure NPS, zero
eval risk.
**Testable:** dotprod path here today; i8mm path on M5.

**Toolchain wrinkle (verified 2026-07-09, rustc 1.94.1 stable):** the
`vdotq_s32` / `vusdotq_s32` **intrinsics are still unstable**
(`stdarch_neon_dotprod` #117224, `stdarch_neon_i8mm` #117223). Coda is
stable-only, so we emit the instructions via **inline `asm!`**
(`sdot`/`usdot` on `.4s,.16b,.16b`), gated by
`#[target_feature(enable = "dotprod" | "i8mm")]` — both confirmed to
assemble on stable, and `sdot` verified numerically correct on the
RK3588. When the intrinsics stabilise, swap the asm for them.

**SDOT sign-correction math (dotprod-only path, no i8mm):** activations
`u ∈ [0,254]` (u8) × weights `w ∈ [-128,127]` (i8). SDOT needs both
signed, so map `s = u XOR 0x80` (= `u − 128` in i8). Then
`Σ u·w = Σ s·w + 128·Σ w`. Per neuron: one SDOT for `Σ s·w`, plus a
`128·Σ w` correction. The XOR of activations is shared across all 4
neurons in the x4 kernel. v1 computes `Σ w` on the fly via a second
SDOT against an all-ones vector (self-contained, no loader changes);
**follow-up NPS opt:** precompute per-neuron `128·Σ w` at load to drop
the wsum dot. USDOT (i8mm) needs no correction — it does u8×i8 directly.

### Step 1b — precompute the SDOT `128·Σw` correction — TESTED, DROPPED
**Idea:** precompute `128·Σw` per neuron at load (in `l1_corr`) so the
SDOT kernel drops the second per-chunk "ones" SDOT it uses to compute Σw
on the fly, halving the SDOT count on dotprod-only cores.
**Result (2026-07-10, RK3588 A76, branch `neon/dotprod-precompute-corr`,
not merged): a ~2% REGRESSION** (6/6 interleaved runs slower; on-the-fly
median ~237.8k vs precompute ~231.9k nps; node count identical 2381675,
tests green). **Diagnosis: the x4 kernel is LOAD-bound, not SDOT-bound**
— 5 loads/chunk (1 activation + 4 weights) dominate, and the "ones" SDOT
pipelines for free on the A76's dual NEON issue, so halving SDOTs buys
nothing while the extra `l1_corr` load + bounds-check costs a hair.
Kept the simpler on-the-fly kernel on main. **Lesson: this L1 kernel is
memory-bound on the A76 — future NPS must come from *fewer/better-laid-out
loads* (Step 2), not fewer arithmetic ops.**

### Step 2 — column-major / L1=32-specialised NEON kernel
**Gap:** x86 grew four L1=32 kernels (`DenseAvx512VnniL1_32`,
`DenseAvxVnniL1_32`, `DenseAvx2L1_32{,X2}`) for the production shape.
NEON `select_l1_kernel` has one arm — `NeonX4` (row-major) — for all
shapes.
**Fix:** column-major L1=32 NEON kernel over the `l1_weights_sparse`
layout to amortise input loads.
**Decision (updated 2026-07-10):** Step 1b showed the kernel is
**load-bound on the A76**, so a layout that reduces/amortises loads is
now the *only* remaining lever that could add NEON L1 NPS — this raises
Step 2's priority from "likely skip" to "the thing to try if we want more
mobile NPS." Still uncertain and more work (column-major over
`l1_weights_sparse`, needs its own parity + bench). Not started. Pure
NEON, fully testable on the RK3588.

### Step 4 (minor) — aarch64 TT prefetch — CLOSED (neutral, abandoned)
**Gap was:** `TranspositionTable::prefetch` (tt.rs) only issued
`_mm_prefetch` on x86_64 — aarch64 got no TT prefetch. ARM has
`PRFM PLDL1KEEP,[addr]` (inline asm). Implemented on branch
`neon/tt-prefetch`.
**Conclusion (2026-07-10): dropped — measured NPS-neutral on BOTH the
RK3588 A76 and a MacBook Air M5** (careful interleaved runs controlling
for the M5's fanless throttling; node count unchanged, ~0 NPS delta on
both). Two independent ARM cores agreeing on "no effect" is conclusive.
Intel's win didn't transfer — the A76 and Apple Silicon both have deep
OoO windows + strong hardware prefetchers that already hide the TT-probe
miss. Not merged; do not revisit without a specific new ARM target that
shows a TT-probe stall in a profile.

**Gotcha logged:** the first pushed version of the branch was missing the
actual PRFM code — the tt.rs edit had been `git stash`ed during
baseline-binary building and never popped, so an initial M5 "1.43M
highest-ever" reading was on a prefetch-LESS binary (pure turbo/thermal
variance). Lesson: when building a baseline binary mid-change, commit
first or verify the feature is present in the built binary; and always
confirm node-count identity AND a real code diff before trusting a
before/after.

### Step 3 — NEON threat-splat enumerator
**Gap:** `threats_splat.rs` is `#![cfg(target_arch = "x86_64")]` end to
end. The default-ON x-ray-aware threat-delta enumerator (+6.7% NPS on
Zen 5) has no NEON path; aarch64 falls back to scalar
`push_threats_for_piece`. Largest ARM-vs-x86 NPS divergence.
**Blocker:** the AVX-512 path leans on VBMI2 compress-store for
emission — **no equivalent on any ARM core, M5 included**. A NEON port
needs a from-scratch mask-drain emission design.
**Decision:** big effort, not justified while ARM carries no fleet-Elo.
Defer; schedule deliberately if/when ARM deployment matters more.
Correctness would be parity-testable against the scalar oracle.

## Progress log

- 2026-07-09: doc created; findings + 3-step plan. Starting Step 1.
- 2026-07-09: **Step 1 implemented** (branch `neon/dotprod-l1-kernel`).
  `detect_dotprod`/`detect_i8mm`; `NeonI8mmX4`/`NeonDotprodX4` kernels
  (inline asm) selected ahead of `NeonX4`; three NEON arms unified under
  one macro; fuzz + boundary tests over the full [0,254] range.
  - **Correctness:** all 19 NEON tests green on the RK3588; the new
    dotprod fuzz/boundary tests validate the SDOT path here (i8mm path
    runtime-gated, compiles, runs on M2+/M5). End-to-end **bench nodes
    identical** to main (2090755 both), i.e. bit-identical eval.
  - **NPS:** on prod net `net-E161C665.nnue` (v10, L1=32/L2=32), pinned
    to an A76 core, dotprod ~193–208k nps vs vmlal main ~149–152k nps —
    **≈ +30% NPS** on the hot L1 kernel. (Unpinned/big.LITTLE numbers are
    noisier; pin to A76 cores 4–7 via `taskset -c 6`.)
  - x86 is untouched (all new code is `#[cfg(target_arch = "aarch64")]`),
    so zero fleet effect; not an OB/SPRT candidate.
  - **Open follow-ups:** (a) validate the i8mm/USDOT path numerically on
    the M5; (b) precompute per-neuron `128·Σw` at load to drop the SDOT
    wsum dot (further NPS); (c) then reassess Step 2.
- 2026-07-10: **USDOT/i8mm path measured on a MacBook Air M5** —
  ~1.06M → ~1.27M nps, **≈ +20% NPS**. Confirms the narrow-vs-wide-core
  prediction (A76 ~+30%, M5 ~+20%: the M5's wider NEON was less
  bottlenecked on the 4-vmlal path, so a smaller but still large relative
  win). Both ARM tiers now have a real NPS win from Step 1. Follow-up (a)
  done; (b) precompute correction and (c) Step 2 reassessment still open.
- 2026-07-10: **Step 4 (TT prefetch) CLOSED — neutral on A76 and M5,
  dropped.** See Step 4 section.
- 2026-07-10: **Step 1b (precompute SDOT correction) TESTED, DROPPED —
  ~2% regression on A76** (kernel is load-bound, not SDOT-bound). See
  Step 1b. Key takeaway: the NEON L1 kernel is memory-bound, so the only
  remaining NPS lever is a load-reducing layout (Step 2), now bumped in
  priority. Step 3 (splat) still deferred. **Shipped and standing: Step 1
  (dotprod/i8mm), +30% A76 / +20% M5.**
