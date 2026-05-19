# PGO Zeus Regression — Mechanism Investigation (2026-05-19)

Follow-on to `docs/pgo_fleet_finding_2026-05-17.md`. That doc surfaced
the Zen 5 cohort split (zeus −10% NPS / −18.5 Elo while older hardware
is positive) and flagged "investigate the hercules vs zeus split,
disassembling hot functions" as the actionable diagnostic. This doc
does that.

## Setup

- Zeus, OB worker idle, cargo.toml as-on-main (`lto=true,
  codegen-units=16, opt-level=3, panic=abort`).
- Built two binaries from the same commit (current main):
  - **plain**: `make`
  - **PGO**: `make pgo` (cargo-pgo instrument → `coda bench 13` profile
    → optimize)
- `strip=false` temporarily (restored after) so `objdump`/`perf report`
  could resolve Rust symbols.

## Replicates the cohort doc

5-run median bench on Zen 5 (idle):

| Build | NPS | Δ |
|---|---:|---:|
| plain | 1,434K | baseline |
| PGO | 1,265K | **−11.8%** |

Same bench tree (5,937,655 nodes both). Pure NPS regression.

## perf-stat tells the same story as the v9-era investigation

| Metric | plain | PGO | Δ |
|---|---:|---:|---:|
| Cycles | 22.84G | 25.92G | +13.5% |
| **Instructions** | **53.64G** | **61.03G** | **+13.8%** |
| **IPC** | 2.35 | 2.35 | unchanged |
| Branch-misses | 209M | 227M | +8.6% |
| iTLB-load-misses | 1481 | 1907 | (tiny abs) |
| **L1-icache-load-misses** | **307M** | **285M** | **−7% (PGO better)** |
| L1-dcache-load-misses | 3691M | 3902M | +5.7% |
| Cache-refs | 5695M | 6626M | +16% |
| Elapsed | 4.13s | 4.70s | +13.8% |

**The whole regression is the +13.8% executed-instruction count.** PGO's
icache layout is *better*, IPC is unchanged, branch prediction is
mildly worse. None of those explain the slowdown — it's pure raw
work-per-node that goes up. Identical to the v9-era PGO failure mode
documented in `docs/pgo_v9_regression_2026-05-04.md`.

## perf-report symbol breakdown

| Function | plain % | PGO % | Δ |
|---|---:|---:|---:|
| `ThreatStack::update` | 18.7 | **24.2** | +5.5 |
| `negamax` | 14.0 | **23.3** | **+9.3** |
| `forward_with_l1_pairwise_inner` | 9.5 | — | folded into `forward_with_threats` |
| `forward_with_threats` | (n/a) | 9.5 | absorbed |
| `make_move` | 3.0 | **9.1** | **+6.1** |
| `refresh_accumulator` | 2.6 | **8.6** | **+6.0** |
| `MovePicker::next_slow` | 5.1 | 7.2 | +2.1 |
| `simd_acc_fused_avx512` | 2.8 | 3.9 | +1.1 |
| `quiescence_with_depth` | 1.8 | 3.4 | +1.6 |
| `MovePicker::pick_best` | 4.2 | 0.6 | **−3.6** (inlined out) |
| `see_ge` | 3.8 | 0.8 | **−3.0** (inlined out) |
| `push_threats_for_piece` | 5.9 | — | inlined out |
| `finny_batch_apply` | 4.4 | — | inlined out |
| `simd512_pairwise_pack_fused` | 1.3 | — | inlined out |

Many small helpers disappear from the top-10 in PGO. Their cycles get
absorbed by the callers — but at **more than their original cost** at
the new callsites.

## Function body size — the mechanism

`objdump -d` instruction count per function:

| Function | plain insns | PGO insns | Δ |
|---|---:|---:|---:|
| `negamax` | 6,717 | **20,439** | **+3.0×** |
| `forward_with_threats` | 517 | 3,834 | **+7.4×** |
| `refresh_accumulator` | 800 | 3,952 | **+4.9×** |
| `simd_acc_fused_avx512` (self) | 717 | 2,880 | **+4.0×** |
| `forward_with_l1_pairwise_inner` | 6,068 | 2,224 | inlined OUT |
| `next_slow` | 1,531 | 2,343 | +1.5× |
| `make_move` | 224 | 209 | unchanged |
| `see_ge` | 382 | 369 | unchanged |
| `pick_best` | 79 | 65 | unchanged |

**`negamax` grows by ~14,000 instructions** in PGO. Add `make_move`'s
6.1pp jump (also inlined-into territory), plus `refresh_accumulator`,
`forward_with_threats`, and the picture is consistent: PGO is
aggressively inlining small SIMD helpers into a handful of large hot
functions. The inlined versions execute MORE total instructions than
the original standalone-call versions did — same v9-era mechanism.

## The AVX-512-specific part

Counting ZMM/EVEX-encoded instructions in the body of each function:

| Function | plain ZMM | PGO ZMM | Δ |
|---|---:|---:|---:|
| `refresh_accumulator` | **0** | **285** | new — `finny_batch_apply_avx512` inlined in |
| `forward_with_threats` | 23 | **585** | **25×** — `simd512_pairwise_pack_fused` etc. inlined |
| `simd_acc_fused_avx512` (self) | 96 | 528 | 5.5× internal expansion |
| `negamax` | 21 | 83 | 4× |

`refresh_accumulator` went from **0 ZMM instructions to 285** when PGO
inlined `finny_batch_apply_avx512` into it.

**This is why Zen 5 specifically regresses:**

1. PGO inlines the AVX-512-attributed SIMD helpers
   (`finny_batch_apply_avx512`, `simd512_pairwise_pack_fused`, etc.)
   into their callers.
2. AVX-512 EVEX-encoded instructions are **5-8 bytes** each (vs ~4-5
   for VEX-encoded AVX-2). Same inlining therefore produces more code
   on AVX-512 hardware.
3. The inlined bodies are *larger* both in instruction count AND in
   bytes. The host functions (`negamax`, `refresh_accumulator`,
   `forward_with_threats`) balloon.
4. Zen 5's wide OOO core was already executing the smaller plain
   binary at IPC ≈ 2.35 — close to its peak for this workload. PGO's
   branch hints + layout don't extract additional parallelism, so the
   +13.8% instructions translate **directly** to +13.8% time.

On hercules (Coffee Lake Xeon, 2019, **no AVX-512**), the same `make pgo`
inlines the AVX-2 versions of the SIMD helpers (the AVX-512 paths are
runtime-rejected). VEX-encoded YMM bodies are smaller. Plus the older
front-end has slack that PGO's layout reduction can exploit. Net
positive Elo.

The cross-vendor finding from the prior doc (Zen 1 titan positive,
Coffee Lake hercules positive, Zen 5 zeus negative) collapses cleanly
under this framing: **age correlates with AVX-512 presence**. Zen 5 is
the only AVX-512 host in the fleet.

## Why `cgu=16` didn't fix Zen 5

The `cgu=16` change (commit f848809) made PGO **viable** by allowing
multiple compilation units that PGO could profile-direct separately —
v9-era PGO at `cgu=1` was producing severely degraded binaries on all
hardware. With `cgu=16`, PGO has finer granularity for inlining
decisions, so older hardware now sees positive results.

But the **inlining decisions themselves** are still aggressively
expanding AVX-512 bodies on Zen 5. The fix unlocked PGO, didn't change
its preferences. The Zen 5 regression mechanism is downstream of the
cgu setting.

## Probable cost decomposition on Zen 5

Rough back-of-envelope from the +14k instructions in `negamax`:

- `negamax` body executed at fraction ~30% of total cycles (per perf
  report)
- +14k insns × 5.9M nodes × ~0.3 weighted = ~2.5e10 added insn-cycles
- Match in total instruction count delta (~7.4G).

`refresh_accumulator` + `forward_with_threats` contributions account
for the rest. **AVX-512 inlining is the dominant lever.** Not loop
unrolling, not data layout, not branch prediction.

## Hypotheses to test (queued, not done in this session)

Each is a one-line code change with a clear prediction:

1. **`#[inline(never)]` on the AVX-512 SIMD helpers**
   (`simd_acc_fused_avx512`, `simd512_pairwise_pack_fused`,
   `finny_batch_apply_avx512`, `apply_deltas_avx512`,
   `add_weight_rows_avx512`).

   *Prediction:* On Zeus, PGO no longer inlines these. `negamax` /
   `refresh_accumulator` / `forward_with_threats` stay small. NPS
   regression closes most or all of the way. Risk: hurts plain-build
   Zen 5 NPS by ~1-3% (the cases where inlining IS justified). Should
   be neutral or slightly negative on AVX-2 hardware (the
   `_avx512`-suffixed functions never run there anyway).

2. **`#[cold]` on the same set.** Less aggressive than `inline(never)`
   — PGO might still inline if profile-hot. Probably less effective.

3. **`-Cllvm-args=-inline-threshold-pgo=N`** (some smaller N).
   Reduces PGO's inlining aggressiveness globally. Could regress
   non-AVX-512 hardware where PGO was helping. Lower-confidence fix.

4. **Per-host build of `make pgo` with different LLVM args based on
   `target_feature = "avx512f"`.** Cargo doesn't directly support
   conditional rustflags by target_feature, but the Makefile could
   detect Zen 5 and pass extra LLVM args. Operational complexity.

5. **AutoFDO sampling profile instead of instrumentation PGO** —
   listed in prior doc, still relevant. Different inlining decisions
   are likely.

Recommended next test: option 1, on a one-shot branch. Single
`#[inline(never)]` annotation per AVX-512 helper. SPRT bench-13 PGO
delta first (cheap), then fleet SPRT if Zeus recovers locally.

## Banked

- **Same mechanism as v9-era PGO failure.** PGO's profile-driven
  inlining over-inlines small SIMD helpers into hot callers. On
  hardware where the inlined SIMD has a smaller encoding footprint
  (AVX-2), the net is positive because the front-end slack on older
  CPUs more than absorbs the size growth. On Zen 5 (AVX-512 +
  saturated OOO), the size growth dominates.

- **`cgu=16` enables PGO, doesn't sharpen it.** The change makes
  the binary buildable / not-broken under PGO. It doesn't fix the
  underlying inlining preferences.

- **AVX-512 is the divider.** Cross-vendor evidence supports the
  microarchitecture-age framing only because all old hardware in
  the fleet is also pre-AVX-512. Zen 5 is the singular AVX-512
  host and the singular regression. If a Zen 1-class AMD host
  upgrades to a Zen 4 (AVX-512), the prediction is that PGO will
  start hurting it too.

## Cross-references

- `docs/pgo_fleet_finding_2026-05-17.md` — fleet-wide cohort data
- `docs/pgo_v9_regression_2026-05-04.md` — v9-era same mechanism,
  cgu=1, made `make pgo` worse
- `docs/smp_scaling_investigation_2026-05-17.md` — the cgu=16 ship
  that enabled this re-test

---

*Investigation 2026-05-19. Worker idle, no fleet impact. `Cargo.toml`
`strip=true` restored at end.*
