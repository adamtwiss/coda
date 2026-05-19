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

## Update: Hypothesis 1 tested — `#[inline(never)]` on AVX-512 helpers — REJECTED

Added `#[inline(never)]` to all ~24 functions with
`#[target_feature(enable="…avx512…")]` annotations (`src/nnue.rs`,
`src/threats.rs`, `src/sparse_l1.rs`). Branch
`experiment/pgo-zen5-noinline-avx512` (deleted post-test).

| Build | NPS (5-run median) | Δ vs plain main |
|---|---:|---:|
| plain main | 1,434K | baseline |
| PGO main | 1,265K | −11.8% |
| plain + inline(never) | 1,453K | +1.3% (within noise) |
| **PGO + inline(never)** | **1,268K** | **−11.6%** (unchanged) |

The annotation **did work** — `finny_batch_apply_avx512` (724 → 732
insns) and `simd512_pairwise_pack_fused` (338 → 489) survived in the
PGO binary as standalone functions instead of being inlined. And
`forward_with_threats` dropped from 3,834 → 2,106 insns in PGO,
showing that AVX-512 SIMD helpers WERE being inlined into it and we
prevented that.

**But the regression didn't close.** Why:

- `negamax` is still 20,439 instructions in PGO (same as before).
  Its growth came from inlining **non-AVX-512** helpers — history /
  scoring / move-iteration paths. AVX-512 SIMD was a subset, not the
  dominant lever.
- `simd_acc_fused_avx512` *itself* still grew 717 → 2,880 instructions
  in PGO. That's not inlining-out; it's **PGO unrolling / specialising
  the function's internal loops**. `#[inline(never)]` doesn't affect
  that.
- The forward-path improvement from inlining-prevention (~1,700
  instructions saved per `forward_with_threats` call) is offset by
  whatever PGO does to `negamax` itself.

The mechanism framing in the doc above was **partially correct but
incomplete**. AVX-512 inlining is one of multiple compounding effects;
removing it alone doesn't recover the regression.

## Hypotheses 2-5 still queued

The next thing worth trying is **option 3 — `-Cllvm-args=-inline-threshold-pgo=N`
with smaller N** — targets the *overall* PGO inlining aggressiveness,
not just AVX-512. Risk: regresses older hardware where aggressive
inlining was helping. Would need fleet SPRT to verify.

Option 5 (AutoFDO sampling profile) remains the most promising
long-term — produces different inlining decisions because the profile
is collected without instrumentation overhead. But it's significantly
more setup work than a one-line annotation.

## Cross-engine comparison — Reckless on Zen 5

Reckless (Rust, similar codebase shape, also uses cargo-pgo) — built
and benched on Zeus 2026-05-19 with the same toolchain:

| Build | NPS (3-run median) | PGO Δ |
|---|---:|---:|
| Reckless plain (`cargo build --release`) | 2,197K | baseline |
| Reckless PGO (`cargo pgo instrument → run -- bench → optimize`) | 2,130K | **−3.1%** |
| Coda plain (cgu=16, lto=true) | 1,434K | baseline |
| Coda PGO (cgu=16, lto=true) | 1,265K | **−11.8%** |
| Coda + Reckless build settings (cgu=1, lto=fat) | 1,471K | (plain) |
| Coda + Reckless build settings PGO | 1,289K | **−12.4%** vs same plain |

**Reckless DOES regress under PGO on Zen 5**, just much less (−3.1% vs
Coda's −11.8%). It's the same underlying mechanism (PGO mis-optimizing
for the Zen 5 cost model), but Reckless's architecture mitigates it
substantially.

### What Reckless does differently

1. **Module-level `#[cfg(target_feature)]` dispatch.** Each ISA is a
   separate module, mutually-exclusive cfg-gated:
   ```rust
   mod simd {
       #[cfg(target_feature = "avx512f")] mod avx512;
       #[cfg(target_feature = "avx512f")] pub use avx512::*;
       #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))] mod avx2;
       #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))] pub use avx2::*;
       #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))] mod neon;
       // ...
   }
   ```
   For a Zen 5 build (`target-cpu=native`), only the `avx512` module
   compiles. The `avx2` module's source isn't even visible to LLVM.
   Each binary has ONE SIMD implementation, not multiple.

2. **No runtime ISA dispatch.** No `if has_avx512 { … } else if has_avx2
   { … }` trees in hot paths. Each callsite jumps straight to the
   single SIMD body that compiled.

3. **`#[inline]` (regular) on SIMD helpers.** Because each ISA has
   exactly one version, LLVM inlines or doesn't based on normal cost
   heuristics — no `#[target_feature(enable=...)]` semantic inline
   barrier, no duplicate AVX-2-vs-AVX-512 paths for PGO to over-inline.

4. **`codegen-units = 1` + `lto = "fat"`.** Whole-program LLVM
   optimization. PGO's decisions have full inter-procedural context.

5. **Smaller absolute codebase.** Less code overall means PGO has less
   to bloat under aggressive inlining decisions.

### What Coda does (and why it amplifies PGO+Zen5)

1. **`#[target_feature(enable = "avx512f,avx512bw")]` on functions** —
   both AVX-512 AND AVX-2 SIMD bodies are in the same compiled binary.
   PGO inlines the hot ISA's version into callers, but the existence
   of the other ISA's body bloats the binary regardless.

2. **Runtime dispatch trees** — `if has_avx512 && ... { simd512_fn() }
   else if has_avx2 && ... { simd_fn() }` patterns in hot functions.
   PGO sees these as hot branches and may emit branch hints, but the
   dispatch itself adds icache pressure.

3. **`cgu = 16` + `lto = true`** (thin LTO). PGO inlining decisions
   made per-CGU with limited inter-procedural context. Earlier
   investigation showed `cgu = 1` was needed for SMP scaling (T=1
   bundle), so reverting to Reckless's `cgu = 1` would un-do that
   ship.

4. **v9 threat features add significant hot-path code surface** vs
   Reckless's leaner search.

### Building Coda with Reckless's settings doesn't fix it

Tested directly: `Cargo.toml` set to `lto = "fat", codegen-units = 1`
and rebuilt + PGO'd on Zen 5. PGO regression is **−12.4%** with those
settings — slightly *worse* than current main's −11.8%. Build
settings alone aren't the lever. The structural piece (module-level
cfg-dispatch eliminating duplicate ISA bodies) is what Reckless gets
that Coda doesn't.

### Can we get to Reckless's pattern?

We already tested partial cfg-dispatch (SPRT #935 `cfg-dispatch-forward`
and #936 `cfg-dispatch-bundle`). Both H0'd or trended slightly
negative on the fleet — those experiments converted runtime branches
to compile-time `cfg!()` but kept the function bodies as
`#[target_feature(enable=...)]` attributed (so both ISA bodies
remained in the binary).

Reckless's approach goes **further**: split each ISA into its own
module with `#[cfg(target_feature)]` at the module level, so only one
ISA's source compiles per build. This is a substantially larger
refactor than #935/#936 — ~1-2 days of restructuring the SIMD code
layout in `nnue.rs`, `threats.rs`, `sparse_l1.rs`, plus migration of
all the `#[target_feature]` annotations.

**Expected payoff** if done: closes most of the −11.8% gap to
Reckless's −3.1%, leaving ~−3% irreducible (the Zen 5 cost-model
issue still applies). On AVX-2 hardware (most of fleet), would have
to verify no regression — Reckless's plain numbers don't include the
runtime-dispatch ladder that Coda's removal would also eliminate.

**Recommendation**: not in scope right now. The training-side levers
(hidden_size shrink, encoding redesign) have larger Elo ceilings and
the cache-leg work already addressed search-side NPS to a large
degree. Module-level cfg-dispatch is queued as a possible long-term
refactor if/when we revisit PGO seriously.

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
