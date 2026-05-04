# PGO regression on v9 — investigation 2026-05-04

`make pgo` produces a binary that is **−12.5% NPS slower** than the
non-PGO build on v9 (1,330K → 1,160K, Zen 5, worker paused, 5-run
median). On v5 the same workflow gave +3% NPS. This doc records what
was tested and why each hypothesis was rejected, plus the mechanism
the data points to.

## Replication

Worker paused for clean numbers:

| Build | Branch | NPS | vs no-PGO main |
|---|---|---:|---:|
| `make` | main | 1,330K | baseline |
| `make pgo` | main | 1,160K | **−12.7%** |
| `make pgo` | feature/cfg-dispatch-bundle | 1,159K | −12.9% |
| `make` | feature/cfg-dispatch-bundle | 1,304K | −2.0% |

PGO regression is reproducible and roughly the same magnitude on the
cfg-dispatch-bundle branch.

## Hypotheses tested and rejected

### H1 — Stale v5-era PGO training profile (REJECTED)

**Premise:** PGO was trained when only v5 paths existed; v9 added
threat enumeration / `apply_threat_deltas` / `ThreatStack` paths that
weren't part of the profile, so PGO mis-laid them in cold blocks.

**Why rejected:** `make pgo` runs `cargo pgo instrument build` →
`./coda bench 13` → `cargo pgo optimize build` every time. The
profile is captured fresh on the current v9 binary. Not stale data.

### H2 — Runtime ISA dispatch (`if has_avx512 …`) confusing PGO (REJECTED)

**Premise:** the runtime dispatch trees in `forward_with_l1_pairwise_inner`,
`apply_threat_deltas`, `materialize`, etc. let PGO record different
arms taken at different times, leading to bad branch-predictor and
layout decisions.

**Why rejected:** `feature/cfg-dispatch-bundle` (commit `166ec32`)
replaces all the runtime gates with `cfg!(target_feature = "…")`,
which resolve to compile-time constants. PGO on that branch is
**−12.9%** vs no-PGO main — same regression. cfg-dispatch is not
the lever.

### H3 — iTLB thrash from PGO code layout (REJECTED, was a measurement artifact)

**Initial reading:** under `perf stat`, PGO-main showed `iTLB-load-misses
≈ 9,300` vs `~1,300` for no-PGO. Looked like 7× iTLB pressure.

**Why rejected:** the OB worker had resumed between bench runs.
Re-measuring with worker paused gave PGO `iTLB-load-misses ≈ 850` vs
no-PGO ≈ 1,200 — **PGO has fewer iTLB misses than no-PGO** at clean
contention. The 9,300 number was a contended-system measurement
artifact.

### H4 — Inline-threshold over-aggression (REJECTED)

**Premise:** PGO inlines too aggressively on v9-class code, blowing up
the hot path beyond what fits in icache or what the register allocator
can handle.

**Tested values via `RUSTFLAGS="-Ctarget-cpu=native -Cllvm-args=…"`:**

| Knob | NPS | Δ from default PGO |
|---|---:|---:|
| Default PGO | 1,156K | baseline (already regressed) |
| `inlinehint-threshold=100` | 1,160K | +0.3% |
| `inlinehint-threshold=50` | 1,160K | +0.3% |
| `inline-threshold=100` | 1,130K | −2.2% (worse) |
| `inline-threshold=25` | 1,158K | +0.2% |
| `hot-cold-split=false` | 1,146K | −0.9% |
| `disable-loop-unrolling` | 1,162K | +0.5% |
| `-Copt-level=s` | 1,160K | +0.3% |
| `-Copt-level=2` | 1,160K | +0.3% |

None of the LLVM inline-control knobs recovered more than ~0.5%. The
problem isn't bulk-tunable via thresholds.

## What the data actually points to

### Clean perf-stat comparison (worker paused)

| Metric | no-PGO | PGO | Δ |
|---|---:|---:|---:|
| Instructions | 9.55 G | 10.74 G | **+12.4%** |
| Cycles | 4.07 G | 4.61 G | +13.3% |
| **IPC** | **2.37** | **2.33** | basically equal |
| `branch-misses` | 34.2 M | 37.0 M | +8% |
| `iTLB-load-misses` | ~1,200 | ~850 | PGO **better** |
| `L1-icache-load-misses` | 43.3 M | 38.7 M | PGO **better** (−10.6%) |
| `L1-dcache-load-misses` | 734 M | 743 M | basically equal |
| Elapsed | 0.735 s | 0.839 s | **+14.1%** |

**The whole regression is the +12.4% executed instructions.** PGO's
cache layout is *better*, IPC is unchanged, branch-predict is barely
worse. None of those explain the slowdown. It's the raw work the CPU
has to do per node.

### `perf record` symbol-level breakdown

PGO redistributes which function bears the cycles, even though total
work goes up:

| Function | no-PGO % | PGO % |
|---|---:|---:|
| `ThreatStack::ensure_computed` | 21.0% | 26.1% |
| `negamax` | 11.6% | **19.0%** |
| `forward_with_l1_pairwise_inner` | 8.0% | (folded into `forward_with_threats` 7.7%) |
| `MovePicker::next` | 6.1% | **10.3%** |
| `push_threats_for_piece` | 6.1% | (inlined, gone from top-10) |
| `finny_batch_apply` | 5.0% | (inlined, gone from top-10) |
| `MovePicker::pick_best` | 3.7% | (inlined, gone from top-10) |
| `refresh_accumulator` | not in top-10 | **9.8%** |
| `make_move` | 2.9% | **7.9%** |

Several small standalone functions disappear from the top-10 in PGO —
**they got inlined into their callers**. The callers' percentages go
up correspondingly (negamax doubles, MovePicker::next doubles, make_move
nearly triples).

### Standalone-function presence diff

```
no-PGO unique SIMD/AVX-512 standalone functions:
  simd512_pairwise_pack_fused        ← AVX-512 SIMD pack
  simd_acc_fused_avx2
  simd_acc_fused_avx512

PGO unique SIMD/AVX-512 standalone functions:
  simd_acc_fused_avx512
  simd512_screlu_dot_i8              ← v5/v7 path
  simd_pairwise_pack_fused           ← AVX-2 SIMD pack (instead of AVX-512!)
  simd_screlu_pack
```

`simd512_pairwise_pack_fused` was inlined and DCE'd; the in-PGO call
inside `forward_with_l1_pairwise_inner` is to the AVX-2 sibling. The
AVX-512 inlined version still exists inside the function body
(179 ZMM instructions), so the AVX-512 path runs at production time,
but the function is fatter than the no-PGO equivalent.

## Mechanism

**PGO + Rust + Coda v9 = bad inlining decisions.** Specifically:

1. v9 has many small SIMD helper functions (`finny_batch_apply`,
   `simd512_pairwise_pack_fused`, `acc_add`, `acc_sub`, `apply_deltas_avx512`,
   etc.) — most marked `#[target_feature(enable = "...")]`.
2. PGO sees these as hot (called per-eval / per-make_move) and
   aggressively inlines them into callers.
3. After inlining, the standalone copies are dead-code-eliminated.
   That's why we see `simd512_pairwise_pack_fused` missing from PGO
   binary and `MovePicker::pick_best` disappearing from the symbol table.
4. **The inlined versions execute more instructions than the
   standalone-call versions did.** Possibly:
   - Register-allocator decisions are worse in larger function bodies
   - Inline-site duplication of prologue/epilogue equivalents
   - Tail-duplication of common code paths
   - Loss of LLVM's normal threshold-based inlining heuristics in
     favour of profile-driven ones that don't trip the same passes
5. icache/iTLB pressure is *not* the bottleneck — PGO's layout is
   actually better there. The bottleneck is just: more work to do.

## What didn't help

- `inline-threshold` / `inlinehint-threshold` knobs (any value)
- `-Copt-level=2` / `-Copt-level=s`
- `-Cllvm-args=-hot-cold-split=false`
- `-Cllvm-args=-disable-loop-unrolling`
- `cfg(target_feature)` dispatch (`feature/cfg-dispatch-bundle`)
- Re-running with explicit `RUSTFLAGS="-Ctarget-cpu=native"` (cargo-pgo
  drops `.cargo/config.toml`'s rustflags during instrumentation —
  worth fixing in the Makefile, but doesn't recover the regression
  because the instrument step is only one half of the issue)

## Cargo-pgo bug discovered en route

`cargo pgo instrument build` does NOT inherit `rustflags = ["-C",
"target-cpu=native"]` from `.cargo/config.toml`. The instrumented
binary lacks AVX-512 functions despite the host having AVX-512.
Setting `RUSTFLAGS="-Ctarget-cpu=native"` explicitly before running
`cargo pgo` fixes the instrumented build but the optimized output
still regresses for the inlining-decision reasons above. Worth fixing
the Makefile to set RUSTFLAGS regardless, as a defensive default.

## Suggested next steps

In rough order of cost:

1. **Skip PGO on v9** — current state is the right call. Re-evaluate
   if the hot-path code surface stabilizes / shrinks meaningfully.
2. **Manually annotate `#[inline(never)]` on small SIMD helpers** —
   prevents PGO from inlining them, restoring the no-PGO behaviour
   for those specific functions while still getting PGO's branch
   layout benefits elsewhere. Mechanical change; medium effort to
   identify the right set.
3. **Try BOLT post-link** — BOLT is a separate post-link optimizer
   that handles basic-block reordering using profile data, *without*
   making inlining decisions. Industry standard for "PGO works but
   for the wrong reasons." Larger setup cost.
4. **Wait for / test newer Rust+LLVM** — the inlining heuristics may
   improve. Less actionable.
5. **Lengthen the PGO training run** (`coda bench 15`, multiple positions)
   — gives PGO better profile data. Doesn't address the core inlining
   issue but may shift the balance.

## Banked

- **Methodology:** before drawing conclusions from `perf stat`, verify
  the OB worker is paused. Contention can swing TLB / cache-miss
  counts by an order of magnitude. (The earlier "7× iTLB" was a
  worker-active artifact; clean run shows PGO has fewer iTLB misses
  than no-PGO.)
- **Diagnostic technique:** when total instructions go up but cache
  metrics improve, the answer is in the function-body / inlining diff.
  `perf record -e cycles` + symbol-level report shows which functions
  the work moved into. `objdump | grep '^[0-9a-f]+ <_ZN4coda'` shows
  which standalone functions disappeared (got inlined).
- **PGO ≠ free.** Profile-guided is good when the profile picks the
  right things to optimize. On v9-class codebases with many small
  SIMD helpers and `#[target_feature]` attributes, PGO's inlining
  heuristics seem to over-inline and produce more total work even
  with better cache layout. Tighten inline knobs first (didn't help
  here); failing that, BOLT or manual hints are the levers.

---

*Investigation 2026-05-04. Worker paused for measurement; restored at
end of session. SPRT queue not affected.*
