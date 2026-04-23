# Coda vs Reckless NPS Investigation — 2026-04-23

## Purpose

We are ~2× slower than Reckless at v9 NPS (about 75 Elo after accounting for
the ~20% we spend on the extra xrays). This accounts for roughly 25-30% of
the Elo gap to Reckless. "Run perf and see where time goes" has been
inconclusive because the cost is spread across many small things.

This investigation decomposes the NPS gap into three measurable buckets:

1. **Per-eval cost** — how long a single NNUE forward pass takes
2. **Evals per node** — how often search invokes NNUE (caching / pruning)
3. **Per-node overhead** — everything else (movegen, move ordering, TT)

Microbenches were added to both engines (`coda eval-bench` and Reckless
`evalbench`) so the per-eval cost can be measured directly, isolated from
search. Results below are on a Ryzen 7 9700X (Zen 5, AVX-512 + VNNI) with
`perf` measuring cache + instruction counters.

Goal of this doc: bring real numbers to inform future performance work. Not
to solve the whole gap in one pass.

## Headline numbers (evals/sec, 80k evals per run)

| Engine | ISA | **fresh** (debug scalar) | **refresh** (production SIMD) | **incremental** |
|---|---|---:|---:|---:|
| Coda | AVX-512+VNNI | 92,227 | **1,330,926** | 3,759,675 |
| Coda | AVX-2 only | 90,940 | 1,129,948 | 2,854,728 |
| Reckless | AVX-512+VNNI | — | **1,458,484** | 5,815,343 |
| Reckless | AVX-2 only | — | 1,258,078 | 5,710,523 |

Reckless / Coda ratio (>1 means Reckless faster):

| ISA | refresh | incremental |
|---|---:|---:|
| AVX-512+VNNI | **1.10×** | **1.55×** |
| AVX-2 only | **1.11×** | **2.00×** |

## Key findings

### 1. Fresh (rebuild-from-scratch) eval is nearly tied

On both AVX-512 and AVX-2, Coda's *production* refresh path (`materialize`
with an invalidated Finny cache, going through `refresh_accumulator` +
`finny_batch_apply`) runs at **90% of Reckless's `full_refresh` speed**.
This is effectively a tie — the ~10% margin could be one missed
micro-optimization and would be real but not world-changing.

→ **Fresh-eval speed is not where the 2× NPS gap lives.**

### 2. Incremental eval has the real gap

Incremental eval — push one dirty piece, forward-pass — is **1.55× slower
on AVX-512 and 2.00× slower on AVX-2**. This is the hot path in search.
Almost every node does incremental eval. This matches the end-to-end NPS
gap we see in `coda bench` vs Reckless's equivalent.

### 3. Cache behavior is dramatically worse in Coda

Same 800k incremental evals, measured with `perf stat`:

| Metric | Coda | Reckless | Notes |
|---|---:|---:|---|
| elapsed | 0.242 s | 0.134 s | 1.80× |
| instructions | 3.75 B | 2.02 B | **1.86× more** |
| cycles | 1.33 B | 0.73 B | 1.83× more |
| IPC | 2.82 | 2.78 | **essentially tied** |
| cache-references | 596 M | 19 M | **31.6× more** |
| cache-misses | 3.29 M | 0.20 M | 16.1× more |
| L1-dcache-loads | 1.93 B | 1.03 B | 1.88× more |
| L1-dcache-load-misses | 304 M (**15.72%**) | 5.6 M (**0.55%**) | **28.4× worse miss rate** |
| branches | 645 M | 137 M | 4.7× more |
| branch-misses | 117 k (0.02%) | 22 k (0.02%) | branch predictor is fine |

The CPU is equally happy executing both engines' code (IPC 2.82 vs 2.78).
The problem is **Coda does 1.86× more instructions per eval and touches
31× more cache lines with a 28× worse L1 hit rate**. Coda is memory-bound
in a way Reckless isn't.

Same pattern holds on AVX-2:

| Metric | Coda AVX-2 | Reckless AVX-2 |
|---|---:|---:|
| L1-dcache-load-misses | **7.44%** | **0.57%** |
| cache-references | 528 M | ~76 M |

Same shape, smaller absolute gap. AVX-2 has more latency-hiding opportunity
per memory access, so Coda's cache unfriendliness hurts less on AVX-2.

### 4. `force_recompute` is a debug-scalar trap (not a production problem)

`eval-bench --mode fresh` measures Coda's `NNUEAccumulator::force_recompute`
— a **scalar** inner loop at `src/nnue.rs:3782`:
```rust
for j in 0..h { dst[j] += row[j]; }
```
This path is used by `CODA_VERIFY_NNUE` and the PSQ fuzzer but **not by
production search**. It runs at ~92k evals/sec, 15× slower than the
production refresh path at ~1.33M evals/sec. A similar scalar pattern
exists in `recompute_threats_full` at `src/nnue.rs:3969`.

Rewriting these to use the existing `finny_batch_apply` SIMD helper would
make the debug path (and the NNUE verification mode) usable, but **won't
affect production NPS**. Noted for completeness; low priority.

## What this tells us about the 2× NPS gap

- **Per-eval cost** (bucket 1): contributes ~1.55-2.00× based on incremental
  throughput. This is most of the NPS gap directly.
- **Evals per node** (bucket 2): not measured yet in this pass. Our
  `tt_entry.static_eval` caching exists, but Reckless has **eval-only
  TT writebacks on NNUE-hit + TT-miss** that we don't
  (`Reckless/src/search.rs:425`). Those reduce redundant NNUE calls. Effect
  unknown without instrumented counters.
- **Per-node overhead** (bucket 3): the 4.7× branch delta on incremental
  eval suggests our per-call dispatch has more branches than Reckless's.
  Runtime dispatch via `has_avx2` / `has_avx512` / `has_avx512_vnni` booleans
  on `NNUENet` adds branches that Reckless's compile-time
  `cfg(target_feature)` dispatch eliminates.

## Working hypotheses for the incremental-path gap

Ordered by rough likelihood of being part of the cost:

1. **ThreatEntry is huge per-ply**: `pub values: [[i16; 768]; 2]` =
   3,072 bytes per ply × 256 plies = **786 KB** of threat-accumulator
   state alone. That's half of L2 cache (usually 1 MB). Every push walks
   through this. `threat_accum.rs:67`.
2. **Threat weight array is 48 MB, accessed scattered**. Each incremental
   delta reads one i8 weight row (48 KB per row × up to ~16 deltas per
   move). Reckless's threat-delta generation is vectorised to collapse
   ranges of delta work into single SIMD instructions; ours isn't.
3. **Finny table writes**: `finny[64]` entries × h bytes each. Touched
   on every king-bucket miss.
4. **Runtime SIMD dispatch branches** — every SIMD helper reads a `has_*`
   flag, which LLVM can't eliminate.
5. **Per-ply `threat_deltas: Vec<RawThreatDelta>`** used to be Vec (heap),
   now a `DeltaVec` (inline array of 80 entries = 2560 bytes per ply,
   another 640 KB of state).
6. **Multiple scalar fallback paths inside the forward function** keep the
   dispatched branches' codegen branchy (see the 250-line if/else-if chain
   at `forward_with_l1_pairwise_inner`).

## Reckless's wins (from git history scan)

Compile a shortlist of Reckless commits that measured real Elo/NPS from
NNUE-path changes, for comparison:

| Commit | Change | Elo |
|---|---|---|
| `a244f689` | AVX-2 threat-update register tiling (REGISTERS=8 across full L1) | +5.77 STC |
| `381ac2f3` | AVX-512 threat-update register tiling (REGISTERS = full L1) | +4.90 STC |
| `42399e79` | AVX-512 "byteboard splat" for slider + x-ray threat deltas | +4.33 STC |
| `d09af1b8` | Replace VPMADDUBSW + VPMADDWD(ones) + VPADDD with VPDPBUSD | +3.38 STC |
| `97be0061` | Avoid redundant add/sub ThreatDelta pairs (observer `on_piece_mutate`) | +3.36 STC |
| `04c88767` | AVX-512 VBMI2 vectorized PST feature-index generation | +2.70 STC |
| `45d9cc5a` | AVX-2 port of threat-splat "byteboard" | +2.27 STC |
| `f20b1a97` | AVX-512 non-slider threat-delta splat | +1.79 STC |
| `66cd450f` | Threat refresh path register-tiled 2-feature unroll | +1.55 STC |
| `859d64cd` | AVX-512 `find_nnz` via `_mm512_maskz_compress_epi16` | +4.69 STC |

Structurally similar pieces Coda already has:
- VPDPBUSD VNNI path (landed on this branch 2026-04-21)
- AVX-512 MAP_HUGETLB TT (2026-04-22)
- Finny cache for PSQ
- Input-chunk-major sparse L1 weight layout
- Pairwise CReLU pack

Pieces Coda does NOT have and Reckless does:
- AVX-2 / AVX-512 **byteboard splat** for threat-delta generation (the big
  5-6 Elo commits above). Our `push_threats_for_piece` is scalar loops over
  magic bitboard lookups.
- **Observer-trait `on_piece_mutate`** to avoid redundant add/sub pairs in
  threat deltas (-1.79 to -4.33 Elo each commit).
- **Eval-only TT writeback on NNUE-hit + TT-miss** (caching gap).
- **Compile-time SIMD dispatch** via `cfg(target_feature)` (eliminates
  runtime branches + lets LLVM inline across the decision).

## Pruning gaps (from search-side scan)

Not eval-cost, but evals-per-node lever. Top items from the comparison:

| Item | Coda | Reckless | Reckless Elo |
|---|---|---|---|
| LMP direct-check carve-out | none | skips moves giving direct check | +2.84 STC / **+9.46 LTC** |
| LMP adaptive improvement scaling | binary improving | continuous via `70*improvement/16/1024` | +3.92 STC / **+5.19 LTC** |
| NMP skip on valuable-piece TT capture | none | `tt_bound=Lower ∧ tt_move.is_capture() ∧ victim≥knight` | +3.43 STC / +3.67 LTC |
| NMP cut_node-only gate | non-PV only | `cut_node` only | (gate effect, not measured alone) |
| RFP depth cap | `depth≤7`, linear margin | no cap, `1165*d²/128 + 25d + 30` | (part of wider changes) |
| SEE pruning: capture depth cap | `depth≤6`, margin `-198*d` linear | no cap, `-8d² - 36d` quadratic | +1.53 LTC (decouple) |
| Razoring | none | d=1-2 with `!tt_move.is_quiet()` gate | +1.10 to +3.64 STC |
| Sibling singular extension propagation | none | `+1` to siblings of extended moves | +2.79 STC |
| Bad-noisy futility depth gate | `depth≤4` | `depth<11` | (within #815 overhaul) |

Cumulative unclaimed Elo from these changes (with diminishing returns
accounted): ~25-35 LTC Elo. All are incremental and independently SPRT-able.

## Methodology — how to reproduce

### Coda microbench

```bash
make                                     # build with embedded v9 net
./coda eval-bench --mode fresh       -n net.nnue   # debug scalar
./coda eval-bench --mode refresh     -n net.nnue   # production SIMD
./coda eval-bench --mode incremental -n net.nnue   # incremental
# Add --no-avx512 to force AVX-2 dispatch at runtime
```

Three modes defined in `src/main.rs:EvalBench` handler. Support routines:
- `NNUEAccumulator::force_recompute` (scalar — debug only)
- `NNUEAccumulator::invalidate_for_bench` + `materialize` (production)
- `NNUEAccumulator::push` + `evaluate_nnue` + `pop` (incremental)

### Reckless microbench

Patched into Reckless at `src/tools/eval_bench.rs` + UCI dispatch in
`src/uci.rs`. Isolated "Coda-comparison hack" so it can be cleanly
removed. Same 8 bench positions as `coda bench`.

```bash
cd ~/chess/engines/Reckless
cargo build --release --no-default-features   # skip syzygy (needs clang)
echo "evalbench fresh 10000
quit" | ./target/release/reckless

# AVX-2 only:
RUSTFLAGS='-Ctarget-cpu=x86-64-v3' cargo build --release --no-default-features --target-dir target-avx2
echo "evalbench incremental 100000
quit" | ./target-avx2/release/reckless
```

Reckless's compile-time dispatch means you need two binaries for two
ISAs — the `target-avx2` build has the AVX-2 path only.

### Cache/perf measurement

```bash
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,instructions,cycles,branches,branch-misses \
  <engine command>
```

OB worker must be stopped first (`~/code/OpenBench/ob-worker.sh stop`)
so CPU cores are idle.

## Recommendations

**Phase 1 — per-eval cost (bucket 1, direct NPS):**

1. Audit what `incremental eval` touches per call. The 31× cache-reference
   gap and 15.7% L1 miss rate point at a memory-layout problem, not a SIMD
   problem. Candidates: ThreatEntry layout (786 KB stack), scattered
   threat-weight reads, Finny placement. **Proper profiler run with
   L1/TLB events scoped to the hot loop would pinpoint this in one
   afternoon.**
2. Port Reckless's **AVX-2 threat-update register tiling** (commit
   `a244f689`, +5.77 Elo). Directly addresses the hot path.
3. Consider compile-time SIMD dispatch. Dropping `has_avx2` / `has_avx512`
   booleans and gating kernels via `cfg(target_feature)` + separate
   binaries would eliminate the runtime branches and let LLVM specialize.
   Modest (+2-4% NPS guess), but also required to match Reckless's
   codegen shape.

**Phase 2 — evals per node (bucket 2):**

4. Add the missing **eval-only TT writeback** on NNUE-hit + TT-miss
   (~3 LoC in `search.rs` after `info.eval(board)` at line 1890). This
   is the gap Hercules identified — Reckless seeds the TT with the eval
   so the next visit of the same position from a different move order is
   free. Should be a strict NPS-increasing change.
5. Instrument counters for `nnue_full_evals` / `nnue_incremental_evals` /
   `tt_static_eval_hits`. Prints at end of `coda bench`. Tells us whether
   evals/node differs between engines, which the microbench can't answer.

**Phase 3 — pruning (bucket 2 and 3):**

6. LMP direct-check carve-out — +9.46 LTC Elo in Reckless, tiny diff,
   needs a cheap `is_direct_check(mv)` predicate on the pre-move board.
7. LMP adaptive improvement scaling — +5.19 LTC Elo.
8. NMP "skip on valuable-piece TT capture" — +3.67 LTC Elo.
9. Remove RFP depth cap + switch to quadratic margin.

These three together are the highest-value untapped Elo in the pruning
landscape and don't touch NNUE.

**Phase 4 — structural NNUE work (ambitious):**

10. Port Reckless's **byteboard splat threat-delta generation** (commits
    `42399e79`, `45d9cc5a`, `97be0061`, total ~+10 Elo). This is the
    biggest structural win. Requires adding `RAY_PERMUTATIONS`,
    `RAY_ATTACKERS_MASK`, etc. tables and a `push_threats_on_mutate`
    observer method. 100-200 LoC, mostly mechanical port.

## What we didn't measure

- **Evals per node in search** — need instrumented counters (Phase 2 #5).
  Without these we can't distinguish "Coda calls NNUE more often per
  node" from "Coda calls NNUE equally often but each call is slower".
- **Cache behavior during a real search** — microbench only stresses
  the hot loop. TT / movegen / history-table cache behavior during
  actual search wasn't captured here.
- **Movegen throughput** — Reckless has AVX-512 setwise movegen
  (`82e30943`, `2d90feb9`, etc.). We don't. Likely contributes to
  bucket 3 (per-node overhead) but not measured.
- **Threat-delta generation cost in isolation** — this is different from
  threat-*apply*. We'd want a third microbench for it.

## Git / commits

- Microbench branch: the changes to `src/main.rs`, `src/nnue.rs`, and
  `src/threat_accum.rs` on `feature/threat-inputs`. Patch is small and
  isolated — see the `EvalBench` enum variant, `invalidate_for_bench`,
  and `reset_for_bench` methods.
- Reckless patch lives outside our repo at `~/chess/engines/Reckless/`
  in `src/tools/eval_bench.rs` + UCI dispatcher edit. Tagged
  `Coda-comparison hack` for easy revert.

## Baseline numbers captured

```
$ coda eval-bench --mode fresh       --reps 10000 -n net.nnue
  evalbench mode=fresh       total_evals=80000 elapsed=0.867s evals/sec=92227
$ coda eval-bench --mode refresh     --reps 10000 -n net.nnue
  evalbench mode=refresh     total_evals=80000 elapsed=0.060s evals/sec=1330926
$ coda eval-bench --mode incremental --reps 10000 -n net.nnue
  evalbench mode=incremental total_evals=80000 elapsed=0.021s evals/sec=3759675

$ coda eval-bench --mode fresh       --reps 10000 -n net.nnue --no-avx512
  evalbench mode=fresh       total_evals=80000 elapsed=0.880s evals/sec=90940
$ coda eval-bench --mode refresh     --reps 10000 -n net.nnue --no-avx512
  evalbench mode=refresh     total_evals=80000 elapsed=0.071s evals/sec=1129948
$ coda eval-bench --mode incremental --reps 10000 -n net.nnue --no-avx512
  evalbench mode=incremental total_evals=80000 elapsed=0.028s evals/sec=2854728

$ echo "evalbench fresh 10000" | reckless               # AVX-512+VNNI
  evalbench mode=fresh       total_evals=80000 elapsed=0.055s evals/sec=1458484
$ echo "evalbench incremental 10000" | reckless         # AVX-512+VNNI
  evalbench mode=incremental total_evals=80000 elapsed=0.014s evals/sec=5815343
$ echo "evalbench fresh 10000" | reckless-avx2          # AVX-2 only
  evalbench mode=fresh       total_evals=80000 elapsed=0.064s evals/sec=1258078
$ echo "evalbench incremental 10000" | reckless-avx2    # AVX-2 only
  evalbench mode=incremental total_evals=80000 elapsed=0.014s evals/sec=5710523
```

---

*Investigation by subagent scans + local profiling, 2026-04-23. Next
action: run instrumented counter build to measure evals/node gap
(Recommendation #5). That's the single missing piece needed to complete
the decomposition.*
