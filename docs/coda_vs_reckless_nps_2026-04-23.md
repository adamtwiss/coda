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

Clean idle-CPU measurements, OB worker stopped. Coda counter
overhead (~2%) is included in Coda's numbers — they're what real
search pays.

| Engine | ISA | **fresh** (debug scalar) | **refresh** (prod SIMD) | **incremental** | **make+unmake** (threats on) |
|---|---|---:|---:|---:|---:|
| Coda | AVX-512+VNNI | 92,040 | 1,328,863 | 3,687,924 | **22,912,303** |
| Coda | AVX-2 only | 91,530 | 1,112,387 | 2,669,350 | 22,840,541 |
| Reckless | AVX-512+VNNI | 1,478,598 | — | **6,112,511** | 19,623,393 |
| Reckless | AVX-2 only | 1,268,017 | — | 5,882,765 | 21,818,698 |
| Reckless | AVX-512, no observer | — | — | — | 52,312,610 (upper bound) |

Reckless / Coda ratio (>1 means Reckless faster):

| ISA | refresh | incremental | make+unmake |
|---|---:|---:|---:|
| AVX-512+VNNI | **1.11×** | **1.66×** | **0.86× (Coda faster)** |
| AVX-2 only | **1.14×** | **2.20×** | **0.96× (tied)** |

(The `make+unmake` values include threat-delta generation on both sides;
see the make-unmake section below for full decomposition.)

## Key findings (updated 2026-04-23 PM after evals/node + make-unmake)

**The 2× NPS gap decomposes into:**
- 1.86× slower per NNUE eval on Coda (weighted average of rebuild + incremental)
- 1.30× more NNUE evals per node in Coda (0.677 vs 0.520)
- Combined: **2.42× worse eval cost per node**, which exceeds the
  observed 2× NPS gap (Coda's faster base make+unmake + movegen
  partially offsets).

**What is NOT the gap**:
- Base make+unmake (Coda is slightly slower, 12%)
- Threat-delta generation (Coda is actually 30% faster here than
  Reckless's NNUE observer, surprisingly)
- L1 cache miss rate on the *full-rebuild* path (both tight)

**Where the real gap lives (ranked):**

1. **Incremental eval L1 dcache miss rate**: Coda 15.72% vs Reckless 0.55%.
   Memory layout problem, not SIMD. See §3.
2. **Full-rebuild rate**: Coda 45.65% vs Reckless ~22% per node.
   Coda's `materialize` doesn't walk back multiple ancestors for
   incremental replay — it forces a rebuild when direct parent is
   unmaterialised.
3. **Evals per node**: Coda 0.677 vs Reckless 0.520. Missing
   eval-only TT writeback + fewer pruning cut-offs = more eval calls.
4. **Per-instruction work**: Coda does 1.86× more instructions per
   incremental eval (at equal IPC). Branching overhead and data
   traversal excess, not cycles-wasted.

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

## Running on other hosts

Script: `scripts/nps_microbench.sh` (on `feature/threat-inputs`).

```bash
# 1. Check out Coda
git clone git@github.com:adamtwiss/coda.git ~/code/coda
cd ~/code/coda && git checkout feature/threat-inputs
make net                   # fetches the v9 production net

# 2. Check out Reckless
git clone https://github.com/codedeliveryservice/Reckless.git ~/chess/engines/Reckless
cd ~/chess/engines/Reckless

# 3. Apply the Coda-comparison evalbench patch to Reckless (one command):
git apply ~/code/coda/scripts/reckless_evalbench.patch
# Tagged 'Coda-comparison hack'; revert with `git checkout -- .` when
# the investigation is done.

# 4. Stop any OB worker on the host for clean numbers
~/code/OpenBench/ob-worker.sh stop   # if one is running

# 5. Run the script
cd ~/code/coda
./scripts/nps_microbench.sh

# 6. Ship back nps_microbench_<host>_<date>.csv
```

The script handles AVX-512 / AVX-2 / ARM dispatch automatically based on
`/proc/cpuinfo`. On ARM hosts it runs native-only (no AVX-2 forcing).

**CPU diversity we want to cover**: Zen 5 (zeus ✓), Zen 4, Zen 2/3,
Intel Alder/Raptor Lake (P-core), Intel Ice/Sapphire Lake (server
AVX-512), Apple Silicon M1/M2/M3, any ARM server (Graviton / Ampere).
Each adds a data point for whether the 2× gap is uArch-specific or
structural.

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

## Update 2026-04-23 PM: evals/node + make-unmake decomposition

Extended the microbench and ran both engines' production search with
eval-path counters. New data reshapes several assumptions from earlier
in the doc.

### Evals per node

Added `NNUEAccumulator::stats_full_rebuilds` / `stats_incremental_updates`
/ `stats_cached_skips` counters on Coda and a `stats_tt_static_eval_hits`
counter on `SearchInfo`, printed after `coda bench 13`. Reckless
instrumented via their existing `dbg_hit` infrastructure at
`evaluate` (slots 0-6 for PST/threat × accurate/refresh/incremental).

Depth 13 bench on the 8 Coda positions (single thread, Hash=16 MB):

| Metric | Coda | Reckless |
|---|---:|---:|
| Total nodes | 2,258,905 | 3,425,249 (their own 50-position set, depth 12) |
| NNUE full rebuilds | 697,761 (45.65% of evals) | 694,562 (19.5% of PST-perspective decisions) |
| NNUE incremental | 830,656 (54.35%) | 2,849,960 (79.9%) |
| TT static-eval hit (avoids NNUE) | 108,442 (6.62% of eval call-sites) | not measured |
| **Evals / node** | **0.677** | **0.520** |

**Coda calls NNUE 30% more often per node** (0.677 vs 0.520). Candidate
causes:
- Missing TT eval-only writeback on NNUE-hit + TT-miss (noted earlier in
  this doc). Reckless seeds the TT with eval; Coda doesn't.
- Different pruning landscape. With a wider pruning net (Reckless has
  razoring, direct-check LMP exemption, wider SEE, no RFP depth cap),
  Reckless's tree is shallower at the leaves → fewer eval calls per
  node.

### Rebuild rate

Coda's rebuild rate is 45.65%. Reckless's is 19.5% (PST) and 3.3%
(threat), per perspective. Not directly comparable (Coda rebuilds both
perspectives together; Reckless separates them), but the upper bound
is "at least one perspective needs refresh": Coda 45.65%, Reckless
~22% → **Coda rebuilds ~2× more often per node than Reckless**.

Candidate causes:
- Coda's `materialize` only checks the *direct parent* (`top - 1`) for
  `computed`. Reckless's `can_update_pst` walks back through multiple
  ancestors and replays from the nearest accurate one. When Coda's
  lazy-accumulator pattern leaves an intermediate parent unmaterialised,
  we pay a full rebuild that Reckless would avoid.
- **Note: multi-ply PSQ walk-back was tried 2026-04-17 (SPRT #424) and
  H0'd at −17.86 Elo.** Replay cost at h=768 exceeded Finny-diff-refresh.
  What did land: per-pov `AccEntry.computed` (a9f6e1f). So the
  rebuild-rate gap vs Reckless is mostly a counting artefact (our
  counter sees "at least one perspective rebuilds" = 45.65%; per-pov
  it's ~22% each side, similar to Reckless's 19.5% PST). The headline
  stat oversold the recoverable delta here.

### Make + unmake with threat-delta generation ACTIVE

Clean results on idle Zen 5 CPU (OB worker stopped, Coda counter
overhead ~2%):

| Mode | Coda | Reckless |
|---|---:|---:|
| make-unmake (threats OFF / no observer) | 46.7M/s | **52.3M/s** (NullBoardObserver) |
| make-unmake (threats ON / NNUE observer) | **22.9M/s** | 19.6M/s |
| Threat-delta marginal cost | **22.3 ns/move** | 31.9 ns/move |
| Base make+unmake cost | 21.4 ns/move | 19.1 ns/move |

**Surprise finding**: with threat deltas active on both sides, **Coda's
make+unmake is 17% FASTER than Reckless's** (22.9M vs 19.6M). The
Reckless "byteboard splat" advantage is real *compared to a naive
scalar impl* (5-6 Elo in their commits), but Coda's current
implementation is already competitive. Base make+unmake is ~12%
slower in Coda, but threat-delta generation is actually *faster* in
Coda — probably because the threat-delta pipeline is smaller-surface
(no x-ray propagation machinery).

→ **Porting the byteboard splat is NOT a priority.** The threat-delta
path is not where the time is going.

### Where the 2× NPS gap actually lives

Decomposition with clean numbers:

**Per-eval cost** (from microbench):
- Coda incremental: 3.69M/s (weighted-average rate)
- Reckless incremental: 6.11M/s
- Coda refresh: 1.33M/s
- Reckless refresh: 1.48M/s
- Effective weighted (at each engine's own full/incremental split):
  - Coda: 1 / (0.4565/1.33M + 0.5435/3.69M) = 1 / (343ns + 147ns) = **2.04M/s**
  - Reckless: 1 / (0.195/1.48M + 0.805/6.11M) = 1 / (132ns + 132ns) = **3.79M/s**
  - Per-eval ratio: **1.86× slower in Coda**

**Evals per node**: 0.677 / 0.520 = **1.30× more calls per node**

**Combined eval impact**: 1.86 × 1.30 = **2.42× worse on eval alone**

Observed NPS gap: ~2.0×. So **per-eval cost + evals-per-node together
more than account for the observed gap**; the difference (~0.4×) is
Coda's faster non-eval overhead (base make+unmake is slightly faster,
movegen likely too) partly offsetting.

**This recasts the priority list:**

1. **Investigate the 28× L1 cache miss rate on incremental eval**
   (15.72% in Coda vs 0.55% in Reckless). Every fix here is multiplied
   by 0.677 evals/node — high leverage.
2. **Reduce evals/node via eval-only TT writeback** (see §3).
   Moving from 45.65% → ~20% rebuilds would close roughly half the
   per-eval-cost gap, since rebuilds are 2.8× slower than incrementals.
3. **Eval-only TT writeback** to directly reduce evals/node.
4. **LMP direct-check exemption** (+9.46 LTC Elo, Reckless #818) — also
   reduces evals/node by reducing node count.

Pushing the byteboard splat to the bottom of the priority stack.

## Cross-reference with 2026-04-18 findings

`docs/v9_nps_findings_2026-04-18.md` covered earlier NPS work and
measured several things that are still valid here. Re-checking what
has/hasn't changed, and what 2026-04-18 got right/wrong:

### Still-valid prior findings

- **Walk-back refresh distance distribution** (2026-04-18):
  47.8% of refreshes have accurate ancestor within 2 plies
  (d1=18.6%, d2=29.2%). Estimated +2-3% NPS. Today's measurement
  shows the full-rebuild rate is 45.65% — in the same ballpark as
  2026-04-18's 40.7%, so the walk-back opportunity is unchanged.
- **apply_threat_deltas histogram** (2026-04-18): avg 10.71 deltas/call,
  uniform distribution. Cap/batch optimisation dead.
- **push_threats section breakdown** (2026-04-18):
  - Section 2b (sliders x-ray TO this square): **27.4% of push_threats,
    2.5% of engine CPU, 0.57 deltas per call at 211 cycles/call.**
    Single biggest hotspot inside the threat pipeline.
  - Section 1 (direct threats FROM piece): 6.4% of push_threats.
- **Const-generic caller doesn't propagate through SIMD helpers**:
  full end-to-end const refactor needed to see gain (2-3 days). Shelved.
- **PGO regresses -10% on v9**: kept shelved. The 2026-04-18 fix
  (fixing threat-gen inliner confusion) wasn't attempted; same
  symptom remains.

### Changed / contradicted

- 2026-04-18 claimed "Coda's NNUE alone consumes ~3.4 μs/node,
  Reckless's entire per-node time is ~3.3 μs". Today's per-call
  microbench numbers don't support that framing — at 548K NPS = 1.82
  μs/node, 61% NNUE = 1.1 μs/node NNUE, not 3.4 μs. The 2026-04-18
  figure looks like a unit mix-up. **The 2× NPS gap persists; it's
  not quite right that "Coda's NNUE alone equals Reckless's entire
  per-node time".**
- 2026-04-18 concluded "remaining gap to Reckless is 15-20% NPS,
  recoverable via 3 small wins". Today's measurement shows the gap
  is still ~2× on Zen 5 (AVX-512+VNNI), with per-eval cost 1.86×
  and evals/node 1.30× — combined 2.42× eval-side cost. **The 2×
  gap is the reality even after VNNI landed; 15-20% understated it.**
- 2026-04-18 said "the last tune delivered +38.6 Elo in a day. Three
  NPS commits in a day delivered +3.8% NPS ≈ +4 Elo." — the
  Elo-economics framing there was right **at that time**. Post-tune
  the easy tuning Elo is harder to find, and the 2× NPS gap remaining
  is structural — at some point NPS IS the dominant Elo lever.

### Walk-back refresh: previous attempt confounded — worth retry

2026-04-18 proposed PSQ walk-back as the top lever (+2-3% NPS). It was
attempted 2026-04-17 (commit 21f02ff) and SPRT #424 H0'd at **−17.86
Elo**. Reading the commit message carefully, the regression was a mix
of walk-back effects **and** unrelated structural changes:

- `AccEntry.computed: bool` → `[bool; 2]` (per-pov) — this DID land.
- Added `DirtyPiece.needs_refresh: [bool; 2]` — 3 extra bytes per
  stack entry.
- Added entry-guard branches in `materialize`'s fast path.
- Added a `#[cold]` outlined slow path doing per-pov walkback +
  chain replay.

Critical note from the commit: **"Common-case fast-path is
byte-identical to old code, so the overhead is elsewhere — struct
size (+3 bytes AccEntry), extra branches in materialize's entry
guard, or code-layout effects."** Bench regressed 4% before SPRT even
started. So the -17.86 Elo reflects *structural overhead on the hot
path* plus whatever walk-back itself contributed — we never
disentangled the two.

The replacement (a9f6e1f) kept per-pov but dropped walk-back, citing
"materialize microbench showed lazy-chain case was 3.5× slower".
That's a real measurement, BUT:

- **Reckless uses exactly this pattern successfully** — their
  `update_pst_accumulator` does `for i in accurate..self.index { ... }`
  to replay from the nearest accurate ancestor.
- Reckless's per-feature weight footprint is **much smaller** than
  Coda's (the 49 MB threat matrix — see sparsity §§). Their replay is
  cache-hot; ours likely spills to DRAM per replayed delta.
- So "walk-back is 3.5× slower" may be specific to Coda's current
  memory layout, not inherent to the approach.

**Retry candidate**: walk-back PSQ is worth re-attempting if one of
the following holds:

1. **Minimal-overhead re-implementation**: keep the current hot-path
   byte-for-byte, add walk-back as a separate entry point called only
   when `!stack[top-1].computed`. No extra AccEntry bytes, no extra
   fast-path branches. If bench stays flat, the 2026-04-17 bench
   regression was the struct/layout overhead, not walk-back itself.
2. **Post-sparsity retry**: once the threat weight matrix shrinks
   enough to fit in L3 (via aggressive L1 reg or width reduction —
   see §§ sparsity), the replay is no longer DRAM-bound. Walk-back's
   cost drops below Finny-refresh's, matching Reckless's regime.

Both are reasonable follow-ups. Option 1 is cheap (half-day) and
isolates the question of "is walk-back itself slow or was the
refactor packaging slow". If option 1 stays flat in bench, run SPRT
carefully (tight bounds, not #424's structural muddle). If option 1
*also* regresses on bench, then the 3.5×-slower-replay claim is
validated and we wait for option 2.

### Rebuild-rate interpretation (narrower)

Coda's "full rebuild" counter reports 45.65%. Per-pov, that's
approximately 22% per side — similar to Reckless's 19.5% PST rebuild.
**The headline "Coda rebuilds 2× more often" in my earlier doc was a
counting artefact** (Coda's counter fires when either perspective
rebuilds; Reckless's is per-perspective). So the recoverable delta
from reducing rebuild rate is smaller than the headline suggested;
the question is whether walk-back is faster than Finny-refresh for
Coda's memory layout — which is what option 1 above would test.

### Structural wins that DID land since 2026-04-18

Not NPS, but worth noting so the NPS lever-list isn't re-proposed as
Elo levers:

- **Reckless-style king-bucket layout (kb10)** — shipped; materially
  affected net file, not per-eval speed.
- **Factoriser in Bullet training** — shipped; Elo gain from better
  net quality, not from faster inference.

### Cross-reference: sparsity investigation (2026-04-19)

`docs/v9_sparsity_investigation_2026-04-19.md` is directly relevant to
today's L1-miss-rate finding. Summary of its key measurements:

- **v9 L1d miss rate (on Hercules via `perf stat`): 26.0%**, vs v5's
  16.5%. v9 is memory-bound, not compute-bound. Matches my Zen-5
  measurement of 15.72% L1 miss on the incremental eval microbench.
- **Threat weight matrix: 66,864 × 768 bytes = 49 MB.** On machines
  with 16-32 MB L3, that matrix spills into DRAM on the cold tail
  of accesses.
- **Activation distribution is heavy-tailed**: top 706 features do
  50% of activations; 74% of features contribute <1% of work.
- **8.4% of weight rows are structural zeros** (impossible
  piece-interaction combinations). Load-time compact captures this
  as a strict win (no Elo risk, bench-identical).
- **L1 regularisation at 1e-6/1e-7 did NOT add more sparsity** —
  cumulative soft-threshold over a training run is negligible vs
  typical weight magnitudes. Need L1 at 1e-4 to 1e-3 to shrink
  beyond the structural floor, with Elo risk from killing marginal
  features.
- **Load-time compact is implemented** on branch
  `experiment/l1-inference-compact`. Bench-neutral. Ready to merge;
  waiting on a sparser net to justify.

**This explains the 15.72% L1 miss rate finding directly.** The gap
to Reckless's 0.55% isn't magical cache-friendliness in their inference
code — it's the **weight matrix being the wrong size**:
- Reckless: ~800 features × ~1 KB row × 50% density ≈ 400 KB
  (fits entirely in L2)
- Coda v9: 49 MB (spills out of L3 on most machines)

The threat weight matrix is the single largest NNUE allocation, and
it's accessed with a heavy-tailed pattern that exercises the cold
parts. Reckless's NNUE footprint is ~100× smaller; every probe stays
in L2.

### Implication for NPS priorities (major revision)

This recasts priorities substantially. The cache-miss finding is NOT
fixable on the inference side alone — the matrix is simply too big.
Real NPS wins from memory footprint require **training-side changes**:

**Training-side levers** (bigger expected NPS impact):
1. **Aggressive L1 regularisation (λ = 1e-4 to 1e-3) retrain** —
   aim to drive 40-60% of feature rows to zero. At 15-20 MB matrix
   we cross the L3 boundary on most machines → **step-function NPS
   gain of 10-30% on memory-constrained hosts** (smaller on beefy
   ones like Zen 5). Requires one training cycle (~4h GPU). Elo
   risk from killed marginal features; tune λ conservatively.
2. **Accumulator width 768 → 640** — 17% memory reduction brute-force.
   Full retrain, uncertain Elo cost, combines with (1).
3. **Piece-interaction feature redesign** — eliminate structurally
   impossible features upfront in the feature space. ~20-30%
   reduction, bigger refactor.

**Inference-side levers** (smaller NPS impact, lower risk):
4. **Ship the load-time compact** (`experiment/l1-inference-compact`)
   — ~4 MB on the current net, bench-neutral. Essentially free.
   Doesn't need waiting for sparser nets.
5. **Eval-only TT writeback** (Hercules's idea) — reduces evals/node,
   not per-eval cost. Orthogonal to the cache issue.

Dropping from the old priority list:
- **Byteboard splat** — Coda's make+unmake is already faster than
  Reckless's with threats active. Not the bottleneck.

### Inference-side cache-friendliness — NOT just a training-side problem

Revising my earlier "can't fix L1 misses on the inference side"
claim. Even when the threat matrix fits in L3 (big-cache hosts like
Threadripper, M3 Pro/Max, server EPYC), **how** the data is accessed
still determines cache hit rate. Reckless's 0.55% miss rate reflects
both (a) smaller footprint AND (b) deliberate layout choices.

Concrete inference-side levers we haven't tried:

**(a) Flatten `AccEntry` to inline arrays** — biggest structural
cache-friendliness lever on the accumulator stack.

Current Coda layout (`src/nnue.rs:3729`):
```rust
pub struct AccEntry {
    pub white: Vec<i16>,                 // heap block
    pub black: Vec<i16>,                 // heap block
    pub threat_white: Vec<i16>,          // heap block
    pub threat_black: Vec<i16>,          // heap block
    pub threat_deltas: Vec<RawThreatDelta>,     // heap block
    pub threat_features_white: Vec<usize>,      // heap block
    pub threat_features_black: Vec<usize>,      // heap block
    ...
}
```
That's **7 heap allocations per AccEntry × 256 entries ≈ 1,800
scattered heap blocks**. When `materialize` walks the stack looking
for an ancestor (or `recompute_threats_if_needed` replays forward),
each pointer dereference is potentially a different cache line.

Reckless's layout (`Reckless/src/nnue/accumulator/psq.rs`,
`threats.rs`):
```rust
pub struct PstEntry {
    pub values: [[i16; L1_SIZE]; 2],  // inline, no heap
    ...
}
pub struct PstStack {
    stack: [PstEntry; MAX_PLY],  // inline, no heap
}
```
Entire stack is one contiguous ~2 MB buffer. HW prefetcher can
predict the access pattern. No pointer-chasing.

**Expected impact**: visible on the incremental-eval L1 miss rate
(15.72%). Half-day refactor, zero retrain, zero algorithm change.
No Elo risk beyond microbench-verifiable correctness. This is
arguably the single biggest "pure inference cache hygiene" lever
on the table, and it was missed in the 2026-04-18 investigation.

**(b) Hot-feature frontloading** — reorder the 49 MB threat weight
matrix so the top 706 features (540 KB, 50% of activations) sit at
the front of the allocation.

2026-04-19 measured the activation distribution: heavy-tailed, with
the top 1% of features doing 50% of the work. Current layout indexes
features by "natural" (piece_pair, square) order — no correlation
with activation frequency. Reordering so hot rows come first means:

- Top 706 rows = 540 KB = fits in L2 on every modern CPU
- Next ~7k rows (90% coverage) = ~5.5 MB = L3 on all, L2 on big caches
- Cold tail stays wherever, rarely touched

Implementation: rank features by activation count (already
instrumented via `measure_feature_sparsity` test), permute the weight
matrix at load time, adjust the feature-index mapping in
`enumerate_threats` / threat-delta code. Zero retrain, ~100 LoC, +
an SPRT to confirm bench unchanged.

This is "free" memory-footprint reduction *of the hot working set*
even when total matrix size stays 49 MB. On big-cache machines
it's plausibly bigger than the raw-compact option because it
targets the hot path specifically. On small-cache machines it
compounds with the compact-at-load work.

**(c) Prefetch next threat-delta's weight row** inside the
apply loop. 2026-04-19 listed this as tier 3 ("smaller win given
L3 is the real frontier"). But with 10 deltas per move × 768-byte
rows = 12 cache lines each, the 120-cache-line scattered pattern is
exactly the case where prefetch helps: kick off the next load before
the current one finishes. Half-day.

**(d) Cache-line alignment on weight rows**: 768 bytes / 64 = 12
cache lines. If weights start on a 64-byte boundary, each row
touches exactly 12 lines. If misaligned, some rows touch 13. Current
allocator may or may not produce 64-byte-aligned starts. Easy check;
easy fix via `#[repr(C, align(64))]` wrappers or explicit alloc.

### Revised lever ranking (final)

All three docs merged. Items 1, 4, 5 specifically target cache hygiene
independent of total footprint — they keep paying off on machines
where 49 MB fits.

| # | Lever | Effort | Where the win lives |
|---|---|---|---|
| 1 | Flatten `AccEntry` → inline arrays | Half day | L1/L2 cache behaviour (all hosts) |
| 2 | Training-side memory shrink (L1 reg / width) | Training cycle | Small-cache L3 fit |
| 3 | Ship load-time compact merge (`experiment/l1-inference-compact`) | Low | 4 MB free today |
| 4 | Hot-feature frontloading | Half day | Hot working set L2 residency |
| 5 | Prefetch in `apply_threat_deltas` | Half day | Per-eval latency (all hosts) |
| 6 | Clean-retry PSQ walk-back (minimal hot-path overhead) | Half day | Rebuild cost |
| 7 | Eval-only TT writeback (Hercules) | Low | Evals/node |
| 8 | LMP direct-check exemption (Hercules, separate) | Low | Tree size, +9.46 LTC Elo |

Items 1, 4, 5 are pure inference-side cache hygiene — the category
I had initially dismissed too quickly. They don't need training
cycles or waiting for sparser nets. Items 7-8 are Hercules territory
(pruning), but reduce evals per node so they land NPS wins too.
Item 2 is the big step-function for small-cache hosts but needs a
training cycle.

Items 1, 4, 5 are **pure inference-side cache hygiene** — the
category I had dismissed too quickly. They don't need training
cycles, they don't need waiting for sparser nets. They specifically
address the "how data is accessed" dimension, which stays relevant
even on machines where total footprint isn't the bottleneck.

### Revised takeaway

- Walk-back PSQ: **previous attempt confounded**; clean re-implementation
  (minimal overhead, isolate walk-back from struct/layout churn) is
  worth half a day. Post-sparsity retry also makes sense.
- Per-pov `computed` tracking: **already done** (a9f6e1f).
- Rebuild-rate gap: smaller than headline once counted per-pov.
- **Current ranked NPS levers** (combining all three docs):
  1. **Training-side memory shrink**: aggressive L1 reg (1e-4 to 1e-3),
     or width reduction 768→640, to cross the 32 MB L3 boundary.
     **Step-function NPS gain on cache-constrained hosts.**
  2. **Load-time compact merge** (`experiment/l1-inference-compact` →
     trunk): ~4 MB free today, more with sparser nets later.
  3. **Clean-re-try PSQ walk-back** with no hot-path overhead —
     isolate whether the 3.5×-slower claim survives when you remove
     the struct/layout confound. Half day.
  4. **Eval-only TT writeback** (Hercules): reduce evals/node.
  5. **LMP direct-check** (Hercules, separate): +9.46 LTC Elo,
     shrinks tree → fewer eval calls.

#1 has the biggest potential NPS step but needs a training run. #2-5
are all incremental wins and independently testable.

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
