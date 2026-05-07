# Perf investigation 2026-05-07 — instructions/node, not cache

## Headline finding

**Coda's NPS gap to Reckless on fleet-class hardware is dominated by
instructions per node, not memory or cache.** Earlier hypotheses (49 MB
threat-matrix L3 spill, AVX-512 path quality, memory bandwidth) are not
the bottleneck. The lever is reducing the instruction budget per node.

## Anchor data — Hercules (Intel Xeon E-2288G, AVX-2, 3.7 GHz)

`perf stat ./engine bench`, single-core, latest main on both:

| Metric              | Coda      | Reckless  | Coda/Reckless |
|---------------------|----------:|----------:|--------------:|
| NPS                 |   664,272 | 1,037,889 |        0.64×  |
| Cycles/node         |     7,288 |     4,800 |       1.52×   |
| **Insns/node**      | **12,910**|  **7,675**|     **1.68×** |
| IPC                 |      1.77 |      1.60 |       1.11×   |
| Cache-refs/node     |       882 |       734 |       1.20×   |
| Cache-misses/node   |      14.2 |      17.4 |       0.82×   |
| Cache miss rate     |     1.61% |     2.37% |       0.68×   |

**Key observations:**
- Coda's IPC is *higher* than Reckless. Pipeline / SIMD execution is fine.
- Coda's cache miss rate is *lower* than Reckless. Memory subsystem is fine.
- Coda's cache-misses per node is *lower* than Reckless. Bandwidth is fine.
- Coda's instructions per node is **1.68× Reckless's** — entire gap.

The 1.52× cycles/node gap = 1.68× instructions / 1.11× IPC. Math closes
exactly — no residual unexplained by instruction count.

## What invalidates earlier hypotheses

| Hypothesis | Status | Evidence |
|---|---|---|
| Threat matrix L3 spill | ❌ killed | Cache miss rate *better* on Coda |
| AVX-512 path quality | ❌ deprioritised | IPC *higher* on Coda; no SIMD gap |
| Memory bandwidth saturation | ❌ killed | Cache-misses/node *lower* on Coda |
| L1 regularization → matrix shrink | ❌ small | Cache isn't bottleneck; helps but not magic |
| HW prefetch already covers refresh | ✅ confirmed | Earlier in-loop prefetch experiment was bench-neutral |

## What the 5,235 extra instructions per node fund

Coda has features Reckless doesn't, each costing per-node instructions:
- **X-ray threat features** (slider through-blocker enumeration)
- **Threat-aware 4D history** indexing
- **Larger move-ordering bonus stack**: offense bonus, mobility delta,
  knight-fork, battery, qsee bonus, escape bonus, discovered attack
- **More pruning checks**: ProbCut, history pruning, hindsight reduction,
  multi-cluster carve-outs
- **Threat enumeration on every NNUE eval**

These bought ~+110 Elo (v9 architecture) and various other Elo wins
individually. The trade was real and measured. The question now is
whether any of this work is wasted vs. actually-Elo-paying.

## Why Atlas is misleading for this investigation

Atlas (Zen 1 EPYC 7351P, 8 MB L3 per chiplet, 2.4 GHz):
- Narrow chiplet L3 makes memory the binding constraint
- Lower clock makes per-cycle work look more efficient than it is
- Coda's heavier per-node work fits in the memory-bound budget the same
  as Reckless's lighter work — masking the instruction-count gap
- On Atlas single-core: Coda 339K NPS vs Reckless 262K NPS (Coda faster!)

Hercules (Coffee Lake-class, 16 MB L3, 3.7 GHz):
- Larger L3 + faster clock = memory budget no longer binding
- Reckless's lighter per-node work pipelines freely → 3.93× scaling vs Atlas
- Coda's heavier work hits an instruction-count ceiling → only 2.0× scaling
- Hercules is the right reference for fleet-deployment perf

**Atlas Claude shouldn't drive instruction-count reduction work** — local
NPS reads will be dominated by memory effects, not the actual lever.

## Candidate hot paths for instruction reduction

Ranked by expected leverage, given the perf attribution from earlier
search-bench profiling on Atlas (treat as directional, not exact for
Hercules — the numbers will shift but the hot functions match):

### Tier 1 — high call frequency, self-contained

1. **`generate_and_score_quiets` move-ordering scoring loop**
   (`src/movepicker.rs:585`). Per quiet move scored:
   - History reads (main 4D, cont-hist × 4, pawn-hist)
   - Atomic loads of CONT_HIST_MULT, MOBILITY_DELTA_WEIGHT, ESCAPE_BONUS_*,
     QUIET_CHECK_BONUS, DISCOVERED_ATTACK_BONUS, QSEE_BONUS,
     KNIGHT_FORK_BONUS, BATTERY_BONUS
   - 5+ conditional bonus computations, each doing piece_type checks +
     bitboard arithmetic + table lookups
   - Many bonuses fail their safety filter — wasted work
   - Order matters: cheapest checks first, hoist invariants out of loop
   - Note: prior `feature/movepicker-hoist` (#937) tried storage-layout
     refactor and H0'd at -2.2 Elo. Distinct from this lever (work
     reduction within scoring, not picker storage).

2. **Search-side per-move pruning checks** (`src/search.rs` negamax move
   loop). Many independent gates: RFP, futility, LMR, LMP, SEE, ProbCut,
   history pruning, bad-noisy, hindsight. Some share preconditions
   (e.g. depth, node-type, history). Folding shared computation could
   eliminate redundant work. Risk: easy to break Elo with reordering.

3. **`is_pseudo_legal`** (`src/movepicker.rs:991`). Called for every
   TT/killer/counter validation. Large function with many sequential
   checks. Hot for TT-move validation per node.

### Tier 2 — moderate call frequency, more invasive

4. **`push_threats_for_piece`** (`src/threats.rs:924`). Per delta
   generation: 5 sections (direct, x-ray, sliders, sliders-2b,
   non-sliders) with magic lookups. Could potentially eliminate redundant
   checks across sections.

5. **`apply_threat_deltas`** (`src/threats.rs:1609`). The threat-index
   computation per delta is per-call; the SIMD apply work is already
   well-tuned. The per-delta `threat_index()` overhead is the lever.

6. **Move generation** (`generate_quiets`, `generate_captures`). High call
   frequency. Some piece-type iteration overhead.

### Tier 3 — measure first, may not move the needle

7. **`MovePicker::next_slow`** state-machine arms. Already optimised in
   the recent fast-path commit. Probably small remaining headroom.

8. **`pick_best`** selection sort. Already attempted unchecked indexing
   on Atlas — bench-neutral. Worth re-measuring on Hercules where
   instruction count is the binding constraint, not branch prediction.

## Suggested investigation flow on Hercules

1. **Confirm the headline.** `perf stat -e cycles,instructions,cache-misses,
   cache-references,branch-misses ./coda bench` and same for Reckless.
   Verify the 1.68× instruction-count gap holds.

2. **Localise the instruction surplus.** `perf record -e cycles ./coda bench`
   then `perf report --no-children`. Compare top functions to Reckless's
   profile. Hot functions with no Reckless analogue are pure-add-on
   instructions (e.g. x-ray related). Hot functions with a Reckless
   analogue but more % share = candidates for instruction reduction.

3. **Annotate candidate functions.** `perf annotate <function>` to find the
   specific instruction sequences eating cycles. Look for:
   - Branch-heavy loops where condition reordering helps
   - Atomic loads inside loops that could be hoisted
   - Redundant work across nested checks
   - Magic table lookups that could be cached

4. **Per-experiment loop:**
   - Make code change that preserves bench (4815017 nodes)
   - Verify on Atlas first that bench is unchanged (nodes-equal)
   - Hercules `perf stat` to confirm instructions/node dropped
   - Hercules NPS comparison vs main
   - SPRT [0, 3] for any cumulative bundle once instruction reduction is
     credible (~3-5% NPS or more)

5. **Things to avoid re-trying** (already tested/dead):
   - Threat-weight prefetching (HW prefetcher already covers it)
   - PSQ walkback port (~3.5% upper bound even in Reckless)
   - MovePicker storage hoisting (#937 H0'd at -2.2 Elo)
   - `inline(always)` on widely-called functions (i-cache bloat regressed)

## Working-tree state at handoff

- `main` at `a383310` (post merge of #955 movepicker fast-path bundle)
- `feature/movepicker-fastpath` branch already merged, can be deleted
- No uncommitted changes
- `/tmp/reckless_walkback` is a clone of Reckless with vanilla source —
  useful as a diagnostic playground for further Reckless ablations.
  Build with `make` (proper `RUSTFLAGS=-Ctarget-cpu=native` via
  `.cargo/config.toml`).

## What's NOT in scope for this investigation

- Training-side levers (L1 regularization, hidden_size shrink) — real
  but separate workstream, requires GPU runs not code changes.
- AVX-512 path optimisation for Zeus — relevant for Zeus specifically
  but secondary to instruction-count work which applies fleet-wide.
- Walkback (PSQ) — upper-bound 3.5% even in Reckless's measured ablation,
  not worth a third attempt given existing Finny table coverage.
