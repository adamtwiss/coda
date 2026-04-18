# Threat-signal search improvements — planning doc (2026-04-19)

## Context

Tests #463 (`experiment/full-attacks-history`, +14.2 Elo) and #466
(`experiment/king-zone-nmp`, ~+7.8 Elo) landed using the simple
`attacks_by_color` union bitboard. This thread tiers the remaining
threat-signal ideas and orders the experiments by ROI.

## Data points driving the ordering

From existing `profile-threats` instrumentation:

- **apply_threat_deltas histogram**: mean 10.71 deltas/call, 67% in
  5-12, 30% in 13-24, 2.7% in 25+. Not long-tailed → B3 (raw delta
  count as tactical signal) is too coarse. C2 (weighted accumulator
  delta) has more resolution.
- **Section 2b profile**: 27.4% CPU for 0.57 deltas/call — 48× more
  cycles-per-delta than section 1. This is the biggest "free-lunch"
  NPS target independent of the Elo work. Addressed on this branch
  via slider-iteration rewrite (see `threats: rewrite section 2b`
  commit). Zero-emitter counter added to confirm skew (see
  `profile-threats: add per-section zero-emitter counters` commit).

## Experiment tiering

### Tier A — consensus patterns, untried in Coda

| # | Item | Status | Notes |
|---|------|--------|-------|
| A1a | Stratified escape ladder (Q from rook+, R from minor+, N/B from pawn) | Not started | Reckless/PlentyChess. +10-20 Elo |
| A1b | Onto-threatened penalty, piece-value-gated | SKIP for now | Unstratified version H0'd at #465 (-4 Elo). Retest with SEE gate after A1a+A1c land |
| A1c | `can_win_material` RFP loosener | Not started | Viridithas. Opportunity-side of RFP widening. +3-8 Elo |
| A2  | 2-bit threat-level history stratification | Not started | Compounds on #463. 16× history table size. +5-10 Elo |
| A3  | ProbCut gate on high threat state | Not started | Trivial. +1-3 Elo |

### Tier B — novel, requires x-ray data surfaced to search

| # | Item | Status | Notes |
|---|------|--------|-------|
| B1 | Discovered-attack movepicker bonus | **Helper landed** (`xray_blockers`). Ready for SPRT branch | Coda-unique. +5-15 Elo speculative |
| B2 | Skewer detector movepicker bonus | Not started | Needs Path 2 refactor to be cheap |
| B3 | Raw delta-magnitude as extension signal | **Shelved** | Delta-count distribution too uniform per histogram |

### Tier C — ambitious

| # | Item | Status | Notes |
|---|------|--------|-------|
| C1 | Trained weight-magnitude oracle | Not started | Novel, high variance |
| C2 | NNUE accumulator delta as extension signal | Not started | Finer resolution than B3 — preferred if delta-magnitude idea tried at all |

## X-ray data in NNUE

Verified in `src/threats.rs:409-600`. X-rays are computed in three
distinct code sections inside `push_threats_for_piece`:

- **1b**: slider-I-am-moving x-rays through blocker to next piece
- **2** (Z-finding): sliders pointing at me, x-ray target flips when I
  appear/disappear
- **2b**: sliders whose x-ray target passes through a blocker to reach
  this square

All three emit into the same `RawThreatDelta` format and go through
the same `threat_index(attacker, from, victim, to)` encoding. The NNUE
feature space (66864 features) makes no distinction between "direct"
and "x-ray"; the network learns one feature per (attacker, from,
victim, to) tuple and x-rays activate the same feature as the
corresponding direct attack along that extended ray.

**Implication**: x-ray info is computed but not exposed to search.
Three paths to change that:

### Path 1 — fresh compute at node entry (simplest)

Standalone `Board::xray_blockers(color) -> Bitboard` using magic
bitboards + ray_extension table. Cost: ~60-120ns per node (~3-5% NPS).

**Landed on this branch**: `Board::xray_blockers` committed. Unused
until B1 SPRT branch consumes it.

### Path 2 — delta-tagging at emission (bit-steal)

`RawThreatDelta` current packing:
`[attacker_cp:8] [from_sq:8] [victim_cp:8] [to_sq:7] [add:1] = 32 bits`.

Both `attacker_cp` and `victim_cp` only use 4 of their 8 bits (12
piece types). Steal one bit from `attacker_cp` for `is_xray`:
`[xray:1] [attacker_cp:7] [from_sq:8] [victim_cp:8] [to_sq:7] [add:1]`.

Sections 1b/2-Z/2b set `is_xray=true`. `apply_threat_deltas` masks it
off before `threat_index()` lookup. NNUE output bit-exact (feature
encoding preserved).

**Earned once B1 Path-1 validates x-ray signal is worth >~6 Elo**
(otherwise the ~5% NPS cost of Path 1 nets to zero and the refactor
is unjustified).

### Path 3 — derived bitboards during emission (biggest)

Emit x-ray info directly into `Board::xray_blockers[2]` bitboards at
emission time. Updated incrementally on every delta. Zero cost at
node entry.

**Not scoped**: threats code has been the #1 correctness hot-spot;
every state addition is another fuzzer case. Do not pursue unless
Paths 1 and 2 both land Elo and NPS remains a binding constraint.

## Recommended experiment order

| # | Experiment | Parallelisable? |
|---|-----------|-----------------|
| 1 | A1a + A1c bundle (stratified escape + can_win_material) | — |
| 2 | 2b slider-iteration NPS win (this branch) | Yes, orthogonal |
| 3 | A2 (2-bit history) | — |
| 4 | A3 (ProbCut threat gate) | — |
| 5 | B1 via Path 1 fresh-compute — validates x-ray lever | — |
| 6 | Path 2 refactor + B2 + x-ray history stratification | IF B1 lands |
| 7 | C2 (NNUE accumulator delta extension) | — |
| 8 | C1 (weight-magnitude oracle) | — |

## What's on this branch

Three commits, in order:

1. **profile-threats: add per-section zero-emitter counters** — extends
   existing feature-gated profile infrastructure to track which
   sections ran their scalar walks but emitted no deltas. Confirms
   whether 2b's 27.4%/0.57 ratio is dominated by zero-emitters (the
   2b rewrite's primary target).

2. **threats: rewrite section 2b as slider-iteration** — replaces
   fixed 8-direction scalar ray walks with slider iteration driven
   by the precomputed `between()` table. Semantics-preserving;
   bench must be unchanged. **Requires fuzzer verification on
   Hercules before merge.** Expected +3-4% NPS.

3. **Board: add xray_blockers(color) for B1** — fresh-compute helper
   that returns squares where one of `color`'s pieces is blocking
   its own slider's attack-through to an enemy piece. Unused in
   this commit; consumed by the eventual B1 SPRT branch.

This docs commit captures the plan for Hercules review before any
A-tier SPRT experiment kicks off.
