# Starzix Review — Survey for Coda (2026-05-01)

Source: `~/chess/engines/Starzix/src/` (~6 KLOC, single-author Rust-style C++).
Compared against Coda v9 reference points (`src/search.rs`, `src/movepicker.rs`,
`src/nnue.rs`, `experiments.md`, `docs/cross_engine_comparison_2026-04-25.md`).

Starzix is small but disciplined: ~45 SPSA-tunable parameters, no killer
backup, no counter-move, no LMP non-PV gating — relies on history quality and
clean composability. Where Starzix differs from the 17-engine consensus, it
tends to be **simplifying** (no LMP-PV gate, no aging in TT) rather than
adding novelty.

---

## Tier 1 — high-confidence, high-leverage

### 1. Multiplicative history bonus on PVS fail-high cascades

`src/search.hpp:690, 738, 749, 756, 812` + `src/thread_data.hpp:227`.

Starzix counts how many fail-highs a single move triggered through the
PVS chain (LMR → ZWS → full-window) and **multiplies** the history bonus
by that count when the move ultimately raises beta:

```cpp
i32 score = 0, numFailHighs = 0;
// LMR re-search
numFailHighs += score > alpha;
// Full ZWS re-search
numFailHighs += score > alpha;
// PV full-window re-search
numFailHighs += score >= beta;
...
histBonus *= numFailHighs;   // applied at line 227
```

A move that "kept earning extensions" on the way to a cutoff (max 3) gets
2-3× its bonus; a move that only fail-highs at the LMR level gets the
normal 1×. This is **distinct from the cutoff_count subtree feedback**
Coda already H0'd at #883 (that was *child* cutoff count gating LMR;
this is *self* fail-high count scaling history bonus). Not in
`experiments.md`.

Effort: **small** — three counter increments + one multiply at the
`update_histories` site. Magnitude: **uncertain, +2 to +6** (it's a
per-move history-magnitude scalar; Coda's history-magnitude work has
been one of our biggest historical gainers). Worth retune-on-branch.

### 2. Razoring with `depth² × margin` (no depth gate)

`src/search.hpp:526-529`, `src/search_params.hpp:91-93`.

```cpp
if (alpha - eval > razoringBase() + depth * depth * razoringDepthMul())
    return qSearch(...);
// 365 + d² * 276
```

No depth cap. At d=2 margin = 1469cp; at d=3 = 2849cp. So in practice
it self-limits via the d² growth, but the *form* differs from Coda's
removed razoring (linear in depth, gated d≤2). Razoring on v9 was
re-added at #389 (RAZOR_BASE=900, RAZOR_MULT=756 linear-in-depth,
trending +1.1 / 5K games — indeterminate). The Starzix form is not
what was re-tested.

Effort: **small** — one-line change to existing razoring. Magnitude:
**+1 to +4** (re-cast of an existing-but-marginal feature with the
d² shape; v9 RMS scale needs RAZOR_BASE ~900-1000 and RAZOR_MULT
~250-300 to match). SPRT [-3, 3].

### 3. Slider attacks x-ray through the STM king for `enemy_attacks`

`src/position.hpp:675-685` (`enemyAttacksNoStmKing`). Starzix builds the
"enemy attacks" bitboard with `occ ^ squareBb(stmKing)`, so sliders
treat the STM king as transparent. Squares behind the king (where the
king might *flee to*) are correctly marked as attacked.

Coda uses `attacks_by_color(them)` with full occupancy
(`board.rs:423-460`), so squares behind our king are unattacked even
when a slider would pin/skewer the king if it moved. This affects
**4D threat-aware history keying** and **threat-aware LMR** — both
read `enemy_attacks` to decide if from/to is threatened.

Coda's threat-aware history is its single biggest move-ordering
investment. Sharpening the underlying threats bitboard with the
king-x-ray correction is a free precision upgrade.

Effort: **small** — one boolean parameter on `attacks_by_color` (or
a new `attacks_excluding_king` variant), thread it through the two
existing call sites in search/qsearch. NPS hit ~0% (one extra
`occupied ^ king_bb`). Magnitude: **+1 to +4** (small per-position
delta, but multiplies into every history read/write).

### 4. Promotion axis on noisy/capture history

`src/history_entry.hpp:31, 79-94`. Starzix's noisy history is keyed
`[stm][piece][to][captured][promotion]` (capture target × promotion
target as separate axes). Coda's capture history is
`[piece 1-12][to][captured 0-6]` (`movepicker.rs:32`) — no
promotion axis, no stm.

Promotions land at a specific square with a specific piece-type
result — currently all four promotion variants share one capture-
history slot per `[piece, to, captured]`. Q-promotes vs N-promotes
are tactically very different.

Effort: **small** — extend `capture` table to
`[piece][to][captured][promo_type]` (5 slots: none/N/B/R/Q).
Memory: 13×64×7 = 5824 i16s → 5824×5 = 29 KB. Magnitude:
**+1 to +3**. STM axis is separately covered in
`cross_engine_comparison_2026-04-25.md` Tier-1 #3 (queued).

---

## Tier 2 — smaller / less certain

### 5. `oppWorsening` signal for RFP

`src/search.hpp:517-523`.

```cpp
const bool oppWorsening
    = !td->pliesData[ply - 1].inCheck && -eval < td->pliesData[ply - 1].correctedEval;
// RFP margin: max(depth - oppWorsening, 1) * rfpDepthMul()
```

This is a one-ply mirror version of "improving" — did the opponent's
move *worsen* their position? If yes, RFP can fire 1 depth easier
(steeper margin). Coda uses `improving` (ply vs ply-2, `search.rs:2405`)
for the same RFP slot, but does not have the symmetric one-ply signal.

Effort: **small** — one extra static-eval comparison from `ply-1`.
Coda already tracks `static_evals[ply]`. Magnitude: **uncertain,
0 to +3** — incremental info beyond `improving` may be small.
SPRT [-3, 3]. Note `oppWorsening` was not surfaced in our 17-engine
survey as a frequent feature; if this shows up only in Starzix and
no Tier-1 engine, weight accordingly.

### 6. RFP "halve toward beta" return value

`src/search.hpp:524`. Starzix returns `(eval + beta) / 2` from RFP
rather than `eval` or `beta`. Coda returns `eval` (let me grep:
`search.rs` ~2410 returns the static eval).

This is the "fail-high blending" pattern — the conservative return
splits credit. Coda's TT FH-blend uses a depth-weighted version
post-search; RFP-time halving is a different application.

Effort: **small** — one expression change. Magnitude: **0 to +2**.
Likely interacts with existing FH-blend; SPRT carefully.

### 7. NMP reduction `R = 4 + depth/3` (no eval term)

`src/search.hpp:541`. Starzix uses `nmpDepth = depth - 4 - depth/3`
(R = 4 + d/3). Coda uses `R = NMP_BASE_R(7) + depth/NMP_DEPTH_DIV(5) +
(eval - beta) / NMP_EVAL_DIV(108)` (`search.rs:2331+`).

Coda's R-base 7 is tied with the most aggressive in our survey;
Starzix's effective R at depth 12 is `4 + 4 = 8`, comparable.
The notable difference is **no eval-term clamp**: Coda's eval-margin
adds up to `NMP_EVAL_MAX` extra plies, sometimes substantial.

Already heavily SPSA'd on Coda. Listed for completeness; **not a
recommendation** — Coda's NMP is well-tuned.

### 8. Reduce more on **expected cut node** (PVS-style)

`src/search.hpp:602`:

```cpp
reducedDepth -= (nodeType == NodeType::Cut) * 2;
```

Starzix carries a node-type enum (PV / Cut / All) and reduces an
extra 2 plies at expected cut nodes. Coda uses `cut_node: bool` and
applies it to NMP, SE negative extension, and IIR — but does not
add an LMR baseline reduction for cut nodes. Reckless's
`cut_node: r += 1` is in the survey; this is the Starzix-specific
`+2` magnitude.

Effort: **small** — single `if cut_node: reduction += 2` at LMR site.
Magnitude: **+1 to +3** if not already covered by Coda's existing
cut_node uses. Pre-check that Coda doesn't already have this
(grep `cut_node` in LMR block); this *might* be a duplicate of
something we tested.

### 9. Soft-time gate at depth ≥ 6 only

`src/search.hpp:389`:

```cpp
if (msElapsed >= (td->rootDepth >= 6 ? softMsScaled() : mSearchConfig.softMs))
    break;
```

Starzix only applies the scaled soft-time (best-move stability ×
score-stability × node-fraction) at root depth ≥ 6, using flat
`softMs` for shallower iterations. Below depth 6 the stability
counters are too noisy to trust.

Coda's TM features were validated at LTC (40+0.4) and are invisible
at STC; the depth-gate may matter at STC. Effort: **small**.
Magnitude: **uncertain at STC, +1 to +3 at LTC**. Worth noting only
because it's a one-line discipline change.

### 10. Shallow razor base (`razoringBase = 365`)

If we don't take Tier-1 #2 (the d² form), the linear-Coda v9 razoring
trial (#389, RAZOR_BASE=900) might still benefit from a tighter base.
Starzix's 365cp base on a SCALE=400 net is much tighter relative to
Coda's #389 (900cp on Coda v9 scale). Effort: tweak existing tunable.
Magnitude: **0 to +2**. Lowest priority; only if #389 lands H0.

---

## Tier 3 — novel / experimental

### 11. NNUE input bucketing on **enemy queen position**

`src/nnue.hpp:20-38`. Starzix uses 6 input buckets, but the bucket key
is **enemy queen's position quadrant**:

```cpp
constexpr size_t NUM_INPUT_BUCKETS = 6;
// Bucket 0: enemy has 0 queens
// Bucket NUM_INPUT_BUCKETS-1 (=5): enemy has multiple queens
// Buckets 1-4: 4-quadrant scheme over enemy queen's square
//   (LQ, LK, RQ, RK quadrants — see INPUT_BUCKETS_MAP)
```

Coda v9 uses 16 king buckets (horizontal mirror + king-quadrant). The
Starzix scheme captures a different bias: positions **with vs without
enemy queen**, and **where her queen lives**, are evaluated by
disjoint feature sets. The Finny table size is comparable to Coda's
16-bucket setup.

This is a network-architecture change: training-side cost, not search.
Mentioned only because it's a *novel* bucketing axis (no other engine
in our 17-engine survey uses queen-position bucketing). Effort:
**large** (requires retrain). Magnitude: **uncertain** — could be
+0 / could be substantial; orthogonal to king-bucketing. File for
the "next net architecture experiment" backlog, not a search PR.

### 12. Fail-low collection at PV nodes drives malus

`src/search.hpp:578-579, 778-781, 815`. Starzix collects fail-low
quiets AND fail-low noisies (with their captured piece) at every
node, then on a fail-high penalises *all* of them via `updateHistories`
— including the captures that fail-lowed.

Coda penalises fail-low quiets on a quiet beta-cutoff. The capture
side (penalising fail-low captures on a beta cutoff regardless of
which move was best) was tested ~partially in
`experiments.md:3971` ("Unconditional capture malus: penalize all
tried captures on any beta cutoff"). Re-check whether the Starzix
form (collected `failLowNoisies` with captured-piece axis) is
materially different from what we tested.

Effort: **small-medium**. Magnitude: **uncertain**. Verify against
`experiments.md` line 3971 before pursuing.

### 13. No killer / counter / non-PV LMP gate

Starzix's MovePicker has TT → noisy → killer (single) → quiet → bad
noisy. **No counter-move heuristic, no killer-2.** And LMP is applied
unconditionally (no `nodeType != NodeType::PV` gate, see
`search.hpp:608`).

Coda has counter-moves dead-coded (per `movepicker.rs:3`) and uses
killer slots that the comments say are dead-coded "between
GoodCaptures and GenerateQuiets". Effectively similar.

Note for confidence: this is the consensus direction (delete dead
ordering slots, lean on history). No action item — observation only,
confirms Coda's recent direction.

---

## Things Starzix has that Coda already has (skip)

- **Cuckoo upcoming-repetition**: `cuckoo.hpp` matches Coda's `cuckoo.rs`.
- **Halfmove eval scaling** `eval * (200-hmc) / 200`: Coda has
  `apply_halfmove_scale` (`search.rs:746`). H0'd twice on Coda
  (#730, #755).
- **2-ply cont-history**: Coda has continuation history at plies 1, 2, 4, 6.
- **Node-fraction TM, score-stability TM, best-move-stability TM**:
  all present in Coda.
- **Negative SE extensions** (-3 on tt_score ≥ beta, -2 on cut node):
  already in Coda (`search.rs:2710-2714`).
- **doDeeper / doShallower** off LMR re-search delta: Coda has
  similar, ditched the previous attempt #1235 H0.
- **Probcut with ttBound depth-4 gate**: equivalent.
- **2-ply cont-corr-hist**: Coda H0'd this at #101 ("ply-2 context
  too noisy for correction signal").

---

## Summary triage

Recommended next-steps in order:

1. **Tier-1 #1** (numFailHighs hist-bonus multiplier) — small, novel,
   plausible +2 to +6. Cheap to try.
2. **Tier-1 #3** (king-x-ray for enemy_attacks) — small, sharpens
   the existing 4D-history infrastructure that already pays large
   dividends.
3. **Tier-1 #4** (promotion axis on capture history) — small, isolated,
   queue alongside the open STM-on-main-history Tier-1 from
   `cross_engine_comparison_2026-04-25.md`.
4. **Tier-1 #2** (d² razoring) — only if v9 #389 lands H1 or H0
   ambiguous; cheap variant.

Tier-2 items are SPSA tweaks or one-line modifications worth
folding into a future retune-on-branch cycle, not standalone SPRTs.

Tier-3 #11 (queen-bucketed NNUE inputs) is a *training* experiment
worth filing in the net-architecture backlog; not a search PR.
