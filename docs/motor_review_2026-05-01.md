# Motor Engine Review (2026-05-01)

Source: `~/chess/engines/motor/` (C++ header-only, ~1.6K LOC of search). CCRL Blitz top-20.

Motor is a small, opinionated engine — most of its search lives in
`search/search.hpp` (459 lines). It maps closely to current Coda but with
a handful of distinctive choices. Nothing here is dramatic; this is a
"refinement / SPSA opportunity" review rather than a "missing feature"
one. Items already known H0 in `experiments.md` are dropped or noted as
such.

## Tier 1 — high confidence / high leverage

### T1.1  PV-extra TT-cutoff depth gate (`tt_entry.depth >= depth + 2 * is_pv`)
**File:** `search/search.hpp:100`.
At PV nodes, Motor only takes the TT cutoff when the stored entry is **2
plies deeper** than the current depth (instead of the usual `>= depth`
gate). For non-PV the gate is the standard `>= depth`. This is the same
pattern Stockfish/Reckless/Obsidian use; we surveyed it for cross-engine
comparison but Coda still uses the symmetric `tt_depth >= depth` at every
node (`src/search.rs:1990`).

Note: this is **not** the same as the catastrophic "TT score refines
static eval" experiment from 2026-03-12 (-77 Elo); that replaced static
eval with a TT minimax score. This change only affects when the TT
*cutoff* fires, and is monotonic (it makes us search more, not less).

**Mechanism:** PV nodes already skip TT cutoffs in many engines — Motor
does cut on PV but only when the entry is overwhelmingly authoritative.

**Expected:** +1 to +3 Elo. Pure pruning-discipline change. Effort: **small**.

### T1.2  PV "would-have-pruned" depth shortening (`if would_tt_prune && is_pv: depth--`)
**File:** `search/search.hpp:109-114`.
When PV gates the TT cutoff out (per T1.1), Motor still extracts a
benefit: it decreases `depth` by 1. This is the lighter, structurally
sound version of the failed "PV+IIR depth--" experiment
(`experiments.md` 2026-03-12 IIR-on-PV at d>=2, H0 -5.5; and the depth-2
IIR-extra test, H0 -26.4). The failed versions reduced *unconditionally*
on PV; Motor's only fires when the TT entry is good enough that the
cutoff would have been valid in non-PV.

**Mechanism:** If the TT says "this is settled," PV verification doesn't
need full depth — search 1 ply shallower. Self-corrects on errors via the
re-search.

**Expected:** +1 to +2 Elo. Effort: **small**. Bench will move
noticeably; expect a retune-on-branch follow-up.

### T1.3  TT-blended `eval` for pruning (motor lines 117-120)
**File:** `search/search.hpp:117-120` (and qs `quiescence_search.hpp:48-51`).
After computing corrected `static_eval`, Motor *replaces* `eval` with the
TT score **only** when the bound direction agrees with our movement
(`tt.bound==LOWER and eval > tt_eval`, or `tt.bound==UPPER and eval < tt_eval`).
This is structurally different from the 2026-03-12 H0 experiment, which
adjusted toward TT in the wrong direction (max for LOWER, min for UPPER)
— that was reversing the bound semantics. Motor's check **only blocks
the override**; if directions disagree the TT value is used. Coda
currently never blends.

This is the same "TT eval refinement" pattern as Stockfish/Reckless and
matches what the 2026-03-12 note suggested ("Stockfish does a milder
version, only adjusting improving").

**Mechanism:** `static_eval` for pruning becomes whichever of (corrected
NNUE, TT score) is *less* informative-against-direction. Conservative
floor/ceiling instead of replacement.

**Expected:** +2 to +5 Elo at most engines; uncertain for Coda given the
prior catastrophic test, but the prior was a different mechanism. Worth
a careful retry — start with bounds [-3, 3] so we cap downside.

Effort: **small**.

### T1.4  Threat-bucketed correction history (5th source)
**File:** `search/tables/history_table.hpp:153-154,178-181,200`.
Motor has 6 corrhist sources: pawn, NP×2, minor, major, **threat-set**,
continuation. The threat key is
`murmur_hash_3(threats & side_occupancy[stm])` — the set of our own
pieces under threat, mixed by murmur3 to spread bits.

Coda has 5 sources (pawn, NP×2, minor, major, continuation). We don't
have a threat-keyed corrhist. This is the *Plenty-style* corrhist that
`docs/cross_engine_comparison_2026-04-25.md` flagged as N-3 (+2 to +4
Elo, "highest expected of corrhist-only items"), but Plenty puts threats
in the *continuation* corrhist key; Motor's treatment is independent —
its own table — and stacks alongside continuation rather than replacing
it. Strictly cheaper to integrate than the Plenty variant: it's a
straight new corrhist table, no movepicker plumbing.

**Mechanism:** When the same set of our pieces is under threat across
visits, eval errors will correlate; corrhist learns the bias.

**Expected:** +2 to +4 Elo. Effort: **medium** (one new 16K-entry table,
murmur3 helper, weight in the corrhist sum). Coda already has the threat
bitboard from `Board::threats`.

## Tier 2 — smaller / less certain

### T2.1  History-bonus depth bump on `best_score > beta + 80`
**File:** `search/search.hpp:348`.
On a fail-high, Motor calls `history->update(... depth + (best_score > beta + 80))`
— effectively giving an extra +1 to the bonus depth when the cutoff was
"clean" (well above beta). Coda passes raw `depth` to `history_bonus()`
at `src/search.rs:3143`. Several SF-family engines (Stockfish, Berserk,
Obsidian) have a similar "clean cutoff" amplifier but the threshold
varies (50-100cp).

**Mechanism:** Reinforce moves that produce decisive cutoffs more than
moves that scrape past beta.

**Expected:** +1 to +2 Elo. Effort: **small**. Pair it with an SPSA on
the threshold (Coda would expose `HIST_CLEAN_CUTOFF_MARGIN`).

### T2.2  Aspiration: depth-decrease on fail-high (Motor's structural variant)
**File:** `search/search.hpp:394-396`.
Motor decreases `search_depth` (not the outer `depth`) by 1 on
aspiration fail-high but keeps it monotonic via `max(1, ...)`. Three
prior Coda attempts H0'd catastrophically (-353, -389, -18 Elo —
`experiments.md` 2026-03-10, V5 ASP Fail-High Depth Reduce). Reading
those notes: the failures were because Coda mutated the outer ID `depth`
rather than an inner `search_depth` variable. Motor has a **separate
inner variable** that resets to `depth` on every fail-low.

This is **not** a recommendation to retry — it's a marker that the prior
H0 was likely an implementation issue (mutating shared depth across
aspiration iterations). If anyone retries, the structural fix is to mirror
Motor's inner `search_depth` and keep `depth` as the ID anchor.

**Expected:** +1 to +3 Elo if the prior bug was implementation-only.
Effort: **small**. Risk: still high given the prior catastrophic
results; gate behind [-3, 3].

### T2.3  Triple extension (`ext = 2 + (is_quiet && s_score + 62 < s_beta)`)
**File:** `search/search.hpp:268`.
Motor's third extension fires only when the quiet TT move is *very*
singular (s_score + 62 < s_beta). This is precisely the structural
pattern Coda's #804/#806 SPSA is currently exploring (PV/quiet/corr
modulators). **Already in flight — not a new recommendation.** Listed
here only because it's worth confirming the current SPSA's defaults
include a quiet bonus comparable to Motor's (-62 vs Coda's
DEXT_MARGIN_QUIET tunable).

Effort: **N/A** (in flight as #806).

### T2.4  Capture-history SEE threshold magnitude (`-cap_score / 40` vs Coda's `-capt_hist / 18`)
**Files:** Motor `search/move_ordering/move_ordering.hpp:35`; Coda
`src/movepicker.rs:562`.
Both engines use cap-history to widen the SEE threshold for good
captures, but the divisors differ by 2.2×. Motor: `-cap_score/40` (more
forgiving SEE for high-history captures). Coda: `-capt_hist/18` (much
more aggressive — large positive history more aggressively re-classifies
losing captures as good).

This is a clear SPSA candidate: expose `CAPT_HIST_SEE_DIVISOR` (currently
hardcoded 18 in `movepicker.rs:562`), tune around 18-50 with c_end ~5.
The fact that Reckless, Berserk, and Obsidian use 32-50 in this divisor
(and our 18 is an outlier) suggests we may be over-classifying losing
captures as good.

**Expected:** +1 to +3 Elo if the SPSA finds a basin away from 18.
Effort: **small** (expose tunable, run focused 1-param SPSA at
~1500 iters).

### T2.5  QS-in-check evasion early break on low score
**File:** `search/quiescence_search.hpp:84`.
Motor's QS-in-check loop **breaks** (not continues) once it sees an
evasion with move-score < 16'000 (i.e. once the killer/good-capture
prefix is done). After the first non-good evasion, the rest are skipped
entirely. Coda searches all evasions in QS-in-check
(`src/search.rs:3511-3547`).

The 16'000 threshold matches the killer-bonus value (`32'000 * (killer
== move)` is killers, captures range to ~10M, anything below 16K is a
plain quiet evasion).

**Mechanism:** Trust QS to find the critical defence among captures and
killers; quiet evasions in qs are mostly noise. Risks missing a quiet
king walk that defends, but the cost-benefit at QS depth tends to favour
the cut.

**Expected:** +0.5 to +2 Elo (NPS-driven; very few QS-in-check positions
need the deep evasion search). Effort: **small**.

### T2.6  Probcut early-out: TT-conflict shortcut
**File:** `search/search.hpp:168`.
`if (depth >= 5 && !(tt_move && tt_entry.depth > depth - 3 && tt_entry.score < probcut_beta))` —
Motor *skips ProbCut entirely* when the TT already says "we won't beat
probcut_beta at near-current depth." Coda has a similar gate via
`probcut_tt_noshot` at `src/search.rs:2433` — confirm whether the gate
includes the `tt_entry.score < probcut_beta` half (skim suggests yes).

If both halves are present, this is a no-op match. If only one is, the
fix is trivial.

**Expected:** 0 to +1 Elo. Effort: **small** (just verify alignment).

## Tier 3 — novel / experimental / lower confidence

### T3.1  Aspiration window growth `window += window/3`
**File:** `search/search.hpp:402`.
Motor uses 4/3 multiplicative growth (`window += window/3`). Coda's
default is also a similar multiplicative grow but with different
constants. Worth a look during the next aspiration tuning pass —
probably not its own SPRT.

Effort: **small**. **Expected: ~0 Elo unless paired with retune.**

### T3.2  TT cluster size 3 (vs Coda's 5)
**File:** `search/tables/transposition_table.hpp:26`.
Motor uses 3-slot clusters; Coda 5-slot. Most top engines (SF, Berserk,
Reckless) use 3. Bigger clusters reduce capacity (more padding-per-bucket)
but improve hit rate via more replacement candidates. Not worth chasing
in isolation; flagged for future TT redesign.

Effort: **medium** (TT layout change + retune). **Expected: 0 to ±2 Elo.**

### T3.3  TT replacement weight `depth + 4 * age` (depth-and-age combined)
**File:** `search/tables/transposition_table.hpp:64,72`.
Motor's relevance score combines depth and age into a single linear
expression. Coda's TT replacement uses a generation-bucketed scheme
(`src/tt.rs`). The 2026-03-12 "TT Age Weight 6" experiment H0'd at
-70 Elo, but that was *evicting more aggressively*. Motor's coefficient
is 4 — same as our prior (allegedly safe) baseline. Not a recommendation;
a sanity-check that our TT replacement is in the same ballpark.

Effort: **small** (verify, no change). **Expected: 0 Elo.**

### T3.4  Move-ordering capture score blend (`10M * see + mvv[victim] + capt_hist`)
**File:** `search/move_ordering/move_ordering.hpp:35-36`.
Motor's good-capture score has SEE as a giant multiplier (10M) that
dominates everything, then mvv (1000-3000) + capt_hist as the
tie-breaker. Coda: `mvv_lva(board, m) + capt_hist` (no SEE in score, SEE
only used for partition). Coda's MVV bucket is also smaller magnitude
(MVV*16). The two-tier "SEE first, then MVV+history" ordering is
qualitatively different from ours.

Coda has tested various capture-ordering shapes (see
`docs/capture_ordering_crossengine_2026-04-20.md` which catalogues 17
engines). The 10M-SEE-multiplier shape is what makes Motor's ordering a
hard 2-tier separation; ours achieves the same partition via the bad
captures bucket. Likely no Elo difference, but worth confirming our
analysis already covers this.

Effort: **N/A — already covered by capture_ordering_crossengine doc.**

## Items deliberately not flagged (already H0 / already done)

- **Razoring at depth < 3, margin 500*depth** — Motor uses
  `eval + 500*depth <= alpha`. Coda's `400 + 100*depth` was tested in
  both directions multiple times (`experiments.md` 2026-03-09, 2026-03-10,
  pattern #10) and is locked. No retry.
- **Continuation corrhist (ply-2 + ply-3, two tables)** — Coda already
  has ply-2 continuation corrhist. Motor's adds a *second* table at
  ply-3 instead of ply-4 (which Arasan/Caissa/Tarnished use). Cross-engine
  norm is ply-2 + ply-4; ply-3 is unusual. Skipping unless someone wants
  to test the variant.
- **Threat NNUE inputs** — Motor doesn't have these. Coda v9 already does.
- **NMP `cut_node` gate** — already merged in Coda (`src/search.rs:2326`).
- **Pawn-endgame NMP guard** — `pawn_endgame()` (`board.hpp:264`) is
  identical in mechanism to Coda's `stm_non_pawn != 0` (`search.rs:2318`).
- **Killer move (single-slot)** — Motor uses 1 killer per ply, dead-coded
  in Coda's movepicker per the CLAUDE.md ("history handles ordering").
  Same posture, no change needed.
- **SE depth gate (Motor `se_depth=6`, Coda `SE_DEPTH=4`)** — Coda's
  lower SE_DEPTH was the result of SPSA tune #743 ("strong signal" per
  `search.rs:108`). Don't revert.

## Summary table

| Item | Tier | File:Line | Expected Elo | Effort |
|------|------|-----------|--------------|--------|
| T1.1 PV `+ 2*is_pv` TT cutoff gate | 1 | search.hpp:100 | +1 to +3 | small |
| T1.2 PV would-tt-prune depth-- | 1 | search.hpp:109-114 | +1 to +2 | small |
| T1.3 TT eval blend (direction-only override block) | 1 | search.hpp:117-120 | +2 to +5 (uncertain) | small |
| T1.4 Threat-bucketed corrhist | 1 | history_table.hpp:153,200 | +2 to +4 | medium |
| T2.1 +1 history-bonus depth on clean cutoff | 2 | search.hpp:348 | +1 to +2 | small |
| T2.2 Asp inner search_depth-- | 2 | search.hpp:394-396 | +1 to +3 (high risk) | small |
| T2.3 Triple-ext quiet bonus | 2 | search.hpp:268 | (in flight #806) | n/a |
| T2.4 capt_hist SEE divisor SPSA | 2 | move_ordering.hpp:35 | +1 to +3 | small |
| T2.5 QS-in-check break on low score | 2 | quiescence_search.hpp:84 | +0.5 to +2 | small |
| T2.6 ProbCut TT-conflict early-out | 2 | search.hpp:168 | 0 to +1 | small |
| T3.1 Aspiration window 4/3 grow | 3 | search.hpp:402 | ~0 | small |
| T3.2 TT cluster 3-slot | 3 | tt.hpp:26 | 0 to ±2 | medium |
| T3.3 TT relevance `d + 4*age` | 3 | tt.hpp:64,72 | 0 (sanity) | small |
| T3.4 Capture ordering 10M-SEE multiplier | 3 | move_ordering.hpp:35-36 | n/a | n/a |

## Recommended next experiments

If anyone is choosing what to try next from this review, the highest-EV
sequence is:

1. **T1.1 + T1.2 together** as one branch — they're a coherent pair
   (PV TT cutoff discipline + would-have-pruned compensation). One SPRT,
   one retune-on-branch. Both small. Cumulative ~+2 to +5 Elo.
2. **T1.4** (threat corrhist) — pure additive, intersects current threat
   agenda, low risk of regression. ~+2 to +4 Elo.
3. **T2.4** (capt_hist SEE divisor SPSA) — focused 1-param tune,
   ~1500 iters. Cheap to run.
4. **T1.3** (TT eval blend) — risky given the 2026-03-12 H0 history but
   structurally different mechanism. Bound at [-3, 3]. ~+2 to +5 if it
   transfers.

Skip everything in Tier 3 unless explicitly needed.
