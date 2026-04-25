# Coda Improvement Portfolio — 2026-04-24

**Written:** 2026-04-24 (v9 merged into main earlier today; retune #747
landed at +7.2; full 45-engine RR settling around CCRL 3520-3620).

**Purpose:** A standalone catalogue of the many +5 Elo opportunities
distributed across diversified improvement threads. Companion to
CLAUDE.md's §Improvement Portfolio section — CLAUDE.md captures the
philosophy (why diversification), this doc captures the list (what
specifically to try). Refresh when threads exhaust or new candidates
surface.

## Why a portfolio mindset

Correlated work plateaus together. Every "clever search idea" thread
of work eventually hits diminishing returns on a flat eval; every
"training recipe tweak" thread eventually bottoms out on fixed search
behaviour. Running multiple orthogonal threads means when one stalls
we've already got 2-3 others producing Elo.

At our current CCRL ~3570 and ~4 weeks old, the cheap Elo is gone but
there are *many* +2-5 Elo items. 12-15 of them would stack toward the
~60-100 Elo jump that gets us into top-20 CCRL. None individually
transformational; all cheap to try.

## The Six Threads

### 1. Eval-Search Flywheel (compounding loop)

`eval → ordering → pruning → depth → eval` — each link reinforces the
next. A +2% eval gain delivers +3-5 Elo raw, but +8-15 after
retune-on-branch captures downstream recalibration.

**Near-term candidates:**
- **Post-shelved-net retune**: once Reckless-KB + factor SB800 trains,
  run 25K-iter full-sweep SPSA to capture the flywheel gain. Biggest
  single leverage point we have queued.
- **`experiment/force-more-pruning` #750 resolution**: tune running
  now, aggressive starting point forcing SPSA to find a new basin.
  Potential +5-10 Elo if the aggressive basin wins.
- **IIR_MIN_DEPTH=2 standalone SPRT** (tune #743 strongly preferred
  this; confirm on trunk as a pure ablation before trusting).
- **FH_BLEND_DEPTH → 0 ablation**: SPSA #743 dialled this toward off.
  Feature-utility diagnostic suggests removing may simplify without
  cost.

### 2. Correctness Audits (bugs in rarely-fired paths)

Historically our highest Elo-per-hour lane. Bugs in 50-move rule, LMR
endgame gate, SMP races, TB handling, and `is_pseudo_legal` EP hole
have each delivered +3-30 Elo.

**R5 top-5 (status updated 2026-04-25):** 3 of 5 merged in the 2026-04-22
batch, 1 in flight, 1 closed-on-re-analysis. See
`research_threads_2026-04-24.md` §Top-5 for per-item resolution log.

- ✅ SEE pawn-promotion recapture — merged `b25366d` (#652 +1.8 Elo H1)
- ✅ Evasion capture-promotion ordering — fixed via C8 audit #25/#26
- 🔄 should_stop 4096-node granularity — SPRT #757 in flight
- ⏭ Forced-move soft_floor — verified intended (10ms saves stockpile)
- ✅ Repetition scan plies_from_null — merged `402e366`

**Next 10 — status updated 2026-04-25 sprint:** 6 actioned, 4 skipped
with rationale. See `research_threads_2026-04-24.md` §Next-10 for the
per-item resolution log. Headline: SE singular_beta clamp, recapture
ext ply>0 guard, FH-blend skip in SE, threats sq=63 bounds, LMR
do_shallower 10cp probe — all in flight as SPRTs #757-#762.

**Remaining audit backlog after this sprint:** mostly defensive /
fix-when-touched / aarch64-only items per
`correctness_audit_2026-04-22.md` §SPECULATIVE. No further high-leverage
queueable SPRTs in this category for now. Future audits should explore
new ground rather than re-walking the resolved 2026-04-22 → 2026-04-25
list.

Each ~1 day to audit + fix historically, but the 2026-04-25 empty-fleet
sprint demonstrated 30-60min per item is more typical when the fix is
well-isolated and the audit doc has the location pinned.

### 3. Comparative Engine Review with Instrumentation

Reading source tells you *what*; instrumentation tells you *when and
how often*. Half-day of instrumented study often beats a full day of
source-reading.

**Near-term:**
- **Reckless NMP R-value distribution**: patch Reckless's search to
  log R per (depth, eval-beta-diff) over a bench run. Compare to
  ours. Our R is probably miscalibrated somewhere.
- **LMR reduction dumps by history bucket**: both engines, same bench,
  diff the per-bucket reduction amounts.
- **Pruning fire rates by depth**: RFP / LMP / Futility / BNFP — what
  % of positions at d=5, d=10, d=15 get pruned by each in Reckless
  vs Coda? Already know BNFP fires far more in Reckless; quantify
  the others.
- **TT hit rate breakdowns by depth / age**: SF vs Coda. Worst-case
  gap informs TT replacement policy tuning.
- **Move-ordering first-cut rate per ply**: quick patch to output at
  bench end. Top engines >85%; we're probably at 75-80%.

Entry point: `scripts/reckless_evalbench.patch` for how we've done
these patches before.

### 4. Training Hyperparameters + Data

Biggest historical wins were 1-line schedule changes (low-final-LR
+47, data filter ply≥16 +22, +80 with retune). Still-unexplored:

- **Output bucket count**: we use 8. Untested on v9. 4, 16, 32 each
  plausible candidates.
- **Batch size**: 16384 is Bullet default. At our current training
  scale this may not be optimal.
- **WDL schedule shape**: currently constant 0.15 for warm30. Linear
  ramp 0.0 → 0.15 or late-ramp 0.0 → 0.3 untested.
- **Training data composition**: take Coda's Lichess blunder positions
  (visible when watching), add to training corpus as supplementary
  data.
- **Warm-up duration**: warm30 beat warm20 on kb10 family. Warm40,
  warm50 untested on current config.

### 5. Long Tunes / Long Training (patience-bounded "free" Elo)

- **25K-iter SPSA pass**: ~200K games, 20-40 fleet hours. Worth once
  every 5-10 merges for calibration lock-in. Top engines run this
  routinely.
- **SB1600+ training**: v9 sparse threat features continue converging
  deep into low-LR tail (+88 Elo SB400 → SB800). Another +10-20 plausible
  at SB1600, tapering. ~80h GPU.

**Priority order:**
1. Shelved-net SB800 (Reckless-KB + factor, +30 already proven short-TC)
2. Post-shelved-net retune (flywheel capture, +5-15)
3. SB1600 on shelved-net recipe (+10-20 more)
4. 25K-iter SPSA lock-in after above stabilises

### 6. Infrastructure — Lichess-Visible, SPRT-Invisible

These don't show up in OB SPRT (OB uses own book, no TB) but matter
for real deployment rating.

#### 6a. Opening Book

Current state: Titans.bin, randomly downloaded from internet.
Problems observed on Lichess:
- Book exit eval varies wildly: **−0.7 to +0.5** pawns. Two compounding
  issues:
  1. **Strength mismatch**: book compiled from GM games (~2600 Elo);
     at our 3500+ search depth, moves that "won statistically" at
     GM level may be concretely inferior when probed deeper.
  2. **Theory staleness**: much of the GM-game corpus predates modern
     engine-driven opening theory. Lines that were considered standard
     in 1980s-2000s Ruy Lopez / Sicilian Najdorf / King's Indian etc.
     have been revised by modern analysis; older books embed the
     revised-to-be-inferior lines with high weights.

**Candidate experiments:**
- **Book A/B on Lichess**: Titans.bin vs alternatives (Zurichess
  small-sharp book, chess-book.info engine-tuned books, PolyGlot's
  standard CCRL book). Play ~100 games of each, rate.
- **Book-move sanity check**: after book returns move M, quickly
  search to depth 10. If our eval of M is >30cp worse than our eval
  of our own best move, exit book here. Prevents the -0.7 cliffs.
- **Shorter book depth**: cap book at move 8-10 instead of letting
  it run to 15+. Our search is stronger than GM-statistics in the
  middlegame transition — trust our eval earlier.
- **Book pre-filtering**: one-time offline pass where Coda searches
  every book position at depth 15, removes positions where book's
  top choice scores >50cp below our best alternative.
- **Book-less Lichess A/B**: test no-book vs Titans.bin as a
  reference. Surprising result possible — our search is strong
  enough that the book may be net-negative.

#### 6b. Syzygy TB Handoff

- **DTZ walkback** (just added 2026-04-24, commit `9b5e364`). Should
  improve Lichess TB endgame display + ponder cache quality.
- **TB entry timing**: currently fires moment popcount drops into
  range. Is DTZ-optimal always practically optimal at our strength?
  Maybe sometimes a more-active king / better pawn structure plan
  beats entering TB immediately.
- **Cursed win / blessed loss handling**: SPECULATIVE items in
  correctness_audit_2026-04-22.md; need review for edge cases.

#### 6c. Time Management

- **Forced-move soft_floor zero** (listed in Correctness §2 — TM bug).
- **Stockpile edge cases**: ponderhit in simple positions, instant-
  emit detection, dynamic soft floor at increment.
- **Lichess-specific TM tuning**: Lichess bot latency, move overhead
  calibration per opponent's server location.
- Hard to SPRT; primary signal is Lichess-watch.

#### 6d. Parallel Search (Lazy SMP)

- **v9 T=4 blunder bug**: 14× more blunders at T=4 vs T=1 on v9 (v5
  unaffected). Coda deploys at T=1 on Lichess. Fixing this
  effectively unlocks more strength without code improvements.
- Root cause unknown; Atlas investigating 2026-04-20.

## Thread-Selection Heuristic

When picking the next experiment:

1. **What thread?** Name it explicitly before starting. Avoids
   reactive picking of whatever's top-of-mind.
2. **How recent was the last win in this thread?** If 4+ weeks,
   prioritise it for diversity reasons.
3. **What's the bounds class?** [0, 3] default for novel features;
   [-3, 3] for correctness fixes; [0, 10] for structural ports;
   Lichess-watch + manual for infrastructure.
4. **Is there an instrumentation variant first?** For comparative
   work especially, 1-hour measurement often redirects the effort
   from a wrong-direction implementation to a right-direction one.

## Per-Thread Status Quick Reference

| Thread | Active? | Last delivery | Next in queue |
|---|---|---|---|
| Flywheel (retune) | Yes (#750 running) | #747 +7.2 | Post-shelved-net retune |
| Correctness | Stale (~1 week) | — | R5 top-5 (SEE pawn-promo easiest) |
| Comparative | Stale (~2 weeks) | — | Reckless NMP distribution |
| Training | Waiting on GPUs | Group-lasso SB200 probe | Shelved-net SB800 |
| Long runs | None yet | — | 25K SPSA lock-in |
| Infrastructure | DTZ walkback landed today | Commit 9b5e364 | Book A/B OR v9 T=4 bug |

## Elo Budget Estimate (if all land over ~2-3 months)

| Thread | Plausible gain |
|---|---:|
| Shelved-net SB800 + retune | +35-45 |
| Correctness backlog (top-5 + next-10) | +20-40 |
| Comparative review → 2-3 retune-on-branch wins | +10-20 |
| Training hyperparameters | +10-20 |
| Long tunes / SB1600 | +15-30 |
| Book + infrastructure | +10-20 (Lichess-only) |
| **Stacked** | **+100-175** |

That puts us at CCRL ~3670-3745 territory if everything lands and
compounds. Realistic-case is more like +50-80 Elo captured (many
items won't hit target, and some will conflict). Even the realistic
case is top-15 CCRL.

## Refresh cadence

Rewrite this doc every ~4-6 weeks or after any 3+ queued items land.
Stale portfolio catalogs steer experimentation toward things that
are already done. Memory notes are more durable than this file; this
is a working list.
