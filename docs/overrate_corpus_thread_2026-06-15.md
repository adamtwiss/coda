# Overrate corpus — mining real games for search/eval blindspots

**Thread goal (Adam, 2026-06-15):** learn from real gameplay. In real games
Coda steers toward positions *it* rates highly, so positions where its
game-speed search prefers a move that deep Stockfish refutes are exactly the
systematic blindspots that cost rating. Build a corpus of these, turn them
into an avoid-move EPD suite, then track down whether SF has pruning
gates/conditions Coda lacks. Potentially unlocks a chunk of Elo.

## Seed case — lichess RpZ9LbYM move 35 (coda_bot Black vs ToromBot, 180+0)

Coda played **35...Bd2**; SF best is **35...Rxe2**. Deep SF: Rxe2 ≈ −1.9
(Black already worse but holding), Bd2 ≈ −4.7 → a **~2.8-pawn error**.

Forensics (deployed Jun-12 binary 263dbea, reproduced single-threaded; full
UCI log pulled from thor):
- **Not TM** — 77s on the clock, spent ~2s, reached depth 28.
- **Not a thread race** — deterministic single-thread.
- **Not a static-eval gap** — Coda's static eval of the start position
  (+1.87 White) is *more accurate* than SF's (−0.47) vs the deep truth
  (+1.9 White); after Bd2/Qxe4 both statics agree (~+0.8). Coda's *deep*
  search of the bad line even reaches −618 by d26 (exceeding SF). So the
  eval can see it.
- **Mechanism = effective-depth on a quiet refutation.** The danger after
  Bd2 only surfaces at depth ~25. At the root the engine reaches d28 overall
  but the *Bd2 subtree* is searched at reduced depth, so it scores Bd2 at
  ~−1.35 (looks fine) instead of −4.7. At equal nominal depth 24 on the
  forced Bd2 line, **SF reads −472, Coda reads −162** — SF extracts ~3× more
  danger per ply on this line.
- **NOT a single over-aggressive Coda prune** (important, tested): at honest
  equal depth 22, disabling LMR (−125), LMP (−119), futility/RFP/SEE
  individually leaves Coda's read of the line unchanged vs baseline (−119).
  DISABLE_ALL is too slow to reach comparable depth single-threaded. So on
  this position it's a diffuse effective-depth/search-efficiency gap (move
  ordering / EBF / extensions / eval-sharpness in the conversion), plus the
  root reducing the non-PV Bd2 move — not one toggle.
- Current trunk (732a176) plays the correct Rxe2 here (the flip lands at d20,
  harmless, not the emit depth), so a redeploy fixes *this* game; the
  underlying class persists.

**Lesson that motivated the corpus:** a single position is too noisy to
isolate the mechanism (the careful ablation came back inconclusive). A
corpus lets you ask "across N of these, does disabling gate X systematically
recover the danger?" — statistically answerable where one position isn't.

## Tooling — `scripts/overrate_corpus.py`

Scan mode: for each Coda-to-move position (move range, not in check) in a set
of games, compute Coda game-speed eval + SF deep ground truth, then SF's eval
of the *played* move and of *current-Coda's preferred* move:
- `played_loss = SF(best) − SF(played)` — the real-game error (any binary).
- `coda_loss   = SF(best) − SF(current-Coda move)` — **LIVE** blindspot.

Flags on `max(played_loss, coda_loss) ≥ thresh`. Build mode merges per-chunk
TSVs into a ranked markdown corpus + an **avoid-move EPD** (`bm` = play this,
`am` = avoid). LIVE entries (current main still errs) are the high-value
targets; `played` entries are real-game errors trunk may already avoid
(regression-test value).

**Calibration (load-bearing):** SF ground-truth depth must be **≥ 24**. At
depth 22 SF *itself* prefers Bd2 — the refutation only surfaces at d24+ — so
a shallow reference is blind to exactly the class we hunt. Validated: at
sf-depth 24 RpZ9LbYM m35 flags with played_loss 370cp, SF best Re2.

Run (parallel, ~8 workers, idle box — stop the OB worker first, restart
after): see `/tmp/overrate_corpus/run_all.sh`. Source = losses + draws (the
bots draw ~70%, so draws hold most of the overrating signal, not just losses).

## Corpus results — run 1 (174 games, sf-depth 24, Coda 0.6s)

**Thin, and the thinness is the finding.** 5554 Coda-to-move positions across
174 losses+draws games (both bots). But **5365 were draw positions vs only
188 loss positions** — the bots draw ~97% vs similar-strength opponents.
Flagged at ≥150cp move-loss: **2, both `played` (historical)**, **0 LIVE**.

Move-loss distribution (cp):
- `played_loss` (real-game errors): ≥100: 3, ≥150: 2, ≥200: 1.
- `coda_loss` (LIVE — current trunk still prefers the refuted move):
  ≥100: 4, **≥150: 0**. Largest LIVE = 149cp (RpZ9LbYM m36, in the already-
  lost position right after the Bd2 blunder).

Takeaways:
1. **Current trunk has essentially no large LIVE search blindspots vs SF-d24**
   across this sample. Its move choice tracks SF-d24 closely. Either the
   "search-blindspot Elo" is smaller than hoped, or this probe is
   under-powered (below).
2. **lichess deploy games are a poor source.** ~97% draws → the
   overrating-into-trouble signal (which lives in losses) is starved; only
   **22 real losses exist** across 400 games/bot. The richer source is
   **Coda-vs-stronger-engine games** (Coda loses more; blindspots exposed),
   not the draw-heavy lichess pool — a generated Coda-vs-SF (handicapped/
   node-capped) gauntlet would feed this far better.
3. **SF-d24 ground truth may be too shallow** — it cannot flag refutations
   that need d27+, which are exactly the hardest (Bd2 itself needed d24). So
   Coda matching SF-d24 ≠ Coda is right at d30.

### Run 2 (refinement) — all 22 real losses, deeper SF
<!-- RUN2_STATS -->

## Next: the SF-gate comparison

Once the corpus has ~20–40 quiet avoid-move positions, the payoff step is to
test, across the whole corpus, which (if any) Coda reduction/pruning gate is
systematically responsible — and where it differs from SF's carve-outs.
SF reduces less / not at all on: TT & PV moves, moves giving check, killer/
counter & high-history moves, `ttCapture`, certain `cutNode`/`improving`
combinations, and has explicit extension logic (singular, check, etc.). The
corpus is the instrument to find which exemption, ported to Coda, recovers
the danger on the most positions without costing speed — then SPRT it.
The EPD suite (`bm`/`am`) is the cheap inner-loop test for any such change.
