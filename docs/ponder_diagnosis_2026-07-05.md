# Ponder deficit diagnosis — 2026-07-05

**Symptom (Adam's H2H study, titan, 10+0.1):** -84 no-ponder → -131 ponder
vs SF ≈ ~50 Elo pondering deficit. Replicated by the instrumented RR
below: **-144 ±32** (150 games, ponder both sides).

Three evidence legs: cross-engine source research (SF/Berserk/
PlentyChess/Viridithas + non-ponder engines), a full Coda ponder
lifecycle audit (code + P7→P13→P14→Option-C history), and an
instrumented 150-game RR (cutechess -debug UCI capture, parser at
scratchpad/ponder_parse.py, log preserved on titan:/tmp/ponder_rr).

## Empirical results (titan RR, 10+0.1, Hash=256, T=1)

| Metric | Coda | Stockfish |
|---|---|---|
| Ponder-hit rate | **63.9%** (5435/8502) | **63.0%** (5469/8686) |
| Bestmoves WITHOUT ponder hint | **4.6%** | 0.5% |
| Post-ponderhit spend, median | **169 ms** | **71 ms** |
| Post-ponderhit spend, p25 | 112 ms | **3 ms** |
| "Instant" replies (≤120ms) | 30.0% | **63.2%** |
| Post-hit spend, mean | 290 ms | 177 ms |
| Post-hit spend, max | **4147 ms** | 2150 ms |

## Diagnosis (in order of magnitude)

**1. Post-ponderhit time policy — the dominant cause (~4s/game).**
Hit prediction quality is EQUAL (63.9 vs 63.0) — the deficit is entirely
in what happens after the hit. SF's model: full charge for pondered time
(budgets fixed at `go ponder`, never recomputed), made profitable by two
compensators — a +25% optimum bump on every move when Ponder is on
(timeman.cpp:134), and `stopOnPonderhit`: soft-stop satisfied during
pondering ⇒ instant emit at ponderhit (p25 = 3ms above is this
mechanism), with a root fail-low revoking the instant-reply so extra
time is spent exactly when the pondered conclusion destabilized. Coda's
Option C (50% credit) SATURATES to the 50ms floor at STC (any ponder
≥ 2×soft zeroes the budget — i.e. nearly every hit), and the realized
spend is then iteration-quantized: finish the in-flight deep iteration
(hard+500ms-grace bound only — the 4147ms max is exactly the audit's
predicted worst case), sometimes start another. Mean excess vs SF ≈
113ms/hit × ~36 hits/game ≈ **4s of own clock per game = 40% of the
10+0.1 base clock**, compounding through every subsequent budget.

**2. Forfeited ponders (secondary, ~few Elo).** Coda emits no ponder
hint on 4.6% of moves (PV<2, illegal-hint drops, root-TB paths) — the
GUI then cannot ponder at all that turn. SF/Berserk manufacture a hint
via TT-probe-after-bestmove (extract_ponder_from_tt / Berserk
search.c:192-205) and drop only 0.5%.

**3. Exculpated:** ponder-move selection quality (hit rates equal);
ponder-miss handling (no reference engine does anything special; our
200ms miss floor is a small overspend but exists to paper over the
T≥2 thread race). Also noted: Reckless, Obsidian, Alexandria have NO
ponder support — past ponder-on pool RRs measured less than assumed.

**Validation asymmetry (why this survived so long):** OB/fastchess
cannot ponder — every SPRT ever run is blind to this entire path. The
PonderhitCreditPct sweep Option C was created for was never run; the
2026-06-13 TM audit's Track C (exactly the SF mechanisms above) was
never fired; 10+0.1-with-ponder was untested territory until this RR.

## Fix plan (the pieces work as a SET — single-knob attempts failed
before: Phase 14 v1 H0 −35, Option C never re-validated vs the −20)

- **P1 `stopOnPonderhit`**: evaluate the soft-stop at iteration
  boundaries DURING pondering; if satisfied, mark done → instant emit at
  ponderhit; root fail-low re-arms thinking. (SF search.cpp:563-571 +
  411-418 pattern.)
- **P2 mid-iteration post-hit enforcement**: when ponderhit arrives with
  the (credited) soft already expired, stop promptly instead of
  finishing the in-flight iteration; shrink/kill the 500ms grace on this
  path. Also arm abs_deadline on the in-flight path (loss55 forfeit
  class, currently only elapsed+hard+grace).
- **P3 ponder-on optimum bump**: +25%-class optimum when the Ponder UCI
  option is on (compute_tm_budgets gains a ponder flag). Funds itself
  via P1's refunds — do not ship P3 without P1.
- **P4 TT-probe ponder-hint fallback** (Berserk-style, with
  is_pseudo_legal + legality verification given the PV_PONDER_BUG
  history) to cut the 4.6% forfeit rate.
- **Validation**: the titan harness + parser above IS the test — target
  the -144 baseline at 10+0.1; also 40+0.4 (deployment-matched);
  non-ponder SPRT [-2,1] to prove the changes are inert where OB can
  see; instant-reply rate and post-hit spend distributions as input
  metrics before Elo.
