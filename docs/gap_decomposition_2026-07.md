# Engine Gap Decomposition — SF & Reckless H2H studies (2026-07-05/06)

Adam's controlled H2H studies isolating WHERE the gaps to the two strongest
engines live: fixed nodes (removes TM+NPS), fixed depth, TC sweep, fixed
time/move, threads, ponder. 300-game H2Hs, noob4moves, Hash 256-512, T=1
no-ponder unless stated. SF runs on titan, Reckless on atlas.

**VINTAGE NOTE (critical for reading the ponder rows):** the SF study ran
BEFORE the 2026-07-05 ponder P1-P4 fix (`def8081`); the Reckless study ran
AFTER it. The SF ponder leg was re-run post-fix on 2026-07-06: −130.9 → −100.0
(both vintages shown in the table).

## The two maps

| Leg | vs SF-dev | vs Reckless 0.10-dev | Reading |
|---|---|---|---|
| Fixed nodes 10k | +4.6 (**overshoot-inflated**; honest ≈ −30..−45) | +82.6 (inflated; honest ≈ +30) | Eval-dominated regime: SF > [Reckless ≈ Coda] ≫ field. Overshoot worth ~50 at 10k, <8 at ≥20k (absolute 4096-boundary artifact) — bigger-budget rows stand |
| Fixed nodes 20k | −51.3 | −8.1 | The reversal point for both |
| Fixed nodes 50k | −58.5 | −29.0 | |
| Fixed nodes 100k | **−78.9** | **−18.5** | Quality-per-node gap: SF huge & compounding; Reckless small |
| Fixed depth 8 | **+110.1** | **−49.0** | Ply density: Reckless > Coda > SF |
| Fixed depth 10 | +10.4 | −86.3 | (pruning weight: SF > Coda > Reckless) |
| Fixed depth 12 | −27.9 | −88.7 | |
| TC 5+0.05 | −119.1 | −52.5 | |
| TC 10+0.1 | −83.8 | −54.9 | |
| TC 20+0.2 | −69.2 | −54.9 | |
| TC 40+0.4 | −59.6 | −62.0 | SF gap SHRINKS with TC; Reckless gap FLAT-to-growing |
| TC 180+1 | −59.6 (81% draws, 1W-18L-81D) | (pending) | SF residual ~−60 is structural |
| Fixed t/move | st=0.1 −102, st=0.2 −90 | (Reckless forfeits in st mode) | Coda TM ≈ SF-par |
| Threads T2 | −94.9 (par −69, post-#2542) | −63.2 (par ~−55) | ~8-10 Elo/doubling relative SMP deficit, |
| Threads T4 | −74.1 (par −60) | −75.3 | consistent vs BOTH opponents |
| **Ponder** | PRE-fix −130.9; **POST-fix rerun −100.0** (2026-07-06) vs −83.8 no-ponder | **−4.6** vs −54.9 no-ponder (POST-fix) | P1-P4 at H2H scale: +31 vs SF, +50 vs Reckless |

## Synthesis (what lives where)

1. **Eval: world #2-tier (tied with Reckless), ~40 behind SF at 10k.**
   (Revised after exact node accounting — the "Coda edges SF" first read
   was a +23%-budget artifact worth ~50 Elo at this TC.) The old model "vs
   Reckless is eval-refinement" stays dead: we're at Reckless parity on
   eval, and the vs-Reckless TC gap decomposes to TM/SMP/search instead.
   vs SF, BOTH components are real: ~40 of eval+QS+shallow machinery even
   at tiny budgets, growing to −60..−79 as their deep-search machinery
   engages. Eval effort = absolute gains (skip-recipe program), not
   chasing.
2. **Search quality-per-node: the SF-specific mountain.** −79 @ 100k and
   compounding, vs only ~−20 to Reckless. SF converts thin plies best
   (their d12 beats our dense d12); Reckless wins with dense plies +
   operational polish. Owner: Hercules — the +1-3 conveyor (Tier 1/2 SF
   ports) + the regime-shift track (fractional LMR + base/moveCount DOF +
   joint SPSA). Verification instrument: re-run the fixed-nodes sweep;
   success = the −51→−79 slope flattening.
3. **Active TM: ≈ par with SF, ~30 behind Reckless.** The TC-sweep SHAPE
   proves the attribution: an NPS-driven gap shrinks with TC (NPS→Elo
   conversion falls from ~100 to ~20/doubling — the SF series' shape); the
   Reckless series is flat-to-growing = TM/scaling signature. Implication:
   Reckless's TM is better than SF's. We have their source; timeman study
   + per-move clock-distribution comparison (tm_pattern_inspect) on these
   PGNs is a readable ~30 Elo. Owner: UNOWNED as of 2026-07-06.
4. **SMP: ~8-10 Elo/doubling relative deficit, confirmed against both.**
   Candidate mechanism from the SF audit: shared histories across threads
   (SF: NUMA-shared atomic conthist/pawn/corrhist; Coda: spawn-copies).
   Owner: Atlas (SMP thread; includes implementing SF-style optimism,
   which SMP work needs — this also covers the search audit's Tier-3
   optimism item, previously #671 H0 untuned).
5. **Ponder: largely fixed; residual vs SF owned.** P1-P4 took vs-Reckless
   from −55 to −4.6 (**level at deployment conditions** — lichess is
   ponder-on, matching live observation) and vs-SF from −131 to −100
   (rerun 2026-07-06). Ponder-quality ordering now: SF > Coda >> Reckless.
   Residual ponder-delta vs SF ≈ 16 on the static −83.8 baseline — but the
   rerun ran on a post-futility-merge Coda, so a FRESH no-ponder baseline
   is needed to pin the true residual (plausibly ~16-22). Known remaining
   mechanisms from the diagnosis doc: instant-reply latency (SF p25=3ms vs
   our 112ms) and hint-less bestmoves (4.6% vs 0.5%). Owner: Zeus.

## Exhibit: fixed-10k-nodes Top-20 RR — the eval leaderboard (FINAL, exact accounting)

nodes=10000/move, no TM/NPS; isolates eval + QS + shallow ordering. First run
used the overshooting Coda (+23% nodes at this budget = +53 Elo measured
field inflation, independently confirmed by a 2500-game self-play H2H of the
exact-vs-overshoot binaries: -49.7 +-8.8). Table below = RERUN with exact
`go nodes` enforcement (158bb00), 950 games/engine, CI ~+-18:

| # | Engine | Elo | | # | Engine | Elo |
|---|--------|-----|-|---|--------|-----|
| 1 | Stockfish | +181 | | 11 | Stormphrax | +6 |
| 2 | Reckless | +147 | | 12 | Hobbes | +3 |
| 3 | **Coda** | **+140** | | 13 | Alexandria | −29 |
| 4 | Starzix | +92 | | 14 | Astra | −36 |
| 5 | PlentyChess | +85 | | 15 | Halogen | −50 |
| 6 | Viridithas | +83 | | 16 | Clover | −80 |
| 7 | Cinder | +48 | | 17 | Berserk | −138 |
| 8 | Obsidian | +37 | | 18 | Caissa | −147 |
| 9 | Raphael | +37 | | 19 | Rubichess | −185 |
| 10 | Integral | +29 | | 20 | Icarus | −257 |

Readings:
- **Coda's eval+QS is Reckless-tier, statistically tied for world #2**
  (147 vs 140, CIs overlap), ~50 clear of the next tier, ~40 behind SF.
  The first run's "Coda #1 / edges SF" was the accounting artifact.
- **SF leads even at eval-dominated budgets** — their eval+QS+shallow
  machinery is ~40 ahead here, then the deep-search machinery widens it to
  −60..−79 as budget grows. Both components are real; the deep one is bigger.
- The podium = the three threat-input NNUE engines (correlation note stands).
- Speed/search-built CCRL-3600 engines still collapse (Berserk −138,
  Rubichess −185): their TC strength is not eval-carried.

## Measurement caveats

- Node counting may not be identical across engines (fixed-node legs).
- Fixed-depth H2H levels are NOT commensurable (nominal depth = different
  tree per engine); the SLOPE and cross-engine ordering are the signal.
- st mode: SF forfeit-contaminated at st=0.05; Reckless forfeits in st
  mode generally — fixed-time legs are Coda-vs-SF only, ≥0.1s.
- 180+1 rows are ~100 games (±28) — read the draw structure, not the point.

## Standing re-run triggers

- Fixed-nodes sweep vs SF: after each search-wave lands (the audit's
  pre-registered verification).
- SF ponder leg: DONE 2026-07-06 (−100.0). Next: fresh no-ponder SF
  baseline on current Coda to pin the residual ponder-delta; re-run again
  after Zeus's follow-up.
- Reckless 180+1: pending.
- Threads legs: after Atlas's SMP work lands.
