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
| Fixed nodes 10k | **+4.6** | **+82.6** | Eval-dominated regime: Coda ≥ SF, ≫ Reckless |
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

1. **Eval: a Coda STRENGTH, not a gap.** At eval-dominated budgets (10k
   nodes) Coda edges SF and crushes Reckless (+83). The old model "vs
   Reckless is eval-refinement" is dead (memory corrected 2026-07-06). Do
   not spend eval effort chasing either engine; spend it on absolute gains
   (the skip-recipe program).
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

## Exhibit: fixed-10k-nodes Top-20 RR (2026-07-06) — the eval leaderboard

Same top-20 pool as the TC round-robins, but every engine limited to
nodes=10000 per move (no TM, no NPS). This isolates eval + QS + shallow
ordering. 858 games/engine, CI ~±20:

| # | Engine | Elo | | # | Engine | Elo |
|---|--------|-----|-|---|--------|-----|
| 1 | **Coda** | **+197** | | 11 | Hobbes | −1 |
| 2 | Stockfish | +179 | | 12 | Stormphrax | −13 |
| 3 | Reckless | +151 | | 13 | Alexandria | −14 |
| 4 | Starzix | +106 | | 14 | Halogen | −55 |
| 5 | PlentyChess | +86 | | 15 | Astra | −70 |
| 6 | Viridithas | +74 | | 16 | Clover | −81 |
| 7 | Cinder | +46 | | 17 | Berserk | −141 |
| 8 | Obsidian | +41 | | 18 | Caissa | −144 |
| 9 | Integral | +30 | | 19 | Rubichess | −176 |
| 10 | Raphael | +17 | | 20 | Icarus | −246 |

Readings (and caveats):
- **Coda-vs-SF is a TIE, not a lead**: Coda's `go nodes` gate checks the
  global counter at 4096-node granularity (search.rs ~1300), consuming
  ~12.3k for a 10k limit (+23%, worth ~+15-30 here); SF/Reckless enforce
  exactly (10001-10005, measured). The +18 point gap ≈ the artifact.
  Defensible claim: **Coda's eval+QS is SF-tier and ahead of everything
  else** (Coda > Reckless is significant: +46 at ±29 combined).
- **The podium = the three threat-input NNUE engines** (Coda v9 threats,
  SF FullThreats/SFNNv12+, Reckless): threat features let static eval see
  tactics others need a ply of search for — maximal value at tiny budgets.
  Correlation, not proof (next-tier archs unverified), but 3-for-3.
- **CCRL-3600 speed/search-built engines collapse** (Berserk −141,
  Rubichess −176, Caissa −144, Alexandria −14): their TC strength is
  NPS/search-shaped. This table + the 100k fixed-node numbers are two ends
  of one curve: Coda starts level-with-SF and SF pulls away as budget
  grows — the cleanest visualization of where the remaining search gap
  lives.
- Fix-before-publishing note: a per-node limit check when max_nodes > 0
  (5 lines) would make future fixed-node tests exact.

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
