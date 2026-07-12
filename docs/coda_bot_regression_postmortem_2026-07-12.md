# coda_bot bullet "regression" — investigation post-mortem (2026-07-10 → 2026-07-12)

## TL;DR

coda_bot (Thor, -t16/-t8, ponder, bullet) lost ~70 rating points from a June-21
peak while codabot (VPS, -t4) gained over the same window. Two days of
systematic investigation — ~15 instruments across code eras, nets, threads,
TCs, hash sizes, ponder, and two uArchs, plus lichess game forensics and
SF-refereed phantom analysis — found **no reproducible engine regression at
any locally reachable operating point**. Every axis measured shows the engine
improving monotonically June → July (~+90 Elo cross-engine at T1/STC, +45 to
+76 differentials at T4/T8, phantom metrics improving era over era). The
lichess decline decomposes into **opponent-pool dynamics** (farm targets left
or strengthened, the 3000+ bullet band is a structural draw-fortress) plus,
possibly, a deployment-only residual our instruments cannot reach.

The hunt still paid: it surfaced and shipped three real fixes (C2 cross-move
TM pollution, TB-band 50mr downgrade, threat-verifier coverage), produced two
reusable instruments (phantom-rate referee, fortress-plateau probe), mapped
the validation stack's structural blind spots, and registered a falsifiable
forecast (below).

**Registered forecast (2026-07-12):** head (with the corrhist residual fix,
C2, TB-band downgrade) deployed to both bots 2026-07-11. If an engine-side
cause our instruments can't reach was real and is now fixed, coda_bot's
bullet curve turns upward by ~2026-07-16 (4-5 day rating lag at ~100
games/day). If it stays flat while codabot holds, the pool explanation
stands.

## Symptom

- coda_bot: bullet peak ~3085 on Jun 21, steady bleed to ~3005-3010 by Jul 10.
  Deploy lag 1-2 days → peak code ≈ main @ ~Jun 19-20. Thor is dedicated to
  the bot (no host contention). Hash=2048, 5-man EGTB, MoveOverhead=100.
  Ran -t16×1 and -t8×2 at different times; switching had no effect.
- codabot: -t4 VPS, Hash=1024, same EGTB/overhead class — GAINED ~60-80 over
  the same window on the same code stream (deployed together, within days).
- Meanwhile a T1/STC/no-ponder cross-engine benchmark showed the code gained
  **~+88** over the window (Coda.today +35 vs Coda.jun18 −53, same pool).
- Initial trigger was a reported rise in time forfeits at no-inc + ponder —
  this turned out NOT to be present in the rated-bullet data (see forensics).

## Instruments and results

### A. Regime grid (Zeus = Zen5/AVX-512, Thor-identical; Titan = Zen1/AVX2)

| Cell | Result | Verdict |
|---|---|---|
| T1, 10+0.1, Hash≥512, field (Zeus) | today +88 over jun18 | code is good in OB's regime |
| T1 + ponder (Zeus) | today fine | ponder per se innocent at T1 |
| T4, 10+0.1, Hash=128, field (Zeus) | apparent FLIP at 40-53 games (−35 vs +17 → −7 vs 0) | **early-N noise** — see below |
| T4, 3+0.03, Hash=128 AND big-hash (Zeus) | unflipped | short-TC cells clean |
| T4, 10+0.1, Hash=512 (Zeus) | unflipped (+45 differential interim) | hash cells clean |
| T4, 10+0.1, Hash=128, H2H (Titan) | head +76 over jun18 (231g) | no flip, other uArch |
| T4, 10+0.1, Hash=128, field (Titan) | head −2.3 vs jun18 −68.0 vs same pool (300g/arm) | **+66 differential — no flip at N** |
| T8, 60+0 bullet, ponder, Hash=2048, field (Titan) | head −11.6 vs jun18 −56.5 (180g/arm) | +45 differential in the deployment cell |

**Lesson (early-N, both directions):** the "T4 flip" that redirected a day of
investigation was declared at 40 games/engine and evaporated by 250. Our own
<500-game rule applies to exciting reversals exactly as much as to boring
ones.

### B. Throughput probe (Titan)

Fixed positions, `go movetime` 200/2000ms, T=1/8/16, three configs
(jun18+oldnet, HEAD+oldnet, HEAD+newnet): **HEAD faster in every cell**;
L1=32 net costs ~11-13% NPS *uniformly* across thread counts (no bandwidth
wall); no per-`go` dispatch tax. Refuted: SMP throughput regression, L1=32
bandwidth saturation, thread-pool refresh overhead.

### C. June-window code audit (4 mechanism-level slices; docs/bot_regression_audit_2026-07-11.md)

Produced ranked thread-scaling suspects (QS corrhist stand-pat into shared
TT, corrhist all-node volume, TT EXACT-override depth inversion, IIR-retune
tree mismatch) and TM/ponder deepeners (Factor-6 thread_bmc unread
ponder-window churn, FL-EXT band suspension). Status: **none convicted for
this symptom** (the T4/T8 differentials above are positive), but the audit
stands as a map of genuinely under-validated SMP surfaces:
- The EXACT-downgrade counter probe measured the depth-inversion mechanism
  as REAL and thread-scaling (274 → 587 same-key intra-gen downgrades per
  M stores from T1 → T16, ~6 plies lost per event) — modest magnitude,
  unconvicted, worth keeping on the SMP-audit list.
- `NO_CORRECTION=1` at T4 made head WORSE (+76 → +36 vs jun18) —
  correction is net-positive under SMP; the corrhist cluster acquitted.
- Factor-6/FL-EXT remain untested in their firing regime (ponder×threads);
  Hercules' #2692/#2696 tests resolved H0 at STC.

### D. Lichess forensics (both bots, all rated bullet since Jun 1, ndjson)

- **Time forfeits: ~zero throughout** (0-4/week, no increase). The forfeit
  narrative that started the thread is not in the rated bullet data.
- coda_bot weekly perf-vs-Elo-expectation: +1.5..+4.4% through Jun 21,
  −1..−2% after. Draw rate 53% → 80%; mates halved; fav-game (exp>60%)
  conversion 78% → 42%. codabot shows the same fav-conversion decline
  (70% → 43%) at stable overall perf.
- **Same-opponent scores were STABLE** (paired total 63.3% → 61.0%;
  styx_reckless 50/50 with 100% draws in BOTH periods). The decline lives in
  the mix: main farm source (lacosox, 47 games pre) left; farmables got
  +70-90 stronger themselves (SleepMind 2617→2707, grail 2712→2784);
  fortress-band peers (90-100% draws vs us, both periods) dominate
  matchmaking at the rating coda_bot climbed to. Success moved the bot into
  a band where bullet bot chess is structurally drawn.
- Elo-expectation math misprices bot styles (a 60% expectation vs a
  draw-fortress peer is unachievable), so "underperformance vs expectation"
  partly measures the model, not the engine.

### E. Net A/B (net-recipe theory: promotions Jun 11/21/28/29 + Jul 4 tracked the decline)

Same HEAD binary, `-n` override: current E161C665 vs pre-Jun-11 E4B66CE4,
T4 @ 15+0.15, vs a deliberately WEAKER field (Halogen/Starzix/Caissa —
the conversion regime), 252 games/arm:
**netnew 61.1% (+78.5) vs netold 56.0% (+41.6), identical 47% draw rates.**
The current net converts better against weaker opposition at depth. Net
recipe **acquitted** (the promotion-timeline correlation was coincidence).

### F. Phantom-overscore forensics (Adam's theory: we misscore a small % of
positions; search gravitates to them; we think we're winning when we're not)

SF-refereed (depth 16-22) every position where Coda claimed sustained +2.0
(3 consecutive moves), across three code eras under identical conditions:

| Era | Hard phantoms (SF≤+0.5) | Mid band (SF 0.5-1.5) conversion | +3.0×10 fortress plateaus in draws |
|---|---|---|---|
| jun18 | 2/111 | 39% won | 7/146 (4.8%) |
| jul1 (mid-bleed: window depth + pre-fix corrhist) | 1/117 | 54% | 4/130 (3.1%) |
| head | 1/124 | 67% | 2/124 (1.6%) |

- Hard phantoms ("winning claim, actually nothing") are **~1% and flat**.
- The soft-overscore band is large (~80% of claims — partly cross-engine
  eval-scale mismatch) but its conversion **improved monotonically**.
- The fortress-plateau signature (high sustained eval, game drawn — the
  corrhist-rail fingerprint) **declined monotonically** era over era; the
  registered prediction of a mid-bleed spike (depth × raw-corrhist
  amplifier) FAILED at T4/STC depth.
- Caveats that keep the theory alive at other operating points: 10+0.1
  games are short, draw adjudication may amputate fortress phases, and T4
  is roughly half the deployment depth. A T8 long-TC plateau probe is the
  remaining unexplored cell.
- The corrhist residual fix (Jul 8-9, +17±15 cross-engine) removed the one
  *proven* phantom amplifier; era-over-era plateau decline is consistent
  with that fix and with general improvement.

## Refuted along the way (kept for the learning)

1. Time-forfeit increase — not present in rated bullet data.
2. C2 cross-move TM pollution as the cause — real bug, fixed (#2683 H1,
   ponder-RR ≈ +7), but born Jul 8: couldn't cause a Jun-21 onset.
3. Host contention on Thor — Thor is dedicated.
4. -t16×1 vs -t8×2 concurrency mix — switching had no effect (Adam).
5. SMP throughput / thread-pool overhead — HEAD faster at every T.
6. L1=32 bandwidth wall — cost flat in thread count.
7. Hash contention — flip absent at Hash=128 short-TC and at Hash=512
   long-TC; bots run 1-2GB.
8. TB probe storms — no locks in the probe path; interior probes REDUCED by
   the Jun-26 changes; EGTB symmetric across bots.
9. The T4 "flip" itself — early-N noise (40-53 games).
10. uArch (Zen5/AVX-512 vs Zen1/AVX2) — nothing reproduced on either.
11. Net-recipe conversion regression — netnew beats netold in the exact
    conversion regime.
12. Corrhist-amplifier bloom (mid-bleed phantom spike) — plateau metric
    declined monotonically instead, at T4/STC depth.
13. TT near-miss / TT-damp field divergences as regressions — both are 0/6
    vs the reference field yet EARN Elo (removal −4.1 H0; raw-return
    CI-stopped ~−0.3): logged as intentional divergences.

## What the hunt shipped anyway

- **fix/tm-cross-publish-gate** (C2): cross-move TM trend no longer polluted
  by ponder-miss/analysis/TB scores. Merged 26f9829, #2683 H1.
- **experiment/tt-tb-50mr-downgrade**: mate+TB-band 50mr downgrade at every
  TT read (SF/Reckless shape) + ply double-count fix. Merged 1e45d0a, #2687 H1.
- **test/threat-verifier-coverage**: CODA_VERIFY_NNUE now covers threat
  drift (was blind); pop/lazy-gap fuzzer for the replay regime of the two
  historical ~200-Elo bugs. Merged 9522f53.
- **Instruments** (scratchpad → worth productizing): phantom-rate referee
  (SF re-scores claimed wins) and fortress-plateau probe (sustained-high-
  eval draws). Candidate additions to net_report as pre-promotion gates —
  NB the existing blindspot metric was separately shown inverted/false.
- **docs/persistent_state_audit_candidates_2026-07-10.md** +
  **docs/bot_regression_audit_2026-07-11.md**: ranked audit maps of
  persistent-state and SMP-validation blind spots (Tier-2 candidates C4-C7
  remain uninvestigated).

## Meta-lessons

1. **The validation stack has a regime hole.** Everything gates at
   T1/no-ponder/STC-LTC vs near-peers. Deployment is T8-16/ponder/bullet vs
   a heterogeneous pool. Four+ mechanisms landed in three weeks that cannot
   fire in any test we run (and, this time, turned out benign or fixed —
   but we couldn't know that without building the instruments). Standing
   fix: a periodic deployment-regime RR (T=8, ponder, bullet, fixed pool)
   as a release gate for TM/SMP/TT-replacement changes.
2. **Self-play and near-peer pools mask exactly the failure classes that
   matter on lichess**: eval-bias defects cancel in self-play (corrhist fix
   was +17 cross-engine, ~0 self-play); conversion defects cost nothing
   against equals. Conversion-sensitive tests need weaker opponents or
   unbalanced books.
3. **Early-N discipline applies to exciting results.** The T4 flip (40
   games) redirected a day. Both directions of the <500-game rule.
4. **Lichess is not a measurement instrument** (~100 games/day, pool
   dynamics, style mispricing in Elo expectations). It can surface a
   problem; it cannot localize or validate one. Now in CLAUDE.md.
5. **Rating ≠ strength when the pool moves.** coda_bot's "regression"
   coexists with the engine measurably improving ~+90. Triaging future
   wobbles: check pool composition/farm-target churn FIRST (the forensics
   scripts in this session's scratchpad took ~an hour and settled more than
   a day of gauntlets).
6. **Field-consensus ≠ Coda-optimal in either direction** (near-miss earns
   Elo at 0/6 consensus; QS-ttPv preservation loses it at 6/6). SPRT the
   mechanism; don't port the consensus blindly.

## Open items

- The registered forecast (top) — check coda_bot's curve ~Jul 16.
- Optional final deep cell: T8 long-TC plateau probe (fortress phases
  un-amputated by adjudication) if the forecast disappoints.
- Productize phantom-rate + plateau metrics into net_report.
- Deployment-regime RR as a standing release gate (design in
  bot_regression_audit doc).
- Tier-2 persistent-state candidates (C4-C7) from the Jul-10 audit.
