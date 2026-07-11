# The SF Gap Is Mid-Band Search Quality — decomposition + input-metrics program (2026-07-11)

## TL;DR

The ~75-Elo STC gap to SF18-dev decomposes as: **per-node mid-band search
quality is the dominant addressable term (~40-50)**, **eval ≈ 0 vs SF17 but
~30 real & accessible vs SF18** (their threat-net — the v10 stack already
targets it), **NPS ≈ parity on deployment hardware**, active TM ≈ 0 ±20.
The search-quality gap is **concentrated in the mid-band (~50k–150k
nodes/move, depth ~14–20)** — which is where real games at STC/LTC/
deployment TCs are actually decided. Proposed next: an **input-metrics
layer for search** (decision suite + search-health dashboard) so ideas —
including novel ones — iterate in minutes, with Elo tests reserved for
candidates that move the metrics.

## Evidence (all 2026-07-10/11, Atlas, conc 32, noob_4moves, Hash=512; SF18
= dev-20260318-d173a065; full tables in experiments.md)

**Node accounting audited first** (both engines count node-entries incl.
qsearch; Coda's final-info under-report fixed in 90dade3) — the fixed-node
frame is fair, and in-situ per-move wall-times showed the fixed-node runs
granted Coda only ~1.16× time odds under contention (idle it would be 1.7×;
contention compresses SF's NPS advantage far more than ours).

| frame | Coda vs SF17 | Coda vs SF18 |
|---|---|---|
| fixed 15k nodes | **+25** | −69 |
| fixed 50k | +5 | −64 |
| fixed 150k | **−33** | −78 |
| fixed 500k (90% draws) | −29 | −55 |
| fixed 0.20s/move | −5 | −52 |
| fixed 0.28s/move (STC-matched) | — | −66 |
| real STC 10+0.1 (1000 games/eng) | −45 | −79 |

- **Eval: ≈0 vs SF17, ~30 accessible vs SF18.** +25 over SF17 at 15k where
  search barely runs — but −69 vs SF18 in the SAME frame. The two columns
  diverge at low nodes because SF18's threat-net eval is stronger than ours
  (SF added threat inputs 17→18, paying NPS for them). That ~30-Elo slice
  is real, eval-shaped, and already targeted by the v10 stack — the SF17
  framing must not quietly retire it. (Hercules review, 2026-07-11.)
- **The slide +25 → −33 vs SF17 (15k→150k) is pure search quality** — NPS
  and TM are excluded by construction. Raw Elo flattens by 500k, but the
  500k point is 90%-draw compressed: "no deep cliff" is UNMEASURABLE there
  (UHO-at-fixed-nodes would resolve it), not proven small. The measured
  leverage is the mid-band.
- **TM absolved at current power — on the MEASUREMENT, not the H0 count**:
  st=0.28 (−66) vs real TC (−79/−73), minus Coda's ~6% movetime overspend
  flattering the st frame → residual ≈ 10 ±20. The nine failed TM
  interventions corroborate but are weaker evidence (OB self-play is
  partly TM-blind — measured 2026-07-10, subtree factor). Blunder-spend
  conditioning is SF-like. The st=0.20→0.28 widening (~14 Elo) is the
  node-scaling confound that had been mis-read as "NPS + active TM".
- **NPS: near-parity on deployment hardware — so the fixed-node frame ≈
  the deployment frame.** On modern Ryzen with all cores stretched, Coda
  is at parity-to-ahead of SF18 (<10% residual; SF paid NPS for threats).
  The 1.7× idle deficit is a Zen-1-class artifact. This is what licenses
  reading the fixed-node decomposition as deployment-relevant, and makes
  the mid-band focus stronger than a hedged "platform-bimodal" implies.
- **Ponder**: separate frame, parked by directive (2026-07-11). We are a
  strong ponderer (#2-tier); SF/Berserk extract ~17-28 more pool-relative
  Elo from it. Revisit after the ponder-off gap closes.

Confidence: HIGH on the decomposition direction (three independent frames
agree: node-ladder, movetime-ladder, TC). Numbers carry Atlas-specific
conditions (contention, this SF build); magnitudes ±15-20.

## Why this reframes the search work

1. **"Search feels exhausted" has been repeatedly wrong** (186 self-play
   Elo from search since 2026-06-06) and this explains why: a ~60-Elo
   mid-band deficit vs SF17 means the ore body is large. Feelings of
   exhaustion reflect the ITERATION COST of the current loop (audit → port
   → SPRT, days per idea), not the size of the deposit.
2. **Validation moves to fixed nodes.** Search-quality candidates should be
   gauged at **fixed 150k vs SF17** (calibrated 2026-07-10: baseline −33
   ±8 at 1600 games, ~1.5h on Atlas) — eval strength can't mask search
   effects there, and TC noise is excluded. SPRT remains the merge gate;
   the gauge is the direction-finder.
3. **The audit-and-port pattern is hitting its ceiling** (Adam, 2026-07-11).
   It's a mimicry gradient: it converges toward the reference engines,
   can't exceed them, and transplants fail when the surrounding system
   differs (measured: TM P1-P4, subtree factor). It stays in the toolbox,
   but the growth path needs to support NOVEL ideas — which requires
   metrics faster than SPRT.

## The input-metrics program for search

Mirror of the TM methodology (input metrics first, Elo second — validated
2026-07-10/11): find fast-measurable quantities that (a) differ between
Coda and SF in the mid-band, (b) move when search improves, (c) cost
seconds-to-minutes per measurement.

### Tier 1 — decision quality (the metric that IS the target)

**Mid-band decision suite.** Harvest positions from our own games where
the mid-band decision went wrong: from the 20k+ archived RR games (plus
future ones), take positions where Coda-at-150k played move A, a deep
reference verdict says B, and the margin is decisive. Curate a few hundred.

**Oracle design** (layered, to keep labels honest):
- Primary labeler: SF-dev at 2-5M nodes, pinned binary+budget, verdicts
  cached per FEN. **Accept-set semantics**: "correct" = any move within
  ~30cp of best; a position is only admitted if Coda's actual move was
  ≥100cp worse (MultiPV=2 for the margin; `searchmoves` to price A). The
  suite measures "avoided a quantified mistake", not "matched SF's move".
- **Coda-deep triage**: run Coda at ~5M nodes on every candidate.
  *Search-fixable* (Coda-deep finds an accept-set move) → the suite: the
  knowledge is in OUR eval+deep search, the mid-band fails to surface it;
  the target is our own deep behavior surfaced earlier. *Eval-limited*
  (Coda-deep can't find it) → excluded from the search suite, banked as a
  pre-filtered **eval-blindspot corpus** — which feeds exactly the ~30-Elo
  threat-net/v10 slice from the decomposition above.
- Corroboration: harvest only mistakes that were punished (mover lost from
  a not-lost position); cross-check a ~50-position sample with an
  architecturally different opinion (lc0) + eyeball.

**CAVEAT (Hercules review): the suite is a faster mimicry gradient, not an
escape from it.** Tier-1 truth is still "SF-dev@5M says so" — optimizing
suite accuracy converges toward SF and caps there; a better-than-SF move
that disagrees with the oracle scores WORSE. That is acceptable for the
current goal (we are behind SF; closing works) and the Coda-deep triage
softens it (the scored target is our own deep verdict) — but it does not
reward EXCEEDING SF. The dashboard (mechanism anomalies) is what supports
going past SF; if/when the goal becomes exceeding, the truth-source must
become stronger than one SF search (deeper multi-engine consensus, game
outcomes).

Then:

- **Suite accuracy @ N nodes** = P(engine finds B) at 15k/50k/150k/500k.
  One full sweep ≈ minutes. SF17/SF18 runs calibrate what "good" looks
  like at each budget; the Coda-vs-SF accuracy gap by budget should
  REPRODUCE the RR mid-band slide (validation step — if it doesn't, the
  suite is mis-harvested).
- **Node-ROI curve** = Δaccuracy per node doubling — the scaling metric.
  A candidate that lifts accuracy@150k without hurting accuracy@15k is a
  mid-band fix by construction; send it to the 150k gauge, then SPRT.
- Refresh the suite periodically from new games (it stales as we fix what
  it measures); keep a held-out control split to catch overfitting. This
  is WAC-methodology but auto-harvested at the exact budget where we're
  weak, and it supports novel-idea iteration in minutes.

### Tier 2 — search-health dashboard (mechanism diagnostics)

One command (extend `coda bench`-style stats; much already exists in
SearchInfo.stats / pruning-stats) over a fixed position set, reporting per
node-budget. Candidate metrics, chosen because each has a *mechanistic*
link to mid-band efficiency and an SF-comparable or self-trackable value:

- **Move-ordering quality**: first-move-cut rate at cut nodes; best-move
  index distribution. Ordering decay at depth = tree bloat exactly where
  we're weak.
- **Re-search overhead**: fraction of nodes inside aspiration re-searches
  and LMR re-searches. Related live finding: **our aspiration windows fail
  low on >50% of searches** (TM Phase-0, flf median 1.34) — SF-anomalous,
  costs re-search nodes AND corrupts TM/LMR signals. First dashboard
  candidate to chase.
- **Root/interior score stability**: root-score variance across iterations,
  best-move flip count per completed depth. Search noise wastes nodes and
  poisons every depth-indexed heuristic. (Also feeds back into TM signal
  quality — the reason SF's TM mechanisms didn't transplant.)
- **Where nodes live**: qsearch fraction, depth-histogram of node spend,
  seldepth/depth ratio, TT hit/cutoff rates by depth, EBF per iteration.
  Fingerprint-comparable Coda-vs-Coda across candidates; coarse vs SF.
- **Wasted-subtree fraction**: nodes spent under root moves that end the
  search refuted (proxy: root_move_nodes of non-best moves at final
  iteration). High = weak early pruning/ordering at root.

Dashboard runs must be CHEAP (~1 min) and deterministic (fixed nodes, T=1)
so any idea — ours or ported — gets a mechanism read before any games.

### Tier 3 — the loop

```
idea → dashboard (mechanism moved? seconds-minutes)
     → decision suite @ budgets (accuracy@150k up, @15k not down? minutes)
     → fixed-150k gauge vs SF17 (Elo direction, ~1.5h local)
     → OB SPRT [0,3] (merge gate, fleet)
```

Novel ideas enter the same loop as ports — the suite doesn't care where an
idea came from, which is the point. Ideas that move accuracy but fail SPRT
(or vice versa) are themselves diagnostic: they localize where the metrics
mis-measure, tightening the instrument.

### Validation duties (before trusting the instrument)

1. Suite accuracy gap Coda-vs-SF17 by budget must reproduce the RR slide.
2. Dashboard metrics must show a Coda-vs-SF or Coda-vs-Coda-at-depth
   anomaly BEFORE we chase them (the flf>50% aspiration anomaly already
   qualifies).
3. Any metric that a known-good historical change (e.g. the June LMP/LMR
   merges) would NOT have moved is a weak metric — spot-check against 2-3
   recent H1s where feasible.

## Known open items adjacent to this doc

- Movetime ~6% overspend fix (hygiene; biases movetime frames, 1 forfeit
  in 1600 at st=0.2 with timemargin=200).
- Aspiration fail-low anomaly (>50% of searches) — first dashboard chase.
- Endgame TM dispersion (bucket 31-50) unmatched vs SF — parked with the
  TM thread.
- TM input-lab branch (`tm-input-lab`) + dashboard (`tm_pattern_inspect
  --shape`) — infrastructure kept; TM_* lab tunables must never enter SPSA
  (self-play blindness, measured 2026-07-10).
