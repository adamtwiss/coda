# Loss / Win Analysis — 1400-game Rivals Gauntlet (2026-04-28..29)

Consolidated writeup of the SF-arbitrator + classifier work on `tough_rivals.pgn`
(1400 games, 8 engines, 40+0.4, hash=512MB, EGTB-on, run 2026-04-28). Combines
the calibration anchors, loss-class taxonomy, win/loss asymmetry, and the
45-moderate-stepped → 12-ablation × 6-depth deep-dive that followed.

CLAUDE.md §Strength Frontier and §Improvement Portfolio reference this doc
as the primary source for mechanism findings and rivals-tier strategy.

## Executive summary

- Coda is **mid-pack at peer tier**: −16 ±9 Elo across the rivals pool at
  40+0.4. Closing 50 Elo here puts us roughly at parity with the top of the
  near-peer band (Tarnished/Horsie). This is the **next-50-Elo target**, not
  closing the SF gap.
- **Coda doesn't blunder.** 92.4% of our per-move searches are SF-accurate
  at depth 18. Median worst-move-per-loss is only −160cp.
- **We lose via dynamic-eval drift, not static-eval errors.** ~60% of losses
  are slow erosion (dynamic eval drifts 0 → −100 → −200 → −400 across many
  moves). The mechanism is search-side: ordering, pruning, LMR, TT. The
  static NNUE is fine — the search-driven refinement is where the gap lives.
- **Win/loss asymmetry:** wins build 92% gradually; losses concede 49% via
  moderate-step or sudden cliffs. Opponents exploit our pruning/horizon
  blind spots ~6× more frequently than we exploit theirs.
- **The +3-6 ply ordering/pruning bucket is the dominant frontier.** Of the
  45 moderate-stepped candidates surfaced from losses, 91% converge to
  SF-best within depth 24, and 40% sit in the +3-6 ply deficit bucket —
  reachable by ordering improvements and pruning carve-outs.
- **Pruning recovery is distributed**, not LMR-exclusive. Single-feature
  carve-outs cap at ~50% coverage; multi-feature carve-outs on a shared
  trigger (e.g. threat-aware loosening of NMP+RFP+FUT+LMR together) have
  higher leverage.

## Calibration anchors

### 100-game H2H vs SF 18 / Reckless (2026-04-27, 60+1, hash=64, no EGTB)

| Metric | vs SF | vs Reckless |
|---|---:|---:|
| Score (W-L-D) | 0-46-54 | 0-41-59 |
| Gap | **−210 ±48** | **−151 ±40** |
| Score % | 23.0% | 29.5% |
| Draw rate | 46% | 59% |
| HORIZON % of losses | 93.5% | 61.5% |
| Median cliff ratio | 0.84 | 0.80 |

### 4-config EGTB + Hash + TC sweep (2026-04-27, 200 games per cell)

| Configuration | Overall | vs SF | vs Reckless | Draw % |
|---|---:|---:|---:|---:|
| 60+1, hash=64, no EGTB (baseline) | −179 ±31 | −210 ±48 | −151 ±40 | 52.5% |
| 60+1, hash=64, EGTB on | −186 ±32 | −182 ±45 | −191 ±46 | 51.0% |
| 60+1, hash=512, EGTB on | −151 ±29 | −139 ±40 | −164 ±42 | 58.0% |
| **180+2, hash=512, EGTB on (deployment-config)** | **−119 ±25** | **−139 ±39** | **−100 ±33** | **67.0%** |

Mechanism decomposition surfaced by this 4-config sweep:

- **EGTB alone (hash controlled)**: essentially flat at −7 Elo overall
  (within noise). Earlier "+86 Elo SF closure from EGTB" reading was a TT-size
  artifact, not an EGTB effect.
- **Hash 64→512 (60+1, EGTB on): +35 Elo overall, +43 vs SF, +27 vs Reckless,
  +7% draw rate.** TT pressure in long endgames is a meaningful gap factor.
  Coda's 64MB TT thrashes at endgame piece counts where transposition density
  is high; bumping to 512MB partially fixes it.
- **TC 60+1 → 180+2 (with hash=512+EGTB): +32 Elo overall, +0 vs SF, +64 vs
  Reckless, +9% draw rate.** SF-side gap is TC-saturated by 60+1; Reckless
  closes via more search time (eval refinement).

vs SF gap is **hash/depth-bound** — closes -71 Elo with hash bump, doesn't
close further with TC. vs Reckless gap is **TC/eval-refinement-bound** —
hash bump barely moves it, long TC closes -64 Elo.

### Rivals gauntlet (2026-04-28, 40+0.4, hash=512, EGTB-on, 8 engines, 1400 games)

| Rank | Engine | Elo | Score | Bench NPS | Speed×Coda |
|---:|---|---:|---:|---:|---:|
|  1 | Horsie | +58 ±23 | 58.3% | 1,052,223 | 1.78× |
|  2 | Tarnished | +56 ±24 | 58.0% | 1,019,000 | 1.72× |
|  3 | PZChessBot | +17 ±26 | 52.5% | (n/a) | (n/a) |
|  4 | Seer | +14 ±25 | 52.0% | 1,500,726 | 2.54× |
|  5 | Velvet | −7 ±24 | 49.0% | 1,780,148 | 3.01× |
|  6 | Clarity | −12 ±26 | 48.3% | 1,690,952 | 2.86× |
|  7 | Arasan | −14 ±26 | 48.0% | 838,335 | 1.42× |
|  8 | **Coda** | **−16 ±9** | 47.7% | **591,523** | 1.0× |

Coda is the **slowest in the rivals pool by 1.4-3× margin**, but rank
ordering shows NPS isn't determinative — top 2 strongest (Horsie, Tarnished)
are at moderate NPS, fastest (Velvet, Clarity) sit mid-pack. v9's
threat-input architecture cost ~20% NPS for +110 Elo (good trade); we
approximately match the field despite being slowest.

#### Per-opponent loss-class breakdown

Of 1400 games / 222 losses analysed via `scripts/classify_losses.py`:
**77.9% HORIZON, 15.8% SELF_BLUNDER, 1.8% POSITIONAL, 4.1% UNCLASSIFIED**.

| Opponent | Losses | HORIZON | SELF_BLUNDER | Read |
|---|---:|---:|---:|---|
| Clarity | 25 | **96.0%** | 0% | Pure horizon — purely depth-bound opponent |
| PZChessBot | 34 | 91.2% | 8.8% | Mostly horizon |
| Tarnished | 43 | 86.0% | 9.3% | Mostly horizon |
| Velvet | 23 | 82.6% | 13.0% | Standard mix |
| Arasan | 25 | 76.0% | 16.0% | Mild SELF_BLUNDER bias |
| Horsie | 41 | 70.7% | **19.5%** | High SELF_BLUNDER + strong |
| **Seer** | 31 | 45.2% | **41.9%** | THE outlier |

**Seer is the durable per-opponent outlier.** 13 of 35 total pool
SELF_BLUNDERs (37%) come from Seer alone. Hypothesis: Seer's positional/
quiet style induces our second-best moves at ~4× the rate of any other
opponent, where the more-tactical engines don't construct the same lures.

**Clarity is the cleanest depth-bound opponent**: 96% HORIZON, 0%
SELF_BLUNDER. Every loss to Clarity is a pure search-depth mismatch.
Search-side work targets the Clarity-class loss directly.

**SELF_BLUNDER is bimodal**: pool aggregate 15.8%, but Seer 42% / Horsie
20% / everyone else 0-16%. The "average" hides two distinct subpopulations.

## SF-vs-Reckless mechanism asymmetry

The two gaps close via different mechanisms — name the predicted gap-move
on a merge before validating, then check.

| Merge type | Expected SF-gap move | Expected Reckless-gap move |
|---|---|---|
| New net / training improvement | small close | **larger close** |
| Pruning / ordering / EBF reduction | **larger close** | moderate close |
| Pure NPS / cache residency | small close | small-to-moderate close |
| Endgame depth (LMR_ENDGAME, TB timing) | **larger close** | moderate close |
| Eval-feature addition (new threat input) | small close | **larger close** |

If a merge moves both gaps equally → mixed mechanism (or measurement noise).
If they move oppositely → something is mis-calibrated; investigate before
accepting.

The path to peer-tier strength runs through the **Reckless gap first**.
Reckless → SF on broad-pool is only ~5 Elo, so once we're peer-tier with
Reckless, search work to close the last bit becomes more justified.

## Loss-class taxonomy (gradual / moderate-stepped / sudden)

Of 222 Coda losses, the first crossing of −100cp splits:

| Class | % of losses | Mechanism | Lever |
|---|---:|---|---|
| Very gradual | 50.9% | Many small drops (<−50cp each) accumulating. Static eval consistently slightly worse | Training (factor net, low-LR tail, group-lasso, data composition) |
| **Moderate stepped** | **20.3%** | Single −50 to −100cp drop pushes Coda below threshold. Pruning blind spot — would have refined at higher depth or without one specific gate | **Search guards / extensions / ablation-driven bug discovery — MOST ACTIONABLE** |
| Sudden | 28.4% | Single ≥−100cp drop. Tactical/search blunder at a specific position | Extensions on tactical-density signals, recapture extensions, threat-aware exemptions |

**The moderate-stepped 20.3% (45 concrete positions) is the highest-leverage
search-side target.** Each candidate has FEN + Coda's chosen move + SF's
bestmove + eval delta — ready for ablation.

## Win/loss asymmetry

SF-arbitrated comparison of how Coda's eval crosses ±100cp on 158 wins
vs 222 losses:

| Crossing class | Wins (cross +100) | Losses (cross −100) |
|---|---:|---:|
| Gradual (no move ≥50cp swing) | **92.4%** | 50.9% |
| Moderate (single 50-100cp swing) | 4.4% | **20.3%** |
| Sudden (single ≥100cp swing) | 3.2% | **28.4%** |

| Per-move accuracy | Wins | Losses |
|---|---:|---:|
| SF-accurate (≥−50cp) | 96.4% | 92.4% |
| Blunder | 0.5% | 1.1% |

The +4pp accuracy gap exists at the per-move level — same engine, same TC,
same hardware.

**Reading:** opponents exploit Coda via moderate-step (pruning blind spots)
and sudden (tactical horizon) crossings ~6× more frequently than we exploit
them. The combined moderate+sudden bucket is **48.7% of −100 crossings in
losses, only 7.6% in wins**.

When we win, we accumulate slowly (92% gradual). When we lose, we hemorrhage
via stepped/sudden mistakes. **Same gradual-rate on both sides; the
asymmetry is purely in stepped/sudden exploitation.**

100% of wins reach +100cp at some point (no pure "opponent blundered from
equal" wins); 72% reach +400cp — we convert significant advantages
routinely. The win-side mechanism is healthy; the loss-side mechanism has
the deficit.

## Coda's per-move search is accurate

SF-arbitrated analysis at depth 18 on 11725 Coda moves across 222 losing
games:

- 92.4% of Coda's moves are SF-accurate (≥−50cp)
- Median worst-move-per-game is only −160cp
- 60.4% of losses have NO single move worse than −200cp
- 74% of cliff moves happen when Coda is already losing per SF
- Only 25% of cliffs come from balanced positions (highest-leverage subset)

**Critical conceptual point:** SF's per-move ΔSF score IS itself a depth-N
search result. "Eval drift" doesn't mean static-NNUE drift — it means
**dynamic-eval drift**, where Coda's search resolves a position to a value
that's worse than SF's search of the same position resolves to. The
mechanism is search-side: ordering, pruning, TT, reductions/extensions, depth.

A 60% slow-erosion loss where dynamic eval drifts 0 → −100 → −200 → −400
across 30 moves is NOT one bad NNUE output. At each move, search should
refine the position; Coda's search misses or under-weights the refinement
(often a late-deep-move that LMR reduced, or ordering placed last, or TT
pollution evicted before it could cut). Drift compounds across plies.

This is the same mechanism family as the 25% balanced-cliff subset, just
spread across many moves. Both are search-bound.

## Drilling in: 45 moderate-stepped candidates

`/tmp/pruning_candidates_moderate.jsonl` from
`scripts/find_pruning_candidates.py --mode moderate_step` over the 1400-game
rivals PGN. Each candidate is a position where Coda played a move that lost
−50 to −100cp vs SF's bestmove at depth 18.

### Convergence-depth probe (post-bug-fix, 2026-04-29)

`scripts/probe_convergence_depth.py` over the 45 candidates at depths
[14, 16, 18, 20, 22, 24], cross-referenced with each candidate's PGN-parsed
played-depth.

**Headline:** 41/45 (91%) converge to SF-best within d24. Only 4/45 are
durable eval-blind spots within depth 24.

**Depth-deficit histogram (probe_d − played_d):**

| Bucket | N | % | Mechanism |
|---:|---:|---:|---|
| Confirmed TT-state-bound (≤0 deficit) | 3 | 7% | Clean hash at played_depth recovers SF-best |
| Non-monotonic depth artifact | 2 | 4% | Probe over-counted (deeper search picks blunder anyway) |
| NPS-bound (+1..+2) | 10 | 22% | Small extra search finds it |
| **Ordering / pruning blind spots (+3..+6)** | **18** | **40%** | **DOMINANT bucket** |
| Big depth deficit (+7..+9) | 4 | 9% | Search-bound |
| Eval-bound (NEVER) | 4 | 9% | Depth 24 doesn't help |
| Probe didn't reach (other) | 4 | 9% | Mixed/skipped |

Notable: 3 of 4 NEVER cases are Seer/Arasan/Tarnished — consistent with
Seer being the SELF_BLUNDER outlier in the rivals gauntlet.

**Methodology gap (worth noting):** the original `probe_convergence_depth.py`
breaks on first SF-best match — biased toward shallower convergence depth
than the true ceiling (caught 2 non-monotonic cases). Future probes should
run all depths.

### Per-feature ablation on the +3-6 ply deficit bucket

Per-feature ablation on the 19 candidates with deficit +3-6 at played-depth
(no extra depth, just feature toggles):

| Ablation | matches_sf | rate |
|---|---:|---:|
| baseline | 1/19 | 5% |
| **NO_LMR** | **10/19** | **53%** |
| NO_RFP | 8/19 | 42% |
| NO_FUTILITY / NO_SEE_PRUNE / NO_BAD_NOISY | 7/19 each | 37% |
| NO_NMP / NO_PROBCUT | 6/19 | 32% |
| NO_HIST_PRUNE | 5/19 | 26% |
| NO_LMP | 4/19 | 21% |
| **NO_ALL_PRUNE** | **11/19** | **58%** |

LMR is the largest single-feature recovery rate (53%). NO_ALL_PRUNE adds
only +1 candidate beyond NO_LMR — disabling everything else on top of LMR
buys ~5% extra coverage.

**LMP is curiously weak** at 21% recovery in this bucket — not the lever
for moderate-stepped exploits. De-prioritise LMP carve-outs as a
moderate-stepped fix.

### Pruning recovery is distributed, not LMR-exclusive

Set arithmetic on `/tmp/abl_ordering_pd0.csv` (18 candidates that fail at
baseline):

**Recovery sets per pruning class:**

| Class | Members | Recovery |
|---|---|---:|
| futility | NO_FUTILITY, NO_RFP | 50% |
| see | NO_SEE_PRUNE, NO_BAD_NOISY | 56% |
| reduction | NO_LMR | 50% |
| null-move | NO_NMP, NO_PROBCUT | 44% |
| move-count | NO_LMP, NO_HIST_PRUNE | 39% |

**Class-exclusive sets all empty** — every NO_LMR-recovered candidate is
ALSO recovered by at least one other ablation class.

**Recovery-breadth distribution:**

| Breadth | N | Cases |
|---:|---:|---|
| 5-7 ablations recover | 7 | heavily over-pruned |
| 2-4 ablations recover | 6 | moderately over-pruned |
| **0 ablations recover** | **5** | **NOT pruning-bound** |

The 5 zero-breadth cases are NOT addressable by any single pruning carve-out
at played_depth. They need either:

- Ordering improvements (move SF-best up the list so it gets searched at all)
- More NPS / depth (deeper effective search)
- Better eval (the move IS searched but ranks low)

These 5 are misfiled as "ordering/pruning" — they're actually
ordering-or-NPS-bound at played_depth.

### Strategic implications

1. **Single-feature LMR carve-outs cap at ~50% coverage** of the +3-6 ply
   bucket. Explains why isolated LMR carve-out SPRTs (#865, #866, #867,
   #868) haven't produced strong H1 signals — the mechanism only matches
   half the cases at best.

2. **Multi-feature carve-outs on a shared trigger** have higher leverage.
   A threat-aware approach should loosen NMP, RFP, FUT, LMR simultaneously
   when major-threat creation is detected, not just LMR. Coverage ~70-80%
   of the recoverable bucket.

3. **NPS work pays on 22% of cases** (the +1-2 ply bucket). Pure
   cache-residency / SIMD dispatch / matrix-shrink wins flow through —
   lower share than initially assumed pre-probe.

4. **TT replacement / aging / collision-tagging work has direct signal on
   ~7% of cases** (down from initial 20% headline once methodology was
   audited). Worth targeting separately from ordering.

5. **Pure-eval improvements address only ~9% of cases** (durable
   eval-blind spots within depth 24). Training work continues but isn't
   the leveraged frontier for the moderate-stepped class.

6. **5 of 18 cases (28%) need ordering or NPS, not pruning.** Move-ordering
   improvements (pos² history, threat-aware ordering, capture-history
   calibration) might address these — single-feature pruning carve-outs
   structurally cannot.

7. **62% of cases are addressable by 1-6 ply gain** — combined NPS +
   ordering + pruning frontier. Each independent improvement compounds.

## Strategic priorities (next-50-Elo target)

The next 50 Elo of work targets **closing the rivals gap**, NOT chasing
SF further. Rivals gap is more closable; SF gap saturates at ~120-140 Elo
regardless of further search work.

Three loss-class buckets, three priority orderings:

1. **Moderate-stepped (20.3% of losses, 45 candidates)** — pruning
   carve-outs, ordering improvements, ablation-driven bug discovery.
   Highest-leverage search-side target.
2. **Sudden (28.4% of losses, 63 candidates)** — tactical extensions,
   recapture/SE-style depth purchases, threat-feature-aware gates.
3. **Gradual (50.9% of losses)** — training/eval lever (factor net,
   low-LR tail, group-lasso, training-data composition).

Within search-side work, prefer:

- **Multi-feature carve-outs on shared triggers** > single-feature
  carve-outs (ceiling 70-80% vs ~50% bucket coverage)
- **Mechanism (3) less-bad-pruning + (4) better-ordering** > mechanism
  (1) NPS + (2) more-pruning. Most easy "tighten the margin" wins are
  already banked
- **Threat-aware approaches** that consume signals v9 has natively

De-prioritise:

- Specific anti-SF tactical features — SF's edge is bulk search depth;
  feature work doesn't address that
- Pure NPS micro-optimisations below the TC sigmoid knee — contribute
  ~0 Elo at 10+0.1 ultra-bullet vs SF; at peer-tier marginally more,
  but EBF wins still beat them 2-5×
- Deep-endgame TT bucket density framed around the SF endgame-cliff
  finding — lower priority now that rivals data exists

## Validation cadence

After every search-side merge cluster of ~+10 Elo:

1. **Rerun the rivals gauntlet** (40+0.4, hash=512, EGTB-on, ~1400 games).
   Track Coda's Elo and the gap to Tarnished/Horsie. If the +50 target is
   being met, Tarnished's gap should compress from +81 toward +30-40 over
   the campaign.
2. **Rerun `classify_losses.py`** on the new PGN. The moderate+sudden
   bucket should shrink in the loss subset; the gradual bucket on the loss
   side should stay stable. If moderate+sudden doesn't shrink, the cluster
   wasn't targeting the right mechanism.
3. **Regenerate the 45-candidate convergence probe** (or its successor).
   Check whether the +3-6 ply bucket shrinks — direct test for whether
   the cluster targeted the right mechanism.
4. **Watch for inverse asymmetry**: if a future merge introduces tactical
   aggression that ALSO produces moderate+sudden cliffs in our games (the
   4.4% / 3.2% wins-side numbers grow), that's a regression signal even
   if SPRT-positive at STC.

## Tools & data

**Scripts:**
- `scripts/classify_losses.py` — gradual/moderate/sudden classifier on PGN
- `scripts/find_pruning_candidates.py` — surfaces candidates by mode
  (`--mode moderate_step` / `--mode sudden`)
- `scripts/blunder_ablation.py` — feature-toggle ablation harness (fresh
  process per ablation, `ucinewgame` clears TT/history/ponder/TC state)
- `scripts/probe_convergence_depth.py` — sweep depths until SF-best emerges
  (NB: stops at first match; biased toward shallower convergence)
- `scripts/ablation_at_played_depth.py` — fixed-depth probe with clean state
- `scripts/sf_arbitrate_blunders.py` — SF-arbiter for per-move accuracy
  (uses persistent process + readline pattern; UCI scripts must read
  `bestmove` before sending `quit`)

**Source data:**
- `tough_rivals.pgn` — 1400 games, 8 engines, 40+0.4, hash=512, EGTB-on
- `/tmp/pruning_candidates_moderate.jsonl` — 45 moderate-stepped candidates
- `/tmp/pruning_candidates_sudden.jsonl` — 63 sudden-drop candidates
- `/tmp/convergence_moderate.csv` — convergence-depth probe output
- `/tmp/abl_ordering_pd0.csv` — per-feature ablation × 18 candidates

**Caveats:**
- The morning-of-2026-04-29 ablation/depth-match data was invalidated by
  a UCI quit-before-bestmove bug (fixed in commit `0fff4f6`). The
  convergence probe is the first post-fix data.
- `probe_convergence_depth.py` stops at first SF-best match — biased
  toward shallower convergence depth. Caught 2 non-monotonic cases on
  audit; future probes should run all depths.

## Connections

- CLAUDE.md §Strength Frontier — calibration tables and per-opponent
  breakdown reference this doc
- CLAUDE.md §Improvement Portfolio — thread-selection heuristic
- `experiments.md` — SPRT entries for #865/#866/#867/#868 (LMR carve-outs)
  and downstream multi-feature triple-ext / threat-aware-bundle attempts
- `docs/cross_engine_comparison_2026-04-25.md` — pruning outliers vs
  Reckless that motivate carve-out experiments
- `docs/tt_hash_sensitivity_2026-04-27.md` — TT-pressure analysis behind
  the hash 64→512 finding
