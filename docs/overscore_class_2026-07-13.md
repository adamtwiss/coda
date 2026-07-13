# The Overscore Class: where the SF gap actually lives (2026-07-13)

**TL;DR.** Two days of decomposition and forensics (2026-07-10 → 07-13)
localize Coda's remaining gap to SF in a specific, now-harvested class of
positions: **quiet, slow-burn middlegame positions that our eval overscores
by ~+30–60cp in ways deeper search does not self-correct — and game
trajectories concentrate on exactly this class, because our search
maximizes our own eval (it walks toward its own optimism) while the
opponent's search cooperates (its eval prices the same positions correctly,
in its favor).** This confirms Adam's long-standing hypothesis, with signs
and magnitudes. Everything else — NPS, time management, pruning policy,
tree shape, aspiration windows, cp-margin calibration — measured ≈ 0 across
~15 falsification tests. The deliverables are a certified overscore corpus,
a net-report metric proposal, and a set of minutes-scale instruments.

Companion doc: `docs/search_gap_decomposition_2026-07-11.md` (the frame
decomposition and input-metrics program this work executed). Experiment
log: `experiments.md` 2026-07-10..13 entries.

---

## 1. What the gap is NOT (the falsification ledger)

All measured, most in two frames (self-play SPRT + cross-engine fixed-150k
differential gauge); full details in experiments.md:

| hypothesis | test(s) | verdict |
|---|---|---|
| NPS | idle/contended/in-situ measurement | ≈ parity on deployment HW (Zen-1 idle artifact only) |
| Active TM | st=0.28 vs TC frames; 9 interventions incl. full SF spend-profile match | ≈ 0 ±20; all interventions ≤ 0 |
| Aspiration width (σ-rescale) | #2703 | H0 |
| cp-margin family ×1.5 (σ-rescale) | #2707 | H0 (−9.7) |
| doDeeper form | treestats gate | mechanism-dead (5-min verdict) |
| Deep-node LMR tilt | #2718 STC, #2723 LTC, tune #2717 | true zero both TCs; tune converged |
| Named pruning gates discard critical lines | per-position tracer, 196 provably-fixable positions | **1.5%** (3/196) — exonerated |
| Tree-shape imitation generally | treestats comparison + P-A/P-B | shape differs (SF: 2.4× more d8+ mass, 5× more LMR re-verification) but nudging toward it buys nothing — shape is a consequence, not a cause |

Methodology finding worth keeping: **imitating a stronger engine's
observable behavior (TM spend shape, tree shape) fails systematically** —
the observables are downstream of their eval-search fit. This is now
measured twice (TM campaign, search campaign).

## 2. The forensic chain (how the class was found)

1. **Lag-ratio metric** (`scripts/harvest_horizon.py`): in sustained
   (≥2-own-move, ≥40cp) eval-divergence episodes from our own fixed-150k
   games, Coda converges to the opponent's earlier assessment vs the
   reverse at **6.8 : 1** against SF17/18 (SF17 seat vs SF18: 1.6 —
   their generation gap is NOT lag-shaped; ours is).
2. **Convergence-at-budget** (`scripts/eval_convergence.py`): on 1019
   episode-start positions (all Coda-to-move), Coda's distance to deep
   truth is FLAT in nodes; on a 300-position neutral control from the same
   games it converges normally. Node-flatness is the knowledge signature.
3. **Forced-line discrimination** (`scripts/forced_line.py`, full set):
   43% SNAP (Coda's own deep search, once walked down the critical line,
   agrees with truth — knowledge is in the net, statically invisible),
   48% STUBBORN, 9% partial.
4. **Per-position gate tracing** (CODA_TRACE_LINE tracer, branch
   `snap-forensics`): in 84% of SNAP positions the search **already
   examines every ply of the critical line** — the failure is depth carried
   along the line, not the line being cut. Lines are ~70% quiet moves at
   every ply (129/196 mostly-quiet) — no non-eval ordering signal can
   select them.
5. **Signed error** (the confirmation of the overscore hypothesis):

| set (Coda@150k vs V*) | signed median | overscore >30cp | underscore <−30cp |
|---|---|---|---|
| neutral control | **0** | 21% | 14% |
| episode-starts (all) | **+30** | 50% | 21% |
| SNAP subset | **+58** | **74%** | 26% |
| STUBBORN subset | +57 | 66% | 34% |

   On neutral positions, low-node optimism (+6 med at 15k) washes out by
   150k — search self-corrects. On the steered-into class it persists to
   500k. **The error is directional (optimism), class-specific, and
   depth-immune.**

## 3. Bias controls (what the magnitudes survive)

- **Selection bias**: membership is self-refutation-based, not
  SF-disagreement-based — (a) in-game: Coda's own later evals converged to
  the opponent's earlier number; (b) offline, for SNAP: Coda's own deep
  search endorses the lower value. The SNAP overscore verdict never
  consults SF.
- **Family bias**: sized by the neutral control — ~⅔ of raw |error| vs the
  SF18@4M oracle is generic cross-family/scale offset (Coda-neutral floor
  ~31cp mean / 14 med at 500k). All class claims are quoted as
  blind-minus-neutral EXCESS (~+16 mean / ~+31 med) or on the
  self-refuting subset.
- **Remaining caveat (open)**: class-correlated oracle lean — SF-favoring
  V* specifically on lag positions — is not yet refereed. Named next step:
  LC0 spot-check on a 30–50 position sample (lc0 in the local engine
  pool). Until then, treat SNAP-subset numbers as certified and
  STUBBORN-subset magnitudes as provisional.
- Eval-scale note: Coda's cp spread is ~2× SF's displayed scale
  (calibration conventions differ); within-Coda comparisons are unaffected.

## 4. Mechanism statement

Search steers games toward eval disagreement (that is what maximizing your
own eval against an opponent's does). Where OUR eval is the optimistic one,
we voluntarily walk in; SF's eval prices the same positions correctly, so
its search cooperates. The positions are quiet pressure-builds (~70% quiet
critical lines) — immobile pieces, pawn weaknesses, growing attacks —
whose refutation sits beyond any depth our budget carries along the line,
while SF's threat-informed eval prices them statically. Accumulated ~0.6–2.5
episodes/game at ~+30–60cp each, this is consistent with (not proven to
equal) the measured −33 mid-band gap vs SF17 and the 6.8:1 lag asymmetry.

This coexists with our eval's Spearman/Pearson parity vs LC0 on broad
samples and richer threat inputs: aggregate parity on average positions
hides a directional deficit on the rare, game-deciding class — precisely
because games are steered INTO the holes, weighting them far above their
base rate. Per Adam's prior finding, SF has a comparable-size (different)
blindspot inventory vs an LC0 oracle; the in-game asymmetry is about whose
search weaponizes whose holes.

## 5. Deliverables and next steps

**Data (testdata/horizon/ + scratchpad TSVs to be promoted on next pass):**
1019 episode-starts (fen, V*, signed errors at 4 budgets, SNAP/STUBBORN
tag, oracle line, source game), 1779 strategic-corpus positions, contested
set. All Coda-to-move.

**Net-report metric (proposed, addresses the historical selection-bias
objection):** mean signed error + %overscore>30cp at a fixed small budget
on the SELF-REFUTED corpus (SNAP subset until the LC0 referee clears the
rest), held-out split, refreshed per net generation. Converges with the
in-progress `eval_compare_nets.py` (net-vs-net metric on oracle EPDs).

**Open work, in order:**
1. **Feature attribution**: correlate signed error with computable
   position features (king-ring pressure, mobility, pawn structure,
   our-attackers vs their-slow-threats) → is a specific feature family
   (e.g. threats) running hot? Names the pattern for training work.
2. **LC0 referee** on the oracle (30–50 sample) → certifies STUBBORN
   magnitudes and the net-report metric.
3. **Scale path to the training corpus**: 3k seeds cannot move a 100B+
   corpus as rows; use them as (a) the measurement instrument above and
   (b) QUERIES — mine T80 for feature-similar positions and seed targeted
   datagen playouts (June stamped-corpus machinery is the precedent) to
   produce class-mass at 0.1–1% corpus scale.
4. Search-side: formally closed for this class (gates 1.5%, shape probes
   zero). Reopen only if feature attribution surfaces a signal search
   could cheaply exploit (e.g. a computable class-marker usable for
   targeted extension — none known today).

**Instruments (all committed):** lag ratio, convergence-at-budget +
neutral control, eval↔search consistency, treestats + instrumented-SF
build, forced-line discriminator, CODA_TRACE_LINE tracer, fixed-150k
differential gauge. Falsification cycle time: minutes-to-hours; ~15
plausible theories retired in three days.
