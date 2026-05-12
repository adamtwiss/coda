# Strength Frontier — Where the Elo Gap Lives

Living reference doc. Working hypothesis derived from a 100-game SF H2H,
Atlas's loss-pattern analysis, and the 1400-game rivals gauntlet. Update
incrementally as new measurements come in; re-author from scratch only
on a major rethinking.

**Cross-refs**: full per-loss-class taxonomy and the 45-moderate-stepped
× 12-ablation deep-dive live in `loss_analysis_2026-04-28.md` (dated
snapshot — don't edit, re-run if rebuilding).

## Calibration

- **Deployment anchor (180+2, hash=512, EGTB-on)**: **−119 ±25** Elo
  combined vs SF + Reckless. Cite this for strength claims. The 60+1
  numbers below are diagnostic instruments for mechanism decomposition,
  not strength reads.
- **60+1 H2H vs SF**: −210 ±48. Gap is hash/depth-bound — closes −71
  Elo with hash 64→512, doesn't close further with TC. 93.5% HORIZON.
- **60+1 H2H vs Reckless**: −151 ±40. Gap is TC/eval-refinement-bound
  — hash bump barely moves it, long TC closes −64 Elo. 61.5% HORIZON.
- **Rivals gauntlet (40+0.4, hash=512, EGTB-on, 1400 games, 8 engines)**:
  Coda **−16 ±9** Elo, mid-pack. Top of pool Horsie/Tarnished +56-58;
  bottom Velvet/Clarity/Arasan −7 to −14. **The next 50 Elo target =
  closing this gap**, not the SF gap. Rivals gap closes linearly with
  merge work; SF gap saturates at ~120-140 Elo regardless of further
  search work.
- **10+0.1 ultra-bullet 45-engine RR**: Coda gap to SF ~270-302 Elo,
  rank 21-23/45. STC SPRT over-measures our deficit vs both SF and
  rivals; the TC sigmoid applies to both. Validate rivals-tier strength
  via 40+0.4 H2H, not SPRT.
- **CCRL inference**: ~3520-3620 band (top-30 territory). See
  `memory/project_ultra_bullet_vs_ccrl_calibration.md`.

**SF-vs-Reckless gap mechanism asymmetry**: vs SF closes via
search/depth; vs Reckless closes via eval refinement. Path to peer-tier
runs through Reckless gap first.

**HORIZON is an outcome class with 4 mechanisms**: (1) faster NPS, (2)
more pruning, (3) less bad pruning, (4) better ordering. Coda's specific
distribution favours (3) and (4) — Reckless outlier-pruning pattern
shows we both over- and under-prune on different thresholds; the
leverage is in specific carve-outs that prevent mis-pruning critical
moves, plus ordering improvements. Mechanism (1) is NPS-discounted at
long TC; mechanism (2) "tighten the margin" wins are mostly already
banked.

**+3-6 ply ordering/pruning bucket = highest-leverage frontier** (40%
of moderate-stepped candidates). Single-feature LMR carve-outs cap at
~50% bucket coverage; multi-feature carve-outs on shared triggers
(e.g. threat-aware loosening of NMP+RFP+FUT+LMR together) go higher.

**SF gap as a function of TC**:
- 10+0.1 (ultra bullet): ~302 Elo
- 60+1 (bullet): ~160 Elo
- 8× (480+8, past the knee): ~35 Elo
- 16× (960+16): ~0 Elo (parity)

## Self-play NPS-Elo conversion (TC and threading sweeps)

Self-play sweeps measure how much our own NPS/depth buys us in our own
regime — directly relevant for sizing up NPS work, threading work, and
SPRT-result-extrapolation. Both sweeps run on Hercules-class hardware,
hash 1024, EGTB on, base TC 10+0.1.

**TC sweep (single thread, ~480 games per point):**

| TC factor | TC | Self-play Elo | Δ vs 1× | Δ vs prev |
|---|---|---:|---:|---:|
| 0.5× | 5+0.05  | **−167** | −194 | — |
| 1× | 10+0.1 | +27 | (anchor) | +194 |
| 2× | 20+0.2 | **+137** | +110 | +110 |

Asymmetric / concave shape — *halving TC hurts more than doubling
helps* (−194 vs +110). Diminishing returns set in fast: each subsequent
doubling buys less. Implication for NPS work: a 2× NPS win is worth
~+110 Elo in our own regime, but stacking 2× NPS wins isn't a linear
path to +220.

**Threading sweep (single TC 10+0.1, 200 games per point):**

| Threads | Self-play Elo (vs 1T) | Δ vs prev doubling |
|---|---:|---:|
| 1 | (anchor) | — |
| 2 | **+37** | +37 |
| 4 | **+68** | +31 |

Lazy SMP scales 2T = +37, 4T = +68 (~85% linear-doubling efficiency on
the second step).

**How to read self-play numbers vs SPRT bounds.** Self-play sweeps
above are absolute Elo at the displayed TC; an SPRT gain *at SPRT TC*
will convert at roughly the same rate as the relevant lever shows
above. A 5% NPS win at 10+0.1 ≈ +5 Elo by linear-interpolating the 2×
doubling; halve that for "NPS-only with no behaviour change" because
SPSA values weren't retuned. Don't double-count — the SF/Rivals
strength gap is measured separately.

**Self-play vs vs-SF dynamics differ.** The previous SF-handicap
sigmoid showed an ultra-flat 1×→2× zone (SF's TM kept the gap stable
until 4×). That dynamic is specific to fighting much stronger
opposition and is not how our own NPS gains compound for SPRT. Use the
self-play tables above when sizing NPS/threading work; consult the
SF-gap framing in the loss-analysis doc when reasoning about
path-to-parity.

## Path to closing the gap (pragmatic, 2026 horizon)

To reach the knee (~8× effective depth-gain) we need compounding across
all levers. Realistic per-lever contribution budget:

| Lever | Plausible gain | Path |
|---|---:|---|
| Net architecture / training upgrades | ~20-30 Elo | Output-bucket layouts, FT-shrink, factor-arch refinements, training-recipe sweeps |
| Cache residency + SIMD dispatch | +0-5 Elo | Most banked; remaining is small. SIMD coverage is in for everything. |
| Pruning equilibrium retunes | +5-15 Elo | Force-more-pruning style branches with full-sweep retune |
| Further eval refinement | +10-20 Elo | Training-recipe iteration on top of new arch |
| Move-ordering improvements | +5-15 Elo | EBF-reducing work; compounds with above |
| **Stacked total** | **~40-85 Elo** | Meaningful jump — top-20 CCRL range |

Per-row gains are rough envelopes, not banked. In-flight experiment
status lives in `experiments.md`, not here.

**Parity (16×-equivalent)** requires dramatic EBF reduction (log(EBF)
halved, 1.8 → 1.35) which in turn requires multi-year investment:
richer training pipeline, more net iteration, possibly novel search
innovations. Don't expect parity in 2026.

## What the losses look like

Loss-pattern analysis (Atlas, 27 SF losses): median max single-ply
eval drop 4.53 cp; cliffs happen at mid-game → endgame transitions
(median ply 83); Coda reaches median depth 15 in losing games vs
SF's 25-35. Coda's eval is accurate at the depth it reaches — the
refuting combination lives at plies SF sees and we don't. **Tactical
blindness from search-depth deficit, not positional mis-eval.**

## What closes the gap

**Effective depth** is the target. Effective depth ≈ log(NPS × time)
/ log(EBF), so depth = f(raw NPS, pruning efficiency, eval quality).

Decomposition of the ~160 Elo SF gap, rough budget:

| Lever | Approx share of gap | Status |
|---|---:|---|
| Raw NPS deficit (Coda ~2× slower than Reckless) | **~100 Elo** (~65%) | Some portion is deliberately bought — x-ray threat features cost ~20% NPS but paid +110 Elo. Most cache-residency / SIMD work now banked; remaining headroom small. |
| Pruning efficiency (Coda under-prunes vs Reckless on several params) | **~30–50 Elo** | "Force more pruning + retune" branches are the lever |
| Eval quality / tactical sharpness | **~20–30 Elo** | Factor-net-quality, training recipe improvements |

**Reckless vs Coda pruning outliers — durable pattern.** Coda has
historically diverged from Reckless on at least five thresholds
(futility margin, SEE quiet prune magnitude, RFP/LMP/BNFP depth caps).
Each direct-port attempt has surfaced the same lesson: **raw-Reckless
values aren't portable because our search context differs** (lmr_d vs
raw-depth formulas, different history scaling). The outliers identify
real miscalibration; the fix is SPSA-retune-on-branch, not value
import. Holistic "force-more-pruning" biased-start full-sweeps have
repeatedly found new equilibria worth +5-15 Elo where direct ports
H0'd. See `experiments.md` for resolved specifics;
`cross_engine_comparison_2026-04-25.md` for the live queue.

## Priors this updates

- **Pruning values matter as much per-param as NPS.** Five clear
  outliers where Coda prunes less than Reckless; each retune-on-branch
  around a tightened threshold ≈ +2-5 Elo, collectively the same
  ballpark as NPS wins.
- **Coda deliberately traded NPS for eval quality** via v9
  threat-inputs. That trade only pays off if better eval feeds better
  pruning decisions. Net Elo > net NPS.
- **"Force more pruning"-style branches** (widen an under-pruning
  threshold, let SPSA find new equilibrium) have worked before and are
  under-built-on.
- **Experiments that buy depth at hot plies** (extensions on
  tactical-density signals, cliff-risk heuristics) are a candidate
  class loss-pattern analysis surfaced. W2-pattern work (signal ×
  pruning/extension) already has high hit rate.

## De-prioritised

- **Eval post-processing tweaks** (optimism calibration, fortress
  caps, shuffle detectors) are low SF-gap leverage. Cheap is fine;
  don't let them displace pruning.
- **Factor net alone** doesn't close the SF gap — improves eval
  quality with indirect leverage but doesn't address depth deficit.
  Still worth doing for Rivals-tier strength + cleaner pruning
  calibration; don't expect it to crack the cliff-miss class.

## Workflow + recalibration

When sizing up a new experiment, ask: (1) does it increase effective
depth? (2) is Coda's pruning value here an outlier vs top-engine
consensus? (3) would it show up in a 100-game SF bullet H2H? If all
three are "no", expected Elo-per-effort is probably low.

Re-run the 100-game SF H2H every ~2 weeks or after any merge cluster
worth ~+20 Elo. Gap should narrow visibly; if it doesn't, on-paper
Elo is overstated.
