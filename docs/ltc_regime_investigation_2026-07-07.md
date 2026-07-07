# Why Coda's relative strength drops at long TC — consolidated investigation (2026-07-07)

Adam-commissioned. Coda is #7 in the STC RR but loses to near-peers (Viridithas,
Integral) at LTC (120+2). Four questions asked; all four answered here with new
empirical data (deep tree-shape + NPS-vs-depth to d24) plus three cross-engine
source audits. **Companion data:** `tree_shape_d24_2026-07-07.csv`,
`tree_shape_study_2026-07-06.{md,csv}` (Stage 1, to d18). Builds on
`ltc_audit_2026-06-13.md`, `tc_scaling_methodology_2026-06-21.md`,
`search_vs_stockfish_2026-06-17.md`, and memory
`project_stc_ltc_strength_reversal_2026-06-19`.

## TL;DR — one root cause

**It is tree shape, and specifically Coda's LMR base reduction over-thins the
deep tree.** Coda's LMR is a *pure* log-log term with a steep coefficient and
**no additive base** — `ln(depth)·ln(moves)/1.45` (effective coefficient 0.69),
snapped to integer plies. Every strong reference (SF, Berserk, Obsidian) uses
`base_const + ln(d)·ln(m)·~0.47` in fine 1024th granularity; SF adds an explicit
depth-*decaying* all-node inflation. Consequences:
1. Coda's reduction grows ~47% faster per unit of `ln(depth)` → the deep tree is
   progressively over-thinned. Measured: EBF collapses 1.7→1.3 while SF holds a
   flat ~1.55; at nominal d24 **Coda reaches depth with 1.74M nodes vs SF 2.47M
   (+42%) and Reckless 3.21M (+85%)** — Coda's deep plies are hollow
   (nominal-depth inflation). The gap *widens* with depth (24% fewer than SF at
   d18 → 30% fewer at d24).
2. Being a single scalar with no depth structure, one SPSA value cannot fit
   shallow AND deep — and **STC tuning at d20-25 is blind to the deep cost**, so
   it settled at the steep 1.45. This is the mechanism by which the tunables
   "floated to STC-optimal."

The NPS and per-node-cost hypotheses are **ruled out empirically** (see Q2).

## The four questions

### Q1. Is it tree shape? — YES, and it worsens at LTC depths.
Median nodes-to-nominal-depth, 60 heldout quiet middlegame positions, `go depth`,
Hash 256, T=1, clean TT/position:

| D | Coda nodes | SF nodes | Reckless nodes | Coda vs SF |
|---|---|---|---|---|
| 12 | 40,965 | 29,057 | 59,694 | Coda +41% (denser, expected) |
| 18 | 294,179 | 389,408 | 586,951 | Coda −24% (already thinner) |
| 24 | **1,735,983** | **2,473,689** | **3,213,594** | **Coda −30%** |

Shallow (d≤12) Coda is denser than SF (strength per nominal ply — matches the
fixed-depth games where Coda beats SF at d8). But the density **inverts** and by
the LTC regime (d24) Coda's tree is the thinnest of the three by a widening
margin. Its extra nominal depth is cheap-and-hollow. Stage-1.5 ablation
(2026-07-06) isolated the cause: **LMR_C_QUIET is the deep-shape lever "by an
order of magnitude"** (setting it 145→200 = +51% deep nodes); RFP depth was
*exonerated* (fires too rarely to shape the tree).

### Q2. Is there an NPS element? — NO. Not per-node, not depth-scaling.
Two independent checks, both negative:
- **Incremental NPS is flat across depth** for all engines (new d24 data):
  Coda 1.60M(d10)→1.72M(d24), SF 1.89M→1.87M, Reckless 2.21M→2.21M. No engine's
  NPS degrades with depth. Coda is a uniform ~9% slower than SF / ~22% slower
  than Reckless **at every depth** — a level offset from the heavy net, not a
  deep-regime penalty.
- **No O(depth) per-node hotspot** (code audit): per-node work is O(1) in ply.
  The PV copy (the hypothesised culprit — "copying an entire PV at each node") is
  triangular and **alpha-raise-gated**, so it runs only on the ~D principal-path
  nodes, and is byte-identical to Reckless / equivalent to SF's memcpy. Whole-array
  ply-indexed clears are per-*search*, not per-node.

Conclusion: the deep-regime gap is **not** an NPS/throughput story. (Minor
footnote: the inline `pv_table` is ~32 KB in `SearchInfo` and the comment above
`MAX_PLY` is stale — it says "Keeping 64 / 128 regressed STC −13 Elo" but the
value is 128, deliberately raised 2026-05-15 in `41ccbf6`. Reconcile the comment;
not a live lever given flat NPS-vs-depth.)

### Q3. Do top engines have depth-specific optimisers Coda lacks? — YES, four.
All keep the deep EBF flat where Coda's collapses:
1. **Additive LMR base constant** (Berserk `+0.2319`, Obsidian `dBase`, SF
   `+1027/1024`). Coda has none — its curve can't decouple the shallow floor from
   the deep slope.
2. **Depth-decaying all-node reduction inflation** (SF `r += r·272/(256·depth+285)`
   — +12% at d8, +3% at d30). Reduces proportionally *more* shallow, *less* deep.
   Coda applies a flat integer `+1` at all-nodes — the wrong depth profile.
3. **Fine (1024th) reduction granularity** — SF/Reckless accumulate reduction in
   1024ths then floor once; Coda works in integer plies with integer ±1
   adjustments, so it cannot express smooth depth-dependent shaping.
4. **Concave moveCount response** (SF `r -= moveCount·62`; Reckless drops
   moveCount from the base entirely). Coda's `ln(m)` grows unbounded → over-reduces
   mid-list moves at deep nodes with long move lists.

Coda *does* already have `LMR_ROOT_COEF` (a deep-tree rebate keyed on root_depth)
— the right idea, but keyed on root not local depth, linear, and STC-tuned tiny
(0.08 ply/root-depth over 15).

### Q4. Have tunables floated to an STC-optimised state? — YES.
The `--core` sweep (LMR/RFP/NMP/LMP/singular/probcut/aspiration) carries STC
values (#2481 STC core, #2603 STC LMR-corr). Only **futility + quiet-SEE** got a
genuine LTC retune (#2548, 40+0.4) and are low-risk. Most STC-biased, ranked:

| Param | Value | STC-bias | LTC direction | Why |
|---|---|---|---|---|
| **LMR_C_QUIET (+ missing base)** | 145 | HIGH | lower slope + add base | The structural #1 (= Q1/Q3). |
| **ASP_SCORE_DIV** | 33378 | HIGH | ~10-15k (much smaller) | 3.3× SF's; score²-widening dead at LTC's big scores → ~68% re-search. Cheap isolated win. |
| **NMP_DEPTH_DIV_10X** | 63 (R=7.9+d/6.3) | HIGH | smaller div (deeper R) | SF R=7+d/3; Coda nulls ~4 plies shallower at d24. |
| **SE_DEPTH_10X** | 41 (4.1) | MED-HIGH | ~6 | Extends singular 2 plies shallower than SF → nominal-depth inflation. |
| **LMP_DEPTH / SEE_CAP_DEPTH** | 8 / 8 | MED | 5 / 4 | The validated codabot deep profile (+11.7@180+2) moved both down. |

Independent empirical ground truth: codabot's 9-param **deep-regime UCI profile**
(+11.7 Elo H1 @ 180+2, −3.3 STC) is a measured list of STC-biased params, and its
directions all say **prune/reduce less at depth**.

## Recommended plan (two tracks)

**Track B — the structural LMR reform (the real Elo; LMR-domain).** Give the LMR
table the degrees of freedom the reference engines have: **fractional/centi-ply
reduction + an additive `LMR_BASE` constant**, then LTC-retune the joint cluster
`{LMR_BASE, LMR_C_QUIET, LMR_C_CAP}` (+ optionally an SF-style depth-decay all-node
term) against the empirical target **tail EBF → ~1.55, shallow density → SF-ish**
(the 10-min `scripts/tree_shape_study.py` + Stage-1.5 setoption sweep is the
mechanism metric). Because it thins shallow while un-thinning deep, expect
**STC-neutral / LTC-positive → mergeable to main**, not a UCI override. This
overlaps Hercules's active LMR work (T1.1/T1.2) — hand off or co-own, don't fork.

**Track A — cheap LTC value/gate tune (independent, non-LMR; can start now).**
A focused `40+0.4 Hash=256` SPSA over the params whose STC↔LTC optima genuinely
differ and are pure values/gates already exposed: **ASP_SCORE_DIV + ASP_DELTA**
(highest-confidence cheap win — the window is demonstrably dead at LTC scores),
**NMP_DEPTH_DIV / NMP_BASE_R / NMP_VERIFY_DEPTH**, **SE_DEPTH**, and the
**LMP_DEPTH / SEE_CAP_DEPTH** gates (seed these two at the deep-profile values 5/4
and let SPSA pull back). Run STC and LTC from the same seed and read the
divergence — params that land far apart are candidates for the existing
`root_depth` parameterisation machinery (`LMR_ROOT_*`, `RFP_ROOT_*`,
`PROBCUT_ROOT_*`). **Do NOT include** futility/RFP/LMP_BASE/probcut (already
LTC-tuned or exonerated) or the TM_* family (its own ponder-validated campaign).

**Also on the LTC list but its own track:** the TM base per-move fraction is
*flat in time* (constant ~2.5% of clock at both 10+0.1 and 40+0.4); SF scales it
with `log10(total_time)`. Coda systematically under-thinks at LTC (finishes with
more clock than opponents every game — confirmed input metric). That is the
single biggest non-search LTC lever and belongs in the TM campaign, not a core
tune.

## What is NOT the cause (ruled out)
- NPS / per-node cost / PV-copy scaling (Q2 — flat NPS-vs-depth, O(1) per node).
- TT overload (2026-06-19: 8× hash did not close the LTC gap).
- Draw-excess (tc_scaling: the LTC deficit is decisive-game quality at depth, not
  extra draws).
- RFP depth (Stage-1.5 ablation exonerated it).
- Futility / quiet-SEE values (already LTC-retuned #2548).
