# Tree-Shape Study, Stage 1 — Coda vs SF vs Reckless (2026-07-06)

Empirical nodes-per-depth / EBF / seldepth comparison. 150 shared positions
(heldout quiet middlegame set), one `go depth 18` per position per engine,
Hash 256, clean TT per position, T=1. Median over positions. Harness:
`scripts/tree_shape_study.py`; raw data: `tree_shape_study_2026-07-06.csv`.
Node counts exact (needed the 6c0fc23 info-line fix; SF/Reckless already
exact). Prompted by Atlas's measure-first plan; companion to the formula-
level LMR analysis in the 2026-07-05 SF search audit (agent F2).

## Median nodes-to-depth

| D | Coda | SF | Reckless | | D | Coda | SF | Reckless |
|---|------|----|----|-|---|------|----|----|
| 8 | 5,634 | 4,311 | 7,034 | | 14 | 89,221 | 81,076 | 103,589 |
| 10 | 17,102 | 10,350 | 18,796 | | 16 | 183,526 | 183,648 | 223,683 |
| 12 | 40,668 | 33,143 | 47,040 | | 18 | **335,686** | **452,136** | 431,833 |

EBF tail (d15→18): Coda 1.44, 1.43, 1.38, **1.32** · SF 1.57, 1.44, 1.51,
**1.63** · Reckless 1.49, 1.45, 1.36, 1.42.
Seldepth: Coda consistently deepest at low D (d10: 21 vs SF 15 / RK 16).

## Findings

1. **Shallow (d≤12): the expected ordering holds** — Reckless densest,
   Coda +25-65% denser than SF. Matches the fixed-depth games (Coda +110
   vs SF at d8; Reckless beats Coda). Dense shallow plies = strength per
   nominal depth.
2. **THE SURPRISE — Coda's density INVERTS with depth**: EBF decays
   1.73→1.32 while SF holds a remarkably flat ~1.5-1.6. We converge with
   SF at d16 and by d18 our tree is the THINNEST of the three (26% fewer
   nodes than SF). Our deep nominal plies are cheap-and-hollow — nominal-
   depth inflation — exactly what the audit's formula analysis predicted
   ("under-reduces deep-early moves, over-reduces deep-late ones": the
   steep C_QUIET=1.5 log-slope + flat additive penalties compound at
   depth).
3. **The quality gap is visible per node**: at d12 SF spends 19% FEWER
   nodes than us and wins the fixed-depth games (−28). Their nodes are
   simply better spent (ordering/selectivity) — consistent with the
   fixed-nodes −51..−79.
4. **Seldepth: Coda's selective spikes run deepest at shallow D**
   (extensions/QS chase long lines early). Part of the shallow-density
   story; possibly extension over-budget at low depth.

## Implication for the regime track (revised target)

The naive reading of "SF prunes more" is WRONG as a global statement. The
measured target shape is REDISTRIBUTION, not more thinning:
- **Thin the shallow/middle tree** (d6-12, where we're 25-65% denser than
  SF): LMR base offset (+1027-class), the shallow-biased gates.
- **STOP over-thinning the deep tree** (d15+, where our EBF collapses to
  1.32 vs SF's sustained ~1.55): moveCount linear rebate (−62/1024·mc),
  depth-decaying (not flat) all-node penalties, spine protection.
SF's flat EBF ≈ 1.5-1.6 across d10-18 is the empirical target shape for
the fractional-LMR cluster SPSA (base + slope + C_QUIET jointly). This is
also why single ported gates keep failing: they shift density at one end
without the compensating shape change at the other.

## Caveats + next

- Nominal depth ≠ comparable quality across engines; these are cost
  curves, not value curves. Position set = quiet middlegame only.
- Stage 2 (needs instrumented SF/Reckless builds, restore discipline):
  first-move-cutoff rate + per-gate prune volumes by depth — the ordering-
  quality comparison proper.
- Re-run this study after regime-track changes land (cheap: ~10 min).

## Stage 1.5 — per-gate ablation (setoption sweep, 80 pos, depth 17)

Same harness, Coda-only, suspects neutralized via UCI setoption:

| Config | nodes @ d17 | tail EBF (d15-17) | verdict |
|---|---|---|---|
| base | 259,800 | 1.25-1.39 | — |
| RFP_DEPTH=8 | 265,953 (+2%) | 1.36-1.46 | RFP depth EXONERATED (Adam's knee hypothesis tested: deep RFP fires too rarely to shape the tree) |
| RFP_DEEP_KNEE=17 | 281,957 (+8%) | 1.35-1.49 | knee terms near-neutral |
| LMR_C_QUIET=200 | 393,231 (+51%) | 1.40-1.55 | **THE deep-shape lever, by an order of magnitude** |

The EBF-tail collapse is LMR's deep-reduction slope. But C_QUIET alone is
the wrong fix: softening it also fattens the SHALLOW tree (+26% at d8)
where we are already 30% denser than SF. Confirms audit F2's 2-DOF
prescription: base offset (thin shallow/early) + flatter deep slope /
moveCount rebate (preserve deep) — the fractional-LMR regime cluster, now
with a measured target (tail EBF -> ~1.5, shallow density -> SF-ish) and a
10-minute verification instrument (this sweep).

## Stage 3 addendum (2026-07-06 evening) — the correction-terms arc, measured

Same 150 fens, `go depth 18`, Hash 256. Three states of the LMR
correction-terms branch vs the Stage-1 main baseline:

| d18 metric | main | enabler+battery+cutoff (full Reckless consts) | + tune #2603 (STC) |
|---|---|---|---|
| EBF d17 | 1.38 | 1.48 | 1.44 |
| EBF d18 | 1.32 | 1.43 | 1.35 |
| nodes d18 | 335,686 | 358,638 | 318,410 |

Arc: the full-strength terms lifted the deep tail to Reckless's shape
(1.43) but lost Elo at STC (#2594/#2596 H0 — magnitude wrong, place
right). The STC cluster tune (#2603) recovered STC to ~flat (#2606) by
**giving back most of the tail lift** — an STC objective cannot see the
deep regime, so SPSA spent the tail on shallow nodes (tree now ~5-11%
thinner than main overall). The branch under LTC SPRT therefore carries
only ~1/4 of the intended redistribution. Pre-registered: LTC flat →
re-tune the cluster AT LTC (warm-start from #2603) before any closure;
only LTC-clearly-negative closes the thread. Lesson for the 2-DOF
follow-up: deep-slope parameters must be tuned with an LTC (or mixed-TC)
objective from the start — STC tuning structurally deletes the tail.
