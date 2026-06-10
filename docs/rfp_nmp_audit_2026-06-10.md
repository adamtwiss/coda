# RFP/NMP Investigation: false-positive audit + structural findings (2026-06-10)

Why Coda's NMP measures as "almost non-additive" (NO_NMP ablation: only
+5.6% nodes vs NO_RFP +152%), what the RFP_AUDIT instrument found, and
the experiments that the arc produced. Companion to the 2026-06-06
bench-stats dig in experiments.md ("NMP weakest pruner").

## TL;DR

1. NMP's classical role is structurally absent in Coda — partly the
   modern-portfolio effect (NNUE-grade RFP + LMR demoted NMP everywhere),
   partly Coda-specific structure: **the verify gate (eff. depth 8) sat
   below the min-depth gate (eff. depth 8), so 100% of NMP cutoffs paid a
   verification re-search. NMP never had a cheap cutoff.**
2. The RFP_AUDIT instrument (env `RFP_AUDIT=1`, in tree since the
   rfp-first branch) shows **~40% of RFP cutoffs are NOT confirmable by a
   null-move check, jumping to ~60% at depth ≥ 8** — RFP's takeover of
   NMP's habitat is not a same-nodes takeover.
3. Coda evolved a **static-pruning-dominant equilibrium** (RFP linear
   margins to depth 17) that is STC-optimal but measurably leaks at LTC
   (#1874: RFP_DEPTH 18→8 = **+3.5 at LTC**). Both banked LTC carve-outs
   move Coda toward the consensus shape, invisible at STC.

## The fire-rate / habitat picture

- Bench tree-shape (depth 12, prod net): RFP 258.6 cutoffs/Kn vs NMP
  1.6/Kn (0.32% of nodes attempt NMP; 42% of attempts cut).
- Disabling LMR raises NMP fire counts ~5×: **LMR compresses the depth
  distribution** (late-move subtrees arrive 3-5 plies shallower), starving
  every depth-gated feature (NMP d≥8, SE, probcut). The pruning portfolio
  is hierarchical: LMR is the first-order tree-shaper; depth-gated
  features are second-order residents of what it leaves standing.
- Single-feature NO_XXX ablations therefore systematically understate
  depth-gated features — they measure marginal value within LMR's
  compressed tree, not mechanism quality.

## Cross-engine comparison (SF / Reckless / Obsidian / Alexandria / Stormphrax)

Coda's NMP at the time of the audit (post tune-#1872 values):

| Parameter | Coda | Consensus | Verdict |
|---|---|---|---|
| min-depth gate | **8** | none (d≥1); Sx 4 | STRONG outlier |
| verify depth | **8** (=100% of cutoffs verified) | 14-16 (Obsidian: none) | STRONG outlier |
| BASE_R | 8.0 | 4-7 | high |
| R depth divisor | 7.8 | 3-5 | flat-R outlier |
| extra gates (king-zone / threat-count / undefended) | 3 | none | Coda-unique (each banked small H1 when added) |
| cut_node gate | yes | SF/Reckless/Obsidian yes; Alex/Sx no | consensus-ok |

Also relevant: Coda RFP runs to depth ≤ 17 with LINEAR margins
(38-42 × d); references use ~d²-growth margins that fade RFP out by
depth ~8-12, leaving deep eval≥beta nodes to NMP. Coda RFP at depth 10
demands ~380-420cp; Reckless ~910cp. **Coda statically cuts deep nodes
at roughly half the consensus confidence margin.**

Known-failed direct ports (do NOT retry naively — slack-shape applies):
LMP_BASE→3 (-7.1 H0 at STC; optimum concave at ~5-8 STC, but see LTC
below), SE_DEPTH 4→6+ttPv (#1087 H0), NMP consensus R-shape standalone
(#1904, below).

## RFP_AUDIT instrument

`RFP_AUDIT=1 ./coda bench N` — at every RFP cutoff, runs a null-move
verification using the live NMP R formula (sans post-capture +1), counts
rejections (null < beta) per remaining-depth bucket, then returns the
RFP cutoff regardless (behavior-preserving). Nested audits suppressed
via `SearchInfo::rfp_audit_active`; skipped for pawn-only STM,
consecutive null, mate-magnitude beta. Per-depth table prints after the
Tree Shape block.

### Results (prod net E4B66CE4, tune-#1872 trunk)

bench 12: 707,887 audited, 42.28% rejected. bench 16: 3,131,675
audited, 39.02% rejected.

| depth | FP% @ b12 | FP% @ b16 | audited (b16) |
|------:|------:|------:|------:|
| 0 | 29.0% | 27.0% | 515,167 |
| 1 | 46.5% | 42.2% | 1,497,820 |
| 2 | 43.9% | 43.0% | 633,704 |
| 3 | 40.0% | 39.2% | 224,238 |
| 4 | 37.3% | 35.0% | 130,910 |
| 5 | 37.3% | 33.6% | 68,769 |
| 6 | 34.8% | 31.3% | 35,253 |
| 7 | 33.9% | 29.0% | 18,023 |
| **8** | **60.8%** | **60.9%** | 4,216 |
| 9 | 58.8% | 60.0% | 1,945 |
| 10 | 46.8% | 58.8% | 954 |
| 11 | 62.5% | 56.4% | 440 |
| 12 | — | 49.1% | 159 |
| 13 | — | 50.0% | 70 |
| TOTAL | 42.3% | 39.0% | 3,131,675 |

### How to read this (caveats are load-bearing)

- **"Rejected" is an UPPER BOUND on true false positives**, not a
  measurement of them. The null verifier hands the opponent a free tempo
  (conservative by construction), and Coda's huge R clamps every
  verification to effective null-depth 1 + QS. Rejected = "not
  dynamically confirmable", i.e. the set where static and dynamic
  pruning DISAGREE — the right metric for the takeover question, not a
  wrongness rate. True-FP measurement would need a no-handicap re-search
  ground truth.
- **Audit perturbs the measured tree** (verification subtrees pollute
  TT/history; node count roughly doubles, futility rate shifts). Treat
  rates as directional with ~±5pp slack.
- The d1→d7 decline (46→29%) tracks the linear margin growing with
  depth (more confident cuts). The **d8 discontinuity (29→61%)** is a
  population shift, explained next.

### The d8 discontinuity: RFP overrode failed NMP

On the pre-reorder trunk, NMP ran BEFORE RFP. At depth ≥ 8, when NMP's
null search failed (< beta — a dynamic refutation, "opponent has a
threat"), control fell through to RFP, which often statically cut the
same node anyway — **discarding the warning the engine just paid a null
search to compute**. Those nodes are guaranteed audit-rejections, and
they (plus NMP-gate-blocked nodes) dominate the d≥8 RFP population. No
reference engine has this path (all run static-prune before null-move).
Tiny at STC (d≥8 is ~0.3% of RFP volume at bench depth) but the region
grows superlinearly toward deployment depth (~50 on lichess hardware).

## Experiment ledger for this arc

| Test | Change | Result | Reading |
|---|---|---|---|
| #1874 | RFP_DEPTH 18→8 via UCI @ LTC | **+3.5** | static cutting at d9-17 is net-negative at LTC — the FP-cost mechanism, measured |
| #1782 | LMP_BASE/DEPTH 5/5 via UCI @ 180+2 | **+4.5** | aggressive movecount pruning pays at depth (consensus LMP_BASE=3) |
| #1882 | NMP moved after RFP | -0.06 ±1.9 (neutral) | reorder is free; an enabler, not a win |
| #1901 | NMP_VERIFY 74→120 | final **H0 +0.2 ±1.1** (early +1 faded) | verification cost alone too small to bank standalone |
| #1903 | NMP threat gates → margins | final **+2.6 ±1.9 H1**, merged | modulate-don't-gate (consensus style) wins |
| #1904 | NMP consensus R-shape standalone (NMP-first order) | **-6.68 H0** | shallow NMP intercepted free RFP cutoffs — wrong without the reorder |
| #1896/#1900 | v6-s5-swa 5000-iter tunes (STC/LTC) | in flight | both dialing RFP down + verify down |
| #1906 | `experiment/rfp-first-nmp-ungated`: reorder + MIN_DEPTH 75→25 + VERIFY 74→120 | final **+1.5 ±1.2 H1 (87,936 games), merged** | the #1904 retry with the interception mechanism removed; bench 3,141,699 → 2,724,601 (-13%). Includes #1901's verify change, which H0'd standalone — the bundle is what banks |

## The unifying claim

At depth, the consensus engines trust move-ordering/movecount pruning
MORE and static-eval-margin pruning LESS; Coda's STC-tuned equilibrium
has both inverted (RFP linear-to-d17 + permissive LMP). Both banked LTC
carve-outs (#1874, #1782) are moves toward the consensus shape that STC
SPSA cannot see. The cross-engine table is best read as a map of
deployment-TC-optimal shapes, while our 10+0.1 SPSA explores
STC-optimal shapes. This is a candidate explanation for part of the
~-50 Elo cross-engine gap at LTC vs STC.

**AMENDED 2026-06-10 (hash discovery):** most of that −50 was TT
starvation, not pruning shape. Adam's 43-engine LTC RR rerun at
Hash=512 recovered ~40 of it (Coda +25, the pool's largest gainer);
the shape claim above applies to the residual ~10, which matches the
banked carve-outs (#1874 +3.5, #1782 +4.5). Mechanism and fill
measurements in experiments.md 2026-06-09/10. Corollary: #1900's LTC
outputs were tuned at Hash=64 (starved regime); #1915 (LTC@256,
warm-started) is the deployment-valid LTC tune.

## Queued follow-ups

1. `experiment/rfp-first-nmp-ungated` SPRT → if viable, focused NMP+RFP
   cluster SPSA on-branch (~1500 iters), then final SPRT (guard-pattern;
   #1904 never got this step).
2. If merged: re-run RFP_AUDIT — the d≥8 anomaly should disappear
   (failed nulls can no longer be overridden).
3. LTC shape bundle: RFP_DEPTH=8 + LMP 5/5, SPRT at 40+0.4 `[0, 3]`,
   sequenced AFTER the NMP experiment lands (domains overlap), with
   #1900 LTC tune outputs applied first if converged.
4. Optional instrument v2: no-handicap ground-truth verifier (real
   reduced search instead of null) to measure TRUE FP rate; and per-depth
   NMP attempt/cutoff histograms in bench stats.

## When to update

Extend (don't replace) when: the rfp-first SPRT resolves; the LTC bundle
runs; #1896/#1900 outputs are applied; or the audit is re-run on a
reordered trunk. Re-run the audit after any RFP margin-shape or
NMP-economics change — the per-depth table is the fingerprint to diff.
