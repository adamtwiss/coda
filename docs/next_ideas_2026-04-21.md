# What to Try Next — Data-Driven Idea Shortlist (2026-04-21)

Written from the vantage of 2026-04-21 after reviewing `experiments.md`
and all move-ordering / threat-signal docs on trunk. **Anchored on
observed win/loss pattern**, not on "X engine does Y".

Docs may lag `experiments.md` — the log is authoritative for H1/H0 state.

## What the data actually says

### Winning patterns (H1 reproducibly)

**Pattern W1 — Specific tactical-motif bonus in MovePick, retuned on branch.**
- B1 discovered-attack (#502, **+52**), knight-fork (#578, **+5.2 post-retune**, would have been H0 without retune), offense-bonus (#554, **+5.7**).
- **What makes it work:** the motif has to be a *named tactic* that NNUE eval already captures but search doesn't project into ordering. Flat scoring nudges ("a piece is attacked" in general) have H0'd repeatedly.

**Pattern W2 — Threat signal gating a pruning DECISION (not widening a margin).**
- king-zone-pressure as NMP gate (#466, +7.9), ProbCut gate (#481, +7.0), LMR-reduction modifier (#482, +6.8), SE gate (#553, +9.7). Four of eight tested contexts landed.
- any_threat_count gating NMP (#539, +6.0) and futility (#484, +7.0).
- unstable-eval gating ProbCut (#542, +6.7).
- **What makes it work:** gates are discrete / binary. The signal says "don't prune this" and search does the work.

**Pattern W3 — Retune-on-branch after 2–4 merges.**
- #490 (**+7.4**) post-4-feature-bundle, #586 (**+6.2**) post-knight-fork-plus.
- **What makes it work:** SPSA compensates for tree-shape changes. Retune captures that "gates let the rest of search be bolder" compound effect.

### Failing patterns (H0'd, **don't re-try verbatim**)

**Pattern L1 — Generic "piece-is-attacked" nudge.** Enter-penalty, hanging-escape, rook-kingring (#555 +0.1), queen-7th (#565 -2.0) all H0'd. Retune on L1 features doesn't rescue. The signal is too diffuse.

**Pattern L2 — Threat signal as MARGIN WIDENER.** RFP-kzp (#503 -9.5), LMP-kzp (#504 -10.1). Gates yes, wideners no.

**Pattern L3 — Dimensional expansion of captHist.** caphist-defender absorbed by B1's ordering fix (not additive). Dynamic-SEE threshold H0'd twice. Stratified threat-history (#480 -3.3) didn't compound on binary-threat gate. **Bucketing doesn't carry signal beyond binary threshold.**

**Pattern L4 — Threat signal for EXTENSION / LMR-amount.** threat-mag-LMR (#500) faded +24 → +0.8, threat-delta-ext (#580) -8.9. Continuous threat magnitude as depth-modifier doesn't work.

### Extracted rules

1. **Scoring mechanism > margin mechanism.** Specific MovePick bonus (+5 to +52 per motif) and pruning gate (+5 to +10) both work. Margin widening and extension don't.
2. **Binary > continuous.** Gates (binary) and motif-bonuses (present/absent) work. Continuous threat-magnitude signals keep fading or failing.
3. **Named-tactic > abstract-feature.** Motifs with a chess name (fork, discovered attack, pin, skewer) beat generic "attack/defend" features.
4. **Retune per ~3 merges banks 6–8 Elo**, non-negotiable — not a nice-to-have.

## Novel ideas — ranked by expected ROI × confidence

### Tier 1 — fits W1 (specific tactical motifs), highest confidence

All follow the B1 recipe: add a MovePick bonus for quiet moves matching a named tactical motif, SPSA-tune the bonus constant on-branch.

**T1.1 — Pin bonus.**
Move that pins an enemy piece to a more-valuable piece/king. Cheap: iterate sliders, check if an enemy piece on our attack ray is the nearest piece and there is a more-valuable piece (or king) beyond it on the same ray. Expected Elo: **+5 to +20** (pins are one of the three Big Three named tactics alongside forks and discovered attacks — both of the other two landed).
- Variant T1.1b: penalty for moves OF our own pinned pieces (they're usually bad).
- **Status 2026-04-22:** tested value-filtered bundle T1.1+T1.2 as `experiment/pin-skewer-value-filtered` → **#618 -6.4 @ 4728g H0**. Did not rescue the unified skewer attempt (#612 -8.0). B1's FROM-based discovered-attack appears to already cover the ray-alignment signal; pin/skewer add noise rather than missing H1 coverage on current trunk. **Dropped unless re-tested under retune-on-branch.**

**T1.2 — Skewer detector (B2 from `threat_ideas_plan`).**
Already specced, gated on B1 landing — **now unblocked**. Estimated +10–30 in that doc. Very likely to H1 given B1 landed and skewer is tactically adjacent.
- **Status 2026-04-22:** **H0** twice. Unified variant #612 -8.0 @ 3848g; value-filtered variant (bundled with T1.1) #618 -6.4 @ 4728g. Same conclusion as T1.1 — the expected signal was absorbed by B1 (+52, merged). Dropped.

**T1.3 — Overload / removing-the-defender.**
Capture/attack on a defender that defends ≥2 of its own pieces. Cheap: count defended-squares per piece, bonus moves that hit a 2+-defender. Expected: **+3 to +10**. Named-tactic class.
- **Status 2026-04-22:** **#634 -1.4 @ 13624g H0.** Quiet-scoring bonus at OVERLOAD_BONUS=4000 did not pass SPRT on the pre-C8-fix trunk.
- **Status 2026-04-23:** retried on tuned C8-fix trunk (#669 experiment/t1-3-overload-c8-retry) → **−4.9 @ 6056g H0 ✗**, past −4 tripwire. Signal genuinely absent across trunks; absorbed by B1 discovered-attack. **Dropped.**

**T1.4 — Battery bonus.**
Quiet move that places a Q/R/B behind another Q/R/B aligned on the same file/rank/diagonal toward an enemy piece or the enemy king zone. Cheap via occupancy + between(). Expected: **+3 to +10**. Adjacent to king-zone-pressure gate pattern.
- **Status 2026-04-22:** **#635 +5.3 @ 8452g H1 ✓**, merged at `2caa96a`. Landed near the top of the expected band.

**T1.5 — Trapped-piece escape.**
Quiet move that moves a piece whose ALL non-loss legal moves are limited to ≤1 square (mobility count trap). This is essentially the dual of "attack a trapped piece" and captures "evacuate before fork". Expected: **+3 to +8**.
- **Status 2026-04-23:** **H0 both variants.** #685 bonus=4000 −2.7 Elo H0 (904g early-stop);
  #687 bonus=2000 H0 −3.7 Elo @ 8174g (re-sampled to rule out single-point bad-value).
  Mobility-proxy signal doesn't surface here — probably absorbed by T2.3 mobility-delta
  (landed +1.4) which captures the "piece wants out of cramped square" directional move.
  **Dropped.**

### Tier 2 — fits W2 (gates) with a NEW signal, moderate confidence

Phase 2 of the signal × context matrix has saturated at 1/10 H1 rate. **Pivot from "more contexts for old signals" to "new signals for the same gate mechanism".**

**T2.1 — Undefended-piece-count as NMP gate.**
Count of our own pieces with ≥1 attacker and 0 defenders ("hanging"). If ≥1, skip NMP (a zugzwang proxy for the opponent exploiting the hanging piece). Expected: **+3 to +8**. Novel signal, old gate context.
- **Status 2026-04-22:** tested as `experiment/undefended-nmp-skip` → **#617 +0.4 @ 44792g H0** on pre-C8 trunk.
- **Status 2026-04-23:** retested on tuned C8-fix trunk (#676 experiment/undefended-nmp-skip-c8-retry) → **+2.9 @ 10122g H1 ✓, merged.** Tuned trunk + cleaner eval signal turned the marginal result positive — same feature, different calibration regime. Lands in the expected band.

**T2.2 — King-mobility as SE / extension gate.**
Number of legal king moves. When ≤1, a quiet king move must be searched deeper. Already implicit in some check-extension logic, but not as an explicit SE trigger. Expected: **+3 to +8**.
- **Status 2026-04-22:** **#632 -1.4 @ 13288g H0.** Dropped — SE-margin widening for constrained-king positions did not convert to Elo.

**T2.3 — Mobility-delta-of-moved-piece as quiet-ordering score.**
Cheap: `popcount(attacks_to(from)) - popcount(attacks_to(to))`. Add as small-weight bonus in quiet-ordering alongside history. Centralization heuristic without writing an eval term. **Ordering, not pruning** — different mechanism class. Expected: **+2 to +5**. Low confidence but cheap.
- **Status 2026-04-23:** **#684 +1.4 Elo H1 ✓** @ 22200g, merged at
  `576d91c` as `MOBILITY_DELTA_WEIGHT=32` tunable. Landed at the
  lower end of expected. Also likely absorbed the signal T1.5 was
  looking for.

### Tier 3 — novel mechanism, higher variance

**T3.1 — Second TT-move slot.**
Store two best moves per TT entry instead of one. Try both before quiet generation. Doubles the "TT move tried" coverage with ~8 bytes per entry extra. No engine in our survey does this explicitly. Expected: **+2 to +10**. Risk: TT size pressure.

**T3.2 — Quiet-SEE for attack-creating moves.**
For quiets that change a square's attack status toward an enemy piece, compute SEE(move, captured=target). If positive, order early (and treat as "tactic-creating quiet"). Gate by "new threat created" to keep cost low. Novel mechanism; closest to Obsidian's "good noisy quiet" concept. Expected: **+3 to +10**.

**T3.3 — Threat-delta tagging (Path 2 refactor from `threat_ideas_plan`).**
Infrastructure work: store per-move threat-delta at make-move time. Unblocks cheap access to "this move extinguished threat X / created threat Y" at any point in search. Currently blocked on... B1 landing (B1 has landed). **Unlocks T1.2, T1.5, T2.1 implementations** at <1% NPS cost. Worth doing as infrastructure even before Elo-capture experiments.

### Tier 4 — training-side, orthogonal, high variance

**T4.1 — Factoriser** (already in `factoriser_design_2026-04-21.md`). Estimated +5–20. Independent of search work.
- **Status 2026-04-23:** **factor arch validated via #688 +13.6 Elo H1**
  and subsequent #700/#704 (+20-23 Elo factor vs non-factor at same SB400
  training length). SB800 factor net pending (~day 0 overnight train).
  Factor tune cycle (#698, #706) found no additional Elo — factor's
  pruning optimum sits close to non-factor pre-tune values. Apply
  factor net with pre-tune params on deploy, do not bundle tune.

**T4.2 — Prune dead FT features.**
L1 sparsity work measured 8.4% structural-zero rows. Next question: **features rarely active AND rarely load-bearing** — can we drop them entirely to shrink FT? Needs a feature-importance measurement (e.g. train with per-feature L0 and see which rows still fire). **+0 Elo but -FT-size →** potential NPS and training-speed gains.
- **Status 2026-04-24:** first sparse L1-reg probe at λ=1e-4 (200 SBs,
  `net-v9-768th16x32-kb10-w15-e200s200-crelu-l1e4.nnue`) produced
  **zero additional sparsity** — 8.4% zero rows, same as the
  structural floor on the unregularised prod net. Mechanism: L1
  shrinkage per step (~1e-7 at `l1_decay=1e-4 × lr=0.001`) is
  swamped by Adam's adaptive updates. **λ=1e-4 is too weak.**
  Next retry: λ=1e-3 (10× stronger) or accumulator width reduction
  768 → 640 as an orthogonal path.
  Also relevant: Zeus's NPS investigation (`docs/coda_vs_reckless_nps_2026-04-23.md`)
  confirmed the 49 MB threat matrix is Coda's single biggest memory-bandwidth
  bottleneck, and `docs/nps_microbench_hostdata.md` shows that cache
  contention (via neighbour VMs sharing L3) hurts Coda disproportionately.
  **Shrinking the threat matrix has cross-fleet compound value** —
  it's not just per-host NPS, it also reduces our own fleet's memory-bandwidth
  footprint.

**T4.3 — Two-layer hidden (32→16).**
Current is 16→32. Try flipping: 32 L1 → 16 L2. Or add a layer: 16→16→32. Tiny retrain cost, could easily H0. Data point worth having before v9 merge.

## Ranked shortlist for Hercules's queue

Recommend in this order (highest expected value to lowest risk):

1. **T1.2 Skewer detector (B2)** — unblocked, spec exists, same class as B1 (+52). Highest confidence high-value.
2. **T1.1 Pin bonus** — Big-Three-tactics completion. High confidence moderate-value.
3. **T3.3 Path-2 infrastructure** — unblocks cheap implementations of T1.3, T1.5, T2.1, T3.2. One-off cost.
4. **T1.3 Overload** & **T1.4 Battery** — run after T3.3 lands.
5. **T2.1 Undefended-piece NMP gate** — new signal × existing gate pattern. Parallel to T1 work.
6. **T4.1 Factoriser** — training-side, parallel to search work.
7. **T3.1 Second TT-move slot** — mechanism novelty; schedule after T1s resolve.

## Stop-trying list (verbatim)

Based on Pattern L1–L4:

- Don't try another dynamic-SEE-threshold variant without changing the `score` input (must be captHist-only, not MVV-dominated).
- Don't try another dimensional captHist expansion unless we have a clear motif to key on (defender bit was absorbed, stm has been tried).
- Don't try threat-magnitude as LMR-amount / extension-amount — only as gates or MovePick bonuses.
- Don't try another margin widener for a signal that already has a gate landed (RFP/LMP king-pressure proved this).
- Don't add "generic piece-attacked" nudges. Named-tactic or nothing.

## Research threads for Titan (2026-04-24)

After the 2026-04-23/24 wave (+25 Elo merged from NPS investigation
+ factor architecture validated at +20-23 Elo), the following
research threads remain genuinely untested and worth deep Titan
analysis. Ordered by expected ROI:

### Untested items from existing Tier lists

Still-untested from `next_ideas_2026-04-21.md` + `peripheral_mechanisms_2026-04-22.md`:

| Item | Source | Why Titan | Status |
|------|--------|-----------|--------|
| T3.1 Second TT-move slot | next_ideas T3 | Novel mechanism, no prior SPRT, needs careful TT-layout design | Untested |
| T3.2 Quiet-SEE for attack-creating moves | next_ideas T3 | "Good noisy quiet" concept; needs SEE-of-quiet + gate design | Untested |
| T3.3 Threat-delta tagging (Path 2 infra) | next_ideas T3 | Infrastructure for future T1/T2 experiments | Untested (infrastructure, no direct SPRT) |
| T4.3 Two-layer hidden 32→16 | next_ideas T4 | Novel architecture probe | Untested |
| P4 NNUE complexity blending | peripheral | Needs proxy-selection research (any_threat_count? corrhist? instability?) | Untested |
| P6 Thread voting for SMP | peripheral | LTC-only win candidate; needs LTC harness | Untested |
| N2 Shuffling detector | peripheral | Novel; shuffle_streak counter + halfmove-conditional blend | Untested |
| N3 Fortress soft-cap | peripheral | High-variance, needs Lichess-visible calibration data first | Untested, low priority |
| N4 Halfmove-scaled pruning margins | peripheral | Novel: scale RFP/futility/NMP margins by (200-hm)/200 | Untested |

### New threads surfaced by 2026-04-23/24 work

**R1. L1 regularisation escalation + accumulator-width path.**
First sparse probe at λ=1e-4 failed to shrink anything beyond the
8.4% structural floor. Next probes: (a) λ=1e-3, (b) accumulator
width 768 → 640 (non-factorised net, 17% memory reduction
brute-force, uncertain Elo cost), (c) combined
`--factoriser` with L1 applied to `l0w - l0f` differential only
(Bullet change required). Titan question: which path is most
likely to deliver 30-60% sparsity without blowing Elo?
Analysis should produce specific λ recommendation + measurement
protocol for each probe.

**R2. Targeted LMP adaptive / NMP TT-capture SPSA.**
Direct ports (#708 LMP adaptive −6.7 H0, #709 NMP TT-capture
−7.8 H0) failed. My #724 (narrower NMP gate) settled at exactly
+0.1 — mechanism is right but Coda has no headroom for the win
Reckless shows. Titan question: is there a different reformulation
of these patterns where Coda *would* win? Specifically:
- **LMP adaptive**: if we expose factor0/factor1 constants as
  tunables and SPSA-refine on Coda's eval distribution, is that
  likely to H1? Worth the tune cycle cost?
- **NMP TT-capture**: is there a "tt_score margin" version
  (only skip when tt_score >= beta + margin) that could land?

**R3. Reckless-derived NPS levers beyond cache layout.**
The 2026-04-23/24 wave exhausted the obvious cache-layout wins
(AccEntry flatten, eval-TT writeback, PSQ walk-back). Next layer:
- **VNNI compile-time dispatch** (vs runtime `has_avx512_vnni`)
  — Zeus flagged; modest expected (~+2-4% NPS).
- **AVX-512 setwise movegen** (Reckless has, Coda doesn't) —
  untouched, benefits bucket 3 (per-node overhead).
- **Input-chunk sparse L1 reordering** — if the loaded net has
  a hot-col structure, permute at load time.
Titan question: scope each, map to expected Elo magnitude and
implementation effort, with priority order for Hercules's queue.

**R4. Post-Item-7 retune opportunity.**
Items 1 (flatten) and 7 (eval-TT writeback) merged in this wave.
Eval-TT writeback changes evals-per-node from 0.677 → 0.581
without tree-shape change, but downstream pruning thresholds
(RFP, futility, SEE) are implicitly calibrated against the old
eval density. A focused post-merge retune of those thresholds
may bank +3-6 Elo per the retune-on-branch methodology. Titan
question: is a retune warranted NOW or after the factor SB800 +
sparse net land?

**R5. Widening tests for audit SPECULATIVE items.**
`correctness_audit_2026-04-22.md` has ~30 SPECULATIVE items, many
untouched. Several are low-risk bench-bit-exact hygiene; others
may expose real bugs under specific conditions (e.g. SEE pawn-
promotion in inner loop, duplicate SEE function in movepicker,
sample-positions filter mismatch). Titan question: rank the
SPECULATIVE items by expected-bug-triggerability (Lichess-reachable
> dev-only) and propose a fuzzer expansion programme.

## Companion docs

- `move_ordering_understanding_2026-04-19.md` — mechanism context.
- `threat_ideas_plan_2026-04-19.md` — has the B2 spec for T1.2.
- `signal_context_sweep_2026-04-19.md` — matrix; T2.x extends it on the signal axis.
- `capture_ordering_crossengine_2026-04-20.md` — context for T3.1, T3.2.
- `factoriser_design_2026-04-21.md` — T4.1 detail.
- `coda_vs_reckless_nps_2026-04-23.md` — NPS investigation (Zeus).
- `nps_microbench_hostdata.md` — cross-fleet cache-residency data (Hercules).
- `experiments.md` — source of truth for resolved SPRTs.
