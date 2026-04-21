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

**T1.2 — Skewer detector (B2 from `threat_ideas_plan`).**
Already specced, gated on B1 landing — **now unblocked**. Estimated +10–30 in that doc. Very likely to H1 given B1 landed and skewer is tactically adjacent.

**T1.3 — Overload / removing-the-defender.**
Capture/attack on a defender that defends ≥2 of its own pieces. Cheap: count defended-squares per piece, bonus moves that hit a 2+-defender. Expected: **+3 to +10**. Named-tactic class.

**T1.4 — Battery bonus.**
Quiet move that places a Q/R/B behind another Q/R/B aligned on the same file/rank/diagonal toward an enemy piece or the enemy king zone. Cheap via occupancy + between(). Expected: **+3 to +10**. Adjacent to king-zone-pressure gate pattern.

**T1.5 — Trapped-piece escape.**
Quiet move that moves a piece whose ALL non-loss legal moves are limited to ≤1 square (mobility count trap). This is essentially the dual of "attack a trapped piece" and captures "evacuate before fork". Expected: **+3 to +8**.

### Tier 2 — fits W2 (gates) with a NEW signal, moderate confidence

Phase 2 of the signal × context matrix has saturated at 1/10 H1 rate. **Pivot from "more contexts for old signals" to "new signals for the same gate mechanism".**

**T2.1 — Undefended-piece-count as NMP gate.**
Count of our own pieces with ≥1 attacker and 0 defenders ("hanging"). If ≥1, skip NMP (a zugzwang proxy for the opponent exploiting the hanging piece). Expected: **+3 to +8**. Novel signal, old gate context.

**T2.2 — King-mobility as SE / extension gate.**
Number of legal king moves. When ≤1, a quiet king move must be searched deeper. Already implicit in some check-extension logic, but not as an explicit SE trigger. Expected: **+3 to +8**.

**T2.3 — Mobility-delta-of-moved-piece as quiet-ordering score.**
Cheap: `popcount(attacks_to(from)) - popcount(attacks_to(to))`. Add as small-weight bonus in quiet-ordering alongside history. Centralization heuristic without writing an eval term. **Ordering, not pruning** — different mechanism class. Expected: **+2 to +5**. Low confidence but cheap.

### Tier 3 — novel mechanism, higher variance

**T3.1 — Second TT-move slot.**
Store two best moves per TT entry instead of one. Try both before quiet generation. Doubles the "TT move tried" coverage with ~8 bytes per entry extra. No engine in our survey does this explicitly. Expected: **+2 to +10**. Risk: TT size pressure.

**T3.2 — Quiet-SEE for attack-creating moves.**
For quiets that change a square's attack status toward an enemy piece, compute SEE(move, captured=target). If positive, order early (and treat as "tactic-creating quiet"). Gate by "new threat created" to keep cost low. Novel mechanism; closest to Obsidian's "good noisy quiet" concept. Expected: **+3 to +10**.

**T3.3 — Threat-delta tagging (Path 2 refactor from `threat_ideas_plan`).**
Infrastructure work: store per-move threat-delta at make-move time. Unblocks cheap access to "this move extinguished threat X / created threat Y" at any point in search. Currently blocked on... B1 landing (B1 has landed). **Unlocks T1.2, T1.5, T2.1 implementations** at <1% NPS cost. Worth doing as infrastructure even before Elo-capture experiments.

### Tier 4 — training-side, orthogonal, high variance

**T4.1 — Factoriser** (already in `factoriser_design_2026-04-21.md`). Estimated +5–20. Independent of search work.

**T4.2 — Prune dead FT features.**
L1 sparsity work measured 8.4% structural-zero rows. Next question: **features rarely active AND rarely load-bearing** — can we drop them entirely to shrink FT? Needs a feature-importance measurement (e.g. train with per-feature L0 and see which rows still fire). **+0 Elo but -FT-size →** potential NPS and training-speed gains.

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

## Companion docs

- `move_ordering_understanding_2026-04-19.md` — mechanism context.
- `threat_ideas_plan_2026-04-19.md` — has the B2 spec for T1.2.
- `signal_context_sweep_2026-04-19.md` — matrix; T2.x extends it on the signal axis.
- `capture_ordering_crossengine_2026-04-20.md` — context for T3.1, T3.2.
- `factoriser_design_2026-04-21.md` — T4.1 detail.
- `experiments.md` — source of truth for resolved SPRTs.
