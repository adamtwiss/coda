# Threat-signal search-consumer plan (2026-04-19)

Post-Hercules-cleanup + V9 > V5 milestone. Captures Titan's revised plan
so the next session can pick up without re-deriving.

## What's already in place

- Full-attacks 4D history key: **+14.2 Elo H1** (#463, merged)
- King-zone-pressure NMP gate: **+7.9 Elo H1** (#466, merged)
- Threat-density LMR reduction (`LMR_THREAT_DIV`): pre-existing
- Escape-capture bonuses (`ESCAPE_BONUS_{Q,R,MINOR}`): pre-existing, gated
  on enemy PAWN attacks only, stratified by defender-piece type only
- 2b slider-iteration rewrite + fix: `fix/threats-2b-rewrite`, fuzzer-clean,
  semantics-preserving, currently SPRT'ing for NPS win (bounds [0, 5])
- `profile-threats` per-section + zero-emit counters: landed
- `Board::xray_blockers(color)`: landed, unused, ready for B1

## What's been tried and rejected

- Enter-threat quiet-move penalty (unstratified): **H0 -4 Elo** (#465)
- Hanging-piece escape bonus (scoring nudge): **H0 -11 Elo** (#467)

Pattern observed: **GATES and BROADER INDEXING work; scoring nudges fail.**
Gates (king-zone NMP, LMR-reduction modifiers) and broader indexing
(full-attacks-history) have landed positive. New scoring bonuses that
compete with existing ordering heuristics have failed twice.

## Tier A — consensus patterns, untried in Coda

### A1a — Stratified escape ladder (bundle w/ A1c)

Reckless pattern: escape bonus stratified by BOTH defender piece type AND
attacker piece class.

- Our Q on square attacked by rook-or-queen → `ESCAPE_BONUS_Q_FROM_RPLUS`
- Our R on square attacked by minor-or-higher → `ESCAPE_BONUS_R_FROM_MINORPLUS`
- Our N/B on square attacked by pawn → `ESCAPE_BONUS_MINOR_FROM_PAWN`
  (matches existing, keep as-is)

Requires three precomputed their-attacks bitboards stratified by attacker:
- `their_pawn_attacks`
- `their_minor_attacks` (N+B)
- `their_rook_plus_attacks` (R+Q)

### A1c — `can_win_material` RFP loosener

When we have a net-positive capture available (e.g. our minor attacks
their rook without being defended by a same-or-lower-value piece), widen
RFP margin so we don't prune a line that could gain material.

Shares the three stratified bitboards with A1a — single branch, single
SPRT.

### A1b — Onto-threatened penalty (deferred)

Unstratified version H0'd as #465. Revisit only after A1a+A1c land,
with proper SEE/piece-value gating so we're not fighting the main
capture ordering.

### A2 — 2-bit threat-level history stratification

Extends the +14.2 full-attacks-history win. Current 4D main-history key
is `[from_threatened][to_threatened][from][to]` (boolean from_threatened,
boolean to_threatened). Upgrade to 2-bit threat level (0=safe, 1=pawn,
2=minor, 3=major) using the same bitboards as A1a.

Table size multiplies by 4 (2×2 → 16 combinations). Compounds on #463 —
same key, richer signal.

### A3 — ProbCut gate on high threat state

Skip ProbCut when our position has many hanging pieces or king under
heavy pressure. Trivial, slots anywhere after A1.

## Tier B — novel, x-ray data surfaced to search

### B1 — Discovered-attack movepicker bonus (Path 1 fresh-compute)

Use `Board::xray_blockers(our_color)` to detect moves that create
discovered attacks. If `move.from() ∈ xray_blockers`, add a bonus
proportional to the x-ray victim's value.

**Note**: This is a scoring nudge pattern. The two prior scoring nudges
(enter-penalty, hanging-escape) H0'd. Approach with a different framing:
pure bonus scaled by victim value, not an "override the capture ladder"
value. Must resolve H1 >6 Elo to justify Path 2 refactor.

### B2 — Skewer-detector movepicker bonus

Blocked on B1 outcome. Cheap only after Path 2 (delta-tagging
infrastructure) refactor.

### B3 — Raw delta-magnitude as extension signal

**Shelved** — Titan measured delta histogram and 67% of deltas are in
magnitude 5-12, 30% in 13-24. Distribution too uniform to give a clean
threshold.

## Tier C — ambitious

### C1 — Trained threat-weight-magnitude oracle

Use an auxiliary head to predict "how much does this threat feature
matter in this position?" Later. High variance.

### C2 — NNUE accumulator delta as extension signal

Independent. Better resolution than shelved B3. Slots anywhere after
A-tier resolves.

## X-ray data paths

### Path 1 — fresh compute at node entry (landed)

`Board::xray_blockers` recomputes x-ray state per call. ~3-5% NPS cost,
correctness-isolated. Used by B1 first.

### Path 2 — delta-tagging at emission (gated on B1 > 6 Elo)

Steal a bit from `attacker_cp` in `RawThreatDelta` to tag deltas as
x-ray-vs-direct. Pure infrastructure, no retraining required. Earn this
refactor only if B1 Path-1 shows >6 Elo — otherwise 5% NPS cost eats the
gain.

### Path 3 — derived bitboards during emission (not scoped)

Threats.rs has been our #1 correctness hot-spot (the 2b rewrite bug just
demonstrated this again). Each additional state emission adds another
fuzzer case. Defer indefinitely unless B1/B2 justify it.

## Suggested experiment order

1. **A1a + A1c bundle** — stratified escape + can_win_material, shared
   bitboards. One branch, one SPRT. **Next up.**
2. **2b rewrite NPS measurement** — currently #477 (then renumbered).
   Orthogonal, runs in parallel with A-tier.
3. **A2** — 2-bit threat-level history, compounds on #463, reuses A1's
   bitboards.
4. **A3** — ProbCut threat gate. Trivial.
5. **B1 via Path 1** — validates x-ray lever. Must hit >6 Elo to justify
   Path 2.
6. **Path 2 + B2 + x-ray history stratification** — gated on B1 magnitude.
7. **C2** — NNUE accumulator delta extension. Independent, slots anywhere.
8. **C1** — weight-magnitude oracle. Last, highest variance.

## Retune timing

A-tier threat work + post-merge tune #470 already landed (+5.7 Elo). Budget
another SPSA cycle after A1 + A2 merge — those will shift the search
landscape again.

## References

- `project_v9_threat_advantage_pattern` memory — "rich signal → search
  consumer" pattern, why these improvements are Coda-specific levers
- `docs/threat_features_design.md` — original v9 architecture design
- `experiment/full-attacks-history` (merged) — +14.2 Elo precedent
- `experiment/king-zone-nmp` (merged) — +7.9 Elo precedent
- `experiment/enter-penalty` (H0) — scoring-nudge cautionary tale
- `experiment/hanging-escape` (H0) — scoring-nudge cautionary tale
