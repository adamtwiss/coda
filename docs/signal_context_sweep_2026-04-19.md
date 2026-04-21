# Signal × Context Sweep — systematic search for compounding threat-signal wins

## Pattern

Landed Elo over the past few days follows a clear shape:

> A **signal** that captures some tactical/positional fact about the
> position is derived. Once derived, it can be plugged into any
> **context** — any search decision point where more information
> improves the gate or margin. Each (signal, context) pair is a small
> experiment; many don't land, but the ones that do add 3-10 Elo each
> for ~1 day of work.

Recent examples of the same signal lighting up in multiple contexts:

| Signal | Context 1 | Context 2 | Context 3 |
|---|---|---|---|
| `enemy_attacks` (full-piece attack bitboard) | History indexing #463 (+14.2) | (escape bonus predated) | — |
| `king_zone_pressure` (popcount of enemy_attacks ∩ king_zone) | NMP skip #466 (+7.94) | LMR modifier #482 (+6.81) | ProbCut gate #481 (+7.03) |

**Lesson**: when a signal wins in one context, check it against every other context. Cost per variation is ~1-2 hours of code + SPRT. Expected hit-rate seems to be ~1 in 2-3 at H1 under [0, 5] bounds.

This doc formalises the sweep as an indexable table.

## Signals (rows) — position-level tactical indicators

| Signal | How computed | Cost | Origin | Status |
|---|---|---|---|---|
| **enemy-attacks bitboard** | `Board::attacks_by_color(them)` — union of every square attacked by any enemy piece | 8-12 magic lookups/node | Added #463 | used in history indexing, escape bonus |
| **pawn-threat-count** | `popcount(enemy_pawn_attacks & our_non_pawns)` | trivial | predates v9 | used in LMR reduction, RFP margin |
| **king-zone-pressure** | `popcount(enemy_attacks & king_zone)`, where king_zone = king_attacks(ksq) ∪ ksq_bb | 1 magic + 1 popcount (reuses `enemy_attacks`) | #466 | NMP gate ✅, LMR modifier ✅, ProbCut gate ✅ |
| **our-king-zone-opportunity** | popcount of *our* attacks on *their* king_zone (mirror of above) | same cost | — | UNTRIED |
| **can_win_material** | Viridithas pattern: any lesser-piece-attack on our higher-value piece (or mirror) | 3 bitboard ops | proposed / #486 testing | RFP 🚧#486 |
| **threat-accumulator magnitude** | sum of \|threat_acc\| over both perspectives (NNUE-derived) | ~300ns scalar pass over 1536 i16 | proposed / #488 testing | LMR 🚧#488 |
| **threat-delta count per move** | size of `board.threat_deltas` after make_move | 0 (already computed) | — | UNTRIED |
| **eval instability** | \|static_eval - (-parent_static_eval)\| > UNSTABLE_THRESH | 0 (existing) | — | used via UNSTABLE_THRESH; untried as a modulator in other contexts |
| **corrhist magnitude** | abs(static_eval - raw_eval) | 0 (existing) | — | used in LMR complexity; untried elsewhere |
| **our-defenses count** | popcount(our attacks on our own occupied squares) — how mutually-defended we are | 2-6 magic lookups | recent (`our_defenses` futility widener) | futility 🚧 |
| **xray-blocker bitboard** | Squares where moving our piece uncovers our slider's attack on enemy (`Board::xray_blockers`) | 4-12 magic lookups | added on `experiment/threats-search-plan` branch | UNTRIED — intended for movepicker bonus |
| **threat-feature-count** | number of active threat features in position (not the magnitude but the cardinality) | derivable from existing threat enumeration | — | UNTRIED |

## Contexts (columns) — search decision points

Each context is an existing decision point where a signal can gate the decision or modulate a margin/reduction.

| Context | What the signal gates/modulates |
|---|---|
| **RFP** | Margin; skip entirely |
| **NMP** | Skip gate; R adjustment |
| **LMR** | Reduction modifier; full-depth force for subset of moves |
| **LMP** | Threshold |
| **Futility** | Margin |
| **ProbCut** | Gate (skip entirely); margin |
| **Singular Extensions** | Gate; margin; double-extension threshold |
| **History pruning** | Threshold |
| **Move ordering** | Score contributions in movepicker |
| **Aspiration window** | Width at root |
| **IIR** | Gate |
| **SEE pruning** | Threshold |

## The matrix (status as of 2026-04-21)

Legend: ✅ landed, 🚧 in flight / queued, ❌ tested and H0, ❓ untried.

| Signal \ Context | RFP | NMP | LMR | LMP | Fut | ProbCut | SE | HistP | MovePick | ASP | IIR | SEE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| enemy-attacks bb | | | ✅history #463 | | | | | | ✅escape | | | |
| pawn-threat-count | ✅widen | | ✅reduce | | | | | | | | | |
| king-zone-pressure | ❌#503 | ✅#466 | ✅#482 | ❌#504 | ❓ | ✅#481 | ✅**#553** | ❓ | ❓ | ❌#512 | ❓ | ❓ |
| our-king-zone-opportunity | ❓ | ❌#605 | ❓ | ❓ | ❓ | ❌#538 | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| can_win_material | ❌#479/#501 | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| threat-mag | ❓ | ❓ | ❌#488/#500 | ❓ | ❓ | ❓ | ❓ | ❓ | ❌#588 | ❓ | ❓ | ❓ |
| any_threat_count | 🚧#606+1.5 | ✅**#539** | ✅reduce | ❌#597 | ✅#484 | ❓ | ❌#590/#607 | ❓ | ❓ | ❓ | ❓ | ❓ |
| eval instability (unstable) | 🚧#589+1.2 | ❌#594 | ❌#541 🚧#548-gated | ❌#597 | ❌#593 | ✅**#542** | ✅#598-alt | ✅guard | ❓ | ❓ | ❓ | ❓ |
| corrhist mag | ❓ | ❓ | ✅complex | ❓ | ❓ | ❌#544 | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| our-defenses | ❓ | ❓ | ❓ | ❓ | ✅#484 | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| xray-blockers | ❓ | ❓ | ❓ | ❓ | ❓ | ❌#591 | 🚧#604~0 | ❓ | ✅**#502 +52** | ❓ | ❓ | ❓ |
| threat-feature-count | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |

**Landed count**: ~11 (added #553 kzp×SE +9.7). **Failed (H0)**: ~19
(+9 H0s from the 2026-04-20/21 unstable-widener family + today's fresh
cells: anythreat×SE half, xray×SE neutral, anythreat×RFP trending H1).
**In flight**: #604 (xray×SE neutral +0.9), #606 (anythreat×RFP +1.5,
probable H1). **Untried**: ~80.

### Hit rate trend (important)

Phase 1 sweep (Apr 19-20, early cells): **~23% hit rate** — the big
wins (kzp×NMP, kzp×ProbCut, kzp×LMR, kzp×SE, xray×MovePick, anythreat×
NMP, unstable×ProbCut). ~4 of first ~17 cells landed.

Phase 2 sweep (Apr 20-21, later cells including the "unstable×{RFP/
NMP/Fut/LMP/bundle}" family and my own fresh-ideas batch): **~1 of 10
landing H1** (only rfp-anythreat-widen trending). The sweep is
saturating — cells remaining are either redundant with SPSA's implicit
exploitation of the signals, or combinations where the signal doesn't
compose.

**Conclusion:** continue to pick individual high-prior cells from this
matrix, but STOP batch-queuing multiple marginal cells. The big wins
are all from the first 40% of the matrix. Remaining picks should be
those with specific mechanism hypotheses, not exhaustive sweeping.

## Key findings from 2026-04-19/20 overnight sweep

**Pattern strongly validates (signal × context)**: of ~13 novel signal-context pairs tested:
- 3 landed strong (+52, +7, +6) = ~23% hit rate
- Several more landed small (#551 tune +2 grinding; #481 at +7.03; #482 at +6.81)

**Specific lessons per signal**:

1. **king-zone-pressure seems to have saturated at 3 contexts** (NMP #466, LMR #482, ProbCut #481). Four subsequent attempts (RFP #503, LMP #504, SE #511, ASP #512) all H0'd. Signal generalizes to 3/7 contexts tested, not 7/7.

2. **any_threat_count is the strongest new "signal" this session**: 3 landings already (futility #484, LMR threat-density, NMP #539). RFP-widener #540 H0'd due to signal overlap with has_pawn_threats widener; not a failure of the signal itself.

3. **unstable (parent-child eval gap) is underused**: landed at ProbCut #542 (+6.7) on first systematic attempt. LMR modifier #541 H0'd for magnitude reasons; depth-gated retry #548 in flight.

4. **Mirror signals don't always transfer**: our-king-zone-opportunity at ProbCut (#538) H0'd despite enemy-king-zone-pressure at same context working (#481 +7.03). Attacker/defender symmetry assumption fails for this signal.

**Priority for next sweep waves**:
- any_threat_count × SE, SEE, LMP, IIR (all untested, strong prior given 3 landings)
- unstable × RFP, NMP, Fut, SEE (one win at ProbCut, likely more)
- xray-blockers × every context (B1 pattern is position-level tactical, may work as gate/modifier elsewhere)

## Prioritised next experiments

Ordered by "plausibility × ease" (plausibility derived from "same signal worked in an adjacent context"). Each is a one-to-two hour code change plus SPRT.

### Tier 1 — king-zone-pressure in adjacent contexts (pattern match with #466/#481/#482 landings)

The king-zone signal has landed in 3 out of 3 tested contexts. Before assuming it saturates, try the obvious remaining slots:

| # | Experiment | Rationale |
|---|---|---|
| S1 | **King-zone-pressure RFP margin widener** | When our king is under pressure, static eval underestimates danger. Widen RFP margin by a scaled pressure bonus. Mirrors pawn-threat-count's existing RFP widening |
| S2 | **King-zone-pressure futility widener** | Same logic as RFP, applied to futility pruning margin. Orthogonal code path |
| S3 | **King-zone-pressure LMP threshold softener** | LMP prunes later moves aggressively; don't prune as many when our king is under pressure |
| S4 | **King-zone-pressure SE margin** | Singular extension margin widened when our king is under pressure (more likely to need extension) |
| S5 | **King-zone-pressure aspiration width** | Wider aspiration at root when king-pressure is high (score volatility expected) |
| S6 | **Mirror: our-king-zone-opportunity gate for ProbCut** | If *we* have high pressure on *their* king, ProbCut is more likely to work (attacker has initiative) |

### Tier 2 — if SPSA #488 (threat-mag-LMR) lands, try it in adjacent contexts

Threat-accumulator magnitude is a NNUE-specific signal. If #488 lands, sweep the same contexts as king-zone-pressure already landed in:

| # | Experiment | Rationale |
|---|---|---|
| S7 | **Threat-mag NMP skip gate** | Same pattern as king-zone-pressure NMP gate — skip NMP when threat-mag is very high |
| S8 | **Threat-mag ProbCut gate** | High threat-mag → ProbCut less reliable (deep tactics) |
| S9 | **Threat-mag RFP margin widener** | Positions with strong threat activations under-predict deep eval |

### Tier 3 — if `can_win_material` (#486) lands, mirror into other margins

| # | Experiment | Rationale |
|---|---|---|
| S10 | **`can_win_material` futility loosener** | Currently proposed for RFP. Futility has separate margin — same logic applies |
| S11 | **`can_win_material` LMP softener** | Same mirror |
| S12 | **`can_lose_material` RFP guard** | If *opponent* has a lesser-piece-attack on our higher piece (mirror), widen RFP (we might be worse than static says) |

### Tier 4 — novel signals in new contexts

| # | Experiment | Rationale |
|---|---|---|
| S13 | **Threat-delta count as movepicker move score** | More threat changes caused = more tactical move. Zero NPS cost (already computed after make_move). Needs speculative computation OR ranks moves after make_move/unmake (expensive) |
| S14 | **xray-blockers movepicker bonus** | The B1 experiment from `threats-search-plan` doc. Bonus for moving pieces that are x-ray-blocking our slider's attack on enemy (discovered attack candidates) |
| S15 | **Eval instability → wider aspiration at root** | When parent-child eval disagrees significantly, score volatility is high; widen asp window |
| S16 | **Eval instability → LMR reduce less** | Positions where eval swings a lot deserve deeper search |
| S17 | **Corrhist magnitude → ProbCut skip** | Already used as LMR complexity modifier; ProbCut is similarly affected by eval uncertainty |

### Tier 5 — lower-confidence / experimental

| # | Experiment | Rationale |
|---|---|---|
| S18 | **Full threat-feature-count in movepicker** | Position-wide count of active threat features as a tactical-complexity proxy |
| S19 | **our-defenses count in NMP R adjustment** | If our pieces are well-defended, NMP can be more aggressive |
| S20 | **Threat-mag difference (parent vs current) as extension signal** | If a move caused the threat accumulator to change significantly, extend depth |

## Calibration — my prior Elo estimates ran low

Test #481 (ProbCut king-zone gate) was in the "trivial +1-3 Elo" slot in my earlier brainstorm. Landed at +7.03. Same pattern for LMR king-pressure (#482) at +6.81.

Going forward I should assume:
- A well-motivated (signal, context) pair that pattern-matches a prior win: **expected ~+5-10 Elo**, not +1-3.
- An experimental (signal, context) pair with no precedent: **still use +2-5 Elo** as estimate.
- The distribution is long-tailed; a few of the above will land +10+ and most will be flat/negative. Run all the Tier 1s under [0, 5] bounds as a batch.

## Suggested execution order

Rather than pick-one-at-a-time, run Tier 1 as a **parallel sweep**:

1. Create six short-lived branches (S1-S6), each one a 10-20 line diff.
2. Bench all six on the same post-trunk HEAD (all should produce identical node counts since each is a different param — no, they'll differ because each changes search behaviour).
3. Submit all six as independent SPRTs with [0, 5] bounds, same dev-network.
4. As they resolve, merge the H1s in order of landing; rebase the H0-but-queued ones on latest trunk.

Expected: 2-4 of 6 land at +3-8 each → +10-25 cumulative Elo from 1-2 days of SPRT time.

After Tier 1 resolves, Tier 2 is gated on #488 resolving. Tier 3 gated on #486. Tier 4 can run anytime — those are orthogonal.

## What NOT to sweep

- **Tweaks to existing SPSA-tuned params** (RFP margin numbers, LMR coefficients) without adding a new signal. SPSA has already explored these.
- **Signal combinations in a single context** without first validating each signal individually (e.g., "king-pressure AND threat-mag in RFP"). That's a local-max search inside SPSA's territory.
- **Anything that requires retraining** the NNUE. This sweep is purely search-side.

## What to watch for

- **Pattern exhaustion**: if three signals × three contexts all land and the fourth doesn't, we may have hit the useful subset. Move to novel signals (Tier 4).
- **Tune decay**: tune #470's SPSA values were calibrated against pre-sweep trunk. After 5-10 signal additions land, a full LMR/margin retune will probably show another +3-8 Elo. Budget one after every ~5 merges.
- **Diminishing tune retunes**: the first retune after signal additions is usually best. Subsequent retunes without new signals find less. That's when the sweep is exhausted and attention should shift back to training / NPS / new architectural ideas.

## Appendix: running/recent tunes and branches relevant to this sweep

- `#486 experiment/stratified-escape-canwin` — A1a + A1c bundle (in flight)
- `#488 experiment/threat-mag-lmr` — threat-mag LMR modifier (in flight; this doc's author)
- `experiment/threats-search-plan` — includes `xray_blockers` helper, ready for S14
- `feature/threat-inputs` — v9 trunk; branch base for all sweep experiments

## Updates

This doc is the living catalogue. When an experiment lands or is killed:
1. Update the matrix row (✅ / ❌ with test number and Elo).
2. Promote the next Tier 2/3/4 items to Tier 1 if the adjacency pattern reveals itself.
3. Note any calibration shifts in the "prior Elo estimates" section.
