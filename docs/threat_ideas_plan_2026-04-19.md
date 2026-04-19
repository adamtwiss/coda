# Threat Ideas Plan — 2026-04-19, with live results

Consolidated plan from the post-Hercules-cleanup / V9 > V5 milestone
thread. Captures the tiered experiment list with current results filled
in where tests have resolved.

**Companion docs:**
- `signal_context_sweep_2026-04-19.md` — signal × context sweep matrix
- `move_ordering_ideas_2026-04-19.md` — move-ordering-specific ideas
- This doc — threat-signal experiment plan with results

## Status change — what's in place

| Item | State |
|---|---|
| 2b slider-iteration rewrite | Merged (#478, +10.0 Elo H1) |
| Profile-threats zero-emitter counter | Landed |
| `Board::xray_blockers(color)` helper | Landed, consumed by #502 |
| V9 > V5 (SPRT #472) | NPS work de-prioritized from load-bearing to compounding |

## Tier A — consensus patterns

| # | Item | Status | Result |
|---|------|--------|--------|
| **A1a** | Stratified escape ladder (Q from rook+, R from minor+, N/B from pawn) | Tested in bundle with A1c | #479 H0 (−0.5 Elo, 20,964 games). **Post-SPSA-486 retest in flight as #501** (+12.7 at 82 games, trending H1) |
| **A1b** | Onto-threatened penalty, SEE/piece-value-gated | Not retried | Unstratified version #465 H0'd at −4.2 Elo. Stratified+SEE version still deferred; revisit after A1a resolves |
| **A1c** | `can_win_material` RFP loosener | Tested in bundle with A1a | Same #479/#501 fate. SPSA barely moved `RFP_CAN_WIN_DIV` (6 → 6). Value-add unclear — likely carried by A1a |
| **A2** | 2-bit threat-level history stratification | **H0'd as #480** | −3.3 Elo. The stratified history idea didn't compound on #463's binary-threat history. Shelved |
| **A3** | ProbCut gate on high threat state | **Merged (#481)** | **+7.03 Elo H1.** Now live in trunk as `PROBCUT_KING_ZONE_MAX` |

**Observations on Tier A outcomes:**

- A1 bundle (A1a+A1c) was surprisingly hard to land — original H0 at −0.5 suggests close to neutral but with unfavorable sign. Post-SPSA retest (#501) will clarify whether the small parameter shifts rescue it.
- A2 stratification DID NOT compound on #463's binary-threat-history gain. More history buckets ≠ better move ordering when the bucketing doesn't carry additional useful signal. Stratifying threats into {safe, pawn, minor, rook} was either too noisy or split the history budget too thin.
- A3 landed clean at +7 Elo. Simplest of the Tier A items.

## Tier B — novel, requires x-ray data surfaced to search

| # | Item | Status | Result |
|---|------|--------|--------|
| **B1** | Discovered-attack movepicker bonus via Path 1 fresh-compute | **In flight as #502** | +9.2 at 114 games, trending H1. Consuming `Board::xray_blockers`. First novel x-ray signal reaching SPRT |
| **B2** | Skewer detector movepicker bonus | Gated on B1 magnitude | Not started |
| **B3** | Raw delta-magnitude as extension signal | **Shelved** | Histogram data (67% in 5-12 range, 30% in 13-24, 2.7% in 25+) too uniform to be useful as tactical-complexity signal |

## Tier C — ambitious

| # | Item | Status | Result |
|---|------|--------|--------|
| **C1** | Trained threat-weight-magnitude oracle | Not started | Later. Novel, high variance |
| **C2** | NNUE accumulator delta as extension signal | **Adjacent variant in flight as #500** | `experiment/threat-mag-lmr` uses position-level `sum(\|threat_acc\|)` as LMR modulator (not the extension-on-delta framing of C2). +10.3 at 740 games, LLR 0.54 trending H1. If lands, the original extension-side framing remains untested as a follow-up |

## X-ray data paths

| Path | Status |
|---|---|
| **Path 1 — fresh compute at node entry** | Landed as `Board::xray_blockers`. ~3-5% NPS cost, correctness-isolated. Consumed by #502 |
| **Path 2 — delta-tagging at emission (bit-steal `is_xray` from `attacker_cp`)** | **Earned IF B1 (#502) lands >6 Elo.** If #502 H1s near +9, Path 2 refactor is justified — amortises the NPS cost to zero |
| **Path 3 — derived bitboards during emission** | Not scoped. threats.rs has been the #1 correctness hot-spot; each state addition is another fuzzer case |

## Suggested experiment order (live-updated)

The original ordering from the planning thread:

1. ~~A1a + A1c bundle~~ → tested (#479 H0, #501 post-tune in flight)
2. ~~2b rewrite NPS measurement~~ → merged (+10.0 Elo)
3. ~~A2 (2-bit history)~~ → H0'd (#480, −3.3 Elo)
4. ~~A3 (ProbCut threat gate)~~ → merged (+7.03 Elo)
5. ~~B1 via Path 1~~ → in flight as #502 (+9.2 trending H1)
6. Path 2 refactor + B2 + x-ray history stratification — **gated on #502 landing**
7. C2 (NNUE accumulator delta extension) — **in flight as #500** in LMR-modulator form; original extension-on-delta form still untested
8. C1 (weight-magnitude oracle) — not started

## What landed beyond the original plan

Several ideas emerged during execution that weren't in the original
Tier A/B/C list:

| Test | Branch | Elo | Status |
|---|---|---|---|
| #482 | `experiment/lmr-king-pressure` | +6.81 | Merged. Same king-zone signal as #466 NMP gate, applied to LMR |
| #484 | `experiment/futility-defenses` | +7.00 | Merged. `our_defenses` count as futility margin widener |
| #490 | `tune/v9-post-merge-r1` | +7.38 | Merged. Post-merge retune capturing ~35% retune bonus on top of the three +7 wins |

Sum of confirmed-landed Elo since the thread started: **~+34-47 Elo** depending on how much of #490's retune is attributed to which feature.

## Calibration data from executing this plan

- **"Trivial" +1-3 Elo predictions ran low.** A3 (+7.03), LMR king-pressure (+6.81), futility-defenses (+7.00), 2b rewrite (+10.0) all landed meaningfully higher than originally estimated.
- **Retune bonus is real and capturable.** Tune #490 banked +7.38 after three +7 wins. Rule of thumb: **budget a retune after every ~3 feature merges** to capture ~33% additional Elo from parameter rebalancing.
- **Small SPSA parameter shifts can still matter.** The #486 tune on stratified-escape-canwin moved params by <5% — I initially dismissed as "within noise" and submitted a duplicate SPRT without applying values. The correct interpretation: small shifts near SPSA's noise floor can still flip a borderline H0 feature to H1 because the *direction* is preserved. #501 retest pending to confirm.
- **A2 being H0 was a surprise.** History stratification by threat level seemed natural after #463's win, but 4-bucket partitioning didn't carry additional signal over 2-bucket. The useful signal is "any threat / no threat", not "pawn threat / minor threat / rook threat".

## What changed vs the pre-Hercules plan

- **2b rewrite**: "REVIEW — needs fuzzer verification" → fuzzer-clean after Hercules's bug fix. Merged at +10.0 Elo.
- **NPS-Elo economics**: V9 > V5 crossing removes the "NPS must close to merge" pressure. NPS wins now compound rather than gate.
- **Retune cadence discovered**: every ~3 merges captures ~33% extra via retune (tune #490 banked on top of #481/#482/#484).

## Updates

Update this doc as in-flight tests resolve:
- #500 threat-mag-lmr
- #501 stratified-escape-canwin (post-SPSA)
- #502 discovered-attack-bonus (B1)

Archive closed entries to a "Closed" section when they H1 and merge.
