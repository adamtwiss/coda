# CapHist-Defender Rebase: B1 Absorption Finding

**Date**: 2026-04-19
**Branch**: `experiment/caphist-defender-v2` (rebase of `8d0c1fc` onto current trunk)
**Status**: **Did NOT proceed with focused SPSA** — Titan's "investigate first" gate (doc §6, Step 1) tripped.

## What I ran

1. Cherry-picked `8d0c1fc` onto `feature/threat-inputs` (post-B1, post-tune #490, post-contempt-removal).
2. Cherry-pick clean — no conflicts in `src/movepicker.rs` or `src/search.rs` despite B1 touching the same files.
3. Built. Measured bench + first-move-cut on v9 prod net (`net-v9-768th16x32-w15-e800s800-xray.nnue`).

## Measurement — pre-B1 vs post-B1 caphist-defender behavior

| Branch / state | Total nodes | First-move-cut |
|---|---|---|
| **Caphist-defender on PRE-B1 trunk** (Titan's original measurement, 8d0c1fc) | 2,061,895 | **82.0%** |
| **Trunk pre-B1** (baseline Titan referenced) | 2,988,580 | 70.6% |
| **Trunk post-B1** (current `feature/threat-inputs` head) | 3,564,297 | 72.4% |
| **Caphist-defender rebased onto POST-B1 trunk** (this measurement) | 3,635,961 | **72.9%** |

## The key observation

Caphist-defender on pre-B1 trunk delivered:
- −31% bench vs its baseline (2.99M → 2.06M)
- +11.4 pp first-move-cut (70.6% → 82.0%)

Caphist-defender rebased onto post-B1 trunk delivers:
- ~0% bench vs post-B1 baseline (3.56M → 3.64M — slight regression within noise)
- +0.5 pp first-move-cut (72.4% → 72.9% — essentially no change)

**B1 absorbed the ordering signal that caphist-defender was providing.** This matches Titan's Scenario B / Scenario D from the proposal's risk section — the mechanical ordering improvement does not survive the rebase onto post-B1 code.

## Why this is consistent with the compensation theory

Both B1 (discovered-attack bonus) and caphist-defender provide additional discrimination between otherwise-identical captures. B1's discriminator: does this capture reveal a discovered attack? Caphist-defender's discriminator: is the victim defended?

For "sitting duck" captures — captures where MVV alone is ambiguous — both features tell the same underlying story: this capture resolves favorably because there's no defender / the attack structure favors us. Once B1 has tagged those, the additional defender bit on captHist provides no new signal.

Mechanically, it also roughly tracks with the `docs/ordering_coupled_pruning_2026-04-19.md` measurement: post-B1, first-move-cut rose +1.8 pp (70.6→72.4) and avg-positions² dropped −12.5%. Caphist-defender was capturing a large chunk of that before B1 existed; now B1 captures it first.

## What this means for Titan's proposal — corrected

**Initial read** (wrong): flat bench + flat first-move-cut means the feature is subsumed by B1, skip focused SPSA.

**Corrected read** (per Adam, 2026-04-19): pre-tune bench is not the right Go/No-Go signal. The retune-on-branch premise is that current trunk tunables OVER-PRUNE relative to the new ordering regime. Even a modest mechanical ordering improvement (+0.5pp first-move-cut) can open retuning headroom that bench alone doesn't reveal, because the current tunables are actively suppressing the feature's potential Elo.

The proper Go/No-Go is **whether focused SPSA moves parameters** toward the v5-direction (SEE_QUIET_MULT down, CAP_HIST_MULT down). If params don't drift, retune thesis is wrong → abandon. If params drift as predicted, retune unlocks the latent gain.

**Corrected recommendation: proceed with focused SPSA** (Titan's Step 2) on `experiment/caphist-defender-v2`. ~4-6 hours of fleet; clean Go/No-Go outcome regardless of direction.

## What the diagnostic told us is still valid

The underlying finding from `docs/move_ordering_understanding_2026-04-19.md` and the tunable-drift analysis — that v9's capture ordering is ~10pp behind v5 and SPSA has compensated with SEE_QUIET_MULT +150%, CAP_HIST_MULT +36%, BAD_NOISY_MARGIN +32% — remains true and useful. The diagnostic correctly identified capture ordering as the locus of the gap.

What the rebase shows is that **the specific fix (defender dim on captHist) isn't the mechanism to close it post-B1** — B1 already closed a meaningful chunk of the gap via a different discriminator, and the remaining gap isn't differentiated by defender status.

## What to try next instead

Ideas for closing the remaining capture-ordering gap that do NOT overlap with B1's discovered-attack signal:

1. **Threat-indexed captHist** (Reckless-style `[piece][to][victim][to_threatened]`): similar structure but threat-indexed, not simple defender. Reckless uses this; its semantics differ from pure defender bit because it's the full threat bitboard intersection, not just any-attacker. Worth benching on a rebase to see if the key distinction is "is this square in ANY threat structure" vs "is this square specifically defended by a lower-value piece."

2. **Capture-history widening via threat-level buckets** (per Titan's §9 reference to tunable anomalies): use SEE(capture) or MVV-class as a bucket dimension instead of the raw captured_pt. Separates "obviously good" from "trade-equal" from "marginal" captures into different history slots.

3. **Post-tune #521 re-measurement**: once #521 converges, re-check first-move-cut. The tune will have recalibrated for post-B1 tree shape; the ordering metric may shift without any new features.

4. **Audit the +7.2pp quiet-first-cut shift** from Titan's decomposition table (v9 22.9% vs v5 15.8%): v9's tree visits more quiet-only nodes where it actually cuts on the first quiet. This is a smaller locus than the −9.5pp capture shift, but easier to instrument since quiets have more tunable signals (pawn history, continuation history, king-zone ideas).

## Fleet state

- **#521** `tune/v9-post-b1-r1` still running (169/1000). Provides the post-B1 tunable landscape independently of caphist-defender.
- Rebased branch `experiment/caphist-defender-v2` pushed for reference. No SPRT or SPSA launched on it.
- No fleet spend on this investigation beyond the ~45s bench run.

## Companion to

- `docs/caphist_retune_proposal_2026-04-19.md` (Titan) — the proposal; this doc is its Step 1 result.
- `docs/move_ordering_understanding_2026-04-19.md` (Titan) — full investigation; still stands.
- `docs/ordering_coupled_pruning_2026-04-19.md` (me, earlier today) — the post-B1 measurement that foreshadowed this absorption.
