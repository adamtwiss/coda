# Capture Ordering — Cross-Engine Deep Dive (2026-04-20)

Scope: survey every top engine source we have for **capture/noisy move scoring** and **good/bad capture split**. Goal: find techniques beyond the `MVV×K + captHist[piece][to][captured]` pattern Coda uses today. Look specifically for (a) additional captHist dimensions (b) dynamic SEE thresholds (c) anything that reshuffles captures after a first-move fail-high miss.

Coda v9 baseline for comparison:
- captHist: 3D `[piece][to][captured]`
- Score: `MVV × 16 + captHist`
- Good/bad split: static `SEE ≥ 0`
- Movepicker order: TT → good captures → quiets → bad captures

## Engines surveyed

| Engine | CaptHist key | MVV weight | Good/bad SEE threshold | Notes |
|---|---|---|---|---|
| Stockfish | `[piece][to][captured]` (3D) | `7 × PieceValue[captured]` | Static SEE | Simpler than Coda. Orders via sophisticated quiet scoring instead. |
| Coda v9 (trunk) | `[piece][to][captured]` (3D) | `16 × SEE_val[captured]` | Static SEE ≥ 0 | Baseline. |
| RubiChess | `[piece][to][captured]` (3D) | `mvv \| lva` (bitwise OR) | Static | Classical. |
| Alexandria | `[piece][to][captured]` (3D) | `16 × SEE_val[captured]` | **Dynamic: `-score/32 + 236`** | Same dims as Coda; dynamic SEE. |
| Caissa | `[color][piece][captured][to]` (4D) | tunable × piece index | Tiered; skips SEE when `attacker < victim` or `attacker == victim` | stm-indexed. Three-tier good/bad split (Winning / Good / Bad). |
| Stormphrax | `[from][to][captured][defended]` (4D) | `see::value(captured)` + `captHist/8` | Static SEE | **`defended` = target square attacked by any enemy.** Matches our `experiment/caphist-defender` branch. |
| Reckless (prior notes) | `[stm][piece][to][captured][to_threatened]` (5D) | `16 × SEE_val[captured]` | Static | `to_threatened` bit — same signal as Stormphrax `defended`, different name. |
| Halogen | `[stm][piece][to][captured]` (4D) | **none — captHist only** | **Dynamic: `-73 - score × 51/1024`** | Radical. MVV is learned by history table, not hardcoded. |
| Quanticade | `[pieceFrom][captured][from][to][from_threat][to_threat]` (6D, ~2.5M entries) | `tunable_piece_values × MO_MVV_MULT` | **Dynamic: `-SEE_THRESH - score/SEE_HIST_DIV`** | Most elaborate. Fully threat-indexed plus from+to. |

Obsidian / PlentyChess covered elsewhere; both use 3D captHist with static SEE.

## Patterns observed

### Extra captHist dimension (past Coda's 3D)

Four engines add a single bit / small dimension:

- **Stormphrax `defended`** (target attacked by enemy). 4D.
- **Reckless `to_threatened`** (same signal). 5D with stm.
- **Halogen / Caissa add `stm` only** (no threat bit).
- **Quanticade adds `from_threat × to_threat` AND keeps from+to** (6D, ~2.5M entries).

The `defended`/`to_threatened` bit is the most consistent extra dimension across threat-aware engines. Coda's `experiment/caphist-defender` branch implements exactly this; it is under full 18-param retune right now.

### Dynamic SEE threshold — score-conditioned good/bad split

Three engines **do not** use a static `SEE ≥ 0` cutoff. Instead, the threshold that decides whether a scored capture is "good" scales with the capture's own score:

```
Alexandria:  thresh = -score / 32 + 236
Halogen:     thresh = -73 - score × 51/1024
Quanticade:  thresh = -MO_SEE_THRESH - score / MO_SEE_HIST_DIV
```

Interpretation: a capture with strong captHist credentials gets an easier SEE threshold (more negative → harder to fail). A capture with poor captHist has to clear a higher bar before being treated as "good". High-score moves get ordered earlier AND tolerated at worse SEE.

This is **not** currently in Coda. Orthogonal to the `defended` dimension.

### MVV weight varies 0–16×

- Halogen: no MVV at all. The captHist learns the victim preference.
- Stockfish: 7× (actual centipawn values).
- Coda/Reckless/Alexandria: 16×.
- RubiChess: bitwise OR with LVA.

Halogen's position implies that captHist alone is sufficient to recover MVV ordering after training — the learning signal pushes captures of valuable pieces high. Coda's strong explicit MVV suggests captHist is not trusted as primary signal.

### Post-factum re-ordering after first-move miss

**None** of the engines surveyed re-order captures after the first good capture fails to cut. Once scored, order is fixed. Failures are addressed structurally (better dynamic SEE split, richer captHist dimensions) rather than reactively.

## Recommendations for Coda

Ordered by expected ROI vs effort:

### Tier 1 — in-flight

- **`experiment/caphist-defender` + retune** (in progress). Adds `defended` bit per Stormphrax/Reckless. Target: close the 9.5pp capture-first-move-cut gap v5 → v9.

### Tier 2 — new experiments worth a branch

- **Caissa 3-tier good/bad split (cheapest).** Replace the binary `SEE ≥ 0` cutoff with a tiered classifier: `attacker < victim` → always good (skip SEE), `attacker == victim` → always good (skip SEE), else → SEE test. Orthogonal to the defender bit and to dynamic SEE — works on MVV-LVA attack geometry, not history. Trivial code change. Expected mechanism: saves SEE calls on obviously-winning and trade-equal captures, classifies them as "good" even when SEE would mark them negative due to recapture sequences deep in the exchange. Worth running as a standalone SPRT on post-defender trunk before stacking dynamic SEE on top, to avoid two untested capture changes at once.

- **Dynamic SEE threshold for good/bad split.** Replace `see_ge(0)` in movepicker with `see_ge(-BASE - score × K / 1024)`. Seen in 3 independent engines (Alexandria/Halogen/Quanticade). Tunable constants — small branch, easy SPRT. Orthogonal to defender bit — stack on top.
  - **Prior attempt (2026-04-03):** `fix-dynamic-capture-see-v3` and `fix-dynamic-see-threshold` both used `see_ge(m, -score/32)`. Neither merged. Two issues in that formulation that the engines surveyed here avoid:
    1. **No baseline term.** Alexandria adds `+236`; Halogen uses `-73 -`. A score-0 move should still get a slightly easier threshold than 0 (these are still captures worth trying). Our old formula collapsed to static SEE at score=0.
    2. **Our `score` was MVV×16 + captHist — dominated by MVV.** A queen capture gets score ≈ 14400 from MVV alone, forcing threshold ≈ -450 even with zero history. The signal being used as "confidence" was really just "is the victim big", which SEE already knows. Halogen's formulation works because their `score` is captHist-only — genuinely new information.
  - **Revisit design:** use `-BASE - captHist/K` (captHist only, not full score), or subtract a known MVV component from `score` first. Combine with the defender bit so captHist carries threat information too.

- **Halogen-style captHist-only ordering.** Ablation: drop MVV from scoring and see if captHist alone reproduces MVV. Probably H0s against main as standalone, but informative — if we're within 5 Elo we know captHist is rich; if we lose 20 Elo we know our captHist is under-trained vs Halogen's.

### Tier 3 — structural, bigger lift

- **Full-threat captHist (Quanticade 6D).** Adds `[from_threat][to_threat]` to captHist — matches what we already do for main quiet history. Memory: 3D→6D is 4× (threat bits × 2 × 2). Only justified if Tier 1+2 leave clear headroom.

## Notes on interactions with Coda's v9 work

- The ordering gap narrative in `move_ordering_understanding_2026-04-19.md` locates the loss in **capture ordering at non-TT nodes** (9.5pp drop vs v5). Of the six techniques surveyed, three attack exactly this: `defended` bit, dynamic SEE threshold, and stm-indexing. All three stack.

- Quanticade's 6D table is the maximalist version; we already have the data structure (4D threat-aware main hist). The unknown is whether the captHist *signal* is strong enough per-cell once you fan out 4× — Coda's v9 has less training than Quanticade and captHist updates are already sparse.

- Halogen's "no MVV" result is worth knowing but not directly actionable: our captHist is younger and less trusted. Revisit after defender + dynamic SEE ship.

## Companion docs

- [`move_ordering_understanding_2026-04-19.md`](move_ordering_understanding_2026-04-19.md) — locus of v9 ordering loss.
- [`caphist_retune_proposal_2026-04-19.md`](caphist_retune_proposal_2026-04-19.md) — in-flight defender-bit retune (matches Stormphrax/Reckless).
- [`ordering_coupled_pruning_2026-04-19.md`](ordering_coupled_pruning_2026-04-19.md) — why fixing ordering may need pruning retune.
