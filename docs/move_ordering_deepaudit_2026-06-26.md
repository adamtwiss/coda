# Move Ordering & History — Deep Structural Audit (2026-06-26)

Second-pass, read-only audit of `src/movepicker.rs` and the history
update sites in `src/search.rs`. Scope excludes correction history and
pruning thresholds (audited elsewhere). North star: the measured v9
first-move-cut gap (71–72% vs v5's 82%), `docs/move_ordering_tracking.md`.

## Headline

**Move ordering is broadly healthy and heavily worked-over.** The 4D
threat-aware main history, cont-hist ply set (1/2/4/6 with read-side
weighting), pawn history, capture history, dynamic SEE good/bad split,
killers-removed quiet ordering, and the bonus/malus symmetry are all
consensus-aligned or intentionally-carved. The big v9 first-move-cut
gap is **mechanical** (v9's flatter eval distribution — median gap_n
1.06σ vs v5's 1.79σ), not an ordering-code bug. This was the conclusion
of the 2026-04-19 investigation and nothing in this pass contradicts it.
The headline capture-ordering fix (caphist `defended`/`to_threatened`
bit, Stormphrax/Reckless structure) was **built, tested, and absorbed
by B1** — do NOT re-propose it.

That said, this pass found **three genuine residual divergences/
asymmetries** that are concrete, untested in their current form, and
cheap to SPRT. None are correctness bugs (no sign errors, no
indexing/threat-key mismatches — read/write threat keys are symmetric
after the C8 fixes; capture/quiet bonus+malus fire on the right move
sets). They are calibration/consistency gaps.

## Ranked findings

| # | Site | What's wrong / divergent | Cross-engine evidence | Severity |
|---|------|--------------------------|-----------------------|----------|
| 1 | `movepicker.rs:1265-1283` (QMovePicker scoring) | **QS capture ordering runs a different scale than main search.** QS scores captures `see_value(victim)*10 - see_value(attacker) + captHist` (MVV-LVA, ×10, *with* LVA, full captHist). Main search (`movepicker.rs:532` / `mvv_lva()`) scores `see_value(victim)*MVV_CAP_MULT(16) + captHist` (MVV-only, ×16, *no* LVA). captHist is the same magnitude in both, so its **relative** weight vs MVV is ~1.6× higher in QS, and QS additionally applies an LVA term that main search deliberately dropped. The QS path was never migrated when main search moved to the tunable MVV-only scheme. | Reckless uses a **single** capture-scoring formula `16*value + noisy_history` for both main and QS (`movepick.rs:172`). Stockfish/Obsidian/Alexandria likewise share one capture scorer across QS and main. Coda is the only one with a forked QS scale. | **Medium-high** |
| 2 | `search.rs:4745` (capture malus on beta cutoff) | **Capture malus is the only beta-cutoff history update missing numFailHighs scaling.** Quiet bonus (`:4618`), quiet malus (`:4622`), and capture bonus (`:4726`, added by #1054) all multiply by `1 + scale_factor*10/NFH_DIV`. `cap_malus = capture_history_malus(depth)` does not. Net effect: on multi-fail-high nodes, the cutoff capture's history is boosted but the *failed* captures' penalty is not → slow inflation of capture-history magnitudes relative to quiets, and an asymmetry vs the carefully-tuned quiet path. | #1054 explicitly added NFH scaling to the capture *bonus* "to match quiet bonus path enrichments" but stopped there; the malus site was not touched. Symmetry argument is Coda-internal (Starzix-derived NFH counter). | **Medium** |
| 3 | `search.rs:3215` (TT-cutoff cont-hist malus) | **Hardcoded magic malus `-min(155*depth, 385)`**, not wired to `history_malus()` or any tunable, and the 385 cap is ~4.5× weaker than the in-tree cont-hist malus (`HIST_MALUS_MAX≈1730`). It writes the opponent's-quiet cont-hist on TT cutoffs (Alexandria pattern). Frozen — SPSA can't touch it, and it's on a scale inconsistent with every other cont-hist write. | Alexandria's value; the magnitude was ported but never re-scaled to Coda's tuned malus formula. | **Low-medium** |
| 4 | `movepicker.rs:534` (dynamic good/bad SEE split) | `see_threshold = -capt_hist / 18` — divisor is **hardcoded, not tunable**, and there is **no baseline term** (collapses to static SEE≥0 at captHist=0). The cross-engine survey (`docs/capture_ordering_crossengine_2026-04-20.md`) noted baselines matter (Alexandria +236, Halogen -73). | Alexandria/Halogen/Quanticade all use a baseline + score term. **Caveat:** baseline-add was already tried (#543 -75, #556 -125) and H0'd. Only the *divisor-as-tunable* angle (so SPSA can find `/18`) is fresh. | **Low** (partly tried) |
| 5 | `search.rs:4631` (comment) | Stale comment: "Ply-1 at full bonus, plies 2/4/6 at half bonus (Obsidian pattern)" but the code applies **uniform** `ch_bonus = bonus` (the B1 change). Documentation rot only — read-side weighting `[cm,cm,1,1]` lives in the picker, not here. | n/a | **Cosmetic** |

## Notes on things that look wrong but are NOT

- **Read-side cont-hist weights `[cm,cm,1,1]` (plies 1,2 at CONT_HIST_MULT≈3; plies 4,6 at 1×) while updates write uniform magnitude.** This is correct SF-style read-side weighting, calibrated by the #1366 cont-hist cluster (+4.7). Not a bug.
- **Main-history threat key symmetry.** Picker reads with `self.threats` = the search's `enemy_attacks` (passed at `search.rs:3882/3884`); writes use the same `enemy_attacks`. Evasion read=write symmetry was the C8 #19 fix and holds. No 4D-index mismatch.
- **`captures_tried` "minus 1" exclusion of the cutoff capture** (`search.rs:4746`) is correct: the cutoff capture is the last element pushed (pushed pre-recursion at `:4251`), bonus applied separately. The `n<32` cap edge case is benign.
- **Killers/counters removed** — consensus-aligned (SF/Reckless removed them). Not a gap.

## Proposed changes & test plans

**Finding 1 — unify QS capture scale (highest expected value).**
Replace QMovePicker's capture branch with the shared `mvv_lva()` +
`capt_hist_score_static()` used by the main picker (MVV-only ×
`MVV_CAP_MULT`, no LVA). Keep the `10000 +` evasion offset. This puts
QS capture ordering on the SPSA-tuned scale and removes the forked LVA
term. QS is a large node fraction and capture ordering is the
documented v9 weak spot, so this is the most leveraged item.
- **Bench:** changes (QS reorders captures). Re-measure on branch.
- **Retune:** not required first pass — it *adopts* the already-tuned
  main scale rather than introducing a new constant. A follow-up
  `MVV_CAP_MULT` confirmation tune is optional.
- **SPRT:** `[-2, 1]` STC 10+0.1 (it's a consistency/non-regression
  change with plausible small upside; could promote to `[0,3]` if STC
  signals positive). Run STC first per gating policy.

**Finding 2 — add NFH scaling to capture malus.**
Mirror the capture-bonus line: `cap_malus = raw_cap_malus +
raw_cap_malus * scale_factor * 10 / NFH_DIV_10X`. Closes the last
asymmetric history-update site.
- **Bench:** changes. **Retune:** no.
- **SPRT:** `[-2, 1]` STC. **Caution:** #1068 showed capture-malus
  *magnitude* is sensitive (a 3× port lost −39.6). NFH scaling is a
  much milder multiplier (≈1.0–1.75× on cascade nodes), but treat a
  clear regression as "malus is already at the right magnitude, leave
  it." Cheap, ~3-line change.

**Finding 3 — rescale / expose the TT-cutoff cont-hist malus.**
Two options: (a) replace `-min(155*d, 385)` with the existing
`history_malus(depth)` (or a fixed fraction of it) so it lives on the
tuned scale; or (b) expose `TT_CUT_CONTHIST_MALUS_{MULT,CAP}` tunables.
Prefer (a) first (no new knobs — the loose-knobs doc warns against
param creep).
- **Bench:** changes slightly (low firing rate — `score_above_beta &&
  ply≥2`). **Retune:** no.
- **SPRT:** `[-2, 1]` STC. Low expected magnitude; bundle with Finding
  2 if fleet-constrained (both are cont-hist/capture malus-scale
  consistency fixes).

**Finding 4 — make the good/bad SEE divisor tunable (optional).**
Expose `SEE_GOODCAP_DIV` (default 18) so SPSA can probe it; do NOT add a
baseline (already H0'd twice). Low confidence — only worth it if
bundled into a routine retune, not as a standalone SPRT.
- **Bench:** unchanged at default 18. **SPRT:** none standalone; let a
  `--core` tune move it.

**Finding 5 — fix the stale comment.** Doc-only, no SPRT.

## Suggested execution order

1. **Finding 1** (QS scale unification) as a standalone branch — highest
   leverage, targets the documented capture-ordering locus.
2. **Findings 2 + 3 bundled** (capture/cont-hist malus-scale consistency)
   — both small, both malus-scale, share a test narrative; bundling
   amplifies SNR per the non-regression-bounds guidance.
3. Finding 4 folded into the next `--core` retune; Finding 5 with any
   doc-touch commit.

## What this audit did NOT find

No sign errors, no white/black or attacker/defender asymmetry, no
missing/duplicate update site, no stale-entry read, no 4D-index or
threat-key mismatch. The capture-history `defended`-bit lever is closed
(absorbed by B1). The ordering code is clean; the residual gap to v5 is
eval-geometry, not code.
