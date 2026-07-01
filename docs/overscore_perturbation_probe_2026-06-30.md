# Overscore perturbation probe + natural-subpopulation follow-up (2026-06-30)

Follow-up to `docs/overrate_eval_investigation_2026-06-30.md`. That doc
established the *what* (Coda overrates ~54% king-safety-motif positions
relative to LC0 truth, where SF does not) without explaining *why* the net
is keying off the wrong signal. This probe asks: if we structurally edit a
worst-overscored position (remove a piece type, relocate the king), does
the Coda-vs-SF eval gap shrink — and if so, which edits move it most?

**Headline result: the synthetic perturbation approach gave an unstable,
confounded signal. A natural-subpopulation follow-up (real positions,
no synthetic edits, true LC0 ground truth) REFUTES the one finding that
looked like it survived (bishop-specificity).** Net conclusion: this line
of attack doesn't isolate a single piece-type cause. See "Open / next" in
the parent doc — the why is still open.

## Part 1 — synthetic perturbation probe

### Method

`scripts/overscore_perturbation_probe.py`. For each sampled position from
`testdata/heldout_overrate_lc0_2023_06.tsv` (Coda-specific overrate
positions, see that file's README), apply a structural edit via
python-chess (remove bishop pair, remove all bishops, remove knight/rook
pair, remove queens, remove the enemy piece closest to / farthest from the
side-to-move's king, remove a random piece, relocate the side-to-move king
to the kingside corner) and re-evaluate with both Coda and a local
Stockfish via UCI `eval` (static, no search).

`gap(position) = coda_eval_stm - sf_eval_stm`. Since the heldout set's own
selection criterion already established SF-static is materially closer to
LC0 truth on this population, `|gap|` shrinking after an edit is read as
"the edit removed something Coda was mis-weighting relative to SF." This
is a proxy, not LC0 truth directly — synthetic edited positions have no
LC0 label to compare against.

`is_sane()` guards reject edits that produce adjacent kings or leave the
non-moving side in check (both crashed the SF subprocess on the first
unfiltered run; fixed by validity-checking before every engine call).

### Run 1 — worst-error tail (king-safety-filtered, n=250, mean baseline \|err\|≈914-1648cp depending on filter)

| perturbation | n | mean\|gap_orig\| | mean\|gap_pert\| | mean_delta |
|---|---|---|---|---|
| remove_closest_attacker_on_stm_king | 245 | 611.7 | 418.3 | **193.3** |
| remove_farthest_enemy_piece_CONTROL | 247 | 611.2 | 427.6 | 183.6 |
| remove_all_bishops | 193 | 596.6 | 442.2 | **154.4** |
| remove_bishop_pair | 193 | 593.8 | 497.6 | 96.2 |
| remove_knight_pair | 185 | 610.5 | 532.4 | 78.1 |
| remove_rook_pair | 238 | 611.1 | 570.2 | 40.9 |
| remove_queens | 155 | 593.8 | 553.6 | 40.3 |
| remove_random_piece_CONTROL | 246 | 611.5 | 585.4 | 26.1 |
| castle_stm_king_to_corner | 193 | 599.3 | 594.1 | 5.2 |

**The closest-attacker and farthest-enemy-piece CONTROL converge** (193.3
vs 183.6) — proximity to the king is not what's driving the shrink. Single-
piece removal of *any* kind, on this extreme tail, shrinks the gap by a
similar large amount — a confound, not a king-safety signal. (The
random-piece control is smaller, 26.1, but still nonzero and in the same
direction.)

### Confound diagnosis

This worst-error tail sample (mean baseline \|err\| up to 1648cp) is
contaminated by extreme outliers — spot-checked one: a move-3 opening
position (`coda=-1455cp`) is a Coda eval bug unrelated to king safety,
caught by the crude `uncastled_center` heuristic flagging a developmentally
normal undeveloped king. Removing *any* single piece from a position this
far off introduces a large, unrelated free-material imbalance that is
itself a large shock to both engines' evals — regression to the mean on
an extreme tail, not a genuine causal signal.

### Run 2 — clean band (err 150-450cp, fullmove≥10, n=250, representative random sample not worst-N)

| perturbation | n | mean\|gap_orig\| | mean\|gap_pert\| | mean_delta |
|---|---|---|---|---|
| remove_closest_attacker_on_stm_king | 249 | 216.4 | 471.1 | **-254.6** |
| remove_random_piece_CONTROL | 249 | 216.8 | 463.3 | -246.5 |
| remove_farthest_enemy_piece_CONTROL | 248 | 216.2 | 460.9 | -244.7 |
| remove_rook_pair | 235 | 213.5 | 285.1 | -71.6 |
| remove_knight_pair | 178 | 202.1 | 270.4 | -68.4 |
| remove_queens | 155 | 194.2 | 260.7 | -66.4 |
| remove_all_bishops | 184 | 200.6 | 297.8 | **-97.2** |
| remove_bishop_pair | 186 | 204.2 | 243.7 | -39.5 |
| castle_stm_king_to_corner | 195 | 211.9 | 247.5 | -35.6 |

**Sign flipped entirely** vs Run 1 — single-piece removal now *worsens*
the gap by ~250cp, and again closest≈farthest≈random (-254.6 / -244.7 /
-246.5), confirming this is a free-material-imbalance artifact, not a
king-proximity effect, in both directions and both samples.

### What replicated across both runs

1. **Bishops are consistently the largest-magnitude mover among the
   piece-pair-type removals** (knight/rook/queen), regardless of sign:
   `remove_all_bishops` 154.4 (run 1) / -97.2 (run 2), both the largest
   magnitude in their respective table, ahead of knight/rook/queen pair
   removals (78.1/40.9/40.3 and -68.4/-71.6/-66.4 respectively).
2. **King-to-corner relocation is consistently the smallest-magnitude
   perturbation in both populations** (5.2 / -35.6) — moving the king
   alone, independent of any material change, does almost nothing to the
   gap. This argues the blind spot isn't really about *king position*
   per se (consistent with the bucket-flat result in the parent doc) but
   something that correlates with material/piece-type composition.

### Why this isn't conclusive

The bishop finding is a *relative magnitude* ranking that held in both
runs, but neither run isolates bishops from the single-piece-removal
confound — both pair-removal numbers still ride on top of "remove 2
pieces of some type" rather than a controlled comparison. A synthetic
edit can't cleanly separate "removing a bishop" from "removing 2 pieces
of value ~330cp," and the sign flip between runs shows the synthetic
approach is not even directionally stable. This motivated the
natural-subpopulation follow-up below — test the bishop hypothesis
without injecting any edit at all.

## Part 2 — natural-subpopulation comparison (the follow-up)

### Method

`scripts/overscore_natural_subpop.py`. Instead of editing positions,
harvest a large **unfiltered, natural** sample directly from a raw T80
binpack via `coda eval-dist --csv` (which already emits
`fen,white_result,coda_eval_white_cp,lc0_score_white_cp,game_id` — true
LC0 ground truth per row, no SF proxy needed). Bin by **total piece
count** (8 bands of width 4, 2-32) and, within each band, split into
"has ≥1 of piece-type X on the board" vs "has none of piece-type X",
for X ∈ {bishop, knight, rook, queen}. Compare mean `|coda_eval_white -
lc0_white|` between the two natural subgroups.

Controlling on **total piece count** rather than editing means a
no-bishop position in the 18-21 band has the same overall material
density as a has-bishop position in that band (the missing bishops are
naturally replaced by other material in real games) — no synthetic
material-imbalance shock, and a real LC0 label for every position.

Sample: 300k quiet positions from
`/training/sf/test80-2024-06-jun-2tb7p.min-v2.v6.binpack` (a different
month than the June-2023 heldout overrate set, so this isn't circular),
`min-fullmove≥5`, optionally restricted to `|lc0|≤600` to mirror the
parent investigation's "balanced position" filter.

### Result: bishop-specificity does NOT replicate

Unfiltered (271k positions), mean\|err\| delta (has-X − no-X) by band:

| band | bishop | knight | rook | queen |
|---|---|---|---|---|
| 2-5 | -81.5 | -87.7 | -57.6 | +456.7 (n_has=1905, noisy) |
| 6-9 | -53.3 | +28.6 | +53.4 | +95.4 |
| 10-13 | +19.6 | +43.9 | -0.2 | +1.6 |
| 14-17 | +8.3 | +20.2 | +53.8 | +85.8 |
| 18-21 | +41.0 | +52.2 | +98.6 | +48.4 |
| 22-25 | +40.8 | +50.4 | +51.1 | +39.2 |
| 26-29 | +59.9 | +43.8 | (n<20) | +27.6 |

Restricted to the balanced band (\|lc0\|≤600, 247k positions, the more
apples-to-apples comparison vs the parent investigation), deltas shrink
across the board but the **cross-piece-type pattern is unchanged**:

| band | bishop | knight | rook | queen |
|---|---|---|---|---|
| 2-5 | +0.9 | +0.5 | -4.5 | +12.9 |
| 6-9 | +1.5 | +18.4 | +1.8 | +3.4 |
| 10-13 | +7.0 | +1.0 | -3.9 | -4.4 |
| 14-17 | +3.7 | +8.6 | +3.9 | +11.4 |
| 18-21 | +16.6 | +14.3 | +24.4 | +17.8 |
| 22-25 | +18.1 | +12.5 | -0.8 | +15.6 |
| 26-29 | +20.0 | +10.1 | (n<20) | +6.5 |

**Bishop is not the standout in either table.** Rook (18-21: +98.6 / +24.4)
and knight (multiple bands) are comparable or larger. The pattern that
*does* hold across all four piece types: presence correlates with
**larger** error at high material counts (mid/late-opening through
early-middlegame, 18-29 pieces) and **smaller or negative** delta at very
low material counts (2-9 pieces, deep endgame) — but this is a generic
"more piece-type diversity → harder static eval, independent of which
type" effect, not a bishop-specific one. The relative-magnitude ranking
that survived both perturbation runs (Part 1) does not survive contact
with natural data.

### Conclusion

The synthetic perturbation probe's bishop finding was itself a confound
of the single-piece-removal artifact — bishops happen to average a value
between knights/rooks, so a pair-removal of "two ~330cp pieces" lands at
a magnitude that's hard to distinguish from "two ~320cp pieces" (knights)
without a much larger, controlled sample. The natural-subpopulation
comparison, which doesn't have this artifact and uses true LC0 labels,
shows no special bishop effect. **The king-safety blind spot identified
in the parent investigation is not explained by bishop presence
specifically** — it's more likely a general material/piece-type-diversity
interaction with dynamic factors (the M1/M2 king-safety motifs in the
parent doc's taxonomy), not isolable to one piece type via either method
tried so far.

## Tools produced

- `scripts/overscore_perturbation_probe.py` — synthetic structural-edit
  probe (Part 1). Useful for future "does removing X change the gap"
  questions, but treat single-piece-type results with caution given the
  demonstrated confound — always include closest/farthest/random
  controls and a clean (non-extreme-tail) sample, not just the worst-N.
- `scripts/overscore_natural_subpop.py` — natural-subpopulation
  comparison (Part 2). Reusable for any future piece-type or material-band
  hypothesis: point it at a raw T80 binpack (or a pre-generated `coda
  eval-dist --csv` output via `--csv ... --reuse`) and it bins + compares
  automatically. No SF dependency — uses the binpack's own LC0 label.

## Open / next

- The motif-level taxonomy in the parent doc (M1 enemy-heavy-piece-attack,
  M2 own-king-stuck-in-center) is still the best lead — it's about
  *dynamic* threat/initiative signals, not static piece-type presence.
  A natural-subpopulation comparison on a *threat-derived* feature (e.g.
  Coda's own threat-accumulator firing count, or attacker-count near the
  king) rather than raw piece-type presence would be the natural next
  step, reusing this script's binning machinery.
- The corrective-data lever (concentrated king-safety/dynamic-motif
  oversampling, ranked #1 in the parent doc) is unaffected by this
  result — it was never bishop-specific, it targets the motif directly.
