# Search Audit — 2026-06-19

Six parallel Sonnet agents; each audited one subsystem against SF, Reckless,
Obsidian, Berserk, Alexandria, PlentyChess, Stormphrax. HEAD at 9ea7794.

Items ranked by confidence × estimated Elo.

---

## TIER 1 — Clear bugs (fix first, test at [-2,1] or [-3,3])

### B1. NMP verification re-triggers NMP (correctness bug)

`search.rs` verification search ~line 3607:

```rust
let v_score = negamax(board, info, beta - 1, beta, depth - r, ply, false);
```

`unmake_null_move()` has already run by this point, so `prev_was_null` (which
checks `board.undo_stack.last().mv == NO_MOVE`) is false inside the subtree.
NMP can fire again during its own verification, making the verification
circular — it no longer catches the zugzwang it was designed to detect.

All five reference engines solve this with a ply-barrier field:
- Reckless: `td.nmp_min_ply = ply + 3*(depth-r)/4`
- Alexandria: `td->nmpPlies = ss->ply + (depth-R)*2/3`
- Stormphrax: `thread.minNmpPly = ply + (depth-R)*3/4`

Fix: add `nmp_min_ply: i32` to `SearchInfo` (default 0), set it before the
verification call, add `ply >= info.nmp_min_ply` to the NMP gate, clear after.
~5-line change. Bounds: `[-2, 1]` (correctness fix, direction uncertain).

---

### B2. Wasted re-search: `do_shallower` when `reduction == 1`

When `reduction == 1` (LMR applied single-ply) and `do_shallower` fires, it
subtracts 1 from `new_depth`, setting `new_depth == lmr_depth`. The
subsequent "full-depth" re-search runs at the exact same depth as the
already-completed LMR search — wasted work, probably -1 to -3 Elo.

All reference engines guard: `if new_depth > lmr_depth { re-search }`.
One-line fix. Bounds: `[-2, 1]` (or `[0, 3]` — the re-search is the bug,
removal should help).

---

### B3. RFP return value not blended toward beta

Coda (`search.rs` ~line 3540):
```rust
return static_eval - margin;
```
All 5 reference engines blend toward beta:
- Obsidian/Stormphrax: `(eval + beta) / 2`
- Reckless: `beta + (estimated - beta) / 3`
- Alexandria: `(eval - margin + beta) / 2`

When `static_eval - margin >> beta`, Coda's return inflates TT entries and
misleads score-trend TM and aspiration windows. The cutoff is correct (> beta)
but the returned score is systematically over-optimistic. Fix: return
`(static_eval - margin + beta) / 2`. One-liner. Bounds: `[-2, 1]`.

---

## TIER 2 — Structural gaps (high confidence, probably positive)

### S1. Correction history never trains on all-nodes (downward errors)

Coda's update gate (~line 4694):
```rust
if !in_check && best_move != NO_MOVE && !best_move_noisy
    && best_score > alpha_orig   // ← blocks fail-low entirely
```

At fail-low nodes `best_score <= alpha_orig` — correction history never gets
trained on positions where static eval was over-optimistic and every move
failed below alpha. That's half the signal gone.

SF and Reckless both update on fail-low when the error direction is consistent
(eval was too high → fire with an upper-bound condition). Reckless gate:
```rust
if !(in_check || best_move.is_noisy()
    || (bound == Upper && best_score >= eval)    // over-pessimistic error
    || (bound == Lower && best_score <= eval))   // over-optimistic error OK
```
Remove the `best_move != NO_MOVE` requirement and add an upper-bound arm.
Estimated +2-4 Elo. Bounds: `[0, 3]`.

---

### S2. SEE capture pruning hard ceiling at depth 7

Coda gates SEE capture pruning on `depth <= tp(&SEE_CAP_DEPTH)` (default 7).
At LTC depths 12-25, ALL losing captures are searched unconditionally.

Reckless/SF use formulas that self-disable gracefully — at depth 16 Reckless
threshold is -2624 (near-impossible to prune). No hard ceiling needed.
Fix: raise `SEE_CAP_DEPTH` to 14-16, or remove the ceiling and let the
`-SEE_CAP_MULT*d` formula self-limit. Bounds: `[0, 3]` LTC 40+0.4.

---

### S3. Capture LMR missing 4 adjustments

Quiet LMR has 12+ adjustments; capture LMR has 2 (history adjustment only).
Missing from capture LMR, which ALL reference engines apply:
- `improving` (Obsidian, Berserk, Alexandria)
- `cut_node` (all five reference engines)
- `tt_pv` (Obsidian, Alexandria, Reckless)
- `tt_depth >= depth` (Obsidian, Berserk)

Fix: unify quiet/capture LMR into a shared adjustment block, differentiating
only the history source and divisor. Needs retune-on-branch (tree shape
changes). Estimated -2 to -4 Elo lost from missing these.

---

### S4. LMR hard floor at 0 blocks favorable extensions

Coda clamps reduction to minimum 0. All reference engines allow reduction to go
negative (= extend by 1 ply) when history is very favorable:
- SF: up to `newDepth + 2 + PvNode`
- Obsidian: clamp to `[1, newDepth + 1]`
- Reckless: explicit `lmr_extension` flag

Fix: allow reduction to go to -1 max, with the existing extension limit cap.
Estimated +1-2 Elo. Needs retune. Bounds: `[0, 3]`.

---

### S5. NMP gate uses raw static_eval without TT refinement

Coda always gates NMP on raw `static_eval >= beta`. Peers refine with TT:
- Reckless: if TT has upper-bound and `tt_score < eval`, use `tt_score` as
  `estimated_score` for the gate. If the position is already known to be weaker
  than eval, NMP is suppressed.

This catches cases where static eval was inflated but the TT already knows it.
One structural add. Estimated +0.5-1.5 Elo.

---

## TIER 3 — Queued experiments that were never executed

### Q1. SE all-node negative extension retune-on-branch (live open item)

SPRT #1202 tested removing the all-node `-1` negative extension: **H0 -1.7
but +17.2% bench shift.** A +17.2% bench shift on removal is a textbook
retune-on-branch signal — adjacent tunables were calibrated WITH this feature
and are now miscalibrated without it. The SE-cluster SPSA (SE_DEPTH,
DEXT_MARGIN_*, DEXT_CAP, SE_XRAY, ~7 params) was explicitly queued after this
test and **never run.** Run 1000-iter focused SPSA on the branch, then re-SPRT.

### Q2. SE cut_node propagation — LTC retry

SPRT #1201: passing parent `cut_node` to SE verification (vs Coda's hardcoded
`false`) came up **H0 +0.2 ±0.9 at 107K games** (LLR -2.95). Barely missed
H1, strong mechanistic argument for LTC benefit (SE fires more at deeper
depths, cut_node matters more). LTC retry was queued but **never submitted.**
Bounds `[0, 3]` at 40+0.4.

---

## TIER 4 — Medium-priority structural ports

| Item | Agent | Finding | Notes |
|------|-------|---------|-------|
| Near-ply contHist weight 1.1× → 2× | Move ordering | Finding A | Berserk's exact table structure; just wrong scoring weight |
| Move-count malus decay (`-K*n`) | Move ordering | Finding B | SF + Reckless both taper the malus across penalized quiets |
| Threat-aware capture history | Move ordering | Finding C | Reckless/Berserk add `[to_defended]` bit; doubles table size |
| Piece-relative threat key for 4D history | Move ordering | Finding D | Lower-noise signal; flat all-attacks is noisier |
| Missing next-ply cutoff count | LMR | Finding 4 | Stack field needed; SF + Reckless have it |
| `tt_score < alpha` LMR reduction increase | LMR | Finding 6 | Reckless: ~0.6 ply when TT had a prior fail-low |
| Cont-corr table undersized (768 vs 590K) | Misc | Finding 3 | Structural: only indexes opponent's last move, not own |
| NMP depth-scaled eval gate | NMP | N3 | Obsidian: `eval + 22*depth - 208 >= beta` |
| NMP TT-upper-bound skip | NMP | N4 | Stormphrax: skip if TT shows ceiling < beta |
| SE ttPv narrowing of singular_beta | Extensions | Finding 2 | Tested in wrong location (DEXT margin); isolate this |
| ASP_SCORE_DIV direction | QS/ASP | Finding 2 | 33378 vs Reckless 23660 vs SF 10208; focused tune |

---

## TIER 5 — Low/cosmetic

- `!tt_move_is_quiet` RFP guard ablation (no reference engine has it)
- LMP `stm_non_pawn != 0` safety guard (rare edge case in pawn-only endings)
- QS capture picker scale inconsistency with main picker
- Small ProbCut TT shortcut (SF positive-LB fast path)
- FH blend depth uncapped vs Reckless's min(depth, 8) cap

---

## Recommended first wave

1. **B2** (do_shallower guard) — one line, free Elo, zero risk
2. **B3** (RFP return blending) — one line, consensus from all 5 engines
3. **B1** (NMP verification barrier) — 5 lines, correctness fix
4. **S1** (correction history all-node update) — ~10 lines, highest Elo potential
5. **S2** (SEE_CAP_DEPTH raise/remove) — LTC SPRT
6. **Q1** (SE all-node negext retune) — fire the queued SPSA that never ran
