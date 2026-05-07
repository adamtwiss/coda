# Hobbes Search/Eval Review (2026-05-01)

CCRL Blitz #19 (~3715). Rust source at `~/chess/engines/hobbes-chess-engine/src/`.
Single-file negamax (`search.rs`, 1286 lines), three nested submodules
(`history.rs`, `correction.rs`, `movepicker.rs`). NNUE is plain HalfKA + king
buckets + horizontal mirror, no threats. Training-side covered separately
(`docs/hobbes_selfplay_case_study_2026-04-30.md`); this is search/eval only.

Hobbes is heavily SPSA-tuned (~270 tunables in `search/parameters.rs`) — many
items below have a tuning component bundled with the structural change.

## Tier-1 — High confidence, high leverage

### 1. Move-encoded prev-move correction-history (countermove + follow-up)

`search/correction.rs:24-25, 55-61, 73-74, 134-140`. Two correction tables
keyed not by a board-hash variant but by the *encoded 16-bit move* of the
previous and the previous-previous move:

```rust
fn prev_move_key(ss, ply, offset) -> Option<u64> {
    if ply >= offset { ss[ply - offset].mv.map(|mv| mv.encoded() as u64) }
}
self.countermove_corrhist  .update(us, ply-1 move, …);   // 1-ply
self.follow_up_move_corrhist.update(us, ply-2 move, …);   // 2-ply
```

Coda has pawn / non-pawn / minor / major / cont (cont_corr is piece+to of the
parent move only — Coda has no 2-ply move-keyed corrhist). Hobbes ports the SF
"countermove correction" + "follow-up correction" pattern and weights them
*above pawn / nonpawn* (`corr_counter_weight=100`, `corr_follow_up_weight=125`,
vs pawn=93, nonpawn=97 — see `parameters.rs:232-237`).

- **Mechanism**: index by `mv.encoded() as u64`, 4096-entry table, gravity
  update. Drops in cleanly next to existing `cont_corr` infrastructure.
- **Expected**: +4 to +10 Elo. SF, Hobbes, Berserk, Obsidian, Reckless all
  carry a 2-ply corrhist axis; we only have 1-ply (cont_corr). The combined
  Hobbes weight on prev-move corrhist (counter+follow_up = 225) outweighs
  pawn (93) — strong signal it's high-leverage.
- **Effort**: small. 2× `[i32; 4096]` per stm, plus weight tunables.

### 2. Stand-pat & fail-high lerp blending in QS

`search.rs:967-972, 1097-1099`. Two lerp gates we don't have:

```rust
// stand-pat: instead of `return alpha;` when stand-pat cuts, blend
if alpha >= beta {
    return lerp(alpha, beta, qs_stand_pat_lerp_factor()) // currently 50
}
// fail-high: blend the returned best_score toward beta on QS fail-high
if best_score >= beta && is_defined(best_score) {
    best_score = lerp(beta, best_score, qs_fail_high_lerp_factor()) // 50
}
```

Plus the same pattern at non-PV RFP cutoff (`search.rs:320`):
`return lerp(beta, static_eval, rfp_lerp_factor())` (33). Coda's
fail-high blending is a `(score*depth+beta)/(depth+1)` rule at non-PV depth
≥ 3 in main search — but we don't blend in QS at all and we don't blend at
RFP cutoff. SPSA midpoints (33, 50, 50) suggest these are real, not no-ops.

- **Expected**: +3 to +6 Elo from RFP+QS combined. Each individually modest.
- **Effort**: small. 3 single-line lerp insertions + 3 tunables.

### 3. Histories: from-square + to-square 1D tables

`search/history.rs:68-70, 86, 105-107, 798-823`. In addition to butterfly
(from×to), piece-to, capture, and 2-ply continuation, Hobbes adds two flat
`[i16; 64]` per-side tables: `from_history[stm][from]` and
`to_history[stm][to]`. Read into the quiet ordering score (history.rs:103-108)
and updated on cutoff (`search.rs:812-823`). SF has from-history; many engines
have neither.

- **Mechanism**: 4× `[i16; 64]` extra (1 KB total), bonus/malus same shape
  as quiet history. Adds two more move-ordering axes — cheap.
- **Expected**: +3 to +5 Elo. SF added from-square history for similar gain.
  The to-square axis is rarer (Hobbes-original?).
- **Effort**: small.

### 4. `cut_node` propagation through the entire negamax + cut-node-driven LMR/IIR

`search.rs:123-130, 347, 380-385, 619, 657, 671`. Hobbes' `alpha_beta` takes
`cut_node: bool`, propagates `!cut_node` into NMP and PVS-zero-window
re-searches (`!cut_node`), and uses it in three places we don't:

- **Cutnode TT-reduction** (`search.rs:380-385`): `if cut_node && depth >= 8
  && (no tt_move || tt_depth + 4 <= depth) { depth -= 1 }`. A second
  shallow-search reduction next to IIR, gated specifically on cut nodes
  with stale TT.
- **LMR cutnode reduction** (`search.rs:619`): `r += lmr_cut_node() *
  cut_node as i32` with weight 1894/1024 ≈ +1.85 plies on cut nodes. We
  have a flat reduction adjustment but not propagated `cut_node`.
- **TT cutoff cut-node guard** (`search.rs:221`): `(tt_score <= alpha ||
  cut_node)` — only allow TT-upper cutoffs at cut nodes (Alexandria
  pattern). Coda has the Alexandria pattern but ours is `node_type` based
  (`search.rs` references). Worth comparing.

`cut_node` is one of the highest-leverage missing search-side primitives
in Coda — multiple SF/Reckless/Obsidian gates depend on it.

- **Expected**: +5 to +12 Elo bundled. Cutnode-LMR alone is consensus.
- **Effort**: medium. Threading `cut_node: bool` through every recursive
  call, then adding the three gates. 1-2 day port + retune-on-branch
  required (LMR cut-node weight reshuffles other LMR axes).

### 5. `tt_pv` flag deep into LMR

`search.rs:212, 616-618`. Coda has the sticky tt_pv flag but uses it only as
a binary "reduce less" in LMR. Hobbes has *three* tt_pv-driven LMR adjustments:

```rust
r -= lmr_ttpv_base()  * tt_pv as i32;                              // 100
r -= lmr_ttpv_score() * (tt_pv && has_tt_score && tt_score > alpha) as i32; // 218
r -= lmr_ttpv_depth() * (tt_pv && has_tt_score && tt_depth >= depth) as i32; // 1181
```

The third (1181/1024 ≈ -1.15 plies when tt_pv ∧ tt_depth≥depth) is the
heaviest single LMR adjustment outside of cut_node. Coda's tt_pv-LMR is
either 0 or a fixed -1; the gradient is missing.

- **Expected**: +2 to +5 Elo, retune-on-branch eligible.
- **Effort**: small. Three new LMR tunables.

### 6. Singular-extension-driven LMR adjustment

`search.rs:632-635`:

```rust
if is_defined(tt_mv_score) && is_defined(singular_score) {
    let margin = tt_mv_score - singular_score;
    r += (lmr_se_mult() * (margin - lmr_se_offset()) / lmr_se_div())
            .clamp(0, lmr_se_max());
}
```

Track the score returned for the TT move at this node, compare to the
singular search's score, and use the gap as an LMR signal: when the TT move
is dominantly best (large margin), reduce non-TT moves more. Reckless does
this, Obsidian does this, SF does this — Coda doesn't. Cheap data flow
(stash one i32 between TT-move search and the rest of the move loop).

- **Caveat**: Coda's SE is fragile (multiple H0 -50 to -100 Elo experiments
  in `experiments.md` §2076-2092). This LMR adjustment is conditional on SE
  *working*, so it only helps if SE is healthy. Re-test after SE diagnosis.
- **Expected**: +2 to +4 Elo *if SE is healthy*; could be 0 with our broken SE.
- **Effort**: small (after SE is fixed).

## Tier-2 — Smaller or less certain

### 7. Hindsight extension (mirror of hindsight reduction)

`search.rs:285-293`:

```rust
// If parent was reduced AND the position is now WORSE than parent expected → extend
if depth >= hindsight_ext_min_depth()  // 1
    && td.stack[ply - 1].reduction >= hindsight_ext_min_reduction()  // 3
    && static_eval + td.stack[ply - 1].static_eval < hindsight_ext_eval_diff()  // -14
{
    depth += 1;
}
```

Coda has hindsight *reduction* (FEAT_HINDSIGHT confirmed −18 Elo without)
but not hindsight *extension*. **Already tried in the carve-out bundle:
H0 #866 individual + H0 #881 bundled** (per `experiments.md:7383`).

- **Recommendation**: Skip unless we revisit. Marked here so a future
  reviewer doesn't re-discover and re-test.

### 8. Alpha-raise depth reduction inside the move loop

`search.rs:741-745`:

```rust
if depth > alpha_raise_min_depth() && depth < alpha_raise_max_depth() && !is_mated {
    depth -= 1;  // applied AFTER score raises alpha, before next move
}
```

Coda has its own "alpha-raise reduction" (`experiments.md:1330` — H1 merged).
Worth diffing the two implementations:
- Hobbes reduces the **enclosing depth** (so future siblings search shallower)
- Coda's progressive alpha-reduce was H0 (`experiments.md:2629`) but a
  flat -1 was kept

Coda's version is specifically gated; Hobbes' version has min/max-depth
bounds (2..12) which we may not have. **Compare bounds; SPSA the gate.**

- **Expected**: +0 to +3 Elo (mostly already captured).
- **Effort**: small.

### 9. NMP eval-margin reduction with clamped ratio

`search.rs:337-340`:

```rust
let r = (5360                  // base
       + 279 * depth
       + 546 * (static_eval - beta).clamp(0, 1155) / 130) / 1024;
```

Coda's NMP R formula is `BASE_R + depth/DEPTH_DIV + (eval-beta)/EVAL_DIV`.
Mechanically the same shape, but Hobbes:
- **Clamps eval−beta to a max** (1155) before dividing — caps the bonus.
- Uses /1024 fixed-point, allowing very fine-grained SPSA on each weight.

Coda's NMP cutoff rate is 30% vs Reckless 57% (per
`docs/cross_engine_comparison_2026-04-25.md`). Worth verifying the eval-clamp
is the missing piece. SPSA over the four-tunable formula.

- **Expected**: +1 to +3 Elo (already-tuned regime).
- **Effort**: small (formula + 4 tunables).

### 10. NMP verification at depth ≤ 14 (low-depth shortcut)

`search.rs:351-364`. Hobbes splits NMP into:
- `depth <= 14 || nmp_min_ply > 0` → return raw score (no verification)
- otherwise → verification search; `nmp_min_ply = 3*(depth-r)/4 + ply`

Coda's verify-NMP gate is `VERIFY_DEPTH` (constant). The Hobbes split is
simpler — verify is reserved for the deepest nodes only. Worth checking
where Coda's threshold sits (the SF-style 12 vs Hobbes-style 14 vs Coda's
current value).

- **Expected**: +1 to +3 Elo.
- **Effort**: small (just retune the gate + threshold).

### 11. Lerped from-to vs piece-to history blend (instead of summed)

`search/history.rs:241, 278`:

```rust
lerp(from_to_score, piece_to_score, quiet_hist_lerp_factor())  // 43/100
```

Hobbes' quiet history is *interpolated* between butterfly (from-to) and
piece-to, not *summed*. The lerp factor is SPSA-tuned to 43% (i.e., weight
piece-to ~43%, butterfly ~57%). Coda sums them. The lerp formulation lets
SPSA find the optimum mix per axis (also done for capture history, lerp
factor 40/100 — `parameters.rs:158`).

- **Mechanism**: replace `quiet_score + cont_score + ...` with
  `lerp(from_to, piece_to, factor) + cont_score + ...` and tune the factor.
- **Expected**: +1 to +3 Elo from finding the right mix; could be 0 if our
  current sum is already near-optimal.
- **Effort**: small.

### 12. SEE-aware noisy split in movepicker (history-modulated)

`search/movepicker.rs:243-253`. Hobbes' good/bad noisy threshold is *driven
by capture history*:

```rust
let threshold = -entry.score / movepick_see_divisor() + movepick_see_offset();
//                          (44 SPSA)              (117 SPSA)
match threshold {
    t if t >  see::value(Queen) => false,    // very-bad → bad pile
    t if t < -see::value(Queen) => true,     // very-good → good pile
    _ => see(board, &mv, threshold, Ordering),
}
```

Coda's good/bad noisy split is a fixed SEE threshold (=0). Hobbes lets
high-history moves cross to "good" even if SEE is slightly negative, and
low-history losing captures get demoted further. This was specifically
SPRT-validated on Reckless / Obsidian — same pattern we found in the
caphist retune (`docs/caphist_retune_proposal_2026-04-19.md`).

- **Expected**: +2 to +5 Elo. Couples capture history magnitude into ordering.
- **Effort**: small. Requires our capture history to be on the right scale
  (per `caphist_retune_proposal`, this might already be in flight).

### 13. SE search-result side-channels

`search.rs:436-444`. Beyond positive extensions, Hobbes has four
*negative* SE outcomes:

```rust
} else if s_beta >= beta {
    return (s_beta * s_depth + beta) / (s_depth + 1);  // multi-cut, blended
} else if tt_score >= beta {
    extension = -3 + pv_node as i32;                    // 3-ply NEGATIVE ext (!)
} else if cut_node {
    extension = -2;
} else if tt_score <= alpha {
    extension = -1;
}
```

Coda has multi-cut and a single negative-ext path (per `experiments.md:2279`).
The −3-ply path on `tt_score >= beta` is unusually aggressive — a clear
Hobbes outlier. Likely SPSA-tuned for Hobbes' specific search shape.

- **Caveat**: SE is currently broken in Coda. Skip until SE is healthy.
- **Effort**: medium (after SE diagnosis).

### 14. `num_fail_highs` tracking → LMR penalty

`search.rs:254, 626, 734`. Hobbes increments `td.stack[ply].num_fail_highs`
on every beta cutoff and uses it in LMR:

```rust
r += lmr_fail_highs() * (td.stack[ply + 1].num_fail_highs > 2) as i32;
//   61/1024
```

If the *child* node has had >2 fail-highs (i.e., it's a cut node that's
been pounded), reduce *us* less aggressively. Two lines of bookkeeping +
one LMR tunable.

- **Expected**: +1 to +3 Elo.
- **Effort**: small.

### 15. Dynamic policy bonus from eval delta (parent-quiet bonus)

`search.rs:265-280`:

```rust
// Hindsight history: at every non-root non-singular node, if the parent's
// quiet move WORKED (static eval flipped favourably), give it an extra
// quiet-history bonus.
let value = dynamic_policy_mult() * -(static_eval + prev_eval);
let bonus = value.clamp(dynamic_policy_min(), dynamic_policy_max()) as i16;
td.history.quiet_history.update(!board.stm, &prev_mv, prev_pc, prev_threats, bonus, bonus);
```

Coda has *prior countermove bonus* (`PRIOR_COUNTERMOVE_BONUS`) but only on
beta cutoff. Hobbes' version applies an eval-driven bonus *every node*.
Novel mechanism — couples NNUE delta directly into policy.

- **Expected**: uncertain. +0 to +5 Elo. Eval-driven history is a known
  pattern but Coda's threat features may already capture this.
- **Effort**: small.

### 16. TT replacement: `depth + 4 > slot.depth` (we have 3)

`search/tt.rs:222`. Hobbes uses `depth + 4 > slot.depth`; Coda uses
`d > slotDepth − 3` (i.e., +3). Hobbes' larger margin keeps stale entries
longer. SPSA candidate, not a structural change.

- **Expected**: +0 to +1 Elo.
- **Effort**: trivial. SPSA the constant.

### 17. Material-phase eval scaling

`evaluation.rs:374-393`:

```rust
fn scale_evaluation(board, eval) -> i32 {
    let phase = scale_value_pawn() * pawns + scale_value_knight() * knights + ...;
    eval * (material_scaling_base() + phase) / 32768 * (200 - hm) / 200
}
```

Hobbes scales NNUE output by total material phase + halfmove clock. Coda
scales by halfmove only (`apply_halfmove_scale`, `search.rs:734-754`). The
material-phase scale handles "evals shrink as material falls" — endgames
get systematic damping. Stockfish, Reckless, Obsidian all do this. Five
SPSA-tuned phase weights (`parameters.rs:248-253`).

- **Expected**: +2 to +5 Elo.
- **Effort**: small (post-NNUE scaling, before search consumes the eval).
  Validate RMS doesn't shift the search-threshold calibration (per
  CLAUDE.md EVAL_SCALE warning).

## Tier-3 — Novel / experimental

### 18. Encoded-move correction-history slot uses the move *only*, not piece+from+to

`search/correction.rs:134-140`. The countermove/follow-up corrhist key is
`mv.encoded() as u64` — i.e., the 16-bit (from, to, flags) of the parent
move, with *no piece type*. Coda's existing `cont_corr` keys on
`piece × to`. Hobbes' choice is unusual; SF uses [piece][to]. May be a
deliberate compactness trade or a leftover. Worth profiling both keying
schemes if we port (#1 above).

### 19. LMR adjustment for "moving into a threatened square requires SEE>=0"

`search.rs:631`:

```rust
r += (is_quiet && to_threatened && !see::see(original_board, &mv, 0, Ordering)) as i32
       * lmr_quiet_see();   // 1313/1024 ≈ +1.28 plies
```

Quiet move where the destination is enemy-threatened *and* SEE<0 → reduce
much more (+1.28 plies). A blunder-detector embedded in LMR. Coda has
`SEE_QUIET_MULT*d²` for SEE pruning but no LMR-axis use.

- **Expected**: +1 to +3 Elo. Cheap (computes a SEE we already need for
  pruning) — only adds a single LMR axis.
- **Effort**: small.

### 20. `do_even_deeper` (3rd-tier LMR re-search re-extension)

`search.rs:649-654`. Coda has do_deeper / do_shallower (`search.rs:3074-3081`).
Hobbes has *three* tiers: `do_deeper`, `do_even_deeper`, `do_shallower`.
After an LMR-fail-high re-search, if score exceeds *both* margins, +2 plies
instead of +1:

```rust
new_depth += (score > do_deeper_margin) as i32;
new_depth += (score > do_even_deeper_margin) as i32;  // EXTRA tier
new_depth -= (score < do_shallower_margin) as i32;
```

- **Expected**: +0 to +2 Elo.
- **Effort**: trivial.

### 21. Per-square `cache.rs` bucket cache (pure NPS)

`evaluation/cache.rs` (referenced from `evaluation.rs:104-110`). Hobbes has
the standard "Finny" per-bucket-perspective cache, indistinguishable from
ours (`nnue.rs` bucket cache in Coda). Listed only to confirm parity — no
porting opportunity.

### 22. `add4`/`sub4` SIMD chunked accumulator updates on full-refresh

`evaluation.rs:147-177`. Refresh path processes 4 features at a time
(`add4`, `sub4`) before the remainder loop. Coda's refresh path is
per-feature (`recompute_threats_full` and PSQ refresh are scalar — already
flagged in `docs/reckless_commit_catalog_2026-05-01.md` items #793, #792).
**Already on the NPS queue** for Reckless port — verifying Hobbes uses the
same pattern strengthens the case.

## Items Coda already has (for completeness)

- IIR (`search.rs:372-377`) — same shape as ours
- Razoring (`search.rs:325-327`) — Coda **removed** razoring (-19.8 Elo,
  `experiments.md:3530`); don't restore.
- Hindsight reduction — Coda has FEAT_HINDSIGHT
- Aspiration windows w/ delta widening — same shape
- LMR base/divisor with separate quiet/noisy tables — same shape
- Improving from ply-2 with ply-4 fallback — same shape (`search.rs:1149-1156`)
- Mate distance pruning, RFP, futility, LMP, SEE pruning, ProbCut — all present

## Suggested SPRT order (by leverage / effort ratio)

1. **#1 prev-move corrhist** (countermove + follow_up) — small, +4-10 Elo
2. **#3 from/to history** — small, +3-5 Elo
3. **#4 cut_node propagation + LMR/IIR gates** — medium, +5-12 Elo bundle
4. **#2 QS lerp blending + RFP lerp** — small, +3-6 Elo
5. **#17 material-phase eval scaling** — small, +2-5 Elo
6. **#5 deep tt_pv-LMR adjustments** — small (retune-on-branch), +2-5 Elo
7. **#11 from-to/piece-to lerp instead of sum** — small, +1-3 Elo
8. **#12 history-modulated SEE noisy split** — small, +2-5 Elo
9. **#19 quiet-into-threatened + SEE<0 LMR axis** — small, +1-3 Elo
10. **#14 num_fail_highs LMR penalty** — small, +1-3 Elo
11. **#8 alpha-raise bounds diff** — small, +0-3 Elo
12. **#20 do_even_deeper** — trivial, +0-2 Elo

Hindsight extension (#7) is already H0 — skip. SE-dependent items (#6, #13)
wait on SE diagnosis.

## Notes

- Hobbes is heavily SPSA-tuned (270 tunables vs Coda's ~80). When porting,
  default values are SPSA midpoints — they will likely need a retune on the
  Coda branch (per `experiments.md` retune-on-branch methodology).
- Hobbes' search is single-file linear (no feature flags / ablation
  infrastructure). Porting one item at a time is straightforward.
- Hobbes uses no killers / counter moves at all — relies entirely on
  history-based ordering. Coda has killers/counter dead-coded with the
  same SF-pattern note. No action needed; Hobbes confirms the SF pattern.
- LMR `cut_node` weight is the heaviest single LMR axis in Hobbes
  (1894/1024). If we port `cut_node`, this is the calibration target.
