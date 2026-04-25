# Good/bad quiet split — v3 plan (2026-04-25)

**Status**: NOT YET IMPLEMENTED. Adam's lead 2026-04-25; queued for
greenlight when he's back. Implementing this is non-trivial enough
(movepicker structural change + new tunable + retune-on-branch) that
auto-execution wasn't appropriate.

## Background

SF / Reckless / Obsidian pattern: after good captures and good quiets,
defer low-history quiets so they fire AFTER bad captures (instead of
mixed in with all quiets).

## Prior attempts

| SPRT | Date | Branch | Result | Notes |
|------|------|--------|--------|-------|
| #199 | 2026-04-11 | `fix-good-bad-quiet-split` | -0.0 / 4444g (stopped, going to tune) | direct SPRT |
| #216 | 2026-04-11 | `fix-good-bad-quiet-split + tune` | **H0 -3.6 / 7680g** | retuned. Comment: "SF-only feature. Neither direct nor tuned helped." |

## Why retry now

A lot has changed since 2026-04-11:

1. **v9 architecture merged** (ea07d93, 2026-04-24) — threat features
   change which quiets are "low history" and what bad-capture pruning
   sees afterward.
2. **Force-more-pruning cluster merged** (#736, #737, #741, #745,
   #747, #752): +40 Elo from aggressive pruning at mid-depth.
   Low-history quiets are now eligible for much more aggressive
   pruning *if they reach the bad-noisy stage* — which is exactly
   what good/bad split would do.
3. **EBF dropped 1.8 → 1.68**, FMC up to 79.4-80.5%. The pruning
   landscape is different.
4. **Capture history magnitude fix** + **discrete capture LMR** etc.
   are all in trunk now — the move-ordering substrate the split
   relies on is materially better than it was 14 days ago.

The H0 at #216 may be specifically because the *old* aggressive
pruning regime didn't hit deferred quiets hard enough to cash in
on the deferral. Post-tune-750, that's plausibly different.

## Implementation sketch

**Files**: `src/movepicker.rs`, `src/search.rs` (tunable).

1. Add `Stage::BadQuiets` between `Stage::BadCaptures` and `Stage::Done`.
2. Add fields to `MovePicker`:
   ```rust
   bad_quiet_moves: [Move; 128],
   bad_quiet_scores: [i32; 128],
   bad_quiet_len: usize,
   ```
3. After `generate_and_score_quiets` finishes scoring, walk
   `self.moves` backwards. For each move with `score <
   QUIET_SPLIT_THRESH`, copy to the bad_quiet arrays and
   swap-remove from `self.moves`.
4. Stage flow:
   - `Stage::Quiets`: pick from (filtered) `self.moves`. Same as today.
   - `Stage::BadCaptures`: same as today.
   - `Stage::BadQuiets` (new): pick from `bad_quiet_*` arrays.
5. Add tunable `(QUIET_SPLIT_THRESH, -14499, -25000, -5000, 1000.0)`
   (SF default is -14499; range explored in old #199 tune).

## Test plan

1. **Direct SPRT** at default tunable values, `[-3, 3]` bounds (small-
   win class). If H1, merge.
2. **If H0 or marginal**: focused SPSA tune on branch — params
   `QUIET_SPLIT_THRESH`, `LMR_C_QUIET`, `LMR_C_CAP`, `BAD_NOISY_MARGIN`,
   `SEE_QUIET_MULT` (5 params, ~1500 iters). Apply tuned values, SPRT
   v2 vs trunk at `[0, 5]`.
3. **If still H0 with retune**: drop, log post-mortem with mechanism
   bucket per `feedback_h0_post_mortem_discipline.md`.

## Estimated path Elo

Per `next_100_elo_2026-04-24.md` ordering-cluster section:
- Direct: maybe +0 to +2 (last result was -3.6, regime change might
  be worth ~5 Elo of tailwind)
- With retune: +2 to +6, similar magnitude to other ordering retunes
  (e.g. cont-hist malus +6.5 with retune)

## Why I didn't auto-execute

1. Movepicker structural change is large, increases risk of subtle
   bugs (TT-move filtering, killer/counter dedup, evasion paths)
2. Old SPRT was H0 even with retune — direct retest without a
   regime-specific reason might just H0 again
3. Worth Adam's input on whether the `[-3, 3]` direct + retune-or-drop
   policy fits, or if he'd prefer to go straight to the focused
   tune + SPRT v2 path.

— Hercules
