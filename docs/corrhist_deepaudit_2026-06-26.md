# Correction History — Deep Structural Audit (2026-06-26)

**Owner:** atlas (background audit agent)
**Scope:** the corrhist subsystem in `src/search.rs` — the source tables
(pawn, white-NP, black-NP, continuation; minor/major dropped), the weight
blend, the proportional-gravity update, the clamps, and the immediate
consumption that produces the corrected static eval (negamax + QS stand-pat).
Read-only; no code modified.

**Reference set (all stronger than Coda):** Stockfish, Reckless, berserk-13,
Obsidian, PlentyChess, Alexandria.

---

## Headline

**Corrhist is fundamentally healthy — the mechanism, update gating, gravity
update, double-correction discipline and QS handling are all consensus-correct.
There is no bug and no dead source to drop.** The one real, bankable structural
divergence is that **Coda's "continuation" correction is the weakest possible
form** — a flat 1-ply `[piece][to]` table — while **all six reference engines
use a 2-D *paired* continuation correction** (previous-move × prior-move
context). This under-powers the cont source (its SPSA weight sits low and was
trimmed −10%, floor lifted to 0), and is the most likely place corrhist is
leaving Elo on the table. A secondary, lower-conviction item is the missing
minor-piece correction (3/6 engines have it; Coda's prior drop was of a *broken*
aliased key, not a correct one). Everything else is clean.

---

## What was verified clean (no action)

- **Double-correction invariant holds everywhere.** TT stores the RAW eval
  (`search.rs:3383`, `:4802`) and QS stores `raw_stand_pat` (`:5166`); every
  consumer (negamax `:3388`, QS `:5169`) re-derives `scaled → corrected` fresh.
  No path double-applies correction.
- **Update gating matches Stockfish's intent and is well-formed.** The
  `corrhist_lower_ok / corrhist_upper_ok` pair (`:4826-4829`) gates *whether* to
  update; the signed `err = best_score - scaled_eval` inside
  `update_correction_history` sets the *direction*. Enumerating the cases:
  exact nodes train in both directions; consistent fail-high trains up;
  consistent fail-low trains down; direction-*inconsistent* fail-high/fail-low
  are correctly skipped. This is exactly SF's `(bestValue < staticEval &&
  bestValue < beta) || (bestValue > staticEval && bestMove)` logic. The S1
  fail-low addition (#2116) closed the one real gap that existed.
- **Gates are all present and correct:** `!in_check`, `!best_move_noisy`
  (capture/EP/promo — matches SF `!(bestMove && pos.capture(bestMove))` and
  Reckless `best_move.is_noisy()`), excluded-move skip, `!is_decisive`,
  mate-range guard, stop-flag skip.
- **Proportional gravity is consensus-identical** (`:1425`): `entry += bonus -
  entry*|bonus|/LIMIT`, clamp ±LIMIT. Matches Reckless/SF/Alexandria exactly.
- **No source is dead weight.** CORR_W_PAWN=301 (range 100–600), CORR_W_NP=71
  (0–400), CORR_W_CONT=81 (0–400) — none floor-pinned at 0, none ceiling-pinned.
  SPSA still resolves all three. Answer to "(a) drop a dead source": **no** —
  cont is the weakest but the fix is to *strengthen its structure*, not drop it.
- **QS stand-pat correction** (`:5169`) is now consensus-aligned (all 6 correct
  the stand-pat; the 2026-06-23 fix removed Coda's last outlier here).
- `correction_value()` (`:1355`, used by the DEXT confidence margin) and
  `corrected_eval()` (`:1383`) share identical source/index/weight math — no
  drift between the consumed correction and the |corr| signal fed to extensions.

---

## Ranked findings

| # | Site | Issue / divergence | Cross-engine evidence | Severity | Proposed change |
|---|------|--------------------|-----------------------|----------|-----------------|
| **F1** | `cont_corr: [[i32;64];12]` decl `:892`; read `:1363-1375` & `:1397-1409`; update `:1455-1467` | **Continuation correction is a flat 1-ply `[piece][to]` table** keyed only on the single last (opponent) move — 768 entries, massively aliased, no pairing/stacking, no check/capture context. | **6/6 use a richer 2-D paired form.** SF `(ss-2)`+`(ss-4)` cont tables indexed by `(ss-1)` move (search.cpp:120-122, 87-90). Reckless `[in_check][capture][piece][to][piece][to]`, ss-2 & ss-4 (history.rs:160, search.rs:1300-1310). Obsidian `ContCorrHist[PIECE*SQ][PIECE*SQ]` (history.h:32). Alexandria `contCorrHist[2][6*64][6*64]` keyed `(ss-1)×(ss-2)` (threads.h:70, history.cpp:176). PlentyChess pointer-per-ply + check/capture flags. **Coda is the sole flat-1-ply outlier.** | **HIGH** (headline) | Implement paired continuation correction: key the table on the `(ply-2)` move and index by the `(ply-1)` move's piece+to (SF/Reckless pattern). Optionally add the `(ply-4)` contribution. The needed per-ply data already exists (`info.moved_piece_stack` / `info.moved_to_stack`, used at `:3861-3868`), so no new board plumbing is required — only a larger table + a subtable index. |
| **F2** | `:244-246` (minor/major dropped); board has no minor key (`board.rs:21-22` only `pawn_hash`+`non_pawn_key`) | **No minor-piece correction source.** Dropped 2026-05-18 (#1318 H1) — but that drop removed a *broken* table aliased to `non_pawn_key`, never a correct `minor_piece_key`. | 3/6 keep a *proper* minor key: SF `minor_piece_key` weight 8620 (search.cpp:84,113). Reckless `minor_key` (search.rs:1296,1318). PlentyChess `minorCorrectionFactor` + even a major one (history.cpp). Obsidian/Alexandria/Berserk do **not** — so this is a split, not a strong consensus. | **MEDIUM** (lower conviction) | Add a real incremental `minor_piece_key` Zobrist on the board (knights+bishops, optionally king per SF), then re-introduce a `minor_corr` source with its own weight. This is a genuinely *new* signal, not a re-add of the H0'd aliased one. Larger change (board Zobrist + recompute/verify). |
| **F3** | `:1376-1378` and `:1412-1414` | **Two-stage integer division** of the blended correction: `total/CORR_HIST_DIV` (1483) **then** `/CORR_HIST_GRAIN_T` (14). The product (~20762) is the only identifiable scale, so the two knobs are collinear, and the intermediate truncation throws away precision a single divide would keep. | Every reference does a **single** division at consumption: SF `cv/131072`, Reckless `sum/77`, Alexandria/PlentyChess one factor+shift. None two-stages it. | **LOW** | Collapse to one divisor: fold GRAIN_T into DIV (single `/COMBINED`), hardcode the second constant out. Note GRAIN_T already has the `--core`-exclude flag (`false` at `:340`), so the collinearity only bites *full-sweep* tunes; the durable win is the precision/clarity, not SPSA-noise removal. Near-bench-neutral. |

### Observation (not a finding)

Coda's **total correction authority is small** — saturated single-source pawn
correction reaches only ~14 cp (`1024*301/1483/14`), all-source same-direction
~25 cp, versus SF's ~95 cp from the pawn term alone (`12153*1024/131072`). This
is SPSA-chosen (DIV=1483 sits mid-range, not pinned) and is *plausibly correct*:
Coda's v9 net is better calibrated than SF's eval, so it needs less correction.
Flagging only so a future tune watching DIV knows the scale is deliberate, not a
bug. The pawn-vs-NP weight ratio is also *inverted* relative to SF (Coda
pawn≫NP; SF NP≥pawn) — again SPSA-resolved, not flagged as wrong.

---

## Per-change test plans

### F1 — paired continuation correction (headline)

- **Class:** search-shape feature change (changes corrhist values → changes the
  tree). This is the retune-on-branch archetype (big node delta, possibly flat
  raw Elo until cont weight + DIV recalibrate).
- **Build/bench:** non-trivial bench delta expected (correction values shift);
  re-measure `coda bench` with the prod net on the branch, pass explicit
  `dev_bench` + `--base-bench`.
- **Step 1 — raw SPRT:** `[0,3]` STC 10+0.1 vs main. If clearly H0, stop.
- **Step 2 — retune-on-branch (recommended even if Step 1 is flat):** focused
  cluster SPSA over `CORR_W_PAWN, CORR_W_NP, CORR_W_CONT, CORR_HIST_DIV,
  CORR_UPDATE_WEIGHT_MAX, CORR_ERR_DIV` (the existing CORR cluster spec). The
  prior is that a richer cont structure *raises* the optimal CORR_W_CONT; if
  SPSA pushes CORR_W_CONT up materially vs main, the source got stronger →
  apply + re-SPRT `[0,3]`. (Mirrors the cont-hist-malus retune-on-branch
  precedent: flat raw, +6.5 with retune.)
- **TC:** STC-first gate; if it banks, confirm non-regression at LTC 40+0.4
  Hash=256 before merge.

### F2 — minor-piece correction (re-add with a correct key)

- **Class:** new source + new board Zobrist key. Two commits: (1) add
  `minor_piece_key` incremental hash with the make/unmake + recompute-verify
  test parity the existing keys have (`board.rs:1583+`); (2) add `minor_corr`
  table, weight `CORR_W_MINOR`, wire into all three corrhist functions.
- **Bench:** changes (new source); re-measure.
- **SPRT:** `[0,3]` STC vs main. Because the prior H0 (#1318) was on a *broken*
  key, this is a genuine "does this feature help?" test, not a re-add of a
  rejected one — default bounds apply. Retune CORR cluster on-branch (the new
  weight needs a value) before the verdict SPRT. Lower priority than F1: do F1
  first, then F2 on top of the F1-retuned baseline.

### F3 — single-division collapse

- **Class:** near-neutral precision/cleanup. Values shift by sub-cp rounding.
- **Bench:** small delta (not zero — truncation changes). Re-measure.
- **SPRT:** `[-2,1]` STC non-regression. No retune needed (DIV range just needs
  widening to absorb the folded GRAIN_T factor; recompute min/max so the new
  default sits mid-range — see the "tunable range inheritance" rule). Bundle
  with another small corrhist cleanup if one is in flight to amplify SNR.

---

## Direct answers to the brief

- **(a) Dead-weight source to drop?** No. All four live sources have non-pinned
  SPSA weights. The continuation source is the *weakest*, but the fix is to
  upgrade its structure (F1), not delete it.
- **(b) Missing source the references have?** Two: a **2-D continuation
  correction** (F1 — 6/6 have it, Coda's flat form is the outlier; highest
  conviction) and a **minor-piece correction** (F2 — 3/6, needs a real key;
  lower conviction).
- **(c) Is the update formula optimal vs SF?** Yes — proportional gravity,
  both-direction bound-consistent gating, noisy-bestmove + in-check + decisive
  exclusions all match SF/Reckless. The only update-path nit is cosmetic (F3
  two-stage divide). The depth-weight cap (`min(depth+1, 17)/8`) is a reasonable
  variant of SF/Reckless's `depth*err` scaling and is SPSA-tuned; not flagged.

## Prior art checked (`experiments.md`)

- #2116 corrhist-allnode (fail-low training) — **H1, merged**; not re-proposed.
- #1318 minor/major drop — H1 on the *aliased* key; F2 is a different (correct-key) feature, not a re-add.
- #2144 npkey-king (king in non_pawn_key) — H0; F2 deliberately scopes minor to
  knights+bishops, with king optional/separate.
- #1970 corrhist-overhaul (T2.4 full-error update) — merged; F1/F2/F3 are
  orthogonal to it.
- #2263 probcut/multicut-corrhist — H0; F1/F2/F3 don't touch the multicut path.
