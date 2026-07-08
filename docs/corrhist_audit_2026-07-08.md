# Correction-history audit — Coda vs SF / Obsidian / Reckless / Berserk / Viridithas (2026-07-08)

Adam-commissioned deep audit of correction history (corrhist), triggered by the
`corrhist_fortress_drift_2026-07-06` finding (corrhist self-reinforcing a phantom
±0.45 in dead positions) and a suspicion it may be reinforcing *other* errors.
Coda's code reviewed in isolation, then diffed against five reference engines
(current sources, three parallel extractions). Reference formulas quoted are from
those engines' current trees.

## TL;DR

The fortress bug was **one symptom of a structural choice**, not a one-off. The
cross-engine diff found the root cause the fortress doc missed:

> **Coda trains corrhist on the error against the RAW eval (`best_score −
> raw_eval`). All five reference engines train against the CORRECTED eval
> (`best_score − correctedEval`) — the residual after their own correction.**

Residual-training self-stabilises at the *true* correction (as the correction
approaches the right value, the residual → 0 and the gravity update stops
pushing). Raw-training has a fixed point at the **rail (±LIMIT), independent of
the true error magnitude** — which is exactly the "rails to a phantom ±0.45"
behaviour the fortress doc observed. This is Finding **#1** and it is the
headline: it explains the rail mechanism *generally* (all material, all position
types), and the current `mat_damp` fortress fix is a narrow material-count
band-aid over it.

Coda's corrhist **hygiene is otherwise correct** and matches the field (TT stores
raw eval; corrected eval only out of check; quiet-bestmove + not-in-check +
direction-consistency + not-decisive update gates). The problems are (1) the
raw-vs-residual baseline, (2) not using correction magnitude as an uncertainty
signal, (3) a Coda-unique `trans_corr` source with no peer precedent, and (4) a
shallower continuation term than everyone else.

## Cross-engine comparison

| | Coda | Stockfish | Obsidian | Reckless | Berserk | Viridithas |
|---|---|---|---|---|---|---|
| Sources | pawn, npW, npB, **cont(1-ply)**, **trans** | pawn, **minor**, npW, npB, cont(2,4) | pawn, npW, npB, cont(2) | pawn, npW, npB, cont(2,4) | pawn, cont(2,3) | pawn, npW, npB, **minor**, **major**, cont(2,4) |
| Update err baseline | **RAW eval** | corrected | corrected | corrected | corrected | corrected |
| Entry limit | 1024 | 1024 | 1024 | 14605 | ~16384 | 1024 |
| Per-update cap | ≈±394 (**38%** of limit) | ±256 (25%) | ±256 (25%) | −4771/+3001 | ±4096 | ±256 (25%) |
| Depth in bonus | `min(depth+1,16)` | `×depth` (uncapped) | `×depth` | `×depth` | `×depth` | `×depth` |
| 50-move eval scale | yes (`apply_halfmove_scale`) | no | yes | yes | yes | yes |
| **50MR index bucketing** | no | no | no | **yes (16 buckets)** | no | no |
| Material eval scale | yes (`#813`) | no | no | yes | no | yes |
| **Correction damping** | **material-count `mat_damp`** | none | none | none | none | none |
| `\|corr\|` as uncertainty | **no** | yes (futility/LMR/SE) | as `complexity` | yes (SEE/LMR/IIR) | no | yes (TT-complexity LR) |
| TT stores | raw ✓ | raw ✓ | raw ✓ | raw ✓ | raw ✓ | raw ✓ |
| Update gates | !chk, quiet-best, dir-consistent, !decisive, !excluded | !chk, quiet-best, dir-consistent | !chk, quiet-best, canUseScore | !chk, !noisy-best, bound-consistent | !chk, quiet-best, bound-consistent | !chk, !winning-SEE-best, bound-consistent |

## Findings (ranked)

### #1 — Update trains on RAW eval, not the corrected residual — the rail mechanism  [ROOT CAUSE]

`search.rs:5898` passes `scaled_eval` (the pre-correction, halfmove-scaled NNUE
eval) as the update baseline:
```rust
update_correction_history(info, board, best_score, scaled_eval, depth);
//   err = search_score - raw_eval   (raw_eval := scaled_eval, uncorrected)
```
`static_evals[ply]` already holds the *corrected* eval — it just isn't used here.

**Every reference engine uses the corrected eval:**
- SF `search.cpp:1540`: `bestValue - ss->staticEval` (staticEval = `to_corrected_static_eval`)
- Obsidian `search.cpp:1229`: `bestScore - ss->staticEval` (adjustEval output)
- Reckless `search.rs:1092`: `best_score - eval` (correct_eval output)
- Berserk `search.c:831`: `bestScore - ss->staticEval` (rawEval + GetCorrectionScore)
- Viridithas `search.rs:1644`: `best_score - fresh_eval` (adj_shuffle(raw)+correction)

**Why it matters (the mechanism, math):** the gravity update is
`entry += b − entry·|b|/LIMIT`, with bonus `b ∝ err`.
- **Residual training** (references): `err = best_score − (raw + k·entry)`.
  As `entry` grows, `err` shrinks, so `b` shrinks; the fixed point is
  `entry ≈ true_error/k` — the correction converges to the *true* value and
  stops. Self-stabilising, magnitude-accurate.
- **Raw training** (Coda): `err = best_score − raw`, which does **not** shrink as
  `entry` grows. The fixed point is `entry = LIMIT·sign(err)` — the **rail**,
  regardless of the true error size.

So Coda's corrhist is effectively a **sign integrator that rails**, applying the
full ~±45cp whenever the error sign is persistent and search can't refute it.
The fortress doc's "fixed point is the rail regardless of magnitude" is a *direct
consequence of the raw baseline* — the references run the same gravity formula
and don't rail because they subtract their own correction. This is not
fortress-specific: any position class where the net has a small persistent
directional bias and search is low-signal (locked middlegames, blocked pawn
chains, shuffizones) gets the full rail correction, not a proportional one.

**Fix:** feed the corrected static eval to the update
(`info.static_evals[ply_u]` instead of `scaled_eval`), matching all five
references. This likely **subsumes `mat_damp`** (drift self-cures because the
correction converges to ~0 in dead positions). **Requires retune-on-branch** —
`CORR_ERR_DIV`, weights, cap were all calibrated for raw-error magnitudes, which
are larger than residuals. Validate: (a) the fortress repro + a high-material
locked repro should read ~0 *without* `mat_damp`; (b) retune history/corr
cluster; (c) SPRT `[0,3]`. Highest priority.

*Caveat:* this is a real behavioural change to a load-bearing +Elo feature, and
raw-training was a deliberate choice (see the `search.rs:5893-5897` comment,
which reasons about halfmove-scale space but not raw-vs-corrected). Test
carefully; it is not guaranteed +Elo, but it is the principled, field-unanimous
shape and the direct cause of the observed pathology.

### #2 — Correction magnitude is never used as an uncertainty signal

4 of 5 references treat a large `|correction|` as "eval is less trustworthy here"
and make search more careful:
- SF: futility margin `+|corr|/182069`; singular DEXT margin `−|corr|/194822`;
  LMR `r −= |corr|/26131` (`search.cpp:946,1203,1259`)
- Reckless: SEE quiet margin `+560·|corr|/1024`; LMR `−3297·|corr|/1024`;
  IIR/TT-ext terms (`search.rs:511,778,658`)
- Viridithas: TT-complexity-scaled *learning rate* on the corrhist update itself
- Obsidian: reuses `|staticEval − rawEval|` as `complexity`

Coda uses `|corr|` **nowhere** in pruning/reduction/extension. So exactly when
corrhist is applying the biggest (and, per #1, most likely rail-inflated)
adjustment, Coda prunes and reduces just as hard — amplifying the damage of a
bad correction. Add `|corr|` terms to futility / LMR / singular margins (SF
shape). Medium-high; independent of #1 and cheap to try. New tunables at 0 =
bench-neutral, so it can go in as `[-2,1]` scaffolding then be tuned on.

### #3 — Fortress fix is a material-count proxy; Reckless buckets the index by 50MR

`mat_damp` (`search.rs:1811`) zeroes correction ≤6 pieces, full ≥16 — a
*piece-count* proxy for "no progress". It cannot catch a **high-material locked**
position (blocked KID/French/Benoni chains, opposite-side-castling lockups),
where the same rail dynamics apply but piece count is high.

Reckless instead **buckets the corrhist index by halfmove clock**
(`board.rs:79-81`, `(hmc−8)/8` → 16 buckets): a position shuffling for N
no-progress plies reads a *different, near-empty* corrhist slot than the same
material at clock 0, so correction physically cannot accumulate to the rail
across a no-progress phase — **material-independent**, and it catches the
high-material case `mat_damp` misses. This is a Reckless-only pattern (not
consensus), but it targets Coda's exact bug class.

**Sequencing:** if #1 lands, fortress drift should largely self-cure and both
`mat_damp` and index-bucketing may be unnecessary — so **test #1 first**, then
decide whether #3 is still needed. If #1 is rejected, #3 is the best structural
mitigation. (Validate the premise cheaply: a high-material locked repro on
current main — corrhist on vs `NO_CORRECTION=1` — should show `mat_damp` does
*not* catch it.)

### #4 — `trans_corr` is a Coda-unique source with no peer precedent

Coda keys a correction table on `hash ^ last.hash` (`search.rs:1757`) — a
move-signature. **No reference engine has any move-signature/transition
correction**; all five use only material (pawn/minor/major/non-pawn) and
continuation keys. A move-signature table rails hardest in shuffling/repetitive
positions (the same transitions recur), so it is the source most exposed to the
#1 rail mechanism, and it is unvalidated against the field. Its weight is already
the lowest (`CORR_W_TRANS=59`). **Ablation test** `CORR_W_TRANS=0` at `[-1.5,1.5]`;
if neutral/positive, drop it (fewer rails, less memory). Cheap, independent of #1.
(Note: the fortress doc found zeroing any *single* source doesn't stop the drift —
the sign just flips — so this is not a fortress fix, but trans is the least
principled source and worth pruning on its own merits.)

### #5 — Continuation correction is 1-ply; the field uses 2-ply (+4-ply)

Coda's `cont_corr` (`search.rs:1749`) is a single ply, keyed by the *opponent's
last move* piece/to, read directly. Every reference engine uses a two-level
continuation correction via stacked subtable pointers: SF/Reckless/Viri
(ss-2)+(ss-4), Berserk (ss-2)+(ss-3), Obsidian (ss-2). Adding a 2-ply (and
4-ply) continuation-correction term (the standard subtable-pointer form, which
Coda already has machinery for in main cont-hist) is a plausible small gain and
brings Coda in line. Lower priority; structural, needs retune. Do after #1/#2.

### #6 — Minor thresholds and dropped sources (low priority)

- **Per-update cap is 38% of the entry limit** (`≈394/1024`) vs 25% for
  SF/Obsidian/Viri (`256/1024`) — Coda rails in ~3 updates vs ~4-5. If #1 is
  *not* adopted, tightening `CORR_BONUS_CAP_DIV_10X` toward the 25% ratio is a
  cheap partial mitigation. If #1 *is* adopted, moot (won't rail).
- **Depth weight capped at `min(depth+1,16)`** — every reference uses uncapped
  linear `×depth`. Deep, reliable searches beyond depth 16 get no extra weight.
  Minor; worth widening `CORR_UPDATE_WEIGHT_MAX` range in the next tune.
- **minor/major sources dropped** (2026-05-19). SF keeps minor; Viri keeps
  minor+major. Coda traded them for `trans`. If #4 drops `trans`, re-testing a
  `minor` correction (more principled, peer-validated) is the natural
  replacement experiment.

## What Coda already gets right (do not "fix")

- **TT stores raw eval** (`search.rs:4221`), corrected recomputed on probe — no
  double-correction. Matches all five references.
- **Corrected eval only out of check**; in-check eval undefined. Matches field.
- **Update gates**: not-in-check, quiet (non-noisy) bestmove, direction-consistent
  both-bounds (`corrhist_lower_ok/upper_ok`), not-decisive (mate/TB), not inside a
  singular-exclusion search. This is a **superset** of SF's gates (SF has no
  explicit `!is_decisive` on the update) — Coda is if anything more conservative.
- Proportional gravity + per-entry clamp ±1024 (consensus limit).
- Trains **both** bound directions (fail-high lower-bound and fail-low
  upper-bound) with correct sign gating — matches SF/Reckless/Obsidian/Viri.

## Recommended sequence

1. **#1 (raw→residual baseline)** on a branch: change the update baseline to the
   corrected eval, keep `mat_damp` initially, run the fortress + a new
   high-material locked repro, then retune the corr/history cluster and SPRT
   `[0,3]`. If it holds, test dropping `mat_damp`. *This is the one to do first —
   it is the root cause and it is field-unanimous.*
2. **#2 (`|corr|` uncertainty)** in parallel (independent): add SF-shape `|corr|`
   terms to futility/LMR/SE as new tunables at 0 (bench-neutral), then tune on.
3. **#4 (`trans_corr` ablation)** — cheap `[-1.5,1.5]`, can run now regardless.
4. **#3 (50MR index bucketing)** only if #1 doesn't fully cure drift on the
   high-material locked repro.
5. **#5 (2-ply/4-ply cont-corr)** and **#6 (minor source)** as later structural
   experiments once the baseline is fixed.

Source extractions (SF/Obsidian/Reckless/Berserk/Viri corrhist, verbatim
formulas + citations) captured in this session's audit; fortress-drift
background in `docs/corrhist_fortress_drift_2026-07-06.md`.
