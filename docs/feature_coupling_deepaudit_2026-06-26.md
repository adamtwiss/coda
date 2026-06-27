# Feature-Coupling / Calibration-Chain Deep Audit — 2026-06-26

Second-order audit: not "is threshold X right" (that's SPSA's job) but "does
feature A's output mis-feed feature B, and are the shared predicates
(`improving`, `cut_node`, `tt_pv`, corrected vs scaled eval, the cp scale)
wired *consistently* across every consumer." Reference set: SF, Reckless,
Berserk, Obsidian, PlentyChess, Alexandria. HEAD 000ace2. `src/search.rs`.

## Headline

**The couplings are largely coherent and SF-parity.** Most cross-feature
wiring that looked suspicious turned out to match a 3+/6 reference consensus
(verified, not assumed). There is **one genuine second-order divergence worth
an experiment** (the pre-move pruning depth estimate is computed a different,
adjustment-free way than the actual reduction — SF unifies them) and **two
minor consistency items**. The calibration/scale chain is sound: the only
cross-scale boundary (classical-SEE cp ↔ NNUE-eval units) is handled
explicitly by `SEE_MATERIAL_SCALE`, and corrected-vs-scaled eval is used
correctly everywhere a margin meets a propagated score. No mis-denominated
margin found.

This "no bug here" result is itself load-bearing: it says the remaining Elo
gap is **not** hiding in pruning-feature interactions, and steers effort back
to eval/TM/LTC-scaling (per `project_elo_gap_decomposition_2026-06-13`).

---

## Finding 1 (MEDIUM) — `lmr_d` (pre-move pruning depth) ignores the LMR adjustments that set the *actual* search depth

**Features coupled:** LMR reduction ↔ futility pruning ↔ SEE-quiet pruning.
**Sites:** `lmr_d` computed `search.rs:3988-3993`; consumed by SEE-quiet
`:4003` and futility `:4160,:4167`; the *actual* reduction with all
adjustments computed `:4275-4392`.

**The defect.** Coda computes the LMR depth used for pre-move pruning **two
different ways**, and SF a third (unified) way:

- `lmr_d` (`:3988`) = `depth − lmr_reduction(depth, move_count)` — the **raw
  table value only**. No `improving`, `tt_pv`, `cut_node`, history, threat,
  king-pressure, or complexity term.
- The reduction the move is *actually* searched at (`:4275-4392`) subtracts
  `improving` (−1), `tt_pv` (−1), `history/LMR_HIST_DIV`, complexity, threats,
  king-pressure, and adds `+1` at cut nodes and `+1` for noisy TT moves. These
  routinely move the effective depth by ±2–4 ply vs `lmr_d`.
- **SF builds ONE `lmrDepth`** (`Stockfish/src/search.cpp:1078` then
  `:1115 lmrDepth += history/2995`) and feeds that single history-adjusted
  value into **both** futility (`:1117`) and SEE-quiet (`:1134`). Berserk and
  Obsidian likewise derive the pruning `lmrDepth` from the reduction `R` they
  will actually use.

**Why it's a coupling bug, not a tuning knob.** The dangerous direction is a
move that the adjustments will search *near full depth* — a good-history,
`improving`, or `tt_pv` quiet (reduction ≈ 0 or negative) — but which `lmr_d`
predicts will be shallow (`depth − base`). Such a move can be SEE-quiet- or
futility-pruned on a depth estimate that's too pessimistic, i.e. we prune a
move we were about to search deeply. The reverse (bad cut-node late move,
actual reduction `base+2`) is under-pruned. Coda's futility partially
self-heals via its own `hist_adj = main_hist/128` (`:4163`), but (a) it uses
*only* main history, not the `cont+pawn` composite SF folds in, and (b)
**SEE-quiet pruning (`:4003`) has zero history/improving/ttPv awareness at
all** — a historically-excellent quiet landing on an attacked square is pruned
at the identical `−SEE_QUIET_MULT·lmr_d²` threshold as a terrible one.

**Cross-engine evidence:** SF, Berserk, Obsidian all derive the pruning depth
from the adjusted reduction (≥3/6); SF additionally injects history into it.
Coda is the outlier in using a pure base-table estimate for the prune and a
fully-adjusted one for the search.

**Honest prior (read before firing).** Coda has H0'd *standalone*
history-pruning (#1697 −6.8) and *composite-history-in-futility* (futility
audit FUT-3). Those test different mechanisms — neither tested "give
good-history/improving/ttPv quiets a more lenient SEE-quiet threshold via a
shared adjusted `lmr_d`." So the prior is cautionary, not disqualifying. The
SEE-quiet leg is the genuinely-untested part.

**Fix.** Compute one `lmr_d_adj` that subtracts at least
`improving + tt_pv + history/DIV` (the cheap, already-in-scope terms) and feed
it to **both** SEE-quiet and futility, mirroring SF's single `lmrDepth`.
Minimal variant: add only a history term to the SEE-quiet threshold (the leg
with *no* history today).
**SPRT:** `[0,3]` STC 10+0.1 first. **Retune-on-branch: yes** — this shifts
SEE-quiet + futility prune volume; `SEE_QUIET_MULT`, `FUT_BASE`,
`FUT_PER_DEPTH` were tuned against the base `lmr_d` and will want to move.
**Bench:** non-trivial change (prune counts move) — re-measure.

---

## Finding 2 (LOW) — LMR uses two different PV definitions within one adjustment block

**Features coupled:** PVS window state ↔ LMR reduction.
**Sites:** `search.rs:4279` vs `:4284`.

```
if beta - alpha > 1   { reduction -= 1; }   // "reduce less at PV" — LIVE window
...
if !is_pv && move_count > 1 { reduction += 1; }  // "reduce more at cut" — alpha_orig
```

`is_pv` (`:3440`) is fixed at node entry (`beta − alpha_orig > 1`). Line 4279
uses the **live** `alpha`, which climbs as moves raise it. At a true PV node
whose running `alpha` has reached `beta−1`, line 4279 stops granting the PV
reduce-less credit while line 4284 still treats the node as PV (no cut bonus)
— the late PV move gets a plain reduction. SF gates the reduce-less on the
static `PvNode` template (= `is_pv`), so it keeps the credit regardless of
window narrowing. Divergence only fires when running alpha sits exactly at
`beta−1` at a PV node (uncommon; usually `beta−alpha > 1` holds), hence LOW.
**Fix:** use `is_pv` at `:4279` for intent-consistency.
**SPRT:** `[-1.5,1.5]` (uncertain direction, near-neutral). Bench ~flat.

---

## Finding 3 (LOW) — SEE-quiet ordered before futility interacts with sticky `skip_quiets`

**Features coupled:** SEE-quiet (`continue`, per-move) ↔ futility (`continue`
**plus** `skip_quiets = true`, kills all remaining quiets).
**Sites:** SEE-quiet `search.rs:3997`, futility `:4156`. Coda order: LMP →
SEE-cap → SEE-quiet → … → futility. SF order: futility → SEE (SF
`search.cpp:1123` then `:1134`).

A move that is *both* SEE-prunable and futility-prunable hits SEE-quiet first
in Coda → `continue` (no skip) → the sticky `skip_quiets` trigger is deferred
to a later, weaker-history move. SF prunes futility first. Because Coda's
futility (unlike SF's) sets the sticky flag, the relative order has a real but
small effect on *when* the quiet stream is cut. Low impact; folds naturally
into a Finding-1 reorder if that's pursued.
**SPRT:** bundle with Finding 1 or skip. Bench moves slightly.

---

## Verified COHERENT (the valuable "not a bug" confirmations)

Each of these *looked* like a second-order defect and was checked against the
reference sources; all are consensus-correct — do not spend SPRT slots here:

- **Sticky `skip_quiets` on futility** (`:4173`). Suspected over-aggressive vs
  SF's per-move `continue`. **But Obsidian (`search.cpp:1021`), Berserk
  (`search.c:634`), and Reckless (`search.rs:734`) all set `skipQuiets` on
  futility** — Coda matches 3/6. SF is the outlier. Not a bug.
- **`tt_pv` sticky flag** (`:3133 = is_pv || (tt_hit && tt_entry.tt_pv)`).
  Correct Alexandria/Obsidian sticky-PV construction; feeds LMR `−1` (`:4325`)
  and TT store (`:4802`) consistently.
- **`improving` consistency.** Computed once (`:3400-3411`) from *corrected*
  `static_eval` vs the stored corrected `static_evals[ply−2]` (ply−4 fallback,
  in-check sentinel handled). The same `improving` and the same corrected
  `static_eval` feed RFP, LMP, futility(lmr_d), ProbCut, and LMR — one
  definition, used uniformly.
- **Corrected vs scaled eval placement.** Margins compare *corrected*
  `static_eval` against propagated (scaled-space) `alpha/beta/best_score`
  everywhere (RFP `:3571`, NMP `:3623`, razor `:3534`, futility `:4167`,
  bonus_depth `:4611`). This is intentional: corrhist makes `static_eval` a
  better predictor of the scaled search score, so the cross-space comparison
  is the *point*, not a mismatch. The LMR complexity term correctly uses
  `scaled_eval` (pre-correction) so it measures only corrhist magnitude
  (`:4361-4363`). corrhist *update* trains on `best_score − scaled_eval` with
  a direction-consistent Lower/Upper gate (`:4826-4844`) matching Reckless.
- **Calibration/scale chain.** All search margins live in NNUE-eval units and
  drift together (SPSA retunes them together) — no margin is mis-denominated.
  SEE pruning thresholds (`−SEE_QUIET_MULT·lmr_d²`, `−cap_margin`) live in
  classical-SEE cp and are compared only against SEE swings (self-consistent).
  The single cross-scale boundary — QS delta pruning — is bridged explicitly
  by `SEE_MATERIAL_SCALE=215` (QS audit QS-B2). `BAD_NOISY_MARGIN` is correctly
  an eval-vs-alpha futility scalar, not a SEE threshold.
- **`cut_node` propagation.** `!cut_node` on NMP/ProbCut/PVS-zero-window,
  `true` into the reduced LMR child, `false` on the full-window PV re-search
  and first move. All match SF. The one deviation (SE verification hardcodes
  `false` vs SF's parent-`cutNode`) is the already-logged Q2 item, not new.
- **Double-extension ↔ new_depth ↔ do_deeper.** `double_ext_count` propagated
  per-move (`:4232`) on the pre-do_deeper `singular_extension`; do_deeper
  mutates `new_depth` and is re-search-guarded `new_depth > lmr_depth`
  (`:4495`, the B2 fix — now on trunk). Negative SE and LMR stack on
  `new_depth` the same way SF stacks negative ext + `r`. No double-count.
- **Hindsight reduce/extend ↔ LMR `prior_reduction`.** `info.reductions[ply]`
  is zeroed after the reduced search and before any re-search (`:4466`), so
  children read the correct parent reduction for hindsight gating (T1.2 fix
  present). Reduction (`prior_reduction≥2`) and extension (`≥3`, non-PV,
  eval_sum≤0) are mutually exclusive bands — no sign fight.
- **Stat-update ordering.** History/contHist/pawnHist/capture bonuses and the
  malus loops fire *after* the `alpha>=beta` cutoff, inside the cutoff block
  (`:4604-4755`), before `break` — bonus before penalties, `quiets_count−1` /
  `n_captures_tried−1` correctly exclude the cutoff move itself. corrhist
  update is post-loop, gated off SE verification and `stop`. No
  best_score-pollution or fire-before-cutoff bug.

---

## Recommended action

Only **Finding 1** is worth fleet time, and only as a careful retune-on-branch
(STC `[0,3]` first; if neutral, it's another confirmation the pruning shape is
saturated). Findings 2–3 are one-line consistency cleanups bundle-able at
`[-1.5,1.5]`. The dominant takeaway is the negative result: the pruning-feature
*interactions* are SF-parity, so the Elo gap is elsewhere.
