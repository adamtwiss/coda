# Peripheral (Non-Pure-Search) Mechanisms — Idea Catalogue (2026-04-22)

## Motivation

SPRT #619 — fixes to our 50-move score-blending formula — landed **+3.3 Elo** on a
single-day branch. That gain didn't come from pruning, move ordering, or NNUE work;
it came from the *eval-post-processing* layer. Looking at `experiments.md`:

- ~1000 experiments on pure search (move ordering, pruning, reductions, extensions).
- A handful on contempt (all H0 by 2026-04-04, one H1 *remove* at #508 +2.5).
- A handful on 50-move scaling (all H0 historically, until #619 landed).
- Essentially none on: optimism, twofold-history blending, halfmove-gated TT cutoff,
  50mr-threatened mate downgrades, fortress detection, shuffling detection.

This is a **blue ocean**. The space is small (maybe 10–20 ideas total) and each win
is small (+1–5 Elo), but they *add up* because every game ends in one of these modes.

## What the top engines do that we don't

Survey of `Stockfish/`, `Reckless/`, `viridithas/`, `Obsidian/`.

### P1. Optimism (SF, Reckless, Viridithas — all three)

**Mechanism.** At root, each ID iteration computes per-side optimism from the root
move's rolling average score:

```
optimism[us]  = K1 * avg / (|avg| + K2);     // SF: 144/91, Reckless: 169/187, Viri: 128/opt_offset
optimism[~us] = -optimism[us];
```

Optimism is then blended into eval at every node:

```
v = (nnue * (MAT_BASE + material) + optimism[stm] * (OPT_BASE + material)) / MAT_BASE
```

**What it does.** Amplifies eval slightly when we're doing well at root, dampens when
we're doing poorly. Same directional idea as contempt, but **adaptive** and
**symmetric between sides** (not just STM).

**Coda context.** We removed contempt at #508 (+2.5) because it was mis-calibrated for
v9's flatter eval. Optimism is *root-relative* and *self-tunes* via the avg formula,
so it doesn't carry contempt's calibration problem. Three independent engines
converged on this design.

**Expected Elo.** +2 to +6. High confidence — unambiguous convergence across three
strong engines.

**Implementation.** One `optimism: [i32; 2]` array on SearchData, update after each
ID iteration, blend in `eval_wrapper`. SPSA `K1` and `K2` (4-param branch tune —
K1, K2, optimism_mat_base, and a gate-off-if-eval-huge threshold).

**Status 2026-04-22:** **#636 -2.0 @ 10986g H0.** Dropped on first pass.

**Status 2026-04-23:** retried on tuned C8-fix trunk (#671 experiment/p1-optimism-c8-retry) →
**−7.1 @ 4268g H0 ✗**, worse on tuned trunk. Untuned port H0s regardless of baseline.
Parked — retry only with K1/K2 focused SPSA on-branch (4-param tune, ~1000 iters).

### P2. TT-cutoff gate on high halfmove (SF: `<96`, Reckless: `<90`)

**Mechanism.** Skip TT cutoff when `halfmove_clock >= 90` (or 96).

```rust
if tt_hit && can_cutoff && board.halfmove < TT_CUTOFF_HALFMOVE_MAX {
    return tt_score;
}
```

**What it does.** TT scores are stored without halfmove context. A cached mate-in-10
stored when halfmove=20 is unreachable when halfmove=92 (only 8 plies until 50mr).
The gate forces re-search near the drawing clock instead of trusting the stale bound.

**Coda context.** We do `score * (200 - hm) / 200` on eval but TT cutoffs bypass
the dampening entirely. This is a consistency bug at minimum — we're scaling
the *static* eval but not the *bound-from-search*.

**Expected Elo.** +1 to +3. Cheap, correctness-adjacent, low variance.

**Status 2026-04-22:** **#628 +1.2 @ 13970g H1 ✓**, merged at `8277799`. Landed inside
the expected band.

### P3. TT-score adjustment when 50mr threatens stored mate/TB (Reckless)

**Mechanism.** `score_from_tt(score, ply, halfmove)`:

```rust
if is_win(score) && Score::MATE - score > 100 - halfmove {
    return Score::TB_WIN_IN_MAX - 1;   // downgrade unreachable mate
}
```

A stored "mate in N plies" is *only reachable* if `N <= 100 - halfmove`. Otherwise
the 50-move rule claims the draw first. Reckless downgrades these to generic
TB_WIN rather than propagating a false mate.

**Coda context.** Our `score_from_tt` only does ply adjustment. We have the same
bug SF fixed years ago and Reckless still guards against. Rare but real — affects
Lichess endgame play (which Adam has been watching).

**Expected Elo.** +1 to +3 (small but real). Bug-fix class; Lichess-visible class.

**Status 2026-04-22:** **#633 -1.4 @ 19668g H0**, merged as confident-correctness at
`19da0e8` (Reckless-style bug-fix class, `[-3, 3]` bounds, −4 tripwire not hit). Didn't
pass positive SPRT but kept as a correctness measure; the score-path is lichess-visible
even if hard to expose under self-play.

### P4. NNUE complexity blending (SF only)

**Mechanism.** SF splits NNUE output into `psqt` and `positional` components and
uses `|psqt - positional|` as a complexity / uncertainty proxy:

```cpp
int complexity = abs(psqt - positional);
nnue     -= nnue     * complexity / 18236;   // dampen eval when uncertain
optimism += optimism * complexity / 476;     // boost optimism when uncertain
```

**Coda context.** We don't have a psqt/positional split, but we have *alternative
uncertainty signals* already computed:

- `any_threat_count` — tactical-density proxy (we use it for pruning gates).
- `corrhist_magnitude` — eval-has-been-wrong signal (already in tree).
- `|eval - parent_eval|` — instability signal (used in #542).
- `threat_deltas` count per move.

Port the mechanism with one of these as complexity. Not guaranteed to transfer —
SF's psqt/positional split specifically separates material-structural signal
from positional NN signal; our substitutes are different in character.

**Expected Elo.** +1 to +5, moderate confidence. Depends on which proxy is chosen.

### P5. Material-weighted eval × material-weighted optimism blend (SF, Viridithas)

**Mechanism.** Extension of P1:

```
v = (eval * (MAT_BASE + material) + optimism * (OPT_BASE + material)) / MAT_BASE
```

Effect: optimism has proportionally *more* weight in material-heavy (middlegame)
positions, less in sparse endgames. Rationale: optimism captures "can we convert
this", which is middlegame-relevant; endgames have narrower conversion windows.

**Coda context.** Requires P1 first. Our current scaling is `v * (22400 + material) / 32 / 1024` —
material-weighted *eval* but no optimism yet.

**Expected Elo.** Bundled with P1 results.

### P6. Thread voting for SMP best-move (Viridithas)

**Mechanism.** Multi-thread best-move selection weighted by `(score - min + 10) × completed_depth`.
Instead of "main thread's best", it's "vote-weighted best".

**Coda context.** Our SMP is now correctness-fixed (#557, #576, +651 Elo in T=4
self-play). Next frontier is SMP *quality*. Thread voting is one path.

**Expected Elo.** Unclear at self-play (might be invisible in self-play for same reason
node-based TM was); **LTC testing required**. Could be +2–5 at LTC. Deferred.

## Novel ideas (chess-knowledge-driven, not copied from engines)

### N1. Twofold-in-game-history eval blend

**Status 2026-04-23:** tested as `experiment/n1-twofold-eval-blend`
→ **#691 +0.01 Elo ±1.7 @ 28,792 games H0** (bounds [0, 5]).
Extraordinarily flat — effectively a statistical zero. The feature
fires rarely enough that even a correct eval blend doesn't move the
needle at STC. Halfmove scaling already catches most of the
structurally-drawish signal twofolds would amplify. Dropped.

---

*Original proposal preserved below.*


**Rationale.** A position that occurred once in the actual game history, but hasn't
tripled yet, is *empirically more drawish* — one side has already chosen to revisit it,
which is weak evidence of being unable to make progress. Currently we score it as
non-draw (correctly, because threefold hasn't triggered) with full eval.

**Mechanism.** In rep-detection helper, return a flag "twofold in game history". In
eval post-processing, blend eval toward 0 by e.g. 25–50% when the flag is set.

**Expected Elo.** +1 to +4. Cheap (already scanning the rep table). No engine I
checked does this explicitly — they rely on pure 3-fold mechanics + halfmove scaling.

**Risk.** Could encourage seeking twofolds (if we're losing, a twofold near-draw is
better than the loss). That's actually the *right* behavior in losing positions.
Root-relative variant: blend twofolds toward 0 only when we're at/above 0cp at root.

### N2. Shuffling detector (non-halfmove-resetting streak)

**Rationale.** Halfmove resets on pawn move/capture. Our 50mr scaling linearly decays
from halfmove=0. But a run of 12 non-resetting moves very early (halfmove=15, say)
is a *shuffle*, and shuffles rarely lead to progress. Halfmove alone misses this.

**Mechanism.** Add `shuffle_streak` counter to board state — reset on pawn-move,
capture, OR check. Blend eval toward 0 proportional to shuffle_streak **beyond
halfmove's own decay**, gated when halfmove is low (so we don't double-count).

**Expected Elo.** +1 to +3. Novel — I didn't find this in any engine. Likely worth
its own SPRT after P1/P2 land.

### N3. Fortress soft-cap

**Rationale.** Strong NNUE nets occasionally read a positional win where no
breakthrough exists (opposite-colour-bishop fortresses, knight-outpost blockades).
Classical fortress heuristic: material-balanced ± ε, no pawn breaks available,
king-opposition static. Soft-cap |eval| at ~50 cp in such positions.

**Mechanism.** Detect:
- `abs(material_balance) < 200` AND
- popcount of legal pawn moves for both sides < 2 AND
- `abs(eval) > 200`

Then clamp `eval = eval.signum() * min(abs(eval), 50)`.

**Expected Elo.** +0 to +4. Risk of mis-clamping legitimate advantages. **Low priority**
unless we can measure a real Lichess-visible fortress-misread rate first.

### N4. Halfmove-scaled pruning margins

**Rationale.** Our 50mr dampening scales eval. But pruning margins (RFP, futility)
are *calibrated against un-dampened eval magnitude*. At halfmove 80, eval is 60%
of its magnitude, but pruning thresholds haven't shrunk — so effective pruning is
40% more aggressive than intended near the drawing clock.

**Mechanism.** Apply the same `(200 - halfmove) / 200` factor to RFP_MARGIN_IMP,
FUT_BASE/FUT_PER_DEPTH, and NMP eval margin.

**Coda context.** Earlier attempts to scale pruning margins (RFP king-pressure #503,
LMP king-pressure #504) H0'd because the signal wasn't coupled to eval magnitude.
Halfmove-scaling IS coupled — the eval is literally being shrunk. Different class
of experiment.

**Expected Elo.** +1 to +4. Must be tested with retune.

### N5. QS insufficient-material short-circuit

**Status 2026-04-23:** tested as `experiment/n5-qs-insufficient-material`
→ **#683 stopped flat at −0.2 @ 50K games** (LLR −0.69, treading in
noise band; would have H0'd on [0, 5] bounds). Dropped.

Reasoning: NNUE already scores KvK/KNvK/KBvK near 0 through training,
so force-returning draw_score=0 is redundant. Not bench-invariant
(+6.4%) — tree shape shifts, but no Elo. The idea is correctness-sound
but produces no measurable benefit on a well-trained net.

---

*Original proposal preserved below.*


**Rationale.** KvK / KNvK / KBvK / K+same-colour-Bs-only are drawn by rule.
Currently we detect these in main-search rep checks but *not in QS*. QS does
stand-pat + captures + possibly mates — can misread insufficient-material
endgames by exploring pointless captures.

**Mechanism.** Add fast insufficient-material check at QS entry. Return `draw_score`
immediately when matched.

**Expected Elo.** +0.5 to +2. Correctness-class and tiny NPS win.

### N6. Promotion-imminent extension

**Rationale.** Pawn on 6th/7th rank often decides games. Many engines extend
pawn-push moves to 7th. Coda has LMR exemptions but no extension.

**Mechanism.** Extend by 1 when the move is a pawn push to 6th/7th rank from STM's
perspective AND the pawn is currently unblocked.

**Expected Elo.** +0 to +3. Low confidence — strong NNUE already overweights
promotion-threat squares; may be redundant.

**Status 2026-04-22:** **#637 +1.6 @ 25886g trending H1**, merged at `742e119`.
Above the zero floor; gained where the existing LMR exemption was not enough for
these critical pushes.

## Ranked shortlist (updated 2026-04-24)

**Tier 1 (high confidence, direct port from multiple strong engines):**

1. ~~**P1 Optimism**~~ — H0 twice (#636, #671). Parked pending K1/K2 SPSA branch tune.
2. ~~**P2 Halfmove-gated TT cutoff**~~ — **+1.2 Elo H1 ✓ merged (#628).**
3. ~~**P3 50mr-threatened mate downgrade in TT**~~ — H0 but merged as confident-correctness (#633).

**Tier 2 (novel but chess-sound):**

4. **N4 Halfmove-scaled pruning margins** — expected +1–4, requires retune. **STILL UNTESTED.**
5. ~~**N1 Twofold-in-history eval blend**~~ — H0 dropped (#691 +0.0 @ 28792g).
6. **P4 NNUE complexity blending** (with alternative proxy) — expected +1–5, moderate. **STILL UNTESTED.**

**Tier 3 (smaller / riskier / more speculative):**

7. ~~**N5 QS insufficient-material short-circuit**~~ — H0 dropped (#683 +0.3 @ 50894g).
8. **N2 Shuffling detector** — expected +1–3, novel. **STILL UNTESTED.**
9. ~~**N6 Promotion-imminent extension**~~ — **+1.6 Elo H1 ✓ merged (#637).**
10. **N3 Fortress soft-cap** — expected +0–4, high variance. **STILL UNTESTED.**

**Still-untested summary**: N4, P4, N2, N3 remain open. P6 SMP thread
voting is also untested (listed above in P6 section). P1 parked pending
focused SPSA. See `next_ideas_2026-04-21.md` §"Research threads for
Titan (2026-04-24)" for the current full research queue.

## Stop-trying list

Based on prior H0 history:

- **Halfmove-XOR Zobrist key (GHI mitigation)** — H0'd twice at 64MB TT. Reckless
  and Viridithas use it but at larger effective hash. **Revisit only if default
  Hash > 256MB becomes common.**
- **Quadratic halfmove scaling** (Reckless/Minic pattern) — H0'd at -3.0.
  Linear beats quadratic on v9 evals.
- **Contempt widening (±5, ±15)** — all H0'd. The contempt=0 point is optimal.
  Don't revisit unless eval-scale changes fundamentally.
- **Eval-only halfmove scaling at ALL halfmoves** (the `(193 - halfmove) / 200` variant
  that applied at halfmove=0) — H0'd because it dampened middlegame eval 3–13%.
  Current `(200 - halfmove) / 200` only starts dampening meaningfully past halfmove 40.

## Meta-observation

**Peripheral mechanisms have been under-experimented-on relative to their ROI.**
Outside pure search is ~1% of our experiment count but contains at least 5–8 +1-to-+5
Elo candidates per the above. Worth treating as a **systematic sweep** the same way
`signal_context_sweep_2026-04-19.md` treated threat-signal × pruning-context.

Proposed plan for Hercules: **P1 → P2 → P3 sequentially** (each unlocks the next or
shares infrastructure), then branch out to Tier 2/3 in parallel.

## Companion docs

- `next_ideas_2026-04-21.md` — move-ordering / search-side idea shortlist.
- `tunable_anomalies_2026-04-19.md` — contempt's removal history.
- `experiments.md` — source of truth for resolved SPRTs.
