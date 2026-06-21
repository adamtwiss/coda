# Why Viridithas scales better with time than Coda — and what to port

> **The ACTIVE work plan + running state lives in
> `tc_scaling_methodology_2026-06-21.md`** (the flex-formula + per-TC-SPSA-
> divergence loop). This doc is the value-level Viri-vs-Coda comparison that
> feeds it.

**Date:** 2026-06-20
**Trigger:** Adam's TC-scaling gauntlets showed Viridithas gains Elo as TC
grows while Coda *loses* it — they cross over around 40+0.4, and the newer
(June 2026) Viridithas scales ~+200 Elo over a 1×→4× range vs Coda's ~+140.
**Method:** git archaeology on the Viridithas repo (March `2536115` → June
`1ec9381`, 24 commits) + structural diff vs Coda `search.rs`/`tt.rs`/
`movepicker.rs`/`nnue.rs`.

## The empirical finding

Elo by TC on a fixed 20-engine gauntlet (Atlas):

| TC | Coda | Viridithas |
|---|---|---|
| 5+0.05 | **+63** | −15 |
| 10+0.1 | **+24** | −1 |
| 40+0.4 | −1 | **+21** |

Coda is far stronger at blitz and *declines* with time; Viridithas *gains*.
Direct self-scaling H2H (1×=10+0.1 → 2× → 4×): the June Viridithas scales
~+196 over the range vs Coda's ~+143, and overtakes Coda at 4×. A "bigger
hash (1024)" rematch at 40+0.4 specifically helped Viridithas — a strong hint
that **TT efficiency at long TC** is part of the story.

Time-scaling = (a) EBF (lower → more depth per time doubling) × (b) how well
the extra depth is *used* (extensions finding critical lines, eval-correction
accuracy, TT reuse). This sits squarely in Coda's known frontier: our **eval
is already SF-class** (Spearman 0.853 vs LC0, #2 of 12 engines), so the gap is
**search/effective-depth**, not eval.

## Headline: the edge is DIFFUSE

There is no single big lever. Viridithas's scaling advantage is a stack of
small EBF/depth-allocation refinements plus three structural levers (TT
replacement, continuation-correction history, pawn-pair eval). Notably **Coda
is *richer* than Viridithas in several areas** — our SE is more elaborate
(PV/quiet/corr-aware DEXT margins) and our threat eval has **x-ray, which
Viridithas entirely lacks** (+110/+187 Elo banked). So this is selective
porting, not wholesale copying. The gap is concentrated in TT retention,
fine-grained LMR, and two eval-correction signals.

## Findings, by leverage

### 1. TT replacement: quadratic age scaling + fractional priority (HIGH)
- **Viridithas** (`src/transpositiontable.rs:328-365`): insert priority =
  `depth + flag_bonus + (age_differential²)/4 + pv`; replaces if
  `insert_priority*3 >= record_priority*2`. Quadratic age keeps moderately-old
  deep entries but evicts very-old ones regardless of depth; flag/PV bonuses
  bias toward retaining Exact/PV entries.
- **Coda** (`src/tt.rs:478,497-498`): linear `slot_score = slot_depth - age*8`;
  key-match gate `depth > slot_depth-3 || gen!=slot_gen || exact`. No PV bonus
  in priority, no fractional 3:2 threshold.
- **Why it scales:** at long TC the TT fills with deep entries; quadratic aging
  is exactly what lets a big-hash search retain still-relevant deep entries
  while flushing stale ones — **directly explains the hash-bump observation.**
- **Port:** low-medium (replacement-policy swap in `store`; SPSA-retune the 3:2
  ratio + age coefficient). **Leverage HIGH. Validate at LTC, hash≥256** —
  TT-pressure-bound, can invert STC→LTC (`feedback_stc_can_invert_ltc_direction`).

### 2. Two LMR reduction terms Coda lacks (MEDIUM-HIGH, cheapest to test)
Both pure-EBF, both landed in Viridithas with **bench drops + Elo gains**
(4.63M→3.68M nodes) — the "force more pruning + retune" signature Coda favors.
- **`alpha_raises`** (commit `72c28e0` #431): `r += alpha_raises *
  LMR_ALPHA_RAISE_MUL` — each time a node raises alpha, *later* moves reduce
  more (once ≥1 improving move is found, remaining moves are less likely to
  beat it).
- **`ttpv_fail_low`** (commit `63a1db8` #432): `r += (ttpv && tt_value <= alpha)
  * LMR_TTPV_FAIL_LOW_MUL` — TTPV nodes whose cached score ≤ alpha look like
  fail-lows → reduce more.
- **Coda** (`src/search.rs:4188-4278`): has cut-node, improving, tt-pv-reduce-
  *less*, check, threat, king-pressure, complexity, history adjustments — but
  **neither** an alpha-raise counter **nor** a TTPV-fail-low term. Coda's
  `tt_pv` only ever reduces *less*; it never adds the fail-low penalty.
- **Port:** LOW (~6 lines each). Two independent `[0,3]` branches + focused
  retune. **Highest EV-per-hour first tests.**

### 3. Continuation-correction history as 2-move tuples (MEDIUM)
- **Viridithas** (commit `b171ce4` #424, `src/history.rs:227-292`): corrhist
  includes two *continuation* terms keyed by `PIECE_KEYS[ch1]^PIECE_KEYS[ch2]`
  for the (1-back,2-back) and (1-back,4-back) pairs, each with its own SPSA
  weight.
- **Coda** (`src/search.rs:856,1310-1317`): `cont_corr` is `[piece][to]` —
  **single-ply** continuation correction keyed only on the immediately
  preceding move. No 2-move tuple, no 1-4 plane.
- **Why it scales:** richer static-eval correction → better RFP/futility/NMP
  gating and the LMR `complexity` term → fewer mis-prunes at depth; eval-
  correction accuracy compounds at long TC.
- **Port:** medium (new zobrist-keyed table + two weights + update site).

### 4. Singular extension: depth-scaled margin + tree-depth cap (MEDIUM)
- **Viridithas** (`src/search.rs:1364-1400`): `r_beta = ce.value - depth*48/64`
  (margin grows ~0.75/ply); SE gate `depth >= 6 + ttpv` (PV nodes need MORE
  depth before SE fires); `height < root_depth*2` cap (don't SE deep in tree);
  triple extension live.
- **Coda** (`src/search.rs:3934-3935,3907`): `singular_beta = tt_score - depth -
  xray_bonus` (fixed 1.0/ply margin); gate `depth >= SE_DEPTH_10X` no TTPV bump;
  **no `height < root_depth*2` cap**; triple ext intentionally excluded.
- **Why it scales:** depth-scaled margin + TTPV-aware gate make SE fire more
  selectively but more often where it matters; the tree-depth cap is a pure-EBF
  guard against SE blowups deep in the tree. Extensions are the prime
  "use extra depth to find critical lines" lever.
- **Port:** LOW for the `48/64` margin + `height < root_depth*2` cap; don't
  wholesale-replace (Coda's SE is otherwise more elaborate).

### 5. Pawn-pair NNUE inputs (MEDIUM — training thread)
- **Viridithas** (commit `331e83f` #445): `PAWN_TUPLE_FEATURES = 96*95/2 = 4560`
  — a dense all-pairs pawn-structure-interaction input family (every unordered
  pawn pair, king-bucketed, learned weight row), in the int8 aux block.
- **Coda:** v9 has only the single P→P threat entry + pawn-hash corrhist — no
  dense pawn-pair table.
- **Why it scales:** pawn-structure games are long, quiet, deep — where extra
  depth pays and a richer pawn-structure eval discriminates.
- **Port:** HIGH (needs a net retrain with the new input family in Bullet —
  GPU/training thread, orthogonal to search). The one genuinely-new eval idea.
  (Do NOT port threat encoding — Coda's x-ray threats are richer than
  Viridithas's.)

## Lower-priority / corroborating

- **Threat-aware capture history** — Viridithas indexes
  `tactical[to_threatened][piece][to][victim]`; Coda's `[piece][to][victim]`
  isn't threat-aware (`movepicker.rs:35`). Small tactical-node EBF gain. LOW.
- **Stat-score-weighted multi-ply cont-hist gravity** — Viridithas modulates
  each cont-plane's gravity by a tunable-weighted sum across planes {1,2,4};
  Coda uses a single-plane base. LOW-MEDIUM.
- **`optimism` term** — Viridithas scales raw eval by a material-weighted
  optimism tracking the root average score; Coda removed contempt (was +2.53 to
  drop). Style/strength knob, not a scaling lever. LOW (skeptical).
- **Aspiration** — Viridithas narrows beta to midpoint on fail-low + reduction-
  scaled delta widening; Coda widens flat `delta += delta/2`. Minor stability.
  LOW.
- **RFP margin** (#444, 73→65) and `do_deeper` margins are parity-class.

## Follow-ups requiring measurement
1. **Direct EBF comparison** Coda vs Viridithas at matched depth — confirm where
   the depth-per-time gap really is (inferred from bench-node drops here). Report
   EBF alongside bench on any port branch.
2. **TT lever (item 1) is the best single bet** for the hash-bump observation
   but is TT-pressure-bound — validate at LTC hash≥256, not STC.
3. Items 1-4 are search-side, independently SPRT-able at `[0,3]` with retune-on-
   branch; item 5 is a GPU/training probe.

## Action plan
1. **LMR terms (#2)** — implement `alpha_raises` + `ttpv_fail_low` as two
   separate `[0,3]` branches with focused retune. Cheapest, highest EV.
2. **SE depth-scaled margin + tree-depth cap (#4)** — separate branch.
3. **TT quadratic age (#1)** — higher leverage, LTC validation, more involved.
4. **Pawn-pair (#5)** — queue as a training probe.

Key files for follow-up: Viridithas `src/transpositiontable.rs:298-385`,
`src/search.rs:1364-1500` (June master); Coda `src/tt.rs:437-508`,
`src/search.rs:3899-4007` (SE), `:4170-4296` (LMR), `:1310-1317`+`:856` (corrhist).

---

# viri.dev OpenBench — measured wins, regimes, and our ports (2026-06-20)

Viridithas runs its own OpenBench at **https://viri.dev** (`/index/`, `/greens/`,
`/test/<id>/`). Far higher-signal than structural archaeology: it shows which
changes actually *banked* Elo, **at what time control**, and by how much. **These
are ALL their notable search/eval winners going back to Feb 2026** — ~6 changes.
(For scale: we ran more non-model search experiments in one day than they ship in
~3 months. Velocity is our edge; their edge is regime-discipline — see the TC
lesson below.)

## Their measured winners (search/eval, excluding version-bump merges)

| change | Elo | their TC | test | what it is |
|---|---|---|---|---|
| **pawn-pawn** (pawn-pair NNUE) | **+12.54 ±5.29** | 25 knodes | /test/663 | dense pawn-pair eval inputs — biggest single win |
| **alpha_raises** (LMR) | **+5.38 ±2.96** | **LTC 40+0.4** | /test/629 | reduce later moves more after alpha rises |
| **ttpv_fail_low** (LMR) | **+5.00 ±2.83** | **STC 8+0.08** | /test/626 | reduce TTPV nodes more when cached value ≤ alpha |
| **bad-capture-corrhist** | +4.77 ±2.84 | STC 8+0.08 | /test/575 | new corrhist source keyed on bad/losing captures |
| branchless-threat-indexing | +2.51 ±1.86 | STC 4+0.04 | — | NPS/threat-indexing opt |
| tune-extract-rfp | +1.27 ±1.01 | STC 8+0.08 | — | RFP parameter tuning |
| (`master vs v19.0.1` = +52.65 LTC — cumulative version bump, not one change) |

## THE LESSON: match their test TC — scaling features are LTC features

The most important methodology takeaway. **alpha_raises and ttpv_fail_low are
both "reduce later/worse moves more" LMR carve-outs, but Viridithas tested one at
LTC and one at STC — deliberately, because they behave differently by regime:**
- **alpha_raises** needs *depth* (multiple alpha-raises per node only happen with
  the depth to search many improving moves). Viridithas: **+5.38 at LTC**. Coda
  STC test (#2160): **−3.5** — it *actively hurts* at STC. The LTC re-test (#2163)
  is the only valid measurement. **We would have wrongly killed a real win on the
  STC result.**
- **ttpv_fail_low** is an STC feature for them (+5.00 STC). Coda STC test (#2161):
  **+1.9** (real but below their +5 and below the [0,3] cliff → likely H0; a
  focused retune of `LMR_TTPV_FAIL_LOW_10X` is the move if so).

**Rule (strong enough to bank): scaling/depth-class features — EBF, extensions,
TT, reductions, "use-the-extra-depth" — default to LTC (40+0.4, Hash=256), not
STC.** STC stays right for genuine STC features and breadth carve-outs. Coda is
*already* strong at STC; the deployment regime (lichess/CCRL, deep) is where the
scaling gap lives, and STC can return the wrong sign there. When porting a
specific Viridithas change, **use their test's TC** — their OB tells us both the
right regime and the expected magnitude.

## Our port status (2026-06-20)

| port | branch | test | regime | status |
|---|---|---|---|---|
| alpha_raises | `experiment/lmr-alpha-raises` | 2163 | LTC (re-fired) | live |
| ttpv_fail_low | `experiment/lmr-ttpv-faillow` | 2161 | STC | +1.9, likely H0 → retune candidate |
| TT quad-age (Step A) | `experiment/tt-quadratic-age` | 2162 | LTC Hash=256 | live, early |
| bad-capture-corrhist | — | — | STC | not yet ported |
| pawn-pair NNUE | — | — | — | training probe (plan below) |

---

# Pawn-pair NNUE features (`pawn-pawn`, +12.54) — implementation plan

Their **biggest single eval win**, and the one **Coda's architecture is uniquely
ready to absorb** (we already have the threat aux block + accumulator it slots
into). Source: Viridithas commit `331e83f`.

## What it is
Every **(file-proximity-masked) pair of pawns** gets its own learned NNUE weight,
**king-bucketed and color-aware**. Single-pawn features can't express pawn-structure
*interactions* (doubled/isolated/chains/phalanxes/opposing tension); the all-pairs
family learns them directly. ~2,000–4,500 features after the file masks
(`PAWN_PAWN_MASKS` restrict by file distance; full unmasked would be 96·95/2 =
4560). Rides in the **same int8 "aux" input block as the threat features**,
concatenated into the FT input (threats offset by `PAWN_TUPLE_FEATURES`), and is
**incrementally updated** by diffing the pawn bitboard `afore`/`after` each move →
adds/removes. Key Viridithas symbols: `pawn_pawn_index()`, `add_pawn_pawn_indexes()`,
`PAWN_PAWN_MASKS`, `AuxUpdateBuffer` (renamed from `ThreatUpdateBuffer`),
`vector_update_aux()`.

## Why it maps cleanly to Coda v9
Coda v9 = FT(1024) + **threat aux block** → L1 → L2 → output, with a full
**threat accumulator** (`threats.rs` / `threat_accum.rs`) already doing exactly
this before/after incremental aux update. Viridithas added pawn-pairs *as a
sibling family in that same aux block*. So the port is "**add a pawn-pair family
alongside threats**," mirroring the threat accumulator — not new infrastructure.
**Genuinely additive**: Coda's v9 threats are pawn *attacks*; pawn-hash corrhist
is search-side; neither is the dense structural all-pairs input.

## Effort — two threads
1. **Bullet (training):** add the pawn-pair input family to the `coda_v9_768_threats`
   config (feature enumeration + masks) and retrain. A new arch variant
   (v9 + pawn-pairs); needs a full SB run to learn the weights. The bigger piece.
2. **Coda (inference):** a `pawn_pairs` accumulator paralleling `threat_accum.rs`
   (masked pair indices, before/after pawn-bitboard diff, int8 apply into the aux
   block). Extension of existing code, not from scratch.

## Caveat (honest leverage)
Coda's eval is already SF-class (#2/12, Spearman 0.853) and the strength frontier
says the bottleneck is **search/depth, not eval** — so +12.54 of *eval quality*
may not transfer 1:1 to Coda strength the way it did for Viridithas (more eval
headroom). BUT pawn-structure games are long/deep (Coda's deployment regime), it
closes a *different* signal than threats, and +12.54 is large enough that even a
partial transfer is worth a probe.

## Recommendation
Scope as a **training probe for when a GPU frees** (post-multi-v7, or gpu2 after
the WDL probe). Highest-value single change *and* the one our architecture is
ready for — but a retrain, so it queues behind the prod run rather than jumping it.

---

# Value-level comparison (2026-06-21): WHY Coda over-prunes at depth

The earlier sections covered structure + measured viri.dev wins but not the
*values*. This section is the param-by-param audit (4-agent sweep over Viri
`1ec9381` latest vs Coda trunk), and it converges on a single thesis.

## The measured scaling curve (Adam's gauntlet, 2026-06-21)

| TC | Coda | Viri | Coda−Viri | Coda gain/doubling | Viri gain/doubling |
|----|------|------|-----------|--------------------|--------------------|
| 1× | −71  | −115 | **+44 Coda** | — | — |
| 2× | +6   | +31  | **−25 Viri** | +77 | +146 |
| 4× | +72  | +81  | −9 (~tied)   | +66 | +50  |

Coda is **stronger at STC (+44)** but Viri overtakes by 2×; the swing is
**front-loaded in the first doubling** (Viri +146 vs Coda +77). By 4× they
re-converge (Viri only +9, within CI). **The symptom is draws, not losses:**
Coda-2× draws **80.2%** vs Viri-2× **69.5%** — Coda is *drawing winnable
games* at longer TC, not getting tactically beaten. That is a
conversion / deep-resource signature, and it points the diagnosis at
**forward pruning that discards the deep quiet resources needed to convert**,
not at tactical oversight.

## THESIS: Coda's forward pruning is STC-shaped; it over-prunes at LTC depths

Coda's tunables are SPSA-tuned at STC, where the tuner inflates forward-pruning
aggression (pruned nodes → depth, which wins at STC). Those same params, at the
depths LTC reaches, prune the long-horizon quiet resources that decide
draw-vs-win. Viri's shapes are structurally LTC-robust (hard depth caps, linear
rather than depth² margins), so they don't over-prune deep. The unit-independent
*shape* divergences (not the absolute values) are the robust findings.

### The forward-pruning over-aggression cluster (highest LTC leverage)

All values normalized to Coda-cp (Viri eval ≈ 2.33× Coda: SEE pawn 233 vs 100).

| feature | Coda | Viri | divergence (shape) |
|---|---|---|---|
| **RFP depth gate** | `depth ≤ 16` | `depth < 9` | Coda makes the reverse-futility static-eval bet to **~2× the depth**. Coda bolts on a superlinear "deep-knee" (DEEP_LINEAR/QUAD) + root-coef widener to stay safe past d8; Viri just **caps at d8**. |
| **SEE-quiet pruning** | `−33·lmr_d²` (**quadratic**) | `−62·d` linear (≈−27·d Coda-cp) | Quadratic is the textbook STC-overfit: at LTC lmr_d hits 6–10 and the spared-quiet threshold explodes (−33·64 ≈ −2112cp at lmr_d=8) → prunes deep defensive/prophylactic quiets. Viri's linear shape spares them. |
| **NMP base reduction** | base R ≈ 7.8 | base R = 4 | Coda nulls away ~2 extra ply at the base of the NMP reduction before depth/eval terms — across the whole zugzwang-risky tree. |
| **Futility** | base 80, per-depth 110, gate lmr_d≤15 | base 37, per-depth 30 (Coda-cp), gate lmr_d<6 | Coda's futility band is 2–4× wider AND gates **2.5× deeper** — still eval-pruning quiets at depths where Viri stopped trusting static eval. |

Coda's RFP-root-coef / deep-knee / unstable-threat margin wideners are evidence
the authors already *diagnosed* this failure mode and patched it with
LTC-protective widening rather than Viri's simpler depth caps. SEE-quiet has no
such LTC term at all.

### LMR: integer ±1 quantization vs Viri fixed-point fractional

Viri accumulates the whole reduction in 1024-units (every per-condition
adjustment is a *fraction* of a ply, one `÷1024` at the end); Coda applies each
condition as integer `±1`. Where Coda's ±1 is materially weaker: **cut-node**
(Viri −1.56 ply vs Coda 1.0 → Coda searches the dominant cut-node tail ~0.5 ply
deeper *per node*), ttpv (1.26 vs 1.0), in-check (1.33 vs 1.0). Per-condition
LMR muls Viri has that Coda lacks: **killer/refutation reduce-less (−0.76 ply)**,
**alpha-raise progressive reduction** (Coda removed FEAT_ALPHA_REDUCE 2026-06-06;
re-added and tested on Coda at BOTH TCs — **+1.09 STC (#2160) / +1.32 LTC
(#2161), ≈flat** — NB the +5.38 figure was Viridithas's own measurement on their
engine, not a Coda LTC number), **ttpv→lmr_depth** (feeds the pruning gate, not
just reduction). Coda's
`do_deeper`/`do_shallower` re-search-depth margins are **hardcoded
(60+10·r / 20)** and flagged un-bisected; Viri exposes them to SPSA — these
literally govern how re-search depth grows (the LTC depth knob).

### Structural capabilities Viri has that Coda lacks

- **Triple extensions** (+3 ply on a quiet TT move failing singular by >text_margin;
  `dextensions≤12` guard). Coda caps at +2. Deep forced lines are the canonical
  SPRT-invisible / LTC-amplifying feature.
- **Correction-history breadth**: Viri has 6 sources (pawn, major, minor,
  nonpawn, cont-12, cont-14); Coda has 4 and **dropped major+minor** (2026-05-19)
  and has only one continuation-corr source. Also Coda's `CORR_W_NP` is **0.21×
  its pawn weight** vs Viri's **~parity** — under-weighting the non-pawn key that
  carries the positional signal in deep endgames (conversion!).
- **Adaptive aspiration delta growth**: Coda widens a fixed ×1.5 on every fail;
  Viri widens gently (×1.34) then *escalates* with consecutive fails — fewer
  nodes on the common single-fail case, the bulk of deep-iteration re-search cost.
- **Optimism**, **material-scaled eval**, **eval-policy (value-difference history
  update)**, **adaptive/eval-scaled ProbCut depth (cj5716)** — all absent in Coda.
- **Shallower singular gate**: Coda SE at depth≥4 vs Viri depth≥6/7 — Coda spends
  verification searches 2 ply shallower (cheap at LTC but noisier TT).

## Ranked portable candidates (LTC-scaling leverage)

The draw-conversion symptom ranks the forward-pruning shape fixes first.

1. **Linearize SEE-quiet pruning** (`−33·lmr_d²` → linear `−k·lmr_d`). The
   cleanest STC-overfit; no LTC-protective term exists today. Directly spares
   deep quiet resources.
2. **RFP depth-gate / shape** — cap nearer Viri's d8–9 (or test whether the
   deep-knee can be removed once capped). Stops the deepest reverse-futility bets.
3. **Triple extensions** — one-tier addition on existing singular/dext
   machinery; SPRT-invisible at STC, the LTC profile we want.
4. **NMP base R + futility width/depth** — pull toward Viri's less-aggressive
   shapes; the STC core tune (#2166) just pushed several of these the *wrong way*
   for LTC, so this is partly an STC-vs-LTC retune-shape question.
5. **CorrHist: reweight CORR_W_NP toward pawn-parity + add cont-14** — eval
   accuracy compounds over depth; cheap (one reweight + one table).
6. **Adaptive aspiration delta growth** — isolated deep-iteration re-search cost.
7. **LMR fixed-point fractional** — meta-change un-quantizing cut-node (1.56 vs
   1.0) and enabling alpha-raise/refutation muls to tune as fractions.

## METHODOLOGY — validate in the scaling gauntlet, not by STC SPRT

Coda is already **+44 ahead at STC**; the deficit is purely a scaling/conversion
problem that appears at 2×+. So an STC-only "does it help?" SPRT is the wrong
acceptance frame here — but note we should NOT assume a big LTC *amplification*
either: Coda's own alpha-raise port was **+1.09 STC / +1.32 LTC (≈flat)**, i.e. a
small TC-stable win, not a STC-invisible-then-large-at-LTC effect (the +5.38 was
Viri's number on their engine). So the honest position is: (a) the
forward-pruning *shape* changes (SEE-quiet quadratic→linear, RFP depth cap, NMP
base, futility width) have a *mechanistic* LTC argument — they bite at depths STC
rarely reaches — but that amplification is **unvalidated for Coda**; (b) we have
no Coda evidence these amplify, and the alpha-raise data point is mildly
cautionary that ports may just be TC-stable.

**Therefore validate directly in the 2×/4× scaling gauntlet** (the harness that
produced the table above) — it measures the actual scaling effect, rather than
relying on an LTC-SPRT assumption. A cheap first pass: apply ONE shape fix and
re-run the gauntlet; if the 2× draw-rate drops / the 2× Elo closes, the lever is
real. Independently, **STC-tuning these would push the wrong way** (toward more
STC aggression — exactly what the #2166 STC core tune did to RFP/FUT/SEE), so
even if a candidate is TC-stable, don't let an STC tune walk it back.
