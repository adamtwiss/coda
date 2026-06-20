# Why Viridithas scales better with time than Coda — and what to port

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
