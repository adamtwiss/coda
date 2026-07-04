# Raphael Search/Eval Review (2026-04-19)

Raphael 4.3.0 (dev), C++20, `src/Raphael/*.cpp` + `src/eval/*.cpp`. CCRL Blitz 4.2.0
= 3740 / SPCC #11 — a top-25 engine, comparable Elo to Coda but stronger in the
latest CCRL. Source of strength is heavily on the **eval side**: an unusual
1024-neuron pairwise net with **60,144 threat features** as int8 weights
(alongside 11×64 = 704 PSQ features as int16), 16 input king buckets, 8 output
buckets, and a 32→32 hidden-layer tower. Search is a solid modern PVS with
fractional depth (`DEPTH_SCALE=128`), NUMA-aware Lazy SMP, and 5-slot TT
clusters, tuned via ~180 exposed SPSA parameters. Not particularly novel on the
search side — this review focuses on the eval and the handful of search bits
that are genuinely new.

## Search architecture

- **Fractional depth** (`Raphael.cpp:431`, `tunable.h:186` `DEPTH_SCALE=128`) —
  every depth quantity is stored `×128`, so LMR/IIR/ext tunables have
  sub-integer granularity. PVS negamax with `cutnode: bool` propagated
  (`Raphael.cpp:474-484`, `!cutnode` into NMP zero-window,
  `!cutnode` into re-searches; `cutnode` gates TT cutoff and drives
  singular-negative extensions).
- **Aspiration** (`Raphael.cpp:418-451`): standard delta-widening +
  `asp_fred` — on every fail-high, `asp_fred += ASP_RED (150)` clamped to
  `ASP_MAX_RED (370)`; the inner search runs at `depth*128 - asp_fred`, i.e. a
  soft *fractional* depth reduction that grows monotonically per fail-high.
- **Pre-move-loop pruning** (`Raphael.cpp:579-681`):
  - RFP: `depth<=RFP_MAX_DEPTH (995/128≈7.8)`, margin =
    `RFP_MARGIN_DEPTH_MUL(36)*d - improving*RFP_MARGIN_IMPROVING(33)
    - opp_worsening*RFP_MARGIN_OPP_WORSENING(17)
    + corrplexity*RFP_MARGIN_CORRPLEXITY(100)/1024`.
  - Razoring (`Raphael.cpp:592-599`): `alpha<=2048`,
    `d<=RAZOR_MAX_DEPTH(5.1)`, margin = 291 + 223·d².
  - NMP (`Raphael.cpp:601-636`): R = `NMP_RED_BASE(420) + d·200/1024
    + min((eval-beta)·82, 384)/128`; gated by `!(ttFlag==UPPER &&
    ttScore<beta)`, `!is_kingpawn(stm)`, prev move exists,
    `eval>=beta+nmp_margin`. Verify at `depth>=NMP_VERIF_MIN_DEPTH(15)`,
    else return score directly. Post-verify sets `min_nmp_ply = ply +
    NMP_VERIF_DEPTH_FACTOR(96) * red_fdepth / (128·128)`.
  - **Hindsight extension** (`Raphael.cpp:580-583`, non-PV pre-loop):
    if parent's LMR reduction ≥ `HINDSIGHT_MIN_RED(420/128)` and
    `staticEval + prevStaticEval <= 0` (opp not worsening), `depth += HINDSIGHT_EXT(150/128)`.
  - ProbCut (`Raphael.cpp:638-680`): `beta+PC_MARGIN(263)`, gated on
    `!ttpv`, `!ttmove || !ttmove_quiet`, TT miss below pc_fdepth. SEE
    threshold scales with `(pc_beta - static_eval) * 125 / 128`.
- **Move-loop pruning** (`Raphael.cpp:702-733`):
  - LMP: `move_searched >= LMP[improving][d] + hist*LMP_HIST_MUL(512)/2^23` —
    threshold *raised* by good history (proceed further before pruning).
  - Futility: `staticEval + FP_BASE(91) + FP_DEPTH_MUL(44)*lmrDepth
    + FP_HIST_MUL(396)*hist/HISTORY_MAX <= alpha`, gated
    `!in_check && !gives_direct_check`.
  - SEE prune: quiet `-23*lmrDepth²/128²`, noisy `-95*d/128`. Note the
    quiet uses lmrDepth (consensus with SF), signed with SPSA min bound `-128`.
- **Extensions/reductions inside move loop** (`Raphael.cpp:736-822`):
  - SE with SE/DE/TE tiers (`Raphael.cpp:754-759`): margins
    `DE_MARGIN_BASE(30) + isPV·245`, `TE_MARGIN_BASE(107) + isPV·723`; triple
    ext only if quiet. Multi-cut returns `(score+beta)/2`. Cutnode negative
    ext `-CUTNODE_NE_RED(132/128)`. Plain negative ext `-NE_RED(123/128)`
    when `tt_score>=beta`.
  - **LDSE** (`Raphael.cpp:767-772`): low-depth singular extension —
    at `fdepth<=LDSE_MAX_DEPTH(732/128≈5.7)`, `!in_check`, `ttFlag==LOWER`,
    if `ss->static_eval <= alpha - LDSE_MARGIN_BASE(24) +
    LDSE_MARGIN_CORRPLEXITY_MUL(102)*corrplexity/16384`, extend by
    `LDSE_EXT(127/128)` **without a verification search**.
  - LMR: separate quiet/noisy base+div tables (`update_lmr_table`), adjustments
    `+LMR_NONPV(38) -LMR_IMPROVING(112) +cutnode·LMR_CUTNODE(208)
    -gives_check·LMR_CHECK(178) -hist/LMR_HIST_DIV -corrplexity/LMR_CORRPLEXITY_DIV(276)`.
  - do_deeper/do_shallower with depth-scaled margins
    (`DO_DEEPER_BASE(40) + 6·d`), fractional-depth ext/red.
- **Correction history** (`corrhist.cpp:99-127`, weights `tunable.h:353-357`):
  five sources — pawn(37) + **major(60)** + white-NP(63) + black-NP(63) +
  cont1(70) + cont2(57). Cont-corr keyed `[prev_piece][prev_to][curr_piece][curr_to]`,
  at ply-1 AND ply-2 offsets (both), thread-local. Pawn/major/nonpawn shared
  across threads, atomic i16. **`major_hash` corrhist is Raphael-specific.**
- **SMP**: NUMA-aware. Each thread has its own history + cont-corrhist; the
  three board-hash-keyed correction tables are shared per NUMA node
  (`Raphael.h:110`, `numa::NumaUniqueAllocation<SharedCorrectionHistory>`).
  Thread voting at end (`Raphael.cpp:257-322`) — weight = `(score-minScore+10)·depth`,
  vote count-weighted by move, tie-break with PV length.

## Move ordering

`movepick.cpp` — clean staged picker copied from Stormphrax (SIMD selection
sort at `movepick.cpp:297-315`). Stages TT → gen noisy → **good_noisy** → gen
quiet → quiet → bad_noisy. Split threshold is
`GOOD_NOISY_SEE_BASE(-1) - captHist·GOOD_NOISY_SEE_MUL(363)/1024` — high
capture history lets a losing SEE cross to good, low capture history demotes
further. History axes:

- **Main quiet**: `butterfly[from][to][from_attacked][to_attacked]` —
  4D threat-aware butterfly (identical shape to Coda's main_hist,
  `history.h:30`).
- **Continuation**: `cont_hist[prev_piece][prev_to][curr_piece][curr_to]`
  at plies 1, 2, 4 with SPSA weights 123, 125, 82 (of 128)
  (`history.cpp:60-78`).
- **Capture**: `capt_hist[from][to][victim]` — from-to-victim (not
  piece-to-victim; Reckless / Stormphrax pattern).
- **`update_with_base`** (`history.cpp:20-24`, applied on cont writes
  `history.cpp:47-52`): every cont-hist update is written against the *sum of
  all three cont scores* rather than the local entry — Stormphrax T6 pattern.
- Quiet score = `butterfly + get_conthist`; noisy score = `capthist/CAPTHIST_DIV(8)
  + SEE_TABLE[victim] + promo bonus` (`movepick.cpp:270-283`).
- **`DIRECT_CHECK_BONUS(5012)`** added to *every* quiet's ordering score if
  it gives check (`movepick.cpp:292`) — unconditional, no SEE gate.
- No killers, no counter-moves. History-only ordering.

## NNUE / eval

Architecture (`arch.h:8-30`):
- **FT**: two feature families feeding a 1024-wide accumulator
  - `W0_psq[16][11·64][1024]` int16 — **11 piece types × 64 squares** (not 12; both
    kings are collapsed to `WHITEKING`, `accumulator.cpp:119-121`); 16 input
    king buckets, 32-square layout mirrored (`arch.h:19-28`).
  - `W0_ti[60144][1024]` int8 — **threat features**, keyed by
    (attacker piece, attacked piece, attacker sq, attacked sq)
    with piece-pair de-dup so mutual attacks encode only once
    (`accumulator.cpp:14-112`). ~60K features; the largest threat-feature
    table of any engine we've reviewed.
- **L1**: SCReLU pairwise on the 1024-accumulator → **512 activated u8**
  (`nnue_multilayer.cpp:66-122`, pairwise mulhi with `<<7` shift), then int8
  matmul via `dpbusd` into 32×float. **Sparse iterator** — tracks nonzero
  4-tile blocks (`sparse.cpp`, `nnue_multilayer.cpp:137-184`), executes L1 only
  on nonzero tiles. 8 output buckets baked into L1/L2/L3 weights, selected by
  `board.occ().count()-2` divided into 8 bins.
- **L2**: 32 → 32 int32 dot-product with i32 weights, SCReLU clamp.
- **L3**: 32 → 1 int32 accumulator + bias. Final scale by `OUTPUT_SCALE(253) / QC⁴`.
- Quantisation: `QA=255` (accumulator), `QB=128` (threat-index weights),
  `QC=64` (hidden). `L1_SHIFT=8` post-pairwise.
- **Finny table** (`accumulator.cpp:140-215`): per-perspective + per-input-bucket
  cache, SIMD 8-chunk `add4`/`sub4` refresh (chunked feature apply).
- **Weight permutation** (`perm/permute.cpp` in the tree): the network file
  is *permuted at load time* to interleave FT weights for
  `packus`-friendly SIMD (`nnue_multilayer.cpp:100-104`, `Nnue::NnuePerm` flag
  in the file header).

**Eval scaling** (`position.h:170-179`): material scaling only,
`eval * (25100 + pawns·110 + knights·340 + bishops·340 + rooks·590 + queens·970)
/ 32768`. Plus halfmove decay `* (200 - hm) / 200` (`Raphael.cpp:390`).
**No optimism/contempt.** Draw score is
`(nodes & 0x2) - 1` — deterministic ±1 alternator per node.

Training documented only via `network.txt` pointing at
`Raphael-Net/releases/yogsothoth_v4.nnue`; the Bullet trainer / T80-style
config is external.

## Notable / novel mechanisms

1. **60K int8 threat features as a distinct FT half.** The threat half is
   int8 (not int16 like PSQ) and applied through a separate weight matrix
   `W0_ti`, keyed by attacker/attacked piece + squares with piece-pair
   de-duplication logic that reads as a first-class geometry table
   (`accumulator.cpp:14-112`, `PIECE_TARGET_MAP`, `ATTACK_INDICES`). Nothing
   else we've reviewed carries this many threat features.
2. **`major_hash` correction axis** (`corrhist.cpp:53-60`) alongside pawn +
   non-pawn. Coda has pawn + WNP + BNP + cont + trans; the major-piece axis is
   Raphael-specific.
3. **`corrplexity` used as a search-shape signal in three places**: RFP
   margin (widen when uncertain), LMR reduction (reduce less when uncertain,
   already in Coda), and LDSE margin (extend more aggressively when uncertain).
4. **RFP `opp_worsening` term** — `staticEval + prevStaticEval > 0` widens
   the RFP margin further. Uses *sum of consecutive static evals* as an
   opponent-worsening proxy rather than the more common self-improving diff.
5. **LDSE** — single-search low-depth singular extension without a verification
   search, gated by static eval below alpha − margin.
6. **Aspiration `asp_fred` cap** — soft, growing fractional depth reduction
   per fail-high with a cap, instead of a boolean `-1` per fail-high.
7. **Deterministic draw-score jitter** — `(nodes & 0x2) - 1` at every node.
8. **Thread voting with PV-length tiebreak** — score, then vote weight, then
   `pv->length > 2` (`Raphael.cpp:317-321`).

## Testable Experiments for Coda (ranked)

Survivors after cross-checking Coda src + grep of `experiments.md`. All of
Raphael's other search bits (hindsight ext, cutnode LMR, from-history,
tt_pv gradient, prev-move corrhist, RFP corrplexity, complexity LMR,
razoring, `update_with_base`, LMP-hist adjust, futility-hist, LMR reduce for
checks) are **already done or already H0'd in Coda** — see the closed-out
list below. What remains is small.

### E1. `major_hash` correction history axis

- **Raphael**: `corrhist.cpp:53-60, 140-141`. Fifth board-hash-keyed
  correction table alongside pawn / white-NP / black-NP, keyed on
  `board.major_hash()` (rook + queen positions only). Weight
  `MAJOR_CORRHIST_WEIGHT=60`, comparable to pawn(37) and nonpawn(63) —
  a first-class signal, not a fallback. Coda's `nonpawn` axis mixes rooks +
  queens with minors; splitting them lets each axis converge on a distinct
  eval bias.
- **Coda today**: `search.rs:911-913` — pawn + white-NP + black-NP +
  cont + trans. No dedicated major or minor sub-key.
- **Prior art**: NONE for `major_hash`. Coda's minor/major split
  attempt in the reject log (`experiments.md:2672`) mixed *pawn + WNP + BNP +
  minor + major* into an equal-weight blend that diluted signal. A single
  additional axis with an SPSA-tunable weight is a different experiment.
- **Sketch**: add `board.major_hash` = zobrist XOR of rook + queen squares
  (already available if hash accumulation is present, else O(64) at leaf),
  4th correction table `[stm][major_hash % CORR_HIST_SIZE]`, `CORR_W_MAJOR`
  weight tunable, plumb through `update_correction_history` and
  `correction_value`. ~50 LoC.
- **Magnitude/risk**: +1-3 Elo. Retune the corr-weight tunables on the branch
  (adding an axis shifts the optimum split). Low-risk, but small.

### E2. `opp_worsening` term in RFP margin

- **Raphael**: `Raphael.cpp:575-590`. `opp_worsening_rate = staticEval +
  prevStaticEval` (i.e. `−(−prevSTM_eval − STM_eval)`); when
  `opp_worsening_rate > 0`, RFP margin is *reduced* by
  `RFP_MARGIN_OPP_WORSENING(17)` — prune more when opponent is worsening.
  Also used in **hindsight extension gate** (`Raphael.cpp:581-583`) —
  only extend if the opponent isn't already worsening.
- **Coda today**: `search.rs` RFP has `RFP_MARGIN_IMP/NOIMP` (self-improving
  only). Hindsight ext gate uses `(prior_static + current_static > X)` on a
  different axis. No `opp_worsening = ss.eval + ss-1.eval` signal exists as
  a search knob.
- **Prior art**: NONE. `experiments.md` "worsening" hits (line 13292) are all
  time-management docstrings, not search signals. RFP-improving is separate.
- **Sketch**: at RFP site, compute `prev_static = (ss-1).static_eval`,
  `opp_worsening = (static_eval + prev_static) > 0`, and subtract
  `RFP_MARGIN_OPP_WORSE * depth` from the margin (i.e. prune slightly *more*).
  Two-liner + one tunable. The `staticEval + prevEval > 0` idiom re-encodes
  "opp is not gaining" cheaply.
- **Magnitude/risk**: +1-3 Elo. Small, coupled to RFP tune. Retune RFP margins
  on the branch. Low risk.

### E3. LDSE (low-depth singular ext without verification)

- **Raphael**: `Raphael.cpp:767-772`. When `fdepth<=LDSE_MAX_DEPTH(5.7)`,
  `!in_check`, `tt_flag==LOWER`, and
  `staticEval <= alpha - LDSE_MARGIN_BASE(24) + LDSE_MARGIN_CORRPLEXITY(102)·corrplexity/16384`,
  extend the TT move by `LDSE_EXT(127/128) ≈ 1` ply. **No search cost** —
  purely a static-eval + TT-flag gate. Also carried by Hobbes (Hobbes
  review §"Gated / lower-priority"). Cross-engine: Reckless / Stormphrax /
  Obsidian variants exist.
- **Coda today**: singular extension is gated on
  `depth >= singular_min_depth` with a full verification search. No
  low-depth zero-cost singular gate.
- **Prior art**: NONE in `experiments.md` for LDSE specifically. Coda's SE
  has a long H0 history (§2076-2092), so this rides on top of a fragile
  primitive.
- **Sketch**: at TT-move loop entry, if the SE-full-search gate fails but the
  low-depth gate passes, set `extension = 1`. Two tunables (margin, corrplexity
  mul). ~15 LoC.
- **Magnitude/risk**: +1-3 Elo if healthy, 0 if Coda's SE is still shaky.
  Do NOT lead with this — SE_health first. Low effort but blocked on
  SE diagnosis.

### E4. `corrplexity` as an RFP margin term

- **Raphael**: `Raphael.cpp:586-590`, RFP margin += `corrplexity·100/1024`.
  Widen the margin when the correction is large — don't prune under uncertainty.
- **Coda today**: RFP has no complexity term. Coda already uses corrplexity
  in LMR (`search.rs:4607-4610`).
- **Prior art**: **H0 #1859/#2442** — "V5 Complexity-Adjusted RFP" tested
  twice with `margin += complexity/2`, both flat. Note: Raphael's coupling
  is `100/1024 ≈ 0.098` — **10× lighter** than Coda's test at `/2 = 0.5`.
  Coda's magnitude was almost certainly too aggressive.
- **Sketch**: `rfp_margin += (correction.abs()) * tp(&RFP_MARGIN_CORR) / 1024`,
  default 100. Since Coda uses `scaled_eval - static_eval` as its
  complexity signal (already computed for LMR), the term is free at RFP.
- **Magnitude/risk**: +0-2 Elo. **Low-priority** — H0 twice at 5× stronger
  coupling. Only worth a retest with the Raphael-scale margin (≤10cp per
  100cp of correction). Consider bundling with E2.

### E5. Aspiration `asp_fred` soft cap (growing fractional reduction with cap)

- **Raphael**: `Raphael.cpp:429-451`. On fail-high, `asp_fred = min(asp_fred +
  ASP_RED(150), ASP_MAX_RED(370))`; the inner search runs at
  `max(depth·128 - asp_fred, 128)`. So the first fail-high softly reduces by
  ~1.17 plies, the second caps at ~2.9 plies.
- **Coda today**: aspiration keeps depth constant across fail-highs.
- **Prior art**: H0 #1319, #2418, #3297-3300 — three separate "aspiration
  fail-high depth reduce" attempts, all at `-1` per fail-high (hard).
  Raphael's `asp_fred` is a *fractional*, *bounded*, *SPSA-tuned* growth —
  a materially different mechanism.
- **Sketch**: introduce fractional aspiration depth in the outer loop
  (either add DEPTH_SCALE OR keep integer depth and reduce only after the
  2nd fail-high). Requires plumbing but small.
- **Magnitude/risk**: 0-2 Elo. **Speculative** — three prior H0s on the
  simpler variant. Do only if E1-E4 all pass and we're mining for margins.

**Not-worth-porting on eval side:**

- 60K threat features (E6-tier NNUE): Coda's v9 threat architecture already
  targets 768-accum with a smaller threat scheme (Reckless-style). Bumping to
  60K would be a full training rewrite, not a search port. Note the *pattern*
  (int8 threat weights + int16 PSQ + sparse iterator across nnz tiles) for
  when we consider v10.
- Weight permutation at load time: Coda already handles quant/perm via the
  Bullet loader.

## Confirmed-clean / not-worth-porting

Items where Coda already matches or has explicitly H0'd Raphael's approach:

- **Cutnode propagation + cutnode LMR bump**: Coda has `cut_node` propagation
  and `LMR_CUTNODE_BUMP` tuned (`search.rs:206-208, 4528`). Raphael's
  `LMR_CUTNODE(208)` is the same axis.
- **Hindsight extension**: Coda merged #1130 (Stormphrax pattern). Raphael's
  version differs only in gate (opp_worsening vs prior+curr static eval sum) —
  see E2 for the reusable half.
- **`update_with_base` cont-hist**: Coda merged #1129 (Stormphrax T6).
- **Cont-hist plies**: Coda uses 1/2/4/6, Raphael uses 1/2/4. Coda is ahead
  here (#1326 4-ply reads).
- **Threat-aware butterfly**: Both use `[from][to][from_att][to_att]`.
- **Razoring**: Coda **removed** (-19.8 Elo, `experiments.md:3575`). Do not
  restore. Raphael keeps it, but Raphael is different fundamentally.
- **Complexity-adjusted LMR**: Coda has it (`search.rs:4607-4610`).
- **Aspiration `-1` on fail-high**: H0 #1319/#2418. E5 is a different shape.
- **Cont-corr multi-source blending**: Coda's cont_corr is 1-ply. Raphael has
  1-ply *and* 2-ply. **A 2-ply `cont_corr` axis is Hobbes E2** already —
  see `engine-notes/hobbes.md` §E2. Not double-listed here.
- **LMR-check reduce-less**: Coda H0'd `reduction--` for checks (#1447),
  keeps full LMR skip. Raphael's `-LMR_CHECK(178)` is the same mechanism
  we rejected.
- **LMP-hist adjustment**: Coda H0'd twice (§317, §1940). Raphael's
  `LMP += hist·LMP_HIST_MUL/2^23` is the same mechanism.
- **Futility history-based margin/gate**: Coda has `hist_adj` in the
  futility formula (`search.rs:4403`). Raphael's `FP_HIST_MUL·hist/HISTORY_MAX`
  is the same axis.
- **NMP TT-upper gate**: Coda has NMP TT flag gates (#864 cut_node gate
  merged). Verify the exact predicate matches — Raphael uses
  `!(ttFlag==UPPER && ttScore<beta)`; if Coda's differs, this is a low-risk
  bench-only alignment.
- **NMP `is_kingpawn` gate**: Coda uses `stm_non_pawn != 0`
  (`search.rs:3830`) — equivalent (Coda skips NMP if only king+pawns, exactly
  Raphael's guard).
- **do_deeper / do_shallower**: both engines have this. Raphael's tiering is
  the same as Coda's, not the Hobbes-style `do_even_deeper` 3rd-tier.
- **Direct-check bonus in quiet ordering**: Coda has `QUIET_CHECK_BONUS`
  (SEE-gated, `movepicker.rs:599-672`). Raphael's is unconditional — one
  simple SPRT could test dropping the SEE gate, but Coda's SEE gate was
  chosen deliberately.

## Sources

- `chess/engines/Raphael/version.txt` — 4.3.0
- `chess/engines/Raphael/README.md` — Elo table, CCRL Blitz #26 → 3740 for 4.2.0
- `chess/engines/Raphael/src/Raphael/Raphael.cpp:474-926` — negamax
- `chess/engines/Raphael/src/Raphael/Raphael.cpp:928-1057` — quiescence
- `chess/engines/Raphael/src/Raphael/Raphael.cpp:401-472` — iterative deepen + aspiration
- `chess/engines/Raphael/src/Raphael/Raphael.cpp:383-398` — adjust_score (halfmove + corrhist)
- `chess/engines/Raphael/src/Raphael/corrhist.cpp:99-158` — 5-source correction update/get
- `chess/engines/Raphael/src/Raphael/corrhist.h:37-39` — pawn/major/nonpawn tables
- `chess/engines/Raphael/src/Raphael/history.cpp:1-138` — quiet/cont/capt history
- `chess/engines/Raphael/src/Raphael/movepick.cpp:1-316` — staged picker
- `chess/engines/Raphael/src/Raphael/see.cpp:1-102` — SEE
- `chess/engines/Raphael/src/Raphael/tunable.h:186-390` — ~180 SPSA parameters
- `chess/engines/Raphael/src/Raphael/transposition.cpp:1-244` — 5-slot TT, XOR-free 16-bit key
- `chess/engines/Raphael/src/Raphael/tm.cpp:1-217` — 3-factor soft time
- `chess/engines/Raphael/src/Raphael/wdl.cpp:1-56` — WDL norm + material polynomial
- `chess/engines/Raphael/src/Raphael/position.h:170-179` — material scaling
- `chess/engines/Raphael/src/Raphael/cuckoo.cpp` — 3668-slot cuckoo table
- `chess/engines/Raphael/src/eval/arch.h:8-30` — NNUE arch constants (60144 threats, 11 PSQ, 1024 accum, 32×32 hidden, 16 in / 8 out buckets)
- `chess/engines/Raphael/src/eval/accumulator.cpp:14-215` — threat indexing + Finny table
- `chess/engines/Raphael/src/eval/nnue_multilayer.cpp:36-276` — SIMD forward pass, sparse L1
- `coda/src/search.rs:206-208, 4528` — Coda cut_node LMR bump
- `coda/src/search.rs:4607-4610` — Coda complexity-adjusted LMR
- `coda/src/search.rs:911-913, 1395-1508` — Coda 5-source correction history
- `coda/experiments.md:1319, 2418, 3297` — asp fail-high H0s
- `coda/experiments.md:1859, 2442` — RFP-complexity H0s
- `coda/experiments.md:317, 1940` — LMP-hist H0s
- `coda/experiments.md:1130` — hindsight extension H1 merge
- `coda/experiments.md:1129` — Stormphrax T6 conthist base-aware H1 merge
- `coda/experiments.md:3575` — Razoring removed −19.8 Elo
- `coda/experiments.md:864` — NMP cut_node gate H1 merge
- `coda/engine-notes/hobbes.md` §E2 — 2-ply cont corrhist already ranked
