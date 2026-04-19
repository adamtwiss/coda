# Clover Chess Engine - Technical Review

Source: `~/chess/engines/CloverEngine/`
Version: 9.1
Language: C++
NNUE: Single-accumulator (768x7 king-bucketed PST -> 2x1280 hidden -> 8 output buckets) with pairwise activation
Rating: 3577 CCRL 40/15 (v6.2, #12); estimated 3650+ for v9.1 based on recent development trajectory

Last reviewed: 2026-04-18

---

## 1. NNUE Architecture

### 1.1 Network Topology: (PST[768x7]) -> 2x1280 -> 8

Clover uses a **single-accumulator HalfKA architecture** with 7 king input buckets and 2 hidden layers of width 1280 each, feeding 8 material-based output buckets.

**Constants** (`net.h:98-122`):
- `KING_BUCKETS = 7` (king bucket count, compact vs finer granularity)
- `INPUT_NEURONS = 768 * 7 = 5376` (total feature inputs)
- `SIDE_NEURONS = 1280` (accumulator width, per perspective)
- `HIDDEN_NEURONS = 2 * SIDE_NEURONS = 2560` (both sides concatenated for output layer)
- `OUTPUT_NEURONS = 8` (material buckets)
- `Q_IN = 255`, `Q_HIDDEN = 64`, `Q_IN_HIDDEN = 255 * 64 = 16320`
- Network scale output: `output * 225 / Q_IN_HIDDEN`

**Coda comparison**: Coda v9 uses 768 FT with 16->32->1x8, Clover uses 5376 inputs (768x7) -> 2x1280 -> 8. Clover's dual 1280-wide layers are **much wider** (vs our 16/32). This greater hidden capacity may allow learning more complex interactions, though with higher training time. Clover lacks the separate threat accumulator that Reckless has.

### 1.2 King Bucket Layout (`net.h:125-142`, `defs.h:133-143`)

**Compact 7-bucket scheme** (`kingIndTable`):
- Buckets indexed by king square mirroring logic
- Back-rank differentiation (ranks 0-1 split into buckets 0-3 with 4 file buckets)
- Mid-board ranks 2-3 use bucket 4-5 (2 file buckets)
- Endgame ranks 4-7 collapse into bucket 6
- Horizontal mirroring at file >= 4

```
Rank 0-1: 4 buckets (buckets 0-3 based on file)
Rank 2-3: 2 buckets (bucket 4-5 based on file pair)
Rank 4-7: 1 bucket  (bucket 6)
```

Net feature index:
```cpp
net_index = 64 * 12 * kingIndTable[kingSq.mirror(side)] 
          + 64 * (piece + side * (piece >= 6 ? -6 : +6))
          + (sq.mirror(side) ^ (7 * ((kingSq >> 2) & 1)))
```

**Coda comparison**: Coda has 16 buckets with all-ranks granularity. Clover's 7-bucket scheme trades accuracy for training/inference speed. The compact representation likely works well because endgames dominate computational time, and opening/middle-game sensitivity to king position varies smoothly.

### 1.3 PST Accumulator (`net.h:160-448`)

**Incremental updates** with king bucket caching:
- `cached_states[color][bucket]` stores accumulator state + piece bitboards for each king bucket
- On king bucket change, only the changed bucket's accumulator is recalculated
- Otherwise, pieces added/removed via SIMD `apply_updates()` (add_ind/sub_ind stacks)

**SIMD loop unrolling** (`apply_updates`, lines 249-279):
- Processes 128-byte chunks (`BUCKET_UNROLL`)
- Unrolled inner loops with `REG_LENGTH` registers (8 for AVX-512, 4 for AVX2)
- Single pass over input weights for all additions, then subtractions

**Forward pass** (`get_output`, lines 450-469):
```cpp
for each perspective (stm, nstm):
    clamped = clamp(accumulator[side], 0, 255)
    output += mullo(clamped, output_weight[side])
    // squared product: clamped_stm^2 + clamped_nstm^2
```
Returns `(bias + acc_product / Q_IN) * 225 / Q_IN_HIDDEN`.

**Coda comparison**: We use similar incremental updates. The key difference is Clover's wider 1280-neuron hidden layers vs our 16/32, which requires more SIMD work. Clover lacks pairwise activation (left*right) that Reckless has.

### 1.4 Output Buckets (Material-based)

8 buckets indexed by piece count (popcount):
```
Popcount  0-8: bucket 0
          9-12: bucket 1
         13-16: bucket 2
         17-19: bucket 3
         20-22: bucket 4
         23-25: bucket 5
         26-28: bucket 6
         29-32: bucket 7
```

**Coda comparison**: Same as ours.

---

## 2. Search

### 2.1 Framework

- Iterative deepening with aspiration windows
- Root node specialization (`rootNode` template parameter)
- Generic PV/cut/all-node types via template parameters
- Killer moves + counter-pawn moves (KP move) stored per position
- Stack includes threat bitboards pre-computed from board

### 2.2 Aspiration Windows (`search.h:169-172`)

Fixed delta-based aspiration:
```cpp
delta = AspirationWindowsValue                           // initial margin (~7 cp)
if fail_low:
    beta = (3*alpha + beta) / 4
    delta += 15
if fail_high:
    beta = score + delta
    delta += 15
```

Simple linear growth on both sides. No eval-dependent scaling like Reckless.

**Coda comparison**: We use similar fixed delta (15). Reckless uses eval-dependent scaling. Clover is simpler but likely sufficient.

### 2.3 Eval Correction (`search.h:414-429`, `history.h:317-429`)

**5-source correction history** indexed by position key hash:
1. Pawn hash correction
2. Material hash (White)
3. Material hash (Black)
4. Continuation correction at ply-2
5. Continuation correction at ply-3 (ply-2 only for non-root)
6. Continuation correction at ply-4

Each source scaled individually, then summed and divided by `16 * 1024 = 16384`:
```cpp
correction = CorrHistPawn * get_corr_hist(turn, pawn_key)
           + CorrHistMat * get_mat_corr_hist(turn, WHITE, white_mat_key)
           + CorrHistMat * get_mat_corr_hist(turn, BLACK, black_mat_key)
           + CorrHistCont2 * get_cont_corr_hist(stack, 2)
           + CorrHistCont3 * get_cont_corr_hist(stack, 3)
           + CorrHistCont4 * get_cont_corr_hist(stack, 4)
corrected_eval = eval + correction / 16384
```

**Coda comparison**: Coda v9 has basic pawn correction. Clover adds material + continuation sources, giving richer eval adjustment. The continuation correction at ply-2/3/4 is worth porting.

### 2.4 Internal Iterative Deepening (`search.h:374-379`)

```cpp
if pvNode && depth >= IIRPvNodeDepth && (!tt_hit || tt_depth + 4 <= depth):
    depth -= IIRPvNodeReduction
if cutNode && depth >= IIRCutNodeDepth && (!tt_hit || tt_depth + 4 <= depth):
    depth -= IIRCutNodeReduction
```

Reduces depth to find a TT move when none available or TT is stale (depth < current - 3).

**Coda comparison**: Standard IIR, we likely have similar.

### 2.5 Eval Difference History Bonus (`search.h:381-388`)

**Hindsight quiet move bonus** based on eval swings:
```cpp
if !null_search && board.captured() == NO_PIECE && previous_move_exists:
    bonus = clamp(-EvalHistCoef * ((stack-1).eval + static_eval), EvalHistMin, EvalHistMax) + EvalHistMargin
    update_main_hist(prev_move, bonus)
```

If the previous move evaluation was good, bonus it retroactively. Uses eval delta to scale.

**Coda comparison**: This is a clever "move was good in hindsight" heuristic we don't have.

### 2.6 Depth Adjustment for Hindsight Reduction (`search.h:390-391`)

```cpp
if previous_R >= 3 && !improving_after_move && !in_check && prev_eval != INF:
    depth += 1 + bad_static_eval
```

If the previous move was heavily reduced AND the current position isn't improving much, restore a ply. This recovers search depth when reductions prove unnecessary.

**Coda comparison**: Similar to our hindsight reduction logic. Nice tactical recovery.

### 2.7 Razoring (`search.h:397-404`)

```cpp
margin = RazoringMargin * depth
if depth <= RazoringDepth && eval + margin <= alpha:
    return quiesce(alpha, alpha+1)
```

Linear depth scaling. Params: `RazoringDepth=2, RazoringMargin=145`.

**Coda comparison**: We use quadratic scaling (`252 * depth^2`). Clover's linear approach is more conservative (shallower reductions at deep depths).

### 2.8 Static Null Move Pruning (`search.h:406-416`)

**Complexity-aware SNMP**:
```cpp
snmp_margin = (SNMPMargin - SNMPImproving * improving) * depth 
            - SNMPImprovingAfterMove * improving_after_move
            - SNMPCutNode * is_cutnode
            + complexity * SNMPComplexityCoef / 1024
            + SNMPBase
if eval - snmp_margin > beta:
    return (eval + beta) / 2
```

Params: `SNMPMargin=89, SNMPImproving=18, SNMPComplexityCoef=608`.

**Coda comparison**: We use simpler SNMP. Clover's complexity scaling is interesting — complex positions get wider pruning margins to avoid over-pruning tactics.

### 2.9 Null Move Pruning (`search.h:418-438`)

Conditions: not null_search, enemy_has_no_threats, non-pawn material, eval >= beta, good static eval.

Reduction: `R = NMPReduction + depth/NMPDepthDiv + (eval-beta)/NMPEvalDiv + improving + is_tt_move_noisy`

On fail-low by margin `NMPHindsightMargin`, restore a ply.

Params: `NMPReduction=4, NMPDepthDiv=3, NMPEvalDiv=133, NMPHindsightMargin=114`.

**Coda comparison**: Standard NMP with eval-dependent reduction. No novel twists.

### 2.10 ProbCut (`search.h:440-478`)

```cpp
probcut_beta = beta + ProbcutMargin - ProbcutImprovingMargin * improving
if depth >= ProbcutDepth && cutNode && tt_depth < depth - 3:
    search qsearch for good noisy moves
    if score >= probcut_beta:
        return score (or re-search at reduced depth if margin is small)
```

Params: `ProbcutMargin=174, ProbcutImprovingMargin=41, ProbcutDepth=4, ProbcutReduction=5`.

**Coda comparison**: Standard ProbCut, no major innovations.

### 2.11 Singular Extensions (`search.h:567-598`)

```cpp
if depth >= SEDepth && tt_depth >= depth - 3 && tt_bound == Lower:
    singular_beta = tt_value - (SEMargin + SEWasPVMargin * !pvNode) * depth / 64
    score = search(..., singular_beta - 1, singular_beta, (depth-1)/2)
    
    if score < singular_beta:
        ex = 1 + double_extend + triple_extend
    else if singular_beta >= beta:
        return singular_beta  // multicut
    else if tt_value >= beta:
        ex = -1 - !pvNode
    else if cutNode:
        ex = -2
```

Extensions: up to +3 for very singular moves, -1/-2 for non-singular.

Params: `SEDepth=5, SEMargin=41, SEWasPVMargin=37, SEDoubleExtensionsMargin=9, SETripleExtensionsMargin=52`.

**Coda comparison**: We disabled SE for causing tactical slowdown. Clover keeps them. The asymmetric extension margins (much tighter for non-PV) are reasonable.

### 2.12 Improvements Detection

```cpp
improving = (eval >= beta) ? 1 
          : (eval_diff > 0) ? 1 
          : (eval_diff < NegativeImprovingMargin) ? -1 
          : 0
improving_after_move = (static_eval + (stack-1).eval > 0)
```

3-state improving flag + binary improving_after_move flag.

### 2.13 Futility Pruning (`search.h:520-523`)

```cpp
new_depth = estimated_depth_after_LMR
if new_depth <= FPDepth && static_eval + FPBias + FPMargin * new_depth <= alpha:
    skip_quiets()
```

Params: `FPDepth=10, FPBias=110, FPMargin=80`.

**Coda comparison**: Similar depth-scaled futility. We use different constants.

### 2.14 Late Move Pruning (`search.h:525-527`)

```cpp
if new_depth <= LMPDepth && played >= (LMPBias + new_depth * new_depth) / (2 - improving):
    skip_quiets()
```

Improvement-adaptive quadratic threshold. Params: `LMPDepth=7, LMPBias=1`.

**Coda comparison**: Similar to ours.

### 2.15 History Pruning (`search.h:529-535`)

```cpp
if depth <= HistoryPruningDepth && bad_static_eval && history < -HistoryPruningMargin * depth:
    skip_quiets()
    continue
```

Prunes quiet moves with terrible history when eval is below alpha and depth is shallow.

Params: `HistoryPruningDepth=2, HistoryPruningMargin=4544`.

**Coda comparison**: We have similar history pruning.

### 2.16 Noisy Move Futility (`search.h:554-562`)

```cpp
noisy_futility = FPNoisyBias + seeVal[captured] + FPNoisyMargin * depth + history / FPNoisyHistoryDiv
if depth <= FPNoisyDepth && static_eval + noisy_futility <= alpha:
    skip
```

Includes captured piece value in the margin.

Params: `FPNoisyDepth=8, FPNoisyBias=98, FPNoisyMargin=88, FPNoisyHistoryDiv=31`.

**Coda comparison**: Similar.

### 2.17 SEE Pruning (`search.h:537-552`)

Quiet: `-SEEPruningQuietMargin * new_depth - history / SEEQuietHistDiv`
Noisy: `-SEEPruningNoisyMargin * depth^2 - history / SEENoisyHistDiv`

History-aware thresholds. Params: `SEEPruningQuietDepth=8, SEEPruningQuietMargin=70, SEEPruningNoisyDepth=8, SEEPruningNoisyMargin=14, SEEQuietHistDiv=123, SEENoisyHistDiv=265`.

**Coda comparison**: Quadratic noisy margin is interesting.

### 2.18 Late Move Reductions (`search.h:627-671`)

**Logarithmic base formula** (`defs.h:274-280`):
```cpp
R = LMRGrain * (LMRQuietBias + log(depth) * log(played) / LMRQuietDiv)
```

Params: `LMRGrain=1024, LMRQuietBias=1.43, LMRQuietDiv=2.53`.

**Adjustments** (all in mille-plies, then divided by `LMRGrain=1024`):
- History: `-LMRGrain * history / (is_quiet ? HistReductionDiv : CapHistReductionDiv)` (history reduces)
- Check: `-LMRGivesCheck * (move gives check)` (check reduces)
- Was PV: `+LMRWasNotPV * !was_pv` (non-pv increases)
- Good eval: `+LMRGoodEval * (quiet && eval > beta + margin)` (increases)
- Eval trend: `+LMREvalDifference * (quiet && eval_trend > margin && parity)` (increases)
- Refutation: `-LMRRefutation * (killer move)` (refutation reduces)
- Bad capture: `+LMRBadNoisy * (bad noisy move)` (increases)
- Improving: `+LMRImprovingM1 * (improving == -1)`, `+LMRImproving0` (increases when not improving)
- PV node: `-LMRPVNode * pvNode` (PV reduces)
- Cut node: `+LMRCutNode * cutNode` (cut increases aggressively)
- Ex-PV: `-LMRWasPVHighDepth * (was_pv && tt_depth >= depth)` (ex-PV reduces)
- TT noisy: `+LMRTTMoveNoisy * is_tt_move_noisy` (increases)
- Bad static: `+LMRBadStaticEval * (eval <= alpha)` (increases)
- Complexity: `-LMRGrain * complexity / LMRCorrectionDivisor` (reduces)
- Fail-low: `+LMRFailLowPV + LMRFailLowPVHighDepth * (tt_depth > depth)` (increases if fail-low PV)
- High cutoffs: `+LMRCutoffCount * (stack->cutoff_cnt > 3)` (increases after cutoffs)

**Post-LMR re-search**:
```cpp
if R > 1 && score > alpha:
    new_depth += (score > best + DeeperMargin)
    new_depth -= (score < best + new_depth)
    re_search(new_depth - 1)
```

Extends if much better, reduces if barely better.

Params: `HistReductionDiv=8488, CapHistReductionDiv=4065, LMRGivesCheck=1166, LMRWasNotPV=1134, LMRGoodEval=1064, LMREvalDifference=1065, LMRRefutation=2200, LMRBadNoisy=1138, LMRImprovingM1=2075, LMRImproving0=1151, LMRPVNode=843, LMRCutNode=2321, LMRWasPVHighDepth=971, LMRTTMoveNoisy=1165, LMRBadStaticEval=903, LMRFailLowPV=861, LMRFailLowPVHighDepth=1096, LMRCutoffCount=1000, EvalDifferenceReductionMargin=208, LMRCorrectionDivisor=53, DeeperMargin=77, PVDeeperMargin=75`.

**Coda comparison**: Clover has a richer set of LMR modifiers than typical engines. The logarithmic formula is cleaner than piecewise adjustments. Notable: separate good_eval vs bad_static conditions, high complexity adjustment, and post-LMR hindsight.

### 2.19 Quiescence Search (`search.h:41-134`)

- TT cutoff (non-PV nodes only)
- Stand-pat blending: `(stand_pat + beta) / 2` to dampen inflated returns
- Delta pruning: `best + DeltaPruningMargin < alpha`
- QS futility: `futility_base + captured_value <= alpha`
- QS LMP: stop after 4 moves if not giving check
- QS SEE: `alpha - eval / 8 - 100`
- Exit blending: `(best + beta) / 2` when best >= beta

Params: `DeltaPruningMargin=1038, QuiesceFutilityBias=192, QuiesceSEEMargin=-12`.

**Coda comparison**: Standard QS. The blending tactics (stand-pat, exit) dampen search instability.

### 2.20 History Updates (`search.h:731-801`)

**Gravity formula**:
```cpp
entry += bonus - abs(bonus) * entry / MAX_VALUE
```

**Bonus/Malus per move type** (all depth-scaled):
```cpp
MainHistory bonus: min(margin * depth - bias, max)  // params: margin=271, bias=329, max=2612
MainHistory malus: -min(margin * depth - bias, max) // params: margin=349, bias=292, max=2491
ContinuationHistory bonus: min(273*d - 343, 2468)
CaptureHistory bonus: min(250*d - 340, 2677)
PawnHistory bonus: min(277*d - 331, 2732)
```

**Tried count scaling**: Each bonus/malus multiplied by `tried_count` (number of moves tried before cutoff). This de-weights early-cutoff moves.

**Correction history update** (`search.h:811`):
```cpp
bonus = clamp(best_score - static_eval) * depth / 8, -256, 256)
update_pawn_corr, update_mat_corr, update_cont_corr
```

**Coda comparison**: Multi-faceted history with tried_count scaling is sophisticated. We have basic history + tried_count in some places.

---

## 3. Move Ordering (`movepick.h`, `history.h`)

### 3.1 Move Picker Stages

1. **TT move**: Transposition table move
2. **Good noisy**: Captures with favorable SEE (`16 * captured_value + noisy_history >= threshold`)
3. **Killer**: Killer move (quiet)
4. **Quiet moves**: All remaining quiet moves, scored by history
5. **Bad noisy**: Failed-SEE captures
6. **QS/ProbCut stages**: Separate QS and ProbCut move pickers

**Bad noisy SEE threshold**: `-score / 46 + 109` (history-dependent, becomes more lenient for high-history moves).

**Coda comparison**: Standard architecture. Clover includes killer moves; Reckless dropped them. Both reasonable depending on history richness.

### 3.2 Quiet Move Scoring (`history.h:367-388`, `movepick.h:134-280`)

```cpp
score = QuietSearchHistCoef * main_history[from][to]
      + QuietSearchContHist1 * continuation_history[ply-1][piece][to]
      + QuietSearchContHist2 * continuation_history[ply-2][piece][to]
      + QuietSearchContHist4 * continuation_history[ply-4][piece][to]
```

In move picker:
```cpp
score = QuietMPHistCoef * main_history
      + QuietMPContHist1 * cont_hist[ply-1]
      + QuietMPContHist2 * cont_hist[ply-2]
      + QuietMPContHist4 * cont_hist[ply-4]
      + QuietMPPawnHist * pawn_history
```

Params (move picker weights): `QuietMPHistCoef=1069, QuietMPContHist1=1921, QuietMPContHist2=1924, QuietMPContHist4=1090, QuietMPPawnHist=1070`.

**Threat-aware history** (`history.h:273-283`):
```cpp
main_history[turn][!from_threatened][!to_threatened][from_to]
```

Each main history entry indexed by threat state of from/to squares. Separate values per threat context.

**Coda comparison**: Threat-aware history buckets are clever. Clover doesn't use as rich threat detection as Reckless, but still leverages threat context.

### 3.3 Noisy Move Scoring (`history.h:296-305`)

```cpp
capture_history[!to_threatened][piece][to][captured_type]
```

Threat-aware capture history. Also scored in move picker:
```cpp
score = 16 * seeVal[captured] + noisy_history
```

### 3.4 Pawn History (`history.h:307-315`)

```cpp
pawn_history[pawn_key & MASK][piece][to]
```

Indexed by pawn structure hash, piece type, and destination square. Used as a tiebreaker in quiet scoring.

**Coda comparison**: Pawn history is a nice tiebreaker when other histories are equal.

---

## 4. Time Management (`search.h:988-1002`)

**Node-based time allocation with score stability tracking**:

Depth check: only adjust time if `id_depth >= TimeManagerMinDepth` (9).

**Score stability factor**:
```cpp
score_scale = clamp(TimeManagerScoreBias + (last_root_score - current_root_score) / TimeManagerScoreDiv,
                     TimeManagerScoreMin, TimeManagerScoreMax)
// Params: Bias=1.058, Min=0.47, Max=1.612, ScoreDiv=116
```

Drops in score extend search time; improvements contract it.

**Best move nodes percentage**:
```cpp
nodesSearchedPercentage = best_move_nodes / total_nodes
best_move_scale = TimeManagerNodesSearchedMaxPercentage - TimeManagerNodesSearchedCoef * nodesSearchedPercentage
// Params: Max=1.858, Coef=1.064
```

If best move uses many nodes, end early (close to solution).

**Best move stability**:
```cpp
best_move_scale = TimeManagerBestMoveMax - TimeManagerbestMoveStep * min(10, best_move_changes)
// Params: Max=1.229, Step=0.075
```

Penalize for best-move changes; reward stability.

**Combined time scale** (all factors multiplied).

**Coda comparison**: Clover uses a multi-factor soft stop with eval, node distribution, and move stability. We use simpler fixed percentages. This adaptive approach likely wins in positions where uncertainty is high.

---

## 5. Miscellaneous / Unique Features

### 5.1 Threat Pre-computation (`board.h`, `search.h:372`)

Board state pre-computes and increments threat bitboards per piece type:
```cpp
threats_pieces[PieceTypes::PAWN], ..., threats.all_threats
```

Updated incrementally during makemove. Used in:
- History indexing (threat-aware buckets)
- Move scoring (escape/check bonuses)
- SEE thresholds

**Coda comparison**: We compute threats on-the-fly. Clover's pre-computed approach is faster for move ordering.

### 5.2 Cuckoo Hash for Repetition (`cuckoo.h`, `search.h:49`)

Uses cuckoo hashing for fast upcoming repetition detection (for draw avoidance). No GHI mitigation visible in the code, unlike Reckless.

**Coda comparison**: We use similar repetition detection.

### 5.3 Killer Move + Counter-Pawn Move (`search.h:482`, `history.h:234`)

Maintains killer moves per ply and counter-pawn moves per position (indexed by `king_pawn_key & KP_MOVE_MASK`). Good simple heuristic for quiet move ordering.

**Coda comparison**: We have similar structures.

### 5.4 Cutoff Count Tracking (`history.h:237`, `search.h:732`)

Tracks cutoff count at each ply level (`stack->cutoff_cnt`). Used in LMR to increase reductions after many cutoffs (position is well-explored).

**Coda comparison**: Interesting tactic to reduce thrashing in already-explored subtrees.

### 5.5 Complexity Measurement (`search.h:323`)

```cpp
complexity = board.complexity()  // likely material/pawn structure measure
```

Used in SNMP margins and LMR. Tight complex positions get more latitude.

**Coda comparison**: We have basic complexity tracking; Clover integrates it more deeply.

### 5.6 Enemy Threats Detection (`search.h:414-415`)

```cpp
bool enemy_has_no_threats = (board.checkers() == 0)  // pre-computed during board updates
```

Used extensively in pruning conditions. Avoids pruning tactical positions.

**Coda comparison**: Standard, we likely do the same.

### 5.7 Singular Search (`search.h:577-579`, `search-info.h`)

Stack exclusion mechanism allows singular extension search to exclude the TT move and try alternatives. Clean implementation using `stack->excluded`.

**Coda comparison**: Standard SE mechanism.

---

## 6. Tuning Parameters

Clover is heavily tuned via SPSA with OpenBench. Notable tunables (all in `params.h`):

**Search pruning**: 18 core pruning params (RFP, NMP, ProbCut, FP, LMP, etc.)
**LMR**: 15 LMR-specific params (was_pv, improving states, eval conditions, cutoffs)
**History**: 12 history growth params (bonus/malus/max per table)
**Move ordering**: 4 threat-aware move scoring bonuses
**Time management**: 8 time allocation params (score, nodes, move stability)
**NNUE evaluation**: 2 params for eval scaling

Total: ~60 SPSA-tuned parameters. Indicates careful engineering to find sweet spots.

**Coda comparison**: Coda v9 has ~50 tunables in `src/search.rs:45-160`. Clover's tuning is similarly comprehensive. Their numbers (e.g., `LMRQuietBias=1.43`) suggest deep VLTC (40s+0.4s) tuning.

---

## 7. Summary: Ideas to Port

### High-impact, moderate effort:

1. **Multi-source correction history** (pawn + material + continuation):
   - Current: Coda has basic pawn correction. Adding material + continuation sources would improve eval accuracy, especially in material-imbalanced positions.
   - Effort: Allocate 3-4 additional history tables, update formulas.
   - Expected gain: ~5-10 Elo

2. **Threat-aware history buckets**:
   - Clover indexes main history by threat state: `hist[from_threatened][to_threatened][from_to]`. This lets the same move be valued differently depending on whether it's defended or escapes threat.
   - Current: Coda uses single history per move.
   - Effort: Double main history dimensions, update all accesses.
   - Expected gain: ~5-8 Elo

3. **Eval-difference history bonus** (`search.h:381-388`):
   - Bonus the previous move retroactively if the current eval is good. Simple hindsight heuristic.
   - Current: We update move histories after cutoff; this bonds previous move to current eval swing.
   - Effort: Add one condition in non-null search path.
   - Expected gain: ~2-5 Elo

4. **Post-LMR hindsight re-search** (`search.h:668-670`):
   - If LMR reduction R > 1 and score > alpha, adjust new_depth based on whether score improved significantly, then re-search.
   - Current: We have basic LMR feedback; Clover's adaptive depth adjustment is more aggressive.
   - Effort: Add 2-3 lines in LMR block.
   - Expected gain: ~2-4 Elo

### Moderate-impact, higher effort:

5. **Logarithmic LMR formula**:
   - Clover uses `log(depth) * log(played) / LMRQuietDiv`, which is smoother than piecewise reduction tables and may generalize better across engine changes.
   - Current: We use LMR tables.
   - Effort: Replace table with formula, retune LMRQuietBias and LMRQuietDiv.
   - Expected gain: ~3-6 Elo (if tuned well)

6. **Threat-aware SEE thresholds**:
   - Clover scales SEE pruning thresholds by piece threat status.
   - Current: Fixed SEE margins in Coda.
   - Effort: Add threat bit to SEE threshold calculation.
   - Expected gain: ~1-3 Elo

7. **Adaptive time management** (multi-factor):
   - Clover uses eval stability, node distribution, and move stability to modulate time. More sophisticated than our simple time percentage.
   - Effort: Requires tracking score/move history and tuning 4-6 time params.
   - Expected gain: ~3-5 Elo in diverse positions (less in fixed time, more in increment)

### Lower priority:

8. **Complexity-aware pruning margins**:
   - SNMP and LMR margins scale with board complexity. Helps avoid over-pruning in tactical middlegames.
   - Effort: Integrate complexity measure into pruning decisions.
   - Expected gain: ~1-3 Elo

9. **Cutoff count LMR bonus** (`search.h:656`):
   - Increase reduction after multiple cutoffs at same ply (well-explored subtree).
   - Effort: Track and use `stack->cutoff_cnt`.
   - Expected gain: ~0.5-2 Elo

10. **Singular extension asymmetry** (non-PV strictness):
    - Clover applies stricter thresholds for non-PV SE, reducing extensions in side variations.
    - Effort: Conditional SE margin in `!pvNode` branches.
    - Expected gain: ~1-2 Elo (or negative if we're correct to avoid SE)

---

## 8. Net-Level Observations

**Architecture simplicity**: Clover's single-accumulator 768x7 -> 1280x2 -> 8 is more tractable than Reckless's dual-accumulator threat network. The wider hidden layers (1280 vs 16/32) may compensate for lack of threat features.

**Training data**: Clover likely trained on Bullet with similar 40s+0.4s or faster time controls. The 7-bucket king scheme suggests either:
- Faster convergence in training (fewer parameters to tune)
- Empirical finding that back-rank + mid-board + endgame distinctions suffice

**Tuning discipline**: ~60 SPSA parameters tuned at VLTC is very thorough. Clover's strength (3650+ estimated) likely comes from meticulous parameter optimization, not architectural novelty.

**Search philosophy**: Clover uses classical alpha-beta + pruning heavily, with strong move ordering (threat-aware history) and eval correction. No radical ideas like separate threat accumulators or sparse matmul.

---

## Coda Porting Priority

**Tier 1 (highest bang-for-buck)**:
- Multi-source correction history
- Threat-aware main history buckets
- Eval-difference history bonus

**Tier 2 (solid incremental gains)**:
- Logarithmic LMR formula (with retuning)
- Post-LMR hindsight adjustment
- Adaptive time management

**Tier 3 (nice-to-have)**:
- Threat-aware SEE thresholds
- Complexity-aware margins
- Cutoff count tracking

Estimated cumulative from Tier 1: +8-20 Elo (depending on tuning rigor).
Estimated cumulative from Tiers 1-2: +15-35 Elo.

