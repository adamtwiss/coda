# Stockfish Technical Review

Source: `~/chess/engines/Stockfish/src/` (current dev branch, 2026)

---

## 1. NNUE Architecture

### Dual-Net System
Stockfish uses **two networks** simultaneously, selected per-evaluation:
- **Big net**: `nn-9a0cc2a62c52.nnue` -- FT=1024, L2=31, L3=32
- **Small net**: `nn-47fc8b7fff06.nnue` -- FT=128, L2=15, L3=32

Selection rule (`evaluate.cpp:49`): `use_smallnet = abs(simple_eval) > 962`
- Simple eval = material imbalance (pawn counts + non-pawn material)
- Re-evaluates with big net if smallnet result is close: `abs(nnue) < 277`

**Coda comparison**: We use a single net. The dual-net approach saves NPS on lopsided positions where accuracy matters less. The small net (128-wide FT) is extremely fast.

### Architecture Per Network
```
(HalfKAv2_hm features --> FT) x2 perspectives
    + FullThreats features (big net only, int8 weights)
    --> SCReLU + CReLU in parallel (dual activation!)
    --> fc_1: (FC_0_OUTPUTS * 2) --> L3
    --> CReLU
    --> fc_2: L3 --> 1
    + skip connection from FT (fc_0 output[FC_0_OUTPUTS])
```

Key detail from `nnue_architecture.h:60-141`:
- **fc_0**: AffineTransformSparseInput (exploits CReLU sparsity)
- **ac_sqr_0**: SqrClippedReLU (SCReLU) on fc_0 output
- **ac_0**: ClippedReLU on same fc_0 output
- These are **concatenated**: `[SCReLU(x) | CReLU(x)]` giving `FC_0_OUTPUTS * 2` inputs to fc_1
- **Skip connection**: `fc_0_out[FC_0_OUTPUTS]` (extra output neuron) is scaled and added to final output

Big net: `1024 --> 2x(31+1) --> [SCReLU(31)|CReLU(31)] = 62 --> 32 --> 1 + skip`
Small net: `128 --> 2x(15+1) --> [SCReLU(15)|CReLU(15)] = 30 --> 32 --> 1 + skip`

**Coda comparison**: We use pure SCReLU or CReLU, not both simultaneously. The dual-activation concatenation is novel -- it preserves both squared (nonlinear) and linear features for the hidden layers. Our v7 architecture (1024 --> 16 --> 32 --> 1) is simpler but doesn't have the dual activation.

### Input Features

**HalfKAv2_hm** (PSQ features):
- Standard HalfKA with horizontal mirroring (king always on e-h files)
- 32 king buckets (4 files x 8 ranks, mirrored)
- `Dimensions = 64 * 11 * 64 / 2 = 22528` per bucket (10 piece types + king, 64 squares, halved by mirroring)
- MaxActiveDimensions = 32

**FullThreats** (big net only):
- Threat-based features: encodes (attacker_piece, from_square, to_square, attacked_piece)
- `Dimensions = 60144`
- Uses int8 weights (`ThreatWeightType = int8_t`)
- MaxActiveDimensions = 128

**Coda comparison**: We use standard HalfKA. The FullThreats feature set is unique to Stockfish and adds threat awareness at the feature level. This is fundamentally different from our approach of relying on the network to learn threats implicitly.

### Quantization
- FT weights: `int16_t` (WeightType)
- FT biases: `int16_t` (BiasType)
- Threat weights: `int8_t` (ThreatWeightType)
- Accumulator values: `int16_t`
- PSQT weights: `int32_t`
- Post-FT layers: `int8_t` output from CReLU/SCReLU
- `OutputScale = 16`, `WeightScaleBits = 6`
- SCReLU shift: `>> (2 * WeightScaleBits + 7)` = shift right 19

### Layer Stacks
`LayerStacks = 8`, `PSQTBuckets = 8` -- the post-FT network is replicated 8 times, selected by material bucket. This means 8 separate sets of fc_0/fc_1/fc_2 weights.

**Coda comparison**: We don't use layer stacks. This is a significant architectural difference -- SF effectively has 8 specialist sub-networks for different material configurations.

### Sparse Input Optimization
`affine_transform_sparse_input.h`: The first hidden layer (fc_0) uses a **sparse matmul** that:
1. Finds non-zero blocks in the CReLU'd accumulator output using `find_nnz()`
2. Only multiplies weight columns corresponding to non-zero input blocks
3. Uses lookup tables (`OffsetIndices`) for efficient index extraction from bitmasks
4. AVX512ICL, AVX2/SSSE3, and NEON paths

This exploits the fact that CReLU clips to [0,127] as uint8, so many values are zero after clipping. The sparse matmul skips those zero blocks entirely.

**Coda comparison**: We don't exploit accumulator sparsity for the hidden layer matmul. This is a major NPS optimization we're missing.

### Accumulator Updates (Finny Tables)
`nnue_accumulator.h`: Per-perspective cache indexed by `[Square][Color]` (king square x perspective).
- Each cache entry stores: accumulation values, PSQT accumulation, piece array, pieceBB
- On king bucket change, diffs cached vs current piece positions
- Forward and backward incremental update paths
- Separate accumulator stacks for PSQ features and Threat features

**Coda comparison**: We have Finny tables with similar design. SF's is more complex due to dual feature sets (PSQ + Threats) requiring separate accumulator stacks.

### Weight Compression
Uses **LEB128** compression for network file storage (`nnue_common.h:175-289`). Signed integers are compressed with variable-length encoding, significantly reducing file size for sparse/small weights.

### Eval Blending
`evaluate.cpp:53-90`:
```cpp
Value nnue = (125 * psqt + 131 * positional) / 128;
int nnueComplexity = abs(psqt - positional);
optimism += optimism * nnueComplexity / 476;
nnue -= nnue * nnueComplexity / 18236;
int material = 534 * count<PAWN>() + non_pawn_material();
int v = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;
v -= v * rule50_count() / 199;
```
- PSQT and positional components blended with slight positional bias (131 vs 125)
- "Complexity" = divergence between PSQT and positional -- reduces eval when complex
- Material-weighted blend of NNUE and optimism
- Rule50 linear damping (divide by 199)

**Coda comparison**: We don't separate PSQT from positional, and we don't have the complexity-based damping or optimism blending.

---

## 2. Search

### Reduction Table
`search.cpp:622-623`:
```cpp
reductions[i] = int(2763 / 128.0 * std::log(i));  // = 21.59 * ln(i)
```

LMR formula (`search.cpp:1756-1758`):
```cpp
Depth reduction(bool improving, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale - delta * 585 / rootDelta + !improving * reductionScale * 206 / 512 + 1133;
}
```
Note: reductions are in **1024ths** (millireductions). Divide by 1024 to get actual depth reduction.

- Base: `21.59*ln(d) * 21.59*ln(mn)` (product of two log terms)
- Delta adjustment: `- delta * 585 / rootDelta` (narrows reductions near PV)
- Non-improving bonus: `+ !improving * reductionScale * 206/512` (~40% increase when not improving)
- Constant offset: `+ 1133`

**Coda comparison**: Our formula is `C * ln(d) * ln(mn)` with C=1.30. SF's is more complex with the delta and improving terms baked into the base formula rather than as post-adjustments.

### Aspiration Windows
`search.cpp:354-419`:
- Initial delta: `5 + threadIdx % 8 + abs(meanSquaredScore) / 10208`
  - Thread-dependent starting width (diversifies search)
  - Score variance adjusts window size
- Center: averageScore (running average across iterations)
- On fail low: widen downward, `delta += delta / 3`
- On fail high: widen upward with anchoring (`alpha = max(beta - delta, alpha)`), `failedHighCnt++`
- Search depth reduced on fail high: `adjustedDepth = rootDepth - failedHighCnt - 3*(searchAgainCounter+1)/4`

**Coda comparison**: Our delta=15 is fixed. SF's variance-adjusted and thread-diversified approach is more sophisticated.

### Optimism
`search.cpp:360-361`:
```cpp
optimism[us]  = 144 * avg / (abs(avg) + 91);
optimism[~us] = -optimism[us];
```
A sigmoid-like function of the average score, added to evaluations via the evaluate() blend. Creates asymmetric search behavior based on who's winning.

### TT Cutoff (Step 4)
`search.cpp:780-820`:
- Standard depth/bound check with twist: `cutNode == (ttData.value >= beta) || depth > 5`
  - Only allows cutoffs when node type agrees with TT bound, or depth is high enough
- **Graph history interaction fix**: For rule50 < 96, does **double-TT verification** at depth >= 7:
  - Makes the TT move, probes TT again at the resulting position
  - Only takes the cutoff if both TT entries agree on whether to cut
  - This prevents stale TT entries from causing GHI-related blunders

**Coda comparison**: We don't have the double-TT verification. This is a significant correctness feature.

### History update on TT cutoff
`search.cpp:786-795`: When TT value fails high with a quiet TT move:
- Bonus for TT move: `min(119 * depth - 74, 855)`
- Penalty for early quiet moves of previous ply: `-2014` if `moveCount < 4` and no prior capture

### Hindsight Depth Adjustments
`search.cpp:774-777`:
```cpp
if (priorReduction >= 3 && !opponentWorsening)
    depth++;
if (priorReduction >= 2 && depth >= 2 && ss->staticEval + (ss-1)->staticEval > 195)
    depth--;
```
- Increase depth if prior move was heavily reduced and opponent isn't worsening
- Decrease depth if both evals are high (quiet position)

**Coda comparison**: We have hindsight reduction (threshold=200, similar to SF's 195). We should consider the asymmetric condition (!opponentWorsening for increase).

### Razoring (Step 7)
`search.cpp:893`:
```cpp
if (!PvNode && eval < alpha - 502 - 306 * depth * depth)
    return qsearch<NonPV>(pos, ss, alpha, beta);
```
Quadratic in depth (d^2 scaling).

**Coda comparison**: Our razoring is `400 + d*100` (linear). SF's quadratic scaling is more aggressive at higher depths.

### Futility Pruning (Step 8)
`search.cpp:898-909`:
```cpp
auto futility_margin = [&](Depth d) {
    Value futilityMult = 76 - 21 * !ss->ttHit;
    return futilityMult * d
         - (2686 * improving + 362 * opponentWorsening) * futilityMult / 1024
         + abs(correctionValue) / 180600;
};
if (!ss->ttPv && depth < 15 && eval - futility_margin(depth) >= beta && eval >= beta
    && (!ttData.move || ttCapture) && !is_loss(beta) && !is_win(eval))
    return (2 * beta + eval) / 3;  // beta blend!
```
- Margin depends on ttHit (76 vs 55 per depth)
- Correction history adjusts margin (wider when correction is large)
- Returns `(2*beta + eval) / 3` not just beta (blended return)
- Only when no quiet TT move exists

**Coda comparison**: Our futility is `60 + d*60`. SF's is more nuanced with ttHit, correction, improving, and returns a blended score.

### Null Move Pruning (Step 9)
`search.cpp:913-945`:
```cpp
if (cutNode && ss->staticEval >= beta - 16 * depth - 53 * improving + 378 && !excludedMove
    && pos.non_pawn_material(us) && ss->ply >= nmpMinPly && !is_loss(beta))
{
    Depth R = 7 + depth / 3;
    // ... null move search ...
    if (nullValue >= beta && !is_win(nullValue))
    {
        if (nmpMinPly || depth < 16)
            return nullValue;
        // Verification search at high depths
        nmpMinPly = ss->ply + 3 * (depth - R) / 4;
        Value v = search<NonPV>(pos, ss, beta-1, beta, depth-R, false);
        nmpMinPly = 0;
        if (v >= beta)
            return nullValue;
    }
}
```
- Only on cutNodes (major restriction vs typical NMP)
- Margin: `beta - 16*depth - 53*improving + 378`
- Reduction: `R = 7 + depth/3` (very aggressive, ~10 at depth 9)
- Verification search at depth >= 16 with nmpMinPly guard against recursion

**Coda comparison**: Our NMP allows on all non-PV nodes. Restricting to cutNodes is interesting -- it means NMP is skipped at all-nodes where null move is less likely to fail high anyway.

### IIR (Step 10)
`search.cpp:952`:
```cpp
if (!ss->followPV && !allNode && depth >= 6 && !ttData.move && priorReduction <= 3)
    depth--;
```
- PV and cut nodes only (not all-nodes)
- Only when prior reduction was small (<= 3)
- followPV check prevents IIR on PV continuation

### ProbCut (Step 11)
`search.cpp:958-1001`:
```cpp
probCutBeta = beta + 224 - 61 * improving;
// depth >= 3, value from TT not below probCutBeta
// For each capture with SEE >= probCutBeta - staticEval:
//   1. qsearch verification
//   2. If qsearch holds AND probCutDepth > 0: full reduced search
// probCutDepth = depth - 4
// Return: value - (probCutBeta - beta)  // discounted!
```

Plus a **small ProbCut** at `search.cpp:1006-1009`:
```cpp
probCutBeta = beta + 416;
if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && ttData.value >= probCutBeta)
    return probCutBeta;
```
TT-only ProbCut that doesn't require a search.

### LMR Adjustments (Steps 14-17)

Base reduction `r` (in 1024ths) is modified by many factors:

**Increases to r (more reduction):**
- `+1013` if ttPv (`search.cpp:1067`)
- `+691` base offset (`search.cpp:1215`)
- `+3611 + 985 * !ttData.move` if cutNode (`search.cpp:1221`)
- `+1054` if ttCapture (`search.cpp:1225`)
- `+251 + 1124*(cutoffCnt>2) + 1042*allNode` if (ss+1)->cutoffCnt > 1 (`search.cpp:1229`)
- `+r * 273 / (256*depth + 260)` if allNode (`search.cpp:1248`)
- `- ss->statScore * 428 / 4096` (negative statScore increases r) (`search.cpp:1244`)
- `- abs(correctionValue) / 25600` (large correction reduces r) (`search.cpp:1217`)

**Decreases to r (less reduction):**
- `-2819 + PvNode*973 + (ttValue>alpha)*905 + (ttDepth>=depth)*(935+cutNode*959)` if ttPv (`search.cpp:1212`)
- `-moveCount * 65` (`search.cpp:1216`)
- `-2239` for TT move (`search.cpp:1233`)

**LMR depth calculation** (`search.cpp:1258`):
```cpp
Depth d = max(1, min(newDepth - r/1024, newDepth + 2)) + PvNode;
```
Allows **negative reduction** (extension up to +2 beyond newDepth), plus +1 for PV nodes.

**Re-search logic** (`search.cpp:1266-1280`):
```cpp
if (value > alpha) {
    const bool doDeeperSearch    = d < newDepth && value > bestValue + 48;
    const bool doShallowerSearch = value < bestValue + 9;
    newDepth += doDeeperSearch - doShallowerSearch;
    if (newDepth > d)
        value = -search<NonPV>(pos, ss+1, -(alpha+1), -alpha, newDepth, !cutNode);
    update_continuation_histories(ss, movedPiece, move.to_sq(), 1426);  // post-LMR bonus
}
```

**Coda comparison**: SF's LMR is vastly more sophisticated than ours. Key differences:
- They have ~15 adjustment factors vs our ~8
- The ttPv/cutNode/allNode interactions are very refined
- The post-LMR deeper/shallower re-search is elegant
- The cutoffCnt feedback from child nodes is unique

### LMP (Late Move Pruning)
`search.cpp:1074`:
```cpp
if (moveCount >= (3 + depth * depth) / (2 - improving))
    mp.skip_quiet_moves();
```
Uses `skip_quiet_moves()` rather than hard break -- still searches captures.

**Coda comparison**: Our formula is `3 + d^2` (no improving divisor). SF's `(3+d^2)/(2-improving)` effectively halves the threshold when improving.

### Capture Pruning
`search.cpp:1080-1100`:
- **Capture futility** (lmrDepth < 7): `staticEval + 218 + 223*lmrDepth + PieceValue[captured] + 131*captHist/1024 <= alpha`
- **SEE pruning for captures**: `margin = max(167*depth + captHist*34/1024, 0)`

### Quiet Pruning
`search.cpp:1102-1136`:
- **Continuation history pruning**: `history < -4097 * depth` (using contHist[0], contHist[1], pawnHist)
- **Futility**: `staticEval + 42 + 151*!bestMove + 120*lmrDepth + 86*(staticEval>alpha)` for lmrDepth < 13
  - Best-move-found adjusts margin by 151
  - staticEval > alpha adjusts by 86
- **SEE pruning**: `-25 * lmrDepth^2`

### Singular Extensions (Step 15)
`search.cpp:1149-1201`:
```cpp
// Trigger: ttMove, depth >= 6 + ttPv, TT lower bound, ttDepth >= depth-3, !shuffling
Value singularBeta  = ttData.value - (60 + 66*(ttPv && !PvNode)) * depth / 55;
Depth singularDepth = newDepth / 2;
// Excluded move search...
if (value < singularBeta) {
    // Double/triple extension margins
    int doubleMargin = -4 + 212*PvNode - 182*!ttCapture - corrValAdj
                     - 906*ttMoveHistory/116517 - (ply > rootDepth)*44;
    int tripleMargin = 73 + 320*PvNode - 218*!ttCapture + 92*ttPv - corrValAdj
                     - (ply > rootDepth)*45;
    extension = 1 + (value < singularBeta - doubleMargin) + (value < singularBeta - tripleMargin);
    depth++;  // bonus depth increase on any singular extension
}
else if (value >= beta && !is_decisive(value))
    return value;  // Multi-cut pruning
else if (ttData.value >= beta)
    extension = -3;  // Negative extension
else if (cutNode)
    extension = -2;  // Negative extension on cut nodes
```

Key features:
- **ttMoveHistory** tracking (`StatsEntry<int16_t, 8192>`): tracks whether TT moves are reliable; adjusts singular margins
- **Multi-cut**: if excluded search >= beta, prune the whole node
- **Triple extensions**: up to +3 extension for very singular moves
- **Negative extensions**: -3 when TT value >= beta (non-singular), -2 on cutNodes
- **Anti-shuffling guard**: `is_shuffling()` checks for move repetition patterns

**Coda comparison**: We removed singular extensions entirely (harmful cross-engine). SF's implementation is extremely refined with triple extensions, multi-cut, ttMoveHistory, and shuffling guards. Worth noting that SF's self-play testing environment may suffer from the same overfitting we observed, but their 3500+ Elo level may make the tradeoffs different.

### Alpha Depth Reduction
`search.cpp:1400-1401`:
```cpp
if (depth > 2 && depth < 14 && !is_decisive(value))
    depth -= 2;
```
After alpha raise, reduce remaining search depth by 2. Only in depth range [3, 13].

### Fail-High Score Blending
`search.cpp:1427-1428`:
```cpp
if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
    bestValue = (bestValue * depth + beta) / (depth + 1);
```
Dampens inflated cutoff scores, weighted by depth.

**Coda comparison**: We have this. Identical concept.

### Quiescence Search
`search.cpp:1510-1754`:
- Stand pat with beta blending: `if (bestValue >= beta) bestValue = (bestValue + beta) / 2`
- Futility base: `staticEval + 328`
- Move count limit: `moveCount > 2` for non-check non-recapture
- SEE threshold: `-73`
- **Stalemate detection**: Special case at `search.cpp:1731-1743` detects when capturing our last non-pawn piece could lead to stalemate (checks for pawn pushes and legal king moves)

---

## 3. Move Ordering

### Stages (movepick.cpp:33-57)
```
MAIN_TT -> CAPTURE_INIT -> GOOD_CAPTURE -> QUIET_INIT -> GOOD_QUIET
        -> BAD_CAPTURE -> BAD_QUIET
EVASION_TT -> EVASION_INIT -> EVASION
PROBCUT_TT -> PROBCUT_INIT -> PROBCUT
QSEARCH_TT -> QCAPTURE_INIT -> QCAPTURE
```

### Good/Bad Capture Split
`movepick.cpp:236-240`: Good captures pass SEE check: `pos.see_ge(*cur, -cur->value / 18)`
- SEE threshold is history-dependent (captures with good history get easier SEE threshold)

### Quiet Scoring (movepick.cpp:158-180)
```cpp
m.value = 2 * mainHistory[us][m.raw()];
m.value += 2 * pawnHistory[pc][to];
m.value += contHist[0][pc][to];  // ply-1
m.value += contHist[1][pc][to];  // ply-2
m.value += contHist[2][pc][to];  // ply-3
m.value += contHist[3][pc][to];  // ply-4
m.value += contHist[5][pc][to];  // ply-6 (skip ply-5!)
// Check bonus: 16384 if gives check with SEE >= -75
m.value += bool(check_squares(pt) & to) && see_ge(m, -75)) * 16384;
// Threat-aware: bonus for escaping attack, penalty for moving into attack
int v = 20 * (bool(threatByLesser[pt] & from) - bool(threatByLesser[pt] & to));
m.value += PieceValue[pt] * v;
// Low ply bonus
if (ply < 5)
    m.value += 8 * lowPlyHistory[ply][m.raw()] / (1 + ply);
```

**Coda comparison**: We use 5 continuation history plies. SF uses 6 (skipping ply-5). The threat-aware scoring (escaping/entering attacked squares) is significant -- we should consider adding this. The low-ply history is also unique.

### Good/Bad Quiet Split
`movepick.cpp:210,254,261`:
- Partial insertion sort with threshold: `-3560 * depth`
- Good quiet threshold: `-14000`
- Bad quiets come AFTER bad captures

### No Killers or Countermoves
Stockfish no longer uses explicit killer moves or countermove tables. The continuation history tables subsume their function.

**Coda comparison**: We still use killers and countermove. SF replaced them with richer history tables (especially the 6-ply continuation history).

---

## 4. History Tables

### Types and Limits (history.h)

| Table | Type | Limit (D) | Indexing |
|-------|------|-----------|----------|
| ButterflyHistory | int16 | 7183 | [color][move_raw] |
| LowPlyHistory | int16 | 7183 | [ply<5][move_raw] |
| CapturePieceToHistory | int16 | 10692 | [piece][to_sq][captured_type] |
| PieceToHistory | int16 | 30000 | [piece][to_sq] |
| ContinuationHistory | -- | -- | [inCheck][capture][piece][sq] -> PieceToHistory |
| PawnHistory | int16 | 8192 | [pawn_key][piece][to_sq] (shared, atomic) |
| TTMoveHistory | int16 | 8192 | single entry |

### Update Formula (StatsEntry::operator<<)
`history.h:77-84`:
```cpp
void operator<<(int bonus) {
    int clampedBonus = clamp(bonus, -D, D);
    T val = *this;
    *this = val + clampedBonus - val * abs(clampedBonus) / D;
}
```
Standard gravity formula: `new = old + bonus - old * |bonus| / D`

### Continuation History Multipliers (CMHC)
`search.cpp:1893-1916`:
```cpp
static constexpr array<ConthistBonus, 6> conthist_bonuses = {
    {{1, 1071}, {2, 753}, {3, 329}, {4, 539}, {5, 124}, {6, 434}}};
constexpr int CMHCMultipliers[] = {96, 100, 100, 100, 115, 118, 129};
```
- Updates plies -1, -2, -3, -4, -5, -6 with decaying weights
- **History consistency multiplier**: tracks how many previous entries are positive; increases bonus when prior history agrees (positive chain = stronger signal)
- Adds flat +73 to first 2 entries
- Only updates first 2 entries when in check

**Coda comparison**: We update 4 continuation history entries with fixed weights. SF's consistency multiplier is novel -- it rewards moves that have consistently good history across multiple plies.

### Shared Histories (NUMA-Aware)
`history.h:222-269`: PawnHistory and CorrectionHistory are **shared between threads on the same NUMA node** using DynStats with atomic operations.
- Size scales with thread count (next power of two)
- Per-NUMA-node to avoid cross-node memory traffic
- Indexed by hash key (pawn_key, minor_piece_key, non_pawn_key)

### Correction History (5 Types!)
`history.h:160-214`:
1. **Pawn**: indexed by pawn structure hash
2. **Minor**: indexed by minor piece (N,B) position hash
3. **NonPawn**: indexed by non-pawn material hash, per color
4. **PieceTo**: indexed by [piece][to_sq]
5. **Continuation**: indexed by [piece][to_sq] pairs (2 and 4 plies back)

All corrections are **unified** into a single `CorrectionBundle` per hash entry containing pawn, minor, nonPawnWhite, nonPawnBlack fields -- reducing cache misses.

Correction value calculation (`search.cpp:79-93`):
```cpp
int pcv   = pawn_correction_entry(pos).at(us).pawn;
int micv  = minor_piece_correction_entry(pos).at(us).minor;
int wnpcv = nonpawn_correction_entry<WHITE>(pos).at(us).nonPawnWhite;
int bnpcv = nonpawn_correction_entry<BLACK>(pos).at(us).nonPawnBlack;
int cntcv = (*(ss-2)->continuationCorrectionHistory)[pc][to]
          + (*(ss-4)->continuationCorrectionHistory)[pc][to];
return 12153*pcv + 8620*micv + 12355*(wnpcv+bnpcv) + 7982*cntcv;
```
Applied as: `v + cv / 131072`

Update weights (`search.cpp:101-124`):
```cpp
pawn_correction << bonus;
minor_correction << bonus * 153 / 128;
nonPawn_correction << bonus * 187 / 128;
contCorr[ply-2] << bonus * 126 / 128;
contCorr[ply-4] << bonus * 63 / 128;
```
Limit: `CORRECTION_HISTORY_LIMIT = 1024`

**Coda comparison**: We have pawn correction history only. SF has 5 types! The minor piece and non-pawn corrections are new, and the continuation correction history is particularly interesting -- it tracks whether specific move pairs lead to eval errors.

### History Aging
`search.cpp:316-318`:
```cpp
for (Color c : {WHITE, BLACK})
    for (int i = 0; i < UINT_16_HISTORY_SIZE; i++)
        mainHistory[c][i] = mainHistory[c][i] * 820 / 1024;
```
Main history decayed by factor `820/1024 = ~0.80` at the start of each iteration.

### EvalDiff History Update
`search.cpp:880-887`: Uses static eval difference between current and previous position to update history:
```cpp
int evalDiff = clamp(-(int((ss-1)->staticEval + ss->staticEval)), -214, 171) + 60;
mainHistory[~us][(ss-1)->currentMove.raw()] << evalDiff * 10;
```
When the opponent's move made our eval worse (high evalDiff), it gets a history bonus. This is separate from the regular search-result-based history updates.

---

## 5. Transposition Table

### Entry Structure (tt.cpp:49-72)
10 bytes per entry:
```
key16:      16 bits (low 16 bits of Zobrist)
depth8:      8 bits (depth + DEPTH_ENTRY_OFFSET)
genBound8:   8 bits (5 gen + 1 pv + 2 bound)
move16:     16 bits
value16:    16 bits
eval16:     16 bits
```

### Cluster Layout
3 entries per cluster + 2 bytes padding = 32 bytes (cache line aligned).
Hash function: `mul_hi64(key, clusterCount)` -- avoids modulo.

### Replacement Strategy (tt.cpp:93-113)
Overwrite if:
1. BOUND_EXACT, OR
2. Different position (key16 mismatch), OR
3. Higher depth (with PV bonus: `depth - offset + 2*pv > depth8 - 4`), OR
4. Stale entry (different generation)

Preserve old move if new store has no move.

### Aging
5-bit generation, incremented by 8 each search (GENERATION_DELTA=8, using 3 low bits for bound/pv).

**Coda comparison**: Our TT is 4-slot buckets, 16 bytes per slot. SF uses 3-slot clusters at 10 bytes per entry = 32 bytes per cluster. Our entries are larger but we store more per cluster.

---

## 6. Lazy SMP

### Thread Architecture (thread.cpp)
- Each thread has its own Worker with private:
  - Board position
  - History tables (mainHistory, captureHistory, continuationHistory, continuationCorrectionHistory)
  - NNUE accumulators + Finny table cache
  - LowPlyHistory, TTMoveHistory
- Shared across threads:
  - TT (racy reads/writes, no locking)
  - PawnHistory (atomic)
  - CorrectionHistory (atomic, unified bundles)
  - Networks (NUMA-replicated)
- NUMA-aware: histories shared only within same NUMA node

### Depth Diversification
No explicit depth offset between threads! SF relies on natural variation from racy TT reads and thread scheduling.

However: aspiration window start varies: `delta = 5 + threadIdx % 8 + ...` gives each thread a different starting window width.

### Best Thread Selection (thread.cpp:347-417)
Vote-based:
```cpp
auto thread_voting_value = [minScore](Thread* th) {
    return (th->rootMoves[0].score - minScore + 14) * int(th->completedDepth);
};
// Each thread votes for its best move, weighted by (score-min+14)*depth
// Thread with most-voted move wins, with tiebreaks on voting value
```
Special handling for proven wins/losses (shortest mate preferred).

**Coda comparison**: We use the deepest-thread-wins approach. SF's vote-weighted scheme is more robust -- it considers both score AND depth, and handles the case where multiple threads agree on a move.

### Node Counting
`search.cpp:571`: `nodes.store(nodes.load(relaxed) + 1, relaxed)` -- non-atomic increment, relaxed ordering. Avoids locked instructions.

### increaseDepth Flag
`search.cpp:543`: `threads.increaseDepth = ponder || elapsed <= totalTime * 0.50`
When false, `searchAgainCounter` increments, reducing effective search depth. This creates natural depth wobble that helps thread diversification.

---

## 7. Time Management

### Init (timeman.cpp:47-139)
```cpp
int centiMTG = movestogo ? min(movestogo * 100, 5000) : 5051;
if (scaledTime < 1000) centiMTG = int(scaledTime * 5.051);

TimePoint timeLeft = max(1, time + (inc * (centiMTG-100) - moveOverhead * (200+centiMTG)) / 100);

// x+inc mode:
originalTimeAdjust = 0.3272 * log10(timeLeft) - 0.4141;
double logTimeInSec = log10(scaledTime / 1000.0);
double optConstant = min(0.0029869 + 0.00033554 * logTimeInSec, 0.004905);
double maxConstant = max(3.3744 + 3.0608 * logTimeInSec, 3.1441);
optScale = min(0.012112 + pow(ply+3.22713, 0.46866) * optConstant, 0.19404 * time / timeLeft) * originalTimeAdjust;
maxScale = min(6.873, maxConstant + ply / 12.352);
```

### Dynamic Adjustments (search.cpp:500-543)
```cpp
// Effort: fraction of nodes spent on best move
uint64_t nodesEffort = rootMoves[0].effort * 100000 / max(1, nodes);

// Falling eval: more time when score is dropping
double fallingEval = (12.44 + 2.318*(prevAvgScore - bestValue) + 0.95*(iterValue - bestValue)) / 100.0;
fallingEval = clamp(fallingEval, 0.581, 1.655);

// Best move stability: less time when best move is stable
double k = 0.476;
double center = lastBestMoveDepth + 11.565;
timeReduction = 0.64 + 0.93 / (0.953 + exp(-k * (completedDepth - center)));

double reduction = (1.5 + previousTimeReduction) / (2.255 * timeReduction);
double bestMoveInstability = 1.088 + 2.315 * totBestMoveChanges / threads.size();
double highBestMoveEffort = nodesEffort > 86000 ? 0.74 : 0.96;

double totalTime = optimum * fallingEval * reduction * bestMoveInstability * highBestMoveEffort;
```

Key features:
- **Effort tracking**: per-root-move node counts, high effort on best move means we can stop sooner
- **Falling eval**: sigmoid-like response to score drops across iterations
- **Move stability**: logistic function of depth since last best-move change
- **Best move changes**: aggregated across all threads
- **Previous time reduction**: smoothed with last iteration's value

**Coda comparison**: Our time management is much simpler. The effort tracking and move stability sigmoid are sophisticated features we don't have.

---

## 8. Novel Features

### followPV
`search.cpp:681-683`:
```cpp
ss->followPV = rootNode
    || ((ss-1)->followPV && ply-1 < lastIterationPV.size()
        && (ss-1)->currentMove == lastIterationPV[ply-1]);
```
Tracks whether current search path follows the previous iteration's PV. Used to:
- Skip IIR on PV continuation
- Skip history pruning on PV continuation

### TTMoveHistory
Single global stat (`StatsEntry<int16_t, 8192>`) tracking TT move reliability:
- After fail-high: `ttMoveHistory << (bestMove == ttData.move ? 805 : -787)`
- Used in singular extension: adjusts double extension margin by `906 * ttMoveHistory / 116517`
- On multi-cut: `ttMoveHistory << max(-424 - 107*depth, -3375)`

### CutoffCnt Feedback
`search.cpp:1228-1229`: Child node cutoff count fed back to parent for reduction:
```cpp
if ((ss+1)->cutoffCnt > 1)
    r += 251 + 1124 * ((ss+1)->cutoffCnt > 2) + 1042 * allNode;
```

### Shuffling Detection
`search.cpp:144-151`:
```cpp
bool is_shuffling(Move move, Stack* ss, Position& pos) {
    if (capture || rule50 < 11) return false;
    if (pliesFromNull <= 6 || ply < 18) return false;
    return move.from_sq() == (ss-2)->currentMove.to_sq()
        && (ss-2)->currentMove.from_sq() == (ss-4)->currentMove.to_sq();
}
```
Detects repetitive piece shuffling and disables singular extensions (prevents infinite extension chains in shuffling endgames).

### MeanSquaredScore
`search.cpp:101,354`: Root moves track mean squared score (actually signed: `value * abs(value)`), used to adjust aspiration window width based on score variance.

### Value Draw Randomization
`search.cpp:127`: `VALUE_DRAW - 1 + (nodes & 0x2)` -- randomly returns -1 or +1 for draws to avoid 3-fold blindness.

### Low Ply History
Separate history table for plies 0-4, helps with root move ordering where standard history is less informative.

### Correction History in Singular Extensions
`search.cpp:1162`: `int corrValAdj = abs(correctionValue) / 210590` -- large correction history (uncertain eval) tightens singular margins.

### Skip Quiet Moves (not hard LMP)
`movepick.cpp:311`: `skip_quiet_moves()` sets a flag that makes the move picker skip all remaining quiets but still return captures. This is softer than traditional LMP which uses `continue`.

---

## 9. Performance Optimizations

### Sparse First Layer
As described above, `affine_transform_sparse_input.h` exploits CReLU sparsity to skip zero-block columns in the first hidden layer matmul.

### Weight Permutation for SIMD
`FeatureTransformer::permute_weights()` reorders weights to match the lane layout of `_mm256_packus_epi16` (AVX2) or `_mm512_packus_epi16` (AVX512). This avoids lane-crossing shuffles during inference.

### TT Prefetching
Position.do_move accepts TT and SharedHistories pointers, prefetches the next TT entry after making a move (before the search at the new position begins).

### Non-Atomic Node Counting
`nodes.store(nodes.load(relaxed) + 1, relaxed)` instead of `fetch_add` -- avoids locked bus cycles on x86.

### NUMA Awareness
- Networks are replicated per NUMA node (`LazyNumaReplicatedSystemWide`)
- Shared histories are per-NUMA-node with atomic operations
- Thread-to-NUMA binding via `OptionalThreadToNumaNodeBinder`
- Large page allocation for TT and history tables

### LEB128 Network Compression
Variable-length encoding for network parameters reduces file size and speeds up loading.

---

## 10. Key Differences from GoChess/Coda

| Feature | Stockfish | GoChess/Coda |
|---------|-----------|-------------|
| NNUE architecture | Dual net (big 1024 + small 128), dual activation (SCReLU+CReLU), 8 layer stacks, skip connection, threat features | Single net, single activation, no layer stacks |
| Sparse matmul | Yes (first hidden layer) | No |
| Killers/Countermove | Removed (replaced by cont. history) | Still used |
| Continuation history | 6 plies with consistency multiplier | 4 plies with fixed weights |
| Correction history | 5 types (pawn, minor, nonpawn, pieceto, continuation) | 1 type (pawn) |
| History aging | Decay by 0.80 per iteration | None |
| Move ordering threats | threatByLesser array in quiet scoring | None |
| NMP restriction | cutNode only | All non-PV |
| Singular extensions | Yes (with triple ext, multi-cut, ttMoveHistory) | Removed |
| Aspiration windows | Thread-diversified, variance-adjusted | Fixed delta |
| Time management | Effort tracking, stability sigmoid, falling eval | Simple |
| SMP best thread | Vote-weighted by (score+14)*depth | Deepest thread |
| TT verification | Double-TT check for GHI at depth >= 7 | None |
| Eval blending | PSQT/positional split, complexity damping, optimism | Direct NNUE output |
| Quiescence | Stalemate detection, movecount limit | Standard |
