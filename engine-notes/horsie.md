# Horsie Chess Engine - Technical Review

Source: `/home/adam/chess/engines/Horsie/`  
Version: 1.1 (May 12 2025)  
Language: C++ (C++20, ported from Lizard C# engine)  
NNUE: Single accumulator (768x14 PST -> 2048 -> 16 -> 32 -> 1x8)  
Rating: 3575 CCRL 40/15 (v1.0), TBD for v1.1  

Last reviewed: 2026-04-19

---

## 1. NNUE Architecture

### Network Topology: (PST[768x14]) -> 2048 -> 16 -> 32 -> 1x8

**Note**: Horsie's first hidden layer (L1_SIZE=2048) is substantially larger than typical engines. This accommodates the pairwise activation scheme without separate threat accumulators.

**Constants** (`arch.h:7-17`):
- `INPUT_BUCKETS = 14`, `INPUT_SIZE = 768`
- `L1_SIZE = 2048` (first hidden layer width — 2.67x larger than Reckless/Coda)
- `L2_SIZE = 16`, `L3_SIZE = 32`
- `FT_QUANT = 255`, `L1_QUANT = 132`, `OutputScale = 400`
- `FT_SHIFT = 10` (used in activation)
- `L1_MUL = (1 << FT_SHIFT) / (FT_QUANT * FT_QUANT * L1_QUANT)` = precise dequant formula

**Coda comparison**: Coda v9 uses single 768 accumulator + 66k threat features feeding to 768 -> 16 -> 32. Horsie consolidates both into a single wider L1 (2048). This is a fundamentally different approach: width instead of parallel feature streams. Network size: Horsie's 768*14*2048 = 21.96M FT weights vs Coda's 768*10*768 + 66864*768 = 5.94M + 51.4M = 57.3M (but split across two streams). Horsie's approach is simpler and more compact in FT but pays for it with a much larger L1.

### 1.1 PST Accumulator

**HalfKA(L) configuration with 14 king buckets** (`nn.h:56-69`):
```
King bucket layout (8 ranks, compressed):
Rank 1: buckets 0-3, 17-14 (mirrored)
Rank 2: buckets 4-7, 21-18
Rank 3: buckets 8-11, 25-22
Rank 4: buckets 8-11, 25-22 (same as rank 3)
Rank 5-8: buckets 12-13, 27-26 (further compressed)
```

The layout has **asymmetric compression**: back ranks (1-3, 6-8) have 8 buckets each, middle ranks (4-5) share buckets. Uses `BucketForPerspective(king_square, perspective)` with mirroring when king file >= 4.

**Incremental updates**: Standard finny table approach with piece add/remove operations (`accumulator.cpp`). Each perspective maintains separate accumulators.

**Coda comparison**: Coda uses 16 king buckets (more granular). Horsie's 14-bucket scheme is less fine-grained but acceptable for the wider L1. The lookup table (`KingBuckets[]`) is a ROM permutation, standard approach.

### 1.2 Feature Computation

PST feature index: `bucket * 768 + 384 * (color != perspective) + 64 * piece_type + square`

Weights: `i16[INPUT_BUCKETS * L1_SIZE * INPUT_SIZE]` = `i16[14 * 2048 * 768]` = ~21.96M weights.

Biases: `i16[L1_SIZE]` = `i16[2048]`.

**No threat features**: Unlike Reckless, Horsie uses only PST. All tactical information must be learned implicitly by the wider L1. This simplifies network training and inference at the cost of requiring more L1 capacity.

**Coda comparison**: We have 66,864 threat features providing explicit tactical information. Horsie's implicit approach trades capacity for simplicity. The threat accumulator in Reckless/Coda costs ~30-50 Elo but requires careful maintenance. Horsie's single-stream approach is simpler to train in Bullet.

### 1.3 Activation Function

**Pairwise activation** (no separate threat stream):
```cpp
for each perspective:
    for each pair (i, i + 1024):
        left = clamp(pst[i], 0, 255)
        right = clamp(pst[i+1024], 0, 255)
        output[i] = (left * right) >> FT_SHIFT
```

The L1 (2048 units) is treated as 1024 pairs. Each pair is independently activated with a pairwise product (similar to Reckless's SCReLU but without asymmetric clamping on right half).

Activation is **symmetric** — both halves of each pair are clamped [0,255], then multiplied. Output is scaled by `1024 / (255*255*132) = 0.0000478...` (FT_SHIFT=10).

**Coda comparison**: Our v9 uses asymmetric pairwise activation (left * right) with different clamp bounds on right half. Horsie's symmetric approach is simpler but potentially less expressive. The FT_SHIFT=10 (vs Reckless FT_SHIFT=9) means slightly different scaling precision.

### 1.4 Hidden Layers (L1, L2, L3)

**L1 matmul** (`nn.cpp:150+`):
- Input: u8[2048] (pairwise activation output)
- Weights: `i8[OUTPUT_BUCKETS][L1_SIZE * L2_SIZE]` = `i8[8 * 2048 * 16]` = 262.144K weights (int8)
- Biases: `float[OUTPUT_BUCKETS][L2_SIZE]`
- **Sparse matmul via NNZ detection**: Finds non-zero 4-byte chunks in u8 activation, uses sparse multiply
- Output: `float[OUTPUT_BUCKETS][L2_SIZE]`

**L2 matmul**:
- Weights: `float[OUTPUT_BUCKETS][L2_SIZE * L3_SIZE]` = `float[8 * 16 * 32]` = 4096 floats
- Biases: `float[OUTPUT_BUCKETS][L3_SIZE]`
- Standard FMA multiply-add with clamp [0, 1]
- Output: `float[OUTPUT_BUCKETS][L3_SIZE]`

**L3 (output)**:
- Weights: `float[OUTPUT_BUCKETS][L3_SIZE]` = 256 floats
- Biases: `float[OUTPUT_BUCKETS]`
- Dot product: `sum(L2[i] * L3_weights[i]) + bias`
- Final: `output * 400`

**SIMD optimizations** (`simd.h`, `nn.cpp`):
- Sparse L1 matmul with bitmask-based NNZ detection
- `_mm256_dpbusd_epi32` (dot-product u8*i8->i32) core operation
- Register blocking with interleaved add/sub for throughput
- AVX512, AVX2, NEON paths

**Coda comparison**: We do dense L1 matmul. Horsie's sparse approach with NNZ detection should be faster when many activations are zero (which they are with pairwise). The int8 L1 weights are compact. Our float L1/L2/L3 is more uniform but less space-efficient.

### 1.5 Output Buckets

Material-based (piece count), 8 buckets:
```
pieces  2-8:  bucket 0
        9-12: bucket 1
       13-16: bucket 2
       17-19: bucket 3
       20-22: bucket 4
       23-25: bucket 5
       26-28: bucket 6
       29-32: bucket 7
```

Index: `(popcount(all pieces) - 2) / ((32 + 8 - 1) / 8) = (pc - 2) / 4` (integer division).

**Coda comparison**: Same as ours. Standard approach.

---

## 2. Search

### 2.1 Framework

- Iterative deepening with aspiration windows
- Template-specialized node types: `RootNode`, `PVNode`, `NonPVNode`
- Stack-based search with offset indexing (`ss[ply-N]` for N in {1,2,4,6})
- Multi-threaded with thread pool and shared TT

### 2.2 Aspiration Windows

Simple fixed window: `AspWindow = 12` (tuned, from `search_options.h:103`)

On fail-low:
- `alpha = max(alpha - window, -INF)`
- `beta = (alpha + beta) / 2`

On fail-high:
- `beta = min(beta + window, INF)`
- `alpha = (alpha + beta) / 2`

Window widens by `window += window / 2` (50% growth per retry).

**Coda comparison**: Reckless uses eval-dependent delta; Horsie uses fixed. Reckless has asymmetric growth (21% fail-low, 49% fail-high). Horsie's simplicity is an advantage for tuning.

### 2.3 TT Cutoff (`search.cpp:287-294`)

Standard early cutoff:
```
tte->Depth >= depth && ttScore != ScoreNone
&& (ttScore < alpha || cutNode)
&& (tte->Bound & (ttScore >= beta ? LOWER : UPPER))
```

Does not allow shallower TT entries to beta-cutoff. Fairly strict depth check.

**Coda comparison**: More conservative than Reckless's `tt_depth > depth - (tt_score < beta)`.

### 2.4 Eval Adjustment (`search.cpp:1000-1006`, `history.h:141-151`)

**Eval-from-correction**:
```cpp
AdjustEval(pos, rawEval):
    halfmove_scaled = rawEval * (200 - halfmove_clock) / 200
    correction = GetCorrection(pos)
    return halfmove_scaled + correction
```

**Correction sources** (3 components, tuned):
```
corr = PawnCorrCoeff * PawnCorrection[stm][pos.PawnHash() % 16384]
     + NonPawnCorrCoeff * NonPawnCorrection[stm][pos.NonPawnHash(WHITE) % 16384]
     + NonPawnCorrCoeff * NonPawnCorrection[stm][pos.NonPawnHash(BLACK) % 16384]
corr /= CorrDivisor (3136)
```

- `PawnCorrCoeff = 129`, `NonPawnCorrCoeff = 70`, `CorrDivisor = 3136`
- **Per-thread correction history** (not shared)

**Coda comparison**: Reckless has 6 correction sources (pawn, minor, non_pawn x2, continuation x2) and shares pawn/minor/non_pawn across threads. Horsie keeps it simple with 3 sources and per-thread storage. The halfmove_clock decay is standard.

### 2.5 Null Move Pruning (`search.cpp:363-396`)

Conditions:
- `!isPV`, `!doSkip`, `depth >= NMPMinDepth (6)`
- `eval >= beta` AND `eval >= prev_eval`
- `pos.HasNonPawnMaterial(us)`
- Not after a null move (single NMP per line)

Reduction formula:
```cpp
reduction = NMPBaseRed (4) 
          + (depth / NMPDepthDiv (4))
          + min((eval - beta) / NMPEvalDiv (151), NMPEvalMin (2))
```

Verification search at depth >= 16 with `NMPPly` (singularity avoidance).

**Coda comparison**: Standard NMP. Horsie's eval-dependent part (`(eval-beta)/151`) is moderate. No exotic gates like Reckless's "no NMP when TT shows capture threat".

### 2.6 Reverse Futility Pruning (`search.cpp:337-347`)

```cpp
RFP fires when:
    !TTPV && !doSkip && depth <= RFPMaxDepth (6)
    && ttMove == Null && eval >= beta
    && (eval - GetRFPMargin(depth, improving)) >= beta

GetRFPMargin(depth, improving):
    margin = RFPMargin (51) + (improving ? 0 : extra)  // likely tuned differently
```

Returns `(eval + beta) / 2` (blended).

**Coda comparison**: Simple depth-based margin. No correction-value awareness like Reckless.

### 2.7 Razoring (`search.cpp:350-360`)

```cpp
if (depth <= RazoringMaxDepth (4)
    && !isPV
    && alpha < 2000
    && eval + RazoringMult (292) * depth <= alpha):
    return QSearch(alpha, alpha+1)
```

Depth capped at 4, linear margin scaling (`292 * depth`).

**Coda comparison**: Standard. Reckless uses quadratic scaling.

### 2.8 ProbCut (`search.cpp:408-451`)

```cpp
probBeta = beta + (improving ? ProbcutBetaImp (97) : ProbcutBeta (252))
```

Cut-node only. Generates captures, checks SEE, performs qsearch, then optional depth-reduced verify search.

Simple implementation, no dynamic depth adjustment like Reckless.

**Coda comparison**: Streamlined, no fancy scoring adjustments.

### 2.9 Singular Extensions (`search.cpp:561-607`)

Conditions: `depth >= SEMinDepth (5) + (isPV && tte->PV())`, `tte->Depth() >= depth - 3`, lower-bound TT.

```cpp
singular_beta = ttScore - (SENumerator (11) * depth / 10)
singular_depth = (depth + SEDepthAdj (-1)) / 2

extension = 1 if score < singular_beta
          + 1 if score < singular_beta - SEDoubleMargin (21)
          + 1 if score < singular_beta - SETripleMargin (101) - (is_capture * SETripleCapSub (73))
```

Also negative extensions (-2 if ttScore >= beta, -1 if ttScore <= alpha, etc.).

**Coda comparison**: Similar structure to Reckless. Horsie uses the same triple-extension idea. Both engines keep SE because their strong NNUE makes singular extensions reliable (unlike older engines).

### 2.10 Late Move Pruning (`search.cpp:475`)

```cpp
LMPTable[improving][depth]:
    not_improving: (3 + depth^2) / 2
    improving:    3 + depth^2
```

Applied in move loop: `legalMoves >= lmpMoves` triggers skipQuiets.

**Coda comparison**: Standard quadratic formula. Horsie implements it simply as a precomputed table.

### 2.11 Late Move Reductions (`search.cpp:513-657`)

Base formula (in 128ths of a ply):
```cpp
R = round(log(depth) * log(moveCount) / 2.25 + 0.25) * 128
```

Stored in precomputed table `LogarithmicReductionTable[depth][moveCount]` (`precomputed.cpp:89-104`).

Adjustments (each modifies R in 128ths):
- `!improving ? +LMRNotImpCoeff (119) : 0`
- `cutNode ? +LMRCutNodeCoeff (280) : 0`
- `TTPV ? -LMRTTPVCoeff (124) : 0`
- `move == KillerMove ? -LMRKillerCoeff (129) : 0`
- History-based: `-(histScore / (isCapture ? LMRCaptureDiv (9229) : LMRQuietDiv (15234)))`
- `R /= 128` (convert back to plies)

**History score computation** (`search.cpp:631-634`):
```cpp
histScore = LMRHist (189) * moveHist
          + LMRHistSS1 (268) * Continuation[ply-1]
          + LMRHistSS2 (128) * Continuation[ply-2]
          + LMRHistSS4 (127) * Continuation[ply-4]
```

Continuation history at offsets 1, 2, 4 only (no offset 6).

**Post-LMR adjustment** (`search.cpp:649-653`):
```cpp
deeper   = score > (bestScore + DeeperMargin (45) + 4*newDepth)
shallower = score < (bestScore + newDepth)
newDepth += deeper - shallower
```

Adaptive: if much better, extend; if barely better, reduce. Applied before full-depth re-search.

**Coda comparison**: Similar logarithmic base formula. Horsie skips continuation offset 6 (uses 1,2,4 vs Coda's 1,2,4,6). History divisors are tuned; Coda likely has different values. The post-LMR adjustment is nice and similar to Reckless's depth adjustments.

### 2.12 Move History Updates

**Bonus/penalty formulas** (`threadpool.h:136-140`):
```cpp
StatBonus(depth)   = min(StatBonusMult (192) * depth - StatBonusSub (91), StatBonusMax (1735))
StatPenalty(depth) = -min(StatPenaltyMult (655) * depth - StatPenaltySub (106), StatPenaltyMax (1267))

LMRBonus(depth)    = min(LMRBonusMult (182) * depth - LMRBonusSub (83), LMRBonusMax (1703))
LMRPenalty(depth)  = -min(LMRPenaltyMult (173) * depth - LMRPenaltySub (82), LMRPenaltyMax (1569))
```

**Gravity formula** (history.h:40):
```cpp
entry += (bonus - bonus.abs() * entry / MAX)
```

Standard gravity with MAX values:
- `HistoryClamp = 16384` (main/capture history)
- `LowPlyClamp = 8192` (ply history at low depths)
- `ContinuationMax = 16384`

**Updates** (`search.cpp:951-997`):
- Best move: `StatBonus(depth)` if quiet, `StatBonus` if noisy
- Failed moves: `StatPenalty(depth)`, with shared penalty for multiple failures

**LMR history updates** (`search.cpp:659-661`):
```cpp
bonus = (score <= alpha) ? LMRPenalty(newDepth) : (score >= beta) ? LMRBonus(newDepth) : 0
```

Applied to continuation history.

**Coda comparison**: Linear-with-offset shape (`MULT*d - OFFSET`). Same gravity formula. Tuned values differ but approach is standard.

### 2.13 Quiet Move Ordering (`search.cpp:326-334`)

**Quiet order from previous move eval difference**:
```cpp
if (prior_move && !prior_in_check && prior_not_capture):
    val = -(QuietOrderMult (153) * (prev_eval + curr_eval)) / 16
    bonus = clamp(val, -QuietOrderMin (101), QuietOrderMax (192))
    UpdateMainHistory(Not(us), priorMove, bonus)
```

Updates the *opponent's* main history based on how well their previous move did. Interesting retrospective heuristic.

**Coda comparison**: We don't have this eval-difference-based quiet history update. Worth porting.

### 2.14 Quiescence Search (`search.cpp:777-918`)

Key features:
- **Stand-pat blending**: `eval = (4*eval + beta) / 5` (dampens inflated stand-pat)
- **TT cutoff** with score bounds check
- **Futility**: `futility = eval + QSFutileMargin (204)`
- **SEE pruning**: `!pos.SEE_GE(m, -QSSeeMargin (75))`
- Move limiting: `!inCheck && legalMoves > 3 : break`

No quiet move generation (only captures/promotions/checks in check).

**Coda comparison**: Standard qsearch. Horsie's stand-pat blending (4:1 toward beta) is specific; we use different ratios.

### 2.15 Time Management (`search.cpp:171-190`)

**Soft-time-management factors**:
```cpp
if (RootDepth > 7):
    nodeFactor = (1.5 - RootMoves[0].Nodes / totalNodes) * 1.75
    bmStability = StabilityCoefficients[min(stability, 6)]
                = {2.2, 1.6, 1.4, 1.1, 1.0, 0.95, 0.9}[stability]
    scoreStability = max(0.85, min(1.15, 0.034 * (score_4plies_ago - current_score)))
    multFactor = nodeFactor * bmStability * scoreStability
    
    stop if GetSearchTime() >= SoftTimeLimit * multFactor
```

Three stability components:
1. **Node allocation**: If best move gets >60% of nodes, reduce multiplier (it's decisive)
2. **Best-move stability**: Stability counter (same move across depths) gives diminishing returns
3. **Score stability**: Dropping score extends search, rising score shortens

**Coda comparison**: Similar multi-factor approach to Reckless. Horsie uses score volatility (0.034 coefficient). All factors are tuned constants.

---

## 3. Move Ordering

### 3.1 Scoring Stages

1. **Hash move**: TT move scored highest
2. **Captures**: Scored by noisy history + MVV bonus
3. **Quiets**: Scored by main history + continuation history (4 plies) + ply history + escape/check/en-prise bonuses
4. **Bad captures**: SEE-failing noisy moves

No killer moves or counter-move table — history alone.

### 3.2 Capture Scoring (`search.cpp:1152-1153`)

```cpp
score = GetNoisyHistory(piece, to, capturedPiece)
      + (MVVMult (363) * GetPieceValue(capturedPiece)) / 32
```

**SEE threshold for good captures** (`search.cpp:554-557`):
```cpp
seeMargin = -ShallowSEEMargin (81) * depth
```

Captures failing SEE go to bad_noisy list.

**Coda comparison**: Standard MVV/LVA variant with capture history. Horsie uses `(363 * piece_value) / 32` scaling.

### 3.3 Quiet Scoring (`search.cpp:1158-1188`)

```cpp
score = NMOrderingMH (419) * mainHist
      + NMOrderingSS1 (491) * contHist[ply-1]
      + NMOrderingSS2 (258) * contHist[ply-2]
      + NMOrderingSS4 (255) * contHist[ply-4]
      + NMOrderingSS6 (246) * contHist[ply-6]    // NOTE: includes ply-6
      + GetPlyHistory(ply, move)
      
score /= 256

if GivesCheck: score += CheckBonus (9816)

if piece == Queen:
    if from & rookThreats: score += 24 * OrderingEnPriseMult (520)
    if to & rookThreats:   score -= 22 * OrderingEnPriseMult

... similar for Rook (minor/pawn threats), Bishop/Knight (pawn threats)
```

**Coda comparison**: Uses ply history + continuation at 4 offsets + threat-based escape/en-prise bonuses. We use continuation at 2 offsets. Horsie's ply-6 offset is interesting (6-ply recency).

### 3.4 History Table Design

**MainHistory** (`history.h:65`):
```cpp
Stats<i16, HistoryClamp(16384), 2, 64*64>
= [color][from*64 + to]
```
2-player x 4096 squares.

**CaptureHistory** (`history.h:66`):
```cpp
Stats<i16, HistoryClamp(16384), 12, 64, 6>
= [piece][to][capturedPiece]
```
12 pieces x 64 squares x 6 piece types.

**PlyHistory** (`history.h:67`):
```cpp
Stats<i16, LowPlyClamp(8192), 4, 64*64>
= [ply % 4][from*64 + to]  // only at ply < 4
```

**ContinuationHistory** (`history.h:72`):
```cpp
NDArray<PieceToHistory, 12, 64>
= [piece][to] -> [incoming_piece][incoming_to]
accessed at ply-1, ply-2, ply-4, ply-6
```

Single-valued entries (no factorization like Reckless's factorizer + threat buckets).

**Coda comparison**: Simpler structure than Reckless's threat-aware factorized history. Single value per entry. Horsie's approach is traditional and less sophisticated but easier to tune and maintain.

### 3.5 En-Prise Bonuses (Move Scoring)

Pre-computed threat bitboards (`search.cpp:1135-1137`):
```cpp
pawnThreats = pos.ThreatsBy<PAWN>(opp)
minorThreats = pos.ThreatsBy<KNIGHT>(opp) | pos.ThreatsBy<BISHOP>(opp) | pawnThreats
rookThreats = pos.ThreatsBy<ROOK>(opp) | minorThreats
```

Bonuses for escaping threats from lower-value pieces. Integrated into move scoring.

---

## 4. Tunable Parameters Summary

**Table of key SPSA-tunable values** (`search_options.h`):

| Category | Param | Value | Notes |
|----------|-------|-------|-------|
| **Aspiration** | AspWindow | 12 | Fixed delta |
| **RFP** | RFPMaxDepth | 6 | Depth limit |
| | RFPMargin | 51 | Beta margin |
| **NMP** | NMPMinDepth | 6 | Depth gate |
| | NMPBaseRed | 4 | Base reduction |
| | NMPDepthDiv | 4 | Depth divisor |
| | NMPEvalDiv | 151 | Eval divisor |
| | NMPEvalMin | 2 | Cap on eval term |
| **Razoring** | RazoringMaxDepth | 4 | Depth limit |
| | RazoringMult | 292 | Margin coefficient |
| **ProbCut** | ProbcutBeta | 252 | Non-improving margin |
| | ProbcutBetaImp | 97 | Improving margin |
| **SE** | SEMinDepth | 5 | Depth gate |
| | SENumerator | 11 | Singular beta offset |
| | SEDoubleMargin | 21 | Double extension threshold |
| | SETripleMargin | 101 | Triple extension threshold |
| | SETripleCapSub | 73 | Capture-specific adjustment |
| | SEDepthAdj | -1 | Singular search depth |
| **Futility** | NMFutileBase | 58 | Base margin |
| | NMFutilePVCoeff | 136 | PV coefficient |
| | NMFutileImpCoeff | 132 | Improving coefficient |
| | NMFutileHistCoeff | 128 | History coefficient |
| | NMFutMarginB | 163 | Base margin |
| | NMFutMarginM | 81 | Depth multiplier |
| | NMFutMarginDiv | 129 | History divisor |
| | ShallowSEEMargin | 81 | SEE pruning margin |
| | ShallowMaxDepth | 9 | Depth limit |
| **LMR** | LMRHist | 189 | Main history weight |
| | LMRHistSS1 | 268 | Continuation[ply-1] weight |
| | LMRHistSS2 | 128 | Continuation[ply-2] weight |
| | LMRHistSS4 | 127 | Continuation[ply-4] weight |
| | LMRNotImpCoeff | 119 | Not-improving penalty |
| | LMRCutNodeCoeff | 280 | Cut-node penalty |
| | LMRTTPVCoeff | 124 | TT-PV reduction |
| | LMRKillerCoeff | 129 | Killer reduction |
| | LMRQuietDiv | 15234 | Quiet history divisor |
| | LMRCaptureDiv | 9229 | Capture history divisor |
| | DeeperMargin | 45 | Post-LMR deeper threshold |
| **Stat Updates** | StatBonusMult | 192 | Bonus multiplier |
| | StatBonusSub | 91 | Bonus offset |
| | StatBonusMax | 1735 | Bonus cap |
| | StatPenaltyMult | 655 | Penalty multiplier |
| | StatPenaltySub | 106 | Penalty offset |
| | StatPenaltyMax | 1267 | Penalty cap |
| | LMRBonusMult | 182 | LMR bonus multiplier |
| | LMRBonusSub | 83 | LMR bonus offset |
| | LMRBonusMax | 1703 | LMR bonus cap |
| | LMRPenaltyMult | 173 | LMR penalty multiplier |
| | LMRPenaltySub | 82 | LMR penalty offset |
| | LMRPenaltyMax | 1569 | LMR penalty cap |
| **Move Ordering** | QuietOrderMin | 101 | Eval diff min |
| | QuietOrderMax | 192 | Eval diff max |
| | QuietOrderMult | 153 | Eval diff multiplier |
| | NMOrderingMH | 419 | Main history weight |
| | NMOrderingSS1 | 491 | Continuation[ply-1] weight |
| | NMOrderingSS2 | 258 | Continuation[ply-2] weight |
| | NMOrderingSS4 | 255 | Continuation[ply-4] weight |
| | NMOrderingSS6 | 246 | Continuation[ply-6] weight |
| | QSOrderingMH | 498 | QS main history weight |
| | QSOrderingSS1 | 500 | QS continuation[ply-1] |
| | QSOrderingSS2 | 258 | QS continuation[ply-2] |
| | QSOrderingSS4 | 247 | QS continuation[ply-4] |
| | QSOrderingSS6 | 261 | QS continuation[ply-6] |
| | OrderingEnPriseMult | 520 | Threat bonus multiplier |
| | CheckBonus | 9816 | Check move bonus |
| | MVVMult | 363 | Capture value multiplier |
| **Correction** | PawnCorrCoeff | 129 | Pawn correction weight |
| | NonPawnCorrCoeff | 70 | Non-pawn correction weight |
| | CorrDivisor | 3136 | Correction scale |
| **IIR** | IIRMinDepth | 3 | Internal iterative reduction |

**~100 tunable parameters** total. Horsie is moderately tuned; not as parameter-heavy as some modern engines but substantial.

**Coda comparison**: Our current search has ~60 tunable params (lines 45-160 of `src/search.rs`). Horsie's parameter count is in the same ballpark. The structure is similar (depth-based penalties, history scaling) but the specific values differ due to engine architecture differences (NNUE, LMR base formula, etc.).

---

## 5. Miscellaneous / Unique Features

### 5.1 Killer Move Optimization

**Killer moves used in move scoring but not updated directly**. Instead, killer bonus comes from history (line 1148):
```cpp
if (m == ss->KillerMove):
    score = INT32_MAX - 1000000
```

Killers are set on beta-cutoff (`search.cpp:971`) but only in quiet move updates, not explicitly maintained.

**Coda comparison**: We use killers more explicitly. Horsie's approach relies on history convergence over time.

### 5.2 Cuckoo Hashing (Repetition Detection)

`cuckoo.cpp`: Simple cuckoo hash table for cycle detection (same game-state repetitions within search tree).

Used in `pos.HasCycle(ply)` checks (`search.cpp:220, 780`).

**Coda comparison**: We use TT-based cycle detection. Horsie's dedicated cuckoo table is faster for repeated checks but uses extra memory.

### 5.3 Network Loading & ZSTD Compression

Network is embedded in binary via `incbin.h` (compile-time inclusion).

Network file: `net-015-2048x16x32-132qb-z-p0.bin` = **ZSTD-compressed** binary.

At engine startup, network is decompressed to RAM (`nn.cpp:54-70`).

**Note on network naming**: "2048x16x32" = L1x L2x L3 sizes. "132qb" = L1_QUANT=132, OUTPUT_BUCKETS. "z" = compressed. Network 015 is a specific trained version.

**Coda comparison**: We load nets from files at runtime. Horsie's embedded, pre-compressed approach:
- Pro: Faster startup, no file dependency, portable binary
- Con: Must recompile for new nets, larger binary

### 5.4 WDL (Win/Draw/Loss) Output

`wdl.cpp`: Estimates game outcome probabilities from eval and material count.

Output in UCI info: `wdl <wins> <draws> <losses>` (out of 1000).

Uses material-based model: `WDL::MaterialModel(score, material)` returns win/loss estimates.

**Coda comparison**: We have WDL. Standard feature for modern engines.

### 5.5 Thread Pool & Shared TT

**Message-passing thread pool** (`threadpool.cpp`):
- Pre-spawned workers waiting on message channel
- Main thread sends search closures to workers
- Lockless shared TT (standard 3-entry clusters)

Per-thread state: board, accumulators, history tables, time manager.

**Coda comparison**: We use similar TT sharing. Horsie's thread pool design is cleaner via message channels.

### 5.6 GHI Mitigation (Graph History Interaction)

Hash function uses halfmove clock bucketing (`zobrist.cpp`):
```cpp
hash ^= ZOBRIST.halfmove_clock[(halfmove_clock.saturating_sub(8) / 8).min(15)]
```

Hashes differ every 8 plies of halfmove clock, preventing GHI where different move histories hash identically.

**Coda comparison**: We don't have this. It's a correctness improvement worth adding.

### 5.7 SEE (Static Exchange Evaluation)

Standard alpha-beta SEE with piece value array (`board/see.rs` conceptually):
- Iterative attackers and defenders
- Accounts for piece gains/losses in sequences
- Used in SEE pruning and capture ordering

**No pin awareness** (unlike Reckless). Horsie's SEE is standard.

---

## 6. Summary: Ideas to Port to Coda

### High-Priority (likely 10-30 Elo each)

1. **Eval-difference-based quiet history update** (`search.cpp:326-334`)
   - Update opponent's main history based on how well their previous move performed
   - Cost: ~5 lines of code, ~1 new parameter
   - Should transfer well cross-engine

2. **Continuation history at ply-6** (not just 1,2,4)
   - Horsie uses offsets {1,2,4,6}; Coda uses {1,2,4}
   - May capture longer-term move correlations
   - Low risk, tuning needed

3. **GHI mitigation in hash** (`zobrist.cpp` halfmove bucketing)
   - Prevent graph history interaction where different histories hash identically
   - Cost: ~1 XOR operation per hash
   - Correctness improvement, no Elo cost, may prevent rare search bugs

### Medium-Priority (5-15 Elo, moderate effort)

4. **Ply history table** (low-depth move ordering)
   - Horsie updates moves at ply < 4 in separate table
   - Helps with move ordering early in search
   - Tuning cost: 1 new parameter (clamping)

5. **Post-LMR adaptive depth adjustment**
   - If LMR re-search much better: extend; if barely better: reduce
   - Horsie's simple 2-factor (deeper/shallower) is elegant
   - Cost: ~3 lines in LMR logic

6. **Quiet order min/max clamping**
   - Horsie clamps eval-diff bonuses to [-101, 192]
   - Prevents extreme history biasing from single positions
   - Parameters: 2 new tunable bounds

### Lower-Priority (interesting but uncertain Elo)

7. **Separate LMR bonus/penalty history**
   - Horsie has dedicated history for re-search bonuses (LMRBonus vs StatBonus)
   - Allows more precise calibration of depth adjustments
   - Cost: 2 new history tables + parameters

8. **Time management stability coefficients**
   - Horsie's StabilityCoefficients array (diminishing returns on repeated best move)
   - More nuanced than simple "if stable, stop early"
   - Tuning-heavy, likely not large Elo gain

9. **Cuckoo hash for repetition detection**
   - Faster than TT-based cycle checks
   - Not required, nice-to-have optimization

### Not Worth Porting (engine-specific design)

10. **Larger L1 (2048 vs 768)** — Horsie's architecture tradeoff for no threat features
    - Requires retraining entire net in Bullet
    - Our threat accumulator is architecturally different

11. **Implicit threat learning vs explicit threat features** — fundamental design choice
    - Horsie shows it's viable but requires more L1 capacity
    - We've committed to explicit threats; no benefit to change

---

## 7. Coda Comparison: Side-by-Side

### NNUE

| Aspect | Horsie | Coda (v9) | Winner |
|--------|--------|-----------|--------|
| FT size | 768 (×14 buckets) | 768 (×10 buckets) | Tie (Coda slightly simpler) |
| L1 width | 2048 | 768 | Coda (faster inference) |
| Threat features | None (implicit) | 66,864 explicit | Coda (more information) |
| Activation | Pairwise symmetric | Pairwise asymmetric | Tie |
| L1 type | int8 weights | int8 weights | Tie |
| L2/L3 | Float | Float | Tie |
| Sparse matmul | Yes (NNZ detection) | No (dense) | Horsie (faster) |
| Output scale | 400 | 380 | Horsie (slightly higher) |

### Search Parameters

| Aspect | Horsie | Coda | Winner |
|--------|--------|------|--------|
| LMR base | log(d)*log(m)/2.25 | Likely similar | Tie |
| LMR history offsets | 1,2,4,6 | 1,2,4 | Horsie (longer view) |
| NMP gates | Simple | Similar | Tie |
| SE gates | Similar | Similar | Tie |
| Eval adjustment | Halfmove + correction | Similar | Tie |
| Correction sources | 3 (pawn, 2x nonpawn) | 6 (more granular) | Coda |
| Time management | Multi-factor stability | Similar | Tie |
| Tunable params | ~100 | ~60 | Coda (simpler) |

### Move Ordering

| Aspect | Horsie | Coda | Winner |
|--------|--------|------|--------|
| History factorization | Single value | Single value | Tie |
| Threat-aware history | No | No (in current version) | Tie |
| Continuation offsets | 1,2,4,6 | 1,2,4 | Horsie |
| Killer moves | Set but not updated explicitly | Explicit updates | Coda |
| En-prise bonuses | Yes (threat-based) | Likely similar | Tie |

---

## 8. Final Notes

Horsie is a **clean, well-engineered C++ port** of the Lizard C# engine. It prioritizes **simplicity and clarity** over exotic optimizations. Key characteristics:

- **Simpler NNUE**: Single-stream PST, wider L1, no threat features. Works well with Horsie's 2048-wide L1 but limits inference speed.
- **Moderate search complexity**: Standard pruning techniques (NMP, RFP, SE, ProbCut) with reasonable tuning. No exotic gates.
- **Excellent move ordering**: Fine-grained continuation history (4 offsets), ply history, eval-based quiet bonuses.
- **Clean code**: C++20, clear separation of concerns, good use of templates.
- **Competitive strength**: 3575 CCRL 40/15 is solid; v1.1 tuning may push it higher.

For Coda:
- **Most valuable port**: Eval-difference quiet history bonus (simple, direct).
- **Highest impact**: Ply-6 continuation offset + adaptive post-LMR depth (tuning opportunity).
- **Correctness**: GHI hash mitigation (low-cost, no risk).
- **Avoid**: Large-scale architectural changes (threat vs no-threat is a fundamental design choice).

