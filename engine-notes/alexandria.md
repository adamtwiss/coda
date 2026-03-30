# Alexandria Chess Engine - Deep Review

Source: https://github.com/PGG106/Alexandria (v9.0.3)
Author: PGG106 (Andrea)
Language: C++ (91.7%)
CCRL Ranking: #6
NNUE: (768x16 -> 1536)x2 -> 16 -> 32 -> 1x8 (SCReLU + dual activation, FinnyTable)

---

## 1. NNUE Architecture

### Network Topology
- **Input**: 768 features (6 piece types x 2 colors x 64 squares) x 16 king buckets = 12,288 virtual inputs
- **Feature Transformer**: 768x16 -> 1536 (int16 weights, FT_QUANT=255)
- **Hidden Layer 1**: 1536 -> 16 (int8 quantized, L1_QUANT=64) -- but with dual activation (see below), effective input is 32
- **Hidden Layer 2**: 32 -> 32 (float)
- **Hidden Layer 3**: 32 -> 1 (float)
- **Output Buckets**: 8 (material-based)
- **NET_SCALE**: 362

### King Buckets (16 buckets, mirrored)
```
 0  1  2  3  3  2  1  0
 4  5  6  7  7  6  5  4
 8  9 10 11 11 10  9  8
 8  9 10 11 11 10  9  8
12 12 13 13 13 13 12 12
12 12 13 13 13 13 12 12
14 14 15 15 15 15 14 14
14 14 15 15 15 15 14 14
```
- Mirrored horizontally when king is on files e-h (`flip = get_file[kingSq] > 3`)
- Perspective: white king square flipped vertically (`kingSq ^ 56`), black king square used directly
- **GoChess comparison**: Our 16 buckets are similar. Their bucket map gives more granularity on ranks 1-2 (8 distinct buckets) vs ranks 3-8 (8 buckets). Compare to our layout.

### Dual Activation (Key Innovation)
After L1, instead of a single activation function, Alexandria produces TWO outputs per neuron:
1. **Linear**: `clamp(z, 0, 1)` (standard CReLU capped at 1)
2. **Squared**: `clamp(z*z, 0, 1)` (squared activation)

This doubles the effective L2 input size from 16 to 32 (`EFFECTIVE_L2_SIZE = 16 * (1 + DUAL_ACTIVATION) = 32`).

**GoChess comparison**: We use CReLU only. Dual activation is a cheap way to add expressiveness without increasing L1 width. This is the key architectural difference vs our planned SCReLU migration.

### Pairwise Multiplication (Feature Transformer)
The FT activation uses pairwise multiplication:
- Split the 1536-element accumulator into two halves (768 each)
- Clip first half to [0, 255], second half to [0, 255] (but second half only clips upper bound, not lower)
- Multiply pairs: `output[i] = clipped0[i] * clipped1[i] >> 10`
- Output is uint8, feeding into int8 L1 weights

**Shift**: `FT_SHIFT = 10` (vs our approach using `/255`)
**Weight clipping**: 1.98 -- L1 weights clipped to [-1.98*L1_QUANT, 1.98*L1_QUANT] = [-127, 127]

### Quantization Scheme
| Layer | Weight Type | Quantization | Notes |
|-------|-----------|-------------|-------|
| FT | int16 | FT_QUANT=255 | Accumulator values in [-32768, 32767] |
| L1 | int8 | L1_QUANT=64 | Sparse multiplication via NNZ tracking |
| L2 | float | None | Small enough that float is fine |
| L3 | float | None | Single output per bucket |

### FinnyTable (Accumulator Cache)
Per-king-bucket cache of accumulator state + occupancy bitmaps:
```cpp
struct FinnyTableEntry {
    PovAccumulator accumCache;     // int16[1536]
    Bitboard occupancies[12] = {}; // per-piece bitboards
};
// Indexed by [side][king_bucket][flip]
using FinnyTable = array<array<array<FinnyTableEntry, 2>, INPUT_BUCKETS>, 2>;
```

On each evaluation, instead of full recompute on king bucket change:
1. Compare cached occupancies vs actual bitboards per piece type
2. Compute add/remove delta lists
3. Apply incremental updates (add/sub/move) to cached accumulator

This avoids full 12288-element recomputation even when the king changes buckets. Very efficient -- only pays the cost proportional to pieces that changed since the cached state.

**GoChess comparison**: **(UPDATE 2026-03-21: GoChess now has Finny tables for v5 NNUE accumulator refresh, merged.)** Previously we used lazy accumulators with delta tracking; FinnyTable persists across king moves and search tree branches.

### Factoriser
The FT has a separate `Factoriser[768 * 1536]` matrix that is added to all king bucket weights during quantization:
```cpp
float w = FTWeights[bucket_offset + i] + Factoriser[i];
```
This is a shared base that all buckets inherit, reducing the effective parameter count while still allowing per-bucket specialization. Applied at quantization time, not inference time.

### Output Bucket Formula
```cpp
int outputBucket = min((63 - pieceCount) * (32 - pieceCount) / 225, 7);
```
Non-linear: finer granularity in the middlegame (20-30 pieces), coarser in endgame.

| Pieces | Bucket |
|--------|--------|
| 32 | 0 |
| 30-31 | 1-2 |
| 27-29 | 3-4 |
| 23-26 | 5-6 |
| <=22 | 7 |

**GoChess comparison**: We use simple `min(pieceCount/4, 7)` which is linear. Their formula puts more emphasis on middlegame transitions.

### L1 Sparse Multiplication
Alexandria tracks non-zero (NNZ) indices in the pairwise output and uses sparse multiplication for L1:
- After pairwise activation, track which 4-byte chunks have any non-zero bytes via NNZ mask
- L1 multiplication only processes non-zero chunks
- Uses VPMADDUBSW (AVX2) / VNNI (AVX-512) for int8*uint8 multiplication
- Unrolled by 2 for additional throughput

**GoChess comparison**: We don't do NNZ-sparse L1. With 1536->16, sparsity exploitation is very valuable since many FT outputs will be zero.

### SIMD Support
- **AVX-512**: Full support with VNNI dpbusd intrinsics
- **AVX2**: Fallback via maddubs emulation
- **No ARM/NEON support** (unlike GoChess which has both AVX2 and NEON)
- Unified abstraction layer in `simd.h` with inline wrappers

---

## 2. Evaluation

### Static Eval Pipeline
```cpp
int eval = NNUE::output(pos, FinnyPointer);  // Raw NNUE output * NET_SCALE(362)
eval = clamp(eval, -MATE_FOUND+1, MATE_FOUND-1);
```

### adjustEval (Post-Processing)
```cpp
int adjustEval(pos, correction, rawEval) {
    eval = rawEval * (200 - fiftyMoveCounter) / 200;  // 50-move decay
    eval = ScaleMaterial(pos, eval);                    // Material scaling
    eval += correction;                                 // Correction history
    return clamp(eval, -MATE_FOUND+1, MATE_FOUND-1);
}
```

### Material Scaling
```cpp
int materialValue = pawns*100 + knights*422 + bishops*422 + rooks*642 + queens*1015;
int scale = (22400 + materialValue) / 32;
eval = eval * scale / 1024;
```
This scales eval upward when more material is on the board, downward in endgames. With all pieces: scale ~ (22400+8280)/32 = 959/1024 ~ 0.94x. With just kings: scale ~ 22400/32 = 700/1024 ~ 0.68x.

**GoChess comparison**: We don't have material scaling of NNUE output. This is a cheap post-processing step that could help with endgame evaluation accuracy.

### 50-Move Rule Decay
```cpp
eval = eval * (200 - fiftyMoveCounter) / 200;
```
Scales eval toward zero as 50-move counter increases. At halfmove=100, eval becomes 50% of raw. At halfmove=0, no scaling.

**GoChess comparison**: We tested this as "50-move eval scaling" and got H0 (-3.0 Elo). Alexandria's is integrated into the adjustEval pipeline alongside material scaling -- the combination may be what makes it work.

---

## 3. Search Architecture

### Iterative Deepening
- Standard ID from depth 1 to max depth
- Average score tracked: `averageScore = (averageScore + score) / 2`
- Root history table cleared before each `SearchPosition` call
- `seldepth` reset between ID iterations

### Aspiration Windows
- **Delta**: 12 (vs our 15)
- **Enabled at**: depth >= 3 (vs our depth >= 5)
- **Fail-low**: Contract beta toward alpha: `beta = (alpha + beta) / 2`, widen alpha by delta
- **Fail-high**: Widen beta by delta, **reduce depth by 1** (`depth = max(depth-1, 1)`)
- **Delta growth**: `delta *= 1.44` (vs our 1.5x)
- **No full-width fallback** -- keeps widening until resolution

**GoChess comparison**: Tighter initial delta (12 vs 15), earlier activation (depth 3 vs 5). **(UPDATE 2026-03-21: GoChess now has asymmetric aspiration contraction: fail-low (3a+5b)/8, fail-high (5a+3b)/8, delta=15, growth 1.5x.)** Fail-high depth reduction is something we tested and got -353.8 Elo, but that was an implementation bug (modified outer loop depth). Worth retesting with correct implementation.

### Draw Detection
- 2-fold repetition within search tree, 3-fold including pre-root
- 50-move rule with checkmate verification
- **Upcoming repetition detection** via Cuckoo hashing (from Stockfish) -- detects when a position *could* repeat via a reversible move without actually searching the move

**GoChess comparison**: We don't have upcoming repetition detection (Cuckoo). This prevents the engine from entering clearly drawn lines earlier, saving search time.

### Mate Distance Pruning
```cpp
alpha = max(alpha, -MATE_SCORE + ply);
beta = min(beta, MATE_SCORE - ply - 1);
if (alpha >= beta) return alpha;
```

**GoChess comparison**: We don't have this. Trivial to implement, 3 lines.

---

## 4. Pruning Techniques

### Reverse Futility Pruning (RFP)
- **Depth**: < 10 (vs our <= 7)
- **Margin**: `75*depth - 61*improving - 76*canIIR` (tuned parameters)
  - At depth 5, not improving, no IIR: 375
  - At depth 5, improving: 314
  - At depth 5, improving + IIR: 238
- **Guard**: `ttMove == NOMOVE || isTactical(ttMove)` -- skips RFP when TT has a quiet best move
- **Return value**: `(eval - margin + beta) / 2` -- blended return (our dampening pattern!)
- **canIIR** = `depth >= 4 && ttBound == HFNONE` -- no TT data means less confident, reduce margin

**GoChess comparison**:
- Our margin is 80+d*80 (merged at +33.6 Elo). Their 75*d is similar for shallow depths but scales differently.
- Their `improving` offset (61) and IIR offset (76) are tuned separately.
- Blended return `(eval-margin+beta)/2` is exactly our dampening pattern. We tested RFP blending and got -16.7 Elo, but that was with our old margins. Worth retesting with current margins.
- **ttMove guard** (skip RFP when TT has quiet move) -- we tested this and it was -17.4 Elo. Alexandria has it gated differently: `ttMove == NOMOVE || isTactical(ttMove)`.
- **IIR margin reduction** is novel -- reduces margin when we have no TT info (higher uncertainty). This is a smart idea.

### Null Move Pruning (NMP)
- **Conditions**: `eval >= staticEval && eval >= beta && staticEval >= beta - 28*depth + 204`
  - Extra condition: `staticEval >= beta - 28*depth + 204` prevents NMP when staticEval is too far below beta
- **Reduction**: `R = 4 + depth/3 + min((eval-beta)/221, 3)` (base 4 vs our 3)
- **badNode reduction**: `-badNode` added to reduction (reduces R by 1 when no TT data)
- **Verification**: At depth >= 15, do verification search with `nmpPlies = ply + (depth-R)*2/3`
- **Mate clamp**: Returns beta if nmpScore is a mate

**GoChess comparison**:
- Base reduction 4 vs our 3 -- more aggressive. The `28*depth + 204` condition compensates.
- Eval divisor 221 matches ours.
- Extra condition `staticEval >= beta - 28*depth + 204` is important -- prevents NMP when eval is barely above beta but staticEval is poor.
- `badNode` reduction is novel -- reduce NMP R by 1 when no TT info.

### Razoring
- **Depth**: <= 5
- **Margin**: `258 * depth`
- Drops to QSearch if `eval + 258*depth < alpha`
- Returns razorScore if <= alpha

**GoChess comparison**: Our razoring uses 400+d*100. Their 258*d is tighter for deeper depths (depth 5: 1290 vs our 900). Different tradeoff.

### ProbCut
- **Depth**: > 4
- **Beta**: `beta + 287 - 54*improving`
- **Guard**: TT score is NONE or lower-bound, and TT depth < depth-3 or ttScore >= pcBeta
- **Two-stage**: First runs QSearch with `-pcBeta`, then if QSearch passes, runs full `Negamax` at `depth-4`
- **SEE threshold**: `pcBeta - staticEval` for capture filtering

**GoChess comparison**:
- Our ProbCut margin is 170 (merged at +10 Elo). Their base 287 is much larger, but with the improving offset of 54, non-improving is 287, improving is 233.
- **Two-stage QSearch pre-filter** is exactly what we identified as idea #6 in SUMMARY.md. This saves significant nodes by not running the expensive depth-4 search when QSearch already fails to confirm.

### Futility Pruning (Quiet Moves)
- **Depth**: `lmrDepth < 13`
- **Margin**: `staticEval + 232 + 118*lmrDepth`
  - depth 1: 350, depth 5: 822
- **Guard**: Not in check, quiet move
- **Best score update**: If `bestScore <= futilityValue`, updates bestScore to futilityValue (avoids returning -infinity)
- Sets `skipQuiets = true` after triggering

**GoChess comparison**: Our futility is `80 + d*80` (depth 1: 160, depth 5: 480). Their margin is significantly larger, but note they use `lmrDepth` (which accounts for LMR reduction) rather than raw depth, so effective depth is often lower.

### Late Move Pruning (LMP)
```cpp
lmp_margin[depth][0] = 1.5 + 0.5 * depth^2  // Not improving
lmp_margin[depth][1] = 3.0 + 1.0 * depth^2  // Improving
```
| Depth | Not Improving | Improving |
|-------|--------------|-----------|
| 1 | 2 | 4 |
| 2 | 3 | 7 |
| 3 | 6 | 12 |
| 4 | 9 | 19 |
| 5 | 14 | 28 |

**GoChess comparison**: Our `3+d^2` formula is similar to their improving formula. Their non-improving formula is tighter.

### History Pruning
```cpp
if (isQuiet && moveHistory < -3753 * depth) skipQuiets = true;
```

**GoChess comparison**: Different formula but similar concept.

### SEE Pruning
- **Quiet moves**: `seeQuietMargin * lmrDepth = -98 * lmrDepth`
- **Noisy moves**: `seeNoisyMargin * lmrDepth^2 = -27 * lmrDepth^2`

**GoChess comparison**: Our single `-20*depth^2` is different. Their separate quiet/noisy thresholds with lmrDepth is more nuanced. Quiet: linear scaling, Noisy: quadratic scaling.

### Hindsight Reduction
```cpp
if (depth >= 2 && (ss-1)->reduction >= 1
    && (ss-1)->staticEval != SCORE_NONE
    && ss->staticEval + (ss-1)->staticEval >= 155)
    depth--;
```
If the parent move was reduced (LMR) and the sum of parent and current static evals is >= 155 (both sides think position is good = quiet position), reduce depth by 1.

**GoChess comparison**: We don't have this. Cheap to implement (2 lines). The insight is that positions where both sides have positive eval are likely quiet and can be searched less deeply.

---

## 5. Extensions

### Singular Extensions (CRITICAL -- our SE is catastrophically failing)
**Conditions**:
```
!rootNode && depth >= 6 && move == ttMove && !excludedMove
&& (ttBound & LOWER) && !isDecisive(ttScore) && ttDepth >= depth - 3
```

**Singular beta**: `ttScore - depth*5/8 - depth*(ttPv && !pvNode)`
- PV-aware: when TT says we were on PV but current node is not PV, reduce margin further (more likely to extend)

**Singular depth**: `(depth - 1) / 2`

**Extension logic**:
```cpp
singularScore = Negamax(singularBeta - 1, singularBeta, singularDepth, cutNode, ...);

if (singularScore < singularBeta) {
    extension = 1;
    // Double extension
    if (!pvNode && singularScore < singularBeta - 10) {
        extension = 2 + (singularScore < singularBeta - 75);  // Triple at -75
        depth += (depth < 10);  // Extra depth boost at shallow depths
    }
}
// Multi-cut: if singular search itself beats beta, return immediately
else if (singularScore >= beta && !isDecisive(singularScore))
    return singularScore;
// Negative extension: TT score >= beta but move not singular
else if (ttScore >= beta)
    extension = -2;
// Negative extension: cut node, not singular, not failing high
else if (cutNode)
    extension = -2;
```

**Extension limit**: `ply * 2 < RootDepth * 5` (allows up to 2.5x root depth in extensions)

**Key differences from GoChess**:
1. **Depth threshold**: 6 vs our 10 -- MUCH more aggressive activation
2. **Singular beta margin**: `ttScore - depth*5/8` vs our `ttScore - depth*3` -- Alexandria's is tighter (smaller margin = harder to be singular = fewer extensions)
3. **PV-awareness**: Extra `depth*(ttPv && !pvNode)` reduces singular beta further on former PV nodes
4. **Double/triple extensions**: At -10 and -75 below singular beta
5. **Multi-cut shortcut**: Returns singularScore directly if >= beta (skips searching the TT move entirely)
6. **Negative extensions of -2** (not -1): Both for ttScore >= beta and for cut nodes
7. **Shallow depth boost**: `depth += (depth < 10)` when double-extending
8. **Extension limiter**: `ply * 2 < RootDepth * 5` prevents explosive tree growth

**Our SE problem diagnosis**: Our SE is -58 to -85 Elo. Possible causes:
- Our depth threshold (10) means we rarely trigger SE, and when we do, the margin (depth*3 = 30 at depth 10) may be too aggressive
- We lack multi-cut (the `singularScore >= beta` return), which is a powerful pruning shortcut
- We lack negative extensions, which help balance the tree growth from positive extensions
- We may lack the extension limiter (`ply * 2 < RootDepth * 5`)
- Our singular depth formula may differ

---

## 6. LMR (Late Move Reductions)

### Base Reduction Tables
```cpp
reductions[noisy][depth][move] = base + log(depth) * log(move) / divisor
```
- **Quiet**: base = 1.07, divisor = 2.27
- **Noisy**: base = -0.36, divisor = 2.47

### Quiet Move Adjustments
| Condition | Adjustment |
|-----------|-----------|
| Cut node | +2 |
| Not improving | +1 |
| Killer or counter move | -1 |
| Gives check | -1 |
| Former PV (ttPv) | -1 - cutNode |
| High complexity (>50) | -1 |
| History score | -moveHistory / 8049 |

### Noisy Move Adjustments
| Condition | Adjustment |
|-----------|-----------|
| Cut node | +2 |
| Not improving | +1 |
| Gives check | -1 |
| Capture history | -moveHistory / 5922 |

### Clamping
```cpp
int reducedDepth = max(1, min(newDepth - depthReduction, newDepth)) + pvNode;
```
- Minimum reduced depth is 1 (never drop to QSearch)
- Maximum is newDepth (extensions limited to +1 in LMR)
- PV nodes get +1 bonus on reduced depth

### DoDeeper/DoShallower (Post-LMR Re-search)
```cpp
const bool doDeeperSearch = score > (bestScore + 77 + 2*newDepth);
const bool doShallowerSearch = score < (bestScore + newDepth);
newDepth += doDeeperSearch - doShallowerSearch;
```
After the reduced search beats alpha:
- If score is very high (>bestScore+77+2*newDepth), search deeper
- If score is barely above bestScore, search shallower
- Then re-search at adjusted depth

**Plus**: Bonus continuation history update based on whether re-search beat alpha.

**GoChess comparison**:
- Cut node +2: We have this from Weiss/Obsidian/BlackMarlin.
- Complexity adjustment: Uses `abs(eval - rawEval) / abs(eval) * 100`. We identified this as idea #27.
- History divisor 8049 vs our 5000 -- less aggressive history-based reduction.
- Noisy history divisor 5922 -- we don't have separate noisy LMR history.
- DoDeeper margin 77 vs the various values we tested. Our test got -13.7 Elo.
- PV node +1 on reduced depth is interesting.

---

## 7. Move Ordering

### Stages
1. **TT Move** (PICK_TT)
2. **Generate Noisy** (GEN_NOISY)
3. **Good Captures** (PICK_GOOD_NOISY) -- SEE filtered with dynamic threshold
4. **Killer Move** (PICK_KILLER) -- single killer per ply
5. **Counter Move** (PICK_COUNTER)
6. **Generate Quiets** (GEN_QUIETS)
7. **Pick Quiets** (PICK_QUIETS) -- partial insertion sort
8. **Bad Captures** (PICK_BAD_NOISY) -- captures that failed SEE

### Capture Scoring
```cpp
score = SEEValue[capturedPiece] * 16 + GetCapthistScore(pos, sd, move);
```
MVV scaled by 16, plus capture history.

### Good Capture SEE Threshold
```cpp
SEEThreshold = -score / 32 + 236;
```
Dynamic: higher-scored captures get a tighter SEE threshold. A capture scored 0 needs SEE >= 236. A capture scored 7680 (queen capture + max capthist) needs SEE >= -4.

### Quiet Scoring
```cpp
score = HH + CH(ply-1) + CH(ply-2) + CH(ply-4) + CH(ply-6) + 4*RH (if root)
```

### History Tables
| Table | Size | Max | Index |
|-------|------|-----|-------|
| Butterfly (HH) | [2][4096] | 8192 | side, from*64+to |
| Root History (RH) | [2][4096] | 8192 | side, from*64+to |
| Capture History | [768][6] | 16384 | piece*64+to, captured_type |
| Continuation History | [768][768] | 16384 | (ss-offset).piece*64+to, piece*64+to |
| Counter Moves | [4096] | N/A | from*64+to of previous move |

### Continuation History Plies
Written to: 1, 2, 4, 6
Read from: 1, 2, 4, 6

**GoChess comparison**: We read/write plies 1 and 2 only. Plies 4 and 6 are in our testing queue.

### Root History
Dedicated history table for root moves, weighted 4x in scoring. Cleared before each search. Gets separate bonus/malus tuning.

**GoChess comparison**: We don't have this. Simple to implement, provides better move ordering at root.

### Opponent History Update (Eval History)
```cpp
if (!inCheck && (ss-1)->staticEval != SCORE_NONE && isQuiet((ss-1)->move)) {
    int bonus = clamp(-10 * ((ss-1)->staticEval + ss->staticEval), -1830, 1427) + 624;
    updateOppHHScore(pos, sd, (ss-1)->move, bonus);
}
```
After computing static eval, if the opponent's last move was quiet, update THEIR butterfly history based on the eval change. If both sides think the position is bad for the side that moved, penalize that move.

**GoChess comparison**: This is the "Eval History / Opponent Move Quality Feedback" idea #23 in our SUMMARY.md. Alexandria's formula is: bonus when opponent's move caused mutual dissatisfaction, penalty when it improved things.

### History Bonus/Malus (Tuned Separately)
Each history type has independent bonus/malus formulas:
```
bonus(depth) = min(mul*depth + offset, max)
```

| History | Bonus mul/offset/max | Malus mul/offset/max |
|---------|---------------------|---------------------|
| HH | 333/159/2910 | 398/139/452 |
| Capture | 349/-76/2296 | 310/-88/1306 |
| ContHist | 159/-135/2538 | 401/114/806 |
| Root | 225/165/1780 | 402/75/892 |

Notable: **HH malus max is only 452** vs bonus max 2910 -- heavily asymmetric. Good moves get strong positive reinforcement, bad moves get only mild penalty.

**GoChess comparison**: We use a single bonus/malus formula for all history types. Asymmetric bonuses could help.

### TT Cutoff Continuation History Malus
```cpp
if (ttMove && ttScore >= beta && (ss-1)->moveCount < 4 && isQuiet((ss-1)->move)) {
    updateCHScore((ss-1), (ss-1)->move, -min(155*depth, 385));
}
```
When we get a TT cutoff, penalize the opponent's last quiet move in continuation history if they've only tried a few moves. This means "the opponent's move led to a position where we already have a good result cached."

**GoChess comparison**: Novel idea we don't have. Uses TT cutoff information to improve move ordering.

---

## 8. Correction History

### 4 Tables
1. **Pawn correction**: `pawnCorrHist[side][pawnKey % 32768]`
2. **White non-pawn correction**: `whiteNonPawnCorrHist[side][whiteNonPawnKey % 32768]`
3. **Black non-pawn correction**: `blackNonPawnCorrHist[side][blackNonPawnKey % 32768]`
4. **Continuation correction**: `contCorrHist[side][pieceTypeTo(ss-1)][pieceTypeTo(ss-2)]`

### Non-Pawn Keys
Separate Zobrist keys for white and black non-pawn pieces, computed by XORing piece-square keys for all non-pawn pieces of each color. Updated incrementally in MakeMove.

### Adjustment Formula
```cpp
adjustment = 29 * pawnCorr + 34 * whiteNonPawnCorr + 34 * blackNonPawnCorr + 26 * contCorr;
adjustment /= 256;
```
Weighted sum with tuned weights, then divided by 256 (CORRHIST_GRAIN).

### Update
```cpp
int bonus = clamp(diff * depth / 8, -CORRHIST_MAX/4, CORRHIST_MAX/4);
// Standard gravity scaling per table entry
```
Where diff = bestScore - staticEval. CORRHIST_MAX = 1024.

### Conditional Update
```cpp
if (!inCheck && (!bestMove || !isTactical(bestMove))
    && !(bound == LOWER && bestScore <= staticEval)
    && !(bound == UPPER && bestScore >= staticEval))
    updateCorrHistScore(...)
```
Only update when the correction is meaningful: not when the bound direction already agrees with static eval vs bestScore.

**GoChess comparison**: We have only pawn correction history. Their 4-table approach with proper Zobrist-based non-pawn keys and continuation correction is exactly what SUMMARY.md idea #7 describes. Our previous XOR-of-bitboards approach failed (-11.8 Elo). Alexandria shows the correct implementation using per-color Zobrist non-pawn keys.

---

## 9. Transposition Table

### Structure
- **10 bytes per entry**: move(2) + score(2) + eval(2) + ttKey(2) + depth(1) + ageBoundPV(1)
- **3 entries per bucket** (32-byte aligned buckets with 2 bytes padding)
- **Key verification**: 16-bit TTKey (upper bits of Zobrist)
- **Age/Bound/PV packed**: lower 2 bits = bound, bit 2 = PV flag, upper 5 bits = age

### Replacement Policy
1. Match on key16 -> replace that entry
2. Otherwise, find lowest-priority entry: `depth - (AGE_DELTA) * 4`
3. Overwrite conditions (SF-derived):
   - Exact bound always overwrites
   - Different key always overwrites
   - Deeper by 5 + 2*pv overwrites
   - Different age overwrites

### TT Cutoff Conditions
```cpp
!pvNode && ttScore != SCORE_NONE && ttDepth >= depth
&& cutNode == (ttScore >= beta)    // <-- KEY CONDITION
&& fiftyMoveCounter < 90
&& (ttBound & (ttScore >= beta ? LOWER : UPPER))
```

The `cutNode == (ttScore >= beta)` condition is interesting: only allow TT cutoffs when the expected node type matches. At cut nodes, only accept fail-highs; at non-cut nodes, only accept fail-lows.

**GoChess comparison**: We don't have the `cutNode == (ttScore >= beta)` guard. This is a precision improvement that prevents using stale TT data that contradicts the expected node behavior.

### Huge Pages
Linux: aligned to 2MB, uses `madvise(MADV_HUGEPAGE)`. Other platforms: 4KB alignment.

---

## 10. Time Management

### Base Allocation
```
optScale = min(25/1000, 200/1000 * time / timeLeft)
optime = optScale * timeLeft
maxtime = 0.76 * time - overhead
```
For cyclic TC: `min(0.90/movesToGo, 0.88 * time / timeLeft)`

### Node-Based Scaling
```cpp
bestMoveNodesFraction = nodeSpentTable[bestmove] / totalNodes;
nodeScalingFactor = (1.53 - bestMoveNodesFraction) * 1.74;
```
When best move uses a small fraction of nodes (unstable), spend more time. When it dominates (stable), spend less.

### Best Move Stability Scaling
5-level stability factor (0-4), with tuned scales:
```
[2.38, 1.29, 1.07, 0.91, 0.71]
```
If best move has been stable for 4+ iterations, multiply time by 0.71 (save 29%).

### Eval Stability Scaling
5-level factor based on whether score is within +/- 10 of average:
```
[1.25, 1.15, 1.03, 0.92, 0.87]
```
Stable eval = less time, volatile eval = more time.

### Final Time Calculation
```cpp
stoptimeOpt = starttime + baseOpt * nodeScale * bestMoveScale * evalScale;
stoptimeOpt = min(stoptimeOpt, stoptimeMax);
```

### Time Check Frequency
`TimeOver` checks every 1024 nodes: `(nodes & 1023) == 1023`

**GoChess comparison**:
- **(UPDATE 2026-03-21: GoChess now has score-drop time extension with 2.0x/1.5x/1.2x tiered scaling, merged.)** Still lacks node-based and bestmove-stability scaling.
- Alexandria has THREE scaling factors (nodes, bestmove stability, eval stability) that multiply together.
- Node-based TM is idea #24 in our SUMMARY.md, estimated +5 to +15 Elo.
- The bestmove stability scaling alone is very high value -- a move that's been best for 4 iterations in a row should not get more time.

---

## 11. Lazy SMP

### Thread Management
- Main thread + N-1 helper threads
- Each thread gets own `ThreadData` (position copy, search data, key history, FinnyTable)
- All threads share the TT (no locks, no atomics visible -- relies on TTKey16 for collision detection)
- Helper threads start at depth 1, main thread manages time
- Main thread can stop helpers via `info.stopped` flag

### Thread Data Isolation
Per-thread: Position, SearchData (all history tables), SearchInfo, keyHistory, FinnyTable, RootDepth, nmpPlies

Shared: TT (global), pvTable (global, only written by main thread)

**GoChess comparison**: Similar to our approach. Key difference: their FinnyTable is per-thread (necessary since it's position-dependent).

---

## 12. Quiescence Search

### Stand-Pat with Beta Blending
```cpp
if (bestScore >= beta) {
    if (!isDecisive(beta) && !isDecisive(bestScore))
        return (bestScore + beta) / 2;
    return bestScore;
}
```
**Double blending**: Both at stand-pat AND at final return:
```cpp
if (bestScore >= beta && !isDecisive(bestScore) && !isDecisive(beta))
    bestScore = (bestScore + beta) / 2;
```

**GoChess comparison**: We have QS beta blending (merged at +4.9 Elo). Alexandria applies it at both stand-pat and final return.

### QS Futility
```cpp
futilityBase = staticEval + 268;
if (futilityBase <= alpha && !SEE(pos, move, 1))
    bestScore = max(futilityBase, bestScore);
    continue;
```
Combined eval-based + SEE gate: only prune moves that don't win material AND where eval + margin is below alpha.

### Upcoming Repetition in QS
Alexandria checks for upcoming repetition (game cycle) even in QSearch, which prevents QS from reporting misleadingly high scores in positions that could lead to draws.

---

## 13. Unique/Novel Techniques

### 1. Complexity Metric
```cpp
int complexity = abs(eval - rawEval) / abs(eval) * 100;
// If complexity > 50, reduce LMR by 1
```
Uses the ratio of correction magnitude to eval as a measure of position complexity. High complexity = uncertain eval = search deeper.

### 2. badNode Flag
```cpp
const bool badNode = depth >= 4 && ttBound == HFNONE;
```
Identifies nodes with no TT information at all. Used in:
- IIR: `depth -= 1` when badNode
- RFP: Extra 76 margin reduction (less aggressive pruning when uncertain)
- NMP: Reduces R by 1

This is a unifying concept: "when we have no TT data, we're more uncertain, so be more conservative with pruning but also use IIR to find information."

### 3. Eval-Based History Depth Bonus
```cpp
UpdateHistories(pos, sd, ss, depth + (eval <= alpha), bestMove, ...);
```
When eval was at or below alpha (the position was worse than expected), give a +1 depth bonus to history updates. This means surprising beta-cutoff moves get stronger history bonuses.

### 4. Dynamic SEE Threshold for Good Captures
```cpp
SEEThreshold = -score / 32 + 236;
```
Higher-valued captures need less SEE safety margin. A queen capture with good history only needs SEE > -4, while a low-scored capture needs SEE > 236.

### 5. Draw Score Randomization
```cpp
return (info->nodes & 2) - 1;  // Returns -1 or 1
```
On draws, returns a small random value based on node count parity. Prevents the engine from entering draw loops while maintaining deterministic behavior per position.

### 6. TT Cutoff Node-Type Guard
```cpp
cutNode == (ttScore >= beta)
```
Only accept TT cutoffs when the expected node type (cut/all) matches the TT score direction.

---

## 14. Summary: Key Differences from GoChess

### Things Alexandria Has That We DON'T (Prioritized by Impact)

1. **4-table correction history** (pawn + white non-pawn + black non-pawn + continuation) with proper Zobrist keys -- HIGH PRIORITY
2. ~~**FinnyTable NNUE accumulator cache**~~ -- **(UPDATE 2026-03-21: MERGED)**
3. **Node-based time management** with bestmove stability + eval stability scaling -- estimated +5-15 Elo
4. **Working singular extensions** with multi-cut and negative extensions -- our SE is broken, understanding their implementation is crucial
5. **Root history table** -- separate history for root moves, 4x weight
6. **Upcoming repetition detection** (Cuckoo hashing)
7. **Dual activation** in NNUE (linear + squared) -- relevant for our NNUE migration
8. **Hindsight reduction** -- 2 lines, cheap
9. **Mate distance pruning** -- 3 lines
10. **ProbCut QSearch pre-filter** (two-stage ProbCut)
11. **Opponent history update** (eval-based feedback)
12. **Continuation history plies 4, 6** (we only use 1, 2)
13. **IIR margin in RFP** (badNode flag reduces RFP margin)
14. **TT cutoff node-type guard** (`cutNode == (ttScore >= beta)`)
15. **Eval-based history depth bonus** (`depth + (eval <= alpha)`)
16. **TT cutoff continuation history malus** (penalize opponent's last move on TT cutoff)
17. **Material scaling of NNUE output**
18. **Asymmetric history bonus/malus** per table type
19. **NNZ-sparse L1 multiplication**
20. **Factoriser in FT** (shared base for all king buckets)

### Significantly Different Parameters
| Parameter | Alexandria | GoChess | Notes |
|-----------|-----------|---------|-------|
| Aspiration delta | 12 | 15 | Tighter |
| RFP depth | < 10 | <= 7 | Deeper |
| RFP margin | 75*d | 80+80*d | Different formula |
| NMP base R | 4 | 3 | More aggressive |
| SE depth threshold | >= 6 | >= 10 | Much more aggressive |
| SE margin | d*5/8 | d*3 | Tighter |
| LMR history divisor | 8049 | 5000 | Less reduction from history |
| LMR noisy divisor | 5922 | N/A | We don't split |
| SEE quiet margin | -98*lmrDepth | -20*d^2 | Different scaling |
| SEE noisy margin | -27*lmrDepth^2 | -20*d^2 | Different scaling |
| QS futility | 268 | 240 | Similar |
| NET_SCALE | 362 | 400 | Different NNUE scale |

### Critical Insight: Our Singular Extensions Problem
Alexandria's SE implementation differs from ours in at least 7 ways (see section 5). The most likely causes of our -58 to -85 Elo:
1. **Missing extension limiter** (`ply*2 < RootDepth*5`) -- without this, extensions can cause exponential tree growth
2. **Missing multi-cut** (`singularScore >= beta` return) -- this is a powerful pruning shortcut that balances the cost of the singular search
3. **Missing negative extensions** (-2 for ttScore >= beta and cut nodes) -- these balance positive extensions
4. **Too high depth threshold** (10 vs 6) -- fewer SE activations means less data to amortize the cost
5. **Too wide margin** (d*3 vs d*5/8) -- at depth 10, our margin is 30 vs their 6.25

The combination of (1) no limiter + (3) no negative extensions likely causes explosive tree growth when SE activates, dwarfing any benefit from the extensions themselves.
