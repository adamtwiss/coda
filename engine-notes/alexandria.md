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

**Coda comparison**: Coda also uses 16 king buckets with HalfKA features. Similar layout.

### Dual Activation (Key Innovation)
After L1, instead of a single activation function, Alexandria produces TWO outputs per neuron:
1. **Linear**: `clamp(z, 0, 1)` (standard CReLU capped at 1)
2. **Squared**: `clamp(z*z, 0, 1)` (squared activation)

This doubles the effective L2 input size from 16 to 32 (`EFFECTIVE_L2_SIZE = 16 * (1 + DUAL_ACTIVATION) = 32`).

**Coda comparison**: Coda's v7 uses SCReLU for hidden layers (squared activation). Dual activation (linear + squared in parallel) is a different approach that could be explored in Bullet training.

### Pairwise Multiplication (Feature Transformer)
The FT activation uses pairwise multiplication:
- Split the 1536-element accumulator into two halves (768 each)
- Clip first half to [0, 255], second half to [0, 255] (but second half only clips upper bound, not lower)
- Multiply pairs: `output[i] = clipped0[i] * clipped1[i] >> 10`
- Output is uint8, feeding into int8 L1 weights

**Shift**: `FT_SHIFT = 10` (vs Coda's `/255`)
**Weight clipping**: 1.98 -- L1 weights clipped to [-1.98*L1_QUANT, 1.98*L1_QUANT] = [-127, 127]

**IMPLEMENTED in Coda**: Coda supports pairwise (v6) with 768pw architecture, including AVX2 SIMD for pairwise CReLU pack.

### Quantization Scheme
| Layer | Weight Type | Quantization | Notes |
|-------|-----------|-------------|-------|
| FT | int16 | FT_QUANT=255 | Accumulator values in [-32768, 32767] |
| L1 | int8 | L1_QUANT=64 | Sparse multiplication via NNZ tracking |
| L2 | float | None | Small enough that float is fine |
| L3 | float | None | Single output per bucket |

**Coda comparison**: Coda uses QA=255 (accumulator), QB=64 (output weights). Same quantization scheme for FT and L1.

### FinnyTable (Accumulator Cache)
Per-king-bucket cache of accumulator state + occupancy bitmaps.

**IMPLEMENTED in Coda**: Coda has per-perspective, per-bucket Finny table cache. On king bucket change, diffs cached vs current bitboards (~5 delta ops vs ~30 full recompute).

### Factoriser
The FT has a separate `Factoriser[768 * 1536]` matrix that is added to all king bucket weights during quantization:
```cpp
float w = FTWeights[bucket_offset + i] + Factoriser[i];
```
This is a shared base that all buckets inherit, reducing the effective parameter count while still allowing per-bucket specialization. Applied at quantization time, not inference time.

**Coda comparison**: Not implemented. This is a training-side optimization that could be added to the Bullet training pipeline.

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

### L1 Sparse Multiplication
Alexandria tracks non-zero (NNZ) indices in the pairwise output and uses sparse multiplication for L1:
- After pairwise activation, track which 4-byte chunks have any non-zero bytes via NNZ mask
- L1 multiplication only processes non-zero chunks
- Uses VPMADDUBSW (AVX2) / VNNI (AVX-512) for int8*uint8 multiplication
- Unrolled by 2 for additional throughput

**IMPLEMENTED in Coda**: Coda has NNZ-sparse L1 via `SparseL1` UCI option (default true for v7). Uses `find_nnz_chunks` + `simd_l1_int8_dot_sparse`.

### SIMD Support
- **AVX-512**: Full support with VNNI dpbusd intrinsics
- **AVX2**: Fallback via maddubs emulation
- **No ARM/NEON support**
- Unified abstraction layer in `simd.h` with inline wrappers

**Coda comparison**: Coda has AVX2 and AVX-512 via `std::arch::x86_64` (runtime detected). No ARM/NEON either.

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
This scales eval upward when more material is on the board, downward in endgames. With all pieces: scale ~ 0.94x. With just kings: scale ~ 0.68x.

**Coda comparison**: Not implemented. Cheap post-processing step that could help with endgame evaluation accuracy. Worth testing.

### 50-Move Rule Decay
```cpp
eval = eval * (200 - fiftyMoveCounter) / 200;
```
Scales eval toward zero as 50-move counter increases.

**Coda comparison**: Not implemented. Was tested in the GoChess era and got H0 (-3.0 Elo). Alexandria's is integrated alongside material scaling -- the combination may be what makes it work.

---

## 3. Search Architecture

### Iterative Deepening
- Standard ID from depth 1 to max depth
- Average score tracked: `averageScore = (averageScore + score) / 2`
- Root history table cleared before each `SearchPosition` call
- `seldepth` reset between ID iterations

### Aspiration Windows
- **Delta**: 12 (vs Coda's 15)
- **Enabled at**: depth >= 3 (vs Coda's depth >= 4)
- **Fail-low**: Contract beta toward alpha: `beta = (alpha + beta) / 2`, widen alpha by delta
- **Fail-high**: Widen beta by delta, **reduce depth by 1** (`depth = max(depth-1, 1)`)
- **Delta growth**: `delta *= 1.44`
- **No full-width fallback** -- keeps widening until resolution

**Coda comparison**: Tighter initial delta and earlier activation. Fail-high depth reduction was tested in the GoChess era and got -353.8 Elo due to implementation bug. Worth retesting with correct implementation.

### Draw Detection
- 2-fold repetition within search tree, 3-fold including pre-root
- 50-move rule with checkmate verification
- **Upcoming repetition detection** via Cuckoo hashing (from Stockfish)

**IMPLEMENTED in Coda**: Coda has Cuckoo cycle detection (proactive repetition avoidance). Used in both main search and QSearch.

### Mate Distance Pruning
```cpp
alpha = max(alpha, -MATE_SCORE + ply);
beta = min(beta, MATE_SCORE - ply - 1);
if (alpha >= beta) return alpha;
```

**Coda comparison**: Not implemented. Trivial to add, 3 lines. Low-impact but free.

---

## 4. Pruning Techniques

### Reverse Futility Pruning (RFP)
- **Depth**: < 10 (vs Coda's <= 7)
- **Margin**: `75*depth - 61*improving - 76*canIIR` (tuned parameters)
  - At depth 5, not improving, no IIR: 375
  - At depth 5, improving: 314
  - At depth 5, improving + IIR: 238
- **Guard**: `ttMove == NOMOVE || isTactical(ttMove)` -- skips RFP when TT has a quiet best move
- **Return value**: `(eval - margin + beta) / 2` -- blended return
- **canIIR** = `depth >= 4 && ttBound == HFNONE` -- no TT data means less confident, reduce margin

**Coda comparison**: Coda uses margin improving?70*d:100*d. Key differences:
- Alexandria's deeper threshold (< 10 vs <= 7) -- more aggressive.
- IIR margin reduction (badNode flag) is an idea Coda doesn't have -- reduces margin when uncertain.
- Blended return `(eval-margin+beta)/2` -- Coda returns staticEval-margin. Could test blended return.

### Null Move Pruning (NMP)
- **Conditions**: `eval >= staticEval && eval >= beta && staticEval >= beta - 28*depth + 204`
  - Extra condition prevents NMP when staticEval is too far below beta
- **Reduction**: `R = 4 + depth/3 + min((eval-beta)/221, 3)` (base 4 vs Coda's 3)
- **badNode reduction**: `-badNode` added to reduction (reduces R by 1 when no TT data)
- **Verification**: At depth >= 15, do verification search with `nmpPlies = ply + (depth-R)*2/3`
- **Mate clamp**: Returns beta if nmpScore is a mate

**Coda comparison**: Coda uses R=3+depth/3 + (eval-beta)/200, verify at depth>=12, NMP score dampening (score*2+beta)/3. Key differences:
- Base reduction 4 vs 3 (more aggressive).
- Extra `staticEval >= beta - 28*depth + 204` guard (Coda doesn't have this).
- `badNode` reduction (reduce R by 1 when no TT info).
- Eval divisor 221 is similar to Coda's 200.

### Razoring
- **Depth**: <= 5
- **Margin**: `258 * depth`
- Drops to QSearch if `eval + 258*depth < alpha`
- Returns razorScore if <= alpha

**Coda comparison**: Coda uses depth <= 2, margin 400+d*100. Alexandria's is deeper (<=5) with tighter per-depth margin (258*d). For depth 2: Alexandria 516 vs Coda 600.

### ProbCut
- **Depth**: > 4
- **Beta**: `beta + 287 - 54*improving`
- **Guard**: TT score is NONE or lower-bound, and TT depth < depth-3 or ttScore >= pcBeta
- **Two-stage**: First runs QSearch with `-pcBeta`, then if QSearch passes, runs full `Negamax` at `depth-4`
- **SEE threshold**: `pcBeta - staticEval` for capture filtering

**Coda comparison**: Coda uses beta+170, staticEval+85 gate, SEE>=0. Key differences:
- Larger base margin (287 vs 170), but with improving offset (54).
- **Two-stage QSearch pre-filter** saves significant nodes. Coda doesn't pre-filter with QSearch before the ProbCut search. Worth testing.

### Futility Pruning (Quiet Moves)
- **Depth**: `lmrDepth < 13`
- **Margin**: `staticEval + 232 + 118*lmrDepth`
  - depth 1: 350, depth 5: 822
- **Guard**: Not in check, quiet move
- **Best score update**: If `bestScore <= futilityValue`, updates bestScore to futilityValue (avoids returning -infinity)
- Sets `skipQuiets = true` after triggering

**Coda comparison**: Coda uses 60+lmrDepth*60, depth<=8. Alexandria's margin is significantly larger (118/ply vs 60/ply). Both use lmrDepth. The `skipQuiets` flag pattern is an optimization to avoid scoring remaining quiets.

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

**Coda comparison**: Coda uses 3+d^2 with improving/failing adjustments, depth<=8. Similar to Alexandria's improving formula. Coda applies only in non-PV.

### History Pruning
```cpp
if (isQuiet && moveHistory < -3753 * depth) skipQuiets = true;
```

**Coda comparison**: Coda uses -1500*depth at depth<=3, !improving guard. Alexandria's threshold (-3753) is more permissive per depth unit but has no depth limit.

### SEE Pruning
- **Quiet moves**: `seeQuietMargin * lmrDepth = -98 * lmrDepth`
- **Noisy moves**: `seeNoisyMargin * lmrDepth^2 = -27 * lmrDepth^2`

**Coda comparison**: Coda uses quiet -20*d^2 at depth<=8, capture -d*100 at depth<=6. Alexandria separates quiet (linear) from noisy (quadratic) with lmrDepth. Different scaling approach.

### Hindsight Reduction
```cpp
if (depth >= 2 && (ss-1)->reduction >= 1
    && (ss-1)->staticEval != SCORE_NONE
    && ss->staticEval + (ss-1)->staticEval >= 155)
    depth--;
```
If the parent move was reduced (LMR) and the sum of parent and current static evals is >= 155 (both sides think position is good = quiet position), reduce depth by 1.

**IMPLEMENTED in Coda**: Coda has hindsight reduction with equivalent logic. Confirmed +18 Elo without it (clean CPU retest).

---

## 5. Extensions

### Singular Extensions
**Conditions**:
```
!rootNode && depth >= 6 && move == ttMove && !excludedMove
&& (ttBound & LOWER) && !isDecisive(ttScore) && ttDepth >= depth - 3
```

**Singular beta**: `ttScore - depth*5/8 - depth*(ttPv && !pvNode)`
- PV-aware: when TT says we were on PV but current node is not PV, reduce margin further

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

**Coda comparison**: Coda has SE at depth >= 8 with singular_beta = tt_score - depth, singular_depth = (depth-1)/2. Coda has multi-cut and negative extensions (-1). Key differences from Alexandria:
1. **Depth threshold**: 8 vs Alexandria's 6 -- less aggressive activation
2. **Singular beta margin**: `ttScore - depth` vs `ttScore - depth*5/8` -- Coda's margin is wider (easier to trigger singularity)
3. **No PV-awareness**: Coda doesn't adjust margin for former PV nodes
4. **No double/triple extensions**: Coda extends by 1 only (no +2 or +3)
5. **Negative extensions**: Coda uses -1 vs Alexandria's -2
6. **No extension limiter**: Coda doesn't cap total ply depth (risk of tree explosion)
7. **No shallow depth boost**: Coda doesn't add extra depth when double-extending

Items 4, 6, and 7 are the most impactful remaining differences. The extension limiter (`ply*2 < RootDepth*5`) is a safety mechanism that should probably be added.

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

**IMPLEMENTED in Coda**: Coda has doDeeper/doShallower with similar logic.

**Coda comparison**: Coda has most LMR adjustments (history, improving, cut-node, contHist, complexity). Key differences:
- Alexandria uses history divisor 8049 (less aggressive than Coda's approach)
- Noisy history divisor 5922 -- separate from quiet
- Former PV (ttPv) adjustment: `-1 - cutNode` -- Coda may not have this
- PV node +1 on reduced depth

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

**Coda comparison**: Similar staged ordering. Coda uses TT move -> good captures (MVV-LVA + captHist/16) -> promotions -> killers -> counter-move -> quiets (main hist + contHist*3 + pawn hist) -> bad captures.

### Capture Scoring
```cpp
score = SEEValue[capturedPiece] * 16 + GetCapthistScore(pos, sd, move);
```
MVV scaled by 16, plus capture history.

### Good Capture SEE Threshold
```cpp
SEEThreshold = -score / 32 + 236;
```
Dynamic: higher-scored captures get a tighter SEE threshold.

**Coda comparison**: Not implemented. Coda uses a fixed SEE threshold for good/bad capture separation. Dynamic thresholds could improve capture ordering quality.

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

**IMPLEMENTED in Coda**: Coda reads/writes continuation history at plies 1, 2, 4, 6 with weights 3x, 3x, 1x, 1x.

### Root History
Dedicated history table for root moves, weighted 4x in scoring. Cleared before each search. Gets separate bonus/malus tuning.

**Coda comparison**: Not implemented. Simple to add, provides better move ordering at root. Low complexity, potentially meaningful impact.

### Opponent History Update (Eval History)
```cpp
if (!inCheck && (ss-1)->staticEval != SCORE_NONE && isQuiet((ss-1)->move)) {
    int bonus = clamp(-10 * ((ss-1)->staticEval + ss->staticEval), -1830, 1427) + 624;
    updateOppHHScore(pos, sd, (ss-1)->move, bonus);
}
```
After computing static eval, if the opponent's last move was quiet, update THEIR butterfly history based on the eval change.

**Coda comparison**: Not implemented. Interesting concept for improving move ordering by penalizing opponent moves that led to bad positions.

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

Notable: **HH malus max is only 452** vs bonus max 2910 -- heavily asymmetric.

**Coda comparison**: Coda uses a single bonus formula (depth^2 capped at 1200). Asymmetric per-table tuning is more sophisticated but requires SPSA.

### TT Cutoff Continuation History Malus
```cpp
if (ttMove && ttScore >= beta && (ss-1)->moveCount < 4 && isQuiet((ss-1)->move)) {
    updateCHScore((ss-1), (ss-1)->move, -min(155*depth, 385));
}
```
When we get a TT cutoff, penalize the opponent's last quiet move in continuation history.

**Coda comparison**: Not implemented. Novel use of TT cutoff information to improve move ordering.

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
```
Where diff = bestScore - staticEval. CORRHIST_MAX = 1024.

### Conditional Update
```cpp
if (!inCheck && (!bestMove || !isTactical(bestMove))
    && !(bound == LOWER && bestScore <= staticEval)
    && !(bound == UPPER && bestScore >= staticEval))
    updateCorrHistScore(...)
```
Only update when the correction is meaningful.

**IMPLEMENTED in Coda**: Coda has multi-source correction history: pawn (512/1024) + white-NP (204/1024) + black-NP (204/1024) + continuation (104/1024). Uses proper per-color non-pawn Zobrist keys, incrementally updated. Coda's weights differ from Alexandria's but the architecture is equivalent.

---

## 9. Transposition Table

### Structure
- **10 bytes per entry**: move(2) + score(2) + eval(2) + ttKey(2) + depth(1) + ageBoundPV(1)
- **3 entries per bucket** (32-byte aligned buckets with 2 bytes padding)
- **Key verification**: 16-bit TTKey (upper bits of Zobrist)
- **Age/Bound/PV packed**: lower 2 bits = bound, bit 2 = PV flag, upper 5 bits = age

### TT Cutoff Conditions
```cpp
!pvNode && ttScore != SCORE_NONE && ttDepth >= depth
&& cutNode == (ttScore >= beta)    // <-- KEY CONDITION
&& fiftyMoveCounter < 90
&& (ttBound & (ttScore >= beta ? LOWER : UPPER))
```

The `cutNode == (ttScore >= beta)` condition: only allow TT cutoffs when the expected node type matches. At cut nodes, only accept fail-highs; at non-cut nodes, only accept fail-lows.

**Coda comparison**: Not implemented. Coda doesn't have the `cutNode == (ttScore >= beta)` guard. This is a precision improvement that could reduce search instability. Worth testing.

### Huge Pages
Linux: aligned to 2MB, uses `madvise(MADV_HUGEPAGE)`.

**Coda comparison**: Coda uses 5-slot buckets (64-byte cache-line aligned), AtomicU64/AtomicU32 for lockless Lazy SMP, XOR key verification. Different TT structure but similar goals.

---

## 10. Time Management

### Base Allocation
```
optScale = min(25/1000, 200/1000 * time / timeLeft)
optime = optScale * timeLeft
maxtime = 0.76 * time - overhead
```

### Node-Based Scaling
```cpp
bestMoveNodesFraction = nodeSpentTable[bestmove] / totalNodes;
nodeScalingFactor = (1.53 - bestMoveNodesFraction) * 1.74;
```

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

### Final Time Calculation
```cpp
stoptimeOpt = starttime + baseOpt * nodeScale * bestMoveScale * evalScale;
stoptimeOpt = min(stoptimeOpt, stoptimeMax);
```

**Coda comparison**: Coda uses simple soft allocation (timeLeft/movesLeft + 80% increment) with MoveOverhead subtraction, emergency mode, and cap at 50% remaining. Lacks all three of Alexandria's scaling factors (node-based, bestmove stability, eval stability). This is probably the single highest-impact area for improvement -- estimated +5-15 Elo.

---

## 11. Lazy SMP

### Thread Management
- Main thread + N-1 helper threads
- Each thread gets own `ThreadData` (position copy, search data, key history, FinnyTable)
- All threads share the TT (no locks, no atomics visible -- relies on TTKey16 for collision detection)
- Helper threads start at depth 1, main thread manages time

**Coda comparison**: Similar approach. Coda uses Lazy SMP with helper threads at offset depths sharing atomic TT and stop flag.

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
**Double blending**: Both at stand-pat AND at final return.

**IMPLEMENTED in Coda**: Coda has QS beta blending at both stand-pat and final return.

### QS Futility
```cpp
futilityBase = staticEval + 268;
if (futilityBase <= alpha && !SEE(pos, move, 1))
    bestScore = max(futilityBase, bestScore);
    continue;
```
Combined eval-based + SEE gate.

**Coda comparison**: Coda uses QS delta of 240. Similar approach.

### Upcoming Repetition in QS
Alexandria checks for upcoming repetition (game cycle) even in QSearch.

**IMPLEMENTED in Coda**: Coda has Cuckoo cycle detection in both main search and QSearch.

---

## 13. Unique/Novel Techniques

### 1. Complexity Metric
```cpp
int complexity = abs(eval - rawEval) / abs(eval) * 100;
// If complexity > 50, reduce LMR by 1
```

**IMPLEMENTED in Coda**: Coda has complexity-aware LMR: `reduction -= complexity / 120` (matches Obsidian style).

### 2. badNode Flag
```cpp
const bool badNode = depth >= 4 && ttBound == HFNONE;
```
Identifies nodes with no TT information at all. Used in IIR, RFP margin reduction, and NMP R reduction.

**Coda comparison**: Not implemented as a unified concept. Coda has IIR but doesn't use the badNode flag to adjust RFP margins or NMP reduction. The unifying concept is worth testing.

### 3. Eval-Based History Depth Bonus
```cpp
UpdateHistories(pos, sd, ss, depth + (eval <= alpha), bestMove, ...);
```
When eval was at or below alpha, give a +1 depth bonus to history updates.

**Coda comparison**: Not implemented. Cheap to add, gives stronger history signal for surprising cutoffs.

### 4. Dynamic SEE Threshold for Good Captures
```cpp
SEEThreshold = -score / 32 + 236;
```

**Coda comparison**: Not implemented. Coda uses fixed SEE threshold for good/bad capture separation.

### 5. Draw Score Randomization
```cpp
return (info->nodes & 2) - 1;  // Returns -1 or 1
```

**Coda comparison**: Coda uses contempt = -10 (prefer playing over drawing). Different approach to the same problem of draw avoidance.

### 6. TT Cutoff Node-Type Guard
```cpp
cutNode == (ttScore >= beta)
```

**Coda comparison**: Not implemented. See TT section above.

---

## 14. Summary: Key Differences from Coda

### Things Alexandria Has That Coda Does NOT (Prioritized by Impact)

1. **Advanced time management** (node-based + bestmove stability + eval stability scaling) -- estimated +5-15 Elo
2. **SE double/triple extensions** and **extension limiter** (`ply*2 < RootDepth*5`) -- Coda has basic SE but lacks these safety/power features
3. **Root history table** -- separate history for root moves, 4x weight
4. **Mate distance pruning** -- 3 lines, free
5. **ProbCut QSearch pre-filter** (two-stage ProbCut)
6. **Opponent history update** (eval-based feedback on opponent's moves)
7. **badNode flag** as unified concept (adjusts RFP margin and NMP R)
8. **TT cutoff node-type guard** (`cutNode == (ttScore >= beta)`)
9. **Eval-based history depth bonus** (`depth + (eval <= alpha)`)
10. **TT cutoff continuation history malus** (penalize opponent's last move on TT cutoff)
11. **Material scaling of NNUE output** (scale down in endgame)
12. **50-move rule eval decay**
13. **Asymmetric history bonus/malus** per table type
14. **Dynamic SEE threshold for good captures**
15. **Factoriser in FT** (shared base for all king buckets -- training-side)

### Things Coda Has That Alexandria Lacks
- NMP score dampening ((score*2+beta)/3)
- Fail-high score blending ((score*depth+beta)/(depth+1))
- TT near-miss (accept 1-ply-short entries with 80cp margin)
- Bad noisy pruning (prune losing captures when eval+depth*75<=alpha)
- Pawn history in move ordering

### IMPLEMENTED Since GoChess Era (No Longer Relevant)
- ~~4-table correction history~~ -- IMPLEMENTED (pawn + white-NP + black-NP + continuation)
- ~~FinnyTable NNUE accumulator cache~~ -- IMPLEMENTED
- ~~Cuckoo cycle detection~~ -- IMPLEMENTED
- ~~Hindsight reduction~~ -- IMPLEMENTED
- ~~DoDeeper/DoShallower~~ -- IMPLEMENTED
- ~~Continuation history plies 4, 6~~ -- IMPLEMENTED
- ~~Complexity-aware LMR~~ -- IMPLEMENTED
- ~~QS beta blending~~ -- IMPLEMENTED
- ~~Singular extensions with multi-cut and negative extensions~~ -- IMPLEMENTED
- ~~NNZ-sparse L1 multiplication~~ -- IMPLEMENTED
- ~~Pairwise FT activation~~ -- IMPLEMENTED (v6)

### Significantly Different Parameters
| Parameter | Alexandria | Coda | Notes |
|-----------|-----------|------|-------|
| Aspiration delta | 12 | 15 | Tighter |
| Aspiration start | depth >= 3 | depth >= 4 | Earlier |
| RFP depth | < 10 | <= 7 | Deeper |
| RFP margin | 75*d | improving?70*d:100*d | Different formula |
| NMP base R | 4 | 3 | More aggressive |
| SE depth threshold | >= 6 | >= 8 | More aggressive |
| SE margin | d*5/8 | d (ttScore-depth) | Different |
| SE negative ext | -2 | -1 | More aggressive |
| LMR history divisor | 8049 | varies | Less reduction from history |
| SEE quiet margin | -98*lmrDepth | -20*d^2 | Different scaling |
| SEE noisy margin | -27*lmrDepth^2 | -d*100 | Different scaling |
| Razoring depth | <= 5 | <= 2 | Deeper |
| ProbCut margin | 287-54*imp | 170 | Larger with improving |
| QS futility | 268 | 240 | Similar |
| History bonus cap | varies per table | 1200 (depth^2) | Per-table tuning |
