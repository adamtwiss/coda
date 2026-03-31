# PlentyChess Technical Analysis

**Source**: `~/chess/engines/PlentyChess/` (C++)
**Strength**: Top-3 on most lists (~3600-3800 Elo), plays in CCC and TCEC
**Key innovation**: Threat-input NNUE with piece-pair attack features

---

## 1. NNUE Architecture

### Network Topology
```
FT: (768 * 12 king buckets + 79856 threat features) -> 640 (int16, split: pieceState + threatState)
L1: 640 -> 16 (int8 weights, sparse matmul)
L2: 2*16 -> 32 (float, with SCReLU-style doubled input)
L3: 32 + 2*16 -> 1 (float, skip connection from L1)
x8 output buckets
```

**Key dimensions** (`nnue.h`):
- `INPUT_SIZE = ThreatInputs::FEATURE_COUNT + 768 * KING_BUCKETS` = 79856 + 9216 = 89072
- `L1_SIZE = 640` (accumulator width)
- `L2_SIZE = 16` (first hidden layer)
- `L3_SIZE = 32` (second hidden layer)
- `OUTPUT_BUCKETS = 8`
- `KING_BUCKETS = 12` (asymmetric layout, more granular near king home ranks)

### Threat Inputs (NOVEL - their key differentiator)

File: `threat-inputs.h/cpp`

79,856 sparse features encoding **which piece attacks which piece on which squares**. This is a piece-pair interaction map:
- For each piece on the board, compute its attack bitboard
- For each attack ray, encode (attackingPiece, attackedPiece, attackingSquare, attackedSquare) as a feature index
- Uses a `PIECE_INTERACTION_MAP[6][6]` that excludes certain pairs (e.g., bishop doesn't interact with queen directly)
- Symmetric pairs (same piece type) are semi-excluded (only one direction counted)
- Features are perspective-relative (flipped for black)

The accumulator has **two separate states** per perspective:
```cpp
struct Accumulator {
    int16_t threatState[2][L1_SIZE];  // threat features (int8 weights)
    int16_t pieceState[2][L1_SIZE];   // standard 768*KB features (int16 weights)
};
```

These are **added together** at inference time before activation. The threat weights are int8 (smaller, since there are 79K features), while piece-square weights are int16 (standard).

**Incremental threat updates**: Dirty threats are tracked per move via `DirtyThreat` structs (4 bytes each). On mirror changes, threats are fully recomputed; otherwise incrementally updated.

> **Coda comparison**: We have no threat inputs. This is the biggest architectural difference between PlentyChess and every other engine. 79K threat features is a massive information injection -- it tells the network exactly which pieces attack which, so it doesn't have to infer attack patterns from piece positions alone. This likely explains a significant portion of their strength. Implementing this in Coda would require:
> 1. Computing attack bitboards for all pieces at accumulator update time
> 2. Encoding piece-pair interactions as features
> 3. Separate int8 weight table for threat features
> 4. Incremental delta tracking for threat features on make/unmake
> The NPS cost would be substantial (attackersTo() for every piece on every move).

### King Buckets
12 buckets with horizontal mirroring:
```
 0  1  2  3  3  2  1  0
 4  5  6  7  7  6  5  4
 8  8  9  9  9  9  8  8
10 10 10 10 10 10 10 10
11 11 11 11 11 11 11 11  (x4 ranks)
```
More granular near the back rank (8 distinct buckets for ranks 1-2), uniform for ranks 4-8. This is similar to what we use.

### Quantization
- `INPUT_QUANT = 255` (QA)
- `L1_QUANT = 64` (QB)
- `INPUT_SHIFT = 9`
- FT weights: int16 (piece-square), int8 (threats)
- L1 weights: int8
- L2/L3: float32

### Pairwise Activation (FT -> L1)

The FT activation is pairwise CReLU-style (`nnue.cpp:407-440`):
```cpp
// First half clipped [0, 255], second half just capped at 255
clipped1 = min(max(piece + threat, 0), 255);  // CReLU
clipped2 = min(piece + threat, 255);           // no floor (can be negative)
// Multiply and pack to uint8
output = (clipped1 * clipped2) >> INPUT_SHIFT;
```
This is effectively pairwise multiplication of the two halves of the accumulator, producing 640 uint8 outputs. The first half is ReLU'd, the second half is not (signed), giving a pairwise product.

> **Coda comparison**: We don't have pairwise in our v5/v7 architecture. PlentyChess uses 640 neurons with pairwise (effectively 320 paired), which compresses information differently than our 1024 CReLU. Their pairwise + threats may compensate for the smaller accumulator.

### Sparse L1 Propagation (PERFORMANCE CRITICAL)

File: `nnue.cpp:446-537`

After pairwise activation produces uint8 outputs, they compute a **non-zero (NNZ) index** of which 4-byte chunks are non-zero. This is the sparse matmul:

1. **NNZ computation**: Scan pairwise outputs as int32 chunks. Use `vecNNZ()` (movemask/cmpgt) to get a bitmask of non-zero chunks. Convert to an index array using a precomputed `nnzLookup[256][8]` table.

2. **Sparse matmul**: Only multiply non-zero chunks against L1 weights:
```cpp
for (int i = 0; i < nnzCount; i += 2) {  // process pairs
    VecIu8 u8_1 = set1Epi32(pairwiseOutputsPacks[nnzIndices[i]]);
    VecI8* weights = &l1Weights[nnzIndices[i] * 4 * L2_SIZE];
    l1MatmulOutputsVec[l1] = dpbusdEpi32x2(...);
}
```

3. **NNZ permutation**: The `NNZ` class (`nnue.h:198-278`) sorts neurons by activation frequency (least active first). This means the most-often-zero neurons are at the beginning, enabling early termination or better cache behavior.

4. **AVX512-VNNI fast path**: Processes 4 NNZ entries at a time with `dpbusdEpi32x2`, accumulating into two separate acc vectors.

> **Coda comparison**: We don't do sparse L1. With our 1024->16 architecture, L1 is only 16 outputs so the dense matmul is already cheap. But if we went wider (1024->64 or similar), sparse L1 would matter. The NNZ permutation trick (sorting neurons by activation frequency) is clever and free at runtime.

### L1 -> L2 (Float, SCReLU-style doubling)

```cpp
l1Outputs[l1] = clamp(l1Result, 0, 1);           // ReLU part
l1Outputs[l1 + L2_SIZE] = clamp(l1Result^2, 0, 1); // Squared part
```
L1 output is doubled: first 16 values are ReLU'd, next 16 are squared. This gives 32 inputs to L2. L2 activation is also SCReLU (clamp then square).

### L3: Skip Connection

```cpp
// L3 takes BOTH L2 outputs AND L1 outputs as input
float l3Weights[OUTPUT_BUCKETS][L3_SIZE + 2 * L2_SIZE];  // 32 + 32 = 64 weights
```
The final layer has a **skip connection** from L1 directly to the output. This means L3 sees both the L2-processed signal and the raw L1 activations. This is important for gradient flow through the narrow layers.

> **Coda comparison**: We don't have skip connections in our v7 architecture. Adding L1 skip to the output layer would be simple: concatenate L1 output with L2 output before the final matmul. Worth testing -- it helps gradient flow and is nearly free computationally (just 32 extra float multiplies).

### Network Scale
`NETWORK_SCALE = 287` -- final output multiplied by this constant.

---

## 2. Search

### Depth Representation
PlentyChess uses **centiplex depth** (depth * 100 internally). This gives finer-grained control over extensions and reductions without floating point. `depth < 100` drops to qsearch. Extensions/reductions are in centiplexes (e.g., `extension = 100` = 1 ply).

> **Coda comparison**: We use integer plies. Centiplex depth is cleaner for fractional extensions. Worth adopting.

### Pre-Search Pruning

**TT Cutoff** (`search.cpp:678`):
```cpp
if (!pvNode && ttDepth >= depth - ttCutOffset + ttCutFailHighMargin * (ttValue >= beta) && ...)
```
Adaptive TT depth requirement -- more lenient when TT value beats beta. `ttCutOffset=44`, `ttCutFailHighMargin=122`.

**IIR** (`search.cpp:779`): Two-stage IIR:
1. Standard IIR when no TT hit and depth >= 288 (or 711 in check)
2. "IIR 2: Electric boolagoo" -- additional reduction for PV nodes without TT hit at depth >= 251

**Static History Adjustment** (`search.cpp:773`):
```cpp
int bonus = clamp(staticHistoryFactor * (staticEval + prevStaticEval) / 10, min, max) + tempo;
history.updateQuietHistory(prevMove, flip(stm), prevBoard, bonus);
```
Adjusts quiet history of the **previous move** based on how much static eval changed. Novel: feeds eval-change information back into history tables.

> **Coda comparison**: We don't adjust history based on eval changes between plies. This is an eval-agnostic information improvement (it tells history "this move caused a big eval shift") and should transfer well cross-engine.

**Post-LMR Opponent Worsening** (`search.cpp:783-789`):
```cpp
if ((stack-1)->inLMR && (stack-1)->reduction >= 220 && staticEval <= -(stack-1)->staticEval)
    depth -= 152;  // reduce more when opponent's reduced search shows position worsened
```
If the parent node was in LMR with a large reduction, and the static eval from the opponent's perspective got worse, reduce further. This is a novel "the situation got worse for our opponent in the reduced search" signal.

**RFP** (`search.cpp:793-806`): Quadratic margin with separate parameters for in-check positions:
```cpp
rfpMargin = rfpBase + rfpFactorLinear * depth/100 + rfpFactorQuadratic * depth^2 / 1e6
```
Returns `(eval + beta) / 2` (blended). Also considers `opponentHasGoodCapture()` -- doesn't count as improving if opponent has good captures available.

**NMP** (`search.cpp:816-856`): Standard with eval-depth condition:
```cpp
staticEval + nmpEvalDepth * depth / 100 - nmpEvalBase >= beta
```
Verification search at high depths (>= 1500 centiplex = 15 ply).

**ProbCut** (`search.cpp:858-907`): Standard ProbCut with qsearch before full re-search.

### Singular Extensions (`search.cpp:1010-1064`)

Standard singular search with **continuous double/triple extensions**:
```cpp
extension = 100;  // base SE
extension += 100 * clamp(singularMargin, 0, 6) / 6;  // up to +100 more
// For quiets or PV nodes, triple extension possible:
extension += 100 * clamp(singularMargin - 6, 0, 35) / 35;
```
The extension scales continuously with how singular the move is (singularMargin = difference between singular beta and singular value). Quiets and PV nodes get more aggressive extensions.

**Multicut on singular fail-high**: When singular search beats beta, stores result and returns early. Also updates correction history from the singular search result.

> **Coda comparison**: Our singular extensions are currently disabled (harmful cross-engine per CLAUDE.md). PlentyChess uses them heavily with continuous scaling. The correction history update on multicut is novel.

### LMR

Three separate reduction tables (quiet, noisy, "important capture"):
```cpp
REDUCTIONS[0] = quiet:    1.12 + ln(i)*ln(j)/2.95
REDUCTIONS[1] = noisy:   -0.23 + ln(i)*ln(j)/2.98
REDUCTIONS[2] = important: -0.13 + ln(i)*ln(j)/3.17
```

"Important captures" = `ttPv && capture && !cutNode`. These get their own reduction table with lower base reductions and different tuning parameters for every LMR adjustment.

LMR adjustments:
- Correction value: `abs(correctionValue / 142K)`
- In check: reduce less
- Cut node: reduce more (+257 centiplex)
- TT PV: reduce less (-169/188)
- TT PV fail-low: reduce more
- Capture history: `history * |history| / divisor` (quadratic scaling!)
- Quiet history: linear scaling
- PV node, improving: reduce less

**Post-LMR deeper/shallower** (`search.cpp:1136-1138`):
```cpp
bool doShallower = value < bestValue + newDepth/100;
bool doDeeperSearch = value > bestValue + 40 + 2*newDepth/100;
newDepth += 116*doDeeperSearch - 114*doShallowerSearch;
```

**LMR research skip** (`search.cpp:1140`):
```cpp
if (value > alpha && reducedDepth < newDepth
    && !(ttValue < alpha && ttDepth - 435 >= newDepth && (ttFlag & UPPER)))
    // skip re-search if TT says it would fail low
```
Skips the full-depth re-search if the TT already has an upper bound suggesting it would fail low. Saves nodes.

> **Coda comparison**: The research-skip optimization is novel and cheap. The quadratic capture history scaling is interesting. Having 3 separate LMR tables (quiet/capture/important-capture) is more granular than our 2 tables.

### Move-Count Pruning (LMP)

```cpp
if (improving)
    lmpMargin = 3863191 + 77*d^2 + 1500*d + history;
else
    lmpMargin = 2958636 + 16*d^2 + 4500*d + history;
moveCount >= lmpMargin / 1000000
```
History-aware LMP -- moves with better history survive longer. The margin incorporates move history directly.

### Futility Pruning

Separate futility for:
1. **Quiets**: `eval + base + factor*lmrDepth + pvBonus + contHistBonus <= alpha`
2. **Bad noisies**: `eval + base + pieceValue + factor*lmrDepth + pvBonus <= alpha` (breaks out of move loop)
3. **Captures**: `eval + base + pieceValue + factor*lmrDepth <= alpha`

Uses `lmrDepth` (reduced depth) rather than actual depth for all futility decisions.

### SEE Pruning

```cpp
int seeMargin = capture ? seeMarginCapture * d^2 / 10000 : seeMarginQuiet * lmrDepth / 100;
```
Capture SEE is quadratic in depth (-22 * d^2 / 10000), quiet SEE is linear in reduced depth.

### History Pruning

```cpp
if (!pvNode && lmrDepth < 468 && moveHistory < hpFactor * depth/100)
    continue;
```
`hpFactor` is -2074 for captures, -6796 for quiets.

### Fail-High Score Blending

```cpp
if (!pvNode && bestValue >= beta && abs(bestValue) < TB && abs(beta) < TB && abs(alpha) < TB)
    bestValue = (bestValue * depth + 100 * beta) / (depth + 100);
```
Blends the fail-high value toward beta, weighted by depth. At low depths, this pulls the value closer to beta (dampening inflated scores).

> **Coda comparison**: We have this too. Their formula is depth-weighted while ours uses a fixed blend.

### Low-Depth PV Reduction

```cpp
if (depth > 404 && depth < 1021 && beta < TB && value > -TB)
    depth -= 105;  // reduce remaining depth after alpha raise in PV
```
After raising alpha in a PV node at moderate depths, reduce the remaining search depth. This is "alpha-reduce" -- we have something similar.

### FMR Hash

```cpp
Hash fmrHash = board->hashes.hash ^ Zobrist::FMR[board->rule50_ply / 10];
```
The TT hash incorporates fifty-move-rule ply in 10-move granularity buckets. This means positions that differ only in rule50 get different TT entries (preventing stale evaluations near the draw threshold).

> **Coda comparison**: We don't incorporate rule50 into TT hash. This is a clean improvement that prevents TT pollution from rule50 differences.

---

## 3. History & Move Ordering

### History Types

1. **Quiet History** (`quietHistory[2][64][2][64][2]`): Indexed by `[stm][from][fromThreatened][to][toThreatened]`. **Threat-aware** -- whether the origin and target squares are under enemy attack. This doubles the effective table size for better discrimination.

2. **Continuation History** (`continuationHistory[2][2][2][Piece::TOTAL][64][Piece::TOTAL * 64 * 2]`): Indexed by `[inCheck][capture][stm][prevPiece][prevTarget][currentPiece*64*2 + currentTarget*2 + stm]`. Uses plies -1, -2, -4, -6 with weights 2, 1, 1, 0.5.

3. **Pawn History** (`pawnHistory[8192][2][6][64]`): Indexed by pawn hash.

4. **Capture History** (`captureHistory[2][6][64][6]`): Standard `[stm][movedPiece][target][capturedPiece]`.

5. **Counter Moves**: Standard `[from][to]` table.

### Correction History (5 types!)

1. **Pawn correction** (pawn hash indexed)
2. **Non-pawn correction** (separate white/black non-pawn hashes, both contribute)
3. **Minor correction** (minor piece hash)
4. **Major correction** (major piece hash)
5. **Continuation correction** (`continuationCorrectionHistory[2][6][64][2][2]` -- indexed by `[stm][piece][target][fromThreatened][toThreatened]`)

All combined with tuned weighting factors:
```cpp
return pawnEntry * 6252 + nonPawnEntry * 5916 + minorEntry * 4109
     + majorEntry * 2627 + contEntry * 6019;
```

Applied as: `eval = eval * (293 - rule50) / 293 + correctionValue / 65536`

> **Coda comparison**: We have pawn correction history only. PlentyChess has 5 types with separate hashes (nonPawn per color, minor, major, continuation). The minor/major hash separation is interesting -- it creates collision-free correction for different piece compositions. The continuation correction is indexed by the previous move's threatened squares, which is novel.

### Move Ordering: Threat-Aware Quiet Scoring

```cpp
// Bonus for moving a piece OUT of a threat zone
if (piece == QUEEN && from & (pawnThreats|knightThreats|bishopThreats|rookThreats))
    score += 19431;
// Penalty for moving INTO a threat zone
if (piece == QUEEN && to & (pawnThreats|knightThreats|bishopThreats|rookThreats))
    score -= 19431;
```
Similar for rooks (~12K) and minor pieces (~8K). Captures don't need this since SEE handles them.

> **Coda comparison**: We don't have threat-aware move ordering for quiets. This is a form of SEE-lite for quiet moves and should be cheap to compute (just bitboard AND checks).

### History Bonus/Malus Scaling

Bonuses scale with `moveSearchCount` (how many times the move was re-searched):
```cpp
history.updateQuietHistory(bestMove, board->stm, board, quietHistBonus * bestMoveSearchCount);
```
Moves that required more re-searches get proportionally larger history updates. This is a self-reinforcing signal: hard-to-prove moves that turn out good get bigger bonuses.

> **Coda comparison**: We don't scale history updates by search count. This is a novel information injection.

### Capture Scoring

```cpp
score = captureHistory + (PIECE_VALUES[captured] - PIECE_VALUES[moved]) * 147 / 100;
```
MVV-LVA plus capture history. Good captures tested with SEE: `SEE(move, -score / 80)` -- the threshold is proportional to the capture's estimated value. Bad captures go to a separate list.

### Sort Optimization

```cpp
void MoveGen::sortMoves() {
    int limit = -3500 * depth;
    for (int i = ...) {
        if (moveListScores[i] < limit)
            continue;  // skip insertion sort for very bad moves at high depths
        // insertion sort
    }
}
```
Skips sorting moves with very low scores at higher depths. They'll be pruned anyway.

---

## 4. Performance & SIMD

### SIMD Tiers
- AVX-512 + VNNI (fastest path, `dpbusdEpi32` as native instruction)
- AVX-512 (fallback via maddubs)
- AVX2 (256-bit)
- SSSE3 (128-bit)
- NEON (ARM64)
- Scalar fallback

### `dpbusdEpi32x2` Optimization

Processes two dot products simultaneously by adding the intermediate 16-bit results before the final 32-bit accumulation:
```cpp
VecI16 mul1 = maddubs(u, i);
VecI16 mul2 = maddubs(u2, i2);
VecI32 sum32 = madd(add(mul1, mul2), set1(1));  // combine before widening
return add(sum32, sum);
```
This saves one madd instruction per pair. Used extensively in the sparse L1 matmul.

### Finny Tables

Standard `FinnyEntry[2][KING_BUCKETS]` with per-perspective piece bitboard caching. Only stores piece state (not threat state -- threats are always recomputed or incrementally updated separately).

### NUMA Support

Full NUMA awareness:
- Detects NUMA topology
- Duplicates network weights per NUMA node (`numa_alloc_onnode`)
- Binds worker threads to specific NUMA nodes
- Uses huge pages (`MADV_HUGEPAGE`)

> **Coda comparison**: We have no NUMA support. This matters for multi-socket servers (TCEC, CCC).

### TT Structure

5-entry clusters (64 bytes = cache line), generation-based replacement, huge page support.

---

## 5. Time Management

### Base Time Allocation
```cpp
totalTime = 10 * time / 134 + 10 * increment / 15;
maxTime = 729 * time / 1000;
optTime = min(maxTime, 0.846 * totalTime);
maxTime = min(maxTime, 2.81 * totalTime);
```

### Soft TM Adjustments (at each depth completion)

1. **Best move stability**: `adjustment *= 1.56 - stability * 0.052` (stability caps at 18). Stable best move = less time.

2. **Eval difference**: `adjustment *= 0.953 + clamp(prevValue - currentValue, -8, 63) * 0.004`. Dropping eval = more time.

3. **Node fraction**: `adjustment *= 1.68 - 0.974 * (bestMoveNodes / totalNodes)`. If best move used few nodes, spend more time.

4. **Score complexity**: `adjustment *= max(0.77 + complexity/386, 1.0)` where `complexity = 0.6 * abs(baseValue - currentValue) * ln(depth)`. Volatile evaluations = more time.

### Hard Time Check
Node-based: checks `getTime()` every 1024 nodes (`nodesSearched % 1024 == 0`).

> **Coda comparison**: Our TM is simpler. The score complexity factor (using base-depth eval difference * log(depth)) is novel and addresses the case where the eval oscillates without the best move changing.

---

## 6. Other Notable Features

### Upcoming Repetition Detection (Cuckoo)

Full cuckoo table implementation (`search.cpp:335-381`). Detects if the side to move can reach a position that appeared earlier in the game, even through intermediate moves. Used to raise alpha to draw score when alpha < 0.

### Optimism

```cpp
int updatedOptimism = 150 * meanScore / (abs(meanScore) + 125);
optimism[stm] = updatedOptimism;
optimism[!stm] = -updatedOptimism;
```
Based on the running mean of root move scores. Applied in eval:
```cpp
eval = (nnueEval * (33060 + material) + optimism * (2000 + material)) / 38912;
```
Material-scaled optimism -- stronger effect in positions with more material.

### Rule50 in Eval

```cpp
eval = eval * (293 - rule50) / 293;
```
Linear decay of eval as rule50 approaches draw. Applied before correction history.

### Aspiration Windows

Delta scales quadratically with mean score:
```cpp
delta = min(10 + meanScore^2 / 12928, INFINITE);
```
Wider windows for volatile positions. On fail-high, reduces search depth: `searchDepth = max(1, depth - failHighs)`.

### Draw Score

```cpp
return 4 - (nodesSearched & 3);  // Returns 1, 2, 3, or 4
```
Slightly randomized draw contempt to avoid 3-fold blindness.

---

## 7. Actionable Items for Coda/GoChess

### High Priority (likely Elo gains)

1. **FMR Hash** (rule50 incorporated into TT hash in 10-move buckets): Simple to implement, prevents TT pollution. No NPS cost.

2. **Multiple correction histories** (nonPawn, minor, major, continuation): We only have pawn correction. Adding 4 more types with separate hash keys is straightforward. The minor/major split is especially interesting.

3. **Threat-aware quiet history** (`[from][fromThreatened][to][toThreatened]`): Doubles the effective history table by distinguishing moves based on whether squares are under attack. Cheap to compute.

4. **Static history adjustment** (adjust previous move's history based on eval change): Novel feedback mechanism. Eval-agnostic, should transfer well.

5. **History bonus scaling by search count**: Moves that required re-search get proportionally larger history updates.

6. **LMR research skip** (skip full-depth re-search when TT upper bound says it would fail low): Saves nodes with no risk.

7. **Threat-aware move ordering** for quiets (bonus for escaping threats, penalty for moving into threats): Simple bitboard checks, no SEE needed.

### Medium Priority (architectural)

8. **Centiplex depth**: Finer-grained extensions/reductions without floating point. Requires refactoring all depth comparisons.

9. **Skip connection in NNUE** (L1 output feeds directly to final layer alongside L2): Nearly free in float inference. Helps gradient flow.

10. **L1 output doubling** (ReLU + squared): PlentyChess doubles L1 output (16 -> 32) by providing both linear and squared activations. This is effectively free and gives L2 both the signal and its square.

### Lower Priority (complex to implement)

11. **Threat input features**: Their key innovation. 79K features encoding piece-pair attacks. Very expensive to implement and requires retraining. Would need to evaluate NPS impact carefully.

12. **Sparse L1 matmul**: Only relevant if we increase L1 output size. With our 16-neuron L1, dense matmul is already fast.

13. **NNZ neuron permutation**: Sort neurons by activation frequency at network processing time. Free at runtime, helps sparse L1.

14. **NUMA support**: Only matters for multi-socket (tournament) hardware.
