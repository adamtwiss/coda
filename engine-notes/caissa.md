# Caissa Engine - Technical Review

Source: `~/chess/engines/Caissa/src/backend/`
Version: 1.24.21
Author: Michal Czardybon

All parameter values are SPRT-tuned via `DEFINE_PARAM` macros. Values listed are the tuned defaults.

---

## 1. NNUE Architecture

### Network Topology
- **Architecture**: HalfKA -> CReLU -> output (direct, no hidden layers)
- **Feature set**: `32 king buckets * 12 piece types * 64 squares = 24,576` input features
- **Accumulator size**: 1024 (int16)
- **Output**: 1 value per variant
- **8 output variants** selected by piece count: `min(numPiecesExcludingKing / 4, 7)`
  - Each variant has independent output weights (2*1024 int16) and bias (1 int32)
  - This is a **piece-count output bucketing** scheme -- rare among top engines

```
PackedNeuralNetwork layout:
  Header (64 bytes)
  accumulatorWeights[24576 * 1024] (int16)  -- ~48MB
  accumulatorBiases[1024] (int16)
  lastLayerVariants[8] { weights[2*1024] (int16), bias (int32) }
```

**Coda comparison**: GoChess uses HalfKA(32kb) -> 1024 CReLU -> 1 output. No output bucketing. Caissa's 8-variant output is interesting -- it means the net has piece-count-specific output weights without needing hidden layers. Could improve endgame accuracy without the NPS cost of hidden layers. Worth investigating for GoChess v5 before committing to v7 hidden layers.

### Quantization
- **First layer**: int16 weights/biases, QA = 256 (ActivationRangeScaling)
- **Output layer**: int16 weights, int32 bias
  - WeightScale = 256 (1 << 8)
  - OutputScale = 1024 (1 << 10)
  - Output weight quant scale = `WeightScale * OutputScale / ActivationRangeScaling = 1024`
  - Output bias quant scale = `WeightScale * OutputScale = 262144`

**Coda comparison**: GoChess uses QA=255, QB=64. Caissa uses powers of 2 (256, 256, 1024) enabling bit shifts instead of division. The output layer uses int16 weights (not int8), giving more precision for the direct-output architecture.

### Activation
- **CReLU**: `clamp(x, 0, 256)` -- standard clipped ReLU
- No SCReLU, no pairwise multiplication
- Applied at inference time in the output linear layer (not stored in accumulator)

### King Buckets (32)
```
 0  1  2  3   3  2  1  0
 4  5  6  7   7  6  5  4
 8  9 10 11  11 10  9  8
12 13 14 15  15 14 13 12
16 17 18 19  19 18 17 16
20 21 22 23  23 22 21 20
24 25 26 27  27 26 25 24
28 29 30 31  31 30 29 28
```
- Horizontally mirrored (file >= 4 flips)
- Each rank has 4 unique buckets (4 files * 8 ranks = 32)

**Coda comparison**: GoChess also uses 32 king buckets with the same layout.

### Accumulator Cache (Finny Table)
- `AccumulatorCache::KingBucket` stores: accumulator + piece bitboards `[2 colors][6 piece types]`
- Cache dimensions: `[2 perspective][2 * 32 king side+bucket]` = 128 entries
- On king bucket change: diffs cached vs current piece bitboards, applies only changed features
- On init: copies biases into all cache entries, zeros piece bitboards
- **Two-stage parent update**: if parent node's accumulator is dirty, updates parent first so siblings can reuse it

**Coda comparison**: GoChess has an identical Finny table design. The two-stage parent update is a nice touch -- ensures that sibling nodes benefit from a single parent refresh rather than each independently walking up the tree.

### Eval Post-Processing (`Evaluate.cpp`)
```cpp
value = nnOutput / (OutputScale * WeightScale / c_nnOutputToCentiPawns);
value = value * (52 + gamePhase) / 64;  // phase scaling (0=endgame, 24=opening)
// castling rights bonus: +5 per remaining right when king has moved off e1/e8
// eval saturation: past +/-8000, compressed by /8
```
- **Game phase scaling**: `gamePhase = minors + 2*rooks + 4*queens`, scales eval from 81% (endgame) to 119% (opening)
- **Endgame evaluation**: checked first for <= 6 pieces (before NNUE eval)

### SIMD Support
- **AVX-512**, AVX2, SSE2, ARM NEON all supported via preprocessor macros
- **VNNI support**: uses `_mm256_dpwssd_epi32` when available (fused dot-product)
- Accumulator operations use tiled register blocking (OptimalRegisterCount tiles)
- Output layer: 2x unrolled sum for STM and NSTM perspectives
- Extra target accumulator: `UpdateImpl<WithExtraTarget>` writes both current and cache in single pass

**Coda comparison**: GoChess has AVX2 and NEON. Caissa additionally supports AVX-512 and VNNI. The VNNI path (`dpwssd`) is relevant for newer Intel CPUs -- could be a meaningful speedup for Zen4/Sapphire Rapids.

---

## 2. Search: Pruning Techniques

### Reverse Futility Pruning (RFP)
- **Depth guard**: `depth <= 6` (RfpDepth=6)
- **Margin**: `83*depth + 0*depth^2 - 145*improving_and_no_opp_material_threat`
  - "improving" only counts if opponent cannot win material (`OppCanWinMaterial` checks threat bitboards: rook attacks on queens, minor attacks on Q/R, pawn attacks on Q/R/B/N)
- **Floor**: margin clamped to at least RfpTreshold=16
- **Return value**: blended `(eval * (1024-525) + beta * 525) / 1024` (not just eval or beta)
- **Conditions**: non-PV, not in check, no filtered move, eval <= KnownWinValue

**Coda comparison**: GoChess uses `60 + 60*depth`. Caissa's `83*depth` is steeper but capped at depth 6 (GoChess: depth 7). The improving+threat interaction is more sophisticated. The blended return value is novel -- GoChess returns raw eval.

### Razoring
- **Depth guard**: `depth <= 4` (RazoringStartDepth=4)
- **Margin**: `22 + 158*depth`
- **Mechanism**: if `eval + margin < beta`, do qsearch; return qscore if still < beta

**Coda comparison**: GoChess uses `400 + 100*depth`. Caissa's margins are tighter (at depth 4: 654 vs 800).

### Null Move Pruning (NMP)
- **Depth guard**: `depth >= 3`
- **Eval condition**: `eval >= beta + (depth < 4 ? 16 : 0)`, `staticEval >= beta`
- **Only on cut nodes** (`node->isCutNode`) -- this is unusual
- **No consecutive null moves**: checks parent and grandparent
- **Reduction R**: `3 + depth/3 + min(3, (eval-beta)/85) + improving`
- **Verification re-search**: if score >= beta AND (`abs(beta) < KnownWinValue && depth < 10`): return immediately. Otherwise reduce depth by 5 and continue.

**Coda comparison**: GoChess applies NMP on any non-PV node, not just cut nodes. Restricting to cut nodes is more conservative -- may miss some pruning opportunities but avoids bad null-move results in all-nodes.

### ProbCut
- **Depth guard**: `depth >= 5`, non-PV, not in check
- **Beta**: `beta + 133`
- **SEE threshold**: `probBeta - staticEval` (dynamic)
- **Process**: qsearch first, then `depth-4` NegaMax verification
- **In-Check ProbCut** (from Stockfish): non-PV, in check, TT move is capture, TT has lower bound with `depth >= depth-4` and `ttScore >= beta+329`. Returns `probCutBeta` directly (no search).

**Coda comparison**: GoChess has ProbCut but not in-check ProbCut. The dynamic SEE threshold (`probBeta - staticEval`) is clever -- it adapts to position evaluation.

### Futility Pruning
- **Depth guard**: `depth < 9`
- **Margin**: `staticEval + 32*depth^2 + moveStatScore/383`
- Note: quadratic depth scaling, plus history influence
- On trigger: `movePicker.SkipQuiets()`, but first quiet still tried

**Coda comparison**: GoChess uses `60 + 60*depth` (linear). Caissa's quadratic scaling is more aggressive at higher depths.

### Late Move Pruning (LMP)
- **Threshold**: `4 + depth^2` when improving, `4 + depth^2/2` otherwise
- Adds `LateMovePruningPVScale * isPvNode` (=2) to depth for PV nodes
- **Optimization**: if in quiets stage, breaks out entirely

**Coda comparison**: GoChess uses `3 + d^2`. Very similar.

### History Pruning
- **Depth guard**: `depth < 9`
- **Threshold**: `0 - 234*depth - 148*depth^2` (quadratic)
- Uses moveStatScore (main history + conthist[0,1,3])

**Coda comparison**: GoChess uses `-5000*depth` (linear). Caissa's quadratic formula is more aggressive at higher depths.

### SEE Pruning
- **Only when move target is on a threatened square** (`move.ToSquare() & allThreats`)
- **Captures**: `depth <= 5`, threshold = `-120*depth`
- **Non-captures**: `depth <= 9`, threshold = `-49*depth - moveStatScore/134`
  - History integrated into SEE threshold for quiets

**Coda comparison**: GoChess uses `-20*depth^2` for quiets, `-depth*100` for captures. Caissa's history-in-SEE-threshold is a good idea we should consider. The "only when threatened" guard is also interesting -- saves SEE computation on unthreatened squares.

### Internal Iterative Reduction (IIR)
- `depth >= 3`, cut node OR PV node
- Fires if no TT move OR `ttEntry.depth + 4 < depth` (stale TT entries also trigger)
- Reduces depth by 1

**Coda comparison**: GoChess IIR fires on missing TT move. Caissa also fires on stale TT entries (depth + 4 < current depth) -- this is a good extension of the idea.

---

## 3. Extensions

### Singular Extensions
- **Conditions**: non-root, `depth >= 3`, TT move with `|ttScore| < KnownWinValue`, TT has lower bound, `ttEntry.depth >= depth - 3`
- **Singular beta**: `ttScore - depth`
- **Singular depth**: `max(1, (59*depth - 215) / 128)`
- **Results**:
  - Singular (score < singularBeta, ply < 2*rootDepth): +1 extension
  - Double extension: `score < singularBeta - 14 - 256*isPvNode`
  - Triple extension: `score < singularBeta - 51 - 256*isPvNode`
  - **Multi-cut**: returns `(singularScore * singularDepth + beta) / (singularDepth + 1)` -- blended return
  - **Negative extensions**: -2 or -3 depending on conditions (ttScore >= beta, isCutNode, ttScore <= alpha)

### Recapture Extension
- PV nodes only, TT move is a recapture (same target square): +1

**Coda comparison**: GoChess has singular extensions but found them harmful cross-engine (see CLAUDE.md). Caissa uses them extensively. The triple extension and blended multi-cut return are more sophisticated than typical implementations. The negative extensions (-2/-3) when not singular are aggressive depth reductions that act like pruning.

---

## 4. Move Ordering

### Stages (MovePicker)
1. **TT Move** (score: INT32_MAX - 1)
2. **Generate & pick captures** (winning/good + queen promotions)
3. **Killer move** (1 per ply, score: 1,000,000)
4. **Counter move** (score: 999,999)
5. **Generate & pick quiets** (sorted by combined score)
6. Bad captures (from step 2, ordered last)

Note: only 1 killer move per ply (not 2). Counter move is a separate stage.

### History Tables

#### Quiet Move History
- `quietMoveHistory[2 stm][2 from_threatened][2 to_threatened][4096 from_to]`
- **Threat-aware**: separate counters for whether source/dest squares are attacked by opponent
- This gives 4x the granularity of standard butterfly history

**Coda comparison**: GoChess uses standard `[2 stm][4096 from_to]`. The threat-aware indexing is a proven Elo gain in many engines. Priority adoption candidate.

#### Continuation History
- 6 plies back: `continuationHistory[2 prevIsCapture][2 prevColor][2 currentColor][6 piece][64 to]`
- **Scoring weights**: conthist[0]=1.0, [1]=1019/1024, [3]=555/1024, [5]=582/1024
- **Update weights**: conthist[0]=1.0, [1]=1014/1024, [2]=300/1024, [3]=978/1024, [5]=978/1024
- Note: conthist[2] updated but NOT scored, conthist[4] neither updated nor scored

**Coda comparison**: GoChess uses 3 plies of continuation history. Caissa uses 6 with weighted scoring. The capture-aware indexing (`prevIsCapture`) is also unusual -- most engines don't split by capture vs quiet predecessor.

#### Capture History
- `capturesHistory[2 stm][6 piece][5 captured_piece][64 to]`
- Standard piece-captured-to indexing

#### Counter Moves
- `counterMoves[2 stm][6 piece][64 to]`

### Quiet Scoring Details
- Main history + weighted conthist (4 tables)
- **Threat-based piece bonuses**:
  - Knight/Bishop: +4000 from pawn-attacked, -4000 to pawn-attacked
  - Rook: +8000 from minor-attacked, -8000 to minor-attacked
  - Queen: +12000 from rook-attacked, -12000 to rook-attacked
- **Node cache bonus** (ply < 3): `4096 * moveNodesSearched / totalNodes`

### Capture Scoring
- `attacker < captured`: WinningCaptureValue (20M)
- `attacker == captured`: GoodCaptureValue (10M)
- SEE >= 0: GoodCaptureValue (10M)
- SEE < 0: INT16_MIN
- Plus `4096 * captured_piece_type` (MVV)
- Plus capture history (shifted by -INT16_MIN)

### History Update Formula
- Gravity-style: `counter += delta - counter * |delta| / 16384`
- **Quiet bonus**: `min(-113 + 164*depth + 148*scoreDiff/64, 2178)`, scoreDiff = `min(bestValue - beta, 256)`
- **Quiet malus**: `-min(-51 + 160*depth + 155*scoreDiff/64, 1844)`
- **History bonus includes scoreDiff**: moves causing larger beta cutoffs get larger updates
- **Non-zero initialization**: quiets=802, conthist=762, captures=346 (biases unexplored moves positive)
- **New search**: history divided by 2, killers cleared

### Prior Counter-Move History Update
- When `bestValue <= oldAlpha` and previous move was quiet: bonus of `min(1200, depth*120 - 100)` to the predecessor's continuation history
- Rationale: if all responses to the previous move are bad, the previous move was probably good

---

## 5. LMR (Late Move Reductions)

### Base Table
- **Separate tables for quiets and captures**
- Quiets: `64 * (0.56 + 0.43 * ln(d) * ln(m))`
- Captures: `64 * (0.68 + 0.42 * ln(d) * ln(m))`
- 64x64 table, units of 1/64

### Quiet Adjustments (in 1/64 units)
| Adjustment | Value | Direction |
|---|---|---|
| Non-PV node | +15 | more reduction |
| TT move is capture | +73 | more reduction |
| Move is killer/counter | -168 | less reduction |
| Cut node | +183 | more reduction |
| Not improving | +38 | more reduction |
| Move gives check | -71 | less reduction |
| History-based | `-(stat + 6877) / 240` | variable |

### Capture Adjustments (in 1/64 units)
| Adjustment | Value | Direction |
|---|---|---|
| Winning capture | -63 | less reduction |
| Bad capture | r += -12 | less reduction |
| Cut node | +81 | more reduction |
| Not improving | r -= (-18) = +18 | more reduction |
| Move gives check | r -= (-4) = +4 | slightly more |

### PV-Specific
- `r -= 64 * depth / (1 + ply + depth)` -- less reduction at low ply
- TT entry with high depth: `r -= 13`

### LMR Deeper/Shallower
- If reduced search beats alpha: `newDepth += (score > bestValue + 85) && (ply < 2*rootDepth)`
- `newDepth -= (score < bestValue + newDepth)`

**Coda comparison**: GoChess has separate quiet/capture LMR (matching Caissa). The capture LMR values are interesting -- bad captures get LESS reduction, which is counterintuitive but may help avoid missing tactical shots.

---

## 6. Quiescence Search

- **Stand-pat beta blending**: returns `(bestValue * 0.49 + beta * 0.51)` instead of just bestValue
- **End-of-qsearch blending**: `(bestValue * 0.47 + beta * 0.53)`
- **Futility base**: `standPat + 77`
- **Move count pruning**: depth < -4: 1 move, < -2: 2 moves, < 0: 3 moves
- **In check**: generates evasions, tries only 1 if it doesn't improve
- Bad captures break immediately (not just skip)
- CanReachGameCycle check: if alpha < 0, raises alpha to 0 (cycle detection)

**Coda comparison**: GoChess has beta blending in qsearch. The move-count pruning by qsearch depth is interesting.

---

## 7. Eval Correction History

### Four Tables
1. **Pawn structure**: `[2 stm][16K]` keyed by pawn hash
2. **Non-pawn white**: `[2 stm][16K]` keyed by white non-pawn hash
3. **Non-pawn black**: `[2 stm][16K]` keyed by black non-pawn hash
4. **Continuation correction**: `[2 stm][384 piece-to][384 piece-to]` for ply-2 and ply-4

### Application
- `corr = 53*pawn + 65*nonPawnW + 65*nonPawnB + 76*cont_ply2 + 76*cont_ply4`
- Divided by 512, applied on top of NNUE eval, then 50-move scaling

### Update
- Bonus: `clamp((bestValue - unadjustedEval) * depth / 4, -249, 249)`
- Gravity: `h += value - h * |value| / 1024`
- Condition: not in check, best move is quiet or loses SEE, score diverged from eval

**Coda comparison**: GoChess has pawn correction history. Caissa additionally has non-pawn white/black (split by color) and continuation correction. The continuation correction is novel -- it adjusts eval based on the move sequence, not just the position hash.

---

## 8. Novel / Notable Features

### 1. Output Bucketing by Piece Count (8 variants)
- `variant = min(numPiecesExcludingKing / 4, 7)`
- Each variant has independent output weights (2*1024 weights + 1 bias)
- Allows the network to learn piece-count-specific evaluation scales
- Very cheap at inference time (just selects which weight vector to dot-product)

**Coda comparison**: This is the most interesting architectural feature. It gives endgame/middlegame specificity without hidden layers. GoChess could implement this for v5 nets with minimal NPS cost. The 8 variants add only 8 * (2048 + 4) = ~16KB of weights.

### 2. Threat-Aware Quiet History
- History indexed by `[from_is_threatened][to_is_threatened]`
- 4x more entries, better move ordering for tactical contexts

### 3. Node Cache (near-root move ordering)
- For ply < 3, tracks nodes spent on each move across iterations
- Boosts moves proportional to `nodesSearched / totalNodes`
- Also used in time management (best move node fraction)

### 4. Fail-High Score Blending
- On beta cutoffs: `bestValue = (bestValue * depth + beta) / (depth + 1)`
- Prevents inflated fail-high scores from propagating

### 5. Depth Reduction After Alpha Improvement
- `if (node->depth > 2) node->depth--` when alpha is raised
- Remaining moves searched at reduced depth after finding a good move

### 6. QSearch Beta Blending
- Stand-pat and end-of-qsearch returns are blended towards beta (~50/50)
- Dampens extreme qsearch values

### 7. Prior Counter-Move History Bonus
- Fail-low nodes give bonus to predecessor's continuation history
- "If all my responses are bad, the move that led here was good"

### 8. Accumulator Two-Stage Parent Update
- When current and parent accumulators are both dirty but share the same king bucket:
  first updates parent, then updates current from parent
- Siblings then get parent's cached accumulator for free

### 9. CanReachGameCycle (Cuckoo Hashing)
- In non-PV nodes with alpha < 0: checks if a drawing move exists via cuckoo tables
- Raises alpha to 0 if cycle is reachable

### 10. NUMA-Aware Threading
- Threads pinned to NUMA nodes
- Correction histories allocated per NUMA node
- ThreadData allocated on specific NUMA nodes

**Coda comparison**: GoChess has no NUMA awareness. This matters for multi-socket systems.

### 11. IIR on Stale TT Entries
- Fires not just on missing TT move, but when `ttEntry.depth + 4 < currentDepth`
- Reduces depth when TT entry exists but is too shallow to be useful

### 12. SEE Pruning Only on Threatened Squares
- `if (move.ToSquare() & allThreats)` guards SEE pruning
- Saves SEE computation on unthreatened target squares

### 13. History Bonus Scaled by Score Difference
- `scoreDiff = min(bestValue - beta, 256)` scaled into bonus formula
- Bigger beta cutoffs produce proportionally larger history updates

---

## 9. Aspiration Windows

- Initial: `6 + |prevScore| / 17`
- Fail-low: `beta = (alpha+beta+1)/2`, `alpha -= window`, restore depth
- Fail-high: `beta += window`, reduce depth by 1 (if depth > 1 and depth+5 > iterationDepth)
- Growth: `window += window / 3`
- Fallback: full window when `window > 547`

**Coda comparison**: GoChess uses delta=15 with doubling. Caissa starts smaller (6) with 33% growth. The asymmetric fail-low (beta contracts toward alpha) is interesting.

---

## 10. Lazy SMP

- Shared: only TT (lockless)
- Per-thread: MoveOrderer, NodeCache, AccumulatorCache, CorrectionHistories, search stack
- NUMA-aware thread pinning and memory allocation
- Best thread selection: picks thread with highest depth+score (prefers deeper, prefers mate)

---

## 11. Time Management

### Moves-Left Estimation (from Lc0)
- `f(moves) = 35 * (1 + 1.5 * (moves/35)^2.19)^(1/2.19) - moves`

### Ideal/Max Time
- `idealTime = 0.823 * (remaining / movesLeft + increment)`
- `maxTime = 4.50 * ((remaining - overhead) / movesLeft + increment)`
- Both clamped to `[0, 0.8 * remaining]`

### Dynamic Adjustments
- **Predicted move**: hit saves 8.5%, miss spends 13.2% more
- **PV stability**: factor from ~1.55 (unstable) to ~0.97 (stable 10+ iterations)
- **Node fraction**: `(1 - bestMoveNodeFraction) * 2.08 + 0.63`
- **Root singularity**: after 20% of ideal time, searches at depth/2 to see if best move is clearly singular

---

## 12. Priority Adoption Candidates for Coda/GoChess

| Feature | Difficulty | Expected Impact | Notes |
|---|---|---|---|
| Output bucketing (8 variants) | Medium (net retraining) | High | Free Elo at near-zero NPS cost |
| Threat-aware quiet history | Low (indexing change) | Medium | 4x table, proven in many engines |
| 6-ply continuation history | Low (extend existing) | Low-Medium | Weighted scoring/update |
| Non-pawn correction history | Low (add tables) | Medium | Split white/black non-pawn hash |
| Continuation correction history | Medium (new table type) | Low-Medium | Novel, worth testing |
| IIR on stale TT entries | Trivial | Low | One extra condition |
| SEE pruning only on threatened | Trivial | Low (NPS) | Saves SEE calls |
| History bonus with scoreDiff | Low | Low | One formula change |
| VNNI SIMD path | Medium | Low (hardware-dependent) | Only helps on Zen4+ / SPR |
| NMP only on cut nodes | Trivial | Unknown | Could help or hurt, test both |
| Prior conthist bonus on fail-low | Low | Low | Rewards good predecessor moves |
