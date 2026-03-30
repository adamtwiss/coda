# Stormphrax Chess Engine - Crib Notes

Source: `~/chess/engines/Stormphrax/`
Author: Ciekce
Version: 7.0.70
Strength: CCRL ~3500+, #8 in our RR at +210 Elo (318 above GoChess-v5)
Net: `net066_255_128_q6_nc.nnue`

---

## 1. NNUE Architecture

### Network: (704x16hm+60144 -> 640)x2 -> (32x2 -> 32 -> 1)x8

This is a highly advanced architecture with several cutting-edge features:

**Feature Transformer (Input Layer)**:
- **PSQ features**: 704 piece-square inputs per king bucket (12 pieces x 64 squares minus kings = approx, but 704 from the bucket config)
- **16 king buckets** with horizontal mirroring (ABCD-side only, so 4 files x ranks grouped). Layout:
  ```
  14 14 15 15    (ranks 7-8 share buckets)
  14 14 15 15
  12 12 13 13    (ranks 5-6 share buckets)
  12 12 13 13
   8  9 10 11    (ranks 3-4 share buckets)
   8  9 10 11
   4  5  6  7    (ranks 1-2, individual)
   0  1  2  3    (rank 0, individual)
  ```
- **Threat features**: 60,144 additional threat inputs (attacker x attackerSq x attacked x attackedSq, king-relative). These are incrementally updated alongside PSQ features, with separate accumulator.
- FT output: 640 neurons per perspective (i16)
- FT quantization: QA=255 (8-bit), scale bits=7

**Pairwise Activation (FT -> L1)**:
- Pairs first-half[i] * second-half[i] (within each perspective's 640 outputs, pairs 320 x 320)
- Clipped to [0, 255], multiplied, shifted right by 7 (scale bits)
- Output is u8 (unsigned 8-bit) -- ready for int8 L1 matmul
- **PSQ + threat accumulator values are summed before pairwise activation**
- Result: 640 u8 values (320 from stm + 320 from nstm)

**L1: 640 -> 32 (int8 weights, sparse)**:
- i8 quantized weights, dpbusd (VNNI) or equivalent SIMD
- **Sparse computation**: tracks non-zero 4-byte chunks in FT output, only multiplies non-zero entries
- 8 output buckets x 32 outputs
- Output undergoes **dual activation**: CReLU side (clamp [0,64], shift left 6) + SCReLU side (square then clip to 64^2)
- Effective L2 input: 64 values (32 CReLU + 32 SCReLU)

**L2: 64 -> 32 (i32 weights)**:
- Standard dense matmul, i32 weights
- Per output bucket

**L3: 32 -> 1 (i32 weights)**:
- CReLU activation (clamp to [0, 64^3])
- Per output bucket
- Final: `l3Out * 400 / (64^3)`

**Output Buckets**: 8, based on linear material count: `(popcount - 2) / 4`

**Refresh Table (FinnyTable)**:
- Per-king-bucket, per-color accumulator cache with stored bitboard state
- On king bucket change, computes delta (added/removed pieces) vs cached state, applies incrementally instead of full recompute
- Separate PSQ and threat refresh paths

### Compared to Our NNUE
| Feature | Stormphrax | GoChess |
|---------|-----------|---------|
| Input features | 704 PSQ + 60K threats per bucket | 12288 HalfKA (40960 total) |
| King buckets | 16 (mirrored, merged) | 16 |
| FT width | 640 | 256 |
| Hidden layers | 3 (640->32->32->1) | v5: shallow wide (Nx2->1x8, pairwise) |
| Pairwise | Yes (FT output) | Yes (FT output) |
| Dual activation | Yes (CReLU + SCReLU on L1 output) | CReLU or SCReLU (UPDATE: now supports both) |
| Threat features | 60,144 per bucket, incremental | None |
| Output buckets | 8 (linear material) | 8 (material-based) |
| FT quantization | 8-bit (QA=255) | 16-bit |
| L1 quantization | 7-bit (int8 weights) | int8 weights |
| Sparse L1 | Yes (NNZ chunk tracking) | No |
| Refresh table | Yes (FinnyTable) | Yes (Finny tables, UPDATE: merged) |
| Scale | 400 | varies |

**Key differences**: Stormphrax's threat feature inputs are unique -- they encode which pieces attack which other pieces, giving the network explicit tactical awareness. The dual CReLU+SCReLU activation on L1 doubles the effective input width to L2. The sparse L1 computation skips zero-valued FT outputs, saving significant compute. The FinnyTable reduces king-bucket-change cost from full recompute to incremental delta.

---

## 2. Search Architecture

Standard iterative deepening with aspiration windows and PVS. C++ templates for PV/root nodes (compile-time specialization). Lazy SMP with barrier-based thread coordination.

### Iterative Deepening
- Depth 1 to kMaxDepth
- Checks soft time limit after each completed depth
- Aspiration windows enabled at depth >= 3

### Aspiration Windows
- Initial delta: 16 (`initialAspWindow = 16`)
- On fail-low: `beta = (alpha + beta) / 2`, widen alpha by delta, reset aspReduction to 0
- On fail-high: widen beta by delta, **aspReduction += 1** (capped at 3). Search depth reduced by aspReduction
- Delta growth: `delta += delta * 17 / 16` (multiply by ~2.06 each iteration)
- **Optimism**: Per-side score bias based on running average: `optimismScale(150) * avgScore / (|avgScore| + optimismStretch(100))`

### Draw Detection
- `pos.isDrawn()` checks fifty-move and repetition
- `pos.hasCycle()` for upcoming repetition (Cuckoo hashing) -- checked early in search and qsearch, only when alpha < 0
- Draw scores randomized: `2 - (nodes % 4)` (range [-1, +2])

### Mate Distance Pruning
- Applied at all non-root nodes: `alpha = max(alpha, -MATE + ply)`, `beta = min(beta, MATE - ply - 1)`
- Cutoff if alpha >= beta

---

## 3. Pruning Techniques

### IIR (Internal Iterative Reduction)
- Conditions: `depth >= 3 && !excluded && (pvNode || cutnode) && (!ttMove || ttEntry.depth + 3 < depth)`
- Reduces depth by 1
- Note: triggered on PV/cut nodes when TT move is missing OR TT depth is too shallow

### Hindsight Reduction/Extension (Parent-Eval Feedback)
- **Extension (+1 depth)**: `parent.reduction >= 3 && parent.staticEval != NONE && curr.staticEval + parent.staticEval <= 0`
  - "If we were reduced and both sides think the position is bad, search deeper"
- **Reduction (-1 depth)**: `depth >= 2 && parent.reduction >= 2 && parent.staticEval != NONE && curr.staticEval + parent.staticEval >= 200`
  - "If we were reduced and both sides think the position is good (quiet), search shallower"
- Compare to ours: We don't have this. 5+ engines use it. Alexandria's variant is simplest.

### Reverse Futility Pruning (RFP)
- Conditions: `!pvNode && !inCheck && !excluded`
- Depth guard: `depth <= 6`
- Margin: `71 * max(depth - improving, 0)` (`rfpMargin = 71`)
- **Complexity adjustment**: `margin += corrDelta * 64 / 128` where corrDelta = |corrected_eval - raw_eval|
  - Higher correction magnitude = higher margin = more conservative pruning
- Return: `(staticEval + beta) / 2` when not decisive (score blending), else staticEval
- Compare to ours: We use 80*d (from futility tightening). They have complexity adjustment (novel) and RFP return blending.

### Razoring
- Conditions: `!pvNode && !inCheck && !excluded`
- Depth guard: `depth <= 4`
- Margin: `315 * depth` (`razoringMargin = 315`), also `abs(alpha) < 2000`
- Formula: `staticEval + 315 * depth <= alpha` => drop to qsearch with null window
- Compare to ours: We use 400 + depth*100 (depth 1: 500, depth 2: 600). They use 315*d (depth 1: 315, depth 2: 630). Tighter at low depth.

### Null Move Pruning (NMP)
- Conditions: `depth >= 4 && ply >= minNmpPly && staticEval >= beta && !parent.move.isNull() && !(ttFlag==UPPER && ttScore < beta) && hasNonPawnMaterial`
- Reduction: `R = 6 + depth / 5` (flat formula, no eval-based component)
- **Verification search at depth > 14**: if NMP score >= beta AND (depth <= 14 || minNmpPly > 0), return. Otherwise set `minNmpPly = ply + (depth - R) * 3/4` and run verification search at depth-R
- Returns `beta` for winning scores, raw score otherwise
- Compare to ours: We use `R = 3 + depth/3 + min((eval-beta)/200, 3)`. They use `R = 6 + depth/5` (simpler, larger base). Our min depth is 3, theirs is 4.

### ProbCut
- Conditions: `!ttpv && depth >= 7 && !decisive(beta) && (!ttMove || ttMoveNoisy) && !(ttHit && ttDepth >= probcutDepth && ttScore < probcutBeta)`
- probcutBeta: `beta + 303` (`probcutMargin = 303`)
- probcutDepth: `max(depth - 3, 1)`
- **SEE threshold**: `(probcutBeta - staticEval) * 17 / 16` -- dynamic, eval-scaled
- **Two-stage**: QSearch first, then full search only if QSearch passes (`-qsearch >= probcutBeta` -> negamax)
- Compare to ours: We use margin 170, gate 85. They use margin 303 (much wider). They have the QS pre-filter that Alexandria also uses (2 engines now). The dynamic SEE threshold is unique.

### Late Move Pruning (LMP)
- Table-based: `[improving][depth]` where threshold = `(3 + depth^2) / (2 - improving)`
  - depth 1: 2 (not imp) / 4 (imp)
  - depth 5: 14 (not imp) / 28 (imp)
  - depth 15: 114 (not imp) / 228 (imp)
- Max depth: 15 (table extends to 15)
- Triggers `skipQuiets()` (staged generator skips all remaining quiet moves)
- Compare to ours: We use `3 + d^2` (+50% improving). Same formula essentially.

### Futility Pruning
- Conditions: `!inCheck && lmrDepth <= 8 && abs(alpha) < 2000`
- Margin: `fpMargin(261) + depth * fpScale(68) + history / fpHistoryDivisor(128)`
  - depth 1: 329 + history/128
  - depth 5: 601 + history/128
- Formula: `staticEval + margin <= alpha` => `skipQuiets()`
- **History-adjusted**: moves with good history have wider futility margin (less likely to be pruned)
- Compare to ours: We use 80 + lmrDepth*80 (from tightening). They use 261 + depth*68, which is wider. The history adjustment is novel.

### History Pruning (Quiet)
- Conditions: quiet, `lmrDepth <= 5`
- Threshold: `history < -2314 * depth + (-1157)` (`quietHistPruningMargin = -2314, quietHistPruningOffset = -1157`)
  - depth 1: history < -3471
  - depth 3: history < -8099
- Triggers `skipQuiets()`
- Compare to ours: We use -2000*depth. They use -2314*d - 1157 (tighter with offset).

### History Pruning (Noisy/Capture)
- Conditions: noisy, `depth <= 4`
- Threshold: `history < -1000 * depth^2 + (-1000)` (`noisyHistPruningMargin = -1000, noisyHistPruningOffset = -1000`)
  - depth 1: -2000, depth 2: -5000, depth 3: -10000
- Per-move continue (skip individual capture)

### SEE Pruning
- Conditions: `quietOrLosing` stage (past good noisy phase), not best score is loss
- Quiet threshold: `-16 * lmrDepth^2` (`seePruningThresholdQuiet = -16`)
  - lmrDepth 1: -16, lmrDepth 3: -144, lmrDepth 5: -400
- Noisy threshold: `min(-112 * depth - history / 64, 0)` (`seePruningThresholdNoisy = -112, seePruningNoisyHistDivisor = 64`)
  - History adjustment: captures with good history get a better SEE threshold
- Compare to ours: We use -20*d^2 for quiets. They use -16*lmrDepth^2 (similar but on lmrDepth). Noisy SEE with history adjustment is novel.

### QSearch Pruning
- **Stand-pat beta blending**: `(eval + beta) / 2` when not decisive (same as QS beta blending, MERGED in GoChess at +4.9 Elo)
- **QSearch futility**: `futility = eval + 135`. If `futility <= alpha && !SEE(move, 1)`, skip. Also prune stand_pat = futility.
- **QSearch SEE threshold**: All moves checked with `SEE(move, -97)` (`qsearchSeeThreshold = -97`)
- **QSearch move limit**: `legalMoves >= 2` => break. Only 2 captures searched in QS!
- **QSearch quiet moves**: Searched in evasions. Also searched when TT has a quiet best move with lower bound flag (`!pvNode && ttMove && ttFlag != UPPER && !isNoisy(ttMove)`).
- Compare to ours: We have QS beta blending. The 2-move QS limit is aggressive and unique.

---

## 4. Extensions

### Singular Extensions
- Depth guard: `depth >= 6 + ttpv` (6 for non-PV, 7+ for ttpv nodes, effectively 6-7)
- TT conditions: `ttEntry.depth >= depth - 5 && ttFlag != UPPER && !decisive(ttScore)`
- sBeta margin: `ttScore - depth * sBetaBaseMargin(14) / 16`, with `+ sBetaPrevPvMargin(16) * (ttpv && !pvNode)`
  - Effective: `ttScore - depth * 14/16` normally, `ttScore - depth * 30/16` on prev-PV non-PV nodes
- Singularity depth: `(depth - 1) / 2`
- Ply limiter: `ply < rootDepth * 2`

**Extension levels**:
- **Single (+1)**: score < sBeta
- **Double (+2)**: `!pvNode && score < sBeta - 11` (`doubleExtMargin = 11`)
- **Triple (+3)**: `!pvNode && !ttMoveNoisy && score < sBeta - 105` (`tripleExtMargin = 105`)
- **Multi-cut**: `!pvNode && score >= beta` => return `(score + beta) / 2` (blended) or score if decisive
- **Negative extension (-2)**: cutnode (when not singular)
- **Negative extension (-1)**: `ttEntry.score >= beta` (when not singular, not cutnode)
- After negative extension: `cutnode |= true`

### LDSE (Low-Depth Singular Extension fallback)
- When SE conditions not met: `depth <= 7 && !inCheck && staticEval <= alpha - 26 && ttFlag == LOWER`
- Extension: +1
- "If we're at low depth, eval is below alpha, but TT says this move is a fail-high, extend it"

### No Check Extension
- No unconditional check extension (unlike Midnight)

Compare to ours: Our SE is broken (tested at -58 to -85 Elo). Stormphrax has the full modern SE suite: double/triple extensions, multi-cut, negative extensions, ply limiter, prev-PV-aware margin. These are the highest-priority fixes for GoChess.

---

## 5. LMR (Late Move Reductions)

### Table Initialization
Two separate tables (stored as fixed-point * 128):
- **Quiet**: `128 * (0.83 + ln(depth) * ln(moves) / 2.18)` (base=83/100, divisor=218/100)
- **Noisy**: `128 * (-0.12 + ln(depth) * ln(moves) / 2.48)` (base=-12/100, divisor=248/100)

Compare to ours: We use quiet C=1.50 / capture C=1.80 (already split, MERGED +43.5 Elo). Their quiet base 0.83 is lower than our ~1.50, meaning more base reduction for quiets. Their noisy base -0.12 is negative, meaning captures start with essentially 0 reduction.

### Application Conditions
- `depth >= 2 && legalMoves >= 2 + kRootNode` (root: move 3+, non-root: move 2+)
- No explicit in_check guard (but checks not extended, so this is implicit)

### Reduction Adjustments (all scaled by /128)
- `+lmrNonPvReductionScale(131)` for non-PV nodes (~+1.02 ply)
- `-lmrTtpvReductionScale(130)` for ttpv nodes (~-1.02 ply)
- `-history * 128 / lmrQuietHistoryDivisor(10835)` for quiets, or `/ lmrNoisyHistoryDivisor(10835)` for noisies
- `-lmrImprovingReductionScale(148)` if improving (~-1.16 ply)
- `-lmrCheckReductionScale(111)` if gives check (~-0.87 ply)
- `+lmrCutnodeReductionScale(257)` at cut nodes (~+2.01 ply) -- **very aggressive**
- `+lmrTtpvFailLowReductionScale(128)` if ttpv and ttScore <= alpha (~+1.00 ply)
- `+lmrAlphaRaiseReductionScale(64) * alphaRaises` (~+0.50 per alpha raise)
- `-lmrHighComplexityReductionScale(128)` if complexity > 70 (~-1.00 ply)
- Additional: `+pvNode + (ttpv && r < -128)` as extension (after clamping)

### Clamping
- `min(max(newDepth - r/128, 1), newDepth) + pvNode + ttPvExtension`
- Never reduces below depth 1; PV nodes get +1

### DoDeeper / DoShallower (after LMR re-search)
- **DoDeeper**: `score > bestScore + 38 + 4 * newDepth` (`lmrDeeperBase = 38, lmrDeeperScale = 4`)
- **DoShallower**: `score < bestScore + newDepth`
- `newDepth += doDeeperSearch - doShallowerSearch`

### Post-LMR Continuation History Update
- If `!noisy && (score <= alpha || score >= beta) && !stopped`: update conthist with bonus/penalty based on outcome
- "Quiet moves that either fail low or cause cutoff after LMR re-search deserve conthist reinforcement"

Compare to ours: Their LMR is significantly more sophisticated. Key differences:
1. Cutnode +2.01 ply reduction (we tested +2 as separate experiment)
2. Alpha-raise count in reduction (novel, 2 engines now)
3. Complexity-based reduction (novel)
4. TT-PV fail-low extra reduction
5. TT-PV extension when reduction is negative
6. Post-LMR conthist update
7. History divisor 10835 vs our 5000 (less history influence)

---

## 6. Move Ordering

### Staged Move Generator
Stages: TT Move -> Gen Noisy -> Good Noisy -> Killer -> Gen Quiet -> Quiet -> Bad Noisy

### Score Hierarchy

1. **TT move**: Returned directly from stage, no scoring needed
2. **Good noisy (captures)**: `noisyHistory / 8 + see::value(captured)` (MVV + capHist/8). Filtered by dynamic SEE: `-score/4 + goodNoisySeeOffset(15)`. Captures failing SEE go to bad noisy list.
3. **Killer**: Single killer per ply (returned directly)
4. **Quiet moves**: `mainHist + contHist`
   - mainHist = `(butterfly + pieceTo) / 2` (average of two tables)
   - contHist = `contHist[ply-1] + contHist[ply-2] + contHist[ply-4] / 2`
5. **Bad noisy**: Moves that failed good-noisy SEE threshold

### History Tables

**Butterfly history**: `[64][64][2][2]` = from, to, fromAttacked, toAttacked
- **4x threat-indexed** (from-threatened x to-threatened booleans)
- This is the threat-aware history that 12+ engines now use

**PieceTo history**: `[12][64][2][2]` = piece, to, fromAttacked, toAttacked
- Same 4x threat indexing as butterfly
- mainHist = average of butterfly and pieceTo

**Continuation history**: `[12][64][12][64]` = prevPiece, prevTo, currPiece, currTo
- Read at plies 1, 2, and 4 (half weight at ply 4)
- Write at plies 1, 2, and 4 (with base-relative update)
- **Base-relative update**: `bonus_applied = bonus - base * |bonus| / maxHistory` where `base = contHist + mainHist/2`
  - Moves with already-high scores get smaller incremental updates (convergence control)

**Capture/Noisy history**: `[64][64][13][2]` = from, to, captured (+1 for promo), defended
- Defended flag: `threats[move.toSq()]`
- Used in noisy scoring (`noisyHistory / 8`), noisy history pruning, noisy SEE pruning adjustment

**Killers**: 1 slot per ply (single killer, not 2)

**History bonus**: `min(depth * 280 - 432, 2576)` (`maxHistoryBonus = 2576, historyBonusDepthScale = 280, historyBonusOffset = 432`)
**History penalty**: `-min(depth * 343 - 161, 1239)` (`maxHistoryPenalty = 1239`)
**Gravity**: `entry += bonus - entry * |bonus| / maxHistory(15769)`

**History depth bonus**: `historyDepth = depth + (!inCheck && staticEval <= bestScore)` -- +1 depth for surprising cutoffs (from Alexandria)

Compare to ours:
| Feature | Stormphrax | GoChess |
|---------|-----------|---------|
| Butterfly | [64][64][2][2] (threat-aware) | [2][64][64] (color only) |
| PieceTo | [12][64][2][2] (threat-aware) | None |
| ContHist plies | 1, 2, 4 (half at 4) | 1, 2 |
| Capture hist | [64][64][13][2] (defended) | [12][64][12] |
| Killers | 1 per ply | 2 per ply |
| Counter-move | None | Yes |
| Max history | 15769 | ~10000 |
| History bonus | min(d*280-432, 2576) | similar |
| Gravity | entry - entry*|bonus|/15769 | entry - entry*|bonus|/divisor |

Key differences: Threat-aware butterfly (4x table), PieceTo table, ply-4 contHist, defended-flag in capture history. They lack counter-move heuristic (we have it).

---

## 7. Correction History

### 7 Sources (4 positional + 3 continuation)
1. **Pawn hash** (`pawnKey % 16384`) -- weight 133
2. **Black non-pawn Zobrist key** (`blackNonPawnKey % 16384`) -- STM weight 142 / NSTM weight 142
3. **White non-pawn Zobrist key** (`whiteNonPawnKey % 16384`) -- STM weight 142 / NSTM weight 142
4. **Major piece key** (`majorKey % 16384`) -- weight 129
5. **Continuation correction ply 1** (`key ^ keyHistory[-1] % 16384`) -- weight 128
6. **Continuation correction ply 2** (`key ^ keyHistory[-2] % 16384`) -- weight 192
7. **Continuation correction ply 4** (`key ^ keyHistory[-4] % 16384`) -- weight 192

### Weighted Blend
- `correction = sum(weight_i * table_i[hash]) / 2048`
- Per-color non-pawn keys (STM gets stmNonPawnCorrhistWeight, NSTM gets nstmNonPawnCorrhistWeight)
- Both weights are 142, but they distinguish STM from NSTM for the black/white keys

### Update
- `bonus = clamp((searchScore - staticEval) * depth / 8, -256, 256)`
- Gravity: `v += bonus - v * |bonus| / 1024`
- Entries are atomic i16 with relaxed ordering
- Updates all 7 tables on every qualifying node

### Complexity Signal
- `corrDelta = |eval_before_correction - eval_after_correction|`
- Used in RFP margin adjustment and LMR reduction adjustment
- High complexity = big correction = uncertain eval

Compare to ours: We only have pawn hash correction. They have 7 sources. This is the #1 priority retry: implement per-color non-pawn keys + continuation correction. Our XOR-bitboard approach failed (-11.8 Elo); proper Zobrist-based keys are needed.

---

## 8. Eval Pipeline

1. **NNUE inference** -> raw eval (i32)
2. **Contempt**: `eval += contempt[stm]`
3. **Material scaling**: `eval = eval * (26500 + npMaterial) / 32768`
   - npMaterial = 450*N + 450*B + 650*R + 1250*Q
   - **Optimism**: `optimism[stm] * (2000 + npMaterial) / 32768` added
4. **50-move decay**: `eval = eval * (200 - halfmove) / 200`
5. **Correction history**: weighted blend of 7 tables
6. **Clamp** to `[-kScoreWin+1, kScoreWin-1]`

Compare to ours: We don't have material scaling, 50-move decay, or optimism. These are all proven in multiple engines.

---

## 9. Transposition Table

### Structure
- 3 entries per cluster, 32-byte aligned clusters
- Entry: 10 bytes (key16, score16, staticEval16, move16, offsetDepth8, agePvFlag8)
- Index: `(key * clusterCount) >> 64` (Fibonacci hashing, not modulo)
- Huge page support (madvise MADV_HUGEPAGE)

### Replacement Policy
Replace if ANY of:
- Flag is EXACT
- Different key (new position)
- Age mismatch (stale entry)
- `depth + 4 + pv*2 > entry.depth` (depth threshold with PV bonus)

### TT Cutoff Conditions
- `!pvNode && ttEntry.depth >= depth && (ttEntry.score <= alpha || cutnode)`
  - AND standard bound checks (exact, upper+alpha, lower+beta)
- The **`cutnode` guard** is notable: at cut nodes, accept TT cutoffs even when score is between alpha and beta (more aggressive)
- **TT cutoff history update**: On beta cutoff with quiet TT move, give it a history bonus

### Static Eval Storage
- `putStaticEval()` stores raw static eval with depth=-6 (below any real depth)
- Used to avoid recomputing static eval at revisited nodes

Compare to ours: We use 4-slot buckets with lockless atomics. They use 3-entry clusters with age/PV replacement. Their cutnode guard on TT cutoffs is novel. We should test `cutNode == (ttScore >= beta)` (Alexandria-style).

---

## 10. Time Management

### Base Allocation
- `movesToGo` = provided or `defaultMovesToGo(19)`
- `baseTime = remaining / movesToGo + increment * 0.83`
- `optTime = min(baseTime * 0.68, maxTime)`
- `maxTime = remaining * 0.56`

### Multi-Dimensional Scaling
Three multiplicative factors, all tuned:

**1. Node-Based Scaling (Best Move Node Fraction)**:
- `bestMoveNodeFraction = pvMove.nodes / totalNodes`
- `scale *= max(2.63 - fraction * 1.7, 0.102)` (nodeTmBase, nodeTmScale, nodeTmScaleMin)
- If best move uses 50% of nodes: 1.78x. If 90%: 1.10x. If 10%: 2.46x.

**2. Best Move Stability**:
- Tracks consecutive iterations with same best move (counter, +1 same / reset to 1 on change)
- `scale *= min(2.4, 0.75 + 9.11 * (stability + 0.8)^(-2.7))` (power law)
- Stability 1: ~2.4x (capped). Stability 5: ~0.85x. Stability 10+: ~0.76x.

**3. Score Trend (Running Average)**:
- Maintains exponential moving average of score (`ilerp<8>`: 1/8 new + 7/8 old)
- `scoreChange = (score - avgScore) / 4.58`
- `invScale = scoreChange * 0.41 / (|scoreChange| + 0.8) * (positive ? 1.09 : 1.04)`
- `scale *= clamp(1.0 - invScale, 0.6, 1.7)`
- Score dropping: extend time. Score rising: save time.

### Final Scale
- `scale = max(combined_scale, 0.07)` (minimum 7% of optTime)
- `stopSoft = time >= optTime * scale`
- `stopHard = time >= maxTime`

### Hard Time Check
- Every 1024 nodes (kTimeCheckInterval)

Compare to ours: We use instability factor (200) based on best move changes. They have a full 3-factor system. This is significantly more sophisticated. Node-based TM alone is estimated at +5-15 Elo across multiple engines. **(UPDATE 2026-03-21: GoChess now has score-drop time extension with 2.0x/1.5x/1.2x tiered scaling, addressing the "score is dropping" case. Node-based TM is still missing.)**

---

## 11. Lazy SMP

### Thread Coordination
- Barrier-based synchronization (not just shared TT)
- Thread data fully per-thread: SearchData, Position, NNUE state, history, search stack
- Correction history shared via NUMA-aware allocation (atomic entries)
- Main thread (id 0) does reporting and time management

### Thread Selection (for bestmove)
- SF-ported algorithm: weighted vote by `(score - lowestScore + 10) * depthCompleted`
- Each thread votes for its best move
- Highest-voted move wins; ties broken by thread weight
- Special handling for decisive scores (wins/losses)

### Node Counting
- Per-thread atomic relaxed loads/stores (no fetch_add overhead)
- Total nodes summed across threads for time management

Compare to ours: We have Lazy SMP with shared TT only. They share correction history atomically across threads. Their thread selection algorithm is more sophisticated than just using main thread's result.

---

## 12. Unique Techniques

### 1. Threat Feature Inputs in NNUE (60,144 features)
- Encodes piece-attacks-piece relationships as direct NNUE inputs
- Incrementally updated alongside PSQ features
- Separate accumulator for threat features, added to PSQ accumulator before pairwise activation
- **No other engine in our review has this** -- completely unique to Stormphrax

### 2. Dual CReLU+SCReLU Activation (L1)
- L1 output goes through both CReLU (linear clamp) and SCReLU (square clamp)
- Doubles effective L2 input width (32 CReLU + 32 SCReLU = 64 inputs)
- Captures both linear and nonlinear patterns from same hidden layer

### 3. Sparse L1 Computation
- Tracks non-zero 4-byte chunks in u8 FT output
- Only multiplies chunks where input is non-zero
- Processes 4 chunks at a time for SIMD efficiency

### 4. Optimism (Eval Bias)
- Per-side bias based on running average score: `150 * avgScore / (|avgScore| + 100)`
- Material-weighted: `optimism * (2000 + npMaterial) / 32768`
- Encourages the engine to play for wins when ahead, draws when behind

### 5. Dynamic ProbCut SEE Threshold
- `seeThreshold = (probcutBeta - staticEval) * 17 / 16`
- Captures must gain enough material to plausibly exceed probcut margin
- Adapts to the gap between static eval and required score

### 6. QSearch 2-Move Limit
- `if (legalMoves >= 2) break;`
- Only searches 2 captures in quiescence (after futility/SEE filtering)
- Very aggressive but saves significant time

### 7. Alpha-Raise Count in LMR
- Tracks `alphaRaises` counter at each node
- Each alpha raise adds `64/128 = 0.5` ply of reduction to subsequent moves
- "If we've already found good moves, remaining moves are less likely to be good"

### 8. Cuckoo Hashing for Upcoming Repetition
- `pos.hasCycle()` detects upcoming repetitions without full search
- Uses Cuckoo hash table of Zobrist key differences
- Called early in search (before depth check) and in QSearch
- Only checked when `alpha < 0` (can only help the side with negative alpha)

---

## 13. Notable Differences from GoChess

### Things Stormphrax has that we don't:
1. **Threat feature NNUE inputs** (60K features, incremental) -- unique in chess engines
2. **Dual CReLU+SCReLU activation** on L1 output
3. **Sparse L1 computation** (NNZ chunk tracking)
4. ~~**FinnyTable**~~ (refresh table for king bucket changes) **(UPDATE 2026-03-21: GoChess now has Finny tables)**
5. **7-source correction history** (pawn + 2 non-pawn + major + 3 continuation)
6. **Threat-aware butterfly history** (4x table: fromThreatened x toThreatened)
7. **PieceTo history table** (averaged with butterfly for mainHist)
8. **Hindsight extension/reduction** (parent eval + child eval feedback)
9. **Optimism** (per-side eval bias)
10. **Material scaling** of eval
11. **50-move decay** of eval
12. **Cuckoo hashing** for upcoming repetition detection
13. **3-factor time management** (node fraction + stability + score trend)
14. **ProbCut QS pre-filter** (2-stage ProbCut)
15. **Dynamic ProbCut SEE threshold**
16. **Alpha-raise count in LMR**
17. **Complexity-adjusted RFP and LMR**
18. **RFP return blending** (`(eval + beta) / 2`)
19. **Full singular extension suite** (double/triple ext, multi-cut, negative ext, ply limiter)
20. **LDSE** (low-depth singular extension fallback)
21. **QSearch 2-move limit**
22. **History-adjusted noisy SEE pruning threshold**
23. **Eval-based history depth bonus** (`depth + (eval <= bestScore)`)
24. **Post-LMR conthist update**

### Things we have that Stormphrax doesn't:
1. **Counter-move heuristic** (they only have 1 killer)
2. **2 killers per ply** (they have 1)
3. **NMP threat detection** (our +12.4 Elo win)
4. **Alpha-reduce** (our +13.0 Elo win)
5. **TT score dampening** (our +22.1 Elo win)
6. **TT near-miss cutoffs** (our +21.7 Elo win)
7. **Failing heuristic** (our eval deterioration detection)

---

## 14. Parameter Comparison Table

| Feature | Stormphrax | GoChess |
|---------|-----------|---------|
| RFP margin | 71*max(d-imp,0), depth<=6 | 80*d, depth<=8 |
| RFP complexity | Yes (corrDelta*64/128) | No |
| RFP return | (eval+beta)/2 | eval |
| Razoring | 315*d, depth<=4 | 400+100*d, depth<=3 |
| NMP min depth | 4 | 3 |
| NMP R formula | 6+d/5 | 3+d/3+min((eval-beta)/200,3) |
| NMP verification | depth > 14 | depth >= 12 |
| ProbCut margin | 303, depth>=7 | 170, depth>=5 |
| ProbCut QS pre-filter | Yes | No |
| LMR quiet base | 0.83 | 1.50 |
| LMR quiet divisor | 2.18 | ~2.36 |
| LMR noisy base | -0.12 | ~1.80 |
| LMR noisy divisor | 2.48 | ~2.36 |
| LMR cutnode | +2.01 ply | No |
| LMR alpha-raise | +0.50/raise | No |
| LMR complexity | -1.00 if high | No |
| LMR history div | 10835 | 5000 |
| LMP formula | (3+d^2)/(2-imp) | 3+d^2 (+50% imp) |
| Futility | 261+68*d+hist/128, d<=8 | 80+80*lmrD |
| SEE quiet | -16*lmrD^2 | -20*d^2 |
| SEE noisy | -112*d-hist/64 | similar |
| SE depth | >=6+ttpv | >=10 |
| SE margin | ttScore-d*14/16 | ttScore-d*3 |
| SE double/triple | Yes (margins 11, 105) | No |
| SE multi-cut | Yes (return (score+beta)/2) | No |
| SE negative ext | -2 cutnode, -1 ttScore>=beta | No |
| Aspiration delta | 16 | 15 |
| Asp fail-high | reduce depth up to 3 | No |
| History max | 15769 | ~10000 |
| Killers | 1 per ply | 2 per ply |
| Counter-move | No | Yes |
| ContHist plies | 1, 2, 4 | 1, 2 |
| Correction sources | 7 | 1 (pawn) |
| Time: movesToGo | 19 | similar |
| Time: soft scale | 0.68 | similar |
| Time: hard scale | 0.56 | similar |
| Time: node-based | Yes (3-factor) | Instability only |

---

## 15. Ideas Worth Testing from Stormphrax

### Already Merged from Stormphrax (via earlier reviews):
- LMR separate tables (cap/quiet) -- +43.5 Elo
- Fail-high score blending -- +14.7 Elo
- QS beta blending -- +4.9 Elo

### Priority 1: Fix Singular Extensions (CRITICAL)
Our SE is broken at -58 to -85 Elo. Stormphrax's full suite confirms the Alexandria pattern:
- Depth >= 6 (not 10), ply limiter `ply < rootDepth * 2`
- Tighter margin: `ttScore - depth*14/16` (not `depth*3`)
- Multi-cut return on `score >= beta`: `(score+beta)/2`
- Double ext at sBeta-11, triple at sBeta-105
- Negative ext -2 at cutnode, -1 when ttScore >= beta
- LDSE fallback at low depth
- **Est. Elo**: +20 to +50 (recovering from -58 is massive)

### Priority 2: Multi-Source Correction History
7 tables vs our 1. This is now confirmed by 11+ engines.
- Per-color non-pawn Zobrist keys (not XOR of bitboards)
- Continuation correction at plies 1, 2, 4
- Major piece key
- Weighted blend with tuned weights
- **Est. Elo**: +5 to +15

### Priority 3: Node-Based Time Management
Their 3-factor system (node fraction + stability + score trend) is much richer than our instability-only approach.
- Start with node fraction scaling: `max(2.63 - fraction * 1.7, 0.102)`
- Add score trend: running average with asymmetric scaling
- 6+ engines now
- **Est. Elo**: +5 to +15

### Priority 4: Threat-Aware Butterfly History
4x table indexed by [from][to][fromAttacked][toAttacked]. 12+ engines now.
- Also: PieceTo table averaged with butterfly
- **Est. Elo**: +3 to +8

### Priority 5: Hindsight Reduction/Extension
Parent eval + child eval feedback. 5 engines now.
- Extension when both evals negative AND parent reduced >= 3
- Reduction when both evals positive AND parent reduced >= 2
- **Est. Elo**: +2 to +5

### Priority 6: ProbCut QS Pre-Filter
Two-stage ProbCut (QS first, then full search). 2 engines (Stormphrax + Alexandria).
- Our ProbCut is already proven (+10 Elo). Adding QS pre-filter saves nodes in deep ProbCut.
- Dynamic SEE threshold: `(probcutBeta - staticEval) * 17/16`
- **Est. Elo**: +3 to +8

### Priority 7: Cutnode +2 LMR
They use +2.01 ply at cut nodes (lmrCutnodeReductionScale = 257). 3 engines now.
- We are currently testing this.
- **Est. Elo**: +2 to +5

### Priority 8: Complexity-Adjusted RFP
Use |corrected_eval - raw_eval| as complexity signal. Unique to Stormphrax.
- Higher complexity = larger RFP margin = less pruning
- Simple: `rfpMargin += corrDelta * 64 / 128`
- **Est. Elo**: +2 to +4

### Lower Priority (from Stormphrax, confirm with other engines):
- Alpha-raise count in LMR (2 engines now: Stormphrax + Altair)
- History-adjusted noisy SEE pruning
- QSearch 2-move limit (unique, risky)
- LDSE (low-depth SE fallback)
- Eval-based history depth bonus (also in Alexandria)
- Optimism (eval bias) -- interesting but complex
- Material scaling + 50-move decay (also in Alexandria)
- Post-LMR conthist update
- Cuckoo repetition detection

---

## 16. NNUE Architecture Lessons

Stormphrax's NNUE is substantially more advanced than ours. Key architectural lessons:

1. **Threat features are powerful**: 60K explicit threat inputs give the network tactical awareness without needing to learn it purely from position. This is unique and likely accounts for significant strength.

2. **Dual activation doubles effective width cheaply**: CReLU+SCReLU on L1 output gives 64 inputs to L2 from 32 neurons. The linear (CReLU) path captures magnitude; the squared (SCReLU) path captures nonlinear interactions.

3. **Sparse L1 is essential at this width**: With 640 FT outputs, many will be zero after pairwise activation. Sparse computation saves ~30-50% of L1 time.

4. **FinnyTable amortizes king bucket changes**: Instead of full recompute on king moves, delta-update from cached state. Critical at 16 king buckets. **(UPDATE 2026-03-21: GoChess now has this.)**

5. **Our v5 architecture** (`(768x16->N)x2->1x8` with SCReLU/pairwise/Finny tables) is now at Stormphrax's scale. **(UPDATE 2026-03-21: GoChess v5 has pairwise mul, SCReLU, dynamic width, and Finny tables.)** Still lacks threat features, dual activation, and NNZ sparsity.
