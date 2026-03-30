# Caissa Engine - Crib Notes

Source: `~/chess/engines/Caissa/src/backend/`

All parameter values are SPRT-tuned via `DEFINE_PARAM` macros. Values listed are the tuned defaults.

---

## 1. Search: Pruning Techniques

### Reverse Futility Pruning (RFP)
- **Depth guard**: `depth <= 6` (RfpDepth=6)
- **Margin**: `83*depth + 0*depth^2 - 145*improving_and_no_opp_material_threat`
  - RfpDepthScaleLinear=83, RfpDepthScaleQuad=0, RfpImprovingScale=145
  - "improving" only counts if opponent cannot win material (checks `OppCanWinMaterial` using threat bitboards)
- **Floor**: margin is clamped to at least RfpTreshold=16
- **Return value**: blended `(eval * (1024-525) + beta * 525) / 1024` (not just eval or beta)
- **Conditions**: non-PV, not in check, no filtered move, eval <= KnownWinValue

### Razoring
- **Depth guard**: `depth <= 4` (RazoringStartDepth=4)
- **Margin**: `22 + 158 * depth` (bias=22, multiplier=158)
- **Mechanism**: if `eval + margin < beta`, do qsearch; return qscore if still < beta
- **Conditions**: non-PV, not in check, beta < KnownWinValue

### Null Move Pruning (NMP)
- **Depth guard**: `depth >= 3` (NmpStartDepth=3)
- **Eval condition**: `eval >= beta + (depth < 4 ? 16 : 0)` (NmpDepthTreshold=4, NmpEvalTreshold=16)
- **Additional**: `staticEval >= beta` AND side has non-pawn material
- **Only on cut nodes** (`node->isCutNode`)
- **No consecutive null moves**: checks both parent and grandparent
- **Reduction R**: `3 + depth/3 + min(3, (eval-beta)/85) + improving`
  - NmpNullMoveDepthReduction=3, NmpEvalRedDiv=3, NmpEvalDiffDiv=85
- **Verification re-search**: if null move score >= beta:
  - If `abs(beta) < KnownWinValue && depth < 10`: return immediately
  - Otherwise: reduce depth by 5 (NmpReSearchDepthReduction=5) and continue searching

### ProbCut
- **Depth guard**: `depth >= 5` (ProbcutStartDepth=5)
- **Beta**: `beta + 133` (ProbcutBetaOffset=133)
- **Conditions**: non-PV, not in check, abs(beta) < TablebaseWinValue
- **Skips if TT entry with sufficient depth has score < probBeta**
- **SEE threshold**: `probBeta - staticEval` (dynamic, position-dependent)
- **Process**: tries captures with sufficient SEE, does qsearch first, then `depth-4` NegaMax verification
- **TT write on success**: stores at `depth-3` with Lower bound

### In-Check ProbCut (from Stockfish)
- **Non-PV only**, when in check and TT move is a capture
- **Beta**: `beta + 329` (ProbcutBetaOffsetInCheck=329)
- **Conditions**: TT has lower bound, `ttEntry.depth >= depth - 4`, ttScore >= probCutBeta
- **Returns probCutBeta directly** (no search needed)

### Futility Pruning (quiet moves)
- **Depth guard**: `depth < 9` (FutilityPruningDepth=9)
- **Margin**: `staticEval + 32*depth^2 + moveStatScore/383`
  - FutilityPruningScale=32, FutilityPruningStatscoreDiv=383
  - Note: uses depth^2 scaling (quadratic), not linear
- **Conditions**: not in check, quiet or underpromotion move
- **On trigger**: calls `movePicker.SkipQuiets()` to stop generating more quiets
- **Exception**: first quiet move (quietMoveIndex > 1 check) still tried

### Late Move Pruning (LMP)
- **Threshold**: `4 + depth^2` when improving, `4 + depth^2/2` otherwise
  - LateMovePruningBase=4, also adds `LateMovePruningPVScale * isPvNode` (=2) to depth
- **Conditions**: not root, bestValue > -KnownWinValue, side has non-pawn material
- **Applies to**: quiet moves and underpromotions
- **Optimization**: if in quiets stage, breaks out entirely rather than continuing

### History Pruning
- **Depth guard**: `depth < 9` (HistoryPruningMaxDepth=9)
- **Threshold**: `0 - 234*depth - 148*depth^2`
  - HistoryPruningLinearFactor=234, HistoryPruningQuadraticFactor=148
- **Conditions**: quietMoveIndex > 1, moveStatScore below threshold
- **Uses moveStatScore** (combined main history + conthist[0] + conthist[1] + conthist[3])

### SEE Pruning
- **Only when move target is on a threatened square** (`move.ToSquare() & allThreats`)
- **Captures**: `depth <= 5`, score < GoodCapture, SEE threshold = `-120*depth`
  - SSEPruningDepth_Captures=5, SSEPruningMultiplier_Captures=120
- **Non-captures**: `depth <= 9`, SEE threshold = `-49*depth - moveStatScore/134`
  - SSEPruningDepth_NonCaptures=9, SSEPruningMultiplier_NonCaptures=49, SSEPruningMoveStatDivNonCaptures=134
  - Note: integrates history into the SEE threshold for quiets

### Internal Iterative Reduction (IIR)
- **Depth guard**: `depth >= 3` (IIRStartDepth=3)
- **Conditions**: cut node OR PV node, AND (no TT move OR TT depth + 4 < current depth)
- **Effect**: reduces depth by 1

### Mate Distance Pruning
- Standard: clamps alpha/beta to `[-Checkmate+ply, Checkmate-ply-1]`
- Non-root only

### 50-Move Rule Eval Scaling
- Eval scaled by `(234 - halfMoveClock) / 234` (FiftyMoveRuleEvalScale=234)

---

## 2. Move Ordering

### Stages (MovePicker)
1. **TT Move** (score: INT32_MAX - 1)
2. **Generate & pick captures** (winning/good captures and queen promotions only in this pass)
3. **Killer move** (1 slot per ply, score: 1,000,000)
4. **Counter move** (score: 999,999)
5. **Generate & pick quiets** (sorted by combined score)
6. Bad captures fall through (scored < GoodCaptureValue threshold)

Note: only 1 killer move per ply (not 2). Counter move is separate stage.

### Capture Scoring
- `attacker < captured` piece value: WinningCaptureValue (20M)
- `attacker == captured`: GoodCaptureValue (10M)
- SEE >= 0: GoodCaptureValue (10M)
- SEE < 0: INT16_MIN (bad capture, ordered last)
- Plus `4096 * captured_piece_type` (MVV)
- Plus capture history score (shifted by -INT16_MIN to stay positive)

### Quiet Scoring
- **Main history**: `quietMoveHistory[stm][from_threatened][to_threatened][from*64+to]`
  - Dimensions: [2 colors][2 from-threat][2 to-threat][4096 from-to]
  - **Threat-aware**: separate counters for whether source/dest squares are attacked by opponent
- **Continuation history**: up to 6 plies back
  - Dimensions: `[2 prevIsCapture][2 prevColor][2 currentColor][6 piece][64 to]`
  - Scoring weights: conthist[0] * 1.0, conthist[1] * (1019/1024), conthist[3] * (555/1024), conthist[5] * (582/1024)
  - Note: conthist[2] and conthist[4] are NOT used in scoring (but conthist[2] IS used in updates)
- **Threat-based piece bonuses** (novel):
  - Knight/Bishop: +4000 if moving FROM pawn-attacked square, -4000 if moving TO pawn-attacked square
  - Rook: +8000/-8000 for minor-attacked squares
  - Queen: +12000/-12000 for rook-attacked squares
- **Node cache bonus** (for ply < 3): `4096 * moveNodesSearched / totalNodesSum`
  - Uses previous iteration's node counts to boost moves that were searched extensively

### Promotion Scoring
- Queen promotion: +5,000,000
- Knight/Bishop/Rook promotions: large negative (underpromotions ordered last)

### History Update Formula
- Gravity-style: `counter += delta - counter * |delta| / 16384`
- **Quiet bonus**: `min(-113 + 164*depth + 148*scoreDiff/64, 2178)`
- **Quiet malus**: `-min(-51 + 160*depth + 155*scoreDiff/64, 1844)`
- **Continuation bonus**: `min(-105 + 166*depth + 162*scoreDiff/64, 2065)`
- **Continuation malus**: `-min(-50 + 162*depth + 174*scoreDiff/64, 2065)`
- **scoreDiff** = `min(bestValue - beta, 256)` (how much score exceeded beta)
- **Capture bonus**: `min(27 + 72*depth, 2658)`, malus: `-min(28 + 44*depth, 1885)`
- **Continuation update weights**: ply-1=1.0, ply-2=1014/1024, ply-3=300/1024, ply-4=978/1024, ply-6=978/1024
  - Note: ply-5 (conthist[4]) is NOT updated
- **New search**: history divided by 2, killers cleared (history persists across searches with decay)
- **Clear values**: quiet=802, continuation=762, captures=346 (non-zero initial values)

### Prior Counter-Move History Update
- When `bestValue <= oldAlpha` and previous move was quiet: bonus of `min(1200, depth*120 - 100)` to the previous move's continuation history
- This rewards moves that precede fail-lows (the previous quiet move led to a good position for the opponent)

---

## 3. Time Management

### Moves-Left Estimation (from Lc0)
- `f(moves) = 35 * (1 + 1.5 * (moves/35)^2.19)^(1/2.19) - moves`
  - Midpoint=35, steepness=219/100=2.19

### Ideal Time
- `idealTime = 0.823 * (remaining / movesLeft + increment)` (TM_IdealTimeFactor=823/1000)
- **Max time**: `4.50 * (remaining - overhead) / movesLeft + increment)` (TM_MaxTimeFactor=450/100)
- Both clamped to `[0, 0.8 * remaining]`

### Predicted Move Adjustment
- If opponent played the predicted move (ponder hit): `idealTime *= 0.915` (save time)
- If opponent played a different move (ponder miss): `idealTime *= 1.132` (spend more time)

### Per-Iteration Updates (after depth >= 5)
- **PV Stability**: tracks how many consecutive iterations the best move stayed the same
  - Factor: `1.549 - 0.058 * min(10, stabilityCounter)` (offset=1549/1000, scale=58/1000)
  - Range: ~1.55 (unstable, move just changed) to ~0.97 (stable for 10+ iterations)
- **Node fraction**: `(1 - bestMoveNodeFraction) * 2.08 + 0.63`
  - TM_NodesCountScale=208/100, TM_NodesCountOffset=63/100
  - If best move used 90% of nodes: factor ~0.84 (save time)
  - If best move used 50% of nodes: factor ~1.67 (spend more time)

### Root Singularity
- Kicks in after 20% of ideal time has elapsed
- Depth guard: `depth >= 9` (SingularitySearchMinDepth=9), `|score| < 1000`
- Searches at `depth/2` with beta = `score - threshold`
  - Threshold: `max(204, 407 - 24*(depth-9))` (decreases with depth)
- If singular (score below beta): stop search immediately (the move is clearly best)

### Stop Conditions
- Hard time limit checked every 512 nodes (or at root)
- Soft time limit checked per ID iteration (also affected by TM adjustments)
- Mate found for 7 consecutive depths: stop (MateCountStopCondition=7)
- Single legal move with time control: return immediately without searching

---

## 4. LMR (Late Move Reductions)

### Base Table
- **Separate tables for quiets and captures**
- Quiets: `64 * (0.56 + 0.43 * ln(depth) * ln(moveIndex))` (bias=56/100, scale=43/100)
- Captures: `64 * (0.68 + 0.42 * ln(depth) * ln(moveIndex))` (bias=68/100, scale=42/100)
- Table is 64x64, values are in units of 1/64 (LmrScale=64)
- Uses natural log (`Log`)

### Activation
- `depth >= 1` (LateMoveReductionStartDepth=1), moveIndex > 1
- In PV nodes: only quiet moves get LMR (captures skip LMR in PV)

### Quiet Move Adjustments (all in units of 1/64)
| Adjustment | Value | Direction |
|---|---|---|
| Non-PV node | +15 | more reduction |
| TT move is capture | +73 | more reduction |
| Move is killer/counter | -168 | less reduction |
| Cut node | +183 | more reduction |
| Not improving | +38 | more reduction |
| Move gives check | -71 | less reduction |
| History-based | `-(moveStatScore + 6877) / 240` | variable |

### Capture Move Adjustments (all in units of 1/64)
| Adjustment | Value | Direction |
|---|---|---|
| Winning capture (score > WinningCapture) | -63 | less reduction |
| Bad capture (score < GoodCapture) | +12 | more reduction (note: -12, so actually LESS) |
| Cut node | +81 | more reduction |
| Not improving | +18 | more reduction (note: -18, so actually LESS) |
| Move gives check | +4 | more reduction (note: -4, so actually less) |

Wait -- re-reading the code more carefully:
- `LmrCaptureBad = -12` means `r += -12` = r decreases (LESS reduction for bad captures)
- `LmrCaptureImproving = -18` means `r -= -18` = r increases when NOT improving (MORE reduction)
- `LmrCaptureInCheck = -4` means `r -= -4` = r increases for checks (MORE reduction?!)

Correction for captures:
| Adjustment | Value | Effect |
|---|---|---|
| Winning capture | r -= 63 | less reduction |
| Bad capture | r += (-12) = r -= 12 | LESS reduction (surprising) |
| Cut node | r += 81 | more reduction |
| Not improving | r += (-(-18)) = r += 18 | more reduction when not improving |
| Move gives check | r -= (-4) = r += 4 | slightly MORE reduction (unusual) |

### PV-Specific Adjustments
- `r -= 64 * depth / (1 + ply + depth)` -- reduces less at low ply relative to depth
- TT entry with high depth: `r -= 13`

### Final Processing
- Result divided by 64 (rounded): `r = (r + 32) / 64`
- Clamped to `[0, newDepth]` -- never drops into qsearch

### LMR Deeper/Shallower
- If reduced search beats alpha: consider adjusting `newDepth`
  - `+1` if `score > bestValue + 85` AND `ply < 2*rootDepth` (LmrDeeperTreshold=85)
  - `-1` if `score < bestValue + newDepth`
  - Re-searches at full depth only if resulting newDepth > lmrDepth

### PV Depth Floor
- In PV search, if TT move at rootDepth > 8 and ttEntry.depth > 1: `newDepth = max(newDepth, 1)` (don't drop into QS for TT moves)

---

## 5. Singular Extensions

### Conditions
- Non-root, `depth >= 3` (SingularExtMinDepth=3)
- Move is TT move with `|ttScore| < KnownWinValue`
- TT entry has lower bound, `ttEntry.depth >= depth - 3`
- No filtered move already active

### Search Parameters
- Singular beta: `ttScore - depth` (subtracts current depth from TT score)
- Singular depth: `max(1, (59*depth - 215) / 128)` (SingularExtDepthRedMul=59, Sub=215)
  - At depth 8: `max(1, (472-215)/128)` = max(1, 2) = 2
  - At depth 16: `max(1, (944-215)/128)` = max(1, 5) = 5

### Extension Results
- **Singular** (score < singularBeta):
  - +1 extension (if ply < 2*rootDepth)
  - +2 if also `score < singularBeta - 14 - 256*isPvNode` (double extension)
  - +3 if also `score < singularBeta - 51 - 256*isPvNode` (triple extension)
- **Multi-cut** (singularScore >= beta): return `(singularScore * singularDepth + beta) / (singularDepth + 1)`
  - Blended return value, not just beta
- **Negative extensions** when not singular:
  - `ttScore >= beta`: -2 (non-PV: -3)
  - `isCutNode`: -2
  - `ttScore <= alpha`: -1
- **Recapture extension**: +1 in PV if TT move recaptures (same target square)

---

## 6. Quiescence Search

### Stand Pat
- Uses TT-adjusted eval as stand pat if TT bounds allow
- **Return value on stand-pat beta cutoff**: blended `(bestValue * (1024-519) + beta * 519) / 1024`
  - NOT just bestValue -- partial blend towards beta (QSearchStandPatBetaScale=519)
- Same blending applied at end of qsearch: `(bestValue * (1024-540) + beta * 540) / 1024`

### Futility in QSearch
- Base = `standPat + 77` (QSearchFutilityPruningOffset=77)
- Skips captures where `futilityBase <= alpha` and target is not recapture and SEE < 1

### Move Count Pruning in QSearch
- Depth < -4: only 1 move
- Depth < -2: only 2 moves
- Depth < 0: only 3 moves

### Other QSearch Details
- Underpromotions always skipped
- Bad captures (SEE < 0) cause immediate break (not just skip)
- In check: generates evasions, only tries 1 evasion move if it doesn't improve
- TT write at end of qsearch
- Captures history updated on beta cutoff

---

## 7. Eval Correction History

### Four Tables
1. **Pawn structure**: `[2 stm][16K entries]` keyed by pawn hash
2. **Non-pawn white**: `[2 stm][16K entries]` keyed by white non-pawn hash
3. **Non-pawn black**: `[2 stm][16K entries]` keyed by black non-pawn hash
4. **Continuation correction**: `[2 stm][384 piece-to][384 piece-to]`
   - Uses moves at ply-2 and ply-4 (two continuation terms)

### Application
- `corr = 53*pawn + 65*nonPawnW + 65*nonPawnB + 76*cont_ply2 + 76*cont_ply4`
- Divided by 512 (EvalCorrectionScale)
- Applied on top of NNUE eval, then 50-move scaling applied

### Update
- Bonus: `clamp((bestValue - unadjustedEval) * depth / 4, -249, 249)`
- Gravity: `h += value - h * |value| / 1024`
- Triggered when: not in check, best move is quiet or loses material (SEE < 0), and score diverged from eval

---

## 8. Aspiration Windows

- Initial window: `6 + |prevScore| / 17` (AspirationWindow=6, AspirationWindowScoreScale=17)
- On fail-low: `beta = (alpha+beta+1)/2`, `alpha -= window`, restore depth
- On fail-high: `beta += window`, reduce depth by 1 (if depth > 1 and depth+5 > iterationDepth)
- Window growth: `window += window / 3` each fail
- Fallback: full window when `window > 547` (AspirationWindowMaxSize=547)

---

## 9. Lazy SMP

- Shared: only TT (lockless)
- Per-thread: MoveOrderer, NodeCache, AccumulatorCache, CorrectionHistories, search stack
- **NUMA-aware**: threads pinned to NUMA nodes, correction histories allocated per-node
- Best thread selection after search: picks thread with highest depth+score (prefers deeper, prefers mate scores)

---

## 10. Novel / Unusual Features

### Threat-Aware Quiet History
- History table indexed by `[from_is_threatened][to_is_threatened]` in addition to color and from-to
- This gives 4x the granularity: a knight retreat from a pawn-attacked square is tracked separately from an unattacked square

### Node Cache (near-root move ordering)
- For ply < 3, tracks nodes spent on each move across iterations
- Adds a bonus proportional to `nodesSearched / totalNodes` when scoring quiet moves
- Acts like a learned "effort" heuristic -- moves that required more search get boosted

### Fail-High Score Adjustment
- On beta cutoffs: `bestValue = (bestValue * depth + beta) / (depth + 1)`
- Blends the fail-high score towards beta based on depth
- This prevents inflated scores from propagating up the tree

### Depth Reduction on Alpha Improvement
- When a new best move raises alpha (but doesn't cause cutoff), depth is reduced by 1
- `if (node->depth > 2) node->depth--`
- This means remaining moves are searched at reduced depth after finding a good move

### QSearch Stand-Pat Beta Blending
- Stand pat returns `(eval * ~0.49 + beta * ~0.51) / 1.0` instead of just eval
- End-of-qsearch also blends: `(bestValue * ~0.47 + beta * ~0.53) / 1.0`
- Prevents extreme QSearch values; damps oscillation

### Prior Counter-Move History Bonus
- When a node fails low and the previous move was quiet, the previous move's continuation history gets a bonus
- Rationale: if all our responses to the previous move are bad, the previous move was probably good

### CanReachGameCycle (Cuckoo Hashing)
- Before searching non-PV nodes with alpha < 0, checks if a drawing move exists
- Uses Stockfish-style cuckoo tables for reversible move detection
- If cycle is reachable, alpha is raised to 0 (can't do worse than draw by repetition)

### Singular Extension Multi-Cut Return Value
- Returns `(singularScore * singularDepth + beta) / (singularDepth + 1)` instead of just beta
- Weights the blended return by singular search depth

### History Bonus Includes Score Difference
- Both quiet and continuation bonuses scale with `scoreDiff/64` where scoreDiff = `min(bestValue - beta, 256)`
- Moves that caused a large beta cutoff get proportionally larger history updates

### Capture LMR (separate table)
- Most engines only do LMR for quiets; Caissa has a separate LMR table for captures
- Captures get LMR even in non-PV nodes (condition: `!isPvNode || move.IsQuiet()` means captures get LMR everywhere)

### Non-Zero History Initialization
- History tables are initialized to non-zero values (quiets=802, conthist=762, captures=346)
- This biases unexplored moves slightly positive rather than neutral

---

## 11. Things We (GoChess) Might Not Have

1. **Threat-aware history** (4x table indexed by from/to threat status)
2. **Node cache for near-root move ordering** (ply < 3, nodes-based scoring)
3. **6-ply continuation history** (we have 3; they use 6 with weighted scoring/updates)
4. **QSearch beta blending** (dampened stand-pat and fail-high returns)
5. **Depth reduction after alpha improvement** (depth-- when alpha raised mid-search)
6. **Fail-high score blending** (`(score*depth + beta) / (depth+1)`)
7. **Prior counter-move history update** (bonus for previous quiet move on fail-low)
8. **Non-pawn correction history** (separate tables for white/black non-pawn hash)
9. **Continuation correction history** (correction based on ply-2 and ply-4 piece-to)
10. **Predicted move time adjustment** (ponder hit saves time, miss spends more)
11. **SEE pruning with history in threshold** (for quiet moves: `-49*d - stat/134`)
12. **Non-zero history initialization** (biases unexplored moves positive)
13. **History bonus scaled by score difference** (larger bonuses for bigger beta cutoffs)
14. **In-check ProbCut** (TT-based, no search needed)
15. **Singular multi-cut blended return** (weighted by singular depth)
16. **Separate LMR table for captures** (different base formula than quiets)
17. **IIR on stale TT entries** (fires if TT depth + 4 < current depth, not just missing)
