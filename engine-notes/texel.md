# Texel 1.13a6 - Engine Crib Notes

Source: `~/chess/engines/texel/`
Author: Peter Osterlund

---

## 1. Search

### Iterative Deepening & Aspiration Windows

- **Aspiration window**: delta = 9 centipawns (`aspirationWindow = 9`)
- First iteration: full window (-MATE0, MATE0)
- On fail high/low: retry with window * 1.5 each time (`betaRetryDelta = betaRetryDelta * 3 / 2`)
- Win scores: aspiration delta jumps to 3000 (essentially full window for mate searches)
- Mate searches: if score is win/lose, set retry delta to MATE0 to avoid aspiration window

### Mate Distance Pruning

- `beta = min(beta, MATE0-ply-1); if alpha >= beta return alpha`
- Standard. Applied at top of search.

### Null-Move Pruning

- **Depth guard**: depth >= 3
- **Conditions**: not in check, NMP allowed at this ply, beta not a win score, non-PV (beta == alpha + 1), not singular search
- **Material requirement**: side to move must have non-pawn material AND at least one pawn
- **TT-based skip**: if TT entry is EXACT or UPPER (T_LE) with depth >= depth-R and score < beta, skip NMP
- **Static eval guard**: if evalScore < beta, skip NMP
- **Reduction R**: `min(depth, 4)` -- fixed R=4 (capped at depth)
- **Verification search**: if score >= beta AND depth >= 10, do a verification search at depth-R with null move disabled. Must still pass beta.
- **Score capping**: if score >= beta and isWinScore(score), clamp score to beta

### Razoring

- **Depth guard**: depth < 4 (depths 1-3)
- **Conditions**: not in check, non-PV (beta == alpha+1), not singular, normal bound (alpha not losing, beta not winning)
- **Margins**:
  - depth <= 1: razorMargin1 = **86**
  - depth 2-3: razorMargin2 = **353**
- **Method**: if eval < beta - margin, do QS at (alpha-margin, beta-margin). If QS score <= alpha-margin, return.

### Reverse Futility Pruning (Static Null Move Pruning)

- **Depth guard**: depth < 5 (depths 1-4)
- **Conditions**: not in check, normal bound, not singular
- **Material requirement**: same as NMP (non-pawn material + pawns)
- **Margins** (per depth, NOT scaled by improving):
  - depth 1: reverseFutilityMargin1 = **102**
  - depth 2: reverseFutilityMargin2 = **210**
  - depth 3: reverseFutilityMargin3 = **267**
  - depth 4: reverseFutilityMargin4 = **394**
- **Return value**: evalScore - margin (not just beta)

### Futility Pruning (Forward)

- **Depth guard**: depth < 5 (depths 1-4)
- **Conditions**: not in check, normal bound, not singular
- **Margins**:
  - depth 1: futilityMargin1 = **61**
  - depth 2: futilityMargin2 = **144**
  - depth 3: futilityMargin3 = **268**
  - depth 4: futilityMargin4 = **334**
- **Method**: set `futilityPrune = true` if eval + margin <= alpha. Then for each reducible quiet move that doesn't give check and isn't a passed pawn push, skip search and use futilityScore as the move's score.

### Late Move Pruning (LMP)

- **Conditions**: non-PV (beta == alpha+1), material OK (non-pawn + pawns)
- **Move count limits** by depth, with tighter limits when `badPrevMove` is true (eval dropped vs 2 plies ago):
  - depth 1: normal **3**, bad prev move **2**
  - depth 2: normal **6**, bad prev move **4**
  - depth 3: normal **12**, bad prev move **7**
  - depth 4: normal **24**, bad prev move **15**
- **Applied to**: reducible moves (score < 30, not captures with good SEE, not promotions) that don't give check and aren't passed pawn pushes
- Default lmpMoveCountLimit = 256 (no pruning) for depth >= 5

### Internal Iterative Deepening / Reduction (IID/IIR)

- **Conditions**: depth > 4, no hash move, not in check
- **PV nodes**: always do IID search at depth-2
- **Non-PV nodes**: do IID search at depth * 3/8 (only if depth > 8)
- **After IID**: probe TT to get hash move, then **always reduce depth by 1** (IIR)

### Singular Extensions

- **Depth guard**: depth > 5
- **Conditions**: hash move selected, not already in singular search, TT entry not upper bound (not T_LE), TT depth >= depth-3, not a win score (or not normal bound), ply+depth < MAX_SEARCH_DEPTH, hash move is legal
- **Search**: at depth/2, with beta = ttScore - depth
- **Result**: if singular search fails low, extend hash move by 1. If singular search score >= beta, return beta (multi-cut).

### Check Extensions

- **Extend by 1** if move gives check AND either: depth <= 1, OR SEE(m) >= 0

### LMR (Late Move Reductions)

- **Depth guard**: depth >= 2
- **Conditions**: move is reducible (score < 30, not a good capture or promotion), not a passed pawn push
- **Base reductions** (move-count based, NOT log-log formula):
  - `lmrCount > lmrMoveCountLimit2 (12)` AND depth >= 5 AND not capture: **lmr = 3**
  - `lmrCount > lmrMoveCountLimit1 (3)` AND depth >= 3 AND not capture: **lmr = 2**
  - mi >= 2: **lmr = 1**
- **Adjustment +1** (applied additively, if lmr > 0, lmr+3 <= depth, non-PV):
  - **Expected cut node**: +1 (see below for definition)
  - **Bad previous move** (eval dropped vs grandparent): +1
  - **Bad history score** (move.score < 20): +1
- **Defense move reduction cancel**: if move is a "defense move" (moving a piece away from threat by lesser piece), reset lmr to 0. Only for non-captures.
- **Re-search**: if LMR score > alpha, re-search at full depth.

### Root LMR

- At root, depth >= 3: reduce by 1 if move index >= rootLMRMoveCount (2) + maxPV, and move is not capture/promotion, doesn't give check, not passed pawn push. Re-search if score > alpha.

### Quiescence Search

- **Checks**: first QS level (depth > -1) generates captures AND checks. Deeper levels: captures only.
- **Check handling in deep QS**: below depth -2, check status is not tracked (saves givesCheck calls)
- **Hard move limit**: at depth < -6 and mi >= 2, stop searching (only 2 moves at very deep QS)
- **Delta pruning**: margin = **152**. If evalScore + captureValue + promoValue + 152 < alpha, skip (unless gives check, or insufficient material for safety). Tracks best optimistic score for the return value.
- **SEE filtering**: all non-evasion captures with negative SEE are skipped. Non-capture checks also skipped if negative SEE.
- **Move sorting**: MVV-LVA, only sort first `quiesceMaxSortMoves = 8` moves (optimization: if 8 moves fail to cut, likely ALL node, don't bother sorting rest)

### ABDADA Parallel Search

- Uses ABDADA (Alpha-Beta Distributed-ish with Aging) parallel search via `abdadaExclusive` flag
- At depth >= 7: uses TT "busy" bit to mark nodes being searched. If a sibling thread finds a node busy, returns BUSY score and defers it to a second pass.
- Two-pass move loop: pass 0 tries all moves but may get BUSY results; pass 1 retries BUSY moves.

---

## 2. Move Ordering

### Stages (not staged picker -- single score-and-sort)

1. **Hash move**: score = 10000, swapped to front
2. **Remaining moves scored in one pass** (if hash move found, scoring deferred until move 1):
   - **Good captures** (SEE > 0): 100 + MVV-LVA, scaled by 100 -> scores ~10000+
   - **Equal captures** (SEE == 0): 50 + MVV-LVA, scaled by 100 -> scores ~5000+
   - **Bad captures** (SEE < 0): -50 + MVV-LVA, scaled by 100 -> scores negative
   - **Killer move (primary)**: score = 64 (4 + 60)
   - **Killer move (secondary)**: score = 63 (3 + 60)
   - **Counter move**: score = 50
   - **Quiet moves**: history score (0-49)
3. **Selection sort**: selectBest() called incrementally for first N moves, then stops sorting when past LMP limit

### MVV-LVA Scoring Detail

- `score = pieceValueOrder[victim] * 8 - pieceValueOrder[attacker]`
- SEE sign test: +100 (good), +50 (equal), -50 (bad), then * 100

### History Table

- **Dimensions**: `ht[pieceType][toSquare]` -- 13 piece types x 64 squares
- **Representation**: each entry stores `nValues` (total weight, capped at 1000) and `scaledScore` (fixed-point score * 1024)
- **Success update**: weighted moving average toward maxVal (50), weight = depthWeight
- **Fail update**: weighted moving average toward 0, weight = depthWeight
- **Depth weights**: `depthTable[0..5] = {0, 1, 6, 19, 42, 56}` -- depth 0 does nothing, depth 5+ = 56
- **Score range**: 0 to 49 (returned by getHistScore, via scaledScore >> 10)
- **Rescale**: every iterative deepening iteration, nValues >>= 2 (halve influence of old data)
- **No continuation history** -- only piece-to-square history
- **No capture history** -- captures scored by SEE only

### Counter Move Heuristic

- **Dimensions**: `cm[pieceType][toSquare]` -- indexed by PREVIOUS move's piece-on-to-square
- Single counter move stored per slot (most recent), no second slot

### Killer Table

- **2 killer slots per ply** (LRU replacement)
- Primary hit = score 4, secondary = score 3 (then +60 added = 64/63)

---

## 3. Time Management

### Base Time Allocation

- `moves = min(movesToGo or 999, timeMaxRemainingMoves=35)`
- `margin = min(bufferTime=1000, time * 9/10)` -- reserve at least 1s or 10% of time
- `timeLimit = (time + inc * (moves-1) - margin) / moves`
- `minTimeLimit = timeLimit`

### Ponder Bonus

- If pondering enabled: estimate opponent's time per move, then add `min(oTimeLimit, timeLimit/(1-k)) * k` where k = timePonderHitRate/100 = 0.35

### Max Time

- `maxTimeLimit = minTimeLimit * clamp(moves * 0.5, 2.0, maxTimeUsage/100=4.0)`
- Both clamped to [1, time - margin]

### One Legal Move

- If only 1 legal move: minTime and maxTime clamped to max(1, original/100) or max depth 2

### Hard Factor (Instability/Difficulty)

- After each completed depth, compute `hardFactor` based on fraction of nodes spent on best move:
  - f < 0.20: hard = 3.5 (very hard -- best move uses few nodes)
  - 0.20 <= f < 0.40: linear interpolation 3.5 -> 1.0
  - 0.40 <= f < 0.60: hard = 1.0 (normal)
  - 0.60 <= f < 0.85: linear interpolation 1.0 -> 0.3
  - f >= 0.85: hard = 0.3 (easy -- best move dominates)
- Averaging: `hardFactor = (hardFactor + hard) / 2` (smoothed with previous)
- **Early stop**: can stop after minTimeMillis * (earlyStopPercentage/100) * hardFactor
  - `earlyStopPercentage` default = minTimeUsage = **85%**

### Fail-Low/High Handling

- On fail-high of non-first move: `needMoreTime = true`, `hardFactor = max(hardFactor, 1.0)`, uses maxTimeMillis
- On fail-low: `needMoreTime = true`, `hardFactor = max(hardFactor, 2.0)`, uses maxTimeMillis
- `searchNeedMoreTime` flag: when set, shouldStop uses maxTimeMillis instead of minTimeMillis
- Adjusted minTimeMillis: if earlyStopPercentage <= 100, use `min(minTime * hardFactor, maxTime)`

### Move-Time Mode

- `earlyStopPercentage = 10000` (effectively disables early stopping)

---

## 4. LMR Details

Texel does NOT use a log-log LMR table. Instead it uses discrete move-count thresholds:

```
lmrCount tracks how many "reducible" moves have been searched.

if lmrCount > 12 && depth >= 5 && !capture:  lmr = 3
elif lmrCount > 3 && depth >= 3 && !capture:  lmr = 2
elif moveIndex >= 2:                           lmr = 1
```

### Reduction Adjustments (additive +1, only when lmr > 0 and lmr+3 <= depth and non-PV)

- **Expected cut node**: +1. Determined by walking up the tree: count consecutive "first moves" from current ply. If the first non-first-move ancestor has even count of first-moves above it, it's an expected cut node.
- **Bad previous move**: +1 if current eval < eval 2 plies ago
- **Bad history**: +1 if move.score() < 20

### Reduction Cancellation

- **Defense move**: if a non-capture quiet move moves a piece away from attack by a lesser piece (e.g., rook away from pawn/knight/bishop attack to a safe square), lmr reset to 0

### "Reducible" Definition

- `move.score() < 30` AND (not a capture, OR move.score() < 0) AND not a promotion
- This means good/equal captures are never reduced, bad captures can be

---

## 5. Notable/Novel Features

### Expected Cut Node Heuristic

Texel computes whether a node is an "expected cut node" by examining the move order in ancestor nodes. It walks up from the current ply counting consecutive first-move nodes. If the count of first-moves is even when a non-first-move is found, the node is expected to cut. This is used to increase LMR by +1 at expected cut nodes. This is a form of node-type prediction that goes beyond simply tracking PV vs non-PV.

### Defense Move Heuristic

The `defenseMove()` function detects when a piece is moving away from a threatening lesser piece. For rooks/queens: checks if the from-square is attacked by enemy pawns/knights/bishops but the to-square is not. For bishops/knights: checks if from-square is attacked by enemy pawns but to-square is not. Defense moves get their LMR cancelled (reduced less). This is specific to Texel and not commonly seen.

### Bad Previous Move (Eval Drop)

`badPrevMove = (evalScore != UNKNOWN && ply >= 2 && evalScore < eval_2_plies_ago)`. Used in two places:
1. LMP: tighter move count limits when badPrevMove (e.g., 2 instead of 3 at depth 1)
2. LMR: +1 reduction adjustment

### History Table Design

Unlike most modern engines that use additive bonus/malus history with gravity, Texel uses a **weighted moving average** with fixed-point arithmetic. Success pushes the score toward 50, failure toward 0, with exponential depth weighting. The depth weight table grows roughly quadratically: {0, 1, 6, 19, 42, 56}. Scores rescale every iteration (nValues >>= 2).

### No Continuation History

Texel has no continuation history (no 2-ply or 4-ply context). Only piece-to-square history. Modern engines typically gain significant Elo from conthist.

### No Capture History

Captures are ordered purely by SEE sign + MVV-LVA. No capture history table.

### No ProbCut

Texel does not implement ProbCut.

### No History Pruning

No explicit history-based pruning threshold for quiet moves. Only LMP and futility.

### Null Move Verification at Depth >= 10

Instead of just using the null move score, Texel does a reduced-depth verification search with null move disabled when depth >= 10 and initial null move search failed high.

### QS Move Sort Cutoff

Only sorts the first 8 moves in QS (quiesceMaxSortMoves = 8). If the first 8 moves didn't cut, assumes it's an ALL node and stops sorting. Saves time on wide QS nodes.

### ABDADA Parallel Search

Uses TT busy-bit marking at depth >= 7. Moves that find a BUSY node are deferred to a second pass. This avoids redundant work when multiple threads explore the same subtree.

### TT Cutoff Restrictions

TT cutoffs only allowed in non-PV or when `depth*2 <= ply` (i.e., deep enough that PV integrity doesn't matter much). This is an unusual condition -- most engines use a simple PV check.

### Swindle Scores

For endgame tablebase draws, Texel computes "swindle scores" (small positive/negative scores) to prefer positions where the opponent is more likely to blunder in practice. `maxFrustrated = 70` centipawns caps the swindle value.

### Half-Move Factor

A table of 10 entries that apparently scales evaluation based on half-move clock proximity to the 50-move rule:
```
{128, 128, 128, 128, 44, 35, 29, 25, 20, 17}
```
Values are out of 128 (full scale). Late in the 50-move clock, evaluation is discounted significantly.

---

## 6. Comparison to GoChess (things we have that Texel lacks, and vice versa)

### Texel has, we lack:
- **Expected cut node** heuristic for LMR adjustment
- **Defense move** heuristic (LMR cancellation for tactical retreats)
- **Bad previous move** dual use (LMP tightening + LMR increase)
- **ABDADA** parallel search (we use Lazy SMP)
- **Null move verification** search at depth >= 10
- **QS sort cutoff** after N moves (quiesceMaxSortMoves = 8)
- **TT cutoff depth*2 <= ply** condition
- **Half-move factor** for 50-move rule eval scaling
- **Swindle scores** for TB draws

### We have, Texel lacks:
- **Continuation history** (we have it, 3x weighted)
- **Capture history**
- **ProbCut**
- **History pruning** (we prune moves with bad history)
- **NNUE evaluation**
- **Log-log LMR table** (ours is more granular)
- **Correction history**
- **Improving flag** for RFP/LMP/futility

### Different approaches:
- **History**: we use additive bonus/malus with divisor; Texel uses weighted moving average
- **RFP**: we use depth-scaled margin with improving flag; Texel uses fixed per-depth margins
- **LMR**: we use log(depth)*log(moveCount) formula with many adjustments; Texel uses discrete thresholds
- **NMP**: we use depth/3 + 3 reduction with eval/200 bonus; Texel uses fixed R=min(depth,4) with verification
- **Futility**: we use lmrDepth*100+100; Texel uses fixed per-depth margins
- **LMP**: we use d^2+3 formula with improving; Texel uses fixed per-depth limits with badPrevMove
