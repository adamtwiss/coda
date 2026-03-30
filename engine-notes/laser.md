# Laser 1.8 Beta - Engine Crib Notes

Source: ~/chess/engines/laser-chess-engine/src/
Authors: Jeffrey An and Michael An (2015-2018)
C++11, classical eval (no NNUE), Syzygy support, Lazy SMP.

---

## 1. Search

### Overview
- Fail-soft PVS with iterative deepening and aspiration windows
- Separate `checkQuiescence()` function for when side-to-move is in check in QS
- Two-fold repetition detection (not three-fold) with special handling: two-fold within search tree terminates immediately, but two-fold involving game history requires a third repetition
- Mate distance pruning present (standard implementation)

### Reverse Futility Pruning (Static Null Move Pruning)
- **Depth guard**: `depth <= 6`
- **Condition**: not PV, not in check, has non-pawn material
- **Margin**: `70 * depth`
- **Returns**: staticEval (not beta)
- Note: no improving distinction -- single margin formula

### Razoring
- **Depth guard**: `depth <= 2`
- **Condition**: not PV, not in check
- **Margin**: 300 (flat, not depth-dependent)
- At depth 1: drops straight into qsearch (full window: alpha, beta)
- At depth 2: does a null-window qsearch at `alpha - 300`; if it fails low, returns the qsearch value

### Null Move Pruning
- **Depth guard**: `depth >= 2`
- **Condition**: not PV, not in check, staticEval >= beta, has non-pawn material
- **Reduction formula**: `2 + (32 * depth + min(staticEval - beta, 384)) / 128`
  - At depth 2, eval==beta: R = 2 + (64+0)/128 = 2
  - At depth 10, eval==beta+200: R = 2 + (320+200)/128 = 2 + 4 = 6
  - At depth 16, eval==beta+384: R = 2 + (512+384)/128 = 2 + 7 = 9
  - The eval component is capped at 384 (so max contribution from eval is 3 at depth 0)
- **Verification search**: at `depth >= 10`, if null move fails high, does a verification search at `depth - 1 - reduction` with full (alpha, beta) window. Returns only if verification also fails high. At depth < 10, returns immediately.
- Killers of child node are cleared after null move

### ProbCut
- **Depth guard**: `depth >= 6`
- **Condition**: not PV, not in check, `staticEval >= beta - 100 - 20 * depth`, `abs(beta) < MAX_PLY_MATE_SCORE`
- **Margin**: `beta + 90`
- **Search depth**: `depth - depth/4 - 4` (e.g. depth 8 -> 2, depth 12 -> 5, depth 16 -> 8)
- Only searches captures (up to 3), excluding hash move
- Captures are SEE-filtered with margin `probCutMargin - staticEval`
- Returns score if any capture scores >= probCutMargin

### Internal Iterative Deepening (IID)
- **Condition**: no hash move, not in check
- PV nodes: `depth >= 6`, IID depth = `depth - depth/4 - 1`
- Non-PV nodes: `depth >= 8`, IID depth = `(depth - 5) / 2`
- This is a true IID (not IIR) -- it does an actual reduced-depth search

### Futility Pruning
- Uses a combined `lmrDepth`-based approach (called `pruneDepth`)
- `pruneDepth` = lmrDepth + 1 (PV), lmrDepth (improving), lmrDepth - 1 (not improving)
- **Depth guard**: `pruneDepth <= 6`
- **Margin**: `alpha - 115 - 90 * pruneDepth`
  - At pruneDepth=1: margin = alpha - 205
  - At pruneDepth=6: margin = alpha - 655
- Only applies to quiet, non-promotion, non-check moves that are not the hash move and where bestScore > -MATE

### Late Move Pruning (Move Count Pruning)
- **Depth guard**: `depth <= 12`
- Two tables, indexed by improving:
  - Not improving: {0, 2, 4, 7, 11, 16, 22, 29, 37, 46, 56, 67, 79}
  - Improving:     {0, 4, 7, 12, 20, 30, 42, 56, 73, 92, 113, 136, 161}
- At PV nodes, the threshold is increased by `depth` (adds depth to move count)
- Same prunable-move conditions as futility

### History Pruning
- **Depth guard**: `pruneDepth <= 2`
- Prunes quiet moves where BOTH counterMoveHistory AND followupMoveHistory are negative
- (Main history is NOT checked -- only continuation histories)

### SEE Pruning (quiet moves)
- **Depth guard**: `pruneDepth <= 6`
- **Threshold**: `-24 * pruneDepth^2`
  - At pruneDepth=1: -24
  - At pruneDepth=3: -216
  - At pruneDepth=6: -864

### SEE Pruning (all non-hash moves including captures)
- **Depth guard**: `depth <= 5` (non-PV only)
- **Threshold**: `-100 * depth`
  - At depth 1: -100
  - At depth 5: -500
- This is separate from the quiet SEE pruning above

### Singular Extensions
- **Depth guard**: `depth >= 8`
- **Condition**: move == hash move, no existing reduction or extension, `abs(hashScore) < NEAR_MATE_SCORE`, hash entry is CUT or PV node, `hashDepth >= depth - 3`
- **SE window**: `hashScore - depth`
- **SE search depth**: `depth / 2 - 1`
- Implementation note: iterates through ALL legal moves (not staged), which is unusually expensive. Most engines use the move picker for SE.
- If all non-hash moves fail low against the SE window, extends by 1

### Check Extensions
- Extends by 1 if the move gives check AND `SEE >= 0`
- NOT applied if move-count pruning is active (so deep in the move list at low depth, checks are not extended)
- Applied both at root and in PVS

### No Other Extensions
- No recapture extensions, no passed pawn extensions, no castling extensions

---

## 2. Move Ordering

### Stages (main search)
1. **Hash move** (tried first, removed from remaining list)
2. **Captures** (scored, with SEE filtering during iteration)
3. **Quiets** (scored)

### Capture Scoring
- Formula: `SCORE_LOSING_CAPTURE + adjustedMVVLVA + captureHistory`
- `adjustedMVVLVA = 8 * MVVLVAScore / (4 + depth)` -- scales MVV/LVA DOWN at higher depths (more reliance on capture history at depth)
- `SCORE_LOSING_CAPTURE = -(1 << 14) = -16384`
- During STAGE_CAPTURES, each capture is SEE-checked before being returned. If SEE < captureMargin (default 0), it is deferred to end of list (searched after quiets). This is done lazily during `nextMove()`.

### Quiet Scoring
- Killers: scored at `SCORE_QUEEN_PROMO - 1` and `SCORE_QUEEN_PROMO - 2` (= 2047, 2046)
- Queen promotions: `SCORE_QUEEN_PROMO = 2048`
- Other quiets: `SCORE_QUIET_MOVE + mainHistory + counterMoveHistory + followupMoveHistory`
  - `SCORE_QUIET_MOVE = -(1 << 12) = -4096` (base offset)

### Quiescence Move Ordering
Stages:
1. Captures (scored by MVV/LVA only, no capture history)
2. Promotions (unsorted)
3. Checks (unsorted, ONLY at QS depth 0 / first ply of qsearch)

### History Tables

**Main history**: `historyTable[2][6][64]` -- [color][pieceType][toSquare]
- Dimensions: color (2) x piece (6, 0-indexed) x square (64)
- No from-square -- this is a "piece-to" table

**Capture history**: `captureHistory[2][6][6][64]` -- [color][movingPiece][capturedPiece][toSquare]

**Counter-move history**: `counterMoveHistory[6][64][6][64]` -- [prevPiece][prevTo][currentPiece][currentTo]
- Indexed by the PREVIOUS move's piece and destination
- NOT color-indexed (shared between colors for the current move dimension)

**Follow-up move history**: `followupMoveHistory[6][64][6][64]` -- [prevPrevPiece][prevPrevTo][currentPiece][currentTo]
- Indexed by the move TWO plies back (the grandparent move)
- Same structure as counter-move history

**History update formula**:
- `historyChange = depth * depth + 5 * depth - 2`
  - depth 1: 4, depth 5: 48, depth 10: 148, depth 15: 298, depth 18: 412
- Gravity: `history -= historyChange * history / 448` (then add/subtract historyChange)
- Cap: updates skipped if `depth > 18`
- On beta cutoff (quiet best move): increase best move history, decrease all previously-searched quiet moves
- On beta cutoff (capture best move): increase best move capture history, decrease all previously-searched captures
- On PV node: same update as cutoff for the best move

### No Counter-Move Table
Laser does NOT have a counter-move table (the direct move-to-move mapping). It only has counter-move HISTORY (the statistical table). This is unusual -- most engines have both.

---

## 3. Time Management

### Base Allocation
- `MOVE_HORIZON = 40` -- assume 40 moves left
- Horizon decreases as game progresses: `movesToGo = 40 - 8 * moveNumber / 80`
  - At move 80+: movesToGo = 32
- `value = timeRemaining / movesToGo + increment`
- Buffer time subtracted from remaining time (default configurable, subtracted before division)

### Special movestogo Handling
When movestogo < 10 and no increment:
- `allotment = max(value, timeRemaining * ALLOTMENT_FACTORS[movestogo])`
- `maxAllotment = min(value * 5.0, timeRemaining * MAX_USAGE_FACTORS[movestogo])`
- ALLOTMENT_FACTORS: {1.0, 0.99, 0.38, 0.28, 0.23, 0.20, 0.18, 0.16, 0.14, 0.12}
- MAX_USAGE_FACTORS: {1.0, 0.99, 0.74, 0.66, 0.62, 0.59, 0.56, 0.54, 0.52, 0.51}

### Normal Time Control
- `allotment = min(value, maxAllotment / 3)`
- `maxAllotment = min(value * 5.0, timeRemaining * 0.95)`
- `MAX_TIME_FACTOR = 5.0` -- hard cap at 5x the base allotment

### TIME_FACTOR
- `TIME_FACTOR = 0.85` -- the soft time limit is `allotment * 0.85 * timeChangeFactor`
- Time check uses `timeLimit = maxAllotment` as the hard limit

### Dynamic Time Adjustment (timeChangeFactor)
- Initialized to 1.0, decays each iteration: `timeChangeFactor = (2 * timeChangeFactor + 1) / 3`
- **PV stability**: if best move unchanged, `pvStreak++` and `timeChangeFactor *= 0.92`
- **PV instability**: if best move changed, `pvStreak = 1`, floor at 1.0, then `*= 1.3`
- **Score change**: `timeChangeFactor *= 0.92 + min(7.0, sqrt(abs(prevScore - bestScore))) / 25.0`
  - Score change 0: *= 0.92 (reduce time)
  - Score change 25: *= 1.12
  - Score change 49+: *= 1.20 (cap from sqrt(49)/25 = 0.28)
- **Fail-low**: `timeChangeFactor *= 1.15`

### Easy Move Detection
- Triggered when: `pvStreak >= 8 + rootDepth/5`, time between 1/16 and 1/2 of allotment, not near-mate, not pondering
- Does a reduced search at `rootDepth - 4 - rootDepth/8` with window `bestScore - 150 - rootDepth - abs(bestScore)/3`
- If second-best move fails low below the window, stops search immediately
- On failure, sets `pvStreak = -128` to prevent re-triggering

### Fail-High Early Exit
- During aspiration loop: if a fail-high occurs but the best move is unchanged, and `timeSoFar >= allotment * 0.85`, breaks out of the aspiration loop without resolving

### One Legal Move
- If only one legal move: `timeLimit = min(timeLimit / 32, 1000ms)`

---

## 4. LMR

### Base Table
```
lmrReductions[depth][movesSearched] = round(0.5 + ln(depth) * ln(movesSearched) / 2.1)
```
- 64x64 table, initialized at startup
- Special override: for depth=1, movesSearched >= 7: reduction forced to 1

### LMR Conditions
- `depth >= 3` and `movesSearched > 1`
- Not a capture, not a promotion
- Applied at both root and interior nodes

### LMR Adjustments (interior nodes only)
| Condition | Adjustment |
|-----------|-----------|
| Killer move | -1 |
| History value (sum of main + counter + followup) | `-historyValue / 512` |
| Expected cut node | +1 |
| PV node | -1 |
| Not improving (non-PV only) | +1 |

### Clamping
- `reduction = max(0, min(reduction, depth - 2))`
- Never reduces directly into qsearch (always leaves at least 1 ply)

### Root LMR
- Same base table but with an additional `-1` adjustment: `reduction = lmrReductions[depth][moves] - 1`
- Only for quiet non-promotions
- Clamped to `max(0, reduction)`

---

## 5. Quiescence Search

### Structure
- Separate `quiescence()` and `checkQuiescence()` functions
- If in check at QS entry, delegates to `checkQuiescence()` which generates ALL evasion moves
- Stand pat used only when not in check

### QS Checks
- Non-capture check moves are generated ONLY at QS depth 0 (first ply)
- After first ply, only captures and promotions

### QS Futility (Delta Pruning)
- `staticEval < alpha - 80` AND `SEE(move) < 1`
- If pruned, `bestScore = max(bestScore, staticEval + 80)` (adjusts bestScore upward)

### QS SEE Pruning
- All moves with `SEE < 0` are pruned

### QS Check Evasions
- In `checkQuiescence()`: quiet evasion moves with `SEE < 0` are pruned UNLESS no other legal move has been found yet (bestScore still -INFTY, to avoid pruning into false checkmate)

### QS Hash Table
- Full TT probes in qsearch, with depth-aware cutoff (`hashEntry->depth >= -plies`)

---

## 6. Lazy SMP

- Standard shared-TT approach
- Per-thread: SearchParameters (killers, history tables), SearchStatistics, SearchStackInfo, TwoFoldStack
- Skip depths scheme (16-entry cycle):
  - `SMP_SKIP_DEPTHS = {1, 2, 2, 4, 4, 3, 2, 5, 4, 3, 2, 6, 5, 4, 3, 2}`
  - `SMP_SKIP_AMOUNT = {1, 1, 1, 2, 2, 2, 1, 3, 2, 2, 1, 3, 3, 2, 2, 1}`
  - Helper threads skip certain depths to diversify search
- `stopSignal` is seq_cst for initial check, relaxed for inner-loop checks

---

## 7. Transposition Table

- Two-bucket system (HashNode = two HashEntry slots, each 16 bytes)
- No lockless/atomic -- plain struct writes (potential for data races in SMP)
- Stores: zobrist key, score (int16), move, eval (int16), depth (int8), age+nodeType (uint8)
- Age-based replacement with 3-state node type (PV=0, CUT=1, ALL=2, NO_INFO=3)
- Static eval is cached in the TT and reused
- TT score used as refined static eval when node type matches direction

---

## 8. Eval Features (Classical)

- Tapered eval using SWAR (mg/eg packed into uint32_t)
- Endgame factor based on piece values: alpha=2010, beta=6410
- Material: P=100/138, N=405/410, B=452/461, R=720/766, Q=1379/1497 (mg/eg)
- Bishop pair: 55
- Tempo: 25
- Material imbalance table
- Mobility tables for N/B/R/Q/K
- King safety: attack units system with safe check bonuses, pawn shield, pawn storm
- Passed pawns with file bonus, free promotion/stop bonus, king proximity
- Pawn structure: doubled, isolated, backward, phalanx, connected
- Threats: undefended pieces, piece-on-piece threats
- Space bonus
- Endgame scaling: opposite bishops, pawnless
- Endgame case detection (known wins, corner distance)

---

## 9. Notable / Unusual Features

### Things Laser Has That GoChess Might Not
1. **Follow-up move history** (grandparent-indexed continuation history) -- this is the move 2 plies back, separate from counter-move history. GoChess has continuation history but should verify it includes both ply-1 and ply-2.

2. **Depth-scaled MVV/LVA**: `8 * MVVLVA / (4 + depth)` -- capture ordering relies more on capture history at higher depths. Novel approach.

3. **ProbCut with SEE margin**: the capture SEE threshold for ProbCut is `probCutMargin - staticEval`, which means only captures likely to reach the ProbCut margin are tried.

4. **ProbCut eligibility guard**: `staticEval >= beta - 100 - 20 * depth` prevents ProbCut at positions already far below beta.

5. **True IID (not IIR)**: does an actual reduced-depth search when no hash move is available, rather than just reducing depth by 1.

6. **Easy move detection**: verifies with a reduced-depth, narrow-window search that the second-best move is substantially worse. Saves time on obvious moves.

7. **Fail-high early termination**: during aspiration loop, if the same best move fails high and we've used enough time, stops without resolving.

8. **Null move verification at depth >= 10**: does a re-search to verify the null-move cutoff is real.

9. **pruneDepth concept**: LMR-adjusted depth is used for all pruning decisions (futility, LMP, SEE, history pruning), with +1 for PV and -1 for not-improving. This creates a unified pruning framework.

### Things GoChess Has That Laser Does Not
1. No counter-move table (only counter-move history)
2. No correction history
3. No NNUE
4. No fractional extensions
5. No passed pawn extensions or recapture extensions
6. No aspiration window delta adapting to score magnitude... wait, it does: `deltaAlpha = 14 - min(rootDepth/4, 6) + abs(bestScore)/25`
7. No history-based LMP threshold adjustment
8. No per-thread pawn hash table (no pawn hash at all visible in search)

### Aspiration Windows
- Initial delta: `14 - min(rootDepth/4, 6) + abs(bestScore)/25`
  - At depth 4, score 0: delta = 13
  - At depth 24, score 0: delta = 8
  - At depth 12, score 500: delta = 11 + 20 = 31
- Enabled at `rootDepth >= 6`
- On fail: window widens by 3/2x, alpha/beta re-centered on the result
- Separate deltaAlpha and deltaBeta (can grow independently)
