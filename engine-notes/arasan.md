# Arasan Chess Engine - Crib Notes

Source: `~/chess/engines/arasan-chess/src/`
Version: 25.x (2026), by Jon Dart

Key constants: `PAWN_VALUE = 128`, `DEPTH_INCREMENT = 2` (fractional plies, so depth 6 = 12 internal units).

---

## 1. Search Architecture

- Negamax alpha-beta with PVS (NegaScout), iterative deepening
- Explicit node types: PvNode=0, CutNode=-1, AllNode=1
- Node type propagation: 1st child of PV = PV, further = Cut; 1st child of Cut = All, further = Cut; All children = Cut
- Mate distance pruning: standard `alpha = max(alpha, -MATE+ply)`, `beta = min(beta, MATE-ply-1)`
- NNUE evaluation (HalfKA with lazy/incremental accumulators stored per-node on the search stack)
- Lazy SMP: threads share only the hash table and correction history. Per-thread: board, SearchContext (killers/history/counter), NNUE accumulators, stats

### Aspiration Windows

Steps: `[0.375P, 0.75P, 1.5P, 3.0P, 6.0P, MATE]` (6 steps). On fail high/low, widen to the next step. After exhausting all steps, open to full window.

At low iterations (`<= widePlies`), the window is wider: `value +/- (wideWindow + aspiration/2)`.

### Internal Iterative Deepening (IID)

When no hash move exists:
- PV nodes: depth >= 8 plies (16 internal)
- Non-PV nodes: depth >= 6 plies (12 internal)
- Reduced search depth: `d = depth - IID_DEPTH + DEPTH_INCREMENT`
- Full IID search (not just IIR): calls `search()` recursively with IID flag set at the same ply, reduced depth

**Difference from GoChess**: Arasan does a full IID search, not just an IIR depth reduction. This is more expensive but may find better hash moves.

---

## 2. Pruning Techniques

### 2a. Reverse Futility Pruning (Static Null Pruning)

- Depth guard: depth <= 6 plies (12 internal)
- Not in check, not PV, not verify/IID
- Condition: `eval >= beta + margin`
- Margin: `max(0.25P, 0.75P * (depth_plies - improving))` (formula from Obsidian)
- Return value: `(eval + beta) / 2` (average, not just eval or beta)
- Requires `eval < TABLEBASE_WIN`

With P=128: margins at depth 1 = 32 (min), depth 6 non-improving = 576 (4.5P), depth 6 improving = 480 (3.75P).

**Difference from GoChess**: Arasan returns `(eval+beta)/2` instead of just `eval`. This is a soft-bounded return that's more conservative.

### 2b. Null Move Pruning

- Depth guard: depth >= 2 plies (4 internal)
- Conditions: not PV, not in check, not IID/verify, side has pieces, eval >= beta, eval >= staticEval, excluded move is null, not mate score alpha, 50-move counter <= 98
- Additional guard: at depth < 4 plies, skip NMP if previous move was a capture/promotion
- Verification: `staticEval >= beta - 0.25P * (depth_plies - 6)` OR `depth >= 12 plies`
- Reduction formula: `R = 3 + depth/3 + min(3, (eval-beta)/(175*P/128))`
  - Low material adjustment: when material level <= 3, additional depth divisor changes: `depth/(3 + 3)` instead of `depth/3`, plus half-ply extension
- Hash-based skip: avoids null move if hash entry is upper bound with depth >= null_depth and value < beta
- Null move verification: at depth >= 6 plies, after null move cutoff, re-search at null depth to verify
- Mate score capping: if null move search returns >= MATE_RANGE, return beta instead

### 2c. Razoring

**DISABLED** by default (commented out with `#ifdef RAZORING`). When enabled:
- Depth guard: depth <= 1 ply (2 internal)
- Margin: `2.75P + depth_plies * 1.25P`
- Drops into quiescence search

### 2d. Futility Pruning (Quiet Moves)

- Depth guard: `pruneDepth <= 8 plies` (16 internal), where pruneDepth = depth - LMR reduction
- History threshold: only prune if `hist < 1000` (improving) or `hist < 2000` (non-improving)
- Margin: `0.25P + max(1, depth_plies) * 0.95P`
- Condition: `eval < alpha - margin`
- Lazy eval: computes eval on demand if not yet computed

With P=128: margins are 153 (d1), 275 (d2), 397 (d3), ... 1000 (d8).

### 2e. Futility Pruning (Captures)

- Depth guard: `pruneDepth <= 5 plies` (10 internal)
- Margin: `2.0P + depth_plies * 2.5P + maxValue(move) + captureHistory/65`
- Condition: `eval + margin < alpha`

### 2f. Late Move Pruning (LMP)

- Depth guard: depth_plies <= 9
- Move count formula: `LMP_BASE + LMP_SLOPE * d^(LMP_EXP/100) / 100`
  - Improving: base=4, slope=75, exp=2.04 => e.g. d1=5, d3=10, d5=20, d9=61
  - Non-improving: base=2, slope=70, exp=1.95 => e.g. d1=3, d3=8, d5=15, d9=45
- Only prunes moves in HISTORY_PHASE or later (not hash, captures, killers, counter)
- Uses raw depth, NOT pruneDepth

### 2g. History Pruning

- Depth guard: `pruneDepth <= (3 - improving) * DEPTH_INCREMENT` = 3 plies (non-improving) or 2 plies (improving)
- Condition: counter-move history < -250 AND follow-up history < -250
- Note: uses individual continuation histories, NOT the combined history score

### 2h. SEE Pruning

- Depth guard: `pruneDepth <= 7 plies` (14 internal)
- Only for moves past WINNING_CAPTURE_PHASE
- Quiet threshold: `-depth_plies * 0.75P` (= -96 per ply with P=128)
- Capture threshold: `-depth_plies^2 * 0.2P` (= -25.6*d^2 with P=128)

### 2i. ProbCut

- Depth guard: depth >= 5 plies (10 internal). Re-allows at depth >= 9 plies if already in ProbCut
- Not PV node
- Beta not near mate
- ProbCut beta: `beta + 1.25P` (= beta + 160)
- Search depth: `depth - 4 plies - depth/8`
- Move generation: captures only, skipping pawn captures (too low value)
- SEE filter: `seeSign(move, probcut_beta - staticEval)`
- Hash skip: if hash has upper bound entry with depth >= nu_depth+1 and value < probcut_beta, skip ProbCut
- Stores result in hash if successful

### 2j. Multi-Cut (from Singular Extension)

- When singular search fails (hash move not singular) and singular_beta >= beta: return singular_beta
- This is a cutoff based on the fact that many moves beat the singular threshold

---

## 3. Extensions

- **Check extension**: 1 ply (DEPTH_INCREMENT) when move gives check
- **Passed pawn push extension**: 1 ply when pawn moves to 7th rank
- **Capture of last piece**: 0.5 ply when capturing the opponent's last non-pawn piece in the endgame
- **Singular extension**: 1-3 plies depending on how singular the move is
  - Depth guard: depth >= 8 plies, hash entry is lower bound, hash depth >= depth-3
  - Singular margin: `P * depth / (64 * DEPTH_INCREMENT)` = roughly `depth_plies/64` pawns
  - Search depth: `depth/2 - 1 ply`
  - Triple extension (3 plies): non-PV and result < `nu_beta - SINGULAR_EXTENSION_TRIPLE`
  - Double extension (2 plies): non-PV and result < `nu_beta - SINGULAR_EXTENSION_DOUBLE`
  - Single extension (1 ply): default case
  - Negative extension (-2 plies): when hash value >= beta but not singular, or at cut nodes
- Extensions are capped at 1 ply total (for regular extensions; singular can go higher)

**Notable**: Arasan allows triple singular extensions (3 plies). SINGULAR_EXTENSION_DOUBLE defaults to 0, SINGULAR_EXTENSION_TRIPLE to 0.33P. So any singularity below nu_beta gives double extension, and below nu_beta-42 gives triple.

---

## 4. LMR (Late Move Reductions)

### Formula

```
LMR_REDUCTION[pv][depth][moves] = floor(2 * (base + log(d) * log(m) / div) + 0.5) / 2
```

Stored as integer * DEPTH_INCREMENT, effectively half-ply granularity.

Parameters:
- PV: base = 0.30, div = 2.25
- Non-PV: base = 0.50, div = 1.80

### Conditions

- Depth >= 3 plies (6 internal = LMR_DEPTH)
- Move index >= 1 + 2*isPV (so >= 3 for PV, >= 2 for non-PV)
- Either quiet move, OR capture with move_index > LMP count

### Adjustments (applied to the table value)

1. **Captures**: reduction -= 1 ply
2. **Non-PV + non-improving**: reduction += 1 ply
3. **Killer/counter move** (phase < HISTORY_PHASE, not in check): reduction -= 1 ply
4. **History-based**: reduction -= `clamp(historyScore / 2200, -2, +2)` plies
   - historyScore = main history + counter-move history + follow-up history

### Final reduction

`r = min(newDepth - 1 ply, reduction)`. If r < 1 ply, no reduction.

**Difference from GoChess**: Arasan's LMR base/div differ: Non-PV base=0.50 (GoChess C=1.5 is different parametrization). History divisor is 2200 (GoChess uses 5000). Arasan adds a ply for non-PV non-improving (GoChess doesn't). Arasan reduces killers/counter moves less.

---

## 5. Move Ordering

### Stages (MoveGenerator::Phase)

1. **HASH_MOVE_PHASE**: TT move
2. **WINNING_CAPTURE_PHASE**: Captures sorted by MVV-LVA (`8*Gain - pieceValue[moved]`), with `initialSortCaptures`. Losing captures deferred to LOSERS_PHASE
3. **KILLER1_PHASE**: Primary killer
4. **KILLER2_PHASE**: Secondary killer
5. **COUNTER_MOVE_PHASE**: Counter move (indexed by [piece][dest] of previous move)
6. **HISTORY_PHASE**: Quiet moves sorted by history score
7. **LOSERS_PHASE**: Losing captures (SEE < 0)

### QSearch Move Ordering

1. Hash move (if capture/promotion)
2. All captures sorted by MVV-LVA via `initialSortCaptures`

### History Tables

1. **Main history**: `ButterflyArray[2][64][64]` = [side][from][to] (standard butterfly)
   - Bonus: `base(-10) + slope(25)*d + slope2(5)*d^2`, capped at d=17
   - Gravity: `val -= val * bonus / 2048` (then +/- bonus)
   - **Negative history**: When a non-pawn quiet move is the best, the reverse direction `[to][from]` gets a negative update. This is a novel anti-history heuristic

2. **Counter-move history**: `PieceTypeToMatrix[8][64][8][64]` = [prevPieceType][prevDest][currPieceType][currDest]
   - Same bonus formula and gravity as main history

3. **Follow-up move history** (continuation at ply-2): `PieceTypeToMatrix[8][64][8][64]`
   - Same structure as counter-move history

4. **Capture history**: `CaptureHistoryArray[16][64][8]` = [piece][dest][capturedPieceType]
   - Separate bonus: `base(-129) + slope(440)*d + slope2(4)*d^2`, capped at d=10
   - Gravity divisor: 2048
   - Used in capture futility with divisor 65
   - Used in capture ordering with divisor 111

5. **Counter moves**: `PieceToArray[16][64]` = [piece][dest] -> Move (single best response)

6. **Killers**: 2 per ply, standard replacement

### History Score Composition

For quiet move ordering and LMR:
```
score = history[side][from][to]
      + counterMoveHistory[prevPiece][prevDest][piece][dest]
      + fuMoveHistory[prev2Piece][prev2Dest][piece][dest]
```

**Notable**: Arasan applies "negative history" (anti-history) for non-pawn quiets: when a quiet move is best, the reverse from/to pair gets a negative update. This penalizes retreats/reversals.

### History Update Skipping (from Ethereal)

When only 1 quiet move was tried and depth <= 3 plies, skip history updates entirely. Prevents inflating scores for forced moves.

---

## 6. Correction History

Shared across threads. Six tables:

1. **Pawn correction**: `[2][16384]` indexed by `pawnHash % 16384`, weight=28
2. **Non-pawn correction (White)**: `[2][16384]` indexed by `nonPawnHash(White) % 16384`, weight=21
3. **Non-pawn correction (Black)**: `[2][16384]` indexed by `nonPawnHash(Black) % 16384`, weight=21
4. **Minor piece correction (White)**: `[2][16384]` indexed by `minorPieceHash(White) % 16384`, weight=13
5. **Minor piece correction (Black)**: `[2][16384]` indexed by `minorPieceHash(Black) % 16384`, weight=13
6. **Continuation correction**: `[2][384][384]` indexed by [side][pieceToIndex(prev1)][pieceToIndex(prev2)] and also [prev1][prev4], weight=60 each. The ply-4 entry gets half bonus.

Correction applied to eval in both search and qsearch. Max correction value: 1024. Scale: `/512`. Eval divisor for bonus: 663. Max bonus: 156.

Update formula: `val = val + bonus - val * |bonus| / 1024`

**Difference from GoChess**: GoChess has only basic pawn correction history. Arasan has 6 tables including non-pawn, minor piece, and continuation correction. The minor piece and continuation corrections are notable additions we don't have.

---

## 7. Time Management

### Base Time Allocation

```
time_target = factor * (moves_left-1)*inc + time_left) / moves_left - move_overhead
```

Where:
- `moves_left` defaults to 28 (with increment) or 40 (sudden death with no increment)
- Factor reduces near time control: `1.0 - 0.05*(6-moves_left)` when moves_left < 6
- Ponder bonus: 1.4x when pondering and moves_left > 4

### Extra Time Budget

- With increment: `extra = max(0, -2.5*target + 2*(2.5/8)*time_left)` when `time_left < 8*target`
- Without increment: `extra = max(0, -2.5*target + 2*(2.5/12)*time_left)` when `time_left < 12*target`, reduced further when time_left < 500ms
- Otherwise: `extra = 2.5 * target`

### Search History Time Adjustment

After each iteration completes, Arasan adjusts the time target based on:

1. **PV change factor**: Looks back up to 6 iterations. Each PV change contributes `1/2^i` (halving weight for older iterations). Range [0, ~2].
2. **Score drop**: `max(0, max_recent_score - current_score - 0.1P)`, normalized by PAWN_VALUE.
3. **Boost formula**: `min(1.0, 0.25*pvChange + scoreDrop*(1.5 + 0.25*pvChange))`
4. **Reduction**: When boost is 0 and max past boost was < 0.25: `1.0 - maxBoost - 0.75 * maxBoostDepth/currentDepth`
5. **Applied as**: `bonus_time = extra_time * boostFactor` or `bonus_time = -floor(reductionFactor * elapsed/3)`

### Fail High/Low Time Extension

- **Fail low at root**: extend by full extra_time
- **Fail high at root**: extend by extra_time/2
- These extensions are temporary and cleared when the fail resolves

### Time Check

Adaptive interval: recalculated as `20 * num_nodes * factor / (elapsed * 16)`. Reduced for short time controls (< 1s).

---

## 8. Quiescence Search

- Depth parameter: always <= 0, decremented each recursive call
- Stand pat: eval with correction history applied, refined by hash table
- Hash table: probed at depth 0, stores results
- Hash move: used in QS if it's a capture/promotion (or if from a depth-0 hash entry)
- Futility pruning: `Gain(move) + 1.4P + eval < best_score` (skip non-promoting, non-PP captures)
- SEE pruning: `seeSign(move, max(0, best_score - eval - 1.25P))`, skips discovered check candidates
- Evasions in QS: when in check, searches all evasions via full MoveGenerator. Limits non-capture evasions to `max(1+depth, 0)` then prunes remaining quiet non-checking evasions

---

## 9. Lazy SMP Details

- Depth variation per thread: non-main threads skip depths based on thread ID and a search_count tracking system
  - Thread starts: offset by `(id+1) % 2` and `(id+1) % 4` at first iteration
  - Later: increments depth to avoid duplicating searches that other threads are already doing
- Best thread selection: highest score at greatest completed depth (with mate score override)
- Monitor thread: one thread handles UCI output and input checking, reassigned if that thread finishes

---

## 10. Notable/Unusual Features

### Negative History (Anti-History)

When a non-pawn quiet move is the best move, Arasan updates `history[side][TO][FROM]` (reversed!) with a negative bonus. This penalizes the reverse of the best move, making the engine less likely to "take back" moves in similar positions. This is novel and not seen in Stockfish.

### IID Instead of IIR

Arasan uses full Internal Iterative Deepening (a recursive search at reduced depth) instead of the simpler IIR (just reducing depth by 1). This is more expensive but may produce better move ordering.

### Null Move Verification

At depth >= 6, after a null move produces a cutoff, Arasan verifies it with a reduced-depth search with the VERIFY flag. This prevents false null-move cutoffs in zugzwang positions. Most modern engines have dropped this.

### Null Move Hash Skip

If the hash table has an upper bound entry with sufficient depth and value < beta, skip the null move search entirely. This saves the cost of a null move search that would likely fail.

### RFP Returns (eval+beta)/2

Instead of returning just eval, Arasan returns the average of eval and beta. This is more conservative and avoids inflated scores.

### ProbCut Skips Pawn Captures

The ProbCut move generator explicitly excludes pawn captures (`~board.pawn_bits[opponent]`), since pawns are too low value to reach the ProbCut threshold.

### Triple Singular Extension

Arasan extends singular moves by up to 3 plies, depending on how far below the singular threshold the non-hash moves scored. Most engines cap at double.

### History Update Skip at Low Depth

From Ethereal: when only 1 quiet was tried and depth <= 3 plies, history updates are skipped entirely.

### Six Correction History Tables

Arasan uses pawn hash, non-pawn hash (per side), minor piece hash (per side), and continuation (prev1 x prev2 and prev1 x prev4) correction tables. The minor piece correction is unusual.

### Strength Reduction Feature

Built-in strength reduction with depth limits, suboptimal move selection with calibrated probabilities per strength level, and time-wasting to simulate human-like play.

---

## 11. Summary: Features We (GoChess) Might Not Have

| Feature | Arasan | GoChess |
|---------|--------|---------|
| Full IID (not just IIR) | Yes (depth >= 6/8) | IIR only |
| Negative/anti-history | Yes (reverse from/to) | No |
| Null move verification | Yes (depth >= 6) | No |
| Null move hash skip | Yes (UB entry check) | No |
| RFP returns (eval+beta)/2 | Yes | Returns eval |
| ProbCut skips pawn captures | Yes | No |
| Triple singular extension | Yes (up to 3 plies) | No (up to 1 ply) |
| Minor piece correction history | Yes | No |
| Non-pawn correction history | Yes (per side) | No |
| Continuation correction history | Yes (prev1xprev2, prev1xprev4) | No |
| Capture futility with history | Yes (capHist/65) | No |
| History update skip (low depth, 1 quiet) | Yes | No |
| LMR killer/counter bonus (-1 ply) | Yes | No |
| LMR non-PV non-improving malus (+1 ply) | Yes | No |
