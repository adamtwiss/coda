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

Steps: `[0.375P, 0.75P, 1.5P, 3.0P, 6.0P, MATE]` (6 steps). On fail high/low, widen to the next step.

At low iterations (`<= widePlies`), the window is wider.

Compare to Coda: we use delta=15 from depth 4. Arasan's stepped approach with 6 predefined widths is more structured.

### Internal Iterative Deepening (IID)

When no hash move exists:
- PV nodes: depth >= 8 plies (16 internal)
- Non-PV nodes: depth >= 6 plies (12 internal)
- Full IID search (not just IIR): calls `search()` recursively with IID flag set at reduced depth

Compare to Coda: we use IIR (reduce depth by 1 at depth >= 4). Arasan does a full recursive search, which is more expensive but may find better hash moves.

---

## 2. Pruning Techniques

### 2a. Reverse Futility Pruning (Static Null Pruning)

- Depth guard: depth <= 6 plies (12 internal)
- Margin: `max(0.25P, 0.75P * (depth_plies - improving))` (formula from Obsidian)
- **Return value: `(eval + beta) / 2`** (average, not just eval or beta -- more conservative)
- Requires `eval < TABLEBASE_WIN`

With P=128: margins at depth 1 = 32 (min), depth 6 non-improving = 576 (4.5P), depth 6 improving = 480 (3.75P).

Compare to Coda: we use 70*d (improving) / 100*d (not improving), depth<=7, returns eval-margin. Arasan's `(eval+beta)/2` return is a soft-bounded alternative.

### 2b. Null Move Pruning

- Depth guard: depth >= 2 plies (4 internal)
- Additional guard: at depth < 4 plies, skip NMP if previous move was a capture/promotion
- Verification: `staticEval >= beta - 0.25P * (depth_plies - 6)` OR `depth >= 12 plies`
- Reduction formula: `R = 3 + depth/3 + min(3, (eval-beta)/(175*P/128))`
  - Low material adjustment: when material level <= 3, uses `depth/(3 + 3)` instead of `depth/3`
- Hash-based skip: avoids null move if hash entry is upper bound with depth >= null_depth and value < beta
- Null move verification: at depth >= 6, re-search at null depth with VERIFY flag
- Mate score capping: if null move search returns >= MATE_RANGE, return beta instead

Compare to Coda: we use R = 4 + d/3 + min((eval-beta)/200, 3) with R-1 after captures and NMP score dampening. Arasan's formula is similar but with hash-based skip and verification. We don't have hash-based NMP skip.

### 2c. Razoring

**DISABLED** by default (behind `#ifdef RAZORING`). When enabled:
- Depth <= 1 ply, margin = `2.75P + depth_plies * 1.25P`
- Compare to Coda: we also don't have razoring in current code.

### 2d. Futility Pruning (Quiet Moves)

- Depth guard: `pruneDepth <= 8 plies` (uses LMR-reduced depth)
- History threshold: only prune if `hist < 1000` (improving) or `hist < 2000` (non-improving)
- Margin: `0.25P + max(1, depth_plies) * 0.95P`

With P=128: margins are 153 (d1), 275 (d2), 397 (d3), ... 1000 (d8).

Compare to Coda: we use 90+lmrDepth*100 with history/128 adjustment. Similar approach.

### 2e. Futility Pruning (Captures)

- Depth guard: `pruneDepth <= 5 plies`
- Margin: `2.0P + depth_plies * 2.5P + maxValue(move) + captureHistory/65`
- **Uses capture history in the margin** -- captures with good history get wider margins

Compare to Coda: we have bad-noisy pruning (eval+depth*75<=alpha for losing captures) but not capture-history-adjusted futility.

### 2f. Late Move Pruning (LMP)

- Depth guard: depth_plies <= 9
- Move count formula: `LMP_BASE + LMP_SLOPE * d^(LMP_EXP/100) / 100`
  - Improving: base=4, slope=75, exp=2.04 => e.g. d1=5, d3=10, d5=20, d9=61
  - Non-improving: base=2, slope=70, exp=1.95 => e.g. d1=3, d3=8, d5=15, d9=45
- Only prunes moves in HISTORY_PHASE or later (not hash, captures, killers, counter)

Compare to Coda: we use 3+d^2 (depth<=8) with improving/failing adjustments.

### 2g. History Pruning

- Depth guard: `pruneDepth <= (3 - improving)` plies
- Condition: counter-move history < -250 AND follow-up history < -250
- Uses individual continuation histories, NOT combined score

Compare to Coda: we use -1500*depth threshold on combined history, depth<=3.

### 2h. SEE Pruning

- Depth guard: `pruneDepth <= 7 plies`
- Quiet threshold: `-depth_plies * 0.75P` (= -96 per ply with P=128)
- Capture threshold: `-depth_plies^2 * 0.2P` (= -25.6*d^2 with P=128)

Compare to Coda: quiet -20*d^2, capture -d*100. Arasan's quiet threshold is linear (not quadratic).

### 2i. ProbCut

- Depth guard: depth >= 5 plies. Re-allows at depth >= 9 if already in ProbCut.
- ProbCut beta: `beta + 1.25P` (= beta + 160)
- Search depth: `depth - 4 plies - depth/8`
- **Skips pawn captures** (too low value to reach ProbCut threshold)
- SEE filter, hash skip
- Stores result in hash if successful

Compare to Coda: we use beta+170, depth>=5 (currently disabled). Arasan's pawn-capture skip is a nice optimization.

### 2j. Multi-Cut (from Singular Extension)

- When singular search fails (hash move not singular) and singular_beta >= beta: return singular_beta
- Compare to Coda: **IMPLEMENTED** -- we have multi-cut from singular extension.

---

## 3. Extensions

- **Check extension**: 1 ply when move gives check
- **Passed pawn push extension**: 1 ply when pawn moves to 7th rank
- **Capture of last piece**: 0.5 ply when capturing opponent's last non-pawn piece in endgame
- **Singular extension**: 1-3 plies depending on singularity
  - Depth guard: depth >= 8 plies, hash entry is lower bound, hash depth >= depth-3
  - Singular margin: roughly `depth_plies/64` pawns
  - **Triple extension (3 plies)**: non-PV and result < `nu_beta - SINGULAR_EXTENSION_TRIPLE` (0.33P = 42cp)
  - **Double extension (2 plies)**: non-PV and result < `nu_beta - SINGULAR_EXTENSION_DOUBLE` (0)
  - Single extension (1 ply): default case
  - Negative extension (-2 plies): when hash value >= beta but not singular, or at cut nodes
- Extensions capped at 1 ply total (for regular; singular can go higher)

**Notable**: Arasan allows triple singular extensions (3 plies). Verified in source: `SINGULAR_EXTENSION_TRIPLE = 0.33*PAWN_VALUE`. Any singularity below nu_beta gives double extension, and below nu_beta-42 gives triple.

Compare to Coda: we have singular extensions with multi-cut and negative extension (-1). No double or triple extension. We have recapture extensions but no check extensions.

---

## 4. LMR (Late Move Reductions)

### Formula

```
LMR_REDUCTION[pv][depth][moves] = floor(2 * (base + log(d) * log(m) / div) + 0.5) / 2
```

Half-ply granularity. Parameters:
- PV: base = 0.30, div = 2.25
- Non-PV: base = 0.50, div = 1.80

### Conditions

- Depth >= 3 plies
- Move index >= 1 + 2*isPV (>= 3 for PV, >= 2 for non-PV)
- Either quiet move, OR capture with move_index > LMP count

### Adjustments

1. **Captures**: reduction -= 1 ply
2. **Non-PV + non-improving**: reduction += 1 ply
3. **Killer/counter move** (not in check): reduction -= 1 ply
4. **History-based**: reduction -= `clamp(historyScore / 2200, -2, +2)` plies

Compare to Coda: separate quiet/capture LMR tables (C=1.30/1.80). History divisor 5000 (vs Arasan's 2200). Coda exempts killers from LMR entirely (not just reduced). Coda has doDeeper/doShallower.

---

## 5. Move Ordering

### Stages (MoveGenerator::Phase)

1. **HASH_MOVE_PHASE**: TT move
2. **WINNING_CAPTURE_PHASE**: Captures sorted by MVV-LVA (`8*Gain - pieceValue[moved]`)
3. **KILLER1_PHASE**: Primary killer
4. **KILLER2_PHASE**: Secondary killer
5. **COUNTER_MOVE_PHASE**: Counter move
6. **HISTORY_PHASE**: Quiet moves sorted by history score
7. **LOSERS_PHASE**: Losing captures (SEE < 0)

### History Tables

1. **Main history**: `ButterflyArray[2][64][64]` = [side][from][to]
   - **Negative history**: When a non-pawn quiet is best, the reverse `[to][from]` gets a negative update. Novel anti-history heuristic (verified in source: `searchc.cpp:187`).
   - Bonus: `base(-10) + slope(25)*d + slope2(5)*d^2`, capped at d=17
   - Gravity: `val -= val * bonus / 2048`

2. **Counter-move history**: `PieceTypeToMatrix[8][64][8][64]` = [prevPieceType][prevDest][currPieceType][currDest]
   - Same bonus/gravity as main history

3. **Follow-up move history** (continuation at ply-2): same structure as counter-move history

4. **Capture history**: `CaptureHistoryArray[16][64][8]` = [piece][dest][capturedPieceType]
   - Separate bonus: `base(-129) + slope(440)*d + slope2(4)*d^2`, capped at d=10
   - Used in capture futility with divisor 65

5. **Counter moves**: `PieceToArray[16][64]` = [piece][dest] -> Move

6. **Killers**: 2 per ply

### History Score Composition

```
score = history[side][from][to]
      + counterMoveHistory[prevPiece][prevDest][piece][dest]
      + fuMoveHistory[prev2Piece][prev2Dest][piece][dest]
```

Compare to Coda: we have main history [color][from][to], capture history [piece][to][victim], continuation history [piece][to][piece][to], and pawn history [pawnHash%512][piece][to]. Arasan has the reverse-direction anti-history and follow-up (ply-2) continuation history which we don't.

### History Update Skipping (from Ethereal)

When only 1 quiet was tried and depth <= 3, skip history updates entirely. Prevents inflating scores for forced moves.

Compare to Coda: we don't have this optimization.

---

## 6. Correction History

Six tables (shared across threads):

1. **Pawn correction**: `[2][16384]` indexed by pawnHash, weight=28
2. **Non-pawn correction (White)**: `[2][16384]` indexed by nonPawnHash(White), weight=21
3. **Non-pawn correction (Black)**: `[2][16384]` indexed by nonPawnHash(Black), weight=21
4. **Minor piece correction (White)**: `[2][16384]` indexed by minorPieceHash(White), weight=13
5. **Minor piece correction (Black)**: `[2][16384]` indexed by minorPieceHash(Black), weight=13
6. **Continuation correction**: `[2][384][384]` indexed by previous moves, weight=60 each. Ply-4 entry gets half bonus.

Correction applied to eval in both search and qsearch. Max correction: 1024. Scale: `/512`.

Compare to Coda: we have multi-source correction history: pawn (512/1024) + white-NP (204/1024) + black-NP (204/1024) + continuation (104/1024). Arasan adds **minor piece correction** (per side) which we don't have. Arasan also uses continuation correction from ply-4 (we don't). Arasan's table sizes are much larger (16384 vs our 512/204/104).

---

## 7. Time Management

### Base Time Allocation

```
time_target = factor * ((moves_left-1)*inc + time_left) / moves_left - move_overhead
```

Where:
- `moves_left` defaults to 28 (with increment) or 40 (sudden death with no increment)
- Factor reduces near time control: `1.0 - 0.05*(6-moves_left)` when moves_left < 6

### Extra Time Budget

Varies based on time remaining relative to target.

### Search History Time Adjustment

After each iteration:
1. **PV change factor**: Looks back up to 6 iterations with halving weight
2. **Score drop**: tracks max recent score minus current
3. **Boost formula**: `min(1.0, 0.25*pvChange + scoreDrop*(1.5 + 0.25*pvChange))`
4. Applied to extra time allocation

Compare to Coda: we use simple time allocation (timeLeft/movesLeft + 80% inc, cap 50%). No PV change tracking or score-drop time extension.

### Fail High/Low Extension

- **Fail low at root**: extend by full extra_time
- **Fail high at root**: extend by extra_time/2

---

## 8. Quiescence Search

- Stand pat with correction history applied, refined by hash table
- Hash table probed at depth 0
- Futility: `Gain(move) + 1.4P + eval < best_score` => skip
- SEE pruning: `seeSign(move, max(0, best_score - eval - 1.25P))`
- Evasions in QS: when in check, searches all evasions. Limits non-capture evasions to `max(1+depth, 0)`.

---

## 9. Lazy SMP Details

- Depth variation per thread: non-main threads skip depths based on thread ID
- Best thread selection: highest score at greatest completed depth
- Monitor thread: handles UCI output and input checking

---

## 10. Notable/Unusual Features

### Negative History (Anti-History)
When a non-pawn quiet move is the best move, Arasan updates `history[side][TO][FROM]` (reversed!) with a negative bonus. Penalizes the reverse of the best move. Verified in source at `searchc.cpp:187`.
Compare to Coda: we don't have this.

### IID Instead of IIR
Full recursive search at reduced depth instead of just reducing depth by 1. More expensive but potentially better move ordering.

### Null Move Verification
At depth >= 6, after null move cutoff, re-search with VERIFY flag. Prevents false cutoffs in zugzwang.
Compare to Coda: we use NMP score dampening instead of verification.

### Null Move Hash Skip
If hash has upper bound entry with sufficient depth and value < beta, skip null move entirely.
Compare to Coda: we don't have this.

### RFP Returns (eval+beta)/2
More conservative than returning just eval. Avoids inflated scores.
Compare to Coda: we return eval-margin.

### ProbCut Skips Pawn Captures
Pawn captures excluded from ProbCut -- too low value to reach threshold.
Compare to Coda: we don't filter ProbCut moves by piece type.

### Triple Singular Extension
Up to 3 plies extension. Verified in source: `SINGULAR_EXTENSION_TRIPLE = 0.33*PAWN_VALUE`.
Compare to Coda: we only do single singular extension (+1).

### History Update Skip at Low Depth
From Ethereal: when only 1 quiet tried at depth <= 3, skip history updates.

### Six Correction History Tables
Pawn, non-pawn (per side), minor piece (per side), continuation (prev1xprev2 and prev1xprev4).
Compare to Coda: we have 4 sources (pawn + NP per side + continuation). Arasan adds minor piece correction and ply-4 continuation.

---

## 11. Summary: Differences from Coda

| Feature | Arasan | Coda |
|---------|--------|------|
| Full IID (not just IIR) | Yes (depth >= 6/8) | IIR only (depth >= 4) |
| Negative/anti-history | Yes (reverse from/to) | No |
| Null move verification | Yes (depth >= 6) | No (NMP score dampening) |
| Null move hash skip | Yes (UB entry check) | No |
| RFP returns (eval+beta)/2 | Yes | Returns eval-margin |
| RFP depth | <=6 | <=7 |
| ProbCut skips pawn captures | Yes | No |
| Triple singular extension | Yes (up to 3 plies) | Single only (+1) |
| Double singular extension | Yes (default) | No |
| Minor piece correction history | Yes (per side) | No |
| Continuation correction (ply-4) | Yes | No |
| Capture futility with capHist | Yes (capHist/65) | No |
| History update skip (1 quiet, d<=3) | Yes | No |
| LMR killer/counter bonus (-1 ply) | Yes | Full exemption |
| LMR non-PV non-improving malus (+1) | Yes | No |
| Check extension | Yes (+1 ply) | No |
| Passed pawn push extension | Yes (7th rank) | No |
| Last piece capture extension | Yes (0.5 ply) | No |
| Multi-cut (from SE) | Yes | Yes (IMPLEMENTED) |
| Singular extensions | Yes (depth >= 8) | Yes (depth >= 8) |
| Recapture extensions | No | Yes (IMPLEMENTED) |
| Hindsight depth adjustment | No | Yes (IMPLEMENTED) |
| NMP R-1 after captures | No | Yes (IMPLEMENTED) |
| NMP score dampening | No | Yes (IMPLEMENTED) |
| Fail-high score blending | No | Yes (IMPLEMENTED) |
| TT score dampening | No | Yes (IMPLEMENTED) |
| TT near-miss cutoffs | No | Yes (IMPLEMENTED) |
| Pawn history | No | Yes (IMPLEMENTED) |
| Bad noisy pruning | No | Yes (IMPLEMENTED) |

### Parameter Comparison

| Parameter | Arasan | Coda |
|-----------|--------|------|
| RFP margin (d6, not imp) | 576 (4.5P) | 600 (100*6) |
| RFP margin (d6, improving) | 480 (3.75P) | 420 (70*6) |
| NMP R formula | 3+d/3+min(3,(eval-beta)/175) | 4+d/3+min(3,(eval-beta)/200) |
| LMR quiet div | 1.80 (non-PV) | 1.30 |
| LMR PV div | 2.25 | same table |
| LMR history div | 2200 | via adjustment |
| Futility margin (d4) | 519 (4.06P) | 490 (90+4*100) |
| SEE quiet | -96*d (linear) | -20*d^2 |
| SEE capture | -25.6*d^2 | -d*100 (linear) |
| Singular margin | ~d/64 pawns | tt_score-depth |
| ProbCut beta | +160 | +170 |
| Aspiration | 6 steps: 48..768 | delta=15 |
| Correction tables | 6 | 4 sources |
| History bonus | -10+25*d+5*d^2 | d^2 capped 1200 |

---

## 12. Ideas Worth Testing from Arasan

### High Priority
1. **Double/triple singular extension** -- Extend +2/+3 when hash move is very singular. Arasan's thresholds: double at any singularity, triple at -42cp below. Could give better tactical resolution.
2. **Negative/anti-history** -- Update `history[to][from]` (reversed) with negative bonus when a non-pawn quiet is best. Novel way to penalize retreats.
3. **Minor piece correction history** -- Per-side hash of minor pieces for eval correction. Arasan uses 6 correction tables total.

### Medium Priority
4. **Null move hash skip** -- If TT has upper bound with depth >= null_depth and value < beta, skip NMP entirely.
5. **RFP (eval+beta)/2 return** -- More conservative return value. Avoids score inflation.
6. **Capture futility with capture history** -- Use capHist/65 in capture futility margin.
7. **History update skip** -- Skip history updates when only 1 quiet tried at depth <= 3.
8. **ProbCut skip pawn captures** -- Filter low-value captures from ProbCut.
9. **Check extension** -- +1 ply when giving check. Consider SEE-filtered variant (Tucano).
10. **Continuation correction at ply-4** -- Add ply-4 move pair to correction history.

### Lower Priority
11. **Full IID** -- Recursive search instead of IIR. More expensive but potentially better ordering.
12. **LMR non-PV non-improving +1** -- Extra ply reduction for non-improving non-PV nodes.
13. **Last piece capture extension** -- +0.5 ply when capturing final enemy piece.
14. **Passed pawn push extension** -- +1 ply for pawn to 7th rank.
