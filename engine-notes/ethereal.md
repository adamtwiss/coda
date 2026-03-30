# Ethereal Engine Crib Notes

Source: `~/chess/engines/Ethereal/src/` (Ethereal 11.80+, by Andrew Grant)

---

## 1. Search Pruning Techniques

### Reverse Futility Pruning (Beta Pruning / Static Null Move Pruning)
- **Depth guard**: `depth <= 8`
- **Margin**: `65 * MAX(0, depth - improving)`
- **Condition**: `!PvNode && !inCheck && !excluded && eval - margin >= beta`
- **Returns**: `eval`
- **Note**: When improving, the effective margin is `65 * (depth - 1)`. When not improving, `65 * depth`. At depth 8 non-improving, margin = 520.

### Alpha Pruning (Deep Futility / Razoring equivalent)
- **Depth guard**: `depth <= 4`
- **Margin**: 3488 (fixed, not depth-scaled)
- **Condition**: `!PvNode && !inCheck && !excluded && eval + 3488 <= alpha`
- **Returns**: `eval`
- **Note**: This is unusual -- a fixed margin of 3488 for depths 1-4. Most engines use depth-scaled razoring. Ethereal's approach is more aggressive: if you're 3488cp below alpha at depth <= 4, just return eval.

### Null Move Pruning
- **Depth guard**: `depth >= 2`
- **Conditions**: `!PvNode && !inCheck && !excluded && eval >= beta && prev_move != NULL_MOVE && has_non_pawn_material && (!ttHit || !(ttBound & BOUND_UPPER) || ttValue >= beta)`
- **Reduction formula**: `R = 4 + depth/5 + MIN(3, (eval - beta) / 191) + (prev_node_was_tactical)`
  - Base: 4
  - Depth scaling: +depth/5
  - Eval scaling: +MIN(3, (eval - beta) / 191)
  - Tactical bonus: +1 if the previous move was a capture/promotion
- **Verification**: None (no verification search)
- **Mate clamping**: If value > TBWIN_IN_MAX, returns `beta` instead (avoids unproven mates)
- **TT guard**: Won't try NMP if TT says upper bound < beta (avoids wasting time when TT predicts fail-low)

### ProbCut
- **Depth guard**: `depth >= 5`
- **Margin**: `beta + 100`
- **Condition**: `!PvNode && !inCheck && !excluded && abs(beta) < TBWIN_IN_MAX && (!ttHit || ttValue >= rBeta || ttDepth < depth - 3)`
- **Search depth**: `depth - 4`
- **Two-tier verification**: At `depth >= 10` (2 * ProbCutDepth), first verifies with qsearch, then reduced search
- **Move source**: Noisy picker with SEE threshold `rBeta - eval`
- **TT store**: Stores result at `depth - 3` on success

### Futility Pruning (per-move, quiet only)
- **Depth guard**: `lmrDepth <= 8` (uses LMR-reduced depth, not raw depth)
- **Margin formula**: `77 + lmrDepth * 52`
  - At lmrDepth 1: 129
  - At lmrDepth 8: 493
- **Two variants**:
  - **14A (history-aware)**: `eval + margin <= alpha && hist < FutilityPruningHistoryLimit[improving]`
    - History limits: improving=14296, not_improving=6004
    - Sets `skipQuiets = 1` (skips all remaining quiets)
  - **14B (no-history)**: `eval + margin + 165 <= alpha` (adds extra 165cp margin, no history check)
    - Sets `skipQuiets = 1`

### Late Move Pruning (LMP / Move Count Pruning)
- **Depth guard**: `depth <= 8` (was 10 in the table, but LateMovePruningDepth=8)
- **Counts formula** (precomputed):
  - Not improving: `2.0767 + 0.3743 * depth^2`
  - Improving: `3.8733 + 0.7124 * depth^2`
  - Example depth 4: not_improving=8, improving=15
  - Example depth 8: not_improving=26, improving=49
- **Condition**: `best > -TBWIN_IN_MAX` (must have found a non-mated line)
- **Sets** `skipQuiets = 1`

### Continuation History Pruning
- **Depth guard**: `lmrDepth <= ContinuationPruningDepth[improving]`
  - Improving: depth <= 3
  - Not improving: depth <= 2
- **History threshold**: `MIN(cmhist, fmhist) < ContinuationPruningHistoryLimit[improving]`
  - Improving: < -1000
  - Not improving: < -2500
- **Condition**: Only after killer/counter stage (`stage > STAGE_COUNTER_MOVE`), quiet moves only
- **Prunes individual move** (continue, not skipQuiets)

### SEE Pruning
- **Depth guard**: `depth <= 10`
- **Margins**:
  - Quiet: `-64 * depth` (linear). At depth 5: -320
  - Noisy: `-20 * depth^2` (quadratic). At depth 5: -500
- **History adjustment**: SEE threshold adjusted by `-hist / 128` (history improves/worsens threshold)
- **Optimization**: Skips SEE check for moves in STAGE_GOOD_NOISY (assumed positive SEE)

### Mate Distance Pruning
- Standard: `rAlpha = MAX(alpha, -MATE + height)`, `rBeta = MIN(beta, MATE - height - 1)`
- Applied in non-root nodes

### TT Research Margin (unusual)
- **Margin**: 141
- **Condition**: `!PvNode && ttDepth >= depth - 1 && (ttBound & BOUND_UPPER) && (cutnode || ttValue <= alpha) && ttValue + 141 <= alpha`
- **Effect**: Accepts TT entries one depth shallower if they show the position is far enough below alpha
- **WE DON'T HAVE THIS** - this is a soft TT cutoff for shallow entries

---

## 2. Move Ordering

### Stages (in order)
1. **STAGE_TABLE** - TT move
2. **STAGE_GENERATE_NOISY** - Generate all noisy (captures + promotions)
3. **STAGE_GOOD_NOISY** - Noisy moves with positive history, passing SEE threshold
4. **STAGE_KILLER_1** - First killer move
5. **STAGE_KILLER_2** - Second killer move
6. **STAGE_COUNTER_MOVE** - Counter move
7. **STAGE_GENERATE_QUIET** - Generate all quiet moves
8. **STAGE_QUIET** - Quiet moves sorted by history
9. **STAGE_BAD_NOISY** - Noisy moves that failed SEE (losing captures)

### Capture/Noisy Scoring
- **Formula**: `64000 + capture_history + MVVAugment[captured_piece]`
- **MVV augments**: Pawn=2400, Knight=2400, Bishop=4800, Rook=9600, Queen not listed (would be higher)
- **Capture history** indexed by: `[piece_type][threat_from][threat_to][to_square][captured_piece]`
  - Dimensions: [6][2][2][64][5] = 15,360 entries
  - `threat_from`: whether the moving piece is on a square attacked by opponent
  - `threat_to`: whether destination is attacked by opponent
- **Queen promotions**: Get +64000 bonus (ensures they sort to top)
- **SEE gating**: Moves with negative capture history are flagged as -1 (bad noisy), deferred to STAGE_BAD_NOISY

### Quiet Scoring
- **Combined score**: `counter_move_history + followup_move_history + butterfly_history`
- **Butterfly history**: `[color][threat_from][threat_to][from][to]`
  - Dimensions: [2][2][2][64][64] = 65,536 entries
  - **Threat-aware**: indexed by whether piece evades/enters a threatened square
- **Counter-move history (continuation history ply-1)**: `[tactical][piece][to][0][piece][to]`
  - First index: whether previous move was tactical (capture/promo)
  - Dimensions: [2][6][64][2][6][64] = 589,824 entries total for continuation table
- **Followup-move history (continuation history ply-2)**: Same table, second CONT_NB index

### Refutation Moves
- **2 killer moves** per ply (standard)
- **1 counter move**: indexed by `[!color][prev_movedPiece][prev_to]`
  - Dimensions: [2][6][64] = 768 entries

### History Update Formula
- **Stat bonus**: `depth > 13 ? 32 : 16*depth^2 + 128*MAX(depth-1, 0)` (from Stockfish)
  - depth 1: 16, depth 5: 912, depth 10: 2752, depth 13: 3840, depth 14+: 32
- **Gravity formula**: `current += delta - current * abs(delta) / 16384`
- **HistoryDivisor**: 16384 (effective max history value)
- **Skip update optimization**: Skips history update if `depth == 0` or `(length == 1 && depth <= 3)` (easy cutoffs don't deserve history credit)

### Threat-Aware History (WE DON'T HAVE THIS)
Both butterfly history and capture history are indexed by two boolean flags:
- `threat_from`: is the origin square attacked by the opponent?
- `threat_to`: is the destination square attacked by the opponent?

This quadruples the history table size but provides much better differentiation. A knight retreating from a threatened square has different history than a knight advancing to a safe square.

### Tactical Continuation History (WE DON'T HAVE THIS)
Continuation history is split into two sub-tables based on whether the previous move was tactical (capture/promotion). Index: `continuation[tactical][piece][to][cont_idx][piece][to]`. This means quiet-after-capture sequences have separate statistics from quiet-after-quiet sequences.

---

## 3. Time Management

### Initial Allocation
- **With moves-to-go (X/Y+Z)**:
  - `ideal = 1.80 * (time - overhead) / (mtg + 5) + inc`
  - `max = 10.00 * (time - overhead) / (mtg + 10) + inc`
- **Sudden death (X+Y)**:
  - `ideal = 2.50 * ((time - overhead) + 25 * inc) / 50`
  - `max = 10.00 * ((time - overhead) + 25 * inc) / 50`
- **Both capped** to `time - overhead`
- **MoveOverhead default**: 300ms

### Dynamic Time Adjustment (tm_finished)
Three multiplicative factors applied to `ideal_usage`:

1. **PV Stability Factor**: `1.20 - 0.04 * pv_stability`
   - pv_stability: 0 to 10 (incremented each depth with same best move, reset to 0 on change)
   - Range: 0.80x (very stable) to 1.20x (just changed)

2. **Score Factor**: `MAX(0.75, MIN(1.25, 0.05 * score_change))`
   - score_change = score 3 depths ago minus current score
   - If score is dropping (positive change), allocate more time (up to 1.25x)
   - If score is rising (negative change), allocate less time (down to 0.75x)

3. **Node Distribution Factor**: `MAX(0.50, 2 * non_best_pct + 0.4)`
   - non_best_pct = fraction of total nodes NOT spent on best move
   - Range: 0.50x (almost all nodes on best move) to 2.40x (nodes spread across many moves)
   - Uses per-move node counting via `tm->nodes[move]` (64K entry table indexed by move encoding)

### Hard Stop
- `max_usage` is the absolute maximum
- Checked every 1024 nodes via `tm_stop_early()`
- Ratio of max to ideal: approximately 4x-5.5x

### Aspiration Window
- **Initial delta**: 10
- **Activation depth**: >= 4
- **On fail-low**: `beta = (alpha+beta)/2`, widen alpha by delta, reset depth
- **On fail-high**: Widen beta by delta, reduce depth by 1 (if score is not a mate)
- **Delta growth**: `delta += delta / 2` (1.5x each iteration)
- **Timer**: Reports UCI info if search takes > 2500ms

---

## 4. LMR Details

### Base Table
```
LMRTable[depth][played] = 0.7844 + log(depth) * log(played) / 2.4696
```
(64x64 precomputed table, natural log)

Example values (depth, played -> R):
- (4, 4): 0.7844 + 1.386 * 1.386 / 2.4696 = ~1.56 -> 1
- (8, 8): 0.7844 + 2.079 * 2.079 / 2.4696 = ~2.53 -> 2
- (16, 16): 0.7844 + 2.773 * 2.773 / 2.4696 = ~3.90 -> 3

### Quiet Move Reductions
Applied when `depth > 2 && played > 1`:

```
R  = LMRTable[depth][played]       // Base from table
R += !PvNode + !improving           // +1 for non-PV, +1 for non-improving (0 to +2)
R += inCheck && king_move           // +1 if evading check with king move
R -= (stage < STAGE_QUIET)          // -1 for killer/counter moves
R -= hist / 6167                    // History adjustment (can be +/- several plies)
```

- **History divisor for LMR**: 6167
- With HistoryDivisor=16384, max history ~16384, so max history reduction: 16384/6167 = ~2.6 plies
- **Clamped**: `R = MIN(depth - 1, MAX(R, 1))` (never extends, never drops into QS)

### Noisy Move Reductions (captures/promotions)
```
R  = 3 - hist / 4952               // Base 3, adjusted by capture history
R -= !!board->kingAttackers         // -1 if move gives check
```
- **Capture history divisor for LMR**: 4952
- **Clamped** same as quiets

### Re-search Logic After LMR
If LMR search returns `value > alpha && R > 1`:
1. Adjust newDepth: `+1 if value > best + 35`, `-1 if value < best + newDepth`
2. If adjusted depth > lmrDepth, do a null-window re-search at adjusted depth
3. If still > alpha, do full-window PVS re-search

---

## 5. Singular Extensions

### Conditions
- `!RootNode && depth >= 8 && move == ttMove && ttDepth >= depth - 3 && (ttBound & BOUND_LOWER)`

### Search
- `rBeta = MAX(ttValue - depth, -MATE)`
- Excluded search: `search(rBeta-1, rBeta, (depth-1)/2, cutnode)` with TT move excluded

### Extension Values
- **Double extension (+2)**: `!PvNode && value < rBeta - 16 && dextensions <= 6`
- **Singular (+1)**: `value < rBeta` (no other move beats reduced beta)
- **Negative extension (-1)**: `ttValue >= beta` (multicut) OR `ttValue <= alpha` (tt already failing low)
- **No extension (0)**: Otherwise

### MultiCut
- If singular search `value >= rBeta && rBeta >= beta`, sets `stage = STAGE_DONE`
- Returns `MAX(ttValue - depth, -MATE)` immediately

### Double Extension Limit
- Tracked per-line as `dextensions` in NodeState
- Maximum 6 double extensions per line from root

---

## 6. Quiescence Search

### Standing Pat
- Standard: `best = eval; alpha = MAX(alpha, eval); if (alpha >= beta) return eval`

### Delta Pruning
- `MAX(QSDeltaMargin, bestCaseCapture) < alpha - eval` -> return eval
- QSDeltaMargin = 142

### Move Generation
- Noisy picker with SEE threshold: `MAX(1, alpha - eval - 123)`
- QSSeeMargin = 123 (so captures must win at least `alpha - eval - 123` by SEE)

### QS Short-Circuit (UNUSUAL)
```c
pessimism = estimatedValue - SEEPieceValues[moving_piece];
if (eval + pessimism > beta && abs(eval + pessimism) < MATE/2) return beta;
```
After applying a capture, if even the worst case (losing our piece immediately) would still beat beta, just return beta without searching deeper. This is an aggressive QS pruning that avoids deep QS trees.

### TT in QS
- Full TT probing with cutoffs (exact, lower/upper bound)
- TT move used in QS (via noisy picker tt_move parameter = NONE_MOVE though, so not actually using TT move in QS generation)
- Stores results back to TT at depth 0

---

## 7. Notable / Novel Features

### Things Ethereal Has That We Don't

1. **TT Research Margin** (TTResearchMargin = 141): Accept TT entries one ply shallower if they show the position is sufficiently below alpha. Avoids re-searching positions that are clearly bad. Worth investigating.

2. **Threat-Aware History Tables**: Butterfly history is [color][threat_from][threat_to][from][to] -- 4x larger than standard [color][from][to]. Capture history similarly uses threat_from/threat_to. This differentiates "retreating from danger" vs "advancing into danger" moves.

3. **Tactical Continuation History**: Continuation table split by whether parent move was tactical: `[tactical][piece][to][cont_idx][piece][to]`. Captures followed by specific responses have different statistics than quiet moves followed by the same responses.

4. **Alpha Pruning** (depth <= 4, margin 3488): A pre-search razoring that returns eval if it's hopelessly below alpha. We have razoring but the threshold/approach differs.

5. **QS Short-Circuit**: After applying a capture, if worst-case (losing the piece) still beats beta, return beta immediately without recursive qsearch. Saves nodes in clearly winning capture sequences.

6. **NMP TT Guard**: Won't attempt null move if TT entry has upper bound < beta, saving a wasted null-move search.

7. **NMP Tactical Bonus**: +1 to null move reduction if the previous move was a capture/promotion (opponent just made a tactical move, so null move is more likely to work). **(UPDATE 2026-03-21: GoChess now has NMP R-1 after captures -- the opposite direction, reducing R when last move was a capture, merged.)**

8. **ProbCut Two-Tier Verification**: At depth >= 10, ProbCut first verifies with qsearch before the reduced search. Prevents false ProbCut cutoffs at high depth.

9. **Aspiration Depth Reduction on Fail-High**: Reduces search depth by 1 on fail-high (if not a mate score). This is common but worth noting -- we should check if we do this.

10. **Double Extension Limit**: Maximum 6 double extensions per search line, tracked in NodeState.dextensions.

11. **IIR on PvNode too**: `depth >= 7 && (PvNode || cutnode) && (ttMove == NONE_MOVE || ttDepth + 4 < depth)` reduces depth by 1. Note it also triggers when TT entry exists but is too shallow (ttDepth + 4 < depth).

12. **History Skip for Easy Cutoffs**: Skips quiet history updates when `depth == 0` or `(only 1 quiet tried && depth <= 3)`. Prevents low-depth noise from polluting history tables.

13. **Cutnode TT Cutoffs**: TT cutoff condition includes `(cutnode || ttValue <= alpha)` -- cutnodes get more liberal TT cutoffs.

14. **Singular Negative Extension for Fail-Low TT**: If `ttValue <= alpha`, singular extension returns -1 (reduces depth). This is beyond the standard multicut negative extension.

### Comparison Table: Key Thresholds

| Feature | Ethereal | GoChess |
|---------|----------|---------|
| RFP depth | <= 8 | <= 8 (same) |
| RFP margin | 65/depth | 85 improving, 60 not |
| NMP min depth | 2 | 3 |
| NMP R formula | 4 + d/5 + min(3,(eval-beta)/191) + tactical | 3 + d/3 + (eval-beta)/200 |
| ProbCut depth | >= 5 | beta+200 |
| ProbCut margin | beta+100 | beta+200 |
| Futility margin | 77 + lmrDepth*52 | 100 + lmrDepth*100 |
| LMR base | 0.7844 + ln(d)*ln(m)/2.4696 | C=1.5 (need to verify formula) |
| LMR hist divisor | 6167 | 5000 |
| SEE quiet margin | -64*depth | (check) |
| SEE noisy margin | -20*depth^2 | (check) |
| Aspiration delta | 10 | 15 |
| Singular depth | >= 8 | (check) |
| Singular margin | ttValue - depth | depth*3 |
| History max | 16384 | 5000 |
| LMP improving d4 | 15 | ~19 |
| LMP not-impr d4 | 8 | ~8 |

### Architecture Notes
- Uses `setjmp/longjmp` for search abort (avoids checking abort flag at every node)
- Board.threats computed after every move (bitboard of all opponent-attacked squares)
- Per-move node counting for time management (64K array indexed by uint16 move encoding)
- Draw score has small randomization: `1 - (nodes & 2)` (returns -1 or +1 to avoid draw blindness)
- Captures history uses MVV augmentation for sorting: P=2400, N=2400, B=4800, R=9600
- No check extension as a separate feature -- in-check extends by 1 via the `extension = inCheck` fallback when not singular
