# Crafty 25.2 - Crib Notes

Source: `~/chess/engines/crafty/` (version 25.2, last modified 08/03/16)

---

## 1. Search Architecture

Negamax alpha-beta with PVS, iterative deepening, aspiration windows.

- **Aspiration window**: +/- 16 centipawns centered on previous iteration score.
  On fail high/low, doubles the delta (starts at 16, doubles each time, caps at 10*PAWN_VALUE=1000, then goes to 99999 = full window).
- **PV node detection**: `pv_node = (alpha != beta - 1)` -- standard.
- **Max ply**: MAXPLY = 129.
- **Mate score**: MATE = 32768, TBWIN = 31000.
- **Piece values**: P=100, N=305, B=305, R=490, Q=1000.

---

## 2. Pruning Techniques

### 2a. Mate Distance Pruning

```c
alpha = Max(alpha, -MATE + ply - 1);
beta  = Min(beta,  MATE - ply);
if (alpha >= beta) return alpha;
```

Applied at every non-root node. Standard implementation.

### 2b. Null Move Pruning

**Formula**: `R = null_depth + depth / null_divisor` = **3 + depth/6** (integer division).

**Conditions** (ALL must be true):
- `do_null` flag set (no consecutive null moves)
- Not a PV node
- `depth > n_depth` where n_depth is dynamically set:
  - n_depth = 1 normally (if pieces > 9, or root moves > 17, or depth > 3)
  - n_depth = 3 otherwise (low piece count endgame with few root moves and shallow depth -- zugzwang protection)
- Not in check
- Side to move has at least one non-pawn piece (`TotalPieces(wtm, occupied) > 0`)

**Hash-based null move avoidance**: If TT probe finds that a null-move search at this position would fail low (stored score from a previous null-move search at appropriate depth was < beta), `do_null` is set to 0, skipping null move entirely. This is the `AVOID_NULL_MOVE` return from HashProbe.

**No verification search**: Crafty does NOT do a verification re-search after null move cutoff (unlike some engines). It simply stores and returns.

### 2c. Futility Pruning

**Depth guard**: `depth < FP_depth` where FP_depth = **7** (so plies 1-6 from QS).

**Condition**: `MaterialSTM(wtm) + FP_margin[depth] <= alpha && !pv_node`

**Margins** (indexed by depth remaining):
```
depth:  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
margin: 0  100  150  200  250  300  400  500  600  700  800  900 1000 1100 1200 1300
```

**Exceptions**: Not applied when:
- In check
- Move has an extension (and is a PV node)
- First move at this ply (order == 1)
- Move is a passed pawn push to 6th+ rank (`PawnPush()`)
- Move gives check

### 2d. Late Move Pruning (LMP)

**Depth guard**: `depth < LMP_depth` where LMP_depth = **16**.

**Condition**: `order > LMP[depth] && !pv_node && !check && alpha > -MATE+300 && !CaptureOrPromote(move)`

**LMP thresholds** computed as: `LMP[i] = LMP_base + pow(i + 0.5, LMP_scale)` where LMP_base=3, LMP_scale=1.9.

Computed values:
```
depth 0:  3 + 0.5^1.9 =  ~3
depth 1:  3 + 1.5^1.9 =  ~5
depth 2:  3 + 2.5^1.9 =  ~8
depth 3:  3 + 3.5^1.9 = ~14
depth 4:  3 + 4.5^1.9 = ~21
depth 5:  3 + 5.5^1.9 = ~30
depth 6:  3 + 6.5^1.9 = ~41
depth 7:  3 + 7.5^1.9 = ~54
depth 8:  3 + 8.5^1.9 = ~69
```

Note: The thresholds grow quickly, so LMP rarely triggers beyond depth ~8 in practice.

**Same exceptions as futility** (in_check, extend, first move, PawnPush).

### 2e. SEE Pruning (in Quiescence)

In quiescence search, captures where the attacker is more valuable than the victim are pruned if:
- `SEE(move) < 0`
- AND the capture doesn't remove the last opponent piece (special exception to allow promotions to win)

Checking moves in QS are also SEE-filtered: only `SEE(move) >= 0` checks are searched.

### 2f. No Razoring, No ProbCut, No Reverse Futility

Crafty does NOT implement:
- **Razoring** (no static eval drop-into-QS)
- **Reverse futility pruning / static null move**
- **ProbCut** (no shallow search to verify beta cutoff)
- **History pruning** (no pruning based on history scores)
- **SEE pruning of quiet moves in main search** (only in QS)

This is notable -- many modern engines have these. Crafty relies more heavily on LMR + LMP + futility.

---

## 3. Extensions

### Check Extension

**Amount**: `check_depth` = **1 ply** (configurable via personality).

**Condition**: Move gives check AND the check is "safe" -- meaning SEE of the checking move (minus the captured piece value) is <= 0. In other words, the checker is not hanging.

```c
if (SEEO(tree, wtm, move) - pcval[Captured(move)] <= 0) {
    extend = check_depth;  // 1
}
```

Unsafe checks (where the checking piece is just being sacrificed pointlessly) are NOT extended.

**Key detail**: An extension disables LMR reduction for that move (`reduce` stays 0 when extend is set). In PV nodes, if extend is set, `reduce` is also decremented.

### No Other Extensions

Crafty 25.2 does NOT have:
- Singular extensions
- Recapture extensions
- Passed pawn extensions
- Castling extensions

Very minimalist -- only safe check extensions.

---

## 4. LMR (Late Move Reductions)

### Formula

```c
LMR[d][m] = Max(Min(log(d * LMR_db) * log(m * LMR_mb) / LMR_s, LMR_max), LMR_min);
LMR[d][m] = Min(LMR[d][m], Max(d - 1 - LMR_rdepth, 0));
```

**Parameters**:
- `LMR_db` = 1.8 (depth bias -- depth is 1.8x as important)
- `LMR_mb` = 1.0 (move count bias)
- `LMR_s`  = 2.0 (scale factor -- smaller = more aggressive reductions)
- `LMR_min` = 1 (minimum reduction: 1 ply)
- `LMR_max` = 15 (maximum reduction: 15 plies)
- `LMR_rdepth` = 1 (minimum remaining depth after reduction: at least 1 full ply)

**Table dimensions**: LMR[32][64] (uint8_t) -- depth 0..31, move order 0..63.

**Effective formula**: `log(1.8*d) * log(1.0*m) / 2.0`, clamped to [1, 15], then further clamped so at least `LMR_rdepth` (1) plies remain after reduction.

The table starts at depth 3 and move 1 (entries for d<3 or m<1 are 0).

### Reduction Adjustments

Very few adjustments compared to modern engines:

1. **PV node or extended move**: `reduce--` (reduce by 1 less)
   ```c
   if (reduce && (pv_node || extend))
       reduce--;
   ```

2. **No history-based reduction adjustment**
3. **No improving/not-improving adjustment**
4. **No capture/quiet distinction in reductions** (LMR only applies to non-pruned moves that passed futility/LMP)

### LMR Re-search

If a reduced search returns `value > alpha`, re-search at full depth with the zero-window:
```c
if (value > alpha && reduce) {
    value = -Search(tree, ply+1, depth-1, ...);  // full depth, zero window
}
```

Then standard PVS re-search if `value > alpha && value < beta && t_beta < beta`.

### Root LMR

Crafty does LMR at the root. Root moves that were "best" in the last 3 iterations (bm_age > 0) are flagged with status bit 4, which sets `phase = DO_NOT_REDUCE` -- these moves are NOT reduced or searched in parallel. All other root moves use `phase = REMAINING` and are subject to normal LMR.

---

## 5. Move Ordering

### Stages (FSM phases)

1. **HASH** -- TT best move (also serves as PV move)
2. **GENERATE_CAPTURES** -- Generate captures (or all evasions if in check)
3. **CAPTURES** -- MVV/LVA ordered captures. Losing captures (attacker > victim AND SEE < 0) are deferred to REMAINING phase
4. **KILLER1** -- Killer move 1 (current ply)
5. **KILLER2** -- Killer move 2 (current ply)
6. **KILLER3** -- Killer move 1 from **ply-2** (2 plies back)
7. **KILLER4** -- Killer move 2 from ply-2
8. **COUNTER_MOVE1** -- Counter-move 1 (refutation of previous ply's move)
9. **COUNTER_MOVE2** -- Counter-move 2
10. **MOVE_PAIR1** -- Move pair 1 (follow-up to the move 2 plies back)
11. **MOVE_PAIR2** -- Move pair 2
12. **GENERATE_QUIET** -- Generate non-captures
13. **HISTORY** -- Top N moves by history score (selection sort, only for depth >= 6)
14. **REMAINING** -- Everything else (includes deferred losing captures), sequential order

**When in check**: GenerateCheckEvasions() produces all legal evasions at once. Captures are sorted by MVV/LVA, then everything goes directly to REMAINING (no killer/counter/history phases).

### Capture Scoring

MVV/LVA via pre-computed `MVV_LVA[victim][attacker]` table. Captures where attacker <= victim (or SEE >= 0) are searched immediately. Others are deferred to the REMAINING phase.

### Quiet Move Scoring

History table only. No piece-to-square tables, no continuation history, no capture history.

The HISTORY phase does a selection sort over the move list, picking the highest-history move first. This only runs for `depth >= 6`. It picks the top `remaining/2` moves via selection sort, then falls through to REMAINING for the rest.

### Killer Moves

- 2 killers per ply (standard)
- Also tries 2 killers from ply-2 (grandparent killers)
- Total: up to 4 killer moves tried before quiet generation
- Killers are only updated for non-capture, non-promotion moves

### Counter Moves

Indexed by `previous_move & 4095` (12-bit: from/to squares only, no piece info).
- 2 counter moves stored per previous-move index
- Tried after killers, before quiet generation

### Move Pairs (Follow-up Moves)

**Novel/unusual**: Indexed by `move_two_plies_back & 4095`.
- Stores 2 moves that worked well after a specific grandparent move
- Similar to continuation history but stores actual best moves instead of scores
- Tried after counter moves, before quiet generation

This is essentially a simplified form of continuation history that stores concrete moves rather than a score table.

---

## 6. History Tables

### Dimensions

`int history[1024]` -- indexed by `HistoryIndex(side, move)`:
```c
#define HistoryIndex(side, m) ((side << 9) + (Piece(m) << 6) + To(m))
```

That is: **side (1 bit) * piece (3 bits, 0-6) * to_square (6 bits)** = 1024 entries.

This is very simple compared to modern engines. No from-square, no piece-to indexing, no capture history, no continuation history tables.

### Update Mechanism

Saturating counters in range [0, 2047]:
- On fail-high (depth > 5): `history[good_move] += (2048 - history[good_move]) >> 5`
  - This adds ~1/32 of the remaining headroom (approaches 2048 asymptotically)
- On fail-high (depth > 5), for each other searched quiet move: `history[bad_move] -= history[bad_move] >> 5`
  - Decays by ~1/32 of current value

**Key detail**: The update does NOT depend on depth. This is intentional and documented -- Hyatt found depth-dependent updates biased too heavily toward root moves.

**No continuation history**: Crafty has no [piece][to][piece][to] or similar multi-dimensional tables.

**No capture history**: Captures are ordered purely by MVV/LVA + SEE.

---

## 7. Time Management

### Time Allocation (TimeSet)

**Sudden death (increment)**:
```c
time_limit = (remaining - safety_margin) / (ponder ? 20 : 26) + increment;
absolute_time_limit = Min(5 * time_limit, remaining / 2);
```

So with pondering ON, it divides by 20. With pondering OFF, divides by 26 (more conservative since no time saved from ponder hits).

**Classical time control**:
```
simple_average = (total_time - safety_margin) / total_moves
surplus = Max(remaining - safety_margin - simple_average * moves_remaining, 0)
average = (remaining - safety_margin + moves_remaining * increment) / moves_remaining
```

If surplus < safety_margin: `time_limit = Min(average, simple_average)`
If surplus >= safety_margin: `time_limit = Min(average, 2 * simple_average)`

**Absolute time limit**:
```c
absolute_time_limit = time_limit + surplus/2 + (remaining - safety) / 4;
// capped at: Min(5 * time_limit, remaining / 2)
```

**Early out-of-book bonus**: If increment > 200 and < 2 moves out of book, multiply time_limit by 1.2.

**"timebook" feature**: Extra time for first N moves out of book, configurable.

### Difficulty-Based Time Scaling

The `difficulty` variable scales the effective time limit: `effective = difficulty * time_limit / 100`.

**Range**: [60%, 200%] of base time_limit.

**Algorithm**:
- Starts at 100%
- **End of iteration, no best-move change**: `difficulty *= 0.9` (easier = less time)
- **Fail high on first move**: No change
- **Fail low on first move**: `difficulty = Max(100, difficulty)` (reset if below 100)
- **Fail high on later move** (mind change):
  - If difficulty < 100: set to 120
  - If difficulty >= 100: add 20
  - Each additional mind change: +20 more
- Clamped to [60, 200]

### TimeCheck Logic

- `busy=0` (called from Iterate between iterations): Stop if `time_used >= difficulty * time_limit / 100`
- `busy=1` (called from Search mid-iteration):
  - If under time limit: continue
  - If only 1 move searched and it didn't fail low: stop
  - If time_used + 300 > remaining: stop (emergency)
  - Otherwise: continue (try to finish iteration)

### One-Move Optimization

If only 1 legal root move and iteration > 10 (not pondering): abort immediately.

---

## 8. IID (Internal Iterative Deepening)

**Condition**: No hash move AND depth >= 6 AND do_null AND ply > 1 AND PV node.

**Reduction**: depth - 2 (search at 2 plies less).

**Purpose**: Get a good first move for PV nodes when TT misses.

---

## 9. Quiescence Search

### Structure

- `Quiesce()`: Captures + optional checking moves
- `QuiesceEvasions()`: Full evasion search when in check

### Stand-pat

Standard: if static eval >= beta, return. If > alpha, update alpha.

### Capture Ordering

MVV/LVA with insertion sort. No TT move probe in QS.

### Capture Pruning

Captures where attacker value > victim value are pruned if:
1. SEE < 0, AND
2. The capture doesn't remove the last opponent piece

### Check Generation in QS

Only at the first ply of quiescence (when `checks=1`, passed from Search). Checks must have SEE >= 0 to be searched. Captures that give check call into QuiesceEvasions which is a full-width evasion search.

### Repetition Detection in QS

Only at first ply of QS (when checks=1). Deeper QS plies skip repetition detection.

---

## 10. Parallel Search (Lazy SMP variant)

Crafty uses a different parallelization than modern Lazy SMP:
- **Tree splitting**: Explicit split points where idle threads join ongoing searches
- **Shared move list at split point**: Threads take moves from a shared list with locking
- **Thread stopping**: When one thread gets a cutoff, it signals siblings to stop
- **Root move aging**: Moves that were best in last 3 iterations are NOT searched in parallel (forced sequential to get good alpha bound fast)
- **Block-based memory**: Each thread has up to 64 TREE blocks for different split points

---

## 11. Hash Table

### Null Move Avoidance

The TT stores enough information to detect when a null move search would be futile:

```c
// In HashProbe:
if (depth - null_depth - depth/null_divisor - 1 <= draft && val < beta)
    return AVOID_NULL_MOVE;
```

If a previous search at sufficient depth scored below beta, we know null move won't help.

### Aspiration Re-search

Standard exponential widening: delta starts at 16, doubles each fail. Once delta > 1000 (10 * PAWN_VALUE), goes to 99999 (effectively full window).

---

## 12. Things Crafty Does NOT Have (vs. modern engines)

- **No NNUE or neural eval** (classical eval only)
- **No reverse futility pruning / static null move pruning**
- **No razoring**
- **No ProbCut**
- **No singular extensions**
- **No continuation history** (only a simple 1024-entry history table)
- **No capture history** (MVV/LVA + SEE only)
- **No improving flag** (no improving/not-improving adjustments to margins)
- **No history-based LMR adjustments**
- **No countermove history** (has counter-moves but not a history table for them)
- **No correction history**
- **No SEE pruning for quiet moves in main search** (only captures in QS)
- **No multi-cut**
- **No aspiration window narrowing based on stability**

---

## 13. Notable / Unusual Features

### Move Pairs (Continuation-like)

The `move_pair[4096]` table stores 2 best follow-up moves for each move played 2 plies ago. This is like a very simplified continuation history that stores concrete moves instead of scores. Indexed by `curmv[ply-2] & 4095` (from/to only).

### 4 Killers (2 current + 2 grandparent)

Most engines use 2 killers. Crafty tries 4: the 2 current-ply killers plus 2 from ply-2.

### History as Saturating Counter

The history update is unusual: it uses a software saturating counter approach where the increment is proportional to the distance from the maximum (2048). This naturally prevents overflow and gives diminishing returns as a move's score approaches the cap. NOT depth-weighted.

### Null Move Depth Guard Variation

The n_depth variable changes based on position characteristics:
- Normal positions: null move allowed at depth > 1
- Endgame with few pieces (<=9), few root moves (<=17), shallow (<=3): depth > 3
This is a targeted zugzwang avoidance heuristic.

### Root Move "Best Move Age" System

Each root move tracks `bm_age` which is set to 4 when the move becomes best, decremented each iteration. Moves with bm_age > 0 get special treatment:
- NOT searched in parallel (all threads focus on it)
- NOT reduced by LMR
- This ensures recently-best moves get full-depth, full-power searches

### Draw Score Asymmetry

```c
draw_score[black] = (wtm) ? -abs_draw_score : abs_draw_score;
draw_score[white] = (wtm) ? abs_draw_score : -abs_draw_score;
```
The draw score is biased: the side to move slightly dislikes draws, the opponent slightly likes them. This encourages the engine to avoid draws when ahead and seek them when behind (contempt factor).

### EGTB Score Fudging

Tablebase draw scores are fudged based on material:
- Blessed loss: -3
- Pure draw: 0
- Cursed win: +3
- Then +1 if ahead in material, -1 if behind
Creates a 9-level ordering that prefers better material even in drawn positions.
