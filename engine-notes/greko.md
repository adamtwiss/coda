# GreKo 2021.12 - Engine Crib Notes

Author: Vladimir Medvedev. Released 31-Dec-2021.
Source: `~/chess/engines/greko-2021.12/src/`

## Search

### Overall Structure

- Negamax alpha-beta with iterative deepening and aspiration windows.
- Aspiration window: `WINDOW_ROOT = 100` (i.e., +/-50 around previous score). On fail, widens to full window and re-searches the same depth (`--s_iter`).
- PVS: first move searched with full window, rest with zero window. Re-search on fail high with full window.
- Node types: `NODE_PV` (PV node) and `NODE_NON_PV`. Most pruning features are **disabled at PV nodes** (see arrays below).

### Feature Enable/Disable by Node Type

Each feature has a `[NODE_PV][NODE_NON_PV]` enable array:

| Feature                  | PV | Non-PV |
|--------------------------|:--:|:------:|
| Futility                 | 0  |   1    |
| Null move                | 0  |   1    |
| LMR                      | 0  |   1    |
| Pawn 7th extensions      | 1  |   1    |
| Recapture extensions     | 1  |   1    |
| Single reply extensions  | 1  |   1    |
| IID                      | 1  |   1    |
| SEE pruning (QS)         | 1  |   1    |
| Mate pruning             | 1  |   1    |
| Hash exact eval cutoff   | 1  |   1    |
| Hash bound pruning       | 1  |   1    |
| QS checks                | 1  |   1    |

**Key observation**: Futility, NMP, and LMR are all disabled at PV nodes. This is conservative by modern standards (most engines do futility/LMR at PV nodes with larger margins).

### Null Move Pruning (NMP)

- **Conditions**: Not in check, last move was not null, side has material (`MatIndex > 0`), `depth >= 2`.
- **Disabled at PV nodes**.
- **Reduction formula**:
  ```
  R = 3 + (depth - 2) / 6.0 + max(0, staticScore - beta) / 120.0
  ```
  - Base R = 3.
  - Depth scaling: `(depth - 2) / 6`. At depth 8, adds 1. At depth 14, adds 2.
  - Eval scaling: `max(0, staticEval - beta) / 120`. If 120 above beta, adds 1 extra ply of reduction.
- No verification search. No NMP in zugzwang-prone endgames (only checks `MatIndex > 0`, which means has at least a minor piece).
- Searches with window `(-beta, -score)` where `score` starts as `alpha`.

### Futility Pruning (Combined Alpha and Beta)

- **Conditions**: Not in check, last move was not null, depth 1-3.
- **Disabled at PV nodes**.
- **Alpha-side (razoring-like)**: If `staticScore <= alpha - margin[depth]`, drop to QS:
  ```
  FUTILITY_MARGIN_ALPHA = { 0, 50, 350, 550 }
  ```
  - Depth 1: margin 50
  - Depth 2: margin 350
  - Depth 3: margin 550
- **Beta-side (reverse futility / static null move)**: If `staticScore >= beta + margin[depth]`, return beta:
  ```
  FUTILITY_MARGIN_BETA = { 0, 50, 350, 550 }
  ```
  - Same margins as alpha side.
- Note: alpha and beta margins are identical. At depth 1 the margin is only 50cp, which is quite tight.

### Mate Distance Pruning

- At both PV and non-PV: if `alpha >= CHECKMATE_SCORE - ply`, return alpha.
- Only prunes upper bound (no lower bound check for being mated).

### Internal Iterative Deepening (IID)

- **Conditions**: No hash move available, `depth > 4`.
- **Enabled at both PV and non-PV nodes**.
- Searches at `depth - 4` (reduction of 4 plies).
- Uses the PV move from the reduced search as the hash move substitute.

### Extensions

All extensions are guarded by: `ply + depth <= 2 * currentIteration` (prevents search explosion).

1. **Check extension**: Always extends +1 when in check (unconditional, before the move loop).
2. **Pawn 7th rank extension**: +1 when a pawn moves to the 7th rank. Both PV and non-PV.
3. **Recapture extension**: +1 when moving to the same square as the last move, and both moves are captures. Both PV and non-PV.
4. **Single reply extension**: +1 when only one legal move exists. Both PV and non-PV. Counts legal moves before the search loop.

Note: The extension `if/else if` chain means only ONE extension (besides check) can fire per node (pawn 7th > recapture > single reply priority).

### Hash Table

- Single-slot (no buckets). 16 bytes per entry.
- Lockless verification via upper 32 bits of Zobrist (`hashLock`).
- Always-replace scheme (no depth-preferred or age-preferred logic beyond the age field).
- Hash cutoffs: at non-root (`ply > 0`), if `hashDepth >= depth`:
  - Exact: return score directly (both PV and non-PV).
  - Alpha bound: return alpha if hashScore <= alpha.
  - Beta bound: return beta if hashScore >= beta.
- Mate score adjustment: standard `score +/- ply` when storing/retrieving.

### Repetition Detection

- Returns `DRAW_SCORE` (0) on 2-fold repetition (`pos.Repetitions() >= 2`), not 3-fold. This is the in-search standard.

### No Pruning Techniques Present

The following are **absent** compared to modern engines:
- **Late Move Pruning (LMP)**: Not implemented. All moves are searched.
- **SEE pruning in main search**: Only in QS. No SEE-based move pruning in the main search.
- **ProbCut**: Not implemented.
- **Razoring** (as a separate feature): The alpha-side futility at depth 1-3 serves as a razoring substitute.
- **Singular extensions**: Not implemented.
- **History pruning**: Not implemented.
- **Improving flag**: Not tracked. No improving/non-improving distinction in any pruning.
- **Counter-move history / continuation history**: Not implemented.
- **Capture history**: Not implemented.
- **Correction history**: Not implemented.

---

## Move Ordering

### Sort Scores (Priority Order)

```
SORT_HASH        = 6,000,000
SORT_CAPTURE     = 5,000,000 + MVV-LVA offset
SORT_MATE_KILLER = 4,000,000
SORT_KILLER      = 3,000,000
SORT_REFUTATION  = 2,000,000
SORT_MAX_HISTORY = 1,000,000
SORT_OTHER       = 0 + SuccessRate(0-100)
```

### Stages

No staged move generation. All moves are generated at once (`GenAllMoves` or `GenMovesInCheck`), scored, and then selection-sorted on the fly (`GetNextBest` does a linear scan to find the highest-scored remaining move).

### Capture Scoring (MVV-LVA)

```
score = SORT_CAPTURE + 6 * (captured/2 + promotion/2) - piece/2
```
Where piece types / 2 gives: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6.

So MVV-LVA: value the captured piece highly (6x multiplier), subtract attacker value.

### Killer Moves

- **One killer per ply** (`m_killers[ply]`).
- **One mate killer per ply** (`m_mateKillers[ply]`). Updated when a quiet move produces a score near checkmate (`score > CHECKMATE_SCORE - 50`).
- Mate killer has higher priority than regular killer (4M vs 3M).
- Only one killer slot each (not two killers like most engines).

### Refutation Table (Counter-Move)

- `m_refutations[ply][lastMove.To()][lastMove.Piece()]` -- indexed by ply, last move's target square (64), and last move's piece (14 types).
- Dimensions: `[MAX_PLY+1][64][14]` of Move.
- Updated on beta cutoffs for quiet moves.
- Priority: below killer (2M), above history.

### History Table

- **Dimensions**: `m_histTry[64][14]` and `m_histSuccess[64][14]` (square x piece type).
- **Success rate**: `100 * histSuccess[to][piece] / histTry[to][piece]`.
- Quiet moves are sorted by: `SORT_OTHER + SuccessRate(mv)` where SuccessRate is 0-100.
- **Delta**: `histTry` incremented by `deltaHist = 1` for every move tried. `histSuccess` incremented by 1 on beta cutoff.
- **No aging/decay**: History is cleared at the start of each search (`NewSearch` calls `ClearHistory`).
- This is a very simple ratio-based history, not the modern gravity-based history (bonus - depth^2 penalty).

### History-Based LMR Guard

The history success rate is also used in LMR: moves with `SuccessRate > 50%` are exempt from LMR (see LMR section).

---

## LMR (Late Move Reductions)

- **Conditions**:
  - `depth >= 4` (LMR_MIN_DEPTH)
  - Not null move parent
  - Not in check (either side)
  - Not a capture
  - Not a promotion
  - Move's `SuccessRate <= 50` (LMR_MAX_SUCCESS_RATE)
  - **Disabled at PV nodes**
- **Quiet move counter**: Incremented for each move passing the above conditions.
- **Min move**: Reductions start at `quietMoves >= 3` (LMR_MIN_MOVE).
- **Reduction formula**:
  ```
  reduction = 1 + (depth - 4) / 10.0 + (quietMoves - 3) / 10.0
  ```
  Truncated to int.

  Examples:
  - depth=4, quietMove=3: `1 + 0 + 0 = 1`
  - depth=8, quietMove=3: `1 + 0.4 + 0 = 1`
  - depth=14, quietMove=13: `1 + 1.0 + 1.0 = 3`
  - depth=20, quietMove=23: `1 + 1.6 + 2.0 = 4`

- **Re-search**: On fail high with reduction, re-search at full depth with zero window. Then if still > alpha and < beta, full window re-search.

### LMR Adjustments (NOT present)

- No reduction for improving/non-improving.
- No reduction for PV nodes (LMR is entirely disabled there).
- No history-based reduction adjustment (only a binary gate: success rate > 50% skips LMR entirely).
- No reduction for check-giving moves (already excluded by "not in check after move").
- No reduction for killers/counter-moves.

---

## Quiescence Search

### Structure

- Separate function `AlphaBetaQ(alpha, beta, ply, qply)`.
- `qply` tracks depth within QS (starts at 0).
- Stand-pat: if not in check, use `staticScore` as lower bound.
- Stand-pat can return beta immediately (fail-high).

### Move Generation

- In check: `GenMovesInCheck` (all evasions).
- Not in check: `GenCapturesAndPromotions` + optional `AddSimpleChecks`.

### Delta Pruning in QS

`GenCapturesAndPromotions` receives a `delta` parameter = `score - staticScore` (how far below alpha).
```
const EVAL QMOVES_DELTA_1 = 500;
```
If `delta > 500`, pawn captures are excluded from generated moves (targets exclude opponent pawns). This is a coarse delta pruning at the movegen level.

### SEE Pruning in QS

- At all qply depths (`SEE_PRUNING_MIN_QPLY = 0`): skip moves with `SEE < 0`.
- Both PV and non-PV.

### QS Checks

- Simple checks (non-capture quiet moves that give check) are generated at `qply < 1` (i.e., only at the first QS ply).
- Enabled at both PV and non-PV.
- `AddSimpleChecks` generates knight, bishop, rook, and queen moves to squares that attack the opponent king.

### Lazy Eval in QS

- `Evaluate(pos, alpha, beta)` has lazy evaluation: `FastEval` (material + PST only). If the fast eval is below `alpha - 200` or above `beta + 200`, returns the bound without full eval.
- `LAZY_MARGIN = 200`.

---

## Time Management

### Extremely Simple

```cpp
void CalculateTimeLimits()
{
    maxTimeSoft = restTime / 40;
    maxTimeHard = restTime / 2;
}
```

- **Soft limit**: `restTime / 40` (2.5% of remaining time).
- **Hard limit**: `restTime / 2` (50% of remaining time).
- Soft limit checked after each completed iteration (if score is within aspiration window).
- Hard limit checked every 8192 nodes.

### No Sophistication

- **No increment handling in time calculation**: `g_increment` is parsed from UCI but never added to the time budget in `CalculateTimeLimits()`. This is likely a significant weakness.
- **No instability detection**: No extension for score changes, fail-lows, or best move changes.
- **No move stability**: No reduced time for stable best move.
- **No pondering time bonus**.
- **No movestogo handling**: `g_restMoves` is parsed but not used in time calculation.
- Single reply terminates search immediately.
- Forced mate terminates search immediately.

---

## Eval

### Tapered Evaluation

- Uses `Pair(mid, end)` with game phase as a pair of weights.
- `DotProduct(stage, score)` = `stage.mid * score.mid + stage.end * score.end`.
- Phase computed from piece counts on the board.

### Features

- PSQ tables (compressed polynomial or tabular, selectable via compile defines: PSQ_5/12/16/64).
- Pawn structure: passed, doubled, isolated, backwards pawns with PSQ bonuses.
- Mobility: Knight (0-8), Bishop (0-13), Rook (0-14), Queen (0-27).
- King distance: per piece type to opponent king.
- King safety: pawn shield (0-9), pawn storm (0-9), king exposure (0-27 queen-mobility-on-king).
- Rook on open/semi-open file, rook on 7th rank.
- Piece pairs (4x4 matrix of N/B/R/Q combinations).
- Attack on king zone (binary per piece).
- Attack on stronger pieces (binary per piece).
- Tempo bonus.

### Pawn Hash Table

- 2^16 = 65536 entries. No replacement policy (direct-mapped).
- Stores: passed/doubled/isolated/backwards bitboards, pawn ranks per file, pawn attacks, safe squares.

### 50-Move Rule Scaling

Score is scaled by `(100 - fiftyMoveCount) / 100`. At 50 half-moves, score is halved.

### Insufficient Material

If one side has no pawns and `MatIndex < 5` (less than a rook equivalent), their winning score is clamped to 0.

### Learned Weights

All eval parameters are stored in `weights.txt` and learned via SGD/coordinate descent from self-play PGNs. This is an early form of Texel tuning.

---

## SMP (Lazy SMP)

- Up to 16 threads (`MAX_NUM_THREADS`).
- Shared hash table (single-slot, no atomics -- just lock-based on the hash verification).
- Helper threads start from different depths (`startDepth = 1 + threadId`) to get natural depth diversity.
- Each thread has its own position, killers, refutations, history, move lists.
- Main thread controls search depth and time. Helpers run independently and are stopped when main finishes.

---

## Notable/Unusual Features

1. **History as success rate**: Uses `successCount / tryCount * 100` rather than modern gravity-based history. This caps at 100 and has no depth weighting. The 50% threshold for LMR exemption is an unusual binary gate.

2. **Mate killers**: Separate killer table for moves that lead to checkmate scores. Higher priority than regular killers. This is uncommon in modern engines.

3. **No LMR/futility at PV nodes**: Completely disabled. Most modern engines do reduced-margin LMR and futility at PV nodes. This likely costs search efficiency.

4. **Alpha-side futility drops to QS**: Rather than returning a static score, the alpha-futility (razoring) falls through to quiescence search for a more accurate answer.

5. **Delta pruning at movegen level**: QS excludes pawn captures when far below alpha (delta > 500), which is unusual -- most engines do delta pruning per-move, not by filtering the move list.

6. **Single extension type per node**: The if/else chain for extensions means pawn-7th, recapture, and single-reply are mutually exclusive (plus the depth budget guard).

7. **No increment in time management**: The UCI increment is parsed but never used. This is almost certainly a bug or oversight that costs significant Elo in increment time controls.

8. **Polynomial PSQ tables**: Optional compile-time choice between 5-coefficient polynomial (X, Y, X^2, Y^2, XY), 12-param (col+row decomposition with file symmetry), 16-param (col+row without symmetry), or full 64-param PSQ tables. The polynomial representation is a compact way to regularize PST learning.

9. **Very simple hash table**: Single slot per index, no bucket structure, no depth/age replacement policy. Just overwrites.

10. **Strength limiting**: Adjustable strength 0-100 via NPS limiting. Exponential curve from 0.1 kNPS to 1000 kNPS.

---

## Comparison with GoChess: What GreKo Has That We Might Not

| Feature | GreKo | GoChess |
|---------|-------|---------|
| Mate killers (separate from regular killers) | Yes | No |
| Alpha-futility drops to QS instead of static return | Yes | No (static return) |
| QS delta pruning at movegen level (exclude pawn captures) | Yes | No |
| History success rate as LMR gate | Yes (binary 50% threshold) | No (continuous history adjustment) |
| IID with depth-4 reduction | Yes | No (uses IIR: depth-1) |
| Pawn 7th rank extensions | Yes | No |
| Single reply extensions | Yes | No |

### Ideas Worth Testing from GreKo

- **Mate killers**: Probably not worth it given modern continuation history captures most of this signal.
- **Alpha-futility to QS**: We already have razoring that drops to QS. Margin differences worth checking: GreKo uses 50/350/550 for depths 1/2/3 vs our 400+d*100.
- **QS delta pruning at movegen**: Could save time by not generating obviously futile captures. But our QS already has SEE pruning which handles this.
- **Single reply extension**: We removed this. Could re-test but unlikely to help with NNUE.
- **Pawn 7th rank extension**: We removed this. Could re-test.

### What GoChess Has That GreKo Lacks

GreKo is missing most modern search features: LMP, SEE pruning in main search, ProbCut, singular extensions, history pruning, continuation history, capture history, correction history, improving detection, sophisticated LMR adjustments, proper time management with increment handling, and NNUE.
