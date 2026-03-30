# Berserk v13 Chess Engine - Detailed Review

Source: `~/chess/engines/berserk-13/`
Author: Jay Honnold
Rating: CCRL #3, +289 Elo above GoChess in our RR (397 total gap)
Language: C with pthreads
NNUE: (768x16 -> 1024)x2 -> 16 -> 32 -> 1 (4-layer, int8 L1 with sparse matmul, float L2/L3)

---

## 1. NNUE Architecture

### Network Topology: (768x16 -> 1024)x2 -> 16 -> 32 -> 1

**Feature Transformer (Input Layer):**
- 16 king buckets x 12 piece types x 64 squares = 12,288 features per perspective
- Hidden size: 1024 neurons (int16 accumulators)
- Perspective: dual accumulator (stm/xstm views), concatenated = 2048 inputs to L1
- King buckets use a custom 16-bucket mapping with horizontal mirroring (flip when king on files e-h):
  ```
  Rank 1: 3  2  1  0  0  1  2  3   (most granular near home)
  Rank 2: 7  6  5  4  4  5  6  7
  Rank 3: 11 10 9  8  8  9  10 11
  Rank 4: 11 10 9  8  8  9  10 11  (same as rank 3)
  Rank 5: 13 13 12 12 12 12 13 13
  Rank 6: 13 13 12 12 12 12 13 13  (same as rank 5)
  Rank 7: 15 15 14 14 14 14 15 15
  Rank 8: 15 15 14 14 14 14 15 15  (same as rank 7)
  ```
- This gives 16 unique buckets with fine-grained king-side vs queen-side distinction on ranks 1-4, coarser on ranks 5-8. More king buckets than our 16 (same count, but different mapping).

**L1 (Hidden 1): 2048 -> 16 (int8 -> int32, sparse matmul)**
- Input: ReLU-clipped int8 (accumulator >> 5, clamped to [0, 127])
- Weights: int8, scrambled for SPARSE_CHUNK_SIZE=4 sparse multiplication
- **NNZ-sparse matmul**: Finds non-zero 4-byte chunks in the int8 input, only multiplies non-zero entries. Uses lookup table (LOOKUP_INDICES) for fast NNZ index extraction.
- Output: int32, converted to float with ReLU
- SIMD: AVX-512, AVX2, SSSE3 paths

**L2 (Hidden 2): 16 -> 32 (float)**
- Weights: float, full matmul with FMA
- Activation: ReLU
- SIMD: AVX-512 with custom hadd, AVX2, SSE3 paths

**L3 (Output): 32 -> 1 (float)**
- Weights: float, dot product + bias
- Output: divided by 32

**Accumulator Management:**
- Stack-based (one Accumulator per ply, allocated MAX_SEARCH_PLY+1 per thread)
- Lazy incremental updates: `correct[color]` flag per perspective. On eval, walks back to last correct state and applies delta chain.
- **RefreshTable** (FinnyTable): Per-king-bucket cache (`AccumulatorKingState`) storing last-known accumulator values and piece bitboards. On king refresh, computes delta from cached state rather than full recompute.
- King move triggers refresh when bucket changes or king crosses the e-file mirror boundary (`(from & 4) != (to & 4)` or bucket differs).
- Specialized update functions: `ApplySubAdd`, `ApplySubSubAdd`, `ApplySubSubAddAdd` for 1-capture, 2-sub+1-add (capture), 2-sub+2-add (castling) patterns.

**Eval Post-Processing:**
- Phase scaling: `score = (128 + phase) * score / 128` — scales from 1.0x (endgame, phase=0) to 1.5x (opening, phase=64)
- Contempt: `phase * contempt[stm] / 64`
- Material draw detection (returns 0 early)
- FMR decay: `(200 - fmr) * eval / 200` applied after correction
- Pawn correction history: `rawEval + correction/2` (halved weight)

### Compared to GoChess NNUE
| Feature | Berserk | GoChess |
|---------|---------|---------|
| Architecture | (768x16->1024)x2->16->32->1 | v5: (768x16->N)x2->1x8 (shallow wide) |
| King buckets | 16 (custom mapping, mirrored) | 16 |
| Hidden size | 1024 | Dynamic (1024/1536/any) |
| Output buckets | None (1 output) | 8 (material-based) |
| L1 computation | int8 sparse matmul (NNZ skip) | int8 dense matmul |
| L2/L3 | float (16->32->1) | int8/int16 (32->32->8) |
| Activation | ReLU | CReLU/SCReLU (UPDATE: now supports both) |
| FinnyTable | Yes (per-bucket delta refresh) | Yes (UPDATE: merged) |
| Phase scaling | Yes (1.0x-1.5x by phase) | No |

**Key differences**: Berserk has a 4-layer architecture while our v5 is shallow wide. The sparse L1 matmul is a major optimization: skipping zero int8 activations saves significant compute when many neurons are inactive post-ReLU. **(UPDATE 2026-03-21: GoChess now has Finny tables, pairwise mul, SCReLU inference, and dynamic NNUE width.)** Float layers L2/L3 are very small (16 and 32 neurons) so the cost is negligible.

---

## 2. Search Architecture

### Iterative Deepening
- Depth 1 to MAX_SEARCH_PLY (201)
- Uses `setjmp`/`longjmp` for hot exit on time/stop
- Per-thread: each thread runs independent `Search()` with its own depth counter

### Aspiration Windows
- Enabled at depth >= 5
- Initial delta: **9** (much tighter than our 15)
- On fail-low: `beta = (alpha + beta) / 2`, `alpha = Max(alpha - delta, -CHECKMATE)`
- On fail-high: `beta = Min(beta + delta, CHECKMATE)`, **searchDepth--** (reduce depth by 1)
- Delta growth: `delta += 17 * delta / 64` (~1.27x per iteration)
- Full-width fallback when alpha < -2000 or beta > 2000

Compare to ours: delta=15. They use 9 (40% tighter). Fail-high depth reduction saves time. Fail-low beta contraction `(alpha+beta)/2` is more aggressive than some engines' `(alpha+3*beta)/4`.

### Draw Detection
- `IsDraw()` for repetition/50-move
- **Cuckoo hashing cycle detection**: `HasCycle(board, ply)` checked at top of Negamax for upcoming repetitions. Returns randomized draw score `2 - (nodes & 0x3)` (range [-1, 2]) — prevents repetition-seeking.
- Draw score: `2 - (thread->nodes & 0x3)` — small random perturbation around 0

### Mate Distance Pruning
- `alpha = Max(alpha, -CHECKMATE + ply)`, `beta = Min(beta, CHECKMATE - ply - 1)` — prunes provably shorter mates
- We do NOT have this. Simple and free.

---

## 3. Pruning Techniques (Exact Parameters)

### Reverse Futility Pruning (RFP)
- Conditions: `!isPV && !inCheck && !ss->skip && eval < TB_WIN_BOUND`
- Depth: `depth <= 9`
- Margin: `eval - 70*depth + 118*(improving && !opponentHasEasyCapture) >= beta`
  - Not improving: 70*depth (depth 1: 70, depth 5: 350, depth 9: 630)
  - Improving without easy captures: 70*depth - 118
- **Guard**: `!hashMove || GetHistory(ss, thread, hashMove) > 11800` — only prune if no hash move, or hash move has strong history
- **Guard**: `!opponentHasEasyCapture` — don't prune when opponent can easily capture hanging pieces
- **Return**: `(eval + beta) / 2` — blended score, not raw eval
- Compare to ours: 85*d (improving) / 60*d (not). They use 70*d with opponent-easy-capture guard + history guard on hash move. The RFP return value blending is novel.

### Razoring
- Conditions: `!isPV && !inCheck`
- Depth: `depth <= 5`
- Margin: `eval + 214*depth <= alpha` (depth 1: 214, depth 3: 642, depth 5: 1070)
- Drops to qsearch, returns score if <= alpha
- Compare to ours: 400+100*d. Their margin is steeper (214*d vs 100*d+400).

### Null Move Pruning (NMP)
- Conditions: `depth >= 4 && (ss-1)->move != NULL_MOVE && !ss->skip && !opponentHasEasyCapture && eval >= beta && HasNonPawn`
- Reduction: `R = 4 + 385*depth/1024 + Min(10*(eval-beta)/1024, 4)`
  - Base R=4 (vs our 3), depth scaling ~0.376*depth (vs our d/3=0.333*d), eval scaling ~0.01*(eval-beta) capped at 4
  - At depth 10, eval=beta+200: R = 4 + 3.75 + 1.95 = ~9 (vs our 3+3.33+1 = ~7)
- **Guard**: `!opponentHasEasyCapture` — unique, prevents NMP when opponent has hanging pieces
- **Verification**: At depth >= 14 (if `!thread->nmpMinPly`), runs full verify search with NMP restricted for `3*(depth-R)/4` plies. Uses `nmpMinPly`/`npmColor` to prevent recursive NMP in verification search.
- Returns `beta` if score >= TB_WIN_BOUND, otherwise returns raw score
- Compare to ours: Higher base R (4 vs 3), opponent-easy-capture guard, deeper verification (our threshold is 12).

### Internal Iterative Reduction (IIR)
- Conditions: `!ss->skip && !inCheck && (isPV || cutnode) && depth >= 4 && !hashMove`
- Reduces depth by 1
- Compare to ours: We apply IIR at any node type at depth >= 4. They restrict to PV or cutnode.

### ProbCut
- Margin: `beta + 172`
- Depth: `depth >= 6`
- Guard: `!(ttHit && ttDepth >= depth-3 && ttScore < probBeta)` — skip if TT already says no cutoff
- **Two-stage**: QSearch first (`-Quiesce(-probBeta, -probBeta+1)`), then full negamax at `depth-4` only if QS passes
- SEE threshold for capture selection: `probBeta > eval` (dynamic)
- Compare to ours: We use margin 170, gate 85, depth >= 5. They have the two-stage QS pre-filter we identified as a testing candidate.

### Late Move Pruning (LMP)
- Table-based: `LMP[improving][depth]`
  - Not improving: `1.305 + 0.3503 * d^2` (d1: 1.66, d5: 10.06, d10: 36.34)
  - Improving: `2.1885 + 0.9911 * d^2` (d1: 3.18, d5: 26.97, d10: 101.3)
- Applied as: `skipQuiets = 1` when `legalMoves >= LMP[improving][depth]`
- Only for `!isRoot` with `bestScore > -TB_WIN_BOUND`
- Compare to ours: We use 3+d^2 (improving 1.5x). Their improving multiplier is ~2.8x (much wider gap).

### Static Exchange Evaluation (SEE) Pruning
- Precomputed tables: `STATIC_PRUNE[quiet][depth]` and `STATIC_PRUNE[capture][depth]`
  - Quiet: `-15.2703 * d^2` (d1: -15, d3: -137, d5: -382, d10: -1527)
  - Capture: `-94.0617 * d` (d1: -94, d3: -282, d5: -470, d10: -941)
- Applied after LMP, using `lmrDepth` for quiets, raw `depth` for captures
- Quiet SEE threshold scales quadratically (more aggressive at depth)
- Compare to ours: We use -20*d^2 for SEE pruning. Their quiet SEE is `-15.27*d^2` (less aggressive), capture is `-94*d` (linear vs our quadratic).

### Futility Pruning
- Conditions: `!inCheck && lmrDepth < 10`
- Margin: `eval + 81 + 46*lmrDepth`
  - lmrDepth 1: 127, lmrDepth 5: 311, lmrDepth 9: 495
- Applied as `skipQuiets = 1` (prunes all remaining quiets)
- Compare to ours: We use 80+lmrDepth*80. Their formula is 81+46*lmrDepth (shallower slope).

### History Pruning
- Conditions: `!killerOrCounter && lmrDepth < 5`
- Threshold: `history < -2788*(depth-1)` (d2: -2788, d5: -11152, d10: -25092)
- Sets `skipQuiets = 1` AND skips the current move
- Compare to ours: We use -2000*depth. They use -2788*(d-1) — more aggressive, but exempt killers/counters.

### Alpha-Reduce (Depth Reduction After Alpha Raise)
- `depth -= (depth >= 2 && depth <= 11)` when `alpha < beta && score > -TB_WIN_BOUND`
- Reduces remaining search depth by 1 after raising alpha in the [2,11] depth range
- Compare to ours: We have this (MERGED, +13.0 Elo).

---

## 4. Extensions

### Singular Extensions
- Depth: `depth >= 6`
- Conditions: `move == hashMove && ttDepth >= depth-3 && (ttBound & BOUND_LOWER) && abs(ttScore) < TB_WIN_BOUND`
- Ply limiter: `ss->ply < thread->depth * 2` (prevents explosive tree growth)
- Singularity beta: `Max(ttScore - 5*depth/8, -CHECKMATE)`
  - At depth 10: ttScore - 6, At depth 20: ttScore - 12 (very tight)
- Singularity depth: `(depth-1)/2`
- **Triple extension (+3)**: if `score < sBeta - 48 && ss->de <= 6 && !IsCap(move)` and `!isPV`
- **Double extension (+2)**: if `score < sBeta - 14 && ss->de <= 6` and `!isPV`
- **Single extension (+1)**: otherwise when singular
- **Multi-cut**: if `sBeta >= beta`, return sBeta (all alternatives exceed beta)
- **Negative extension (-2+isPV)**: if `ttScore >= beta` (was previously a fail-high)
- **Cut-node reduction (-2)**: if `cutnode`
- **Alpha reduction (-1)**: if `ttScore <= alpha`
- Double-extension counter: `ss->de` incremented on double/triple ext, inherited from parent, capped at 6
- Compare to ours: Our SE is broken (-58 to -85 Elo). Berserk has the full toolkit: ply limiter, multi-cut, triple/double ext with counters, negative ext for cut/beta/alpha nodes. The tight margin (5*d/8 vs our 3*d) is critical.

### No Check Extension
- No explicit check extension in main search (handled implicitly via LMR `-1` for giving check)

---

## 5. Late Move Reductions (LMR)

### Table Initialization
- Single table (not split captures/quiets): `LMR[d][m] = log(d) * log(m) / 2.0385 + 0.2429`
- Divisor 2.0385, base 0.2429
- Compare to ours: We use split tables (cap C=1.80 / quiet C=1.50). They use a single table.

### Application
- `depth > 1 && legalMoves > 1 && !(isPV && IsCap(move))`
- PV captures are never reduced (always full window search)

### Reduction Adjustments
Pre-computed R from table, then:
- `R -= history / 8192` — history-based (continuous, not bucketed)
- `R += (IsCap(hashMove) || IsPromo(hashMove))` — +1 if TT move is noisy (MERGED in GoChess as +34.4 Elo)
- `R += 2` if `!ttPv` (non-PV TT node gets +2 reduction — very aggressive)
- `R += 1` if `!improving`
- `R -= 2` if `killerOrCounter`
- `R -= 1` if move gives check (`board->checkers` after MakeMove)
- `R += 1 + !IsCap(move)` if `cutnode` (+1 for captures, +2 for quiets at cut nodes)
- `R -= 1` if `ttDepth >= depth` (TT has deep entry for this position)
- Clamped to `[1, newDepth]`

### DoDeeper/DoShallower
- After LMR re-search beats alpha and R > 1:
  - `newDepth += (score > bestScore + 69)` — deeper if score significantly above best
  - `newDepth -= (score < bestScore + newDepth)` — shallower if score close to best
- Also: CH bonus/malus based on re-search result:
  - `bonus = score <= alpha ? -HistoryBonus(newDepth-1) : score >= beta ? HistoryBonus(newDepth-1) : 0`

### Compare to GoChess LMR
| Adjustment | Berserk | GoChess |
|-----------|---------|---------|
| Table | log*log/2.04+0.24 | split cap/quiet C=1.80/1.50 |
| History | -history/8192 | -history/5000 |
| !ttPv | +2 | N/A (no TT PV flag) |
| !improving | +1 | +1 |
| killerOrCounter | -2 | N/A |
| Gives check | -1 | N/A |
| cutnode | +1 to +2 | N/A |
| ttDepth deep | -1 | N/A |
| TT noisy | +1 | +1 (merged) |
| DoDeeper margin | +69 | N/A (tested -13.7 Elo) |

---

## 6. Move Ordering

### Staged Move Picker
Phases: HASH_MOVE -> GEN_NOISY -> PLAY_GOOD_NOISY -> KILLER_1 -> KILLER_2 -> COUNTER -> GEN_QUIET -> PLAY_QUIETS -> PLAY_BAD_NOISY

### Capture Scoring
- `GetCaptureHistory(move) / 16 + SEE_VALUE[captured_piece_type]`
- Good capture threshold: `SEE(move, -score/2)` — dynamic SEE threshold based on capture ordering score! Higher-scored captures get more lenient SEE filtering.
- Bad captures (fail SEE) deferred to PLAY_BAD_NOISY stage after quiets

### Quiet Scoring
- `HH(stm, move, threatened) * 2 + contHist[-1] * 2 + contHist[-2] * 2 + contHist[-4] * 1 + contHist[-6] * 1`
- Weights: butterfly x2, contHist plies 1,2 x2, contHist plies 4,6 x1 (deeper contHist at half weight)
- **Threat-aware escape/enter bonuses**: For non-pawn, non-king pieces:
  - Danger zones by piece type: pawns threaten all, minors add to pawn threats, rooks add to minor threats
  - `+16384` if piece is leaving a threatened square (escape bonus)
  - `-16384` if piece is entering a threatened square (enter penalty)
  - This is the threat-aware history idea with hardcoded bonuses instead of table-based

### History Tables — Threat-Indexed Butterfly
- `hh[2][2][2][64*64]` — stm, from_threatened, to_threatened, from*64+to
- 4x the size of standard butterfly history (indexed by threat status of both from and to squares)
- `board->threatened` bitmap computed once per position
- Compare to ours: We have `hh[2][4096]` — no threat indexing. This is the "threat-aware history" idea from SUMMARY.md (12 engine consensus).

### Capture History — Defended-Indexed
- `caph[12][64][2][7]` — piece, to_square, defended, captured_piece_type
- Defended flag: `!GetBit(board->threatened, to)` — is the target square defended?
- Compare to ours: We have `caph[12][64][12]` — no defended flag. The defended dimension is novel.

### Continuation History
- `ch[2][12][64][12][64]` — capture_flag, prev_piece, prev_to, curr_piece, curr_to
- **Capture-indexed**: Separate tables for capture vs quiet context (the [2] dimension)
- Read at plies -1, -2, -4, -6 (4-level deep)
- Written at plies -1, -2, -4, -6 (all 4)
- Compare to ours: We read/write at plies -1, -2 only, no capture-indexing, no deeper plies.

### Killers and Counter-Move
- 2 killers per ply
- Counter-move: `counters[piece][to]` of parent move
- Both exempt from history pruning (killerOrCounter check)

---

## 7. Transposition Table

### Structure
- 3-entry buckets (BUCKET_SIZE=3), 32-byte buckets with 2-byte padding
- Entry: 10 bytes packed: hash(2) + depth(1) + agePvBound(1) + evalAndMove(4) + score(2)
- Move stored in lower 20 bits of evalAndMove, eval in upper 12 bits (offset by 2048)
- Age uses upper 5 bits of agePvBound (32 generations), PV flag is bit 2, bound is bits 0-1
- Index: `((u128)hash * count) >> 64` (Fibonacci-like hashing, no power-of-2 restriction)

### Replacement Policy
- Probe scans bucket for hash match or empty slot (depth==0)
- On hash hit: refresh age
- Replace if: `(bound == BOUND_EXACT) || shortHash != tt->hash || depth + 4 > TTDepth(tt)`
- Replacement victim selection: minimizes `depth - (age_diff / 2)` — prefers replacing old, shallow entries

### TT Cutoff Conditions
- `!isPV && ttScore != UNKNOWN && ttDepth >= depth && (cutnode || ttScore <= alpha)`
- `(ttBound & (ttScore >= beta ? BOUND_LOWER : BOUND_UPPER))`
- **cutnode guard**: At cut nodes, always allow cutoff. At non-cut non-PV nodes, only allow if ttScore <= alpha.
- Compare to ours: We don't have the cutnode guard.

### TT PV Flag
- Stored in bit 2 of agePvBound
- Sticky: once PV, always PV (OR'd on probe: `*pv = *pv || TTPV(entry)`)
- Used in LMR: `!ttPv` adds +2 reduction — massive impact

### Depth offset
- DEPTH_OFFSET = -2, allowing QS entries (depth 0 stored as 2, depth -1 stored as 1)

---

## 8. Time Management

### Base Allocation (No movestogo)
```
total = Max(1, time + 50*inc - 50*MOVE_OVERHEAD)
alloc = Min(time * 0.4193, total * 0.0575)
max   = Min(time * 0.9221 - MOVE_OVERHEAD, alloc * 5.928) - 10
```
- Soft limit: ~5.75% of adjusted total, capped at ~42% of remaining time
- Hard limit: ~5.5x soft limit, capped at ~92% of remaining time

### Movestogo Mode
```
total = Max(1, time + movesToGo * inc - MOVE_OVERHEAD)
alloc = Min(time * 0.9, 0.9 * total / Max(1, movesToGo / 2.5))
max   = Min(time * 0.8 - MOVE_OVERHEAD, alloc * 5.5) - 10
```

### Single Root Move
- If only 1 legal move: `max = Min(250, max)` — don't waste time

### Soft Time Scaling (Main Thread Only, Depth >= 5)
Three multiplicative factors:

**1. Stability Factor:**
- Track consecutive iterations where best move is the same
- `searchStability = sameBestMove ? Min(10, searchStability + 1) : 0`
- `stabilityFactor = 1.311 - 0.0533 * searchStability`
- Range: [0.778, 1.258] — stable best move reduces time, unstable extends

**2. Score Change Factor:**
- `searchScoreDiff = scores[depth-3] - bestScore` (3 iterations ago vs now)
- `prevScoreDiff = previousSearchScore - bestScore` (last search's final score vs now)
- `scoreChangeFactor = 0.1127 + 0.0262*searchScoreDiff*(>0) + 0.0261*prevScoreDiff*(>0)`
- Clamped to [0.503, 1.656]
- Only positive diffs (score drops) extend time

**3. Node Count Factor:**
- `pctNodesNotBest = 1.0 - bestMoveNodes / totalNodes`
- `nodeCountFactor = Max(0.563, pctNodesNotBest * 2.267 + 0.450)`
- If best move is TB win: factor = 0.5 (stop quickly)
- Range: [0.450 (100% on best), 2.717 (0% on best)]

**Combined**: `elapsed > alloc * stability * scoreChange * nodeCount` triggers stop.

### Compare to GoChess TM
| Feature | Berserk | GoChess |
|---------|---------|---------|
| Soft base | ~5.75% of total | similar |
| Hard limit | ~5.5x soft | similar |
| Stability | 10-level counter, continuous scaling | 200-based instability counter |
| Score change | 3-iter-ago + previous search, clamped | 2.0x/1.5x/1.2x tiered (UPDATE: merged) |
| Node fraction | Continuous, range [0.45, 2.72] | Not implemented |
| Single move | Cap at 250ms | Not implemented |

**Key gap**: Berserk's node-fraction-based scaling is much more sophisticated than ours. This is a high-value improvement candidate (6 engines have it).

---

## 9. Lazy SMP

### Thread Selection (Vote-Based)
- After all threads stop, compute vote scores: `ThreadValue = (score - worstScore) * depth`
- Build vote map indexed by FromTo of best move
- Select thread that maximizes vote score, with tie-breaking by:
  - Always prefer faster mate avoidance
  - Equal votes: prefer higher ThreadValue with PV length > 2

### Thread Communication
- Only shared state: TT (global) and atomic stop/ponder flags
- All history tables, board state, accumulators, pawn correction are per-thread
- pthread-based sleeping threads (CFish pattern)
- No depth offsets or node-based selection — all threads search the same depths
- Ponder move extracted from TT if PV doesn't have one

---

## 10. Quiescence Search

### QS Structure
- QS depth parameter (starts at 0, decrements)
- At depth >= -1: generates noisy moves + quiet checks
- At depth < -1: generates noisy moves only (no checks)
- In-check: generates all evasions (noisy first, then quiets) using hash move

### QS Pruning
- **Futility**: `futility = bestScore + 63` — if `futility <= alpha && !SEE(move, 1)`, skip
- **SEE filter**: All moves must pass `SEE(move, 0)` (non-negative SEE)
- **QS beta blending**: If `bestScore >= beta`: `bestScore = (bestScore + beta) / 2` — dampens inflated QS scores
- After first non-check capture is searched and scores > -TB_WIN_BOUND, quiet checks are suppressed
- Compare to ours: We have QS beta blending (MERGED, +4.9 Elo). Their futility margin (63) is similar to Midnight's (60).

### QS History Updates
- On beta cutoff in QS: full `UpdateHistories` with depth=1
- This means QS cutoffs feed back into the main search's move ordering

---

## 11. Unique Techniques

### 1. OpponentsEasyCaptures Guard
Computes whether opponent has hanging major/minor pieces:
```c
(queens & rookThreats) | (rooks & minorThreats) | (minors & pawnThreats)
```
This bitmap detects pieces attacked by lower-value pieces. Used to:
- **Gate RFP**: Don't prune when opponent can capture hanging pieces
- **Gate NMP**: Don't null-move when opponent can capture hanging pieces
- `!!` to convert to boolean, then used as `!opponentHasEasyCapture` guard

This is cheap (bitwise ops on existing attack maps) and prevents pruning in tactically unstable positions. **12 engines** have some variant of this.

### 2. Cuckoo Hashing Cycle Detection
`HasCycle(board, ply)` detects upcoming repetitions before they happen, using Cuckoo hash table of all possible move zobrist diffs. This is Stockfish's technique — detects if a sequence of reversible moves could lead to a repetition, and returns a draw-like score early.

### 3. RFP Returns Blended Score
`return (eval + beta) / 2` instead of raw eval or beta. This dampens over-optimistic RFP returns.

### 4. Dynamic Good-Capture SEE Threshold
In move picker: `SEE(board, move, -score/2)` — the SEE threshold for "good capture" classification is based on the capture ordering score. Higher-valued captures are allowed more loosely. This means a queen capture (score ~1015) needs SEE >= -507, while a pawn capture (score ~100) needs SEE >= -50.

### 5. Capture-Indexed Continuation History
The `ch` table has dimension `[2][12][64][12][64]` — the first [2] separates continuation history for contexts where the parent move was a capture vs quiet. This doubles the table size but gives context-dependent patterns.

### 6. LMR CH Bonus After Re-Search
After LMR re-search with R>1, gives CH bonus/malus based on the re-search score relative to alpha/beta. This feeds search information back into move ordering without waiting for the full search to complete.

### 7. avgScore for Aspiration Center
Root moves track `avgScore = (avgScore + score) / 2` as running average. Aspiration window centered on avgScore, not last iteration's score. This smooths the aspiration center.

---

## 12. Notable Differences from GoChess

### Things Berserk has that we don't:
1. **Mate distance pruning** — trivial, 3 lines, universal
2. **Cuckoo cycle detection** — medium complexity, SF-originated
3. **OpponentsEasyCaptures guard** on RFP + NMP — cheap, high-value
4. **Threat-indexed butterfly history** (4x table) — proven in 12+ engines
5. **Capture-indexed continuation history** — doubles CH quality
6. **4-level deep continuation history** (plies 1,2,4,6) with write at all levels
7. ~~**FinnyTable**~~ (accumulator refresh cache) **(UPDATE 2026-03-21: GoChess now has Finny tables)**
8. **TT PV flag** in LMR (+2 reduction for non-PV) — very aggressive reduction
9. **Cutnode LMR** (+1 cap, +2 quiet at cut nodes)
10. **Dynamic good-capture SEE threshold** (-score/2) in move picker
11. **Defended dimension in capture history**
12. **ProbCut QS pre-filter** (two-stage: QS then negamax)
13. **NMP verification** with ply-restricted NMP re-search
14. **Phase-based eval scaling** (1.0x-1.5x)
15. **RFP blended return** (eval+beta)/2
16. **DoDeeper/DoShallower** after LMR re-search
17. **Node-fraction time management** (triple-factor: stability + score + nodes)
18. **Aspiration fail-high depth reduction**
19. **NNZ-sparse L1 matmul** (skip zero activations)
20. **TT cutnode guard** (cutnode || ttScore <= alpha)
21. **Negative extensions** (-2 for ttScore >= beta or cutnode, -1 for ttScore <= alpha)

### Things we have that Berserk doesn't:
1. **Output buckets** in NNUE (8 material-based)
2. **SCReLU activation** **(UPDATE 2026-03-21: GoChess now supports SCReLU inference with SIMD)** — they use plain ReLU
3. **Correction history** as separate table (they have pawn correction only, /2 weight)
4. **TT score dampening** (they don't blend TT cutoff scores)
5. **Fail-high score blending** in main search (they don't)
6. **QS delta pruning** (they rely on SEE + futility)
7. **NMP threat detection** (they use opponentEasyCaptures instead)
8. **TT near-miss cutoffs** (they don't accept shallower TT entries)

---

## 13. Parameter Comparison Table

| Feature | Berserk | GoChess |
|---------|---------|---------|
| Aspiration delta | 9 | 15 |
| Aspiration growth | x1.27 | N/A |
| RFP margin | 70*d | 85*d (imp) / 60*d (not) |
| RFP return | (eval+beta)/2 | eval |
| Razoring margin | 214*d | 400+100*d |
| Razoring max depth | 5 | 3 |
| NMP base R | 4 | 3 |
| NMP eval scaling | 10*(eval-beta)/1024, cap 4 | (eval-beta)/200, cap 3 |
| NMP min depth | 4 | 3 |
| NMP verify depth | 14 | 12 |
| ProbCut margin | 172 | 170 |
| ProbCut depth | >= 6 | >= 5 |
| ProbCut QS pre-filter | Yes | No |
| LMR table | log*log/2.04+0.24 | split cap/quiet |
| LMR history div | 8192 | 5000 |
| LMR !ttPv | +2 | N/A |
| LMR cutnode | +1/+2 | N/A |
| LMR killer/counter | -2 | N/A |
| LMR gives check | -1 | N/A |
| LMR ttDepth deep | -1 | N/A |
| LMP not-imp | 1.3+0.35*d^2 | 3+d^2 |
| LMP improving | 2.19+0.99*d^2 | (3+d^2)*1.5 |
| Futility margin | 81+46*lmrD | 80+80*lmrD |
| Futility max depth | lmrD < 10 | similar |
| SEE quiet | -15.27*d^2 | -20*d^2 |
| SEE capture | -94.06*d | -20*d^2 |
| History pruning | -2788*(d-1) | -2000*d |
| SE min depth | 6 | 10 |
| SE margin | 5*d/8 | 3*d |
| SE double ext | < sBeta-14 | N/A |
| SE triple ext | < sBeta-48 | N/A |
| SE ply limit | ply < depth*2 | N/A |
| SE multi-cut | Yes | N/A |
| SE negative ext | -2/-1 | N/A |
| IIR | PV/cutnode, d>=4 | any node, d>=4 |
| ContHist depth | plies 1,2,4,6 | plies 1,2 |
| ContHist capture-indexed | Yes ([2]) | No |
| Pawn correction | /2 weight, grain 256 | /2 weight |
| QS futility | +63 | N/A |
| QS beta blend | Yes | Yes |
| History bonus | 4*d^2+164*d-113, cap 1729 | similar |
| History gravity div | 16384 | similar |

---

## 14. Ideas Worth Testing from Berserk (Prioritized)

### Already tested / merged from Berserk:
- TT noisy detection (+1 LMR for quiets): **MERGED +34.4 Elo**
- Fail-high score blending: **MERGED +14.7 Elo**
- QS beta blending: **MERGED +4.9 Elo**
- Alpha-reduce: **MERGED +13.0 Elo**
- DoDeeper/DoShallower: tested, -13.7 Elo (margin needs calibration)
- 50-move eval scaling: tested, H0

### High priority (not yet tested, high Berserk-specific value):

1. **Fix Singular Extensions** (CRITICAL) — Berserk's SE is the single biggest search difference. They use depth>=6, margin 5*d/8, ply limiter `ply<depth*2`, triple/double ext with counter, multi-cut return, negative extensions. Our SE is catastrophically broken (-58 Elo). Copying Berserk's exact parameters could recover 50-100+ Elo.

2. **OpponentsEasyCaptures guard on RFP + NMP** — Cheap bitwise computation of hanging pieces, used to prevent pruning in tactical positions. Guards both RFP and NMP. Aligns with pattern #6 (guards prevent over-pruning). Est. +5-15 Elo.

3. **TT PV flag with +2 LMR** — Store PV flag in TT, use to add +2 reduction on non-PV nodes. This is one of Berserk's most aggressive reductions. Requires adding 1 bit to our packed TT. Est. +5-10 Elo.

4. **Threat-indexed butterfly history** — 4x table indexed by `[from_threatened][to_threatened]`. 12+ engines have this. Berserk's exact implementation: `hh[2][2][2][4096]`. Est. +5-15 Elo.

5. **Cutnode LMR (+1/+2)** — `R += 1 + !IsCap(move)` at cut nodes. 3+ engines. Weiss uses flat +2. Already in our testing queue. Est. +3-8 Elo.

6. **Node-fraction time management** — Triple multiplicative scaling (stability, score change, node fraction). Berserk's implementation is the most sophisticated we've seen. Est. +5-15 Elo.

7. **RFP blended return** `(eval+beta)/2` — Dampens optimistic RFP. Aligns with pattern #1 (dampening at boundaries). Previously tested as "RFP score dampening" and got -16.7 Elo — but that may have been a different formula. Berserk blends eval+beta, not dampening toward a fixed point. Retry candidate.

8. **Capture-indexed continuation history** — Separate CH for capture vs quiet parent context. Doubles CH effectiveness. Medium complexity but high engine consensus. Est. +3-8 Elo.

9. **4-level deep continuation history** (plies 1,2,4,6) with read at x1 weight for plies 4,6 — tested as "ContHist4" and got -58 Elo. But Berserk uses x1 weight (half the x2 weight of plies 1-2), not our 1/4 weight. Retry with Berserk's exact weighting.

10. **ProbCut QS pre-filter** — Two-stage ProbCut (QS first, then negamax only if QS passes). Already a testing candidate from Alexandria. Est. +3-8 Elo.

11. **Aspiration delta 9** — Much tighter than our 15. With fail-high depth reduction. Multiple parameters to tune together. Est. +2-5 Elo.

12. ~~**FinnyTable**~~ (accumulator refresh cache) — **(UPDATE 2026-03-21: MERGED)**

### Lower priority:
13. **Mate distance pruning** — 3 lines, universal, free.
14. **Cuckoo cycle detection** — Medium complexity but prevents repetition-seeking.
15. **Dynamic good-capture SEE threshold** (-score/2 in picker) — Novel, worth testing.
16. **Defended dimension in capture history** — Adds `[2]` to capture history, small extra info.
17. **LMR ttDepth deep bonus** (-1 if ttDepth >= depth) — 1 line.
18. **LMR killer/counter bonus** (-2) — 1 line, our killers already get ordering priority.
19. **NMP base R=4** (vs our 3) — More aggressive null move, but compensated by easy-capture guard.
20. **Phase-based eval scaling** — Requires phase tracking infrastructure.
