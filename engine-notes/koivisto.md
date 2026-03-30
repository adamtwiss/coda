# Koivisto Chess Engine - Technical Review

Source: `~/chess/engines/Koivisto/src_files/`
Authors: Kim Kahre, Finn Eggers, Eugenio Bruno
License: GPLv3
NNUE: HalfKP-like 12288->2x512->1 (16 king buckets, CReLU, int16)

---

## 1. NNUE Architecture

### Network Topology

```
Input: 6 piece_types * 64 squares * 2 colors * 16 king_buckets = 12,288 features
    |
    v
Feature Transform: 12288 -> 512 (int16, per-perspective)
    |
    v
Concatenate: [active_perspective(512), opponent_perspective(512)] = 1024
    |
    v
CReLU: max(x, 0)
    |
    v
Output: 1024 -> 1 (single output, no buckets)
```

**Ref**: `nn/defs.h:35-38`, `nn/weights.h:26-29`, `nn/eval.cpp:88-110`

- `INPUT_SIZE = 12288`, `HIDDEN_SIZE = 512`, `HIDDEN_DSIZE = 1024`, `OUTPUT_SIZE = 1`
- Weights: int16 feature weights, int16 hidden weights, int16 input bias, int32 hidden bias
- Quantization: `INPUT_WEIGHT_MULTIPLIER = 32` (QA), `HIDDEN_WEIGHT_MULTIPLIER = 128` (QB)
- Final: `(sum + hiddenBias) / 32 / 128` (divide by 4096)
- No output buckets -- single scalar output

**Coda comparison**: Our v7 has 12288->1024->16->32->1x8 (with output buckets, SCReLU, hidden layers). Koivisto is simpler/faster (single hidden layer, no output buckets) but less expressive. We match top engine convention (16->32->1x8); Koivisto uses an older-gen 512-wide single layer.

### King Buckets (16 buckets, horizontally mirrored)

**Ref**: `nn/index.h:28-62`

```cpp
constexpr int kingSquareIndices[64] {
    0,  1,  2,  3,  3,  2,  1,  0,     // rank 1 (white POV)
    4,  5,  6,  7,  7,  6,  5,  4,     // rank 2
    8,  9,  10, 11, 11, 10, 9,  8,     // rank 3-4 (same buckets)
    8,  9,  10, 11, 11, 10, 9,  8,
    12, 12, 13, 13, 13, 13, 12, 12,    // rank 5-6
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,    // rank 7-8
    14, 14, 15, 15, 15, 15, 14, 14,
};
```

- Vertically symmetric (left-right mirrored at file d/e boundary)
- Horizontal mirroring: if king is on files e-h (`kingSquare & 0x4`), XOR square by 7
- Feature index: `square + pieceType*64 + (sameColor^view)*64*6 + ksIndex*64*6*2`
- View flipping: `square ^= 56 * view` for black perspective

**Coda comparison**: Our king bucket system is identical (16 buckets, same mirroring). Their `kingSquareIndex` table has slightly different rank groupings (ranks 3-4 share buckets, ranks 5-6 share), ours may differ.

### Accumulator (Stack-Based with Lazy Copy)

**Ref**: `nn/accumulator.h:31-33`, `nn/eval.cpp:128-148`

```cpp
struct Accumulator {
    alignas(ALIGNMENT) int16_t summation[2][512];  // ~2KB per accumulator
};
```

- `history[]` vector of Accumulators, indexed by `history_index`
- `addNewAccumulation()`: increments index, marks both colors as uninitialized
- **Lazy copy on first write**: when `accumulator_is_initialised[side] == false`, first feature update copies from `history[index-1]` to `history[index]` before adding delta
- `popAccumulation()`: decrements index, marks both as initialized (parent is still valid)
- King moves that change bucket or cross file-d/e boundary trigger `resetAccumulator()`

**Coda comparison**: Our accumulator stack works similarly. Their lazy copy pattern (defer copy until first write per color) is a minor optimization -- we copy both perspectives immediately.

### AccumulatorTable (Finny Table)

**Ref**: `nn/accumulator.cpp:22-78`

```cpp
struct AccumulatorTable {
    AccumulatorTableEntry entries[2][32];  // [color][king_bucket_index * 2 + king_side]
};
struct AccumulatorTableEntry {
    U64 piece_occ[2][6];     // bitboards per color per piece type
    Accumulator accumulator;
};
```

- 32 entries per color = 16 king buckets * 2 halves (queen-side vs king-side)
- Entry indexed by: `king_side * 16 + kingSquareIndex`
- On king reset: diffs stored `piece_occ` against current board, applies only changed features
- After update: copies result to evaluator's history accumulator
- Reset: initializes all accumulators to input bias

**Coda comparison**: This is the same FinnyTable design we have (merged). Their 32-entry structure (with king-side split) is identical to what Alexandria/Obsidian use. Our implementation (`[NNUEKingBuckets][mirror]`) matches this pattern.

### SIMD Implementation

**Ref**: `nn/defs.h:40-102`, `nn/eval.cpp:26-52,88-110`, `nn/accumulator.h:65-202`

- Compile-time ISA detection: AVX-512 > AVX2 > SSE2 > NEON
- Unified register abstraction: `avx_register_type_16`, `avx_register_type_32` + generic ops
- NEON: custom `avx_madd_epi16` using `vmull_s16 + vmull_high_s16 + vpaddq_s32`
- Accumulator updates: 4x loop unrolling (`HIDDEN_SIZE / STRIDE_16_BIT / 4`) with register tiling
- Output: dot product of `max(acc, 0)` with hidden weights via `avx_madd_epi16`
- Horizontal sum via `sumRegisterEpi32`: cascade reduce 512->256->128->scalar

**Batched multi-delta updates** (`nn/accumulator.h:93-202`): Specialized functions for 1-delta (`setUnsetPiece`), 2-delta (`setUnsetUnsetPiece`), and 3-delta (`setSetUnsetUnsetPiece`) accumulator updates. Each loads all involved weight rows into registers and applies all adds/subs in a single pass with `REG_COUNT=16` register tiling and `CHUNK_UNROLL_SIZE=BIT_ALIGNMENT` chunking.

```cpp
// Example: setSetUnsetUnsetPiece (castling: add king+rook destinations, remove king+rook origins)
for (size_t c = 0; c < HIDDEN_SIZE / CHUNK_UNROLL_SIZE; c++) {
    for (size_t i = 0; i < REG_COUNT; i++) regs[i] = load(&acc_in[i]);
    for (size_t i = 0; i < REG_COUNT; i++) regs[i] = add(regs[i], wgt_set1[i]);
    for (size_t i = 0; i < REG_COUNT; i++) regs[i] = add(regs[i], wgt_set2[i]);
    for (size_t i = 0; i < REG_COUNT; i++) regs[i] = sub(regs[i], wgt_unset1[i]);
    for (size_t i = 0; i < REG_COUNT; i++) regs[i] = sub(regs[i], wgt_unset2[i]);
    for (size_t i = 0; i < REG_COUNT; i++) store(&acc_out[i], regs[i]);
}
```

**Coda comparison**: We do single-delta updates. Their multi-delta batching (especially for castling = 4 weight changes) is more cache-friendly since all weight rows are loaded and processed in one pass. Worth considering for hot-path optimization. Our SIMD is equivalent quality (AVX2/NEON with runtime detection).

---

## 2. Search

### Iterative Deepening + Aspiration Windows

**Ref**: `search.cpp:322-357`

- Depth 1 to MAX_PLY (128), no aspiration below depth 6
- Initial window: `score +/- 10`
- On fail-high: `beta += window; sDepth--` (unique: depth reduced on fail-high)
- On fail-low: `beta = (alpha+beta)/2; alpha -= window` (beta contracts toward alpha)
- Window doubles: `window += window`, goes full-width at 500
- sDepth floor: `max(sDepth, depth-3)`
- Single legal move at root: stop search early (only with match time limit)

**Coda comparison**: We use delta=15, double on fail, full at 500. Their fail-low beta contraction is something we already have. Their fail-high depth reduction (`sDepth--`) is an interesting trick we haven't tried (the idea: if we're failing high, we can afford slightly less depth to confirm).

### Pruning

#### Razoring (search.cpp:622-625)
- `depth <= 3 && staticEval + 190*depth < beta` => qSearch; return if `< beta`
- Margins: d1=190, d2=380, d3=570

#### Reverse Futility Pruning (search.cpp:633-639)
- `depth <= 7 && staticEval >= beta + (depth - (isImproving && !enemyThreats)) * 68 && staticEval < MIN_MATE_SCORE`
- **Threat guard**: margin reduces by 1*FUTILITY_MARGIN when improving AND no enemy threats
- Extra depth-1 pruning: `staticEval > beta + (improving ? 0 : 30) && !enemyThreats` => return beta

#### Null Move Pruning (search.cpp:647-670)
- `staticEval >= beta + (depth < 5 ? 30 : 0) && !(depth < 5 && enemyThreats) && !hasOnlyPawns`
- **Threat guard**: disabled at `depth < 5` when opponent has threats (Koivisto original)
- R = `depth/4 + 3 + futilityAdj`
- FutilityAdj: `(staticEval-beta)/68` when `staticEval-beta < 300`, else 3
- Mate score clamped to beta

#### IIR (search.cpp:730-731)
- `depth >= 4 && !hashMove` => `depth--`

#### ProbCut (search.cpp:683-724)
- `!inCheck && !pv && !skipMove && depth > 4 && ownThreats > 0`
- **Threat guard**: only when we have our own threats (Koivisto original)
- betaCut = `beta + 130`
- Skip if `hashMove && en.depth >= depth-3 && ttScore < betaCut`
- Two-phase: qSearch first, then pvSearch at `depth-4` if qSearch passes

#### Mate Distance Pruning (search.cpp:747-762)
- Standard: tighten alpha/beta based on `MAX_MATE_SCORE - ply`

#### Late Move Pruning (search.cpp:808-814)
- Table-based: `lmp[2][8]`:
  - Not improving: `{0, 2, 3, 5, 8, 12, 17, 23}`
  - Improving: `{0, 3, 6, 9, 12, 18, 28, 40}`
- `depth <= 7`
- When triggered: `skip()` flag on move generator, remaining quiets skipped except those attacking checker squares

#### Quiet Futility Pruning (search.cpp:817-823)
- `!inCheck && moveDepth <= 7`
- `maxImprovement[from][to] + moveDepth*68 + 100 + evalHistory < alpha`
- Uses per-move improvement tracking as adaptive margin

#### History Pruning (search.cpp:831-836)
- `!inCheck && getHistories(m) < min(140 - 30*depth*(depth+isImproving), 0)`
- Quadratic threshold: gets very aggressive at deeper depths

#### SEE Pruning (search.cpp:844-847)
- `moveDepth <= 5 + quiet*3` (8 for quiets, 5 for captures)
- Requires: captured piece type < moving piece type
- Quiet margin: `-40 * moveDepth`
- Capture margin: `-100 * moveDepth`

### Extensions

#### Singular Extensions (search.cpp:871-899)
- `depth >= 8 && !skipMove && !inCheck && m == hashMove && legalMoves == 0 && ply > 0`
- `en.depth >= depth-3 && abs(ttScore) < MIN_MATE_SCORE && (CUT_NODE || PV_NODE)`
- Singularity beta: `min(ttScore - 2*depth, beta)` (narrow margin)
- Singularity depth: `depth >> 1`
- **LMR cancellation**: if singular, parent's LMR can be undone via `lmrFactor` pointer
  ```cpp
  if (score < betaCut) {
      if (lmrFactor != nullptr) {
          depth += *lmrFactor;  // restore parent's depth
          *lmrFactor = 0;
      }
      extension++;
  }
  ```
- **Multi-cut 1**: `score >= beta` => return score
- **Multi-cut 2**: `ttScore >= beta` => re-search at `(depth>>1)+3`; return if >= beta
- After SE, re-initializes move generator

**Coda comparison**: Our SE uses `depth >= 10` and `ttScore - 3*depth` margin (wider). Their SE LMR cancellation is unique and theoretically interesting: a singular child should never have been LMR'd. However, SE is an extension-class feature that we found amplifies 3:1 cross-engine -- use caution.

#### Check Extension (search.cpp:932-933)
- `extension == 0 && depth > 4 && opponent in check` after move => +1
- Only at deeper depths

#### Non-PV Hash Move Extension (search.cpp:935-936)
- `sameMove(hashMove, m) && !pv && en.type > ALL_NODE` => +1
- Extends TT move in non-PV nodes with CUT/PV-type entry

#### Eval-Based Extension (search.cpp:909-917)
- `depth < 8 && !skipMove && !inCheck && m == hashMove && ply > 0`
- `evalHistory < alpha - 25 && en.type == CUT_NODE` => +1
- Extends when eval suggests we're in trouble but TT says cutoff

### LMR

**Ref**: `search.cpp:37-43,166-235,239-245`

#### Table
```cpp
lmrReductions[d][m] = 1.25 + log(d) * log(m) * 100 / 267
```
Single table for both quiets and captures. LMR_DIV = 267.

#### Skip Conditions (lmr = 0)
- `legalMoves < 2 - (hashMove != 0) + pv` (first 1-2 moves depending on PV/hash)
- `depth <= 2`
- `isCapture && SEE > 0` (winning captures)
- `isPromotion && queen promotion`

#### Adjustments (search.cpp:182-233)
1. `+1` if behind null move and same side (`activePlayer == behindNMP`)
2. `-history/150` based on combined history scores
3. `+1` if not improving
4. `-1` if PV node
5. `+1` if `!targetReached` (< 10% of allocated time used)
6. `-1` if killer move
7. `+1` if `sd->reduce && sd->sideToReduce != activePlayer` (PV first-move propagation)
8. `+min(2, abs(staticEval - alpha) / 350)` (eval-alpha distance)
9. `-bitCount(getNewThreats(b, m))` (moves creating new threats)
10. Clamped to `[0, depth-2]`
11. **Override to 0**: if `history > 256 * (2 - isCapture)` (512 for quiets, 256 for captures)

**Coda comparison**: We have separate cap/quiet tables with C=1.80/1.50 (div ~55/67), they use single table with div 267 (much lower base reduction). Their adjustment list is longer and more nuanced. Key unique features: behind-NMP tracking, targetReached, eval-alpha distance, new-threats, high-history override. Our LMR is simpler with fewer adjustments.

### PVS Re-Search

**Ref**: `search.cpp:938-962`

Standard pattern:
1. First move: full window `(-beta, -alpha)`
2. Later moves: ZWS with LMR `(-alpha-1, -alpha, depth-1-lmr)`
3. If LMR and score > alpha: re-search at full depth `(-alpha-1, -alpha, depth-1)`
4. If score > alpha and score < beta: full window `(-beta, -alpha, depth-1)`

**LMR factor passing**: `lmr != 0 ? opponent : behindNMP` propagates null-move context. The `&lmr` pointer allows child SE to cancel parent LMR.

---

## 3. Move Ordering

### Stages

**Ref**: `newmovegen.h:41-53`, `newmovegen.cpp:66-141`

```
GET_HASHMOVE -> GEN_NOISY -> GET_GOOD_NOISY -> KILLER1 -> KILLER2 ->
GEN_QUIET -> GET_QUIET -> GET_BAD_NOISY -> END
```

QSearch: `GET_HASHMOVE -> GEN_NOISY -> GET_GOOD_NOISY -> END`
QSearch in check: adds `QS_EVASIONS` (king quiet moves)

### Noisy Scoring (newmovegen.cpp:143-162)

All captures get SEE computed during generation (`staticExchangeEvaluation`).
- Good (SEE >= 0): `100000 + getHistories(m) + SEE + 150*(sqTo == prevSqTo)`
  - Recapture bonus: +150
- Bad (SEE < 0): `10000 + getHistories(m)`
- `goodNoisyCount` tracks the boundary; good noisies searched before killers/quiets, bad noisies after

### Quiet Scoring (newmovegen.cpp:164-176)

Direct sum of three tables (precomputed pointers during init):
```cpp
quietScores[i] = m_th[getSqToSqFromCombination(m)]          // threat history
               + m_cmh[getPieceTypeSqToCombination(m)]       // counter-move history
               + m_fmh[getPieceTypeSqToCombination(m)];      // followup-move history
```

No traditional butterfly history (from-to). Threat history IS the butterfly replacement.

### Selection Sort
Both noisy and quiet use selection sort (find max, swap to front). No insertion sort or partial sort.
After `skip()` (LMP triggered), quiets scanned linearly for moves that attack checker squares only.

**Coda comparison**: We use staged move ordering with similar structure. Key differences: they score captures with full history (not just MVV-LVA), they have no butterfly history (replaced by threat history), and they batch SEE computation during generation (we compute SEE lazily or on demand).

---

## 4. History Tables

### Threat History (replaces butterfly)

**Ref**: `history.h:38-40,74-76`

```cpp
int th[2][65][4096];  // [side][threatSquare][fromTo]
// 65 = 64 squares + 1 for "no threat" (square 64)
```

This is Koivisto's signature innovation. Instead of plain butterfly `history[side][from*64+to]`, it indexes by the opponent's primary threat square. The same move gets different scores depending on what the opponent is threatening.

### Counter-Move History (CMH)

```cpp
int cmh[384][2][384];  // [prev_piece*64+prev_to][side][curr_piece*64+curr_to]
```
Standard 1-ply continuation history.

### Followup-Move History (FMH)

```cpp
int fmh[385][2][384];  // [followup_piece*64+followup_to][side][curr_piece*64+curr_to]
```
Standard 2-ply continuation history. Extra entry (385) for null previous move.

### Capture History

```cpp
int captureHistory[2][4096];  // [side][fromTo]
```
Simple from-to indexed, captures only.

### Combined Score (history.cpp:37-47)

```cpp
int getHistories(Move m, Color side, Move previous, Move followup, Square threatSquare) {
    if (isCapture(m)) return captureHistory[side][fromTo];
    auto fmh_value = followup ? FMH(followup, side, m) : 0;
    auto cmh_value = CMH(previous, side, m);
    auto th_value  = THREAT_HISTORY(side, threatSquare, m);
    return (2*fmh + 2*cmh + 2*th) / 3;  // weighted average
}
```

### History Update (Gravity)

**Ref**: `newmovegen.cpp:642-677`

```cpp
weight = min(depth^2 + 5*depth, 384);
UPDATE_UP(value):   value += weight - weight * value / 512
UPDATE_DOWN(value): value += -weight - weight * value / 512
```

- MAX_HIST = 512 (gravity cap)
- Weight formula: d^2 + 5d capped at 384 (d3=24, d5=50, d10=150, d12=204)
- Best move: bonus on all relevant tables
- All searched moves (not just best): malus, capped at weight=128
- **Eval-based bonus**: `depth + (staticEval < alpha)` -- surprising cutoffs get +1 depth in weight
- Captures: only capture history updated (bonus/malus)
- Quiets: all 3 tables (th, cmh, fmh) updated

**Coda comparison**: We use divisor 5000 for history scaling. Their MAX_HIST=512 with gravity formula keeps values bounded tighter. Their eval-based weight bonus (+1 when eval < alpha) is shared with Alexandria.

### Killers

```cpp
Move killer[2][MAX_PLY+2][2];  // [color][ply][id]
```
- 2 killers per ply per color
- Reset grandchildren killers at each node (`ply+2`)
- Color-indexed (not just ply-indexed like many engines)

### spentEffort Table (for time management)

```cpp
int64_t spentEffort[64][64];  // [from][to]
```
- At root (depth 1): reset to 0
- After each root move: `+= (nodes after - nodes before)`
- Used in time management: `bestMoveEffort * 100 / totalNodes`

### maxImprovement Table (for futility)

```cpp
int maxImprovement[64][64];  // [from][to]
```
- Updated at every non-check node for the previous move:
  `improvement = -staticEval - evalHistory[opponent][ply-1]`
- Used in quiet futility pruning as per-move adaptive margin

---

## 5. Threat Computation (Koivisto Original)

**Ref**: `search.cpp:66-164`

At every non-check node, computes threats for BOTH sides:

```cpp
template<Color color>
U64 getThreatsOfSide(Board* b, SearchData* sd, Depth ply) {
    pawn_attacks  = pawnAttacks(pawns) & (opp_major | opp_minor);  // pawns threatening pieces
    minor_attacks = knightAttacks | bishopAttacks & opp_major;      // minors threatening majors
    rook_attacks  = rookAttacks & opp_queen;                        // rooks threatening queens
    THREAT_COUNT(sd, ply, color) = popcount(all threats);
    return pawn_attacks | minor_attacks | rook_attacks;
}
```

Stored as:
- `threatCount[ply][side]` -- count of threatening attacks per side
- `mainThreat[ply]` -- square of first threat to active player (or 64 if none)

Used in:
1. **RFP**: margin adjusted when improving AND no enemy threats
2. **NMP**: disabled at d<5 when enemy has threats
3. **ProbCut**: only triggers when own threats exist
4. **Threat History**: mainThreat indexes the history table
5. **LMR**: `-bitCount(getNewThreats(b, m))` for moves creating new threats

### getNewThreats (per-move threat creation)

Computes attacks the move CREATES (new attacks minus old attacks from same piece):
- Rook: new attacks on queens
- Bishop: new attacks on rooks/queens
- Knight: new attacks on rooks/queens
- Pawn: new attacks on all non-pawn pieces
- Queen/King: return 0

**Coda comparison**: We have no threat computation. This is the backbone of 5+ features in Koivisto. The computational cost is moderate (attack lookups, already have magic bitboard infrastructure). The threat history replacement for butterfly history is the biggest payoff.

---

## 6. Transposition Table

**Ref**: `transpositiontable.h`, `transpositiontable.cpp`

### Structure
- **Single entry per index** (no buckets)
- Index: `zobrist & mask` (power-of-2)
- Key: `zobrist >> 32` (upper 32 bits)
- Entry: 12 bytes (U32 key, Move move [upper 8 bits = age], Depth u8, NodeType u8, Score i16, Eval i16)

### Node Types
```cpp
PV_NODE = 0, CUT_NODE = 1, ALL_NODE = 2, FORCED_ALL_NODE = 3
```
- `FORCED_ALL_NODE`: stored when `depth > 7` and best move consumed > 50% of node effort
- `ALL_NODE` check uses `en.type & ALL_NODE` (catches both 2 and 3)

### Replacement Policy (transpositiontable.cpp:118-149)
Replace if ANY:
1. Entry empty (key == 0)
2. Different age (new search)
3. New entry is PV_NODE
4. Non-PV entry with `depth <= newDepth`
5. Same hash with `depth <= newDepth + 3`

### TT Probe in Search (search.cpp:511-529)

```cpp
if (!pv && en.depth + (!b->getPreviousMove() && ttScore >= beta) * 100 >= depth) {
```
**Unique**: After null move (`!previousMove`), TT entries treated as depth+100 for fail-highs. Rationale: NMP already proved the position is strong, so a TT fail-high in the child is highly trustworthy regardless of stored depth.

### Static Eval Adjustment (search.cpp:578-585)
If TT entry matches, adjust staticEval:
- PV_NODE: use ttScore
- CUT_NODE: use ttScore if > staticEval
- ALL_NODE: use ttScore if < staticEval

**Coda comparison**: We use 4-slot buckets with lockless atomic packing (more robust for SMP). Their single-entry TT is simpler but has more hash collisions. Their null-move depth relaxation is clever and low-cost. Their FORCED_ALL_NODE distinction is interesting for replacement policy.

---

## 7. Time Management

**Ref**: `timemanager.cpp:98-279`

### Base Allocation
```cpp
timeToUse = 2*inc + 2*time/movesToGo
upperTimeBound = time / 2
// Both clamped to: time - overhead - inc - overheadPerGame
```
Under 1000ms with no increment: `time *= 0.7`

### Root Time Decision (rootTimeLeft)
```cpp
nodeScore = 110 - min(spentEffort[best] * 100 / totalNodes, 90);  // range 20-110
evalScore = min(max(50, 50 + evalDrop), 80);                       // range 50-80
stop if: timeToUse * nodeScore / 100 * evalScore / 65 < elapsed
```
- Best move stability: 90%+ effort on best => 0.20x time, distributed effort => 1.10x time
- Score drop: big drop => 0.80x multiplier (spend more time)
- Combined range: 0.15x to 1.35x of base time

### targetReached Flag
- `false` when `elapsed * 10 < timeToUse` (using < 10% of time)
- Used in LMR: +1 reduction when target not reached (search faster early)

**Coda comparison**: Our time management is simpler. Their multiplicative nodeScore * evalScore scaling is elegant. The targetReached->LMR connection is a unique cross-cutting optimization.

---

## 8. QSearch

**Ref**: `search.cpp:1073-1187`

### Key Features
- TT probing with full cutoffs (no depth requirement)
- Stand-pat from TT eval if hit, otherwise full NNUE eval
- TT eval adjustment (same as main search)
- Move generation: good captures only (SEE >= 0) in normal QS, all captures + king evasions in check
- **Good-delta pruning**: `SEE + stand_pat > beta + 200 => return beta` (unique)
- SEE < 0: skip
- TT storage: depth = `!inCheckOpponent` (0 or 1)

**Coda comparison**: We have QS beta blending `(bestScore+beta)/2` which they don't. Their good-delta pruning (`SEE + stand_pat > beta + 200`) is interesting -- it's a beta-side delta prune we don't have.

---

## 9. Novel Features Summary (Actionable Items)

### High Priority (unique Koivisto features worth testing)

1. **Threat History** -- Replace butterfly history with `th[side][threatSquare][fromTo]`. The opponent's primary threat square contextualizes the history. This is a "category 1" change (accuracy/information improvement) that should transfer ~1:1 to cross-engine.
   - Estimated: +5 to +10 Elo
   - Complexity: Medium

2. **Bilateral Threat Computation** -- Compute threats at every non-check node for both sides. Enables 5+ downstream features. Cost: ~2% NPS for attack lookups (already have magic bitboards).
   - Estimated: +5 to +15 Elo (across all dependent features)
   - Complexity: Medium (but amortized across many features)

3. **maxImprovement Table** -- Track per-move `[from][to]` eval improvement. Use in futility as adaptive margin: `maxImprovement[from][to] + moveDepth*margin + 100 + evalHistory < alpha`.
   - Estimated: +3 to +8 Elo
   - Complexity: Low

4. **LMR behind-NMP tracking** -- Pass `behindNMP` color through recursion. +1 LMR when same side is behind null move (their position was already proven strong).
   - Estimated: +2 to +5 Elo
   - Complexity: Low (add parameter to pvSearch)

5. **NMP threat guard** -- Disable NMP at `depth < 5` when `enemyThreats > 0`. Prevents null-move in tactical positions at shallow depth.
   - Estimated: +2 to +5 Elo
   - Complexity: Low (needs threat computation)

### Medium Priority

6. **TT depth relaxation after null move** -- `!previousMove && ttScore >= beta` => treat TT depth as depth+100. Trivial implementation, theoretically sound.
   - Estimated: +2 to +5 Elo
   - Complexity: Trivial

7. **Multi-delta accumulator batching** -- Specialized SIMD functions for 2/3/4-delta updates (captures, castling). More cache-friendly than sequential single-delta calls.
   - Estimated: +1-3% NPS
   - Complexity: Medium

8. **Good-delta QS pruning** -- `SEE + stand_pat > beta + 200 => return beta`. Beta-side only.
   - Estimated: +2 to +5 Elo
   - Complexity: Trivial

9. **Mate distance pruning** -- Standard. We don't have it.
   - Estimated: +1 to +3 Elo
   - Complexity: Trivial

10. **FORCED_ALL_NODE TT type** -- When best move consumed > 50% of effort at depth > 7, store as distinct ALL subtype. Improves replacement policy.
    - Estimated: +1 to +3 Elo
    - Complexity: Trivial

### Lower Priority / Caution

11. **SE LMR cancellation** -- Singular child cancels parent LMR. Interesting but SE is extension-class (3:1 amplification cross-engine).
    - Test with caution

12. **LMR targetReached** -- +1 reduction when < 10% of time used. Tight coupling to time management.
    - Estimated: +1 to +3 Elo

13. **LMR eval-alpha distance** -- `+min(2, abs(eval-alpha)/350)`. This is a pruning-class change -- verify cross-engine.

14. **Aspiration fail-high depth reduction** -- `sDepth--` on fail-high. Previously tested at -353.8 Elo (implementation bug suspected).

---

## 10. Parameter Comparison Table

| Feature | Koivisto | GoChess/Coda |
|---------|----------|--------------|
| NNUE hidden | 2x512, CReLU, int16 | v7: 1024->16->32->1x8, SCReLU |
| Output buckets | None | 8 |
| QA/QB | 32/128 | 255/64 |
| RFP margin | 68*(d-imp_no_threats), d<=7 | 80*d, d<=8 |
| Razoring | 190*d, d<=3 | 400+100*d, d<=3 |
| NMP base R | d/4+3 | 3+d/3 |
| NMP futility | (eval-beta)/68, max 3 | min((eval-beta)/200, 3) |
| LMR divisor | 267 (single table) | Separate cap/quiet tables |
| LMR adjustments | 11 factors | ~6 factors |
| History type | Threat-history (no butterfly) | Butterfly + cont-hist |
| History gravity | MAX=512, d^2+5d cap 384 | Divisor 5000 |
| SE depth | >=8 | >=10 |
| SE margin | ttScore - 2*depth | ttScore - 3*depth |
| ProbCut margin | 130 | 170 |
| ProbCut guard | ownThreats required | None |
| TT structure | Single-entry, 12 bytes | 4-slot buckets, lockless |
| TT null-move trick | +100 depth for fail-highs | No |
| Aspiration | 10, fail-high sDepth-- | 15 |
| Finny table | Yes (32 entries/color) | Yes |
| Mate distance | Yes | No |
| QS good-delta | SEE+stand_pat > beta+200 | No |
