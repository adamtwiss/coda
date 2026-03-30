# Berserk 13 — Technical Analysis

**Source:** `~/chess/engines/berserk-13/src/`
**Author:** Jay Honnold
**Rating:** ~3500+ (top 10)
**Language:** C, pthreads, AVX-512/AVX2/SSE2/SSSE3 SIMD

---

## 1. NNUE Architecture

**Architecture:** `(16×12×64 → 1024)×2 → 16 → 32 → 1` (FT → L1 → L2 → output)

This is the exact same topology we're targeting with v7. Key dimensions from `types.h`:

```
N_KING_BUCKETS = 16
N_FEATURES     = 16 * 12 * 64 = 12288
N_HIDDEN       = 1024       (accumulator width)
N_L1           = 2048       (2 * N_HIDDEN, both perspectives concatenated)
N_L2           = 16         (first hidden layer)
N_L3           = 32         (second hidden layer)
N_OUTPUT       = 1
```

### Quantization Scheme

| Layer | Weight type | Bias type | Notes |
|-------|-----------|-----------|-------|
| FT (input) | int16 | int16 | Standard HalfKA accumulator |
| L1 (FT→16) | int8 | int32 | Sparse matmul with NNZ detection |
| L2 (16→32) | **float32** | **float32** | Float inference, no quantization |
| Output (32→1) | **float32** | **float32** | Float inference |

**Coda comparison:** We use int16 for L1 and int16 for L2. Berserk uses int8 for L1 (faster VPMADDUBSW) and float for L2/output. The int8 L1 is critical for the sparse matmul optimization. We should match this: int8 L1 with float L2/output.

### Activation: CReLU (not SCReLU)

The `InputReLU` function clamps to [0, 127] after right-shifting by 5 (dividing by QA=32). This is plain CReLU — no squaring. The accumulator values (int16) are shifted right by 5 bits, packed to int8, and clamped to [0, max].

```c
// evaluate.c:63-69 (AVX-512 path)
__m512i s0 = _mm512_srai_epi16(in[2 * i + 0], 5);
out[i]     = _mm512_max_epi8(_mm512_packs_epi16(s0, s1), _mm512_setzero_si512());
```

**Coda comparison:** We use SCReLU for v7. Berserk uses CReLU but with int8 quantization for L1. The key difference: CReLU doesn't have the dying ReLU problem when combined with int8 (the dead neurons issue is a training problem with narrow hidden layers, not an inference one). Berserk likely handles this via training strategy.

### Sparse L1 Matmul (Major Performance Feature)

The L1 layer (2048→16) uses **sparse matmul**: since CReLU zeros out many inputs, Berserk detects which int8 groups are nonzero (NNZ = non-zero detection) and only processes those.

**Chunk size = 4**: The input is treated as groups of 4 int8 values (cast to int32). Only groups where at least one value is nonzero are processed.

```c
// evaluate.h:23
#define SPARSE_CHUNK_SIZE 4
```

**NNZ Detection** (`evaluate.c:146-182`):
1. Cast the int8 input to int32 (groups of 4 bytes)
2. Use SIMD `cmpgt` against zero to get a bitmask of nonzero chunks
3. Use a precomputed lookup table (`LOOKUP_INDICES[256][8]`) to expand the bitmask into indices
4. Process only the nonzero chunks in the sparse matmul

```c
// AVX2 NNZ detection
INLINE uint32_t NNZ(__m256i chunk) {
  return _mm256_movemask_ps(_mm256_castsi256_ps(
    _mm256_cmpgt_epi32(chunk, _mm256_setzero_si256())));
}
```

**Sparse matmul** (`L1AffineReLU`, line 280-322, AVX2):
- Process nonzero chunks in pairs for efficiency (`dpbusd_epi32x2`)
- `VPMADDUBSW` + `VPMADDWD` chain: multiply uint8 × int8, accumulate to int32
- Output is converted to float with ReLU: `_mm256_cvtepi32_ps(_mm256_max_epi32(...))`

**Weight scrambling** (`WeightIdxScrambled`, line 673-676): L1 weights are rearranged at load time so that the sparse chunk layout matches SIMD register layout. This avoids scatter/gather during inference.

**Coda comparison:** This is a MAJOR optimization we don't have. With CReLU, ~50-70% of accumulator values are zero, so sparse L1 cuts the matmul work roughly in half. This requires:
1. int8 accumulator output (CReLU, shift+pack)
2. NNZ detection infrastructure
3. Scrambled weight layout
4. Sparse matmul kernel

With SCReLU, sparsity is lower (squared values are all positive for positive inputs, zero only for truly zero inputs). We'd need to measure our actual sparsity ratio to know if this optimization pays off for SCReLU.

### L2/L3 (Float Inference)

L2 (16→32) and L3 (32→1) are entirely float32. This avoids integer truncation issues in narrow layers.

```c
float L2_WEIGHTS[N_L2 * N_L3] ALIGN;
float L2_BIASES[N_L3] ALIGN;
float OUTPUT_WEIGHTS[N_L3 * N_OUTPUT] ALIGN;
float OUTPUT_BIAS;
```

L2 uses FMA (`_mm256_fmadd_ps`) with horizontal add reduction (`_mm256_hadd_ps`). The output uses FMA dot product with horizontal reduction.

**Final scaling:** `Propagate` returns `L3Transform(x2) / 32`.

**Coda comparison:** We use integer for L2→output. Float is simpler and avoids precision loss. The 16→32 matmul is tiny (512 multiplies) so the float overhead is negligible vs. the complexity of integer quantization.

### King Bucketing

16 king buckets with file mirroring (left/right halves):

```c
KING_BUCKETS[64] = {15, 15, 14, 14, 14, 14, 15, 15,
                    15, 15, 14, 14, 14, 14, 15, 15,
                    13, 13, 12, 12, 12, 12, 13, 13,
                    ...
                    3,  2,  1,  0,  0,  1,  2,  3};
```

Mirror index: `N_KING_BUCKETS * (File(kingSq) > 3)` — doubles the refresh table for left/right king placement.

Feature index: `KING_BUCKETS[oK] * 12 * 64 + oP * 64 + oSq` with horizontal flip when king is on queenside (`7 * !(kingSq & 4)`).

Refresh table dimensions: `2 (color) × 2 (mirror) × 16 (king buckets) = 64` entries.

### Finny Table (RefreshAccumulator)

Same approach as ours: per-perspective cache of accumulated weights + piece bitboards. On king bucket change, diff cached vs current piece bitboards and apply only changed features.

```c
// accumulator.c:40-75
void RefreshAccumulator(Accumulator* dest, Board* board, const int perspective) {
  // diff against cached state
  for (int pc = WHITE_PAWN; pc <= BLACK_KING; pc++) {
    BitBoard rem = prev & ~curr;
    BitBoard add = curr & ~prev;
    // build delta from diffs
  }
  ApplyDelta(state->values, state->values, delta);
  memcpy(dest->values[perspective], state->values, sizeof(acc_t) * N_HIDDEN);
}
```

### Lazy Accumulator Updates

`CanEfficientlyUpdate` walks backward through the accumulator stack checking if the own king moved to a different bucket. If not, `ApplyLazyUpdates` chains incremental updates from the last correct state.

Specialized update functions for common cases:
- `ApplySubAdd` — quiet move (1 sub, 1 add)
- `ApplySubSubAdd` — capture (2 subs, 1 add)
- `ApplySubSubAddAdd` — castling (2 subs, 2 adds)

**Coda comparison:** We have similar lazy updates but use a single generic delta function. Berserk's specialized 2/3/4-delta functions avoid loop overhead and keep values in registers. Worth adopting.

### Weight Permutation for SIMD

At load time (`CopyData`), input weights and biases are permuted for `_mm256_packs_epi16` ordering. The packs instruction interleaves lanes in a non-intuitive way (low 128 bits from each input, then high 128 bits). Berserk pre-swaps the data so the natural sequential access produces correct results after packing.

```c
// evaluate.c:754-777 (AVX2)
for (size_t i = 0; i < WEIGHT_CHUNKS; i += 2) {
  __m128i a1 = _mm256_extracti128_si256(weights[i], 1);
  __m128i b0 = _mm256_extracti128_si256(weights[i + 1], 0);
  weights[i]     = _mm256_inserti128_si256(weights[i], b0, 1);
  weights[i + 1] = _mm256_inserti128_si256(weights[i + 1], a1, 0);
}
```

**Coda comparison:** We don't do this pre-permutation. If we use `_mm256_packs_epi16` in our InputReLU, we need the same weight permutation or we'll get scrambled outputs.

---

## 2. Search

### Negamax Structure

Standard PVS with aspiration windows. Key parameters:

| Feature | Berserk Value | GoChess Value | Notes |
|---------|--------------|---------------|-------|
| Aspiration delta | 9 | 15 | Tighter initial window |
| Delta growth | ×1.266 (+17/64) | — | Grows 26.6% per fail |
| LMR formula | `ln(d)*ln(m)/2.0385 + 0.2429` | `ln(d)*ln(m)/C + ...` | C=2.04 vs our 1.30 |
| RFP depth | ≤9 | — | |
| RFP margin | `70*d - 118*improving` | — | |
| Razoring depth | ≤5, margin `214*d` | — | |
| NMP R | `4 + 385*d/1024 + min(10*(eval-beta)/1024, 4)` | — | |
| ProbCut beta | `beta + 172` | — | |
| LMP improving | `2.19 + 0.99*d²` | — | |
| LMP not improving | `1.31 + 0.35*d²` | — | |
| Futility | `81 + 46*lmrDepth` | — | |

### Singular Extensions (Full Implementation)

Berserk has a sophisticated singular extension system (`search.c:663-694`):

```c
if (depth >= 6 && move == hashMove && ttDepth >= depth - 3 && (ttBound & BOUND_LOWER)) {
  int sBeta  = Max(ttScore - 5 * depth / 8, -CHECKMATE);
  int sDepth = (depth - 1) / 2;

  ss->skip = move;
  score = Negamax(sBeta - 1, sBeta, sDepth, cutnode, thread, pv, ss);
  ss->skip = NULL_MOVE;

  if (score < sBeta) {
    // Double/triple extension for deeply singular moves
    if (!isPV && score < sBeta - 48 && ss->de <= 6 && !IsCap(move)) {
      extension = 3;          // triple extension!
      ss->de = (ss-1)->de + 1;
    } else if (!isPV && score < sBeta - 14 && ss->de <= 6) {
      extension = 2;          // double extension
      ss->de = (ss-1)->de + 1;
    } else {
      extension = 1;          // single extension
    }
  } else if (sBeta >= beta)
    return sBeta;              // multicut
  else if (ttScore >= beta)
    extension = -2 + isPV;     // negative extension (reduction)
  else if (cutnode)
    extension = -2;
  else if (ttScore <= alpha)
    extension = -1;
}
```

Key detail: `ss->de` tracks "double extension" depth to cap triple/double extensions at 6 levels.

**Coda comparison:** We removed singular extensions because they hurt cross-engine. Berserk (as a top-10 engine) benefits because their eval is strong enough that the verification search is rarely wrong. For weaker engines, singular extension's verification search wastes nodes on eval-biased positions.

### IIR (Internal Iterative Reduction)

```c
if ((isPV || cutnode) && depth >= 4 && !hashMove)
  depth--;
```

Reduces by 1 when no hash move found in PV or cut nodes at depth >= 4.

### Cycle Detection

```c
if (!isRoot && board->fmr >= 3 && alpha < 0 && HasCycle(board, ss->ply)) {
  alpha = 2 - (thread->nodes & 0x3);  // randomized draw score
  if (alpha >= beta) return alpha;
}
```

Uses Cuckoo hashing (like Stockfish) to detect upcoming repetitions. Only applied when alpha < 0 (losing side benefits from detecting draws). Randomized draw value prevents search instability.

**Coda comparison:** We don't have cycle detection. This is eval-agnostic (it detects forced repetitions regardless of eval quality) and transfers well cross-engine. Worth implementing.

### Alpha-Reduce (Depth Reduction on Alpha Raise)

```c
if (score > alpha) {
  bestMove = move;
  alpha = score;
  if (alpha < beta && score > -TB_WIN_BOUND)
    depth -= (depth >= 2 && depth <= 11);
}
```

When alpha is raised in PV nodes, reduce remaining search depth by 1 (for depth 2-11). This is exactly our "alpha-reduce" feature.

### Deeper/Shallower Re-Search

After LMR re-search:
```c
newDepth += (score > bestScore + 69);   // do deeper if much better
newDepth -= (score < bestScore + newDepth);  // do shallower if not great
```

### QSearch Beta Blending

```c
if (bestScore >= beta && abs(bestScore) < TB_WIN_BOUND)
  bestScore = (bestScore + beta) / 2;
```

Identical to our beta blending in quiescence.

### History Bonus on LMR Re-Search

```c
int bonus = score <= alpha ? -HistoryBonus(newDepth - 1)
          : score >= beta  ?  HistoryBonus(newDepth - 1) : 0;
UpdateCH(ss, move, bonus);
```

Updates continuation history based on whether the re-search confirmed or denied the LMR result.

### Opponent Easy Captures (Novel)

Used to gate RFP and NMP:

```c
INLINE BitBoard OpponentsEasyCaptures(Board* board) {
  // Our queens threatened by rook+ attacks
  // Our rooks/queens threatened by minor+ attacks
  // Our minors+ threatened by pawn attacks
  return (queens & rookThreats) | (rooks & minorThreats) | (minors & pawnThreats);
}
```

If the opponent has easy captures, RFP gets a stricter margin (-118 improving bonus removed) and NMP is skipped entirely. This prevents over-pruning when we have hanging pieces.

**Coda comparison:** This is a simple, eval-agnostic heuristic. When the opponent has obvious captures of our pieces, our static eval is unreliable, so we should be more cautious about pruning. Worth adopting.

### FMR Eval Adjustment

```c
INLINE int AdjustEvalOnFMR(Board* board, int eval) {
  return (200 - board->fmr) * eval / 200;
}
```

Scales eval toward 0 as the 50-move rule counter increases. At fmr=100 (50 moves), eval is halved.

### Pawn Correction History

Single pawn-hash based correction with simple EMA update:

```c
#define PAWN_CORRECTION_SIZE  131072  // 128K entries
#define PAWN_CORRECTION_GRAIN 256

int GetPawnCorrection(Board* board, ThreadData* thread) {
  return thread->pawnCorrection[board->pawnZobrist & PAWN_CORRECTION_MASK] / PAWN_CORRECTION_GRAIN;
}

// Update: exponential moving average (255/256 old + 1/256 new)
thread->pawnCorrection[idx] = (thread->pawnCorrection[idx] * 255 + correction) / 256;
```

Applied at half weight: `eval = rawEval + GetPawnCorrection(...) / 2`

Update condition: only when bestMove is quiet and bound matches direction.

**Coda comparison:** We have pawn correction history. Berserk's is simpler (single table, no material/non-pawn corrections). Modern SF has 6+ correction history tables. Single pawn correction is the highest-value one.

### History-Gated Pruning

RFP is skipped if the hash move has poor history:
```c
eval - 70 * depth + 118 * (improving && !opponentHasEasyCapture) >= beta &&
(!hashMove || GetHistory(ss, thread, hashMove) > 11800)
```

**Coda comparison:** We don't gate RFP on hash move history. This prevents RFP from cutting off when the "best" move is historically bad.

### Quiet History Pruning

```c
if (!killerOrCounter && lmrDepth < 5 && history < -2788 * (depth - 1)) {
  skipQuiets = 1;
  continue;
}
```

Aggressively skips remaining quiets when history is deeply negative.

---

## 3. Move Ordering

### History Tables

| Table | Dimensions | Index | Notes |
|-------|-----------|-------|-------|
| HH (main history) | `[2][2][2][64*64]` | `[stm][!threatened_from][!threatened_to][from_to]` | **Threat-aware**: 4× the usual butterfly table |
| CH (continuation) | `[2][12][64][12][64]` | `[cap][piece][to] → [piece][to]` | 1-ply, 2-ply, 4-ply, 6-ply lookback |
| CaptureH | `[12][64][2][7]` | `[piece][to][defended][captured_type]` | Defended = not on threatened square |
| Counters | `[12][64]` | `[prev_piece][prev_to]` | Standard counter-move |
| Killers | 2 per ply | Standard | |

### Threat-Aware History (Key Feature)

The main history table is indexed by whether the from/to squares are threatened:

```c
#define HH(stm, m, threats) (thread->hh[stm][!GetBit(threats, From(m))][!GetBit(threats, To(m))][FromTo(m)])
```

This creates 4 sub-tables per side:
- Move FROM safe, TO safe
- Move FROM safe, TO threatened
- Move FROM threatened, TO safe (escaping!)
- Move FROM threatened, TO threatened

**Coda comparison:** We have `[stm][from_to]` history (2 sub-tables). Berserk's 4x threat-aware indexing separates "escaping from threat" moves from normal moves, which is eval-agnostic information. This transfers well cross-engine. The `threatened` bitboard is computed once per node anyway (for legality). Adding `!GetBit(threats, From/To)` indexing is simple.

### Continuation History Depths

CH lookback: 1-ply, 2-ply, 4-ply, 6-ply (for scoring)
CH update: 1-ply, 2-ply, 4-ply, 6-ply (for bonus/malus)

```c
// Scoring (movepick.c:64-68)
current->score = HH * 2 +
                 (*(ss-1)->ch)[pc][to] * 2 +   // 1-ply
                 (*(ss-2)->ch)[pc][to] * 2 +   // 2-ply
                 (*(ss-4)->ch)[pc][to] +        // 4-ply (half weight)
                 (*(ss-6)->ch)[pc][to];         // 6-ply (half weight)
```

Note the weighting: recent plies get 2× weight, distant plies get 1×.

**Coda comparison:** We use 1-ply and 2-ply CH. Adding 4-ply and 6-ply CH is pure information gain and should transfer well cross-engine.

### Capture History with Defended Flag

```c
#define TH(p, e, d, c) (thread->caph[p][e][d][c])
```

Indexed by: piece, to-square, **whether target is defended** (not on opponent's threat map), captured piece type. This separates captures of defended vs undefended pieces — a key tactical distinction.

### Quiet Move Scoring with Threat Awareness

In `ScoreMoves`, quiet moves get a bonus/penalty based on piece type and threat level:

```c
if (pt != PAWN && pt != KING) {
  const BitBoard danger = threats[Max(0, pt - BISHOP)];
  if (GetBit(danger, from))  current->score += 16384;  // escaping threat
  if (GetBit(danger, to))    current->score -= 16384;  // moving into threat
}
```

Where `threats[0]` = pawn threats, `threats[1]` = pawn+minor threats, `threats[2]` = pawn+minor+rook threats. The danger level matches piece value: bishops/knights fear pawn attacks, rooks fear minor attacks, queens fear rook attacks.

**Coda comparison:** We don't have this piece-type-aware threat scoring in move ordering. This is a clean, eval-agnostic signal worth adding.

### History Bonus Formula

```c
INLINE int16_t HistoryBonus(int depth) {
  return Min(1729, 4 * depth * depth + 164 * depth - 113);
}

INLINE void AddHistoryHeuristic(int16_t* entry, int16_t inc) {
  *entry += inc - *entry * abs(inc) / 16384;
}
```

Gravity-based update: entries naturally decay toward 0. Max bonus = 1729, gravity divisor = 16384.

### Conditional History Update

Only increases best move history when the search was non-trivial:
```c
if (nQ > 1 || depth > 4) {
  AddHistoryHeuristic(&HH(...), inc);
  UpdateCH(ss, bestMove, inc);
}
```

**Coda comparison:** We always update history for the best move. Berserk's conditional update prevents inflating history for trivially-won positions.

---

## 4. Performance / SIMD

### SIMD Tiers

| Tier | Accumulator Update | InputReLU | L1 Sparse | L2 Float | L3 Float |
|------|-------------------|-----------|-----------|----------|----------|
| AVX-512 | 512-bit int16 | 512-bit pack+clamp | 512-bit DPBUSD | 512-bit FMA | 512-bit FMA |
| AVX2 | 256-bit int16 | 256-bit pack+clamp | 256-bit MADDUBS | 256-bit FMA | 256-bit FMA |
| SSSE3 | 128-bit int16 | 128-bit pack+clamp | 128-bit MADDUBS | 128-bit mul+add | 128-bit mul+add |
| SSE2 | 128-bit int16 | 128-bit pack+clamp | scalar sparse | scalar | scalar |
| Scalar | scalar | scalar | scalar | scalar | scalar |

### Key SIMD Patterns

**DPBUSD emulation** (the core int8 matmul):
```c
// VPMADDUBSW: uint8 × int8 → int16, pairwise add
// VPMADDWD: int16 × 1 → int32, pairwise add (horizontal reduction)
__m256i p0 = _mm256_maddubs_epi16(a, b);
p0         = _mm256_madd_epi16(p0, _mm256_set1_epi16(1));
*acc       = _mm256_add_epi32(*acc, p0);
```

**Pair processing**: Sparse matmul processes two nonzero chunks at once, combining their MADDUBS results before the MADD reduction. This halves the horizontal add overhead.

**Lookup table for NNZ expansion**: `LOOKUP_INDICES[256][8]` — for each byte pattern, stores the indices of set bits. This converts a bitmask to a dense index array without branching.

---

## 5. Lazy SMP

### Thread Pool Design

Berserk uses the "sleeping threads" pattern from CFish:

```c
typedef struct {
  ThreadData* threads[256];    // up to 256 threads
  int count;
  pthread_mutex_t mutex, lock;
  pthread_cond_t sleep;
  uint8_t init, searching, sleeping, stopOnPonderHit;
  atomic_uchar ponder, stop;   // atomic stop flag
} ThreadPool;
```

### Shared State

Only the TT is shared (global `TTTable TT`). Each thread has its own:
- `Board` (copied from main at search start)
- `Accumulator*` stack (allocated per thread)
- `AccumulatorKingState*` refresh table (allocated per thread)
- All history tables (HH, CH, CaptureH, counters, killers)
- Pawn correction history
- Root moves list

### Best Thread Selection (Vote System)

After search, all threads vote for the best move:

```c
int ThreadValue(ThreadData* thread, const int worstScore) {
  return (thread->rootMoves[0].score - worstScore) * thread->depth;
}
```

Voting weight = (score - worst_score) × depth. The move with highest total vote wins. Tiebreaker: PV length > 2 (prefer moves with deeper PV).

**Coda comparison:** Similar to our approach. The PV-length tiebreaker is interesting — it prefers threads that found a real PV over those with a fail-high-only result.

### Stop Mechanism

- `Threads.stop` is `atomic_uchar`, checked with `LoadRlx` every N nodes
- On stop, thread does `longjmp(thread->exit, 1)` for hot exit
- `thread->calls` decrements and resets to `Limits.hitrate` for batched time checks

### Thread Communication Pattern

1. Main thread wakes all helper threads with `THREAD_SEARCH`
2. All threads run `Search()` independently
3. Main thread finishes, sets `Threads.stop = 1`
4. Main thread waits for all helpers to sleep
5. Vote on best move across all threads

---

## 6. Transposition Table

### Entry Format

10 bytes per entry, 3 entries per bucket (30 bytes + 2 padding = 32 bytes):

```c
typedef struct __attribute__((packed)) {
  uint16_t hash;         // 2 bytes: verification hash
  uint8_t depth;         // 1 byte: search depth (offset by DEPTH_OFFSET=-2)
  uint8_t agePvBound;    // 1 byte: age(5 bits) | pv(1 bit) | bound(2 bits)
  uint32_t evalAndMove;  // 4 bytes: eval(12 bits) | move(20 bits)
  int16_t score;         // 2 bytes
} TTEntry;               // total: 10 bytes
```

**Move is 20 bits** (not 16 like ours). Eval is packed into 12 bits (offset by 2048).

**Bucket size = 3** (vs our 4). Each bucket is exactly 32 bytes for cache line alignment.

### Replacement Policy

Age-depth hybrid: `depth - (age_diff)/2`. Prefer replacing old, shallow entries.

### TT Cutoff

Non-PV TT cutoff has a `cutnode` condition:
```c
if (!isPV && ttScore != UNKNOWN && ttDepth >= depth &&
    (cutnode || ttScore <= alpha) &&
    (ttBound & (ttScore >= beta ? BOUND_LOWER : BOUND_UPPER)))
  return ttScore;
```

The `(cutnode || ttScore <= alpha)` allows TT cutoffs at cut nodes even when ttScore > alpha, which is more aggressive than standard.

**Coda comparison:** We don't have the `cutnode` condition on TT cutoffs. This is a structural optimization that doesn't depend on eval quality.

---

## 7. Novel/Notable Features Summary

### High Priority (Should Adopt)

1. **Threat-aware history** (`hh[stm][!threat_from][!threat_to][from_to]`) — 4× the standard butterfly table, separates escaping-threat moves from normal moves. Pure information gain, eval-agnostic. `board.h:26`, `movepick.c:64`.

2. **Opponent easy captures** — gates RFP and NMP when opponent has obvious hanging-piece captures. Prevents over-pruning when static eval is unreliable. `board.h:67-78`, `search.c:513`.

3. **Sparse L1 matmul with NNZ detection** — only processes nonzero accumulator groups in the FT→L1 matmul. With CReLU, ~50-70% are zero. Requires int8 L1 weights. `evaluate.c:184-322`.

4. **4-ply and 6-ply continuation history** — extends our 2-ply lookback to 4-ply and 6-ply with weighted scoring. `history.h:31-33`, `movepick.c:64-68`.

5. **Cycle detection** (Cuckoo hashing) — detects upcoming repetitions without playing them. Eval-agnostic, transfers perfectly. `board.c:733`, `search.c:383`.

6. **Float L2/output inference** — avoids integer quantization precision loss in narrow hidden layers. `evaluate.c:43-47`.

### Medium Priority (Consider)

7. **Conditional history update** — only update best move history when search was non-trivial (nQ > 1 or depth > 4). `history.c:50-53`.

8. **Capture history with defended flag** — separates captures of defended vs undefended pieces. `types.h:198`, `history.h:27`.

9. **FMR eval adjustment** — scale eval toward 0 as 50-move counter increases. `search.c:81-83`.

10. **History-gated RFP** — skip RFP when hash move has poor history. `search.c:519`.

11. **Cutnode TT cutoff** — allow TT cutoffs at cut nodes even when ttScore > alpha. `search.c:433`.

12. **Threat-aware quiet scoring** — bonus for escaping threat, penalty for moving into threat, scaled by piece type. `movepick.c:70-77`.

13. **Weight permutation for packs** — pre-permute weights for `_mm256_packs_epi16` lane behavior. `evaluate.c:754-777`.

### Low Priority (Reference)

14. **Specialized update functions** (SubAdd, SubSubAdd, SubSubAddAdd) — avoid loop overhead for common delta patterns. `accumulator.h:101-195`.

15. **LMR re-search history update** — update CH with bonus/malus based on re-search result. `search.c:747-748`.

16. **Aspiration delta = 9** with 1.266× growth — tighter than our delta=15. `search.c:263-303`.
