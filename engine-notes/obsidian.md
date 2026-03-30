# Obsidian Chess Engine - Technical Review

Source: `~/chess/engines/Obsidian/`
Author: gab8192
Version: dev-16.15
NNUE: (768x13 -> 1536)x2 -> 16 -> 32 -> 1x8 (pairwise mul, dual-activation L2, NNZ-sparse L1, Finny table)

---

## 1. NNUE Architecture

### Network: (768x13 -> 1536) x2 -> 16 -> 32 -> 1 x8

**nnue.h:25-29:**
```cpp
constexpr int FeaturesWidth = 768;
constexpr int L1 = 1536;
constexpr int L2 = 16;
constexpr int L3 = 32;
```

**Feature Transformer (FT):**
- Input: HalfKA -- 768 features (12 piece types x 64 squares) per king bucket
- 13 king buckets (mirrored horizontally when king on files e-h) `nnue.h:30-40`
- FT width: 1536 (int16)
- Weight layout: `int16_t FeatureWeights[KingBuckets][2][6][64][L1]` (nnue.cpp:24) -- [bucket][enemy?][piece_type-1][square][1536]
- King-side mirroring: when `kingSq & 0b100` (files e-h), feature square is XORed with 7 `nnue.cpp:57`

**Quantization:**
- QA = 255, QB = 128 `nnue.h:46-47`
- NetworkScale = 400 `nnue.h:44`
- FT: int16 (QA=255)
- L1 weights: **int8** `nnue.cpp:28` -- this is key, enables dpbusd/maddubs
- L1 biases: **float** `nnue.cpp:31`
- L2 weights: **float** `nnue.cpp:33`
- L2 biases: **float** `nnue.cpp:34`
- L3 weights: **float** `nnue.cpp:36`
- L3 biases: **float** `nnue.cpp:37`

**Coda comparison:** We use int16 L1 weights with QA=255. Obsidian uses int8 L1 weights, enabling the much faster dpbusd/maddubs kernel. This is a massive NPS difference. All layers after L1 are float -- no quantization issues for narrow layers.

### FT Activation: Pairwise Multiplication (not CReLU/SCReLU)

**nnue.cpp:242-257** -- The FT activation is a pairwise product:
```cpp
for (int i = 0; i < L1 / 2; i += I8InVec) {
    VecI c0 = minEpi16(maxEpi16(AsVecI(acc[i]), veciZero), veciOne);        // clamp(acc[i], 0, QA)
    VecI c1 = minEpi16(AsVecI(acc[i + L1/2]), veciOne);                     // min(acc[i+L1/2], QA)  -- NO clamp to 0!
    ...
    VecI cProd = mulhiEpi16(slliEpi16(c0, 16 - FtShift), c1);              // (c0 << 7) * c1 >> 16
    VecI packed = packusEpi16(cProd, dProd);                                 // pack to uint8
```

This is **pairwise multiplication**: the 1536-element accumulator is split in half (768+768). First half is clamped to [0, QA], second half is clamped to [-inf, QA] (note: NOT clamped to 0 on the low end). The product is taken, shifted, and packed to uint8 for the L1 matmul.

Key detail: `FtShift = 9` (nnue.cpp:21). The shift `slliEpi16(c0, 16 - 9)` = shift left by 7, then `mulhiEpi16` takes the high 16 bits, effectively computing `(c0 * c1) >> 9`.

Output: 768 uint8 values (half the FT width, because pairwise halves it), then concatenated for both perspectives = 1536 uint8 total for L1 input.

**Coda comparison:** We use CReLU (clamp + concat perspectives). Obsidian's pairwise approach is more expressive (multiplicative interaction within features). This is the same technique used by Stockfish. We should investigate pairwise for our next net architecture.

### NNZ (Non-Zero) Sparse L1 Matmul

This is the most important performance feature. **nnue.cpp:260-271** (AVX2 path):
```cpp
uint16_t nnzIndexes[L1 / 4];   // indices of non-zero 4-byte chunks
int nnzCount = 0;
...
uint16_t nnzMask = getNnzMask(packed);  // bitmask: which int32s are >0
for (int lookup = 0; lookup < FloatInVec; lookup += 8) {
    uint8_t slice = (nnzMask >> lookup) & 0xFF;
    __m128i indexes = _mm_loadu_si128((__m128i*)nnzTable[slice]);
    _mm_storeu_si128((__m128i*)(nnzIndexes + nnzCount), _mm_add_epi16(base, indexes));
    nnzCount += BitCount(slice);
    base = _mm_add_epi16(base, lookupInc);
}
```

The NNZ table (`nnzTable[256][8]`) maps each byte to the positions of its set bits. After FT activation + packus, we get uint8 outputs. `getNnzMask` checks which int32 chunks are non-zero (using `_mm256_movemask_ps` on AVX2, or `_mm512_cmpgt_epi32_mask` on AVX-512).

Then L1 matmul only iterates over non-zero chunks **nnue.cpp:290-302**:
```cpp
for (int i = 0; i < nnzCount; i++) {
    int l1in = nnzIndexes[i] * 4;
    VecI vecFtOut = set1Epi32(*(uint32_t*)(ftOut + l1in));  // broadcast 4 uint8s
    for (int j = 0; j < L2; j += FloatInVec) {
        VecI vecWeight = AsVecI(Weights->L1Weights[bucket][l1in + j/4]);
        AsVecI(sums[j]) = dpbusdEpi32(AsVecI(sums[j]), vecFtOut, vecWeight);
    }
}
```

This uses `dpbusdEpi32` (nnue.cpp emulates it via `maddubsEpi16` + `maddEpi16` + `addEpi32` in simd.h:204-207). Each iteration processes 4 uint8 FT outputs against 4 int8 weights, accumulated into int32.

**dpbusd weight preprocessing** (nnue.cpp:170-176): On load, L1 weights are transposed so that 4 consecutive FT inputs for each L2 neuron are contiguous, enabling the vectorized dpbusd pattern.

**Coda comparison:** This is a huge win. With L1=16 output neurons:
- Dense: 1536 * 16 = 24576 int8 multiplies per perspective
- Sparse (typical ~50% NNZ): ~12288 multiplies, but with dpbusd's 4-way accumulation it's ~3072 dpbusd ops
- The sparsity comes naturally from pairwise mul (products of near-zero values are zero)

We don't have NNZ sparse matmul. This is the #1 actionable item for NPS improvement.

### L1 -> L2 Propagation (float, dual activation)

**nnue.cpp:304-311:**
```cpp
for (int i = 0; i < L2; i += FloatInVec) {
    VecF vecBias = AsVecF(Weights->L1Biases[bucket][i]);
    VecF prod = mulAddPs(castEpi32ToPs(AsVecI(sums[i])), L1MulVec, vecBias);
    VecF squared = mulPs(prod, prod);
    AsVecF(l1Out[i]) = minPs(maxPs(prod, vecfZero), vecfOne);          // CReLU: clamp [0,1]
    AsVecF(l1Out[i + L2]) = minPs(squared, vecfOne);                    // Squared: clamp [0,1]
}
```

The L1 output has a **dual activation**: both `clamp(x, 0, 1)` (ReLU) and `clamp(x^2, 0, 1)` (squared). This produces `L2 * 2 = 32` float values fed into L2. The scale factor `L1Mul = 1.0 / (QA * QA * QB >> FtShift)`.

**Coda comparison:** We use SCReLU for the hidden layer. Obsidian's dual activation (linear ReLU + squared) for L1 output is a different approach -- it feeds both the value and its square as separate inputs to L2, giving the next layer access to both. This is more parameter-efficient than pure SCReLU since it doubles the effective L2 input width without doubling L1 neurons.

### L2 -> L3 and L3 -> Output (float, standard)

**nnue.cpp:316-328:** L2 is a simple float matmul with ReLU clamp [0,1].
**nnue.cpp:330-344:** L3 is dot product + bias, then `* NetworkScale (400)`.

Both use FMA (`mulAddPs`) for performance.

### Finny Table

**nnue.h:73-81:**
```cpp
struct FinnyEntry {
    Bitboard byColorBB[COLOR_NB][COLOR_NB];
    Bitboard byPieceBB[COLOR_NB][PIECE_TYPE_NB];
    Accumulator acc;
};
using FinnyTable = FinnyEntry[2][KingBuckets];  // [mirror][bucket]
```

Standard Finny table indexed by `[fileOf(king) >= FILE_E][bucket]`. Refresh diffs cached vs current piece bitboards (search.cpp:270-296). Optimized with movePiece when both add and remove exist for same piece type.

**Coda comparison:** Very similar to our implementation. They pair add/remove into movePiece which reduces ops slightly.

### Output Buckets

8 output buckets, selected by piece count: `bucket = (popcount(pieces) - 2) / divisor` where `divisor = (32 + 8 - 1) / 8 = 4` (nnue.cpp:218-219).

---

## 2. Search

### LMR Formula

**search.cpp:119-131:**
```cpp
double dBase = LmrBase / 100.0;  // 99/100 = 0.99
double dDiv = LmrDiv / 100.0;    // 314/100 = 3.14
lmrTable[i][m] = dBase + log(i) * log(m) / dDiv;
```

LMR = 0.99 + ln(depth) * ln(moveIndex) / 3.14

**LMR adjustments** (search.cpp:1069-1091):
- `-= history / (quiet ? 9621 : 5693)` -- history-based
- `-= complexity / 120` -- correction history complexity
- `-= (newPos.checkers != 0)` -- gives check
- `-= (ttDepth >= depth)` -- deep TT hit
- `-= ttPV + IsPV` -- PV nodes
- `+= ttMoveNoisy` -- TT move was noisy
- `+= !improving` -- not improving
- `+= cutNode ? (2 - ttPV) : 0` -- cut nodes get +2 (or +1 if ttPV)

**Coda comparison:** Similar to ours. Notable: they use `complexity` (abs difference between raw and adjusted eval) as an LMR adjustment -- positions where correction history adjusts eval a lot get less reduction. We don't have this.

### Razoring

**search.cpp:854-859:** `eval < alpha - 352 * depth` -> qsearch, return if <= alpha. Guarded by `alpha < 2000`.

### Reverse Futility Pruning (RFP)

**search.cpp:864-868:** `depth <= 11`, margin = `max(87 * (depth - improving), 22)`. Returns `(eval + beta) / 2` (blended).

**Coda comparison:** They return `(eval + beta) / 2` -- same beta blending we use.

### Null Move Pruning (NMP)

**search.cpp:872-891:**
- Guard: `cutNode && !excludedMove && (ss-1)->playedMove != MOVE_NONE && eval >= beta && staticEval + 22*depth - 208 >= beta`
- Reduction: `R = min((eval-beta)/147, 4) + depth/3 + 4 + ttMoveNoisy`
- Returns: `score < TB_WIN ? score : beta`

Notable: NMP only fires on **cutNode**. This is unusual -- most engines allow NMP on all non-PV nodes. Guard includes `staticEval + 22*depth - 208 >= beta` (depth-dependent eval guard).

**Coda comparison:** We don't restrict NMP to cutNode. Worth investigating. Their NMP formula uses `ttMoveNoisy` as a reduction boost (noisy TT move = opponent has forcing moves, so null move search should be deeper).

### Internal Iterative Reduction (IIR)

**search.cpp:894-895:** `(IsPV || cutNode) && depth >= 2+2*cutNode && !ttMove` -> `depth--`.

IIR depth threshold: 2 for PV, 4 for cutNode.

### ProbCut

**search.cpp:897-939:** Standard ProbCut with `beta + 190` margin. Uses SEE margin `(probcutBeta - staticEval) * 10 / 16`. Depth >= 5 guard.

### SEE Pruning

**search.cpp:1000-1004:**
- Quiet: `PvsQuietSeeMargin * lmrDepth^2` = `-21 * lmrDepth^2`
- Capture: `PvsCapSeeMargin * depth` = `-96 * depth`

Where `lmrDepth = max(0, depth - lmrRed)` includes early LMR history adjustment.

### History Pruning

**search.cpp:1006-1008:** `isQuiet && history < -7471 * depth` -> skip all remaining quiets.

### Late Move Pruning (LMP)

**search.cpp:1012-1013:** `seenMoves >= (depth*depth + 3) / (2 - improving)`

### Futility Pruning

**search.cpp:1017-1023:** `lmrDepth <= 10 && staticEval + 159 + 153 * lmrDepth <= alpha` -> skip quiets. Guarded by `quietCount >= 1` (at least one quiet already seen).

### Singular Extensions

**search.cpp:1029-1058:**
- Guard: `depth >= 5 && move == ttMove && |ttScore| < TB_WIN && ttBound & LOWER && ttDepth >= depth - 3`
- sBeta = `ttScore - (depth * 64) / 64` = `ttScore - depth`
- Search: `negamax(sBeta-1, sBeta, (depth-1)/2, cutNode, ...)`
- Single extension if `seScore < sBeta`
- Double extension if `!IsPV && seScore < sBeta - 13`
- Triple extension if `isQuiet && seScore < sBeta - 121`
- Multicut if `sBeta >= beta`
- Negative extensions: `-3 + IsPV` if `ttScore >= beta`, `-2` if cutNode

**Coda comparison:** We don't have singular extensions (deliberately removed because they hurt cross-engine). Obsidian uses them aggressively with triple extensions. This is typical for top engines in self-play but may be eval-dependent.

### LMR Re-search Adjustments

**search.cpp:1095-1106:** After LMR re-search:
- Deeper if `score > bestScore + 43 + 2*newDepth`
- Shallower if `score < bestScore + 11`
- Bonus cont history adjustment on re-search

### Aspiration Windows

**search.cpp:1378-1425:**
- Start depth: 4
- Initial delta: `6 + avgScore^2 / 13000` (score-dependent)
- On fail low: `beta = (alpha + beta) / 2; alpha = max(-INF, score - window)`
- On fail high: `beta = min(INF, score + window); failHighCount++` (only if score < 2000)
- Window widening: `window += window / 3`
- Depth reduction on fail high: `max(1, rootDepth - failHighCount)`

**Coda comparison:** Score-dependent initial window is interesting. The `avgScore^2 / 13000` term means volatile positions get wider windows.

### QSearch

**nnue.cpp N/A, search.cpp:484-634:**
- Standing pat with beta blending: `return (bestScore + beta) / 2` on initial stand-pat cutoff
- Final beta blending: `bestScore = (bestScore + beta) / 2` if `bestScore >= beta`
- Quiet check generation at depth 0 only
- FP margin: `staticEval + 156`
- SEE margin: -32
- In-check evasion: no stand pat, full move gen
- Limit: 3 non-check captures, or break after first quiet in check

### 50-Move Rule Zobrist

**zobrist.cpp:47-52:**
```cpp
memset(ZOBRIST_50MR, 0, sizeof(ZOBRIST_50MR));
for (int i = 14; i <= 100; i += 8) {
    uint64_t key = dis(gen);
    for (int j = 0; j < 8; j++)
        ZOBRIST_50MR[i+j] = key;
}
```

The TT key includes 50-move clock hash. Keys are zero for HMC 0-13, then change every 8 plies from 14 onward. This makes TT entries from different 50mr counts appear as different positions (avoiding stale draws). The coarse 8-ply granularity prevents excessive TT pollution.

**search.cpp:502:** `posTtKey = pos.key ^ ZOBRIST_50MR[pos.halfMoveClock]`

Additionally guarded by `pos.halfMoveClock < 90` in TT cutoff (search.cpp:749).

**Coda comparison:** We don't have 50mr-aware TT keys. This is a clean, cheap improvement to avoid draw-related TT bugs.

---

## 3. Move Ordering

### History Types

Per-thread histories (search.h:90-98):
1. **MainHistory** `[color][from*64+to]` -- butterfly history (history.h:12)
2. **PawnHistory** `[pawnKey % 1024][piece*64+to]` -- pawn-structure indexed (history.h:24)
3. **CaptureHistory** `[piece*64+to][captured_type]` (history.h:15)
4. **ContinuationHistory** `[isCap][piece*64+to][piece*64+to]` (history.h:21) -- two-ply indexed, with capture/quiet split
5. **CounterMoveHistory** `[piece*64+to] -> Move` (history.h:18)
6. **PawnCorrHist** `[pawnKey % 32768][stm]` (history.h:27)
7. **NonPawnCorrHist** x2 (white/black non-pawn keys) `[key % 32768][stm]` (history.h:29)
8. **ContCorrHist** `[piece*64+to][piece*64+to]` -- continuation correction history (history.h:32)

### Quiet Move Scoring (movepick.cpp:60-98)

Score = threatScore + mainHistory + pawnHistory + contHist[-1] + contHist[-2] + contHist[-4] + contHist[-6]

**Threat-aware ordering:** `calcThreats()` computes opponent pawn, minor, and rook attack maps. Then:
- Queen on rook-attacked square: +32768 for leaving, -32768 for entering
- Rook on minor-attacked square: +16384 for leaving, -16384 for entering
- Minor on pawn-attacked square: +16384 for leaving, -16384 for entering

**Coda comparison:** We use 4 continuation history plies (1, 2, 4, 6) -- same as Obsidian. The threat scoring is interesting: it directly adds a large bonus/penalty based on whether the move escapes or walks into a threat. We don't have this. The bonuses are large (16384-32768) -- comparable to the entire history range -- so this has a significant effect on move ordering.

### Capture Move Scoring (movepick.cpp:101-119)

Score = capturedPieceValue * 16 + promotionBonus + captureHistory

### Move Ordering Stages (movepick.h:15-35)

TT -> Good Captures (SEE >= -score/32) -> Killer -> Counter -> Quiets -> Bad Captures

Notable: Good capture SEE threshold is `-move.score / 32` (dynamic, based on capture history score). In probcut mode, uses explicit SEE margin.

### Continuation History Depth

4 plies: `(ss-1)`, `(ss-2)`, `(ss-4)`, `(ss-6)` for move ordering.
For updates: same 4 plies, but `(ss-4)` and `(ss-6)` get `bonus/2`.
For getQuietHistory (used in pruning): only 3 plies: `(ss-1)`, `(ss-2)`, `(ss-4)`.

### Eval-Based History (EvalHist)

**search.cpp:840-844:**
```cpp
if (!(ss-1)->playedCap && (ss-1)->staticEval != SCORE_NONE) {
    int theirLoss = (ss-1)->staticEval + ss->staticEval - 58;
    int bonus = clamp(-492 * theirLoss / 64, -534, 534);
    addToHistory(mainHistory[~pos.sideToMove][move_from_to((ss-1)->playedMove)], bonus);
}
```

After evaluation, retroactively adjusts the butterfly history of the opponent's last move based on how much eval changed. If the opponent's move caused a big eval drop for them (`theirLoss` is large), their move gets a penalty. If their move was strong, it gets a bonus.

**Coda comparison:** This is novel. We don't have retroactive eval-based history updates. It essentially teaches the engine about opponent move quality through the eval signal.

---

## 4. Correction History

### 4 Correction History Tables

1. **PawnCorrHist** `[pawnKey % 32768][stm]` -- pawn structure correction
2. **White NonPawnCorrHist** `[nonPawnKey[WHITE] % 32768][stm]`
3. **Black NonPawnCorrHist** `[nonPawnKey[BLACK] % 32768][stm]`
4. **ContCorrHist** `[piece*64+to][piece*64+to]` -- continuation-indexed correction

### Adjustment Formula (search.cpp:374-387)

```cpp
eval = (eval * (200 - pos.halfMoveClock)) / 200;  // 50mr scaling
eval += 30 * pawnCorrhist[key][stm] / 512;
eval += 35 * wNonPawnCorrhist[key][stm] / 512;
eval += 35 * bNonPawnCorrhist[key][stm] / 512;
eval += 27 * contCorrHist[piece_to][piece_to_prev] / 512;  // if prev move exists
```

### Update (search.cpp:1223-1239)

Only when: not in check, best move is quiet (or no best move), and TT bound confirms score direction.
Bonus: `clamp((bestScore - staticEval) * depth / 8, -256, 256)`

**Coda comparison:** We have pawn correction history. They additionally have per-color non-pawn correction and continuation correction. The per-color non-pawn keys are clever -- they track material/position corrections for each side independently. The continuation correction history is unique: it corrects eval based on the last two moves (piece-to indexing), capturing positional patterns that follow specific move sequences.

---

## 5. SIMD / Performance

### Instruction Set Support

**simd.h** provides unified wrappers for AVX-512, AVX2, and SSSE3 (3 tiers).

| Tier | VecI | VecF | Alignment |
|------|------|------|-----------|
| AVX-512 | `__m512i` | `__m512` | 64 |
| AVX2 | `__m256i` | `__m256` | 32 |
| SSSE3 | `__m128i` | `__m128` | 16 |

### Key SIMD Operations

- **dpbusdEpi32** (simd.h:204-207): Emulated via `maddubsEpi16` + `maddEpi16(x, 1)` + `addEpi32`. This is the unsigned*signed 8-bit dot product used for sparse L1 matmul. On VNNI-capable hardware, this could be a single `_mm256_dpbusd_epi32` instruction, but Obsidian emulates it for compatibility.

- **getNnzMask**: AVX-512 uses `_mm512_cmpgt_epi32_mask` (returns uint16). AVX2 uses `_mm256_movemask_ps` after `_mm256_cmpgt_epi32` (returns uint8). SSSE3 uses `_mm_movemask_ps` (returns uint8, but only 4 bits used).

- **packusEpi16 reordering** (nnue.cpp:189-213): Feature weights and biases are permuted at load time using `PackusOrder` to compensate for `packus` lane interleaving. AVX2: `{0,2,1,3}`, AVX-512: `{0,2,4,6,1,3,5,7}`.

- **FT activation**: Uses `mulhiEpi16` (high 16 bits of 16x16 multiply) with a left shift to implement the pairwise product with controlled precision.

- **Float layers** (L2, L3): Use `mulAddPs` (FMA) throughout. AVX-512 and AVX2 use hardware FMA; SSSE3 falls back to `add(mul(x,y), z)`.

**Coda comparison:** We don't have VNNI/dpbusd emulation or NNZ sparse matmul. Our SIMD covers accumulator updates and forward pass but misses the biggest optimization opportunity. Adding dpbusd-style int8 L1 matmul with NNZ sparsity would be the single largest NPS improvement.

### No VNNI Native Support

Despite the function being named `dpbusdEpi32`, there's no actual `_mm256_dpbusd_epi32` VNNI intrinsic used. It's always emulated with the 3-instruction sequence. A VNNI build target would be a free ~15% speedup on modern CPUs.

---

## 6. Lazy SMP

### Implementation (threads.cpp)

Standard Lazy SMP: all threads share only the TT. Per-thread: Position, SearchInfo stack, accumulator stack, all history tables, Finny table.

**Thread launching** (threads.cpp:74-78): Each thread creates its own `Search::Thread` object. Threads are synchronized with mutex + condition_variable (threads.cpp:44-49).

**Thread voting** (search.cpp:1488-1524): Best-thread selection uses weighted voting:
```cpp
votes[move] += (score - minScore + 9) * completeDepth;
```

Highest vote wins (unless TB scores are involved). This is similar to Stockfish's voting mechanism.

**searchStopped**: `std::atomic<bool>`, read with `memory_order_relaxed` (threads.cpp:19).

**Coda comparison:** Our Lazy SMP is similar. Their voting mechanism for best-thread selection is worth examining -- it weights votes by both score margin and search depth.

---

## 7. Time Management

### Calculation (timeman.cpp)

Simple and clean:
```cpp
int mtg = movestogo ? min(movestogo, 50) : 50;
int64_t timeLeft = max(1, time + inc*(mtg-1) - overhead*(2+mtg));
if (sudden death):
    optScale = min(0.025, 0.214 * time / timeLeft);
else:
    optScale = min(0.95/mtg, 0.88 * time / timeLeft);
optimumTime = optScale * timeLeft;
maximumTime = time * 0.8 - overhead;
```

### Dynamic TM (search.cpp:1457-1472)

At root, after each iteration:
```cpp
nodesFactor = 0.63 + notBestNodes * 2.00;
stabilityFactor = 1.71 - searchStability * 0.08;
scoreLoss = 0.86 + 0.010*(prevScore - score) + 0.025*(searchPrevScore - score);
scoreFactor = clamp(scoreLoss, 0.81, 1.50);
if (elapsed > stabilityFactor * nodesFactor * scoreFactor * optimumTime) break;
```

Three factors:
1. **nodesFactor**: How concentrated search effort is on best move (more concentrated = use less time)
2. **stabilityFactor**: How many iterations best move has been stable (more stable = use less time)
3. **scoreFactor**: Whether score is dropping (dropping score = use more time, up to 1.5x)

Also uses `searchPrevScore` (score from previous search, not just previous iteration) for longer-term score trend.

**Coda comparison:** Standard but clean. We have similar node/stability/score factors. Their use of `searchPrevScore` (from previous `go` command) for cross-search score trending is a nice touch.

---

## 8. Transposition Table

### Entry Structure (tt.h:19-76)

10 bytes per entry:
```cpp
uint16_t key16;       // 2 bytes
int16_t staticEval;   // 2 bytes
uint8_t agePvBound;   // 1 byte: age(5 bits) | pv(1 bit) | bound(2 bits)
uint8_t depth;        // 1 byte
uint16_t move;        // 2 bytes
int16_t score;        // 2 bytes
```

3 entries per bucket + 2 bytes padding = 32 bytes per bucket.

### TT Probe/Store

- **Probe** (tt.cpp:70-89): Linear search over 3 entries. On miss, returns lowest-quality entry for replacement.
- **Store** (tt.cpp:108-131): Overwrites if: exact bound, key mismatch, age distance >0, or `depth + 4 + 2*isPV > existing depth`.
- **No atomics**: Unlike some engines, Obsidian's TT store/load is not explicitly atomic. This could lead to data races under SMP, but the key16 matching provides a probabilistic guard.

**Coda comparison:** We use lockless TT with XOR-verified packed atomics (stronger SMP safety). Their TT is simpler but has potential race conditions. 3 entries per bucket vs our 4.

### TT Cutoff Enhancement

**search.cpp:743-758:** The TT cutoff has several interesting conditions:
- `cutNode == (ttScore >= beta)` -- only cut if the node type matches the score direction
- `pos.halfMoveClock < 90` -- avoid TT-induced draw blindness
- Retroactive cont history malus: when TT cutoff fires and opponent's last move was quiet with few alternatives (seenMoves <= 3), penalize opponent's last move in cont history

**Coda comparison:** The `cutNode == (ttScore >= beta)` condition is interesting -- it means upper-bound TT entries can't cut in cut nodes and lower-bound can't cut in non-cut nodes. This is more selective than our approach. The retroactive cont history penalty on TT cutoff is novel.

---

## 9. Novel / Notable Features

### Complexity-Aware LMR
The `ss->complexity` field (search.cpp:825) stores `abs(staticEval - rawStaticEval)` (difference between corrected and raw eval). Used to reduce LMR: `R -= complexity / 120`. Positions where correction history makes a large adjustment are searched deeper.

### Cuckoo Tables for Upcoming Repetition
**cuckoo.h/cpp**: Standard Cuckoo hash table for `hasUpcomingRepetition()` detection before making a move. Used in both qsearch and negamax to detect draws early.

### Copy-Make (not Make/Unmake)
Obsidian uses `Position newPos = pos; newPos.doMove(move, ...)` instead of make/unmake. The undo is just discarding the copy. This simplifies the code but may have memory implications.

### Eval History Update
The eval-based retroactive history update (search.cpp:840-844) is unique. After evaluating a position, it looks at the eval swing and adjusts the opponent's last move's history score accordingly. This provides a feedback mechanism where the engine learns from eval results.

### TT Static Eval Sharing
**search.cpp:827-832:** When evaluation is computed for a position not in TT, it's immediately stored (even without a search result) so other SMP threads can reuse the eval. This avoids redundant NNUE evaluations across threads.

---

## 10. Actionable Items for Coda/GoChess

### High Priority
1. **NNZ Sparse L1 Matmul**: The #1 NPS opportunity. Requires int8 L1 weights (our L1 is int16). With pairwise FT activation, ~50% of outputs are zero. Even with CReLU, some sparsity exists. Need to:
   - Train with int8 L1 quantization (QA_L1=64 or 127)
   - Implement NNZ tracking during FT activation
   - Use dpbusd-style kernel for L1 matmul (maddubsEpi16 emulation)

2. **Pairwise FT Activation**: More expressive than CReLU, provides natural sparsity for NNZ. Would require retraining.

3. **50mr-Aware TT Keys**: Cheap to implement, prevents stale draw evaluations in TT. Just XOR halfmove clock hash into TT probe key.

4. **Per-Color Non-Pawn Correction History**: Two additional correction tables keyed by per-color non-pawn material hash. Cheap memory, may improve eval accuracy.

### Medium Priority
5. **Continuation Correction History**: `[piece*64+to][piece*64+to]` indexed correction -- captures move-sequence-dependent eval errors.

6. **Dual Activation for Hidden Layers**: Feed both ReLU(x) and x^2 as separate inputs to next layer. Doubles effective L2 input width without retraining FT.

7. **Threat-Aware Move Ordering**: Large bonuses for escaping threatened squares, penalties for moving into threats. Simple to implement in scoreQuiets.

8. **Eval-Based Retroactive History**: Update opponent's last move history based on eval swing. Novel feedback mechanism.

### Lower Priority
9. **Complexity-Aware LMR**: Use correction history magnitude as LMR adjustment.
10. **cutNode-restricted NMP**: Only allow NMP on cut nodes (current: we allow on all non-PV).
11. **Score-dependent aspiration window**: `delta + avgScore^2 / 13000`.
12. **TT cutoff cont history penalty**: Penalize opponent's last move when TT cuts off.
13. **dpbusd preprocessing**: Transpose L1 weights at load time for optimal SIMD access pattern.

---

## Key Parameter Values (for reference)

| Parameter | Value | Notes |
|-----------|-------|-------|
| LMR base | 0.99 | |
| LMR divisor | 3.14 | |
| NMP base R | 4 | |
| NMP depth div | 3 | |
| NMP eval div | 147 | |
| RFP max depth | 11 | |
| RFP depth mul | 87 | |
| Razoring depth mul | 352 | |
| Futility base | 159 | |
| Futility depth mul | 153 | |
| Futility max depth | 10 | (lmrDepth) |
| LMP base | 3 | formula: (d*d+3)/(2-improving) |
| History pruning | -7471 * depth | |
| SEE quiet margin | -21 * lmrDepth^2 | |
| SEE capture margin | -96 * depth | |
| Singular beta | ttScore - depth | (depth*64/64) |
| Double ext margin | 13 | |
| Triple ext margin | 121 | |
| ProbCut margin | 190 | |
| QS FP margin | 156 | |
| QS SEE margin | -32 | |
| Aspiration start delta | 6 | + avgScore^2/13000 |
| Aspiration start depth | 4 | |
| NetworkScale | 400 | |
| QA | 255 | |
| QB | 128 | |
| Phase formula | score * (230 + phase) / 330 | phase = 2P+3N+3B+5R+12Q |
| Corr hist limit | 1024 | |
| Corr hist size | 32768 | |
| Pawn hist size | 1024 | |
| History divisor | 16384 | |

*Last updated: 2026-03-29, from Obsidian dev-16.15 source*
