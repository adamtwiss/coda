# Stormphrax Chess Engine - Technical Review

Source: `~/chess/engines/Stormphrax/`
Author: Ciekce
Version: 7.0.70
Strength: CCRL ~3500+, top-15 engine
Net: `net066_255_128_q6_nc.nnue`
Language: C++20, Clang-only (no GCC support)

---

## 1. NNUE Architecture

### Current: `(704x16hm+60144 -> 640)x2 -> (32x2 -> 32 -> 1)x8`

**Ref:** `src/eval/arch.h`, `src/eval/nnue/arch/multilayer.h`, `src/eval/nnue/network.h`

This is one of the most sophisticated NNUE architectures in any open-source engine.

### Feature Transformer

**PSQ Features** (`src/eval/nnue/features/psq.h`):
- 704 inputs per king bucket (merged kings: 11 piece types x 64 squares)
- 16 king buckets with ABCD-side horizontal mirroring
- King bucket layout (visually flipped, a1=0):
  ```
  14 14 15 15    ranks 7-8 share
  14 14 15 15
  12 12 13 13    ranks 5-6 share
  12 12 13 13
   8  9 10 11    ranks 3-4 share
   8  9 10 11
   4  5  6  7    rank 2
   0  1  2  3    rank 1
  ```
- Total PSQ features: 704 x 16 = 11,264 per perspective
- Refresh table size: 32 (16 buckets x 2 mirror states)

**Threat Features** (`src/eval/nnue/features/threats.h`, `threats.cpp`):
- 60,144 additional threat input features
- Encoded as (attacker, attackerSq, attacked, attackedSq) tuples, king-relative
- Separate i8-typed weights (vs i16 for PSQ)
- Separate threat accumulator alongside PSQ accumulator, both lazy/incremental
- Incrementally updated: on each move, computes threat deltas (adds/removes) rather than recomputing
- Per-piece attack tables with directional encoding (forwards vs backwards)
- Threat refresh triggered when king crosses the e-file (mirror flip changes)

> **Coda comparison:** We have no threat features. This is the most novel NNUE feature in Stormphrax. The 60K threat features are a second parallel accumulator that encodes attack relationships. This is fundamentally different from what any engine we've reviewed has -- most use only PSQ (piece-square) features. The threat accumulator is additive with the PSQ accumulator at the FT activation stage (lines 93-95, 116-121 of `multilayer.h`): `inputs = add(psqInputs, threatInputs)`. This means threat features act as a bias correction on the main accumulator.

### Hidden Layers

**Architecture** (`src/eval/nnue/arch/multilayer.h`):
- FT output: 640 neurons (320 per perspective, pairwise)
- L1: 640 -> 32 (int8 weights, int32 biases) x 8 output buckets
- L2: 64 -> 32 (32 CReLU + 32 SCReLU via dual activation) x 8 output buckets
- L3: 32 -> 1 x 8 output buckets
- Output bucketing: `MaterialCount<8>` -- (popcount - 2) / 4

**Activation Chain:**
1. **FT -> L1**: Pairwise CReLU (multiply pairs, clip to [0, 255], pack to u8)
2. **L1 -> L2**: Dual activation (`kDualActivation = true`):
   - CReLU side: clamp [0, Q], shift left by kQuantBits
   - SCReLU side: square then clip to Q^2 (SF-style square-then-clip)
   - Concatenated: L2 gets 32 CReLU + 32 SCReLU = 64 inputs
3. **L2 -> L3**: CReLU (clamp [0, Q^3])
4. **L3 -> output**: Linear dot product

> **Coda comparison:** Our v7 arch is 1024 -> 16 -> 32 -> 1x8. Stormphrax uses 640 accumulator (pairwise from 1280) with the same 3-layer hidden structure but wider (32 -> 32 -> 1 vs our 16 -> 32 -> 1). The dual CReLU+SCReLU activation at L1 is unique -- they concatenate both activations to double the L2 input width without doubling L1 output neurons. **This is worth investigating:** it's essentially free from a training perspective (Bullet supports it) and doubles L2 information for minimal inference cost.

### Quantization

**Ref:** `src/eval/arch.h`

```
kFtQBits  = 8   (FT quantization: 256)
kL1QBits  = 7   (L1 quantization: 128)
kFtScaleBits = 7 (pairwise scale)
kQuantBits = 6   (hidden layer Q = 64)
```

- FT weights: i16 (PSQ), i8 (threat)
- FT output: i16 (accumulator), packed to u8 after pairwise activation
- L1 weights: i8 (enables VNNI `dpbusd`)
- L1 biases: i32
- L2/L3 weights: i32
- L2/L3 biases: i32
- Scale: 400

> **Coda comparison:** We use QA=255, QB=64. Stormphrax uses power-of-2 quantization (256, 128, 64) which is shift-friendly. The i8 L1 weights are key for VNNI performance. Our v7 also uses int8 L1 but with QA_L1=64.

### Sparse L1 MatMul

**Ref:** `src/eval/nnue/arch/util/sparse_default.h`, `sparse_vbmi2.h`

This is one of Stormphrax's most important performance innovations.

**Concept:** After FT activation, many outputs are zero (pairwise CReLU). Instead of doing a dense 640x32 matmul, track which 4-byte chunks are nonzero and only multiply those.

**Default (AVX2/NEON) implementation** (`sparse_default.h`):
- After FT activation, scan output in SIMD-width chunks
- For each chunk pair (a, b), compute `nonzeroMask` -- a bitmask of which i32-sized groups are nonzero
- Use a precomputed 256-entry LUT (`kNonZeroIndices`) to convert each byte of the mask to indices of nonzero elements
- Store nonzero indices into a compact array, track count
- L1 matmul iterates only over nonzero chunks (in groups of 4 for ILP):
  ```cpp
  for (chunk = 0; chunk < quadChunks; chunk += 4) {
      // load 4 nonzero input groups, broadcast each to SIMD
      // load corresponding weight rows, dpbusd accumulate
  }
  ```

**VBMI2 (AVX-512) implementation** (`sparse_vbmi2.h`):
- Uses `_mm512_maskz_compress_epi16` (VBMI2 instruction) to compress nonzero indices in a single instruction
- Much simpler: just compress, store, popcount
- `_mm512_kunpackw` to merge two 32-bit masks into a 64-bit kmask

> **Coda comparison:** We do not have sparse L1. This is a significant NPS optimization -- typically 30-60% of FT outputs are zero after pairwise CReLU, meaning the L1 matmul can skip 30-60% of work. **High priority to adopt.** The default implementation works on AVX2 without special instructions. The VBMI2 path is a bonus for Ice Lake+.

### Finny Tables (Refresh Table)

**Ref:** `src/eval/nnue.h`, lines 483-546

Standard Finny table implementation:
- Per-perspective `[RefreshTableSize]` cache (32 entries = 16 buckets x 2 mirror states)
- On king bucket change: diff cached vs current bitboards, apply only changed features
- Batched 4-at-a-time for adds and subs (`activateFourFeatures`, `deactivateFourFeatures`)
- Lazy accumulator with backtrack: scans backward to find last clean accumulator, then forward-applies all deltas

> **Coda comparison:** Similar to our Finny table. The 4-at-a-time batching is nice -- we could batch our delta applications similarly for better ILP.

### Network Loading

**Ref:** `src/eval/header.h`, `src/eval/nnue/loader.cpp`

- 64-byte header with magic, version, flags, arch ID, activation type, dimensions
- Supports zstd compression (embedded nets are zstd-compressed)
- Pre-permutation step: a separate `preprocess/permute.cpp` tool rearranges FT weights for SIMD-friendly memory layout at build time (AVX-512 `packus` produces non-sequential output, so weights must be reordered)
- NUMA-aware: network replicated per NUMA node

---

## 2. Search

**Ref:** `src/search.cpp` (1400+ lines)

### Iterative Deepening + Aspiration Windows
- Initial window: 16 (tunable)
- Widening factor: `delta += delta * 17/16`
- Aspiration reduction: on fail-high, reduce search depth by up to 3
- Re-center on fail-low: `beta = (alpha + beta) / 2`

### Null Move Pruning
- R = 6 + depth/5 (no eval-based reduction)
- Verification search at depth > 14: re-search at depth - R with cutnode=true
- NMP ply limit: prevents recursive NMP verification within 3/4 of remaining depth
- Guard: TT upper bound < beta prevents NMP

### Reverse Futility Pruning
- `depth <= 6 && staticEval - margin >= beta`
- Margin: `rfpMargin (71) * max(depth - improving, 0) + complexity * rfpCorrplexityScale / 128`
- Complexity from correction history delta
- Soft return: `(staticEval + beta) / 2` unless decisive

### Razoring
- `depth <= 4 && staticEval + 315 * depth <= alpha` => qsearch verification

### ProbCut
- `depth >= 7 && !ttpv && beta + 303 undecisive`
- SEE threshold: `(probcutBeta - staticEval) * 17/16`
- Two-phase: qsearch first, then full search at probcutDepth - 1

### Internal Iterative Reduction (IIR)
- `depth >= 3 && !excluded && (pvNode || cutnode) && (!ttMove || ttEntry.depth + 3 < depth)` => depth--

### Singular Extensions
- Conditions: `depth >= 6 + ttpv`, TT depth >= depth - 5, not upper bound, not decisive
- sBeta: `ttEntry.score - depth * (14 + 16 * (ttpv && !pvNode)) / 16`
- Search at `(depth - 1) / 2` with excluded move
- Double extension: `score < sBeta - 11` (non-PV only)
- Triple extension: `score < sBeta - 105` (non-PV, quiet TT move)
- Negative extensions: cutnode => -2, TT score >= beta => -1
- `cutnode |= extension < 0` (propagate cutnode on negative extension)

**Low-Depth Singular Extension (LDSE):**
- `depth <= 7 && !inCheck && staticEval <= alpha - 26 && ttFlag == LowerBound` => extension = 1

> **Coda comparison:** We removed singular extensions as harmful cross-engine. Stormphrax keeps them. Key differences: their sBeta includes a ttpv adjustment, they have LDSE (a simpler extension for shallow depths), and the `cutnode |= extension < 0` propagation is interesting -- negative singular extensions make the node act as a cutnode.

### Late Move Reductions (LMR)

**Table:** `[noisy][depth][legalMoves]` precomputed with:
- Quiet: `base=83, divisor=218` => `base/128 + ln(depth) * ln(moves) / (divisor/128)`
- Noisy: `base=-12, divisor=248`

**Adjustments (all scaled by /128):**
- +131 non-PV
- -130 ttpv
- -history/10835 (quiet) or -history/10835 (noisy)
- -148 improving
- -111 gives check
- +257 cutnode
- +128 ttpv fail-low (TT hit with score <= alpha)
- +64 * alphaRaises (number of times alpha was raised)
- -128 high complexity (corrDelta > 70)

**Deeper/shallower re-search:**
- If LMR result > alpha and reduced < newDepth:
  - Deeper: score > bestScore + 38 + 4 * newDepth
  - Shallower: score < bestScore + newDepth
  - `newDepth += doDeeperSearch - doShallowerSearch`

**LMR ttpv extension:**
- If ttpv and r < -128: add +1 to reduced depth (effectively extend by 1 in PV-adjacent nodes)

> **Coda comparison:** Similar LMR framework to ours. Notable differences: (1) separate quiet/noisy LMR tables (we have this), (2) alphaRaises counter as a reduction term (novel -- tracks how many moves raised alpha, increases reduction for later moves), (3) ttpv extension threshold, (4) high complexity reduction from correction history. The alphaRaises feature is interesting -- it's a cheap proxy for "this node is well-explored" that doesn't require additional data structures.

### Hindsight Reduction
- `depth >= 2 && parent.reduction >= 2 && parent.staticEval + curr.staticEval >= 200` => depth--

### Hindsight Extension (Inverse)
- `parent.reduction >= 3 && parent.staticEval + curr.staticEval <= 0` => depth++

> **Coda comparison:** We have hindsight reduction at margin 200 (same). Their hindsight extension (both evals sum to <= 0) is the inverse -- extend when the position became much worse than expected. We should test this.

### Late Move Pruning (LMP)
- Table: `(3 + depth^2) / (2 - improving)`
- Identical to our formula

### Futility Pruning
- `lmrDepth <= 8 && |alpha| < 2000 && staticEval + 261 + depth * 68 + history/128 <= alpha`

### History Pruning
- Quiet: `lmrDepth <= 5 && history < -2314 * depth + -1157`
- Noisy: `depth <= 4 && history < -1000 * depth^2 + -1000`

### SEE Pruning
- Quiet: `-16 * lmrDepth^2` (after good-noisy stage)
- Noisy: `min(-112 * depth - history/64, 0)` (history-adjusted)

### Good Noisy SEE Threshold
- `-score/4 + 15` -- SEE threshold is history-adjusted for good captures

> **Coda comparison:** The good noisy SEE using `-score/4 + offset` where score is the capture history is clever -- it means well-scored captures need less SEE margin to be considered "good". We use a fixed threshold.

### Fail-High Score Blending
- `bestScore = (bestScore * depth + beta) / (depth + 1)` when beta is exceeded (undecisive scores)

### QSearch
- Standing pat with beta blending: `(eval + beta) / 2` when undecisive
- Futility: `eval + 135`
- SEE threshold: -97
- Evasion handling with quiet generation
- Quiet search in qsearch: if TT move is quiet and TT flag is not upper bound

### Cycle Detection (Cuckoo Hashing)
**Ref:** `src/cuckoo.h`, `src/cuckoo.cpp`

Uses Stockfish-style cuckoo hashing to detect upcoming repetitions before they happen. `pos.hasCycle(ply, keyHistory)` is checked early in both search and qsearch.

> **Coda comparison:** We check for drawn positions via `isDrawn()` which includes repetition. The cuckoo approach is more proactive -- it detects when a repetition is *available* to either side, allowing the engine to score the position as a draw before the repetition actually occurs. This prevents the engine from entering lines where the opponent can force a draw. **Worth adopting.**

---

## 3. Move Ordering

**Ref:** `src/movepick.h`

### Stages
1. TT move
2. Generate noisy -> good noisy (SEE-filtered with history-adjusted threshold)
3. Killer (single killer)
4. Generate quiet -> quiet (history-scored)
5. Bad noisy (failed SEE in stage 2)

### Good Noisy SEE Filter
Captures that fail SEE at threshold `-score/4 + goodNoisySeeOffset()` are deferred to bad noisy stage. The score includes capture history, so well-scored captures have a lower SEE bar.

### Scoring

**Noisy:** `captureHistory/8 + SEE_value(captured) + promo_bonus`

**Quiet:** `mainHist + conthist` where:
- mainHist = `(butterfly + pieceTo) / 2`
- conthist = `cont[ply-1] + cont[ply-2] + cont[ply-4]/2`

### Selection Sort
Uses an optimized `findNext()` that packs score and index into a u64 for branchless max-finding:
```cpp
auto best = toU64(score) | (256 - idx);
best = std::max(best, curr);
bestIdx = 256 - (best & 0xFFFFFFFF);
```

> **Coda comparison:** Similar to our ordering. They use a single killer (we use 2). Their branchless selection sort trick is nice but minor. The SEE-adjusted good noisy filter is the main novelty.

---

## 4. History Tables

**Ref:** `src/history.h`

### Threat-Aware History

**All quiet history tables are 4-dimensional with threat awareness:**
- Butterfly: `[from][to][fromAttacked][toAttacked]` (64x64x2x2)
- PieceTo: `[piece][to][fromAttacked][toAttacked]` (12x64x2x2)

The `threats` bitboard (all opponent attacks) is used to index whether the from-square and to-square are under attack. This doubles the history table granularity.

**Update formula:**
```cpp
value += bonus - value * abs(bonus) / maxHistory  // gravity
```
maxHistory = 15769 (tunable)

**Conthist with base-aware update:**
```cpp
// base = conthist_score + mainHist/2
value += bonus - base * abs(bonus) / maxHistory
```
The conthist update uses the *combined* quiet score as the gravity base rather than the entry's own value. This means heavily-used moves in other contexts get less aggressive updates.

### Capture History
- `[from][to][captured][defended]` (64x64x13x2)
- Extra slot for non-capture queen promotions (index 12)
- `defended` = whether the target square is attacked by opponent

### Continuation History
- Standard `[piece][to]` subtables at ply-1, ply-2, ply-4
- ply-4 contribution is halved in scoring

### History Bonus/Penalty
- Bonus: `min(depth * 280 - 432, 2576)`
- Penalty: `-min(depth * 343 - 161, 1239)`
- Asymmetric: penalty cap (1239) is lower than bonus cap (2576)

> **Coda comparison:** Our history is `[from][to]` (4096 entries). Stormphrax's `[from][to][fromAttacked][toAttacked]` is 4x larger (16384 entries) but much more informative. Moves from/to attacked squares are fundamentally different from safe moves. The conthist base-aware update is also novel. **Both are high-priority improvements.**

---

## 5. Correction History

**Ref:** `src/correction.h`

### Tables (per color, shared across threads via atomics)
1. **Pawn** hash (16384 entries)
2. **Black non-pawn** material hash (16384 entries)
3. **White non-pawn** material hash (16384 entries)
4. **Major** piece hash (16384 entries)
5. **Continuation** hash at ply-1, ply-2, ply-4 (single table, keyed by `pos.key() ^ keyHistory[size - offset]`)

### Weights (tunable)
```
pawnCorrhistWeight        = 133
stmNonPawnCorrhistWeight  = 142
nstmNonPawnCorrhistWeight = 142
majorCorrhistWeight       = 129
contCorrhist1Weight       = 128
contCorrhist2Weight       = 192
contCorrhist4Weight       = 192
```

### Update
- `bonus = clamp((searchScore - staticEval) * depth / 8, -256, 256)`
- Gravity: `v += bonus - v * |bonus| / 1024`
- Entries are `atomic<i16>` (lock-free sharing across threads)

### Correction Application
- `correction = sum(weight_i * table_i) / 2048`
- STM/NSTM non-pawn weights are swapped based on side to move
- Applied after material scaling and halfmove scaling

### Eval Adjustment Pipeline
1. Static eval from NNUE
2. Contempt addition
3. Material scaling: `eval * (26500 + npMaterial) / 32768`
4. Optimism: `optimism * (2000 + npMaterial) / 32768`
5. Halfmove scaling: `eval * (200 - halfmove) / 200`
6. Correction history application
7. Clamp to [-ScoreWin+1, ScoreWin-1]

**Complexity:**
- `corrDelta = abs(eval_before_correction - eval_after_correction)`
- Used in RFP margin and LMR high-complexity reduction

> **Coda comparison:** We have pawn hash correction only. Stormphrax has 7 correction sources (pawn, 2x nonpawn, major, 3x continuation). The continuation correction history (keyed by XOR of current key with ancestor keys) is particularly interesting -- it captures position-pair relationships. The "complexity" metric derived from correction magnitude is used to modulate search depth. **Multi-source correction history is high priority.**

---

## 6. Performance / Build System

**Ref:** `Makefile`, `src/arch.h`

### Build Tiers
| Target | Flags | Key Features |
|--------|-------|-------------|
| `avx512` | `-march=icelake-client -mtune=znver4` | VNNI512, VBMI2, BMI2 |
| `avx2-bmi2` | `-march=haswell -mtune=znver3` | AVX2, BMI2 |
| `zen2` | `-march=bdver4 -mtune=znver2` | AVX2, no BMI2 (slow PEXT) |
| `armv8_4` | `-march=armv8.4-a` | NEON, dotprod |
| `native` | `-march=native` | Auto-detect |

### VNNI Usage
- `dpbusd` (VPDPBUSD): u8 x i8 dot product accumulate. Used in L1 matmul (sparse).
- `dpwssd` (VPDPWSSD): i16 x i16 dot product accumulate. Used in mulAddAdjAcc (FT activation).
- Both have 512-bit (VNNI512) and fallback (madd+add) paths.
- VNNI256 explicitly disabled (`SP_HAS_VNNI256 0`) with comment "slowdown on any cpu that would use it"

### Weight Permutation
A separate preprocessing step (`preprocess/permute.cpp`) reorders FT weights for SIMD-friendly memory access. AVX-512 `packus` produces non-sequential output (interleaved 256-bit halves), so weights must be pre-permuted to match. This happens at build time, not runtime.

### Network Embedding
- Net file is zstd-compressed and linked as binary data
- Decompressed at startup
- Pre-permuted for the target architecture at build time

### NUMA Support
- Optional libnuma integration (`-DSP_USE_LIBNUMA`)
- Network replicated per NUMA node
- Thread binding via `numa::bindThread(threadId)`
- Correction history allocated per NUMA node

### Attack Generation
- Two backends: **Black Magic** (magic bitboards) and **BMI2** (PEXT-based)
- BMI2 used on Intel + Zen3+, Black Magic on Zen1/Zen2 (slow PEXT)
- Zen2 build excludes BMI2 sources entirely

### TT Implementation
- 10-byte entries, 3 per cluster (30 bytes + 2 padding = 32-byte aligned cluster)
- 5-bit age, 1-bit PV, 2-bit flag packed into single byte
- Index: `(u128(key) * u128(count)) >> 64` (single multiply, no modulo)
- Lazy initialization: TT memory allocated but not touched until first use (avoids startup cost)

### Optimism
- Per-side optimism from average root score: `scale * avgScore / (|avgScore| + stretch)`
- scale=150, stretch=100
- Applied during eval adjustment alongside material scaling

> **Coda comparison:** The VBMI2 sparse matmul is the biggest NPS win we're missing. Weight pre-permutation at build time is a nice trick (we do runtime detection). NUMA support is relevant for server testing. The TT lazy init is a good UX improvement.

---

## 7. Novel Features Summary (Priority for Coda)

### HIGH PRIORITY
1. **Sparse L1 MatMul**: Track NNZ in FT output, skip zero chunks in L1 matmul. 30-60% speedup on L1. Default impl works on AVX2. (`sparse_default.h`)
2. **Threat-Aware History**: `[from][to][fromAttacked][toAttacked]` doubles history granularity. Simple to implement, likely +10-20 Elo. (`history.h`)
3. **Multi-Source Correction History**: Pawn + nonpawn(x2) + major + continuation(x3). Seven tables vs our one. (`correction.h`)
4. **Dual CReLU+SCReLU Activation**: Concatenate both activation types at L1 to double L2 input width. Free expressiveness. (`multilayer.h` lines 230-248)

### MEDIUM PRIORITY
5. **Cuckoo Cycle Detection**: Detect upcoming repetitions proactively. Prevents entering drawable lines. (`cuckoo.h`)
6. **Conthist Base-Aware Update**: Use combined quiet score as gravity base for continuation history updates. (`history.h` lines 112-125)
7. **Complexity from Correction Delta**: Use magnitude of correction history adjustment as a search depth modulator. (`search.cpp` lines 704-732)
8. **Alpha Raises Counter**: Track how many moves raised alpha as an LMR adjustment. (`search.cpp` line 1033)
9. **Good Noisy SEE History Adjustment**: `-captureScore/4 + offset` as SEE threshold for good captures. (`movepick.h` line 113)

### LOW PRIORITY / SITUATIONAL
10. **VBMI2 Sparse Path**: Only relevant for Ice Lake+ CPUs. Nice bonus over default sparse.
11. **Weight Pre-Permutation**: Build-time tool for SIMD-optimal weight layout. Saves ~1ms per startup.
12. **NUMA Replication**: Network + correction history per NUMA node. Matters for multi-socket.
13. **Hindsight Extension**: Inverse of hindsight reduction -- extend when position became worse. (`search.cpp` lines 748-752)
14. **TT Lazy Init**: Don't clear TT until first use after resize.

### DO NOT ADOPT
- **Singular Extensions**: Stormphrax uses them but our cross-engine testing showed them harmful for our engine. Their implementation has novel details (LDSE, cutnode propagation) but the fundamental concern remains.
- **VNNI256 disabled**: They found it's a slowdown. Don't bother with AVX2-VNNI.

---

## 8. Architecture Comparison Table

| Feature | Stormphrax | GoChess/Coda |
|---------|-----------|-------------|
| FT width | 640 (pairwise from 1280) | 1024 |
| King buckets | 16, ABCD-mirrored | Varies by net version |
| Merged kings | Yes (704 inputs) | No (768 inputs) |
| Threat features | 60,144 | None |
| FT activation | Pairwise CReLU -> u8 | CReLU or SCReLU -> i16 |
| L1 | 640->32, i8 weights, sparse | 1024->16, i8 weights, dense |
| L1 activation | Dual CReLU+SCReLU | SCReLU |
| L2 | 64->32, i32 weights | 16->32, i16 weights |
| L3 | 32->1, i32 weights | 32->1, i16 weights |
| Output buckets | 8 (material count) | 8 (material count) |
| Sparse L1 | Yes (NNZ tracking) | No |
| History dims | 4D (from/to/attacked/attacked) | 2D (from/to) |
| Correction sources | 7 | 1 (pawn) |
| Cycle detection | Cuckoo hashing | Repetition check |
| Build system | Makefile, Clang-only | Go build |
