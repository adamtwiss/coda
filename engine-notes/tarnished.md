# Tarnished Chess Engine - Technical Review

Source: `~/chess/engines/Tarnished/`
Version: 5.0 (rel. 2026-02-07)
Language: C++ (C++20)
NNUE: (768x16hm->1792)x2->(16->32->32)x8 (trained with Bullet, relabeled from (768x8hm->4096)x2)
Rating: #14 SP-CC, 3700 Blitz (CCRL)

Last reviewed: 2026-04-19

---

## 1. NNUE Architecture

### Network Topology: (768x16 PST -> 1792) x2 -> (16->32->32) x8

A complex multi-layer NNUE with **dual activation** function and 8 material-indexed output buckets. The network is trained from scratch with self-generated data using Bullet, and relabeled using a larger secondary network.

**Architecture summary** (`nnue.h:36-44`, `parameters.h:46-56`):
- Input features: `(piece, square, color)` standard HalfKA with 16 king input buckets
- Accumulator: `768x16hm -> 1792` (1792 = 768 + expansion layer)
- L1: `1792 -> 16` (integer quantized, i8)
- L2: `2x16 -> 32` (float)
- L3: `32 -> 1x8` (float, 8 output buckets for material)
- Network scale: `NNUE_SCALE = 369`

**Quantization constants** (`parameters.h:47-60`):
- `QA = 255` (feature quantization)
- `QB = 64` (FT output quantization)
- `FT_SHIFT = 10` (pairwise activation shift)
- `L1_MUL = (1 << FT_SHIFT) / (QA * QA * QB)` = 0.0010378...
- Weight clipping: `WEIGHT_CLIPPING = 1.98f`

**Coda comparison**: We use 768 FT -> 1024 accumulator (1 perspective) -> 16->32->1x8. Tarnished's L1_SIZE=1792 (168% of ours) but is still single perspective. The expansion to 1792 happens on-the-fly in the accumulator (likely residual or factorized). This gives more intermediate capacity at a memory cost.

### 1.1 PST Accumulator (`nnue.h`, `search.cpp:22-150`)

Standard HalfKA piece-square transformer:
- 16 king input buckets with detailed king-position granularity (`BUCKET_LAYOUT`, `parameters.h:72-81`)
- Horizontal mirroring when king file >= FILE_E
- Feature index: `bucket * 768 + ci * 384 + piece_type * 64 + square` (ci = color index)
- Weights: `i16[INPUT_BUCKETS * L1_SIZE * 768]` = `i16[16 * 1792 * 768]` (25M entries, 50 MB)
- Biases: `i16[L1_SIZE]` = `i16[1792]`

**Finny table cache** (`nnue.h:49-85`, `InputBucketCache`): Caches accumulated features per perspective, king bucket, and horizontal flip (2x2x16 = 64 cache entries). On king bucket change, diffs cached vs current bitboards to incrementally update.

**Incremental updates** (`search.cpp:22-150`, MakeMove): Lazy updates with feature deltas tracked per move. Stack contains `accumulator->featureDeltas[0/1]` for white/black perspectives.

**Refresh and lazy updates**:
- `needsRefresh` flag set when king moves to different bucket
- `applyDelta()` applies batched feature updates
- `fused refresh` mentioned in README but implementation unclear (likely combines multiple updates)

**Coda comparison**: Identical structure to our Finny table approach. 16 king buckets is more granular than our equivalent (we have fewer but with similar coverage).

### 1.2 Dual Activation (`nnue.cpp:72-114`)

**Pairwise product activation** (lines 72-114):
```
For each perspective (white and black):
    For i in [0, L1_SIZE/2):
        left = clamp(acc[i], 0, QA)
        right = clamp(acc[i + L1_SIZE/2], 0, QA)
        output[i] = mulhi_i16(left << (16 - FT_SHIFT), right) >> 0
```

The accumulator is split in half: left half clamped [0, 255], right half also clamped. The pairwise product uses high-order bits of a 16-bit multiply to approximate `(left * right) >> FT_SHIFT`. The `mulhi_epi16` instruction efficiently computes the top 16 bits.

**Two perspectives**: White and black perspectives are both computed, doubling the activation output to 1792 (2x896? or 1792 for each?). The code stores results in `output[i]` and `output[i + L1_SIZE/2]` for two sides.

**Coda comparison**: We use SCReLU (`x^2 clamp [0,1]`). Tarnished uses pairwise product which is more expressive but requires both halves. This is similar to Reckless's pairwise activation. The benefit is capturing feature interactions (both halves must be active for high output).

### 1.3 L1 Forward Pass (`nnue.cpp:116-158`)

**Sparse matmul with dpbusd**:
```
For input chunks i in [0, L1_SIZE):
    For output j in [0, L2_SIZE):
        sum[j] += dpbusd(input32[i], weights[i][j])
```

- L1 weights: `i8[OUTPUT_BUCKETS][L1_SIZE * L2_SIZE]` (quantized to i8, not i16)
- L1 biases: `float[OUTPUT_BUCKETS][L2_SIZE]`
- dpbusd is AVX2 instruction: `_mm256_maddubs_epi16` (dot-product of u8 * i8 -> i32)
- Input is u8 (from pairwise activation clamped to [0, 255])
- Output is dequantized: `sum_float = sum_int32 * L1_MUL + bias`
- Final clamp: `[0, 1]`

**Material-based output buckets** (8 buckets indexed by popcount):
```
Popcount:  0-8:  bucket 0
           9-12: bucket 1
          13-16: bucket 2
          17-19: bucket 3
          20-22: bucket 4
          23-25: bucket 5
          26-28: bucket 6
          29-32: bucket 7
```

**Coda comparison**: We use dense matmul with i32 weights. Tarnished uses i8 and dpbusd which is much faster on modern CPUs with AVX2. The material bucketing is identical to ours.

### 1.4 L2 and L3 Layers (`nnue.cpp:150-158`)

**L2**: Standard matmul
- Weights: `float[OUTPUT_BUCKETS][L2_SIZE * 2 * L3_SIZE]` (L2_SIZE=16, L3_SIZE=32)
- Biases: `float[OUTPUT_BUCKETS][L3_SIZE]`
- Input: `float[L2_SIZE * 2]` (from L1 output concatenated with its square)
- Activation: `ReLU` clamp `[0, 1]`

**L3**: Final layer
- Weights: `float[OUTPUT_BUCKETS][L3_SIZE]`
- Biases: `float[OUTPUT_BUCKETS]`
- Output: `dot(L3_input, L3_weights) + bias` * `NNUE_SCALE` (369)

**Network relabeling**: The architecture is trained at (768x8hm->4096)x2->(96->192->192->1)x8 then relabeled into the smaller (768x16hm->1792)x2->(16->32->32)x8 form. This is a knowledge distillation approach where the larger network teaches the smaller one.

**Coda comparison**: L2/L3 are float, same as Reckless. The L2 double-width (concatenate with square) is the same approach we use. Network relabeling is novel for us and may provide better net quality than direct training at small sizes.

---

## 2. Search

### 2.1 Framework (`search.cpp`)

- Iterative deepening with aspiration windows
- Principle Variation Search (PVS)
- Quiescence search with capture-only moves
- Cut-node tracking via `bool cutnode` parameter
- Stack structure with ply tracking and per-ply info

### 2.2 Aspiration Windows (`search.cpp:206-250`)

```
delta = INITIAL_ASP_WINDOW() = 28 (tunable)
depth >= MIN_ASP_WINDOW_DEPTH() (3) to enable
```

On fail-low:
- `beta = (3*alpha + beta) / 4` (asymmetric tightening)
- `delta *= ASP_WIDENING_FACTOR()` (multiply by tunable, default 1)

On fail-high:
- `alpha = max(alpha, score - delta)`
- `delta *= ASP_WIDENING_FACTOR()`

No explicit reduction on fail-high like Reckless.

**Coda comparison**: Our aspiration is simpler (fixed delta=15). Tarnished uses a tunable initial delta and widening factor. The asymmetric tightening on fail-low is interesting.

### 2.3 TT Cutoff (`search.cpp:417-450`)

Standard TT cutoff with depth/age checks:
- Requires `ttData.depth >= depth - 4` for lower bound, or exact match
- Downgrades mate scores near 50-move boundary (halfmove_clock based)
- Lazy updates: stores eval in TT via `ttData.staticEval`

**Coda comparison**: Standard TT logic. No novel gates observed.

### 2.4 Null Move Pruning (`search.cpp:505-545`)

```
Conditions:
  - depth >= 2
  - ss->staticEval >= beta + NMP_BETA_M_OFFSET() - NMP_BETA_M_SCALE() * depth
  - cut node only
  - not in check
  - has non-pawns

Reduction: NMP_BASE_REDUCTION() + depth / NMP_REDUCTION_SCALE() + 
           min(2, (ss->eval - beta) / NMP_EVAL_SCALE())
```

Tunable parameters:
- `NMP_BASE_REDUCTION = 5` (base reduction in plies)
- `NMP_REDUCTION_SCALE = 3` (depth divisor)
- `NMP_EVAL_SCALE = 195` (eval margin divisor)
- `NMP_BETA_M_OFFSET = 140` (beta margin offset)
- `NMP_BETA_M_SCALE = 7` (depth scale in margin)

**Coda comparison**: Similar formula to ours. The eval-margin scaling is standard.

### 2.5 Small ProbCut (`search.cpp:548-560`)

```
spcBeta = beta + SPROBCUT_MARGIN() = 400 (tunable)
Condition: ttData.bound == BETA_CUT && ttData.depth >= depth - 4 && 
           ttData.score >= spcBeta
Return: spcBeta (not full re-search)
```

This is a fast heuristic pruning: if TT shows a cutoff at a much higher score, prune.

**Coda comparison**: Simple heuristic without depth-dependent search like Reckless. Very fast.

### 2.6 Razoring (`search.cpp:486-500`)

```
depth < 2 only
margin = RAZORING_SCALE() * depth = 321 * depth / 100
if estimated_score < alpha - margin:
    return qsearch(alpha, beta)
```

Linear depth scaling (321/100 = 3.21 pawns per depth).

**Tunable**: `RAZORING_SCALE = 321` (per 100 units of depth)

**Coda comparison**: Our razoring uses quadratic depth. Tarnished uses linear, which is simpler and less aggressive.

### 2.7 Reverse Futility Pruning (`search.cpp:491-504`)

```
rfpMargin = RFP_SCALE() * depth - RFP_IMPROVING_SCALE() * improving + 
            corrplexity * RFP_CORRPLEXITY_SCALE() / 128
if depth >= 2 && !inCheck && ss->staticEval >= beta + rfpMargin:
    return ss->staticEval (or blend)
```

**Tunable parameters**:
- `RFP_SCALE = 89` (main margin per depth)
- `RFP_IMPROVING_SCALE = 70` (reducing when improving)
- `RFP_CORRPLEXITY_SCALE = 37` (complexity modulation, per 128)

The "corrplexity" is a measure of position complexity (likely TT score variance or move complexity).

**Coda comparison**: We use quadratic RFP. Tarnished uses linear depth but adds complexity modulation. Our version:
```
margin = 1125 * d^2 / 128 + 26 * d - 77 * improving + ...
```
Tarnished is simpler: linear depth with an improving heuristic and complexity scaling.

### 2.8 History Pruning (`search.cpp:587-592`)

```
if depth <= 4 && !isPV && getQuietHistory(move) <= -HIST_PRUNING_SCALE() * depth:
    skipQuiets = true
    continue
```

**Tunable**: `HIST_PRUNING_SCALE = 2820` (per depth)

Prunes quiet moves with bad history at low depths. For depth=1, threshold is -2820.

**Coda comparison**: Simple heuristic. We don't have explicit history pruning.

### 2.9 Futility Pruning (`search.cpp:596-604`)

```
futility = ss->staticEval + FP_SCALE() * depth + FP_OFFSET() + 
           ss->historyScore / FP_HIST_DIVISOR()
if !inCheck && isQuiet && lmrDepth <= 8 && alpha < 2000 && futility <= alpha:
    continue (skip move)
```

**Tunable parameters**:
- `FP_SCALE = 141` (per depth)
- `FP_OFFSET = 93` (base margin)
- `FP_HIST_DIVISOR = 93` (history scaling)

History-aware futility: better history moves get more leeway.

**Bad Noisy Futility** (lines 603-609):
```
futility = ss->staticEval + BNFP_DEPTH_SCALE() * depth + 
           BNFP_MOVECOUNT_SCALE() * moveCount / 128
```

For moves that fail SEE (bad captures).

**Coda comparison**: Similar logic. Tarnished's history-aware futility and move-count-aware bad noisy futility are nice refinements.

### 2.10 SEE Pruning (`search.cpp:610-614`)

```
seeMargin = isQuiet ? SEE_QUIET_SCALE() * lmrDepth - historyScore / SEE_QUIET_HIST_DIVISOR() :
                      SEE_NOISY_SCALE() * lmrDepth - historyScore / SEE_NOISY_HIST_DIVISOR()
if !SEE(board, move, seeMargin):
    continue
```

**Tunable parameters**:
- `SEE_QUIET_SCALE = -70` (negative, so less aggressive as depth increases)
- `SEE_QUIET_HIST_DIVISOR = 99`
- `SEE_NOISY_SCALE = -90`
- `SEE_NOISY_HIST_DIVISOR = 90`

History-aware SEE thresholds.

**Coda comparison**: Similar approach. Negative scales mean SEE pruning is tighter at high depth (move must be very good tactically).

### 2.11 Singular Extensions (`search.cpp:616-650`)

```
Conditions: depth >= 5, ttData.bound == BETA_CUT, ttData.depth >= depth - 4,
            valid TT score
            
sBeta = ttData.score - depth
sDepth = (depth - 1) / 2
score = search(..., sDepth, sBeta - 1, sBeta, ...)

extension = 0
if score < sBeta:
    extension = 1
    if depth <= 14 && score < sBeta - SE_DOUBLE_MARGIN():
        extension = 2
        if score < sBeta - SE_TRIPLE_MARGIN():
            extension = 3
```

**Tunable parameters**:
- `SE_BETA_SCALE = 27` (singular beta = tt_score - depth_with_this_scale)
- `SE_DOUBLE_MARGIN = 22`
- `SE_TRIPLE_MARGIN = 91`

Double and triple extensions based on how singular the move is.

**Negative extensions**: -3 if TT score >= beta (non-singular fail-high), -2 if cut-node.

**Coda comparison**: We removed singular extensions. Tarnished keeps them with fine-tuned margins. Their stronger NNUE may make SE more accurate. The distinction between double/triple margins and the negative extension for cut-nodes are sophisticated.

### 2.12 Late Move Reductions (`search.cpp:641-710`)

**Base reduction** (factorized LMR):
```
// Logarithmic table
lmrTable[isQuiet][depth][movecount] = base + log(depth) * log(movecount) / divisor

LMR_BASE_QUIET = 155/100 = 1.55
LMR_DIVISOR_QUIET = 257/100 = 2.57
LMR_BASE_NOISY = 26/100 = 0.26
LMR_DIVISOR_NOISY = 330/100 = 3.30
```

**Factorized LMR** (lines 683-706):
```
// Tuned via convolution of 8 boolean features
features = [isQuiet, !isPV, improving, cutNode, ttPV, ttHit, failHigh>2, corrplexity>margin]
reduction = lmrConvolution(features)  // precomputed lookup table
```

Then adjusts:
```
baseLMR = LMR_BASE_SCALE() * lmrTable[isQuiet][depth][movecount]
lmrDepth = max(depth - baseLMR / 1024, 0)
reduction -= historyScore / LMR_HIST_DIVISOR()
```

Adjustments by context:
- History scaling: `LMR_HIST_DIVISOR = 8212`
- Deeper conditions: `LMR_DEEPER_BASE = 30`, `LMR_DEEPER_SCALE = 6`

**Coda comparison**: Very sophisticated. The factorized LMR reduces using a 256-element lookup table (8 boolean features) instead of explicit depth-count tables. The convolution of features (8-bit number) indexes a precomputed reduction table. This is more expressive than our simple `log(d) * log(mc)` formula. The `corrplexity` feature (position complexity) gates reduction differently.

### 2.13 Quiescence Search (`search.cpp:281-393`)

Key features:
- TT cutoff (non-decisive scores only in PV)
- Stand-pat (eval >= beta early exit)
- Captures and promotions only (no quiet moves in standard QS)
- 3-move limit if not giving check
- SEE pruning with `QS_SEE_MARGIN = -71` (tunable)
- Prefetch TT entries

**Coda comparison**: Standard QS. Reckless does quiet moves in QS conditionally; Tarnished keeps it simple.

---

## 3. Move Ordering (`movepicker.cpp`, `movepicker.h`, `search.h:271-360`)

### 3.1 Move Picker Stages

1. **TTMOVE**: TT move (or QS move if capture/check/evasion)
2. **GEN_NOISY**: Generate and score all captures/promotions
3. **NOISY_GOOD**: Filter good captures (passing SEE), score by `16 * captured_value + capthist`
4. **KILLER**: Killer move heuristic
5. **GEN_QUIET**: Generate and score all quiet moves
6. **QUIET**: Iterate through quiet moves by score
7. **BAD_NOISY**: Failed-SEE captures (in order)

**Coda comparison**: Uses killers (we've moved to threat-aware history). Reckless relies entirely on history; Tarnished uses explicit killers as a shortcut.

### 3.2 Capture Scoring (`movepicker.cpp:10-48`)

```
score = getCapthist(move) + MVV_VALUES[captured_piece]
```

- MVV values: `[800, 2400, 2400, 4800, 7200]` for pawn-rook
- Capture history is threat-aware (4 buckets per threat context)
- SEE filter threshold: `-score / 4 + 15` (history-aware)

**Coda comparison**: Similar MVV/LVA-style ordering with history offset. The SEE threshold scaling with history is nice.

### 3.3 Quiet Move Scoring (`movepicker.cpp:20-46`)

```
score = getQuietHistory(move) + threat_bonuses

if piece == QUEEN:
    bonus += threats_on_from ? THREAT_QUEEN_BONUS : 0
    bonus -= threats_on_to ? THREAT_QUEEN_MALUS : 0
    // Similar for ROOK, BISHOP, KNIGHT
```

**Threat bonuses** (`search.h:150-156`):
- `THREAT_QUEEN_BONUS = 13224`
- `THREAT_QUEEN_MALUS = 10828`
- `THREAT_ROOK_BONUS = 10421`
- `THREAT_ROOK_MALUS = 8149`
- `THREAT_MINOR_BONUS = 7639`
- `THREAT_MINOR_MALUS = 5967`

Moves that escape threats get bonuses; moves into threats get maluses. Threats are precomputed bitboards (pawnThreats, knightThreats, etc.).

**Coda comparison**: We compute threats but don't use them in move ordering. Tarnished's threat-aware bonuses are a natural extension that should improve move ordering quality.

### 3.4 History Tables (`search.h:235-250`)

**Butterfly history** (`MultiArray<int, 2, 64, 64, 4>`):
- Indexed by `[stm][from][to][threatIndex]`
- threatIndex computed as `2 * threat_on_from + threat_on_to`
- Max value: `MAX_HISTORY = 16384`
- Bonus cap: `MAX_HISTORY_BONUS = 4096`

**Capture history** (`MultiArray<int, 2, 6, 6, 64, 4>`):
- Indexed by `[stm][moving_piece][captured_piece][to][threatIndex]`
- Same max values

**Continuation history** (`MultiArray<int16_t, 2, 6, 64, 2, 6, 64>`):
- Indexed by `[prev_stm][prev_piece][prev_to][stm][piece][to]`
- Updated at ply-1, ply-2, ply-4 (not ply-1,2,3,4)
- Max: `16384`, bonus cap: `4096`

**Pawn history** (`MultiArray<int16_t, 2, 1024, 6, 64>`):
- Indexed by `[stm][pawnKey % 1024][piece][to]`
- Max: `16384`

**Continuation correction history** (`MultiArray<int16_t, 2, 6, 64, 2, 6, 64>`):
- Same indexing as continuation history
- Used for eval correction

**Gravity formula** (`search.h:277-285`):
```
entry += bonus - entry * abs(bonus) / MAX_VALUE
```

Standard gravity with per-table MAX values. Bonus/malus are clamped to `MAX_BONUS = 4096`.

**Coda comparison**: Threat-indexed history is novel for us. The ply-1,2,4 continuation history matches ours (we also skip ply-3). The gravity formula is identical to ours.

### 3.5 History Bonuses/Maluses (`parameters.h:115-132`)

Quadratic depth-based bonuses/maluses:

```
historyBonus(d) = min(HIST_BONUS_QUADRATIC * d^2 + HIST_BONUS_LINEAR * d + HIST_BONUS_OFFSET, 2048)
                = min(9 * d^2 + 263 * d - 157, 2048)

historyMalus(d) = -min(HIST_MALUS_QUADRATIC * d^2 + HIST_MALUS_LINEAR * d + HIST_MALUS_OFFSET, 1024)
                = -min(4 * d^2 + 263 * d + 79, 1024)

Continuation: QUAD=7, LIN=227, OFF=-129 (bonus), QUAD=8, LIN=293, OFF=248 (malus)
Capture:      QUAD=5, LIN=242, OFF=-184 (bonus), QUAD=7, LIN=303, OFF=141 (malus)
```

**Coda comparison**: We use similar quadratic formulas. The specific coefficients differ, suggesting Tarnished has tuned these more aggressively. The offset is negative for bonuses (tighter curvature).

---

## 4. Evaluation & Correction History (`search.h:336-406`)

### 4.1 Static Eval Correction (`search.h:336-406`)

Five correction sources:

```
correction = PAWN_CORR_WEIGHT * pawnCorrhist[stm][pawnKey % ENTRIES]
           + MAJOR_CORR_WEIGHT * majorCorrhist[stm][majorKey % ENTRIES]
           + MINOR_CORR_WEIGHT * minorCorrhist[stm][minorKey % ENTRIES]
           + NON_PAWN_STM_CORR_WEIGHT * nonPawnCorrhist[stm][nonPawnKey[stm] % ENTRIES]
           + NON_PAWN_NSTM_CORR_WEIGHT * nonPawnCorrhist[stm][nonPawnKey[!stm] % ENTRIES]
           + CONT_CORR_WEIGHT * contCorrhist[ply-2] or [ply-4]
```

**Weights** (tunable):
- `PAWN_CORR_WEIGHT = 187`
- `MAJOR_CORR_WEIGHT = 197`
- `MINOR_CORR_WEIGHT = 171`
- `NON_PAWN_STM_CORR_WEIGHT = 202`
- `NON_PAWN_NSTM_CORR_WEIGHT = 202`
- `CONT_CORR_WEIGHT = 265`

Scaled by `/2048` and clamped.

**Coda comparison**: We have pawn correction only. Tarnished adds major, minor, and separate non-pawn tables for white and black. This 6-source correction is significantly richer and likely worth 20-40 Elo.

### 4.2 Correction Update (`search.h:337-357`)

Updated when eval is wrong:
```
bonus = (eval - best_score) clamped to [-MAX_CORR_HIST/4, +MAX_CORR_HIST/4]
entry += bonus - entry * abs(bonus) / MAX_CORR_HIST (gravity)
```

Updated at ply-2 and ply-4 for continuation correction.

**Coda comparison**: Similar logic. Their multi-source approach learns correction patterns more granularly.

---

## 5. Time Management (`search.h:177-209`)

Node-based time management with soft/hard limits:

```
softtime = NODE_TM_BASE / 100 * main - proportion * NODE_TM_SCALE / 100
complexity_scale = max((COMPLEXITY_TM_BASE / 100) + clamp(complexity, 0, 200) / COMPLEXITY_TM_DIVISOR, 1.0)
best_move_scale = max(1.8 - 0.1 * bmStability, 0.9)
softtime_adjusted = softtime * complexity_scale * best_move_scale
```

**Tunable parameters**:
- `NODE_TM_BASE = 154` (per 100)
- `NODE_TM_SCALE = 152` (per 100)
- `COMPLEXITY_TM_BASE = 78` (per 100)
- `COMPLEXITY_TM_DIVISOR = 382`
- `SOFT_TM_SCALE = 62` (per 100)

**Coda comparison**: Node-fraction and stability-based scaling. The complexity measure (TT score volatility?) and best-move stability help balance time allocation. Simpler than Reckless's exponential ramp but effective.

---

## 6. Transposition Table (`tt.h`)

- 3 entries per 32-byte cluster (10 bytes each + 2 padding)
- Entry: key(u16) + move(u16) + score(i16) + staticEval(i16) + depth(i8) + flags(u8)
- Flags: bound(2 bits) + pv(1 bit) + age(5 bits)
- Prefetch on move generation: `thread.searcher.TT.prefetch(prefetchKey(...))`

**Lemire fast modulo**: Likely uses `(hash as u128 * len as u128) >> 64` for fast indexing.

**Coda comparison**: Standard TT. Prefetching is a nice performance touch.

---

## 7. Board & SEE (`util.h`, `util.cpp`)

### 7.1 SEE Implementation

Standard alpha-beta SEE with piece values embedded in tunable parameters:
```
PAWN_VALUE = 105
KNIGHT_VALUE = 312
BISHOP_VALUE = 309
ROOK_VALUE = 492
QUEEN_VALUE = 1003
```

No special pin-awareness or x-ray logic visible in grep output.

**Coda comparison**: Reckless has pin-aware SEE. Tarnished's simpler approach is adequate and faster.

### 7.2 Threat Calculation (`search.cpp:345-346`)

Pre-computed per-ply:
```
ss->threats = calculateThreats(board)
```

Returns 7 bitboards (or 4 piece-specific):
- `threats[0]` = pawn threats
- `threats[1]` = knight threats
- `threats[2]` = bishop threats
- `threats[3]` = rook threats
- `threats[4-6]` = unclear from grep

Used for threat-aware history and move ordering.

**Coda comparison**: We don't pre-compute threats. Their approach provides cheap threat info for move ordering.

---

## 8. Network Training & Data Generation

From README.md:
- Self-generated training data
- Bullet trainer
- ~22 billion positions
- 5000/20000 soft nodes for self-play
- 8 random opening plies
- Relabeled with larger network: (768x8hm->4096)x2->(96->192->192->1)x8 -> smaller net

**Coda comparison**: We use external training data (Lichess). Tarnished's self-play is expensive but may yield superior nets due to self-consistent evaluation. The relabeling/distillation is novel and suggests high confidence in the larger net.

---

## 9. Notable Tunable Parameters Summary

| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| NNUE_SCALE | 369 | - | Network output scaling |
| INITIAL_ASP_WINDOW | 28 | 8-64 | Aspiration window delta |
| NMP_BASE_REDUCTION | 5 | 2-5 | Null move base reduction |
| SPROBCUT_MARGIN | 400 | 128-750 | Small ProbCut threshold |
| RFP_SCALE | 89 | 30-100 | Reverse futility per depth |
| SE_BETA_SCALE | 27 | 8-64 | Singular extension beta offset |
| SE_DOUBLE_MARGIN | 22 | 0-40 | Double extension threshold |
| SE_TRIPLE_MARGIN | 91 | 32-128 | Triple extension threshold |
| LMR_HIST_DIVISOR | 8212 | 4096-16385 | LMR history scaling |
| HIST_PRUNING_SCALE | 2820 | 512-4096 | History pruning threshold |
| FP_SCALE | 141 | 30-200 | Futility pruning depth scale |

**Tuning infrastructure**: SPSA support via weather-factory tuner. Extensive parameter tuning via self-play.

**Coda comparison**: Our tunable parameter count is similar. Tarnished's values reflect more aggressive tuning (e.g., HIST_PRUNING_SCALE=2820 vs our lighter pruning).

---

## 10. Unique/Novel Features

1. **Threat-aware move ordering**: Explicit threat bonuses/maluses for move piece and destination square. Cheap but effective.

2. **Threat-indexed history**: Butterfly and capture history bucketed by threat context (from_threat, to_threat).

3. **6-source correction history**: Pawn, major, minor, non-pawn(white), non-pawn(black), continuation. Much richer than single-source.

4. **Factorized LMR**: 8-bit feature vector -> 256-entry lookup table for reduction. Replaces explicit depth-movecount table.

5. **Network relabeling**: Large (768x8->4096)x2 net distills into smaller (768x16->1792)x2 net. Knowledge transfer from larger model.

6. **Dual activation with pairwise product**: Left/right halves of accumulator multiplied element-wise, capturing feature interactions.

7. **Material-aware eval scaling** (not NNUE-specific): Static eval scaled by material amount for complexity adjustment.

8. **Pre-computed threat bitboards**: Threats calculated once per ply, used throughout move ordering and evaluation.

9. **Simple small ProbCut**: Fast heuristic using TT score; no depth-dependent re-search.

10. **Complexity-aware search parameters**: RFP, TM, and LMR all scaled by a "corrplexity" metric.

---

## 11. Coda Comparison (Detailed)

### NNUE

**Coda v9**: 768 FT -> 1024 accumulator -> 16->32->1x8, SCReLU activation
**Tarnished**: 768 FT -> 1792 accumulator -> 16->32->32->1x8, pairwise product activation

Differences:
- Tarnished's 1792 is 75% larger (likely residual or factorized inside accumulator)
- Pairwise product is more expressive than SCReLU
- Relabeling via distillation may produce superior weights
- 16 king input buckets vs our ~16-equivalent (trade granularity vs different bucketing scheme)

**Potential port**: 
- Consider pairwise activation instead of SCReLU (requires retraining)
- Consider network relabeling from a larger teacher net
- 1792 accumulator may not be worth the cost (NPS hit + memory), but pairwise activation is worth testing

### Search: Selectivity

**Coda**: RFP quadratic, NMP eval-based, no SE, no history pruning
**Tarnished**: RFP linear + complexity, NMP similar, SE + double/triple, history pruning

Key wins:
- History pruning (HIST_PRUNING_SCALE) is a simple win for low-depth nodes
- Singular extensions with careful margins (may require stronger eval to be effective)
- Complexity gating on RFP/LMR is elegant

**Potential port**:
- Add history pruning at depth <= 4 (low risk, simple to implement)
- Reconsider SE with better gating (e.g., tied to eval stability)
- Add complexity metric to LMR/RFP (requires defining complexity)

### Search: Move Ordering

**Coda**: Killer moves + counter-move + history
**Tarnished**: Killer moves + threat-aware history + threat bonuses

Key win:
- Threat-aware bonuses in move ordering (e.g., THREAT_QUEEN_BONUS = 13224)
- Cheap to compute (pre-computed threat bitboards)

**Potential port**:
- Replace killer heuristic with threat bonuses (or supplement)
- Pre-compute and cache threat bitboards per ply
- Add threat-aware history buckets (4 instead of 1 per move)

### Search: History & Correction

**Coda**: Simple butterfly history + pawn correction (1 source)
**Tarnished**: Butterfly + capture + continuation + pawn/major/minor/non-pawn correction (6 sources)

Wins:
- Multi-source correction allows learning piece-specific eval errors
- Threat-indexed history captures move-quality variation by threat context

**Potential port**:
- Add major/minor/non-pawn correction (major undertaking, significant tuning)
- Add threat indexing to butterfly history (requires pre-computed threats)

### Time Management

**Coda**: Simpler node-based with stability
**Tarnished**: Similar, with explicit complexity scaling

Minimal difference. Tarnished's complexity metric is an enhancement.

**Potential port**: Low priority

### Tuning

**Coda**: SPSA on v5 tunables (~150 parameters)
**Tarnished**: Extensive SPSA + weather-factory (~100+ parameters)

Tarnished's parameters are well-tuned via self-play. Their values (e.g., SPROBCUT_MARGIN=400, SE_TRIPLE_MARGIN=91) should not be directly adopted without retuning on Coda's net/search.

**Potential port**: Tune Coda's parameters against a representative test set if adopting search/ordering changes.

---

## 12. Summary: Ideas to Port (Prioritized)

### Tier 1: High Impact, Low Effort (20-50 Elo potential)

1. **History Pruning** (`if depth <= 4 && getHistory(move) <= -SCALE*depth: skip`). 
   - Simple if-check before move is searched. Low false-positive rate because depth is capped at 4.
   - Tunable parameters: `HIST_PRUNING_SCALE` (start with 2820, tune down for Coda's weaker eval).

2. **Threat-Aware Move Ordering Bonuses** (THREAT_QUEEN_BONUS, THREAT_ROOK_BONUS, THREAT_MINOR_BONUS).
   - Requires pre-computing threat bitboards (pawn, knight, bishop, rook attacks).
   - Add bonus/malus to quiet move scores based on from_threat + to_threat.
   - Pre-computed threats are fast and reusable for other heuristics.

3. **Simple Small ProbCut** (heuristic using TT if score >> beta).
   - One-liner: `if ttScore >= beta + MARGIN && ttDepth >= depth - 4: return beta`
   - Low risk, fast win.

### Tier 2: Medium Impact, Medium Effort (10-30 Elo potential)

4. **Threat-Indexed History** (4 buckets per move instead of 1).
   - Requires reindexing all history tables from `[stm][from][to]` to `[stm][from][to][threatBucket]`.
   - Learns move quality variation by threat context (e.g., escaping threats vs moving into threats).
   - Moderate refactoring; large tuning effort.

5. **Multi-Source Correction History** (add major, minor, non-pawn sources to pawn-only).
   - Significant refactoring (5 new correction tables per side).
   - Requires separate hash keys for major/minor pieces (Tarnished tracks them incrementally).
   - Moderate code change, significant tuning.

### Tier 3: High Impact, High Effort (30-50 Elo potential, but risky)

6. **Singular Extensions with Careful Margins** (double/triple extension gates).
   - We removed SE after finding it harmful with our eval. Tarnished's success suggests their NNUE accuracy makes SE work.
   - Only viable if we significantly improve eval quality (NNUE architecture changes, retraining, relabeling).
   - High risk without strong confidence in eval.

7. **Pairwise Product Activation in NNUE** (instead of SCReLU).
   - Requires retraining network with Bullet.
   - More expressive than SCReLU but adds compute (multiply two halves).
   - Tuning infrastructure already exists.

8. **Network Relabeling / Distillation** (train large net, compress to smaller).
   - Requires training large (768x8->4096)x2 net, then distillation into (768x16->1792)x2.
   - Expensive (large datagen + training), but may yield superior weights.
   - Coda's training infrastructure may not support this workflow.

### Tier 4: Niche/Lower Priority

9. **Complexity-Aware LMR/RFP** (gate reduction by TT score variance or move complexity).
   - Requires defining a "complexity" metric. Tarnished's metric is unclear from code.
   - Medium effort to implement; benefits unclear for Coda.

10. **Factorized LMR** (256-entry lookup table from 8-bit feature vector).
    - Replaces logarithmic depth-movecount table.
    - Requires recomputing table; benefit depends on how well Coda's features map to Tarnished's.
    - Medium effort, unclear benefit without retuning.

---

## Final Notes

**Tarnished's Strengths**:
- Well-tuned parameters via extensive SPSA (self-play corpus reflects optimal parameters)
- Threat-aware systems (history, move ordering, bonuses) are effective and relatively cheap
- Multi-source eval correction learns position-type-specific eval errors
- Network relabeling/distillation may produce weights superior to direct training

**Tarnished vs Reckless**: 
- Reckless has threat *accumulator* (NNUE feature). Tarnished has threat *heuristics* (move ordering, history).
- Reckless is ~200 Elo higher but requires major NNUE architecture changes. Tarnished's threat heuristics are low-hanging fruit.

**Recommend starting with**: History pruning + threat bonuses in move ordering, then multi-source correction. These are highest bang-for-buck with manageable complexity.

---

**Reviewed by**: Hercules (Coda project)
**Date**: 2026-04-19
**Net**: lichdragon-3.bin (v5.0)
**Repo**: https://github.com/Bobingstern/Tarnished
