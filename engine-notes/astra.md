# Astra Chess Engine - Technical Review

Source: `~/chess/engines/Astra/`
Version: 7.0-dev
Language: C++
NNUE: Single-accumulator (13x768 -> 1536) -> 16 -> 32 -> 1x8 with CReLU activations
Rating: Strong modern engine (competitive gauntlet strength)

Last reviewed: 2026-04-19

---

## 1. NNUE Architecture

### Network Topology: (13x768 -> 1536) -> 16 -> 32 -> 1x8 with CReLU

A clean, traditional NNUE with single PST accumulator and careful quantization. Smaller than Reckless but benefits from tightly-tuned parameters and sparse L1 processing.

**Constants** (`constants.h:5-18`):
- `INPUT_BUCKETS = 13` (king position + piece placement granularity)
- `FEATURE_SIZE = 768` (standard half-KA feature set)
- `FT_SIZE = 1536` (output of feature transformer, two perspectives concatenated)
- `L1_SIZE = 16` (first hidden layer)
- `L2_SIZE = 32` (second hidden layer)
- `OUTPUT_BUCKETS = 8` (material-based bucketing)
- `FT_SHIFT = 9`, `FT_QUANT = 255`, `L1_QUANT = 64`
- `EVAL_SCALE = 400` (final output multiplier)
- `DEQUANT_MULT = (1 << 9) / (255 * 255 * 64) = 0.00048828...`

**Coda comparison**: We use 768 FT with 16->32 hidden layers, same structure. Astra's 1536 FT is because they concatenate both perspectives (stm + nstm) before the hidden layers, then apply activation pairwise. We fuse them in the forward pass more flexibly. Both approaches work; theirs is more conventional.

### 1.1 Input Buckets (`constants.h:23-32`)

13 king input buckets with front-rank/back-rank differentiation:
```
Rank 8 (back):    0, 1, 2, 3, 3, 2, 1, 0
Rank 7:           4, 5, 6, 7, 7, 6, 5, 4
Rank 6:           8, 8, 9, 9, 9, 9, 8, 8
Ranks 5-1 (symmetric):  10, 10, ..., 10 (all same)
```
More granular at the back and middle ranks, less at the edge. The vertical asymmetry reflects piece activity differences between ranks.

**Coda comparison**: Our 16 buckets have even more granularity (4 files × 4 rank clusters). Theirs is more compact with 13 buckets. Both work, theirs is more memory-efficient.

### 1.2 Feature Transformer / PST Accumulator (`accum.h`, `nnue.cpp:230-242`)

Standard HalfKA approach:
- King mirroring when king file >= 4 (`feature_idx`: line 86)
- Feature index: `bucket * 768 + piece_type * 64 + (color != view) * 384 + relative_square`
- Weights: `int16_t[INPUT_SIZE * FT_SIZE]` = `int16_t[13*768 * 1536]`
- Biases: `int16_t[FT_SIZE]`
- Accumulators store full `int16_t[FT_SIZE]` for each perspective

Incremental updates via `put()` and `remove()` (`nnue.cpp:230-257`): simple SIMD add/sub of weight rows.

**Accumulator caching** (`accum.h:95-111`): `AccumEntry` stores piece bitboards per color/type for each cached king bucket position. On king bucket change, diff against cache. Otherwise, incremental update with add/sub tracking.

**Coda comparison**: Identical to our caching strategy. Same structure, same efficiency.

### 1.3 Forward Pass / CReLU Activation (`nnue.cpp:82-228`)

**Preparation** (`prep_l1_input`, lines 94-119):
```
For each perspective (stm, nstm):
  For each pair (i, i + 384):
    right = clamp(acc[i], 0, 255)              // CReLU right half
    left = min(acc[i + 384], 255)              // Max-clamped left half
    shifted_right = right << (16 - FT_SHIFT)   // Make room for multiply
    product = mulhi_i16(shifted_right, left)   // High 16 bits of product
    output[i] = product (as u8, i.e., lower 8 bits)
```
This is **pairwise CReLU activation** similar to Reckless' SCReLU but using CReLU (clamp [0, 255]) instead of squashing. The high-16-bit multiply compresses the product into u8 range without overflow.

**Sparse L1 matmul** (`find_nnz` and `forward_l1`, lines 121-195):
1. Find non-zero 32-bit chunks in the u8 activation (NNZ mask)
2. Use lookup table to convert bitmask to u16 indices of non-zero chunks
3. Sparse multiply: only process non-zero input chunks against L1 weights
4. Uses `dpbusd_epi32` (dot-product u8*i8 -> i32) with pairs processed together

L1 weights layout: transposed for fast access (`l1_weights[bucket][idx * L1_SIZE * 4 + ...]`).

L1 biases: `float[OUTPUT_BUCKETS][L1_SIZE]`, added after dequantization.

**L2/L3 layers** (`forward_l2`, `forward_l3`, lines 197-228):
- L2: `float[OUTPUT_BUCKETS][L1_SIZE][L2_SIZE]`, standard FMA with CReLU
- L3: `float[OUTPUT_BUCKETS][L2_SIZE]`, dot product + bias, multiply by `EVAL_SCALE = 400`

**Coda comparison**: Nearly identical forward pass to ours. Both use:
1. Pairwise activation (theirs CReLU, ours pairwise SCReLU)
2. Sparse L1 matmul with NNZ detection
3. Float L2/L3
The exact activation differs (CReLU vs pairwise SCReLU), but both achieve similar compression and sparsity benefits.

### 1.4 Output Buckets

Material-based bucketing (`nnue.cpp:85`):
```
bucket = (popcount(occupancy) - 2) / 4
```
8 buckets from 0 (endgame, 2-6 pieces) to 7 (full board, 26-32 pieces).

**Coda comparison**: Same bucket scheme.

---

## 2. Search

### 2.1 Framework (`search.h`, `search.cpp`)

- Iterative deepening with aspiration windows
- Root-level multipv support (multiple best moves tracked)
- `Stack` structure with offset indexing for continuation history lookback
- Generic `NodeType` trait (ROOT, PV, NON_PV) for specialization

### 2.2 Aspiration Windows (`search.cpp:145-185`)

```rust
delta = asp_delta (default 11)
On fail-low:
  beta = (alpha + beta) / 2
  alpha = max(alpha - delta, -INF)
  delta += delta / 3
On fail-high:
  alpha = beta - delta
  beta = min(beta + delta, INF)
  fail_high_count++
  new_depth = depth - fail_high_count  // Reduce on repeated fail-highs
```

Fixed delta (unlike Reckless' eval-dependent delta). Multiplicative growth factor of 1.33x per iteration.

**Coda comparison**: We use fixed delta=15. Astra's 11 is slightly smaller. Their fail-high reduction is useful, worth adding to our aspiration.

### 2.3 Evaluation (`search.cpp:870-909`)

**Raw NNUE evaluation**: `nnue.forward(board, accumulator)` returns i32.

**Material scaling** (line 900-906):
```cpp
int material = 122*P + 401*N + 427*B + 663*R + 1237*Q
eval = (230 + material / 42) * nnue_eval / 440
```
Stronger eval weight when more material on board. The formula `(230 + m/42) / 440` scales from ~0.52 (bare king + pawns) to ~0.75 (full board).

**50-move rule decay** (line 912):
```cpp
eval = eval * (200 - halfmove_clock) / 200
```
Linearly decay toward 0 as halfmove clock approaches 100.

**Correction histories** (line 913):
```cpp
eval += (corr_histories.get(board) + cont_corr_history.get(board, stack)) / 1024
```
Adds pawn/minor/non-pawn + continuation-based corrections.

**Coda comparison**: Material scaling is novel and sensible. We don't scale eval by material. The decay is standard. Correction histories, we have only pawn; they have 4 sources + continuation. Worth porting.

### 2.4 Correction Histories

**Sources** (`history.h:130-165`):
1. `pawn[color][8192]` indexed by pawn_hash
2. `minor_piece[color][8192]` indexed by minor_piece_hash
3. `w_non_pawn[color][8192]` indexed by white's non-pawn hash
4. `b_non_pawn[color][8192]` indexed by black's non-pawn hash

Combined in `get()` (line 145-151):
```cpp
int corr = p_corr_weight * pawn[stm][idx]
         + m_corr_weight * minor_piece[stm][idx]
         + np_corr_weight * w_non_pawn[stm][idx]
         + np_corr_weight * b_non_pawn[stm][idx]
```

Weights are tunable (default 4 each).

**Continuation correction history** (`history.h:167-200`): Indexes `[piece][square]` -> `PieceToHistory` (2D `[piece][square]`). Gets piece at previous move destination.

**Updates** (`search.cpp:682-690`):
```cpp
if (!in_check && !best_move.is_capture() && valid_tt_score(best_score, eval, bound)):
    bonus = clamp((best_score - raw_eval) * depth / 8, -256, 256)
    corr_histories.update(board, bonus)
    cont_corr_history.update(board, bonus, stack)
```

Updates only when eval was wrong (score disagreed with bound). Bonus magnitude proportional to error and depth.

**Coda comparison**: We have only pawn correction history. This 4-source + continuation approach is worth adopting. Simple to implement, likely +10-20 Elo.

### 2.5 Null Move Pruning (`search.cpp:375-414`)

Conditions:
- Cut node
- Not in check, not skipped
- `eval >= beta` and `eval + nmp_depth_mult * depth - nmp_base >= beta` (double gate)
- Non-pawn material exists
- `ply >= nmp_min_ply` (avoids at low ply)

Reduction:
```cpp
R = nmp_rbase + depth / nmp_rdepth_div + min(nmp_rmin, (eval - beta) / nmp_eval_div)
```

Default params: `nmp_rbase=4, nmp_rdepth_div=3, nmp_rmin=4, nmp_eval_div=193, nmp_depth_mult=26, nmp_base=192`.

Verification: at depth >= 16, re-search without reduction.

**Coda comparison**: Standard NMP. Their double-gating (both eval and adjusted_eval must exceed beta) is sensible. Depth divisor of 3 is similar to ours.

### 2.6 RFP (Reverse Futility Pruning) (`search.cpp:364-372`)

```cpp
if (!pv_node && depth < 11 && eval >= beta - (103*depth - 85*improving)):
    return (eval + beta) / 2
```

Conditions: non-PV, not skipped, depth < rfp_depth=11, not losing beta.

Margin: `rfp_depth_mult * depth - rfp_improving_mult * improving` (defaults: 103, 85).

Returns blended value, not raw beta.

**Coda comparison**: Similar structure. Their improving multiplier (85) is high; ours is typically lower. Worth testing their values.

### 2.7 Razoring (`search.cpp:360-361`)

```cpp
if (!pv_node && eval < alpha - (318 - 234*depth^2)):
    return quiescence(alpha, beta)
```

Quadratic margin scaling (wider at shallower depths). Parameters: `rzr_base=318, rzr_mult=234`.

**Coda comparison**: Standard razoring. Their coefficients seem reasonable.

### 2.8 Late Move Pruning (`search.cpp:492-493`)

```cpp
quiets.size() > (3 + depth^2) / (2 - improving)
```

More aggressive when improving (higher denominator = threshold increases faster).

**Coda comparison**: Same formula as we use.

### 2.9 Futility Pruning (`search.cpp:498-501`)

```cpp
futility = eval + fp_base + r_depth * fp_mult
if (r_depth < fp_depth && futility <= alpha):
    skip_quiets = true
```

Parameters: `fp_base=88, fp_mult=104, fp_depth=10`.

Uses reduced depth `r_depth = max(0, depth - lmr_reduction + history_score / hist_div)`.

**Coda comparison**: History-aware futility (adjusted for move quality). We don't condition on r_depth explicitly. Worth testing.

### 2.10 History Pruning (`search.cpp:504-507`)

```cpp
if (r_depth < hp_depth && history_score < hp_depth_mult * depth):
    skip_quiets = true
    continue
```

Parameters: `hp_depth=6, hp_depth_mult=-5429`.

Prunes low-history moves at shallow depths.

**Coda comparison**: We have history pruning. Their multiplier (-5429) is quite negative, meaning only very-low-history moves get pruned.

### 2.11 SEE Pruning (`search.cpp:510-516`)

```cpp
Quiet threshold: -16*depth^2 + 52*depth - 21*history/1024 + 22, min(0)
Noisy threshold: -8*depth^2 - 36*depth - 32*history/1024 + 11, min(0)
```

Both history-aware (negative history makes threshold easier to pass). Parametric coefficients all tunable.

**Coda comparison**: Similar quadratic structure with history awareness.

### 2.12 Late Move Reductions (`search.cpp:476-598`)

Base formula (lines 864-868):
```cpp
int lmr_base = 111, lmr_div = 297
double reduction = lmr_base / 100.0 + log(depth) * log(move_count) / (lmr_div / 100.0)
```

Adjustments (`search.cpp:563-576`):
```cpp
r += !improving                          // +1 if not improving
r += 2 * cut_node                        // +2 at cut nodes
r += (tt_move && tt_move.is_noisy())     // +1 if TT move is noisy
r -= tt_pv                               // -1 if PV in TT
r -= board.in_check()                    // -1 if in check after move
r -= (tt_depth >= depth)                 // -1 if good TT move
r -= history_score / (is_quiet ? quiet_hist_div : noisy_hist_div)  // History adjustment
```

History adjustment: divide by `quiet_hist_div=8026` or `noisy_hist_div=4566`.

**Re-search and depth adjustment** (`search.cpp:581-586`):
```cpp
if (score > alpha && new_depth > r_depth):
    new_depth += (score > best_score + 39 + 2*new_depth)  // Extend if much better
    new_depth -= (score < best_score + new_depth)         // Reduce if barely better
```

Parameter `zws_margin=39` for zero-window re-search bonus.

**Coda comparison**: Classic logarithmic LMR with multiplicative adjustments. Our formula is similar. Their history divisors (8026, 4566) are large; ours are smaller. The re-search adjustment with `zws_margin` is sensible.

### 2.13 Singular Extensions (`search.cpp:520-552`)

Conditions: `depth >= 6`, `tt_depth >= depth - 3`, TT has lower bound, score is non-decisive, `ply < 2 * root_depth`.

```cpp
sbeta = tt_score - (1 + tt_pv && !pv_node) * depth
singular_depth = (depth - 1) / 2
score = negamax<NON_PV>(singular_depth, sbeta - 1, sbeta, ...)

if (score < sbeta):
    extensions = 1
    if (!pv_node && score < sbeta - 13):           // double_ext_margin = 13
        extensions = 2 + (quiet && score < sbeta - 93)  // tripple_ext_margin = 93
else if (sbeta >= beta):
    return sbeta  // Multi-cut
else if (tt_score >= beta):
    extensions = -3
else if (cut_node):
    extensions = -2
```

Parameters: `double_ext_margin=13, tripple_ext_margin=93` (tunable).

**Coda comparison**: We disabled singular extensions. Astra keeps them with modest thresholds (13, 93). Their multi-cut condition is nice. If you retune, singular extensions might be worth revisiting.

### 2.14 Internal Iterative Reduction (`search.cpp:417-418`)

```cpp
if (depth >= iir_depth && !tt_move && (pv_node || cut_node)):
    depth--
```

Parameter: `iir_depth=4` (depths 4+). Reduces when no TT move.

**Coda comparison**: We have IIR. Same depth threshold.

### 2.15 ProbCut (`search.cpp:421-455`)

```cpp
probcut_beta = beta + pc_margin - pc_improving_mult * improving
probcut_depth = max(depth - 4, 0)
```

Parameters: `pc_margin=217, pc_improving_mult=60`.

Only captures, only non-PV. Returns `score - (probcut_beta - beta)` if succeeds (blended cutoff).

**Coda comparison**: Similar structure. Their margin and improving adjustment are tunable.

### 2.16 Quiescence Search (`search.cpp:698-839`)

Key features:
- Stand pat: `best_score = eval` (without blending initially)
- TT lookup (non-PV nodes only)
- Generates noisy moves (captures/promotions)
- **Conditional quiet generation**: If in check, generates quiet moves (evasions)
- Futility pruning in qsearch: `futility <= alpha && !board.see(move, 1)` (line 793)
- SEE pruning: `qsee_margin=-20` (default)
- Blending at exit: `(best_score + beta) / 2` (line 831)

**Coda comparison**: Standard qsearch with quiet generation in check. Our qsearch doesn't generate quiets in non-check positions. Their conditional approach is fine.

### 2.17 TT Cutoff (`search.cpp:262-284`)

```cpp
if (!pv_node && !skipped && valid_score(tt_score) 
    && tt_depth > depth - (tt_score <= beta)
    && valid_tt_score(tt_score, beta, tt_bound)
    && (cut_node == (tt_score >= beta) || depth > 5)):
    
    // Update history on cutoff
    if (tt_move && tt_score >= beta):
        if (tt_move.is_quiet()):
            quiet_history.update(stm, tt_move, bonus)
        if (depth <= 3 && prev_move):
            cont_history.update(piece, prev_sq, -malus, ...)
    
    if (halfmove < 90):
        return tt_score
```

The `cut_node == (tt_score >= beta) || depth > 5` condition allows TT cutoff when node type and bound direction agree, or at depth > 5 unconditionally.

History updates on cutoff are nice (reinforces good moves found shallowly).

**Coda comparison**: Similar structure. The history update on cutoff is a small detail we could adopt.

### 2.18 Quiet Move Evaluation (`search.cpp:346-357`)

```cpp
if (!skipped && prev_move && prev_move.is_quiet() && valid(prev_eval)):
    bonus = clamp(static_h_mult * (current_eval + prev_eval) / 16, static_h_min, static_h_max)
    quiet_history.update(~stm, prev_move, bonus)
```

Parameters: `static_h_mult=-51, static_h_min=-108, static_h_max=226`.

Updates opponent's (previous mover's) quiet history based on how the position turned out. This is a form of eval-based move quality assessment.

**Coda comparison**: We have similar eval-based adjustments elsewhere. This approach is cleaner.

---

## 3. Move Ordering (`movepicker.cpp`, `history.h`)

### 3.1 Move Picker Stages

1. **PLAY_TT_MOVE**: TT move (if pseudo-legal)
2. **GEN_NOISY** / **PLAY_NOISY**: Generate captures/promotions, score by `noisy_history + 16*captured_value`, SEE threshold
3. **GEN_QUIETS** / **PLAY_QUIETS**: Generate quiet moves (conditional), score by history/continuation/threats
4. **PLAY_BAD_NOISY**: Failed-SEE captures

**SEE threshold for noisy moves** (line 70):
```cpp
threshold = (st == N_SEARCH) ? -move.score / 32 : probcut_threshold
```

In normal search, apply a -score/32 SEE threshold (higher-scoring moves allowed more SEE slack).

**Coda comparison**: No killer moves, no counter-move table. History alone suffices due to richer design.

### 3.2 Noisy Move Scoring (`movepicker.cpp:124-131`)

```cpp
for (auto& m : ml_main):
    captured = m.is_ep() ? PAWN : piece_type(board.piece_at(m.to()))
    m.score = noisy_history.get(board, m) + 16 * piece_values(captured)
```

Combines noisy history with 16x the captured piece value. The 16x scalar ensures material dominates but history can override.

**Coda comparison**: Same as ours.

### 3.3 Quiet Move Scoring (`movepicker.cpp:133-164`)

```cpp
score = 2 * (quiet_history + pawn_history)
for i in {1,2,4,6}:
    score += cont_hist[i][piece][to]

// Threat bonuses
if (pt != PAWN && pt != KING):
    danger = threats[PAWN]
    if (pt >= ROOK):
        danger |= threats[BISHOP] | threats[KNIGHT]
    if (pt == QUEEN):
        danger |= threats[ROOK]
    
    bonus = QUEEN_BONUS (20480) / ROOK_BONUS (12288) / MINOR_BONUS (7168)
    if (piece threatened):     score += bonus
    if (destination threatened): score -= bonus

if (board.gives_check() && board.see(m, -quiet_checker_bonus)):
    score += 16384
```

Uses 4-ply continuation history lookback (1,2,4,6). Heavy threat-based bonuses (up to 20k). Check moves get big bonus if SEE is good.

**Coda comparison**: We don't explicitly use threat bitboards in move ordering. Their threat bonuses (escape/danger/check) are a good port target. The 4-ply continuation lookback (vs our 2) gives more context.

### 3.4 History Table Design

**QuietHistory** (`history.h:22-41`):
```cpp
int16_t data[NUM_COLORS][NUM_SQUARES][NUM_SQUARES]
```
Simple `[color][from][to]` table. No factorization.

**NoisyHistory** (`history.h:43-70`):
```cpp
int16_t data[NUM_PIECES][NUM_SQUARES][NUM_PIECE_TYPES + 1]
```
Indexed by `[piece][to][captured_type]`. Depends on what was captured.

**ContinuationHistory** (`history.h:104-128`):
```cpp
int16_t data[2][2][NUM_PIECES+1][NUM_SQUARES][NUM_PIECES+1][NUM_SQUARES]
```
6D array: `[in_check][is_cap][piece][to][cont_piece][cont_to]`. Accessed at ply-1, ply-2, ply-4, ply-6.

**PawnHistory** (`history.h:72-102`):
```cpp
int16_t data[8192][NUM_PIECES][NUM_SQUARES]
```
Indexed by pawn hash (8192 entries), then piece and destination.

**ContinuationCorrectionHistory** (`history.h:167-200`):
Same indexing as continuation history but for correction values.

**Gravity updates** (`history.h:14-19`):
```cpp
int history_bonus(int d) {
    return min(max_hist_bonus, hist_bonus_mult * d + hist_bonus_minus)
}
int history_malus(int d) {
    return min(max_hist_malus, hist_malus_mult * d + hist_malus_minus)
}
```

Linear bonus/malus with tunable multiplier/offset/max. Parameters:
- Quiet: `mult=308, minus=4, max=2445` for bonus; `mult=311, minus=104, max=1646` for malus
- Noisy: `mult=308, minus=4, max=2445` / `mult=311, minus=104, max=1646` (same)
- Continuation: separate

**Coda comparison**: No factorizer (unlike Reckless). Simpler design, fewer moving parts. Continuation history indexed by `[in_check][is_cap]` provides context. Pawn history is good. The linear bonus shape matches Reckless, unlike our `min(MAX, MULT*d)`. Their shape is `MULT*d + OFFSET` which allows negative values. This might be worth testing.

---

## 4. Time Management (`timeman.h`)

**Soft/hard time calculation** (`get_optimum` lines 29-44):
```cpp
mtg = min(50, moves_to_go ? moves_to_go : 50)
adj_time = time_left + inc * (mtg - 1) - overhead * (mtg + 2)

scale = min(
    moves_to_go ? 1.034612 / mtg : 0.029935,
    (moves_to_go ? 0.88 : 0.213) * time_left / max(1, adj_time)
)

optimum = time_left * scale
maximum = time_left * 0.8 - overhead
```

Different formulas for `moves_to_go` known vs unknown. Includes overhead adjustment.

**Soft stop during search** (`search.cpp:105-119`):
```cpp
stability_factor = (157 / 100.0) - stability * (35 / 1000.0)
result_change_factor = (70/100) + (15/1000)*(prev_depth_score - current)
                     + (29/1000)*(prev_depth-2_score - current)
result_change_factor = clamp(result_change_factor, 101/100, 147/100)

node_ratio = best_move.nodes / total_nodes
node_count_factor = (1.0 - node_ratio) * (219/100) + (58/100)

if (elapsed > optimum * stability * result_change * node_count):
    break
```

Multi-factor soft stop: stability (best-move consistency), score changes (eval volatility), node distribution.

Parameters: `tm_stability_base=157, tm_stability_mult=35, tm_results_base=70, tm_results_mult1=15, tm_results_mult2=29, tm_node_mult=219, tm_node_base=58`.

**Coda comparison**: More sophisticated than typical. The multi-factor approach (stability + eval volatility + node distribution) is worth studying. Most engines use simpler stopping logic.

---

## 5. Transposition Table

### 5.1 Structure (`tt.h`)

- 3 entries per 32-byte cluster
- Entry: `key(u16) + move(u16) + score(i16) + eval(i16) + depth(i8) + flags(u8)`
- Flags: `bound(2 bits) + tt_pv(1 bit) + age(5 bits)`
- Lemire fast modular indexing

### 5.2 Replacement Policy

Priority: exact key match > empty slot > lowest quality.
Quality = `depth - 4 * relative_age`.

**Coda comparison**: Standard replacement policy.

---

## 6. Board / Move Generation

### 6.1 Threats (`movepicker.cpp:136-159`)

Pre-computed threat bitboards per piece type:
```cpp
Threats threats = board.threats()
danger = threats[PAWN]  // bitboard of squares attacked by enemy pawns
```

Used for escape/danger bonuses in move ordering.

**Coda comparison**: We compute threats in the NNUE accumulator. Using them in move ordering too is the missing piece.

### 6.2 SEE (Static Exchange Evaluation)

Standard alpha-beta SEE. Called in move ordering, pruning, and search.

**Coda comparison**: We have SEE; all engines do.

---

## 7. Tunable Parameters Summary

Astra has extensive SPSA parameters (from `tune_params.h`). Notable ones:

| Parameter | Value | Min | Max | Purpose |
|-----------|-------|-----|-----|---------|
| `asp_delta` | 11 | 1 | 30 | Aspiration window size |
| `asp_depth` | 4 | 2 | 6 | Aspiration window depth |
| `lmr_base` | 111 | 40 | 200 | LMR base (percent) |
| `lmr_div` | 297 | 150 | 500 | LMR divisor (percent) |
| `nmp_rbase` | 4 | 1 | 5 | NMP reduction base |
| `nmp_depth_mult` | 26 | 1 | 58 | NMP depth multiplier |
| `rfp_depth` | 11 | 2 | 20 | RFP max depth |
| `fp_base` | 88 | 1 | 300 | Futility pruning base |
| `hp_depth` | 6 | 1 | 15 | History pruning depth |
| `zws_margin` | 39 | 10 | 160 | LMR re-search threshold |
| `quiet_hist_div` | 8026 | 1 | 16384 | Quiet history divisor |
| `hist_bonus_mult` | 308 | 1 | 1536 | History bonus multiplier |
| `tm_stability_base` | 157 | 100 | 200 | Time mgmt stability |
| `tm_results_mult1` | 15 | 0 | 27 | Time mgmt eval volatility |

**Coda comparison**: Our tunable count is similar. Astra's divisors (8026, 4566) are large compared to ours. Their multipliers are reasonable. The time management parameters are well-tuned.

---

## 8. Unique / Novel Features

### 8.1 Material-Scaled Eval

Eval weight increases with material: `(230 + material/42) / 440`. Simple but effective for endgame accuracy.

### 8.2 4-Source Correction History

Pawn, minor, non-pawn (both sides) + continuation. Richer than our 1-source.

### 8.3 Threat-Based Move Ordering Bonuses

Escape, danger, and check bonuses with large multipliers (7k-20k). Makes move ordering much smarter.

### 8.4 Multi-Factor Soft Stop

Combines stability, eval volatility, and node distribution for sophisticated stopping logic.

### 8.5 Linear History Bonus Shape

`bonus = MULT*depth + OFFSET` allows negative bonuses and is more flexible than `min(MAX, MULT*d)`.

---

## 9. Coda Comparison: Architecture and Search

### NNUE

| Aspect | Astra | Coda v9 |
|--------|-------|---------|
| FT width | 1536 (13*768, concat) | 768*2 (fused) |
| Accumulator | Single PST | Single PST + threats (66k) |
| Activation | CReLU pairwise | Pairwise SCReLU + threats |
| Hidden | 16 -> 32 -> 1x8 | 16 -> 32 -> 1x8 |
| Sparse L1 | Yes (NNZ) | Yes (NNZ) |
| Advantage | Clean, efficient | Richer features (threats) |

### Search

| Aspect | Astra | Coda v9 |
|--------|-------|---------|
| LMR formula | log(d)*log(mc)/div | Similar |
| NMP | Cut-node only | Cut-node only |
| RFP | Depth-based margin | Similar |
| Singular ext | Yes (+1/+2/-3) | Disabled (was harmful) |
| Correction hist | 4 sources | 1 source (pawn) |
| Eval scaling | Material-weighted | Fixed |
| Time mgmt | Multi-factor soft stop | Simpler |
| Move ordering | Threat bonuses | History-only |

### Key Differences

**Astra strengths**:
1. Correction history with 4 sources (pawn, minor, non-pawn, continuation)
2. Material-scaled eval
3. Threat-based move ordering bonuses (escape, danger, check)
4. Multi-factor soft-stop time management
5. Singular extensions (if retrained)

**Coda strengths**:
1. Threat accumulator in NNUE (unique)
2. Slightly more tuned LMR/NMP parameters
3. (When retrained) Threat-aware history (factorized)

---

## 10. Summary: Ideas to Port

### High Priority (10-20 Elo each, moderate effort)

1. **4-source correction history**: Pawn + minor + non-pawn(white) + non-pawn(black). Replace our single pawn table.
   - Implementation: Add 3 more correction history tables, combine in `adjust_eval()`.
   - Tuning: 4 new weight parameters per source (similar structure).

2. **Threat-based move ordering bonuses**: Escape, danger, check bonuses.
   - Implementation: Compute threat bitboards in move picker, add bonuses to quiet move scores.
   - Tuning: 3 bonus values (escape, danger, check) and SEE threshold.

3. **Material-scaled eval**: `eval = (230 + material/42) * raw_eval / 440`.
   - Implementation: 1 line in `evaluate()`.
   - Tuning: 230 and 42 are tunable.

### Medium Priority (5-15 Elo, low effort)

4. **Linear history bonus shape**: `MULT*d + OFFSET` instead of `min(MAX, MULT*d)`.
   - Implementation: Change bonus formula; adjust gravity update slightly.
   - Tuning: Multiplier and offset per history type.

5. **4-ply continuation history lookback**: Use ply-1, ply-2, ply-4, ply-6 instead of 1, 2.
   - Implementation: Already indexed, just extend stack offset checks.
   - Tuning: None (structure change).

6. **Aspiration window fail-high reduction**: Reduce depth on repeated fail-highs.
   - Implementation: Track `fail_high_count`, adjust `root_depth` in aspiration loop.
   - Tuning: How much to reduce (they subtract fail_high_count).

7. **Soft stop: eval volatility factor**: Track score changes across depths, reduce time if volatile.
   - Implementation: Store past depth scores, compute volatility metrics.
   - Tuning: Multipliers for 1-depth and 3-depth score deltas.

8. **TT cutoff: history update on cutoff**: Reinforce moves that cause TT cutoff.
   - Implementation: Update quiet/continuation history when TT move causes cutoff.
   - Tuning: Bonus formula.

### Low Priority (marginal, high effort)

9. **Threat accumulator in NNUE**: Would require retraining net with threat features. High Elo (30-50) but large effort.

10. **Factorized threat-aware history**: Separate factorizer + threat-conditioned buckets (from Reckless). Requires new history table structure and tuning.

---

## Technical Debt & Observations

1. **Coda already has most of Astra's core techniques**. The gap is primarily in correction history richness and move ordering sophistication.

2. **Singular extensions**: We disabled them; Astra keeps them. If you retrain, worth revisiting with modest thresholds (their 13/93 are conservative).

3. **History bonus shapes**: Astra's linear `MULT*d + OFFSET` is philosophically cleaner than our `min(MAX, MULT*d)`. Both work, but the difference in tuned parameters suggests theirs may be slightly better calibrated.

4. **Time management**: Astra's multi-factor approach (stability + volatility + node distribution) is more sophisticated. Our node-ratio-only approach is simpler but potentially leaving Elo on the table.

5. **C++ vs Rust**: Astra's C++ is more conservative (simpler NNUE, no threat accumulator), but that's language/effort choice, not a gap in concept.

---

## Recommended Next Steps

1. **Immediate**: Port 4-source correction history and threat bonuses in move ordering. These are low-risk, straightforward, and likely +15-25 Elo combined.

2. **Short-term**: Experiment with material-scaled eval and linear history bonus shape. Easy tuning targets.

3. **Medium-term**: If preparing a major update, retrain net with threat features (threat accumulator). This is the biggest unlock but requires Bullet integration.

4. **Long-term**: Compare multi-factor soft-stop against your current approach. Astra's implementation is well-tuned for typical hardware.

