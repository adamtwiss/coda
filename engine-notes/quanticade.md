# Quanticade Engine Analysis

C engine, top-5 strength. Clean codebase, heavily SPSA-tuned. Interesting architecture choices throughout.

Source: `~/chess/engines/Quanticade/Source/`

## NNUE Architecture

**4-layer network with output buckets** (`nnue.h`, `nnue.c`):
```
(768 * 13 king_buckets -> 1536) x2 perspectives -> L1(16) -> L2(32) -> L3(1) x8 output_buckets
```

Key parameters:
- `INPUT_WEIGHTS = 768`, `L1_SIZE = 1536`, `L2_SIZE = 16`, `L3_SIZE = 32`, `OUTPUT_BUCKETS = 8`
- `KING_BUCKETS = 13` (custom asymmetric bucket map, not uniform)
- `INPUT_QUANT = 255`, `L1_QUANT = 128`, `INPUT_SHIFT = 9`
- `SCALE = 400`

**Quantization scheme** (`nnue.c:106-155`):
- Feature weights: float -> int16 at QA=255 (standard)
- L1 weights: float -> int8 at L1_QUANT=128 (allows VPMADDUBSW kernel)
- L1 bias, L2 weights, L2 bias, L3 weights, L3 bias: all kept as **float**
- Network stored as raw floats on disk, quantized at load time in `transpose()`

**Activation: Paired SCReLU** (`nnue.c:487-520`):
The 1536-wide accumulator is split in half (768+768). For each side:
```c
clipped1 = clip(acc[i], 0, 255);       // first half
clipped2 = min(acc[i + 768], 255);     // second half (no floor clamp!)
result = (clipped1 * clipped2) >> INPUT_SHIFT;  // paired product, packed to int8
```
This is NOT standard SCReLU (x^2). It's a **paired/pairwise** activation where each output neuron is the product of two accumulator values. The first value is clipped [0, QA], the second only has an upper clamp (can go negative). Packed to uint8 via `packus_epi16`.

**Coda comparison**: We use standard SCReLU (v^2) with clamp [0, QA]. Quanticade's paired activation is what Stockfish calls "sqr_clipped_relu" with the pairwise twist. This is a strictly more expressive activation -- each output depends on two accumulator neurons. We should consider adopting pairwise activation for v7.

**L1 -> L2: int8 matmul** (`nnue.c:522-533`):
Uses `dpbusd_epi32` (VPMADDUBSW + VPMADDWD + VPADDD) for the sparse L1 dot product. L1 neurons are uint8, weights are int8. This is the standard efficient kernel.

**L2, L3: Float inference** (`nnue.c:536-582`):
After L1 matmul, converts to float and applies SCReLU (clip [0,1] then square) for both L2 and L3 activations. Uses FMA (`fmadd_ps`) for the float matmul.

**Coda comparison**: We should adopt float inference for L2+ layers. The precision loss from int quantization in narrow layers (16, 32 neurons) is significant. Quanticade shows this is the right approach.

**Finny tables** (`nnue.c:256-310`, `structs.h:71-74`):
Standard Finny table with `[2][13]` = `[mirror][king_bucket]` indexing. Diffs current vs cached bitboards per piece, applies add/sub/move deltas. Same as our implementation.

**King bucket map** (`nnue.c:33-37`):
```
12 12 12 12 12 12 12 12   (rank 8)
12 12 12 12 12 12 12 12   (rank 7)
11 11 11 11 11 11 11 11   (rank 6)
11 11 11 11 11 11 11 11   (rank 5)
10 10 10 10 10 10 10 10   (rank 4)
 8  8  9  9  9  9  8  8   (rank 3)
 4  5  6  7  7  6  5  4   (rank 2)
 0  1  2  3  3  2  1  0   (rank 1)
```
Fine-grained on home ranks (4 buckets for rank 1, 4 for rank 2), coarse for advanced kings. Plus horizontal mirroring detection for refresh (`need_refresh` checks both bucket change AND file-half change).

**Coda comparison**: We use simpler king buckets. Their 13-bucket asymmetric scheme with file-half mirroring is more granular and likely worth adopting.

**Output bucket selection** (`nnue.c:101-104`):
```c
uint8_t pieces = popcount(pos->occupancies[2]);
return (pieces - 2) / 4;
```
8 buckets based on piece count. Standard approach.

**Eval scaling** (`evaluate.c:17-29`):
```c
int phase = 384*knights + 384*bishops + 640*rooks + 1280*queens;
eval = eval * (25600 + phase) / 32768;
```
Material-based eval scaling applied on top of NNUE output. Effectively scales eval by ~0.78 (empty board) to ~1.2 (full material). This is post-NNUE scaling, unusual.

**Coda comparison**: We don't do post-NNUE phase scaling. Worth testing -- it effectively gives the network a material-aware output scaling that the network itself might not learn perfectly.

## Search

### Pruning Features (`search.c`)

**Razoring** (depth <= 7, margin = 266 * depth):
Drops to qsearch if static eval is far below alpha. Standard.

**Reverse Futility Pruning** (depth <= 9):
```
eval >= beta + 26 + 57*depth - 62*improving - 14*opponent_worsening
```
Returns `beta + (eval - beta) / 3` (partial return, not full eval). Uses both `improving` and `opponent_worsening` flags.

**Coda comparison**: The partial RFP return `beta + (eval-beta)/3` is interesting -- dampens inflated static evals. We return `eval` directly. The opponent_worsening factor is novel.

**Null Move Pruning** (`search.c:716-798`):
- Condition: `cutnode && !in_check && eval >= beta && static_eval >= beta - 20*depth + 189 && !only_pawns`
- Reduction: `depth/3 + 6` (capped at depth)
- NMP verification: at depth > 14, does verification search with `nmp_min_ply` mechanism (sets minimum ply for next NMP attempt to `ply + 3*(depth-R)/4`)

**Coda comparison**: The `cutnode` restriction on NMP is interesting -- only does NMP in cut nodes. Standard engines allow it in all non-PV nodes. The verification at depth > 14 with ply-based lockout is Stockfish-style.

**ProbCut** (depth >= 5, margin = 208):
Standard: generate captures, SEE filter at threshold 101, qsearch then shallow search with `depth - 3 - 1`.

**Futility Pruning** (lmr_depth <= 10):
```
static_eval + lmr_depth * 137 + 163 + history_score / 30 <= alpha
```
Uses LMR-reduced depth for the futility check. History-adjusted. Also checks `!might_give_check`.

**Late Move Pruning** (powered, SPSA-tuned):
```
LMP_MARGIN[depth][improving] = base + factor * pow(depth, power)
```
Non-improving: `1.47 + 0.39 * d^1.62`
Improving: `2.83 + 0.87 * d^1.99`
With a `beta + 15` margin for the improving flag. Power-law LMP margins are more flexible than our polynomial.

**SEE Pruning** (depth <= 10):
```
quiet: -41 * depth  (linear)
capture: -29 * depth^2  (quadratic!)
```
History-adjusted: `SEE_MARGIN[depth][quiet] - history_score / 39`

**Coda comparison**: Quadratic SEE margins for captures is unusual and interesting. Most engines use linear for both.

### Singular Extensions (`search.c:998-1040`)

Full implementation with double and triple extensions:
- Depth >= 6, TT depth >= depth - 6
- s_beta = tt_score - depth, s_depth = depth / 2
- Single extension on s_score < s_beta
- Double extension on s_score < s_beta - (0 + 1*pv_node)
- Triple extension on s_score < s_beta - 37 (quiet moves only)
- Multicut: if s_beta >= beta, return s_beta
- Negative extensions: -2 or -3 based on cutnode/pv status

**Coda comparison**: We removed singular extensions as harmful cross-engine. Quanticade keeps them with aggressive triple extensions. Worth revisiting with cross-engine testing if our eval improves.

### Hindsight Reduction (`search.c:681-689`)

Two-part mechanism using parent's reduction value:
1. **Extension**: If parent had large reduction (>= 2905/1024) AND opponent not worsening, extend by 1
2. **Reduction**: If depth >= 2 AND parent had reduction >= 2113/1024 AND combined eval (our static + parent's eval) > 93, reduce by 1

**Coda comparison**: Similar to our hindsight reduction but uses the parent's reduction value rather than comparing evals. The extension half (increase depth when parent reduced a lot and position isn't getting worse) is novel.

### LMR (`search.c:1082-1121`)

Standard log-log formula with many additive adjustments (all SPSA-tuned, expressed in 1024ths):
- Base: `0.73 + ln(d)*ln(m)/1.69` (quiet), `-0.12 + ln(d)*ln(m)/2.53` (noisy)
- Adjustments: +!pv, -history (separate quiet/noisy divisors), -in_check, -was_in_check, +cutnode, -tt_depth_sufficient, -tt_pv, +(tt_pv && cutnode), +(tt_pv && tt_score <= alpha), +cutoff_cnt>3, -improving

After LMR search, does deeper/shallower re-search:
```c
new_depth += (score > best_score + 30 + round(1.94 * new_depth));  // go deeper
new_depth -= (score < best_score + 6);                              // go shallower
```

**Coda comparison**: The `cutoff_cnt` LMR adjustment is interesting -- reduces less at nodes that have many cutoffs (unstable). The deeper/shallower thresholds with depth-dependent deeper margin are well-tuned.

### Aspiration Windows (`search.c:1265-1320`)

Score-dependent window widening:
```c
window += score^2 / 31548;  // wider windows for extreme scores
```
On fail high, reduces depth: `depth - fail_high_count` (only if alpha < 2000).
Window widening: `window += 462 * window / 1024` per iteration (~45% growth).

**Coda comparison**: The score-dependent initial window is novel -- opens wider for positions with extreme evals where windows are more likely to fail. The fail-high depth reduction is aggressive.

### Beta Blending (`search.c:1200-1203`)

At fail-high:
```c
best_score = (best_score * depth + beta) / (depth + 1);
```
Dampens inflated cutoff scores toward beta, weighted by depth.

**Coda comparison**: We have fail-high blending too. Their depth-weighted formula is clean.

### QSearch Beta Blending (`search.c:392-398`)

At QSearch stand-pat cutoff:
```c
if (abs(best_score) < MATE_SCORE && abs(beta) < MATE_SCORE)
    best_score = (best_score + beta) / 2;
```

### Correction History Update Gating (`search.c:1216-1221`)

Only updates correction history when the hash flag direction matches the eval direction:
```c
(hash_flag != LOWER_BOUND || best_score > raw_static_eval) &&
(hash_flag != UPPER_BOUND || best_score <= raw_static_eval)
```
Prevents updating correction history from fail-high/fail-low nodes where the score direction contradicts the correction needed.

**Coda comparison**: This gating is smart -- prevents noisy fail-high/low scores from polluting correction history. We should adopt this.

## Move Ordering (`search.c:209-272`)

### Score computation

**Captures**:
```
score = MVV[target_piece] * 1011 + capture_history * 1039
score /= 1024
see_threshold = -119 - score / 33
score += SEE(pos, move, see_threshold) ? 1e9 : -1e9
```
Dynamic SEE threshold based on MVV + capture history. Good captures (SEE pass) get +1B, bad captures get -1B. The history-adjusted SEE threshold is novel.

**Quiets**:
```
score = quiet_history * 1087
      + conthist(ply-1) * 940
      + conthist(ply-2) * 940
      + conthist(ply-4) * 1075
      + pawn_history * 989
score /= 1024
```
No killers or counter-move heuristic! All ordering is through history tables.

**Coda comparison**: No killers is bold and clean. The multiplier weights on each history component are SPSA-tuned. The MVV values are also SPSA-tuned (not standard piece values).

### Threat-Aware History (`structs.h:119-121`, `history.c:242-248`)

Both quiet and capture history are indexed by `[from_threatened][to_threatened]`:
```c
int16_t quiet_history[2][64][64][2][2];     // [side][from][to][from_threat][to_threat]
int16_t capture_history[12][13][64][64][2][2]; // [piece][captured][from][to][from_threat][to_threat]
```

`is_square_threatened(ss, sq)` checks if the square is attacked by any enemy piece (computed once per node via `calculate_threats`). This gives 4x the history entries -- moves from/to threatened squares have different statistics than moves from/to safe squares.

**Coda comparison**: We don't have threat-aware history. This is a significant information gain -- a knight retreat from a threatened square is very different from an aggressive knight move to a safe outpost. The 4x table size is modest. HIGH PRIORITY to adopt.

### Continuation History (`history.c:284-313`)

Indexed at ply-1, ply-2, and ply-4 (not ply-3). The update uses a **combined total score** for gravity:
```c
int64_t total_score = conthist(ply-1) + conthist(ply-2) + conthist(ply-4);
bonus = bonus - total_score * abs(bonus) / HISTORY_MAX;
```
All three continuation history tables share the same gravity term based on the sum of all three lookups.

**Coda comparison**: We use ply-1 and ply-2. The ply-4 addition captures longer-range move sequences. The shared gravity is interesting -- prevents continuation history from growing too fast when all three agree.

### Pawn History (`history.c:315-324`)

```c
int16_t pawn_history[2048][12][64];  // [pawn_key % 2048][piece][target]
```
Standard pawn hash-indexed history. 2048 buckets.

### History for Search Decisions

Separate history computation for search decisions vs move ordering:
```c
// Move ordering uses MO_* multipliers
// Search pruning uses SEARCH_* multipliers (all 1024 = unity)
```
The search history adds MVV for captures. All multipliers are SPSA-tunable but search ones are currently at unity.

## Correction History (`history.c:151-236`)

Three correction history tables:
1. **Pawn correction**: `[side][pawn_key & 16383]` (16K entries)
2. **White non-pawn correction**: `[side][white_non_pawn_key & 16383]`
3. **Black non-pawn correction**: `[side][black_non_pawn_key & 16383]`

Weighted combination:
```c
correction = pawn_corr * 35 + w_non_pawn_corr * 22 + b_non_pawn_corr * 22;
adjusted = static_eval + correction / 1024;
```

**Fifty-move scaling** applied before correction:
```c
static_eval *= (193 - fifty) / 200;
```

**Coda comparison**: We have pawn correction history. The per-color non-pawn correction history is a Stockfish innovation we should adopt. The fifty-move scaling on eval is simple and effective.

## Time Management (`search.c:180-193`, `uci.c:586-598`)

Three-factor soft limit scaling:
1. **Best move stability** (5 levels, 0-4): scale factors `[2.43, 1.36, 1.10, 0.90, 0.71]` -- unstable best move gets 2.4x more time
2. **Node fraction**: `max(2.41 * (1 - bm_node_fraction) + 0.46, 0.55)` -- if best move uses few nodes, allocate more time
3. **Eval stability** (0-8 levels): `1.20 - stability * 0.044` -- stable evals reduce time

Base time: `time * 0.091 + inc * 0.848`
Hard limit: `time * 0.748`
Soft limit: `min(base * 0.778, hard)`

Scales applied after depth 7:
```c
soft_limit = min(starttime + base_soft * bm_scale * eval_scale * node_scale, max_time + starttime);
```

**Coda comparison**: Well-structured 3-factor time management. The bestmove stability array with 5 discrete levels and SPSA-tuned scale factors is cleaner than our approach. The node-fraction scaling is standard but well-calibrated.

## Performance / SIMD

### AVX2/AVX512 L1 Activation (`nnue.c:487-520`)

The paired activation uses a clever trick to avoid int32 multiply:
```c
shift = slli_epi16(clipped1, 16 - INPUT_SHIFT);  // shift left by 7
mul = mulhi_epi16(shift, clipped2);               // takes upper 16 bits of 16x16->32 product
```
This computes `(clipped1 * clipped2) >> INPUT_SHIFT` entirely in int16 using `pmulhw`, avoiding the expensive int16->int32 widening multiply. Then packs to uint8 via `packus_epi16` with lane fixup.

### AVX2/AVX512 L1 Matmul (`nnue.c:522-533`)

Uses `dpbusd_epi32` (VPMADDUBSW + VPMADDWD + VPADDD chain) for the uint8 * int8 dot product. Weight layout is transposed at load time to enable sequential memory access:
```c
// Transposed layout: [output_bucket][l1 / 4 * L2_SIZE + l2 * 4 + l1 % 4]
```
Groups of 4 int8 weights are packed for the dpbusd kernel.

### Float L2/L3 (`nnue.c:536-582`)

Uses FMA for L2 and L3 matmul. L3 uses chunked accumulation with `reduce_add_ps` for the final sum.

**AVX512 VNNI note** (`simd.h:38-44`):
```c
// On Zen4 VNNI is slower so lets disable it by default
```
They detected that `_mm512_dpbusd_epi32` is slower than the manual 3-instruction sequence on Zen4. Uses manual emulation even when `__AVX512VNNI__` is defined.

**Coda comparison**: The VNNI Zen4 performance note is valuable. We should test this if we add AVX512 support.

### TT Structure (`structs.h:27-39`)

3-slot buckets with 10-byte entries + 2 bytes padding = 32 bytes per bucket:
```c
struct tt_entry {
    uint16_t hash_key;     // 2 bytes (only low 16 bits of hash!)
    uint16_t move;         // 2
    int16_t score;         // 2
    int16_t static_eval;   // 2
    uint8_t depth;         // 1
    uint8_t flag:2, tt_pv:1; // 1 (bitfield)
};
// 3 entries = 30 bytes + 2 padding = 32 bytes
```

Only 16-bit hash verification -- very collision-prone but compact. Index via `(uint128_t)hash * num_entries >> 64` (Fibonacci hashing).

Replacement: depth-based with PV bonus: `depth + 4 + 2*tt_pv > stored_depth`.

**Coda comparison**: Their 16-bit hash key is much weaker than our approach. 3 entries per 32-byte cacheline-aligned bucket is efficient though.

## SPSA Tuning Infrastructure (`spsa.c`)

Extensive SPSA parameter catalog with ~100+ tunable parameters. Every search constant has explicit min/max/rate tuning parameters. Parameters marked as `tunable` are exposed to OpenBench for automated tuning.

Integer and float parameters supported. The `func` pointer allows re-initialization (e.g., `init_reductions()`) when LMR parameters change.

**Coda comparison**: We should build a similar SPSA infrastructure for Coda. The ability to auto-tune 100+ parameters simultaneously is how top engines find optimal configurations.

## Novel Features Worth Adopting

### HIGH PRIORITY
1. **Threat-aware history** (`[from_threatened][to_threatened]` dimensions) -- 4x history entries, captures move context
2. **Non-pawn correction history** (per-color) -- separate correction for white/black non-pawn pieces
3. **Correction history update gating** -- only update when hash flag direction matches
4. **Float L2+ inference** -- avoid quantization precision loss in narrow layers
5. **Fifty-move eval scaling** -- simple `(193 - fifty) / 200` multiplier

### MEDIUM PRIORITY
6. **Pairwise activation** -- paired SCReLU (product of two accumulator halves) instead of standard SCReLU
7. **History-adjusted SEE threshold** in move ordering -- better good/bad capture separation
8. **Power-law LMP margins** -- more flexible than polynomial
9. **Opponent worsening** flag for RFP -- uses `static_eval + parent_eval > 1`
10. **No killers/counter-move** -- pure history-based ordering (simplifies code, may not help)

### LOW PRIORITY / INFORMATIONAL
11. **VNNI Zen4 regression** -- manual dpbusd emulation even with VNNI available
12. **Score-dependent aspiration windows** -- wider for extreme scores
13. **LMR cutoff_cnt adjustment** -- reduce less at unstable nodes
14. **Partial RFP return** -- `beta + (eval-beta)/3` dampens inflated evals
15. **Debug stats infrastructure** -- atomic hit/mean/stdev/correlation tracking

## Architecture Summary

| Feature | Quanticade | Coda/GoChess |
|---------|-----------|--------------|
| FT width | 1536 | 1024 |
| Activation | Paired SCReLU | SCReLU |
| Hidden layers | 16 -> 32 | 16 -> 32 |
| L1 quantization | int8 (QA=128) | int8 (planned) |
| L2+ inference | Float | Integer |
| King buckets | 13 (asymmetric) | Simpler |
| Correction history | Pawn + 2x non-pawn | Pawn only |
| Threat history | Yes (4x tables) | No |
| Killers | No | Yes |
| Continuation history | ply-1,2,4 | ply-1,2 |
| Singular extensions | Full (double+triple) | Removed |
| SPSA tuning | ~100+ params | Manual |
