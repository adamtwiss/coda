# Viridithas Chess Engine - Deep Review

Source: https://github.com/cosmobobak/viridithas (v19.0.1)
Author: Cosmo (cosmobobak)
Language: Rust
CCRL Ranking: ~Top 20 (superhuman, ~3400+ Elo)
NNUE: (704x32 -> 2560)x2 -> 16 -> 32 -> 1x8 (pairwise CReLU, SCReLU hidden layers, NNZ-sparse L1)

---

## 1. NNUE Architecture

### Network Topology
- **Input**: 704 features (11 piece types x 64 squares, king planes merged) x 32 king buckets = 22,528 virtual inputs
- **Feature Transformer (L0)**: 704x32 -> 2560 (int16 weights, QA=255)
- **FT Activation**: Pairwise CReLU (split 2560 into two halves of 1280, multiply pairs) -> 2560 u8 outputs (both perspectives)
- **Hidden Layer 1 (L1)**: 2560 -> 16 (int8 weights, QB=64, NNZ-sparse multiplication)
- **L1 Activation**: Squared Clipped ReLU (SCReLU): `clamp(x, 0, 1)^2`
- **Hidden Layer 2 (L2)**: 16 -> 32 (float32)
- **L2 Activation**: Squared Clipped ReLU
- **Output (L3)**: 32 -> 1 (float32, linear - no activation)
- **Output Buckets**: 8 (material-based, linear: `(piece_count - 2) / 4`)
- **Output Scale**: 240
- **HEADS**: 1 (but supports 3-head WDL mode with softmax)

`src/nnue/network.rs:46-58`

### King Buckets (32 buckets = 16 half-buckets x 2 sides)
```
 0  1  2  3 | mirrored
 4  5  6  7 |
 8  9 10 11 |
 8  9 10 11 |
12 12 13 13 |
12 12 13 13 |
14 14 15 15 |
14 14 15 15 |
```
Left half (files a-d) maps to bucket 0-15. Right half (files e-h) maps to bucket 16-31 (separate weights, not just mirrored features).

`src/nnue/network.rs:67-96`

**GoChess/Coda comparison**: Our 16 buckets use the same half-bucket layout but don't separate left/right king positions into different weight sets. Viridithas has 32 effective king buckets (16 x 2 for mirror), giving it finer-grained positional specialization. This means 2x the FT weight count.

### King Plane Merging (MERGE_KING_PLANES = true)
The input layer is only 704 features (11 piece types) instead of 768 (12). The own-king is omitted from each bucket's inputs, since the king position is already encoded by the bucket selection. This saves 64 x 2560 = 163,840 weights per bucket (meaningful memory/cache savings).

`src/nnue/network.rs:42-46, src/nnue/network/feature.rs:36-38`

**GoChess/Coda comparison**: We use 768 inputs (12 piece types). King plane merging is a clean optimization - we should adopt this. Reduces FT size by ~8%.

### Pairwise CReLU (Feature Transformer Activation)
The 2560-element accumulator is split into two halves (1280 each). The first half is clamped to [0, QA], the second half is clamped from above only (no lower clamp). The products are computed via `shift_mul_high_i16` with `FT_SHIFT=9`:

```rust
// Pairwise: left_half * right_half, packed to u8
let cl = clamp(left, 0, QA);      // CReLU on left half
let cr = min(right, QA);           // only upper-clip on right half (key difference!)
output = (cl * cr) >> FT_SHIFT;    // u8 result
```

`src/nnue/network/layers.rs:9,31-50`

The asymmetric clipping (only upper-clip on right half, allowing negative values through the multiply) is a subtle design choice. The right half effectively acts as a learned gate.

**GoChess/Coda comparison**: We use CReLU without pairwise. Pairwise adds expressiveness at no inference cost (just a different activation pattern). This is a major architectural difference.

### NNZ-Sparse L1 Multiplication (Key Performance Feature)
After the pairwise activation produces u8 outputs, many values will be zero. Viridithas tracks which 4-byte chunks are non-zero via a bitmask, then only multiplies the non-zero entries against L1 weights. This is a massive speedup since the 2560-element input to L1 is typically ~30-50% sparse.

Key implementation details:
1. During FT activation, build a mask of which i32-sized chunks have any non-zero bytes
2. Use a pre-computed NNZ lookup table (`NNZ_TABLE`) for fast index extraction from the bitmask
3. L1 multiplication only processes non-zero blocks, unrolled by 4 with auxiliary accumulator

```rust
// For each non-zero block (4 bytes = 4 u8 activations):
let input32 = splat_i32(reinterpret_as_i32(ft_outputs[nnz_idx]));
// Multiply against all L2_SIZE output neurons at once:
for k in 0..NUM_ACCS {
    let weight = load_i8(weights + w_offset + k * U8_CHUNK);
    acc = madd_u8_to_i32(acc, input32, weight);
}
```

`src/nnue/network/layers.rs:191-490`

The auxiliary accumulator (`acc` + `aux`, folded at the end) helps with VNNI instruction latency on modern CPUs (reference: Stockfish PR #6336).

**GoChess/Coda comparison**: We don't do NNZ-sparse L1. With our 1024-element FT going through pairwise/CReLU, sparsity exploitation would be similarly beneficial. This is probably the single biggest NPS optimization we're missing.

### Repermutation (Weight Ordering Optimization)
Viridithas uses a 1280-element permutation table (`REPERMUTE_INDICES`) to reorder FT neurons such that correlated activations are grouped together. This maximizes the number of consecutive zero blocks in the NNZ computation, improving cache locality and reducing the number of NNZ blocks processed.

`src/nnue/network.rs:187-257`

The permutation is computed offline via feature co-occurrence analysis (enabled by the `nnz-counts` feature flag, which tracks which neurons fire together across evaluation positions).

**GoChess/Coda comparison**: We don't do this. It's a pure NPS optimization that could be worth +5-10% NPS when combined with NNZ-sparse L1.

### Quantization Scheme
| Layer | Weight Type | Quantization | Notes |
|-------|-----------|-------------|-------|
| FT (L0) | int16 | QA=255 | Accumulator values in [-32768, 32767] |
| L1 | int8 | QB=64 | Sparse multiplication via NNZ tracking |
| L1 bias | float32 | None | Small enough that float is fine |
| L2 | float32 | None | 16->32, too small to quantize |
| L3 | float32 | None | 32->1, linear output |

Weight clipping at quantization: `1.98 * QA = 504.9` for FT, `1.98 * QB = 126.7` for L1.

`src/nnue/network.rs:412-489`

The L1->L2 conversion uses a carefully computed multiplier:
```rust
const L1_MUL: f32 = (1 << FT_SHIFT) / (QA^2 * QB);  // = 512 / (255*255*64) = 0.000123
```
This converts the integer L1 accumulator back to float, accounting for the pairwise shift and L1 quantization in one step.

`src/nnue/network/layers.rs:12`

**GoChess/Coda comparison**: Very similar to our quantization scheme. The key difference is pairwise activation (FT_SHIFT) changes the scale chain. Our v7 uses QA=255 for FT, QA_L1=64 for L1 - same idea.

### FinnyTable (Bucket Accumulator Cache)
Per-bucket cache of accumulated weights and board state:
```rust
pub struct BucketAccumulatorCache {
    accs: [Accumulator; BUCKETS * 2],       // 32 cached accumulators
    board_states: [[PieceLayout; BUCKETS * 2]; 2],  // per-perspective piece layouts
}
```

On each evaluation, instead of full recompute on king bucket change:
1. Load the cached accumulator for the target bucket
2. Diff cached `PieceLayout` vs actual piece bitboards
3. Apply incremental add/sub updates (typically 3-5 ops vs 30+ full recompute)
4. Save the updated state back to cache

`src/nnue/network.rs:1147-1189`

### hint_common_access (Lazy Accumulator Optimization)
Before using the accumulator, Viridithas attempts to find a previously-computed accumulator in the stack and batch all pending updates into a single operation, rather than applying them one-by-one. Uses a budget system based on piece count to decide whether batching is worthwhile:

```rust
fn try_find_computed_accumulator(&self, pos: &Board) -> Option<usize> {
    let mut budget = pos.state.bbs.occupied().count() as i32;
    // Walk back through acc stack, deducting adds+subs from budget
    // If budget goes negative, it's cheaper to use FinnyTable instead
}
```

`src/nnue/network.rs:1336-1394`

This is called during singular verification searches and TT hit eval reuse to pre-compute accumulators that will be needed repeatedly.

**GoChess/Coda comparison**: We don't have hint_common_access. This is an important optimization for singular extensions where we evaluate the same position multiple times.

### Output Bucket Formula
```rust
const DIVISOR: usize = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS; // = 4
(piece_count - 2) / DIVISOR
```
Linear bucketing: bucket 0 = 2-5 pieces, bucket 7 = 30-32 pieces.

**GoChess/Coda comparison**: We use `min(pieceCount/4, 7)` which is almost identical. Alexandria uses a quadratic formula that gives more middlegame granularity.

### SIMD Support
- **AVX-512**: Full support with VNNI `dpbusd` intrinsics for L1 sparse multiply
- **AVX2**: Fallback via `vpmaddubsw` emulation of u8*i8 multiply-add
- **SSE2**: Basic x86-64 fallback with 128-bit registers
- **NEON (ARM64)**: Full support via `vdotq_s32` for i8 dot products
- **Generic**: Scalar fallback for other architectures

The SIMD abstraction is compile-time selected via `cfg(target_feature)`, not runtime-detected. This means different binaries for different CPU targets.

`src/nnue/simd.rs`, `src/nnue/network/layers.rs:173-563`

**GoChess/Coda comparison**: We do runtime SIMD detection (AVX2 on x86-64), which is more flexible but adds a small branch overhead. Compile-time selection is standard for Rust engines since Cargo supports target-cpu easily.

### Zstd-Compressed Embedded Network
The NNUE weights are embedded as zstd-compressed data and decompressed at startup. The decompressed, permuted weights are cached to a memory-mapped file in the temp directory, shared between processes:

```rust
let weights_file_name = format!(
    "viridithas-shared-network-weights-{}-{}-{}-{:X}.bin",
    ARCH, OS, SIMD_ARCH, checksum
);
```

`src/nnue/network.rs:700-917`

This is a clever optimization: the first instance decompresses and writes to disk; subsequent instances just mmap the file. The checksum in the filename ensures different nets don't collide.

**GoChess/Coda comparison**: We embed the net file directly. Zstd compression would reduce binary size significantly (from ~120MB to ~30MB for our v5 net).

---

## 2. Evaluation

### Static Eval Pipeline
```rust
fn evaluate(t: &mut ThreadData, nodes: u64) -> i32 {
    // 1. Material draw detection (KvK, KBvK, KNvK, etc.)
    // 2. Force accumulator materialization
    // 3. Raw NNUE forward pass
    // 4. Clamp to valid range
}

fn adj_shuffle(t: &ThreadData, raw_eval: i32, clock: u8) -> i32 {
    // 1. Material scaling: eval * (856 + material) / 1024
    // 2. Optimism blending: add side-dependent optimism term
    // 3. 50-move decay: eval * (200 - clock) / 200
}
```

`src/evaluation.rs:110-137, src/search.rs:1894-1913`

### Material Scaling
```rust
let material = (knight_val*N + bishop_val*B + rook_val*R + queen_val*Q) / 32;
let mat_mul = MATERIAL_SCALE_BASE(856) + material;
let opt_mul = OPTIMISM_MAT_BASE(1869) + material;
let raw_eval = (raw_eval * mat_mul + optimism * opt_mul / 32) / 1024;
```

This scales the eval based on total piece material. With all pieces: ~(856+259)/1024 = ~1.09x. With just kings: ~856/1024 = ~0.84x. The optimism term is also material-scaled with a much larger base.

`src/search.rs:1894-1913, src/evaluation.rs:73-84`

### Optimism
Side-to-move optimism is computed at the start of each ID iteration:
```rust
t.optimism[us] = 128 * average_value / (average_value.abs() + OPTIMISM_OFFSET(196));
t.optimism[!us] = -t.optimism[us];
```

This biases the eval toward the side that's currently winning. When blended into the eval via `adj_shuffle`, it makes the engine more aggressive when ahead and more defensive when behind.

`src/search.rs:376-379`

**GoChess/Coda comparison**: We don't have optimism. This is an interesting concept - essentially a self-adjusting contempt that makes the eval asymmetric based on the current search trajectory.

### Correction History (6 separate tables!)
Viridithas uses SIX correction history tables:
1. **Pawn structure** (pawn hash): weight 1890
2. **Non-pawn white** (non-pawn hash for white pieces): weight 1887
3. **Non-pawn black** (non-pawn hash for black pieces): weight 1887
4. **Minor pieces** (minor hash): weight 1292
5. **Major pieces** (major hash): weight 1461
6. **Continuation** (indexed by prev 2 moves' piece+to): weight 1942

All tables are 16,384 entries x 2 sides, stored as int16, capped at +/-1024.

The combined correction is:
```rust
let adjustment = pawn * 1890 + major * 1461 + minor * 1292
    + (white + black) * 1887 + cont * 1942;
(adjustment * 12 / 0x40000) as i32
```

`src/history.rs:176-256, src/historytable.rs:340-435`

**GoChess/Coda comparison**: We have pawn-hash correction history only. Adding non-pawn, minor, major, and continuation correction histories is a major accuracy improvement. The continuation correction history is particularly interesting - it's indexed by `[prev_move.to][prev_move.piece_type][prev_prev_move.to][prev_prev_move.piece_type][side]`.

The `tt_complexity` factor in the update is novel:
```rust
let tt_complexity_factor = ((1.0 + (tt_complexity + 1.0).log2() / 10.0) * 8.0) as i32;
let bonus = diff * depth * tt_complexity_factor / 64;
```

This scales the correction update by how much the TT value disagrees with static eval - larger disagreements produce larger corrections.

---

## 3. Search Architecture

### Aspiration Windows
- **Delta**: 12 (vs our 15)
- **Delta grows with eval**: `delta += average_value^2 / 30155`
- **Fail-low**: Contract beta toward alpha: `beta = midpoint(alpha, beta)`, widen alpha
- **Fail-high**: Widen beta by delta, **reduce root depth by 1** (if score not decisive), increment `reduction` counter
- **Delta growth**: `delta += delta * (43 + 19 * reduction) / 128` (accelerating growth on repeated fails)

`src/search.rs:354-434`

**GoChess/Coda comparison**: The eval-dependent delta widening is interesting - it means wider initial windows in sharp positions where the eval is large. The fail-high depth reduction saves time when the score is running away.

### Singular Extensions (with Multi-Cut and Double/Triple Extensions)
Full singular extension implementation with several sub-features:

```rust
// Singular verification search:
let r_beta = tte.value - depth * 48 / 64;
let r_depth = (depth - 1) / 2;
// Search without the TT move:
let value = alpha_beta(r_depth, r_beta - 1, r_beta, cut_node);

if value < r_beta {
    if !PV && dextensions <= 12 && value < r_beta - DEXT_MARGIN(13) {
        // Double extension
        extension = 2 + (is_quiet && value < r_beta - TEXT_MARGIN(201));
        // (triple extension for very quiet moves that fail extremely low)
    } else {
        extension = 1;  // Normal singular extension
    }
} else if !PV && value >= beta && !is_decisive(value) {
    return value;  // **MULTI-CUT**: another move beats beta, cut the node
} else if tte.value >= beta {
    extension = -3 + PV;  // Light negative extension (effective -3 for non-PV)
} else if cut_node {
    extension = -2;  // Strong negative extension on cut nodes
}
```

`src/search.rs:1389-1444`

The `dextensions` counter (in the search stack) limits double/triple extensions to 12 cumulative across the entire line. This prevents search explosion.

**GoChess/Coda comparison**: We tested singular extensions and found them harmful cross-engine (-41 to -140 Elo). However, Viridithas uses them at the top level of engine strength where they're clearly beneficial. The key differences: (a) much deeper TT depth threshold (`depth >= 6 + ttpv`), (b) multi-cut provides a cutoff path that recovers wasted nodes, (c) negative extensions (-2, -3) actively save nodes when singular fails. Our singular implementation may have been too aggressive.

### Null-Move Pruning with Verification
```rust
// NMP condition: cut_node, previous move exists, depth > 2, eval >= beta
let r = 4 + depth / 3 + min((static_eval - beta) / 174, 4) + (tt_capture exists);
let null_score = -alpha_beta(depth - r, -beta, -beta + 1, false);

if null_score >= beta {
    if depth < 12 && !is_decisive(beta) {
        return null_score;  // Low-depth: trust NMP directly
    }
    // High-depth: verify with a reduced-depth search (NMP banned for STM)
    t.ban_nmp_for(turn);
    let veri_score = alpha_beta(depth - r, beta - 1, beta, false);
    t.unban_nmp_for(turn);
    if veri_score >= beta { return veri_score; }
}
```

`src/search.rs:1112-1167`

Key detail: NMP banning is per-side (2-bit bitmask), not per-node. The ban persists through the verification subtree but is lifted for the opponent, so a mutual NMP scenario at great depth gets verified.

The condition `cut_node && previous_move.is_some()` restricts NMP to expected cut nodes where a real move was played (no NMP after null move).

The TT upper-bound guard is notable: `!matches!(tt_hit, Some(TTHit { value: v, bound: Bound::Upper, .. }) if v < beta)` -- if the TT says the position is likely below beta, skip NMP.

**GoChess/Coda comparison**: Our NMP uses `div 200` for the eval reduction term (vs 174). We don't have verification search, NMP banning, or the TT upper-bound guard. The cut_node restriction is standard.

### Reverse Futility Pruning (RFP)
```rust
if eval - rfp_margin(depth, improving, correction) >= beta {
    return beta + (eval - beta) / 3;
}

fn rfp_margin(depth, improving, correction) -> i32 {
    73 * depth
    - (improving && !can_win_material) * 76
    + correction.abs() / 2
}
```

`src/search.rs:1098-1107, src/search.rs:1708-1712`

The `can_win_material` check is notable: RFP improving margin is only applied when we can't trivially win material. The correction history magnitude is used to widen the margin in volatile positions.

The return value `beta + (eval - beta) / 3` is a blended score rather than just returning beta.

**GoChess/Coda comparison**: Our RFP is similar. The correction-adjusted margin and can_win_material check are refinements we could adopt.

### Razoring
```rust
if alpha < 2000 && static_eval < alpha - 123 - 295 * depth {
    let v = quiescence(alpha, beta);
    if v <= alpha { return v; }
}
```

`src/search.rs:1084-1091`

**GoChess/Coda comparison**: Nearly identical to our implementation (400+d*100 vs 123+295*d). The alpha < 2000 guard prevents razoring near mate scores.

### ProbCut (with Adaptive Depth)
Viridithas has a sophisticated ProbCut implementation with adaptive depth reduction based on QS results:

```rust
// Base: beta + 176 - improving * 78
let pc_depth_base = depth - 3 - (static_eval - beta) / 289;

// After QS confirms a winning capture:
let pc_depth = (pc_depth_base - clamp((value - pc_beta - 50) / 300, 0, 3))
    .clamp(0, depth - 1);
// Higher beta for shallower search:
let ada_beta = pc_beta + (base_pc_depth - pc_depth) * 300;
```

If QS returns a very high value, the verification search can be even shallower but with a proportionally higher beta threshold. If that fails, a full-depth ProbCut search is tried.

`src/search.rs:1190-1284`

**GoChess/Coda comparison**: We don't have ProbCut. This is a well-established technique that could save significant nodes in capture-heavy positions.

### Hindsight Extensions and Reductions
```rust
// Hindsight extension: if parent was heavily reduced and both sides agree position is bad
if prev_reduction >= 1419 && static_eval + prev_static_eval < 0 {
    depth += 1;
}
// Hindsight reduction: if parent was heavily reduced and both sides agree position is quiet
if depth >= 2 && prev_reduction >= 2494
    && prev_static_eval != VALUE_NONE
    && static_eval + prev_static_eval > 128
{
    depth -= 1;
}
```

`src/search.rs:1067-1079`

**GoChess/Coda comparison**: We have hindsight reduction (threshold 200). Viridithas uses much larger thresholds (1419/2494) because their reduction values are in milliplies (x1024). Their hindsight extension is novel - extending when both sides see the position as bad (sum of evals < 0).

### IIR (Internal Iterative Reduction)
Two forms:
1. **PV IIR**: If no TT hit with sufficient depth (`tte.depth + 4 > depth`), reduce by 1 on PV nodes at depth >= 4
2. **Cut-node IIR**: Same condition but on cut nodes, reduce at depth >= 8

`src/search.rs:1171-1181`

### LMR (Late Move Reductions)
Base table: `lm_reduction = ln(depth) * ln(moves_made) * 1024 / 260 + 99` (values in milliplies)

Adjustments (all additive in milliplies):
- **Non-PV**: +987
- **TT-PV**: -1289
- **Cut node**: +1601
- **History-based**: `-stat_score * 1024 / 17017`
- **Killer move**: -775
- **Not improving**: +613
- **TT capture**: +999
- **Gives check**: -1361
- **Correction magnitude**: `-correction.abs() * 448 / 16384`
- **Base offset**: +226

After LMR re-search, do-deeper/do-shallower logic:
```rust
let do_deeper = score > best_score + 32 + 8 * r;
let do_shallower = score < best_score + new_depth;
new_depth += do_deeper - do_shallower;
```

`src/search.rs:1466-1532`

**GoChess/Coda comparison**: Our LMR uses C=1.30 (corresponding to their base/division). They have more LMR adjustments than us: TT-PV, TT-capture, check, and correction magnitude adjustments. The do-deeper/do-shallower re-search logic is more nuanced than our simple re-search.

### Fail-High Score Blending
```rust
// At the end of alpha-beta, if best_score >= beta:
if !is_decisive(best_score) && !is_decisive(alpha) && !is_decisive(beta) {
    best_score = (best_score * depth + beta) / (depth + 1);
}
```

This dampens inflated cutoff scores, weighting them toward beta as depth increases.

`src/search.rs:1578-1579`

**GoChess/Coda comparison**: We have this too. Same concept.

### QS Beta Blending
```rust
// In quiescence, if stand_pat >= beta:
return i32::midpoint(stand_pat, beta);
// And at the end:
if !is_decisive(best_score) && best_score > beta {
    best_score = i32::midpoint(best_score, beta);
}
```

`src/search.rs:647-649, 728-730`

### Value-Difference Based Policy Update (Eval Policy)
Novel feature: after making a move, Viridithas updates the history score of the previous move based on how much the static eval improved/worsened:

```rust
// After computing static_eval at a new node:
let improvement = -(prev_static_eval + static_eval) + EVAL_POLICY_OFFSET(-16);
let delta = clamp(improvement * 227 / 32, -94, 94);
update_history(prev_move, delta);
```

`src/search.rs:1012-1036`

This is essentially learning from the eval: if a quiet move made the eval much worse for the side that played it (improvement is negative), reduce its history score; if it improved things, boost it. This is done continuously during search, not just at fail-high/fail-low.

**GoChess/Coda comparison**: We don't have this. It's a fascinating approach to history heuristic - using eval differences as immediate feedback rather than waiting for search results.

---

## 4. Move Ordering

### Move Picker Stages
1. **TT Move**: Yield the hash move
2. **Generate Captures**: Generate + score all captures
3. **Yield Good Captures**: Yield captures with score >= WINNING_CAPTURE_BONUS (10M), with lazy SEE validation
4. **Yield Killer**: Single killer move (pseudo-legality checked)
5. **Generate Quiets**: Generate + score all quiet moves
6. **Yield Remaining**: Yield all remaining moves (bad captures intermixed with quiets, sorted by score)

`src/movepicker.rs:19-28,58-193`

Captures are scored as: `10M + MVV_SCORE + tactical_history`

MVV scores: `Pawn=0, Knight=2400, Bishop=2400, Rook=4800, Queen=9600, King=0`

SEE is deferred (lazy): captures start with the WINNING_CAPTURE_BONUS and are SEE-tested only when selected for play. If SEE fails, the score is reduced by WINNING_CAPTURE_BONUS, making them sort below quiets.

`src/movepicker.rs:290-310`

### Quiet Move Scoring
Quiets are scored by combining:
1. **Main history**: `threats_history[from_threatened][to_threatened][piece][to]`
2. **Continuation history**: `cont_hist[prev_move][piece][to]` (1-ply and 2-ply back)
3. **Pawn history**: `pawn_hist[pawn_hash % 1024][piece][to]`
4. **Threat-based bonuses**: Per-piece-type bonuses for escaping/entering attacked squares

`src/movepicker.rs:195-287`

The threat-based bonuses are tiered:
- Minor piece escaping pawn attack: +4000 / entering: -4000
- Rook escaping minor attack: +8000 / entering: -8000
- Queen escaping rook attack: +12000 / entering: -12000
- Pawn creating a supported threat: +1000 to +10000 (based on target piece value)

### fast_select (Move Picking Optimization)
Instead of traditional partial insertion sort, Viridithas uses a branchless selection method that packs score and index into a single u64 for max-finding:

```rust
fn fast_select(entries: &[Cell<MoveListEntry>]) -> Option<&Cell<MoveListEntry>> {
    let best = entries.iter().enumerate()
        .map(|(i, e)| to_u64(e.get()) | i as u64)
        .max()?;
    let best_idx = best & 0xFFFF_FFFF;
    entries.get_unchecked(best_idx)
}
```

`src/movepicker.rs:40-56`

---

## 5. History Tables

### Main History (Threat-Aware)
4-dimensional: `[from_threatened][to_threatened][piece][square]`

The from/to threat dimensions mean the engine tracks separate histories for:
- Moving from a safe square to a safe square
- Moving from a safe square to a threatened square
- Moving from a threatened square to a safe square
- Moving from a threatened square to a threatened square

This gives the engine a nuanced understanding of whether quiet moves gain or lose material implicitly.

`src/historytable.rs:135-198`

**GoChess/Coda comparison**: We have basic `[piece][square]` main history without threat awareness. This is 4x the table size but provides much better move ordering in positions with tactical tension.

### Continuation History
Triple indexed: 1-ply back, 2-ply back, and **4-ply back** (a full move ago for the same side).

Each continuation history entry is indexed by `[prev_piece][prev_to][current_piece][current_to]`.

The 4-ply-back continuation history is novel -- most engines only use 1-ply and 2-ply.

`src/history.rs:140-167`

### Pawn Hash History
`[pawn_hash % 1024][piece][square]` -- captures structure-dependent move ordering.

`src/history.rs:68-95`

### Tactical History (Threat-Aware Capture History)
`[to_threatened][capture_type][piece][square]` -- separate histories for captures landing on safe vs threatened squares.

`src/history.rs:98-118`

### History Update Formula
Gravity-based with modulator:
```rust
fn gravity_update_with_modulator<const MAX: i32>(val: &mut i16, modulator: i32, delta: i32) {
    let new = val + delta - modulator * delta.abs() / MAX;
    *val = clamp(new, -MAX, MAX);
}
```

For continuation history, the modulator is the sum of weighted contributions from cont1/cont2/cont4:
```rust
let sum = (cont1_mul * cont1_val + cont2_mul * cont2_val + cont4_mul * cont4_val) / 32;
gravity_update_with_modulator(val, sum, delta);
```

This means the continuation history update is "context-aware" -- moves that already have high continuation history scores get smaller bonuses (gravity pulls them back). This prevents runaway history inflation.

`src/historytable.rs:108-133`

Separate bonus/malus depth-scaled formulas with individual multipliers, offsets, and caps for each history type (main, cont1, cont2, cont4, pawn, tactical). Example for main history:
- Bonus: `min(357 * depth + 226, 2241)`
- Malus: `min(111 * depth + 561, 915)`

Asymmetric bonus/malus (bonus can be much larger) -- this means good moves get boosted more than bad moves get penalized.

`src/historytable.rs:12-83`

---

## 6. Lazy SMP

### Thread Management
Viridithas uses a worker thread pool with synchronization via condition variables:

```rust
pub struct WorkerThread {
    handle: JoinHandle<()>,
    comms: WorkSender,  // SyncChannel + completion signal
}
```

Worker threads are long-lived (created once at startup). Search work is dispatched via `spawn_into()` which sends a closure through the channel. `ReceiverHandle::join()` waits on the completion condvar.

`src/threadpool.rs:1-148`

Main thread and helper threads use the same `iterative_deepening` function, but:
- Main thread (`MainThread`): Has time management, reports UCI info, can stop search
- Helper threads (`HelperThread`): Use `SearchLimit::Infinite`, check `global_stopped` flag

`src/search.rs:266-291`

### Best Thread Selection
After search completes, the engine selects the best thread by comparing completed depth and score:

```rust
// Prefer higher depth, but allow same-depth threads with better scores
if (this_depth == best_depth || this_score >= TB_WIN) && this_score > best_score {
    best_thread = thread;
}
if this_depth > best_depth && (this_score > best_score || best_score < TB_WIN) {
    best_thread = thread;
}
```

`src/search.rs:1916-1950`

### Shared State
Only the TT is shared between threads (via `TTView` reference). All other state is per-thread:
- Board
- NNUE state (accumulators, bucket cache)
- All history tables
- All correction history tables
- Search stack
- Killer moves
- PV arrays

### TT Design
3-entry clusters, 32 bytes per cluster (aligned), using atomic u64 loads/stores:

```rust
struct TTClusterMemory {
    memory: [AtomicU64; 4],  // 4 x 8 bytes = 32 bytes
}

struct TTEntry {
    key: u16,           // 2 bytes
    m: Option<Move>,    // 2 bytes
    score: i16,         // 2 bytes
    depth: u8,          // 1 byte
    info: PackedInfo,   // 1 byte (5-bit age + 1-bit PV + 2-bit bound)
    evaluation: i16,    // 2 bytes
}
// Total: 10 bytes per entry, 3 entries = 30 bytes + 2 padding = 32 bytes
```

`src/transpositiontable.rs:130-172`

Age uses 5 bits (32 generations). The replacement scheme and probing order are in the cluster-level methods.

**GoChess/Coda comparison**: Our TT uses 4-slot buckets with lockless XOR verification. Viridithas uses 3-slot clusters with 16-bit key verification. Our entries are larger (16 bytes) because we store the full 64-bit key for verification.

---

## 7. Time Management

### Time Allocation
```rust
// Base time computation:
let max_time = our_clock * 600 / 1000 - 30;  // Max 60% of clock
let hard_time = our_clock * 46 / 100;         // Hard limit: 46% of clock
let opt_time = our_clock / 24 + our_inc * 94 / 100 - 30;  // Optimal
let opt_time = (opt_time * 73 / 100).min(hard_time);
```

`src/timemgmt.rs:96-129`

### Dynamic Time Multipliers
Four multiplicative factors adjust hard and optimal times each iteration:

1. **Best-move stability** (from Stash engine):
   - Stability 0: x2.50 (best move just changed -- think longer!)
   - Stability 1: x1.20
   - Stability 2: x0.90
   - Stability 3: x0.80
   - Stability 4+: x0.75

2. **Fail-low bonus**: `1.0 + failed_low_count * 340 / 1000` (up to 2 failures, max x1.68)

3. **Forced move detection**: Reduced time for positions where one move is clearly best
   - "Weakly forced" (depth >= 12): x0.627
   - "Strongly forced" (depth >= 8): x0.386

4. **Node-TM subtree size**: `(1.62 - best_move_nodes_fraction) * 140 / 100`
   - If best move has ~50% of nodes: ~1.57x (uncertain, think more)
   - If best move has ~90% of nodes: ~1.01x (very certain, can stop)

`src/timemgmt.rs:360-428`

### Forced Move Detection
At the end of each ID iteration, if the engine detects a forced move (only one clearly best move, others are much worse), it reduces thinking time. This is done via a singular-like search:

```rust
fn is_forced(margin, t, best_move, score, depth) -> bool {
    let r_beta = score - margin;  // margin typically 170 or 400 cp
    // Search without the best move to see if alternatives exist
    alpha_beta(excluded=best_move, depth/2, r_beta-1, r_beta)
}
```

`src/search.rs:1751-1768`

**GoChess/Coda comparison**: We don't have forced move detection, node-based TM, or fail-low time extensions. Our TM is basic allocation without dynamic adjustment. This is a significant area for improvement.

---

## 8. Upcoming Repetition Detection (Cuckoo Hashing)

Viridithas uses Stockfish's Cuckoo hashing technique for upcoming repetition detection. This detects positions that *could* repeat via a single reversible move, without actually making the move:

```rust
// In both alpha_beta and quiescence:
if alpha < 0 && t.board.has_game_cycle(height) {
    alpha = 0;
    if alpha >= beta { return alpha; }
}
```

The Cuckoo tables are pre-computed: 8192 entries of (key, move) pairs for every piece type on every pair of squares. The `has_game_cycle` function walks back through the position history checking if any single move would create a repetition.

`src/cuckoo.rs, src/search.rs:565-570, 823-828`

**GoChess/Coda comparison**: We don't have upcoming repetition detection. This prevents the engine from entering drawn lines and improves play near the 50-move boundary. Worth implementing.

---

## 9. Training / Datagen

### Self-Play Data Generation
Built-in datagen with the `datagen` feature flag:
- Random opening moves: 8 random moves from startpos or DFRC, 0 from book positions
- SEE threshold for random moves: -1000 (allows most moves)
- Node-limited search (soft/hard limits)
- Marlinformat output (ChessBoard from `bulletformat` crate)
- Supports DFRC (Double Fischer Random) for position diversity
- Book support via EPD files
- Syzygy adjudication during datagen

`src/datagen.rs:51-96`

### Quantization Pipeline
The conversion from Bullet's float output to inference-ready format:
1. `UnquantisedNetwork::read()` - Load raw float weights from Bullet
2. `merge()` - Fold factoriser into per-bucket weights, merge L2/L3 bucket-specific + shared weights
3. `quantise()` - Convert FT to int16 (QA=255), L1 to int8 (QB=64), rest stays float
4. `permute()` - Reorder weights for SIMD-optimal layout + NNZ locality

The factoriser merging is interesting: L2 and L3 have both bucket-specific (`l2x_weights`) and shared (`l2f_weights`) components that are summed during merge:
```rust
net.l2_weights[i][bucket][j] = self.l2x_weights[i][bucket][j] + self.l2f_weights[i][j];
```

This is a training technique that encourages shared structure across output buckets while still allowing specialization.

`src/nnue/network.rs:260-489`

---

## 10. Novel / Notable Features

### Threat Maps
The board state pre-computes threat maps at different levels:
- `threats.all`: All squares attacked by the opponent
- `threats.leq_pawn`: Squares attacked by opponent pawns
- `threats.leq_minor`: Squares attacked by opponent pawns or minors
- `threats.leq_rook`: Squares attacked by opponent pawns, minors, or rooks

These are used in:
- Main history (4D threat indexing)
- Quiet move scoring (piece-type-specific threat bonuses)
- SEE pruning (only prune if target square is threatened)
- RFP (`can_win_material` uses threat maps)

**GoChess/Coda comparison**: We don't pre-compute threat maps. These are relatively cheap to compute (one pass over opponent pieces) and provide rich information for move ordering and pruning decisions.

### TT Prefetching
Before making a move, Viridithas prefetches the TT entry that the child position will need:
```rust
t.tt.prefetch(t.board.key_after(m));
```

This is done both in the move loop and in ProbCut. The `key_after` function computes what the zobrist hash will be after the move without actually making it.

`src/search.rs:1305`

**GoChess/Coda comparison**: We don't prefetch TT entries. This is a simple NPS optimization that can be worth +2-5% by hiding memory latency.

### Eval from TT with Bound-Aware Correction
When a TT hit provides a score, Viridithas uses it to correct the static eval based on the bound type:
```rust
if tte.value != VALUE_NONE && match tte.bound {
    Bound::Upper => tte.value < static_eval,
    Bound::Lower => tte.value > static_eval,
    Bound::Exact => true,
    Bound::None => false,
} {
    eval = tte.value;  // Use TT value instead of static eval
}
```

`src/search.rs:963-974`

### Halfmove-Clock Zobrist Keys
The TT key is XORed with a halfmove-clock-dependent Zobrist key:
```rust
let key = board.zobrist ^ HM_CLOCK_KEYS[fifty_move_counter as usize];
```

This ensures that the TT doesn't conflate positions that differ only in their 50-move counter, which is important near the draw boundary.

`src/search.rs:770, src/lookups.rs`

### History Bonus for TT Fail-High
When a non-PV TT cutoff occurs with a quiet move, the engine updates history for that move even though it wasn't searched at this node. This propagates knowledge about good quiet moves through the TT.

`src/search.rs:851-863`

### Fail-Low History Bonus for Inbound Edge
When a node fails low (all-node), the quiet move that led to this node (the parent's move) gets a history bonus, since it led to a position where the opponent couldn't find anything good:

```rust
if flag == Bound::Upper && let Some(mov) = prev_searching && !prev_searching_tactical {
    t.update_history_single(from, to, moved, threats, depth, true);
}
```

`src/search.rs:1616-1629`

### NMP Refutation History Boost
When a quiet move raises alpha after null-move pruning was tried on the parent (parent's searching was None), it gets an extra depth bonus in history updates:

```rust
let nmp = i32::from(!ROOT && prev_searching.is_none());
update_quiet_history(moves, best_move, depth + low + nmp);
```

`src/search.rs:1601-1606`

---

## 11. Parameter Summary

### Key Search Parameters (for tuning comparison)
| Parameter | Viridithas | GoChess/Coda |
|-----------|-----------|--------------|
| Aspiration delta | 12 | 15 |
| RFP margin | 73/depth | ~60+60*depth |
| NMP base R | 4 + depth/3 | Similar |
| NMP eval divisor | 174 | 200 |
| Razoring | 123 + 295*d | 400 + 100*d |
| Futility | 86 + 70*d | 60 + 60*d |
| SEE quiet margin | -62*d | -20*d^2 |
| SEE tactical margin | -28*d^2 | -100*d |
| LMR base/div | 99/260 | C=1.30 |
| History LMR divisor | 17017 | 5000 |
| History pruning margin | -3186 | N/A |
| ProbCut margin | 176 | N/A |
| QS futility | 350 | N/A |
| Output scale | 240 | 362 |
| Material scale base | 856 | N/A |

---

## 12. Actionable Items for GoChess/Coda

### High Priority (likely +10-30 Elo each)
1. **Multiple correction history tables** (pawn, non-pawn, minor, major, continuation) - 6 tables vs our 1
2. **Threat-aware main history** (`[from_threat][to_threat][piece][square]`) - better move ordering
3. **NNZ-sparse L1** for NNUE inference - major NPS gain
4. **TT prefetching** (`key_after` + prefetch before make_move) - +2-5% NPS
5. **Upcoming repetition detection** (Cuckoo hashing) - saves time in drawn positions

### Medium Priority (likely +5-15 Elo each)
6. **King plane merging** - 8% FT size reduction
7. **Pairwise CReLU** for FT activation - more expressive at zero cost
8. **4-ply continuation history** (2 full moves back, same side)
9. **Node-based time management** (subtree size multiplier)
10. **Best-move stability TM** - dynamic thinking time based on PV stability
11. **Forced move TM** - reduce time when position is forced
12. **Eval-policy history updates** (value-difference based)
13. **ProbCut** with adaptive depth

### Lower Priority (refinements)
14. **Pawn hash history** (`[pawn_hash % 1024][piece][square]`)
15. **Threat-based quiet move scoring** (escape/enter attacked square bonuses)
16. **TT halfmove-clock Zobrist keys**
17. **Weight repermutation** for NNZ locality
18. **Zstd-compressed embedded network**
19. **Optimism** (eval asymmetry based on score trajectory)
20. **hint_common_access** for lazy accumulator batching

### Do NOT adopt (our testing shows these are harmful cross-engine)
- Singular extensions (we tested, -41 to -140 Elo cross-engine)
- Double/triple extensions
- SEE pruning tightening
- Capture LMR fine-tuning

Note: Viridithas is strong enough that singular extensions work for them. At our Elo level, the overhead may not be worth the benefit. Revisit if/when we gain 200+ Elo from other improvements.
