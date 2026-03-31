# Halogen Engine Analysis

Top-tier C++ engine. Notable for threat NNUE inputs, sparse L1 matmul, NUMA-aware shared correction history, SMP vote-based best move, and Cuckoo upcoming repetition detection.

Source: `~/chess/engines/Halogen/src/`

---

## 1. NNUE Architecture

### Network Shape
```
(King-bucketed HalfKA + Threat inputs) -> FT[640] -> pairwise SCReLU -> L1[16] -> CReLU+SCReLU -> L2[32] -> CReLU -> L3[1] x8 output buckets
```
Defined in `network/arch.hpp`:
- `FT_SIZE = 640` (320 per perspective after pairwise)
- `L1_SIZE = 16`
- `L2_SIZE = 32`
- `OUTPUT_BUCKETS = 8`
- `FT_SCALE = 255`, `L1_SCALE = 64`, `SCALE_FACTOR = 192.5`

### Dual Input Feature Transformers

**King-bucketed piece-square** (`network/inputs/king_bucket.h`):
- 8 king buckets (asymmetric: 4 for rank 1, 2 for rank 2, 1 for ranks 3-4, 1 for ranks 5-8)
- Horizontal mirroring when king on files E-H
- Total: `8 * 64 * 6 * 2 = 6144` features per perspective
- Standard HalfKA with vertical/horizontal flipping

**Threat inputs** (`network/inputs/threat.h`):
- Encodes (attacker_piece, attacker_sq, victim_piece, victim_sq) on empty-board attack masks
- ~79,856 total threat features with restriction rules:
  - Pawns only threaten pawns/knights/rooks
  - Bishops/rooks don't threaten queens
  - Kings only threaten non-queen pieces
- Incrementally updated via deltas (add/sub threats affected by a move)
- X-ray discovery handled: when a piece vacates a square, sliding piece threats through that square are discovered
- Recalculates one side from scratch when king crosses the center file boundary (file D/E)

**Coda comparison**: We have no threat inputs. This is a significant novel feature. The threat accumulator is a separate int16 array added to the king-bucket accumulator before activation. Weights are int8 (not int16 like king-bucket weights), keeping the total threat weight matrix small. The incremental update is expensive (scanning all changed squares for threats), but the information is very rich. Worth investigating for training but adds complexity.

### Pairwise SCReLU Activation (FT -> L1)

`network/inference.hpp`, `FT_activation()`:
```
For each perspective (STM, NSTM):
  combined = king_bucket[i] + threat[i]  // int16 addition
  a = clamp(combined[0..319], 0, 255)
  b = clamp(combined[320..639], 0, 255)
  output[i] = (a * (b << 7)) >> 16       // pairwise mul, fits in uint8
```
Uses `mulhi_epi16` trick from Alexandria to compute SCReLU in int16 without overflow. Left shift by 7 (or 6 on NEON due to `vqdmulhq_s16` doubling), then mulhi gives `a*b*128/65536 = a*b/512`, which for max values `255*255*128/65536 = 127`. Output is uint8[640].

**Key insight**: Skip the `max_epi16(zero)` for the second operand. Since mulhi preserves sign, negative values become positive after squaring, and `packus` clips them. Saves 2 instructions per pair.

### Sparse L1 Matmul

**This is the most important performance feature.**

During FT activation, non-zero 4-byte nibbles (groups of 4 adjacent uint8 activations) are tracked. A `deposit_nonzero_4xu8_block_indices` function uses:
- AVX512: `_mm512_mask_compressstoreu_epi16` with `cmpgt_i32_mask`
- AVX2/SSE: `cmpgt_i32_mask` -> byte mask -> 256-entry lookup table -> index extraction
- NEON: Similar approach with 4-bit masks

In `L1_activation()`, instead of iterating all 640 FT neurons:
```cpp
for (size_t i = 0; i < sparse_nibbles_size; i++) {
    const auto& nibble_idx = sparse_nibbles[i];
    const auto ft_nibble = *(uint32_t*)(ft_activation + nibble_idx);
    const auto ft_vec = set_u8_from_u32(ft_nibble);  // broadcast 4 bytes
    for (j in L1_SIZE) {
        output_reg[j] = dpbusd_i32(output_reg[j], ft_vec, load(l1_weight[nibble_idx * 4 * L1_SIZE + j * 4]));
    }
}
```
Only processes non-zero blocks. With pairwise SCReLU zeroing ~40-50% of activations, this gives substantial speedup.

**Coda comparison**: We do dense L1 matmul. Sparse L1 is a significant NPS gain, especially with our 1024-wide FT where SCReLU zeroes many neurons. This should be a high priority. The key is grouping neurons into aligned 4-byte blocks and tracking which blocks have any non-zero activation.

### Neuron Grouping for Sparsity

`tools/sparse_shuffle.hpp` contains a `SparseL1Shuffle` class that:
1. Collects activation statistics across many positions
2. Groups neurons into 4-element blocks that maximize co-zero probability
3. Reorders weights at network load time to match the optimal grouping

This means the network weights are permuted so that neurons likely to be zero together are in the same 4-byte block, maximizing the number of all-zero blocks skipped.

**Coda comparison**: Critical optimization we're missing. Without this, sparse L1 would still help but be less effective.

### L1 -> L2: Float with CReLU+SCReLU Dual Output

After int32 accumulation in L1, converts to float and produces **both** CReLU and SCReLU:
```cpp
float_output = int_output * (1.0f / (127 * L1_SCALE));  // = 1/8128
output[i] = clamp(float_output, 0, 1);           // CReLU
output[i + L1_SIZE] = clamp(float_output^2, 0, 1); // SCReLU
```
So L2 receives 32 inputs (16 CReLU + 16 SCReLU). This is clever -- it provides both linear and quadratic signal.

L2 and L3 are fully float. L2 uses CReLU (clamp 0..1), L3 is linear.

**Coda comparison**: We only pass SCReLU to hidden layers. Adding dual CReLU+SCReLU is worth testing. Doubles L2 input width (from L1_SIZE to L1_SIZE*2) at minimal cost since L1 is tiny.

### Material Scaling

Post-NNUE eval is scaled by material on board (`evaluation/evaluate.cpp`):
```cpp
npMaterial = 64*pawns + 576*knights + 560*bishops + 487*rooks + 1654*queens;
eval = eval * (15486 + npMaterial) / 32768;
```
All SPSA-tuned. This compresses eval toward draw as material decreases.

**Coda comparison**: We don't have this. Simple and likely worth a few Elo.

---

## 2. Search

### Full Feature List (from `search.cpp`)

1. **Aspiration windows**: delta = 550/64 (~8.6cp), growth = 86/64 (~1.34x). On fail low, contracts beta: `beta = (alpha+beta)/2`. On fail high, increments `fail_high_count`, reduces depth.

2. **Mate distance pruning**: Standard.

3. **Upcoming cycle detection (Cuckoo)**: `upcoming_rep()` in both search and qsearch. If `alpha < 0` and a legal move leads to repetition, raises alpha to 0. Uses Cuckoo hash tables (`search/cuckoo.h`) -- a fast O(1) check without making moves.

4. **Generalized TT fail-high**: Even if TT depth is insufficient (`depth - 5`), if TT has LOWER_BOUND and `tt_score >= beta + 445`, return `beta + 445`. Novel aggressive TT usage.

5. **Hindsight depth extension**: If `(ss-1)->reduction >= 3` and `eval + (ss-1)->eval < 5`, increment depth. This re-searches positions that turned out quieter than expected.

6. **Razoring** (depth <= 3): Per-depth margins `{0, 391, 610, 757}`. Full razoring (straight to qsearch) at depth 1, or depth 2 with extra margin. Verification qsearch at depth 2-3. If verification fails high at depth <= 3, trim depth by `razor_trim=1`.

7. **Reverse futility pruning** (depth < 9): Quadratic margin `(-140 + 1682*(d-improving) + 73*(d-improving)^2)/64`. Threat-aware: reduces margin by 61 when `active_lesser_threats()` (STM pieces attacked by lesser pieces). Returns `(beta+eval)/2` blend.

8. **Null move pruning**: Reduction = `376/64 + depth*570/4096 + min(306/64, (eval-beta)*73/16384)`. NMP verification search at depth >= 10 (Stockfish enhanced NMP). Disabled when only pawns+king remain or < 5 pieces. Requires `eval > beta + 35`.

9. **ProbCut** (depth >= 3): Beta = `beta + 211`. Qsearch verification before full search (Ethereal idea). Depth = `depth - 5`.

10. **IIR**: If no TT entry at depth > 1, or no TT move at depth > 6, reduce depth by 1.

11. **LMP**: Margin = `(421 + 411*d*(1+improving) + 23*d^2*(1+improving))/64`. Max depth 7.

12. **Futility pruning** (depth < 12): Margin = `(2378 + 689*d + 545*d^2)/64`.

13. **SEE pruning** (depth <= 7): History-adjusted:
    - Loud: `-45*d^2 - history/101`
    - Quiet: `-109*d - history/118`

14. **History pruning**: Skip quiets if `history < -1650*depth - 421`.

15. **Singular extensions** (depth >= 6, tt_depth + 5 >= depth):
    - Triple ext: se_score < sbeta - 32 (non-PV)
    - Double ext: se_score < sbeta - 0 (non-PV)
    - Single ext: se_score < sbeta
    - Multi-cut: if sbeta >= beta, return sbeta
    - Negative ext (-2): if tt_score >= beta, or if cut_node

16. **LMR**: 4-term log formula with depth*move interaction:
    ```
    LMR = -1.805 + 1.427*ln(d) + 2.665*ln(m) - 0.752*ln(d)*ln(m)
    ```
    Adjustments: -1.34 PV, +1.92 cut_node, -0.86 improving, -0.72 loud, -history*2232/16M, +0.56 offset. Minimum = -pv_node (can extend by 1 in PV).

17. **LMR shallower re-search**: After LMR re-search beats alpha, if `score < best_score + 9`, reduce full-depth re-search by 1.

18. **Fail-high score blending**: `score = (score*depth + beta) / (depth+1)` when score >= beta (non-mate).

19. **Prior counter-move bonus**: If node fails low, give history bonus to (ss-1)->move in threat_hist, pawn_hist, and cont_hist at (ss-2), (ss-3), (ss-5).

20. **Qsearch**: LMP at 2 moves. SEE pruning with history-adjusted threshold for quiet evasions. Beta blending: `score = (score+beta)/2`.

### Correction History

Four types, all shared between threads via NUMA-aware allocation:

1. **PawnCorrHistory**: `[2 sides][16384 pawn_hash]`, max=95, scale=85
2. **NonPawnCorrHistory**: `[2 sides][16384 hash]` x2 (one for white non-pawn, one for black non-pawn), max=99, scale=174
3. **ContinuationCorrHistory**: `[2 sides][6 piece_types][64 squares] -> PieceMoveCorrHistory`, max=91, scale=130

Applied in `get_search_eval()` to adjust raw NNUE eval.

**Coda comparison**: We have pawn correction history. They add non-pawn (per color) and continuation correction history. The non-pawn correction is keyed by a separate `non_pawn_key[side]` hash. Continuation correction history is indexed by the previous move's piece-to-square.

---

## 3. Move Ordering

### Stages (`staged_movegen.h`)
```
TT_MOVE -> GEN_LOUD -> GIVE_GOOD_LOUD -> GEN_QUIET -> GIVE_QUIET -> GIVE_BAD_LOUD
```

**Good/bad loud split**: Uses SEE with history adjustment:
```cpp
see_ge(move, -73 - score * 51 / 1024)
```
History-positive captures get more lenient SEE threshold.

**Quiet chunked sorting**: Sorts in chunks of 5 (not full sort), using selection sort.

### History Tables

**Per-thread (local):**
1. **PawnHistory**: `[2][512 pawn_hash][6 piece_types][64 sq]`, max=7394, scale=38
2. **ThreatHistory**: `[2][from_threatened][to_threatened][64 from][64 to]`, max=11400, scale=46
3. **CaptureHistory**: `[2][6 piece][64 sq][6 cap_piece]`, max=19241, scale=51
4. **ContinuationHistory**: Single shared table for 1-ply and 2-ply continuations (plus 4-ply for ordering).
   - For search scoring, uses (ss-1) and (ss-2) cont hist
   - For ordering, additionally uses (ss-4) cont hist

**Quiet search history** (used in SEE pruning threshold, history pruning):
```
pawn_hist + threat_hist + cont_hist[ss-1] + cont_hist[ss-2]
```

**Quiet ordering history** (used in move scoring):
```
pawn_hist + threat_hist + cont_hist[ss-1] + cont_hist[ss-2] + cont_hist[ss-4]
```

**History update formula** (gravity):
```cpp
adjust = clamp(change * scale / 64, -max/4, max/4);
entry += adjust - entry * abs(adjust) / max;
```

**History bonus/penalty** are quadratic in depth:
- Bonus: `(737 - 79*d + 113*d^2) / 64`
- Penalty: `-(1899 - 90*d + 21*d^2) / 64`

**Coda comparison**: Their ThreatHistory is interesting -- indexes by whether from/to squares are under attack by lesser pieces. This is cheap (just a bitboard lookup) and encodes tactical context. We don't have this. Also, their pawn history uses a 512-entry hash (not the standard from/to), which is a compact context signal.

---

## 4. Performance

### Sparse L1 (described above)
The biggest performance feature. Processes only non-zero 4-byte blocks of FT activations.

### Pairwise SCReLU in int16
Avoids float conversion for FT activation. Uses mulhi trick for `a*b >> 16` in int16, producing uint8 output. Very efficient.

### SIMD
Full AVX512/AVX2/SSE4/NEON support:
- Accumulator updates: SIMD add/sub
- FT activation: SIMD clamp + mulhi + pack
- L1 sparse: dpbusd (u8*i8 -> i32) per non-zero nibble
- L2/L3: float FMA, scalar hsum

### TT Prefetch
```cpp
shared.transposition_table.prefetch(Zobrist::get_fifty_move_adj_key(position.board()));
```
Called immediately after apply_move, before any search work. Comment says "~5% speedup".

### Accumulator Table (Finny Tables)
`AccumulatorTable` caches `[KING_BUCKET_COUNT * 2]` entries with piece bitboards for diffing. Standard Finny approach.

### NUMA-Aware Shared History
`PerNumaAllocation<SharedHistory>` allocates one `SharedHistory` per NUMA node. Threads on the same node share correction history, avoiding cross-node latency.

**Coda comparison**: We don't have NUMA awareness. On multi-socket systems this matters. Low priority for now.

---

## 5. Time Management

### Three Factors Combined
```
search_time_usage_scale = node_factor * stability_factor * score_stability_factor
```

**Node factor** (node-based TM):
```
node_tm_base + node_tm_scale * (1 - root_move_effort / total_nodes)
= 0.326 + 2.736 * (1 - effort_fraction)
```
If best move used few nodes (effort_fraction low), node_factor is high -> keep searching.

**Move stability**:
```
1.018 * exp(-0.306 * stable_count) + 0.615
```
Exponential decay: stable_count=0 -> 1.63x, stable_count=5 -> 0.84x, stable_count=10 -> 0.66x.

**Score stability**:
```
0.605 + 1.418 / (1 + exp(-0.062 * (prev_score - curr_score - 16)))
```
Sigmoid: if score dropped, extend time; if stable/improved, save time. Center at score drop of 16cp.

### Soft/Hard Limits
- `should_continue_search`: checks `elapsed < soft_limit * factor * 0.294`
- `should_abort_search`: checks `elapsed >= soft_limit`
- For `go movetime`, soft == hard, never aborts early

### SMP Thread Voting
When half the threads `thread_wants_to_stop`, all threads abort. Prevents one slow thread from wasting time.

### TC Calculation
```
blitz: time/42 + inc*248/256
sudden_death: time/51
repeating_tc: time*96/256
```

**Coda comparison**: We have basic node-based + stability TM. The score stability sigmoid and thread voting for stop are novel. The three-factor product is cleaner than additive approaches.

---

## 6. SMP Best Move Selection

`get_best_root_move()` in `data.cpp`:

1. If any thread found a win, take the shortest win (highest score)
2. If any thread found a loss, take the shallowest loss (lowest score)
3. Otherwise: **weighted popular vote** across threads:
   ```
   weight = (depth + 2.126) * (score_diff + 1.461) + 356.7
   ```
   where score_diff is relative to the minimum thread score. Highest total weight wins. Among threads playing that move, pick deepest/highest-scoring.

**Coda comparison**: We use thread 0's result. This voting system is much better for SMP scaling. Worth implementing.

---

## 7. Novel / Notable Features

### Threat NNUE Inputs
The most distinctive feature. A separate accumulator tracks which pieces attack which other pieces, with restricted threat types (pawns don't threaten queens, etc.). Incrementally updated with x-ray discovery. Uses int8 weights (vs int16 for king-bucket), keeping memory reasonable despite ~80K features.

### Lesser Threats in Search
`BoardState::lesser_threats[piece_type]` tracks which squares are attacked by pieces of lesser value. Updated on every move. Used in:
- **RFP margin**: reduced when pieces are under attack
- **ThreatHistory**: indexes by whether from/to are threatened
- **Legality**: king can't move to squares attacked by lesser pieces

### Upcoming Cycle Detection (Cuckoo)
Used in both search and qsearch. When `alpha < 0`, checks if any legal move leads to a position in the hash history. If yes, raises alpha to 0 (draw score). Avoids wasting time on positions that are draws by repetition.

Works with singular extension exclusion: `upcoming_rep(distance_from_root, ss->singular_exclusion)`.

### Generalized TT Fail-High
Even when TT depth is insufficient, if the TT has a lower bound score significantly above beta (`+445`), cut. Requires depth within 5 of needed.

### 50-Move Rule Eval Scaling
```cpp
eval = eval * (279 - fifty_move_count) / 231
```
Compresses eval toward 0 as 50-move counter increases. Simple and effective.

### TT Entry: 10 bytes, 3-way bucket
Each entry is 10 bytes (key16 + move + score + static_eval + depth + meta), bucket is 32 bytes aligned (3 entries + 2 bytes padding). Generation stored in 6 bits of meta byte.

---

## 8. Actionable Items for Coda

### High Priority
1. **Sparse L1 matmul**: Track non-zero 4-byte blocks during FT activation, skip zero blocks in L1 matmul. With our 1024-wide FT + SCReLU, ~40-50% sparsity expected. Estimate 30-50% NPS gain on L1.
2. **Neuron grouping optimization**: Collect activation statistics, reorder FT weights so co-zero neurons are adjacent. Maximizes sparse L1 benefit.
3. **CReLU+SCReLU dual L1 output**: Feed both clamp(x,0,1) and clamp(x^2,0,1) to L2. Doubles L2 input width at trivial cost (L1 is 16 neurons). Needs training support.
4. **Non-pawn correction history**: Add per-color non-pawn key hashing and correction tables. Proven in multiple engines.
5. **Continuation correction history**: Correction indexed by previous move's piece-to-square.
6. **Material scaling**: Simple post-NNUE eval scale by remaining material. Easy to implement.

### Medium Priority
7. **SMP vote-based best move**: Weighted popular vote across threads instead of thread-0 only. Better SMP scaling.
8. **Score stability time management**: Sigmoid function on score drop. Cleaner than threshold-based approaches.
9. **ThreatHistory**: History table indexed by whether from/to squares are under attack by lesser pieces.
10. **Pawn hash history**: History indexed by pawn structure hash instead of from/to.
11. **Generalized TT fail-high**: Use insufficient-depth TT entries for cutoffs when score is far above beta.
12. **Prior counter-move bonus**: On fail-low, give bonus to parent's move in multiple history tables.

### Low Priority / Investigate
13. **Threat NNUE inputs**: Major architectural change requiring training support. ~80K features with incremental update. High potential but high effort.
14. **Upcoming cycle detection (Cuckoo)**: Raises alpha to draw when repetition is available. O(1) check.
15. **NUMA-aware shared history**: Relevant only for multi-socket systems.
16. **RFP threat adjustment**: Reduce RFP margin when pieces are under attack by lesser pieces.
17. **50-move eval scaling**: Compress eval toward 0 based on fifty-move counter.
