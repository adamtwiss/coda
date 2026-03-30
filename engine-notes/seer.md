# Seer 2.8.0 - Technical Analysis

Source: `~/chess/engines/seer-nnue/`
Author: Connor McMonigle (C++)
License: GPLv3
Rating: ~3200+ CCRL (strong engine, well above GoChess)

---

## 1. NNUE Architecture

### Network Topology

**HalfKA -> 2x1024 -> 8 -> 8 -> 8 -> 1** with residual concatenation.

```
Input features: 64 * 12 * 64 = 49,152  (HalfKA: all 64 king squares, no bucketing)
Feature transformer: int16, 1024 per perspective, QA=512
fc0: 2048 -> 8   CReLU255 (int8 weights, int16 bias @ QA*Qw=524288), dequantized to float
fc1:    8 -> 8   ReLU, float
fc2:   16 -> 8   ReLU, float  (input = concat(fc1_in, fc1_out) -- residual)
fc3:   24 -> 1   ReLU, float  (input = concat(fc2_in, fc2_out) -- residual)
```

**Coda comparison**: We use v5 1024 direct output or v7 1024->16->32->1x8. Seer's residual concatenation is architecturally distinct -- each layer sees ALL prior intermediate features, not just the previous layer's output. This is cheap because the hidden widths are tiny (8). The concat pattern means fc3 has 24 inputs covering all abstraction levels. Our v7 with L1=16, L2=32 is wider but without residual connections.

### Quantization

| Layer | Weight type | Bias type | Scale |
|-------|-----------|----------|-------|
| FT (shared) | int16 | int16 | 512 |
| fc0 | int8 | int32 | W: 1024, B: 512*1024 |
| fc1-fc3 | float | float | none |

Key: fc0 uses int8 weights with CReLU255 (clamp to [0, 255]) via VPMADDUBSW-style kernel. The dequantization constant is `1/(512*1024)`. Hidden layers fc1-fc3 are entirely float -- no integer quantization for the small layers.

**Coda comparison**: We quantize the FT at QA=255, fc0 at QA_L1=64 (int8). Seer's QA=512 is higher precision. Their float hidden layers avoid all quantization noise in the small layers. Worth considering float L1/L2 for our v7 (Viridithas does this too).

### Feature Reset Cache (Finny Table)

`sided_feature_reset_cache`: per-king-square (64 entries per side) cache of accumulator + piece configuration.

```cpp
// feature_reset_cache.h
struct feature_reset_cache_entry {
  const weights_type* weights_;
  chess::sided_piece_configuration config;  // bitboard snapshot
  aligned_slice<parameter_type, dim> slice_;  // cached accumulator
};
```

On king move: diff cached piece config vs current board, apply only changed features. Per-square caching means returning to a king square reuses the cached state.

**Coda comparison**: GoChess has Finny tables with `[NNUEKingBuckets][mirror]` keying. Seer uses raw king square (64 entries). With our 16 king buckets, multiple king squares map to the same bucket, so we get more cache hits but lower granularity.

### Lazy Evaluation (Dirty Nodes)

`eval_node` is a union of `context` (parent pointer + move) or materialized `eval`. Child nodes are created "dirty" -- NNUE update only happens when `evaluator()` is called.

```cpp
// eval_node.h
struct eval_node {
  bool dirty_;
  union {
    context context_;  // parent + move (lazy)
    eval eval_;        // materialized accumulator
  } data_;

  const eval& evaluator() {  // lazy materialization
    if (!dirty_) return data_.eval_;
    dirty_ = false;
    // ... compute from parent
  }
};
```

This saves NNUE computation for nodes pruned before static eval is needed (NMP verification, SEE-only decisions, etc).

**Coda comparison**: We push/pop accumulator stack on every MakeMove/UnmakeMove. Seer's lazy approach skips NNUE work entirely for pruned nodes. Could be significant NPS gain for our hidden-layer nets where forward pass is expensive.

### Phase-Scaled Output (No Output Buckets)

Instead of output buckets (indexed by material count), Seer uses continuous phase interpolation:

```cpp
eval = phase * 0.7 * prediction + (1 - phase) * 0.55 * prediction;
result = 1024 * clamp(eval, -8.0, 8.0);
```

MG coefficient 0.7, EG coefficient 0.55. Effectively scales down eval in endgames.

**Coda comparison**: We use 8 output buckets. Seer's approach is simpler but less flexible -- output buckets allow the net to learn different piece value relationships per game phase.

### SIMD

Compile-time dispatch via template overload sets (AVX2 > SSSE3 > scalar). Key kernels:

- **FT accumulator updates**: int16 add/sub with 4x unrolled AVX2 (64 elements per iteration)
- **Combined copy+insert+erase**: `add_add_sub` and `add_add_sub_sub` fused operations
- **CReLU255 matmul** (fc0): int8 weights x uint8 clipped input via `_mm256_maddubs_epi16` -> horizontal reduction for 8 outputs
- **Float matmul** (fc1-fc3): 8-wide output unrolling with hadd reduction

The CReLU255 kernel (`int16_crelu255_matrix_vector_product_x32_x8`) packs int16 accumulators to uint8 via `_mm256_packus_epi16`, then uses MADDUBSW for the multiply-accumulate. This is the same approach as Stockfish's CReLU kernel.

**Coda comparison**: Our AVX2 kernels are similar for the FT layer. For fc0 we use VPMADDUBSW too. The fused `add_add_sub` operations (copy parent + insert - erase in one pass) are a nice optimization we should consider -- saves a memory pass vs separate copy + insert + erase.

---

## 2. Search

### Overview

Iterative deepening, PVS, aspiration windows. C++ templates eliminate runtime branching for `is_pv`, `is_root`, `use_tt`. Uses `chess::player_type reducer` instead of explicit `cutNode` bool.

### Aspiration Windows

- Enabled at depth >= 4, initial delta = 21
- On fail-low: `beta = (alpha+beta)/2; alpha = score - delta`
- On fail-high: `beta = score + delta; ++consecutive_failed_high_count`
- **Cumulative fail-high depth reduction**: `adjusted_depth = max(1, depth - consecutive_failed_high_count)`. Each consecutive fail-high reduces depth by 1 more. Most engines reduce by just 1.
- Delta growth: `delta += delta / 3` (~1.33x per iteration)

**Coda comparison**: We use delta=15. The cumulative fail-high reduction is unique and aggressive -- worth testing. Our previous test was bugged (-353 Elo).

### "Reducer" Cut-Node Mechanism

Instead of passing `cutNode` boolean, Seer passes `chess::player_type reducer` (none/white/black):

```cpp
// In ZW search:
const chess::player_type next_reducer =
    (is_pv || zw_depth < next_depth) ? chess::player_from(bd.turn()) : reducer;

// In LMR:
if (is_player(reducer, !bd.turn())) { ++reduction; }
```

When the current side is the "reducer's opponent", they get +1 LMR reduction. This is semantically equivalent to Stockfish's cutNode LMR increase but expressed as "which player's reduced search created this node."

**Coda comparison**: We don't have cutNode. This is a clean way to implement it. The reducer propagation through the tree is more nuanced than a simple boolean toggle.

### Draw Detection

- Material draw, rule-50
- **Cuckoo hash table** for upcoming cycle detection: checks if any single move could reach a previously seen position by looking up Zobrist XOR deltas in a precomputed hash table of all possible single moves.

```cpp
// cuckoo_hash_table.h
// Precomputed: for every (piece, from, to), store hash delta
// At runtime: XOR current hash with each ancestor hash, look up in cuckoo table
// If found, a single move exists that creates a repetition
```

This is more sophisticated than our simple 3-fold repetition check. It detects *upcoming* cycles (positions reachable in 1 move that would create a repetition).

**Coda comparison**: We check exact repetition from the hash history. Cuckoo detects upcoming repetitions that haven't happened yet. Complex to implement but proven in Stockfish/Seer.

---

## 3. Pruning

### Reverse Futility Pruning (SNMP)
- `!is_pv && !excluded && !is_check && depth <= 6 && value > loss_score`
- Margin: `297 * (depth - (improving && !threats)) + (threats ? 112 : 0)`
- **Threat-aware**: adds 112 margin when under threat, subtracts a depth level when improving with no threats
- **Blended return**: `(beta + value) / 2` instead of raw value

**Coda comparison**: We use 85*d improving / 60*d not-improving, depth<=8, no threat awareness, no blending. Note Seer's logit_scale=1024, so 297 ~ 30cp. Our margins are comparable in cp terms. The threat awareness and blended return are worth testing.

### Razoring
- `!is_pv && !is_check && !excluded && depth <= 3`
- Margin: `896 * depth` (~87cp * depth)
- Drops to QSearch at `(alpha, alpha+1)`

### Null Move Pruning
- `!is_pv && !excluded && !is_check && depth >= 3 && value > beta && nmp_valid && has_non_pawn_material`
- `nmp_valid`: requires counter AND follow moves non-null (no NMP at ply 0-1)
- `(!threatened.any() || depth >= 4)`: no NMP at depth 3 under threat
- **TT SEE guard**: `!see_gt(tt_move, 226)` -- blocks NMP when TT best move is a strong capture
- Reduction: `3 + depth/4 + min(4, (value-beta)/242)`

**Coda comparison**: We use `3 + depth/3 + min(3, (eval-beta)/200)`. The TT SEE guard is novel and interesting -- prevents NMP in positions where the TT suggests tactical activity. Our NMP div 200 is well-calibrated per experiments.

### Late Move Pruning (LMP)
- `depth <= 7`, table-based thresholds:
  - Improving: `{0, 5, 8, 12, 20, 30, 42, 65}`
  - Worsening: `{0, 3, 4, 8, 10, 13, 21, 31}`
- Applied as **break** (stops all remaining moves)
- Requires `!bd_.is_check()` (checks legality of resulting position)

**Coda comparison**: We use `3+d^2` with improving adjustment. Their hand-tuned tables are more flexible but harder to optimize.

### Futility Pruning
- `mv.is_quiet() && depth <= 5`, margin: `1544 * depth` (~150cp * depth)
- Per-move **continue** (skips individual moves)

### History Pruning
- `mv.is_quiet() && history_value <= -1397 * depth^2`
- **Quadratic scaling** -- much more aggressive at higher depths than linear

**Coda comparison**: We use -2000*d (linear). Quadratic is steeper: at depth 4, Seer prunes at -22352 vs our -8000. At depth 1, -1397 vs -2000 (Seer is looser). The quadratic shape matches the idea that deep pruning should be more aggressive.

### SEE Pruning
- Quiet: `-54 * depth`, depth <= 9
- Noisy: `-111 * depth`, depth <= 6

### ProbCut
- `!is_pv && !excluded && depth >= 5`
- Beta: `beta + 315`, depth: `depth - 3`
- **Two-stage**: QSearch first, full search only if QSearch passes
- Guards: TT quiet best move blocks ProbCut; TT with sufficient depth and score below ProbCut beta blocks

```cpp
const score_type q_score = -q_search<false>(ss.next(), eval_node_, bd_, -probcut_beta, -probcut_beta + 1, 0);
const score_type probcut_score = (q_score >= probcut_beta) ? pv_score() : q_score;
```

**Coda comparison**: We have ProbCut with margin 170, depth-4, no QS pre-filter. The QS pre-filter is used by Alexandria and Tucano too -- saves the expensive reduced-depth search when QSearch already fails low.

### QSearch Pruning

Three pruning mechanisms beyond standard SEE filtering:

1. **SEE filter**: `!see_ge(mv, 0)` => break (standard)
2. **Delta pruning**: `!is_pv && !is_check && !see_gt(mv, 0) && value + 506 < alpha` => break
3. **Good capture pruning**: `!is_pv && !is_check && !tt_hit && see_ge(mv, 270) && value + 265 > beta` => return beta

The good capture pruning is novel: when we have a very strong capture (SEE >= 270) and we're already close to beta, just cut immediately.

**Coda comparison**: We have SEE filtering and delta pruning. The good-capture pruning is a clean addition for positions where the tactical situation is overwhelmingly favorable.

### IIR
- `!tt_hit && !excluded && depth >= 2` => depth--
- Lower threshold than typical (depth >= 2 vs most engines' 4+)

---

## 4. Extensions

### Singular Extensions
- `!is_root && !excluded && depth >= 6 && tt_hit && mv == tt_move && tt_bound != upper && tt_depth + 3 >= depth`
- Singular beta: `tt_score - 2 * depth`
- Singular depth: `depth/2 - 1`
- **Double extension**: `!is_pv && excluded_score + 166 < singular_beta` => extend by 2
- **Single extension**: `excluded_score < singular_beta` => extend by 1
- **Multi-cut**: `excluded_score >= beta` => return beta immediately
- **Negative extension**: non-PV, none triggered => extend by -1 (reduce)

**Coda comparison**: Our SE is disabled (was -58 to -85 Elo cross-engine). Seer's implementation at depth >= 6 with double extensions and multi-cut is a good reference. However, per our cross-engine testing findings, extensions amplify eval biases 3:1 against diverse opponents. Seer's SE works because their NNUE eval is stronger.

### No Check Extension, No Recapture Extension
Seer has zero other extensions besides singular. This is consistent with our finding that check extensions hurt cross-engine.

---

## 5. LMR (Late Move Reductions)

### Table
Single table (not split by capture/quiet):
```
lmr[depth][played] = 0.4711 + log(depth) * log(played) / 2.5651
```

### Adjustments
| Condition | Adjustment |
|-----------|-----------|
| Improving | -1 |
| Gives check | -1 |
| Creates threat | -1 |
| Is killer | -1 |
| Not tt_pv | +1 |
| Reducer opponent (cut-node) | +1 |
| Quiet history | `clamp(-history/5872, -2, +2)` |

Final reduction clamped to >= 0, reduced depth clamped to >= 1.

### Creates-Threat Reduction
`bd.creates_threat(mv)` checks if the move creates a new attack on a higher-value piece. Moves creating threats get -1 reduction.

**Coda comparison**: We have 7 LMR adjustments. Seer has 7 too but different ones. Key differences: we don't have creates-threat or cut-node reduction. We have separate cap/quiet tables; they use one. Our history divisor is 5000, theirs 5872 (similar). The creates-threat idea is from Koivisto and is eval-agnostic (should transfer well cross-engine per our model).

---

## 6. Move Ordering

### Structure
**Not staged** -- generates ALL moves upfront, scores them, then selection-sort (pick-best). TT move is tried first via lazy initialization: if `first` is set and legal, yield it before initializing the full move list.

```cpp
// move_orderer.cc
move_orderer_iterator::move_orderer_iterator(const move_orderer_data& data) noexcept : data_{data} {
  if (data.first.is_null() || !data.bd->is_legal<mode>(data.first)) {
    stepper_.initialize(data, data.bd->generate_moves<mode>());
  }
  // else: first move will be yielded before initialization
}
```

### Sort Key Hierarchy
Bit-packed 64-bit key with flag bits for priority:
1. **TT move**: separate path (yielded before move gen)
2. **Positive noisy** (SEE > 0): highest priority bit + MVV-LVA score
3. **Killer**: killer bit + history value
4. **Quiet**: history value only
5. **Negative noisy** (SEE <= 0): no priority bits + history value (ranks below quiets)

**Coda comparison**: We use staged generation (TT -> good captures -> killers -> counter -> quiets -> bad captures). Seer's approach is simpler but does all movegen upfront. For our Go implementation, staged generation avoids generating quiets when a capture causes a cutoff.

### History Tables (5 components)

All stored in a `combined<threat_info, pawn_structure_info, counter_info, follow_info, capture_info>` and summed:

| Table | Index | Size | Applies to |
|-------|-------|------|-----------|
| **Threat** | `[threatened_from][from][to]` | 2*64*64 = 8K | Quiets |
| **Pawn structure** | `[pawn_hash & 511][piece][to]` | 512*6*64 = 196K | Quiets |
| **Counter (ply -1)** | `[prev_piece][prev_to][piece][to]` | 6*64*6*64 = 147K | Quiets |
| **Follow (ply -2)** | `[prev2_piece][prev2_to][piece][to]` | 6*64*6*64 = 147K | Quiets |
| **Capture** | `[piece][to][captured]` | 6*64*6 = 2.3K | Captures |

Total: ~500K entries * 2 bytes = ~1MB per side.

### History Update Formula
```
gain = min(400, depth^2)
delta = gain * 32 - clamp(value, -16384, 16384) * |gain| / 512
```

Best move gets positive delta, all tried moves get negative delta. Only quiet and losing-capture moves are tracked.

**Coda comparison**: We have 4 tables (butterfly, counter, follow, capture). Missing: threat history and pawn structure history. Threat history (2x butterfly indexed by from-square-threatened) is the most widely adopted (12+ engines). Pawn history (512-bucket hash) is in Obsidian/Weiss/Seer. Both are high-priority additions.

### Only 1 Killer Per Ply
Seer stores a single killer per ply. No counter-move table -- counter-move information is embedded in the continuation history.

---

## 7. Eval & Correction History

### Evaluation Pipeline
1. NNUE inference -> raw logit
2. Phase scaling: `phase * 0.7 * score + (1-phase) * 0.55 * score`
3. Clamp to [-8, +8] logit, multiply by 1024
4. **Eval cache** lookup (8MB per-thread, avoids NNUE recompute for transpositions)
5. TT adjustment: use TT score if it bounds tighter than static eval
6. **Correction history** (4 tables) applied

### Eval Cache
8MB per-thread cache keyed by position hash. Stores the raw NNUE eval plus an "eval feature hash" (Zobrist of which final-layer outputs are positive). Avoids recomputing NNUE for positions seen via transpositions.

**Coda comparison**: We don't have an eval cache. With our expensive v7 hidden-layer forward pass (~439 kNPS), caching NNUE results could be a meaningful NPS improvement. Simple to implement: hash table of (hash -> score).

### Correction History (4 tables)

`composite_eval_correction_history<4>` with 4 separate 4096-entry tables, each indexed by a different feature hash:

| Table | Hash source | What it captures |
|-------|------------|-----------------|
| Pawn | `lower_quarter(pawn_hash)` | Pawn structure influence on eval error |
| NNUE output | Zobrist of positive final-layer outputs | NNUE's internal confidence signature |
| Continuation | `counter_hash ^ follow_hash` | Move-sequence patterns (ply -1 ^ -2) |
| CC-continuation | `counter_hash ^ ccounter_hash` | Deeper move-sequence (ply -1 ^ -3) |

All 4 corrections are **summed** and added to the raw eval.

### Correction Update
- Only when: `!is_check && best_move.is_quiet()`
- **Bound-aware**: skip if `bound == upper && error >= 0` or `bound == lower && error <= 0`
- Depth-dependent learning rate: `alpha = 16 * (1 - 1/(1 + depth/8))`
  - depth 1: ~2, depth 8: ~8, depth 16: ~11
- EMA update: `correction = correction * (256-alpha) + error * 256 * alpha) / 256`
- Clamped to [-65536, +65536], divided by 256 for correction

**Coda comparison**: We have single pawn-hash correction. Seer has 4 tables. The NNUE output hash is unique to Seer -- it hashes which final-layer neurons are active, creating a "signature" of what the NNUE thinks about the position. The continuation hashes are cheaper to implement (reuse existing move Zobrist keys). Multi-source correction is priority #1 for Coda search improvements.

---

## 8. Transposition Table

### Structure
- Bucketed: 4 entries per cache line (64 bytes / 16 bytes per entry)
- Entry: `key ^ value` XOR encoding (Hyatt lockless trick)
- Fields: bound(2) + score(16) + best_move(16) + depth(8) + gen(6) + tt_pv(1) + was_exact_or_lb(1)
- Index: `hash % bucket_count` (modulo)
- 6-bit generation counter for aging

### Replacement Policy
Within bucket: prefer replacing non-current-gen entries > empty > same-gen-lower-depth. Same hash always replaces.

### TT Entry Merging (Novel)
When storing an upper-bound entry that matches an existing exact/lower-bound entry, copies the best move from the existing entry:

```cpp
if (bound() == bound_type::upper && other.was_exact_or_lb() && key() == other.key()) {
    // Copy best move from the stronger entry
    best_move_::set(value_, other.best_move().data);
}
```

This preserves the best move from deeper searches even when the current search (upper bound) has no useful move.

**Coda comparison**: Our TT uses 4-slot buckets with packed atomics. The merging trick is interesting -- we could adopt it to preserve TT moves when overwriting with fail-low results.

### tt_pv Flag
Stored in TT and propagated: `tt_pv = is_pv || (tt_hit && entry.tt_pv())`. Used in LMR: `if (!tt_pv) ++reduction`. This allows PV information to persist across searches.

**Coda comparison**: We don't store tt_pv. This is a widely adopted feature (Stockfish, many others). Worth adding.

---

## 9. Time Management

### Budget Allocation (Increment)
```
min_budget = (remaining - 150ms + 25*inc) / 25
max_budget = (remaining - 150ms + 25*inc) / 10
Both clamped to: min(4/5 * (remaining - 150ms), budget)
```

### Node-Based Soft Time Scaling
After each iteration: `should_stop = elapsed >= min_budget * 50 / max(best_move_percent, 20)`

Where `best_move_percent = 100 * best_move_nodes / total_nodes`.

Effect: if best move uses 50% of nodes, soft limit is 1.0x. At 20% (floor), 2.5x. At 80%, 0.625x.

**Coda comparison**: Simple and effective. We use an instability factor. The node-concentration approach is more direct.

---

## 10. Lazy SMP

- `worker_orchestrator` manages threads with shared TT only
- Per-thread: search stack, history, eval cache, correction history, NNUE scratchpad, feature reset cache
- Thread depth staggering: `start_depth = 1 + (i % 2)` (simple odd/even alternation)
- Stop via atomic `go` bool

---

## 11. Actionable Items for Coda

### High Priority (proven, high-confidence transfer)

1. **Multi-source correction history** -- 4 tables (pawn + NNUE output + 2 continuation hashes). The continuation hashes are free (reuse move Zobrist keys). This is an accuracy/information improvement (Tier 1 in our transfer model, ~1:1 transfer). Seer, Stockfish, Obsidian, Alexandria all have multi-source correction.

2. **Threat-aware butterfly history** -- `[from_threatened][from][to]`, 2x table size. 12+ engines have this. Again an accuracy improvement.

3. **Pawn structure history** -- `[pawn_hash & 511][piece][to]`. Seer, Obsidian, Weiss have this. Captures pawn-structure-specific move patterns.

4. **tt_pv flag** -- Store and propagate PV-ness in TT. Used for LMR adjustment. Simple, widely adopted.

5. **ProbCut QS pre-filter** -- QSearch before full ProbCut search. Saves nodes. 3+ engines.

### Medium Priority (worth testing)

6. **Creates-threat LMR reduction** (-1 for moves creating threats). Eval-agnostic, should transfer well. Koivisto + Seer.

7. **Eval cache** (8MB per-thread NNUE result cache). With our expensive v7 forward pass, this could give meaningful NPS.

8. **Fused accumulator operations** (`add_add_sub` in one SIMD pass instead of copy + insert + erase). Saves a memory pass over the 1024-element accumulator.

9. **Lazy NNUE evaluation** (dirty nodes, compute only when needed). Saves forward pass for pruned nodes. More impactful with expensive v7 nets.

10. **Cut-node LMR** via reducer parameter or explicit cutNode bool. +1 reduction at expected cut nodes.

### Lower Priority / Risky

11. **Singular extensions** -- Seer's implementation is clean but our cross-engine testing shows extensions amplify eval biases 3:1. Only revisit if we significantly improve NNUE eval quality.

12. **Cumulative aspiration fail-high depth reduction** -- Aggressive, our previous test was bugged. Retry with correct implementation.

13. **Cuckoo upcoming-cycle detection** -- Sophisticated but complex. Low priority given other available gains.

14. **RFP blended return** (`(beta+value)/2`) -- Our test showed -16.7 Elo but may have had different formula.

### Do NOT Adopt

- **Single LMR table** -- We have separate cap/quiet tables that tested well.
- **Non-staged move generation** -- Our staged approach is better for Go (avoids unnecessary movegen).
- **Single killer per ply** -- Harmless to keep 2.
- **All-64-king-square features** -- We use 16 buckets which is the modern standard.

---

## 12. Key Architectural Insights

### What Makes Seer Strong

1. **NNUE quality**: Residual concatenation means the output layer sees all intermediate features. Float hidden layers avoid quantization noise. FinnyTable for efficient king-move recomputes.

2. **Correction history breadth**: 4 tables covering orthogonal position aspects. This is likely the biggest search-side advantage over engines with 1-2 correction tables.

3. **History table richness**: 5 components including threat-aware and pawn-structure tables provide much better move ordering signal than simpler history implementations.

4. **Clean singular extensions**: Activated at depth 6 with double extensions and multi-cut. Works because their NNUE is strong enough that the eval biases don't dominate.

5. **Consistent threat awareness**: Threats appear in RFP margins, NMP guards, history indexing, and LMR adjustments. This thread of tactical safety runs through the entire search.

### Design Philosophy

C++ template-heavy style eliminates runtime branching for PV/non-PV, root/non-root, color. The `reducer` parameter for cut-node tracking is elegant. Union-based lazy NNUE nodes. Composite pattern for multi-table histories with automatic summing and updating. Code is well-organized and readable despite complexity.
