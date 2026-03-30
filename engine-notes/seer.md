# Seer Chess Engine - Crib Notes

Source: `~/chess/engines/seer-nnue/`
Author: Connor McMonigle
NNUE: HalfKA (64 king squares) -> 2x1024 -> 8 -> 8 -> 8 -> 1 (CReLU255 + ReLU, float hidden layers, residual concat)
Rating: #10 in our RR at +185 Elo (293 above GoChess-v5)

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. C++ templates for color-dependent logic (no runtime branching on side-to-move). Supports Lazy SMP via `worker_orchestrator` with shared TT. Uses a `player_type reducer` parameter to implement a cut-node-like mechanism without explicit cut-node tracking.

### Iterative Deepening
- Depth 1 to 128 (`max_depth`)
- Thread 0 starts at depth 1, thread 1 at depth 2, alternating: `start_depth = 1 + (i % 2)`
- Checks soft/hard time limits via callbacks (`on_update` every 512 nodes, `on_iter` after each iteration)

### Aspiration Windows
- Enabled at depth >= 4 (`aspiration_depth = 4`)
- Initial delta: 21 (`aspiration_delta = 21`)
- On fail-low: center window: `beta = (alpha + beta) / 2`, `alpha = score - delta`
- On fail-high: `beta = score + delta`, increment `consecutive_failed_high_count`
- Depth reduced by `consecutive_failed_high_count` on each aspiration re-search (not just -1, but cumulative)
- Delta growth: `delta += delta / 3` (~1.33x per iteration)
- No full-width fallback -- keeps growing delta until score falls in window

### Draw Detection
- Trivially drawn: insufficient material check
- Rule50: `half_clock >= 100`
- **Upcoming cycle detection** (Cuckoo hashing): `upcoming_cycle_exists()` checks if any single move could reach a previously seen position. If cycle detected and `draw_score >= beta`, returns draw. Otherwise narrows alpha to draw.
- Used in both main search and QSearch -- more sophisticated than simple repetition detection

### "Reducer" Cut-Node Mechanism
- Instead of passing a `cutNode` boolean, passes a `chess::player_type reducer` (none, white, or black)
- At ZW search, if PV or depth was reduced (LMR), sets `next_reducer = current_turn`; otherwise propagates parent's reducer
- In LMR: `if (is_player(reducer, !bd.turn())) { ++reduction; }` -- this is equivalent to Stockfish's `cutNode` LMR increase
- Novel framing: "if our opponent is the reducing player, an errant fail low will, at worst, induce a re-search"

---

## 2. Pruning Techniques

### Reverse Futility Pruning (Static Null Move Pruning / SNMP)
- Conditions: `!is_pv && !excluded && !is_check && depth <= 6 && value > loss_score`
- Margin: `297 * (depth - (improving && !threats)) + (threats ? 112 : 0)`
  - depth 1 not-improving: 297, depth 6 improving: 1485
  - **Threat-aware**: adds 112 when threats exist, subtracts one depth level when improving AND no threats
- **Score blending on return**: returns `(beta + value) / 2` instead of raw value
- Compare to ours: we use 85*d improving / 60*d not-improving, depth<=8, no threat awareness, no blending

### Razoring
- Conditions: `!is_pv && !is_check && !excluded && depth <= 3`
- Margin: `896 * depth` (depth 1: 896, depth 2: 1792, depth 3: 2688)
- Drops to QSearch with `(alpha, alpha+1)` window; returns if score <= alpha
- Compare to ours: we use 400+100*d (much tighter margins -- depth 1: 500, depth 2: 600, depth 3: 700)

### Null Move Pruning (NMP)
- Conditions: `!is_pv && !excluded && !is_check && depth >= 3 && value > beta && nmp_valid && has_non_pawn_material`
- **Extra guards**:
  - `nmp_valid`: requires both counter move and follow move to be non-null (prevents NMP at ply 0 and 1)
  - `(!threatened.any() || depth >= 4)`: no NMP at depth 3 when under threat
  - **TT move SEE guard**: if TT has a move, only allow NMP if `!see_gt(tt_move, 226)` -- blocks NMP when TT suggests a very good capture exists. This is unique.
- Reduction: `3 + depth/4 + min(4, (value - beta) / 242)`
- Returns raw nmp_score (no clamping to beta)
- Compare to ours: we use `3 + depth/3 + min(3, (eval-beta)/200)`. Their R is slightly lower at shallow depths (depth/4 vs depth/3) but allows higher eval bonus (min 4 vs 3). The TT SEE guard is novel.

### Internal Iterative Reduction (IIR)
- Condition: `!tt_hit && !excluded && depth >= 2`
- Reduces depth by 1
- Lower threshold than most (depth >= 2 vs typical 4+)

### Late Move Pruning (LMP)
- Conditions: `!is_root && idx >= 2 && best_score > loss_score && !bd_.is_check() && depth <= 7`
- **Table-based thresholds** (hard-coded arrays):
  - Improving: `{0, 5, 8, 12, 20, 30, 42, 65}`
  - Worsening: `{0, 3, 4, 8, 10, 13, 21, 31}`
- This is a **break** (stops all remaining moves)
- Compare to ours: we use `3+d^2` with improving adjustment. Their table goes up to depth 7 with hand-tuned values.

### Futility Pruning
- Conditions: `mv.is_quiet() && depth <= 5`
- Margin: `1544 * depth` (depth 1: 1544, depth 2: 3088, depth 3: 4632)
- Applied per-quiet-move as a **continue** (skips individual moves)
- Compare to ours: we use 80+d*80 (much tighter). Their margins are extremely wide -- appears calibrated for WDL/logit-scale eval.
- **Note**: Seer uses `logit_scale = 1024` for evaluation, so all scores are scaled differently. Their 1544 in logit space ~ 150cp in classical space.

### History Pruning
- Conditions: `mv.is_quiet() && history_value <= threshold`
- Threshold: `-1397 * depth^2` (depth 1: -1397, depth 2: -5588, depth 3: -12573)
- **Quadratic scaling** -- steeper than linear. Much more aggressive at higher depths.
- Compare to ours: we use -2000*depth (linear).

### SEE Pruning
- **Quiet moves**: depth <= 9, threshold `-54 * depth`
- **Noisy moves**: depth <= 6, threshold `-111 * depth`
- Compare to ours: quiet -20*d^2, noisy similar. Their quiet pruning extends deeper (9 vs typical 7).

### QSearch Pruning
- **Standard SEE filter**: non-check captures with SEE < 0 are pruned (break)
- **Delta pruning**: `!is_pv && !is_check && !see_gt(mv, 0) && value + 506 < alpha` => break
  - Margin 506 (in logit scale, ~50cp classical equivalent)
- **Good capture pruning** (unique): `!is_pv && !is_check && !tt_hit && see_ge(mv, 270) && value + 265 > beta` => return beta
  - This is the "good-delta" from Minic -- when we have a very strong capture AND we're already above beta, just cut
  - SEE threshold 270 ensures the capture is decisively winning
  - Score margin 265 ensures we're already close to or above beta
- Compare to ours: we have SEE filtering and delta pruning at 240. The good-capture pruning is novel.

### ProbCut
- Conditions: `!is_pv && !excluded && depth >= 5 && !(tt has quiet best move) && !(tt depth >= probcut_depth && tt score < probcut_beta)`
- Beta: `beta + 315`
- Search depth: `depth - 3`
- **Two-stage**: runs QSearch first (`-q_search(..., -probcut_beta, -probcut_beta+1)`), then only does full `pv_search` if QSearch passes
  - This is the ProbCut QS pre-filter from our experiment queue (2 engines: Alexandria, Tucano, now 3 with Seer)
- Compare to ours: we use margin 170, gate 85. No QS pre-filter.

---

## 3. Extensions

### Singular Extensions
- Depth guard: `depth >= 6` (`singular_extension_depth`)
- Conditions: `!is_root && !excluded && tt_hit && mv == tt_move && tt_bound != upper && tt_depth + 3 >= depth`
- Singular beta: `tt_score - 2 * depth`
- Singular depth: `depth / 2 - 1`
- **Double extension**: if `!is_pv && excluded_score + 166 < singular_beta` => extend by 2 (not 1)
- **Single extension**: if `excluded_score < singular_beta` => extend by 1
- **Multi-cut**: if `excluded_score >= beta` => `multicut = true; return beta`
- **Negative extension**: if non-PV and none of the above triggered => `-1` extension (reduce by 1)
- Compare to ours: we use depth >= 10, margin 3*depth. Their activation is much earlier (depth 6 vs 10), tighter margin (2*depth vs 3*depth), has double extensions and multi-cut. **Our SE is known to be problematic (-58 to -85 Elo). Seer's implementation is a reference for how to do it right.**

### No Check Extension
- Unlike many engines, Seer does NOT extend for checks
- No recapture extension either

---

## 4. LMR (Late Move Reductions)

### Table Initialization
Single LMR table (not split by capture/quiet):
- `0.4711 + log(depth) * log(played) / 2.5651`
- Base ~0.47, divisor ~2.57
- Compare to ours: we use separate tables (cap C=1.80, quiet C=1.50). Seer uses a single table with lower base.

### Application Conditions
- `!is_check && (mv.is_quiet() || !see_ge(mv, 0)) && idx >= 2 && depth >= 2`
- Applies to ALL moves that are quiet OR losing captures (SEE < 0)
- Single table covers both types

### Reduction Adjustments
- `-1` for improving
- `-1` if move gives check (`bd_.is_check()`)
- `-1` if move creates a threat (`bd.creates_threat(mv)`) -- unique, see below
- `-1` if move is the killer
- `+1` if not tt_pv (stored in TT, propagated)
- `+1` if `is_player(reducer, !bd.turn())` (cut-node equivalent)
- History-based: `clamp(-history_value / 5872, -2, +2)` -- bounded [-2, +2]
- Final: clamped to >= 0, reduced depth clamped to >= 1

### Notable: Creates-Threat Reduction
- `bd.creates_threat(mv)` checks if a move creates a new attack on a higher-value piece
- Moves that create threats get -1 reduction (searched deeper)
- This is the "new threats in LMR" idea from Koivisto, also in our experiment queue

---

## 5. Move Ordering

### Structure
Not staged -- generates ALL moves, scores upfront, selection sort (pick-best). Same approach as Midnight. TT move is handled specially: it's tried first (before move generation) via the iterator's lazy initialization pattern.

### Score Hierarchy (encoded in bit-packed sort key)
The sort key uses bit flags for priority levels:
1. **TT move**: tried first via `set_first()` -- no score needed, separate path
2. **Positive noisy** (SEE > 0): `positive_noisy` bit set + `mvv_lva_key` as value
3. **Killer move**: `killer` bit set + history value
4. **Quiet moves**: history value only
5. **Negative noisy** (SEE <= 0): no priority bits + history value (ranks below quiets)

### History Tables (5 component tables)
All stored in a single `combined` type and summed for move scoring:

1. **Threat history**: `[2_threat_states][64_from][64_to]` -- indexed by whether `from` square is threatened
   - This is **threat-aware butterfly history** (from SUMMARY.md item #5, 12-engine consensus)
   - Only uses `from_threatened` (1 bit), not `to_threatened`
2. **Pawn structure history**: `[512_pawn_states][6_pieces][64_to]` -- indexed by pawn hash & 511
   - This is **pawn history** (from SUMMARY.md item #22, Obsidian/Weiss)
   - 512-bucket hash table keyed on pawn structure
3. **Counter-move history**: `[64_to0][6_piece0][64_to1][6_piece1]` -- continuation history ply -1
4. **Follow-up history**: `[64_to0][6_piece0][64_to1][6_piece1]` -- continuation history ply -2
5. **Capture history**: `[6_piece][64_to][6_captured]` -- for noisy moves

### History Update
- Bonus: `min(400, depth^2)` -- capped at 400 (vs typical depth^2+depth-1 uncapped)
- Gravity formula: `delta = gain * 32 - clamp(value, -16384, 16384) * |gain| / 512`
- All 5 tables updated simultaneously for best move (positive bonus) and tried moves (negative bonus)
- **Malus only for quiet/losing-capture moves**: `if (mv.is_quiet() || !see_gt(mv, 0))` guards both best move update AND tried-move tracking

### Context
- `follow` = move 2 plies ago (for follow-up history)
- `counter` = move 1 ply ago (for counter-move history)
- `ccounter` = move 3 plies ago (used for correction history feature hash)
- Only 1 killer per ply (not 2)
- No explicit counter-move table -- counter-move info is in the continuation history

---

## 6. Eval & Correction History

### Evaluation Pipeline
1. NNUE inference -> raw score (in logit scale, 1024 units per logit)
2. Phase-adjusted: `phase * 0.7 * score + (1-phase) * 0.55 * score` (MG weighted higher)
3. Clamped to [-8, 8] logit range, scaled by 1024
4. **Eval cache**: 8MB per-thread cache of NNUE evals, keyed by position hash (avoids recomputing NNUE for same position in different search paths)
5. TT-adjusted: if TT has upper bound below static eval, use TT score (and vice versa)
6. **Correction history** applied: `static_value += correction_for(feature_hash)`

### Correction History (4 tables)
Seer uses `composite_eval_correction_history<4>` -- 4 separate correction tables indexed by different feature hashes:

1. **Pawn hash** (`pawn_feature_hash`): `lower_quarter(bd.pawn_hash())`
2. **NNUE output feature hash** (`eval_feature_hash`): Zobrist hash of which NNUE final-layer outputs are positive
3. **Continuation feature hash** (`cont_feature_hash`): `lower_quarter(counter_move_hash ^ follow_move_hash)` -- captures move sequence context
4. **CC-continuation feature hash** (`ccont_feature_hash`): `lower_quarter(counter_move_hash ^ ccounter_move_hash)` -- deeper move sequence (ply -1 ^ ply -3)

Each table has 4096 entries. All 4 corrections are summed.

Update:
- Only when: `!is_check && best_move.is_quiet()`
- **Bound-aware**: skips update if `bound == upper && error >= 0` or `bound == lower && error <= 0`
- Alpha (learning rate) is depth-dependent via lookup table: `alpha = 16 * (1 - 1/(1 + depth/8))`
  - depth 1: ~2, depth 8: ~8, depth 16: ~11, depth 32: ~13
- Update: `correction = (correction * (256 - alpha) + error * 256 * alpha) / 256`
- Clamped to [-65536, 65536], divided by 256 for actual correction

**Compare to ours**: We have single pawn-hash correction. Seer has 4 tables with unique hash sources. The NNUE output hash is novel -- it captures which aspects of the position the NNUE thinks are important. The continuation hashes capture move-sequence context.

---

## 7. Transposition Table

### Structure
- Bucketed: `cache_line_size / sizeof(entry)` entries per bucket (64 / 16 = 4 entries per bucket)
- Entry: 16 bytes (key XOR value for lockless access)
- Fields packed via bit ranges: bound (2), score (16), best_move (16), depth (8), gen (6), tt_pv (1), was_exact_or_lb (1)
- Index: `hash % bucket_count` (modulo, not power-of-2)
- **Generation-based aging**: 6-bit generation counter incremented per search

### Lockless Access
- Key stored as `key ^ value` (Hyatt XOR trick)
- `__attribute__((no_sanitize("thread")))` on insert/find
- No atomics -- relies on XOR verification for data integrity

### Replacement Policy
Within bucket, find `to_replace`:
- First priority: same hash (always replace)
- Then pick worst entry: prefer non-current-gen > empty > same-gen-lower-depth

For insertion: replace if `exact node || different hash || new_depth + 2 >= old_depth`
- The `+2` offset means entries are replaced fairly aggressively

### TT Entry Merging
Novel: when storing an upper-bound entry that matches an existing exact/lower-bound entry, it copies the best move from the existing entry. This preserves the best move from deeper searches even when the current search doesn't improve on it.

### TT PV Flag
- `tt_pv` bit stored in TT entries
- At search start: `tt_pv = is_pv || (tt_hit && entry.tt_pv())`
- Used in LMR: `if (!tt_pv) { ++reduction; }` -- non-PV moves get more reduction
- Persists across searches via TT

---

## 8. NNUE Architecture

### Network Topology
**HalfKA -> 2x1024 -> 8 -> 8 -> 8 -> 1** with residual concatenation

- Input features: `64 * 12 * 64 = 49152` (HalfKA: 64 king squares x 12 piece types x 64 squares)
  - Uses all 64 king squares (no bucketing, no mirroring reduction)
  - Features are mirrored by XOR with 56 for black perspective
- Feature transformer: int16 accumulator, 1024 dimensions per perspective
- **First hidden layer (fc0)**: `2048 -> 8` with CReLU255 activation
  - Input is concat of white/black accumulators (2x1024 = 2048)
  - CReLU255: `clamp(x, 0, 255)` -- this is the quantized clipped ReLU
  - int8 weights, int16 biases, output dequantized to float
  - Separate pre-flipped weight matrices for white/black POV (`white_fc0`, `black_fc0`)
- **Second hidden layer (fc1)**: `8 -> 8` with ReLU, float
- **Third hidden layer (fc2)**: `16 -> 8` with ReLU, float
  - Input is concat of fc1 input (8) and fc1 output (8) = **residual connection**
- **Output layer (fc3)**: `24 -> 1` with ReLU, float
  - Input is concat of fc2 input (16) and fc2 output (8) = **residual connection**
  - 24 total features feed into single output

### Residual Concatenation Pattern
```
ft_output (2048) -> fc0 -> x1 (8)
concat(x1, fc1(x1)) -> x2 (16)
concat(x2, fc2(x2)) -> x3 (24)
fc3(x3) -> scalar
```
Each layer's input is the concatenation of the previous input and output. This means all intermediate features are available to the final layer.

### Quantization
- Feature transformer: int16, scale 512
- fc0 weights: int8, weight scale 1024, bias scale 512*1024
- fc0 output: dequantized to float via `1/(512*1024)`
- fc1/fc2/fc3: float (no quantization)

### Phase-Scaled Output
- Not output buckets -- instead uses game phase to interpolate:
  - `eval = phase * 0.7 * prediction + (1-phase) * 0.55 * prediction`
  - MG coefficient 0.7, EG coefficient 0.55
  - Effectively scales down in endgame
- Final: `1024 * clamp(eval, -8, 8)`

### Feature Reset Cache (FinnyTable equivalent)
- `sided_feature_reset_cache`: per-king-square (64) cache of accumulator state + piece configuration
- On king move: `half_feature_partial_reset_` compares cached piece config vs actual board, applies only deltas
- This is the **FinnyTable** concept from our experiment queue (also in Obsidian, Alexandria)
- Key difference from a full recompute: only needs to process pieces that changed since last time king was on that square

### Lazy Evaluation (Dirty Nodes)
- `eval_node` uses a union of `context` (dirty) and `eval` (clean)
- Child nodes are created as "dirty" with just the parent reference and move
- Actual NNUE update only happens when `evaluator()` is called (lazy materialization)
- Saves NNUE computation for nodes that are pruned before evaluation

### SIMD
- Compile-time SIMD detection via `nnue/simd.h`
- CReLU255 matrix-vector product for fc0 (most compute-intensive)
- Standard matrix-vector products for float layers

---

## 9. Time Management

### Base Allocation (Increment mode)
```
min_budget = (remaining - 150ms + 25 * inc) / 25
max_budget = (remaining - 150ms + 25 * inc) / 10
Both clamped to: min(4/5 * (remaining - 150ms), budget)
```
- Overhead: 150ms (conservative)
- Soft/hard ratio: 2.5x (hard is 2.5x soft)

### Sudden Death (no increment)
```
min_budget = (remaining - 150ms) / 25
max_budget = (remaining - 150ms) / 10
```

### Moves-to-Go
```
min_budget = 2/3 * (remaining - 150ms) / mtg
max_budget = 10/3 * (remaining - 150ms) / mtg
```

### Node-Based Soft Time Scaling
- After each iteration: `should_stop = elapsed >= (min_budget * 50 / max(best_move_percent, 20))`
- `best_move_percent = 100 * best_move_nodes / total_nodes`
- Effect: if best move uses 50% of nodes, soft limit multiplied by 1.0x (unchanged). If 20% (floor), multiplied by 2.5x. If 80%, multiplied by 0.625x.
- This is the **node-based time management** from our queue (6 engines)
- Simple but effective: single-factor scaling based on node concentration

### Aspiration Fail-High Depth Reduction
- `adjusted_depth = max(1, depth - consecutive_failed_high_count)`
- **Cumulative**: each consecutive fail-high reduces depth by 1 more
- Most engines reduce by just 1 on fail-high; Seer's cumulative approach is unique and aggressive

---

## 10. Lazy SMP

### Thread Management
- `worker_orchestrator` manages threads, shared TT, shared constants
- Each thread has independent: search stack, history tables, eval cache, correction history, NNUE scratchpad, feature reset cache
- Thread depth staggering: `start_depth = 1 + (i % 2)` (odd/even alternation)
- Shared: TT only (via XOR-verified lockless access)
- Stop signal: `go` atomic bool checked via `keep_going()`

### Node Distribution Tracking
- Only thread 0 tracks `node_distribution[move]` for time management
- `best_move_percent()` computed from thread 0's node distribution

---

## 11. Notable Differences from GoChess

### Things Seer has that we don't:
1. **4-component correction history** (pawn hash + NNUE output hash + 2 continuation hashes) -- 4 tables vs our 1
2. **Threat-aware butterfly history** (`[threatened_from][from][to]`) -- 2x table size
3. **Pawn structure history** (512-bucket hash of pawn config in move ordering) -- entirely new table
4. **ProbCut QS pre-filter** (run QSearch first, only do full search if QS passes) -- saves nodes
5. **NMP TT-move SEE guard** (block NMP when TT suggests good capture at SEE > 226) -- unique
6. **Upcoming cycle detection** (Cuckoo hashing for upcoming repetitions) -- more sophisticated than our repetition check
7. ~~**Feature reset cache (FinnyTable)**~~ for NNUE king-move recomputes **(UPDATE 2026-03-21: GoChess now has Finny tables)**
8. **Residual concatenation** in NNUE hidden layers -- all intermediate features available to output
9. **NNUE output feature hash** for correction history -- captures "what the net thinks is important"
10. **Creates-threat LMR reduction** (-1 for moves creating threats)
11. **Cumulative aspiration fail-high depth reduction** -- more aggressive than single -1
12. **RFP score blending** (`(beta+value)/2` return) -- dampened RFP
13. **Quadratic history pruning** (`-1397*d^2` vs our linear -2000*d)
14. **Eval cache** (8MB per-thread cache of NNUE results)
15. **Lazy NNUE evaluation** (dirty nodes, only compute when needed)

### Things we have that Seer doesn't:
1. **Separate LMR tables for captures vs quiets** (they use single table)
2. **Counter-move table** (they embed counter info in continuation history only)
3. **2 killers per ply** (they use 1)
4. **Staged move generation** (they generate all moves upfront)
5. **Check extension** (they have none)
6. **Recapture extension** (they have none)
7. **Fail-high score blending** in main search
8. **Alpha-reduce** (depth-1 after alpha raised)
9. **TT near-miss cutoffs**
10. **TT noisy move detection** for LMR
11. **NMP threat detection** escape bonus
12. **QS beta blending**

### Parameter Comparison Table

Note: Seer uses `logit_scale = 1024`, so direct centipawn comparisons require dividing by ~10.

| Feature | Seer | GoChess |
|---------|------|---------|
| RFP margin | 297*d (threat/improving adjusted), depth<=6 | 85*d imp / 60*d not, depth<=8 |
| RFP return | (beta+value)/2 (blended) | raw value |
| Razoring | 896*d, depth<=3 | 400+100*d, depth<=3 |
| NMP reduction | 3+d/4+min(4,(eval-beta)/242) | 3+d/3+min(3,(eval-beta)/200) |
| NMP extra guard | TT SEE <= 226, threat-aware | Threat detection |
| IIR depth | >= 2 | >= 4 |
| LMR table | 0.47 + log(d)*log(m)/2.57 | Split: cap C=1.80 / quiet C=1.50 |
| LMR history | /5872, clamp [-2,+2] | /5000 |
| LMR creates-threat | -1 | N/A |
| LMP depth | <= 7, table-based | 3+d^2 |
| Futility margin | 1544*d, depth<=5 | 80+d*80 |
| SEE quiet | -54*d, depth<=9 | -20*d^2 |
| SEE noisy | -111*d, depth<=6 | similar |
| History pruning | -1397*d^2 | -2000*d |
| Singular depth | >= 6 | >= 10 |
| Singular beta | tt_val - 2*d | tt_val - 3*d |
| Singular double ext | +166 margin | N/A |
| Multi-cut | Yes | N/A |
| Aspiration delta | 21 | 15 |
| Aspiration fail-high | cumulative depth-- | N/A |
| ProbCut margin | +315 | +170 |
| ProbCut depth | depth-3, min depth 5 | depth-4, min depth 5 |
| ProbCut QS pre-filter | Yes | No |
| History tables | 5 (threat+pawn+counter+follow+capture) | 4 (butterfly+counter+follow+capture) |
| Correction tables | 4 (pawn+nnue_out+cont+ccont) | 1 (pawn) |
| Time soft/hard | 1:2.5 ratio | similar |
| Time node scaling | 50/max(bm_pct,20) | instability factor |

---

## 12. Ideas Worth Testing from Seer

### High Priority (align with proven patterns)

1. **4-table correction history** -- Seer uses pawn hash + NNUE output hash + counter^follow hash + counter^ccounter hash. The continuation-based hashes are novel and don't require new Zobrist keys (use existing move Zobrist hashing). 11+ engines have multi-source correction; Seer's move-hash approach is simplest to implement.
   - **Status**: Multi-source correction is priority #7 in our queue. Seer's implementation is a clean reference.

2. **ProbCut QS pre-filter** -- run QSearch before full ProbCut search. If QSearch doesn't reach probcut_beta, skip the expensive reduced-depth search. 3 engines now (Seer, Alexandria, Tucano).
   - **Status**: Priority #13 in queue. Seer confirms the pattern.

3. **Pawn structure history** -- 512-bucket table indexed by pawn hash, used for quiet move ordering. Complements existing history tables.
   - **Status**: Priority #22 in queue (Obsidian, Weiss, now Seer = 3 engines).

4. **Creates-threat LMR reduction** -- `-1` for moves that create new threats on higher-value pieces.
   - **Status**: Priority ~35 in queue (Koivisto). Seer adds a second implementation.

5. **NMP TT-move SEE guard** -- block NMP when TT best move is a winning capture (SEE > threshold). Prevents NMP in tactically rich positions.
   - **Status**: Already in our queue as item #33. Seer uses SEE threshold 226.

### Medium Priority

6. **Eval cache** (8MB per-thread NNUE result cache) -- avoids recomputing NNUE for positions seen via transpositions or repetitions. Cheap memory, could give NPS boost.

7. **Cumulative aspiration fail-high depth reduction** -- reduce by more than 1 on consecutive fail-highs. Our previous test was bugged (-353 Elo), but this is a valid approach.
   - **Status**: Retry candidate. Seer's cumulative approach is more aggressive than typical -1.

8. **RFP blended return** -- `(beta + value) / 2` instead of raw value. Dampens the score, which aligns with our success pattern #1 (score dampening works at noisy boundaries).
   - **Status**: We tested RFP dampening and got -16.7 Elo. But our test may have used a different formula. Seer's (beta+value)/2 is simpler.

9. **Quadratic history pruning** -- `-1397 * d^2` steepens the curve at higher depths. Worth comparing to our linear -2000*d.

10. **Threat-aware RFP** -- Seer adds 112 to margin when threats exist and subtracts an improving depth level when no threats. Guards pruning in tactical positions.

### Lower Priority / Already Covered

11. ~~**Feature reset cache (FinnyTable)**~~ -- **(UPDATE 2026-03-21: GoChess now has Finny tables, merged.)**

12. **Lazy NNUE evaluation** -- we already have lazy accumulators. Seer's dirty-node pattern is equivalent.

13. **Upcoming cycle detection (Cuckoo)** -- sophisticated but complex to implement. Low priority given other gains available.

14. **Single LMR table** -- we already have separate tables which tested at +43.5 Elo. Not adopting.

---

## 13. Key Architectural Insights

### Seer's Strength Profile
Seer is 293 Elo above GoChess-v5, which is a massive gap. The main structural advantages appear to be:

1. **NNUE quality**: HalfKA with all 64 king squares (vs our 16 buckets), residual connections, FinnyTable caching. **(UPDATE 2026-03-21: GoChess now also has Finny tables, SCReLU, pairwise mul, and dynamic width v5 architecture.)** The network is deeper (4 layers) with residual skip connections ensuring gradient flow. The CReLU255 + float hidden layers is a pragmatic mix.

2. **Correction history breadth**: 4 tables covering different position aspects (pawns, NNUE output signature, move sequences) vs our 1. This is likely the single biggest search-side difference.

3. **Singular extensions**: Activated at depth 6 with double extensions and multi-cut. This alone could account for significant Elo given our SE is broken.

4. **History table richness**: 5 history components including threat-aware and pawn-structure tables. The pawn history is especially interesting as it captures position-type knowledge.

5. **NMP sophistication**: The TT SEE guard and threat-awareness in NMP show attention to tactical safety that our NMP lacks.

### Design Philosophy
Seer is notable for its clean C++ design: templated color/mode parameters eliminating runtime branching, union-based lazy NNUE nodes, composite pattern for multi-table histories. The "reducer" mechanism instead of explicit cut-node tracking is elegant. The code is well-organized and readable despite the template-heavy style.
