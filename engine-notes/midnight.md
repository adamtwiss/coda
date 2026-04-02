# Midnight Chess Engine - Crib Notes

Source: `~/chess/engines/MidnightChessEngine/`
Author: Archishmaan Peyyety
NNUE: 768->768->1 (SCReLU activation, no king buckets, no output buckets)

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. C++ templates for color (no runtime branching on side-to-move). Single-threaded only (no Lazy SMP -- ThreadData exists but engine.cpp shows no thread spawning).

### Iterative Deepening
- Depth 1 to params.depth (default MAX_DEPTH=100)
- Checks soft time limit before each iteration
- Soft time limit scaled by node distribution (see Time Management below)

### Aspiration Windows
- Enabled at depth > 6 (`ASP_WINDOW_MIN_DEPTH = 6`)
- Initial window: prev_score +/- 12 (`ASP_WINDOW_INIT_WINDOW = 12`)
- Initial delta: 16 (`ASP_WINDOW_INIT_DELTA = 16`)
- On fail-low: widen alpha by delta, also pull beta toward alpha: `beta = (alpha + 3*beta) / 4`
- On fail-high: widen beta by delta, **reduce depth by 1** (`depth = max(depth-1, 1)`)
- Delta growth: `delta += delta * 2 / 3` (i.e., multiply by ~1.67 each iteration)
- Full-width fallback when alpha < -3500 or beta > 3500
- **Coda comparison**: Coda uses eval-dependent delta (13+avg^2/23660), fail-low contraction (3a+5b)/8, fail-high contraction (5a+3b)/8 + depth reduce, delta growth 1.5x. Both have fail-high depth reduction and fail-low beta contraction.

### Draw Detection
- Fifty-move rule: `>=100` half-moves
- Repetition: three-fold at root (ply==0), **two-fold** at non-root (ply>0)

### Check Extension
- **Unconditional +1 depth when in check** (applied before depth==0 qsearch drop)
- No fractional extensions anywhere

---

## 2. Pruning Techniques

### Reverse Futility Pruning (RFP)
- Conditions: `!in_check && !pv_node && !excluding_move`
- Depth guard: `depth < 9` (`RFP_MAX_DEPTH = 9`)
- Margin: `75 * depth` (`RFP_MARGIN = 75`)
- Formula: `static_eval >= beta + 75 * depth` => return static_eval
- **No improving adjustment** (same margin regardless of improving flag)
- **Coda comparison**: Coda uses 70*d (improving) / 100*d (not), depth<=7. Midnight allows deeper RFP (depth 8) with a slightly higher margin.

### Razoring
- Conditions: `!pv_node && !in_check && !excluding_move`
- Depth guard: `depth <= 3`
- Margin: `-63 + 182 * depth` (i.e., depth 1: 119, depth 2: 301, depth 3: 483)
- Formula: `static_eval - 63 + 182 * depth <= alpha` => drop to qsearch
- **Coda comparison**: Coda has no razoring.

### Null Move Pruning (NMP)
- Conditions: `depth >= 3 && !in_check && !pv_node && !excluding_move && do_null && static_eval >= beta`
- Reduction: `3 + depth/3 + min((static_eval - beta) / 200, 3)`, clamped to min 3
- Returns null_eval if >= beta (does NOT clamp to beta, returns raw null_eval)
- **Coda comparison**: Coda uses R=4+d/3+(eval-beta)/200 (base R is 4, not 3), verify at depth>=12, post-capture R--, score dampening (2s+b)/3. Coda's base R=4 is more aggressive.

### Internal Iterative Reduction (IIR)
- Condition: `!static_eval_tt.entry_found && depth >= 4`
- Reduces depth by 1
- Note: this is triggered when there's no TT eval entry at all (not just no TT move)
- Uses probe_eval() which matches on hash only (no depth requirement)
- **Coda comparison**: Coda has IIR at depth>=4, no TT move, PV/cut node only.

### Late Move Pruning (LMP) - Two variants

**Variant 1 (break -- stops all remaining moves):**
- Conditions: `!pv_node && depth <= 3` (`LMP_MIN_DEPTH = 3`)
- Table-based thresholds by depth and improving flag:
  - Not improving: `(3 + depth^2) / 2` => depth 1: 2, depth 2: 3, depth 3: 6
  - Improving: `3 + depth^2` => depth 1: 4, depth 2: 7, depth 3: 12
- Triggers when `move_idx > lmp_table[depth][improving]`

**Variant 2 (continue -- skips individual quiet moves):**
- Conditions: `!pv_node && depth <= 6 && move.is_quiet()`
- Threshold: `move_idx > depth * 9`
- This is essentially a second, looser LMP for deeper depths (4-6)
- **Coda comparison**: Coda uses 3+d^2 with improving +50% / failing -33%, depth<=8. Coda's single variant is simpler but applies at higher depths.

### Futility Pruning
- Conditions: `value > -MATE_BOUND && depth < 6` (`FP_DEPTH = 6`)
- Margin: `depth * 100 + 75` (`FP_COEFFICIENT = 100, FP_MARGIN = 75`)
  - depth 1: 175, depth 2: 275, depth 3: 375, depth 4: 475, depth 5: 575
- Formula: `static_eval + margin <= alpha` => break (prunes ALL remaining moves)
- **Note: this is a break, not a continue** -- it prunes ALL remaining moves once triggered
- **Coda comparison**: Coda uses 60+lmrDepth*60, per-move (continue). Midnight's break is more aggressive.

### History Pruning
- Conditions: `!pv_node && value > -MATE_BOUND && depth < 3`
- Threshold: `history[color][from][to] < -1024 * depth`
  - depth 1: history < -1024, depth 2: history < -2048
- **Coda comparison**: Coda uses -1500*depth at depth<=3.

### SEE Pruning
- Conditions: `!pv_node && depth < 7 && value > -MATE_BOUND` (`SEE_PVS_MIN_DEPTH = 7`)
- Quiet margin: `-50 * depth` (`SEE_PVS_QUIET_MARGIN = -50`)
- Tactical margin: `-90 * depth` (`SEE_PVS_TACTICAL_MARGIN = -90`)
- Prunes if SEE fails the threshold
- **Coda comparison**: Coda uses quiet -20d^2 (depth<=8, quadratic) and capture -d*100 (depth<=6, linear). Midnight uses linear margins for both.

### QSearch Futility
- Margin: stand_pat + 60 (`Q_SEARCH_FUTILITY_MARGIN = 60`)
- Combined with SEE: if `futility <= alpha && !SEE(move, 1)`, skip the capture
- The SEE threshold of 1 means it filters captures that don't win material

---

## 3. Extensions

### Singular Extensions
- Depth guard: `depth >= 8`
- Conditions: `!excluding_move && ply > 0 && tt_hit && move == tt_move && tt_depth >= depth - 3 && tt_node != UPPER_NODE`
- Singularity beta: `max(tt_value - 2 * depth, -MATE_BOUND)`
- Singularity depth: `(depth - 1) / 2` (integer division)
- If singularity score < singularity_beta: **+1 extension**
- **Negative extension (-1)**: if tt_value >= beta OR tt_value <= alpha (multi-cut / negative extension)
- **Coda comparison**: Both use depth>=8. Midnight uses margin=2*depth, Coda uses margin=depth (tighter). Both have multi-cut and negative extension.

### Check Extension
- +1 depth unconditionally when in_check (applied at top of pvs before depth==0 check)

### No other extensions
- No recapture extension, no passed pawn extension

---

## 4. LMR (Late Move Reductions)

### Table Initialization
Two separate tables for captures vs quiets:
- **Captures**: `LMR_BASE_CAPTURE(1.40) + log(depth) * log(moveIdx) / LMR_DIVISOR_CAPTURE(1.80)`
- **Quiets**: `LMR_BASE_QUIET(1.50) + log(depth) * log(moveIdx) / LMR_DIVISOR_QUIET(1.75)`

**Coda comparison**: Coda uses ln(d)*ln(m)/1.30 (quiet) and ln(d)*ln(m)/1.80 (capture), no additive base. Midnight's additive base (1.40/1.50) makes reductions more aggressive at all depths.

### Application Conditions
- `depth >= 3 && move_idx > lmr_depth && !in_check`
- `lmr_depth` = 5 for PV nodes or root, 3 for non-PV nodes
- So PV nodes start LMR at move 6+, non-PV at move 4+

### Reduction Adjustments
- `-1` for PV node
- `+1` for not improving
- Clamped to `[0, depth-1]`

### What Midnight lacks (that Coda has):
- No history-based reduction adjustment
- No capture-specific reduction logic beyond the separate table
- No TT-based adjustments
- No doDeeper/doShallower
- No cutnode LMR adjustment

---

## 5. Move Ordering

### Score Hierarchy (higher = searched first)
All scores are computed upfront and negated for selection sort:

1. **TT move**: +10,000,000 (`PREVIOUS_BEST_MOVE_BONUS`)
2. **Promotions**: piece_value + 1,000,000 (`PROMOTION_BONUS`)
3. **Good captures**: `(100,000 + capture_history) * SEE_pass + MVV-LVA`
   - SEE threshold for "good": -107
   - Captures that pass SEE get 100,000 + capture_history bonus
   - Captures that fail SEE get only MVV-LVA (victim_value - attacker_value), which is low/negative
4. **Killer moves**: 10,000 + 2000 (first killer) or 10,000 + 1000 (second killer)
5. **Quiet moves**: main_history + continuation_history(1-ply) + continuation_history(2-ply)
6. **Penalty**: -350 if move lands on square attacked by opponent pawn

### Not staged
Unlike most engines, Midnight does NOT use a staged move picker. It generates ALL moves, scores them all upfront, then uses selection sort (pick-best) during iteration. This means:
- All moves are generated even if beta cutoff happens on the first move
- No separate capture-only generation phase
- Simple but potentially slower than staged generation

### History Tables

**Main history**: `history[2][64][64]` -- color, from, to
- Bonus: `depth^2 + depth - 1`
- Gravity formula: `entry -= (entry * |bonus|) / 324; entry += bonus * 32`

**Continuation history**: `continuation_history[12][64][12][64]` -- prev_piece, prev_to, curr_piece, curr_to
- 1-ply and 2-ply lookback
- Same bonus and gravity formula as main history

**Capture history**: `capture_history[12][64][12]` -- attacker_piece, to_square, victim_piece

**Killers**: 2 slots per ply

### Novel: Opponent Pawn Territory Penalty
- Quiet moves that land on squares attacked by opponent pawns get -350 penalty
- Effectively deprioritizes moves into pawn-attacked squares
- **Coda**: Does not have this. Related to Coda's threat-aware 4D history which achieves a similar goal through different means.

---

## 6. Time Management

### Base Allocation
**Standard (no movestogo):**
- Soft limit: `time_remaining / 40 + 3 * increment / 4`
- Hard limit: `time_remaining / 5 + 3 * increment / 4`

**Moves-to-go:**
- moves_to_go capped at 50
- scale = 0.7 / moves_to_go
- optimum = min(scale * time_remaining, 0.95 * time_remaining)
- Hard limit: min(5 * optimum, 0.95 * time_remaining)
- Soft limit: optimum

### Node-Based Soft Time Scaling (Best Move Stability)
- Only active at depth > 9
- Tracks `nodes_spent[from][to]` for root moves (thread 0 only)
- `percent_nodes_spent = best_move_nodes / total_nodes`
- `scaled_soft_limit = soft_limit * (1.5 - percent_nodes_spent) * 1.35`
- Clamped to hard_limit
- Effect: if best move uses 50% of nodes, scale = 1.35x. If 90%, scale = 0.81x. If 10%, scale = 1.89x
- **No fail-low extension** -- there's no special handling for fail-low at root

**Coda comparison**: Coda uses time_left/20 + 3*inc/4 (soft), 5x (hard). Similar base allocation. No node-ratio scaling.

---

## 7. NNUE Architecture

### Network: 768 -> 768 -> 1 (SCReLU)
- Input: 64 squares * 12 piece types = 768 features (no king buckets!)
- Hidden layer: 768 neurons (much wider than typical, but only one hidden layer)
- Output: single value
- Activation: SCReLU (squared clipped ReLU): `clamp(x, 0, QA)^2` where QA=181
- Quantization: QA=181, QB=64
- Scale: 400
- Final: `(sum + output_bias) * 400 / (181 * 64)`

### Compared to Coda's NNUE
- Coda: HalfKA with 16 king buckets, 8 output buckets, v5/v6/v7 architectures, CReLU/SCReLU/pairwise, Finny tables, AVX2/AVX-512 SIMD
- Midnight: Simple 768->768->1 (piece-square only, no king buckets, no output buckets)
- Midnight's simpler architecture means fewer features but faster inference

### Accumulator
- Stack-based (push_copy on make, pop on unmake)
- Incremental updates for feature add/remove
- SIMD: compile-time detection (AVX2/AVX512/NEON/None)

---

## 8. Transposition Table

### Structure
- Single-entry per index (no buckets)
- Index: `hash % entry_count` (modulo, not power-of-2 masking)
- No lockless/atomic access (not thread-safe)
- Fields: zobrist_hash, depth (i16), value (i32), node_type, best_move

### Replacement Policy
Replace if ANY of:
- New entry is EXACT node type
- Different hash (new position)
- `new_depth + 7 + 2*pv_node > old_depth - 4` (heavily favors replacement; threshold is depth+11 for PV, depth+7 for non-PV vs old_depth-4)

### Probe for Search
- Requires `entry.depth >= depth && ply != 0` (never cuts at root)
- Standard alpha/beta/exact cutoff logic

### Probe for Eval (IIR trigger)
- Hash match only (no depth requirement)
- Used to get static_eval from TT and to trigger IIR if no entry found

---

## 9. Notable Differences from Coda

### Things Midnight has that Coda doesn't:
1. **Opponent pawn territory penalty in move ordering** (-350 for moves into pawn-attacked squares)
2. **Two-tier LMP**: strict table-based LMP at depth<=3, plus loose `depth*9` quiet LMP at depth<=6
3. **Razoring** (-63 + 182*depth, depth<=3)

### IMPLEMENTED (Coda already has these):
- **Separate LMR tables for captures vs quiets** (Coda: C=1.30/C=1.80)
- **Aspiration fail-high depth reduction**
- **Aspiration fail-low beta narrowing** (Coda: (3a+5b)/8)
- **Capture history in capture ordering**

### Things Coda has that Midnight doesn't:
1. **Lazy SMP** (Midnight is single-threaded)
2. **King buckets in NNUE** (HalfKA vs simple 768)
3. **Counter-move heuristic**
4. **ProbCut** (disabled in Coda, but implemented)
5. **History-based LMR adjustments**
6. **Recapture extensions**
7. **Multi-source correction history**
8. **Staged move generation** (Midnight generates all moves upfront)
9. **Multi-bucket TT** (Midnight uses single-entry)
10. **Lockless TT** (Midnight has no thread safety)
11. **Pawn history**
12. **Threat-aware 4D history**
13. **DoDeeper/DoShallower**
14. **Cuckoo cycle detection**
15. **Hindsight reduction**
16. **Fail-high score blending**

### Parameter Comparison Table

| Feature | Midnight | Coda |
|---------|----------|------|
| RFP margin | 75*depth, depth<9 | 70*d (imp) / 100*d (not), depth<=7 |
| RFP improving | No distinction | Yes, different margins |
| Razoring | -63+182*d, depth<=3 | None |
| NMP reduction | 3+d/3+min((eval-beta)/200,3) | 4+d/3+(eval-beta)/200, R-- after cap |
| NMP min depth | 3 | 4 |
| NMP verification | None | depth >= 12 |
| NMP dampening | None | (2*score+beta)/3 |
| LMR base (quiet) | 1.50 + ln(d)*ln(m)/1.75 | ln(d)*ln(m)/1.30 |
| LMR base (capture) | 1.40 + ln(d)*ln(m)/1.80 | ln(d)*ln(m)/1.80 |
| LMR start (non-PV) | move 4+ | move 3+ |
| LMP depth | <=3 (strict) + <=6 (loose) | <=8, 3+d^2 |
| Futility margin | 100*d+75, depth<6 | 60+lmrDepth*60, depth<=8 |
| SEE quiet margin | -50*d, depth<7 | -20*d^2, depth<=8 |
| SEE tactical margin | -90*d, depth<7 | -d*100, depth<=6 |
| Singular depth | >=8 | >=8 |
| Singular beta | tt_val - 2*depth | tt_score - depth |
| Aspiration delta | 12 initial, delta 16 | 13+avg^2/23660 |
| QS futility | stand_pat + 60 | QS delta 240 |

---

## 10. Ideas Worth Testing from Midnight

1. **Opponent pawn territory penalty** in move ordering -- cheap to compute, might improve ordering. However, Coda's 4D threat-aware history may already capture this effect.
2. **Two-tier LMP** with a loose quiet-only tier at depth<=6 (`depth*9` threshold) -- simple and complementary to existing LMP.
3. **Razoring** -- Coda has no razoring at all. Midnight's -63+182*d is worth trying, though razoring results are often marginal for NNUE engines.
