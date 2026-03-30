# Altair Chess Engine - Crib Notes

Source: `~/chess/engines/AltairChessEngine/`
Author: Alexander Tian
Version: 7.2.1
RR rank: #11 at +112 Elo (220 above GoChess-v5)
NNUE: (768x5 -> 1024)x2 -> 1x8 (SCReLU, 5 king buckets, 8 output buckets)

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. Lazy SMP with shared TT. C++ with `search_parameters.h` exposing all tunable params with SPRT-style min/max/step ranges (OpenBench-compatible).

### Iterative Deepening
- Depth 1 to max_depth (default MAX_AB_DEPTH-1 = 255)
- Checks soft time limit after each iteration (depth >= min_depth)
- Separate `asp_depth` counter starts at 6, increments each iteration from depth 6+, used for aspiration delta scaling

### Aspiration Windows
- Enabled at depth >= 6 (`MINIMUM_ASP_DEPTH = 6`)
- Initial delta: `max(6 + 85/(asp_depth-2), ASP_delta_min=9)` -- starts ~49 at depth 6, shrinks as asp_depth grows
- On fail-low: widen alpha by delta, contract beta: `beta = (3*alpha + 5*beta) / 8`
- On fail-high: widen beta by delta, **reduce depth by 1** (only when return_eval < MATE_BOUND)
- On fail-low: decrease asp_depth (clamp >= 6)
- On fail-high: decrease asp_depth (clamp >= 6)
- Delta growth: `delta += delta * ASP_delta_scaler(2) / ASP_delta_divisor(5)` = multiply by 1.4 each re-search
- Full-width fallback when alpha <= -1000 or beta >= 1000
- Compare to ours: delta=15, growth 1.5x. **(UPDATE 2026-03-21: GoChess now has fail-low contraction (3a+5b)/8 and fail-high contraction (5a+3b)/8, matching Altair's asymmetric style. No depth-adaptive delta yet.)**

### Draw Detection
- Fifty-move rule: >= 100 half-moves
- Repetition: **two-fold** at any ply (single loop checking all positions back to fifty_move reset)
- Draw score: `3 - (node_count & 8)` -- randomized between -5 and 3 to avoid draw-seeking

### Check Extension
- +1 depth unconditionally when in check (applied after move is made, via `in_check` flag)
- Only when move was not extended by singular extension or passed-pawn extension

### Mate Distance Pruning
- `alpha = max(alpha, -MATE_SCORE + ply)`, `beta = min(beta, MATE_SCORE - ply)`, cutoff if alpha >= beta
- **We don't have this** -- trivial to add, 3 lines

---

## 2. Pruning Techniques

All forward pruning gated by: `!pv_node && !in_check && !singular_search && abs(beta) < MATE_BOUND`.

### Reverse Futility Pruning (RFP)
- Depth guard: `depth <= 7` (`RFP_depth = 7`)
- Margin: `110 * (depth - improving)` (`RFP_margin = 110`)
- Formula: `eval - 110 * (depth - improving) >= beta` => return `(eval + beta) / 2`
- **Returns average of eval and beta** (score dampening), not raw eval
- Compare to ours: we return raw eval, use 85*d (improving) / 60*d (not improving), depth<=8

### Null Move Pruning (NMP)
- Min depth: `depth >= 1` (`NMP_depth = 1`) -- extremely low threshold
- Conditions: `do_null && eval >= static_eval && eval >= beta`
- Extra condition: `static_eval >= beta + NMP_condition_base(128) - (NMP_condition_depth(11) + improving*NMP_condition_improving(3)) * depth`
  - This is a depth-dependent activation gate: only NMP when eval is sufficiently above beta relative to depth
- Material guard: `non_pawn_material_count >= 1 + (depth >= 10)` (needs 2+ non-pawn pieces at depth >= 10)
- Reduction: `NMP_base(5) + depth/NMP_depth_divisor(4) + clamp((eval-beta)/NMP_eval_divisor(400), -1, 3)`
  - Base R=5 (vs our R=3!), depth divisor 4 (vs our 3), eval divisor 400 (vs our 200)
  - **Much more aggressive NMP** -- R=5 base is very high
- Returns `beta` if return_eval >= MATE_BOUND, else returns return_eval
- **NMP Verification at depth > 15**: runs full `negamax(beta-1, beta, depth-reduction)` with do_null=false
  - Only verifies when depth > 15 (classical anti-zugzwang safety net)
  - Compare to ours: we also have NMP verification

### Internal Iterative Reduction (IIR)
- Condition: `tt_move == NO_MOVE && depth >= IIR_base_depth(4)`
- Reduction: `depth -= 1 + pv_node` -- **extra -1 on PV nodes** (we only do -1)
- This is novel: PV nodes without TT move get reduced by 2, which is aggressive

### Late Move Pruning (LMP) -- Two tiers

**Tier 1 (strict, noisy+quiet, break):**
- Conditions: `!pv_node && depth <= LMP_depth(4)`
- Threshold: `legal_moves >= depth * LMP_margin(10)`
  - depth 1: 10, depth 2: 20, depth 3: 30, depth 4: 40
- Triggers break (stops all remaining moves)

**Tier 2 (quiet only, break):**
- Conditions: `!pv_node && quiet`
- Threshold: `legal_moves >= LMP_margin_quiet(2) + depth*depth / (1 + !improving + failing)`
  - Not improving, not failing: `2 + d^2/2`
  - Improving, not failing: `2 + d^2`
  - Not improving, failing: `2 + d^2/3` (most aggressive)
  - The `failing` flag **tightens LMP** for deteriorating positions
- Compare to ours: we use `3 + d^2` (+50% improving). They have a more nuanced 3-way split with the failing heuristic.

### Futility Pruning
- Conditions: `!pv_node && quiet && depth <= FP_depth(7)`
- Margin: `(depth - !improving) * FP_multiplier(170) + FP_margin(70)`
  - depth 1 improving: 240, depth 2: 410, depth 3: 580, depth 4: 750, depth 5: 920, depth 6: 1090, depth 7: 1260
  - Not improving: uses (depth-1) instead of depth
- Formula: `static_eval + margin <= alpha` => break (prunes ALL remaining moves)
- Compare to ours: we use 80+d*80 (much tighter). They have much wider margins but also check deeper.

### History Pruning
- Conditions: `!pv_node && (quiet || !winning_capture) && depth <= history_pruning_depth(10)`
- Threshold: `move_history_score <= (depth + improving) * -history_pruning_divisor(9600)`
  - Combined history (butterfly + continuation) must be below threshold
  - depth 1 not improving: < -9600, improving: < -19200
- Uses `continue` (per-move skip, not break)
- Compare to ours: we use -2000*depth, they use -9600*(depth+improving) -- much larger threshold but includes cont-hist sum

### SEE Pruning
- Conditions: `depth <= SEE_base_depth(2) + SEE_noisy_depth(0)*!quiet + SEE_pv_depth(5)*pv_node`
  - Quiet: depth <= 2 (non-PV) or depth <= 7 (PV)
  - Noisy: depth <= 2 (non-PV) or depth <= 7 (PV)
- Extra conditions: `legal_moves >= SEE_base_moves(4) && move_history_score <= SEE_base_history(5800)`
  - History gate: only prune by SEE when history is not strong
- Quiet margin: `-SEE_quiet_multiplier(40) * depth`
- Noisy margin: `-SEE_noisy_multiplier(85) * depth`
- Compare to ours: similar margins. Their history gate on SEE pruning is novel -- prevents pruning moves with strong history.

### QSearch SEE Pruning
- Threshold: `eval + QSEE_base(70) <= alpha && !SEE(move, 1)`
- `best_score = max(best_score, eval + QSEE_base)` -- sets floor at eval+70 for failed captures
- Only winning captures searched (bad captures filtered at generation time)

---

## 3. Extensions

### Singular Extensions
- Depth guard: `depth >= SE_base_depth(6)`
- Conditions: `!root && move == tt_move && tt_entry.depth >= depth - 2 && tt_flag != UPPER && !excluding && |tt_score| < MATE_BOUND`
- Singular beta: `tt_score - depth` (margin = 1*depth, tighter than many engines)
- Singular depth: `(depth - 1) / 2`
- If singular score < singular_beta: **+1 extension**
- **Double Extension**: if `!pv_node && return_eval < singular_beta - SE_dext_margin(12) && double_extensions <= SE_dext_limit(7)`: +2 total
- **Multi-cut Pruning**: if return_eval >= beta: return return_eval immediately
- **Negative Extension (-1)**: if `tt_score >= beta || tt_score <= alpha` (same as Midnight)
- Double extension counter tracked per ply (inherited from parent, incremented on double ext)
- Compare to ours: our SE had -58 to -85 Elo. Altair's is well-structured: depth>=6, margin=1*depth, double ext with limiter, multi-cut.

### Passed Pawn Extension
- +1 for passed pawns reaching 7th rank (pawn on rank 6 for white, rank 1 for black)
- Only when `move_score >= 0` (not a losing move)
- Also +1 for queen promotions specifically

### Check Extension
- +1 unconditionally when the side to move is in check
- Only if not already extended by singular/passed-pawn

### Extension Capping
- `extension = min(extension, min(2, MAX_AB_DEPTH - 1 - depth))`
- Prevents extending beyond max depth

---

## 4. LMR (Late Move Reductions)

### Table Initialization
Single table for quiets (captures use the same table with adjustments):
- **Quiets**: `max(0, log(depth) * log(moves) / (LMR_divisor_quiet(230)/100.0) + LMR_base_quiet(140)/100.0)`
- Effective: `log(d)*log(m)/2.30 + 1.40`
- Compare to ours: we have split tables -- cap C=1.80, quiet C=1.50 (from our LMR-split merge)

### Application Conditions
- `legal_moves >= 1 + root + pv_node && depth >= 3 && (quiet || !winning_capture)`
- So root PV: move 3+, non-root PV: move 2+, non-PV: move 2+
- Bad captures also get LMR (losing captures treated like quiets)

### Reduction Adjustments (all applied to the single base table value)
- `-1` for noisy moves (`!quiet`)
- `-1` for PV node
- `-1` for tt_pv (was PV in TT)
- `-1` for improving
- `+1` for failing
- `-1` for "interesting" moves (passed pawn, queen promotion, gives check, is killer)
- `-1` for in_check
- `-move_history_score / LMR_history_divisor(10000)` -- history-based adjustment
- `+alpha_raised_count * (0.5 + 0.5 * tt_move_noisy)` -- **novel: alpha-raised count scaling**
  - More reduction after alpha has been raised multiple times
  - Extra aggressive when TT move was noisy (captures suggest tactical position)
- `+1` for cutnode
- Clamped to `[0, new_depth - 1]`

### DoDeeper / DoShallower (after LMR re-search)
- After ZW LMR search exceeds alpha:
  - `new_depth += (return_eval >= best_score + deeper_margin(80))` -- search deeper if score far above best
  - `new_depth -= (return_eval < best_score + new_depth)` -- search shallower if score only marginally above
- Compare to ours: we tested do-deeper/shallower, got -13.7 Elo. Their thresholds differ: +80 for deeper, best_score+new_depth for shallower.

### Key Differences from GoChess LMR
- Single table (not split cap/quiet like ours)
- The `failing` flag adds reduction in deteriorating positions (novel)
- `alpha_raised_count` scaling is unique to Altair
- History divisor 10000 (vs our 5000) -- less aggressive history adjustment
- `tt_pv` reduction bonus (reduces less when TT says this was on PV)
- `interesting` catch-all flag covers checks, killers, passed pawns

---

## 5. Move Ordering

### Staged Generation
Altair uses a **proper staged move picker** (unlike Midnight which generates all upfront):

1. **Stage TT_probe**: Return TT move immediately if pseudo-legal
2. **Stage GenNoisy**: Generate all noisy moves, score with SEE + capture history
3. **Stage Noisy**: Pick-best among good captures (SEE >= -85 threshold). Bad captures get BAD_SCORE and are deferred.
4. **Stage GenQ_BN**: Generate quiet moves, append to same vector (bad captures already there)
5. **Stage Q_BN**: Pick-best among quiets + bad captures. Uses **MaxHeap** when gen_all flag is set and >8 moves remain.

The MaxHeap optimization is interesting: on PV or exact TT nodes where all moves are expected to be searched, it builds a heap after the first quiet is picked for O(log n) extraction instead of O(n) selection sort.

### Score Hierarchy

1. **TT move**: 500,000 (`MO_Margin::TT`)
2. **Good captures** (SEE >= -85): 50,000 + capture_history[winning][piece][victim][to] + MVV-LVA scale
3. **Queen promotions**: 100,000
4. **Other promotions**: -30,000 + piece value
5. **Killers**: killer_1 = 30,000, killer_2 = 25,000
6. **Castling**: 1,200
7. **Quiet moves**: threat-aware history + continuation history (3 plies)
8. **Bad captures** (SEE < -85): scored in Q_BN stage with base_capture(-3000) + MVV-LVA + capture_history

### History Tables

**Butterfly History (threat-aware)**: `history_moves[12][64][2][2]` -- piece, target_square, origin_threatened, target_threatened
- 4x the standard butterfly table via threat indexing
- `threats` bitboard computed per position (all opponent attacks)
- `(threats >> move.origin()) & 1` and `(threats >> move.target()) & 1` as boolean indices
- **This is the threat-aware history from SUMMARY.md (#5, 12 engines)** -- Altair implements it

**Capture History**: `capture_history[2][12][12][64]` -- winning_capture_flag, attacker_piece, victim_piece, target_square
- Indexed by whether the capture was "winning" (passed SEE threshold)
- This is a 2-tier capture history, which is unusual

**Continuation History**: `continuation_history[12][64][12][64]` -- prev_piece, prev_to, curr_piece, curr_to
- Lookback plies: 1, 2, and **4** (`LAST_MOVE_PLIES = {1, 2, 4}`)
- Ply 4 lookback is significant -- most engines only do 1-2, but Altair goes to 4
- Compare to ours: we use plies 1-2. Our ply-4 test got -58 Elo, but that was at full weight. Altair uses it at equal weight.

**Killers**: 2 slots per ply
- Special handling: after null-move search, if killer[0] is empty, store as killer[0]; otherwise store as killer[1]
- Prevents null-move from overwriting important killers

**No counter-move table** -- relies on continuation history instead.

### History Bonus/Gravity
- Bonus: `depth * (depth + 1 + null_search + pv_node + improving) - 1`
  - Extra bonus in null-search nodes, PV nodes, and improving positions
  - Extra `+ (alpha >= eval)` on alpha raises (surprising cutoffs get more bonus)
  - Compare to ours: we use `d*d + d - 1`
- Gravity: `score += bonus*32 - (score * abs(bonus*32)) / max_score`
  - Max scores: quiet=9000, noisy=12000, continuation=11000
  - Compare to ours: similar formula with different divisors

---

## 6. Time Management

### Position-Based Time Scaling
Altair has a unique `position_time_scale()` function that considers:
- **Game phase**: weighted piece count (P=1, N/B=3, R=7, Q=15, K=0) as fraction of max (96)
- **Low-rank pawns**: count of pawns on ranks 1-2 (white) or 7-8 (black)
- Returns a factor between 0.2 and 1.2 used to scale time allocation
- Low material: 0.2x. Early opening (many low-rank pawns): reduced time. Complex middlegame: full time.

### Base Allocation
**With increment, no movestogo:**
- `time_amt = time_remaining / 20 + inc * 0.75`
- Scaled by `(0.7 + 0.3 * position_time_scale)`

**No increment, no movestogo:**
- `time_amt = time_remaining / 24`
- Scaled by `(0.8 + 0.2 * position_time_scale)`

**Movestogo (with or without inc):**
- `movestogo_ratio = clamp(atan((movestogo + 20) / 24), 0.84, 1.1)` -- arctangent smoothing
- `time_amt = time * movestogo_ratio / movestogo + inc * 0.75`
- Special case: movestogo == 1 => use 90% of remaining time

### Hard/Soft Limits
- Hard limit: `max(time_amt, time_amt * TM_hard_scalar(281)/100) - move_overhead`
  - Effective: hard = time_amt * 2.81
- Soft limit: `time_amt * TM_soft_scalar(78)/100`
  - Effective: soft = time_amt * 0.78
- Both capped at 80% of remaining time

### Node-Based Soft Time Scaling
- Active at depth >= 8
- `best_node_percentage = best_move_nodes / total_nodes`
- `node_scaling_factor = (TM_node_margin(135)/100 - best_node_pct) * TM_node_scalar(180)/100`
  - If best move uses 50%: factor = (1.35-0.50)*1.80 = 1.53x
  - If best move uses 90%: factor = (1.35-0.90)*1.80 = 0.81x
  - If best move uses 10%: factor = (1.35-0.10)*1.80 = 2.25x

### Score-Based Time Scaling
- Tracks `low_depth_score` (from depth <= 4) and current score
- `score_difference = abs(clamp(current - low_depth, -TM_score_min(276), TM_score_max(136)))`
- `score_scaling_factor = TM_score_base(98)/100 + score_difference / TM_score_divisor(381)`
  - Base: 0.98. Max positive shift (score dropping 276cp): 0.98 + 276/381 = 1.70x
  - Max negative shift (score rising 136cp): 0.98 + 136/381 = 1.34x
- The asymmetry is intentional: score drops extend time more than score rises
- `soft_time_limit = original_soft * node_scaling * score_scaling`

Compare to ours: **(UPDATE 2026-03-21: GoChess now has score-drop time extension with 2.0x/1.5x/1.2x tiered scaling, merged.)** Their system is more granular with continuous node scaling and asymmetric score scaling; ours is tiered but goes up to 2.0x.

---

## 7. NNUE Architecture

### Network: (768x5 -> 1024)x2 -> 1x8 (SCReLU)
- Input: 768 = 64 squares * 12 piece types (HalfKP-style, perspective-relative)
- **5 king buckets** -- asymmetric mapping:
  ```
  0 0 0 1 1 2 2 2    (rank 8, from white's perspective)
  0 0 1 1 1 1 2 2
  0 3 3 3 4 4 4 2
  3 3 3 3 4 4 4 4    (ranks 5 down to 1 share same buckets)
  3 3 3 3 4 4 4 4
  3 3 3 3 4 4 4 4
  3 3 3 3 4 4 4 4
  3 3 3 3 4 4 4 4    (rank 1)
  ```
  - Corner-heavy: 3 buckets for back rank (left/center/right), 2 buckets for rest
  - Black bucket computed from `square ^ 56` (vertically flipped)
  - Total feature weights: 5 * 768 * 1024 = 3,932,160

- Hidden layer: 1024 neurons (wide single hidden)
- **8 material output buckets**: `bucket = (popcount(all_pieces) - 2) / 4`
- Output weights: `1024 * 2 * 8 = 16,384` (transposed at load time for cache efficiency)
- Activation: **SCReLU**: `clamp(x, 0, QA=255)^2`
- Quantization: QA=255, QB=64, SCALE=400
- Final: `(sum + output_bias[bucket]) * 400 / (255 * 64)`

### Compared to our NNUE
- Ours: v5 (768x16->N)x2->1x8, 16 king buckets, 8 output buckets, CReLU/SCReLU, Finny tables **(UPDATE 2026-03-21: GoChess now has v5 shallow wide architecture with SCReLU inference support, pairwise multiplication, dynamic width, and Finny tables)**
- Theirs: (768x5 -> 1024)x2 -> 1x8, 5 king buckets, 8 output buckets, SCReLU
- Fewer king buckets (5 vs our 16) but still king-relative
- Both use SCReLU now; both have shallow wide architecture
- No Finny tables (full recompute on king bucket change) -- we now have Finny tables

### Accumulator
- Stack-based: push copy on make_move, pop on undo_move
- Incremental updates for feature add/remove
- **Per-side king bucket refresh**: when king changes bucket, only that side's accumulator is recomputed from scratch; the other side gets incremental updates applied to the copy
- SIMD: compile-time detection (AVX-512 / AVX2 / NEON / scalar fallback)
- SCReLU SIMD: clamp with min/max intrinsics, multiply with mullo, madd for dot product
- No lazy accumulator (eager incremental updates)
- No FinnyTable (full recompute on king bucket change)

---

## 8. Transposition Table

### Structure
- Single-entry per index (no buckets), 24 bytes per entry
- Fields: key (8B), score (4B), evaluation (4B), move (2B), depth (2B), flag (2B), pv_node (1B)
- Power-of-2 sizing with bitmask indexing: `hash & (size - 1)`
- **Not lockless** -- no atomic operations. Shared between threads in Lazy SMP.

### Replacement Policy
Replace if ANY of:
- Different hash (new position)
- `artificial_depth >= old_depth`, where `artificial_depth = new_depth + 3 + 2*tt_pv`
  - PV entries get +5 depth bonus, non-PV get +3 bonus
  - This strongly favors replacing with PV entries and retaining PV entries
- New entry is EXACT node type
- Move is only updated if flag != UPPER or old entry had no move

### Probe
- Full search: requires `tt_entry.depth >= depth` for score cutoff; move always returned on hash match
- Qsearch: no depth requirement for cutoffs (any hash match can cut)
- Eval probe: separate function, returns stored static evaluation on hash match
- TT eval correction: if TT has a value and the bound matches direction (lower bound when tt_value >= eval, upper when <), use tt_value as eval

### TT PV Flag
- Stored as `pv_node` bool in TT entry
- Retrieved as `tt_pv = tt_entry.pv_node || pv_node` (sticky PV flag)
- Used in: replacement policy (+2 depth), LMR (-1 reduction for tt_pv moves)
- Compare to ours: we don't store PV flag in TT

---

## 9. Lazy SMP

- Thread 0 runs iterative_search as main thread
- Helper threads: copies of thread 0's state, each runs independent iterative_search
- All threads share the TT (not lockless -- potential data races)
- Best move taken from thread 0 only
- Node counting: each thread has separate counter, summed for NPS/time checks
- Time checking only on thread 0
- No per-thread depth offset (unlike some engines that stagger depths)

---

## 10. Correction History

### 4 Tables (pawn, non-pawn, major, minor)
- `correction_history[2][16384]` -- color, hash-indexed
- `correction_history_np[2][16384]` -- non-pawn hash key
- `correction_history_major[2][16384]` -- major piece hash key
- `correction_history_minor[2][16384]` -- minor piece hash key

### Hash Keys
- Pawn hash: XOR of Zobrist keys for all pawns
- Non-pawn hash: XOR of Zobrist keys for all non-pawn pieces + castling rights
- Major hash: XOR of Zobrist keys for rooks/queens/kings + castling rights
- Minor hash: XOR of Zobrist keys for knights/bishops
- All incrementally updated in make_move

### Update Formula
- `new_weight = min((1+depth)^2, 144)` -- depth-squared, capped at 144 (depth 11)
- `entry = entry * (1024 - weight) / 1024 + scaled_diff * weight / 1024`
  - scaled_diff = `(best_score - static_eval) * 256`
- Clamped to +/- `256 * 64 = 16384`

### Application
- `corrected = raw_eval + sum(all 4 correction scores / 256)`
- All 4 corrections summed equally (no weighting between tables)
- Updated when: not in check, best move is quiet or no move, and score doesn't contradict TT bound direction

Compare to ours: we only have pawn correction history. They have 4 tables with dedicated hash keys. This is the multi-source correction history from SUMMARY.md (#7, 11 engines). Their implementation uses single Zobrist keys shared with the main hash (not separate per-color like Alexandria).

---

## 11. Notable Unique Techniques

### The "Failing" Heuristic
```
failing = !pv_node && static_eval < past_eval - (60 + 40 * depth)
```
- Detects when the position is deteriorating significantly
- Used in: LMR (+1 reduction), quiet LMP (tighter divider, 3 instead of 2)
- The depth-dependent margin means deeper searches need more deterioration to trigger
- **Already in SUMMARY.md as idea #29** from this engine

### Alpha-Raised Count in LMR
```
reduction += alpha_raised_count * (0.5 + 0.5 * tt_move_noisy)
```
- Tracks how many times alpha was raised at current node
- Each raise increases reduction for subsequent moves
- Amplified when TT move was noisy (captures)
- **Already in SUMMARY.md as idea #30** from this engine + Reckless

### RFP Score Dampening
- Returns `(eval + beta) / 2` instead of raw eval
- Prevents RFP from returning inflated scores that confuse parent nodes
- Compare to ours: we return raw eval. We tested RFP dampening and got -16.7 Elo. But our test may have used different parameters.

### Null-Move Killer Protection
- After null-move search, killers are stored in slot 1 (not slot 0) if slot 0 is already occupied
- Prevents null-move refutation from overwriting the primary killer

### Position-Phase Time Scaling
- Unique feature: game phase + pawn structure analysis to scale time allocation
- Low-rank pawn count as proxy for opening vs middlegame vs endgame
- Most engines only use simple time/moves formulas; Altair adds positional awareness

### History Bonus Depth Adjustment
```
depth_adjusted = depth + (alpha >= eval)
bonus = depth_adjusted * (depth_adjusted + 1 + null_search + pv_node + improving) - 1
```
- The `+ (alpha >= eval)` means surprising cutoffs (from worse-looking positions) get stronger history reinforcement
- Similar to Alexandria's eval-based history depth bonus (**SUMMARY.md #14f**)

### SEE Pruning History Gate
```
move_history_score <= SEE_base_history(5800)
```
- SEE pruning only activates when the move's combined history score is below 5800
- Prevents SEE from pruning moves that history says are good
- This is a safety net: if history strongly recommends a move, skip SEE check

### Continuation History at Ply 4
- LAST_MOVE_PLIES = {1, 2, 4} -- reads cont-hist at plies 1, 2, and 4
- **Already in SUMMARY.md as idea #20** (11 engines). Our test at full weight failed (-58 Elo).
- Altair uses it at equal weight for both scoring and update, which suggests it works when well-tuned.

---

## 12. Notable Differences from GoChess

### Things Altair has that we don't:
1. **4-table correction history** (pawn, non-pawn, major, minor) with dedicated hash keys
2. **Threat-aware butterfly history** `[piece][to][from_threatened][to_threatened]`
3. **"Failing" heuristic** in LMR/LMP (position deterioration detection)
4. **Alpha-raised count in LMR** (progressive reduction after alpha raises)
5. **Mate distance pruning** (trivial, 3 lines)
6. **NMP base R=5** (vs our R=3 -- significantly more aggressive)
7. **IIR extra reduction on PV** (depth -= 1 + pv_node)
8. **Aspiration depth-adaptive delta** (starts ~49, shrinks over iterations)
9. ~~**Aspiration fail-low beta contraction**~~ (beta = (3*alpha + 5*beta) / 8) **(UPDATE 2026-03-21: GoChess now has this)**
10. **RFP score dampening** (return (eval+beta)/2)
11. **Position-phase time scaling** (game phase + pawn structure)
12. **Score-based time scaling** (asymmetric: drops extend more than gains)
13. **Staged move picker with MaxHeap** on expected-all-nodes
14. **TT PV flag** stored and used in replacement policy + LMR
15. **Continuation history at ply 4**
16. **SEE pruning history gate** (skip SEE if history > 5800)
17. ~~**SCReLU activation**~~ in NNUE **(UPDATE 2026-03-21: GoChess now supports SCReLU inference with SIMD)**
18. **Draw score randomization** (3 - (nodes & 8))
19. **DoDeeper/DoShallower** after LMR re-search

### Things we have that Altair doesn't:
1. **Lockless TT** (they share TT without atomics -- data race risk)
2. **4-bucket TT** (they use single-entry)
3. **Counter-move heuristic** (they rely on cont-hist instead)
4. **ProbCut** (they don't have it)
5. **NMP threat detection** (our +8000 escape bonus)
6. **QS beta blending** (our (bestScore+beta)/2)
7. **TT score dampening** (our (3*score+beta)/4 on cutoffs)
8. **Fail-high score blending** (our (score*d+beta)/(d+1))
9. **Recapture extensions**
10. **Lazy NNUE accumulator** (they do eager updates)
11. **TT near-miss cutoffs** (our 1-ply/64cp relaxation)
12. **TT noisy move detection** for LMR (our merged +34.4 Elo feature)

### Parameter Comparison Table

| Feature | Altair | GoChess |
|---------|--------|---------|
| RFP margin | 110*(d-improving), d<=7 | 85*d (imp) / 60*d (not), d<=8 |
| RFP return value | (eval+beta)/2 | raw eval |
| NMP base R | 5 | 3 |
| NMP depth divisor | 4 | 3 |
| NMP eval divisor | 400 | 200 |
| NMP min depth | 1 | 3 |
| NMP verification | depth > 15 | depth > 12 |
| LMR table | single, div=2.30, base=1.40 | split cap/quiet C=1.80/1.50 |
| LMR history div | 10000 | 5000 |
| LMP tier 1 | depth*10, d<=4 | 3+d^2 (+50% improving) |
| LMP tier 2 | 2+d^2/(1+!imp+fail) | N/A |
| Futility | (d-!imp)*170+70, d<=7 | 80+d*80 |
| History pruning | -9600*(d+imp), d<=10 | -2000*d |
| SEE quiet margin | -40*d, d<=2+5*pv | -20*d^2 |
| SEE noisy margin | -85*d | -20*d^2 |
| Singular depth | >=6 | >=10 |
| Singular beta | tt_score - depth | tt_score - 3*depth |
| Double ext margin | 12 | N/A |
| Double ext limit | 7 | N/A |
| Aspiration delta | 6+85/(asp_d-2) | 15 (UPDATE: now with contraction) |
| ContHist plies | 1, 2, 4 | 1, 2 |
| History max | quiet=9000, noisy=12000, cont=11000 | ~10000 |
| Correction tables | 4 (pawn/np/major/minor) | 1 (pawn) |
| King buckets | 5 | 16 |
| Output buckets | 8 | 8 |
| Hidden layer | 1024 | Dynamic (1024/1536/768pw/any) |
| Activation | SCReLU | CReLU/SCReLU (UPDATE: now supports both) |

---

## 13. Ideas Worth Testing from Altair

### Already Tracked in SUMMARY.md (new evidence from Altair):
1. **Threat-aware history** (#5, now 12+ engines including Altair) -- `[piece][to][from_threat][to_threat]`
2. **Multi-source correction history** (#7, now 11+ engines) -- Altair's 4-table approach with separate major/minor hash keys
3. **"Failing" heuristic** (#29) -- position deterioration flag in LMR/LMP
4. **Alpha-raised count in LMR** (#30) -- progressive reduction scaling
5. **Continuation history ply 4** (#20) -- Altair uses at full weight, our test failed at full weight
6. **IIR extra PV reduction** (#39) -- depth -= 1 + pvNode

### New Ideas from Altair:
7. **NMP base R=5** -- their NMP is significantly more aggressive (R=5, div=4, eval_div=400). Combined with the activation gate condition, this could be a net win. Worth testing R=4 first.
8. **SEE pruning history gate** -- skip SEE pruning when history > threshold. Prevents over-pruning tactically important moves. Easy: 1 condition.
9. **Null-move killer protection** -- store refutation in slot 1 instead of overwriting slot 0 after NMP. Trivial.
10. **Position-phase time scaling** -- material+pawn structure to allocate less time in openings/endgames.
11. **Asymmetric score-based time scaling** -- score drops extend time more than score rises. **(UPDATE 2026-03-21: GoChess now has score-drop time extension 2.0x/1.5x/1.2x, merged)**
12. **MaxHeap for expected-all nodes** -- build heap for PV/exact TT nodes after first quiet. Could save move ordering time.

### Priority for GoChess Testing:
1. **Mate distance pruning** (3 lines, free, already in queue)
2. **4-table correction history** (Altair confirms major/minor split works; our XOR approach failed, but proper hash keys should work)
3. **Threat-aware history** (Altair's boolean from_threat x to_threat indexing is the simplest variant)
4. **NMP R=4** (split the difference between our R=3 and their R=5)
5. **SEE pruning history gate** (1 line, protects good moves from being SEE-pruned)
6. **Failing heuristic** (3 lines, cheap)
7. **Alpha-raised count in LMR** (3 lines, novel idea with two engines now)
