# Velvet Chess Engine - Crib Notes

Source: `~/chess/engines/velvet-chess/engine/src/`
Author: Martin Honert (mhonert)
Version: 9.0.0-dev7
Language: Rust (like Coda)
Eval: NNUE (768-wide CReLU, 32 king buckets, embedded weights)
CCRL Blitz 1T: ~3660 Elo (8T: 3736). Approximately +100 Elo above Coda.

---

## 1. Overview and Architecture

Velvet is a UCI chess engine written in Rust with NNUE evaluation. It has an embedded neural network (weights compiled into the binary via `include_bytes!`). The engine features Lazy SMP, Syzygy tablebase support (via fathomrs crate), configurable playing strength (UCI_LimitStrength with node limits), and a tunable search parameter framework.

### Board Representation
- Bitboards via `BitBoards` struct (piece-type + color bitboards)
- Mailbox: `items: [i8; 64]` for O(1) piece lookup (signed: positive = white, negative = black)
- Magic bitboards for sliding pieces (classic magic multiply, NOT PEXT)
- Magics initialized at runtime via `std::sync::Once` (Coda detects PEXT at runtime and uses it when available)
- Cuckoo cycle detection tables also initialized at runtime

### Rust-Specific Notes
- SIMD: compile-time feature gating via `#[cfg(target_feature = "avx2")]` etc. Supports AVX-512 (feature-gated), AVX2, SSE4.1, and NEON (aarch64). No runtime detection -- relies on compile flags.
- Coda uses runtime SIMD detection via `std::is_x86_feature_detected!` which is more flexible but has a small overhead.
- Embedded NNUE weights via `include_bytes!` + `transmute_copy` into aligned arrays. No external .nnue file loading.
- Aligned memory: custom `A64<T>` wrapper for 64-byte alignment (same concept as Coda's alignment).
- Uses `unsafe` for magic table lookups with `get_unchecked` and raw pointers for performance.
- Thread communication via `std::sync::mpsc` channels (Coda uses shared atomics + condvar).

---

## 2. NNUE Architecture

### Network Structure
- **Architecture**: (32 king buckets) x (6 pieces x 64 squares x 2 colors) = 24576 inputs per bucket
- **Hidden layer**: 768 per perspective (1536 total = `HL1_NODES`). Single hidden layer, NO deeper layers.
- **Output**: 1 node (single output, no output buckets)
- **Activation**: CReLU (clipped ReLU, clamp [0, 255])
- **Quantization**: FP_IN = 6 bits (multiplier 64), FP_OUT = 6 bits (multiplier 64), SCORE_SCALE = 256

### King Buckets
- 32 king buckets: `row * 4 + (col & 3)` -- 8 rows x 4 columns (horizontally mirrored)
- Feature index: `bucket_offset + piece_idx * 64 * 2 + pos ^ xor_pov`
- Horizontal mirroring via XOR of position bits

### Accumulator Updates
- **Per-bucket accumulator cache**: maintains `BUCKETS * 2 = 64` accumulator states (32 per perspective, double-buffered)
- **Delta vs full refresh**: compares delta cost (popcount of changed squares) vs full refresh cost (total piece count). Uses cheaper option.
- **Lazy update via update queue**: moves push `UpdateAction` variants (RemoveAdd, RemoveRemoveAdd, RemoveAddAdd) to a Vec. Applied lazily before eval.
- **Fast undo**: on undo_move, pops update entries instead of recomputing (O(1) undo when updates haven't been applied yet)
- No Finny table per se, but the per-bucket cache with delta/refresh decision serves a similar purpose.

### SIMD Forward Pass
- AVX-512: processes `VALUES_PER_REG = 32` i16 values per register, 2 chunks per loop iteration
- AVX2: processes 16 i16 values per register, 4 chunks per loop iteration
- SSE4.1: processes 8 i16 values per register, 8 chunks per loop iteration
- NEON (aarch64): same structure as SSE4.1
- Forward pass: CReLU clamp, VPMADDWD dot product with output weights (same as Coda's CReLU path)
- Input weights stored as `i8`, biases as `i8`, output weights as `i8`

### Eval Scaling
- `clock_scaled_eval`: `eval * (128 - halfmove_clock) / 128` -- linear decay with 50-move counter
- Applied both in search (via `get_or_calc_eval`) and in score reporting
- Coda does NOT have 50-move eval scaling.

---

## 3. Search Features

### Iterative Deepening
- Depth 1 to `limits.depth_limit` for main thread
- Helper threads: depth 1 to `MAX_DEPTH` (255)
- Single root move: `time_mgr.reduce_timelimit()` divides time by 32

### Aspiration Windows
- Initial delta: 16 (constant, not score-dependent)
- Starts at depth > 7 (later than Coda's depth 4)
- On fail-low: `beta = (alpha + beta) / 2`, widen alpha by step
- On fail-high: widen beta by step, reduce depth by 1 (max -5 from original), only for non-mate scores
- Step growth: `max(delta, prev_step) * (4/3)^attempt` -- exponential growth
- **Coda comparison**: Coda uses score-dependent initial delta (13+avg^2/23660), starts at depth 4. Velvet's fixed delta=16 is simpler.

### Draw Detection
- Repetition: standard repetition via `PositionHistory`
- 50-move rule: `halfmove_clock >= 100`, with checkmate test in check positions
- Upcoming repetition: Cuckoo-based cycle detection (`has_upcoming_repetition`). Applied when alpha < 0.
- Draw score: `-1 + (node_count & 2)` -- oscillates between -1 and +1 (draw randomization)
- Insufficient material detection
- **Coda comparison**: Coda has Cuckoo detection. Both have draw randomization (Velvet's is simpler).

### Check Extension
- `depth = (depth + 1).max(1)` when in check, only when NOT in SE search
- Applied before move loop

### Mate Distance Pruning
- `alpha = max(alpha, MATED_SCORE + ply)`
- `beta = min(beta, MATE_SCORE - ply - 1)`
- Cutoff if alpha >= beta
- **Coda comparison**: Coda does NOT have mate distance pruning.

---

## 4. Pruning Techniques

### Reverse Futility Pruning (RFP)
- Depth guard: `depth <= 8`
- Margin: `rfp_margin_multiplier * (depth - improving)` = **64 * (depth - improving)**
- Uses `ref_score` (TT-adjusted eval) not raw eval
- Additional guard: `!is_mate_or_mated_score(beta) && !is_mate_or_mated_score(ref_score)`
- Return: `beta + (ref_score - beta) / 2` -- blended return, not just beta!
- **Coda comparison**: Coda uses 70*d (improving) / 100*d (not improving), depth<=7. Coda returns staticEval-margin. Velvet's blended RFP return is notable.

### Razoring
- Conditions: `!improving && depth <= 4 && ref_score + (1 << (depth-1)) * 200 <= alpha`
- Margins by depth: d1=200, d2=400, d3=800, d4=1600 (exponential growth)
- Falls through to QS: if QS score <= alpha, returns QS score
- **Coda comparison**: Coda uses depth<=2, margin 400+d*100 (linear). Velvet's exponential margins and deeper razoring (d<=4) are different.

### Null Move Pruning (NMP)
- Conditions: `ref_score >= beta && !in_se_search && has_non_pawns(active_player)`
- Additional guard: `!(tt_score_type == UpperBound && tt_depth >= reduced_depth)` -- skips NMP when TT suggests position might not be as good
- Reduction: `(768 + depth * 256^2 / 668) / 256` = approximately `3 + depth * 98.2 / 256` which simplifies to roughly `3 + depth * 0.384`
- At depth 10: R = 3 + 3.84 = ~6.8, so R = 6
- Verification at depth >= 12: reduces to `reduced_depth` instead of returning score
- Mate score guard: returns beta instead of mate score
- No NMP after null move (not explicitly checked, but se_move == NO_MOVE ensures it)
- **Coda comparison**: Coda uses R=4+d/3+(eval-beta)/200. Velvet's NMP has no eval-beta component, just depth-based reduction. Velvet does NOT have NMP score dampening.

### ProbCut
- Margin: `beta + 152`
- Depth: `depth - 4` (reduced search depth)
- Conditions: `!in_se_search && tt_move != NO_MOVE && !UpperBound(tt_score_type) && tt_score >= prob_cut_beta && prob_cut_depth > 0 && is_eval_score(beta)`
- Only tries the TT move (not a full capture search!)
- Makes the TT move, does a full reduced search, cuts if score >= prob_cut_beta
- **Coda comparison**: Coda has ProbCut with beta+170. Velvet's ProbCut is simpler (only tries TT move, no SEE filtering of captures).

### Futility Pruning
- Conditions: `!is_pv && !improving && depth <= 6 && !in_check`
- Margin: `(fp_margin_multiplier << depth) + fp_base_margin` = **(22 << depth) + 17**
- By depth: d1=61, d2=105, d3=193, d4=369, d5=721, d6=1425
- Applied as pruning flag `allow_futile_move_pruning`
- Futile quiet moves get `+2 reductions` (FUTILE_MOVE_REDUCTIONS)
- If reductions >= depth-1 AND has non-pawns AND !gives_check AND !queen_promotion: prune entirely
- **Coda comparison**: Coda uses 60+lmrDepth*60, depth<=8. Velvet's exponential futility margins are much wider at higher depths.

### Late Move Pruning (LMP)
- Conditions: `!is_pv && !in_check && has_non_pawns && depth <= 4` (lmp_max_depth)
- Improving: `(d^2 + 3) * 65 / 64` -- roughly d^2 + 3
- Not improving: `(d^2 + 2) * 35 / 64` -- roughly (d^2 + 2) * 0.55

| Depth | Improving | Not improving |
|-------|-----------|---------------|
| 1     | 4         | 1             |
| 2     | 7         | 3             |
| 3     | 12        | 6             |
| 4     | 19        | 9             |

- **Coda comparison**: Coda uses 3+d^2 with improving 1.5x / failing 0.67x, depth<=8. Velvet is tighter (especially not-improving) and only goes to depth 4.

### IIR (Internal Iterative Reduction)
- `!is_tt_hit && depth > 3`: reduce depth by 1
- Applied once, no separate PV/cut-node distinction
- **Coda comparison**: Coda uses depth>=4 (equivalent threshold). Both similar.

### SEE Pruning
- No dedicated SEE pruning pass in the move loop
- Instead, moves with negative SEE are:
  1. Placed in `bad_capture_moves` list during move generation (sorted last)
  2. Given `+2 reductions` if quiet with negative SEE and negative history
  3. Given `+2 reductions` (NEG_SEE_REDUCTIONS) if quiet with negative SEE and score < QUIET_BASE_SCORE
- **Coda comparison**: Coda has explicit SEE pruning thresholds (-20d^2 quiet, -d*100 capture). Velvet uses SEE more as a reduction signal than a hard prune.

---

## 5. LMR (Late Move Reductions)

### Table Initialization
- LMR table: `lmr[moves] = log2(base + moves / divisor)`
- Base: `253/256 = 0.988`, Divisor: `1024/256 = 4.0`
- Formula: `log2(0.988 + moves/4.0)` as i16
- Values: moves=1 -> 0, moves=2 -> 0, moves=3 -> 0, moves=4 -> 1, moves=8 -> 1, moves=13 -> 2, moves=25 -> 2, moves=49 -> 3

### Application Conditions
- `depth > 2 && quiet_move_count > 1 && !queen_promotion`
- Only for quiet moves (captures handled separately)

### Reduction Adjustments
- `+= 1` if not PV (`!is_pv`)
- `+= 1` if singular move found OR tt_move is capture/queen_promotion
- `-= history_diff` if `!is_pv && !improving && history_diff < 0` (bad history increases reduction)
- `-= 1` if `is_pv && improving && history_diff > 0` (good history + PV + improving reduces)
- `+= 1` if negative SEE score AND negative history score (below QUIET_BASE_SCORE)
- `-= 1` if pawn on passed pawn square
- `-= 1` if killer move (applied after all other adjustments)
- `+= 1` if `!is_pv && next_ply_cutoff_count > 3` (applied to ALL moves, not just quiets)
- No minimum clamp (reductions can go to 0 or negative, meaning extensions)

### Capture Reductions
- Bad captures: `+= 2` if `!in_check && !gives_check && is_bad_capture && !improving`
- Not using LMR table for captures at all

### Re-search
- If reduced search beats alpha: re-search at full depth with full window
- No doDeeper/doShallower logic

### Futile Move Reductions
- Quiet moves matching futility criteria: `+= 2` (FUTILE_MOVE_REDUCTIONS)
- Passed pawn adjustment: `-= 1`
- If total reductions >= depth-1: prune entirely

### Negative History Reductions
- Moves with score <= NEGATIVE_HISTORY_SCORE: `+= 2` (NEG_HISTORY_REDUCTIONS)

**Coda comparison**: Coda uses LMR tables with `log(d)*log(m)/C` formula (C_quiet=1.30, C_cap=1.80). Coda has doDeeper/doShallower. Velvet's LMR is simpler with more ad-hoc adjustments (cutoff count, singular, passed pawn). No cut-node +2 like some engines.

---

## 6. Extensions

### Singular Extensions
- Depth guard: `depth >= 6` (higher than Weiss's 5, lower than Coda's 8)
- Conditions: `!in_se_search && tt_move != NO_MOVE && !in_check && !UpperBound(tt_score_type) && !is_mate_or_mated_score(tt_score) && tt_depth >= depth - 3`
- Singular beta: `tt_score - depth`
- Singular depth: `depth / 2`
- **Single extension (+1)**: result < se_beta
- **Double extension (+2)**: `!is_pv && result + se_double_ext_margin(4) < se_beta && double_extensions < se_double_ext_limit(12)`
- **Multi-cut**: `se_beta >= beta` -> return se_beta
- **Negative extension (-1)**: `tt_score <= alpha || tt_score >= beta` (TT score unhelpful)
- Double extension counter propagated via `PlyEntry` (incremented on double ext, inherited from parent)
- **Coda comparison**: Coda uses SE at depth>=8, margin tt_score-depth, multi-cut, negative ext. Coda has NO double extensions. Velvet's lower SE depth (6 vs 8) and double extensions with limit 12 are key differences.

### Check Extension
- `depth += 1` when in check, only outside SE search
- Applied before the move loop (unconditional)

---

## 7. Move Ordering

### Staged Move Picker
Stages: HashMove -> GenerateMoves -> CaptureMoves -> PrimaryKiller -> SecondaryKiller -> CounterMove -> PostponedBadCaptures -> GenerateQuiets -> QuietMoves

### Capture Scoring
- `captured_piece_id * 16 - moving_piece_id` (simple MVV-LVA, no capture history)
- Captures sorted by score, bad captures (negative SEE) deferred to PostponedBadCaptures stage
- **Coda comparison**: Coda uses MVV-LVA + captHist/16. Velvet has NO capture history.

### Quiet Scoring
- `QUIET_BASE_SCORE + history_score + queen_promotion_bonus`
- History score = follow_up_score + counter_score (from HistoryTable)
- No separate butterfly history, pawn history, or continuation history in quiet scoring
- **Coda comparison**: Coda uses main hist + contHist*3 + pawn hist. Velvet's quiet scoring is much simpler.

### Priority Moves
- 2 killers per ply (primary + secondary)
- Counter-move heuristic (indexed by opponent's piece-end combination, 512 entries)
- Checked priority moves tracked in array to avoid duplicates

### History Tables

**HistoryTable** (combined counter + follow-up):
- Structure: `[512][2][512]` -- indexed by `(related_move.piece_end_index, color, current_move.piece_end_index)`
- Each entry stores TWO values: `(counter_score, follow_up_score)`
- Counter: indexed by opponent's last move
- Follow-up: indexed by own previous move (2 plies ago)
- Gravity: `value += scale * 4 - value / 32`
- Scale: +1 (positive history) or +4 (no positive history) for best move; -1 for played non-best moves
- Range: i8 (-128 to 127)
- **No butterfly history, no pawn history, no continuation history at depth > 2**
- **Coda comparison**: Coda has butterfly[2][64][64], pawn[512][12][64], capture[12][64][12], cont[12][64][12][64] at plies 1,2,4,6, plus killers and counter-move. Velvet's history is MUCH simpler -- just counter + follow-up in a single combined table.

---

## 8. Correction History

### Components (4 sources)
1. **Pawn correction**: `[2][16384]` indexed by pawn hash
2. **Non-pawn correction (white)**: `[2][16384]` indexed by white non-pawn hash
3. **Non-pawn correction (black)**: `[2][16384]` indexed by black non-pawn hash
4. **Move correction**: `[2][16384]` indexed by `move_history_hash` (Zobrist of own_move XOR opp_move)

### Blending
- Simple sum: `pawn + white_non_pawn + black_non_pawn + move_corr`
- Clamped to [-255, 255]
- No per-component weights (unlike Weiss's weighted blend)
- Stored as fixed-point: `CORR_HISTORY_GRAIN = 256`, max = `64 * 256 = 16384`
- Score extraction: `value / 256`

### Update
- Weight = depth (linear, up to MAX_DEPTH + 1)
- Formula: `((MAX_WEIGHT - weight) * old + diff * weight * GRAIN) / MAX_WEIGHT`
- Applied when:
  - Beta cutoff: `!(in_check || best_move.is_capture() || is_mate_or_mated_score(best_score) || best_score <= eval)`
  - Alpha improvement: `!(in_check || best_move.is_capture() || is_mate_or_mated_score(best_score) || (UpperBound && best_score >= eval))`
- Diff: `best_score - eval`

### Move History Hash
- Unique feature: correction indexed by a hash of the last two moves (own + opponent)
- Computed as `piece_zobrist_key(opp_piece, opp_end) XOR piece_zobrist_key(own_piece, own_end)`
- This captures move-order-dependent eval errors

**Coda comparison**: Coda has pawn (512) + white-NP (204) + black-NP (204) + continuation (104) correction with weights 512/1024, 204/1024, 204/1024, 104/1024. Velvet has 4 sources with equal weights and 16384 entries each. Velvet's move-correction is unique. Coda has more sources but Velvet has more entries per source.

---

## 9. Transposition Table

### Structure
- 8-slot segments, each slot is an `AtomicU64` (64-bit)
- Segment = 64 bytes (cache-line aligned)
- Each entry packs into 64 bits: 20-bit hash check + 32-bit move+score + 2-bit clock + 2-bit score type + 8-bit depth
- Index: `hash & index_mask` (power-of-2 sizing)

### Slot Selection
- Slot ID derived from `halfmove_clock`: `(clock >> 2) >> 2 + offset_based_on_hash_bit_40`
- Clock bits: `(clock >> 2) & 0b11` stored in entry for age detection
- This means different game phases naturally use different slots (clever aging mechanism)

### Replacement Policy
- Same hash check + Exact type or `new_depth >= old_depth - 4`: overwrite
- Different hash: always overwrite
- Single-entry probe (no bucket scanning for best replacement target)

### Eval Cache
- Separate eval caching path: `get_or_calc_eval` uses first slot of segment
- Hash check uses `EVAL_HASHCHECK_MASK` (more bits than TT entry hash check)
- Stores raw eval + adds correction on retrieval
- **Coda comparison**: Coda stores raw eval in TT entries. Velvet has a dedicated eval cache slot.

### Prefetch
- `_mm_prefetch::<_MM_HINT_T0>` on x86, `__prefetch` on aarch64
- Called after `perform_move` (same as Coda)

### TT Cutoffs
- Complex clock-aware logic: entries matching current halfmove_clock get standard cutoffs
- Non-matching clock entries: additional `clock_scaled_eval` adjustment before cutoff check
- PV nodes: cutoffs only at `depth <= 0`
- **Coda comparison**: Coda uses 5-slot buckets with XOR key verification, generation-based replacement. Velvet's clock-based slot selection is unique. Coda has TT score dampening and near-miss cutoffs which Velvet lacks.

---

## 10. Time Management

### Base Allocation
- `time_for_move = time_left / moves_to_go` (default mtg=25)
- Add increment if `time_for_move + increment <= time_left`
- Subtract `move_overhead_ms` (default 20ms) from time_left first

### Time Extension
- Extension multiplier: **3x** (triple the time limit)
- Triggered when: `best_move_changed() || score_dropped()`
- `best_move_changed`: current iteration best != previous iteration best
- `score_dropped`: previous score > current score
- Only one extension allowed per search (flag cleared after extension)
- If already extended and neither condition holds: stop early

### Iteration Decision
- `estimated_iteration_duration = previous_iteration * 7/4`
- Stop if `remaining_time <= estimated * 7/4 AND best_move_consistent`
- `best_move_consistent`: all iterations returned the expected best move
- `expected_best_move`: set via ponder hit

### Single Root Move
- Divides time by 32

### No Node-Based TM
- No node-ratio scaling, no continuous time scaling
- Simpler than Weiss's node-ratio approach, similar simplicity to Coda

**Coda comparison**: Coda uses time_left/20 + 3*inc/4 soft, 5x hard. Both are simple TM systems. Velvet's 3x extension on instability is more aggressive than Coda's fixed allocation.

---

## 11. Lazy SMP

- Helper threads via `std::sync::mpsc` channels (Coda uses shared state + atomics)
- Shared: TT only (via `Arc<TranspositionTable>`, `AtomicU64` entries)
- Per-thread: Board, HistoryHeuristics, SearchContext
- Global node count and TB hits via `Arc<AtomicU64>` with periodic bulk updates
- Helper threads search from depth 1 to MAX_DEPTH (no depth offset like some engines)
- Stop signal: `Arc<AtomicBool>` shared between all threads (same as Coda)
- **Coda comparison**: Very similar design. Coda also uses shared TT + per-thread history. Main difference is communication mechanism (channels vs shared state).

---

## 12. Other Notable Features

### Syzygy Tablebases
- Via `fathomrs` crate (Rust bindings to Fathom)
- Root probing: returns TB result + skipped moves
- In-search probing: at `depth >= tb_probe_depth` (configurable, default 5)
- TB results: Win/Loss get ply-adjusted scores, Draw/CursedWin/BlessedLoss get near-zero
- **Coda comparison**: Coda also has Syzygy support.

### Configurable Strength
- UCI_LimitStrength with node limits per move (72 Elo steps from 1225-3000)
- Simulated thinking time feature
- Multiple playing styles via different embedded nets
- **Coda comparison**: Coda does not have strength limiting or style options.

### Tunable Parameters
- `tune` feature flag exposes all search params as UCI spin options
- Params: fp_base_margin=17, fp_margin_multiplier=22, razor_margin_multiplier=200, rfp_margin_multiplier=64, nmp_base=768, nmp_divider=668, se_double_ext_margin=4, se_double_ext_limit=12, prob_cut_margin=152, prob_cut_depth=4, lmr_base=253, lmr_divider=1024, lmp_max_depth=4
- **Coda comparison**: Coda uses env-var feature flags for ablation but not tunable numeric params at runtime.

---

## 13. QSearch

- TT probe with cutoffs (all score types)
- Eval from `get_or_calc_eval` (with correction history and clock scaling)
- Stand pat: if position_score >= beta, return immediately
- TT move searched first if available
- In check: searches all evasions first, then captures
- Captures: sorted by MVV-LVA, only "good" captures (SEE-filtered)
- No delta pruning (relies on SEE filtering)
- TT writes on cutoffs and best moves found
- Upcoming repetition detection in QS (via Cuckoo)
- **Coda comparison**: Coda has QS delta pruning (240), QS TT probe. Velvet's QS has no delta pruning but does have upcoming repetition detection and TT writes for non-cutoff best moves.

---

## 14. Parameter Comparison Table

| Feature | Velvet | Coda |
|---------|--------|------|
| RFP margin | 64*(d-improving), depth<=8 | 70*d (imp) / 100*d (not), depth<=7 |
| RFP return | beta + (ref-beta)/2 (blended) | staticEval - margin |
| Razoring | depth<=4, 200*(2^(d-1)) | depth<=2, 400+d*100 |
| NMP reduction | ~3 + d*0.384 | 4 + d/3 + (eval-beta)/200 |
| NMP verify | depth >= 12 | depth >= 12 |
| NMP dampening | None | (2*score+beta)/3 |
| NMP TT guard | Skip if TT UpperBound at reduced depth | None |
| ProbCut margin | beta + 152 | beta + 170 |
| ProbCut method | TT move only | Full capture search |
| Futility | (22<<d)+17, depth<=6, exponential | 60+lmrD*60, depth<=8, linear |
| LMR formula | log2(0.988 + m/4.0) | ln(d)*ln(m)/1.30 |
| LMR depth-dependent | No (move-count only) | Yes (depth x move) |
| LMR cutoff count | +1 if next_ply cutoffs > 3 | None |
| LMR passed pawn | -1 for passed pawn push | None |
| LMP (improving) | (d^2+3)*65/64, depth<=4 | (3+d^2)*1.5, depth<=8 |
| LMP (not improving) | (d^2+2)*35/64, depth<=4 | (3+d^2)*0.67, depth<=8 |
| SE min depth | 6 | 8 |
| SE margin | tt_score - depth | tt_score - depth |
| SE double ext | +2 if < se_beta-4, limit 12 | None |
| Aspiration delta | 16 (fixed) | 13+avg^2/23660 (score-dep) |
| Aspiration start | depth > 7 | depth >= 4 |
| History tables | Counter+followup (i8, 512x2x512) | Butterfly+cont+pawn+capture |
| Capture history | None | Yes [12][64][12] |
| Pawn history | None | Yes [512][12][64] |
| Cont-hist plies | Effectively 1, 2 (counter + followup) | 1, 2, 4, 6 |
| Killers | 2 per ply | 2 per ply |
| Counter-move | Yes (512 entries) | Yes [12][64] |
| Correction sources | 4 (pawn, NP-W, NP-B, move-corr) | 6 (pawn, NP-W, NP-B, minor, major, cont) |
| Correction entries | 16384 each | 512-1024 |
| Correction weights | Equal (unweighted sum) | Weighted (512/204/204/104 out of 1024) |
| King buckets | 32 (8x4 mirror) | 16 |
| NNUE activation | CReLU | CReLU/SCReLU/Pairwise |
| NNUE hidden layers | 1 (768 per side) | v5: 1, v7: 2-3 |
| NNUE loading | Embedded (include_bytes!) | External .nnue file |
| 50-move eval scaling | Yes (linear decay) | No |
| Mate distance pruning | Yes | No |
| Draw randomization | -1 or +1 (node_count & 2) | None |
| TT slots | 8 per segment | 5 per bucket |
| TT key verification | 20-bit hash check | XOR key (full hash) |
| TT aging | Clock-based slot selection | Generation counter |
| SIMD detection | Compile-time (#[cfg]) | Runtime (is_x86_feature_detected!) |
| Attack tables | Runtime magic init (Once) | Runtime magic + PEXT detection |

---

## 15. Ideas for Coda from Velvet

### High priority (likely Elo gains):

1. **Mate distance pruning** -- 5 lines, universal technique. Every strong engine has it. Simple alpha/beta clamping at each node. EASY WIN.
2. **Double extensions** (+2 when well below singular beta, capped). Velvet uses margin=4, limit=12. Coda has no double extensions. HIGH PRIORITY.
3. **Lower SE depth** (6 vs Coda's 8) -- more aggressive SE activation. Medium risk.
4. **50-move eval scaling** -- `eval * (128 - halfmove_clock) / 128`. Applied in eval. Simple, reduces draw-blindness in long games.
5. **Draw randomization** -- small oscillating draw score to avoid draw blindness. Velvet uses -1/+1 based on node count.
6. **RFP blended return** -- `beta + (ref_score - beta) / 2` instead of just returning eval-margin. Dampens RFP overconfidence.

### Medium priority:

7. **Cutoff count LMR adjustment** -- `+1 reduction if next_ply_cutoff_count > 3`. Cheap heuristic, reduces search in positions where opponent consistently gets cutoffs. NOVEL.
8. **Move correction history** -- correction indexed by Zobrist hash of last two moves. Captures move-order-dependent eval errors. Velvet's 4th correction source.
9. **NMP TT upper bound guard** -- skip NMP when TT says position has upper bound at sufficient depth. Prevents wasteful null move searches.
10. **Passed pawn LMR reduction** -- `-1 for passed pawn pushes`. Simple positional awareness in LMR.
11. **Embedded NNUE weights** -- compile net into binary via `include_bytes!`. Eliminates file loading, simplifies distribution. QUALITY OF LIFE.
12. **Larger correction history tables** -- 16384 entries vs Coda's 512-1024. More precision.

### Lower priority / already covered:

13. **IMPLEMENTED**: Killers (2 per ply)
14. **IMPLEMENTED**: Counter-move heuristic
15. **IMPLEMENTED**: Cuckoo cycle detection (upcoming repetition)
16. **IMPLEMENTED**: SE multi-cut
17. **IMPLEMENTED**: SE negative extension
18. **IMPLEMENTED**: NMP verification at depth >= 12
19. **IMPLEMENTED**: TT prefetch after make_move
20. **IMPLEMENTED**: Lazy SMP
21. **IMPLEMENTED**: Syzygy tablebases
22. **IMPLEMENTED**: IIR

---

## 16. Things Coda Has That Velvet Doesn't

1. **Capture history** -- Velvet has NO capture history at all. Coda uses [piece][to][victim].
2. **Pawn history** -- Velvet has none. Coda uses [pawnHash%512][piece][to].
3. **Deep continuation history** (plies 4, 6) -- Velvet only has effective plies 1-2.
4. **SCReLU / Pairwise activations** -- Velvet uses only CReLU.
5. **Multi-layer NNUE** (v7 hidden layers) -- Velvet is single hidden layer.
6. **More king buckets variety** (16 in Coda vs 32 in Velvet, but Coda tests various configs)
7. **Runtime SIMD detection** -- Velvet requires compile-time flags.
8. **PEXT for magic bitboards** -- Velvet uses classic magic multiply only.
9. **Deeper LMP** (depth <= 8 vs Velvet's depth <= 4)
10. **Fail-high score blending** in main search
11. **TT score dampening** at cutoffs
12. **TT near-miss cutoffs** (1 ply shallower with margin)
13. **NMP score dampening** ((2*score+beta)/3)
14. **Hindsight reduction** (doDeeper/doShallower)
15. **Recapture extensions**
16. **Weighted correction history** (per-component weights)
17. **More correction sources** (minor + major piece corrections)
18. **Depth x move LMR tables** (Velvet's LMR only depends on move count, not depth)
19. **QS delta pruning** (240)
20. **Contempt** (Coda has -10)

---

## 17. Key Takeaways

Velvet is ~100 Elo stronger than Coda despite having simpler infrastructure in several areas:
- **Simpler history**: just counter + follow-up in one table (no butterfly, no pawn history, no capture history)
- **Simpler move ordering**: basic MVV-LVA for captures, no capture history integration
- **Simpler LMR**: move-count-only table (not depth x move), fewer adjustments
- **Single-layer CReLU NNUE**: no SCReLU, no pairwise, no hidden layers

This suggests Velvet's Elo advantage comes from:
1. **Better training data / net quality** (768-wide with 32 king buckets, well-trained)
2. **Double extensions** (significant tree extension for critical moves)
3. **Lower SE depth** (6 vs 8 -- more singular extensions fire)
4. **Mate distance pruning** (universal safety feature)
5. **50-move eval scaling** (better endgame handling)
6. **Clock-based TT aging** (more effective TT utilization)
7. **RFP blended return** (less pruning overconfidence)

The biggest low-hanging fruit for Coda: **mate distance pruning + double extensions + lower SE depth**. These are consensus features that Velvet has and Coda lacks.

---

## 18. GPU Trainer Analysis (2026-04-06)

Velvet uses a **custom PyTorch trainer** (via `tch-rs` Rust bindings to libtorch),
NOT Bullet. Key differences from our Bullet-based training:

### Loss Function: Power 2.6
```rust
const ERR_EXP: f64 = 2.6;
// loss = abs(predicted - target).pow(2.6).mean()
```
NOT MSE (power 2.0). Cosmo's research found power-2.5 gained +16-24 Elo.
Velvet uses 2.6 which penalizes large errors even more heavily.

### LR Schedule: Patience-Based Decay (NOT cosine)
- Initial LR: 0.001
- On validation plateau: multiply LR by 0.4
- Initial patience: 8 epochs (each epoch = 200M positions)
- Patience halves on each decay: 8 → 4 → 2 → 2 → ...
- Stops at LR ≤ 0.00000166
- Reloads best model checkpoint on patience exhaustion

This is classic ML early-stopping + LR decay, completely different from our
fixed cosine schedule. It adapts to training dynamics — decays faster when
learning stalls, keeps high LR while making progress.

### Optimizer: AdamW with weight decay 0.01
```rust
let mut opt = nn::AdamW::default().build(vs, initial_lr).unwrap();
opt.set_weight_decay(0.01);
```
Our Bullet configs don't set explicit weight decay (uses Bullet's default).

### Batch Size: 32000
Double our 16384. Larger batches = more stable gradients = potentially better
for single-layer training.

### Target: Sigmoid of Scaled Score (no WDL)
```rust
// K_DIV = K / (400.0 / SCORE_SCALE)
target = (label * K_DIV).sigmoid()
predicted = forward(inputs).sigmoid() * K_DIV
```
Pure score-based, no WDL blending. Score is scaled and passed through sigmoid
before loss computation.

### Validation-Based Checkpointing
- Checks validation error every 200M positions
- Saves model only when validation improves
- Reloads best model on patience exhaustion
- Always saves after first 4 epochs regardless of improvement

### Data Format
- Custom LZ4-compressed format, NOT binpack
- 200K positions per set file
- Self-play data, not Stockfish/LC0 data
- Shuffled per batch

### Implications for Coda Training
1. **Power-2.6 loss**: High priority to test. Can be set in Bullet via custom loss_fn.
2. **Patience-based LR**: Would need custom schedule in Bullet (not built-in).
3. **Weight decay 0.01**: Should add to our AdamW config.
4. **Batch size 32K**: Easy to test in Bullet.
5. **No WDL**: Confirms that pure score training can work well.
6. **Validation checkpointing**: Bullet doesn't support this — would need wrapper.
