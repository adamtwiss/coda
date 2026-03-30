# Black Marlin Chess Engine - Crib Notes

Source: `~/chess/engines/blackmarlin/`
Author: Doruk Sekercioglu (Rust)
Rating: #13 in RR at +95 Elo (203 above GoChess-v5)
NNUE: HalfKA with threats (28672->Nx2->1x8), SCReLU, 32 king buckets (file-mirrored), output buckets

---

## 1. NNUE Architecture

### Network Topology
- **Input**: 32 king buckets (64 squares, horizontally mirrored when king on files e-h) x 2 colors x 7 feature types x 64 squares = **28672 features**
- The 7th "piece type" is a **threat feature** -- squares where pieces can be captured by weaker pieces (pawn attacks pieces, minor attacks majors, rook attacks queen). This is baked into the NNUE input, not just a search heuristic.
- **Hidden layer**: MID neurons (architecture-dependent, read from network file header -- 3 u32 values for INPUT, MID, OUTPUT)
- **Output layer**: Dense MID*2 -> OUTPUT, with **output buckets** selected by material count: `bucket = min(7, (63-pc)(32-pc)/225)` -- identical non-linear formula to Alexandria
- **Activation**: SCReLU -- `clamp(x, 0, 255)^2 >> 8` producing uint8 for int8 L1 weights
- **Quantization**: FT_SCALE=255, SCALE=64, UNITS=400. Final: `output * 400 / (255 * 64)`
- **Accumulator**: Stack-based, incremental updates. King moves trigger full recompute for that perspective only.

### Threat Features in NNUE (Unique)
This is BlackMarlin's most distinctive NNUE feature. The `threats()` function computes bitboards of pieces that can be captured by weaker pieces:
- Pawn attacks on any non-pawn piece
- Minor (knight/bishop) attacks on major pieces (rook/queen)
- Rook attacks on queen
- These threat squares are encoded as additional NNUE input features (using the 7th "piece type" slot)
- When a move changes the threat map, the difference (XOR) is incrementally updated in the accumulator
- This means the NNUE directly "sees" hanging and en-prise pieces, not just piece placement

### Compared to GoChess NNUE
- Ours: v5 (768x16->N)x2->1x8 (16 king buckets, 8 output buckets, CReLU/SCReLU, pairwise mul, Finny tables) **(UPDATE 2026-03-21)**
- Theirs: 28672->2xMID->1x8 (32 king buckets mirrored, threat features, SCReLU, output buckets)
- Key differences: (1) threat awareness baked into network inputs, (2) 32 king buckets from 64 squares. **(UPDATE 2026-03-21: GoChess now also has SCReLU, output buckets, pairwise mul, and Finny tables)**
- The threat feature input is **extremely unusual** -- only BlackMarlin does this among all 27 engines reviewed

### SIMD
- AVX2 for L1 feed-forward (VPMADDUBSW -- same as ours)
- NEON code exists but is commented out
- Accumulator updates use chunked processing (256 elements for AVX2, 128 otherwise)

---

## 2. Search Architecture

### Iterative Deepening
- Depth 1 upward, checks soft time limit between iterations
- Uses Rust generics for search type (Pv/Zw/NoNm) -- zero-cost abstractions, no runtime branching

### Aspiration Windows
- Initial window: 15 (`Window::new(15, 45, 100, 9)`)
- On fail-low: beta contracts toward alpha: `beta = (alpha + beta) / 2`, alpha widens by window
- On fail-high: beta widens by window
- Window expansion: `window += window * 45 / 100 + 9` (multiply by ~1.45 then add 9)
- Full-width fallback after 10 consecutive fails, or when eval > 1000
- Aspiration enabled at depth > 4

### Draw Detection
- 50-move rule: halfmove_clock >= 100
- **Two-fold repetition within search** (positions after root), three-fold for pre-root positions
- Insufficient material: K vs K, K+minor vs K

### Lazy SMP
- Shared TT via atomic u32 pairs (hash + analysis split into two AtomicU32)
- Board, history, killer moves, NNUE accumulator all per-thread
- Node counting via per-thread AtomicU64
- Time management decisions only on thread 0

---

## 3. Pruning Techniques

### Reverse Futility Pruning (RFP)
- Conditions: `!PV && !in_check && !skip_move`
- Depth guard: `depth <= 9`
- Margin: `71 * depth - 62 * improving_with_no_threats`
  - The `improving` flag is further conditioned on `nstm_threats.is_empty()` -- only counts as improving when opponent has no threats
  - Not improving: 71*d (depth 4: 284, depth 8: 568)
  - Improving + no threats: 9*d (depth 4: 36, depth 8: 72) -- drastically reduced margin
- **Returns `(eval + beta) / 2`** -- score blending at cutoff (not raw eval or beta)
- Compare to ours: we use 85*d (improving) / 60*d (not improving), no threat guard, no score blending

### Razoring
- Conditions: `!PV && !in_check && !skip_move`
- Depth guard: `depth <= 4`
- Margin: `306 * depth` (depth 1: 306, depth 2: 612, depth 3: 918, depth 4: 1224)
- If `eval + margin <= alpha`, do QSearch with window `alpha - 277` (zero-window)
- Returns QSearch result if it confirms fail-low
- Compare to ours: we use 400+100*d (much tighter). They have larger margins.

### Null Move Pruning (NMP)
- Conditions: `NM_allowed && depth > 4 && eval >= beta && not_all_pawns_and_king`
- **Threat guard**: `!(nstm_threat && depth <= 7)` -- skips NMP at low depth when opponent has threats
- **TT-adjusted eval for NMP**: If TT hit, uses bounded TT score (lower bound: max(tt, eval), exact: tt, upper: min(tt, eval)) instead of raw eval. Novel.
- Reduction: `R = 4 + depth * 23 / 60 + (eval - beta) / 204`
  - depth 8, eval=beta: R = 4+3+0 = 7, search at depth 1
  - depth 12, eval=beta+200: R = 4+4+0 = 8, search at depth 4
- **NMP Verification**: At depth >= 10, runs a verification search if NMP returns >= beta. This prevents NMP from being fooled by zugzwang at high depths.
- Compare to ours: we use R = 3 + d/3 + (eval-beta)/200, depth > 3, no verification, no TT-adjusted eval, no threat guard

### Internal Iterative Reduction (IIR)
- Condition: `depth >= 4 && (no_tt_entry || tt_entry.depth + 4 < depth)`
- Reduces depth by 1
- Note: triggers both when no TT entry at all AND when TT entry is significantly shallower (depth+4 < search depth). The second condition is unique.

### Futility Pruning (FP)
- Conditions: `!PV && non_mate_line && moves_seen > 0 && !is_capture && depth <= 9`
- Margin: `86 * lmr_depth` (uses LMR-reduced depth, not full depth)
  - lmr_depth 1: 86, lmr_depth 3: 258, lmr_depth 5: 430
- Triggers `skip_quiets()` -- prunes ALL remaining quiet moves when triggered
- Compare to ours: we use 80 + lmrDepth*80 per-move (continue, not skip-all)

### Late Move Pruning (LMP)
- Table-based thresholds from `lmp_lookup`:
  - Formula: `2.97 + depth^2`, divided by 1.94 when not improving
  - Not improving: depth 1: 2, depth 2: 3, depth 3: 6, depth 4: 9, depth 5: 14
  - Improving: depth 1: 3, depth 2: 6, depth 3: 11, depth 4: 18, depth 5: 27
- Triggers `skip_quiets()` -- skips all remaining quiets
- Compare to ours: we use 3+d^2 (+50% improving). Their formula is very similar.

### History Pruning (HP)
- Conditions: `!PV && non_mate_line && moves_seen > 0 && depth <= 6 && (!good_capture || eval <= alpha)`
- Threshold: `h_score < -(depth^2 * 138 / 10)`
  - depth 1: -13, depth 2: -55, depth 3: -124, depth 4: -220, depth 5: -345, depth 6: -496
- The `(!good_capture || eval <= alpha)` guard is novel -- allows HP on captures only when eval is bad
- Compare to ours: we use -2000*depth (much more aggressive)

### SEE-Assisted Futility Pruning
- Conditions: `!PV && non_mate_line && moves_seen > 0 && depth <= 6 && !good_capture`
- SEE margin: `alpha - eval - 123*depth + 1`
- If `margin > 0` (eval already above alpha) or SEE fails the margin, prune
- This combines futility and SEE into one check for captures

### QSearch Pruning
- **SEE >= 0 filter**: All captures with negative SEE are pruned
- **Good-delta SEE cutoff**: If `stand_pat + 1000 >= beta` and `SEE(move, beta - stand_pat + 192) >= 0`, return beta immediately. This is a novel QSearch shortcut -- if a capture's SEE is strong enough to push past beta, cut immediately without searching.
- **Bad-delta futility**: If `stand_pat + 192 <= alpha` and `SEE(move, 1) < 0`, skip. Prunes neutral captures when position is bad.
- **QSearch move limit**: Only searches up to **3 captures** (`move_cnt >= 2` triggers break after incrementing). This is extremely aggressive -- most engines search all good captures.
- **No evasion handling in QSearch**: When in check at QS entry, only generates captures (no quiet evasions). This could miss critical defensive moves.

---

## 4. Extensions

### Singular Extensions
- Depth guard: Two tiers:
  - `depth >= 7` ("multi_cut" path): Full singular search at `depth/2 - 1`
  - `depth < 7` ("low depth" path): Uses raw eval instead of searching, no multi-cut
- Conditions: `moves_seen == 0 && tt_move == make_move && ply != 0 && !tt_mate && tt_depth + 2 >= depth && (lower_bound || exact)`
- Additional gate for low-depth: `eval <= alpha` (only extends when losing)
- Singular beta: `tt_score - depth` (compare: ours is `tt_score - 3*depth`, theirs is much tighter)
- Singular search depth: `depth/2 - 1`

**Extension amounts**:
- Base: +1 if s_score < s_beta
- **Double extension** (non-PV, multi_cut): +2 if s_score < s_beta (always, when singular)
- **Triple extension** (non-PV, non-capture): +3 if s_score + 197 < s_beta (very singular quiet)
- **PV double extension**: +2 if s_score + 120 < s_beta (PV, multi_cut)
- **Low-depth double**: +2 if eval + 100 <= alpha (non-PV, low depth, struggling)

**Multi-cut**:
- If s_beta >= beta (singular search already beats beta), return s_beta immediately
- This is a powerful pruning shortcut

**Negative extensions**:
- -2 if tt_score >= beta (TT says position is already good enough)
- -2 if cut_node (expected to fail high, reduce)

**History bonus on singular move**: When a move is confirmed singular, it gets a history bonus of `depth` -- reinforcing the singular move for future ordering.

Compare to ours: Our SE is broken (-58 to -85 Elo). BlackMarlin's implementation is clean with: tighter margin (1*depth vs our 3*depth), multi-cut, negative extensions, low-depth eval-based fallback, and no extension limiter (relies on depth/2 search depth instead).

### Check Extensions
- +1 when gives_check, applied as `extension = extension.max(1)` (doesn't stack with SE)
- This means a move that gives check gets at least +1 extension even if SE didn't trigger

---

## 5. LMR (Late Move Reductions)

### Table Initialization
Single table (not split for captures/quiets):
- `0.50 + ln(depth) * ln(moveIdx) / 2.05`
- Base 0.50, divisor 2.05
- Compare to ours: we use separate tables (cap C=1.80, quiet C=1.50)

### LMR Application
- Only for `moves_seen > 0`
- History-based reduction: `reduction -= h_score / 112`
  - For quiets: h_score = (quiet_hist + counter_move_hist) / 2
  - For captures: h_score = capture_hist
  - History divisor 112 (compare to our 5000 -- very different scale due to MAX_HIST=512)

### Reduction Adjustments
- **Near-root bonus**: `-1` if `ply <= (depth + ply) * 2 / 5` (reduces less near root). This is novel -- approximately active when ply < 0.4 * depth.
- `+1` for non-PV node
- `+1` for not improving
- `-1` for killer move
- `+1` for cut_node
- **Threat-aware**: `-1` if move creates more threats than existed before (`new_stm_threat.len() > stm_threats.len()`)
- Clamped to `[0, depth-2]`

### What BlackMarlin Has vs GoChess
- Near-root reduction bonus (unique)
- Threat-aware LMR adjustment (using the same threat computation as NNUE)
- Killer move LMR bonus
- Cut-node +1 (we should have this from SUMMARY testing)

### What GoChess Has vs BlackMarlin
- Separate capture/quiet LMR tables (we have this, they use single table)
- TT noisy detection for LMR (we have this)
- Alpha-reduce (depth-1 after alpha raised)

---

## 6. Move Ordering

### Score Hierarchy (staged)
1. **TT move** (returned first, before any generation)
2. **Good captures** (SEE >= 0): scored by `capture_history + MVV * 32`
3. **Killer moves** (2 slots per ply, verified for legality)
4. **Quiet moves**: `quiet_hist + counter_move_hist + followup_hist_2ply + followup_hist_4ply`
   - Queen promotions get MAX score, underpromotions get MIN
5. **Bad captures** (SEE < 0): scored by `capture_history + MVV * 32`

### Staged Move Generation
Unlike Midnight, BlackMarlin uses proper staged generation:
- Phase 1: TT move only
- Phase 2: Generate all pseudo-legal moves, score captures
- Phase 3: Pick-best selection sort for good captures (SEE filter)
- Phase 4: Killer moves
- Phase 5: Score and generate quiet list
- Phase 6: Pick-best for quiets
- Phase 7: Pick-best for bad captures

The `skip_quiets()` method allows futility/LMP to skip directly to bad captures.

### History Tables

**Quiet history**: `[color][threatened][from][to]` -- **threat-indexed**
- Indexed by whether the from-square is attacked by opponent (`nstm_threats.has(make_move.from)`)
- This doubles the table size (2x for threatened flag)
- Effectively: moves escaping threats get different history than moves from safe squares

**Capture history**: `[color][threatened][from][to]` -- also threat-indexed
- Same threat-awareness as quiet history

**Counter-move history (continuation history)**: `[color][prev_piece][prev_to][curr_piece][curr_to]`
- 1-ply lookback

**Follow-up move history**: Same structure as counter-move, but 2-ply lookback
- Uses the **same table** as counter-move (shared weights). The `followup_move` and `counter_move` use different source indices but identical table structure.

**Follow-up move history 2**: 4-ply lookback
- Also uses the same `followup_move` table structure
- All three cont-hist lookbacks write and read

**History bonus/malus**:
- `hist_stat(amt) = min(amt * 13, 512)` -- caps at MAX_HIST=512
- Bonus: `change - change * hist / MAX_HIST` (gravity formula)
- Malus: `change + change * hist / MAX_HIST`
- History amount at cutoff: `depth + (eval <= initial_alpha)` -- **eval-based history depth bonus** (same as Alexandria idea #14f in SUMMARY)

**Correction History**:
- Pawn-only: `[color][pawn_hash_16bit]`
- Grain: 256, max correction: 32 (effective range: -32 to +32 centipawns)
- Update weight: `min(depth * 8, 128)`
- Update conditions: not mate score, not in check, score agrees with eval direction, best move is quiet

### No Counter-Move Table
BlackMarlin does NOT have a counter-move heuristic (specific move after opponent's move). It only has continuation history (piece-to indexed).

---

## 7. Time Management

### Base Allocation
- `base = min(inc + time / expected_moves, time * 4/5)`
- `expected_moves` starts at 64, decremented by 1 each move (decaying expected game length)
- Hard limit: `time * 4/5` (80% of remaining time)
- Target initially set to hard limit, then adjusted by deepen()

### Triple-Factor Soft Time Scaling
After each iteration (thread 0 only, depth > 4):

1. **Move stability factor**: `(41 - stability) * 0.024`
   - Stability: 0-14, incremented when best move unchanged, reset to 0 on change
   - Stability 0 (just changed): factor = 0.984
   - Stability 14 (very stable): factor = 0.648
   - Range: 0.648 to 0.984

2. **Node factor**: `(1.0 - move_nodes / total_nodes) * 3.42 + 0.52`
   - If best move uses 10% of nodes: factor = 3.60 (extend time significantly)
   - If best move uses 50% of nodes: factor = 2.23
   - If best move uses 90% of nodes: factor = 0.86 (save time)

3. **Eval factor**: `clamp(prev_eval - eval, 18, 20) * 0.088`
   - This is nearly constant! Range: 1.584 to 1.760
   - The clamp(18, 20) means eval changes barely affect time -- essentially a fixed multiplier of ~1.67
   - **This looks like a bug or vestigial code** -- the eval factor provides almost no dynamic scaling

Final: `target = base * stability * node * eval`

Compare to ours: We use instability factor (200) based on best move changes + scoreDelta. Their node-based scaling is more sophisticated but the eval factor is essentially dead code.

---

## 8. Transposition Table

### Structure
- Single entry per index (no buckets)
- Index: fixed-point multiplication trick: `(hash as u128 * len as u128) >> 64`
- Hash verification: lower 32 bits stored separately
- Atomic access: hash in AtomicU32, analysis split across two AtomicU32s (total 12 bytes/entry)
- No XOR verification between hash and data (unlike our lockless scheme)

### Entry Layout (8 bytes packed into u64)
- depth: u8
- entry_type: u16 (2-bit bounds + 14-bit eval)
- score: i16
- table_move: u16 (compressed: 6-bit from + 6-bit to + 4-bit promotion)
- age: u8

### Replacement Policy
Replace if `new_depth * 2 + age_difference + 1 >= old_depth + old_extra_depth`
- Extra depth: +1 for Exact or LowerBound entries
- Age difference: wrapping subtraction of ages
- This heavily favors newer entries (age bonus) and deeper searches

### TT Cutoff
- Non-PV only, requires `entry.depth >= search_depth`
- Standard bound-based cutoffs (exact returns immediately, lower bound >= beta, upper bound <= alpha)

### Prefetch
- SSE prefetch on x86_64 for TT lookup ahead of search

### TT in Eval
- Stores raw_eval in TT (not corrected eval)
- On TT hit, uses stored eval instead of recomputing NNUE

---

## 9. Aggression System (Unique)

BlackMarlin has a novel "aggression" adjustment to evaluation:
- `aggression = 2 * non_pawn_piece_count * clamp(root_eval, -200, 200) / 100`
- Positive when current position's STM matches root STM, negative otherwise
- Effect: if the engine is winning at root (+200cp), all evaluations get boosted by up to ~2*15*200/100 = 60cp in the middlegame (15 non-pawn pieces)
- This creates a snowball effect: winning positions are evaluated more optimistically
- The aggression is added to eval but NOT stored in TT (only raw_eval is stored)
- Also added to QSearch stand-pat

This is very unusual among engines. It's essentially a form of contempt that scales with material and position advantage.

---

## 10. Notable Differences from GoChess

### Things BlackMarlin Has That We Don't
1. **Threat features in NNUE** -- hanging piece information baked into network inputs (unique among all reviewed engines)
2. **Threat-indexed history tables** -- quiet and capture history doubled by whether from-square is under attack
3. **NMP verification at depth >= 10** -- prevents zugzwang-based NMP failures
4. **NMP TT-adjusted eval** -- uses bounded TT score for NMP eval (tighter than raw eval)
5. **NMP threat guard** -- disables NMP at depth <= 7 when opponent has hanging piece threats
6. **Near-root LMR reduction** -- reduces less near the root based on ply/depth ratio
7. **Threat-aware LMR** -- reduces less for moves that create new threats
8. **Aggression/contempt system** -- eval scaled by root advantage and material
9. **Low-depth singular extensions** -- uses raw eval at depth < 7 instead of searching
10. **RFP score blending** -- returns `(eval+beta)/2` instead of raw eval
11. **QSearch capture limit** -- only searches 3 captures in QSearch
12. **Eval-based history depth bonus** -- `depth + (eval <= alpha)` for history amount
13. **4-ply continuation history** -- reads and writes at ply -4 (not just -1 and -2)
14. **Output buckets in NNUE** -- material-based with non-linear formula

### Things We Have That BlackMarlin Doesn't
1. **Counter-move heuristic** (specific move table)
2. **TT score dampening** at cutoffs
3. **Alpha-reduce** (depth-1 after alpha raise)
4. **Fail-high score blending** in main search
5. **QS beta blending**
6. **Separate LMR tables** for captures vs quiets (they use one table)
7. **TT noisy move detection** for LMR
8. **Pawn correction history** with higher resolution (they use u16 hash, we presumably use larger)
9. **NMP threat escape bonus** in move ordering
10. **Recapture extensions**
11. **Multi-bucket TT** (4-slot buckets vs their single entry)

---

## 11. Parameter Comparison Table

| Feature | BlackMarlin | GoChess |
|---------|-------------|---------|
| RFP margin | 71*d - 62*improving_nothreats, d<=9 | 85*d (imp) / 60*d (not), d<=8 |
| RFP return | (eval+beta)/2 | eval |
| Razoring | 306*d, d<=4 | 400+100*d, d<=3 |
| NMP min depth | 5 | 3 |
| NMP reduction | 4+d*23/60+(eval-beta)/204 | 3+d/3+(eval-beta)/200 |
| NMP verification | depth >= 10 | No |
| NMP threat guard | nstm_threats && depth<=7 | No |
| IIR trigger | depth>=4, no TT or TT shallow | depth>=4, no TT move |
| Futility margin | 86*lmr_depth, d<=9 | 80+lmrDepth*80 |
| Futility action | skip_quiets | per-move continue |
| LMP formula | 2.97+d^2 (/1.94 not improving) | 3+d^2 (+50% improving) |
| LMR base | 0.50 | 1.50 (quiet), 1.80 (cap) |
| LMR divisor | 2.05 | 2.36 (quiet), 2.05 (cap) |
| History LMR div | 112 (MAX_HIST=512) | 5000 |
| History pruning | -d^2*13.8, d<=6 | -2000*d |
| SEE quiet margin | alpha-eval-123*d+1, d<=6 | -20*d^2 |
| SE depth gate | >=7 (full) / any (low-depth) | >=10 |
| SE beta | tt_score - depth | tt_score - 3*depth |
| SE double ext | +2/+3 (non-PV), +2 (PV) | None |
| SE multi-cut | return s_beta | None |
| SE neg ext | -2 (ttScore>=beta or cutNode) | None |
| Aspiration delta | 15 | 15 |
| ASP growth | *1.45 + 9 | *1.67 |
| ASP fail-low | beta = (a+b)/2 | No |
| Check extension | +1 on gives_check | No |
| QS capture limit | 3 | Unlimited |
| History depth bonus | depth + (eval<=alpha) | depth |
| Cont-hist plies | -1, -2, -4 | -1, -2 |
| Correction history | pawn only, u16 hash | pawn only |
| TT entry size | 12 bytes, single entry | 10 bytes, 4-slot buckets |
| SMP | Lazy SMP, atomic u32 pairs | Lazy SMP, XOR lockless |

---

## 12. Ideas Worth Testing from BlackMarlin

### High Priority

1. **NMP TT-adjusted eval** -- Use bounded TT score instead of raw static eval for NMP decision. If TT says lower bound is higher than eval, use that. Costs nothing, 3 lines.
   - Est. Elo: +2 to +5.

2. **RFP score blending** -- Return `(eval+beta)/2` instead of raw eval on RFP cutoff. We already know score dampening works (+22 Elo from TT dampening).
   - Est. Elo: +3 to +8. Note: we tested "RFP score dampening" before at -16.7 Elo. But that may have been a different formula. BlackMarlin's `(eval+beta)/2` is symmetric.

3. **NMP verification at high depth** -- At depth >= 10, verify NMP result with a real search. Prevents zugzwang-related errors. Classical technique that BlackMarlin and Altair both keep.
   - Est. Elo: +2 to +5.

4. **4-ply continuation history** -- We tested contHist4 and it failed (-58 Elo), but BlackMarlin uses it with full read+write weight. May need different integration (at full weight, not 1/4).
   - Status: Previously tested negative. Retry only with different weight/integration.

5. **Threat-indexed history** -- This is the same idea as "threat-aware history" (#5 in SUMMARY, 12 engines). BlackMarlin's implementation: butterfly history indexed by `[color][from_threatened][from][to]`. Simple boolean index, 2x table size. Aligns with our highest-consensus untested idea.
   - Est. Elo: +5 to +15.

6. **NMP threat guard** -- Disable NMP at low depth (<=7) when opponent has hanging piece threats. BlackMarlin computes threats anyway for NNUE; we'd need to compute them. Medium complexity.
   - Est. Elo: +2 to +5.

7. **Eval-based history depth bonus** -- `depth + (eval <= alpha)` for history update amount. Same as Alexandria idea #14f. Trivial.
   - Est. Elo: +1 to +3.

### Medium Priority

8. **Near-root LMR reduction** -- Reduce less when `ply <= (depth+ply)*2/5`. Novel, only in BlackMarlin. The idea: near the root, reductions are more costly because errors propagate to the PV.
   - Est. Elo: +1 to +4.

9. **Low-depth singular extensions** -- At depth < 7, use raw eval instead of searching for singularity. Much cheaper, allows SE to activate at lower depths. Inspired by Koivisto.
   - Est. Elo: +2 to +5. (Only relevant after we fix our SE implementation)

10. **QSearch capture limit** -- Limit QSearch to 3 captures. Extremely aggressive but BlackMarlin is strong. May gain NPS at cost of accuracy.
    - Est. Elo: -5 to +5 (risky, NPS gain may not translate to Elo).

11. **Threat-aware LMR** -- Reduce less for moves that create new threats. BlackMarlin checks `new_threats.len() > old_threats.len()`. Requires computing threats after each move.
    - Est. Elo: +2 to +5.

### Lower Priority

12. **Aggression system** -- Scale eval by root advantage. Novel but could cause instability. Low priority.

13. **IIR on shallow TT** -- Extend IIR trigger to also fire when `tt_depth + 4 < depth` (not just missing TT). BlackMarlin's refinement.
    - Est. Elo: +1 to +3.

14. ~~**Aspiration fail-low beta contraction**~~ -- `beta = (alpha+beta)/2` on fail-low. **(UPDATE 2026-03-21: GoChess now has aspiration contraction, merged.)**
    - Est. Elo: +1 to +3.

---

## 13. Cross-Reference with SUMMARY

Ideas from BlackMarlin that appear in SUMMARY (already tracked):
- **Threat-aware history** (#5 in Tier 1) -- BlackMarlin is engine #13 with this. Now 13 engines.
- **Deeper cont-hist** (#20 in Tier 2) -- BlackMarlin uses ply -4. Now 12 engines.
- **Cut-node LMR +1** (#25 in Tier 2) -- BlackMarlin has this. Now 4 engines (Weiss/Obsidian/BlackMarlin/GoChess-testing).
- **Node-based time management** (#24 in Tier 2) -- BlackMarlin's triple-factor scaling. Now 6 engines.
- **Double/triple SE + multi-cut** (#4 in Tier 1) -- BlackMarlin has all three. Now 11 engines.
- **Eval-based history depth bonus** (#14f in Tier 2) -- BlackMarlin matches Alexandria. Now 2 engines.
- ~~**Aspiration fail-low beta contraction**~~ -- **(UPDATE 2026-03-21: GoChess now has this, merged.)**

New ideas unique to BlackMarlin (not in SUMMARY):
- **Threat features in NNUE input** -- completely novel
- **NMP TT-adjusted eval** -- novel refinement
- **NMP threat guard at low depth** -- novel
- **RFP conditioned on threat-free improving** -- novel
- **Near-root LMR reduction** -- novel
- **Low-depth eval-based singular extensions** -- inspired by Koivisto but distinct
- **Aggression/contempt system** -- novel
- **QSearch 3-capture limit** -- novel
