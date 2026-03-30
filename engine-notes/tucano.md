# Tucano Chess Engine - Crib Notes

Source: `~/chess/engines/Tucano/`
Author: Alcides Schulz (Brazil)
Language: C (not Rust as initially assumed)
NNUE: HalfKP 40960->2x256->32->32->1 (Stockfish-derived, CReLU, no king buckets, no output buckets)
RR rank: #15 at +35 Elo (143 above GoChess-v5)

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. Multi-threaded via Lazy SMP (pthreads). Per-thread: GAME struct with board, search data, move ordering, NNUE accumulators, eval table.

### Iterative Deepening
- Depth 1 to max_depth (default MAX_DEPTH=120)
- Non-main threads skip depths: if > half of helper threads are already at current depth, increment depth by 1 (simple Lazy SMP depth diversification)
- Checks soft/extended time limits between iterations

### Aspiration Windows
- Enabled at depth > 4
- Initial window: prev_score +/- 25
- On fail: widen to +/- 100, then +/- 400 (window *= 4 each step)
- Falls back to full-width search after 3 fails or if mate score found
- **No fail-high depth reduction** (unlike Midnight/Alexandria)
- **No fail-low beta narrowing**
- Compare to ours: we use delta=15. Tucano's 25 is wider and their growth rate (4x) is aggressive.

### Draw Detection
- `is_draw()` called at every node (combines repetition, 50-move, insufficient material)

### Mate Distance Pruning
- Present at both main search and QSearch: `alpha = MAX(-MATE_SCORE + ply, alpha); beta = MIN(MATE_SCORE - ply, beta)`
- Compare to ours: we do NOT have MDP. This is a trivial 3-line addition we should test.

---

## 2. Pruning Techniques

### Reverse Futility Pruning (Static Null Move)
- Conditions: `!pv_node && !incheck && !singular_move_search && depth <= 6 && eval_score >= beta && !is_losing_score(beta)`
- **TT quiet-move guard**: only applies when `trans_move == MOVE_NONE || move_is_capture(trans_move)` -- i.e., RFP is skipped when a quiet TT move exists
- Margin: `100 * depth - improving * 80 - opponent_worsening * 30 + (ply > 0 ? eval_hist[ply-1] / 350 : 0)`
  - depth 1, not improving, not worsening: 100
  - depth 6, improving, worsening: 600 - 80 - 30 = 490
  - Parent eval also adjusts margin (eval_hist[ply-1] / 350, ~0-28 cp contribution)
- Returns `eval_score - margin` (not just beta)
- Compare to ours: we use 85*d (improving) / 60*d (not improving), depth<=8, no quiet-move guard, no parent eval adjustment

### Razoring
- Conditions: `!pv_node && !incheck && !singular_move_search && trans_move == MOVE_NONE`
- Depth guard: `depth < 6` (RAZOR_DEPTH=6)
- Margins by depth: [0, 250, 500, 750, 1000, 1250] (250*depth)
- Formula: if `eval_score + RAZOR_MARGIN[depth] < alpha`, drop to QSearch with `alpha - margin` window
- Only prunes if QSearch confirms score below margin (two-stage)
- Compare to ours: we use 400+100*d, depth<=3. Tucano goes deeper (to depth 5) with wider margins.

### Null Move Pruning (NMP)
- Conditions: `has_pieces(turn) && !has_recent_null_move && depth >= 2 && eval_score >= beta`
- Reduction: `5 + (depth - 4) / 4 + MIN(3, (eval_score - beta) / 200) + (last_move_is_quiet ? 0 : 1)`
  - Base R is 5, not 3 -- **much more aggressive** than ours
  - `(depth - 4) / 4` scaling (vs our `depth / 3`)
  - **-1 R after captures**: when last move was not quiet, adds 1 to null_depth reduction (less reduction = deeper null search)
- Clamps mate scores to beta
- **No NMP verification** (unlike some engines at deep depths)
- Compare to ours: R = 3 + d/3 + min((eval-beta)/200, 3). Tucano's R starts at 5, making NMP significantly more aggressive.

### ProbCut
- Conditions: `depth >= 5 && !is_mate_score(beta) && !pv_node && !incheck && !singular_move_search`
- Beta: `beta + 100`
- **TT guard**: only proceeds if no TT hit, or TT score >= pc_beta, or TT depth < depth-3
- **Quiet TT move guard**: only proceeds if `trans_move == MOVE_NONE || !move_is_quiet(trans_move)`
- Iterates captures only, skips quiet moves and captures where `eval_score + SEE(move) < pc_beta`
- **Two-stage QS pre-filter at depth > 10**: runs QSearch first; only does full negamax if QS passes
- Full search: `depth - 4 - pc_move_count/5` (extra reduction based on move count -- novel)
- Saves to TT on cutoff at `depth - 3`
- Compare to ours: we use margin 170, gate 85, depth>=5. Tucano has wider margin (100) and the QS pre-filter at depth>10 is a technique we should test. The move-count-based reduction in ProbCut is unique.

### Recapture Depth Reduction
- After a capture (`move_is_capture(last_move_made)`):
  - If `!improving && depth > 3 && depth <= 10 && eval_score + MAX(200, best_capture_value) < alpha`: `depth--`
  - `get_best_capture()` returns value of opponent's most valuable piece (+ promotion bonus if pawn on 7th)
- **Unique technique**: reduces depth when a recapture is unlikely to improve the position
- Compare to ours: we don't have this. Already in SUMMARY.md as idea #15.

### IIR (Internal Iterative Reduction)
- Conditions: `depth > 3 && trans_move == MOVE_NONE && !incheck`
- Reduces depth by 1
- Compare to ours: identical (we use depth > 3, no TT move)

### Move Count Pruning (LMP)
- Conditions: `!root_node && !pv_node && move_has_bad_history && depth <= 6 && !incheck && move_is_quiet(move) && !is_free_passer`
- Threshold: `4 + depth * 2 + (improving ? 0 : -3)`
  - depth 1, improving: 6; not improving: 3
  - depth 6, improving: 16; not improving: 13
- **Bad history gate**: only prunes moves with < 60% beta cutoff rate
- **Free passer exception**: never prunes pawn moves that create passed pawns
- Compare to ours: we use 3+d^2 (+50% improving). Tucano's thresholds are looser at depth 1-3 but tighter at depth 5-6.

### Futility Pruning
- Conditions: `!root_node && depth < 8 && (!pv_node || !incheck) && !is_mate_score(alpha) && move_is_quiet(move)`
- **Not a killer or counter-move** (inside nested if for non-killer, non-counter)
- Margin: `depth * (50 + get_pruning_margin(move_order, turn, move))`
  - `get_pruning_margin` returns `cutoff_count * 100 / search_count` for that piece+to combination (0-100)
  - If unsearched, returns 100 (avoids premature pruning)
  - Effective margin: `depth * 50` (bad history, 0%) to `depth * 150` (perfect history, 100%)
  - depth 1: 50-150; depth 7: 350-1050
- Formula: `eval_score + margin < alpha` => skip (continue, per-move)
- **Unique: history-driven margin** -- moves with good beta-cutoff history get wider margins (harder to prune)
- Compare to ours: we use 80+d*80 (fixed per depth). Tucano's per-move history-driven margin is a distinctive feature.

### SEE Pruning
- Conditions: `!root_node && depth <= 8 && !incheck` (non-killer, non-counter only)
- Quiet margin: `-60 * depth`
- Capture margin: `-10 * depth * depth`
- Compare to ours: our quiet margin is similar; our capture margin is -20*d^2 (Tucano's is -10*d^2, less aggressive)

### SEE Pruning for Captures at Depth 1
- Extra capture pruning: `!pv_node && !root_node && move_count > 5 && !extensions && !incheck && depth == 1 && !move_is_quiet(move) && !improving`
- Formula: `best_score + 200 + SEE(move) <= alpha` => skip
- Combines best-score-so-far, fixed buffer, and SEE value

### History Pruning
- Indirect via `move_has_bad_history` flag: cutoff_rate < 60% counts as "bad"
- Used as a gate for LMP (only bad-history moves get LMP'd)
- Compare to ours: we use -2000*depth threshold on butterfly history. Tucano uses cutoff ratio instead.

### QSearch Futility (Delta Pruning)
- Stand-pat + `MAX(100, 500 + depth * 10)` + captured piece value <= alpha => skip
- Note: `depth` is negative in QSearch (starts at 0, decrements)
- At qs depth 0: buffer = 500; at depth -5: buffer = 450; at depth -10: buffer = 400
- Also: if captured piece worth less than capturing piece, verify with full SEE
- Compare to ours: we use 240 delta buffer. Tucano's 500 is much wider but shrinks with QS depth.

---

## 3. Extensions

### Check Extension
- `gives_check && (depth < 4 || SEE(move) >= 0)` => +1
- **SEE-filtered at depth >= 4**: only extends checks that aren't losing material
- Compare to ours: we tested unconditional check extension (-11.2 Elo). Tucano's SEE-filtered variant is different -- worth retesting.

### Singular Extensions
- Depth guard: `depth >= 8`
- Conditions: `!root_node && !extensions && !singular_move_search && tt_hit && move == trans_move && tt_flag != TT_UPPER && tt_depth >= depth - 3 && !is_mate_score(tt_score)`
- Singularity beta: `trans_score - 4 * depth`
- Singularity depth: `depth / 2`
- **Double extension**: if `singular_score + 50 < reduced_beta` => extensions = 2
- **Single extension**: if `singular_score < reduced_beta` => extensions = 1
- **Multi-cut**: if `reduced_beta >= beta` => return reduced_beta (direct cutoff)
- **Negative extension**: if `trans_score <= alpha || trans_score >= beta` => reductions = 1
- Compare to ours: our SE is broken (-58 Elo). Tucano's margin is 4*depth (vs Alexandria's d*5/8 which is ~0.625*d). Tucano's double ext uses a 50cp additional threshold below singularity beta.

### No other extensions
- No recapture extension, no passed pawn extension

---

## 4. LMR (Late Move Reductions)

### Table Initialization
- Single table: `1.0 + log(d) * log(m) * 0.5`
- Compare to ours: we use separate tables (captures C=1.80, quiets C=1.50). Tucano uses a single table with coefficient 0.5 (equivalent to divisor 2.0).

### Application Conditions
- Only for non-killer, non-counter-move quiets (captures never get LMR)
- `move_count > 1 && !extensions`

### Reduction Adjustments (non-PV)
- `+1` for bad_history OR not_improving OR (in_check AND moving king)
- `+1` if TT move is noisy (`trans_move != MOVE_NONE && !move_is_quiet(trans_move)`)

### Reduction Adjustments (PV)
- `-1` if not bad_history
- `-1` if in_check
- `-1` if root_node

### What they DON'T have (that we do):
- No history-based continuous adjustment (we use history/5000)
- No threat-aware LMR
- No cut-node LMR
- No alpha-reduce interaction
- No failing heuristic interaction
- Captures are never reduced (no separate capture LMR table)

---

## 5. Move Ordering

### Staged Move Generation
Stages: TT -> Good Captures -> Quiet Moves -> Bad Captures (late moves)

1. **TT move**: returned first, validated via make/unmake for legality
2. **Good captures** (MVV-LVA style): `SORT_CAPTURE(100M) + VICTIM_VALUE[captured] + ATTACKER_VALUE[piece]`
   - Victim values: P=6, N=12, B=13, R=18, Q=24, K=0
   - Attacker values: P=4, N=3, B=3, R=2, Q=1, K=9
   - Bad captures (SEE < 0 where victim < attacker) deferred to late_moves
   - Under-promotions also deferred
3. **Quiet moves**: scored by `beta_cutoff_percent + SORT_CAPTURE(100M) if killer + SORT_KILLER(10M) if counter`
   - Killers scored at 100M (same as captures!), counters at 10M
   - Base score: cutoff_count * 100 / search_count (0-100 range)
   - **Sort stops early**: when score becomes 0, stops sorting remaining moves
4. **Bad captures**: returned in original order (no sorting)

### History Tables

**Cutoff History** (unique approach): `cutoff_history[2][6][64]` -- color, piece, to_square
- Tracks search_count and cutoff_count per piece+to combination
- On beta cutoff: increment both search_count and cutoff_count for best move; increment only search_count for all other searched quiet moves
- Overflow handling: when search_count reaches UINT16_MAX, both counts are right-shifted by 4 (divide by 16)
- Used for: move ordering (cutoff percentage), futility margin scaling, bad-history detection (<60%)
- Compare to ours: we use standard bonus/malus butterfly history. Tucano's ratio-based approach is fundamentally different.

**Killers**: 2 slots per ply per color
- Shift-down on update (standard)

**Counter-moves**: `counter_move[2][6][64][2]` -- prev_color, prev_piece, prev_to, 2 slots
- **Two counter-moves** per parent move (FIFO -- new one shifts old one to slot 1)
- Compare to ours: we have 1 counter-move per slot. Already in SUMMARY.md as idea #41.

### What they DON'T have:
- No continuation history (no [piece][to][piece][to] tables)
- No capture history
- No pawn history
- No correction history
- No threat-aware ordering

---

## 6. Unique Techniques

### Hindsight Depth Adjustment (Eval Progress)
Two adjustments based on parent reduction and eval trajectory:
1. **Bonus**: if `prior_reduction >= 3 && !opponent_worsening` => `depth++`
   - Parent was heavily reduced, and opponent's position didn't worsen -- the reduced search may have been insufficient
2. **Penalty**: if `prior_reduction >= 1 && depth >= 2 && eval_hist[ply] + eval_hist[ply-1] > 175` => `depth--`
   - Both sides think position is okay (sum of evals > 175) -- quiet position, reduce
- `opponent_worsening = eval_hist[ply] > -eval_hist[ply-1]` (our eval improved more than opponent's worsened)
- Compare to ours: we don't have this. Already in SUMMARY.md as idea #8 (5 engines). Tucano's implementation is among the clearest.

### Score-Drop Time Extension
- Tracks `var` counter across iterations (initial 0):
  - If `score + 20 < prev_score`: var -= 3
  - Else: var += 1
  - If `score + 40 < prev_score`: var = -3 (hard reset)
- When var < 0: `score_drop = TRUE`
- Effect: uses `extended_move_time` (4x normal) instead of `normal_move_time` when score is dropping
- Won't start new iteration when `used_time >= normal_move_time` UNLESS score_drop is true
- Also: won't start new iteration when `used_time >= extended_move_time * 0.6` regardless
- Compare to ours: **(UPDATE 2026-03-21: GoChess now has score-drop time extension with 2.0x/1.5x/1.2x tiered scaling, merged.)** Tucano's running counter approach gives up to 4x; ours uses fixed tiers.

### ProbCut Move-Count Reduction
- In ProbCut, the search depth is reduced further based on how many captures have been tried: `depth - 4 - pc_move_count/5`
- First 4 captures searched at full ProbCut depth; each subsequent 5 captures lose 1 ply
- **Novel**: not seen in other reviewed engines

### Cutoff History (Ratio-Based)
- Unlike standard engines that use additive bonus/malus history, Tucano tracks actual beta-cutoff ratios (cutoff_count / search_count)
- Used for: move ordering (sort by cutoff%), futility margin (50 + cutoff%), LMP gating (<60% = bad)
- Advantage: naturally bounded [0, 100], doesn't need gravity/clamping, directly interpretable
- Disadvantage: no depth weighting (shallow and deep searches contribute equally), coarser than continuous history

### Eval Cache Table
- Per-thread eval table: `eval_table[EVAL_TABLE_SIZE]` where EVAL_TABLE_SIZE = 64536
- Key + Score, indexed by `key % EVAL_TABLE_SIZE`
- Avoids redundant NNUE evaluations at same position
- Compare to ours: we evaluate fresh each time. This could save significant NNUE inference cost.

### Free Passer Exception in LMP
- Pawn moves that create passed pawns (no enemy pawns in front) are exempt from LMP
- Simple but effective -- prevents pruning of potentially game-changing pawn advances
- Compare to ours: we have no such exception

---

## 7. Time Management

### Base Allocation
**No movestogo (sudden death + increment):**
- `moves_to_go = 40 - played_moves / 2` (decreases as game progresses, min 1)
- `normal_move_time = total_time / moves_to_go`
- `extended_move_time = normal_move_time * 4`, capped at `total_time - 1000`
- Time buffer: 1000ms, reduced for very short TC

**Movestogo provided:**
- Uses provided moves_to_go directly
- Same formula: time / moves_to_go, with buffer

### Score-Drop Extension
- See "Unique Techniques" above -- var counter system gives up to 4x time on score drops

### Iteration-Based Stopping
- Stop if `used_time >= normal_move_time` (unless score_drop)
- Stop if `used_time >= extended_move_time * 0.6`

### What they DON'T have:
- No node-based time scaling (no best-move node fraction tracking)
- No best-move stability tracking
- No eval stability scaling
- Compare to ours: similar simplicity, but we have a basic instability factor

---

## 8. NNUE Architecture

### Network: HalfKP 40960 -> 2x256 -> 32 -> 32 -> 1
- Input: HalfKP (10 piece types * 64 squares * 64 king squares = 40960 per perspective)
- Feature transformer: 40960 -> 256 (per perspective, 512 total)
- Hidden layer 1: 512 -> 32 (CReLU, sparse multiplication with mask)
- Hidden layer 2: 32 -> 32 (CReLU)
- Output: 32 -> 1
- Scale: FV_SCALE = 16, SHIFT = 6
- No king buckets, no output buckets
- Net file: `tucano_nn03.bin` (external, not embedded)

### Compared to Our NNUE
- Theirs: HalfKP -> 2x256 -> 32 -> 32 -> 1
- Ours: v5 (768x16->N)x2->1x8 (8 output buckets, shallow wide, CReLU/SCReLU, Finny tables) **(UPDATE 2026-03-21: GoChess v5 is now a completely different, more modern architecture)**
- They lack output buckets (material-based) -- our 8-bucket output is an advantage
- They lack king buckets (no horizontal mirroring) -- we have 16 king buckets
- CReLU activation only (we now also support SCReLU)

### Accumulator Updates
- Stack-based: `nnue_data[MAX_HIST]` stores accumulator + changes at each ply
- Incremental updates for non-king moves
- King moves trigger full recompute (walks back in history to find last computed accumulator)
- `nnue_can_update()` searches backward through history for a computed accumulator without king moves
- `nnue_update_tree()` recursively updates all accumulators in the chain
- SIMD: AVX2, SSE4.1, SSE3, SSE2, MMX, NEON (compile-time selection)
- Hidden layer 1 uses **sparse multiplication with NNZ mask** (only multiplies non-zero clipped inputs)

---

## 9. Transposition Table

### Structure
- Single-entry per index (no buckets)
- Entry: 16 bytes total (8-byte key + 8-byte record packed as union)
- Record union: move (4 bytes), score (2 bytes), depth (1 byte), age:6 + flag:2 (1 byte)
- Index: `key & mask` (power-of-2 masking, not modulo)
- **Not lockless**: no atomic operations, potential race conditions with Lazy SMP

### Replacement Policy
Replace if ANY of:
- Same hash (update in place)
- New depth >= existing depth
- Existing entry is from an older age
- On same-hash replacement with MOVE_NONE, preserves existing move

### Probe
- Exact match on full 64-bit key (no XOR verification)
- Non-PV: depth >= required, standard alpha/beta/exact cutoff
- QSearch: uses TT too (with quiesce_depth = 0 or -1)

### Compared to ours
- We have 4-slot buckets (much better utilization)
- We use lockless XOR-verified packed atomics (thread-safe)
- Their single-entry is a significant weakness for Lazy SMP

---

## 10. Lazy SMP

- Main thread + N helper threads (configurable via UCI `Threads` option)
- Each thread gets full copy of: board, search data, move_order
- Threads share only the TT (no atomic protection -- race-prone)
- Depth diversification: helper threads skip depths where > half of helpers are already searching
- Main thread controls time; helpers check abort flag
- Helper threads don't post UCI info
- Compare to ours: similar approach but we have proper lockless TT

---

## 11. Notable Differences from GoChess

### Things Tucano has that we don't:
1. **Mate distance pruning** (3 lines, trivial, in both main and QSearch)
2. **Eval cache table** (64K entries, avoids redundant NNUE calls)
3. **RFP quiet-move guard** (skip RFP when quiet TT move exists) -- already in SUMMARY.md #2
4. **ProbCut QS pre-filter at depth > 10** (two-stage) -- already in SUMMARY.md #6
5. **ProbCut move-count reduction** (novel, pc_move_count/5)
6. **Recapture depth reduction** (not improving + eval + best_capture < alpha) -- already in SUMMARY.md #15
7. **Hindsight depth adjustment** (parent reduction + eval trajectory) -- already in SUMMARY.md #8
8. **Score-drop time extension** (var counter, up to 4x) -- **(UPDATE 2026-03-21: GoChess now has score-drop time extension 2.0x/1.5x/1.2x, merged)**
9. **NMP -1 R after captures** (less reduction when last move was capture) -- **(UPDATE 2026-03-21: GoChess now has NMP R-1 after captures, merged)**
10. **Free passer exception** in LMP
11. **SEE-filtered check extension** (SEE >= 0 at depth >= 4)
12. **Two counter-moves** per parent move -- already in SUMMARY.md #41
13. **Cutoff-ratio-driven futility margin** (per-move, history-scaled)
14. **Double singular extension** (+2 when score is 50cp below singularity beta)

### Things we have that Tucano doesn't:
1. **Continuation history** (they have none -- major gap)
2. **Capture history**
3. **Correction history**
4. **History-based LMR adjustments**
5. **Separate LMR tables for captures vs quiets**
6. **Output buckets in NNUE** (8 material-based)
7. **King buckets in NNUE** (16)
8. **4-slot TT buckets** (they have single-entry)
9. **Lockless TT** (they have no thread safety)
10. **Alpha-reduce** after alpha raise
11. **Fail-high score blending**
12. **TT score dampening**
13. **TT near-miss cutoffs**
14. **TT noisy move detection** for LMR
15. **QS beta blending**
16. **NMP threat detection**
17. **Failing heuristic**

### Parameter Comparison Table

| Feature | Tucano | GoChess |
|---------|--------|---------|
| RFP margin | 100*d - 80*imp - 30*opp_worse | 85*d (imp) / 60*d (not) |
| RFP max depth | 6 | 8 |
| RFP quiet guard | Yes (skip if quiet TT move) | No |
| Razoring | 250*d, depth<6 | 400+100*d, depth<=3 |
| NMP base R | 5 | 3 |
| NMP depth scaling | (d-4)/4 | d/3 |
| NMP eval scaling | min((eval-beta)/200, 3) | min((eval-beta)/200, 3) |
| NMP post-capture | R-1 when last move capture | Yes (UPDATE: merged) |
| LMR formula | 1.0 + log(d)*log(m)*0.5 | Separate cap/quiet tables |
| LMR captures | Never reduced | Reduced (C=1.80) |
| LMP depth | <=6 | 3+d^2 |
| LMP threshold | 4+2*d (+/- 3 improving) | 3+d^2 (+50% improving) |
| Futility margin | d*(50+cutoff%) depth<8 | 80+d*80 |
| SEE quiet margin | -60*d, depth<=8 | similar |
| SEE capture margin | -10*d^2, depth<=8 | -20*d^2 |
| Singular depth | >=8 | >=10 |
| Singular beta | tt_val - 4*depth | tt_val - 3*depth |
| Double sing ext | +2 at -50cp below beta | None |
| Multi-cut | return reduced_beta | None |
| ProbCut margin | +100, depth>=5 | +170, depth>=5 |
| ProbCut QS gate | depth > 10 | No |
| Check extension | SEE-filtered at depth>=4 | None |
| Aspiration delta | 25, growth 4x | 15 |
| TT structure | Single-entry, non-atomic | 4-slot, lockless |
| NNUE | HalfKP 40960->2x256->32->32->1 | v5: (768x16->N)x2->1x8 (UPDATE) |
| Counter-moves | 2 per slot | 1 per slot |
| Cont-hist | None | Plies 1-2 |

---

## 12. Ideas Worth Testing from Tucano

### Already in SUMMARY.md (priority reinforced):
1. **ProbCut QS pre-filter** (SUMMARY #6) -- Tucano confirms two-stage approach at depth>10
2. **Recapture depth reduction** (SUMMARY #15) -- `!improving && eval + bestCapture < alpha`
3. ~~**Score-drop time extension**~~ (SUMMARY #16) -- **(UPDATE 2026-03-21: MERGED, 2.0x/1.5x/1.2x)**
4. ~~**NMP -1 R after captures**~~ (SUMMARY #17) -- **(UPDATE 2026-03-21: MERGED)**
5. **Hindsight depth adjustment** (SUMMARY #8) -- `prior_reduction >= 3 && !opponent_worsening => depth++`
6. **Two counter-moves** (SUMMARY #41)

### New ideas from Tucano:
1. **Mate distance pruning** -- 3 lines at top of search+QSearch. Universal, trivial, 0-risk. Should have been added ages ago.
2. **Eval cache table** -- 64K-entry hash table avoiding redundant NNUE inference. Given NNUE is 35% of our CPU time, even a modest hit rate could be +5-10% NPS.
3. **ProbCut move-count reduction** -- reduce ProbCut search depth by pc_move_count/5. Unique to Tucano, cheap.
4. **SEE-filtered check extension** -- only extend checks with SEE >= 0 at depth >= 4. Our unconditional check ext was -11 Elo; this variant filters losing checks.
5. **Free passer LMP exception** -- exempt passed pawn advances from LMP. Very cheap.
6. **NMP base R=5** -- Tucano starts NMP reduction at 5 instead of 3. Their engine is significantly stronger than ours, and aggressive NMP could be a factor. Worth testing R=4 as a step.
7. **Cutoff-ratio futility** -- per-move futility margin driven by beta-cutoff percentage. Novel approach, interacts with our existing 80+d*80 futility.
8. **Wider razoring** -- depth < 6 with 250*d margin (vs our depth <= 3 with 400+100*d). Tests deeper razoring.
