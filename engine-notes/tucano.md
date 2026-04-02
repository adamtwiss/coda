# Tucano Chess Engine - Crib Notes

Source: `~/chess/engines/Tucano/`
Author: Alcides Schulz (Brazil)
Language: C
NNUE: HalfKP 40960->2x256->32->32->1 (Stockfish-derived, CReLU, no king buckets, no output buckets)
GoChess-era RR rank: #15 at +35 Elo (143 above GoChess-v5). Coda gauntlet opponent (Texel tier).

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. Multi-threaded via Lazy SMP (pthreads). Per-thread: GAME struct with board, search data, move ordering, NNUE accumulators, eval table.

### Iterative Deepening
- Depth 1 to max_depth (default MAX_DEPTH=120)
- Non-main threads skip depths: if > half of helper threads are already at current depth, increment depth by 1 (simple Lazy SMP depth diversification)

### Aspiration Windows
- Enabled at depth > 4
- Initial window: prev_score +/- 25
- On fail: widen to +/- 100, then +/- 400 (window *= 4 each step)
- Falls back to full-width search after 3 fails or if mate score found
- Compare to Coda: we use delta=15 from depth 4. Tucano's 25 is wider and their growth rate (4x) is aggressive.

### Mate Distance Pruning
- Present at both main search and QSearch: `alpha = MAX(-MATE_SCORE + ply, alpha); beta = MIN(MATE_SCORE - ply, beta)`
- Verified in source (search.c lines 56-57).
- Compare to Coda: we do NOT have MDP. Trivial 3-line addition.

---

## 2. Pruning Techniques

### Reverse Futility Pruning (Static Null Move)
- Conditions: `!pv_node && !incheck && !singular_move_search && depth <= 6 && eval_score >= beta && !is_losing_score(beta)`
- **TT quiet-move guard**: only applies when `trans_move == MOVE_NONE || move_is_capture(trans_move)` -- i.e., RFP is skipped when a quiet TT move exists
- Margin: `100 * depth - improving * 80 - opponent_worsening * 30 + (ply > 0 ? eval_hist[ply-1] / 350 : 0)`
- Returns `eval_score - margin` (not just beta)
- Compare to Coda: we use 70*d (improving) / 100*d (not improving), depth<=7, with TT quiet-move guard (**IMPLEMENTED from Tucano**). No parent eval adjustment.

### Razoring
- Conditions: `!pv_node && !incheck && !singular_move_search && trans_move == MOVE_NONE`
- Depth guard: `depth < 6` (RAZOR_DEPTH=6)
- Margins by depth: [0, 250, 500, 750, 1000, 1250] (250*depth)
- Two-stage: if eval + margin < alpha, drop to QSearch with `alpha - margin` window
- Compare to Coda: we don't currently have razoring in the code.

### Null Move Pruning (NMP)
- Conditions: `has_pieces(turn) && !has_recent_null_move && depth >= 2 && eval_score >= beta`
- Reduction: `5 + (depth - 4) / 4 + MIN(3, (eval_score - beta) / 200) + (last_move_is_quiet ? 0 : 1)`
  - Base R is 5, not 4 -- more aggressive than Coda
  - `(depth - 4) / 4` scaling (vs Coda's `depth / 3`)
  - **-1 R after captures**: when last move was not quiet, adds 1 to reduction -- **IMPLEMENTED in Coda** (R-1 after captures)
- Clamps mate scores to beta
- **No NMP verification** (unlike some engines at deep depths)
- Compare to Coda: R = 4 + d/3 + min((eval-beta)/200, 3) with R-1 after captures and NMP score dampening.

### ProbCut
- Conditions: `depth >= 5 && !is_mate_score(beta) && !pv_node && !incheck && !singular_move_search`
- Beta: `beta + 100`
- **TT guard**: only proceeds if no TT hit, or TT score >= pc_beta, or TT depth < depth-3
- **Quiet TT move guard**: only proceeds if `trans_move == MOVE_NONE || !move_is_quiet(trans_move)`
- **Two-stage QS pre-filter at depth > 10**: runs QSearch first; only does full negamax if QS passes
- Full search: `depth - 4 - pc_move_count/5` (extra reduction based on move count -- novel)
- Compare to Coda: we use margin 170, gate 85, depth>=5. ProbCut is currently disabled. Tucano's QS pre-filter and move-count reduction are unique.

### IIR (Internal Iterative Reduction)
- Conditions: `depth > 3 && trans_move == MOVE_NONE && !incheck`
- Compare to Coda: we use depth >= 4, PV or cut nodes only.

### LMP (Late Move Pruning)
- Conditions: `!root_node && !pv_node && move_has_bad_history && depth <= 6 && !incheck && move_is_quiet(move) && !is_free_passer`
- Threshold: `4 + depth * 2 + (improving ? 0 : -3)`
- **Bad history gate**: only prunes moves with < 60% beta cutoff rate
- **Free passer exception**: never prunes pawn moves that create passed pawns
- Compare to Coda: we use 3+d^2, depth<=8, with improving/failing adjustments. No free passer exception.

### Futility Pruning
- **History-driven margin**: `depth * (50 + get_pruning_margin(move_order, turn, move))`
  - Margin per move: 50 (bad history) to 150 (perfect history) per ply
- Not applied to killers or counter-moves
- Compare to Coda: we use 90+lmrDepth*100 with history/128 adjustment.

### SEE Pruning
- Quiet margin: `-60 * depth` (depth<=8)
- Capture margin: `-10 * depth * depth` (depth<=8)
- Compare to Coda: quiet -20*d^2 depth<=8, capture -d*100 depth<=6.

### Recapture Depth Reduction
- After capture: if `!improving && depth > 3 && depth <= 10 && eval + MAX(200, best_capture_value) < alpha`: `depth--`
- **Unique technique**: reduces depth when a recapture is unlikely to improve position
- Compare to Coda: we don't have this.

### History Pruning
- Indirect via `move_has_bad_history` flag: cutoff_rate < 60% counts as "bad"
- Used as gate for LMP
- Compare to Coda: we use -1500*depth threshold on combined history.

### QSearch Futility (Delta Pruning)
- Stand-pat + `MAX(100, 500 + depth * 10)` + captured piece value <= alpha => skip
- Compare to Coda: we use 240 delta buffer. Tucano's 500 is much wider but shrinks with QS depth.

---

## 3. Extensions

### Check Extension
- `gives_check && (depth < 4 || SEE(move) >= 0)` => +1
- **SEE-filtered at depth >= 4**: only extends checks that aren't losing material
- Compare to Coda: we don't have check extensions.

### Singular Extensions
- Depth guard: `depth >= 8`
- Singularity beta: `trans_score - 4 * depth`
- Singularity depth: `depth / 2`
- **Double extension**: if `singular_score + 50 < reduced_beta` => extensions = 2
- **Single extension**: if `singular_score < reduced_beta` => extensions = 1
- **Multi-cut**: if `reduced_beta >= beta` => return reduced_beta
- **Negative extension**: if `trans_score <= alpha || trans_score >= beta` => reductions = 1
- Compare to Coda: we have singular extensions with multi-cut and negative extension (-1). No double extension. Coda's margin is `tt_score - depth` (vs Tucano's `4*depth`).

---

## 4. LMR (Late Move Reductions)

### Table Initialization
- Single table: `1.0 + log(d) * log(m) * 0.5`
- Compare to Coda: we use separate tables (captures C=1.80, quiets C=1.30).

### Application Conditions
- Only for non-killer, non-counter-move quiets (captures never get LMR)
- `move_count > 1 && !extensions`

### Reduction Adjustments (non-PV)
- `+1` for bad_history OR not_improving OR (in_check AND moving king)
- `+1` if TT move is noisy

### Reduction Adjustments (PV)
- `-1` if not bad_history
- `-1` if in_check
- `-1` if root_node

### What Tucano doesn't have (that Coda does):
- No history-based continuous adjustment
- No capture LMR (captures never reduced)
- No doDeeper/doShallower

---

## 5. Move Ordering

### Staged Move Generation
Stages: TT -> Good Captures -> Quiet Moves -> Bad Captures

1. **TT move**: returned first, validated via make/unmake for legality
2. **Good captures** (MVV-LVA style): Victim values P=6, N=12, B=13, R=18, Q=24. Bad captures (SEE < 0) deferred.
3. **Quiet moves**: scored by cutoff percentage + killer/counter bonuses
4. **Bad captures**: returned unsorted

### History Tables

**Cutoff History** (unique approach): `cutoff_history[2][6][64]` -- color, piece, to_square
- Tracks search_count and cutoff_count per piece+to combination
- Ratio-based: naturally bounded [0, 100], no gravity needed
- Compare to Coda: we use standard bonus/malus butterfly history + continuation history + pawn history + capture history. Tucano's ratio-based approach is fundamentally different.

**Killers**: 2 slots per ply per color

**Counter-moves**: `counter_move[2][6][64][2]` -- 2 counter-moves per parent move (FIFO)
- Compare to Coda: we have 1 counter-move per slot.

### What Tucano doesn't have:
- No continuation history
- No capture history
- No pawn history
- No correction history

---

## 6. Unique Techniques

### Hindsight Depth Adjustment (Eval Progress)
- **Bonus**: if `prior_reduction >= 3 && !opponent_worsening` => `depth++`
- **Penalty**: if `prior_reduction >= 1 && depth >= 2 && eval_hist[ply] + eval_hist[ply-1] > 175` => `depth--`
- Compare to Coda: **IMPLEMENTED** -- Coda has hindsight depth adjustment (confirmed +18 Elo in ablation).

### Score-Drop Time Extension
- Running `var` counter: decremented on score drops, incremented on stable scores
- Effect: uses extended_move_time (4x normal) when score is dropping
- Compare to Coda: we don't have score-drop time extension.

### ProbCut Move-Count Reduction
- In ProbCut, search depth reduced by `pc_move_count/5` -- first 4 captures at full depth, each 5 more lose 1 ply
- **Novel**: not seen in other reviewed engines

### Cutoff History (Ratio-Based)
- Unlike standard engines that use additive bonus/malus, Tucano tracks actual beta-cutoff ratios
- Used for: move ordering, futility margin scaling, LMP gating
- Advantage: naturally bounded, no gravity needed

### Eval Cache Table
- Per-thread: `eval_table[64536]` (key + score)
- Avoids redundant NNUE evaluations at same position
- Compare to Coda: we evaluate fresh each time. Could save NNUE inference cost.

### Free Passer Exception in LMP
- Pawn moves that create passed pawns exempt from LMP
- Compare to Coda: we don't have this.

---

## 7. Time Management

### Base Allocation
- `moves_to_go = 40 - played_moves / 2` (sudden death)
- `normal_move_time = total_time / moves_to_go`
- `extended_move_time = normal_move_time * 4`, capped at `total_time - 1000`

### Score-Drop Extension
- var counter system gives up to 4x time on score drops
- Compare to Coda: we don't have score-drop time extension.

---

## 8. NNUE Architecture

### Network: HalfKP 40960 -> 2x256 -> 32 -> 32 -> 1
- Input: HalfKP (10 piece types * 64 squares * 64 king squares = 40960 per perspective)
- Hidden layers: 512 -> 32 (CReLU, sparse multiplication with mask) -> 32 -> 1
- No king buckets, no output buckets
- Net file: `tucano_nn03.bin`

### Compared to Coda NNUE
- Coda: v5 (768x16->N)x2->1x8 with 16 king buckets, 8 output buckets, CReLU/SCReLU/pairwise, Finny tables; v7 adds hidden layers
- Tucano lacks output buckets and king buckets -- Coda's architecture is more modern
- Tucano uses sparse multiplication with NNZ mask for hidden layer 1

---

## 9. Transposition Table

- Single-entry per index (no buckets), 16 bytes
- **Not lockless**: potential race conditions with Lazy SMP
- Replace if: same hash, new depth >= existing, or older age
- Compare to Coda: we have 5-slot buckets with lockless XOR-verified atomics. Much better.

---

## 10. Summary: Differences from Coda

### Things Tucano has that Coda doesn't:
1. **Mate distance pruning** (3 lines, trivial)
2. **Eval cache table** (64K entries, avoids redundant NNUE calls)
3. **ProbCut QS pre-filter at depth > 10** (two-stage)
4. **ProbCut move-count reduction** (pc_move_count/5)
5. **Recapture depth reduction** (not improving + eval + best_capture < alpha)
6. **Free passer LMP exception**
7. **SEE-filtered check extension** (SEE >= 0 at depth >= 4)
8. **Two counter-moves** per parent move
9. **Cutoff-ratio-driven futility margin** (per-move, history-scaled)
10. **Double singular extension** (+2 when 50cp below singularity beta)
11. **Score-drop time extension** (up to 4x)

### Things Coda has that Tucano doesn't:
1. **Continuation history**
2. **Capture history**
3. **Pawn history**
4. **Multi-source correction history** (pawn + NP + continuation)
5. **History-based LMR adjustments**
6. **Separate LMR tables for captures vs quiets** (captures reduced in Coda)
7. **Output buckets in NNUE** (8 material-based)
8. **King buckets in NNUE** (16)
9. **5-slot TT buckets** with lockless atomics
10. **Fail-high score blending** ((score*d+beta)/(d+1))
11. **TT score dampening** ((3*score+beta)/4)
12. **TT near-miss cutoffs**
13. **NMP score dampening** ((score*2+beta)/3)
14. **Bad noisy pruning**
15. **Hindsight depth adjustment** (IMPLEMENTED)
16. **doDeeper/doShallower** in LMR

### Parameter Comparison Table

| Feature | Tucano | Coda |
|---------|--------|------|
| RFP margin | 100*d - 80*imp - 30*opp_worse | 70*d (imp) / 100*d (not) |
| RFP max depth | 6 | 7 |
| RFP quiet guard | Yes | Yes (IMPLEMENTED) |
| NMP base R | 5 | 4 |
| NMP depth scaling | (d-4)/4 | d/3 |
| NMP eval scaling | min((eval-beta)/200, 3) | min((eval-beta)/200, 3) |
| NMP post-capture | R-1 | R-1 (IMPLEMENTED) |
| LMR formula | 1.0 + log(d)*log(m)*0.5 | Separate cap/quiet tables |
| LMR captures | Never reduced | Reduced (C=1.80) |
| LMP depth | <=6 | <=8, 3+d^2 |
| Futility margin | d*(50+cutoff%) depth<8 | 90+lmrDepth*100 |
| SEE quiet margin | -60*d, depth<=8 | -20*d^2, depth<=8 |
| SEE capture margin | -10*d^2, depth<=8 | -d*100, depth<=6 |
| Singular depth | >=8 | >=8 |
| Singular beta | tt_val - 4*depth | tt_val - depth |
| ProbCut margin | +100, depth>=5 | +170, depth>=5 |
| ProbCut QS gate | depth > 10 | No |
| Check extension | SEE-filtered at depth>=4 | None |
| Aspiration delta | 25, growth 4x | 15 |
| TT structure | Single-entry, non-atomic | 5-slot, lockless |
| Counter-moves | 2 per slot | 1 per slot |
| Cont-hist | None | 1-ply |
| Delta pruning | 500 (shrinks with QS depth) | 240 |

---

## 11. Ideas Worth Testing from Tucano

### High Priority
1. **Mate distance pruning** -- 3 lines at top of search+QSearch. Universal, trivial, zero risk.
2. **Eval cache table** -- 64K-entry hash table avoiding redundant NNUE inference. NNUE is a significant fraction of CPU time.
3. **ProbCut QS pre-filter** -- Two-stage ProbCut at depth>10. Also seen in Minic.
4. **Score-drop time extension** -- up to 4x on score drops. Multiple engines have this.

### Medium Priority
5. **SEE-filtered check extension** -- Only extend checks with SEE >= 0 at depth >= 4.
6. **Recapture depth reduction** -- `!improving && eval + bestCapture < alpha => depth--`.
7. **Double singular extension** -- +2 when score is 50cp below singularity beta.
8. **ProbCut move-count reduction** -- reduce ProbCut depth by pc_move_count/5.
9. **Two counter-moves** per parent move.

### Lower Priority
10. **Free passer LMP exception** -- exempt passed pawn advances from LMP.
11. **NMP R=5** -- Tucano starts NMP reduction at 5 instead of 4. More aggressive.
12. **Wider razoring** -- depth < 6 with 250*d margin. Coda currently has no razoring.
