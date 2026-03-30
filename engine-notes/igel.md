# Igel Chess Engine - Crib Notes

Source: `~/chess/engines/igel/`
Author: Volodymyr Shcherbyna (derived from GreKo 2018.01)
NNUE: HalfKP-like with 32 king buckets, 22528->2x1024->16->32->1 x8 output buckets, pairwise mul + dual activation (SCReLU + CReLU)
RR rank: #14 at +87 Elo (195 above GoChess-v5)

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. Lazy SMP with vote-based best-move selection across threads. Derived from GreKo but heavily modernized.

### Iterative Deepening
- Depth 1 to MAX_PLY (128), or limited by level setting
- Checks soft time limit after each completed iteration
- Score-drop time adjustment (see Time Management)

### Aspiration Windows
- Enabled at depth >= 4 (`aspiration = depth >= 4 ? 5 : CHECKMATE_SCORE`)
- **Very tight initial delta: 5** (vs our 15, Midnight 12, Alexandria 12)
- On fail-low: `beta = (alpha + beta) / 2`, widen alpha by aspiration
- On fail-high: widen beta by aspiration
- Delta growth: `aspiration += 2 + aspiration / 2` (multiply by ~1.5 each iteration)
- No fail-high depth reduction (unlike Midnight, Alexandria)
- Compare to ours: our delta=15. Their delta=5 means more re-searches but tighter bounds. Already listed in SUMMARY as item #40.

### Draw Detection
- Two-fold repetition (`Repetitions() >= 2`) at all nodes
- Fifty-move rule: `Fifty() >= 100`
- Insufficient material: no pawns and MatIndex < 5 for both sides

### Mate Distance Pruning
- Standard MDP: tighten alpha/beta by ply distance from root
- `rAlpha = max(alpha, -CHECKMATE + ply)`, `rBeta = min(beta, CHECKMATE - ply - 1)`
- Return rAlpha if rAlpha >= rBeta
- Compare to ours: we don't have MDP. Already in SUMMARY item #2 as "mate distance pruning" priority item.

---

## 2. Pruning Techniques

### Reverse Futility Pruning (Static Null Move Pruning)
- Conditions: `!inCheck && !onPV`
- Depth guard: `depth <= 8`
- Margin: `85 * (depth - improving)`
  - When improving: 85*(depth-1) = depth 1: 0, depth 2: 85, depth 3: 170, depth 4: 255
  - Not improving: 85*depth = depth 1: 85, depth 2: 170, depth 3: 255, depth 4: 340
- Formula: `bestScore - 85*(depth - improving) >= beta` => return bestScore
- Uses TT-corrected `bestScore` (not raw staticEval)
- Compare to ours: we use 85*d (improving) / 60*d (not improving), depth<=8. Their margin is slightly different: `85*(d-1)` improving vs our `85*d` improving; `85*d` not-improving vs our `60*d` not-improving. Their non-improving is significantly wider (85 vs 60), meaning they prune less aggressively when not improving.

### Razoring
- Conditions: `!inCheck && !onPV`
- Depth guard: `depth <= 2`
- Margin: 150 (fixed, not depth-dependent beyond the guard)
- Formula: `staticEval + 150 < alpha` => drop to qsearch
- Compare to ours: we use 400+100*d (depth 1: 500, depth 2: 600, depth 3: 700), depth<=3. Their margin is much tighter (150 at all depths vs our 500-700), and shallower (depth<=2 vs our depth<=3). This means they razor much more aggressively.

### Null Move Pruning
- Conditions: `!isNull && depth >= 3 && bestScore >= beta && (!ttHit || !(type==HASH_BETA) || ttScore >= beta) && NonPawnMaterial()`
- **TT guard**: prevents NMP when TT has a fail-high entry with score < beta (the TT already suggests null-move won't hold)
- Reduction: `R = 5 + depth/6 + min(3, (bestScore-beta)/100)`
  - At depth 12, eval=beta: R = 5 + 2 + 0 = 7
  - At depth 12, eval=beta+300: R = 5 + 2 + 3 = 10
- Returns beta for mate scores, raw nullScore otherwise
- Compare to ours: we use `R = 3 + depth/3 + min(3, (eval-beta)/200)`. Their base is much higher (5 vs 3) and divisor is smaller (6 vs 3), meaning more aggressive NMP reductions. At depth 12: their R=7-10 vs our R=7-10 (similar range but different distribution). Their eval scaling `/100` is twice as aggressive as our `/200`.
- **No NMP verification** at deep searches (unlike Altair at depth>15)

### ProbCut
- Conditions: `!inCheck && !onPV && depth >= 5`
- Beta cut: `beta + 100`
- TT guard: skip if TT entry exists with `depth >= (depth-4)` and `score < betaCut`
- **Two-stage QS pre-filter**: runs qsearch first (`-qSearch(-betaCut, -betaCut+1)`), only does full `depth-4` search if QS passes
- SEE filter: `SEE(move) >= betaCut - staticEval` (only considers captures that could reach the beta cutoff)
- Compare to ours: we have ProbCut with margin 170, gate 85. Their margin is 100 (tighter). They have the two-stage QS pre-filter that Alexandria and Tucano also use -- already in SUMMARY item #6 as "ProbCut QS pre-filter."

### Late Move Pruning (LMP)
- Conditions: `depth <= 8` (`m_lmpDepth = 8`), quiet moves only
- Table-based thresholds by depth and improving:
  - Not improving: `{0, 1, 2, 3, 5, 9, 13, 18, 25}` (depth 1-8)
  - Improving: `{0, 5, 7, 11, 17, 26, 36, 48, 63}` (depth 1-8)
- Sets `skipQuiets = true` when threshold exceeded (skips all remaining quiets)
- Compare to ours: we use `3+d^2` (+50% improving). Their thresholds are tighter at low depths (depth 2: 2 vs our 7) but extend to depth 8 (we use variable depth). Their improving multiplier is much larger (~3-5x vs our 1.5x).

### Countermove History Pruning (CMP)
- Conditions: quiet, `depth <= m_cmpDepth[improving]`
  - Not improving: `depth <= 3, cmhistory < 0`
  - Improving: `depth <= 2, cmhistory < -1000`
- Prunes quiet moves with poor counter-move history
- Compare to ours: we don't have separate CMP. Already in SUMMARY item #35.

### Followup Move History Pruning (FMP)
- Conditions: quiet, `depth <= m_fmpDepth[improving]`
  - Not improving: `depth <= 3, fmhistory < -2000`
  - Improving: `depth <= 2, fmhistory < -4000`
- Prunes quiet moves with poor follow-up move history
- Compare to ours: we don't have separate FMP. Already in SUMMARY item #35.

### Futility Pruning
- Conditions: `depth <= 8`, quiet moves only
- Margin: `staticEval + 90 * depth`
- **History gate**: only prunes if `history + cmhistory + fmhistory < m_fpHistoryLimit[improving]`
  - Not improving: combined history < 12000
  - Improving: combined history < 6000
- Sets `skipQuiets = true` (all remaining quiets pruned)
- Compare to ours: we use `80 + lmrDepth*80` per-move (continue, not break). Their margin is looser (90*d vs ~80*d at similar depths). The **history gate** is novel -- don't futility-prune moves with very strong history. Already in SUMMARY item #36.

### SEE Pruning
- Conditions: `depth <= 8 && !inCheck`
- Quiet margin: `-60 * depth` (`SEEQuietMargin = -60`)
- Noisy margin: `-10 * depth^2` (`SEENoisyMargin = -10`)
- Compare to ours: we use `-20*d^2` for SEE. Their quiet margin is linear (-60*d) vs our quadratic approach. Their noisy margin (-10*d^2) is gentler than our unified approach.

---

## 3. Extensions

### Check Extension
- +1 depth unconditionally when in check
- Applied via `extensionRequired()` function

### History-Based Extension
- Conditions: `!onPV && cmhistory >= 10000 && fmhistory >= 10000`
- Extends by 1 ply when both counter-move history and follow-up history are very strong
- Already in SUMMARY item #37.

### Singular Extensions
- Depth guard: `depth >= 8`
- Conditions: `!skipMove && hashMove == mv && !rootNode && !isCheckMateScore(ttScore) && hEntry.type == HASH_BETA && hEntry.depth >= depth - 3`
- Singularity beta: `ttScore - depth` (i.e., margin = 1*depth)
- Singularity depth: `depth / 2`
- If score < betaCut: +1 extension
- **Double extension**: if `!onPV && score < betaCut - 50`: +2 extension
- **Multi-cut shortcut**: if `betaCut >= beta`: return betaCut directly (skips remaining search)
- **Negative extension**: if `ttScore >= beta`: -2 extension
- Compare to ours: our SE is currently broken (-58 to -85 Elo). Their margin `1*depth` is much tighter than our `3*depth`. They have double extension (+2) for very singular moves, multi-cut for pruning, and negative extension for non-singular. These are all features Alexandria also has.

---

## 4. LMR (Late Move Reductions)

### Table Initialization
Single table (not split for captures/quiets):
- Formula: `0.75 + log(depth) * log(moves) / 2.25`
- Base: 0.75 (lower than typical), divisor: 2.25

Compare to ours: we use split tables -- captures C=1.80, quiets C=1.50. Their single table with base 0.75 and divisor 2.25 is different from both.

### Application Conditions
- `depth >= 3 && quietMove && legalMoves > 1 + 2*rootNode`
- Only applies to quiet moves (no capture LMR)
- Root: starts at move 4+ (1 + 2*1 = 3), non-root: move 2+ (1 + 2*0 = 1)

### Reduction Adjustments
- `+1` for cut node
- `-2` for PV node
- `-1` for killer move (either slot)
- History-based: `-max(-2, min(2, (history + cmhistory + fmhistory) / 5000))`
- Clamped: `reduction >= newDepth` -> `newDepth - 1`; `reduction < 0` -> `0`

### What they DON'T have (that we or others do):
- No capture LMR (only quiet moves get reduced)
- No TT-based adjustments (ttPv flag, noisy TT move detection)
- No DoDeeper/DoShallower after LMR re-search
- No cutnode extra reduction beyond +1
- No improving-based adjustment
- No eval-alpha distance adjustment

---

## 5. Move Ordering

### Score Hierarchy
All scores computed upfront, selection sort during iteration:

1. **TT move**: 7,000,000 (`s_SortHash`) -- swapped to position 0 immediately
2. **Good captures**: 6,000,000 + 10*(captured_value + promo_value) - piece_value (`s_SortCapture`)
3. **Killer moves**: 5,000,000 (`s_SortKiller`, both slots treated equally)
4. **Quiet moves**: `history[color][from][to] + followTable[0][counterPiece][counterTo][piece][to] + followTable[1][followPiece][followTo][piece][to]`
5. **Bad captures**: 1,000,000 + SEE score (`s_SortBadCapture`) -- still above most quiets

### Not Staged
Like Midnight, Igel generates ALL moves upfront, scores them, then uses selection sort (pick-best). Not staged.

### History Tables

**Main history**: `m_history[2][64][64]` -- color, from, to
- Bonus formula: `entry += 32 * delta - entry * |delta| / 512`
  - Where `delta = bonus` for best move, `delta = -bonus` for non-best quiets
  - Bonus = `depth * depth` (depth squared)
- Max bonus capped at 400 (`s_historyMax`)
- Gravity: `s_historyMultiplier=32`, `s_historyDivisor=512`
- Compare to ours: we use `bonus - entry*|bonus|/divisor` with divisor=5000. Their divisor (512) is much smaller, meaning faster decay / more responsive to recent data.

**Continuation history**: `m_followTable[2][14][64][14][64]`
- Index 0: counter-move history (prev piece+to -> current piece+to)
- Index 1: follow-up move history (2-ply-ago piece+to -> current piece+to)
- Same bonus/gravity formula as main history
- Compare to ours: same 2-ply structure. They don't have deeper continuation history (plies 4, 6).

**Killer moves**: 2 slots per ply, FIFO replacement

### Notable Absences
- No counter-move heuristic (just killer + continuation history for ordering)
- No capture history table (captures ordered by MVV-LVA + SEE only)
- No pawn territory penalty (unlike Midnight)

---

## 6. Time Management

### Base Allocation
**Normal time control (no movestogo):**
- Reserves 100ms overhead if remaining > 200ms
- Hard limit: `remainingTime / 13 + increment / 2 + enemyLowTimeBonus`
- Soft limit: `hardLimit / 4` (with increment) or `hardLimit / 6` (without increment)
- Compare to ours: similar structure but different ratios. Their `/13` hard limit is more conservative than typical `/5` or `/10`.

**Moves-to-go:**
- Hard limit: `remainingTime / movestogo + increment / 2 + enemyLowTimeBonus`
- If movestogo == 1: hard limit halved
- Otherwise: middle game time bonus (1.5x for first 20 moves)
- Soft limit: `hardLimit / 2`

### Score-Drop Time Extension
- Active at `depth >= 8`
- If score drops from previous iteration:
  - `softLimit *= min(1.0 + (prevScore - score) / 80.0, 1.5)`
  - Capped at hardLimit
- Example: 40cp drop -> softLimit *= 1.5 (maximum)
- Compare to ours: **(UPDATE 2026-03-21: GoChess now has score-drop time extension with 2.0x/1.5x/1.2x tiered scaling.)** Their system is more granular (continuous) but caps at 1.5x; ours has higher max (2.0x) but is tiered.

### Enemy Time Bonus
- When we have more time than opponent (but not > 5x more): `bonus = (ourTime - theirTime) / 10`
- Unique feature: uses opponent's time pressure as a signal to think more carefully

### Notable Absences
- **No node-based time management** (no bestmove stability tracking)
- **No fail-low time extension** (just score-drop scaling)
- **No aspiration fail-high depth reduction**

---

## 7. NNUE Architecture

### Network Topology
HalfKP with 32 king buckets, mirrored horizontally:
- **Input**: 11 piece types (10 non-king + 1 king type) x 64 squares x 32 king buckets = 22528 features
- **Feature transformer**: 22528 -> 2x1024 (perspective, with PSQT side accumulation)
- **Pairwise multiplication**: first 512 x second 512 of each 1024-dim accumulator = 512 output pairs per side, 1024 total
- **Hidden layer 1**: 1024 -> 16 (int8 quantized, VPMADDUBSW on AVX2)
- **Dual activation on L1 output**:
  - First 15 outputs: SCReLU path (`(x^2) >> 12 / 128`)
  - Last 15 outputs: standard CReLU path (clamp to [0,127])
  - Combined: 30 features for L2 (15 SCReLU + 15 CReLU, with shared element at position 15)
- **Hidden layer 2**: 32 -> 32 (int8)
- **Output layer**: 32 -> 1
- **Output skip connection**: L1 output[15] * (600*16)/(127*64) added directly to final output
- **8 output buckets** by piece count: `bucket = (popcount(all) - 1) / 4`
- **PSQT blending**: `((128-delta)*psqt + (128+delta)*eval) / 128 / 16` where delta=7 (slight bias toward eval)

### Eval Post-Processing
- Material scaling: `scale = 600 + 20 * nonPawnMaterial / 1024`
- Applied: `eval = NnueEval * scale / 1024`
- **50-move decay**: `eval * (208 - fifty) / 208` (slower decay than typical `(200-fmr)/200`)
- Tempo: +20cp added at the end

### Compared to Our NNUE
- Ours: v5 (768x16->N)x2->1x8 (16 king buckets, CReLU/SCReLU, 8 output buckets, pairwise mul, Finny tables) **(UPDATE 2026-03-21)**
- Theirs: 22528->2x1024->16->32->1 x8 (32 king buckets, pairwise mul + dual SCReLU/CReLU, 8 output buckets)
- Their feature transformer is much wider (2x1024 vs 2x256) but uses pairwise multiplication to compress to 1024 before L1
- Their dual activation (SCReLU + CReLU) is unusual -- most engines use one or the other
- Their output skip connection from L1[15] to final output is a form of residual connection
- Their material scaling and 50-move decay are post-processing steps we don't have (material scaling is in SUMMARY item #14g)

### Accumulator
- Stack-based (stored in Undo records, full copy on MakeMove)
- Incremental updates for feature add/remove
- King moves trigger full recompute (per king bucket)
- SIMD: AVX2 required (no fallback, no ARM/NEON)

---

## 8. Transposition Table

### Structure
- **4-slot buckets** (`TTCluster` of 4 `TEntry`s, 64 bytes = 1 cache line)
- Each entry: 16 bytes (8-byte packed data + 8-byte key)
- Packed data: move(24), age(8), type(2), score(22), depth(8) -- all in a union with U64
- Index: `hash % m_hashSize` (modulo, not power-of-2 masking)
- **XOR-verified**: `m_key = hash0 ^ m_data.raw` on store; `(hentry.m_key ^ hentry.m_data.raw) == hash` on retrieve
- Age-based replacement

### Replacement Policy
Searches for empty slot or hash match first (break immediately). Otherwise, replaces the entry with the lowest priority based on:
- `(same_age) - (replace_same_age) - (shallower_than_replace) < 0`
- Prefers replacing: different-age entries, then deeper-than-replace entries

### Probe / Cutoff Logic
- Main search: requires `hEntry.depth >= depth && (depth==0 || !onPV)` and `Fifty() < 90`
  - Never gives cutoffs on PV nodes (unless depth==0)
  - Respects 50-move rule proximity (won't cut at fifty>=90)
- QSearch: requires `hEntry.depth >= tteDepth` where tteDepth = 0 (in check or depth>=0) or -1 (otherwise)
  - Never cuts on PV in QSearch either
- Standard alpha/beta/exact cutoff logic

### TT Score Correction
- TT-corrected bestScore used for pruning decisions: if TT has a bound that improves on staticEval, use ttScore
  - `(HASH_BETA && ttScore > staticEval)` or `(HASH_ALPHA && ttScore < staticEval)` or `HASH_EXACT`
- This is standard but worth noting: bestScore (used in RFP, NMP) may differ from staticEval

### Compare to Ours
- Same 4-slot bucket structure
- Same XOR-verified lockless design
- Their modulo indexing vs our power-of-2 masking (modulo is slightly more cache-friendly but slower on the index computation)
- Their age-based replacement is more sophisticated than ours (we use depth + generation)
- Their 50-move safety check (`Fifty() < 90`) is a nice guard we don't have

---

## 9. Lazy SMP

### Thread Management
- Main thread + N worker threads (configurable via UCI `Threads` option)
- Workers share only the TT (global singleton)
- Board, search state, history tables, killer moves, eval stacks all per-thread
- Workers launched via condition variables; signaled to start/stop

### Vote-Based Best Move Selection
After all threads finish, Igel uses a **vote-based** system to pick the best move:
- Each thread votes for its best move with weight: `(score - worst_score + 20) * depth`
- Votes are aggregated per move in a map
- The move with the highest total vote weight wins
- This is more sophisticated than simple "pick deepest/best score" -- it considers consensus across threads

### Compare to Ours
- We use simple "main thread decides" approach
- Their vote-based system could find better moves when threads disagree
- However, it adds complexity and the main thread's result is usually correct

---

## 10. Notable Differences from GoChess

### Things Igel has that we don't:
1. **Mate distance pruning** -- standard 3-line technique, trivial to add. Already priority #3 in SUMMARY.
2. **Countermove history pruning (CMP)** -- separate from general history pruning, granular per-component thresholds. Already in SUMMARY #35.
3. **Followup move history pruning (FMP)** -- same as above for ply-2 continuation. Already in SUMMARY #35.
4. **Futility history gate** -- don't prune moves with combined history > 12000/6000. Already in SUMMARY #36.
5. **History-based extensions** -- extend when both cmhist and fmhist > 10000. Already in SUMMARY #37.
6. **ProbCut QS pre-filter** -- two-stage ProbCut. Already in SUMMARY #6.
7. **Singular double extension** -- +2 when score < betaCut - 50. Standard in many engines.
8. **Singular multi-cut** -- return betaCut when betaCut >= beta. Powerful pruning shortcut.
9. **Singular negative extension** -- -2 when ttScore >= beta. Standard in many engines.
10. **TT NMP guard** -- skip NMP when TT fail-high with score < beta. Novel safety check.
11. **Vote-based Lazy SMP best-move selection** -- consensus across threads.
12. **Enemy time bonus** -- use opponent's time pressure in time allocation.
13. **Aspiration delta=5** -- much tighter than our 15. Already in SUMMARY #40. **(UPDATE 2026-03-21: GoChess now has aspiration contraction but delta is still 15)**
14. **50-move eval decay** -- `eval * (208-fifty) / 208`. Already tested (H0 at -3.0 Elo, SUMMARY #32).
15. **Material-dependent NNUE scaling** -- `(600 + 20*nonPawnMaterial/1024) / 1024`. Similar to Alexandria's. Already in SUMMARY #14g.
16. **TT 50-move safety** -- skip TT cutoffs at Fifty() >= 90.
17. **Passer push detection in move ordering** -- `isSpecialMove()` checks pawn rank 5/6, though it doesn't appear to be actively used in scoring.

### Things we have that Igel doesn't:
1. **Split LMR tables** (separate capture/quiet) -- our key +43.5 Elo win
2. **Capture LMR** -- they only reduce quiet moves
3. **Counter-move heuristic** in move ordering
4. **Capture history table**
5. **TT score dampening** -- our +22.1 Elo win
6. **Fail-high score blending** -- our +14.7 Elo win
7. **Alpha-reduce** -- our +13.0 Elo win
8. **NMP threat detection** -- our +12.4 Elo win
9. **QS beta blending** -- our +4.9 Elo win
10. **TT near-miss cutoffs** -- our +21.7 Elo win
11. **TT noisy move detection for LMR** -- our +34.4 Elo win
12. **Correction history** (any form)
13. **Staged move generation** -- they generate all moves upfront
14. **Node-based time management** (bestmove stability)
15. **DoDeeper/DoShallower after LMR**

### Parameter Comparison Table

| Feature | Igel | GoChess |
|---------|------|---------|
| RFP margin | 85*(d-improving), d<=8 | 85*d (imp) / 60*d (not), d<=8 |
| Razoring | eval+150 < alpha, d<=2 | 400+100*d, d<=3 |
| NMP base R | 5 + d/6 | 3 + d/3 |
| NMP eval scale | (bestScore-beta)/100 | (eval-beta)/200 |
| NMP TT guard | Yes (skip if TT fail-high < beta) | No |
| ProbCut margin | beta + 100 | beta + 170 |
| ProbCut QS pre-filter | Yes | No |
| ProbCut depth | >= 5 | >= 5 |
| LMR base | 0.75 | split: cap 1.80, quiet 1.50 |
| LMR divisor | 2.25 | split: cap 2.36 impl, quiet 2.36 impl |
| LMR captures | No (quiet only) | Yes (separate table) |
| LMR cut-node | +1 | +1 |
| LMR PV | -2 | -1 |
| LMR history div | /5000 (combined) | /5000 |
| LMP depth | <= 8, table-based | 3+d^2, +50% improving |
| Futility margin | 90*d, d<=8 | 80+d*80 |
| Futility history gate | Yes (12000/6000) | No |
| SEE quiet | -60*d, d<=8 | similar |
| SEE noisy | -10*d^2, d<=8 | -20*d^2 |
| CMP | Yes, d<=3/2 | No |
| FMP | Yes, d<=3/2 | No |
| Singular depth | >= 8 | >= 10 (broken) |
| Singular beta | ttScore - depth | ttScore - 3*depth |
| Singular double ext | +2 at betaCut-50 | No |
| Singular multi-cut | Yes (return betaCut) | No |
| Singular neg ext | -2 when ttScore>=beta | No |
| History ext | cmhist>=10000 && fmhist>=10000 | No |
| Aspiration delta | 5 | 15 |
| Aspiration growth | +2 + asp/2 | similar |
| TT buckets | 4-slot | 4-slot |
| TT entries | 16 bytes | similar |
| History bonus | d^2, capped at 400 | similar |
| History gravity | 32*delta - entry*|delta|/512 | bonus - entry*|bonus|/divisor |
| Cont-hist depth | plies 1, 2 | plies 1, 2 |
| Time soft | hard/4 (w/inc) or hard/6 | similar |
| Time hard | remaining/13 + inc/2 | similar |
| Time score-drop | min(1+(drop/80), 1.5) | 2.0x/1.5x/1.2x tiered (UPDATE: merged) |

---

## 11. Ideas Worth Testing from Igel

### Already tracked in SUMMARY:
- **Mate distance pruning** -- #3 priority, trivial, 3 lines.
- **CMP/FMP** -- #35, 5 lines each. More granular than general history pruning.
- **Futility history gate** -- #36, 2 lines. Don't futility-prune strong-history moves.
- **History-based extensions** -- #37, 2 lines. Extend on very strong cont-hist.
- **ProbCut QS pre-filter** -- #6, medium complexity.
- **Aspiration delta=5** -- #40, tighter bounds.
- **Material scaling of NNUE output** -- #14g, cheap post-processing.

### New or refined ideas from Igel:

1. **NMP TT guard** -- skip NMP when TT has fail-high with score < beta. This prevents NMP in positions where TT data contradicts the null-move hypothesis. 1 condition. We previously tested "NMP TT guard" at 0 Elo, but their specific condition (`!ttHit || !(type==HASH_BETA) || ttScore >= beta`) is different from what we may have tested.

2. **NMP base R=5** -- their `5 + depth/6` is significantly more aggressive than our `3 + depth/3`. At depth 12: both give R=7 without eval bonus, so the main difference is at low/medium depths. At depth 6: their R=6, ours R=5. Worth testing higher base R.

3. **Razor at depth<=2 with fixed margin 150** -- much more aggressive razoring than our 400+100*d. If their tighter razor works, it means more positions drop to QSearch sooner. Could be net-dependent.

4. **TT 50-move safety** -- skip TT cutoffs at Fifty() >= 90. Prevents stale TT data from causing incorrect cutoffs near the 50-move boundary. Trivial to add (1 condition).

5. **Vote-based Lazy SMP** -- when threads disagree, pick the most-voted move weighted by score and depth. Could be better than always trusting the main thread. Medium complexity.

6. **RFP with TT-corrected eval** -- they use `bestScore` (which may be TT-adjusted) instead of raw `staticEval` for RFP. This is more informative since TT data refines the eval estimate. We may already do this.

7. **NMP eval scaling /100** -- twice as aggressive as our /200. Each 100cp above beta adds +1 to R (vs our 200cp per +1R). Could be too aggressive but worth testing alongside the higher base R.
