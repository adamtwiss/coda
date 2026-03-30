# Koivisto Chess Engine - Crib Notes

Source: `~/chess/engines/Koivisto/`
Authors: Kim Kahre, Finn Eggers, Eugenio Bruno
NNUE: HalfKP-like 12288->2x512->1 (16 king buckets, CReLU, int16 hidden layer)
RR Ranking: #9 at +194 Elo (302 above GoChess-v5)

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. Lazy SMP with shared TT (like us). C++ with template-based color for move generation. `pvSearch` has a `behindNMP` parameter tracking which color is behind a null move (unique -- used for LMR adjustments).

### Iterative Deepening
- Depth 1 to MAX_PLY (or depth limit)
- At root: single legal move detection stops search early
- `prevScore` tracked for eval-based time management

### Aspiration Windows
- Enabled at depth >= 6
- Initial window: `score +/- 10`
- On fail-high: `beta += window; sDepth--` (depth reduced on fail-high)
- On fail-low: `beta = (alpha + beta) / 2; alpha -= window` (beta contracts toward alpha)
- Window doubles each iteration: `window += window`
- Full-width at window > 500
- `sDepth` floor: `max(sDepth, depth - 3)`

### Draw Detection
- `isDraw()` handles 50-move and repetition
- Draw randomization (Beal effect): returns `8 - (nodes & 15)` (range -7 to +8)
- Special handling: at 50-move boundary, check for legal evasions (stalemate vs checkmate)

### Lazy SMP
- Shared TT across all threads (up to 256 threads)
- Per-thread: Board copy, ThreadData (nodes, seldepth, SearchData, move generators, PV table)
- Main thread (ID 0) spawns worker threads, handles time management and info strings
- TT not lockless (no atomic packing) -- relies on same-index replacement being benign

---

## 2. Pruning Techniques

### Reverse Futility Pruning (RFP / "Static Null Move Pruning")
- Conditions: `!inCheck && !pv && !skipMove`
- Depth guard: `depth <= 7`
- Margin: `(depth - (isImproving && !enemyThreats)) * FUTILITY_MARGIN` where `FUTILITY_MARGIN = 68`
- Formula: `staticEval >= beta + margin && staticEval < MIN_MATE_SCORE` => return staticEval
- **Key difference**: margin scales by `depth - 1` when improving AND no enemy threats (effectively tighter margin). Enemy threats guard against pruning.
- Compare to ours: we use 80*d improving / 80*d not-improving (merged futility tightening), depth<=8. They use 68*(depth-improving_and_no_threats), depth<=7.

### Extra RFP at Depth 1
- `depth == 1 && staticEval > beta + (isImproving ? 0 : 30) && !enemyThreats` => return beta
- A separate shallow pruning with tighter margin

### Razoring
- Conditions: `!inCheck && !pv && !skipMove`
- Depth guard: `depth <= 3`
- Margin: `RAZOR_MARGIN * depth` where `RAZOR_MARGIN = 190`
  - depth 1: 190, depth 2: 380, depth 3: 570
- Formula: `staticEval + 190 * depth < beta` => drop to qSearch; return score if `< beta`
- Compare to ours: we use 400+100*d (depth 1: 500, depth 2: 600, depth 3: 700). Theirs is tighter.

### Null Move Pruning (NMP)
- Conditions: `staticEval >= beta + (depth < 5 ? 30 : 0) && !(depth < 5 && enemyThreats > 0) && !hasOnlyPawns`
- **Threat guard**: disables NMP at shallow depth (< 5) when opponent has threats. This is a Koivisto original.
- Reduction: `nmpDepthAdjustment = depth/4 + 3`
- Futility adjustment: `nmpFutilityAdjustment = (staticEval - beta) / 68` when `staticEval - beta < 300`, otherwise 3
- Total R: `depth/4 + 3 + futilityAdj` (typically 3-9)
- Mate score clamping: returns `beta` instead of raw score when `score >= TB_WIN_SCORE_MIN`
- Compare to ours: we use `3 + d/3 + min((eval-beta)/200, 3)`. Theirs uses d/4+3 base (slightly larger at depth 12+), and the futility part uses their smaller FUTILITY_MARGIN=68 as divisor (more aggressive scaling).

### Internal Iterative Reduction (IIR)
- Condition: `depth >= 4 && !hashMove`
- Reduces depth by 1
- Standard implementation

### Late Move Pruning (LMP)
- Table-based: `lmp[2][8]`
  - Not improving: `{0, 2, 3, 5, 8, 12, 17, 23}`
  - Improving:     `{0, 3, 6, 9, 12, 18, 28, 40}`
- Depth guard: `depth <= 7`
- When triggered: sets `skip()` flag on move generator, which causes all remaining quiets to be skipped (but still searches bad noisy moves)
- After LMP, only noisy moves that attack king capture bitboard (checker squares) are searched among quiets
- Compare to ours: we use `3+d^2` base with +50% improving. Their table is hand-tuned with more values.

### Futility Pruning (quiet moves)
- Uses `maxImprovement` table (tracks largest eval improvement for each from-to pair)
- Formula: `maxImprovement[from][to] + moveDepth * 68 + 100 + evalHistory <= alpha`
- `moveDepth = max(1, 1 + depth - lmrReductions[depth][legalMoves])`
- Depth guard: `moveDepth <= 7`
- Compare to ours: we use `80 + lmrDepth*80`. Their approach uses per-move improvement history + eval history, making it adaptive.

### History Pruning
- Conditions: `!inCheck && depth > 0`
- Threshold: `getHistories(m) < min(140 - 30 * depth * (depth + isImproving), 0)`
  - depth 1 improving: `140 - 30*1*2 = 80` (doesn't prune, > 0)
  - depth 2 not improving: `140 - 30*2*2 = 20` (doesn't prune, > 0)
  - depth 3 not improving: `140 - 30*3*3 = -130`
  - depth 4 not improving: `140 - 30*4*4 = -340`
- Applies to quiet moves only
- Compare to ours: we use -2000*depth. Their threshold is quadratic (tighter at deeper depth).

### SEE Pruning
- Conditions: `legalMoves >= 1 && highestScore > -MIN_MATE_SCORE`
- `moveDepth <= 5 + quiet*3` (depth 8 for quiets, depth 5 for captures)
- Also requires: captured piece type < moving piece type (captures)
- Quiet margin: `-40 * moveDepth`
- Capture margin: `-100 * moveDepth`
- Compare to ours: similar margins. Their `moveDepth` includes LMR effect (effectively tighter for reduced moves).

### ProbCut
- Conditions: `!inCheck && !pv && !skipMove && depth > 4 && ownThreats > 0`
- betaCut = `beta + 130`
- **Unique**: Only triggers when we have our own threats (ownThreats > 0). This is a Koivisto original.
- Two-stage: first qSearch at `-betaCut`, then pvSearch at `depth - 4` if qSearch passes
- Guard: skipped if `hashMove && en.depth >= depth-3 && ttScore < betaCut`
- Stores result in TT at `depth - 3`
- Compare to ours: we use margin 170, gate 85. They use 130 margin but the threat guard is unique.

---

## 3. Extensions

### Singular Extensions
- Depth guard: `depth >= 8`
- Conditions: `!skipMove && !inCheck && m == hashMove && legalMoves == 0 && ply > 0 && en.depth >= depth-3 && abs(ttScore) < MIN_MATE_SCORE && (en.type == CUT_NODE || en.type == PV_NODE)`
- Singularity beta: `min(ttScore - SE_MARGIN_STATIC - depth * 2, beta)` where `SE_MARGIN_STATIC = 0`
  - So effectively: `min(ttScore - 2*depth, beta)`
- Singularity depth: `depth >> 1` (half depth)
- If `score < betaCut`:
  - **LMR cancellation**: if parent passed an `lmrFactor` pointer, adds `*lmrFactor` back to depth and zeroes it. This is unique -- singular moves undo their parent's LMR.
  - +1 extension
- **Multi-cut**: if `score >= beta` => return score immediately
- **Secondary multi-cut**: if `ttScore >= beta`, runs another search at `(depth >> 1) + 3`; returns score if >= beta
- After SE, re-initializes the move generator to resume from the hash move
- Compare to ours: we use `depth*3` margin (much wider). Their LMR cancellation feature is unique.

### Check Extension
- `depth > 4 && b->isInCheck(opponent)` after making the move => extension = 1
- Only when no other extension was applied (extension == 0)
- Compare to ours: we removed check extensions (-11.2 Elo). They restrict to deeper depths only.

### Non-PV Hash Move Extension
- `sameMove(hashMove, m) && !pv && en.type > ALL_NODE` => extension = 1
- Extends when playing the TT move in a non-PV node with a cut/PV-type TT entry

### Eval-Based Extension (Non-PV)
- `depth < 8 && !skipMove && !inCheck && m == hashMove && ply > 0`
- `evalHistory < alpha - 25 && en.type == CUT_NODE` => extension = 1
- Extends when eval is below alpha but TT says this was a cut node (position is potentially tricky)

---

## 4. LMR (Late Move Reductions)

### Table Initialization
- Single table: `lmrReductions[256][256]`
- Formula: `1.25 + log(d) * log(m) * 100 / 267`
- `LMR_DIV = 267`
- Compare to ours: we use separate cap/quiet tables. Their divisor 267 = 100/267 ~ 0.375, vs our quiet 100/150 ~ 0.667. Their base reduction is significantly lower.

### Application Conditions
- `legalMoves < 2 - (hashMove != 0) + pv || depth <= 2 || (isCapture && SEE > 0) || (isPromotion && queen promotion)` => lmr = 0
- Effectively: no reduction for first 1-2 moves (depending on PV/hash), depth <= 2, winning captures, queen promotions

### Reduction Adjustments
1. `+1` if behind null move and same side as NMP (`activePlayer == behindNMP`). Unique feature: tracks which color is behind a null move through recursion.
2. `-history/150` based on combined history scores
3. `+1` if not improving
4. `-1` if PV node
5. `+1` if `!targetReached` (haven't used 10% of allocated time yet). Unique: LMR increases early in the search when we're still "exploring".
6. `-1` if killer move
7. `+1` if PV node, first move reduced at root (`sd->reduce && sd->sideToReduce != activePlayer`). This is the PV-first-move-reduce-hint propagation.
8. `+min(2, abs(staticEval - alpha) / 350)` -- eval-alpha distance. More reduction when eval is far from alpha.
9. `-bitCount(getNewThreats(b, m))` -- reduce less for moves creating new threats (attacks on higher-value pieces)
10. Clamped to `[0, depth-2]`
11. **Override to 0**: if `history > 256 * (2 - isCapture)` (512 for quiets, 256 for captures), force no reduction

### LMR Re-Search
- Standard PVS: ZWS with reduced depth, then ZWS at full depth if score > alpha, then full window if score > alpha
- **LMR factor passing**: the `lmrFactor` pointer is passed to child pvSearch. If the child is singular, it can cancel the parent's LMR. This is a deeply unique feature.

---

## 5. Move Ordering

### Staged Move Generation
Order: hash move -> good noisy -> killer1 -> killer2 -> quiet -> bad noisy

### Noisy Move Scoring
- SEE computed upfront for all captures during generation
- Good captures (SEE >= 0): `100000 + getHistories(m) + SEE + 150 * (sqTo == prevSqTo)`
  - Recapture bonus: +150 for captures on same square as previous move
- Bad captures (SEE < 0): `10000 + getHistories(m)`
- Good captures sorted by selection sort; bad captures searched after all quiets

### Quiet Move Scoring
- `threatHistory[side][threatSquare][fromTo] + cmh[prevPiece*64+prevTo][side][pieceTo] + fmh[followupPiece*64+followupTo][side][pieceTo]`
- Three tables combined directly (not averaged)
- No butterfly history (from-to only) -- replaced by threat history

### History Tables

**Threat History**: `th[2][65][4096]` -- side, threat_square (0-64), from-to combination
- This is Koivisto's MAIN butterfly-equivalent table, indexed by the opponent's primary threat square
- 65 entries for threat square: 64 actual squares + index 64 for "no threat"
- This replaces traditional butterfly history with a threat-context-aware version

**Counter-Move History (CMH)**: `cmh[384][2][384]` -- prev_piece*64+prev_to, side, curr_piece*64+curr_to
- Standard 1-ply continuation history

**Followup-Move History (FMH)**: `fmh[385][2][384]` -- followup_piece*64+followup_to, side, curr_piece*64+curr_to
- Standard 2-ply continuation history

**Capture History**: `captureHistory[2][4096]` -- side, from-to combination
- Used for capture scoring. Same from-to indexing as butterfly (but captures only)

**Killers**: 2 per ply per color. Grandchildren killers reset at each node.

### getHistories() -- Combined Score
- Captures: returns `captureHistory[side][fromTo]`
- Quiets: returns `(2*fmh + 2*cmh + 2*th) / 3` (effectively weighted average of 3 tables)

### History Update (Gravity Formula)
- Weight: `min(depth^2 + 5*depth, 384)` (e.g., depth 3: 24, depth 5: 50, depth 10: 150, depth 12: 204, capped at 384)
- Update: `value += weight - weight * value / MAX_HIST` (bonus) or `value += -weight - weight * value / MAX_HIST` (malus)
- `MAX_HIST = 512`
- All 3 quiet tables (th, cmh, fmh) updated on cutoff
- Capture history updated for all captures searched (bonus for best, malus for others)
- **Eval-based bonus**: `depth + (staticEval < alpha)` adds +1 depth to history update weight when eval was below alpha (surprising cutoff)
- Malus weight capped at 128 for non-best moves

### Unique: spentEffort Table
- `spentEffort[64][64]` -- from, to
- Tracks node count per root move, used for time management

### Unique: maxImprovement Table
- `maxImprovement[64][64]` -- from, to
- Tracks the maximum eval improvement when opponent played move from-to
- Used in futility pruning as per-move margin adjustment

---

## 6. Threat Computation (Koivisto Original)

At every non-check node, Koivisto computes threats for BOTH sides:
- **Pawn threats**: pawn attacks on opponent minors + majors
- **Minor threats**: knight/bishop attacks on opponent rooks + queens
- **Rook threats**: rook attacks on opponent queens

Stored as:
- `threatCount[ply][side]` -- count of threatening attacks
- `mainThreat[ply]` -- square of first threat to the active player (or 64 if none)

Used in:
1. **RFP**: margin reduced when improving AND no enemy threats
2. **NMP**: disabled at shallow depth when enemy has threats
3. **ProbCut**: only triggers when we have our own threats
4. **Threat History**: main threat square indexes the butterfly-replacement history table
5. **LMR**: `-bitCount(getNewThreats(b, m))` reduces LMR for moves creating new threats

`getNewThreats()` computes threats the move CREATES by looking at attack differences:
- Rook: new rook attacks on queen
- Bishop: new bishop attacks on rook/queen
- Knight: new knight attacks on rook/queen
- Pawn: new pawn attacks on all non-pawn pieces
- Queen/King: returns 0 (queens already threaten, kings don't create threats)

---

## 7. Time Management

### Base Allocation
- `timeToUse = 2*inc + 2*time/movesToGo`
- Hard limit: `time / 2`
- Both clamped to `time - overhead - inc - overheadPerGame`
- Under 1000ms with no increment: multiply time by 0.7

### Node-Based Soft Time Scaling
- `nodeScore = spentEffort[bestFrom][bestTo] * 100 / max(1, nodes)`
- Converted: `nodeScore = 110 - min(nodeScore, 90)` (range 20-110)
- So 90%+ effort on best move => nodeScore=20 (cut time); <10% effort => nodeScore=110 (extend time)

### Eval-Based Time Scaling
- `evalScore = min(max(50, 50 + evalDrop), 80)`
- `evalDrop = prevScore - score` (positive when score dropped)
- If score dropped significantly, evalScore increases toward 80 (spend more time)

### Combined Decision
- `match_time_limit.time_to_use * nodeScore / 100 * evalScore / 65 < elapsed` => stop
- This is a multiplicative scaling: nodeScore and evalScore both scale the target time
- Maximum multiplier: 110/100 * 80/65 = 1.35x (when best move unstable + score dropping)
- Minimum multiplier: 20/100 * 50/65 = 0.15x (when best move dominant + score stable)

### targetReached Flag
- Set `false` when `elapsed*10 < timeToUse` (using < 10% of time)
- Set `true` otherwise
- Used in LMR: `+1` reduction when target not reached (search faster early on)

---

## 8. QSearch

### Stand-Pat
- bestScore = stand_pat from TT eval if hit, otherwise full evaluation
- TT adjustment: if TT entry matches, adjust bestScore to ttScore (same as main search)
- Standard alpha/beta cutoff at stand_pat

### TT in QSearch
- Full TT probing with node-type cutoffs (PV returns, CUT >= beta, ALL <= alpha)
- No depth requirement (accepts any TT depth)

### Move Generation
- Q_SEARCH mode: only good noisy moves (SEE >= 0)
- Q_SEARCHCHECK mode: all noisy + king evasion quiets (when in check)

### QSearch Pruning
- **SEE < 0**: skip the capture
- **Good delta pruning**: `SEE + stand_pat > beta + 200` => return beta immediately
  - If SEE is winning AND stand_pat is already well above beta, no need to search further
  - Compare: we don't have this. It's a beta-side delta pruning (found previously in Seer/Minic).

### QSearch TT Storage
- Stores beta cutoffs at depth = `!inCheckOpponent` (0 or 1)
- Stores PV/ALL nodes at depth 0

---

## 9. Transposition Table

### Structure
- Single entry per index (no buckets)
- Index: `zobrist & mask` (power-of-2 masking)
- Key: upper 32 bits of zobrist (`zobrist >> 32`)
- Fields: U32 key, Move move (32-bit, upper 8 bits = age), Depth (8-bit), NodeType (8-bit), Score (16-bit), Eval (16-bit)
- Total: 12 bytes per entry
- Age stored in move's upper 8 bits (via getScore/setScore)

### Replacement Policy
Replace if ANY of:
1. Entry is empty (key == 0)
2. Different age (new search)
3. PV node type (always replaces)
4. Non-PV entry with `depth <= newDepth` (depth-preferred)
5. Same hash with `depth <= newDepth + 3` (very generous for same position)

### TT Probe in Main Search
- Non-PV only, requires: `en.depth + (!previousMove && ttScore >= beta) * 100 >= depth`
  - **Unique**: after null move (no previous move), TT entries are treated as if they have depth+100 for fail-high scores. This is the "child nodes of null moves" optimization -- since NMP already requires high proof, a TT fail-high after null move is almost always trustworthy.
- Standard alpha/beta/exact cutoffs

### Node Types
- `PV_NODE = 0, CUT_NODE = 1, ALL_NODE = 2, FORCED_ALL_NODE = 3`
- `FORCED_ALL_NODE`: stored when at deep nodes (depth > 7), the best move consumed > 50% of the node's effort. This distinguishes "forced" all-nodes from true all-nodes.
- ALL_NODE check uses `en.type & ALL_NODE` which matches both 2 and 3

### Eval Storage
- Static eval stored separately from search score
- Used for eval retrieval without depth requirement

---

## 10. NNUE Architecture

### Network: HalfKP-like with 16 King Buckets -> 2x512 -> 1
- **Input**: 6 piece types * 64 squares * 2 colors * 16 king buckets = 12,288 features
  - `INPUT_SIZE = N_PIECE_TYPES * N_SQUARES * 2 * 16` = 6 * 64 * 2 * 16 = 12,288
- **Hidden layer**: 512 neurons per perspective (1024 total with both perspectives)
- **Output**: single value (no output buckets)
- **Activation**: CReLU (clipped to [0, max_int16])
- **Quantization**: INPUT_WEIGHT_MULTIPLIER=32, HIDDEN_WEIGHT_MULTIPLIER=128
- **Final**: `(dotProduct + hiddenBias) / 32 / 128` = divide by 4096

### King Buckets
- 16 buckets, vertically symmetric (mirrored left-right)
- King bucket table: `{0,1,2,3,3,2,1,0, 4,5,6,7,7,6,5,4, 8,9,10,11,...}`
- Horizontal mirroring: if king on files e-h, flip square by `XOR 7`

### Accumulator
- Stack-based: `history[]` vector of Accumulators, `history_index` counter
- `addNewAccumulation()` increments index, sets `accumulator_is_initialised[color] = false`
- On first feature update for a color, copies from `history[index-1]` (lazy copy on first write)
- King moves trigger reset only if king bucket or half changes

### AccumulatorTable (FinnyTable equivalent)
- `AccumulatorTable::entries[2][32]` -- color, 32 entries (16 king buckets * 2 halves)
- Each entry stores: `piece_occ[2][6]` (bitboards per piece per color) + accumulator
- On king reset: computes diff between stored position and current, applies only deltas
- This is exactly the FinnyTable technique found in Alexandria/Obsidian
- **Memory**: 32 entries * 2 colors * (1024 bytes accumulator + 96 bytes occupancy) ~ 70KB per side

### Compared to Our NNUE
- Ours: v5 (768x16->N)x2->1x8 (8 output buckets, shallow wide, CReLU/SCReLU, Finny tables) **(UPDATE 2026-03-21: GoChess now has v5 architecture with pairwise mul, SCReLU, dynamic width, and Finny tables)**
- Theirs: 12288->2x512->1 (no output buckets, 2 perspectives, single hidden layer)
- No output buckets (we have 8)
- int16 weights throughout (we use int8 for hidden layer 1)
- Both have FinnyTable for efficient king recomputation

### SIMD
- AVX-512, AVX2, SSE2, NEON -- compile-time detection via preprocessor
- Evaluation: dot product of `max(accumulator, 0)` with hidden weights
- Both perspectives concatenated for output: active player first, then opponent

---

## 11. Notable Differences from GoChess

### Things Koivisto has that we don't:
1. **Threat computation at every node** -- bidirectional threat analysis driving RFP, NMP, ProbCut, history, and LMR
2. **Threat history** -- butterfly history indexed by opponent's main threat square (replaces plain from-to)
3. ~~**FinnyTable (AccumulatorTable)**~~ -- **(UPDATE 2026-03-21: GoChess now has Finny tables)**
4. **LMR behind-NMP adjustment** -- tracks which color is behind null move through recursion
5. **LMR targetReached** -- searches faster when still in early time allocation
6. **LMR new-threats reduction** -- moves creating threats get less reduction
7. **LMR eval-alpha distance** -- `+min(2, abs(eval-alpha)/350)`
8. **LMR high-history override** -- forces lmr=0 when history > 256/512
9. **LMR cancellation from child SE** -- singular extension in child can undo parent's LMR
10. **maxImprovement table** in futility pruning -- per-move adaptive margins
11. **Eval-based history depth bonus** -- +1 depth when staticEval < alpha (from Alexandria too)
12. **Mate distance pruning** -- standard but we don't have it
13. **FORCED_ALL_NODE** TT type -- distinguishes forced from true all-nodes
14. **TT depth relaxation after null move** -- TT entries after NMP treated as much deeper for fail-highs
15. **ProbCut threat guard** -- only ProbCut when own threats exist
16. **NMP threat guard** -- disable NMP at shallow depth when enemy has threats
17. ~~**Aspiration fail-low beta contraction**~~ -- `beta = (alpha + beta) / 2` **(UPDATE 2026-03-21: GoChess now has aspiration contraction)**
18. **Aspiration fail-high depth reduction** -- `sDepth--`
19. **QSearch good-delta pruning** -- `SEE + stand_pat > beta + 200 => return beta`
20. **Draw randomization (Beal effect)** -- `8 - (nodes & 15)`

### Things we have that Koivisto doesn't:
1. **Multi-layer NNUE** (4 layers vs their 1)
2. **Output buckets** (8 material-based buckets)
3. **Counter-move heuristic** (they have FMH but no explicit counter-move table)
4. **Correction history**
5. **Fail-high score blending** (`(score*d+beta)/(d+1)`)
6. **TT score dampening** (`(3*score+beta)/4`)
7. **Alpha-reduce** (depth-1 after alpha raised)
8. **TT noisy move detection** (+1 LMR for quiets when TT move is capture)
9. **TT near-miss cutoffs** (1 ply shallower acceptance)
10. **Multi-bucket TT** (4-slot buckets)
11. **Lockless TT** (atomic packing)
12. **QS beta blending** (`(bestScore+beta)/2`)

---

## 12. Parameter Comparison Table

| Feature | Koivisto | GoChess |
|---------|----------|---------|
| RFP margin | 68*(d-improving_no_threats), d<=7 | 80*d, d<=8 |
| RFP threat guard | Yes (enemyThreats) | No |
| Razoring | 190*d, d<=3 | 400+100*d, d<=3 |
| NMP base R | d/4+3 | 3+d/3 |
| NMP futility adj | (eval-beta)/68, max 3 | min((eval-beta)/200, 3) |
| NMP threat guard | Yes (d<5 && enemyThreats) | No |
| LMR table | 1.25 + log(d)*log(m)*100/267 | Separate cap C=1.80 / quiet C=1.50 |
| LMR history adj | -history/150 | -history/5000 |
| LMR behind-NMP | +1 for same side | No |
| LMR eval-alpha | +min(2, abs(eval-alpha)/350) | No |
| LMR new threats | -bitCount(threats) | No |
| LMR high-history | Force 0 if hist > 256/512 | No |
| LMP table | Hand-tuned [2][8] | 3+d^2 (+50% improving) |
| Futility margin | maxImprove + 68*moveDepth + 100 + evalHist | 80+d*80 |
| History pruning | 140-30*d*(d+imp), quadratic | -2000*d, linear |
| SEE quiet margin | -40*moveDepth | similar |
| SEE cap margin | -100*moveDepth | similar |
| SE depth | >=8 | >=10 |
| SE margin | ttScore - 2*depth | ttScore - 3*depth |
| SE multi-cut | Yes (2 variants) | No |
| SE LMR cancel | Yes (unique) | No |
| ProbCut margin | 130 | 170 |
| ProbCut guard | ownThreats required | None |
| Aspiration init | 10 | 15 |
| Aspiration fail-high | sDepth-- | No |
| History bonus | min(d^2+5d, 384) | similar |
| History gravity | MAX_HIST=512 | divisor 5000 |
| QS good-delta | SEE+stand_pat > beta+200 | No |
| King buckets | 16 | 16 |
| NNUE hidden | 2x512 | v5: dynamic width (1024/1536/any) |
| Output buckets | None | 8 |
| FinnyTable | Yes | Yes (UPDATE: merged) |
| Mate distance | Yes | No |
| Draw randomization | 8-(nodes&15) | No |

---

## 13. Ideas Worth Testing from Koivisto

### Already tested/merged (from SUMMARY.md):
- LMR eval-alpha distance (`+min(2, abs(staticEval-alpha)/350)`) -- listed as Tier 3 #34
- New threats in LMR (`-bitCount(getNewThreats)`) -- listed as Tier 3 #35
- Node count move ordering near root -- listed as Tier 3 #38
- Draw randomization -- listed as Tier 3 #37
- Multi-cut in SE -- tested, -28.5 Elo
- Aspiration fail-high depth reduce -- tested, -353.8 Elo (implementation bug)
- ProbCut QS pre-filter -- already in Tier 1 queue

### New ideas from Koivisto (not yet in SUMMARY.md):

1. **Threat computation framework** -- Compute bilateral threats at every node. This is the backbone of 5+ search features in Koivisto. Computing pawn/minor/rook threats once per node enables RFP/NMP/ProbCut guards and threat history. Medium complexity but very high payoff potential if any 2-3 of the dependent features gain Elo.
   - Est. Elo: +5 to +15 (across all uses)
   - Complexity: Medium (attack computation, but we already compute attacks for SEE)

2. **Threat history replacing butterfly history** -- Instead of `history[color][from][to]`, use `history[color][threatSquare][from*64+to]`. Indexes by opponent's primary threat, so the same move gets different history scores depending on what the opponent is threatening.
   - Already in SUMMARY.md as "threat-aware history" (12 engine consensus). Koivisto's implementation is the cleanest reference.
   - Est. Elo: +5 to +10

3. **LMR behind-null-move tracking** -- Pass the color behind the null move through recursion. Add +1 LMR when same color is behind NMP (their position was already proven strong enough to pass). This is theoretically sound and unique to Koivisto.
   - Est. Elo: +2 to +5
   - Complexity: Low -- add `behindNMP` parameter to pvSearch

4. **LMR targetReached (time-based reduction)** -- When less than 10% of allocated time has been used, add +1 LMR to all moves. The idea: early iterations should be fast/exploratory, later iterations more careful.
   - Est. Elo: +1 to +3
   - Complexity: Trivial -- 1 flag, 1 line in LMR

5. **LMR high-history override** -- When history score exceeds a threshold (512 for quiets, 256 for captures), force LMR to 0 regardless of other adjustments. Acts as a hard guarantee that historically excellent moves are never reduced.
   - Est. Elo: +1 to +3
   - Complexity: Trivial -- 1 condition after LMR computation

6. **SE LMR cancellation** -- When singular extension fires in a child node, cancel the parent's LMR reduction. The idea: if the child found its move was singular (clearly best), the parent should have given it full depth.
   - Est. Elo: +2 to +5
   - Complexity: Medium -- need to pass lmr factor pointer through search

7. **NMP threat guard** -- Disable null move pruning at shallow depth (< 5) when opponent has threats. The opponent's threats mean the position is more tactical.
   - Est. Elo: +2 to +5
   - Complexity: Low (needs threat computation)

8. **ProbCut threat guard** -- Only attempt ProbCut when we have our own threats. The idea: ProbCut assumes we can prove a cutoff with a shallow search, which is more likely when we have active threats.
   - Est. Elo: +2 to +5
   - Complexity: Low (needs threat computation)

9. **TT depth relaxation after null move** -- After a null move, trust TT fail-high entries regardless of depth (add +100 to TT depth for the check). Since NMP already proved position is strong, a TT fail-high in the child is extremely reliable.
   - Est. Elo: +2 to +5
   - Complexity: Trivial -- 1 condition change in TT probe

10. **maxImprovement table for futility** -- Track the maximum eval improvement for each `[from][to]` across the search. Use as per-move futility margin. Moves that historically caused big improvements are harder to futility-prune.
    - Est. Elo: +3 to +8
    - Complexity: Low -- 4096 int table, updated at each node

11. **QSearch good-delta pruning** -- When `SEE + stand_pat > beta + 200`, return beta immediately. If a capture is winning AND we're already above beta, don't bother searching deeper.
    - Previously tested as "QS two-sided delta" at -37.4 Elo. But our test included a bad-delta component. The good-delta alone might work. Koivisto's version is simpler (single condition).
    - Est. Elo: +2 to +5

12. **FinnyTable for NNUE** -- Already in SUMMARY.md (#42). Koivisto's implementation is a good reference: 32 entries per color (16 buckets * 2 king halves), stores per-piece occupancy bitboards.
    - Est. NPS: +5-10%

13. **Mate distance pruning** -- Already in SUMMARY.md queue. Koivisto has standard implementation.
    - Est. Elo: +1 to +3

14. **FORCED_ALL_NODE TT type** -- When at deep all-nodes (depth > 7), if the best move consumed > 50% of effort, store as FORCED_ALL_NODE (value 3). The `& ALL_NODE` check catches both. This gives more information to the replacement policy and TT probe.
    - Est. Elo: +1 to +3
    - Complexity: Trivial -- 1 condition in TT store

### Recommended Testing Order (Koivisto-specific ideas):

1. **Mate distance pruning** -- trivial, universal
2. **TT depth relaxation after null move** -- trivial, theoretically sound
3. **LMR high-history override** -- trivial, Koivisto-proven
4. **LMR behind-NMP tracking** -- low complexity, unique and theoretically sound
5. **maxImprovement table for futility** -- low complexity, adaptive margins
6. **Threat computation + NMP/ProbCut guards** -- medium complexity but enables many features
7. **QSearch good-delta (only)** -- retry with only the beta-side condition
8. **FinnyTable** -- medium complexity, NPS gain
