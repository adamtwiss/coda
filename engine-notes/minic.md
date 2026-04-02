# Minic Chess Engine - Crib Notes

Source: `~/chess/engines/Minic/`
Author: Vivien Clauzon
NNUE: 768->2x384->8->8->8->1 (CReLU input, ReLU hidden, 2 output buckets by game phase, no king buckets)
Lazy SMP: Yes (Stockfish-style skip-blocks)
GoChess-era RR rank: #20 at -15 Elo (93 above GoChess-v5). Coda gauntlet opponent.

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows and PVS. C++ templates for PV/non-PV nodes. Lazy SMP with Stockfish-style skip-block depth distribution across threads.

### Iterative Deepening
- Depth 1 to MAX_DEPTH-6 (MAX_DEPTH=127)
- EBF-based time stop: `usedTime * 1.2 * EBF > currentMoveMs` => stop before next iteration
- EBF computed from depth > 12: `nodes[depth] / nodes[depth-1]`

### Aspiration Windows
- Enabled at depth > 4 (`aspirationMinDepth = 4`)
- Initial delta: `aspirationInit + max(0, aspirationDepthInit - aspirationDepthCoef * depth)` = `5 + max(0, 40 - 3*depth)`
  - depth 5: 5+25=30, depth 10: 5+10=15, depth 14+: 5
- On fail-low: `beta = (alpha + beta) / 2`, then widen alpha by delta
- On fail-high: `--windowDepth` (reduce search depth by 1), widen beta by delta
- Delta growth: `delta += (delta / 4) * exp(1 - gamePhase)` (faster growth in endgames)
- Full-width fallback when delta > `max(128, 4096/depth)`
- Compare to Coda: we use delta=15 from depth 4. Minic's depth-dependent initial delta (5-30) and endgame-aware growth are more sophisticated.

### TT Move Tried First (Before Move Generation)
- If valid TT move exists, it is searched BEFORE generating remaining moves
- Singular extension applied only to TT move
- If TT move fails low by > `failLowRootMargin` (118cp) at root, return alpha - 118 immediately

### Draw Detection
- 50-move rule at 100+ half-moves (pre-loop at 101, post-loop at 100)
- Repetition: template-parameterized `<pvnode>` for 2-fold vs 3-fold
- Draw score: `(-1 + 2 * (nodes % 2))` -- randomized +1/-1 based on node count

### Lazy SMP
- Stockfish-style skip-blocks: 20-entry table of (skipSize, skipPhase) per thread
- Threads skip depths where `((depth + skipPhase[i]) / skipSize[i]) % 2 != 0`
- All threads share TT (single-entry, always-replace)
- Best-thread PV selection: pick thread with deepest completed depth
- Compare to Coda: we use per-thread depth offsets. Minic's skip-block approach is more sophisticated.

---

## 2. NNUE Architecture

### Network: 768 -> 2x384 -> 8 -> 8 -> 8 -> 1 (x2 output buckets)

- **Input features**: 12 piece types x 64 squares = 768 (relative: us/them perspective, no king buckets)
- **Feature transformer**: 768 -> 384, separate white/black accumulators
- **Hidden layers**: Concat(white_384, black_384) = 768 -> fc0(768->8) -> fc1(8->8) -> fc2(16->8) -> fc3(24->1)
- **Skip connections**: Each layer's output is spliced with next layer's output:
  - x1 = ReLU(fc0(x0)) [8]
  - x2 = splice(x1, ReLU(fc1(x1))) [16]
  - x3 = splice(x2, ReLU(fc2(x2))) [24]
  - output = fc3(x3) [1]
- **Output buckets**: 2 buckets based on game phase (gp < 0.45 => bucket 0, else bucket 1)
- **Activation**: CReLU for input (clamp to [0, 512]), ReLU for hidden (clamp to [0, 1])
- **Quantization**: Input layer quantized to int16 (scale=512), hidden layers remain float
- **Output scale**: 600
- **Net size**: ~151 MB (float weights, very large for a chess net)

### Compared to Coda NNUE
- Coda: v5 (768x16->N)x2->1x8 (16 king buckets, 8 output buckets, CReLU/SCReLU/pairwise, Finny tables); v7 adds hidden layers
- Minic: Simple 768->2x384 (no king buckets, 2 output buckets by game phase)
- Minic has wider FT (384 vs our 256/1024) but simpler input features
- Their skip-connection architecture (concatenating all previous layer outputs) is unique
- They still use float for inner layers (only input quantized to int16)
- No king-relative features at all -- significantly simpler feature set

---

## 3. Pruning Techniques

### Reverse Futility Pruning (Static Null Move)
- Uses the `Coeff` system with game-phase dependent thresholds
- Two variants based on whether eval is from search or hash score
- **Max depth: 6** (vs Coda's 7)
- Compare to Coda: we use 70*d (improving) / 100*d (not-improving), depth<=7, with TT quiet-move guard. Minic's game-phase scaling is more complex.

### Threats Pruning (Koivisto-inspired)
- Condition: `!goodThreats[opponent]` -- opponent has no "good threats" (pawn attacks non-pawn, minor attacks rook/queen, rook attacks queen)
- If no opponent threats AND eval > beta + margin => prune
- **This is an additional forward pruning layer beyond RFP, gated on threat absence**
- Compare to Coda: we don't have this.

### Razoring
- Tight margins with threat-based guard: if `!haveThreats[us]` (we have no threats), return alpha immediately without QSearch
- Compare to Coda: we don't currently have razoring in the code (documented in CLAUDE.md but removed from source).

### Null Move Pruning
- Min depth: 2 (vs Coda's 4)
- Condition: `lessZugzwangRisk` (has non-pawn material or mobility > 4) AND `evalScore >= beta + 18` (`nullMoveMargin = 18`)
- Also requires: `evalScore >= stack[halfmoves].eval` (eval not declining)
- Reduction: `5 + depth/6 + min((eval-beta)/147, 7)`
  - Init=5 (vs Coda's 4), depth divisor=6 (vs Coda's 3), dynamic divisor=147 (vs Coda's 200), cap=7 (vs Coda's 3)
  - **Much more aggressive reduction than Coda**
- **Verification search**: When `(!lessZugzwangRisk || depth > 6)` and nullMoveMinPly == 0 -- uses Botvinnik-Markoff style ply restriction
- **NMP threat extraction**: After null move, probes TT for opponent's best reply ("refutation"). If NMP fails low with mate score, sets `mateThreat = true`. Refutation move used in move ordering.
- Compare to Coda: we have R = 4 + d/3 + min((eval-beta)/200, 3), R-1 after captures, NMP score dampening. We don't have NMP threat extraction.

### ProbCut
- Min depth: 10 (vs Coda's 5)
- Beta margin: `beta + 124 - 53*improving` (124/71 vs Coda's 170)
- Max moves to try: 5
- **Two-stage**: QSearch first, then `depth/3` search only if QSearch passes
- Gates: `haveThreats[us]` required (only try ProbCut when we have threats)
- Compare to Coda: we have ProbCut (disabled by default, showed +4 Elo without it). We don't have the QSearch pre-filter stage.

### IIR (Internal Iterative Reduction)
- Condition: `depth >= 3 && !ttHit`
- Reduction: 1 ply
- Compare to Coda: we use depth >= 4, PV or cut nodes only.

### LMP (Late Move Pruning)
- Table-based: `lmpLimit[improving][depth]` for depth 0-10
- Not improving: {0, 2, 3, 5, 9, 13, 18, 25, 34, 45, 55}
- Improving:     {0, 5, 6, 9, 14, 21, 30, 41, 55, 69, 84}
- **Max depth 10** (vs Coda's depth<=8 with 3+d^2)
- Compare to Coda: we use `3+d^2` with improving/failing adjustments.

### Futility Pruning
- Game-phase dependent with improving bonus
- **Max depth 10** (vs Coda's depth<=8)
- Compare to Coda: we use 90+lmrDepth*100 with history adjustment. Minic's is game-phase dependent.

### History Pruning
- Triggers `skipQuiet = true` for ALL remaining quiet moves
- Compare to Coda: we use -1500*depth threshold per-move, depth<=3.

### Capture History Pruning
- Separate from combined history pruning, depth 0-4
- Compare to Coda: we have bad-noisy pruning (losing captures when eval+depth*75<=alpha) but not capture-history-based pruning.

### CMH Pruning (Continuation History)
- Standalone continuation history pruning at max depth 4
- Threshold: CMHMargin = -256

### SEE Pruning
- **Danger-modulated**: SEE thresholds loosen when we have a big attack vs small defense
- Quiet margin: `-(32 + 61 * (nd-1) * (nd + danger/11))` -- quadratic, danger-adjusted
- Capture margin: `-(65 + 180 * (d-1 + danger_adjustment/8))`
- Compare to Coda: we use quiet -20*d^2 depth<=8, capture -d*100 depth<=6. Minic's danger modulation is unique.

---

## 4. Extensions

### Singular Extensions
- Depth guard: `depth >= 8` (vs Coda's `depth >= 8`)
- Singularity beta: `e.s - 2 * depth` (vs Coda's `tt_score - depth`)
- Singularity depth: `depth / 2`
- **Double extension (+2)**: `score < betaC - 4*depth && extensions <= 6`
- **Multi-cut**: If `betaC >= beta`, return `betaC` (fail-soft)
- **Negative extension**: If `e.s >= beta` => extension = -2 + pvNode; if `e.s <= alpha` => extension = -1
- Compare to Coda: we have singular extensions with multi-cut and single negative extension (-1). No double extension.

### Other Extensions (ALL DISABLED)
- Check, in-check, mate threat, queen threat, castling, recapture, good history, CMH extensions -- all disabled
- **Minic relies entirely on singular extensions and LMR adjustments**

---

## 5. LMR (Late Move Reductions)

### Table Initialization
Single table: `LMR[d][m] = log2(d * 1.6) * log2(m) * 0.4`

Uses `log2` (not natural log). Compare to Coda's separate tables:
- Coda quiet: `ln(d) * ln(m) / 1.30`
- Coda capture: `ln(d) * ln(m) / 1.80`

### Application Conditions
- `depth >= 2`
- Applied to move 2+ (not first move or TT move)
- NOT applied to killers or advanced pawn pushes

### Reduction Adjustments (Quiets)
- `+1` for not improving
- `+1` if TT move is a capture (`ttMoveIsCapture`) -- **IMPLEMENTED in Coda**
- `+1` if likely fail-high (`cutNode && evalScore - failHighReductionCoeff > beta`)
- `-min(3, HISTORY_DIV(2 * moveScore))` -- history-based (max 3 plies)
- `-1` for formerPV or ttPV (PV prudence)
- `-1` for known endgame

### Reduction Adjustments (Captures)
- Base LMR from same table as quiets
- `-max(-2, min(2, capHistoryScore))` -- capture history adjustment [-2, +2]
- `+1` if bad capture AND reduction > 1
- `-1` for PV node
- `-1` if improving

### What Minic has that Coda doesn't:
- **Fail-high extra reduction with game-phase-dependent threshold** at cut nodes
- **FormerPV / ttPV flag reducing** (we don't track ttPV flag in TT)
- **Known endgame prudence**
- **Bad capture extra reduction for captures**

---

## 6. Move Ordering

### Score Hierarchy (higher = searched first)
1. **Previous root best**: +15000
2. **TT move**: +14000 (searched before generation)
3. **Promotion captures**: base score for type + promotion scoring
4. **Good captures**: SEE + PST delta/2 + recapture bonus (512) + type bonus
   - Bad captures: SEE < -80 => penalized (drops to ~-7000)
5. **Killer 0**: +1900
6. **Grandparent killer 0** (height-2): +1700
7. **Counter move**: +1500
8. **Killer 1**: +1300
9. **Quiet moves**: `history[c][from][to]/3 + historyP[piece][to]/3 + CMH/3 + PST_delta/2`
   - Escaping NMP threat square: +512 (if SEE_GE >= -80)
10. **Bad captures**: deep in the negative range

### Not Staged (with partial sort)
Minic generates ALL moves upfront but uses partial sorting via `pickNext`/`pickNextLazy`:
- First N moves are selection-sorted (pick best from remaining)
- After a move score drops below `lazySortThreshold` (-491), stop sorting and iterate in order

### History Tables

**Butterfly history**: `history[2][64][64]` -- color, from, to
- Bonus: `4 * min(depth, 32)^2` (HSCORE = SQR(min(d,32)) * 4)
- Gravity: `entry += bonus - HISTORY_DIV(entry * |bonus|)` where `HISTORY_DIV(x) = x >> 10`

**Piece/to history**: `historyP[12][64]` -- piece, to
- Same bonus and gravity as butterfly
- **Unique -- most engines don't have a separate piece/to table**
- Compare to Coda: we have pawn history [pawnHash%512][piece][to] which is different but serves a similar purpose.

**Continuation history**: `counter_history[12][64][12*64]` -- prev_piece, prev_to, curr_piece*64+curr_to
- Only 1-ply lookback (MAX_CMH_PLY = 1)
- Compare to Coda: we use [piece][to][piece][to] continuation history (similar 1-ply structure).

**Capture history**: `historyCap[12][64][6]` -- piece_moved, to_square, victim_piece
- Compare to Coda: we have capture history [piece][to][victim] -- **IMPLEMENTED**.

**Killers**: 2 per ply, checked at height and height-2 (grandparent)

**Counter move**: `counter[64][64]` -- from/to of opponent's last move

### Novel: Grandparent Killer
- Killer from 2 plies earlier scored at +1700 (between killer0 at +1900 and counter at +1500)
- Compare to Coda: we don't have grandparent killers.

### Novel: NMP Refutation Escape Bonus
- Quiet moves that move a piece FROM the NMP refutation's target square get +512 bonus
- Guard: must pass `SEE_GE(move, -80)` to avoid rewarding unsafe escapes
- Compare to Coda: we don't have NMP threat extraction/escape bonus.

---

## 7. QSearch

### Standard QSearch
- Stand-pat with mate distance pruning at top
- TT probe at depth -2 (accepts any TT entry for cutoffs)
- TT move tried first before generation (if capture or in check)
- QSearch depth tracked (`qply`) for recapture-only filtering

### Recapture-Only at Deep QSearch
- At `qply > 5`: only recaptures are searched
- Compare to Coda: we don't limit QS depth this way.

### SEE Pruning
- All captures with SEE < -50 pruned

### Two-Sided Delta Pruning (from Seer)
- **Bad-delta**: SEE <= 20 AND staticScore + 169 < alpha => skip
- **Good-delta**: SEE >= 162 AND staticScore + 47 > beta => return beta
- Gated on `!ttPV`
- Compare to Coda: we use simple delta pruning (240cp). Minic's two-sided approach is more nuanced.

---

## 8. Transposition Table

### Structure
- Single entry per index (no buckets)
- XOR verification: `h ^= _data1 ^= _data2` (3-part XOR, not lockless atomics)
- Additional pseudo-legal validation of TT moves before use

### Replacement Policy
- **Always replace** -- no depth/age/bound priority
- Compare to Coda: we use 5-slot buckets with depth+age replacement. Minic's always-replace is simple but loses deep entries.

### TT Entry Flags
- Stores `ttPV` flag (PV node marker) in bound byte
- Stores `isCheck` and `isInCheck` flags (saves check detection work)

### Near-Miss TT Cutoffs (from Ethereal)
- **Alpha cutoff**: TT depth >= `depth - 1` AND bound == ALPHA AND `e.s + 60 * (depth - 1) <= alpha`
- **Beta cutoff**: TT depth >= `depth - 1` AND bound == BETA AND `e.s - 64 * (depth - 1) >= beta`
- Compare to Coda: **IMPLEMENTED** -- we have TT near-miss cutoffs with 80cp margin.

### TT History Update on Cutoff
- On TT cutoff, updates history for TT move (quiet and capture)
- Compare to Coda: we don't update history on TT cutoffs.

---

## 9. Time Management

### Base Allocation (Sudden Death)
- Computes `nmoves` = estimated remaining moves
- `frac = (remaining - margin) / nmoves`
- `incrBonus = min(0.9 * increment, remaining - margin - frac)`
- maxTime: `min(remaining - margin, targetTime * 7)` (hard limit = 7x soft limit)

### Move Difficulty
- **Forced move**: Only 1 legal move => time / 16
- **Emergency (IID moob)**: Score dropped > 64cp during ID => multiply time by 3x

### Variability Factor
- Tracks score variability during ID: if score changes > 16cp between iterations, `variability *= (1 + depth/100)`, else `variability *= 0.98`
- Applied as sigmoid: `variabilityFactor = 2 / (1 + exp(1 - variability))` -- range [0.5, 2.0]

### EBF-Based Next-Iteration Prediction
- At depth > 12: `usedTime * 1.2 * EBF > currentMoveMs` => stop
- Compare to Coda: we use simple time allocation (soft = timeLeft/movesLeft + 80% inc). No EBF prediction.

---

## 10. Eval Pipeline (NNUE)

1. Game-phase bucket selection (gp < 0.45 => bucket 0, else 1)
2. Game-phase scaling between MG and EG
3. NNUE scaling: `nnueScore * NNUEScaling / 64`
4. Contempt: adds game-phase-weighted contempt
5. **50-move rule scaling**: `score * (1 - fifty^2 / 10000)` -- quadratic decay
   - Compare to Coda: we don't have 50-move rule scaling.
6. Dynamic contempt: `25 * tanh(score/400)` if base contempt >= 0

---

## 11. Unique Techniques

### 1. Danger-Modulated SEE Thresholds
- King danger computed from attack maps even with NNUE
- SEE thresholds adjusted by net attack differential
- Effect: When attacking, accept deeper sacrifices. When defending, prune more conservatively.

### 2. Good Threats Forward Pruning
- Computes "good threats" bitboard: pawn attacks non-pawn, minor attacks rook/queen, rook attacks queen
- Used to gate both RFP-style pruning and razoring

### 3. Fail-High Extra Reduction at Cut Nodes
- When `cutNode && evalScore - margin > beta`, add +1 LMR for quiets
- Margin: game-phase-dependent, tighter at higher depth

### 4. Lazy Sort with Threshold
- Uses selection sort but stops when scores drop below -491
- Remaining moves iterate unsorted

### 5. History Update on Strong Beta Cutoff
- When beta cutoff exceeds `beta + 170`, increase history bonus depth by +1

### 6. TT isCheck / isInCheck Flags
- Saves isPosInCheck calls when replaying TT moves

### 7. Quadratic 50-Move Scaling
- `1 - fifty^2/10000` preserves more eval in normal play but decays faster near the limit

### 8. Root Fail-Low Early Return
- If first root move fails low by > 118cp, immediately returns `alpha - 118`

---

## 12. Notable Differences from Coda

### Things Minic has that Coda doesn't:
1. **Good threats forward pruning** (additional layer beyond RFP)
2. **Danger-modulated SEE thresholds** (loosens SEE when attacking, tightens when defending)
3. **Piece/to history table** (separate `historyP[piece][to]`)
4. **Grandparent killer** (killer[height-2][0] scored at +1700)
5. **QSearch recapture-only at qply > 5**
6. **ProbCut QSearch pre-filter** (two-stage ProbCut)
7. **Fail-high extra reduction at cut nodes** (game-phase dependent)
8. **Root fail-low early return** (alpha - 118)
9. **Quadratic 50-move rule scaling** (fifty^2/10000)
10. **Lazy sort threshold** (stop sorting bad moves)
11. **TT isCheck flag** (avoid recomputing check detection)
12. **Capture history pruning** (separate from combined history pruning)
13. **CMH pruning** (standalone continuation history pruning)
14. **EBF-based iteration prediction** (stop if next iteration would exceed time)
15. **Dynamic aspiration delta** (tighter at higher depth, faster growth in endgames)
16. **Skip-block Lazy SMP** (Stockfish-style thread depth distribution)
17. **NNUE skip-connection architecture** (splice outputs of all previous layers)
18. **NMP threat extraction** with escape bonus in move ordering
19. **History update on TT cutoff**
20. **Double singular extension** (+2 when far below singularity beta)

### Things Coda has that Minic doesn't:
1. **King buckets in NNUE** (HalfKA vs simple 768)
2. **8 output buckets** (vs 2 game-phase buckets)
3. **Multi-slot TT** (5-slot buckets vs single entry)
4. **Lockless TT** (atomic XOR vs simple struct copy)
5. **Multi-source correction history** (pawn + white-NP + black-NP + continuation)
6. **Fail-high score blending** ((score*d+beta)/(d+1))
7. **TT score dampening** ((3*score+beta)/4)
8. **NMP score dampening** ((score*2+beta)/3)
9. **TT near-miss cutoffs** (1-ply-short entries with margin)
10. **Recapture extensions**
11. **Hindsight depth adjustment** (parent reduction + eval trajectory)
12. **Pawn history** [pawnHash%512][piece][to]
13. **Bad noisy pruning** (losing captures when eval+depth*75<=alpha)

### Parameter Comparison Table

| Feature | Minic | Coda |
|---------|-------|------|
| RFP max depth | 6 | 7 |
| RFP margin | game-phase dependent (~82*d) | 70*d imp / 100*d not |
| NMP min depth | 2 | 4 |
| NMP reduction init | 5 | 4 |
| NMP depth divisor | 6 | 3 |
| NMP dynamic divisor | 147 | 200 |
| NMP dynamic cap | 7 | 3 |
| LMR formula | log2(d*1.6)*log2(m)*0.4 | split cap/quiet tables |
| LMR quiet C | ~0.4 (log2) | 1.30 (ln) |
| LMR capture | same table + adjustments | C=1.80 (separate table) |
| LMP max depth | 10 | 8 (3+d^2) |
| Futility margin | game-phase, depth 0-10 | 90+lmrDepth*100 |
| SEE quiet | -32-61*d^2 (danger-mod) | -20*d^2 |
| SEE capture | -65-180*d (danger-mod) | -d*100 |
| Singular depth | >=8 | >=8 |
| Singular beta | tt_val-2*depth | tt_val-depth |
| Aspiration init | 5+max(0,40-3*d) | 15 |
| TT | 1 entry, always-replace | 5-slot, depth+age |
| History bonus | 4*min(d,32)^2 | d^2 capped 1200 |
| Cont-hist plies | 1 | 1 |
| QS SEE threshold | -50 | similar |
| Output buckets | 2 (game phase) | 8 (material) |
| King buckets | 0 | 16 |
| Contempt | dynamic 25*tanh | -10 (fixed) |

---

## 13. Ideas Worth Testing from Minic

### High Priority
1. **ProbCut QSearch pre-filter** -- Two-stage ProbCut: QSearch first, full search only if QSearch confirms. Seen in Minic + Tucano.
2. **NMP threat extraction** -- After null move, probe TT for opponent's best reply. Use in move ordering as escape bonus. Minic does this with SEE guard.
3. **History update on TT cutoff** -- When TT causes early cutoff, update history for the TT move.

### Medium Priority
4. **Good threats forward pruning** -- Additional pruning layer gated on absence of opponent threats.
5. **Fail-high extra reduction at cut nodes** -- When `cutNode && eval - margin > beta`, add +1 LMR.
6. **Double singular extension** -- Extend +2 when score is far below singularity beta.
7. **EBF-based iteration prediction** -- `usedTime * 1.2 * EBF > moveTime => stop`.
8. **Grandparent killer** -- Use killer[height-2] in move ordering between killer0 and counter.

### Lower Priority
9. **Quadratic 50-move scaling** -- `score * (1 - fifty^2/10000)`.
10. **Root fail-low early return** -- If first root move fails low by >118cp, return immediately.
11. **Capture history pruning** -- Prune captures with bad capture history at shallow depth.
12. **History bonus +1 depth on strong beta cutoff** -- When `score > beta + 170`, use `depth+1` for history bonus.
13. **QSearch recapture-only at depth > 5** -- Limit deep QSearch to recaptures only.
