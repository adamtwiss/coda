# Minic Chess Engine - Crib Notes

Source: `~/chess/engines/Minic/`
Author: Vivien Clauzon
NNUE: 768->2x384->8->8->8->1 (CReLU input, ReLU hidden, 2 output buckets by game phase, no king buckets)
Lazy SMP: Yes (Stockfish-style skip-blocks)
RR rank: #20 at -15 Elo (93 above GoChess-v5)

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
- Compare to ours: We use per-thread depth offsets. Minic's skip-block approach is more sophisticated.

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

### Compared to our NNUE
- Ours: v5 (768x16->N)x2->1x8 (16 king buckets, 8 output buckets, CReLU/SCReLU, pairwise mul, Finny tables) **(UPDATE 2026-03-21)**
- Theirs: Simple 768->2x384 (no king buckets, 2 output buckets by game phase)
- They have wider hidden layer (384 vs 256) but simpler input features
- Their skip-connection architecture (concatenating all previous layer outputs) is unique
- They still use float for inner layers (only input quantized to int16)
- No king-relative features at all -- significantly simpler feature set

---

## 3. Pruning Techniques

### Reverse Futility Pruning (Static Null Move)
- Uses the `Coeff` system with game-phase dependent thresholds
- `staticNullMoveCoeff`: init={-15,-32}, bonus={404,112}, slopeD={82,75}, slopeGP={149,21}, depth 0-6
- Two variants based on whether eval is from search or hash score
- Threshold: `(init + d*slopeDepth + gp*slopeGamePhase) * bonus / 256`
- Also uses "improving" flag via the bonus array
- **Max depth: 6** (vs our 8)
- Compare to ours: We use simple 85*d (improving) / 60*d (not-improving). Minic's game-phase scaling is more complex.

### Threats Pruning (Koivisto-inspired)
- Condition: `!goodThreats[opponent]` -- opponent has no "good threats" (pawn attacks non-pawn, minor attacks rook/queen, rook attacks queen)
- If no opponent threats AND eval > beta + margin => prune
- `threatCoeff`: init={33,10}, slopeD={24,18}, depth 0-2
- **This is an additional forward pruning layer beyond RFP, gated on threat absence**
- Compare to ours: We have NMP threat detection but not this kind of forward pruning guard. This was noted in SUMMARY.md as "Opponent-Threats Guard on RFP/NMP" (Tier 2, item #12).

### Razoring
- `razoringCoeff`: init={123,119}, slopeD={69,66}, depth 0-2
- Extra guard: if `!haveThreats[us]` (we have no threats), return alpha immediately without QSearch
- Otherwise falls through to QSearch
- Compare to ours: We use 400+d*100. Minic has tighter margins and threat-based guard.

### Null Move Pruning
- Min depth: 2 (`nullMoveMinDepth = 2`, vs our 3)
- Condition: `lessZugzwangRisk` (has non-pawn material or mobility > 4) AND `evalScore >= beta + 18` (`nullMoveMargin = 18`)
- Also requires: `evalScore >= stack[halfmoves].eval` (eval not declining)
- Reduction: `5 + depth/6 + min((eval-beta)/147, 7)`
  - Init=5 (vs our 3), depth divisor=6 (vs our 3), dynamic divisor=147 (vs our 200), cap=7 (vs our 3)
  - **Much more aggressive reduction than ours**
- **Verification search**: When `(!lessZugzwangRisk || depth > 6)` and nullMoveMinPly == 0
  - Sets `nullMoveMinPly = height + 3*nullDepth/4`
  - Re-searches at nullDepth with restricted NMP
  - Compare to ours: We do simple verification. Minic uses a ply-based restriction (Botvinnik-Markoff style).
- **NMP threat extraction**: After null move, probes TT for opponent's best reply ("refutation")
  - If NMP fails low with mate score, sets `mateThreat = true`
  - Refutation move used in move ordering (bonus for escaping threat square)
  - Compare to ours: We also extract NMP threats (+12.4 Elo merged). Minic does the same.

### ProbCut
- Min depth: 10 (`probCutMinDepth = 10`, vs our ... similar)
- Beta margin: `beta + 124 - 53*improving` (124/71 vs our 170/85)
- Max moves to try: 5
- **Two-stage**: QSearch first, then `depth/3` search only if QSearch passes
  - `scorePC = -qsearch(-betaPC, -betaPC+1, ...)` first
  - If scorePC >= betaPC: `scorePC = -pvs<false>(-betaPC, -betaPC+1, ..., depth/3, ...)`
- Gates: `haveThreats[us]` required (only try ProbCut when we have threats)
- Also gated on: `(cutNode || staticScore >= betaPC)` -- skip ProbCut at non-cut nodes when eval is low
- Compare to ours: We have ProbCut but without the QSearch pre-filter stage. The two-stage ProbCut is item #6 in Tier 1 of SUMMARY.md.

### IIR (Internal Iterative Reduction)
- Condition: `depth >= 3 && !ttHit`
- Reduction: 1 ply
- Compare to ours: Similar, we use depth >= 4.

### LMP (Late Move Pruning)
- Table-based: `lmpLimit[improving][depth]` for depth 0-10
- Not improving: {0, 2, 3, 5, 9, 13, 18, 25, 34, 45, 55}
- Improving:     {0, 5, 6, 9, 14, 21, 30, 41, 55, 69, 84}
- **Max depth 10** (vs our implicit depth limit from 3+d^2)
- Triggers `skipQuiet = true` which skips ALL remaining quiet moves
- Compare to ours: We use `3+d^2` (+50% improving). Minic has a much wider depth range (up to 10 vs ~6 effective).

### Futility Pruning
- `futilityPruningCoeff`: init={7,6}, slopeD={161,170}, depth 0-10
- Threshold: `alpha - ((7 + d*161 + gp*(-3)) * bonus[improving] / 256)`
  - Not improving bonus = 155/256 ~= 0.61
  - Improving bonus = 605/256 ~= 2.36
- **Max depth 10** (vs our futility at depth limit ~5-6)
- Sets `skipQuiet = true` and also `skipCap = true` when futility + bad capture
- Compare to ours: We use 80+d*80 (recently tightened from 100+d*100, +33.6 Elo). Minic's is game-phase dependent.

### History Pruning
- `historyPruningCoeff`: depth 0-2 (not-improving) or 0-1 (improving)
- Triggers `skipQuiet = true` for ALL remaining quiet moves
- Compare to ours: We use -2000*depth threshold per-move.

### Capture History Pruning
- `captureHistoryPruningCoeff`: depth 0-4 (not-improving) or 0-3 (improving)
- Prunes individual captures (continue, not break)
- Compare to ours: We don't have separate capture history pruning.

### CMH Pruning (Continuation History)
- Separate from combined history pruning
- Max depth: 4 (both improving and not-improving)
- Threshold: CMHMargin = -256
- Prunes individual quiet moves when continuation history is bad

### SEE Pruning (Captures)
- For bad captures (SEE < badCapLimit = -80):
  - Margin: `-(seeCaptureInit + seeCaptureFactor * (depth - 1 + max(0, dangerGoodAttack - dangerUnderAttack)/seeCapDangerDivisor))`
  - = `-(65 + 180 * (depth-1 + danger_adjustment/8))`
  - **Danger-modulated**: The SEE threshold loosens when we have a big attack vs small defense
- Sets `skipCap = true` on first pruned bad capture (skips all remaining bad caps)

### SEE Pruning (Quiets)
- Applied AFTER LMR reduction computation (uses nextDepth, not depth)
- Margin: `-(seeQuietInit + seeQuietFactor * (nextDepth-1) * (nextDepth + danger_adjustment/11))`
- = `-(32 + 61 * (nd-1) * (nd + danger/11))`
- **Also danger-modulated**
- Compare to ours: We use simple -d*20 threshold. Minic's quadratic formula is much more aggressive at depth.

### Danger-Based Pruning Adaptation
- Computes `dangerFactor = (danger[White] + danger[Black]) / 183`
- Three thresholds: `dangerLimitPruning = 16`, `dangerLimitForwardPruning = 16`, `dangerLimitReduction = 16`
- Currently NOT used for gating pruning (all three checks are tracked but not applied)
- However, danger IS used to modulate SEE thresholds for captures and quiets (see above)
- Compare to ours: We don't have danger-based pruning. This is a unique feature, noted as Tier 3 #36 in SUMMARY.md.

---

## 4. Extensions

### Singular Extensions
- Depth guard: `depth >= 8` (`singularExtensionDepth = 8`)
- Condition: `!rootnode && !mate_score && (exact || beta bound) && e.d >= depth - 5`
- Singularity beta: `e.s - 2 * depth`
- Singularity depth: `depth / 2`
- If score < betaC: **+1 extension** (singular)
- **Double extension (+2)**: `score < betaC - 4*depth && extensions <= 6`
  - Triple extension commented out (`betaC - 8*depth`)
- **Multi-cut**: If `betaC >= beta`, return `betaC` (fail-soft)
- **Negative extension**: If `e.s >= beta` => extension = -2 + pvNode
  - Also: if `e.s <= alpha` => extension = -1
- Compare to ours: Our SE was tested and failed badly. Minic's singularity beta `2*depth` is tighter than our `3*depth`. The multi-cut return and negative extensions are key.

### Botvinnik-Markoff Extension
- Tracks refutation (NMP threat) across plies
- If same refutation at height and height-2 (persistent threat), or if current threat captures on same square as previous threat => eligible for extension
- Currently disabled (`enableBMExtension = false`) but infrastructure exists

### Other Extensions (ALL DISABLED)
- Check extension: disabled
- In-check extension: disabled
- Mate threat extension: disabled
- Queen threat extension: disabled
- Castling extension: disabled
- Recapture extension: disabled
- Good history extension: disabled (threshold 512)
- CMH extension: disabled
- **All extensions except singular are disabled** -- Minic relies entirely on singular extensions and LMR adjustments

---

## 5. LMR (Late Move Reductions)

### Table Initialization
Single table: `LMR[d][m] = log2(d * 1.6) * log2(m) * 0.4`

Uses `log2` (not natural log). Compare to our separate tables:
- Our quiet: `1.50 + ln(d) * ln(m) / 1.50`
- Our capture: `1.50 + ln(d) * ln(m) / 1.80`
- Minic's base values are somewhat different due to the log2 vs ln scaling (log2(x) = ln(x)/0.693), but the overall shape is similar.

### Application Conditions
- `depth >= 2` (`lmrMinDepth = 2`)
- Applied to move 2+ (not first move or TT move)
- NOT applied to killers or advanced pawn pushes

### Reduction Adjustments (Quiets -- all enabled)
- `+1` for not improving
- `+1` if TT move is a capture (`ttMoveIsCapture`)
- `+1` if likely fail-high (`cutNode && evalScore - failHighReductionCoeff > beta`)
  - `failHighReductionCoeff`: init={119,139}, slopeD={-54,7} -- loosens with depth
- `-min(3, HISTORY_DIV(2 * moveScore))` -- history-based (max 3 plies reduction/extension)
- `-1` for formerPV or ttPV (PV prudence)
- `-1` for known endgame

### Reduction Adjustments (Captures -- all enabled)
- Base LMR from same table as quiets
- `-max(-2, min(2, capHistoryScore))` -- capture history adjustment [-2, +2]
- `+1` if bad capture AND reduction > 1
- `-1` for PV node
- `-1` if improving

### Clamp
- `extension - reduction` cannot be > 0 (never extend more than reduce, except for first move without TT hit)
- `depth - 1 + extension - reduction` cannot be <= 0

### What Minic has that we don't:
- **TT move is capture => +1 reduction** (we have this as TT noisy detection, MERGED +34.4 Elo)
- **Fail-high reduction with game-phase-dependent threshold** (noted as SUMMARY.md Tier 2 #13)
- **FormerPV / ttPV flag reducing** (we don't track ttPV flag in TT)
- **Known endgame prudence**
- **Bad capture extra reduction for captures**

---

## 6. Move Ordering

### Score Hierarchy (higher = searched first)
1. **Previous root best**: +15000
2. **TT move**: +14000 (searched before generation, not just scored)
3. **Promotion captures**: base score for type + promotion scoring
4. **Good captures**: SEE + PST delta/2 + recapture bonus (512) + type bonus
   - Bad captures: SEE < -80 => penalized by `-2 * MoveScoring[T_capture]` (drops to ~-7000)
   - Without SEE (in check or QS): recapture bonus + MVVLVA*2 + capHistory/4
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
- This is a hybrid: staged-like for top moves, unsorted for bad moves

### History Tables

**Butterfly history**: `history[2][64][64]` -- color, from, to
- Bonus: `4 * min(depth, 32)^2` (HSCORE = SQR(min(d,32)) * 4)
- Gravity: `entry += bonus - HISTORY_DIV(entry * |bonus|)` where `HISTORY_DIV(x) = x >> 10`
- Max: HISTORY_MAX = 1024

**Piece-to history**: `historyP[12][64]` -- piece, to
- Same bonus and gravity as butterfly
- **This is unique -- most engines don't have a separate piece/to table**
- Used with weight = 1/3 in move scoring

**Continuation history**: `counter_history[12][64][12*64]` -- prev_piece, prev_to, curr_piece*64+curr_to
- Only 1-ply lookback (MAX_CMH_PLY = 1)
- Initialized to -1 (slight bias toward non-continuation moves)
- Compare to ours: We use 1-ply and 2-ply continuation history. Minic only uses 1-ply but has the extra piece/to table.

**Capture history**: `historyCap[12][64][6]` -- piece_moved, to_square, victim_piece
- Same bonus and gravity formula

**Killers**: 2 per ply, checked at height and height-2 (grandparent)

**Counter move**: `counter[64][64]` -- from/to of opponent's last move

### Novel: Grandparent Killer
- Killer from 2 plies earlier scored at +1700 (between killer0 at +1900 and counter at +1500)
- This is Tier 3 #29 in SUMMARY.md

### Novel: NMP Refutation Escape Bonus
- Quiet moves that move a piece FROM the NMP refutation's target square get +512 bonus
- Guard: must pass `SEE_GE(move, -80)` to avoid rewarding unsafe escapes
- Compare to ours: We have NMP threat detection with +8000 escape bonus (MERGED +12.4 Elo). Minic's is similar but with SEE guard and lower bonus.

### Novel: Piece/To History Table
- Separate `historyP[piece][to]` table alongside standard butterfly `history[color][from][to]`
- Both contribute equally (both divided by 3) to quiet move scoring
- This is Tier 2 #21 in SUMMARY.md

---

## 7. QSearch

### Standard QSearch
- Stand-pat with mate distance pruning at top
- TT probe at depth -2 (accepts any TT entry for cutoffs)
- TT move tried first before generation (if it is a capture or we are in check)
- QSearch depth tracked (`qply`) for recapture-only filtering

### Recapture-Only at Deep QSearch
- At `qply > 5`: only recaptures are searched (move must target the same square as opponent's last capture)
- This is Tier 3 #31 in SUMMARY.md

### SEE Pruning
- All captures with SEE < -50 (`seeQThreshold = -50`) are pruned
- Compare to ours: We use SEE filtering as well.

### Two-Sided Delta Pruning (from Seer)
- **Bad-delta**: SEE <= 20 AND staticScore + 169 < alpha => skip (neutral/losing capture won't help)
  - `deltaBadSEEThreshold = 20`, `deltaBadMargin = 169`
- **Good-delta**: SEE >= 162 AND staticScore + 47 > beta => return beta (winning capture already enough)
  - `deltaGoodSEEThreshold = 162`, `deltaGoodMargin = 47`
- Gated on `!ttPV` (don't apply on PV nodes)
- Compare to ours: We tested this and it was -37.4 Elo (good-delta too aggressive). Minic's thresholds are quite different and the ttPV guard may be important.

### QS Futility (DISABLED)
- `doQFutility = false` -- the QSearch futility margin exists but is turned off
- When enabled: margin = 1024cp (very wide)

---

## 8. Transposition Table

### Structure
- Single entry per index (no buckets)
- Index: `h & (ttSize - 1)` (power-of-2 masking)
- 12 bytes per entry: hash(32) + score+eval(32) + move+bound+depth(32)
- XOR verification: `h ^= _data1 ^= _data2` (3-part XOR, not lockless atomics)
- Additional pseudo-legal validation of TT moves before use

### Replacement Policy
- **Always replace** -- no depth/age/bound priority. This heavily favors leaf nodes.
- Compare to ours: We use 4-slot buckets with depth+age replacement. Minic's always-replace is simple but loses deep entries.

### Generation/Aging
- 3-bit generation counter (0-7) encoded in the bound byte
- Incremented each search

### TT Entry Flags
- Stores `ttPV` flag (PV node marker) in bound byte
- Stores `isCheck` flag (whether TT move gives check)
- Stores `isInCheck` flag (whether position was in check)
- These extra flags save work: check detection from TT move avoids isPosInCheck call

### Near-Miss TT Cutoffs (from Ethereal)
- **Alpha cutoff**: If TT depth >= `depth - 1` AND bound == ALPHA AND `e.s + 60 * (depth - ttAlphaCutDepth) <= alpha`
  - `ttAlphaCutDepth = 1`, `ttAlphaCutMargin = 60`
- **Beta cutoff**: If TT depth >= `depth - 1` AND bound == BETA AND `e.s - 64 * (depth - ttBetaCutDepth) >= beta`
  - `ttBetaCutDepth = 1`, `ttBetaCutMargin = 64`
- Compare to ours: We MERGED this as "TT near-miss cutoffs" at +21.7 Elo. The idea originated from Minic/Ethereal.

### TT History Update on Cutoff
- On TT cutoff (non-PV, returning hash score), updates history for TT move:
  - Quiet TT moves: updates killers, counter, butterfly + CMH
  - Capture TT moves: updates capture history
- Compare to ours: We don't update history on TT cutoffs.

---

## 9. Time Management

### Base Allocation (Sudden Death)
- Computes `nmoves` = estimated remaining moves:
  - Risky (increment < 20ms): nmoves = 28
  - Normal: nmoves = 16 - correction (0-15 based on game length and phase)
- `frac = (remaining - margin) / nmoves`
- `incrBonus = min(0.9 * increment, remaining - margin - frac)` (non-risky only)
- `ms = (frac + incrBonus) * ponderingCorrection`
- Margin: `max(min(3000, 0.01 * remaining), moveOverhead)`
- maxTime: `min(remaining - margin, targetTime * 7)` (hard limit = 7x soft limit)

### Move Difficulty
- **Forced move**: Only 1 legal move => time / 16
- **Standard**: Normal allocation
- **Emergency (IID moob)**: Score dropped > 64cp during ID => multiply time by 3x (capped at 1/5 of remaining)

### Position Evolution (Game History)
- Tracks "booming" (rapid score increase) and "moobing" (rapid score decrease) patterns from game history
- Booming attack/defence: 1x multiplier (no change)
- Moobing attack/defence: 2x multiplier

### Variability Factor
- Tracks score variability during ID: if score changes > 16cp between iterations, `variability *= (1 + depth/100)`, else `variability *= 0.98`
- Applied as sigmoid: `variabilityFactor = 2 / (1 + exp(1 - variability))` -- range [0.5, 2.0]
- Multiplied into time allocation

### EBF-Based Next-Iteration Prediction
- At depth > 12: `usedTime * 1.2 * EBF > currentMoveMs` => stop
- EBF = `nodes[depth] / nodes[depth-1]`
- Compare to ours: We use instability factor (200). Minic's variability factor is similar but continuous (sigmoid). The EBF prediction is unique.

---

## 10. Eval Pipeline (NNUE)

1. NNUE propagation with game-phase bucket selection (gp < 0.45 => bucket 0, else 1)
2. Game-phase scaling: `ScaleScore({nnueScore, nnueScore}, gp, scalingFactor)` -- scales between MG and EG
3. NNUE scaling: `nnueScore * NNUEScaling / 64` (configurable UCI option)
4. Contempt: adds `ScaleScore(contempt, gp)` -- game-phase weighted contempt
5. **50-move rule scaling**: `score * (1 - fifty^2 / 10000)` -- quadratic decay
   - At fifty=50: factor = 0.75, at fifty=70: factor = 0.51, at fifty=100: factor = 0
   - Compare to ours: We tested linear 50-move scaling (H0, -3.0 Elo). Minic uses quadratic which preserves eval better in normal play.
6. Clamp to valid score range
7. Dynamic contempt: `25 * tanh(score/400)` added if base contempt >= 0

---

## 11. Unique Techniques

### 1. Danger-Modulated SEE Thresholds
- King danger computed from attack maps even with NNUE (partial HCE computation)
- `dangerGoodAttack` = opponent king danger / 183 (our attack strength)
- `dangerUnderAttack` = our king danger / 183 (how attacked we are)
- SEE capture threshold adjusted by `max(0, dangerGoodAttack - dangerUnderAttack) / 8`
- SEE quiet threshold adjusted by `max(0, dangerGoodAttack - dangerUnderAttack) / 11`
- Effect: When attacking, accept deeper sacrifices. When defending, prune more conservatively.

### 2. Good Threats Forward Pruning
- Computes "good threats" bitboard: pawn attacks non-pawn, minor attacks rook/queen, rook attacks queen
- Used to gate both RFP-style pruning (threats pruning) and razoring (skip QSearch if no threats)
- Compare to ours: Item #12 in Tier 2 of SUMMARY.md.

### 3. Fail-High Extra Reduction at Cut Nodes
- `failHighReductionCoeff`: when `cutNode && evalScore - margin > beta`, add +1 LMR reduction
- Margin: `(119 + d*(-54)) * bonus[improving] / 256` -- tighter at higher depth
- This is item #13 in Tier 2 of SUMMARY.md.

### 4. Lazy Sort with Threshold
- After scoring moves, uses selection sort (`pickNext`) but stops sorting when move scores drop below -491
- Remaining moves iterate unsorted (saves ~5-10% in sort overhead for deep nodes with many moves)
- QSearch has separate threshold (-512)

### 5. History Update on Strong Beta Cutoff
- When beta cutoff score exceeds `beta + betaMarginDynamicHistory` (170cp), increase history bonus depth by +1
- Applied to both quiet and capture history
- Compare to ours: This is item #14f from Alexandria ("Eval-Based History Depth Bonus") in SUMMARY.md.

### 6. TT isCheck / isInCheck Flags
- TT entry stores whether the TT move gives check and whether the position was in check
- Saves isPosInCheck calls when replaying TT moves
- Small but measurable speedup

### 7. Quadratic 50-Move Scaling
- `1 - fifty^2/10000` instead of linear `(200-fifty)/200`
- Preserves more eval in normal play (factor > 0.99 until move 10) but decays faster near the limit

### 8. Root Fail-Low Early Return
- If TT move or first move at root fails low by > 118cp, immediately returns `alpha - 118`
- Forces aspiration window re-search without wasting time on remaining root moves

---

## 12. Notable Differences from GoChess

### Things Minic has that we don't:
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
14. **Always-replace TT** (single entry, no buckets)
15. **EBF-based iteration prediction** (stop if next iteration would exceed time)
16. **Dynamic aspiration delta** (tighter at higher depth, faster growth in endgames)
17. **Skip-block Lazy SMP** (Stockfish-style thread depth distribution)
18. **NNUE skip-connection architecture** (splice outputs of all previous layers)

### Things we have that Minic doesn't:
1. **King buckets in NNUE** (HalfKA vs simple 768)
2. **8 output buckets** (vs 2 game-phase buckets)
3. **Multi-slot TT** (4-slot buckets vs single entry)
4. **Lockless TT** (atomic XOR vs simple struct copy)
5. **Correction history** (pawn hash correction)
6. **Alpha-reduce** (depth-1 after alpha raise)
7. **QS beta blending**
8. **TT score dampening** ((3*score+beta)/4)
9. **Fail-high score blending** ((score*d+beta)/(d+1))
10. **2-ply continuation history** (Minic only uses 1-ply)

### Parameter Comparison Table

| Feature | Minic | GoChess |
|---------|-------|---------|
| RFP max depth | 6 | 8 |
| RFP margin | game-phase dependent (~82*d) | 85*d imp / 60*d not |
| Razoring depth | <=2 | <=3 |
| NMP min depth | 2 | 3 |
| NMP reduction init | 5 | 3 |
| NMP depth divisor | 6 | 3 |
| NMP dynamic divisor | 147 | 200 |
| NMP dynamic cap | 7 | 3 |
| LMR formula | log2(d*1.6)*log2(m)*0.4 | split cap/quiet tables |
| LMP max depth | 10 | ~6 (3+d^2) |
| Futility depth | 0-10 | 0-5 (80+d*80) |
| SEE quiet | -32-61*d^2 (danger-mod) | similar |
| SEE capture | -65-180*d (danger-mod) | similar |
| Singular depth | >=8 | >=10 (disabled) |
| Singular beta | tt_val-2*depth | tt_val-3*depth |
| Aspiration init | 5+max(0,40-3*d) | 15 |
| TT | 1 entry, always-replace | 4-slot, depth+age |
| TT near-miss | 1 ply, 60/64cp margin | 1 ply, 64cp (merged) |
| History HSCORE | 4*min(d,32)^2 | similar |
| History gravity | >> 10 | /divisor |
| Cont-hist plies | 1 | 2 |
| QS SEE threshold | -50 | similar |
| Output buckets | 2 (game phase) | 8 (material) |
| King buckets | 0 | 16 |
| Time hard/soft ratio | 7:1 | similar |

---

## 13. Ideas Worth Testing from Minic

### Already Merged (from Minic)
- **TT near-miss cutoffs**: +21.7 Elo (ttAlphaCutDepth=1, margin 60-64)
- **NMP threat extraction**: +12.4 Elo (escape bonus for threatened pieces)
- **TT noisy detection for LMR**: +34.4 Elo (ttMoveIsCapture => +1 reduction)

### Already Tested (from Minic)
- **QS two-sided delta**: -37.4 Elo (good-delta too aggressive). Minic's thresholds differ significantly (SEE 20/162, margin 169/47) and uses ttPV guard. Consider re-test with Minic's exact params.
- **Fail-high depth reduction in aspiration**: -353.8 Elo (implementation bug). Minic uses `--windowDepth` in aspiration loop (inner variable only). Retry candidate.
- **QS recapture-only at depth > 5**: Not tested yet.

### New Ideas to Test

1. **ProbCut QSearch pre-filter** -- Two-stage ProbCut: QSearch first, full search only if QSearch confirms cutoff. Already in SUMMARY.md Tier 1 #6 (2 engines: Minic + Tucano + Alexandria).
   - Complexity: Medium. Add QSearch call before existing ProbCut search.
   - Est. Elo: +3 to +8.

2. **Good threats forward pruning** -- Additional RFP-like layer gated on absence of opponent "good threats" (pawn attacks non-pawn, minor attacks rook/queen, rook attacks queen). Already in SUMMARY.md Tier 2 #12.
   - Complexity: Medium. Need attack map computation.
   - Est. Elo: +3 to +8.

3. **Fail-high extra reduction at cut nodes** -- When `cutNode && eval - margin > beta`, add +1 LMR for quiets. Already in SUMMARY.md Tier 2 #13.
   - Complexity: Low. One condition in LMR.
   - Est. Elo: +2 to +4.

4. **Piece/to history table** -- Separate `historyP[piece][to]` alongside butterfly `history[color][from][to]`. Already in SUMMARY.md Tier 2 #21.
   - Complexity: Low. Small table (12x64), same update logic.
   - Est. Elo: +2 to +4.

5. **Grandparent killer** -- Use killer[height-2][0] in move ordering between killer0 and counter. Already in SUMMARY.md Tier 3 #29.
   - Complexity: Trivial. 2 comparisons.
   - Est. Elo: +1 to +3.

6. **Quadratic 50-move scaling** -- `score * (1 - fifty^2/10000)` instead of linear. Our linear test was H0. The quadratic form preserves eval better in normal play.
   - Complexity: Trivial. 1 line.
   - Est. Elo: +1 to +3.

7. **NMP R=5 + depth/6** -- Much more aggressive NMP than our R=3+d/3. Worth testing intermediate values.
   - Complexity: Trivial. Parameter change.
   - Caution: May need verification search adjustment.

8. **EBF-based iteration prediction** -- `usedTime * 1.2 * EBF > moveTime => stop`. Prevents starting iterations that can't finish.
   - Complexity: Low. Track EBF from node counts, one check per iteration.
   - Est. Elo: +1 to +3.

9. **Root fail-low early return** -- If first root move fails low by large margin (118cp), return immediately.
   - Complexity: Low. Already have root move scoring.
   - Est. Elo: +1 to +3.

10. **History update on TT cutoff** -- When TT entry causes early cutoff, update history for the TT move. Both quiet and capture history.
    - Complexity: Low. Add update calls in TT cutoff path.
    - Est. Elo: +2 to +5.

11. **Capture history pruning** -- Prune captures with very bad capture history at shallow depth (0-4).
    - Complexity: Low.
    - Est. Elo: +1 to +3.

12. **History bonus +1 depth on strong beta cutoff** -- When `score > beta + 170`, use `depth+1` for history bonus. Same as Alexandria's eval-based history depth bonus.
    - Already in SUMMARY.md Tier 2 #14f (2 engines: Minic + Alexandria).
    - Complexity: Trivial.
