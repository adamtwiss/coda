# Rodent III -- Crib Notes

Source: `~/chess/engines/Rodent_III/sources/src/`
Derived from Sungorus 1.4 by Pablo Vazquez. Author: Pawel Koziol.

---

## 1. Search Framework

- Iterative deepening with aspiration windows (Senpai-style)
- PVS (principal variation search)
- Separate `SearchRoot()` and `Search()` functions
- Quiescence: 3-layer design -- `QuiesceChecks()` -> `QuiesceFlee()` -> `Quiesce()`
- Lazy SMP: odd-numbered threads search at depth+1; threads skip depths with >50% coverage; lagging threads skip iterations

### Aspiration Windows (`Widen()`)
- Kicks in at depth > 6
- Initial margin: 8, doubles each iteration (8, 16, 32, 64, 128, 256)
- Falls back to full window on failure or mate score
- **Score jump detection**: if fail-low/fail-high with margin > 50, sets `Glob.scoreJump = true` (used by time management)
- Score increase detection only for single-thread mode

### Mate Distance Pruning
- Present in `Search()` (not root)
- Standard MDP: prune if alpha >= MATE-ply or beta <= -(MATE-ply)

### Early Mate Exit
- At root: if mate score found, compute `max_mate_depth = (MATE - |score| + 1) * 4/3`
- Stop deepening if `max_mate_depth <= rootDepth`

### Single Root Move Optimization
- If only one legal root move (`mFlRootChoice == false`) and depth >= 8, stop searching

---

## 2. Pruning Techniques

### 2a. Static Null Move Pruning (Reverse Futility Pruning)
- **Depth guard**: depth <= 7 (`mscSnpDepth` is 3 but the actual check in code is `depth <= 7`)
- **Condition**: non-PV, not in check, no mate scores, has non-pawn material (`MayNull()`), not after null move
- **Skill guard**: searchSkill > 7
- **Margin**: `eval - (175 - 50*improving) * depth > beta`
  - Improving: `175 * depth` margin (e.g. d1=175, d4=700, d7=1225)
  - Not improving: `125 * depth` (e.g. d1=125, d4=500, d7=875)
- **Returns**: the adjusted eval score (not just beta)

### 2b. Null Move Pruning
- **Depth guard**: depth > 1
- **Skill guard**: searchSkill > 1
- **Conditions**: non-PV, not in check, not after null move, has non-pawn material, eval >= beta
- **Reduction formula** (`SetNullReductionDepth()`):
  ```
  R = (823 + 67*depth) / 256 + min(3, (eval-beta)/200)
  ```
  - At depth 8: base R = (823+536)/256 = 5.3 -> 5, plus eval-dependent 0-3
  - At depth 12: base R = (823+804)/256 = 6.3 -> 6
  - At depth 4: base R = (823+268)/256 = 4.2 -> 4
- **TT-based null move avoidance**: before doing null move, probes TT at the reduced depth. If TT score < beta, skips null move entirely (goto avoidNull). This saves the null move search when it is likely to fail.
- **Verification search**: if reduced depth > 6 and searchSkill > 9, re-searches at `newDepth - 5` with `wasNull=true`. Only accepts null cutoff if verification also >= beta.
- **Mate score clamping**: if null move score > MAX_EVAL, clamp to beta (Stockfish-style)
- **Null refutation tracking**: after null move, retrieves TT to find the move that refuted null. Its target square (`refutationSquare`) is passed to move ordering to prioritize escaping that piece.

### 2c. Razoring (Toga II style)
- **Depth guard**: depth <= 4 (`mscRazorDepth = 4`)
- **Skill guard**: searchSkill > 3
- **Conditions**: non-PV, not in check, no TT move, not after null move, no pawns on 7th rank
- **Margins**: `{0, 300, 360, 420, 480}` (indexed by depth)
- **Logic**: if `eval < beta - margin[depth]`, do QS. If QS score also < threshold, return it. Otherwise continue to normal search.

### 2d. Futility Pruning
- **Depth guard**: depth <= 6 (`mscFutDepth = 6`)
- **Skill guard**: searchSkill > 4
- **Margins**: `{0, 100, 150, 200, 250, 300, 400}` (indexed by depth)
- **Conditions**: non-PV, not in check, quiet move, first quiet hasn't been tried yet when flag is set
- **Logic**: sets `flagFutility = true` if `eval + futMargin[depth] < beta`. Then prunes subsequent quiet moves that don't give check and have history < histLimit and are MV_NORMAL type. Requires movesTried > 1.

### 2e. Late Move Pruning (LMP)
- **Depth guard**: depth <= 10
- **Skill guard**: searchSkill > 5
- **Conditions**: non-PV, not in check, quiet move, move doesn't give check, history < histLimit
- **Table** (indexed by [improving][depth]):
  ```
  Not improving: {0, 3, 4, 6, 10, 15, 21, 28, 36, 45, 55}
  Improving:     {0, 5, 6, 9, 15, 23, 32, 42, 54, 68, 83}
  ```
  - e.g. depth 4, not improving: prune after 10 quiet moves
  - e.g. depth 4, improving: prune after 15 quiet moves

### 2f. SEE Pruning of Bad Captures
- **Depth guard**: depth <= 3
- **Conditions**: non-PV, bad capture (MV_BADCAPT), not in check
- **Threshold**: if `SEE_score > 150 * depth`, prune (note: SEE score is negative for bad captures, so `> 150*depth` means "not too negative" -- actually this prunes captures where SEE is *positive* but classified as bad??)
  - Wait -- bad captures have negative SEE. `150 * depth` is positive. So `moveSEEscore > 150*depth` would never be true for bad captures. This looks like it might prune if the bad capture is less bad than expected. Actually, re-reading: `BadCapture()` returns `Swap() < 0`, so bad captures have negative SEE. The condition `moveSEEscore > 150*depth` would mean the SEE is positive -- which contradicts it being a bad capture. This code path may be effectively dead.
  - **Correction**: Looking more carefully, `moveSEEscore` is set to `p->Swap()` only when `moveType == MV_BADCAPT`. The Swap values are actual SEE scores. For bad captures, these are negative. `150*depth` is always positive (depth 1-3 = 150-450). So this condition `moveSEEscore > 150*depth` is always false for bad captures. **This pruning code is dead/no-op.**

### 2g. No ProbCut
- Rodent III does **not** have ProbCut.

---

## 3. Late Move Reductions (LMR)

### 3a. LMR Table (Stockfish-derived)
- **Formula**: `r = 0.33 + log(min(moveCount,63)) * log(min(depth,63)) / 2.00`
- Two tables: `msLmrSize[0]` for zero-window, `msLmrSize[1]` for PV (= zero-window - 1)
- Capped at `depth - 1`

### 3b. LMR for Quiet Moves
- **Depth guard**: depth > 2
- **Move count guard**: movesTried > 3
- **Conditions**: not in check (either side), quiet move (MV_NORMAL), not a castle, history < histLimit, table value > 0
- **Reduction adjustments**:
  1. **Bad history** (history < 0): reduction += 1
  2. **Very bad history** (non-PV, history < -MAX_HIST/2 = -16384): reduction += 1 (additional)
  3. **Not improving** (non-PV, !improving): reduction += 1
  4. Cap: `reduction >= newDepth` -> `reduction = newDepth - 1`
- **Re-search**: if reduced search returns > alpha, re-searches at full depth
- Example at depth 10, move 8, non-PV, bad history, not improving:
  - Base: `0.33 + log(8)*log(10)/2 = 0.33 + 2.08*2.30/2 = 0.33 + 2.39 = 2.72` -> 2
  - +1 (bad history) + 1 (very bad history) + 1 (not improving) = 5 total

### 3c. LMR for Bad Captures
- **Depth guard**: depth > 2
- **Skill guard**: searchSkill > 8
- **Move count guard**: movesTried > 6
- **Conditions**: non-PV, bad capture, not in check (either side), no mate scores in window
- **Fixed reduction**: 1 ply
- This is unusual -- most engines don't separately reduce bad captures

---

## 4. Extensions

### 4a. Check Extension
- +1 ply if the move gives check
- **Only in PV nodes OR depth < 8** (restricted in deep non-PV)

### 4b. Recapture Extension (Search only, not Root)
- +1 ply in PV nodes if the move's target square == last capture square
- This requires passing `lastCaptSquare` through the search

### 4c. Pawn to 7th Rank Extension
- +1 ply in PV nodes at depth < 6 if a pawn lands on rank 2 or rank 7
- Only at shallow depth to avoid explosion

### 4d. Singular Extension (Senpai-style)
- **PV nodes only**, depth > 5
- Probes TT at depth-4 for a LOWER bound move
- Verification: searches at depth-4 with window `[-singScore-50, -singScore-49]`
- If verification score <= that alpha, extends by +1 ply
- Allows stacking with other extensions (the `flagExtended == false` check is commented out in Search)

---

## 5. Move Ordering

### 5a. Stages (8 phases)
1. **TT move** (phase 0)
2. **Good captures** (phase 1-2): generated, scored by MVV-LVA, bad captures saved aside
3. **Killer 1** (phase 3)
4. **Killer 2** (phase 4)
5. **Refutation move** (phase 5): counter-move indexed by [from][to] of the previous move
6. **Quiet moves** (phase 6-7): generated, scored by history
7. **Bad captures** (phase 8): returned in insertion order (no re-sorting)

### 5b. Capture Scoring (MVV-LVA)
- `victim_type * 6 + 5 - attacker_type`
- Promotions: `promType - 5`
- En passant: score 5

### 5c. Bad Capture Detection
- A capture is "bad" if:
  1. `tp_value[victim] < tp_value[attacker]` (capturing down), AND
  2. Not en passant, AND
  3. `Swap(from, to) < 0`
- All minor-for-minor exchanges pass (equal piece values)

### 5d. Quiet Move Scoring
- Base: `mHistory[piece][to_square]`
- Bonus: +2048 if `from_square == refutation_square` (the square of the piece whose escape is being prioritized due to null move refutation)
- **This is novel**: the null move refutation tracking bumps the escape move of the threatened piece

### 5e. History Table
- Dimensions: `mHistory[12][64]` -- 12 piece types (WP,BP,WN,BN,...) x 64 squares
- **Update on beta cutoff**: `mHistory[piece][to] += 2 * depth * depth`
- **Decrease for tried moves**: `mHistory[piece][to] -= depth * depth`
- **Max value**: MAX_HIST = 32768. When exceeded, all entries halved (`TrimHist()`)
- **Age between searches**: all entries divided by 8, killers cleared
- **No separate capture history**

### 5f. Killer Moves
- 2 killers per ply
- Standard replacement: new killer goes to slot 0, old slot 0 goes to slot 1

### 5g. Refutation Table (Counter-Move)
- `mRefutation[64][64]` -- indexed by [from_sq][to_sq] of the previous move
- Updated on beta cutoff for quiet moves
- Also supports null move (lastMove == 0)

### 5h. History Limit for Pruning/Reduction
- `histLimit = -MAX_HIST + (MAX_HIST * hist_perc / 100)` where hist_perc defaults to 175
- Default: `-32768 + (32768 * 175 / 100) = -32768 + 57344 = 24576`
- Moves with history >= histLimit are **exempt** from LMR, futility pruning, and LMP
- Exposed as "Selectivity" UCI option (10-500)

---

## 6. Quiescence Search

### 6a. Three-Layer Design
1. **QuiesceChecks()**: entry from main search (depth 0). Tries TT move, good captures, killers, AND checking moves (via `GenerateSpecial()`). No bad captures.
2. **QuiesceFlee()**: called when in check. Tries ALL moves (full move generation). No stand-pat. This is an evasion handler.
3. **Quiesce()**: standard QS. Stand-pat, then captures only with delta pruning.

### 6b. Delta Pruning (in `Quiesce()`)
- Only when opponent has more than 1 non-pawn piece (to avoid pruning in near-endgame)
- Stage 1: skip if `floor + tp_value[victim] + 150 < alpha` (delta margin = 150)
- Stage 2: skip if `BadCapture()` returns true
- `floor` = stand-pat score (evaluation)

### 6c. QuiesceChecks Special Move Generator
- Uses `NextSpecialMove()` which has its own ordering:
  - TT move -> good captures -> killer 1 -> killer 2 -> checking quiet moves
  - Bad captures are pruned entirely (not saved)
  - This means the first QS ply considers checks, but deeper plies (Quiesce) do not

---

## 7. Time Management

### 7a. Base Time Allocation
```
SetMoveTime(base, inc, movestogo):
  if movestogo == 1: base -= min(1000, base/10)
  moveTime = (base + inc * (movestogo - 1)) / movestogo
```
- Default movestogo = 40 (if not specified by GUI)
- Percentage adjustment: `moveTime = moveTime * time_percentage / 100` (default 100, "SlowMover" option)
  - Only applied if `2 * moveTime > base` (safety check)
- Capped to not exceed total base time
- Subtracted by `time_buffer` (configurable, default likely 0)

### 7b. Bullet Correction
- time < 200ms: use 72% (23/32)
- time < 400ms: use 81% (26/32)
- time < 1200ms: use 91% (29/32)
- time >= 1200ms: no correction

### 7c. Score Instability ("TimeTricks")
- `scoreJump` flag set when aspiration window fails with margin > 50
- **If enabled** (`timeTricks` option, default false): time is **doubled** when scoreJump is true
- Simple binary decision -- no gradual scaling
- Score increase only triggers scoreJump in single-thread mode

### 7d. No Fail-Low/Fail-High Time Extension Beyond ScoreJump
- No sophisticated fail-low handling or ponder-hit integration beyond the basic scoreJump x2 multiplier
- No "instability" counter or best-move change tracking

---

## 8. Transposition Table

- 8-byte entries: key(64) + date(16) + move(16) + score(16) + flags(8) + depth(8) = 16 bytes per entry
- Replacement: age-priority scheme (via `tt_date`)
- No bucket/multi-slot probing visible in the header (single entry per hash position)
- Stores/retrieves with standard alpha/beta/depth filtering

---

## 9. Internal Iterative Deepening (IID)

- **PV nodes only**, not in check, no TT move, depth > 6
- Searches at depth - 2 with full [alpha, beta] window
- Then retrieves the TT move from hash
- No IID in cut nodes (explicitly noted as TODO in source)

---

## 10. Improving Detection

- `improving = true` if `ply > 2 && !inCheck && eval > evalStack[ply-2]`
- Used in: static null move margin, LMP table, LMR reduction
- Simple 2-ply comparison, no fallback to 4-ply

---

## 11. Eval Hash Adjustment
- After computing eval, if TT entry exists, adjusts eval toward hashScore:
  ```
  if (hashFlag matches direction of correction)
      eval = hashScore
  ```
- This is a simple "use TT score as better eval estimate" -- common in many engines

---

## 12. Skill Levels (SearchSkill Feature)

The `searchSkill` parameter (0-10) gates which search features are active:
- 0: no TT cutoffs
- 1: TT cutoffs enabled
- 2: null move enabled
- 3: LMR enabled
- 4: razoring enabled
- 5: futility pruning enabled
- 6: LMP enabled
- 7: aspiration windows enabled
- 8: static null move (RFP) enabled
- 9: bad capture LMR enabled
- 10: null move verification enabled (full strength)

---

## 13. Lazy SMP Details

- Odd threads start at depth+1, even at depth+1 (offset by threadId & 1)
- Skip iterations where >50% of threads already reached that depth
- Skip iterations if thread lags behind `depthReached` by >1
- All threads share only TT and global abort flag
- History, killers, refutation, eval hash, pawn hash are per-engine (per-thread)

---

## 14. Notable/Novel Features vs. Standard Engine

1. **Null move refutation tracking**: After null move search, retrieves the TT move that refuted null. The target square of that refutation move is passed to move ordering as `refutationSquare`. Quiet moves whose from-square matches get a +2048 bonus. This effectively prioritizes escaping the piece that the opponent threatened to capture after null move.

2. **Three-layer quiescence**: QuiesceChecks (first ply with checks + killers) -> QuiesceFlee (evasions) -> Quiesce (captures only). Most engines have two layers.

3. **TT-based null move skip**: Before executing null move, probes TT at the reduced depth. If TT score < beta, skips null move entirely. This avoids wasting time on null moves that are unlikely to cause a cutoff.

4. **History-gated pruning/reduction**: The `histLimit` parameter (default 24576 = ~75% of max) exempts high-history moves from LMR, futility, and LMP. Exposed as "Selectivity" UCI option. This is a crude but effective way to avoid pruning moves the engine has historically found good.

5. **Bad capture LMR**: Separately reduces bad captures by 1 ply (after move 6). Most engines either don't reduce captures or lump them with quiets.

6. **Configurable contempt**: `DrawScore()` returns `+/- Par.drawScore` depending on which side the engine is playing. Asymmetric evaluation possible.

7. **Personality system**: Asymmetric attack/mobility weights (own vs opponent), configurable piece values, eval blurring for weaker play, NPS limiting, and the skill level system that disables search features.

8. **No features Rodent has that we lack**:
   - We already have: NMP with verification, RFP, razoring, futility, LMP, LMR with history adjustments, singular extensions, check extensions, recapture extensions, PVS, aspiration windows, IID, killer/counter-move/history
   - Rodent lacks: ProbCut, continuation history, capture history, history pruning, IIR (only has IID in PV), SEE pruning of quiet moves at depth

---

## 15. Thresholds Summary Table

| Feature | Rodent III | GoChess |
|---------|-----------|---------|
| RFP depth | <= 7 | depth-dependent |
| RFP margin (improving) | 125 * depth | 60 * depth |
| RFP margin (not improving) | 175 * depth | 85 * depth |
| NMP base R | (823+67*d)/256 | 3 + depth/3 |
| NMP eval bonus | min(3, (eval-beta)/200) | (eval-beta)/200 |
| NMP verification | depth > 6, re-search at d-5 | none |
| Razoring depth | <= 4 | 400 + depth*100 |
| Razoring margins | 300/360/420/480 | flat |
| Futility depth | <= 6 | 100 + lmrDepth*100 |
| Futility margins | 100/150/200/250/300/400 | linear |
| LMP depth | <= 10 | unlimited (3+d^2 formula) |
| LMR formula | 0.33 + log(d)*log(mc)/2.00 | C=1.5 + log(d)*log(mc)/C |
| LMR PV reduction | base - 1 | less aggressive |
| Aspiration initial | 8 | 15 |
| Aspiration widening | x2 | x2 |
| Singular margin | singScore + 50 | depth * 3 |
| Delta pruning margin | 150 | (check our QS) |
| History dimensions | [12][64] | [12][64] + capture + conthist |
| History bonus | 2*depth^2 | depth^2 |
| History penalty | depth^2 | depth^2 |
| History max | 32768, halve on overflow | gravity-based |
