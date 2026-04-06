# Clarity Chess Engine - Crib Notes

Source: `~/chess/engines/Clarity/src/`
Author: Joseph Pasfield ("Vast")
Version: V7.2.0
Language: C++
Eval: NNUE (SCReLU, embedded net)
Approximate strength: ~+200 Elo above Coda

---

## 1. Overview

Clarity is a C++ UCI engine with embedded NNUE evaluation. It uses standard iterative deepening with aspiration windows, PVS, and Lazy SMP (via `std::jthread`). The search is clean and relatively straightforward -- no `setjmp`/`longjmp` or fancy abort mechanisms, just an atomic `timesUp` flag checked every 4096 nodes.

The engine has a strong SPSA-tuned parameter set exposed via `tunables.h`, with JSON/OB export for automated tuning. All major search parameters are tunable at runtime.

---

## 2. NNUE Architecture

**Current net**: `cn_028`
**Architecture**: `(768x6 -> 1024)x2 -> 1x8`
- **Input**: 768 features (12 piece types x 64 squares)
- **Input buckets**: 6 king buckets with **horizontal mirroring** (left/right symmetry)
  - Bucket layout (half-board): `{0,0,1,1, 2,2,3,3, 4,4,4,4, 4,4,4,4, 5,5,5,5, 5,5,5,5, 5,5,5,5, 5,5,5,5}`
  - Mirrored: king on files e-h mirrors to files a-d, doubling to 12 effective buckets
- **Hidden layer**: 1024 neurons
- **Activation**: SCReLU (squared clipped ReLU)
- **Output buckets**: 8 (material-count based, `(pieceCount - 2) / 4`)
- **Quantization**: QA=256, QB=64, Scale=400
- **Finny table**: Per-perspective, per-bucket refresh table for accumulator updates on king bucket changes
- **SIMD**: AVX-512, AVX2, SSE2 fallback. Uses SomeLizard's CReLU-decomposition trick for SCReLU.
- **No hidden layers beyond FT**: Direct FT -> output (equivalent to Coda's v5 architecture)

**Coda comparison**: Coda uses 16 king buckets (no horizontal mirroring), QA=255 (vs 256). Clarity's horizontal mirroring halves the FT weight count, allowing more effective training with less data. Coda supports v5/v6/v7 architectures; Clarity is v5-equivalent only. The 6-bucket layout with mirroring is notably different from Coda's 16-bucket layout.

---

## 3. Search Features

### Aspiration Windows
- Base delta: `13` (tunable `aspBaseDelta`)
- Delta multiplier on fail: `1.22x` (tunable `aspDeltaMultiplier`)
- Activation: depth > `5` (tunable `aspDepthCondition`)
- On fail-high: widen beta, reduce search depth by 1 (capped at `depth - 5`)
- On fail-low: contract beta to `(alpha + beta) / 2`, widen alpha, restore full depth
- **Coda comparison**: Coda uses delta=13+avg^2/23660 (score-dependent), growth 1.5x. Clarity uses flat delta=13, growth 1.22x. Both reduce depth on fail-high.

### Reverse Futility Pruning (RFP)
- Depth guard: `depth < 9` (tunable `rfpDepthCondition`)
- Margin: `85 * (depth - improving)` (tunable `rfpMultiplier`)
- Returns `(staticEval + beta) / 2` (averaged, not raw staticEval)
- Guards: `!inSingularSearch && !inCheck && !isPV`
- **Coda comparison**: Coda uses depth<=7, margin 70*d (improving) / 100*d (not improving), returns staticEval-margin. Clarity's depth 9 cutoff is more aggressive. Clarity returns `(eval+beta)/2` which is a score-clamped return -- more conservative than returning eval directly.

### Null Move Pruning (NMP)
- Conditions: `!isPV && !inSingularSearch && !board.isPKEndgame() && nmpAllowed && !inCheck`
- Eval gate: `staticEval >= beta && staticEval >= beta + 175 - 25 * depth`
  - At depth 3: eval >= beta + 100
  - At depth 7: eval >= beta + 0 (threshold vanishes)
  - At depth 10: eval >= beta - 75 (relaxes past beta)
- Reduction: `3 + depth/3 + min((staticEval - beta) / 131, 4)` (tunable `nmpDivisor`, `nmpSubtractor`)
- Returns raw score (no dampening or beta clamping)
- No NMP verification search
- **Coda comparison**: Coda uses R=3+d/3+(eval-beta)/200, verify at depth>=12, score dampening (2s+b)/3. Clarity has the depth-dependent eval gate (175-25*d) which Coda lacks. Coda's NMP verification prevents zugzwang in endgames; Clarity uses `isPKEndgame()` guard instead.

### Razoring
- Condition: `!isPV && staticEval < alpha - 436 * depth` (tunable `razDepthMultiplier`)
- Action: drop to QS, return if QS < alpha
- No explicit depth guard (margin grows fast enough)
- **Coda comparison**: Coda uses depth<=2, margin 400+d*100. Clarity's 436*d is similar scale but no depth cap.

### Depth Extension on High Reduction
- Unique feature: `if(!isPV && !inCheck && ply > 0 && info.stack[ply-1].reduction >= 3 && !opponent_worsening) ++depth`
- `opponent_worsening` = `staticEval + info.stack[ply-1].staticEval > 1` (both sides think they're winning = not worsening)
- Extends search by 1 ply when the parent node was heavily reduced and the opponent isn't worsening
- **Coda comparison**: Coda has no equivalent. This is a form of "reduced move was surprisingly important" extension.

### Internal Iterative Reduction (IIR)
- Condition: `depth > 2` and (TT miss or TT has no best move)
- Single condition (not split PV/cut like Weiss)
- Reduction: `depth--`
- **Coda comparison**: Coda uses depth>=6. Clarity's depth>2 threshold is much more aggressive.

### Mate Distance Pruning
- Non-PV only: clamp alpha/beta to mating scores at current ply
- **Coda comparison**: Coda has no mate distance pruning. MISSING FEATURE.

### Futility Pruning
- Condition: `bestScore > matedScore && !inCheck && depth <= 0 && staticEval + 347 + depth * 44 <= alpha`
  - Note: `fpDepthCondition` = 0, so this is effectively **disabled** (depth <= 0 means only at depth 0, which goes to QS anyway)
  - Base: 347 (tunable `fpBase`), multiplier: 44 per depth (tunable `fpMultiplier`)
- **Coda comparison**: Coda uses futility at depth<=8, margin 60+lmrDepth*60. Clarity has this effectively disabled.

### Late Move Pruning (LMP)
- Condition: `!isPV && isQuiet && bestScore > matedScore+256 && legalMoves > (3 + depth * depth) / (2 - improving)`
- Base: 3 (tunable `lmpBase`)
- At depth 3 (improving): `12`, at depth 3 (not improving): `6`
- **Coda comparison**: Coda uses 3+d^2 with improving +50% / failing -33%. Very similar formulas. At depth 3: Clarity=12/6, Coda=18/8.

### SEE Pruning
- Depth guard: `depth <= 1` (tunable `sprDepthCondition`)
  - This is very conservative -- only at depth 1!
- Capture threshold: `-97 * depth` (tunable `sprCaptureThreshold`)
- Quiet threshold: `-36 * depth` (tunable `sprQuietThreshold`)
- Also requires `isQuietOrBadCapture` (moveValue <= 5*historyCap)
- **Coda comparison**: Coda uses SEE pruning at depth<=8 (quiet) and depth<=6 (capture) with larger margins. Clarity's SEE pruning is almost entirely disabled at depth>1.

### History Pruning
- Condition: `ply > 0 && !isPV && isQuiet && depth <= 9 && moveValue < -2335 * depth`
- Depth guard: 9 (tunable `hipDepthCondition`)
- Threshold: -2335 * depth (tunable `hipDepthMultiplier`)
- Uses `break` (not `continue`) -- skips all remaining quiets
- **Coda comparison**: Coda uses -1500*depth at depth<=3. Clarity's threshold is larger (-2335) and applies to much deeper nodes (depth<=9).

---

## 4. Extensions

### Check Extension
- `depth += inCheck` -- unconditional +1 when in check
- Applied after pruning, before move loop

### Singular Extensions
- Depth guard: `depth >= 7` (tunable `sinDepthCondition`)
- TT depth requirement: `entry->depth >= depth - 3` (tunable `sinDepthMargin`)
- TT bound: `entry->flag != FailLow` (not upper bound)
- Singular beta: `entry->score - depth * 30 / 16` (= `depth * 1.875`) (tunable `sinDepthScale`)
- Singular depth: `(depth - 1) / 2`
- **Single extension (+1)**: score < singularBeta
- **Double extension (+2)**: `!isPV && score < singularBeta - 33 && doubleExtensions <= 21`
  - Cap of 21 double extensions per line (tunable `dexLimit`)
  - Additional depth extension: `depth += (depth < 14)` when double extending (tunable `deiDepth`)
- **Triple extension (+3)**: `score < singularBeta - 67` (tunable `texMargin`)
  - Same double extension counter increment
- **Multi-cut**: `singularBeta >= beta` -> return singularBeta
- **Negative extension (-2)**: `entry->score >= beta` (TT score beats beta, not singular)
- **Cut-node extension (-1)**: `isCutNode` and not singular
- **Coda comparison**: Coda has SE at depth>=8, margin=tt_score-depth, multi-cut, negative ext (-1). Clarity has:
  - Lower SE depth threshold (7 vs 8)
  - Double extensions (capped at 21, very generous -- Coda has none)
  - Triple extensions (Coda has none)
  - Stronger negative extension (-2 vs -1)
  - Cut-node -1 extension on non-singular TT move (Coda has none)
  - Depth boost when double extending at shallow depth

---

## 5. LMR (Late Move Reductions)

### Table
Single unified table (not separate quiet/capture):
- Formula: `0.97 + ln(depth) * ln(moves) * 0.54`
- Base: 0.97 (tunable `lmrBase`), multiplier: 0.54 (tunable `lmrMultiplier`)

### Conditions
- `depth > 1` (tunable `lmrDepth`)
- Applied to all non-first moves (no moveCount threshold beyond legalMoves > 1)

### Adjustments
- `-= isPV` (reduce less in PV)
- `-= improving` (reduce less when improving)
- `-= !isQuietOrBadCapture` (reduce less for good captures)
- `-= corrhistUncertain` (reduce less when correction history has large magnitude -- uncertain eval)
- `+= isCutNode * 2` (reduce more at cut nodes)
- For quiets: `-= moveValue / 8711` (tunable `hmrDivisor`)
- For captures: `-= captureHistory / 4677` (tunable `cmrDivisor`) -- uses opponent's perspective history for the capture
- Clamped to `[0, depth - 1]`
- **Stores reduction in stack**: `info.stack[ply].reduction = lmr` for use in depth extension

### Re-search
- If score > alpha and (score < beta or lmr > 0): full-window re-search at `depth - 1`
- No doDeeper/doShallower logic

**Coda comparison**: Coda uses separate quiet (C=1.30) and capture (C=1.80) tables. Coda has doDeeper/doShallower. Clarity's `corrhistUncertain` LMR adjustment is unique -- reducing less when the static eval correction is large (position is hard to evaluate). Clarity's cut-node +2 is very aggressive. Clarity stores the reduction for use by child nodes (depth extension feature).

---

## 6. Move Ordering

### Stages
TTMove -> GenAll -> All (selection sort) -> QSGenAll -> QSAll

### Score Hierarchy

1. **TT move**: 1,000,000,000 (returned first)
2. **Captures**: `MVV[victim] + captureHistory[color][piece][to][victim][toAttacked]`
   - If SEE >= 0: add 500,000 bonus (good captures)
   - SEE < 0: no bonus (bad captures, sorted among quiets by history)
3. **Killer move**: 81,922
4. **Counter-move**: 81,921
5. **Quiets**: `historyTable[color][from][fromAttacked][to][toAttacked] + contHist(ply-1) + contHist(ply-2) + contHist(ply-4) + pawnHistory[pawnHash][color][piece][to]`

### History Tables

**Threat-aware butterfly history**: `historyTable[2][64][2][64][2]` -- color, from, fromAttacked, to, toAttacked
- This is a 4x larger history table than standard butterfly, using attack status of both source and destination squares
- Gravity: `entry += bonus - entry * abs(bonus) / 16384`
- Bonus: `min(1632, 276 * depth - 119)` (tunables `hstMaxBonus`, `hstAdder`, `hstSubtractor`)
- Malus: symmetric (same magnitude, negated)
- **Coda comparison**: Coda uses `[2][64][64]` (no threat awareness). Clarity's threat-aware history is a significant structural difference. MISSING FEATURE for Coda.

**Capture history (noisy)**: `noisyHistoryTable[2][6][64][7][2]` -- color, piece, to, victim, toAttacked
- Also threat-aware (toAttacked dimension)
- Gravity divisor: 16384

**QS capture history**: `qsHistoryTable[2][6][64][7]` -- separate table for QS, no attack dimension
- Updated on QS beta cutoffs with depth-dependent bonus: `min(1525, 6*qDepth^2 + 79*qDepth - 92)`
- **Coda comparison**: Coda has no separate QS history table. MISSING FEATURE.

**Pawn history**: `pawnHistoryTable[32768][2][6][64]` -- pawnHash, color, piece, to
- 32768 buckets (Coda uses 512)
- Gravity divisor: 16384
- **Coda comparison**: Coda has 512-bucket pawn history. Clarity's 32768 buckets provide much finer granularity.

**Continuation history**: `CHTable = [2][7][64][7]` per entry -- indexed by [color][piece][to][victim]
- 4D structure that unifies quiet and capture continuation by including victim type (None for quiets)
- Plies used in ordering: 1, 2, 4 (same as Coda's ply 1, 2, 4)
- Gravity divisor: 16384
- **Coda comparison**: Coda uses [piece][to][piece][to] (2D per entry) at plies 1, 2, 4, 6. Clarity uses [color][piece][to][victim] at plies 1, 2, 4.

**Killer**: 1 per ply (Coda has 2)

**Counter-move**: `counterMoves[64][64]` -- indexed by previous move's start/end squares
- **Coda comparison**: Coda has counter-move indexed by [piece][to]. Similar concept.

---

## 7. Correction History

### Components (4 sources)
1. **Pawn correction**: `pawnTable[16384][2]` -- indexed by pawnHash, weight 512/512
2. **Non-pawn correction (per color)**: `nonPawnTables[16384][2][2]` -- indexed by nonPawnHash per color, weight 512/512
3. **Major piece correction**: `majorTable[16384][2]` -- indexed by majorHash, weight 512/512
4. **Minor piece correction**: `minorTable[16384][2]` -- indexed by minorHash, weight 512/512

### Blending Formula
```
correction = (pawnTable * 512 + (nonPawnTables[ctm] + nonPawnTables[1-ctm]) * 512
            + majorTable * 512 + minorTable * 512) / 512
correctedEval = staticEval + correction / 256
```

All weights are 512/512 = 1.0 (equal weighting), scale divisor = 256.

### Update
- Weight: `min(depth - 1, 16)`
- Error: `bestScore - staticEval`
- Exponential moving average: `entry = (entry * (256 - weight) + error * 256 * weight) / 256`
- Clamp: `[-64 * 256, 64 * 256]` = `[-16384, 16384]`

### Uncertainty Detection
- `corrhistUncertain = abs(correction) > 128` (tunable `chUncertaintyMargin`)
- Used in LMR: `-= corrhistUncertain` (reduce less when correction is large)
- **Coda comparison**: Coda has 6 sources (pawn + nonpawn*2 + major + minor + continuation) with tuned weights. Clarity has 4 sources (same as Coda minus continuation correction) with equal weights. Coda does NOT have the uncertainty-based LMR adjustment. Clarity's correction history is simpler but the uncertainty feature is novel.

---

## 8. Transposition Table

### Structure
- Single entry per slot (no buckets!)
- Entry: 10 bytes packed (`score i16, move u16, key u16, staticEval i16, flag u8, depth u8`)
- Index: Lemire's fast modulo `((u128)key * size) >> 64`
- Key stored as 16-bit truncation (higher collision risk than Coda's XOR method)

### Replacement
- Always replace (no replacement policy -- new entry overwrites old)

### Probe
- TT cutoff: `!isPV && entry->depth >= depth && (exact || lower+failHigh || upper+failLow)`
- TT eval adjustment: if TT score is more informative than staticEval, use it

### Prefetch
- `__builtin_prefetch(TT->getEntry(afterKey))` before make_move (using `keyAfter`)
- Applied in both main search and QS

**Coda comparison**: Coda uses 5-slot buckets with lockless atomics and XOR key verification. Clarity's single-slot always-replace TT is much simpler but has higher collision/overwrite rates. Coda's design is significantly more sophisticated here.

---

## 9. Time Management

### Base Allocation
- Default moves to go: 20
- Move overhead: 10ms (default)
- Soft bound: `0.6 * (time / movestogo + inc * 3 / 4)` (tunables `tmsMultiplier`, `tmsNumerator`, `tmsDenominator`)
- Hard bound: `time / 2` (tunable `tmhDivisor`)
- Hard limit checked every 4096 nodes

### Node-Based Time Management
- Tracks nodes per root move: `nodeTMTable[from][to]`
- After each iteration: `frac = bestMoveNodes / totalNodes`
- Soft bound scaling at depth > 9: `softBound * (1.51 - frac) * 1.44` (tunables `ntmSubtractor`, `ntmMultiplier`)
- At depth <= 9: `softBound * 1.25` (tunable `ntmDefault`)
- Effect: if best move gets 60% of nodes, factor = (1.51 - 0.6) * 1.44 = 1.31. If 20%, factor = 1.89. If 90%, factor = 0.88.
- **Coda comparison**: Coda uses simpler time management (time_left/moves + 80% inc, 5x hard cap). No node-ratio scaling. MISSING FEATURE.

### Best-Move Stability
- Tracks consecutive iterations where best move doesn't change
- Stability multiplier table (7 entries, tuned):
  - 0 stable: 2.2x
  - 1 stable: 1.6x
  - 2 stable: 1.4x
  - 3 stable: 1.1x
  - 4 stable: 1.0x
  - 5 stable: 0.95x
  - 6+ stable: 0.9x
- Applied as multiplier to soft bound after node TM
- **Coda comparison**: Coda has no best-move stability time management. MISSING FEATURE. This is a major time management improvement -- spending less time when confident, more when uncertain.

---

## 10. Lazy SMP

- Thread count configurable via UCI `Threads` option
- Each thread has own `Engine` instance with own history tables, correction history, stack
- Shared: `TranspositionTable` only (no atomic access -- relies on x86 aligned write atomicity)
- Main thread (index 0) handles info output and time management
- All threads search same position to same depth (no offset depths)
- Stop via atomic `timesUp` flag

**Coda comparison**: Very similar design. Coda uses proper atomic access for TT (XOR key verification). Coda's helper threads search at offset depths for depth diversity. Clarity's threads all search the same depth.

---

## 11. Other Notable Features

### Opponent Worsening Detection
- `opponent_worsening = staticEval + info.stack[ply-1].staticEval > 1`
- Used to gate the "extend after high reduction" feature
- The idea: if both sides think they're winning (both evals positive from their perspective), the situation is dynamic and shouldn't be extended

### Correction History Uncertainty in LMR
- When correction history makes a large adjustment (>128cp), reduce LMR
- Rationale: uncertain positions deserve deeper search

### QS History
- Separate capture history table for QS, updated on QS beta cutoffs
- Bonus scales with `seldepth - ply` (QS depth proxy)

### Syzygy Tablebase Support
- Root probe via Fathom library
- Probes at <= 7 pieces at root before search

### Score Normalization
- WDL-based score normalization for UCI output
- Logistic curve parameters tuned to ply count

---

## 12. Parameter Comparison Table

| Feature | Clarity | Coda |
|---------|---------|------|
| **RFP depth** | < 9 | <= 7 |
| **RFP margin** | 85*(d-improving) | 70*d (imp) / 100*d (not) |
| **RFP return** | (eval+beta)/2 | eval-margin |
| **NMP base R** | 3 + d/3 | 3 + d/3 |
| **NMP eval div** | 131 | 200 |
| **NMP eval clamp** | min(..., 4) | no clamp |
| **NMP eval gate** | eval >= beta+175-25*d | none |
| **NMP PK guard** | isPKEndgame() | none |
| **NMP verification** | none | depth >= 12 |
| **NMP dampening** | none | (2*score+beta)/3 |
| **IIR depth** | > 2 | >= 6 |
| **Razoring** | 436*d (no depth cap) | depth<=2, 400+d*100 |
| **LMR formula** | 0.97 + ln(d)*ln(m)*0.54 | ln(d)*ln(m)/C |
| **LMR quiet C** | 0.54 (multiplier) | 1.30 (divisor) |
| **LMR cutnode** | +2 | none |
| **LMR corrhistUncertain** | -1 when uncertain | none |
| **LMR history div (quiet)** | 8711 | varies |
| **LMR history div (cap)** | 4677 | varies |
| **LMP base** | 3 | 3 |
| **LMP formula** | (3+d*d)/(2-improving) | 3+d^2 with improving adj |
| **SEE prune depth** | <= 1 | quiet<=8, cap<=6 |
| **SEE quiet threshold** | -36*d | -20*d^2 |
| **SEE capture threshold** | -97*d | -d*100 |
| **History pruning depth** | <= 9 | <= 3 |
| **History pruning threshold** | -2335*d | -1500*d |
| **SE min depth** | >= 7 | >= 8 |
| **SE margin** | score - d*30/16 | tt_score - depth |
| **SE double ext** | +2 (cap 21) | none |
| **SE triple ext** | +3 (margin 67) | none |
| **SE negative ext** | -2 | -1 |
| **SE cutnode ext** | -1 | none |
| **Aspiration delta** | 13 (flat) | 13+avg^2/23660 |
| **Aspiration growth** | 1.22x | 1.5x |
| **Aspiration depth** | > 5 | >= 4 |
| **History bonus** | min(1632, 276*d-119) | depth^2 capped 1200 |
| **History symmetry** | symmetric | symmetric |
| **History cap** | 16384 | 1200 |
| **Butterfly hist** | [2][64][2][64][2] (threat-aware) | [2][64][64] |
| **Capture hist** | [2][6][64][7][2] (threat-aware) | [piece][to][victim] |
| **Pawn hist buckets** | 32768 | 512 |
| **ContHist plies** | 1, 2, 4 | 1, 2, 4, 6 |
| **Killers** | 1 per ply | 2 per ply |
| **Counter-move** | yes [64][64] | yes [piece][to] |
| **Correction sources** | 4 (pawn+nonpawn+major+minor) | 6 (pawn+nonpawn+major+minor+cont) |
| **CorrHist uncertainty** | yes (LMR adjustment) | none |
| **TT structure** | single entry, always replace | 5-slot buckets, lockless atomic |
| **TT key bits** | 16-bit | XOR key verification |
| **Time mgmt node TM** | yes (node fraction) | none |
| **Time mgmt stability** | yes (7-level multiplier) | none |
| **Mate distance pruning** | yes (non-PV) | none |
| **Depth ext after reduction** | yes (reduction>=3, !worsening) | none |
| **QS history** | separate table | none |
| **SEE values** | P=117 N=370 B=422 R=606 Q=1127 | P=100 N=320 B=330 R=500 Q=900 |
| **MVV values** | P=95 N=462 B=498 R=647 Q=1057 | standard MVV-LVA |
| **King buckets** | 6 (horizontal mirror) | 16 (no mirror) |
| **NNUE activation** | SCReLU | SCReLU/CReLU/Pairwise |
| **Output buckets** | 8 | 8 |
| **Contempt** | none visible | -10 |
| **Syzygy** | root probe | yes |

---

## 13. Things Clarity Has That Coda Doesn't

1. **Threat-aware butterfly history** `[color][from][fromAttacked][to][toAttacked]` -- 4x larger, conditions on whether squares are attacked. MISSING FEATURE.
2. **Threat-aware capture history** -- includes `toAttacked` dimension. MISSING FEATURE.
3. **Best-move stability time management** -- 7-level multiplier (0.9x to 2.2x). MISSING FEATURE.
4. **Node-based time management** -- fraction of nodes on best move scales soft bound. MISSING FEATURE.
5. **Double/triple singular extensions** -- +2 (cap 21) and +3 with tuned margins. MISSING FEATURE.
6. **Mate distance pruning** -- simple alpha/beta clamping in non-PV. MISSING FEATURE.
7. **Correction history uncertainty in LMR** -- reduce less when eval correction is large. MISSING FEATURE.
8. **Depth extension after high reduction** -- extend when parent was reduced by 3+ and opponent not worsening. MISSING FEATURE.
9. **Opponent worsening detection** -- `eval + parentEval > 1`. MISSING FEATURE.
10. **QS capture history** -- separate history table updated on QS cutoffs. MISSING FEATURE.
11. **Larger pawn history** -- 32768 buckets vs 512. MISSING FEATURE (can increase).
12. **Cut-node +2 LMR** -- aggressive reduction at expected cut nodes. MISSING FEATURE.
13. **Stronger negative extensions** -- -2 (Coda has -1) and cut-node -1. MISSING FEATURE.
14. **RFP returns (eval+beta)/2** -- averaged return prevents overshooting. MISSING FEATURE.
15. **NMP depth-dependent eval gate** -- `eval >= beta + 175 - 25*d`. MISSING FEATURE.
16. **IIR at depth > 2** -- much more aggressive than Coda's depth >= 6. TUNING DIFFERENCE.
17. **Horizontal mirroring in NNUE** -- halves effective feature weights. ARCHITECTURE DIFFERENCE.
18. **Aspiration fail-low beta contraction** -- `beta = (alpha+beta)/2`. PARTIAL (Coda has similar).
19. **Aspiration fail-high depth reduction** capped at `depth - 5`. PARTIAL.
20. **Unified quiet/capture continuation history** indexed by victim type. ARCHITECTURE DIFFERENCE.

---

## 14. Things Coda Has That Clarity Doesn't

1. **5-slot TT buckets** with lockless atomic access and XOR key verification (Clarity has single-entry always-replace)
2. **NMP verification search** at depth >= 12 (Clarity has none)
3. **NMP score dampening** (2*score+beta)/3
4. **2 killers per ply** (Clarity has 1)
5. **ContHist ply 6** (Clarity uses plies 1, 2, 4 only)
6. **Continuation correction history** (Clarity has no cont-corr)
7. **Recapture extensions**
8. **Fail-high score blending** in main search
9. **TT score dampening** at cutoffs
10. **TT near-miss cutoffs** (1 ply shallower with margin)
11. **v6/v7 NNUE architectures** (pairwise, hidden layers)
12. **Contempt** (-10, anti-draw)
13. **doDeeper/doShallower** after LMR re-search
14. **Score-dependent aspiration delta** (Coda's scales with previous score)
15. **SEE pruning at meaningful depths** (Clarity's is effectively depth<=1 only)
16. **Helper thread depth offset** for Lazy SMP diversity
17. **Polyglot opening book support**

---

## 15. Ideas Worth Testing from Clarity

### High priority (biggest Elo potential):

1. **Node-based time management + best-move stability** -- This is likely one of Clarity's biggest advantages. The 7-level stability multiplier (0.9x-2.2x) combined with node-fraction scaling means Clarity spends dramatically more time on uncertain positions and less on stable ones. Both features should be implemented together.

2. **Double/triple singular extensions** with cap -- `+2 when score < singBeta - 33, +3 when < singBeta - 67, capped at 21 per line`. Widely adopted across top engines. The generous cap of 21 (vs Weiss's 5) is notable.

3. **Threat-aware butterfly history** `[color][from][attacked][to][attacked]` -- 4x larger table that distinguishes quiet positions from tactical ones. This is a significant move ordering improvement.

4. **Cut-node +2 LMR** -- `lmr += isCutNode * 2`. Very aggressive but consensus feature.

5. **Mate distance pruning** -- 3 lines of code, universal technique, no downside.

### Medium priority:

6. **Correction history uncertainty in LMR** -- `corrhistUncertain = abs(correction) > 128; lmr -= corrhistUncertain`. Novel technique that reduces less in hard-to-evaluate positions. Cheap to implement.

7. **Depth extension after high reduction** -- `if(parent.reduction >= 3 && !opponent_worsening) depth++`. Interesting idea that compensates for aggressive LMR.

8. **NMP depth-dependent eval gate** -- `staticEval >= beta + 175 - 25*depth`. Prevents NMP at shallow depth when eval barely exceeds beta.

9. **Stronger negative extensions** -- -2 instead of -1, plus cut-node -1 on non-singular TT moves.

10. **Larger pawn history** -- 32768 buckets vs 512. Simple change, may improve move ordering.

11. **RFP averaged return** -- `(staticEval + beta) / 2` instead of raw eval. Prevents excessive pruning scores.

12. **QS capture history** -- separate table for QS with depth-aware bonuses.

### Lower priority:

13. **More aggressive IIR** -- depth > 2 (currently depth >= 6). Test carefully, may be destabilizing.

14. **RFP at depth < 9** -- Coda uses depth <= 7. Wider application may help.

15. **History pruning at depth <= 9** with -2335 threshold -- Coda uses depth <= 3 with -1500. Much wider application.

16. **Horizontal mirroring in NNUE** -- architecture change, would need retraining. Long-term.

17. **Unified continuation history** with victim dimension -- architectural change to cont-hist tables.

---

## 16. Key Takeaways

Clarity's +200 Elo advantage over Coda likely comes from several compounding factors:

1. **Time management** is the biggest gap. Clarity's node TM + stability multiplier means it allocates time much more intelligently. This alone could be worth 30-50 Elo.

2. **Threat-aware history** tables give better move ordering in tactical positions, which translates to better pruning decisions.

3. **Double/triple singular extensions** allow deeper search on critical lines, compensating for aggressive pruning elsewhere.

4. **Cut-node +2 LMR** enables much more aggressive reduction at expected cut nodes, saving time for important lines.

5. **Better tuning** -- Clarity's SPSA-tuned parameters are likely well-optimized. Many of Coda's parameters come from GoChess and may not be optimal for the current search structure.

6. Clarity's TT is notably simpler (single entry, 16-bit key) which suggests the search features and time management are doing the heavy lifting, not the TT design. Coda's TT is objectively better here.
