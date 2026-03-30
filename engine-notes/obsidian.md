# Obsidian Chess Engine - Crib Notes

Source: `~/chess/engines/Obsidian/`
Author: gab8192
Version: dev-16.15
Strength: #2 in our RR at +318 Elo (426 above GoChess-v5)
NNUE: (768x13->1536)x2 -> 16 -> 32 -> 1x8 (pairwise mul, dual activation L2, NNZ-sparse L1, FinnyTable)

---

## 1. NNUE Architecture

### Network: (768x13 -> 1536) x2 -> 16 -> 32 -> 1 x8

This is a deep, wide, modern architecture with several advanced techniques.

**Feature Transformer (FT):**
- Input: HalfKA — 768 features (12 piece types x 64 squares) per king bucket
- 13 king buckets (mirrored horizontally when king on files e-h):
  ```
  0  1  2  3  3  2  1  0
  4  5  6  7  7  6  5  4
  8  8  9  9  9  9  8  8
  10 10 10 10 10 10 10 10
  11 11 11 11 11 11 11 11
  11 11 11 11 11 11 11 11
  12 12 12 12 12 12 12 12
  12 12 12 12 12 12 12 12
  ```
- Output: 1536 int16 per perspective (very wide)
- Weights: `[KingBuckets][2][6][64][1536]` — color-relative, piece-type-indexed
- Horizontal mirroring: if king file >= E, square is XOR'd with 7

**Pairwise Multiplication (FT Activation):**
- Clips first half to [0, QA=255], second half to [-inf, QA=255]
- Multiplies corresponding pairs: `c0 * c1` where c0 is from first half, c1 from second half (offset by L1/2 = 768)
- Uses `mulhi_epi16` with left-shift for the product: `mulhi(slli(c0, 16-FtShift), c1)` where FtShift=9
- Packs result to uint8 via `packus_epi16` — output is 1536 uint8 values (768 per perspective)
- This is the same pairwise technique as Alexandria

**NNZ (Non-Zero) Sparse L1:**
- While computing FT output, builds a sparse index of non-zero 4-byte chunks
- Uses `getNnzMask()` — a movemask on int32 > 0 — to identify active chunks
- Lookup table `nnzTable[256][8]` maps each byte bitmask to active indices
- L1 propagation only processes non-zero entries, skipping zeros entirely
- **This is a significant speedup** — typical positions have many zero FT outputs

**L1 (1536 -> 16):**
- int8 weights, int32 accumulation via `dpbusd` (dot product of unsigned bytes and signed bytes)
- Only non-zero FT outputs are processed (NNZ sparse)
- Output: float, via `cvtepi32_ps` then multiply by `L1Mul = 1.0 / (QA * QA * QB >> FtShift)`
- Biases are float
- 8 output buckets (by material count: `(pieceCount - 2) / divisor` where divisor = `(32 + 8 - 1) / 8 = 4`)

**L2 (16*2 -> 32) — Dual Activation:**
- Input is 32 floats: 16 from `clamp(L1, 0, 1)` + 16 from `min(L1^2, 1)`
- Both linear and squared activations are concatenated, doubling the effective input
- Standard float matmul, CReLU activation clamp [0, 1]

**L3 (32 -> 1):**
- Float dot product, add bias
- Final score: `l3Out * NetworkScale` where NetworkScale = 400

**Quantization:** QA=255, QB=128, Scale=400

**FinnyTable (Accumulator Cache):**
- `FinnyEntry[2][KingBuckets]` — indexed by [file>=E mirror][bucket]
- Each entry stores: `byColorBB[2][2]`, `byPieceBB[2][PIECE_TYPE_NB]`, and a full `Accumulator`
- On king refresh: compares cached vs actual bitboards, applies only deltas (add/remove/move)
- Avoids full recompute when king moves within the same bucket
- Reset at start of each search

**SIMD:**
- AVX-512 (primary), AVX2 (fallback), SSSE3 (minimum)
- No ARM/NEON support
- Weight transpose at load time to match `packus_epi16` interleaving behavior

### Compared to GoChess NNUE
| Feature | Obsidian | GoChess |
|---------|----------|---------|
| Topology | (768x13->1536)x2->16->32->1x8 | v5: (768x16->N)x2->1x8 (shallow wide) |
| FT width | 1536 | Dynamic (1024/1536/any) |
| King buckets | 13 (mirror) | 16 |
| Output buckets | 8 (linear material) | 8 (material) |
| FT activation | Pairwise mul (uint8 out) | Pairwise + CReLU/SCReLU (UPDATE: now supports both) |
| L1 sparse | NNZ-sparse (skip zeros) | Dense |
| L2 activation | Dual (linear + squared) | CReLU |
| Quantization | QA=255, QB=128 | QA=127, QB=64 |
| Accumulator cache | FinnyTable | Finny tables (UPDATE: merged) |
| SIMD | AVX-512/AVX2/SSSE3 | AVX2/NEON |

**Key NNUE takeaways:**
1. Pairwise multiplication is a huge architectural advantage — doubles effective information without doubling weights
2. NNZ sparse L1 saves significant compute by skipping zeros
3. Dual activation (linear + squared) in L2 captures non-linear interactions cheaply
4. FinnyTable reduces king-bucket refresh cost
5. Their FT is wider (1536 vs our dynamic width which can match) **(UPDATE 2026-03-21: GoChess v5 now has pairwise mul, SCReLU, dynamic width up to 1536, and Finny tables)**

---

## 2. Eval Pipeline

The evaluation function is minimalist — pure NNUE with post-processing:

```cpp
score = NNUE::evaluate(pos, accumulator);
score += contempt;  // relative to root STM
score = score * (230 + phase) / 330;  // material scaling
score = clamp(score, TB_LOSS+1, TB_WIN-1);
```

**Material phase scaling:**
- Phase = 2*pawns + 3*knights + 3*bishops + 5*rooks + 12*queens
- Full material ~= 100 (16P + 4N + 4B + 4R + 2Q = 32+12+12+20+24 = 100)
- Scale = `(230 + phase) / 330`
- At full material: (230+100)/330 = 1.0. At endgame phase 10: 240/330 = 0.73
- **This dampens eval significantly in endgames**

**adjustEval (correction history + 50mr):**
```cpp
eval = (eval * (200 - halfMoveClock)) / 200;  // 50-move decay
eval += PawnChWeight(30) * pawnCorrhist / 512;
eval += NonPawnChWeight(35) * wNonPawnCorrhist / 512;
eval += NonPawnChWeight(35) * bNonPawnCorrhist / 512;
eval += ContChWeight(27) * contCorrHist / 512;
```

**Correction History (4 tables):**
1. Pawn correction: `pawnCorrhist[pawnKey % 32768][stm]`
2. White non-pawn correction: `wNonPawnCorrhist[nonPawnKey[WHITE] % 32768][stm]`
3. Black non-pawn correction: `bNonPawnCorrhist[nonPawnKey[BLACK] % 32768][stm]`
4. Continuation correction: `contCorrHist[pieceTo(lastMove)][pieceTo(move_before_last)]`

Per-color non-pawn Zobrist keys are maintained incrementally in `putPiece`/`removePiece`/`movePiece`:
```cpp
if (piece_type(pc) == PAWN)
    pawnKey ^= ZOBRIST_PSQ[pc][sq];
else
    nonPawnKey[piece_color(pc)] ^= ZOBRIST_PSQ[pc][sq];
```

Correction history gravity: `history += value - history * abs(value) / CORRHIST_LIMIT(1024)`

**Correction history update (at node end):**
- Conditions: not in check, best move is quiet, and bound allows use
- Bonus: `clamp((bestScore - staticEval) * depth / 8, -CORRHIST_LIMIT/4, CORRHIST_LIMIT/4)`
- Updates all 4 tables with same bonus

**Eval History (opponent move feedback):**
```cpp
theirLoss = (ss-1)->staticEval + ss->staticEval - 58;
bonus = clamp(-492 * theirLoss / 64, -534, 534);
addToHistory(mainHistory[~stm][from_to(lastMove)], bonus);
```
If opponent's move made things worse for them (theirLoss positive), we give negative bonus to that move in their butterfly history. This is a form of opponent-move quality feedback.

### Compared to GoChess
| Feature | Obsidian | GoChess |
|---------|----------|---------|
| Material scaling | Phase-based `(230+phase)/330` | None |
| 50-move decay | `(200-hmClock)/200` | None |
| Correction tables | 4 (pawn, wNonPawn, bNonPawn, cont) | 1 (pawn only) |
| Non-pawn corr keys | Per-color Zobrist | N/A |
| Cont correction | piece-to indexed | N/A |
| Eval history | Opponent move feedback | None |
| Contempt | Root-STM relative | Fixed |

---

## 3. Search Architecture

### Iterative Deepening
- Depth 1 to settings.depth (MAX_PLY-4 = 123 default)
- Early exit if only 1 legal move and elapsed >= 200ms
- Main thread handles info output and time decisions

### Aspiration Windows
- Start depth: **4** (`AspWindowStartDepth = 4`) — very early, ours is implicit
- Initial delta: **6** (`AspWindowStartDelta = 6`) + `avgScore^2 / 13000` — score-adaptive!
  - At score +-100: delta = 6 + 0.77 = ~7
  - At score +-500: delta = 6 + 19 = 25
  - At score +-1000: delta = 6 + 77 = 83
- On fail-low: `beta = (alpha + beta) / 2`, alpha widens by window
- On fail-high: beta widens by window, `failHighCount++` (if score < 2000)
- Adjusted depth: `max(1, rootDepth - failHighCount)` — reduces search depth on repeated fail-highs
- Window growth: `window += window / 3` (1.33x per iteration)

### Draw Detection
- 50-move rule: `halfMoveClock >= 100`
- Repetition: two-fold in tree, three-fold across root
- **Upcoming repetition detection (Cuckoo hashing)**: checks if any reversible move in history could lead to a repetition, adjusts alpha upward to DRAW score. This is a sophisticated Stockfish technique.
- Draw score: `SCORE_DRAW = 0` (no randomization)

### 50-Move TT Key Modification
```cpp
posTtKey = pos.key ^ ZOBRIST_50MR[pos.halfMoveClock];
```
ZOBRIST_50MR is non-zero only for halfMoveClock >= 14, and changes every 8 half-moves. This prevents TT entries from being reused when the 50-move clock is significantly different — positions near draw by 50MR should not share TT entries with fresh positions.

### Mate Distance Pruning
```cpp
alpha = max(alpha, ply - SCORE_MATE);
beta = min(beta, SCORE_MATE - ply - 1);
if (alpha >= beta) return alpha;
```
Standard, applied at non-root nodes.

---

## 4. Pruning Techniques

### Razoring
- Conditions: `!IsPV && alpha < 2000 && eval < alpha - 352 * depth`
  - Depth 1: eval < alpha - 352
  - Depth 2: eval < alpha - 704
  - Depth 3: eval < alpha - 1056
- Drops to qsearch, returns score if `score <= alpha`
- No depth guard — applies at any depth (but margin grows rapidly)
- Compare to ours: we use 400+100*depth (depth 1: 500, depth 2: 600)

### Reverse Futility Pruning (RFP)
- Conditions: `!IsPV && depth <= 11 && eval < TB_WIN`
- Margin: `max(87 * (depth - improving), 22)`
  - depth 1 not improving: max(87, 22) = 87
  - depth 1 improving: max(0, 22) = 22
  - depth 5 not improving: 87*5 = 435
  - depth 5 improving: 87*4 = 348
- Returns `(eval + beta) / 2` — **blended return** (dampened toward beta)
- Compare to ours: 85*d improving / 60*d not-improving, depth<=8, returns eval

### Null Move Pruning (NMP)
- Conditions: `cutNode && !excludedMove && lastMove != NONE && eval >= beta && staticEval + 22*depth - 208 >= beta && hasNonPawns(stm) && beta > TB_LOSS`
- **cutNode only** — NMP is restricted to expected cut nodes
- Reduction: `min((eval - beta) / 147, 4) + depth/3 + 4 + ttMoveNoisy`
  - Base R = **4** (vs our 3)
  - Extra +1 if TT best move is noisy (capture/promotion)
  - eval-beta contribution capped at 4 (vs our 3 with div 200)
  - depth/3 same as ours
- Mate score guard: returns `score < TB_WIN ? score : beta`
- Compare to ours: we allow NMP in non-cut nodes too, base R=3, no ttMoveNoisy bonus

### IIR (Internal Iterative Reduction)
- Conditions: `(IsPV || cutNode) && depth >= 2+2*cutNode && !ttMove`
  - PV nodes: depth >= 2
  - Cut nodes: depth >= 6
- Reduces depth by 1
- Compare to ours: we apply IIR at depth >= 4 regardless of node type

### ProbCut
- Conditions: `!IsPV && depth >= 5 && abs(beta) < TB_WIN && !(ttDepth >= depth-3 && ttScore < probcutBeta)`
- Beta: `beta + 190`
- **Two-stage**: QSearch first, then full negamax only if QSearch confirms
  ```cpp
  score = -qsearch(newPos, -probcutBeta, -probcutBeta+1, 0, ss+1);
  if (score >= probcutBeta)
      score = -negamax(newPos, -probcutBeta, -probcutBeta+1, depth-4, !cutNode, ss+1);
  ```
- **Dynamic SEE margin**: `(probcutBeta - staticEval) * 10 / 16` — adjusts capture threshold based on how much we need to gain
- TT move included only if it passes SEE at this margin
- Compare to ours: we have ProbCut (margin 170) but without QS pre-filter

### History Pruning (skipQuiets)
- Conditions: `isQuiet && history < -7471 * depth`
  - Depth 1: history < -7471
  - Depth 2: history < -14942
- Sets `skipQuiets = true` (prunes all remaining quiet moves)
- Compare to ours: we use -2000*depth (much looser)

### Late Move Pruning (LMP)
- Conditions: `!IsRoot && bestScore > TB_LOSS && hasNonPawns(stm)`
- Threshold: `(depth * depth + 3) / (2 - improving)`
  - Not improving: depth 1: 2, depth 2: 3, depth 3: 6, depth 4: 9
  - Improving: depth 1: 4, depth 2: 7, depth 3: 12, depth 4: 19
- Sets `skipQuiets = true`
- Compare to ours: similar formula `3+d^2` with improving adjustment

### Futility Pruning
- Conditions: `isQuiet && quietCount >= 1 && lmrDepth <= 10 && !pos.checkers`
- Margin: `159 + 153 * lmrDepth`
  - lmrDepth 1: 312, lmrDepth 2: 465, lmrDepth 3: 618
- Formula: `staticEval + margin <= alpha` => skipQuiets + continue
- Note: uses **lmrDepth** (reduced depth) not raw depth — more accurate
- Compare to ours: we use 80+80*depth (merged tighter margins)

### SEE Pruning
- Conditions: `!IsRoot && bestScore > TB_LOSS && hasNonPawns(stm)`
- Quiet margin: `-21 * lmrDepth^2` — quadratic, uses lmrDepth
  - lmrDepth 1: -21, lmrDepth 2: -84, lmrDepth 3: -189
- Capture margin: `-96 * depth` — linear, uses raw depth
  - depth 3: -288, depth 5: -480
- Compare to ours: we use -20*d^2 for quiets, similar magnitude

### QSearch Pruning
- **QS beta blending**: `(bestScore + beta) / 2` on stand-pat cutoff and on fail-high — both entry and exit blending
- **QS futility**: `stand_pat + 156`, combined with SEE(move, 1) for captures
- **QS SEE threshold**: `-32` — prunes captures that lose more than 32cp
- **QS move count limit**: break after 3 captures when not in check
- **QS quiet check break**: when in check and a quiet move is seen, break
- **QS quiet checks**: generated at depth==0 only (first ply of QS)

---

## 5. Extensions

### Singular Extensions
- Depth guard: `depth >= 5` (vs our 10 — much lower threshold)
- Ply limiter: `ply < 2 * rootDepth` — prevents explosive SE growth
- Conditions: `!IsRoot && !excludedMove && move == ttMove && abs(ttScore) < TB_WIN && ttBound & FLAG_LOWER && ttDepth >= depth-3`
- Singular beta: `ttScore - (depth * 64) / 64 = ttScore - depth`
  - Wait — SBetaMargin = 64, so: `ttScore - (depth * 64) / 64 = ttScore - depth`
  - Actually: `ttScore - depth * SBetaMargin / 64 = ttScore - depth * 1 = ttScore - depth`
  - This is a very tight margin
- Singular depth: `(depth - 1) / 2`
- **Single extension (+1)**: when `seScore < singularBeta`
- **Double extension (+2)**: when `!IsPV && seScore < singularBeta - 13` (DoubleExtMargin=13)
- **Triple extension (+3)**: when additionally `isQuiet && seScore < singularBeta - 121` (TripleExtMargin=121)
- **Multi-cut**: when `singularBeta >= beta`, return singularBeta
- **Negative extension (-3+IsPV)**: when `ttScore >= beta` (non-PV: -3, PV: -2)
- **Cut-node negative extension (-2)**: when none of the above apply at a cut node

### No Check Extension
- Obsidian does NOT have unconditional check extension (unlike many engines)
- This is notable — the check handling is purely through the evasion move generation path

### Compared to GoChess
| Feature | Obsidian | GoChess |
|---------|----------|---------|
| SE min depth | 5 | 10 |
| SE margin | ttScore - depth | ttScore - 3*depth |
| Ply limiter | ply < 2*rootDepth | None |
| Double ext | +2 at margin-13 | None |
| Triple ext | +3 at margin-121 (quiet) | None |
| Multi-cut | return singularBeta | None |
| Negative ext | -3+isPV / -2 cutNode | None |
| Check extension | None | None (removed, -11 Elo) |

**Our SE is broken (-58 to -85 Elo). Obsidian's SE is a reference implementation of what works.** The ply limiter `ply < 2*rootDepth` is critical — without it, double/triple extensions cause explosive tree growth.

---

## 6. LMR (Late Move Reductions)

### Table
Single table (not split captures/quiets):
```cpp
lmrTable[i][m] = 0.99 + log(i) * log(m) / 3.14
```
Base = LmrBase/100 = 0.99, Divisor = LmrDiv/100 = 3.14.

Compare to ours: split tables with capture C=1.80, quiet C=1.50.

### Application
- Conditions: `depth >= 2 && seenMoves > 1 + 2*IsRoot`
  - Non-root: seenMoves > 1 (starts at move 2)
  - Root: seenMoves > 3 (starts at move 4)

### Reduction Adjustments
```
R = lmrTable[depth][seenMoves]
R -= history / (isQuiet ? 9621 : 5693)        // history-based
R -= complexity / 120                          // complexity-based (correction magnitude)
R -= (newPos.checkers != 0)                    // -1 if move gives check
R -= (ttDepth >= depth)                        // -1 for deep TT entry
R -= ttPV + IsPV                               // -1 for PV, -1 for ttPV
R += ttMoveNoisy                               // +1 if TT move is noisy (MERGED in GoChess)
R += !improving                                // +1 if not improving
if (cutNode) R += 2 - ttPV                     // +2 at cut nodes, -1 if ttPV
```

Clamped: `reducedDepth = clamp(newDepth - R, 1, newDepth + 1)`
- Note: upper bound is `newDepth + 1` — allows R to go negative (extending by 1)

### DoDeeper / DoShallower
After LMR re-search beats alpha:
```cpp
newDepth += (score > bestScore + ZwsDeeperMargin(43) + 2 * newDepth);
newDepth -= (score < bestScore + ZwsShallowerMargin(11));
```
- DoDeeper: score must exceed bestScore + 43 + 2*newDepth (depth-adaptive)
- DoShallower: score below bestScore + 11

### LMR Re-Search History Bonus
After ZWS re-search (whether it extends or not):
```cpp
bonus = score <= alpha ? -statMalus(newDepth) : score >= beta ? statBonus(newDepth) : 0;
addToContHistory(pos, bonus, move, ss);
```

### Compared to GoChess
| Feature | Obsidian | GoChess |
|---------|----------|---------|
| LMR table | Single (base 0.99, div 3.14) | Split (cap 1.80, quiet 1.50) |
| History in LMR | quiet/9621, cap/5693 | /5000 |
| Complexity in LMR | abs(corrected-raw)/120 | None |
| Check gives -1 | Yes | No |
| ttDepth gives -1 | Yes (when >= depth) | No |
| ttPV gives -1 | Yes | No |
| PV gives -1 | Yes | Yes |
| Cut node +2 | Yes (2-ttPV) | No |
| DoDeeper | 43+2*newDepth | Tested, -13.7 Elo |
| DoShallower | 11 | Tested, -13.7 Elo |
| Re-search hist bonus | cont-hist only | No |

---

## 7. Move Ordering

### Staged Move Picker
Stages: TT -> Good Captures -> Killer -> Counter -> Quiets -> Bad Captures

**Good capture threshold**: Dynamic SEE — `- move.score / 32` where score is MVV + capHist. This means well-ordered captures need a lower SEE threshold, poorly ordered ones need higher. Effectively, captures with high history/MVV are treated more leniently.

### Capture Scoring
```
score = PIECE_VALUE[captured] * 16 + isPromotion * 16384 + capHist[pieceTo][captured]
```
MVV-based with capture history. No LVA component directly — the attacker type only matters via capture history.

### Quiet Scoring
```
score = threatScore
      + mainHistory[stm][from_to]
      + pawnHistory[pawnKey % 1024][pieceTo]
      + contHistory(ply-1)[chIndex]
      + contHistory(ply-2)[chIndex]
      + contHistory(ply-4)[chIndex]
      + contHistory(ply-6)[chIndex]
```

**Threat-aware scoring** (computed via `calcThreats()`):
- Queen escaping rook attack: +32768. Queen entering rook attack: -32768
- Rook escaping minor attack: +16384. Rook entering minor attack: -16384
- Minor escaping pawn attack: +16384. Minor entering pawn attack: -16384

This is a highly differentiated threat system — piece-type-specific, with large bonuses/penalties that dominate over history.

### History Tables
- **Butterfly history**: `mainHistory[2][4096]` (color x from-to)
- **Pawn history**: `pawnHistory[1024][pieceTo]` — indexed by `pawnKey % 1024`
- **Capture history**: `captureHistory[pieceTo][pieceType]`
- **Continuation history**: `contHistory[2][pieceTo][pieceTo]` — [isCap][prev pieceTo][curr pieceTo]
  - 4 plies used in scoring: 1, 2, 4, 6
  - Updates to plies 1, 2 at full bonus; plies 4, 6 at **half bonus**
- **Counter move**: `counterMoveHistory[pieceTo]` — indexed by previous move's piece-to

### History Bonus/Malus
```cpp
statBonus(d) = min(175*d + 15, 1409)
statMalus(d) = min(196*d - 25, 1047)
```
**Strong fail-high boost**: when `bestScore > beta + 95`, bonus/malus use `depth+1`:
```cpp
bonus = statBonus(depth + (bestScore > beta + StatBonusBoostAt(95)));
```

History gravity: `history += value - history * abs(value) / 16384`

### Quality-Gated History Updates
```cpp
if (depth > 3 || quietCount)  // Ethereal-credited
    updateMoveHistories(pos, bestMove, bonus, ss);
```
Skip promoting the best move's history if it was a trivially quick low-depth cutoff with no competing moves tried.

### Compared to GoChess
| Feature | Obsidian | GoChess |
|---------|----------|---------|
| Stages | TT/GoodCap/Killer/Counter/Quiet/BadCap | Same |
| Pawn history | Yes (1024 entries) | No |
| Cont-hist plies | 1,2,4,6 | 1,2 |
| Threat-aware scoring | Per-piece-type escape/enter | None |
| Cap ordering | MVV*16 + capHist | MVV-LVA + capHist (merged) |
| Dynamic SEE threshold | -score/32 | Fixed |
| History bonus boost | +1 depth when score > beta+95 | None |
| Quality-gated update | depth>3 or quietCount | No |

---

## 8. Transposition Table

### Structure
- 3 entries per bucket (EntriesPerBucket = 3)
- Bucket size: 32 bytes (3 * 10 bytes + 2 padding)
- Entry: 10 bytes — key16(2) + staticEval(2) + agePvBound(1) + depth(1) + move(2) + score(2)
- Index: multiply-shift `(uint128(key) * bucketCount) >> 64` — Fibonacci hashing
- Age: 5-bit (32 ages), stored in agePvBound alongside PV flag and bound bits
- Multithreaded clear using multiple std::thread workers

### Replacement Policy
Replace if ANY of:
- Bound is EXACT
- Different hash (new position)
- Age differs (any age distance)
- `new_depth + 4 + 2*isPV > old_depth` (generous depth threshold)

### Probe — Search Cutoff
```cpp
if (!IsPV && !excludedMove && ttScore != NONE
    && ttDepth >= depth + (ttScore >= beta)     // stricter for fail-high
    && (cutNode == (ttScore >= beta))            // node-type guard
    && canUseScore(ttBound, ttScore, beta)
    && halfMoveClock < 90)                       // avoid 50mr tricks
```
**Notable features:**
1. `ttDepth >= depth + (ttScore >= beta)` — fail-high requires 1 extra ply of depth
2. `cutNode == (ttScore >= beta)` — **node-type guard**: cut nodes only accept fail-highs, all-nodes only accept fail-lows
3. `halfMoveClock < 90` — avoid using TT near 50-move draw boundary

### TT Cutoff Cont-Hist Malus
When TT cutoff gives beta cutoff and opponent tried <= 3 moves:
```cpp
if (ttScore >= beta && prevSq != SQ_NONE && !(ss-1)->playedCap && (ss-1)->seenMoves <= 3)
    addToContHistory(chIndex, -statMalus(depth), ss-1);
```
Penalizes the opponent's last move in continuation history — "your quiet move led to a known losing position."

### 50-Move Zobrist Key
```cpp
posTtKey = pos.key ^ ZOBRIST_50MR[pos.halfMoveClock];
```
ZOBRIST_50MR is zero for halfMoveClock 0-13, then changes every 8 half-moves (same key for a block of 8). This differentiates positions that are near/far from 50-move draw in the TT.

### Compared to GoChess
| Feature | Obsidian | GoChess |
|---------|----------|---------|
| Entries/bucket | 3 | 4 |
| Entry size | 10 bytes | 16 bytes (packed atomic) |
| Lockless | No (not needed, copy-on-search) | Yes (XOR-verified atomics) |
| Node-type guard | Yes | No |
| Depth bonus for fail-high | +1 ply | No |
| 50mr key modification | Yes | No |
| TT cutoff CH malus | Yes | No |
| PV flag in TT | Yes | No |
| Age bits | 5 (32 ages) | 2 (4 generations) |

---

## 9. Time Management

### Base Allocation
```cpp
int mtg = movestogo ? min(movestogo, 50) : 50;
timeLeft = max(1, time + inc*(mtg-1) - overhead*(2+mtg));
if (!movestogo)
    optScale = min(0.025, 0.214 * time / timeLeft);
else
    optScale = min(0.95/mtg, 0.88 * time / timeLeft);
optimumTime = optScale * timeLeft;
maxTime = time * 0.8 - overhead;
```

### Soft Time Scaling (Triple Factor)
At depth >= 4, main thread computes:
```cpp
notBestNodes = 1.0 - (bmNodes / nodesSearched);
nodesFactor = 0.63 + notBestNodes * 2.00;            // [0.63, 2.63]
stabilityFactor = 1.71 - searchStability * 0.08;     // [1.07, 1.71]  (stability 0-8)
scoreLoss = 0.86 + 0.010*(prevScore-score) + 0.025*(searchPrevScore-score);
scoreFactor = clamp(scoreLoss, 0.81, 1.50);
if (elapsed > stabilityFactor * nodesFactor * scoreFactor * optimumTime)
    stop;
```

Three multiplicative factors:
1. **Node distribution**: If best move uses few nodes (unstable), spend more time
2. **Best move stability**: Counter 0-8, decays stability factor from 1.71 to 1.07
3. **Score loss**: If score dropped from previous iteration or previous search, extend time (up to 1.5x)

The `searchPrevScore` is the score from the PREVIOUS search (different position) — provides cross-position context.

### Compared to GoChess
| Feature | Obsidian | GoChess |
|---------|----------|---------|
| Soft time base | time*0.025 or 0.214*time/timeLeft | time/20 + 3*inc/4 |
| Hard time | time*0.8 - overhead | time/5 + 3*inc/4 |
| Node-based scaling | Yes (continuous) | No |
| Best move stability | 0-8 counter, multiplicative | Instability factor 200 |
| Score loss scaling | prev iter + prev search | 2.0x/1.5x/1.2x tiered (UPDATE: merged) |
| Triple factor | nodesFactor * stabilityFactor * scoreFactor | Single factor |

---

## 10. Lazy SMP

### Thread Model
- All threads run full ID from depth 1
- Shared TT (only shared state)
- Per-thread: Position (copy-on-search), SearchInfo stack, all history tables, accumulator stack, FinnyTable
- Main thread handles time checks and info output
- **No asymmetric depths** — all threads search the same depth sequence

### Best Move Voting
When multiple threads complete:
```cpp
votes[move] += (score - minScore + 9) * completeDepth;
```
Best thread selected by highest vote for its best move. TB wins override voting.

### Thread-Safety
- `Threads::isSearchStopped()` uses relaxed atomic
- Position is copied (not shared) — `Position newPos = pos` before each `doMove`
- No lockless TT needed because positions are copied, not shared references

---

## 11. Unique / Notable Techniques

### 1. Pairwise Multiplication in NNUE FT
The FT activation multiplies first-half * second-half neurons, producing uint8 output. This is fundamentally different from standard CReLU and captures feature interactions at the FT level.

### 2. NNZ-Sparse L1
Tracking non-zero FT outputs and only multiplying those in L1 is a significant optimization. Typical chess positions activate only ~30-50% of features, so this nearly doubles L1 throughput.

### 3. Score-Adaptive Aspiration Window
`delta = 6 + avgScore^2 / 13000` — wider windows for large scores, tight for small. This is elegant and avoids repeated fail-highs in winning/losing positions.

### 4. NMP restricted to cutNode
Most engines allow NMP at any non-PV node. Obsidian restricts it to expected cut nodes only. This is aggressive but saves NMP overhead at all-nodes where it's unlikely to succeed.

### 5. Copy-on-Search (no Undo)
```cpp
Position newPos = pos;
newPos.doMove(move, dirtyPieces);
```
No UnmakeMove at all — position is copied before each move. Simpler code, no undo stack bugs. Trade-off: more memory pressure but likely better cache behavior with modern copy-on-write.

### 6. 50-Move Zobrist TT Keys
Differentiating positions by 50-move clock state in the TT prevents stale evaluations near draw boundaries.

### 7. Quality-Gated History Updates (Ethereal credit)
Don't boost the history of a move that got a trivial instant cutoff at low depth with no competing moves — it tells us nothing about the move's quality.

### 8. Dynamic ProbCut SEE Margin
`seeMargin = (probcutBeta - staticEval) * 10 / 16` — only consider captures that could plausibly bridge the gap between current eval and probcut beta.

---

## 12. Parameter Comparison Table

| Feature | Obsidian | GoChess |
|---------|----------|---------|
| **Razoring** | 352*d, any depth | 400+100*d, depth<=3 |
| **RFP margin** | max(87*(d-imp), 22), d<=11 | 85*d imp / 60*d not, d<=8 |
| **RFP return** | (eval+beta)/2 | eval |
| **NMP base R** | 4 | 3 |
| **NMP depth div** | 3 | 3 |
| **NMP eval div** | 147 | 200 |
| **NMP guard** | cutNode + eval+22*d-208>=beta | None (always try) |
| **NMP ttMoveNoisy** | +1 R | No |
| **IIR** | PV: d>=2, cut: d>=6 | d>=4 |
| **ProbCut beta** | beta+190 | beta+170 |
| **ProbCut QS pre-filter** | Yes | No |
| **ProbCut SEE** | Dynamic (eval-relative) | Fixed |
| **History pruning** | -7471*d | -2000*d |
| **LMP** | (d^2+3)/(2-imp) | 3+d^2 (+50% imp) |
| **Futility** | 159+153*lmrD, lmrD<=10 | 80+80*d |
| **SEE quiet** | -21*lmrD^2 | -20*d^2 |
| **SEE capture** | -96*d | Similar |
| **SE depth** | >=5 | >=10 |
| **SE margin** | ttScore - depth | ttScore - 3*depth |
| **SE ply limiter** | ply<2*rootDepth | None |
| **SE double ext** | margin-13 | None |
| **SE triple ext** | margin-121 (quiet) | None |
| **SE multi-cut** | return singularBeta | None |
| **SE negative ext** | -3+isPV / -2 cutNode | None |
| **LMR base** | 0.99 | Split (cap 1.80, quiet 1.50) |
| **LMR div** | 3.14 | Split |
| **LMR history div** | quiet 9621 / cap 5693 | 5000 |
| **LMR complexity** | -abs(corr)/120 | None |
| **LMR check** | -1 | No |
| **LMR ttDepth** | -1 if >=depth | No |
| **LMR ttPV** | -1 | No |
| **LMR cutNode** | +2-ttPV | No |
| **LMR ttMoveNoisy** | +1 | +1 (merged) |
| **DoDeeper** | 43+2*newDepth | Tested -13.7 |
| **DoShallower** | 11 | Tested -13.7 |
| **Asp delta** | 6 + score^2/13000 | 15 |
| **Asp fail-high** | depth-- | Tested (bug) |
| **QS beta blend** | (bestScore+beta)/2 | (bestScore+beta)/2 (merged) |
| **QS futility** | stand_pat+156 | None (SEE filter) |
| **QS SEE** | -32 | Similar |
| **QS move limit** | 3 (no check) | None |
| **Correction hist** | 4 tables | 1 (pawn only) |
| **Eval history** | Opponent move feedback | None |
| **Pawn history** | Yes (1024) | No |
| **Cont-hist plies** | 1,2,4,6 | 1,2 |
| **Cont-hist write plies** | 1,2 full; 4,6 half | 1,2 |
| **Threat-aware ordering** | Per-piece escape/enter | None |
| **FinnyTable** | Yes | Yes (UPDATE: merged) |
| **Cuckoo repetition** | Yes | No |
| **50mr TT key** | Yes | No |
| **TT node-type guard** | Yes | No |
| **TT PV flag** | Yes | No |
| **TT cutoff CH malus** | Yes | No |
| **Material eval scaling** | (230+phase)/330 | None |
| **50mr eval decay** | (200-hmClock)/200 | None |

---

## 13. Ideas Worth Testing from Obsidian

### HIGH PRIORITY (address known GoChess weaknesses)

1. **Fix Singular Extensions using Obsidian's framework**: depth>=5, margin=depth, ply<2*rootDepth limiter, double/triple ext, multi-cut return singularBeta, negative ext -3+isPV / -2 cutNode. Our SE is -58 to -85 Elo due to missing ply limiter and over-wide margin.

2. **4-table correction history**: Per-color non-pawn Zobrist keys (already maintained in Position via putPiece/removePiece), continuation correction. Obsidian shows clean implementation. Our single pawn table is insufficient. *11+ engines have this.*

3. **Node-based time management**: Triple-factor (nodesFrac * stability * scoreLoss). Our time management is basic. *6+ engines.*

4. **NMP cutNode restriction**: Only allow NMP at cut nodes. Simple change, may reduce NMP failures at all-nodes.

### MEDIUM PRIORITY (proven in Obsidian, moderate expected gain)

5. **Pawn history** in move ordering: `pawnHistory[pawnKey % 1024][pieceTo]`. Captures pawn-structure-dependent move quality. *2+ engines.*

6. **Complexity-adjusted LMR**: `R -= abs(correctedEval - rawEval) / 120`. High correction = uncertain = search deeper. *Already in SUMMARY.*

7. **Threat-aware quiet scoring**: Per-piece-type escape/enter bonuses (+32768/-32768 for queen, +16384/-16384 for rook/minor). *12+ engines have threat-aware ordering.*

8. **TT cutoff cont-hist malus**: When TT cutoff at beta and opponent tried <=3 moves, penalize opponent's last quiet move. Novel and cheap.

9. **RFP blended return** `(eval+beta)/2` instead of raw eval. Dampens RFP over-confidence.

10. **Cont-hist plies 4 and 6**: Read at full weight in scoring, write at half bonus. Our test of contHist4 was -58 Elo but may have used wrong weight.

### LOWER PRIORITY (interesting but less certain)

11. **Score-adaptive aspiration window**: `delta = 6 + avgScore^2 / 13000`. Avoids repeated fail-highs in winning positions.

12. **50-move Zobrist TT keys**: Differentiate positions by halfmove clock. Prevents stale evaluations near 50MR boundary.

13. **Quality-gated history updates**: Skip history boost when `depth <= 3 && !quietCount` (trivial cutoff).

14. **Dynamic ProbCut SEE**: `(probcutBeta - staticEval) * 10/16` instead of fixed margin.

15. **NMP +1R for ttMoveNoisy**: Already similar via our TT noisy detection, but this adds it directly to NMP reduction.

16. **Copy-on-search (no UnmakeMove)**: Architectural change, likely not worth the effort for GoChess.

### NNUE Architecture Ideas

17. ~~**Pairwise multiplication FT activation**~~: **(UPDATE 2026-03-21: GoChess now has pairwise multiplication with SIMD support)**

18. **NNZ-sparse L1**: Track non-zero FT outputs, skip zeros in L1 matmul. Significant NPS gain.

19. **Dual activation L2** (linear + squared): Cheap way to capture non-linear interactions.

20. ~~**FinnyTable**~~: **(UPDATE 2026-03-21: GoChess now has Finny tables, merged)**

---

## 14. What Makes Obsidian 426 Elo Stronger

The Elo gap between Obsidian and GoChess-v5 is large. The primary sources:

1. **NNUE quality** (~200+ Elo): 1536-wide FT with pairwise multiplication, NNZ-sparse L1, dual activation L2, 13 king buckets, FinnyTable. **(UPDATE 2026-03-21: GoChess v5 now has pairwise mul, SCReLU, dynamic width, and Finny tables. Still lacks NNZ-sparse L1 and dual activation. Net quality gap narrowing as training scales up.)**

2. **Singular extensions** (~50-80 Elo): Full SE framework with double/triple/multi-cut/negative vs our broken SE. This is pure search Elo we're leaving on the table.

3. **Correction history depth** (~20-40 Elo): 4 tables vs 1. Better static eval leads to better pruning decisions everywhere.

4. **Time management** (~15-25 Elo): Triple-factor node/stability/score scaling vs our basic approach.

5. **Move ordering** (~15-25 Elo): Pawn history + threat-aware scoring + deeper cont-hist + dynamic SEE.

6. **Accumulated pruning refinements** (~30-50 Elo): NMP cutNode guard, complexity-adjusted LMR, quality-gated updates, eval history, 50mr TT keys, node-type TT guard, etc.

**(UPDATE 2026-03-21: NNUE architecture migration is largely complete -- GoChess v5 has pairwise, wider FT, SCReLU, Finny tables. Still need NNZ sparse L1 and dual activation.)** The remaining biggest priorities are NNZ-sparse L1 and fixing singular extensions.
