# Weiss Chess Engine - Crib Notes

Source: `~/chess/engines/weiss/src/`
Author: Terje Kirstihagen
Rating: #19 in our RR at -2 Elo (106 above GoChess-v5)
Eval: Classical (tapered HCE with mobility, king safety, passed pawns, threats). No NNUE.

---

## 1. Search Architecture

Standard iterative deepening with aspiration windows, PVS, and Lazy SMP. Written in C with pthread-based threading. Uses `setjmp`/`longjmp` for time-abort (clean unwind from deep search).

### Iterative Deepening
- Main thread: depth 1 to `Limits.depth` (default MAX_PLY=100)
- Helper threads: depth 1 to MAX_PLY (no depth limit)
- Time check and abort via `longjmp(thread->jumpBuffer, true)` on time expiry
- Soft time limit checked after each iteration with node-ratio scaling (see Time Management)

### Aspiration Windows
- Dynamic initial delta: `9 + prevScore * prevScore / 16384` (score-dependent -- wider for large scores)
- On fail-low: widen alpha by delta, contract beta: `beta = (alpha + 3*beta) / 4`
- On fail-high: widen beta by delta, **reduce depth by 1** (only for non-terminal scores)
- Delta growth: `delta += delta / 3` (multiply by ~1.33 each iteration)
- **Trend bonus**: `x = CLAMP(prevScore/2, -32, 32)`, applied as S(x, x/2) to eval. Biases eval in direction of last score.
- Compare to ours: we use fixed delta=15, growth 1.5x. **(UPDATE 2026-03-21: GoChess now has fail-low contraction (3a+5b)/8 and fail-high contraction (5a+3b)/8.)** Their score-dependent delta is novel. Trend bonus is unique.

### Draw Detection
- Repetition: **any** two-fold (checks all `i += 2` up to `min(rule50, histPly)`)
- Upcoming repetition: `HasCycle(pos, ply)` -- Cuckoo-based upcoming repetition detection (alpha < 0 only)
- 50-move rule: `rule50 >= 100`
- Draw randomization: `8 - (nodes & 0x7)` -- small positive score +1 to +8

### Check Extension
- `extension = MAX(extension, 1)` when in check, after singular extension logic
- Means singular extension takes priority: a singular move in check gets +2 (double ext)

---

## 2. Pruning Techniques

### Pruning Gate (doPruning flag)
**Unique technique**: Weiss has a `doPruning` flag that delays enabling forward pruning until either:
- `usedTime >= optimalUsage / 64` (time-based), OR
- `depth > 2 + optimalUsage / 270` (depth-based), OR
- Using node-time mode
- In infinite mode: after 1000ms or 5000ms (varies by context)

This prevents aggressive pruning in very early iterations where eval is unstable. The flag is checked before RFP, NMP, and all per-move pruning.

Compare to ours: we have no such gating. This is listed in SUMMARY.md as idea #43 (Tier 3).

### Pruning Skip Conditions
All pruning is skipped when: `inCheck || pvNode || !doPruning || ss->excluded || isTerminal(beta) || lastMoveNullMove`
- Note: skips after null move (`(ss-1)->move == NOMOVE`), preventing NMP chains

### Reverse Futility Pruning (RFP)
- Depth guard: `depth < 7`
- Margin: `77 * (depth - improving)` -- improving flag reduces effective depth by 1
- Additional guard: history of last move: `(ss-1)->histScore / 131` subtracted from margin
- **TT move guard**: `!ttMove || GetHistory(thread, ss, ttMove) > 6450`
  - If there IS a TT move with low history (<=6450), RFP is skipped
  - This prevents RFP when there's a known quiet TT move that might be strong
- Formula: `eval - 77*(depth-improving) - (ss-1)->histScore/131 >= beta`
- Compare to ours: we use 85*d (improving) / 60*d (not improving), depth<=8. They use 77*(d-imp), depth<7. Their histScore adjustment and TT move guard are novel.

### Null Move Pruning (NMP)
- Conditions: `eval >= beta && eval >= staticEval && staticEval >= beta + 138 - 13*depth`
  - The `staticEval >= beta + 138 - 13*depth` is a **depth-dependent verification gate** -- at low depth, staticEval must be well above beta; at high depth, the threshold relaxes
- Additional guard: `(ss-1)->histScore < 28500` -- skip NMP if opponent's last move had very high history
- Material guard: `nonPawnCount[stm] > (depth > 8)` -- at depth > 8, need at least 2 non-pawn pieces
- Reduction: `4 + depth/4 + MIN(3, (eval-beta)/227)`
- Returns raw score (no beta clamping) if `>= beta`, but filters terminal wins to just beta
- Compare to ours: we use `3 + depth/3 + MIN(3, (eval-beta)/200)`. Their base R=4 (vs our 3) and divisor d/4 (vs our d/3) means less reduction overall. Their histScore gate is novel.

### Internal Iterative Reduction (IIR)
- Two separate conditions:
  1. PV node: `pvNode && depth >= 3 && !ttMove` -> `depth--`
  2. Cut node: `cutnode && depth >= 8 && !ttMove` -> `depth--`
- These are **additive** -- a PV node at depth 8 with no TT move doesn't get double IIR (pvNode skips pruning so cutnode branch is never hit on PV)
- Compare to ours: we have IIR at depth >= 4 with no TT move. Their PV/cutnode split is more targeted.

### ProbCut
- Threshold: `beta + 200`
- Depth guard: `depth >= 5`
- TT guard: `!ttHit || ttScore >= probCutBeta`
- **Two-stage**: QS first, then `depth-4` search only if QS passes
- SEE threshold for captures: `probCutBeta - staticEval` (dynamic, based on margin needed)
- Only searches moves that pass NOISY_GOOD stage (SEE-filtered)
- Return adjustment: `score - 160` for non-terminal wins (dampens ProbCut score)
- Compare to ours: we have ProbCut margin 170, gate 85, depth >= 5. Their two-stage QS pre-filter is in SUMMARY.md as idea #6 (Tier 1). Their score dampening (`-160`) is novel.

### Late Move Pruning (LMP)
- Formula: `moveCount > (improving ? 2 + depth*depth : depth*depth/2)`
- Sets `mp.onlyNoisy = true` (skips all remaining quiets, but still searches captures)
- Compare to ours: we use `3 + d^2` (+50% improving). Their formula: `depth*depth/2` (not improving) vs our `3+d^2`, and `2+depth*depth` (improving). At depth 3: theirs = 4/11, ours = 12/18.

### History Pruning
- Conditions: `quiet && lmrDepth < 3 && histScore < -1024 * depth`
- Note: uses `lmrDepth` (reduced depth) not raw depth
- Compare to ours: we use `-2000*depth`. They use `-1024*depth` but on lmrDepth (smaller), so effective threshold is tighter.

### SEE Pruning
- Conditions: `lmrDepth < 7`
- Single threshold: `-73 * depth` for ALL moves (quiet and noisy)
- Compare to ours: we have separate quiet/tactical margins. They use one uniform margin.

### QSearch Pruning
- Futility: `eval + 165` is the futility value
  - `futility + PieceValue[EG][captured] <= alpha && !promotion` -> skip (updates bestScore)
- SEE filter: `futility <= alpha && !SEE(pos, move, 1)` -> skip
- Stage filter: `mp.stage > NOISY_GOOD` -> break (stop searching bad captures entirely)
- Mate distance pruning in QS (applied before moves)
- Upcoming repetition detection in QS (`HasCycle`)
- Compare to ours: our QS delta = 240. Their futility margin of 165 is tighter. They also have the SEE(1) filter.

---

## 3. Extensions

### Singular Extensions
- Depth guard: `depth > 4` (i.e., depth >= 5, much lower than many engines)
- Conditions: `move == ttMove && !excluded && ttDepth > depth-3 && ttBound != UPPER && !isTerminal(ttScore)`
- Singular beta: `ttScore - depth * (2 - pvNode)` -- tighter on PV nodes (margin = depth vs 2*depth)
- Singular depth: `depth / 2`
- **Single extension (+1)**: score < singularBeta
- **Double extension (+2)**: `!pvNode && score < singularBeta - 1 && doubleExtensions <= 5`
  - Hard cap of 5 double extensions per line (via `ss->doubleExtensions` counter)
- **Multi-cut**: `singularBeta >= beta` -> return singularBeta immediately
- **Negative extension (-1)**: `ttScore >= beta` (not singular, but likely still beats beta)
- Extension limiter: `ss->ply >= thread->depth * 2` skips all extensions
- Compare to ours: our SE is currently problematic. Key differences:
  - Weiss depth >= 5 vs our 10
  - Weiss margin 2*depth (non-PV) / depth (PV) vs our 3*depth
  - Weiss has double ext cap (5), multi-cut return, negative ext, ply limiter
  - These are all recommended fixes in SUMMARY.md idea #4

### Check Extension
- `extension = MAX(extension, 1)` unconditionally when in check
- Applied after singular extension, so singular + check can give +2

### No other extensions
- No recapture extension, no passed pawn extension

---

## 4. LMR (Late Move Reductions)

### Table Initialization
Two separate tables for captures and quiets:
- **Captures**: `0.38 + log(depth) * log(moves) / 3.76`
- **Quiets**: `2.01 + log(depth) * log(moves) / 2.32`

Compare to ours: we use separate tables with C_cap=1.80 / C_quiet=1.50. Weiss has a much higher quiet base (2.01 vs our 1.50) -- very aggressive quiet reduction. Lower capture base (0.38 vs ~0.69 in our formula).

### Application Conditions
- `depth > 2 && moveCount > MAX(1, pvNode + !ttMove + root + !quiet) && doPruning`
- The threshold `MAX(1, pvNode + !ttMove + root + !quiet)` means:
  - Non-PV with TT move at non-root with quiet: LMR at move 2+
  - PV with TT move at root with noisy: LMR at move 5+
  - Many combinations in between

### Reduction Adjustments
- `-= histScore / 8870` (history-based, uses combined quiet+cont history)
- `-= pvNode` (reduce less in PV)
- `-= improving` (reduce less when improving)
- `+= moveIsCapture(ttMove)` (reduce more when TT move is noisy -- **same as our TT noisy detection**)
- `+= nonPawnCount[opponent] < 2` (reduce more in endgame with few opponent pieces)
- `+= 2 * cutnode` (reduce more at cut nodes -- **+2 per cut node, very aggressive**)
- Clamped to `[1, newDepth]`

### DoDeeper Search
After LMR re-search beats alpha:
- `deeper = score > bestScore + 1 + 6 * (newDepth - lmrDepth)`
- If deeper is true: `newDepth += 1` before full re-search
- Reduction-proportional threshold: the more the move was reduced, the higher the bar to deepen
- Compare to ours: we don't have doDeeper/doShallower. Listed in SUMMARY.md idea #1 (Tier 1).

### Continuation History Update on Re-search
- If quiet and re-search result `<= alpha || >= beta`: update cont histories with bonus/malus
- This gives history feedback even for LMR re-searches

### What they DON'T have:
- No doShallower (only doDeeper)
- No TT-PV flag adjustment
- No eval-distance-based adjustment

---

## 5. Move Ordering

### Staged Move Picker
Stages: TTMOVE -> GEN_NOISY -> NOISY_GOOD -> KILLER -> GEN_QUIET -> QUIET -> NOISY_BAD

### Score Hierarchy

1. **TT move**: returned directly (not scored)
2. **Good captures**: `captureHistory[piece][to][victimType] + PieceValue[MG][captured]`
   - Filtered by: `score > 11046` (always good), OR `score > -9543 && SEE(move, threshold)` (decent + SEE pass)
   - Moves that fail get saved as "bad" for NOISY_BAD stage
3. **Killer move**: single killer per ply (returned directly)
4. **Quiet moves**: `quietHistory[stm][from][to] + pawnHistory[pawnKey%512][piece][to] + contHist(1) + contHist(2) + contHist(4)`
5. **Bad captures**: searched last in NOISY_BAD stage

### Partial Insertion Sort
- After scoring, uses partial insertion sort with threshold `-750 * depth`
- Only moves above threshold are fully sorted; the rest are left in generation order
- This is a performance optimization: at shallow depth, don't bother sorting all moves

### History Tables

**Butterfly history**: `history[2][64][64]` -- color, from, to
- Gravity: `entry += bonus - entry * abs(bonus) / 4373`
- Bonus: `MIN(2418, 251*depth - 267)`
- Malus: `-MIN(693, 532*depth - 163)` (asymmetric -- malus is smaller than bonus!)

**Pawn history**: `pawnHistory[512][16][64]` -- pawn structure hash % 512, piece, to square
- Gravity divisor: 8663 (slower adaptation than butterfly)
- Used in both scoring and updates

**Capture history**: `captureHistory[16][64][8]` -- piece, to, captured piece type
- Gravity divisor: 14387 (very slow adaptation)

**Continuation history**: `continuation[2][2][16][64]` -- [inCheck][isCapture] x piece x to
- 4 tables indexed by (inCheck, isCapture) of the move that led to this position
- Plies used: 1, 2, **4** (skips ply 3!)
- Gravity divisor: 16384 (slowest adaptation)
- **Updates**: At depth > 2 for bonus; always for malus. Cont-hist at plies 1, 2, 4.

**Single killer**: 1 killer per ply (most engines use 2)

### Notable: No Counter-Move Heuristic
Weiss has NO counter-move table. Only 1 killer per ply. This is compensated by the richer history tables (pawn history, deeper cont-hist).

Compare to ours: we have 2 killers + counter-move. They have 1 killer + pawn history + cont-hist ply 4.

---

## 6. Correction History (Multi-Source)

This is Weiss's most sophisticated feature. **10 correction history components** with tuned weights:

### Components
1. **Pawn correction**: `pawnCorrHistory[2][16384]` indexed by pawnKey, weight 5868
2. **Minor piece correction**: `minorCorrHistory[2][16384]` indexed by minorKey, weight 7217
3. **Major piece correction**: `majorCorrHistory[2][16384]` indexed by majorKey, weight 4416
4. **Non-pawn correction (per color)**: `nonPawnCorrHistory[2][2][16384]` indexed by nonPawnKey[color], weight 7025 each
5. **Continuation correction plies 2-7**: `contCorrHistory[16][64]` indexed by piece/to of move at that ply
   - Ply 2: weight 4060
   - Ply 3: weight 3235
   - Ply 4: weight 2626
   - Ply 5: weight 3841
   - Ply 6: weight 3379
   - Ply 7: weight 2901

### Blending Formula
```
correction = (5868*pawn + 7217*minor + 4416*major
            + 7025*(nonPawnW + nonPawnB)
            + 4060*cont2 + 3235*cont3 + 2626*cont4
            + 3841*cont5 + 3379*cont6 + 2901*cont7) / 131072
```

### Zobrist Keys
Position struct has separate incremental Zobrist keys:
- `pawnKey`: pawn positions
- `minorKey`: minor piece (N/B) positions
- `majorKey`: major piece (R/Q) positions
- `nonPawnKey[WHITE]`, `nonPawnKey[BLACK]`: per-color non-pawn pieces

### Update Conditions
Correction history is updated when:
- Not in check
- Best move is not a capture
- Not `(bestScore >= beta && bestScore <= staticEval)` -- not a trivially confirmed eval
- Not `(!bestMove && bestScore >= staticEval)` -- not a stand-pat-like result

Bonus: `CLAMP((score - eval) * depth / 4, -172, 289)` (asymmetric clamp)
Gravity divisors: 1651 (pawn), 1142 (minor), 1222 (major), 1063 (non-pawn), 1514 (cont-corr)

### 50-Move Rule Decay
Applied in `CorrectEval`: `correctedEval *= (256 - rule50) / 256.0` when `rule50 > 7`
- This is integrated into the correction pipeline, not eval itself

Compare to ours: we have single pawn correction history. Weiss has 10 components with tuned weights. This is SUMMARY.md idea #7 (Tier 1, 11+ engines). The key implementation detail: separate Zobrist keys for minor, major, and per-color non-pawn pieces.

---

## 7. Transposition Table

### Structure
- 2-entry buckets (`BUCKET_SIZE = 2`)
- Entry: 12 bytes (key i32, move u32, score i16, eval i16, depth u8, genBound u8)
- Index: Lemire's fast modulo reduction `((u128)key * count) >> 64`
- Key stored as truncated `int32_t` (32-bit)

### Replacement Policy
Replace if ANY of:
- Different key (new position)
- `depth + 4 >= old_depth` (new depth is close to or exceeds old)
- Bound is EXACT
- Entry is from a previous generation (`Age(tte)`)

### Probe
- Iterates 2 entries per bucket
- Returns first matching key or first empty slot
- If no match and no empty: replaces lowest `EntryValue(depth - age)` entry
- TT cutoff: `!pvNode && ttDepth >= depth && TTScoreIsMoreInformative(ttBound, ttScore, beta)`
- History bonus on TT cutoff: quiet TT moves that cause cutoff get `Bonus(depth)` in both butterfly and pawn history

### Prefetch
- `TTPrefetch(key)` via `__builtin_prefetch` (called in makemove)

Compare to ours: we use 4-slot buckets with lockless atomics. Their 2-entry buckets are simpler. They store eval in TT (we do too). Their generation-based aging with combined genBound byte is clean.

---

## 8. Time Management

### Base Allocation

**Standard (no movestogo)**:
- `mtg = 50`
- `timeLeft = MAX(0, time + 50*inc - 50*6)` (6ms overhead per move)
- `scale = 0.022`
- `optimalUsage = MIN(timeLeft * 0.022, 0.2 * time)`
- `maxUsage = MIN(5 * optimalUsage, 0.8 * time)`

For 10s+0.1s: timeLeft = ~10000 + 5000 - 300 = ~14700. optimal = ~323ms. max = ~1617ms.

**Moves-to-go**:
- `scale = 0.7 / MIN(mtg, 50)`
- `optimalUsage = MIN(timeLeft * scale, 0.8 * time)`
- `maxUsage = MIN(5 * optimalUsage, 0.8 * time)`

### Node-Based Soft Time Scaling
- `nodeRatio = 1.0 - bestMoveNodes / totalNodes`
- `timeRatio = 0.52 + 3.73 * nodeRatio`
- Stop after iteration if: `!uncertain && timeSince > optimalUsage * timeRatio`
- `uncertain` flag: set when aspiration PV line[0] differs from rootMoves[0].move after stable window

Effect: if best move uses 50% of nodes, nodeRatio=0.5, timeRatio=2.39. If 90%, timeRatio=0.89. If 10%, timeRatio=3.89. Very aggressive scaling.

### Forced Move Detection
- When only 1 legal root move: `optimalUsage = MIN(500, optimalUsage)` -- cap at 500ms

### doPruning as Time Gate
- The `doPruning` flag also acts as a time gate in `OutOfTime`:
  - When `!doPruning && elapsed >= optimalUsage/32`, enable pruning
  - This means early iterations run without pruning (more accurate but slower)

Compare to ours: we use instability factor (200) based on best move changes. Their continuous node-ratio scaling is more granular and includes an `uncertain` flag from aspiration results. Listed in SUMMARY.md idea #24 (Tier 2).

---

## 9. Evaluation (Classical HCE)

Weiss uses a hand-crafted evaluation with tapered scoring (midgame/endgame phase interpolation). **No NNUE.**

### Material + PST
- Piece values: P=104/204, N=420/632, B=427/659, R=569/1111, Q=1485/1963
- Incremental material + PSQT maintained in `pos->material`
- Phase: based on PhaseValue (N=1, B=1, R=2, Q=4)
- Tempo: +18 for side to move

### Pawn Evaluation (cached)
- Doubled pawns (adjacent and 1-gap): -11/-48 and -10/-25
- Isolated pawns: -8/-16
- Pawn support (pawns defending pawns): +22/+17
- Open pawns (no opposing pawn ahead, not pawn-defended): -14/-19
- Phalanx: rank-based bonus up to +231/+367 at rank 7
- Passed pawns: rank-based, with defended bonus
- Pawn cache: `pawnKey % PAWN_CACHE_SIZE` with single entry per slot

### Piece Evaluation
- Mobility: x-ray attack BBs filtered by mobility area (not blocked by own pawns on rank 2/3 or behind pawn, not attacked by enemy pawns)
- Bishop pair: +33/+110
- Minor behind pawn: +9/+32 per piece
- Bad bishop pawns: penalty per (same-color pawn count * central blocked pawn count)
- Rook forward: +28/+31 (fully open), +17/+15 (semi-open)

### King Safety
- King zone attacks: accumulated per attacking piece type
- AttackPower: N=36, B=22, R=23, Q=78
- CheckPower: N=68, B=44, R=88, Q=92
- CountModifier: table indexed by attack count (0-7)
- KingLineDanger: 28-entry table based on queen-like mobility from king
- Pawn shelter: count of own pawns in front of king, not attacked by opponent pawns

### Passed Pawn Evaluation
- Rank-based bonus with defended bonus
- PassedBlocked / PassedFreeAdv by rank (rank 4+)
- Distance to own and enemy king
- Rook behind passed pawn bonus
- Square rule (promotion race)

### Threats
- PawnThreat: pawns attacking non-pawns (+80/+34 per)
- PushThreat: pawn-push attacks on non-pawns (+25/+6 per)
- ThreatByMinor/ThreatByRook: piece-type specific tables

### Scale Factor
- Fewer pawns for stronger side: `128 - (8-pawnCount)^2`
- Pawns only on one side: -20
- Opposite-colored bishop endgame: 64 (only minors) or 96 (1 extra piece each)

### Endgame Table
- Material-key indexed endgame recognition (KK, KNK, KBK, KNkn, KBkb, KNNk, etc.)
- Currently only trivial draws

Compare to ours: we use NNUE. Their classical eval is well-structured but fundamentally weaker than NNUE. Not relevant for borrowing eval ideas, but their correction history infrastructure is excellent.

---

## 10. Lazy SMP

- Threads allocated via `calloc`, started via `pthread_create`
- Each thread has own: Position, pawn cache, all history tables, continuation history, correction history
- Shared: TT only (no atomic access -- relies on aligned writes being naturally atomic on x86)
- Helper threads search to MAX_PLY (no depth limit), main thread has depth limit
- Thread index 0 is main thread; only main thread handles time and prints output
- `ABORT_SIGNAL`: atomic bool, checked every 2048 nodes

Compare to ours: very similar design. We use packed atomics for TT, they don't (simpler but technically racy on non-x86).

---

## 11. Notable Differences from GoChess

### Things Weiss has that we don't:
1. **10-component correction history** with tuned weights (pawn + minor + major + per-color non-pawn + cont-corr plies 2-7)
2. **Pawn history table** (`pawnHistory[512][piece][to]` indexed by pawn structure hash)
3. **Continuation history ply 4** (plies 1, 2, 4 -- skips 3)
4. **ProbCut QS pre-filter** (two-stage: QS first, then reduced search)
5. **ProbCut score dampening** (`score - 160` on non-terminal returns)
6. **DoDeeper search** (reduction-proportional threshold after LMR re-search)
7. **Aspiration score-dependent delta** (`9 + prevScore^2/16384`)
8. ~~**Aspiration fail-low beta contraction**~~ (`beta = (alpha+3*beta)/4`) **(UPDATE 2026-03-21: GoChess now has aspiration contraction (3a+5b)/8)**
9. **Aspiration fail-high depth reduction** (inner loop only)
10. **Time-adaptive pruning enable** (`doPruning` flag delayed by time/depth)
11. **NMP depth-dependent eval gate** (`staticEval >= beta + 138 - 13*depth`)
12. **NMP opponent history gate** (`(ss-1)->histScore < 28500`)
13. **RFP history adjustment** (`(ss-1)->histScore / 131` subtracted from margin)
14. **RFP TT move history guard** (`!ttMove || GetHistory(ttMove) > 6450`)
15. **Cut-node +2 LMR** (`r += 2 * cutnode`)
16. **Opponent material LMR** (`r += nonPawnCount[opponent] < 2`)
17. **TT cutoff history bonus** (quiet TT moves get butterfly + pawn history bonus on cutoff)
18. **Upcoming repetition detection** (Cuckoo-based `HasCycle`)
19. **Eval trend bonus** (biases eval toward previous iteration's score)
20. **50-move decay in correction** (`(256-rule50)/256` factor)
21. **Cont-hist update on LMR re-search** (score <= alpha or >= beta -> cont-hist update)
22. **Asymmetric history bonus/malus** (bonus up to 2418, malus capped at 693)
23. **Mate distance pruning** (both main search and QS)
24. **Double extension cap** (max 5 per line via `ss->doubleExtensions`)

### Things we have that Weiss doesn't:
1. **NNUE** (they use classical eval -- major strength difference)
2. **Counter-move heuristic** (they have none)
3. **2 killers per ply** (they have 1)
4. **Lockless TT** (they rely on natural alignment atomicity)
5. **Recapture extensions**
6. **Alpha-reduce** (depth-1 after alpha raised)
7. **Fail-high score blending** in main search
8. **TT score dampening** at cutoffs
9. **TT near-miss cutoffs** (1 ply shallower with margin)
10. **TT noisy move detection** (+1 LMR for quiets when TT is noisy)
11. **QS beta blending**
12. **NMP threat detection** (escape bonus)

---

## 12. Parameter Comparison Table

| Feature | Weiss | GoChess |
|---------|-------|---------|
| RFP margin | 77*(d-improving), depth<7 | 85*d (imp) / 60*d (not), depth<=8 |
| RFP extra | -(ss-1)->histScore/131, TT guard | None |
| NMP base R | 4 + d/4 | 3 + d/3 |
| NMP eval-beta div | 227 | 200 |
| NMP extra gate | staticEval >= beta+138-13*d, histScore<28500 | None |
| LMR base (quiet) | 2.01 | 1.50 |
| LMR div (quiet) | 2.32 | ~2.36 |
| LMR base (capture) | 0.38 | ~0.69 (C=1.80) |
| LMR div (capture) | 3.76 | ~2.76 |
| LMR cutnode | +2 | +0 |
| LMR endgame | +1 if oppo nonPawn<2 | None |
| LMR history div | 8870 | 5000 |
| LMP (not improving) | d^2/2 | 3+d^2 |
| LMP (improving) | 2+d^2 | (3+d^2)*1.5 |
| Futility | 80+lmrD*80 (ours) | 80+lmrD*80 |
| SEE pruning | -73*d (uniform) | Separate quiet/tactical |
| SE min depth | 5 | 10 |
| SE margin | d*(2-pvNode) | d*3 |
| SE double ext | +2 if score < singBeta-1, cap 5 | None |
| SE multi-cut | return singBeta if >= beta | None |
| SE negative ext | -1 if ttScore >= beta | None |
| ProbCut margin | beta+200 | beta+170 |
| ProbCut QS pre-filter | Yes | No |
| Aspiration delta | 9+score^2/16384 | 15 |
| History bonus | MIN(2418, 251*d-267) | similar |
| History malus | -MIN(693, 532*d-163) | symmetric with bonus |
| History gravity (butterfly) | div 4373 | div ~5000 |
| Correction sources | 10 | 1 (pawn only) |
| Pawn history | Yes (512 buckets) | No |
| ContHist plies | 1, 2, 4 | 1, 2 |
| Killers | 1 per ply | 2 per ply |
| Counter-move | No | Yes |
| TT buckets | 2 entries | 4 slots |
| Draw randomization | 1-8 | None |

---

## 13. Ideas Worth Testing from Weiss

### Already noted in SUMMARY.md (reinforces priority):
1. **Multi-source correction history** -- Weiss is the reference implementation for 10-component weighted blend. Priority HIGH. (SUMMARY #7)
2. **ProbCut QS pre-filter** -- two-stage ProbCut confirmed in Weiss. (SUMMARY #6)
3. **DoDeeper search** -- reduction-proportional threshold `score > bestScore + 1 + 6*(newDepth-lmrDepth)`. (SUMMARY #1)
4. **Cut-node +2 LMR** -- `r += 2 * cutnode`. (SUMMARY #25)
5. **Opponent material LMR** -- `r += nonPawnCount[opponent] < 2`. (SUMMARY #26)
6. **Pawn history** -- `[pawnKey%512][piece][to]`. (SUMMARY #22)
7. **Cont-hist ply 4** -- adds ply 4 to ordering and updates, skips ply 3. (SUMMARY #20)
8. **Time-adaptive pruning** -- `doPruning` flag. (SUMMARY #43)
9. **ASP fail-high depth reduce** -- inner loop `depth = MAX(1, depth-1)`. (SUMMARY #14)
10. **ASP fail-low beta contraction** -- `beta = (alpha+3*beta)/4`. (SUMMARY #14)

### New or reinforced ideas from Weiss:
11. **ProbCut score dampening** (`score - 160`) -- prevents ProbCut from returning inflated scores. Very simple, aligns with our pattern #1 (dampening at noisy boundaries). **Est. Elo: +2 to +5**.
12. **NMP depth-dependent eval gate** (`staticEval >= beta + 138 - 13*depth`) -- adds a second eval condition that relaxes with depth. Prevents NMP at shallow depth when eval is barely above beta. **Est. Elo: +2 to +5**.
13. **NMP opponent history gate** (`(ss-1)->histScore < 28500`) -- skip NMP when opponent's last move has very high history (strongly expected move). **Est. Elo: +1 to +3**.
14. **RFP history adjustment** (`(ss-1)->histScore / 131`) -- tighten RFP margin when opponent's last move had good history. Aligns with pattern #6 (guards). **Est. Elo: +2 to +4**.
15. **Asymmetric history bonus/malus** -- malus capped much lower than bonus (693 vs 2418). Prevents over-penalization of moves that failed in one position but may be good elsewhere. **Est. Elo: +1 to +3**.
16. **Eval trend bonus** -- `CLAMP(prevScore/2, -32, 32)` as S(x, x/2) added to eval. Biases toward expected score trajectory. Novel, untested. **Est. Elo: +1 to +3**.
17. **Cont-hist update on LMR re-search** -- when LMR re-search score `<= alpha || >= beta`, update cont-hist with bonus/malus. Provides history feedback from reduced searches. **Est. Elo: +1 to +3**.
18. **TT cutoff history bonus** -- quiet TT moves that cause cutoff get butterfly + pawn history bonus. Leverages TT cutoffs for move ordering improvement. **Est. Elo: +2 to +4**.
19. **Score-dependent aspiration delta** (`9 + prevScore^2/16384`) -- larger windows for extreme scores where volatility is higher. **Est. Elo: +1 to +3**.
20. **Mate distance pruning** -- 3 lines, both main search and QS. Universal technique. **Est. Elo: +1 to +2**.
