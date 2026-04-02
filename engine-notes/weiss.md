# Weiss Chess Engine - Crib Notes

Source: `~/chess/engines/weiss/src/`
Author: Terje Kirstihagen
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
- **Coda comparison**: Coda uses eval-dependent delta (13+avg^2/23660), fail-low beta contraction (3a+5b)/8, fail-high alpha contraction (5a+3b)/8 + depth reduce, delta growth 1.5x. Both use score-dependent delta. Weiss's trend bonus is unique.

### Draw Detection
- Repetition: **any** two-fold (checks all `i += 2` up to `min(rule50, histPly)`)
- Upcoming repetition: `HasCycle(pos, ply)` -- Cuckoo-based upcoming repetition detection (alpha < 0 only)
- 50-move rule: `rule50 >= 100`
- Draw randomization: `8 - (nodes & 0x7)` -- small positive score +1 to +8
- **Coda comparison**: Coda has Cuckoo cycle detection (FEAT_CUCKOO). No draw randomization.

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

This prevents aggressive pruning in very early iterations where eval is unstable. The flag is checked before RFP, NMP, and all per-move pruning. Coda has no such gating.

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
- **Coda comparison**: Coda uses 70*d (improving) / 100*d (not), depth<=7. Coda has an RFP TT quiet guard (skip when TT has quiet move) but not the history adjustment from opponent's last move.

### Null Move Pruning (NMP)
- Conditions: `eval >= beta && eval >= staticEval && staticEval >= beta + 138 - 13*depth`
  - The `staticEval >= beta + 138 - 13*depth` is a **depth-dependent verification gate** -- at low depth, staticEval must be well above beta; at high depth, the threshold relaxes
- Additional guard: `(ss-1)->histScore < 28500` -- skip NMP if opponent's last move had very high history
- Material guard: `nonPawnCount[stm] > (depth > 8)` -- at depth > 8, need at least 2 non-pawn pieces
- Reduction: `4 + depth/4 + MIN(3, (eval-beta)/227)`
- Returns raw score (no beta clamping) if `>= beta`, but filters terminal wins to just beta
- **Coda comparison**: Coda uses R=4+d/3+(eval-beta)/200, verify at depth>=12, post-capture R--, score dampening (2s+b)/3. No depth-dependent eval gate or history gate.

### Internal Iterative Reduction (IIR)
- Two separate conditions:
  1. PV node: `pvNode && depth >= 3 && !ttMove` -> `depth--`
  2. Cut node: `cutnode && depth >= 8 && !ttMove` -> `depth--`
- These are **additive** -- a PV node at depth 8 with no TT move doesn't get double IIR (pvNode skips pruning so cutnode branch is never hit on PV)
- **Coda comparison**: Coda has IIR at depth>=4 with no TT move on PV/cut nodes.

### ProbCut
- Threshold: `beta + 200`
- Depth guard: `depth >= 5`
- TT guard: `!ttHit || ttScore >= probCutBeta`
- **Two-stage**: QS first, then `depth-4` search only if QS passes
- SEE threshold for captures: `probCutBeta - staticEval` (dynamic, based on margin needed)
- Only searches moves that pass NOISY_GOOD stage (SEE-filtered)
- Return adjustment: `score - 160` for non-terminal wins (dampens ProbCut score)
- **Coda comparison**: Coda has ProbCut (beta+170, depth>=5, eval+85 gate) but it is DISABLED. Weiss's two-stage QS pre-filter and score dampening are notable improvements if ProbCut is re-enabled.

### Late Move Pruning (LMP)
- Formula: `moveCount > (improving ? 2 + depth*depth : depth*depth/2)`
- Sets `mp.onlyNoisy = true` (skips all remaining quiets, but still searches captures)
- **Coda comparison**: Coda uses 3+d^2 with improving +50% / failing -33%. At depth 3: Weiss=4/11, Coda=12/18. Weiss is much tighter.

### History Pruning
- Conditions: `quiet && lmrDepth < 3 && histScore < -1024 * depth`
- Note: uses `lmrDepth` (reduced depth) not raw depth
- **Coda comparison**: Coda uses -1500*depth at depth<=3.

### SEE Pruning
- Conditions: `lmrDepth < 7`
- Single threshold: `-73 * depth` for ALL moves (quiet and noisy)
- **Coda comparison**: Coda uses separate quiet (-20d^2) and tactical (-d*100) margins.

### QSearch Pruning
- Futility: `eval + 165` is the futility value
  - `futility + PieceValue[EG][captured] <= alpha && !promotion` -> skip (updates bestScore)
- SEE filter: `futility <= alpha && !SEE(pos, move, 1)` -> skip
- Stage filter: `mp.stage > NOISY_GOOD` -> break (stop searching bad captures entirely)
- Mate distance pruning in QS (applied before moves)
- Upcoming repetition detection in QS (`HasCycle`)

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
- **Coda comparison**: Coda has SE at depth>=8, margin=tt_score-depth, multi-cut, negative ext. No double extensions or extension limiter. Weiss's lower SE depth (5 vs 8) and double extensions with cap are worth testing.

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

Weiss has a much higher quiet base (2.01) -- very aggressive quiet reduction. Lower capture base (0.38).

**Coda comparison**: Coda uses C_quiet=1.30 / C_cap=1.80 (divisor-style). Different formula but Coda's quiet reductions are less aggressive.

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
- `+= moveIsCapture(ttMove)` (reduce more when TT move is noisy)
- `+= nonPawnCount[opponent] < 2` (reduce more in endgame with few opponent pieces)
- `+= 2 * cutnode` (reduce more at cut nodes -- **+2 per cut node, very aggressive**)
- Clamped to `[1, newDepth]`
- **Coda**: Does not have cutnode +2 LMR or opponent material LMR adjustment.

### DoDeeper Search (IMPLEMENTED in Coda)
After LMR re-search beats alpha:
- `deeper = score > bestScore + 1 + 6 * (newDepth - lmrDepth)`
- If deeper is true: `newDepth += 1` before full re-search
- Coda has doDeeper (+1 if score > best+50) and doShallower (-1 if score < best+new_depth).

### Continuation History Update on Re-search
- If quiet and re-search result `<= alpha || >= beta`: update cont histories with bonus/malus
- This gives history feedback even for LMR re-searches
- **Coda**: Does not update history on LMR re-searches.

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

### History Tables

**Butterfly history**: `history[2][64][64]` -- color, from, to
- Gravity: `entry += bonus - entry * abs(bonus) / 4373`
- Bonus: `MIN(2418, 251*depth - 267)`
- Malus: `-MIN(693, 532*depth - 163)` (asymmetric -- malus is smaller than bonus!)

**Pawn history** (IMPLEMENTED in Coda): `pawnHistory[512][16][64]` -- pawn structure hash % 512, piece, to square
- Gravity divisor: 8663 (slower adaptation than butterfly)

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

**Coda comparison**: Coda has 2 killers + counter-move + pawn history + cont-hist plies 1, 2, 4, 6.

---

## 6. Correction History (Multi-Source)

This is Weiss's most sophisticated feature. **10 correction history components** with tuned weights.

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

**Coda comparison**: Coda has multi-source correction history: pawn + non-pawn(per color) + minor + major + continuation. Similar architecture to Weiss. Coda does NOT have continuation correction at plies 3-7 (only one ply of continuation correction), the 50-move decay, or tuned per-component weights. Weiss's 10-component weighted blend with 6 continuation correction plies is more sophisticated.

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

**Coda comparison**: Coda uses 5-slot buckets with lockless atomics (XOR key verification). TT cutoff history bonus is something Coda doesn't have.

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

**Coda comparison**: Coda uses simpler time management (time_left/20 + 3*inc/4 soft, 5x hard). No node-ratio scaling, no uncertain flag, no pruning gate.

---

## 9. Evaluation (Classical HCE)

Weiss uses a hand-crafted evaluation with tapered scoring (midgame/endgame phase interpolation). **No NNUE.** Not relevant for borrowing eval ideas, but their search and correction history infrastructure is excellent.

---

## 10. Lazy SMP

- Threads allocated via `calloc`, started via `pthread_create`
- Each thread has own: Position, pawn cache, all history tables, continuation history, correction history
- Shared: TT only (no atomic access -- relies on aligned writes being naturally atomic on x86)
- Helper threads search to MAX_PLY (no depth limit), main thread has depth limit
- Thread index 0 is main thread; only main thread handles time and prints output
- `ABORT_SIGNAL`: atomic bool, checked every 2048 nodes

**Coda comparison**: Very similar design. Coda uses packed atomics for TT (proper lockless access).

---

## 11. Notable Differences from Coda

### Things Weiss has that Coda doesn't:
1. **Continuation correction plies 3-7** with tuned weights (Coda has 1 ply of cont-corr)
2. **ProbCut QS pre-filter** (two-stage: QS first, then reduced search)
3. **ProbCut score dampening** (`score - 160` on non-terminal returns)
4. **Time-adaptive pruning enable** (`doPruning` flag delayed by time/depth)
5. **NMP depth-dependent eval gate** (`staticEval >= beta + 138 - 13*depth`)
6. **NMP opponent history gate** (`(ss-1)->histScore < 28500`)
7. **RFP history adjustment** (`(ss-1)->histScore / 131` subtracted from margin)
8. **Cut-node +2 LMR** (`r += 2 * cutnode`)
9. **Opponent material LMR** (`r += nonPawnCount[opponent] < 2`)
10. **TT cutoff history bonus** (quiet TT moves get butterfly + pawn history bonus on cutoff)
11. **Eval trend bonus** (biases eval toward previous iteration's score)
12. **50-move decay in correction** (`(256-rule50)/256` factor)
13. **Cont-hist update on LMR re-search** (score <= alpha or >= beta -> cont-hist update)
14. **Asymmetric history bonus/malus** (bonus up to 2418, malus capped at 693)
15. **Mate distance pruning** (both main search and QS)
16. **Double extension cap** (max 5 per line via `ss->doubleExtensions`)
17. **Node-ratio time management** (continuous scaling vs Coda's fixed allocation)
18. **Draw randomization** (+1 to +8)
19. **Singular extension depth >= 5** (Coda uses >= 8)
20. **Singular PV-aware margin** (depth on PV vs 2*depth on non-PV)

### Things Coda has that Weiss doesn't:
1. **NNUE** (Weiss uses classical eval)
2. **Counter-move heuristic** (Weiss has none)
3. **2 killers per ply** (Weiss has 1)
4. **Lockless TT** (proper atomic access)
5. **Recapture extensions**
6. **Fail-high score blending** in main search
7. **TT score dampening** at cutoffs
8. **TT near-miss cutoffs** (1 ply shallower with margin)
9. **NMP verification search** at depth >= 12
10. **Hindsight reduction**
11. **Threat-aware 4D history**
12. **Cont-hist plies 4 and 6** (Weiss uses 1, 2, 4; Coda uses 1, 2, 4, 6)

---

## 12. Parameter Comparison Table

| Feature | Weiss | Coda |
|---------|-------|------|
| RFP margin | 77*(d-improving), depth<7 | 70*d (imp) / 100*d (not), depth<=7 |
| RFP extra | -(ss-1)->histScore/131, TT guard | TT quiet guard |
| NMP base R | 4 + d/4 | 4 + d/3 |
| NMP eval-beta div | 227 | 200 |
| NMP extra gate | staticEval >= beta+138-13*d, histScore<28500 | None |
| NMP verification | None | depth >= 12 |
| NMP dampening | None | (2*score+beta)/3 |
| LMR base (quiet) | 2.01 | ln(d)*ln(m)/1.30 |
| LMR div (quiet) | 2.32 | 1.30 |
| LMR base (capture) | 0.38 | ln(d)*ln(m)/1.80 |
| LMR div (capture) | 3.76 | 1.80 |
| LMR cutnode | +2 | None |
| LMR endgame | +1 if oppo nonPawn<2 | None |
| LMR history div | 8870 | (various) |
| LMP (not improving) | d^2/2 | 3+d^2 (failing: *2/3) |
| LMP (improving) | 2+d^2 | (3+d^2)*3/2 |
| Futility | 80+lmrD*80 | 60+lmrD*60 |
| SEE pruning | -73*d (uniform) | Quiet: -20d^2, Cap: -d*100 |
| SE min depth | 5 | 8 |
| SE margin | d*(2-pvNode) | tt_score - depth |
| SE double ext | +2 if score < singBeta-1, cap 5 | None |
| SE multi-cut | return singBeta if >= beta | return singBeta if >= beta |
| ProbCut margin | beta+200 | beta+170 (disabled) |
| ProbCut QS pre-filter | Yes | No |
| Aspiration delta | 9+score^2/16384 | 13+avg^2/23660 |
| History bonus | MIN(2418, 251*d-267) | depth^2 capped 1200 |
| History malus | -MIN(693, 532*d-163) | symmetric with bonus |
| Correction sources | 10 (pawn+minor+major+nonpawn+contCorr 2-7) | 6 (pawn+minor+major+nonpawn+contCorr) |
| Pawn history | Yes (512 buckets) | Yes (512 buckets) |
| ContHist plies | 1, 2, 4 | 1, 2, 4, 6 |
| Killers | 1 per ply | 2 per ply |
| Counter-move | No | Yes |
| TT buckets | 2 entries | 5 slots |
| Draw randomization | 1-8 | None |

---

## 13. Ideas Worth Testing from Weiss

### High priority (reinforces consensus features):
1. **Additional continuation correction plies** (3-7) with tuned weights -- Weiss has 6 plies of cont-corr, Coda has 1. This is the biggest gap.
2. **Cut-node +2 LMR** -- `r += 2 * cutnode`. Aggressive but widely adopted.
3. **Opponent material LMR** -- `r += nonPawnCount[opponent] < 2`. Simple endgame awareness.
4. **Lower SE depth** (5 vs 8) -- more aggressive singular extension activation.
5. **Double extensions** (+2 when well below singular beta, capped at 5 per line).
6. **Mate distance pruning** -- 3 lines, universal technique.

### Medium priority:
7. **ProbCut QS pre-filter** -- two-stage ProbCut. Relevant if ProbCut is re-enabled.
8. **ProbCut score dampening** (`score - 160`) -- prevents inflated ProbCut scores.
9. **NMP depth-dependent eval gate** (`staticEval >= beta + 138 - 13*depth`).
10. **RFP history adjustment** (`(ss-1)->histScore / 131`).
11. **TT cutoff history bonus** -- quiet TT moves that cause cutoff get history bonus.
12. **Asymmetric history bonus/malus** (malus much smaller than bonus).
13. **Node-ratio time management** -- continuous scaling vs fixed allocation.

### Lower priority:
14. **Eval trend bonus** -- biases eval toward previous score. Novel, untested.
15. **50-move decay in correction** -- simple multiplication factor.
16. **Cont-hist update on LMR re-search** -- history feedback from reduced searches.
17. **NMP opponent history gate** (`histScore < 28500`).
18. **Draw randomization** -- small positive draw score. Simple anti-draw-blindness.
