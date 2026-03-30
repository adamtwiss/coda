# RubiChess Engine - Crib Notes

Source: `~/chess/engines/RubiChess/`
Author: Andreas Matthies
Version: 2026 (git)
NNUE: Dual architecture — V1 (HalfKP 40960->256x2->32->32->1) and V5 (HalfKAv2_hm with horizontal mirroring, 1024x2->16+16x32->1, 8 output buckets, 8 PSQT buckets)
Strength: #6 in our RR at +247 Elo (355 above GoChess-v5)

---

## 1. NNUE Architecture

### V5 (Default, SFNNv5 compatible)
- **Input**: HalfKAv2 with horizontal mirroring — `64 * 11 * 64 / 2 = 22528` input features (11 piece types including own king, 64 squares, 32 king buckets from horizontal mirroring)
- **Feature transformer**: 22528 -> 1024 (configurable; 512/768/1024 variants exist)
- **Hidden layer 1**: 1024 -> 16 neurons. The 16th neuron is a "fwd" output that bypasses the rest via a scaled direct contribution: `fwdout = hidden1[15] * (600*1024/nnuevaluescale) / (127 * 64)`
- **Activation between L1 and L2**: Dual path — SqrClippedReLU on first 15 neurons (squared, for non-linearity) + ClippedReLU on first 15 neurons. These are concatenated to form 30-dim input to L2
- **Hidden layer 2**: 30 -> 32
- **Output layer**: 32 -> 1
- **Output buckets**: 8 layer stacks, selected by `(popcount - 1) / 4`
- **PSQT buckets**: 8, also by piece count. PSQT output added to network output
- **Quantization**: ClippedReLU shift = 6 (divide by 64)
- **Value scale**: `(psqt + positional) * nnuevaluescale / 1024` where nnuevaluescale = 59
- **King buckets**: 32 (half-board, horizontally mirrored). The KingBucket table maps squares to 0-31 with files e-h only (a-d mirrored)
- **Accumulator cache**: Per-king-bucket cache (`accucache`) with piece bitboard state — avoids full recompute on king bucket changes (FinnyTable equivalent)

### V1 (Legacy HalfKP)
- Classic HalfKP: 40960 -> 256x2 -> 32 -> 32 -> 1
- No PSQT buckets, no output buckets
- Standard ClippedReLU activation

### NNUE-Classical Hybrid
- Uses NNUE when PSQT-based eval is within threshold: `abs(GETEGVAL(psqval)) < NnuePsqThreshold` (760)
- Falls back to full classical eval (mobility, king safety, threats, pawns, passed pawns, complexity) when outside threshold
- NNUE eval is phase-scaled: `score * (116 + phcount) / 128`
- Adds tempo (20cp) and FRC correction for Chess960

### Compared to GoChess NNUE
- RubiChess V5 is a much more modern SFNNv5-compatible architecture with horizontal mirroring, PSQT buckets, and the dual SqrCReLU+CReLU activation path
- Our v5 (768x16->N)x2->1x8 with 8 output buckets, CReLU/SCReLU, pairwise mul, Finny tables **(UPDATE 2026-03-21)**
- RubiChess has FinnyTable-style accumulator cache; **(UPDATE 2026-03-21: GoChess now has Finny tables too)**
- The fwd/bypass neuron design in V5 is unique — one neuron directly contributes a PSQT-like signal

---

## 2. Search Architecture

Standard iterative deepening with aspiration windows, PVS, and Lazy SMP (shared TT only, per-thread everything else).

### Iterative Deepening
- Depth 1 to MAXDEPTH-1 (255)
- Lazy SMP uses Laser's skip-depth pattern: `SkipSize[16]` and `SkipDepths[16]` tables, indexed by `threadindex % 16`
- Best thread selection at end: picks thread with highest score among those with `lastCompleteDepth >= main thread's depth`

### Aspiration Windows
- Initial delta: 8 (tunable `aspinitialdelta=15` for next-depth window, but initial ID starts with full window)
- After depth > 4: `alpha = score - 15`, `beta = score + 15`
- On fail-low: `beta = (alpha + beta) / 2`, `alpha = max(SCOREBLACKWINS, alpha - delta)` (beta contraction!)
- On fail-high: `beta = min(SCOREWHITEWINS, beta + delta)`
- Delta growth: `delta += delta / aspincratio + aspincbase` = `delta += delta/4 + 2` (multiply by ~1.25 + 2 each widen)
- Opens to full window when `abs(score) > 1000`
- **Root LMR uses `inWindow`**: quiet root moves get `reductiontable[1][depth][i+1] + inWindowLast - 1` reduction. When `inWindow=0` (fail-low), reduction is -1; when `inWindow=2` (fail-high), reduction is +1. Clever implicit depth management

### Draw Detection
- Three-fold repetition: `testRepetition() >= 2`
- Fifty-move rule: `halfmovescounter >= 100` (checks for checkmate in check positions)
- Root position: detects immediate 3-fold and opponent-enabled 3-fold, fixes TT entry to SCOREDRAW

### Mate Distance Pruning
- Template-based: only active when `Pt == MatePrune`
- Standard: `alpha = max(alpha, SCOREBLACKWINS + ply + 1)`, `beta = min(beta, SCOREWHITEWINS - ply)`

---

## 3. Pruning Techniques

### Reverse Futility Pruning (RFP)
- Conditions: `!isCheckbb && depth <= MAXPRUNINGDEPTH(8) && POPCOUNT(threats) < 2`
- Margin: `depth * (futilityreversedepthfactor - futilityreverseimproved * positionImproved)` = `depth * (39 - 4*improving)`
- Formula: `staticeval - depth * margin > beta` => return staticeval
- **Threat guard**: `POPCOUNT(threats) < 2` — skips RFP when 2+ pieces are threatened (important tactical guard)
- Compare to ours: we use 85*d improving / 60*d not-improving, depth<=8. Their base margin is much smaller (39 vs 60-85) but has the threat guard

### Razoring
- Conditions: `!PVNode && !isCheckbb && depth <= 2`
- Margin: `razormargin + depth * razordepthfactor` = `360 + depth * 67` (depth 1: 427, depth 2: 494)
- At depth 1: drops to QS with full alpha/beta if `ralpha < alpha`
- At depth 2: drops to QS with null-window `(ralpha, ralpha+1)` and confirms
- Compare to ours: we use 400+d*100 (depth 1: 500, depth 2: 600). Their margins are tighter

### Threat Pruning (Koivisto-inspired)
- Conditions: `!PVNode && !isCheckbb && depth == 1 && !threats`
- Margin: `staticeval > beta + (positionImproved ? 3 : 29)`
- When no threats exist at depth 1 with static eval well above beta, prune immediately
- **Novel**: We don't have this. It's a depth-1-specific RFP variant conditioned on no threats

### Futility Pruning (Forward)
- Conditions: `depth <= MAXPRUNINGDEPTH(8)`
- Margin: `futilitymargin + futilitymarginperdepth * depth` = `12 + 80*depth`
  - depth 1: 92, depth 2: 172, depth 3: 252, depth 4: 332
- Formula: `staticeval < alpha - margin` => set `futility = true` (per-move application later)
- Per-move: skips quiet non-check moves when `futility && !ISTACTICAL && !isCheckbb && !moveGivesCheck && legalMoves > 0`
- Compare to ours: we use 80+d*80 (merged). Their margins are slightly different (12+80d vs 80+80d) — similar

### Null Move Pruning (NMP)
- Conditions: `!isCheckbb && !threats && depth >= 4 && bestknownscore >= beta && phcount > 0`
- Reduction: `min(depth, nmmredbase + depth/nmmreddepthratio + (bestknownscore-beta)/nmmredevalratio + !PVNode*nmmredpvfactor)` = `min(depth, 3 + depth/5 + (eval-beta)/125 + 2*(!PVNode))`
- **Threat guard**: `!threats` — disables NMP when any piece is threatened (unique among reviewed engines)
- `bestknownscore` = hashscore if available, else staticeval
- **Verification at deep depths**: When `depth >= nmverificationdepth(11)` AND `!nullmoveply`, does a verification search at `depth - nmreduction` with NMP disabled for `3*(depth-R)/4` plies
- Returns `beta` (clamped, not raw score)
- Compare to ours: we use 3+d/3+(eval-beta)/200 min depth 3. Their R formula is similar but with `depth/5` instead of `depth/3` (less reduction), PV bonus of 2, and eval divisor 125 (more aggressive eval component). The threat guard and verification are both notable

### Late Move Pruning (LMP)
- Conditions: `depth <= MAXPRUNINGDEPTH(8) && !ISTACTICAL`
- Table-based with improving flag:
  - Not improving: `lmptable[0][d] = 2.5 + 0.43 * d^0.62`
  - Improving: `lmptable[1][d] = 4.0 + 0.70 * d^1.68`
- Approximate values:
  - depth 1: 3/5, depth 2: 3/5, depth 3: 3/6, depth 4: 4/8, depth 5: 4/14, depth 6: 4/22, depth 7: 5/33, depth 8: 5/47
- Triggers `ms->state++` to skip to next moveselector phase (important optimization)
- Compare to ours: we use `3+d^2` with improving adjustment. Their table is more nuanced with tunable exponents

### SEE Pruning
- Conditions: `!isCheckbb && ms->state >= QUIETSTATE && depth <= MAXPRUNINGDEPTH(8)`
- Margin: `seeprunemarginperdepth * depth * (ISTACTICAL ? depth : seeprunequietfactor)` = `-14 * depth * (tactical ? depth : 3)`
  - Quiet: `-42*depth` (depth 1: -42, depth 4: -168, depth 8: -336)
  - Tactical: `-14*depth^2` (depth 1: -14, depth 4: -224, depth 8: -896)
- Compare to ours: similar concept, different scaling

### QSearch Delta Pruning
- **Position-level**: `staticeval + deltapruningmargin(389) + getBestPossibleCapture() < alpha` => return staticeval
- **Per-move**: `staticeval + materialvalue[capture] + deltapruningmargin(389) <= alpha` => skip capture
- Compare to ours: we have QS delta pruning with 240 margin. Their margin is much larger (389)

---

## 4. Extensions

### Check Extension
- **Guarded by extension budget**: `extensionguard < extguardcheckext(3) * 16`
- The `extensionguard` is a single integer with upper bits (>>4) counting check extensions and lower bits (& 0xf) counting double singular extensions
- Maximum 3 check extensions per branch (48/16 = 3)
- +1 depth when in check
- **When NOT in check**, computes threats instead (threats computation skipped when check-extending)

### Singular Extensions
- Depth guard: `depth >= singularmindepth(8)`
- Conditions: `move == hashmove && !excludeMove && (ttbound & HASHBETA) && ttdepth >= depth - 3`
- Singularity beta: `max(hashscore - singularmarginperdepth(1) * depth, SCOREBLACKWINS)` = `hashscore - depth`
- Singularity depth: `depth / 2`
- If `redScore < sBeta`:
  - **Double extension (+2)**: `!PVNode && redScore < sBeta - singularmarginfor2(17) && (extensionguard & 0xf) <= extguarddoubleext(7)`
  - **Single extension (+1)**: otherwise
- **Multi-cut**: If `bestknownscore >= beta && sBeta >= beta` => return sBeta
- **Negative extension (-1)**: If `hashscore >= beta` (not singular, but hash already cuts)
- Extension guard: double extensions limited to 7 per branch via `extensionguard` lower nibble
- Compare to ours: we use margin `depth*3`, they use `depth*1` (much tighter!). Multi-cut returns sBeta, not beta. Double extension guard is clever

### Endgame Capture Extension
- Conditions: `phcount < 6 && GETCAPTURE(mc) >= WKNIGHT` (capturing knight or better in <= 6 piece count)
- +1 depth extension
- **Novel**: We don't have this specifically

### History Extension (Auto-Tuning Threshold)
- Conditions: `!ISTACTICAL(mc)` and `conthistptr[ply-1][pieceTo] > he_threshold && conthistptr[ply-2][pieceTo] > he_threshold`
- Both continuation history values for the move must exceed the threshold
- +1 depth extension
- **Auto-tuning**: Every 4M nodes (`he_all & 0x3fffff`), adjusts threshold:
  - If extension rate > 1/512 (~0.2%): increase threshold by factor 257/256
  - If extension rate < 1/32768 (~0.003%): decrease threshold by factor 255/256
- This is a self-calibrating history extension — unique among all reviewed engines
- Compare to ours: We don't have history extensions. Igel has a fixed-threshold variant

---

## 5. LMR (Late Move Reductions)

### Table Initialization
Two separate tables for improving vs not-improving:
- **Not improving**: `1 + round(log(d * 1.31) * log(m) * 0.41)`
- **Improving**: `round(log(d * 1.52) * log(m*2) * 0.30)`

The log-factor products differ significantly from a single C parameter. Not-improving gets more reduction.

### Application Conditions
- `depth >= lmrmindepth(3)`
- Applied to ALL moves including captures (no separate capture table)
- Uses `legalMoves + 1` as move index

### Reduction Adjustments
1. **+1 for cut nodes** (quiet moves only): `+= (cutnode && !ISTACTICAL)`
2. **History-based**: `-= stats / (lmrstatsratio(908) * 8)` = `-= stats / 7264`
3. **-1 for PV node**: `-= PVNode`
4. **-1 for PV node with good hash**: `-= (PVNode && (!tpHit || hashmovecode != mc || hashscore > alpha))`
5. **-1 for opponent's high move count**: `-= (CurrentMoveNum[ply-1] >= lmropponentmovecount(24))`
6. **Fail-high count feedback**: If `failhighcount[ply] < 4`: `-= (5 - failhighcount[ply]) / 2`
   - 0 fail-highs: -2, 1: -2, 2: -1, 3: -1

### Clamped to `[0, depth]`

### Notable differences from GoChess
- No separate capture LMR table (we have split tables since +43.5 Elo merge)
- **Opponent move count adjustment**: reduces LMR when opponent had many moves to try (they chose poorly = our move may be good)
- **Fail-high count feedback**: Novel — tracks how many beta cutoffs occurred at the next ply. Fewer cutoffs = more volatile = reduce less. This is a child-to-parent feedback mechanism
- **PV node gets double reduction bonus**: -1 for being PV, plus -1 more if hash entry supports it

---

## 6. Move Ordering

### Staged Move Picker
States: `HASHMOVE -> TACTICAL_INIT -> TACTICAL -> KILLER1 -> KILLER2 -> COUNTER -> QUIET_INIT -> QUIET -> BAD_TACTICAL`
Evasion path: `EVASION_INIT -> EVASION`

### Score Hierarchy
1. **TT move**: returned directly from hash, not scored
2. **Good captures** (SEE >= margin): MVV-LVA + tactical history. Selection sort (pick-best, mark used)
3. **Killer moves**: 2 slots per ply, validated with `moveIsPseudoLegal()`
4. **Counter-move**: `countermove[piece][to]`, one per piece-to combination
5. **Quiet moves**: `history[color][threatSquare][from][to] + conthistptr[ply-1][pieceTo] + conthistptr[ply-2][pieceTo] + (conthistptr[ply-4][pieceTo] + conthistptr[ply-6][pieceTo]) / 2`
6. **Bad captures** (SEE < 0): flagged in tactical phase, replayed at end

### History Tables

**Main history**: `history[2][65][64][64]` — color, **threat square**, from, to
- **Threat-aware**: Index 0-63 = square of first threatened piece, 64 = no threats
- This is RubiChess's implementation of threat-aware history — a 65-value index, not a boolean flag
- Bonus: `depth^2`, capped to `[-400, 400]`
- Gravity: `delta = value * 32 - history[...] * abs(value) / 256`

**Continuation history**: `counterhistory[14][64][14*64]` — prev_piece (14), prev_to (64), curr_pieceTo (14*64)
- Used at plies -1, -2, -4 for scoring
- Updated at plies 0, 1, 3 (offsets {0, 1, 3} from current)
- Read at plies -4 and -6 at half weight for move ordering only

**Tactical history**: `tacticalhst[7][64][6]` — attacker_type, to_square, victim_type
- Same gravity formula as main history

**Counter-move**: `countermove[14][64]` — piece, to

**Correction history**: 3 tables (see section 8)

### Compared to GoChess
- **Threat-aware history** is the major differentiator. 65-value index (threatened square) vs our flat butterfly table. This is equivalent to a 65x expansion of the history table (~530KB per color)
- **Deeper continuation history** at plies -4 and -6 (half weight). We use plies -1 and -2 only
- **No capture history in ordering** — tactical history is used for good capture scoring but not the same way other engines combine it

---

## 7. Transposition Table

### Structure
- **3-entry buckets** (`TTBUCKETNUM = 3`)
- Each entry: 10 bytes (hashupper 2B, movecode 2B, value 2B, staticeval 2B, depth 1B, boundAndAge 1B)
- Cluster: `3 * 10 = 30 bytes` + padding to 32 bytes
- Index: `hash & sizemask` (power-of-2 masking)
- Huge page support on Linux via `madvise(MADV_HUGEPAGE)`

### Probe
- Search all 3 entries for hash match (upper 16 bits)
- If match found or empty slot: return it, refresh age
- If no match: return least-valuable entry based on `depth - age_penalty * 2`

### Replacement Policy
Replace if ANY of:
- New entry is EXACT bound
- Different hash (new position)
- `ttdepth + 3 >= old_depth` (generous — new entry within 3 plies of old)

### Age Management
- Age stored in `boundAndAge` field (upper 6 bits)
- Incremented each search via `nextSearch()`
- Replacement weight: `depth - (age_difference) * 2`

### Compared to GoChess
- We use 4-slot buckets with lockless XOR-verified atomics; they use 3-slot with no locking
- Their replacement is simpler: just `ttdepth + 3 >= old_depth`
- They store `staticeval` in the TT entry (we do too)

---

## 8. Correction History

### Three Tables
1. **Pawn correction**: `pawncorrectionhistory[2][CORRHISTSIZE(16384)]` indexed by `pawnhash & (CORRHISTSIZE-1)`
2. **White non-pawn correction**: `nonpawncorrectionhistory[WHITE][2][CORRHISTSIZE]` indexed by `nonpawnhash[WHITE] & (CORRHISTSIZE-1)`
3. **Black non-pawn correction**: `nonpawncorrectionhistory[BLACK][2][CORRHISTSIZE]` indexed by `nonpawnhash[BLACK] & (CORRHISTSIZE-1)`

### Non-pawn Zobrist Keys
- `nonpawnhash[WHITE]` and `nonpawnhash[BLACK]` computed incrementally in `playMove()`:
  - XOR of `boardtable[(sq << 4) | piece]` for all non-pawn pieces of that color
  - Kings included in non-pawn hash
- This is the "proper Zobrist-based non-pawn keys separated by color" approach that 11+ engines use

### Update Formula
- `weight = min(1 + depth, 16)`
- `corrHist = (corrHist * (256 - weight) + scaledValue * weight) / 256`
- `scaledValue = (bestScore - staticeval) * 256`
- Clamped to `[-8192, 8192]`
- Division ratios: `pawncorrectionhistoryratio(102)` and `nonpawncorrectionhistoryratio(102)`

### Correction Applied
- `correctEvalByHistory(rawEval)` adds all three corrections divided by their ratios
- Applied to raw static eval before use in search decisions

### Update Conditions
- On beta cutoff: `!ISCAPTURE && !isCheckbb && !(bestscore < staticeval)`
- On search completion: `!ISCAPTURE && !isCheckbb && !(eval_type == HASHALPHA && bestscore > staticeval)`

### Compared to GoChess
- We have pawn correction only. Their 3-table approach with per-color non-pawn Zobrist keys is the standard modern design
- **This is a HIGH priority for us** — 11+ engines have multi-source correction history, and Alexandria/RubiChess show the cleanest implementations

---

## 9. Time Management

### Dual Time Limits
- `endtime1`: soft limit — stop before starting next iteration
- `endtime2`: hard limit — stop immediately mid-search

### Base Allocation (Sudden Death with Increment)
- Phase: `ph = (materialPhase + min(255, fullmovescounter * 6)) / 2` (blends material and move number)
- Soft: `f1 = max(5, 17 - constance) * bestmovenodesratio`; `endtime1 = thinkStartTime + max(inc, f1 * (time + inc) / 128 / (256 - ph))`
- Hard: `f2 = max(15, 27 - constance) * bestmovenodesratio`; `endtime2 = clockStartTime + min(time - overhead, f2 * (time + inc) / 128 / (256 - ph))`

### Constance Factor
- `constance = constantRootMoves * 2 + ponderhitbonus`
- `constantRootMoves` increments each iteration when best move unchanged
- Halved when score drops by > 10cp

### Node-Based Best Move Ratio
- `bestmovenodesratio = 128 * (2.5 - 2 * bestMoveNodes / totalNodes)`
- Range: [64, 320] approximately
- When best move uses 50% of nodes: ratio = 192 (1.5x soft limit)
- When best move uses 90%: ratio = 64 (0.5x soft limit)
- When best move uses 10%: ratio = 294 (2.3x soft limit)

### Score Drop Handling
- When `lastiterationscore > bestmovescore + 10`: `constantRootMoves /= 2`
- This indirectly extends time by reducing constance

### Compared to GoChess
- More sophisticated than our instability-based approach. They combine:
  - Best move stability (constantRootMoves counter)
  - Node distribution ratio (continuous, not binary)
  - Score drop detection
  - Material phase in allocation
  - Ponder hit bonus
- **(UPDATE 2026-03-21: GoChess now has score-drop time extension 2.0x/1.5x/1.2x, merged.)** We use instability * 200 for best-move changes. Their system is more granular

---

## 10. Unique/Notable Techniques

### 1. Threat-Aware History (65-value index)
Instead of a boolean from-threatened/to-threatened flag (like Berserk/Seer), RubiChess uses the actual square of the first threatened piece as an index into the history table. When no threats: index = 64. This gives finer granularity than boolean approaches.

`history[2][65][64][64]` = 2 colors x 65 threat indices x 64 from x 64 to = ~32MB per thread

### 2. Fail-High Count Feedback in LMR
`failhighcount[ply]` tracks how many beta cutoffs occurred at each ply. When `failhighcount[ply] < 4`, child nodes get reduced LMR: `-(5 - failhighcount) / 2`. This provides direct child-to-parent feedback about position volatility.

Updated: `failhighcount[ply] += (!hashmovecode + 1)` — gives 2 points when no hash move existed (more surprising cutoff), 1 point otherwise.

Reset: `failhighcount[ply + 2] = 0` at each node entry.

### 3. Self-Tuning History Extension Threshold
The `he_threshold` adjusts itself every 4M nodes to maintain a target extension rate between 0.003% and 0.2%. This avoids the need for manual tuning of the threshold parameter.

### 4. Extension Guard (Unified Budget)
A single `extensionguard` integer tracks both check extensions (upper bits) and double singular extensions (lower nibble). Limits: max 3 check extensions, max 7 double extensions per branch. This prevents explosive tree growth.

### 5. Threat Guard on NMP and RFP
- NMP: `!threats` (disables NMP when any piece is threatened)
- RFP: `POPCOUNT(threats) < 2` (allows RFP with 0-1 threats, disables with 2+)
- Threat computation: pawn attacks on non-pawns, minor attacks on rooks/queens, rook attacks on queens

### 6. NNUE-Classical Hybrid with PSQ Threshold
Falls back to full classical eval when PSQT-based eval (from NNUE feature transformer) is large (`>= 760`), indicating a position where material imbalance is extreme enough that NNUE may be unreliable.

### 7. Opponent Move Count in LMR
`-= (CurrentMoveNum[ply-1] >= 24)` — if the opponent tried 24+ moves before finding one (high move number), our current position is likely good and we should reduce less.

### 8. Speculative NNUE Evaluation
`NnueSpeculativeEval()` called before ProbCut and singular extension searches to ensure the NNUE accumulator is incrementally refreshed, avoiding full recomputes later.

### 9. Root LMR Modulated by Aspiration Window State
Root LMR reduction includes `inWindowLast - 1`: when last iteration failed low (inWindow=0), reduction is -1 (search deeper); when failed high (inWindow=2), reduction is +1 (search shallower). Elegant coupling of aspiration state with root move ordering.

---

## 11. Parameter Comparison Table

| Feature | RubiChess | GoChess |
|---------|-----------|---------|
| RFP margin | d*(39-4*imp), d<=8 | 85*d (imp) / 60*d (not), d<=8 |
| RFP guard | threats < 2 | none |
| Razoring | 360+67*d, d<=2 | 400+100*d, d<=3 |
| NMP min depth | 4 | 3 |
| NMP R base | 3 + d/5 + (eval-beta)/125 + 2*!PV | 3 + d/3 + (eval-beta)/200 |
| NMP guard | !threats | none |
| NMP verification | depth >= 11 | depth >= 12 |
| LMR tables | improving/not-improving | capture/quiet split |
| LMR history div | 7264 | 5000 |
| LMR cutnode | +1 (quiets only) | +1 |
| LMR fail-high feedback | yes (child cutoff count) | no |
| LMR opponent movecount | yes (-1 at >=24) | no |
| LMP depth | <=8 (table) | 3+d^2 (+50% imp) |
| Futility margin | 12+80*d, d<=8 | 80+80*d |
| SEE quiet margin | -42*d | -20*d^2 |
| Singular depth | >=8 | >=10 |
| Singular beta | hashscore - 1*depth | hashscore - 3*depth |
| Double SE guard | <=7 per branch | none |
| Multi-cut | yes (return sBeta) | no |
| History table | threat-aware (65 indices) | flat butterfly |
| Cont-hist plies | -1,-2,-4 (read -4,-6 at half) | -1,-2 |
| Correction history | 3 tables (pawn + 2x non-pawn) | 1 table (pawn only) |
| Aspiration delta | 15 (depth>4) | 15 |
| Aspiration fail-low | beta contraction | contraction (3a+5b)/8 (UPDATE: merged) |
| Time management | node ratio + stability + score | instability + score-drop 2.0x/1.5x/1.2x (UPDATE) |
| NNUE arch | SFNNv5 1024, 8 output buckets | v5: dynamic width, 8 output buckets (UPDATE) |

---

## 12. Ideas Worth Testing from RubiChess

### HIGH PRIORITY

1. **Multi-source correction history** (3 tables: pawn + white non-pawn + black non-pawn)
   - RubiChess shows a clean implementation with per-color Zobrist non-pawn keys
   - Our single pawn-only correction is leaving Elo on the table
   - 11+ engines confirm this; RubiChess + Alexandria are the reference implementations
   - **Status**: Our XOR-bitboard approach failed (-11.8 Elo). Need proper Zobrist-based keys as shown here

2. **Fail-high count feedback in LMR** (-2 to 0 reduction based on child cutoff count)
   - Novel mechanism: fewer cutoffs at child = more volatile = reduce less
   - Similar to Reckless's `cutoff_count > 2 -> reduction += 1604` but reverse direction
   - **Status**: Reckless variant is in SUMMARY.md as idea #31. RubiChess variant is cleaner

3. **Threat guard on NMP and RFP**
   - NMP: disable when threats exist (opponent has hanging piece = might be zugzwang-like)
   - RFP: disable when 2+ pieces threatened (position too tactical)
   - Aligns with our pattern #6 (guards prevent over-pruning in tactical positions)
   - **Status**: Related to SUMMARY.md idea #12 (opponent-threats guard). RubiChess shows the exact implementation

4. **Deeper continuation history** (plies -4 and -6 at half weight for ordering)
   - Already in SUMMARY.md as idea #20 (11 engines). RubiChess adds the half-weight detail
   - **Status**: Our test of contHist4 at full weight got -58 Elo. RubiChess uses half weight — worth retesting

### MEDIUM PRIORITY

5. **Self-tuning history extension threshold**
   - Auto-calibrates to maintain ~0.003%-0.2% extension rate
   - Eliminates a hard-to-tune parameter
   - **Status**: In SUMMARY.md as Tier 3 idea #32. RubiChess shows the only auto-tuning implementation

6. **Extension guard** (unified budget for check + double SE extensions)
   - Prevents explosive tree growth from unbounded extensions
   - Our SE was -58 to -85 Elo — possibly due to missing extension limiter
   - Alexandria uses `ply*2 < RootDepth*5`; RubiChess uses the extensionguard nibble approach
   - **Status**: Critical for fixing our singular extensions

7. ~~**Aspiration fail-low beta contraction**~~: `beta = (alpha + beta) / 2` **(UPDATE 2026-03-21: GoChess now has aspiration contraction)**
   - Tighter re-search window after fail-low
   - Midnight also has this (`beta = (alpha + 3*beta) / 4`)
   - **Status**: Not tested yet

8. **Opponent move count LMR adjustment**: `-= (opponentMoveNum >= 24)`
   - Cheap to implement — just read the CurrentMoveNum stack
   - **Status**: Novel, not in SUMMARY.md yet

9. **Threat pruning at depth 1** (Koivisto-inspired)
   - When `!threats && staticeval > beta + margin`: prune at depth 1
   - Very targeted — only depth 1, only when no threats
   - **Status**: Not tested yet

### LOW PRIORITY

10. **Endgame capture extension** (captures of >= knight when < 6 pieces)
    - Already in SUMMARY.md as Tier 3 idea #39
    - **Status**: Not tested

11. **Root LMR modulated by aspiration state**
    - Interesting coupling but may interact poorly with our root search structure
    - **Status**: Not tested

12. **NNUE-classical hybrid with PSQ threshold**
    - Only relevant if we implement classical eval fallback
    - Not applicable to our current NNUE-only approach
