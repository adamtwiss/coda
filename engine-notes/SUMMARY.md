# Engine Review Summary — Ideas for GoChess

Cross-referencing 27 engines against GoChess. Ranked by estimated impact, implementation complexity, and alignment with our proven success patterns.

**Engines reviewed**: Ethereal, Caissa, Midnight, Laser, Winter, Texel, Wasp, Arasan, Crafty, ExChess, GreKo, Rodent III, Berserk (v13), Koivisto, Stormphrax, RubiChess, Seer, Minic, Tucano, Demolito, Weiss, Obsidian, BlackMarlin, Altair, Reckless, Igel, Alexandria

**Our success patterns**:
1. Score dampening works at noisy boundaries (TT cutoffs, QS) — not at well-calibrated margins (RFP)
2. Search byproducts improve move ordering (NMP threat, TT move in QS)
3. Table-based approaches beat hard-coded thresholds (LMR split)
4. History table changes tend to fail — our tables are well-tuned
5. Node-level pruning benefits from tightening; per-move pruning needs slack
6. Guards that prevent pruning in tactical positions (threats, captures) are high-value

---

## MERGED — Proven Winners

| Idea | Elo | Source | Commit |
|------|-----|--------|--------|
| LMR separate tables (cap C=1.80 / quiet C=1.50) | +43.5 | Midnight, Caissa | d8fdc3c |
| QS delta pruning buffer 200→240 | +31.2 | Parameter tuning | 0a4b4d1 |
| TT score dampening `(3*score+beta)/4` | +22.1 | Winter, Caissa | 2b37d25 |
| Fail-high score blending `(score*d+beta)/(d+1)` | +14.7 | Caissa, Stormphrax, Berserk | c7b65d0 |
| Alpha-reduce (depth-1 after alpha raised) | +13.0 | Caissa | 4a49d1f |
| NMP threat detection (+8000 escape bonus) | +12.4 | Rodent, ExChess, Texel | 464a425 |
| ProbCut margin 200→170, gate 100→85 | +10.0 | Multiple | 4c2c92c |
| QS beta blending (bestScore+beta)/2 | +4.9 | Caissa, Berserk, Stormphrax | 00af62c |
| TT near-miss cutoffs (1 ply, 64cp margin) | +21.7 | Minic, Ethereal | a412cbe |
| Futility tightened 100+d*100 → 80+d*80 | +33.6 | Parameter tuning | ec9d8fa |
| TT noisy move detection (+1 LMR for quiets) | +34.4 | Obsidian, Berserk | 330dcd4 |

**Total merged from engine reviews: ~241 self-play Elo**

---

## Tier 1: High Confidence — Multiple Engines, Aligns With Success Patterns

### 1. LMR doDeeperSearch/doShallowerSearch ⭐ TESTING
After LMR re-search beats alpha, dynamically adjust newDepth before full re-search.
- **Engines**: Berserk (`+1 if score > bestScore+69, -1 if score < bestScore+newDepth`), Stormphrax (similar), Stockfish (originated), **Weiss** (`+1 if score > bestScore+1+6*(newDepth-lmrDepth)`), **Obsidian** (`+1 if score > bestScore+ZwsDeeperMargin+2*newDepth, -1 if score < bestScore+ZwsShallowerMargin`)
- **Evidence**: 2 lines of code, Stockfish-originated. 5 engines now. Weiss uses reduction-proportional threshold. Obsidian has tunable margins (43/11).
- **Complexity**: Very low — 2 lines after LMR re-search score check.
- **Est. Elo**: +3 to +6.
- **Status**: TESTING — running SPRT, 152 games, +2.5 Elo (very early).

### 2. RFP with TT Quiet-Move Guard ⭐ TESTING
Skip RFP when TT has a quiet best move — if we know a good quiet move exists, don't prune based on static eval alone.
- **Engines**: Tucano (guards RFP when `ttMove != NONE && !isCapture(ttMove)`), **Weiss** (guards RFP when `!ttMove || GetHistory(thread, ss, ttMove) > 6450`)
- **Evidence**: Aligns with pattern #6 (guards prevent over-pruning). Weiss adds history threshold check. Very targeted condition.
- **Complexity**: 1 line — add `&& (ttMove == NoMove || isCap(ttMove))` to RFP condition.
- **Est. Elo**: +3 to +8.
- **Status**: TESTING — running SPRT, 161 games, -17.4 Elo (looking bad).

### 3. TT Near-Miss Cutoffs ⭐ TESTING
Accept TT entries 1 ply shallower than required, with a score margin proportional to the depth gap.
- **Engines**: Minic (margin 60-64cp per ply gap, credited to Ethereal), Ethereal
- **Evidence**: Reduces re-searching positions where we have a near-hit. Low risk with margin guard.
- **Complexity**: Very low — 10 lines after existing TT cutoff block.
- **Est. Elo**: +2 to +5.
- **Status**: TESTING — running SPRT, 120 games, +31.9 Elo (very promising early).

### 3. QSearch Two-Sided Delta Pruning ⭐ NEW
Two distinct delta conditions beyond standard QS futility:
- **Bad-delta**: SEE ≈ 0 AND eval + margin < alpha → skip (won't help)
- **Good-delta**: SEE is winning AND eval + margin > beta → return beta (already won)
- **Engines**: Minic (credited to Seer), Seer
- **Evidence**: Aligns with pattern #1 (QS is a noisy boundary). The good-delta early return is especially valuable.
- **Complexity**: Very low — two 4-line blocks in QSearch.
- **Est. Elo**: +2 to +5.

### 4. Double/Triple Singular Extensions + Multi-Cut + Negative Extensions
When singular search score is *very* far below threshold, extend by 2 instead of 1. Multi-cut shortcut when alternatives all exceed beta. Negative extension (-1 ply) when not singular but ttScore >= beta.
- **Engines**: Seer (2×depth, +2 at 166), Berserk (+2 at 12, +3 at 43), Stormphrax (+2/+3), Koivisto (multi-cut), RubiChess (+2 at 17), Minic (+2 at 4×depth, multi-cut), Tucano (+2 at 50cp below, multi-cut return), **Weiss** (+2 if !pvNode && score < singBeta-1 && doubleExts≤5; multi-cut; negative ext -1 if ttScore≥beta), **Obsidian** (+2/+3 by margin; negative ext -3+isPV if ttScore≥beta, -2 if cutNode), **BlackMarlin** (double/triple SE)
- **Our status**: sing-d2 margin tested and rejected (-8.5 Elo). Double extension, multi-cut, and negative extension are separate experiments.
- **Complexity**: 5 lines for double ext, 3 lines for multi-cut, 2 lines for negative ext.
- **10 engines** — highest consensus. Multi-cut alone found in 7+ engines. Negative extensions in Weiss/Obsidian.

### 5. Threat-Aware History Tables / Move Ordering
Index butterfly history by from/to threat status (4x table size). Also: threat-aware quiet move scoring.
- **Engines**: Ethereal, Caissa, Berserk, Koivisto, Stormphrax, RubiChess, Seer, **Obsidian**, **BlackMarlin**, **Altair** (`[piece][sq][threatened_origin][threatened_target]`), **Reckless** (factorized: 1852 for threat, 6324 for bucket; also escape bonus Q+20000, R+14000, N/B+8000)
- **Implementations vary**:
  - Boolean from_threatened × to_threatened (Berserk, Stormphrax, Seer, Altair): 4x table
  - Actual threat square index (RubiChess): 65-value index
  - Berserk adds +16863 from-attacked bonus, -13955 to-attacked penalty
  - Obsidian scores threats directly in move ordering (bonus for escaping, penalty for entering)
  - Reckless: piece-type-differentiated escape bonus (Q+20000, R+14000, minor+8000) + queen-into-threat penalty -10000
- **Complexity**: Medium. Need attack map per position, 4x history memory (~256KB extra).
- **Confidence**: Highest-consensus idea across all reviewed engines (**12 engines now**).

### 6. ProbCut QS Pre-Filter ⭐ NEW
At depth > 10, run QS before the full ProbCut negamax search. Only do the expensive reduced-depth search if QS already confirms the cutoff.
- **Engines**: Tucano (two-stage: QS first, then negamax only if QS passes), **Alexandria** (identical two-stage: QS at -pcBeta, then Negamax at depth-4 only if QS passes)
- **Evidence**: Saves significant nodes in deep ProbCut paths. Our ProbCut is already proven (+10 Elo). **2 engines now**.
- **Complexity**: Medium — conditional QS call before existing ProbCut search.
- **Est. Elo**: +3 to +8.

### 7. Multi-Source Correction History
Beyond pawn hash — use non-pawn keys, continuation hashes, material buckets.
- **Engines**: Stormphrax (5 sources), Seer (4 sources), Arasan (6 tables), Berserk (3 tables), RubiChess (3 tables), **Weiss** (pawn + minor + major + per-color non-pawn + cont corr plies -2 to -7, weighted blend), **Obsidian** (pawn + per-color non-pawn + cont corr), **Altair** (pawn + non-pawn + major + minor — 4 tables), **Reckless** (pawn + minor + non-pawn white + non-pawn black + cont corr ply-2 + cont corr ply-4 — 6 components, div 77), **Igel** (same as Reckless), **Alexandria** (pawn + white non-pawn + black non-pawn + continuation corr; weighted blend 29/34/34/26 / 256)
- **Key insight from our failure**: Our XOR-of-bitboards approach failed (-11.8). Successful engines use:
  - Separate per-color non-pawn Zobrist keys (Stormphrax, RubiChess, Weiss, Obsidian, **Alexandria**)
  - Continuation correction history indexed by opponent's last move (Obsidian, Weiss plies -2 to -7, **Alexandria** by `pieceTypeTo(ss-1) x pieceTypeTo(ss-2)`)
  - Minor/major piece count buckets (Weiss — separate tables for minor and major piece counts)
  - Major piece key (Stormphrax)
  - **Alexandria** shows cleanest implementation: per-color Zobrist non-pawn keys computed by `XOR(PieceKeys[piece][sq])` for each non-pawn piece of that color, updated incrementally in MakeMove
  - Weiss blends all sources with tuned weights: `5868*pawn + 7217*minor + 4416*major + 7025*nonPawn + 4060*cont2 + ...` / 131072
- **Priority**: HIGH — retry with proper Zobrist-based non-pawn keys separated by color. Alexandria's code is the reference implementation. **11 engines now**.

### 8. Hindsight Reduction/Bonus (Parent LMR Compensation)
Adjust depth based on parent's reduction and eval trajectory.
- **Engines**: Stormphrax (reduction≥2: depth-- if evals sum positive, depth++ if negative), Berserk (reduction≥3: depth++ if not declining), Koivisto (similar), **Tucano** (reduction≥3 + !opponent_worsening: depth++), **Alexandria** (reduction≥1 + `(ss-1)->staticEval + ss->staticEval >= 155` → depth--)
- **Evidence**: **5 engines now**. Alexandria's variant is simplest: just depth-- when both sides think position is okay (quiet). 2 lines.
- **Complexity**: 2-5 lines. Track parent reduction on search stack.

---

## Tier 2: Promising — Strong Engine Support, Worth Testing

### 9. QS Beta Blending (TESTING)
Stand-pat and fail-high returns blended with beta in QS.
- **Engines**: Caissa, Berserk (`(bestScore+beta)/2`), Stormphrax (`(eval+beta)/2`)
- **Status**: Currently running, +11.5 Elo at 1074 games, LLR 70% — nearing H1.

### 10. 50-Move Rule Eval Scaling (TESTING)
Scale eval toward zero as halfmove clock advances.
- **Engines**: Texel, Berserk (`(200-fmr)/200 * eval`)
- **Status**: Currently running, flat at 798 games.

### 11. Fail-High Score Blending (TESTING)
Blend beta cutoff score in main search: `(score*depth + beta)/(depth+1)`.
- **Engines**: Caissa, Stormphrax, Berserk
- **Status**: Currently running, +2.0 Elo at 537 games.

### 12. Opponent-Threats Guard on RFP/NMP ⭐ NEW
Compute "good threats" bitboard (pawn attacks non-pawn, minor attacks rook, rook attacks queen). Only allow RFP when opponent has no good threats.
- **Engines**: Minic (from Koivisto), Berserk (disables NMP when opponent has hanging Q/R)
- **Complexity**: Medium — need attack map computation at search time.
- **Est. Elo**: +3 to +8.

### 13. Fail-High Extra Reduction at Cut-Nodes ⭐ NEW
At cut-nodes where eval already exceeds beta by a margin, add +1 LMR reduction for quiet moves.
- **Engines**: Minic (`evalScore - margin > beta` at cutNode → reduction++)
- **Complexity**: Low — one condition in LMR formula.
- **Est. Elo**: +2 to +4.

### 14. Aspiration Fail-High Depth Reduction
On aspiration fail-high, reduce depth by 1 before re-searching with wider window.
- **Engines**: Midnight, ExChess, Seer, **Minic** (--windowDepth on fail-high), **Alexandria** (`depth = max(depth-1, 1)` on fail-high)
- **Complexity**: Trivial — 1 line in aspiration loop.
- **Est. Elo**: +1 to +3.
- **Note**: Our test got -353.8 Elo but was an implementation bug (modified outer loop depth). Alexandria's implementation confirms: only modify the inner loop depth variable.

### 14b. TT Cutoff Node-Type Guard ⭐ NEW (from Alexandria)
Only accept TT cutoffs when expected node type matches: `cutNode == (ttScore >= beta)`.
- **Engines**: **Alexandria** (in TT cutoff conditions alongside depth/bound checks)
- **Evidence**: Prevents stale TT data from contradicting expected node behavior. Cut nodes should only accept fail-highs, all nodes should only accept fail-lows.
- **Complexity**: Trivial — 1 extra condition in TT cutoff check.
- **Est. Elo**: +2 to +5.

### 14c. IIR-Aware RFP Margin (badNode flag) ⭐ NEW (from Alexandria)
When no TT data at all (`depth >= 4 && ttBound == NONE`), reduce RFP margin by a tuned offset (76cp in Alexandria). Also reduce NMP R by 1.
- **Engines**: **Alexandria** (`badNode` flag used in RFP margin `-76*canIIR`, NMP R `- badNode`, IIR `depth -= 1`)
- **Evidence**: Unifying concept: uncertain positions (no TT) should be more conservative with pruning.
- **Complexity**: Low — compute flag once, use in 3 places.
- **Est. Elo**: +2 to +5.

### 14d. Root History Table ⭐ NEW (from Alexandria)
Dedicated butterfly history table for root moves only, weighted 4x in quiet move scoring. Cleared before each search.
- **Engines**: **Alexandria** (`rootHistory[2][4096]`, separate bonus/malus tuning: 225/165/1780 bonus, 402/75/892 malus)
- **Evidence**: Root move ordering is disproportionately important. A dedicated table prevents contamination from non-root data.
- **Complexity**: Low — duplicate butterfly table, zero on search start, update only at root.
- **Est. Elo**: +3 to +8.

### 14e. TT Cutoff Continuation History Malus ⭐ NEW (from Alexandria)
When TT cutoff gives beta cutoff at a node where opponent has tried < 4 moves, penalize opponent's last quiet move in cont-hist.
- **Engines**: **Alexandria** (`updateCHScore((ss-1), (ss-1)->move, -min(155*depth, 385))`)
- **Evidence**: Uses TT cutoff information to improve opponent's move ordering: "your move led to a position we already know is lost for you."
- **Complexity**: Low — 3 lines in TT cutoff path.
- **Est. Elo**: +2 to +4.

### 14f. Eval-Based History Depth Bonus ⭐ NEW (from Alexandria)
When bestMove causes beta cutoff and eval was <= alpha (surprising cutoff), give +1 depth to history bonus.
- **Engines**: **Alexandria** (`UpdateHistories(... depth + (eval <= alpha) ...)`)
- **Evidence**: Moves that beat alpha from a poor position are more surprising and deserve stronger history reinforcement.
- **Complexity**: Trivial — change 1 argument.
- **Est. Elo**: +1 to +3.

### 14g. Material Scaling of NNUE Output ⭐ NEW (from Alexandria)
Scale NNUE eval by material count: `eval * (22400 + materialValue) / 32 / 1024`. Dampens eval in low-material endgames.
- **Engines**: **Alexandria** (material values: P=100, N=422, B=422, R=642, Q=1015)
- **Evidence**: NNUE tends to overestimate advantages in endgames. Simple post-processing correction.
- **Complexity**: Low — 3 lines in eval function.
- **Est. Elo**: +2 to +5.

### 15. Recapture Depth Reduction ⭐ NEW
When in a recapture position, not improving, and eval + best available capture < alpha, reduce depth by 1.
- **Engines**: Tucano (`!improving && eval + bestCapture < alpha → depth--`)
- **Complexity**: Easy — need `bestCapture()` helper + one condition.
- **Est. Elo**: +3 to +8.

### 16. Score-Drop Time Extension ⭐ MERGED
When ID score drops 20+ cp from previous iteration, extend time allowance significantly (2-4x).
- **Engines**: Tucano (running counter: -3 on 20cp drop, +1 on stable; sigmoid scaling)
- **Our status**: **(UPDATE 2026-03-21: GoChess now has score-drop time extension with 2.0x/1.5x/1.2x tiered scaling, merged.)**
- **Complexity**: Easy — strengthen existing score-delta time scaling.
- **Est. Elo**: +3 to +10.

### 17. NMP Less Reduction After Captures ⭐ MERGED
Reduce NMP R by 1 when the previous move was a capture (position is more forcing).
- **Engines**: Tucano (`R -= 1 if !move_is_quiet(lastMove)`)
- **(UPDATE 2026-03-21: GoChess now has NMP R-1 after captures, merged.)**
- **Complexity**: Trivial — one condition.
- **Est. Elo**: +2 to +5.

### 18. TT PV Flag in LMR
Reduce non-PV TT entries more aggressively in LMR.
- **Engines**: Berserk (+2 reduction if !ttPv), Stormphrax (+131/128 if !PV)
- **Complexity**: 2 lines. Store PV flag in TT, use in LMR adjustment.
- **Caveat**: Requires adding a PV flag bit to our packed TT entries.

### 19. Complexity-Aware RFP
Use correction history magnitude as "complexity" signal to adjust RFP margins.
- **Engines**: Stormphrax (complexity * scale/128 added to margin)
- **Complexity**: 3 lines. We already compute correction — just feed it into RFP.

### 20. Deeper Continuation History (Read-Only, Plies 4 & 6)
Use continuation history at plies 4 and 6 for move ordering and LMR, at half weight.
- **Engines**: Berserk (plies 1,2,4,6), Stormphrax (plies 1,2,4), RubiChess (plies 1,2,4,6), Caissa (plies 0,1,3,5), Seer, **Weiss** (plies 1,2,4 — skips 3), **Obsidian** (plies 1,2,4,6 — writes at half bonus to 4/6), **BlackMarlin** (ply -4), **Altair** (plies 1,2,4), **Reckless** (plies 1,2,4,6), **Alexandria** (plies 1,2,4,6 — both read and write)
- **Key distinction**: Read deeper contHist — Obsidian also writes to plies 4/6 at half bonus. **11 engines now**.

### 21. Piece/To History Table
Separate history table indexed by `[pieceType][toSquare]` (not from/to).
- **Engines**: Minic (historyP, combined with butterfly and CMH in scoring)
- **Complexity**: Low — small table (12×64), update alongside butterfly history.
- **Est. Elo**: +2 to +4.

### 22. Pawn History ⭐ NEW
Separate history table indexed by pawn structure hash (captures position-type patterns).
- **Engines**: **Obsidian** (`pawnHistory[pawnKey % PAWNHIST_SIZE][pieceTo]`), Weiss (similar)
- **Evidence**: Captures the idea that certain moves are good/bad depending on pawn structure.
- **Complexity**: Low-Medium — needs pawn hash integration, ~256KB table.
- **Est. Elo**: +3 to +8.

### 23. Eval History (Opponent Move Quality Feedback) ⭐ NEW
Update opponent's move history based on eval change caused by their move.
- **Engines**: **Obsidian** (`theirLoss = (ss-1)->staticEval + ss->staticEval - 58; bonus = clamp(-492*theirLoss/64, -534, 534)`), **Alexandria** (`bonus = clamp(-10*((ss-1)->staticEval + ss->staticEval), -1830, 1427) + 624; updateOppHHScore()`)
- **Evidence**: **2 engines now**. Both use sum of parent+current static eval as signal. Alexandria updates opponent's butterfly history directly.
- **Complexity**: Low — 4 lines, uses existing history infrastructure.
- **Est. Elo**: +2 to +5.

### 24. Node-Based Time Management ⭐ NEW
Use node count ratio (bestmove nodes / total nodes) to decide time allocation.
- **Engines**: **BlackMarlin**, Berserk, Caissa, Koivisto, **Alexandria** (`nodeScale = (1.53 - bestMoveNodesFrac) * 1.74`; also 5-level bestmove stability + 5-level eval stability multiplicative scaling)
- **Evidence**: When best move uses >80% of nodes, the move is stable — stop early. When <30%, the position is volatile — extend time. **5 engines now**.
- **Complexity**: Medium — track per-root-move node counts, adjust time based on ratio.
- **Est. Elo**: +5 to +15.

### 25. Cutnode LMR Extra Reduction ⭐ NEW
Extra LMR reduction (+2 plies) at cut nodes.
- **Engines**: **Weiss** (`r += 2 * cutnode`), **Obsidian** (similar), **BlackMarlin** (cut-node LMR)
- **Complexity**: Trivial — 1 line.
- **Est. Elo**: +2 to +5.

### 26. Opponent Material-Based LMR ⭐ NEW
Increase LMR reduction when opponent has few non-pawn pieces (endgame simplification).
- **Engines**: **Weiss** (`r += pos->nonPawnCount[opponent] < 2`)
- **Complexity**: Trivial — 1 line.
- **Est. Elo**: +1 to +3.

### 27. Complexity-Adjusted LMR ⭐ NEW
Use the difference between raw static eval and corrected eval as "complexity" signal to reduce LMR.
- **Engines**: **Obsidian** (`ss->complexity = abs(staticEval - rawStaticEval); R -= complexity / 120`)
- **Evidence**: High complexity = uncertain eval = search deeper. Low complexity = confident = can reduce more.
- **Complexity**: Low — 2 lines.
- **Est. Elo**: +2 to +4.

### 28. TT Move Noisy Detection for NMP/LMR ⭐ MERGED
If the TT best move is a capture/promotion, add extra LMR reduction for quiet moves.
- **Engines**: **Obsidian**, Berserk, **Reckless** (similar)
- **Result**: MERGED, +34.4 Elo. Commit 330dcd4.

### 29. "Failing" Heuristic (Position Deterioration) ⭐ NEW
Detect when eval has deteriorated significantly: `failing = staticEval < pastEval - (60 + 40*d)`. Use to increase LMR, tighten LMP.
- **Engines**: **Altair** (failing flag → +1 LMR, tighter LMP divider)
- **Complexity**: 1 line to compute, 2-3 lines to use.
- **Est. Elo**: +2 to +5.

### 30. Alpha-Raised Count in LMR ⭐ NEW
Track how many times alpha has been raised at current node. Each raise adds to LMR reduction for subsequent moves.
- **Engines**: **Altair** (`reduction += alpha_raised * (0.5 + 0.5*ttMoveNoisy)`), **Reckless** (`reduction += 1300 * alpha_raises`)
- **Complexity**: 3 lines.
- **Est. Elo**: +2 to +5.

### 31. Cutoff Count (Child Node Feedback) ⭐ NEW
Track beta cutoffs at child nodes. If child cutoff_count > threshold, increase parent LMR reduction.
- **Engines**: **Reckless** (`cutoff_count > 2 → reduction += 1604`; also gates NMP: `cutoff_count < 2 → NMP margin -= 20`)
- **Complexity**: 5 lines.
- **Est. Elo**: +3 to +8.

### 32. Halfmove Clock Eval Decay ⭐ NEW
Scale eval toward zero as 50-move rule approaches: `eval * (200 - halfmove) / 200`.
- **Engines**: **Reckless**, previously tested as "50-move eval scaling" (H0, -3.0 Elo) — but Reckless formula differs.
- **Complexity**: 1 line.
- **Note**: Our previous test may have used a different formula. Worth retesting with exact Reckless formula.

### 33. Main Search Beta-Cutoff Score Blending ⭐ NEW
When bestScore >= beta: `bestScore = (bestScore * min(depth,8) + beta) / (min(depth,8) + 1)`.
- **Engines**: **Reckless** (depth-weighted blending toward beta)
- **Complexity**: 2 lines.
- **Note**: We already have QS beta blending (+4.9 Elo). Main search version is deeper.
- **Est. Elo**: +2 to +5.

### 34. Bad Noisy Futility Pruning ⭐ NEW
Separate futility stage for losing captures (bad SEE): `margin = 71*d + 69*history/1024 + 81*capturedValue/1024`.
- **Engines**: **Reckless** (breaks move loop entirely for bad noisy moves)
- **Complexity**: 8 lines.
- **Est. Elo**: +2 to +5.

### 35. Countermove/Followup History Pruning (CMP/FMP) ⭐ NEW
Prune quiet moves at shallow depths when individual continuation history components are below threshold.
- **Engines**: **Igel** (CMP: depth <= 3, contHist < 0; FMP: depth <= 3, contHist2 < -2000)
- **Complexity**: 5 lines each.
- **Note**: More granular than our combined history pruning check.

### 36. Futility History Gate ⭐ NEW
Don't futility-prune moves with very strong combined history (>12000 non-improving, >6000 improving).
- **Engines**: **Igel** (`history + cmhist + fmhist < fpHistoryLimit[improving]`)
- **Complexity**: 2 lines.
- **Est. Elo**: +2 to +4.

### 37. History-Based Extensions ⭐ NEW
Extend by 1 ply when both contHist1 and contHist2 exceed 10000 for a move (off-PV only).
- **Engines**: **Igel** (`cmhist >= 10000 && fmhist >= 10000 → ext = 1`), **Altair** (deeper/shallower after LMR re-search: `+1 if score > bestScore + 80`)
- **Complexity**: 2 lines.
- **Est. Elo**: +2 to +5.

### 38. NMP Verification at Deep Searches ⭐ NEW
After NMP returns >= beta at depth > 15, run verification search. Prevents NMP from being fooled by zugzwang at high depth.
- **Engines**: **Altair** (depth > 15 → verification search)
- **Complexity**: 5 lines.
- **Note**: Classical technique. Most modern engines dropped it but Altair keeps it for deep searches only.

### 39. IIR Extra Reduction on PV Nodes ⭐ NEW
When no TT move found, reduce PV nodes by 2 instead of 1.
- **Engines**: **Altair** (`depth -= 1 + pvNode`)
- **Complexity**: 1 line change.
- **Est. Elo**: +1 to +3.

### 40. Aspiration Window Tightening ⭐ PARTIALLY IMPLEMENTED
Initial delta = 5-6 instead of 15. Also asymmetric widening: on fail-low, contract beta toward alpha.
- **Engines**: **Igel** (delta=5), **Altair** (delta=6+85/(depth-2); on fail-low: `beta = (3*alpha+5*beta)/8`)
- **Complexity**: 2-3 lines.
- **Note**: **(UPDATE 2026-03-21: GoChess now has asymmetric aspiration window contraction: fail-low (3a+5b)/8, fail-high (5a+3b)/8, delta=15, growth 1.5x. The contraction is implemented; tighter initial delta is still untested.)**

---

## Tier 3: Interesting but Lower Priority

### 29. Grandparent Killer
Use first killer from 2 plies earlier (grandparent) in move ordering, scored between killer[0] and counter-move.
- **Engines**: Minic (score +1700 between killer[0] +1900 and counter +1500)
- **Complexity**: Trivial — 2 comparisons.

### 30. Per-Move Futility Margin via History
Futility margin adjusted per-move based on history: `depth * (50 + histPercent)`. Bad-history moves get pruned earlier.
- **Engines**: Tucano (cutoff-ratio history drives margin)
- **Complexity**: Easy-Moderate.
- **Est. Elo**: +3 to +8 (but interacts with our existing futility tuning).

### 31. QSearch Recapture-Only at Depth > 5
After 5+ capture plies in QSearch, restrict to recaptures only.
- **Engines**: Minic (filters non-recaptures at qply > 5)
- **Complexity**: Very low — 4 lines.

### 32. History Extension with Auto-Tuning Threshold
Extend moves with high continuation history; dynamically tune threshold.
- **Engines**: RubiChess (adjusts ±1/256 per node based on actual rate)

### 33. NMP with Good Capture Guard
Only allow NMP if TT move is not a good capture (SEE ≤ threshold).
- **Engines**: Seer (SEE ≤ 226 guard)

### 34. LMR Based on Eval-Alpha Distance
More reduction when eval is far from alpha.
- **Engines**: Koivisto (`+min(2, abs(staticEval-alpha)/350)`)

### 35. New Threats in LMR
Reduce LMR for moves that create new threats.
- **Engines**: Koivisto (`-bitCount(getNewThreats(b, m))`)

### 36. Danger-Based Pruning Adaptation
Use king danger to adjust pruning/reduction aggressiveness.
- **Engines**: Minic (danger metric from king attack, gates all pruning at danger≥16, modulates SEE thresholds)
- **Note**: Minic also uses danger to adjust SEE capture/quiet thresholds dynamically. Requires computing danger at search time.

### 37. Draw Randomization
Return small random value on draws to prevent repetition-seeking.
- **Engines**: Koivisto (`8 - (nodes & 15)`)

### 38. Node Count Move Ordering (Near Root)
At ply < 3, boost moves that consumed more nodes in prior iterations.
- **Engines**: Caissa, Koivisto

### 39. Endgame Capture Extension
Extend captures of pieces ≥ knight when < 6 pieces remain.
- **Engines**: RubiChess

### 40. EBF-Based Time Stop
Predict whether there's time for the next iteration using effective branching factor.
- **Engines**: Minic (`usedTime * 1.2 * EBF > moveTime → stop`)

### 41. Two Counter-Moves per Parent
Store 2 counter-moves per `[piece][to]` instead of 1, FIFO-updated.
- **Engines**: Tucano

### 42. FinnyTable (NNUE Accumulator Cache) ⭐ MERGED
Per-king-bucket cache of board state + NNUE accumulator. On king refresh, compare cached vs actual and apply only deltas instead of full recompute.
- **Engines**: **Obsidian** (`FinnyEntry[2][KingBuckets]` with byColor/byPiece BBs + accumulator), **Alexandria** (`FinnyTableEntry[2][INPUT_BUCKETS][2]` with per-piece occupancy BBs + accumCache; indexed by [side][kingBucket][flip])
- **(UPDATE 2026-03-21: GoChess now has Finny tables for v5 NNUE accumulator refresh, merged.)**
- **Complexity**: Medium — need per-thread cache of accumulators per king bucket.
- **Est. NPS**: +5-10% (search speed, not Elo directly).

### 43. Time-Adaptive Pruning Enable ⭐ NEW
Enable forward pruning only after a time threshold (not by depth). Prevents aggressive pruning in early iterations.
- **Engines**: **Weiss** (`doPruning = usedTime >= optimalUsage/64 || depth > 2 + optimalUsage/270`)
- **Complexity**: Low — condition on pruning gates.

### 44. StatBonus Boost on Strong Fail-High ⭐ NEW
When bestScore exceeds beta by a tunable margin, use `statBonus(depth+1)` instead of `statBonus(depth)` for history updates.
- **Engines**: **Obsidian** (`bonus = statBonus(depth + (bestScore > beta + 95))`)
- **Complexity**: Trivial — 1 line.

---

## Already Have / Already Tested

These appeared in multiple engines but GoChess already has them or has tested:
- Continuation history (plies 1-2) ✓
- Counter-move heuristic ✓
- Improving flag ✓
- Correction history (pawn) ✓
- ProbCut ✓ (tightened to +170)
- Singular extensions ✓ (single only — still broken/disabled as of 2026-03-21, every attempt -60 to -140 Elo)
- IIR ✓ (tested true IID, neutral)
- Alpha-reduce ✓ (MERGED, +13.0 Elo)
- Check extension: tested and removed (-11.2 Elo). Tucano uses SEE-filtered variant — different but low priority given our result.
- NMP verification at depth 12 ✓
- NMP R-1 after captures ✓ (MERGED 2026-03-21)
- History pruning ✓
- FinnyTable ✓ (MERGED 2026-03-21 for v5 NNUE)
- Score-drop time extension ✓ (MERGED 2026-03-21, 2.0x/1.5x/1.2x tiered)
- Aspiration window contraction ✓ (MERGED 2026-03-21, fail-low (3a+5b)/8, fail-high (5a+3b)/8)
- SCReLU activation ✓ (v5 NNUE inference support with SIMD)
- Pairwise multiplication ✓ (v5 NNUE inference support with SIMD)
- Dynamic NNUE width ✓ (loads 1024/1536/768pw/any width)
- NMP TT guard: tested, 0 Elo (retry candidate)
- RFP score dampening: tested, -16.7 Elo
- History bonus by score diff: tested, -33.9 Elo
- Non-pawn correction (XOR bitboards): tested, -11.8 Elo (retry with Zobrist keys)
- ContHist2 ordering: tested, -7.5 Elo
- ContHist2 updates: tested, -7.5 Elo
- LMR reduce less for checks: tested, -8.1 Elo
- ProbCut MVV-LVA ordering: tested, -18.3 Elo
- SkipQuiets after LMP: tested, -35.0 Elo
- DoDeeper/DoShallower: tested, -13.7 Elo (thresholds may need calibration)
- Multi-cut in SE: tested, -28.5 Elo (SE framework itself harmful)
- QS two-sided delta: tested, -37.4 Elo (good-delta too aggressive)
- TT noisy detection: ✓ MERGED (+34.4 Elo)
- ASP fail-high depth reduce: tested, -353.8 Elo (implementation bug — modified outer loop depth). Alexandria confirms correct impl: modify inner loop depth only. Retry candidate.
- Futility 80+d*80: ✓ MERGED (+33.6 Elo)
- TT near-miss cutoffs: ✓ MERGED (+21.7 Elo)
- Contempt 10→15: tested, H0 (-5.2 Elo)

---

## Recommended Testing Order

Based on success pattern alignment, multi-engine consensus, and implementation ease. Updated with 26-engine review.

**Currently TESTING:**
- **Cutnode LMR +2** (Weiss/Obsidian/BlackMarlin — 110 games, +23.2 Elo)
- **NMP-170** (NMP divisor 200→170 — 483 games, +5 Elo, weakening)
- **ContHist4** (4-ply cont-hist at 1/4 weight — 49 games, -58 Elo, likely H0)
- ~~**NMP postcap** (R-1 after captures — 32 games, early)~~ **(MERGED 2026-03-21)**

**Next up (post-Alexandria review, re-prioritized):**
1. **FIX SINGULAR EXTENSIONS** (Alexandria-style: depth>=6, margin d*5/8, ply limiter, multi-cut, negative ext -2. Our SE is broken at -58 Elo — this is the single highest-value fix)
2. **TT cutoff node-type guard** (1 line, Alexandria — `cutNode == (ttScore >= beta)`)
3. **Mate distance pruning** (3 lines, Alexandria/Igel/many — universal, trivial)
4. **badNode flag in RFP/NMP** (Alexandria — reduce RFP margin and NMP R when no TT data)
5. **Root history table** (Alexandria — dedicated root ordering, 4x weight)
6. **Hindsight reduction** (2 lines, 5 engines — depth-- when both evals sum high)
7. **Multi-source correction history** (11+ engines — proper Zobrist non-pawn keys, Alexandria shows exact implementation)
8. **Node-based time management** (6 engines — +5-15 Elo potential)
9. **Alpha-raised count in LMR** (3 lines, Altair + Reckless — 2 engines, novel)
10. **Cutoff count child feedback** (5 lines, Reckless — child cutoffs → parent LMR increase)
11. **"Failing" heuristic** (3 lines, Altair — position deterioration → more aggressive pruning)
12. **Main search beta-cutoff score blending** (2 lines, Reckless — depth-weighted like our QS blend)
13. **ProbCut QS pre-filter** (2 engines — Alexandria + Tucano)
14. **TT cutoff CH malus** (Alexandria — penalize opponent's move on TT cutoff)
15. **Opponent eval history** (2 engines — Alexandria + Obsidian)
16. **Threat-aware history** (12 engines — highest consensus but medium complexity)
17. **Complexity-adjusted LMR** (Obsidian + Alexandria — 2 lines)
18. **Aspiration delta=12** (Alexandria — tighter than our 15; **(UPDATE 2026-03-21: GoChess now has fail-low/fail-high contraction (3a+5b)/8 and (5a+3b)/8, only tighter delta untested)**)
19. **Eval-based history depth bonus** (Alexandria — +1 depth when eval <= alpha)
20. **Material scaling of NNUE output** (Alexandria — cheap endgame correction)

---

## Alexandria Deep-Dive (2026-03-19)

Alexandria (v9.0.3) is CCRL #6, using similar NNUE architecture class to our migration target. Full review in `engine-notes/alexandria.md`.

### NNUE Architecture
- **Topology**: (768×16→1536)×2 → 16 → 32 → 1×8 with **dual activation** (linear + squared, doubles effective L2 input to 32)
- **Pairwise**: Clips to [0,255], multiplies pairs, shifts `>> 10`. Output is uint8 for int8 L1
- **Output buckets**: Non-linear `(63-pc)(32-pc)/225` — finer middlegame granularity
- **King buckets**: 16, mirrored horizontally (flip when king on files e-h)
- **Quantization**: FT_QUANT=255, L1_QUANT=64, NET_SCALE=362
- **FinnyTable**: Per-[side][bucket][flip] accumulator cache with per-piece occupancy BBs — avoids full recompute even on king bucket changes **(UPDATE 2026-03-21: GoChess now has Finny tables)**
- **Factoriser**: Shared 768×1536 base matrix added to all king bucket weights at quantization time
- **NNZ-sparse L1**: Tracks non-zero output chunks, only multiplies non-zero entries
- **SIMD**: AVX-512 + AVX2 (no ARM/NEON)

### Search — Key Differences from GoChess

**Singular Extensions** (CRITICAL — our SE is -58 to -85 Elo):
- Depth >= **6** (vs our 10), margin `ttScore - depth*5/8` (vs our `depth*3`)
- PV-aware: extra `depth*(ttPv && !pvNode)` reduces margin on former PV nodes
- Double ext at singularBeta-10, triple at singularBeta-75 (+ depth boost `depth += depth<10`)
- **Multi-cut**: returns singularScore directly if >= beta (powerful pruning shortcut)
- **Negative extensions**: -2 for ttScore >= beta AND for cut nodes
- **Extension limiter**: `ply*2 < RootDepth*5` prevents explosive tree growth
- **Diagnosis of our failure**: Missing extension limiter + negative extensions causes explosive tree growth. Missing multi-cut means we pay the SE cost without the pruning benefit. Margin too wide (30 at depth 10 vs 6.25 at depth 10).

**NMP**: Base R=4 (vs our 3), extra condition `staticEval >= beta - 28*depth + 204`, badNode reduces R by 1. Verification at depth >= 15 with `nmpPlies = ply + (depth-R)*2/3`.

**LMR**: Quiet base=1.07/div=2.27, Noisy base=-0.36/div=2.47. History divisor 8049 (vs our 5000). Separate noisy divisor 5922. DoDeeper margin 77+2*newDepth. PV node bonus: `reducedDepth + pvNode`.

**Correction History**: 4 tables (pawn, white non-pawn, black non-pawn, continuation) with weighted blend (29/34/34/26 weights / 256). Per-color Zobrist non-pawn keys updated incrementally in MakeMove.

**Eval Pipeline**: NNUE → 50-move decay `*(200-fmr)/200` → material scaling `*(22400+matValue)/32/1024` → correction → clamp.

**Other notable**: badNode flag (unifying concept for IIR/RFP/NMP), opponent history update, TT cutoff CH malus, eval-based history depth bonus, TT cutoff node-type guard `cutNode == (ttScore >= beta)`, draw score randomization `(nodes & 2) - 1`, upcoming repetition detection (Cuckoo), dynamic SEE threshold for good captures `-score/32 + 236`.

### Top Ideas to Test (prioritized by impact and alignment)

1. **Fix singular extensions** — Use Alexandria's parameters: depth>=6, margin d*5/8, ply limiter ply*2<RootDepth*5, multi-cut return, negative ext -2. This alone could recover massive Elo.
2. **4-table correction history** — Proper Zobrist non-pawn keys + continuation correction. 11+ engines have this.
3. **Node-based time management** — Triple scaling (nodes + bestmove stability + eval stability). 6 engines.
4. **Root history table** — Dedicated root move ordering, 4x weight. Easy.
5. **TT cutoff node-type guard** — 1 line, free.
6. **badNode flag** — Reduce RFP margin + NMP R when no TT data. 3 uses from 1 flag.
7. **ProbCut QS pre-filter** — Two-stage ProbCut. 2 engines now.
8. **Hindsight reduction** — 2 lines, 5 engines.
9. **Opponent eval history** — 2 engines now.
10. **TT cutoff CH malus** — Penalize opponent's move on TT cutoff. Novel.
11. **Material scaling of NNUE output** — Cheap post-processing.
12. **FinnyTable** — +5-10% NPS, 2 top engines. **(UPDATE 2026-03-21: MERGED)**
