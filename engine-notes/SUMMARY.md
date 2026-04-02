# Engine Review Summary — Ideas for Coda

Cross-referencing 29 engines against Coda. Ranked by estimated impact, implementation complexity, and alignment with our proven success patterns.

**Engines reviewed**: Ethereal, Caissa, Midnight, Winter, Texel, Wasp, Arasan, Berserk, Koivisto, Stormphrax, RubiChess, Seer, Minic, Tucano, Weiss, Obsidian, BlackMarlin, Altair, Reckless, Igel, Alexandria, Stockfish, Viridithas, Halogen, PlentyChess, Quanticade

**Removed** (too weak to be informative): Crafty, ExChess, GreKo, Rodent III, Laser

**Our success patterns** (from GoChess era, still valid):
1. Score dampening works at noisy boundaries (TT cutoffs, QS) — not at well-calibrated margins (RFP)
2. Search byproducts improve move ordering (NMP threat, TT move in QS)
3. Table-based approaches beat hard-coded thresholds (LMR split)
4. History table changes tend to fail — our tables are well-tuned
5. Node-level pruning benefits from tightening; per-move pruning needs slack
6. Guards that prevent pruning in tactical positions (threats, captures) are high-value

**Key ablation finding (2026-04-01)**: All pruning features confirmed helpful via self-play SPRT on quiet CPU. Only ProbCut is dead weight (disabled). Bad noisy pruning (-26 Elo), hindsight (-18), hist prune (-17), SEE prune (-17), correction (-10), extensions (-9) all confirmed valuable.

---

## IMPLEMENTED — Already in Coda

These ideas from engine reviews have been implemented and confirmed:

| Feature | Source engines | Status |
|---------|--------------|--------|
| LMR separate tables (cap C=1.80 / quiet C=1.30) | Midnight, Caissa | Implemented |
| TT score dampening `(3*score+beta)/4` | Winter, Caissa | Implemented |
| Fail-high score blending `(score*d+beta)/(d+1)` | Caissa, Stormphrax, Berserk | Implemented |
| Alpha-raise reduction (alpha_raised_count/2) | Caissa, Altair, Reckless | Implemented |
| QS beta blending (bestScore+beta)/2 | Caissa, Berserk, Stormphrax | Implemented |
| TT near-miss cutoffs (1 ply, 80cp margin) | Minic, Ethereal | Implemented |
| TT noisy move detection (+1 LMR for quiets) | Obsidian, Berserk | Implemented |
| DoDeeper/DoShallower after LMR re-search | Stockfish, Weiss, Obsidian, Berserk | Implemented |
| Multi-source correction history (5 tables) | Stormphrax, Seer, Obsidian, Alexandria | Implemented |
| Continuation history plies 1,2,4,6 (writes at half bonus) | Berserk, Obsidian, Reckless, Alexandria | Implemented |
| Pawn history [pawnHash%512][piece][to] | Obsidian, Weiss | Implemented |
| 4D threat-aware history [from_thr][to_thr][from][to] | Ethereal, Berserk, Stormphrax | Implemented |
| Hindsight reduction (parent compensation) | Stormphrax, Berserk, Tucano, Alexandria | Implemented |
| Singular extensions (positive/double/multi-cut/negative) | All strong engines | Implemented |
| Cuckoo cycle detection | Weiss, Stockfish | Implemented |
| Finny table (NNUE accumulator cache) | Obsidian, Alexandria | Implemented |
| Aspiration fail-low beta contraction | Midnight, Altair, Alexandria | Implemented |
| NMP R-1 after captures | Tucano | Implemented |
| Recapture extensions | Multiple | Implemented |
| Bad noisy pruning (eval+depth*75<=alpha) | Reckless | Implemented |
| Score-drop time extension | Tucano | Implemented |
| IIR (depth >= 6, no TT move) | Multiple | Implemented |
| NMP verification (depth >= 12) | Altair, classical | Implemented |

---

## Tier 1: High Confidence — Test Next

### 1. Node-Based Time Management
Use node count ratio (bestmove nodes / total nodes) to decide time allocation. When best move uses >80% of nodes, stop early. When <30%, extend time.
- **Engines**: BlackMarlin, Berserk, Caissa, Koivisto, Alexandria (`nodeScale = (1.53 - bestMoveNodesFrac) * 1.74`)
- **Complexity**: Medium — track per-root-move node counts.
- **Est. Elo**: +5 to +15. Highest expected gain of any untested idea.
- **5+ engines** — strong consensus.

### 2. Root History Table
Dedicated butterfly history table for root moves only, weighted 4x in quiet move scoring. Cleared before each search.
- **Engines**: Alexandria (`rootHistory[2][4096]`), Stockfish
- **Complexity**: Low — duplicate butterfly table, zero on search start, update only at root.
- **Est. Elo**: +3 to +8.

### 3. Cutnode LMR Extra Reduction
Extra LMR reduction (+2 plies) at cut nodes.
- **Engines**: Weiss (`r += 2 * cutnode`), Obsidian, BlackMarlin, Stockfish
- **Complexity**: Trivial — 1 line.
- **Est. Elo**: +2 to +5.

### 4. TT Cutoff Node-Type Guard
Only accept TT cutoffs when expected node type matches: `cutNode == (ttScore >= beta)`.
- **Engines**: Alexandria
- **Complexity**: Trivial — 1 extra condition in TT cutoff check.
- **Est. Elo**: +2 to +5.

### 5. badNode Flag (IIR-Aware Pruning)
When no TT data at all (`depth >= 4 && ttBound == NONE`), reduce RFP margin, reduce NMP R by 1.
- **Engines**: Alexandria (`badNode` flag used in RFP margin, NMP R, IIR)
- **Complexity**: Low — compute flag once, use in 3 places.
- **Est. Elo**: +2 to +5.

### 6. Opponent-Threats Guard on RFP/NMP
Compute "good threats" bitboard (pawn attacks non-pawn, minor attacks rook, rook attacks queen). Only allow RFP when opponent has no good threats.
- **Engines**: Minic (from Koivisto), Berserk
- **Complexity**: Medium — need attack map computation.
- **Est. Elo**: +3 to +8.

### 7. TT PV Flag in LMR
Reduce non-PV TT entries more aggressively in LMR.
- **Engines**: Berserk (+2 reduction if !ttPv), Stormphrax
- **Complexity**: 2 lines. Store PV flag in TT.
- **Caveat**: Requires adding a PV flag bit to packed TT entries.

### 8. TT Cutoff Continuation History Malus
When TT cutoff gives beta cutoff, penalize opponent's last quiet move in cont-hist.
- **Engines**: Alexandria (`updateCHScore((ss-1), (ss-1)->move, -min(155*depth, 385))`)
- **Complexity**: Low — 3 lines in TT cutoff path.
- **Est. Elo**: +2 to +4.

---

## Tier 2: Promising — Worth Testing

### 9. Eval-Based History Depth Bonus
When bestMove causes beta cutoff and eval was <= alpha (surprising cutoff), give +1 depth to history bonus.
- **Engines**: Alexandria (`UpdateHistories(... depth + (eval <= alpha) ...)`)
- **Complexity**: Trivial — change 1 argument.

### 10. Material Scaling of NNUE Output
Scale NNUE eval by material count: `eval * (22400 + materialValue) / 32 / 1024`.
- **Engines**: Alexandria
- **Complexity**: Low — 3 lines in eval function.

### 11. Complexity-Adjusted LMR
Use difference between raw static eval and corrected eval as "complexity" signal.
- **Engines**: Obsidian (`R -= complexity / 120`)
- **Complexity**: Low — 2 lines. Already have raw/corrected eval.

### 12. Opponent Material-Based LMR
Increase LMR reduction when opponent has few non-pawn pieces.
- **Engines**: Weiss (`r += pos->nonPawnCount[opponent] < 2`)
- **Complexity**: Trivial — 1 line.

### 13. Eval History (Opponent Move Quality Feedback)
Update opponent's move history based on eval change caused by their move.
- **Engines**: Obsidian, Alexandria
- **Complexity**: Low — 4 lines.

### 14. Cutoff Count (Child Node Feedback)
Track beta cutoffs at child nodes. If child cutoff_count > threshold, increase parent LMR reduction.
- **Engines**: Reckless (`cutoff_count > 2 → reduction += 1604`)
- **Complexity**: 5 lines.

### 15. Fail-High Extra Reduction at Cut-Nodes
At cut-nodes where eval already exceeds beta by a margin, add +1 LMR reduction for quiet moves.
- **Engines**: Minic
- **Complexity**: Low — one condition in LMR formula.

### 16. History-Based Extensions
Extend by 1 ply when both contHist1 and contHist2 exceed 10000 for a move.
- **Engines**: Igel, Altair
- **Complexity**: 2 lines.

### 17. Futility History Gate
Don't futility-prune moves with very strong combined history (>12000).
- **Engines**: Igel
- **Complexity**: 2 lines.

### 18. StatBonus Boost on Strong Fail-High
When bestScore exceeds beta by a margin, use `statBonus(depth+1)` for history updates.
- **Engines**: Obsidian (`bonus = statBonus(depth + (bestScore > beta + 95))`)
- **Complexity**: Trivial — 1 line.

### 19. Aspiration Window Tighter Initial Delta
Initial delta = 5-6 instead of 15.
- **Engines**: Igel (delta=5), Altair (delta=6+85/(depth-2))
- **Note**: Our current delta=15 works, but tighter initial delta resolves faster for stable positions.

---

## Tier 3: Lower Priority / Experimental

### 20. Mate Distance Pruning
If we already have a forced mate shorter than the current ply distance, prune.
- **Engines**: Tucano, RubiChess, Altair, Alexandria
- **Complexity**: 5 lines. Classical technique.

### 21. Grandparent Killer
Use first killer from 2 plies earlier in move ordering.
- **Engines**: Minic (score +1700)
- **Complexity**: Trivial.

### 22. QSearch Recapture-Only at Depth > 5
After 5+ capture plies, restrict to recaptures only.
- **Engines**: Minic
- **Complexity**: Very low — 4 lines.

### 23. Per-Move Futility Margin via History
Futility margin adjusted per-move based on history.
- **Engines**: Tucano
- **Complexity**: Easy-Moderate.

### 24. Two Counter-Moves per Parent
Store 2 counter-moves per `[piece][to]` instead of 1, FIFO-updated.
- **Engines**: Tucano
- **Complexity**: Low.

### 25. NMP with Good Capture Guard
Only allow NMP if TT move is not a good capture.
- **Engines**: Seer (SEE ≤ 226 guard)
- **Complexity**: 1 line.

### 26. Draw Randomization
Return small random value on draws to prevent repetition-seeking.
- **Engines**: Koivisto (`8 - (nodes & 15)`)
- **Complexity**: 1 line.

### 27. Halfmove Clock Eval Decay
Scale eval toward zero as 50-move rule approaches.
- **Engines**: Reckless, Berserk
- **Note**: Previously tested as neutral, but formula may have differed.

### 28. Time-Adaptive Pruning Enable
Enable forward pruning only after a time threshold.
- **Engines**: Weiss
- **Complexity**: Low.

---

## Rejected / Not Applicable

- **ProbCut**: Disabled in Coda. Ablation confirmed dead weight. Two rewrite attempts failed (-24 and -2 Elo). Not viable at current eval quality.
- **Razoring**: Fully removed from codebase. Confirmed -20 Elo on quiet CPU.
- **NMP threat detection**: Was in GoChess, not ported to Coda. May retest later.
- **50-move eval scaling**: Tested neutral in GoChess era.
- **RFP score dampening**: Tested negative — RFP margins are well-calibrated, dampening hurts.
