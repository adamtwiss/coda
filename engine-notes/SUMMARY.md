# Engine Review Summary — Ideas for Coda

Cross-referencing 34 engines against Coda. Ranked by estimated impact, implementation complexity, and alignment with our proven success patterns.

**Engines reviewed**: Ethereal, Caissa, Midnight, Winter, Texel, Wasp, Arasan, Berserk, Koivisto, Stormphrax, RubiChess, Seer, Minic, Tucano, Weiss, Obsidian, BlackMarlin, Altair, Reckless, Igel, Alexandria, Stockfish, Viridithas, Halogen, PlentyChess, Quanticade, **Clarity** (+200 above us), **Velvet** (+100 above us), **Astra**, **Clover**, **Horsie**, **Tarnished**, **integral** (added 2026-04-19)

**Removed** (too weak to be informative): Crafty, ExChess, GreKo, Rodent III, Laser

**Our success patterns** (from GoChess era, still valid):
1. Score dampening works at noisy boundaries (TT cutoffs, QS) — not at well-calibrated margins (RFP)
2. Search byproducts improve move ordering (NMP threat, TT move in QS)
3. Table-based approaches beat hard-coded thresholds (LMR split)
4. ~~History table changes tend to fail~~ History magnitude fix (+31.6 Elo) changed everything — history is now the primary ordering signal
5. Node-level pruning benefits from tightening; per-move pruning needs slack
6. Guards that prevent pruning in tactical positions (threats, captures) are high-value

**Key ablation finding (2026-04-01)**: All pruning features confirmed helpful via self-play SPRT on quiet CPU. ProbCut re-enabled after fixing missing qsearch filter, SEE threshold, and excluded_move guard. Bad noisy pruning (-26 Elo), hindsight (-18), hist prune (-17), SEE prune (-17), correction (-10), extensions (-9) all confirmed valuable.

**Updated 2026-04-12**: Many Tier 1/2 ideas now implemented. Killers/counters removed (history handles ordering). 48 SPSA-tunable parameters. Production net: 768pw v5 with filtered data + low final LR + power-2.5 loss.

---

## IMPLEMENTED — Already in Coda

These ideas from engine reviews have been implemented and confirmed:

| Feature | Source engines | Status |
|---------|--------------|--------|
| LMR separate tables (cap C=1.80 / quiet C=1.30) | Midnight, Caissa | Implemented |
| TT score dampening `(3*score+beta)/4` | Winter, Caissa | Implemented |
| Fail-high score blending `(score*d+beta)/(d+1)` | Caissa, Stormphrax, Berserk | Implemented |
| ~~Alpha-raise reduction~~ | Caissa, Altair, Reckless | Removed — SPSA zeroed it out |
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
| NMP R+1 after captures (flipped from R-1) | Tucano | Implemented, retune +3.5 |
| Recapture extensions | Multiple | Implemented |
| Bad noisy pruning (eval+depth*75<=alpha) | Reckless | Implemented |
| Score-drop time extension | Tucano | Implemented |
| IIR (depth >= 6, no TT move) | Multiple | Implemented |
| NMP verification (depth >= 12) | Altair, classical | Implemented |
| Node-based TM (node fraction + stability + score drop) | BlackMarlin, Berserk, Clarity | Implemented |
| TT PV flag in LMR (-1 reduction for PV positions) | Berserk, Stormphrax, Clarity | Implemented, +4.5 |
| TT cutoff cont-hist malus | Alexandria | Implemented, +6.5 (with retune) |
| Complexity-adjusted LMR (raw-corrected eval) | Obsidian | Implemented |
| Killer/counter-move removal (pure history ordering) | Stockfish | Implemented, +3.77 (with retune) |
| Quiet check bonus in move ordering | Multiple | Implemented, +6.0 |
| Double extensions (with ply cap) | Seer, Berserk, Clarity, Velvet | Implemented |
| ProbCut (re-enabled after fixes) | Multiple | Implemented |
| Contempt (draw avoidance) | Multiple | Implemented, SPSA-tunable |

---

## Tier 1: High Confidence — Test Next

### 1. ~~Node-Based Time Management~~ ✅ IMPLEMENTED
~~Use node count ratio (bestmove nodes / total nodes) + best-move stability multiplier.~~
- **Status**: Implemented. 3-factor model: node fraction + best-move stability + score trend.

### 2. Mate Distance Pruning
If we already have a forced mate shorter than current ply, prune.
- **Engines**: Tucano, RubiChess, Altair, Alexandria, **Clarity**, **Velvet**
- **Complexity**: Trivial — 5 lines at top of negamax.
- **Est. Elo**: +1 to +3. Universal technique, 6+ engines.
- **Note**: Both engines +100-200 above us have this. We don't. Tested neutral on old baseline (OB #163), worth retrying with current eval.

### 3. ~~Double/Triple Singular Extensions~~ ✅ IMPLEMENTED
- **Status**: Implemented with DEXT_MARGIN and DEXT_CAP tunables. Ply guard included.

### 4. Cutnode LMR Extra Reduction
Extra LMR reduction (+2 plies) at cut nodes.
- **Engines**: Weiss, Obsidian, BlackMarlin, Stockfish, **Clarity** (+2)
- **Complexity**: Trivial — 1 line.
- **Est. Elo**: +2 to +5.
- **Note**: Not yet tested in Coda.

### 5. ~~Root History Table~~ ❌ REJECTED
- **Status**: Tested 3× (OB era), -0.8 to -7.9 Elo. Root ordering already good with 4D history.

### 6. TT Cutoff Node-Type Guard
Only accept TT cutoffs when expected node type matches: `cutNode == (ttScore >= beta)`.
- **Engines**: Alexandria
- **Complexity**: Trivial — 1 extra condition in TT cutoff check.
- **Est. Elo**: +2 to +5.
- **Note**: Not yet tested.

### 7. badNode Flag (IIR-Aware Pruning)
When no TT data at all (`depth >= 4 && ttBound == NONE`), reduce RFP margin, reduce NMP R by 1.
- **Engines**: Alexandria (`badNode` flag used in RFP margin, NMP R, IIR)
- **Complexity**: Low — compute flag once, use in 3 places.
- **Est. Elo**: +2 to +5.
- **Note**: Not yet tested.

### 8. ~~50-Move Eval Scaling~~ ❌ REJECTED
- **Status**: Tested twice (OB #223, GoChess era). Neutral both times.

### 9. Opponent-Threats Guard on RFP/NMP
Compute "good threats" bitboard. Only allow RFP when opponent has no good threats.
- **Engines**: Minic (from Koivisto), Berserk
- **Complexity**: Medium — need attack map computation.
- **Est. Elo**: +3 to +8.
- **Note**: Not yet tested.

### 10. ~~TT PV Flag in LMR~~ ✅ IMPLEMENTED
- **Status**: Implemented. Sticky PV bit, -1 LMR for PV positions. +4.5 Elo.

### 11. ~~TT Cutoff Continuation History Malus~~ ✅ IMPLEMENTED
- **Status**: Implemented with retune. +6.5 Elo.

### 12. ~~Lower SE Depth Threshold~~ ✅ IMPLEMENTED (SPSA-tunable)
- **Status**: SE_DEPTH is SPSA-tunable, currently at 5 (SPSA converged from 8).

---

## Tier 2: Promising — Worth Testing

### 9. Eval-Based History Depth Bonus
When bestMove causes beta cutoff and eval was <= alpha (surprising cutoff), give +1 depth to history bonus.
- **Engines**: Alexandria (`UpdateHistories(... depth + (eval <= alpha) ...)`)
- **Complexity**: Trivial — change 1 argument.
- **Note**: Not yet tested.

### 10. Material Scaling of NNUE Output
Scale NNUE eval by material count: `eval * (22400 + materialValue) / 32 / 1024`.
- **Engines**: Alexandria
- **Complexity**: Low — 3 lines in eval function.
- **Note**: Not yet tested.

### 11. ~~Complexity-Adjusted LMR~~ ✅ IMPLEMENTED
- **Status**: Implemented. LMR_COMPLEXITY_DIV is SPSA-tunable (currently 118).

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

- **Razoring**: Fully removed from codebase. Confirmed -20 Elo on quiet CPU.
- **NMP threat detection**: Was in GoChess, not ported to Coda. May retest later.
- **50-move eval scaling**: Tested neutral twice (OB #223, GoChess era). Both directions tried.
- **RFP score dampening**: Tested negative — RFP margins are well-calibrated, dampening hurts.
- **Root history table**: Tested 3× (OB era), -0.8 to -7.9. Root ordering good with 4D history.
- **Alpha-raise LMR reduction**: Removed via SPSA — retune found it unhelpful. Was in "implemented" but SPSA zeroed it out.
- **Good/bad quiet split**: SF-only feature, tested -3.6 (OB #216). May revisit with better nets.
- **Low-ply history bonus**: Tested -2.9 (OB #209). 8× at ply 0 doesn't help.

---

## 2026-04-19 additions (Astra, Clover, Horsie, Tarnished, integral)

Full reviews in `engine-notes/{astra,clover,horsie,tarnished,integral}.md`. Consensus patterns across the 5:

### New Tier 1 candidates

- **Eval-difference quiet history bonus** (Clover + Horsie): retroactively update the previous quiet move's history based on eval-change after it was played. Formula variant: `bonus = -(prev_eval + curr_eval) / 16`. Novel signal, trivial change. Est. +3-8 Elo. **Recommended next.**
- **Material-scaled NNUE output** (Astra + Integral): `eval *= (230 + material/42) / 440` or `eval *= (27600 + mat_popcount) / 32768`. One-line weakening of eval as pieces leave the board. Est. +2-5 Elo.

### New Tier 2 candidates

- **Logarithmic LMR formula** (Clover): `reduction = log(d) * log(n) / C` vs our lookup table. Simpler, possibly smoother tuning surface. Medium-high effort refactor.
- **Post-LMR depth adjustment** (Horsie + Integral): after re-search, adjust depth ±1 based on score-delta magnitude (e.g., `newer += (score > bestScore + 45 + 4*d) - (score < bestScore + d)`).
- **History pruning at very low depth** (Tarnished): explicit `hist < -SCALE*d` gate at d≤4. We have HIST_PRUNE_MULT=5148 — worth cross-checking against Tarnished's SCALE=2820.

### Architectural validation (already doing it, confirmation from new engines)

- **Linear-with-offset history bonus shape** — Astra's tuned values (MULT=308, OFFSET=4) validate the direction of our in-flight Experiment 1 (`experiment/history-shape-offset`, SPSA #515). If our SPSA drifts OFFSET past 50, we're in Astra's basin.
- **Multi-source correction history** — 4 sources (Astra), 5 sources (Clover), 6 sources (Tarnished). We have 5 (pawn/NP/minor/major/cont). Not a gap.
- **Cumulative continuation history over {1,2,4,6} plies** — Integral uses the sum; we already use all 4 offsets. Not a gap.

### Novel-to-us training ideas (not easily ported)

- **Net distillation** (Tarnished): train large (4096 L1) net, distill into small (1792 L1) production net. Needs training infrastructure rewrite; speculative +20-40 Elo if it works.
- **Sparse L1 via NNZ bitmask detection** (Horsie): if our pairwise activation also sparsifies, bitmask-skip zero lanes in L1 matmul. ~5-10% NPS. Implementation effort: medium.
- **Killers/counters**: Removed (+3.77 Elo). History tables fully replaced them post magnitude fix.
