# Wasp Chess Engine - Crib Notes

Author: John Stanback. Written in C. Closed source (MIT license, but source not publicly distributed).
Current version: 7.00 (June 2024). ~3200+ CCRL.

Primary sources: [Technical page](https://waspchess.com/wasp_technical.html),
[Release notes](https://waspchess.com/release_notes.txt),
[CPW](https://www.chessprogramming.org/Wasp).

---

## 1. Search Framework

- Iteratively-deepened alpha-beta negamax with PVS and aspiration windows
- Quiescence search: captures, checks, and promotions only
- Hash table NOT cleared between moves; reserved for main search only
- Effective branching factor: ~1.75
- Aspiration window: 15cp (v3.5+); widened for TB win/loss or mate scores (v6.50)

---

## 2. Pruning Techniques

### Null Move Pruning (NMP)
- **Conditions**: depth >= 3, eval >= beta, not in check, not pawn ending
- **Reduction**: R=3 if depth < 8; R=4 if depth >= 8 (changed to depth >= 10 in v4.0)
- **Verification search**: depth-4 verification if fails high at depth >= 5 (v3.5+)
- *Compare to GoChess*: We use R = 3 + depth/3 + min((eval-beta)/200, 3). Wasp uses simpler fixed R with verification.

### Cut Node Pruning / Static Null Move Pruning (Reverse Futility Pruning)
- **Conditions**: depth <= 3, eval >= beta + margin, not in check, no mate/promotion threat, not pawn ending
- **Margin**: MAX(hung_material/2, 75*depth)
- Alternative form (v3.0+): depth <= 2, eval >= beta + 25cp, or margin = swapoff(best safe capture) + 100cp
- v6.00: Also detects hung pieces and enemy pawn threats on 6th/7th ranks before pruning
- v6.50: Allows static null-move pruning even with mated beta score
- **Notable**: The hung-material-aware margin is unusual. Most engines use a simple linear margin.
- *Compare to GoChess*: We use 85*depth improving, 60*depth non-improving, depth <= 7.

### Futility Pruning
- **Condition**: non-PV nodes, eval < alpha - (100 + 100*depth)
- When triggered: only generates captures/checks/promotions/pawn-to-7th (skip quiets)
- Combined with razoring at low depth (v3.5+)
- *Compare to GoChess*: We use 100 + 100*lmrDepth. Wasp uses actual depth, not reduced depth.

### Razoring
- Low depth, static eval well below alpha -> quiescence search only
- Combined with futility (v3.0+); uses QS at current depth to retain active moves
- *Compare to GoChess*: We use 400 + depth*100, depth <= 3.

### Late Move Pruning (LMP)
- v1.25: Prune non-tactical moves after 1 + 3*depth moves tried
- Additional condition: prune if moves >= 3 AND eval < alpha - 100*depth
- v6.50: Less aggressive if eval is improving
- *Compare to GoChess*: We use 3 + depth^2 (non-improving) / 1.5x that (improving).

### ProbCut (v3.0+)
- **Conditions**: depth >= 4, static eval >= beta
- Searches safe captures to depth-3
- Prunes if score >= beta + 100
- v5.00: Eliminated for winning captures that already fail-high with reduced depth
- *Compare to GoChess*: We use beta + 200.

### SEE Pruning
- Uses "swapoff" (recursive alpha-beta on a single square) to determine if captures are winning/losing
- Losing captures deferred to last stage of move ordering
- *No explicit SEE threshold pruning in search mentioned* -- handled via move ordering stages

---

## 3. Move Ordering

### Stages (5-stage generation)
1. **Hash move** (or IID move if no hash move)
2. **Winning captures/promotions** (scored by MVV-LVA, filtered by SEE)
3. **Decent quiet moves** (killer moves, countermoves, history-sorted)
4. **Losing captures** (SEE-negative captures deferred here)
5. **Poor quiet moves** (remaining low-history quiets)

Note: "pawn to 7th" was a separate stage in earlier versions (v1.25), between captures and quiets.

### History Tables
- **Main history**: standard history heuristic for quiet moves
- **Countermove history** (v4.0+): indexed by [color][prev_piece_type][curr_piece_type][to_square]
  - This is unusual -- most engines index countermove by [prev_piece][prev_to] only
  - Wasp includes current piece type in the index, making it a form of continuation history
- **Killer moves**: standard 2-killer table
- **Refutation table**: mentioned separately from killers (likely countermove table)
- No explicit mention of: capture history, continuation history (1-ply/2-ply), correction history

### Capture Scoring
- MVV-LVA (Most Valuable Victim, Least Valuable Attacker)
- SEE ("swapoff") to classify winning vs losing captures

### Internal Iterative Deepening (IID)
- When no hash move: R=3 for PV nodes, R=4 for non-PV nodes
- *Compare to GoChess*: We use IIR (reduce depth by 1) instead of IID.

---

## 4. Late Move Reduction (LMR)

### Formula Evolution
- **v1.25**: Reduce by 3 plies if depth >= 6 AND moves_searched >= 40 - 2*depth
- **v2.80**: Reduction varies 0-3 plies based on depth and move count
  - Tactical moves (captures, checks, promotions, pawn-to-7th): NOT reduced at PV nodes, max 1 ply at non-PV
  - PV nodes get ~1 ply less reduction than non-PV nodes
  - "Mirrors Stockfish's approach but less aggressively"
- **v3.5**: +1 extra ply reduction if move is unsafe OR score dropped >= 1.5 pawns from 2 plies ago
- **v4.0**: "Slightly higher depth reduction as depth and move count increase" (tuned upward)
- **v4.5**: Less aggressive at PV, more aggressive at non-PV; less aggressive if eval improving
- **v6.50**: Further tweaks to criteria

### Conditions
- depth >= 3, moves_searched >= 1, not in check
- Tactical moves limited or excluded from reduction (especially at PV)

### Adjustments
- **PV nodes**: ~1 ply less reduction
- **Improving**: Less aggressive reduction (v4.5+)
- **Unsafe moves**: +1 ply extra reduction (v3.5+)
- **Score drop**: +1 ply if eval dropped >= 150cp from 2 plies earlier (v3.5+)
- **Tactical moves**: Not reduced at PV; max 1 ply at non-PV (safe) or 2 plies (unsafe) in v4.5

### Notable
- Max reduction capped at 3 plies (at least in v2.80)
- The "score drop from 2 plies ago" adjustment is uncommon -- most engines don't track this
- No mention of history-based LMR adjustments (unlike SF/GoChess)
- *Compare to GoChess*: We use log(depth)*log(moveCount) formula with C=1.5, plus history adjustments.

---

## 5. Extensions

- **Check extension**: +1 ply when in check (always, even if checking piece capturable, v4.5+)
- **Pawn to 7th**: +1 ply for safe pawn push to 7th rank (v5.00: also 6th rank if depth <= 2)
- **Last piece capture**: +1 ply when capturing the last remaining enemy piece
- **Extension limit**: Max 2 plies total, except at PV nodes or depth <= 4

### Notable Absences
- **No singular extensions** -- "Singular Extension does not seem beneficial for Wasp"
- No fractional extensions in current version (removed v5.20)
- No recapture extensions
- *Compare to GoChess*: We have singular extensions and recapture extensions. No check extensions.

---

## 6. Time Management

Limited details available:
- v3.0: "Simplified time management; increased fraction of remaining time usable when failing low/high or score drops"
- **Move overhead**: UCI option, default 50ms, subtracted from target search time
- Ponder support (UCI, default true)
- v4.5, v7.00: "Modified time management" (no specifics)

### Selectivity UCI Option
- Default 100; increase for deeper/narrower search, decrease for wider/shallower search
- Adjusts pruning aggressiveness globally
- *This is unusual* -- most engines don't expose a global selectivity knob

### Fail-Low / Instability
- Score drops and fail-lows cause increased time allocation (v3.0+)
- No specific multiplier values documented
- *Compare to GoChess*: We use instability multiplier of 200 (2x time on PV changes).

---

## 7. Parallel Search (Lazy SMP)

- Shared hash table only -- each thread has own eval hash and pawn hash (v2.80+)
- Up to 160 threads supported
- Thread skipping: if >1 + nthreads/2 threads at depth N, skip to N+1 (v3.5)
- v6.50: if >15 threads at given depth, skip to depth+1; if >45, skip to depth+2
- Master thread updates PV/score; others just contribute to hash table
- v7.00: Hash store/probe modified for depth < 0 (stores QS results)

### Notable
- Storing QS results (depth < 0) in hash table is interesting -- most engines don't do this
- The thread-skipping thresholds are unusually detailed for high thread counts

---

## 8. NNUE / Evaluation

### v7.00 Architecture (HalfKA-like)
- **Input**: 768 piece-square features (12 pieces x 64 squares, no kings in features)
- **King buckets**: 2 zones per side (ranks 1-3 vs ranks 4-8)
- **Hidden layer**: 2x1536 neurons (white perspective + black perspective), leaky ReLU
- **Output**: 5 nodes, selected by non-pawn material on board
- **Weight sets**: 2 sets from input->hidden (by king zone), 2 sets from hidden->output (by side to move)
- Square mirroring for king position symmetry
- Integer math for inference (~18% speedup over float, v6.00+)
- Incremental update via SIMD (vector copy/add/subtract/dot-product)

### Training
- ~800M+ positions from Wasp vs similar-strength engines
- Depth 7-8 searches for position scores
- Target: 65% search score + 35% game result (v7.00)
- 300M positions/epoch, weights updated every 72K positions
- Momentum added to gradient descent (v7.00)
- Net merging: train two 2048-neuron nets ~300 epochs each, merge, prune to 3072, train ~200 more
- Converted to 16-bit integer weights for embedding

### Notable Differences from GoChess NNUE
- Only 2 king buckets (vs our 16)
- 5 output nodes selected by material (output buckets) -- we have 8
- Leaky ReLU (vs our CReLU/SCReLU)
- Net merging/pruning technique for training
- Hybrid mode: switches to hand-crafted eval if material advantage >= 4 pawns (v5.20)
- *Compare to GoChess*: We use v5 (768x16->N)x2->1x8 with 16 king buckets, CReLU/SCReLU, pairwise mul, Finny tables **(UPDATE 2026-03-21)**

---

## 9. Hash Table

- 4-slot buckets (probe 4 consecutive entries per index)
- Replacement: least useful entry based on depth and age
- Lockless via Bob Hyatt's XOR method
- Stores: score, best move, depth
- v7.00: Static eval stored in both main hash and per-thread eval hash
- v7.00: Stores/probes at depth < 0 (quiescence search entries)
- Hash moves fully verified before use (anti-corruption for SMP)
- Hash cutoffs enabled for mate scores regardless of depth (v2.60+)

---

## 10. Opening Book / Tablebases

### Opening Book
- Polyglot .bin format
- Weight formula: 4*nwins - 2*nlosses + ndraws
- Moves with <10 weight or <3 wins pruned
- OwnBook_Depth: default 24 half-moves
- OwnBook_Variety: default 25 (higher = more variety)
- Supports two books (OwnBook2 as fallback)

### Syzygy Tablebases
- Via Pyrrhic/Fathom library, up to 7 pieces
- DTZ ("rtbz") at root only
- WDL ("rtbw") during search when depth >= 0 and pieces <= 7
- SyzygyProbeDepth6: min probe depth for 6-piece TBs (default 1, recommend 3-4 for HDD)

---

## 11. Notable / Unusual Features

### Things Wasp Has That We Don't
1. **Hung-material-aware RFP margin**: MAX(hung_material/2, 75*depth) -- adapts to tactical danger
2. **Threat detection in RFP**: Checks for hung pieces and enemy pawn threats on 6th/7th before pruning (v6.00)
3. **Score-drop LMR bonus**: +1 ply reduction if eval dropped >= 150cp from 2 plies ago
4. **Selectivity UCI option**: Global knob to trade depth vs width
5. **QS hash entries**: Stores quiescence search results in TT at depth < 0 (v7.00)
6. **Multiple output nodes**: NNUE selects output based on material count (5 outputs in v7.00)
7. **Net merging/pruning**: Trains two larger nets, merges them, prunes to target size
8. **Hybrid eval fallback**: Uses hand-crafted eval when material advantage >= 4 pawns
9. **Last piece capture extension**: +1 ply when capturing final enemy piece
10. **NMP verification search**: At depth >= 5, does verification search at depth-4 after null-move fail-high
11. **Static eval in hash**: Stores static eval in main TT (saves re-evaluation)
12. **Countermove indexed by piece types**: [color][prev_piece][curr_piece][to_sq] -- richer than standard

### Things Wasp Doesn't Have That We Do
1. No singular extensions (explicitly tried, no benefit)
2. No history-based LMR adjustments
3. No continuation history (1-ply, 2-ply) -- only countermove with piece-type indexing
4. No capture history
5. No correction history
6. No IIR (uses traditional IID instead)
7. No recapture extensions
8. No aspiration window widening strategy details beyond initial 15cp

### Potential Ideas to Test in GoChess
1. **QS TT entries at depth < 0**: We already store TT move in QS; could store scores too
2. **NMP verification search**: depth-4 verification after null-move fail-high at depth >= 5
3. **Hung-material RFP**: Detect hung pieces and increase RFP margin accordingly
4. **Score-drop LMR**: If eval dropped >= 150cp from grandparent node, increase LMR
5. **Static eval in TT**: Avoid re-computing static eval on TT hits
6. **Last-piece-capture extension**: +1 ply when capturing the final non-pawn piece

---

## 12. Material Values (Classical, pre-NNUE)

For reference (v2.80 hand-crafted eval):
| Piece  | Opening | Endgame |
|--------|---------|---------|
| Pawn   | 80      | 102     |
| Knight | 335     | 325     |
| Bishop | 335     | 330     |
| Rook   | 470     | 550     |
| Queen  | 1000    | 1080    |
| Bishop pair | 35 | 35     |

---

## 13. Performance Reference

On Ryzen 7950x:
- Single-thread depth 17: ~160K NPS
- 32-thread depth 21: ~23M NPS (144x single-thread with 32 threads)
