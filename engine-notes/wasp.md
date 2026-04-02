# Wasp Chess Engine - Crib Notes

Author: John Stanback. Written in C. Closed source (MIT license, but source not publicly distributed).
Current version: 7.00 (June 2024). ~3200+ CCRL.

Primary sources: [Technical page](https://waspchess.com/wasp_technical.html),
[Release notes](https://waspchess.com/release_notes.txt),
[CPW](https://www.chessprogramming.org/Wasp).

Note: No source code available for direct verification. All details from documentation and release notes.

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
- Compare to Coda: we use R = 4 + d/3 + min((eval-beta)/200, 3) with R-1 after captures and NMP score dampening. Wasp uses simpler fixed R with verification.

### Reverse Futility Pruning (Static Null Move Pruning)
- **Conditions**: depth <= 3, eval >= beta + margin, not in check, no mate/promotion threat, not pawn ending
- **Margin**: MAX(hung_material/2, 75*depth)
- v6.00: detects hung pieces and enemy pawn threats on 6th/7th ranks
- **Notable**: The hung-material-aware margin is unusual.
- Compare to Coda: we use 70*d (improving) / 100*d (not improving), depth<=7. No hung-material awareness.

### Futility Pruning
- **Condition**: non-PV nodes, eval < alpha - (100 + 100*depth)
- When triggered: only generates captures/checks/promotions/pawn-to-7th (skip quiets)
- Compare to Coda: we use 90+lmrDepth*100. Wasp uses actual depth, not reduced depth.

### Razoring
- Low depth, static eval well below alpha -> quiescence search only
- Combined with futility (v3.0+)
- Compare to Coda: we don't currently have razoring.

### Late Move Pruning (LMP)
- v1.25: Prune non-tactical moves after 1 + 3*depth moves tried
- Additional condition: prune if moves >= 3 AND eval < alpha - 100*depth
- v6.50: Less aggressive if eval is improving
- Compare to Coda: we use 3+d^2 (with improving/failing adjustments), depth<=8.

### ProbCut (v3.0+)
- **Conditions**: depth >= 4, static eval >= beta
- Searches safe captures to depth-3
- Prunes if score >= beta + 100
- Compare to Coda: we use beta + 170 (currently disabled).

### SEE Pruning
- Uses "swapoff" (recursive alpha-beta on a single square)
- Losing captures deferred to last stage of move ordering
- No explicit SEE threshold pruning in search -- handled via move ordering stages

---

## 3. Move Ordering

### Stages (5-stage generation)
1. **Hash move** (or IID move if no hash move)
2. **Winning captures/promotions** (MVV-LVA, SEE-filtered)
3. **Decent quiet moves** (killer moves, countermoves, history-sorted)
4. **Losing captures** (SEE-negative deferred here)
5. **Poor quiet moves** (remaining low-history quiets)

### History Tables
- **Main history**: standard butterfly history
- **Countermove history** (v4.0+): indexed by [color][prev_piece_type][curr_piece_type][to_square]
  - Unusual: includes current piece type in the index, making it a form of continuation history
- **Killer moves**: standard 2-killer table
- **Refutation table**: mentioned separately from killers (likely countermove table)
- No explicit mention of: capture history, pawn history, correction history

### Internal Iterative Deepening (IID)
- When no hash move: R=3 for PV nodes, R=4 for non-PV nodes
- Compare to Coda: we use IIR (reduce depth by 1) instead of full IID.

---

## 4. Late Move Reduction (LMR)

### Formula Evolution
- v2.80: Reduction varies 0-3 plies based on depth and move count
- Tactical moves not reduced at PV, max 1 ply at non-PV
- PV nodes get ~1 ply less reduction than non-PV

### Adjustments
- **PV nodes**: ~1 ply less
- **Improving**: Less aggressive (v4.5+)
- **Unsafe moves**: +1 ply extra (v3.5+)
- **Score drop**: +1 ply if eval dropped >= 150cp from 2 plies earlier (v3.5+)
- Max reduction capped at 3 plies (at least in v2.80)

### Notable
- The "score drop from 2 plies ago" adjustment is uncommon
- No mention of history-based LMR adjustments
- Compare to Coda: we use ln(d)*ln(m) formula with separate quiet/capture tables (C=1.30/1.80), plus history and doDeeper/doShallower.

---

## 5. Extensions

- **Check extension**: +1 ply when in check (always, even if checking piece capturable, v4.5+)
- **Pawn to 7th**: +1 ply for safe pawn push to 7th rank (v5.00: also 6th rank if depth <= 2)
- **Last piece capture**: +1 ply when capturing the last remaining enemy piece
- **Extension limit**: Max 2 plies total, except at PV nodes or depth <= 4

### Notable Absences
- **No singular extensions** -- "Singular Extension does not seem beneficial for Wasp"
- No recapture extensions
- Compare to Coda: we have singular extensions (with multi-cut, negative extension) and recapture extensions. No check extensions.

---

## 6. Time Management

Limited details available:
- v3.0: "Simplified time management; increased fraction of remaining time usable when failing low/high or score drops"
- **Move overhead**: UCI option, default 50ms
- Ponder support

### Selectivity UCI Option
- Default 100; increase for deeper/narrower search, decrease for wider/shallower
- Adjusts pruning aggressiveness globally
- Unusual -- most engines don't expose a global selectivity knob

### Fail-Low / Instability
- Score drops and fail-lows cause increased time allocation (v3.0+)

---

## 7. Parallel Search (Lazy SMP)

- Shared hash table only -- each thread has own eval hash and pawn hash (v2.80+)
- Up to 160 threads supported
- Thread skipping: if >1 + nthreads/2 threads at depth N, skip to N+1 (v3.5)
- v6.50: if >15 threads at given depth, skip to depth+1; if >45, skip to depth+2
- Master thread updates PV/score; others contribute to hash table
- v7.00: Hash store/probe modified for depth < 0 (stores QS results)

### Notable
- Storing QS results (depth < 0) in hash table is interesting -- most engines don't do this
- Thread-skipping thresholds are detailed for high thread counts

---

## 8. NNUE / Evaluation

### v7.00 Architecture (HalfKA-like)
- **Input**: 768 piece-square features (12 pieces x 64 squares, no kings in features)
- **King buckets**: 2 zones per side (ranks 1-3 vs ranks 4-8)
- **Hidden layer**: 2x1536 neurons (white + black perspective), leaky ReLU
- **Output**: 5 nodes, selected by non-pawn material on board
- **Weight sets**: 2 sets from input->hidden (by king zone), 2 sets from hidden->output (by side to move)
- Square mirroring for king position symmetry
- Integer math for inference (~18% speedup over float, v6.00+)
- Incremental update via SIMD

### Compared to Coda NNUE
- Only 2 king buckets (vs Coda's 16)
- 5 output nodes selected by material (vs Coda's 8 output buckets)
- Leaky ReLU (vs Coda's CReLU/SCReLU)
- Net merging/pruning technique for training
- Hybrid mode: switches to hand-crafted eval if material advantage >= 4 pawns (v5.20)

---

## 9. Hash Table

- 4-slot buckets (probe 4 consecutive entries per index)
- Replacement: least useful entry based on depth and age
- Lockless via Bob Hyatt's XOR method
- v7.00: Static eval stored in both main hash and per-thread eval hash
- v7.00: Stores/probes at depth < 0 (QS entries)
- Hash moves fully verified before use
- Compare to Coda: similar philosophy (5-slot buckets, lockless XOR). Wasp also stores static eval in TT, which Coda does (raw/uncorrected).

---

## 10. Opening Book / Tablebases

### Opening Book
- Polyglot .bin format
- Weight formula: 4*nwins - 2*nlosses + ndraws
- Compare to Coda: we also use Polyglot .bin format.

### Syzygy Tablebases
- Via Pyrrhic/Fathom library, up to 7 pieces
- DTZ at root only, WDL during search when depth >= 0 and pieces <= 7

---

## 11. Summary: Differences from Coda

### Things Wasp has that Coda doesn't:
1. **Hung-material-aware RFP margin**: MAX(hung_material/2, 75*depth) -- adapts to tactical danger
2. **Threat detection in RFP**: Checks for hung pieces and enemy pawn threats before pruning (v6.00)
3. **Score-drop LMR bonus**: +1 ply reduction if eval dropped >= 150cp from 2 plies ago
4. **Selectivity UCI option**: Global knob to trade depth vs width
5. **QS hash entries**: Stores quiescence search results in TT at depth < 0 (v7.00)
6. **Hybrid eval fallback**: Uses hand-crafted eval when material advantage >= 4 pawns
7. **Last piece capture extension**: +1 ply when capturing final enemy piece
8. **NMP verification search**: At depth >= 5, does verification search at depth-4
9. **Static eval in hash**: Stores static eval in main TT
10. **Countermove indexed by piece types**: [color][prev_piece][curr_piece][to_sq]
11. **Check extension**: +1 ply when in check
12. **Pawn to 7th extension**: +1 ply for safe pawn push to 7th/6th rank

### Things Coda has that Wasp doesn't:
1. **Singular extensions** (Wasp explicitly tried, no benefit for them)
2. **History-based LMR adjustments**
3. **Continuation history** (1-ply) -- Wasp's countermove with piece-type indexing is different
4. **Capture history**
5. **Pawn history**
6. **Multi-source correction history**
7. **IIR** (Wasp uses traditional IID)
8. **Recapture extensions**
9. **Fail-high score blending**
10. **TT score dampening**
11. **TT near-miss cutoffs**
12. **NMP score dampening**
13. **Bad noisy pruning**
14. **Hindsight depth adjustment**
15. **doDeeper/doShallower** in LMR

### Parameter Comparison Table

| Feature | Wasp | Coda |
|---------|------|------|
| RFP depth | <=3 | <=7 |
| RFP margin | MAX(hung/2, 75*d) | 70*d (imp) / 100*d (not) |
| NMP reduction | R=3 (d<8), R=4 (d>=8) | R=4+d/3+min((eval-beta)/200,3) |
| NMP verification | depth>=5, at depth-4 | None (NMP score dampening instead) |
| Futility | 100+100*depth | 90+lmrDepth*100 |
| LMP | 1+3*depth | 3+d^2, depth<=8 |
| ProbCut | beta+100, depth>=4 | beta+170, depth>=5 (disabled) |
| LMR | 0-3 plies, capped | ln(d)*ln(m)/C, separate tables |
| Extensions | check, pawn-7th, last-piece | singular, recapture |
| TT | 4-slot, lockless | 5-slot, lockless |
| NNUE | 768->2x1536, 2 king zones, 5 outputs | HalfKA 16 king buckets, 8 outputs |

---

## 12. Ideas Worth Testing from Wasp

### High Priority
1. **NMP verification search**: depth-4 verification after null-move fail-high at depth >= 5. Alternative to Coda's NMP score dampening.
2. **Hung-material RFP**: Detect hung pieces and increase RFP margin accordingly.
3. **Static eval in TT**: Avoid re-computing static eval on TT hits. Coda already stores raw eval in TT but could leverage it more.

### Medium Priority
4. **Score-drop LMR**: If eval dropped >= 150cp from grandparent node, increase LMR.
5. **QS TT entries at depth < 0**: Store QS results in main TT.
6. **Last-piece-capture extension**: +1 ply when capturing the final non-pawn piece.
7. **Check extension**: +1 ply when in check. Wasp uses unconditional; Tucano's SEE-filtered variant may be better.

### Lower Priority
8. **Pawn to 7th extension**: Extend safe pawn pushes to 7th rank.
9. **Selectivity UCI option**: Global pruning aggressiveness knob (mostly a user-facing feature).
