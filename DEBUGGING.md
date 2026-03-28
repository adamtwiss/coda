# Coda vs GoChess: The 300 Elo Gap

## The Problem

Coda is a Rust port of GoChess (Go chess engine). Despite literal line-by-line translations of the search, movepicker, and all constants, Coda is **~300 Elo weaker** than GoChess in head-to-head play.

- **H2H result**: ~6W-155L-39D over 200 games at 10+0.1s TC
- **Elo gap**: -334 ± 55
- **Draw rate**: 19.5%
- **NPS**: Coda is ~9% faster (Rust speed advantage)
- **Depth**: Both reach depth 18 in 1 second on test positions

This gap is enormous — equivalent to being 4 plies shallower or 16x slower. With identical NNUE evaluation and a faster language, the gap must come from the search making worse decisions.

## What Has Been Verified Identical

- **NNUE eval**: 0/10,000 mismatches during search verification
- **Zobrist keys**: Exact same PRNG, same seed — hashes match on all positions
- **LMR tables**: Same formula (C=1.30 quiet, C=1.80 capture), values verified
- **SEE values**: Compared on multiple positions, all match exactly
- **Pruning thresholds**: All constants aligned (MATE_SCORE=29000, INFINITY=30000, CONTEMPT=10, all sentinel thresholds)
- **Perft**: All standard positions pass (move generation correct)
- **TT pack/unpack**: Roundtrip verified for all field types
- **Depths 1-3**: Produce identical eval sequences (172 evals, same MD5)

## What Was Fixed (Session of 2026-03-28)

### Major Fixes
1. **MATE_SCORE was 30000** (same as INFINITY) — couldn't distinguish mate from infinity. Fixed to 29000. This corrupted TT mate score storage.
2. **CONTEMPT sign bug** — returning +10 for draws (seek draws) instead of -10 (avoid draws)
3. **QSearch had no TT move and no capture history** — every QS node explored moves in suboptimal order. Fixed the depth-16 search instability completely.
4. **SEEAfterQuiet returned binary 0/-1000** instead of actual SEE value — caused 2.2x over-pruning of quiet moves per node
5. **Sentinel thresholds off by 1000cp** — used -INFINITY+100 (-29900) where GoChess uses -MateScore+100 (-28900) in 6 places

### Other Fixes
6. TT store argument order aligned with GoChess
7. TT Exact cutoff: removed erroneous history bonus update
8. Repetition detection: step by 2 (same-side only), matching GoChess
9. Continuation history: pre-compute prev_piece before move loop (was looking up post-make_move)
10. Time management: added absolute hard cap and next-iteration estimate
11. Aspiration: added mate score guard, removed delta>500 fallback
12. Literal rewrite of negamax/quiescence matching GoChess line-by-line
13. Literal rewrite of movepicker (stages, scoring, history types)
14. Zobrist keys matched exactly (same Go math/rand PRNG)
15. History table types: cont_hist changed to i16, capture/counter dimensions aligned

## The Smoking Gun: Different Search Trees

At the same depth on the same position, Coda searches a **structurally different tree**:

| Metric | GoChess | Coda | Ratio |
|--------|---------|------|-------|
| Total nodes (d14) | 142,463 | 175,188 | **1.23x** |
| QSearch nodes | 86,408 | 55,744 | **0.52x** |
| LMP prunes/node | 1.086 | 1.145 | 1.05x |
| Futility prunes/node | 0.632 | 0.634 | 1.00x |
| SEE prunes/node | 0.054 | 0.056 | 1.04x |
| LMR searches/node | 0.233 | 0.210 | 0.90x |
| FMR | 81% | 79% | ~same |

Key observations:
- **1.23x more main-search nodes** but **only 0.52x QSearch entries**
- Per-node pruning ratios are almost identical (~1.0x)
- This means the tree is just BIGGER, not differently pruned
- Coda stays in main search longer instead of dropping to QSearch

## Open Theories

### 1. Effective depth off by ~1 (most likely)
Something causes Coda's search to be effectively 1 ply deeper in parts of the tree. Moves that GoChess reduces to depth 0 (→ QSearch) stay at depth 1+ in Coda (→ main search). This would explain both the bigger main tree AND the smaller QSearch. Possible causes:
- LMR reductions are systematically 1 less in Coda (despite identical tables)
- Some adjustment to `new_depth` differs
- Alpha-reduce or extension behaves differently in edge cases
- The `reduction > 0` guard filters differently

### 2. Move ordering quality
FMR is 81% vs 79% — Coda's first move is the best 2% less often. This means more LMR re-searches (wasted work). With identical history tables types and scoring, this could come from:
- Different tie-breaking in the movepicker selection sort
- Move generation order affecting which move comes first at equal scores
- Pawn history or continuation history values accumulating differently (i16 vs i32 ranges)

### 3. Something in the PVS re-search logic
After LMR, both engines do doDeeper/doShallower adjustments and PVS re-search. A subtle difference in the re-search conditions could cause Coda to re-search more often at full depth.

### 4. QSearch structural difference
GoChess's QSearch uses the main MovePicker (with skipQuiet=true). Coda uses a separate QMovePicker. Despite adding TT move and capture history, the QMovePicker:
- Doesn't partition captures into good/bad upfront (does SEE per-move)
- Doesn't have the full staging (TTMove → GoodCaptures → BadCaptures)
- May order captures slightly differently

### 5. Board/make_move subtle difference
Something in board.make_move() or unmake_move() could leave state subtly different. Perft passes, but perft doesn't test correction history keys, NNUE accumulator state, or hash table interactions.

### 6. Time management still different
Despite fixes, Coda might allocate time differently in practice, causing it to stop searching at suboptimal moments.

## What to Try Next

1. **Add LMR reduction histograms** to both engines — directly compare the distribution of reductions applied. If Coda's average reduction is 0.1 less, that explains the tree size difference.

2. **Compare `new_depth` distributions** — what values of new_depth does each engine pass to recursive negamax calls? This would directly show if Coda searches deeper.

3. **Node-by-node trace at depth 5-6** — with matching Zobrist keys, trace every node visited and find exactly where the trees first diverge beyond the known 1-node difference at depth 4.

4. **Disable all LMR adjustments** in both engines — if the base LMR tables match but adjustments diverge, disabling adjustments should make the trees converge.

5. **Test the old Coda search** (pre-literal-rewrite) — it was a cleaner Rust implementation. If it's stronger than the literal translation, the translation introduced a bug.

## Key Test Positions

- **Startpos**: Trees match at d1-3, diverge at d4 (1 extra eval in GoChess)
- **Critical endgame** (`6k1/1p3pp1/p7/2Pp1p2/PPn1rP1P/2P5/2KB3P/3R4 w`): QSearch TT fix resolved the depth-16 collapse. Scores now match through depth 18.
- **After 1.Nf3 f5** (`rnbqkbnr/ppppp1pp/8/5p2/8/5N2/PPPPPPPP/RNBQKB1R w KQkq f6`): The 1-node divergence at depth 4 occurs in QSearch from this position.

## Key Binaries
- `/tmp/gochess-see` — GoChess with eval/see/stats
- `/tmp/coda-seeq` — Latest Coda with all fixes
- `/tmp/coda-zobrist` — Coda with matching Zobrist (before SEEAfterQuiet fix)

## Commits (this session)
See `git log --oneline` in the coda repo — 15 commits from this session.

## Update: Correction History Amplifies Divergence (late 2026-03-28)

### LMR Adjustment Analysis
Average LMR reduction: GoChess 2.493, Coda 2.414 (0.08 less → bigger tree).

Key LMR adjustment rates (at depth 14):
| Adjustment | GoChess | Coda | Effect |
|-----------|---------|------|--------|
| PV (-1) | 16.5% | 14.2% | Similar |
| cut (+1) | 85.4% | 87.0% | Similar |
| improving (-1) | 40.5% | 42.0% | Coda reduces less 1.5% more |
| failing (+1) | 0.4% | 0.8% | Coda reduces more 2x |
| **unstable (-1)** | **1.9%** | **3.3%** | **Coda reduces less 1.7x** |
| hist_good (-N) | 0.08% | 0.0% | Negligible |

### Correction History Ablation
Without correction history, the unstable difference nearly disappears:
- WITH correction: GoChess unstable=616, Coda=1212 (1.97x)  
- WITHOUT correction: GoChess unstable=602, Coda=665 (1.10x)

Correction history is amplifying a small underlying divergence. The corrected evals differ between engines (despite same raw NNUE eval), causing the unstable threshold (|eval swing| > 200) to fire 2x more in Coda. This extra unstable → less reduction → bigger tree → more nodes → worse play.

### Root Cause Theory
The search trees diverge slightly at depth 4+ (1 node difference). This causes different positions to be searched, which updates correction history differently. The different correction values then modify the eval used for pruning/reducing, amplifying the original divergence. This feedback loop is the 300 Elo gap.

### Possible Fix
1. Disable correction history in both engines to remove the amplifier (test Elo gap)
2. Find and fix the original depth-4 divergence that seeds the correction difference
3. The depth-4 divergence is likely in move ordering at tie-break (different move generation order)
