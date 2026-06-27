# Caissa Engine - Technical Review

Source: `~/chess/engines/Caissa/src/backend/`
Version: 1.24.21 (git HEAD `d77117c`, 2026-03-12 — source UNCHANGED since this note's last body refresh)
Author: Michal Czardybon

> **Caissa is rank #13 in our local RR — NOT stronger than Coda (#7).** Its
> choices are HYPOTHESES, not authority. The body below was written against the
> old "GoChess" predecessor and many "Coda comparison" lines are now stale (Coda
> has since shipped most of the headline ideas). See **Review refresh 2026-06-27**
> at the bottom for what Coda already does and what was H0'd. Use the ranked
> experiment list immediately below — everything in it has been prior-art checked.

All parameter values are SPRT-tuned via `DEFINE_PARAM` macros. Values listed are the tuned defaults.

---

## Testable Experiments for Coda (ranked, refreshed 2026-06-27)

Only ideas that (a) Coda does NOT already do and (b) are not already H0'd in
`experiments.md` survive here. Everything else is consolidated under "Already in
Coda / already H0'd" in the refresh section.

### 1. Node-count move-ordering bonus at shallow plies (HEADLINE — genuinely untested)
- **Caissa mechanism**: a persistent per-position `NodeCache` keyed by hash, populated
  for `ply < 3` (`Search.cpp:1667-1671`). After each move it records nodes spent
  (`Search.cpp:2000-2001 AddMoveStats`). On the NEXT iteration the move orderer adds
  `score += 4096 * moveInfo.nodesSearched / nodesSum` when `nodesSum > 256`
  (`MoveOrderer.cpp:417-421`). Moves that consumed the most subtree last iteration get
  ordered earlier — a cross-iteration "this branch was hard/important" signal.
- **How Coda differs today**: Coda has `root_move_nodes: Box<[u64;4096]>` but it is
  **root-only (ply 0), reset every iteration** (`search.rs:825, 2141`) and feeds ONLY
  the TM `subtree_size_multiplier` (`search.rs:2739-2777`). It is never used for move
  ordering, and never at ply 1-2. `movepicker.rs` has no node-count term.
- **Prior-art check**: no hit in `experiments.md` for node-count/node-cache *move
  ordering* (only node-fraction-for-TM at line 3257). Genuinely untested direction.
- **Sketch**: extend `root_move_nodes` accounting to ply<3 keyed by (ply, from*64+to)
  or a small hash-keyed cache; do NOT zero it between iterations; in the quiet/capture
  scoring loop add `BONUS * nodes_this_move / nodes_sum` (BONUS ~ one history unit) when
  `nodes_sum > threshold`. Gate to ply<3 and `info.root_depth >= 4` so iteration 1 has data.
- **Magnitude / risk / transfer**: moderate upside (used by SF/Caissa/Obsidian-class);
  medium effort (cross-iteration accounting + new ordering term); risk it overlaps with
  Coda's already-rich 4D threat history. Worth a single clean SPRT `[0,3]` STC-first.

### 2. IIR also fires on STALE TT entries (trivial, untested)
- **Caissa mechanism** (`Search.cpp:1487-1490`): IIR reduces depth by 1 when
  `depth >= IIRStartDepth(3) && (cutNode||pv) && (!ttMove.valid || ttEntry.depth + 4 < depth)`.
  The second clause fires IIR even when a TT move exists, if that entry is too shallow to trust.
- **How Coda differs today** (`search.rs:3747`): Coda's IIR fires ONLY on
  `tt_move == NO_MOVE`. A present-but-shallow TT move never triggers IIR.
- **Prior-art check**: no `experiments.md` hit for "IIR stale TT" / "IIR tt depth". Untested.
- **Sketch**: change the gate to
  `tt_move == NO_MOVE || (tt_hit && tt_entry.depth + SLACK < depth)` with SLACK as a new
  tunable defaulting ~4. One-line behavioral change behind `FEAT_IIR`.
- **Magnitude / risk / transfer**: low magnitude; low effort; low risk. Caveat: Coda's
  IIR ordering vs NMP was recently audited (search.rs:3745 note, NMP audit N2) — keep the
  same call-site, only widen the condition. Good cheap `[0,3]` probe.

### 3. Threat-gate the RFP "improving" discount (low effort, untested angle)
- **Caissa mechanism** (`Search.cpp:1508-1514`): the improving term is
  `- RfpImprovingScale * (isImproving && !OppCanWinMaterial(position, threats))` — i.e. the
  improving discount is *suppressed* when the opponent has a material-winning threat
  (rook-on-queen, minor-on-Q/R, pawn-on-Q/R/B/N from the threat bitboards).
- **How Coda differs today** (`search.rs:3582`): Coda picks `RFP_MARGIN_IMP` vs
  `RFP_MARGIN_NOIMP` on the raw `improving` flag with no threat qualification.
- **Prior-art check**: line 642 ("improving doesn't help per-move pruning") is about
  futility/SEE, NOT a threat-gated RFP improving term; line 78 4D-threat-history is
  unrelated. The specific "suppress improving discount under enemy material threat" is
  not in `experiments.md`. Coda already computes full threat bitboards (`threats.rs`), so
  the gate is cheap.
- **Sketch**: when `improving`, additionally require "no opponent material-winning threat"
  before using `RFP_MARGIN_IMP`; else fall back to `RFP_MARGIN_NOIMP`. Reuse existing
  threat masks.
- **Magnitude / risk / transfer**: low magnitude; low effort; low risk. Plausibly redundant
  with NNUE+threat eval already encoding the threat — but cheap to falsify. `[0,3]`.

### Considered and DROPPED (do not propose)
- **History folded into quiet SEE-pruning threshold** — H0: `#2072 quiet-see-hist`
  (-0.3, 58k games) and line 3496 "SEE history gate" (raw -21). Caissa does this
  (`Search.cpp:1788`) but Coda has tested it both ways.
- **Capture-history into SEE threshold** — H0 (line 374, early-noise fade).
- **In-check ProbCut** — the old note claimed Caissa has it; current source does NOT
  (probcut at `Search.cpp:1582` sits inside the `!isInCheck` block). Non-idea.
- **NMP only on cut nodes** — Coda ALREADY does this (`search.rs:3656`).
- **ProbCut dynamic SEE threshold `probBeta - staticEval`** — Coda ALREADY does this
  (`search.rs:3792`).
- **4D threat-aware history / recapture extension / non-pawn + continuation corrhist** —
  all ALREADY in Coda (CLAUDE.md history+corrhist sections; recapture H1 #758/#1817).

---

## 1. NNUE Architecture

### Network Topology
- **Architecture**: HalfKA -> CReLU -> output (direct, no hidden layers)
- **Feature set**: `32 king buckets * 12 piece types * 64 squares = 24,576` input features
- **Accumulator size**: 1024 (int16)
- **Output**: 1 value per variant
- **8 output variants** selected by piece count: `min(numPiecesExcludingKing / 4, 7)`
  - Each variant has independent output weights (2*1024 int16) and bias (1 int32)
  - This is a **piece-count output bucketing** scheme -- rare among top engines

```
PackedNeuralNetwork layout:
  Header (64 bytes)
  accumulatorWeights[24576 * 1024] (int16)  -- ~48MB
  accumulatorBiases[1024] (int16)
  lastLayerVariants[8] { weights[2*1024] (int16), bias (int32) }
```

**Coda comparison**: GoChess uses HalfKA(32kb) -> 1024 CReLU -> 1 output. No output bucketing. Caissa's 8-variant output is interesting -- it means the net has piece-count-specific output weights without needing hidden layers. Could improve endgame accuracy without the NPS cost of hidden layers. Worth investigating for GoChess v5 before committing to v7 hidden layers.

### Quantization
- **First layer**: int16 weights/biases, QA = 256 (ActivationRangeScaling)
- **Output layer**: int16 weights, int32 bias
  - WeightScale = 256 (1 << 8)
  - OutputScale = 1024 (1 << 10)
  - Output weight quant scale = `WeightScale * OutputScale / ActivationRangeScaling = 1024`
  - Output bias quant scale = `WeightScale * OutputScale = 262144`

**Coda comparison**: GoChess uses QA=255, QB=64. Caissa uses powers of 2 (256, 256, 1024) enabling bit shifts instead of division. The output layer uses int16 weights (not int8), giving more precision for the direct-output architecture.

### Activation
- **CReLU**: `clamp(x, 0, 256)` -- standard clipped ReLU
- No SCReLU, no pairwise multiplication
- Applied at inference time in the output linear layer (not stored in accumulator)

### King Buckets (32)
```
 0  1  2  3   3  2  1  0
 4  5  6  7   7  6  5  4
 8  9 10 11  11 10  9  8
12 13 14 15  15 14 13 12
16 17 18 19  19 18 17 16
20 21 22 23  23 22 21 20
24 25 26 27  27 26 25 24
28 29 30 31  31 30 29 28
```
- Horizontally mirrored (file >= 4 flips)
- Each rank has 4 unique buckets (4 files * 8 ranks = 32)

**Coda comparison**: GoChess also uses 32 king buckets with the same layout.

### Accumulator Cache (Finny Table)
- `AccumulatorCache::KingBucket` stores: accumulator + piece bitboards `[2 colors][6 piece types]`
- Cache dimensions: `[2 perspective][2 * 32 king side+bucket]` = 128 entries
- On king bucket change: diffs cached vs current piece bitboards, applies only changed features
- On init: copies biases into all cache entries, zeros piece bitboards
- **Two-stage parent update**: if parent node's accumulator is dirty, updates parent first so siblings can reuse it

**Coda comparison**: GoChess has an identical Finny table design. The two-stage parent update is a nice touch -- ensures that sibling nodes benefit from a single parent refresh rather than each independently walking up the tree.

### Eval Post-Processing (`Evaluate.cpp`)
```cpp
value = nnOutput / (OutputScale * WeightScale / c_nnOutputToCentiPawns);
value = value * (52 + gamePhase) / 64;  // phase scaling (0=endgame, 24=opening)
// castling rights bonus: +5 per remaining right when king has moved off e1/e8
// eval saturation: past +/-8000, compressed by /8
```
- **Game phase scaling**: `gamePhase = minors + 2*rooks + 4*queens`, scales eval from 81% (endgame) to 119% (opening)
- **Endgame evaluation**: checked first for <= 6 pieces (before NNUE eval)

### SIMD Support
- **AVX-512**, AVX2, SSE2, ARM NEON all supported via preprocessor macros
- **VNNI support**: uses `_mm256_dpwssd_epi32` when available (fused dot-product)
- Accumulator operations use tiled register blocking (OptimalRegisterCount tiles)
- Output layer: 2x unrolled sum for STM and NSTM perspectives
- Extra target accumulator: `UpdateImpl<WithExtraTarget>` writes both current and cache in single pass

**Coda comparison**: GoChess has AVX2 and NEON. Caissa additionally supports AVX-512 and VNNI. The VNNI path (`dpwssd`) is relevant for newer Intel CPUs -- could be a meaningful speedup for Zen4/Sapphire Rapids.

---

## 2. Search: Pruning Techniques

### Reverse Futility Pruning (RFP)
- **Depth guard**: `depth <= 6` (RfpDepth=6)
- **Margin**: `83*depth + 0*depth^2 - 145*improving_and_no_opp_material_threat`
  - "improving" only counts if opponent cannot win material (`OppCanWinMaterial` checks threat bitboards: rook attacks on queens, minor attacks on Q/R, pawn attacks on Q/R/B/N)
- **Floor**: margin clamped to at least RfpTreshold=16
- **Return value**: blended `(eval * (1024-525) + beta * 525) / 1024` (not just eval or beta)
- **Conditions**: non-PV, not in check, no filtered move, eval <= KnownWinValue

**Coda comparison**: GoChess uses `60 + 60*depth`. Caissa's `83*depth` is steeper but capped at depth 6 (GoChess: depth 7). The improving+threat interaction is more sophisticated. The blended return value is novel -- GoChess returns raw eval.

### Razoring
- **Depth guard**: `depth <= 4` (RazoringStartDepth=4)
- **Margin**: `22 + 158*depth`
- **Mechanism**: if `eval + margin < beta`, do qsearch; return qscore if still < beta

**Coda comparison**: GoChess uses `400 + 100*depth`. Caissa's margins are tighter (at depth 4: 654 vs 800).

### Null Move Pruning (NMP)
- **Depth guard**: `depth >= 3`
- **Eval condition**: `eval >= beta + (depth < 4 ? 16 : 0)`, `staticEval >= beta`
- **Only on cut nodes** (`node->isCutNode`) -- this is unusual
- **No consecutive null moves**: checks parent and grandparent
- **Reduction R**: `3 + depth/3 + min(3, (eval-beta)/85) + improving`
- **Verification re-search**: if score >= beta AND (`abs(beta) < KnownWinValue && depth < 10`): return immediately. Otherwise reduce depth by 5 and continue.

**Coda comparison**: GoChess applies NMP on any non-PV node, not just cut nodes. Restricting to cut nodes is more conservative -- may miss some pruning opportunities but avoids bad null-move results in all-nodes.

### ProbCut
- **Depth guard**: `depth >= 5`, non-PV, not in check
- **Beta**: `beta + 133`
- **SEE threshold**: `probBeta - staticEval` (dynamic)
- **Process**: qsearch first, then `depth-4` NegaMax verification
- **In-Check ProbCut** (from Stockfish): non-PV, in check, TT move is capture, TT has lower bound with `depth >= depth-4` and `ttScore >= beta+329`. Returns `probCutBeta` directly (no search).

**Coda comparison**: GoChess has ProbCut but not in-check ProbCut. The dynamic SEE threshold (`probBeta - staticEval`) is clever -- it adapts to position evaluation.

### Futility Pruning
- **Depth guard**: `depth < 9`
- **Margin**: `staticEval + 32*depth^2 + moveStatScore/383`
- Note: quadratic depth scaling, plus history influence
- On trigger: `movePicker.SkipQuiets()`, but first quiet still tried

**Coda comparison**: GoChess uses `60 + 60*depth` (linear). Caissa's quadratic scaling is more aggressive at higher depths.

### Late Move Pruning (LMP)
- **Threshold**: `4 + depth^2` when improving, `4 + depth^2/2` otherwise
- Adds `LateMovePruningPVScale * isPvNode` (=2) to depth for PV nodes
- **Optimization**: if in quiets stage, breaks out entirely

**Coda comparison**: GoChess uses `3 + d^2`. Very similar.

### History Pruning
- **Depth guard**: `depth < 9`
- **Threshold**: `0 - 234*depth - 148*depth^2` (quadratic)
- Uses moveStatScore (main history + conthist[0,1,3])

**Coda comparison**: GoChess uses `-5000*depth` (linear). Caissa's quadratic formula is more aggressive at higher depths.

### SEE Pruning
- **Only when move target is on a threatened square** (`move.ToSquare() & allThreats`)
- **Captures**: `depth <= 5`, threshold = `-120*depth`
- **Non-captures**: `depth <= 9`, threshold = `-49*depth - moveStatScore/134`
  - History integrated into SEE threshold for quiets

**Coda comparison**: GoChess uses `-20*depth^2` for quiets, `-depth*100` for captures. Caissa's history-in-SEE-threshold is a good idea we should consider. The "only when threatened" guard is also interesting -- saves SEE computation on unthreatened squares.

### Internal Iterative Reduction (IIR)
- `depth >= 3`, cut node OR PV node
- Fires if no TT move OR `ttEntry.depth + 4 < depth` (stale TT entries also trigger)
- Reduces depth by 1

**Coda comparison**: GoChess IIR fires on missing TT move. Caissa also fires on stale TT entries (depth + 4 < current depth) -- this is a good extension of the idea.

---

## 3. Extensions

### Singular Extensions
- **Conditions**: non-root, `depth >= 3`, TT move with `|ttScore| < KnownWinValue`, TT has lower bound, `ttEntry.depth >= depth - 3`
- **Singular beta**: `ttScore - depth`
- **Singular depth**: `max(1, (59*depth - 215) / 128)`
- **Results**:
  - Singular (score < singularBeta, ply < 2*rootDepth): +1 extension
  - Double extension: `score < singularBeta - 14 - 256*isPvNode`
  - Triple extension: `score < singularBeta - 51 - 256*isPvNode`
  - **Multi-cut**: returns `(singularScore * singularDepth + beta) / (singularDepth + 1)` -- blended return
  - **Negative extensions**: -2 or -3 depending on conditions (ttScore >= beta, isCutNode, ttScore <= alpha)

### Recapture Extension
- PV nodes only, TT move is a recapture (same target square): +1

**Coda comparison**: GoChess has singular extensions but found them harmful cross-engine (see CLAUDE.md). Caissa uses them extensively. The triple extension and blended multi-cut return are more sophisticated than typical implementations. The negative extensions (-2/-3) when not singular are aggressive depth reductions that act like pruning.

---

## 4. Move Ordering

### Stages (MovePicker)
1. **TT Move** (score: INT32_MAX - 1)
2. **Generate & pick captures** (winning/good + queen promotions)
3. **Killer move** (1 per ply, score: 1,000,000)
4. **Counter move** (score: 999,999)
5. **Generate & pick quiets** (sorted by combined score)
6. Bad captures (from step 2, ordered last)

Note: only 1 killer move per ply (not 2). Counter move is a separate stage.

### History Tables

#### Quiet Move History
- `quietMoveHistory[2 stm][2 from_threatened][2 to_threatened][4096 from_to]`
- **Threat-aware**: separate counters for whether source/dest squares are attacked by opponent
- This gives 4x the granularity of standard butterfly history

**Coda comparison**: GoChess uses standard `[2 stm][4096 from_to]`. The threat-aware indexing is a proven Elo gain in many engines. Priority adoption candidate.

#### Continuation History
- 6 plies back: `continuationHistory[2 prevIsCapture][2 prevColor][2 currentColor][6 piece][64 to]`
- **Scoring weights**: conthist[0]=1.0, [1]=1019/1024, [3]=555/1024, [5]=582/1024
- **Update weights**: conthist[0]=1.0, [1]=1014/1024, [2]=300/1024, [3]=978/1024, [5]=978/1024
- Note: conthist[2] updated but NOT scored, conthist[4] neither updated nor scored

**Coda comparison**: GoChess uses 3 plies of continuation history. Caissa uses 6 with weighted scoring. The capture-aware indexing (`prevIsCapture`) is also unusual -- most engines don't split by capture vs quiet predecessor.

#### Capture History
- `capturesHistory[2 stm][6 piece][5 captured_piece][64 to]`
- Standard piece-captured-to indexing

#### Counter Moves
- `counterMoves[2 stm][6 piece][64 to]`

### Quiet Scoring Details
- Main history + weighted conthist (4 tables)
- **Threat-based piece bonuses**:
  - Knight/Bishop: +4000 from pawn-attacked, -4000 to pawn-attacked
  - Rook: +8000 from minor-attacked, -8000 to minor-attacked
  - Queen: +12000 from rook-attacked, -12000 to rook-attacked
- **Node cache bonus** (ply < 3): `4096 * moveNodesSearched / totalNodes`

### Capture Scoring
- `attacker < captured`: WinningCaptureValue (20M)
- `attacker == captured`: GoodCaptureValue (10M)
- SEE >= 0: GoodCaptureValue (10M)
- SEE < 0: INT16_MIN
- Plus `4096 * captured_piece_type` (MVV)
- Plus capture history (shifted by -INT16_MIN)

### History Update Formula
- Gravity-style: `counter += delta - counter * |delta| / 16384`
- **Quiet bonus**: `min(-113 + 164*depth + 148*scoreDiff/64, 2178)`, scoreDiff = `min(bestValue - beta, 256)`
- **Quiet malus**: `-min(-51 + 160*depth + 155*scoreDiff/64, 1844)`
- **History bonus includes scoreDiff**: moves causing larger beta cutoffs get larger updates
- **Non-zero initialization**: quiets=802, conthist=762, captures=346 (biases unexplored moves positive)
- **New search**: history divided by 2, killers cleared

### Prior Counter-Move History Update
- When `bestValue <= oldAlpha` and previous move was quiet: bonus of `min(1200, depth*120 - 100)` to the predecessor's continuation history
- Rationale: if all responses to the previous move are bad, the previous move was probably good

---

## 5. LMR (Late Move Reductions)

### Base Table
- **Separate tables for quiets and captures**
- Quiets: `64 * (0.56 + 0.43 * ln(d) * ln(m))`
- Captures: `64 * (0.68 + 0.42 * ln(d) * ln(m))`
- 64x64 table, units of 1/64

### Quiet Adjustments (in 1/64 units)
| Adjustment | Value | Direction |
|---|---|---|
| Non-PV node | +15 | more reduction |
| TT move is capture | +73 | more reduction |
| Move is killer/counter | -168 | less reduction |
| Cut node | +183 | more reduction |
| Not improving | +38 | more reduction |
| Move gives check | -71 | less reduction |
| History-based | `-(stat + 6877) / 240` | variable |

### Capture Adjustments (in 1/64 units)
| Adjustment | Value | Direction |
|---|---|---|
| Winning capture | -63 | less reduction |
| Bad capture | r += -12 | less reduction |
| Cut node | +81 | more reduction |
| Not improving | r -= (-18) = +18 | more reduction |
| Move gives check | r -= (-4) = +4 | slightly more |

### PV-Specific
- `r -= 64 * depth / (1 + ply + depth)` -- less reduction at low ply
- TT entry with high depth: `r -= 13`

### LMR Deeper/Shallower
- If reduced search beats alpha: `newDepth += (score > bestValue + 85) && (ply < 2*rootDepth)`
- `newDepth -= (score < bestValue + newDepth)`

**Coda comparison**: GoChess has separate quiet/capture LMR (matching Caissa). The capture LMR values are interesting -- bad captures get LESS reduction, which is counterintuitive but may help avoid missing tactical shots.

---

## 6. Quiescence Search

- **Stand-pat beta blending**: returns `(bestValue * 0.49 + beta * 0.51)` instead of just bestValue
- **End-of-qsearch blending**: `(bestValue * 0.47 + beta * 0.53)`
- **Futility base**: `standPat + 77`
- **Move count pruning**: depth < -4: 1 move, < -2: 2 moves, < 0: 3 moves
- **In check**: generates evasions, tries only 1 if it doesn't improve
- Bad captures break immediately (not just skip)
- CanReachGameCycle check: if alpha < 0, raises alpha to 0 (cycle detection)

**Coda comparison**: GoChess has beta blending in qsearch. The move-count pruning by qsearch depth is interesting.

---

## 7. Eval Correction History

### Four Tables
1. **Pawn structure**: `[2 stm][16K]` keyed by pawn hash
2. **Non-pawn white**: `[2 stm][16K]` keyed by white non-pawn hash
3. **Non-pawn black**: `[2 stm][16K]` keyed by black non-pawn hash
4. **Continuation correction**: `[2 stm][384 piece-to][384 piece-to]` for ply-2 and ply-4

### Application
- `corr = 53*pawn + 65*nonPawnW + 65*nonPawnB + 76*cont_ply2 + 76*cont_ply4`
- Divided by 512, applied on top of NNUE eval, then 50-move scaling

### Update
- Bonus: `clamp((bestValue - unadjustedEval) * depth / 4, -249, 249)`
- Gravity: `h += value - h * |value| / 1024`
- Condition: not in check, best move is quiet or loses SEE, score diverged from eval

**Coda comparison**: GoChess has pawn correction history. Caissa additionally has non-pawn white/black (split by color) and continuation correction. The continuation correction is novel -- it adjusts eval based on the move sequence, not just the position hash.

---

## 8. Novel / Notable Features

### 1. Output Bucketing by Piece Count (8 variants)
- `variant = min(numPiecesExcludingKing / 4, 7)`
- Each variant has independent output weights (2*1024 weights + 1 bias)
- Allows the network to learn piece-count-specific evaluation scales
- Very cheap at inference time (just selects which weight vector to dot-product)

**Coda comparison**: This is the most interesting architectural feature. It gives endgame/middlegame specificity without hidden layers. GoChess could implement this for v5 nets with minimal NPS cost. The 8 variants add only 8 * (2048 + 4) = ~16KB of weights.

### 2. Threat-Aware Quiet History
- History indexed by `[from_is_threatened][to_is_threatened]`
- 4x more entries, better move ordering for tactical contexts

### 3. Node Cache (near-root move ordering)
- For ply < 3, tracks nodes spent on each move across iterations
- Boosts moves proportional to `nodesSearched / totalNodes`
- Also used in time management (best move node fraction)

### 4. Fail-High Score Blending
- On beta cutoffs: `bestValue = (bestValue * depth + beta) / (depth + 1)`
- Prevents inflated fail-high scores from propagating

### 5. Depth Reduction After Alpha Improvement
- `if (node->depth > 2) node->depth--` when alpha is raised
- Remaining moves searched at reduced depth after finding a good move

### 6. QSearch Beta Blending
- Stand-pat and end-of-qsearch returns are blended towards beta (~50/50)
- Dampens extreme qsearch values

### 7. Prior Counter-Move History Bonus
- Fail-low nodes give bonus to predecessor's continuation history
- "If all my responses are bad, the move that led here was good"

### 8. Accumulator Two-Stage Parent Update
- When current and parent accumulators are both dirty but share the same king bucket:
  first updates parent, then updates current from parent
- Siblings then get parent's cached accumulator for free

### 9. CanReachGameCycle (Cuckoo Hashing)
- In non-PV nodes with alpha < 0: checks if a drawing move exists via cuckoo tables
- Raises alpha to 0 if cycle is reachable

### 10. NUMA-Aware Threading
- Threads pinned to NUMA nodes
- Correction histories allocated per NUMA node
- ThreadData allocated on specific NUMA nodes

**Coda comparison**: GoChess has no NUMA awareness. This matters for multi-socket systems.

### 11. IIR on Stale TT Entries
- Fires not just on missing TT move, but when `ttEntry.depth + 4 < currentDepth`
- Reduces depth when TT entry exists but is too shallow to be useful

### 12. SEE Pruning Only on Threatened Squares
- `if (move.ToSquare() & allThreats)` guards SEE pruning
- Saves SEE computation on unthreatened target squares

### 13. History Bonus Scaled by Score Difference
- `scoreDiff = min(bestValue - beta, 256)` scaled into bonus formula
- Bigger beta cutoffs produce proportionally larger history updates

---

## 9. Aspiration Windows

- Initial: `6 + |prevScore| / 17`
- Fail-low: `beta = (alpha+beta+1)/2`, `alpha -= window`, restore depth
- Fail-high: `beta += window`, reduce depth by 1 (if depth > 1 and depth+5 > iterationDepth)
- Growth: `window += window / 3`
- Fallback: full window when `window > 547`

**Coda comparison**: GoChess uses delta=15 with doubling. Caissa starts smaller (6) with 33% growth. The asymmetric fail-low (beta contracts toward alpha) is interesting.

---

## 10. Lazy SMP

- Shared: only TT (lockless)
- Per-thread: MoveOrderer, NodeCache, AccumulatorCache, CorrectionHistories, search stack
- NUMA-aware thread pinning and memory allocation
- Best thread selection: picks thread with highest depth+score (prefers deeper, prefers mate)

---

## 11. Time Management

### Moves-Left Estimation (from Lc0)
- `f(moves) = 35 * (1 + 1.5 * (moves/35)^2.19)^(1/2.19) - moves`

### Ideal/Max Time
- `idealTime = 0.823 * (remaining / movesLeft + increment)`
- `maxTime = 4.50 * ((remaining - overhead) / movesLeft + increment)`
- Both clamped to `[0, 0.8 * remaining]`

### Dynamic Adjustments
- **Predicted move**: hit saves 8.5%, miss spends 13.2% more
- **PV stability**: factor from ~1.55 (unstable) to ~0.97 (stable 10+ iterations)
- **Node fraction**: `(1 - bestMoveNodeFraction) * 2.08 + 0.63`
- **Root singularity**: after 20% of ideal time, searches at depth/2 to see if best move is clearly singular

---

## 12. Priority Adoption Candidates for Coda/GoChess

| Feature | Difficulty | Expected Impact | Notes |
|---|---|---|---|
| Output bucketing (8 variants) | Medium (net retraining) | High | Free Elo at near-zero NPS cost |
| Threat-aware quiet history | Low (indexing change) | Medium | 4x table, proven in many engines |
| 6-ply continuation history | Low (extend existing) | Low-Medium | Weighted scoring/update |
| Non-pawn correction history | Low (add tables) | Medium | Split white/black non-pawn hash |
| Continuation correction history | Medium (new table type) | Low-Medium | Novel, worth testing |
| IIR on stale TT entries | Trivial | Low | One extra condition |
| SEE pruning only on threatened | Trivial | Low (NPS) | Saves SEE calls |
| History bonus with scoreDiff | Low | Low | One formula change |
| VNNI SIMD path | Medium | Low (hardware-dependent) | Only helps on Zen4+ / SPR |
| NMP only on cut nodes | Trivial | Unknown | Could help or hurt, test both |
| Prior conthist bonus on fail-low | Low | Low | Rewards good predecessor moves |

---

## Review refresh 2026-06-27

**Source state:** Caissa HEAD `d77117c` (2026-03-12) — the C++ source is byte-identical
to when the body above was written. The only thing that has moved is **Coda**, so this
refresh re-frames the body against Coda's *current* feature set and supersedes the
"Coda comparison" / "Priority Adoption Candidates" tables above where they conflict.

### Verified against current Caissa source (file:line)
- RFP: `Search.cpp:1507-1516` — confirmed `RfpDepthScaleLinear*d + RfpDepthScaleQuad*d² -
  RfpImprovingScale*(improving && !OppCanWinMaterial)`, blended return
  `(eval*(1024-RfpAdjBetaScale)+beta*RfpAdjBetaScale)/1024`.
- Razoring `Search.cpp:1521-1528`; NMP cut-node-only `1531-1535`; ProbCut dynamic SEE
  `1583-1606`; IIR incl. stale-TT clause `1487-1490`; quiet SEE pruning with history
  `1787-1788`; LMP/history/futility `1744-1772`; node cache ply<3 `1667-1671` + ordering
  bonus `MoveOrderer.cpp:417-421`; qsearch capture-count by depth `Search.cpp:1204-1205`.

### Already in Coda since this note was written (REMOVE from "adoption candidates")
The body's "Priority Adoption Candidates" table is now largely **done** — do not re-propose:
- **Threat-aware quiet history** → Coda ships the richer 4D
  `[from_threatened][to_threatened][from][to]` main history (CLAUDE.md; `e7f52b5`).
- **6-ply / weighted continuation history** → Coda has continuation history (plies 1,2,4,6).
- **Non-pawn white/black + continuation correction** → Coda's multi-source corrhist
  (pawn, NP-white, NP-black, minor, major, continuation) is in production (CLAUDE.md).
- **NMP only on cut nodes** → `search.rs:3656` (Reckless gate, already merged).
- **ProbCut dynamic SEE threshold** → `search.rs:3792`.
- **Fail-high score blending** → Coda has it (FH_BLEND, CLAUDE.md TT section).
- **Recapture extension** → H1 and gated (`#758`, `#1817`); ablation -18 Elo.
- **Singular extensions** → Coda found the *positive* extension worth ~-30 Elo
  (`experiments.md` SE v6/v7); keeps multi-cut + negative ext only. Caissa's
  triple-extension/blended-multicut is therefore NOT a transfer candidate for Coda.

### Already H0'd in Coda (do not retest as "Caissa idea")
- Quiet SEE threshold + history (`#2072`, line 3496) — H0.
- Capture history into SEE threshold (line 374) — H0.
- History bonus scaled by score-diff (line 1188) — strongly negative.
- Equal-weight expanded correction blends (lines 1201, 1860-1866, 2672) — the *weighted*
  version is what Coda kept; equal-weight dilutes pawn signal.

### Stale claims in the body, corrected
- **"In-check ProbCut (from Stockfish)"** (body §ProbCut): NOT present in current source —
  all of RFP/razor/NMP/ProbCut sit inside the `!node->isInCheck` block. Treat as removed.
- All **"GoChess"** comparison lines predate Coda's v9 + 5-source corrhist + 4D history;
  read them as historical, not current gaps.

### Net result — what actually survives as testable for Coda
Three ideas, ranked in the top section: (1) **node-count move ordering at ply<3**
(headline, genuinely untested, medium effort), (2) **IIR on stale TT entries** (trivial),
(3) **threat-gated RFP improving discount** (low effort). Minor/marginal also-rans not
promoted: Caissa's depth-scaled qsearch capture-count cap (`Search.cpp:1204`) — Coda's
`QS_MAX_CAPTURES` infra exists at a no-op 32 (`experiments.md` line 3936); the aggressive
1/2/3-move-at-deep-qs variant is untested but low-magnitude.
