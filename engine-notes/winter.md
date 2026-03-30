# Winter 4.15b - Engine Crib Notes

Source: `~/chess/engines/Winter/src/`

## Overview

Winter is a UCI engine by Jonathan Rosenthal. Key distinguishing feature: **native WDL (Win/Draw/Loss) scoring** throughout search. Scores are not centipawns -- they are `WDLScore{win, loss}` where `win` and `loss` are integers on a 0..4000 scale representing win% and loss% (draw% is implied). Comparison uses `value() = win - loss`. This means all pruning margins, aspiration deltas, etc. operate in WDL-space, not centipawn-space.

Net-based evaluation (NNUE-like). No classical eval fallback.

---

## 1. Search: Pruning Techniques

### Internal Iterative Reduction (IIR)
- **Depth guard**: `depth >= 2` and no TT entry and not in singular extension search (`!exclude_move`)
- **Action**: `depth--` (reduce by 1 ply)
- Applied in both PV and non-PV nodes

### IIR for PV nodes without TT entry (separate)
- **Condition**: PV node, not root, `depth >= 5`, no TT entry
- **Action**: `depth--`
- Credited to talkchess idea (http://talkchess.com/forum3/viewtopic.php?f=7&t=74769)
- This stacks with the general IIR above, so PV nodes without TT entry at depth>=5 get depth reduced by 2 total

### Static Null Move Pruning (Reverse Futility Pruning)
- **Node type**: Non-PV only (`kNW`)
- **Depth guard**: `depth <= 5`
- **Condition**: `!exclude_move`, `beta.is_static_eval()`, not in check
- **Margin formula**: `margin = kSNMPScaling * depth + kSNMPOffset - kSNMPImproving * improving * depth`
  - `kSNMPScaling = 709`
  - `kSNMPOffset = -122`
  - `kSNMPImproving = 190`
  - When improving: effective scaling = `(709 - 190) * depth - 122 = 519*depth - 122`
  - When not improving: `709*depth - 122`
- **Prune if**: `eval > beta + margin`
- **Return value**: `(eval + beta) / 2` (average of eval and beta, not just beta)

### Null Move Pruning (NMP)
- **Condition**: `eval >= beta`, `depth > 1`, side to move has non-pawn material, non-PV, not in check, `beta.is_static_eval()`
- **Reduction formula**: `R = (kNMPBase + depth * kNMPScale) / 128`
  - `kNMPBase = 485`, `kNMPScale = 40`
  - At depth 4: R = (485 + 160) / 128 = 5 (i.e., search at depth -1)
  - At depth 8: R = (485 + 320) / 128 = 6
  - At depth 16: R = (485 + 640) / 128 = 8
  - Effectively: R = 3.79 + 0.3125*depth
- **No verification search**
- **Tracks NMP failure**: `nmp_failed_node = true` if NMP fails, used to grant double singular extension

### Futility Pruning
- **Node type**: Non-PV only
- **Depth guard**: `depth - reduction <= 3` (uses LMR-reduced depth)
- **Condition**: not in check, move not giving direct check, not root, lower_bound >= kMinStaticEval, move is quiet (type < kEnPassant)
- **Margin formula**: `kFutileMargin[depth] + kFutilityImproving * depth * improving`
  - `kFutileMargin[d] = kFutilityOffset + kFutilityScaling * d`
  - `kFutilityOffset = 127`, `kFutilityScaling = 685`
  - `kFutilityImproving = 149`
  - Margin at depth 1: 127 + 685 = 812 (+ 149 if improving)
  - Margin at depth 2: 127 + 1370 = 1497 (+ 298 if improving)
  - Margin at depth 3: 127 + 2055 = 2182 (+ 447 if improving)
  - (These are WDL-space values, scale=4000, not centipawns)
- **Prune if**: `eval < alpha - margin`

### Late Move Pruning (LMP)
- **Depth guard**: `depth < 6` (kLMP array size is 6)
- **Condition**: not root, not in check, move not giving direct check, lower_bound >= kMinStaticEval, move is quiet (type < kEnPassant)
- **Formula**: `lmp[isPV][improving][depth] = (base + scalar*(d-1) + quad*(d-1)^2) / 128`
  - Non-PV, not improving: base=408, scalar=122, quad=53
  - Non-PV, improving: base=678, scalar=188, quad=83
  - PV, not improving: base=817, scalar=122, quad=53
  - PV, improving: base=569, scalar=188, quad=83

  Computed LMP thresholds (moves allowed before pruning):

  | depth | NW | NW-imp | PV | PV-imp |
  |-------|-----|--------|-----|--------|
  | 1     | 3   | 5      | 6   | 4      |
  | 2     | 4   | 7      | 8   | 7      |
  | 3     | 5   | 10     | 10  | 11     |
  | 4     | 7   | 15     | 13  | 16     |
  | 5     | 10  | 21     | 17  | 23     |

### SEE Pruning (depth 1)
- **Condition**: `depth == 1`, move is not en passant, not in check, not giving check
- **Action**: Skip moves with negative SEE
- Only at depth 1; no deeper SEE pruning for quiets

### QSearch SEE Pruning
- In quiescence search: skip captures with negative SEE (except en passant and when in check)

### TT Score Capping (Non-PV)
- When TT score > beta and not a mate score: returns `(score * 3 + beta) / 4` instead of raw TT score
- Dampens TT cutoff scores toward beta

---

## 2. Extensions

### Singular Extensions
- **Depth guard**: `depth >= kSingularExtensionDepth - 2 = 7`
- **TT depth requirement**: `entry->depth >= max(depth, kSingularExtensionDepth) - 3 = max(depth, 9) - 3`
- **Condition**: First move (i==0), not root, TT entry exists, TT bound is not upper bound, TT score is static eval, not (PV with only 1 legal move)
- **Singular beta**: `rBeta = WDLScore{beta.win - 2*depth, beta.loss + 2*depth}` (shrink toward draw by 2*depth on each side)
- **Singular depth**: `rDepth = (depth - 3) / 2`
- **On singularity confirmed** (score <= rAlpha): extend by 1 ply. If NMP also failed at this node, extend by 2 plies (double extension)
- **Multi-cut**: If singular search score >= beta, return score immediately (prune the whole subtree)

### One-move PV Extension
- If PV node and only 1 legal move: `depth++`

### No check extensions, no passed pawn extensions, no recapture extensions

---

## 3. Move Ordering

### Two sorting systems:

**Sort()** - Used in QSearch only. Simple MVV-LVA style:
- TT move: priority 20000
- Promotions: 10000 + moveType - (target piece exists ? 0 : 1)
- Captures: 1000 + 10*target_piece - moving_piece
- Quiets: history_score / 1000

**SortML()** - Used in main search. Machine-learned feature-weighted scoring:

All moves scored by summing weighted features (separate weight vectors for in-check vs not-in-check positions). 117 features total. Key features:

1. **TT move**: Returns maximum score immediately (32767)
2. **Killer moves**: Two killer slots per ply, separate feature weights for killer[0] and killer[1]
3. **Counter move**: Feature for matching counter-move table entry
4. **Continuation history**: 1-ply and 2-ply continuation history scores, each multiplied by learned weight
5. **Main history**: Color x Source x Destination history score, multiplied by learned weight
6. **Piece-type x Target-type**: 6x6 = 36 features for (moving_piece, captured_piece) pairs
7. **Move type**: 9 features (normal, castle, double-pawn, en-passant, capture, 4 promotions)
8. **Source/Destination PST**: Symmetric 10-entry PST for move source and destination (separate for knights)
9. **SEE negative**: Feature for losing captures
10. **Gives direct check**: Separate features for check with capture vs quiet check
11. **Taboo destination**: Feature for moving to king-attacked square
12. **Captures last moved piece**: Bonus feature
13. **Piece under attack**: Whether source square is attacked by opponent's last move
14. **Passed pawn rank**: Destination rank of passed pawn moves (6 features by rank)
15. **Pawn rank destination**: Rank of non-passed pawn moves (6 features)
16. **Pawn attacks piece**: Pawn move destination attacks enemy non-pawn
17. **Forcing state changes**: 4 features for transitions between forcing/non-forcing moves

The scoring is additive with pre-tuned integer weights (hardcoded in `hardcoded_params.h`). These weights were tuned via SPSA (`TUNE_ORDER` mode).

### Move ordering in search:
- TT move is swapped to front (not sorted)
- Remaining moves sorted by SortML on first non-TT move (lazy: only sort when i==1)

### History Tables

**Main history**: `history[2][64][64]` - Color x Source x Destination
- Update formula: `history[c][s][d] += 32 * score - history[c][s][d] * abs(score) / 512`
- Score = `min(depth * depth, 512)` (gravity-style update, capped at 512)

**Continuation history**: `continuation_history[2][6][64][6][64]` - two tables (1-ply and 2-ply ago) x PieceType x Destination x PieceType x Destination
- Same gravity update formula as main history
- 1-ply: indexed by opponent's last moved piece type and destination
- 2-ply: indexed by own previous move's piece type and destination

**Counter move table**: `counter_moves[2][6][64]` - Color x PieceType x Destination -> single best counter move

### No capture history table (captures scored by MVV-LVA + SEE in SortML features)

---

## 4. LMR (Late Move Reductions)

### Formula
4 separate LMR tables: {non-PV quiet, non-PV capture, PV quiet, PV capture}

Base formula: `floor(offset + log(depth) * log(moveCount) * multiplier)`

Parameters (internal representation, all multiplied by 0.01 scale):
- **Non-PV quiet**: offset = kLMROffset * 0.01 = -0.16, mult = kLMRMult * 0.01 = 0.86
  - `floor(-0.16 + log(d+1) * log(m+1) * 0.86)`
- **Non-PV capture**: offset = kLMROffsetCap * 0.01 = 0.15, mult = 0.86 * kLMRMultCap * 0.01 = 0.86 * 0.51 = 0.4386
  - `floor(0.15 + log(d+1) * log(m+1) * 0.4386)`
- **PV quiet**: offset = kLMROffsetPV * 0.01 = 0.21, mult = 0.86 * kLMRMultPV * 0.01 = 0.86 * 0.86 = 0.7396
  - `floor(0.21 + log(d+1) * log(m+1) * 0.7396)`
- **PV capture**: offset = kLMROffsetPVCap * 0.01 = 0.23, mult = 0.86 * 0.86 * 0.51 = 0.3772
  - `floor(0.23 + log(d+1) * log(m+1) * 0.3772)`

Result clamped to `[0, depth-1]`.

### LMR conditions
- Applied to all moves after the first (i > 0), or at root after the first 3 (i > 2, with index shifted by -2)
- **NOT applied when**: in check, or move gives direct check
- No history-based reduction adjustment
- No improving-based reduction adjustment
- No separate quiet vs capture adjustments beyond the table selection
- PV re-search: if reduced search beats alpha, full-depth re-search with full window (PVS pattern)

### Example LMR values (non-PV quiet):

| depth\move | 4   | 8   | 16  | 32  |
|------------|-----|-----|-----|-----|
| 4          | 1   | 2   | 2   | 3   |
| 8          | 2   | 3   | 3   | 4   |
| 16         | 2   | 3   | 4   | 5   |
| 32         | 3   | 4   | 5   | 6   |

---

## 5. Time Management

### Base time allocation
```
base_time = min(maxtime(clock), (58 * clock / min(50, moves_to_go) + 58 * inc) / 50)
maxtime(t) = max(0.8 * t, t - 100)
```

Default `moves_to_go = 22` when not specified (sudden death).

### Move stability (best move change)
- Tracks `time_factor`, starts at 1.0
- If best move is unchanged between iterations: `time_factor = max(time_factor * 0.9, 0.5)`
- If best move changes: `time_factor = 1.0`
- Stops early if `time_used > base_time * time_factor`
- Effect: stable best move can cut time to 50% of allocation; changing best move uses full time

### No fail-low extension, no score instability factor, no pondering-specific logic

### Time check frequency
- Only main thread (id==0) checks time
- `skip_time_check` counter: decremented each node, actual time check only when counter hits 0
- Counter reset to `min(512, max_nodes - current_nodes)` after each actual check

---

## 6. Aspiration Windows

- **Initial delta**: `kInitialAspirationDelta = 72` (in WDL-space, applied as `{+72, -72}` to win/loss)
- **Kick in at**: depth > 4 (depths 1-4 use full window)
- **On fail**: delta doubles, window widened in the failing direction
- On fail-low: also tightens beta slightly toward score: `beta = (max(score, kMinStaticEval) + beta*3) / 4`
- On fail-high: tightens alpha similarly: `alpha = (alpha*3 + min(kMaxStaticEval, score)) / 4`

---

## 7. Correction History (Error History)

Winter has a sophisticated **multi-dimensional correction history** that adjusts the static evaluation based on search error patterns. This is analogous to Stockfish's correction history but more elaborate.

### Types of correction history:
1. **Pawn correction**: Keyed by pawn hash, scale = 0.705
2. **Major piece correction**: Keyed by major piece hash, scale = 0.466
3. **Minor piece correction**: Keyed by major_hash >> 32, scale = 0.466
4. **RNG correction**: 4 independent hash functions x 4 16-bit sub-hashes = 16 correction entries total, each with scale = 0.34375

### Table size: 32768 entries each (`kErrorHistorySize = 1 << 15`), each entry stores `{float win_error, float loss_error}`

### Application:
- Raw eval's win% and loss% are adjusted: `win += sum(win_errors * scales) / 8192`, same for loss
- Clamped to valid probability range

### Update:
- On beta cutoff or alpha improvement (quiet best moves only, not in check)
- `win_val = clamp((eval.win_pct - static_eval.win_pct) * depth * 64, -200, 200)`
- Gravity update: `entry += value - entry * (abs(value) + leak) / 1024`
- Leak factor: `kCorrectionLeakScale * min(depth, 16)` where `kCorrectionLeakScale = 1.2`

---

## 8. Lazy SMP

- Shared TT only (16-byte entries, 4 per cache line)
- Per-thread: board, killers, counter moves, history, continuation history, error history, eval stack
- Helper thread depth scheduling: if >= half of threads are at or above current depth, skip depths based on `(id % 3) != (depth % 3)` and increment depth (avoids redundant work at same depth)
- Thread initialization is lazy (first search call)

---

## 9. Novel/Unusual Features

### WDL-native search
The most distinctive feature. All scores are 2D (win, loss) rather than 1D centipawns. This means:
- Pruning margins operate on win/loss probabilities
- Aspiration windows adjust both win and loss bounds
- Singular extension beta shrinks both win and loss by `2*depth`
- TT stores win and loss separately (int16 each)
- Draw detection returns contempt-adjusted WDL draw score, not 0

### ML-trained move ordering weights
Move ordering uses ~117 pre-tuned feature weights (separate sets for in-check and not-in-check), trained via SPSA. This is unusual -- most engines use ad-hoc scoring with MVV-LVA + history. Winter's approach is more like a lightweight linear model over hand-crafted features.

### TT score dampening
On non-PV TT cutoffs where score > beta and not mate: returns `(3*score + beta)/4` instead of raw score. This dampens the effect of possibly inflated TT scores.

### SNMP returns average
SNMP (reverse futility) returns `(eval + beta) / 2` rather than just `eval` or `beta`. This softens the pruning return value.

### No check extensions
Winter does NOT extend checks. It only has singular extensions and a one-move PV extension. This is unusual for a competitive engine.

### No history-based LMR adjustment
LMR has no adjustment for history score, improving status, or any other dynamic factor. The reduction is purely a function of depth and move count, with separate tables for PV/non-PV and quiet/capture.

### Double singular extension on NMP-failed nodes
If both NMP failed (score < beta after null move) AND the TT move is singular, extend by 2 plies instead of 1.

### Correction history in WDL space
Rather than correcting a single centipawn value, Winter corrects win% and loss% independently, using pawn hash, major piece hash, minor piece hash, and 16 RNG-based hash corrections. This is significantly more correction dimensions than typical engines.

### No razoring, no ProbCut, no history pruning
Winter lacks several common techniques:
- No razoring (dropping to QSearch at low depth with bad eval)
- No ProbCut (shallow search to prove a beta cutoff)
- No history-based pruning (pruning quiet moves with very negative history)
- No SEE pruning for quiets at depth > 1

---

## 10. Comparison with GoChess

### Techniques Winter has that GoChess might benefit from:
- **TT score dampening**: `(3*score + beta)/4` for non-PV TT cutoffs -- could reduce search instability
- **SNMP returning (eval+beta)/2** instead of eval -- softer pruning
- **Multi-dimensional correction history**: pawn hash + major hash + minor hash + RNG hash corrections
- **ML-tuned move ordering weights**: systematic optimization of feature weights for move ordering
- **Double singular extension on NMP-failed nodes**: extra extension when both NMP and singularity signal

### Techniques GoChess has that Winter lacks:
- Check extensions
- Razoring
- ProbCut
- History-based LMR adjustment
- History-based pruning
- SEE pruning at depth > 1 for quiets
- Recapture extensions
- Passed pawn extensions (removed in GoChess too)
- Capture history table (separate from continuation history)

### Parameter comparison (approximate, WDL values converted where possible):

| Feature | Winter | GoChess |
|---------|--------|---------|
| NMP R | 3.79 + 0.31*d | 3 + d/3 |
| RFP depth | <= 5 | <= 8 |
| Futility depth | <= 3 (after LMR) | lmrDepth-based |
| LMP depth | < 6 | depth-based |
| Singular depth | >= 7 | depth-based |
| Aspiration delta | 72 (WDL) | 15 (cp) |
| IIR depth | >= 2 | varies |
| Check extension | NO | YES (was removed) |
