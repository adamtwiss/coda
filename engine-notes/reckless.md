# Reckless Chess Engine - Crib Notes

Source: `~/chess/engines/Reckless/`
Language: Rust
NNUE: Dual-accumulator (768x10 PST + 66864 threats -> 2x768 -> 16 -> 32 -> 1x8) with pairwise activation
Rating: #1 in local RR at +362 Elo, +470 Elo above GoChess-v5

---

## 1. NNUE Architecture

### Network Topology: (768x10 + threats) -> 2x768 -> 16 -> 32 -> 1x8

This is the most sophisticated NNUE architecture in our review set.

**Feature Transformer (two independent accumulators):**

1. **PST Accumulator** (piece-square): `INPUT_BUCKETS * 768 -> L1_SIZE (768)` per perspective
   - 10 king input buckets (not 16 like Alexandria/us), layout:
     ```
     0 1 2 3 3 2 1 0    (rank 8)
     4 5 6 7 7 6 5 4    (rank 7)
     8 8 8 8 8 8 8 8    (rank 6)
     9 9 9 9 9 9 9 9    (ranks 1-5)
     ```
   - Fine granularity in back ranks (king safety), coarse elsewhere
   - Horizontally mirrored (king file >= 4 flips)
   - Uses **FinnyTable-style accumulator cache** (`AccumulatorCache`): per `[pov][flip][bucket]` with stored piece/color bitboards. On king bucket change, computes delta from cached state rather than full recompute
   - Feature index: `bucket * 768 + 384 * (color != pov) + 64 * piece_type + (square ^ flip)`

2. **Threat Accumulator**: `66864 -> L1_SIZE (768)` per perspective
   - Encodes actual piece-to-piece attack relationships on the board
   - 66864 threat features indexed by `(attacking_piece, from_sq, attacked_piece, to_sq)` with interaction filtering
   - Uses a `PIECE_INTERACTION_MAP` that excludes certain pairs (e.g., pawn-pawn same color attacks)
   - Semi-exclusion for same-piece-type pairs (only one direction counted)
   - Threat weights are int8 (vs int16 for PST), expanded via `convert_i8_i16` at inference time
   - Incrementally updated: `on_piece_move`, `on_piece_change`, `on_piece_mutate` callbacks push deltas

**Pairwise Activation (FT output):**
- Combines PST + threat accumulators: `pst[i] + threat[i]` before activation
- Splits each perspective's 768 outputs into first-half and second-half (384 each)
- First half: `clamp(x, 0, QA=255)` (CReLU)
- Second half: `min(x, QA=255)` (no lower clamp — signed)
- Pairwise multiply: `first_half[i] * second_half[i]` via `mul_high_i16` with shift
- Output: uint8 packed (768 values per perspective = 1536 total, but pairwise reduces to 768)
- Result is `[u8; L1_SIZE]` (768 bytes)

**Hidden Layers (float32 after L1):**
- L1: `768 -> 16` (int8 weights, NNZ-sparse matmul with output buckets)
  - Uses non-zero element tracking (`find_nnz`) — only multiplies non-zero FT outputs
  - Processes 4 bytes at a time (`CHUNKS=4`) via `dpbusd` (AVX2 dot product)
  - Dequantized to float after L1: `pre_act * DEQUANT_MULTIPLIER + bias`
  - Activation: `clamp(x, 0.0, 1.0)` (ReLU clamped to [0,1])
- L2: `16 -> 32` (float32 weights, standard matmul)
  - Activation: `clamp(x, 0.0, 1.0)`
- L3: `32 -> 1` (float32 weights, dot product + bias)
  - Per output bucket

**Output Buckets:** 8, indexed by piece count:
```
pieces:  0-8 -> bucket 0
pieces:  9-12 -> bucket 1
pieces: 13-16 -> bucket 2
pieces: 17-19 -> bucket 3
pieces: 20-22 -> bucket 4
pieces: 23-25 -> bucket 5
pieces: 26-28 -> bucket 6
pieces: 29-32 -> bucket 7
```

**Quantization:**
- FT_QUANT = 255, L1_QUANT = 64, FT_SHIFT = 9
- DEQUANT_MULTIPLIER = `(1 << 9) / (255 * 255 * 64)` = 0.0001228
- NETWORK_SCALE = 380
- Final: `l3_output * 380`

**SIMD:** AVX-512, AVX2, NEON (compile-time selected)

### Compared to GoChess NNUE
| Feature | Reckless | GoChess |
|---------|----------|---------|
| Input features | PST (768x10) + Threats (66864) | HalfKA (40960) |
| King buckets | 10 (non-uniform) | 16 (uniform) |
| Output buckets | 8 (piece count) | 8 (material) |
| Hidden layers | 768->16->32->1 | v5: Nx2->1x8 (shallow wide) |
| FT activation | Pairwise (CReLU * signed) | Pairwise + CReLU/SCReLU (UPDATE: now supports both) |
| L1 activation | ReLU [0,1] | CReLU |
| Threat features | 66864 attack-relation features | None |
| FinnyTable cache | Yes (per pov/flip/bucket) | Yes (UPDATE: merged) |
| NNZ-sparse L1 | Yes (skip zero FT outputs) | No |
| FT width | 768 | Dynamic (1024/1536/768pw/any) |

**Key insight**: The threat accumulator is a major novelty. Instead of relying solely on piece-square features (which encode static placement), Reckless explicitly encodes which pieces attack which other pieces. This gives the network direct access to tactical information that other architectures must learn indirectly. The 66864-feature threat space is large but sparse (typically <100 active threats per position), and incremental updates keep it efficient.

---

## 2. Search Architecture

### Iterative Deepening
- Depth 1 to MAX_PLY
- Uses Rust generics for node type (Root/PV/NonPV) — zero-cost compile-time dispatch

### Aspiration Windows
- Initial delta: `13 + average^2 / 23660` (score-adaptive — wider at extreme scores)
- On fail-low: `beta = (3*alpha + beta) / 4`, `alpha = score - delta`, `delta += 27*delta/128`
- On fail-high: `alpha = max(beta - delta, alpha)`, `beta = score + delta`, `reduction += 1`, `delta += 63*delta/128`
- Average maintained as running average: `(prev_average + score) / 2`
- **Optimism**: `169 * best_avg / (best_avg.abs() + 187)` — Stockfish-style contempt from shared best stats
- Compare to ours: delta=15, growth 1.5x. **(UPDATE 2026-03-21: GoChess now has aspiration contraction: fail-low (3a+5b)/8, fail-high (5a+3b)/8.)** They have score-adaptive delta and optimism.

### Draw Detection
- `is_draw(ply)` — repetition/50-move
- **Upcoming repetition**: `upcoming_repetition(ply)` — detects if a repetition is reachable, adjusts alpha to draw score. Applied at both main search and qsearch entry. This avoids searching into positions that will inevitably draw.

### Mate Distance Pruning
- `alpha = max(alpha, mated_in(ply))`, `beta = min(beta, mate_in(ply+1))`
- Compare to ours: We don't have this. Trivial to add.

---

## 3. Pruning Techniques

### Reverse Futility Pruning (RFP)
- Conditions: `!tt_pv && !excluded && !in_check && estimated_score >= beta && !is_loss(beta) && !is_win(estimated_score)`
- Margin: `1125*d^2/128 + 26*d - 77*improving + 519*|correction|/1024 + 32*(d==1) - 64*(no_threats && !in_check)`
  - At depth 5: ~220 + 130 = ~350 (non-improving), ~273 (improving)
  - Correction-aware: larger correction = stricter margin (harder to prune)
  - Threat-aware: if our pieces are not under attack, reduce margin by 64
- Returns: `beta + (estimated_score - beta) / 3` (dampened return, not raw eval)
- **No depth guard** — applies at all depths where margin is met
- Compare to ours: We use `85*d (improving) / 60*d (non-improving)`, depth<=8. They have quadratic margin, correction awareness, and dampened return.

### Razoring
- Conditions: `!PV && !in_check && estimated_score < alpha - 299 - 252*d^2 && alpha < 2048 && !tt_move.is_quiet()`
- TT-move guard: only razors when TT move is noisy (not quiet) — avoids razoring when a good quiet exists
- Margin at depth 1: 551, depth 2: 1307, depth 3: 2567 (very large, quadratic growth)
- Compare to ours: We use `400+100*d` (linear). They use quadratic with TT-move guard.

### Null Move Pruning (NMP)
- Conditions: `cut_node && !in_check && !excluded && !potential_singularity && estimated_score >= beta && estimated_score >= eval && eval >= beta - 9*d + 126*tt_pv - 128*improvement/1024 + 286 - 20*(cutoff_count < 2) && ply >= nmp_min_ply && has_non_pawns && !is_loss(beta)`
- Additional guard: `!(tt_bound == Lower && tt_move.is_capture() && captured >= Knight)` — don't NMP when TT says a good capture exists
- Reduction: `(5154 + 271*d + 535*clamp(estimated_score-beta, 0, 1073)/128) / 1024`
  - At depth 10, eval=beta+100: (5154 + 2710 + 535*100/128)/1024 ≈ 8
- **NMP Verification at depth >= 16**: If `nmp_min_ply == 0 && depth >= 16`, sets `nmp_min_ply = ply + 3*(depth-R)/4` and does a verification search. Prevents zugzwang-based NMP failures at high depth.
- **Cutoff count feedback**: `-20*(cutoff_count < 2)` — less aggressive when child nodes haven't been cutting off
- Returns raw score (not clamped to beta)
- Compare to ours: `3 + d/3 + min((eval-beta)/200, 3)`. They have much more nuanced conditions (cut_node only, verification, TT capture guard, cutoff count feedback, potential singularity guard).

### ProbCut
- Conditions: `cut_node && !is_decisive(beta) && (!valid(tt_score) || tt_score >= probcut_beta && !decisive) && !tt_move.is_quiet()`
- Beta: `beta + 269 - 72*improving`
- **Three-stage process**:
  1. QS pre-filter: `-qsearch(-probcut_beta, -probcut_beta+1)`
  2. Adaptive depth: `probcut_depth = clamp(base_depth - (score - probcut_beta)/295, 0, base_depth)` where `base_depth = max(depth-4, 0)`
  3. If QS passes and probcut_depth > 0: full search at adjusted_beta. If that fails, retry at base_depth
- **Dampened return**: `(3*score + beta) / 4` for non-decisive scores
- Only runs during good noisy moves (breaks at BadNoisy stage)
- Compare to ours: We have basic ProbCut at margin 170. They have QS pre-filter (tested in 2 other engines), adaptive depth, and dampened return.

### Singular Extensions (SE)
- Depth guard: `depth >= 5 + tt_pv` (depth 5 normally, depth 6 if tt_pv)
- Conditions: `!ROOT && !excluded && tt_depth >= depth-3 && tt_bound != Upper && valid(tt_score) && !decisive(tt_score) && ply < 2*root_depth`
- Singular beta: `tt_score - depth - depth*(tt_pv && !PV)` (tighter for former-PV nodes)
- Singular depth: `(depth-1) / 2`
- **Triple extensions**:
  - Extension 1: `score < singular_beta`
  - Extension 2: `score < singular_beta - double_margin` where `double_margin = 200*PV - 16*tt_move.is_quiet() - 16*|correction|/128`
  - Extension 3: `score < singular_beta - triple_margin` where `triple_margin = 288*PV - 16*tt_move.is_quiet() - 16*|correction|/128 + 32`
- **Multi-Cut**: `score >= beta && !decisive` => return `(2*score + beta) / 3` (dampened)
- **Negative Extensions**: `tt_score >= beta` => extension = -2; `cut_node` => extension = -2
- **Ply limiter**: `ply < 2 * root_depth` prevents explosive growth
- **Recapture extension**: If not SE-eligible and PV and tt_move is noisy recapture, extend +1
- Compare to ours: Our SE is broken (-58 Elo). Reckless has all the modern SE features: ply limiter, triple extensions, correction-aware margins, multi-cut with dampened return, aggressive negative extensions.

### Late Move Pruning (LMP)
- Conditions: `!ROOT && !in_check && !is_loss(best_score)`
- Threshold: `(2515 + 130*clamp(improvement,-100,218)/16 + (946 + 79*clamp(improvement,-100,218)/16)*d^2) / 1024`
  - At depth 1: ~3-4, depth 2: ~5-7, depth 3: ~10-14
  - **Continuous improvement scaling** — not just boolean improving, but proportional to eval improvement magnitude
- Sets `skip_quiets = true` (all remaining quiets skipped)
- Compare to ours: `3+d^2` with 50% more when improving. They use continuous improvement magnitude.

### Futility Pruning (FP)
- Conditions: `!in_check && is_quiet && depth < 14 && !is_direct_check(mv) && !is_loss(best_score)`
- Value: `eval + 88*d + 63*history/1024 + 88*(eval >= beta) - 114`
  - History-adjusted: moves with good history get higher futility value (harder to prune)
  - Beta-aware: +88 when eval already beats beta (less likely to need the move)
  - Depth 1: eval - 26 + 63*hist/1024, depth 5: eval + 326 + 63*hist/1024
- If `futility_value <= alpha` and `!is_direct_check(mv)`: sets `skip_quiets = true` and continues
- Also updates `best_score` to `futility_value` if applicable
- Compare to ours: `80+d*80`. They have much richer formula with history, beta-awareness, and check guard.

### Bad Noisy Futility Pruning (BNFP)
- Conditions: `!in_check && depth < 12 && stage == BadNoisy && !is_direct_check(mv) && !is_loss(best_score)`
- Value: `eval + 71*d + 69*history/1024 + 81*captured_value/1024 + 25`
- If `noisy_futility <= alpha`: **breaks** (stops all remaining bad noisy moves)
- Compare to ours: We don't have separate bad-noisy futility. Novel idea.

### SEE Pruning
- Conditions: `!ROOT && !is_loss(best_score)`
- Quiet threshold: `min(-16*d^2 + 52*d - 21*history/1024 + 22, 0)`
  - Depth 1: min(58, 0) = 0 (no pruning), depth 2: min(62, 0) = 0, depth 3: min(34, 0) = 0, depth 5: min(-254, 0) = -254
  - Quadratic growth, history-adjusted (bad-history moves pruned more aggressively)
- Noisy threshold: `min(-8*d^2 - 36*d - 32*history/1024 + 11, 0)`
  - Always negative at depth >= 1 (always prunes losing captures)
- Compare to ours: We use linear `-20*d^2`. They have separate quadratic formulas for quiet/noisy with history adjustment.

### Hindsight Reductions (from parent)
- **Depth increase (+1)**: If `parent_reduction >= 2247 && eval + parent_eval < 0` (parent was heavily reduced and this side is worse than expected — search deeper to compensate)
- **Depth decrease (-1)**: If `!tt_pv && parent_reduction > 0 && depth >= 2 && eval + parent_eval > 59` (both sides think position is fine — reduce)
- Compare to ours: We don't have this. 5+ engines have it (Alexandria, Tucano, Berserk, Stormphrax, Koivisto).

### IIR (Internal Iterative Reduction)
- Not explicit in the search — instead, when no TT entry exists, the TT write of `(hash, SOME, raw_eval, NONE, None, NULL)` serves as a shallow probe marker. The depth penalty comes implicitly through the estimatedScore/eval path.

---

## 4. LMR (Late Move Reductions)

### Two Separate LMR Paths

**Path 1: Full LMR (depth >= 2 && move_count >= 2)**

Base reduction: `250 * log2(move_count) * log2(depth)` (in 1/1024 units)

Adjustments (all in 1/1024 units):
- `-65 * move_count` (less reduction for early moves)
- `-3183 * |correction| / 1024` (less reduction when position is uncertain)
- `+1300 * alpha_raises` (more reduction after alpha has been raised — subsequent moves less likely to beat new alpha)
- Quiet: `+1972 - 154*history/1024`
- Noisy: `+1452 - 109*history/1024`
- PV: `-411 - 421*(beta-alpha)/root_delta` (less reduction in PV, scaled by window width)
- tt_pv: `-371`
- tt_pv && tt_score > alpha: additional `-656`
- tt_pv && tt_depth >= depth: additional `-824`
- Noisy recapture: `-910`
- !tt_pv && cut_node: `+1762`
- !tt_pv && cut_node && no tt_move: additional `+2116`
- !improving: `+(438 - 279*improvement/128).min(1288)` (continuous improving scale)
- In check (after move): `-966`
- Child cutoff_count > 2: `+1604`
- tt_score < alpha: `+600`
- !PV && parent_reduction > reduction + 512: `+128`

**LMR Extension**: If `reduction < -3072 && move_count <= 3`, extend by 1 (very good early moves get deeper search)

Reduced depth: `clamp(new_depth - reduction/1024, 1, new_depth + 1 + lmr_extension) + 2*PV`

**Do-Deeper/Do-Shallower after LMR re-search:**
- `+1 if score > best_score + 50 + 4*reduced_depth` (surprisingly good — search deeper)
- `-1 if score < best_score + 5 + reduced_depth` (barely better — reduce)

**Path 2: Non-LMR FDS (depth < 2 || move_count < 2, non-PV, non-first-move)**

Similar formula but different coefficients:
- Base: `238 * log2(move_count) * log2(depth)`
- Different history divisors, cut_node bonuses
- tt_move: `-3316` (massive reduction decrease for TT move)
- Threshold-based: reduces depth by 1 if reduction >= 3072, by 2 if >= 5687 and depth >= 3

### Compared to GoChess LMR
| Feature | Reckless | GoChess |
|---------|----------|---------|
| Base formula | 250*log2*log2 / 1024 | C=1.5 + log*log/2.36 (split cap/quiet) |
| History adjustment | -154*hist/1024 (quiet), -109/1024 (noisy) | -history/5000 |
| Correction awareness | -3183*|corr|/1024 | No |
| Alpha-raises count | +1300 per raise | No |
| Child cutoff count | +1604 if >2 | No |
| tt_pv adjustments | 3 levels (-371, -656, -824) | No |
| Cut-node extra | +1762 (+2116 if no tt_move) | No |
| Continuous improving | 438 - 279*impr/128 | Boolean +1 |
| In-check reduction | -966 | No |
| Recapture bonus | -910 | No |
| Do-deeper/shallower | Yes (50+4*rd / 5+rd thresholds) | No |
| LMR extension | Yes (reduction < -3072) | No |
| Parent reduction | +128 if parent was heavily reduced | No |

---

## 5. Move Ordering

### Staged Move Picker
Stages: HashMove -> GenerateNoisy -> GoodNoisy -> GenerateQuiet -> Quiet -> BadNoisy

### Hash Move
- Pseudo-legality check before returning
- No TT move in QSearch picker (separate `new_qsearch()` starts at GenerateNoisy)

### Good Noisy Scoring
- Score: `16 * captured_value + noisy_history`
- Noisy history: threat-aware, `factorizer + bucket[to_threatened][captured_type]`
- In check: simplified — `10000 - 1000 * piece_type` (lighter pieces first)
- SEE threshold for good/bad split: `-score/46 + 109` (dynamic — better-scored captures get a stricter SEE threshold; worse-scored get slack)

### Quiet Scoring
- `quiet_history + conthist(ply-1) + conthist(ply-2) + conthist(ply-4) + conthist(ply-6)`
- **Escape bonus**: threatened pieces get large bonuses for moving:
  - Queen under rook/minor/pawn attack: +20000
  - Rook under minor/pawn attack: +14000
  - Knight/Bishop under pawn attack: +8000
- **Check bonus**: +10000 for moves that give check (uses precomputed `checking_squares`)
- **Queen danger malus**: -10000 for queen moves into minor-attacked squares
- **No killers, no counter-moves** — all quiet ordering is history-based + tactical bonuses
- Compare to ours: We have killers + counter-move + main history + cont-hist (2 plies). They skip killers/countermoves entirely but use 4-ply cont-hist + tactical bonuses.

### Opponent Move Ordering (quiet move ordering using eval difference)
- After opponent's quiet move, compute: `bonus = clamp(819 * (-(eval + parent_eval)) / 128, -124, 312)`
- Updates opponent's quiet history: if opponent's move made eval bad for them, give their move a bonus (it was actually good for them from their perspective — wait, actually this is the reverse: if `-(eval + parent_eval)` is positive, meaning the position deteriorated for us, the opponent's move gets a bonus)
- Compare to ours: We don't have this. Alexandria and Obsidian have similar.

### Bad Noisy
- Captures that failed the dynamic SEE threshold are deferred to BadNoisy stage
- Searched last, after all quiets

### History Tables

**Quiet History**: Factorized threat-aware butterfly
- `[color][from][to]` with sub-structure:
  - `factorizer: i16` (MAX 1852) — general quality of this from-to pair
  - `buckets[from_threatened][to_threatened]: i16` (MAX 6324) — threat-aware quality
- Get: `factorizer + bucket[from_threatened][to_threatened]`
- Update: both factorizer and bucket updated with gravity formula
- Compare to ours: `[color][from][to]` simple butterfly, no threat awareness

**Noisy History**: Factorized threat-aware capture history
- `[piece][to]` with sub-structure:
  - `factorizer: i16` (MAX 4524) — general quality
  - `buckets[captured_type][to_threatened]: i16` (MAX 7826) — per-capture-type, threat-aware
- Get: `factorizer + bucket[captured][to_threatened]`
- Compare to ours: `[attacker][to][victim]` simple capture history

**Continuation History**: `[in_check][capture][piece][to][piece][to]`
- 4 ply lookback: plies 1, 2, 4, 6
- Indexed by (in_check, is_capture) of the parent move — separate tables for different move types
- MAX 15168
- Compare to ours: 2-ply only, no check/capture indexing

**Gravity formula**: `entry += bonus - |bonus| * entry / MAX` (standard)

---

## 6. Transposition Table

### Structure
- 3 entries per cluster (32-byte aligned clusters)
- Entry: 10 bytes (key16, mv16, score16, raw_eval16, depth8, flags8)
- Flags: bound(2 bits) + tt_pv(1 bit) + age(5 bits)
- Age cycle: 32 (5-bit counter)

### Index Computation
- Lemire fast modulo: `((hash as u128) * (len as u128)) >> 64`

### Replacement Policy
- Priority 1: Same key or empty slot
- Priority 2: Lowest quality = `depth - 4 * relative_age`
- Skip write if: `key == entry.key && depth + 4 + 2*tt_pv <= entry.depth && same_age`

### TT Cutoff Conditions
- `!PV && !excluded`
- `tt_depth > depth - (tt_score < beta)` (1 ply slack when tt_score < beta)
- Bound-type-aware: Upper bound needs score <= alpha (with depth > 5 guard if cut_node), Lower bound needs score >= beta (with depth > 5 guard if !cut_node)
- Halfmove clock guard: no cutoff when `halfmove_clock >= 90`
- **TT cutoff history bonus**: When quiet TT move gives beta cutoff and parent move_count < 4, update quiet and continuation history with the TT move

### Mate Score Handling
- Ply-adjusted on write: `score + signum * ply`
- On read: `score - ply` (win) or `score + ply` (loss)
- Halfmove clock aware: downgrades potentially false mate/TB scores when 50-move rule is near

### Compare to GoChess TT
| Feature | Reckless | GoChess |
|---------|----------|---------|
| Entries/cluster | 3 | 4 |
| Entry size | 10 bytes | Similar |
| Cluster size | 32 bytes | 32 bytes |
| Age bits | 5 (32 cycle) | Similar |
| Lockless | No (single-writer per cluster) | Yes (XOR-verified atomics) |
| tt_pv stored | Yes | No |
| raw_eval stored | Yes | No |
| Index method | Lemire fast modulo | Power-of-2 masking |
| Huge pages | Yes (mmap + MADV_HUGEPAGE) | No |

---

## 7. Eval Pipeline

```
raw_eval = nnue.evaluate(board)
eval = raw_eval * (21061 + material) / 26556    // material scaling
     + optimism * (1519 + material) / 26556      // optimism/contempt
eval = eval * (200 - halfmove_clock) / 200       // 50-move decay
eval += correction_value                          // correction history
eval = clamp(eval, -TB_WIN_IN_MAX+1, TB_WIN_IN_MAX-1)
```

**Material scaling**: `(21061 + material) / 26556` — eval is scaled down in endgames. With material=0, scale = 0.793. With material=8000, scale = 1.095. This replaces a separate endgame scaling function.

**Optimism**: Stockfish-style, `169 * best_avg / (|best_avg| + 187)`, added scaled by `(1519 + material) / 26556`.

**50-move decay**: `(200 - halfmove_clock) / 200` — linear fade to zero as 50-move rule approaches.

**Correction History (6 components, div 77):**
1. Pawn key
2. Minor key
3. Non-pawn White key
4. Non-pawn Black key
5. Continuation correction ply-2
6. Continuation correction ply-4

All are `CorrectionHistory` tables (65536 entries, atomically updated, MAX 14605). Continuation correction is `ContinuationCorrectionHistory` (indexed by `[in_check][capture][piece][to]` of parent, then `[piece][to]` of current, MAX 16282).

Compare to ours: We have pawn correction only. They have 6 components including per-color non-pawn keys, minor keys, and deep continuation correction. This is the 11-engine consensus approach.

---

## 8. Time Management

### Soft/Hard Bounds

**Fischer (main + increment):**
- `soft_scale = 0.024 + 0.042 * (1 - exp(-0.045 * fullmove_number))`
  - At move 1: ~0.026, move 10: ~0.050, move 30: ~0.064, move 60: ~0.066
  - Asymptotic: approaches 0.066 in endgame
- `soft = soft_scale * (main - overhead) + 0.75 * inc`
- `hard = 0.742 * (main - overhead) + 0.75 * inc`
- Both capped at `main - overhead`
- Overhead: 15ms built-in

**Cyclic (moves to go):**
- `base = main/moves + 0.75*inc`
- Soft: `1.0 * base`
- Hard: `5.0 * base`
- Both capped at `main + inc`

### Soft Stop Multiplier (5 factors, multiplicative)

1. **Nodes factor**: `max(2.7168 - 2.2669 * (best_move_nodes / total_nodes), 0.5630)`
   - 10% of nodes: 2.49x, 50%: 1.58x, 90%: 0.68x
2. **PV stability**: `max(1.25 - 0.05 * pv_stability_count, 0.85)` (min after 8 stable iterations)
3. **Eval stability**: `max(1.2 - 0.04 * eval_stability_count, 0.88)` (min after 8 stable iterations)
4. **Score trend**: `clamp(0.8 + 0.05 * (prev_best_score - current_score), 0.80, 1.45)`
   - Score dropping: extend time up to 1.45x
   - Score rising: use less time (floor 0.80x)
5. **Best move changes**: `1.0 + best_move_changes / 4.0` (halved each iteration)

All 5 factors multiply together: `nodes * pv_stability * eval_stability * score_trend * best_move_changes`

### Soft Stop Voting (SMP)
- Each thread votes independently on whether to soft-stop
- Majority rule: `(thread_count * 65 + 99) / 100` votes needed (65% majority)
- Thread can retract its vote if conditions change

### Hard Stop
- Checked every 2048 nodes (thread 0 only for timed games)
- All threads check shared status flag every 2048 nodes

### Compare to GoChess
| Feature | Reckless | GoChess |
|---------|----------|---------|
| Node-based scaling | 5-factor multiplicative | Best-move instability only |
| PV stability | Dedicated factor (0.85-1.25) | Instability counter |
| Eval stability | Dedicated factor (0.88-1.2) | No |
| Score trend | 0.80-1.45x | 2.0x/1.5x/1.2x tiered (UPDATE: merged) |
| Fullmove scaling | Exponential approach | No |
| SMP voting | 65% majority voting | No (main thread decides) |

---

## 9. Lazy SMP

- Multiple threads share TT and correction histories
- NUMA-aware: `NumaReplicator` for correction history, thread pinning via `bind_thread`
- Per-thread: Board, SearchInfo, NNUE accumulator, quiet/noisy/continuation histories, stack
- Node counter: per-thread sharded atomics (cache-line aligned via `#[repr(align(64))]`)
- Worker threads use persistent thread pool with send/receive channels
- Soft-stop voting prevents premature termination when only one thread thinks search is done

Compare to ours: Similar TT-sharing model. They additionally share correction history (ours is not shared). NUMA awareness and soft-stop voting are novel.

---

## 10. QSearch

### Stand Pat
- If `best_score >= beta` and not decisive: return `beta + (best_score - beta) / 3` (dampened)
- Stores stand-pat to TT if no existing entry

### Move Generation
- Generates noisy moves only (unless in check/TT has quiet lower-bound)
- Skip quiets: `!(in_check && is_loss(best_score)) && !(tt_move.is_quiet() && tt_bound != Upper)`
  - In check with mating threat: search all evasions
  - TT says a quiet move has a lower bound: search quiets

### QS Pruning
- **LMP**: Break after 3 non-checking moves
- **SEE threshold**: `(alpha - eval) / 8 - 100` (alpha-relative, scales with how far behind)

### QS Beta Blending
- After all moves: if `best_score >= beta && !decisive`: `best_score = (best_score + beta) / 2`

### QS History Update
- On beta cutoff: update noisy_history (bonus 106) or quiet_history (bonus 172)

---

## 11. Notable Differences from GoChess

### Things Reckless Has That We Don't:
1. **Threat accumulator in NNUE** — 66864 attack-relationship features, incrementally updated
2. ~~**FinnyTable accumulator cache**~~ — **(UPDATE 2026-03-21: GoChess now has Finny tables)**
3. **NNZ-sparse L1 matmul** — only processes non-zero FT outputs
4. **6-component correction history** (pawn, minor, non-pawn white, non-pawn black, cont-corr ply-2, cont-corr ply-4)
5. **Mate distance pruning** — trivial to add
6. **Upcoming repetition detection** — in both main search and qsearch
7. **Alpha-raises count in LMR** (+1300 per raise)
8. **Child cutoff count in LMR** (+1604 if cutoff_count > 2)
9. **Continuous improvement scaling** in LMP/LMR (not just boolean)
10. **Correction-aware LMR** (-3183*|correction|/1024)
11. **Hindsight reductions** (depth +1/-1 from parent's reduction + eval trajectory)
12. **Bad noisy futility pruning** — separate futility for losing captures
13. **Dynamic SEE threshold** for good-capture/bad-capture split: `-score/46 + 109`
14. **Check bonus in quiet scoring** (+10000 for checking moves)
15. **Escape bonus in quiet scoring** (Q+20000, R+14000, minor+8000)
16. **Queen danger malus** (-10000 for queen into minor-attacked squares)
17. **4-ply continuation history** (plies 1, 2, 4, 6)
18. **Opponent eval history** — updates opponent's history based on eval change
19. **5-factor time management** with multiplicative scaling
20. **SMP soft-stop voting** (65% majority)
21. **NUMA-aware thread pool**
22. **Huge pages** (Linux mmap + MADV_HUGEPAGE for TT)
23. **NMP cut_node restriction** — only NMP at cut nodes
24. **NMP verification at depth >= 16**
25. **ProbCut QS pre-filter + adaptive depth**
26. **Triple singular extensions** with correction-aware margins
27. **Optimism/contempt** from shared best-score stats
28. **tt_pv flag** stored in TT and used for 3-level LMR adjustment
29. **No killers/countermoves** — entirely history-based quiet ordering

### Things We Have That Reckless Doesn't:
1. **Killers** (they rely entirely on history)
2. **Counter-move heuristic** (they don't have it)
3. **Lockless TT** (they use non-atomic writes, safe because of SMP design)
4. **Classical eval fallback** (they're NNUE-only)

---

## 12. Ideas Worth Testing from Reckless (Prioritized)

### Already Referenced in SUMMARY.md:
These ideas from Reckless are already logged. Reckless confirms/strengthens the case:
- Multi-source correction history (6 components) — **11-engine consensus, HIGH priority**
- Alpha-raises count in LMR — **2 engines (Reckless + Altair)**
- Child cutoff count feedback — **Reckless-specific, 5 lines**
- Halfmove clock eval decay — **already tested, H0 at -3.0 Elo**
- Beta-cutoff score blending — **already merged in QS; main search version untested**
- Bad noisy futility — **Reckless-specific**
- Deeper cont-hist (plies 4, 6) — **11 engines, tested at -58 Elo but implementation may have been wrong**
- Threat-aware history — **12 engines**, Reckless has factorized approach
- Escape bonus in move ordering — **Reckless + Obsidian**
- Opponent eval history — **Reckless + Obsidian + Alexandria**
- ProbCut QS pre-filter — **Reckless + Alexandria + Tucano**
- Hindsight reductions — **5 engines**
- Node-based TM — **6 engines**

### New/Strengthened Ideas from This Review:

1. **Mate Distance Pruning** — Trivial (3 lines), universal technique. `alpha = max(alpha, mated_in(ply)); beta = min(beta, mate_in(ply+1))`. Free Elo.

2. **Correction-Aware LMR** — `-3183*|correction|/1024`. When correction history is large (position type is systematically misevaluated), reduce less. 1 line. We already compute correction value.

3. **NMP cut_node Restriction** — Only allow NMP at cut nodes. This is a major structural difference from our NMP. Worth testing as it prevents NMP at all-nodes where null move might miss a crucial resource.

4. **RFP Correction Awareness** — `+519*|correction|/1024` in margin. High correction = uncertain eval = harder to prune. 1 line.

5. **RFP Dampened Return** — `beta + (estimated_score - beta) / 3` instead of raw eval. Prevents over-estimation at cutoff. Already tested and rejected for us as "RFP score dampening" at -16.7 Elo, but Reckless formula is different (dampens toward beta, not toward eval).

6. **Dynamic Good-Capture SEE Threshold** — `-score/46 + 109` for good/bad noisy split. Better-scored captures need to prove themselves more; worse-scored captures get more slack. Very different from a fixed threshold.

7. **Check Bonus in Quiet Ordering** — +10000 for moves that give check. Uses precomputed `checking_squares[piece_type]`. Cheap if we already compute check squares.

8. **Upcoming Repetition Detection** — In both main search and QSearch. Adjusts alpha to draw score when repetition is reachable. Prevents wasting time searching dead-draw lines.

9. **LMR Extension for Very Good Early Moves** — If `reduction < -3072 && move_count <= 3`, add +1. The first few moves that get massive negative reduction (= very promising) get an extra ply.

10. **tt_pv Flag in TT** — 1 bit, used for 3-level LMR adjustment (-371 base, -656 if tt_score > alpha, -824 if tt_depth >= depth). Would require adding a bit to our packed TT entries.

11. **NMP TT Capture Guard** — Don't NMP when TT has a lower-bound capture of piece >= Knight. Prevents NMP when tactical resources exist.

12. **Quadratic SEE Pruning** — `-16*d^2 + 52*d - 21*hist/1024` for quiets. Our linear `-20*d^2` is simpler but less tuned.

13. **Threat Accumulator in NNUE** — Massive architectural innovation. Would require network retraining but explains much of the +470 Elo gap. Long-term consideration for NNUE architecture migration.
