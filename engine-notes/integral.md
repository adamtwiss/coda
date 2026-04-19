# Integral Chess Engine - Technical Review

Source: `~/chess/engines/integral/`
Version: Zekrom-v7 (git hash 3f15ba6b)
Language: C++20 (GCC >= 13 or Clang >= 10)
NNUE: Single factorized accumulator (768×12 → 1536) → 16 → 32 → 1×8, with i8 L1 weights
Rating: ~3650 CCRL / Top commercial tier

Last reviewed: 2026-04-18

---

## 1. NNUE Architecture

### 1.1 Network Topology: (768×12 Factorized) → 1536 → 16 → 32 → 1×8

**Architecture constants** (`shared/nnue/definitions.h:13-22`):
```
L1_SIZE = 1536
L2_SIZE = 16
L3_SIZE = 32
INPUT_BUCKETS = 12
OUTPUT_BUCKETS = 8
FT_QUANTIZATION = 255
L1_QUANTIZATION = 128
EVAL_SCALE = 200
```

Integral uses a **factorized HalfKA** input layer with 12 king-based input buckets. The feature transformer has dimension `[12][2][6][64][1536]` (bucket × perspective × piece_type × square × neurons), producing a 1536-neuron L1 for each perspective.

**Coda comparison**: We use 768×10 with a separate threat accumulator (66,864 threat features). Integral uses only PST features but achieves comparable strength through a larger L1 (1536 vs 768+768). Single accumulator is simpler; no threat detection adds complexity but may lose tactical precision. Integral's approach is more scalable for larger networks.

### 1.2 Input Buckets

12 king input buckets indexed by king file (0-7) and rank groups:
- Fine granularity on back ranks (buckets 0-4 for white king on ranks 1-2)
- Coarser granularity in middlegame (buckets 5-11)
- Horizontal mirroring when king file >= 4

**Coda comparison**: We have 10 buckets with similar back-rank focus. Integral has one additional middlegame bucket for finer differentiation.

### 1.3 Feature Weights & Biases

- **Feature weights**: `i16[12][2][6][64][1536]` (12.6 MB)
- **Feature biases**: `i16[1536]`
- **L1 weights**: `i8[8][1536][16]` (0.2 MB) — **i8 quantization saves bandwidth**
- **L1 biases**: `float[8][16]` per output bucket
- **L2 weights**: `float[8][16][32]` per output bucket
- **L2 biases**: `float[8][32]`
- **L3 weights**: `float[8][32]` per output bucket
- **L3 bias**: `float[8]`

Network weights embedded at compile time via `include_bytes!()` and transmuted directly (requires recompilation for new nets).

**Coda comparison**: We load nets from files at startup. Integral's compile-time embedding is faster (no I/O) but less flexible.

### 1.4 Output Buckets

Material-based buckets by piece count (`popcount`):
```
Pieces    0-8:  bucket 0
          9-12: bucket 1
         13-16: bucket 2
         17-19: bucket 3
         20-22: bucket 4
         23-25: bucket 5
         26-28: bucket 6
         29-32: bucket 7
```

**Coda comparison**: Identical to ours.

### 1.5 Forward Pass

Standard dense computation:
1. **Accumulator activation**: Clamp each L1 neuron to [0, 255] (unsigned i8 range)
2. **L1 matmul**: `float[16]` = (u8[1536] @ i8[1536][16]) + bias
3. **L1 activation**: ReLU [0, 1] range
4. **L2 matmul**: `float[32]` = (float[16] @ float[16][32]) + bias, ReLU [0, 1]
5. **L3 matmul**: `float` = (float[32] @ float[32]) + bias
6. **Output scale**: `score = L3_output * 200`

No pairwise SCReLU (unlike Reckless). Plain ReLU throughout.

---

## 2. Search

### 2.1 Framework

- Iterative deepening with aspiration windows
- Depth-based LMR table pre-computed at startup
- PVS with null-window searcher at reduced depths
- Stack-based state with offset indexing

### 2.2 Aspiration Windows (`search.cc:93-140`)

```
initial_delta = 10 + (avg_score^2) / 15954
fail_low:  alpha -= window; window *= 1.3983
fail_high: beta += window; window *= 1.3983
```

Asymmetric but same growth on both sides (unlike Reckless which grows fail-high faster). `fail_high_count` limits re-searches to < 3 iterations when not winning.

**Coda comparison**: Our aspiration is simpler (fixed delta=15). Integral's eval-dependent delta is more adaptive.

### 2.3 Static Evaluation Adjustment (`search.cc:220-245`)

Three-source correction:
1. **Material scaling** (`kMaterialScaleBase = 27600`):
   ```
   eval *= (27600 + material_popcount) / 32768
   ```
   Evaluation gets weaker with fewer pieces.

2. **Correction history** — single unified table indexed by pawn hash, minor piece hash, non-pawn hash, continuation keys
3. **50-move-rule decay**:
   ```
   eval *= (220 - halfmove_clock) / 220
   ```

Unlike Reckless (6-source correction), Integral uses a single correction history table that varies by key type.

### 2.4 Improving/Opponent-Worsening Heuristics (`search.cc:696-711`)

**Improving**: `stack->eval > (stack-2)->eval || (stack-2)->eval == kScoreNone && stack->eval > (stack-4)->eval`
**Opponent-worsening**: `stack->eval + (stack-1)->eval > 1`

Used throughout to gate pruning and adjustments.

### 2.5 Reverse Futility Pruning (`search.cc:725-737`)

```
margin = depth * 50 - improving * 94 - opponent_worsening * 25
        + eval_complexity * 12/32 + (stack-1)->history_score / 586
if eval - max(margin, 13) >= beta:
    return lerp(eval, beta, 0.516)
```

Depth-1 bonus removed. Lerp factor 0.516 is non-standard (closer to beta than Reckless's 0.516). Pruning only at depth <= 11.

**Coda comparison**: We use quadratic margins. Integral's linear depth scaling is simpler.

### 2.6 Razoring (`search.cc:741-748`)

```
if eval + 393 * (depth - !improving) < alpha && alpha < 2000:
    return qsearch(alpha, alpha+1)
```

Linear depth scaling (not quadratic). Condition `alpha < 2000` prevents razoring when winning.

### 2.7 Null Move Pruning (`search.cc:752-799`)

**Conditions**:
- eval >= beta
- static_eval >= beta + 146 - 7*depth
- Has non-pawn pieces
- Not at root, not in check

**Reduction**:
```
eval_reduction = min(2, (eval - beta) / 169)
reduction = depth/3 + 4 + eval_reduction + improving
reduction = clamp(reduction, 0, depth)
```

**Verification** at depth >= 16 with `nmp_min_ply` (standard singularity avoidance).

No TT-awareness guards like Reckless (no check for TT lower bound with capture threat).

### 2.8 ProbCut (`search.cc:801-865`)

```
probcut_beta = beta + 213
```

Non-standard probe:
1. Quiescence at `probcut_beta`
2. If score >= `probcut_beta`, search at depth-3 with null window
3. Return `(3*score + beta)/4` on success

No dynamic depth adjustment like Reckless.

### 2.9 Singular Extensions (`search.cc:985-1030`)

**Conditions**: depth >= 5, tt_depth + 3 >= depth, tt_entry is lower bound

```
reduced_depth = kSeDepthReduction * (depth - 1) / 16 = 7 * (depth - 1) / 16
singular_beta = tt_score - 17 * depth / 16

if excluded_score < singular_beta - 8 * (depth/16) - 150*PV:
    extensions = 2 + (is_quiet && excluded_score < singular_beta - 58)
else if excluded_score >= beta:
    return excluded_score (multi-cut)
else if tt_score >= beta:
    extensions = -3
else if cut_node:
    extensions = -2
```

Simpler than Reckless. Static margins (no correction-aware adjustment).

### 2.10 Internal Iterative Reduction (`search.cc:871-874`)

```
if (in_pv || cut_node) && depth >= 4 && !tt_move && depth--
```

Standard IIR.

### 2.11 Late Move Reductions (`search.cc:907-909, 1054-1104`)

**Base formula** (pre-computed table):
```
reduction = log(depth) * log(moves) * base / divisor  [in milliples]
kLmrQuietBase = 0.794, kLmrQuietDiv = 2.071
kLmrTactBase = -0.340, kLmrTactDiv = 2.860
```

**Adjustments** (in milliples; 1024 = 1 ply):
- Non-PV: `+743`
- Was PV: `-1168`
- Cut node: `+2205`
- Gives check: `-942`
- Quiet history: `-history * 522 / 11680`
- Capture history: `-history * 1229 / 11008`
- Not improving: `+949`
- Eval complexity delta > 73: `-778`
- Killer move: `-886`
- Rounding cutoff: `+698`

Fractional reductions clamped to integer: `new_depth - (reduction + 698) / 1024`

**Hindsight adjustment** (`search.cc:1117-1121`):
```
if score > best_score + 30 + 33*new_depth/16:
    do_deeper()
if score < best_score + 5:
    do_shallower()
```

**Quiet update after full-depth re-search** (`search.cc:1135-1141`):
```
if reduction > 0 && is_quiet:
    if score <= alpha: penalty
    if score >= beta: bonus
```

This is **unusual** — bonus/penalty applied to continuation history based on re-search result.

**Coda comparison**: We use similar base logarithmic formula. Integral's adjustment constants are heavily tuned (SPSA). The quiet history update after re-search is novel.

### 2.12 Late Move Pruning (`search.cc:936-937`)

```
lmp_threshold = (5 + depth*depth) / (3 - (improving || eval >= beta))
if moves_seen >= lmp_threshold: break
```

More aggressive when improving. Simple formula.

### 2.13 Futility Pruning (`search.cc:945-954`)

```
margin = 128 + 81 * (lmr_depth) + history_score / 124
if static_eval + margin <= alpha: prune
```

Includes history. Reuses `lmr_depth` (fractional) in calculation.

### 2.14 History Pruning (`search.cc:958-964`)

```
margin = -250 - 2198*depth  (quiet)
         -447 - 1773*depth  (capture)
if history_score <= margin: prune
```

Quadratic depth scaling.

### 2.15 SEE Pruning (`search.cc:968-978`)

```
threshold = -17 * lmr_depth^2  (quiet)
           = -98 * depth - history / 131  (noisy)
```

Static exchange evaluates to threshold.

**Coda comparison**: Formula-heavy; we use simpler constants.

### 2.16 Quiescence Search (`search.cc:248-481`)

**Key features**:
- Stand-pat with lerp blending: `lerp(best, beta, 0.281)`
- Fail-high blending: `lerp(best, beta, 0.591)` when best >= beta
- Futility margin for captures: `eval + 173`
- Limited move generation: stops after good captures unless in check
- Conditional quiet generation: generates quiets only if in_check or tt_move is quiet

**Coda comparison**: Same quiet generation logic as Reckless. We only generate captures.

### 2.17 Killer Moves

Standard 2-move killer table indexed by ply. No counter-move table (unlike Reckless).

---

## 3. Move Ordering

### 3.1 Move Picker Stages

1. **TT move**: Pseudo-legal check
2. **Good noisy**: Captures/promotions with winning SEE
3. **Killer 1**: First killer move (history bonus applied earlier)
4. **Killer 2**: Second killer move
5. **Quiets**: Remaining quiets
6. **Bad noisy**: Losing captures (failed SEE)

No separate counter-move list (integral uses history only).

### 3.2 Capture Scoring (`move_picker.cc:209-214`)

```
score = victim_value * 100 + capture_history_score
```

Standard MVV (material-value-of-victim) plus history.

**SEE threshold with history** (`move_picker.cc:78-79`):
```
threshold = see_threshold - capture_history / 92
```

History reduces SEE stringency for high-history captures.

### 3.3 Quiet Move Scoring (`move_picker.cc:240-246`)

```
score = threat_score + history_score
```

**Threat scoring** (`move_picker.cc:216-239`):
- Queen on rook-attacked square: `+19451`
- Queen moving to rook-attacked square: `-17612`
- Rook on minor-attacked square: `+13177`
- Rook moving to minor-attacked square: `-13758`
- Minor on pawn-attacked square: `+7852`
- Minor moving to pawn-attacked square: `-8475`
- Move gives direct check: `+2048`

**Coda comparison**: Threat bonuses are constant (tuned) values. We compute escape/check/danger heuristically. Integral's approach is more principled.

### 3.4 Quiet History Tables

**Quiet history** (`history/quiet_history.h:14-44`):
- Indexed by `[side][from][to][from_threatened][to_threatened]`
- Dimension: `[2][64][64][2][2]` = 32 KB
- Threat index computed from threat bitboard (`threatened_by[Pawn|Knight|Bishop]`)

**Capture history** (`history/capture_history.h:14-44`):
- Indexed by `[side][from_piece][to][captured_piece]`
- Dimension: `[2][6][64][6]` = 4.6 KB

**Continuation history** (`history/continuation_history.h:14-42`):
- Indexed by `[side][from_piece][to]`
- Accessed at offsets -1, -2, -4, -6 from current ply
- Updates use **total score** from all 4 ply offsets as bonus influence:
  ```
  total = prev[-1] + prev[-2] + prev[-4] + prev[-6]
  bonus = ScaleBonus(total, history_bonus)  // gravity update
  ```

This is **novel**: continuation history updates consider cumulative past context, not just individual history.

### 3.5 Pawn History

Indexed by `[pawn_board_state][from][to]`. Updated similarly to quiet history.

### 3.6 History Bonus Formula (`history/bonus.h:15-31`)

```
bonus = clamp(scale*depth - bias, -max_bonus, max_bonus)
       where scale=168, bias=126, max_bonus=1294
penalty = -clamp(scale*depth + bias, -max_bonus, max_bonus)
        where scale=156

gravity_update(score, bonus):
    return bonus - score * |bonus| / gravity
    where gravity = 11422
```

Linear depth scaling with gravity dampening. Simpler than Reckless's more sophisticated adjustments.

**Coda comparison**: Similar gravity mechanics. Constants are SPSA-tuned.

---

## 4. Time Management

### 4.1 Time Allocation (Fischer increment)

```
soft_scale = 0.024 + 0.042 * (1 - exp(-0.045 * fullmove))
hard_scale = 0.742
soft = soft_scale * main + 0.75 * inc
hard = hard_scale * main + 0.75 * inc
```

Exponential ramp-up: early game ~2.4%, late game ~6.6%, caps at 74.2%.

### 4.2 Soft Stop Decision

Main thread checks `ShouldStop()` every iteration. Factors:
- Iterative depth completed
- Soft time limit reached

No soft-stop multiplier visible in header. Likely simple hard limit approach.

**Coda comparison**: We track stability multipliers (PV, eval, best-move stability). Integral uses simpler approach.

### 4.3 Hard Stop

Node counter checked every ~4000 nodes. Main thread only checks time limit.

---

## 5. Transposition Table

### 5.1 Structure

Standard 3-entries-per-32-byte cluster:
```
Entry: key(u16) + move(u16) + score(i16) + raw_eval(i16) + depth(i8) + flags(u8)
Flags: bound(2) + was_in_pv(1) + age(5 bits) = 32 age cycles
```

Lemire fast modular indexing.

### 5.2 Replacement Policy

Same as Reckless: depth-adjusted quality metric, skip write if entry is better.

---

## 6. Unique/Novel Features

### 6.1 Continuation History with Cumulative Context

Most distinctive: continuation history updates consider **sum of prior 4 ply** (offsets -1, -2, -4, -6) to compute bonus/penalty. This is more sophisticated than independent per-ply updates.

### 6.2 Threat-Aware Quiet Move Bonuses

Bonuses for piece escape/attacked-square avoidance are **tuned constants** rather than heuristics. Cleaner and more predictable.

### 6.3 Quiet History Update After LMR Re-Search

When a move is reduced then re-searched at full depth:
- If score <= alpha: penalty to continuation history
- If score >= beta: bonus to continuation history

This provides feedback from deeper search back to history.

### 6.4 Simple, Tuned Approach

Integral is heavily SPSA-tuned (100k-game LTC completed recently, git log shows 8babfda9). No complex hand-crafted heuristics like Reckless's eval-dependent aspiration delta or alpha-raise tracking. The engine is simpler but better-tuned.

### 6.5 Material-Scaled Eval

Evaluation weakens with fewer pieces, aiding tabletop play.

---

## 7. Comparison with Coda v9

### NNUE Architecture

| Aspect | Coda v9 | Integral |
|--------|---------|----------|
| Input features | PST 768 + Threat 66864 | PST 768 (factorized) |
| L1 size | 768 + 768 | 1536 (single) |
| L1 weights | i16 | i8 |
| L1 activation | SCReLU pairwise | Plain ReLU |
| L2/L3 | float | float |
| Network weights | Runtime loaded | Compile-time embedded |
| Buckets | 10 input, 8 output | 12 input, 8 output |

**Assessment**: Integral trades off threat information for a larger single accumulator. Simpler architecture; easier to train/tune. Coda's dual accumulator should give better tactical awareness (estimated +30-50 Elo from Reckless review).

### Search

| Aspect | Coda v9 | Integral |
|--------|---------|----------|
| Aspiration | Fixed delta=15 | Eval-dependent + growth |
| Correction history | Single (pawn hash) | Single (pawn + minor + non-pawn keys) |
| Singular extensions | Disabled | Enabled (static margins) |
| IIR | Standard | Standard |
| LMR base | Logarithmic | Logarithmic (tuned) |
| LMR adjustments | ~10 modifiers | ~10 modifiers (heavily tuned) |
| Killer moves | 2-move table | 2-move table |
| Counter-move | Separate table | None (history only) |
| Time mgmt | Sophisticated multipliers | Simple hard limit |

**Assessment**: Integral is simpler in design, heavier in constant tuning. Both use similar search structure; Integral's simpler time management may leave rating on table in bullet/blitz.

### Move Ordering & History

| Aspect | Coda v9 | Integral |
|--------|---------|----------|
| Capture scoring | MVV + history | MVV + history |
| Quiet scoring | History + cont-hist | History + 4-ply cont-hist |
| Threat bonuses | Heuristic | Tuned constants |
| History gravity | Standard | Standard (gravity=11422) |
| Continuation hist | 2-ply lookback | 4-ply lookback (cumulative) |
| Pawn history | Yes | Yes |

**Assessment**: Integral's 4-ply cumulative continuation history is more sophisticated. Threat bonuses more disciplined. Overall move ordering probably comparable.

---

## 8. Summary: Ideas to Port (Prioritized)

### High Priority (Moderate Effort)

1. **Continuation history with cumulative context (4-ply, offsets -1, -2, -4, -6)**
   - Coda currently uses 2-ply. This should be straightforward to extend.
   - Expected: +5-15 Elo

2. **Material-scaled evaluation**
   ```
   eval *= (27600 + material_popcount) / 32768
   ```
   - Simple to add; helps endgame transitions.
   - Expected: +2-5 Elo

3. **Eval-dependent aspiration window delta**
   ```
   delta = 10 + avg_score^2 / 15954
   ```
   - Adapts window size to position volatility.
   - Expected: +2-5 Elo

4. **Quiet history update after LMR re-search**
   - Provides feedback from deeper search to history.
   - Expected: +3-8 Elo (needs careful tuning)

### Medium Priority (Higher Effort)

5. **Larger single accumulator (1536 neurons)**
   - Replace threat accumulator with larger PST accumulator.
   - Requires retraining in Bullet.
   - Expected: Neutral (we gain simplicity, lose threat precision; should roughly offset).

6. **Static threat-based move bonuses**
   - Replace escape/check heuristics with tuned constants.
   - Need fresh SPSA tuning.
   - Expected: +2-3 Elo (cleaner code, not higher ELO)

### Lower Priority

7. **IIR simplification**
   - Integral uses standard form; worth checking if we can simplify our variant.
   - Expected: Neutral (refactoring only).

8. **Pawn history extension**
   - Both engines have it; no change needed.

---

## Final Notes

Integral is a **finely tuned, simple-by-design** engine. It lacks Reckless's sophistication (threat accumulators, pairwise SCReLU) and Coda's threat information, but compensates with:
- Excellent SPSA tuning (recent 100k-game LTC)
- Larger, simpler accumulator
- 4-ply cumulative continuation history
- Clean threat-bonus constants

The engine demonstrates that **deep tuning beats complex heuristics** at modern strengths. Recommended ports are the 4-ply continuation history (significant) and material-scaled eval (easy win). Singular extensions are enabled here but we disabled them for good reason; not recommended without re-tuning.

