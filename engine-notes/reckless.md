# Reckless Chess Engine - Technical Review

Source: `~/chess/engines/Reckless/`
Version: 0.10.0-dev
Language: Rust (edition 2024)
NNUE: Dual-accumulator (768x10 PST + 66864 threats -> 2x768 -> 16 -> 32 -> 1x8) with pairwise SCReLU
Rating: #1 in local gauntlet (among non-SF)

Last reviewed: 2026-03-29
Last v9-refreshed: 2026-04-19

---

## v9 Refresh (2026-04-19)

Coda has converged architecturally with Reckless since the original review:
- **v9 architecture matches Reckless**: HalfKA PST + threats (66864) → pairwise → hidden layers → output
- **Discovered-attack bonus** (Coda-unique, #502 landed +52 Elo) — leverages x-ray threat enumeration that Reckless doesn't compute
- **LMP king-pressure NMP gate, ProbCut king-zone gate, LMR king-pressure, futility-defenses** — all landed as novel signal × context experiments (#466/#481/#482/#484)
- **2b slider-iteration rewrite** (NPS, +10 Elo #478) — reduced threat-delta generation cost
- **Target-feature AVX2 codegen win** (~+5% NPS) on same ISA as Reckless

### What Coda still does worse/differently than Reckless

- **First-move-cut rate**: Coda v9 72.2% vs estimate ~80-85% for Reckless. Same move-ordering code as v5 (which gets 82.3% with linear eval). Gap is the eval-surface-is-flatter tax.
- **Attacker-type-stratified escape bonuses** — Reckless uses `[0, 8k, 8k, 14k, 20k, 0]` per moving-piece AND gates by `threatened[pt]` (the threat-class-specific bitboard: pawn/minor/rook). Coda only has pawn-attacker scaling. Tested unstratified; A1a+A1c with SPSA retune (#501) in flight.
- **Onto-threatened penalty** — Reckless: `-8000 * threatened[pt].contains(to)`. Coda's unstratified version H0'd (#465); stratified+SEE-gated not yet tested.
- **Offense bonus** — Reckless: `+6000 * offense[pt].contains(to)` — quiet moves that land on squares attacking enemy pieces *safely*. Uniquely Reckless; not in Obsidian, Plenty, or Coda. **Untested.**
- **Rook into king-ring-ortho** — Reckless: `+5000 * (pt == Rook && king_ring_ortho.contains(to))`. Also Reckless-unique. Untested.
- **NPS**: Reckless ~2× Coda on same hardware. Despite Coda's recent +5% from target_feature and +10% aggregate from 2b/target-feature/psq-fuse wins, the base const-generic/static-PARAMETERS refactor still outstanding (see `v9_nps_findings_2026-04-18.md`).

### What Coda does better than Reckless

- **X-ray threat enumeration** — Reckless's threat features are direct-only. Coda's include x-ray (+110 Elo gain in training). Enables discovered-attack-bonus.
- **4D threat-indexed history** — matches Reckless's structure but more granular via from_threatened+to_threatened.
- **Correction history**: Coda has 5 sources (pawn, wNP, bNP, minor, major, cont) vs Reckless's 5 (no minor/major split). Marginal structural advantage.

### Untested Reckless ideas worth flagging

1. **Offense bonus** — computation is non-trivial (needs per-piece-type "safe attacker" bitboards) but zero per-node cost once computed at node entry. Pre-search signal: pure move-ordering.
2. **Rook king-ring-ortho** — cheap (1 magic lookup + mask). Specific enough that might be noise; could bundle with other king-ring ideas.

Both untested in Coda. Could be valuable especially given Coda's flatter eval needs more pre-search discrimination (see `move_ordering_understanding_2026-04-19.md`).

---

## 1. NNUE Architecture

### Network Topology: (PST[768x10] + Threats[66864]) -> 2x768 -> 16 -> 32 -> 1x8

The most sophisticated NNUE architecture in our test suite. Uses **two separate accumulators** (PST + threats) that are summed at activation time, with a 3-layer hidden network.

**Constants** (`nnue.rs:51-68`):
- `NETWORK_SCALE = 380` (final output multiplier)
- `INPUT_BUCKETS = 10`, `OUTPUT_BUCKETS = 8`
- `L1_SIZE = 768` (accumulator width, each perspective)
- `L2_SIZE = 16` (first hidden layer)
- `L3_SIZE = 32` (second hidden layer)
- `FT_QUANT = 255`, `L1_QUANT = 64`
- `FT_SHIFT = 9` (used in pairwise activation)
- `DEQUANT_MULTIPLIER = (1 << 9) / (255 * 255 * 64) = 0.0001225...`

**Coda comparison**: We use 768 FT with 16->32->1x8 hidden layers too (v7 architecture), but we have a single PST accumulator. Reckless has a **separate threat accumulator** which is the key differentiator. Our L1_SIZE=1024 vs their 768, but they get more information from threats.

### 1.1 PST Accumulator (`nnue/accumulator.rs`)

Standard HalfKA piece-square feature transformer:
- 10 king input buckets with fine back-rank granularity (`nnue.rs:71-80`)
- Horizontally mirrored when king file >= 4
- Feature index: `bucket * 768 + 384 * (color != pov) + 64 * piece_type + (square ^ flip)` (`accumulator.rs:226-233`)
- Weights: `i16[INPUT_BUCKETS * 768][L1_SIZE]` (standard)
- Biases: `i16[L1_SIZE]`

**Finny Table cache** (`AccumulatorCache`): `[pov][flip][bucket]` cache entries store accumulated values plus piece/color bitboards. On king bucket change, diffs cached vs current bitboards (`accumulator.rs:54-93`). This is the same approach we use.

**Incremental updates** (`accumulator.rs:95-188`): Handles add1_sub1, add1_sub2, add2_sub2 patterns with direct SIMD loops stepping by `simd::I16_LANES`.

**Coda comparison**: Nearly identical to our Finny table approach. Same king bucket mirroring. Our bucket layout has 16 entries vs their 10 -- theirs is more compact with back-rank differentiation only.

### 1.2 Threat Accumulator (`nnue/accumulator.rs:235-403`, `nnue/threats.rs`, `nnue/threats/vectorized.rs`)

**This is the novel feature.** A second accumulator that encodes piece-on-piece attack relationships.

**Threat features**: For every piece P attacking square S where piece Q sits, there is a threat feature `(P, from, Q, to)`. The feature index is computed by `threat_index()` (`threats.rs:112-130`) which encodes:
- Attacker piece (color-relative to POV)
- Attacked piece (color-relative to POV)
- Attacker square (mirrored if king file >= 4)
- Attack square (mirrored)
- A piece-pair interaction map that determines which piece combinations are tracked

**Piece interaction map** (`threats.rs:44-49`):
```
              P  N  B  R  Q  K
Pawn       [  0, 1,-1, 2,-1,-1]
Knight     [  0, 1, 2, 3, 4,-1]
Bishop     [  0, 1, 2, 3,-1,-1]
Rook       [  0, 1, 2, 3,-1,-1]
Queen      [  0, 1, 2, 3, 4,-1]
King       [  0, 1, 2, 3,-1,-1]
```
-1 means excluded (not tracked). Semi-excluded pairs (same piece type, enemy, or same non-pawn) use directional deduplication.

Total threat feature count: **66,864** features (`ft_threat_weights: i8[66864][768]`).

**Weights are i8** (not i16 like PST). This makes sense -- threats are more numerous but individually less important, and i8 halves the memory bandwidth for updates.

**SIMD threat detection** (`threats/vectorized/avx2.rs`): This is extraordinarily clever. Uses AVX2 byte shuffles to:
1. **Ray permutation**: Maps the 64-square board into 64-byte ray representation (8 directions x 8 squares per direction) centered on the focus piece
2. **Board-to-rays**: Shuffles the mailbox board through the permutation to get pieces along each ray
3. **Closest-on-rays**: Finds the nearest piece on each ray using bit tricks (`o ^ (o - 0x030303...)`)
4. **Attack classification**: Uses bitmask tables to identify which ray-occupants can attack the focus square
5. **X-ray threats**: Detects sliding attackers behind the closest piece

The whole thing runs in ~10 AVX2 instructions per operation. The permutation table (`RAY_PERMUTATIONS`) is a compile-time const using 0x88 coordinate math.

**Incremental threat updates** (`threats/vectorized.rs:109-182`):
- `push_threats_on_change`: When a piece is added/removed, compute all its attacks and all attacks on it
- `push_threats_on_move`: When a piece moves, compute threats at source (remove) and destination (add), plus x-ray changes
- `push_threats_on_mutate`: For promotions -- remove old piece threats, add new piece threats

These push `ThreatDelta` entries (packed u32: piece, from, attacked, to, add flag) onto `ThreatAccumulator.delta`. The actual accumulator update is deferred and batched.

**Coda comparison**: We have nothing like this. This is likely worth 30-50 Elo and is the primary reason Reckless is stronger. The threat information gives the net direct access to tactical features (pins, forks, threats) that our PST-only net must learn implicitly. Adopting this would require:
1. Training a new net architecture in Bullet with threat features
2. Implementing the ray-based threat detection (complex but well-defined)
3. Adding the second accumulator stack

### 1.3 Forward Pass / Output Transformer (`nnue/forward/vectorized.rs`)

**Pairwise SCReLU activation** (`activate_ft`, lines 10-53):
```
for each perspective (stm, nstm):
    for each pair (i, i + L1_SIZE/2):
        left = clamp(pst[i] + threat[i], 0, 255)
        right = min(pst[i+384] + threat[i+384], 255)
        output[i] = (left * right) >> FT_SHIFT   // pairwise product
```
The PST and threat accumulators are **summed** before activation. Left half is clamped [0, 255], right half is only min-clamped. The pairwise product uses `mulhi_i16` with a left-shift for precision. Output is u8.

**Sparse L1 matmul** (`propagate_l1`, lines 55-108):
- FT output is u8[768]. Treated as i32[192] (4 bytes per chunk).
- **NNZ detection** (`find_nnz`): Finds non-zero 4-byte chunks. Uses `_mm256_cmpgt_epi32` to get a bitmask, then a lookup table (`nnz_table`) to expand to u16 indexes. AVX-512 VBMI path uses `_mm512_maskz_compress_epi16` for even faster NNZ detection.
- **Sparse matmul**: Only multiplies non-zero chunks against L1 weights. Uses `dpbusd` (dot-product of u8 * i8 -> i32) with pairs of NNZ entries processed together (`double_dpbusd`).
- L1 weights: `i8[OUTPUT_BUCKETS][L1_SIZE / 4 * L2_SIZE * 4]` -- stored in a layout optimized for the sparse access pattern.
- L1 biases: `f32[OUTPUT_BUCKETS][L2_SIZE]`
- After integer matmul, dequantize to f32, add biases, clamp [0, 1].

**L2 and L3** (`propagate_l2`, `propagate_l3`, lines 110-153):
- L2: `f32[OUTPUT_BUCKETS][L2_SIZE][L3_SIZE]` weights, `f32` biases. Standard matmul with FMA, clamp [0, 1].
- L3: `f32[OUTPUT_BUCKETS][L3_SIZE]` weights, `f32` biases. Dot product -> scalar + bias.
- Final output: `l3_out * 380`

**Coda comparison**: Key differences from our v7:
1. They use **pairwise SCReLU** (left * right) not plain SCReLU (x^2). This is more expressive.
2. **Sparse L1 matmul** with NNZ detection -- we do dense. With SCReLU many activations are zero, so sparsity is high. This is a major NPS optimization.
3. **Float L2/L3** -- we should do this too, avoids integer quantization artifacts.
4. The `double_dpbusd` trick processes two NNZ entries per iteration, improving instruction-level parallelism.

### 1.4 Output Buckets

Material-based output buckets (`nnue.rs:83-92`):
```
Popcount  0-8: bucket 0
          9-12: bucket 1
         13-16: bucket 2
         17-19: bucket 3
         20-22: bucket 4
         23-25: bucket 5
         26-28: bucket 6
         29-32: bucket 7
```
8 output buckets indexed by total piece count.

**Coda comparison**: Same as our output bucket scheme.

---

## 2. Search

### 2.1 Framework (`search.rs`)

- Iterative deepening with aspiration windows
- Generic `NodeType` trait: `Root`, `PV`, `NonPV` (compile-time specialization)
- Cut-node tracking passed through search
- `Stack` with offset indexing (`stack[ply - N]` for N up to 8, sentinel entries)

### 2.2 Aspiration Windows (`search.rs:101-176`)

```rust
delta = 13 + average^2 / 23660    // eval-dependent initial delta
```
On fail-low:
- `beta = (3*alpha + beta) / 4` (asymmetric tightening)
- `alpha = (score - delta).max(-INF)`
- `reduction = 0`
- `delta += 27 * delta / 128` (~21% growth)

On fail-high:
- `alpha = (beta - delta).max(alpha)`
- `beta = (score + delta).min(INF)`
- `reduction += 1` (reduces search depth on repeated fail-highs)
- `delta += 63 * delta / 128` (~49% growth, much faster than fail-low)

**Optimism** (`search.rs:123-127`): Based on best score across threads:
```rust
optimism[stm] = 169 * best_avg / (best_avg.abs() + 187)
optimism[!stm] = -optimism[stm]
```
Used in `correct_eval` to adjust eval.

**Coda comparison**: Our aspiration is simpler (fixed delta=15). The eval-dependent delta and asymmetric growth rates are interesting. The reduction on fail-high is novel -- reduces wasted effort on clearly winning positions.

### 2.3 TT Early Cutoff (`search.rs:339-361`)

Standard TT cutoff with several twists:
- `tt_depth > depth - (tt_score < beta)` -- allows shallower TT entries when score is below beta
- Cut-node and non-cut-node have different depth requirements: `(!cut_node || depth > 5)` for upper, `(cut_node || depth > 5)` for lower
- On TT cutoff with quiet beta-cutoff and low parent move count (< 4): updates quiet history and continuation history with the TT move
- Halfmove clock >= 90 disables TT cutoffs (near 50-move rule)

**Coda comparison**: The cut-node-aware TT cutoff depth is interesting -- allows more aggressive cutoffs when the node type agrees with the bound direction. The TT move history update on cutoff is a nice touch.

### 2.4 Eval Correction (`search.rs:1291-1340`, `evaluation.rs`)

**Correction history** combines 6 sources (`eval_correction`, line 1291):
1. Pawn correction (pawn hash key)
2. Minor piece correction (minor hash key)
3. Non-pawn correction for White (non-pawn key)
4. Non-pawn correction for Black (non-pawn key)
5. Continuation correction at ply-2
6. Continuation correction at ply-4

All divided by 77 to scale.

**Shared correction history** (`thread.rs:90-94`): Pawn, minor, and non-pawn correction histories are **shared across threads** (via atomics on each entry). This is unusual -- most engines keep correction history per-thread. The sharing means all threads contribute to and benefit from correction data.

**Continuation correction history** (`history.rs:160-188`): Indexed by `[in_check][capture][piece][to]` -> `PieceToHistory<i16>`. This is a correction history that uses continuation-style indexing -- it predicts eval error based on the last move made.

**Eval calculation** (`evaluation.rs:3-13`):
```rust
eval = (raw_eval * (21061 + material) + optimism * (1519 + material)) / 26556
eval = eval * (200 - halfmove_clock) / 200
eval += correction_value
```
- Material-scaled eval (stronger eval with more material)
- Halfmove clock decay (eval trends toward 0 near 50-move draw)
- Optimism injection based on thread-aggregate score estimates

**Coda comparison**: We have pawn correction history. Reckless adds minor, non-pawn, and continuation correction -- 6 sources vs our 1. The shared correction history across threads is particularly interesting for Lazy SMP. The material-scaled eval is also novel.

### 2.5 Eval Estimation / TT Tightening (`search.rs:424-456`)

Uses TT score to tighten eval when bound direction agrees:
```
if tt_bound == Upper && tt_score < eval: estimated_score = tt_score
if tt_bound == Lower && tt_score > eval: estimated_score = tt_score
if Exact: estimated_score = tt_score
```
`estimated_score` is used for pruning decisions. `eval` stays as the NNUE+correction value.

Special handling for in-check: uses TT score for eval when bound allows cutoff.

### 2.6 Quiet Move Ordering via Eval Difference (`search.rs:460-465`)

```rust
value = 819 * (-(eval + prev_eval)) / 128
bonus = value.clamp(-124, 312)
```
Updates quiet history for the previous move based on how well it performed (measured by eval difference). This is a form of "malus retrospectively" -- if the position after a quiet move evaluated poorly for the opponent, that quiet move gets a bonus.

**Coda comparison**: We don't have this. It's an eval-agnostic improvement to move ordering that should transfer well cross-engine.

### 2.7 Hindsight Reductions (`search.rs:468-483`)

Two hindsight conditions:

1. **Depth increase** when `prev_reduction >= 2247` and `eval + prev_eval < 0`:
   - If the previous move was heavily reduced AND the position is bad for both sides, give an extra ply.

2. **Depth decrease** when `prev_reduction > 0` and `eval + prev_eval > 59`:
   - Non-tt-pv nodes where the previous move was reduced and the position is quiet for both sides: reduce by 1.

**Coda comparison**: Similar concept to our hindsight reduction. Their threshold of 59 for the quiet condition is tighter than our 200. The depth-increase case (condition 1) is novel -- we don't extend based on mutual badness.

### 2.8 Razoring (`search.rs:502-509`)

```rust
if estimated_score < alpha - 299 - 252 * depth^2
   && alpha < 2048
   && !tt_move.is_quiet():
    return qsearch(alpha, beta)
```
Quadratic depth scaling. The condition `!tt_move.is_quiet()` means razoring only fires when the TT move is noisy (or no TT move) -- if a quiet TT move exists, we trust the position may have tactical complexity.

### 2.9 Reverse Futility Pruning (`search.rs:512-525`)

```rust
margin = 1125 * depth^2 / 128 + 26 * depth
       - 77 * improving
       + 519 * |correction_value| / 1024
       + 32 * (depth == 1)
       - 64 * (no_threats_in_check)
```
Returns `beta + (score - beta) / 3` (blended, not raw beta).

Conditions: non-tt-pv, not excluded, valid estimated_score >= beta, not losing beta, not winning estimate.

**Coda comparison**: The correction-value-aware margin is interesting -- high correction means the eval is less reliable, so widen the margin. The depth-1 bonus and threat-awareness are nice tweaks.

### 2.10 Null Move Pruning (`search.rs:528-582`)

Conditions:
- Cut node only
- Not in check, not excluded, not potential singularity
- `estimated_score >= beta` and `estimated_score >= eval`
- `eval >= beta - 9*depth + 126*tt_pv - 128*improvement/1024 + 286 - 20*(cutoff_count < 2)`
- Has non-pawns, not losing beta
- **Not when TT has a lower-bound capture of knight+ value** (avoids NMP when there's a known tactical threat)

Reduction: `(5154 + 271*depth + 535*clamp(estimated_score - beta, 0, 1073)/128) / 1024`

Verification at depth >= 16 with `nmp_min_ply` mechanism.

**Coda comparison**: The cut-node-only NMP is notable -- many engines allow NMP on non-cut nodes too. The "no NMP when TT shows capture threat" condition is novel and sensible.

### 2.11 ProbCut (`search.rs:585-637`)

```rust
probcut_beta = beta + 269 - 72 * improving
```
Cut-node only. Requires non-quiet TT move or no valid TT score.

**Dynamic depth adjustment**: After qsearch confirms score >= probcut_beta, calculates a reduced depth based on how much the score exceeds the threshold:
```rust
probcut_depth = (base_depth - (score - probcut_beta) / 295).clamp(0, base_depth)
```
If the margin is huge, search at reduced depth. If the reduced search fails, re-search at full `base_depth`.

Returns `(3*score + beta) / 4` (blended).

### 2.12 Singular Extensions (`search.rs:640-678`)

Conditions: `depth >= 5 + tt_pv`, `tt_depth >= depth - 3`, lower bound TT, valid non-decisive TT score.

```rust
singular_beta = tt_score - depth - depth * (tt_pv && !PV)
singular_depth = (depth - 1) / 2
```

**Triple extensions** (up to +3):
```rust
extension = 1
extension += (score < singular_beta - double_margin)    // double extend
extension += (score < singular_beta - triple_margin)    // triple extend

double_margin = 200*PV - 16*tt_quiet - 16*|correction|/128
triple_margin = 288*PV - 16*tt_quiet - 16*|correction|/128 + 32
```

**Multi-cut**: If singular search score >= beta, return `(2*score + beta) / 3`.

**Negative extensions**: -2 if TT score >= beta or cut_node.

**Recapture extension**: +1 in PV when TT move is noisy and targets recapture square.

**Coda comparison**: We removed singular extensions after cross-engine testing showed they were harmful. Reckless keeps them -- their stronger NNUE may make singular extensions viable because the eval is more accurate. The triple extension with correction-aware margins is sophisticated.

### 2.13 Late Move Pruning (`search.rs:718-725`)

```rust
adjust = improvement.clamp(-100, 218)
factor0 = 2515 + 130 * adjust / 16
factor1 = 946 + 79 * adjust / 16
lmp_threshold = (factor0 + factor1 * depth^2) / 1024
```
Improvement-scaled LMP. More aggressive when improving (higher threshold = more moves allowed).

### 2.14 Futility Pruning (`search.rs:728-736`)

```rust
futility_value = eval + 88*depth + 63*history/1024 + 88*(eval >= beta) - 114
```
For quiet moves at depth < 14, not in check, not giving check. History-aware: high-history moves get more leeway.

Also updates `best_score` to `futility_value` if applicable (prevents returning -INF when everything is pruned).

### 2.15 Bad Noisy Futility (`search.rs:739-752`)

```rust
noisy_futility = eval + 71*depth + 69*history/1024 + 81*piece_value/1024 + 25
```
For bad noisy moves (failed SEE) at depth < 12. Includes captured piece value in the margin.

### 2.16 SEE Pruning (`search.rs:754-763`)

Quiet threshold: `(-16*depth^2 + 52*depth - 21*history/1024 + 22).min(0)`
Noisy threshold: `(-8*depth^2 - 36*depth - 32*history/1024 + 11).min(0)`

History-aware: bad-history moves get tighter SEE requirements.

### 2.17 Late Move Reductions (`search.rs:775-849`)

Base formula (logarithmic):
```rust
reduction = 250 * log2(move_count) * log2(depth)    // in milliplex (1024 = 1 ply)
```

Adjustments:
- `-65 * move_count` (more reduction for late moves)
- `-3183 * |correction_value| / 1024` (less reduction when correction is high)
- `+1300 * alpha_raises` (more reduction after alpha raises -- the position is well-explored)
- Quiet: `+1972` base, `-154 * history / 1024`
- Noisy: `+1452` base, `-109 * history / 1024`
- PV: `-411 - 421 * (beta-alpha) / root_delta`
- tt_pv: `-371`, `-656` if TT score > alpha, `-824` if TT depth >= depth
- Recapture: `-910`
- Cut-node (non-tt-pv): `+1762`, `+2116` if no TT move
- Not improving: `+(438 - 279*improvement/128).min(1288)`
- In check after move: `-966`
- High cutoff count at ply+1: `+1604`
- TT score < alpha: `+600`
- Parent had larger reduction: `+128` (non-PV)

**LMR extension**: If reduction < -3072 and move_count <= 3, allow new_depth to be extended by 1.

**Depth adjustment after LMR re-search**:
```rust
new_depth += (score > best_score + 50 + 4*reduced_depth)   // extend if much better
new_depth -= (score < best_score + 5 + reduced_depth)       // reduce if barely better
```

**Full-depth search** (depth < 2 or move_count < 2) also has reduction logic:
```rust
reduction = 238 * log2(move_count) * log2(depth)   // slightly less aggressive base
// Similar adjustments but different coefficients
```
Applies discrete reductions: `new_depth - (reduction >= 3072) - (reduction >= 5687 && new_depth >= 3)`.

**Coda comparison**: The `alpha_raises` tracking is novel -- it means positions where many moves raise alpha get more reduction on subsequent moves (the best move has been found, no need to search deeply). The correction-value awareness in LMR is also interesting. The discrete reduction for full-depth search is simpler than continuous -- worth testing.

### 2.18 Quiescence Search (`search.rs:1098-1289`)

Key features:
- Upcoming repetition check (prunes early draws)
- TT cutoff (non-decisive scores only in PV)
- **Stand pat blending**: `beta + (best_score - beta) / 3` (dampens inflated stand-pat)
- **Quiet moves in qsearch**: Conditionally generated based on `(in_check && is_loss) || (tt_move.is_quiet() && tt_bound != Upper)`
- QS LMP: break after 3 moves if not giving check
- QS SEE: `(alpha - eval) / 8 - 100`
- **Exit blending**: `(best_score + beta) / 2` when best_score >= beta

**Coda comparison**: The conditional quiet generation in qsearch is clever -- if the TT suggests a quiet move is good, generate quiets. Our qsearch only does captures.

### 2.19 Beta Blending (`search.rs:1070-1073`)

```rust
if best_score >= beta && !decisive(best_score) && !decisive(alpha):
    weight = depth.min(8)
    best_score = (best_score * weight + beta) / (weight + 1)
```
Blends the score toward beta to dampen inflated cutoff scores. Depth-weighted: deeper searches get more trust.

**Coda comparison**: We have fail-high score blending. Their depth-weighting is a nice refinement.

### 2.20 History Updates (`search.rs:991-1066`)

**Bonus/malus scaling**:
```rust
noisy_bonus = (106*depth).min(808) - 54 - 80*cut_node
noisy_malus = (164*depth).min(1329) - 52 - 23*noisy_count
quiet_bonus = (172*depth).min(1459) - 78 - 54*cut_node
quiet_malus = (144*depth).min(1064) - 45 - 39*quiet_count
cont_bonus  = (108*depth).min(977) - 67 - 52*cut_node
cont_malus  = (352*depth).min(868) - 47 - 19*quiet_count
```
Cut-node reduces bonuses (less reliable). Number of tried moves reduces maluses (shared blame).

**Countermove bonus on fail-low** (`search.rs:1034-1066`):
When bound == Upper (fail-low), the **previous move** (parent's move) gets a bonus:
```rust
factor = 95
    + 156*(depth > 5)
    + 215*(prev_move_count > 8)
    + 113*(pcm_move == tt_move)
    + 156*(!in_check && best_score <= eval.min(raw_eval) - 96)
    + 317*(valid_prev_eval && best_score <= -prev_eval - 120)
scaled_bonus = factor * (153*depth - 34).min(2474) / 128
```
This is a **multi-factor countermove heuristic** -- the bonus is higher when the position after the parent move is deeply searched, the parent was the TT move, the current eval is much worse than expected, etc.

**Coda comparison**: The multi-factor countermove bonus is very sophisticated. We have a simple counter-move table. This graduated approach based on context should be much more accurate.

### 2.21 Correction History Updates (`search.rs:1084-1090`, `1312-1340`)

Updated when: not in check, not noisy best move, and bound direction doesn't agree with eval (i.e., the eval was wrong about which side of the score the best move falls on):
```rust
!(bound == Upper && best_score >= eval)
!(bound == Lower && best_score <= eval)
```
Bonus: `(142 * depth * diff / 128).clamp(-4923, 3072)` where diff = best_score - eval.

Also updates continuation correction at ply-2 and ply-4.

---

## 3. Move Ordering (`movepick.rs`, `history.rs`)

### 3.1 Move Picker Stages

1. **HashMove**: TT move (pseudo-legality checked)
2. **GenerateNoisy**: Generate all captures/promotions
3. **GoodNoisy**: Score by `16 * captured_value + noisy_history`. SEE threshold: `-score/46 + 109` (history-dependent). Failing moves go to bad_noisy.
4. **GenerateQuiet**: Generate quiet moves (skipped if `skip_quiets`)
5. **Quiet**: Score by quiet_history + conthist[1] + conthist[2] + conthist[4] + conthist[6], plus escape/check/danger bonuses
6. **BadNoisy**: Failed-SEE captures

**No killers, no counter-move table**: History alone handles move ordering.

**Coda comparison**: We use killers and counter-move table. Reckless relies entirely on history -- their richer history (threat-aware, factorized) makes explicit killers unnecessary.

### 3.2 Quiet Move Scoring (`movepick.rs:185-232`)

```rust
score = quiet_history(threats, stm, mv)
      + conthist(ply-1, mv)
      + conthist(ply-2, mv)
      + conthist(ply-4, mv)
      + conthist(ply-6, mv)
```

**Escape bonus**: If the piece is currently threatened by a lower-value attacker:
- Queen threatened by minor/rook: +20000
- Rook threatened by minor: +14000
- Minor threatened by pawn: +8000

**Check bonus**: +10000 for moves to checking squares.

**Danger malus**: -10000 for queen moves to squares attacked by minors.

Threat detection uses pre-computed per-piece-type threat bitboards stored in board state.

**Coda comparison**: The escape/check/danger bonuses in move scoring are nice. We compute threats but don't use them this way in move ordering.

### 3.3 History Table Design (`history.rs`)

**QuietHistory** (`history.rs:17-69`):
- Indexed by `[stm][from][to]`
- Each entry has a **factorizer** (unconditional, MAX=1852) and **buckets** `[from_threatened][to_threatened]` (MAX=6324)
- Score = factorizer + bucket
- This is "threat-aware" history: same move gets different scores depending on threat context

**NoisyHistory** (`history.rs:71-120`):
- Indexed by `[piece][to]`
- Factorizer (MAX=4524) + buckets `[captured_piece_type][to_threatened]` (MAX=7826)
- 7 capture types x 2 threat states

**ContinuationHistory** (`history.rs:190-218`):
- `[in_check][capture][piece][to]` -> `PieceToHistory<i16>` (MAX=15168)
- Accessed at offsets 1, 2, 4, 6 from current ply
- Indexed by `[in_check][capture]` -- separate tables for check/non-check, capture/quiet contexts

**ContinuationCorrectionHistory** (`history.rs:160-188`):
- Same indexing as continuation history but for correction values (MAX=16282)
- Accessed at ply-2 and ply-4

**Gravity formula** (`history.rs:12-15`):
```rust
*entry += (bonus - bonus.abs() * (*entry) / MAX) as i16
```
Standard gravity with per-table MAX values.

**Coda comparison**: The factorizer + bucket design is novel. Factorizer captures move quality unconditionally, buckets capture threat-dependent quality. Our history is a single value per entry. The separate MAX values per component allow independent tuning. The continuation history with 4 plies of lookback (1,2,4,6) is more than our 2.

### 3.4 Noisy Move Scoring in Check (`movepick.rs:174-183`)

When in check, noisy moves are scored by inverse piece value:
```rust
score = 10000 - 1000 * piece_type
```
This prioritizes capturing/moving with pawns first (least valuable attacker).

---

## 4. SIMD / Performance

### 4.1 SIMD Tiers

Three tiers compiled via target features (not runtime detected):
1. **AVX-512** (`nnue/simd/avx512.rs`): Full 512-bit operations
2. **AVX2** (`nnue/simd/avx2.rs`): 256-bit operations, FMA
3. **NEON** (`nnue/simd/neon.rs`): ARM 128-bit
4. **Scalar** fallback

**Coda comparison**: We use runtime AVX2 detection. Reckless compiles different binaries. Compile-time is faster (no function pointer overhead) but less portable.

### 4.2 Key SIMD Tricks

**dpbusd** (`avx2.rs:75-79`): Dot-product of unsigned bytes * signed bytes -> i32. Uses `_mm256_maddubs_epi16` + `_mm256_madd_epi16`. This is the core L1 matmul operation.

**double_dpbusd** (`avx2.rs:81-86`): Processes two pairs at once by adding the pairwise results before the final madd. Saves one `_mm256_add_epi32`.

**NNZ bitmask** (`avx2.rs:104-107`): Converts u8 activations to a bitmask of non-zero 4-byte chunks using `_mm256_cmpgt_epi32` + `_mm256_movemask_ps`.

**Pairwise activation** (`forward/vectorized.rs:10-53`): Uses `_mm256_mulhi_epi16` (multiply high 16-bit) with a left-shift to get the top bits of the pairwise product. Avoids expensive 32-bit multiply.

**TT prefetch** (`transposition.rs:229-242`): `_mm_prefetch::<_MM_HINT_T1>` (prefetch to L2 cache) on the TT cluster before making a move.

### 4.3 Threat Accumulator SIMD

The threat accumulator update (`accumulator.rs:311-391`) uses a register-blocking strategy:
- 8 registers (or L1_SIZE/I16_LANES for AVX-512)
- Interleaved add/sub processing to maximize throughput
- i8 threat weights are converted to i16 via `convert_i8_i16` before SIMD add

### 4.4 Memory Layout

Network weights are embedded at compile time via `include_bytes!(env!("MODEL"))` and transmuted directly. The `Aligned<T>` wrapper ensures 64-byte alignment.

All history tables use `zeroed_box()` for zero-initialized heap allocation, avoiding default constructors for large arrays.

**Coda comparison**: We load nets from files at startup. Embedded weights are faster (no I/O, no parsing) but require recompilation for new nets.

---

## 5. Lazy SMP (`threadpool.rs`, `thread.rs`)

### 5.1 Shared State

- **TT**: Shared, lockless (same as everyone)
- **Correction history**: `SharedCorrectionHistory` (pawn, minor, non_pawn[2]) shared via `AtomicI16` entries, accessed through `NumaReplicator`
- **Node counter**: Per-thread sharded `Counter` with 64-byte aligned atomics (avoids false sharing)
- **TB hits counter**: Same sharded design
- **Status**: Atomic stop flag
- **Best stats**: `AtomicU32[MAX_MOVES]` for sharing best score + depth across threads (packed as `(depth << 16) | (score + 32768)`)
- **Soft stop votes**: Atomic counter for consensus-based time management

### 5.2 Per-Thread State

- Board, stack, NNUE network (both accumulator stacks + cache), quiet/noisy/continuation history, continuation correction history, root moves, PV table, time manager.

### 5.3 Worker Thread Model (`threadpool.rs`)

Uses a **message-passing architecture** with pre-spawned worker threads:
- Workers wait on a `SyncChannel` for work closures
- `ReceiverHandle` with Condvar for join semantics
- Workers signal completion via `(Mutex<bool>, Condvar)` pair
- Unsafe lifetime transmute for scoped thread work (`search.rs` references are 'scope, channel needs 'static)

### 5.4 Soft Stop Consensus (`search.rs:231-244`)

Novel: threads **vote** on soft stop:
```rust
if soft_limit_reached:
    votes = soft_stop_votes.fetch_add(1) + 1
    if votes >= 65% of threads:
        set STOPPED
```
If a thread un-votes (soft limit no longer reached), it decrements. This prevents premature stopping when one thread has a bad estimate.

**Coda comparison**: We use main-thread-only time management. The consensus approach is more robust with many threads.

### 5.5 NUMA Support (`numa.rs`)

Optional feature (`--features numa`):
- Discovers NUMA topology via `libnuma`
- Binds threads to NUMA nodes via `numa_run_on_node` + `numa_set_preferred`
- `NumaReplicator<T>`: Allocates shared data per NUMA node via `numa_alloc_onnode`
- Only binds when thread count >= concurrency/2 (avoids overhead for few threads)

Correction history is replicated per NUMA node.

**Coda comparison**: We have no NUMA support. For multi-socket testing this matters.

---

## 6. Time Management (`time.rs`)

### 6.1 Soft/Hard Bounds

**Fischer**:
```rust
soft_scale = 0.024 + 0.042 * (1.0 - exp(-0.045 * fullmove_number))
hard_scale = 0.742
soft = soft_scale * main + 0.75 * inc
hard = hard_scale * main + 0.75 * inc
```
Exponential ramp-up: early game uses ~2.4% of time, late game uses ~6.6%.

### 6.2 Soft Stop Multiplier (`search.rs:217-229`)

```rust
multiplier = nodes_factor * pv_stability * eval_stability * score_trend * best_move_stability
```

- **nodes_factor**: `(2.72 - 2.27 * (best_move_nodes / total_nodes)).max(0.56)`
  - If best move used most of the nodes, stop early
- **pv_stability**: `(1.25 - 0.05 * stability_count).max(0.85)`
  - Stable PV = stop earlier
- **eval_stability**: `(1.2 - 0.04 * stability_count).max(0.88)`
  - Stable eval = stop earlier
- **score_trend**: `(0.8 + 0.05 * (prev_score - current_score)).clamp(0.80, 1.45)`
  - Dropping score = search longer
- **best_move_stability**: `1.0 + best_move_changes / 4.0`
  - Many best-move changes = search longer

### 6.3 Hard Stop

Only thread 0 checks hard limit, every 2048 nodes. Other threads only check for STOPPED status.

**Coda comparison**: Similar time management structure to ours. The exponential soft_scale ramp and multi-factor soft multiplier are well-tuned. The consensus voting is the main innovation.

---

## 7. Transposition Table (`transposition.rs`)

### 7.1 Structure

- 3 entries per 32-byte cluster (10 bytes each + 2 padding)
- Entry: key(u16) + move(u16) + score(i16) + raw_eval(i16) + depth(i8) + flags(u8)
- Flags pack: bound(2 bits) + tt_pv(1 bit) + age(5 bits) = 32 age cycles
- Lemire fast modular indexing: `(hash as u128 * len as u128) >> 64`

### 7.2 Replacement Policy

Priority: exact key match > empty slot > lowest quality. Quality = `depth - 4 * relative_age`.

Write-protection: skip if same-age, same-key, and `depth + 4 + 2*tt_pv <= existing_depth`.

### 7.3 GHI Mitigation (`board.rs:77-81`)

```rust
hash ^ ZOBRIST.halfmove_clock[(halfmove_clock.saturating_sub(8) / 8).min(15)]
```
Hashes differ every 8 plies of the halfmove clock, preventing Graph History Interaction where positions with different histories hash identically.

**Coda comparison**: We don't have GHI mitigation. This is a correctness improvement worth adding.

### 7.4 Score From TT (`transposition.rs:269-305`)

Sophisticated handling of mate/TB scores:
- Downgrades potentially false mate scores when `MATE - score > 100 - halfmove_clock`
- Downgrades potentially false TB scores with same logic
- Prevents stale mates/TB wins from the 50-move rule from being treated as real

### 7.5 Linux Huge Pages

TT allocation on Linux uses `mmap` with `MADV_HUGEPAGE`. Parallel clearing with multi-threaded `write_bytes`.

---

## 8. Board / SEE

### 8.1 SEE (`board/see.rs`)

Standard alpha-beta SEE with:
- **Pin-aware**: Excludes pinned pieces from attacking set when their pinner is still on the board (`see.rs:54-56`). Uses pre-computed `pinned` and `pinners` bitboards.
- **King ray check**: Only allows pinned pieces to capture along the pin ray (`king_rays[stm]`)

**Coda comparison**: Our SEE doesn't consider pins. This is more accurate.

### 8.2 Pre-computed State

Board state includes pre-computed: piece threats per type, all threats, pinned/pinners per side, checkers, checking squares per type, recapture square. All computed incrementally during makemove.

---

## 9. Novel Features Summary

### What Reckless Has That We Don't

1. **Threat accumulator** (NNUE): Separate i8 accumulator encoding all piece-on-piece attack relationships with SIMD-vectorized incremental updates. This is the biggest differentiator.

2. **Pairwise SCReLU activation**: `left * right` pairing of accumulator halves, more expressive than simple SCReLU.

3. **Sparse L1 matmul**: NNZ detection + sparse multiply. Major NPS win when many activations are zero.

4. **Factorized threat-aware history**: Each history entry has an unconditional factorizer + threat-conditioned buckets.

5. **6-source correction history**: Pawn, minor, non-pawn(white), non-pawn(black), continuation(ply-2), continuation(ply-4).

6. **Shared correction history across threads**: Via atomics, all threads contribute to eval correction.

7. **Soft stop consensus**: Threads vote on when to stop, majority required.

8. **Multi-factor countermove bonus**: Graduated history bonus on fail-low based on depth, move count, TT match, eval surprise.

9. **GHI mitigation**: Hash key varies with halfmove clock bucket.

10. **Pin-aware SEE**: Excludes pinned pieces.

11. **Eval-dependent aspiration delta**: Wider windows for volatile positions.

12. **Alpha-raise tracking in LMR**: More reduction after alpha has been raised.

13. **NUMA support**: Thread binding and per-node memory allocation.

14. **Quiet generation in qsearch**: When TT suggests a quiet move is good.

### Priority Adoption List (for Coda/GoChess)

**High impact, moderate effort:**
1. 6-source correction history (pawn + minor + non_pawn + continuation)
2. Factorized threat-aware history tables
3. GHI mitigation in hash
4. Pin-aware SEE
5. Shared correction history across SMP threads

**High impact, high effort:**
6. Threat accumulator (requires Bullet training changes)
7. Pairwise activation (same)
8. Sparse L1 matmul

**Medium impact, low effort:**
9. Eval-dependent aspiration delta
10. Alpha-raise tracking in LMR
11. Multi-factor countermove bonus
12. Soft stop consensus
13. Quiet moves in qsearch based on TT

**Low priority:**
14. NUMA support (only matters for multi-socket)
15. Compile-time net embedding
