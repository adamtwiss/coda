# NNUE Training Guide

Comprehensive guide for training Coda's NNUE nets — from v5 improvements
through v7 hidden layers. Covers architecture options, training, known issues,
and experiment plan.

## Architecture Options

### Option A: Velvet-Style (simplest, proven +100-200 Elo above us)

**768 → 1 (single output, no hidden layers, 32 king buckets, CReLU)**

Velvet is +100-200 Elo above Coda with this simple architecture:
- 768 FT width, CReLU activation
- 32 king buckets (vs our 16) — more positional granularity
- **1 output bucket** (no material-count bucketing)
- **No hidden layers** — direct FT→output
- No pairwise, no SCReLU

Note: early check-net analysis suggested output bucketing broke piece value
ordering, but this was a misleading metric (sigmoid saturation — see Known
Issues section). Output buckets are standard and used by most top engines.
32 king buckets provide more positional granularity.

This is the **recommended first step** — a well-trained Velvet-style net on our
infrastructure, before attempting hidden layers.

### Option B: Single Hidden Layer (stepping stone to v7)

**768pw → 16 → 1×8 (or 768 → 16 → 1)**

One hidden layer adds capacity without the complexity of v7's dual layers.
Easier to train (fewer parameters, simpler gradients). Can validate that our
training pipeline produces healthy hidden layers before adding L2.

### Option C: Full v7 (target, matching top engines)

**768pw → 16 → 32 → 1×8** (matching Obsidian, Viridithas, Reckless, Halogen)

```
Input: HalfKA 768 features (12 pieces × 64 squares) per king bucket
  → Feature Transformer: 768 × 16 king buckets → 1536
  → CReLU → Pairwise multiplication → 768 per perspective
  → Concat both perspectives → 1536
  → L1: 1536 → 16 (int8 weights, QA=64, SCReLU activation)
  → L2: 16 → 32 (float weights, SCReLU activation)
  → Output: 32 → 1 × 8 output buckets (float, linear/sigmoid)
```

### Top Engine Architecture Comparison

| Engine | FT width | Pairwise | FT activation | L1 | L2 | Output | L1 quant | L2+ | Extra features |
|--------|---------|----------|--------------|-----|-----|--------|----------|-----|----------------|
| Stockfish | 1024 | No | SCReLU+CReLU dual | 31 | 32 | 1 | int8 | float | Dual net (big+small) |
| Obsidian | 1536 | Yes | CReLU→pairwise | 16 | 32 | 1×8 | int8 sparse | float | None |
| Viridithas | 2560 | Yes | CReLU→pairwise | 16 | 32 | 1×8 | int8 sparse | float | None |
| Reckless | 768 | Yes | SCReLU→pairwise | 16 | 32 | 1×8 | int8 | float | 66K threat features |
| Berserk | 1024 | No | CReLU | 16 | 32 | 1 | int8 sparse | float | Threat score |
| Alexandria | 1536 | No | dual (CReLU+SCReLU) | 16→32 | 32 | 1×8 | int8 | float | None |
| Halogen | 640 | Yes | SCReLU→pairwise | 16 | 32 | 1×8 | int8 | float | Threat features |
| Stormphrax | 128+ | Yes | SCReLU | 16 | 32 | 1×8 | int8 | float | 60K threat features |
| **Velvet** | 768 | No | CReLU | — | — | **1** | — | — | 32 king buckets |
| **Coda v5 (current)** | 768pw | Yes | CReLU→pairwise | — | — | 1×8 | — | — | 16 king buckets |
| **Coda v7 (target)** | 768pw | Yes | CReLU→pairwise | 16 | 32 | 1×8 | int8 | float | None |

**Note**: Velvet is +100-200 Elo above us with NO hidden layers, NO output buckets,
and simpler move ordering. This suggests net quality and training matter more than
architecture complexity at our level.

### FT→L1 Packing: Consensus Analysis (2026-04-13)

**How top engines pack FT output to int8 for L1 matmul:**

| Engine | FT activation | Packing to u8/i8 | Shift | L1 QB |
|--------|--------------|-------------------|-------|-------|
| Viridithas | CReLU→paired product | `clamp(left,0,255) * clamp(right,0,255) >> 9` → u8 | 9 | 64 |
| Obsidian | CReLU→paired product | `clamp(left,0,255) * clamp(right,0,255) >> 9` → u8 | 9 | 128 |
| Berserk | CReLU | `clamp(x, 0, 127<<5) >> 5` → i8 | 5 | integer chain |
| Reckless | SCReLU→pairwise | (not yet verified) | — | — |
| Stockfish | SCReLU+CReLU dual | (unique dual-net architecture) | — | — |
| **Coda (broken)** | **SCReLU (x²)** | `clamp(x,0,255)² >> 8` → u8 | 8 | 64 |

**Key insight:** Viridithas and Obsidian both use **paired/gated product** (left_half × right_half),
NOT SCReLU squaring. This is effectively CReLU→pairwise_mul — the same as our v5 FT activation.
The paired product naturally preserves perspective symmetry and has better dynamic range than
self-squaring.

**Our SCReLU i8 failure mode:** SCReLU squares the accumulator values (x²), which compresses
the value range differently than the paired product. With only 16 L1 neurons and i8 precision,
the squaring destroys the sign structure needed for perspective-relative evaluation. The paired
product (two independent halves multiplied) maintains better information content per u8 value.

**Decision (2026-04-13):** Our v7 target architecture should use CReLU→pairwise on the FT
(matching Viridithas/Obsidian consensus and our v5 architecture), with i8 L1 using the
standard >>9 shift paired product packing. SCReLU on hidden layers (L1→L2, L2→output) is
fine — that's where top engines use it.

**What failed and why (history):**
1. Factoriser (l0f + init_with_effective_input_size) — kills hidden layers entirely
2. Bucketed hidden layers (.select(output_buckets) on L1/L2) — gradient starvation
3. SCReLU on FT with i8 L1 — perspective symmetry corruption
4. SCReLU on FT with i16 L1 — works but slow (no VPMADDUBSW kernel)
5. CReLU→pairwise on FT with i16 L1 — works (GoChess-style, proven)
6. CReLU→pairwise on FT with i8 L1 — **target** (matches consensus, not yet tested)

### Key Architecture Insights (from Cosmo/Viridithas research)

- **SCReLU dominates CReLU** on hidden layers — worth ~50% network size increase
- **Pairwise multiplication** halves FT width before L1, making hidden layers practical
- **Sparse L1 matmul** is essential — with ~70% block-sparsity, effective L1 cost is ~0.3× dense
- **Float for L2/L3** — too small for quantization to be worthwhile, avoids precision loss
- **Hard-Swish on L1** gained +14 Elo over SCReLU in Viridithas (but needs L1 activation regularization to maintain sparsity)
- **SwiGLU on L2** gained +5.5 Elo on top of Hard-Swish in Viridithas
- **Activation choice differs by layer** — PlentyChess found CReLU best for L1, SCReLU for L2
- **L1 activation regularization** (L1-norm penalty on FT outputs) drives sparsity for sparse matmul

## Current Status

### What We Have
- v7 inference code: FT→16→32→1×8, SCReLU, int8 L1, float L2+
- AVX2 and NEON SIMD for all paths
- Bullet training configs for v7_1024h16x32s and v7_768pw_h16x32
- Converter: `coda convert-bullet`
- Check-net diagnostic tool

### Known Issues

**1. Check-net piece value ordering — RESOLVED (misleading metric)**

Early check-net analysis showed apparent piece value misordering (knight > queen)
by comparing absolute evals of "remove one piece" positions. This was misleading
due to **sigmoid saturation**: at high skill levels, being up a bishop or up a
queen from the start both map to ~99% win probability. The absolute scores
("miss knight" = -951 vs "miss queen" = -718) are compressed into the sigmoid
plateau and don't reflect relative piece values.

The correct methodology: remove a queen from one side AND a bishop from the
other, measuring the **relative** eval difference. This produces logical piece
value orderings because the position is competitive (not saturated).

The earlier "root cause hypothesis" (T80 data lacking queen-down positions)
was incorrect — the issue was purely the check-net methodology, not the training
data or output bucketing.

**2. v7 hidden layers collapsed — RESOLVED: factoriser was the cause**

All Coda v7 nets trained with our standard config produced collapsed evals
(all scores identical, wrong signs on endgames). Ruled out as causes:
- Bucketed vs unbucketed hidden layers (both collapsed)
- WDL proportion (w=0 and w=0.4 both collapsed)
- Data volume (single file has 3-4B positions, plenty for s100)
- Data filtering (not the issue)

**Root cause: the factoriser** (`l0f` weight sharing + `init_with_effective_input_size(32)`).
The GoChess v7 config (which produces healthy nets) uses plain `new_affine` without a
factoriser. Removing the factoriser from our config immediately produced a healthy v7 net
(all 8 check-net tests pass, good differentiation).

Secondary factor: **i16 quantisation** for hidden layers. GoChess uses i16 at QA=255 for
L1/L2/L3. Our config used i8 at QA=64 for L1 and float for L2/L3. The i8 quantisation
may contribute to precision loss in the narrow 16-neuron L1 layer. Not yet tested
independently — the factoriser removal alone was sufficient.

**Working v7 training recipe** (2026-04-13):
- No factoriser (plain `new_affine` for FT)
- i16 quantisation for all hidden layers (QA=255 for L1/L2, QB=64 for output)
- SCReLU activation
- LR warmup (20 SBs)
- Position filtering + power-2.5 loss + low final LR
- Config: `v7_1024h16x32s_gochess_style.rs`
- Convert: `coda convert-bullet --screlu --hidden 16 --hidden2 32 --ft-size 1024 --int16-hidden`

**3. 768pw training config uses ReLU, not SCReLU**

The `v7_768pw_h16x32.rs` config uses `.relu()` on hidden layers instead of `.screlu()`. This contradicts consensus — every top engine uses SCReLU on hidden layers. The `v7_1024h16x32s.rs` config correctly uses SCReLU.

**4. LR warmup helps but may not be enough**

20 SB linear warmup prevents immediate neuron death. But with the factoriser, the overall training dynamics still produced collapsed evals. Without the factoriser, warmup + SCReLU produces healthy nets.

## Training Configs

### Current configs in `training/configs/`

| Config | FT | Hidden | Activation | Status |
|--------|-----|--------|-----------|--------|
| v5_768pw.rs | 768pw | None | CReLU→pairwise | Production |
| v5_1024s.rs | 1024 | None | SCReLU | Working |
| v7_1024h16x32s.rs | 1024 | 16→32 | SCReLU | Has factoriser (broken) |
| v7_768pw_h16x32.rs | 768pw | 16→32 | CReLU+ReLU | Has factoriser (broken) |
| v7_1024h16x32s_gochess_style.rs | 1024 | 16→32 | SCReLU | **No factoriser — WORKS** |

### Training Infrastructure
- **Bullet GPU trainer** (Rust, CUDA)
- **Data**: 12× T80 binpack files (~24B positions of LC0-scored Stockfish games)
- **Supplemental**: 100M Coda self-play positions with blunder diversity
- **GPU hosts**: 3 available for parallel experiments
- **Timing**: e100 ≈ 1 hour, e800 ≈ 8 hours

### Key Training Parameters (updated 2026-04-12)
- **eval_scale**: 400.0 (converts sigmoid to centipawn space)
- **batch_size**: 16,384
- **batches_per_superbatch**: 6104 (~100M positions per SB)
- **Position filter**: ply≥16, no checks, no captures, no tactical moves (+22-48 Elo)
- **LR**: 0.001 peak, cosine decay to **5e-6** (was 1e-4 — **+47 Elo from fix**)
- **Warmup**: 20 SB linear ramp (0.0001 → 0.001) — essential for v7 hidden layers
- **WDL for v5**: 0.07 (pure score works for v5's 12M FT params)
- **WDL for v7**: **0.4** (Viridithas uses this — hidden layers need game outcome signal)
- **Optimizer**: AdamW with stricter clipping (0.99) for FT/factoriser weights
- **Training length**: 400 SBs is the sweet spot (Viridithas finding). 800 gives marginal +1-2 Elo.

### Training Improvements Roadmap (from Viridithas research, 2026-04-12)

| Change | Expected Impact | Effort | Status |
|--------|----------------|--------|--------|
| Low final LR (5e-6) | **+47 Elo (proven)** | Config change | ✅ Done |
| Position filtering | **+22-48 Elo (proven)** | Config change | ✅ Done |
| WDL 0.4 for v7 | +10-20 Elo | Config change | Training overnight |
| Power loss 2.5 | +16-24 Elo (Cosmo) | 1-line config change | Not yet tried |
| Fresh self-play data | +37 Elo (Viridithas "stalker") | Datagen run | Generating on Titan |
| AdamW beta1=0.95 | +4 Elo | 1-line config change | Not yet tried |
| L1 activation regularization | Unknown but positive | May need custom code | Not yet tried |

Power loss is available in Bullet:
```rust
// One-line change from MSE to power-2.5:
.loss_fn(|output, target| output.sigmoid().power_error(target, 2.5))
```

## Velvet Trainer Findings (2026-04-06)

Velvet (+100-200 Elo above us) uses a **custom PyTorch trainer**, not Bullet.
Key differences that likely contribute to their net quality:

1. **Power-2.6 loss** (not MSE) — penalizes large errors more. Cosmo found +16-24 Elo.
2. **Patience-based LR decay** — starts 0.001, ×0.4 on plateau, patience halves each time.
   Adapts to training dynamics vs our fixed cosine schedule.
3. **Weight decay 0.01** (AdamW) — we don't set explicit weight decay.
4. **Batch size 32K** — double our 16384. More stable gradients.
5. **Validation-based checkpointing** — saves best model, reloads on plateau.
6. **No WDL blending** — pure sigmoid(score) targets.
7. **Self-play data** — not T80/LC0 data.

### What We Should Try (in Bullet)
- Power-2.6 loss: `|output.sigmoid() - target|.pow(2.6).mean()` (custom loss_fn)
- Weight decay: add to AdamW config
- Batch size 32K: change batch_size param
- Score filter tightening: Velvet scores are self-play (capped), not LC0 (uncapped)

### v7 Training Failure (2026-04-06)
All three 768pw v7 experiments (A, C, E) produced **completely collapsed nets**:
- Exp A (SCReLU baseline): all scores ≈ -6409 (dead hidden layers)
- Exp C (lower final LR): all scores = 0 (zero output)
- Exp E (position weighting): all scores ≈ -38403 (diverged)

Raw weight analysis of Exp A quantised.bin: FT weights ±11 (should be ±100-300),
output layer weights exploded to 10³¹. Classic gradient instability: output layer
gets huge gradients, FT gets vanishing gradients through hidden layer bottleneck.

**The 768pw + hidden layers combination doesn't train in our Bullet pipeline.**
Previous 1024h (non-pairwise) models worked — the pairwise→hidden path is the
specific failure mode. The CReLU→pairwise creates a gradient bottleneck that
prevents the FT from learning while the output explodes.

**Revised strategy**: Focus on Velvet-style v5 improvements (32 king buckets,
training quality) before attempting hidden layers.

### Power-2.6 Loss Failure (2026-04-06)
Experiments G, H, I all collapsed with power-2.6 loss — scores of 44M.
The loss function produces larger gradients than MSE, causing output weight
explosion without explicit weight decay (Velvet uses 0.01). Bullet's
default AdamW may not include sufficient weight decay.

### Output Bucket Hypothesis — DISPROVEN (2026-04-06)
Tested 2 output buckets vs 8 (exp L vs production). Both showed similar
check-net results. The apparent "piece ordering issue" was a check-net
methodology problem (sigmoid saturation), not a real eval defect.
See Known Issues section for full explanation.

LC0 evaluates based on **win probability**, not material. A position where
White is down a knight might be scored -200cp (modest disadvantage) if there's
compensation. A position down a queen might be -300cp if there's activity.
LC0 doesn't score proportionally to material value — it scores based on how
likely each side is to win from that position.

Our NNUE learns these LC0 score distributions. It never sees "queen down =
-900cp" because LC0 doesn't think in material terms. The net correctly learns
LC0's evaluation style, which happens to not differentiate piece values the
way traditional engines expect.

This explains:
- Why all our models (v4, v5, 2-bucket, 8-bucket) show similar queen/knight confusion
- Why blunder self-play data (scored by our engine with material-aware eval) showed
  better calibration — our eval scores queen-down much more negatively than LC0
- Why Velvet (which trains on self-play, not LC0 data) may have better piece ordering
- Why Obsidian (also LC0 data) might compensate with different training techniques

**Key finding (2026-04-06 evening)**: The net DOES correctly differentiate piece
values in *relative* comparisons. Position with White missing queen + Black missing
knight scores -583cp — exactly the queen-knight difference (~580cp classical).

The check-net issue is sigmoid saturation: being down ANY major piece vs full
material gives ~95%+ win probability for the opponent, compressing all "piece down"
scores into a narrow band. But in positions where both sides have imbalances
(which is what actually occurs in games), the relative values are correct.

**Conclusion**: The piece ordering in check-net is a diagnostic artifact, not a
playing strength issue. Focus on Elo, not check-net piece ordering. The engine
correctly evaluates relative material differences in real game positions.

### Experiment Results Summary (2026-04-06)

| # | Experiment | Architecture | Change | Result |
|---|-----------|-------------|--------|--------|
| A | 768pw v7 SCReLU | Hidden 16→32 | Baseline with SCReLU fix | COLLAPSED (gradient instability) |
| C | 768pw v7 lower LR | Hidden 16→32 | Lower final LR | COLLAPSED (zero output) |
| E | 768pw v7 weight imb | Hidden 16→32 | 2× weight for imbalanced | COLLAPSED (diverged) |
| G | 1024 v7 + power-2.6 | Hidden 16→32 | Power-2.6 loss | COLLAPSED (output explosion) |
| H | 768pw v5 + power-2.6 | No hidden | Power-2.6 loss, 8 buckets | COLLAPSED (output explosion) |
| I | 768pw v5 + power-2.6 | No hidden | Power-2.6 loss, 1 bucket | COLLAPSED (output explosion) |
| J | 768pw v5, 2 bucket | No hidden | Wrong ft_size in config | CONVERTER ERROR |
| K | 768pw v5, 8 bucket | No hidden | Wrong ft_size in config | WRONG SCALE (44M) |
| L | 768pw v5, 2 bucket | No hidden | Clean production copy | HEALTHY — piece ordering still wrong |

11 of 12 experiments failed due to config bugs or untested loss functions.
Only exp L (exact production copy with 1 line changed) produced a healthy model.

## Research Findings

### From Cosmo/Viridithas (2024-2026)

**Architecture:**
- SCReLU ≈ 1.5× CReLU capacity. Our CReLU L1=16 effectively equals SCReLU L1=10.
- Pairwise multiplication is the standard dimensionality reduction before L1.
- Hard-Swish (+14 Elo over SCReLU on L1) and SwiGLU (+5.5 on L2) are the latest frontier, but require activation regularization.

**Training:**
- **Power-2.5 loss** gained +16-24 Elo over MSE in Motor. Penalizes large errors more.
- **AdamW beta1=0.95** (vs 0.9) gained ~4 Elo.
- **Tighter weight clipping** for quantization gained +7-9 Elo.
- Training trick gains don't always compose — test at LTC, not just fixed-nodes.
- LR schedule choice (cosine/step/linear) matters little; cosine marginally best.
- Overfitting unlikely with billions of positions and small nets.

**Inference:**
- Sparse L1 matmul with ~70% block-sparsity cuts L1 cost to ~0.3×.
- L1 activation regularization during training drives sparsity.
- `VPMADDUBSW` (u8×i8→i16) is the workhorse instruction for L1.
- Finny tables, lazy accumulators, fused updates all critical for NPS.

### From Stockfish NNUE Wiki

- Most knowledge is in the first layer. Hidden layers add refinement.
- Quantization error accumulates with depth — float for post-L1 layers.
- Dual-net approach (big+small) saves NPS on lopsided positions.
- King bucket count and layout affect training efficiency.

## v7 Inference Investigation (2026-04-12)

### Validated: Inference is Correct for Unbucketed Nets

Loaded GoChess v7 `net-v7-1024h16x32s-w0-e1200-12f-s1200.nnue` (unbucketed, flags=0x05)
in Coda successfully. Results match GoChess direction:

| Position | Coda (v7 unbucketed) | GoChess | Status |
|----------|---------------------|---------|--------|
| startpos | +24 | ~same | ✓ OK |
| miss pawn | -12 | ~same | ✓ OK |
| miss rook | -475 | -1803 | ✓ Same sign |
| EG rook up | +2172 | ~same | ✓ OK |
| EG queen up | +2118 | ~same | ✓ OK |

All SIMD paths (AVX2, NEON) and scalar fallback produce identical results.

### Bucketed vs Unbucketed Formats

Two v7 .nnue formats exist:
- **Unbucketed** (GoChess, flags bit 3 = 0): L1/L2 are flat (per-bucket size).
  Output layer has bucket selection. File ~25.2MB.
- **Bucketed** (Coda, flags bit 3 = 1): L1/L2 dimensions are multiplied by
  NUM_OUTPUT_BUCKETS. Bucket selection at each layer. File ~25.7MB.

Both are now supported in Coda's inference code.

### Root Cause: Training, Not Inference

The Coda v7 net (w0 + filtered, s800) produces near-constant output (~16cp for
all positions). Investigation confirmed:
- L1 weights are alive in quantised.bin: ~42% nonzero, mean|w|=2.3
- L1 weights correctly transferred to .nnue file: ~36.5% nonzero
- Both SIMD and scalar paths produce identical (constant) results
- Output weights alive in .nnue file (nonzero, correct offset)

The hidden layers simply didn't learn position-dependent features during training.

### Why Training Failed: w0 + Filtering

- **w0 (0% WDL)**: Hidden layers only see search scores as training signal.
  The 33K parameters of 16→32 hidden layers need cleaner gradients than noisy
  search scores provide. w0 works for v5's 12M FT parameters but not for tiny
  hidden layers.
- **Position filtering** (quiet positions only): Removes exactly the positions
  where material differences are most visible — captures and tactical positions.
  Hidden layers need these to learn piece value relationships.
- **Viridithas uses w=0.4 (40% WDL)** for their v7 nets. Game outcome signal
  provides cleaner gradients: "this side won" is unambiguous vs "search scored
  this +247 or maybe +198 depending on depth."

### Fix: Train with w=0.4 WDL

Next v7 training run should use `--wdl 0.4` (matching Viridithas). Config
`v7_1024h16x32s.rs` is ready. Everything else stays the same:
- 1024 SCReLU, L1=16, L2=32
- Filtering (quiet positions)
- Low final LR (5e-6 or Bullet formula)
- Warmup 20 SBs
- e800, 12 T80 files

### Bugs Fixed During Investigation

1. **L1 bucket indexing**: `b_off = bucket * l1_per_bucket` for biases and weights
   (AVX2, NEON, scalar paths)
2. **L2 weight stride**: `i * l2_size + bucket * l2_per_bucket` (was `(b_off + i) * l2_per_bucket`)
3. **Scalar fallback buffers**: `[64]` → `[256]` for bucketed nets
4. **Unbucketed support**: `b_off = 0`, `l2_off = 0` when `bucketed_hidden = false`
5. **Display**: check-net now shows per-bucket L1/L2 sizes, not total bucketed

## Open Questions

1. ~~Is the piece value ordering a v5 problem too?~~ RESOLVED — check-net methodology was misleading (sigmoid saturation).
2. ~~Is it a data distribution issue?~~ RESOLVED — same root cause as above.
3. ~~**Is eval_scale=400 correct?**~~ RESOLVED — eval_scale=400 is correct for all nets. Clipped eval RMS matches across nets. The apparent scale differences were from tails, not the search-relevant range.
4. ~~**Is the score filter (10000) too loose?**~~ RESOLVED — standard filtering (ply≥16, no checks, no captures, no tactical) gives +22-48 Elo.
5. **Are 8 output buckets by material count optimal?** Berserk uses 1 bucket successfully.
6. **Are 16 king buckets right?** Obsidian uses 13, Reckless 10. Different granularity.
7. **Do we need threat features?** Reckless, Halogen, Stormphrax all use them. Biggest architectural gap.
8. **Should we try power-2.5 loss?** +16-24 Elo in Cosmo's testing.
9. **Should we try beta1=0.95?** +4 Elo in Motor.
10. **Is suicide-chess data generation viable?** Natural material-imbalance positions via forced-capture self-play.
11. **v7 with w=0.4 WDL**: Will this produce viable hidden layers? GPU4 training queued.

## Experiment Plan

### Principle: Fail fast with e100 runs, validate with e800

Each experiment changes ONE variable from a known baseline. Define success criteria before running. Use 3 GPU hosts for parallel experiments.

### Phase 0: Velvet-Style Baseline (fastest path to better net)

Test whether simpler architecture with better training produces stronger nets.

| # | Experiment | Change | Success metric | Time |
|---|-----------|--------|----------------|------|
| 0a | 768 CReLU, 1 output, 32KB | Velvet-style: no pairwise, no buckets, 32 king buckets | Check-net ordering + bench first-move% | 1.5h |
| 0b | 768pw, 1 output, 16KB | Our pairwise FT but no output buckets | Compare vs 0a and current v5 | 1.5h |
| 0c | 768pw, 1 output, 32KB | Pairwise + 32 king buckets, no output buckets | Best of 0a and 0b combined | 1.5h |

**Decision gate**: If any of 0a-0c has correct piece ordering (queen > knight)
and competitive first-move%, it validates that output buckets are the problem.
Scale to e800 and SPRT against production v5.

### Phase 1: V7 Training Experiments (running 2026-04-06)

Six SB100 experiments on 768pw v7 (16→32 hidden layers). Configs in
`training/configs/experiments/`.

| # | Config | Change from baseline | Status |
|---|--------|---------------------|--------|
| A | exp_a_baseline_screlu | Control: SCReLU fix (was ReLU) | Running |
| B | exp_b_wdl030 | WDL=0.3 | Queued |
| C | exp_c_lower_final_lr | Final LR ×0.3⁵ (match Bullet) | Running |
| D | exp_d_wdl030_low_lr | WDL=0.3 + lower final LR | Queued |
| E | exp_e_weight_imbalance | 2× loss weight for \|score\|>500 | Running |
| F | exp_f_wdl_dynamic | Dynamic WDL scaling with \|score\| | Queued |

**Evaluation**: check-net values, bench first-move%, self-play H2H.

### Phase 1b: V5 Eval Quality (data experiments)

| # | Experiment | Change | Result |
|---|-----------|--------|--------|
| ✓ | Blunders-only e20/e60 | 100% blunder data | Better calibration but much weaker play |
| ✓ | T80+blunders blend e60 | 95% T80 + 5% blunders | No improvement — blend too dilute |
| — | Force-capture data | 70M positions generated, not yet trained | Pending |
| — | Higher blend ratio | 20-30% diverse data | Need more data first |

**Finding**: Blunder data improves calibration but can't replace T80 for playing
strength. The apparent "check-net ordering issue" was a methodology artifact
(sigmoid saturation), not a real training defect. See Known Issues section.

### Phase 2: Single Hidden Layer (stepping stone)

After Phase 0 establishes a good single-layer baseline:

| # | Experiment | Change | Success metric | Time |
|---|-----------|--------|----------------|------|
| 2a | 768pw→16→1 | Single hidden, SCReLU, no output buckets | Better than Phase 0 best | 1.5h |
| 2b | Frozen FT from 0-best | Load best v5 FT, freeze, train L1 only | L1 learns healthy weights | 1.5h |
| 2c | Unfreeze FT | 2b then unfreeze at 0.1× LR | Better than 2b | 2h |

### Phase 3: Full v7 (dual hidden layers)

Only after Phase 2 validates single hidden layer training:

| # | Experiment | Change | Success metric | Time |
|---|-----------|--------|----------------|------|
| 3a | 768pw→16→32→1×8 SCReLU | Full v7 with best training recipe | Better than Phase 2 | 1.5h |
| 3b | + lower final LR | If Phase 1 C/D show this helps | Better convergence | 1.5h |
| 3c | + position weighting | If Phase 1 E/F show this helps | Better calibration | 1.5h |

### Phase 4: Training Tricks (apply to whichever phase works)

| # | Experiment | Change | Time |
|---|-----------|--------|------|
| 4a | AdamW beta1=0.95 | +4 Elo in Cosmo's testing | 1.5h |
| 4b | Power-2.5 loss | +16-24 Elo in Cosmo's testing | 1.5h |
| 4c | Knowledge distillation | Train new arch to match best v5 output | 1.5h |

### Phase 5: Scale Up

Take best recipe and train e800. Compare against production v5 via cross-engine RR.

## Data Generation

### T80 Data (primary training data)
- 12× T80 binpack files, **~30B total positions** (confirmed via binpack-stats)
- 2.6 bytes/position (excellent chain compression, avg chain 108 positions)
- 6.2% in-check positions included, 55.7% draws
- `min-v2` = filtered (low-signal positions removed), `v6` = binpack format version
- Unfiltered — all positions from games, filtering at Bullet training time

### Self-Play Data
- **Blunders**: 69M positions, 16.1 bytes/pos, 6.8% draws, avg chain 69
- **Force-captures**: 66M positions, 13.3 bytes/pos, 19.9% draws, avg chain 66
- **Pure selfplay**: 10.1 bytes/pos, avg chain 111

### Compression Finding (2026-04-06)
Our data compresses 4-6× worse than T80 (10-16 bytes/pos vs 2.6). Root cause:
position filtering in datagen broke chain continuity. Fix applied: write ALL
positions unfiltered, filter at training time via Bullet loader. Also must store
the actually-played move (not best_move) for chain compression.

Remaining gap (still ~4× vs T80) likely due to sfbinpack Rust crate vs
Stockfish's C++ writer, or shorter game lengths.

### Key Data Insight (updated 2026-04-12)
The apparent "piece value ordering issue" in check-net was a misleading metric
caused by sigmoid saturation at extreme material imbalances — not a real eval
defect. See Known Issues section. Output buckets are standard and not harmful.

## Quantization Reference

| Layer | Our approach | Consensus | Notes |
|-------|-------------|-----------|-------|
| FT weights | int16 (QA=255) | int16 (QA=255) | Standard |
| FT bias | int16 (QA=255) | int16 (QA=255) | Standard |
| L1 weights | int8 (QA=64) | int8 (QA=64) | With NNZ-sparse matmul |
| L1 bias | float | float | Some engines use int32 |
| L2 weights | float | float | Too small to quantize |
| L2 bias | float | float | |
| Output weights | float | float | |
| Output bias | float | float | |

### SCReLU Scale Chain (critical for v7)

SCReLU squares the accumulator values, so the scale becomes QA². The full chain:
```
FT output: int16 at scale QA=255
SCReLU: v² at scale QA² = 65025
L1 matmul: bias×QA² + sum(v²×w), then /QA² after matmul
```
Do NOT divide by QA before the L1 matmul — keep v² at scale QA² through the matmul for precision.
Hidden→output activation is linear in Bullet (no SCReLU at final layer).
