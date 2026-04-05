# V7 NNUE Training Guide

Comprehensive guide for training Coda's v7 hidden-layer NNUE nets.
Covers architecture, training, quantization, known issues, and experiment plan.

## Target Architecture

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
| **Coda (target)** | 1536 (768pw) | Yes | CReLU→pairwise | 16 | 32 | 1×8 | int8 | float | None |

**Universal consensus**: L1=16, L2=32, float for L2+, 8 output buckets, SCReLU on hidden layers.

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

**1. Eval scale miscalibration (affects v5 AND v7)**

Check-net comparison shows our nets have wrong piece value ordering:

| Position | Obsidian | Our v5 | Our v7 e800i8 | Expected |
|----------|---------|--------|---------------|----------|
| startpos | +10 | +54 | +47 | ~0 |
| miss pawn | -62 | -150 | -102 | -100 |
| miss knight | -615 | -951 | -2044 | -320 |
| miss bishop | -674 | -1062 | -2032 | -330 |
| miss rook | -739 | -1091 | -1529 | -500 |
| miss queen | -998 | -718 | -580 | -900 |

Our v5: queen(-718) scored less negative than knight(-951) — fundamentally wrong.
Our v7: even worse, knight(-2044) is 3.5× queen(-580).
Obsidian: correct ordering queen > rook > bishop > knight > pawn.

**Root cause hypothesis**: T80 training data never contains "queen-down in opening" positions. The net has never learned the relative value of major pieces in material-imbalanced positions.

**2. v7 plays ~50-100 Elo behind v5**

Partly NPS (~2.5× penalty from hidden layer matmul), partly eval quality. The v7 eval scale is more extreme (wider scores for minor pieces), which interacts badly with search thresholds tuned for v5.

**3. 768pw training config uses ReLU, not SCReLU**

The `v7_768pw_h16x32.rs` config uses `.relu()` on hidden layers instead of `.screlu()`. This contradicts consensus — every top engine uses SCReLU on hidden layers. The `v7_1024h16x32s.rs` config correctly uses SCReLU.

**4. LR warmup helps but may not be enough**

20 SB linear warmup prevents immediate neuron death. But the overall training dynamics still produce miscalibrated evals. The warmup solves stability, not quality.

## Training Configs

### Current configs in `training/configs/`

| Config | FT | Hidden | Activation | Status |
|--------|-----|--------|-----------|--------|
| v5_768pw.rs | 768pw | None | CReLU→pairwise | Production |
| v5_1024s.rs | 1024 | None | SCReLU | Working |
| v7_1024h16x32s.rs | 1024 | 16→32 | SCReLU | Trained, weak eval |
| v7_768pw_h16x32.rs | 768pw | 16→32 | CReLU+ReLU (BUG) | Needs SCReLU fix |

### Training Infrastructure
- **Bullet GPU trainer** (Rust, CUDA)
- **Data**: 12× T80 binpack files (~24B positions of LC0-scored Stockfish games)
- **Supplemental**: 100M Coda self-play positions with blunder diversity
- **GPU hosts**: 3 available for parallel experiments
- **Timing**: e100 ≈ 1 hour, e800 ≈ 8 hours

### Key Training Parameters
- **eval_scale**: 400.0 (converts sigmoid to centipawn space)
- **batch_size**: 16,384
- **batches_per_superbatch**: 6104 (~100M positions per SB)
- **score_filter**: `unsigned_abs() < 10000` (consider tightening to 2000)
- **LR**: 0.001 peak, cosine decay to 0.00001
- **Warmup**: 20 SB linear ramp (0.0001 → 0.001)
- **WDL**: 0.0 (pure score training, but w7 = 0.07 was optimal for v5)
- **Optimizer**: AdamW with stricter clipping (0.99) for FT/factoriser weights

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

## Open Questions

1. **Is the piece value ordering a v5 problem too?** If yes, fixing it on v5 first is faster feedback.
2. **Is it a data distribution issue?** T80 never has "queen down in opening." Need diverse material-imbalance training data.
3. **Is eval_scale=400 correct?** Does it match what other Bullet-trained engines use?
4. **Is the score filter (10000) too loose?** Tightening to 2000 might help by excluding extreme positions.
5. **Are 8 output buckets by material count optimal?** Berserk uses 1 bucket successfully.
6. **Are 16 king buckets right?** Obsidian uses 13, Reckless 10. Different granularity.
7. **Do we need threat features?** Reckless, Halogen, Stormphrax all use them. Biggest architectural gap.
8. **Should we try power-2.5 loss?** +16-24 Elo in Cosmo's testing.
9. **Should we try beta1=0.95?** +4 Elo in Motor.
10. **Is suicide-chess data generation viable?** Natural material-imbalance positions via forced-capture self-play.

## Experiment Plan

### Principle: Fail fast with e100 runs, validate with e800

Each experiment changes ONE variable from a known baseline. Define success criteria before running. Use 3 GPU hosts for parallel experiments.

### Phase 1: Fix V5 Eval Quality (fastest feedback loop)

**Baseline**: current production v5-768pw-w7-e800

| # | Experiment | Change | Success metric | Time |
|---|-----------|--------|----------------|------|
| 1a | Suicide-chess data | Blend 5% forced-capture self-play data | Check-net queen > knight | 1h datagen + 1h train |
| 1b | Tighter score filter | 10000 → 2000 | Check-net values closer to Obsidian | 1h |
| 1c | Power-2.5 loss | MSE → power-2.5 | Lower test loss, better check-net | 1h |
| 1d | WDL=0.07 + suicide data | Combine best of 1a with proven WDL | Self-play vs baseline | 1h |

**Decision gate**: If any of 1a-1d fixes v5 check-net ordering, apply same fix to v7 training.

### Phase 2: V7 Architecture Fixes

**Baseline**: v7-1024h16x32s with best training recipe from Phase 1

| # | Experiment | Change | Success metric | Time |
|---|-----------|--------|----------------|------|
| 2a | Fix 768pw config | ReLU→SCReLU on hidden layers | Better loss than CReLU | 1h |
| 2b | Single hidden layer | 768pw→16→1×8 (no L2) | Does L1 train at all? | 1h |
| 2c | Frozen FT | Load v5 FT weights, freeze, train L1+L2 only | Hidden layers learn | 1h |
| 2d | Frozen FT + unfreeze | 2c then unfreeze FT at 0.1× LR for 50 SB | Better than 2c | 2h |

### Phase 3: Training Tricks

| # | Experiment | Change | Success metric | Time |
|---|-----------|--------|----------------|------|
| 3a | AdamW beta1=0.95 | Default 0.9 → 0.95 | Lower test loss | 1h |
| 3b | Tighter weight clipping | 0.99 → 0.5 | Better quantized eval | 1h |
| 3c | L1 activation sparsity | Add L1-norm penalty on FT activations | >70% block-sparsity | 1h |
| 3d | Knowledge distillation | Train v7 to match v5 output | Faster convergence | 1h |

### Phase 4: Scale Up

Take the best recipe from Phases 1-3 and train a full e800. Compare against production v5 via cross-engine RR.

## Data Generation

### Current data
- 12× T80 binpack files (~24B positions)
- 100M Coda self-play with blunder diversity

### Planned: Suicide-Chess Generator

Add a `--force-captures` mode to Coda's datagen command:
- Play normal self-play games
- At configurable probability (e.g. 30-50%), force a random capture move instead of best move
- Creates natural games with organic material imbalances
- Sequential positions compress well in binpack format
- Labels come from the engine's own eval at each position

This directly addresses the T80 data distribution gap — the training data will include positions where one side is materially down, which T80 almost never has.

### Piece-Removal Generator (built, not yet used)

Loads EPD positions and randomly removes pieces. Creates material imbalances but:
- Non-sequential positions (bad binpack compression)
- Artificial piece placement patterns
- No search scores (needs separate evaluation pass)

Suicide-chess approach is preferred.

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
