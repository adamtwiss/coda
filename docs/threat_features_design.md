# Threat Features Design Doc

## Overview

Add ~67K threat input features to Coda's NNUE, following the Reckless implementation
pattern. Threat features encode (attacker, attacker_sq, victim, victim_sq) relationships
— which pieces attack which pieces on which squares. These are fed into the FT accumulator
via a separate i8 weight matrix, summed with the existing i16 PSQ accumulator at activation
time.

**Expected gain**: +40-60 Elo of eval quality, based on comparison of threat vs non-threat
engines at similar FT widths.

**Reference implementation**: Reckless (`/home/adam/chess/engines/Reckless/src/nnue/`)
— same base architecture as Coda (768 accumulator, pairwise, 16→32 hidden layers).

## Architecture

**Target: 768 accumulator (matching Reckless), not our current 1536.**

Reckless is #2 on CCRL with 768 accumulator + threats. Our current 1536 accumulator
without threats is ~50-60 Elo behind v5. The extra width isn't earning its NPS cost.
Narrowing to 768 gives us: 2× faster FT updates, half the L1 matmul work, half the
threat weight memory (~50MB instead of ~100MB), and a proven reference architecture.

```
Current v7/v8 (1536 accumulator, NO threats):
  768 PSQ features × i16 weights → 1536 accum → pairwise → 768/persp → 1536 L1 input

Target v9 (768 accumulator, WITH threats):
  768 PSQ features × i16 weights ─┐
                                   ├─ add → 768 accum → pairwise → 384/persp → 768 L1 input
  ~67K threat features × i8 weights┘
```

This halves both the FT and L1 dimensions while adding threat features. The hidden
layers (L1=16, L2=32, output) are unchanged. Net effect: smaller, faster FT with
richer input signal. Should improve both NPS and eval quality.

## Phase 1: Threat Feature Encoding (inference-only, no training)

### What to encode

Each threat feature represents: "piece P on square S attacks piece Q on square T."
Not all combinations are valid — a piece-pair interaction map filters which
attacker×victim pairs are tracked.

**Piece interaction map** (from Reckless):
```
                Victim: P  N  B  R  Q  K
Attacker Pawn:         [✓  ✓  ✗  ✓  ✗  ✗]   3 targets
Attacker Knight:       [✓  ✓  ✓  ✓  ✓  ✗]   5 targets
Attacker Bishop:       [✓  ✓  ✓  ✓  ✗  ✗]   4 targets
Attacker Rook:         [✓  ✓  ✓  ✓  ✗  ✗]   4 targets
Attacker Queen:        [✓  ✓  ✓  ✓  ✓  ✗]   5 targets
Attacker King:         [✓  ✓  ✓  ✓  ✗  ✗]   4 targets
```

Targets are counted per side (friendly + enemy), giving PIECE_TARGET_COUNT = [6,10,8,8,10,8].

### Index computation

For each piece on the board:
1. Compute its attack bitboard (we already do this for movegen)
2. For each attacked occupied square, compute threat_index:
   - Perspective-flip squares (rank flip for black, file flip if king on e-h files)
   - Look up piece-pair in interaction map (skip if -1)
   - Index = base_offset[attacker_type][attacker_sq] + attack_ray_index[attacker][from][to]
   - Ray index = popcount of attack squares below target in the attack bitboard

### Feature count

~66,864 features total. The exact count depends on the piece-pair filtering and the
number of valid attack squares per piece per square (pawns exclude ranks 1/8, etc).

### King-side mirroring

When the perspective king is on files e-h, all squares are horizontally mirrored
(XOR by 7). This halves the feature space by exploiting board symmetry. A king move
crossing the e-file boundary invalidates all threat indices and forces full recompute.

### Deliverable

- `src/threats.rs`: threat_index function, piece interaction map, offset tables
- Unit tests: verify feature indices match expected values for known positions
- Benchmark: measure cost of computing all threat features for a position

**Validation**: Compare our threat feature indices against Reckless's for the same
positions. We can instrument both engines to dump feature lists and verify they match
(modulo our wider accumulator).

## Phase 2: Threat Accumulator (inference)

### Data structures

```rust
struct ThreatAccumulator {
    values: [[i16; FT_SIZE]; 2],  // per perspective, FT_SIZE=768 (matching Reckless)
    accurate: [bool; 2],
}
```

Separate from the existing PSQ accumulator. Both are `i16[2][1536]`.

### Forward pass change

At activation time, add threat accumulator to PSQ accumulator before pairwise:
```rust
// Current:
let clipped = acc[i].clamp(0, QA);
// New:
let combined = psq_acc[i] + threat_acc[i];
let clipped = combined.clamp(0, QA);
```

One SIMD add per 16-byte chunk — negligible cost.

### Threat weights

```rust
threat_weights: Vec<i8>,  // [num_threat_features × FT_SIZE]
```

i8 quantization (not i16 like PSQ weights). Widened to i16 at update time via
`_mm256_cvtepi8_epi16`. Memory: 67K × 768 × 1 byte ≈ 50MB (matching Reckless exactly).

### Incremental updates

On each make_move:
1. Compute threat deltas (which threats were added/removed by the move)
2. For each removed threat: subtract i8 weight row from accumulator
3. For each added threat: add i8 weight row to accumulator

A typical move generates ~10-30 threat deltas (the moved piece's attacks change,
pieces that were attacking/defending the from/to squares change, discovered attacks
through the vacated square).

Full recompute triggers:
- King crosses the e-file boundary (mirroring changes)
- No accurate ancestor in the stack

### Deliverable

- Threat accumulator struct with push/pop semantics matching existing NNUE accumulator
- SIMD update functions (AVX2: `_mm256_cvtepi8_epi16` + `_mm256_add_epi16`)
- Integration into `forward_with_l1` / `forward_with_l1_pairwise` paths
- Full recompute function for positions without incremental history

**Validation**: Load a position, compute threat accumulator from scratch, make a move,
compute incrementally, verify values match full recompute. Fuzz with random positions.

## Phase 3: Bullet Trainer Fork

### Required changes to adamtwiss/bullet

The graph/IR layer (acyclib) already supports the architecture. Changes needed in
bullet_lib data pipeline (4-5 files):

**1. New threat feature encoder** (`crates/bullet_lib/src/game/threats.rs`):
```rust
pub struct ThreatFeatures;
impl SparseInputType for ThreatFeatures {
    fn num_inputs(&self) -> usize { 66864 }
    fn max_active(&self) -> usize { 256 }  // max threats per position
    fn map_features(&self, pos: &ChessBoard, ...) -> Vec<(usize, usize)> { ... }
}
```

**2. PreparedData** (`loader.rs`): Add second SparseInput field for threats.

**3. PreparedBatchHost** (`dataloader.rs`): Insert threat sparse matrix under key
"stm_threats" / "nstm_threats".

**4. ValueTrainerBuilder** (`builder.rs`): Accept optional second SparseInputType.
The build closure receives threat input nodes alongside PSQ input nodes.

**5. Training config** (`examples/coda_v9_768pw_threats.rs`):
```rust
.build(|builder, stm_inputs, ntm_inputs, stm_threats, ntm_threats, output_buckets| {
    let ft_psq = builder.new_affine("ft_psq", 768 * BUCKETS, 768);  // 768 accum (not 1536)
    let ft_threat = builder.new_affine("ft_threat", 66864, 768);
    
    let stm = ft_psq.forward(stm_inputs) + ft_threat.forward(stm_threats);
    let ntm = ft_psq.forward(ntm_inputs) + ft_threat.forward(ntm_threats);
    
    let stm_pw = stm.crelu().pairwise_mul();  // 768 → 384 per perspective
    let ntm_pw = ntm.crelu().pairwise_mul();
    let hl1 = stm_pw.concat(ntm_pw);          // 384 + 384 = 768 L1 input
    // L1(768→16) → L2(16→32) → output(32→1×8) ... unchanged
})
.save_format(&[
    SavedFormat::id("ft_psq_w").quantise::<i16>(255),
    SavedFormat::id("ft_psq_b").quantise::<i16>(255),
    SavedFormat::id("ft_threat_w").quantise::<i8>(64),
    // ... hidden layers unchanged
])
```

### Validation

1. Train a tiny net (FT=32, 100 threat features) to verify the pipeline works end-to-end
2. Check that threat weight gradients are non-zero (features are actually contributing)
3. Compare loss curves: with threats vs without (should converge faster/lower)

## Phase 4: Converter and File Format

### .nnue format extension

New version **v9** (or v8 flag bit):
```
Header:
  u32 magic ("NNUE")
  u32 version (9)
  u8  flags (existing bits + bit 6 = has_threats)
  u16 ft_size
  u16 l1_size
  u16 l2_size
  u32 num_threat_features  // NEW

Weights:
  i16[INPUT_SIZE × FT_SIZE]     // PSQ FT weights (existing)
  i16[FT_SIZE]                   // PSQ FT biases (existing)
  i8[num_threats × FT_SIZE]     // Threat FT weights (NEW)
  // NO threat biases (threats have no bias — zero-initialised accumulator)
  i16[L1_input × L1_SIZE]       // L1 weights (existing)
  ... rest unchanged
```

### Converter changes

`convert-bullet` gains `--threats <count>` flag. Reads the additional weight block from
quantised.bin and writes it into the .nnue file.

### Backwards compatibility

v5/v6/v7/v8 nets load as before (no threat weights, threat accumulator stays zero).
v9 nets require the threat inference code but gracefully degrade if loaded on old binary
(version check rejects with clear error).

## Implementation Plan

### Branch strategy

All work on a single feature branch `feature/threat-inputs`. Merge to main only after
end-to-end validation (training → conversion → inference → check-net passes).

### Phase breakdown with estimated effort

| Phase | What | Effort | Dependencies | Testable independently? |
|-------|------|--------|-------------|------------------------|
| 1a | Threat index computation | 2-3 days | None | Yes — unit tests, compare vs Reckless |
| 1b | Threat enumeration for a position | 1 day | 1a | Yes — dump features, verify counts |
| 2a | Threat accumulator (full recompute only) | 2-3 days | 1b | Yes — check-net with random weights |
| 2b | Forward pass integration (PSQ + threat sum) | 1 day | 2a | Yes — verify eval changes with threat weights |
| 2c | Incremental threat updates | 3-5 days | 2a | Yes — compare incremental vs full recompute |
| 2d | SIMD optimisation | 2-3 days | 2c | Yes — NPS benchmarks |
| 3 | Bullet trainer fork | 3-5 days | 1a (for feature encoder) | Yes — train tiny net, verify gradients |
| 4 | Converter + file format | 1-2 days | 3 | Yes — round-trip test |
| 5 | First real training run + evaluation | 1-2 days | 3, 4 | H2H vs non-threat net |

**Total: ~3-4 weeks**, with testable milestones every 2-3 days.

### Critical path

Phase 3 (Bullet fork) is the highest-risk item. If Bullet modifications prove harder than
expected, this blocks training. Mitigation: start Phase 3 early in parallel with Phase 1.

### Testing strategy

1. **Phase 1 validation**: Dump threat features for 10 standard positions. Manually verify
   a few (e.g., "e4 pawn should have threats on d5/f5 if occupied"). Cross-check feature
   count against Reckless for the same positions.

2. **Phase 2 validation**: Load a net with random threat weights. Verify eval differs from
   zero-threat eval. Make/unmake moves, verify incremental accumulator matches full
   recompute. Fuzz test with 10K random positions.

3. **Phase 3 validation**: Train tiny net (FT=32, 100 threats, 100 SBs). Verify loss
   decreases. Check that threat weight gradients are non-zero. Then train full-size net
   (FT=1536, 67K threats, s100) and run check-net.

4. **End-to-end**: Convert trained net, load in engine, run check-net (all 8 tests pass),
   bench (NPS reasonable), H2H vs best non-threat v7/v8 net.

### Naming convention

```
net-v9-768th16x32-wNN-eNNNsNNN.nnue
```
768 = accumulator width (matching Reckless), `t` = threat features, version v9.
No `pw` suffix needed — pairwise is implied by the 768 accumulator with hidden layers.

### Open questions

1. ~~**Accumulator width**~~ DECIDED: 768, matching Reckless. Halves FT/L1 cost, proven
   architecture, 50MB threat weights instead of 100MB.

2. **Which piece-pair interactions to include?** Start with Reckless's map (proven) or
   start with a simpler subset (e.g., only enemy attacks, not defenses)?

3. **Threat feature count**: Match Reckless (~67K) or start smaller (~30K with fewer
   piece-pair types) for faster iteration?

4. **X-ray attacks**: Discovered attacks through moved pieces. Reckless tracks these.
   Add from the start or simplify the first version?

5. **Dual L1 activation**: Do we also include dual activation (v8) in the v9 target,
   or keep it simpler? Reckless uses CReLU on L1, not dual. Recommend matching
   Reckless first, add dual later if needed.

6. **King bucket layout**: Use consensus fine-near/coarse-far or Reckless's 10-bucket
   layout? Recommend our consensus 16-bucket layout (already implemented) unless
   Reckless's specific layout proves important.

## Recommended decomposition for first iteration

For the fastest path to a testable net:

1. **Start with ALL piece-pair interactions** (match Reckless). The interaction map is
   simple to implement and reducing features means less signal for hidden layers.

2. **Skip x-ray/discovery threats initially**. Only direct attacks (piece's attack
   bitboard & occupied). This simplifies incremental updates significantly — we only
   need to track the moved piece's attacks and attacks on the from/to squares.

3. **Full recompute only** for the first training run. Incremental updates are an NPS
   optimisation — the eval quality comes from the features themselves. We can train and
   test with full recompute, then add incremental updates for production NPS.

4. **Match Reckless's encoding exactly** for the feature indices. This lets us validate
   against their implementation and avoids encoding bugs that could waste a training run.
