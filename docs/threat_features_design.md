# Threat Features Design Doc

## Status: IMPLEMENTED AND VALIDATED (2026-04-17)

v9 architecture with threat features (including x-rays) is shipped on
`feature/threat-inputs`. Key milestones:

- **Inference & training**: complete. Converter, SIMD paths, incremental
  deltas, Finny table, king-bucket mirroring all working.
- **Correctness**: 16-scenario regression test suite in
  `src/threat_accum.rs::incremental_tests` enforces incremental delta path
  matches full refresh for every slider geometry change (including x-rays,
  captures that reveal x-rays, and unrelated moves that shift sliders'
  x-ray targets).
- **Strength**: x-ray-fixed v9 at s200 is **+110 Elo H1** vs non-x-ray
  baseline (test #407, 2026-04-17). Represents a ~200 Elo swing from the
  pre-fix regression (-60 to -85) and validates the entire threat pipeline.
- **Scale**: 66,864 threat features, i8 weights at QB=64, 768 accumulator
  matching Reckless. Our x-ray enumeration is 2-deep (vs Reckless's direct-only)
  — this is the one architectural choice that differs from Reckless's
  reference implementation.

Sections below describe the design as of implementation. The "Lessons
Learned" and "Open Questions" sections at the bottom reflect what was
actually discovered during build-out.

## Overview

Add ~67K threat input features to Coda's NNUE, following the Reckless implementation
pattern. Threat features encode (attacker, attacker_sq, victim, victim_sq) relationships
— which pieces attack which pieces on which squares. These are fed into the FT accumulator
via a separate i8 weight matrix, summed with the existing i16 PSQ accumulator at activation
time.

**Initial expected gain**: +40-60 Elo of eval quality, based on comparison of threat vs non-threat
engines at similar FT widths.

**Realized gain** (2026-04-17): +110 Elo H1 at s200 with x-ray-fixed inference
(test #407). Expected to grow further with s800 training, SPSA retune on the
fixed eval, and stacked architecture wins (no-ply-filter + Reckless 10 KB).

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

### Finny table / caching

The threat accumulator needs invalidation when the king changes bucket (same as PSQ
accumulator). Reckless does NOT use a Finny table for threats — it does full recompute
on king bucket change. We should match this initially.

When a king moves:
- PSQ Finny table handles the PSQ accumulator recompute (existing)
- Threat accumulator does full recompute (new — zero + iterate all threats)
- Both accumulators must be invalidated together

The PSQ and threat accumulators must stay synchronised — if PSQ is accurate for a
position, threat must be too. The `accurate` flags should be tracked independently per
perspective per accumulator type.

### Deliverable

- Threat accumulator struct with push/pop semantics matching existing NNUE accumulator
- SIMD update functions (AVX2: `_mm256_cvtepi8_epi16` + `_mm256_add_epi16`)
- Integration into `forward_with_l1` / `forward_with_l1_pairwise` paths
- Full recompute function for positions without incremental history
- Invalidation on king bucket changes (coordinated with PSQ Finny table)

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

**Parallel tracks**: Phase 1 (inference) and Phase 3 (Bullet) should run in parallel.
Phase 3 is the highest-risk item and the threat feature encoder for Bullet doesn't
depend on the inference-side code — only the index computation (Phase 1a) is shared.

| Phase | What | Effort | Dependencies | Testable independently? |
|-------|------|--------|-------------|------------------------|
| **Track A: Inference** | | | | |
| 1a | Threat index computation | 2-3 days | None | Yes — unit tests, compare vs Reckless |
| 1b | Threat enumeration for a position | 1 day | 1a | Yes — dump features, verify counts |
| 2a | Threat accumulator (full recompute) + Finny | 2-3 days | 1b | Yes — check-net with random weights |
| 2b | Forward pass integration (PSQ + threat sum) | 1 day | 2a | Yes — verify eval changes |
| 2c | Incremental threat updates | 3-5 days | 2a | Yes — compare vs full recompute |
| 2d | SIMD optimisation | 2-3 days | 2c | Yes — NPS benchmarks |
| **Track B: Training** | | | | |
| 3 | Bullet trainer fork | 5-7 days | 1a (shared encoder) | Yes — train tiny net, verify gradients |
| **Integration** | | | | |
| 4 | Converter + file format | 1-2 days | 3 | Yes — round-trip test |
| 5 | First training run + evaluation | 1-2 days | 2b, 3, 4 | check-net, fixed-node H2H |
| 5b | Time-controlled H2H (needs incremental) | 1 day | 2c, 5 | SPRT at real TC |

**Total: ~4-5 weeks**, with testable milestones every 2-3 days.
Budget extra time for debugging — the first training run will likely need 2-3 iterations
to fix encoding bugs, weight init issues, or gradient problems.

**Fallback for Phase 3**: If Bullet modifications stall, a simple PyTorch training script
can validate the concept. Even a slow CPU trainer is fine for producing a test net.

### Critical path

```
Phase 1a ──→ Phase 1b ──→ Phase 2a ──→ Phase 2b ──→ Phase 5 (fixed-node)
    │                                                      ↑
    └──→ Phase 3 (Bullet fork, parallel) ──→ Phase 4 ─────┘
                                                           ↓
                                        Phase 2c ──→ Phase 5b (time-controlled)
```

Phase 3 and Phase 1/2 run in parallel. First testable net at Phase 5 (full recompute,
check-net + fixed-node only). Real strength testing at Phase 5b after incremental updates.

### NPS expectations

- **Full recompute** (Phase 2a-2b): NPS will be poor (~50-100K estimated). Computing
  ~60-100 active threats from scratch per eval means ~60-100 random 768-byte reads from
  a 50MB weight table per position. Fine for check-net and fixed-node testing, NOT for
  time-controlled play.
- **With incremental updates** (Phase 2c): NPS should be reasonable (~300-400K estimated).
  Only ~10-30 threat deltas per move, each a single 768-byte weight row add/sub.
  The 768 accumulator (half our current 1536) also speeds up FT and L1.
- **With SIMD optimisation** (Phase 2d): Target ~400K+ NPS.

**Note on all-piece attack computation** (Atlas review point): Threat feature computation
needs attack bitboards for ALL pieces on the board, not just side to move. This means
~32 magic lookups per eval in the full recompute path. We already have the infrastructure
(attacks.rs) but the per-eval cost should be benchmarked in Phase 1b.

### Testing strategy

1. **Phase 1 validation**: Dump threat features for 10 standard positions. Manually verify
   a few (e.g., "e4 pawn should have threats on d5/f5 if occupied"). Cross-check feature
   count against Reckless for the same positions if possible.

2. **Phase 2 validation**: Load a net with random threat weights. Verify eval differs from
   zero-threat eval. Make/unmake moves, verify incremental accumulator matches full
   recompute. Fuzz test with 10K random positions. Verify Finny table invalidation on
   king moves that cross the e-file mirror boundary.

3. **Phase 3 validation**: Train tiny net (FT=32, 100 threats, 100 SBs). Verify loss
   decreases. Check that threat weight gradients are non-zero. Then train full-size net
   (FT=768, 67K threats, s100) and run check-net.

4. **End-to-end (fixed-node)**: Convert trained net, load in engine, run check-net (all 8
   tests pass), fixed-node H2H vs best non-threat v7/v8 net. This validates eval quality
   WITHOUT needing good NPS.

5. **Time-controlled (after incremental)**: SPRT at real TC once incremental updates are
   working. This is the real strength test.

### Naming convention

```
net-v9-768pwth16x32-wNN-eNNNsNNN.nnue
```
768 = accumulator width (matching Reckless), `pw` = pairwise, `t` = threat features.
Keep `pw` suffix for clarity since pairwise is an architectural choice, not implied by width.

### Open questions (resolved)

1. ~~**Accumulator width**~~ → **DECIDED: 768**, matching Reckless.

2. ~~**Which piece-pair interactions**~~ → **DECIDED: full Reckless map**.
   Interaction map is at `src/threats.rs:18-25`. Pawn→{P,N,R} only (B/Q/K excluded),
   Rook→{P,N,B,R} (Q/K excluded), etc. 66,864 total features.

3. ~~**Threat feature count**~~ → **DECIDED: 66,864** (matches Reckless exactly).

4. ~~**X-ray attacks**~~ → **DECIDED: included, 2-deep** (the one place we
   diverge from Reckless). For each slider's direct target, we also enumerate
   the next piece on the ray behind it. The incremental delta path handles
   three distinct cases: x-rays from the moving piece, x-rays from other
   sliders whose direct target is the subject square, and x-rays from sliders
   whose x-ray target is the subject square (through one blocker). Bugs in
   each of these three cases cost ~200 Elo before being fixed on 2026-04-17.

5. ~~**Dual L1 activation**~~ → **DECIDED: no dual**. v9 uses single SCReLU
   on L1/L2 (matching Reckless's shape, not their activation — Reckless uses
   clipped ReLU `clamp(0,1)` on L1/L2, which is still an open experiment for us).

6. **King bucket layout**: still the main open architectural question.
   Currently using 16 uniform buckets (default) or 16 consensus fine-near /
   coarse-far (toggleable via `init_king_buckets(true)`, trained on v5 for
   +15 Elo but not yet re-tested on v9). Reckless uses 10 with aggressive
   far-rank merging. Queued as T1 (consensus 16) and T2 (Reckless 10) in the
   training experiment queue. See `docs/v7_training_guide.md` T1-T2.

## Lessons Learned (2026-04-17)

### The delta path must match full refresh to the feature, not just the sum

The implementation was initially "tested" by running 16 random moves and
comparing the overall accumulator sum to a full recompute. That test
passed — but 8 out of 12 targeted scenarios (in `threat_accum.rs::incremental_tests`)
fail on that code. The random test missed three distinct bugs:

1. The moving slider's x-rays from its new position were never emitted.
2. Section 2's x-ray delta targeted the *unchanged* feature (first piece on
   the ray past the appearing/disappearing piece — same feature index in
   direct and x-ray form).
3. Sliders whose x-ray target is the changed square (but whose direct target
   is elsewhere) were entirely missed.

**Takeaway**: For any incremental accumulator, the test must be
per-feature-activation element-wise, not a summed scalar. A scalar test
can't distinguish "features A and B active" from "features C and D active"
if A+B = C+D numerically. Random move fuzzing is not a substitute for
handcrafted scenarios that exercise each delta-path code branch.

### Training-inference pair validation

When training and inference diverge silently, the symptoms look like "the
trainer makes bad models". We spent hours blaming training instability
(s800 regressed -50, x-ray nets regressed -60 to -85) before realising the
inference path was corrupting the accumulator on every capture. Cross-check
training-vs-inference feature indices on test positions EARLY — don't let
a training run finish before confirming the two paths agree on multiset
feature activation.

### The "brittle training" hypothesis was wrong

The pre-fix data suggested training was unstable (s800 loss went up after
SB 720, x-ray nets regressed). Post-fix, the data shows training is fine
and x-rays add +110 Elo. Multiple "training is fragile" hypotheses were
actually masking one structural inference bug. When several experiments
all regress in correlated ways, suspect a shared dependency before
concluding the training pipeline is unstable.

### Scalar benchmarks don't catch feature-level bugs

The scale bug fix (QA=255→QA=64 converter rescale) and the x-ray delta bug
were both inference-side correctness issues that no training-time metric
could expose. For any future NNUE feature:

1. Unit-test inference correctness per-feature before shipping.
2. Cross-check feature indices against reference implementation on curated
   positions.
3. When inference seems fine but strength regresses, assume an inference
   bug before retraining.

## Recommended decomposition for first iteration

For the fastest path to a testable net:

1. **Start with ALL piece-pair interactions** (match Reckless). The interaction map is
   simple to implement and reducing features means less signal for hidden layers.

2. **Skip x-ray/discovery threats initially**. Only direct attacks (piece's attack
   bitboard & occupied). This simplifies incremental updates significantly — we only
   need to track the moved piece's attacks and attacks on the from/to squares.

3. **Full recompute for training** (GPU doesn't care about NPS). For inference validation,
   full recompute is fine for check-net and fixed-node H2H. But implement incremental
   updates BEFORE any time-controlled testing — full recompute will give ~50-100K NPS
   which is too slow for meaningful SPRT.

4. **Match Reckless's encoding exactly** for the feature indices. This lets us validate
   against their implementation and avoids encoding bugs that could waste a training run.

5. **Start Phase 3 (Bullet fork) immediately** in parallel with Phase 1. It's the
   highest-risk item and doesn't depend on inference code. The threat feature encoder
   only needs the index computation from Phase 1a.
