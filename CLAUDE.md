# CLAUDE.md — Coda Chess Engine

Coda is a UCI chess engine written in Rust. Rewritten from GoChess with all accumulated knowledge.

**Chess Optimised, Developed Agentically** — built through human-AI collaboration.

## Build and Test

```bash
cargo build --release
cargo test                           # Run all tests including perft
cargo bench                          # Run benchmarks

./target/release/coda                # UCI mode (default)
./target/release/coda bench          # Deterministic node count + NPS
./target/release/coda bench -classical  # Classical eval only (no NNUE)
```

## Architecture

### Board Representation
- Bitboards only: `pieces: [u64; 6]` (by piece type) + `colors: [u64; 2]` (by color)
- No mailbox/Squares[64] — piece-at-square via bitboard scan when needed
- Magic bitboards for sliding pieces, PEXT on Intel/Zen3+ (runtime detected)
- Incremental Zobrist hashing

### Move Encoding
16 bits: from(6) + to(6) + flags(4). Same encoding as GoChess.
**Critical rule**: Check non-promotion flags with equality (==), not bitwise AND.

### Search
Negamax with alpha-beta, iterative deepening, PVS, aspiration windows.

Features: NMP (with eval >= beta guard, verification at depth >= 12 with ply+1), RFP (depth <= 7), futility (60+d*60), LMR (separate capture/quiet tables, doDeeper/doShallower), LMP, SEE pruning, ProbCut, history pruning (-1500*depth), razoring, hindsight reduction (200cp threshold), recapture extensions, alpha-reduce, failing heuristic, fail-high score blending.

Quiescence with SEE filtering, evasion handling (captHist/16), QS delta (240), beta blending.

Move ordering: TT move → good captures (MVV-LVA + captHist/16) → killers → counter-move → quiets (history + contHist) → bad captures.

History tables: main, capture, continuation (1-ply), pawn history. Multi-source correction history (pawn 512 + white-NP 204 + black-NP 204 + cont 104, /1024).

### TT
- 5-slot buckets, 64 bytes (cache-line aligned)
- 14-bit staticEval (±8191 cp range, NOT int8*4)
- Replacement: d > slotDepth-3 for key match (prevents shallow re-searches overwriting deep)
- Depth-4*age scoring for non-match replacement
- TT score dampening: (3*score+beta)/4 on non-PV TTLower cutoffs
- Fail-high blending: (score*depth+beta)/(depth+1) at non-PV, depth >= 3
- TT near-miss: accept 1-ply-short entries with 80cp margin
- PV guard: skip TTLower/TTUpper bound tightening at PV nodes
- Prefetch: prefetch next TT bucket before move loop

### NNUE
Production: v5/v6 format, 1024 SCReLU, 8 output buckets by material count.
HalfKA features: 16 king buckets × 12 piece types × 64 squares = 12288 inputs.

- Lazy accumulator (MakeMove stores deltas, Materialize applies on demand)
- Finny table for king bucket refresh (diff cached vs current bitboards)
- SIMD via Rust intrinsics: `std::arch::x86_64` for AVX2, `std::arch::aarch64` for NEON
- CReLU: clamp [0, QA=255], VPMADDWD dot product
- SCReLU: clamp then square, byte decomposition for VPMADDWD compatibility
- Quantization: QA=255 (accumulator), QB=64 (output weights)

### Validation Methodology
**Three-tier (never rely on self-play alone):**
1. **Self-play SPRT** — fast fail, pass even if mildly negative (down to -10)
2. **Rival engine gauntlet** (6-8 engines near our Elo) — acceptance criterion
3. **Broad RR** (10+ engines) — production validation

Self-play SPRT gives contradictory signals to cross-engine. Bug fixes can show -9 self-play but +12 cross-engine. Always run Tier 2.

### NNUE Training (Bullet GPU)
- Bullet trainer (Rust, CUDA) on T80 binpack data (12 files, 2023+2024)
- SCReLU activation, wdl=0.05 for near-zero resolution
- Warmup LR for hidden layers: 5 SB ramp 0.0001→0.001, then cosine decay
- Score filter: unsigned_abs() < 10000
- Monitor neuron health at every checkpoint
- CReLU kills hidden layer neurons — always use SCReLU for multi-layer

### Key Search Parameters (tuned, do not change without cross-engine validation)
- NMP: R=3+depth/3, divisor 200, verify depth >= 12 with ply+1
- RFP: depth <= 7, margin 70+100*improving
- Futility: 60+lmrDepth*60, depth <= 8
- LMR: quiet C=1.30, capture C=1.80
- History pruning: -1500*depth
- SEE quiet: -20*depth²
- SEE capture: -depth*100
- QS delta: 240
- Aspiration: delta=15
- Hindsight: 200cp threshold
- Correction history: clamp ±128, depth >= 3 gate
