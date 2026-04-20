# Coda

**Chess Optimised, Developed Agentically** — a UCI chess engine written in Rust.

Coda is a rewrite of [GoChess](https://github.com/adamtwiss/gochess) in Rust, built through human-AI collaboration using Claude Code. The entire engine — board representation, search, NNUE evaluation, SIMD, and UCI protocol — was written from scratch in a single session.

## Features

- **NNUE evaluation** — HalfKA network with lazy accumulator and Finny table caching. Supports v5 (direct FT→output), v6 (SCReLU+pairwise), v7 (hidden layers), v8 (dual L1 activation), v9 (threat features).
- **AVX2 / AVX-512 SIMD** — runtime auto-detected for NNUE inference
- **Full search** — alpha-beta with 25+ pruning features (NMP, LMR, RFP, singular extensions, ProbCut, futility, LMP, SEE pruning, history pruning, cuckoo cycles, hindsight reduction, recapture extensions, etc.) — all SPSA-tunable via the `tunables!` macro
- **Lazy SMP** — multi-threaded search with shared transposition table
- **Syzygy tablebases** — endgame tablebase probing
- **Cuckoo cycle detection** — proactive repetition avoidance
- **Polyglot opening book** — weighted random selection from .bin book files
- **Magic bitboards** — with PEXT runtime detection on BMI2 hardware
- **Training data generation** — multi-threaded self-play datagen in SF binpack format

## Build

```bash
make                # Build with native CPU optimizations
make pgo            # PGO-optimized build (~5% faster)
make net            # Download production NNUE net
cargo build --release  # Plain release build
```

Requires Rust 1.70+ (uses `std::arch` intrinsics).

For PGO builds, install prerequisites:
```bash
rustup component add llvm-tools-preview
cargo install cargo-pgo
```

## Run

```bash
# UCI mode (default)
./target/release/coda

# With NNUE evaluation (recommended)
./target/release/coda -nnue path/to/net.nnue

# With NNUE + opening book
./target/release/coda -nnue net.nnue -book book.bin

# Search benchmark
./target/release/coda bench 13 -nnue net.nnue

# EPD test suite
./target/release/coda epd testdata/wac.epd 1000 0 -nnue net.nnue

# Perft verification
./target/release/coda perft-bench

# Download net from net.txt URL
./target/release/coda fetch-net

# Help
./target/release/coda help
```

## UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin (1-4096) | 64 | Transposition table size in MB |
| Threads | spin (1-256) | 1 | Lazy SMP thread count |
| NNUEFile | string | | Path to .nnue network file |
| OwnBook | check | true | Use opening book |
| BookFile | string | | Path to Polyglot .bin book |
| MoveOverhead | spin (0-5000) | 100 | Communication latency in ms |
| Ponder | check | false | Enable pondering |
| SparseL1 | check | true | Sparse L1 matmul for v7/v8/v9 hidden layers |
| HiddenActivation | combo (screlu, crelu) | screlu | Hidden-layer activation for v7+ nets |
| SyzygyPath | string | | Path to Syzygy tablebase files |

Plus 60+ tunable search parameters exposed as spin options for SPSA.

## Strength

Deployed on Lichess as `codabot`, currently ~2900 Lichess rating. In
local round-robin testing Coda matches engines rated ~3500 CCRL (Lichess
and CCRL scales are not directly comparable — Lichess ratings are
compressed at the top). v9 development branch (threat features + hidden
layers) under active SPRT/retune. Uses HalfKA NNUE networks (v5-v9
formats, CReLU or SCReLU activation).

## License

MIT
