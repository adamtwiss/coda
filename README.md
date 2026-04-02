# Coda

**Chess Optimised, Developed Agentically** — a UCI chess engine written in Rust.

Coda is a rewrite of [GoChess](https://github.com/adamtwiss/gochess) in Rust, built through human-AI collaboration using Claude Code. The entire engine — board representation, search, NNUE evaluation, SIMD, and UCI protocol — was written from scratch in a single session.

## Features

- **NNUE evaluation** — HalfKA network with lazy accumulator and Finny table caching
- **AVX2 / AVX-512 SIMD** — runtime auto-detected for NNUE inference
- **Full search** — alpha-beta with 20+ pruning features (NMP, LMR, RFP, singular extensions, etc.)
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
| SparseL1 | check | true | Sparse L1 matmul for v7 hidden layers |
| SyzygyPath | string | | Path to Syzygy tablebase files |

## Strength

In the Winter/Midnight/Weiss/Minic tier (~2800-2900 CCRL estimate). Uses HalfKA NNUE networks (v5/v6/v7 formats, CReLU/SCReLU activation).

## License

MIT
