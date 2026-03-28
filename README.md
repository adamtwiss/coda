# Coda

**Chess Optimised, Developed Agentically** — a UCI chess engine written in Rust.

Coda is a rewrite of [GoChess](https://github.com/adamtwiss/gochess) in Rust, built through human-AI collaboration using Claude Code. The entire engine — board representation, search, NNUE evaluation, SIMD, and UCI protocol — was written from scratch in a single session.

## Features

- **NNUE evaluation** — HalfKA network with lazy accumulator and Finny table caching
- **AVX2 / AVX-512 SIMD** — runtime auto-detected for NNUE inference
- **Full search** — alpha-beta with 20+ pruning features (NMP, LMR, RFP, ProbCut, etc.)
- **Polyglot opening book** — weighted random selection from .bin book files
- **Magic bitboards** — with PEXT runtime detection on BMI2 hardware

## Build

```bash
cargo build --release
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
./target/release/coda epd testdata/wac.epd 1000 -nnue net.nnue

# Perft verification
./target/release/coda perft-bench

# Help
./target/release/coda help
```

## UCI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| Hash | spin (1-4096) | 64 | Transposition table size in MB |
| NNUEFile | string | | Path to .nnue network file |
| OwnBook | check | true | Use opening book |
| BookFile | string | | Path to Polyglot .bin book |
| MoveOverhead | spin (0-5000) | 100 | Communication latency in ms |
| Ponder | check | false | Enable pondering |

## Strength

Competitive with Crafty, ExChess, and GreKo (~3000 Elo estimate based upon CCRL ratings for those engines). Uses the same NNUE networks as GoChess (v5/v6 format, CReLU or SCReLU activation).

## License

MIT
