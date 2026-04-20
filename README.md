# Coda

**Chess Optimised, Developed Agentically** — a UCI chess engine written in Rust.

Coda is a rewrite of [GoChess](https://github.com/adamtwiss/gochess) in Rust, 100% vibe-coded with Claude Code. The initial implememntation - board representation, search, NNUE evaluation, SIMD, and UCI protocol — was written from scratch in a single session - building on top of the work done in gochess.

## Features

- **NNUE evaluation** — HalfKA network, with threats, with lazy accumulator and Finny table caching
- **AVX2 / NEON / AVX-512 SIMD** — runtime auto-detected for NNUE inference
- **Full search** — alpha-beta with 20+ pruning features (NMP, LMR, RFP, singular extensions, etc.)
- **Lazy SMP** — multi-threaded search with shared transposition table
- **Syzygy tablebases** — endgame tablebase probing
- **Polyglot opening book** — weighted random selection from .bin book files
- **Magic bitboards** — with PEXT runtime detection on BMI2 hardware
- **Training data generation** — multi-threaded self-play datagen in SF binpack format

## Build

```bash
make                # Build with native CPU optimizations
make pgo            # PGO-optimized build (~3-5% faster)
make net            # Download production NNUE net
```
All the above will build a 'coda' binary in the current directory.

Alternatively, you can build using cargo:
```
cargo build --release  # Plain release build into target/releases
```

Requires Rust 1.70+ (uses `std::arch` intrinsics).

For PGO builds, install prerequisites:
```bash
rustup component add llvm-tools-preview
cargo install cargo-pgo
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
| SyzygyPath | string | | Path to Syzygy tablebase files |

## Strength

Plays around ~2900 on lichess, competes with most engines ranked around ~3500 on CCRL

## License

MIT
