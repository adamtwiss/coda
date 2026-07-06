# Coda

**Chess Optimised, Developed Agentically** — a UCI chess engine written in Rust.

Coda is a strong chess engine written in Rust, it is 100% vibe-coded with Claude Code. Coda started off in late Jan 2026 as [GoChess](https://github.com/adamtwiss/gochess) (similarly vibe-coded) before being rewritten in Rust in late-March. I had written a few hobby engines in the past, and I wanted to see how far I could get with a new engine with the help of Claude.

## Features

- **NNUE evaluation** - Coda evaluates with a from-scratch NNUE whose input layer combines HalfKA-style piece-square features (16 king buckets) with ~67k explicit threat and x-ray attack features — the network sees not just where pieces stand but what they attack — feeding a 1024-wide per-perspective accumulator with pairwise activation, then two small int8 hidden layers (32→32) with material-bucketed output heads. Networks are trained from scratch on mix of full set of LC0 data (100s of billions of positions) along with self-generated data with a customized Bullet trainer. Inference is fully incremental (lazy accumulator, Finny tables, incremental threat deltas) with runtime-dispatched AVX2/AVX-512-VNNI/NEON SIMD kernels.
- **Full search** - alpha-beta with complete set of modern pruning features (NMP, LMR, RFP, singular extensions, etc.). Search supports pondering, and multi-threaded search with shared transposition table.
- **Tablebases and opening books** - Syzygy endgame tablebase probing, and native support for polyglot opening books.
- **Training data generation** - multi-threaded self-play datagen in SF binpack format.

## Pre-built binaries

As we get closer to a first proper release, we will shortly add pre-built binaries for all major platforms.

## Build from Source

```bash
make                # Downloads the production network, and builds a binary with embedded NNUE net.
```
All the above will build a 'coda' binary in the current directory. There is a pgo build option, but this currently regresses performance on most hardware, so please avoid using.

Alternatively, you can build using cargo, but this won't embed an NNUE network.
```
cargo build --release  # Plain release build into target/releases
```

Requires Rust 1.70+

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

Plays around 3000-3080 on lichess ([coda-bot](https://lichess.org/@/coda_bot) and [codabot](https://lichess.org/@/codabot)) where it is one of the strongest non-SF engines. In local testing competes with most engines ranked around 3500-3600 on CCRL. In local RR, STC testing it's a top-10 engine. So far most most performance tuning has been done for short time controls. Coda is still a young engine that is evolving fast, so bug-reports, testing and feedback are always welcome. 

## Credits

Coda development has been made much easier by many other projects - most notably the [OpenBench](https://github.com/andygrant/openbench) distributed engine-testing framework, the [Bullet](https://github.com/jw1912/bullet) NNUE trainer and [LC0 training data](https://github.com/LeelaChessZero/lczero-training). We're grateful to the authors/maintainers of these projects along with the wider chess community.

## License

**GPL-3.0-or-later** — see [LICENSE](LICENSE).

Coda was briefly labelled MIT in this repo's early days, which was likely
accurate for the code as it stood then. Our dependencies have drifted since:
after auditing them (July 2026) the engine now links several GPLv3 libraries
(the shakmaty family for Syzygy tablebase probing and PGN handling, and
sfbinpack), so binaries can only be distributed under GPL terms — and rather
than carry a split notice, we think GPL is simply the better license for
Coda: it matches the engine ecosystem we learn from and contribute back to.
Thanks to Disservin for pointing out the mismatch.
