# Coda

**Chess Optimised, Developed Agentically** — a UCI chess engine written in Rust.

Coda is a strong UCI chess engine, 100% vibe-coded with Claude Code. It started in late January 2026 as [GoChess](https://github.com/adamtwiss/gochess) (similarly vibe-coded) before being rewritten in Rust in late March. I had written a few hobby engines in the past, and I wanted to see how far I could get with a new engine with the help of Claude.

## Features

- **NNUE evaluation** — a from-scratch NNUE whose input layer combines HalfKA-style piece-square features (16 king buckets) with ~67k explicit threat and x-ray attack features — the network sees not just where pieces stand but what they attack — feeding a 1024-wide per-perspective accumulator with pairwise activation, then two small int8 hidden layers (32→32) with material-bucketed output heads. Networks are trained from scratch on a mix of the full LC0 dataset (hundreds of billions of positions) and self-generated data, using a customized Bullet trainer. Inference is fully incremental (lazy accumulator, Finny tables, incremental threat deltas) with runtime-dispatched AVX2/AVX-512-VNNI/NEON SIMD kernels.
- **Search** — principal-variation alpha-beta with iterative deepening and aspiration windows, carrying the full modern battery: null-move pruning with verification, reverse futility (with a depth-aware knee), razoring, futility and late-move pruning, SEE pruning for quiets and captures, ProbCut, internal iterative reductions, late-move reductions shaped by history/complexity/threat signals, and singular extensions with double and negative variants. Repetition handling goes beyond the rules with cuckoo-table upcoming-cycle detection.
- **Move ordering & history** — a staged move picker driven by an unusually rich history stack: threat-aware 4D main history, capture history, four-ply continuation history, and pawn-structure history, plus tactical ordering bonuses (threat escapes, discovered attacks, safe checks). A multi-source **correction history** (pawn / non-pawn / continuation / transition tables) continuously corrects the static eval from search feedback.
- **Multi-threading & transposition table** — Lazy SMP over a lockless, XOR-verified transposition table (5-slot cache-line buckets, huge-page backed). Fully atomic with acquire/release ordering, so SMP is correct on ARM as well as x86 — Apple Silicon and ARM servers are first-class targets.
- **Time management** — an adaptive multi-factor model (per-move node fraction, best-move stability, score trend) with full pondering support.
- **Tablebases and opening books** — Syzygy endgame tablebase probing with a dedicated probe cache, and native Polyglot opening-book support.
- **Training data generation** — multi-threaded self-play and material-imbalance datagen in SF binpack format, plus converters to and from Bullet checkpoint formats.

## Pre-built binaries

As we get closer to a first proper release, we will shortly add pre-built binaries for all major platforms.

## Build from Source

```bash
make                # Downloads the production network, and builds a binary with embedded NNUE net.
```
This builds a `coda` binary in the current directory, targeting the native CPU. (A `make pgo` option exists but currently regresses performance on most modern hardware — avoid it.)

Alternatively, you can build with cargo, but this won't embed an NNUE network (pass one at runtime with `--nnue <file>` or the `NNUEFile` UCI option):
```
cargo build --release  # Plain release build into target/release
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
| TBHash | spin (0-1024) | 16 | Tablebase probe cache size in MB |
| SyzygyProbeDepth | spin (1-100) | 4 | Minimum depth for TB probes during search |

The engine also exposes its ~130 search parameters as UCI options for SPSA tuning; these are not intended for end users.

## Strength

Plays around 3000-3080 on lichess ([coda_bot](https://lichess.org/@/coda_bot) and [codabot](https://lichess.org/@/codabot)), where it is one of the strongest non-Stockfish-derived engines. In local testing it competes with engines rated around 3500-3600 on CCRL, and in our local round-robin pool — which includes the strongest available C/C++/Rust engines — it places top-10 at short time controls. Most performance tuning so far has targeted short time controls. Coda is a young engine evolving fast, so bug reports, testing and feedback are always welcome.

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
