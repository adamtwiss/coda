# CLAUDE.md — Coda Chess Engine

Coda is a UCI chess engine written in Rust. Rewritten from GoChess with all accumulated knowledge.

**Chess Optimised, Developed Agentically** — built through human-AI collaboration.

## Build and Test

```bash
cargo build --release
cargo test                                      # Run all tests including perft

./target/release/coda                            # UCI mode (default)
./target/release/coda -nnue net.nnue             # UCI with NNUE evaluation
./target/release/coda -nnue net.nnue -book book.bin  # UCI with NNUE + opening book
./target/release/coda bench [depth]              # Search benchmark (default depth 13)
./target/release/coda bench 13 -nnue net.nnue    # Bench with NNUE
./target/release/coda epd wac.epd 1000 0 -nnue net.nnue  # EPD test suite
./target/release/coda perft [depth] [fen...]     # Perft with divide
./target/release/coda perft-bench                # Perft benchmark (6 positions)
./target/release/coda datagen [options]          # Generate training data (SF binpack)
./target/release/coda convert-bullet [options]   # Bullet quantised.bin → .nnue
./target/release/coda convert-checkpoint [opts]  # .nnue → Bullet checkpoint (transfer learning)
./target/release/coda check-net <net.nnue>       # NNUE health check (TODO)
./target/release/coda help                       # Show all options
```

## Project Structure

```
src/
  main.rs          Entry point, CLI argument parsing, subcommands
  board.rs         Board struct (bitboards + mailbox), FEN, make/unmake, Zobrist
  types.rs         Color, Piece, Square, Move encoding (16-bit), castling
  bitboard.rs      Bitboard ops, between/line tables
  attacks.rs       Magic bitboards (PEXT runtime detected), knight/king/pawn tables
  movegen.rs       Pseudo-legal + capture-only move generation, perft
  zobrist.rs       Zobrist hash keys (deterministic PRNG)
  eval.rs          PeSTO material+PST eval (fallback), SEE values, NNUE eval wrapper
  see.rs           Static Exchange Evaluation
  tt.rs            Transposition table (4-slot buckets, XOR key verification)
  movepicker.rs    Staged move ordering, history tables, killer/counter moves
  search.rs        Negamax, pruning, LMR, correction history, pruning stats
  nnue.rs          NNUE v5/v6/v7 inference, accumulator stack, Finny table, AVX2/AVX-512 SIMD
  uci.rs           UCI protocol (position, go, stop, ponder, setoption)
  epd.rs           EPD test suite runner with SAN formatting
  book.rs          Polyglot opening book support
  polyglot_randoms.rs  Standard Polyglot Zobrist random table (781 entries)
  datagen.rs       Multi-threaded training data generation (self-play, material removal)
  binpack.rs       SF BINP binpack format writer (chain-compressed)
  bullet_convert.rs  Bullet quantised.bin → .nnue converter (v5/v6/v7)
  nnue_export.rs   .nnue → Bullet checkpoint converter (for transfer learning)
testdata/
  wac.epd          Win At Chess test suite (201 positions)
```

## Architecture

### Board Representation
- Bitboards: `pieces: [u64; 6]` (by type) + `colors: [u64; 2]` (by color)
- Mailbox: `mailbox: [u8; 64]` for O(1) piece-at-square lookup
- Magic bitboards for sliding pieces, PEXT on BMI2 hardware (runtime detected)
- Incremental Zobrist hashing + incremental pawn hash

### Move Encoding
16 bits: from(6) + to(6) + flags(4). Flags: None=0, EP=1, Castle=2, PromoteN=4..PromoteQ=7. Double push has no flag (FLAG_DOUBLE_PUSH=0), detected by distance in make_move.
**Critical rule**: Check non-promotion flags with equality (==), not bitwise AND.

### Search
Negamax with alpha-beta, iterative deepening, PVS, aspiration windows (from depth 4). Lazy SMP: helper threads search at offset depths sharing the TT (atomic) and stop flag.

**Pruning features:**
- NMP: R=3+depth/3 + (eval-beta)/200, eval>=beta guard, verify at depth>=12, post-capture R--, NMP score dampening (score*2+beta)/3
- RFP: depth<=7, margin improving?70*d:100*d, returns staticEval-margin
- Futility: 60+lmrDepth*60, depth<=8, uses estimated LMR depth
- LMR: separate quiet (C=1.30) and capture (C=1.80) tables, doDeeper/doShallower
- LMP: non-PV only, depth<=8, threshold 3+d² with improving/failing adjustments
- SEE pruning: quiet -20d² at depth<=8, capture -d*100 at depth<=6
- ProbCut: beta+170, staticEval+85 gate, SEE>=0
- History pruning: -1500*depth at depth<=3, !improving guard
- Bad noisy: prune losing captures when eval+depth*75<=alpha
- Razoring: depth<=2, margin 400+d*100
- IIR: depth>=6, !inCheck, no TT move
- Recapture extensions
- Fail-high score blending: (score*depth+beta)/(depth+1) at non-PV

**Move ordering:** TT move → good captures (MVV-LVA + captHist/16) → promotions → killers → counter-move → quiets (main hist + contHist*3 + pawn hist) → bad captures

**Exemptions:** TT move and killers exempt from all pre-make pruning and from LMR. Promotions exempt from LMR.

**History tables:** main [color][from][to], capture [piece][to][victim], continuation [piece][to][piece][to], pawn [pawnHash%512][piece][to], killers [ply][2], counter [piece][to].

**Correction history:** Multi-source static eval correction. Pawn (512/1024) + white-NP (204/1024) + black-NP (204/1024) + continuation (104/1024). Updated on bestScore > originalAlpha with depth >= 3.

**Other:** Contempt = -10 (prefer playing over drawing). Insufficient material detection (KvK, KNvK, KBvK). Repetition detection via undo stack hash comparison.

### TT
- 5-slot buckets, 64 bytes (cache-line aligned), AtomicU64/AtomicU32 for lockless Lazy SMP
- XOR key verification: `key_xor = hash ^ data` (detects torn reads from concurrent writes)
- 14-bit staticEval (±8191 cp range)
- Replacement: d > slotDepth-3 for same-gen key match; always replace if generation differs
- TT score dampening: (3*score+beta)/4 on non-PV TTLower cutoffs
- TT near-miss: accept 1-ply-short entries with 80cp margin (else-if, not unconditional)
- Fail-high blending: (score*depth+beta)/(depth+1) at non-PV, depth >= 3
- QS TT probe with cutoffs
- Stores raw (uncorrected) static eval to avoid double correction

### NNUE
Supports v5 (CReLU), v6 (SCReLU, pairwise), and v7 (hidden layers) formats.
Strongest cross-engine: 768pw w5 (`net-v5-768pw-w5-e800-s800.nnue`). Strongest self-play: 1024s w5 (`net-v5-1024s-w5-e800-s800.nnue`).

HalfKA features: 16 king buckets × 12 piece types × 64 squares = 12288 inputs.
Quantization: QA=255 (accumulator), QB=64 (output weights).

- **Lazy accumulator**: push stores DirtyPiece info, materialize on demand (saves work for pruned nodes)
- **Finny table**: per-perspective, per-bucket cache. On king bucket change, diffs cached vs current bitboards (~5 delta ops vs ~30 full recompute)
- **SIMD**: AVX2 and AVX-512 via `std::arch::x86_64` (runtime detected). Int8 weight quantization for SCReLU forward pass.
- **CReLU**: clamp [0, 255], VPMADDWD dot product
- **SCReLU**: clamp [0, 255], square, int8 byte decomposition for VPMADDUBSW. Scale correction ×0.8 for search threshold compatibility.
- **Pairwise**: split accumulator halves, CReLU-clamp, multiply pairs. SIMD byte decomposition like SCReLU.
- **v7 hidden layers**: SCReLU pack to uint8, int8 L1 matmul via VPMADDUBSW, float L2→output. 673K NPS (12% faster than GoChess).
- **Fused accumulator update**: copy + delta in single pass for incremental updates
- **TT prefetch**: prefetch TT bucket after make_move, before child node TT probe

### Opening Book
Polyglot .bin format. Weighted random selection. Polyglot Zobrist hashing with standard 781-entry random table. Castling encoded as king-to-rook, converted to king-to-destination. EP hash only when capture is actually possible.

### UCI Options
- `Hash` (spin, 1-4096, default 64) — TT size in MB
- `Threads` (spin, 1-256, default 1) — Lazy SMP thread count
- `NNUEFile` (string) — path to .nnue network file
- `OwnBook` (check, default true) — use opening book
- `BookFile` (string) — path to Polyglot .bin book
- `MoveOverhead` (spin, 0-5000, default 100) — communication latency in ms
- `Ponder` (check, default false)
- `SparseL1` (check, default false) — sparse L1 matmul for v7 (experimental)
- `SyzygyPath` (string) — path to Syzygy tablebase files

### Time Management
- Soft allocation: timeLeft/movesLeft + 80% of increment
- MoveOverhead subtracted from available time
- Emergency mode below 1 second: cap at timeLeft/10
- Default 25 moves left for sudden death
- Cap at 50% remaining time

## NNUE Training (Bullet GPU)

We train on **Bullet** (Rust, CUDA, fork: `adamtwiss/bullet`) using T80 binpack data (~30B positions across 12 files). Training produces `quantised.bin` which is converted to `.nnue` via `coda convert-bullet`.

### GPU Host Setup

All GPU hosts should use the forked Bullet trainer:
```bash
git clone git@github.com:adamtwiss/bullet.git
cd bullet && cargo build --release
```

**Output directory**: All trained models and checkpoints go in `coda/nets/`:
```
coda/nets/
  v5-1024s-w5-e800/           # training run directory
    quantised-s100.bin         # checkpoint at SB 100
    quantised-s400.bin         # checkpoint at SB 400
    quantised-s800.bin         # final checkpoint
    net-v5-1024s-w5-e800s800.nnue   # converted .nnue
  v7-1024h16x32s-w0-e800/     # v7 training run
    ...
```

Create the directory if it doesn't exist: `mkdir -p coda/nets`

### Model Conversion

Convert Bullet output to .nnue format (run on the GPU host after training):
```bash
# v5 (no hidden layers)
coda convert-bullet -input quantised.bin -output net.nnue -screlu

# v5 pairwise
coda convert-bullet -input quantised.bin -output net.nnue -pairwise

# v7 (hidden layers)
coda convert-bullet -input quantised.bin -output net.nnue -screlu -hidden 16 -hidden2 32 -int8l1
```

### v7 Transfer Learning (Frozen FT)

To train v7 hidden layers using pre-trained v5 FT weights:
```bash
# 1. Convert best v5 .nnue to Bullet checkpoint
coda convert-checkpoint -nnue net-v5-1024s-w5-e800s800.nnue -output v7_checkpoint -l1 16 -l2 32

# 2. In Bullet training config, load checkpoint and freeze FT:
#    trainer.load_from_checkpoint("v7_checkpoint");
#    trainer.optimiser.freeze("l0w");
#    trainer.optimiser.freeze("l0b");
#    // After 50 SBs, unfreeze in superbatch_callback
```

The `adamtwiss/bullet` fork adds `freeze()`/`unfreeze()` support for per-weight training control.

### Training Data Generation

Coda can generate supplementary training data (material-imbalance positions, self-play with blunders):
```bash
# Material removal: remove pieces from EPD positions, deep-search each
coda datagen -nnue net.nnue -epd positions.epd -depth 10 -threads 32 -output material.binpack

# Self-play with blunders
coda datagen -nnue net.nnue -games 50000 -depth 8 -threads 32 -blunder 0.1 -output selfplay.binpack
```

Output is SF BINP binpack format, directly usable by Bullet.

### Key Training Findings

- **CReLU kills hidden layer neurons** during long training (dying ReLU). SCReLU prevents this.
- **LR warmup is critical for hidden layers**: 5-10 SB linear warmup 0.0001→0.001, then cosine 0.001→0.0001.
- **SCReLU scale chain**: keep v² at QA² through matmul, bias×QA² to match, /QA² after.
- **Hidden→output activation is linear** in Bullet (no SCReLU before output buckets).
- **v5 architecture is saturated**: 1024 CReLU/SCReLU/768pw all within ~20 Elo cross-engine. Hidden layers (v7) needed to break through.
- **WDL blend**: w0 is better than w5 for v7 hidden layer nets (+30 Elo). w3-w5 are equivalent for v5.
- **12-file training data** gives +33 Elo over 6-file for 768pw (data diversity matters).
- **v7 hidden layers need better training**: current v7 nets are ~40-80 Elo weaker than v5 due to hidden layer quality. Transfer learning (frozen FT) and supplementary material-imbalance data are being investigated.

Bullet LR schedule for hidden layer nets:
```rust
lr_scheduler: Sequence {
    first: LinearDecayLR { initial_lr: 0.0001, final_lr: 0.001, final_superbatch: 5 },
    second: CosineDecayLR { initial_lr: 0.001, final_lr: 0.0001, final_superbatch: N-5 },
    first_scheduler_final_superbatch: 5,
}
```

## NNUE Model Naming Convention

**v5 (direct FT→output):**
```
net-v5-{width}{activation}-w{wdl}-e{epochs}s{snap}.nnue
```

**v7 (FT→hidden→output):**
```
net-v7-{ftWidth}h{layers}{activation}-w{wdl}-e{epochs}s{snap}.nnue
```

Where:
- **ftWidth**: 1024, 1536, 768pw (pairwise)
- **h{layers}**: h16 (L1=16), h16x32 (L1=16, L2=32)
- **activation**: omit for CReLU, `s` for SCReLU
- **w{wdl}**: w0 (0.0), w5 (0.05), w10 (0.1)
- **e{epochs}**: total superbatches in cosine schedule
- **s{snap}**: snapshot checkpoint

Examples: `net-v5-1024-w0-e120s120.nnue`, `net-v5-1024s-w5-e400s400.nnue`, `net-v7-1024h16x32s-w0-e800s800.nnue`

## Key Search Parameters (from GoChess, tuned)
- NMP: R=3+depth/3, divisor 200, verify depth>=12
- RFP: depth<=7, margin improving?70*d:100*d
- Futility: 60+lmrDepth*60, depth<=8
- LMR: quiet C=1.30, capture C=1.80
- History pruning: -1500*depth, depth<=3, !improving
- SEE quiet: -20*depth², SEE capture: -depth*100
- QS delta: 240
- Aspiration: delta=15, from depth 4
- ProbCut: beta+170, staticEval+85 gate
- SEE values: P=100, N=320, B=330, R=500, Q=900
- History bonus cap: depth²  capped at 1200
- Contempt: -10

## Current Status (2026-03-30)

- **Strength**: -7 Elo in 13-engine cross-engine gauntlet (with 768pw net). In the Winter/Midnight/Weiss/Minic tier.
- **NPS**: ~1,350K (v5 1024 CReLU), ~1,464K (768pw), ~673K (v7 hidden layer) — all with AVX2 SIMD
- **vs GoChess**: +19 Elo from NPS advantage (15-26% faster depending on net type)
- **Lazy SMP**: Implemented. 4 threads gives ~4x NPS and +2 depth.
- **Features implemented**: All search features from GoChess, TT prefetch, v5/v6/v7 NNUE, datagen (multi-threaded), SF binpack output, Bullet convert, transfer learning checkpoint converter

## Key Gotchas
- Move flag equality vs bitwise: check non-promotion flags with ==, not &
- EP moves only valid when EP square is empty (occupied square = corruption)
- TT stores raw (uncorrected) eval to avoid double correction on probe
- Correction history only updated when bestScore > originalAlpha
- **is_pseudo_legal must be thorough**: TT hash collisions inject illegal moves. Pawn validation must check direction, intermediate squares (double push), starting rank, destination empty (pushes), enemy piece (captures). Castling must check rights, path clear, king/intermediate/destination not attacked, king on correct square. All three bugs cost 320 Elo combined.
- **PV error warnings = TT collision bugs**: Every "Illegal PV move" from cutechess-cli means a TT collision passed is_pseudo_legal and corrupted the search tree. Treat as critical, not cosmetic.
- **Feature flag ablation**: env var controlled flags (NO_XXX, ENABLE_XXX, DISABLE_ALL) for systematic search feature testing. Parsed once at startup via std::sync::Once.
- LMR contHist weight: 3x in move ordering, 1x in reduction adjustment
- Killers fully exempt from LMR (not just r -= 1)
- All pruning (LMP, futility, history, SEE) exempts TT move and killers
- PV nodes skip all TT cutoffs and QS beta blending
- Polyglot book encodes castling as king-to-rook (must convert to king-to-destination)
