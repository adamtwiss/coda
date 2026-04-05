# CLAUDE.md — Coda Chess Engine

Coda is a UCI chess engine written in Rust. Rewritten from GoChess with all accumulated knowledge.

**Chess Optimised, Developed Agentically** — built through human-AI collaboration.

## Build and Test

```bash
make                                             # Build with native CPU optimizations
make pgo                                         # PGO-optimized build (~5% faster)
make net                                         # Download production NNUE net
make openbench                                   # OpenBench-compatible build
cargo build --release                            # Plain release build
cargo test                                       # Run all tests including perft

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
./target/release/coda fetch-net                  # Download NNUE net from net.txt URL
./target/release/coda sample-positions [options]  # Sample positions from binpack to EPD
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
  zobrist_keys.rs  Auto-generated Zobrist key constants
  eval.rs          PeSTO material+PST eval (fallback), SEE values, NNUE eval wrapper
  see.rs           Static Exchange Evaluation
  tt.rs            Transposition table (5-slot buckets, XOR key verification)
  movepicker.rs    Staged move ordering, 4D history tables, killer/counter moves
  search.rs        Negamax, pruning, LMR, correction history, cuckoo, pruning stats
  cuckoo.rs        Cuckoo cycle detection for proactive repetition avoidance
  tb.rs            Syzygy tablebase probing (via shakmaty-syzygy)
  nnue.rs          NNUE v5/v6/v7 inference, accumulator stack, Finny table, AVX2/AVX-512 SIMD
  uci.rs           UCI protocol (position, go, stop, ponder, setoption)
  epd.rs           EPD test suite runner with SAN formatting
  book.rs          Polyglot opening book support
  polyglot_randoms.rs  Standard Polyglot Zobrist random table (781 entries)
  datagen.rs       Multi-threaded training data generation (self-play, material removal)
  binpack.rs       SF BINP binpack format writer (chain-compressed)
  bullet_convert.rs  Bullet quantised.bin → .nnue converter (v5/v6/v7)
  nnue_export.rs   .nnue → Bullet checkpoint converter (for transfer learning)
Makefile           Build targets: make, make pgo, make openbench, make net
scripts/
  ob_submit.py     OpenBench job submission script
  ob_stop.py       Stop an OpenBench test by ID
  ob_status.py     Fleet status and test results
training/configs/  Bullet training configs (.rs) for each net architecture
testdata/
  wac.epd          Win At Chess test suite (201 positions)
net.txt            Production NNUE net URL (used by make net / fetch-net)
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
- NMP: R=3+depth/4 + (eval-beta)/164, eval>=beta guard, depth>=3, verify at depth>=13 (depth-r), post-capture R--, NMP score dampening (score*2+beta)/3. All parameters SPSA-tunable (values from SPSA round 1).
- RFP: depth<=7, margin improving?70*d:100*d, returns staticEval-margin
- Futility: 90+lmrDepth*100+histAdj/128, depth<=8, uses estimated LMR depth
- LMR: separate quiet (C=1.30) and capture (C=1.80) tables, doDeeper/doShallower
- LMP: non-PV only, depth<=8, threshold 3+d² with improving/failing adjustments
- SEE pruning: quiet -20d² at depth<=8, capture -d*100 at depth<=6
- ProbCut: beta+170, staticEval+85 gate, SEE>=0 **(disabled by default — needs better eval)**
- History pruning: -1500*depth at depth<=3, !improving guard
- Bad noisy: prune losing captures when eval+depth*75<=alpha
- IIR: depth>=4, !inCheck, no TT move, PV or cut node
- Singular extensions (FEAT_SINGULAR)
- Cuckoo cycle detection for proactive repetition avoidance (FEAT_CUCKOO)
- Hindsight reduction: reduce when parent was LMR-reduced (>=2) and both sides quiet (FEAT_HINDSIGHT)
- Recapture extensions
- Fail-high score blending: (score*depth+beta)/(depth+1) at non-PV

**Move ordering:** TT move → good captures (MVV-LVA + captHist/16) → killers → counter-move → quiets (queen promos scored high, then main hist + contHist*3 + pawn hist) → bad captures

**Exemptions:** TT move and killers exempt from all pre-make pruning and from LMR. Promotions exempt from LMR.

**History tables:** main [from_threatened][to_threatened][from][to] (4D threat-aware, FEAT_4D_HISTORY), capture [piece][to][victim], continuation [piece][to][piece][to], pawn [pawnHash%512][piece][to], killers [ply][2], counter [piece][to].

**Correction history:** Multi-source static eval correction (6 sources). Pawn (384/1024) + white-NP (154/1024) + black-NP (154/1024) + minor pieces (102/1024) + major pieces (102/1024) + continuation (128/1024). Updated on bestScore > originalAlpha with depth >= 3.

**Other:** Contempt = 10 (applied as -CONTEMPT for draws, prefer playing over drawing). Insufficient material detection (KvK, KNvK, KBvK). Repetition detection via undo stack hash comparison + cuckoo cycle detection.

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
- `SparseL1` (check, default true) — sparse L1 matmul for v7
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

All GPU hosts should use the forked Bullet trainer at `~/code/bullet`:
```bash
# Should already be cloned; if not:
git clone https://github.com/adamtwiss/bullet ~/code/bullet
cd ~/code/bullet && cargo build --release
```

The Coda repo should also be cloned for the converter:
```bash
git clone https://github.com/adamtwiss/coda ~/code/coda
cd ~/code/coda && cargo build --release
```

### Training Data Locations

**GPU hosts** (cloud): `/workspace/data/`
**Dev hosts** (Hercules, Atlas, Titan): `/training/data/`

```
# T80 binpack data (~30B positions across 12 files)
T80_test-80-2024-10-d12-3B.binpack          # 3B positions
T80_test-80-2024-01-d9-12B.binpack          # 12B positions
T80_test-80-2024-04-d12-6B.binpack          # 6B positions
T80_test-80-2024-06-d12-6B.binpack          # 6B positions
...                                          # additional T80 files

# Supplementary data
blunders-0.2.binpack                         # Self-play with 20% blunder rate (~1B positions)
```

Bullet training configs reference these paths directly. When setting up a new GPU host, ensure the data directory is populated (symlink or copy from storage).

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

## Key Search Parameters
- NMP: R=4+depth/3, divisor 200, depth>=3, verify depth>=12 (depth-r), dampening (score*2+beta)/3
- RFP: depth<=7, margin improving?70*d:100*d
- Futility: 90+lmrDepth*100+histAdj/128, depth<=8
- LMR: quiet C=1.30, capture C=1.80
- History pruning: -1500*depth, depth<=3, !improving
- SEE quiet: -20*depth², SEE capture: -depth*100
- QS delta: 240
- Aspiration: delta=13+score²/23660 (eval-dependent), from depth 4
- ProbCut: beta+170, staticEval+85 gate (disabled by default)
- SEE values: P=100, N=320, B=330, R=500, Q=900
- History bonus cap: depth² capped at 1200
- Contempt: 10 (applied as -CONTEMPT)

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

## Testing Methodology

### CRITICAL LESSON (2026-04-01)

Narrow cross-engine gauntlets (3 engines, 200 games) gave us inflated Elo estimates (+20 to +67) for changes that self-play SPRT later proved were 0 to -17. The narrow gauntlet overfits to specific opponents just like self-play overfits to shared eval blindspots. **Never merge based on narrow gauntlet results alone.**

Self-play SPRT with tight bounds is the primary acceptance criterion. It is disciplined (runs until statistically significant), reproducible, and — while it has known limitations — the direction of its results consistently matched broader cross-engine testing in our validation experiments.

### Self-Play SPRT (primary acceptance test)

All search/eval changes must pass self-play SPRT before merging.

**Two tiers of bounds:**

**Tier A: `elo0=-5 elo1=5`** — "Don't ship regressions"
- For: code cleanup, refactoring, consensus features, infrastructure changes
- Question: "is this at least not harmful?"
- Typically resolves in 1000-3000 games
- Use when the change has structural value beyond pure Elo (e.g., cuckoo, 4D history)

**Tier B: `elo0=0 elo1=10`** — "Prove it helps"
- For: novel search ideas, parameter tuning, pruning changes
- Question: "is this genuinely positive?"
- Typically resolves in 1000-5000 games (longer for small gains)
- Use for anything that adds complexity or changes search behaviour

**SPRT command template:**
```bash
cutechess-cli \
    -tournament gauntlet \
    -engine name=Contender cmd=<NEW_BINARY> proto=uci \
        option.NNUEFile=<NET> option.OwnBook=false option.Hash=64 option.MoveOverhead=100 \
    -engine name=Base cmd=<OLD_BINARY> proto=uci \
        option.NNUEFile=<NET> option.OwnBook=false option.Hash=64 option.MoveOverhead=100 \
    -each tc=0/10+0.1 \
    -rounds 10000 -concurrency 4 \
    -sprt elo0=<ELO0> elo1=<ELO1> alpha=0.05 beta=0.05 \
    -openings file=/home/adam/code/gochess/testdata/noob_3moves.epd format=epd order=random \
    -pgnout /tmp/sprt_<name>.pgn -recover -ratinginterval 100 \
    -draw movenumber=20 movecount=10 score=10 \
    -resign movecount=3 score=500 twosided=true
```

**Key rules:**
- Both engines MUST use the same net file
- Contender = prior commit + one change. Never stack untested changes.
- Wait for H0 or H1. Do not stop early based on "looks good" — that's optimism bias.
- H0 = reject (revert). H1 = accept (merge). No exceptions.
- Log result to experiments.md with: H0/H1, Elo, games, LLR, tier used.

### Commit Messages

**Every commit that changes search/eval must include `Bench: <nodes>` in the commit message.** OpenBench uses this to verify the correct binary was built. Get the value by running `coda bench` with the production net loaded. Example:

```
Fix razoring margin at depth 2

Bench: 1443162
```

If the change doesn't affect bench (e.g. comments, docs, tooling), the bench line is optional.

### SPRT Testing Policy

**Every change that affects node count or playing strength must be SPRT tested before merging to main.** This includes:
- Search logic changes (pruning, reduction, extension parameters)
- Move ordering changes (history, scoring, sorting)
- Bug fixes in search code (even "obvious" fixes can regress)
- NPS optimizations (faster code = deeper search = potential Elo gain or regression)
- NNUE inference changes

**Workflow:**
1. Create a feature branch from main with the change
2. Add `Bench: <nodes>` to the commit message (run `coda bench` with production net)
3. Push the branch to GitHub
4. Submit SPRT test via OpenBench (https://ob.atwiss.com/) or `scripts/ob_submit.py`
5. Wait for H0 (reject) or H1 (accept). Do not stop early.
6. If H1: merge to main, update bench in main's commit message
7. If H0: do not merge, log result in experiments.md

**What does NOT need SPRT:**
- Comments, documentation, tooling changes
- Code cleanup that doesn't change compiled output (verify with bench)
- New feature flags (disabled by default)

**OpenBench scripts** (all require `OPENBENCH_PASSWORD` env var, username defaults to `claude`):
```bash
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch> <bench> [--bounds '[0,10]'] [--priority 1]
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_stop.py <test_id>
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_status.py
```
Reference NPS is 500K. Default bounds [0.00, 5.00] for novel changes. Always verify bench matches before submitting.

**Current bench: 1443162** (with production net, x86-64). Update this when search changes are merged.

### SPSA Parameter Tuning

26 search parameters are exposed as UCI options for SPSA optimization via OpenBench. Create a "Tune" workload (not "Test") on the web UI. Key parameters: NMP (base R, depth divisor, eval divisor, eval max, verify depth), RFP (depth, margins), futility (base, per-depth), history pruning (depth, multiplier), SEE pruning (quiet/capture multipliers), LMR (history divisor, C values for quiet/capture), LMP (base, depth), aspiration (delta, score divisor), probcut margin, hindsight threshold, SE depth.

SPSA format per parameter: `NAME, int, default, min, max, c_end, r_end`

When LMR_C_QUIET or LMR_C_CAP change, LMR tables are automatically reinitialized.

**Practical guidance (from round 1):**
- 1500-2000 iterations is sufficient for 16 parameters. Values stabilise well before 5000.
- c_end ~5-10% of parameter range, r_end 0.002 are good defaults.
- Alpha 0.602, gamma 0.101, A_ratio 0.1 (standard SPSA constants).
- Create via web UI at `/tune/new/` (scripts API doesn't support tune creation).
- SPRT the final values against main before merging — SPSA can overfit.
- Round 1 found +31 Elo from 16 parameters. Parameters interact — NMP eval divisor only works at 164 when RFP margins widen simultaneously.
- Plan SPSA after merging correctness fixes and before/after net switches (eval changes shift optimal parameters).

### Cross-Engine Validation (secondary, for milestones)

Self-play SPRT is reliable for direction but magnitude may differ cross-engine. Periodically validate cumulative progress with a peer-group gauntlet.

**Peer-group gauntlet** (after every 3-5 merged changes):
- Base binary + contender binary + 4-6 peer engines
- Peers: Midnight, Winter, Minic, Tucano, Weiss, Texel (adjust as Coda's strength changes)
- Gauntlet format (every game involves a Coda)
- 200+ games, check that contender ≥ base relative to peers

**Broad RR** (monthly milestone):
- 15-30 engines spanning full strength range
- Maximum 2 Coda variants (more causes Coda-on-Coda contamination that distorts results)
- 100+ rounds for ±30 error bars

### Model Comparison Testing

For comparing NNUE nets (architectures, training, WDL blends):

**5-way RR**: two nets under test + 3-4 rival engines.
Self-play between nets is uninformative for WDL changes (WDL differences are invisible in self-play). Cross-engine RR is essential.

```bash
cutechess-cli \
    -tournament round-robin \
    -engine name=NetA cmd=./coda proto=uci option.NNUEFile=net_a.nnue option.OwnBook=false \
    -engine name=NetB cmd=./coda proto=uci option.NNUEFile=net_b.nnue option.OwnBook=false \
    -engine name=Weiss cmd=... proto=uci \
    -engine name=Texel cmd=... proto=uci \
    -engine name=Winter cmd=... proto=uci \
    -each tc=0/10+0.1 \
    -rounds 50 -concurrency 16 \
    -openings file=noob_3moves.epd format=epd order=random \
    -pgnout /tmp/model_test.pgn -recover -ratinginterval 20 \
    -draw movenumber=20 movecount=10 score=10 -resign movecount=3 score=500 twosided=true
```

Note: RR is correct here (not gauntlet) — we want both nets to play each other AND the rivals.

### Known Testing Pitfalls

- **CPU contention**: Ensure CPU cores are fairly idle before starting SPRT tests. Background load (other tests, RRs, compilation) halves effective TC and distorts marginal results. In our ablation sweep, CPU contention flipped history pruning from +3 (false H1) to -17 (real H0) and correction history from -3 to -10. Always check `htop` or equivalent before launching tests.
- **Narrow gauntlet inflation**: 3 engines × 200 games gave +67 for a change that was actually 0. Use SPRT instead.
- **Coda-on-Coda contamination**: Multiple Coda variants in a RR amplify shared eval biases. Keep to max 2 Coda variants.
- **Optimism bias**: Stopping a test early because "it looks positive" leads to false positives. Let SPRT decide.
- **False negatives**: A change rejected at -5 with wide error bars might be +5. Retest when conditions change.
- **WDL blindspot**: WDL blending improvements are invisible in self-play. Must use cross-engine RR.
- **Self-play discount**: Self-play Elo ≠ cross-engine Elo. Direction is usually reliable, magnitude varies.

### Feature Improvement Cycle (Detect → Diagnose → Fix → Tune)

Systematic approach for finding and fixing search feature issues. Each cycle compounds — fixing one feature shifts the optimal parameters for everything else, revealing the next weak feature.

**1. Detect weak features**
- **SPSA detuning**: If SPSA is aggressively moving a parameter away from its starting value (>30% shift), the feature may have a structural flaw. The tuner compensates for bugs by detuning. Example: SEE_QUIET_MULT driven from 17 to 6 (nearly disabled) because the implementation was broken.
- **Ablation anomaly**: If disabling a feature gains Elo, or gains less than expected, investigate.
- **Cross-engine parameter divergence**: If our value for a parameter is far outside the consensus range, understand why before assuming we're special.

**2. Diagnose via cross-engine comparison**
- Compare the specific feature implementation against 6-8 top engines with source code available. Engine sources are in `/home/adam/chess/engines/`.
- Reference engines (strongest, most relevant): Stockfish, Viridithas (Rust), Obsidian, Berserk, Reckless (Rust), PlentyChess, Caissa, RubiChess, Halogen, Stormphrax.
- For each engine: exact formula, gating conditions, position in move loop (before/after MakeMove), depth variable used (raw depth vs lmrDepth), history adjustments, numeric values.
- Common structural issues found so far:
  - **Pre-move vs post-move**: Pruning after MakeMove wastes make/unmake + NNUE push/pop per pruned move, and makes the feature redundant with earlier pruning (futility, LMP catch most candidates first).
  - **Raw depth vs lmrDepth**: Using raw depth for depth² scaling is far more aggressive than intended. Engines using depth² all use lmrDepth. Engines using raw depth compensate with linear scaling.
  - **Duplicate SEE functions**: Using a separate SEE implementation instead of the standard `see_ge` risks correctness bugs.

**3. Fix and SPRT test**
- Create a feature branch with the structural fix.
- Set parameter defaults to match consensus (e.g., Stockfish's value for the same formula).
- SPRT test against main with bounds [0, 10]. The fix should be positive even with untuned constants if the structural change is correct.
- If SPRT fails, review for secondary bugs (missing ply>0 guard, bestScore pollution, etc.).

**4. SPSA tune the corrected feature**
- **Focused tune first** (2-4 params, ~1000 iterations): Just the new/changed parameters. Fast convergence, finds the right ballpark.
- **Full tune after** (all 32 params, 5000+ iterations): Rebalances everything — other params were compensating for the broken feature and need to readjust.
- Merge the focused tune values, then run the full tune as the next round.

**5. Repeat**
The retuned baseline exposes the next weak feature. Check SPSA trends for the next parameter being aggressively detuned.

**Results from this approach (2026-04-05):**
- SEE quiet pruning: SPSA flagged (17→6), deep dive found 3 structural issues (post-move, raw depth², duplicate SEE fn), fix tested +10.8 Elo.
- Futility pruning: SPSA flagged (116→99), deep dive found gate/margin mismatch (raw depth gate, lmrDepth margin), fix in testing.

### Model Comparison Testing

For comparing NNUE nets (different architectures, training schedules, WDL blends):

1. **Quick H2H** (200 games, self-play): same engine, two different nets. Fast signal.
2. **5-way RR** (50 rounds): two nets under test + 3 rival engines (Minic/Ethereal/Texel).
   This gives both self-play delta AND cross-engine validation in one test.
   Self-play and cross-engine can give different signals for models — the RR catches this.

```bash
cutechess-cli \
    -tournament round-robin \
    -engine name=NetA cmd=./coda proto=uci option.NNUEFile=net_a.nnue option.OwnBook=false \
    -engine name=NetB cmd=./coda proto=uci option.NNUEFile=net_b.nnue option.OwnBook=false \
    -engine name=Minic cmd=... proto=uci \
    -engine name=Ethereal cmd=... proto=uci \
    -engine name=Texel cmd=... proto=uci \
    -each tc=0/10+0.1 \
    -rounds 50 -concurrency 16 \
    -openings file=noob_3moves.epd format=epd order=random \
    -pgnout /tmp/model_test.pgn -recover -ratinginterval 20 \
    -draw movenumber=20 movecount=10 score=10 -resign movecount=3 score=500 twosided=true
```

Note: RR is correct here (not gauntlet) — we want both nets to play each other AND the rivals.
