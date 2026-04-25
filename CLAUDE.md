# CLAUDE.md — Coda Chess Engine

Coda is a UCI chess engine written in Rust. Rewritten from GoChess with all accumulated knowledge.

**Chess Optimised, Developed Agentically** — built through human-AI collaboration.

## Supported CPU families

**x86_64 (primary):** OpenBench fleet, Lichess deployment, CCRL, all SPRT
gating. Default target.

**aarch64 (first-class, since 2026-04-25):** Apple M-series and ARM
servers (e.g. Graviton). New code must use correct memory ordering for
SMP — `Acquire/Release` on shared atomics with reader-publish patterns,
not `Relaxed`. x86's strong memory model masks ordering bugs that fire
on aarch64. See `docs/arm_correctness_2026-04-25.md` for the
ARM-correctness sweep status, the coding standard for new atomics, and
remaining audit items. When adding shared atomics or SIMD paths,
default to `Acquire/Release` and explicit NEON tests; `Relaxed` is
only correct when there's no data-dependency on the synchronization.

## Build and Test

**Prerequisites:** Rust 1.70+. For PGO builds:
```bash
cargo install cargo-pgo
rustup component add llvm-tools-preview
```

```bash
make                                             # Build with native CPU optimizations
make pgo                                         # PGO-optimized build (~3-5% faster v5 on main; regresses v9 — see Makefile)
make net                                         # Download production NNUE net
make openbench                                   # OpenBench-compatible build
cargo build --release                            # Plain release build (no embedded net)
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
  tb_cache.rs      Lockless Zobrist-keyed WDL probe cache (UCI TBHash)
  nnue.rs          NNUE v5/v7/v9 inference, accumulator stack, Finny table, AVX2/AVX-512/VNNI SIMD
  sparse_l1.rs     Sparse/dense int8 L1 matmul kernels (AVX2, AVX-VNNI, AVX-512 VNNI)
  threats.rs       Threat-feature enumeration + delta generation (v9)
  threat_accum.rs  Per-ply threat accumulator stack (v9)
  uci.rs           UCI protocol (position, go, stop, ponder, setoption)
  epd.rs           EPD test suite runner with SAN formatting
  book.rs          Polyglot opening book support
  polyglot_randoms.rs  Standard Polyglot Zobrist random table (781 entries)
  datagen.rs       Multi-threaded training data generation (self-play, material removal)
  binpack.rs       SF BINP binpack format writer (chain-compressed)
  bullet_convert.rs  Bullet quantised.bin → .nnue converter (v5/v7/v9)
  nnue_export.rs   .nnue → Bullet checkpoint converter (for transfer learning)
Makefile           Build targets: make, make pgo, make openbench, make net
scripts/
  ob_submit.py     OpenBench SPRT job submission
  ob_tune.py       OpenBench SPSA tune submission
  ob_tune_status.py  Read SPSA tune results and compare branches
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

**Pruning features** (all SPSA-tunable via `tunables!` macro — see the macro in
`search.rs` for the current parameter list and defaults; count grows over time):
- NMP: R=BASE_R+depth/DEPTH_DIV + (eval-beta)/EVAL_DIV, verify at depth>=VERIFY_DEPTH, post-capture R++
- RFP: depth<=RFP_DEPTH, margin improving?RFP_MARGIN_IMP*d:RFP_MARGIN_NOIMP*d
- Futility: FUT_BASE+lmrDepth*FUT_PER_DEPTH, history adjusts effective lmr_depth (SF pattern)
- LMR: separate quiet and capture tables (C_QUIET/C_CAP), doDeeper/doShallower, tt_pv reduces less
- LMP: non-PV only, depth<=LMP_DEPTH, threshold (LMP_BASE+d²)/(2-improving)
- SEE pruning: quiet -SEE_QUIET_MULT*d² at shallow depth, capture -SEE_CAP_MULT*d
- ProbCut: beta+PROBCUT_MARGIN, staticEval gate, SEE>=0
- History pruning: -HIST_PRUNE_MULT*depth at depth<=HIST_PRUNE_DEPTH
- Bad noisy: prune losing captures when SEE < -BAD_NOISY_MARGIN
- IIR: depth>=4, !inCheck, no TT move, PV or cut node
- Singular extensions + double extensions (margin=DEXT_MARGIN, cap=DEXT_CAP)
- Cuckoo cycle detection for proactive repetition avoidance
- Hindsight reduction: reduce when parent was LMR-reduced and both sides quiet
- Recapture extensions
- Fail-high score blending at non-PV nodes
- TT cutoff node-type guard (Alexandria pattern)
- TT cutoff cont-hist malus (penalize opponent's quiet on cutoff)
- Mate distance pruning (non-PV, ply+1 offset)

**Move ordering:** TT move → good captures (MVV×16 + captHist) → quiets (main hist + contHist×3 + pawn hist + quiet check bonus) → bad captures. Good/bad quiet split being tested (SF pattern: defer low-history quiets after bad captures).

**Exemptions:** TT move exempt from pruning. Promotions exempt from LMR.

**History tables:** main [from_threatened][to_threatened][from][to] (4D threat-aware), capture [piece][to][victim], continuation [piece][to][piece][to] (4 plies: 1,2,4,6), pawn [pawnHash%512][piece][to]. Linear bonus formula: min(HIST_BONUS_MAX, HIST_BONUS_MULT*depth - HIST_BONUS_BASE).

**Correction history:** Multi-source static eval correction (5 sources, SPSA-tunable weights). Pawn + white-NP + black-NP + minor + major + continuation. Proportional gravity update.

**Time management:** 3-factor multiplicative model (Obsidian/Clarity). Node fraction (tracks per-root-move nodes), best-move stability (linear), score trend. Validated at LTC (40+0.4) — TM features invisible at STC.

**Other:** Contempt = 10. Insufficient material detection. Repetition + cuckoo cycle detection.

### TT
- 5-slot buckets, 64 bytes (cache-line aligned), AtomicU64/AtomicU32 for lockless Lazy SMP
- XOR key verification: `key_xor = hash ^ data` (detects torn reads from concurrent writes)
- 13-bit staticEval (±4095 cp range), 1-bit tt_pv flag (sticky PV marker for LMR)
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
- **Low final LR is critical**: Our cosine 0.001→0.0001 final LR was 20× too high. Bullet examples use `0.001 * 0.3^5 = 2.4e-6`. Reducing to ~5e-6 gave **+47 Elo**. The net was oscillating in late training instead of converging.
- **Data filtering**: Training on quiet positions only (ply≥16, no checks, no captures, no tactical moves) gave **+22 Elo untuned, +48 tuned**. Aligns with how NNUE eval is used (only at quiet nodes after QS).
- **Combined (filter + low LR)**: +32 untuned, +80 with retune at s120. The gains are partially additive.
- **WDL 0.25**: -24 Elo due to eval scale mismatch. Needs retune to show benefit. Our w7 (0.07) is an outlier vs consensus (0.3-0.4).

### EVAL_SCALE Calibration

`EVAL_SCALE` (nnue.rs:33, default 400) converts raw network output to centipawns. Different training configs (filtering, WDL, LR) produce different eval scales. When the scale changes, all search thresholds (RFP, futility, SEE, LMR) become miscalibrated.

**Measuring scale**: Evaluate 500 positions from selfplay data and compute RMS:
```bash
# Sample positions
./coda sample-positions -i /training/coda/selfplay.binpack -o /tmp/sample.epd -n 500 --rate 0.001

# Evaluate with the net and compute RMS
python3 -c "
import subprocess, re, math
positions = [l.strip() for l in open('/tmp/sample.epd').readlines()[:500]]
cmds = 'uci\nisready\n'
for fen in positions:
    cmds += f'position fen {fen}\neval\n'
cmds += 'quit\n'
result = subprocess.run(['./target/release/coda', '-n', 'path/to/net.nnue'],
    input=cmds, capture_output=True, text=True, timeout=30)
scores = [int(m.group(1)) for m in re.finditer(r'raw_nnue\s+([-\d]+)', result.stdout)]
rms = math.sqrt(sum(s*s for s in scores) / len(scores))
print(f'RMS={rms:.0f}, ratio to baseline(580)={rms/580:.2f}x')
"
```

**Baseline RMS**: 580 (production net with EVAL_SCALE=400). Target: RMS ≈ 580 after adjustment.

**WARNING**: EVAL_SCALE does NOT scale linearly for pairwise nets. The pairwise architecture squares values, so large EVAL_SCALE causes integer overflow in quantized inference. Always verify RMS after changing — don't just compute `400 * baseline_rms / net_rms`.

**Alternatives to EVAL_SCALE adjustment**: SPSA retune (preferred). The tune recalibrates all thresholds to the net's natural scale. EVAL_SCALE adjustment is a quick hack; retune is the proper fix.

Bullet LR schedule for hidden layer nets:
```rust
lr_scheduler: Sequence {
    first: LinearDecayLR { initial_lr: 0.0001, final_lr: 0.001, final_superbatch: 5 },
    second: CosineDecayLR { initial_lr: 0.001, final_lr: 0.0001, final_superbatch: N-5 },
    first_scheduler_final_superbatch: 5,
}
```

## NNUE Model Naming Convention

### Accumulator width clarification (2026-04-14)

**IMPORTANT**: The "width" in net names refers to different things historically:
- `768pw` in **v5** names: 768 accumulator per perspective (768 FT outputs, pairwise → 384)
- `768pw` in **v7** names: **actually 1536 accumulator** (1536 FT outputs, pairwise → 768).
  The "768" refers to the 768 input features (12×64), not the accumulator width.
- `1024` / `1024s`: 1024 accumulator, no pairwise (direct concat → 2048 L1 input)

This naming inconsistency is historical. Going forward, new architectures use
accumulator width explicitly.

Cross-engine comparison:
- Coda v7 "768pw": accum=1536, same as Alexandria/Obsidian
- Reckless: accum=768, pairwise → 384, plus 67K threat features
- Our threat target (v9): accum=768, matching Reckless

**v5 (direct FT→output):**
```
net-v5-{width}{activation}-w{wdl}-e{epochs}s{snap}.nnue
```

**v7/v8 (FT→hidden→output):**
```
net-v7-{ftWidth}h{layers}{activation}-w{wdl}-e{epochs}s{snap}.nnue
```

**v9 (FT + threats → hidden → output, target architecture):**
```
net-v9-{accumWidth}t{activation}h{layers}-w{wdl}-e{epochs}s{snap}.nnue
```

Where:
- **width/ftWidth**: historical — 768pw, 1024, 1536 (see clarification above)
- **accumWidth**: actual accumulator width per perspective (v9+)
- **t**: threat features present
- **d**: dual L1 activation (CReLU+SCReLU)
- **h{layers}**: h16 (L1=16), h16x32 (L1=16, L2=32)
- **activation**: omit for CReLU, `s` for SCReLU
- **w{wdl}**: w0 (0.0), w5 (0.05), w10 (0.1), w15 (0.15)
- **e{epochs}**: total superbatches in cosine schedule
- **s{snap}**: snapshot checkpoint

Examples:
- `net-v5-768pw-w7-e800s800.nnue` — v5 production (768 accum, pairwise)
- `net-v7-768pwh16x32-w15-e100s100.nnue` — v7 (actually 1536 accum, historical name)
- `net-v9-768th16x32-w15-e400s400.nnue` — v9 target (768 accum, threats, 16→32)

## Key Search Parameters

All parameters are SPSA-tunable via the `tunables!` macro in `search.rs`
(~77 parameters as of 2026-04-24, count grows over time). Current values
reflect multiple SPSA rounds + retune-on-branch calibration, most
recently the full-sweep #743 retune merged as #747. See the macro in
`search.rs` for authoritative defaults.

- SEE values: P=100, N=320, B=330, R=500, Q=900
- History bonus: linear formula min(MAX, MULT*depth - BASE)
- Contempt: 10 (applied as -CONTEMPT)

## Current Status

Living state (strength, bench, net, SPSA progress, deployment) changes too
quickly to keep accurate in a checked-in file. Authoritative sources:

- **Current production nets:** `docs/net_catalog.md` (v5 + v9 prod hashes,
  retired/active nets, invariants)
- **Recent experiment history / lessons:** `experiments.md`
- **Current bench:** the `Bench: <nodes>` line of the latest commit on the
  branch you're submitting. Re-measure on the exact branch+net before any
  SPRT; don't carry bench values across branches.
- **Lichess bot:** `codabot` (deployment state and thread count varies)
- **OpenBench fleet:** `ob.atwiss.com`, composition varies

Testing methodology is durable and documented below (Self-play SPRT primary,
retune-on-branch for tree-shape-changing features, LTC for TM features).

## Strength Frontier — Where the Elo Gap Lives (2026-04-24)

Findings from a 100-game bullet H2H vs Stockfish 18 and Atlas's
loss-pattern analysis. This is a working hypothesis, not dogma —
update as more data accumulates.

### Calibration

Three anchor points:

- **60+1 bullet H2H vs SF 18**: Coda −159.8 ±41.6 Elo (100 games).
- **10+0.1 ultra-bullet 45-engine RR** (Adam's local RR, every
  top-10 CCRL engine present): Coda gap to SF settling in the
  **~270-302 Elo** range across snapshots (±55 error bars at
  ~88 games). Rank drifting 21-23/45, cluster with
  Clarity/Velvet/PZChessBot at latest snapshot, above
  Igel/Arasan/Altair cluster.
- **CCRL 40/15 + 4CPU**: SF 3652, top 45 bunched within 100 Elo
  (Reckless 3647 only 5 behind SF). Coda not yet listed; our
  ultra-bullet → CCRL inference (see
  `memory/project_ultra_bullet_vs_ccrl_calibration.md`) puts us in
  the **3520-3620 band** — top-30 territory on CCRL. Caveat:
  pool composition matters — Coda is tactical-archetype so likely
  at the lower end of that band in a broad pool, higher in a
  close-rivals pool.

SF gap as a function of TC (see TC-handicap sigmoid below for the
full curve):
- 10+0.1 (ultra bullet): ~302 Elo
- 60+1 (bullet): ~160 Elo
- 8× (480+8, past the knee): ~35 Elo
- 16× (960+16): ~0 Elo (parity)

### The TC-handicap sigmoid (2026-04-24 calibration)

Bullet H2H vs SF 18 with asymmetric TC (Coda at 60+1 vs SF at
reduced TC). 100 games per data point:

| HC | Score (W-L-D) | Coda % | Gap Elo (±) | Draw % | Δ vs prev |
|---:|:---|---:|---:|---:|---:|
| 1× (baseline) | 0-43-57 | 28.5% | **−159.8** ±41.6 | 57% | — |
| 2× | 0-47-53 | 26.5% | **−177.2** ±43.9 | 53% | 0 (within noise) |
| 4× | 0-35-65 | 32.5% | **−127.0** ±37.2 | 65% | +50 |
| 8× | 3-13-84 | 45.0% | **−34.9** ±26.7 | 84% | **+92 (knee)** |
| 16× | 15-16-69 | 49.5% | **−3.5** ±38.0 | 69% | +31 |

**Classic sigmoid** with an ultra-flat (possibly slightly negative)
first doubling. 2× TC doesn't help at all — SF's time-management is
so efficient at short TC that Coda's extra time doesn't translate
to useful extra depth. Knee at 4×→8× (+92 Elo). Taper into parity
at 16× (LOS 43%).

**The 1×→2× flat zone is important**: a 2× NPS gain from our
inference work alone is worth ~0 Elo vs SF. NPS work only pays once
we're near the 4× knee threshold.

**Draw-rate peak at 8×** (84%!): both engines draw nearly everything
at that TC — we've reached "solid drawing depth" but still can't
out-play SF from equal position. At 16× Coda finally wins enough
games to match SF's wins (15W-16L-69D, essentially symmetric).

**Caveat on absolute numbers**: the sigmoid shape is preserved across
net/tune configurations — treat absolute Elo values as anchors with
~±10 Elo uncertainty rather than exact points. Updated anchors would
be useful post- any major retune + merge cluster (most recently
#747 retune + v9-into-main merge, ~+7 Elo on trunk, not yet
re-calibrated on the sigmoid).

**What this tells us:**

- Our **eval is at SF parity** — at equal depth, we play as well.
  The full 160 Elo gap at normal TC is *horizon*, not quality.
- Horizon is **nonlinear in depth against SF-class opponents**.
  Partial depth gains below the knee (1-4×) return near-zero Elo.
  Above the knee (4-8×), each incremental depth-ply delivers 20-30+
  Elo. This is because specific tactical refutations live at
  specific depths — seeing them at all vs not matters far more than
  seeing them half-a-ply earlier.
- **EBF-ratio intuition pump**: with EBF≈2, a 4-ply depth advantage
  is a 2⁴ = 16× search-effort ratio *both* at d=6-vs-10 and at
  d=26-vs-30. Geometrically identical. But at d=6 vs d=10 the
  stronger engine sees whole classes of 8-ply combinations the
  weaker engine literally cannot — knight forks in 3, quiet moves
  winning material in 5, etc. At d=26 vs d=30, both engines already
  catch every standard tactical motif; the extra 4 plies refine
  positional eval and resolve obscure endgame nuances. Same ratio,
  very different Elo value. This is why SF's ~4-ply NPS+pruning
  advantage is worth ~100 Elo over Reckless at 10+0.1 but only 5
  Elo at CCRL 40/15: both engines have crossed the "sees the
  tactic" threshold at the longer TC.
- **The sigmoid applies specifically to wide-Elo-gap opposition.**
  Vs same-tier opponents (e.g. our Rivals pool at 3430-3585 CCRL),
  the tactical-cliff dynamic isn't present — both engines have
  similar horizon, so NPS wins convert more linearly to Elo. NPS
  work still pays for closing small gaps and holding ground in
  close matchups; it just plateaus against engines far above us.
- **Compound to cross the knee or plateau below it (vs SF).**
  Individual experiments at +3% NPS / +3 Elo pre-knee feel
  thankless against SF; they deliver Elo only insofar as they stack
  with other gains toward the knee threshold. BUT those same +3 Elo
  wins are cleanly valuable in Rivals-tier competition (Lichess bot
  ranking, CCRL position).

### Path to closing the gap (pragmatic, 2026 horizon)

To reach the knee (~8× effective depth-gain) we need compounding
across all levers. Realistic per-lever contribution budget:

| Lever | Plausible gain | Path |
|---|---:|---|
| **Shelved net upgrade** (already validated, not yet deployed) | **~30 Elo** | Reckless-KB (+15, SPRT'd) + factor arch (+15, SPRT'd) — need clean SB800 training as full prod replacement. Current prod is still consensus-buckets SB800. |
| Cache residency + SIMD dispatch | +10-25 Elo | L1 permutation, VNNI dispatch, group-lasso-driven matrix shrink (in progress 2026-04-24) |
| Force-more-pruning retune | +5-10 Elo | `experiment/force-more-pruning` #750 running 2026-04-24 |
| Further eval refinement | +10-20 Elo | Post-shelved-net training-recipe iteration |
| Move-ordering improvements | +5-15 Elo | EBF-reducing work; compounds with above |
| **Stacked total** | **~60-100 Elo** | Meaningful jump — top-20 CCRL range |

**The shelved net is the cheapest lever** — the Elo is already
earned in SPRT, just needs one SB800 training run to deploy. Don't
chase NPS or search wins that individually outweigh 30 Elo without
this deployment in flight.

That total moves us from rank ~42 CCRL → top-20 range — but does
NOT reach parity with SF.

**Parity (16×-equivalent)** requires dramatic EBF reduction (log(EBF)
halved, 1.8 → 1.35) which in turn requires multi-year investment:
richer training pipeline, more net iteration, possibly novel search
innovations. Don't expect parity in 2026.

### What the losses look like

Atlas's analysis of 27 SF losses (2026-04-24):

- Median max single-ply eval drop: **4.53 cp**. Mean 5.56 cp.
  18/27 losses have a >3 cp cliff, 7/27 have >8 cp.
- Cliffs happen mid-game → endgame transition (median ply 83,
  range 42–123). Openings survive fine.
- Coda median depth in losing games: **15**, max 21.
  SF at 60+1 typically reaches 25–35.
- Coda's eval says "fine"; one ply later the eval collapses
  4–14 pawns.

**This is tactical blindness from search-depth deficit, not
positional mis-eval.** Coda's eval is accurate at the depth it
reaches; the refuting combination lives at plies SF sees and we
don't.

### What closes the gap

**Effective depth** is the target. Effective depth ≈ log(NPS × time)
/ log(EBF), so depth = f(raw NPS, pruning efficiency, eval quality).

Decomposition of the ~160 Elo SF gap, rough budget:

| Lever | Approx share of gap | Status |
|---|---:|---|
| Raw NPS deficit (Coda is ~2× slower than Reckless) | **~100 Elo** (~65%) | Some portion is deliberately bought — x-ray threat features cost ~20% NPS but paid +110 Elo (good trade). We won't fully close this; target is partial recovery. |
| Pruning efficiency (Coda under-prunes vs Reckless on several params) | **~30–50 Elo** | Under-exploited; "force more pruning + retune" branches are the lever |
| Eval quality / tactical sharpness | **~20–30 Elo** | Factor-net-quality, training recipe improvements |

**Key NPS framing**: we don't expect to match Reckless's raw NPS.
Some of our deficit is a deliberate eval-architecture trade
(x-ray threats, wider FT chunks). The recoverable NPS ceiling is
bounded by what doesn't regress eval quality.

**Where the recoverable NPS lives**: almost entirely in
**cache-residency improvements tied to sparsity**. The current
49 MB threat weight matrix spills L3 on most fleet hosts (measured
on 9-host microbench; see `docs/nps_microbench_hostdata.md`). If
we can shrink it below ~32 MB via training-side L1 regularisation
(Item 2 in the NPS investigation) and Viridithas-style input-chunk
L1 permutation (R3 in research_threads_2026-04-24.md), we get
a step-function NPS gain across the entire fleet. That's the
biggest remaining NPS lever by far — cache-residency wins
compound across every node of every search.

Both NPS (via sparsity/cache work) and pruning (via retune branches
targeting the Reckless outliers) attack the SAME target — depth —
and compound.

**Concrete Reckless vs Coda pruning outliers (measured 2026-04-24,
status updated post-experiment batch):**

- **Futility margin**: Coda pre-#736 had 78+160·d; Reckless uses
  ~35+65·d (Coda ~2.4× less aggressive). #736 tightened to
  Reckless-scale (40/65), H1 +4.4 on detuned trunk. Post-retune #747
  reverted to tune's landing point (80/156) per Adam's override call,
  because tune-landscape preferred the wider values once other
  pruning recalibrated. Net effect in trunk: the margin matters
  less than the surrounding pruning cluster calibration.
- **SEE quiet prune magnitude**: Coda threshold −1175 at d=5 vs
  Reckless −145 (8× more lenient). Direct Reckless-shape port
  H0'd twice (#738 catastrophic at −167; #740 calibrated at −13.3).
  SPSA converges at SEE_QUIET_MULT≈46, not near Reckless's value —
  seemingly our lmr_d-based formula and Reckless's raw-depth
  formula aren't directly portable. Mechanism-level work remains.
- **RFP depth cap**: Coda d≤10 (now 11 post-retune #747); Reckless
  uncapped. #737 raised cap 10→16 on detuned trunk, H1 +2.2; retested
  on retuned trunk as `experiment/rfp-depth-uncap-v2` (bench-neutral
  with retuned tight margin, SPRT running at [0, 3]).
- **LMP depth cap**: Coda d≤8; Reckless uncapped. #735 uncap tested
  H0 at −1.1.
- **BNFP depth cap**: Coda d≤4; Reckless d<11. #734 uncap tested
  H0 at −1.2.

Of the five outliers, direct-port experiments landed 1 clear H1 (RFP
depth, +2.2 detuned; retune pending), 3 H0, and 1 mixed-result
(futility). Pattern: the raw-Reckless values aren't directly portable
— our different search context (lmr_d vs raw-depth formulas, different
history scaling) means the OUTLIERS identify real miscalibration but
the FIX requires SPSA-retune-on-branch, not direct value import.

`experiment/force-more-pruning` tune #750 (2026-04-24, 2000 iters,
77 params) tests this holistically — aggressive starting point,
SPSA finds new equilibrium.

### Priors this updates

- **NPS is a dominant lever (~2/3 of the gap), not secondary.**
  Cache-residency wins (flatten AccEntry, eval-TT writeback, L1
  permutation, training-side matrix shrink) and SIMD dispatch each
  translate directly to depth. A 10% NPS gain ≈ ~5–8 Elo at this
  regime.
- **Pruning values matter just as much per-param.** We have at
  least five clear outliers where Coda prunes less than Reckless.
  Each retune-on-branch around a tightened threshold is probably
  +2–5 Elo. Collectively they're in the same ballpark as NPS wins.
- **Coda deliberately traded NPS for eval quality** via v9
  threat-inputs (Reckless architecture). That trade only pays off
  if the better eval feeds better pruning decisions. Net Elo >
  net NPS. The eval-quality lever is smallest in this decomposition
  but orthogonal — it compounds with both others.
- **"Force more pruning"-style branches** (intentionally widen
  an under-pruning threshold, let SPSA find new equilibrium)
  have worked before and are under-built-on. The Reckless
  comparison gives specific targets.
- **Experiments that buy depth at hot plies** (extension on
  tactical-density signals, cliff-risk heuristics) are a new
  candidate class Atlas's analysis surfaced. W2-pattern work
  (signal × pruning/extension decision) already has high hit
  rate in Coda's history.

### De-prioritised by this finding

- **Eval post-processing tweaks** (contempt, optimism
  calibration, fortress caps, shuffle detectors) are low leverage
  for the SF-gap. Keep doing them when they're cheap, but don't
  let them displace pruning/NPS work on the queue.
- **Factor net (architecture work) alone** doesn't close the SF
  gap — it improves eval quality, which has indirect leverage
  (see above) but doesn't address depth deficit directly. Still
  worth doing because it helps Rivals-tier strength and makes
  pruning calibration cleaner; just don't expect it to crack the
  cliff-miss class.

### Workflow implication for future Claudes

When sizing up a new experiment, ask three questions in order:

1. Does it increase effective depth? (Via pruning efficiency,
   NPS, or cliff-avoiding extensions.)
2. Is Coda's pruning value in this area an outlier vs top-engine
   consensus? (SF/Reckless/Obsidian/Viridithas — check actual
   values, not just presence/absence of a feature.)
3. Would it show up in a 100-game SF bullet H2H? (±~40 Elo
   bars at that sample.)

If all three are "no", the expected Elo-per-effort is probably low
compared to items that say yes to at least (1) and (2).

### Recalibration cadence

Re-run the 100-game SF H2H every ~2 weeks or after any merge
cluster worth ~+20 Elo. The gap should narrow visibly; if it
doesn't, our on-paper Elo is overstated.

## Improvement Portfolio — Diversified Threads (2026-04-24)

Beyond the flywheel (eval → ordering → pruning → depth → eval), the
gap-closing strategy runs multiple ORTHOGONAL threads in parallel.
Correlated work can plateau together; diversified threads hedge.
When picking the next experiment, explicitly name which thread it
sits in — it helps avoid the trap of chasing one axis while others
decay.

**Eval-search flywheel** — compounding feedback loop:
- Better eval → better ordering → safer pruning → more depth → better
  eval (via stronger self-play training data)
- Non-linear: a +2% eval improvement often delivers +8-15 Elo AFTER
  retune-on-branch, versus +3-5 Elo raw. Flywheel-captured gain
  doubles or more.
- Retune-on-branch (§SPSA) is the mechanism by which we capture
  flywheel gain after each feature. Don't drop a +1-2 Elo raw feature
  without running the retune — that's usually where the real Elo is.
- Reckless-value imports fail directly because Reckless's values are
  calibrated against Reckless's own flywheel state. Our optimum is
  different. Outliers are directional signals; retune-on-branch is
  the portable-translation mechanism.

**Correctness audits** — bugs in rarely-fired paths:
- Historically our highest Elo-per-hour lane. Bugs in 50-move rule,
  LMR endgame gate, SMP race conditions, TB integration, and a
  critical `is_pseudo_legal` EP hole have each delivered +3-30 Elo.
- `correctness_audit_2026-04-22.md` SPECULATIVE list has queued
  candidates. R5 in `research_threads_2026-04-24.md` ranks the top 5
  by Lichess reachability.
- Under-invested relative to payoff. If in doubt between a feature
  experiment at expected +2 Elo and a correctness audit branch,
  audit first.

**Comparative engine review with instrumentation** — leverage
multiplier we're under-using:
- Reading top-engine source tells you *what* — instrumentation tells
  you *when and how often*. Different information class.
- Practical recipe: patch Reckless (or SF, Obsidian) to log pruning
  rates / first-move-cut / R-value distributions per depth. Run the
  same bench or `coda_blunders.epd`. Compare tree shape directly.
- `scripts/reckless_evalbench.patch` is the entry point we've used
  for this so far. A half-day of instrumented comparison often
  surfaces gaps that a full day of source-reading misses.
- Specific under-studied axes: NMP R-value distribution, LMR
  reduction amounts by history bucket, pruning fire rate by depth,
  TT hit rate breakdowns.

**Training hyperparameters + data** — upstream from the eval:
- Biggest historical wins: low-final-LR (+47), data filtering ply≥16
  (+22, +80 with retune), WDL calibration. Each was 1-line schedule
  change.
- Still-unexplored: output bucket count (we use 8, untested on v9),
  batch size (16384 default, not defended empirically), WDL schedule
  shape (constant vs linear vs late-ramp), data composition additions
  (Lichess-blunder positions into training set).
- Lichess-watch provides a feedback loop for "what positions does
  Coda actually blunder" → those positions become training-data
  candidates.

**Long tunes / long training — "free" Elo at cost of patience**:
- 25K-iter SPSA (~200K games). Top engines use this routinely; our
  2-2.5K standard almost certainly leaves small-gradient parameter
  gains on the table. Worth running once every 5-10 merges as a
  "settle calibration" pass — not routine.
- SB1600+ training. V9 pattern is specifically different from v5/v7:
  sparse threat features continue converging deep into the low-LR
  tail. SB400 → SB800 delivered +88 Elo on v9 (vs flat on v7). Each
  additional 400 SB plausibly +10-20 Elo with tapering returns.
  Roughly 40h GPU per SB800.
- Priority: shelved-net (Reckless-KB + factor, +30 proven) at SB800
  FIRST, then SB1600 on top of that recipe if the SB800 lands clean.

**Infrastructure — Lichess-visible, SPRT-invisible**:
- Opening book: current Titans.bin is randomly grabbed from the
  internet, untested vs alternatives. Book A/B + book-move-sanity-
  checking are 1-5 Elo each, only measurable on Lichess.
- TB entry timing + DTZ walkback quality: just added DTZ walkback
  2026-04-24; remaining questions include whether entering TB the
  moment popcount drops is optimal at all strength levels.
- Time management edge cases: stockpile bugs, forced-move
  soft_floor, increment flooring — visible on Lichess bot as time
  forfeits or poor practical play. Hard to SPRT.
- Parallel search (Lazy SMP): v9 T=4 blunder bug known, Coda deploys
  at T=1 on Lichess as result. Fixing this unlocks more effective
  strength at the deployment without code changes.

**Thread-selection heuristic** — when picking the next experiment:

1. What thread is this in? (Forces explicit framing vs reactive
   picking.)
2. What's the last win in this thread, and how recent? If a thread
   hasn't delivered in 4+ weeks, pick it — diversity of attempts.
3. For flywheel/correctness/comparative/training: run SPRT at [0, 3]
   (default) or [-3, 3] (correctness fix). For infrastructure:
   Lichess-watch + manual inspection.

## Key Gotchas
- Move flag equality vs bitwise: check non-promotion flags with ==, not &
- EP moves only valid when EP square is empty (occupied square = corruption)
- TT stores raw (uncorrected) eval to avoid double correction on probe
- Correction history only updated when bestScore > originalAlpha
- **is_pseudo_legal must be thorough**: TT hash collisions inject illegal moves. Pawn validation must check direction, intermediate squares (double push), starting rank, destination empty (pushes), enemy piece (captures). Castling must check rights, path clear, king/intermediate/destination not attacked, king on correct square. All three bugs cost 320 Elo combined.
- **PV error warnings = TT collision bugs**: Every "Illegal PV move" from cutechess-cli means a TT collision passed is_pseudo_legal and corrupted the search tree. Treat as critical, not cosmetic.
- **Feature flag ablation**: env var controlled flags (NO_XXX, ENABLE_XXX, DISABLE_ALL) for systematic search feature testing. Parsed once at startup via std::sync::Once.
- LMR contHist weight: 3x in move ordering, ply-1+ply-2 in reduction adjustment
- PV nodes skip all TT cutoffs and QS beta blending
- Polyglot book encodes castling as king-to-rook (must convert to king-to-destination)

## Code Hygiene

- **Keep compiler warnings at zero.** `cargo build --release` should emit no
  warnings. Warnings accumulate into noise that masks real issues — every
  new warning is one that hid the serious one next to it. When adding
  code, run `cargo build --release` and fix any new warnings before
  committing. When a warning IS intentional (e.g. a placeholder), suppress
  it explicitly (`#[allow(unused_variables)]`) so the intent is clear.
- Common patterns: unused imports → delete; `let mut x = ...` that's
  never mutated → drop `mut`; `let mut x = false;` then reassigned before
  read → move declaration into the scope where it's used.

## Testing Methodology

### CRITICAL LESSON (2026-04-01)

Narrow cross-engine gauntlets (3 engines, 200 games) gave us inflated Elo estimates (+20 to +67) for changes that self-play SPRT later proved were 0 to -17. The narrow gauntlet overfits to specific opponents just like self-play overfits to shared eval blindspots. **Never merge based on narrow gauntlet results alone.**

Self-play SPRT with tight bounds is the primary acceptance criterion. It is disciplined (runs until statistically significant), reproducible, and — while it has known limitations — the direction of its results consistently matched broader cross-engine testing in our validation experiments.

### Self-Play SPRT (primary acceptance test)

All search/eval changes must pass self-play SPRT before merging.

**Default bounds: `[0.00, 5.00]`** — standard for all changes. Resolves in 2-20K games depending on effect size. For small gains (+2-3 Elo), consider `[-3.00, 3.00]` which resolves genuine +3 in ~15K games.

**LTC testing (40+0.4)** for time management changes — TM features are invisible at STC (10+0.1) where each move gets ~200ms. Node-based TM failed 3x at STC but passed at +11.9 LTC.

**SPRT via OpenBench** (preferred):
```bash
# Let OB auto-detect bench from commit messages:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch>

# With explicit base commit (if main ref is stale):
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch> --base-branch <commit>

# Custom TC or bounds:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch> --tc '40.0+0.4' --bounds '[-3, 3]'
```

**Key rules:**
- One change per branch. Never stack untested changes.
- Wait for H0 or H1. Do not stop early based on "looks good".
- H0 = reject. H1 = accept. Log result to experiments.md.
- For tree-shape-changing features: retune-on-branch before deciding (see methodology below).
- Do not pass explicit bench values unless OB fails to auto-detect.

### Commit Messages

**Every commit that changes search/eval must include `Bench: <nodes>` in the commit message.** OpenBench uses this to verify the correct binary was built. Run `coda bench` with the production net.

```
Fix razoring margin at depth 2

Bench: 1780721
```

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

**Choosing SPRT bounds** — encode your prior belief about the change:

| Bounds | When to use | Example |
|--------|-------------|---------|
| **`[0, 3]`** | **Default for "does this feature help?"** at Coda's current strength. Small-gain regime: most new ideas target +1-3 Elo. | Small pruning tweak, small bonus adjustment, incremental feature |
| `[0, 5]` | Expect clearly positive gain +3-5 Elo or structural change with measurable impact. | Multi-param port, force-more-pruning w/ retune, new tactical motif |
| `[-3, 3]` | Small-win correctness fix where regression ≤ -3 would be a block. | 50mr mate downgrade, stale-bound gate, SE-margin tweak |
| `[0, 10]` | Expect big gain from a named structural change. | SEE magnitude fix, EBF-massive rewrite, major algorithm port |
| `[-5, 5]` | Pure non-regression / infrastructure check. | Adding tunables at default values, NPS-only bench-neutral change |

**`[0, 3]` is the new default** for most experiments in 2026. At our
current rating (~CCRL 3570) the easy Elo is gone and most ideas that
land deliver +1-3 Elo. `[0, 5]` accumulates negative LLR on a true
+1.5 ±1.5 effect because H1 target 5 is out of reach — we'd waste
fleet cycles proving something we could have accepted with `[0, 3]`.
**Reckless uses [0, 3] as their default** (see
recklesschess.space/index); adopting the same discipline means we
don't throw away genuinely-positive small wins.

**Why tight bounds matter strategically**: the gap from current Coda
to 2nd-best-engine-in-the-world is genuinely countable. At CCRL
scale it's ~77 Elo (~16 × +5 Elo wins); at ultra-bullet scale it's
~200 Elo (~40 × +5 wins). Either way the remaining work is a
**finite, enumerable list of small wins** — not "insurmountable
years of catching up." This reframes how to value a +5 Elo change:
it's **1/16th to 1/40th** of the path to the #2 position in the
world. Not marginal. Stack them, don't dismiss them. The wide-bounds
SPRT that misses a real +3-5 win is throwing away 2-6% of the total
path to #2.

Use `[0, 5]` or `[0, 10]` only when the change class has a
well-grounded prior for a larger magnitude (structural port,
retune-on-branch of multi-merge cluster, algorithmic rewrite).

**Don't use `[-10, 5]`.** In a regime where single experiments target +1-3 Elo, a floor of -10 is too permissive — H1 can fire on data that still leaves room for a small negative true effect. Use `[-3, 3]` for correctness fixes instead.

When picking, don't hedge toward wider bounds out of uncertainty. Name the class ("small feature", "correctness bug", "structural port", "NPS-only"), pick the matching row, submit.

**What does NOT need SPRT:**
- Comments, documentation, tooling changes
- Code cleanup that doesn't change compiled output (verify with bench)
- New tunables at default values that don't change behavior (verify bench unchanged)

**OpenBench scripts** (all require `OPENBENCH_PASSWORD` env var, username defaults to `claude`):
```bash
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch> <bench> [--bounds '[0,10]'] [--priority 1]
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_stop.py <test_id>
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_status.py
```
Default bounds [0.00, 5.00] for novel changes. Always verify bench matches before submitting.

**Reference NPS defaults to 250000** (v9 is main since 2026-04-24 merge).
V9 runs at ~240-280K NPS per core; the default fits everything on main
and all standard feature branches. Explicit v5-only work (legacy
bullet_convert experiments, v5 net comparisons) must pass `--scale-nps
500000`. Wrong scale_nps means games run at wrong time budgets —
experiments at 500K on v9 code run at ~2× wall-clock, halving fleet
throughput.

**Bench lives in git commit messages, not this file.** Measure with
`coda bench -n <current-prod-net>` on the exact branch you're submitting;
never hardcode a bench value here or reuse one across branches
(branch-specific tunables + net changes both move it). See `docs/net_catalog.md`
for the current production net hash.

**Do not pass explicit bench values** when submitting — let OB auto-detect from
commit messages. Only override if OB fails to parse.

### SPSA Parameter Tuning

Search parameters are exposed as UCI options for SPSA optimization via OpenBench.
The `tunables!` macro in `search.rs` is the single source of truth for defaults,
ranges, and c_end values.

**CRITICAL RULE — always tune against the net in `net.txt`:**

Trunk's param defaults and the currently-deployed production net
must stay calibrated together. If trunk is tuned for Net-A but
we deploy Net-B and run SPRTs against Net-B, every result is on
a **detuned baseline** — usually costing 5-15 Elo of baseline
strength and producing confusing SPSA movements on top.

Before firing any trunk retune:

1. `cat net.txt` on trunk (main; v9 merged 2026-04-24)
2. The filename in net.txt IS the production net for that trunk
3. Pass the matching `--dev-network <SHA8>` to `ob_tune.py`
4. If the SHA doesn't match, DO NOT SUBMIT — fix the mismatch
   (either update net.txt for a new prod, or use the correct net)

**This is not optional.** We burnt on this 2026-04-24: #682/#686
retune was on factor-SB400, deployed net stayed prod-SB800, and
subsequent SPRTs ran on a silently-detuned trunk. Only noticed
via confusing movements in tune #733.

**Pre-merge check for any experiment:**

Any branch proposed for merge to trunk must have been SPRT'd
against the current net in net.txt. If the dev-network used in
the submission doesn't match, don't merge — re-SPRT with the
right net first.

**Post-tune / net-deploy discipline:**

- When a net changes (new prod in net.txt), plan an immediate
  trunk retune against the new net before landing large clusters
  of eval-dependent experiments on top.
- Don't ship "new net with old trunk" as a hidden detune.
- Periodic SF H2H (100 games, 60+1) every ~5-10 merges to
  confirm trunk still holds; re-tune if the baseline has drifted.

**Submitting tunes:**
```bash
# Submit via script (preferred) — params file covers the tunables you want to sweep:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune.py <branch> [bench] --params-file scripts/tune_nmp_cluster.txt --iterations 1000

# Or with inline params:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune.py <branch> [bench] --params "LMR_C_QUIET, int, 132, 80, 300, 10.0, 0.002
LMR_C_CAP, int, 164, 100, 350, 10.0, 0.002"

# For full-sweep tunes, generate the spec from the tunables! macro:
#   python3 -c 'import re; [print(f"{m[1]}, int, {m[2]}, {m[3]}, {m[4]}, {m[5]}, 0.002") for m in (re.match(r"    \\((\\w+), (-?\\d+), (-?\\d+), (-?\\d+), ([\\d.]+)\\),", l) for l in open("src/search.rs")) if m and m[1] != "name"]' > scripts/tune_all.txt
# (skips the macro-definition line). See scripts/tune_force_more_pruning.txt for an example of the full sweep.

# Bench is auto-detected from commit message. Pass explicitly only if OB can't parse it.
```

**Reading tune results:**
```bash
# Show all active tunes with parameter values and % change:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune_status.py

# Show specific tune:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune_status.py 175

# Get SPSA outputs (for applying to code):
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune_status.py 175 --outputs

# Compare two tunes side by side:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune_status.py --compare 175 176

# Raw digest API (used by scripts):
# GET /api/spsa/{tune_id}/digest/   → CSV of current values
# GET /api/spsa/{tune_id}/outputs/  → SPSA input format with current values
# GET /api/spsa/{tune_id}/inputs/   → Original SPSA inputs
```

**SPSA format per parameter:** `NAME, int, default, min, max, c_end, r_end`

When LMR_C_QUIET or LMR_C_CAP change, LMR tables are automatically reinitialized.

**Practical guidance:**
- 2500 iterations (×8 pairs = 40000 games) is standard. Values stabilise by ~800 iterations.
- Focused tunes (4-8 params) need only ~1000 iters; full-sweep (60+ params) benefits from 2000-2500.
- c_end ~5-10% of parameter range, r_end 0.002 are good defaults.
- Alpha 0.602, gamma 0.101, A_ratio 0.1 (standard SPSA constants).
- SPRT the final values against main before merging — SPSA can overfit.
- Plan SPSA after merging structural fixes (eval/search changes shift optimal parameters).
- Focused tune specs for common clusters: `scripts/tune_nmp_cluster.txt`
  (NMP), `scripts/tune_history_shape.txt` (history-bonus shape),
  `scripts/tune_caphist_focused.txt` (capture history).

### Retune-on-Branch Methodology (discovered 2026-04-07)

Some features are neutral without retuning but gain significant Elo when pruning parameters are recalibrated on their branch. The workflow:

1. **Create feature branch** on current main
2. **Submit SPSA tune** on the branch (same 18 pruning params as baseline)
3. **Compare parameter convergence** against a baseline tune on main
4. **If parameters diverge significantly** (>5% on multiple params): the feature is shifting the search landscape. Apply tuned values and SPRT the branch+tune against main.
5. **If parameters converge to same values as main**: the feature is truly neutral, drop it.

**Validated examples:**
- TT PV flag: +4.5 raw → retune added +4.0 more (nearly doubled)
- Cont-hist malus: flat (-0.15 at 16K games) → +6.5 with retune
- Pattern: big bench/node change but flat Elo → retune candidate

#### Guard / Safety-Gate Sub-Pattern (2026-04-24)

A special case of retune-on-branch: experiments that ADD a guard
around an existing pruning feature (e.g. "skip NMP when TT move is a
capture", "don't LMR when X", "require Y before probcut").

A vanilla SPRT of the guard at default tunables measures only the
DIRECT safety gain — "these specific unsafe prunes no longer happen."
It does NOT measure the REBALANCING gain: adjacent pruning tunables
were globally calibrated to avoid the unsafe case the guard now
handles. With the guard in place, those tunables have latent headroom
they're no longer using — the rest of the pruning can become more
aggressive because the worst case is defended.

**Right workflow for guard experiments:**

1. SPRT the guard at default tunables (confirms direction isn't
   negative and captures the direct gain)
2. SPSA retune the ADJACENT cluster on branch (e.g. for an NMP
   guard: NMP_BASE_R, NMP_DEPTH_DIV, NMP_EVAL_DIV, NMP_EVAL_MAX,
   NMP_VERIFY_DEPTH — 5-6 params, 1200-1500 iters is usually
   enough for a focused cluster)
3. SPRT guard + retuned-cluster vs trunk

A guard that SPRTs at +1 without retune is often +3-5 with retune —
same multiplier as TT PV flag and cont-hist malus. Don't drop a guard
at +1 Elo without completing step 2.

**Test #732 (NMP TT-noisy R++) 2026-04-24** is a concrete example:
the guard added `r += tt_move.is_capture()`, SPRT'd at +1.1 ±1.9 /
30K games trending H0 on [0,3] bounds. Retune-on-branch of the NMP
cluster is the missing step, not a new feature.

### Cross-Engine and Model Testing

Self-play SPRT is the primary acceptance criterion for search changes.

**Model comparison** (different NNUE nets): Self-play H2H is effective — two nets in the same engine. The eval biases are in the nets, not the search. 200+ games gives a fast signal.

### Known Testing Pitfalls

- **CPU contention**: Ensure CPU cores are fairly idle before starting SPRT tests. Background load (other tests, RRs, compilation) halves effective TC and distorts marginal results. In our ablation sweep, CPU contention flipped history pruning from +3 (false H1) to -17 (real H0) and correction history from -3 to -10. Always check `htop` or equivalent before launching tests.
- **Narrow gauntlet inflation**: 3 engines × 200 games gave +67 for a change that was actually 0. Use SPRT instead.
- **Coda-on-Coda contamination**: Multiple Coda variants in a RR amplify shared eval biases. Keep to max 2 Coda variants.
- **Optimism bias**: Stopping a test early because "it looks positive" leads to false positives. Let SPRT decide.
- **False negatives**: A change rejected at -5 with wide error bars might be +5. Retest when conditions change.
- **WDL blindspot**: WDL blending improvements are invisible in self-play. Must use cross-engine RR.
- **Self-play discount**: Self-play Elo ≠ cross-engine Elo. Direction is usually reliable, magnitude varies.
- **Dig deeper on consensus H0s**: When a feature that every top engine uses fails (H0), the feature isn't wrong — your implementation or its dependencies are broken. Don't accept "doesn't work for our engine." Ask **why** it doesn't work, compare actual numeric values with reference engines, and look for magnitude/scaling bugs. Example: dynamic capture SEE H0'd three times because capture history values were 27× too small. Fixing the magnitude (+31.6 Elo) was the biggest single gain in Coda's history, discovered by refusing to give up on a consensus feature.

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
- **Full tune after** (all tunable parameters, 2500+ iterations): Rebalances everything — other params were compensating for the broken feature and need to readjust.
- Merge the focused tune values, then run the full tune as the next round.

**5. Repeat**
The retuned baseline exposes the next weak feature. Check SPSA trends for the next parameter being aggressively detuned.

**Historical examples (early 2026 — kept as illustrations of the method):**
- SEE quiet pruning: SPSA flagged, 3 structural issues found, +11.4 Elo
- History magnitude: consensus H0 investigated, found 27× scaling bug, +31.6 Elo
- LMR simplify: removed unique adjustments + retune, +6.3 Elo
- Cont-hist malus: fixed indexing bug + retune, +6.5 Elo
- NMP capture R: consensus flip + retune, +3.5 Elo
- Node-based TM: failed 3x at STC, retested at LTC, +11.9 Elo

For the current cumulative list, see `experiments.md`.

