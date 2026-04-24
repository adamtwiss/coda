# CLAUDE.md — Coda Chess Engine

Coda is a UCI chess engine written in Rust. Rewritten from GoChess with all accumulated knowledge.

**Chess Optimised, Developed Agentically** — built through human-AI collaboration.

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

All parameters are SPSA-tunable via the `tunables!` macro in `search.rs` (45 parameters).
Current values reflect SPSA rounds 1-10 + retune-on-branch calibration. See the macro
for authoritative defaults — values below are approximate.

- SEE values: P=100, N=320, B=330, R=500, Q=900
- History bonus: linear formula min(MAX, MULT*depth - BASE), ~170*d-50 capped at 1505
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

- 60+1 bullet H2H vs SF 18: **Coda −159.8 ±41.6 Elo** (100 games,
  57% draws, 43 SF wins, 0 Coda wins).
- CCRL 40/15 top 10 sits at **3633–3652** (SF 18 #1 at 3652,
  Reckless 0.9.0 #2 at 3647). Rivals pool spans 3434–3585;
  Coda is below the CCRL top-100 cutoff (3390) in the posted
  snapshot but effectively ~3480–3500 on current trunk +
  SB800 prod net.

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

**Effective depth** is the target. Effective depth = f(pruning
efficiency, raw NPS, eval quality). Ranked by leverage for the
SF-gap class of loss:

1. **Pruning efficiency** (biggest lever). Same NPS, more
   effective depth. Two-row LMP, NMP refinements, LMR
   recalibration, direct-check exemptions. This is Reckless's
   playbook — they're slower than SF in raw NPS but #2 on CCRL
   via better pruning + eval.
2. **Raw NPS** (secondary). Cache residency (flatten AccEntry,
   eval-TT writeback, L1 permutation, training-side matrix
   shrink), compile-time SIMD dispatch, prefetch hygiene.
3. **Eval quality** (contributes indirectly). A cleaner eval
   lets pruning/ordering decisions be sharper, which amplifies
   lever 1. But better eval alone doesn't stop tactical cliffs
   — SF's cliffs happen at deeper plies and are out of Coda's
   search reach regardless of eval signal.

### Priors this updates

- **Coda deliberately traded NPS for eval quality** via v9
  threat-inputs (Reckless architecture). The ROI on that trade
  only realises if the better eval translates to better pruning
  decisions → greater effective depth. Net Elo > net NPS.
- **Raw NPS gains are only as valuable as their depth conversion
  ratio.** A 5% NPS win that doesn't translate to measurable
  depth (because pruning is the bottleneck) is worth less than
  a 2% NPS win in a depth-limited regime.
- **Reckless at #2 CCRL (3647) despite being slower than SF**
  demonstrates the limit isn't NPS. Before proposing "just more
  NPS" to close the gap, compare Coda's pruning parameters
  against Reckless/Obsidian/Viridithas and look for systematic
  under-pruning.
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
| `[0, 5]` | Novel feature, expect small gain, confirm not harmful. | New search heuristic, new bonus in movepick |
| `[-3, 3]` | Small win / correctness bug fix. Tight resolution around zero. | 50mr mate downgrade, stale-bound gate, SE-margin tweak |
| `[0, 10]` | Expect meaningful gain from structural fix. | SEE quiet fix, futility reimplementation |
| `[-5, 5]` | Pure non-regression check. | Adding tunables at default values, NPS-only changes |

**Don't use `[-10, 5]`.** In a regime where single experiments target +1-3 Elo, a floor of -10 is too permissive — H1 can fire on data that still leaves room for a small negative true effect. Use `[-3, 3]` for correctness fixes instead.

When picking, don't hedge toward wider bounds out of uncertainty. Name the class ("correctness bug", "novel feature", "NPS-only"), pick the matching row, submit.

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

**Reference NPS is auto-detected from branch name** (as of 2026-04-19):
- `feature/threat-inputs`, `experiment/*`, `tune/v9-*`, `fix/threats-*` → **250K** (v9 runs slower due to threat features)
- main / other → **500K**

Both `ob_submit.py` and `ob_tune.py` print `[auto] scale_nps=...` when the auto-detect fires. Verify it matches expectation before each submission. Wrong scale_nps means games run at wrong time budgets; experiments at 500K on v9 branches run at ~2× wall-clock, halving fleet throughput. Override only for branches the patterns don't cover — and when you do, add the new pattern to `v9_patterns` in both scripts.

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

**Submitting tunes:**
```bash
# Submit via script (preferred):
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune.py <branch> [bench] --params-file scripts/tune_pruning_18.txt --iterations 2500

# Or with inline params:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_tune.py <branch> [bench] --params "LMR_C_QUIET, int, 132, 80, 300, 10.0, 0.002
LMR_C_CAP, int, 164, 100, 350, 10.0, 0.002"

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
- c_end ~5-10% of parameter range, r_end 0.002 are good defaults.
- Alpha 0.602, gamma 0.101, A_ratio 0.1 (standard SPSA constants).
- SPRT the final values against main before merging — SPSA can overfit.
- Plan SPSA after merging structural fixes (eval/search changes shift optimal parameters).
- A standard 18-param pruning tune spec is at `scripts/tune_pruning_18.txt`.

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

