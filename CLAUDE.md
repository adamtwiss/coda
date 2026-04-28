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
make                  # Build with embedded NNUE net + native CPU (produces ./coda)
make pgo              # PGO build — helps v5 on main; regresses v9. See Makefile.
make net              # Download production NNUE net (from net.txt)
make openbench        # OpenBench-compatible build (alias for `make`)
cargo build --release # Plain release (no embedded net) — NOT what OB workers use
cargo test            # Run all tests including perft

./coda bench [depth]                  # Search benchmark, default depth 13 — use THIS for OB
./coda                                # UCI mode
./coda -nnue net.nnue                 # UCI with explicit NNUE
./coda -nnue net.nnue -book book.bin  # ... + opening book
./coda epd wac.epd 1000 0 -nnue net.nnue
./coda perft [depth] [fen...]
./coda perft-bench                    # 6-position perft suite
./coda datagen [options]              # Self-play / material-removal data
./coda convert-bullet [options]       # quantised.bin → .nnue
./coda convert-checkpoint [opts]      # .nnue → Bullet checkpoint
./coda fetch-net                      # Pull net from net.txt URL
./coda sample-positions [options]     # binpack → EPD samples
./coda help
```

**Bench-for-OB ritual.** OB workers use `make` (which emits `./coda` at the
repo root via `--emit link=coda`). To get a bench number that matches what
OB will measure, **always** `make && ./coda bench`. `cargo build --release`
+ `./target/release/coda bench` produces a DIFFERENT number (no embedded
net, different code paths) — submitting that bench triggers OB "Wrong
Bench" rejections.

After every SPRT submission, check https://ob.atwiss.com/errors/ 5-10 min
later before assuming success — workers report build/bench errors async,
not in the test page or `ob_status.py`. Don't resubmit on stall without
checking errors first.

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
Production architecture is **v9** (FT + threats → hidden layers). The inference path
also supports legacy v5 (CReLU/SCReLU/pairwise direct FT→output) and v7 (FT→hidden→output)
formats for compatibility with retired nets. Current prod hash and active nets live in
`docs/net_catalog.md`.

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

GPU hosts use the forked Bullet trainer at `~/code/bullet` (clone
`adamtwiss/bullet`) plus a Coda checkout at `~/code/coda` for the converter.
Both `cargo build --release` once after cloning.

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

### Transfer Learning (Frozen FT)

The `adamtwiss/bullet` fork adds `freeze()`/`unfreeze()` support for per-weight
training control. Used historically for v5 → v7 transfer; same machinery is
available for v9 hidden-layer experiments. Use
`coda convert-checkpoint -nnue <src.nnue> -output <ckpt> -l1 16 -l2 32` to
seed a Bullet checkpoint from an existing .nnue.

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
- **WDL blend**: w0 better than w5 for v7-style hidden-layer nets (+30 Elo); w3-w5 equivalent for v5.
- **12-file training data** gives +33 Elo over 6-file for 768pw (data diversity matters).
- **Low final LR is critical**: cosine `final_lr 0.0001` was 20× too high. Reducing to ~5e-6 (Bullet examples use `0.001 * 0.3^5 = 2.4e-6`) gave **+47 Elo** — net was oscillating, not converging.
- **Data filtering**: quiet positions only (ply≥16, no checks/captures/tactical moves) gave **+22 untuned, +48 tuned**. Aligns with how NNUE is consumed (quiet nodes after QS).
- **v9 low-LR tail is load-bearing**: sparse threat features keep converging deep into the tail; SB400 → SB800 delivered +88 Elo on v9 (vs flat on v7). v9 wants LOWER final LR than v5's 5e-6. See `memory/project_v9_low_lr_tail_critical.md`.

### EVAL_SCALE Calibration

`EVAL_SCALE` (nnue.rs, default 400) converts raw network output to centipawns.
Different training configs (filtering, WDL, LR) produce different eval scales.
When the scale changes, all search thresholds (RFP, futility, SEE, LMR) become
miscalibrated. **Preferred fix is SPSA retune** — recalibrates all thresholds
to the net's natural scale. EVAL_SCALE adjustment is a quick hack.

To measure the scale of a candidate net: bench `coda eval` over 500 sampled
positions and compute RMS; baseline is ~580 for prod-tuned nets. **Pairwise
nets do NOT scale linearly with EVAL_SCALE** — large values overflow int8
quantization. Always verify RMS empirically; never compute it as
`400 * baseline / candidate`.

Standard Bullet LR schedule for hidden-layer nets: linear warmup
0.0001→0.001 over 5 SBs, then cosine 0.001→`final_lr` over N-5 SBs.
For v9, target final_lr ~2.4e-6 (Bullet examples) — Coda's earlier 1e-4
was 20× too high (cost +47 Elo when fixed).

## NNUE Model Naming Convention

Format: `net-v{N}-{accumWidth}[t][d]h{layers}[s]-w{wdl}-e{epochs}s{snap}.nnue`

- **`v{N}`**: architecture generation. v5 (direct FT→output), v7 (FT→hidden→output), v9 (FT + threats → hidden → output, current production).
- **`accumWidth`**: accumulator width per perspective. For v9+ this is literal (`768t` = 768 accum + threats). Legacy v5/v7 names confusingly use the input feature count `768pw` to mean 1536 accum on v7 — see git history if decoding old names.
- **`t`**: threat features present (v9). **`d`**: dual L1 activation (CReLU+SCReLU).
- **`h{layers}`**: hidden sizes — `h16` (L1=16), `h16x32` (L1=16, L2=32).
- **`s`**: SCReLU activation (else CReLU).
- **`w{n}`**: WDL blend ×100 — `w0`, `w5`, `w15`, etc.
- **`e{N}s{M}`**: total superbatches / snapshot checkpoint.

Example: `net-v9-768th16x32-w15-e800s800.nnue` — v9 prod-shape (768 accum,
threats, 16→32 hidden, w=0.15, full 800-SB run).

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

## Strength Frontier — Where the Elo Gap Lives

Findings from a 100-game bullet H2H vs Stockfish 18 and Atlas's
loss-pattern analysis. This is a working hypothesis, not dogma —
update as more data accumulates.

### Calibration

Three anchor points:

- **60+1 bullet H2H vs SF 18 + Reckless** (2026-04-27, Atlas, 100 games per pairing):
  - vs SF: **−210 ±48** Elo (0 wins / 46 draws / 54 losses)
  - vs Reckless: **−151 ±40** Elo (0 wins / 59 draws / 41 losses)
  - Combined: −179 ±31, 26.3% score, 52.5% draw rate, **0 wins in 200 games**
  - Loss-mechanism analysis (`scripts/classify_losses.py` on the gauntlet PGN):
    - 78.9% HORIZON-class (Coda eval cliffs late while opp eval was steady-correct earlier)
    - 7% POSITIONAL drift, 10.5% SELF_BLUNDER, 3.5% UNCLASSIFIED
    - **Median cliff-ratio 0.89** — eval drop happens at 89% through the game
    - 31 of 45 HORIZON losses (69%) cliff in the endgame or final 5%
    - Implication: the gap is concentrated in **endgame depth/eval**, not middlegame
  - **Lag-near-zero finding**: when both sides "commit" to the bad evaluation
    (5-ply consistency at threshold), median lag is −3 plies vs SF, −1 vs Reckless.
    Both engines see the loss at roughly the same time. This rebuts the
    naive "SF outsearches us 10 plies deep" reading of HORIZON. The mechanism
    is more **eval-refinement** than search-depth: Coda's eval underestimates
    badness by 1-3 pawns persistently (mean late-game eval gap 354cp vs SF),
    until the position becomes concrete enough that even our less-refined
    eval catches up.

- **EGTB + Hash + TC sweep** (2026-04-27, 200 games per cell, all on prod 1EF1C3E5):

| Configuration | Overall | vs SF | vs Reckless | Draw % |
|---|---:|---:|---:|---:|
| 60+1, hash=64, no EGTB (baseline) | −179 ±31 | −210 ±48 | −151 ±40 | 52.5% |
| 60+1, hash=64, EGTB on | −186 ±32 | −182 ±45 | −191 ±46 | 51.0% |
| 60+1, hash=512, EGTB on | −151 ±29 | −139 ±40 | −164 ±42 | 58.0% |
| **180+2, hash=512, EGTB on (deployment-config)** | **−119 ±25** | **−139 ±39** | **−100 ±33** | **67.0%** |

  - **EGTB alone (hash controlled, 60+1): essentially flat overall (-7 within noise)**.
    Mixed by opponent: maybe small help vs SF (+28), maybe small hurt vs Reckless
    (-40), both within ±60 combined error bars.
  - **Hash 64→512 (60+1, EGTB on): +35 Elo overall, +43 vs SF, +27 vs Reckless,
    +7% draw rate.** Dominant lever vs SF.
  - **TC 60+1 → 180+2 (with hash=512+EGTB): +32 Elo overall, +0 vs SF, +64 vs Reckless,
    +9% draw rate.** Dominant lever vs Reckless. SF-side gap is TC-saturated by 60+1.
  - **Mechanism split made cleanly visible by this 4-config sweep:**
    - vs SF: gap is **hash/depth-bound**. Closes -71 Elo (210→139) with hash bump.
      Doesn't close further with TC. SF outsearches us in tactical density that
      hash size partly compensates for; once we have the hash, SF's edge is
      fully manifest.
    - vs Reckless: gap is **TC/eval-refinement-bound**. Hash bump barely moves
      it (-13, possibly null at this N). Long TC closes it -64 (164→100).
      Tracks the eval-refinement frame: at long TC both engines see more,
      and Coda's eval has more headroom to converge on positions Reckless
      already evaluated correctly.
    - These are complementary mechanisms, both load-bearing for different
      opponents. Combined, the deployment-config 180+2+512+EGTB number is
      the relevant strength anchor.
  - **First wins ever against SF/Reckless** (1 win at hash=512+EGTB 60+1, more
    at 180+2). Qualitative crossing of zero — but 200 games per cell is still
    a small sample.
  - The earlier "EGTB closes the SF gap by 86 Elo" reading was largely a
    TT-size artifact, not an EGTB effect.

  **Action items from this finding:**
  - **Strength claims should cite the 180+2 deployment-config anchor** (-119 ±25),
    not the 60+1 H2H number. The 60+1 numbers are diagnostic instruments
    for mechanism decomposition, not strength reads.
  - SF gap is hash-bound; further tightening requires search-side work
    (EBF, NPS, tactical density). Pure eval improvements won't close it
    much further.
  - Reckless gap is TC-bound; eval-refinement work (training, factor net,
    longer SBs) closes it. The path to peer-tier strength runs through
    the Reckless gap first per `feedback_sf_vs_reckless_gaps_are_different.md`.
  - Investigate Coda TT efficiency in long endgames: hit rate at
    piece-count buckets, replacement effectiveness at low piece count.
    See `docs/tt_hash_sensitivity_2026-04-27.md` for the structural
    candidates (bucket density, age weight, QS in-check static_eval).
  - SPRT at 10+0.1 measures search/eval changes well, but is structurally
    blind to a specific class: changes whose Elo only manifests at long-TC
    mechanisms (deep-search TT pressure, ponder, TM nuance, >25-ply tactics).
    OB hash CAN be bumped if we audit RAM headroom (workers run N concurrent
    games × hash MB); we just don't currently. For hash/TT-sensitive merges,
    validate via 180+2 H2H gauntlet alongside SPRT.

- Earlier baseline (recorded here for trajectory): −159.8 ±41.6 vs SF (CI overlaps the new −210, so the gap may not have widened — but it also hasn't tightened despite recent merge cluster)
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

- **Rivals gauntlet — 40+0.4, hash=512MB, EGTB-on, 8 engines, 1400 games**
  (Adam, 2026-04-28, FINAL: 200 games per opponent, the "near-peers" pool):

  | Rank | Engine | Elo | Score | Bench NPS | Speed×Coda | Δ-to-Coda |
  |---:|---|---:|---:|---:|---:|---:|
  |  1 | Horsie | +58 ±23 | 58.3% | 1,052,223 | 1.78× | +74 |
  |  2 | Tarnished | +56 ±24 | 58.0% | 1,019,000 | 1.72× | +72 |
  |  3 | PZChessBot | +17 ±26 | 52.5% | (n/a) | (n/a) | +33 |
  |  4 | Seer | +14 ±25 | 52.0% | 1,500,726 | **2.54×** | +30 |
  |  5 | Velvet | −7 ±24 | 49.0% | **1,780,148** | **3.01×** | +9 |
  |  6 | Clarity | −12 ±26 | 48.3% | 1,690,952 | **2.86×** | +4 |
  |  7 | Arasan | −14 ±26 | 48.0% | 838,335 | 1.42× | +2 |
  |  8 | **Coda** | **−16 ±9** | 47.7% | **591,523** | 1.0× | 0 |

  Bench NPS measured 2026-04-28 on Hercules with OB worker stopped
  (clean CPU). PZChessBot's bench wouldn't terminate cleanly under
  any invocation tried; revisit. Reference values: Reckless 942K,
  Stockfish 1.12M.

  **Coda is the slowest in the rivals pool by 1.4-3× margin** — but
  the data argues this is a *deliberate and approximately neutral*
  trade-off, not a problem. Look at the rank correlation:
  - Top 2 strongest engines (Horsie +58, Tarnished +56) = 1.72-1.78× NPS
  - Strongest fast engine (Seer +14) = 2.54× NPS
  - Top-3 fastest engines (Velvet, Clarity, Seer at 2.54-3.01×) = -12 to +14 Elo
  - Coda at 1.0× = -16 Elo

  **NPS doesn't determine rank in this pool**. Tarnished/Horsie are
  the strongest at moderate NPS; the fastest engines (Velvet, Clarity)
  sit mid-pack. Reckless is also a slower-but-strong design pattern
  (942K NPS vs Coda's 591K, only 1.6× faster). v9's threat-input
  architecture cost ~20% NPS for +110 Elo (good trade); we approximately
  match the field's strength despite being slowest. The lever for
  closing the next 50 Elo isn't *only* NPS — it's improvements that
  produce eval/ordering wins per node, of which NPS is one ingredient.

  Trajectory across sample sizes (showing how early reads misled):
  180g →  Coda −31 (sampling)
  240g →  Coda −22
  1400g → **Coda −16 ±9** (final, ±9 tighter than initial ±24)

  - **Coda is mid-pack** in this pool, gap-to-centroid ~22 Elo.
    Velvet sits below us; Clarity/Seer are at our level; Arasan,
    PZChessBot, Horsie, Tarnished are above (the +33 → +103 band).
  - **Loss-class profile** (1400 games / 222 losses analysed via
    `classify_losses.py`): **77.9% HORIZON, 15.8% SELF_BLUNDER,
    1.8% POSITIONAL, 4.1% UNCLASSIFIED, 0.5% MUTUAL_TACTIC**.
    HORIZON-dominance is **mechanism-invariant** across the rivals
    tier — same ballpark as the 78.9% HORIZON vs SF/Reckless
    (210/151 Elo gaps). Tactical density / search-depth mismatches
    drive losses across the entire opposition spectrum we've
    measured.

  **Per-opponent breakdown** (now reliable at 23-43 losses each):

  | Opponent | N | HORIZON | SELF_BLUNDER | Read |
  |---|---:|---:|---:|---|
  | Clarity | 25 | **96.0%** | 0% | Pure horizon |
  | PZChessBot | 34 | 91.2% | 8.8% | Mostly horizon |
  | Tarnished | 43 | 86.0% | 9.3% | Mostly horizon |
  | Velvet | 23 | 82.6% | 13.0% | Standard mix |
  | Arasan | 25 | 76.0% | 16.0% | Mild SELF_BLUNDER bias |
  | Horsie | 41 | 70.7% | **19.5%** | High SELF_BLUNDER + strong |
  | **Seer** | 31 | **45.2%** | **41.9%** | THE outlier |

  - **Seer is the durable per-opponent outlier**. 13 of 35 total
    pool SELF_BLUNDERs (37%) are vs Seer alone. Seer's style induces
    our second-best moves at ~4× the rate of any other opponent.
    Hypothesis: Seer's positional/quiet emphasis lures us into
    mis-evaluated lines that the more-tactical engines don't
    construct. Concrete investigation: pull a sample of Seer-loss
    PGNs, look qualitatively at the move class (which piece type,
    which phase, which tactical motif) where our eval mis-rates.
  - **Horsie is a secondary SELF_BLUNDER source** (19.5%) AND
    top-of-pool (+58 Elo vs Coda). Likely a different mechanism
    from Seer's: Horsie's strength forces difficult positions where
    we pick second-best, vs Seer's *style* luring eval errors. Both
    worth qualitative investigation.
  - **Clarity is striking — 96% HORIZON, 0% SELF_BLUNDER.** Every
    loss to Clarity is a pure search-depth mismatch. This is the
    cleanest "purely depth-bound" opponent in the pool. Search-side
    work targets the Clarity-class loss directly. Clarity is also
    2.86× faster (NPS), which compounds the depth deficit.
  - **NPS-vs-HORIZON correlation (Adam's hypothesis 2026-04-28)**:
    mostly confirmed. Top 2 fastest engines (Velvet 3.01×, Clarity
    2.86×) → high HORIZON (82-96%). Slowest non-Coda Arasan 1.42×
    → still 76% HORIZON but smallest Elo gap (+2). Seer 2.54×
    breaks the pattern at 45% HORIZON / 42% SELF_BLUNDER — speed
    alone doesn't explain HORIZON when opponent's *style* doesn't
    expose tactical mismatches. **Clean reading**: speed × style →
    HORIZON share. Pure tactical engines (Clarity, PZChess) at any
    NPS dominate via HORIZON; positional engines (Seer) produce
    SELF_BLUNDER-heavy losses regardless of NPS.
  - **SELF_BLUNDER bimodal distribution**: pool aggregate is 15.8%
    but it's bimodal — Seer 42%, Horsie 20%, everyone else 0-16%.
    The "average" hides two distinct subpopulations. When SELF_BLUNDER
    bucket grows in future gauntlets, check whether it's Seer-driven
    style noise or a broader move-ordering issue.
  - **POSITIONAL is essentially zero** (1.8%) vs rivals (was 7% vs
    SF). At peer-tier eval depth, drift losses don't manifest.
  - **An earlier 139-game / 29-loss snapshot showed 20.7%
    SELF_BLUNDER** — reframed once N grew. The "doubling vs SF"
    framing was small-N noise; the 1400-game value (15.8%) is in
    the same band as the 10.5% vs SF/Reckless. **Move-ordering work
    is still valuable** but as mechanism (4) HORIZON-reduction,
    not because peer-tier games have a fundamentally different
    loss profile.
  - **The next 50 Elo target is closing the rivals gap, not chasing
    SF.** Adam's framing 2026-04-28: this pool spans the entire near-
    peer band, and a +50 Elo improvement moves us from −22 to roughly
    parity with Tarnished (+81). Pure SF-gap-closing work has lower
    leverage — vs SF the gap is hash-bound and saturates at ~120-140
    Elo regardless of further search work, but vs rivals every
    EBF/ordering/eval improvement compounds linearly across 6+ peer
    opponents.
  - **HORIZON is an outcome class, not a single mechanism.** When we
    say "75.9% of losses are HORIZON" we mean "the eval cliff signature
    matches search-depth-mismatch." Multiple distinct mechanisms can
    produce that signature:
    1. **Faster NPS** (raw search throughput → more nodes in the
       same wall-clock → more effective depth).
    2. **More pruning** (tighter RFP/FUT/SEE/LMP margins → smaller
       sub-tree per node → deeper effective depth, but at the cost
       of occasionally cutting moves that mattered).
    3. **Less BAD pruning** (carve-outs / better gates that prevent
       pruning the specific moves that were tactically critical —
       e.g. LMP direct-check carve-out, recapture extensions,
       singular extensions, threat-feature-aware gates).
    4. **Better move ordering** (first-move-cutoff rate ↑ → more
       beta cutoffs → less wasted search → deeper effective depth).

    These mechanisms have OPPOSITE risk profiles. (2) gains depth
    by accepting more mistakes; (3) accepts less depth in exchange
    for fewer mistakes. (4) is a free lunch in principle but
    bounded by ordering quality.
  - **Adam's reading of our specific HORIZON-mechanism distribution**
    (2026-04-28): the Reckless outlier-pruning comparison
    (`docs/cross_engine_comparison_2026-04-25.md`) shows we both
    over-prune AND under-prune relative to Reckless on different
    thresholds — five outliers in different directions. This pattern
    suggests our HORIZON losses are more about **(3) less bad
    pruning** than **(2) more pruning**. The wins from "force more
    pruning + retune" cluster were real but bounded; the bigger
    wins (LMP direct-check, recapture extensions, threat-aware
    gates) come from specific carve-outs that prevent mis-pruning
    critical moves.
  - **Implication for experiment selection**: prioritise work that
    targets HORIZON via mechanisms (3) and (4) for our archetype —
    specific pruning carve-outs (Reckless-pattern direct-check gates,
    threat-aware exemptions, recapture/SE extensions) and ordering
    improvements (4D history shape, capture-history sufficiency,
    quiet-check bonus calibration). Mechanism (1) NPS still pays
    but has discount factor at long TC per
    `feedback_nps_elo_conversion_drops_with_tc.md`. Mechanism (2)
    "more pruning" should now be the smaller part of our experiment
    portfolio — most of the easy "tighten the margin" wins are
    captured. De-prioritise SF-specific anchor optimisation
    (deep-endgame TT bucket density was a candidate; lower priority
    now that rivals data exists). The SELF_BLUNDER 20.7% bucket is
    primarily mechanism (4) (better ordering) — second-best moves
    are an ordering/eval-quality artifact, not a depth deficit.
  - **Refresh cadence**: rerun the gauntlet + classifier monthly or
    after any cluster of merges worth ~+10 Elo. The Tarnished/Horsie
    gap should compress visibly as we improve; if it doesn't, the
    work isn't moving the right buckets.
  - **TC sigmoid applies vs rivals too** (Adam, 2026-04-28). The
    SF-gap-shrinks-with-TC pattern documented in §The TC-handicap
    sigmoid below also holds for rivals: gap at 10+0.1 is meaningfully
    larger than at 40+0.4. At 10+0.1 ultra-bullet (the 45-engine RR
    snapshot), Coda rank 21-23/45 cluster with Clarity/Velvet/
    PZChessBot, with gaps to nearby rivals in the 50-150 Elo range
    and gap to top of pool 270-302 Elo. At 40+0.4 (this gauntlet),
    the same engines collapse to within ±20 Elo of Coda — Velvet
    actually ends up below us. Two implications:
    1. **STC SPRT (10+0.1) over-measures our deficit vs rivals** —
       same blindspot as vs SF per `feedback_sprt_blind_to_long_game_effects.md`.
       A change that delivers +5 Elo at STC SPRT may convert to
       less at 40+0.4 deployment TC; conversely, changes whose
       mechanism only fires at LTC (TT pressure, deep tactics, TM)
       may be invisible in SPRT yet pay vs rivals at deployment.
    2. **The rivals gap is more closable than the STC view suggests.**
       Closing 50 Elo at 40+0.4 is moving from -22 to +28 — past
       Tarnished (+81 → ≈+30 post-improvement). At 10+0.1 the same
       improvement might be 100-150 Elo of apparent shift, but the
       deployment-relevant number is the LTC one.
    3. **Rivals-tier validation should be 40+0.4 or 60+1**, not
       SPRT. Reuse the `tough_rivals.pgn` gauntlet pattern after
       any cluster of search/eval merges — it's the deployment-
       relevant strength read for the next 50 Elo target.

SF gap as a function of TC (see TC-handicap sigmoid below for the
full curve):
- 10+0.1 (ultra bullet): ~302 Elo
- 60+1 (bullet): ~160 Elo
- 8× (480+8, past the knee): ~35 Elo
- 16× (960+16): ~0 Elo (parity)

### The TC-handicap sigmoid (last calibrated 2026-04-24)

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
~±10 Elo uncertainty rather than exact points. Sigmoid was calibrated
2026-04-24; recalibration is queued every ~2 weeks or after any merge
cluster worth ~+20 Elo (see §Recalibration cadence below).

**What this tells us:**

- **Our eval is at SF parity** — at equal depth we play as well. The full 160 Elo gap at normal TC is *horizon*, not quality.
- **Horizon is nonlinear in depth vs SF-class opponents.** Partial depth gains below the knee (1-4×) return near-zero Elo; above the knee, each incremental ply delivers 20-30+ Elo. Tactical refutations live at specific depths — seeing them at all matters far more than seeing them half-a-ply earlier.
- **The sigmoid is specific to wide-Elo-gap opposition.** Vs same-tier opponents (Rivals pool, 3430-3585 CCRL), there's no tactical-cliff dynamic — NPS wins convert more linearly to Elo.
- **Compound to cross the knee, or plateau below it (vs SF).** A +3 Elo win pre-knee feels thankless vs SF but is cleanly valuable in Rivals-tier competition (Lichess bot ranking, CCRL position). Stack them.

### Path to closing the gap (pragmatic, 2026 horizon)

To reach the knee (~8× effective depth-gain) we need compounding across all
levers. Realistic per-lever contribution budget:

| Lever | Plausible gain | Path |
|---|---:|---|
| Net architecture / training upgrades | ~30 Elo | Reckless-KB + factor arch each SPRT'd at +15; deploy as SB800 retrain |
| Cache residency + SIMD dispatch | +10-25 Elo | L1 permutation, VNNI dispatch, group-lasso-driven matrix shrink |
| Pruning equilibrium retunes | +5-15 Elo | Force-more-pruning style branches with full-sweep retune |
| Further eval refinement | +10-20 Elo | Post-shelved-net training-recipe iteration |
| Move-ordering improvements | +5-15 Elo | EBF-reducing work; compounds with above |
| **Stacked total** | **~60-100 Elo** | Meaningful jump — top-20 CCRL range |

**Parity (16×-equivalent)** requires dramatic EBF reduction (log(EBF) halved,
1.8 → 1.35) which in turn requires multi-year investment: richer training
pipeline, more net iteration, possibly novel search innovations. Don't expect
parity in 2026.

### What the losses look like

Loss-pattern analysis (Atlas, 27 SF losses): median max single-ply eval drop
4.53 cp; cliffs happen at mid-game → endgame transitions (median ply 83);
Coda reaches median depth 15 in losing games vs SF's 25-35. Coda's eval is
accurate at the depth it reaches — the refuting combination lives at plies
SF sees and we don't. **Tactical blindness from search-depth deficit, not
positional mis-eval.**

### What closes the gap

**Effective depth** is the target. Effective depth ≈ log(NPS × time)
/ log(EBF), so depth = f(raw NPS, pruning efficiency, eval quality).

Decomposition of the ~160 Elo SF gap, rough budget:

| Lever | Approx share of gap | Status |
|---|---:|---|
| Raw NPS deficit (Coda is ~2× slower than Reckless) | **~100 Elo** (~65%) | Some portion is deliberately bought — x-ray threat features cost ~20% NPS but paid +110 Elo (good trade). We won't fully close this; target is partial recovery. |
| Pruning efficiency (Coda under-prunes vs Reckless on several params) | **~30–50 Elo** | Under-exploited; "force more pruning + retune" branches are the lever |
| Eval quality / tactical sharpness | **~20–30 Elo** | Factor-net-quality, training recipe improvements |

**Recoverable NPS** lives almost entirely in **cache-residency improvements
tied to sparsity**. The 49 MB threat weight matrix spills L3 on most fleet
hosts (measured on 9-host microbench; see `docs/nps_microbench_hostdata.md`).
Shrinking it below ~32 MB via training-side L1 regularisation +
Viridithas-style input-chunk L1 permutation gives a step-function NPS gain
across the entire fleet. We don't expect to match Reckless's raw NPS — some
deficit is a deliberate eval-architecture trade. Both NPS (sparsity/cache)
and pruning (Reckless-outlier retunes) attack the SAME target (depth) and
compound.

**Reckless vs Coda pruning outliers — durable pattern.** Coda has
historically diverged from Reckless on at least five thresholds (futility
margin, SEE quiet prune magnitude, RFP/LMP/BNFP depth caps). Each direct-port
attempt has surfaced the same lesson: **raw-Reckless values aren't portable
because our search context differs** (lmr_d vs raw-depth formulas, different
history scaling). The outliers identify real miscalibration; the fix is
SPSA-retune-on-branch, not value import. Holistic "force-more-pruning"
biased-start full-sweeps have repeatedly found new equilibria worth
+5-15 Elo where direct ports H0'd. See `experiments.md` for resolved
specifics; `docs/cross_engine_comparison_2026-04-25.md` for the live queue.

### Priors this updates

- **NPS is a dominant lever (~2/3 of the gap)**, not secondary. 10% NPS ≈ +5-8 Elo at this regime. Cache-residency wins (flatten AccEntry, eval-TT writeback, L1 permutation, training-side matrix shrink) and SIMD dispatch translate directly to depth.
- **Pruning values matter just as much per-param.** Five clear outliers where Coda prunes less than Reckless; each retune-on-branch around a tightened threshold ≈ +2-5 Elo, collectively the same ballpark as NPS wins.
- **Coda deliberately traded NPS for eval quality** via v9 threat-inputs. That trade only pays off if better eval feeds better pruning decisions. Net Elo > net NPS.
- **"Force more pruning"-style branches** (widen an under-pruning threshold, let SPSA find new equilibrium) have worked before and are under-built-on.
- **Experiments that buy depth at hot plies** (extensions on tactical-density signals, cliff-risk heuristics) are a candidate class loss-pattern analysis surfaced. W2-pattern work (signal × pruning/extension) already has high hit rate.

### De-prioritised

- **Eval post-processing tweaks** (contempt, optimism calibration, fortress caps, shuffle detectors) are low SF-gap leverage. Cheap is fine; don't let them displace pruning/NPS.
- **Factor net alone** doesn't close the SF gap — improves eval quality with indirect leverage but doesn't address depth deficit. Still worth doing for Rivals-tier strength + cleaner pruning calibration; don't expect it to crack the cliff-miss class.

### Workflow + recalibration

When sizing up a new experiment, ask: (1) does it increase effective depth?
(2) is Coda's pruning value here an outlier vs top-engine consensus? (3)
would it show up in a 100-game SF bullet H2H? If all three are "no",
expected Elo-per-effort is probably low.

Re-run the 100-game SF H2H every ~2 weeks or after any merge cluster worth
~+20 Elo. Gap should narrow visibly; if it doesn't, on-paper Elo is overstated.

## Improvement Portfolio — Diversified Threads

Beyond the flywheel (eval → ordering → pruning → depth → eval), the
gap-closing strategy runs multiple ORTHOGONAL threads in parallel.
Correlated work can plateau together; diversified threads hedge.
When picking the next experiment, explicitly name which thread it
sits in — it helps avoid the trap of chasing one axis while others
decay.

**Eval-search flywheel** — compounding loop: better eval → better ordering →
safer pruning → more depth → better self-play training data → better eval.
Captured via **retune-on-branch** after each feature: a +1-2 Elo raw feature
is often +5-10 Elo post-retune. Reckless-value imports fail because their
values are calibrated against their flywheel; ours is different. Outliers are
directional signals; SPSA-retune-on-branch is the portable-translation
mechanism.

**Correctness audits** — bugs in rarely-fired paths historically deliver
+3-30 Elo (50-move rule, LMR endgame gate, SMP races, TB integration,
`is_pseudo_legal` EP hole). Highest Elo-per-hour lane, under-invested. If
torn between a +2 Elo feature and a correctness audit branch, audit first.

**Comparative engine review with instrumentation** — reading top-engine
source tells you *what*; patching them with `dbg_hit`-style counters tells
you *when and how often*. Different information class. A half-day of
instrumented comparison often surfaces gaps a full day of source-reading
misses. `scripts/reckless_evalbench.patch` is the entry point.

**Training hyperparameters + data** — biggest historical wins are 1-line
schedule changes (low-final-LR +47, ply≥16 filter +22→+80 with retune).
Still-unexplored: output bucket count, batch size, WDL schedule shape,
data composition additions (Lichess-blunder positions into training set).

**Long tunes / long training** — top engines run 25K-iter SPSA routinely;
our 2-2.5K default leaves small-gradient gains on the table. v9 sparse
threat features converge deep into the low-LR tail (SB400 → SB800 = +88
Elo on v9 vs flat on v7). SB1600+ likely worth +10-20 with tapering.

**Infrastructure (Lichess-visible, SPRT-invisible)** — opening book A/B,
TB entry timing / DTZ walkback quality, TM edge cases (stockpile, forced-move
soft_floor, increment flooring), Lazy SMP correctness (v9 T=4 blunder bug
known; deploys at T=1 until fixed).

**Thread-selection heuristic.** Name the thread first ("flywheel",
"correctness", "comparative", "training", "long-tune", "infrastructure")
before picking the next experiment — prevents accidental concentration on
one axis. If a thread hasn't delivered in 4+ weeks, prefer it.

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

**Keep compiler warnings at zero.** `cargo build --release` should emit none.
Warnings accumulate into noise that masks real issues. Fix or
`#[allow(...)]`-suppress with intent before committing.

## Testing Methodology

### Self-Play SPRT (primary acceptance test)

Narrow cross-engine gauntlets (3 engines, 200 games) overfit to specific
opponents just like self-play overfits to shared eval blindspots. Self-play
SPRT with tight bounds is disciplined, reproducible, and matches the direction
of broader cross-engine testing — primary acceptance criterion. **Never merge
based on a narrow gauntlet alone.**

All search/eval changes must pass self-play SPRT before merging.

**Default bounds: `[0, 3]`** — see the bounds table in §SPRT Testing Policy
below for when to deviate.

**LTC testing (40+0.4)** for time management changes — TM features are invisible at STC (10+0.1) where each move gets ~200ms. Node-based TM failed 3x at STC but passed at +11.9 LTC.

**SPRT via OpenBench** (preferred):
```bash
# Standard submission — pass dev bench explicitly when commit at HEAD lacks
# a Bench: line (e.g. doc-only commits don't have one):
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch> <bench> --base-bench <main_bench>

# Or let OB auto-detect when both HEADs have Bench: lines:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch>

# Custom TC or bounds:
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch> --tc '40.0+0.4' --bounds '[-3, 3]'
```

**Key rules:**
- One change per branch. Never stack untested changes.
- Wait for H0 or H1. Do not stop early based on "looks good".
- H0 = reject. H1 = accept. Log result to experiments.md.
- For tree-shape-changing features: retune-on-branch before deciding (see methodology below).
- Pass explicit `dev_bench` + `--base-bench` whenever there's any chance of staleness, branch-switch confusion, or commit-without-Bench at HEAD.

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

**Choosing SPRT bounds.**

> **Standing policy: `[0, 3]` is the default for ALL "does this feature help?"
> SPRTs.** Reckless uses the same default. Adam reaffirmed 2026-04-25 after
> a `[0, 5]` submission slipped through. Use a wider bound only when the
> prior magnitude is genuinely larger; otherwise tighten to `[0, 3]`.

| Bounds | When to use | Example |
|--------|-------------|---------|
| **`[0, 3]` (DEFAULT)** | "Does this feature help?" at Coda's current strength. Most new ideas target +1-3 Elo. **Pick this unless you have a specific reason for one of the rows below.** | Pruning/ordering tweak, parameter probe, small bonus adjustment, incremental feature, audit correctness fix |
| `[0, 5]` | Prior is for clearly positive **+3-5 Elo or larger** structural change with measurable impact. Reserve for changes whose magnitude has a load-bearing prior, not "could be larger." | Multi-param port with retune, force-more-pruning cluster, new tactical motif |
| `[-3, 3]` | Small-win correctness fix where a regression ≤ -3 would be a block. Use for fixes whose direction is uncertain but where the correctness side of the trade matters. | SE margin tweak, 50mr mate downgrade, stale-bound gate |
| `[0, 10]` | Big gain from a named structural change. Use sparingly — wrong-bound risk in either direction is real. | SEE magnitude fix, EBF-massive rewrite, major algorithm port |
| `[-5, 5]` | Pure non-regression / infrastructure check. | NPS-only bench-neutral change, adding tunables at default values, ARM ordering change |

**Why `[0, 3]`.** Most ideas at our current rating (~CCRL 3570) land in the
+1-3 Elo range. `[0, 5]` accumulates negative LLR on a true +1.5 effect
because H1=5 is out of reach — wasting fleet cycles on something `[0, 3]`
would bank. Reframe: closing the gap to top-engine #2 is a **finite,
enumerable list of small wins** (~16 × +5 Elo at CCRL, ~40 × +5 at
ultra-bullet). A +3-5 win missed by wide bounds is throwing away 2-6% of
the path. Stack them, don't dismiss them.

Use `[0, 5]` or `[0, 10]` only when the change class has a well-grounded
prior for larger magnitude (structural port, multi-merge retune, algorithmic
rewrite). **Don't use `[-10, 5]`** — too permissive at +1-3 Elo target. Use
`[-3, 3]` for correctness fixes. Name the class first, pick the matching
row, submit — don't hedge toward wider bounds out of uncertainty.

**What does NOT need SPRT:**
- Comments, documentation, tooling changes
- Code cleanup that doesn't change compiled output (verify with bench)
- New tunables at default values that don't change behavior (verify bench unchanged)

**OpenBench scripts** (all require `OPENBENCH_PASSWORD` env var, username defaults to `claude`):
```bash
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py <branch> <bench> [--bounds '[0, 3]'] [--priority 1]
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_stop.py <test_id>
OPENBENCH_PASSWORD=<pw> python3 scripts/ob_status.py
```

**Reference NPS** for OB scaling defaults to 250000 (v9 prod). Explicit v5-only
work (legacy bullet_convert experiments, v5 net comparisons) must pass
`--scale-nps 500000`. Wrong scale_nps means wrong time budgets — v9 code at
500K runs ~2× wall-clock, halving fleet throughput.

**Bench measurement.** Always `make && ./coda bench` on the exact branch you're
submitting (see Build and Test §Bench-for-OB ritual). Never reuse bench values
across branches — branch-specific tunables AND the production net both move
the number. Pass the result as `dev_bench` (positional arg to `ob_submit.py`)
and the corresponding main bench as `--base-bench`. The Bench: line in commit
messages is also useful for OB auto-detection but isn't a substitute for
re-measuring before submission.

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

### Cross-Engine and Model Testing

Self-play SPRT is the primary acceptance criterion for search changes.

**Model comparison** (different NNUE nets): Self-play H2H is effective — two nets in the same engine. The eval biases are in the nets, not the search. 200+ games gives a fast signal.

### Known Testing Pitfalls

- **CPU contention**: idle cores before launching SPRT. Background load halves effective TC and distorts marginal results — flipped history pruning from "false H1 +3" to "real H0 −17" in our ablation sweep.
- **Coda-on-Coda contamination**: max 2 Coda variants in a RR; more amplifies shared eval biases.
- **Optimism bias**: don't stop on "looks positive" — let SPRT decide.
- **WDL blindspot**: WDL blending improvements are invisible in self-play; needs cross-engine RR.
- **Self-play discount**: direction usually reliable, magnitude varies.
- **Dig deeper on consensus H0s**: when a feature every top engine uses H0s, your implementation or its dependencies are broken. Don't accept "doesn't work for our engine" — ask why, compare numeric values, look for magnitude/scaling bugs. Capture history was 27× too small (3 prior H0s), fixing it was +31.6 Elo — biggest single gain in Coda's history, found by refusing to give up.

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

For historical examples (SEE quiet pruning +11.4, history magnitude +31.6,
LMR simplify +6.3, cont-hist malus +6.5, NMP capture R +3.5, node-based TM
+11.9, etc.) and the cumulative list of resolved experiments, see
`experiments.md`.

