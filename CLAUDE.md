# CLAUDE.md — Coda Chess Engine

Coda is a UCI chess engine written in Rust. Rewritten from GoChess with all accumulated knowledge.

**Chess Optimised, Developed Agentically** — built through human-AI collaboration.

## Where durable knowledge goes (read this before "saving" anything)

**Put durable, cross-session knowledge in THIS file (CLAUDE.md) or an in-repo
skill (`.claude/skills/`) — NOT in per-machine `~/.claude` memories.** Adam works
with Claude across multiple machines; memories are local to one machine and don't
travel, so they fragment and rot. In-repo docs travel with the checkout and get
human-reviewed. Two more discipline notes that have bitten us:
- **No war-stories / unverified specifics.** Don't assert "+X Elo, biggest gain
  ever" or similar anecdotes — they're frequently wrong and add noise. State the
  rule/mechanism; cite a specific number only if you can verify it right now.
- **Keep OB mechanics in the `ob` skill; local cutechess RR/profiling in the
  `local-rr` skill.** CLAUDE.md holds methodology/policy and points to the skills.

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
make pgo              # PGO build — helps v5 on main; regresses v9 on most x86 platforms. See Makefile.
make net              # Download production NNUE net (from net.txt)
make openbench        # OpenBench-compatible build (alias for `make`)
cargo build --release # Plain release (no embedded net) — NOT what OB workers use
cargo test            # Run all tests including perft

./coda bench [depth]                  # Search benchmark, 48 positions @ default depth 12 — use THIS for OB
./coda                                # UCI mode
./coda --nnue net.nnue                # UCI with explicit NNUE (-n is the short form; -nnue single-dash is INVALID)
./coda --nnue net.nnue --book book.bin  # ... + opening book
./coda epd wac.epd --nnue net.nnue -t 1000    # epd: -t <ms/pos>, -m <max>, -n/--nnue <net>
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
OB will measure, **always** 

* Make sure you have the latest code on the branch
* Run coda bench with the net that you want OB to test with (coda bench -n {NNUEFILE})

For further information on using OB see the OB skill.

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
  movepicker.rs    Staged move ordering, 4D history tables, continuation history
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
- Bad noisy futility (BNFP): prune captures with `static_eval + depth*BAD_NOISY_MARGIN <= alpha && SEE < 0`, gated to `depth <= BAD_NOISY_DEPTH`. BAD_NOISY_MARGIN is a futility scalar (eval-vs-alpha), NOT a SEE threshold.
- IIR: depth>=IIR_MIN_DEPTH (currently effective 2, _10X form), !inCheck, no TT move, PV or cut node
- Singular extensions + double extensions (margin=DEXT_MARGIN, cap=DEXT_CAP)
- Cuckoo cycle detection for proactive repetition avoidance
- Hindsight reduction: reduce when parent was LMR-reduced and both sides quiet
- Fail-high score blending at non-PV nodes
- TT cutoff node-type guard (Alexandria pattern)
- TT cutoff cont-hist malus (penalize opponent's quiet on cutoff)
- Mate distance pruning (non-PV, ply+1 offset)

Do not add tunables to the core set unless you are confident that they are not 'lose knobs' that have little elo impact. They will likely negatively impact SPSA effectiveness.

**Move ordering:** TT move → good captures (MVV×16 + captHist) → quiets (main hist + contHist×3 + pawn hist + quiet check bonus) → bad captures.

**Exemptions:** TT move exempt from pruning. Promotions exempt from LMR.

**History tables:** main [from_threatened][to_threatened][from][to] (4D threat-aware), capture [piece][to][victim], continuation [piece][to][piece][to] (4 plies: 1,2,4,6), pawn [pawnHash%8192][piece][to]. Linear bonus formula: clamp(0, HIST_BONUS_MAX, HIST_BONUS_MULT*depth - HIST_BONUS_OFFSET).

**Correction history:** Multi-source static eval correction. Six source tables (pawn, white-NP, black-NP, minor, major, continuation) with five SPSA-tunable weights (the two NP tables share `CORR_W_NP`). Proportional gravity update.

**Time management:** 3-factor multiplicative model (Obsidian/Clarity). Node fraction (tracks per-root-move nodes), best-move stability (linear), score trend. Validated at LTC (40+0.4); TM is also high-leverage at STC (#1568 TM rework +135 self-play STC) — see §TM-class changes for the methodology.

**Other:** Insufficient material detection. Repetition + cuckoo cycle detection. (Contempt removed 2026-04-19 per SPRT #508 — modern engine practice; was net +2.53 Elo to drop.)

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
- **SIMD**: AVX2 and AVX-512 via `std::arch::x86_64` (runtime detected). Int8 weight quantization for SCReLU forward pass. ARM NEON support also.
- **CReLU**: clamp [0, 255], VPMADDWD dot product
- **SCReLU**: clamp [0, 255], square, int8 byte decomposition for VPMADDUBSW. Scale correction ×0.8 for search threshold compatibility.
- **Pairwise**: split accumulator halves, CReLU-clamp, multiply pairs. SIMD byte decomposition like SCReLU.
- **v7/v9 hidden layers**: SCReLU pack to uint8, int8 L1 matmul via VPMADDUBSW, float L2→output.
- **Fused accumulator update**: copy + delta in single pass for incremental updates
- **TT prefetch**: prefetch TT bucket after make_move, before child node TT probe

### Opening Book
Polyglot .bin format. Weighted random selection. Polyglot Zobrist hashing with standard 781-entry random table. Castling encoded as king-to-rook, converted to king-to-destination. EP hash only when capture is actually possible.

### UCI Options
- `Hash` (spin, 1-4096, default 64) — TT size in MB. For anything above STC (10+0.1) consider increasing this as the default will fill quickly on a modern CPU.
- `Threads` (spin, 1-256, default 1) — Lazy SMP thread count
- `NNUEFile` (string) — path to .nnue network file
- `OwnBook` (check, default true) — use opening book
- `BookFile` (string) — path to Polyglot .bin book
- `MoveOverhead` (spin, 0-5000, default 100) — communication latency in ms
- `Ponder` (check, default false)
- `SyzygyPath` (string) — path to Syzygy tablebase files

### Time Management (Phase 13 model, 2026-05-26 — see `compute_tm_budgets`)
- Viridithas-style windows: max = 60% of clock (absolute single-move
  ceiling), hard = 46%, opt = 73% of (timeLeft/mtg + 94% inc) capped by
  hard. Default mtg 24; no-inc sudden death uses mtg 40 with tighter
  caps (max 15%, hard 10% of clock — flag-fall protection).
- Phase multiplier on opt: 0.36 + 0.64·(1 − e^(−0.045·fullmove))
  (Reckless pattern — spend less in the opening).
- Dynamic factors on opt (the 3-factor model + extras): stability table
  [2.50, 1.20, 0.90, 0.80, 0.75]; subtree/node-fraction multiplier;
  score-trend drop term; aspiration fail-low factor 1 + 0.34·min(2, fl)
  (applies to opt AND hard); forced-move exclusion-search multipliers
  (×0.386 margin 400 @ d8, ×0.627 margin 170 @ d12).
- abs_deadline = clock − MoveOverhead − 50ms: absolute forfeit guard,
  checked first, no grace, no ponder exception. NO emergency mode —
  the no-inc caps + abs_deadline replace the old timeLeft/10 rule.
- Soft floor (stockpile prevention) on no-inc; ponderhit paths arm
  absolute deadlines (H7). Ponder-path constants deliberately excluded
  from `tunables!` (OB has no ponder).
- Living detail + audit findings: `docs/tm_audit_2026-06-13.md`.

## NNUE Training (Bullet GPU)

We train on **Bullet** (Rust, CUDA, fork: `adamtwiss/bullet`) using T80 binpack data. The core set (/workspace/data on training hosts) ~47B positions across 12 files; measured 2026-05). We have about 4x this (full SF trainng set) on GPU4 that we use for production builds. Training produces `quantised.bin` which is converted to `.nnue` via `coda convert-bullet`.

### GPU Host Setup

GPU hosts use the forked Bullet trainer at `~/code/bullet` (clone
`adamtwiss/bullet`) plus a Coda checkout at `~/code/coda` for the converter.
Both `cargo build --release` once after cloning.

### Training Data Locations

**GPU hosts** (cloud): `/workspace/data/`
**Dev hosts** (Hercules, Atlas, Titan): `/training/`

The  T80 binpack data (~47B positions across core 12 files, more on GPU4).

These are LC0 generated positions that are WDL scored (100cp = 50% win chance). The T80 dataset uses 800 MCTS nodes (broadly equivalent to depth 20+ for SF)


### Model Conversion

Convert Bullet output to .nnue format (run on the GPU host after training):
```bash

# Current v9 models -(hidden layers, L1=32, typical modern config is something like)

./coda convert-bullet     -i quantised.bin     -o net.nnue     --pairwise --screlu --hidden 32 --hidden2 32 --int8l1 --threats 66864  --kb-layout reckless --hl-crelu
```

### Training Data Generation

Coda can generate supplementary training data (material-imbalance positions, self-play with blunders). We don't currently train using this data though.

```bash
# Material removal: remove pieces from EPD positions, deep-search each
coda datagen --nnue net.nnue --epd positions.epd --depth 10 --threads 32 --output material.binpack

# Self-play with blunders
coda datagen --nnue net.nnue --games 50000 --depth 8 --threads 32 --blunder 0.1 --output selfplay.binpack
```

Output is SF BINP binpack format, directly usable by Bullet.

### Key Training Findings

- **LR warmup is helpful for hidden layers**: 5-10 SB linear warmup 0.0001→0.001, then cosine 0.001→0.0001.
- **SCReLU scale chain**: keep v² at QA² through matmul, bias×QA² to match, /QA² after.
- **Hidden→output activation is linear** in Bullet (no SCReLU before output buckets).
- **WDL blend (linear / fixed-blend)**: **v5 optimum is w0.07; v9 optimum is w0.20 on modern nets**. **Direction of optimum**: as architecture richness increases (V5 → V9 with threats), optimal WDL weight *increases* — eval-quality improves, so the WDL ground-truth signal can take more weight without polluting eval signal.
- **WDL schedule (alternative to fixed blend)**: Hobbes uses a *ramping* schedule (`100sb constant(0.2), 700sb linear(0.2 → 0.4)` early; `0.6` to `0.75` in late iterations h-33+). Our bullet fork supports this but we've yet to see this really work for us..
- **12-file training data** gives +33 Elo over 6-file for 768pw (data diversity matters).
- **Low final LR is critical**: cosine `final_lr 0.0001` was 20× too high. Reducing to **2.43e-6** (Bullet default `0.001 * 0.3^5`) gave **+47 Elo** — net was oscillating, not converging. For our long multi-stage runs we now prefer 1e-6.
- **Data filtering**: quiet positions only (ply≥16, no checks/captures/tactical moves) gave **+22 untuned, +48 tuned**. Aligns with how NNUE is consumed (quiet nodes after QS).
- **v9 schedule completion is load-bearing, not "more SBs"**: the +88 Elo we used to attribute to "SB400 → SB800 tail" was actually *schedule mismatch*: an `e800` net stopped mid-cosine at SB600 is −88 vs the same `e800` run completed to SB800. A *fully-baked* `e400s400` (own cosine ending at SB400) is only ~5–6 Elo behind `e800s800`. Schedule doubling is roughly +4.7 Elo per doubling. Lesson: complete the schedule you started; don't half-bake. v9 has *not* been tested below the 2.43e-6 final-LR default; floor is ~2.43e-7 (regressed). See `memory/project_v9_low_lr_tail_critical.md`.

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


## NNUE Model Naming Convention

**Production nets (going forward, since 2026-05-31): hash-based.** Name a
promoted prod net `net-v{N}-{OB_HASH}.nnue` — reuse the 8-char OpenBench
content hash you already pass to `--dev-network` (e.g.
`net-v9-E2773E50.nnue`). Rationale: descriptive filenames encode recipe
*inferences* that rot or turn out wrong (the retired `...C8fix-factor.nnue`
prod actually never contained C8fix-2 — the filename lied), and a content
hash can't. Reusing the OB hash also collapses the net's two identities
(filename + OB hash) into one, killing a class of mismatch bugs. Keep only
the `v{N}` arch-generation prefix — v5/v7/v9 coexist and the inference path
is generation-load-bearing. **Recipe / provenance lives in
`docs/net_catalog.md`**, keyed by hash, where it can be corrected without a
rename. This matches SF (`nn-<hash>.nnue`) and most engines.

**Legacy descriptive format** (for decoding pre-2026-05-31 names — do NOT
use for new prod nets; fine for throwaway experiment nets where a
self-describing name is convenient):

`net-v{N}-{accumWidth}[t][d]h{layers}[s]-w{wdl}-e{epochs}s{snap}.nnue`

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
(count grows over time — see the macro for the live list). Current values
reflect multiple SPSA rounds + retune-on-branch calibration. See the macro
in `search.rs` for authoritative defaults and `experiments.md` for which
tunes shaped them.

- SEE values: P=100, N=320, B=330, R=500, Q=900
- History bonus: linear formula min(MAX, MULT*depth - BASE)

## Current Status

Living state (strength, bench, net, SPSA progress, deployment) changes too
quickly to keep accurate in a checked-in file. Authoritative sources:

- **Current production nets:** `docs/net_catalog.md` (v5 + v9 prod hashes,
  retired/active nets, invariants)
- **Recent experiment history / lessons:** `experiments.md`. Always log new experiments (H1 or H0) here for our memory.
- **Current bench:** the `Bench: <nodes>` line of the latest commit on the
  branch you're submitting. Re-measure on the exact branch+net before any
  SPRT; don't carry bench values across branches.
- **Lichess bot:** `codabot` (deployment state and thread count varies)
- **OpenBench fleet:** `ob.atwiss.com`, composition varies

Testing methodology is durable and documented below (Self-play SPRT primary,
retune-on-branch for tree-shape-changing features, LTC for TM features).

## Key Gotchas
- Move flag equality vs bitwise: check non-promotion flags with ==, not &
- EP moves only valid when EP square is empty (occupied square = corruption)
- TT stores raw (uncorrected) eval to avoid double correction on probe
- **is_pseudo_legal must be thorough**: TT hash collisions inject illegal moves. Pawn validation must check direction, intermediate squares (double push), starting rank, destination empty (pushes), enemy piece (captures). Castling must check rights, path clear, king/intermediate/destination not attacked, king on correct square. All three bugs cost 320 Elo combined.
- **PV error warnings = TT collision bugs**: Every "Illegal PV move" from cutechess-cli means a TT collision passed is_pseudo_legal and corrupted the search tree. Treat as critical, not cosmetic.
- **Feature flag ablation**: env var controlled flags (NO_XXX, ENABLE_XXX, DISABLE_ALL) for systematic search feature testing. Parsed once at startup via std::sync::Once.
- PV nodes skip all TT cutoffs and QS beta blending
- Polyglot book encodes castling as king-to-rook (must convert to king-to-destination)

## Code Hygiene

**Keep compiler warnings at zero.** `cargo build --release` should emit none.
Warnings accumulate into noise that masks real issues. Fix or
`#[allow(...)]`-suppress with intent before committing.

## Testing Methodology

### → Before any OB operation: invoke the `ob` skill

For all OpenBench operations (submitting SPRTs, submitting SPSA tunes,
benching for OB, stopping tests, reading results), invoke the **`ob`
skill** at `.claude/skills/ob.md`. It is the canonical reference for
OB usage and supersedes any scattered per-Claude memories on the topic.

The skill covers: bench measurement (including the critical
net-override case), SPRT submission with the bounds policy, SPSA tune
submission, stopping tests, reading results with early-N caveats,
common failure modes. **Recurring bench-mismatch and stop-didn't-stop
issues have been from skipping this skill** — invoking it first is
cheap and prevents the failure modes.

### Self-Play SPRT (primary acceptance test)

SPRT on OB with tight bounds is disciplined, reproducible, and matches the direction of broader cross-engine testing. This is our primary acceptance criterion.

All search/eval changes should normally pass self-play SPRT before merging. This serves the purpose of stress testing, and checking for non-regressions.

**Default bounds: `[0, 3]`** — see the bounds table in §SPRT Testing Policy below for when to deviate.

**STC-first gating.** A good change should work at **all** time controls. STC
(10+0.1) runs ~4× faster than LTC, so **fire STC first as a cheap gating/initial
signal** before committing fleet to LTC. And for any change aimed at a specific
TC (e.g. an LTC-targeted pruning/reduction tweak), **also validate it doesn't
disproportionately hurt STC before merging** — if it does, a future STC tune
will just move it back. Don't merge a change that only holds at one TC.

For some time management (TM) changes (e.g. involving pondering that OB can't do) then OB is not effective. It's better in these cases to do local cutechess RRs. TM changes often need the behaviours of other engines to provoke our behaviour.

**Methodology for TM-class changes:**
1. **Inspect mechanism first**: 5-10 local games at the target TC, parse
   per-move clocks from PGN, verify the change actually fires as designed.
   See `scripts/tm_pattern_inspect.py`. Catches "governor never fires" /
   "wrong TC for mechanism" bugs cheaply before burning fleet/CPU.
2. **Primary signal: cross-engine RR with ponder-enabled opponents.** Use
   similar-strength engines with `ponder` flag on the engine line. TC at deployment-matched ratios (30+0.5 for 60:1 ratio, etc.). Target ≥200 games per
   engine for ±20 Elo CI; default 30-50 rounds × 2 games × 21+ pairs.
3. **Cross-check: SPRT before merging** at `[0, 3]` LTC. Required for
   non-regression confirmation. Accept any verdict short of clear regression
   — do NOT gate merge on SPRT magnitude for TM changes (it WILL undersell
   ponder-asymmetric gains).

Some, but not all, TM changes do show up in SPRT. We have seen tests that have given us over 100 Elo from TM measurable from STC SPRT.

**Common-trap regex bug (2026-05-25)**: when parsing per-move spend from
cutechess PGN comments `{+0.43/17 0.60s}`, the non-greedy regex
`[^}]*?([0-9]+\.[0-9]+)s?[^}]*?\}` extracts the FIRST decimal (the score,
`0.43`), NOT the spend (`0.60`). Use `([0-9]+\.[0-9]+)s\b` instead. Caught
this after multiple wrong "mechanism not firing" conclusions on Phase
10f/10g/10h analyses; tooling is now correct in `tm_pattern_inspect.py`
and `tm_variation_analyzer.py`.

**Concurrency for local RR**
- It's generally best to have one engine running per CPU thread. Without ponder only one engine runs at a time, so on an 8C/16T CPU you can run cutechess with concurrency 16.

**Key rules:**
- One change per branch. Never stack untested changes.
- Wait for H0 or H1. Do not stop early based on "looks good".
- H0 = reject. H1 = accept. Always log result to experiments.md.
- For tree-shape-changing features consider retune-on-branch before deciding (see methodology below).
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

**Standing policy: `[0, 3]` is the default for ALL "does this feature help?"
SPRTs.** Reckless uses the same default. Avoid temptation to use wider bounds in almost any circumstances

| Bounds | When to use | Example |
|--------|-------------|---------|
| **`[0, 3]` (DEFAULT)** | "Does this feature help enough to be worth making a change?" at Coda's current strength. Most new ideas target +1-3 Elo. **Pick this unless you have a specific reason for one of the rows below.** | Pruning/ordering tweak, parameter probe, small bonus adjustment, incremental feature, audit correctness fix, tune-applied retest, structural ports |
| `[-2, 1]` | "Ship if not a meaningful regression." Bench-neutral refactors, NPS-only changes, ARM ordering, adding tunables at default values. Forces enough games to actually discriminate near zero. | Code cleanup with possible perf delta, OnceLock migration, defensive guard whose direction is uncertain |
| `[-1.5, 1.5]` | Where the cost of a change is zero, and you are comparing two neutral things (net-vs-net, alternative-net compare). | New candidate net vs prod/baseline, SE margin tweak, 50mr mate downgrade, stale-bound gate |

**Do-NOT-use (all range > 3): `[-3, 3]`, `[-5, 5]`, `[0, 5]`, `[0, 10]`.**
They lock H1 on noise and/or return without separating. `[-3, 3]`
specifically (the trap I keep falling into): use **`[-1.5, 1.5]`** instead
for uncertain-direction, or `[-2, 1]` for non-regression. Adam pushed back
on `[-5, 5]` 2026-05-26 and on `[-3, 3]` 2026-06-13.

**What does NOT need SPRT:**
- Comments, documentation, tooling changes
- Code cleanup that doesn't change compiled output (verify with bench)
- New tunables at default values that don't change behavior (verify bench unchanged)

### Mini-prod Branch for S200 Experiments

S200-targeted experiments (anything trained to ~SB200: training-recipe
probes, architecture probes, factor/threat variants) **must fork from
the `mini-prod` branch, NOT main**, and SPRT with `--base-branch mini-prod`.

**Why.** Main's tunables are SPSA-calibrated for the prod SB800 net.
Using main as the SPRT base for an S200 experiment gives the DEV side
a ~+4-6 Elo tune-flation handicap on BASE — BASE runs mistuned trunk
values for its S200 net. Empirically measured 2026-05-11 via SPRT
#1117 (mini-prod tuned-vs-untuned, same net both sides): the latent
tuning headroom on an S200 net was ~+4 Elo. Main does NOT have this
problem because main's tunables are always calibrated for main's
current net.

**Workflow:**

```bash
git checkout mini-prod
git checkout -b experiment/s200-<description>
# ... make S200 experimental changes ...
make && ./coda bench

OPENBENCH_PASSWORD=<pw> python3 scripts/ob_submit.py experiment/s200-<...> \
    --base-branch mini-prod --base-bench <mini-prod-bench> \
    --dev-network <CANDIDATE_SHA> --base-network <baby-prod-SHA>
```

The branch name is always `mini-prod` (net-agnostic); the current
baby-prod reference net is in net.txt and a comment at the top of the
`tunables!` macro on that branch.

**Asymmetric merge cadence** (load-bearing — read this):
- Search/eval H1 wins merge to `main` ASAP via the normal SPRT flow.
- The SAME wins propagate to `mini-prod` **only during scheduled
  refresh windows when no S200 experiments are in flight**. A
  mid-flight base shift would invalidate in-progress mini-prod SPRTs.

This means mini-prod intentionally lags main between refreshes — by
design, not bug. Don't try to "keep mini-prod fresh" on every main
merge.

**Refresh = rebase + retune** (NOT recreate from scratch).
Trigger when ANY of:
1. Main changed the `tunables!` macro structure (new/renamed tunable,
   widened range, new feature).
2. Main landed a search-shape change that may interact with tunings.
3. Main accumulated ~5+ Elo of merged changes since the last refresh.
4. **AND** no S200 experiments are currently in flight.

Procedure: rebase mini-prod onto main, on conflicts take main's
STRUCTURAL changes but keep mini-prod's tuned VALUES (S200-calibrated);
fire focused ~1500-iter SPSA against the baby-prod net; apply outputs;
SPRT-validate at `[-1.5, 1.5]` vs pre-refresh mini-prod; push.

**Net rotation** (when training methodology produces a new baby-prod
S200 net, not just a more-baked version): archive the current
mini-prod, branch fresh from main, fire a ~2500-iter tune from main
defaults. Force-push mini-prod — rotation is the one time history
restarts. See `docs/mini_prod_branch_workflow.md` for the full
procedure.

See `docs/mini_prod_branch_workflow.md` for runnable detail and the
methodology behind these policies.

### SPSA Parameter Tuning

Search parameters are exposed as UCI options for SPSA optimization via OpenBench.
The `tunables!` macro in `search.rs` is the single source of truth for defaults,
ranges, and c_end values.

**TC convention for tunes (Adam, 2026-06-12):**

- **Production tunes (anything whose outputs will be applied to trunk
  defaults): LTC, `--tc 40.0+0.4 --options 'Threads=1 Hash=256'`.**
  Hash=256 is load-bearing — Hash=64 at LTC is a TT-starved regime that
  distorts pruning economics (tune #1900 vs #1915 decomposition,
  2026-06-11; the starved LTC RR cost ~25 Elo vs healthy hash). Slower
  per iteration; warm-start from the previous production tune's outputs
  and use ~2500 iters instead of 5000 (the #1915 pattern) to keep cost
  sane. Avoid Hash=512: 8GB workers OOM at 6 concurrent games (worker
  391, #1911 post-mortem).
- **Quick retune-on-branch validation tunes (focused clusters proving a
  feature direction): STC is fine.** The LTC-healthy production values
  measure ~neutral at STC (#1926), so STC branch-tunes don't fight the
  production calibration much. Anything that graduates to trunk
  application should get its final values from (or be validated at) the
  LTC regime.
- Rationale: every deployment regime (lichess on bare metal, CCRL) is
  the deep regime; OB STC is a measurement frame, not a deployment
  target. STC-optimal and LTC-optimal pruning shapes can genuinely differ.


**Post-tune / net-deploy discipline:**

- When a net changes (new prod in net.txt), plan an immediate
  trunk retune against the new net before landing large clusters
  of eval-dependent experiments on top.
- Don't ship "new net with old trunk" as a hidden detune.

**Submitting tunes:**

See the OB skill.

# Static "all params" cache files (e.g. scripts/tune_all_main.txt) are
# FORBIDDEN — they drift from src/search.rs after every applied tune,
# silently restarting SPSA from stale defaults. ob_tune.py auto-derives
# the spec from the binary when no --params/--params-file is given.

# Bench is auto-detected from commit message. Pass explicitly only if OB can't parse it.

**SPSA format per parameter:** `NAME, int, default, min, max, c_end, r_end`

When LMR_C_QUIET or LMR_C_CAP change, LMR tables are automatically reinitialized.

**Practical guidance:**
- **SPSA iteration counts.** SPSA uses **2 objective evals per iteration
  regardless of param count** — iters do NOT scale linearly with params (the old
  √N "150-200 iter/param, 12-16K full-sweep" rule here was wrong and is removed).
  Total iters-to-converge scales with **dimensionality + noise**, not param count
  directly: moderate dim (≤~50-60 params) **500-2000 iters**; high dim (100+
  params) 5000-10000+ (ref: pennylane.ai/demos/tutorial_spsa). **Our tunes sit in
  the moderate band (~60 params), so Adam's working convention is: ~2500 full
  production tune; 1000-1500 retune-on-branch over all params (incl. the
  ~74-param `--core` set); 800-1000 limited set.** Longer tunes have been tried
  repeatedly **at our param count** and not consistently helped — so don't propose
  10K+ runs or flag a 1500-iter all-params tune as undersized *at our scale*.
- **Watch param-creep / "loose knobs."** The `--core` count has crept up (now
  ~74). Past ~60-70, extra params add dimensionality + noise (each loose,
  non-load-bearing knob the tuner wanders) and push toward the high-dim regime.
  The fix is **audit + prune loose knobs** (SPSA-detuning signal: params it
  pushes hard off-default or wanders without converging), NOT just more iters.
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
2. **Submit SPSA tune** on the branch (cluster of pruning params relevant to the change, or full-sweep if scope is unclear)
3. **Compare parameter convergence** against a baseline tune on main
4. **If parameters diverge significantly** (>5% on multiple params): the feature is shifting the search landscape. Apply tuned values and SPRT the branch+tune against main.
5. **If parameters converge to same values as main**: the feature is truly neutral, drop it.

**Validated examples:**
- TT PV flag: +4.5 raw → retune added +4.0 more (nearly doubled)
- Cont-hist malus: flat (-0.15 at 16K games) → +6.5 with retune
- Pattern: big bench/node change but flat Elo → retune candidate


### Feature Improvement Cycle (Detect → Diagnose → Fix → Tune)

Systematic approach for finding and fixing search feature issues. Each cycle compounds — fixing one feature shifts the optimal parameters for everything else, revealing the next weak feature.

**1. Detect weak features**
- **SPSA detuning**: If SPSA is aggressively moving a parameter away from its starting value (>30% shift), the feature may have a structural flaw. The tuner compensates for bugs by detuning. Example: SEE_QUIET_MULT driven from 17 to 6 (nearly disabled) because the implementation was broken.
- **Ablation anomaly**: If disabling a feature gains Elo, or gains less than expected, investigate.
- **Cross-engine parameter divergence**: If our value for a parameter is far outside the consensus range, understand why before assuming we're special.

**2. Diagnose via cross-engine comparison**
- Compare the specific feature implementation against 6-8 top engines with source code available. Engine sources are in `/home/adam/chess/engines/`.
- **Only survey engines STRONGER than Coda** for "what should we do" consensus.
  Weaker engines' choices are not evidence (they may be weak *because* of them).
  Coda currently ranks #7 in our local top-20 RR — **engines #1-6 are the
  valid reference set** (Stockfish, Reckless, Berserk, Obsidian,
  PlentyChess, Alexandria). A "14-of-16 engines do X" consensus that
  leans on mid-table engines no longer carries weight; weigh agreement
  among the six above us, and treat near-peers (Integral, Viridithas)
  as corroboration only.

**Current local RR (10+0.1, ~218 games/engine, 2026-06-12, top-20 pool —
includes every CCRL-top engine in C/C++/Rust)** — Elo relative to the
pool, NOT absolute. Coda = #7 (+16), up from #14 a few days earlier
(+25-30 banked on main that week; rivals-RR stretches internal Elo ~2×):

| # | Engine | Elo | | # | Engine | Elo |
|---|--------|-----|-|---|--------|-----|
| 1 | Stockfish | +148 | | 11 | Hobbes | −13 |
| 2 | Reckless | +107 | | 12 | Astra | −21 |
| 3 | Berserk | +66 | | 13 | Clover | −27 |
| 4 | Obsidian | +66 | | 14 | Stormphrax | −32 |
| 5 | PlentyChess | +48 | | 15 | Halogen | −32 |
| 6 | Alexandria | +37 | | 16 | Quanticade | −40 |
| **7** | **Coda** | **+16** | | 17 | Starzix | −58 |
| 8 | Integral | −2 | | 18 | Tarnished | −61 |
| 9 | Viridithas | −2 | | 19 | Motor | −63 |
| 10 | Caissa | −11 | | 20 | Clarity | −119 |

Engines below Coda (Integral, Viridithas, Caissa, Hobbes, Clover,
Halogen, Starzix, Quanticade, Tarnished, Astra, Stormphrax, and the
rest of the historical pool: Horsie, Koivisto, Seer, Motor, Clarity,
Velvet, ... down to Crafty/Monolith at −500+) are NOT references for
"should we" consensus. **Monolith/Crafty/Greko/Rodent are 500-600+
Elo behind us — do NOT cite them as references.**

Reference engines for cross-engine review (strongest first, the #1-6
set): Stockfish, Reckless (Rust), Berserk, Obsidian, PlentyChess,
Alexandria.
- For each engine: exact formula, gating conditions, position in move loop (before/after MakeMove), depth variable used (raw depth vs lmrDepth), history adjustments, numeric values.
- Common structural issues found so far:
  - **Pre-move vs post-move**: Pruning after MakeMove wastes make/unmake + NNUE push/pop per pruned move, and makes the feature redundant with earlier pruning (futility, LMP catch most candidates first).
  - **Raw depth vs lmrDepth**: Using raw depth for depth² scaling is far more aggressive than intended. Engines using depth² all use lmrDepth. Engines using raw depth compensate with linear scaling.
  - **Duplicate SEE functions**: Using a separate SEE implementation instead of the standard `see_ge` risks correctness bugs.

**3. Fix and SPRT test**
- Create a feature branch with the structural fix.
- Set parameter defaults to match consensus (e.g., Stockfish's value for the same formula).
- SPRT test against main with the standard bounds policy (default `[0, 3]`; widen only if the change class has a load-bearing prior for larger magnitude — see §SPRT Testing Policy bounds table). The fix should be positive even with untuned constants if the structural change is correct.
- If SPRT fails, review for secondary bugs (missing ply>0 guard, bestScore pollution, etc.).

**4. SPSA tune the corrected feature**
- **Focused tune first** (2-4 params, ~1000 iterations): Just the new/changed parameters. Fast convergence, finds the right ballpark.
- **Full tune after** (all tunable parameters, 2500+ iterations): Rebalances everything — other params were compensating for the broken feature and need to readjust.
- Merge the focused tune values, then run the full tune as the next round.

**5. Repeat**
The retuned baseline exposes the next weak feature. Check SPSA trends for the next parameter being aggressively detuned.

For historical examples and the cumulative list of resolved experiments, see
`experiments.md`.
