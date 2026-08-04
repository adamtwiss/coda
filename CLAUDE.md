# CLAUDE.md — Coda Chess Engine

Coda is a UCI chess engine in Rust, rewritten from GoChess.
**Chess Optimised, Developed Agentically** — built through human-AI collaboration.

> Coda's development methodology (testing, SPRT/SPSA, OpenBench), NNUE training,
> cross-engine research, tooling, and the Claude skills live in a **separate
> private research repository**. This public repo is the engine: source, build,
> tests, and the licensing docs (`docs/license_analysis_2026-07-13.md`).
>
> **Never add scripts or tuning artefacts to this repo.** There is no `scripts/`
> directory and there should not be one — it is `.gitignore`d, along with
> `tune_*.txt`. Research tooling, launchers, analysis scripts and SPSA specs all
> belong in the private repo. If a tool seems to need to live here, it is almost
> certainly a `coda` subcommand instead (`bench`, `perft`, `epd`, `tune-spec`,
> `fetch-net`), which is how this project ships tooling. **SPSA specs especially
> must never be committed**: they drift from the `tunables!` macro as soon as a
> tune is applied, and `coda tune-spec` emits the live values on demand, so a
> checked-in copy is guaranteed to be the stale one.

## Supported CPU families

- **x86_64 (primary):** default target.
- **aarch64 (first-class, since 2026-04-25):** Apple M-series, ARM servers. New
  SMP code must use correct memory ordering — `Acquire/Release` on shared atomics
  with reader-publish patterns, not `Relaxed` (x86's strong model masks ordering
  bugs that fire on ARM). Default to `Acquire/Release` + explicit NEON tests.

## Build and Test

Prerequisites: Rust 1.70+. PGO builds also need `cargo install cargo-pgo` +
`rustup component add llvm-tools-preview`.

```bash
make                  # Build with embedded NNUE net + native CPU (produces ./coda)
make net              # Download production NNUE net (from net.txt)
cargo build --release # Plain release (no embedded net)
cargo test            # Run all tests including perft

./coda bench [depth]                  # Search benchmark, 48 positions @ default depth 12
./coda                                # UCI mode
./coda --nnue net.nnue                # UCI with explicit NNUE (-n is the short form; -nnue single-dash is INVALID)
./coda --nnue net.nnue --book book.bin  # ... + opening book
./coda epd wac.epd --nnue net.nnue -t 1000    # epd: -t <ms/pos>, -m <max>, -n/--nnue <net>
./coda perft [depth] [fen...]
./coda perft-bench                    # 6-position perft suite
./coda convert-bullet [options]       # quantised.bin → .nnue
./coda fetch-net                      # Pull net from net.txt URL
./coda help
```

## Project Structure

```
src/
  main.rs          Entry point, CLI argument parsing, subcommands
  board.rs         Board struct (bitboards + mailbox), FEN, make/unmake, Zobrist
  types.rs         Color, Piece, Square, Move encoding (16-bit), castling
  bitboard.rs      Bitboard ops, between/line tables
  attacks.rs       Magic bitboards (PEXT runtime detected), knight/king/pawn tables
  setwise.rs       Setwise (batched) attack generation — all pieces of one type at once
  movegen.rs       Pseudo-legal + capture-only move generation, perft
  zobrist.rs       Zobrist hash keys (deterministic PRNG)
  zobrist_keys.rs  Auto-generated Zobrist key constants
  eval.rs          PeSTO material+PST eval (fallback), SEE values, NNUE eval wrapper
  see.rs           Static Exchange Evaluation
  tt.rs            Transposition table (5-slot buckets, XOR key verification)
  movepicker.rs    Staged move ordering, 4D history tables, continuation history
  search.rs        Negamax, pruning, LMR, correction history, cuckoo, pruning stats
  thread_pool.rs   Persistent Lazy-SMP helper thread pool (reused across go commands)
  cuckoo.rs        Cuckoo cycle detection for proactive repetition avoidance
  tb.rs            Syzygy tablebase probing (via shakmaty-syzygy)
  tb_cache.rs      Lockless Zobrist-keyed WDL probe cache (UCI TBHash)
  nnue.rs          NNUE v5/v7/v9 inference, accumulator stack, Finny table, AVX2/AVX-512/VNNI SIMD
  nnue_simd.rs     NNUE SIMD primitive abstractions (cfg(target_feature)-gated)
  sparse_l1.rs     Sparse/dense int8 L1 matmul kernels (AVX2, AVX-VNNI, AVX-512 VNNI)
  threats.rs       Threat-feature enumeration + delta generation (v9)
  threat_accum.rs  Per-ply threat accumulator stack (v9)
  uci.rs           UCI protocol (position, go, stop, ponder, setoption)
  epd.rs           EPD test suite runner with SAN formatting
  book.rs          Polyglot opening book support
  polyglot_randoms.rs  Standard Polyglot Zobrist random table (781 entries)
  datagen.rs       Multi-threaded training data gen; writes SF BINP binpack via the sfbinpack crate
  bullet_convert.rs  Bullet quantised.bin → .nnue converter (v5/v7/v9)
  nnue_export.rs   .nnue → Bullet checkpoint converter (for transfer learning)
Makefile           Build targets: make, make pgo, make openbench, make net
net.txt            Production NNUE net URL (used by make net / fetch-net)
```

## Architecture

### Board
Bitboards (`pieces[6]` by type + `colors[2]`) + mailbox (`[u8;64]` for O(1)
piece-at-square). Magic bitboards for sliders (PEXT on BMI2, runtime-detected).
Incremental Zobrist + pawn hash.

### Move encoding
16 bits: from(6) + to(6) + flags(4). Flags: None=0, EP=1, Castle=2,
PromoteN=4..PromoteQ=7. Double-push has no flag (detected by distance in make_move).
**Check non-promotion flags with `==`, not bitwise `&`.**

### Search
Negamax + alpha-beta, iterative deepening, PVS, aspiration windows (from depth 4).
Lazy SMP: helper threads search at offset depths sharing the TT (atomic) + stop flag.

**Pruning / extension features** (all SPSA-tunable via the `tunables!` macro in
`search.rs` — that macro is the authoritative feature list, defaults, and ranges):
NMP, RFP, futility (history adjusts effective lmr_depth), LMR (separate
quiet/capture tables, doDeeper/doShallower, tt_pv reduces less), LMP, SEE pruning
(quiet d², capture linear), ProbCut, bad-noisy futility (BNFP — futility scalar +
SEE<0 gate, not a SEE threshold), IIR, singular + double extensions, hindsight
reduction, cuckoo repetition detection, fail-high blending at non-PV, TT-cutoff
node-type guard + cont-hist malus, mate-distance pruning.

**Move ordering:** TT move → good captures (MVV + captHist) → quiets (main +
cont-hist×3 + pawn hist + quiet-check bonus) → bad captures. SEE uses modern
consensus piece values (minors/rook/queen higher than the old textbook set) — see
`eval::see_value`.

**History tables:** main `[from_threatened][to_threatened][from][to]` (4D
threat-aware); capture `[piece][to][victim]`; continuation `[piece][to][piece][to]`
(plies 1,2,4,6); pawn `[pawnHash&511][piece][to]`. Linear bonus:
`clamp(0, MAX, MULT·depth − OFFSET)`.

**Correction history:** multi-source static-eval correction with proportional
gravity update. Five sources — pawn, white-NP, black-NP, continuation, transition
(zobrist-delta); weights are `CORR_W_*` tunables.

**Time management:** 3-factor multiplicative model (per-root-move node fraction,
best-move stability, score trend) over opt/hard/max windows with an opening-phase
damp and an absolute forfeit deadline. Full model in `compute_tm_budgets`.
Ponder-path constants are deliberately excluded from `tunables!`.

**Other:** insufficient-material detection; repetition + cuckoo cycle detection.
Contempt removed (modern practice).

### TT
- 5-slot buckets, 64 bytes (cache-line aligned), Atomic{U64,U32} for lockless SMP.
- XOR key verification (`key ^ data`) detects torn reads from concurrent writes.
- Packs a 13-bit static eval (±4095cp) + 1-bit tt_pv (sticky PV marker for LMR).
- Replacement: overwrite a same-gen key match unless the stored entry is much
  deeper; always replace on generation change or an exact entry.
- Non-PV cutoff score dampening; 1-ply near-miss acceptance (bounded margin);
  fail-high blending at non-PV depth≥3; QS probe with cutoffs.
- Stores **raw** (uncorrected) static eval to avoid double-correction on probe.

### NNUE
Production arch is **v9** (FT + threats → hidden layers). Inference also supports
legacy v5 (direct FT→output) and v7 (FT→hidden→output) for retired nets. The
production net is referenced by `net.txt`.
- HalfKA: 16 king buckets × 12 piece types × 64 squares = 12288 inputs.
  Quantization QA=255 (accumulator), QB=64 (output weights).
- **Lazy accumulator** (materialize on demand — saves work for pruned nodes),
  **Finny table** (per-perspective per-bucket cache, diffs on king-bucket change),
  **fused accumulator update**, **TT prefetch** after make_move.
- **SIMD** (runtime-detected): AVX2 / AVX-512 / VNNI on x86, NEON on ARM.
  CReLU / SCReLU (int8 byte-decomposition for VPMADDUBSW) / pairwise activations;
  int8 L1 matmul; float L2→output for v7/v9.

### Opening book
Polyglot `.bin`, weighted random selection. Standard 781-entry Zobrist table.
Castling encoded king-to-rook (convert to king-to-destination). EP hash only when a
capture is actually possible.

### UCI options
`Hash` (MB, default 64), `Threads` (default 1), `NNUEFile`, `OwnBook` (default
true), `BookFile`, `MoveOverhead` (ms, default 100), `Ponder`, `SyzygyPath`,
`TBHash` (WDL-cache MB, default 16), `SyzygyProbeDepth` (default 4). Debug/internal
only: `HiddenActivation`, `LoadAnyway`, `TMDebug`, `PonderhitCreditPct`. All
`tunables!` params are also exposed as spin options for SPSA (not for manual use).

### EVAL_SCALE
`EVAL_SCALE` (nnue.rs) converts raw net output to centipawns. When a net's natural
scale changes, all search thresholds (RFP/futility/SEE/LMR) miscalibrate — the
preferred fix is an SPSA retune, not an EVAL_SCALE hack. Measure a candidate's
scale empirically; pairwise nets don't scale linearly (int8 overflow), so never
compute it analytically.

## NNUE net naming
**Prod nets (since 2026-05-31): hash-based** — `net-{HASH}.nnue`, reusing the
8-char content hash (e.g. `net-DB9B5605.nnue`). Descriptive filenames encode
recipe inferences that rot; a content hash can't. **No architecture-version
prefix** — the arch generation is already in the `.nnue` header and the loader
dispatches on it, so a filename prefix would be a second source of truth that
can disagree with the file.

## Key gotchas
- Non-promotion move flags: compare with `==`, not `&`.
- EP move valid only when the EP square is empty (occupied square = corruption).
- TT stores raw (uncorrected) eval — avoids double-correction on probe.
- **`is_pseudo_legal` must be thorough** — TT hash collisions inject illegal moves;
  incomplete pawn/castling validation (direction, intermediate squares, start rank,
  dest empty/enemy; castle rights + path + attacked squares) has cost hundreds of
  Elo. Any **"Illegal PV move"** warning is a **critical TT-collision bug**, not
  cosmetic — it makes the lichess bot resign.
- PV nodes skip all TT cutoffs and QS beta blending.
- Feature-flag ablation via env vars (NO_XXX / ENABLE_XXX / DISABLE_ALL), parsed
  once at startup for systematic search-feature testing.

## Code hygiene
Keep `cargo build --release` at **zero warnings** — fix or `#[allow(...)]`-suppress
with intent before committing.

## Commit messages
Every commit that changes search/eval must include `Bench: <nodes>` (from
`./coda bench` with the production net), so the built binary can be verified:
```
Fix razoring margin at depth 2

Bench: 1780721
```
