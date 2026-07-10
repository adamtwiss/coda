# CLAUDE.md — Coda Chess Engine

Coda is a UCI chess engine in Rust, rewritten from GoChess.
**Chess Optimised, Developed Agentically** — built through human-AI collaboration.

## Where durable knowledge goes (read before "saving" anything)

Put durable, cross-session knowledge in **THIS file or an in-repo skill**
(`.claude/skills/`) — NOT in per-machine `~/.claude` memories, which don't travel
between Adam's machines and rot. In-repo docs travel with the checkout and get
human-reviewed.
- **No war-stories, unverified specifics, or tunable values.** State the
  rule/mechanism; don't paste "+X Elo" anecdotes or current parameter values —
  they rot, and source is authoritative. Cite a number only if you can verify it now.
- **OB mechanics live in the `ob` skill; local cutechess RR/profiling in
  `local-rr`.** CLAUDE.md holds methodology/policy and points to the skills.

## Supported CPU families

- **x86_64 (primary):** OpenBench fleet, Lichess, CCRL, all SPRT gating. Default target.
- **aarch64 (first-class, since 2026-04-25):** Apple M-series, ARM servers. New
  SMP code must use correct memory ordering — `Acquire/Release` on shared atomics
  with reader-publish patterns, not `Relaxed` (x86's strong model masks ordering
  bugs that fire on ARM). Default to `Acquire/Release` + explicit NEON tests. See
  `docs/arm_correctness_2026-04-25.md`.

## Build and Test

Prerequisites: Rust 1.70+. PGO builds also need `cargo install cargo-pgo` +
`rustup component add llvm-tools-preview`.

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

**Bench-for-OB ritual.** OB workers build with `make` (emits `./coda` at the repo
root). For a bench that matches what OB measures: pull the latest code on the
branch, then run `./coda bench -n <net>` with the net OB will test. Full detail in
the `ob` skill.

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
  threats_splat.rs AVX-512 byteboard-splat threat-delta enumeration (v9 threat pipeline)
  uci.rs           UCI protocol (position, go, stop, ponder, setoption)
  epd.rs           EPD test suite runner with SAN formatting
  book.rs          Polyglot opening book support
  polyglot_randoms.rs  Standard Polyglot Zobrist random table (781 entries)
  datagen.rs       Multi-threaded training data gen; writes SF BINP binpack via the sfbinpack crate
  bullet_convert.rs  Bullet quantised.bin → .nnue converter (v5/v7/v9)
  nnue_export.rs   .nnue → Bullet checkpoint converter (for transfer learning)
Makefile           Build targets: make, make pgo, make openbench, make net
scripts/           OB tooling: ob_submit.py, ob_tune.py, ob_status.py, ob_stop.py, ob_tune_status.py, ...
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
NMP, RFP, futility (history adjusts effective lmr_depth, SF pattern), LMR (separate
quiet/capture tables, doDeeper/doShallower, tt_pv reduces less), LMP, SEE pruning
(quiet d², capture linear), ProbCut, bad-noisy futility (BNFP — futility scalar +
SEE<0 gate, not a SEE threshold), IIR, singular + double extensions, hindsight
reduction, cuckoo repetition detection, fail-high blending at non-PV, TT-cutoff
node-type guard (Alexandria) + cont-hist malus, mate-distance pruning.

Don't add tunables to the core set unless confident they carry Elo — loose knobs
degrade SPSA effectiveness.

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
(zobrist-delta); weights are `CORR_W_*` tunables. (Minor/major tables ablated and
dropped 2026-05-19.)

**Time management:** 3-factor multiplicative model (per-root-move node fraction,
best-move stability, score trend) over Viridithas-style opt/hard/max windows with an
opening-phase damp and an absolute forfeit deadline. Full model in
`compute_tm_budgets`; audit findings in `docs/tm_audit_2026-06-13.md`. Ponder-path
constants are deliberately excluded from `tunables!` (OB has no ponder). TM is
high-leverage at both STC and LTC.

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
legacy v5 (direct FT→output) and v7 (FT→hidden→output) for retired nets. Current
prod hash + active nets: `docs/net_catalog.md`.
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
`Hash` (MB, default 64 — raise above STC), `Threads` (default 1), `NNUEFile`,
`OwnBook` (default true), `BookFile`, `MoveOverhead` (ms, default 100), `Ponder`,
`SyzygyPath`, `TBHash` (WDL-cache MB, default 16), `SyzygyProbeDepth` (default 4).
Debug/internal only: `HiddenActivation`, `LoadAnyway`, `TMDebug`,
`PonderhitCreditPct`. All `tunables!` params are also exposed as spin options for
SPSA (not for manual use).

## NNUE Training (Bullet GPU)

Train on **Bullet** (fork `adamtwiss/bullet`, Rust/CUDA) using T80 LC0 binpack data
(WDL-scored, 100cp≈50% win; 800 MCTS nodes ≈ SF depth 20+). Core set ~47B positions
across 12 files on training hosts; GPU4 holds the full SF set used for prod builds.
Output `quantised.bin` → `.nnue` via `coda convert-bullet`.

- **GPU host setup:** forked trainer at `~/code/bullet` + a Coda checkout at
  `~/code/coda` (for the converter); `cargo build --release` both once.
- **Data locations:** `/workspace/data/` (cloud GPU hosts), `/training/` (dev hosts).
- **Convert** (flags must exactly match the training config):
  ```
  ./coda convert-bullet -i quantised.bin -o net.nnue --pairwise --screlu \
      --hidden 32 --hidden2 32 --int8l1 --threats 66864 --kb-layout reckless --hl-crelu
  ```
- **Datagen** (supplementary material-imbalance / self-play; not currently used in
  prod training): `coda datagen …` writes SF BINP binpack, directly usable by Bullet.

**Durable training rules:**
- LR warmup helps hidden layers (short linear ramp → cosine to a low final LR).
- SCReLU scale chain: keep v² at QA² through the matmul, bias×QA² to match, /QA²
  after. Hidden→output is linear in Bullet (no SCReLU before output buckets).
- WDL blend rises with architecture richness (v5 ~0.07, v9 ~0.20) — richer eval
  tolerates more WDL ground-truth without polluting the eval signal.
- Low final-LR is critical (oscillation vs convergence). Prefer ~1e-6 for long
  multi-stage runs; don't go below the ~2.4e-6 default without SWA.
- Quiet-position filtering (ply≥16, no checks/captures/tactical moves) helps —
  matches how NNUE is consumed at quiet nodes after QS.
- **Complete the schedule you start.** A net stopped mid-cosine is much weaker than
  one whose cosine ends at the snapshot; schedule doubling is a small steady gain.
  Don't half-bake.
- 12-file data beats 6-file (diversity). Quantified versions live in memory/docs.

### EVAL_SCALE
`EVAL_SCALE` (nnue.rs) converts raw net output to centipawns. When a net's natural
scale changes, all search thresholds (RFP/futility/SEE/LMR) miscalibrate —
**preferred fix is an SPSA retune**, not an EVAL_SCALE hack. Measure a candidate's
scale empirically (RMS over ~500 positions via `coda eval-dist`/`eval-fens`);
pairwise nets don't scale linearly (int8 overflow), so never compute it analytically.

## NNUE net naming

**Prod nets (since 2026-05-31): hash-based** — `net-v{N}-{OB_HASH}.nnue`, reusing
the 8-char OpenBench content hash (e.g. `net-v9-E2773E50.nnue`). Descriptive
filenames encode recipe inferences that rot; a content hash can't, and reusing the
OB hash collapses the net's filename + OB identities into one (killing a class of
mismatch bugs). Keep only the `v{N}` arch-generation prefix (v5/v7/v9 coexist and
the inference path is generation-load-bearing). Recipe/provenance lives in
`docs/net_catalog.md`, keyed by hash. Legacy descriptive names (pre-2026-05-31):
decode via `docs/net_catalog.md`.

## Current status (living state lives elsewhere)

Strength, bench, prod net, SPSA progress, and deployment change too fast for a
checked-in file:
- **Prod nets:** `docs/net_catalog.md` (hashes, active/retired, invariants).
- **Experiment history / lessons:** `experiments.md` — log every H0/H1.
- **Current bench:** the `Bench:` line of the branch you're submitting; re-measure
  per branch+net, never carry across branches.
- **Lichess bot:** `codabot`. **OB fleet:** `ob.atwiss.com`.

### Versioned releases (policy 2026-07-05)

**When to release (Adam decides the moment):** a new prod net promoted AND soaked
~a week on lichess AND trunk quiet (no half-landed tune/merge trains). A code-only
patch with no net change skips the net-soak gate. No backfilling; no off-cadence releases.

**Versioning** (single source of truth = Cargo.toml `version`): MINOR per release,
PATCH for hotfix re-release, MAJOR for era-class changes. Currently `0.9.x`;
`1.0.0` is a *confidence* milestone ("stable, proven"), NOT the gate for being
publicly testable — the `0.x` number is the only pre-1.0 signal we use. `build.rs`
stamps `CODA_VERSION` from `git describe` (engine tags only; `*-nets*` excluded so
net buckets can't read as engine versions). Never hand-edit a version string elsewhere.

**Ceremony:** bump Cargo.toml `version` → commit → `git tag vX.Y.Z && git push
origin vX.Y.Z`. `release.yml` builds the static embedded-net matrix (linux
x86-64-v2 + v3, linux-aarch64, windows-x86-64-v3, macos-aarch64 — all musl/static),
smoke-benches each, attaches binaries + sha256s. Hand-write notes (`gh release edit
… --notes-file`; auto-notes are disabled). **Do NOT set `--prerelease` for normal
releases (0.x included)** — CCRL gates on it, silently blocking the coverage we
want; reserve it for genuine RCs (e.g. `1.0.0-rc1`). Post-release, verify one
asset's sha256 + `id name`. Dry run without tagging: `gh workflow run release.yml`.
Ship both x86 v2 and v3 (v3 is meaningfully faster general codegen).

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
with intent before committing. Warnings accumulate into noise that masks real issues.

## Testing Methodology

### Before any OB operation → invoke the `ob` skill
The `ob` skill (`.claude/skills/ob`) is canonical for all OpenBench use: bench
measurement (incl. the critical net-override case), SPRT/SPSA submission, bounds
policy, stopping tests, reading results with early-N caveats, common failure modes.
Recurring bench-mismatch / stop-didn't-stop failures come from skipping it.

### Self-Play SPRT (primary acceptance test)
SPRT on OB with tight bounds is the primary acceptance criterion — all search/eval
changes normally pass it before merging (stress test + non-regression). Default
bounds **`[0, 3]`**.

**STC-first gating (even for LTC work).** STC (10+0.1) runs ~4× faster than LTC —
fire it first: it fail-fasts a doomed change ~4× quicker, can reveal bankable STC
wins that LTC misses, and checks the change doesn't disproportionately hurt the
other TC. Don't merge a change that only holds at one TC (unless it's a clear
bankable win at the other and neutral where it isn't).

**TM-class changes** (especially ponder-dependent — OB can't ponder, so SPRT
undersells them): (1) inspect the mechanism on 5-10 local games — parse per-move
clocks, confirm it fires as designed (`scripts/tm_pattern_inspect.py`); (2) primary
signal = ponder-enabled cross-engine RR at deployment-matched TC ratios (see the
`local-rr` skill); (3) cross-check with an LTC `[0,3]` SPRT for non-regression, but
don't gate merge on SPRT magnitude. Some TM gains still show up in STC SPRT.

**Core rules:** one change per branch (never stack untested); wait for H0/H1 (don't
stop on "looks good"); log every result to `experiments.md`; consider
retune-on-branch for tree-shape-changing features.

### Commit Messages
**Every commit that changes search/eval must include `Bench: <nodes>`** (OB uses it
to verify the built binary). Run `coda bench` with the production net.
```
Fix razoring margin at depth 2

Bench: 1780721
```

### SPRT Testing Policy
Every change affecting node count or strength (search logic, move ordering, "obvious"
bug fixes, NPS optimizations, NNUE inference) must be SPRT-tested before merging.
Workflow: feature branch → `Bench:` in the commit → push → submit SPRT → wait for
H0/H1 → merge (H1) or log to `experiments.md` (H0).

**Bounds: always range-3** (width H1−H0 = 3); **never range-6+** (`[-3,3]`,
`[-5,5]`, `[0,5]` lock H1 on noise or never separate). Default **`[0,3]`** ("does
this help?"). Deviations: **`[-1,2]`** complexity-free one-liner with an
externally-validated mechanism; **`[-2,1]`** ship-if-not-a-regression (bench-neutral
refactors, NPS-only, ARM ordering, tunables at default); **`[-1.5,1.5]`**
uncertain-direction / net-vs-net. Full table + rationale in the `ob` skill.
**No SPRT needed for** comments/docs/tooling, or cleanups with no compiled-output
change (verify bench unchanged).

### Mini-prod branch — DORMANT (since 2026-06-29)
Don't use `mini-prod` as an SPRT base (far behind main on code + tunables). For
current S200 net work, test **net-vs-net on `main`** with both candidates as
`--dev-network`/`--base-network` — the prod-tunable handicap is symmetric and
cancels — at `[-1.5,1.5]`. The asymmetry that motivated mini-prod only bites S200
*search/tunable* experiments (which perturb the trunk), not pure net/data ablations.
Revive only for such experiments (rebase onto main, keep S200-calibrated values,
focused retune); full workflow in `docs/mini_prod_branch_workflow.md`.

### SPSA Parameter Tuning
The `tunables!` macro in `search.rs` is the single source of truth (defaults,
ranges, c_end). `ob_tune.py` auto-derives the spec from the binary — **static
"all-params" cache files are forbidden** (they drift from source after every applied
tune and restart SPSA from stale defaults).
- **TC:** production tunes (outputs applied to trunk) run **LTC `40+0.4`,
  Hash=256** — Hash=64 at LTC is a TT-starved regime that distorts pruning
  economics; avoid Hash=512 (worker OOM at concurrency). Quick retune-on-branch
  validation can use STC; anything graduating to trunk gets final values from LTC.
- **Iterations** (at our ~60-param scale): ~2500 production; 1000-1500
  retune-on-branch / all-params (incl. `--core`); 800-1000 focused cluster. Longer
  is not consistently better at our scale — don't propose 10K+ runs.
- **Watch loose knobs.** `--core` has crept toward ~74; past ~60-70 params,
  non-load-bearing knobs add dimensionality + noise and can flip adjacent gradients.
  Fix by auditing/pruning (params SPSA pushes hard off-default or won't converge),
  not by adding iters. Default to `--core`, not full-sweep.
- Standard constants: c_end ~5-10% of range, r_end 0.002, alpha 0.602, gamma 0.101,
  A_ratio 0.1. LMR tables auto-reinitialize when `LMR_C_QUIET`/`LMR_C_CAP` change.
- **Always SPRT tuned values against main before merging** (SPSA overfits). Plan a
  retune after structural changes and after a net swap ("new net + old trunk" is a
  hidden detune). Submitting: see the `ob` skill.

### Retune-on-Branch
Some features are neutral untuned but gain Elo once pruning params are recalibrated
on their branch. Workflow: branch → SPSA tune the relevant cluster → compare
convergence vs a main baseline → if params diverge (>5% on several), apply + SPRT
the branch+tune against main; if they converge to main's values, the feature is
truly neutral — drop it. Tell: a big bench/node change with flat Elo → retune candidate.

### Feature Improvement Cycle (Detect → Diagnose → Fix → Tune)
Each fix compounds — it shifts optimal params elsewhere, exposing the next weak feature.
1. **Detect:** SPSA detuning a param >30% off its start (tuner compensating for a
   structural bug); ablation anomaly (disabling a feature gains Elo, or gains more
   than expected); our value far outside cross-engine consensus.
2. **Diagnose vs top engines** (sources in `/home/adam/chess/engines/`): learn the
   *idea* — exact formula, gating, position in the move loop (pre/post-MakeMove),
   depth variable (raw vs lmrDepth), whether it reuses the standard `see_ge`. Write
   Coda's own version; don't copy code. **Weigh the strongest engines** — reference
   set (strongest first): **Stockfish, Reckless, Obsidian** primary; Berserk,
   PlentyChess, Alexandria secondary. Coda is top-5 in our local pool, so mid-table
   consensus mostly tells us what we already do. **Never reference Ethereal** anywhere
   in the repo (code, docs, experiments, commits, test data). Common structural
   traps: pruning after MakeMove (wastes make/unmake + NNUE push/pop, redundant with
   earlier pruning); raw depth for depth² scaling (far more aggressive than intended);
   a duplicate SEE instead of `see_ge`.
3. **Fix + SPRT** at `[0,3]` with consensus-default constants — a correct structural
   fix should be positive even untuned. If it fails, look for secondary bugs (missing
   `ply>0` guard, bestScore pollution).
4. **SPSA the fix** — focused cluster first (~1000 iters), full `--core` after
   (rebalances params that were compensating for the old behavior).
5. **Repeat** — the retuned baseline exposes the next weak feature.

Historical examples + the cumulative list of resolved experiments: `experiments.md`.
