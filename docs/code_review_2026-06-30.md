# Full codebase review (2026-06-30)

First full-codebase sweep in a while. Method: `cargo clippy` (lint pass) +
7 parallel agents each reading a subsystem in full (search/TT, NNUE core,
NNUE threat/SIMD kernels, board/movegen, move-ordering/eval/SEE/book/TB,
UCI/CLI/EPD, training tooling) looking for dead code, bugs, stale comments,
sign errors, and refactor/optimization opportunities. ~37.7K lines covered.

**Nothing here has been fixed yet — this is the findings doc.** Each item
below should become its own branch + (where it touches search/eval/play)
an SPRT, per standing methodology. Pure-tooling/CLI/comment fixes don't
need SPRT, just a bench-neutral check where relevant.

## Headline finding: cuckoo repetition-cycle detection is mostly inert

`src/cuckoo.rs:130-141`, `has_game_cycle`. **[bug, confirmed]**

```rust
let mut other: u64 = original_key ^ key_at(1) ^ side_key();
for i in (3..=end).step_by(2) {
    other ^= key_at(i - 1) ^ key_at(i) ^ side_key();
    if other != 0 { continue; }
    let diff = original_key ^ key_at(i);   // correct value, used regardless
    ...
```

`diff` (the value actually used for the cuckoo-table lookup at line 140) is
computed correctly and independently every iteration. `other` is meant as a
cheap early-reject gate before paying for that lookup — but it doesn't
compute (or approximate) `original_key ^ key_at(i)`. Tracing the
accumulation: `other` telescopes to the XOR of **every** intermediate key
from `key_at(0)` through `key_at(i)` (the recurrence uses `key_at(i-1)`,
not `key_at(i-2)`, so it never cancels back down to just the two
endpoints). Real Stockfish's `has_game_cycle` has no such pre-filter at
all — it computes `moveKey = originalKey ^ stp->key` fresh every iteration
and looks it up directly, exactly like Coda's `diff`. Coda's `other` gate
is a bolted-on "optimization" that is mathematically wrong.

Net effect: `other == 0` is essentially uncorrelated with whether
`key_at(i)` is a real single-move-reversal away from the current position,
for `i >= 5` it requires an astronomical 64-bit coincidence. The `i == 3`
case only passes the one existing unit test
(`cuckoo_detects_knight_dance_at_ply_3`) because that test is a symmetric
out-and-back move, which makes the (wrong) accumulator zero by
construction (XOR is direction-symmetric). `fuzz_cuckoo_sanity` discards
the return value (`let _ = has_game_cycle(...)`), so it can't catch the
false negatives either.

**Practical impact:** proactive repetition/cycle avoidance — the entire
purpose of `cuckoo.rs` — is largely non-functional outside this narrow
coincidence case. Real strength gap, not a crash.

**Fix direction:** drop the `other` pre-filter entirely (just compute
`diff` directly every iteration, like upstream SF — it's already computed
regardless, the gate isn't saving meaningful work) — or, if the gate is
worth keeping for the array-lookup avoidance, fix the recurrence to
`other ^= key_at(i - 2) ^ key_at(i);` (no `side_key()`, 2-ply step). Once
fixed, `src/cuckoo.rs:176-182`'s historical-repetition rescan (currently a
linear O(end) scan per hit) becomes hot and is worth revisiting at the
same time — Stockfish uses a sticky per-position `repetition` flag instead
of rescanning.

**This is the single highest-value item in this review** — it's a defined
feature (CLAUDE.md: "Cuckoo cycle detection for proactive repetition
avoidance") silently not doing its job. Should get its own branch + SPRT.

## Other high-priority bugs

**1. `--nnue`/`-n` CLI flag broken for `convert-checkpoint`; `-n` collides
with `count` in 3 other subcommands.** `src/main.rs:91` declares a global
`--nnue`/`-n`; `ConvertCheckpoint` (`main.rs:528-529`) redeclares its own
local `--nnue` field with the same long name, so the global flag is never
visible to it — reproduced on the built binary (`--nnue X convert-checkpoint`
errors "required arguments not provided: --nnue"). Separately, `-n` is
reused as the short form for unrelated `count` fields in `InspectBinpack`,
`FuzzThreats`, `SamplePositions` — confirmed via `--help` showing two `-n`
bindings and runtime testing that `-n` always binds to the local `count`
there. Fix: drop `ConvertCheckpoint`'s local `nnue` field in favor of
`cli.nnue` (matches every other net-consuming subcommand); drop the
conflicting `short = 'n'` from the three count fields.

**2. `import_game_tsv`/`import_pgn` silently break binpack chain
compression.** `src/datagen.rs`, `write_game_entry` (`:109-122`) builds
every position via `SfPosition::from_fen(&board.to_fen())` instead of
chaining via `after_move()`. Coda's `Board::to_fen()` always emits the EP
square after a double pawn push regardless of capture legality; the
vendored `sfbinpack` crate's `after_move()` only sets `enpassant` when an
EP capture is actually legal. Since `is_continuation()` compares full
`Position` equality (including `enpassant`), this mismatch breaks the
chain on essentially every double-pawn-push position with no immediately
available EP capture — every such position gets written as a full ~32+
byte stem instead of a 2-3 byte continuation. **The same bug was already
found and fixed for self-play data in `play_one_game` in this same file**
(see its comment at `:673-675`) but the fix was never ported to the
PGN/TSV import paths, which are the documented production replacement for
the old Python PGN pipeline. Not data corruption — scores/moves/results
stay correct — but bloats the primary import tooling's output. Fix:
chain via `after_move()` the same way `play_one_game` does.

**3. `nnue_export.rs` ignores `use_pairwise`, hardcodes non-pairwise L1
shape.** `nnue_to_bullet_checkpoint` (`src/nnue_export.rs:43-64`) reads
`net.use_screlu` but never `net.use_pairwise`, and unconditionally
zero-inits `l1w` sized `2*h*l1_size` (the direct/CReLU-concat convention).
Production v7+ configs are pairwise, where L1 input is `h` not `2h`
(confirmed against `training/configs/v7_768pw_h16x32.rs` vs
`v7_1024h16x32s.rs`). Loading this checkpoint into a pairwise training
config will produce a parameter-count mismatch. This is the
import/export-direction drift the review brief specifically asked about —
`bullet_convert.rs` (import) and `nnue_export.rs` (export) have drifted on
net-shape assumptions.

**4. `bullet_convert.rs::convert_v5` size check is too weak to catch a
wrong net shape.** `:113` only checks `data_len < expected` where
`expected` is recomputed from a hidden size (`h`) that was itself derived
via integer division of `data_len` — so a non-evenly-dividing shape can
silently produce a wrong `h` that still passes the `<` check. Contrast
`convert_v7`'s `if expected != data_len` exact check (`:265`). Fix:
make `convert_v5` use exact equality too.

**5. `bullet_convert.rs::convert_v5` silently drops `kb_count`/`kb_layout`
for plain-CReLU (v5) output.** `write_extended_kb` is computed
unconditionally (`:170`) but only written `if version == 6` (`:172`); for
plain v5 output (no flags/kb-header bytes written at all), a non-default
king-bucket config is silently discarded rather than erroring, while the
loader's v5 arm hardcodes the 16-bucket default. Narrow real-world
likelihood (v5 is legacy) but a real unguarded gap.

**6. `src/nnue.rs` memory-safety: `HIDDEN32_BUF`/`H2_BUF` stack guards are
`debug_assert!`, unlike the sibling `PW_BUF` guard a few lines above.**
`:3343-3346` / `:3659-3662` in `forward_with_l1_pairwise_body`, vs the
deliberately-hardened `PW_BUF` check at `:3250-3254` (real `assert!`, with
a comment that an oversized net slipping past load-time validation "must
abort here ... genuine memory unsafety"). `l1_size`/`l2_size` are read
from the net-file header with no upper-bound load-time validation (only
`hidden_size` is checked, `:2797`). In a `--release` build (what OB/prod
actually run, `debug_assert!` is compiled out), a crafted/buggy net with
oversized per-bucket L1/L2 overflows these fixed stack buffers. Fix:
upgrade both to real `assert!`s, matching `PW_BUF`'s precedent, or add
load-time bounds validation for `l1_size`/`l2_size`.

**7. `src/nnue.rs`: `FinnyEntry.acc` sized for a half-width bound but the
loader permits double that.** `:4739`, `acc: [i16; NNUE_PW_BUF]` where
`NNUE_PW_BUF` is documented (`:173`) as a half-width bound (intended
`hidden_size ≤ 2×this`); the loader enforces `hidden_size ≤ 2*NNUE_PW_BUF
= 2048` (`:2793-2802`), but `refresh_accumulator` slices `entry.acc[..h]`
with the **full** `hidden_size` (`:5498,5576,5584`). Any net with
`hidden_size` in (1024, 2048] panics/OOBs on first Finny refresh after a
king-bucket crossing. Currently dormant (prod uses 768) but a real
landmine for future net widening.

**8. `src/nnue.rs::forward_with_threats` no-hidden-layer fallback drops
the NTM perspective; non-pairwise arm returns a position-independent
constant.** `:4282-4294`, reached when `has_threats && l1_size == 0`. Only
`stm_acc`/`t_stm` are used; `ntm_acc`/`t_ntm` are never read (every other
forward path in the file uses both). The whole block is further gated
`if self.use_pairwise { ... }` with no `else`, so `!use_pairwise` silently
returns `self.output_bias[bucket]` regardless of the position. Nothing at
load time rejects `has_threats && l1_size == 0`. Unreached by current
production v9 nets (which always have hidden layers), but real and would
silently corrupt eval if such a net were ever loaded.

## Dead code

| File | Item | Notes |
|---|---|---|
| `src/binpack.rs` | **entire file (~530 lines)** | Hand-rolled SF BINP writer with zero callers anywhere in the workspace — production binpack writing exclusively uses the external `sfbinpack` crate. Never exercised, never tested; any latent bug in it (it has its own unused-even-internally `nth_set_bit` helper) goes undetected forever. Delete, or mark clearly experimental. |
| `src/threat_profile.rs` | **entire file (~75 lines)** | `pub mod threat_profile;` registered in `main.rs:57` but no symbol referenced anywhere else. Delete or wire up. |
| `src/threats.rs:1629-1961` | `compute_move_deltas` (~330 lines) | Fully-implemented alternate delta-computation strategy, structurally different from the live `push_threats_on_move`/`push_threats_for_piece` path, zero callers. Risk of silent bit-rot if resurrected without re-audit. |
| `src/sparse_l1.rs` | `sparse_l1_avx2`, `sparse_l1_avx512_vnni`, `find_nnz_chunks4` | Zero-skip sparse kernels, deliberately superseded by dense kernels per `nnue.rs:3465-3473` (measured 1.8-2.4× slower at every density). `select_l1_kernel` only dispatches dense. Test-only currently; delete or mark test-only explicitly. |
| `src/movepicker.rs:968-998` | `fixup_move_flags` | Zero call sites; already flagged dead in `docs/research_threads_2026-04-24.md` but never removed. |
| `src/movepicker.rs:962-965` | `is_capture` | Zero call sites; equivalent logic is inlined ad-hoc at several other call sites instead of reusing it. |
| `src/tt.rs:892-918` | `TT::dump_to_file` | No call sites, not gated behind a debug subcommand. |
| `src/nnue.rs` | `simd_acc_copy_add`/`_sub` (AVX2), `simd512_acc_copy_add`/`_sub` (AVX-512), `neon_acc_copy_add`/`_sub` (NEON) — 6 functions | Superseded by the fused N-delta kernel `simd_acc_fused_avx2` (`:406-414`), whose own doc says it replaces this copy+per-delta pattern. |
| `src/nnue.rs:1081-1151` | `simd_screlu_dot` (16-bit-weight AVX2 variant) | Distinct from the live `simd_screlu_dot_i8`; unused. |
| `src/nnue.rs:1208-1219` | `hsum_epi32_to_i64` | Unused. |
| `src/nnue.rs:5775-5813` | `acc_add`/`acc_sub` dispatchers | Superseded by fused incremental path; the lower-level SIMD primitives they wrap remain used elsewhere. |
| `src/nnue.rs:4634-4635` | `AccEntry::threat_features_white`/`_black` fields | Never read/written outside declaration. |
| `src/nnue.rs:4599` | `MAX_HIDDEN_SIZE` constant | Unused as an actual bound, only appears in stale doc comments (see below). |

## Stale / misleading comments

- `src/setwise.rs:304-305` — references a `shiftv_avx512` function "below" that doesn't exist anywhere in the file or codebase (grepped for `rolv_epi64`/`shiftv_avx512`).
- `src/sparse_l1.rs:1-11` — module doc still describes the abandoned sparse approach ("~89% sparsity... processes only ~11%") as the shipped design; the correction already exists in `nnue.rs:3465-3473` but wasn't backported here. Misleads a reader of this file in isolation.
- `src/nnue.rs:1` — file header says "NNUE v5/v6 inference"; should be v5/v7/v9 (v6 isn't a real generation label elsewhere in the codebase).
- `src/nnue.rs:4601-4624` — `AccEntry` doc block describes a pre-refactor layout ("inline i16 arrays, no allocator calls") that no longer matches the struct (now heap-allocated `Vec`s). Directly contradicts the correct, adjacent `AccDataStack` doc a few lines below.
- `src/nnue.rs:5599-5602` — `finny_batch_apply_avx512` doc says "CHUNK=256, 3 passes for h=768"; actual constants (`:5632-5634`) are `REGS=24, CHUNK=768`, i.e. 1 pass — stale from a REGS 8→24 upgrade. Same comment block also mislabels h=1024 as "(current prod)", contradicting both this file (`:4650`) and CLAUDE.md (prod is 768).
- `src/nnue.rs:709` — `simd_crelu_dot` comment overstates the drain threshold by ~2× (says "after 128 pairs", actual drain is every 128 *elements* = 64 pairs). Code correct, comment imprecise.
- `src/nnue.rs:3135-3144` — `forward_with_l1_pairwise_fused` doc has accumulated edit cruft, describes a different/sibling function.
- `src/search.rs:1857` — orphaned `#[allow(dead_code)]` above `search_smp`, which is actively the main Lazy-SMP entry point (called from `uci.rs:526,660`). Misleading, not actually dead — safe to delete the attribute.
- `src/bullet_convert.rs:1-9` — module doc omits the v8 (dual-L1)/v9-v10 (threats) path, which is actually the file's main current-day output per `convert_v7`'s own version derivation (`:439`) and CLAUDE.md ("v9 is production").

## Medium-confidence bugs / gaps

- **`src/epd.rs:92-94`** — `move_to_san` returns `"O-O"`/`"O-O-O"` early for castling, skipping the check/mate-suffix logic below, so a castling move that gives check renders without `+`/`#`. Currently harmless (`run_epd`'s comparison strips `+`/`#` from both sides) but a real gap in a function documented as general SAN formatting.
- **`src/uci.rs:290-313`** — `ucinewgame` resets TT/history/corrhist/NNUE/TB-cache but never resets `ponder_limits`/`pondermiss_pending`. A stale pondermiss flag can survive into a new game if the GUI sends `ucinewgame` instead of `ponderhit`/`go` after a miss, applying the post-pondermiss min-think floor to an unrelated first move. Low impact (≤200ms), real state-hygiene gap.
- **`src/board.rs:833-842`** — `make_move`'s EP defensive validation checks the capture square holds an enemy pawn and the destination is empty, but never checks the *mover* is a pawn — unlike `movepicker::is_pseudo_legal`, which explicitly guards `pt != PAWN`. Not currently reachable (every call site gates through `is_pseudo_legal` first) but `make_move`'s own safety net is incomplete relative to its sibling check; cheap to add for parity/defense-in-depth.
- **`src/threats.rs:444-467`** — `PiecePair::base()`/`new()` bit-width mismatch: `new()` masks to 30 bits, `base()`'s accessor mask only preserves bit 31 + bits 0-23 (silently drops bits 24-29); doc comment claims 24 bits. Harmless today (feature count ~66,864 is far below 2^24) but a real width inconsistency between constructor/accessor/doc.
- **Test quality, `src/nnue.rs:7749-7810`** — `test_finny_incremental_consistency` is vacuous: both its "incremental" and "full recompute" branches call `force_recompute()`, never exercising `push`/`materialize`/Finny at all. Would pass even if the real incremental path were broken. (Other tests in the file, e.g. `fuzz_psq_accumulator`, do cover incremental correctness, so the feature isn't uncovered overall — just this specific test is dead weight.)
- **`src/sparse_l1.rs:1227-1488`** — fuzz tests cap input to stay inside int8-saturation envelope, citing "a separate `saturation` test below" that doesn't exist. Underlying gap is real-but-dormant: AVX2 kernels route through `VPMADDUBSW` (i16 intermediate saturation), AVX-512/VNNI kernels use `VPDPBUSD` (straight to i32, no intermediate saturation) — a genuine latent cross-kernel divergence if pairwise output ever drifts outside today's documented bound.
- **`src/nnue.rs`** cross-path inconsistency: `simd_screlu_dot_i8` (AVX2) accumulates the whole dot product in `i32` with no periodic drain; its AVX-512 sibling explicitly drains to `i64` every 512 elements "to prevent i32 overflow." Adversarial worst case at h=1024 exceeds `i32::MAX`; unlikely with real trained-net weight distributions but a real asymmetry.
- **`src/nnue.rs`** — `l1_out` stack buffer is `[0.0f32; 128]` on the AVX2/AVX-512 int8 path vs `[0.0f32; 512]` on the NEON/scalar fallback for the same logical quantity — bounds-checked so it panics rather than corrupts, but suggests one path wasn't updated when L1 width support was extended.
- **`src/nnue.rs:3882-3883`** — `simd512_l1_int8_dot_sparse` (the non-VNNI AVX-512 path) has no direct scalar-reference unit test, unlike its VNNI sibling; may go unexercised on typical modern hardware that has both AVX-512BW and VNNI.

## Refactor / duplication opportunities

- **Three independent reimplementations of move-legality rules** (`movepicker::is_pseudo_legal`, `movegen::generate_captures`/`generate_quiets`/`generate_castling`, `board::make_move`'s defensive checks) — no current divergence found, but this is exactly the pattern CLAUDE.md's "is_pseudo_legal must be thorough" note identifies as the root cause of three historical 320-Elo bugs. Worth a deliberate note even with nothing currently broken: any future rule change (new castling variant, EP rule tweak) must touch all three in lockstep.
- `src/movegen.rs:412-423` / `:427-437` — `is_attacked`/`is_attacked_with_occ` have byte-for-byte duplicate bodies; `is_attacked` could just call `is_attacked_with_occ(board, sq, by_color, board.occupied())`.
- `src/board.rs:110` — `ep_capture_available` reimplements `bitboard::east`/`west` inline instead of calling the existing helpers. Cosmetic only.
- `src/movepicker.rs:628-648` and `:803-822` — near-identical main-history/cont-hist/pawn-hist scoring blocks in `generate_and_score_quiets` and the quiet branch of `generate_and_score_evasions`. Currently in sync; a shared helper would prevent future drift.
- `src/see.rs:77-101` — `see_ge` recomputes pin masks from scratch (4 slider lookups + scans) on every call that reaches the iterative loop, even though pin masks only depend on board state, not the move. Repeated per-capture during move-picker scoring and again during search-side SEE pruning. Caching once per node would require widening `see_ge`'s signature — real but non-trivial.
- `src/uci.rs:1229-1234` and `:950-955` — the name/value token scan in `setoption` handling is implemented twice (once in `parse_option()`, once again right after for option-specific special-casing).
- `src/main.rs` — the `File::open(...).unwrap_or_else(...)` + `CompressedTrainingDataEntryReader::new(...)` boilerplate is repeated near-verbatim across 5 subcommands (`InspectBinpack`, `ChainStats`, `BinpackStats`, `SamplePositions`, `EvalDist`). A shared `open_binpack_reader()` helper would remove the duplication.
- `src/threat_accum.rs:167-177` — `ThreatStack::push(mv, moved_pt)` parameters are dead weight in production: every call site (`search.rs:3631,3685,3824,4239,5149,5360`) passes `NO_MOVE`/`NO_PIECE_TYPE`, and `absorb_deltas()` unconditionally overwrites them from board state anyway. Only unit tests pass real values. Worth simplifying the signature.
- `src/threat_accum.rs:418-444` — `ensure_computed` calls `can_update()` twice per perspective when ancestors diverge; the per-pov fallback could reuse the already-computed ancestor instead of re-walking. Pure redundant work, O(depth) per node when triggered.

## Clippy (cargo clippy --all-targets --features embedded-net,bz2)

Clean — **45 warnings, all minor style, zero correctness lints**:
- `manual checked division` (×9) — `main.rs`, `search.rs:2460`, `threat_profile.rs` (×6, in the now-flagged-dead module)
- `this if can be collapsed into the outer match` (×9) — `search.rs`, `uci.rs`, `board.rs`, `datagen.rs`, `movepicker.rs`
- `unsafe function's docs are missing a # Safety section` (×7) — all in `nnue_simd.rs`
- `consider using sort_by_key` (×4) — `main.rs` (×3), `book.rs`
- `named constant with interior mutability` (×4) — `search.rs:1512-1518`
- `manual implementation of .is_multiple_of()` (×2) — `datagen.rs`
- misc singles: empty-line-after-doc-comment (`search.rs:1857`, `threats.rs:8`), `this map_or can be simplified` (`main.rs:570`), unnecessary `&mut` (`movegen::generate_evasions`)

None of these are worth a dedicated pass on their own merit, but several overlap with findings above (the `search.rs:1857` empty-line warning is literally the orphaned `#[allow(dead_code)]` finding) and could be swept in the same cleanup commit.

## Recommended priority order

1. **Cuckoo fix** (own branch, SPRT `[0,3]` — likely a real, bankable Elo gain; this is the kind of "feature defined but not actually doing its job" bug the project's improvement cycle methodology is built to catch).
2. **CLI flag collisions** (`main.rs`) — no SPRT needed (doesn't touch search/eval), just a build + manual verification that `--help`/the affected subcommands work after the fix.
3. **NNUE memory-safety hardening** (`HIDDEN32_BUF`/`H2_BUF` asserts, `FinnyEntry.acc` sizing) — bundle as a defensive-only, bench-neutral commit; verify bench unchanged, no SPRT needed since it only changes behavior for malformed/oversized nets that don't currently exist in production.
4. **Training-tooling fixes** (`datagen.rs` chain-compression, `nnue_export.rs` pairwise shape, `bullet_convert.rs` v5 checks) — offline tooling, no SPRT, but worth fixing before the next training run that touches these paths.
5. **Dead-code sweep** — `binpack.rs`, `threat_profile.rs`, `compute_move_deltas`, the 6 superseded acc-copy SIMD functions, `fixup_move_flags`/`is_capture`, `dump_to_file`. Bench-neutral by construction (removing unreachable code); good batch cleanup commit.
6. **Stale-comment sweep** — cheap, no risk, can ride along with #5.
7. Everything else (refactor/duplication items, medium-confidence gaps) — opportunistic, pick up when next touching the relevant file rather than as a dedicated pass.
