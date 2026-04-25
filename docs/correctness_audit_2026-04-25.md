# Correctness Audit — Coda (2026-04-25)

## Methodology

Six parallel agents reviewed disjoint surfaces post-v9-merge-into-main. Building on the 2026-04-22 audit (which resolved ~40 of 44 actionable items); this fresh pass focused on **NEW code introduced in the ~107 commits since then** plus completeness checks on recent fix-merges. Severity tiers as before: CRITICAL / LIKELY / SPECULATIVE.

Audit surfaces:
1. NEW: eval-only TT writeback (#713) + AccEntry flatten (#711) + ARM Acquire/Release (#764)
2. NEW: recently exposed tunables (#741, #745) + applied tune #750 (+12.3 Elo)
3. Completeness check on recent fix-class merges (threats blocker, recapture-ext, FH-blend SE, SEE promo, qsearch OOB, NMP stale reductions, TT-malus symmetry, book/EGTB/ponder)
4. Broad fresh-eye scan of search/NNUE/SMP/UCI/build hygiene
5. Bullet trainer fork + factor SB800 + L1-sparsity branches
6. Coda training pipeline / net I/O hygiene

Summary: **4 CRITICAL, ~14 HIGH/LIKELY, ~20 SPECULATIVE.** Nothing in the search loop or NNUE/TT/SMP cores is structurally broken — the codebase is in better shape than 2026-04-22. The CRITICAL findings cluster in **(a) the Bullet C8 fix being incomplete on x-ray paths**, **(b) tooling regressions introduced by 2026-04-22 audit fixes themselves** (`fetch-net`, `--no-xray-trained` flag), and **(c) two more sites of the silent-zero-threat-features bug pattern**.

## CRITICAL findings

### C2026-04-25-1. Bullet C8 fix is INCOMPLETE — x-ray same-type-pair semi-exclusion still buggy

**File:** `bullet/crates/bullet_lib/src/game/inputs/chess_threats.rs:516, 548` (positive and negative x-ray rays)

The 2026-04-22 audit's headline bug (C8) was the training/inference frame mismatch on semi-exclusion of same-type pairs. Commit `a8e2c7d` patched the **direct-attacks path at line 460** to use physical-frame comparison `(sq ^ phys_flip) < (to ^ phys_flip)`. **The two x-ray sites at lines 516 and 548 were missed** — they still use raw `sq < xray_sq` (bulletformat-normalised frame).

Concrete trace: black-rook a8 with white-rook a1 blocker on the file, real STM=Black:
- Coda inference (`coda/src/threats.rs:200-204`): keeps `(black-rook a8 → white-rook a1)`, base index from `offset_table[3]`.
- Bullet training: keeps `(white-rook bf-sq=56 → black-rook bf-sq=0)`, base index from `offset_table[7]`.
- Different feature indices.

Affects same-type-pair x-rays — rook-behind-rook on files (very common in middlegame), bishop-behind-bishop on diagonals, queen behind R/B/Q. ~25–35% of middlegame positions have this configuration when STM=Black. The +8.25 Elo retune that landed after the C8 partial-fix was therefore measured on a partially-fixed pipeline. Net trained today still has divergent training/inference signal on this subset.

**Fix:** replace both `sq < xray_sq` with `(sq ^ phys_flip) < (xray_sq ^ phys_flip)` at lines 516 and 548. Verify with extended fuzzer covering same-type x-ray cases.

### C2026-04-25-2. `coda fetch-net` rejects every valid NNUE since 2026-04-22

**File:** `src/main.rs:958-964` (commit `b7c0312` from the 2026-04-22 audit fix)

```rust
if bytes.len() < 16 || bytes.iter().take(4).all(|&b| b.is_ascii_alphabetic()) {
    eprintln!("Error: downloaded file doesn't look like a .nnue ...");
    let _ = std::fs::remove_file(fname);
    std::process::exit(1);
}
```

The NNUE magic is `0x4E4E5545 LE` → bytes `[0x45, 0x55, 0x4E, 0x4E]` = `"EUNN"`, all ASCII-alphabetic. The check intends "rejects HTML pages with leading text" but every valid net trips it. **`coda fetch-net` has been broken in production since the audit hygiene fix landed.**

**Fix:** invert to positive validation: `bytes[0..4] != [0x45, 0x55, 0x4E, 0x4E]` rejects.

### C2026-04-25-3. UCI `eval` and CLI `eval-dist` silently drop v9 threat features

**Files:** `src/uci.rs:552-554` (UCI `eval`), `src/main.rs:1313` (`run_eval_dist`)

Both build `ThreatStack::new(0)` with default `active=false`. For v9 nets `forward_with_threats` (`src/nnue.rs:3407`) gates on `threat_stack.active && self.has_threats` — falls through to `forward()`, which then sees `acc.current().threat_accurate[..]` as false and uses `empty_threat = &[]`. **The threat half of the eval is silently zeroed.** In release builds `debug_assert!(!self.has_threats)` is a no-op.

The 2026-04-22 audit's C8 LIKELY #17/#20 patched this for `forward()` and `check-net` (now `run_check_net`). Two sites were missed:
- UCI `eval` command (user types `eval` at console)
- `coda eval-dist` (used by CLAUDE.md `EVAL_SCALE` calibration recipe — calibration measurements on v9 nets have been invalid)

**Fix pattern:** mirror `main.rs:411-413` — build a real `ThreatStack` with `active = net.has_threats`, refresh both perspectives before evaluating. Add a regression test that compares `bench` first-eval output to `eval` output for a v9 net.

**Cross-cutting recommendation:** sweep all `evaluate_nnue` call sites — there may be a third site with this same pattern.

### C2026-04-25-4. `--no-xray-trained` flag documented but doesn't exist

**File:** `src/main.rs:321-322` (commit `d290831`)

```rust
#[arg(long, default_value_t = true)]
xray_trained: bool,
```

Clap with `default_value_t = true` on bool requires `--xray-trained=<bool>` syntax (and `--xray-trained false`, `--xray-trained=false`, `--xray-trained=true` all error — only bare `--xray-trained` works as a no-op). Clap does **not** auto-generate `--no-xray-trained`. The help text and commit message tell users to "pass `--no-xray-trained`"; clap responds with `error: unexpected argument '--no-xray-trained' found`.

The intended escape hatch for converting `--xray 0` Bullet nets is unusable. Today moot (no `--xray 0` nets in production), but the L1-sparsity probe runs do flip `--xray`.

**Fix:** rename to a positive flag (e.g. `--xray-untrained`) without default, or use `clap::ArgAction::Set` with `BoolishValueParser`.

## HIGH / LIKELY findings (clustered)

### Search / new code

| Location | Issue |
|---|---|
| `src/search.rs:2715-2716` | **N6 promotion-imminent extension (#637, +1.6 H1) is dead code.** Reads `board.side_to_move` AFTER `make_move`, so STM is the OPPONENT. Both branches unreachable: white-to-rank-7 sees `side_to_move==BLACK`, black-to-rank-2 sees `side_to_move==WHITE`. Use pre-move `us` from line 2151. **Re-test could land bigger.** |
| `src/search.rs:90` | **`SEE_CAP_MULT` is a dead tunable.** Declared in `tunables!` macro and exposed via UCI/SPSA but never read anywhere in `src/`. Tune #750 perturbed it 152→171 — wasted SPSA budget. The capture-SEE-prune formula uses `SEE_MATERIAL_SCALE`. CLAUDE.md docs the formula incorrectly. Either delete or wire it in. |
| `src/search.rs:76` | **`NMP_VERIFY_DEPTH` SPSA-pinned to floor.** Range `[8, 20]`; tune #750 drove it 13→8 (exact floor). Strong directional signal that the optimum is below 8. Widen min to 4-6 next tune. |
| `src/search.rs:2136-2147` | **Hindsight reduction not gated on `excluded_move`.** Other modifiers touching SE re-search (NMP, RFP, ProbCut, FH-blend, TT-store, corrhist) are all gated. Consistency gap. May weaken singular detection at the SE re-search ply. |
| `src/uci.rs:552-554` | UCI `eval` silently zeros v9 threat features (Critical C2026-04-25-3, listed here for cross-reference). |
| `src/movepicker.rs:644-647` | **Mobility-delta uses pre-move occupancy for `to_mob` on sliders.** Own piece at `from` blocks the back-ray. Under-counts slider mobility. SPSA partially compensated by tuning `MOBILITY_DELTA_WEIGHT` 32→30 but signal is noisier than intended. |
| `src/movepicker.rs:643` | **Mobility-delta color extraction `(piece >> 3) & 1` is wrong for black pawns/knights** (piece=6,7 → 0). Latent — pawn/knight count is color-symmetric, no measurable impact today, but a trap if anything else gets wired into this code path. |

### Bullet trainer

| Location | Issue |
|---|---|
| `bullet/crates/bullet_lib/src/value/loader/viribinpack.rs:178-211`, `text.rs` | **C8 fix doesn't cover viribinpack or text loaders.** They produce `bulletformat::ChessBoard` without setting `extra[0]`, so `phys_flip` reads 0 even for real-STM=Black — silently re-introduces C8. Production T80 binpack goes through `sfbinpack` (covered), so production safe today; gap is latent. |
| `bullet/` (local working tree) | **Local checkout 2 commits behind origin** (`a1530a9` vs `281efb3`). Missing the C8 direct-path fix and the `MAX_THREAT_ACTIVE 256→512` + `--factoriser`/`--l1-decay` mutex. Anyone training from this tree pre-pull regenerates the C8 bug. One `git pull --ff-only` plus rebuild. |
| Coda main branch | **C8 threat-frame fuzzer never landed.** Lives only on `origin/fix/c8-threat-fuzzer` (`9571622`). Both `enumerate_threats_bullet_ref` and `coda fuzz-threats` subcommand absent from main. Could have caught the x-ray frame hole (CRITICAL #1). Cherry-pick recommended; expand to cover real-STM=White same-type-pair x-rays too. |
| `bullet/crates/bullet_hip_backend/kernels/base/optimiser.cu:107-140` | **C9 HIP Adam ABI still broken** on both `feature/threat-inputs` and `feature/decouple-l1-lr` branches. C++ kernel ignores the `l1` param, every subsequent param shifts by one slot, `gradients` becomes a wild pointer. Production unaffected (CUDA path), but flagged as deferred. |
| `bullet/crates/bullet_lib/src/game/inputs/chess_threats.rs:482-484` | **`--xray 0` training still produces an under-trained net.** Random-init x-ray rows that activate at inference. Coda v10 file format + `refuse-to-load` mitigates if user honors `--xray-trained`/`--no-xray-trained`. Gap: `--no-xray-trained` flag doesn't exist (CRITICAL C2026-04-25-4). |

### Training pipeline / net I/O

| Location | Issue |
|---|---|
| `src/bullet_convert.rs:170-183` (writer) vs `src/nnue.rs:2143-2168` (reader) | **v5/v6 extended_kb header writer/reader still mismatched.** `convert-bullet --kb-layout reckless` (without `--hidden`) writes 2 extra header bytes the reader doesn't consume. Silent corruption. Audit-flagged 2026-04-22, still unfixed. |
| `src/nnue_export.rs:14, 64` | **`nnue_to_bullet_checkpoint` hardcoded for non-pairwise + 16-bucket + no-threats.** `NNUE_INPUT_SIZE = 12288`, `l1w = vec![0.0f32; 2 * h * l1_size]` (wrong for pairwise), no `has_threats` branch. v9 nets cannot be checkpointed for transfer learning. |
| `src/datagen.rs:452-456` | **Material-removal `is_valid_position` doesn't verify opponent-not-in-check.** Audit fix was partial (castling rights + EP cleanup landed; opponent-in-check check missing). Illegal positions get scored by deep search and pollute training data. Add `if board.attackers_to(opp_king_sq, opp_color) != 0 { return false; }`. |
| `src/main.rs:801-838` (`coda dump-threats`) | **Advertises ground-truth output but only prints direct attacks.** Collects features via `enumerate_threats` (which emits direct AND xray) into a Vec, then **never reads it** — re-iterates piece by piece and prints only direct via `threat_index(...)`. Useless as fuzzer ground truth. Undermines C8 verification flow. |
| `Makefile` `net:` target | Uses `curl -sL` without `-f`. The `fetch-net` hygiene fix from 2026-04-22 was never propagated. HTTP 4xx/5xx response bodies become "net" file with curl exit 0; `cargo build --features embedded-net` then embeds an HTML error page. |
| `src/main.rs:1313` (`run_eval_dist`) | Same v9 silent-zero-threats bug as UCI `eval` (CRITICAL C2026-04-25-3). |

### NNUE / accumulator

| Location | Issue |
|---|---|
| `src/threat_accum.rs:67, 83` | **`ThreatEntry::values: [[i16; 768]; 2]` hardcoded.** Consumers slice with `..hidden_size`; any future net with `hidden_size > 768` panics on first refresh. Add `assert!(hidden_size <= 768)` in `ThreatStack::new` for defence-in-depth. |
| `src/nnue.rs:3656-3945` | **Legacy threat pipeline is unreachable but still in source.** `store_threat_deltas` and `recompute_threats_if_needed` have zero callers (production v9 routes through `info.threat_stack`). Kept alive only by `debug_assert!`. Future contributor may re-route v9 through it expecting it works. Recommend deletion or `#[deprecated]`. |
| `src/nnue.rs:4153-4156` | `NNUEAccumulator::push()` doesn't reset legacy `threat_move`/`threat_moved_pt`/`threat_moved_color` fields (still on `AccEntry`). Dormant because path is dead. Cleanup-when-touched. |

## SPECULATIVE findings

Brief list — fix when touching the area.

- `src/threat_accum.rs:926` — same UB-at-63 shift pattern as the threats.rs:456 fix, inside `#[cfg(test)]` only. Apply same guard for symmetry.
- `src/search.rs:2125` — IIR not explicitly gated on `excluded_move`. Incidentally safe today via `tt_move != NO_MOVE`.
- `src/search.rs:2077-2078, 2101-2102` vs `1703, 1759, 1889, ...` — inconsistent `ply_u < MAX_PLY` vs `<= MAX_PLY` writes. Both in-bounds. Standardise.
- `src/search.rs:2925` — `LMR shallower margin v2 (#679)` `best_score + 20` is silently inactive in heavy-pruning regimes (`best_score == -INFINITY`). Not wrong, just feature-degenerate.
- `src/search.rs:2462` — `gives_direct_check` magic-bitboard cost in the move-loop hot path adds ~1-2% search time. Net positive (+2.5 Elo merged).
- `src/search.rs:74` — `NMP_EVAL_DIV` trending toward floor (104 → min=100). Widen min to 50 next tune.
- `src/search.rs:701` — `apply_halfmove_scale` sentinel guard `score <= -INFINITY + 1` is fragile to future sentinel changes.
- `src/search.rs:1696-1698` — at `ply_u == MAX_PLY` overflow path, `pv_table[MAX_PLY][..]` may hold stale moves from prior visits. Bestmove still correct (derived from root); only printed PV may show garbage suffixes.
- `src/tt.rs:514-543` — `score_to_tt`/`score_from_tt` thresholds match `TB_WIN - 128` but `MAX_PLY = 64` lives in `search.rs`. If MAX_PLY ever raised past 128, TB scores corrupt again.
- `src/tt.rs:30` — TT `static_eval` clamp ±4095cp shifts on TT-cache-hit. Pre-existing; eval-only TT writeback (#713 +14.7) makes it fire ~3× more often (6.62% → 19.88%). **Load-bearing for #713's measured gain.** Worth instrumenting.
- `src/tt.rs:102-103` — `clear()` uses `Relaxed` while readers use `Acquire`. Single-threaded today. Defensive Release upgrade is free on x86.
- `src/threat_accum.rs:135-138` — `ThreatStack::pop` underflow only `debug_assert!`'d. Catches via slice OOB in release. Prefer saturating-sub.
- `src/board.rs:493` (`gives_direct_check`) — skips EP capture removal in occupancy. Documented as intentional (heuristic carve-out for futility); discovered checks via EP missed.
- `src/movepicker.rs:250, 358` (`cont_hist_subs`) — raw pointers into `info.history.cont_hist`; assumes no concurrent mutation during move loop. Currently safe.
- `src/main.rs:684-727` — `coda patch-net` writes byte 8 with no NNUE-magic check. Destroys non-NNUE files passed by typo.
- `src/main.rs:1249-1250` — `sample-positions` filter ("score < 2000 AND not in check") doesn't match training quiet-filter (no checks/captures/tactical, ply ≥ 16, score ≤ 10000). Misleading for net diagnostics.
- `src/binpack.rs:148-150` — EP-pawn nibble doesn't differentiate by color (W and B both → 12). Dead code today.
- `src/main.rs:486-514` (`EvalBench --mode incremental`) — `tstack.ensure_computed` called once before inner loop, never updated as make/unmake execute. Mean_eval is stale for v9 nets. Diagnostic-only.
- `src/nnue.rs:3911-3946` (`force_recompute`) — for v9 nets in `CODA_VERIFY_NNUE` mode, only refreshes white/black PSQ; verify path uses incremental threat_stack on both sides → catches PSQ bugs but not threat-incremental bugs.
- `src/search.rs:1769, 582` — `should_stop` time-check granularity is effectively 4096 nodes despite the 1024-node call cadence. Known constraint.
- Bullet `crates/acyclib/src/device/cpu/cmp.rs` — cross-backend test only exercises `l1_decay = 0.0`. No CPU↔CUDA equivalence test for active L1 or group-lasso. Add one before merging the lr-decouple branch.
- Bullet `coda_v9_768_threats.rs:362-365` — `group_l1_row_size` parameter name is misleading (with column-major storage, contiguous chunk is one input column not row). Cosmetic.

## Confirmed-clean (verified during scan)

Search loop / NNUE / SMP cores:
- Push/pop balance for `nnue_acc` and `info.threat_stack` across NMP, ProbCut, main move loop, qsearch.
- `info.excluded_move[ply_u]` set/clear pair around SE has no early return between them.
- `info.reductions[ply_u]` reset at node entry (audit C3 fix correctly applied).
- `moved_piece_stack` / `moved_to_stack` null sentinel on NMP (audit C3 fix correctly applied).
- Helper threads correctly mirror main's threat_stack initialization for v9 nets.
- TT XOR + Acquire/Release on aarch64 (commit `54586b5`) correctly applied at probe and store; reader Acquire-pair correctly synchronizes with writer Release-pair.
- TB cache `effective_key` mixes halfmove (audit C2 fix correctly applied).
- Threats blocker bounds at `sq=63` (commit `97b805f`) correctly fixed in both `threats.rs` enumeration paths.
- FH-blend skip during SE verification (commit `15bb1aa`) correctly skips when `info.excluded_move[ply_u] != NO_MOVE`.
- Eval-only TT writeback (#713) verified safe at every consumer: TT cutoff, near-miss, ProbCut, SE, IIR, QS — all gated.
- AccEntry flatten (#711) verified bit-equivalent: every consumer slices with `[..h]`, SIMD bounded by `h` parameter, `MAX_HIDDEN_SIZE=2048` covers v9.
- Recapture-extension `ply > 0` guard correct; other `undo_stack` reads are semantically distinct (don't need the same guard).
- TT-cutoff cont-hist malus correctly reads from `moved_piece_stack` for read/write symmetry on promotions.
- Bullet v10 file format `training_flags` byte + `--load-anyway` triple-route correctly converges on a single atomic.
- Bullet factoriser save-time fold-in is correct (column-major `repeat(NKB)` semantics matched).
- Bullet `--xray` + Coda refuse-to-load is consistent for sfbinpack/montybinpack production data path.
- Tune #750 big-movers all within ranges; LMR table init handles changed `LMR_C_QUIET=120, LMR_C_CAP=93`.
- Build warning-free for both release and `--tests`.
- CLAUDE.md spot-check: "LMP non-PV only", "TT 5-slot buckets", "RFP depth<=RFP_DEPTH" all match current code.

## Recommended fix order

**Batch 1 — CRITICAL, quick wins:**
1. **Bullet C8 x-ray frame fix** (lines 516, 548) → `bullet/fix/c8-xray-semi-exclusion`. One-line each. Then re-train v9 net.
2. **`fetch-net` magic check** → `fix/fetch-net-magic-positive`. One-line.
3. **UCI `eval` + `eval-dist` v9 threat-stack** → `fix/eval-v9-threat-stack`. Pattern from `main.rs:411-413`.
4. **`--no-xray-trained` flag** → rename to `--xray-untrained` (positive flag) or use `BoolishValueParser`. One-line per fix.

**Batch 2 — HIGH, structural:**
5. **N6 promotion-imminent extension dead code** — fix STM check, re-test. Could land bigger than the original +1.6.
6. **C8 fuzzer cherry-pick** to main — would have caught Batch 1 #1. Extend to cover same-type x-ray cases.
7. **`SEE_CAP_MULT` dead tunable** — wire it in or delete.
8. **`NMP_VERIFY_DEPTH` floor pin** — widen min to 4 next tune.
9. Mobility-delta sliders / color extraction fixes.

**Batch 3 — Tooling hygiene cluster:**
10. v5/v6 extended_kb header consistency.
11. `nnue_export` v9 support (pairwise / threats / kb_count).
12. `dump-threats` actually print all features.
13. Material-removal opponent-in-check check.
14. Makefile `net:` target `curl -f`.
15. `bullet` repo `git pull` before next training run.

**Batch 4 — SPECULATIVE / fix-when-touched:**
- All SPECULATIVE items above. Lower urgency, opportunistic.

## Cross-cutting observations

- **The 2026-04-22 audit's C8 fix was incomplete in two ways:** (a) only the direct-attacks path was patched; same-type x-ray paths remain divergent, and (b) the fuzzer that should have caught (a) was never merged to main. The +8.25 Elo retune that landed after the C8 partial-fix was therefore measured on a partially-fixed pipeline. Re-train on full C8 fix is likely worth more Elo.
- **Tooling regressions cluster around the audit hygiene fixes themselves** — `fetch-net` broke its own audit fix; `--no-xray-trained` was specified but not actually implementable in clap. Suggests audit-merged-as-correctness fixes deserve a manual smoke-test pass before relying on them.
- **Three sites of the same v9 silent-zero-threats bug pattern** (audit C8 LIKELY #17/#20 fixed two; UCI `eval` + `eval-dist` still affected). Sweep all `evaluate_nnue` call sites.
- **Tune #750 had three issues simultaneously**: dead tunable being perturbed, floor-pinned tunable, and a feature (#637 N6 promotion-imminent extension) that doesn't actually fire because of a post-make_move STM bug. The +12.3 Elo SPRT was real, but it was measured on a search where one feature was secretly disabled and one tunable was secretly noise. Worth re-testing after fixes to triangulate.
- **No new structural bugs in search/NNUE/SMP/TT cores.** The +25 Elo of NPS optimisations + ~+50 Elo of feature/fix merges since 2026-04-22 didn't introduce any deep architectural issues. The codebase is in better shape than the prior audit.

## Companion docs

- `correctness_audit_2026-04-22.md` — prior audit; resolution log shows what was actioned.
- `next_ideas_2026-04-21.md` / `peripheral_mechanisms_2026-04-22.md` / `research_threads_2026-04-24.md` — research-side ideation context.
- `improvement_portfolio_2026-04-24.md` / `next_100_elo_2026-04-24.md` — Elo-portfolio context.
- `experiments.md` — source of truth for resolved SPRTs.
