# Correctness Audit — Coda (2026-04-22)

## Methodology

Nine parallel agents (ten counting the Bullet-trainer audit, which is still running at write-time) reviewed disjoint subsystems for correctness bugs that pass SPRT but are silently wrong. Each agent was briefed with recent bug history in its area and asked to find file:line issues in three severity tiers: CRITICAL, LIKELY, SPECULATIVE.

Audits performed:
1. Threat subsystem (`threats.rs`, `threat_accum.rs`, consumers)
2. NNUE + accumulator state (`nnue.rs`, `sparse_l1.rs`, SMP init)
3. Search pruning gates (all of `search.rs`)
4. TT + history + correction (`tt.rs`, `tb_cache.rs`, `movepicker.rs` scoring, corrhist)
5. MovePicker + movegen + `is_pseudo_legal` (`movepicker.rs`, `movegen.rs`, `board.rs`)
6. Time management + ponder + UCI (`search.rs` TM, `uci.rs`, `main.rs`)
7. SMP + helper threads (`search.rs` helpers, `tt.rs` atomics)
8. SEE + Zobrist + Cuckoo + TB (`see.rs`, `zobrist.rs`, `cuckoo.rs`, `tb.rs`)
9. Training pipeline + net I/O (`datagen.rs`, `bullet_convert.rs`, `nnue_export.rs`)
10. Bullet trainer additions (feature/threat-inputs branch) — *still running, separate doc*

Summary: **9 CRITICAL, ~35 LIKELY, ~30 SPECULATIVE.** Many of the CRITICAL items are latent (affect edge conditions or near-50mr play) but are the kind that show up on Lichess, long TC, or when users mix UCI modes — classic "SPRT never saw it" bugs. **The Bullet audit found a potentially high-impact training-inference frame mismatch (C8) affecting ~50% of training positions — flagged for immediate investigation.**

## Reading the severity

- **CRITICAL** — actively wrong in reachable code paths today. Fix first.
- **LIKELY** — structural bug that bites under reachable but narrower conditions. Fix batch 2.
- **SPECULATIVE** — latent trap, dead code, or defensive hole. Fix when touching the area.

## CRITICAL findings

### C1. `is_pseudo_legal` EP validation missing from-rank and file-adjacency checks

**File:** `src/movepicker.rs:899-908`

```rust
if flags == FLAG_EN_PASSANT {
    if pt != PAWN { return false; }
    if to != board.ep_square { return false; }
    let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
    if cap_sq >= 64 || (1u64 << cap_sq) & board.pieces[PAWN as usize] & board.colors[them as usize] == 0 {
        return false;
    }
    return true;
}
```

A TT-collision move `(from=a2, to=d6=ep_square, flag=FLAG_EN_PASSANT)` in a position where `ep_square=d6` passes validation. `make_move`'s EP branch only checks `cap_sq` has enemy pawn and dest is empty — both satisfied. Result: white pawn **teleports from a2 to d6** and black's d5 pawn is removed. `is_legal` simulates EP and checks for discoveries but passes. Same bug class as the historical 320-Elo hole.

The existing `fuzz_is_pseudo_legal_rejects_corrupted` only mutates one field at a time; to hit this you need from + to + flag corrupted simultaneously.

**Fix:** add from-rank check (rank 5 for white, rank 4 for black) and file-adjacency (`|from_file - to_file| == 1`). Extend fuzzer to mutate multiple fields when `ep_square != NO_SQUARE`.

### C2. TB probe cache keyed on Zobrist alone — drops halfmove context

**File:** `src/tb_cache.rs:89-114`, also `src/tb.rs:64-76`

```rust
pub fn probe(&self, key: u64) -> Option<i32> {
    // key = board.hash — encodes pieces, side, castling, ep. NOT halfmove.
    let slot = &self.slots[(key & self.mask) as usize];
    ...
}
```

`shakmaty-syzygy::probe_wdl` is halfmove-aware (via FEN halfmove); the same piece layout returns `Win` at halfmove=0 vs `CursedWin` at halfmove=80. First probe caches the result. The same position later reached at reset halfmove=0 still gets the cached `CursedWin`, so search accepts a draw it could have won. Interior-node TB cutoffs at `search.rs:1688-1707` use this cache directly.

**Fix:** mix halfmove into cache key (e.g. `key ^ HALFMOVE_KEY[halfmove]`), or disable cache when `halfmove > 0`.

### C3. NMP leaves `moved_piece_stack[ply_u]` stale before recursing

**File:** `src/search.rs:2008-2013`

```rust
board.make_null_move();
info.tt.prefetch(board.hash);
if let Some(acc) = &mut info.nnue_acc { acc.push(DirtyPiece::recompute()); }
if info.threat_stack.active { info.threat_stack.push(...); }
let null_score = -negamax(board, info, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
```

`moved_piece_stack[ply_u]` and `moved_to_stack[ply_u]` are never written for the null move. The child at ply+1 reads `moved_piece_stack[ply_u]` as "parent's move" — gets whatever prior sibling or unrelated visit left there. That stale `(piece, to)` feeds cont-hist lookups, history-pruning, and LMR history adjustment in every null-move subtree. Stockfish explicitly sets the null sentinel on make_null.

**Fix:** `info.moved_piece_stack[ply_u] = 0; info.moved_to_stack[ply_u] = 0;` before recursing.

### C4. NEON SCReLU pack uses `>>9`, but scalar tail and x86 use `>>8`

**File:** `src/nnue.rs:1711-1712`

```rust
// neon_screlu_pack (SIMD body)
let d0 = vreinterpretq_s16_u16(vshrq_n_u16::<9>(vreinterpretq_u16_s16(sq0)));
let d1 = vreinterpretq_s16_u16(vshrq_n_u16::<9>(vreinterpretq_u16_s16(sq1)));
// ...
// scalar tail:
out[i] = ((v * v) >> 8) as u8;
```

`simd_screlu_pack` (line 692) and `simd512_screlu_pack` (line 1323) use `>>8`. Commit `44baa95` intentionally changed `neon_pairwise_pack` to `>>9` (matching FT_SHIFT) but also touched `neon_screlu_pack` by mistake. On aarch64 with a v7 **non-pairwise** SCReLU net, the NEON-packed activations are half size of x86; `h_val * h_val` is ~4× smaller downstream. Silent eval drift on aarch64.

No `neon_screlu_pack` vs scalar test exists (unlike `neon_pairwise_pack_fused`).

**Fix:** align the shift between NEON body and scalar tail (both `>>8` or both `>>9`, matching x86). Add NEON-vs-scalar consistency test.

### C5. `score_to_tt` / `score_from_tt` threshold at `MATE_SCORE - 100 = 28900` excludes TB_WIN scores (28800 band)

**File:** `src/tt.rs:489-512`

```rust
pub const MATE_SCORE: i32 = 29000;
pub const TB_WIN: i32 = 28800;
pub fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score > MATE_SCORE - 100 { score + ply }
    else if score < -(MATE_SCORE - 100) { score - ply }
    else { score }   // TB_WIN = 28800 lands here
}
```

TB_WIN is 100cp below the ply-adjust threshold. `TB_WIN - ply` scores pass through TT store/load unadjusted — retrieved at different ply they are wrong by `ply_diff`. All mate-guard sites in `search.rs` (e.g. 1738-1743, 1849, 2278, 3125) use the same threshold and so bypass TB scores too.

Stockfish convention: `VALUE_TB = VALUE_MATE - MAX_PLY - 1` (above the threshold). Reckless similarly.

**Fix:** widen the threshold (e.g. `TB_WIN - MAX_PLY`) or redefine TB_WIN above the threshold. Update all mate-guard sites uniformly.

### C6. Stale `soft_limit` / `hard_limit` persist across `go` commands

**File:** `src/search.rs:1168-1255`

```rust
if limits.infinite { /* zeroes all four */ }
else if limits.movetime > 0 { info.time_limit = limits.movetime; info.soft_floor = ...; }
// soft_limit and hard_limit NOT touched
else if our_time > 0 { /* sets all four */ }
else if !limits.infinite { info.time_limit = 0; }
// soft_limit and hard_limit NOT touched
```

`SearchInfo` is reused across `go` commands. `ucinewgame` doesn't clear `soft_limit`/`hard_limit` either. After any prior `go wtime/btime`, a subsequent `go movetime 30000`, `go depth 20`, or `go nodes N` enters the dynamic-TM gate at line 1463 with stale values, scales them, and breaks the ID loop at `elapsed >= adjusted_soft`. `go movetime 30000` can return at 1–3 s. `go depth 20` can cut short.

Triggers with any client mixing TC modes (Lichess analysis, SPRT harnesses, cutechess-cli edge cases).

**Fix:** explicitly zero `soft_limit`, `hard_limit`, `soft_floor`, and `tm_*` at the top of every non-`our_time` branch (or unconditionally before all branches).

### C8. Bullet training / Coda inference semi-exclusion frame mismatch — affects ~50% of positions

**File:** `bullet/crates/bullet_lib/src/game/inputs/chess_threats.rs:435` vs `coda/src/threats.rs:200-204`

```rust
// Training (bullet):
if att_type == vic_type && (att_color != vic_color || !is_pawn_att) && sq < to {
    continue;
}
```

Training semi-excludes same-type pairs using `sq < to` where both squares are in the **bulletformat-normalized frame** (rank-XOR-flipped when real STM = black). Coda inference's `PiecePair::base(from, to)` uses **physical squares** (per commit comment "Semi-exclusion uses PHYSICAL squares to match Bullet training" — but the semantics diverged).

Consequence: when real STM is black, the rank-XOR reverses the ordering relation for pieces on different ranks. Concrete trace: black rooks at physical c3 and c6 with real STM = black:
- Training emits `stm_idx = base + POL[45] + AIL[45][21]`
- Inference emits `stm_idx = base + POL[21] + AIL[21][45]`
- **Different feature indices.**

Also: Reckless inference applies semi-exclusion in POV-relative frame (post-flip), which is yet a **third** scheme. All three engines' training/inference/reference are inconsistent.

Affects: same-type pairs (doubled rooks, double bishops, opposing-colour pawn confrontations, queen-queen, knight-knight, king-king) on different ranks in positions where real STM is black. Roughly 50% of training positions × several features each — a meaningful fraction of threat emissions are **trained on features that never activate at inference, and activated at features that were never trained**.

This is likely a multi-Elo bug on v9 nets. It may also explain some of the v9 NNUE "flatter eval distribution" observation we've attributed to architecture rather than a training-signal leak.

**Fix path:**
1. Build a fuzzer that round-trips random positions through both `ChessBucketsWithThreats::map_features` (bullet) and `coda::threats::enumerate_threats` for both perspectives. Diff the feature sets. Any non-empty diff is a mismatch.
2. Use `coda dump-threats` CLI as the ground-truth source.
3. Decide the canonical frame: (a) physical squares everywhere — change training, retrain — or (b) STM-relative squares — change inference.
4. Coordinate with any reference engines whose training data we may share (Reckless) for consistency.

**Severity:** CRITICAL. Potentially the highest-impact finding in this audit because it's been silently corrupting training signal on every v9 net.

### C9. Bullet HIP-backend Adam kernel ABI mismatch — drops `l1` parameter

**File:** `bullet/crates/bullet_hip_backend/kernels/base/optimiser.cu:107-140`

Rust extern declares 10 scalar args (`decay, l1, min, max, ...`); C++ kernel signature reads 9 scalars (no `l1`). On HIP/AMD: undefined behaviour — `l1` float is interpreted as `min`, `min` as `max`, `max` as pointer-aligned junk. Inner kernel invocation at line 125 also drops `l1`.

**Not active in Coda production** — our Bullet training uses CUDA (`bullet_cuda_backend`), not HIP. But if any GPU host ever runs the HIP path (AMD GPU training), all optimiser calls silently corrupt. Future-proofing concern only.

**Fix:** align HIP `Adam` C ABI with Rust + CUDA signature, rebuild HIP kernel artifact.

### C7. `convert-checkpoint` momentum/velocity files miss the output layer when `l2_size == 0`

**File:** `src/nnue_export.rs:102-113`

```rust
write_weight_entry(&mut zero_buf, "l1w", ...);
write_weight_entry(&mut zero_buf, "l1b", ...);
if l2_size > 0 {
    write_weight_entry(&mut zero_buf, "l2w", ...);
    write_weight_entry(&mut zero_buf, "l2b", ...);
    write_weight_entry(&mut zero_buf, out_name_w, ...);
    write_weight_entry(&mut zero_buf, out_name_b, ...);
}
// When l2_size == 0, weights.bin writes l2w/l2b AS the output layer,
// but momentum.bin/velocity.bin only write l0*/l1*
```

For a v7 no-L2 transfer-learning run, Bullet's optimiser loader will fail or leave the output-layer momentum uninitialised. Silently produces a wrong trained net.

**Fix:** mirror weights.bin's entry list exactly in momentum.bin / velocity.bin.

## LIKELY findings (grouped)

### Search pruning

| Location | Issue |
|---|---|
| `search.rs:2312-2320` | `is_pv` shadow inside singular extension uses current `alpha` not `alpha_orig`. After prior moves raise alpha, `beta - alpha` collapses to 1 and double-extension can activate on PV nodes, violating no-DEXT-at-PV. |
| `search.rs:1915-1917` | `improving` wrongly `true` when ply-2 was in check (`static_evals[ply_u-2] = -INFINITY`, so `static_eval > -INF` trivially true). Feeds RFP, LMP, futility, LMR. SF falls back to ply-4. |
| `search.rs:1953-1966, 2008-2013` | Stale `info.reductions[ply_u]` leaks into NMP child's hindsight reduction (reductions is written in the move loop but never reset at node entry or before NMP recursion). |
| `search.rs:2995-3017` | QS TT cutoff has no halfmove guard; main-search does (`<90`). |
| `search.rs:1855-1870` | TT near-miss accepts entries 1 ply short with no halfmove guard. |
| `search.rs:1774-1789` | TT-cutoff cont-hist malus reads `board.piece_at(our_to)` — for promotions this is the promoted piece; the write-side via `moved_piece_stack` stores the pre-promotion pawn. Read/write key asymmetry on promotions. |
| `search.rs:2079` | ProbCut "TT says no chance" skip uses unadjusted `tt_entry.score` — inconsistent with all other mate-aware uses. Also missing a TT-bound-type gate (`!= TT_FLAG_LOWER`); a LOWER bound below `probcut_beta` doesn't mean "no chance". |
| `search.rs:2076-2082` | ProbCut has no `!is_pv` guard. Consensus (SF/Obsidian/Viri/Berserk) gate to non-PV only. |
| `search.rs:2396-2406` | LMP has no `!is_pv` guard despite CLAUDE.md saying non-PV only. Either doc or code is stale. |
| `search.rs:2981-2986` | QS cuckoo missing `ply > 0` guard; main-search cuckoo has it (line 1715). |
| `search.rs:2364` | `pawn_hash % info.pawn_hist.len()` inside history-pruning block vs `& (PAWN_HIST_SIZE - 1)` everywhere else. Equivalent today but latent if PAWN_HIST_SIZE becomes non-power-of-2. |
| `search.rs:2879-2892` | `update_correction_history` runs without `info.stop` guard — TT-store has one. On stop, children returned 0 → polluted `best_score` may still pass `> alpha_orig` and poison per-thread corrhist for future iterations. |

### NNUE / accumulator

| Location | Issue |
|---|---|
| `nnue.rs:3653, 4028, 4038` | `AccEntry.computed: bool` (single) — the +21 Elo "PSQ refresh per-pov" optimisation (`[bool; 2]` per-side) appears lost in the later fuse refactor (commit `317deab`). Confirm from experiments log; if unintentional revert, re-apply for +21 Elo. |
| `nnue.rs:3362-3368, 3426-3430` | `[0i16; 768]` hardcoded stack buffers in `forward_with_threats` and `forward` non-pairwise paths. Any future v9 with accum > 768 panics on startup. |
| `threat_accum.rs:67, 83` | `ThreatEntry::values: [[i16; 768]; 2]` hardcoded. Same risk as above. |
| `nnue.rs:3371-3383` | `forward_with_threats` non-hidden pairwise path only sums STM — NTM half missing. Dead in production (v9 has `l1_size > 0`) but a regression net without hidden layers would silently halve the eval. |
| `nnue.rs:3656-3945` | Legacy threat pipeline (`threat_white/black` fields on `NNUEAccumulator`, `store_threat_deltas`, `recompute_threats_if_needed`) is unreachable. If future code routes through `forward()` directly, threats silently zero out via the `empty_threat` fallback at lines 3407-3415. Attractive-nuisance trap. |

### Threats

| Location | Issue |
|---|---|
| `threat_accum.rs:160-179` | Refresh indices buffer `[0usize; 256]` silently drops overflow with no `overflowed` flag; `entry.accurate[p]` is set `true` so subsequent incremental deltas compound on corrupted baseline. |
| `search.rs:1824, 2717, 2756` vs `movepicker.rs:375, 714` | **In-check history key asymmetry.** Beta-cutoff writes use `enemy_attacks` key; evasion picker reads with `threats = 0`. Different 4D slots → in-check history writes are invisible to in-check reads. Reckless/SF keep them symmetric. |
| `main.rs:743-751` | `coda eval` CLI builds an accumulator but calls `forward()` (not `forward_with_threats`) — for v9 threat nets, the displayed eval score ignores threat features entirely. User-facing wrong number. |

### TT / history / score handling

| Location | Issue |
|---|---|
| `tt.rs:431-440, 455-456` | Two `Relaxed` stores (`data` then `keys`) on aarch64 can reorder. XOR verification catches false hits (~1/2³²), so correctness holds by construction, not by ordering. Use `Release`/`Acquire` for principled robustness. |
| `tb_cache.rs:112-113` | Same two-store pattern, same XOR-only defence. |
| `tt.rs:438` | Same-gen same-key replacement allows `depth > slot_depth - 3`, so a depth-18 UPPER can evict a depth-20 EXACT. Tuning choice; verify intent. |

### Move ordering / move picker

| Location | Issue |
|---|---|
| `movepicker.rs:528-533` | Bad-capture overflow silently drops moves past the 64-slot `bad_moves` buffer. Pathological tactical positions (two queens + rooks) can exceed. Enlarge to 256. |
| `movepicker.rs:524-539` | Non-capture promotions scored with `mvv_lva = 0` + `capt_hist[empty]`. Queen promotion ranks below a NxP + history capture in good-captures list. |
| `movepicker.rs:700-710` | Evasion-mode: capture-promotion matched by `is_promotion` first and scored 9000, LOWER than a plain capture (10000+...). Should score above both. |
| `movepicker.rs:441-454` | CounterMove stage dedup against `killers[0]/[1]` uses pre-fixup flag on counter_move; killers have been fixup'd. Stage is currently dead code (GoodCaptures → GenerateQuiets direct), so this is latent. |

### Time management / UCI / ponder

| Location | Issue |
|---|---|
| `search.rs:1241-1250` | `info.soft_limit = soft` assigned BEFORE the `if soft > hard { soft = hard }` clamp. Local `soft` is corrected; `info.soft_limit` is not. In movestogo=1 regime can yield `info.soft_limit > info.hard_limit`. |
| `search.rs:1555-1563` vs `search.rs:1025` | Stockpile-prevention sleep inside `search()` runs before `search_smp()` sets the shared stop. Helpers burn CPU for the entire sleep window (tens–hundreds of ms at blitz+inc). Skews SPSA/OB fair-time accounting. |
| `uci.rs:673-730, uci.rs:83` | `Ponder` check-option has no handler in `parse_option`. `setoption name Ponder value true` falls through silently. Protocol hygiene. |
| `uci.rs:149` vs `uci.rs:352` | On `go` with TB-valid: override move even for `wdl=0` draws. On `ponderhit`: override only for `wdl != 0`. Inconsistency — drawn TB endgame on ponderhit plays the NNUE move. In 50mr-adjacent positions that move can be 50mr-losing. |
| `uci.rs:599-603` | `parse_go` parses `depth` with `unwrap_or(100)`. A malformed `depth ???` becomes virtually infinite. Other integer fields use `unwrap_or(0)` (fail-closed). |
| `uci.rs:391-417` | `go ponder movetime X` (no wtime/btime) → on ponderhit, `our_time == 0` → instant stop. Discards ponder work. |
| `uci.rs:150-157` | TB probe at `go` reports `score cp ±TB_WIN` with `depth 1`. Some GUIs misinterpret extreme cp scores; SF uses `score mate N` when DTZ is known. |

### SMP / helper threads

| Location | Issue |
|---|---|
| `search.rs:938-963` | `helper.ponderhit_time` is NOT cloned from `main.ponderhit_time` — every other coordination atomic is. Helpers' `ponderhit_time` stays 0 forever, so they ignore ponderhit deadlines and only stop when main sets the shared stop. CPU waste during grace window. |
| `search.rs:411-422` | `load_nnue` activates `threat_stack` when `net.has_threats` but doesn't reset it to false otherwise. Switching v9 → v5 via `NNUEFile setoption` leaves `threat_stack.active = true` while `net.has_threats = false`. Main and helper can disagree on `active` after a net swap. |
| `search.rs:981-1022` | Helpers spawn before `info.tt.new_search()` (which bumps generation). In the µs race window helpers write TT entries with the old gen. Not correctness, but replacement picks them first later. |

### SEE / Zobrist / cuckoo / TB

| Location | Issue |
|---|---|
| `search.rs:1670-1682, 2944-2952` | Repetition detection uses `halfmove` only; ignores `plies_from_null`. `halfmove` increments on null moves (board.rs:947); scan walks back across nulls. Cuckoo at `cuckoo.rs:114` already uses the `min` — search/QS rep should match. |
| `see.rs:95-102` | SEE king-capture gate: both branches `break` with the same state and produce the same `side_to_move != stm` outcome. Dormant (KING value 20000 dominates balance), but if a huge-promotion gain pushes balance `>= 0` a king recapture returns the wrong side. SF flips the return. |
| `see.rs:76-93` | SEE inner loop doesn't handle pawn-promotion recaptures. A pawn reaching `to` on rank 1/8 during exchange is valued at PAWN=100, not Q=900. Rare but wrong. |
| `tb.rs:118-126` | DTZ cursed-win / blessed-loss collapsed into `±20000` at root — drawn positions report "mate-ish" scores. Inconsistent with `ambiguous_wdl_to_score` (same file) which scores them as ±1. |
| `tb.rs:57-77` | Interior TB probe comment says "Only valid when halfmove clock is 0"; probe-site at `search.rs:1688-1707` doesn't enforce. Combined with C2 (cache drops halfmove), results stored from halfmove>0 probes pollute halfmove=0 queries. |
| `tb.rs:54-57` (+ `search.rs:1688`) | No castling-rights gate. Syzygy rejects positions with castling → probe wastes FEN formatting + `Chess::from_setup`. Add `if board.castling != 0 { return None; }` at probe entry. |

### Training pipeline / net I/O

| Location | Issue |
|---|---|
| `bullet_convert.rs:170-183` | v6 writer emits extended_kb header bytes (bit 7 + kb_count + kb_layout) that the v6 reader at `nnue.rs:2126-2140` never parses. Hidden-size inference hardcodes `NNUE_INPUT_SIZE = 12288`. `convert-bullet --kb-layout reckless` produces a file the reader mis-parses. |
| `bullet_convert.rs:165-168` | Version 5 has no flags byte, so no way to encode non-16 king-bucket layouts. `convert_v5 --kb-layout reckless` writes `10*768*h` i16s the reader treats as `16*768*h`. Silent corruption. |
| `nnue_export.rs:64` | `l1w = vec![0.0f32; 2 * h * l1_size]` hardcoded for non-pairwise. For pairwise v7 (768pw → L1=16) Bullet expects `h × l1_size`. Transfer-learning checkpoint misaligns. |
| `datagen.rs:384-405` | Material-removal datagen: `remove_piece` doesn't touch `board.castling` or `board.ep_square`. Remove a rook from A1/H1/A8/H8 → castling bit stays set with no rook. EP square survives pawn removal. `is_valid_position` only checks both kings exist — doesn't verify opponent-not-in-check (illegal "you moved into check" positions possible). |
| `main.rs:707-720` | `fetch-net` uses `curl -sL` without `-f`. HTTP errors (404/500) captured as file content; curl exits 0. Partial downloads become "successful" nets. No post-download NNUE magic validation. |
| `bullet_convert.rs:292-293` | Threat weight clamp uses `[-127, 127]` instead of full `[-128, 127]` i8 range. More importantly, silent clipping below the 1% warning threshold goes unreported. |

### Bullet trainer (feature/threat-inputs branch)

| Location | Issue |
|---|---|
| `bullet/crates/bullet_lib/src/game/inputs/chess_threats.rs:455-458` | `--xray 0` training produces a net with `num_inputs() = threat_offset + NUM_THREAT_FEATURES` — x-ray rows exist in the weight matrix with **random init** (never gradient-updated), but Coda inference always emits x-ray features. Random weights activate at inference → silent eval corruption. No matching `--xray` flag on inference side. |
| `bullet/examples/coda_v9_768_threats.rs:219-234` | `--l1-decay` + `--factoriser` interaction: L1 drives `l0w[i]` → 0, but saved value is `l0w[i] + l0f[i % l0f_len]`, which is non-zero wherever factoriser has learned shared weights. Defeats L1's stated sparsity goal. Either train with one or the other, or document incompatibility. |
| `bullet/crates/bullet_lib/src/game/inputs/chess_threats.rs:17` | `MAX_THREAT_ACTIVE = 256` + 32 PSQ = 288 upper bound. Worst-case position with many slider/xray features could exceed. Panics the loader mid-training run. Quiet-position filter reduces probability but doesn't eliminate. Bump to 384–512 or run the data through a max-count probe. |
| `bullet/examples/coda_v9_768_threats.rs:298-305` + `coda/src/main.rs:266` | `--hidden-activation crelu` at training and `--hl-crelu` at converter are independent flags. If they disagree, .nnue claims wrong activation; inference applies wrong activation silently. Human-process footgun. Fix: emit `training.meta` sidecar, have `convert-bullet` read it automatically. |

## SPECULATIVE (latent / dead-code / defensive)

Brief list; fix when touching the area.

- `threats.rs:456` — `1u64 << (blocker_sq + 1)` undefined at `blocker_sq == 63`. Currently guarded upstream by `revealed == 0` but fragile.
- `threats.rs:892-1224` — dead `compute_move_deltas` function duplicates delta pipeline; any drift goes unnoticed.
- `threat_accum.rs:122-132` — `ThreatStack::push(mv, moved_pt)` args always `NO_MOVE, NO_PIECE_TYPE`; absorbed post-make. New caller forgetting the absorb is silent corruption.
- `search.rs` threat_accum absorb — stores `moved_pt` from post-move mailbox, so promotions record QUEEN not PAWN. Currently only consumed by `== KING` check, safe today.
- `tt.rs:489-512` — combine with C5 fix.
- `cuckoo.rs:170-174` — root-boundary STM check picks first non-empty square; if both squares have pieces of the same colour, heuristic can misclassify. Knight-dance passes.
- `cuckoo.rs:216-245` — `cuckoo_entries_valid_moves` test accepts ANY piece matching the XOR key; no unique-piece assertion per slot.
- `see.rs:46-53` — SEE initial `risk_value -=` subtraction is strictly wrong when opponent has no defender; monotonicity of `see_ge` threshold still gives correct answer. Fuzzer covers it.
- `search.rs:2647` — LMR `do_shallower` uses `new_depth` (integer, ~5–20) as a centipawn threshold. Almost certainly accidental; SF uses a proper cp margin.
- `search.rs:2292-2294` — SE `singular_beta = tt_score - depth - kzp*SE_KING_PRESSURE_MARGIN - xray_bonus` can drop below mate-bypass range. Downstream comparisons still work, but multi-cut return value interpretation could misfire.
- `search.rs:2441-2449` — recapture extension has no `ply > 0` guard; at root it can fire on game-history captures.
- `search.rs:1619-1626` — `any_threat_count` excludes king from `our_non_pawns`. Deliberate? Verify intent vs SF pattern.
- `search.rs:2894-2899` — FH blending applies inside SE verification; dampened score in `singular_score` may flip DEXT to single-ext in edges.
- `search.rs:500-527` — `should_stop` time-check bucket is `& 4095 == 0` (~4–7 ms). For <100 ms emergency budgets this causes overruns; SF checks every 1024–2048.
- `search.rs:1273-1278` — forced-move path zeros `soft_floor`, defeating stockpile prevention on ponderhit fresh-search with 1 legal move.
- `search.rs:938-963` — `helper.root_stm = main.root_stm` is read before main sets it in `search()`. Currently unused; trap for future stm-aware helper code.
- `search.rs:1013` — helper `max_depth` doesn't map 0 → MAX_PLY/2 the way main does. `go depth 0` makes helpers empty-loop.
- `tt.rs:431-457` vs `tb_cache.rs:101-114` — aarch64 Relaxed-store ordering relies entirely on XOR verification.
- `see.rs` — duplicate SEE functions in movepicker.rs and see.rs? (mentioned in CLAUDE.md as "risks correctness bugs" — verify single source of truth.)
- `types.rs:103-110` — `is_promotion` + `promotion_piece_type` returns garbage for `flags >= 8`. Guarded by `is_pseudo_legal` today.
- `movepicker.rs:856-861` — `fixup_move_flags` adds FLAG_CASTLE for any king-move-by-2. Rejected by `is_pseudo_legal`'s castling branch, but brittle.
- `movepicker.rs:956-960` — promotion validation checks from-rank but not to-rank. Closed in practice by pawn-geometry constraints.
- `bullet_convert.rs:153-159` — output-bucket remap drops buckets when `ob > 8`. Not used today.
- `nnue.rs:690-705` — SCReLU scale uses `>>8` (÷256) instead of `÷255`. ~0.8% drift, absorbed into SPSA tune. New training recipes that shift SCReLU expectations could surface this.
- `nnue.rs:1289-1294, 755-763` — SIMD pack tails (AVX2/AVX-512 half-chunk) have no scalar round-trip test. Production `h` sizes are multiples of 64 so the tail path is dead.
- `main.rs:460-508` — `patch-net` no NNUE magic validation before in-place modify.
- `binpack.rs` — Coda's (unused) BINP writer has latent pawn-colour nibble collision potential; dead code, delete or assert.
- `main.rs:898-945` — `sample-positions` filter doesn't match CLAUDE.md's documented training quiet-filter (ply ≥16, no checks, no captures, no tactical).
- `search.rs:1649-1656, 2971-2978` — redundant `stop.load` after `should_stop` in negamax. If intentional "check every node", it's correct but costly; if not, remove.

## Confirmed clean (noted for future reference)

- `is_pseudo_legal` non-EP paths (pawn direction, double-push intermediate, castling path/attack/rights).
- Zobrist EP conditional (all 4 sites: compute_hash, make_move remove+add, make_null_move). Post `+3.3 Elo` fix.
- Castling rights XOR symmetry and mask coverage.
- TT XOR-key verification on every probe.
- Helper thread threat-state init (post `+651 Elo` fix).
- King-bucket `NNUENet` field (post `+3.4 Elo` static-mut race fix).
- Core x-ray threat math at `threats.rs` sections 2 and 2b (post `+110 Elo` rewrite).
- Corrhist update guards: excluded, noisy-move, mate-score, alpha-orig (with the LIKELY exception of missing stop-flag guard, listed above).
- SE recursion excluded-move propagation (blocks recursive SE correctly).
- NMP mate guards (`beta.abs() < MATE_SCORE - 100`).
- Mate-distance pruning (non-PV, ply+1 offset).
- Hindsight reduction sign (post sign-flip fix).
- FinnyTable piece_bbs staleness (always updated after delta apply; fits `[usize; 32]` with 32 pieces max).
- King-bucket crossing detection returns `DirtyPiece::recompute()` correctly on bucket or mirror change.
- ThreatStack `can_update`/`update` mirror-crossing rejection for moving pov.
- AVX-512 pairwise pack threats (post fuse fix).
- Perft on 6 standard positions (fuzzed).
- Main-search cuckoo `ply > 0` guard.
- Cuckoo table population (3668 entries).
- Move flag `==` vs `&` — all call sites use `==` correctly.
- Duplicate-SEE risk — single implementation at `see.rs`; movepicker calls through it.

## Recommended fix order

Fix in batches. Each batch is independently SPRT-able and small enough for one review branch.

**Batch 0 — investigate first, potentially highest-impact:**
0. **C8 Bullet training/inference semi-exclusion frame mismatch.** Build the map_features vs enumerate_threats fuzzer BEFORE deciding fix path. If confirmed, this likely requires a one-time retrain of v9. Budget: investigation days + one retrain cycle. Do not queue anything dependent on net behaviour until this is resolved.

**Batch 1 — CRITICAL, straightforward, no retune needed:**
1. C1 `is_pseudo_legal` EP checks (+ fuzzer update) → `fix/is-pseudo-legal-ep`
2. C2 TB cache halfmove key → `fix/tb-cache-halfmove`
3. C5 TB_WIN ply-adjust threshold → `fix/tt-tb-win-threshold`
4. C6 Stale soft_limit/hard_limit across `go` → `fix/tm-stale-limits`
5. C7 convert-checkpoint output-layer momentum → `fix/convert-checkpoint-l2`
6. C4 NEON SCReLU `>>9` vs `>>8` (+ scalar-vs-NEON test) → `fix/nnue-neon-screlu-shift`
7. C9 HIP Adam ABI (only if we ever run HIP training; low-priority vs the CUDA-only production path)

**Batch 2 — CRITICAL, may affect Elo / tree shape:**
8. C3 NMP `moved_piece_stack` sentinel → `fix/nmp-sentinel-moved-piece` (expect small Elo either way; retune trigger)

**Batch 3 — LIKELY, clusters of low-variance fixes:**
8. `is_pv` SE shadow, `improving` in-check fallback, stale `reductions`, QS TT halfmove, TT near-miss halfmove, TT-cutoff cont-hist malus promotion asymmetry — all narrow one-liners, bundle per subsystem.
9. In-check history key asymmetry — make evasion reads match writes (or vice versa).
10. Refresh-indices overflow flag.
11. `helper.ponderhit_time` clone, `load_nnue` threat_stack reset, corrhist stop-flag guard.
12. Repetition scan + cuckoo plies_from_null.
13. SEE pawn-promotion in inner loop + king-capture gate cleanup.
14. DTZ cursed-win/blessed-loss consistency.

**Batch 4 — training pipeline / I/O hygiene (separate from engine):**
15. convert-checkpoint L1W pairwise, v5/v6 kb-layout guards, fetch-net `-f` + magic, material-removal datagen state cleanup.

**Batch 5 — SPECULATIVE / hygiene:**
16. Dead-code removal, tighter defensive guards, the long list of minor items.

## Bullet audit — confirmed clean

The Bullet audit independently verified these are consistent with Coda inference:
- Mirror/flip math (`ntm_flip = 56 ^ optional_file_flip`) matches bulletformat's storage convention.
- Empty-board pawn/knight attack tables (unit tests pass at expected totals: 84, 336).
- Slider ray attacks vs naive ray-walk — match across occupancy patterns.
- `base_pair` encoding: training's `pt*2+color` and inference's `color*6+pt` are pov-remap-equivalent (same iteration order).
- X-ray ray-bitscan optimisation (ebdf398) matches old attack-recomputation path.
- Threat hot-path optimisation (ceef197 +45%) is semantically equivalent.
- CUDA AdamW + L1 argument order matches kernel declaration; CPU AdamW path uses standard proximal soft-thresholding.
- Sign-at-zero: L1 kernel `else { p=0 }` produces exactly 0, not NaN.
- Factoriser init: `InitSettings::Zeroed` for l0f; l0w gets normal init before rebinding.
- Factoriser shape math and gradient flow through `.repeat(ones^T)` are correct.
- `l0f` correctly excluded from `l1_decay` (no zero-collapse).
- Reckless-10-bucket flat expansion has a startup assertion against the Reckless reference.

## Methodology notes

- Several bugs were independently flagged by two different agents (C2, C5). Those get automatic high confidence.
- "Hasn't been caught by SPRT" is not evidence of correctness. SPRT's signal-to-noise floor around ±2 Elo is fine for structural features but low for rare-case bugs that trigger in <5% of games.
- The Lichess watch methodology (CG5ZXe5Z endgame gate fix) and bench-reveal methodology are qualitatively better at finding this class of bug. The #619 Zobrist EP fix was similarly surfaced by "weird score on what should be a forced draw".
- Fuzzers caught the existing `is_pseudo_legal` holes but miss the multi-field-corruption class (C1). Worth extending fuzzers to mutate pairs/triples of fields when EP is active.
