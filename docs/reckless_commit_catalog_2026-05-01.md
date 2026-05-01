# Reckless NPS Commit Catalog (Phase 3 of NPS leg plan)

Scan range: Reckless `45d9cc5a` (#728) → HEAD `286ae5b8`. Built 2026-05-01.

Companion to `/home/adam/code/coda/docs/coda_vs_reckless_nps_2026-04-23.md`
and `/home/adam/code/coda/docs/v9_nps_findings_2026-04-18.md`.

## Method

Walked 199 Reckless commits from `45d9cc5a` (Threat Input Incremental
Update for AVX2, the first commit on the v9-equivalent threat-input
architecture) up to `286ae5b8`. Filtered titles for NPS-relevant
keywords (speedup, perf, vectoriz/AVX/SIMD/VNNI, prefetch, setwise,
movegen, accumulator, threat, NUMA), then read commit bodies for the
~50 candidates. Excluded pure correctness fixes, refactors with no
Elo/NPS claim, search-feature additions (different work axis),
network-only releases, and CI/clippy churn.

For each NPS-relevant commit, classified against Coda by file-grep
plus `git log --grep` on the Coda repo. Status breakdown is at the
end. **Coda v9 architecture** (768-wide accum, threat features, h16x32
SCReLU L1+L2) is the reference for "incompatible" calls.

The dominant NPS themes in this Reckless window are: (a) AVX2/AVX-512
SIMD ports of new SIMD threat-update kernels, (b) compile-time
target-feature dispatch via cfg-gated SIMD modules, (c) AVX2/AVX-512
*setwise* movegen replacing per-square magic lookups, (d) prefetch /
TT cacheline tweaks, and (e) NUMA-aware NNUE weight replication.

## Catalog

Sort: untried (priority order) first, then partially ported, then
already ported, then dropped/incompatible.

| Reckless | PR | Title | Their Elo | Mechanism | Coda status | Effort | Priority |
|---|---|---|---|---|---|---|---|
| `b974fbe8` | #909 | Set-wise knight/bishop/rook attack generation | +1.64 STC (AVX-512); locally 1.03× faster | New `src/setwise.rs`. Generates attacks for *multiple pieces of one type* in a single SIMD operation, replacing per-square magic lookups. Used in board.rs threat enumeration and movepick. | **Implemented** (independent reimpl, Coda merge `d865591` 2026-05-01). New `src/setwise.rs` with scalar + AVX2 paths; integrated in `attacks_by_color`. SPRT #900 aggregate +0.4 ±1.6 / 33K games (won't H1); Zeus AVX-512 alone +2.1 / 4096g. Merged as forward-looking infra. | — | done |
| `146cfe47` | #914 | Set-wise attacks (AVX2) | +3.01 STC | AVX2 port of #909. Larger STC margin than the AVX-512 case. | **Implemented** with #909 above (single Coda merge covers both — runtime dispatch, single binary). | — | done |
| `66cd450f` | #793 | Speed up threat accumulator refresh | +1.55 STC | Register-tile the threat-refresh path; 2-feature unroll. `src/nnue/accumulator.rs` 64 lines. | **Untried — pending verification.** Our `recompute_threats_full` path (`nnue.rs:3969`) is scalar; `add_weight_rows_avx512`/`avx2` exist for the apply path but not for the refresh-from-scratch loop. | Half day | **HIGH** |
| `04c88767` | #792 | Vectorize PST feature index generation on accum refresh (AVX-512 VBMI2) | +2.70 STC | Generate the 64 feature indices for the king-bucket+piece-type fan in one VBMI2 shuffle pass instead of per-square scalar. | **Untried — pending verification.** Coda's `enumerate_features_for_perspective` (PSQ Finny refresh) is scalar. AVX-512 VBMI2 (`_mm512_permutexvar_epi8`) needed. | Half day, AVX-512 VBMI2 only | **MED-HIGH** |
| `319dd161` | #753 | Prefetch the TT entry before NMP search | +3.33 VSTC | One added prefetch line in `search.rs` before the NMP recursive call. Trivial 1-LoC. | **Already ported** (verified 2026-05-01). `search.rs:2346-2347`: `board.make_null_move(); info.tt.prefetch(board.hash);` — identical to Reckless's diff. | — | done |
| `fa8f25a5` | #800 | Prefetch TT entries into all levels of cache | +1.09 STC (173k games) | Reckless went `_MM_HINT_T1` → `_MM_HINT_T0`. Catalog initially read this as Coda needing change; misread. | **Already at parity** (verified 2026-05-01). Coda already uses `_MM_HINT_T0` in `tt.rs:487`. The change brought Reckless TO our state, not vice versa. | — | done |
| `2889b28f` | #927 | Overhaul NUMA + NNUE weight replication | +1.59 STC 1th non-reg (real win is at SMP) | Per-NUMA-domain replication of NNUE weights, libnuma replacement. Stockfish-style NUMA. | **Untried — major work.** Coda has zero NUMA awareness (only 2 grep hits in `threats.rs`/`threat_accum.rs` and they're not NUMA-related). Only matters at multi-socket SMP — irrelevant for single-thread bench, relevant for OB SMP and bot deployment on multi-socket hosts. | 4-7 days, Stockfish reference port | **LOW** (no current multi-socket) |
| `11c2edcd` | #859 | Improve ARM NEON, add DOTPROD, harmonize | "12% speedup" overall on ARM | NEON `vqdmulhq_s16` swap, asm-block DOTPROD path, horizontal-reduction NNZ bitmask. ARM-only. | **Untried.** Coda has NEON `add_weight_rows_neon` but no DOTPROD asm path and no NEON-side NNZ horizontal-reduction. Only matters on ARM hosts (lichess bot if redeployed, Apple Silicon dev). | 1 day, ARM only | **LOW** |
| `45d9cc5a` | #728 | Threat Input Incremental Update for AVX2 | +2.27 VSTC AVX2 | Restructured threat-feature enumeration into `nnue/threats/{scalar,vectorized/{avx2,avx512}}.rs`. Compile-time SIMD dispatch via cfg modules. | **Already ported (architecturally).** Coda v9 has the threat-input architecture (`src/threats.rs`) and incremental updates (`src/threat_accum.rs`). Coda dispatch is **runtime-flag-based**, not cfg-module-based — see `coda_vs_reckless_nps_2026-04-23.md` §"runtime SIMD dispatch branches". | — | (informational) |
| `d09af1b8` | #784 | Speedup dpbusd with VNNI | +3.38 VSTC | Replace `_mm512_madd_epi16(_mm512_maddubs_epi16(...), ones)` triple with a single `_mm512_dpbusd_epi32`. | **Already ported.** Coda has the VNNI path on `feature/nnue-avx512-vnni-v9` (current branch), production. | — | done |
| `f7863555` | #920 | Add feature detection for AVX512_VNNI | (no Elo, gating fix) | Runtime gate so VNNI binary doesn't ICE on non-VNNI AVX-512 hardware. | **Already ported** — Coda has runtime `has_avx512_vnni` detection in `nnue.rs`. | — | done |
| `c505c44f` | #850 | Fix build on non-ICL AVX512 | (no Elo, build fix) | Strict cfg-target-feature gating so non-Ice-Lake AVX-512 builds don't pull in VBMI2. | **Already conformant.** Coda's runtime dispatch sidesteps the issue. | — | done |
| `1f2f0720` | #825 | Correct target_feature flags | (no Elo, build fix) | Tighten VBMI2 / AVX512VL flag handling. | **Already conformant.** | — | done |
| `4ce46a83` | #790 | Switch to fully legal movegen | +1.48 VSTC | Drop pseudo-legal generation; legal-only at gen time. Removes pseudo-legality checks from `is_legal` callers. | **Architecturally incompatible (cost-benefit).** Coda v9 already generates fully legal evasions when in check (`src/movegen.rs:4`) and uses pseudo-legal + `is_pseudo_legal`-validated path otherwise. Switching to fully-legal everywhere is a multi-week rewrite of move encoding/pinning machinery for +1.48 STC; not worth right now. The structural issues that cost us 320 Elo previously (CLAUDE.md key gotcha) live in `is_pseudo_legal`, which we'd be deleting. | (multi-week) | LOW |
| `587f6354` | #788 | En passant legality | +2.16 STC non-reg | Only set `en_passant` square when EP is actually legal — guards repetition detection. | **Already ported.** Coda has the `ep_capture_available` Zobrist guard at `src/board.rs:96`. | — | done |
| `bc7981f0` | #778 | Revert previous EP legality logic | (revert) | Reverted #767-era EP change. | (predecessor reverted, see #788) | — | — |
| `350fc251` | #735 | Cache checking squares + 10k movepicker bonus for direct checks | +5.09 LTC | Cache the per-square checking-piece bitboards on the board, used both by movegen and by movepicker bonus for direct-check moves. | **Untried — pending ablate.** Coda has no `checking_squares` cache; `gives_check` is computed on demand at `search.rs:2840` as `board.in_check()` *post-move*. Cache would let us score direct-check moves *pre-move* in movepicker — feeds into LMP direct-check carve-out (which we tried at `8199f17` and reverted at `e9011b8`). The carve-out failure may have been because we couldn't classify direct-check moves cheaply. **Worth retrying with the cache**. | 1-2 days | **MED** |
| `77720632` | #826 | Reorder accumulator update operations | +1.81 VSTC | Reorder `psq.rs`/`threats.rs` add/sub instructions to improve LLVM register allocation. ~12 lines changed. | **Untried — pending verification.** Cheap targeted look at `simd_acc_fused_avx2/avx512` and `apply_deltas_avx*` — see if Reckless's reorder pattern (likely "all subs before any adds" or similar) produces better codegen on Coda's helpers. | Half day | **MED** |
| `12d4beba` | #853 | Store intermediate search results into TT | failed STC -3.61, passed VSTC SMP +3.99 | TT-stores intermediate aspirate-loop results in addition to final root result. SMP-only win. | **Untried, dubious.** Failed at 1-thread STC; only gains at 8-thread SMP. Low priority for Coda's primary single-thread NPS lever. | (skip) | LOW |
| `045d4d52` | #928 | Refactor board passing in thread pool | (no Elo, NUMA-locality nudge) | Allocate `Board` per-thread to improve NUMA locality. | **Architecturally incompatible** without prior NUMA infrastructure (#927). | — | (with #927 only) |
| `e3961057` | #946 | Simplify pawn attacks setwise | -0.04 STC non-reg | Followup to #909/#914 setwise infrastructure. | (Bundle with #909/#914 port) | — | with setwise port |
| `d608b446` | #923 | Simplify some threat generation | +1.38 STC | Touches `board.rs` threat generation 4-line simplification. Code-readability claim, not perf. | **Untried — likely irrelevant** to Coda since our threat enumeration lives in `src/threats.rs`, separate code shape. | (skip) | LOW |
| `65d05229` | #894 | Consolidate some threats code | (no Elo, refactor) | Threat generation refactor in `board.rs`. | (Skip — refactor) | — | — |
| `6174933d` | #874 | is_legal perft generator | (tooling) | Perft tooling for is_legal validation. | (Skip — tooling) | — | — |
| `b974fbe8` setwise + `146cfe47` AVX2 setwise | #909+#914 | (see above) | Combined +4.65 STC | New SIMD movegen pillar. | (See HIGH-priority entries above) | | |
| `dae74990` | #770 | Use Vec instead of ArrayVec for board stack | +0.38 VSTC non-reg | `Vec<Board>` replaces `ArrayVec<Board, MAX_PLY>`. Heap once instead of large-on-stack. | **Architecturally different.** Coda uses `[BoardState; MAX_PLY]` style. Effort vs. negligible Elo — skip. | — | LOW |
| `f08c91b1` | #877 | Replace unchecked access with safe indexing | +1.00 STC non-reg | Code hygiene; `get_unchecked` → safe indexing. | (Skip — claim is bench-equiv) | — | — |
| `83855ed9` | #879 | Simplify and speedup is_legal | +1.25 STC non-reg | Tightens `is_legal` paths after #790 fully-legal switch. | (Conditional on #790 port) | — | — |

## Top untried port candidates (ranked)

EV ranking = (their Elo signal) × (Coda applicability) ÷ (effort). Top candidates we should ablate before porting:

1. ~~**Setwise movegen (`b974fbe8` + `146cfe47`, #909+#914)**~~ — **DONE
   2026-05-01.** Independent reimplementation in `src/setwise.rs`,
   merged at `d865591` after SPRT #900 stopped at +0.4 ±1.6 Elo /
   33K games (Zeus AVX-512 alone +2.1 Elo / 4096g). Merged as
   forward-looking infrastructure for AVX-512-ubiquity future. See
   `experiments.md` 2026-05-01 entry for full write-up.

2. **Threat accumulator refresh pair-unroll (`66cd450f`, #793)** —
   **In flight (SPRT #903).** Implemented as
   `experiment/threat-refresh-pair-unroll`: 2-feature unroll on
   `add_weight_rows_avx2` + `add_weight_rows_avx512`. Trending H0
   (-0.3 ±2.0 / 22K games) at time of writing. Same low-signal regime
   as setwise; merge decision will follow same pattern (forward-
   looking AVX-512 infra if Hercules agrees, drop otherwise).

3. **PST feature-index vectorisation (`04c88767`, #792)** — +2.70 STC, AVX-512 VBMI2. Drops a scalar enum-features loop that runs once per Finny refresh fallback. Coda hits this on king-bucket transitions (45.65% of evals are full rebuilds, per microbench). Half day, gated on AVX-512 VBMI2 detection (already detected by `cpufeatures` crate; we'd add the gate).

4. **Cache checking squares + retry LMP direct-check (`350fc251`, #735)** — +5.09 LTC headline. We previously tried LMP direct-check carve-out and reverted at `e9011b8` — see Coda commit log. The Reckless approach pre-computes a `checking_squares` bitboard on the board so the predicate is O(1) at movepicker time. **Worth retrying our LMP carve-out with this cache** — our previous attempt may have been killed by the cost of computing direct-check membership per move. 1-2 days.

5. **TT prefetch level audit (`fa8f25a5` + `319dd161`, #800+#753)** — combined +4.42 across the two. Cheap (1-2 LoC each). Verify which prefetch hint Reckless settled on (`fa8f25a5` says "all levels"; that's typically `_MM_HINT_T2` for the deeper levels paired with T0 for the immediate). Add the missing prefetch before NMP descent if absent. 30-60 min combined.

6. **Accumulator update reorder (`77720632`, #826)** — +1.81 VSTC for a codegen reorder. Half day to compare Reckless's reordered psq.rs/threats.rs against Coda's `simd_acc_fused_avx2/avx512` helpers; may auto-replicate gain on our LLVM build.

7. **NUMA NNUE weight replication (`2889b28f`, #927)** — +1.59 STC 1th (the SMP gain isn't measured solo). Only matters on multi-socket hosts; the Coda lichess bot doesn't currently run multi-socket. Defer until/unless we deploy on EPYC dual-socket. 4-7 days when we do. **LOW priority right now.**

8. **NEON DOTPROD harmonisation (`11c2edcd`, #859)** — "12% speedup" on ARM only. Apple Silicon dev hosts and any future ARM bot deployment benefit. 1 day. Defer until ARM hosts are part of the perf story.

## Already ported / dropped / incompatible (summary)

**Already ported (status verified)**:
- `45d9cc5a` (#728 threat input architecture) — Coda has v9 threat features + `threat_accum.rs`
- `d09af1b8` (#784 VNNI dpbusd) — landed on `feature/nnue-avx512-vnni-v9`
- `dbb4ab7` (Coda 2026-05-01) — AVX-512 `apply_threat_deltas` + `add_weight_rows`, +4.4 Elo H1
- `587f6354` (#788 EP legality) — `ep_capture_available` Zobrist guard at `board.rs:96`
- `f7863555` (#920 VNNI feature detection) — Coda's `has_avx512_vnni` runtime flag
- `c505c44f` / `1f2f0720` (#850/#825 build flags) — Coda's runtime dispatch sidesteps

**Architecturally incompatible / not pursuing**:
- `4ce46a83` (#790 fully legal movegen) — multi-week rewrite for +1.48 Elo, sidesteps `is_pseudo_legal` which we audited carefully (see CLAUDE.md gotchas, +320 Elo cumulative on those bugs)
- `dae74990` (#770 Vec board stack) — Coda already has fixed-size stack
- `045d4d52` (#928 NUMA per-thread board alloc) — gated on #927

**Tested and dropped on Coda side**:
- LMP direct-check carve-out (Coda `8199f17` → reverted `e9011b8`). Retry candidate paired with `350fc251` cache (#735 above).
- Multi-ply PSQ walk-back (Coda SPRT #424, H0 -17.86 Elo). Discussed at length in `coda_vs_reckless_nps_2026-04-23.md` §"Walk-back refresh: previous attempt confounded — worth retry"; clean re-implementation is on the broader lever ranking.

**Coda has these and Reckless caught up**:
- AVX-512 `apply_threat_deltas` (Coda `dbb4ab7`, 2026-05-01) — Reckless equivalent is folded into their `nnue/threats/vectorized/avx512.rs` from #728 onward. Both engines now have parity here.

## Notes / themes

- **Reckless's NPS work in this window is dominated by setwise movegen + vectorised threat updates.** No silver-bullet gains; mostly +1.5 to +5 Elo each. The aggregate is significant — probably 15-25 STC Elo across the catalog if all ported, with diminishing returns. That's a meaningful chunk but won't close the 2× NPS gap on its own.

- **Compile-time SIMD dispatch is a recurring theme** in their NNUE module (`#728`/`#796`/`#920`/`#825`). Their `cfg(target_feature)` cfg-module approach gives LLVM full inlining freedom; our runtime `has_avx512_vnni` boolean check forces branches. Switching to compile-time dispatch is independently flagged in `coda_vs_reckless_nps_2026-04-23.md` §"Working hypotheses" #4 — modest expected gain (+2-4% NPS), but also required to match Reckless's codegen shape. Not a single commit; it's their architectural choice.

- **TT-side prefetch tuning is easy money.** Reckless landed two TT-prefetch tweaks (#800, #753) for combined +4.42 Elo with 1-2 LoC each. Cheapest cluster in the catalog.

- **Their movegen is moving toward setwise / batched.** Three commits (#909, #914, #946) build out `setwise.rs` over 6 weeks. Coda is still single-square magic-lookup throughout. This is the single biggest "structural" port gap and the highest-EV untried lever.

- **NUMA work is a 2026-Q2 theme for Reckless** (#927, #928 most recent). For Coda this is a parking-lot item — only matters when we deploy multi-socket. Note for completeness; not a Phase-3 priority.

- **Their threat-side has stabilised**: after the #728 architectural intro and the AVX-512 follow-ups (#793, #792), Reckless hasn't touched the threat-update SIMD kernels for ~6 weeks. That suggests we're close to parity once Phase-3 ports land.

- **Reckless does NOT have eval-only TT writeback**. We landed it as Coda #713 (+14.7 Elo). They didn't have a parallel commit in this window — confirmed by grep across the 199 commits. Their lower evals/node (0.520 vs Coda 0.677 per microbench) comes from somewhere else (likely the wider pruning landscape per #815, #818, #893).

- **Razoring** (`b2ad9338`, #893) — they have it, we don't. Listed in `coda_vs_reckless_nps_2026-04-23.md` pruning gaps as "+1.10 to +3.64 STC". Not in this NPS-focused catalog, but worth noting for the search-side leg.

---

*Catalog built 2026-05-01 by walking 199 Reckless commits, body-reading
~50 candidates, classifying against Coda's `src/`. Quality > exhaustiveness:
some ultra-marginal NPS-adjacent commits are batched into the "summary"
list rather than promoted to the main table.*
