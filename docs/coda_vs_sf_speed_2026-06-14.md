# Coda vs Stockfish per-node speed review (2026-06-14)

**Why:** SF is ~44% faster single-thread and ~80% faster under 16× contention
than Coda on identical hardware (Hercules). At Coda's measured ~130-140 STC
Elo/NPS-doubling, the contended gap is **~115 STC Elo — ~80% of the ~130-150
SF gap**. Critically, **SF runs the SAME architecture** (FT1024 + threats), so
this is pure implementation efficiency, fully recoverable without touching eval
quality. Mechanism: Coda runs higher IPC (1.52 vs 1.37) and less memory/node
(102 vs 115 B) yet lower NPS → **more instructions per node**.

**EMPIRICAL PROFILE (perf, single-thread, Hercules AVX2, `bench 14` / SF `bench`,
2026-06-14) — this supersedes the source-review guesses below.** SF is faster
on THIS AVX2 box, so the gap is NOT an AVX-512 story. Flat self-cycles:

| area | **Coda** | **Stockfish** |
|---|---|---|
| **Threat accumulator** | **~31%** (ThreatStack::update 23.4, push_threats 6.3, refresh 1.7) | **~5.5%** (update_piece_threats 2.8, FullThreats::apply 1.5, swap_piece 1.2) |
| NNUE FT + L1/L2/out | ~25% (fwd_l1_pairwise 12.3, simd_acc_fused 9.2, finny 1.8, corrected_eval 1.4) | folded into `evaluate` 35.7 (incl. its threats) |
| Search + movepick | ~20% (negamax 12.1, next_slow 6.1, pick_best 2.0) | ~27% (search 14, next_move 10, qsearch 2.7) |
| movegen/make/legal | ~5% (make_move 1.9, is_legal 1.7, gen_captures 1.1) | do_move 4.9 |

**THE GAP IS THE THREAT ACCUMULATOR: Coda ~31% vs SF ~5.5% (~25 points, ~most
of the 44% speed gap).** `threat_accum::ThreatStack::update` (23.4%) is Coda's
single hottest function. Annotation of `update` (apply_threat_deltas inlined):
~37% is the SIMD weight-apply (`vpmovsxbw` i8→i16 + streaming threat rows,
mod.rs:547), ~10% is the `threat_index` 48KB-`attack_index`-table chase
(threats.rs:509-510). SF maintains threats cheaply: builds the DirtyThreats
list incrementally inside `do_move` (`update_piece_threats`/`swap_piece`) and
applies once (`FullThreats::apply` 1.5%) — no replay, indices computed once.

**ROOT CAUSE — X-RAY THREATS (Coda does them, SF/Reckless do NOT).** Replay
depth is ~1.24 plies/update (4.4% refresh) → NOT a replay problem; cost is
per-node single-ply apply of ~10 deltas (profile-threats: apply avg 9.98
deltas/call, max 78). The threat-GENERATION breakdown (profile-threats):
- direct (step 1): 492 Mcy, 1.55 d/call, 28% zero-emit
- **own-xray (1b): 420 Mcy, 0.43 d/call, 72.8% zero-emit**
- sliders (2): 788 Mcy, 0.83 d/call, 50% zero-emit
- **slider-xray (2b): 434 Mcy, 0.37 d/call, 77.8% zero-emit**
- nonsliders (3): 268 Mcy, 0.56 d/call, 58% zero-emit

**The two X-ray steps are ~854 Mcy = ~36% of threat-generation cycles, 73-78%
WASTED (zero-emit)** — plus X-ray deltas inflate the ~10/node the apply streams.
SF/Reckless skip X-ray entirely → that's the bulk of why their threats cost
~5.5% vs Coda's ~31%. **The "+100 Elo X-ray" prior is SUSPECT: the +110 (H1
2026-04-17, project_xray_bug_fix) was a BUG FIX of broken X-ray, NOT a clean
"X-ray vs no-X-ray" A/B — that comparison may never have been run.** Revalidate:
(1) quick — ablate X-ray enumeration (env flag), measure NPS upside (profile
implies ~15-20% of total, ~30-50 STC Elo of speed at 135/doubling); (2) real —
RETRAIN a no-X-ray net (current net learned X-ray, so inference-disable = garbage
eval) + SPRT for the eval cost. If X-ray eval value < its NPS cost, dropping it
is a large net win AND closes most of the SF threat gap.

**Other threat target: make Coda's threat accumulator ~SF-cheap.** Candidates,
in order: (1) **threat-index precompute at delta-gen** (kill the ~10% table
chase, agent item E — was underranked); (2) **understand why Coda's apply is
~5x SF's** — likely Coda's replay-from-ancestor walks many plies vs SF's eager
per-`do_move` incremental, OR more active deltas/node; (3) the i8→i16 apply
cost. The L1-sparse and movegen-cache theories are BOTH refuted by this profile
(movegen ~5%, L1 a fraction of the 12% forward). This also re-confirms the
contention finding (threat_accum::update was #1 there too) — the threat
accumulator is Coda's central inefficiency vs SF, for both cycles AND bandwidth.

---
Comparative source review (read-only, 2 agents) — PARTLY SUPERSEDED by the
profile above (it over-weighted movegen/check-info, which is only ~5%).
SF source: `/home/adam/chess/engines/Stockfish/src`.

## Bit-identical wins (pure speed, no Elo risk — non-regression SPRT only)

### A. L1 fc_0 sparse-input — TESTED AND DEAD (dense is correct, 2026-06-14)
- **FINAL VERDICT (microbench, proper SF-style branch-free kernel): sparse LOSES at every density.** Built a vectorized `find_nnz` (cmpeq+movemask+LUT) → compact list → straight-line matmul (bit-identical to scalar, asserted). Result vs dense, PW=512: L1=16 @40% **1.79x slower**, @58% (real) **2.37x slower**; L1=32 @40% **1.83x**, @58% **1.95x**. The inline-branch `sparse_l1_avx2` was even worse (and is L1=16-only — panics at 32).
- **Why: Coda's L1 is too small (16-32 neurons).** list-sparse time is ~CONSTANT across density (~900ns L1=16 regardless of 40%/100%) → `find_nnz` *detection* dominates and the matmul *savings* are tiny (~144ns of skippable work at 40%). Detection cost > skip savings because each chunk's matmul is only 2-4 YMM ops. The dense decision was RIGHT all along (its 89% rationale was stale, but the conclusion holds at the real 58%).
- **Kills the sparsity-training lever for THIS target too:** `find_nnz` cost is density-INDEPENDENT, so more activation sparsity can't rescue it (list-sparse ~900ns at any density). Sparsity-training only helps if detection is ~free, which it isn't at this layer width.
- **Sliver (not worth chasing):** a maximally-optimized find_nnz (mine likely has 4KB-LUT cache misses; SF's is more tuned) *might* squeak L1=16/40% marginally — but it's marginal vs the much larger movegen/check-info wins. SF's sparse-input is likely near-neutral for SF too; its speed comes from the movegen/state-caching side.
- Method preserved: re-run via the `bench_l1_kernels` ignored test (set DENSITY sweep + add list kernel); reverted from main to keep it clean.

### (historical) the density re-measurement that motivated the test:
- **The dense-over-sparse decision was wrong-data.** Comment `nnue.rs:3332-3335` justified `DenseAvx2` with "pairwise-CReLU inputs have high density (~89%)." **Re-measured live (env `CODA_L1_DENSITY`, bench corpus): the input is ~58-60% nonzero → ~40% SKIPPABLE, not ~11%.** L1=16 prod 58.1% nonzero (41.9% skip); L1=32 multi-v6-l132-s5-swa 59.8% nonzero (40.2% skip). The 89% was stale.
- **SF is pairwise too (verified, NOT raw CReLU).** SF `transform()` (`nnue_feature_transformer.h:364-367`) does `vec_mulhi_16(sum0,sum1)` = pairwise multiply of the two accumulator halves (comment: "pairwise multiplication"); `sfnnv13_architecture_review_2026-05-23.md` is correct. So **pairwise does NOT preclude sparse** — SF runs pairwise + sparse-input + FT=1024 (fc_0 1024->31, ~= our L1=32) and wins. The earlier "SF wider" and "pairwise densifies → kills sparse" claims were BOTH wrong (inferred, not measured).
- **Why Coda's old test lost:** it skipped only ~11% (per the wrong 89%) AND Coda's `find_nnz_chunks4` is SCALAR (read u32 + branch per chunk) vs SF's vectorized cmpgt+movemask+256-LUT. At the real ~40% skip + a vectorized find_nnz, Coda is in SF's exact regime.
- **PATH (top speed candidate): vectorize `find_nnz` SF-style + select the sparse kernel + measure NPS.** Largest matmul, every node, ~40% skippable, at both L1=16 and L1=32. Coda already has `sparse_l1_avx2` (`sparse_l1.rs:134`) + the input-chunk-major layout; missing piece is the vectorized nnz pre-pass + selection. Bit-identical. Does NOT require dropping pairwise.
- **METHOD LESSON: three architecture claims this thread were wrong from inference (SF wider / SF not pairwise / pairwise kills sparse). Read source + measure; never infer arch.**
- **Lesson: agent source-reviews can miss explanatory history — verify each "gap" against in-code comments/git before implementing.**

### B. Incremental check/pin/check-square cache (SF StateInfo) — UNTESTED, lower prior now
- Coda recomputes `checkers()` + `pinned()` (each = 2 slider magics + leapers, `board.rs:664/692`) **from scratch every node** (`search.rs:3169-3170`, QS `4789`).
- SF's `set_check_info()` (`position.cpp:473`) computes `checkersBB`, `blockersForKing`, `pinners`, `checkSquares[all pt]` ONCE incrementally at the tail of `do_move`; movepicker/legal/gives_check read them as O(1) StateInfo loads.
- Fix: per-ply `CheckInfo` populated in/around `make_move`, threaded through the search stack. Bit-identical search. **Large, correctness-critical (320-Elo-class) refactor.** Was framed "biggest movegen-side win" — but C (below) measured NEUTRAL, and `checkers`/`pinned` don't surface as a standalone hotspot in perf (inlined, small). Prior on B is now LOW-value/HIGH-risk: don't attempt speculatively; profile that checkers/pinned is a real slice first.

### C. gives_check from cached check-squares — TESTED NEUTRAL (2026-06-15)
- Implemented: `Board::check_squares()` (per-node table) + `Board::gives_check_cached()`, lazily computed once per node, replacing the per-move `gives_direct_check` (post-move occ + magic) in the futility/LMP/bad-noisy carve-outs (`search.rs` ~3937/3955/3975). Branch `experiment/check-info-cache` (9f81d84).
- **Bit-identical** (bench 2325223; proven vs `gives_direct_check` by differential test `board::tests::gives_check_cached_matches` over a depth-3 perft tree — sliders/unblock/promo/EP/castle).
- **NPS-NEUTRAL** both single-thread (BASE ~777k vs ~775k) and 16× contended (3.437M vs 3.431M), clean Hercules. The per-move `gives_direct_check` was too small a slice. **Fourth scalar micro-opt to measure neutral** (after the zero-emit cull, the 48KB→6KB table shrink; Fix A regressed). Parked, do NOT SPRT/merge.
- **Pattern: small scalar micro-opts (~2-3% slices each) don't move NPS** — but this is NOT "the gap is eval-bound / unrecoverable" (that was an overreach; corrected by the hard data below).

### NPS DECOMPOSITION — hard data (2026-06-15, Adam pushed for it)
Gating build, x-ray on/off via same-net `CODA_NO_XRAY` isolation, clean
Hercules, vs SF `bench`:

| | Coda **+xray** | Coda **−xray** | Stockfish |
|---|---|---|---|
| single-thread | 793,748 | 881,829 (**+11.1%**) | ~1,140,000 |
| 16× contended (agg) | ~3,406,000 | ~3,892,000 (**+14.3%**) | ~6,589,000 |

- **x-ray NPS cost is REAL: ~11% single / ~14% contended.** The micro-opts were
  neutral because they shave loop-overhead/tables, not the emission+apply
  volume (only removing the feature does that).
- **x-ray is only ~25% of the single-thread gap, ~15% contended.** With x-ray
  removed, SF is STILL **+29% single / +69% contended** faster on the SAME
  FT1024+threats arch. **~75-85% of the SF gap is non-x-ray implementation
  headroom — recoverable** (re-confirms the top-of-doc "~80% recoverable" thesis).
- **Keep x-ray**: SPRT #2014 = +187 Elo for ~3-5 deployment Elo of NPS.
- **The real lever (UNTOUCHED): the threat-accumulator APPLY** — Coda ~31% of
  cycles vs SF ~5.5%, even with x-ray gone. SF builds the dirty-threat list once
  in `do_move`, applies once. The 4 neutral micro-opts did NOT touch it. First
  confirm cost = replay DEPTH (doc says ~1.24 plies, likely not) vs per-node
  apply VOLUME, then scope.

### D. Direct-write move generation (kill the 514-byte MoveList copies)
- Coda `generate_captures`/`generate_quiets` return `MoveList` (~514 B) BY VALUE (`movegen.rs:97/271`), then the picker re-pushes into `self.moves`. `generate_all_moves` (`movegen.rs:444`) copies caps+quiets into a 3rd list (every evasion/QS-in-check node).
- SF generates straight into the picker buffer; no by-value return, no re-push, dedicated `MoveList<EVASIONS>`.
- Fix: `&mut MoveList`/`&mut MovePicker` direct write. Localized, bit-identical, removes 1-2 full list copies per node.

### E. Threat feature-index precompute — TWO VARIANTS TESTED NEUTRAL (2026-06-15)
- `apply_threat_deltas` (`threats.rs:1644`) re-derives every delta's index via `threat_index` (`threats.rs:472`: color remap + `piece_pair` LUT + `pair.base()` + flip + `piece_offset` + a 48KB `attack_index[12][64][64]` chase) twice per ply (once per perspective).
- **The premise that "the 48KB table thrash is the cost" was wrong.** Two bit-identical micro-opts on the threat-index machinery measured NPS-NEUTRAL (clean Hercules, OB worker stopped, 16× contended aggregate, 4-5 rounds each, direction flips round-to-round = noise floor ~0.5%):
  - **48KB `attack_index` → 6KB `empty_attacks` + on-demand `popcount`** (`experiment/threat-index-compute`, commit `6f1e329`): base mean 3,439,591 vs 3,442,296 (+0.08%). An 8× table shrink moved nothing — a 48KB table is negligible against the multi-MB NNUE weight working set that dominates the contended L3 pressure.
  - **X-ray zero-emit cull** (gate the 1b own-x-ray loop, skips 72.8% of zero-emit calls; `experiment/xray-zeroemit-cull`, commit `23b7e05`): single-thread +0.3%, 16× contended +0.06%. Skipping the cheapest 73% of a cheap loop is Amdahl-invisible.
- **Conclusion (two angles — instruction count AND memory footprint — both neutral):** the threat *index/generation* subsystem (scalar code + small tables) is NOT the lever. The contended cost is **streaming the NNUE threat-row weights in the apply** (the ~37% `vpmovsxbw` weight-apply), whose volume is set by **how many threat features fire** — which x-ray inflates (see line 42 / §"WASTED" above). The threat-side speed lever and the −157 Elo x-ray finding are the SAME lever: the feature set. Pursue via the x-ray SB800 A/B, not index micro-opts.
- Both branches are bit-identical and harmless but pointless on NPS grounds — do not SPRT (measured-neutral; would burn fleet). Drop or leave parked.

### F. Fixed-array undo stack (replace Vec push/pop)
- Coda `undo_stack: Vec<UndoInfo>` push/pop per make/unmake (`board.rs:837/959`) — capacity check + `Option` unwrap on the absolute hottest path (pre-reserved, so no realloc).
- SF: caller-stack `StateInfo`, `memcpy` prefix + pointer swap; undo = single pointer load.
- Fix: `[UndoInfo; MAX_PLY]` indexed by ply → indexed store/load, no capacity check/Option. Bit-identical.

## Behavior-affecting (NPS-vs-Elo trade — needs full SPRT)

### G. Per-quiet-move scoring is much heavier than SF
- Coda `generate_and_score_quiets` (`movepicker.rs:557-773`): per quiet move computes a mobility delta (2 slider magics + popcounts), an offense bonus (`bishop/rook/queen_attacks` from `to` + inner SEE-target `while` loop), a knight-fork popcount, discovered-attack test, escape match — ~4-6 magics + loops PER QUIET MOVE.
- SF `score<QUIETS>` (`movepick.cpp:158`): ~7 history reads + one `check_squares & to` + one `see_ge`, with `threatByLesser[]` built ONCE per node (not per move).
- Sub-fixes (i) hoist per-node-constant attack sets out of the per-move loop (bit-identical) and (ii) gate the offense-bonus SEE-loop behind cheaper pre-tests. The bonuses are tuned-in Elo so deletion isn't free — isolate the hoists, SPRT the rest.

### H. L2/L3/output: drop the f32 pipeline for folded-scale integer VPDPBUSD
- Coda dequantizes L1 output to f32 and runs L2 + SCReLU + output in float (`nnue.rs:3482-3650`), width doubled by `dual_l1` (CReLU‖SCReLU = 32). Explicit `/qa_l1_f`, `/pw_scale`, clamps per neuron per eval.
- SF keeps every layer int8→int32 (VPDPBUSD) with all dequant folded into activation shifts (`sqr_clipped_relu.h` square+shift; `clipped_relu.h` `>>WeightScaleBits`). No float pipeline, no separate scale pass.
- Structural (needs int-quantized L2/L3 weights → requant/retrain). Standing per-node tax, lower priority than A.

## Already SF-class — no action
- PSQ accumulator update (`simd_acc_fused`, lazy + register-tiled + Finny cache) ≈ SF's design.
- Pairwise pack (`simd_pairwise_pack_impl`) ≈ SF `transform()`.
- `is_pseudo_legal` weight ≈ SF `pseudo_legal`, same frequency (TT move only).

## Recommended sequencing (REVISED 2026-06-14 — A tested DEAD, movegen-side is the thread)
1. **B+C (check-info cache)** — biggest verified win: recompute checkers/pinned/check-squares every node vs SF's once-in-do_move `set_check_info`. Large refactor, bit-identical. This is the real movegen/search-overhead slice of the 44% gap. NOT yet tested — distinct subsystem from the neutral threat-index results below.
2. **D (direct-write movegen) + F (fixed undo stack)** — localized bit-identical wins. NOTE: the two threat-side micro-opts (E) measured neutral at the ~0.5% noise floor, which is a prior that small scalar wins in an NNUE-dominated path may not register; measure D/F at conc≤2 before assuming they bank. Lower priority than B+C.
3. ~~E (threat index precompute)~~ — TESTED NEUTRAL both as table-shrink and zero-emit cull (see §E). The threat-index machinery is not the lever; the cost is NNUE weight-streaming volume (feature count / x-ray). DEAD for NPS.
4. ~~A (L1 sparse-input)~~ — TESTED DEAD (loses 1.8-2.4x at all densities; L1 too small for skip-detection to pay). Dense is correct.
5. **G/H** — behavior-affecting / structural, SPRT-gated. "Drop pairwise" is a SEPARATE untested eval question (copied Reckless; never measured vs plain CReLU) — but NOT a speed lever (SF is pairwise and fast).

**Process note:** A's already-rejected status (in-code comment) was missed by the
agent review and caught by manual verification. Verify each remaining target the
same way (search comments + git log) before implementing — the agents surface
candidates, not vetted work.

Bit-identical changes need only a bench-speed check + non-regression SPRT `[-2,1]`
(they can't lose Elo except via a bug; the win is NPS that converts at ~130-140
STC Elo/doubling). Validate speed at conc≤2 or with the Hercules OB worker
stopped (see project_hercules_ob_worker_vs_local_contention).
