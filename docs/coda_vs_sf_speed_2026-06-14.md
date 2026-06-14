# Coda vs Stockfish per-node speed review (2026-06-14)

**Why:** SF is ~44% faster single-thread and ~80% faster under 16× contention
than Coda on identical hardware (Hercules). At Coda's measured ~130-140 STC
Elo/NPS-doubling, the contended gap is **~115 STC Elo — ~80% of the ~130-150
SF gap**. Critically, **SF runs the SAME architecture** (FT1024 + threats), so
this is pure implementation efficiency, fully recoverable without touching eval
quality. Mechanism: Coda runs higher IPC (1.52 vs 1.37) and less memory/node
(102 vs 115 B) yet lower NPS → **more instructions per node**.

Comparative source review (read-only, 2 agents). Targets ordered by leverage.
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

### B. Incremental check/pin/check-square cache (SF StateInfo) — biggest movegen-side sink
- Coda recomputes `checkers()` + `pinned()` (each = 2 slider magics + leapers, `board.rs:664/692`) and the movepicker's `checking_sqs` (`movepicker.rs:562`) **from scratch every node** (`search.rs:3145-3146`, QS `4751`).
- SF's `set_check_info()` (`position.cpp:473`) computes `checkersBB`, `blockersForKing`, `pinners`, `checkSquares[all pt]` ONCE incrementally at the tail of `do_move`; movepicker/legal/gives_check read them as O(1) StateInfo loads.
- Fix: per-ply `CheckInfo` populated in/around `make_move`, threaded through the search stack. Bit-identical search. Large refactor but the single biggest movegen-side instructions/node win. **Enables C.**

### C. gives_check from cached check-squares (depends on B)
- Coda: `let gives_check = board.in_check()` AFTER make_move (`search.rs:3974`) = full attacker scan w/ 2 magics, every made move (and the child recomputes `checkers()` a 3rd time).
- SF: `gives_check(move)` BEFORE do_move from cached `check_squares(pt) & to` + discovered test — handful of bitboard ops, and the result seeds `checkersBB`. Bit-identical once B exists.

### D. Direct-write move generation (kill the 514-byte MoveList copies)
- Coda `generate_captures`/`generate_quiets` return `MoveList` (~514 B) BY VALUE (`movegen.rs:97/271`), then the picker re-pushes into `self.moves`. `generate_all_moves` (`movegen.rs:444`) copies caps+quiets into a 3rd list (every evasion/QS-in-check node).
- SF generates straight into the picker buffer; no by-value return, no re-push, dedicated `MoveList<EVASIONS>`.
- Fix: `&mut MoveList`/`&mut MovePicker` direct write. Localized, bit-identical, removes 1-2 full list copies per node.

### E. Threat feature-index precompute (avoid double table-chase per ply)
- `apply_threat_deltas` (`threats.rs:1644`) re-derives every delta's index via `threat_index` (`threats.rs:472`: color remap + `piece_pair` LUT + `pair.base()` + flip + `piece_offset` + a 48KB `attack_index[12][64][64]` chase) **twice per ply** (once per perspective), every evaluated make_move. The apply SIMD kernel itself is SF-class; the scalar index loop + 48KB table thrash is the cost.
- SF builds the changed-index list once in `append_changed_indices`; `double_inc_update` cancels recapture toggles.
- Fix: precompute per-perspective indices at delta-generation (`push_threats_*`) or cache in `RawThreatDelta`. Bit-identical, medium effort.

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
1. **B+C (check-info cache)** — biggest verified win: recompute checkers/pinned/check-squares every node vs SF's once-in-do_move `set_check_info`. Large refactor, bit-identical. This is the real movegen/search-overhead slice of the 44% gap.
2. **D (direct-write movegen) + F (fixed undo stack)** — localized bit-identical wins, bank them.
3. **E (threat index precompute)** — medium bit-identical.
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
