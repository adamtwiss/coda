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

### A. L1 fc_0 sparse-input — ALREADY TESTED AND REJECTED at L1=16 (NOT a free win)
- **VERIFIED 2026-06-14 (corrects the initial agent finding):** Coda's `L1Kernel::DenseAvx2` is a *deliberate* choice, not an oversight. Comment `nnue.rs:3332-3335`: "Dense variant (no zero-check): pairwise-CReLU inputs have high density (~89%), so the if-check overhead in the sparse variant exceeded the skip savings at L1=16." Coda HAS `sparse_l1_avx2` (`sparse_l1.rs:134`) + `find_nnz_chunks4` (`sparse_l1.rs:61`) and tested them — dense won empirically.
- **CORRECTED 2026-06-14 (Adam): SF is NOT wider.** SF dims: FT=1024 (same as Coda), `fc_0 = 1024 -> FC_0_OUTPUTS=L2=31` (`nnue_architecture.h:43,62-66`). SF's L1 output (31) ~= our L1=32 candidate. The real reason Coda's sparse lost is **the pairwise activation, not width**: SF feeds the raw FT output (naturally ClippedReLU-sparse) directly into the sparse fc_0; Coda applies **pairwise-CReLU first** (multiply accumulator-half pairs), which DENSIFIES the L1 input to ~89% nonzero — almost nothing to skip. Compounded (not caused) by L1=16's narrow output and Coda's SCALAR `find_nnz_chunks4` vs SF's vectorized cmpgt+movemask+LUT.
- **LIVE angles (sparse is NOT dead, just density-gated):** (a) measure Coda's ACTUAL per-position L1-input nnz density — the 89% may be a bulk average; if real positions are sparser, vectorized `find_nnz` + sparse could win now. (b) the L1=32 candidate matches SF's width (31~=32), so if it becomes prod the only question is its density. (c) the real unlock is architectural — feeding raw FT output to fc_0 (plain CReLU) instead of pairwise would give SF-style sparsity, but pairwise was an Elo win, so that's an eval trade to SPRT, not a free swap. Next step: instrument L1-input density (single-thread, no fleet impact) for both nets before any reimplementation.
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

## Recommended sequencing (REVISED after verifying A is already-rejected)
1. **D (direct-write movegen) + F (fixed undo stack)** — localized bit-identical wins, no prior-rejection history, bank them first.
2. **B+C (check-info cache)** — biggest verified movegen-side prize (recompute checkers/pinned/check-squares every node); large refactor but bit-identical. The single highest-leverage *open* target.
3. **E (threat index precompute)** — medium bit-identical.
4. **A (L1 sparse)** — only via vectorized find_nnz re-measure OR the L1=32 nets; settled loss at L1=16 otherwise.
5. **G/H** — behavior-affecting / structural, SPRT-gated, later.

**Process note:** A's already-rejected status (in-code comment) was missed by the
agent review and caught by manual verification. Verify each remaining target the
same way (search comments + git log) before implementing — the agents surface
candidates, not vetted work.

Bit-identical changes need only a bench-speed check + non-regression SPRT `[-2,1]`
(they can't lose Elo except via a bug; the win is NPS that converts at ~130-140
STC Elo/doubling). Validate speed at conc≤2 or with the Hercules OB worker
stopped (see project_hercules_ob_worker_vs_local_contention).
