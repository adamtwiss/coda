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

### A. L1 fc_0 matmul: Coda runs DENSE, SF runs sparse-input (nnz-skipping) — top eval target
- Coda's production AVX2 L1 kernel is `DenseAvx2` (`nnue.rs:2464` select, `sparse_l1.rs:324`): processes ALL `pw/4` input chunks every eval (~256 chunks). The kernel file is *named* sparse but the dense arm is selected (comment `nnue.rs:3332` removed the zero-check citing ~89% density).
- SF (`layers/affine_transform_sparse_input.h`): builds a nonzero-block index list (`find_nnz`: cmpgt+movemask + 256-entry offset LUT) and multiplies only weight columns for nonzero input blocks. The 1024→32 fc_0 is the largest matmul, every node.
- **Coda already HAS `sparse_l1_avx2` (`sparse_l1.rs:134`) and the input-chunk-major weight layout** — it just isn't selected. Missing piece: the `find_nnz` pre-pass + flip the selection. Bit-identical (skipping zero inputs doesn't change the dot product). Medium effort, **highest eval-side leverage**.
- Also closes a VNNI/AVX-512 gap on the rest of the fleet (SF's VNNI L1 is sparse + 3-chain; Coda's VNNI arms are dense).

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

## Recommended sequencing
1. **A (L1 sparse kernel)** — highest eval leverage, bit-identical, Coda already has the kernel+layout. Start here.
2. **D (direct-write movegen) + F (fixed undo stack)** — localized bit-identical wins, bank them.
3. **B+C (check-info cache)** — biggest movegen-side prize but large refactor; do after the cheap wins prove the thread.
4. **E (threat index precompute)** — medium bit-identical.
5. **G/H** — behavior-affecting / structural, SPRT-gated, later.

Bit-identical changes need only a bench-speed check + non-regression SPRT `[-2,1]`
(they can't lose Elo except via a bug; the win is NPS that converts at ~130-140
STC Elo/doubling). Validate speed at conc≤2 or with the Hercules OB worker
stopped (see project_hercules_ob_worker_vs_local_contention).
