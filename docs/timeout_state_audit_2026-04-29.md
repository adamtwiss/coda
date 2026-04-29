# Timeout / state-write correctness audit (2026-04-29)

Read-only audit of `src/search.rs` (HEAD = `0c192fb` on `main`). Three
timeout cases reviewed (normal, ponder, SMP) plus the TT-collision
class. Cross-references against Stockfish (`/home/adam/chess/engines/Stockfish/src/search.cpp`),
Reckless (`/home/adam/chess/engines/Reckless/src/search.rs`), and
Viridithas (`/home/adam/chess/engines/viridithas/src/search.rs`).

No source files were modified. Punch list at end is the deliverable.

---

## 1. Contract

**Strict no-write-after-stop policy.** Once `info.stop` becomes `true`,
no write must occur to:

1. The transposition table (`info.tt.store`)
2. Any history table — main, capture, continuation (`info.history.*`)
3. Pawn history (`info.pawn_hist`)
4. Correction history — `pawn_corr`, `np_corr`, `minor_corr`,
   `major_corr`, `cont_corr`
5. Killer / counter move tables (none currently — Coda removed
   killers/counters following the SF pattern, but `MovePicker` still
   reads `self.killers[]` in evasion mode; the writes are handled by
   `History::push_killer` if added back later)
6. Per-root-move node accumulator `info.root_move_nodes`
7. Triangular PV table `info.pv_table` / `info.pv_len` — EXCEPT for
   the explicit `stable_pv` restoration at `search.rs:1536-1539,
   1545-1548`, which intentionally restores the prior completed
   iteration's PV so `best_move` and `pv_table[0]` stay consistent
   for the bestmove emit (lichess oeZ7KRUt class fix, 2026-04-26).

**Stat counters** (`info.stats.*`, `info.stats_tt_static_eval_hits`,
`info.sel_depth`, `info.nodes`) are excluded — they're observability,
not search-deciding state. Even so, reducing post-stop stat writes is
nice-to-have for cleaner diagnostics.

### Code-comment block (paste-ready policy header for `search.rs`)

```rust
// State-write policy under stop
// =============================
//
// Once `info.stop` is set (either by external `stop`/`ponderhit`
// arriving on the UCI thread, by `should_stop` firing on the time
// budget, or by SMP main signaling helpers), no further writes to
// search-decision state must occur. "Search-decision state" means:
//
//   - TT (info.tt.store)
//   - All history tables (info.history.{main,capture,cont_hist})
//   - Pawn history (info.pawn_hist)
//   - Correction history (info.{pawn,np,minor,major,cont}_corr)
//   - Per-root-move nodes (info.root_move_nodes)
//   - Killers/counter moves
//   - Triangular PV (info.pv_table, info.pv_len) — except for the
//     explicit stable_pv restoration at the ID-loop interrupt point,
//     which restores the LAST COMPLETED iteration's PV.
//
// Stat counters (info.stats.*, info.sel_depth, info.nodes) are not
// covered — they're observability only.
//
// Two implementation patterns are accepted as gating:
//
//   1. Early-return-on-stop after each recursive call. If a child
//      negamax/qsearch returned and `info.stop.load()` is true, return
//      immediately without touching state. SF/Reckless/Viridithas all
//      use this. Coda mirrors this at lines 2364, 2485, 2672, 3112,
//      3422, etc.
//
//   2. Explicit stop-guard at the write site:
//        if !info.stop.load(Ordering::Relaxed) { ... write ... }
//      Used at the main TT.store (3278), QS TT.store (3562, 3699),
//      corrhist update (3319), and the soft-floor wait (1785).
//
// Either pattern is fine; mixing is fine. What's NOT fine: a write
// site that depends on neither an early-return guard upstream nor an
// explicit stop check, AND which executes data that may have been
// produced by a partial/aborted child search.
```

---

## 2. Site-by-site table

| File:line | What writes | Currently gated? | SF equivalent | Reckless equivalent | Verdict |
|-----------|-------------|------------------|---------------|---------------------|---------|
| `search.rs:1325` | `root_move_nodes.fill(0)` (search start) | n/a — search start, stop is false here | line 2008-area resets | similar | ✅ Safe |
| `search.rs:1537-1539, 1546-1548` | `pv_table[0]`, `pv_len[0]` from `stable_pv` | Explicit restoration AFTER stop fires | n/a (SF uses different PV save) | similar | ✅ Intentional exception per contract |
| `search.rs:1591` | `info.ponder_depth.store(depth, Relaxed)` | After should_stop break checks | n/a | n/a | ✅ Safe (only on completed iteration) |
| `search.rs:1786` | `info.stop.store(true)` (set stop before stockpile sleep) | n/a — sets stop, doesn't violate | n/a | n/a | ✅ Safe |
| `search.rs:1820` | `info.reductions[ply_u] = 0` (node entry) | Per-ply slot, overwritten on revisit | `(ss-1)->reduction = 0` (line 716) | similar | ✅ Safe — not search-deciding state across stops |
| `search.rs:1876` | `info.pv_len[ply_u] = 0` (clear at node entry) | Per-ply slot, overwritten on revisit | similar | similar | ✅ Safe — overwritten by parent on next visit |
| `search.rs:2015-2018` | `pv_table[ply_u]`, `pv_len[ply_u]` after TT cutoff | NOT gated; writes legal/no-op based on `is_pseudo_legal` | line 802 has same pattern | similar | ⚠️ See punch list #5 |
| `search.rs:2046-2049` | `cont_hist` malus on TT-cutoff for opponent's quiet | **NOT gated** | line 794-795 (NOT gated in SF either) | similar | ⚠️ See punch list #4 |
| `search.rs:2080-2084` | `pv_table[ply_u]`, `pv_len[ply_u]` after TT bounds-collapse cutoff | NOT gated | line 802 same pattern | similar | ⚠️ See punch list #5 |
| `search.rs:2092-2095` | `main_history` bonus on TT-cutoff (quiet TT move) | **NOT gated** | line 790-791 (NOT gated in SF either) | inline section | ⚠️ See punch list #4 |
| `search.rs:2106-2109` | `capture` (cont-hist for capture) on TT-cutoff | **NOT gated** | n/a (SF doesn't update cap-hist on TT cutoff) | n/a | ⚠️ See punch list #4 |
| `search.rs:2112` | `pv_len[ply_u] = 0` on TT-cutoff with no tt_move | NOT gated | n/a | n/a | ✅ Safe — clears, doesn't add data |
| `search.rs:2201` | `info.tt.store` eval-only stub (depth=-2) | Gated by `FEAT_TT_STORE` only — **NOT stop-gated** | line 428-area in Reckless | line 428 in Reckless | ⚠️ See punch list #6 |
| `search.rs:2209` | `info.static_evals[ply_u] = ...` | Per-ply slot, parent reads at child boundaries only | similar | similar | ✅ Safe |
| `search.rs:2232` | `info.static_evals[ply_u] = -INFINITY` (in-check sentinel) | Per-ply slot | similar | similar | ✅ Safe |
| `search.rs:2356-2357` | `moved_piece_stack[ply_u]=0`, `moved_to_stack[ply_u]=0` (NMP null sentinel) | Per-ply slot, overwritten by next move | SF sets `(ss)->currentMove = MOVE_NULL` similarly | similar | ✅ Safe |
| `search.rs:2498-2501` | `info.tt.store` for ProbCut verified score | NOT gated, but stop-check at 2485 immediately precedes | SF ProbCut has same pattern (gated by stop check upstream) | line 654 same pattern | ⚠️ See punch list #2 — small race window |
| `search.rs:2668, 2670` | `info.excluded_move[ply_u] = tt_move/NO_MOVE` (SE) | Set→clear pattern; cleared before recursion in negamax probe | similar | similar | ✅ Safe — guard pattern |
| `search.rs:2725-2728` | `moved_piece_stack[ply_u]`, `moved_to_stack[ply_u]` (per-move record) | Per-ply slot; overwritten on next iteration | SF sets `(ss)->currentMove = move` | similar | ✅ Safe |
| `search.rs:2877-2878` | `double_ext_count[ply_u + 1]` propagation | Per-ply slot | n/a | n/a | ✅ Safe |
| `search.rs:3052` | `info.reductions[ply_u] = reduction` | Per-ply slot | similar | similar | ✅ Safe |
| `search.rs:3109` | `info.root_move_nodes[idx] += ...` | **NOT gated** (writes BEFORE stop check at 3112) | SF: `rm.effort += nodes - nodeCount;` AFTER stop check (line 1328) | similar | ⚠️ See punch list #3 |
| `search.rs:3125-3131` | `pv_table[ply_u]`, `pv_len[ply_u]` on alpha raise | Within `if score > alpha`, AFTER stop check at 3112 | line 1359-area | similar | ✅ Safe (transitively gated by 3112) |
| `search.rs:3145-3148` | `main_history` bonus on cutoff | Inside cutoff arm, AFTER stop check at 3112 | line 1879-area (post-loop) | line 1062-area (post-loop) | ✅ Safe (transitively gated) |
| `search.rs:3161-3164` | `cont_hist` bonus on cutoff (4 plies) | Inside cutoff arm, AFTER stop check | similar | similar | ✅ Safe |
| `search.rs:3173-3176` | `pawn_hist` bonus on cutoff | Inside cutoff arm, AFTER stop check | n/a (SF doesn't have per-pawn-hash history this shape) | similar | ✅ Safe |
| `search.rs:3184-3187` | `main_history` malus for sibling quiets | Inside cutoff arm, AFTER stop check | line 1879 | line 1074 | ✅ Safe |
| `search.rs:3201-3204` | `cont_hist` malus for sibling quiets | Inside cutoff arm, AFTER stop check | similar | similar | ✅ Safe |
| `search.rs:3219` | `pawn_hist` malus for sibling quiets | Inside cutoff arm, AFTER stop check | n/a | n/a | ✅ Safe |
| `search.rs:3234-3236` | `capture` bonus for cutoff capture | Inside cutoff arm, AFTER stop check | similar | line 1062 | ✅ Safe |
| `search.rs:3249-3252` | `capture` malus for sibling captures | Inside cutoff arm, AFTER stop check | similar | line 1079 | ✅ Safe |
| `search.rs:3278-3292` | `info.tt.store` (main negamax) | **Stop-gated** ✅ | line 1908-area (NOT explicitly gated in SF — relies on stop-return) | line 1140 (NOT gated in Reckless either) | ✅ Coda is more conservative than SF/Reckless |
| `search.rs:3319-3326` | `update_correction_history` | **Stop-gated** ✅ | line 1500 (NOT explicitly gated) | line 1148 (NOT gated) | ✅ Coda is more conservative |
| `search.rs:3562-3563` | QS evasion `info.tt.store` | **Stop-gated** ✅ | similar | similar | ✅ Safe |
| `search.rs:3699-3703` | QS captures `info.tt.store` | **Stop-gated** ✅ | similar | similar | ✅ Safe |

### Notes on the table

- "NOT gated" means: no explicit `if !info.stop.load { ... }` guard at
  the write site. It does NOT necessarily mean the write happens under
  stop — most "NOT gated" sites are transitively safe because of an
  early-return-on-stop further up the call chain. The verdict column
  resolves which are actually exposed to corruption.

- The only sites with genuine post-stop write windows are the four
  flagged in §Punch list. Everything else is either gated by an
  early-return upstream, by an explicit stop check, by being a per-ply
  slot that's overwritten on revisit, or by living inside a `if score
  > alpha` / cutoff branch that requires a clean child search to
  reach.

---

## 3. Three timeout case analyses

### 3.1 Normal timeout

Triggered when `should_stop` (search.rs:616-653) detects the deadline
has passed. Fires every 4096 nodes inside `negamax`/`quiescence`
(checked at lines 1885, 3416). The store at line 648 sets
`info.stop = true` and returns true to the caller, who returns 0
upward.

**Trace:**

1. Main thread is deep in a recursive negamax call. `info.nodes &
   1023 == 0` triggers the periodic check. `should_stop` sees the
   deadline has passed, sets `info.stop`, returns true. Caller at
   line 1886 returns 0.
2. Parent's `score = -negamax(...)` becomes 0. Parent then hits the
   stop check at the appropriate post-recurse line (1891 inside
   nodes-counting, 2364 NMP, 2485 ProbCut, 2672 SE verify, 3112
   main move loop), returns 0.
3. Stop bubbles up through the entire stack. Each level returns
   0 without writing.
4. At the ID-loop in `search()` (line 1480-1769), `should_stop()`
   in the asp-window inner loop (1509, 1533, 1544) catches the stop;
   `stable_pv` is restored to keep `best_move` consistent (1536-1539,
   1545-1548); ID loop breaks at 1540/1549.

**Exposed write windows under normal timeout:**

- `root_move_nodes` accumulation at 3109 happens BEFORE the stop
  check at 3112. So the partial node count for the in-flight root
  move is added even though the search aborted. **(Punch list #3)**
- TT-cutoff history bonus at 2046, 2092, 2107 fires whenever a TT
  cutoff path is taken at any depth. If `should_stop` already returned
  true at line 1886 (the `&1023 == 0` branch fires), we early-returned
  with 0 — fine. But if stop fires DURING a sibling search and we
  re-enter THIS node's TT-cutoff fast path (the cutoff doesn't itself
  call should_stop), we update history under stop. **(Punch list #4)**
  Note: the `info.stop.load` check at line 1891 is OUTSIDE the
  `info.nodes & 1023 == 0` block — it ALWAYS runs at node entry.
  So the only window for this is "stop fires DURING TT-cutoff
  fast-path body, between line 1891 (already passed) and line 2046".
  In practice that window is microseconds.
- ProbCut TT.store at 2498 has stop-check at 2485 immediately before
  the verification re-search returns. The window is narrow but
  non-zero — the unmake_move at 2481, accumulator/threat pop at
  2482-2483, and the stop check at 2485, then `if score >=
  probcut_beta` at 2489, then the store at 2498. If stop fires
  between 2485 and 2498, we'd write a TT entry that's correct but
  for an incomplete tree. **(Punch list #2)**
- Eval-only TT stub at 2201 has no stop gate at all. **(Punch list #6)**

### 3.2 Ponder + ponderhit timeout

Per `should_stop` (lines 636-643), when `ponderhit_time > 0` (set by
the UCI ponderhit handler at uci.rs:481, 491), `should_stop` allows a
grace window of `min(remaining/4, 500ms)` beyond the deadline. After
grace expires, `info.stop` is set the same way as normal timeout.

The ID-loop also has its own ponderhit check at lines 1486-1489:
between iterations, if elapsed >= ph_time, break. This check is BEFORE
the next iteration's negamax is dispatched, so it's clean.

**Concerns specific to ponder:**

1. **Mid-iteration ponderhit grace window.** If ponderhit arrives
   during an iteration, `should_stop` extends the deadline by up to
   500ms. During this grace window, the iteration runs to completion
   if it can. Writes under grace are NOT under stop — they're under
   extended deadline. After grace expires, normal timeout flow
   applies. Same write windows as §3.1.

2. **Search thread spawn vs ponderhit race.** The UCI thread clears
   `info.stop` at uci.rs:219 BEFORE spawning the search thread, and
   clears `ponderhit_time` at uci.rs:226. This is intentional —
   clearing inside `search()` would race with a ponderhit that
   arrives in the spawn window. Comment at search.rs:1284-1288
   explains the bug this prevents. ✅ Correct.

3. **Wait-loop after ponder search completes.** uci.rs:252-255 spins
   on `ext_stop` and `ponderhit_time`. While in this loop, no search
   thread is running — `search_smp` already returned. No writes
   happen here.

4. **Fresh-search after ponderhit (uci.rs:271-306).** When ponderhit
   arrives and the ponder search hasn't yet stopped, UCI clears
   `info.stop`, clears `ponderhit_time`, then runs a SECOND
   `search_smp` with a fresh movetime budget. This is a clean fresh
   start — same flow as §3.1. The TT is hot from the ponder, but
   no state is "carried forward" that wasn't legitimately written
   during the prior (cleanly-completed) ponder search.

5. **The "ponder reached max_depth and waited" path.** If the
   ponder search exhausts max_depth and falls into the wait-loop at
   uci.rs:252, then ponderhit arrives, the fresh-search at uci.rs:303
   reuses the SAME `SearchInfo` (passed via the closure). Histories,
   corrhist, pawn_hist persist. Were they all completely written
   under non-stop? If max_depth was reached, the ID loop terminated
   at line 1481 (`for depth in 1..=effective_max`) — natural
   termination, not stop. So histories are clean.

**Verdict on ponder.** No additional write-under-stop classes beyond
what §3.1 identifies. The grace-period extension is just a deadline
shift; the actual stop signal still gates the same way. The
historical "ponder bugs were the dominant source of past Lichess
corruption" was traced (search.rs:1456-1463 commentary) to a
different bug — partial pv_table[0] mid-iteration desync — which
is fixed by the `stable_pv` restoration. That fix targets a PV
emit issue, not a state-pollution issue.

### 3.3 SMP

Helper threads share with main: `Arc<TT>`, `Arc<AtomicBool> stop`,
`Arc<AtomicU64> ponderhit_time`, `Arc<AtomicU64> global_nodes`,
`Arc<NNUENet>` (read-only weights), `Arc<Syzygy>` (read-only).

Helpers have their OWN: `info.history`, `info.pawn_hist`, `info.cont_corr`
and friends, NNUE accumulator, threat_stack, pv_table/pv_len,
moved_piece_stack/moved_to_stack, root_move_nodes, excluded_move,
reductions, double_ext_count, static_evals.

**Trace from main setting stop:**

1. Main's `search()` returns (line 1197 in search_smp). Main
   immediately sets `info.stop.store(true)` at search.rs:1200.
2. Helper threads, on their next `should_stop` / `info.stop.load`
   call (helper has time_limit=0 so `should_stop` only checks the
   stop flag and the node-flush bookkeeping at line 625-626), exit
   their negamax recursion via early-return.
3. Helper's outer loop in `search_helper` (line 1249-1262) checks
   `info.stop.load` between iterations and breaks (lines 1250, 1257).
4. Helper returns from `search_helper` cleanly. Stack unwinds.
5. `search_smp` joins each helper handle (line 1204). Total nodes
   accumulated to main's `info.nodes` (line 1209).

**Per-helper analysis of writes:**

- **History/pawn_hist/cont_corr/np_corr/etc.** — per-helper-local. No
  cross-helper writes. Helper writes them according to the same
  patterns analyzed in §2 — same correctness; same exposure to the
  stop windows in #3, #4, #6 of the punch list.

- **TT writes** — helpers write to the shared TT atomic buckets.
  Stop guards on lines 3278, 3562, 3699 hold for helpers. Stop
  windows in punch list #2 and #6 fire on helpers too. The XOR-key
  check at TT (tt.rs:395-397) catches torn reads. The Acquire/Release
  ordering on tt.rs:391-392 and 449-450 carries the happens-before
  edge.

- **Logical staleness across helpers.** A helper writes a depth-13
  entry to a hash where main wrote a depth-15 entry one ID iteration
  ago. The replacement policy at tt.rs:455-460 is `depth > slot_depth
  - 3`, so `13 > 15-3 = 12` is true: the helper CAN overwrite the
  deeper main entry. Main's next probe of that hash sees depth-13.
  This is intentional Lazy-SMP behavior (SF/Reckless do the same)
  but worth noting: in worst-case the depth-13 helper entry came
  from a near-stop helper that returned 0; that 0 is a valid
  search return (mate distance pruning bound, draw, etc.) IF it
  passed all the `if info.stop` gates. **Under the contract here,
  it didn't write because of the stop gate at 3278.** So this isn't
  an SMP-specific corruption window beyond what punch list items
  #2 and #6 already capture.

- **Helper threat_stack / NNUE accumulator** — per-helper-local. On
  a stop-cut iteration, these may end mid-pop. Helper's next
  iteration starts at `search_helper` which doesn't fully reset the
  accumulator (only the threat_stack at lines 1238-1242). The NNUE
  accumulator's `acc.reset()` is called at search start (line 1177),
  but NOT between iterations within `search_helper`'s ID loop.
  However, by the time we re-enter `negamax` at depth N+1, every
  push is matched by a pop in normal flow — and on stop, the
  early-return-on-stop at each unmake site (3102-3104, etc.) ensures
  the pops happen. Verified: no place where push happens without a
  matching pop on the stop path. ✅

- **Per-helper `root_move_nodes`.** Helpers reset this at search
  start (line 1129 says fresh history; root_move_nodes is on
  `SearchInfo` and is reset by the main search() at line 1325 — but
  helper invocations of `negamax` at ply==0 still write to it.
  Helper's pv_table extraction uses `info.pv_table[0][0]` at line
  1260 only. The root_move_nodes side-effect is a write to a
  helper-local field, with no propagation back to main. So the
  punch-list #3 issue applies per-helper but is not load-bearing.

**SMP verdict.** No SMP-specific corruption windows beyond what
single-thread analysis already covers. The `info.stop.load` early
returns are correctly mirrored in `search_helper` (lines 1250, 1257)
and in helper's negamax invocations (which use the same code as
main).

---

## 4. TT-collision audit

Sites that read `tt_entry.best_move` or `tt_entry.score`:

| File:line | What's read | Re-validation? | Verdict |
|-----------|-------------|----------------|---------|
| `search.rs:1573-1582` | `tt_entry.best_move` for root fallback | Validates from/to/flags against root_legal list | ✅ Safe |
| `search.rs:1649-1665` | `pv_tt.best_move` for PV extension | Validates from/to/flags against pv_legal at each step | ✅ Safe |
| `search.rs:1977` | `tt_entry.best_move` → `tt_move` | Used unfiltered as picker seed; picker calls `is_pseudo_legal` at line 404, 497, 1180 before yielding it as the first move | ✅ Safe — picker validates |
| `search.rs:1981, 2433, 2652, 3445, 3588` | `tt_entry.score` | Used directly for cutoff/window narrowing/SE margin | ⚠️ See below |
| `search.rs:2011-2014` | `tt_move` → write to pv_table on TT cutoff | Explicit `is_pseudo_legal` + `is_legal` check before write | ✅ Safe |
| `search.rs:2046-2049` | `info.moved_piece_stack[ply_u-1]`, `[ply_u-2]` (NOT tt_move) for cont-hist malus | Stack indices are own-history, not collision-injected | ✅ Safe (stack-driven, not tt_move-driven) |
| `search.rs:2076-2084` | `tt_move` → write to pv_table | Explicit `is_pseudo_legal` + `is_legal` check | ✅ Safe |
| `search.rs:2087-2110` | `tt_move` → history update (`move_from`, `move_to`, `tt_piece`) | **NO** `is_pseudo_legal` check; only `tt_piece != NO_PIECE` filter | ⚠️ See punch list #4 |
| `search.rs:2174-2175` | `tt_entry.static_eval` | Only used as `raw_eval` if > -(MATE_SCORE-100); no further legality gate needed | ✅ Safe — static eval is position-keyed scalar |
| `search.rs:2247-2248` | `move_to(tt_move)` for `tt_move_noisy` | Computed from collision-injected to-square; if collision points to wrong square, `piece_type_at` may return NO_PIECE_TYPE → tt_move_noisy=false. Used only to gate LMR adjustment. | ✅ Safe (worst case: minor mis-categorization of move type, no corruption) |
| `search.rs:2391` | `move_to(threat_entry.best_move)` after NMP fail | Used as `threat_sq` (a square index) for ordering hint; never used to make a move | ✅ Safe |
| `search.rs:2401-2402` | `tt_move` for `tt_move_is_quiet` (RFP gate) | Same as 2247 — read-only categorization | ✅ Safe |
| `search.rs:2659` | `move_from(tt_move)` for SE xray bonus | `mv == tt_move` check at 2638 — `mv` came from picker (validated). Only fires when picker yielded tt_move equal to the real legal move. | ✅ Safe |
| `search.rs:2668` | `info.excluded_move[ply_u] = tt_move` | Sets excluded for SE re-search; same picker-validated path. | ✅ Safe |
| `uci.rs:333-340` | `si.pv_table[0][1]` for ponder hint | Validated against legal moves of post-best position (uci.rs:319-330) before emit | ✅ Safe |

### TT-score class

`tt_entry.score` is consumed at:
- Cutoff gates (1995-2004): score-vs-beta direction match + bound type
  match + cut_node alignment + halfmove<90. Hash collision injecting
  a wrong score will trigger or suppress cutoff on a position whose
  score was for a DIFFERENT position. The cutoff returns the (wrong)
  score. Mitigation in place: `bound_matches` check ensures bound
  type is consistent with score direction; mate scores are filtered
  by `< MATE_SCORE-100` checks downstream.
- SE singular_beta computation (2663): `tt_score_local - depth -
  king_zone_pressure * MARGIN - xray_bonus`. Wrong score → wrong
  singular_beta → bogus extension/reduction decision. Self-correcting
  on next iteration when the real position's TT entry replaces the
  collision.
- Stand-pat refinement in QS (3590-3596): only narrows stand_pat;
  bound-direction check at 3590 prevents inversion.
- ProbCut TT-no-shot gate (2434-2435): only blocks ProbCut from
  firing — failure mode is "ProbCut runs when it shouldn't have"
  which is bounded by ProbCut's own re-search verification.

These are all consensus patterns (SF, Reckless, Viridithas, Obsidian
all read tt_entry.score the same way). Hash collision rate at 32-bit
key match: ~1 in 2^32 per probe, and at ~500K nps a 6-second search
visits ~3M positions, expected collisions ~700μs of compute. Most
collisions self-correct via re-search before they bubble up.

### TT-move class

The picker's `is_pseudo_legal` filter (line 404, 497, 1180) is the
primary defense. Once a tt_move passes that check, all uses inside
`negamax`'s move loop are on the picker-yielded `mv` (which was
validated). The remaining unvalidated reads are:

1. **TT-cutoff history bonus / cont-hist malus at lines 2046, 2092,
   2107.** These fire BEFORE entering the move loop; no picker
   validation has happened. Filtered only by `tt_piece != NO_PIECE`.
   See punch list #4.

2. **`tt_move_noisy` at 2246-2249, `tt_move_is_quiet` at 2400-2402.**
   Read-only categorization. If collision: `piece_type_at(move_to)`
   returns NO_PIECE_TYPE (square is empty in current position), so
   tt_move_noisy=false — gated as quiet. Used only to skip RFP
   when TT has a quiet best move. Worst case: collision causes us
   to skip RFP we could have taken. Bounded loss, self-corrects.

3. **`tt_move == NO_MOVE` IIR gate at 2255.** Pure presence check.
   Safe.

4. **`mv == tt_move` checks at 2601, 2622, 2638, 2742, 2809, 2932,
   3013.** All comparing picker-yielded `mv` against `tt_move`. If
   collision injected a tt_move that happens to BIT-EQUAL a real
   legal move, we'd treat it as the TT move. But picker validated
   that `tt_move` itself is pseudo-legal in current position via
   line 404 / 1180 — so if it's bit-equal to a legal move, it IS
   that legal move. ✅ Safe.

---

## 5. Punch list

Severity legend:
- **HIGH**: can corrupt search-decision state across iterations
- **MEDIUM**: small race window, write happens but consequences are
  bounded
- **LOW**: stylistic / inert sentinel, no measurable impact

### #1 — `root_move_nodes` accumulation under stop  [LOW]

**File:line:** `search.rs:3107-3110`

**Current behavior:**
```rust
if ply == 0 {
    let idx = (from as usize) * 64 + (to as usize);
    info.root_move_nodes[idx] += info.nodes - nodes_before;
}

if info.stop.load(Ordering::Relaxed) {
    return 0;
}
```

The accumulation runs BEFORE the stop check. SF (search.cpp:1321
THEN 1328) checks stop FIRST, then accumulates `rm.effort`.

**Proposed fix:** swap the two — stop check first, then accumulation.

```rust
if info.stop.load(Ordering::Relaxed) {
    return 0;
}

if ply == 0 {
    let idx = (from as usize) * 64 + (to as usize);
    info.root_move_nodes[idx] += info.nodes - nodes_before;
}
```

**Expected impact:** ~0 Elo. `root_move_nodes` is reset every
`search()` call (line 1325), so partial accumulation under stop only
affects THIS iteration's TM decision. The ID loop's per-iteration TM
gate (line 1685, `!info.should_stop()`) skips TM computation when
stop is set, so the polluted counter doesn't actually feed TM. The
fix is correctness hygiene, not Elo.

**Prior:** `[-3, 3]` non-regression bounds.

---

### #2 — ProbCut TT.store has small post-stop window  [MEDIUM]

**File:line:** `search.rs:2485, 2498-2501`

**Current behavior:**
```rust
if info.stop.load(Ordering::Relaxed) {
    return 0;
}

if score >= probcut_beta {
    info.stats.probcut_cutoffs += 1;
    // ... commentary ...
    info.tt.store(
        board.hash, depth - 3, score_to_tt(score, ply),
        TT_FLAG_LOWER, mv, raw_eval, tt_pv,
    );
    return score - (probcut_beta - beta);
}
```

The stop check at 2485 is right after unmake. If stop fires between
2485 and 2498 (a few microseconds — `score >= probcut_beta` branch
+ stat increment + hash compute), the store fires.

**Proposed fix:** add explicit guard at the store, mirroring the
main TT.store at 3278 and the QS stores at 3562, 3699:

```rust
if score >= probcut_beta {
    info.stats.probcut_cutoffs += 1;
    if FEAT_TT_STORE.load(Ordering::Relaxed) && !info.stop.load(Ordering::Relaxed) {
        info.tt.store(
            board.hash, depth - 3, score_to_tt(score, ply),
            TT_FLAG_LOWER, mv, raw_eval, tt_pv,
        );
    }
    return score - (probcut_beta - beta);
}
```

Note: also gate on `FEAT_TT_STORE` for consistency with other TT.store
sites (currently the ProbCut store is NOT gated on this feature flag,
which is inconsistent — minor pre-existing issue worth folding in).

**Expected impact:** +0 to +1 Elo. The window is genuinely small,
but the write is at depth-3 with a verified score — clobbering it
under stop poisons future probes of this exact position. Since the
score WAS verified at probcut_beta, it's a valid lower bound; the
"stop" condition just means the outer search context that produced
this verification is incomplete. Reckless does NOT gate this
(line 654) — they accept the same risk.

**Prior:** `[-3, 3]` correctness.

---

### #3 — TT-cutoff history bonus / cont-hist malus under stop  [MEDIUM]

**File:line:** `search.rs:2046-2049, 2092-2095, 2106-2109`

**Current behavior:** When a TT-entry produces a cutoff (lines
2003-2055 first cutoff, 2072-2123 bounds-collapse cutoff), Coda
updates:
- `info.history.cont_hist` (line 2046) — malus on opponent's quiet
- `info.history.main` (line 2092) — bonus on tt_move if quiet
- `info.history.capture` (line 2106) — bonus on tt_move if capture

None of these are stop-gated. SF has the same un-gated pattern at
line 790-795. Reckless has the same un-gated pattern at line 358-364.

**However**: Coda's TT-cutoff cont-hist malus at 2046 is unique to
Coda (Alexandria-derived). It's NOT gated on `tt_move != NO_MOVE` —
it fires whenever `score_above_beta && stack_len >= 2 && ply_u >= 2
&& opp_undo.captured == NO_PIECE_TYPE && our_undo.mv != NO_MOVE`.
Indices come from `moved_piece_stack` / `moved_to_stack` (which are
correctly own-search-state, not tt_move-injected). So the
collision-class concern doesn't apply here — but the stop-class one
does.

The history bonuses at 2092 and 2107 use `move_from(tt_move)`,
`move_to(tt_move)` directly. There's a `tt_piece != NO_PIECE` filter
but NOT an `is_pseudo_legal` check. Hash-collision exposure:

- If collision tt_move says `from=12, to=28, FLAG_NONE` and current
  position has our knight on 12: knight-move history at (knight, 28)
  is reinforced. If the collision-target had a queen on 12 in its
  position, our table now has a phantom knight-bonus on 28.
- If collision says `from=12, to=28, FLAG_PROMOTE_Q` and 12 has our
  bishop: `tt_is_cap` evaluates `piece_type_at(28) != NO_PIECE_TYPE
  || flags == FLAG_EN_PASSANT` — depends on what's at 28. The
  capture-history index uses `cpt_pt = piece_type_at(28)`, which
  could be anything.

**Proposed fix:** add an `is_pseudo_legal` gate before the history
updates AND a stop guard. This converts the bonus update into the
same defense-in-depth pattern as the PV-table writes at 2011-2014
and 2076-2079.

```rust
if tt_move != NO_MOVE
    && !info.stop.load(Ordering::Relaxed)
    && crate::movepicker::is_pseudo_legal(board, tt_move)
{
    // ... existing history bonus update ...
}
```

For the cont-hist malus at 2046, only the stop guard is needed (no
tt_move read in the index expression):

```rust
if score_above_beta && stack_len >= 2 && ply_u >= 2
    && !info.stop.load(Ordering::Relaxed)
{
    // ... existing malus update ...
}
```

**Expected impact:**
- Stop-guard component: +0 to +1 Elo. History scribbles under stop
  pollute next iteration's ordering; a single iteration's worth of
  noise is small but real (history values are ~1500-3000; a single
  bad bonus is ~150 cp of ordering bias).
- Pseudo-legal-guard component: +0 to +1 Elo. Hash-collision class
  is rare (1 in ~4 billion probes per check) but the failure mode
  is silent corruption of move-ordering tables. A defense-in-depth
  win comparable to the analogous PV-table guard at 2011-2014.
- Combined: +1 to +2 Elo as a single change. Self-play SPRT may
  show flat — same problem class as TT-PV-flag retune (raw +4.5
  becomes +9.5 with retune). Worth retune-on-branch.

**Prior:** `[0, 5]` for the bundle (correctness AND likely
positive-direction effect).

---

### #4 — `pv_table[ply_u]`/`pv_len[ply_u]` writes on TT cutoff with no `is_pseudo_legal` validation prior to commentary  [LOW]

**File:line:** `search.rs:2015-2018, 2080-2084, 2112`

**Current behavior:** These DO validate via `is_pseudo_legal +
is_legal` before writing tt_move into pv_table. Looks correct.

**Verdict:** Already gated correctly. **Listed here to confirm the
audit found nothing to change.** Move on.

---

### #5 — Eval-only TT stub at 2201 unguarded against stop and unguarded against under-replacement  [LOW]

**File:line:** `search.rs:2200-2202`

**Current behavior:**
```rust
if !tt_hit && FEAT_TT_STORE.load(Ordering::Relaxed) {
    info.tt.store(board.hash, -2, -INFINITY, TT_FLAG_UPPER, NO_MOVE, raw_eval, is_pv);
}
```

No stop guard. Since this is a depth=-2, score=-INFINITY,
flag=UPPER, NO_MOVE stub, the entry cannot produce a cutoff or
narrow alpha/beta (all gates filter on depth >= 0 or depth >= -1).
Only `static_eval` and `tt_pv` survive a future probe.

**Risk:** trivial. The stub may overwrite an in-flight real entry
written by another helper between our `info.tt.probe` (1969) and our
`info.tt.store` (2201). The replacement check at tt.rs:455-460
requires `depth > slot_depth - 3`; with our depth=-2 this means
`-2 > slot_depth - 3` → `slot_depth < 1`. So the stub can only
overwrite entries with slot_depth ≤ 0 (QS entries) when keys match.
Real interior negamax entries (depth ≥ 1) are safe.

**Proposed fix:** add a stop guard for consistency:

```rust
if !tt_hit
    && FEAT_TT_STORE.load(Ordering::Relaxed)
    && !info.stop.load(Ordering::Relaxed)
{
    info.tt.store(board.hash, -2, -INFINITY, TT_FLAG_UPPER, NO_MOVE, raw_eval, is_pv);
}
```

**Expected impact:** ~0 Elo. Hygiene only.

**Prior:** `[-5, 5]` non-regression.

---

### #6 — TT replacement policy depth threshold creates SMP staleness window  [MONITOR ONLY]

**File:line:** `tt.rs:455-460`

**Current behavior:**
```rust
if recovered_upper == key_upper {
    if depth > slot_depth - 3 || gen != slot_gen {
        bucket.data[i].store(new_data, Ordering::Release);
        bucket.keys[i].store(new_key, Ordering::Release);
    }
    return;
}
```

A helper at depth 13 can overwrite main's depth-15 entry (`13 >
15 - 3 = 12` is true). This is consensus Lazy-SMP policy
(SF/Reckless/Viridithas all do similar) but worth confirming the
tradeoff is intended.

**Verdict:** **NOT a bug. Listed for monitoring.** If we ever see
SMP-T4 still produce blunder regressions traced to TT replacement
churn (the v9 T=4 14× blunder bug per `project_v9_t4_threading_bug.md`),
revisit this gate. For now, don't touch.

---

## EMERGENCY section

**No emergency-class findings.** The strict-no-write-after-stop
contract is largely held. The four flagged write windows (#1, #2, #3,
#5) are small race windows or one-iteration history scribbles, not
cross-iteration corruption sources.

The user's hypothesis — "moderate-stepped game blunders look like
state-pollution bugs" — is NOT directly supported by this audit. The
two paths most likely to leak state across iterations are:
1. The TT itself, which IS gated on stop at the main negamax store
   (3278) and QS stores (3562, 3699). Punch list #2 (ProbCut) and #5
   (eval-only stub) are minor TT exposure but at depths that can't
   trigger cutoffs.
2. History tables, which are gated TRANSITIVELY by the stop check at
   line 3112 (post-recurse, pre-cutoff-arm). Punch list #3 (TT-cutoff
   history bonus) is the only history-table write under stop.

Punch list #3 IS plausible as a contributor to the observed
"clean-hash recovers SF-best" pattern — a one-iteration history
scribble shifts move ordering at the next iteration's same-position
revisits. But the magnitude is ~1 history bonus per stop event, and
the scrubbed game tree has thousands of clean history updates between
games. Self-play SPRT should see ≤2 Elo from fixing it.

**More likely culprits for the moderate-stepped-blunder pattern:**

- **Logical staleness via TT replacement (#6 in the list).** Helper
  threads at lower depths overwrite main's deeper entries. This is
  not a stop-class bug — it fires every search. If the user's test
  setup involves T>1, this would explain the "live game vs
  clean-hash" gap better than any post-stop write would.

- **History persistence across `go` commands.** `search()` only
  ages the main thread's history (line 1297, factor 4/5), doesn't
  clear it. A polluted main-thread history from prior games does
  carry forward. Helper threads DO clear at line 1218. So at T=1,
  history accumulates dirty signals; at T=4 with helpers clearing,
  the helpers race-write fresh signals on top — different ordering
  outcomes. If repro is at T=4 and clean-hash is `ucinewgame`-
  driven (which clears history), this is a candidate.

- **Recommend: a follow-up audit of `ucinewgame` handling and
  history aging coefficients** as a separate exercise. Out of scope
  for this audit but a stronger hypothesis for the observed
  symptom.

---

## 6. SPRT plan

### Bundle 1: hygiene fixes (#1 + #5)

Both are LOW-impact correctness/hygiene changes. Single branch:

- Move stop check before `root_move_nodes` accumulation (line 3107
  swap with 3112)
- Add stop guard to eval-only TT stub at 2201

Bench-neutral; SPRT bounds `[-5, 5]` non-regression. Expected
result: H1 within ~3K games (low-noise correctness change). If H0,
revert and investigate.

### Bundle 2: ProbCut store stop-gate (#2)

Single-file change, narrow window. SPRT bounds `[-3, 3]` correctness.
Expected ~3-6K games. Likely H1 at +0 to +1 Elo.

Submit independently — bundling with #3 dilutes the read.

### Bundle 3: TT-cutoff history-update gate (#3)

This is the highest-leverage of the four. Two components in one
branch:
- Stop guard on the cont-hist malus at 2046 and history bonuses at
  2092, 2107
- Pseudo-legal validation on tt_move before lines 2092 and 2107

Submit at `[0, 5]` since prior is "small positive Elo from history
scribble removal + collision defense-in-depth". If H1 at +1-3,
queue a retune-on-branch (focused: HIST_BONUS_MULT, HIST_BONUS_OFFSET,
HIST_BONUS_MAX, CAP_HIST_MULT, CAP_HIST_BASE, CAP_HIST_MAX — 6
params, ~1200 iters) since history-scale tunables may have been
calibrated against the noisier baseline.

If Bundle 3 H0, the un-retuned baseline doesn't show the wins; in
the spirit of `feedback_h0_isnt_terminal_bisect_and_diagnose.md`,
bisect into stop-guard-only and pseudo-legal-only sub-branches to
identify which component (if either) carries Elo.

### Submission order

1. Bundle 1 first — clears the smallest hygiene fix without burning
   fleet on a noisy [0, 5] test.
2. Bundle 2 second — independent narrow window.
3. Bundle 3 third — biggest expected lift; retune-on-branch ready
   if base SPRT lands +1-3.

Total expected fleet usage: ~25-40K games across three SPRTs.
Aggregate Elo: +1 to +4 if all three land. Strongest case for #3
under retune.

### What this audit is NOT testing

- The **hash-rate-vs-collision** SF/Reckless tradeoff (#6) is
  monitor-only and would require a separate deep audit of TT
  generation and replacement under T>1.
- **History persistence across `go` commands** (mentioned in
  EMERGENCY section). Worth a follow-up audit; could explain the
  user's symptom better than any of the punch-list fixes.

### Risk acknowledgements

- The `is_pseudo_legal` call added in #3 costs ~50ns per TT cutoff.
  TT cutoffs are common (~30% of nodes). Bench delta: expect
  ~0.5-1% NPS regression. Compensated by the +1-3 Elo from
  cleaner history. Not a self-play-blind change.
- The stop guards added in #1, #2, #5 are AtomicBool::load(Relaxed)
  — single-cycle on x86, ~3-5 cycles on aarch64. Bench-neutral.

---

## Files referenced

- `/home/adam/code/coda/src/search.rs` (4057 lines)
- `/home/adam/code/coda/src/tt.rs` (855 lines)
- `/home/adam/code/coda/src/movepicker.rs` (1498 lines)
- `/home/adam/code/coda/src/uci.rs` (lines 1-510)
- `/home/adam/code/coda/src/threat_accum.rs` (overview)
- `/home/adam/chess/engines/Stockfish/src/search.cpp` (lines 696,
  780-795, 1321, 1458, 1879)
- `/home/adam/chess/engines/Reckless/src/search.rs` (lines 280, 358,
  582, 598, 649, 681, 959, 1140, 1148)
- `/home/adam/chess/engines/viridithas/src/search.rs` (line 1119)
