# Persistent SMP Thread Pool — Design (2026-07-04)

Motivation: external-search audit finding (`docs/external_search_review_2026-07-04.md`,
SMP F1). Coda respawns `threads-1` helper threads on **every** `go`, each paying
a full `create_helper_info` copy (~260 KB history/corr) + NNUE materialize, then
discarding all learned state on join. Deployment is multi-threaded (lichess
4–8T, CCRL) — threading is our *deployment config*, and (correction 2026-07-04)
**OB does support Threads>1 SPRTs**, so this is directly OB-measurable, not
local-RR-only.

Two separable levers (build + test independently):
- **Stage 1 (overhead):** persist the worker threads across `go` (park/unpark),
  keep the per-search copy-from-main → behavior-identical, saves spawn + alloc +
  NNUE-acc-alloc per move. Validate: OB Threads=8, `[-2,1]` non-regression.
- **Stage 2 (diversity):** stop copying from main; workers keep + age their own
  history across moves (like main's `history.age(4,5)`) → genuine Lazy-SMP
  diversity Coda currently throws away. Validate: OB Threads=8, `[0,3]`.

## Ownership / lifetime

`search_smp` runs on a **per-`go` spawned search thread** (uci.rs:515-535 swaps
`info` out and `move`s it into a fresh thread so the UCI loop stays responsive).
The pool must outlive individual searches, so it **cannot** live on the search
thread. Decision:

- **`Arc<ThreadPool>` owned by the UCI loop**, created lazily on the first
  multi-thread `go` (and rebuilt when the `Threads` option changes), cloned into
  each per-`go` search thread and handed to `search_smp`.
- The pool owns `N-1` **persistent worker threads** and, crucially, **owns each
  worker's `SearchInfo`** (not the worker thread) — the SF/Reckless model. A
  worker thread holds a raw pointer to its slot; the park/signal protocol
  guarantees main and worker never touch a slot concurrently.

## Worker slot + park/signal protocol

Per worker `i`: a `WorkerSlot` owned by the pool:
```
struct WorkerSlot {
    info: SearchInfo,        // persistent across go (Stage 2 keeps history)
    board: Board,            // set each go
    limits: SearchLimits,
    result: Option<(u64 nodes, Move, i32 score, i32 depth, Move ponder)>,
    epoch: u64,              // bumped by main to release the worker
    state: WorkerState,      // Idle | Running | Done
}
```
Synchronization: one `Mutex<PoolShared>` + `Condvar` (workers wait; main
notifies), plus per-slot `AtomicU64 epoch` for the release edge.

**Release edge (main → worker), ARM-correct:**
1. main (workers all Idle): writes `slot.board`, `slot.limits`; for Stage 1 also
   copies main's history/corr into `slot.info` (exclusive access — workers Idle).
2. main bumps `epoch` with **`Release`** and `notify_all()` the Condvar.
3. worker wakes, loads `epoch` with **`Acquire`** (pairs with the Release →
   sees the board/limits/history writes), sets `state=Running`, runs
   `search_helper` on `slot.info`/`slot.board`.

**Completion edge (worker → main):**
4. worker writes `slot.result`, sets `state=Done` under the mutex, `notify`s.
5. main (after its own search + `stop.store(true)`) waits until all slots are
   `Done`, reads results, returns slots to `Idle`.

`stop` is the existing shared `Arc<AtomicBool>` — workers poll it exactly as
today (`search_helper` already stops on it). No change to the stop mechanism.

## What moves where

- `create_helper_info`'s **one-time** setup (share TT/stop/ponderhit/nnue_net/
  syzygy, alloc nnue_acc + threat_stack, `silent=true`) runs **once** at worker
  creation, not per go.
- `create_helper_info`'s **per-go** part (copy history/pawn_corr/np_corr/
  cont_corr/trans_corr from main; NNUE `acc.reset()` + `materialize`) runs per go
  in the release step. **Stage 2 drops the history/corr copy** and instead ages
  the worker's own tables once per go (`info.history.age(4,5)` + corr aging to
  match main's decay), keeping NNUE materialize (position changed).
- Per-go resets that must still happen each go: `start_time`, `time_limit=0`,
  `max_depth`, `sel_depth`, node counters, PV/pv_len, `completed_depth`,
  `threat_stack` reset. Audit the full per-go reset list against
  `create_helper_info` + the current spawn closure so nothing carries over
  incorrectly (Stage 1 must be byte-behavior-identical).

## Threads-option change / quit

- `Threads` set (uci.rs:~1252 neighborhood): rebuild the pool to `N-1` workers
  (join old workers with a shutdown flag, create new). Cheap, rare.
- `quit`: set a `shutdown` flag, bump epoch, join all workers before exit (no
  detached threads).
- `Hash` resize / `ucinewgame`: pool workers share the TT `Arc`; on resize the
  UCI loop swaps the TT Arc — workers must pick up the **new** Arc. Simplest:
  rebuild the pool on Hash resize too (rare), or re-share the TT into each slot
  at the next release. Decision: **re-share TT + stop + ponderhit Arcs into each
  slot at every release** (cheap Arc clones) so option changes can't leave a
  worker on a stale TT.

## Risks / validation

- **Deadlock/race:** the highest-risk class. Validate with local Threads=8 runs
  (bench parity at T=1 unaffected; `go` under load; repeated `go`/`stop`/
  `ucinewgame`/`Threads` cycles) BEFORE any OB submit. Stop the OB worker first
  (local-rr skill) for clean timing.
- **ARM ordering:** the epoch Release/Acquire is the one new reader-publish pair;
  the history/board writes are ordered behind it. Matches the CLAUDE.md standard.
- **Determinism:** bench is T=1 (single-thread path in `search_smp` untouched) →
  bench unchanged, OB bench-gate unaffected. SMP is inherently non-deterministic;
  Stage 1 must still be *behaviorally* identical (same per-go copy), only cheaper.
- **Ponder pairing:** helpers already return their 2nd PV move; the vote/select
  logic in `search_smp` is unchanged. Run the ponder-RR gate before merge.

## Staging plan

1. **Stage 1a — pool infra, T=1 no-op.** Add `thread_pool.rs`; wire `search_smp`
   to use the pool for `threads>1`; keep per-go copy. Verify T=1 bench unchanged,
   T=8 runs correctly and is behavior-identical (local RR self-play parity vs
   current main at T=8). OB Threads=8 `[-2,1]`.
2. **Stage 1b (if 1a clean).** — nothing; 1a IS stage 1.
3. **Stage 2 — diversity.** Drop the per-go history/corr copy; age worker tables
   instead. OB Threads=8 `[0,3]`. Keep Stage 1 merged first so the lever is
   isolated.

Files: new `src/thread_pool.rs`; edits to `src/search.rs` (`search_smp`,
`create_helper_info` split into one-time vs per-go), `src/uci.rs` (own the
`Arc<ThreadPool>`, rebuild on Threads/Hash change, shutdown on quit), `src/main.rs`
(module decl).
