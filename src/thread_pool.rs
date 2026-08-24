//! Persistent Lazy-SMP helper thread pool.
//!
//! Coda used to spawn `threads-1` helper threads on **every** `go`, each paying
//! a fresh `SearchInfo` allocation (~13 MB of history/pawn/corr tables) + NNUE
//! accumulator alloc, then discarding everything on join. This pool keeps the
//! worker threads (and their `SearchInfo`s) alive across searches: parked on a
//! condvar between `go`s, woken per search, refreshed from main.
//!
//! Ownership: a process-global `Mutex<Option<ThreadPool>>`. There is exactly
//! one search in flight at a time, so a single global pool is sufficient and
//! avoids threading an `Arc<ThreadPool>` through the per-`go` search thread that
//! uci.rs spawns. The pool is rebuilt when the worker count (`Threads`) or the
//! NNUE net (`NNUEFile`) changes; the TT / stop / ponderhit Arcs are re-shared
//! into each worker every dispatch (via `refresh_helper_per_go`), so `Hash`
//! resize and the per-`go` `SearchInfo` swap need no rebuild. On `quit` the
//! process exits with workers still parked — harmless (no owned resource beyond
//! the threads themselves, which the OS reclaims).
//!
//! Stage 1 (this file): behavior-identical to the old per-go spawn — workers run
//! the same `helper_run`, and each dispatch still copies main's history/corr in.
//! The only change is *how* the threads are managed. Stage 2 will drop the
//! per-go copy in favour of aging each worker's own persisted history.

use crate::board::Board;
use crate::search::{
    create_helper_info, helper_run, nnue_net_identity, refresh_helper_per_go, SearchInfo,
};
use crate::types::{Move, NO_MOVE};
use std::sync::{Condvar, Mutex};
use std::thread::JoinHandle;

/// The vote tuple returned per worker:
/// (nodes, best_move, score, depth, seldepth, tb_hits, ponder).
type HelperResult = (u64, Move, i32, i32, i32, u64, Move);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Parked, no work assigned.
    Idle,
    /// Main has published board/limits — worker should search.
    Go,
    /// Worker finished; result is ready for main to collect.
    Done,
    /// Pool is being torn down — worker should exit.
    Shutdown,
}

/// Per-worker state. The mutex is held by the worker for the duration of its
/// search (there is no contention — main only touches a slot while the worker
/// is Idle/Done), so no lock is taken on the search hot path itself.
struct SlotData {
    phase: Phase,
    info: SearchInfo,
    board: Board,
    thread_id: usize,
    max_depth: i32,
    result: HelperResult,
}

struct Slot {
    data: Mutex<SlotData>,
    to_worker: Condvar, // main -> worker (start / shutdown)
    to_main: Condvar,   // worker -> main (done)
}

struct Worker {
    slot: std::sync::Arc<Slot>,
    handle: JoinHandle<()>,
}

struct ThreadPool {
    workers: Vec<Worker>,
    /// NNUE net identity the worker accumulators were sized for.
    net_id: usize,
}

static POOL: Mutex<Option<ThreadPool>> = Mutex::new(None);

fn worker_loop(slot: std::sync::Arc<Slot>) {
    loop {
        let mut g = slot.data.lock().unwrap();
        // Wait for work. The predicate check before waiting handles a
        // notify that arrives before we park (main sets Go then notifies;
        // if we weren't waiting yet we still observe Go here and skip the wait).
        loop {
            match g.phase {
                Phase::Go => break,
                Phase::Shutdown => return,
                Phase::Idle | Phase::Done => {
                    g = slot.to_worker.wait(g).unwrap();
                }
            }
        }
        // phase == Go, holding the lock. Run the search directly on the slot's
        // owned SearchInfo/Board; no one else touches them until we set Done.
        {
            let sd = &mut *g;
            let res = helper_run(&mut sd.info, &mut sd.board, sd.max_depth, sd.thread_id);
            sd.result = res;
            sd.phase = Phase::Done;
        }
        slot.to_main.notify_one();
        drop(g); // release so main's collect() can lock and read the result
    }
}

/// Build `num_workers` fresh worker threads sized for `main`'s net.
fn build(num_workers: usize, main: &SearchInfo, board: &Board) -> ThreadPool {
    let net_id = nnue_net_identity(main);
    let mut workers = Vec::with_capacity(num_workers);
    for id in 1..=num_workers {
        let info = create_helper_info(main); // one-time: shared Arcs + acc/threat alloc
        let slot = std::sync::Arc::new(Slot {
            data: Mutex::new(SlotData {
                phase: Phase::Idle,
                info,
                board: board.clone(),
                thread_id: id,
                max_depth: 0,
                result: (0, NO_MOVE, 0, 0, 0, 0, NO_MOVE),
            }),
            to_worker: Condvar::new(),
            to_main: Condvar::new(),
        });
        let worker_slot = slot.clone();
        let handle = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || worker_loop(worker_slot))
            .expect("Failed to spawn SMP pool worker");
        workers.push(Worker { slot, handle });
    }
    ThreadPool { workers, net_id }
}

/// Tear down a pool: signal Shutdown, wake, and join every worker.
fn shutdown(pool: ThreadPool) {
    for w in &pool.workers {
        let mut g = w.slot.data.lock().unwrap();
        g.phase = Phase::Shutdown;
        drop(g);
        w.slot.to_worker.notify_one();
    }
    for w in pool.workers {
        let _ = w.handle.join();
    }
}

/// Dispatch `num_workers` helper searches for the current `go`: (re)build the
/// pool if the worker count or net changed, refresh each worker from `main`, set
/// its board + depth, and release it. Returns immediately; the caller runs the
/// main search and then calls `collect()`.
pub fn dispatch(num_workers: usize, main: &SearchInfo, board: &Board, max_depth: i32) {
    let mut guard = POOL.lock().unwrap();
    let net_id = nnue_net_identity(main);
    let needs_rebuild = match guard.as_ref() {
        Some(p) => p.workers.len() != num_workers || p.net_id != net_id,
        None => true,
    };
    if needs_rebuild {
        if let Some(old) = guard.take() {
            shutdown(old);
        }
        if num_workers > 0 {
            *guard = Some(build(num_workers, main, board));
        }
    }

    if let Some(pool) = guard.as_ref() {
        for w in &pool.workers {
            let mut g = w.slot.data.lock().unwrap();
            refresh_helper_per_go(&mut g.info, main);
            g.board.clone_from(board);
            g.max_depth = max_depth;
            g.phase = Phase::Go;
            drop(g);
            w.slot.to_worker.notify_one();
        }
    }
}

/// Wait for every dispatched worker to finish and return their vote tuples.
/// Must be called after the main search has set the shared stop flag (workers
/// exit their search loop on it). Returns an empty vec if there are no workers.
pub fn collect() -> Vec<HelperResult> {
    let guard = POOL.lock().unwrap();
    let mut out = Vec::new();
    if let Some(pool) = guard.as_ref() {
        out.reserve(pool.workers.len());
        for w in &pool.workers {
            let mut g = w.slot.data.lock().unwrap();
            while g.phase != Phase::Done {
                g = w.slot.to_main.wait(g).unwrap();
            }
            out.push(g.result);
            g.phase = Phase::Idle;
        }
    }
    out
}
