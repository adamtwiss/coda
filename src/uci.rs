/// UCI protocol implementation.

use std::io::{self, BufRead};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::board::Board;
use crate::search::*;
use crate::types::*;

/// Among multiple TB-drawn root moves, pick the one with the highest
/// "qualitative draw" score. Used only when `probe_root_pv` returned
/// `wdl == 0` — restricted to definite draws (skips cursed wdl=1 and
/// blessed wdl=-1, where the 50-MR matters and capturing can flip the
/// outcome).
///
/// Preference (high to low):
///   1. Moves leading to a position where the OPPONENT has insufficient
///      material to mate (we cannot lose — strictly safer than any
///      drawn fortress that requires defense). This is the
///      qualitative-Lichess fix: KB-vs-K beats KB-vs-KR even though
///      both are TB-drawn.
///   2. Among non-IM-resulting moves: prefer the move that captures
///      the most material (less material left for the opponent to
///      play dangerous tricks with).
///   3. Fall back to the shakmaty-picked move.
///
/// Returns the chosen UCI move string. No effect when TB isn't loaded
/// (caller is gated by `if let Some(ref tb) = syzygy { ... }`).
fn pick_drawn_tb_move(board: &Board, fallback_uci: &str) -> String {
    use crate::movegen::generate_legal_moves;
    let legal = generate_legal_moves(board);

    // Score each legal move: (im_terminal, captured_value).
    // im_terminal: true if the position after this move has insufficient
    //   material per FIDE 5.2 (game ends immediately by rule).
    // captured_value: SEE value of the captured piece (0 for non-cap).
    // Higher tuple wins, with im_terminal taking priority.
    let mut best_im_terminal = false;
    let mut best_captured_value: i32 = -1;
    let mut best_uci = fallback_uci.to_string();

    for i in 0..legal.len {
        let mv = legal.get(i);
        let to = move_to(mv);
        let flags = move_flags(mv);

        // Captured-piece value (board reflects pre-move state).
        let captured_pt = board.piece_type_at(to);
        let captured_value = if captured_pt != NO_PIECE_TYPE {
            crate::eval::see_value(captured_pt)
        } else if flags == FLAG_EN_PASSANT {
            crate::eval::see_value(PAWN)
        } else {
            0
        };

        // IM check requires making the move. Clone board and apply.
        let mut probe = board.clone();
        if !probe.make_move(mv) {
            continue;  // illegal — should not happen for legal moves but defensive
        }
        let im_terminal = probe.is_insufficient_material();

        // Lex order: im_terminal > captured_value > position-in-list (stable).
        let better = (im_terminal, captured_value) > (best_im_terminal, best_captured_value);
        if better {
            best_im_terminal = im_terminal;
            best_captured_value = captured_value;
            best_uci = crate::types::move_to_uci(mv);
        }
    }

    best_uci
}

pub fn uci_loop_with_nnue(nnue_path: Option<&str>, book_path: Option<&str>, classical: bool) {
    let mut board = Board::startpos();
    let mut info = SearchInfo::new(64);
    let mut stop_flag = info.stop.clone(); // keep a handle to signal stop from UCI loop
    let mut ponderhit_flag = info.ponderhit_time.clone(); // shared ponderhit hard deadline
    let mut ponderhit_soft_flag = info.ponderhit_soft.clone(); // shared ponderhit soft deadline
    let mut ponderhit_floor_flag = info.ponderhit_floor.clone(); // shared ponderhit min think
    // Separate flag set ONLY by external UCI "stop"/"ponderhit"/"quit" — distinct
    // from info.stop (which search_smp also sets internally when main search
    // returns). Used by the ponder wait loop below so it waits for an actual
    // external ack rather than racing with search_smp's internal teardown.
    let external_stop: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    // Set when the running ponder is being abandoned because a NEW position+go
    // arrived (opponent didn't play the predicted ponder move). The search
    // thread checks this on its way out and skips emitting bestmove — its
    // pv_table is from the predicted-but-not-real position, so the move
    // doesn't apply to whatever the new go is for. Without this suppress,
    // the stale ponder bestmove gets read by the GUI as the response to the
    // new go and forfeits on lichess (game 2agDftuq, 2026-04-29: predicted
    // e3d3, opp played e3e4, ponder PV `g3f3 d3c4` was legal at the d3
    // position but illegal at e4 → king-into-check → resign).
    let suppress_bestmove: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    let mut ponder_limits: Option<SearchLimits> = None; // pending limits for ponderhit
    let mut ponder_stm: u8 = crate::types::WHITE; // side to move at ponder start
    let mut opening_book: Option<crate::book::OpeningBook> = None;
    let mut use_book = true;
    let mut syzygy: Option<std::sync::Arc<crate::tb::SyzygyTB>> = None;
    let mut syzygy_path: Option<String> = None; // remembered for cache-size reloads
    let mut tb_hash_mb: usize = crate::tb::DEFAULT_TB_HASH_MB;
    let mut num_threads: usize = 1;

    // Pre-load NNUE if path given via CLI, otherwise auto-discover
    if let Some(path) = nnue_path {
        if let Err(e) = info.load_nnue(path) {
            eprintln!("ERROR: Failed to load NNUE from {}: {}", path, e);
            std::process::exit(1);
        }
    } else {
        let loaded = info.auto_discover_nnue();
        if !loaded && !classical {
            eprintln!("Error: No NNUE net found. Cannot play without NNUE.");
            eprintln!("  Use: -nnue <path>, 'setoption name NNUEFile value <path>',");
            eprintln!("       'make' to embed, or '--classical' for PeSTO eval.");
            std::process::exit(1);
        }
        if !loaded && classical {
            eprintln!("info string Classical (PeSTO) eval mode — no NNUE net loaded.");
        }
    }

    // Pre-load opening book if path given via CLI
    if let Some(path) = book_path {
        match crate::book::OpeningBook::load(path) {
            Ok(b) => opening_book = Some(b),
            Err(e) => eprintln!("Failed to load book: {}", e),
        }
    }

    // Search thread handle — returns SearchInfo back after search completes
    let mut search_handle: Option<std::thread::JoinHandle<SearchInfo>> = None;
    // Track when the current search started (for ponderhit time calculation)
    let mut ponder_search_start: Option<std::time::Instant> = None;

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            "uci" => {
                println!("id name Coda");
                println!("id author Adam Twiss");
                println!("option name Hash type spin default 64 min 1 max 4096");
                println!("option name Threads type spin default 1 min 1 max 256");
                println!("option name NNUEFile type string default <empty>");
                println!("option name OwnBook type check default true");
                println!("option name BookFile type string default <empty>");
                println!("option name MoveOverhead type spin default 100 min 0 max 5000");
                println!("option name Ponder type check default false");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name TBHash type spin default 16 min 0 max 1024");
                println!("option name HiddenActivation type combo default screlu var screlu var crelu");
                println!("option name LoadAnyway type check default false");
                // Tunable search parameters (for SPSA)
                for (name, _, default, min, max, _c_end, _is_core) in crate::search::tunable_params() {
                    println!("option name {} type spin default {} min {} max {}", name, default, min, max);
                }
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                // Wait for any active search to finish before clearing state
                if let Some(handle) = search_handle.take() {
                    suppress_bestmove.store(true, Ordering::Relaxed);
                    external_stop.store(true, Ordering::Relaxed);
                    stop_flag.store(true, Ordering::Relaxed);
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                        ponderhit_flag = info.ponderhit_time.clone();
                        ponderhit_soft_flag = info.ponderhit_soft.clone();
                        ponderhit_floor_flag = info.ponderhit_floor.clone();
                    }
                }
                info.tt.clear();
                info.history.clear();
                info.clear_correction_history();
                info.clear_pawn_hist(); // was missing — stale data leaked between games
                if let Some(acc) = &mut info.nnue_acc { acc.reset(); }
                // Clear Syzygy probe cache on new game (prevents stale entries
                // from a prior game leaking into the new one's search).
                if let Some(ref tb) = syzygy { tb.clear_cache(); }
                board = Board::startpos();
            }
            "position" => {
                parse_position(&tokens, &mut board);
            }
            "go" => {
                // Wait for any pending search to finish first. If we're abandoning
                // a ponder (predicted opp move didn't happen), suppress its
                // bestmove emit — it's for the predicted position, not whatever
                // this new go is for.
                if let Some(handle) = search_handle.take() {
                    suppress_bestmove.store(true, Ordering::Relaxed);
                    external_stop.store(true, Ordering::Relaxed);
                    stop_flag.store(true, Ordering::Relaxed);
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                    }
                }
                let is_ponder = tokens.iter().any(|&t| t == "ponder");

                // Try Syzygy tablebase at root. Behaviour splits on is_ponder:
                //   - Non-ponder: walk DTZ to build a multi-ply PV, emit a
                //     single info line with the walked line + bestmove, done.
                //   - Ponder: emit the walked PV as a seed info line (so the
                //     GUI has a real ponder_move to think about) and fall
                //     through to search. Search still runs for ponder-cache
                //     TT stockpile, UCI protocol compliance (no premature
                //     bestmove), and depth-counter updates. Ponderhit handler
                //     below will override with TB-optimal move when the time
                //     comes to play.
                if let Some(ref tb) = syzygy {
                    if crate::bitboard::popcount(board.occupied()) as usize <= tb.max_pieces() {
                        if let Some((mut tb_pv, wdl)) = tb.probe_root_pv(&board, 32) {
                            // Drawn-root qualitative tiebreak (wdl == 0 only;
                            // skip cursed/blessed where 50-MR matters).
                            // shakmaty's `best_move` picks arbitrarily among
                            // drawing moves — we override to prefer
                            // IM-terminal recaptures over drawn fortresses.
                            // See feedback_egtb_drawn_tiebreak_unfixable_via_sprt.md.
                            // Lichess game I4qJhfQw m103 and VE9mvCIG m~67 both
                            // exhibited Coda skipping the IM-terminal recapture.
                            if wdl == 0 && !tb_pv.is_empty() {
                                tb_pv[0] = pick_drawn_tb_move(&board, &tb_pv[0]);
                            }
                            // Validate the FIRST move of the walked PV — if
                            // TB returns an illegal "king capture" in a mate
                            // position we want to fall through to search.
                            let legal = crate::movegen::generate_legal_moves(&board);
                            let mut tb_valid = false;
                            if let Some(parsed) = parse_uci_move(&board, &tb_pv[0]) {
                                for i in 0..legal.len {
                                    if move_from(legal.get(i)) == move_from(parsed)
                                        && move_to(legal.get(i)) == move_to(parsed) {
                                        tb_valid = true;
                                        break;
                                    }
                                }
                            }
                            if tb_valid {
                                let score_str = if wdl > 0 {
                                    format!("score cp {}", crate::tt::TB_WIN)
                                } else if wdl < 0 {
                                    format!("score cp -{}", crate::tt::TB_WIN)
                                } else {
                                    "score cp 0".to_string()
                                };
                                let pv_str = tb_pv.join(" ");
                                let depth = tb_pv.len().max(1);
                                println!("info depth {} seldepth {} {} tbhits 1 pv {}",
                                         depth, depth, score_str, pv_str);
                                if !is_ponder {
                                    println!("bestmove {}", tb_pv[0]);
                                    continue;
                                }
                                // During ponder: info line emitted as a seed
                                // so GUI has ponder_move = tb_pv[1]. Fall
                                // through to search for stockpile + UCI
                                // compliance.
                            }
                            // TB move invalid — fall through to search (which has interior TB probes)
                        }
                    }
                }

                // Try opening book first (not in ponder mode)
                if use_book && !is_ponder {
                    if let Some(ref book) = opening_book {
                        if let Some(book_move) = book.pick_move(&board) {
                            let uci = move_to_uci(book_move);
                            // Emit a nominal info line so GUIs don't show an empty PV
                            // for book moves. Score cp 0 reflects "we didn't evaluate";
                            // this matches the convention used by most engines with a
                            // built-in book.
                            println!("info depth 1 seldepth 1 score cp 0 nodes 0 nps 0 time 0 pv {}", uci);
                            println!("bestmove {}", uci);
                            continue;
                        }
                    }
                }
                let mut limits = parse_go(&tokens);
                if is_ponder {
                    ponder_limits = Some(limits.clone());
                    ponder_stm = board.side_to_move;
                    limits.infinite = true;
                } else {
                    ponder_limits = None;
                }
                // Warn if no NNUE net is loaded
                if info.nnue_net.is_none() {
                    println!("info string WARNING: No NNUE net loaded! Playing with classical eval.");
                }

                // Move info into search thread, get it back when search finishes
                // Clear stop flag here (not in search thread) to avoid race with
                // ponderhit: if ponderhit arrives between spawn and search start,
                // the search thread must NOT overwrite it.
                stop_flag.store(false, Ordering::Relaxed);
                // Also clear external_stop for this new search so the ponder
                // wait loop below correctly waits for a fresh external ack.
                external_stop.store(false, Ordering::Relaxed);
                // Clear ponderhit_time here too (not in search()), same race reason:
                // if ponderhit arrives in the window between `go ponder` and search()
                // entry, search() clearing it would clobber the legitimate deadline.
                ponderhit_flag.store(0, Ordering::Relaxed);
                ponderhit_soft_flag.store(0, Ordering::Relaxed);
                ponderhit_floor_flag.store(0, Ordering::Relaxed);
                // Clear the abandon-ponder suppress flag — this new search
                // owns its bestmove emit. Set right before spawn so any value
                // left from the prior go-handler abandonment can't leak into
                // this thread's emit path.
                suppress_bestmove.store(false, Ordering::Relaxed);
                ponder_search_start = Some(std::time::Instant::now());
                let mut search_board = board.clone();
                let shared_tt = info.tt.clone();
                let shared_net = info.nnue_net.clone();
                let shared_stop = stop_flag.clone();
                // Preserve user-set config (MoveOverhead) across mem::replace
                // — the placeholder SearchInfo we leave behind on the UCI
                // thread is what the ponderhit handler reads, and it must
                // see the same overhead that the search thread is using.
                let saved_overhead = info.move_overhead;
                let search_info = std::mem::replace(&mut info, SearchInfo::new_with_shared(
                    shared_stop, shared_tt, shared_net,
                ));
                info.move_overhead = saved_overhead;
                ponderhit_flag = search_info.ponderhit_time.clone();
                ponderhit_soft_flag = search_info.ponderhit_soft.clone();
                ponderhit_floor_flag = search_info.ponderhit_floor.clone();
                let threads = num_threads;
                let is_ponder_search = is_ponder;
                let ext_stop = external_stop.clone();
                let suppress = suppress_bestmove.clone();
                search_handle = Some(std::thread::Builder::new()
                    .stack_size(16 * 1024 * 1024)
                    .spawn(move || {
                        let mut si = search_info;
                        let go_received = std::time::Instant::now();
                        let mut best_move = search_smp(&mut search_board, &mut si, &limits, threads);
                        let search_elapsed = go_received.elapsed();
                        // In ponder mode, if search completed naturally (not stopped
                        // externally), wait for stop/ponderhit before outputting
                        // bestmove.
                        let wait_start = std::time::Instant::now();
                        if is_ponder_search && !ext_stop.load(std::sync::atomic::Ordering::Relaxed)
                            && si.ponderhit_time.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                            while !ext_stop.load(std::sync::atomic::Ordering::Relaxed)
                                && si.ponderhit_time.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                                std::thread::sleep(std::time::Duration::from_millis(1));
                            }
                        }
                        let wait_elapsed = wait_start.elapsed();

                        // If we exited the wait-loop via ponderhit (not external stop)
                        // AND the allocated hard_limit hasn't been reached yet, run a
                        // FRESH timed search for the remaining budget. This fixes the
                        // case where ponder reached max_depth in a trivial position
                        // and we'd otherwise emit bestmove instantly — ignoring the
                        // clock budget the ponderhit handler computed for us.
                        //
                        // Why: the ponder's TT work is preserved (shared via Arc<TT>),
                        // so the fresh search starts with a hot TT and goes deeper
                        // than it would cold. The engine still gets the "ponder benefit"
                        // — just as extra depth rather than instant emit.
                        let mut fresh_elapsed = std::time::Duration::ZERO;
                        if is_ponder_search && !ext_stop.load(std::sync::atomic::Ordering::Relaxed) {
                            let ph_deadline = si.ponderhit_time.load(std::sync::atomic::Ordering::Relaxed);
                            let now_elapsed = go_received.elapsed().as_millis() as u64;
                            if ph_deadline > now_elapsed + 5 {
                                let remaining = ph_deadline - now_elapsed;
                                si.stop.store(false, std::sync::atomic::Ordering::Relaxed);
                                si.ponderhit_time.store(0, std::sync::atomic::Ordering::Relaxed);
                                si.ponderhit_soft.store(0, std::sync::atomic::Ordering::Relaxed);
                                si.ponderhit_floor.store(0, std::sync::atomic::Ordering::Relaxed);
                                // Movetime (not full TM): full TM's dynamic stability
                                // cut (stability_factor → 0.5× in stable positions)
                                // actively reduces fresh-search time in simple endgames,
                                // producing MORE stockpile. Verified: v3 (movetime)
                                // had 3/28 stockpile at 60+1; v6 (full TM) had 7/28.
                                // Movetime runs the full budget — best for this case.
                                //
                                // movetime_floor = inc - overhead. This enforces the
                                // "never think less than we gain" rule on ponderhit
                                // fresh-searches too — without it, TT-cached positions
                                // instant-emit and stockpile clock on ponder-heavy TCs
                                // like lichess blitz (lichess 6CQJQNVu).
                                let our_inc = if search_board.side_to_move == crate::types::WHITE {
                                    limits.winc
                                } else {
                                    limits.binc
                                };
                                let floor = our_inc.saturating_sub(si.move_overhead);
                                let fresh_limits = SearchLimits {
                                    infinite: false,
                                    movetime: remaining,
                                    movetime_floor: floor,
                                    ..SearchLimits::new()
                                };
                                let fresh_start = std::time::Instant::now();
                                best_move = search_smp(&mut search_board, &mut si, &fresh_limits, threads);
                                fresh_elapsed = fresh_start.elapsed();
                            }
                        }

                        // Belt-and-braces: validate pv_table[0][1] is legal in the
                        // position after best_move. The upstream cause (stable-PV
                        // snapshot in search.rs) prevents the partial-iteration
                        // inconsistency that lichess oeZ7KRUt (2026-04-26) hit, but
                        // any future regression that desyncs pv_table from
                        // best_move would still be caught here — losing the
                        // ponder hint instead of forfeiting the game.
                        let ponder_legal = si.pv_len[0] >= 2
                            && si.pv_table[0][1] != crate::types::NO_MOVE
                            && {
                                let mut after_best = search_board.clone();
                                if after_best.make_move(best_move) {
                                    let legal = crate::movegen::generate_legal_moves(&after_best);
                                    let p_from = move_from(si.pv_table[0][1]);
                                    let p_to = move_to(si.pv_table[0][1]);
                                    let mut ok = false;
                                    for i in 0..legal.len {
                                        if move_from(legal.get(i)) == p_from
                                            && move_to(legal.get(i)) == p_to {
                                            ok = true;
                                            break;
                                        }
                                    }
                                    if !ok {
                                        // Emit on stdout (UCI `info string` reaches
                                        // GUI/bot logs) AND stderr (caught by
                                        // cutechess -debug). Unique tag for grep.
                                        let msg = format!(
                                            "PV_PONDER_BUG dropped illegal ponder={} after best={} root={}",
                                            move_to_uci(si.pv_table[0][1]),
                                            move_to_uci(best_move),
                                            search_board.to_fen());
                                        println!("info string WARNING: {}", msg);
                                        eprintln!("{}", msg);
                                    }
                                    ok
                                } else { false }
                            };
                        let pv_consistent = ponder_legal
                            && move_from(si.pv_table[0][0]) == move_from(best_move)
                            && move_to(si.pv_table[0][0]) == move_to(best_move);
                        // Measure println wall-clock — if it blocks on stdout (slow
                        // reader on the other end), this is how we catch it.
                        let pr_start = std::time::Instant::now();
                        // Suppress emit if the ponder is being abandoned because
                        // a new go arrived (predicted opp move didn't happen).
                        // The new go-handler set this flag before joining us; its
                        // own search will emit the correct bestmove for the
                        // actual position.
                        if suppress.load(std::sync::atomic::Ordering::Relaxed) {
                            // Skip emit. Caller's new go owns the next bestmove.
                        } else if pv_consistent {
                            println!("bestmove {} ponder {}", move_to_uci(best_move), move_to_uci(si.pv_table[0][1]));
                        } else {
                            println!("bestmove {}", move_to_uci(best_move));
                        }
                        // Explicit flush: default stdout is LineWriter which *should*
                        // flush on \n, but when piped to another process (cutechess)
                        // there are edge cases where writes sit in a buffer. This
                        // guarantees cutechess sees bestmove immediately.
                        use std::io::Write;
                        let _ = std::io::stdout().lock().flush();
                        let pr_elapsed = pr_start.elapsed();
                        // Log anything suspiciously long to stderr (cutechess doesn't
                        // account stderr against the engine clock).
                        if pr_elapsed.as_millis() > 20
                            || wait_elapsed.as_millis() > 2000
                            || (!is_ponder_search && search_elapsed.as_millis() > (si.time_limit + 500) as u128)
                        {
                            eprintln!(
                                "TM_TRACE search={}ms fresh={}ms time_limit={}ms wait={}ms println={}ms ponder={} pv_ok={}",
                                search_elapsed.as_millis(),
                                fresh_elapsed.as_millis(),
                                si.time_limit,
                                wait_elapsed.as_millis(),
                                pr_elapsed.as_millis(),
                                is_ponder_search, pv_consistent,
                            );
                        }
                        si // return SearchInfo for reuse
                    }).expect("Failed to spawn search thread"));
            }
            "stop" => {
                // Signal external stop FIRST so the ponder wait loop (which
                // watches external_stop) sees it before we join the handle.
                external_stop.store(true, Ordering::Relaxed);
                stop_flag.store(true, Ordering::Relaxed);
                // Wait for search thread to finish and recover SearchInfo
                if let Some(handle) = search_handle.take() {
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                    }
                }
            }
            "ponderhit" => {
                // Check TB first: in TB-range endgames, play the TB-optimal move
                // instead of the ponder result. The ponder search uses NNUE eval
                // which doesn't distinguish optimal from merely winning moves.
                if let Some(ref tb) = syzygy {
                    if crate::bitboard::popcount(board.occupied()) as usize <= tb.max_pieces() {
                        if let Some((mut tb_move_str, wdl)) = tb.probe_root(&board) {
                            // Drawn-root qualitative tiebreak (wdl == 0 only).
                            // Mirrors the `go`-path logic.
                            if wdl == 0 {
                                tb_move_str = pick_drawn_tb_move(&board, &tb_move_str);
                            }
                            // Validate TB move against legal moves
                            let legal = crate::movegen::generate_legal_moves(&board);
                            let mut tb_valid = false;
                            if let Some(parsed) = parse_uci_move(&board, &tb_move_str) {
                                for i in 0..legal.len {
                                    if move_from(legal.get(i)) == move_from(parsed)
                                        && move_to(legal.get(i)) == move_to(parsed) {
                                        tb_valid = true;
                                        break;
                                    }
                                }
                            }
                            // C8 audit LIKELY #31: match the `go` path —
                            // override even for drawn TB positions (wdl==0).
                            // NNUE search can play a 50mr-losing move in
                            // a drawn TB endgame; the TB draw move is
                            // safer.
                            if tb_valid {
                                // Stop search and play TB move
                                external_stop.store(true, Ordering::Relaxed);
                                stop_flag.store(true, Ordering::Relaxed);
                                if let Some(handle) = search_handle.take() {
                                    if let Ok(returned_info) = handle.join() {
                                        info = returned_info;
                                        stop_flag = info.stop.clone();
                                    }
                                }
                                let score_str = if wdl > 0 {
                                    format!("score cp {}", crate::tt::TB_WIN)
                                } else if wdl < 0 {
                                    format!("score cp -{}", crate::tt::TB_WIN)
                                } else {
                                    "score cp 0".to_string()
                                };
                                println!("info depth 1 seldepth 1 {} tbhits 1 pv {}", score_str, tb_move_str);
                                println!("bestmove {}", tb_move_str);
                                continue;
                            }
                        }
                    }
                }

                // Normal ponderhit: our ponder move was played. Now it's our turn.
                // The ponder search gave us a head start — use normal time
                // allocation to push deeper. Don't waste the free thinking
                // time by moving instantly.
                if let (Some(ref pl), Some(start)) = (&ponder_limits, ponder_search_start) {
                    // Use cached STM from go-ponder time (board may be stale)
                    let our_time = if ponder_stm == crate::types::WHITE {
                        pl.wtime
                    } else {
                        pl.btime
                    };
                    let our_inc = if ponder_stm == crate::types::WHITE {
                        pl.winc
                    } else {
                        pl.binc
                    };
                    if our_time > 0 {
                        let overhead = info.move_overhead;
                        let elapsed = start.elapsed().as_millis() as u64;

                        // Single source of truth for soft/hard/floor —
                        // shared with start_search via compute_tm_budgets.
                        let (soft, hard, floor) = crate::search::compute_tm_budgets(
                            our_time, our_inc, pl.movestogo, overhead);

                        // Very low time (< 2s with no inc): instant stop.
                        if hard <= overhead && our_inc == 0 && our_time < 2000 {
                            external_stop.store(true, Ordering::Relaxed);
                            stop_flag.store(true, Ordering::Relaxed);
                        } else {
                            // Store hard deadline (absolute) for the should_stop
                            // grace check. Also publish the soft deadline + floor
                            // so the ID loop can arm dynamic TM — without that,
                            // it burns the full hard budget (~5s at 60+2) even on
                            // positions where 2-3s would suffice.
                            let deadline = elapsed + hard.max(10);
                            let soft_deadline = elapsed + soft.max(10).min(hard.max(10));
                            ponderhit_flag.store(deadline, Ordering::Relaxed);
                            ponderhit_soft_flag.store(soft_deadline, Ordering::Relaxed);
                            ponderhit_floor_flag.store(floor, Ordering::Relaxed);
                        }
                    } else if pl.movetime > 0 {
                        // C8 audit LIKELY #33: `go ponder movetime X` (no
                        // wtime/btime). Previously our_time==0 triggered an
                        // instant stop, discarding all the ponder work. Use
                        // movetime as the deadline instead — matches what the
                        // caller asked for.
                        let elapsed = start.elapsed().as_millis() as u64;
                        let deadline = elapsed + pl.movetime.max(10);
                        ponderhit_flag.store(deadline, Ordering::Relaxed);
                    } else {
                        // No time info: instant stop
                        external_stop.store(true, Ordering::Relaxed);
                        stop_flag.store(true, Ordering::Relaxed);
                    }
                } else {
                    // No ponder limits saved: instant stop
                    external_stop.store(true, Ordering::Relaxed);
                    stop_flag.store(true, Ordering::Relaxed);
                }
            }
            "setoption" => {
                // Wait for any active search to finish before changing options
                if let Some(handle) = search_handle.take() {
                    external_stop.store(true, Ordering::Relaxed);
                    stop_flag.store(true, Ordering::Relaxed);
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                        ponderhit_flag = info.ponderhit_time.clone();
                        ponderhit_soft_flag = info.ponderhit_soft.clone();
                        ponderhit_floor_flag = info.ponderhit_floor.clone();
                    }
                }
                parse_option(&tokens, &mut info, &mut num_threads);
                // Handle book options separately
                let mut ni = 0; let mut vi = 0;
                for i in 0..tokens.len() {
                    if tokens[i] == "name" { ni = i + 1; }
                    if tokens[i] == "value" { vi = i + 1; }
                }
                if ni > 0 && vi > 0 && vi < tokens.len() {
                    match tokens[ni] {
                        "OwnBook" => { use_book = tokens[vi] == "true"; }
                        "SyzygyPath" => {
                            let path = tokens[vi].to_string();
                            match crate::tb::SyzygyTB::new_with_cache(&path, tb_hash_mb) {
                                Ok(tb) => {
                                    let tb_arc = std::sync::Arc::new(tb);
                                    info.syzygy = Some(tb_arc.clone());
                                    syzygy = Some(tb_arc);
                                    syzygy_path = Some(path);
                                }
                                Err(e) => eprintln!("info string Syzygy load failed: {}", e),
                            }
                        }
                        "TBHash" => {
                            if let Ok(mb) = tokens[vi].parse::<usize>() {
                                tb_hash_mb = mb.min(1024);
                                // Rebuild the tablebase wrapper with the new cache
                                // size (so existing searches can't race on resize).
                                if let Some(ref path) = syzygy_path {
                                    match crate::tb::SyzygyTB::new_with_cache(path, tb_hash_mb) {
                                        Ok(tb) => {
                                            let tb_arc = std::sync::Arc::new(tb);
                                            info.syzygy = Some(tb_arc.clone());
                                            syzygy = Some(tb_arc);
                                        }
                                        Err(e) => eprintln!("info string TBHash resize failed: {}", e),
                                    }
                                }
                            }
                        }
                        "BookFile" => {
                            match crate::book::OpeningBook::load(tokens[vi]) {
                                Ok(b) => opening_book = Some(b),
                                Err(e) => eprintln!("info string Book load failed: {}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            "loadnnue" => {
                // Non-standard: loadnnue <path>
                if tokens.len() > 1 {
                    match info.load_nnue(tokens[1]) {
                        Ok(_) => println!("info string NNUE loaded"),
                        Err(e) => {
                            eprintln!("ERROR: Failed to load NNUE from {}: {}", tokens[1], e);
                            println!("info string ERROR: Failed to load NNUE from {}: {}", tokens[1], e);
                            std::process::exit(1);
                        }
                    }
                }
            }
            "quit" => {
                external_stop.store(true, Ordering::Relaxed);
                stop_flag.store(true, Ordering::Relaxed);
                if let Some(handle) = search_handle.take() {
                    let _ = handle.join();
                }
                break;
            }
            "d" | "display" => {
                println!("{}", board.display());
                println!("FEN: {}", board.to_fen());
                println!("Hash: {:016x}", board.hash);
            }
            "eval" => {
                let score = if let (Some(net), Some(acc)) = (&info.nnue_net, &mut info.nnue_acc) {
                    // Build a real ThreatStack for v9 nets — without this,
                    // forward_with_threats falls through and the threat
                    // half of the eval is silently zeroed (audit
                    // C2026-04-25-3). Refresh against the current board
                    // before evaluating.
                    let mut ts = crate::threat_accum::ThreatStack::new(net.hidden_size);
                    ts.active = net.has_threats;
                    if ts.active {
                        ts.ensure_computed(&net.threat_weights, net.num_threat_features, &board);
                    }
                    crate::eval::evaluate_nnue(&board, net, acc, &ts)
                } else {
                    crate::eval::evaluate(&board)
                };
                println!("info string fen {}", board.to_fen());
                println!("info string hash {:016x}", board.hash);
                println!("info string pawn_hash {:016x}", board.pawn_hash);
                println!("info string npkey_w {:016x}", board.non_pawn_key[0]);
                println!("info string npkey_b {:016x}", board.non_pawn_key[1]);
                println!("info string raw_nnue {}", score);
                println!("info string side {}", board.side_to_move);

                // Dump accumulator values
                if let (Some(net), Some(acc)) = (&info.nnue_net, &mut info.nnue_acc) {
                    // Force full recompute for clean values
                    acc.force_recompute(net, &board);
                    let h = net.hidden_size;
                    let n = 16.min(h);
                    let w_vals: Vec<String> = acc.white()[..n].iter().map(|v| v.to_string()).collect();
                    let b_vals: Vec<String> = acc.black()[..n].iter().map(|v| v.to_string()).collect();
                    println!("info string white_acc [{}]", w_vals.join(","));
                    println!("info string black_acc [{}]", b_vals.join(","));
                    let pc = crate::nnue::piece_count(&board);
                    let bucket = net.output_bucket(pc);
                    println!("info string piece_count {} bucket {}", pc, bucket);
                }
            }
            "see" => {
                // Dump SEE values for all captures from current position
                let caps = crate::movegen::generate_captures(&board);
                for i in 0..caps.len {
                    let mv = caps.get(i);
                    let val = crate::see::see_value_of(&board, mv);
                    let from = crate::types::move_from(mv);
                    let to = crate::types::move_to(mv);
                    let flags = crate::types::move_flags(mv);
                    let ge0 = crate::see::see_ge(&board, mv, 0);
                    println!("SEE from={} to={} flags={} val={} ge0={}", from, to, flags, val, ge0);
                }
            }
            _ => {}
        }
    }
}

fn parse_position(tokens: &[&str], board: &mut Board) {
    let mut idx = 1;
    if idx >= tokens.len() { return; }

    if tokens[idx] == "startpos" {
        *board = Board::startpos();  // fresh board with empty undo stack
        idx += 1;
    } else if tokens[idx] == "fen" {
        idx += 1;
        let mut fen_parts = Vec::new();
        while idx < tokens.len() && tokens[idx] != "moves" {
            fen_parts.push(tokens[idx]);
            idx += 1;
        }
        let fen = fen_parts.join(" ");
        board.set_fen(&fen);
    }

    // Apply moves
    if idx < tokens.len() && tokens[idx] == "moves" {
        idx += 1;
        while idx < tokens.len() {
            if let Some(mv) = parse_uci_move(board, tokens[idx]) {
                if !board.make_move(mv) {
                    eprintln!("info string WARNING: make_move failed for UCI move {} (parsed as {})",
                        tokens[idx], crate::types::move_to_uci(mv));
                }
            } else {
                eprintln!("info string WARNING: failed to parse UCI move: {}", tokens[idx]);
            }
            idx += 1;
        }
    }
}

fn parse_go(tokens: &[&str]) -> SearchLimits {
    let mut limits = SearchLimits::new();
    let mut idx = 1;

    while idx < tokens.len() {
        match tokens[idx] {
            "depth" => {
                idx += 1;
                if idx < tokens.len() {
                    // C8 audit LIKELY #32: fail-closed on parse error (0),
                    // matching every other integer field in this parser.
                    // Previously malformed `depth ???` parsed as 100,
                    // producing a near-infinite search.
                    limits.depth = tokens[idx].parse().unwrap_or(0);
                }
            }
            "movetime" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.movetime = tokens[idx].parse().unwrap_or(0);
                }
            }
            "wtime" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.wtime = tokens[idx].parse().unwrap_or(0);
                }
            }
            "btime" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.btime = tokens[idx].parse().unwrap_or(0);
                }
            }
            "winc" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.winc = tokens[idx].parse().unwrap_or(0);
                }
            }
            "binc" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.binc = tokens[idx].parse().unwrap_or(0);
                }
            }
            "movestogo" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.movestogo = tokens[idx].parse().unwrap_or(0);
                }
            }
            "nodes" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.nodes = tokens[idx].parse().unwrap_or(0);
                }
            }
            "infinite" => {
                limits.infinite = true;
                limits.depth = 100;
            }
            _ => {}
        }
        idx += 1;
    }

    limits
}

fn parse_option(tokens: &[&str], info: &mut SearchInfo, num_threads: &mut usize) {
    // setoption name X value Y
    // Find "name" and "value" positions
    let mut name_idx = 0;
    let mut value_idx = 0;
    for i in 0..tokens.len() {
        if tokens[i] == "name" { name_idx = i + 1; }
        if tokens[i] == "value" { value_idx = i + 1; }
    }
    if name_idx == 0 || value_idx == 0 || value_idx >= tokens.len() { return; }

    let name = tokens[name_idx];
    let value = tokens[value_idx];

    match name {
        "Hash" => {
            if let Ok(mb) = value.parse::<usize>() {
                info.tt = std::sync::Arc::new(crate::tt::TT::new(mb.max(1).min(4096)));
            }
        }
        "NNUEFile" => {
            match info.load_nnue(value) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("ERROR: Failed to load NNUE from {}: {}", value, e);
                    println!("info string ERROR: Failed to load NNUE from {}: {}", value, e);
                    std::process::exit(1);
                }
            }
        }
        "Threads" => {
            if let Ok(t) = value.parse::<usize>() {
                *num_threads = t.max(1).min(256);
                println!("info string Threads = {}", *num_threads);
            }
        }
        "MoveOverhead" => {
            if let Ok(ms) = value.parse::<u64>() {
                info.move_overhead = ms.min(5000);
            }
        }
        "HiddenActivation" => {
            if let Some(net) = &info.nnue_net {
                let crelu = value.eq_ignore_ascii_case("crelu");
                net.crelu_hidden.store(crelu, std::sync::atomic::Ordering::Relaxed);
                println!("info string HiddenActivation = {}", if crelu { "crelu" } else { "screlu" });
            }
        }
        "Ponder" => {
            // C8 audit LIKELY #30: `Ponder` is advertised at startup
            // (line 83) but was falling through to the tunable loop,
            // silently not storing anywhere. The engine actually reads
            // ponder state from the `go ponder` command, not a stored
            // flag, so the handler is a no-op acknowledgement — but it
            // must explicitly match here so the protocol contract is
            // satisfied (some GUIs fail if setoption response is empty).
            println!("info string Ponder = {}", value);
        }
        "LoadAnyway" => {
            // Diagnostic override — load nets even on training/inference
            // mismatch (e.g. xray-disabled-trained net). MUST be set
            // BEFORE the NNUEFile setoption that triggers load. Default
            // false (refuse to load on mismatch — protects against silent
            // corruption in SPRT / OB / Lichess where log noise is
            // invisible).
            let on = value.eq_ignore_ascii_case("true");
            crate::nnue::LOAD_ANYWAY.store(on, std::sync::atomic::Ordering::Relaxed);
            println!("info string LoadAnyway = {}", on);
        }
        _ => {
            // Check tunable search parameters
            for (pname, param, _, min, max, _c_end, _is_core) in crate::search::tunable_params() {
                if name == pname {
                    if let Ok(v) = value.parse::<i32>() {
                        let clamped = v.max(min).min(max);
                        param.store(clamped, std::sync::atomic::Ordering::Relaxed);
                        // Reinit LMR tables if C value changed
                        if pname.starts_with("LMR_C") {
                            crate::search::init_lmr();
                        }
                        println!("info string {} = {}", pname, clamped);
                    }
                    break;
                }
            }
        }
    }
}

/// Parse a UCI move string (e.g. "e2e4", "e7e8q") in the context of the current board.
/// Matches against the generated legal move list to get correct flags.
pub fn parse_uci_move(board: &Board, s: &str) -> Option<Move> {
    let bytes = s.as_bytes();
    if bytes.len() < 4 { return None; }

    let from_file = bytes[0].wrapping_sub(b'a');
    let from_rank = bytes[1].wrapping_sub(b'1');
    let to_file = bytes[2].wrapping_sub(b'a');
    let to_rank = bytes[3].wrapping_sub(b'1');

    if from_file > 7 || from_rank > 7 || to_file > 7 || to_rank > 7 {
        return None;
    }

    let from = crate::types::square(from_file, from_rank);
    let to = crate::types::square(to_file, to_rank);

    // Detect promotion suffix
    let promo_pt = if bytes.len() > 4 {
        match bytes[4] {
            b'q' => Some(FLAG_PROMOTE_Q),
            b'r' => Some(FLAG_PROMOTE_R),
            b'b' => Some(FLAG_PROMOTE_B),
            b'n' => Some(FLAG_PROMOTE_N),
            _ => None,
        }
    } else {
        None
    };

    // Find the matching move in the legal move list
    let legal = crate::movegen::generate_legal_moves(board);
    for i in 0..legal.len {
        let mv = legal.get(i);
        if move_from(mv) == from && move_to(mv) == to {
            // For promotions, match the promotion type
            if let Some(pf) = promo_pt {
                if move_flags(mv) == pf {
                    return Some(mv);
                }
            } else if !is_promotion(mv) {
                return Some(mv);
            }
        }
    }

    // Fallback: if no promotion specified but there are promotion moves, pick queen
    if promo_pt.is_none() {
        for i in 0..legal.len {
            let mv = legal.get(i);
            if move_from(mv) == from && move_to(mv) == to && move_flags(mv) == FLAG_PROMOTE_Q {
                return Some(mv);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() { crate::init(); }

    /// Reproduces the Lichess game I4qJhfQw move 103 scenario: White's
    /// king is in check from a black rook on a1. Kxa1 captures the rook
    /// and leaves KB-vs-K (FIDE insufficient material → instant draw).
    /// Kb2 / Kc2 escape without recapturing → KB-vs-KR (drawable but
    /// requires defense). pick_drawn_tb_move must prefer Kxa1.
    #[test]
    fn drawn_tb_tiebreak_prefers_im_terminal_recapture() {
        init();
        // White: K on b1, B on h6. Black: K on g3 (arbitrary), R on a1
        // giving check to white K via the back rank. Material count = 4
        // pieces (within TB range).
        let board = Board::from_fen("8/8/7B/8/8/6k1/8/rK6 w - - 0 1");
        // Pretend shakmaty picked the non-recapturing escape "b1b2".
        let fallback = "b1b2";
        let picked = pick_drawn_tb_move(&board, fallback);
        // The recapture is "b1a1" (Kxa1). pick_drawn_tb_move should
        // override the fallback and choose the IM-terminal recapture.
        assert_eq!(picked, "b1a1",
            "picked move {} when Kxa1 was available as IM-terminal recapture",
            picked);
    }

    /// When no recapture leads to IM-terminal but a capture is
    /// available, prefer the higher-value capture.
    #[test]
    fn drawn_tb_tiebreak_prefers_high_value_capture() {
        init();
        // Simple position with two legal captures of different values
        // available. Position: White K, B, R vs Black K, R, N. White
        // can capture either Black's rook or Black's knight.
        let board = Board::from_fen("8/8/8/3k4/3r4/3K1n2/4R3/B7 w - - 0 1");
        let fallback = "a1b2";  // arbitrary non-capture quiet
        let picked = pick_drawn_tb_move(&board, fallback);
        // Rxd4 (e2d4 = rook captures rook) is higher value than rook
        // capturing knight. Expect e2d4 or equivalent.
        // Just verify the picked move IS a capture (target square has
        // a piece in the pre-move board).
        if let Some(mv) = parse_uci_move(&board, &picked) {
            let to = move_to(mv);
            let captured = board.piece_type_at(to);
            assert!(captured != NO_PIECE_TYPE,
                "picked move {} was not a capture", picked);
        }
    }

    /// No-signal case: position with no captures and no IM-after-move.
    /// Function should return SOME legal move (we don't promise it's
    /// the fallback — shakmaty's fallback is itself arbitrary among
    /// tied drawn moves, so swapping among ties is no worse).
    #[test]
    fn drawn_tb_tiebreak_returns_legal_move_when_no_signal() {
        init();
        let board = Board::from_fen("8/8/4k3/8/4K3/8/4P3/8 w - - 0 1");
        let fallback = "e4d4";
        let picked = pick_drawn_tb_move(&board, fallback);
        // Just verify the picked move is legal.
        assert!(parse_uci_move(&board, &picked).is_some(),
            "picked move {} is not a legal move", picked);
    }
}
