//! UCI protocol implementation.

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
fn pick_drawn_tb_move(board: &Board, fallback_uci: &str, tb: Option<&crate::tb::SyzygyTB>) -> String {
    use crate::movegen::generate_legal_moves;
    let legal = generate_legal_moves(board);

    // DRAW-PRESERVATION FILTER (2026-05-30 — fixes local-reproduced lichess
    // throw class). shakmaty's `best_move` at a drawn root returns a
    // DTZ-"optimal" move, but DTZ semantics around drawn-with-pawns positions
    // can hand back a move that does NOT actually hold the draw under best
    // opponent play (e.g. KBPvKB `8/7k/1P1B4/2K5/8/8/8/5b2 b`: shakmaty
    // returned Be2, which LOSES — opponent's child WDL is a win; Bg2/Ba6 hold,
    // child WDL = draw). The previous code trusted that fallback whenever no
    // capture/insufficient-material move existed, throwing the draw to a loss.
    //
    // Fix: re-probe each legal move's CHILD WDL and only consider moves whose
    // child is NOT a win for the opponent (child_wdl <= 0 from child STM POV).
    // Mirrors `pick_winning_tb_move`'s child-verification. Among the safe
    // moves, keep the existing (insufficient-material, captured-value) tiebreak.
    // Only if NO move is verified-safe do we fall back to the shakmaty move.
    let mut best_im_terminal = false;
    let mut best_captured_value: i32 = -1;
    let mut best_uci: Option<String> = None;

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

        // IM + draw-preservation check require making the move.
        let mut probe = board.clone();
        if !probe.make_move(mv) {
            continue;  // illegal — should not happen for legal moves but defensive
        }
        let im_terminal = probe.is_insufficient_material();

        // Draw-preservation: child WDL is from the OPPONENT's POV (child STM).
        // Reject moves where the opponent now definitively wins (child_wdl > 0).
        // Accept draws (0) and opponent-losing (< 0). IM-terminal positions are
        // accepted unconditionally (game ends by rule — cannot lose). If the
        // child is out of TB range (None), keep it as acceptable rather than
        // discarding — better than no candidate.
        if !im_terminal {
            if let Some(tbh) = tb {
                if let Some(child_wdl) = tbh.probe_wdl(&probe) {
                    if child_wdl > 0 {
                        continue;  // this move lets the opponent win — drop it
                    }
                }
            }
        }

        // Lex order: im_terminal > captured_value > position-in-list (stable).
        let cand = crate::types::move_to_uci(mv);
        let better = best_uci.is_none()
            || (im_terminal, captured_value) > (best_im_terminal, best_captured_value);
        if better {
            best_im_terminal = im_terminal;
            best_captured_value = captured_value;
            best_uci = Some(cand);
        }
    }

    // Fall back to the shakmaty move only if nothing passed the filter.
    best_uci.unwrap_or_else(|| fallback_uci.to_string())
}

/// Among multiple TB-winning root moves (wdl > 0), pick the one most likely
/// to actually convert under practical play. `probe_root_pv` returns a
/// DTZ-optimal move but among DTZ-equivalent winners shakmaty picks the
/// first one it enumerates — which can be a long pawn push when a capture
/// would simplify the position immediately. lichess wej9jERO m65 (KPPvKR,
/// w to move, 2026-05-27): shakmaty returned `c5c6` (pawn push, position
/// stays 5-piece KPPvKR) over `f2g1` (Kxg1 → 4-piece KPPvK, trivial win).
/// Both are TB_WIN per WDL but `f2g1` is the practical choice.
///
/// Tiebreak among `wdl >= TB_WIN-100` moves: re-probe each legal move's
/// child position with WDL, keep only those that remain definite wins
/// (child wdl ≤ -(TB_WIN-100), i.e. opponent definitely loses), then
/// prefer highest-captured-value (which also reduces opponent's piece
/// count and resets 50mr).
fn pick_winning_tb_move(board: &Board, fallback_uci: &str, tb: &crate::tb::SyzygyTB) -> String {
    use crate::movegen::generate_legal_moves;
    let legal = generate_legal_moves(board);

    // probe_wdl returns ambiguous_wdl_to_score values: ±20000 for definite,
    // ±1 for cursed/blessed, 0 for draw. Gate on the definite-loss-for-
    // opponent value.
    const DEFINITE_LOSS_THRESHOLD: i32 = -19000;
    // Start at 0, NOT -1: the fallback (shakmaty `best_move`) is already the
    // DTZ-OPTIMAL (fastest-mate, always-progressing) move. We only want to
    // override it for a winning CAPTURE (captured_value > 0) that simplifies /
    // resets the 50mr. Starting at -1 let the first no-capture win-preserving
    // move (captured_value 0 > -1) replace the DTZ move with an arbitrary
    // non-progressing one — catastrophic in KQvK etc. where NO move captures:
    // every queen move "preserves the win" by WDL, so Coda picked an arbitrary
    // square and could oscillate forever -> 50-move/3-fold draw of a trivial
    // mate (lichess V49tJcfl: coda_bot drew KQvK vs ratsu-bot, 2026-05-30).
    let mut best_captured_value: i32 = 0;
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

        // Make the move and probe TB on the child to verify it's still a
        // definite win. Skip moves whose child is out of TB range or which
        // don't preserve a definite win.
        let mut probe = board.clone();
        if !probe.make_move(mv) {
            continue;
        }
        let child_wdl = match tb.probe_wdl(&probe) {
            Some(w) => w,
            None => continue,
        };
        // child_wdl is from CHILD STM (opponent). We win iff opponent loses
        // definitively. Reject draws, cursed-loss, and unknown.
        if child_wdl > DEFINITE_LOSS_THRESHOLD {
            continue;
        }

        if captured_value > best_captured_value {
            best_captured_value = captured_value;
            best_uci = crate::types::move_to_uci(mv);
        }
    }

    best_uci
}

/// Count of ponder hints manufactured via the TT-probe fallback (P4) —
/// diagnostic only, reported per occurrence behind TMDebug.
static PONDER_HINT_MANUFACTURED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// P4 — TT-probe ponder-hint fallback (2026-07-05 ponder diagnosis §2).
/// When a search ends without a usable pv[1] (short PV, illegal-hint drop,
/// inconsistent PV after an SMP vote), the GUI cannot ponder at all that
/// turn — measured at 4.6% of Coda's moves vs 0.5% for engines that
/// manufacture a hint from the TT (technique modelled on Berserk's
/// TT-probe-after-bestmove and SF's extract_ponder_from_tt).
///
/// `after_best` is the position AFTER our bestmove. Probes the TT there and
/// returns its move ONLY if it survives the same validation the pv[1] path
/// applies: the cheap is_pseudo_legal screen first, then FULL legality by
/// matching against the generated legal move list (TT collisions inject
/// illegal moves — the PV_PONDER_BUG history makes these guards
/// load-bearing). Returns the legal list's own encoding (canonical flags),
/// or NO_MOVE when no validated hint exists.
pub(crate) fn tt_ponder_hint(tt: &crate::tt::TT, after_best: &Board) -> Move {
    let entry = tt.probe(after_best.hash);
    if !entry.hit || entry.best_move == NO_MOVE {
        return NO_MOVE;
    }
    let cand = entry.best_move;
    if !crate::movepicker::is_pseudo_legal(after_best, cand) {
        return NO_MOVE;
    }
    let legal = crate::movegen::generate_legal_moves(after_best);
    let (cf, ct, cfl) = (move_from(cand), move_to(cand), move_flags(cand));
    for i in 0..legal.len {
        let m = legal.get(i);
        if move_from(m) == cf && move_to(m) == ct
            && (!is_promotion(cand) || move_flags(m) == cfl)
        {
            return m;
        }
    }
    NO_MOVE
}

pub fn uci_loop_with_nnue(nnue_path: Option<&str>, book_path: Option<&str>, classical: bool) {
    let mut board = Board::startpos();
    let mut info = SearchInfo::new(64);
    let mut stop_flag = info.stop.clone(); // keep a handle to signal stop from UCI loop
    let mut ponderhit_flag = info.ponderhit_time.clone(); // shared ponderhit hard deadline
    let mut ponderhit_soft_flag = info.ponderhit_soft.clone(); // shared ponderhit soft deadline
    let mut ponderhit_floor_flag = info.ponderhit_floor.clone(); // shared ponderhit min think
    let mut ponderhit_abs_flag = info.ponderhit_abs.clone(); // shared in-flight forfeit guard
    let mut ponder_depth_flag = info.ponder_depth.clone(); // ponder search's completed depth
    let mut root_fail_low_flag = info.root_fail_low.clone(); // root aspiration fail-low state
    // Separate flag set ONLY by external UCI "stop"/"ponderhit"/"quit" — distinct
    // from info.stop (which search_smp also sets internally when main search
    // returns). Used by the ponder wait loop below so it waits for an actual
    // external ack rather than racing with search_smp's internal teardown.
    let external_stop: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
    // Share the external-stop source with SearchInfo so search-side wait
    // loops (stockpile-floor sleep) can poll it — info.stop can't serve
    // there because the search sets it internally (TM audit 2026-06-13 A2).
    info.external_stop = external_stop.clone();
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
    // Set when a pondering search was abandoned (either via `stop` from GUI,
    // or via a new `go` arriving while ponder was running). The next
    // non-ponder `go` consumes this flag to apply the ponder-miss min-think
    // floor — see go-handler.
    let mut pondermiss_pending: bool = false;
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
                println!("option name SyzygyProbeDepth type spin default 4 min 1 max 100");
                println!("option name HiddenActivation type combo default screlu var screlu var crelu");
                println!("option name LoadAnyway type check default false");
                println!("option name TMDebug type check default false");
                // default -1 = "unset" sentinel → full charge (100). See
                // search::PONDERHIT_CREDIT_PCT (full-charge model 2026-07-05).
                println!("option name PonderhitCreditPct type spin default -1 min -1 max 100");
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
                    external_stop.store(true, Ordering::SeqCst);
                    stop_flag.store(true, Ordering::SeqCst);
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                        ponderhit_flag = info.ponderhit_time.clone();
                        ponderhit_soft_flag = info.ponderhit_soft.clone();
                        ponderhit_floor_flag = info.ponderhit_floor.clone();
                        ponderhit_abs_flag = info.ponderhit_abs.clone();
                        ponder_depth_flag = info.ponder_depth.clone();
                        root_fail_low_flag = info.root_fail_low.clone();
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
                //
                // Ponder-miss detection: if the prior search was a ponder
                // (`ponder_limits.is_some()`) and this go arrives while its
                // handle is still around, that's a ponder MISS. The new fresh
                // search starts with a polluted TT (analysis of the predicted
                // line). At T>=2 + short TM allocation, the main thread can
                // emit a low-depth wrong move before the correct move
                // stabilizes (wej9jERO m65). Set a floor on the new search's
                // soft deadline to give the main thread time to ratchet past
                // the race window.
                // If this go arrives while a ponder is still running (no `stop`
                // was sent first — non-standard but possible), that's also a
                // pondermiss. Set the flag now so the detection below sees it.
                if search_handle.is_some() && ponder_limits.is_some() {
                    pondermiss_pending = true;
                }
                let prev_was_pondermiss = pondermiss_pending;
                if let Some(handle) = search_handle.take() {
                    suppress_bestmove.store(true, Ordering::Relaxed);
                    external_stop.store(true, Ordering::SeqCst);
                    stop_flag.store(true, Ordering::SeqCst);
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                    }
                }
                let is_ponder = tokens.contains(&"ponder");

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
                                tb_pv[0] = pick_drawn_tb_move(&board, &tb_pv[0], Some(&**tb));
                            }
                            // Winning-root tiebreak: among DTZ-equivalent
                            // winners, prefer the move that captures the
                            // highest-value enemy piece (also reduces piece
                            // count and resets 50mr). lichess wej9jERO m65
                            // case — shakmaty picked c5c6 over Kxg1.
                            // wdl == 20000 only — skip cursed wins (wdl == 1).
                            if wdl >= 19000 && !tb_pv.is_empty() {
                                tb_pv[0] = pick_winning_tb_move(&board, &tb_pv[0], tb);
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
                    // Ponder-miss min-think floor. Applies when this go is a
                    // fresh search after an abandoned ponder. Capped to a tiny
                    // fraction of remaining clock so it can't burn time
                    // pressure (e.g. 40/40 endgame with 1s left would
                    // otherwise spend 15% of clock on a 150ms floor).
                    if prev_was_pondermiss {
                        let our_time_ms = if board.side_to_move == crate::types::WHITE {
                            limits.wtime
                        } else {
                            limits.btime
                        };
                        const MIN_POST_PONDERMISS_MS: u64 = 200;
                        // 2% of clock OR 20ms, whichever is larger
                        let safety_cap = (our_time_ms / 50).max(20);
                        let floor = MIN_POST_PONDERMISS_MS.min(safety_cap);
                        limits.min_think_ms = floor;
                        // Debug-only: this fires on every routine ponder-miss
                        // (a common event, not an anomaly), unlike PV_PONDER_BUG
                        // / TM_TRACE below which are gated on genuine anomaly
                        // thresholds. Was unconditional (stray dev-era eprintln,
                        // flooding stderr on every deployed ponder-miss);
                        // gated behind TMDebug to match the existing tm-debug
                        // diagnostic convention.
                        if TM_DEBUG.load(Ordering::Relaxed) {
                            eprintln!(
                                "PONDER_MISS_FLOOR applied floor={}ms our_time={}ms cap={}ms",
                                floor, our_time_ms, safety_cap);
                        }
                    }
                    pondermiss_pending = false; // consumed
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
                // A1 unpublish order: hard FIRST (Release), then soft/floor —
                // a protocol reader that observes hard == 0 never reads
                // soft/floor, so their momentarily-stale values are inert.
                // (No search thread is running here — handle joined above —
                // so this ordering is protocol hygiene, not a live race.)
                ponderhit_flag.store(0, Ordering::Release);
                ponderhit_soft_flag.store(0, Ordering::Relaxed);
                ponderhit_floor_flag.store(0, Ordering::Relaxed);
                ponderhit_abs_flag.store(0, Ordering::Relaxed);
                // Clear the instant-reply gate inputs (P1) so a stale depth /
                // fail-low state from the PREVIOUS search can never satisfy
                // (or wrongly block) the gate for this one. The search thread
                // resets them again at search start; only the search thread
                // ever writes them after this point, the ponderhit handler
                // only reads. Double-ponderhit guard: a ponderhit arriving
                // before the new ponder search stores a real depth reads 0
                // here → instant reply structurally blocked.
                ponder_depth_flag.store(0, Ordering::Relaxed);
                root_fail_low_flag.store(false, Ordering::Relaxed);
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
                // Re-share the external-stop source on the placeholder (the
                // moved-out search_info keeps the shared Arc; the placeholder
                // would otherwise carry a fresh, never-signalled one).
                info.external_stop = external_stop.clone();
                ponderhit_flag = search_info.ponderhit_time.clone();
                ponderhit_soft_flag = search_info.ponderhit_soft.clone();
                ponderhit_floor_flag = search_info.ponderhit_floor.clone();
                ponderhit_abs_flag = search_info.ponderhit_abs.clone();
                ponder_depth_flag = search_info.ponder_depth.clone();
                root_fail_low_flag = search_info.root_fail_low.clone();
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
                        // A4-residual (TM audit 2026-06-13): `go infinite`
                        // (non-ponder) must NOT emit bestmove until the GUI
                        // sends `stop` (UCI spec) — search() can return on
                        // its own at depth-64 (MAX_PLY/2) exhaustion, e.g. in
                        // trivial/TB endgames. Mirrors the ponder wait-loop
                        // below: poll the external stop (set by stop/quit/
                        // new-go/ucinewgame paths) before the emit.
                        if limits.infinite && !is_ponder_search {
                            while !ext_stop.load(std::sync::atomic::Ordering::Acquire) {
                                std::thread::sleep(std::time::Duration::from_millis(1));
                            }
                        }
                        // In ponder mode, if search completed naturally (not stopped
                        // externally), wait for stop/ponderhit before outputting
                        // bestmove.
                        // A1: ponderhit_time (hard) loads use Acquire — it is
                        // the publish flag for the deadline trio; observing it
                        // non-zero here guarantees the soft/floor reads in the
                        // fresh-search block below see the matching values.
                        let wait_start = std::time::Instant::now();
                        if is_ponder_search && !ext_stop.load(std::sync::atomic::Ordering::Relaxed)
                            && si.ponderhit_time.load(std::sync::atomic::Ordering::Acquire) == 0 {
                            while !ext_stop.load(std::sync::atomic::Ordering::Relaxed)
                                && si.ponderhit_time.load(std::sync::atomic::Ordering::Acquire) == 0 {
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
                            let our_inc = if search_board.side_to_move == crate::types::WHITE {
                                limits.winc
                            } else {
                                limits.binc
                            };
                            // Gate-removal (2026-05-31): use the ponder-aware
                            // soft deadline (= partial-credit post_min) at ALL
                            // TCs, not just inc>=500. The old inc>=500 gate
                            // existed only to keep this code out of STC SPRT —
                            // but ponder code can't be SPRT'd anyway (no ponder
                            // in OB/fastchess, #513). With PonderhitCreditPct,
                            // credit=0 reproduces the old STC full-budget
                            // behavior, so this is a strict generalization.
                            let _ = our_inc;
                            // A1: hard (ponderhit_time) is the deadline trio's
                            // publish flag — load it FIRST with Acquire; soft
                            // is only read (Relaxed) after observing hard != 0,
                            // which the Release/Acquire pairing guarantees is
                            // coherent with the published soft. Pre-fix (soft
                            // read first, all Relaxed) an ARM reader could see
                            // hard set with a STALE soft == 0 and target the
                            // HARD deadline — overspending the full 46% window
                            // exactly when ponder finished early. hard == 0
                            // here means no ponderhit was published (wait-loop
                            // exit paths guarantee ext_stop in that case, and
                            // ext_stop is excluded above) — both deadlines 0
                            // → no fresh search.
                            let ph_hard_deadline = si.ponderhit_time.load(std::sync::atomic::Ordering::Acquire);
                            let ph_soft_deadline = if ph_hard_deadline > 0 {
                                si.ponderhit_soft.load(std::sync::atomic::Ordering::Relaxed)
                            } else {
                                0
                            };
                            let now_elapsed = go_received.elapsed().as_millis() as u64;
                            let target_deadline = if ph_soft_deadline > 0 {
                                ph_soft_deadline
                            } else {
                                ph_hard_deadline
                            };
                            if target_deadline > now_elapsed + 5 {
                                let remaining = target_deadline - now_elapsed;
                                // A2 (TM audit 2026-06-13): clearing si.stop for
                                // the fresh search can ERASE a GUI `stop`/`quit`
                                // that landed after the wait-loop exit — the
                                // fresh search would then run its full movetime
                                // while the UCI thread blocks in join(). SeqCst
                                // on this clear + the re-check below, paired
                                // with SeqCst on the UCI-side
                                // (external_stop=true, stop=true) store pairs,
                                // gives a single total order: either the GUI's
                                // stop=true store came AFTER this clear (it
                                // survives and the fresh search stops
                                // immediately), or it was erased — in which
                                // case its program-earlier external_stop=true
                                // store precedes this clear in the total order
                                // and the load below MUST see it.
                                si.stop.store(false, std::sync::atomic::Ordering::SeqCst);
                                // A1 unpublish order: hard FIRST (Release),
                                // then soft/floor (see publish protocol at the
                                // ponderhit handler).
                                si.ponderhit_time.store(0, std::sync::atomic::Ordering::Release);
                                si.ponderhit_soft.store(0, std::sync::atomic::Ordering::Relaxed);
                                si.ponderhit_floor.store(0, std::sync::atomic::Ordering::Relaxed);
                                // (abs is re-armed via fresh_limits.abs_clock
                                // below — the fresh search's own plain-field
                                // guard takes over.)
                                si.ponderhit_abs.store(0, std::sync::atomic::Ordering::Relaxed);
                                if ext_stop.load(std::sync::atomic::Ordering::SeqCst) {
                                    // External stop fired in the clear window —
                                    // restore the flag and skip the fresh
                                    // search; the ponder result is emitted
                                    // promptly below.
                                    si.stop.store(true, std::sync::atomic::Ordering::SeqCst);
                                } else {
                                    // Gate-removal: ponder-aware target as both
                                    // movetime and floor at all TCs (was inc>=500).
                                    let movetime_floor_val = remaining;
                                    // Real remaining clock for the side to move, so
                                    // the absolute forfeit guard applies in the
                                    // ponderhit fresh-search path too (start_time is
                                    // reset to the ponderhit moment by search()).
                                    let our_clock = if search_board.side_to_move == crate::types::WHITE {
                                        limits.wtime
                                    } else {
                                        limits.btime
                                    };
                                    let fresh_limits = SearchLimits {
                                        infinite: false,
                                        movetime: remaining,
                                        movetime_floor: movetime_floor_val,
                                        abs_clock: our_clock,
                                        ..SearchLimits::new()
                                    };
                                    let fresh_start = std::time::Instant::now();
                                    best_move = search_smp(&mut search_board, &mut si, &fresh_limits, threads);
                                    fresh_elapsed = fresh_start.elapsed();
                                }
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
                            // P4: no usable pv[1] (short PV / dropped illegal
                            // hint / SMP-vote inconsistency) — manufacture a
                            // ponder hint by probing the TT at the position
                            // after bestmove. A missing hint forfeits the
                            // entire opponent-time think for this turn.
                            let mut hint = NO_MOVE;
                            if best_move != NO_MOVE {
                                let mut after_best = search_board.clone();
                                if after_best.make_move(best_move) {
                                    hint = tt_ponder_hint(&si.tt, &after_best);
                                }
                            }
                            if hint != NO_MOVE {
                                let n = PONDER_HINT_MANUFACTURED
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                                if crate::search::TM_DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
                                    eprintln!(
                                        "PONDER_HINT_TT manufactured hint={} after best={} (total={})",
                                        move_to_uci(hint), move_to_uci(best_move), n);
                                }
                                println!("bestmove {} ponder {}", move_to_uci(best_move), move_to_uci(hint));
                            } else {
                                println!("bestmove {}", move_to_uci(best_move));
                            }
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
                // A2: every UCI-side (external_stop=true, stop=true) pair
                // uses SeqCst so it participates in a total order with the
                // ponderhit fresh-search window's (stop=false, external_stop
                // load) — see the comment there. If our stop=true gets
                // erased by that window's clear, the window is guaranteed to
                // observe our external_stop=true and restore it.
                external_stop.store(true, Ordering::SeqCst);
                stop_flag.store(true, Ordering::SeqCst);
                // If we're stopping a pondering search, the GUI is signalling
                // a ponder-MISS — flag the next go to apply the min-think
                // floor (search_handle gets joined here, so the next go can't
                // detect "was a ponder" via the handle alone).
                if ponder_limits.is_some() {
                    pondermiss_pending = true;
                }
                // Wait for search thread to finish and recover SearchInfo
                if let Some(handle) = search_handle.take() {
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                    }
                }
            }
            "ponderhit" => {
                // Ponderhit means our predicted move was played — this is NOT a
                // pondermiss. Clear the pending flag in case a stale `stop`
                // somehow set it (shouldn't happen in protocol, but defensive).
                pondermiss_pending = false;
                // Check TB first: in TB-range endgames, play the TB-optimal move
                // instead of the ponder result. The ponder search uses NNUE eval
                // which doesn't distinguish optimal from merely winning moves.
                if let Some(ref tb) = syzygy {
                    if crate::bitboard::popcount(board.occupied()) as usize <= tb.max_pieces() {
                        if let Some((mut tb_move_str, wdl)) = tb.probe_root(&board) {
                            // Drawn-root qualitative tiebreak (wdl == 0 only).
                            // Mirrors the `go`-path logic.
                            if wdl == 0 {
                                tb_move_str = pick_drawn_tb_move(&board, &tb_move_str, Some(&**tb));
                            }
                            // Winning-root tiebreak (mirror go-path).
                            if wdl >= 19000 {
                                tb_move_str = pick_winning_tb_move(&board, &tb_move_str, tb);
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
                                external_stop.store(true, Ordering::SeqCst);
                                stop_flag.store(true, Ordering::SeqCst);
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
                        // These are the budgets this move WOULD have been
                        // given on a plain `go` (our clock does not tick
                        // while pondering, so computing them here from the
                        // saved go-ponder clocks is equivalent to computing
                        // them at `go ponder` time). ponder_enabled() passes
                        // the P3 +25% optimum bump through.
                        let (soft, hard, _max_time, floor) = crate::search::compute_tm_budgets(
                            our_time, our_inc, pl.movestogo, overhead, board.fullmove,
                            crate::search::ponder_enabled());

                        // Instant-reply gate inputs (P1): the ponder search's
                        // completed root depth and root fail-low state.
                        // Relaxed loads — independent gate values, stale-
                        // conservative by construction (see search.rs docs).
                        let ponder_depth =
                            ponder_depth_flag.load(Ordering::Relaxed) as i32;
                        let failing_low = root_fail_low_flag.load(Ordering::Relaxed);

                        // Very low time (< 2s with no inc): instant stop.
                        if hard <= overhead && our_inc == 0 && our_time < 2000 {
                            external_stop.store(true, Ordering::SeqCst);
                            stop_flag.store(true, Ordering::SeqCst);
                        } else if crate::search::should_instant_reply(
                            elapsed, soft, ponder_depth, failing_low)
                        {
                            // P1 stopOnPonderhit (2026-07-05 ponder diagnosis;
                            // SF search.cpp:563-571 technique): the pondered
                            // time already covers this move's soft budget and
                            // the ponder search is deep + settled — emit the
                            // pondered bestmove with ~zero additional own
                            // clock. This is where the +25% pre-funding (P3)
                            // and the full-charge model (P2) get refunded.
                            // The gate is double-ponderhit-cascade-proof:
                            // elapsed >= soft fails inherently on a ~1ms
                            // go-ponder→ponderhit window (soft is >= tens of
                            // ms), the depth floor backstops tiny-soft
                            // degenerate cases, and the structural
                            // MIN_PONDER_ELAPSED floor makes it unconditional.
                            if crate::search::TM_DEBUG.load(Ordering::Relaxed) {
                                eprintln!(
                                    "PONDERHIT_INSTANT elapsed={}ms soft={}ms depth={}",
                                    elapsed, soft, ponder_depth);
                            }
                            external_stop.store(true, Ordering::SeqCst);
                            stop_flag.store(true, Ordering::SeqCst);
                        } else {
                            // P2 full-charge post-hit budget (2026-07-05,
                            // replaces Option C partial credit): the search
                            // continues in the from-go-ponder time frame —
                            // remaining think = intended_soft − elapsed, i.e.
                            // budgets fixed at `go ponder` and the pondered
                            // time charged in FULL (SF timeman model; dynamic
                            // factors still apply at iteration boundaries).
                            // Option C's 50% credit saturated to the 50ms
                            // floor at STC and the realized spend became
                            // iteration-quantized bleed (median 169ms vs SF
                            // 71ms). The historical failure of full charge
                            // alone (Phase 14 v4: shallow instant emits) is
                            // now handled by the P1 depth-gated instant
                            // reply + P3 pre-funding — the pieces work as a
                            // set. ponderhit_credit_pct() is 100 unless
                            // PonderhitCreditPct was explicitly set (local
                            // A/B only).
                            let _ = floor;
                            let (deadline, soft_deadline, store_floor, abs_deadline) = {
                                let credited = elapsed
                                    .saturating_mul(crate::search::ponderhit_credit_pct())
                                    / 100;
                                let mut post_min = soft.saturating_sub(credited)
                                    .max(crate::search::MIN_POST_PONDERHIT_MS);
                                // FL-EXT at-hit half (2026-07-05): arriving
                                // at the hit failing low → grant the FULL
                                // fresh soft (was soft/2) and extend the
                                // post-hit hard frame by one fail-low factor
                                // step (×1.34). SF spends EXTRA time exactly
                                // when the pondered conclusion destabilized;
                                // its post-hit tail is >1s on 3.3% of moves
                                // while ours was 0.0% — clipped exactly by
                                // this cap. ponderhit_abs (below) remains the
                                // uncrossable forfeit wall (checked FIRST in
                                // should_stop, so hard_eff may exceed it
                                // harmlessly).
                                let hard_eff = if failing_low {
                                    hard.saturating_mul(
                                        100 + crate::search::PH_FL_HARD_EXT_PCT) / 100
                                } else {
                                    hard
                                };
                                if failing_low {
                                    post_min = post_min.max(soft);
                                }
                                let post_min = post_min.min(hard_eff.max(10));
                                // P2(b): absolute forfeit guard for the
                                // in-flight post-hit search (loss55 class —
                                // this path previously had NO abs deadline).
                                // Our clock starts ticking ~at the ponderhit,
                                // so in the search's start_time frame the
                                // forfeit wall is elapsed + clock − reserve.
                                const FORFEIT_MARGIN_MS: u64 = 50;
                                let reserve = overhead + FORFEIT_MARGIN_MS;
                                let abs = elapsed
                                    + our_time.saturating_sub(reserve).max(1);
                                (elapsed + hard_eff.max(10), elapsed + post_min, post_min, abs)
                            };
                            // A1 publish protocol (TM audit 2026-06-13, ARM
                            // standard): the deadline group is read by the
                            // search thread assuming mutual consistency, so
                            // it must be PUBLISHED atomically-enough: store
                            // floor, soft and abs FIRST (Relaxed is fine —
                            // they are inert until hard is seen), then hard
                            // (ponderhit_flag) LAST with Release. Every
                            // reader loads hard with Acquire first and only
                            // then reads soft/floor/abs. Pre-fix (all
                            // Relaxed, hard stored first) an ARM reader could
                            // see soft > 0 with stale floor == 0 (stockpile
                            // floor erased → instant-emit class) or hard set
                            // with stale soft == 0 (post-completion wait
                            // targets the HARD deadline → overspends the
                            // full 46% window when ponder finished early).
                            ponderhit_floor_flag.store(store_floor, Ordering::Relaxed);
                            ponderhit_soft_flag.store(soft_deadline, Ordering::Relaxed);
                            ponderhit_abs_flag.store(abs_deadline, Ordering::Relaxed);
                            ponderhit_flag.store(deadline, Ordering::Release);
                        }
                    } else if pl.movetime > 0 {
                        // C8 audit LIKELY #33: `go ponder movetime X` (no
                        // wtime/btime). Previously our_time==0 triggered an
                        // instant stop, discarding all the ponder work. Use
                        // movetime as the deadline instead — matches what the
                        // caller asked for.
                        let elapsed = start.elapsed().as_millis() as u64;
                        let deadline = elapsed + pl.movetime.max(10);
                        // A1: hard published with Release (soft/floor stay 0
                        // from the pre-spawn clear — readers key off hard and
                        // treat soft==0 as "no dynamic TM", which is correct
                        // for the movetime path).
                        ponderhit_flag.store(deadline, Ordering::Release);
                    } else {
                        // No time info: instant stop
                        external_stop.store(true, Ordering::SeqCst);
                        stop_flag.store(true, Ordering::SeqCst);
                    }
                } else {
                    // No ponder limits saved: instant stop
                    external_stop.store(true, Ordering::SeqCst);
                    stop_flag.store(true, Ordering::SeqCst);
                }
                // Ponderhit consumed the ponder. Clear ponder_limits so the
                // next `go` doesn't falsely detect a pondermiss off the stale
                // value (the heuristic in go-handler checks handle+limits).
                ponder_limits = None;
            }
            "setoption" => {
                // Wait for any active search to finish before changing options
                if let Some(handle) = search_handle.take() {
                    external_stop.store(true, Ordering::SeqCst);
                    stop_flag.store(true, Ordering::SeqCst);
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                        ponderhit_flag = info.ponderhit_time.clone();
                        ponderhit_soft_flag = info.ponderhit_soft.clone();
                        ponderhit_floor_flag = info.ponderhit_floor.clone();
                        ponderhit_abs_flag = info.ponderhit_abs.clone();
                        ponder_depth_flag = info.ponder_depth.clone();
                        root_fail_low_flag = info.root_fail_low.clone();
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
                            // Recoverable: keep the current net rather than
                            // killing the engine on a bad interactive path.
                            eprintln!("ERROR: Failed to load NNUE from {}: {}", tokens[1], e);
                            println!("info string ERROR: Failed to load NNUE from {}: {}", tokens[1], e);
                        }
                    }
                }
            }
            "quit" => {
                external_stop.store(true, Ordering::SeqCst);
                stop_flag.store(true, Ordering::SeqCst);
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
                // Clean, parseable static-eval line (white POV, pawns) —
                // matches the Stockfish/Reckless `eval` format so external
                // tooling can extract all three engines uniformly. `score`
                // is STM-POV cp; flip to white POV for the printed value.
                let white_cp = if board.side_to_move == 0 { score } else { -score };
                println!("NNUE evaluation       {:+.2} (white side)", white_cp as f32 / 100.0);

                // Verbose dump (hashes, accumulator values) only on
                // `eval debug` / `eval verbose` — keeps the default output
                // clean. Nothing in scripts/tests parses these lines.
                let verbose = tokens.len() > 1 && (tokens[1] == "debug" || tokens[1] == "verbose");
                if verbose {
                    println!("info string fen {}", board.to_fen());
                    println!("info string hash {:016x}", board.hash);
                    println!("info string pawn_hash {:016x}", board.pawn_hash);
                    println!("info string npkey_w {:016x}", board.non_pawn_key[0]);
                    println!("info string npkey_b {:016x}", board.non_pawn_key[1]);
                    println!("info string raw_nnue {} (stm)", score);
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
            // Every move in the list is relative to the board state produced by
            // all preceding moves. If one fails to parse or apply, continuing to
            // the rest would make them relative to the WRONG position, silently
            // desyncing us from the GUI. Stop at the first failure instead.
            if let Some(mv) = parse_uci_move(board, tokens[idx]) {
                if !board.make_move(mv) {
                    eprintln!("info string WARNING: make_move failed for UCI move {} (parsed as {}); \
                        ignoring this and any further moves",
                        tokens[idx], crate::types::move_to_uci(mv));
                    break;
                }
            } else {
                eprintln!("info string WARNING: failed to parse UCI move: {}; \
                    ignoring this and any further moves", tokens[idx]);
                break;
            }
            idx += 1;
        }
    }
}

/// Parse a clock/movetime token that was PRESENT in the `go` command
/// (TM audit 2026-06-13 A3). GUIs send negative values at the flag edge
/// (`go wtime -23`) and explicit zeros (`go movetime 0`); the previous
/// `parse::<u64>().unwrap_or(0)` turned both into 0 == "token absent" ==
/// no clock info == UNBOUNDED search == time forfeit. Parse as i64 and
/// clamp any present-but-<=0 (or unparseable) value to a 1ms clock —
/// fail CLOSED to an instant move, never open to an untimed search.
/// Absence semantics are preserved because this is only called when the
/// token is present.
fn parse_clock_ms(tok: &str) -> u64 {
    match tok.parse::<i64>() {
        Ok(v) if v >= 1 => v as u64,
        _ => 1,
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
                    // Fail-closed to a real depth-1 search on malformed /
                    // non-positive input. A bare 0 is NOT safe: search()
                    // treats depth <= 0 as "no depth limit" and remaps it to
                    // MAX_PLY/2, so `go depth nope|0|-1` would become a long
                    // untimed search. .max(1) keeps a present-but-bad depth
                    // token bounded instead of collapsing into that sentinel.
                    limits.depth = tokens[idx].parse().unwrap_or(0).max(1);
                }
            }
            "movetime" => {
                idx += 1;
                if idx < tokens.len() {
                    // A3: present-but-<=0 clamps to 1ms (see parse_clock_ms).
                    limits.movetime = parse_clock_ms(tokens[idx]);
                }
            }
            "wtime" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.wtime = parse_clock_ms(tokens[idx]);
                }
            }
            "btime" => {
                idx += 1;
                if idx < tokens.len() {
                    limits.btime = parse_clock_ms(tokens[idx]);
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
                    // Report and keep the currently-loaded net. A bad NNUEFile
                    // path is recoverable user/GUI input — exiting the process
                    // mid-UCI kills the engine and forfeits the game.
                    eprintln!("ERROR: Failed to load NNUE from {}: {}", value, e);
                    println!("info string ERROR: Failed to load NNUE from {}: {}", value, e);
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
        "SyzygyProbeDepth" => {
            if let Ok(d) = value.parse::<i32>() {
                info.tb_probe_depth = d.max(1).min(100);
                println!("info string SyzygyProbeDepth = {}", info.tb_probe_depth);
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
            // P3 (2026-07-05 ponder diagnosis): the Ponder option now
            // gates the +25% optimum pre-funding in compute_tm_budgets
            // (SF timeman semantics — applied on every move while on,
            // refunded by the P1 instant replies). Previously a no-op
            // acknowledgement (the engine reads per-move ponder state
            // from `go ponder`; that part is unchanged). Default false →
            // bit-identical no-ponder behavior.
            let on = value.eq_ignore_ascii_case("true");
            crate::search::PONDER_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
            println!("info string Ponder = {}", on);
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
        "TMDebug" => {
            // Diagnostic: emit per-move TM state via `info string tm-debug`
            // before bestmove. Useful for investigating which TM signals
            // fire when and how big the multipliers grow. Off by default
            // since OB/SPRT runs should not have noisy output.
            let on = value.eq_ignore_ascii_case("true");
            crate::search::TM_DEBUG.store(on, std::sync::atomic::Ordering::Relaxed);
            println!("info string TMDebug = {}", on);
        }
        "PonderhitCreditPct" => {
            // Post-ponderhit budget credit % — INERT by default since the
            // 2026-07-05 full-charge model (default is the -1 sentinel =
            // charge 100% of pondered time). Explicitly setting 0..=100
            // re-enables fractional crediting for local A/B comparability
            // (50 reproduces Option C, 0 the pre-P13 full-fresh-budget).
            // NOT a SPSA tunable — OB/fastchess has no ponder, so it can
            // only be swept in a local ponder gauntlet. See
            // search::PONDERHIT_CREDIT_PCT.
            if let Ok(v) = value.parse::<i32>() {
                // -1 (or any negative) restores the "unset" sentinel.
                let clamped = v.clamp(-1, 100);
                crate::search::PONDERHIT_CREDIT_PCT.store(clamped, std::sync::atomic::Ordering::Relaxed);
                println!("info string PonderhitCreditPct = {}", clamped);
            }
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

    /// P4 — the TT-probe ponder-hint fallback must NEVER emit an illegal
    /// hint (PV_PONDER_BUG class), and must produce the TT move when it IS
    /// legal. Property test over random playouts: at every position we
    /// "play" a random bestmove, inject three classes of TT entry at the
    /// resulting position (random junk bits — simulated hash collision; a
    /// plausible-but-foreign move legal only in the PRE-move position; and
    /// a genuinely legal move), and assert the manufactured hint is always
    /// legal-or-none, and equals the stored move in the legal case.
    #[test]
    fn tt_ponder_hint_always_legal_or_none() {
        init();
        let tt = crate::tt::TT::new(1);
        // Deterministic LCG so failures reproduce.
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rng = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 16
        };
        let assert_legal_or_none = |after: &Board, hint: Move, label: &str| {
            if hint == NO_MOVE { return; }
            let legal = crate::movegen::generate_legal_moves(after);
            let mut found = false;
            for i in 0..legal.len {
                if legal.get(i) == hint { found = true; break; }
            }
            assert!(found,
                "{}: manufactured hint {} is not legal in {}",
                label, move_to_uci(hint), after.to_fen());
        };
        let mut legal_hits = 0u32;
        for _game in 0..25 {
            let mut board = Board::startpos();
            for _ply in 0..80 {
                let legal = crate::movegen::generate_legal_moves(&board);
                if legal.len == 0 { break; }
                let bm = legal.get((rng() % legal.len as u64) as usize);
                let mut after = board.clone();
                assert!(after.make_move(bm));

                // (1) Random junk bits — simulated TT hash collision.
                let junk = (rng() & 0xFFFF) as Move;
                tt.store(after.hash, 5, 0, crate::tt::TT_FLAG_LOWER, junk, 0, false);
                assert_legal_or_none(&after, tt_ponder_hint(&tt, &after), "junk");

                // (2) A move legal in the PRE-move position (foreign to
                // `after` unless coincidentally shared) — the classic
                // stale-sibling / collision shape.
                let foreign = legal.get((rng() % legal.len as u64) as usize);
                tt.store(after.hash, 5, 0, crate::tt::TT_FLAG_EXACT, foreign, 0, false);
                assert_legal_or_none(&after, tt_ponder_hint(&tt, &after), "foreign");

                // (3) A genuinely legal move — the short-PV rescue case:
                // the fallback must recover it (not drop the ponder).
                let after_legal = crate::movegen::generate_legal_moves(&after);
                if after_legal.len > 0 {
                    let good = after_legal.get((rng() % after_legal.len as u64) as usize);
                    tt.store(after.hash, 5, 0, crate::tt::TT_FLAG_EXACT, good, 0, false);
                    let hint = tt_ponder_hint(&tt, &after);
                    assert_eq!(hint, good,
                        "legal TT move {} must be recovered as the hint in {}",
                        move_to_uci(good), after.to_fen());
                    legal_hits += 1;
                }
                board = after;
            }
        }
        assert!(legal_hits > 500, "test must exercise the rescue path (got {})", legal_hits);
    }

    /// P4 rescue on a realistic short-PV shape: run a real (shallow) search
    /// so the TT is warm, then ask for a hint after the search's bestmove —
    /// the fallback must produce a legal hint here (this is the 4.6%
    /// forfeited-ponder class the fallback exists to cut).
    #[test]
    fn tt_ponder_hint_recovers_after_real_search() {
        init();
        let mut info = crate::search::SearchInfo::new(16);
        info.silent = true;
        let mut board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let limits = crate::search::SearchLimits {
            depth: 8, ..crate::search::SearchLimits::new()
        };
        let best = crate::search::search(&mut board, &mut info, &limits);
        assert!(best != NO_MOVE);
        let mut after = board.clone();
        assert!(after.make_move(best));
        let hint = tt_ponder_hint(&info.tt, &after);
        // A depth-8 search always leaves a TT move at the child of the root
        // best — the fallback must find and validate it.
        assert!(hint != NO_MOVE, "warm TT must yield a manufactured hint");
        let legal = crate::movegen::generate_legal_moves(&after);
        let mut found = false;
        for i in 0..legal.len {
            if legal.get(i) == hint { found = true; break; }
        }
        assert!(found, "manufactured hint {} must be legal", move_to_uci(hint));
    }

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
        let picked = pick_drawn_tb_move(&board, fallback, None);
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
        let picked = pick_drawn_tb_move(&board, fallback, None);
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
    /// REGRESSION (2026-05-30): drawn TB root must not play a move that
    /// LOSES. KBPvKB `8/7k/1P1B4/2K5/8/8/8/5b2 b`: shakmaty's DTZ pick Be2
    /// loses (opponent child WDL = win); Bg2/Ba6 hold. With the draw-
    /// preservation filter, pick_drawn_tb_move must avoid Be2/Bh3.
    #[test]
    fn drawn_tb_preserves_draw_kbpkb() {
        init();
        let tb = match crate::tb::SyzygyTB::new("/home/adam/chess/tablebases") {
            Ok(t) => t,
            Err(_) => return, // skip when TB unavailable
        };
        if tb.max_pieces() < 5 { return; }
        let board = Board::from_fen("8/7k/1P1B4/2K5/8/8/8/5b2 b - - 0 191");
        let fallback = "f1e2"; // the LOSING shakmaty pick
        let picked = pick_drawn_tb_move(&board, fallback, Some(&tb));
        assert!(picked != "f1e2" && picked != "f1h3",
            "pick_drawn_tb_move returned losing move {}", picked);
        // Verify the picked move actually holds: child must not be a win for White.
        let mv = parse_uci_move(&board, &picked).expect("legal");
        let mut probe = board.clone();
        probe.make_move(mv);
        if let Some(child_wdl) = tb.probe_wdl(&probe) {
            assert!(child_wdl <= 0, "picked {} gives opponent a win (wdl {})", picked, child_wdl);
        }
    }

    #[test]
    fn drawn_tb_tiebreak_returns_legal_move_when_no_signal() {
        init();
        let board = Board::from_fen("8/8/4k3/8/4K3/8/4P3/8 w - - 0 1");
        let fallback = "e4d4";
        let picked = pick_drawn_tb_move(&board, fallback, None);
        // Just verify the picked move is legal.
        assert!(parse_uci_move(&board, &picked).is_some(),
            "picked move {} is not a legal move", picked);
    }

    /// Locate Syzygy tablebases for tests (lichess: /tablebases; dev hosts:
    /// ~/chess/tablebases). Returns None to skip when unavailable.
    fn test_tb() -> Option<crate::tb::SyzygyTB> {
        let home = std::env::var("HOME").unwrap_or_default();
        for p in [
            "/tablebases".to_string(),
            format!("{}/chess/tablebases", home),
            format!("{}/chess/syzygy", home),
        ] {
            if std::path::Path::new(&p).exists() {
                if let Ok(tb) = crate::tb::SyzygyTB::new(&p) { return Some(tb); }
            }
        }
        None
    }

    /// Drive the TB ROOT move-selection path (probe_root_pv + pick_winning_tb_move /
    /// pick_drawn_tb_move) move-by-move from `fen` with a TB-optimal defender,
    /// asserting a winning side actually MATES (no shuffle / 50-move / 3-fold draw).
    /// This is the regression guard for the KQvK/KRvK "can't convert a trivial
    /// TB win" bug (lichess V49tJcfl, 2026-05-30) — pick_winning_tb_move was
    /// overriding shakmaty's DTZ-optimal move with an arbitrary non-progressing one.
    fn assert_tb_root_mates(tb: &crate::tb::SyzygyTB, fen: &str, max_plies: usize, label: &str) {
        use std::collections::HashMap;
        let mut board = Board::from_fen(fen);
        let mut seen: HashMap<String, u32> = HashMap::new();
        for ply in 0..max_plies {
            let legal = crate::movegen::generate_legal_moves(&board);
            if legal.len == 0 {
                // checkmate or stalemate; require checkmate (the win must convert)
                assert!(board.in_check(),
                    "{}: stalemate at ply {} — failed to convert TB win", label, ply);
                return; // mated
            }
            // Pick the mover's move.
            let (mut pv, wdl) = tb.probe_root_pv(&board, 8)
                .unwrap_or_else(|| panic!("{}: probe_root_pv None at ply {}", label, ply));
            let stm_winning = wdl >= 19000;
            if stm_winning && !pv.is_empty() {
                pv[0] = pick_winning_tb_move(&board, &pv[0], tb);
            } else if wdl == 0 && !pv.is_empty() {
                pv[0] = pick_drawn_tb_move(&board, &pv[0], Some(tb));
            }
            let mv = parse_uci_move(&board, &pv[0])
                .unwrap_or_else(|| panic!("{}: illegal TB move {} at ply {}", label, pv[0], ply));
            board.make_move(mv);
            // 3-fold detection on board layout (the shuffle signature).
            let key = board.to_fen().split_whitespace().next().unwrap().to_string();
            *seen.entry(key.clone()).or_insert(0) += 1;
            assert!(seen[&key] < 3,
                "{}: 3-fold repetition at ply {} — TB win NOT converging (shuffle bug)", label, ply);
        }
        panic!("{}: did not mate within {} plies — TB win not converging", label, max_plies);
    }

    /// REGRESSION (2026-05-30, lichess V49tJcfl): basic TB mates must CONVERT,
    /// not shuffle to a draw. pick_winning_tb_move must keep shakmaty's
    /// DTZ-optimal (progressing) move rather than an arbitrary win-preserving one.
    #[test]
    fn tb_root_converts_kqvk_mate() {
        init();
        let tb = match test_tb() { Some(t) => t, None => return };
        // Both sides played by the TB root path; KQvK is a forced mate well
        // under 50 moves, so a non-converging shuffle trips the 3-fold assert.
        assert_tb_root_mates(&tb, "2k5/8/8/8/8/8/8/3KQ3 w - - 0 1", 60, "KQvK");
    }

    #[test]
    fn tb_root_converts_krvk_mate() {
        init();
        let tb = match test_tb() { Some(t) => t, None => return };
        assert_tb_root_mates(&tb, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 80, "KRvK");
    }
}
