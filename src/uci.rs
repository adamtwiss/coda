/// UCI protocol implementation.

use std::io::{self, BufRead};
use std::sync::atomic::Ordering;

use crate::board::Board;
use crate::search::*;
use crate::types::*;

pub fn uci_loop_with_nnue(nnue_path: Option<&str>, book_path: Option<&str>, classical: bool) {
    let mut board = Board::startpos();
    let mut info = SearchInfo::new(64);
    let mut stop_flag = info.stop.clone(); // keep a handle to signal stop from UCI loop
    let mut ponderhit_flag = info.ponderhit_time.clone(); // shared ponderhit time limit
    let mut ponder_limits: Option<SearchLimits> = None; // pending limits for ponderhit
    let mut opening_book: Option<crate::book::OpeningBook> = None;
    let mut use_book = true;
    let mut syzygy: Option<crate::tb::SyzygyTB> = None;
    let mut num_threads: usize = 1;

    // Pre-load NNUE if path given via CLI, otherwise auto-discover
    if let Some(path) = nnue_path {
        if let Err(e) = info.load_nnue(path) {
            println!("info string Failed to load NNUE from {}: {}", path, e);
        }
    } else {
        let loaded = info.auto_discover_nnue();
        if !loaded && !classical {
            println!("info string WARNING: No NNUE net found. Use 'setoption name NNUEFile value <path>', -nnue flag, or 'coda fetch-net'.");
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
                println!("option name SparseL1 type check default true");
                // Tunable search parameters (for SPSA)
                for (name, _, default, min, max) in crate::search::tunable_params() {
                    println!("option name {} type spin default {} min {} max {}", name, default, min, max);
                }
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                info.tt.clear();
                info.history.clear();
                info.clear_correction_history();
                if let Some(acc) = &mut info.nnue_acc { acc.reset(); }
                board = Board::startpos();
            }
            "position" => {
                parse_position(&tokens, &mut board);
            }
            "go" => {
                // Wait for any pending search to finish first
                if let Some(handle) = search_handle.take() {
                    stop_flag.store(true, Ordering::Relaxed);
                    if let Ok(returned_info) = handle.join() {
                        info = returned_info;
                        stop_flag = info.stop.clone();
                    }
                }
                // Try Syzygy tablebase at root
                if let Some(ref tb) = syzygy {
                    if crate::bitboard::popcount(board.occupied()) as usize <= tb.max_pieces() {
                        if let Some((tb_move, _wdl)) = tb.probe_root(&board) {
                            println!("bestmove {}", tb_move);
                            continue;
                        }
                    }
                }

                // Try opening book first (not in ponder mode)
                let is_ponder = tokens.iter().any(|&t| t == "ponder");
                if use_book && !is_ponder {
                    if let Some(ref book) = opening_book {
                        if let Some(book_move) = book.pick_move(&board) {
                            println!("bestmove {}", move_to_uci(book_move));
                            continue;
                        }
                    }
                }
                let mut limits = parse_go(&tokens);
                if is_ponder {
                    ponder_limits = Some(limits.clone());
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
                ponder_search_start = Some(std::time::Instant::now());
                let mut search_board = board.clone();
                let shared_tt = info.tt.clone();
                let shared_net = info.nnue_net.clone();
                let shared_stop = stop_flag.clone();
                let search_info = std::mem::replace(&mut info, SearchInfo::new_with_shared(
                    shared_stop, shared_tt, shared_net,
                ));
                ponderhit_flag = search_info.ponderhit_time.clone();
                let threads = num_threads;
                search_handle = Some(std::thread::Builder::new()
                    .stack_size(16 * 1024 * 1024)
                    .spawn(move || {
                        let mut si = search_info;
                        let best_move = search_smp(&mut search_board, &mut si, &limits, threads);
                        // Output ponder move from PV if available.
                        // Validate PV[0] matches best_move: if the search was stopped
                        // mid-iteration, the PV may belong to a different root move.
                        let pv_consistent = si.pv_len[0] >= 2
                            && si.pv_table[0][1] != crate::types::NO_MOVE
                            && move_from(si.pv_table[0][0]) == move_from(best_move)
                            && move_to(si.pv_table[0][0]) == move_to(best_move);
                        if pv_consistent {
                            println!("bestmove {} ponder {}", move_to_uci(best_move), move_to_uci(si.pv_table[0][1]));
                        } else {
                            println!("bestmove {}", move_to_uci(best_move));
                        }
                        si // return SearchInfo for reuse
                    }).expect("Failed to spawn search thread"));
            }
            "stop" => {
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
                // Play the ponder result immediately. At fast TC (3+2), the ponder
                // search already had several seconds — the result is deep and good.
                // Giving additional time risks destabilizing deeper iterations.
                // This matched our 2950-rated behavior before the TM changes.
                stop_flag.store(true, Ordering::Relaxed);
            }
            "setoption" => {
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
                            match crate::tb::SyzygyTB::new(tokens[vi]) {
                                Ok(tb) => syzygy = Some(tb),
                                Err(e) => eprintln!("info string Syzygy load failed: {}", e),
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
                        Err(e) => println!("info string NNUE load failed: {}", e),
                    }
                }
            }
            "quit" => {
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
                    crate::eval::evaluate_nnue(&board, net, acc)
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
                    let cur = acc.current();
                    let h = net.hidden_size;
                    let n = 16.min(h);
                    let w_vals: Vec<String> = cur.white[..n].iter().map(|v| v.to_string()).collect();
                    let b_vals: Vec<String> = cur.black[..n].iter().map(|v| v.to_string()).collect();
                    println!("info string white_acc [{}]", w_vals.join(","));
                    println!("info string black_acc [{}]", b_vals.join(","));
                    let pc = crate::nnue::piece_count(&board);
                    let bucket = crate::nnue::output_bucket(pc);
                    println!("info string piece_count {} bucket {}", pc, bucket);
                }
            }
            "see" => {
                // Dump SEE values for all captures from current position
                let caps = crate::movegen::generate_captures(&board);
                for i in 0..caps.len {
                    let mv = caps.moves[i];
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
                    limits.depth = tokens[idx].parse().unwrap_or(100);
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
                Err(e) => println!("info string Failed to load NNUE from {}: {}", value, e),
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
        "SparseL1" => {
            if let Some(net) = &info.nnue_net {
                net.use_sparse_l1.store(value == "true", std::sync::atomic::Ordering::Relaxed);
                println!("info string SparseL1 = {}", value == "true");
            }
        }
        _ => {
            // Check tunable search parameters
            for (pname, param, _, min, max) in crate::search::tunable_params() {
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
fn parse_uci_move(board: &Board, s: &str) -> Option<Move> {
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
        let mv = legal.moves[i];
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
            let mv = legal.moves[i];
            if move_from(mv) == from && move_to(mv) == to && move_flags(mv) == FLAG_PROMOTE_Q {
                return Some(mv);
            }
        }
    }

    None
}
