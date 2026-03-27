/// UCI protocol implementation.

use std::io::{self, BufRead};

use crate::board::Board;
use crate::search::*;
use crate::types::*;

pub fn uci_loop_with_nnue(nnue_path: Option<&str>, book_path: Option<&str>) {
    let mut board = Board::startpos();
    let mut info = SearchInfo::new(64);
    let mut opening_book: Option<crate::book::OpeningBook> = None;
    let mut use_book = true;

    // Pre-load NNUE if path given via CLI
    if let Some(path) = nnue_path {
        if let Err(e) = info.load_nnue(path) {
            eprintln!("Failed to load NNUE: {}", e);
        }
    }

    // Pre-load opening book if path given via CLI
    if let Some(path) = book_path {
        match crate::book::OpeningBook::load(path) {
            Ok(b) => opening_book = Some(b),
            Err(e) => eprintln!("Failed to load book: {}", e),
        }
    }

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
                println!("option name NNUEFile type string default <empty>");
                println!("option name OwnBook type check default true");
                println!("option name BookFile type string default <empty>");
                println!("option name MoveOverhead type spin default 100 min 0 max 5000");
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
                // Try opening book first
                if use_book {
                    if let Some(ref book) = opening_book {
                        if let Some(book_move) = book.pick_move(&board) {
                            println!("bestmove {}", move_to_uci(book_move));
                            continue;
                        }
                    }
                }
                let limits = parse_go(&tokens);
                let best_move = search(&mut board, &mut info, &limits);
                println!("bestmove {}", move_to_uci(best_move));
            }
            "setoption" => {
                parse_option(&tokens, &mut info);
                // Handle book options separately
                let mut ni = 0; let mut vi = 0;
                for i in 0..tokens.len() {
                    if tokens[i] == "name" { ni = i + 1; }
                    if tokens[i] == "value" { vi = i + 1; }
                }
                if ni > 0 && vi > 0 && vi < tokens.len() {
                    match tokens[ni] {
                        "OwnBook" => { use_book = tokens[vi] == "true"; }
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
                break;
            }
            "d" | "display" => {
                println!("{}", board.display());
                println!("FEN: {}", board.to_fen());
                println!("Hash: {:016x}", board.hash);
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

fn parse_option(tokens: &[&str], info: &mut SearchInfo) {
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
                info.tt = crate::tt::TT::new(mb.max(1).min(4096));
            }
        }
        "NNUEFile" => {
            match info.load_nnue(value) {
                Ok(_) => {}
                Err(e) => eprintln!("info string Failed to load NNUE: {}", e),
            }
        }
        "MoveOverhead" => {
            if let Ok(ms) = value.parse::<u64>() {
                info.move_overhead = ms.min(5000);
            }
        }
        _ => {}
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
