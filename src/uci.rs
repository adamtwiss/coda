/// UCI protocol implementation.

use std::io::{self, BufRead};

use crate::board::Board;
use crate::search::*;
use crate::types::*;

pub fn uci_loop() {
    let mut board = Board::startpos();
    let mut info = SearchInfo::new(64); // 64 MB default hash

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
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                info.tt.clear();
                info.history.clear();
                board = Board::startpos();
            }
            "position" => {
                parse_position(&tokens, &mut board);
            }
            "go" => {
                let limits = parse_go(&tokens);
                let best_move = search(&mut board, &mut info, &limits);
                println!("bestmove {}", move_to_uci(best_move));
            }
            "setoption" => {
                parse_option(&tokens, &mut info);
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
        *board = Board::startpos();
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
                board.make_move(mv);
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
        _ => {}
    }
}

/// Parse a UCI move string (e.g. "e2e4", "e7e8q") in the context of the current board.
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

    // Detect promotion
    let promo_flag = if bytes.len() > 4 {
        match bytes[4] {
            b'q' => FLAG_PROMOTE_Q,
            b'r' => FLAG_PROMOTE_R,
            b'b' => FLAG_PROMOTE_B,
            b'n' => FLAG_PROMOTE_N,
            _ => FLAG_NONE,
        }
    } else {
        FLAG_NONE
    };

    if promo_flag != FLAG_NONE {
        return Some(make_move(from, to, promo_flag));
    }

    // Detect special moves by context
    let pt = board.piece_type_at(from);

    // Castling: king moves 2 squares
    if pt == KING {
        let diff = (to as i32 - from as i32).abs();
        if diff == 2 {
            return Some(make_move(from, to, FLAG_CASTLE));
        }
    }

    // En passant
    if pt == PAWN && to == board.ep_square {
        return Some(make_move(from, to, FLAG_EN_PASSANT));
    }

    // Double pawn push
    if pt == PAWN {
        let diff = (to as i32 - from as i32).abs();
        if diff == 16 {
            return Some(make_move(from, to, FLAG_DOUBLE_PUSH));
        }
    }

    Some(make_move(from, to, FLAG_NONE))
}
