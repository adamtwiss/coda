/// EPD file loading and test suite runner.

use std::fs;
use std::time::Instant;

use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::search::*;
use crate::types::*;

/// A single EPD test position.
pub struct EpdPosition {
    pub fen: String,
    pub best_moves: Vec<String>,  // "bm" field: correct moves in SAN or coordinate
    pub avoid_moves: Vec<String>, // "am" field: moves to avoid
    pub id: String,
}

/// Parse an EPD file into positions.
pub fn parse_epd(path: &str) -> Vec<EpdPosition> {
    let content = fs::read_to_string(path).expect("Failed to read EPD file");
    let mut positions = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(pos) = parse_epd_line(line) {
            positions.push(pos);
        }
    }

    positions
}

fn parse_epd_line(line: &str) -> Option<EpdPosition> {
    // EPD format: <FEN without move counters> <operations>
    // FEN has 4 fields: position, side, castling, ep
    // Then operations like: bm Rg3; id "WAC.003";

    let parts: Vec<&str> = line.splitn(5, ' ').collect();
    if parts.len() < 4 {
        return None;
    }

    // The FEN is the first 4 fields + default move counters
    let fen = format!("{} {} {} {} 0 1", parts[0], parts[1], parts[2], parts[3]);

    // Parse operations (everything after the 4th field)
    let ops_str = if parts.len() > 4 { parts[4] } else { "" };

    let mut best_moves = Vec::new();
    let mut avoid_moves = Vec::new();
    let mut id = String::new();

    // Split by semicolon for separate operations
    for op in ops_str.split(';') {
        let op = op.trim();
        if op.is_empty() { continue; }

        if op.starts_with("bm ") {
            let moves_str = &op[3..];
            for m in moves_str.split_whitespace() {
                best_moves.push(m.trim_end_matches(';').to_string());
            }
        } else if op.starts_with("am ") {
            let moves_str = &op[3..];
            for m in moves_str.split_whitespace() {
                avoid_moves.push(m.trim_end_matches(';').to_string());
            }
        } else if op.starts_with("id ") {
            id = op[3..].trim_matches('"').to_string();
        }
    }

    Some(EpdPosition {
        fen,
        best_moves,
        avoid_moves,
        id,
    })
}

/// Convert a move to SAN notation for comparison with EPD best moves.
pub fn move_to_san(board: &Board, mv: Move) -> String {
    let from = move_from(mv);
    let to = move_to(mv);
    let flags = move_flags(mv);
    let pt = board.piece_type_at(from);

    // Castling
    if flags == FLAG_CASTLE {
        return if to > from { "O-O".to_string() } else { "O-O-O".to_string() };
    }

    let mut san = String::new();

    // Piece letter (not for pawns)
    if pt != PAWN {
        san.push(match pt {
            KNIGHT => 'N',
            BISHOP => 'B',
            ROOK => 'R',
            QUEEN => 'Q',
            KING => 'K',
            _ => '?',
        });
    }

    // Disambiguation: check if another piece of same type can go to the same square
    if pt != PAWN && pt != KING {
        let legal = generate_legal_moves(board);
        let mut same_piece_to = false;
        let mut same_file = false;
        let mut same_rank = false;

        for i in 0..legal.len {
            let other = legal.moves[i];
            if other == mv { continue; }
            let other_from = move_from(other);
            let other_to = move_to(other);
            if other_to != to { continue; }
            if board.piece_type_at(other_from) != pt { continue; }

            same_piece_to = true;
            if file_of(other_from) == file_of(from) { same_file = true; }
            if rank_of(other_from) == rank_of(from) { same_rank = true; }
        }

        if same_piece_to {
            if !same_file {
                san.push((b'a' + file_of(from)) as char);
            } else if !same_rank {
                san.push((b'1' + rank_of(from)) as char);
            } else {
                san.push((b'a' + file_of(from)) as char);
                san.push((b'1' + rank_of(from)) as char);
            }
        }
    }

    // Capture
    let is_capture = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
    if is_capture {
        if pt == PAWN {
            san.push((b'a' + file_of(from)) as char);
        }
        san.push('x');
    }

    // Destination square
    san.push((b'a' + file_of(to)) as char);
    san.push((b'1' + rank_of(to)) as char);

    // Promotion
    if is_promotion(mv) {
        san.push('=');
        san.push(match promotion_piece_type(mv) {
            KNIGHT => 'N',
            BISHOP => 'B',
            ROOK => 'R',
            QUEEN => 'Q',
            _ => '?',
        });
    }

    // Check/checkmate suffix
    let mut board_copy = board.clone();
    if !board_copy.make_move(mv) { return san; }
    if board_copy.in_check() {
        // Check if it's checkmate
        let legal_after = generate_legal_moves(&board_copy);
        if legal_after.len == 0 {
            san.push('#');
        } else {
            san.push('+');
        }
    }

    san
}

/// Run an EPD test suite.
pub fn run_epd(path: &str, time_per_pos: u64, max_positions: usize, nnue_path: Option<&str>) {
    let positions = parse_epd(path);
    let total = if max_positions > 0 { max_positions.min(positions.len()) } else { positions.len() };

    println!("Running {} positions from {}", total, path);
    println!("Time per position: {}ms", time_per_pos);
    println!();

    let mut info = SearchInfo::new(64);
    if let Some(path) = nnue_path {
        if let Err(e) = info.load_nnue(path) {
            eprintln!("Warning: failed to load NNUE: {}", e);
        }
    }
    let mut passed = 0;
    let mut failed = 0;
    let suite_start = Instant::now();

    for (i, pos) in positions.iter().enumerate() {
        if i >= total { break; }

        let mut board = Board::from_fen(&pos.fen);
        info.nodes = 0;
        info.history.clear();
        info.tt.clear();

        let limits = SearchLimits {
            movetime: time_per_pos,
            ..SearchLimits::new()
        };

        let best = search(&mut board, &mut info, &limits);
        let best_san = move_to_san(&board, best);
        let best_uci = move_to_uci(best);

        // Check if the move matches any of the expected best moves
        let is_correct = pos.best_moves.iter().any(|bm| {
            // Compare SAN (with and without check/mate symbols)
            let bm_clean = bm.trim_end_matches('+').trim_end_matches('#');
            let san_clean = best_san.trim_end_matches('+').trim_end_matches('#');
            bm_clean == san_clean || bm == &best_uci
        });

        if is_correct {
            passed += 1;
            print!(".");
        } else {
            failed += 1;
            print!("X");
            // Print details for failures
            eprint!("\n  {} FAIL: played {} ({}), expected {:?}",
                pos.id, best_san, best_uci, pos.best_moves);
        }

        // Flush periodically
        if (i + 1) % 50 == 0 {
            println!(" [{}/{}]", i + 1, total);
        }
    }

    let elapsed = suite_start.elapsed();
    println!("\n\nResults: {}/{} passed ({:.1}%)",
        passed, passed + failed,
        100.0 * passed as f64 / (passed + failed) as f64);
    println!("Total time: {:.1}s", elapsed.as_secs_f64());
}
