/// Polyglot opening book support.
///
/// Format: sorted array of 16-byte entries (key u64 BE, move u16 BE, weight u16 BE, learn u32).
/// Uses its own Zobrist hash scheme (standard Polyglot randoms).

use std::collections::HashMap;
use std::fs;

use crate::bitboard::*;
use crate::board::Board;
use crate::movegen::generate_legal_moves;
use crate::types::*;

/// A book move with weight.
struct BookMove {
    raw_move: u16,
    weight: u16,
}

/// Opening book loaded from a Polyglot .bin file.
pub struct OpeningBook {
    entries: HashMap<u64, Vec<BookMove>>,
}

impl OpeningBook {
    /// Load a Polyglot .bin opening book.
    pub fn load(path: &str) -> Result<Self, String> {
        let data = fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
        if data.len() % 16 != 0 {
            return Err(format!("book size {} not multiple of 16", data.len()));
        }

        let mut entries: HashMap<u64, Vec<BookMove>> = HashMap::new();
        let num = data.len() / 16;

        for i in 0..num {
            let off = i * 16;
            let key = u64::from_be_bytes(data[off..off+8].try_into().unwrap());
            let mv = u16::from_be_bytes(data[off+8..off+10].try_into().unwrap());
            let weight = u16::from_be_bytes(data[off+10..off+12].try_into().unwrap());

            if weight == 0 { continue; }

            entries.entry(key).or_default().push(BookMove { raw_move: mv, weight });
        }

        // Sort by weight descending
        for moves in entries.values_mut() {
            moves.sort_by(|a, b| b.weight.cmp(&a.weight));
        }

        let count = entries.len();
        eprintln!("info string Book loaded: {} positions from {}", count, path);

        Ok(OpeningBook { entries })
    }

    /// Pick a weighted random move for the position.
    pub fn pick_move(&self, board: &Board) -> Option<Move> {
        let hash = polyglot_hash(board);
        let bmoves = self.entries.get(&hash)?;
        if bmoves.is_empty() { return None; }

        let legal = generate_legal_moves(board);

        // Collect matching legal moves with weights
        let mut candidates: Vec<(Move, u32)> = Vec::new();
        let mut total_weight = 0u32;

        for bm in bmoves {
            if let Some(m) = match_polyglot_move(bm.raw_move, &legal, board) {
                let w = bm.weight as u32;
                candidates.push((m, w));
                total_weight += w;
            }
        }

        if candidates.is_empty() { return None; }

        // Weighted random selection using wall-clock nanoseconds as entropy
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let r = nanos % total_weight;
        let mut cumulative = 0;
        for (m, w) in &candidates {
            cumulative += w;
            if r < cumulative {
                return Some(*m);
            }
        }

        Some(candidates[0].0)
    }
}

/// Match a Polyglot-encoded move against the legal move list.
fn match_polyglot_move(pm: u16, legal: &crate::movegen::MoveList, _board: &Board) -> Option<Move> {
    let to_file = (pm & 7) as u8;
    let to_rank = ((pm >> 3) & 7) as u8;
    let from_file = ((pm >> 6) & 7) as u8;
    let from_rank = ((pm >> 9) & 7) as u8;
    let promo = ((pm >> 12) & 7) as u8;

    let from = square(from_file, from_rank);
    let mut to = square(to_file, to_rank);

    // Polyglot encodes castling as king-to-rook; convert to king-to-destination
    if from_file == 4 {
        if from_rank == 0 && to_rank == 0 {
            if to_file == 7 { to = 6; }       // e1h1 → e1g1
            else if to_file == 0 { to = 2; }  // e1a1 → e1c1
        } else if from_rank == 7 && to_rank == 7 {
            if to_file == 7 { to = 62; }      // e8h8 → e8g8
            else if to_file == 0 { to = 58; } // e8a8 → e8c8
        }
    }

    let promo_flag = match promo {
        1 => Some(FLAG_PROMOTE_N),
        2 => Some(FLAG_PROMOTE_B),
        3 => Some(FLAG_PROMOTE_R),
        4 => Some(FLAG_PROMOTE_Q),
        _ => None,
    };

    for i in 0..legal.len {
        let m = legal.moves[i];
        if move_from(m) == from && move_to(m) == to {
            if let Some(pf) = promo_flag {
                if move_flags(m) == pf { return Some(m); }
            } else if !is_promotion(m) {
                return Some(m);
            }
        }
    }

    None
}

/// Compute the Polyglot Zobrist hash for a board position.
/// Uses the standard Polyglot random number table.
pub fn polyglot_hash(board: &Board) -> u64 {
    let mut hash = 0u64;

    // Pieces
    for color in 0..2u8 {
        for pt in 0..6u8 {
            let mut bb = board.pieces[pt as usize] & board.colors[color as usize];
            while bb != 0 {
                let sq = pop_lsb(&mut bb) as usize;
                let idx = polyglot_piece_index(color, pt);
                hash ^= POLYGLOT_RANDOMS[idx * 64 + sq];
            }
        }
    }

    // Castling
    if board.castling & CASTLE_WK != 0 { hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET]; }
    if board.castling & CASTLE_WQ != 0 { hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 1]; }
    if board.castling & CASTLE_BK != 0 { hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 2]; }
    if board.castling & CASTLE_BQ != 0 { hash ^= POLYGLOT_RANDOMS[CASTLE_OFFSET + 3]; }

    // En passant (only if a pawn can actually capture)
    if board.ep_square != NO_SQUARE && polyglot_has_ep_capture(board) {
        let ep_file = file_of(board.ep_square) as usize;
        hash ^= POLYGLOT_RANDOMS[EP_OFFSET + ep_file];
    }

    // Side to move: Polyglot XORs when WHITE to move
    if board.side_to_move == WHITE {
        hash ^= POLYGLOT_RANDOMS[TURN_OFFSET];
    }

    hash
}

/// Map (color, piece_type) to Polyglot piece index (0-11).
/// Polyglot order: BlackPawn=0, WhitePawn=1, BlackKnight=2, WhiteKnight=3, ...
fn polyglot_piece_index(color: u8, pt: u8) -> usize {
    // Polyglot: black pieces at even indices, white at odd
    // Order: pawn, knight, bishop, rook, queen, king
    let base = pt as usize * 2;
    if color == WHITE { base + 1 } else { base }
}

/// Check if any pawn can actually make the EP capture.
fn polyglot_has_ep_capture(board: &Board) -> bool {
    if board.ep_square == NO_SQUARE { return false; }
    let ep_file = file_of(board.ep_square);
    let pawns = board.pieces[PAWN as usize] & board.colors[board.side_to_move as usize];

    if board.side_to_move == WHITE {
        // White pawns on rank 5 adjacent to EP file
        let rank4_mask = RANK_5;
        let mut check = pawns & rank4_mask;
        while check != 0 {
            let sq = pop_lsb(&mut check) as u8;
            let f = file_of(sq);
            if f == ep_file.wrapping_sub(1) || f == ep_file.wrapping_add(1) {
                if f < 8 { return true; }
            }
        }
    } else {
        let rank3_mask = RANK_4;
        let mut check = pawns & rank3_mask;
        while check != 0 {
            let sq = pop_lsb(&mut check) as u8;
            let f = file_of(sq);
            if f == ep_file.wrapping_sub(1) || f == ep_file.wrapping_add(1) {
                if f < 8 { return true; }
            }
        }
    }

    false
}

// Polyglot random number table offsets
const CASTLE_OFFSET: usize = 768;
const EP_OFFSET: usize = 772;
const TURN_OFFSET: usize = 780;

// The standard Polyglot random numbers (781 entries).
// Generated by the standard Polyglot PRNG with seed.
include!("polyglot_randoms.rs");
