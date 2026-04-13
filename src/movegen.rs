/// Move generation: pseudo-legal moves with legality filter.
///
/// GenerateAllMoves returns pseudo-legal moves (captures then quiets).
/// GenerateEvasions produces fully legal evasions when in check.

use crate::bitboard::*;
use crate::attacks::*;
use crate::types::*;
use crate::board::Board;

pub const MAX_MOVES: usize = 256;

/// Move list with stack-allocated storage.
pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    #[inline]
    pub fn new() -> Self {
        MoveList {
            moves: [0; MAX_MOVES],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len] = mv;
        self.len += 1;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

/// Add pawn promotions (Q, R, B, N).
#[inline]
fn add_promotions(list: &mut MoveList, from: u8, to: u8) {
    list.push(make_move(from, to, FLAG_PROMOTE_Q));
    list.push(make_move(from, to, FLAG_PROMOTE_R));
    list.push(make_move(from, to, FLAG_PROMOTE_B));
    list.push(make_move(from, to, FLAG_PROMOTE_N));
}

// ---------------------------------------------------------------------------
// Capture move generation
// Order: pawn captures (left, right with promo), non-capture promotions, EP,
//        knight caps, bishop caps, rook caps, queen caps, king caps
// ---------------------------------------------------------------------------

/// Generate all pseudo-legal captures (including promotions and en passant).
pub fn generate_captures(board: &Board) -> MoveList {
    let mut list = MoveList::new();
    let us = board.side_to_move;
    let them = flip_color(us);
    let occ = board.occupied();
    let our_pieces = board.colors[us as usize];
    let their_pieces = board.colors[them as usize];

    let pawns = board.pieces[PAWN as usize] & our_pieces;
    let promo_rank = if us == WHITE { RANK_8 } else { RANK_1 };

    if us == WHITE {
        // Capture left (a-file pawns excluded, shift <<7 = northwest)
        let capture_l = ((pawns & NOT_FILE_A) << 7) & their_pieces;
        // Capture right (h-file excluded, shift <<9 = northeast)
        let capture_r = ((pawns & NOT_FILE_H) << 9) & their_pieces;

        // Left captures: promotion then non-promotion
        let mut promo = capture_l & promo_rank;
        let mut non_promo = capture_l & !promo_rank;
        while promo != 0 {
            let to = pop_lsb(&mut promo) as u8;
            add_promotions(&mut list, to - 7, to);
        }
        while non_promo != 0 {
            let to = pop_lsb(&mut non_promo) as u8;
            list.push(make_move(to - 7, to, FLAG_NONE));
        }

        // Right captures: promotion then non-promotion
        let mut promo = capture_r & promo_rank;
        let mut non_promo = capture_r & !promo_rank;
        while promo != 0 {
            let to = pop_lsb(&mut promo) as u8;
            add_promotions(&mut list, to - 9, to);
        }
        while non_promo != 0 {
            let to = pop_lsb(&mut non_promo) as u8;
            list.push(make_move(to - 9, to, FLAG_NONE));
        }

        // Non-capture promotions (push to 8th rank)
        let empty = !occ;
        let mut push_promo = ((pawns << 8) & empty) & promo_rank;
        while push_promo != 0 {
            let to = pop_lsb(&mut push_promo) as u8;
            add_promotions(&mut list, to - 8, to);
        }

        // En passant
        if board.ep_square != NO_SQUARE {
            let ep_bb = 1u64 << board.ep_square;
            let ep_l = ((pawns & NOT_FILE_A) << 7) & ep_bb;
            let ep_r = ((pawns & NOT_FILE_H) << 9) & ep_bb;
            if ep_l != 0 {
                list.push(make_move(board.ep_square - 7, board.ep_square, FLAG_EN_PASSANT));
            }
            if ep_r != 0 {
                list.push(make_move(board.ep_square - 9, board.ep_square, FLAG_EN_PASSANT));
            }
        }
    } else {
        // Black pawns
        // Capture left (a-file excluded, shift >>9 = southwest)
        let capture_l = ((pawns & NOT_FILE_A) >> 9) & their_pieces;
        // Capture right (h-file excluded, shift >>7 = southeast)
        let capture_r = ((pawns & NOT_FILE_H) >> 7) & their_pieces;

        let mut promo = capture_l & promo_rank;
        let mut non_promo = capture_l & !promo_rank;
        while promo != 0 {
            let to = pop_lsb(&mut promo) as u8;
            add_promotions(&mut list, to + 9, to);
        }
        while non_promo != 0 {
            let to = pop_lsb(&mut non_promo) as u8;
            list.push(make_move(to + 9, to, FLAG_NONE));
        }

        let mut promo = capture_r & promo_rank;
        let mut non_promo = capture_r & !promo_rank;
        while promo != 0 {
            let to = pop_lsb(&mut promo) as u8;
            add_promotions(&mut list, to + 7, to);
        }
        while non_promo != 0 {
            let to = pop_lsb(&mut non_promo) as u8;
            list.push(make_move(to + 7, to, FLAG_NONE));
        }

        // Non-capture promotions
        let empty = !occ;
        let mut push_promo = ((pawns >> 8) & empty) & promo_rank;
        while push_promo != 0 {
            let to = pop_lsb(&mut push_promo) as u8;
            add_promotions(&mut list, to + 8, to);
        }

        // En passant
        if board.ep_square != NO_SQUARE {
            let ep_bb = 1u64 << board.ep_square;
            let ep_l = ((pawns & NOT_FILE_A) >> 9) & ep_bb;
            let ep_r = ((pawns & NOT_FILE_H) >> 7) & ep_bb;
            if ep_l != 0 {
                list.push(make_move(board.ep_square + 9, board.ep_square, FLAG_EN_PASSANT));
            }
            if ep_r != 0 {
                list.push(make_move(board.ep_square + 7, board.ep_square, FLAG_EN_PASSANT));
            }
        }
    }

    // Knight captures
    let mut knights = board.pieces[KNIGHT as usize] & our_pieces;
    while knights != 0 {
        let from = pop_lsb(&mut knights) as u8;
        let mut attacks = knight_attacks(from as u32) & their_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Bishop captures
    let mut bishops = board.pieces[BISHOP as usize] & our_pieces;
    while bishops != 0 {
        let from = pop_lsb(&mut bishops) as u8;
        let mut attacks = bishop_attacks(from as u32, occ) & their_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Rook captures
    let mut rooks = board.pieces[ROOK as usize] & our_pieces;
    while rooks != 0 {
        let from = pop_lsb(&mut rooks) as u8;
        let mut attacks = rook_attacks(from as u32, occ) & their_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Queen captures
    let mut queens = board.pieces[QUEEN as usize] & our_pieces;
    while queens != 0 {
        let from = pop_lsb(&mut queens) as u8;
        let mut attacks = queen_attacks(from as u32, occ) & their_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // King captures
    let ksq = board.king_sq(us);
    let mut attacks = king_attacks(ksq as u32) & their_pieces & !our_pieces;
    while attacks != 0 {
        let to = pop_lsb(&mut attacks) as u8;
        list.push(make_move(ksq, to, FLAG_NONE));
    }

    list
}

// ---------------------------------------------------------------------------
// Quiet move generation
// Order: pawn single push (non-promo), pawn double push,
//        knight, bishop, rook, queen, king, castling
// ---------------------------------------------------------------------------

/// Generate pseudo-legal quiet moves (non-captures, non-promotions).
pub fn generate_quiets(board: &Board) -> MoveList {
    let mut list = MoveList::new();
    let us = board.side_to_move;
    let occ = board.occupied();
    let our_pieces = board.colors[us as usize];
    let empty = board.empty();

    let pawns = board.pieces[PAWN as usize] & our_pieces;
    let promo_rank = if us == WHITE { RANK_8 } else { RANK_1 };

    if us == WHITE {
        // Single push (exclude promotions)
        let push1 = ((pawns << 8) & empty) & !promo_rank;
        let mut p = push1;
        while p != 0 {
            let to = pop_lsb(&mut p) as u8;
            list.push(make_move(to - 8, to, FLAG_NONE));
        }

        // Double push
        let push1_all = (pawns << 8) & empty;
        let push2 = ((push1_all & RANK_3) << 8) & empty;
        let mut p = push2;
        while p != 0 {
            let to = pop_lsb(&mut p) as u8;
            list.push(make_move(to - 16, to, FLAG_DOUBLE_PUSH));
        }
    } else {
        let push1 = ((pawns >> 8) & empty) & !promo_rank;
        let mut p = push1;
        while p != 0 {
            let to = pop_lsb(&mut p) as u8;
            list.push(make_move(to + 8, to, FLAG_NONE));
        }

        let push1_all = (pawns >> 8) & empty;
        let push2 = ((push1_all & RANK_6) >> 8) & empty;
        let mut p = push2;
        while p != 0 {
            let to = pop_lsb(&mut p) as u8;
            list.push(make_move(to + 16, to, FLAG_DOUBLE_PUSH));
        }
    }

    // Knight quiets
    let mut knights = board.pieces[KNIGHT as usize] & our_pieces;
    while knights != 0 {
        let from = pop_lsb(&mut knights) as u8;
        let mut attacks = knight_attacks(from as u32) & empty;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Bishop quiets
    let mut bishops = board.pieces[BISHOP as usize] & our_pieces;
    while bishops != 0 {
        let from = pop_lsb(&mut bishops) as u8;
        let mut attacks = bishop_attacks(from as u32, occ) & empty;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Rook quiets
    let mut rooks = board.pieces[ROOK as usize] & our_pieces;
    while rooks != 0 {
        let from = pop_lsb(&mut rooks) as u8;
        let mut attacks = rook_attacks(from as u32, occ) & empty;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Queen quiets
    let mut queens = board.pieces[QUEEN as usize] & our_pieces;
    while queens != 0 {
        let from = pop_lsb(&mut queens) as u8;
        let mut attacks = queen_attacks(from as u32, occ) & empty;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // King quiets
    let ksq = board.king_sq(us);
    let mut attacks = king_attacks(ksq as u32) & empty;
    while attacks != 0 {
        let to = pop_lsb(&mut attacks) as u8;
        list.push(make_move(ksq, to, FLAG_NONE));
    }

    // Castling
    generate_castling(board, &mut list, us, occ);

    list
}

// ---------------------------------------------------------------------------
// Castling move generation
// ---------------------------------------------------------------------------

fn generate_castling(board: &Board, list: &mut MoveList, us: Color, occ: Bitboard) {
    if us == WHITE {
        // Kingside: f1 and g1 must be empty, e1/f1/g1 not attacked
        if board.castling & CASTLE_WK != 0 {
            if occ & 0x60 == 0 {
                if !is_attacked(board, 4, BLACK) && !is_attacked(board, 5, BLACK) && !is_attacked(board, 6, BLACK) {
                    list.push(make_move(4, 6, FLAG_CASTLE));
                }
            }
        }
        // Queenside: b1, c1, d1 must be empty, e1/d1/c1 not attacked
        if board.castling & CASTLE_WQ != 0 {
            if occ & 0x0E == 0 {
                if !is_attacked(board, 4, BLACK) && !is_attacked(board, 3, BLACK) && !is_attacked(board, 2, BLACK) {
                    list.push(make_move(4, 2, FLAG_CASTLE));
                }
            }
        }
    } else {
        // Kingside
        if board.castling & CASTLE_BK != 0 {
            if occ & (0x60u64 << 56) == 0 {
                if !is_attacked(board, 60, WHITE) && !is_attacked(board, 61, WHITE) && !is_attacked(board, 62, WHITE) {
                    list.push(make_move(60, 62, FLAG_CASTLE));
                }
            }
        }
        // Queenside
        if board.castling & CASTLE_BQ != 0 {
            if occ & (0x0Eu64 << 56) == 0 {
                if !is_attacked(board, 60, WHITE) && !is_attacked(board, 59, WHITE) && !is_attacked(board, 58, WHITE) {
                    list.push(make_move(60, 58, FLAG_CASTLE));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Attack detection
// ---------------------------------------------------------------------------

/// Is a square attacked by the given color?
pub fn is_attacked(board: &Board, sq: u8, by_color: Color) -> bool {
    let occ = board.occupied();
    let attackers = board.colors[by_color as usize];

    if pawn_attacks(flip_color(by_color), sq as u32) & board.pieces[PAWN as usize] & attackers != 0 { return true; }
    if knight_attacks(sq as u32) & board.pieces[KNIGHT as usize] & attackers != 0 { return true; }
    if king_attacks(sq as u32) & board.pieces[KING as usize] & attackers != 0 { return true; }
    if bishop_attacks(sq as u32, occ) & (board.pieces[BISHOP as usize] | board.pieces[QUEEN as usize]) & attackers != 0 { return true; }
    if rook_attacks(sq as u32, occ) & (board.pieces[ROOK as usize] | board.pieces[QUEEN as usize]) & attackers != 0 { return true; }

    false
}

/// Is a square attacked by the given color, using a custom occupancy?
/// Used for king-move legality (king removed from occ) and evasions.
fn is_attacked_with_occ(board: &Board, sq: u8, by_color: Color, occ: Bitboard) -> bool {
    let attackers = board.colors[by_color as usize];

    if pawn_attacks(flip_color(by_color), sq as u32) & board.pieces[PAWN as usize] & attackers != 0 { return true; }
    if knight_attacks(sq as u32) & board.pieces[KNIGHT as usize] & attackers != 0 { return true; }
    if king_attacks(sq as u32) & board.pieces[KING as usize] & attackers != 0 { return true; }
    if bishop_attacks(sq as u32, occ) & (board.pieces[BISHOP as usize] | board.pieces[QUEEN as usize]) & attackers != 0 { return true; }
    if rook_attacks(sq as u32, occ) & (board.pieces[ROOK as usize] | board.pieces[QUEEN as usize]) & attackers != 0 { return true; }

    false
}

// ---------------------------------------------------------------------------
// All moves = captures + quiets
// ---------------------------------------------------------------------------

/// Generate all pseudo-legal moves (captures first, then quiets).
pub fn generate_all_moves(board: &Board) -> MoveList {
    let mut list = MoveList::new();

    // Captures
    let caps = generate_captures(board);
    for i in 0..caps.len {
        list.push(caps.moves[i]);
    }

    // Quiets
    let quiets = generate_quiets(board);
    for i in 0..quiets.len {
        list.push(quiets.moves[i]);
    }

    list
}

// ---------------------------------------------------------------------------
// Evasion move generation (in check)
// Fully legal evasion moves when in check. No IsLegal filtering needed.
// ---------------------------------------------------------------------------

/// Generate all legal evasion moves when in check.
/// checkers and pinned should be precomputed via board.checkers() / board.pinned().
/// The returned moves are fully legal.
pub fn generate_evasions(board: &mut Board, checkers: Bitboard, pinned: Bitboard) -> MoveList {
    let mut list = MoveList::new();
    let us = board.side_to_move;
    let them = flip_color(us);
    let king_sq = board.king_sq(us);
    let our_pieces = board.colors[us as usize];

    // King evasions: always generated (both single and double check).
    // Remove king from occupancy so sliders see through it.
    let occ = board.occupied() ^ (1u64 << king_sq);
    let mut targets = king_attacks(king_sq as u32) & !our_pieces;
    while targets != 0 {
        let to = pop_lsb(&mut targets) as u8;
        if !is_attacked_with_occ(board, to, them, occ) {
            list.push(make_move(king_sq, to, FLAG_NONE));
        }
    }

    // Double check: only king moves are legal
    if more_than_one(checkers) {
        return list;
    }

    // Single check: can also block or capture the checker
    let checker_sq = lsb(checkers);
    // target = capture the checker OR block the ray between king and checker.
    // between() is empty for non-sliding checkers (knight, pawn).
    let target = (1u64 << checker_sq) | between(king_sq as u32, checker_sq);
    let block_target = between(king_sq as u32, checker_sq); // blocking squares only

    // Only non-pinned pieces can resolve check
    let non_pinned = our_pieces & !pinned & !(1u64 << king_sq);

    // Knights
    let mut knights = board.pieces[KNIGHT as usize] & non_pinned;
    while knights != 0 {
        let from = pop_lsb(&mut knights) as u8;
        let mut attacks = knight_attacks(from as u32) & target;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Bishops
    let mut bishops = board.pieces[BISHOP as usize] & non_pinned;
    while bishops != 0 {
        let from = pop_lsb(&mut bishops) as u8;
        let mut attacks = bishop_attacks(from as u32, board.occupied()) & target;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Rooks
    let mut rooks = board.pieces[ROOK as usize] & non_pinned;
    while rooks != 0 {
        let from = pop_lsb(&mut rooks) as u8;
        let mut attacks = rook_attacks(from as u32, board.occupied()) & target;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Queens
    let mut queens = board.pieces[QUEEN as usize] & non_pinned;
    while queens != 0 {
        let from = pop_lsb(&mut queens) as u8;
        let mut attacks = queen_attacks(from as u32, board.occupied()) & target;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Pawns
    let pawns = board.pieces[PAWN as usize] & non_pinned;
    let checker_bb = 1u64 << checker_sq;
    let empty = !board.occupied();

    if us == WHITE {
        let promo_rank = RANK_8;

        // Pawn captures onto checker square
        let capture_l = ((pawns & NOT_FILE_A) << 7) & checker_bb;
        let capture_r = ((pawns & NOT_FILE_H) << 9) & checker_bb;

        let mut cl = capture_l;
        while cl != 0 {
            let to = pop_lsb(&mut cl) as u8;
            let from = to - 7;
            if (1u64 << to) & promo_rank != 0 {
                add_promotions(&mut list, from, to);
            } else {
                list.push(make_move(from, to, FLAG_NONE));
            }
        }
        let mut cr = capture_r;
        while cr != 0 {
            let to = pop_lsb(&mut cr) as u8;
            let from = to - 9;
            if (1u64 << to) & promo_rank != 0 {
                add_promotions(&mut list, from, to);
            } else {
                list.push(make_move(from, to, FLAG_NONE));
            }
        }

        // Pawn pushes onto blocking/capture squares
        let push1 = ((pawns << 8) & empty) & target;
        let push2 = ((((pawns << 8) & empty & RANK_3) << 8) & empty) & block_target;

        let mut p = push1;
        while p != 0 {
            let to = pop_lsb(&mut p) as u8;
            let from = to - 8;
            if (1u64 << to) & promo_rank != 0 {
                add_promotions(&mut list, from, to);
            } else {
                list.push(make_move(from, to, FLAG_NONE));
            }
        }
        let mut p = push2;
        while p != 0 {
            let to = pop_lsb(&mut p) as u8;
            list.push(make_move(to - 16, to, FLAG_DOUBLE_PUSH));
        }

        // En passant: only if the captured pawn IS the checking piece
        if board.ep_square != NO_SQUARE {
            let captured_pawn_sq = board.ep_square - 8;
            if captured_pawn_sq as u32 == checker_sq {
                let ep_bb = 1u64 << board.ep_square;
                let ep_l = ((pawns & NOT_FILE_A) << 7) & ep_bb;
                let ep_r = ((pawns & NOT_FILE_H) << 9) & ep_bb;
                if ep_l != 0 {
                    let m = make_move(board.ep_square - 7, board.ep_square, FLAG_EN_PASSANT);
                    board.make_move(m);
                    if !is_attacked(board, king_sq, them) {
                        list.push(m);
                    }
                    board.unmake_move();
                }
                if ep_r != 0 {
                    let m = make_move(board.ep_square - 9, board.ep_square, FLAG_EN_PASSANT);
                    board.make_move(m);
                    if !is_attacked(board, king_sq, them) {
                        list.push(m);
                    }
                    board.unmake_move();
                }
            }
        }
    } else {
        // Black pawns
        let promo_rank = RANK_1;

        let capture_l = ((pawns & NOT_FILE_A) >> 9) & checker_bb;
        let capture_r = ((pawns & NOT_FILE_H) >> 7) & checker_bb;

        let mut cl = capture_l;
        while cl != 0 {
            let to = pop_lsb(&mut cl) as u8;
            let from = to + 9;
            if (1u64 << to) & promo_rank != 0 {
                add_promotions(&mut list, from, to);
            } else {
                list.push(make_move(from, to, FLAG_NONE));
            }
        }
        let mut cr = capture_r;
        while cr != 0 {
            let to = pop_lsb(&mut cr) as u8;
            let from = to + 7;
            if (1u64 << to) & promo_rank != 0 {
                add_promotions(&mut list, from, to);
            } else {
                list.push(make_move(from, to, FLAG_NONE));
            }
        }

        let push1 = ((pawns >> 8) & empty) & target;
        let push2 = ((((pawns >> 8) & empty & RANK_6) >> 8) & empty) & block_target;

        let mut p = push1;
        while p != 0 {
            let to = pop_lsb(&mut p) as u8;
            let from = to + 8;
            if (1u64 << to) & promo_rank != 0 {
                add_promotions(&mut list, from, to);
            } else {
                list.push(make_move(from, to, FLAG_NONE));
            }
        }
        let mut p = push2;
        while p != 0 {
            let to = pop_lsb(&mut p) as u8;
            list.push(make_move(to + 16, to, FLAG_DOUBLE_PUSH));
        }

        // En passant
        if board.ep_square != NO_SQUARE {
            let captured_pawn_sq = board.ep_square + 8;
            if captured_pawn_sq as u32 == checker_sq {
                let ep_bb = 1u64 << board.ep_square;
                let ep_l = ((pawns & NOT_FILE_A) >> 9) & ep_bb;
                let ep_r = ((pawns & NOT_FILE_H) >> 7) & ep_bb;
                if ep_l != 0 {
                    let m = make_move(board.ep_square + 9, board.ep_square, FLAG_EN_PASSANT);
                    board.make_move(m);
                    if !is_attacked(board, king_sq, them) {
                        list.push(m);
                    }
                    board.unmake_move();
                }
                if ep_r != 0 {
                    let m = make_move(board.ep_square + 7, board.ep_square, FLAG_EN_PASSANT);
                    board.make_move(m);
                    if !is_attacked(board, king_sq, them) {
                        list.push(m);
                    }
                    board.unmake_move();
                }
            }
        }
    }

    list
}

// ---------------------------------------------------------------------------
// Legal moves (for perft and verification)
// ---------------------------------------------------------------------------

/// Generate all legal moves (for perft and verification).
pub fn generate_legal_moves(board: &Board) -> MoveList {
    let pseudo = generate_all_moves(board);
    let pinned = board.pinned();
    let checkers = board.checkers();
    let mut legal = MoveList::new();

    for i in 0..pseudo.len {
        let mv = pseudo.moves[i];
        if board.is_legal(mv, pinned, checkers) {
            legal.push(mv);
        }
    }

    legal
}

// ---------------------------------------------------------------------------
// Perft
// ---------------------------------------------------------------------------

/// Perft: count leaf nodes at given depth.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = generate_legal_moves(board);

    if depth == 1 {
        return moves.len as u64;
    }

    let mut nodes = 0u64;
    for i in 0..moves.len {
        let mv = moves.moves[i];
        board.make_move(mv);
        nodes += perft(board, depth - 1);
        board.unmake_move();
    }

    nodes
}

/// Perft with divide: show per-move counts at the root.
pub fn perft_divide(board: &mut Board, depth: u32) -> u64 {
    let moves = generate_legal_moves(board);
    let mut total = 0u64;

    for i in 0..moves.len {
        let mv = moves.moves[i];
        board.make_move(mv);
        let count = if depth <= 1 { 1 } else { perft(board, depth - 1) };
        board.unmake_move();
        println!("{}: {}", move_to_uci(mv), count);
        total += count;
    }

    println!("\nTotal: {}", total);
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() { crate::init(); }

    #[test]
    fn test_startpos_moves() {
        init();
        let b = Board::startpos();
        let moves = generate_legal_moves(&b);
        assert_eq!(moves.len, 20, "Starting position should have 20 legal moves");
    }

    #[test]
    fn test_perft_startpos() {
        init();
        let mut b = Board::startpos();
        assert_eq!(perft(&mut b, 1), 20);
        assert_eq!(perft(&mut b, 2), 400);
        assert_eq!(perft(&mut b, 3), 8902);
        assert_eq!(perft(&mut b, 4), 197281);
        assert_eq!(perft(&mut b, 5), 4865609);
    }

    #[test]
    fn test_perft_kiwipete() {
        init();
        // Position 2: Kiwipete
        let mut b = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        assert_eq!(perft(&mut b, 1), 48);
        assert_eq!(perft(&mut b, 2), 2039);
        assert_eq!(perft(&mut b, 3), 97862);
        assert_eq!(perft(&mut b, 4), 4085603);
    }

    #[test]
    fn test_perft_position3() {
        init();
        let mut b = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        assert_eq!(perft(&mut b, 1), 14);
        assert_eq!(perft(&mut b, 2), 191);
        assert_eq!(perft(&mut b, 3), 2812);
        assert_eq!(perft(&mut b, 4), 43238);
        assert_eq!(perft(&mut b, 5), 674624);
    }

    #[test]
    fn test_perft_position4() {
        init();
        let mut b = Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
        assert_eq!(perft(&mut b, 1), 6);
        assert_eq!(perft(&mut b, 2), 264);
        assert_eq!(perft(&mut b, 3), 9467);
        assert_eq!(perft(&mut b, 4), 422333);
    }

    #[test]
    fn test_perft_position5() {
        init();
        let mut b = Board::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
        assert_eq!(perft(&mut b, 1), 44);
        assert_eq!(perft(&mut b, 2), 1486);
        assert_eq!(perft(&mut b, 3), 62379);
        assert_eq!(perft(&mut b, 4), 2103487);
    }

    #[test]
    fn test_perft_position6() {
        init();
        let mut b = Board::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
        assert_eq!(perft(&mut b, 1), 46);
        assert_eq!(perft(&mut b, 2), 2079);
        assert_eq!(perft(&mut b, 3), 89890);
        assert_eq!(perft(&mut b, 4), 3894594);
    }

    #[test]
    fn test_ep_legal() {
        init();
        // Position where EP capture would leave king in check (discovered check)
        let b = Board::from_fen("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1");
        let moves = generate_legal_moves(&b);
        // EP capture e4xd3 should be illegal because it discovers check from Qh4
        for i in 0..moves.len {
            let mv = moves.moves[i];
            if move_flags(mv) == FLAG_EN_PASSANT {
                panic!("EP move should be illegal in this position");
            }
        }
    }

    /// Helper: classify a move as "quiet" (not a capture, not a promotion, not EP).
    fn is_quiet(board: &Board, mv: Move) -> bool {
        let to = move_to(mv);
        let flags = move_flags(mv);
        if flags == FLAG_EN_PASSANT { return false; }
        if is_promotion(mv) { return false; }
        if board.piece_type_at(to) != NO_PIECE_TYPE { return false; }
        true
    }

    /// Verify generate_quiets() produces the exact same set of moves as
    /// the quiet subset of generate_all_moves().
    fn verify_quiets_match(fen: &str) {
        let b = Board::from_fen(fen);

        // Get all moves, filter to quiets
        let all = generate_all_moves(&b);
        let mut expected: Vec<Move> = (0..all.len)
            .map(|i| all.moves[i])
            .filter(|&mv| is_quiet(&b, mv))
            .collect();
        expected.sort();

        // Get quiet moves directly
        let quiets = generate_quiets(&b);
        let mut actual: Vec<Move> = (0..quiets.len)
            .map(|i| quiets.moves[i])
            .collect();
        actual.sort();

        assert_eq!(
            expected.len(), actual.len(),
            "Quiet move count mismatch for FEN: {}\nExpected {} moves, got {}\nExpected: {:?}\nActual:   {:?}",
            fen, expected.len(), actual.len(),
            expected.iter().map(|&m| move_to_uci(m)).collect::<Vec<_>>(),
            actual.iter().map(|&m| move_to_uci(m)).collect::<Vec<_>>(),
        );
        assert_eq!(
            expected, actual,
            "Quiet move set mismatch for FEN: {}",
            fen,
        );
    }

    #[test]
    fn test_generate_quiets() {
        init();
        // Starting position
        verify_quiets_match("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        // Kiwipete (lots of captures, EP, castling)
        verify_quiets_match("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        // Position 3 (pawns, rook)
        verify_quiets_match("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        // Position 4 (promotions, heavy captures)
        verify_quiets_match("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
        // Black to move with promotions
        verify_quiets_match("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1");
        // Position 5 (promotion on d8)
        verify_quiets_match("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
        // Position 6 (symmetrical)
        verify_quiets_match("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
        // EP position
        verify_quiets_match("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1");
        // No pawns
        verify_quiets_match("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
        // Endgame with passed pawns
        verify_quiets_match("8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1");
    }

    /// Verify captures + quiets == all moves (complete union, no duplicates, no missing).
    fn verify_union_match(fen: &str) {
        let b = Board::from_fen(fen);

        let all = generate_all_moves(&b);
        let mut all_set: Vec<Move> = (0..all.len).map(|i| all.moves[i]).collect();
        all_set.sort();

        let caps = generate_captures(&b);
        let quiets = generate_quiets(&b);
        let mut union: Vec<Move> = (0..caps.len).map(|i| caps.moves[i])
            .chain((0..quiets.len).map(|i| quiets.moves[i]))
            .collect();
        union.sort();

        assert_eq!(
            all_set.len(), union.len(),
            "Union count mismatch for FEN: {}\nAll={}, Caps+Quiets={} (caps={}, quiets={})\nAll:   {:?}\nUnion: {:?}",
            fen, all_set.len(), union.len(), caps.len, quiets.len,
            all_set.iter().map(|&m| move_to_uci(m)).collect::<Vec<_>>(),
            union.iter().map(|&m| move_to_uci(m)).collect::<Vec<_>>(),
        );
        assert_eq!(all_set, union, "Union set mismatch for FEN: {}", fen);
    }

    #[test]
    fn test_captures_plus_quiets_equals_all() {
        init();
        verify_union_match("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        verify_union_match("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        verify_union_match("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        verify_union_match("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
        verify_union_match("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 b kq - 0 1");
        verify_union_match("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
        verify_union_match("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");
        verify_union_match("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1");
        verify_union_match("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
        verify_union_match("8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 0 1");
    }

    #[test]
    fn test_promotion_captures() {
        init();
        let b = Board::from_fen("r3k2r/1P6/8/8/8/8/8/4K3 w kq - 0 1");
        let moves = generate_legal_moves(&b);
        // b7b8 promotions (4) + b7xa8 capture promotions (4) = 8 pawn moves minimum
        let promo_moves: Vec<_> = (0..moves.len)
            .filter(|&i| is_promotion(moves.moves[i]))
            .collect();
        assert!(promo_moves.len() >= 4, "Should have at least 4 promotion moves, got {}", promo_moves.len());
    }

    // -----------------------------------------------------------------------
    // Evasion generator tests
    // -----------------------------------------------------------------------

    /// Verify that generate_evasions produces the same set of legal moves
    /// as the filtered generate_all_moves when in check.
    fn verify_evasions_match(fen: &str) {
        let mut b = Board::from_fen(fen);
        let checkers = b.checkers();
        if checkers == 0 {
            panic!("verify_evasions_match called on non-check position: {}", fen);
        }
        let pinned = b.pinned();

        // Legal moves via standard path
        let all = generate_all_moves(&b);
        let mut expected: Vec<Move> = (0..all.len)
            .map(|i| all.moves[i])
            .filter(|&mv| b.is_legal(mv, pinned, checkers))
            .collect();
        expected.sort();

        // Legal moves via evasion generator
        let evasions = generate_evasions(&mut b, checkers, pinned);
        let mut actual: Vec<Move> = (0..evasions.len)
            .map(|i| evasions.moves[i])
            .collect();
        actual.sort();

        assert_eq!(
            expected.len(), actual.len(),
            "Evasion count mismatch for FEN: {}\nExpected {} moves, got {}\nExpected: {:?}\nActual:   {:?}",
            fen, expected.len(), actual.len(),
            expected.iter().map(|&m| move_to_uci(m)).collect::<Vec<_>>(),
            actual.iter().map(|&m| move_to_uci(m)).collect::<Vec<_>>(),
        );
        assert_eq!(
            expected, actual,
            "Evasion move set mismatch for FEN: {}\nExpected: {:?}\nActual:   {:?}",
            fen,
            expected.iter().map(|&m| move_to_uci(m)).collect::<Vec<_>>(),
            actual.iter().map(|&m| move_to_uci(m)).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_evasions_single_check() {
        init();
        // White king on d3 in check from black queen on e4
        verify_evasions_match("rnb1kbnr/pppppppp/8/8/4q3/3K4/PPPPPPPP/RNBQ1BNR w kq - 0 1");
        // Black king on e8 in check from white rook on e1
        verify_evasions_match("4k3/8/8/8/8/8/8/4R2K b - - 0 1");
        // Black king on e8 in check from white knight on f6
        verify_evasions_match("4k3/8/5N2/8/8/8/8/4K3 b - - 0 1");
        // White king on e1 in check from black bishop on b4
        verify_evasions_match("4k3/8/8/8/1b6/8/8/4K3 w - - 0 1");
    }

    #[test]
    fn test_evasions_double_check() {
        init();
        // Double check: white rook on e7 + white knight on f6 both check black king on e8
        verify_evasions_match("4k3/4R3/5N2/8/8/8/8/4K3 b - - 0 1");
    }

    #[test]
    fn test_evasions_ep_check() {
        init();
        // Pawn gives check, EP can capture the checking pawn
        verify_evasions_match("8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1");
    }

    #[test]
    fn test_evasions_promotion_check() {
        init();
        // White king on e1 in check from black rook on a1, pawn on d7 can promote
        verify_evasions_match("8/3P4/8/4k3/8/8/8/r3K3 w - - 0 1");
    }

    #[test]
    fn test_evasions_perft() {
        init();
        // Run perft on positions that involve checks, using evasion-based perft
        // Position 3 has lots of checks
        let mut b = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        assert_eq!(perft(&mut b, 1), 14);
        assert_eq!(perft(&mut b, 2), 191);
        assert_eq!(perft(&mut b, 3), 2812);
        assert_eq!(perft(&mut b, 4), 43238);
        assert_eq!(perft(&mut b, 5), 674624);
    }
}
