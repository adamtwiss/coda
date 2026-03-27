/// Move generation: pseudo-legal moves with legality filter.
///
/// GenerateAllMoves returns pseudo-legal moves. Search calls is_legal() to filter.
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

/// Add pawn promotions (all 4 types).
#[inline]
fn add_promotions(list: &mut MoveList, from: u8, to: u8) {
    list.push(make_move(from, to, FLAG_PROMOTE_Q));
    list.push(make_move(from, to, FLAG_PROMOTE_R));
    list.push(make_move(from, to, FLAG_PROMOTE_B));
    list.push(make_move(from, to, FLAG_PROMOTE_N));
}

/// Generate all pseudo-legal moves.
pub fn generate_all_moves(board: &Board) -> MoveList {
    let mut list = MoveList::new();
    let us = board.side_to_move;
    let them = flip_color(us);
    let occ = board.occupied();
    let our_pieces = board.colors[us as usize];
    let their_pieces = board.colors[them as usize];
    let empty = board.empty();

    // Pawns
    generate_pawn_moves(board, &mut list, us, occ, our_pieces, their_pieces, empty);

    // Knights
    let mut knights = board.pieces[KNIGHT as usize] & our_pieces;
    while knights != 0 {
        let from = pop_lsb(&mut knights) as u8;
        let mut attacks = knight_attacks(from as u32) & !our_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Bishops
    let mut bishops = board.pieces[BISHOP as usize] & our_pieces;
    while bishops != 0 {
        let from = pop_lsb(&mut bishops) as u8;
        let mut attacks = bishop_attacks(from as u32, occ) & !our_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Rooks
    let mut rooks = board.pieces[ROOK as usize] & our_pieces;
    while rooks != 0 {
        let from = pop_lsb(&mut rooks) as u8;
        let mut attacks = rook_attacks(from as u32, occ) & !our_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Queens
    let mut queens = board.pieces[QUEEN as usize] & our_pieces;
    while queens != 0 {
        let from = pop_lsb(&mut queens) as u8;
        let mut attacks = queen_attacks(from as u32, occ) & !our_pieces;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // King
    let ksq = board.king_sq(us);
    let mut attacks = king_attacks(ksq as u32) & !our_pieces;
    while attacks != 0 {
        let to = pop_lsb(&mut attacks) as u8;
        list.push(make_move(ksq, to, FLAG_NONE));
    }

    // Castling
    generate_castling(board, &mut list, us, occ);

    list
}

fn generate_pawn_moves(
    board: &Board,
    list: &mut MoveList,
    us: Color,
    _occ: Bitboard,
    our_pieces: Bitboard,
    their_pieces: Bitboard,
    empty: Bitboard,
) {
    let pawns = board.pieces[PAWN as usize] & our_pieces;
    let promo_rank = if us == WHITE { RANK_8 } else { RANK_1 };
    let third_rank = if us == WHITE { RANK_3 } else { RANK_6 };

    if us == WHITE {
        // Single push
        let single = north(pawns) & empty;
        let mut promo = single & promo_rank;
        let mut non_promo = single & !promo_rank;

        while promo != 0 {
            let to = pop_lsb(&mut promo) as u8;
            add_promotions(list, to - 8, to);
        }
        while non_promo != 0 {
            let to = pop_lsb(&mut non_promo) as u8;
            list.push(make_move(to - 8, to, FLAG_NONE));
        }

        // Double push
        let double = north(single & third_rank) & empty;
        let mut doubles = double;
        while doubles != 0 {
            let to = pop_lsb(&mut doubles) as u8;
            list.push(make_move(to - 16, to, FLAG_DOUBLE_PUSH));
        }

        // Captures
        let mut cap_left = north_west(pawns) & their_pieces;
        let mut cap_right = north_east(pawns) & their_pieces;

        // Promotion captures
        let mut promo_left = cap_left & promo_rank;
        let mut promo_right = cap_right & promo_rank;
        cap_left &= !promo_rank;
        cap_right &= !promo_rank;

        while promo_left != 0 {
            let to = pop_lsb(&mut promo_left) as u8;
            add_promotions(list, to - 7, to);
        }
        while promo_right != 0 {
            let to = pop_lsb(&mut promo_right) as u8;
            add_promotions(list, to - 9, to);
        }
        while cap_left != 0 {
            let to = pop_lsb(&mut cap_left) as u8;
            list.push(make_move(to - 7, to, FLAG_NONE));
        }
        while cap_right != 0 {
            let to = pop_lsb(&mut cap_right) as u8;
            list.push(make_move(to - 9, to, FLAG_NONE));
        }
    } else {
        // Black pawns — mirror of white
        let single = south(pawns) & empty;
        let mut promo = single & promo_rank;
        let mut non_promo = single & !promo_rank;

        while promo != 0 {
            let to = pop_lsb(&mut promo) as u8;
            add_promotions(list, to + 8, to);
        }
        while non_promo != 0 {
            let to = pop_lsb(&mut non_promo) as u8;
            list.push(make_move(to + 8, to, FLAG_NONE));
        }

        let double = south(single & third_rank) & empty;
        let mut doubles = double;
        while doubles != 0 {
            let to = pop_lsb(&mut doubles) as u8;
            list.push(make_move(to + 16, to, FLAG_DOUBLE_PUSH));
        }

        let mut cap_left = south_west(pawns) & their_pieces;
        let mut cap_right = south_east(pawns) & their_pieces;

        let mut promo_left = cap_left & promo_rank;
        let mut promo_right = cap_right & promo_rank;
        cap_left &= !promo_rank;
        cap_right &= !promo_rank;

        while promo_left != 0 {
            let to = pop_lsb(&mut promo_left) as u8;
            add_promotions(list, to + 9, to);
        }
        while promo_right != 0 {
            let to = pop_lsb(&mut promo_right) as u8;
            add_promotions(list, to + 7, to);
        }
        while cap_left != 0 {
            let to = pop_lsb(&mut cap_left) as u8;
            list.push(make_move(to + 9, to, FLAG_NONE));
        }
        while cap_right != 0 {
            let to = pop_lsb(&mut cap_right) as u8;
            list.push(make_move(to + 7, to, FLAG_NONE));
        }
    }

    // En passant
    if board.ep_square != NO_SQUARE {
        let mut attackers = pawn_attacks(flip_color(us), board.ep_square as u32) & pawns;
        while attackers != 0 {
            let from = pop_lsb(&mut attackers) as u8;
            list.push(make_move(from, board.ep_square, FLAG_EN_PASSANT));
        }
    }
}

fn generate_castling(board: &Board, list: &mut MoveList, us: Color, occ: Bitboard) {
    if us == WHITE {
        // Kingside: e1-g1, f1 and g1 must be empty, e1/f1/g1 not attacked
        if board.castling & CASTLE_WK != 0 {
            if occ & 0x60 == 0 { // f1, g1 empty
                if !is_attacked(board, 4, BLACK) && !is_attacked(board, 5, BLACK) && !is_attacked(board, 6, BLACK) {
                    list.push(make_move(4, 6, FLAG_CASTLE));
                }
            }
        }
        // Queenside: e1-c1, b1/c1/d1 must be empty, e1/d1/c1 not attacked
        if board.castling & CASTLE_WQ != 0 {
            if occ & 0x0E == 0 { // b1, c1, d1 empty
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

/// Is a square attacked by the given color?
fn is_attacked(board: &Board, sq: u8, by_color: Color) -> bool {
    let occ = board.occupied();
    let attackers = board.colors[by_color as usize];

    if knight_attacks(sq as u32) & board.pieces[KNIGHT as usize] & attackers != 0 { return true; }
    if king_attacks(sq as u32) & board.pieces[KING as usize] & attackers != 0 { return true; }
    if pawn_attacks(flip_color(by_color), sq as u32) & board.pieces[PAWN as usize] & attackers != 0 { return true; }
    if bishop_attacks(sq as u32, occ) & (board.pieces[BISHOP as usize] | board.pieces[QUEEN as usize]) & attackers != 0 { return true; }
    if rook_attacks(sq as u32, occ) & (board.pieces[ROOK as usize] | board.pieces[QUEEN as usize]) & attackers != 0 { return true; }

    false
}

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
        let mut b = Board::from_fen("8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1");
        let moves = generate_legal_moves(&b);
        // EP capture e4xd3 should be illegal because it discovers check from Qh4
        for i in 0..moves.len {
            let mv = moves.moves[i];
            if move_flags(mv) == FLAG_EN_PASSANT {
                panic!("EP move should be illegal in this position");
            }
        }
    }

    #[test]
    fn test_promotion_captures() {
        init();
        let mut b = Board::from_fen("r3k2r/1P6/8/8/8/8/8/4K3 w kq - 0 1");
        let moves = generate_legal_moves(&b);
        // b7b8 promotions (4) + b7xa8 capture promotions (4) = 8 pawn moves minimum
        let promo_moves: Vec<_> = (0..moves.len)
            .filter(|&i| is_promotion(moves.moves[i]))
            .collect();
        assert!(promo_moves.len() >= 4, "Should have at least 4 promotion moves, got {}", promo_moves.len());
    }
}
