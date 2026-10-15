//! Move generation: pseudo-legal moves with legality filter.
//!
//! GenerateAllMoves returns pseudo-legal moves (captures then quiets).
//! GenerateEvasions produces fully legal evasions when in check.

use crate::bitboard::*;
use crate::attacks::*;
use crate::types::*;
use crate::board::Board;

pub const MAX_MOVES: usize = 256;

/// Move list with stack-allocated storage.
///
/// `moves` is `[MaybeUninit<Move>; MAX_MOVES]` rather than `[Move; MAX_MOVES]`
/// so the constructor can skip the 512-byte zero-init that the compiler
/// otherwise emits. With ~600k MoveList constructions per bench (movegen
/// callsites + every MovePicker), the eliminated store traffic is ~300 MB.
/// Same memset-skip pattern as `forward_with_l1_pairwise_inner` and
/// `apply_threat_deltas`; push() writes slot N before incrementing
/// len, so as_slice() only ever exposes initialized slots.
pub struct MoveList {
    moves: [std::mem::MaybeUninit<Move>; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    #[inline]
    pub fn new() -> Self {
        // SAFETY: `[MaybeUninit<Move>; N]::uninit().assume_init()` is sound —
        // each slot is itself a `MaybeUninit<Move>`, which has no validity
        // invariants. Reading these slots (e.g. via `assume_init_ref`)
        // requires writing them first; that invariant is enforced by `push`
        // and `as_slice` only exposes [..len].
        MoveList {
            moves: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len].write(mv);
        self.len += 1;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: push() writes slot `len` before incrementing, so [..len]
        // is fully initialized. MaybeUninit<T> has the same layout as T.
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr() as *const Move, self.len) }
    }

    /// Index access — equivalent to `as_slice()[i]` but slightly more concise
    /// for callers that already index by integer.
    #[inline(always)]
    pub fn get(&self, i: usize) -> Move {
        debug_assert!(i < self.len, "MoveList::get out of bounds: {} >= len {}", i, self.len);
        // SAFETY: caller invariant `i < len` → slot is initialized by push.
        unsafe { self.moves[i].assume_init() }
    }

    /// Mutable index access — used by sort/swap helpers in MovePicker.
    /// Caller must ensure `i < len` (debug-asserted).
    #[inline(always)]
    pub fn set(&mut self, i: usize, mv: Move) {
        debug_assert!(i < self.len, "MoveList::set out of bounds: {} >= len {}", i, self.len);
        self.moves[i].write(mv);
    }

    /// Swap two slots — used by MovePicker's selection-sort partial pass.
    /// Both indices must be `< len` (debug-asserted).
    #[inline(always)]
    pub fn swap(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.len && j < self.len);
        self.moves.swap(i, j);
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
    generate_castling(board, &mut list, us);

    list
}

// ---------------------------------------------------------------------------
// Castling move generation
// ---------------------------------------------------------------------------

fn generate_castling(board: &Board, list: &mut MoveList, us: Color) {
    // Rights for `us`: bit 0 = kingside, bit 1 = queenside.
    let rights = (board.castling >> (2 * us)) & 0b11;
    if rights == 0 {
        return;
    }
    // Encoding is king-from -> king-destination (g-file kingside, c-file
    // queenside) in standard chess and Chess960 alike; in 960 the two can be
    // the same square. `castle_is_legal` does the full check (right, rook on
    // its start square, empty paths, king's path unattacked with king and
    // castling rook lifted), so these are fully legal, not just pseudo-legal.
    // Kingside first, then queenside — the order movegen always used.
    if board.std_castling {
        // Standard geometry: fixed squares and masks, the same code as before
        // Chess960 support, so standard chess pays nothing for the generic path.
        let occ = board.occupied();
        if us == WHITE {
            if rights & 0b01 != 0 && occ & 0x60 == 0
                && !is_attacked(board, 4, BLACK) && !is_attacked(board, 5, BLACK) && !is_attacked(board, 6, BLACK) {
                list.push(make_move(4, 6, FLAG_CASTLE));
            }
            if rights & 0b10 != 0 && occ & 0x0E == 0
                && !is_attacked(board, 4, BLACK) && !is_attacked(board, 3, BLACK) && !is_attacked(board, 2, BLACK) {
                list.push(make_move(4, 2, FLAG_CASTLE));
            }
        } else {
            if rights & 0b01 != 0 && occ & (0x60u64 << 56) == 0
                && !is_attacked(board, 60, WHITE) && !is_attacked(board, 61, WHITE) && !is_attacked(board, 62, WHITE) {
                list.push(make_move(60, 62, FLAG_CASTLE));
            }
            if rights & 0b10 != 0 && occ & (0x0Eu64 << 56) == 0
                && !is_attacked(board, 60, WHITE) && !is_attacked(board, 59, WHITE) && !is_attacked(board, 58, WHITE) {
                list.push(make_move(60, 58, FLAG_CASTLE));
            }
        }
        return;
    }
    let ksq = board.king_sq(us);
    let back_rank: u8 = if us == WHITE { 0 } else { 56 };
    if rights & 0b01 != 0 && board.castle_is_legal(us, ksq, back_rank + 6) {
        list.push(make_move(ksq, back_rank + 6, FLAG_CASTLE));
    }
    if rights & 0b10 != 0 && board.castle_is_legal(us, ksq, back_rank + 2) {
        list.push(make_move(ksq, back_rank + 2, FLAG_CASTLE));
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
        list.push(caps.get(i));
    }

    // Quiets
    let quiets = generate_quiets(board);
    for i in 0..quiets.len {
        list.push(quiets.get(i));
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
pub fn generate_evasions(board: &Board, checkers: Bitboard, pinned: Bitboard) -> MoveList {
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
    let occ_full = board.occupied();
    // target = capture the checker OR block the ray between king and checker.
    // between() is empty for non-sliding checkers (knight, pawn).
    let target = (1u64 << checker_sq) | between(king_sq as u32, checker_sq);
    let block_target = between(king_sq as u32, checker_sq); // blocking squares only

    // Non-king candidates: exclude PINNED pieces up front, which makes every
    // generated non-king, non-EP evasion legal by construction and lets the
    // per-move legality oracle go.
    //
    // Why a pinned piece can never resolve a check: it may only move along
    // line(king, pinner), while every square that resolves the check (the
    // checker's square, or a square between king and checker) lies on
    // line(king, checker). The pinner is never the checker (a pinner has our
    // piece between it and the king, so it is not giving check), so these
    // are two distinct lines through the king square, and they meet only at
    // the king square -- which no non-king move can reach. In double check
    // we have already returned above. En passant is the one exception, since
    // removing the captured pawn can open a rank discovered attack, so EP
    // keeps the full oracle below.
    //
    // Previously every candidate paid is_legal(): for a non-pinned piece in
    // single check that call re-derives the check mask the target set already
    // encodes, then finds the piece unpinned and returns true. Same output,
    // strictly less work. Perft pins the equivalence.
    let non_king = our_pieces & !(1u64 << king_sq) & !pinned;

    // Knights
    let mut knights = board.pieces[KNIGHT as usize] & non_king;
    while knights != 0 {
        let from = pop_lsb(&mut knights) as u8;
        let mut attacks = knight_attacks(from as u32) & target;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Bishops
    let mut bishops = board.pieces[BISHOP as usize] & non_king;
    while bishops != 0 {
        let from = pop_lsb(&mut bishops) as u8;
        let mut attacks = bishop_attacks(from as u32, occ_full) & target;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Rooks
    let mut rooks = board.pieces[ROOK as usize] & non_king;
    while rooks != 0 {
        let from = pop_lsb(&mut rooks) as u8;
        let mut attacks = rook_attacks(from as u32, occ_full) & target;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Queens
    let mut queens = board.pieces[QUEEN as usize] & non_king;
    while queens != 0 {
        let from = pop_lsb(&mut queens) as u8;
        let mut attacks = queen_attacks(from as u32, occ_full) & target;
        while attacks != 0 {
            let to = pop_lsb(&mut attacks) as u8;
            list.push(make_move(from, to, FLAG_NONE));
        }
    }

    // Pawns
    let pawns = board.pieces[PAWN as usize] & non_king;
    let checker_bb = 1u64 << checker_sq;
    let empty = !occ_full;

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
                    if board.is_legal(m, pinned, checkers) {
                        list.push(m);
                    }
                }
                if ep_r != 0 {
                    let m = make_move(board.ep_square - 9, board.ep_square, FLAG_EN_PASSANT);
                    if board.is_legal(m, pinned, checkers) {
                        list.push(m);
                    }
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
                    if board.is_legal(m, pinned, checkers) {
                        list.push(m);
                    }
                }
                if ep_r != 0 {
                    let m = make_move(board.ep_square + 7, board.ep_square, FLAG_EN_PASSANT);
                    if board.is_legal(m, pinned, checkers) {
                        list.push(m);
                    }
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
        let mv = pseudo.get(i);
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
        let mv = moves.get(i);
        // make_move refuses a king capture, which only exists in an invalid
        // position (side not to move in check). Skip it rather than unmake a
        // move that was never made.
        if !board.make_move(mv) { continue; }
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
        let mv = moves.get(i);
        if !board.make_move(mv) { continue; } // king capture: invalid position
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
            let mv = moves.get(i);
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
            .map(|i| all.get(i))
            .filter(|&mv| is_quiet(&b, mv))
            .collect();
        expected.sort();

        // Get quiet moves directly
        let quiets = generate_quiets(&b);
        let mut actual: Vec<Move> = (0..quiets.len)
            .map(|i| quiets.get(i))
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
        let mut all_set: Vec<Move> = (0..all.len).map(|i| all.get(i)).collect();
        all_set.sort();

        let caps = generate_captures(&b);
        let quiets = generate_quiets(&b);
        let mut union: Vec<Move> = (0..caps.len).map(|i| caps.get(i))
            .chain((0..quiets.len).map(|i| quiets.get(i)))
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
            .filter(|&i| is_promotion(moves.get(i)))
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
            .map(|i| all.get(i))
            .filter(|&mv| b.is_legal(mv, pinned, checkers))
            .collect();
        expected.sort();

        // Legal moves via evasion generator
        let evasions = generate_evasions(&mut b, checkers, pinned);
        let mut actual: Vec<Move> = (0..evasions.len)
            .map(|i| evasions.get(i))
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

    // -----------------------------------------------------------------------
    // Chess960
    // -----------------------------------------------------------------------

    /// Chess960 positions with depth-3 perft counts. The counts come from
    /// python-chess (`chess.Board(fen, chess960=True)`, recursive perft over
    /// `legal_moves`), an independent move generator, not from Coda itself.
    /// The set covers: start setups with the king on the b/c/f/g file next to
    /// a rook; the king already on its castling square (castling moves only
    /// the rook); the rook already on its castling square (castling moves
    /// only the king); king and rook swapping squares; an enemy rook or queen
    /// on the back rank behind the castling rook (the castle must be
    /// illegal, since the rook was the only shield); an attacked destination
    /// or transit square; an inner rook as the castling rook; and mid-game
    /// positions from random play, most with castling rights left.
    ///
    /// Also: kings castling AWAY from their rook (king on the b-file castling
    /// queenside moves right, both colours) and kings passing over their own
    /// rook to a destination beyond it (king b1, inner kingside rook c1).
    const CHESS960_PERFT3: &[(&str, u64)] = &[
        ("r5kr/8/8/8/8/8/8/R5KR w HAha - 0 1", 11069), // king_stays_ks
        ("1r4kr/8/8/8/8/8/8/1R4KR w HBhb - 0 1", 10954), // ks_stay
        ("r1k4r/8/8/8/8/8/8/R1K4R w HAha - 0 1", 13776), // qs_kingstay
        ("1rk4r/8/8/8/8/8/8/1RK4R w HBhb - 0 1", 10990), // qs_kingstay_b
        ("r2k1r2/8/8/8/8/8/8/R2K1R2 w FAfa - 0 1", 11735), // rook_stays_ks
        ("3r1k1r/8/8/8/8/8/8/3R1K1R w HDhd - 0 1", 11601), // rook_stays_qs
        ("rk5r/8/8/8/8/8/8/RK5R w HAha - 0 1", 11099), // king_b_qs
        ("r4kr1/8/8/8/8/8/8/R4KR1 w GAga - 0 1", 9927), // swap_ks
        ("2rk3r/8/8/8/8/8/8/2RK3R w HChc - 0 1", 9924), // swap_qs_d
        ("1r1k3r/8/8/8/8/8/8/rR1K3R w HBh - 0 1", 10576), // xray_qs
        ("1r1k3r/8/8/8/8/8/8/qR1K3R w HBh - 0 1", 11904), // xray_qs_q
        ("r3k1r1/8/8/8/8/8/8/R3K1Rq w GAga - 0 1", 11782), // xray_ks
        ("r4kr1/8/8/8/8/8/8/R4KRr w GAga - 0 1", 7889), // xray_ks2
        ("r3k2r/8/8/8/8/8/8/1RK3r1 w B - 0 1", 1480), // xray_qs_far
        ("r3k2r/8/8/8/8/8/8/RK5r w A - 0 1", 1046), // dest_attacked
        ("r3k2r/8/8/8/8/8/6b1/RK5R w HA - 0 1", 13207), // transit_bishop
        ("rk2r2r/8/8/8/8/8/8/RK2R2R w EAea - 0 1", 25073), // inner_rook
        ("rk5r/1p6/8/8/8/8/8/1R2K2R w HBha - 0 1", 13222), // b8 king castles q-side rightward
        ("rk5r/1p6/8/8/8/8/8/1R2K2R b HBha - 0 1", 13627), // same, black to move
        ("r3k2r/8/8/8/8/8/8/RK5R w HAha - 0 1", 11291), // b1 king castles q-side rightward
        ("rk5r/8/8/8/8/8/8/R3K2R b HAha - 0 1", 11291), // b8 king, black to move
        ("rkr4r/8/8/8/8/8/8/RKR4R w CAca - 0 1", 22404), // king passes over inner k-side rook
        ("rkr4r/8/8/8/8/8/8/RKR4R b CAca - 0 1", 22404), // same, black to move
        ("r2k3r/8/8/8/8/8/8/R2K3R w HAha - 0 1", 13177), // rook passes king dest, lands on king start
        ("2rk3r/8/8/8/8/8/8/2RK3R b HChc - 0 1", 9924), // black_swap
        ("Rr1k3r/8/8/8/8/8/8/1R1K3R b HBhb - 0 1", 10738), // black_xray
        ("br1bqnr1/2pk2pp/5p1n/p3p3/4PP1P/8/PPPP1QP1/BRK2NRN w GB - 0 8", 27965), // mid801_14
        ("bqnrk1rb/ppp2pp1/4n3/4P3/4pPp1/8/PPP1P2P/BQNR1RKB b gd - 0 8", 26442), // mid195_15
        ("nbbrkr2/pppp1ppq/6Bp/4p3/3P4/2P5/PPN1PPPP/2BRKRQN b FDfd - 2 5", 14327), // mid356_9
        ("nbbrq1kr/3pppp1/pp2n2p/2p3P1/8/PP5P/2PPPPK1/NBBRQN1R b hd - 1 6", 17127), // mid132_11
        ("1rnkrbnq/p1p3pp/2b1pp2/1p1p4/P7/3N1P2/1PPPP1PP/BR1KRBNQ w EBeb - 0 6", 16185), // mid658_10
        ("b1kr1rqn/1pbpn1pp/2p1pp2/8/p1PPP3/7P/PPB2PPQ/BRN1KR1N w FB - 0 9", 25102), // mid641_16
        ("bnnbrkrq/pppp2p1/4p3/5p1p/8/P2P3P/1PP1PPP1/BNNBRRKQ b ge - 0 4", 15561), // mid81_7
        ("bbq1nrk1/2rQ2pp/1p3p2/4p3/2p2nP1/P2PP3/1PP2PNP/BBR2KNR w HC - 1 12", 45357), // mid496_22
        ("bqr1kbnr/1pppppQ1/4n1pp/p7/P4P2/1P5P/2PPP1PR/B1RNKBN1 b Chc - 6 7", 18400), // mid482_13
        ("r2bk2r/pppnp1pb/2n4p/3p1p2/8/1P2PB1q/P1PPNPPP/R1Q1KNBR b ha - 2 8", 36701), // mid525_15
        ("brknr1qn/p1pp1ppp/1p1bp3/8/5P1P/8/PPPPPQP1/BRKNRB1N w EBeb - 2 4", 21497), // mid834_6
        ("nq2b1kr/n1pr1p2/3bp3/p3pp1p/2p3PP/1P5R/P4K2/NQNRBB2 w h - 0 15", 53916), // mid26_28
        ("rk2n1qr/ppb1ppp1/2pp4/4nb1p/BPP3P1/P2P4/RK2PP1P/2B1NNQR b ha - 2 8", 43536), // mid741_15
        ("rk3brq/ppp1pnp1/3p1p1n/1b6/2P1P2p/P7/1P1P1PPP/RKNNB1RQ w GAga - 1 10", 14194), // mid762_18
        ("brqk1brn/p1nppppp/1pp5/8/6PP/1P4N1/P1PPPP2/BRQKNBR1 b GBgb - 0 4", 21369), // mid786_7
        ("brkbrqnn/1pp2ppp/p7/3p3B/Q3p3/4P3/PPPP1PPP/BRK1R1NN w EBeb - 0 5", 40732), // mid913_8
        ("rkbnqb1r/pp1ppp2/7p/2p3p1/P1P3nP/4P1NB/RP1P1P2/K1BNQ2R b ha - 1 10", 16027), // mid726_19
        ("n1bkrnqb/1rp1pppp/1p6/p2p4/8/P3N1P1/1PPPPPBP/NRBKR1Q1 w EBe - 0 5", 17409), // mid375_8
        ("2rqkbbr/2pppppp/pp1n1Q2/8/4P3/1P1n2P1/P1PP1P1P/RN2KBBR w HAh - 0 8", 3040), // mid446_14
        ("r1bknbqr/p3pp2/npPp2p1/7p/P7/2P5/2KPPPPP/RNB1NBQR b ha - 1 7", 19186), // mid550_13
        ("r1kq1rn1/nppbpp1p/8/p1p3pP/4P3/2b2Q2/PPPP1PP1/RK2BRNB b FA - 3 9", 32278), // mid827_17
        ("b2nnbr1/p1p2ppk/P7/Qr1pN2p/2PP1P2/7P/1P2PNP1/BR3BKR w HB - 0 15", 40635), // mid402_28
        ("rkn1qrbb/1ppppp1p/6p1/p1N5/3P4/4P3/PP1P1PPP/RK1NQRBB b FAfa - 0 5", 14358), // mid751_9
        ("nbbr2qr/p2k1ppp/2pppn2/8/1p3P1P/2PPP2R/PPN3P1/1BBRKQN1 w D - 1 9", 23843), // mid244_16
        ("rbbnnkqr/ppppppp1/8/7p/1P4N1/8/P1PPPPPP/RBB1NKQR w HAha - 0 4", 12184), // mid452_6
        ("rnbbqrkn/p1pppppp/8/1p6/PP4P1/8/2PPPP1P/RNBBQRKN w - - 0 5", 9838), // mid613_8
        ("1krnqn1b/r3p1p1/1ppp1p2/7p/pPb1P2Q/P3BP2/1NPP2PP/RKR2N1B w CAc - 0 12", 43886), // mid943_22
        ("1qkrbbnn/r1pppppp/pp6/8/2P5/P4P1N/1P1PP1PP/RQKRBB1N b DAd - 0 4", 15146), // mid890_7
        ("n2rq1br/ppkp1p2/B6p/2p3p1/P2P2P1/2P4P/1P1K1P1b/NN1RQ1BR w - - 3 14", 29407), // mid78_26
        ("Qrn1rbn1/p1pkp2p/3p1p2/1p4p1/1P3PP1/P7/1NPPP2P/qR1KRBN1 w EB - 0 8", 21336), // mid658_14
        ("bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1", 9006), // start0
        ("nbbnrkrq/pppppppp/8/8/8/8/PPPPPPPP/NBBNRKRQ w GEge - 0 1", 8914), // start84
        ("qbbnrknr/pppppppp/8/8/8/8/PPPPPPPP/QBBNRKNR w HEhe - 0 1", 9010), // start196
        ("bbqrnnkr/pppppppp/8/8/8/8/PPPPPPPP/BBQRNNKR w HDhd - 0 1", 9024), // start384
        ("bbrqnkrn/pppppppp/8/8/8/8/PPPPPPPP/BBRQNKRN w GCgc - 0 1", 8970), // start592
        ("rknnqbbr/pppppppp/8/8/8/8/PPPPPPPP/RKNNQBBR w HAha - 0 1", 9032), // start750
        ("rkbnrbnq/pppppppp/8/8/8/8/PPPPPPPP/RKBNRBNQ w EAea - 0 1", 8934), // start854
        ("brkrnnqb/pppppppp/8/8/8/8/PPPPPPPP/BRKRNNQB w DBdb - 0 1", 8950), // start947
    ];

    /// An INVALID position (side not to move in check: the b1 rook attacks the
    /// b8 king) must not panic. make_move refuses the king capture and perft
    /// skips it; python-chess, which does generate king captures here,
    /// agrees once those are excluded (77403 at depth 4).
    #[test]
    fn perft_invalid_position_skips_king_capture() {
        init();
        let mut b = Board::from_fen("rk5r/8/8/8/8/8/8/1R2K2R w HBha - 0 1");
        assert_eq!(perft(&mut b, 4), 77403);
    }

    #[test]
    fn perft_chess960_matches_python_chess() {
        init();
        for &(fen, expected) in CHESS960_PERFT3 {
            let mut b = Board::from_fen(fen);
            assert_eq!(perft(&mut b, 3), expected, "chess960 perft(3) mismatch: {}", fen);
        }
    }

    /// Random games from the Chess960 set: make/unmake must restore the board
    /// exactly, the incremental hash must equal a from-scratch hash after
    /// every move, every legal move must pass `is_pseudo_legal` + `is_legal`
    /// (the TT-move path), and FEN must round-trip. Castles are favoured so
    /// the 960 make/unmake paths get exercised heavily.
    #[test]
    fn chess960_make_unmake_hash_and_fen_roundtrip() {
        init();
        let mut state = 0x9600_0001u32;
        let mut rnd = || { state ^= state << 13; state ^= state >> 17; state ^= state << 5; state };
        let mut castles = 0usize;
        for &(fen, _) in CHESS960_PERFT3 {
            let start = Board::from_fen(fen);
            // FEN round-trip: what we print parses back to the same position.
            let again = Board::from_fen(&start.to_fen());
            assert_eq!(again.castling, start.castling, "castling round-trip {}", fen);
            assert_eq!(again.castle_rooks, start.castle_rooks, "castle rooks round-trip {}", fen);
            assert_eq!(again.hash, start.hash, "hash round-trip {}", fen);
            for _game in 0..8 {
                let mut b = start.clone();
                for _ply in 0..60 {
                    let legal = generate_legal_moves(&b);
                    if legal.len == 0 { break; }
                    let pinned = b.pinned();
                    let checkers = b.checkers();
                    for i in 0..legal.len {
                        let m = legal.get(i);
                        assert!(crate::movepicker::is_pseudo_legal(&b, m) && b.is_legal(m, pinned, checkers),
                            "legal move {} rejected by is_pseudo_legal/is_legal in {}", move_to_uci(m), b.to_fen());
                    }
                    let mut mv = legal.get(rnd() as usize % legal.len);
                    for i in 0..legal.len {
                        if move_flags(legal.get(i)) == FLAG_CASTLE && rnd() % 2 == 0 {
                            mv = legal.get(i);
                        }
                    }
                    if move_flags(mv) == FLAG_CASTLE { castles += 1; }
                    let before = b.clone();
                    assert!(b.make_move(mv));
                    assert_eq!(b.hash, b.compute_hash(), "incremental hash after {} from {}", move_to_uci(mv), before.to_fen());
                    let fen_after = b.to_fen();
                    assert_eq!(Board::from_fen(&fen_after).hash, b.hash, "FEN round-trip after {}", move_to_uci(mv));
                    b.unmake_move();
                    assert_eq!(b.pieces, before.pieces, "unmake pieces {}", move_to_uci(mv));
                    assert_eq!(b.colors, before.colors, "unmake colors {}", move_to_uci(mv));
                    assert_eq!(b.mailbox, before.mailbox, "unmake mailbox {}", move_to_uci(mv));
                    assert_eq!(b.castling, before.castling);
                    assert_eq!(b.hash, before.hash);
                    b.make_move(mv);
                }
            }
        }
        assert!(castles > 100, "only {} castles played", castles);
    }

    /// FEN parsing: X-FEN K/Q mean the outermost rook, Shredder letters name
    /// the file, and standard positions print exactly as before.
    #[test]
    fn chess960_fen_castling_fields() {
        init();
        // Standard: unchanged output.
        let b = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_eq!(b.castle_rooks, crate::board::STANDARD_CASTLE_ROOKS);
        assert!(b.to_fen().contains(" KQkq "));
        // Shredder letters and X-FEN letters give the same position.
        let s = Board::from_fen("1r1k2r1/8/8/8/8/8/8/1R1K2R1 w GBgb - 0 1");
        let x = Board::from_fen("1r1k2r1/8/8/8/8/8/8/1R1K2R1 w KQkq - 0 1");
        assert_eq!(s.castling, 0b1111);
        assert_eq!(s.castle_rooks, [6, 1, 62, 57]);
        assert_eq!(x.castle_rooks, s.castle_rooks);
        assert_eq!(s.hash, x.hash);
        assert!(s.to_fen().contains(" GBgb "), "{}", s.to_fen());
        // K = OUTERMOST rook: with rooks on f1 and h1, K is h1; F names f1.
        let k = Board::from_fen("4k3/8/8/8/8/8/8/R3KR1R w K - 0 1");
        assert_eq!(k.castle_rooks[0], 7);
        let f = Board::from_fen("4k3/8/8/8/8/8/8/R3KR1R w F - 0 1");
        assert_eq!(f.castle_rooks[0], 5);
        assert!(f.to_fen().contains(" F "), "{}", f.to_fen());
        // A right naming a missing rook is dropped.
        let none = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w KQ - 0 1");
        assert_eq!(none.castling, 0);
    }

    /// UCI move strings: standard form by default, king-takes-rook in 960
    /// mode, and the two kinds of king move that share from/to in 960 (king
    /// step f1g1 vs. castle with the king going f1->g1) stay distinct.
    #[test]
    fn chess960_uci_castle_strings() {
        init();
        let b = Board::from_fen("r4k1r/8/8/8/8/8/8/R4K1R w HAha - 0 1");
        let legal = generate_legal_moves(&b);
        let mut castle_ks = NO_MOVE;
        let mut step = NO_MOVE;
        for i in 0..legal.len {
            let m = legal.get(i);
            if move_flags(m) == FLAG_CASTLE && move_to(m) == 6 { castle_ks = m; }
            if move_flags(m) == FLAG_NONE && move_from(m) == 5 && move_to(m) == 6 { step = m; }
        }
        assert!(castle_ks != NO_MOVE && step != NO_MOVE);
        assert_eq!(b.uci_move_mode(castle_ks, true), "f1h1");
        assert_eq!(b.uci_move_mode(castle_ks, false), "f1g1");
        assert_eq!(b.uci_move_mode(step, true), "f1g1");
        assert!(!crate::types::move_matches(step, castle_ks));
        assert!(!crate::types::move_matches(castle_ks, step));
        // Standard position, 960 mode: e1h1 / e1a1.
        let s = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_eq!(s.uci_move_mode(make_move(4, 6, FLAG_CASTLE), true), "e1h1");
        assert_eq!(s.uci_move_mode(make_move(4, 2, FLAG_CASTLE), true), "e1a1");
        assert_eq!(s.uci_move_mode(make_move(4, 6, FLAG_CASTLE), false), "e1g1");
    }
}
