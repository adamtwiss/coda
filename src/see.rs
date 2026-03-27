/// Static Exchange Evaluation (SEE).
/// Determines if a capture sequence is winning/losing.

use crate::attacks::*;
use crate::bitboard::*;
use crate::board::Board;
use crate::eval::see_value;
use crate::types::*;

/// Returns true if the SEE of the move is >= threshold.
pub fn see_ge(board: &Board, mv: Move, threshold: i32) -> bool {
    let from = move_from(mv) as u32;
    let to = move_to(mv) as u32;
    let flags = move_flags(mv);

    // Special moves
    if flags == FLAG_EN_PASSANT {
        // EP always captures a pawn, and attacker is a pawn
        return see_value(PAWN) - see_value(PAWN) >= threshold;
    }
    if flags == FLAG_CASTLE {
        return 0 >= threshold;
    }

    let target_pt = board.piece_type_at(to as u8);
    let attacker_pt = board.piece_type_at(from as u8);

    // Initial balance: capture value minus threshold
    let mut balance = if target_pt != NO_PIECE_TYPE {
        see_value(target_pt)
    } else {
        0
    };

    // Promotion bonus
    if is_promotion(mv) {
        let promo_pt = promotion_piece_type(mv);
        balance += see_value(promo_pt) - see_value(PAWN);
    }

    balance -= threshold;
    if balance < 0 {
        return false;
    }

    // Now assume we lose the attacker
    balance -= if is_promotion(mv) {
        see_value(promotion_piece_type(mv))
    } else {
        see_value(attacker_pt)
    };

    if balance >= 0 {
        return true;
    }

    // Iterative SEE
    let mut occ = board.occupied() ^ (1u64 << from) ^ (1u64 << to);
    if target_pt != NO_PIECE_TYPE {
        // Target piece is removed
    }

    let bishops = board.pieces[BISHOP as usize] | board.pieces[QUEEN as usize];
    let rooks = board.pieces[ROOK as usize] | board.pieces[QUEEN as usize];

    let mut stm = flip_color(board.side_to_move); // opponent moves next
    let mut attackers = board.attackers_to(to, occ);
    // Remove the initial attacker
    attackers &= occ;

    loop {
        let stm_attackers = attackers & board.colors[stm as usize];
        if stm_attackers == 0 {
            break;
        }

        // Find least valuable attacker
        let (att_pt, att_sq) = find_lva(board, stm_attackers, stm);

        // Remove attacker from occ
        occ ^= 1u64 << att_sq;

        // Add x-ray attackers through this square
        if att_pt == PAWN || att_pt == BISHOP || att_pt == QUEEN {
            attackers |= bishop_attacks(to, occ) & bishops;
        }
        if att_pt == ROOK || att_pt == QUEEN {
            attackers |= rook_attacks(to, occ) & rooks;
        }
        attackers &= occ;

        stm = flip_color(stm);
        balance = -balance - 1 - see_value(att_pt);

        if balance >= 0 {
            // If the attacker is king and opponent still has attackers, king can't capture
            if att_pt == KING && (attackers & board.colors[stm as usize]) != 0 {
                // Side that just moved with king loses
                break;
            }
            break;
        }
    }

    // If it's our opponent's turn and balance >= 0, we win
    // stm is the side that needs to make a move but chose not to (or can't)
    board.side_to_move != stm
}

/// Find the least valuable attacker and its square.
fn find_lva(board: &Board, stm_attackers: Bitboard, stm: Color) -> (u8, u32) {
    for pt in 0..6u8 {
        let bb = board.pieces[pt as usize] & stm_attackers;
        if bb != 0 {
            return (pt, lsb(bb));
        }
    }
    (NO_PIECE_TYPE, 0) // shouldn't happen
}
