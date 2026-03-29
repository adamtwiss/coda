/// Static Exchange Evaluation (SEE).
/// Determines if a capture sequence is winning/losing.

use crate::attacks::*;
use crate::bitboard::*;
use crate::board::Board;
use crate::eval::see_value;
use crate::types::*;

/// Returns true if the SEE of the move is >= threshold.
#[inline]
pub fn see_ge(board: &Board, mv: Move, threshold: i32) -> bool {
    let from = move_from(mv) as u32;
    let to = move_to(mv) as u32;
    let flags = move_flags(mv);

    // Castling is never a capture
    if flags == FLAG_CASTLE {
        return 0 >= threshold;
    }

    let target_pt = board.piece_type_at(to as u8);
    let attacker_pt = board.piece_type_at(from as u8);

    // Initial balance: capture value minus threshold
    // GoChess: gain[0] = SEEPieceValues[captured]
    let mut balance = if flags == FLAG_EN_PASSANT {
        see_value(PAWN) // EP always captures a pawn
    } else if target_pt != NO_PIECE_TYPE {
        see_value(target_pt)
    } else {
        0
    };

    // GoChess does NOT handle promotion in SEE — match this behavior
    // (promotions are treated as pawn captures for SEE purposes)

    balance -= threshold;
    if balance < 0 {
        return false;
    }

    // Now assume we lose the attacker (GoChess: nextVictimValue = SEEPieceValues[attacker])
    balance -= see_value(attacker_pt);

    if balance >= 0 {
        return true;
    }

    // Iterative SEE
    // GoChess: occupied &^= SquareBB(from) — only removes initial attacker
    // (victim at 'to' stays in occupied; for EP, captured pawn also stays)
    let mut occ = board.occupied() ^ (1u64 << from);

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
#[inline]
fn find_lva(board: &Board, stm_attackers: Bitboard, _stm: Color) -> (u8, u32) {
    for pt in 0..6u8 {
        let bb = board.pieces[pt as usize] & stm_attackers;
        if bb != 0 {
            return (pt, lsb(bb));
        }
    }
    (NO_PIECE_TYPE, 0) // shouldn't happen
}

/// Compute full SEE value (for testing/comparison). Not used in search.
pub fn see_value_of(board: &Board, mv: Move) -> i32 {
    // Binary search for the SEE value using see_ge
    let mut lo = -2000i32;
    let mut hi = 2000i32;
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if see_ge(board, mv, mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

#[cfg(test)]
mod see_tests {
    use super::*;
    use crate::board::Board;
    use crate::types::*;
    use crate::movegen::generate_legal_moves;

    fn init() { crate::init(); }

    #[test]
    fn test_see_values() {
        init();
        
        // Italian game: test Bxf7+ and Nxe5
        let b = Board::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
        let moves = generate_legal_moves(&b);
        
        for i in 0..moves.len {
            let mv = moves.moves[i];
            let to = move_to(mv);
            let target = b.piece_type_at(to);
            if target != NO_PIECE_TYPE || move_flags(mv) == FLAG_EN_PASSANT {
                let val = see_value_of(&b, mv);
                let from_sq = move_from(mv);
                println!("  {}{}→{}{}: SEE={}", 
                    (b'a' + (from_sq % 8)) as char, (b'1' + (from_sq / 8)) as char,
                    (b'a' + (to % 8)) as char, (b'1' + (to / 8)) as char,
                    val);
            }
        }

        // Test a position with x-ray attacks
        println!("\nX-ray test: rook behind rook on open file");
        let b2 = Board::from_fen("1k1r3r/pp6/8/3p4/8/8/PP3R2/1K3R2 w - - 0 1");
        let moves2 = generate_legal_moves(&b2);
        for i in 0..moves2.len {
            let mv = moves2.moves[i];
            let to = move_to(mv);
            let target = b2.piece_type_at(to);
            if target != NO_PIECE_TYPE {
                let val = see_value_of(&b2, mv);
                let from_sq = move_from(mv);
                println!("  {}{}→{}{}: SEE={}", 
                    (b'a' + (from_sq % 8)) as char, (b'1' + (from_sq / 8)) as char,
                    (b'a' + (to % 8)) as char, (b'1' + (to / 8)) as char,
                    val);
            }
        }
        
        // Position with pawn x-ray
        println!("\nPawn structure test");
        let b3 = Board::from_fen("r1bqkbnr/ppp2ppp/2n5/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq d6 0 3");
        let moves3 = generate_legal_moves(&b3);
        for i in 0..moves3.len {
            let mv = moves3.moves[i];
            let to = move_to(mv);
            let target = b3.piece_type_at(to);
            if target != NO_PIECE_TYPE || move_flags(mv) == FLAG_EN_PASSANT {
                let val = see_value_of(&b3, mv);
                let from_sq = move_from(mv);
                println!("  {}{}→{}{}: SEE={}", 
                    (b'a' + (from_sq % 8)) as char, (b'1' + (from_sq / 8)) as char,
                    (b'a' + (to % 8)) as char, (b'1' + (to / 8)) as char,
                    val);
            }
        }
    }
}

#[cfg(test)]
mod see_xray_tests {
    use super::*;
    use crate::board::Board;
    use crate::types::*;
    use crate::movegen::generate_legal_moves;

    fn init() { crate::init(); }

    #[test]
    fn test_see_xray() {
        init();
        let positions = vec![
            ("Rook x-ray on d-file", "3r4/8/8/3p4/8/8/3R4/3RK2k w - - 0 1"),
            ("Bishop x-ray", "8/8/8/3p4/8/1B6/B7/4K2k w - - 0 1"),
            ("Queen behind bishop", "8/8/8/3p4/8/5B2/8/3QK2k w - - 0 1"),
            ("Pawn defended piece", "8/8/8/3p4/2P5/8/8/4K2k w - - 0 1"),
            ("Complex: NxP defended by B,R", "r1b1k2r/ppp2ppp/2n2n2/3Np3/2B5/8/PPPP1PPP/R1BQK2R w KQkq - 0 6"),
        ];
        
        for (name, fen) in &positions {
            let b = Board::from_fen(fen);
            println!("{}:", name);
            let moves = generate_legal_moves(&b);
            for i in 0..moves.len {
                let mv = moves.moves[i];
                let to = move_to(mv);
                if b.piece_type_at(to) != NO_PIECE_TYPE || move_flags(mv) == FLAG_EN_PASSANT {
                    let val = see_value_of(&b, mv);
                    let from = move_from(mv);
                    println!("  {}{}→{}{}: SEE={}", 
                        (b'a' + (from % 8)) as char, (b'1' + (from / 8)) as char,
                        (b'a' + (to % 8)) as char, (b'1' + (to / 8)) as char,
                        val);
                }
            }
        }
    }
}
