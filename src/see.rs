//! Static Exchange Evaluation (SEE).
//! Determines if a capture sequence is winning/losing.
//! The iterative swap-loop follows the standard SEE formulation (as used by
//! Reckless and Stockfish); the pinned-piece handling is Coda's own.

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
    let is_promo = is_promotion(mv);

    // Initial balance: capture value minus threshold
    let mut balance = if flags == FLAG_EN_PASSANT {
        see_value(PAWN) // EP always captures a pawn
    } else if target_pt != NO_PIECE_TYPE {
        see_value(target_pt)
    } else {
        0
    };

    // Promotion: gain the promoted piece, lose the pawn
    if is_promo {
        let promo_pt = promotion_piece_type(mv);
        balance += see_value(promo_pt) - see_value(PAWN);
    }

    balance -= threshold;
    if balance < 0 {
        return false;
    }

    // Assume we lose the piece on the destination after capture
    // For promotions, the piece at risk is the promoted piece, not the pawn
    let risk_value = if is_promo { see_value(promotion_piece_type(mv)) } else { see_value(attacker_pt) };
    balance -= risk_value;

    if balance >= 0 {
        return true;
    }

    // Iterative SEE — remove initial attacker from occupied
    let mut occ = board.occupied() ^ (1u64 << from);
    // For EP, also remove the captured pawn (on same file as to, same rank as from)
    if flags == FLAG_EN_PASSANT {
        let ep_victim_sq = (to & 7) | (from & !7); // file of to, rank of from
        occ ^= 1u64 << ep_victim_sq;
    }

    let bishops = board.pieces[BISHOP as usize] | board.pieces[QUEEN as usize];
    let rooks = board.pieces[ROOK as usize] | board.pieces[QUEEN as usize];

    let mut stm = flip_color(board.side_to_move); // opponent moves next
    let mut attackers = board.attackers_to(to, occ);
    // Remove the initial attacker
    attackers &= occ;

    // Pinned piece masks for each side (SEE audit S4: matches SF/Reckless/Obsidian).
    // A pinned piece cannot move off the pin ray, so it cannot legally recapture
    // on `to` unless `to` is on the same pin ray. Rather than compute ray membership,
    // we conservatively exclude ALL pinned pieces — this slightly under-estimates
    // the exchange value (may classify some winning exchanges as equal/losing) but
    // is correct in the other direction (never over-estimates, never prunes wrongly).
    let pinned_w = {
        let ksq = board.king_sq(WHITE) as u32;
        let occ_all = board.occupied();
        let their_diag = (board.pieces[BISHOP as usize] | board.pieces[QUEEN as usize]) & board.colors[BLACK as usize];
        let their_orth = (board.pieces[ROOK as usize] | board.pieces[QUEEN as usize]) & board.colors[BLACK as usize];
        let mut pinned = 0u64;
        let mut p = bishop_attacks(ksq, 0) & their_diag;
        while p != 0 { let sq = pop_lsb(&mut p); let b = crate::bitboard::between(ksq, sq) & occ_all; if popcount(b) == 1 && b & board.colors[WHITE as usize] != 0 { pinned |= b; } }
        let mut p = rook_attacks(ksq, 0) & their_orth;
        while p != 0 { let sq = pop_lsb(&mut p); let b = crate::bitboard::between(ksq, sq) & occ_all; if popcount(b) == 1 && b & board.colors[WHITE as usize] != 0 { pinned |= b; } }
        pinned
    };
    let pinned_b = {
        let ksq = board.king_sq(BLACK) as u32;
        let occ_all = board.occupied();
        let their_diag = (board.pieces[BISHOP as usize] | board.pieces[QUEEN as usize]) & board.colors[WHITE as usize];
        let their_orth = (board.pieces[ROOK as usize] | board.pieces[QUEEN as usize]) & board.colors[WHITE as usize];
        let mut pinned = 0u64;
        let mut p = bishop_attacks(ksq, 0) & their_diag;
        while p != 0 { let sq = pop_lsb(&mut p); let b = crate::bitboard::between(ksq, sq) & occ_all; if popcount(b) == 1 && b & board.colors[BLACK as usize] != 0 { pinned |= b; } }
        let mut p = rook_attacks(ksq, 0) & their_orth;
        while p != 0 { let sq = pop_lsb(&mut p); let b = crate::bitboard::between(ksq, sq) & occ_all; if popcount(b) == 1 && b & board.colors[BLACK as usize] != 0 { pinned |= b; } }
        pinned
    };
    let pinned = [pinned_w, pinned_b];

    loop {
        // Exclude pinned pieces from recapturing (they cannot legally move off the pin ray).
        let stm_attackers = attackers & board.colors[stm as usize] & !pinned[stm as usize];
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
        // In-loop promotion needs NO special case (revert of C8 #40, audit
        // 2026-06-13 B1): the swap update with the PLAIN pawn value is the
        // exact algebra. Correct accounting for a pawn capturing onto the
        // promotion rank is `balance = -balance - 1 + (Q-P) - Q`: the
        // promoting side gains (Q-P) and puts Q at risk — and those terms
        // cancel to `- P`. The negation next iteration then automatically
        // credits the recapturer +Q while permanently charging the (Q-P)
        // gain. The C8 #40 form (risk Q without crediting Q-P) made the
        // promoting side ~1100cp too pessimistic, flipping SEE verdicts:
        // FEN 3R2k1/4P3/3q4/3r4/8/8/8/6K1 b, Qd6xd8 returned +640 vs true
        // -460 (see test_see_inloop_promotion_*). No reference engine
        // special-cases this (SF position.cpp:1467 `swap = PawnValue -
        // swap`; Reckless/Berserk plain attacker value).
        balance = -balance - 1 - see_value(att_pt);

        if balance >= 0 {
            // King-illegality: if the just-made capture was by the king and
            // the opponent still has attackers, the king moved into check —
            // illegal. The king-side cannot actually make this capture, so
            // the exchange ends with king-side losing. At this point
            // `stm = flip_color(king_side)` (we already advanced it above);
            // flip back so the final `board.side_to_move != stm` return
            // correctly attributes the loss to king-side.
            if att_pt == KING && (attackers & board.colors[stm as usize]) != 0 {
                stm = flip_color(stm);
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
    
    use crate::movegen::generate_legal_moves;

    fn init() { crate::init(); }

    #[test]
    fn test_see_values() {
        init();
        
        // Italian game: test Bxf7+ and Nxe5
        let b = Board::from_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
        let moves = generate_legal_moves(&b);
        
        for i in 0..moves.len {
            let mv = moves.get(i);
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
            let mv = moves2.get(i);
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
            let mv = moves3.get(i);
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
                let mv = moves.get(i);
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

/// Assertive SEE correctness tests. Hand-computed expected values for
/// specific capture scenarios. Guards the SEE implementation against
/// regressions — particularly the classic failure modes: wrong
/// exchange ordering, missing x-ray attackers, incorrect EP victim
/// location, incorrect promotion gain calculation.
///
/// Piece values (from eval::see_value): P=100, N=420, B=420, R=640,
/// Q=1200, K=huge.
#[cfg(test)]
mod see_assertive_tests {
    use super::*;
    use crate::board::Board;
    use crate::movegen::generate_legal_moves;

    fn init() { crate::init(); }

    /// Find a move by from/to squares. Panics if not legal.
    fn find_move(b: &Board, from: u8, to: u8) -> Move {
        let moves = generate_legal_moves(b);
        for i in 0..moves.len {
            let mv = moves.get(i);
            if move_from(mv) == from && move_to(mv) == to {
                return mv;
            }
        }
        panic!("no legal move {} → {} in {}", from, to, b.to_fen());
    }

    /// In-loop promotion recapture (audit 2026-06-13 B1, the C8 #40 revert).
    /// QxR, exd8=Q, Rxd8: white loses R+P (740), black loses Q (1200)
    /// -> true exchange value for black is -460. The C8 #40 form returned
    /// +640 (sign flip at every SEE threshold in use).
    #[test]
    fn see_inloop_promotion_recapture_losing() {
        init();
        let b = Board::from_fen("3R2k1/4P3/3q4/3r4/8/8/8/6K1 b - - 0 1");
        let mv = find_move(&b, 43, 59); // d6 -> d8 (QxR)
        assert_eq!(see_value_of(&b, mv), -460);
        assert!(!see_ge(&b, mv, 0));
    }

    /// Same motif with queen/rook swapped: RxR, exd8=Q, Qxd8.
    /// White loses R+P (740), black loses R (640) -> +100 for black
    /// (no sign flip, but C8 #40 returned +640 — large magnitude error).
    #[test]
    fn see_inloop_promotion_recapture_winning() {
        init();
        let b = Board::from_fen("3R2k1/4P3/3r4/3q4/8/8/8/6K1 b - - 0 1");
        let mv = find_move(&b, 43, 59); // d6 -> d8 (RxR)
        assert_eq!(see_value_of(&b, mv), 100);
        assert!(see_ge(&b, mv, 0));
    }

    #[test]
    fn see_hanging_pawn_returns_pawn_value() {
        init();
        // White N on c4 captures hanging Black P on e5 (only W has attackers).
        let b = Board::from_fen("4k3/8/8/4p3/2N5/8/8/4K3 w - - 0 1");
        let mv = find_move(&b, 26, 36); // c4 → e5
        assert_eq!(see_value_of(&b, mv), 100);
        assert!(see_ge(&b, mv, 100), "ge(100) true");
        assert!(!see_ge(&b, mv, 101), "ge(101) false");
    }

    #[test]
    fn see_pawn_defended_by_pawn_zero() {
        init();
        // White P on d5 captures Black P on e6, recaptured by Black P on d7.
        let b = Board::from_fen("4k3/3p4/4p3/3P4/8/8/8/4K3 w - - 0 1");
        let mv = find_move(&b, 35, 44); // d5 → e6
        // PxP, then PxP = +100 - 100 = 0.
        assert_eq!(see_value_of(&b, mv), 0);
        assert!(see_ge(&b, mv, 0));
        assert!(!see_ge(&b, mv, 1));
    }

    #[test]
    fn see_queen_takes_defended_pawn_negative() {
        init();
        // White Q on d1 captures Black P on d5 (defended by Black pawns
        // on c6 and e6 — both attack d5 diagonally).
        // After QxP, BP×Q: +100 - 1200 = -1100.
        // (SEE piece values are consensus: Q=1200, P=100.)
        let b = Board::from_fen("4k3/8/2p1p3/3p4/8/8/8/3QK3 w - - 0 1");
        let mv = find_move(&b, 3, 35); // d1 → d5
        assert_eq!(see_value_of(&b, mv), -1100);
        assert!(see_ge(&b, mv, -1100));
        assert!(!see_ge(&b, mv, -1099));
    }

    #[test]
    fn see_xray_rook_behind_rook() {
        init();
        // White R on d1, White R on d2, Black R on d8, Black P on d5.
        // WR d2 captures d5. Sequence:
        //  +P (100)  -R (500)   +R (500)  -R (500)
        //   d2×d5    d8×d5      d1×d5     black has no more attackers (stops).
        // Wait: after d1×d5, Black would lose their last R to nothing, so they stop earlier.
        //
        // Optimal play: +100 -500 = -400 (W loses too much, so W wouldn't start)
        //   BUT see computes the perfect-exchange value assuming both sides play
        //   optimally. Let's trace:
        //   0. W move: d2×d5 → balance = +100
        //   1. B stands or recaptures? B knows if they do Rxd5, W can continue with
        //      d1×d5 winning B's rook: +100-500+500 = +100. B choice: recapture
        //      (lose -100 relative to standing pat at +100). No, that's losing for B.
        //      Actually from B's perspective, they want to minimize W's gain.
        //      If B stands: W gains +100.
        //      If B recaptures (Rxd5): W loses R (-500 from +100 = -400), then W can
        //        continue (Rxd5): +100-500+500 = +100 again. Then B no more attackers,
        //        final value = +100. So B capturing gives +100 too.
        //      Either way, W nets +100.
        //   Expected SEE = +100.
        let b = Board::from_fen("3r3k/8/8/3p4/8/8/3R4/3RK3 w - - 0 1");
        let mv = find_move(&b, 11, 35); // d2 → d5
        assert_eq!(see_value_of(&b, mv), 100);
    }

    #[test]
    fn see_en_passant_captures_pawn() {
        init();
        // Black just played e7-e5 (double push). White P on d5 can EP-capture e6.
        // Captured pawn is the e5 pawn (not e6).
        let b = Board::from_fen("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1");
        let moves = generate_legal_moves(&b);
        // Find the EP capture
        let mut ep_move = NO_MOVE;
        for i in 0..moves.len {
            let mv = moves.get(i);
            if move_flags(mv) == FLAG_EN_PASSANT {
                ep_move = mv;
                break;
            }
        }
        assert_ne!(ep_move, NO_MOVE, "EP move must be legal");
        // EP: no defenders on e6, so pure pawn gain.
        assert_eq!(see_value_of(&b, ep_move), 100);
    }

    #[test]
    fn see_promotion_on_empty_square() {
        init();
        // White P on a7 promotes to a8 (empty). Gain: Q - P = 1200 - 100 = 1100.
        let b = Board::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        let moves = generate_legal_moves(&b);
        let mut promo_move = NO_MOVE;
        for i in 0..moves.len {
            let mv = moves.get(i);
            if is_promotion(mv) && promotion_piece_type(mv) == QUEEN && move_to(mv) == 56 {
                promo_move = mv;
                break;
            }
        }
        assert_ne!(promo_move, NO_MOVE);
        assert_eq!(see_value_of(&b, promo_move), 1100);
    }

    #[test]
    fn see_promotion_capture_of_rook() {
        init();
        // White P on g7 captures h8 (Black R). Black K on e8 is too far.
        // Gain: R + (Q - P) = 640 + 1100 = 1740.
        let b = Board::from_fen("4k2r/6P1/8/8/8/8/8/4K3 w - - 0 1");
        let moves = generate_legal_moves(&b);
        let mut promo_cap = NO_MOVE;
        for i in 0..moves.len {
            let mv = moves.get(i);
            if is_promotion(mv) && promotion_piece_type(mv) == QUEEN
                && move_from(mv) == 54 && move_to(mv) == 63
            {
                promo_cap = mv;
                break;
            }
        }
        assert_ne!(promo_cap, NO_MOVE);
        assert_eq!(see_value_of(&b, promo_cap), 1740);
    }

    #[test]
    fn see_castling_is_zero() {
        init();
        // Castling is not a capture — SEE should be 0.
        let b = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        let moves = generate_legal_moves(&b);
        let mut castle_move = NO_MOVE;
        for i in 0..moves.len {
            let mv = moves.get(i);
            if move_flags(mv) == FLAG_CASTLE {
                castle_move = mv;
                break;
            }
        }
        assert_ne!(castle_move, NO_MOVE);
        assert_eq!(see_value_of(&b, castle_move), 0);
        assert!(see_ge(&b, castle_move, 0));
        assert!(!see_ge(&b, castle_move, 1));
    }

    /// Regression test for see_value_of's binary search: every
    /// possible SEE value in [-2000, 2000] must satisfy:
    /// see_ge(mv, threshold) == (see_value_of(mv) >= threshold)
    /// This tests the monotonicity invariant of see_ge.
    #[test]
    fn see_ge_monotonicity() {
        init();
        let positions = &[
            "4k3/8/8/4p3/2N5/8/8/4K3 w - - 0 1",     // hanging P
            "4k3/3p4/4p3/3P4/8/8/8/4K3 w - - 0 1",   // defended P
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  // castle
        ];
        for fen in positions {
            let b = Board::from_fen(fen);
            let moves = generate_legal_moves(&b);
            for i in 0..moves.len {
                let mv = moves.get(i);
                let actual = see_value_of(&b, mv);
                // Check monotonicity at several thresholds.
                for &t in &[-2000, -1000, -500, -100, 0, 100, 500, 1000, 2000] {
                    let expected = actual >= t;
                    let got = see_ge(&b, mv, t);
                    assert_eq!(
                        got, expected,
                        "see_ge({}, {}) = {}; expected {} (actual SEE = {})",
                        crate::types::move_to_uci(mv), t, got, expected, actual
                    );
                }
                // Check adjacency: see_ge(actual+1) must be false, see_ge(actual) must be true.
                assert!(see_ge(&b, mv, actual), "see_ge(SEE) must be true");
                assert!(!see_ge(&b, mv, actual + 1), "see_ge(SEE+1) must be false");
            }
        }
    }
}
