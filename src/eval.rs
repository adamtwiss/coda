//! NNUE evaluation wrapper plus SEE piece values.
//!
//! Coda's real evaluation is NNUE (`evaluate_nnue`); an NNUE net is required to
//! run. This module also holds the endgame mop-up gradient that supplements the
//! NNUE score, and `see_value` (SEE piece values used by move ordering / search,
//! unrelated to evaluation).

/// DOMINANT endgame "mop-up" gradient. The NNUE is materially correct in won
/// endgames but provides no MATING gradient and swings ~200cp across king
/// positions (noise w.r.t. the mate). To steer the search to a bare-king mate
/// the drive-to-edge gradient must EXCEED that noise — so EDGE spans ~600cp.
/// Gated to a lone king vs material that can FORCE mate (Q, R, BB, BN);
/// KB/KN/KNN-v-K are excluded (drawn — no drive). WHITE-relative centipawns.
// Magnitude window: must EXCEED the ~200cp NNUE position-noise but stay UNDER
// the protected material (~500cp rook) so king-driving never outweighs piece
// safety (EDGE=100 hung rooks). Env-tunable for sweeping the sweet spot.
#[inline(always)]
fn mopup_params() -> (i32, i32, i32) { (70, 15, 80) }
fn endgame_mopup(board: &crate::board::Board) -> i32 {
    use crate::bitboard::popcount;
    let (k, b, n) = (crate::types::KING as usize, crate::types::BISHOP as usize,
                     crate::types::KNIGHT as usize);
    let wpc = popcount(board.colors[0]);
    let bpc = popcount(board.colors[1]);
    let winner = if bpc == 1 && wpc > 1 { 0usize }
                 else if wpc == 1 && bpc > 1 { 1usize }
                 else { return 0 };
    let cw = board.colors[winner];
    let q = popcount(board.pieces[crate::types::QUEEN as usize] & cw);
    let r = popcount(board.pieces[crate::types::ROOK as usize] & cw);
    let bi = popcount(board.pieces[b] & cw);
    let kn = popcount(board.pieces[n] & cw);
    // matable material only — KB/KN/KNN cannot force mate, leave them alone
    if !(q > 0 || r > 0 || bi >= 2 || (bi >= 1 && kn >= 1)) {
        return 0;
    }
    let loser = 1 - winner;
    let lk = (board.pieces[k] & board.colors[loser]).trailing_zeros() as i32;
    let wk = (board.pieces[k] & board.colors[winner]).trailing_zeros() as i32;
    let (lf, lr) = (lk % 8, lk / 8);
    let (wf, wr) = (wk % 8, wk / 8);
    let cmd = (if lf < 4 { 3 - lf } else { lf - 4 }) + (if lr < 4 { 3 - lr } else { lr - 4 });
    let md = (lf - wf).abs() + (lr - wr).abs();
    let (p_edge, p_prox, p_corner) = mopup_params();
    let mut term = p_edge * cmd + p_prox * (14 - md);
    // KBN: only the bishop-coloured corners mate — bias the bare king there.
    let wb = board.pieces[b] & cw;
    if popcount(cw) == 3 && bi == 1 && kn == 1 {
        let bsq = wb.trailing_zeros() as i32;
        let light = ((bsq % 8) + (bsq / 8)) & 1 == 1;
        let corners: [(i32, i32); 2] = if light { [(0, 7), (7, 0)] } else { [(0, 0), (7, 7)] };
        let dmin = corners.iter().map(|&(cf, cr)| (lf - cf).abs() + (lr - cr).abs()).min().unwrap();
        term += p_corner * (7 - dmin);
    }
    if winner == 0 { term } else { -term }
}

/// Evaluate the position with the NNUE net, from the side-to-move's
/// perspective (centipawns), plus the endgame mop-up gradient.
pub fn evaluate_nnue(
    board: &crate::board::Board,
    net: &crate::nnue::NNUENet,
    acc: &mut crate::nnue::NNUEAccumulator,
    threat_stack: &crate::threat_accum::ThreatStack,
) -> i32 {
    acc.materialize(net, board);
    let pc = crate::nnue::piece_count(board);

    // DEBUG: compare ThreatStack vs full recompute
    #[cfg(debug_assertions)]
    if threat_stack.active && net.has_threats {
        static DBG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let c = DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if c < 20 {
            let h = net.hidden_size;
            let occ = board.colors[0] | board.colors[1];
            let wk = (board.pieces[crate::types::KING as usize] & board.colors[0]).trailing_zeros();
            let bk = (board.pieces[crate::types::KING as usize] & board.colors[1]).trailing_zeros();
            // Full recompute from scratch
            let mut check_w = vec![0i16; h];
            let mut check_b = vec![0i16; h];
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, crate::types::WHITE, (wk % 8) >= 4,
                |idx| { if idx < net.num_threat_features { let w = idx * h; for j in 0..h { check_w[j] += net.threat_weights[w + j] as i16; } } },
            );
            crate::threats::enumerate_threats(
                &board.pieces, &board.colors, &board.mailbox,
                occ, crate::types::BLACK, (bk % 8) >= 4,
                |idx| { if idx < net.num_threat_features { let w = idx * h; for j in 0..h { check_b[j] += net.threat_weights[w + j] as i16; } } },
            );
            let ts_w = threat_stack.values(crate::types::WHITE);
            let ts_b = threat_stack.values(crate::types::BLACK);
            let mut w_diff: i32 = 0;
            let mut b_diff: i32 = 0;
            for j in 0..h { w_diff += (ts_w[j] as i32 - check_w[j] as i32).abs(); }
            for j in 0..h { b_diff += (ts_b[j] as i32 - check_b[j] as i32).abs(); }
            if w_diff > 0 || b_diff > 0 {
                eprintln!("THREAT_STACK MISMATCH #{}: wdiff={} bdiff={} w_acc=[{},{}] w_chk=[{},{}]",
                    c, w_diff, b_diff, ts_w[0], ts_w[1], check_w[0], check_w[1]);
            }
        }
    }


    let mut v = net.forward_with_threats(acc, board.side_to_move, pc, threat_stack);
    // Dominant endgame mop-up gradient (lone-king-vs-matable only). WHITE-rel -> stm.
    let mu = endgame_mopup(board);
    v += if board.side_to_move == crate::types::WHITE { mu } else { -mu };
    // Eval-scale normalization (EVAL_SCALE_PCT, default 100 = no-op).
    let pct = crate::search::EVAL_SCALE_PCT.load(std::sync::atomic::Ordering::Relaxed);
    if pct != 100 { v * pct / 100 } else { v }
}

/// Material-only value of a piece type (midgame, for SEE).
///
/// A flat table rather than a `match`: the match compiled to a jump table
/// on `pt`, and with the piece type varying call to call (SEE exchange loop,
/// MVV ordering) that indirect jump was one of the hottest mispredicting
/// branches in the engine (LBR: ~5% of all mispredicts across `see_ge` and
/// `next_slow`). A load from an 8-entry table is branch-free. The `& 7`
/// makes the index provably in range so no bounds check is emitted;
/// NO_PIECE_TYPE (6) and 7 map to 0, exactly as the old `_ => 0` arm did.
pub const fn see_value(pt: u8) -> i32 {
    // Values aligned with consensus from top engines (Berserk, Obsidian,
    // Stormphrax). Old textbook values (100/320/330/500/900)
    // underestimated minor pieces by ~25% and rook/queen by ~20%.
    const SEE_VALUES: [i32; 8] = [
        100,   // PAWN
        420,   // KNIGHT (was 320)
        420,   // BISHOP (was 330, N=B consensus)
        640,   // ROOK (was 500)
        1200,  // QUEEN (was 900)
        20000, // KING
        0,     // NO_PIECE_TYPE
        0,
    ];
    SEE_VALUES[(pt & 7) as usize]
}
