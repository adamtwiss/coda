/// Main search: negamax with alpha-beta, iterative deepening, PVS, aspiration windows.
/// All pruning parameters ported from GoChess (tuned values).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::bitboard::*;
use crate::board::Board;
use crate::eval::{evaluate, see_value};
use crate::movegen::generate_legal_moves;
use crate::movepicker::*;
use crate::see::see_ge;
use crate::tt::*;
use crate::types::*;

const MAX_PLY: usize = 128;
const INFINITY: i32 = 32000;

/// Search limits.
pub struct SearchLimits {
    pub depth: i32,
    pub movetime: u64,    // milliseconds
    pub wtime: u64,
    pub btime: u64,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: u32,
    pub nodes: u64,
    pub infinite: bool,
}

impl SearchLimits {
    pub fn new() -> Self {
        SearchLimits {
            depth: 100,
            movetime: 0,
            wtime: 0,
            btime: 0,
            winc: 0,
            binc: 0,
            movestogo: 0,
            nodes: 0,
            infinite: false,
        }
    }
}

/// Search state for one thread.
pub struct SearchInfo {
    pub nodes: u64,
    pub tt: TT,
    pub history: History,
    pub stop: AtomicBool,
    pub start_time: Instant,
    pub time_limit: u64,  // ms
    pub max_depth: i32,
    pub max_nodes: u64,
    pub sel_depth: i32,
    prev_moves: [Move; MAX_PLY], // previous move at each ply
}

impl SearchInfo {
    pub fn new(tt_mb: usize) -> Self {
        SearchInfo {
            nodes: 0,
            tt: TT::new(tt_mb),
            history: History::new(),
            stop: AtomicBool::new(false),
            start_time: Instant::now(),
            time_limit: 0,
            max_depth: 100,
            max_nodes: 0,
            sel_depth: 0,
            prev_moves: [NO_MOVE; MAX_PLY],
        }
    }

    fn should_stop(&self) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }
        if self.max_nodes > 0 && self.nodes >= self.max_nodes {
            return true;
        }
        // Check time every 4096 nodes
        if self.nodes & 4095 == 0 && self.time_limit > 0 {
            let elapsed = self.start_time.elapsed().as_millis() as u64;
            if elapsed >= self.time_limit {
                self.stop.store(true, Ordering::Relaxed);
                return true;
            }
        }
        false
    }
}

/// LMR reduction table.
static mut LMR_TABLE: [[i32; 64]; 64] = [[0; 64]; 64];

pub fn init_lmr() {
    for depth in 1..64 {
        for moves in 1..64 {
            // Quiet: C=1.30, Capture: C=1.80 (we use quiet here, adjust at call site)
            unsafe {
                LMR_TABLE[depth][moves] = (1.30 + (depth as f64).ln() * (moves as f64).ln() / 2.36) as i32;
            }
        }
    }
}

fn lmr_reduction(depth: i32, moves: i32) -> i32 {
    let d = (depth as usize).min(63);
    let m = (moves as usize).min(63);
    unsafe { LMR_TABLE[d][m] }
}

/// Run iterative deepening search.
pub fn search(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits) -> Move {
    info.start_time = Instant::now();
    info.stop.store(false, Ordering::Relaxed);
    info.nodes = 0;
    info.sel_depth = 0;

    // Time management
    let (our_time, our_inc) = if board.side_to_move == WHITE {
        (limits.wtime, limits.winc)
    } else {
        (limits.btime, limits.binc)
    };

    if limits.movetime > 0 {
        info.time_limit = limits.movetime;
    } else if our_time > 0 {
        let moves_left = if limits.movestogo > 0 { limits.movestogo as u64 } else { 30 };
        info.time_limit = our_time / moves_left + our_inc / 2;
        // Don't use more than 50% of remaining time
        info.time_limit = info.time_limit.min(our_time / 2);
    } else if !limits.infinite {
        info.time_limit = 0; // No time limit (depth-limited)
    }

    info.max_depth = limits.depth;
    info.max_nodes = limits.nodes;

    info.tt.new_search();

    let mut best_move = NO_MOVE;
    let mut prev_score = 0i32;

    // Get a fallback move
    let legal = generate_legal_moves(board);
    if legal.len > 0 {
        best_move = legal.moves[0];
    }

    for depth in 1..=info.max_depth {
        if info.should_stop() { break; }

        let score;

        // Aspiration windows
        if depth >= 5 {
            let mut delta = 15i32;
            let mut alpha = (prev_score - delta).max(-INFINITY);
            let mut beta = (prev_score + delta).min(INFINITY);
            let mut asp_result = prev_score; // fallback

            loop {
                let result = negamax(board, info, alpha, beta, depth, 0, false);

                if info.should_stop() {
                    asp_result = result;
                    break;
                }

                if result <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (result - delta).max(-INFINITY);
                } else if result >= beta {
                    beta = (result + delta).min(INFINITY);
                } else {
                    asp_result = result;
                    break;
                }

                delta += delta / 2;
                if delta > 500 {
                    alpha = -INFINITY;
                    beta = INFINITY;
                }
            }

            score = asp_result;
            if info.should_stop() && depth > 1 { break; }
        } else {
            score = negamax(board, info, -INFINITY, INFINITY, depth, 0, false);
            if info.should_stop() && depth > 1 { break; }
        }

        // Get best move from TT
        let tt_entry = info.tt.probe(board.hash);
        if tt_entry.hit && tt_entry.best_move != NO_MOVE {
            best_move = tt_entry.best_move;
        }

        prev_score = if !info.should_stop() { score } else { prev_score };

        // UCI info output
        let elapsed = info.start_time.elapsed().as_millis() as u64;
        let nps = if elapsed > 0 { info.nodes * 1000 / elapsed } else { 0 };
        let score_str = if is_mate_score(prev_score) {
            let mate_in = if prev_score > 0 {
                (MATE_SCORE - prev_score + 1) / 2
            } else {
                -(MATE_SCORE + prev_score + 1) / 2
            };
            format!("score mate {}", mate_in)
        } else {
            format!("score cp {}", prev_score)
        };

        println!(
            "info depth {} seldepth {} {} nodes {} nps {} time {} hashfull {} pv {}",
            depth, info.sel_depth, score_str, info.nodes, nps, elapsed,
            info.tt.hashfull(), move_to_uci(best_move)
        );
    }

    best_move
}

/// Negamax alpha-beta search.
fn negamax(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    mut beta: i32,
    mut depth: i32,
    ply: i32,
    cut_node: bool,
) -> i32 {
    if info.should_stop() {
        return 0;
    }

    // Quiescence at depth 0
    if depth <= 0 {
        return quiescence(board, info, alpha, beta, ply);
    }

    info.nodes += 1;
    let ply_u = ply as usize;
    let is_root = ply == 0;
    let is_pv = beta - alpha > 1;

    if ply_u >= MAX_PLY - 1 {
        return evaluate(board);
    }

    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    // Draw detection (not at root)
    if !is_root {
        if board.halfmove >= 100 {
            return 0;
        }
        // Simple repetition: check if current hash appeared before
        // Full repetition detection would require hash history
    }

    // TT probe
    let tt_entry = info.tt.probe(board.hash);
    let tt_move = if tt_entry.hit { tt_entry.best_move } else { NO_MOVE };
    let tt_score = if tt_entry.hit { score_from_tt(tt_entry.score, ply) } else { 0 };

    // TT cutoff (non-PV only)
    if !is_pv && tt_entry.hit && tt_entry.depth >= depth {
        match tt_entry.flag {
            TT_FLAG_EXACT => return tt_score,
            TT_FLAG_LOWER => {
                if tt_score >= beta { return tt_score; }
            }
            TT_FLAG_UPPER => {
                if tt_score <= alpha { return tt_score; }
            }
            _ => {}
        }
    }

    let in_check = board.in_check();

    // Check extension
    if in_check {
        depth += 1;
    }

    // Static eval
    let static_eval = if in_check {
        -INFINITY // Don't trust eval in check
    } else if tt_entry.hit && tt_entry.static_eval != 0 {
        tt_entry.static_eval
    } else {
        evaluate(board)
    };

    let improving = !in_check && ply >= 2 && static_eval > evaluate(board); // simplified

    // Razoring
    if !is_pv && !in_check && depth <= 3 {
        let razor_margin = 400 + depth as i32 * 100;
        if static_eval + razor_margin <= alpha {
            let q_score = quiescence(board, info, alpha, beta, ply);
            if q_score <= alpha {
                return q_score;
            }
        }
    }

    // Reverse Futility Pruning (RFP)
    if !is_pv && !in_check && depth <= 7 {
        let rfp_margin = 70 + 100 * if improving { 1 } else { 0 };
        if static_eval - rfp_margin * depth as i32 >= beta {
            return static_eval;
        }
    }

    // Null Move Pruning
    if !is_pv && !in_check && depth >= 3 && static_eval >= beta {
        // Ensure we have non-pawn material
        let us = board.side_to_move;
        let non_pawn = board.colors[us as usize] & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
        if non_pawn != 0 {
            let r = 3 + depth / 3 + ((static_eval - beta) / 200).min(3);

            board.make_null_move();
            let null_score = -negamax(board, info, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
            board.unmake_null_move();

            if null_score >= beta {
                // Verification at high depth
                if depth >= 12 {
                    let v = negamax(board, info, beta - 1, beta, depth - r, ply + 1, false);
                    if v >= beta {
                        return null_score;
                    }
                } else {
                    return null_score;
                }
            }
        }
    }

    // ProbCut
    if !is_pv && !in_check && depth >= 5 {
        let probcut_beta = beta + 200;
        let mut pc_picker = QMovePicker::new(board);
        let pc_in_check = in_check;

        loop {
            let mv = pc_picker.next(board, pc_in_check);
            if mv == NO_MOVE { break; }

            if !see_ge(board, mv, probcut_beta - static_eval) { continue; }

            board.make_move(mv);
            let mut score = -quiescence(board, info, -probcut_beta, -probcut_beta + 1, ply + 1);
            if score >= probcut_beta {
                score = -negamax(board, info, -probcut_beta, -probcut_beta + 1, depth - 4, ply + 1, !cut_node);
            }
            board.unmake_move();

            if score >= probcut_beta {
                return score;
            }
        }
    }

    // IIR: Internal Iterative Reduction
    if tt_move == NO_MOVE && depth >= 4 {
        depth -= 1;
    }

    let prev_move = if ply > 0 { info.prev_moves[ply_u - 1] } else { NO_MOVE };
    let mut picker = MovePicker::new(board, tt_move, ply_u, &info.history, prev_move);

    let mut best_score = -INFINITY;
    let mut best_move = NO_MOVE;
    let mut moves_tried = 0;
    let mut quiets_tried: Vec<Move> = Vec::new();

    loop {
        let mv = picker.next(board, &info.history, prev_move);
        if mv == NO_MOVE { break; }

        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);
        let is_capture = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
        let is_promo = is_promotion(mv);

        moves_tried += 1;

        // Pruning for non-root, non-PV, late moves
        if !is_root && best_score > -TB_WIN && !in_check {
            // Late Move Pruning (LMP)
            if !is_capture && !is_promo && moves_tried > 3 + depth as usize * depth as usize {
                continue;
            }

            // History pruning for quiets
            if !is_capture && !is_promo && depth <= 8 {
                let hist = info.history.quiet_score(board, mv, prev_move);
                if hist < -1500 * depth as i32 {
                    continue;
                }
            }

            // Futility pruning
            if !is_capture && !is_promo && depth <= 8 {
                let futility_margin = 60 + (depth as i32) * 60;
                if static_eval + futility_margin <= alpha {
                    continue;
                }
            }

            // SEE pruning
            if depth <= 8 {
                let see_threshold = if is_capture {
                    -(depth as i32) * 100
                } else {
                    -20 * depth as i32 * depth as i32
                };
                if !see_ge(board, mv, see_threshold) {
                    continue;
                }
            }
        }

        // Record previous move for continuation history
        if ply_u < MAX_PLY {
            info.prev_moves[ply_u] = mv;
        }

        board.make_move(mv);

        let mut score;
        let new_depth = depth - 1;

        // LMR
        if depth >= 3 && moves_tried > 1 + if is_pv { 1 } else { 0 } {
            let mut r = lmr_reduction(depth, moves_tried as i32);

            // Adjustments
            if !is_pv { r += 1; }
            if cut_node { r += 1; }
            if is_capture { r -= 1; }

            // History-based adjustment
            if !is_capture {
                let hist = info.history.quiet_score(board, mv, prev_move);
                r -= (hist / 5000).clamp(-2, 2);
            }

            r = r.max(1).min(new_depth);

            // Reduced-depth search
            score = -negamax(board, info, -alpha - 1, -alpha, new_depth - r + 1, ply + 1, true);

            // doDeeper / doShallower
            if score > alpha && r > 1 {
                // Re-search at full depth with null window
                score = -negamax(board, info, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
            } else if score <= alpha && r < 1 {
                // doShallower: already searched deep enough
            }
        } else if !is_pv || moves_tried > 1 {
            // Non-PV null window search
            score = -negamax(board, info, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
        } else {
            score = alpha + 1; // Force full window search for first PV move
        }

        // Full window PV search
        if is_pv && (moves_tried == 1 || score > alpha) {
            score = -negamax(board, info, -beta, -alpha, new_depth, ply + 1, false);
        }

        board.unmake_move();

        if info.should_stop() {
            return best_score.max(0);
        }

        if score > best_score {
            best_score = score;
            best_move = mv;

            if score > alpha {
                alpha = score;

                if score >= beta {
                    // Fail high: update history, killers, counter
                    if !is_capture {
                        let bonus = (depth as i32 * depth as i32).min(400);

                        // Update killer
                        if ply_u < 128 {
                            if info.history.killers[ply_u][0] != mv {
                                info.history.killers[ply_u][1] = info.history.killers[ply_u][0];
                                info.history.killers[ply_u][0] = mv;
                            }
                        }

                        // Update counter-move
                        if prev_move != NO_MOVE {
                            let prev_to = move_to(prev_move);
                            let prev_piece = board.piece_at(prev_to);
                            if prev_piece != NO_PIECE && (prev_piece as usize) < 12 {
                                info.history.counter[prev_piece as usize][prev_to as usize] = mv;
                            }
                        }

                        // Update main history
                        let color = board.side_to_move;
                        History::update_history(
                            &mut info.history.main[color as usize][from as usize][to as usize],
                            bonus,
                        );

                        // Penalize other tried quiets
                        for &q in &quiets_tried {
                            let qf = move_from(q);
                            let qt = move_to(q);
                            History::update_history(
                                &mut info.history.main[color as usize][qf as usize][qt as usize],
                                -bonus,
                            );
                        }

                        // Update continuation history
                        if prev_move != NO_MOVE {
                            let prev_to = move_to(prev_move);
                            let prev_piece = board.piece_at(prev_to);
                            let piece = board.piece_at(to); // after unmake, piece is back at from... need to reconsider
                            let our_piece = board.piece_at(from);
                            if prev_piece != NO_PIECE && (prev_piece as usize) < 12
                                && our_piece != NO_PIECE && (our_piece as usize) < 12
                            {
                                History::update_history(
                                    &mut info.history.cont_hist[prev_piece as usize][prev_to as usize][our_piece as usize][to as usize],
                                    bonus,
                                );
                            }
                        }
                    } else {
                        // Capture history bonus
                        let piece = board.piece_at(from);
                        let victim = if flags == FLAG_EN_PASSANT {
                            PAWN
                        } else {
                            board.piece_type_at(to)
                        };
                        if piece != NO_PIECE && (piece as usize) < 12 && (victim as usize) < 6 {
                            let bonus = (depth as i32 * depth as i32).min(400);
                            History::update_history(
                                &mut info.history.capture[piece as usize][to as usize][victim as usize],
                                bonus,
                            );
                        }
                    }

                    break; // beta cutoff
                }
            }
        }

        if !is_capture && !is_promo {
            quiets_tried.push(mv);
        }
    }

    // Checkmate / stalemate
    if moves_tried == 0 {
        if in_check {
            return -MATE_SCORE + ply;
        } else {
            return 0;
        }
    }

    // Store in TT
    let tt_flag = if best_score >= beta {
        TT_FLAG_LOWER
    } else if best_score > alpha - 1 && is_pv { // improved alpha
        TT_FLAG_EXACT
    } else {
        TT_FLAG_UPPER
    };

    info.tt.store(
        board.hash,
        best_move,
        tt_flag,
        static_eval,
        score_to_tt(best_score, ply),
        depth,
    );

    best_score
}

/// Quiescence search.
fn quiescence(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    beta: i32,
    ply: i32,
) -> i32 {
    if info.should_stop() {
        return 0;
    }

    info.nodes += 1;

    if ply as usize >= MAX_PLY - 1 {
        return evaluate(board);
    }

    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    let in_check = board.in_check();
    let static_eval = if in_check { -INFINITY } else { evaluate(board) };

    if !in_check {
        if static_eval >= beta {
            return static_eval;
        }
        if static_eval > alpha {
            alpha = static_eval;
        }

        // Delta pruning
        if static_eval + 240 + 900 <= alpha { // QS delta + queen value
            return alpha;
        }
    }

    let mut best_score = if in_check { -INFINITY } else { static_eval };
    let mut picker = QMovePicker::new(board);

    loop {
        let mv = picker.next(board, in_check);
        if mv == NO_MOVE { break; }

        // SEE pruning in QS
        if !in_check && !see_ge(board, mv, 0) {
            continue;
        }

        // Delta pruning per-move
        if !in_check && !is_promotion(mv) {
            let to = move_to(mv);
            let captured_pt = board.piece_type_at(to);
            if captured_pt != NO_PIECE_TYPE {
                if static_eval + see_value(captured_pt) + 240 <= alpha {
                    continue;
                }
            }
        }

        board.make_move(mv);
        let score = -quiescence(board, info, -beta, -alpha, ply + 1);
        board.unmake_move();

        if score > best_score {
            best_score = score;
            if score > alpha {
                alpha = score;
                if score >= beta {
                    break;
                }
            }
        }
    }

    // In check with no moves = checkmate
    if in_check && best_score == -INFINITY {
        return -MATE_SCORE + ply;
    }

    best_score
}

/// Run bench: fixed-depth search on standard positions, return total nodes.
pub fn bench(depth: i32) -> u64 {
    let positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "2r3k1/pp3ppp/2n1b3/3pP3/3P4/2NB4/PP3PPP/R4RK1 w - - 0 1",
    ];

    let mut info = SearchInfo::new(16);
    let mut total_nodes = 0u64;

    let limits = SearchLimits {
        depth,
        infinite: true,
        ..SearchLimits::new()
    };

    for fen in &positions {
        let mut board = Board::from_fen(fen);
        info.nodes = 0;
        info.history.clear();
        info.tt.new_search();

        let mv = search(&mut board, &mut info, &limits);
        total_nodes += info.nodes;
    }

    total_nodes
}
