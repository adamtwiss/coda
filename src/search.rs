/// Main search: negamax with alpha-beta, iterative deepening, PVS, aspiration windows.
/// All pruning parameters ported from GoChess (tuned values).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::bitboard::*;
use crate::board::Board;
use crate::eval::{evaluate, evaluate_nnue, see_value};
use crate::nnue::{NNUENet, NNUEAccumulator, DirtyPiece};
use crate::movegen::generate_legal_moves;
use crate::movepicker::*;
use crate::see::see_ge;
use crate::tt::*;
use crate::types::*;

const MAX_PLY: usize = 128;
const INFINITY: i32 = 32000;

// Correction history constants
const CORR_HIST_SIZE: usize = 16384;
const CORR_HIST_GRAIN: i32 = 256;
const CORR_HIST_MAX: i32 = 128;
const CORR_HIST_LIMIT: i32 = 32000;

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
    prev_moves: [Move; MAX_PLY],
    static_evals: [i32; MAX_PLY],
    /// Pawn correction history: [color][pawn_hash % size]
    pawn_corr: Box<[[i32; CORR_HIST_SIZE]; 2]>,
    /// Continuation correction history: [piece][to_square]
    cont_corr: Box<[[i32; 64]; 12]>,
    pub nnue_net: Option<std::sync::Arc<crate::nnue::NNUENet>>,
    pub nnue_acc: Option<crate::nnue::NNUEAccumulator>,
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
            static_evals: [0; MAX_PLY],
            pawn_corr: Box::new([[0; CORR_HIST_SIZE]; 2]),
            cont_corr: Box::new([[0; 64]; 12]),
            nnue_net: None,
            nnue_acc: None,
        }
    }

    /// Load an NNUE network.
    pub fn load_nnue(&mut self, path: &str) -> Result<(), String> {
        let net = crate::nnue::NNUENet::load(path)?;
        let acc = crate::nnue::NNUEAccumulator::new(net.hidden_size);
        self.nnue_net = Some(std::sync::Arc::new(net));
        self.nnue_acc = Some(acc);
        Ok(())
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

    pub fn clear_correction_history(&mut self) {
        self.pawn_corr = Box::new([[0; CORR_HIST_SIZE]; 2]);
        self.cont_corr = Box::new([[0; 64]; 12]);
    }

    /// Evaluate using NNUE if loaded, otherwise classical PeSTO.
    fn eval(&mut self, board: &Board) -> i32 {
        if let (Some(net), Some(acc)) = (&self.nnue_net, &mut self.nnue_acc) {
            evaluate_nnue(board, net, acc)
        } else {
            evaluate(board)
        }
    }
}

/// Build a DirtyPiece for lazy NNUE accumulator update.
/// `us`/`them` are the sides BEFORE the move.
fn build_dirty_piece(
    mv: Move,
    us: u8,
    them: u8,
    moved_pt: u8,
    captured_pt: u8,
) -> DirtyPiece {
    let from = move_from(mv);
    let to = move_to(mv);
    let flags = move_flags(mv);

    // King moves: need full recompute (bucket may change)
    if moved_pt == KING {
        return DirtyPiece::recompute();
    }

    let mut changes: [(bool, u8, u8, u8); 5] = [(false, 0, 0, 0); 5];
    let mut n = 0;

    // Remove moved piece from origin
    changes[n] = (false, us, moved_pt, from); n += 1;

    // Remove captured piece
    if flags == FLAG_EN_PASSANT {
        let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
        changes[n] = (false, them, PAWN, cap_sq); n += 1;
    } else if captured_pt != NO_PIECE_TYPE {
        changes[n] = (false, them, captured_pt, to); n += 1;
    }

    // Add piece at destination (possibly promoted)
    let placed_pt = if is_promotion(mv) { promotion_piece_type(mv) } else { moved_pt };
    changes[n] = (true, us, placed_pt, to); n += 1;

    // Castling: also move the rook
    if flags == FLAG_CASTLE {
        let (rook_from, rook_to) = if to > from {
            if us == WHITE { (7u8, 5u8) } else { (63u8, 61u8) }
        } else {
            if us == WHITE { (0u8, 3u8) } else { (56u8, 59u8) }
        };
        changes[n] = (false, us, ROOK, rook_from); n += 1;
        changes[n] = (true, us, ROOK, rook_to); n += 1;
    }

    let mut d = DirtyPiece::recompute();
    d.kind = 1;
    d.n_changes = n as u8;
    d.changes = changes;
    d
}

/// Apply correction history to raw static eval.
fn corrected_eval(info: &SearchInfo, board: &Board, raw_eval: i32) -> i32 {
    let stm = board.side_to_move as usize;

    // Pawn correction
    let pawn_idx = (board.pawn_hash as usize) % CORR_HIST_SIZE;
    let pawn_corr = info.pawn_corr[stm][pawn_idx] as i64;

    // Continuation correction (from opponent's last move)
    let cont_corr = if !board.undo_stack.is_empty() {
        let last = &board.undo_stack[board.undo_stack.len() - 1];
        if last.mv != NO_MOVE {
            let to = move_to(last.mv);
            let pt = board.piece_type_at(to);
            if pt < 6 {
                let piece = make_piece(flip_color(board.side_to_move), pt);
                if (piece as usize) < 12 {
                    info.cont_corr[piece as usize][to as usize] as i64
                } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 };

    // Weighted blend: pawn 512/1024, cont 104/1024 (simplified from GoChess)
    let total_corr = (pawn_corr * 512 + cont_corr * 104) / 1024;
    let adjusted = raw_eval + (total_corr as i32) / CORR_HIST_GRAIN;
    adjusted.clamp(-MATE_SCORE + 100, MATE_SCORE - 100)
}

/// Update correction history entry with gravity.
fn update_corr_entry(entry: &mut i32, err: i32, weight: i32) {
    let err_clamped = err.clamp(-CORR_HIST_MAX, CORR_HIST_MAX);
    let new_val = (*entry * (CORR_HIST_GRAIN - weight) + err_clamped * CORR_HIST_GRAIN * weight) / CORR_HIST_GRAIN;
    *entry = new_val.clamp(-CORR_HIST_LIMIT, CORR_HIST_LIMIT);
}

/// Update all correction history tables.
fn update_correction_history(info: &mut SearchInfo, board: &Board, search_score: i32, raw_eval: i32, depth: i32) {
    if depth < 3 { return; }

    let err = search_score - raw_eval;
    let weight = (depth + 1).min(16);
    let stm = board.side_to_move as usize;

    // Pawn correction
    let pawn_idx = (board.pawn_hash as usize) % CORR_HIST_SIZE;
    update_corr_entry(&mut info.pawn_corr[stm][pawn_idx], err, weight);

    // Continuation correction
    if !board.undo_stack.is_empty() {
        let last = &board.undo_stack[board.undo_stack.len() - 1];
        if last.mv != NO_MOVE {
            let to = move_to(last.mv);
            let pt = board.piece_type_at(to);
            if pt < 6 {
                let piece = make_piece(flip_color(board.side_to_move), pt);
                if (piece as usize) < 12 {
                    update_corr_entry(&mut info.cont_corr[piece as usize][to as usize], err, weight);
                }
            }
        }
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

    // Reset and initialize NNUE accumulator for root position
    if let Some(acc) = &mut info.nnue_acc {
        acc.reset();
    }
    // Materialize root accumulator (populates Finny table)
    if let (Some(net), Some(acc)) = (&info.nnue_net, &mut info.nnue_acc) {
        acc.materialize(net, board);
    }

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

    // Get a fallback move and keep the legal list for final validation
    let root_legal = generate_legal_moves(board);
    if root_legal.len > 0 {
        best_move = root_legal.moves[0];
    }

    let effective_max = info.max_depth.min(MAX_PLY as i32 / 2);
    for depth in 1..=effective_max {
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

        // Get best move from TT — validate it's a legal move for the root position
        let tt_entry = info.tt.probe(board.hash);
        if tt_entry.hit && tt_entry.best_move != NO_MOVE {
            let tt_from = move_from(tt_entry.best_move);
            let tt_to = move_to(tt_entry.best_move);
            // Find matching move in root legal list (ensures correct flags)
            for i in 0..root_legal.len {
                let m = root_legal.moves[i];
                if move_from(m) == tt_from && move_to(m) == tt_to {
                    best_move = m; // use the legal move (with correct flags)
                    break;
                }
            }
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
        return alpha; // return current bound, not 0 (which fakes a draw)
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
        return info.eval(board);
    }

    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    // Draw detection (not at root)
    if !is_root {
        if board.halfmove >= 100 {
            return 0;
        }
        // Insufficient material: KvK, KNvK, KBvK
        let occ = board.occupied();
        let pc = popcount(occ);
        if pc <= 3 {
            if pc == 2 { return 0; } // KvK
            // KN vs K or KB vs K
            if board.pieces[PAWN as usize] == 0
                && board.pieces[ROOK as usize] == 0
                && board.pieces[QUEEN as usize] == 0
            {
                return 0;
            }
        }
        // Repetition detection: compare current hash against previous positions.
        // undo_stack[i].hash = hash of position BEFORE move i was made.
        // Current position has board.hash. The position 2 half-moves ago
        // (same side to move) is at undo_stack[len-2].hash (hash before the
        // opponent's last move, which equals the position after our last move).
        // Actually: undo[len-1].hash = position before opponent moved = our position.
        // Wait — no. undo[len-1] was pushed by the OPPONENT's make_move call.
        // undo[len-1].hash was the position when the opponent was about to move.
        // That's the position after OUR last move = same side as current (us).
        // But that's only 1 ply back, not 2.
        //
        // Simple approach: just check every entry. The hash includes the side-to-move
        // key, so only positions with the same side to move will match.
        let hash = board.hash;
        let stack_len = board.undo_stack.len();
        let check_back = (board.halfmove as usize).min(stack_len);
        for i in 0..check_back {
            if board.undo_stack[stack_len - 1 - i].hash == hash {
                return 0;
            }
        }
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

    // Check extension (capped to prevent explosion at extreme depths)
    if in_check && ply < MAX_PLY as i32 - 10 {
        depth += 1;
    }

    // Static eval with correction history
    let raw_eval;
    let static_eval;
    if in_check {
        raw_eval = -INFINITY;
        static_eval = -INFINITY;
    } else {
        raw_eval = if tt_entry.hit && tt_entry.static_eval != 0 {
            tt_entry.static_eval
        } else {
            info.eval(board)
        };
        static_eval = corrected_eval(info, board, raw_eval);
    }

    // Store static eval for improving detection
    if ply_u < MAX_PLY {
        info.static_evals[ply_u] = static_eval;
    }

    // Position is "improving" if our eval is better than 2 plies ago.
    // When ply-2 was in check, we stored -INFINITY, so improving=true
    // (conservative: don't reduce extra when uncertain).
    let improving = !in_check && ply >= 2 && (
        ply_u >= MAX_PLY || static_eval > info.static_evals[ply_u - 2]
    );

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
            if let Some(acc) = &mut info.nnue_acc { acc.push(DirtyPiece::recompute()); }
            let null_score = -negamax(board, info, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
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
        let mut pc_picker = QMovePicker::new(board, false);
        let pc_in_check = in_check;

        loop {
            let mv = pc_picker.next(board, pc_in_check);
            if mv == NO_MOVE { break; }

            if !see_ge(board, mv, probcut_beta - static_eval) { continue; }

            let pc_moved_pt = board.piece_type_at(move_from(mv));
            let pc_captured_pt = if move_flags(mv) == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(move_to(mv)) };
            let pc_dirty = build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), pc_moved_pt, pc_captured_pt);

            if let Some(acc) = &mut info.nnue_acc { acc.push(pc_dirty); }
            if !board.make_move(mv) {
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                continue;
            }
            let mut score = -quiescence(board, info, -probcut_beta, -probcut_beta + 1, ply + 1);
            if score >= probcut_beta {
                score = -negamax(board, info, -probcut_beta, -probcut_beta + 1, depth - 4, ply + 1, !cut_node);
            }
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }

            if score >= probcut_beta {
                return score;
            }
        }
    }

    // IIR: Internal Iterative Reduction
    if tt_move == NO_MOVE && depth >= 4 {
        depth -= 1;
    }

    let original_alpha = alpha;

    let safe_ply = ply_u.min(MAX_PLY - 1);
    let prev_move = if ply > 0 { info.prev_moves[safe_ply - 1] } else { NO_MOVE };
    let mut picker = MovePicker::new(board, tt_move, safe_ply, &info.history, prev_move);

    let mut best_score = -INFINITY;
    let mut best_move = NO_MOVE;
    let mut moves_tried = 0;
    let mut quiets_tried = [NO_MOVE; 64];
    let mut n_quiets_tried = 0usize;

    loop {
        let mv = picker.next(board, &info.history, prev_move);
        if mv == NO_MOVE { break; }

        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);
        let is_capture = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
        let is_promo = is_promotion(mv);

        // Pruning for non-root, non-PV, late moves
        if !is_root && best_score > -TB_WIN && !in_check {
            // Late Move Pruning (LMP) — more aggressive when not improving
            let lmp_threshold = if improving {
                3 + depth as usize * depth as usize
            } else {
                (3 + depth as usize * depth as usize) / 2
            };
            if !is_capture && !is_promo && moves_tried > lmp_threshold {
                continue;
            }

            // History pruning for quiets
            if !is_capture && !is_promo && depth <= 8 {
                let hist = info.history.quiet_score(board, mv, prev_move);
                if hist < -1500 * depth as i32 {
                    continue;
                }
            }

            // Futility pruning (uses estimated LMR depth)
            if !is_capture && !is_promo && depth <= 8 {
                let est_r = lmr_reduction(depth, moves_tried as i32);
                let lmr_depth = (depth - est_r).max(1);
                let futility_margin = 60 + lmr_depth * 60;
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
        if safe_ply < MAX_PLY {
            info.prev_moves[safe_ply] = mv;
        }

        // Build NNUE dirty piece info BEFORE make_move
        let moved_pt = board.piece_type_at(from);
        let captured_pt = if flags == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(to) };
        let us = board.side_to_move;
        let them = flip_color(us);
        let dirty = build_dirty_piece(mv, us, them, moved_pt, captured_pt);

        // Push NNUE accumulator with lazy update
        if let Some(acc) = &mut info.nnue_acc { acc.push(dirty); }

        if !board.make_move(mv) {
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            continue;
        }

        moves_tried += 1;

        // Recapture extension: extend when capturing on the same square as the previous capture
        let mut extension = 0;
        if is_capture && board.undo_stack.len() >= 2 {
            let prev = &board.undo_stack[board.undo_stack.len() - 2];
            if prev.captured != NO_PIECE_TYPE && to == move_to(prev.mv) {
                extension = 1;
            }
        }

        let mut score;
        let mut new_depth = depth - 1 + extension;

        // LMR
        if depth >= 3 && moves_tried > 1 + if is_pv { 1 } else { 0 } {
            let mut r = lmr_reduction(depth, moves_tried as i32);

            // Adjustments
            if !is_pv { r += 1; }
            if cut_node { r += 1; }
            if is_capture { r -= 1; }

            // History-based adjustment (use `us` not board.side_to_move — board is post-make)
            if !is_capture {
                let from = move_from(mv);
                let to = move_to(mv);
                let hist = info.history.main[us as usize][from as usize][to as usize];
                r -= (hist / 5000).clamp(-2, 2);
            }

            r = r.max(1).min(new_depth);

            // Reduced-depth search
            score = -negamax(board, info, -alpha - 1, -alpha, new_depth - r, ply + 1, true);

            // doDeeper / doShallower
            if score > alpha && r > 0 {
                // LMR failed high — check if we need deeper or shallower re-search
                let mut adj = 0;
                if score > best_score + 60 + 10 * r {
                    adj = 1; // score greatly exceeds best → LMR too aggressive → deeper
                } else if score < best_score + new_depth {
                    adj = -1; // score barely above alpha → LMR about right → shallower
                }
                let re_depth = (new_depth + adj).max(1);
                score = -negamax(board, info, -alpha - 1, -alpha, re_depth, ply + 1, !cut_node);
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
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }

        if info.should_stop() {
            return best_score;
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
                        for &q in &quiets_tried[..n_quiets_tried] {
                            let qf = move_from(q);
                            let qt = move_to(q);
                            History::update_history(
                                &mut info.history.main[color as usize][qf as usize][qt as usize],
                                -bonus,
                            );
                        }

                        // Update continuation history
                        // Note: board is post-unmake, piece is at `from`
                        if prev_move != NO_MOVE {
                            let prev_to = move_to(prev_move);
                            let prev_piece = board.piece_at(prev_to);
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
            if n_quiets_tried < 64 {
                quiets_tried[n_quiets_tried] = mv;
                n_quiets_tried += 1;
            }
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

    // Update correction history (only for non-mate, non-check positions with sufficient depth)
    if !in_check && !is_mate_score(best_score) && raw_eval != -INFINITY {
        update_correction_history(info, board, best_score, raw_eval, depth);
    }

    // Store in TT
    let tt_flag = if best_score >= beta {
        TT_FLAG_LOWER
    } else if is_pv && best_score > original_alpha {
        TT_FLAG_EXACT
    } else {
        TT_FLAG_UPPER
    };

    info.tt.store(
        board.hash,
        best_move,
        tt_flag,
        raw_eval,  // store RAW eval, not corrected — avoids double correction on TT hit
        score_to_tt(best_score, ply),
        depth,
    );

    // Fail-high score blending: dampen inflated cutoff scores at non-PV nodes.
    // Deeper cutoffs are more trustworthy, so weight raw score by depth.
    if best_score >= beta && !is_pv && depth >= 3
        && !is_mate_score(best_score)
    {
        return (best_score * depth + beta) / (depth + 1);
    }

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
        return alpha;
    }

    info.nodes += 1;

    if ply as usize >= MAX_PLY - 1 {
        return info.eval(board);
    }

    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    let in_check = board.in_check();
    let static_eval = if in_check { -INFINITY } else { info.eval(board) };

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
    let mut picker = QMovePicker::new(board, in_check);

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

        // Build lazy NNUE update
        let qs_moved_pt = board.piece_type_at(move_from(mv));
        let qs_captured_pt = if move_flags(mv) == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(move_to(mv)) };
        let qs_dirty = build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), qs_moved_pt, qs_captured_pt);

        if let Some(acc) = &mut info.nnue_acc { acc.push(qs_dirty); }
        if !board.make_move(mv) {
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            continue;
        }
        let score = -quiescence(board, info, -beta, -alpha, ply + 1);
        board.unmake_move();
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }

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
