/// Main search: negamax with alpha-beta, iterative deepening, PVS, aspiration windows.
/// All pruning parameters ported from GoChess (tuned values).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::bitboard::*;
use crate::board::Board;
use crate::eval::{evaluate, evaluate_nnue, see_value};
use crate::nnue::DirtyPiece;
use crate::movegen::generate_legal_moves;
use crate::movepicker::*;
use crate::see::see_ge;
use crate::tt::*;
use crate::types::*;

const MAX_PLY: usize = 128;
const INFINITY: i32 = 32000;
const CONTEMPT: i32 = -10; // prefer playing on over drawing

// Pawn history table size
const PAWN_HIST_SIZE: usize = 512;

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

/// Pruning counters for diagnostics.
#[derive(Default)]
pub struct PruneStats {
    pub tt_cutoffs: u64,
    pub tt_near_miss: u64,
    pub nmp_attempts: u64,
    pub nmp_cutoffs: u64,
    pub rfp_cutoffs: u64,
    pub razor_cutoffs: u64,
    pub lmp_prunes: u64,
    pub futility_prunes: u64,
    pub history_prunes: u64,
    pub see_prunes: u64,
    pub probcut_cutoffs: u64,
    pub lmr_searches: u64,
    pub recapture_ext: u64,
    pub qnodes: u64,
    pub beta_cutoffs: u64,
    pub first_move_cutoffs: u64,
}

/// Search state for one thread.
pub struct SearchInfo {
    pub nodes: u64,
    pub stats: PruneStats,
    pub tt: TT,
    pub history: History,
    pub stop: AtomicBool,
    pub start_time: Instant,
    pub time_limit: u64,  // ms
    pub max_depth: i32,
    pub max_nodes: u64,
    pub move_overhead: u64, // ms
    // Dynamic time management state
    tm_prev_best: Move,
    tm_prev_score: i32,
    tm_best_stable: i32,
    tm_has_data: bool,
    soft_limit: u64,  // ms — can be extended/shortened dynamically
    hard_limit: u64,  // ms — absolute maximum
    pub sel_depth: i32,
    prev_moves: [Move; MAX_PLY],
    static_evals: [i32; MAX_PLY],
    /// Pawn history: [pawn_hash % PAWN_HIST_SIZE][piece][to_square]
    pawn_hist: Box<[[[i16; 64]; 12]; PAWN_HIST_SIZE]>,
    /// Pawn correction history: [stm][pawn_hash % size]
    pawn_corr: Box<[[i32; CORR_HIST_SIZE]; 2]>,
    /// Non-pawn correction history: [stm][color][nonpawn_hash % size]
    np_corr: Box<[[[i32; CORR_HIST_SIZE]; 2]; 2]>,
    /// Continuation correction history: [piece][to_square]
    cont_corr: Box<[[i32; 64]; 12]>,
    pub nnue_net: Option<std::sync::Arc<crate::nnue::NNUENet>>,
    pub nnue_acc: Option<crate::nnue::NNUEAccumulator>,
}

impl SearchInfo {
    pub fn new(tt_mb: usize) -> Self {
        SearchInfo {
            nodes: 0,
            stats: PruneStats::default(),
            tt: TT::new(tt_mb),
            history: History::new(),
            stop: AtomicBool::new(false),
            start_time: Instant::now(),
            time_limit: 0,
            max_depth: 100,
            max_nodes: 0,
            move_overhead: 100,
            tm_prev_best: NO_MOVE,
            tm_prev_score: 0,
            tm_best_stable: 0,
            tm_has_data: false,
            soft_limit: 0,
            hard_limit: 0,
            sel_depth: 0,
            prev_moves: [NO_MOVE; MAX_PLY],
            static_evals: [0; MAX_PLY],
            pawn_hist: Box::new([[[0i16; 64]; 12]; PAWN_HIST_SIZE]),
            pawn_corr: Box::new([[0; CORR_HIST_SIZE]; 2]),
            np_corr: Box::new([[[0; CORR_HIST_SIZE]; 2]; 2]),
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
        self.np_corr = Box::new([[[0; CORR_HIST_SIZE]; 2]; 2]);
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

    // Non-pawn corrections (per color). Derive non-pawn key from hash ^ pawn_hash.
    let np_key = board.hash ^ board.pawn_hash;
    let np_idx = (np_key as usize) % CORR_HIST_SIZE;
    let white_np_corr = info.np_corr[stm][WHITE as usize][np_idx] as i64;
    // Use a different index for black NP (rotate the key)
    let np_idx_b = (np_key.rotate_right(32) as usize) % CORR_HIST_SIZE;
    let black_np_corr = info.np_corr[stm][BLACK as usize][np_idx_b] as i64;

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

    // Weighted blend: pawn 512, whiteNP 204, blackNP 204, cont 104 = 1024
    let total_corr = (pawn_corr * 512 + white_np_corr * 204 + black_np_corr * 204 + cont_corr * 104) / 1024;
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

    // Non-pawn corrections (per color)
    let np_key = board.hash ^ board.pawn_hash;
    let np_idx = (np_key as usize) % CORR_HIST_SIZE;
    update_corr_entry(&mut info.np_corr[stm][WHITE as usize][np_idx], err, weight);
    let np_idx_b = (np_key.rotate_right(32) as usize) % CORR_HIST_SIZE;
    update_corr_entry(&mut info.np_corr[stm][BLACK as usize][np_idx_b], err, weight);

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

/// LMR reduction tables (quiet and capture).
static mut LMR_TABLE: [[i32; 64]; 64] = [[0; 64]; 64];
static mut LMR_TABLE_CAP: [[i32; 64]; 64] = [[0; 64]; 64];

pub fn init_lmr() {
    for depth in 1..64 {
        for moves in 1..64 {
            unsafe {
                // Quiet table: C=1.30
                LMR_TABLE[depth][moves] = ((depth as f64).ln() * (moves as f64).ln() / 1.30) as i32;
                // Capture table: C=1.80 (less reduction for captures)
                if depth >= 3 && moves >= 3 {
                    let r = ((depth as f64).ln() * (moves as f64).ln() / 1.80) as i32;
                    LMR_TABLE_CAP[depth][moves] = r.min((depth - 2) as i32);
                }
            }
        }
    }
}

fn lmr_cap_reduction(depth: i32, moves: i32) -> i32 {
    let d = (depth as usize).min(63);
    let m = (moves as usize).min(63);
    unsafe { LMR_TABLE_CAP[d][m] }
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
        // Subtract move overhead (communication latency)
        let overhead = info.move_overhead;
        let time_left = our_time.saturating_sub(overhead).max(1);

        let moves_left = if limits.movestogo > 0 { limits.movestogo as u64 } else { 25 };

        // Soft allocation: time/movesLeft + 80% of increment
        let mut soft = time_left / moves_left + our_inc * 4 / 5;

        // Cap at half remaining time
        let max_alloc = time_left / 2;
        if soft > max_alloc { soft = max_alloc; }

        // Emergency: below 1 second, be very conservative
        if time_left < 1000 {
            let mut emergency = time_left / 10;
            if our_inc > 0 && our_inc < emergency { emergency = our_inc; }
            if emergency < 10 { emergency = 10; }
            if soft > emergency { soft = emergency; }
        }

        // Floor at 10ms
        if soft < 10 { soft = 10; }

        // Hard limit: 3x soft for sudden death, 2x for tournament
        let hard = if limits.movestogo > 0 {
            (soft * 2).min(time_left / 2)
        } else {
            (soft * 3).min(time_left * 3 / 4)
        };

        info.soft_limit = soft;
        info.hard_limit = hard.max(soft);
        info.time_limit = hard; // search uses hard as absolute limit
        info.tm_has_data = false;
        info.tm_best_stable = 0;
    } else if !limits.infinite {
        info.time_limit = 0;
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
        if depth >= 4 {
            let mut delta = 15i32;
            let mut alpha = (prev_score - delta).max(-INFINITY);
            let mut beta = (prev_score + delta).min(INFINITY);
            #[allow(unused_assignments)]
            let mut asp_result = prev_score;

            loop {
                let result = negamax(board, info, alpha, beta, depth, 0, false);

                if info.should_stop() {
                    asp_result = result;
                    break;
                }

                if result <= alpha {
                    // Fail low: contract beta aggressively toward alpha, widen alpha
                    beta = (3 * alpha + 5 * beta) / 8;
                    alpha = (result - delta).max(-INFINITY);
                } else if result >= beta {
                    // Fail high: contract alpha toward beta, widen beta
                    alpha = (5 * alpha + 3 * beta) / 8;
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

        // Extract PV from TT
        let mut pv_str = String::new();
        {
            let mut pv_board = board.clone();
            let mut pv_moves = 0;
            // First move is always the best_move
            pv_str.push_str(&move_to_uci(best_move));
            pv_board.make_move(best_move);
            pv_moves += 1;
            // Follow TT chain for remaining PV moves
            while pv_moves < depth as usize + 5 {
                let pv_tt = info.tt.probe(pv_board.hash);
                if !pv_tt.hit || pv_tt.best_move == NO_MOVE { break; }
                let pv_from = move_from(pv_tt.best_move);
                let pv_to = move_to(pv_tt.best_move);
                // Validate move exists in legal move list
                let pv_legal = generate_legal_moves(&pv_board);
                let mut found = NO_MOVE;
                for i in 0..pv_legal.len {
                    let m = pv_legal.moves[i];
                    if move_from(m) == pv_from && move_to(m) == pv_to {
                        found = m;
                        break;
                    }
                }
                if found == NO_MOVE { break; }
                pv_str.push(' ');
                pv_str.push_str(&move_to_uci(found));
                pv_board.make_move(found);
                pv_moves += 1;
            }
        }

        println!(
            "info depth {} seldepth {} {} nodes {} nps {} time {} hashfull {} pv {}",
            depth, info.sel_depth, score_str, info.nodes, nps, elapsed,
            info.tt.hashfull(), pv_str
        );

        // Dynamic time management: adjust soft limit based on stability
        if info.soft_limit > 0 && depth >= 4 && !info.should_stop() {
            // Track best-move stability
            if info.tm_has_data {
                let bm_from = move_from(best_move);
                let bm_to = move_to(best_move);
                let prev_from = move_from(info.tm_prev_best);
                let prev_to = move_to(info.tm_prev_best);
                if bm_from == prev_from && bm_to == prev_to {
                    info.tm_best_stable += 1;
                } else {
                    info.tm_best_stable = 0;
                }
            }

            // Score delta
            let score_delta = if info.tm_has_data && !is_mate_score(prev_score) && !is_mate_score(info.tm_prev_score) {
                (prev_score - info.tm_prev_score).abs()
            } else {
                0
            };

            info.tm_prev_best = best_move;
            info.tm_prev_score = prev_score;
            info.tm_has_data = true;

            // Scale factor for soft limit
            let mut scale = 1.0f64;

            // Stable best move → stop early
            if info.tm_best_stable >= 8 { scale *= 0.35; }
            else if info.tm_best_stable >= 5 { scale *= 0.5; }
            else if info.tm_best_stable >= 3 { scale *= 0.7; }
            else if info.tm_best_stable >= 1 { scale *= 0.85; }

            // Very stable + stable score → extra reduction
            if info.tm_best_stable >= 5 && score_delta <= 10 { scale *= 0.8; }

            // Unstable score → extend time
            if score_delta > 50 { scale *= 2.0; }
            else if score_delta > 25 { scale *= 1.5; }
            else if score_delta > 10 { scale *= 1.2; }

            // Check if we should stop at the soft limit
            let adjusted_soft = (info.soft_limit as f64 * scale) as u64;
            let adjusted_soft = adjusted_soft.min(info.hard_limit);
            if elapsed >= adjusted_soft {
                break;
            }
        }
    }

    best_move
}

/// Negamax alpha-beta search.
fn negamax(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    beta: i32,
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
        return info.eval(board);
    }

    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    // Draw detection (not at root)
    if !is_root {
        if board.halfmove >= 100 {
            return CONTEMPT;
        }
        // Insufficient material: KvK, KNvK, KBvK
        let occ = board.occupied();
        let pc = popcount(occ);
        if pc <= 3 {
            if pc == 2 { return CONTEMPT; } // KvK
            // KN vs K or KB vs K
            if board.pieces[PAWN as usize] == 0
                && board.pieces[ROOK as usize] == 0
                && board.pieces[QUEEN as usize] == 0
            {
                return CONTEMPT;
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
                return CONTEMPT;
            }
        }
    }

    // TT probe
    let tt_entry = info.tt.probe(board.hash);
    let tt_move = if tt_entry.hit { tt_entry.best_move } else { NO_MOVE };
    let tt_score = if tt_entry.hit { score_from_tt(tt_entry.score, ply) } else { 0 };

    // TT cutoff (non-PV only)
    if !is_pv && tt_entry.hit && tt_entry.depth >= depth {
        let mut tt_cutoff = false;
        let mut tt_return_score = tt_score;
        match tt_entry.flag {
            TT_FLAG_EXACT => { tt_cutoff = true; }
            TT_FLAG_LOWER => {
                if tt_score >= beta {
                    tt_cutoff = true;
                    // TT score dampening: blend toward beta to prevent inflation
                    if !is_mate_score(tt_score) {
                        tt_return_score = (3 * tt_score + beta) / 4;
                    }
                }
            }
            TT_FLAG_UPPER => {
                if tt_score <= alpha { tt_cutoff = true; }
            }
            _ => {}
        }
        if tt_cutoff {
            info.stats.tt_cutoffs += 1;
            // History bonus for TT cutoff move (reinforces move ordering)
            if tt_move != NO_MOVE {
                let bonus = (depth as i32 * depth as i32).min(1200);
                let tt_from = move_from(tt_move);
                let tt_to = move_to(tt_move);
                let tt_target = board.piece_type_at(tt_to);
                let tt_is_cap = tt_target != NO_PIECE_TYPE || move_flags(tt_move) == FLAG_EN_PASSANT;
                if !tt_is_cap {
                    // Quiet TT cutoff: update main history
                    History::update_history(
                        &mut info.history.main[board.side_to_move as usize][tt_from as usize][tt_to as usize],
                        bonus,
                    );
                } else {
                    // Capture TT cutoff: update capture history
                    let piece = board.piece_at(tt_from);
                    let victim = if move_flags(tt_move) == FLAG_EN_PASSANT { PAWN } else { tt_target };
                    if piece != NO_PIECE && (piece as usize) < 12 && (victim as usize) < 6 {
                        History::update_history(
                            &mut info.history.capture[piece as usize][tt_to as usize][victim as usize],
                            bonus,
                        );
                    }
                }
            }
            return tt_return_score;
        }
    }

    // TT near-miss: accept entries 1 ply short with a score margin (only when full-depth didn't cut)
    else if !is_pv && tt_entry.hit && tt_entry.depth >= depth - 1
        && !is_mate_score(tt_score)
    {
        let margin = 80;
        if tt_entry.flag == TT_FLAG_LOWER && tt_score - margin >= beta {
            info.stats.tt_near_miss += 1; return tt_score - margin;
        }
        if tt_entry.flag == TT_FLAG_UPPER && tt_score + margin <= alpha {
            info.stats.tt_near_miss += 1; return tt_score + margin;
        }
    }

    let in_check = board.in_check();

    // Check extensions disabled — proven harmful (-11.2 Elo SPRT in GoChess).
    // Only recapture extensions are beneficial cross-engine.

    // Static eval with correction history
    let raw_eval;
    let static_eval;
    if in_check {
        raw_eval = -INFINITY;
        static_eval = -INFINITY;
    } else {
        raw_eval = if tt_entry.hit && tt_entry.static_eval > -INFINITY + 100 {
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

    // Failing heuristic: detect significant position deterioration.
    // When eval has dropped well below 2-ply-ago eval, prune/reduce more aggressively.
    let failing = !in_check && ply >= 2 && ply_u < MAX_PLY
        && info.static_evals[ply_u - 2] > -INFINITY + 100
        && static_eval < info.static_evals[ply_u - 2] - (60 + 40 * depth as i32);

    // Eval instability: sharp eval swing from parent → something tactical happening
    let unstable = !in_check && ply >= 1 && ply_u > 0
        && info.static_evals[ply_u - 1] > -INFINITY + 100
        && static_eval > -INFINITY
        && (static_eval + info.static_evals[ply_u - 1]).abs() > 200;

    // IIR: Internal Iterative Reduction — reduce depth when no TT move exists.
    // Applied FIRST so all subsequent pruning decisions use the reduced depth.
    if tt_move == NO_MOVE && depth >= 6 && !in_check {
        depth -= 1;
    }

    // Razoring
    if !is_pv && !in_check && depth <= 2 && ply > 0 {
        let razor_margin = 400 + depth as i32 * 100;
        if static_eval + razor_margin <= alpha {
            let q_score = quiescence(board, info, alpha, beta, ply);
            if q_score <= alpha {
                info.stats.razor_cutoffs += 1; return q_score;
            }
        }
    }

    // Hindsight reduction: when both sides think the position is quiet
    // (parent eval + current eval both positive, sum > 200), reduce depth by 1.
    if !in_check && ply >= 1 && depth >= 3 && ply_u > 0
        && info.static_evals[ply_u - 1] > -INFINITY + 100
        && static_eval > -INFINITY
    {
        let eval_sum = info.static_evals[ply_u - 1] + static_eval;
        if eval_sum > 200 {
            depth -= 1;
        }
    }

    // Reverse Futility Pruning (RFP)
    // Reverse Futility Pruning
    if !is_pv && !in_check && depth <= 7 && ply > 0 {
        let rfp_margin = if improving { depth as i32 * 70 } else { depth as i32 * 100 };
        if static_eval - rfp_margin >= beta {
            info.stats.rfp_cutoffs += 1; return static_eval - rfp_margin;
        }
    }

    // Null Move Pruning
    if !is_pv && !in_check && depth >= 3 && ply > 0 && static_eval >= beta {
        // Ensure we have non-pawn material
        let us = board.side_to_move;
        let non_pawn = board.colors[us as usize] & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
        if non_pawn != 0 {
            let mut r = 3 + depth / 3 + ((static_eval - beta) / 200).min(3);
            // Reduce R when last move was a capture (position may be more volatile)
            if !board.undo_stack.is_empty() && board.undo_stack.last().unwrap().captured != NO_PIECE_TYPE {
                r -= 1;
            }
            // Clamp: ensure R >= 1 and search depth >= 1 (don't go to QS)
            let r = r.max(1).min(depth - 2);

            info.stats.nmp_attempts += 1;
            board.make_null_move();
            if let Some(acc) = &mut info.nnue_acc { acc.push(DirtyPiece::recompute()); }
            let null_score = -negamax(board, info, -beta, -beta + 1, depth - 1 - r, ply + 1, false);
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            board.unmake_null_move();

            if null_score >= beta {
                // NMP score dampening: blend toward beta
                let dampened = (null_score * 2 + beta) / 3;

                // Verification at high depth
                if depth >= 12 {
                    let v = negamax(board, info, beta - 1, beta, depth - 1 - r, ply + 1, false);
                    if v >= beta {
                        info.stats.nmp_cutoffs += 1; return dampened;
                    }
                } else {
                    info.stats.nmp_cutoffs += 1; return dampened;
                }
            }
        }
    }

    // ProbCut (with static eval pre-check gate)
    if !is_pv && !in_check && depth >= 5 {
        let probcut_beta = beta + 170;
        if static_eval + 85 >= probcut_beta {
        let mut pc_picker = QMovePicker::new(board, false);
        let pc_in_check = in_check;

        loop {
            let mv = pc_picker.next(board, pc_in_check);
            if mv == NO_MOVE { break; }

            if !see_ge(board, mv, 0) { continue; }

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
                info.stats.probcut_cutoffs += 1;
                return score;
            }
        }
        } // static eval pre-check gate
    }

    let original_alpha = alpha;

    let safe_ply = ply_u.min(MAX_PLY - 1);
    let prev_move = if ply > 0 { info.prev_moves[safe_ply - 1] } else { NO_MOVE };
    let tt_move_is_noisy = tt_move != NO_MOVE && {
        let tt_to = move_to(tt_move);
        board.piece_type_at(tt_to) != NO_PIECE_TYPE || is_promotion(tt_move)
    };
    let mut picker = MovePicker::new(board, tt_move, safe_ply, &info.history, prev_move);

    let mut best_score = -INFINITY;
    let mut best_move = NO_MOVE;
    let mut moves_tried = 0;
    let mut quiets_tried = [NO_MOVE; 64];
    let mut n_quiets_tried = 0usize;
    let mut captures_tried: [(u8, u8, u8); 32] = [(0, 0, 0); 32]; // (piece, to, victim)
    let mut n_captures_tried = 0usize;
    let mut alpha_raised_count = 0;

    loop {
        let ph_idx = (board.pawn_hash as usize) % PAWN_HIST_SIZE;
        let mv = picker.next(board, &info.history, prev_move, Some(&info.pawn_hist[ph_idx]));
        if mv == NO_MOVE { break; }

        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);
        let is_capture = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
        let is_promo = is_promotion(mv);

        // Check if this is a special move (TT, killer, counter) — exempt from pruning
        let safe = safe_ply.min(127);
        let is_tt_move = tt_move != NO_MOVE && move_from(mv) == move_from(tt_move) && move_to(mv) == move_to(tt_move);
        let is_killer = !is_capture && (mv == info.history.killers[safe][0] || mv == info.history.killers[safe][1]);
        let is_counter = if prev_move != NO_MOVE {
            let prev_to = move_to(prev_move);
            let prev_piece = board.piece_at(prev_to);
            prev_piece != NO_PIECE && (prev_piece as usize) < 12
                && mv == info.history.counter[prev_piece as usize][prev_to as usize]
        } else { false };
        let is_special = is_tt_move || is_killer || is_counter;

        // Pre-make pruning: only things that don't need gives-check info
        // History pruning (pre-make, doesn't need check detection)
        if !is_root && !is_pv && !is_special
            && best_score > -TB_WIN && !in_check
        {
            if !is_capture && !is_promo && depth <= 3 && !improving && !unstable {
                let ph_idx2 = (board.pawn_hash as usize) % PAWN_HIST_SIZE;
                let hist = info.history.quiet_score(board, mv, prev_move, Some(&info.pawn_hist[ph_idx2]));
                if hist < -1500 * depth as i32 {
                    info.stats.history_prunes += 1; continue;
                }
            }
        }

        // Pre-compute SEE results before make_move (SEE needs pre-make board state)
        // Only compute when the post-make pruning block will actually use them
        let do_see = !is_root && !is_pv && !is_special && !in_check
            && best_score > -TB_WIN && depth <= 8;
        let see_capture_ok = if do_see && is_capture && depth <= 6 {
            see_ge(board, mv, -(depth as i32) * 100)
        } else { true };
        let see_quiet_ok = if do_see && !is_capture {
            let see_quiet_threshold = -20 * depth as i32 * depth as i32
                - if unstable { 100 } else { 0 };
            see_ge(board, mv, see_quiet_threshold)
        } else { true };
        let see_bad_noisy = if do_see && is_capture && !is_promo && depth <= 4 {
            !see_ge(board, mv, 0)
        } else { false };

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

        // Post-make pruning: now we can detect gives-check
        let gives_check = board.in_check(); // opponent is now in check = our move gave check

        if !is_root && !is_pv && !is_special
            && best_score > -TB_WIN && !in_check && depth <= 8
        {
            // Bad noisy: prune losing captures when eval is far below alpha
            if is_capture && !is_promo && !gives_check && depth <= 4
                && static_eval > -INFINITY && static_eval + depth as i32 * 75 <= alpha
                && see_bad_noisy
            {
                board.unmake_move();
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                moves_tried -= 1;
                continue;
            }

            // LMP (exempt checks)
            let mut lmp_threshold = 3 + depth as usize * depth as usize;
            if improving && depth >= 3 { lmp_threshold += lmp_threshold / 2; }
            if failing { lmp_threshold = lmp_threshold * 2 / 3; }
            if !is_capture && !is_promo && !gives_check && moves_tried > lmp_threshold {
                board.unmake_move();
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                info.stats.lmp_prunes += 1;
                continue;
            }

            // Futility (exempt checks)
            if !is_capture && !is_promo && !gives_check {
                let est_r = lmr_reduction(depth, moves_tried as i32);
                let lmr_depth = (depth - est_r).max(1);
                let futility_margin = 60 + lmr_depth * 60;
                if static_eval + futility_margin <= alpha {
                    board.unmake_move();
                    if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                    info.stats.futility_prunes += 1;
                    continue;
                }
            }

            // SEE pruning (exempt checks) — uses pre-computed SEE results
            if !gives_check {
                if is_capture && depth <= 6 && !see_capture_ok {
                    board.unmake_move();
                    if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                    info.stats.see_prunes += 1;
                    continue;
                }
                if !is_capture && depth <= 8 && !see_quiet_ok {
                    board.unmake_move();
                    if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                    info.stats.see_prunes += 1;
                    continue;
                }
            }
        }

        // Recapture extension: extend when capturing on the same square as the previous capture
        let mut extension = 0;
        if is_capture && board.undo_stack.len() >= 2 {
            let prev = &board.undo_stack[board.undo_stack.len() - 2];
            if prev.captured != NO_PIECE_TYPE && to == move_to(prev.mv) {
                extension = 1;
                info.stats.recapture_ext += 1;
            }
        }

        let mut score;
        let mut new_depth = depth - 1 + extension;

        // Alpha-reduce: after alpha has been raised, reduce subsequent moves by 1 ply
        if alpha_raised_count > 0 {
            new_depth -= 1;
        }
        if new_depth < 0 { new_depth = 0; }

        // LMR (exempt promotions, killers, counter-moves, and check-giving moves)
        if depth >= 3 && !is_promo && !is_killer && !is_counter && !gives_check && moves_tried > 1 + if is_pv { 1 } else { 0 } {
            let mut r = if is_capture {
                // Capture LMR: separate table (C=1.80), non-PV only
                if !is_pv { lmr_cap_reduction(depth, moves_tried as i32) } else { 0 }
            } else {
                lmr_reduction(depth, moves_tried as i32)
            };

            // Quiet-only adjustments
            if !is_capture {
                if !is_pv { r += 1; }
                // Use GoChess's cut-node heuristic instead of explicit cut_node flag
                if !is_pv && moves_tried > 1 { r += 1; }

                // Reduce more when multiple moves have already raised alpha
                if alpha_raised_count > 1 {
                    r += alpha_raised_count / 2;
                }
            }

            // (killers are fully exempt from LMR via the condition above)

            // History-based adjustment (use `us` not board.side_to_move — board is post-make)
            if !is_capture {
                let from = move_from(mv);
                let to = move_to(mv);
                let mut hist = info.history.main[us as usize][from as usize][to as usize];
                // Add continuation history (1x weight for LMR adjustment)
                if prev_move != NO_MOVE {
                    let prev_to = move_to(prev_move);
                    let prev_piece = board.piece_at(prev_to);
                    let our_piece = make_piece(us, moved_pt);
                    if prev_piece != NO_PIECE && (prev_piece as usize) < 12
                        && (our_piece as usize) < 12
                    {
                        hist += info.history.cont_hist[prev_piece as usize][prev_to as usize][our_piece as usize][to as usize];
                    }
                }
                r -= hist / 5000;
            }

            // Reduce less when improving
            if improving { r -= 1; }

            // Reduce more when position is deteriorating
            if failing { r += 1; }

            // Reduce less in volatile positions
            if unstable { r -= 1; }

            // Reduce more when TT best move is a capture (quiet alternatives less likely)
            if !is_capture && tt_move_is_noisy {
                r += 1;
            }

            r = r.max(0).min(new_depth);

            info.stats.lmr_searches += 1;
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

        // Full window PV search (only if score is between alpha and beta — if already >= beta, skip)
        if is_pv && (moves_tried == 1 || (score > alpha && score < beta)) {
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
                alpha_raised_count += 1;

                if score >= beta {
                    info.stats.beta_cutoffs += 1;
                    if moves_tried == 1 { info.stats.first_move_cutoffs += 1; }
                    // Fail high: update history, killers, counter
                    if !is_capture {
                        let bonus = (depth as i32 * depth as i32).min(1200);

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

                        // Update pawn history
                        {
                            let ph_idx = (board.pawn_hash as usize) % PAWN_HIST_SIZE;
                            let our_piece = board.piece_at(from);
                            if our_piece != NO_PIECE && (our_piece as usize) < 12 {
                                let v = info.pawn_hist[ph_idx][our_piece as usize][to as usize] as i32;
                                let clamped = bonus.clamp(-16384, 16384);
                                let new_v = v + clamped - v * clamped.abs() / 16384;
                                info.pawn_hist[ph_idx][our_piece as usize][to as usize] = new_v.clamp(-32000, 32000) as i16;
                            }
                        }

                        // Penalize other tried quiets (main + continuation + pawn history)
                        for &q in &quiets_tried[..n_quiets_tried] {
                            let qf = move_from(q);
                            let qt = move_to(q);
                            History::update_history(
                                &mut info.history.main[color as usize][qf as usize][qt as usize],
                                -bonus,
                            );
                            // Continuation history penalty
                            if prev_move != NO_MOVE {
                                let prev_to = move_to(prev_move);
                                let prev_piece = board.piece_at(prev_to);
                                let q_piece = board.piece_at(qf);
                                if prev_piece != NO_PIECE && (prev_piece as usize) < 12
                                    && q_piece != NO_PIECE && (q_piece as usize) < 12
                                {
                                    History::update_history(
                                        &mut info.history.cont_hist[prev_piece as usize][prev_to as usize][q_piece as usize][qt as usize],
                                        -bonus,
                                    );
                                }
                            }
                            // Pawn history penalty
                            {
                                let ph_idx2 = (board.pawn_hash as usize) % PAWN_HIST_SIZE;
                                let q_piece = board.piece_at(qf);
                                if q_piece != NO_PIECE && (q_piece as usize) < 12 {
                                    let v = info.pawn_hist[ph_idx2][q_piece as usize][qt as usize] as i32;
                                    let clamped = (-bonus).clamp(-16384, 16384);
                                    let new_v = v + clamped - v * clamped.abs() / 16384;
                                    info.pawn_hist[ph_idx2][q_piece as usize][qt as usize] = new_v.clamp(-32000, 32000) as i16;
                                }
                            }
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
                            let bonus = (depth as i32 * depth as i32).min(1200);
                            History::update_history(
                                &mut info.history.capture[piece as usize][to as usize][victim as usize],
                                bonus,
                            );

                            // Penalize failed captures
                            for i in 0..n_captures_tried {
                                let (cp, ct, cv) = captures_tried[i];
                                History::update_history(
                                    &mut info.history.capture[cp as usize][ct as usize][cv as usize],
                                    -bonus,
                                );
                            }
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
        } else if is_capture {
            // Track captures for penalty on cutoff
            let piece = board.piece_at(from);
            let victim = if flags == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(to) };
            if piece != NO_PIECE && (piece as usize) < 12 && (victim as usize) < 6 && n_captures_tried < 32 {
                captures_tried[n_captures_tried] = (piece, to, victim);
                n_captures_tried += 1;
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

    // Update correction history (only when search improved on eval, non-mate, non-check)
    if !in_check && best_score > original_alpha && !is_mate_score(best_score) && raw_eval != -INFINITY {
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
        return 0;
    }

    info.nodes += 1;
    info.stats.qnodes += 1;

    if ply as usize >= MAX_PLY - 1 {
        return info.eval(board);
    }

    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    // TT probe in QS
    let tt_entry = info.tt.probe(board.hash);
    let tt_score = if tt_entry.hit { score_from_tt(tt_entry.score, ply) } else { 0 };

    // TT cutoff in QS (non-PV only)
    if tt_entry.hit && tt_entry.depth >= 0 && beta - alpha == 1 {
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
    let static_eval = if in_check {
        -INFINITY
    } else if tt_entry.hit && tt_entry.static_eval > -INFINITY + 100 {
        tt_entry.static_eval
    } else {
        info.eval(board)
    };

    if !in_check {
        if static_eval >= beta {
            // QS beta blending: dampen stand-pat cutoff at non-PV only
            if beta - alpha == 1 && !is_mate_score(static_eval) {
                return (static_eval + beta) / 2;
            }
            return static_eval;
        }
        if static_eval > alpha {
            alpha = static_eval;
        }

        // Delta pruning
        if static_eval + 240 + 900 <= alpha {
            return alpha;
        }
    }

    let original_alpha = alpha;
    let mut best_score = if in_check { -INFINITY } else { static_eval };
    let mut best_move = NO_MOVE;
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
            best_move = mv;
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

    // Store in TT from QS (with best move and exact flag)
    let qs_flag = if best_score >= beta {
        TT_FLAG_LOWER
    } else if best_score > original_alpha {
        TT_FLAG_EXACT
    } else {
        TT_FLAG_UPPER
    };
    info.tt.store(
        board.hash,
        best_move,
        qs_flag,
        static_eval,
        score_to_tt(best_score, ply),
        0, // depth 0 for QS entries
    );

    // QS capture fail-high blending (non-PV only)
    if best_score >= beta && beta - alpha == 1 && !is_mate_score(best_score) {
        return (best_score + beta) / 2;
    }

    best_score
}

/// Run bench: fixed-depth search on standard positions, return total nodes.
pub fn bench(depth: i32, nnue_path: Option<&str>) -> u64 {
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
    if let Some(path) = nnue_path {
        if let Err(e) = info.load_nnue(path) {
            eprintln!("Warning: failed to load NNUE: {}", e);
        }
    }
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

        let _mv = search(&mut board, &mut info, &limits);
        total_nodes += info.nodes;
    }

    // Print pruning stats
    let s = &info.stats;
    eprintln!("=== Pruning Stats (cumulative across all bench positions) ===");
    eprintln!("TT cutoffs:     {:>8}  ({:.1}% of nodes)", s.tt_cutoffs, s.tt_cutoffs as f64 / total_nodes as f64 * 100.0);
    eprintln!("TT near-miss:   {:>8}", s.tt_near_miss);
    eprintln!("NMP attempts:   {:>8}  cutoffs: {} ({:.0}%)", s.nmp_attempts, s.nmp_cutoffs,
        if s.nmp_attempts > 0 { s.nmp_cutoffs as f64 / s.nmp_attempts as f64 * 100.0 } else { 0.0 });
    eprintln!("RFP cutoffs:    {:>8}  ({:.1}% of nodes)", s.rfp_cutoffs, s.rfp_cutoffs as f64 / total_nodes as f64 * 100.0);
    eprintln!("Razor cutoffs:  {:>8}", s.razor_cutoffs);
    eprintln!("LMP prunes:     {:>8}", s.lmp_prunes);
    eprintln!("Futility prunes:{:>8}", s.futility_prunes);
    eprintln!("History prunes: {:>8}", s.history_prunes);
    eprintln!("SEE prunes:     {:>8}", s.see_prunes);
    eprintln!("ProbCut cutoffs:{:>8}", s.probcut_cutoffs);
    eprintln!("LMR searches:   {:>8}  ({:.1}% of nodes)", s.lmr_searches, s.lmr_searches as f64 / total_nodes as f64 * 100.0);
    eprintln!("Recapture ext:  {:>8}", s.recapture_ext);
    eprintln!("QS nodes:       {:>8}  ({:.1}% of total)", s.qnodes, s.qnodes as f64 / total_nodes as f64 * 100.0);
    eprintln!("Total nodes:    {:>8}", total_nodes);

    total_nodes
}
