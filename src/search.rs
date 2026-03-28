/// Main search: negamax with alpha-beta, iterative deepening, PVS, aspiration windows.
/// All pruning parameters ported from GoChess (tuned values).
/// This is a literal, faithful translation of GoChess's search.go.

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

const MAX_PLY: usize = 64;
const INFINITY: i32 = 30000;
const CONTEMPT: i32 = 10; // prefer playing on over drawing (GoChess: +10)

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
    pub nmp_verify: u64,
    pub nmp_verify_fail: u64,
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
        let score = if let (Some(net), Some(acc)) = (&self.nnue_net, &mut self.nnue_acc) {
            evaluate_nnue(board, net, acc)
        } else {
            evaluate(board)
        };
        // Debug: dump hash + eval (enabled with CODA_DUMP_EVAL=1)
        static DUMP_INIT: std::sync::Once = std::sync::Once::new();
        static DUMP_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static mut DUMP_ENABLED: bool = false;
        DUMP_INIT.call_once(|| {
            unsafe { DUMP_ENABLED = std::env::var("CODA_DUMP_EVAL").is_ok(); }
        });
        if unsafe { DUMP_ENABLED } {
            let n = DUMP_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n < 3000 {
                eprintln!("EVAL n={} hash={:016x} eval={}", n, board.hash, score);
            }
        }
        score
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

    // Non-pawn corrections (per color)
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) % CORR_HIST_SIZE;
    let white_np_corr = info.np_corr[stm][WHITE as usize][white_np_idx] as i64;
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) % CORR_HIST_SIZE;
    let black_np_corr = info.np_corr[stm][BLACK as usize][black_np_idx] as i64;

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
    let new_val = (*entry * (CORR_HIST_GRAIN - weight) + err * CORR_HIST_GRAIN * weight) / CORR_HIST_GRAIN;
    *entry = new_val.clamp(-CORR_HIST_LIMIT, CORR_HIST_LIMIT);
}

/// Update all correction history tables.
fn update_correction_history(info: &mut SearchInfo, board: &Board, search_score: i32, raw_eval: i32, depth: i32) {
    let err = (search_score - raw_eval).clamp(-CORR_HIST_MAX, CORR_HIST_MAX);
    let weight = (depth + 1).min(16);
    let stm = board.side_to_move as usize;

    // Pawn correction
    let pawn_idx = (board.pawn_hash as usize) % CORR_HIST_SIZE;
    update_corr_entry(&mut info.pawn_corr[stm][pawn_idx], err, weight);

    // Non-pawn corrections (per color)
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) % CORR_HIST_SIZE;
    update_corr_entry(&mut info.np_corr[stm][WHITE as usize][white_np_idx], err, weight);
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) % CORR_HIST_SIZE;
    update_corr_entry(&mut info.np_corr[stm][BLACK as usize][black_np_idx], err, weight);

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
                // Quiet table: C=1.30 (GoChess: depth>=3 && moveNum>=3)
                if depth >= 3 && moves >= 3 {
                    let r = ((depth as f64).ln() * (moves as f64).ln() / 1.30) as i32;
                    LMR_TABLE[depth][moves] = r.min((depth - 2) as i32);
                }
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

        // Hard limit (match GoChess)
        let hard = if limits.movestogo > 0 {
            // Tournament TC: cap by moves remaining (generous early, tight late)
            let hard_raw = soft * 2;
            let cap_pct = (20 + limits.movestogo as u64 / 2).min(40);
            let mtg_cap = time_left * cap_pct / 100;
            hard_raw.min(mtg_cap)
        } else {
            // Sudden death: allow up to 3x soft
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

        // Aspiration windows (GoChess: skip for mate scores)
        if depth >= 4 && prev_score > -MATE_SCORE + 100 && prev_score < MATE_SCORE - 100 {
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
/// Faithful translation of GoChess's negamax() from search.go.
fn negamax(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    mut beta: i32,
    mut depth: i32,
    ply: i32,
    _cut_node: bool, // kept for API compatibility, not used (GoChess doesn't have cut_node)
) -> i32 {
    // Guard against stack overflow
    let ply_u = ply as usize;
    if ply_u >= MAX_PLY - 1 {
        return info.eval(board);
    }

    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    // Check time periodically
    if info.nodes & 1023 == 0 {
        if info.should_stop() {
            return 0;
        }
    }

    if info.stop.load(Ordering::Relaxed) {
        return 0;
    }

    info.nodes += 1;

    let is_root = ply == 0;
    let is_pv = beta - alpha > 1;
    let alpha_orig = alpha;

    // Draw detection (not at root)
    if !is_root {
        if board.halfmove >= 100 {
            return -CONTEMPT;
        }
        // Insufficient material: KvK, KNvK, KBvK
        let occ = board.occupied();
        let pc = popcount(occ);
        if pc <= 3 {
            if pc == 2 { return -CONTEMPT; } // KvK
            if board.pieces[PAWN as usize] == 0
                && board.pieces[ROOK as usize] == 0
                && board.pieces[QUEEN as usize] == 0
            {
                return -CONTEMPT;
            }
        }
        // Repetition detection
        let hash = board.hash;
        let stack_len = board.undo_stack.len();
        let check_back = (board.halfmove as usize).min(stack_len);
        for i in 0..check_back {
            if board.undo_stack[stack_len - 1 - i].hash == hash {
                return -CONTEMPT;
            }
        }
    }

    // TT probe
    let tt_entry = info.tt.probe(board.hash);
    let tt_move = if tt_entry.hit { tt_entry.best_move } else { NO_MOVE };
    let tt_hit = tt_entry.hit;

    // TT cutoff (GoChess: returns TTExact at PV nodes, narrows bounds at non-PV)
    if tt_hit && !is_root {
        let tt_depth = tt_entry.depth;
        let tt_score = score_from_tt(tt_entry.score, ply);

        if tt_depth >= depth {
            match tt_entry.flag {
                TT_FLAG_EXACT => {
                    info.stats.tt_cutoffs += 1;
                    // History bonus for TT cutoff
                    if tt_move != NO_MOVE {
                        let bonus = (depth * depth).min(1200);
                        update_tt_cutoff_history(info, board, tt_move, bonus);
                    }
                    return tt_score;
                }
                TT_FLAG_LOWER => {
                    if !is_pv && tt_score > alpha {
                        alpha = tt_score;
                    }
                }
                TT_FLAG_UPPER => {
                    if !is_pv && tt_score < beta {
                        beta = tt_score;
                    }
                }
                _ => {}
            }

            if alpha >= beta {
                info.stats.tt_cutoffs += 1;
                if tt_move != NO_MOVE {
                    let bonus = (depth * depth).min(1200);
                    update_tt_cutoff_history(info, board, tt_move, bonus);
                }
                // TT score dampening at non-PV lower-bound cutoffs
                if !is_pv && tt_entry.flag == TT_FLAG_LOWER && !is_mate_score(tt_score) {
                    return (3 * tt_score + beta) / 4;
                }
                return tt_score;
            }
        } else if tt_depth >= depth - 1 && !is_pv && !is_mate_score(tt_score) {
            // TT near-miss cutoffs
            let margin = 80;
            if tt_entry.flag == TT_FLAG_LOWER && tt_score - margin >= beta {
                info.stats.tt_near_miss += 1;
                return tt_score - margin;
            }
            if tt_entry.flag == TT_FLAG_UPPER && tt_score + margin <= alpha {
                info.stats.tt_near_miss += 1;
                return tt_score + margin;
            }
        }
    }

    // Leaf node - go to quiescence search
    if depth <= 0 {
        return quiescence(board, info, alpha, beta, ply);
    }

    let in_check = board.in_check();

    // Static eval with correction history
    let raw_eval: i32;
    let static_eval: i32;
    let improving: bool;
    let failing: bool;
    if !in_check {
        raw_eval = if tt_hit && tt_entry.static_eval > -INFINITY + 100 {
            tt_entry.static_eval
        } else {
            info.eval(board)
        };
        static_eval = corrected_eval(info, board, raw_eval);
        if ply_u < MAX_PLY {
            info.static_evals[ply_u] = static_eval;
        }
        // Improving: our eval is better than 2 plies ago
        improving = ply >= 2 && ply_u >= 2 && static_eval > info.static_evals[ply_u - 2];
        // Failing: significant position deterioration
        failing = ply >= 2 && ply_u >= 2
            && info.static_evals[ply_u - 2] > -INFINITY + 100
            && static_eval < info.static_evals[ply_u - 2] - (60 + 40 * depth);
    } else {
        raw_eval = -INFINITY;
        static_eval = -INFINITY;
        if ply_u < MAX_PLY {
            info.static_evals[ply_u] = -INFINITY;
        }
        improving = false;
        failing = false;
    }

    // Eval instability: sharp eval swing from parent
    let unstable = !in_check && ply >= 1 && ply_u >= 1
        && info.static_evals[ply_u - 1] > -INFINITY + 100
        && static_eval > -INFINITY
        && {
            let parent_eval = -info.static_evals[ply_u - 1];
            let diff = (static_eval - parent_eval).abs();
            diff > 200
        };

    // Detect if TT move is a capture
    let tt_move_noisy = tt_move != NO_MOVE && {
        let tt_to = move_to(tt_move);
        board.piece_type_at(tt_to) != NO_PIECE_TYPE || move_flags(tt_move) == FLAG_EN_PASSANT
    };

    // IIR: Internal Iterative Reduction
    if depth >= 6 && tt_move == NO_MOVE && !in_check {
        depth -= 1;
    }

    // Threat square from null-move failure (-1 = no threat detected)
    let mut _threat_sq: i32 = -1;

    // Hindsight reduction
    if !in_check && ply >= 1 && depth >= 3 && ply_u >= 1
        && info.static_evals[ply_u - 1] > -INFINITY + 100
        && static_eval > -INFINITY
    {
        let eval_sum = info.static_evals[ply_u - 1] + static_eval;
        if eval_sum > 200 {
            depth -= 1;
        }
    }

    // Null-move pruning (GoChess uses beta-alpha==1, not !is_pv)
    let us = board.side_to_move;
    let stm_non_pawn = board.colors[us as usize]
        & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
    if depth >= 3 && !in_check && !is_root && stm_non_pawn != 0
        && !is_pv && static_eval >= beta
    {
        // Adaptive reduction
        let mut r = 3 + depth / 3;
        // Reduce less after captures
        if !board.undo_stack.is_empty() && board.undo_stack.last().unwrap().captured != NO_PIECE_TYPE {
            r -= 1;
        }
        if static_eval > beta {
            let eval_r = ((static_eval - beta) / 200).min(3);
            r += eval_r;
        }
        // Clamp: null-move search is at least depth 1
        if depth - 1 - r < 1 {
            r = depth - 2;
        }

        info.stats.nmp_attempts += 1;
        board.make_null_move();
        let null_key = board.hash; // save hash for threat detection
        if let Some(acc) = &mut info.nnue_acc { acc.push(DirtyPiece::recompute()); }
        let null_score = -negamax(board, info, -beta, -beta + 1, depth - 1 - r, ply + 1, false);
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        board.unmake_null_move();

        if info.stop.load(Ordering::Relaxed) {
            return 0;
        }

        if null_score >= beta {
            // NMP score dampening
            let dampened = (null_score * 2 + beta) / 3;

            // Verification at high depth
            if depth >= 12 {
                info.stats.nmp_verify += 1;
                let v = negamax(board, info, beta - 1, beta, depth - 1 - r, ply + 1, false);
                if v >= beta {
                    info.stats.nmp_cutoffs += 1;
                    return dampened;
                }
                info.stats.nmp_verify_fail += 1;
            } else {
                info.stats.nmp_cutoffs += 1;
                return dampened;
            }
        } else {
            // NMP failed low: extract opponent's best reply from TT for threat detection
            let threat_entry = info.tt.probe(null_key);
            if threat_entry.hit && threat_entry.best_move != NO_MOVE {
                _threat_sq = move_to(threat_entry.best_move) as i32;
            }
        }
    }

    if !in_check {
        // Reverse Futility Pruning
        if depth <= 7 && !is_root {
            let rfp_margin = if improving { depth * 70 } else { depth * 100 };
            if static_eval - rfp_margin >= beta {
                info.stats.rfp_cutoffs += 1;
                return static_eval - rfp_margin;
            }
        }

        // Razoring
        if depth <= 2 && !is_root {
            let razor_margin = 400 + depth * 100;
            if static_eval + razor_margin < alpha {
                let q_score = quiescence(board, info, alpha, beta, ply);
                if q_score < alpha {
                    return q_score;
                }
                info.stats.razor_cutoffs += 1;
            }
        }
    }

    // ProbCut
    let probcut_beta = beta + 170;
    if !in_check && !is_root && depth >= 5 && static_eval + 85 >= probcut_beta {
        let pc_depth = depth - 4;
        let mut pc_picker = QMovePicker::new(board, false);
        loop {
            let mv = pc_picker.next(board, false);
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
            let score = -negamax(board, info, -probcut_beta, -probcut_beta + 1, pc_depth, ply + 1, false);
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }

            if info.stop.load(Ordering::Relaxed) {
                return 0;
            }

            if score >= probcut_beta {
                info.stats.probcut_cutoffs += 1;
                return score;
            }
        }
    }

    // Get killers for this ply
    let safe_ply = ply_u.min(MAX_PLY - 1).min(127);
    let killers = info.history.killers[safe_ply];

    // Counter-move and continuation history lookup from opponent's last move
    let prev_move = if !is_root && !board.undo_stack.is_empty() {
        board.undo_stack[board.undo_stack.len() - 1].mv
    } else {
        NO_MOVE
    };

    // Compute enemy pawn attacks for threat-aware LMR
    let enemy_pawn_attacks = if !in_check {
        let them = flip_color(us);
        let enemy_pawns = board.pieces[PAWN as usize] & board.colors[them as usize];
        if us == WHITE {
            south_west(enemy_pawns) | south_east(enemy_pawns)
        } else {
            north_west(enemy_pawns) | north_east(enemy_pawns)
        }
    } else {
        0
    };

    // Record prev_move for continuation history
    if ply_u < MAX_PLY {
        info.prev_moves[safe_ply] = NO_MOVE; // will be updated per-move
    }

    let mut picker = MovePicker::new(board, tt_move, safe_ply, &info.history, prev_move);

    let mut best_score = -INFINITY;
    let mut best_move = NO_MOVE;
    let mut move_count = 0i32;
    let mut alpha_raised_count = 0i32;

    // Track quiet moves searched before beta cutoff
    let mut quiets_tried = [NO_MOVE; 64];
    let mut quiets_count = 0usize;

    // Track captures searched before beta cutoff
    let mut captures_tried: [(u8, u8, u8); 32] = [(0, 0, 0); 32]; // (piece, to, victim)
    let mut n_captures_tried = 0usize;

    loop {
        let ph_idx = (board.pawn_hash as usize) % PAWN_HIST_SIZE;
        let mv = picker.next(board, &info.history, prev_move, Some(&info.pawn_hist[ph_idx]));
        if mv == NO_MOVE { break; }

        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);
        let is_cap = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
        let is_promo = is_promotion(mv);

        // Save moved piece before make_move for consistent history indexing
        let moved_pt = board.piece_type_at(from);
        let moved_piece = board.piece_at(from);
        let captured_pt = if is_cap {
            if flags == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(to) }
        } else {
            NO_PIECE_TYPE
        };

        // SEE capture pruning (before MakeMove, GoChess order)
        if is_cap && !is_root && !in_check && depth <= 6
            && mv != tt_move && best_score > -MATE_SCORE + 100
            && !see_ge(board, mv, -(depth * 100))
        {
            continue;
        }

        // SEE quiet pruning: compute SEE before MakeMove
        let is_killer = !is_cap && (mv == killers[0] || mv == killers[1]);
        let counter_move = get_counter_move(&info.history, board, prev_move);
        let is_counter = mv == counter_move;

        let mut check_see_quiet = false;
        let mut see_quiet_ok = true;
        if !is_root && !in_check && depth <= 8
            && !is_cap && !is_promo
            && !is_killer && mv != counter_move && mv != tt_move
            && best_score > -MATE_SCORE + 100
        {
            let mut see_quiet_threshold = -20 * depth * depth;
            if unstable {
                see_quiet_threshold -= 100;
            }
            check_see_quiet = true;
            see_quiet_ok = see_ge(board, mv, see_quiet_threshold);
        }

        // History-based pruning (GoChess: exempt ttMove, killers, counterMove)
        if !is_root && !in_check && !improving && !unstable && depth <= 3
            && !is_cap && !is_promo
            && mv != tt_move && !is_killer && !is_counter
            && best_score > -MATE_SCORE + 100
        {
            let mut hist_prune_score = info.history.main[from as usize][to as usize];
            // Add continuation history
            if prev_move != NO_MOVE {
                let prev_to = move_to(prev_move);
                let prev_piece = board.piece_at(prev_to);
                if prev_piece != NO_PIECE && (prev_piece as usize) < 12
                    && moved_piece != NO_PIECE && (moved_piece as usize) < 12
                {
                    hist_prune_score += info.history.cont_hist[prev_piece as usize][prev_to as usize][moved_piece as usize][to as usize];
                }
            }
            if hist_prune_score < -1500 * depth {
                info.stats.history_prunes += 1;
                continue;
            }
        }

        // Bad noisy flag: identify losing captures for tighter futility pruning
        let is_bad_noisy = is_cap && !in_check && !is_root && depth <= 4 && mv != tt_move
            && !is_promo && best_score > -MATE_SCORE + 100
            && static_eval > -INFINITY && static_eval + depth * 75 <= alpha
            && !see_ge(board, mv, 0);

        // Build NNUE dirty piece info BEFORE make_move
        let dirty = build_dirty_piece(mv, us, flip_color(us), moved_pt, captured_pt);

        // Push NNUE accumulator
        if let Some(acc) = &mut info.nnue_acc { acc.push(dirty); }

        if !board.make_move(mv) {
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            continue;
        }

        move_count += 1;

        // Record previous move for continuation history
        if safe_ply < MAX_PLY {
            info.prev_moves[safe_ply] = mv;
        }

        // Check if move gives check (opponent is now in check)
        let gives_check = board.in_check();

        // Bad noisy futility: prune losing captures when eval is far below alpha
        if is_bad_noisy && !gives_check {
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            move_count -= 1;
            continue;
        }

        // Futility pruning (GoChess: no !is_pv guard)
        if static_eval > -INFINITY && depth <= 8 && !in_check && !gives_check
            && !is_cap && !is_promo
            && best_score > -MATE_SCORE + 100
        {
            // Estimate LMR reduction for this move
            let mut lmr_depth = depth;
            if move_count > 1 && depth >= 2 {
                let d = (depth as usize).min(63);
                let m = (move_count as usize).min(63);
                let r = lmr_reduction(d as i32, m as i32);
                if r > 0 {
                    lmr_depth = (depth - r).max(1);
                }
            }
            if static_eval + 60 + lmr_depth * 60 <= alpha {
                info.stats.futility_prunes += 1;
                board.unmake_move();
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                continue;
            }
        }

        // Late Move Pruning (GoChess: beta-alpha==1, not !is_pv)
        if !is_root && !in_check && depth >= 1 && depth <= 8
            && !is_cap && !is_promo && !gives_check
            && best_score > -MATE_SCORE + 100 && !is_pv
        {
            let mut lmp_limit = 3 + (depth * depth) as i32;
            if improving && depth >= 3 {
                lmp_limit += lmp_limit / 2;
            }
            if failing {
                lmp_limit = lmp_limit * 2 / 3;
            }
            if move_count > lmp_limit {
                info.stats.lmp_prunes += 1;
                board.unmake_move();
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                continue;
            }
        }

        // SEE quiet pruning (uses pre-computed result)
        if check_see_quiet && !gives_check && !see_quiet_ok {
            info.stats.see_prunes += 1;
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            continue;
        }

        // Recapture extension
        let mut extension = 0;
        if is_cap && board.undo_stack.len() >= 2 {
            let prev = &board.undo_stack[board.undo_stack.len() - 2];
            if prev.captured != NO_PIECE_TYPE && to == move_to(prev.mv) {
                extension = 1;
                info.stats.recapture_ext += 1;
            }
        }

        let mut new_depth = depth - 1 + extension;

        // Alpha-reduce
        if alpha_raised_count > 0 {
            new_depth -= 1;
        }
        if new_depth < 0 { new_depth = 0; }

        let score;

        // Track quiet moves for history penalty on beta cutoff
        if !is_cap && !is_promo && quiets_count < 64 {
            quiets_tried[quiets_count] = mv;
            quiets_count += 1;
        }

        // Track captures for capture history penalty
        if is_cap && n_captures_tried < 32 {
            let piece = moved_piece;
            let victim = captured_pt;
            if piece != NO_PIECE && (piece as usize) < 12 && (victim as usize) < 6 {
                captures_tried[n_captures_tried] = (piece, to, victim);
                n_captures_tried += 1;
            }
        }

        // LMR + PVS
        let mut reduction = 0i32;

        // Quiet LMR
        if !in_check && !is_cap && !is_promo && !is_killer && !gives_check {
            let d = (depth as usize).min(63);
            let m = (move_count as usize).min(63);
            reduction = lmr_reduction(d as i32, m as i32);

            if reduction > 0 {
                // Reduce less at PV nodes
                if is_pv {
                    reduction -= 1;
                }

                // Reduce more at expected cut nodes
                if !is_pv && move_count > 1 {
                    reduction += 1;
                }

                // Reduce less when improving
                if improving {
                    reduction -= 1;
                }

                // Reduce more when failing
                if failing {
                    reduction += 1;
                }

                // Reduce more when multiple alpha raises
                if alpha_raised_count > 1 {
                    reduction += alpha_raised_count / 2;
                }

                // Reduce less when eval is unstable
                if unstable {
                    reduction -= 1;
                }

                // Reduce more when TT move is a capture
                if tt_move_noisy {
                    reduction += 1;
                }

                // Reduce more when opponent has few non-pawn pieces
                let opp = flip_color(board.side_to_move); // note: board is post-make, opponent is now us
                let opp_non_pawn = board.colors[opp as usize]
                    & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
                if popcount(opp_non_pawn) < 3 {
                    reduction += 1;
                }

                // Reduce less when moving piece away from pawn-attacked square
                if enemy_pawn_attacks & (1u64 << from) != 0 {
                    reduction -= 1;
                }

                // Continuous history adjustment
                let mut hist_score = info.history.main[from as usize][to as usize];
                if prev_move != NO_MOVE {
                    let prev_to = move_to(prev_move);
                    let prev_piece = board.piece_at(prev_to);
                    // Note: after make_move the piece is at 'to' now. But for cont_hist
                    // we need the piece index. moved_piece was captured before make_move.
                    if prev_piece != NO_PIECE && (prev_piece as usize) < 12
                        && moved_piece != NO_PIECE && (moved_piece as usize) < 12
                    {
                        hist_score += info.history.cont_hist[prev_piece as usize][prev_to as usize][moved_piece as usize][to as usize];
                    }
                }
                reduction -= hist_score / 5000;

                // Clamp: never extend, never reduce past depth 1
                if reduction < 0 { reduction = 0; }
                if reduction > new_depth - 1 { reduction = new_depth - 1; }
            }
        }

        // Capture LMR: separate capture table with capture history adjustments
        if !in_check && is_cap && !is_promo && !gives_check && move_count > 1 && mv != tt_move {
            // Only reduce at non-PV nodes
            if !is_pv {
                let d = (depth as usize).min(63);
                let m = (move_count as usize).min(63);
                reduction = lmr_cap_reduction(d as i32, m as i32);

                if reduction > 0 {
                    // Capture history adjustment
                    let cpt = captured_pt;
                    if moved_piece != NO_PIECE && (moved_piece as usize) < 12 && (cpt as usize) < 6 {
                        let capt_hist_val = info.history.capture[moved_piece as usize][to as usize][cpt as usize];
                        if capt_hist_val > 2000 { reduction -= 1; }
                        if capt_hist_val < -2000 { reduction += 1; }
                    }

                    if reduction < 0 { reduction = 0; }
                    if reduction > new_depth - 1 { reduction = new_depth - 1; }
                }
            }
        }

        if reduction > 0 {
            info.stats.lmr_searches += 1;

            // LMR: reduced depth, zero window
            let lmr_depth = new_depth - reduction;
            let mut lmr_score = -negamax(board, info, -alpha - 1, -alpha, lmr_depth, ply + 1, true);

            if lmr_score > alpha && !info.stop.load(Ordering::Relaxed) {
                // doDeeper / doShallower
                let mut do_deeper_adj = 0;
                if lmr_score > best_score + 60 + 10 * reduction {
                    do_deeper_adj = 1;
                } else if lmr_score < best_score + new_depth {
                    do_deeper_adj = -1;
                }

                lmr_score = -negamax(board, info, -alpha - 1, -alpha, new_depth + do_deeper_adj, ply + 1, false);
            }

            if lmr_score > alpha && lmr_score < beta && !info.stop.load(Ordering::Relaxed) {
                // PVS failed high → full window re-search
                score = -negamax(board, info, -beta, -alpha, new_depth, ply + 1, false);
            } else {
                score = lmr_score;
            }
        } else if move_count > 1 {
            // PVS: zero-window for non-first moves
            let mut pvs_score = -negamax(board, info, -alpha - 1, -alpha, new_depth, ply + 1, false);
            if pvs_score > alpha && pvs_score < beta && !info.stop.load(Ordering::Relaxed) {
                pvs_score = -negamax(board, info, -beta, -alpha, new_depth, ply + 1, false);
            }
            score = pvs_score;
        } else {
            // First move: always full window
            score = -negamax(board, info, -beta, -alpha, new_depth, ply + 1, false);
        }

        board.unmake_move();
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }

        if info.stop.load(Ordering::Relaxed) {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = mv;

            if score > alpha {
                alpha = score;
                alpha_raised_count += 1;

                if alpha >= beta {
                    info.stats.beta_cutoffs += 1;
                    if move_count == 1 { info.stats.first_move_cutoffs += 1; }

                    // Beta cutoff: update history
                    if !is_cap {
                        let bonus = (depth * depth).min(1200);

                        // Update killer
                        if safe_ply < 128 {
                            if info.history.killers[safe_ply][0] != mv {
                                info.history.killers[safe_ply][1] = info.history.killers[safe_ply][0];
                                info.history.killers[safe_ply][0] = mv;
                            }
                        }

                        // Update main history
                        let color = us;
                        History::update_history(
                            &mut info.history.main[from as usize][to as usize],
                            bonus,
                        );

                        // Update continuation history
                        if prev_move != NO_MOVE {
                            let prev_to = move_to(prev_move);
                            let prev_piece = board.piece_at(prev_to);
                            if prev_piece != NO_PIECE && (prev_piece as usize) < 12
                                && moved_piece != NO_PIECE && (moved_piece as usize) < 12
                            {
                                History::update_history(
                                    &mut info.history.cont_hist[prev_piece as usize][prev_to as usize][moved_piece as usize][to as usize],
                                    bonus,
                                );
                            }
                        }

                        // Update pawn history
                        {
                            let ph_idx = (board.pawn_hash as usize) % PAWN_HIST_SIZE;
                            if moved_piece != NO_PIECE && (moved_piece as usize) < 12 {
                                let v = info.pawn_hist[ph_idx][moved_piece as usize][to as usize] as i32;
                                let clamped = bonus.clamp(-16384, 16384);
                                let new_v = v + clamped - v * clamped.abs() / 16384;
                                info.pawn_hist[ph_idx][moved_piece as usize][to as usize] = new_v.clamp(-32000, 32000) as i16;
                            }
                        }

                        // Penalize quiets tried before cutoff (excluding the cutoff move itself)
                        for i in 0..quiets_count.saturating_sub(1) {
                            let q = quiets_tried[i];
                            let qf = move_from(q);
                            let qt = move_to(q);
                            History::update_history(
                                &mut info.history.main[qf as usize][qt as usize],
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

                        // Store counter-move
                        if prev_move != NO_MOVE {
                            let prev_to = move_to(prev_move);
                            let prev_piece = board.piece_at(prev_to);
                            if prev_piece != NO_PIECE && (prev_piece as usize) < 12 {
                                info.history.counter[prev_piece as usize][prev_to as usize] = mv;
                            }
                        }
                    } else {
                        // Capture caused beta cutoff — update capture history
                        let bonus = (depth * depth).min(1200);
                        let piece = moved_piece;
                        let victim = captured_pt;
                        if piece != NO_PIECE && (piece as usize) < 12 && (victim as usize) < 6 {
                            History::update_history(
                                &mut info.history.capture[piece as usize][to as usize][victim as usize],
                                bonus,
                            );

                            // Penalize captures tried before cutoff
                            for i in 0..n_captures_tried.saturating_sub(1) {
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
    }

    // Checkmate / stalemate
    if move_count == 0 {
        if in_check {
            return -MATE_SCORE + ply;
        } else {
            return 0;
        }
    }

    // Store in TT
    let tt_flag = if best_score <= alpha_orig {
        TT_FLAG_UPPER
    } else if best_score >= beta {
        TT_FLAG_LOWER
    } else {
        TT_FLAG_EXACT
    };

    info.tt.store(
        board.hash,
        depth,
        score_to_tt(best_score, ply),
        tt_flag,
        best_move,
        raw_eval,
    );

    // Update correction history
    if !in_check && best_move != NO_MOVE && depth >= 3
        && best_score > alpha_orig
        && !is_mate_score(best_score)
        && raw_eval > -INFINITY + 100
    {
        update_correction_history(info, board, best_score, raw_eval, depth);
    }

    // Fail-high score blending
    if best_score >= beta && !is_pv && depth >= 3
        && !is_mate_score(best_score)
    {
        return (best_score * depth + beta) / (depth + 1);
    }

    best_score
}

/// Helper: update history for TT cutoff moves
fn update_tt_cutoff_history(info: &mut SearchInfo, board: &Board, tt_move: Move, bonus: i32) {
    let tt_from = move_from(tt_move);
    let tt_to = move_to(tt_move);
    let tt_target = board.piece_type_at(tt_to);
    let tt_is_cap = tt_target != NO_PIECE_TYPE || move_flags(tt_move) == FLAG_EN_PASSANT;
    if !tt_is_cap {
        History::update_history(
            &mut info.history.main[tt_from as usize][tt_to as usize],
            bonus,
        );
    } else {
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

/// Helper: get counter-move for the given previous move
fn get_counter_move(history: &History, board: &Board, prev_move: Move) -> Move {
    if prev_move != NO_MOVE {
        let prev_to = move_to(prev_move);
        let prev_piece = board.piece_at(prev_to);
        if prev_piece != NO_PIECE && (prev_piece as usize) < 12 {
            return history.counter[prev_piece as usize][prev_to as usize];
        }
    }
    NO_MOVE
}

/// Quiescence search.
/// Faithful translation of GoChess's quiescenceWithDepth() from search.go.
fn quiescence(
    board: &mut Board,
    info: &mut SearchInfo,
    alpha: i32,
    beta: i32,
    ply: i32,
) -> i32 {
    quiescence_with_depth(board, info, alpha, beta, ply, 0)
}

fn quiescence_with_depth(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    beta: i32,
    ply: i32,
    qs_depth: i32,
) -> i32 {
    info.stats.qnodes += 1;

    // Limit quiescence depth
    if qs_depth >= 32 {
        return info.eval(board);
    }

    info.nodes += 1;

    if ply as usize >= MAX_PLY - 1 {
        return info.eval(board);
    }

    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    // Check time periodically
    if info.nodes & 1023 == 0 {
        if info.should_stop() {
            return 0;
        }
    }

    if info.stop.load(Ordering::Relaxed) {
        return 0;
    }

    // TT probe
    let tt_entry = info.tt.probe(board.hash);
    let _tt_move = if tt_entry.hit { tt_entry.best_move } else { NO_MOVE };
    let alpha_orig = alpha;
    let tt_hit = tt_entry.hit;

    // TT cutoff in QS (GoChess: depth >= -1, no !is_pv guard)
    if tt_hit && tt_entry.depth >= -1 {
        let tt_score = score_from_tt(tt_entry.score, ply);

        match tt_entry.flag {
            TT_FLAG_EXACT => { return tt_score; }
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

    // In-check evasion path
    if in_check {
        // Continuation history pointer for evasion ordering
        let _prev_move = if !board.undo_stack.is_empty() {
            board.undo_stack[board.undo_stack.len() - 1].mv
        } else { NO_MOVE };

        let mut evasion_picker = QMovePicker::new(board, true);
        let mut best_score = -INFINITY;
        let mut best_move = NO_MOVE;
        let mut move_count = 0;

        loop {
            let mv = evasion_picker.next(board, true);
            if mv == NO_MOVE { break; }
            move_count += 1;

            let qs_moved_pt = board.piece_type_at(move_from(mv));
            let qs_captured_pt = if move_flags(mv) == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(move_to(mv)) };
            let qs_dirty = build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), qs_moved_pt, qs_captured_pt);

            if let Some(acc) = &mut info.nnue_acc { acc.push(qs_dirty); }
            if !board.make_move(mv) {
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                move_count -= 1;
                continue;
            }
            let score = -quiescence_with_depth(board, info, -beta, -alpha, ply + 1, qs_depth + 1);
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }

            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            if score > alpha {
                alpha = score;
                if score >= beta {
                    break;
                }
            }
        }

        // Checkmate detection
        if move_count == 0 {
            return -MATE_SCORE + ply;
        }

        // Store in TT at depth -1
        let store_score = score_to_tt(best_score, ply);
        let flag = if best_score >= beta {
            TT_FLAG_LOWER
        } else if best_score <= alpha_orig {
            TT_FLAG_UPPER
        } else {
            TT_FLAG_EXACT
        };
        info.tt.store(board.hash, -1, store_score, flag, best_move, -INFINITY);
        return best_score;
    }

    // Stand pat (only when not in check)
    let stand_pat = if tt_hit && tt_entry.static_eval > -INFINITY + 100 {
        tt_entry.static_eval
    } else {
        info.eval(board)
    };
    let mut best_score = stand_pat;

    if best_score >= beta {
        // QS beta blending: dampen stand-pat cutoff at non-PV nodes
        if beta - alpha == 1 && !is_mate_score(best_score) {
            return (best_score + beta) / 2;
        }
        return best_score;
    }

    if best_score > alpha {
        alpha = best_score;
    }

    let mut best_move = NO_MOVE;
    let mut picker = QMovePicker::new(board, false);

    loop {
        let mv = picker.next(board, false);
        if mv == NO_MOVE { break; }

        // Delta pruning per-move (GoChess: standPat + SEEPieceValues[captured] + 240 <= alpha)
        if !is_promotion(mv) {
            let cap_to = move_to(mv);
            let cap_pt = if move_flags(mv) == FLAG_EN_PASSANT {
                PAWN
            } else {
                board.piece_type_at(cap_to)
            };
            if cap_pt != NO_PIECE_TYPE && (cap_pt as usize) < 6 {
                if stand_pat + see_value(cap_pt) + 240 <= alpha {
                    continue;
                }
            }
        }

        // SEE pruning: skip bad captures
        if !see_ge(board, mv, 0) {
            continue;
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
        let score = -quiescence_with_depth(board, info, -beta, -alpha, ply + 1, qs_depth + 1);
        board.unmake_move();
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }

        if score > best_score {
            best_score = score;
            best_move = mv;
        }
        if score > alpha {
            alpha = score;
            if score >= beta {
                break;
            }
        }
    }

    // Store in TT at depth -1
    let store_score = score_to_tt(best_score, ply);
    let flag = if best_score >= beta {
        TT_FLAG_LOWER
    } else if best_score <= alpha_orig {
        TT_FLAG_UPPER
    } else {
        TT_FLAG_EXACT
    };
    info.tt.store(board.hash, -1, store_score, flag, best_move, stand_pat);

    // QS capture fail-high blending (non-PV only)
    if best_score >= beta && beta - alpha_orig == 1 && !is_mate_score(best_score) {
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
