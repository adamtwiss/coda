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

// Feature flags for ablation testing. All true = normal play.
pub static mut FEAT_NMP: bool = true;
pub static mut FEAT_RFP: bool = true;
pub static mut FEAT_RAZORING: bool = true;
pub static mut FEAT_PROBCUT: bool = true;
pub static mut FEAT_LMR: bool = true;
pub static mut FEAT_LMP: bool = true;
pub static mut FEAT_FUTILITY: bool = true;
pub static mut FEAT_SEE_PRUNE: bool = true;
pub static mut FEAT_HIST_PRUNE: bool = true;
pub static mut FEAT_BAD_NOISY: bool = true;
pub static mut FEAT_EXTENSIONS: bool = true;
pub static mut FEAT_ALPHA_REDUCE: bool = true;
pub static mut FEAT_IIR: bool = true;
pub static mut FEAT_HINDSIGHT: bool = true;
pub static mut FEAT_CORRECTION: bool = true;
pub static mut FEAT_PVS: bool = true;
pub static mut FEAT_TT_CUTOFF: bool = true;
pub static mut FEAT_TT_NEARMISS: bool = true;
pub static mut FEAT_TT_STORE: bool = true;
pub static mut FEAT_QS_CAPTURES: bool = true; // false = QS returns eval immediately

/// Disable all features (pure negamax + eval)
pub fn disable_all_features() {
    unsafe {
        FEAT_NMP = false; FEAT_RFP = false; FEAT_RAZORING = false;
        FEAT_PROBCUT = false; FEAT_LMR = false; FEAT_LMP = false;
        FEAT_FUTILITY = false; FEAT_SEE_PRUNE = false; FEAT_HIST_PRUNE = false;
        FEAT_BAD_NOISY = false; FEAT_EXTENSIONS = false; FEAT_ALPHA_REDUCE = false;
        FEAT_IIR = false; FEAT_HINDSIGHT = false; FEAT_CORRECTION = false;
        FEAT_PVS = false; FEAT_TT_CUTOFF = false; FEAT_TT_NEARMISS = false;
        FEAT_TT_STORE = false; FEAT_QS_CAPTURES = false;
    }
}

/// Enable all features (normal play)
pub fn enable_all_features() {
    unsafe {
        FEAT_NMP = true; FEAT_RFP = true; FEAT_RAZORING = true;
        FEAT_PROBCUT = true; FEAT_LMR = true; FEAT_LMP = true;
        FEAT_FUTILITY = true; FEAT_SEE_PRUNE = true; FEAT_HIST_PRUNE = true;
        FEAT_BAD_NOISY = true; FEAT_EXTENSIONS = true; FEAT_ALPHA_REDUCE = true;
        FEAT_IIR = true; FEAT_HINDSIGHT = true; FEAT_CORRECTION = true;
        FEAT_PVS = true; FEAT_TT_CUTOFF = true; FEAT_TT_NEARMISS = true;
        FEAT_TT_STORE = true; FEAT_QS_CAPTURES = true;
    }
}

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
    // LMR reduction histogram: [0]=reduction 0 (no LMR), [1]=1, ... [7]=7+
    pub lmr_reductions: [u64; 8],
    // new_depth histogram for recursive calls: [0..15]=depth 0-14, [15]=15+
    pub depth_hist: [u64; 16],
    // LMR adjustment tracking
    pub lmr_adj_improving: u64,
    pub lmr_adj_failing: u64,
    pub lmr_adj_pv: u64,
    pub lmr_adj_cut: u64,
    pub lmr_adj_unstable: u64,
    pub lmr_adj_history_neg: u64,
    pub lmr_adj_history_pos: u64,
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
    /// Triangular PV table (matching GoChess)
    pv_table: [[Move; MAX_PLY + 1]; MAX_PLY + 1],
    pv_len: [usize; MAX_PLY + 1],
    prev_moves: [Move; MAX_PLY + 1],
    static_evals: [i32; MAX_PLY + 1],
    /// Excluded move for singular extension verification search (always NoMove when disabled)
    excluded_move: [Move; MAX_PLY + 1],
    /// Pawn history: [pawn_hash % PAWN_HIST_SIZE][piece 1-12][to_square] (GoChess indexing, slot 0 unused)
    pawn_hist: Box<[[[i16; 64]; 13]; PAWN_HIST_SIZE]>,
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
            prev_moves: [NO_MOVE; MAX_PLY + 1],
            static_evals: [0; MAX_PLY + 1],
            excluded_move: [NO_MOVE; MAX_PLY + 1],
            pv_table: [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1],
            pv_len: [0; MAX_PLY + 1],
            pawn_hist: Box::new([[[0i16; 64]; 13]; PAWN_HIST_SIZE]),
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
            let s = evaluate_nnue(board, net, acc);
            // NNUE verification: recompute from scratch and compare
            static VERIFY_INIT: std::sync::Once = std::sync::Once::new();
            static mut VERIFY_ENABLED: bool = false;
            static VERIFY_MISMATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static VERIFY_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            VERIFY_INIT.call_once(|| {
                unsafe { VERIFY_ENABLED = std::env::var("CODA_VERIFY_NNUE").is_ok(); }
            });
            if unsafe { VERIFY_ENABLED } {
                let n = VERIFY_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                acc.force_recompute(net, board);
                let s2 = evaluate_nnue(board, net, acc);
                if s != s2 {
                    let m = VERIFY_MISMATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if m < 20 {
                        eprintln!("NNUE MISMATCH n={} hash={:016x} incremental={} recomputed={} diff={}",
                            n, board.hash, s, s2, s - s2);
                    }
                }
                if n == 9999 {
                    let mm = VERIFY_MISMATCHES.load(std::sync::atomic::Ordering::Relaxed);
                    eprintln!("NNUE verify: {}/{} mismatches after 10000 evals", mm, n + 1);
                }
                s2 // use recomputed value when verifying
            } else {
                s
            }
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
    // Feature flag control via env var (for ablation testing)
    if std::env::var("DISABLE_ALL").is_ok() {
        disable_all_features();
    }

    info.start_time = Instant::now();
    info.stop.store(false, Ordering::Relaxed);
    info.nodes = 0;
    info.sel_depth = 0;

    // Clear all search heuristics (matches GoChess: fresh SearchInfo per go command).
    // Only TT persists across moves. History, killers, counters, correction — all reset.
    info.history.clear();
    info.clear_correction_history();
    info.stats = PruneStats::default();
    // Clear pawn history
    for entry in info.pawn_hist.iter_mut() {
        *entry = [[0i16; 64]; 13];
    }
    // Clear static evals and excluded moves
    info.static_evals = [0; MAX_PLY + 1];
    info.excluded_move = [NO_MOVE; MAX_PLY + 1];
    info.pv_table = [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1];
    info.pv_len = [0; MAX_PLY + 1];
    // Clear TM state
    info.tm_prev_best = NO_MOVE;
    info.tm_prev_score = 0;
    info.tm_best_stable = 0;
    info.tm_has_data = false;

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
        let mut hard = if limits.movestogo > 0 {
            // Tournament TC: cap by moves remaining (generous early, tight late)
            let hard_raw = soft * 2;
            let cap_pct = (20 + limits.movestogo as u64 / 2).min(40);
            let mtg_cap = time_left * cap_pct / 100;
            hard_raw.min(mtg_cap)
        } else {
            // Sudden death: allow up to 3x soft
            soft * 3
        };

        // Absolute hard cap (GoChess: never use more than timeLeft/5 + inc)
        let mut max_hard = time_left / 5 + our_inc;
        if max_hard > time_left * 3 / 4 {
            max_hard = time_left * 3 / 4;
        }
        if hard > max_hard {
            hard = max_hard;
        }

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
        let iter_start = std::time::Instant::now();

        let score;

        // Dump TT after iteration 4 (before depth 5)
        if depth == 5 && std::env::var("DUMP_TT").is_ok() {
            let _ = info.tt.dump_to_file("/tmp/coda_tt_d4.txt");
            eprintln!("Coda TT dumped after d4");
        }

        // Aspiration windows (GoChess: skip for mate scores)
        if depth >= 4 && prev_score > -MATE_SCORE + 100 && prev_score < MATE_SCORE - 100 {
            let mut delta = 15i32;
            let mut alpha = (prev_score - delta).max(-INFINITY);
            let mut beta = (prev_score + delta).min(INFINITY);
            #[allow(unused_assignments)]
            let mut asp_result = prev_score;

            loop {
                // Probe divergent hash at start of each aspiration attempt
                if std::env::var("TRACE_NODES").is_ok() && depth == 5 {
                    let probe = info.tt.probe(0x5cac71485b008015u64);
                    if probe.hit {
                        eprintln!("PRE-D5 asp a={} b={}: TT HIT at 5cac71485b008015 mv={} sc={} d={} f={}",
                            alpha, beta, crate::types::move_to_uci(probe.best_move), probe.score, probe.depth, probe.flag);
                    } else {
                        eprintln!("PRE-D5 asp a={} b={}: TT MISS at 5cac71485b008015", alpha, beta);
                    }
                }
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
            if info.should_stop() { break; }
        } else {
            score = negamax(board, info, -INFINITY, INFINITY, depth, 0, false);
            if info.should_stop() { break; }
        }

        // Get best move from PV table (GoChess: bestMove = info.pvTable[0][0])
        // Fall back to TT probe if PV table is empty
        if info.pv_len[0] > 0 {
            let pv_move = info.pv_table[0][0];
            // Validate against root legal list (ensures correct flags)
            let pv_from = move_from(pv_move);
            let pv_to = move_to(pv_move);
            for i in 0..root_legal.len {
                let m = root_legal.moves[i];
                if move_from(m) == pv_from && move_to(m) == pv_to {
                    best_move = m;
                    break;
                }
            }
        } else {
            // Fallback: probe TT
            let tt_entry = info.tt.probe(board.hash);
            if tt_entry.hit && tt_entry.best_move != NO_MOVE {
                let tt_from = move_from(tt_entry.best_move);
                let tt_to = move_to(tt_entry.best_move);
                for i in 0..root_legal.len {
                    let m = root_legal.moves[i];
                    if move_from(m) == tt_from && move_to(m) == tt_to {
                        best_move = m;
                        break;
                    }
                }
            }
        }

        prev_score = score;

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

        // Extract PV from PV table (GoChess style), extend with TT if short
        let mut pv_str = String::new();
        {
            // Use PV table first
            let pv_len = info.pv_len[0].min(MAX_PLY);
            for i in 0..pv_len {
                if i > 0 { pv_str.push(' '); }
                pv_str.push_str(&move_to_uci(info.pv_table[0][i]));
            }
            // If PV table is short, extend with TT
            if pv_len < depth as usize {
                let mut pv_board = board.clone();
                for i in 0..pv_len {
                    pv_board.make_move(info.pv_table[0][i]);
                }
                let mut pv_moves = pv_len;
                while pv_moves < depth as usize + 5 {
                    let pv_tt = info.tt.probe(pv_board.hash);
                    if !pv_tt.hit || pv_tt.best_move == NO_MOVE { break; }
                    let pv_from = move_from(pv_tt.best_move);
                    let pv_to = move_to(pv_tt.best_move);
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
                    if !pv_str.is_empty() { pv_str.push(' '); }
                    pv_str.push_str(&move_to_uci(found));
                    pv_board.make_move(found);
                    pv_moves += 1;
                }
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

            // Next-iteration estimate: stop if next iteration would exceed hard limit.
            // Use 2x last iteration time as estimate (exponential branching). GoChess lines 685-693.
            if info.hard_limit > 0 {
                let iter_elapsed = iter_start.elapsed().as_millis() as u64;
                if elapsed > 0 && info.hard_limit > elapsed && (info.hard_limit - elapsed) < 2 * iter_elapsed {
                    break;
                }
            }
        }
    }

    best_move
}

/// Negamax alpha-beta search.
/// Line-by-line translation of GoChess's negamax() from search.go (lines 976-1963).
fn negamax(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    mut beta: i32,
    mut depth: i32,
    ply: i32,
    _cut_node: bool, // kept for API compatibility, not used (GoChess doesn't have cut_node)
) -> i32 {
    let ply_u = ply as usize;

    // Guard against stack overflow (GoChess: ply >= MaxPly)
    if ply_u >= MAX_PLY {
        return info.eval(board);
    }

    // Prefetch TT bucket early to hide memory latency
    info.tt.prefetch(board.hash);

    // Clear PV for this node (GoChess: info.pvLen[ply] = 0)
    if ply_u <= MAX_PLY {
        info.pv_len[ply_u] = 0;
    }

    // Track seldepth
    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    // Check time periodically (GoChess: info.Nodes&1023 == 0)
    if info.nodes & 1023 == 0 {
        if info.should_stop() {
            return 0;
        }
    }

    if info.stop.load(Ordering::Relaxed) {
        return 0;
    }

    info.nodes += 1;
    {
        static TRACE_INIT: std::sync::Once = std::sync::Once::new();
        static mut TRACE_NODES: bool = false;
        TRACE_INIT.call_once(|| { unsafe { TRACE_NODES = std::env::var("TRACE_NODES").is_ok(); } });
        if unsafe { TRACE_NODES } && info.nodes <= 5000 {
            eprintln!("NM {} d={} p={} a={} b={} h={:016x}", info.nodes, depth, ply, alpha, beta, board.hash);
        }
    }
    info.stats.depth_hist[depth.clamp(0, 15) as usize] += 1;

    // Draw detection: repetition and 50-move rule (GoChess: ply > 0)
    if ply > 0 {
        if board.halfmove >= 100 {
            return -CONTEMPT;
        }
        // Repetition detection
        let stack_len = board.undo_stack.len();
        let limit = (board.halfmove as usize).min(stack_len);
        let mut i = 2usize;
        while i <= limit {
            if board.undo_stack[stack_len - i].hash == board.hash {
                return -CONTEMPT;
            }
            i += 2;
        }
    }

    // Syzygy WDL probe: skipped (Coda handles Syzygy elsewhere)

    // Probe transposition table
    let mut tt_move = NO_MOVE;
    let alpha_orig = alpha;
    let tt_entry = info.tt.probe(board.hash);
    let tt_hit = tt_entry.hit;

    if tt_hit {
        tt_move = tt_entry.best_move;

        if info.excluded_move[ply_u] == NO_MOVE && ply > 0 {
            let tt_depth = tt_entry.depth;
            let mut tt_score = tt_entry.score;
            // Adjust mate scores for distance from root
            if tt_score > MATE_SCORE - 100 {
                tt_score -= ply;
            } else if tt_score < -(MATE_SCORE - 100) {
                tt_score += ply;
            }

            if tt_depth >= depth && unsafe { FEAT_TT_CUTOFF } {
                match tt_entry.flag {
                    TT_FLAG_EXACT => {
                        // Update PV table with TT move (matching GoChess)
                        if tt_move != NO_MOVE && ply_u <= MAX_PLY {
                            info.pv_table[ply_u][0] = tt_move;
                            info.pv_len[ply_u] = 1;
                        } else if ply_u <= MAX_PLY {
                            info.pv_len[ply_u] = 0;
                        }
                        info.stats.tt_cutoffs += 1;
                        return tt_score;
                    }
                    TT_FLAG_LOWER => {
                        if beta - alpha_orig == 1 && tt_score > alpha {
                            alpha = tt_score;
                        }
                    }
                    TT_FLAG_UPPER => {
                        if beta - alpha_orig == 1 && tt_score < beta {
                            beta = tt_score;
                        }
                    }
                    _ => {}
                }

                if alpha >= beta {
                    if tt_move != NO_MOVE {
                        info.stats.tt_cutoffs += 1;
                        // Update PV table with TT move (matching GoChess)
                        if ply_u <= MAX_PLY {
                            info.pv_table[ply_u][0] = tt_move;
                            info.pv_len[ply_u] = 1;
                        }

                        // History bonus for TT cutoff: reinforce move ordering
                        let bonus = history_bonus(depth);
                        let tt_piece = board.piece_at(move_from(tt_move));
                        let tt_is_cap = board.piece_type_at(move_to(tt_move)) != NO_PIECE_TYPE
                            || move_flags(tt_move) == FLAG_EN_PASSANT;
                        if !tt_is_cap && tt_piece != NO_PIECE {
                            History::update_history(
                                &mut info.history.main[move_from(tt_move) as usize][move_to(tt_move) as usize],
                                bonus,
                            );
                        } else if tt_is_cap && tt_piece != NO_PIECE {
                            let cpt_pt = board.piece_type_at(move_to(tt_move));
                            let ct = if move_flags(tt_move) == FLAG_EN_PASSANT {
                                captured_type(PAWN)
                            } else if cpt_pt != NO_PIECE_TYPE {
                                captured_type(cpt_pt)
                            } else {
                                0 // empty
                            };
                            History::update_cont_history(
                                &mut info.history.capture[go_piece(tt_piece)][move_to(tt_move) as usize][ct],
                                bonus,
                            );
                        }
                    } else if ply_u <= MAX_PLY {
                        info.pv_len[ply_u] = 0;
                    }
                    // TT score dampening: at non-PV nodes with non-mate lower-bound cutoffs,
                    // blend the TT score toward beta to prevent score inflation
                    if beta - alpha_orig == 1
                        && tt_entry.flag == TT_FLAG_LOWER
                        && tt_score > -(MATE_SCORE - 100) && tt_score < MATE_SCORE - 100
                    {
                        return (3 * tt_score + beta) / 4;
                    }
                    return tt_score;
                }
            } else if tt_depth >= depth - 1
                && beta - alpha_orig == 1
                && tt_score > -(MATE_SCORE - 100) && tt_score < MATE_SCORE - 100
                && unsafe { FEAT_TT_NEARMISS }
            {
                // TT near-miss cutoffs: accept entries 1 ply short with a score margin
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
    }

    // Leaf node - go to quiescence search
    if depth <= 0 {
        return quiescence(board, info, alpha, beta, ply);
    }

    // Compute pinned, checkers, in_check
    let pinned = board.pinned();
    let checkers = board.checkers();
    let in_check = checkers != 0;

    // Compute static eval for pruning and LMR improving detection
    let mut static_eval = -INFINITY;
    let mut raw_eval = -INFINITY;
    let mut improving = false;
    let mut failing = false;
    if !in_check {
        if tt_hit && tt_entry.static_eval > -(MATE_SCORE - 100) {
            raw_eval = tt_entry.static_eval;
        } else {
            raw_eval = info.eval(board);
        }
        // Apply correction history
        static_eval = if unsafe { FEAT_CORRECTION } { corrected_eval(info, board, raw_eval) } else { raw_eval };
        if ply_u < MAX_PLY {
            info.static_evals[ply_u] = static_eval;
        }
        // Improving: our eval is better than 2 plies ago
        if ply >= 2 && ply_u >= 2 {
            improving = static_eval > info.static_evals[ply_u - 2];
        }
        // Failing heuristic: detect significant position deterioration
        failing = ply >= 2 && ply_u >= 2
            && info.static_evals[ply_u - 2] > -(MATE_SCORE - 100)
            && static_eval < info.static_evals[ply_u - 2] - (60 + 40 * depth);
    } else {
        if ply_u < MAX_PLY {
            info.static_evals[ply_u] = -INFINITY;
        }
    }

    // Eval instability: detect sharp eval swings from parent node
    let unstable = !in_check && ply >= 1 && ply_u >= 1
        && info.static_evals[ply_u - 1] > -INFINITY
        && {
            let parent_eval = -info.static_evals[ply_u - 1];
            let diff = (static_eval - parent_eval).abs();
            diff > 200
        };

    // Detect if TT move is a capture
    let tt_move_noisy = tt_move != NO_MOVE && {
        board.piece_type_at(move_to(tt_move)) != NO_PIECE_TYPE
            || move_flags(tt_move) == FLAG_EN_PASSANT
    };

    // Internal Iterative Reduction: reduce depth when no TT move exists
    if depth >= 6 && tt_move == NO_MOVE && !in_check && unsafe { FEAT_IIR } {
        depth -= 1;
    }

    // Threat square from null-move failure
    let mut threat_sq: i32 = -1;

    // Hindsight reduction: when both sides think the position is quiet
    if !in_check && ply >= 1 && depth >= 3 && ply_u >= 1
        && info.static_evals[ply_u - 1] > -(MATE_SCORE - 100)
        && static_eval > -INFINITY
        && unsafe { FEAT_HINDSIGHT }
    {
        let eval_sum = info.static_evals[ply_u - 1] + static_eval;
        if eval_sum > 200 {
            depth -= 1;
        }
    }

    // Null-move pruning
    let us = board.side_to_move;
    let stm_non_pawn = board.colors[us as usize]
        & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
    if depth >= 3 && !in_check && ply > 0 && stm_non_pawn != 0
        && beta - alpha == 1 && static_eval >= beta
        && unsafe { FEAT_NMP }
    {
        // Adaptive reduction: scales with depth and eval margin above beta
        let mut r = 3 + depth / 3;
        // Reduce less after captures
        if !board.undo_stack.is_empty() && board.undo_stack.last().unwrap().captured != NO_PIECE_TYPE {
            r -= 1;
        }
        if static_eval > beta {
            let eval_r = ((static_eval - beta) / 200).min(3);
            r += eval_r;
        }
        // Clamp so null-move search is at least depth 1
        if depth - 1 - r < 1 {
            r = depth - 2;
        }

        board.make_null_move();
        let null_key = board.hash; // save hash for threat detection after unmake
        if let Some(acc) = &mut info.nnue_acc { acc.push(DirtyPiece::recompute()); }
        let null_score = -negamax(board, info, -beta, -beta + 1, depth - 1 - r, ply + 1, false);
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        board.unmake_null_move();

        if info.stop.load(Ordering::Relaxed) {
            return 0;
        }

        if null_score >= beta {
            // NMP score dampening: blend toward beta to prevent inflated scores
            let dampened = (null_score * 2 + beta) / 3;

            // Verification search at high depths to guard against zugzwang
            if depth >= 12 {
                info.stats.nmp_verify += 1;
                let v_score = negamax(board, info, beta - 1, beta, depth - 1 - r, ply + 1, false);
                if v_score >= beta {
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
                threat_sq = move_to(threat_entry.best_move) as i32;
            }
        }
    }

    if !in_check {
        // Reverse Futility Pruning (Static Null Move Pruning)
        if depth <= 7 && ply > 0 && unsafe { FEAT_RFP } {
            let margin = if improving { depth * 70 } else { depth * 100 };
            if static_eval - margin >= beta {
                info.stats.rfp_cutoffs += 1;
                return static_eval - margin;
            }
        }

        // Razoring: at shallow depths, if eval is far below alpha, drop to quiescence
        if depth <= 2 && ply > 0 && unsafe { FEAT_RAZORING } {
            let razoring_margin = 400 + depth * 100;
            if static_eval + razoring_margin < alpha {
                let q_score = quiescence(board, info, alpha, beta, ply);
                if q_score < alpha {
                    return q_score;
                }
                info.stats.razor_cutoffs += 1;
            }
        }
    }

    // ProbCut: at moderate+ depths, if a shallow search of captures with
    // raised beta confirms the position is winning, prune the node
    let probcut_beta = beta + 170;
    if !in_check && ply > 0 && depth >= 5 && static_eval + 85 >= probcut_beta && unsafe { FEAT_PROBCUT } {
        let pc_depth = depth - 4;
        let mut pc_picker = QMovePicker::new(board, NO_MOVE, false, &info.history);
        loop {
            let mv = pc_picker.next(board);
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
    let safe_ply = ply_u.min(MAX_PLY - 1).min(63);
    let killers = if safe_ply < 64 {
        info.history.killers[safe_ply]
    } else {
        [NO_MOVE; 2]
    };

    // Counter-move and continuation history lookup from opponent's last move
    let mut counter_move = NO_MOVE;
    let mut prev_piece_go: usize = 0; // GoChess piece index (1-12), 0 = none
    let mut prev_to_for_cont: u8 = 0;
    if !board.undo_stack.is_empty() {
        let undo = &board.undo_stack[board.undo_stack.len() - 1];
        let pm = undo.mv;
        if pm != NO_MOVE {
            let prev_piece = board.piece_at(move_to(pm));
            if prev_piece != NO_PIECE {
                let gp = go_piece(prev_piece);
                counter_move = info.history.counter[gp][move_to(pm) as usize];
                prev_piece_go = gp;
                prev_to_for_cont = move_to(pm);
            }
        }
    }

    // Pawn history pointer for this position's pawn structure
    let ph_idx = (board.pawn_hash as usize) % PAWN_HIST_SIZE;

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

    // Use MovePicker for staged move generation
    let prev_move = if !board.undo_stack.is_empty() {
        board.undo_stack[board.undo_stack.len() - 1].mv
    } else {
        NO_MOVE
    };
    let pawn_hist_ref = Some(&info.pawn_hist[ph_idx] as &[[i16; 64]; 13]);
    let mut picker = if in_check {
        MovePicker::new_evasion(board, tt_move, safe_ply, checkers, pinned, &info.history, prev_move, pawn_hist_ref)
    } else {
        MovePicker::new(board, tt_move, safe_ply, &info.history, prev_move, pawn_hist_ref)
    };
    picker.threat_sq = threat_sq;

    let mut best_move = NO_MOVE;
    let mut best_score = -INFINITY;
    let mut move_count = 0i32;
    let mut alpha_raised_count = 0i32;

    // Track quiet moves searched before beta cutoff for history penalty
    let mut quiets_tried = [NO_MOVE; 64];
    let mut quiets_count = 0usize;

    // Track captures searched before beta cutoff for capture history penalty
    let mut captures_tried: [(u8, u8, u8); 32] = [(0, 0, 0); 32]; // (piece, to, victim)
    let mut n_captures_tried = 0usize;

    loop {
        let mv = picker.next(board);
        if mv == NO_MOVE { break; }

        // Skip excluded move (singular extension verification search)
        if mv == info.excluded_move[ply_u] {
            continue;
        }

        // Legality check: evasion picker returns legal moves, non-evasion needs explicit check
        if !in_check && !board.is_legal(mv, pinned, checkers) {
            continue;
        }

        // Increment move count BEFORE pruning (matches GoChess: moveCount++ at line 1433).
        // Pruned moves still count for LMR/LMP purposes — later moves in the ordering
        // should be reduced more regardless of whether earlier moves were pruned.
        move_count += 1;

        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);

        // Check if capture BEFORE making the move
        let is_cap = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
        let is_promo = is_promotion(mv);

        // SEE capture pruning: at shallow depths, prune captures that lose material
        if is_cap && ply > 0 && !in_check && depth <= 6
            && mv != tt_move && best_score > -(MATE_SCORE - 100)
            && !see_ge(board, mv, -(depth * 100))
            && unsafe { FEAT_SEE_PRUNE }
        {
            continue;
        }

        // SEE quiet pruning: compute SEE before MakeMove (doesn't modify board)
        let mut see_quiet_score = 0i32;
        let mut check_see_quiet = false;
        if ply > 0 && !in_check && depth <= 8
            && !is_cap && !is_promo
            && mv != killers[0] && mv != killers[1]
            && mv != counter_move && mv != tt_move
            && best_score > -(MATE_SCORE - 100)
            && unsafe { FEAT_SEE_PRUNE }
        {
            see_quiet_score = see_after_quiet(board, mv);
            check_see_quiet = true;
        }

        // Singular extension: disabled (SingularExtEnabled = false in GoChess)
        // Code preserved structurally but will never trigger
        let singular_extension = 0i32;
        let _ = singular_extension; // singular extensions currently disabled

        // Save moved piece before MakeMove for consistent history indexing
        let moved_piece = board.piece_at(from);
        let moved_pt = board.piece_type_at(from);
        let captured_pt = if is_cap {
            if flags == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(to) }
        } else {
            NO_PIECE_TYPE
        };

        // History-based pruning: prune quiet moves with deeply negative history at shallow depths
        if ply > 0 && !in_check && !improving && !unstable && depth <= 3
            && !is_cap && !is_promo
            && mv != tt_move
            && mv != killers[0] && mv != killers[1]
            && mv != counter_move
            && best_score > -(MATE_SCORE - 100)
            && unsafe { FEAT_HIST_PRUNE }
        {
            let mut hist_prune_score = info.history.main[from as usize][to as usize];
            if prev_piece_go != 0
                && moved_piece != NO_PIECE
            {
                hist_prune_score += info.history.cont_hist[prev_piece_go][prev_to_for_cont as usize][go_piece(moved_piece)][to as usize] as i32;
            }
            if hist_prune_score < -1500 * depth as i32 {
                info.stats.history_prunes += 1;
                continue;
            }
        }

        // Bad noisy flag: identify losing captures for tighter futility pruning
        let is_bad_noisy = unsafe { FEAT_BAD_NOISY } && is_cap && !in_check && ply > 0 && depth <= 4 && mv != tt_move
            && !is_promo && best_score > -(MATE_SCORE - 100)
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

        // Check if move gives check (opponent is now in check after make_move)
        let gives_check = board.in_check();

        // Bad noisy futility: prune losing captures when eval is far below alpha
        if is_bad_noisy && !gives_check {
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            continue;
        }

        // Futility pruning: use estimated post-LMR depth for tighter margin
        if static_eval > -INFINITY && depth <= 8 && !in_check && !gives_check
            && !is_cap && !is_promo
            && best_score > -(MATE_SCORE - 100)
            && unsafe { FEAT_FUTILITY }
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

        // Late Move Pruning: at shallow depths, skip late quiet moves
        if ply > 0 && !in_check && depth >= 1 && depth <= 8
            && !is_cap && !is_promo && !gives_check
            && best_score > -(MATE_SCORE - 100) && beta - alpha == 1
            && unsafe { FEAT_LMP }
        {
            let mut lmp_limit = 3 + depth * depth;
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

        // SEE quiet pruning: prune quiet moves where piece lands on a losing square
        let mut see_quiet_threshold = -20 * depth * depth;
        if unstable {
            see_quiet_threshold -= 100; // more lenient when position is volatile
        }
        if check_see_quiet && !gives_check && see_quiet_score < see_quiet_threshold {
            info.stats.see_prunes += 1;
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            continue;
        }

        // Recapture extension: extend when recapturing on the same square
        let mut extension = 0;
        if is_cap && board.undo_stack.len() >= 2 {
            let prev_undo = &board.undo_stack[board.undo_stack.len() - 2];
            if prev_undo.captured != NO_PIECE_TYPE && to == move_to(prev_undo.mv) {
                extension = if unsafe { FEAT_EXTENSIONS } { 1 } else { 0 };
                if extension > 0 { info.stats.recapture_ext += 1; }
            }
        }

        let mut new_depth = depth - 1 + extension;

        // Alpha-reduce: after alpha has been raised, reduce subsequent moves by 1 ply
        if alpha_raised_count > 0 && unsafe { FEAT_ALPHA_REDUCE } {
            new_depth -= 1;
        }
        if new_depth < 0 {
            new_depth = 0;
        }

        let score;

        // Track quiet moves for history penalty on beta cutoff
        if !is_cap && !is_promo && quiets_count < 64 {
            quiets_tried[quiets_count] = mv;
            quiets_count += 1;
        }

        // Track captures for capture history penalty on beta cutoff
        // Store GoChess-indexed (go_piece, to, captured_type) for direct use in history penalty
        if is_cap && n_captures_tried < 32 {
            if moved_piece != NO_PIECE && captured_pt != NO_PIECE_TYPE {
                let ct = if flags == FLAG_EN_PASSANT { captured_type(PAWN) } else { captured_type(captured_pt) };
                captures_tried[n_captures_tried] = (go_piece(moved_piece) as u8, to, ct as u8);
                n_captures_tried += 1;
            }
        }

        // Late Move Reductions (LMR) + Principal Variation Search (PVS)
        let is_killer = mv == killers[0] || mv == killers[1];

        let mut reduction = 0i32;
        if !in_check && !is_cap && !is_promo && !is_killer && !gives_check && unsafe { FEAT_LMR } {
            let d = (depth as usize).min(63);
            let m = (move_count as usize).min(63);
            reduction = lmr_reduction(d as i32, m as i32);

            if reduction > 0 {
                // Reduce less at PV nodes where accuracy matters most
                if beta - alpha > 1 {
                    reduction -= 1;
                    info.stats.lmr_adj_pv += 1;
                }

                // Reduce more at expected cut nodes (zero window, not first move)
                if beta - alpha == 1 && move_count > 1 {
                    reduction += 1;
                    info.stats.lmr_adj_cut += 1;
                }

                // Reduce less when the position is improving
                if improving {
                    reduction -= 1;
                    info.stats.lmr_adj_improving += 1;
                }

                // Reduce more when position is deteriorating significantly
                if failing {
                    reduction += 1;
                    info.stats.lmr_adj_failing += 1;
                }

                // Reduce more when multiple moves have already raised alpha
                if alpha_raised_count > 1 {
                    reduction += alpha_raised_count / 2;
                }

                // Reduce less when eval is unstable (sharp swing from parent)
                if unstable {
                    reduction -= 1;
                    info.stats.lmr_adj_unstable += 1;
                }

                // Reduce more when TT move is a capture
                if tt_move_noisy {
                    reduction += 1;
                }

                // Reduce more when opponent has few non-pawn pieces
                // Note: board is post-make_move, so SideToMove is now the opponent
                let opp = flip_color(board.side_to_move);
                let opp_non_pawn = board.colors[opp as usize]
                    & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
                if popcount(opp_non_pawn) < 3 {
                    reduction += 1;
                }

                // Reduce less when moving a piece away from a pawn-attacked square
                if enemy_pawn_attacks & (1u64 << from) != 0 {
                    reduction -= 1;
                }

                // Continuous history adjustment: good history reduces less, bad more
                let mut hist_score = info.history.main[from as usize][to as usize];
                if prev_piece_go != 0
                    && moved_piece != NO_PIECE
                {
                    hist_score += info.history.cont_hist[prev_piece_go][prev_to_for_cont as usize][go_piece(moved_piece)][to as usize] as i32;
                }
                let hist_adj = hist_score / 5000;
                reduction -= hist_adj;
                if hist_adj > 0 { info.stats.lmr_adj_history_neg += 1; } // reduces reduction = good history
                if hist_adj < 0 { info.stats.lmr_adj_history_pos += 1; } // increases reduction = bad history

                // Clamp: never extend (negative), never reduce past depth 1
                if reduction < 0 {
                    reduction = 0;
                }
                if reduction > new_depth - 1 {
                    reduction = new_depth - 1;
                }
            }
        }

        // LMR for captures: use separate capture LMR table with capture history adjustments
        if !in_check && is_cap && !is_promo && !gives_check && move_count > 1 && mv != tt_move && unsafe { FEAT_LMR } {
            // Only reduce at non-PV nodes (zero window search)
            if beta - alpha == 1 {
                let d = (depth as usize).min(63);
                let m = (move_count as usize).min(63);
                reduction = lmr_cap_reduction(d as i32, m as i32);

                if reduction > 0 {
                    // Continuous capture history adjustment
                    if moved_piece != NO_PIECE && captured_pt != NO_PIECE_TYPE {
                        let ct = if flags == FLAG_EN_PASSANT { captured_type(PAWN) } else { captured_type(captured_pt) };
                        let capt_hist_val = info.history.capture[go_piece(moved_piece)][to as usize][ct];
                        // Positive capture history: reduce less
                        if capt_hist_val > 2000 {
                            reduction -= 1;
                        }
                        // Negative capture history: reduce more
                        if capt_hist_val < -2000 {
                            reduction += 1;
                        }
                    }

                    if reduction < 0 {
                        reduction = 0;
                    }
                    // Never reduce past depth 1
                    if reduction > new_depth - 1 {
                        reduction = new_depth - 1;
                    }
                }
            }
        }

        if reduction > 0 {
            info.stats.lmr_searches += 1;
            info.stats.lmr_reductions[reduction.min(7) as usize] += 1;

            // LMR: reduced depth, zero window
            let lmr_depth = new_depth - reduction;
            let mut lmr_score = -negamax(board, info, -alpha - 1, -alpha, lmr_depth, ply + 1, false);

            if lmr_score > alpha && !info.stop.load(Ordering::Relaxed) {
                // LMR failed high: doDeeper/doShallower before re-search
                let mut do_deeper_adj = 0;
                if lmr_score > best_score + 60 + 10 * reduction {
                    do_deeper_adj = 1;
                } else if lmr_score < best_score + new_depth {
                    do_deeper_adj = -1;
                }

                lmr_score = -negamax(board, info, -alpha - 1, -alpha, new_depth + do_deeper_adj, ply + 1, false);
            }

            if lmr_score > alpha && lmr_score < beta && !info.stop.load(Ordering::Relaxed) {
                // PVS failed high: full window re-search
                score = -negamax(board, info, -beta, -alpha, new_depth, ply + 1, false);
            } else {
                score = lmr_score;
            }
        } else if move_count > 1 && unsafe { FEAT_PVS } {
            // PVS: zero-window for non-first moves
            let mut pvs_score = -negamax(board, info, -alpha - 1, -alpha, new_depth, ply + 1, false);
            if pvs_score > alpha && pvs_score < beta && !info.stop.load(Ordering::Relaxed) {
                // Failed high: full window re-search
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

                // Update PV table (triangular) matching GoChess
                if ply_u <= MAX_PLY {
                    info.pv_table[ply_u][0] = mv;
                    let child_len = if ply_u + 1 <= MAX_PLY { info.pv_len[ply_u + 1] } else { 0 };
                    let copy_len = child_len.min(MAX_PLY - ply_u);
                    for i in 0..copy_len {
                        info.pv_table[ply_u][1 + i] = info.pv_table[ply_u + 1][i];
                    }
                    info.pv_len[ply_u] = 1 + child_len;
                }

                if alpha >= beta {
                    info.stats.beta_cutoffs += 1;
                    if move_count == 1 { info.stats.first_move_cutoffs += 1; }

                    // Beta cutoff - update killer moves, history, and counter-move for quiet moves
                    if !is_cap {
                        let bonus = history_bonus(depth);

                        // Store killer
                        if safe_ply < 64 {
                            if info.history.killers[safe_ply][0] != mv {
                                info.history.killers[safe_ply][1] = info.history.killers[safe_ply][0];
                                info.history.killers[safe_ply][0] = mv;
                            }
                        }

                        // Update main history
                        History::update_history(
                            &mut info.history.main[from as usize][to as usize],
                            bonus,
                        );

                        // Update continuation history
                        if prev_piece_go != 0
                            && moved_piece != NO_PIECE
                        {
                            History::update_cont_history(
                                &mut info.history.cont_hist[prev_piece_go][prev_to_for_cont as usize][go_piece(moved_piece)][to as usize],
                                bonus,
                            );
                        }

                        // Update pawn history
                        if moved_piece != NO_PIECE {
                            let gp = go_piece(moved_piece);
                            let v = info.pawn_hist[ph_idx][gp][to as usize] as i32;
                            let clamped = bonus.clamp(-16384, 16384);
                            let new_v = v + clamped - v * clamped.abs() / 16384;
                            info.pawn_hist[ph_idx][gp][to as usize] = new_v.clamp(-32000, 32000) as i16;
                        }

                        // Penalize all quiet moves tried before the cutoff move
                        for i in 0..quiets_count.saturating_sub(1) {
                            let q = quiets_tried[i];
                            let qf = move_from(q);
                            let qt = move_to(q);
                            History::update_history(
                                &mut info.history.main[qf as usize][qt as usize],
                                -bonus,
                            );

                            // Penalize continuation history
                            if prev_piece_go != 0 {
                                let q_piece = board.piece_at(qf);
                                if q_piece != NO_PIECE {
                                    History::update_cont_history(
                                        &mut info.history.cont_hist[prev_piece_go][prev_to_for_cont as usize][go_piece(q_piece)][qt as usize],
                                        -bonus,
                                    );
                                }
                            }

                            // Penalize pawn history
                            {
                                let q_piece = board.piece_at(qf);
                                if q_piece != NO_PIECE {
                                    let gp = go_piece(q_piece);
                                    let v = info.pawn_hist[ph_idx][gp][qt as usize] as i32;
                                    let clamped = (-bonus).clamp(-16384, 16384);
                                    let new_v = v + clamped - v * clamped.abs() / 16384;
                                    info.pawn_hist[ph_idx][gp][qt as usize] = new_v.clamp(-32000, 32000) as i16;
                                }
                            }
                        }

                        // Store counter-move
                        if !board.undo_stack.is_empty() {
                            let undo = &board.undo_stack[board.undo_stack.len() - 1];
                            let pm = undo.mv;
                            if pm != NO_MOVE {
                                let prev_piece = board.piece_at(move_to(pm));
                                if prev_piece != NO_PIECE {
                                    info.history.counter[go_piece(prev_piece)][move_to(pm) as usize] = mv;
                                }
                            }
                        }
                    } else {
                        // Capture caused beta cutoff: update capture history
                        let bonus = history_bonus(depth);
                        if moved_piece != NO_PIECE && captured_pt != NO_PIECE_TYPE {
                            let cpt = if flags == FLAG_EN_PASSANT {
                                captured_type(PAWN)
                            } else {
                                captured_type(captured_pt)
                            };
                            History::update_cont_history(
                                &mut info.history.capture[go_piece(moved_piece)][to as usize][cpt],
                                bonus,
                            );

                            // Penalize captures tried before cutoff
                            for i in 0..n_captures_tried.saturating_sub(1) {
                                let (cp, ct, cv) = captures_tried[i];
                                History::update_cont_history(
                                    &mut info.history.capture[cp as usize][ct as usize][cv as usize],
                                    -bonus,
                                );
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // Check for checkmate or stalemate
    if move_count == 0 {
        if info.excluded_move[ply_u] != NO_MOVE {
            // Singular verification: no alternative found, return alpha
            return alpha;
        }
        if in_check {
            // Checkmate - return negative mate score adjusted for ply
            return -MATE_SCORE + ply;
        }
        // Stalemate
        return 0;
    }

    // Store in transposition table (skip during singular verification)
    if info.excluded_move[ply_u] == NO_MOVE {
        let flag = if best_score <= alpha_orig {
            TT_FLAG_UPPER
        } else if best_score >= beta {
            TT_FLAG_LOWER
        } else {
            TT_FLAG_EXACT
        };

        // Adjust mate score for storage (relative to this position)
        let store_score = score_to_tt(best_score, ply);

        if unsafe { FEAT_TT_STORE } {
            info.tt.store(board.hash, depth, store_score, flag, best_move, raw_eval);
        }
    }

    // Update pawn-hash correction history when we have a reliable score
    if !in_check && best_move != NO_MOVE && depth >= 3
        && info.excluded_move[ply_u] == NO_MOVE
        && best_score > alpha_orig
        && best_score > -(MATE_SCORE - 100) && best_score < MATE_SCORE - 100
        && raw_eval > -(MATE_SCORE - 100)
    {
        update_correction_history(info, board, best_score, raw_eval, depth);
    }

    // Fail-high score blending: dampen inflated cutoff scores at non-PV nodes
    if best_score >= beta && beta - alpha_orig == 1 && depth >= 3
        && best_score > -(MATE_SCORE - 100) && best_score < MATE_SCORE - 100
    {
        return (best_score * depth + beta) / (depth + 1);
    }

    best_score
}

/// History bonus: depth-based bonus for history updates, capped to avoid
/// over-weighting very deep searches. Matches GoChess historyBonus().
fn history_bonus(depth: i32) -> i32 {
    (depth * depth).min(1200)
}

/// Compute SEE score for a quiet move (GoChess: SEEAfterQuiet).
/// This is a placeholder that calls see_ge with the threshold.
/// For a true SEE score, we'd need a full SEE implementation that returns the value.
/// SEE for a quiet move: how much material do we lose if the opponent captures
/// the piece we moved? Returns negative if we lose material (e.g., -320 for knight).
/// Matches GoChess SEEAfterQuiet exactly.
fn see_after_quiet(board: &Board, mv: Move) -> i32 {
    use crate::attacks::*;
    use crate::eval::see_value;

    let from = move_from(mv);
    let to = move_to(mv);
    let pt = board.piece_type_at(from);
    if pt == NO_PIECE_TYPE { return 0; }
    let piece_value = see_value(pt);

    // Our piece moves from 'from' to 'to'
    let mut occ = (board.occupied() & !(1u64 << from)) | (1u64 << to);

    // Opponent tries to capture our piece first
    let us = board.side_to_move;
    let them = flip_color(us);

    let (att_pt, att_sq) = find_lva_for_see(board, to as u32, them, occ);
    if att_pt == NO_PIECE_TYPE {
        return 0; // No attacker, piece is safe
    }

    // Build gain array
    let mut gain = [0i32; 32];
    let mut gain_len = 1usize;
    gain[0] = piece_value; // opponent captures our piece
    let mut next_victim = see_value(att_pt);
    occ ^= 1u64 << att_sq;
    let mut stm = us; // our turn to recapture

    let bishops = board.pieces[BISHOP as usize] | board.pieces[QUEEN as usize];
    let rooks = board.pieces[ROOK as usize] | board.pieces[QUEEN as usize];

    while gain_len < 32 {
        let (lva_pt, lva_sq) = find_lva_for_see(board, to as u32, stm, occ);
        if lva_pt == NO_PIECE_TYPE { break; }

        gain[gain_len] = next_victim - gain[gain_len - 1];
        gain_len += 1;
        next_victim = see_value(lva_pt);
        occ ^= 1u64 << lva_sq;

        // X-ray updates
        if lva_pt == PAWN || lva_pt == BISHOP || lva_pt == QUEEN {
            // Recompute bishop attacks through the hole
        }
        if lva_pt == ROOK || lva_pt == QUEEN {
            // Recompute rook attacks through the hole
        }

        stm = flip_color(stm);
    }

    // Negamax backward
    let mut i = gain_len as i32 - 2;
    while i >= 0 {
        if -gain[i as usize + 1] < gain[i as usize] {
            gain[i as usize] = -gain[i as usize + 1];
        }
        i -= 1;
    }

    -gain[0] // Negate: gain[0] is opponent's result
}

/// Find least valuable attacker of square `sq` by `color` given `occ`.
fn find_lva_for_see(board: &Board, sq: u32, color: u8, occ: u64) -> (u8, u8) {
    use crate::attacks::*;

    let color_bb = board.colors[color as usize] & occ;

    // Pawns
    let pawn_att = if color == WHITE {
        // Black pawns attack downward, so white pawns attacking sq are south of it
        ((1u64 << sq) >> 7) & !0x0101010101010101u64 | ((1u64 << sq) >> 9) & !0x8080808080808080u64
    } else {
        ((1u64 << sq) << 7) & !0x8080808080808080u64 | ((1u64 << sq) << 9) & !0x0101010101010101u64
    };
    let pawns = pawn_att & board.pieces[PAWN as usize] & color_bb;
    if pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        return (PAWN, sq);
    }

    // Knights
    let knights = knight_attacks(sq) & board.pieces[KNIGHT as usize] & color_bb;
    if knights != 0 {
        return (KNIGHT, knights.trailing_zeros() as u8);
    }

    // Bishops
    let bishop_att = bishop_attacks(sq, occ);
    let bishops = bishop_att & board.pieces[BISHOP as usize] & color_bb;
    if bishops != 0 {
        return (BISHOP, bishops.trailing_zeros() as u8);
    }

    // Rooks
    let rook_att = rook_attacks(sq, occ);
    let rooks = rook_att & board.pieces[ROOK as usize] & color_bb;
    if rooks != 0 {
        return (ROOK, rooks.trailing_zeros() as u8);
    }

    // Queens
    let queens = (bishop_att | rook_att) & board.pieces[QUEEN as usize] & color_bb;
    if queens != 0 {
        return (QUEEN, queens.trailing_zeros() as u8);
    }

    // King
    let kings = king_attacks(sq) & board.pieces[KING as usize] & color_bb;
    if kings != 0 {
        return (KING, kings.trailing_zeros() as u8);
    }

    (NO_PIECE_TYPE, 0)
}

/// Helper: get counter-move for the given previous move
fn get_counter_move(history: &History, board: &Board, prev_move: Move) -> Move {
    if prev_move != NO_MOVE {
        let prev_to = move_to(prev_move);
        let prev_piece = board.piece_at(prev_to);
        if prev_piece != NO_PIECE {
            return history.counter[go_piece(prev_piece)][prev_to as usize];
        }
    }
    NO_MOVE
}

/// Quiescence search wrapper.
/// Faithful translation of GoChess's quiescence() from search.go.
fn quiescence(
    board: &mut Board,
    info: &mut SearchInfo,
    alpha: i32,
    beta: i32,
    ply: i32,
) -> i32 {
    quiescence_with_depth(board, info, alpha, beta, ply, 0)
}

/// Quiescence search with depth tracking.
/// Line-by-line translation of GoChess's quiescenceWithDepth() from search.go (lines 1973-2201).
fn quiescence_with_depth(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    beta: i32,
    ply: i32,
    qs_depth: i32,
) -> i32 {
    info.stats.qnodes += 1;

    // Limit quiescence depth to prevent stack overflow
    if qs_depth >= 32 {
        return info.eval(board);
    }

    // Prefetch TT bucket early
    info.tt.prefetch(board.hash);

    info.nodes += 1;

    // QSearch node trace
    static QS_TRACE_INIT: std::sync::Once = std::sync::Once::new();
    static mut QS_TRACE: bool = false;
    QS_TRACE_INIT.call_once(|| { unsafe { QS_TRACE = std::env::var("TRACE_NODES").is_ok(); } });
    if unsafe { QS_TRACE } && info.nodes <= 5000 {
        eprintln!("QS {} p={} a={} b={} h={:016x} qd={}", info.nodes, ply, alpha, beta, board.hash, qs_depth);
    }

    // Track seldepth
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

    // Probe transposition table
    let tt_entry = info.tt.probe(board.hash);
    let tt_move = if tt_entry.hit { tt_entry.best_move } else { NO_MOVE };
    let alpha_orig = alpha;

    // Trace TT probe for divergent hash
    if unsafe { QS_TRACE } && board.hash == 0x5cac71485b008015u64 {
        if tt_entry.hit {
            eprintln!("QS-TT h={:016x} hit mv={} score={} depth={} flag={}",
                board.hash, crate::types::move_to_uci(tt_entry.best_move),
                tt_entry.score, tt_entry.depth, tt_entry.flag);
        } else {
            eprintln!("QS-TT h={:016x} miss", board.hash);
        }
    }
    let tt_hit = tt_entry.hit;

    if tt_hit && tt_entry.depth >= -1 {
        let mut tt_score = tt_entry.score;
        // Adjust mate scores for distance from root
        if tt_score > MATE_SCORE - 100 {
            tt_score -= ply;
        } else if tt_score < -(MATE_SCORE - 100) {
            tt_score += ply;
        }

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

    // Check detection (matches GoChess: PinnedAndCheckers for both check and evasion data)
    let qs_pinned = board.pinned();
    let qs_checkers = board.checkers();
    let qs_in_check = qs_checkers != 0;

    // When in check, generate all evasion moves using main MovePicker
    // (matches GoChess: InitEvasion with full history scoring for quiet evasions)
    if qs_in_check {
        let qs_prev_move = if !board.undo_stack.is_empty() {
            board.undo_stack[board.undo_stack.len() - 1].mv
        } else {
            NO_MOVE
        };
        let qs_ph_idx = if !info.pawn_hist.is_empty() {
            (board.pawn_hash as usize) % info.pawn_hist.len()
        } else {
            0
        };
        let qs_pawn_hist_ref = if !info.pawn_hist.is_empty() {
            Some(&info.pawn_hist[qs_ph_idx] as &[[i16; 64]; 13])
        } else {
            None
        };
        let mut evasion_picker = MovePicker::new_evasion(
            board, tt_move, 0, qs_checkers, qs_pinned, &info.history, qs_prev_move, qs_pawn_hist_ref,
        );
        let mut best_score = -INFINITY;
        let mut best_move = NO_MOVE;
        let mut move_count = 0i32;

        loop {
            let mv = evasion_picker.next(board);
            if mv == NO_MOVE { break; }

            let qs_moved_pt = board.piece_type_at(move_from(mv));
            let qs_captured_pt = if move_flags(mv) == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(move_to(mv)) };
            let qs_dirty = build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), qs_moved_pt, qs_captured_pt);

            if let Some(acc) = &mut info.nnue_acc { acc.push(qs_dirty); }
            if !board.make_move(mv) {
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                continue;
            }
            move_count += 1;

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

        // Store in TT
        let store_score = score_to_tt(best_score, ply);
        let flag = if best_score >= beta {
            TT_FLAG_LOWER
        } else if best_score <= alpha_orig {
            TT_FLAG_UPPER
        } else {
            TT_FLAG_EXACT
        };
        if unsafe { FEAT_TT_STORE } {
            info.tt.store(board.hash, -1, store_score, flag, best_move, -INFINITY);
        }
        return best_score;
    }

    // Stand pat - evaluate the current position (only when not in check)
    // Use TT staticEval when available to avoid recomputing
    let stand_pat = if tt_hit && tt_entry.static_eval > -(MATE_SCORE - 100) {
        tt_entry.static_eval
    } else {
        info.eval(board)
    };
    let mut best_score = stand_pat;

    if best_score >= beta {
        // QS beta blending: dampen stand-pat cutoff at non-PV nodes
        if beta - alpha == 1
            && best_score < MATE_SCORE - 100 && best_score > -(MATE_SCORE - 100)
        {
            return (best_score + beta) / 2;
        }
        return best_score;
    }

    if best_score > alpha {
        alpha = best_score;
    }

    // FEAT_QS_CAPTURES: when disabled, skip the capture loop entirely
    if !unsafe { FEAT_QS_CAPTURES } {
        return best_score;
    }

    // Use main MovePicker in quiescence mode (matching GoChess InitQuiescence).
    // This partitions captures into good (SEE>=0) and bad, and uses staged ordering.
    // Previously used QMovePicker which had no SEE partition — different capture order
    // caused different TT best moves to be stored, seeding all search divergence.
    let mut picker = MovePicker::new_quiescence(board, tt_move, &info.history);
    let mut best_move = NO_MOVE;

    loop {
        let mv = picker.next(board);
        if mv == NO_MOVE { break; }

        // Trace captures at the divergent hash
        if unsafe { QS_TRACE } && board.hash == 0x5cac71485b008015u64 {
            eprintln!("QS-CAP h={:016x} mv={} see={}", board.hash,
                crate::types::move_to_uci(mv), if see_ge(board, mv, 0) { "Y" } else { "N" });
        }

        // Delta pruning: skip captures that can't possibly raise alpha
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

        // Skip bad captures (SEE < 0)
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

    // Store in TT
    let store_score = score_to_tt(best_score, ply);
    let flag = if best_score >= beta {
        TT_FLAG_LOWER
    } else if best_score <= alpha_orig {
        TT_FLAG_UPPER
    } else {
        TT_FLAG_EXACT
    };
    if unsafe { FEAT_TT_STORE } {
        info.tt.store(board.hash, -1, store_score, flag, best_move, stand_pat);
    }

    // QS beta blending: dampen capture fail-high at non-PV nodes
    if best_score >= beta && beta - alpha_orig == 1
        && best_score < MATE_SCORE - 100 && best_score > -(MATE_SCORE - 100)
    {
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
