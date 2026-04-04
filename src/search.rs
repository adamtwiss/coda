/// Negamax alpha-beta search with iterative deepening, PVS, aspiration windows, and Lazy SMP.
/// Features: NMP, RFP, LMR, LMP, futility, SEE pruning, history pruning,
/// singular extensions, cuckoo cycle detection, correction history.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
const CONTEMPT: i32 = 10; // prefer playing on over drawing

// Pawn history table size
const PAWN_HIST_SIZE: usize = 512;

// ============================================================================
// Tunable search parameters (exposed as UCI options for SPSA tuning)
// ============================================================================
use std::sync::atomic::AtomicI32;

macro_rules! tunable {
    ($name:ident, $default:expr, $min:expr, $max:expr) => {
        pub static $name: AtomicI32 = AtomicI32::new($default);
    };
}

// NMP parameters
tunable!(NMP_BASE_R,         4,    2,    8);
tunable!(NMP_DEPTH_DIV,      3,    2,    6);    // R = base + depth/div
tunable!(NMP_EVAL_DIV,     200,  100,  400);    // eval bonus = (eval-beta)/div
tunable!(NMP_EVAL_MAX,       3,    1,    6);    // max eval bonus
tunable!(NMP_VERIFY_DEPTH, 12,    8,   20);    // depth threshold for verification

// RFP parameters
tunable!(RFP_DEPTH,          7,    4,   10);
tunable!(RFP_MARGIN_IMP,    70,   30,  150);    // margin when improving
tunable!(RFP_MARGIN_NOIMP, 100,   50,  200);    // margin when not improving

// Futility parameters
tunable!(FUT_BASE,          90,   20,  200);
tunable!(FUT_PER_DEPTH,    100,   40,  250);

// History pruning
tunable!(HIST_PRUNE_DEPTH,   3,    1,    8);
tunable!(HIST_PRUNE_MULT, 1500,  500, 5000);   // threshold = -mult * depth

// SEE pruning
tunable!(SEE_QUIET_MULT,   20,    5,   50);    // threshold = -mult * depth²
tunable!(SEE_CAP_MULT,    100,   30,  200);    // threshold = -mult * depth

// LMR history divisor
tunable!(LMR_HIST_DIV,   5000, 2000, 15000);

// Singular extensions
tunable!(SE_DEPTH,           8,    4,   12);

// Aspiration windows
tunable!(ASP_DELTA,         13,    5,   30);     // initial delta
tunable!(ASP_SCORE_DIV,  23660, 8000, 50000);   // score-dependent delta divisor

// LMR (C value * 100 for integer representation)
tunable!(LMR_C_QUIET,     130,   80,  300);     // quiet LMR constant (divided by 100)
tunable!(LMR_C_CAP,       180,  100,  350);     // capture LMR constant (divided by 100)

// LMP
tunable!(LMP_BASE,           3,    1,    6);
tunable!(LMP_DEPTH,          8,    4,   12);

// Bad noisy
tunable!(BAD_NOISY_MARGIN,  75,   30,  150);    // depth * margin

// ProbCut
tunable!(PROBCUT_MARGIN,   170,   80,  300);

// Hindsight
tunable!(HINDSIGHT_THRESH, 195,   50,  400);

/// Get a tunable parameter value (inline for hot paths)
#[inline(always)]
fn tp(param: &AtomicI32) -> i32 {
    param.load(Ordering::Relaxed)
}

/// List of all tunable parameters for UCI/SPSA
pub fn tunable_params() -> Vec<(&'static str, &'static AtomicI32, i32, i32, i32)> {
    vec![
        ("NMP_BASE_R",         &NMP_BASE_R,         4,    2,    8),
        ("NMP_DEPTH_DIV",      &NMP_DEPTH_DIV,      3,    2,    6),
        ("NMP_EVAL_DIV",       &NMP_EVAL_DIV,      200, 100,  400),
        ("NMP_EVAL_MAX",       &NMP_EVAL_MAX,        3,    1,    6),
        ("NMP_VERIFY_DEPTH",   &NMP_VERIFY_DEPTH,   12,    8,   20),
        ("RFP_DEPTH",          &RFP_DEPTH,           7,    4,   10),
        ("RFP_MARGIN_IMP",     &RFP_MARGIN_IMP,     70,   30,  150),
        ("RFP_MARGIN_NOIMP",   &RFP_MARGIN_NOIMP,  100,   50,  200),
        ("FUT_BASE",           &FUT_BASE,            90,   20,  200),
        ("FUT_PER_DEPTH",      &FUT_PER_DEPTH,      100,   40,  250),
        ("HIST_PRUNE_DEPTH",   &HIST_PRUNE_DEPTH,    3,    1,    8),
        ("HIST_PRUNE_MULT",    &HIST_PRUNE_MULT,  1500,  500, 5000),
        ("SEE_QUIET_MULT",     &SEE_QUIET_MULT,     20,    5,   50),
        ("SEE_CAP_MULT",       &SEE_CAP_MULT,      100,   30,  200),
        ("LMR_HIST_DIV",       &LMR_HIST_DIV,     5000, 2000, 15000),
        ("SE_DEPTH",           &SE_DEPTH,             8,    4,   12),
        ("ASP_DELTA",          &ASP_DELTA,            13,    5,   30),
        ("ASP_SCORE_DIV",      &ASP_SCORE_DIV,     23660, 8000, 50000),
        ("LMR_C_QUIET",        &LMR_C_QUIET,        130,   80,  300),
        ("LMR_C_CAP",          &LMR_C_CAP,          180,  100,  350),
        ("LMP_BASE",           &LMP_BASE,              3,    1,    6),
        ("LMP_DEPTH",          &LMP_DEPTH,             8,    4,   12),
        ("BAD_NOISY_MARGIN",   &BAD_NOISY_MARGIN,     75,   30,  150),
        ("PROBCUT_MARGIN",     &PROBCUT_MARGIN,       170,   80,  300),
        ("HINDSIGHT_THRESH",   &HINDSIGHT_THRESH,     195,   50,  400),
    ]
}

// Feature flags for ablation testing. All true = normal play.
pub static FEAT_NMP: AtomicBool = AtomicBool::new(true);
pub static FEAT_RFP: AtomicBool = AtomicBool::new(true);
pub static FEAT_PROBCUT: AtomicBool = AtomicBool::new(true); // re-enabled after fixing missing qsearch filter, SEE threshold, and excluded_move guard
pub static FEAT_LMR: AtomicBool = AtomicBool::new(true);
pub static FEAT_LMP: AtomicBool = AtomicBool::new(true);
pub static FEAT_FUTILITY: AtomicBool = AtomicBool::new(true);
pub static FEAT_SEE_PRUNE: AtomicBool = AtomicBool::new(true); // confirmed: -17 Elo without (clean CPU retest)
pub static FEAT_HIST_PRUNE: AtomicBool = AtomicBool::new(true); // confirmed: -17 Elo without (retested without CPU contention)
pub static FEAT_BAD_NOISY: AtomicBool = AtomicBool::new(true); // confirmed: -26 Elo without (retested without CPU contention)
pub static FEAT_EXTENSIONS: AtomicBool = AtomicBool::new(true);
pub static FEAT_ALPHA_REDUCE: AtomicBool = AtomicBool::new(true); // confirmed: -4 Elo without and trending keep (clean CPU retest)
pub static FEAT_IIR: AtomicBool = AtomicBool::new(true);
pub static FEAT_HINDSIGHT: AtomicBool = AtomicBool::new(true); // confirmed: -18 Elo without (clean CPU retest)
pub static FEAT_CORRECTION: AtomicBool = AtomicBool::new(true);
pub static FEAT_PVS: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_CUTOFF: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_NEARMISS: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_STORE: AtomicBool = AtomicBool::new(true);
pub static FEAT_QS_CAPTURES: AtomicBool = AtomicBool::new(true); // false = QS returns eval immediately
pub static FEAT_SINGULAR: AtomicBool = AtomicBool::new(true); // singular extensions specifically
pub static FEAT_CUCKOO: AtomicBool = AtomicBool::new(true);
pub static FEAT_4D_HISTORY: AtomicBool = AtomicBool::new(true); // threat-aware 4D history indexing

/// Disable all features (pure negamax + eval)
pub fn disable_all_features() {
    FEAT_NMP.store(false, Ordering::Relaxed); FEAT_RFP.store(false, Ordering::Relaxed);
    FEAT_PROBCUT.store(false, Ordering::Relaxed); FEAT_LMR.store(false, Ordering::Relaxed); FEAT_LMP.store(false, Ordering::Relaxed);
    FEAT_FUTILITY.store(false, Ordering::Relaxed); FEAT_SEE_PRUNE.store(false, Ordering::Relaxed); FEAT_HIST_PRUNE.store(false, Ordering::Relaxed);
    FEAT_BAD_NOISY.store(false, Ordering::Relaxed); FEAT_EXTENSIONS.store(false, Ordering::Relaxed); FEAT_ALPHA_REDUCE.store(false, Ordering::Relaxed);
    FEAT_IIR.store(false, Ordering::Relaxed); FEAT_HINDSIGHT.store(false, Ordering::Relaxed); FEAT_CORRECTION.store(false, Ordering::Relaxed);
    FEAT_PVS.store(false, Ordering::Relaxed); FEAT_TT_CUTOFF.store(false, Ordering::Relaxed); FEAT_TT_NEARMISS.store(false, Ordering::Relaxed);
    FEAT_TT_STORE.store(false, Ordering::Relaxed); FEAT_QS_CAPTURES.store(false, Ordering::Relaxed);
}

/// Enable all features (normal play)
#[allow(dead_code)]
pub fn enable_all_features() {
    FEAT_NMP.store(true, Ordering::Relaxed); FEAT_RFP.store(true, Ordering::Relaxed); FEAT_PROBCUT.store(true, Ordering::Relaxed);
    FEAT_LMR.store(true, Ordering::Relaxed); FEAT_LMP.store(true, Ordering::Relaxed);
    FEAT_FUTILITY.store(true, Ordering::Relaxed); FEAT_SEE_PRUNE.store(true, Ordering::Relaxed); FEAT_HIST_PRUNE.store(true, Ordering::Relaxed);
    FEAT_BAD_NOISY.store(true, Ordering::Relaxed); FEAT_EXTENSIONS.store(true, Ordering::Relaxed); FEAT_ALPHA_REDUCE.store(true, Ordering::Relaxed);
    FEAT_IIR.store(true, Ordering::Relaxed); FEAT_HINDSIGHT.store(true, Ordering::Relaxed); FEAT_CORRECTION.store(true, Ordering::Relaxed);
    FEAT_PVS.store(true, Ordering::Relaxed); FEAT_TT_CUTOFF.store(true, Ordering::Relaxed); FEAT_TT_NEARMISS.store(true, Ordering::Relaxed);
    FEAT_TT_STORE.store(true, Ordering::Relaxed); FEAT_QS_CAPTURES.store(true, Ordering::Relaxed);
}

// Correction history constants
const CORR_HIST_SIZE: usize = 16384;
const CORR_HIST_GRAIN: i32 = 256;
const CORR_HIST_MAX: i32 = 128;
const CORR_HIST_LIMIT: i32 = 32000;

/// Search limits.
#[derive(Clone)]
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

impl Default for SearchLimits {
    fn default() -> Self { Self::new() }
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
    pub global_nodes: std::sync::Arc<AtomicU64>,  // aggregate nodes across SMP threads
    pub silent: bool,  // suppress UCI output (for datagen)
    pub stats: PruneStats,
    pub tt: std::sync::Arc<TT>,  // shared across Lazy SMP threads
    pub history: Box<History>,
    pub stop: std::sync::Arc<AtomicBool>,  // shared stop flag
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
    /// Ponderhit: shared atomic time limit (ms). 0 = ponder mode (infinite).
    /// Set by UCI thread on ponderhit to switch from infinite to timed search.
    pub ponderhit_time: std::sync::Arc<AtomicU64>,
    pub sel_depth: i32,
    pub last_score: i32,
    /// Triangular PV table
    pv_table: [[Move; MAX_PLY + 1]; MAX_PLY + 1],
    pv_len: [usize; MAX_PLY + 1],
    static_evals: [i32; MAX_PLY + 1],
    /// LMR reduction applied at each ply (for hindsight reduction gating)
    reductions: [i32; MAX_PLY + 1],
    /// Excluded move for singular extension verification search (always NoMove when disabled)
    excluded_move: [Move; MAX_PLY + 1],
    /// Pawn history: [pawn_hash & (PAWN_HIST_SIZE - 1)][piece 1-12][to_square] (slot 0 unused)
    pawn_hist: Box<[[[i16; 64]; 13]; PAWN_HIST_SIZE]>,
    /// Pawn correction history: [stm][pawn_hash % size]
    pawn_corr: Box<[[i32; CORR_HIST_SIZE]; 2]>,
    /// Non-pawn correction history: [stm][color][nonpawn_hash % size]
    np_corr: Box<[[[i32; CORR_HIST_SIZE]; 2]; 2]>,
    /// Minor piece correction history: [stm][minor_hash % size]
    minor_corr: Box<[[i32; CORR_HIST_SIZE]; 2]>,
    /// Major piece correction history: [stm][major_hash % size]
    major_corr: Box<[[i32; CORR_HIST_SIZE]; 2]>,
    /// Continuation correction history: [piece][to_square]
    cont_corr: Box<[[i32; 64]; 12]>,
    pub nnue_net: Option<std::sync::Arc<crate::nnue::NNUENet>>,
    pub nnue_acc: Option<crate::nnue::NNUEAccumulator>,
}

impl SearchInfo {
    pub fn new(tt_mb: usize) -> Self {
        SearchInfo {
            nodes: 0,
            global_nodes: std::sync::Arc::new(AtomicU64::new(0)),
            silent: false,
            stats: PruneStats::default(),
            tt: std::sync::Arc::new(TT::new(tt_mb)),
            history: alloc_zeroed_box(),
            stop: std::sync::Arc::new(AtomicBool::new(false)),
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
            ponderhit_time: std::sync::Arc::new(AtomicU64::new(0)),
            sel_depth: 0,
            last_score: 0,
            static_evals: [0; MAX_PLY + 1],
            reductions: [0; MAX_PLY + 1],
            excluded_move: [NO_MOVE; MAX_PLY + 1],
            pv_table: [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1],
            pv_len: [0; MAX_PLY + 1],
            pawn_hist: alloc_zeroed_box(),
            pawn_corr: alloc_zeroed_box(),
            np_corr: alloc_zeroed_box(),
            minor_corr: alloc_zeroed_box(),
            major_corr: alloc_zeroed_box(),
            cont_corr: alloc_zeroed_box(),
            nnue_net: None,
            nnue_acc: None,
        }
    }

    /// Create a placeholder SearchInfo sharing TT, stop flag, and NNUE net.
    /// Used by UCI loop while the real SearchInfo is in the search thread.
    pub fn new_with_shared(
        stop: std::sync::Arc<AtomicBool>,
        tt: std::sync::Arc<crate::tt::TT>,
        nnue_net: Option<std::sync::Arc<crate::nnue::NNUENet>>,
    ) -> Self {
        let mut si = Self::new(1); // tiny dummy TT, replaced below
        si.stop = stop;
        si.tt = tt;
        si.nnue_net = nnue_net;
        si
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
        // Flush local node count to global counter every 4096 nodes
        // (skip at nodes==0 to avoid phantom 4096 at search start)
        if self.nodes & 4095 == 0 && self.nodes > 0 {
            self.global_nodes.fetch_add(4096, Ordering::Relaxed);
        }
        // Check time every 4096 nodes
        if self.nodes & 4095 == 0 {
            let elapsed = self.start_time.elapsed().as_millis() as u64;
            // Check ponderhit: UCI thread sets this to switch from infinite to timed
            let ph_time = self.ponderhit_time.load(Ordering::Relaxed);
            let effective_limit = if ph_time > 0 { ph_time } else { self.time_limit };
            if effective_limit > 0 && elapsed >= effective_limit {
                self.stop.store(true, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    pub fn clear_correction_history(&mut self) {
        for row in self.pawn_corr.iter_mut() { row.fill(0); }
        for mat in self.np_corr.iter_mut() { for row in mat.iter_mut() { row.fill(0); } }
        for row in self.minor_corr.iter_mut() { row.fill(0); }
        for row in self.major_corr.iter_mut() { row.fill(0); }
        for row in self.cont_corr.iter_mut() { row.fill(0); }
    }

    /// Evaluate using NNUE if loaded, otherwise classical PeSTO.
    fn eval(&mut self, board: &Board) -> i32 {
        let score = if let (Some(net), Some(acc)) = (&self.nnue_net, &mut self.nnue_acc) {
            let s = evaluate_nnue(board, net, acc);
            // NNUE verification: recompute from scratch and compare
            static VERIFY_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            static VERIFY_MISMATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static VERIFY_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if *VERIFY_ENABLED.get_or_init(|| std::env::var("CODA_VERIFY_NNUE").is_ok()) {
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
        score
    }
}

/// Build a DirtyPiece for lazy NNUE accumulator update.
/// `us`/`them` are the sides BEFORE the move.
#[inline]
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

    // King moves: check if bucket+mirror change for the moving side's perspective
    if moved_pt == KING {
        let mut from_ks = from as usize;
        let mut to_ks = to as usize;
        if us == BLACK { from_ks ^= 56; to_ks ^= 56; }

        let from_bucket = crate::nnue::king_bucket_pub(from_ks);
        let to_bucket = crate::nnue::king_bucket_pub(to_ks);
        let from_mirror = crate::nnue::king_mirror_pub(from_ks);
        let to_mirror = crate::nnue::king_mirror_pub(to_ks);

        if from_bucket != to_bucket || from_mirror != to_mirror {
            // Bucket or mirror changed: full recompute needed
            return DirtyPiece::recompute();
        }

        // Same bucket+mirror: only the king feature changes for our perspective.
        // The opponent's perspective is always incremental (their king didn't move).
        // We can treat this as a normal incremental update.
        let mut changes: [(bool, u8, u8, u8); 5] = [(false, 0, 0, 0); 5];
        let mut n = 0;

        // Remove king from origin
        changes[n] = (false, us, KING, from); n += 1;

        // Remove captured piece (king captures)
        if captured_pt != NO_PIECE_TYPE {
            changes[n] = (false, them, captured_pt, to); n += 1;
        }

        // Add king at destination
        changes[n] = (true, us, KING, to); n += 1;

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
        return d;
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
#[inline]
fn corrected_eval(info: &SearchInfo, board: &Board, raw_eval: i32) -> i32 {
    let stm = board.side_to_move as usize;

    // Pawn correction
    let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
    let pawn_corr = info.pawn_corr[stm][pawn_idx] as i64;

    // Non-pawn corrections (per color)
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
    let white_np_corr = info.np_corr[stm][WHITE as usize][white_np_idx] as i64;
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
    let black_np_corr = info.np_corr[stm][BLACK as usize][black_np_idx] as i64;

    // Minor piece correction (knight+bishop hash)
    let minor_idx = (board.minor_key[WHITE as usize] ^ board.minor_key[BLACK as usize]) as usize & (CORR_HIST_SIZE - 1);
    let minor_corr = info.minor_corr[stm][minor_idx] as i64;

    // Major piece correction (rook+queen hash)
    let major_idx = (board.major_key[WHITE as usize] ^ board.major_key[BLACK as usize]) as usize & (CORR_HIST_SIZE - 1);
    let major_corr = info.major_corr[stm][major_idx] as i64;

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

    // Weighted blend: pawn 384, whiteNP 154, blackNP 154, minor 102, major 102, cont 128 = 1024
    let total_corr = (pawn_corr * 384 + white_np_corr * 154 + black_np_corr * 154
        + minor_corr * 102 + major_corr * 102 + cont_corr * 128) / 1024;
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
    let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.pawn_corr[stm][pawn_idx], err, weight);

    // Non-pawn corrections (per color)
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.np_corr[stm][WHITE as usize][white_np_idx], err, weight);
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.np_corr[stm][BLACK as usize][black_np_idx], err, weight);

    // Minor piece correction
    let minor_idx = (board.minor_key[WHITE as usize] ^ board.minor_key[BLACK as usize]) as usize & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.minor_corr[stm][minor_idx], err, weight);

    // Major piece correction
    let major_idx = (board.major_key[WHITE as usize] ^ board.major_key[BLACK as usize]) as usize & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.major_corr[stm][major_idx], err, weight);

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
                // Quiet table: C from tunable (default 130 = 1.30)
                if depth >= 3 && moves >= 3 {
                    let c = tp(&LMR_C_QUIET) as f64 / 100.0;
                    let r = ((depth as f64).ln() * (moves as f64).ln() / c) as i32;
                    LMR_TABLE[depth][moves] = r.min((depth - 2) as i32);
                }
                // Capture table: C from tunable (default 180 = 1.80)
                if depth >= 3 && moves >= 3 {
                    let c = tp(&LMR_C_CAP) as f64 / 100.0;
                    let r = ((depth as f64).ln() * (moves as f64).ln() / c) as i32;
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

/// Initialize feature flags from environment variables (called once at process startup).
/// NO_XXX=1 disables individual features. DISABLE_ALL=1 disables everything,
/// then ENABLE_XXX=1 re-enables individual features.
fn init_feature_flags() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if std::env::var("DISABLE_ALL").is_ok() {
            disable_all_features();
            if std::env::var("ENABLE_NMP").is_ok() { FEAT_NMP.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_RFP").is_ok() { FEAT_RFP.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_PROBCUT").is_ok() { FEAT_PROBCUT.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_LMR").is_ok() { FEAT_LMR.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_LMP").is_ok() { FEAT_LMP.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_FUTILITY").is_ok() { FEAT_FUTILITY.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_SEE_PRUNE").is_ok() { FEAT_SEE_PRUNE.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_HIST_PRUNE").is_ok() { FEAT_HIST_PRUNE.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_BAD_NOISY").is_ok() { FEAT_BAD_NOISY.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_EXTENSIONS").is_ok() { FEAT_EXTENSIONS.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_ALPHA_REDUCE").is_ok() { FEAT_ALPHA_REDUCE.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_IIR").is_ok() { FEAT_IIR.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_HINDSIGHT").is_ok() { FEAT_HINDSIGHT.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_CORRECTION").is_ok() { FEAT_CORRECTION.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_PVS").is_ok() { FEAT_PVS.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_TT_CUTOFF").is_ok() { FEAT_TT_CUTOFF.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_TT_NEARMISS").is_ok() { FEAT_TT_NEARMISS.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_TT_STORE").is_ok() { FEAT_TT_STORE.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_QS_CAPTURES").is_ok() { FEAT_QS_CAPTURES.store(true, Ordering::Relaxed); }
        } else {
            if std::env::var("NO_NMP").is_ok() { FEAT_NMP.store(false, Ordering::Relaxed); }
            if std::env::var("NO_RFP").is_ok() { FEAT_RFP.store(false, Ordering::Relaxed); }
            if std::env::var("NO_PROBCUT").is_ok() { FEAT_PROBCUT.store(false, Ordering::Relaxed); }
            if std::env::var("NO_LMR").is_ok() { FEAT_LMR.store(false, Ordering::Relaxed); }
            if std::env::var("NO_LMP").is_ok() { FEAT_LMP.store(false, Ordering::Relaxed); }
            if std::env::var("NO_FUTILITY").is_ok() { FEAT_FUTILITY.store(false, Ordering::Relaxed); }
            if std::env::var("NO_SEE_PRUNE").is_ok() { FEAT_SEE_PRUNE.store(false, Ordering::Relaxed); }
            if std::env::var("NO_HIST_PRUNE").is_ok() { FEAT_HIST_PRUNE.store(false, Ordering::Relaxed); }
            if std::env::var("NO_BAD_NOISY").is_ok() { FEAT_BAD_NOISY.store(false, Ordering::Relaxed); }
            if std::env::var("NO_EXTENSIONS").is_ok() { FEAT_EXTENSIONS.store(false, Ordering::Relaxed); }
            if std::env::var("NO_ALPHA_REDUCE").is_ok() { FEAT_ALPHA_REDUCE.store(false, Ordering::Relaxed); }
            if std::env::var("NO_IIR").is_ok() { FEAT_IIR.store(false, Ordering::Relaxed); }
            if std::env::var("NO_HINDSIGHT").is_ok() { FEAT_HINDSIGHT.store(false, Ordering::Relaxed); }
            if std::env::var("NO_CORRECTION").is_ok() { FEAT_CORRECTION.store(false, Ordering::Relaxed); }
            if std::env::var("NO_PVS").is_ok() { FEAT_PVS.store(false, Ordering::Relaxed); }
            if std::env::var("NO_TT_CUTOFF").is_ok() { FEAT_TT_CUTOFF.store(false, Ordering::Relaxed); }
            if std::env::var("NO_TT_NEARMISS").is_ok() { FEAT_TT_NEARMISS.store(false, Ordering::Relaxed); }
            if std::env::var("NO_TT_STORE").is_ok() { FEAT_TT_STORE.store(false, Ordering::Relaxed); }
            if std::env::var("NO_QS_CAPTURES").is_ok() { FEAT_QS_CAPTURES.store(false, Ordering::Relaxed); }
            if std::env::var("NO_SINGULAR").is_ok() { FEAT_SINGULAR.store(false, Ordering::Relaxed); }
            if std::env::var("NO_CUCKOO").is_ok() { FEAT_CUCKOO.store(false, Ordering::Relaxed); }
            if std::env::var("NO_4D_HISTORY").is_ok() { FEAT_4D_HISTORY.store(false, Ordering::Relaxed); }
        }
    });
}

/// Allocate a zeroed Box on the heap without stack intermediary.
fn alloc_zeroed_box<T>() -> Box<T> {
    unsafe {
        let layout = std::alloc::Layout::new::<T>();
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() { std::alloc::handle_alloc_error(layout); }
        Box::from_raw(ptr)
    }
}

/// Create a helper SearchInfo that shares TT and stop flag with the main thread.
fn create_helper_info(main: &SearchInfo) -> SearchInfo {
    let mut helper = SearchInfo::new(1); // dummy TT, will be replaced
    helper.tt = main.tt.clone();             // share the same TT
    helper.stop = main.stop.clone();         // share the same stop flag
    helper.global_nodes = main.global_nodes.clone(); // share node counter
    helper.silent = true;                // helpers don't output UCI
    helper.nnue_net = main.nnue_net.clone(); // share NNUE weights (read-only)
    // Create fresh NNUE accumulator for the helper
    if let Some(net) = &helper.nnue_net {
        helper.nnue_acc = Some(crate::nnue::NNUEAccumulator::new(net.hidden_size));
    }
    helper.time_limit = 0; // helpers don't do time management
    helper.move_overhead = main.move_overhead;
    helper
}

/// Run Lazy SMP search: main thread + N-1 helper threads.
pub fn search_smp(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits, threads: usize) -> Move {
    if threads <= 1 {
        return search(board, info, limits);
    }

    // Reset shared state (TT generation is advanced in search(), not here,
    // to avoid double-increment which makes entries appear 2x staler)
    info.stop.store(false, Ordering::Relaxed);
    info.global_nodes.store(0, Ordering::Relaxed); // Reset before helpers start

    // Spawn helper threads
    let mut handles = Vec::new();
    for thread_id in 1..threads {
        let mut helper = create_helper_info(info);
        let mut helper_board = board.clone();
        let helper_limits = SearchLimits {
            depth: limits.depth,
            movetime: limits.movetime,
            wtime: limits.wtime, btime: limits.btime,
            winc: limits.winc, binc: limits.binc,
            movestogo: limits.movestogo,
            nodes: 0, // helpers don't have node limits
            infinite: limits.infinite,
        };

        handles.push(std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                // Helpers search at offset depths for diversity
                helper.start_time = Instant::now();
                // Reset NNUE for this position
                if let Some(acc) = &mut helper.nnue_acc {
                    acc.reset();
                }
                if let (Some(net), Some(acc)) = (&helper.nnue_net, &mut helper.nnue_acc) {
                    acc.materialize(net, &helper_board);
                }
                // Helpers don't do time management — they stop when the main
                // thread sets the shared stop flag. Only main thread controls timing.
                helper.time_limit = 0;
                helper.soft_limit = 0;
                helper.hard_limit = 0;
                helper.max_depth = helper_limits.depth;

                // Search with depth offset for diversity (standard Lazy SMP trick)
                let _mv = search_helper(&mut helper_board, &mut helper, &helper_limits, thread_id);
                helper.nodes // return node count
            }).expect("Failed to spawn SMP helper"));
    }

    // Main thread searches normally
    let best_move = search(board, info, limits);

    // Signal all helpers to stop
    info.stop.store(true, Ordering::Relaxed);

    // Collect helper node counts
    let mut total_nodes = info.nodes;
    for h in handles {
        if let Ok(helper_nodes) = h.join() {
            total_nodes += helper_nodes;
        }
    }
    info.nodes = total_nodes;

    best_move
}

/// Helper thread search — same as main but silent and with depth offset.
fn search_helper(board: &mut Board, info: &mut SearchInfo, _limits: &SearchLimits, thread_id: usize) -> Move {
    init_feature_flags();

    info.history.clear();
    info.clear_correction_history();
    info.stats = PruneStats::default();
    for entry in info.pawn_hist.iter_mut() {
        *entry = [[0i16; 64]; 13];
    }
    info.static_evals = [0; MAX_PLY + 1];
    info.reductions = [0; MAX_PLY + 1];
    info.excluded_move = [NO_MOVE; MAX_PLY + 1];
    info.pv_table = [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1];
    info.pv_len = [0; MAX_PLY + 1];
    info.nodes = 0;

    let root_legal = generate_legal_moves(board);
    let mut best_move = if root_legal.len > 0 { root_legal.moves[0] } else { NO_MOVE };

    let effective_max = info.max_depth.min(MAX_PLY as i32 / 2);
    for depth in 1..=effective_max {
        if info.stop.load(Ordering::Relaxed) { break; }

        // Depth offset for thread diversity: odd threads +1, even threads +0
        let search_depth = depth + (thread_id % 2) as i32;
        if search_depth > effective_max { break; }

        let _score = negamax(board, info, -INFINITY, INFINITY, search_depth, 0, false);
        if info.stop.load(Ordering::Relaxed) { break; }

        if info.pv_len[0] > 0 {
            best_move = info.pv_table[0][0];
        }
    }

    best_move
}

/// Run iterative deepening search.
pub fn search(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits) -> Move {
    init_feature_flags();

    info.start_time = Instant::now();
    info.stop.store(false, Ordering::Relaxed);
    info.nodes = 0;
    info.global_nodes.store(0, Ordering::Relaxed);
    info.sel_depth = 0;

    // Age history tables (×0.80) to preserve useful move ordering from prior searches.
    // Killers and counter-moves are cleared (position-specific). Correction history reset.
    info.history.age(4, 5);
    info.clear_correction_history();
    info.stats = PruneStats::default();
    // Clear pawn history
    for entry in info.pawn_hist.iter_mut() {
        *entry = [[0i16; 64]; 13];
    }
    // Clear static evals and excluded moves
    info.static_evals = [0; MAX_PLY + 1];
    info.reductions = [0; MAX_PLY + 1];
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

        // Cap soft allocation: scale with moves remaining
        // movestogo=1: allow up to 90%, movestogo=2: 70%, sudden death (25): 50%
        let max_pct = if limits.movestogo > 0 {
            (95 - limits.movestogo as u64 * 5).max(30).min(90)
        } else {
            50
        };
        let max_alloc = time_left * max_pct / 100;
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

        // Hard limit
        let mut hard = if limits.movestogo > 0 {
            // Tournament TC: cap based on moves remaining
            // movestogo=1: 90%, movestogo=2: 60%, scales down to 30% min
            let hard_raw = soft * 2;
            let hard_pct = (95 - limits.movestogo as u64 * 10).max(30).min(90);
            let mtg_cap = time_left * hard_pct / 100;
            hard_raw.min(mtg_cap)
        } else {
            // Sudden death: allow up to 3x soft
            soft * 3
        };

        // Absolute hard cap: never use more than timeLeft/5 + inc
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

        // Aspiration windows (skip for mate scores)
        if depth >= 4 && prev_score > -MATE_SCORE + 100 && prev_score < MATE_SCORE - 100 {
            // Eval-dependent aspiration delta: wider for extreme scores (Reckless pattern)
            // Calm positions (avg~0): delta=13, winning (avg~500): delta=24, crushing (avg~1000): delta=55
            let avg = prev_score;
            let mut delta = tp(&ASP_DELTA) + (avg as i64 * avg as i64 / tp(&ASP_SCORE_DIV) as i64) as i32;
            let mut alpha = (prev_score - delta).max(-INFINITY);
            let mut beta = (prev_score + delta).min(INFINITY);
            let mut asp_depth = depth;
            #[allow(unused_assignments)]
            let mut asp_result = prev_score;

            loop {
                let result = negamax(board, info, alpha, beta, asp_depth, 0, false);

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
                    // Reduce depth for re-search (Alexandria/Midnight/Seer pattern)
                    asp_depth = (asp_depth - 1).max(1);
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

        // Get best move from PV table
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
        info.last_score = score;

        // UCI info output
        let elapsed = info.start_time.elapsed().as_millis() as u64;
        let global = info.global_nodes.load(Ordering::Relaxed);
        let nps = if elapsed > 0 { global * 1000 / elapsed } else { 0 };
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

        // Extract PV from PV table, extend with TT if short
        let mut pv_str = String::new();
        {
            // Use PV table first
            let pv_len = info.pv_len[0].min(MAX_PLY);
            for i in 0..pv_len {
                if i > 0 { pv_str.push(' '); }
                pv_str.push_str(&move_to_uci(info.pv_table[0][i]));
            }
            // If PV table is short, extend with TT (detect cycles to avoid looping PVs)
            if pv_len < depth as usize {
                let mut pv_board = board.clone();
                for i in 0..pv_len {
                    pv_board.make_move(info.pv_table[0][i]);
                }
                let mut pv_moves = pv_len;
                let mut seen_hashes = Vec::new();
                while pv_moves < depth as usize + 5 {
                    // Stop at draw conditions: cycle or fifty-move rule
                    if seen_hashes.contains(&pv_board.hash) { break; }
                    if pv_board.halfmove >= 100 { break; }
                    seen_hashes.push(pv_board.hash);

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

        if !info.silent {
            println!(
                "info depth {} seldepth {} {} nodes {} nps {} time {} hashfull {} pv {}",
                depth, info.sel_depth, score_str,
                global, nps, elapsed,
                info.tt.hashfull(), pv_str
            );
        }

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
            // Use 2x last iteration time as estimate (exponential branching)
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
/// Main negamax search with all pruning, extensions, and reductions.
fn negamax(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    mut beta: i32,
    mut depth: i32,
    ply: i32,
    cut_node: bool, // true at expected cut nodes (child of all-node, non-first child of PV)
) -> i32 {
    let ply_u = ply as usize;

    // Guard against stack overflow
    if ply_u >= MAX_PLY {
        return info.eval(board);
    }

    // Prefetch TT bucket early to hide memory latency
    info.tt.prefetch(board.hash);

    // Compute enemy pawn attacks for threat-aware history indexing (cheap: ~3 ops)
    let them_color = flip_color(board.side_to_move);
    let their_pawns = board.pieces[PAWN as usize] & board.colors[them_color as usize];
    let enemy_attacks: u64 = if them_color == WHITE {
        ((their_pawns & !0x0101010101010101u64) << 7) | ((their_pawns & !0x8080808080808080u64) << 9)
    } else {
        ((their_pawns & !0x8080808080808080u64) >> 7) | ((their_pawns & !0x0101010101010101u64) >> 9)
    };

    // Clear PV for this node
    if ply_u <= MAX_PLY {
        info.pv_len[ply_u] = 0;
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

    info.nodes += 1;


    // Draw detection: repetition and 50-move rule
    if ply > 0 {
        if board.halfmove >= 100 {
            return -CONTEMPT;
        }
        // Repetition detection: look back up to halfmove clock entries
        // Note: null moves change hash via side_key XOR, so false matches across
        // null moves are extremely unlikely. No pliesFromNull limit needed here
        // (that limit only applies to cuckoo cycle detection).
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

    // Cuckoo cycle detection: proactive repetition avoidance (Stockfish/Berserk/Viridithas)
    // If we're losing (alpha < 0) and a repetition can be forced, raise alpha to draw score.
    if ply > 0 && alpha < 0 && FEAT_CUCKOO.load(Ordering::Relaxed) && crate::cuckoo::has_game_cycle(board, ply) {
        alpha = 0;
        if alpha >= beta {
            return alpha;
        }
    }

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

            if tt_depth >= depth && FEAT_TT_CUTOFF.load(Ordering::Relaxed) {
                match tt_entry.flag {
                    TT_FLAG_EXACT if beta - alpha == 1 => {
                        // TT exact cutoff: only at non-PV nodes to avoid truncating the PV
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
                        // Update PV table with TT move
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
                                info.history.main_entry(move_from(tt_move), move_to(tt_move), enemy_attacks),
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
                && FEAT_TT_NEARMISS.load(Ordering::Relaxed)
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
        static_eval = if FEAT_CORRECTION.load(Ordering::Relaxed) { corrected_eval(info, board, raw_eval) } else { raw_eval };
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

    // Internal Iterative Reduction: reduce depth when no TT move exists.
    // Restricted to PV/cut nodes (Obsidian/Berserk/Stormphrax pattern).
    // All-nodes have tight bounds already, IIR there wastes depth.
    let is_pv = beta - alpha_orig > 1;
    if depth >= 4 && tt_move == NO_MOVE && !in_check && (is_pv || cut_node) && FEAT_IIR.load(Ordering::Relaxed) {
        depth -= 1;
    }

    // Threat square from null-move failure
    let mut threat_sq: i32 = -1;

    // Hindsight reduction: when parent was LMR-reduced and both sides
    // think the position is quiet, reduce depth further.
    // Gate on prior_reduction (Stockfish >= 2, Alexandria >= 1).
    let prior_reduction = if ply_u >= 1 { info.reductions[ply_u - 1] } else { 0 };
    if !in_check && ply >= 1 && depth >= 2 && ply_u >= 1
        && prior_reduction >= 2
        && info.static_evals[ply_u - 1] > -(MATE_SCORE - 100)
        && static_eval > -INFINITY
        && FEAT_HINDSIGHT.load(Ordering::Relaxed)
    {
        // Both sides optimistic about their position (eval_sum > threshold)
        // correlates with quiet positions where reduction is safe.
        let eval_sum = info.static_evals[ply_u - 1] + static_eval;
        if eval_sum > tp(&HINDSIGHT_THRESH) {
            depth -= 1;
        }
    }

    // Null-move pruning
    let us = board.side_to_move;
    let stm_non_pawn = board.colors[us as usize]
        & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
    if depth >= 3 && !in_check && ply > 0 && stm_non_pawn != 0
        && beta - alpha == 1 && static_eval >= beta
        && info.excluded_move[ply_u] == NO_MOVE  // Skip NMP during SE verification
        && FEAT_NMP.load(Ordering::Relaxed)
    {
        info.stats.nmp_attempts += 1;
        // Adaptive reduction: scales with depth and eval margin above beta
        let mut r = tp(&NMP_BASE_R) + depth / tp(&NMP_DEPTH_DIV);
        // Reduce less after captures
        if !board.undo_stack.is_empty() && board.undo_stack[board.undo_stack.len() - 1].captured != NO_PIECE_TYPE {
            r -= 1;
        }
        if static_eval > beta {
            let eval_r = ((static_eval - beta) / tp(&NMP_EVAL_DIV)).min(tp(&NMP_EVAL_MAX));
            r += eval_r;
        }
        // Clamp so null-move search is at least depth 1
        if depth - r < 1 {
            r = depth - 1;
        }

        board.make_null_move();
        info.tt.prefetch(board.hash);
        let null_key = board.hash; // save hash for threat detection after unmake
        if let Some(acc) = &mut info.nnue_acc { acc.push(DirtyPiece::recompute()); }
        let null_score = -negamax(board, info, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        board.unmake_null_move();

        if info.stop.load(Ordering::Relaxed) {
            return 0;
        }

        if null_score >= beta {
            // NMP score dampening: blend toward beta to prevent inflated scores
            let dampened = (null_score * 2 + beta) / 3;

            // Verification search at high depths to guard against zugzwang
            if depth >= tp(&NMP_VERIFY_DEPTH) {
                info.stats.nmp_verify += 1;
                let v_score = negamax(board, info, beta - 1, beta, depth - r, ply + 1, false);
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
        // RFP TT quiet guard: skip RFP when TT has a quiet best move (Tucano/Weiss).
        // If we know a good quiet move exists, don't prune based on static eval alone.
        let tt_move_is_quiet = tt_move != NO_MOVE
            && board.piece_type_at(move_to(tt_move)) == NO_PIECE_TYPE
            && move_flags(tt_move) != FLAG_EN_PASSANT;
        if depth <= tp(&RFP_DEPTH) && ply > 0 && !is_pv && !tt_move_is_quiet && info.excluded_move[ply_u] == NO_MOVE && FEAT_RFP.load(Ordering::Relaxed) {
            let margin = if improving { depth * tp(&RFP_MARGIN_IMP) } else { depth * tp(&RFP_MARGIN_NOIMP) };
            if static_eval - margin >= beta {
                info.stats.rfp_cutoffs += 1;
                return static_eval - margin;
            }
        }

    }

    // ProbCut: at moderate+ depths, if a shallow search of captures with
    // raised beta confirms the position is winning, prune the node
    let probcut_beta = beta + tp(&PROBCUT_MARGIN);
    if !in_check && ply > 0 && depth >= 5
        && beta.abs() < MATE_SCORE - 100  // skip for mate/TB scores
        && info.excluded_move[ply_u] == NO_MOVE  // skip during SE verification
        && !(tt_hit && tt_entry.depth >= depth - 3 && tt_entry.score < probcut_beta)  // TT says no chance
        && FEAT_PROBCUT.load(Ordering::Relaxed)
    {
        // SEE threshold: only consider captures that gain enough material
        let see_threshold = (probcut_beta - static_eval).max(0);
        let pc_depth = depth - 4;
        let mut pc_picker = QMovePicker::new(board, NO_MOVE, false, &info.history);
        loop {
            let mv = pc_picker.next(board);
            if mv == NO_MOVE { break; }

            if !see_ge(board, mv, see_threshold) { continue; }

            let pc_moved_pt = board.piece_type_at(move_from(mv));
            let pc_captured_pt = if move_flags(mv) == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(move_to(mv)) };
            let pc_dirty = build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), pc_moved_pt, pc_captured_pt);

            if let Some(acc) = &mut info.nnue_acc { acc.push(pc_dirty); }
            if !board.make_move(mv) {
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                continue;
            }
            info.tt.prefetch(board.hash);

            // Cheap qsearch verification before expensive negamax (Stockfish pattern)
            let mut score = -quiescence(board, info, -probcut_beta, -probcut_beta + 1, ply + 1);

            // Only do deeper search if qsearch also beats probcut_beta
            if score >= probcut_beta && pc_depth > 0 {
                score = -negamax(board, info, -probcut_beta, -probcut_beta + 1, pc_depth, ply + 1, !cut_node);
            }

            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }

            if info.stop.load(Ordering::Relaxed) {
                return 0;
            }

            if score >= probcut_beta {
                info.stats.probcut_cutoffs += 1;
                // Store in TT as lower bound so sibling nodes benefit
                info.tt.store(board.hash, depth - 3, score_to_tt(score, ply), TT_FLAG_LOWER, mv, raw_eval);
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
    let mut prev_piece_for_cont: usize = 0; // piece index (1-12), 0 = none
    let mut prev_to_for_cont: u8 = 0;
    if !board.undo_stack.is_empty() {
        let undo = &board.undo_stack[board.undo_stack.len() - 1];
        let pm = undo.mv;
        if pm != NO_MOVE {
            let prev_piece = board.piece_at(move_to(pm));
            if prev_piece != NO_PIECE {
                let gp = go_piece(prev_piece);
                counter_move = info.history.counter[gp][move_to(pm) as usize];
                prev_piece_for_cont = gp;
                prev_to_for_cont = move_to(pm);
            }
        }
    }

    // Pawn history pointer for this position's pawn structure
    let ph_idx = (board.pawn_hash as usize) & (PAWN_HIST_SIZE - 1);

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
        MovePicker::new(board, tt_move, safe_ply, &info.history, prev_move, pawn_hist_ref, enemy_attacks)
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

        // Count before pruning: move ordering position affects LMR/LMP thresholds.
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
            && FEAT_SEE_PRUNE.load(Ordering::Relaxed)
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
            && FEAT_SEE_PRUNE.load(Ordering::Relaxed)
        {
            see_quiet_score = see_after_quiet(board, mv);
            check_see_quiet = true;
        }

        // Singular extension verification search (v7: multi-cut + negative ext, no positive ext)
        // Singular extensions: verify TT move is uniquely best by searching with excluded move.
        // NMP must be gated during singular extension verification search.
        // All components working: positive ext (+1), double ext (+2), multi-cut, negative ext (-1).
        let mut singular_extension = 0i32;
        if mv == tt_move
            && tt_move != NO_MOVE
            && ply > 0
            && depth >= tp(&SE_DEPTH)
            && !in_check
            && info.excluded_move[ply_u] == NO_MOVE
            && tt_hit
            && tt_entry.flag != TT_FLAG_UPPER
            && tt_entry.depth >= depth - 3
            && FEAT_SINGULAR.load(Ordering::Relaxed)
        {
            let tt_score_local = {
                let mut s = tt_entry.score;
                if s > MATE_SCORE - 100 { s -= ply; }
                else if s < -(MATE_SCORE - 100) { s += ply; }
                s
            };

            // Skip SE for mate scores (margin comparison meaningless)
            if tt_score_local > -(MATE_SCORE - 100) && tt_score_local < MATE_SCORE - 100 {
                let singular_beta = tt_score_local - depth;
                let singular_depth = (depth - 1) / 2;

                info.excluded_move[ply_u] = tt_move;
                let singular_score = negamax(board, info, singular_beta - 1, singular_beta, singular_depth, ply, false);
                info.excluded_move[ply_u] = NO_MOVE;

                if info.stop.load(Ordering::Relaxed) {
                    return 0;
                }

                if singular_score >= singular_beta && singular_beta >= beta {
                    // Multi-cut: alternatives are also good enough — prune the whole node
                    return singular_beta;
                }

                if singular_score < singular_beta {
                    // TT move is singular — no competitive alternatives. Extend +1.
                    singular_extension = 1;
                } else {
                    // Alternatives are competitive — negative extension (reduce TT move)
                    singular_extension = -1;
                }
            }
        }

        // Save moved piece before MakeMove for consistent history indexing
        let moved_piece = board.piece_at(from);
        let moved_pt = board.piece_type_at(from);
        let captured_pt = if is_cap {
            if flags == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(to) }
        } else {
            NO_PIECE_TYPE
        };

        // History-based pruning: prune quiet moves with deeply negative history at shallow depths
        if ply > 0 && !in_check && !improving && !unstable && depth <= tp(&HIST_PRUNE_DEPTH)
            && !is_cap && !is_promo
            && mv != tt_move
            && mv != killers[0] && mv != killers[1]
            && mv != counter_move
            && best_score > -(MATE_SCORE - 100)
            && FEAT_HIST_PRUNE.load(Ordering::Relaxed)
        {
            let mut hist_prune_score = info.history.main_score(from, to, enemy_attacks);
            if prev_piece_for_cont != 0
                && moved_piece != NO_PIECE
            {
                hist_prune_score += info.history.cont_hist[prev_piece_for_cont][prev_to_for_cont as usize][go_piece(moved_piece)][to as usize] as i32;
            }
            if hist_prune_score < -tp(&HIST_PRUNE_MULT) * depth as i32 {
                info.stats.history_prunes += 1;
                continue;
            }
        }

        // Bad noisy flag: identify losing captures for tighter futility pruning
        let is_bad_noisy = FEAT_BAD_NOISY.load(Ordering::Relaxed) && is_cap && !in_check && ply > 0 && depth <= 4 && mv != tt_move
            && !is_promo && best_score > -(MATE_SCORE - 100)
            && static_eval > -INFINITY && static_eval + depth * tp(&BAD_NOISY_MARGIN) <= alpha
            && !see_ge(board, mv, 0);

        // Build NNUE dirty piece info BEFORE make_move
        let dirty = build_dirty_piece(mv, us, flip_color(us), moved_pt, captured_pt);

        // Push NNUE accumulator
        if let Some(acc) = &mut info.nnue_acc { acc.push(dirty); }

        if !board.make_move(mv) {
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            continue;
        }

        // Prefetch TT bucket for the new position
        info.tt.prefetch(board.hash);

        // Check if move gives check (opponent is now in check after make_move)
        let gives_check = board.in_check();

        // Bad noisy futility: prune losing captures when eval is far below alpha
        if is_bad_noisy && !gives_check {
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
            continue;
        }

        // Futility pruning: use estimated post-LMR depth for margin
        // Margin widened from 60+60*d to 90+100*d (SF uses 42+120*d, Viridithas 86+70*d)
        // History adjustment: good history widens margin (harder to prune)
        if static_eval > -INFINITY && depth <= 8 && !in_check && !gives_check
            && !is_cap && !is_promo
            && best_score > -(MATE_SCORE - 100)
            && FEAT_FUTILITY.load(Ordering::Relaxed)
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
            let hist_adj = info.history.main_score(from, to, enemy_attacks) / 128;
            if static_eval + tp(&FUT_BASE) + lmr_depth * tp(&FUT_PER_DEPTH) + hist_adj <= alpha {
                info.stats.futility_prunes += 1;
                board.unmake_move();
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                continue;
            }
        }

        // Late Move Pruning: at shallow depths, skip late quiet moves
        if ply > 0 && !in_check && depth >= 1 && depth <= tp(&LMP_DEPTH)
            && !is_cap && !is_promo && !gives_check
            && best_score > -(MATE_SCORE - 100) && beta - alpha == 1
            && FEAT_LMP.load(Ordering::Relaxed)
        {
            let mut lmp_limit = tp(&LMP_BASE) + depth * depth;
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
        let mut see_quiet_threshold = -tp(&SEE_QUIET_MULT) * depth * depth;
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
                extension = if FEAT_EXTENSIONS.load(Ordering::Relaxed) { 1 } else { 0 };
                if extension > 0 { info.stats.recapture_ext += 1; }
            }
        }

        let mut new_depth = depth - 1 + extension + singular_extension;

        // Alpha-reduce: after alpha has been raised, reduce subsequent moves by 1 ply
        if alpha_raised_count > 0 && FEAT_ALPHA_REDUCE.load(Ordering::Relaxed) {
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
        // Store piece/to/captured for history updates after search
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
        if !in_check && !is_cap && !is_promo && !is_killer && FEAT_LMR.load(Ordering::Relaxed) {
            let d = (depth as usize).min(63);
            let m = (move_count as usize).min(63);
            reduction = lmr_reduction(d as i32, m as i32);

            if reduction > 0 {
                // Reduce less at PV nodes where accuracy matters most
                if beta - alpha > 1 {
                    reduction -= 1;
                }

                // Reduce more at expected cut nodes (zero window, not first move)
                if !is_pv && move_count > 1 {
                    reduction += 1;
                }

                // Reduce less when the position is improving
                if improving {
                    reduction -= 1;
                }

                // Reduce more when position is deteriorating significantly
                if failing {
                    reduction += 1;
                }

                // Reduce more when multiple moves have already raised alpha
                if alpha_raised_count > 1 {
                    reduction += alpha_raised_count / 2;
                }

                // Reduce less when eval is unstable (sharp swing from parent)
                if unstable {
                    reduction -= 1;
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
                if enemy_attacks & (1u64 << from) != 0 {
                    reduction -= 1;
                }

                // Reduce less when move gives check (Obsidian/Alexandria/Berserk pattern)
                if gives_check {
                    reduction -= 1;
                }

                // Continuous history adjustment: good history reduces less, bad more
                let mut hist_score = info.history.main_score(from, to, enemy_attacks);
                if prev_piece_for_cont != 0
                    && moved_piece != NO_PIECE
                {
                    hist_score += info.history.cont_hist[prev_piece_for_cont][prev_to_for_cont as usize][go_piece(moved_piece)][to as usize] as i32;
                }
                let hist_adj = hist_score / tp(&LMR_HIST_DIV);
                reduction -= hist_adj;

                // Complexity-aware LMR: reduce less when correction history
                // magnitude is high (uncertain eval → search deeper).
                // Matches Obsidian: R -= complexity / 120.
                if raw_eval > -INFINITY {
                    let complexity = (static_eval - raw_eval).abs();
                    reduction -= complexity / 120;
                }

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
        if !in_check && is_cap && !is_promo && move_count > 1 && mv != tt_move && FEAT_LMR.load(Ordering::Relaxed) {
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

                    // Reduce less for captures that give check
                    if gives_check {
                        reduction -= 1;
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

        // Store reduction for child's hindsight gating
        info.reductions[ply_u] = reduction;

        if reduction > 0 {
            info.stats.lmr_searches += 1;

            // LMR: reduced depth, zero window
            let lmr_depth = new_depth - reduction;
            let mut lmr_score = -negamax(board, info, -alpha - 1, -alpha, lmr_depth, ply + 1, true);

            if lmr_score > alpha && !info.stop.load(Ordering::Relaxed) {
                // LMR failed high: doDeeper/doShallower before re-search
                let mut do_deeper_adj = 0;
                if lmr_score > best_score + 60 + 10 * reduction {
                    do_deeper_adj = 1;
                } else if lmr_score < best_score + new_depth {
                    do_deeper_adj = -1;
                }

                lmr_score = -negamax(board, info, -alpha - 1, -alpha, new_depth + do_deeper_adj, ply + 1, !cut_node);
            }

            if lmr_score > alpha && lmr_score < beta && !info.stop.load(Ordering::Relaxed) {
                // PVS failed high: full window re-search
                score = -negamax(board, info, -beta, -alpha, new_depth, ply + 1, false);
            } else {
                score = lmr_score;
            }
        } else if move_count > 1 && FEAT_PVS.load(Ordering::Relaxed) {
            // PVS: zero-window for non-first moves
            let mut pvs_score = -negamax(board, info, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
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

                // Update triangular PV table
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
                            info.history.main_entry(from, to, enemy_attacks),
                            bonus,
                        );

                        // Update continuation history at plies 1, 2, 4, 6
                        // Ply-1 at full bonus, plies 2/4/6 at half bonus (Obsidian pattern)
                        if moved_piece != NO_PIECE {
                            let gp_mv = go_piece(moved_piece);
                            let stack_len = board.undo_stack.len();
                            let ch_offsets = [1usize, 2, 4, 6];
                            for &off in &ch_offsets {
                                if stack_len >= off {
                                    let undo = &board.undo_stack[stack_len - off];
                                    if undo.mv != NO_MOVE {
                                        let uto = move_to(undo.mv);
                                        let upiece = board.piece_at(uto);
                                        if upiece != NO_PIECE {
                                            let ch_bonus = if off <= 1 { bonus } else { bonus / 2 };
                                            History::update_cont_history(
                                                &mut info.history.cont_hist[go_piece(upiece)][uto as usize][gp_mv][to as usize],
                                                ch_bonus,
                                            );
                                        }
                                    }
                                }
                            }
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
                                info.history.main_entry(qf, qt, enemy_attacks),
                                -bonus,
                            );

                            // Penalize continuation history at plies 1, 2, 4, 6
                            {
                                let q_piece = board.piece_at(qf);
                                if q_piece != NO_PIECE {
                                    let gp_q = go_piece(q_piece);
                                    let stack_len = board.undo_stack.len();
                                    let ch_offsets = [1usize, 2, 4, 6];
                                    for &off in &ch_offsets {
                                        if stack_len >= off {
                                            let undo = &board.undo_stack[stack_len - off];
                                            if undo.mv != NO_MOVE {
                                                let uto = move_to(undo.mv);
                                                let upiece = board.piece_at(uto);
                                                if upiece != NO_PIECE {
                                                    let ch_pen = if off <= 1 { -bonus } else { -bonus / 2 };
                                                    History::update_cont_history(
                                                        &mut info.history.cont_hist[go_piece(upiece)][uto as usize][gp_q][qt as usize],
                                                        ch_pen,
                                                    );
                                                }
                                            }
                                        }
                                    }
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

        if FEAT_TT_STORE.load(Ordering::Relaxed) {
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
/// over-weighting very deep searches.
fn history_bonus(depth: i32) -> i32 {
    (depth * depth).min(1200)
}

/// SEE for a quiet move: how much material do we lose if the opponent captures
/// the piece we moved? Returns negative if we lose material (e.g., -320 for knight).
/// Full SEE with gain array and negamax backward pass.
#[inline]
fn see_after_quiet(board: &Board, mv: Move) -> i32 {
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

    while gain_len < 32 {
        let (lva_pt, lva_sq) = find_lva_for_see(board, to as u32, stm, occ);
        if lva_pt == NO_PIECE_TYPE { break; }

        gain[gain_len] = next_victim - gain[gain_len - 1];
        gain_len += 1;
        next_victim = see_value(lva_pt);
        occ ^= 1u64 << lva_sq;
        // X-ray attacks handled implicitly: find_lva_for_see recomputes
        // slider attacks with updated occ, revealing pieces behind removed ones.

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
#[inline]
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
/// Quiescence search wrapper.
/// Quiescence search wrapper.
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
/// Quiescence search with depth tracking.
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

    // Cuckoo cycle detection in quiescence
    if alpha < 0 && FEAT_CUCKOO.load(Ordering::Relaxed) && crate::cuckoo::has_game_cycle(board, ply) {
        alpha = 0;
        if alpha >= beta {
            return alpha;
        }
    }

    // Probe transposition table
    let tt_entry = info.tt.probe(board.hash);
    let tt_move = if tt_entry.hit { tt_entry.best_move } else { NO_MOVE };
    let alpha_orig = alpha;

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

    // Check detection
    let qs_pinned = board.pinned();
    let qs_checkers = board.checkers();
    let qs_in_check = qs_checkers != 0;

    // When in check, generate all evasion moves using main MovePicker
    // Full history scoring for quiet evasions
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
            info.tt.prefetch(board.hash);
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
        if FEAT_TT_STORE.load(Ordering::Relaxed) {
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

    // TT bound refinement of stand-pat (consensus: every top engine does this)
    // Use TT score as a better estimate when the bound direction agrees
    if tt_hit {
        let tt_score = {
            let mut s = tt_entry.score;
            if s > MATE_SCORE - 100 { s -= ply; }
            else if s < -(MATE_SCORE - 100) { s += ply; }
            s
        };
        if tt_score.abs() < MATE_SCORE - 100 {
            if (tt_entry.flag == TT_FLAG_LOWER && tt_score > best_score)
                || (tt_entry.flag == TT_FLAG_UPPER && tt_score < best_score)
                || tt_entry.flag == TT_FLAG_EXACT
            {
                best_score = tt_score;
            }
        }
    }

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
    if !FEAT_QS_CAPTURES.load(Ordering::Relaxed) {
        return best_score;
    }

    // Use main MovePicker in quiescence mode.
    // This partitions captures into good (SEE>=0) and bad, and uses staged ordering.
    let mut picker = MovePicker::new_quiescence(board, tt_move, &info.history);
    let mut best_move = NO_MOVE;

    loop {
        let mv = picker.next(board);
        if mv == NO_MOVE { break; }

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
        info.tt.prefetch(board.hash);
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
    if FEAT_TT_STORE.load(Ordering::Relaxed) {
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
    bench_inner(depth, nnue_path, true)
}

/// Run bench without printing stats (for multi-threaded bench).
pub fn bench_silent(depth: i32, nnue_path: Option<&str>) -> u64 {
    bench_inner(depth, nnue_path, false)
}

fn bench_inner(depth: i32, nnue_path: Option<&str>, print_stats: bool) -> u64 {
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
    info.silent = !print_stats;
    if let Some(path) = nnue_path {
        if let Err(e) = info.load_nnue(path) {
            eprintln!("Warning: failed to load NNUE: {}", e);
        }
    } else {
        // Auto-discover NNUE net

        // 1. Embedded net (compiled in via CODA_EVALFILE env var during build)
        #[cfg(feature = "embedded-net")]
        {
            static EMBEDDED_NET: &[u8] = include_bytes!(env!("CODA_EVALFILE"));
            let net = crate::nnue::NNUENet::load_from_bytes(EMBEDDED_NET).expect("embedded NNUE corrupt");
            let acc = crate::nnue::NNUEAccumulator::new(net.hidden_size);
            info.nnue_net = Some(std::sync::Arc::new(net));
            info.nnue_acc = Some(acc);
        }

        // 2. net.nnue in exe dir or CWD
        if info.nnue_net.is_none() {
            let try_paths = [
                std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("net.nnue"))),
                Some(std::path::PathBuf::from("net.nnue")),
            ];
            for maybe_path in &try_paths {
                if let Some(path) = maybe_path {
                    if path.exists() {
                        if let Ok(()) = info.load_nnue(path.to_str().unwrap()) {
                            break;
                        }
                    }
                }
            }
        }

        // 3. net.txt discovery (extract filename from URL)
        if info.nnue_net.is_none() {
            let try_paths = [
                std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("net.txt"))),
                Some(std::path::PathBuf::from("net.txt")),
            ];
            for maybe_path in &try_paths {
                if let Some(path) = maybe_path {
                    if path.exists() {
                        if let Ok(contents) = std::fs::read_to_string(path) {
                            let url = contents.trim();
                            if let Some(fname) = url.rsplit('/').next() {
                                let net_dir = path.parent().unwrap_or(std::path::Path::new("."));
                                let net_path = net_dir.join(fname);
                                if net_path.exists() {
                                    if let Ok(()) = info.load_nnue(net_path.to_str().unwrap()) {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
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

    if !print_stats { return total_nodes; }

    // Print pruning stats
    let s = &info.stats;
    eprintln!("=== Pruning Stats (cumulative across all bench positions) ===");
    eprintln!("TT cutoffs:     {:>8}  ({:.1}% of nodes)", s.tt_cutoffs, s.tt_cutoffs as f64 / total_nodes as f64 * 100.0);
    eprintln!("TT near-miss:   {:>8}", s.tt_near_miss);
    eprintln!("NMP attempts:   {:>8}  cutoffs: {} ({:.0}%)", s.nmp_attempts, s.nmp_cutoffs,
        if s.nmp_attempts > 0 { s.nmp_cutoffs as f64 / s.nmp_attempts as f64 * 100.0 } else { 0.0 });
    eprintln!("RFP cutoffs:    {:>8}  ({:.1}% of nodes)", s.rfp_cutoffs, s.rfp_cutoffs as f64 / total_nodes as f64 * 100.0);
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
