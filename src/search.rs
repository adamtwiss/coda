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
// CONTEMPT is now tunable as CONTEMPT_VAL

// Pawn history table size
const PAWN_HIST_SIZE: usize = 512;

// ============================================================================
// Tunable search parameters (exposed as UCI options for SPSA tuning)
// ============================================================================
use std::sync::atomic::AtomicI32;

/// Declare a tunable search parameter with default, min, max.
/// Single source of truth — used for both the static AtomicI32 and the UCI/SPSA parameter list.
macro_rules! tunables {
    ( $( ($name:ident, $default:expr, $min:expr, $max:expr) ),* $(,)? ) => {
        // Declare each as a pub static AtomicI32
        $( pub static $name: AtomicI32 = AtomicI32::new($default); )*

        /// List of all tunable parameters for UCI/SPSA
        pub fn tunable_params() -> Vec<(&'static str, &'static AtomicI32, i32, i32, i32)> {
            vec![
                $( (stringify!($name), &$name, $default, $min, $max), )*
            ]
        }
    };
}

tunables!(
    // v9 post-#489 retune (feature/threat-inputs; post-merge of 2b rewrite,
    // ProbCut gate, LMR king-pressure, futility-defenses; landed +7.38 H1).
    (NMP_BASE_R, 5, 2, 8),
    (NMP_DEPTH_DIV, 3, 1, 6),
    (NMP_EVAL_DIV, 122, 100, 400),
    (NMP_EVAL_MAX, 5, 1, 6),
    (NMP_VERIFY_DEPTH, 12, 8, 20),
    // RFP
    (RFP_DEPTH, 7, 2, 12),
    (RFP_MARGIN_IMP, 82, 30, 150),
    (RFP_MARGIN_NOIMP, 128, 50, 200),
    // Futility
    (FUT_BASE, 69, 20, 200),
    (FUT_PER_DEPTH, 163, 40, 250),
    // History pruning
    (HIST_PRUNE_DEPTH, 4, 1, 8),
    (HIST_PRUNE_MULT, 5148, 500, 50000),
    // SEE pruning
    (SEE_QUIET_MULT, 45, 5, 80),
    (SEE_CAP_MULT, 146, 30, 200),
    // LMR
    (LMR_HIST_DIV, 7123, 2000, 100000),
    (LMR_C_QUIET, 124, 40, 300),
    (LMR_C_CAP, 101, 100, 350),
    // Singular extensions
    (SE_DEPTH, 5, 4, 20),
    // Aspiration windows
    (ASP_DELTA, 12, 5, 30),
    (ASP_SCORE_DIV, 33333, 8000, 50000),
    // LMP
    (LMP_BASE, 13, 1, 15),
    (LMP_DEPTH, 9, 4, 20),
    // Bad noisy
    (BAD_NOISY_MARGIN, 125, 30, 150),
    // ProbCut
    (PROBCUT_MARGIN, 193, 80, 300),
    // Hindsight
    (HINDSIGHT_THRESH, 181, 50, 400),
    // Unstable position detection
    (UNSTABLE_THRESH, 155, 50, 500),
    // SEE piece value scaling
    (SEE_MATERIAL_SCALE, 191, 30, 300),
    // QS
    (QS_DELTA_MARGIN, 358, 100, 500),
    (QS_SEE_THRESHOLD, -35, -200, 0),
    (QS_MAX_CAPTURES, 27, 2, 32),
    // Correction history weights
    (CORR_W_PAWN, 301, 100, 600),
    (CORR_W_NP, 106, 50, 400),
    (CORR_W_MINOR, 57, 30, 300),
    (CORR_W_MAJOR, 92, 30, 300),
    (CORR_W_CONT, 39, 30, 400),
    // Fail-high blend
    (FH_BLEND_DEPTH, 1, 0, 8),
    // History bonus
    (HIST_BONUS_MULT, 300, 50, 400),
    (HIST_BONUS_MAX, 1584, 500, 3000),
    // Capture history bonus
    (CAP_HIST_MULT, 263, 50, 400),
    (CAP_HIST_BASE, 15, 0, 200),
    (CAP_HIST_MAX, 1635, 500, 3000),
    // Double extensions
    (DEXT_MARGIN, 10, 2, 50),
    (DEXT_CAP, 18, 4, 32),
    // Quiet check bonus
    (QUIET_CHECK_BONUS, 8270, 2000, 30000),
    // LMR complexity
    (LMR_COMPLEXITY_DIV, 184, 30, 500),
    // Contempt
    (CONTEMPT_VAL, 19, 0, 50),
    // Correction history divisor
    (CORR_HIST_DIV, 1263, 256, 4096),
    // Correction history update weight cap.
    (CORR_UPDATE_WEIGHT_MAX, 17, 4, 48),
    (CORR_BONUS_CAP_DIV, 4, 1, 16),
    (CORR_HIST_GRAIN_T, 9, 1, 32),
    (CORR_HIST_ERR_MAX, 4, 1, 64),
    // Escape-capture bonuses (Reckless pattern): move ordering bonus for
    // moving a piece off a square attacked by enemy pawns
    (ESCAPE_BONUS_Q, 15627, 5000, 40000),
    (ESCAPE_BONUS_R, 13736, 3000, 30000),
    (ESCAPE_BONUS_MINOR, 10172, 2000, 20000),
    // v9 threat-family gates/modifiers (v9-specific — require threat-aware net).
    (NMP_KING_ZONE_MAX, 5, 2, 9),
    (PROBCUT_KING_ZONE_MAX, 5, 2, 9),
    (LMR_THREAT_DIV, 2, 1, 5),
    (LMR_KING_PRESSURE_DIV, 4, 2, 9),
    (FUT_THREATS_MARGIN, 40, 0, 200),
    // B1: Discovered-attack movepicker bonus (+52 Elo H1, #502). Flat
    // bonus added to quiet move score when `move.from()` is one of our
    // pieces currently blocking our own slider's attack on an enemy.
    // Moving it creates a discovered attack. Uses Board::xray_blockers.
    (DISCOVERED_ATTACK_BONUS, 8000, 0, 30000),
    // S5: king-zone-pressure aspiration widener. Adds king_zone_pressure
    // * ASP_KING_PRESSURE_MARGIN to the aspiration delta at root,
    // anticipating wider score swings in tactical king positions.
    // 0 = disabled.
    (ASP_KING_PRESSURE_MARGIN, 3, 0, 20),
    // MVV multiplier + cont-hist plies-1/2 weight.
    (MVV_CAP_MULT, 15, 4, 64),
    (CONT_HIST_MULT, 3, 1, 8),
);

/// Get a tunable parameter value (inline for hot paths)
#[inline(always)]
fn tp(param: &AtomicI32) -> i32 {
    param.load(Ordering::Relaxed)
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
    FEAT_SINGULAR.store(false, Ordering::Relaxed); FEAT_CUCKOO.store(false, Ordering::Relaxed);
    FEAT_4D_HISTORY.store(false, Ordering::Relaxed);
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
    FEAT_SINGULAR.store(true, Ordering::Relaxed); FEAT_CUCKOO.store(true, Ordering::Relaxed);
    FEAT_4D_HISTORY.store(true, Ordering::Relaxed);
}

// Correction history constants
const CORR_HIST_SIZE: usize = 16384;
const CORR_HIST_GRAIN: i32 = 8;       // Scaled with LIMIT: 256/32000 ≈ 8/1024
const CORR_HIST_MAX: i32 = 4;         // Scaled: 128/32000 ≈ 4/1024
const CORR_HIST_LIMIT: i32 = 1024;    // Consensus (SF, Viridithas, Obsidian)

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
    // Move ordering quality: sum of move_count² at beta cutoff (lower = better ordering)
    pub cutoff_movecount_sq_sum: u64,
    pub cutoff_movecount_sum: u64,
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
    /// Minimum think time per move: the increment we're about to gain, minus
    /// move overhead. Floors the dynamically-scaled soft limit so stability
    /// cuts in stable endgames can't push think time below the increment,
    /// which would grow the clock instead of spending it (stockpile). 0 when
    /// there is no increment.
    soft_floor: u64,
    /// Per-root-move node counts for node-based time management.
    /// Indexed by from_sq * 64 + to_sq. Reset each search.
    root_move_nodes: Box<[u64; 4096]>,
    /// Ponderhit: shared atomic time limit (ms). 0 = ponder mode (infinite).
    /// Set by UCI thread on ponderhit to switch from infinite to timed search.
    pub ponderhit_time: std::sync::Arc<AtomicU64>,
    /// Completed search depth (shared atomic). Updated by search thread after
    /// each completed iteration. Read by UCI thread on ponderhit to scale budget.
    pub ponder_depth: std::sync::Arc<AtomicU64>,
    pub sel_depth: i32,
    pub last_score: i32,
    /// Root side-to-move (for contempt: penalize draws from our perspective)
    pub root_stm: u8,
    /// Per-depth cumulative node counts (for EBF calculation in bench)
    pub depth_nodes: [u64; MAX_PLY + 1],
    pub completed_depth: i32,
    /// Triangular PV table
    pub pv_table: [[Move; MAX_PLY + 1]; MAX_PLY + 1],
    pub pv_len: [usize; MAX_PLY + 1],
    static_evals: [i32; MAX_PLY + 1],
    /// LMR reduction applied at each ply (for hindsight reduction gating)
    reductions: [i32; MAX_PLY + 1],
    /// Excluded move for singular extension verification search (always NoMove when disabled)
    pub excluded_move: [Move; MAX_PLY + 1],
    /// Double extension counter — propagated from parent, capped to prevent search explosion
    double_ext_count: [i32; MAX_PLY + 1],
    /// Per-ply moved piece (go_piece index 1-12, 0=none). Set before make_move.
    /// Used for correct cont hist lookups at ply-2+ (avoids stale board.piece_at).
    moved_piece_stack: [u8; MAX_PLY + 1],
    /// Per-ply move destination square. Used alongside moved_piece_stack.
    moved_to_stack: [u8; MAX_PLY + 1],
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
    pub threat_stack: crate::threat_accum::ThreatStack,
    /// Syzygy tablebases (shared, read-only). Interior WDL probes in search.
    pub syzygy: Option<std::sync::Arc<crate::tb::SyzygyTB>>,
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
            soft_floor: 0,
            root_move_nodes: alloc_zeroed_box(),
            ponderhit_time: std::sync::Arc::new(AtomicU64::new(0)),
            ponder_depth: std::sync::Arc::new(AtomicU64::new(0)),
            sel_depth: 0,
            last_score: 0,
            root_stm: WHITE,
            depth_nodes: [0; MAX_PLY + 1],
            completed_depth: 0,
            static_evals: [0; MAX_PLY + 1],
            reductions: [0; MAX_PLY + 1],
            excluded_move: [NO_MOVE; MAX_PLY + 1],
            double_ext_count: [0; MAX_PLY + 1],
            moved_piece_stack: [0; MAX_PLY + 1],
            moved_to_stack: [0; MAX_PLY + 1],
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
            threat_stack: crate::threat_accum::ThreatStack::new(768), // max v9 accum size
            syzygy: None,
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
        // Activate threat stack if net has threat features
        if net.has_threats {
            self.threat_stack = crate::threat_accum::ThreatStack::new(net.hidden_size);
            self.threat_stack.active = true;
        }
        self.nnue_net = Some(std::sync::Arc::new(net));
        self.nnue_acc = Some(acc);
        Ok(())
    }

    /// Auto-discover NNUE net. Single source of truth for all code paths (UCI, bench, etc).
    /// Priority: embedded (fat binary) > net.nnue on disk > net.txt filename discovery.
    /// Returns true if a net was loaded.
    pub fn auto_discover_nnue(&mut self) -> bool {
        // 1. Embedded net (compiled in via CODA_EVALFILE env var during build)
        #[cfg(feature = "embedded-net")]
        {
            static EMBEDDED_NET: &[u8] = include_bytes!(env!("CODA_EVALFILE"));
            match crate::nnue::NNUENet::load_from_bytes(EMBEDDED_NET) {
                Ok(net) => {
                    let acc = crate::nnue::NNUEAccumulator::new(net.hidden_size);
                    if net.has_threats {
                        self.threat_stack = crate::threat_accum::ThreatStack::new(net.hidden_size);
                        self.threat_stack.active = true;
                    }
                    self.nnue_net = Some(std::sync::Arc::new(net));
                    self.nnue_acc = Some(acc);
                    return true;
                }
                Err(e) => {
                    eprintln!("WARNING: embedded NNUE corrupt: {}", e);
                }
            }
        }

        // 2. net.nnue in exe dir or CWD
        let net_nnue_paths = [
            std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("net.nnue"))),
            Some(std::path::PathBuf::from("net.nnue")),
        ];
        for maybe_path in &net_nnue_paths {
            if let Some(path) = maybe_path {
                if path.exists() {
                    if let Ok(()) = self.load_nnue(path.to_str().unwrap()) {
                        return true;
                    }
                }
            }
        }

        // 3. net.txt discovery (extract filename from URL)
        let net_txt_paths = [
            std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("net.txt"))),
            Some(std::path::PathBuf::from("net.txt")),
        ];
        for maybe_path in &net_txt_paths {
            if let Some(path) = maybe_path {
                if path.exists() {
                    if let Ok(contents) = std::fs::read_to_string(path) {
                        let url = contents.trim();
                        if let Some(fname) = url.rsplit('/').next() {
                            let net_dir = path.parent().unwrap_or(std::path::Path::new("."));
                            let net_path = net_dir.join(fname);
                            if net_path.exists() {
                                if let Ok(()) = self.load_nnue(net_path.to_str().unwrap()) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }

        false
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
            // For ponderhit: allow a grace period beyond the deadline so the
            // current iteration can finish cleanly. But hard-stop if the grace
            // period expires to prevent time loss. The ID loop also checks the
            // deadline (without grace) between iterations to prevent starting
            // new iterations after the budget expires.
            let ph_time = self.ponderhit_time.load(Ordering::Relaxed);
            let effective_limit = if ph_time > 0 {
                // Grace period scales with remaining budget: enough to finish
                // an iteration but not enough to risk flagging. Caps at 500ms
                // and shrinks to near-zero when budget is almost used.
                let remaining = if ph_time > elapsed { ph_time - elapsed } else { 0 };
                let grace = (remaining / 4).min(500);
                ph_time + grace
            } else {
                self.time_limit
            };
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

    pub fn clear_pawn_hist(&mut self) {
        for entry in self.pawn_hist.iter_mut() {
            *entry = [[0i16; 64]; 13];
        }
    }

    /// Evaluate using NNUE if loaded, otherwise classical PeSTO.
    fn eval(&mut self, board: &Board) -> i32 {
        // Ensure threat accumulator is computed before eval
        if self.threat_stack.active {
            if let Some(ref net) = self.nnue_net {
                self.threat_stack.ensure_computed(&net.threat_weights, net.num_threat_features, board);
            }
        }
        let score = if let (Some(net), Some(acc)) = (&self.nnue_net, &mut self.nnue_acc) {
            let s = evaluate_nnue(board, net, acc, &self.threat_stack);
            // NNUE verification: recompute from scratch and compare
            static VERIFY_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            static VERIFY_MISMATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static VERIFY_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if *VERIFY_ENABLED.get_or_init(|| std::env::var("CODA_VERIFY_NNUE").is_ok()) {
                let n = VERIFY_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                acc.force_recompute(net, board);
                let s2 = evaluate_nnue(board, net, acc, &self.threat_stack);
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
        // Material scaling: dampen eval in low-material endgames (Alexandria pattern).
        // NNUE tends to overestimate advantages when fewer pieces remain.
        // P=100, N=422, B=422, R=642, Q=1015
        let material = {
            let pawns = popcount(board.pieces[PAWN as usize]) as i32 * 100;
            let knights = popcount(board.pieces[KNIGHT as usize]) as i32 * 422;
            let bishops = popcount(board.pieces[BISHOP as usize]) as i32 * 422;
            let rooks = popcount(board.pieces[ROOK as usize]) as i32 * 642;
            let queens = popcount(board.pieces[QUEEN as usize]) as i32 * 1015;
            pawns + knights + bishops + rooks + queens
        };
        let score = score * (22400 + material) / 32 / 1024;

        // 50-move eval scaling: decay eval toward zero as halfmove clock advances.
        let hm = board.halfmove.min(200) as i32;
        score * (200 - hm) / 200
    }
}

/// Build a DirtyPiece for lazy NNUE accumulator update.
/// `us`/`them` are the sides BEFORE the move.
#[inline]
pub fn build_dirty_piece(
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
    let total_corr = (pawn_corr * tp(&CORR_W_PAWN) as i64 + white_np_corr * tp(&CORR_W_NP) as i64 + black_np_corr * tp(&CORR_W_NP) as i64
        + minor_corr * tp(&CORR_W_MINOR) as i64 + major_corr * tp(&CORR_W_MAJOR) as i64 + cont_corr * tp(&CORR_W_CONT) as i64) / tp(&CORR_HIST_DIV) as i64;
    let adjusted = raw_eval + (total_corr as i32) / tp(&CORR_HIST_GRAIN_T);
    adjusted.clamp(-MATE_SCORE + 100, MATE_SCORE - 100)
}

/// Update correction history entry with gravity.
fn update_corr_entry(entry: &mut i32, err: i32, weight: i32, cap_div: i32) {
    // Proportional gravity (consensus: every top engine uses this)
    // Self-limiting: values near the limit get pulled back harder
    let cap = CORR_HIST_LIMIT / cap_div.max(1);
    let bonus = (err * weight).clamp(-cap, cap);
    *entry += bonus - *entry * bonus.abs() / CORR_HIST_LIMIT;
    *entry = (*entry).clamp(-CORR_HIST_LIMIT, CORR_HIST_LIMIT);
}

/// Update all correction history tables.
fn update_correction_history(info: &mut SearchInfo, board: &Board, search_score: i32, raw_eval: i32, depth: i32) {
    let err_max = tp(&CORR_HIST_ERR_MAX);
    let err = (search_score - raw_eval).clamp(-err_max, err_max);
    let weight = (depth + 1).min(tp(&CORR_UPDATE_WEIGHT_MAX));
    let cap_div = tp(&CORR_BONUS_CAP_DIV);
    let stm = board.side_to_move as usize;

    // Pawn correction
    let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.pawn_corr[stm][pawn_idx], err, weight, cap_div);

    // Non-pawn corrections (per color)
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.np_corr[stm][WHITE as usize][white_np_idx], err, weight, cap_div);
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.np_corr[stm][BLACK as usize][black_np_idx], err, weight, cap_div);

    // Minor piece correction
    let minor_idx = (board.minor_key[WHITE as usize] ^ board.minor_key[BLACK as usize]) as usize & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.minor_corr[stm][minor_idx], err, weight, cap_div);

    // Major piece correction
    let major_idx = (board.major_key[WHITE as usize] ^ board.major_key[BLACK as usize]) as usize & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.major_corr[stm][major_idx], err, weight, cap_div);

    // Continuation correction
    if !board.undo_stack.is_empty() {
        let last = &board.undo_stack[board.undo_stack.len() - 1];
        if last.mv != NO_MOVE {
            let to = move_to(last.mv);
            let pt = board.piece_type_at(to);
            if pt < 6 {
                let piece = make_piece(flip_color(board.side_to_move), pt);
                if (piece as usize) < 12 {
                    update_corr_entry(&mut info.cont_corr[piece as usize][to as usize], err, weight, cap_div);
                }
            }
        }
    }
}

/// LMR reduction tables (quiet and capture).
/// Safety: initialized once at startup (main.rs) and on setoption (UCI thread).
/// Search threads read concurrently — technically a race on setoption during search,
/// but values change monotonically and a stale read produces a slightly wrong reduction,
/// not UB in practice (i32 reads/writes are atomic on x86-64).
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
    helper.root_stm = main.root_stm; // contempt needs to know who started the search
    helper.syzygy = main.syzygy.clone(); // share tablebases (read-only)
    // Helpers start with fresh history for SMP diversity (cleared in search_helper)
    helper
}

/// Run Lazy SMP search: main thread + N-1 helper threads.
pub fn search_smp(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits, threads: usize) -> Move {
    if threads <= 1 {
        info.global_nodes.store(0, Ordering::Relaxed);
        return search(board, info, limits);
    }

    // Reset shared state (TT generation is advanced in search(), not here,
    // to avoid double-increment which makes entries appear 2x staler)
    // Note: stop flag is cleared by the UCI thread before spawning the search
    // thread, not here. Clearing here races with ponderhit (which sets stop
    // before the search thread starts).
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
                helper.soft_floor = 0;
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
    info.moved_piece_stack = [0; MAX_PLY + 1];
    info.double_ext_count = [0; MAX_PLY + 1];
    info.moved_to_stack = [0; MAX_PLY + 1];
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

    // Enable threat delta generation if we have a threat net
    board.generate_threat_deltas = info.nnue_net.as_ref().map_or(false, |n| n.has_threats);

    // Initialize root position threat accumulator
    if info.threat_stack.active {
        info.threat_stack.reset();
        if let Some(ref net) = info.nnue_net {
            info.threat_stack.refresh(&net.threat_weights, net.num_threat_features, board, WHITE);
            info.threat_stack.refresh(&net.threat_weights, net.num_threat_features, board, BLACK);
        }
    }

    info.start_time = Instant::now();
    // Note: stop flag AND ponderhit_time are cleared by the UCI thread before
    // spawning the search thread, not here. Clearing here races with ponderhit:
    // if ponderhit arrives in the ~ms between `go ponder` and this line, UCI
    // sets ponderhit_time → search() clobbers it → ponder runs truly infinite →
    // wait-loop → eventual time forfeit (observed at blitz TC).
    info.nodes = 0;
    // Note: global_nodes reset is done by callers (search_smp, bench) to avoid
    // clobbering helper thread contributions in SMP mode.
    info.sel_depth = 0;
    info.root_stm = board.side_to_move;

    // Age history tables (×0.80) to preserve useful move ordering from prior searches.
    // Killers and counter-moves are cleared (position-specific). Correction history reset.
    info.history.age(4, 5);
    info.clear_correction_history();
    info.stats = PruneStats::default();
    // Age pawn history (×0.80, matching main/capture history aging)
    for entry in info.pawn_hist.iter_mut() {
        for piece in entry.iter_mut() {
            for val in piece.iter_mut() {
                *val = (*val as i32 * 4 / 5) as i16;
            }
        }
    }
    // Clear static evals, excluded moves, depth tracking
    info.static_evals = [0; MAX_PLY + 1];
    info.depth_nodes = [0; MAX_PLY + 1];
    info.completed_depth = 0;
    info.reductions = [0; MAX_PLY + 1];
    info.excluded_move = [NO_MOVE; MAX_PLY + 1];
    info.moved_piece_stack = [0; MAX_PLY + 1];
    info.double_ext_count = [0; MAX_PLY + 1];
    info.moved_to_stack = [0; MAX_PLY + 1];
    info.pv_table = [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1];
    info.pv_len = [0; MAX_PLY + 1];
    // Clear TM state
    info.tm_prev_best = NO_MOVE;
    info.tm_prev_score = 0;
    info.tm_best_stable = 0;
    info.tm_has_data = false;
    // Reset per-root-move node counts
    for v in info.root_move_nodes.iter_mut() { *v = 0; }

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

    if limits.infinite {
        info.time_limit = 0;
        info.soft_limit = 0;
        info.hard_limit = 0;
        info.soft_floor = 0;
    } else if limits.movetime > 0 {
        info.time_limit = limits.movetime;
        info.soft_floor = 0;
    } else if our_time > 0 {
        // Subtract move overhead (communication latency)
        let overhead = info.move_overhead;
        let time_left = our_time.saturating_sub(overhead).max(1);

        let moves_left = if limits.movestogo > 0 { limits.movestogo as u64 } else { 25 };

        // TC regime classification: seconds per move estimate

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

        // Save base soft before dynamic factors scale it.
        // CRITICAL: hard limit uses base_soft, not dynamically-scaled soft.
        // Dynamic factors can scale soft by up to ~2.5x (stability=0 ×
        // node_fraction=2.23). If hard uses scaled soft, the effective maximum
        // becomes soft × 2.5 × 3 = soft × 7.5, recreating the overspend problem.
        let base_soft = soft;

        // Hard limit: 3x base soft, but never more than time/20 + inc
        // The time/20 + inc cap prevents overspend at all TCs:
        //   60+0: max = 3000ms    60+2: max = 5000ms
        //   180+2: max = 11000ms  600+5: max = 35000ms
        let mut hard = if limits.movestogo > 0 {
            let hard_raw = base_soft * 2;
            let hard_pct = (95 - limits.movestogo as u64 * 10).max(30).min(90);
            let mtg_cap = time_left * hard_pct / 100;
            hard_raw.min(mtg_cap)
        } else {
            base_soft * 3
        };

        // Absolute hard cap: time/20 + inc (never risk more than 5% of clock + 1 increment)
        let mut max_hard = time_left / 20 + our_inc;
        if max_hard > time_left * 3 / 4 {
            max_hard = time_left * 3 / 4;
        }
        if hard > max_hard {
            hard = max_hard;
        }

        info.soft_limit = soft;
        // Ensure hard >= soft (but soft is also capped by max_hard for movestogo safety)
        if soft > hard { soft = hard; }
        info.hard_limit = hard;
        // Soft floor: never spend less than the increment we gain, so the
        // dynamic stability cut can't produce clock-growing instant emits
        // in stable endgames (lichess PZ7pCyrx stockpile). Capped at hard
        // to preserve the absolute maximum. Zero when inc is zero.
        info.soft_floor = our_inc.saturating_sub(overhead).min(hard);
        info.time_limit = hard.max(soft); // search uses hard as absolute limit
        info.tm_has_data = false;
        info.tm_best_stable = 0;
    } else if !limits.infinite {
        info.time_limit = 0;
    }

    info.max_depth = if limits.depth > 0 { limits.depth } else { MAX_PLY as i32 / 2 };
    info.max_nodes = limits.nodes;

    info.tt.new_search();

    let mut best_move = NO_MOVE;
    let mut prev_score = 0i32;

    // Get a fallback move and keep the legal list for final validation
    let root_legal = generate_legal_moves(board);
    if root_legal.len > 0 {
        best_move = root_legal.moves[0];
    }

    // Forced move: only one legal move, skip full search (just return it quickly).
    // Still search to depth 1 for a score to display, but cap time at 10ms.
    if root_legal.len == 1 && (info.soft_limit > 0 || info.time_limit > 0) {
        info.soft_limit = 10;
        info.hard_limit = 10;
        info.time_limit = 10;
        info.soft_floor = 0;
    }

    let effective_max = info.max_depth.min(MAX_PLY as i32 / 2);
    for depth in 1..=effective_max {
        if info.should_stop() { break; }
        // Ponderhit check: stop between iterations (not mid-search) to avoid
        // partial TT entries and PV inconsistency. The engine completes the
        // current iteration fully before stopping, producing clean state.
        let ph = info.ponderhit_time.load(std::sync::atomic::Ordering::Relaxed);
        if ph > 0 && info.start_time.elapsed().as_millis() as u64 >= ph {
            break;
        }
        let iter_start = std::time::Instant::now();

        let score;

        // Aspiration windows (skip for mate scores)
        if depth >= 4 && prev_score > -MATE_SCORE + 100 && prev_score < MATE_SCORE - 100 {
            // Eval-dependent aspiration delta: wider for extreme scores (Reckless pattern)
            // Calm positions (avg~0): delta=13, winning (avg~500): delta=24, crushing (avg~1000): delta=55
            let avg = prev_score;
            // S5: widen delta when our king is under attack at root.
            let root_king_sq = board.king_sq(board.side_to_move);
            let root_king_zone = crate::attacks::king_attacks(root_king_sq as u32) | (1u64 << root_king_sq);
            let root_enemy_attacks = board.attacks_by_color(flip_color(board.side_to_move));
            let root_king_pressure = popcount(root_enemy_attacks & root_king_zone) as i32;
            let mut delta = tp(&ASP_DELTA) + (avg as i64 * avg as i64 / tp(&ASP_SCORE_DIV) as i64) as i32
                + root_king_pressure * tp(&ASP_KING_PRESSURE_MARGIN);
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
            // Validate against root legal list (match from/to/flags for promotions)
            let pv_from = move_from(pv_move);
            let pv_to = move_to(pv_move);
            let pv_flags = move_flags(pv_move);
            for i in 0..root_legal.len {
                let m = root_legal.moves[i];
                if move_from(m) == pv_from && move_to(m) == pv_to
                    && (!is_promotion(pv_move) || move_flags(m) == pv_flags)
                {
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
                let tt_flags = move_flags(tt_entry.best_move);
                for i in 0..root_legal.len {
                    let m = root_legal.moves[i];
                    if move_from(m) == tt_from && move_to(m) == tt_to
                        && (!is_promotion(tt_entry.best_move) || move_flags(m) == tt_flags)
                    {
                        best_move = m;
                        break;
                    }
                }
            }
        }

        prev_score = score;
        info.last_score = score;
        info.ponder_depth.store(depth as u64, std::sync::atomic::Ordering::Relaxed);

        // Record cumulative nodes at this depth (for EBF calculation)
        if (depth as usize) < MAX_PLY {
            info.depth_nodes[depth as usize] = info.nodes;
            info.completed_depth = depth;
        }

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
        // Track game history hashes throughout to stop at threefold repetition
        let mut pv_str = String::new();
        {
            let mut seen_hashes: Vec<u64> = board.undo_stack.iter().map(|u| u.hash).collect();
            seen_hashes.push(board.hash);
            let mut pv_board = board.clone();
            let mut pv_moves = 0usize;

            // Use PV table first (stop at repetition)
            let pv_len = info.pv_len[0].min(MAX_PLY);
            for i in 0..pv_len {
                pv_board.make_move(info.pv_table[0][i]);
                if seen_hashes.iter().filter(|&&h| h == pv_board.hash).count() >= 2 { break; }
                seen_hashes.push(pv_board.hash);
                if !pv_str.is_empty() { pv_str.push(' '); }
                pv_str.push_str(&move_to_uci(info.pv_table[0][i]));
                pv_moves += 1;
            }

            // Extend with TT if PV table was short
            if pv_moves < depth as usize {
                while pv_moves < depth as usize + 5 {
                    if seen_hashes.iter().filter(|&&h| h == pv_board.hash).count() >= 2 { break; }
                    if pv_board.halfmove >= 100 { break; }
                    seen_hashes.push(pv_board.hash);

                    let pv_tt = info.tt.probe(pv_board.hash);
                    if !pv_tt.hit || pv_tt.best_move == NO_MOVE { break; }
                    let pv_from = move_from(pv_tt.best_move);
                    let pv_to = move_to(pv_tt.best_move);
                    let pv_flags = move_flags(pv_tt.best_move);
                    let pv_legal = generate_legal_moves(&pv_board);
                    let mut found = NO_MOVE;
                    for i in 0..pv_legal.len {
                        let m = pv_legal.moves[i];
                        if move_from(m) == pv_from && move_to(m) == pv_to
                            && (!is_promotion(pv_tt.best_move) || move_flags(m) == pv_flags)
                        {
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

        // Dynamic time management: 3-factor model (Obsidian/Clarity pattern)
        // Combines node fraction, best-move stability, and score trend.
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

            // Score trend: how much has the score dropped since last iteration?
            // Positive = score is dropping (need more time), negative = improving
            let score_drop = if info.tm_has_data && !is_mate_score(prev_score) && !is_mate_score(info.tm_prev_score) {
                info.tm_prev_score - prev_score  // positive when eval is falling
            } else {
                0
            };

            info.tm_prev_best = best_move;
            info.tm_prev_score = prev_score;
            info.tm_has_data = true;

            // Factor 1: Node fraction (Obsidian pattern)
            // How concentrated is the search on the best move?
            // High fraction → confident → use less time. Low fraction → uncertain → use more.
            let nodes_factor = if depth > 9 && best_move != NO_MOVE {
                let bm_from = move_from(best_move) as usize;
                let bm_to = move_to(best_move) as usize;
                let best_nodes = info.root_move_nodes[bm_from * 64 + bm_to];
                let total = info.nodes;
                if total > 0 {
                    let frac = best_nodes as f64 / total as f64;
                    // Obsidian: 0.63 + (1.0 - frac) * 2.0
                    // frac=0.9 → 0.83, frac=0.5 → 1.63, frac=0.2 → 2.23
                    0.63 + (1.0 - frac) * 2.0
                } else {
                    1.25  // default when no data (Clarity pattern)
                }
            } else {
                1.25  // early depths: use default multiplier
            };

            // Factor 2: Best-move stability (Obsidian linear pattern)
            // Each stable iteration reduces time by 8%
            // 0 stable: 1.71x, 5 stable: 1.31x, 10 stable: 0.91x
            let stability_factor = (1.71 - info.tm_best_stable as f64 * 0.08).max(0.5);

            // Factor 3: Score trend (Obsidian pattern, simplified)
            // Dropping score → use more time. Rising score → slightly less.
            // scoreFactor = clamp(0.86 + 0.010 * scoreDrop, 0.81, 1.50)
            let score_factor = (0.86 + 0.010 * score_drop as f64).clamp(0.81, 1.50);

            // Combined: all three factors multiply against the soft limit
            let scale = nodes_factor * stability_factor * score_factor;

            // Check if we should stop at the soft limit.
            // Floor at soft_floor (≈ increment) so stability cuts in stable
            // endgames can't produce clock-growing instant emits.
            let adjusted_soft = (info.soft_limit as f64 * scale) as u64;
            let adjusted_soft = adjusted_soft.max(info.soft_floor).min(info.hard_limit);
            if elapsed >= adjusted_soft {
                break;
            }

            // Next-iteration estimate: stop if next iteration would exceed time limit.
            // Use 2x last iteration time as estimate (exponential branching).
            // Check both hard_limit (normal) and ponderhit_time (after ponderhit).
            // Without this, ponder searches start arbitrarily deep iterations after
            // ponderhit, get stopped mid-search, and leave incomplete TT entries.
            let effective_hard = {
                let ph = info.ponderhit_time.load(std::sync::atomic::Ordering::Relaxed);
                if ph > 0 { ph } else { info.hard_limit }
            };
            if effective_hard > 0 {
                let iter_elapsed = iter_start.elapsed().as_millis() as u64;
                if elapsed > 0 && effective_hard > elapsed && (effective_hard - elapsed) < 2 * iter_elapsed {
                    break;
                }
            }
        }
    }

    // Don't stockpile: if the ID loop finished below the soft_floor (e.g. all
    // iterations were TT hits in a repetitive endgame), wait out the rest of
    // the floor time before emitting. Prevents clock growth from instant emits
    // at 1s-inc bullet on lichess (PZ7pCyrx). Polls the stop flag so the UCI
    // thread can still interrupt. Skip when there's no time budget (depth/
    // node-limited search) or when already stopped.
    if info.soft_floor > 0 && !info.stop.load(Ordering::Relaxed) {
        loop {
            let elapsed = info.start_time.elapsed().as_millis() as u64;
            if elapsed >= info.soft_floor { break; }
            if info.stop.load(Ordering::Relaxed) { break; }
            let remaining = info.soft_floor - elapsed;
            std::thread::sleep(std::time::Duration::from_millis(remaining.min(25)));
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

    // Mate distance pruning — applies to all nodes (standard form)
    let is_pv = beta - alpha > 1;
    {
        let mating_score = MATE_SCORE - ply - 1;
        if mating_score < beta {
            beta = mating_score;
            if alpha >= mating_score { return mating_score; }
        }
        let mated_score = -MATE_SCORE + ply;
        if mated_score > alpha {
            alpha = mated_score;
            if beta <= mated_score { return mated_score; }
        }
    }

    // Prefetch TT bucket early to hide memory latency
    info.tt.prefetch(board.hash);

    // Threat-aware history indexing: upgrade from pawn-only to all-enemy-pieces.
    // `enemy_attacks` keys the 4D main history slot (from_threatened, to_threatened);
    // broader threat coverage → finer move-ordering distinctions.
    // Cost: 8-12 extra magic lookups per node, only at non-QS non-TT-cut nodes.
    let them_color = flip_color(board.side_to_move);
    let enemy_attacks: u64 = board.attacks_by_color(them_color);

    // Pawn-specific threat count kept separate: RFP margin adjustment and
    // LMR_THREAT_DIV are tuned on the pawn-only scale.
    let their_pawns = board.pieces[PAWN as usize] & board.colors[them_color as usize];
    let enemy_pawn_attacks: u64 = if them_color == WHITE {
        ((their_pawns & !0x0101010101010101u64) << 7) | ((their_pawns & !0x8080808080808080u64) << 9)
    } else {
        ((their_pawns & !0x8080808080808080u64) >> 7) | ((their_pawns & !0x0101010101010101u64) >> 9)
    };
    let our_non_pawns = board.colors[board.side_to_move as usize]
        & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
    let has_pawn_threats = (enemy_pawn_attacks & our_non_pawns) != 0;
    let threat_count = popcount(enemy_pawn_attacks & our_non_pawns) as i32;
    // our_defenses signal for futility widener: count of our non-pawn
    // pieces under any enemy attack (pawn OR piece). Widens margin in
    // tactical positions. Uses existing enemy_attacks — no new bitboard.
    let any_threat_count = popcount(enemy_attacks & our_non_pawns) as i32;
    // B1: Discovered-attack bitboard. Our pieces that are currently
    // blocking one of our sliders' attack on an enemy piece — moving
    // any such piece uncovers a slider attack. Used as a quiet-move
    // ordering bonus in MovePicker. Cost: 10-20 magic lookups.
    let our_xray_blockers: u64 = if tp(&DISCOVERED_ATTACK_BONUS) > 0 {
        board.xray_blockers(board.side_to_move)
    } else {
        0
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
    // Contempt is relative to the ROOT side (us): we get -CONTEMPT for draws
    // (we don't want to draw), opponent gets +CONTEMPT (we're happy if they draw).
    // This prevents us from playing INTO repetitions when we have an advantage.
    if ply > 0 {
        let contempt = tp(&CONTEMPT_VAL);
        // Jitter draw score by ±2 to break ties between draw paths (Koivisto pattern)
        let jitter = 2 - (info.nodes & 3) as i32; // range: -1 to +2
        let draw_score = if board.side_to_move == info.root_stm { -contempt + jitter } else { contempt + jitter };
        if board.halfmove >= 100 {
            return draw_score;
        }
        let stack_len = board.undo_stack.len();
        let limit = (board.halfmove as usize).min(stack_len);
        let mut i = 2usize;
        while i <= limit {
            if board.undo_stack[stack_len - i].hash == board.hash {
                return draw_score;
            }
            i += 2;
        }
    }

    // Syzygy tablebase probe at interior nodes.
    // Probe WDL when piece count is within TB range. Returns a score that
    // causes a cutoff, so the search doesn't waste time in solved endgames.
    // Only at non-root (ply > 0) and non-excluded (not in singular verification).
    if ply > 0 && info.excluded_move[ply_u] == NO_MOVE {
        if let Some(ref tb) = info.syzygy {
            if crate::bitboard::popcount(board.occupied()) as usize <= tb.max_pieces() {
                if let Some(wdl) = tb.probe_wdl(board) {
                    // wdl from ambiguous_wdl_to_score: ±20000 = definite, ±1 = ambiguous, 0 = draw
                    // Only use large TB scores for definite Win/Loss.
                    // Ambiguous results (CursedWin=1, MaybeLoss=-1) stay small
                    // so the search treats them as near-draw, not resignation triggers.
                    let tb_score = if wdl > 1 {
                        TB_WIN - ply  // definite win
                    } else if wdl < -1 {
                        -TB_WIN + ply  // definite loss
                    } else {
                        wdl  // ambiguous (±1) or draw (0): use as-is
                    };

                    if tb_score >= beta { return tb_score; }
                    if tb_score <= alpha { return tb_score; }
                    // Exact score in window: tighten bounds
                    alpha = tb_score;
                }
            }
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

    // Sticky PV flag: once a position is searched as PV, it stays PV in the TT.
    // Used to reduce LMR for moves that lead to historically important positions.
    let tt_pv = is_pv || (tt_hit && tt_entry.tt_pv);

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
                // Unified TT cutoff with node-type guard (Alexandria pattern):
                // At non-PV nodes, accept TT cutoff when:
                // - cut_node matches score direction (cut expects fail-high, all expects fail-low)
                // - TT bound type matches (LOWER for fail-high, UPPER for fail-low)
                // - Not too close to 50-move rule (avoid drawing positions incorrectly)
                let score_above_beta = tt_score >= beta;
                let bound_matches = if score_above_beta {
                    tt_entry.flag == TT_FLAG_LOWER || tt_entry.flag == TT_FLAG_EXACT
                } else {
                    tt_entry.flag == TT_FLAG_UPPER || tt_entry.flag == TT_FLAG_EXACT
                };
                if !is_pv && cut_node == score_above_beta && bound_matches
                    && board.halfmove < 90
                {
                    info.stats.tt_cutoffs += 1;
                    if tt_move != NO_MOVE && ply_u <= MAX_PLY {
                        info.pv_table[ply_u][0] = tt_move;
                        info.pv_len[ply_u] = 1;
                    } else if ply_u <= MAX_PLY {
                        info.pv_len[ply_u] = 0;
                    }
                    // TT cutoff cont-hist malus: penalize opponent's last quiet move
                    // in context of our move before that (Alexandria pattern).
                    // "Your move led to a position we already know is lost for you."
                    let stack_len = board.undo_stack.len();
                    if score_above_beta && stack_len >= 2 {
                        let opp_undo = &board.undo_stack[stack_len - 1];
                        let our_undo = &board.undo_stack[stack_len - 2];
                        if opp_undo.mv != NO_MOVE && opp_undo.captured == NO_PIECE_TYPE
                            && our_undo.mv != NO_MOVE
                        {
                            let opp_to = move_to(opp_undo.mv);
                            let opp_piece = board.piece_at(opp_to);
                            let our_to = move_to(our_undo.mv);
                            let our_piece = board.piece_at(our_to);
                            if opp_piece != NO_PIECE && our_piece != NO_PIECE {
                                let malus = -((155 * depth).min(385));
                                History::update_cont_history(
                                    &mut info.history.cont_hist[go_piece(our_piece)][our_to as usize][go_piece(opp_piece)][opp_to as usize],
                                    malus,
                                );
                            }
                        }
                    }
                    return tt_score;
                }

                // Fall through: use TT bounds to narrow alpha/beta window at non-PV nodes
                match tt_entry.flag {
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
                        let tt_piece = board.piece_at(move_from(tt_move));
                        let tt_is_cap = board.piece_type_at(move_to(tt_move)) != NO_PIECE_TYPE
                            || move_flags(tt_move) == FLAG_EN_PASSANT;
                        if !tt_is_cap && tt_piece != NO_PIECE {
                            let bonus = history_bonus(depth);
                            History::update_history(
                                info.history.main_entry(move_from(tt_move), move_to(tt_move), enemy_attacks),
                                bonus,
                            );
                        } else if tt_is_cap && tt_piece != NO_PIECE {
                            let bonus = capture_history_bonus(depth);
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
            diff > tp(&UNSTABLE_THRESH)
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
    // Guard against consecutive null moves
    let prev_was_null = !board.undo_stack.is_empty()
        && board.undo_stack[board.undo_stack.len() - 1].mv == NO_MOVE;
    // King-zone-pressure gate: skip NMP when enemy has many attackers
    // on our king zone. A null move in an attacking position gives
    // opponent an extra tempo at the worst moment.
    let our_king_sq = board.king_sq(board.side_to_move);
    let king_zone = crate::attacks::king_attacks(our_king_sq as u32) | (1u64 << our_king_sq);
    let king_zone_pressure = popcount(enemy_attacks & king_zone) as i32;

    if depth >= 3 && !in_check && ply > 0 && stm_non_pawn != 0
        && beta - alpha == 1 && static_eval >= beta
        && !prev_was_null  // Prevent consecutive null moves
        && beta.abs() < MATE_SCORE - 100  // Skip NMP for mate/TB scores
        && info.excluded_move[ply_u] == NO_MOVE  // Skip NMP during SE verification
        && king_zone_pressure < tp(&NMP_KING_ZONE_MAX)  // New gate
        && FEAT_NMP.load(Ordering::Relaxed)
    {
        info.stats.nmp_attempts += 1;
        // Adaptive reduction: scales with depth and eval margin above beta
        let mut r = tp(&NMP_BASE_R) + depth / tp(&NMP_DEPTH_DIV);
        // Reduce more after captures: opponent just captured, null move more likely to work
        // (Consensus: SF/Obsidian increase R after captures, not decrease)
        if !board.undo_stack.is_empty() && board.undo_stack[board.undo_stack.len() - 1].captured != NO_PIECE_TYPE {
            r += 1;
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
        if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }
        let null_score = -negamax(board, info, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }
        board.unmake_null_move();

        if info.stop.load(Ordering::Relaxed) {
            return 0;
        }

        if null_score >= beta {
            // Return null score directly (no dampening — no top engine uses it)
            // Clamp mate scores to beta to avoid inflated mate distance
            let nmp_score = if null_score.abs() > MATE_SCORE - 100 { beta } else { null_score };

            // Verification search at high depths to guard against zugzwang
            if depth >= tp(&NMP_VERIFY_DEPTH) {
                info.stats.nmp_verify += 1;
                // Verification re-searches current position (no move made), so ply stays same
                let v_score = negamax(board, info, beta - 1, beta, depth - r, ply, false);
                if v_score >= beta {
                    info.stats.nmp_cutoffs += 1;
                    return nmp_score;
                }
                info.stats.nmp_verify_fail += 1;
            } else {
                info.stats.nmp_cutoffs += 1;
                return nmp_score;
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
            let mut margin = if improving { depth * tp(&RFP_MARGIN_IMP) } else { depth * tp(&RFP_MARGIN_NOIMP) };
            // Widen margin when opponent pawns attack our pieces (Minic/Berserk pattern)
            if has_pawn_threats { margin += margin / 3; }
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
        && king_zone_pressure < tp(&PROBCUT_KING_ZONE_MAX)  // A3: skip in high-threat positions
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
        if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }
            if !board.make_move(mv) {
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }
                continue;
            }
                    if info.threat_stack.active { let entry = info.threat_stack.current_mut(); entry.delta.clear(); for d in board.threat_deltas.iter() { entry.delta.push(*d); } let ul = board.undo_stack.len(); if ul > 0 { let u = &board.undo_stack[ul-1]; entry.mv = u.mv; if u.mv != crate::types::NO_MOVE { entry.moved_pt = board.mailbox[crate::types::move_to(u.mv) as usize]; entry.moved_color = crate::types::flip_color(board.side_to_move); } } }
            info.tt.prefetch(board.hash);

            // Cheap qsearch verification before expensive negamax (Stockfish pattern)
            let mut score = -quiescence(board, info, -probcut_beta, -probcut_beta + 1, ply + 1);

            // Only do deeper search if qsearch also beats probcut_beta
            if score >= probcut_beta && pc_depth > 0 {
                score = -negamax(board, info, -probcut_beta, -probcut_beta + 1, pc_depth, ply + 1, !cut_node);
            }

            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }

            if info.stop.load(Ordering::Relaxed) {
                return 0;
            }

            if score >= probcut_beta {
                info.stats.probcut_cutoffs += 1;
                // TT stores the RAW verified score (a tighter lower bound than
                // the dampened value) and preserves the sticky PV flag — matches
                // Stockfish search.cpp:994-995. Prior code stored `dampened` and
                // hardcoded tt_pv=false, losing both pruning information on
                // future probes and the PV stickiness used by LMR reduction
                // decisions. Return value is still dampened — score was
                // verified at probcut_beta = beta+margin, not beta.
                info.tt.store(
                    board.hash, depth - 3, score_to_tt(score, ply),
                    TT_FLAG_LOWER, mv, raw_eval, tt_pv,
                );
                return score - (probcut_beta - beta);
            }
        }
    }

    // Continuation history lookup from search stack (killers/counter removed — SF pattern)
    let safe_ply = ply_u.min(MAX_PLY - 1).min(63);
    let mut prev_piece_for_cont: usize = 0; // go_piece index (1-12), 0 = none
    let mut prev_to_for_cont: u8 = 0;
    let mut prev2_piece_for_cont: usize = 0; // ply-2 (grandparent move)
    let mut prev2_to_for_cont: u8 = 0;

    // Ply-1: parent's move (for continuation history)
    if ply_u >= 1 {
        let gp = info.moved_piece_stack[ply_u - 1] as usize;
        let to_sq = info.moved_to_stack[ply_u - 1];
        if gp != 0 {
            prev_piece_for_cont = gp;
            prev_to_for_cont = to_sq;
        }
    }

    // Ply-2: grandparent's move (correct — uses stack, not stale board.piece_at)
    if ply_u >= 2 {
        let gp2 = info.moved_piece_stack[ply_u - 2] as usize;
        let to_sq2 = info.moved_to_stack[ply_u - 2];
        if gp2 != 0 {
            prev2_piece_for_cont = gp2;
            prev2_to_for_cont = to_sq2;
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
        MovePicker::new_evasion(tt_move, safe_ply, checkers, pinned, &info.history, prev_move, pawn_hist_ref, &info.moved_piece_stack, &info.moved_to_stack)
    } else {
        MovePicker::new(board, tt_move, safe_ply, &info.history, prev_move, pawn_hist_ref, enemy_attacks, our_xray_blockers, &info.moved_piece_stack, &info.moved_to_stack)
    };
    picker.threat_sq = threat_sq;

    let mut best_move = NO_MOVE;
    let mut best_score = -INFINITY;
    let mut move_count = 0i32;
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
            && !see_ge(board, mv, -(depth * tp(&SEE_MATERIAL_SCALE)))
            && FEAT_SEE_PRUNE.load(Ordering::Relaxed)
        {
            continue;
        }

        // Estimated LMR depth for pre-MakeMove pruning (SEE quiet, futility).
        // Computed once and shared — no depth ceiling; at high depths lmr_d
        // collapses to 1, so thresholds naturally become permissive.
        let lmr_d = if move_count > 1 && depth >= 2 {
            let r = lmr_reduction((depth as usize).min(63) as i32, (move_count as usize).min(63) as i32);
            if r > 0 { (depth - r).max(1) } else { depth }
        } else {
            depth
        };

        // SEE quiet pruning: prune quiet moves landing on attacked squares.
        // Use lmrDepth² scaling (matching Stockfish/Berserk/Obsidian).
        if ply > 0 && !in_check
            && !is_cap && !is_promo
            && mv != tt_move
            && best_score > -(MATE_SCORE - 100)
            && FEAT_SEE_PRUNE.load(Ordering::Relaxed)
        {
            let see_quiet_threshold = -tp(&SEE_QUIET_MULT) * lmr_d * lmr_d;
            if !see_ge(board, mv, see_quiet_threshold) {
                info.stats.see_prunes += 1;
                continue;
            }
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
                    // TT move is singular — no competitive alternatives.
                    let is_pv = beta - alpha > 1;
                    if !is_pv && singular_score < singular_beta - tp(&DEXT_MARGIN)
                        && info.double_ext_count[ply_u] < tp(&DEXT_CAP)
                    {
                        // Double extension (+2): well below singular beta (margin=10, Velvet uses 4)
                        singular_extension = 2;
                    } else {
                        singular_extension = 1;
                    }
                } else if tt_score_local >= beta {
                    // TT move fails high and alternatives competitive — strong reduce
                    // Consensus: -3 non-PV (SF/Viridithas/Obsidian)
                    singular_extension = -3;
                } else if cut_node {
                    // Cut node with competitive alternatives — moderate reduce
                    singular_extension = -2;
                } else {
                    // All-node with competitive alternatives — mild reduce
                    singular_extension = -1;
                }
            }
        }

        // Save moved piece before MakeMove for consistent history indexing
        let moved_piece = board.piece_at(from);
        let moved_pt = board.piece_type_at(from);

        // Record on search stack for correct ply-2+ cont hist lookups
        if moved_piece != NO_PIECE && ply_u <= MAX_PLY {
            info.moved_piece_stack[ply_u] = go_piece(moved_piece) as u8;
            info.moved_to_stack[ply_u] = to;
        }
        let captured_pt = if is_cap {
            if flags == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(to) }
        } else {
            NO_PIECE_TYPE
        };

        // History-based pruning: prune quiet moves with deeply negative history at shallow depths
        if ply > 0 && !in_check && !improving && !unstable && depth <= tp(&HIST_PRUNE_DEPTH)
            && !is_cap && !is_promo
            && mv != tt_move
            && best_score > -(MATE_SCORE - 100)
            && FEAT_HIST_PRUNE.load(Ordering::Relaxed)
        {
            let mut hist_prune_score = info.history.main_score(from, to, enemy_attacks);
            if moved_piece != NO_PIECE {
                let gp = go_piece(moved_piece);
                if prev_piece_for_cont != 0 {
                    hist_prune_score += info.history.cont_hist[prev_piece_for_cont][prev_to_for_cont as usize][gp][to as usize] as i32;
                }
                // Pawn history in pruning decision
                let ph_idx = (board.pawn_hash as usize) % info.pawn_hist.len();
                hist_prune_score += info.pawn_hist[ph_idx][gp][to as usize] as i32;
            }
            if hist_prune_score < -tp(&HIST_PRUNE_MULT) * depth as i32 {
                info.stats.history_prunes += 1;
                continue;
            }
        }

        // Futility pruning: skip quiet moves when static eval + margin is below alpha.
        // Uses shared lmr_d for both gate and margin (SF/Obsidian/Berserk consensus).
        if ply > 0 && static_eval > -INFINITY && !in_check
            && !is_cap && !is_promo
            && best_score > -(MATE_SCORE - 100)
            && FEAT_FUTILITY.load(Ordering::Relaxed)
            && lmr_d <= 10
        {
            let main_hist = info.history.main_score(from, to, enemy_attacks);
            let hist_adj = main_hist / 128;
            // our_defenses widener: add margin per our-piece-under-attack so
            // tactical positions keep more lines from being pruned on eval.
            let threats_adj = any_threat_count * tp(&FUT_THREATS_MARGIN);
            let futility_value = static_eval + tp(&FUT_BASE) + lmr_d * tp(&FUT_PER_DEPTH) + hist_adj + threats_adj;
            // Don't futility-prune moves with very strong history (Igel pattern)
            if futility_value <= alpha && main_hist < 12000 {
                info.stats.futility_prunes += 1;
                continue;
            }
        }

        // Late Move Pruning: at shallow depths, skip late quiet moves.
        // Applied before MakeMove. Formula: (LMP_BASE + depth²) / (2 - improving)
        if ply > 0 && !in_check && depth >= 1 && depth <= tp(&LMP_DEPTH)
            && !is_cap && !is_promo
            && best_score > -(MATE_SCORE - 100)
            && FEAT_LMP.load(Ordering::Relaxed)
        {
            let lmp_limit = (tp(&LMP_BASE) + depth * depth) / (2 - improving as i32);
            if move_count > lmp_limit {
                info.stats.lmp_prunes += 1;
                continue;
            }
        }

        // Bad noisy pruning: skip losing captures when eval is far below alpha.
        // Applied before MakeMove — no gives_check exemption (matches pre-move pattern).
        if FEAT_BAD_NOISY.load(Ordering::Relaxed) && is_cap && !in_check && ply > 0 && depth <= 4 && mv != tt_move
            && !is_promo && best_score > -(MATE_SCORE - 100)
            && static_eval > -INFINITY && static_eval + depth * tp(&BAD_NOISY_MARGIN) <= alpha
            && !see_ge(board, mv, 0)
        {
            continue;
        }

        // Build NNUE dirty piece info BEFORE make_move
        let dirty = build_dirty_piece(mv, us, flip_color(us), moved_pt, captured_pt);

        // Push NNUE accumulator
        if let Some(acc) = &mut info.nnue_acc { acc.push(dirty); }
        if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }

        if !board.make_move(mv) {
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }
            continue;
        }
        // Store threat deltas from make_move into accumulator stack
                if info.threat_stack.active { let entry = info.threat_stack.current_mut(); entry.delta.clear(); for d in board.threat_deltas.iter() { entry.delta.push(*d); } let ul = board.undo_stack.len(); if ul > 0 { let u = &board.undo_stack[ul-1]; entry.mv = u.mv; if u.mv != crate::types::NO_MOVE { entry.moved_pt = board.mailbox[crate::types::move_to(u.mv) as usize]; entry.moved_color = crate::types::flip_color(board.side_to_move); } } }

        // Prefetch TT bucket for the new position
        info.tt.prefetch(board.hash);

        // Check if move gives check (opponent is now in check after make_move)
        let gives_check = board.in_check();

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

        // Propagate double extension counter to child
        if ply_u + 1 <= MAX_PLY {
            info.double_ext_count[ply_u + 1] = info.double_ext_count[ply_u]
                + if singular_extension >= 2 { 1 } else { 0 };
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
        let mut reduction = 0i32;
        if !in_check && !is_cap && !is_promo && FEAT_LMR.load(Ordering::Relaxed) {
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

                // Reduce more when TT move is a capture
                if tt_move_noisy {
                    reduction += 1;
                }

                // Reduce more when opponent has few non-pawn pieces (simpler position)
                // Note: board is post-make_move, so side_to_move IS the opponent
                let opp_non_pawn = board.colors[board.side_to_move as usize]
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

                // Reduce less when position was previously a PV node (Alexandria/Obsidian/Seer pattern).
                // Sticky: once a position is searched as PV, tt_pv stays set even at non-PV nodes.
                if tt_pv {
                    reduction -= 1;
                }

                // Continuous history adjustment: good history reduces less, bad more
                // Uses main history + ply-1 + ply-2 continuation history (consensus).
                // Ply-2 weighted at half to avoid over-scaling the total.
                let mut hist_score = info.history.main_score(from, to, enemy_attacks);
                if moved_piece != NO_PIECE {
                    let gp = go_piece(moved_piece);
                    if prev_piece_for_cont != 0 {
                        hist_score += info.history.cont_hist[prev_piece_for_cont][prev_to_for_cont as usize][gp][to as usize] as i32;
                    }
                    if prev2_piece_for_cont != 0 {
                        hist_score += info.history.cont_hist[prev2_piece_for_cont][prev2_to_for_cont as usize][gp][to as usize] as i32 / 2;
                    }
                    // Pawn history: pawn-structure-aware move quality (SF/Alexandria pattern)
                    let ph_idx = (board.pawn_hash as usize) % info.pawn_hist.len();
                    hist_score += info.pawn_hist[ph_idx][gp][to as usize] as i32;
                }
                let hist_adj = hist_score / tp(&LMR_HIST_DIV);
                reduction -= hist_adj;

                // Complexity-aware LMR: reduce less when correction history
                // magnitude is high (uncertain eval → search deeper).
                // Matches Obsidian: R -= complexity / 120.
                if raw_eval > -INFINITY {
                    let complexity = (static_eval - raw_eval).abs();
                    reduction -= complexity / tp(&LMR_COMPLEXITY_DIV);
                }

                // Threat-density LMR: reduce less when multiple pieces are
                // under pawn attack. Tactical positions need deeper search.
                reduction -= threat_count / tp(&LMR_THREAT_DIV);

                // King-pressure LMR modifier: reduce less when enemy has
                // many attackers on our king zone. Parent-node signal reused
                // from NMP/ProbCut gates — tactical king positions need depth.
                reduction -= king_zone_pressure / tp(&LMR_KING_PRESSURE_DIV);

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

        // Track nodes per root move for node-based time management
        let nodes_before = if ply == 0 { info.nodes } else { 0 };

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
        if info.threat_stack.active { info.threat_stack.pop(); }

        // Accumulate nodes for this root move
        if ply == 0 {
            let idx = (from as usize) * 64 + (to as usize);
            info.root_move_nodes[idx] += info.nodes - nodes_before;
        }

        if info.stop.load(Ordering::Relaxed) {
            return 0;
        }

        if score > best_score {
            best_score = score;
            best_move = mv;

            if score > alpha {
                alpha = score;

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
                    info.stats.cutoff_movecount_sum += move_count as u64;
                    info.stats.cutoff_movecount_sq_sum += (move_count as u64) * (move_count as u64);

                    // Beta cutoff - update history for quiet moves (killers/counter removed — SF pattern)
                    if !is_cap {
                        let bonus = history_bonus(depth);

                        // Update main history
                        History::update_history(
                            info.history.main_entry(from, to, enemy_attacks),
                            bonus,
                        );

                        // Update continuation history at plies 1, 2, 4, 6
                        // Ply-1 at full bonus, plies 2/4/6 at half bonus (Obsidian pattern)
                        if moved_piece != NO_PIECE {
                            let gp_mv = go_piece(moved_piece);
                            let ch_offsets = [1usize, 2, 4, 6];
                            for &off in &ch_offsets {
                                if ply_u >= off {
                                    let prior_piece = info.moved_piece_stack[ply_u - off] as usize;
                                    let prior_to = info.moved_to_stack[ply_u - off] as usize;
                                    if prior_piece > 0 && prior_piece < 12 && prior_to < 64 {
                                        let ch_bonus = if off <= 1 { bonus } else { bonus / 2 };
                                        History::update_cont_history(
                                            &mut info.history.cont_hist[prior_piece][prior_to][gp_mv][to as usize],
                                            ch_bonus,
                                        );
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
                                    let ch_offsets = [1usize, 2, 4, 6];
                                    for &off in &ch_offsets {
                                        if ply_u >= off {
                                            let prior_piece = info.moved_piece_stack[ply_u - off] as usize;
                                            let prior_to = info.moved_to_stack[ply_u - off] as usize;
                                            if prior_piece > 0 && prior_piece < 12 && prior_to < 64 {
                                                let ch_pen = if off <= 1 { -bonus } else { -bonus / 2 };
                                                History::update_cont_history(
                                                    &mut info.history.cont_hist[prior_piece][prior_to][gp_q][qt as usize],
                                                    ch_pen,
                                                );
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

                    } else {
                        // Capture caused beta cutoff: bonus the cutoff capture
                        let cap_bonus = capture_history_bonus(depth);
                        if moved_piece != NO_PIECE && captured_pt != NO_PIECE_TYPE {
                            let cpt = if flags == FLAG_EN_PASSANT {
                                captured_type(PAWN)
                            } else {
                                captured_type(captured_pt)
                            };
                            History::update_cont_history(
                                &mut info.history.capture[go_piece(moved_piece)][to as usize][cpt],
                                cap_bonus,
                            );
                        }
                    }

                    // Unconditionally penalize all tried captures that didn't cause cutoff
                    // (matching Stockfish/Obsidian/Viridithas — captures that fail should be
                    // penalized regardless of whether the best move was quiet or tactical)
                    {
                        let cap_malus = capture_history_bonus(depth);
                        let cap_count = if is_cap { n_captures_tried.saturating_sub(1) } else { n_captures_tried };
                        for i in 0..cap_count {
                            let (cp, ct, cv) = captures_tried[i];
                            History::update_cont_history(
                                &mut info.history.capture[cp as usize][ct as usize][cv as usize],
                                -cap_malus,
                            );
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
    // Also skip if search was stopped — partial results corrupt the TT.
    // Child nodes that completed before stop are individually valid but
    // the parent's best_score is based on an incomplete move list.
    if info.excluded_move[ply_u] == NO_MOVE && !info.stop.load(Ordering::Relaxed) {
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
            info.tt.store(board.hash, depth, store_score, flag, best_move, raw_eval, tt_pv);
        }
    }

    // Update pawn-hash correction history when we have a reliable score.
    //
    // Skip when best_move is a capture/promotion: the score delta
    // (best_score - raw_eval) is then dominated by material change, not the
    // positional-eval miscalibration correction history is trying to learn.
    // Training on noisy bestmoves pollutes the tables. Matches Stockfish
    // (search.cpp:1495: `!(bestMove && pos.capture(bestMove))`) and Reckless
    // (search.rs:1085: `|| best_move.is_noisy()`).
    let best_move_noisy = best_move != NO_MOVE && {
        board.piece_type_at(move_to(best_move)) != NO_PIECE_TYPE
            || move_flags(best_move) == FLAG_EN_PASSANT
            || is_promotion(best_move)
    };
    if !in_check && best_move != NO_MOVE
        && !best_move_noisy
        && info.excluded_move[ply_u] == NO_MOVE
        && best_score > alpha_orig
        && best_score > -(MATE_SCORE - 100) && best_score < MATE_SCORE - 100
        && raw_eval > -(MATE_SCORE - 100)
    {
        update_correction_history(info, board, best_score, raw_eval, depth);
    }

    // Fail-high score blending: dampen inflated cutoff scores at non-PV nodes
    if best_score >= beta && beta - alpha_orig == 1 && depth >= tp(&FH_BLEND_DEPTH)
        && best_score > -(MATE_SCORE - 100) && best_score < MATE_SCORE - 100
    {
        return (best_score * depth + beta) / (depth + 1);
    }

    best_score
}

/// History bonus: linear depth-based bonus for history updates.
/// Consensus: SF min(1469, 155*d-93), Clarity min(1632, 276*d-119),
/// Obsidian min(1400, 175*d-50). Our old depth² formula gave 25 at d=5
/// vs SF's 682 — history values were 27× too small to influence ordering.
fn history_bonus(depth: i32) -> i32 {
    (tp(&HIST_BONUS_MULT) * depth).min(tp(&HIST_BONUS_MAX))
}

fn capture_history_bonus(depth: i32) -> i32 {
    (tp(&CAP_HIST_MULT) * depth - tp(&CAP_HIST_BASE)).clamp(0, tp(&CAP_HIST_MAX))
}

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
fn quiescence_with_depth(
    board: &mut Board,
    info: &mut SearchInfo,
    mut alpha: i32,
    beta: i32,
    ply: i32,
    qs_depth: i32,
) -> i32 {
    info.stats.qnodes += 1;

    // Draw detection: repetition and 50-move rule
    let draw_score = if info.root_stm == board.side_to_move { -tp(&CONTEMPT_VAL) } else { tp(&CONTEMPT_VAL) };
    if board.halfmove >= 100 {
        return draw_score;
    }
    // Check for repetition in game history
    let hash = board.hash;
    for undo in board.undo_stack.iter().rev().skip(1).step_by(2) {
        if undo.hash == hash { return draw_score; }
        if undo.halfmove == 0 { break; } // irreversible move
    }

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

        let qs_is_pv = beta - alpha > 1;
        match tt_entry.flag {
            TT_FLAG_EXACT => {
                if !qs_is_pv { return tt_score; }
            }
            TT_FLAG_LOWER => {
                if !qs_is_pv && tt_score >= beta { return tt_score; }
            }
            TT_FLAG_UPPER => {
                if !qs_is_pv && tt_score <= alpha { return tt_score; }
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
            tt_move, ply as usize, qs_checkers, qs_pinned, &info.history, qs_prev_move, qs_pawn_hist_ref,
            &info.moved_piece_stack, &info.moved_to_stack,
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
        if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }
            if !board.make_move(mv) {
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }
                continue;
            }
                    if info.threat_stack.active { let entry = info.threat_stack.current_mut(); entry.delta.clear(); for d in board.threat_deltas.iter() { entry.delta.push(*d); } let ul = board.undo_stack.len(); if ul > 0 { let u = &board.undo_stack[ul-1]; entry.mv = u.mv; if u.mv != crate::types::NO_MOVE { entry.moved_pt = board.mailbox[crate::types::move_to(u.mv) as usize]; entry.moved_color = crate::types::flip_color(board.side_to_move); } } }
            info.tt.prefetch(board.hash);
            move_count += 1;

            let score = -quiescence_with_depth(board, info, -beta, -alpha, ply + 1, qs_depth + 1);
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }

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

        // Store in TT (skip if stopped — partial QS results corrupt TT)
        let store_score = score_to_tt(best_score, ply);
        let flag = if best_score >= beta {
            TT_FLAG_LOWER
        } else if best_score <= alpha_orig {
            TT_FLAG_UPPER
        } else {
            TT_FLAG_EXACT
        };
        if FEAT_TT_STORE.load(Ordering::Relaxed) && !info.stop.load(Ordering::Relaxed) {
            info.tt.store(board.hash, -1, store_score, flag, best_move, -INFINITY, false);
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
    let mut qs_move_count = 0i32;
    let qs_max_caps = tp(&QS_MAX_CAPTURES);

    loop {
        let mv = picker.next(board);
        if mv == NO_MOVE { break; }

        // Move count cutoff: stop searching after N captures (Obsidian: 3)
        qs_move_count += 1;
        if qs_move_count > qs_max_caps {
            break;
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
                if stand_pat + see_value(cap_pt) * tp(&SEE_MATERIAL_SCALE) / 100 + tp(&QS_DELTA_MARGIN) <= alpha {
                    continue;
                }
            }
        }

        // Skip bad captures (SEE below threshold)
        // Negative threshold allows slightly losing captures (e.g. BxN)
        // Obsidian uses -32, Viridithas -141
        if !see_ge(board, mv, tp(&QS_SEE_THRESHOLD)) {
            continue;
        }

        // Build lazy NNUE update
        let qs_moved_pt = board.piece_type_at(move_from(mv));
        let qs_captured_pt = if move_flags(mv) == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(move_to(mv)) };
        let qs_dirty = build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), qs_moved_pt, qs_captured_pt);

        if let Some(acc) = &mut info.nnue_acc { acc.push(qs_dirty); }
        if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }
        if !board.make_move(mv) {
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }
            continue;
        }
                if info.threat_stack.active { let entry = info.threat_stack.current_mut(); entry.delta.clear(); for d in board.threat_deltas.iter() { entry.delta.push(*d); } let ul = board.undo_stack.len(); if ul > 0 { let u = &board.undo_stack[ul-1]; entry.mv = u.mv; if u.mv != crate::types::NO_MOVE { entry.moved_pt = board.mailbox[crate::types::move_to(u.mv) as usize]; entry.moved_color = crate::types::flip_color(board.side_to_move); } } }
        info.tt.prefetch(board.hash);
        let score = -quiescence_with_depth(board, info, -beta, -alpha, ply + 1, qs_depth + 1);
        board.unmake_move();
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }

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

    // Store in TT (skip if stopped — partial QS results corrupt TT)
    let store_score = score_to_tt(best_score, ply);
    let flag = if best_score >= beta {
        TT_FLAG_LOWER
    } else if best_score <= alpha_orig {
        TT_FLAG_UPPER
    } else {
        TT_FLAG_EXACT
    };
    if FEAT_TT_STORE.load(Ordering::Relaxed) && !info.stop.load(Ordering::Relaxed) {
        info.tt.store(board.hash, -1, store_score, flag, best_move, stand_pat, false);
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
        info.auto_discover_nnue();
    }
    let mut total_nodes = 0u64;
    let mut ebf_ln_sum = 0.0f64;
    let mut ebf_count = 0u32;
    let mut total_stats = PruneStats::default();

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

        // Accumulate stats across all positions
        total_stats.tt_cutoffs += info.stats.tt_cutoffs;
        total_stats.tt_near_miss += info.stats.tt_near_miss;
        total_stats.nmp_attempts += info.stats.nmp_attempts;
        total_stats.nmp_cutoffs += info.stats.nmp_cutoffs;
        total_stats.rfp_cutoffs += info.stats.rfp_cutoffs;
        total_stats.lmp_prunes += info.stats.lmp_prunes;
        total_stats.futility_prunes += info.stats.futility_prunes;
        total_stats.history_prunes += info.stats.history_prunes;
        total_stats.see_prunes += info.stats.see_prunes;
        total_stats.probcut_cutoffs += info.stats.probcut_cutoffs;
        total_stats.lmr_searches += info.stats.lmr_searches;
        total_stats.recapture_ext += info.stats.recapture_ext;
        total_stats.qnodes += info.stats.qnodes;
        total_stats.beta_cutoffs += info.stats.beta_cutoffs;
        total_stats.first_move_cutoffs += info.stats.first_move_cutoffs;
        total_stats.cutoff_movecount_sum += info.stats.cutoff_movecount_sum;
        total_stats.cutoff_movecount_sq_sum += info.stats.cutoff_movecount_sq_sum;

        // Accumulate EBF data across all positions
        let max_d = info.completed_depth as usize;
        for d in 5..max_d {
            let prev = info.depth_nodes[d];
            let curr = info.depth_nodes[d + 1];
            if prev > 100 && curr > prev {
                ebf_ln_sum += (curr as f64 / prev as f64).ln();
                ebf_count += 1;
            }
        }
    }

    if !print_stats { return total_nodes; }

    // Print pruning stats (accumulated across all positions)
    let s = &total_stats;
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
    if s.beta_cutoffs > 0 {
        let avg_pos = s.cutoff_movecount_sum as f64 / s.beta_cutoffs as f64;
        let avg_sq = s.cutoff_movecount_sq_sum as f64 / s.beta_cutoffs as f64;
        let first_pct = s.first_move_cutoffs as f64 / s.beta_cutoffs as f64 * 100.0;
        eprintln!("Move ordering:  avg cutoff pos {:.2}, avg pos² {:.1}, first-move {:.1}%",
            avg_pos, avg_sq, first_pct);
    }
    // Effective branching factor: geometric mean of node ratios between consecutive depths
    // Accumulated across all bench positions for a robust estimate
    if ebf_count > 0 {
        let mean_ebf = (ebf_ln_sum / ebf_count as f64).exp();
        eprintln!("EBF (depth 5+): {:.2} (geometric mean, {} transitions across {} positions)",
            mean_ebf, ebf_count, positions.len());
    }

    // Tree shape fingerprint: per-1K-node rates for easy diffing between branches.
    // A change in any of these rates indicates the tree shape has changed,
    // even if total node count is similar. Prune counts can exceed nodes
    // (multiple prunes per node in the move loop), so per-1K is clearer.
    let kn = total_nodes as f64 / 1000.0;
    eprintln!("--- Tree Shape (per 1K nodes) ---");
    eprintln!("TT cutoffs:     {:>6.1}/Kn", s.tt_cutoffs as f64 / kn);
    eprintln!("NMP cutoffs:    {:>6.1}/Kn  ({:.0}% of attempts)", s.nmp_cutoffs as f64 / kn,
        if s.nmp_attempts > 0 { s.nmp_cutoffs as f64 / s.nmp_attempts as f64 * 100.0 } else { 0.0 });
    eprintln!("RFP cutoffs:    {:>6.1}/Kn", s.rfp_cutoffs as f64 / kn);
    eprintln!("LMP prunes:     {:>6.1}/Kn", s.lmp_prunes as f64 / kn);
    eprintln!("Futility:       {:>6.1}/Kn", s.futility_prunes as f64 / kn);
    eprintln!("Hist prune:     {:>6.1}/Kn", s.history_prunes as f64 / kn);
    eprintln!("SEE prune:      {:>6.1}/Kn", s.see_prunes as f64 / kn);
    eprintln!("LMR searches:   {:>6.1}/Kn", s.lmr_searches as f64 / kn);
    eprintln!("QS nodes:       {:>5.1}%", s.qnodes as f64 / total_nodes as f64 * 100.0);
    eprintln!("First-move cut: {:>5.1}%", if s.beta_cutoffs > 0 { s.first_move_cutoffs as f64 / s.beta_cutoffs as f64 * 100.0 } else { 0.0 });

    eprintln!("Total nodes:    {:>8}", total_nodes);

    #[cfg(feature = "profile-materialize")]
    crate::nnue::mat_stats::report();

    #[cfg(feature = "profile-threats")]
    {
        crate::threats::thr_stats::report();
        crate::threats::apply_stats::report();
    }

    total_nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Singular extensions set `info.excluded_move[ply]` during verification
    /// search and MUST clear it after. A leak would silently corrupt the
    /// next search iteration (subsequent SE would skip, or move loops would
    /// skip a random move).
    ///
    /// Audit 2026-04-17: confirmed the set/clear pair at search.rs:2103-2105
    /// has no early-return path between them. This test guards the invariant
    /// against future regressions.
    #[test]
    fn test_excluded_move_cleared_after_search() {
        use crate::board::Board;

        crate::init();
        let mut info = SearchInfo::new(16);
        info.silent = true;

        let mut board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

        let limits = SearchLimits {
            depth: 8, // enough to hit SE at SE_DEPTH threshold
            movetime: 0,
            wtime: 0, btime: 0, winc: 0, binc: 0,
            movestogo: 0, nodes: 0, infinite: false,
        };

        search(&mut board, &mut info, &limits);

        for (i, &mv) in info.excluded_move.iter().enumerate() {
            assert_eq!(
                mv, NO_MOVE,
                "excluded_move[{}] = {} after search — SE verification leaked",
                i, crate::types::move_to_uci(mv)
            );
        }
    }

    /// Correction-history update primitive (`update_corr_entry`) must:
    /// (a) move the entry in the direction of `err * weight`,
    /// (b) respect the bound ±CORR_HIST_LIMIT,
    /// (c) apply proportional gravity (saturates at the bound),
    /// (d) be symmetric for positive vs negative errors (equal magnitude
    ///     updates produce equal magnitude changes from 0).
    #[test]
    fn corr_entry_update_basics() {
        // (d) Symmetry from zero.
        let mut pos = 0i32;
        let mut neg = 0i32;
        update_corr_entry(&mut pos, 4, 5, 4);   // err=+4
        update_corr_entry(&mut neg, -4, 5, 4);  // err=-4
        assert_eq!(pos, -neg, "symmetric updates from zero: pos={}, neg={}", pos, neg);
        assert!(pos > 0, "positive err must raise entry: got {}", pos);

        // (a) Directional.
        let mut e = 0i32;
        update_corr_entry(&mut e, 3, 2, 4);
        assert!(e > 0, "err > 0, weight > 0 → entry must rise, got {}", e);

        // (b) Bounded at ±CORR_HIST_LIMIT.
        let mut e = 0i32;
        for _ in 0..10000 {
            update_corr_entry(&mut e, 1000, 1000, 1); // saturate hard
        }
        assert!(e <= CORR_HIST_LIMIT, "entry must stay ≤ LIMIT, got {}", e);
        assert!(e >= -CORR_HIST_LIMIT, "entry must stay ≥ -LIMIT, got {}", e);

        // (c) Proportional gravity: repeated same-sign updates saturate,
        //     don't grow without bound.
        let mut e = CORR_HIST_LIMIT / 2;
        let before = e;
        update_corr_entry(&mut e, 1, 1, 4);
        let delta = e - before;
        // Small update near saturation should be small.
        assert!(delta.abs() < 4, "near-saturation delta should be tiny, got {}", delta);
    }

    /// Zero err must leave entry unchanged (neither grows nor decays).
    /// If this fails, we're either applying decay-in-error-free case
    /// (bad) or have a sign bug.
    #[test]
    fn corr_entry_zero_err_noop() {
        let mut e = 500i32;
        update_corr_entry(&mut e, 0, 5, 4);
        assert_eq!(e, 500, "zero err must not change entry");

        let mut e = -500i32;
        update_corr_entry(&mut e, 0, 5, 4);
        assert_eq!(e, -500, "zero err must not change negative entry either");
    }

    /// Read/write index symmetry: for every correction-history table,
    /// corrected_eval reads the slot that update_correction_history
    /// writes for the same position. This test populates a table via
    /// a single update, reads via corrected_eval, and verifies the
    /// expected delta appears.
    ///
    /// Using a position with distinctive piece layout so hash
    /// collisions with default zero-state are unlikely.
    #[test]
    fn corr_read_write_index_symmetry() {
        use crate::board::Board;
        crate::init();

        let mut info = SearchInfo::new(16);
        info.silent = true;

        // Distinctive position
        let board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

        let raw = 100;
        // Before any update: corrected == raw (all tables zero).
        let corrected_before = corrected_eval(&info, &board, raw);
        assert_eq!(corrected_before, raw, "zero tables must give corrected == raw");

        // Apply a large positive update at depth=20.
        update_correction_history(&mut info, &board, raw + 400, raw, 20);

        let corrected_after = corrected_eval(&info, &board, raw);
        assert!(
            corrected_after > corrected_before,
            "after positive-err update, corrected eval must rise: before={} after={}",
            corrected_before, corrected_after
        );

        // Reading with a DIFFERENT board that hashes to the same
        // indices is improbable; reading with a fresh board should
        // NOT see the update (different position → different indices).
        let other = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let other_corrected = corrected_eval(&info, &other, raw);
        // For startpos, pawn_hash/np/minor/major keys are entirely
        // different from the test fen, so any match would be a random
        // index collision at 1/16384 probability — extremely unlikely
        // to drift more than ~0.5 cp.
        let drift = (other_corrected - raw).abs();
        assert!(drift < 100,
            "unrelated position should see near-zero drift, got {} (raw {})",
            other_corrected, raw);
    }
}

