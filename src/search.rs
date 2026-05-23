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

/// Maximum ply depth supported by per-SearchInfo arrays.
///
/// Public because MovePicker and History share the cap for fixed-size
/// ply-indexed arrays (killers, pv_table, moved_piece_stack, etc.).
///
/// Tried 128 to lift the iterative-deepening cap to depth 64, but the
/// resulting pv_table grew from ~17 KB to ~67 KB per SearchInfo, spilled
/// L1 on the hot pv-copy path, and regressed STC by ~-13 Elo (OB #664).
/// Keeping 64 — the original crash is fixed by the ply clamp in qsearch
/// + bounds check in MovePicker, not by raising the ceiling.
pub const MAX_PLY: usize = 128;
const INFINITY: i32 = 30000;
// Contempt removed 2026-04-19 (SPRT #508 H1 +2.53).

// Pawn history table size
const PAWN_HIST_SIZE: usize = 512;

// ============================================================================
// Tunable search parameters (exposed as UCI options for SPSA tuning)
// ============================================================================
use std::sync::atomic::AtomicI32;

/// Declare a tunable search parameter with default, min, max, c_end.
/// Single source of truth — used for both the static AtomicI32 and the
/// UCI/SPSA parameter list. c_end is the SPSA end-of-tune perturbation;
/// target >= 1.5 for narrow-range int params (so int boundaries can be
/// crossed) or ~5% of range for wider params.
macro_rules! tunables {
    ( $( ($name:ident, $default:expr, $min:expr, $max:expr, $c_end:expr, $core:expr) ),* $(,)? ) => {
        // Declare each as a pub static AtomicI32
        $( pub static $name: AtomicI32 = AtomicI32::new($default); )*

        /// List of all tunable parameters for UCI/SPSA.
        /// Tuple: (name, &atomic, default, min, max, c_end, is_core).
        ///
        /// `is_core` marks the curated subset that's worth tuning in routine
        /// retunes. Non-core tunables are kept in the source (still loadable
        /// via UCI for full-sweep tunes) but excluded from --core SPSA runs
        /// to improve per-parameter SNR on the meaningful axes.
        pub fn tunable_params() -> Vec<(&'static str, &'static AtomicI32, i32, i32, i32, f32, bool)> {
            vec![
                $( (stringify!($name), &$name, $default, $min, $max, $c_end, $core), )*
            ]
        }
    };
}

tunables!(
    // v9 #784 tune applied (2500 iters, 40010 games, on C8-fix-factor SB800
    // net 1EF1C3E5). Full 77-param sweep against the factor SB800 candidate
    // (in flight as #782 net-vs-net, +2.4 trending H1).
    //
    // Big movers vs prior trunk (#743/#747-derived):
    //   NMP cluster heavy aggressive shift: BASE_R 5→6 (+13%), DEPTH_DIV 4→3
    //     (-21%), EVAL_DIV 104→97 (-7%), VERIFY_DEPTH 8→9, MIN_DEPTH 5→6
    //     (NMP wants more reduction, deeper verification, later activation)
    //   HINDSIGHT_MIN_DEPTH 2→4 (+85%), HIST_PRUNE_MULT 11753→13272 (+13%),
    //     HIST_PRUNE_DEPTH 4→5 (+16%) — pruning generally MORE aggressive
    //   LMR_C_QUIET 120→130, LMR_C_CAP 93→101, LMR_KING_PRESSURE_DIV 4→5,
    //     LMR_THREAT_DIV 4→3 — LMR fully recalibrated
    //   CORR_W_CONT -10%, CORR_HIST_GRAIN_T 9→10 — corrhist reweighting
    //     (CORR_W_MINOR/MAJOR contributions dropped 2026-05-19, were
    //     +14%/+11% in this tune but ablated post-#1318)
    //   ESCAPE_BONUS_Q 12758→14437 (+13%), KNIGHT_FORK_BONUS 8782→9716,
    //     DISCOVERED_ATTACK_BONUS 5808→6672 — tactical bonuses up
    //
    // Overrides applied to SPSA output:
    //   LMR_ENDGAME_PIECES_10X restored to 50 (effective 5) with floor 45
    //     (effective 5). The orphaned restore commit 74666f5 had set it to
    //     5; the _10X migration (855f35b) silently captured the drifted
    //     trunk value 4 → 40. Play-quality load-bearing per
    //     feedback_play_quality_params_narrow_range. SPSA can still
    //     explore 5..=9 within the [45, 90] clamp.
    //
    // Flags for future investigation:
    //   NMP_UNDEFENDED_MAX float-converged at 0.6 (int rounds to 1, no
    //     change); two consecutive tunes have drifted this toward feature-
    //     disable. Candidate for ablation SPRT (set to 0).
    (NMP_BASE_R_10X, 69, 20, 80, 15.0, true),
    // Ceiling lifted from 60 → 200 (audit 2026-05-20): SPSA at 55, 90%
    // from min, only ~9% headroom. Symmetric to a floor pin — gradient
    // clamped at the top. Lifting lets SPSA find the true optimum.
    (NMP_DEPTH_DIV_10X, 61, 10, 200, 15.0, true),
    (NMP_EVAL_DIV, 129, 50, 400, 17.5, true),
    (NMP_EVAL_MAX_10X, 24, 10, 60, 5.0, true),
    (NMP_VERIFY_DEPTH_10X, 105, 40, 200, 20.0, true),
    (RFP_DEPTH, 17, 2, 20, 2.0, true),
    // Floors lifted to 0 (audit 2026-05-20): both pinned within ~10% of floor.
    (RFP_MARGIN_IMP, 40, 0, 150, 6.0, true),
    (RFP_MARGIN_NOIMP, 57, 0, 200, 7.5, true),
    // Futility margin reduced to Reckless scale. At lmr_d=5:
    //   Old: 78 + 160*5 = 878 (Coda 2.4× wider than Reckless 364)
    //   New: 40 + 65*5 = 365 (matches Reckless)
    // Force-more-pruning experiment: Coda was under-pruning at mid-depth
    // where Reckless prunes confidently. SPSA retune-on-branch expected.
    // Floor lifted from 20 → 0 (audit 2026-05-20): pinned at 23, 2% from floor.
    (FUT_BASE, 27, 0, 200, 9.0, true),
    (FUT_PER_DEPTH, 79, 40, 250, 10.5, true),
    (HIST_PRUNE_DEPTH_10X, 10, 10, 80, 15.0, true),
    (HIST_PRUNE_MULT, 10262, 500, 50000, 2475.0, true),
    (SEE_QUIET_MULT, 35, 5, 80, 3.75, true),
    (LMR_HIST_DIV, 8173, 2000, 100000, 4900.0, true),
    // 2026-05-18 audit (outlier #2 deep-dive): capture-LMR was using a
    // step function (±1 at |capt_hist|>2000), while quiet-LMR uses
    // continuous `hist_score / LMR_HIST_DIV`. Obsidian uses continuous
    // `R -= hist/(isQuiet?LmrQuietHistoryDiv:LmrCapHistoryDiv)` with
    // LmrQuietHistoryDiv=9621, LmrCapHistoryDiv=5693 (cap divisor ~60%
    // of quiet — single-source capt_hist needs smaller divisor for
    // equivalent reduction magnitude). Coda's quiet div is 7736; same
    // ratio gives ~4500. Defaulting 5000 as a starting point.
    (LMR_HIST_DIV_CAP, 5000, 1000, 20000, 1500.0, true),
    (LMR_C_QUIET, 140, 40, 300, 13.0, true),
    (LMR_C_CAP, 108, 80, 350, 12.5, true),
    // 2026-05-09 cross-engine port (Tier 5.1): SF gates SE at >=6+ttPv,
    // Reckless at >=5+ttPv. Coda's 4 fires SE at shallower depth where
    // singular_depth is too low to judge singularity reliably. Bumping
    // 4 → 6 first; ttPv add deferred to a follow-up if H1.
    (SE_DEPTH_10X, 45, 40, 200, 20.0, true),
    (ASP_DELTA, 11, 5, 30, 1.5, false),
    (ASP_SCORE_DIV, 33378, 8000, 50000, 2100.0, false),
    // 2026-05-09 cross-engine bisect (Tier 5.3a): SF/Obsidian/Reckless all
    // use LMP_BASE=3 with the same `(BASE + d²)/(2 - improving)` formula.
    // Coda's 9 is 3× consensus at d=1: allows 5-10 quiets vs SF's 2-4.
    // Bisecting 9 → 5 first.
    (LMP_BASE, 9, 1, 15, 2.0, true),
    (LMP_DEPTH, 7, 4, 20, 2.0, true),
    (BAD_NOISY_MARGIN, 80, 30, 150, 6.0, true),
    (PROBCUT_MARGIN, 185, 80, 300, 11.0, true),
    (HINDSIGHT_THRESH, 169, 50, 400, 17.5, true),
    (UNSTABLE_THRESH, 310, 50, 500, 22.5, false),
    (SEE_MATERIAL_SCALE, 215, 30, 300, 13.5, false),
    (QS_DELTA_MARGIN, 367, 100, 500, 20.0, true),
    (QS_SEE_THRESHOLD, -26, -200, 0, 10.0, false),
    (QS_MAX_CAPTURES, 24, 2, 32, 2.0, false),
    (CORR_W_PAWN, 299, 100, 600, 25.0, true),
    // Floor lifted from 50 → 0 (audit 2026-05-20): pinned at 63, 4% from floor.
    (CORR_W_NP, 61, 0, 400, 17.5, true),
    // CORR_W_MINOR / CORR_W_MAJOR were dropped 2026-05-18 (ablated to 0
    // via #1318 H1; minor_key/major_key are strict subsets of
    // non_pawn_key, so the contributions were redundant with np_corr).
    // Tunables and supporting tables removed 2026-05-19 — they sat at
    // weight 0 burning ~5% of every --core SPSA tune's iteration budget
    // on parameters with no gradient.
    //
    // Floor on CORR_W_CONT lifted from 30 → 0 (audit 2026-05-19): SPSA
    // converged 33, ~1% from floor. Lifting allows finding true optimum
    // including disabling cont-corr if SPSA wants. Default unchanged.
    (CORR_W_CONT, 33, 0, 400, 18.5, true),
    (FH_BLEND_DEPTH_10X, 33, 0, 80, 15.0, false),
    // Re-expose 4 hardcoded search constants (audit 2026-05-21).
    // All bench-neutral at current defaults.
    //
    // TT_DAMP_TT_WEIGHT: weight of tt_score in TT-LOWER non-PV cutoff score
    // dampening. Formula: (W*tt_score + beta) / (W+1). Old hardcoded W=3.
    (TT_DAMP_TT_WEIGHT, 3, 1, 10, 0.5, false),
    // PROBCUT_TT_DEPTH_SLACK: TT depth must be >= current depth - SLACK for
    // ProbCut-TT-noshot to consider the entry. Old hardcoded 3.
    (PROBCUT_TT_DEPTH_SLACK, 3, 0, 10, 0.5, false),
    (HIST_BONUS_MULT, 309, 50, 400, 17.5, true),
    (HIST_BONUS_MAX, 1936, 500, 3000, 125.0, true),
    // Shape experiment 1 (Titan's shape_experiments_proposal_2026-04-19):
    // history bonus adopts Stockfish/cap-hist offset shape:
    //   old: min(MAX, MULT * d)
    //   new: clamp(0, MAX, MULT * d - OFFSET)
    // Rationale: at d=5 the old formula saturates at ~1500; d=5 and d=10
    // get the same bonus. New shape with offset 72 (SF's value) gives
    // wider depth discrimination. cap-history already uses the offset
    // shape (CAP_HIST_MULT * d - CAP_HIST_BASE) — main history is the
    // only inconsistent one. Starting offset 72 mirrors SF.
    (HIST_BONUS_OFFSET, 24, 0, 400, 25.0, false),
    (CAP_HIST_MULT, 289, 50, 400, 17.5, true),
    (CAP_HIST_BASE, 42, 0, 200, 10.0, false),
    (CAP_HIST_MAX, 1881, 500, 3000, 125.0, true),
    // BONUS_BOOST_AT removed 2026-05-17: ablation #1277 at [0, 3] H0
    // (+0.3 ±1.0, CI [-0.7, +1.3] at 136K games). Depth-boost trigger
    // confirmed neutral; both call sites updated to drop the +1 clause.
    // numFailHighs multiplicative scaling (#1020 / Starzix T1 #1):
    // bonus = raw + raw * min(num_fail_highs, NFH_CAP) / NFH_DIV.
    // 0..NFH_CAP cascades produce 1.0× .. (1 + NFH_CAP/NFH_DIV)× bonus.
    (NFH_CAP_10X, 31, 10, 60, 10.0, false),
    // Was 47 (tp10→5). Now consumed as FIXED-POINT (stored/10) so SPSA's
    // sub-integer precision is preserved. Default 50 → eff 5.0 ≡ old behavior.
    (NFH_DIV_10X, 50, 20, 120, 10.0, false),
    // Reckless-pattern PV/quiet/correction-aware DEXT margin.
    // Matches SF (search.cpp:1153) and Reckless (search.rs:686-689).
    //
    // dext_margin = DEXT_MARGIN_PV   * is_pv
    //             - DEXT_MARGIN_QUIET * is_tt_quiet
    //             - DEXT_MARGIN_CORR * |corr| / 128
    //             + DEXT_MARGIN_BASE
    //
    // BASE term is Coda-specific: pure Reckless has dext_margin=-16 at
    // non-PV quiet (always fires on singular), which exploded our bench
    // +67% at #804. BASE shifts the non-PV baseline to a positive
    // threshold so default is sane; SPSA explores the basin where
    // pruning compensates (Yin/Yang frame).
    //
    // CORR modulator reduces threshold when correction history has been
    // correcting — extend less on uncertain evals.
    //
    // TRIPLE extension intentionally not included here. Original test
    // (#787 H0, SPSA #792 no basin) showed signal-not-there for Coda's
    // regime; bundling it into #815 dragged the result negative. Tested
    // alone in this branch.
    (DEXT_MARGIN_PV, 155, 50, 400, 15.0, true),
    (DEXT_MARGIN_QUIET, 3, 0, 100, 4.0, true),
    (DEXT_MARGIN_CORR, 21, 0, 64, 3.0, true),
    (DEXT_MARGIN_BASE, 44, -50, 150, 6.0, true),
    (DEXT_CAP, 14, 4, 32, 2.0, true),
    (QUIET_CHECK_BONUS, 14805, 2000, 30000, 1400.0, false),
    (LMR_COMPLEXITY_DIV, 152, 30, 500, 23.5, false),
    (CORR_HIST_DIV, 1559, 256, 4096, 192.0, true),
    (CORR_UPDATE_WEIGHT_MAX, 13, 4, 48, 2.2, true),
    // Was 32 (tp10→3). Now FIXED-POINT. Default 30 → eff 3.0 ≡ old behavior.
    (CORR_BONUS_CAP_DIV_10X, 30, 10, 160, 15.0, false),
    (CORR_HIST_GRAIN_T, 14, 1, 32, 1.55, false),
    // Floor lifted from 10 → 0 (audit 2026-05-19): SPSA converged 25, ~2%
    // from the floor. Lifting allows exploration of looser clamps.
    (CORR_HIST_ERR_MAX_10X, 25, 0, 640, 5.0, false),
    // ESCAPE_BONUS_Q / _MINOR removed 2026-05-17: ablations #1256/#1255
    // H0 at [-3, 3]. Slightly load-bearing (central -0.6/-1.3 to ablate),
    // hardcoded at current SPSA values in movepicker.rs.
    (ESCAPE_BONUS_R, 8181, 3000, 30000, 1350.0, false),
    // ESCAPE_BONUS_Q / _MINOR were hardcoded post-ablation (#1255/#1256
    // H0). Re-exposing as tunables 2026-05-21 — after this session's
    // big cont-hist + NMP + shallow-margin shifts, optimal values may
    // have drifted from the post-ablation snapshot. Bench-neutral at
    // current defaults.
    (ESCAPE_BONUS_Q, 17819, 0, 30000, 1500.0, false),
    (ESCAPE_BONUS_MINOR, 5250, 0, 30000, 1000.0, false),
    // Null-move threat-escape bonus in quiet ordering (was hardcoded 8000).
    (NULL_THREAT_ESCAPE_BONUS, 8000, 0, 30000, 1000.0, false),
    (NMP_KING_ZONE_MAX_10X, 72, 20, 90, 15.0, true),
    // T2.1 (Titan's next_ideas 2026-04-21): undefended-piece NMP skip
    // threshold. Count our pieces with ≥1 enemy attacker AND zero of
    // our own defenders ("hanging"). If count >= this threshold, skip
    // NMP — opponent's free tempo is very likely to exploit the hanger.
    // Fits Titan's W2 pattern (binary signal gating a pruning decision).
    // Default 1 = skip NMP whenever any piece is hanging.
    (NMP_UNDEFENDED_MAX_10X, 17, 0, 50, 10.0, true),
    // T2.3 (next_ideas_2026-04-21): mobility-delta quiet-ordering weight.
    // Bonus applied in movepicker quiets = (to_mobility - from_mobility) × this.
    // Default 32 = ±256 typical range, additive to history (~1000s scale).
    (MOBILITY_DELTA_WEIGHT, 34, 0, 256, 8.0, false),
    (PROBCUT_KING_ZONE_MAX_10X, 58, 20, 90, 15.0, true),
    // Was 38 (tp10→4). Now FIXED-POINT. Default 40 → eff 4.0 ≡ old behavior.
    (LMR_THREAT_DIV_10X, 40, 10, 50, 15.0, true),
    // Was 68 (tp10→7). Now FIXED-POINT. Default 70 → eff 7.0 ≡ old behavior.
    (LMR_KING_PRESSURE_DIV_10X, 70, 20, 90, 15.0, true),
    (FUT_THREATS_MARGIN, 23, 0, 200, 10.0, true),
    (DISCOVERED_ATTACK_BONUS, 3534, 0, 30000, 1500.0, false),
    // BATTERY_BONUS removed 2026-05-17: ablation #1278 at [0, 3] H0
    // (+0.2 ±1.1, CI [-0.9, +1.3] at 114K games). Feature confirmed
    // neutral; movepicker.rs T1.4 battery-bonus block removed.
    // QSEE_BONUS removed 2026-05-17: ablation #1257 at [-3, 3] H0
    // (-2.1 ±3.0, central +2 Elo from the feature — load-bearing).
    // Feature kept, hardcoded at SPSA value in movepicker.rs.
    // SE_KING_PRESSURE_MARGIN removed 2026-05-15: tune at _10X precision
    // (range -5..+30, direct /10 scaling) confirmed optimum is genuinely 0.
    // Historical conflicting reads (0.22 vs 1-2 across tunes) were SPSA
    // noise on integer-rounded values. Direction closed.
    // xray-SE: widen singular test margin when TT move is from an x-ray
    // blocker square (moving it uncovers our slider's attack on an enemy).
    // Signal already delivered +52 in movepicker (#502). Flat bonus
    // subtracted from singular_beta → easier to judge singular → more
    // extensions for tactically significant moves.
    (SE_XRAY_BLOCKER_MARGIN_10X, 47, 0, 400, 20.0, true),
    // 2026-05-19 audit: floor was pinned at 10 (=1.0 effective), preventing
    // SPSA from exploring below 1× even though SPSA had repeatedly driven
    // the value to the floor across tunes. Widened to allow 0× (full disable)
    // so SPSA can find the genuine optimum. CLAUDE.md previously claimed
    // "3× in move ordering" — stale; corrected to "1× current SPSA basin".
    (CONT_HIST_MULT_10X, 19, 0, 80, 15.0, true),
    // Pawn-history weight in quiet move ordering. Was hardcoded at 1×;
    // making tunable lets SPSA find the right pawn-structure weighting
    // relative to main/cont/etc. Default 10 = eff 1× (bench-neutral).
    // core: false — newly exposed, not yet validated Elo-positive (mini-tune
    // #1385 was flat). Keep out of --core to avoid loose-knob false gradients.
    (PAWN_HIST_MULT_10X, 10, 0, 80, 10.0, false),
    (KNIGHT_FORK_BONUS, 8722, 0, 20000, 1000.0, false),
    // LMR endgame gate: skip LMR when popcount(occupied) <= this value.
    // +5.0 Elo H1 in SPRT #583. Fixes endgame-conversion blunders where
    // LMR over-reduces king-restriction queen moves that complete mates.
    //
    // NARROW RANGE [5, 9]: correctness-load-bearing per Lichess play-quality
    // (rook on open board over-reduced as "late"). 2026-04-22 SPSA #660
    // drifted to 4; restore commit 74666f5 set it to 5 with floor 5 but
    // that branch was never merged. The 2026-05-10 _10X migration (855f35b)
    // then captured the drifted trunk value 4 → 40 (effective 4), so the
    // intent was lost. Restored here as 50 (effective 5) with floor 45
    // (also effective 5 via tp10 rounding); SPSA can explore 5..=9.
    (LMR_ENDGAME_PIECES_10X, 50, 45, 90, 15.0, true),
    // --- Previously-hardcoded pruning depth gates, now tunable ---
    // Per 2026-04-24 strategy: at our strength/eval regime, optimal
    // depth caps/gates are sensitive to eval quality and will need
    // re-tuning after each net change. Exposing them as SPSA-tunables
    // lets retunes re-calibrate without code changes. Defaults match
    // the previously-hardcoded values so this commit is bench-neutral.
    //
    // Future retune-on-branch cycles will sweep these with the
    // eval+pruning co-tune; expect meaningful movement as net quality
    // changes.
    // IIR floor lifted from 20 → 5 (audit 2026-05-19): tune #743 drove
    // value to 20 (eff depth 2). With floor=20 SPSA can't explore below
    // depth 2; lifting to 5 (eff 0.5) lets SPSA find effective optimum,
    // including "fire at any depth ≥ 1".
    (IIR_MIN_DEPTH_10X, 20, 5, 100, 15.0, true),          // was hardcoded 4; tune #743 converged to 2 (strong signal)
    // ProbCut floor lifted from 30 → 10 (audit 2026-05-19): SPSA at 32,
    // ~2% from floor. Lifting to 10 (eff 1) allows exploration of more
    // aggressive ProbCut activation.
    (PROBCUT_MIN_DEPTH_10X, 32, 10, 120, 15.0, true),     // was hardcoded 5 (ProbCut activation gate)
    (SEE_CAP_DEPTH, 6, 3, 15, 1.5, true),         // was hardcoded 6 (SEE capture prune depth cap)
    (BAD_NOISY_DEPTH, 8, 4, 15, 1.5, true),       // was hardcoded 4 (BNFP depth cap)
    // Second pass — additional gates exposed for the feature-utility
    // audit tune. Widened ranges allow SPSA to reach disable-endpoint
    // values where appropriate (per feedback_spsa_as_feature_utility_diagnostic).
    (NMP_MIN_DEPTH_10X, 59, 20, 200, 15.0, true),              // was hardcoded 3 (NMP activation gate, 2 sites)
    // Floor lifted from 10 → 0 (audit 2026-05-20): pinned at 25, 8% from floor.
    (HINDSIGHT_MIN_DEPTH_10X, 23, 0, 200, 15.0, true),        // was hardcoded 2 (hindsight reduction gate)
    (TT_CUTOFF_HALFMOVE_MAX, 89, 50, 100, 3.0, false),  // was hardcoded 90 (TT cutoff halfmove gate, 5 sites)
);

// Demoted loose knobs (2026-05-22 cross-tune analysis): SPSA drift dominated
// signal, so removed from SPSA surface to improve SNR for the rest. Values
// frozen at their pre-demotion defaults. Bench-neutral; UCI-invisible.
pub static FH_BLEND_OFFSET: AtomicI32 = AtomicI32::new(1);
pub static SE_TT_DEPTH_SLACK: AtomicI32 = AtomicI32::new(3);
pub static MVV_CAP_MULT: AtomicI32 = AtomicI32::new(28);
pub static FUT_LMR_DEPTH: AtomicI32 = AtomicI32::new(15);

/// Get a tunable parameter value (inline for hot paths)
#[inline(always)]
fn tp(param: &AtomicI32) -> i32 {
    param.load(Ordering::Relaxed)
}

/// Read a `_10X`-scaled tunable, returning the effective integer value
/// (round-half-away-from-zero of stored/10). Tunables with the `_10X`
/// suffix store 10× their effective value so SPSA can express decimal
/// precision and retain decimal progress across tune cycles. See
/// memory feedback_floor_pin_tunables_cross_recipe.md for rationale.
#[inline(always)]
pub fn tp10(param: &AtomicI32) -> i32 {
    let v = param.load(Ordering::Relaxed);
    if v >= 0 { (v + 5) / 10 } else { (v - 5) / 10 }
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
    /// Minimum think time to enforce on a movetime search. Normally 0 (pure
    /// movetime). Ponderhit fresh-searches set this to `inc - overhead` so
    /// they don't instant-emit on TT-cached positions (which would stockpile
    /// the clock on every move). See `search()`'s movetime branch.
    pub movetime_floor: u64,
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
            movetime_floor: 0,
        }
    }
}

/// Pruning counters for diagnostics.
#[derive(Default)]
pub struct PruneStats {
    pub tt_probes: u64,
    pub tt_hits: u64,
    pub tt_cross_gen_hits: u64,
    pub tt_cross_gen_cutoffs: u64,
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
    /// Number of quiet moves reaching the history-prune gate (passed all
    /// preceding checks). hist_prune_eligible >= history_prunes.
    pub hist_prune_eligible: u64,
    /// Distribution of `score / (HIST_PRUNE_MULT * depth)` at the gate.
    /// Buckets: 0=>=1.0, 1=[0,1), 2=[-0.5,0), 3=[-1,-0.5), 4=[-1.5,-1),
    /// 5=[-2,-1.5), 6=[-3,-2), 7=<-3.0. Buckets 4-7 = "would have fired".
    pub hist_prune_ratio_buckets: [u64; 8],
    /// Cont-hist read magnitude distribution per offset, sampled once per
    /// hist-prune-eligible move. Offsets [ply-1, ply-2, ply-4, ply-6].
    /// Mag buckets: 0=[0,200), 1=[200,1k), 2=[1k,5k), 3=[5k,10k), 4=[10k+).
    pub cont_hist_mag_buckets: [[u64; 5]; 4],
    /// Cont-hist write counts and magnitude sums per offset.
    /// Indexed [ply-1, ply-2, ply-4, ply-6].
    pub cont_hist_writes: [u64; 4],
    pub cont_hist_write_mag_sum: [u64; 4],
    /// Main-history (4D threat-aware) cell-density at end of search.
    /// Indexed [ft*2 + tt]: 0=[0][0], 1=[0][1], 2=[1][0], 3=[1][1].
    /// Each value: count of cells with |entry|>1000 in that bucket
    /// (0..4096 cells per bucket).
    pub main_hist_density: [u64; 4],
    /// Read counts of main-history per (ft, tt) bucket, sampled at
    /// hist-prune gate. Same indexing as main_hist_density.
    pub main_hist_bucket_reads: [u64; 4],
    /// Hypothetical hist-prune fire rates with different cont-hist offset
    /// combinations in the score:
    ///   [0] = main + cont[1] + pawn (CURRENT gate sources)
    ///   [1] = main + cont[1,2] + pawn
    ///   [2] = main + cont[1,2,4] + pawn
    ///   [3] = main + cont[1,2,4,6] + pawn
    /// Marginal fire-rate gain of adding each deeper offset.
    pub hist_prune_what_if_fires: [u64; 4],
    /// Sign-agreement between main_hist and sum-of-cont-hist at gate:
    /// [0]=both-positive, [1]=both-negative, [2]=disagree, [3]=one-zero.
    pub cont_hist_sign_buckets: [u64; 4],
    /// Per-offset dominance at gate (which |contribution| is largest).
    pub cont_hist_dominant_offset: [u64; 4],
    pub see_prunes: u64,
    pub probcut_cutoffs: u64,
    pub lmr_searches: u64,
    pub recapture_ext: u64,
    pub singular_ext: u64,
    pub double_ext: u64,
    pub negative_ext: u64,
    pub multicut: u64,
    pub qnodes: u64,
    pub beta_cutoffs: u64,
    pub first_move_cutoffs: u64,
    // Move ordering quality: sum of move_count² at beta cutoff (lower = better ordering)
    pub cutoff_movecount_sq_sum: u64,
    pub cutoff_movecount_sum: u64,
}

/// Forced-move detection state (Viridithas pattern, set by `detect_forced_move`).
/// Once a position is classified at the root, the result is sticky for the rest of
/// the search — both the verification's TT pollution and the result itself are
/// monotonic. `None` is the default; once `Weak` or `Strong` is observed, the TM
/// multiplier scales down accordingly.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ForcedState {
    None,
    /// Best alternative was within `SLIGHTLY_FORCED_MARGIN` of TT score at depth ≥ 12.
    /// Multiplier reduces soft by ~37% (Viridithas: 627/1000).
    Weak,
    /// Best alternative collapsed by ≥ `VERY_FORCED_MARGIN` at depth ≥ 8.
    /// Multiplier reduces soft by ~61% (Viridithas: 386/1000).
    Strong,
}

/// Search state for one thread.
pub struct SearchInfo {
    pub nodes: u64,
    pub global_nodes: std::sync::Arc<AtomicU64>,  // aggregate nodes across SMP threads
    pub silent: bool,  // suppress UCI output (for datagen)
    pub stats: PruneStats,
    // Eval-path decomposition counters (see docs/coda_vs_reckless_nps_*.md).
    // `stats_tt_static_eval_hits` counts nodes where we used the TT's
    // cached static_eval and did NOT call NNUE. The NNUE counters live on
    // `nnue_acc` (full rebuilds vs incremental updates vs computed skips).
    pub stats_tt_static_eval_hits: u64,
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
    /// Cumulative count of root best-move changes between iterations,
    /// reset at search start. Drives a Reckless/Stockfish-style upward
    /// multiplier on tactical/unstable positions (Phase 1 TM redesign,
    /// docs/tm_redesign_phase1_2026-05-19.md). Pairs with `tm_best_stable`
    /// (which tracks CONSECUTIVE stable iterations and reduces time);
    /// this tracks TOTAL changes and increases time.
    tm_best_move_changes: u32,
    /// Forced-move detection state (Viridithas pattern). Set after an ID iteration
    /// at the root when `detect_forced_move` finds that excluding the current best
    /// move collapses the alternative score by a meaningful margin. Sticky once
    /// set — verification only fires while state == None and depth ≥ 8. Drives a
    /// downward multiplier in the TM scale; this is the position-intrinsic signal
    /// Coda was missing (every other signal is search-progress-derived).
    tm_forced_state: ForcedState,
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
    /// Ponderhit: shared atomic soft deadline (absolute ms since start_time).
    /// 0 = unset. Written by UCI thread on ponderhit; read by the ID loop to
    /// enable dynamic TM post-ponderhit (so stable positions don't burn the
    /// full hard deadline at deep iterations).
    pub ponderhit_soft: std::sync::Arc<AtomicU64>,
    /// Ponderhit: minimum think time post-ponderhit (relative duration in
    /// ms — typically ≈ increment-overhead). Floors the dynamically-scaled
    /// soft so we still spend some time after a ponderhit even when the
    /// position is rock-solid (prevents instant-emit).
    pub ponderhit_floor: std::sync::Arc<AtomicU64>,
    /// Time-management baseline: elapsed-ms at which the soft budget starts
    /// counting. 0 for normal `go` (TM starts at search start). Set to the
    /// elapsed-at-ponderhit value when post-ponderhit dynamic TM kicks in,
    /// so soft/floor are interpreted as durations from the ponderhit moment
    /// rather than from the original `go ponder` start.
    pub tm_baseline: u64,
    /// Completed search depth (shared atomic). Updated by search thread after
    /// each completed iteration. Read by UCI thread on ponderhit to scale budget.
    pub ponder_depth: std::sync::Arc<AtomicU64>,
    pub sel_depth: i32,
    pub last_score: i32,
    /// Root side-to-move (was used for contempt; retained for potential future use)
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
        Self::new_with_tt(std::sync::Arc::new(TT::new(tt_mb)))
    }

    /// Construct a SearchInfo with a pre-existing shared TT. Used by helper
    /// threads to avoid allocating a throwaway 1 MB TT (which prints a
    /// misleading "TT 1 MB" info string before the shared TT is swapped in).
    pub fn new_with_tt(tt: std::sync::Arc<TT>) -> Self {
        SearchInfo {
            nodes: 0,
            global_nodes: std::sync::Arc::new(AtomicU64::new(0)),
            silent: false,
            stats: PruneStats::default(),
            stats_tt_static_eval_hits: 0,
            tt,
            history: alloc_zeroed_box(),
            stop: std::sync::Arc::new(AtomicBool::new(false)),
            start_time: Instant::now(),
            time_limit: 0,
            max_depth: 100,
            max_nodes: 0,
            move_overhead: 100,
            tm_prev_best: NO_MOVE,
            tm_best_move_changes: 0,
            tm_forced_state: ForcedState::None,
            tm_prev_score: 0,
            tm_best_stable: 0,
            tm_has_data: false,
            soft_limit: 0,
            hard_limit: 0,
            soft_floor: 0,
            root_move_nodes: alloc_zeroed_box(),
            ponderhit_time: std::sync::Arc::new(AtomicU64::new(0)),
            ponderhit_soft: std::sync::Arc::new(AtomicU64::new(0)),
            ponderhit_floor: std::sync::Arc::new(AtomicU64::new(0)),
            tm_baseline: 0,
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
        // Use the shared TT directly — avoids allocating a throwaway 1 MB
        // TT and the misleading "TT 1 MB" info string that prints before
        // the swap. Same pattern as create_helper_info.
        let mut si = Self::new_with_tt(tt);
        si.stop = stop;
        si.nnue_net = nnue_net;
        si
    }

    /// Load an NNUE network.
    pub fn load_nnue(&mut self, path: &str) -> Result<(), String> {
        let net = crate::nnue::NNUENet::load(path)?;
        let acc = crate::nnue::NNUEAccumulator::new(net.hidden_size);
        // Activate threat stack if net has threat features.
        //
        // C8 audit LIKELY #36: previously only the `has_threats` branch
        // touched threat_stack. On net swap from v9 (has_threats=true) to
        // v5 (has_threats=false), the existing threat_stack would keep
        // `active=true` even though the new net doesn't use threats —
        // search would try to run threat computation against a net that
        // doesn't consume it. Reset unconditionally first; activate only
        // when the new net needs it.
        self.threat_stack = crate::threat_accum::ThreatStack::new(net.hidden_size);
        if net.has_threats {
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
        // Material scaling: dampen eval in low-material endgames (SF/Stormphrax/
        // Halogen/Integral pattern — non-pawn material only). Pawn-up endgames
        // shouldn't get dampened toward zero; pawns retain decisive value at
        // low non-pawn material counts.
        // N=422, B=422, R=642, Q=1015
        let material = {
            let knights = popcount(board.pieces[KNIGHT as usize]) as i32 * 422;
            let bishops = popcount(board.pieces[BISHOP as usize]) as i32 * 422;
            let rooks = popcount(board.pieces[ROOK as usize]) as i32 * 642;
            let queens = popcount(board.pieces[QUEEN as usize]) as i32 * 1015;
            knights + bishops + rooks + queens
        };
        // `eval()` now returns the halfmove-INDEPENDENT score (material
        // scaling only). 50-move scaling is applied at every consumption
        // site (via `apply_halfmove_scale`) using the *current* halfmove.
        //
        // Rationale: TT stores the static eval and gets probed again at
        // potentially very different halfmove values. If we scaled inside
        // `eval()` the scaling factor would be frozen into the TT entry at
        // write time, and re-probing the same position at a higher halfmove
        // would use a stale scale. With aggressive scaling (our current
        // formula) that bakes in gross errors and wipes out the
        // correction — hence SPRT #610 showed −8 Elo at 1000 games before
        // we caught this. The fix is structural: keep TT storage
        // halfmove-independent, apply scale freshly on read.
        score * (22400 + material) / 32 / 1024
    }
}

/// Scale a raw (halfmove-independent) eval toward zero as the halfmove
/// clock approaches the 100-ply draw horizon.
///
/// Formula: `score * (100 - clamp(hm, 0, 100)) / 100`. At `hm=0` returns
/// `score` unchanged; at `hm=100` returns `0`. Callers apply this at the
/// *point of use*, never before storing to TT — see the comment in
/// `SearchInfo::eval`.
///
/// Consensus-aligned with Obsidian/Reckless-style `(100 - hm) / 100`,
/// which unlike the previous `(200 - hm) / 200` actually reaches zero
/// at the draw cliff rather than topping out at 0.5×.
#[inline]
fn apply_halfmove_scale(score: i32, halfmove: u16) -> i32 {
    // Leave sentinel scores untouched so downstream comparisons with
    // `-INFINITY` / `MATE_SCORE - ply` keep their absolute magnitudes.
    if score <= -INFINITY + 1 || score.abs() >= MATE_SCORE - 100 {
        return score;
    }
    let hm = (halfmove as i32).min(100);
    score * (100 - hm) / 100
}

/// Build a DirtyPiece for lazy NNUE accumulator update.
/// `us`/`them` are the sides BEFORE the move.
/// `net`: NNUE net whose king-bucket layout determines bucket/mirror
///   changes on king moves. Must be the same net that will later apply
///   the DirtyPiece to the accumulator; using the wrong net here produces
///   silently-wrong "refresh needed" decisions.
#[inline]
pub fn build_dirty_piece(
    mv: Move,
    us: u8,
    them: u8,
    moved_pt: u8,
    captured_pt: u8,
    net: &crate::nnue::NNUENet,
) -> DirtyPiece {
    let from = move_from(mv);
    let to = move_to(mv);
    let flags = move_flags(mv);

    // King moves: check if bucket+mirror change for the moving side's perspective
    if moved_pt == KING {
        let mut from_ks = from as usize;
        let mut to_ks = to as usize;
        if us == BLACK { from_ks ^= 56; to_ks ^= 56; }

        let from_bucket = net.king_bucket(from_ks);
        let to_bucket = net.king_bucket(to_ks);
        let from_mirror = net.king_mirror(from_ks);
        let to_mirror = net.king_mirror(to_ks);

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

/// Compute the correction value alone (the centipawn delta corrhist would
/// apply to raw eval). Used by SE-margin formulas that want to gate
/// extension confidence on |correction| — when the eval has been
/// drifting (large |corr|), reduce extension thresholds so we
/// extend less on uncertain evals (Reckless pattern).
#[inline]
fn correction_value(info: &SearchInfo, board: &Board) -> i32 {
    let stm = board.side_to_move as usize;
    let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
    let pawn_corr = info.pawn_corr[stm][pawn_idx] as i64;
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
    let white_np_corr = info.np_corr[stm][WHITE as usize][white_np_idx] as i64;
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
    let black_np_corr = info.np_corr[stm][BLACK as usize][black_np_idx] as i64;
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
    let total_corr = (pawn_corr * tp(&CORR_W_PAWN) as i64 + white_np_corr * tp(&CORR_W_NP) as i64 + black_np_corr * tp(&CORR_W_NP) as i64
        + cont_corr * tp(&CORR_W_CONT) as i64) / tp(&CORR_HIST_DIV) as i64;
    (total_corr as i32) / tp(&CORR_HIST_GRAIN_T)
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

    // Weighted blend: pawn, whiteNP, blackNP, cont (minor/major dropped 2026-05-19)
    let total_corr = (pawn_corr * tp(&CORR_W_PAWN) as i64 + white_np_corr * tp(&CORR_W_NP) as i64 + black_np_corr * tp(&CORR_W_NP) as i64
        + cont_corr * tp(&CORR_W_CONT) as i64) / tp(&CORR_HIST_DIV) as i64;
    let adjusted = raw_eval + (total_corr as i32) / tp(&CORR_HIST_GRAIN_T);
    adjusted.clamp(-MATE_SCORE + 100, MATE_SCORE - 100)
}

/// Update correction history entry with gravity.
fn update_corr_entry(entry: &mut i32, err: i32, weight: i32, cap_div_10x: i32) {
    // Proportional gravity (consensus: every top engine uses this)
    // Self-limiting: values near the limit get pulled back harder
    // cap_div_10x is stored × 10 (fixed-point); cap = LIMIT * 10 / cap_div_10x.
    let cap = CORR_HIST_LIMIT * 10 / cap_div_10x.max(1);
    let bonus = (err * weight).clamp(-cap, cap);
    *entry += bonus - *entry * bonus.abs() / CORR_HIST_LIMIT;
    *entry = (*entry).clamp(-CORR_HIST_LIMIT, CORR_HIST_LIMIT);
}

/// Update all correction history tables.
fn update_correction_history(info: &mut SearchInfo, board: &Board, search_score: i32, raw_eval: i32, depth: i32) {
    let err_max = tp10(&CORR_HIST_ERR_MAX_10X);
    let err = (search_score - raw_eval).clamp(-err_max, err_max);
    let weight = (depth + 1).min(tp(&CORR_UPDATE_WEIGHT_MAX));
    // Pass raw stored value; consumer treats it as fixed-point (×10).
    let cap_div = CORR_BONUS_CAP_DIV_10X.load(Ordering::Relaxed);
    let stm = board.side_to_move as usize;

    // Pawn correction
    let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.pawn_corr[stm][pawn_idx], err, weight, cap_div);

    // Non-pawn corrections (per color)
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.np_corr[stm][WHITE as usize][white_np_idx], err, weight, cap_div);
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.np_corr[stm][BLACK as usize][black_np_idx], err, weight, cap_div);

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
    // Use the shared TT directly (avoids allocating a throwaway 1 MB TT
    // and the misleading "TT 1 MB" info string that prints before swap).
    let mut helper = SearchInfo::new_with_tt(main.tt.clone());
    helper.stop = main.stop.clone();         // share the same stop flag
    // C8 audit LIKELY #35: share ponderhit_time so helpers respect the
    // ponderhit deadline set by the UCI thread. Previously helpers kept
    // their own AtomicU64 stuck at 0, so they ignored the ponderhit
    // deadline and only stopped when main set the shared stop flag —
    // burning CPU for the grace window on every ponderhit.
    helper.ponderhit_time = main.ponderhit_time.clone();
    helper.ponderhit_soft = main.ponderhit_soft.clone();
    helper.ponderhit_floor = main.ponderhit_floor.clone();
    helper.global_nodes = main.global_nodes.clone(); // share node counter
    helper.silent = true;                // helpers don't output UCI
    helper.nnue_net = main.nnue_net.clone(); // share NNUE weights (read-only)
    // Create fresh NNUE accumulator for the helper
    if let Some(net) = &helper.nnue_net {
        helper.nnue_acc = Some(crate::nnue::NNUEAccumulator::new(net.hidden_size));
        // Mirror main's threat_stack for v9 nets — helpers must evaluate
        // consistently with main, otherwise shared-TT entries disagree and
        // the search diverges badly at T>1.
        if net.has_threats {
            helper.threat_stack = crate::threat_accum::ThreatStack::new(net.hidden_size);
            helper.threat_stack.active = true;
        }
    }
    helper.time_limit = 0; // helpers don't do time management
    helper.move_overhead = main.move_overhead;
    helper.root_stm = main.root_stm; // kept in sync for potential future stm-aware features
    helper.syzygy = main.syzygy.clone(); // share tablebases (read-only)

    // Seed helpers with the main thread's accumulated, aged history.
    // Cross-engine consensus (SF/Reckless/Obsidian/Alexandria/Viridithas/
    // PlentyChess): helpers never start from zero — they either share
    // history with main lockless, persist it across `go` calls, or get
    // a copy. Coda spawns helpers per `go`, so we copy from main
    // (which has just been aged ×0.80 at the top of `start_search`).
    //
    // Without this, helpers spend the first few iterations re-learning
    // move ordering from scratch, generating TT entries with worse
    // ordering than main; this both wastes their work AND poisons
    // shared-TT ordering, costing most of the potential SMP Elo.
    helper.history.copy_from(&main.history);
    // Pawn-hist and the correction-history tables are large
    // (pawn_hist alone is ~13 MB) and main aggressively clears
    // correction history at the top of every search anyway, so a copy
    // there has no value. Main history (which includes the
    // load-bearing 4D main + cont_hist) is the one that matters.

    helper
}

/// Compute time-management budgets from clock state. Returns
/// (soft, hard, soft_floor) all in milliseconds.
///
/// Shared by `start_search` (initial allocation on `go wtime/btime`)
/// and the UCI ponderhit handler (allocation after ponderhit when the
/// ponder search is already running). Keeping both paths on the same
/// helper prevents the formulas from drifting — previously the
/// handler reimplemented the formula and silently lost emergency-mode
/// reduction at time<1s and movestogo-aware caps.
///
/// Inputs: `our_time` and `our_inc` are wtime/winc or btime/binc in
/// ms. `movestogo` is 0 for sudden death. `overhead` is the
/// MoveOverhead UCI option (default 100ms).
pub fn compute_tm_budgets(
    our_time: u64,
    our_inc: u64,
    movestogo: u32,
    overhead: u64,
    fullmove: u16,
) -> (u64, u64, u64) {
    let time_left = our_time.saturating_sub(overhead).max(1);
    let moves_left = if movestogo > 0 { movestogo as u64 } else { 25 };

    // Soft allocation: time/movesLeft + 80% of increment.
    let mut soft = time_left / moves_left + our_inc * 4 / 5;

    // Phase 4 v2 (2026-05-21): ply-aware soft scaling.
    //
    // Diagnosis from main/phase2/phase4 RR + clock-drain analyzer
    // (scripts/tm_clock_drain.py): Coda spends 7.1s/move avg on moves 1-5
    // at 120+1, vs SF's ~2-3s/move. Even main has this overspend pattern.
    // Result: clock at 50% by move 10 vs SF at 78%, no reserve for
    // middlegame.
    //
    // SF's optScale = 0.012112 + (ply+3.22713)^0.46866 × c — soft target
    // GROWS sub-linearly with ply, low in opening, high in middlegame.
    // Coda's `time/movesLeft + 0.8*inc` is uniform regardless of phase.
    //
    // Linear ramp piecewise approximation (full-move number):
    //   fm=1-5:   soft × 0.4  (sharp early discipline)
    //   fm=6-10:  soft × 0.6
    //   fm=11-15: soft × 0.85
    //   fm=16+:   soft × 1.0  (full middlegame TM)
    //
    // Only soft is scaled — hard cap stays as catastrophe ceiling.
    // Multipliers (Phase 1's bmc, stability, nodes, score) still push
    // tactical opening positions higher above this lower base; routine
    // opening positions emit at the reduced soft target.
    //
    // Skip scaling when movestogo>0: the movestogo path already
    // computes soft from remaining moves directly.
    if movestogo == 0 {
        let phase_x100: u64 = if fullmove <= 5 { 40 }
                              else if fullmove <= 10 { 60 }
                              else if fullmove <= 15 { 85 }
                              else { 100 };
        soft = soft * phase_x100 / 100;
    }

    // Cap soft allocation: scale with moves remaining.
    // movestogo=1: 90%, movestogo=2: 85%, sudden death (effective 25): 50%.
    let max_pct = if movestogo > 0 {
        (95 - movestogo as u64 * 5).max(30).min(90)
    } else {
        50
    };
    let max_alloc = time_left * max_pct / 100;
    if soft > max_alloc { soft = max_alloc; }

    // Emergency: below 1 second, be very conservative.
    if time_left < 1000 {
        let mut emergency = time_left / 10;
        if our_inc > 0 && our_inc < emergency { emergency = our_inc; }
        if emergency < 10 { emergency = 10; }
        if soft > emergency { soft = emergency; }
    }

    // Floor at 10ms.
    if soft < 10 { soft = 10; }

    // Save base soft before any clamp to hard. Hard uses base_soft, NOT
    // a dynamically-scaled soft — dynamic factors at runtime can scale
    // soft by up to ~2.5× (stability=0 × node_fraction=2.23), so hard
    // derived from scaled soft would let max = soft × 2.5 × 3 = soft ×
    // 7.5, recreating the overspend problem.
    let base_soft = soft;

    // Hard limit:
    //   movestogo>0: 2× base_soft, capped at a movestogo-scaled fraction
    //   sudden death: 3× base_soft.
    let mut hard = if movestogo > 0 {
        let hard_raw = base_soft * 2;
        let hard_pct = (95 - movestogo as u64 * 10).max(30).min(90);
        let mtg_cap = time_left * hard_pct / 100;
        hard_raw.min(mtg_cap)
    } else {
        base_soft * 3
    };

    // Universal hard cap (sudden-death only). Phase 2 TM redesign
    // (docs/tm_redesign_phase2_2026-05-20.md, follows
    // docs/tm_redesign.md Issue 1):
    //
    // The old `time/20 + inc` cap pinned sudden-death hard to ~5% of
    // clock + 1 increment. Phase 1 self-play gauntlets across
    // 30+0.5/60+1/10+0.1 (200 games each, 2026-05-19) showed that
    // every upward multiplier (bmc_factor, nodes_factor, score_factor)
    // saturated against this ceiling on tactical positions, producing
    // a uniform spend pattern indistinguishable from main. Phase 2
    // widens the cap so the multipliers can express variance, guarded
    // by an absolute minimum-reserve floor that prevents the v1 blitz-
    // catastrophe regression (40-second moves at move 3).
    //
    // Two safety layers:
    //   1. TC-aware multiplier (`mult_cap`) — looser at classical
    //      where deep think pays, tighter at bullet where one bad
    //      move loses. Combined with a per-TC max-single-move
    //      percentage of remaining time.
    //   2. Minimum-reserve floor (`max_consumable`) — never spend so
    //      much that remaining clock would drop below
    //      max(TM_RESERVE_INC × inc, TM_RESERVE_ABS_MS). The safety
    //      that lets the cap widen without recreating v1's blitz
    //      catastrophe.
    if movestogo == 0 {
        // Inc-aware per-move estimate: 1m+5s plays nothing like 1m+0s.
        // The natural per-move budget is base/25 + inc (you regain the
        // inc each move). Without including inc, a 1m+5s game was being
        // classified as "blitz" (60s/25 = 2.4s → 9% pct_cap = 5.4s),
        // and soft was clamped to a 5.4s hard cap while soft_floor at
        // ~4.9s left a <500ms variance band — observed on lichess as
        // "always uses 5s per move" at 1m+5s.
        let estimated_spm_ms = time_left / 25 + our_inc;
        // TC-aware hard multiplier × 10 (integer math). Conservative
        // initial values; SPSA-tunable later.
        let hard_mult_x10: u64 = if estimated_spm_ms < 2000 { 20 }       // bullet 2.0×
                                 else if estimated_spm_ms < 5000 { 25 }   // blitz 2.5×
                                 else if estimated_spm_ms < 15000 { 30 }  // rapid 3.0×
                                 else { 40 };                              // classical 4.0×
        // Max-single-move percentage of remaining clock. Aligned with
        // doc design table — bullet/blitz tight, rapid/classical loose.
        let max_single_pct: u64 = if estimated_spm_ms < 2000 { 8 }        // bullet 8%
                                  else if estimated_spm_ms < 5000 { 9 }    // blitz 9%
                                  else if estimated_spm_ms < 15000 { 12 }  // rapid 12%
                                  else { 15 };                              // classical 15%

        let mult_cap = base_soft * hard_mult_x10 / 10;
        let pct_cap = time_left * max_single_pct / 100;
        let new_hard = mult_cap.min(pct_cap);

        // Minimum-reserve floor: never spend such that remaining clock
        // drops below max(TM_RESERVE_INC × inc, TM_RESERVE_ABS_MS).
        // K_ABS = 2s is below the doc's 3s proposal to avoid strangling
        // short STC games (10+0.1 would lose 30% of its clock otherwise).
        const TM_RESERVE_INC_MULT: u64 = 5;
        const TM_RESERVE_ABS_MS: u64 = 2000;
        let min_reserve = (TM_RESERVE_INC_MULT * our_inc).max(TM_RESERVE_ABS_MS);
        let max_consumable = time_left.saturating_sub(min_reserve);
        let new_hard = new_hard.min(max_consumable);

        if hard > new_hard {
            hard = new_hard;
        }
        // Phase 4 v2 (2026-05-21): also scale hard cap by phase factor in
        // opening. Soft scaling alone wasn't enough because the dynamic
        // TM multipliers (especially stability_factor starting at 1.71
        // when 0 stable iterations seen) compensated for the reduced soft
        // by pushing scale × soft back up toward original. Hard-cap
        // scaling makes the early ceiling an absolute, multiplier-proof
        // bound.
        let phase_x100: u64 = if fullmove <= 5 { 40 }
                              else if fullmove <= 10 { 60 }
                              else if fullmove <= 15 { 85 }
                              else { 100 };
        hard = hard * phase_x100 / 100;

        // Final absolute safety: never spend > 3/4 of remaining time on
        // a single move (preserved from old formula).
        if hard > time_left * 3 / 4 {
            hard = time_left * 3 / 4;
        }
    }

    // Clamp soft to hard. C8 audit LIKELY #28: must clamp BEFORE
    // returning; otherwise downstream TM code could see soft > hard
    // and try to spend more than the absolute cap.
    if soft > hard { soft = hard; }

    // Soft floor: prevents instant-emit stockpile in stable endgames
    // (lichess PZ7pCyrx) without crushing downward variance. Set at
    // half the increment (overhead-adjusted) so dynamic stability cuts
    // can take spend down to ~50% of inc, but no further. The old
    // full-inc floor (`our_inc - overhead`) collapsed the variance band
    // at high-inc TCs — 1m+5s was floored at 4.9s and capped near 5.4s,
    // leaving no room for position-aware variance. Capped at hard.
    // Zero when (inc - overhead) ≤ 1.
    let soft_floor = (our_inc.saturating_sub(overhead) / 2).min(hard);

    (soft, hard, soft_floor)
}

/// Run Lazy SMP search: main thread + N-1 helper threads.
pub fn search_smp(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits, threads: usize) -> Move {
    // C8 audit LIKELY #37: advance TT generation here (before spawning
    // helpers) rather than inside search(). Previously helpers could
    // start writing TT entries with the old generation in the microsecond
    // window between spawn and main's new_search() call, leaving them
    // looking freshest in replacement. Main's search() no longer bumps;
    // single-thread path bumps here too for consistency.
    info.tt.new_search();

    if threads <= 1 {
        info.global_nodes.store(0, Ordering::Relaxed);
        return search(board, info, limits);
    }

    // Reset shared state.
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
            movetime_floor: 0, // helpers don't need the floor — only main sleeps
        };

        handles.push(std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
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

                let mv = search_helper(&mut helper_board, &mut helper, &helper_limits, thread_id);
                // Return (nodes, best_move, score, depth) for vote aggregation
                (helper.nodes, mv, helper.last_score, helper.completed_depth)
            }).expect("Failed to spawn SMP helper"));
    }

    // Main thread searches normally
    let main_move = search(board, info, limits);
    let main_score = info.last_score;
    let main_depth = info.completed_depth;

    // Signal all helpers to stop
    info.stop.store(true, Ordering::Relaxed);

    // Collect helper results
    let mut total_nodes = info.nodes;
    let mut thread_results: Vec<(Move, i32, i32)> = Vec::with_capacity(threads);
    if main_move != NO_MOVE && main_depth > 0 {
        thread_results.push((main_move, main_score, main_depth));
    }
    for h in handles {
        if let Ok((helper_nodes, mv, score, depth)) = h.join() {
            total_nodes += helper_nodes;
            if mv != NO_MOVE && depth > 0 {
                thread_results.push((mv, score, depth));
            }
        }
    }
    info.nodes = total_nodes;

    // Vote-based best-move selection (SF/Obsidian/Plenty pattern).
    // weight = depth * (score - min_score + 14). The +14 keeps the worst-
    // scored thread's vote nonzero so depth still matters for tied scores;
    // multiplying by depth makes shallow helpers count less than deep ones.
    if thread_results.len() <= 1 {
        return main_move;
    }
    let min_score = thread_results.iter().map(|(_, s, _)| *s).min().unwrap();
    let mut votes: Vec<(Move, i64)> = Vec::with_capacity(thread_results.len());
    for (mv, score, depth) in &thread_results {
        let weight = *depth as i64 * (*score as i64 - min_score as i64 + 14);
        if let Some(entry) = votes.iter_mut().find(|(m, _)| *m == *mv) {
            entry.1 += weight;
        } else {
            votes.push((*mv, weight));
        }
    }
    votes.iter().max_by_key(|(_, w)| *w).map(|(m, _)| *m).unwrap_or(main_move)
}

/// Helper thread search — full aspiration ID loop matching the main
/// thread's `search()`, just silent (no UCI output, no TM).
///
/// Previously this was a stripped-down `negamax(-INF, +INF)` per depth
/// with no aspiration, no score carry, an empty history table, and a
/// `thread_id % 2` depth offset. Cross-engine review (SF/Reckless/
/// Obsidian/Alexandria/Viridithas/PlentyChess) showed every reference
/// engine runs full aspiration in helpers — aspiration-window variance
/// + slight asynchrony IS the diversity mechanism, not depth offsets.
/// Helpers with full windows search far more nodes per depth than they
/// need to and contribute worse-ordered TT entries.
///
/// History is seeded from main in `create_helper_info` — see comment
/// there. We deliberately do NOT clear it here.
fn search_helper(board: &mut Board, info: &mut SearchInfo, _limits: &SearchLimits, _thread_id: usize) -> Move {
    init_feature_flags();

    // History was just seeded from main in create_helper_info — do
    // NOT clear it here. Reset only per-search scratch state.
    info.stats = PruneStats::default();
    info.static_evals = [0; MAX_PLY + 1];
    info.reductions = [0; MAX_PLY + 1];
    info.excluded_move = [NO_MOVE; MAX_PLY + 1];
    info.moved_piece_stack = [0; MAX_PLY + 1];
    info.double_ext_count = [0; MAX_PLY + 1];
    info.moved_to_stack = [0; MAX_PLY + 1];
    info.pv_table = [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1];
    info.pv_len = [0; MAX_PLY + 1];
    info.nodes = 0;
    info.tm_has_data = false;
    info.tm_best_stable = 0;
    info.tm_best_move_changes = 0;
    info.tm_forced_state = ForcedState::None;

    // Mirror search()'s threat setup — helpers must evaluate consistently
    // with main or shared-TT entries disagree and search diverges at T>1.
    board.generate_threat_deltas = info.nnue_net.as_ref().map_or(false, |n| n.has_threats);
    if info.threat_stack.active {
        info.threat_stack.reset();
        if let Some(ref net) = info.nnue_net {
            info.threat_stack.refresh(&net.threat_weights, net.num_threat_features, board, WHITE);
            info.threat_stack.refresh(&net.threat_weights, net.num_threat_features, board, BLACK);
        }
    }

    let root_legal = generate_legal_moves(board);
    let mut best_move = if root_legal.len > 0 { root_legal.get(0) } else { NO_MOVE };

    // Iterative deepening with aspiration windows — same flow as main
    // `search()` but no UCI output and no TM decisions. Helpers stop
    // when main sets the shared stop flag.
    let effective_max = info.max_depth.min(MAX_PLY as i32 / 2);
    let mut prev_score = 0i32;
    for depth in 1..=effective_max {
        if info.stop.load(Ordering::Relaxed) { break; }

        let score;

        // Aspiration windows (skip for mate scores) — mirrors search().
        if depth >= 4 && prev_score > -MATE_SCORE + 100 && prev_score < MATE_SCORE - 100 {
            let avg = prev_score;
            let mut delta = tp(&ASP_DELTA) + (avg as i64 * avg as i64 / tp(&ASP_SCORE_DIV) as i64) as i32;
            let mut alpha = (prev_score - delta).max(-INFINITY);
            let mut beta = (prev_score + delta).min(INFINITY);
            let mut asp_depth = depth;
            #[allow(unused_assignments)]
            let mut asp_result = prev_score;
            loop {
                let result = negamax(board, info, alpha, beta, asp_depth, 0, false);
                if info.stop.load(Ordering::Relaxed) {
                    asp_result = result;
                    break;
                }
                if result <= alpha {
                    beta = (3 * alpha + 5 * beta) / 8;
                    alpha = (result - delta).max(-INFINITY);
                } else if result >= beta {
                    alpha = (5 * alpha + 3 * beta) / 8;
                    beta = (result + delta).min(INFINITY);
                    asp_depth = (asp_depth - 1).max(1);
                } else {
                    asp_result = result;
                    break;
                }
                delta += delta / 2;
            }
            score = asp_result;
        } else {
            score = negamax(board, info, -INFINITY, INFINITY, depth, 0, false);
        }

        if info.stop.load(Ordering::Relaxed) { break; }

        if info.pv_len[0] > 0 {
            best_move = info.pv_table[0][0];
        }
        prev_score = score;
        info.last_score = score;
        info.completed_depth = depth;
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
    info.tm_best_move_changes = 0;
    info.tm_forced_state = ForcedState::None;
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

    // C6 (2026-04-22 audit): SearchInfo persists across `go` commands.
    // Without explicit reset, stale soft_limit/hard_limit from a prior
    // `go wtime/btime` leak into subsequent `go movetime`, `go depth`, or
    // `go nodes` — the dynamic-TM gate then scales them and can break the
    // ID loop early. Zero all four unconditionally up-front; each branch
    // below sets only the ones it needs.
    info.time_limit = 0;
    info.soft_limit = 0;
    info.hard_limit = 0;
    info.soft_floor = 0;
    info.tm_baseline = 0;

    if limits.infinite {
        // Already zero above.
    } else if limits.movetime > 0 {
        info.time_limit = limits.movetime;
        // Respect caller-supplied minimum think time (ponderhit fresh-search uses
        // this to enforce the increment floor; plain `go movetime` callers leave
        // it at 0 so they get exactly the movetime they asked for).
        info.soft_floor = limits.movetime_floor.min(limits.movetime);
    } else if our_time > 0 {
        let (soft, hard, soft_floor) =
            compute_tm_budgets(our_time, our_inc, limits.movestogo, info.move_overhead, board.fullmove);
        info.soft_limit = soft;
        info.hard_limit = hard;
        info.soft_floor = soft_floor;
        info.time_limit = hard; // search uses hard as absolute limit
        info.tm_has_data = false;
        info.tm_best_stable = 0;
        info.tm_best_move_changes = 0;
    } else if !limits.infinite {
        // No clock info (e.g. `go depth N` or `go nodes N`). Already zeroed
        // above; explicit reset kept for clarity.
        info.time_limit = 0;
    }

    info.max_depth = if limits.depth > 0 { limits.depth } else { MAX_PLY as i32 / 2 };
    info.max_nodes = limits.nodes;

    // TT generation is advanced by the entry-point caller (search_smp or
    // datagen), not here — see C8 audit LIKELY #37 fix.

    let mut best_move = NO_MOVE;
    let mut prev_score = 0i32;

    // Stable PV snapshot. Updated only at the end of a *completed* iteration.
    // On a mid-iteration interrupt (should_stop fires inside negamax) we
    // restore from this so `best_move` and `pv_table[0]` stay consistent —
    // otherwise the bestmove emit can pair the prior iteration's best_move
    // with the current iteration's *partial* pv_table[0][1], producing a
    // ponder move that doesn't apply to the actual position-after-best
    // (lichess oeZ7KRUt forfeit, 2026-04-26).
    let mut stable_pv_len: usize = 0;
    let mut stable_pv: [Move; MAX_PLY + 1] = [NO_MOVE; MAX_PLY + 1];

    // Get a fallback move and keep the legal list for final validation
    let root_legal = generate_legal_moves(board);
    if root_legal.len > 0 {
        best_move = root_legal.get(0);
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
        // Post-ponderhit dynamic TM setup. If UCI just stored a soft deadline
        // AND the search has no soft_limit yet (i.e. this started as
        // `go ponder` with no time budget), arm dynamic TM from here onward.
        // Without this, the loop only honours the hard deadline and burns
        // the full ~5s at 60+2 even on stable positions where a 2-3s emit
        // would suffice. The floor (≈ inc-overhead) prevents instant-emit
        // when stability has been confidently held through ponder.
        let ph_soft = info.ponderhit_soft.load(std::sync::atomic::Ordering::Relaxed);
        if ph_soft > 0 && info.soft_limit == 0 {
            let now = info.start_time.elapsed().as_millis() as u64;
            let soft_remaining = ph_soft.saturating_sub(now).max(1);
            let hard_remaining = if ph > now { ph - now } else { soft_remaining };
            let hard_remaining = hard_remaining.max(soft_remaining);
            let floor = info.ponderhit_floor.load(std::sync::atomic::Ordering::Relaxed)
                .min(soft_remaining);
            info.tm_baseline = now;
            info.soft_limit = soft_remaining;
            info.hard_limit = hard_remaining;
            info.soft_floor = floor;
        }
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
            if info.should_stop() {
                // Mid-iteration interrupt — restore the last completed iteration's
                // PV so pv_table[0] stays in sync with `best_move`.
                if stable_pv_len > 0 {
                    info.pv_len[0] = stable_pv_len;
                    for i in 0..stable_pv_len { info.pv_table[0][i] = stable_pv[i]; }
                }
                break;
            }
        } else {
            score = negamax(board, info, -INFINITY, INFINITY, depth, 0, false);
            if info.should_stop() {
                if stable_pv_len > 0 {
                    info.pv_len[0] = stable_pv_len;
                    for i in 0..stable_pv_len { info.pv_table[0][i] = stable_pv[i]; }
                }
                break;
            }
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
                let m = root_legal.get(i);
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
                    let m = root_legal.get(i);
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

        // Snapshot the completed iteration's pv_table[0] so a future
        // mid-iteration interrupt can restore consistency between best_move
        // and pv_table[0]. See comment at stable_pv declaration.
        stable_pv_len = info.pv_len[0].min(stable_pv.len());
        for i in 0..stable_pv_len {
            stable_pv[i] = info.pv_table[0][i];
        }

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
                        let m = pv_legal.get(i);
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

        // Track best-move stability and score trend on every iteration so
        // that ponder iterations accumulate TM state — when ponderhit fires
        // mid-deep-search, dynamic TM (below) can immediately see "best move
        // has been stable for N iterations" and scale down accordingly. With
        // tracking gated behind `soft_limit > 0`, ponderhit started cold
        // (tm_best_stable = 0, stability_factor = 1.71) and the dynamic
        // adjustment couldn't bite.
        let score_drop = if depth >= 4 {
            if info.tm_has_data {
                let bm_from = move_from(best_move);
                let bm_to = move_to(best_move);
                let prev_from = move_from(info.tm_prev_best);
                let prev_to = move_to(info.tm_prev_best);
                if bm_from == prev_from && bm_to == prev_to {
                    info.tm_best_stable += 1;
                } else {
                    info.tm_best_stable = 0;
                    // Phase 1: cumulative count of root best-move changes since
                    // search start. Drives an upward multiplier on tactically
                    // unstable positions (Reckless `1 + changes/4`, Stockfish
                    // `1.096 + 2.29 * totBestMoveChanges` patterns).
                    info.tm_best_move_changes = info.tm_best_move_changes.saturating_add(1);
                }
            }
            let drop = if info.tm_has_data && !is_mate_score(prev_score) && !is_mate_score(info.tm_prev_score) {
                info.tm_prev_score - prev_score
            } else {
                0
            };
            info.tm_prev_best = best_move;
            info.tm_prev_score = prev_score;
            info.tm_has_data = true;
            drop
        } else {
            0
        };

        // Forced-move detection (Viridithas pattern). Once-per-search verification
        // that the chosen best move is meaningfully better than all alternatives.
        // Fires at depth boundaries 8+ (state == None), excludes best_move at root,
        // and runs a narrow-window search at reduced depth. If the alternative
        // collapses by `margin` or more, the position is "forced" and the TM
        // multiplier scales soft DOWN — this is the position-intrinsic signal
        // Coda was missing (every other signal is search-progress-derived,
        // correlated with stability_factor).
        //
        // - Strong (depth 8-11, margin 400cp): soft × 0.386
        // - Weak   (depth 12+,   margin 170cp): soft × 0.627
        //
        // Sticky once set — the verification itself is expensive (depth ~3-5
        // re-search with NMP/RFP/probcut gated by excluded_move) so we run it
        // at most once per search. Skip for mate scores and early depths.
        // TC gate (Phase 6b, 2026-05-22): skip the detector when the floor
        // already occupies ≥ 1/3 of the soft budget. At high-inc TCs (e.g.
        // 60+5 → floor 2.45s vs soft 6.4s = 38%) the detector's downward
        // multiplier can't actually push adjusted_soft below the floor by a
        // meaningful amount — verification cost is paid but actual spend
        // barely changes. Local 60+5 RR (n=170) showed phase6 (with detector)
        // 13 Elo worse than phase6a (floor only) at that TC. At low-inc TCs
        // the floor fraction stays small and the detector pays back (LTC
        // SPRT delta phase6 − phase6a = +2.8 Elo).
        //
        // Behaviour table:
        // 60+5: floor/soft = 0.38 → skip
        // 60+1: floor/soft = 0.14 → fire
        // LTC:  floor/soft = 0.00 → fire
        let floor_dominates = info.soft_floor * 3 >= info.soft_limit;
        if info.tm_forced_state == ForcedState::None
            && depth >= 8
            && best_move != NO_MOVE
            && info.soft_limit > 0
            && !floor_dominates
            && !info.should_stop()
            && !is_mate_score(prev_score)
        {
            const FORCED_MARGIN_WEAK: i32 = 170;
            const FORCED_MARGIN_STRONG: i32 = 400;
            let margin = if depth >= 12 { FORCED_MARGIN_WEAK } else { FORCED_MARGIN_STRONG };
            let r_beta = (prev_score - margin).max(-MATE_SCORE + 1);
            // Viridithas r_depth: (min(12, depth-1) - 1) / 2 — caps verification at depth 5.
            let r_depth = (depth.min(13) - 2) / 2;

            // Save PV state — the verification call at ply=0 will overwrite
            // pv_table[0] with the alternative line, which corrupts the
            // bestmove path. Save and restore around the call.
            let saved_pv_len = info.pv_len[0];
            let mut saved_pv: [Move; MAX_PLY + 1] = [NO_MOVE; MAX_PLY + 1];
            for i in 0..saved_pv_len { saved_pv[i] = info.pv_table[0][i]; }

            info.excluded_move[0] = best_move;
            let value = negamax(board, info, r_beta - 1, r_beta, r_depth, 0, false);
            info.excluded_move[0] = NO_MOVE;

            // Restore PV state regardless of stop flag (the saved PV is the
            // current iteration's completed result).
            info.pv_len[0] = saved_pv_len;
            for i in 0..saved_pv_len { info.pv_table[0][i] = saved_pv[i]; }

            if !info.stop.load(Ordering::Relaxed) && value < r_beta {
                info.tm_forced_state = if depth >= 12 {
                    ForcedState::Weak
                } else {
                    ForcedState::Strong
                };
            }
        }

        // Dynamic time management: 3-factor model (Obsidian/Clarity pattern)
        // Combines node fraction, best-move stability, and score trend.
        if info.soft_limit > 0 && depth >= 4 && !info.should_stop() {
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

            // Factor 4: Best-move-changes upward boost (Phase 1 TM redesign).
            // Mirrors Reckless `1.0 + changes/4.0` and SF `1.096 + 2.29 *
            // bestMoveChanges` patterns — the upward instability signal that
            // Coda was structurally missing. Clamped at 2.5 so it can't
            // multiplicatively explode on pathological positions. Combined
            // with the other factors and bounded by hard_limit downstream,
            // so the catastrophe ceiling is unchanged.
            //
            // tm_best_move_changes is the cumulative count of root best-move
            // flips between iterations since search start, reset at `go`.
            let bmc_factor = (1.0 + info.tm_best_move_changes as f64 / 4.0).min(2.5);

            // Factor 5: Forced-move downward boost (Viridithas pattern,
            // Phase 6 TM redesign). When the verification above has
            // classified the position as forced, scale down decisively.
            // Numbers verbatim from Viridithas (per-mille → fraction):
            //   Strong (depth 8-11):  0.386 — best alternative was -400cp
            //   Weak   (depth 12+):   0.627 — best alternative was -170cp
            // This is the orthogonal downward signal: stability/score-factor
            // both REACT to the search settling, while forced-state CAUSES
            // a single discrete drop based on whether other moves are
            // verifiably worse. Tying time-spend to position-intrinsic
            // shape decorrelates adjacent-move spend (the autocorrelation
            // gap to top engines).
            let forced_factor = match info.tm_forced_state {
                ForcedState::Strong => 0.386,
                ForcedState::Weak   => 0.627,
                ForcedState::None   => 1.0,
            };

            // Combined: all five factors multiply against the soft limit.
            // adjusted_soft is downstream-clamped to hard_limit, so this
            // factor pushes us toward the existing hard cap on tactical
            // positions but cannot exceed it.
            let scale = nodes_factor * stability_factor * score_factor * bmc_factor * forced_factor;

            // Check if we should stop at the soft limit.
            // Floor at soft_floor (≈ increment) so stability cuts in stable
            // endgames can't produce clock-growing instant emits.
            let adjusted_soft = (info.soft_limit as f64 * scale) as u64;
            let adjusted_soft = adjusted_soft.max(info.soft_floor).min(info.hard_limit);
            // Subtract tm_baseline so soft is measured from the TM-start
            // moment, not search start. tm_baseline is 0 for normal `go`
            // (unchanged behaviour); set to elapsed-at-ponderhit when
            // post-ponderhit dynamic TM arms above.
            let elapsed_since_tm = elapsed.saturating_sub(info.tm_baseline);
            if elapsed_since_tm >= adjusted_soft {
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
    //
    // C8 audit LIKELY #29: set the shared stop flag BEFORE the sleep so
    // helper threads stop searching immediately rather than burning CPU
    // through the entire stockpile-prevention window. Previously helpers
    // kept running until hitting their own hard_limit or main unblocked,
    // wasting tens-hundreds of ms of CPU per ponderhit grace window at
    // blitz+inc. Main thread already has its best move, just waiting to
    // emit.
    if info.soft_floor > 0 && !info.stop.load(Ordering::Relaxed) {
        info.stop.store(true, Ordering::Relaxed);
        loop {
            let elapsed = info.start_time.elapsed().as_millis() as u64;
            // Floor is a duration measured from tm_baseline (0 for normal
            // `go`; elapsed-at-ponderhit for post-ponderhit dynamic TM).
            let elapsed_since_tm = elapsed.saturating_sub(info.tm_baseline);
            if elapsed_since_tm >= info.soft_floor { break; }
            let remaining = info.soft_floor - elapsed_since_tm;
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

    // Drawn-position detection must run BEFORE the MAX_PLY guard. At ply
    // >= MAX_PLY in drawn-material or repetition positions, the fall-back
    // `apply_halfmove_scale(info.eval(board), halfmove)` returns a possibly-
    // nonzero scaled eval — eval doesn't know about insufficient material
    // or repetition (only halfmove via the scale itself handles 50mr).
    // Found via correctness audit 2026-05-23, adjacent to the prior
    // MAX_PLY=64→128 fix family.
    if ply > 0 {
        let draw_score: i32 = 0;
        if board.halfmove >= 100 {
            return draw_score;
        }
        if board.is_insufficient_material() {
            return draw_score;
        }
        // Repetition scan (bounded by min(halfmove, plies_from_null) to avoid
        // walking across null-move boundaries).
        let stack_len = board.undo_stack.len();
        let scan_limit = (board.halfmove as usize).min(board.plies_from_null as usize);
        let limit = scan_limit.min(stack_len);
        let mut i = 2usize;
        while i <= limit {
            if board.undo_stack[stack_len - i].hash == board.hash {
                return draw_score;
            }
            i += 2;
        }
    }

    // Guard against stack overflow — only reached for non-drawn positions
    // at ply >= MAX_PLY.
    if ply_u >= MAX_PLY {
        return apply_halfmove_scale(info.eval(board), board.halfmove);
    }

    // C8 audit LIKELY #3: reset reductions slot at node entry so NMP and
    // any other pre-move-loop child call reads "no prior reduction", not
    // a sibling's stale LMR value from an earlier visit to this ply.
    if ply_u <= MAX_PLY {
        info.reductions[ply_u] = 0;
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


    // Draw detection moved above the MAX_PLY guard (2026-05-23 audit fix
    // for ply >= MAX_PLY in drawn positions). Note: kept the broader
    // contempt + jitter context comments at the new location for history.

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
    let tt_cur_gen = info.tt.current_generation();
    let tt_cross_gen = tt_hit && tt_entry.generation != tt_cur_gen;
    info.stats.tt_probes += 1;
    if tt_hit {
        info.stats.tt_hits += 1;
        if tt_cross_gen {
            info.stats.tt_cross_gen_hits += 1;
        }
    }

    // Sticky PV flag: once a position is searched as PV, it stays PV in the TT.
    // Used to reduce LMR for moves that lead to historically important positions.
    let tt_pv = is_pv || (tt_hit && tt_entry.tt_pv);

    if tt_hit {
        tt_move = tt_entry.best_move;

        if info.excluded_move[ply_u] == NO_MOVE && ply > 0 {
            let tt_depth = tt_entry.depth;
            let tt_score = score_from_tt(tt_entry.score, ply);

            // P2 (halfmove-gated TT cutoff): TT scores are stored without halfmove
            // context. Near the 50-move cliff a cached mate-in-N may be unreachable,
            // and a stored bound may be over/understated by the time we revisit.
            // Gate ALL return-from-TT paths (direct + bounds-narrow collapse +
            // near-miss + QS) on halfmove < 90. Window-narrowing is still applied —
            // it only biases the search, while returning stale tt_score is unsafe.
            let halfmove_ok = (board.halfmove as i32) < tp(&TT_CUTOFF_HALFMOVE_MAX);
            if tt_depth >= depth && FEAT_TT_CUTOFF.load(Ordering::Relaxed) {
                // Unified TT cutoff with node-type guard (Alexandria pattern):
                // At non-PV nodes, accept TT cutoff when:
                // - cut_node matches score direction (cut expects fail-high, all expects fail-low)
                // - TT bound type matches (LOWER for fail-high, UPPER for fail-low)
                let score_above_beta = tt_score >= beta;
                let bound_matches = if score_above_beta {
                    tt_entry.flag == TT_FLAG_LOWER || tt_entry.flag == TT_FLAG_EXACT
                } else {
                    tt_entry.flag == TT_FLAG_UPPER || tt_entry.flag == TT_FLAG_EXACT
                };
                if !is_pv && cut_node == score_above_beta && bound_matches
                    && halfmove_ok
                {
                    info.stats.tt_cutoffs += 1;
                    if tt_cross_gen {
                        info.stats.tt_cross_gen_cutoffs += 1;
                    }
                    // Defence-in-depth: validate tt_move is fully legal before
                    // stuffing it into pv_table. Diagnosed during PV_PONDER_BUG
                    // chase as a path that *could* plant an illegal move (hash
                    // collision, torn-write surviving XOR) — empirically never
                    // fires, but the cost is O(1) and the failure mode is a
                    // forfeited game on lichess (oeZ7KRUt 2026-04-26).
                    if tt_move != NO_MOVE && ply_u <= MAX_PLY
                        && crate::movepicker::is_pseudo_legal(board, tt_move)
                        && board.is_legal(tt_move, board.pinned(), board.checkers())
                    {
                        info.pv_table[ply_u][0] = tt_move;
                        info.pv_len[ply_u] = 1;
                    } else if ply_u <= MAX_PLY {
                        info.pv_len[ply_u] = 0;
                    }
                    // TT cutoff cont-hist malus: penalize opponent's last quiet move
                    // in context of our move before that (Alexandria pattern).
                    // "Your move led to a position we already know is lost for you."
                    //
                    // C8 audit LIKELY #6: read pieces from moved_piece_stack (set
                    // pre-move, captures pre-promotion pawn) rather than
                    // board.piece_at(to) (post-move, reports promoted piece for
                    // promotions). Write-side (beta-cutoff bonuses) uses
                    // moved_piece_stack; the old asymmetry meant malus on promotion
                    // moves landed in the queen/rook bin where reads never look.
                    let stack_len = board.undo_stack.len();
                    if score_above_beta && stack_len >= 2 && ply_u >= 2 {
                        let opp_undo = &board.undo_stack[stack_len - 1];
                        let our_undo = &board.undo_stack[stack_len - 2];
                        if opp_undo.mv != NO_MOVE && opp_undo.captured == NO_PIECE_TYPE
                            && our_undo.mv != NO_MOVE
                        {
                            let opp_gp = info.moved_piece_stack[ply_u - 1] as usize;
                            let our_gp = info.moved_piece_stack[ply_u - 2] as usize;
                            let opp_to = info.moved_to_stack[ply_u - 1] as usize;
                            let our_to = info.moved_to_stack[ply_u - 2] as usize;
                            if opp_gp > 0 && opp_gp < 13
                                && our_gp > 0 && our_gp < 13
                                && opp_to < 64 && our_to < 64
                            {
                                let malus = -((155 * depth).min(385));
                                History::update_cont_history(
                                    &mut info.history.cont_hist[our_gp][our_to][opp_gp][opp_to],
                                    malus,
                                );
                            }
                        }
                    }
                    // P3: downgrade stored mate if 50mr will fire before mate.
                    return downgrade_50mr_mate(tt_score, ply, board.halfmove);
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

                if alpha >= beta && halfmove_ok {
                    if tt_move != NO_MOVE {
                        info.stats.tt_cutoffs += 1;
                        if tt_cross_gen {
                            info.stats.tt_cross_gen_cutoffs += 1;
                        }
                        // Defence-in-depth: validate tt_move (see note at first cutoff site).
                        if ply_u <= MAX_PLY
                            && crate::movepicker::is_pseudo_legal(board, tt_move)
                            && board.is_legal(tt_move, board.pinned(), board.checkers())
                        {
                            info.pv_table[ply_u][0] = tt_move;
                            info.pv_len[ply_u] = 1;
                        } else if ply_u <= MAX_PLY {
                            info.pv_len[ply_u] = 0;
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
                        let w = tp(&TT_DAMP_TT_WEIGHT);
                        return (w * tt_score + beta) / (w + 1);
                    }
                    // P3: downgrade stored mate if 50mr will fire before mate.
                    return downgrade_50mr_mate(tt_score, ply, board.halfmove);
                }
            } else if tt_depth >= depth - 1
                && beta - alpha_orig == 1
                && tt_score > -(MATE_SCORE - 100) && tt_score < MATE_SCORE - 100
                && FEAT_TT_NEARMISS.load(Ordering::Relaxed)
                && halfmove_ok
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

    // Compute static eval for pruning and LMR improving detection.
    //
    // Three variables — same value at different processing stages:
    //   raw_eval    : halfmove-INDEPENDENT  (what goes in/out of TT, also
    //                                        passed to corrhist *update*)
    //   scaled_eval : halfmove-scaled, pre-correction (corrhist sees this as
    //                                                   its input base)
    //   static_eval : scaled + corrected (what pruning/LMR decisions use)
    //
    // Keeping these separate fixes the TT staleness bug where a position
    // first seen at hm=20 stored an eval scaled against hm=20, then
    // revisited at hm=85 used the stale-scaled value — exposed by the
    // aggressive `(100-hm)/100` scaling (SPRT #610, flat at -8 Elo after
    // 1000g). TT now stores raw_eval; every consumer scales fresh.
    let mut static_eval = -INFINITY;
    let mut raw_eval = -INFINITY;
    let mut scaled_eval = -INFINITY;
    let mut improving = false;
    if !in_check {
        // Consumer threshold matches pack_data's static_eval clamp range
        // (-4095..4095). Stores that pass -INFINITY (in-check positions
        // where eval is undefined) get clamped to -4095; we reject that
        // value here and recompute. The legitimate-eval-at-exactly-(-4095)
        // false positive case (~-40 pawns) is rare enough that re-eval
        // is harmless.
        if tt_hit && tt_entry.static_eval > -4095 {
            raw_eval = tt_entry.static_eval;
            info.stats_tt_static_eval_hits += 1;
        } else {
            raw_eval = info.eval(board);
            // Eval-only TT writeback: when we paid for an NNUE eval AND
            // there's no existing TT entry, seed the TT with static_eval
            // so later visits of this position (from different move
            // orders or ID re-searches) skip the NNUE call.
            //
            // Reckless pattern (`search.rs:425`). Phase-2 lever from
            // `docs/coda_vs_reckless_nps_2026-04-23.md`: reduces
            // evals/node (Coda 0.677 vs Reckless 0.520).
            //
            // Safety:
            //   - Gated on `!tt_hit` so we never overwrite a real entry
            //     with a shallow stub.
            //   - depth=-2 means `tt_depth >= depth` is false for any
            //     real search depth, so this entry never triggers TT
            //     cutoffs or alpha/beta window narrowing.
            //   - flag=TT_FLAG_UPPER + score=-INFINITY makes any
            //     score read trivially rejected.
            //   - tt_move=NO_MOVE preserves IIR behaviour at this
            //     position on later visits.
            //   - tt_pv carries current node's is_pv context so PV
            //     propagation is correct on re-visit.
            if !tt_hit && FEAT_TT_STORE.load(Ordering::Relaxed) {
                info.tt.store(board.hash, -2, -INFINITY, TT_FLAG_UPPER, NO_MOVE, raw_eval, is_pv);
            }
        }
        scaled_eval = apply_halfmove_scale(raw_eval, board.halfmove);
        // Apply correction history to the halfmove-scaled value
        static_eval = if FEAT_CORRECTION.load(Ordering::Relaxed) { corrected_eval(info, board, scaled_eval) } else { scaled_eval };
        if ply_u < MAX_PLY {
            info.static_evals[ply_u] = static_eval;
        }
        // Improving: our eval is better than 2 plies ago.
        //
        // C8 audit LIKELY #2: when ply-2 was in-check, static_evals[ply-2]
        // was set to -INFINITY (see the else branch below). Comparing
        // `static_eval > -INFINITY` trivialised to true, so improving fired
        // on every post-check comeback, inflating RFP/LMP/futility/LMR.
        // SF/Viridithas fall back to ply-4 when ply-2 is unavailable; skip
        // entirely when neither ply is usable.
        if ply >= 2 && ply_u >= 2 {
            let prev2 = info.static_evals[ply_u - 2];
            if prev2 > -INFINITY + 1 {
                improving = static_eval > prev2;
            } else if ply_u >= 4 {
                let prev4 = info.static_evals[ply_u - 4];
                if prev4 > -INFINITY + 1 {
                    improving = static_eval > prev4;
                }
                // else: leave improving=false (no usable baseline).
            }
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
    if depth >= tp10(&IIR_MIN_DEPTH_10X) && tt_move == NO_MOVE && !in_check && (is_pv || cut_node) && FEAT_IIR.load(Ordering::Relaxed) {
        depth -= 1;
    }

    // Threat square from null-move failure
    let mut threat_sq: i32 = -1;

    // Hindsight reduction: when parent was LMR-reduced and both sides
    // think the position is quiet, reduce depth further.
    // Gate on prior_reduction (Stockfish >= 2, Alexandria >= 1).
    let prior_reduction = if ply_u >= 1 { info.reductions[ply_u - 1] } else { 0 };
    if !in_check && ply >= 1 && depth >= tp10(&HINDSIGHT_MIN_DEPTH_10X) && ply_u >= 1
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

    // Hindsight extension (Stormphrax search.cpp:749-752): mirror of the
    // reduction. When parent reduced aggressively (>=3) but the combined
    // eval shows position has worsened (eval_sum <= 0), extend +1 ply to
    // find the threat we missed. Non-PV only (PV already searched fully).
    if !in_check && ply >= 1 && ply_u >= 1
        && !is_pv
        && prior_reduction >= 3
        && info.static_evals[ply_u - 1] > -(MATE_SCORE - 100)
        && static_eval > -INFINITY
        && FEAT_HINDSIGHT.load(Ordering::Relaxed)
    {
        let eval_sum = info.static_evals[ply_u - 1] + static_eval;
        if eval_sum <= 0 {
            depth += 1;
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

    // T2.1: undefended ("hanging") piece count. Our non-pawn pieces
    // that are attacked by enemy AND NOT defended by any of our own
    // pieces. Zero-cost-when-skipped: computation only runs for
    // NMP-eligible nodes (most nodes either fail the depth gate or are
    // in_check). ~10-15 magic lookups per computed node — comparable
    // to king-zone-pressure's cost.
    let undefended_count: i32 = {
        // Only bother computing when NMP might actually fire.
        let nmp_gate_cheap = depth >= tp10(&NMP_MIN_DEPTH_10X) && !in_check && ply > 0
            && stm_non_pawn != 0 && beta - alpha == 1
            && static_eval >= beta && !prev_was_null
            && beta.abs() < MATE_SCORE - 100
            && info.excluded_move[ply_u] == NO_MOVE;
        if nmp_gate_cheap && tp10(&NMP_UNDEFENDED_MAX_10X) > 0 {
            let our_non_pawn = board.colors[board.side_to_move as usize]
                & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
            let attacked = our_non_pawn & enemy_attacks;
            let our_attacks = board.attacks_by_color(board.side_to_move);
            popcount(attacked & !our_attacks) as i32
        } else {
            0
        }
    };

    if depth >= tp10(&NMP_MIN_DEPTH_10X) && !in_check && ply > 0 && stm_non_pawn != 0
        && beta - alpha == 1 && static_eval >= beta
        && !prev_was_null  // Prevent consecutive null moves
        && beta.abs() < MATE_SCORE - 100  // Skip NMP for mate/TB scores
        && info.excluded_move[ply_u] == NO_MOVE  // Skip NMP during SE verification
        && king_zone_pressure < tp10(&NMP_KING_ZONE_MAX_10X)  // New gate
        && any_threat_count < 3  // S7-style: skip NMP when many of our pieces are under threat
        && undefended_count < tp10(&NMP_UNDEFENDED_MAX_10X)  // T2.1: skip when hanging pieces
        && cut_node  // Reckless gate: only attempt NMP at expected fail-high nodes (closes 30%->57% NMP cutoff-rate gap)
        && FEAT_NMP.load(Ordering::Relaxed)
    {
        info.stats.nmp_attempts += 1;
        // Adaptive reduction: scales with depth and eval margin above beta
        let mut r = tp10(&NMP_BASE_R_10X) + depth / tp10(&NMP_DEPTH_DIV_10X);
        // Reduce more after captures: opponent just captured, null move more likely to work
        // (Consensus: SF/Obsidian increase R after captures, not decrease)
        if !board.undo_stack.is_empty() && board.undo_stack[board.undo_stack.len() - 1].captured != NO_PIECE_TYPE {
            r += 1;
        }
        if static_eval > beta {
            let eval_r = ((static_eval - beta) / tp(&NMP_EVAL_DIV)).min(tp10(&NMP_EVAL_MAX_10X));
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
        // C3 (2026-04-22 audit): set null sentinel on moved_piece_stack /
        // moved_to_stack at ply_u. Without this, child at ply+1 reads
        // stale (piece, to) from a prior sibling move at this ply, feeding
        // cont-hist, history pruning and LMR-history adjustment with
        // unrelated data. Stockfish sets the null sentinel similarly.
        if ply_u <= MAX_PLY {
            info.moved_piece_stack[ply_u] = 0;
            info.moved_to_stack[ply_u] = 0;
        }
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
            if depth >= tp10(&NMP_VERIFY_DEPTH_10X) {
                info.stats.nmp_verify += 1;
                // Verification re-searches current position (no move made), so ply stays same
                let v_score = negamax(board, info, beta - 1, beta, depth - r, ply, false);
                // Stop-during-verification returns 0 from negamax; with
                // beta <= 0 (fail-low re-searches / losing branches),
                // `0 >= beta` is true and we'd return `nmp_score` (the
                // unverified null_score) as a real cutoff. Mirror the
                // post-recursion stop-check pattern used at every other
                // negamax call site in this function.
                if info.stop.load(Ordering::Relaxed) {
                    return 0;
                }
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
            // E2: widen margin when position is unstable (parent-child eval gap
            // > UNSTABLE_THRESH). Static eval can't be trusted for RFP when
            // eval is volatile. Mirrors unstable × ProbCut skip (#542 +6.7).
            if unstable { margin += margin / 3; }
            if static_eval - margin >= beta {
                info.stats.rfp_cutoffs += 1;
                return static_eval - margin;
            }
        }

    }

    // ProbCut: at moderate+ depths, if a shallow search of captures with
    // raised beta confirms the position is winning, prune the node.
    //
    // C8 audit LIKELY #7/#8: two fixes to this gate.
    // - #8: add !is_pv — SF/Obsidian/Viridithas/Berserk all gate ProbCut to
    //   non-PV. Pruning PV nodes on a raised-beta shallow search is too
    //   aggressive for a node whose score we need exactly.
    // - #7: the "TT says no chance" skip read `tt_entry.score` raw (no ply
    //   adjust) and accepted ANY flag. A LOWER bound < probcut_beta means
    //   "score is AT LEAST X" — it does not mean "no chance at probcut_beta",
    //   the true score can be much higher. Only UPPER/EXACT bounds are
    //   evidence of a ceiling. Switch to ply-adjusted score + bound gate.
    let probcut_beta = beta + tp(&PROBCUT_MARGIN);
    let probcut_tt_noshot = if tt_hit && tt_entry.depth >= depth - tp(&PROBCUT_TT_DEPTH_SLACK) {
        let adj_score = score_from_tt(tt_entry.score, ply);
        (tt_entry.flag == TT_FLAG_UPPER || tt_entry.flag == TT_FLAG_EXACT)
            && adj_score < probcut_beta
    } else {
        false
    };
    if !in_check && ply > 0 && !is_pv && depth >= tp10(&PROBCUT_MIN_DEPTH_10X)
        && beta.abs() < MATE_SCORE - 100  // skip for mate/TB scores
        && info.excluded_move[ply_u] == NO_MOVE  // skip during SE verification
        && !probcut_tt_noshot  // TT says no chance
        && king_zone_pressure < tp10(&PROBCUT_KING_ZONE_MAX_10X)  // A3: skip in high-threat positions
        && !unstable  // Skip ProbCut in eval-unstable positions (eval can't be trusted)
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
            let pc_dirty = if let Some(net) = info.nnue_net.as_deref() {
                build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), pc_moved_pt, pc_captured_pt, net)
            } else { DirtyPiece::recompute() };

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
    let safe_ply = ply_u.min(MAX_PLY - 1);
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
        MovePicker::new_evasion(tt_move, safe_ply, checkers, pinned, &info.history, prev_move, pawn_hist_ref, enemy_attacks, &info.moved_piece_stack, &info.moved_to_stack)
    } else {
        MovePicker::new(board, tt_move, safe_ply, &info.history, prev_move, pawn_hist_ref, enemy_attacks, our_xray_blockers, &info.moved_piece_stack, &info.moved_to_stack)
    };
    picker.threat_sq = threat_sq;

    let mut best_move = NO_MOVE;
    let mut best_score = -INFINITY;
    let mut move_count = 0i32;
    // EXPERIMENT (Starzix T1 #1): track PVS fail-high cascades at this node.
    // Each child that triggers a re-search (LMR failed high → re-search at
    // full depth, or zero-window PVS failed high → full-window re-search)
    // increments. On the eventual beta cutoff, scale the history bonus by
    // this count — more fail-highs at this node = stronger signal that the
    // cutoff move is genuinely good.
    let mut num_fail_highs: i32 = 0;
    // Track quiet moves searched before beta cutoff for history penalty
    let mut quiets_tried = [NO_MOVE; 64];
    let mut quiets_count = 0usize;

    // Track captures searched before beta cutoff for capture history penalty
    let mut captures_tried: [(u8, u8, u8); 32] = [(0, 0, 0); 32]; // (piece, to, victim)
    let mut n_captures_tried = 0usize;

    // Skip-quiets flag: once LMP fires, skip remaining quiets without
    // re-running gates (Reckless pattern). Bisection found this produces
    // +22% bench (vs expected bench-neutral perf-only) — mechanism not
    // localised; SPRT'd anyway as a data point per Adam.
    let mut skip_quiets = false;

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

        if skip_quiets && !is_cap && !is_promo {
            continue;
        }

        // SEE capture pruning: at shallow depths, prune captures that lose material
        if is_cap && ply > 0 && !in_check && depth <= tp(&SEE_CAP_DEPTH)
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
            && depth >= tp10(&SE_DEPTH_10X)
            && !in_check
            && info.excluded_move[ply_u] == NO_MOVE
            && tt_hit
            && tt_entry.flag != TT_FLAG_UPPER
            && tt_entry.depth >= depth - tp(&SE_TT_DEPTH_SLACK)
            && FEAT_SINGULAR.load(Ordering::Relaxed)
        {
            // Ply-only adjustment here — P3 downgrade deliberately not applied
            // at SE: would cause over-extension on downgraded mate scores that
            // pass the < MATE_SCORE - 100 check below.
            let tt_score_local = score_from_tt(tt_entry.score, ply);

            // Skip SE for mate scores (margin comparison meaningless)
            if tt_score_local > -(MATE_SCORE - 100) && tt_score_local < MATE_SCORE - 100 {
                // xray bonus: if TT move uncovers our slider's attack on enemy
                // (from-square ∈ our_xray_blockers), widen margin → easier
                // singular → more extensions on tactical discoveries.
                let xray_bonus = if our_xray_blockers & (1u64 << move_from(tt_move)) != 0 {
                    tp10(&SE_XRAY_BLOCKER_MARGIN_10X)
                } else { 0 };
                let singular_beta = tt_score_local - depth - xray_bonus;
                let singular_depth = (depth - 1) / 2;

                info.excluded_move[ply_u] = tt_move;
                let singular_score = negamax(board, info, singular_beta - 1, singular_beta, singular_depth, ply, false);
                info.excluded_move[ply_u] = NO_MOVE;

                if info.stop.load(Ordering::Relaxed) {
                    return 0;
                }

                if singular_score >= singular_beta && singular_beta >= beta {
                    // Multi-cut: alternatives are also good enough — prune the whole node.
                    // Return singular_score (SF pattern, search.cpp:1183) — tighter score
                    // for downstream TT propagation than singular_beta floor.
                    info.stats.multicut += 1;
                    return singular_score;
                }

                if singular_score < singular_beta {
                    // TT move is singular — no competitive alternatives.
                    //
                    // Reckless/SF-pattern additive extensions with PV/quiet/
                    // correction-aware margins. PV nodes get LARGER margin
                    // (suppressed); quiet TT moves get SMALLER margin (easier);
                    // large |corrhist| REDUCES threshold (eval is uncertain →
                    // extend less). DEXT_CAP propagation gates the additive
                    // count so cumulative extensions stay safe.
                    //
                    // Yin/Yang frame: aggressive extensions on tactical hits
                    // ENABLE more aggressive pruning of the rest. Default
                    // BASE term puts us in a sensible starting basin; SPSA
                    // explores the equilibrium where pruning compensates.
                    let is_tt_quiet = !is_cap && !is_promo;
                    let corr_abs = correction_value(info, board).abs();
                    let dext_margin = tp(&DEXT_MARGIN_PV) * is_pv as i32
                                    - tp(&DEXT_MARGIN_QUIET) * is_tt_quiet as i32
                                    - tp(&DEXT_MARGIN_CORR) * corr_abs / 128
                                    + tp(&DEXT_MARGIN_BASE);

                    singular_extension = 1;
                    info.stats.singular_ext += 1;
                    if info.double_ext_count[ply_u] < tp(&DEXT_CAP) {
                        let de = (singular_score < singular_beta - dext_margin) as i32;
                        singular_extension += de;
                        if de > 0 { info.stats.double_ext += 1; }
                    }
                } else if tt_score_local >= beta {
                    // TT move fails high and alternatives competitive — strong reduce
                    // Consensus: -3 non-PV (SF/Viridithas/Obsidian)
                    singular_extension = -3;
                    info.stats.negative_ext += 1;
                } else if cut_node {
                    // Cut node with competitive alternatives — moderate reduce
                    singular_extension = -2;
                    info.stats.negative_ext += 1;
                } else {
                    // All-node with competitive alternatives — mild reduce
                    singular_extension = -1;
                    info.stats.negative_ext += 1;
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

        // History-based pruning: prune quiet moves with deeply negative history at shallow depths.
        // Removed `!improving && !unstable` gate clauses per
        // cross_engine_comparison_2026-04-25.md item #10 — SF/Obsidian/
        // Halogen/Reckless gate hist-prune on neither. Coda's prior gate
        // suppressed firing in ~50% of nodes hist-prune was meant for.
        // 2026-05-14 audit: gate on lmr_d instead of raw depth (SF/Obsidian/
        // Reckless consensus). Standard "shallow after LMR reductions" semantic.
        if ply > 0 && !in_check && lmr_d <= tp10(&HIST_PRUNE_DEPTH_10X)
            && !is_cap && !is_promo
            && mv != tt_move
            && best_score > -(MATE_SCORE - 100)
            && FEAT_HIST_PRUNE.load(Ordering::Relaxed)
        {
            let mut hist_prune_score = info.history.main_score(from, to, enemy_attacks);
            if moved_piece != NO_PIECE {
                let gp = go_piece(moved_piece);
                // Cont-hist at offsets {1, 2, 4, 6} — full set used by
                // Coda's move-ordering already; bringing hist-prune score
                // in line. Diagnostic data showed including ply-6 doubles
                // fire rate at unchanged threshold (most-often-dominant
                // offset). See docs/history_prune_cont_hist_data_2026-05-08.md.
                let offsets = [1usize, 2, 4, 6];
                for &off in &offsets {
                    if ply_u >= off {
                        let p = info.moved_piece_stack[ply_u - off] as usize;
                        let pt = info.moved_to_stack[ply_u - off] as usize;
                        if p > 0 && p < 13 && pt < 64 {
                            hist_prune_score += info.history.cont_hist[p][pt][gp][to as usize] as i32;
                        }
                    }
                }
                // Pawn history in pruning decision
                let ph_idx = (board.pawn_hash as usize) % info.pawn_hist.len();
                hist_prune_score += info.pawn_hist[ph_idx][gp][to as usize] as i32;
            }

            // Diagnostics: bucket the gate score and per-offset cont-hist magnitudes.
            // One sample per gate-eligible move — cost ~12 i32 reads + 12 increments.
            // Findings drive the next experiments; see
            // docs/history_prune_cont_hist_review_2026-05-08.md.
            info.stats.hist_prune_eligible += 1;
            let threshold = tp(&HIST_PRUNE_MULT) * depth as i32;
            if threshold > 0 {
                let ratio_x100 = (hist_prune_score * 100) / threshold;
                let bucket = if ratio_x100 >= 100 { 0 }       // >= +1.0 (positive)
                    else if ratio_x100 >= 0 { 1 }              // [0, 1)
                    else if ratio_x100 >= -50 { 2 }            // [-0.5, 0)
                    else if ratio_x100 >= -100 { 3 }           // [-1, -0.5)
                    else if ratio_x100 >= -150 { 4 }           // [-1.5, -1)  — FIRES
                    else if ratio_x100 >= -200 { 5 }           // [-2, -1.5)
                    else if ratio_x100 >= -300 { 6 }           // [-3, -2)
                    else { 7 };                                // < -3.0
                info.stats.hist_prune_ratio_buckets[bucket] += 1;
            }
            // Per-offset cont-hist read magnitudes
            if moved_piece != NO_PIECE {
                let gp = go_piece(moved_piece);
                let offsets = [1usize, 2, 4, 6];
                for (i, &off) in offsets.iter().enumerate() {
                    if ply_u >= off {
                        let p = info.moved_piece_stack[ply_u - off] as usize;
                        let pt = info.moved_to_stack[ply_u - off] as usize;
                        if p > 0 && p < 13 && pt < 64 {
                            let v = info.history.cont_hist[p][pt][gp][to as usize] as i32;
                            let abs_v = v.unsigned_abs() as u64;
                            let mb = if abs_v < 200 { 0 }
                                else if abs_v < 1000 { 1 }
                                else if abs_v < 5000 { 2 }
                                else if abs_v < 10000 { 3 }
                                else { 4 };
                            info.stats.cont_hist_mag_buckets[i][mb] += 1;
                        }
                    }
                }
            }
            // 4D main-history bucket-read counter (ft, tt)
            if crate::search::FEAT_4D_HISTORY.load(Ordering::Relaxed) {
                let ft = ((enemy_attacks >> from) & 1) as usize;
                let tt = ((enemy_attacks >> to) & 1) as usize;
                info.stats.main_hist_bucket_reads[ft * 2 + tt] += 1;
            } else {
                info.stats.main_hist_bucket_reads[0] += 1;
            }

            // What-if fire rates for hypothetical hist-prune scores with
            // varying cont-hist offset combinations. Tells us the marginal
            // fire-rate gain of including each deeper offset in the score.
            if moved_piece != NO_PIECE && threshold > 0 {
                let gp = go_piece(moved_piece);
                let main_score_only = info.history.main_score(from, to, enemy_attacks);
                let ph_idx = (board.pawn_hash as usize) % info.pawn_hist.len();
                let pawn_score = info.pawn_hist[ph_idx][gp][to as usize] as i32;
                // Read all 4 cont-hist offsets (regardless of ply_u sufficiency:
                // 0 if not enough plies, treated as no contribution).
                let mut conts = [0i32; 4];
                let offsets = [1usize, 2, 4, 6];
                for (i, &off) in offsets.iter().enumerate() {
                    if ply_u >= off {
                        let p = info.moved_piece_stack[ply_u - off] as usize;
                        let pt = info.moved_to_stack[ply_u - off] as usize;
                        if p > 0 && p < 13 && pt < 64 {
                            conts[i] = info.history.cont_hist[p][pt][gp][to as usize] as i32;
                        }
                    }
                }
                // V0 = main + cont1 + pawn (CURRENT)
                let s_v0 = main_score_only + conts[0] + pawn_score;
                let s_v1 = s_v0 + conts[1];
                let s_v2 = s_v1 + conts[2];
                let s_v3 = s_v2 + conts[3];
                if s_v0 < -threshold { info.stats.hist_prune_what_if_fires[0] += 1; }
                if s_v1 < -threshold { info.stats.hist_prune_what_if_fires[1] += 1; }
                if s_v2 < -threshold { info.stats.hist_prune_what_if_fires[2] += 1; }
                if s_v3 < -threshold { info.stats.hist_prune_what_if_fires[3] += 1; }

                // Sign agreement: main_hist vs sum-of-cont-hist
                let cont_sum = conts[0] + conts[1] + conts[2] + conts[3];
                let agree_bucket = if main_score_only > 0 && cont_sum > 0 { 0 }
                    else if main_score_only < 0 && cont_sum < 0 { 1 }
                    else if main_score_only != 0 && cont_sum != 0 { 2 }
                    else { 3 };
                info.stats.cont_hist_sign_buckets[agree_bucket] += 1;

                // Per-offset dominance: which |cont_hist[i]| is largest?
                let mags = [conts[0].unsigned_abs(), conts[1].unsigned_abs(),
                            conts[2].unsigned_abs(), conts[3].unsigned_abs()];
                let total: u32 = mags.iter().sum();
                if total > 0 {
                    let mut best = 0;
                    for i in 1..4 { if mags[i] > mags[best] { best = i; } }
                    info.stats.cont_hist_dominant_offset[best] += 1;
                }
            }

            if hist_prune_score < -tp(&HIST_PRUNE_MULT) * depth as i32 {
                info.stats.history_prunes += 1;
                // 2026-05-14 audit: Obsidian/Alexandria/Stormphrax/Halogen all
                // use skipQuiets here — once a quiet is hist-pruned, all later
                // ones at this node should be too (they're worse-ordered).
                //
                // 2026-05-18 LMP investigation: plumb skip_quiet into the picker
                // so it stops generating/scoring quiets entirely (vs picker
                // yielding them and search discarding via `continue`). Cuts
                // wasted ordering work, and changes move_count seen by bad
                // captures (they yield after quiets) — lower move_count for
                // bad-cap LMR. SF/Obsidian both wire this into the picker.
                skip_quiets = true;
                picker.skip_quiet = true;
                continue;
            }
        }

        // Futility pruning: skip quiet moves when static eval + margin is below alpha.
        // Uses shared lmr_d for both gate and margin (SF/Obsidian/Berserk consensus).
        if ply > 0 && static_eval > -INFINITY && !in_check
            && !is_cap && !is_promo
            && best_score > -(MATE_SCORE - 100)
            && FEAT_FUTILITY.load(Ordering::Relaxed)
            && lmr_d <= tp(&FUT_LMR_DEPTH)
        {
            let main_hist = info.history.main_score(from, to, enemy_attacks);
            let hist_adj = main_hist / 128;
            // our_defenses widener: add margin per our-piece-under-attack so
            // tactical positions keep more lines from being pruned on eval.
            let threats_adj = any_threat_count * tp(&FUT_THREATS_MARGIN);
            let futility_value = static_eval + tp(&FUT_BASE) + lmr_d * tp(&FUT_PER_DEPTH) + hist_adj + threats_adj;
            // Don't futility-prune moves with very strong history (Igel pattern)
            // Direct-check carve-out: don't prune moves that give direct check
            // (Reckless #410 +1.62 STC).
            if futility_value <= alpha && main_hist < 12000 && !board.gives_direct_check(mv) {
                info.stats.futility_prunes += 1;
                continue;
            }
        }

        // Late Move Pruning: at shallow depths, skip late quiet moves.
        // Applied before MakeMove. Formula: (LMP_BASE + depth²) / (2 - improving)
        //
        // 2026-05-14 audit: removed !is_pv gate (SPRT #1209 +5.0 H1).
        // SF/Obsidian/Reckless all run LMP on PV nodes; Coda's PV nodes
        // previously had zero LMP coverage — a real gap.
        // 2026-05-15: depth-gated check carve (SPRT #1227 +1.3 H1). Keep
        // check-protection at shallow depths where tactical checks matter
        // most; drop at depth ≥ 4 where the carve mostly preserves
        // low-quality late checks.
        if ply > 0 && !in_check && depth >= 1 && depth <= tp(&LMP_DEPTH)
            && !is_cap && !is_promo
            && (depth >= 4 || !board.gives_direct_check(mv))
            && best_score > -(MATE_SCORE - 100)
            && FEAT_LMP.load(Ordering::Relaxed)
        {
            let lmp_limit = (tp(&LMP_BASE) + depth * depth) / (2 - improving as i32);
            if move_count > lmp_limit {
                info.stats.lmp_prunes += 1;
                skip_quiets = true;
                picker.skip_quiet = true;
                continue;
            }
        }

        // Bad noisy pruning: skip losing captures when eval is far below alpha.
        // Applied before MakeMove. Direct-check carve-out: don't prune moves
        // that give direct check (Reckless #630 +1.85 STC).
        if FEAT_BAD_NOISY.load(Ordering::Relaxed) && is_cap && !in_check && ply > 0 && depth <= tp(&BAD_NOISY_DEPTH) && mv != tt_move
            && !is_promo && best_score > -(MATE_SCORE - 100)
            && static_eval > -INFINITY && static_eval + depth * tp(&BAD_NOISY_MARGIN) <= alpha
            && !see_ge(board, mv, 0)
            && !board.gives_direct_check(mv)
        {
            continue;
        }

        // Build NNUE dirty piece info BEFORE make_move
        let dirty = if let Some(net) = info.nnue_net.as_deref() {
            build_dirty_piece(mv, us, flip_color(us), moved_pt, captured_pt, net)
        } else { DirtyPiece::recompute() };

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

        // Recapture extension: extend when recapturing on the same square.
        // ply > 0 guard: at root, undo_stack contains game-history moves;
        // without the guard, the last played game move (if a capture) would
        // count as a "previous capture" against the first root-move capture
        // even though it's not an in-search recapture pattern.
        let mut extension = 0;
        if ply > 0 && is_cap && board.undo_stack.len() >= 2 {
            let prev_undo = &board.undo_stack[board.undo_stack.len() - 2];
            if prev_undo.captured != NO_PIECE_TYPE && to == move_to(prev_undo.mv) {
                extension = if FEAT_EXTENSIONS.load(Ordering::Relaxed) { 1 } else { 0 };
                if extension > 0 { info.stats.recapture_ext += 1; }
            }
        }
        // N6 Promotion-imminent extension: pawn push to 7th rank (from STM's
        // perspective) very often decides the game. Extend by 1. Gated by
        // FEAT_EXTENSIONS to share the ablation flag with recapture ext.
        if extension == 0
            && FEAT_EXTENSIONS.load(Ordering::Relaxed)
            && !is_cap
            && !is_promo
            && moved_pt == PAWN
        {
            // STM=WHITE: 7th rank is row 6 (squares 48..56). STM=BLACK: 7th
            // rank is row 1 (squares 8..16). `to` is the destination.
            //
            // CRITICAL: must use pre-move STM `us`, not board.side_to_move.
            // make_move (line 3020) flips side_to_move to the opponent, so
            // reading board.side_to_move here checks the WRONG side and the
            // condition can never satisfy. Bug present since N6 introduced
            // 2026-04 (~3 weeks dead); audit C2026-04-25-N6 + 2026-05-09
            // correctness sweep. `us` is the value board.side_to_move had
            // BEFORE make_move; reused here.
            let to_rank = to >> 3; // to / 8
            let on_seventh = (us == WHITE && to_rank == 6)
                || (us == BLACK && to_rank == 1);
            if on_seventh {
                extension = 1;
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
        // Endgame gate: skip LMR in low-piece-count positions where
        // mate-completing king-restriction moves would be over-reduced.
        let endgame_threshold = tp10(&LMR_ENDGAME_PIECES_10X) as u32;
        let is_endgame_skip = endgame_threshold > 0
            && crate::bitboard::popcount(board.occupied()) <= endgame_threshold;
        if !in_check && !is_cap && !is_promo && !is_endgame_skip && FEAT_LMR.load(Ordering::Relaxed) {
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
                //
                // Compare against `scaled_eval` (pre-correction, post-hm-scale)
                // so "complexity" measures only the corrhist delta magnitude,
                // not the halfmove scaling factor. Using raw_eval here would
                // conflate corrhist drift with halfmove decay, artificially
                // inflating complexity in long-halfmove positions.
                if scaled_eval > -INFINITY {
                    let complexity = (static_eval - scaled_eval).abs();
                    reduction -= complexity / tp(&LMR_COMPLEXITY_DIV);
                }

                // Threat-density LMR: reduce less when multiple pieces are
                // under pawn attack. Tactical positions need deeper search.
                // Fixed-point divisor: stored × 10. Avoids tp10 swallowing
                // sub-integer SPSA precision on this multiplicative use.
                reduction -= threat_count * 10 / LMR_THREAT_DIV_10X.load(Ordering::Relaxed).max(1);

                // King-pressure LMR modifier: reduce less when enemy has
                // many attackers on our king zone. Parent-node signal reused
                // from NMP/ProbCut gates — tactical king positions need depth.
                reduction -= king_zone_pressure * 10 / LMR_KING_PRESSURE_DIV_10X.load(Ordering::Relaxed).max(1);

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
        if !in_check && is_cap && !is_promo && move_count > 1 && mv != tt_move && !is_endgame_skip && FEAT_LMR.load(Ordering::Relaxed) {
            // Only reduce at non-PV nodes (zero window search)
            if beta - alpha == 1 {
                let d = (depth as usize).min(63);
                let m = (move_count as usize).min(63);
                reduction = lmr_cap_reduction(d as i32, m as i32);

                if reduction > 0 {
                    // Continuous capture history adjustment — replaced the
                    // prior step function (±1 at |capt_hist|>2000) with
                    // continuous `capt_hist / LMR_HIST_DIV_CAP` to mirror
                    // quiet-LMR's `hist_score / LMR_HIST_DIV` and Obsidian's
                    // `R -= hist/(isQuiet?Q_DIV:C_DIV)`. 2026-05-18 outlier
                    // audit traced LMR_C_CAP<LMR_C_QUIET inversion to this
                    // step-vs-continuous asymmetry — SPSA had to compress
                    // C_CAP because there was no per-feature carve-out for
                    // tactical capt_hist signal beyond the binary fire.
                    if moved_piece != NO_PIECE && captured_pt != NO_PIECE_TYPE {
                        let ct = if flags == FLAG_EN_PASSANT { captured_type(PAWN) } else { captured_type(captured_pt) };
                        let capt_hist_val = info.history.capture[go_piece(moved_piece)][to as usize][ct] as i32;
                        reduction -= capt_hist_val / tp(&LMR_HIST_DIV_CAP);
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
                // LMR failed high: doDeeper/doShallower before re-search.
                //
                // Audit SPECULATIVE fix (v2 retry): `new_depth` (integer depth
                // 5-20) was used as a cp margin — near-certain typo. #673 at
                // 30cp H0'd −1.7 @ 17286g; retrying with 20cp, closer to the
                // old "new_depth ≈ 10-15" effective threshold but with proper
                // cp semantics. If this also H0s, the true value is smaller
                // still (try 10cp) or the feature wants a depth-scaled margin.
                num_fail_highs += 1; // Starzix T1 #1: LMR fail-high cascade.
                let mut do_deeper_adj = 0;
                if lmr_score > best_score + 60 + 10 * reduction {
                    do_deeper_adj = 1;
                } else if lmr_score < best_score + 20 {
                    do_deeper_adj = -1;
                }

                lmr_score = -negamax(board, info, -alpha - 1, -alpha, new_depth + do_deeper_adj, ply + 1, !cut_node);

                // EXPERIMENT: post-LMR-research cont-hist nudge (Berserk pattern,
                // search.c:747-748). After the zero-window re-search, nudge cont-hist
                // based on whether re-search confirmed or refuted the LMR prediction:
                //   score >= beta:  re-search confirmed move good → +bonus
                //   score <= alpha: re-search confirmed move bad  → -malus
                //   else (alpha < score < beta): PVS decides; no nudge here.
                // Adds signal density beyond beta-cutoff updates. Quiet moves only.
                if !is_cap && moved_piece != NO_PIECE {
                    let nudge_depth = (new_depth - 1).max(1);
                    let nudge_bonus = if lmr_score >= beta {
                        history_bonus(nudge_depth)
                    } else if lmr_score <= alpha {
                        -history_bonus(nudge_depth)
                    } else {
                        0
                    };
                    if nudge_bonus != 0 {
                        let gp_mv = go_piece(moved_piece);
                        // T6: base = current cont_hist + main_hist / 2 (Stormphrax history.h:120).
                        let main_score_v = info.history.main_score(from, to, enemy_attacks);
                        let ch_offsets = [1usize, 2, 4, 6];
                        for &off in &ch_offsets {
                            if ply_u >= off {
                                let prior_piece = info.moved_piece_stack[ply_u - off] as usize;
                                let prior_to = info.moved_to_stack[ply_u - off] as usize;
                                if prior_piece > 0 && prior_piece < 13 && prior_to < 64 {
                                    // B1 (audit 2026-05-19): uniform bonus across offsets
                                    // {1,2,4,6}. Coda was unique in [bonus, b/2, b/2, b/2]
                                    // shape; Reckless/Berserk/Alexandria/Stormphrax use
                                    // uniform `bonus`. See docs/history_prune_cont_hist_
                                    // review_2026-05-08.md Experiment B1.
                                    let ch_b = nudge_bonus;
                                    let cur_cont = info.history.cont_hist[prior_piece][prior_to][gp_mv][to as usize] as i32;
                                    let base = cur_cont + main_score_v / 2;
                                    History::update_cont_history_with_base(
                                        &mut info.history.cont_hist[prior_piece][prior_to][gp_mv][to as usize],
                                        base,
                                        ch_b,
                                    );
                                }
                            }
                        }
                    }
                }
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
                num_fail_highs += 1; // Starzix T1 #1: PVS fail-high cascade.
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
                        // Depth-boost on big fail-high. BONUS_BOOST_AT trigger
                        // removed 2026-05-17 (ablation #1277 H0). Two remaining
                        // triggers (Stormphrax: cutoff beat static eval;
                        // improving) can stack for +2 depth.
                        let bonus_depth = depth
                            + if !in_check && static_eval <= best_score { 1 } else { 0 }
                            + if improving { 1 } else { 0 };
                        // numFailHighs multiplicative scaling (#1020, Starzix T1 #1) —
                        // more cascades = stronger cutoff confidence.
                        let raw_bonus = history_bonus(bonus_depth);
                        let scale_factor = num_fail_highs.min(tp10(&NFH_CAP_10X));
                        // Fixed-point divisor (stored × 10).
                        let bonus = raw_bonus + raw_bonus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);

                        // Update main history
                        History::update_history(
                            info.history.main_entry(from, to, enemy_attacks),
                            bonus,
                        );

                        // Update continuation history at plies 1, 2, 4, 6
                        // Ply-1 at full bonus, plies 2/4/6 at half bonus (Obsidian pattern)
                        if moved_piece != NO_PIECE {
                            let gp_mv = go_piece(moved_piece);
                            // T6: base = current cont_hist + main_hist / 2 (Stormphrax history.h:120).
                            let main_score_v = info.history.main_score(from, to, enemy_attacks);
                            let ch_offsets = [1usize, 2, 4, 6];
                            for (i, &off) in ch_offsets.iter().enumerate() {
                                if ply_u >= off {
                                    let prior_piece = info.moved_piece_stack[ply_u - off] as usize;
                                    let prior_to = info.moved_to_stack[ply_u - off] as usize;
                                    if prior_piece > 0 && prior_piece < 13 && prior_to < 64 {
                                        // B1: uniform bonus (see LMR nudge site above).
                                        let ch_bonus = bonus;
                                        let cur_cont = info.history.cont_hist[prior_piece][prior_to][gp_mv][to as usize] as i32;
                                        let base = cur_cont + main_score_v / 2;
                                        History::update_cont_history_with_base(
                                            &mut info.history.cont_hist[prior_piece][prior_to][gp_mv][to as usize],
                                            base,
                                            ch_bonus,
                                        );
                                        info.stats.cont_hist_writes[i] += 1;
                                        info.stats.cont_hist_write_mag_sum[i] += ch_bonus.unsigned_abs() as u64;
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

                            // Penalize continuation history at plies 1, 2, 4, 6.
                            // T6: base uses qf,qt move's main_score (the move being malused).
                            {
                                let q_piece = board.piece_at(qf);
                                if q_piece != NO_PIECE {
                                    let gp_q = go_piece(q_piece);
                                    let q_main_score = info.history.main_score(qf, qt, enemy_attacks);
                                    let ch_offsets = [1usize, 2, 4, 6];
                                    for (i, &off) in ch_offsets.iter().enumerate() {
                                        if ply_u >= off {
                                            let prior_piece = info.moved_piece_stack[ply_u - off] as usize;
                                            let prior_to = info.moved_to_stack[ply_u - off] as usize;
                                            if prior_piece > 0 && prior_piece < 13 && prior_to < 64 {
                                                // B1: uniform penalty (see bonus site above).
                                                let ch_pen = -bonus;
                                                let cur_cont = info.history.cont_hist[prior_piece][prior_to][gp_q][qt as usize] as i32;
                                                let base = cur_cont + q_main_score / 2;
                                                History::update_cont_history_with_base(
                                                    &mut info.history.cont_hist[prior_piece][prior_to][gp_q][qt as usize],
                                                    base,
                                                    ch_pen,
                                                );
                                                info.stats.cont_hist_writes[i] += 1;
                                                info.stats.cont_hist_write_mag_sum[i] += ch_pen.unsigned_abs() as u64;
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
                        // Capture caused beta cutoff: bonus the cutoff capture.
                        // BONUS_BOOST_AT depth-boost removed 2026-05-17
                        // (ablation #1277 H0).
                        let cap_bonus_depth = depth;
                        // numFailHighs multiplicative scaling (#1054 ext of #1020):
                        // more cascades = stronger cutoff confidence.
                        let raw_cap_bonus = capture_history_bonus(cap_bonus_depth);
                        let scale_factor = num_fail_highs.min(tp10(&NFH_CAP_10X));
                        // Fixed-point divisor (stored × 10).
                        let cap_bonus = raw_cap_bonus + raw_cap_bonus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);
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
        && scaled_eval > -(MATE_SCORE - 100)
        // C8 audit LIKELY #12: TT-store has a stop guard (see tt write
        // path); corrhist update previously didn't. On a stop, children
        // returned 0, which can bubble up as best_score > alpha_orig
        // from a polluted baseline. Writing that into corrhist poisons
        // per-thread tables for every subsequent iteration.
        && !info.stop.load(Ordering::Relaxed)
    {
        // Train corrhist on the halfmove-scaled pre-correction value.
        // `best_score` is in scaled-space (propagated up from scaled leaf
        // evals), so the err term `best_score - scaled_eval` captures the
        // positional miscalibration we want corrhist to learn — not the
        // halfmove decay, which is already priced into best_score.
        update_correction_history(info, board, best_score, scaled_eval, depth);
    }

    // Fail-high score blending: dampen inflated cutoff scores at non-PV nodes.
    //
    // Skip blending when we're inside an SE verification search (excluded_move
    // is set on this ply). The dampened return value would feed into the
    // singular_score → DEXT-margin comparison upstream, biasing DEXT toward
    // single extensions on otherwise-double-extension-eligible TT moves.
    // SE verification needs the raw cutoff score to make the right extension
    // call.
    if best_score >= beta && beta - alpha_orig == 1 && depth >= tp10(&FH_BLEND_DEPTH_10X)
        && best_score > -(MATE_SCORE - 100) && best_score < MATE_SCORE - 100
        && info.excluded_move[ply_u] == NO_MOVE
    {
        // Divisor floor: SPSA can perturb FH_BLEND_OFFSET to 0 AND
        // FH_BLEND_DEPTH_10X low enough that the gate admits depth=0,
        // producing a 0+0 divisor → div-by-zero panic. Clamp.
        return (best_score * depth + beta) / (depth + tp(&FH_BLEND_OFFSET)).max(1);
    }

    best_score
}

/// History bonus: linear depth-based bonus for history updates.
/// Consensus: SF min(1469, 155*d-93), Clarity min(1632, 276*d-119),
/// Obsidian min(1400, 175*d-50). Our old depth² formula gave 25 at d=5
/// vs SF's 682 — history values were 27× too small to influence ordering.
fn history_bonus(depth: i32) -> i32 {
    // Offset shape — mirrors Stockfish's `155*d - 93` and our own
    // capture-history's `MULT * d - BASE`. Clamped at 0 to avoid
    // negative bonuses at very shallow depth (which would corrupt
    // gravity updates) and at MAX to cap the late-depth plateau.
    (tp(&HIST_BONUS_MULT) * depth - tp(&HIST_BONUS_OFFSET)).clamp(0, tp(&HIST_BONUS_MAX))
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

    // Draw detection: repetition and 50-move rule. Contempt removed (#508).
    let draw_score = 0;
    if board.halfmove >= 100 {
        return draw_score;
    }
    // Check for repetition in game history.
    // C8 audit LIKELY #38: also break on null-move boundary — scanning
    // past a null move looks for repetitions in a different search line.
    let hash = board.hash;
    for undo in board.undo_stack.iter().rev().skip(1).step_by(2) {
        if undo.hash == hash { return draw_score; }
        if undo.halfmove == 0 { break; } // irreversible move
        if undo.mv == NO_MOVE { break; } // null-move boundary
    }

    // Limit quiescence depth to prevent stack overflow
    if qs_depth >= 32 {
        return apply_halfmove_scale(info.eval(board), board.halfmove);
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
    // C8 audit LIKELY #10: gate QS cuckoo on ply > 0, mirroring the
    // main-search check at line 1763. Cuckoo's root-boundary STM check
    // is undefined at ply 0.
    if ply > 0 && alpha < 0 && FEAT_CUCKOO.load(Ordering::Relaxed) && crate::cuckoo::has_game_cycle(board, ply) {
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
    let tt_cur_gen = info.tt.current_generation();
    info.stats.tt_probes += 1;
    if tt_hit {
        info.stats.tt_hits += 1;
        if tt_entry.generation != tt_cur_gen {
            info.stats.tt_cross_gen_hits += 1;
        }
    }

    if tt_hit && tt_entry.depth >= -1 {
        let tt_score = score_from_tt(tt_entry.score, ply);
        // P3: downgrade stored mate if 50mr will fire before mate.
        let tt_ret = downgrade_50mr_mate(tt_score, ply, board.halfmove);

        // P2: skip QS TT cutoff near 50mr — stale bound unsafe
        let halfmove_ok = (board.halfmove as i32) < tp(&TT_CUTOFF_HALFMOVE_MAX);
        let qs_is_pv = beta - alpha > 1;
        match tt_entry.flag {
            TT_FLAG_EXACT => {
                if !qs_is_pv && halfmove_ok { return tt_ret; }
            }
            TT_FLAG_LOWER => {
                if !qs_is_pv && halfmove_ok && tt_score >= beta { return tt_ret; }
            }
            TT_FLAG_UPPER => {
                if !qs_is_pv && halfmove_ok && tt_score <= alpha { return tt_ret; }
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
        // C8 audit LIKELY #19: evasion history reads now use enemy_attacks
        // (symmetric with beta-cutoff writes). Compute here since QS doesn't
        // otherwise need the bitboard.
        let qs_enemy_attacks = board.attacks_by_color(
            crate::types::flip_color(board.side_to_move)
        );
        // Clamp ply to the moved_piece_stack / moved_to_stack bounds.
        // Qsearch can recurse past MAX_PLY via tactical extensions and evasion
        // chains; without this clamp MovePicker::new_evasion indexes
        // `moved_piece_stack[ply - off]` with ply > MAX_PLY and panics (observed
        // on lichess ASuoXT9f — game thrown from a +21.84 winning position).
        let qs_safe_ply = (ply as usize).min(MAX_PLY - 1);
        let mut evasion_picker = MovePicker::new_evasion(
            tt_move, qs_safe_ply, qs_checkers, qs_pinned, &info.history, qs_prev_move, qs_pawn_hist_ref,
            qs_enemy_attacks,
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
            let qs_dirty = if let Some(net) = info.nnue_net.as_deref() {
                build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), qs_moved_pt, qs_captured_pt, net)
            } else { DirtyPiece::recompute() };

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
    // Use TT staticEval when available to avoid recomputing. TT stores
    // halfmove-independent values (see SearchInfo::eval doc), so we apply
    // the scale freshly against `board.halfmove` after reading.
    let raw_stand_pat = if tt_hit && tt_entry.static_eval > -4095 {
        // Threshold matches pack_data's clamp range. -INFINITY sentinels
        // (from in-check TT stores) get clamped to -4095 and would
        // otherwise pass a wider check.
        info.stats_tt_static_eval_hits += 1;
        tt_entry.static_eval
    } else {
        info.eval(board)
    };
    let stand_pat = apply_halfmove_scale(raw_stand_pat, board.halfmove);
    let mut best_score = stand_pat;

    // TT bound refinement of stand-pat (consensus: every top engine does this)
    // Use TT score as a better estimate when the bound direction agrees
    if tt_hit {
        // Ply-only adjustment; refinement is explicitly for non-mate scores
        // per the abs check below. P3 downgrade would turn a stored mate
        // into a huge TB_WIN signal that passes the check and pollutes
        // stand-pat.
        let tt_score = score_from_tt(tt_entry.score, ply);
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
        let qs_dirty = if let Some(net) = info.nnue_net.as_deref() {
            build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), qs_moved_pt, qs_captured_pt, net)
        } else { DirtyPiece::recompute() };

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
        // Store the halfmove-INDEPENDENT value so later probes at a
        // different halfmove get a correct scale — see the doc comment
        // in `SearchInfo::eval`.
        info.tt.store(board.hash, -1, store_score, flag, best_move, raw_stand_pat, false);
    }

    // QS beta blending: dampen capture fail-high at non-PV nodes
    if best_score >= beta && beta - alpha_orig == 1
        && best_score < MATE_SCORE - 100 && best_score > -(MATE_SCORE - 100)
    {
        return (best_score + beta) / 2;
    }

    best_score
}

/// Standard bench position list — 48 positions, imported from Stockfish's
/// `Defaults` array (chess960 + setoption control lines dropped, two endgame
/// FENs padded to 6 fields, SF Pohl knight-saturation test dropped — see
/// 2026-05-07: 50% of fresh SB200 random seeds and 1-of-5 SB800 nets had
/// elevated tree size on it, distorting bench aggregates and OB scale_nps).
/// Used by `coda bench` and `coda eval-bench` so the prune-stats /
/// move-ordering / NPS aggregates have N=48 sample size, matching the field
/// convention (Reckless 46, Halogen 49, Stormphrax 50, Viridithas 50,
/// Alexandria 51, Stockfish 51) rather than the historical 8 we used to
/// ship with.
pub const BENCH_POSITIONS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    "rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14",
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14",
    "r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13",
    "r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16",
    "4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17",
    "2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11",
    "r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16",
    "3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22",
    "r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    "3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
    "3b4/5kp1/1p1p1p1p/pP1PpP1P/P1P1P3/3KN3/8/8 w - - 0 1",
    "2K5/p7/7P/5pR1/8/5k2/r7/8 w - - 0 1",
    "8/6pk/1p6/8/PP3p1p/5P2/4KP1q/3Q4 w - - 0 1",
    "7k/3p2pp/4q3/8/4Q3/5Kp1/P6b/8 w - - 0 1",
    "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
    "8/1p3pp1/7p/5P1P/2k3P1/8/2K2P2/8 w - - 0 1",
    "8/pp2r1k1/2p1p3/3pP2p/1P1P1P1P/P5KR/8/8 w - - 0 1",
    "8/3p4/p1bk3p/Pp6/1Kp1PpPp/2P2P1P/2P5/5B2 b - - 0 1",
    "5k2/7R/4P2p/5K2/p1r2P1p/8/8/8 b - - 0 1",
    "6k1/6p1/P6p/r1N5/5p2/7P/1b3PP1/4R1K1 w - - 0 1",
    "1r3k2/4q3/2Pp3b/3Bp3/2Q2p2/1p1P2P1/1P2KP2/3N4 w - - 0 1",
    "6k1/4pp1p/3p2p1/P1pPb3/R7/1r2P1PP/3B1P2/6K1 w - - 0 1",
    "8/3p3B/5p2/5P2/p7/PP5b/k7/6K1 w - - 0 1",
    "5rk1/q6p/2p3bR/1pPp1rP1/1P1Pp3/P3B1Q1/1K3P2/R7 w - - 93 90",
    "4rrk1/1p1nq3/p7/2p1P1pp/3P2bp/3Q1Bn1/PPPB4/1K2R1NR w - - 40 21",
    "r3k2r/3nnpbp/q2pp1p1/p7/Pp1PPPP1/4BNN1/1P5P/R2Q1RK1 w kq - 0 16",
    "3Qb1k1/1r2ppb1/pN1n2q1/Pp1Pp1Pr/4P2p/4BP2/4B1R1/1R5K b - - 11 40",
    "4k3/3q1r2/1N2r1b1/3ppN2/2nPP3/1B1R2n1/2R1Q3/3K4 w - - 5 1",
    "1r6/1P4bk/3qr1p1/N6p/3pp2P/6R1/3Q1PP1/1R4K1 w - - 1 42",
    "K7/8/8/BNQNQNB1/N5N1/R1Q1q2r/n5n1/bnqnqnbk w - - 0 1",
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",
    "8/3k4/8/8/8/4B3/4KB2/2B5 w - - 0 1",
    "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
    "8/2p4P/8/kr6/6R1/8/8/1K6 w - - 0 1",
    "8/8/3P3k/8/1p6/8/1P6/1K3n2 b - - 0 1",
    "8/R7/2q5/8/6k1/8/1P5p/K6R w - - 0 124",
    "6k1/3b3r/1p1p4/p1n2p2/1PPNpP1q/P3Q1p1/1R1RB1P1/5K2 b - - 0 1",
    "r2r1n2/pp2bk2/2p1p2p/3q4/3PN1QP/2P3R1/P4PP1/5RK1 w - - 0 1",
    "8/8/8/8/8/6k1/6p1/6K1 w - - 0 1",
    "7k/7P/6K1/8/3B4/8/8/8 b - - 0 1",
];

/// Pathological positions used to flush out tree-shape blow-ups —
/// positions where a misbehaving net or a broken pruning/extension
/// interaction produces a tree that is orders of magnitude bigger than
/// expected. Run via `coda bench-pathology`. The default node-budget
/// threshold below (5M @ depth 8) flags clear pathology; well-trained
/// prod nets land ~1M.
///
/// Add new positions whenever an investigation surfaces a class of
/// position that drives non-linear search-time blow-up. Prefer
/// well-known stress positions (SF defaults, Pohl, ECM hardest)
/// over engine-specific corner cases.
pub const BENCH_PATHOLOGY_POSITIONS: &[&str] = &[
    // SF Pohl knight-saturation test. 14 minor pieces, 2 kings, no
    // pawns. Eval-driven non-convergence: 50% of fresh SB200 random
    // seeds and 1-of-5 SB800 nets had elevated tree size; some
    // exceeded 100M nodes at depth 8. Removed from main bench list
    // 2026-05-07; kept here as a tripwire.
    "k7/2n1n3/1nbNbn2/2NbRBn1/1nbRQR2/2NBRBN1/3N1N2/7K w - - 0 1",
];

/// Run bench: fixed-depth search on standard positions, return total nodes.
pub fn bench(depth: i32, nnue_path: Option<&str>) -> u64 {
    bench_inner(depth, nnue_path, true)
}

/// Run pathology bench: per-position node + wall-clock report. Returns
/// the count of positions exceeding `node_threshold` so callers can
/// fail-fast on regressions.
pub fn bench_pathology(depth: i32, node_threshold: u64, nnue_path: Option<&str>) -> u32 {
    let mut info = SearchInfo::new(16);
    info.silent = true;  // suppress UCI info lines, want only the per-position report
    if let Some(path) = nnue_path {
        if let Err(e) = info.load_nnue(path) {
            eprintln!("Warning: failed to load NNUE: {}", e);
        }
    } else {
        info.auto_discover_nnue();
    }
    let limits = SearchLimits {
        depth,
        infinite: true,
        ..SearchLimits::new()
    };
    let mut over = 0u32;
    let mut total_nodes = 0u64;
    let total_start = std::time::Instant::now();
    println!("idx |  nodes        | nps      | time(s) | flag | fen");
    println!("----+---------------+----------+---------+------+----");
    for (i, fen) in BENCH_PATHOLOGY_POSITIONS.iter().enumerate() {
        let mut board = Board::from_fen(fen);
        info.nodes = 0;
        info.history.clear();
        info.tt.new_search();
        let start = std::time::Instant::now();
        let _mv = search(&mut board, &mut info, &limits);
        let elapsed = start.elapsed();
        let nodes = info.nodes;
        let nps = if elapsed.as_secs_f64() > 0.0 {
            (nodes as f64 / elapsed.as_secs_f64()) as u64
        } else { 0 };
        let flag = if nodes > node_threshold { "WARN" } else { "ok  " };
        if nodes > node_threshold { over += 1; }
        total_nodes += nodes;
        println!("{:>3} | {:>13} | {:>8} | {:>7.2} | {} | {}", i, nodes, nps, elapsed.as_secs_f64(), flag, fen);
    }
    println!("\nTotal: {} positions, {} nodes, {:.2}s wall-clock, {} over threshold ({}M nodes @ depth {})",
        BENCH_PATHOLOGY_POSITIONS.len(), total_nodes, total_start.elapsed().as_secs_f64(),
        over, node_threshold / 1_000_000, depth);
    over
}

/// Run bench without printing stats (for multi-threaded bench).
pub fn bench_silent(depth: i32, nnue_path: Option<&str>) -> u64 {
    bench_inner(depth, nnue_path, false)
}

fn bench_inner(depth: i32, nnue_path: Option<&str>, print_stats: bool) -> u64 {
    let positions = BENCH_POSITIONS;

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

    for fen in positions {
        let mut board = Board::from_fen(fen);
        info.nodes = 0;
        info.history.clear();
        info.tt.new_search();

        let _mv = search(&mut board, &mut info, &limits);
        total_nodes += info.nodes;

        // Accumulate stats across all positions
        total_stats.tt_probes += info.stats.tt_probes;
        total_stats.tt_hits += info.stats.tt_hits;
        total_stats.tt_cross_gen_hits += info.stats.tt_cross_gen_hits;
        total_stats.tt_cross_gen_cutoffs += info.stats.tt_cross_gen_cutoffs;
        total_stats.tt_cutoffs += info.stats.tt_cutoffs;
        total_stats.tt_near_miss += info.stats.tt_near_miss;
        total_stats.nmp_attempts += info.stats.nmp_attempts;
        total_stats.nmp_cutoffs += info.stats.nmp_cutoffs;
        total_stats.rfp_cutoffs += info.stats.rfp_cutoffs;
        total_stats.lmp_prunes += info.stats.lmp_prunes;
        total_stats.futility_prunes += info.stats.futility_prunes;
        total_stats.history_prunes += info.stats.history_prunes;
        total_stats.hist_prune_eligible += info.stats.hist_prune_eligible;
        for i in 0..8 { total_stats.hist_prune_ratio_buckets[i] += info.stats.hist_prune_ratio_buckets[i]; }
        for off in 0..4 {
            for mb in 0..5 {
                total_stats.cont_hist_mag_buckets[off][mb] += info.stats.cont_hist_mag_buckets[off][mb];
            }
            total_stats.cont_hist_writes[off] += info.stats.cont_hist_writes[off];
            total_stats.cont_hist_write_mag_sum[off] += info.stats.cont_hist_write_mag_sum[off];
            total_stats.main_hist_bucket_reads[off] += info.stats.main_hist_bucket_reads[off];
            total_stats.hist_prune_what_if_fires[off] += info.stats.hist_prune_what_if_fires[off];
            total_stats.cont_hist_sign_buckets[off] += info.stats.cont_hist_sign_buckets[off];
            total_stats.cont_hist_dominant_offset[off] += info.stats.cont_hist_dominant_offset[off];
        }
        // Sample 4D main-history density: count cells with |val|>1000 per bucket.
        // Sampled once per bench position; final sum is approximate density.
        for ft in 0..2usize {
            for tt in 0..2usize {
                let mut count: u64 = 0;
                for from in 0..64 {
                    for to in 0..64 {
                        if info.history.main[ft][tt][from][to].unsigned_abs() > 1000 {
                            count += 1;
                        }
                    }
                }
                total_stats.main_hist_density[ft * 2 + tt] += count;
            }
        }
        total_stats.see_prunes += info.stats.see_prunes;
        total_stats.probcut_cutoffs += info.stats.probcut_cutoffs;
        total_stats.lmr_searches += info.stats.lmr_searches;
        total_stats.recapture_ext += info.stats.recapture_ext;
        total_stats.singular_ext += info.stats.singular_ext;
        total_stats.double_ext += info.stats.double_ext;
        total_stats.negative_ext += info.stats.negative_ext;
        total_stats.multicut += info.stats.multicut;
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
    eprintln!("TT probes:      {:>8}  hits: {} ({:.1}%)  cross-gen hits: {} ({:.1}% of hits)",
        s.tt_probes,
        s.tt_hits,
        if s.tt_probes > 0 { s.tt_hits as f64 / s.tt_probes as f64 * 100.0 } else { 0.0 },
        s.tt_cross_gen_hits,
        if s.tt_hits > 0 { s.tt_cross_gen_hits as f64 / s.tt_hits as f64 * 100.0 } else { 0.0 });
    eprintln!("TT cutoffs:     {:>8}  ({:.1}% of nodes)  cross-gen: {} ({:.1}% of cutoffs)",
        s.tt_cutoffs,
        s.tt_cutoffs as f64 / total_nodes as f64 * 100.0,
        s.tt_cross_gen_cutoffs,
        if s.tt_cutoffs > 0 { s.tt_cross_gen_cutoffs as f64 / s.tt_cutoffs as f64 * 100.0 } else { 0.0 });
    eprintln!("TT near-miss:   {:>8}", s.tt_near_miss);
    eprintln!("NMP attempts:   {:>8}  cutoffs: {} ({:.0}%)", s.nmp_attempts, s.nmp_cutoffs,
        if s.nmp_attempts > 0 { s.nmp_cutoffs as f64 / s.nmp_attempts as f64 * 100.0 } else { 0.0 });
    eprintln!("RFP cutoffs:    {:>8}  ({:.1}% of nodes)", s.rfp_cutoffs, s.rfp_cutoffs as f64 / total_nodes as f64 * 100.0);
    eprintln!("LMP prunes:     {:>8}", s.lmp_prunes);
    eprintln!("Futility prunes:{:>8}", s.futility_prunes);
    eprintln!("History prunes: {:>8}  ({:.1}% of eligible {})", s.history_prunes,
        if s.hist_prune_eligible > 0 { s.history_prunes as f64 / s.hist_prune_eligible as f64 * 100.0 } else { 0.0 },
        s.hist_prune_eligible);
    if s.hist_prune_eligible > 0 {
        let total = s.hist_prune_eligible as f64;
        let pct = |n: u64| n as f64 / total * 100.0;
        eprintln!("Hist-prune score / threshold buckets (sum=eligible):");
        eprintln!("    >= +1.0      (positive history)         {:>8} ({:>5.1}%)", s.hist_prune_ratio_buckets[0], pct(s.hist_prune_ratio_buckets[0]));
        eprintln!("    [0.0, +1.0)                              {:>8} ({:>5.1}%)", s.hist_prune_ratio_buckets[1], pct(s.hist_prune_ratio_buckets[1]));
        eprintln!("    [-0.5, 0.0)                              {:>8} ({:>5.1}%)", s.hist_prune_ratio_buckets[2], pct(s.hist_prune_ratio_buckets[2]));
        eprintln!("    [-1.0, -0.5)  (close to gate)            {:>8} ({:>5.1}%)", s.hist_prune_ratio_buckets[3], pct(s.hist_prune_ratio_buckets[3]));
        eprintln!("    [-1.5, -1.0)  FIRES (just over gate)     {:>8} ({:>5.1}%)", s.hist_prune_ratio_buckets[4], pct(s.hist_prune_ratio_buckets[4]));
        eprintln!("    [-2.0, -1.5)  FIRES                      {:>8} ({:>5.1}%)", s.hist_prune_ratio_buckets[5], pct(s.hist_prune_ratio_buckets[5]));
        eprintln!("    [-3.0, -2.0)  FIRES (deep)               {:>8} ({:>5.1}%)", s.hist_prune_ratio_buckets[6], pct(s.hist_prune_ratio_buckets[6]));
        eprintln!("    < -3.0        FIRES (very deep)          {:>8} ({:>5.1}%)", s.hist_prune_ratio_buckets[7], pct(s.hist_prune_ratio_buckets[7]));
    }
    // Cont-hist read magnitude distribution per offset (sampled at hist-prune gate).
    // Tells us whether deeper offsets are saturating, near-zero, or in the noise band.
    if s.hist_prune_eligible > 0 {
        eprintln!("Cont-hist read magnitudes per offset (sampled at hist-prune gate):");
        eprintln!("    offset    [0,200)   [200,1k)    [1k,5k)   [5k,10k)    [10k+)");
        let labels = ["ply-1", "ply-2", "ply-4", "ply-6"];
        for off in 0..4 {
            let row = &s.cont_hist_mag_buckets[off];
            let row_total: u64 = row.iter().sum();
            if row_total == 0 {
                eprintln!("    {}    (no samples — offset never reached)", labels[off]);
                continue;
            }
            let pct_of = |n: u64| n as f64 / row_total as f64 * 100.0;
            eprintln!("    {}  {:>7}({:>4.1}%) {:>7}({:>4.1}%) {:>7}({:>4.1}%) {:>7}({:>4.1}%) {:>7}({:>4.1}%)",
                labels[off],
                row[0], pct_of(row[0]),
                row[1], pct_of(row[1]),
                row[2], pct_of(row[2]),
                row[3], pct_of(row[3]),
                row[4], pct_of(row[4]));
        }
    }
    // Cont-hist write counts and average magnitudes per offset.
    let total_writes: u64 = s.cont_hist_writes.iter().sum();
    if total_writes > 0 {
        eprintln!("Cont-hist writes per offset (cutoff bonus + non-cutoff malus):");
        let labels = ["ply-1", "ply-2", "ply-4", "ply-6"];
        for off in 0..4 {
            let n = s.cont_hist_writes[off];
            let avg_mag = if n > 0 { s.cont_hist_write_mag_sum[off] as f64 / n as f64 } else { 0.0 };
            eprintln!("    {}  count {:>9}  avg |bonus| {:>7.1}", labels[off], n, avg_mag);
        }
    }
    // What-if hist-prune fire rates with varying cont-hist offset combos.
    // Tells us the marginal benefit of adding each deeper offset to the
    // gate score. Same threshold (HIST_PRUNE_MULT * depth) for each variant.
    let v0 = s.hist_prune_what_if_fires[0];
    if v0 > 0 || s.hist_prune_eligible > 0 {
        eprintln!("Hist-prune what-if fire counts (same threshold, varying score sources):");
        eprintln!("    main + cont[1] + pawn (CURRENT):       {:>8} fires", s.hist_prune_what_if_fires[0]);
        eprintln!("    + cont[2]:                              {:>8} fires", s.hist_prune_what_if_fires[1]);
        eprintln!("    + cont[2,4]:                            {:>8} fires", s.hist_prune_what_if_fires[2]);
        eprintln!("    + cont[2,4,6] (all offsets):            {:>8} fires", s.hist_prune_what_if_fires[3]);
    }
    // Sign agreement between main_hist and sum of cont-hist
    let sign_total: u64 = s.cont_hist_sign_buckets.iter().sum();
    if sign_total > 0 {
        let pct = |n: u64| n as f64 / sign_total as f64 * 100.0;
        eprintln!("Cont-hist vs main_hist sign agreement (at hist-prune gate):");
        eprintln!("    both positive (reinforce good move):   {:>8} ({:>5.1}%)", s.cont_hist_sign_buckets[0], pct(s.cont_hist_sign_buckets[0]));
        eprintln!("    both negative (reinforce bad move):    {:>8} ({:>5.1}%)", s.cont_hist_sign_buckets[1], pct(s.cont_hist_sign_buckets[1]));
        eprintln!("    DISAGREE (cont fights main):           {:>8} ({:>5.1}%)", s.cont_hist_sign_buckets[2], pct(s.cont_hist_sign_buckets[2]));
        eprintln!("    one or both zero:                      {:>8} ({:>5.1}%)", s.cont_hist_sign_buckets[3], pct(s.cont_hist_sign_buckets[3]));
    }
    // Per-offset dominance: which offset has the largest |contribution|
    let dom_total: u64 = s.cont_hist_dominant_offset.iter().sum();
    if dom_total > 0 {
        let pct = |n: u64| n as f64 / dom_total as f64 * 100.0;
        eprintln!("Cont-hist dominant offset (largest |value| at gate):");
        let labels = ["ply-1", "ply-2", "ply-4", "ply-6"];
        for i in 0..4 {
            eprintln!("    {} dominant:                          {:>8} ({:>5.1}%)", labels[i], s.cont_hist_dominant_offset[i], pct(s.cont_hist_dominant_offset[i]));
        }
    }
    // 4D main-history bucket distribution (read counts at hist-prune gate).
    // Tells us if (ft, tt) buckets are evenly populated for cell-density purposes.
    let total_bucket_reads: u64 = s.main_hist_bucket_reads.iter().sum();
    if total_bucket_reads > 0 {
        eprintln!("4D main-history (ft, tt) bucket distribution (sampled at hist-prune gate):");
        let labels = ["[ft=0][tt=0]", "[ft=0][tt=1]", "[ft=1][tt=0]", "[ft=1][tt=1]"];
        for b in 0..4 {
            let n = s.main_hist_bucket_reads[b];
            let pct_b = n as f64 / total_bucket_reads as f64 * 100.0;
            // Density: cells with |val|>1000 out of 4096 cells per bucket * positions
            let density_cells = s.main_hist_density[b];
            eprintln!("    {}  reads {:>8} ({:>5.1}%)   |val|>1000 cells (sum across positions): {:>6}",
                labels[b], n, pct_b, density_cells);
        }
    }
    eprintln!("SEE prunes:     {:>8}", s.see_prunes);
    eprintln!("ProbCut cutoffs:{:>8}", s.probcut_cutoffs);
    eprintln!("LMR searches:   {:>8}  ({:.1}% of nodes)", s.lmr_searches, s.lmr_searches as f64 / total_nodes as f64 * 100.0);
    eprintln!("Recapture ext:  {:>8}", s.recapture_ext);
    eprintln!("Singular ext:   {:>8}  (single +1 ply)", s.singular_ext);
    eprintln!("Double ext:     {:>8}  (additional +1 on top of singular)", s.double_ext);
    eprintln!("Negative ext:   {:>8}  (-1/-2/-3 fail-high reduce)", s.negative_ext);
    eprintln!("Multi-cut:      {:>8}  (return singular_beta)", s.multicut);
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
    eprintln!("TT probes:      {:>6.1}/Kn  hits: {:.1}/Kn  cross-gen hits: {:.1}/Kn",
        s.tt_probes as f64 / kn,
        s.tt_hits as f64 / kn,
        s.tt_cross_gen_hits as f64 / kn);
    eprintln!("TT cutoffs:     {:>6.1}/Kn  cross-gen: {:.1}/Kn", s.tt_cutoffs as f64 / kn, s.tt_cross_gen_cutoffs as f64 / kn);
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

    // Eval-path decomposition — supports the "evals/node" investigation
    // (see docs/coda_vs_reckless_nps_2026-04-23.md). Reports how search
    // splits its static-eval calls between full rebuilds, incremental
    // updates, already-computed skips, and TT-cached bypasses.
    if let Some(acc) = info.nnue_acc.as_ref() {
        let full = acc.stats_full_rebuilds;
        let incr = acc.stats_incremental_updates;
        let skip = acc.stats_cached_skips;
        let tt   = info.stats_tt_static_eval_hits;
        let total_evals = full + incr;
        let eval_call_attempts = total_evals + skip + tt;
        eprintln!("--- NNUE Eval Decomposition ---");
        eprintln!("NNUE full rebuilds: {:>10} ({:>5.2}% of evals)", full,
                  if total_evals > 0 { full as f64 * 100.0 / total_evals as f64 } else { 0.0 });
        eprintln!("  by cause:  king-bucket={} ({:>5.2}%)  root={} ({:>5.2}%)  chain-break={} ({:>5.2}%)",
                  acc.stats_rebuild_kind0,
                  if full > 0 { acc.stats_rebuild_kind0 as f64 * 100.0 / full as f64 } else { 0.0 },
                  acc.stats_rebuild_root,
                  if full > 0 { acc.stats_rebuild_root as f64 * 100.0 / full as f64 } else { 0.0 },
                  acc.stats_rebuild_chain,
                  if full > 0 { acc.stats_rebuild_chain as f64 * 100.0 / full as f64 } else { 0.0 });
        eprintln!("NNUE incremental:   {:>10} ({:>5.2}% of evals)", incr,
                  if total_evals > 0 { incr as f64 * 100.0 / total_evals as f64 } else { 0.0 });
        eprintln!("TT static-eval hit: {:>10} ({:>5.2}% of call sites)", tt,
                  if eval_call_attempts > 0 { tt as f64 * 100.0 / eval_call_attempts as f64 } else { 0.0 });
        eprintln!("Already-computed:   {:>10}", skip);
        eprintln!("Evals / node:       {:>10.3}", total_evals as f64 / total_nodes as f64);
        eprintln!("Call-sites / node:  {:>10.3}", eval_call_attempts as f64 / total_nodes as f64);
    }

    #[cfg(feature = "profile-materialize")]
    crate::nnue::mat_stats::report();

    #[cfg(feature = "profile-threats")]
    {
        crate::threats::thr_stats::report();
        crate::threats::apply_stats::report();
        crate::threats::refresh_stats::report();
    }

    total_nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 50-move eval scaling helper. Locks in both the formula (linear decay
    /// over 100 plies, reaches zero at the 50-move claim cliff) and the
    /// sentinel preservation that downstream search relies on when
    /// comparing against `-INFINITY` and mate scores.
    #[test]
    fn test_apply_halfmove_scale() {
        // Linear decay from full to zero across the draw horizon.
        assert_eq!(apply_halfmove_scale(100, 0), 100);
        assert_eq!(apply_halfmove_scale(100, 25), 75);
        assert_eq!(apply_halfmove_scale(100, 50), 50);
        assert_eq!(apply_halfmove_scale(100, 75), 25);
        assert_eq!(apply_halfmove_scale(100, 99), 1);
        assert_eq!(apply_halfmove_scale(100, 100), 0);
        // Sign preserved.
        assert_eq!(apply_halfmove_scale(-400, 50), -200);
        // Saturation past 100 (hm > 100 is normally intercepted as a draw,
        // but the scale function still clamps so it never flips sign).
        assert_eq!(apply_halfmove_scale(100, 150), 0);
        // Zero in → zero out at any hm.
        assert_eq!(apply_halfmove_scale(0, 50), 0);
        // Sentinel scores are not scaled — comparisons against -INFINITY /
        // mate-adjusted scores in the search body rely on this.
        assert_eq!(apply_halfmove_scale(-INFINITY, 50), -INFINITY);
        assert_eq!(apply_halfmove_scale(MATE_SCORE - 5, 99), MATE_SCORE - 5);
        assert_eq!(apply_halfmove_scale(-(MATE_SCORE - 5), 99), -(MATE_SCORE - 5));
    }

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
            movetime_floor: 0,
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
    /// writes for the same position.
    ///
    /// Tested two ways:
    /// 1. Direct entry check — after one update, the per-table slots
    ///    indexed by the test position must be non-zero, while a
    ///    reference position's slots remain zero. Independent of
    ///    `CORR_HIST_ERR_MAX_10X` / `CORR_HIST_GRAIN_T` defaults.
    /// 2. corrected_eval drift — after enough updates to escape
    ///    integer-division flooring, corrected_eval(test_pos) must
    ///    rise above raw, while corrected_eval(reference_pos) must
    ///    stay near raw.
    ///
    /// Using a position with distinctive piece layout so hash
    /// collisions with the all-zero-state are unlikely.
    #[test]
    fn corr_read_write_index_symmetry() {
        use crate::board::Board;
        crate::init();

        let mut info = SearchInfo::new(16);
        info.silent = true;

        // Distinctive position vs fresh startpos — different
        // pawn_hash, non_pawn_key.
        let board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let other = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let raw = 100;
        // Before any update: corrected == raw (all tables zero).
        assert_eq!(corrected_eval(&info, &board, raw), raw,
            "zero tables must give corrected == raw");

        // === Part 1: direct entry check after one update ===
        update_correction_history(&mut info, &board, raw + 400, raw, 20);

        let stm = board.side_to_move as usize;
        let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
        let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
        let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);

        // The slot indexed by the test position's hash must be non-zero
        // in every per-position table. cont_corr is excluded — needs a
        // last-move undo entry, which the test position doesn't have.
        assert!(info.pawn_corr[stm][pawn_idx] != 0,
            "pawn_corr slot must be written");
        assert!(info.np_corr[stm][WHITE as usize][white_np_idx] != 0,
            "white np_corr slot must be written");
        assert!(info.np_corr[stm][BLACK as usize][black_np_idx] != 0,
            "black np_corr slot must be written");

        // Apply repeatedly to escape integer-division flooring at
        // current default grain (CORR_HIST_GRAIN_T=11). Each call
        // bumps each slot by the gravity-clamped bonus; ~30 iterations
        // is enough to push entries near steady-state given the small
        // err clamp (CORR_HIST_ERR_MAX_10X=10, effective 1).
        for _ in 0..50 {
            update_correction_history(&mut info, &board, raw + 400, raw, 20);
        }

        // === Part 2: corrected_eval drift ===
        let corrected_after = corrected_eval(&info, &board, raw);
        assert!(
            corrected_after > raw,
            "after sustained positive-err updates, corrected eval must rise: \
             raw={} corrected={}",
            raw, corrected_after
        );

        // Reference position: pawn_hash / non_pawn_key / minor / major
        // are entirely different from the test fen, so any match would
        // be a 1/16384 random collision — extremely unlikely.
        let other_corrected = corrected_eval(&info, &other, raw);
        let drift = (other_corrected - raw).abs();
        assert!(drift < 100,
            "unrelated position should see near-zero drift, got {} (raw {})",
            other_corrected, raw);
    }
}

