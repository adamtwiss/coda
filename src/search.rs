//! Negamax alpha-beta search with iterative deepening, PVS, aspiration windows, and Lazy SMP.
//! Features: NMP, RFP, LMR, LMP, futility, SEE pruning,
//! singular extensions, cuckoo cycle detection, correction history.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use crate::bitboard::*;
use crate::board::Board;
use crate::eval::{evaluate_nnue, see_value};
use crate::nnue::DirtyPiece;
use crate::movegen::generate_legal_moves;
use crate::movepicker::*;
use crate::see::see_ge;
use crate::tt::*;
use crate::types::*;

/// Maximum ply depth supported by per-SearchInfo arrays.
///
/// Public because MovePicker and History share the cap for fixed-size
/// ply-indexed arrays (pv_table, moved_piece_stack, etc.).
///
/// The iterative-deepening cap is `MAX_PLY / 2` (see `effective_max`), so
/// this value is twice the deepest nominal depth; the other half is headroom
/// for extensions and QS running ahead of nominal depth.
///
/// Raising this is not free: `pv_table` is O(MAX_PLY²) and sits on the hot
/// pv-copy path, so an increase can spill L1 and cost more than the extra
/// depth is worth — an earlier bump did exactly that, for about -13 Elo at
/// STC. That was measured when `Move` was 4 bytes; at `u16` the footprint is
/// half, so the pressure is materially lower now.
///
/// UPPER BOUND — do not exceed 199. `MATE_IN_MAX_PLY = MATE_SCORE - MAX_PLY`
/// (tt.rs) must stay strictly above `TB_WIN = 28800`, or tablebase scores are
/// misclassified as mates by `is_mate_score`. At 160 the mate band starts at
/// 28840, clearing TB by 40cp; at 256 it would be 28744 and the whole TB band
/// would read as mate. A unit test in tt.rs asserts this invariant.
pub const MAX_PLY: usize = 160;

/// Maximum ROOT depth for iterative deepening.
///
/// This was `MAX_PLY / 2`, a stack-safety margin left so that extensions below
/// the root could not push `ply` past the MAX_PLY-sized stacks. That margin is
/// redundant: `negamax` hard-guards `ply_u >= MAX_PLY` and returns a static
/// eval, so a deep root cannot overrun anything — an over-extended line simply
/// stops getting searched, exactly as it does today.
///
/// The halving cost us the entire deep-endgame tail. In the 2026-08 CCRL corpus
/// every one of the 78 moves that hit the ceiling was an endgame (median 6
/// pieces), and Coda returned in a median 3s where a position one ply shallower
/// took 11-13s — it had run out of DEPTH, not time, while peers in the same
/// event reported up to 256.
///
/// 3/4 keeps 40 plies of extension headroom instead of 80. It cannot change any
/// search that does not reach depth 80, so bench is unaffected.
pub const ROOT_DEPTH_MAX: i32 = (MAX_PLY as i32) * 3 / 4;
const INFINITY: i32 = 30000;

// Pawn history table size
const PAWN_HIST_SIZE: usize = 512;
const ROOT_MOVE_BUCKETS: usize = 5;
const ROOT_MOVE_TABLE_SIZE: usize = 64 * 64 * ROOT_MOVE_BUCKETS;

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
    // THIS MACRO IS THE AUTHORITATIVE LIST of search tunables — their defaults,
    // ranges, SPSA c_end, and --core membership. Nothing else should restate
    // them; `coda tune-spec` emits the live values on demand, which is why a
    // checked-in SPSA spec is guaranteed to be the stale one.
    //
    // The DEFAULTS ARE SPSA OUTPUT and move on every applied tune. Comments
    // here should therefore describe what a parameter DOES and what constrains
    // it — mechanism, interactions, and any range that is deliberately narrow —
    // NOT the sequence of adjustments that produced today's number. A comment
    // recording one tune's movers is stale by the next tune and will be read as
    // current intent.
    //
    // RANGES ARE PART OF THE DESIGN. A floor or ceiling that a parameter pins
    // against is suppressing gradient, not expressing an optimum; several here
    // run to 0 specifically so SPSA can disable a dead term. Where a bound
    // instead exists to STOP SPSA (play-quality guards), the comment says so.
    //
    // `_10X` PARAMETERS ARE FIXED-POINT, consumed via `tp10` = (v+5)/10, which
    // ROUNDS rather than truncates. Whole raw bands therefore collapse to one
    // effective value, so such a parameter can post a large SPSA percentage
    // while changing nothing at all. Check the bucket before acting on a mover.
    (NMP_BASE_R_10X, 78, 20, 80, 15.0, true),
    (NMP_DEPTH_DIV_10X, 47, 10, 200, 15.0, true),
    (NMP_EVAL_DIV, 75, 50, 400, 17.5, true),
    (NMP_EVAL_MAX_10X, 35, 10, 60, 5.0, false),
    // Depth at/above which an NMP cutoff must be re-searched to verify it.
    // Must stay ABOVE the min-depth gate: if it sits below, every NMP cutoff
    // pays a verification re-search and NMP never gets a cheap cutoff. Below
    // this depth the cutoff is taken unverified; above it the re-search acts
    // as the zugzwang guard.
    (NMP_VERIFY_DEPTH_10X, 68, 40, 200, 20.0, true),
    (RFP_DEPTH, 17, 2, 20, 2.0, true),
    (RFP_MARGIN_IMP, 24, 0, 150, 6.0, true),
    (RFP_MARGIN_NOIMP, 35, 0, 200, 7.5, true),
    // Root-depth-aware RFP relaxation (single-set, self-adapts STC<->LTC):
    // demand MORE static-eval confidence to RFP-cut as the OVERALL search
    // depth grows past RFP_ROOT_THRESH (diminishing-returns of depth — the
    // marginal ply is cheap at LTC so deep pruning trades blindness for
    // worthless depth); relaxes deep RFP at LTC. SPSA tunes both.
    //
    // NOT STC-neutral — same correction as the LMR_ROOT_* block below. Gated on
    // root_depth, not on TC. Warm-TT measurement at 250ms/move: root_depth > 18
    // on 39% of moves overall — 0% in the opening but 92% through the late
    // middlegame. Cold-TT probes understate this badly.
    (RFP_ROOT_THRESH, 17, 6, 30, 1.5, true),
    (RFP_ROOT_COEF, 17, 0, 150, 7.5, false),
    // Additional depth-local RFP relaxation: current main already scales RFP
    // by overall root depth; this term raises the margin for high remaining
    // depth regardless of TC. Consensus engines either cap RFP around d9-11
    // or use a quadratic/deepening margin so static eval does not keep
    // cheaply pruning d12+ nodes.
    (RFP_DEEP_KNEE_10X, 41, 40, 170, 20.0, true),
    (RFP_DEEP_LINEAR, 50, 0, 200, 10.0, true),
    // Razoring: drop straight to qsearch when static eval is far enough below
    // alpha that a full search is unlikely to recover it. Margin scales with
    // depth, gated to shallow depths only.
    (RAZOR_MULT, 286, 100, 500, 20.0, false),
    (RAZOR_DEPTH_10X, 39, 10, 80, 5.0, true),
    // Futility margin: base + per-depth, compared against alpha at the
    // frontier. History adjusts the effective lmr_depth used here, so these
    // interact with the LMR history terms — retune the pair together.
    (FUT_BASE, 77, 0, 200, 9.0, true),
    (FUT_PER_DEPTH, 101, 40, 250, 10.5, true),
    // Strong-history exemption for quiet futility: a quiet whose main history
    // exceeds this is never futility-pruned. It was the hardcoded literal 12000
    // sitting inside the gate at search.rs while every other term around it was
    // SPSA-tunable -- so SPSA has been optimising a formula with one frozen
    // input. Exposing it costs nothing and is behaviour-identical at 12000.
    (FUT_HIST_EXEMPT, 12000, 2000, 16384, 900.0, true),
    (FUT_LMR_DEPTH, 14, 6, 24, 2.0, true),
    (SEE_QUIET_MULT, 23, 5, 80, 3.75, true),
    // Low-increment TM multiplier ceiling. The factor product
    // (stability×fail-low×forced×subtree×score-trend, up to ~13.8×) is only
    // clamped for no_inc; at increments that are SMALL RELATIVE TO THE CLOCK
    // it ran uncapped, so a complex middlegame drew deep on a RUN of moves the
    // increment can't refill -> clock drained by early middlegame -> flag
    // (seen at rapid 10+1 = 600s+1s). The discriminator is increment relative
    // to the per-move budget: inc_cover = inc / (timeLeft/mtg). Cap by an
    // inc_cover-scaled ceiling: cmin at inc_cover->0 (starved), rising to cmax
    // (≈ uncapped) at inc_cover >= TM_INC_COVER_REF/100.
    //   inc_cover ≈ 0.04 at 600+1 (capped);  0.24 at 10+0.1
    //   and 0.4 at 600+10 (both ~uncapped) — so short, long and big-inc TCs
    //   are untouched, and only true low-inc-vs-clock (rapid) is throttled.
    // NB: keying this on ABSOLUTE increment rather than inc_cover does not work
    // — an inc/12000 form gave a 1.6x cap at inc=100ms, crushing short TCs for
    // about -24 Elo. _10X ceilings are /10.
    //
    // WHY THE WHOLE TM_* BLOCK IS NON-CORE: not because TM is invisible at STC
    // — it is one of the highest-leverage levers in BOTH directions, and a bad
    // TM form has cost ~24 Elo. It is excluded from the routine --core sweep
    // precisely because it is that high-leverage and deployment-critical: TM
    // wants deliberate, TC-matched, cross-engine and ponder-validated tuning,
    // not incidental perturbation by a broad STC sweep where a noisy movement
    // regresses real games. Still UCI-loadable for deliberate TM tunes.
    (TM_INC_COVER_REF, 20, 5, 60, 4.0, false),
    (TM_MULT_CEIL_MIN_10X, 15, 10, 40, 2.0, false),
    (TM_MULT_CEIL_MAX_10X, 130, 40, 140, 8.0, false),
    // Cross-thread best-move-instability TM factor (concept from SF).
    // factor = BASE/1000 + MULT/1000 * (Σ per-thread bmc)/n_threads, applied
    // to the soft budget only at Threads>1. Defaults are SF's 1.088 / 2.315
    // instability shape; will want a focused TM-cluster retune-on-branch since
    // Coda's TM uses the standard opt/hard/max + factor-product shape.
    // Fixed-point /1000 for the sub-integer precision these multiplicative
    // constants need.
    // BASE defaults to 1000 (=1.0), NOT SF's 1088: SF's base is balanced
    // against SF's OWN factor product; on Coda's already-calibrated product a
    // >1.0 base would add a blanket ~9% time to EVERY position (settled or not),
    // contaminating the raw test. At 1.0 the factor is neutral when the pool
    // agrees and only scales UP on genuine cross-thread churn — the retune can
    // lift BASE if beneficial. MULT starts at SF's 2.315.
    (TM_BMC_INSTAB_BASE, 1000, 900, 1500, 25.0, false),
    (TM_BMC_INSTAB_MULT, 2315, 500, 4000, 100.0, false),
    // Subtree factor = (BASE/100 - best_move_node_fraction) * 1.4, floor 0.55
    // (the floor cannot bind at the 1.62 default — frac would need to exceed
    // 1.23).
    //
    // DO NOT TUNE THIS IN SELF-PLAY. The factor inflates ~66% of moves and
    // looks over-generous, but re-centring it measured EVEN in self-play while
    // costing ~48 Elo against stronger opponents. "Our best move dominated our
    // own search" is a poor confidence proxy against a stronger engine, so the
    // upward bias is deliberate insurance — and self-play cannot see the cost.
    // Cross-engine RR only.
    (TM_SUBTREE_BASE_100, 162, 100, 180, 4.0, false),
    // Low-inc absolute single-move ceiling — the companion to the inc_cover
    // multiplier cap above.
    // inc_cover caps the factor MULTIPLIER, so adjusted_soft stays
    // ~11% of clock — but a single deep iteration that starts just under
    // adjusted_soft runs uninterrupted (the soft check only fires BETWEEN
    // iterations) until the mid-iteration hard check stops it at hard = 46%
    // of clock. At low-inc-ratio TCs (600+1, inc/base ~0.0017) the engine
    // reaches deep enough for one iteration to span soft->hard, so a single
    // move eats 46% of the clock — repeatedly, geometric-draining it. This
    // lost several live games before it was fixed.
    // Fix: lower the hard/max ceiling directly when the increment is small,
    // keyed on the (constant) increment so it never flips mid-game:
    //   inc_ceiling = inc * TM_INC_HARD_MULT + TM_INC_HARD_FLOOR_MS
    // MULT=30, FLOOR=10s leaves standard OB TCs (10+0.1/40+0.4/60+0.6) and
    // rich TCs (600+10) essentially untouched (the 46%/60% windows still
    // bind), while capping 600+1 at 40s (was 276s) and 60+0.1 at 13s.
    (TM_INC_HARD_MULT, 30, 0, 120, 4.0, false),
    (TM_INC_HARD_FLOOR_MS, 10000, 0, 60000, 1000.0, false),
    // No-inc adaptive mtg divisor: base assumed moves-to-go, and the growth
    // rate once a game outlives that assumption — see compute_tm_budgets for
    // the full derivation.
    (NO_INC_MTG_BASE, 34, 20, 80, 4.0, false),
    (NO_INC_MTG_GROWTH_PCT, 94, 0, 200, 10.0, false),
    // TM window + factor constants. Coda's own parameters, tuned and validated
    // on Coda's own search and net; provenance in
    // docs/license_analysis_2026-07-13.md. All non-core, for the reason given
    // in the TM_* block above. Note TM constants are bench-invariant (bench is
    // fixed-depth and never consults the budgets), so a TM tune cannot move the
    // bench — do not read an unchanged bench as "the tune did nothing".
    (TM_MAX_BANK_1000, 599, 400, 750, 15.0, false),   // max_time = clock * N/1000
    (TM_HARD_WINDOW_PCT, 47, 25, 65, 2.5, false),     // hard_time = clock * N/100
    (TM_OPT_WINDOW_PCT, 73, 45, 95, 3.0, false),      // opt = computed * N/100
    (TM_INC_FRAC_PCT, 93, 40, 100, 4.0, false),       // computed += inc * N/100
    (TM_DEFAULT_MTG, 24, 14, 40, 1.5, false),         // sudden-death moves-to-go
    (TM_STAB_0_100, 175, 100, 260, 8.0, false),       // stability table [0] * 1/100
    (TM_STAB_1_100, 122, 80, 180, 5.0, false),        // stability table [1]
    (TM_STAB_2_100, 90, 60, 130, 3.0, false),         // stability table [2]
    (TM_STAB_3_100, 80, 50, 120, 3.0, false),         // stability table [3]
    (TM_STAB_4_100, 74, 40, 110, 3.0, false),         // stability table [4+]
    (TM_FAIL_LOW_BONUS_1000, 350, 100, 700, 20.0, false), // 1 + N/1000 * fail_lows
    (TM_FORCED_STRONG_1000, 381, 150, 700, 20.0, false),  // strong-forced * N/1000
    (TM_FORCED_WEAK_1000, 631, 300, 950, 25.0, false),    // weak-forced * N/1000
    (TM_SUBTREE_MULT_100, 140, 90, 200, 4.0, false),      // (base-frac) * N/100
    (TM_FORCED_MARGIN_WEAK, 170, 80, 320, 8.0, false),    // weak-forced cp margin
    (TM_FORCED_MARGIN_STRONG, 400, 200, 620, 12.0, false),// strong-forced cp margin
    (LMR_HIST_DIV, 20620, 2000, 100000, 4900.0, true),
    // Capture-LMR history divisor. Separate from the quiet divisor above:
    // capture history is single-source, so it needs a smaller divisor than
    // quiet history to produce an equivalent reduction magnitude. Both are
    // continuous (`R -= hist / DIV`), not stepped.
    (LMR_HIST_DIV_CAP, 3152, 1000, 20000, 1500.0, true),
    (LMR_C_QUIET, 169, 40, 300, 13.0, true),
    (LMR_C_CAP, 240, 80, 350, 12.5, true),
    // Two independent degrees of freedom on the LMR curve, both in centi-ply
    // (they need the fractional accumulator to express):
    //
    //   BASE      shifts the curve's INTERCEPT — a constant offset on every
    //             reduction.
    //   DECAY_NUM changes the curve's SHAPE — multiplicative all-node
    //             inflation `r += r*NUM/(256d+285)`, so proportionally MORE
    //             reduction shallow and LESS deep. A flat +1-ply all-node
    //             bump is the wrong shape for this.
    (LMR_BASE_CENTI, 41, 0, 120, 6.0, true),
    (LMR_ALLNODE_DECAY_NUM, 427, 0, 1600, 80.0, true),
    // Cut-node LMR bump, in centi-ply. Cut nodes reduce by this amount
    // (plus a further ply with no TT move); all-nodes keep +1.
    (LMR_CUTNODE_BUMP_CENTI, 184, 100, 500, 40.0, true),
    // LMR correction battery — sub-ply terms in centi-ply, needing the
    // fractional accumulator to express. These were seeded well below their
    // source values because Coda's ln(d)·ln(m) base already carries a
    // move-count term that the source shape omits, so the constants would
    // otherwise double-count. Ranges run to 0 so SPSA can kill dead terms.
    (LMR_WINBETA_CENTI, 32, 0, 250, 12.0, false),
    (LMR_TTALPHA_CENTI, 17, 0, 150, 8.0, true),
    (LMR_EXPECT_MULT, 17, 0, 120, 6.0, true),
    // cutoff_count LMR terms. When the child ply has failed high more than
    // twice under this node, reduce late moves more (with extra at non-PV
    // all-nodes). Seeded below source values for the double-counting reason
    // above. The >2 threshold is fixed, not a knob.
    (LMR_CUTOFF_CNT_CENTI, 67, 0, 250, 12.0, true),
    (LMR_CUTOFF_ALLNODE_CENTI, 18, 0, 150, 8.0, true),
    // Minimum depth at which singular extension is attempted. Too low and
    // singular_depth is itself too shallow to judge singularity reliably.
    (SE_DEPTH_10X, 41, 40, 200, 20.0, true),
    (ASP_DELTA, 11, 5, 30, 1.5, false),
    (ASP_SCORE_DIV, 12000, 8000, 50000, 2100.0, false),
    // Late move pruning: quiets searched before the cutoff, on the shape
    // `(BASE + d²·DEPTH) / (2 - improving)`. BASE dominates at shallow depth
    // and sets how many quiets survive at d=1.
    (LMP_BASE_10X, 55, 10, 150, 20.0, true),
    (LMP_DEPTH_10X, 130, 40, 200, 20.0, true),
    // Margin-aware LMP. Coda's LMP limit keys on depth and improving only; the
    // fail-low histogram (b_probe_*) shows late-quiet work concentrating at
    // nodes that end up failing low, but its margin is only knowable AFTER the
    // node completes. These give the count limit a PREDICTIVE margin
    // dimension: when the static eval is already LMP_MARGIN_THRESH below alpha,
    // scale the limit to LMP_MARGIN_PCT percent so late quiets are cut sooner.
    //
    // Prior measurement says expect little: the safely-prunable tail (post-hoc
    // margin >= 150cp) is ~1% of all move-searches, measured 2026-07-27 and
    // again on the v10 net 2026-08-17, and the idea was closed then without
    // implementation. Tested anyway at Adam's request (2026-08-17) since the
    // predictor here differs from the post-hoc selector that was measured.
    (LMP_MARGIN_THRESH, 75, 50, 500, 35.0, true),
    (LMP_MARGIN_PCT, 60, 40, 100, 6.0, true),
    // Root-depth-aware LMR relaxation (single-set, self-adapts STC<->LTC):
    // reduce LESS as the OVERALL search depth grows past LMR_ROOT_THRESH
    // (diminishing returns — at LTC the reduced re-search is cheap vs the
    // budget and a wrong reduction costs more). SPSA tunes both.
    //
    // NOT STC-neutral — the gate is root_depth, and nothing here sees the
    // clock. Measured on a 200-ply game replayed with a WARM TT at 250ms/move
    // (2026-08-17), root_depth > 15 on 61% of moves overall, and the effect is
    // phase-concentrated: 7% in the opening, 58% in the middlegame, 100% from
    // the late middlegame into the endgame. A cold-TT probe shows 0% and is
    // what makes this term look dead — warm caches roughly double reached
    // depth, so measure with the TT warm or the answer inverts.
    (LMR_ROOT_THRESH, 15, 6, 30, 1.5, false),
    (LMR_ROOT_COEF_10X, 9, 0, 800, 40.0, true),
    (BAD_NOISY_MARGIN, 79, 30, 150, 6.0, true),
    (PROBCUT_MARGIN, 153, 80, 300, 11.0, true),
    // ProbCut margin reduction when improving (Stockfish/Alexandria shape):
    // improving positions verify against a lower beta, non-improving nodes
    // keep the safer base margin. Effective improving margin is
    // PROBCUT_MARGIN - PROBCUT_MARGIN_IMP.
    (PROBCUT_MARGIN_IMP, 46, 0, 120, 8.0, true),
    // Root-depth-aware ProbCut: a more conservative margin is wanted at
    // shallow root depths than deep ones, so add an offset below
    // PROBCUT_ROOT_THRESH and fade it out as root depth grows.
    (PROBCUT_ROOT_THRESH, 16, 8, 28, 1.5, true),
    (PROBCUT_ROOT_FADE_10X, 32, 10, 120, 10.0, true),
    (PROBCUT_ROOT_MARGIN, 69, 0, 120, 8.0, false),
    (HINDSIGHT_THRESH, 152, 50, 400, 17.5, true),
    (QS_DELTA_MARGIN, 374, 100, 500, 20.0, true),
    // Cap on captures actually SEARCHED in qsearch (delta/SEE-pruned moves
    // are not charged against it). Counting pruned moves here would let SPSA
    // detune the cap to near-off, which is what an earlier counting bug did.
    (QS_MAX_CAPTURES, 3, 2, 32, 2.0, false),
    (CORR_W_PAWN, 125, 100, 600, 25.0, true),
    (CORR_W_NP, 144, 0, 400, 17.5, true),
    // There is deliberately no minor-key or major-key correction source:
    // both are strict subsets of non_pawn_key, so such terms are redundant
    // with np_corr and simply consume SPSA budget at weight 0.
    (CORR_W_CONT, 215, 0, 400, 18.5, true),
    // Transition (zobrist-delta) correction weight (Cinder idea): correction
    // keyed by hash(ply-1) ^ hash(ply) — a hash of the last move IN CONTEXT
    // (from+to+captured+side), richer than cont_corr's [piece][to]. Captures
    // "this structural CHANGE tends to be mis-evaluated."
    (CORR_W_TRANS, 101, 0, 400, 18.5, true),
    (FH_BLEND_DEPTH_10X, 21, 0, 80, 15.0, false),
    // TT_DAMP_TT_WEIGHT: weight of tt_score in TT-LOWER non-PV cutoff score
    // dampening. Formula: (W*tt_score + beta) / (W+1).
    (TT_DAMP_TT_WEIGHT_10X, 31, 10, 100, 5.0, false),
    // PROBCUT_TT_DEPTH_SLACK: TT depth must be >= current depth - SLACK for
    // ProbCut-TT-noshot to consider the entry.
    (PROBCUT_TT_DEPTH_SLACK, 3, 0, 10, 0.5, false),
    (HIST_BONUS_MULT, 245, 50, 400, 17.5, true),
    (HIST_BONUS_MAX, 1653, 500, 3000, 125.0, true),
    // History bonus uses the offset shape `clamp(0, MAX, MULT*d - OFFSET)`
    // rather than `min(MAX, MULT*d)`. Without the offset the formula
    // saturates early and d=5 and d=10 earn the same bonus; the offset buys
    // depth discrimination. Capture history uses the same shape.
    (HIST_BONUS_OFFSET, 18, 0, 400, 25.0, false),
    (CAP_HIST_MULT, 324, 50, 400, 17.5, true),
    (CAP_HIST_MAX, 1997, 500, 3000, 125.0, true),
    // Malus constants are SEPARATE from the bonus constants rather than
    // hardwired to -bonus, so SPSA can tune the two slopes independently.
    // Whether malus should be steeper or shallower than bonus is
    // engine-specific and only discoverable by tuning.
    (HIST_MALUS_MULT, 558, 50, 900, 40.0, true),
    (HIST_MALUS_OFFSET, 30, 0, 400, 25.0, false),
    (HIST_MALUS_MAX, 1037, 500, 4000, 175.0, true),
    (CAP_HIST_MALUS_MULT, 278, 50, 900, 40.0, true),
    (CAP_HIST_MALUS_BASE, 42, 0, 400, 25.0, false),
    (CAP_HIST_MALUS_MAX, 2325, 500, 4000, 175.0, true),
    // numFailHighs multiplicative history scaling (Starzix pattern):
    //   bonus = raw + raw * min(num_fail_highs, NFH_CAP) / NFH_DIV
    // so 0..NFH_CAP cascades produce 1.0x .. (1 + NFH_CAP/NFH_DIV)x bonus.
    (NFH_CAP_10X, 33, 10, 60, 10.0, false),
    // Consumed as FIXED-POINT (stored/10) so SPSA keeps sub-integer
    // precision on the divisor.
    (NFH_DIV_10X, 49, 20, 120, 10.0, false),
    // Sibling-count history-bonus scaling (Stockfish pattern). At non-PV
    // cutoffs, amplify the best move's bonus by
    // (quiets+caps searched)/HIST_SIBLING_DIV — a move that cut off after
    // more competition proved itself more strongly.
    (HIST_SIBLING_DIV, 172, 64, 1024, 40.0, true),
    // PV/quiet/correction-aware double-extension margin (Stockfish shape).
    //
    // dext_margin = DEXT_MARGIN_PV   * is_pv
    //             - DEXT_MARGIN_QUIET * is_tt_quiet
    //             - DEXT_MARGIN_CORR * |corr| / 128
    //             + DEXT_MARGIN_BASE
    //
    // The BASE term is Coda-specific and load-bearing: without it the margin
    // goes negative at non-PV quiet nodes, so a double extension fires on
    // every singular hit and the tree explodes. BASE shifts the non-PV
    // baseline to a positive threshold.
    //
    // CORR modulator reduces threshold when correction history has been
    // correcting — extend less on uncertain evals.
    //
    // TRIPLE extension is intentionally NOT part of this shape — it has been
    // tested for Coda's regime and the signal was not there.
    (DEXT_MARGIN_PV, 169, 50, 400, 15.0, false),
    (DEXT_MARGIN_QUIET, 17, 0, 100, 4.0, false),
    (DEXT_MARGIN_CORR, 12, 0, 64, 3.0, true),
    (DEXT_MARGIN_BASE, 41, -50, 150, 6.0, true),
    (DEXT_CAP, 8, 4, 32, 2.0, true),
    (QUIET_CHECK_BONUS, 14805, 2000, 30000, 1400.0, false),
    // SEE gate on the quiet check bonus (SF movepick.cpp: check bonus only
    // applies when see_ge(m, -75)). Without it Coda orders losing check-sacs
    // into the first-searched slot. Margin on Coda's pawn=100 SEE scale:
    // a check that loses more than this by SEE gets no ordering bonus.
    (QUIET_CHECK_SEE_MARGIN, 81, 0, 300, 12.0, true),
    // Effective correction magnitude is sum(W) / (DIV * GRAIN_T), so this
    // divisor trades off directly against the CORR_W_* weights — the pair is
    // degenerate and must be read together, never one in isolation. The floor
    // is deliberately low: a bound this parameter pins against would be setting
    // the value instead of the optimum. Keep c_end well under the operating
    // point, or perturbations clamp against the floor every iteration.
    (CORR_HIST_DIV, 323, 64, 4096, 64.0, true),
    // Caps the per-update weight. The ceiling suits depth-proportional weights;
    // a much lower cap only makes sense in a sign-only (error-clamped) regime.
    (CORR_UPDATE_WEIGHT_MAX, 17, 4, 48, 2.2, true),
    // Fixed-point /10.
    (CORR_BONUS_CAP_DIV_10X, 38, 10, 160, 15.0, false),
    (CORR_HIST_GRAIN_T, 13, 1, 32, 1.55, false),
    // Correction-history output scaling — the output is scaled rather than the
    // input pre-clamped:
    //   bonus = err * (depth+1).min(W) / CORR_ERR_DIV
    // clamped at the gravity cap only.
    (CORR_ERR_DIV_10X, 55, 20, 640, 30.0, false),
    (ESCAPE_BONUS_R, 8181, 3000, 30000, 1350.0, false),
    // Threat-escape ordering bonuses by escaping piece type. Ablating the
    // queen/minor terms measured only slightly load-bearing, so they are kept
    // tunable rather than hardcoded — their optimum drifts with the
    // cont-hist / NMP / margin shape around them.
    (ESCAPE_BONUS_Q, 17819, 0, 30000, 1500.0, false),
    (ESCAPE_BONUS_MINOR, 5250, 0, 30000, 1000.0, false),
    // Null-move threat-escape bonus in quiet ordering.
    (NULL_THREAT_ESCAPE_BONUS, 8321, 0, 30000, 1000.0, false),
    (NMP_KING_ZONE_MAX_10X, 26, 20, 90, 15.0, true),
    (PROBCUT_KING_ZONE_MAX_10X, 71, 20, 90, 15.0, true),
    (LMR_THREAT_DIV_10X, 32, 10, 50, 15.0, true),
    (LMR_KING_PRESSURE_DIV_10X, 72, 20, 90, 15.0, true),
    // Reduce later moves more once this node has already raised alpha N times
    // (alpha_raises reduction, a known LMR refinement). Fixed-point ×10: reduction += raises *
    // VALUE/10. Only fires at PV nodes (cut nodes break on the first fail-high
    // before alpha is raised). Default 10 = +1.0 reduction per prior alpha-raise.
    (LMR_ALPHA_RAISE_10X, 5, 0, 40, 5.0, false),
    (FUT_THREATS_MARGIN, 52, 0, 200, 10.0, true),
    (DISCOVERED_ATTACK_BONUS, 0, 0, 30000, 1500.0, false),
    // xray-SE: when the TT move is from an x-ray blocker square (moving it
    // uncovers our slider's attack on an enemy), this flat bonus is
    // SUBTRACTED from singular_beta (`singular_beta = tt_score - depth -
    // xray_bonus`). That LOWERS singular_beta → WIDENS the SE margin →
    // STRICTER singularity test → FEWER extensions on x-ray-blocker TT moves,
    // not more. This reads backwards but is correct: SPSA drives the value UP
    // away from the 0 floor, so do NOT "fix" the sign.
    // NON-CORE, and this is not a judgement call: the parameter is PROVABLY
    // INERT at every value in its range. Its only consumer gates on
    // `our_xray_blockers`, which is computed as 0 whenever
    // DISCOVERED_ATTACK_BONUS is 0 -- and that has been 0 since SPSA #660 drove
    // it there in the v9 era. Measured 2026-08-26: bench is bit-identical at
    // 2521318 with this set to 0, 46 and 400, while a control
    // (DEXT_MARGIN_PV=400) moves bench to 2112498.
    //
    // It was flagged core, so every core sweep has been spending gradient
    // budget on a parameter that cannot affect the search. Per
    // docs/loose_spsa_knobs_2026-05-16.md that is not merely wasted: SPSA
    // averages the loss change across ALL parameters per iteration, so an inert
    // knob injects noise into the gradient seen by every useful knob. A recent
    // core tune moved it +101.7% (23 -> 46), which is pure noise by
    // construction.
    //
    // Left in place rather than deleted because the SE half of the xray feature
    // has never been tested live -- only the movepicker half has (that is
    // try/discovered-attack-on, H0 at about -5.8 Elo). Decoupling the two so
    // this one can actually fire is a separate, untested experiment.
    (SE_XRAY_BLOCKER_MARGIN_10X, 46, 0, 400, 20.0, false),
    // Continuation-history weight in quiet move ordering. Range runs to 0 so
    // SPSA can disable the term entirely rather than pinning at a floor.
    (CONT_HIST_MULT_10X, 20, 0, 80, 15.0, true),
    // Pawn-history weight in quiet move ordering, relative to main/cont/etc.
    // core: false — not yet validated Elo-positive, so kept out of --core to
    // avoid contributing loose-knob false gradients to the sweep.
    (PAWN_HIST_MULT_10X, 14, 0, 80, 10.0, false),
    (KNIGHT_FORK_BONUS, 8722, 0, 20000, 1000.0, false),
    // LMR endgame gate: skip LMR entirely when popcount(occupied) <= this.
    // Fixes endgame-conversion blunders where LMR over-reduces the
    // king-restriction moves that complete a mate.
    //
    // DELIBERATELY NARROW RANGE: this is correctness-load-bearing on live-play
    // quality (a rook on an open board gets over-reduced as "late"), and SPSA
    // has previously drifted it below the safe band. The floor is set so the
    // effective value cannot fall under 5.
    (LMR_ENDGAME_PIECES_10X, 47, 45, 90, 15.0, true),
    // --- Pruning depth gates ---
    // These are sensitive to eval quality and want re-calibrating after a net
    // change, which is why they are tunable rather than hardcoded.
    //
    // Minimum depth for internal iterative reduction. Floor runs low so SPSA
    // can explore "fire at any depth >= 1" rather than being clamped out of it.
    (IIR_MIN_DEPTH_10X, 46, 5, 100, 15.0, true),
    (PROBCUT_MIN_DEPTH_10X, 15, 10, 120, 15.0, false),     // ProbCut activation gate
    (PROBCUT_ROOT_MIN_DEPTH_10X, 29, 0, 80, 8.0, true),
    (SEE_CAP_DEPTH_10X, 82, 30, 150, 15.0, true),         // SEE capture prune depth cap
    // Capture-SEE prune margin, SF-shaped (search.cpp): margin = depth*MULT +
    // capt_hist*HIST/1024, prune if SEE < -margin. MULT is ~1.1 pawn/depth,
    // toward SF's 0.84; HIST ≈ SF's 34/1024 rescaled for Coda's ±16384
    // capt-hist range.
    //
    // The capt-hist term is load-bearing, not decoration: it protects
    // historically-good captures (the ones that produce cutoffs) so the base
    // can be lowered without over-pruning them. Dropping the base alone,
    // without the history term, cost +17% bench nodes.
    (SEE_CAP_MULT, 101, 40, 250, 12.0, true),
    (SEE_CAP_HIST, 8, 0, 40, 2.0, true),
    (BAD_NOISY_DEPTH_10X, 63, 40, 150, 15.0, true),       // BNFP depth cap
    // NMP activation gate (2 sites). This can sit low because RFP runs FIRST:
    // shallow NMP then only sees nodes static pruning could not already cut,
    // so it no longer intercepts free cutoffs. Reordering NMP ahead of RFP
    // would require pushing this gate back up to compensate.
    (NMP_MIN_DEPTH_10X, 64, 20, 200, 15.0, true),
    (HINDSIGHT_MIN_DEPTH_10X, 21, 0, 200, 15.0, true),
    // Net output scale in percent: the final NNUE eval is multiplied by
    // PCT/100. Nets train to very different natural scales (eval RMS has
    // ranged 219-369 across same-recipe runs) while every cp-denominated
    // search margin is calibrated to the production net's scale. This knob
    // rescales a candidate onto that scale so a net-vs-net SPRT measures the
    // net rather than its units. 100 = off.
    (EVAL_SCALE_PCT, 100, 50, 200, 5.0, false),
    // Piece count at or below which the search runs in threat-REFRESH mode:
    // no per-move delta generation, and the accumulator re-enumerates from the
    // board instead of replaying delta edges.
    //
    // Delta generation is EAGER (every make_move) while consumption is LAZY
    // (only on eval), so every node pays generation but only the evaluating
    // minority consumes it. Measured evals/node is 0.572 in the opening and
    // 0.186 in the endgame, i.e. the endgame discards the work ~5 times out of
    // 6 — which is why `push_threats_for_piece` costs MORE there (5.9% of
    // runtime) than in the opening (4.1%) despite a third the eval rate.
    //
    // Refresh mode deletes that generation and pays a re-enumeration instead.
    // Measured +7.8% nps at <=10 pieces (median +8.6%, positive in 17/20
    // positions, against a same-config control at 8/20) but -21.6% at 17-24
    // and -38.2% at 25-32, so it is worth it only where evals are rare.
    //
    // Keyed on ROOT piece count because the generate/consume contract must be
    // decided before the first make_move, and both sides must agree — replaying
    // from deltas that were never generated would silently corrupt the
    // accumulator. 0 = off.
    (THREAT_REFRESH_PIECE_MAX, 12, 0, 32, 2.0, false),
    // Fail-low prior-countermove cont-hist bonus, % of history_bonus(depth)
    // (SF fail-low history harvesting).
    (FAIL_LOW_PREV_BONUS_PCT, 59, 0, 150, 15.0, false),
    // Cross-MOVE score-trend TM coefficient (×1e-4). Folds the deterioration
    // across MOVES (prev-`go` final score − current running score) into the
    // score-trend multiplier, giving more time when the position has been
    // worsening over the game horizon — the regime where LTC games are lost.
    // Complements the within-search drop term (fixed 0.0025). Default matches
    // that scale (25 → 0.0025). TM change: validate via local cross-engine RR.
    (CROSS_MOVE_TREND, 25, 0, 150, 8.0, false),
    // Converts SEE material (centipawns) into eval units for QS delta pruning:
    // `stand_pat + see_value(victim) * SCALE/100`. It is therefore the
    // material<->eval exchange rate, and a net whose eval sits on a different
    // scale needs this moved with it. Non-core, but it carries loose-knob
    // gradient noise — exclude it from full sweeps if it destabilises
    // neighbours.
    (SEE_MATERIAL_SCALE, 211, 30, 300, 13.5, false),
    // Endgame eval-scaling base: eval *= (MAT_SCALE_BASE + non_pawn_material) / 32768.
    // Lower base = more aggressive damp of the net's output in low-material
    // endgames — the lever against the net over-rating simplified positions.
    // Non-core / experimental.
    (MAT_SCALE_BASE, 22400, 14000, 30000, 1000.0, false),
);

// Demoted loose knobs: cross-tune analysis found SPSA drift dominating signal
// on these, so they were taken off the SPSA surface to improve per-parameter
// SNR for everything else. Values frozen at their pre-demotion defaults.
// Bench-neutral and UCI-invisible. Re-promote only on evidence from a focused
// single-parameter tune, not on a full-sweep mover.
pub static FH_BLEND_OFFSET: AtomicI32 = AtomicI32::new(1);
pub static SE_TT_DEPTH_SLACK: AtomicI32 = AtomicI32::new(3);
pub static MVV_CAP_MULT: AtomicI32 = AtomicI32::new(28);
// Same rationale, second batch. Demotion is reversible: SEE_MATERIAL_SCALE was
// demoted from this list and later put back after a focused single-parameter
// tune found real Elo in it. A knob dismissed as noise under a broad sweep can
// still pay under a targeted one.
pub static QS_SEE_THRESHOLD: AtomicI32 = AtomicI32::new(-26);
pub static CAP_HIST_BASE: AtomicI32 = AtomicI32::new(42);
pub static LMR_COMPLEXITY_DIV: AtomicI32 = AtomicI32::new(152);
pub static TT_CUTOFF_HALFMOVE_MAX: AtomicI32 = AtomicI32::new(89);

/// Post-ponderhit budget credit: PERCENT of elapsed ponder time deducted from
/// the fresh post-hit think budget.
///
/// INERT BY DEFAULT (-1 sentinel = full 100% charge for pondered time, budgets
/// fixed at `go ponder`). Fractional crediting was tried and abandoned: any
/// credit below 100% saturates to the 50ms floor at STC — a ponder of twice the
/// soft budget zeroes it outright — after which realized spend is
/// iteration-quantized bleed up to the hard limit plus grace. That was the
/// dominant cause of a large ponder deficit.
///
/// Full charge is affordable because of its two compensators: the instant
/// reply on a settled ponder hit (`should_instant_reply`) and the ponder-on
/// optimum bump in `compute_tm_budgets`.
///
/// Kept ONLY for local A/B comparability — set 0..=100 via
/// `setoption name PonderhitCreditPct` to re-enable fractional crediting.
///
/// DELIBERATELY NOT a `tunables!` entry: SPSA runs on OB/fastchess, which has
/// NO ponder support (fastchess#513 open), so the ponderhit path never fires
/// there — SPSA would random-walk this on noise and could silently detune it.
pub static PONDERHIT_CREDIT_PCT: AtomicI32 = AtomicI32::new(-1);

/// Effective post-ponderhit credit percent. -1 sentinel (default, "unset")
/// means FULL charge (100). Explicit values are clamped 0..=100.
#[inline(always)]
pub fn ponderhit_credit_pct() -> u64 {
    let v = PONDERHIT_CREDIT_PCT.load(Ordering::Relaxed);
    if v < 0 { 100 } else { v.min(100) as u64 }
}

/// `Ponder` UCI option state. Set by the GUI (cutechess/lichess-bot set it
/// when pondering is enabled). Gates the +25% optimum pre-funding in
/// `compute_tm_budgets` (SF ponder-optimum pre-funding semantics — applied on EVERY
/// move when on; refunded on average by the instant replies of
/// `should_instant_reply`). Default false → bit-identical no-ponder behavior.
pub static PONDER_ENABLED: AtomicBool = AtomicBool::new(false);

#[inline(always)]
pub fn ponder_enabled() -> bool {
    PONDER_ENABLED.load(Ordering::Relaxed)
}

/// Minimum completed root depth of the ponder search before a ponderhit may
/// instant-emit the pondered bestmove. Backstops the elapsed>=soft condition
/// against degenerate cases (tiny soft at low clock, stale/early depth
/// readings): an instant reply must be backed by a real search. NOT a
/// tunable — OB can't ponder, SPSA would detune it on noise.
pub const MIN_PONDER_DEPTH_FOR_INSTANT: i32 = 10;

/// Hard structural floor on the go-ponder→ponderhit window for instant
/// replies. Immune-by-construction guard against DOUBLE-PONDERHIT CASCADES:
/// if the opponent instant-replied out of their own ponderhit, our
/// `go ponder`→`ponderhit` window can be ~1ms — no flag combination may
/// instant-emit then (we would be echoing moves at zero depth back and
/// forth). `elapsed >= intended_soft` already fails in any sane case (soft
/// is tens of ms to seconds), and MIN_PONDER_DEPTH_FOR_INSTANT backstops
/// tiny-soft cases; this floor makes the guarantee unconditional.
pub const MIN_PONDER_ELAPSED_FOR_INSTANT_MS: u64 = 10;

/// Minimum post-ponderhit think slice (ms) when the budget is exhausted but
/// the instant-reply gate did not fire. Shared by the uci.rs ponderhit
/// handler (slice computation) and `should_stop` (mid-iteration soft
/// enforcement floor).
pub const MIN_POST_PONDERHIT_MS: u64 = 50;

/// Hard-frame extension per root fail-low event in the post-ponderhit frame,
/// as a percent of the hard budget — the aspiration fail-low factor
/// (1 + 0.34·min(2, fl), SF shape) applied to the post-hit deadlines.
///
/// Without it, a post-hit cap clips exactly the fail-low re-thinks: SF spends
/// >1s post-hit on ~3.3% of moves where we spent none.
///
/// Consts, DELIBERATELY NOT tunables: OB cannot ponder, so SPSA would detune
/// them on noise; sweep in a local ponder gauntlet only.
pub const PH_FL_HARD_EXT_PCT: u64 = 34;
/// Max fail-low deadline extensions per post-hit search (SF's min(2, fl)).
pub const PH_FL_MAX_EXTENSIONS: u32 = 2;
/// Minimum root depth for a during-post-hit fail-low to trigger a deadline
/// extension. Without this floor, shallow aspiration-window misses (d4-8) —
/// which are routine noise — burn the entire extension budget within
/// milliseconds. Only a fail-low at a real search frontier signals genuine
/// destabilization.
pub const PH_FL_MIN_DEPTH: i32 = 10;

/// stopOnPonderhit-class instant-reply decision (SF stopOnPonderhit
/// pattern, evaluated at the ponderhit instead of during pondering — our
/// clock doesn't tick while pondering, so the budgets the move would have
/// are computable at either point and the handler already has all inputs).
/// Instant-emit the pondered bestmove iff:
///   - the pondered time already covers the soft budget the move would have
///     been given (`elapsed >= intended_soft`), AND
///   - the ponder search completed a real search (depth floor), AND
///   - the root is not currently failing low (SF fail-low pattern: a root
///     fail-low revokes the instant reply — spend extra time exactly when
///     the pondered conclusion destabilized), AND
///   - the elapsed window is not a double-ponderhit cascade artifact (see
///     MIN_PONDER_ELAPSED_FOR_INSTANT_MS).
/// Stability-scaled soft threshold for the instant reply (percent of the
/// intended soft the pondered elapsed must cover, indexed by the ponder
/// search's best-move stability). SAME SHAPE as the dynamic-TM stability
/// table (`TM_STAB_0_100`..`TM_STAB_4_100`) — SF arms stopOnPonderhit
/// against its instability-inflated optimum, so an unstable ponder (stab 0)
/// must have covered 1.71x soft before it may instant-emit, while a settled
/// one (4+) qualifies at 0.75x. Const, not a tunable (OB cannot ponder).
pub const INSTANT_STAB_PCT: [u64; 5] = [171, 120, 90, 80, 75];

#[inline]
pub fn should_instant_reply(
    elapsed_ms: u64,
    intended_soft_ms: u64,
    ponder_completed_depth: i32,
    root_failing_low: bool,
    ponder_stability: u64,
) -> bool {
    let need = intended_soft_ms
        .saturating_mul(INSTANT_STAB_PCT[(ponder_stability as usize).min(4)])
        / 100;
    elapsed_ms >= MIN_PONDER_ELAPSED_FOR_INSTANT_MS
        && elapsed_ms >= need
        && ponder_completed_depth >= MIN_PONDER_DEPTH_FOR_INSTANT
        && !root_failing_low
}

/// Get a tunable parameter value (inline for hot paths)
#[inline(always)]
fn tp(param: &AtomicI32) -> i32 {
    param.load(Ordering::Relaxed)
}

/// Read a `_10X`-scaled tunable, returning the effective integer value
/// (round-half-away-from-zero of stored/10). Tunables with the `_10X`
/// suffix store 10× their effective value so SPSA can express decimal
/// precision and retain decimal progress across tune cycles.
#[inline(always)]
pub fn tp10(param: &AtomicI32) -> i32 {
    let v = param.load(Ordering::Relaxed);
    if v >= 0 { (v + 5) / 10 } else { (v - 5) / 10 }
}

// Feature flags for ablation testing. All true = normal play.
pub static FEAT_NMP: AtomicBool = AtomicBool::new(true);
/// TM diagnostic mode — emit per-move TM state via `info string tm-debug`.
/// Off by default, controlled by UCI option `TMDebug`.
pub static TM_DEBUG: AtomicBool = AtomicBool::new(false);
pub static FEAT_RFP: AtomicBool = AtomicBool::new(true);
pub static FEAT_PROBCUT: AtomicBool = AtomicBool::new(true);
pub static FEAT_LMR: AtomicBool = AtomicBool::new(true);
pub static FEAT_LMP: AtomicBool = AtomicBool::new(true);
pub static FEAT_FUTILITY: AtomicBool = AtomicBool::new(true);
pub static FEAT_SEE_PRUNE: AtomicBool = AtomicBool::new(true); // load-bearing: -17 Elo without
pub static FEAT_BAD_NOISY: AtomicBool = AtomicBool::new(true); // load-bearing: -26 Elo without
pub static FEAT_EXTENSIONS: AtomicBool = AtomicBool::new(true);
pub static FEAT_FH_BLEND: AtomicBool = AtomicBool::new(true); // gates fail-high score blending
pub static FEAT_IIR: AtomicBool = AtomicBool::new(true);
pub static FEAT_HINDSIGHT: AtomicBool = AtomicBool::new(true); // load-bearing: -18 Elo without
pub static FEAT_CORRECTION: AtomicBool = AtomicBool::new(true);
pub static FEAT_PVS: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_CUTOFF: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_NEARMISS: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_STORE: AtomicBool = AtomicBool::new(true);
// Static-eval cache: reuse TT-stored static_eval instead of calling NNUE.
// Ablate (NO_TT_STATIC_EVAL=1) to test whether skipping evals hurts more
// than it helps via deeper lazy-replay gaps (fatter threat/finny applies).
pub static FEAT_TT_STATIC_EVAL: AtomicBool = AtomicBool::new(true);
/// Saturation material tiebreak (issue #18). Deliberately NOT cleared by
/// `disable_all_features()`: it is an eval-correctness term, not a search
/// heuristic, and letting DISABLE_ALL switch it off would fold an eval change
/// into every pruning ablation.
pub static FEAT_SAT_TIEBREAK: AtomicBool = AtomicBool::new(true);
pub static FEAT_QS_CAPTURES: AtomicBool = AtomicBool::new(true); // false = QS returns eval immediately
pub static FEAT_SINGULAR: AtomicBool = AtomicBool::new(true); // singular extensions specifically
pub static FEAT_CUCKOO: AtomicBool = AtomicBool::new(true);
pub static FEAT_4D_HISTORY: AtomicBool = AtomicBool::new(true); // threat-aware 4D history indexing
// Diagnostic-only (env RFP_AUDIT=1): null-verify every RFP cutoff and count
// false positives per depth. NOT a play feature — costs NPS; bench/EPD use.
pub static RFP_AUDIT: AtomicBool = AtomicBool::new(false);

/// Disable all features (pure negamax + eval)
pub fn disable_all_features() {
    FEAT_NMP.store(false, Ordering::Relaxed); FEAT_RFP.store(false, Ordering::Relaxed);
    FEAT_PROBCUT.store(false, Ordering::Relaxed); FEAT_LMR.store(false, Ordering::Relaxed); FEAT_LMP.store(false, Ordering::Relaxed);
    FEAT_FUTILITY.store(false, Ordering::Relaxed); FEAT_SEE_PRUNE.store(false, Ordering::Relaxed);
    FEAT_BAD_NOISY.store(false, Ordering::Relaxed); FEAT_EXTENSIONS.store(false, Ordering::Relaxed); FEAT_FH_BLEND.store(false, Ordering::Relaxed);
    FEAT_IIR.store(false, Ordering::Relaxed); FEAT_HINDSIGHT.store(false, Ordering::Relaxed); FEAT_CORRECTION.store(false, Ordering::Relaxed);
    FEAT_PVS.store(false, Ordering::Relaxed); FEAT_TT_CUTOFF.store(false, Ordering::Relaxed); FEAT_TT_NEARMISS.store(false, Ordering::Relaxed);
    FEAT_TT_STORE.store(false, Ordering::Relaxed); FEAT_TT_STATIC_EVAL.store(false, Ordering::Relaxed);
    FEAT_QS_CAPTURES.store(false, Ordering::Relaxed);
    FEAT_SINGULAR.store(false, Ordering::Relaxed); FEAT_CUCKOO.store(false, Ordering::Relaxed);
    FEAT_4D_HISTORY.store(false, Ordering::Relaxed);
}

/// Enable all features (normal play)
#[allow(dead_code)]
pub fn enable_all_features() {
    FEAT_NMP.store(true, Ordering::Relaxed); FEAT_RFP.store(true, Ordering::Relaxed); FEAT_PROBCUT.store(true, Ordering::Relaxed);
    FEAT_LMR.store(true, Ordering::Relaxed); FEAT_LMP.store(true, Ordering::Relaxed);
    FEAT_FUTILITY.store(true, Ordering::Relaxed); FEAT_SEE_PRUNE.store(true, Ordering::Relaxed);
    FEAT_BAD_NOISY.store(true, Ordering::Relaxed); FEAT_EXTENSIONS.store(true, Ordering::Relaxed); FEAT_FH_BLEND.store(true, Ordering::Relaxed);
    FEAT_IIR.store(true, Ordering::Relaxed); FEAT_HINDSIGHT.store(true, Ordering::Relaxed); FEAT_CORRECTION.store(true, Ordering::Relaxed);
    FEAT_PVS.store(true, Ordering::Relaxed); FEAT_TT_CUTOFF.store(true, Ordering::Relaxed); FEAT_TT_NEARMISS.store(true, Ordering::Relaxed);
    FEAT_TT_STORE.store(true, Ordering::Relaxed); FEAT_TT_STATIC_EVAL.store(true, Ordering::Relaxed);
    FEAT_QS_CAPTURES.store(true, Ordering::Relaxed);
    FEAT_SINGULAR.store(true, Ordering::Relaxed); FEAT_CUCKOO.store(true, Ordering::Relaxed);
    FEAT_4D_HISTORY.store(true, Ordering::Relaxed);
}

// Correction history constants
// --- Saturation material tiebreak (issue #18) ---------------------------
// Ramp start, in SEE units: the measured knee is a 4-pawn imbalance, so the
// term is exactly zero for anything less lopsided than that.
const SAT_MAT_KNEE: i32 = 400;
// Ramp reaches full weight a queen's worth of material past the knee.
const SAT_MAT_FULL: i32 = 900;
// Weight per pawn (100 SEE units) at full ramp. Three times the measured
// -13 cp/pawn inversion, so ordering is monotone with margin while the term
// stays a tiebreak rather than a second material eval.
const SAT_TIEBREAK_W: i32 = 40;

const CORR_HIST_SIZE: usize = 16384;
const CORR_HIST_LIMIT: i32 = 1024;    // Consensus (SF, Obsidian)


/// Search limits.
#[derive(Clone)]
pub struct SearchLimits {
    pub depth: i32,
    /// True when `depth` came from an explicit fixed-depth request. Clocked
    /// searches also carry a high depth ceiling, so `depth` alone cannot tell
    /// SMP whether helper voting is allowed.
    pub fixed_depth: bool,
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
    /// Minimum think time floor in clock-mode (`our_time > 0` branch). Set
    /// after a ponder MISS to prevent instant-emit from a polluted TT
    /// (predicted-line analysis leaked into move ordering) racing the main
    /// thread on positions where the correct move only stabilizes at depth
    /// 10+. Caller must cap this against time-pressure before passing it
    /// in (see uci.rs ponder-miss handling). Zero = no floor.
    pub min_think_ms: u64,
    /// Real remaining clock (ms) for the side to move, for the absolute forfeit
    /// guard on a `movetime` search. Set by the ponderhit fresh-search so the
    /// guard applies there too (start_time is reset to the ponderhit moment).
    /// 0 = no clock concept (plain `go movetime`/depth). See SearchInfo::abs_deadline.
    pub abs_clock: u64,
}

impl Default for SearchLimits {
    fn default() -> Self { Self::new() }
}

impl SearchLimits {
    pub fn new() -> Self {
        SearchLimits {
            depth: 100,
            fixed_depth: false,
            movetime: 0,
            wtime: 0,
            btime: 0,
            winc: 0,
            binc: 0,
            movestogo: 0,
            nodes: 0,
            infinite: false,
            movetime_floor: 0,
            min_think_ms: 0,
            abs_clock: 0,
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
    pub razor_cutoffs: u64,
    pub lmp_prunes: u64,
    pub futility_prunes: u64,
    pub see_prunes: u64,
    pub probcut_cutoffs: u64,
    pub lmr_searches: u64,
    pub singular_ext: u64,
    pub double_ext: u64,
    pub negative_ext: u64,
    pub multicut: u64,
    pub qnodes: u64,
    pub beta_cutoffs: u64,
    pub first_move_cutoffs: u64,
    // fh1 source split: [tt_move, noisy, quiet]
    pub cut_by_source: [u64; 3],
    pub first_cut_by_source: [u64; 3],
    // fh1 conditioned on TT-move presence: [no_tt, has_tt]
    pub cut_by_ttpresence: [u64; 2],
    pub first_cut_by_ttpresence: [u64; 2],
    // RFP-audit FP bucketed by corr-source spread (cp): [<8, 8-24, >=24]
    pub rfp_audit_var_attempts: [u64; 3],
    pub rfp_audit_var_fp: [u64; 3],
    pub cut_quiet_rank1: u64,
    pub cut_quiet_rank_sum: u64,
    // Dual-net dispatch instrumentation: |material-proxy| buckets of 100
    // SEE units, index 11 = 1100+.
    pub dualnet_evals: [u64; 12],
    pub dualnet_abseval: [u64; 12],
    pub dualnet_neareq: [u64; 12],
    // Fail-low node histogram, indexed
    // [depth band 0-2][margin band 0-3][quiet-count band 0-3]:
    // depth {<=4, 5-8, >=9}, margin {<50, 50-150, 150-300, >=300}cp,
    // quiets {0-2, 3-5, 6-9, 10+}. _nodes counts nodes, _quiets sums
    // quiets tried, _late sums max(quiets-2, 0) — the late-quiet tail a
    // margin-aware LMP could target. Measured 2026-07-27 and again on the
    // v10 net 2026-08-17: the tail at margin>=150 (the only safely prunable
    // part) is ~1% of all move-searches, so that idea is closed. Kept
    // because it is the standing measurement of fail-low node shape.
    pub b_probe_nodes: [[u64; 16]; 3],
    pub b_probe_quiets: [[u64; 16]; 3],
    pub b_probe_late: [[u64; 16]; 3],
    pub moves_searched: u64,
    // Move ordering quality: sum of move_count² at beta cutoff (lower = better ordering)
    pub cutoff_movecount_sq_sum: u64,
    pub cutoff_movecount_sum: u64,
    // RFP false-positive audit (diagnostic, env RFP_AUDIT=1). At each RFP
    // cutoff, additionally run an NMP-style null-move verification (same R
    // formula as real NMP) and count cutoffs the null search REJECTS
    // (null_score < beta), bucketed by remaining depth. Answers "is RFP's
    // expanded habitat cutting nodes a dynamic threat check would refuse?"
    // Behavior-preserving: the RFP cutoff is returned regardless.
    pub rfp_audit_attempts: [u64; 24],
    pub rfp_audit_fp: [u64; 24],
    // TREESTATS parity counters, for tree-shape comparison against an
    // instrumented SF build; dumped by the UCI `treestats` command in the same
    // line format that patch emits. Bucket 0 = qsearch;
    // interior nodes bucket by ENTRY depth min(31) — same convention both
    // sides so per-depth lines stay mutually consistent. Reset per `go`
    // (Coda's existing stats convention; harness dumps after each go).
    pub nodes_by_depth: [u64; 32],
    pub cuts_by_depth: [u64; 32],
    pub first_cuts_by_depth: [u64; 32],
    pub width_sum_by_depth: [u64; 32],
    pub width_cnt_by_depth: [u64; 32],
    pub ts_lmr_research: u64,
    pub ts_asp_fail_low: u64,
    pub ts_asp_fail_high: u64,
}

/// Forced-move detection state (set by `detect_forced_move`).
/// Once a position is classified at the root, the result is sticky for the rest of
/// the search — both the verification's TT pollution and the result itself are
/// monotonic. `None` is the default; once `Weak` or `Strong` is observed, the TM
/// multiplier scales down accordingly.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ForcedState {
    None,
    /// Best alternative was within `TM_FORCED_MARGIN_WEAK` of TT score at depth ≥ 12.
    /// Multiplier reduces soft by ~37% (627/1000).
    Weak,
    /// Best alternative collapsed by ≥ `TM_FORCED_MARGIN_STRONG` at depth ≥ 8.
    /// Multiplier reduces soft by ~61% (386/1000).
    Strong,
}

/// Search state for one thread.
/// Stop-time snapshot of the dynamic-TM factor values, captured on the last
/// iteration that evaluated the factor product. TMDebug-gated diagnostics
/// only — never read by search logic.
#[derive(Default, Clone, Copy)]
struct TmDbg {
    stab: f64,
    fail_low: f64,
    forced: f64,
    subtree: f64,
    trend: f64,
    /// Best-move node fraction feeding the subtree factor; -1 when not computed.
    frac: f64,
    /// Post-ceiling factor product applied to soft_limit.
    product: f64,
    adjusted_soft: u64,
}

pub struct SearchInfo {
    pub nodes: u64,
    /// Interior Syzygy WDL probe hits this search, aggregated across SMP
    /// workers after collection. Cosmetic — surfaced in the UCI `tbhits`
    /// field so TB usage is observable. Root TB-move hits are reported
    /// separately at their own info lines.
    pub tb_hits: u64,
    pub global_nodes: std::sync::Arc<AtomicU64>,  // aggregate nodes across SMP threads
    /// Cross-thread best-move-changes, PER THREAD (concept from SF). Thread `i` writes
    /// slot `i` on a root best-move change; main reads+sums+resets per ID
    /// iteration for the instability TM factor. A PER-THREAD array (not one
    /// shared counter) so hundreds of TCEC-scale threads don't contend a single
    /// cache line — writes are rare (≤ once/iteration/thread) so the packed
    /// array's adjacent-slot false-sharing is negligible. Shared Arc like
    /// global_nodes; re-shared each `go`.
    pub thread_bmc: std::sync::Arc<[AtomicU32; 256]>,
    /// Thread count for the current search (1 for the single-thread path). Set
    /// by search_smp; gates the cross-thread instability factor to Threads>1.
    pub num_threads: usize,
    /// Last per-thread node count flushed into global_nodes (delta flushing).
    /// Cell because should_stop takes &self.
    last_flushed_nodes: std::cell::Cell<u64>,
    pub silent: bool,  // suppress UCI output (for datagen)
    /// SMP defers search-end output until helpers have been collected and the
    /// winning thread and aggregate counters are final.
    defer_final_info: bool,
    pub stats: PruneStats,
    // Eval-path decomposition counters.
    // `stats_tt_static_eval_hits` counts nodes where we used the TT's
    // cached static_eval and did NOT call NNUE. The NNUE counters live on
    // `nnue_acc` (full rebuilds vs incremental updates vs computed skips).
    pub stats_tt_static_eval_hits: u64,
    /// True while inside an RFP_AUDIT verification subtree — suppresses
    /// nested audits (each audited cutoff would otherwise spawn audits at
    /// every RFP cutoff inside its own verification, compounding cost).
    pub rfp_audit_active: bool,
    pub tt: std::sync::Arc<TT>,  // shared across Lazy SMP threads
    pub history: Box<History>,
    pub stop: std::sync::Arc<AtomicBool>,  // shared stop flag
    /// EXTERNAL stop flag: set ONLY by the UCI
    /// thread on `stop`/`quit`/abandon (uci.rs shares its external_stop
    /// Arc here). Distinct from `stop`, which the search itself also sets
    /// internally (e.g. to halt helpers during the stockpile-floor sleep)
    /// — so `stop` cannot be used to detect a GUI interrupt once the
    /// search has set it. Polled by the stockpile-floor sleep loop so a
    /// GUI `stop`/`quit` interrupts the wait (on the ponderhit
    /// fresh-search path the floor can be large). Defaults to a fresh
    /// Arc (never set) for non-UCI callers (bench/datagen/epd), which
    /// preserves their behavior exactly.
    pub external_stop: std::sync::Arc<AtomicBool>,
    pub start_time: Instant,
    pub time_limit: u64,  // ms
    pub max_depth: i32,
    pub max_nodes: u64,
    pub move_overhead: u64, // ms
    // Dynamic time management state
    tm_prev_best: Move,
    tm_prev_score: i32,
    tm_best_stable: i32,
    /// Cumulative count of aspiration fail-lows in the current search.
    /// Reset at search start. Consumed by the fail-low TM factor
    /// `1.0 + 0.34 * min(2, asp_fail_low)`, applied to both the opt and hard
    /// windows as a smooth ramp rather than a threshold.
    tm_asp_fail_low: u32,
    /// Cumulative count of aspiration fail-highs in the current search.
    /// Currently diagnostic-only; not used by TM yet (asymmetric vs
    /// fail-low because fail-high means we found a BETTER move than
    /// expected — already captured by score_factor's upward sense).
    tm_asp_fail_high: u32,
    /// Cumulative count of root best-move changes between iterations,
    /// reset at search start. DIAGNOSTIC-ONLY (TMDebug output) — the upward
    /// multiplier it once drove was dropped. A candidate for re-use as an
    /// SF-style within-iteration instability factor.
    tm_best_move_changes: u32,
    /// Forced-move detection state. Set after an ID iteration
    /// at the root when `detect_forced_move` finds that excluding the current best
    /// move collapses the alternative score by a meaningful margin. Sticky once
    /// set — verification only fires while state == None and depth ≥ 8. Drives a
    /// downward multiplier in the TM scale; this is the position-intrinsic signal
    /// Coda was missing (every other signal is search-progress-derived).
    tm_forced_state: ForcedState,
    tm_has_data: bool,
    /// Best score of the PREVIOUS `go` (our-side-to-move, so consecutive
    /// searches are sign-comparable). Persists across `go` commands and is
    /// reset only on ucinewgame; sentinel `i32::MIN` means "no previous move".
    /// Feeds the cross-MOVE score-trend TM term (game-horizon deterioration),
    /// distinct from `tm_prev_score` which is within-search iteration trend.
    pub(crate) tm_cross_prev_score: i32,
    soft_limit: u64,  // ms — can be extended/shortened dynamically
    hard_limit: u64,  // ms — absolute maximum
    /// Minimum think time per move: the increment we're about to gain, minus
    /// move overhead. Floors the dynamically-scaled soft limit so stability
    /// cuts in stable endgames can't push think time below the increment,
    /// which would grow the clock instead of spending it (stockpile). 0 when
    /// there is no increment.
    soft_floor: u64,
    /// True when our increment is exactly zero (3+0, 60+0, 180+0, 40/15).
    ///
    /// Test the increment directly, NOT `soft_floor == 0`: at 10+0.1 the
    /// increment equals the move overhead, so soft_floor lands on 0 by
    /// coincidence and the looser form silently disables the forced-move
    /// detector at short TCs — worth about 3 Elo, for reasons unrelated to
    /// no-increment play.
    tm_no_inc: bool,
    /// Absolute max time we will ever spend on a single move, computed as 60%
    /// of our_clock. A single max-bank ceiling rather than a separate cap on
    /// hard; the TM factors multiply soft up against this.
    tm_max_time: u64,
    /// Our increment (ms) for the current search — feeds the low-increment
    /// multiplier ceiling. 0 when there is no increment.
    tm_our_inc: u64,
    /// Our remaining clock (ms, post-overhead) for the current search — feeds
    /// the inc-relative-to-budget ceiling discriminator.
    tm_time_left: u64,
    /// Per-root-move node counts for node-based time management. The fifth
    /// bucket distinguishes ordinary moves from the four promotion choices.
    /// Reset each search.
    root_move_nodes: Box<[u64; ROOT_MOVE_TABLE_SIZE]>,
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
    /// FL-EXT v2: the INTENDED FULL soft budget (duration ms, from-go-ponder
    /// frame) this move would get on a plain `go`. Stored by the ponderhit
    /// handler with the deadline group; read by the fail-low extension to
    /// inflate the optimum SF-style (soft x (1 + 0.34 x min(2, fl))).
    pub ponderhit_isoft: std::sync::Arc<AtomicU64>,
    /// FL-EXT v3: MAIN thread's "deep root fail-low unresolved in the
    /// post-hit frame" state. While true, should_stop suspends the
    /// mid-iteration soft band (hard + abs still bind) — SF semantics: a
    /// root fail-low revokes the optimum stop entirely; only maximum time
    /// bounds the re-think (the >1s tail source; a soft multiple cannot
    /// reach it: STC intended-soft x1.68 ~ 340ms). Written ONLY by the main
    /// thread (helpers' aspiration state must not clobber it) but SHARED to
    /// helpers so they ride along instead of tripping the shared stop at
    /// the stale band.
    pub ph_fl_active: std::sync::Arc<AtomicBool>,
    /// Time-management baseline: elapsed-ms at which the soft budget starts
    /// counting. 0 for normal `go` (TM starts at search start). Set to the
    /// elapsed-at-ponderhit value when post-ponderhit dynamic TM kicks in,
    /// so soft/floor are interpreted as durations from the ponderhit moment
    /// rather than from the original `go ponder` start.
    pub tm_baseline: u64,
    /// ABSOLUTE forfeit-guard deadline (ms since start_time). Hard ceiling that
    /// makes flagging structurally impossible: the search is force-stopped once
    /// `elapsed >= abs_deadline`, with NO grace and NO ponder exception. Set to
    /// `our_time - overhead - margin` when a real clock is present (0 = disabled,
    /// e.g. movetime/depth/infinite searches). This is independent of the soft/
    /// hard TM budget — it's the last line of defence against search overrun
    /// (iteration overflow, ponder accounting, thread startup, clock lag) that
    /// the fractional `hard` cap can't prevent at low clock. The rule it
    /// enforces is "never forfeit with time on the clock".
    pub abs_deadline: u64,
    /// ABSOLUTE forfeit-guard deadline for the IN-FLIGHT post-ponderhit search
    /// (ms since start_time; 0 = unset). Shared atomic counterpart of
    /// `abs_deadline` for the path where the ponder search keeps running
    /// through a ponderhit (the plain-field guard is only armed by
    /// `start_search`, which that path never re-enters — the loss55
    /// forfeit-class gap). Published by the UCI ponderhit handler as part of
    /// the A1 deadline group (stored Relaxed BEFORE the hard `ponderhit_time`
    /// Release store; readers Acquire hard first) and enforced by
    /// `should_stop` with NO grace.
    pub ponderhit_abs: std::sync::Arc<AtomicU64>,
    /// Root aspiration fail-low state (shared atomic). Set true by the main
    /// ID loop when a root aspiration search fails low, cleared when the
    /// widening re-search resolves inside the window. Read by the UCI thread
    /// at ponderhit to REVOKE the instant reply (SF fail-low
    /// pattern: spend extra time exactly when the pondered conclusion
    /// destabilized). Relaxed ordering is correct: an independent bool gate
    /// with no data-dependency on other shared state — a stale `true` blocks
    /// an instant reply (safe/conservative); a stale `false` is the same
    /// benign race SF's async time check has.
    pub root_fail_low: std::sync::Arc<AtomicBool>,
    /// Completed search depth (shared atomic). Updated by search thread after
    /// each completed iteration. Read by UCI thread on ponderhit to scale budget.
    pub ponder_depth: std::sync::Arc<AtomicU64>,
    /// FL-EXT: count of fail-low deadline extensions granted in THIS post-hit
    /// search (main thread only writes; capped at PH_FL_MAX_EXTENSIONS).
    /// Plain field — never shared; helpers' copies stay 0 (silent-gated).
    pub ph_fl_extensions: u32,
    /// Best-move stability of the (ponder) search (shared atomic; mirrors
    /// tm_best_stable, which is tracked on EVERY iteration including pure
    /// ponder). Read by the UCI thread at ponderhit: the instant-reply gate
    /// scales its elapsed-vs-soft threshold by the SAME stability table the
    /// dynamic TM uses (SF arms stopOnPonderhit against its instability-
    /// inflated optimum — an unstable-but-deep ponder must NOT instant-emit).
    pub ponder_stability: std::sync::Arc<AtomicU64>,
    pub sel_depth: i32,
    pub last_score: i32,
    /// Root side-to-move (was used for contempt; retained for potential future use)
    pub root_stm: u8,
    /// Per-depth cumulative node counts (for EBF calculation in bench)
    pub depth_nodes: [u64; MAX_PLY + 1],
    pub completed_depth: i32,
    /// Current ID iteration's target (root) depth, set at the top of each
    /// iteration in both ID loops. Visible at every node so depth-dependent
    /// formulas adjust by the OVERALL search depth (= the time control's
    /// reach), giving a single tunable set that self-adapts STC<->LTC instead
    /// of two constant sets.
    pub root_depth: i32,
    /// TMDebug-only stop-time snapshot of the dynamic-TM factors (see TmDbg).
    tm_dbg: TmDbg,
    /// Line-trace forensics (CODA_TRACE_LINE env): zobrist hashes of the
    /// positions along a target line from the root, and the line's moves.
    /// Empty (= disabled, zero hot-path cost beyond one is_empty check) unless
    /// the env var is set. When a search node's hash matches trace_hashes[ply],
    /// hooks log which pruning gate discards trace_line_mv[ply], to stderr.
    pub trace_hashes: Vec<u64>,
    pub trace_line_mv: Vec<Move>,
    /// Ply barrier for NMP verification: prevents NMP from re-triggering
    /// inside its own verification subtree (all peers: Alexandria,
    /// Stormphrax use nmpMinPly / nmp_min_ply). Default 0 = no barrier.
    pub nmp_min_ply: i32,
    /// Triangular PV table
    pub pv_table: [[Move; MAX_PLY + 1]; MAX_PLY + 1],
    pub pv_len: [usize; MAX_PLY + 1],
    /// MultiPV: number of top root lines to report (analysis only). Default 1.
    pub multipv: usize,
    /// Root moves banned from the current search (MultiPV secondary lines).
    /// Empty except during a MultiPV>1 secondary search, so single-PV play is
    /// byte-identical.
    pub root_ban: Vec<Move>,
    static_evals: [i32; MAX_PLY + 1],
    /// LMR reduction applied at each ply (for hindsight reduction gating)
    reductions: [i32; MAX_PLY + 1],
    /// Excluded move for singular extension verification search (always NoMove when disabled)
    pub excluded_move: [Move; MAX_PLY + 1],
    /// Double extension counter — propagated from parent, capped to prevent search explosion
    double_ext_count: [i32; MAX_PLY + 1],
    /// Per-ply beta-cutoff counter (SF cutoffCnt). Incremented at the fail-high
    /// site; each node clears its GRANDCHILD slot on entry so
    /// `cutoff_count[ply+1]` reflects only fail-highs under this node's own
    /// subtree. Read in LMR: a child ply that keeps failing high means
    /// refutations come easy there — reduce late moves more. +4 padding
    /// allows unconditional ply+2 indexing.
    cutoff_count: [i32; MAX_PLY + 4],
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
    /// Paired continuation correction: [prev_piece][prev_to][cur_piece][cur_to],
    /// go_piece 1-12 (slot 0 unused). Read/updated at ply-2 and ply-4 offsets.
    cont_corr: Box<[[[[i32; 64]; 13]; 64]; 13]>,
    /// Transition correction history: [stm][(hash(ply-1) ^ hash(ply)) % size]
    trans_corr: Box<[[i32; CORR_HIST_SIZE]; 2]>,
    pub nnue_net: Option<std::sync::Arc<crate::nnue::NNUENet>>,
    pub nnue_acc: Option<crate::nnue::NNUEAccumulator>,
    pub threat_stack: crate::threat_accum::ThreatStack,
    /// Syzygy tablebases (shared, read-only). Interior WDL probes in search.
    pub syzygy: Option<std::sync::Arc<crate::tb::SyzygyTB>>,
    /// Min depth at which to probe Syzygy WDL when at the maximum loaded
    /// piece count (SF `SyzygyProbeDepth`). Below the max piece count we
    /// always probe regardless of depth. Default 4: our deploy set is
    /// 5-man-everything, so max-men interior probes at depth<4 are frequent
    /// and largely redundant (re-probed deeper up the tree). Local RR
    /// (5-man, STC, no-adjudication) measured =4 vs =1 at +2.0 Elo, LOS
    /// 92.2%, N=10000. UCI option `SyzygyProbeDepth`.
    pub tb_probe_depth: i32,
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
            tb_hits: 0,
            global_nodes: std::sync::Arc::new(AtomicU64::new(0)),
            thread_bmc: std::sync::Arc::new(std::array::from_fn(|_| AtomicU32::new(0))),
            num_threads: 1,
            last_flushed_nodes: std::cell::Cell::new(0),
            silent: false,
            defer_final_info: false,
            stats: PruneStats::default(),
            stats_tt_static_eval_hits: 0,
            tt,
            history: alloc_zeroed_box(),
            stop: std::sync::Arc::new(AtomicBool::new(false)),
            external_stop: std::sync::Arc::new(AtomicBool::new(false)),
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
            tm_asp_fail_low: 0,
            tm_asp_fail_high: 0,
            tm_has_data: false,
            tm_cross_prev_score: i32::MIN,
            soft_limit: 0,
            hard_limit: 0,
            soft_floor: 0,
            tm_no_inc: false,
            tm_max_time: 0,
            tm_our_inc: 0,
            tm_time_left: 0,
            root_move_nodes: alloc_zeroed_box(),
            ponderhit_time: std::sync::Arc::new(AtomicU64::new(0)),
            ponderhit_soft: std::sync::Arc::new(AtomicU64::new(0)),
            ponderhit_floor: std::sync::Arc::new(AtomicU64::new(0)),
            ponderhit_isoft: std::sync::Arc::new(AtomicU64::new(0)),
            ph_fl_active: std::sync::Arc::new(AtomicBool::new(false)),
            tm_baseline: 0,
            abs_deadline: 0,
            ponderhit_abs: std::sync::Arc::new(AtomicU64::new(0)),
            root_fail_low: std::sync::Arc::new(AtomicBool::new(false)),
            ponder_depth: std::sync::Arc::new(AtomicU64::new(0)),
            ph_fl_extensions: 0,
            ponder_stability: std::sync::Arc::new(AtomicU64::new(0)),
            sel_depth: 0,
            last_score: 0,
            root_stm: WHITE,
            depth_nodes: [0; MAX_PLY + 1],
            completed_depth: 0,
            root_depth: 0,
            tm_dbg: TmDbg::default(),
            trace_hashes: Vec::new(),
            trace_line_mv: Vec::new(),
            nmp_min_ply: 0,
            static_evals: [0; MAX_PLY + 1],
            reductions: [0; MAX_PLY + 1],
            excluded_move: [NO_MOVE; MAX_PLY + 1],
            double_ext_count: [0; MAX_PLY + 1],
            cutoff_count: [0; MAX_PLY + 4],
            moved_piece_stack: [0; MAX_PLY + 1],
            moved_to_stack: [0; MAX_PLY + 1],
            pv_table: [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1],
            pv_len: [0; MAX_PLY + 1],
            multipv: 1,
            root_ban: Vec::new(),
            pawn_hist: alloc_zeroed_box(),
            pawn_corr: alloc_zeroed_box(),
            np_corr: alloc_zeroed_box(),
            cont_corr: alloc_zeroed_box(),
            trans_corr: alloc_zeroed_box(),
            nnue_net: None,
            nnue_acc: None,
            threat_stack: crate::threat_accum::ThreatStack::new(768), // max v9 accum size
            syzygy: None,
            tb_probe_depth: 4,
            rfp_audit_active: false,
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
        // Reset unconditionally first, then activate only if the new net needs
        // it. Touching threat_stack only inside the `has_threats` branch leaks
        // state across a net swap: going from a threats net to one without,
        // the stack stays `active=true` and search runs threat computation
        // against a net that never consumes it.
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
        for path in net_nnue_paths.iter().flatten() {
            if path.exists() {
                if let Ok(()) = self.load_nnue(path.to_str().unwrap()) {
                    return true;
                }
            }
        }

        // 3. net.txt discovery (extract filename from URL)
        let net_txt_paths = [
            std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("net.txt"))),
            Some(std::path::PathBuf::from("net.txt")),
        ];
        for path in net_txt_paths.iter().flatten() {
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

        false
    }

    fn should_stop(&self) -> bool {
        if self.stop.load(Ordering::Relaxed) {
            return true;
        }
        // Flush local node count to global counter every 4096 nodes. Track the
        // DELTA rather than adding a flat 4096: should_stop can be re-invoked
        // from ID-loop sites while the count rests on a boundary, and a flat
        // add double-counts there.
        if self.nodes & 4095 == 0 && self.nodes > self.last_flushed_nodes.get() {
            self.global_nodes.fetch_add(self.nodes - self.last_flushed_nodes.get(), Ordering::Relaxed);
            self.last_flushed_nodes.set(self.nodes);
        }
        // SMP-correct node-limit gate: helpers don't get max_nodes propagated
        // (helper init at clone_for_helper leaves max_nodes=0), and main's
        // per-thread `self.nodes` excludes helper contributions. Both effects
        // made `go nodes N` overshoot by ~N*T at T threads. Check global
        // counter so all threads stop together; up to T*4096 unflushed nodes
        // of slack is acceptable.
        // Include this thread's UNFLUSHED local delta, not just the
        // 4096-granular global counter. Checking the global alone lets
        // `go nodes N` overshoot by up to 4095 nodes — +23% at N=10000, which
        // is enough to inflate fixed-node results against an engine that
        // enforces the limit exactly. Helpers still contribute at flush
        // granularity (the bounded slack above); the main thread, which is the
        // one that matters for T=1 fixed-node testing, is node-exact. Zero
        // cost unless max_nodes > 0.
        if self.max_nodes > 0
            && self.global_nodes.load(Ordering::Relaxed)
                + (self.nodes - self.last_flushed_nodes.get())
                >= self.max_nodes
        {
            // Set the shared stop flag (like the time branches below) so
            // ancestors abort their epilogues instead of storing partial
            // best_score/best_move to the TT and firing fake history bonuses,
            // and so helper threads see the limit too. Without this, `go nodes`
            // returned 0 up the tree while nodes completed their TT/history
            // stores at full claimed depth (persisting across the game).
            self.stop.store(true, Ordering::Relaxed);
            return true;
        }
        // Check time every 4096 nodes
        if self.nodes & 4095 == 0 {
            let elapsed = self.start_time.elapsed().as_millis() as u64;
            // ABSOLUTE forfeit guard — checked FIRST, with NO grace and NO
            // ponder exception. Makes flagging impossible regardless of what the
            // soft/hard budget, ponder accounting, or iteration overflow do.
            if self.abs_deadline > 0 && elapsed >= self.abs_deadline {
                self.stop.store(true, Ordering::Relaxed);
                return true;
            }
            // For ponderhit: allow a grace period beyond the deadline so the
            // current iteration can finish cleanly. But hard-stop if the grace
            // period expires to prevent time loss. The ID loop also checks the
            // deadline (without grace) between iterations to prevent starting
            // new iterations after the budget expires.
            // A1 publish protocol: ponderhit_time (hard) is the publish
            // flag for the deadline trio — always loaded with Acquire.
            // (Only hard is consumed here, but uniform Acquire keeps every
            // reader on the protocol.)
            let ph_time = self.ponderhit_time.load(Ordering::Acquire);
            let effective_limit = if ph_time > 0 {
                // P2(b) — ABSOLUTE forfeit guard for the in-flight
                // post-ponderhit search (loss55 class: only the wait-loop
                // fresh-search path used to arm abs_deadline; the in-flight
                // path had no guarantee at all). NO grace. Published with
                // the deadline group (Relaxed store before the hard Release;
                // hard observed non-zero above ⇒ this value is coherent).
                let ph_abs = self.ponderhit_abs.load(Ordering::Relaxed);
                if ph_abs > 0 && elapsed >= ph_abs {
                    self.stop.store(true, Ordering::Relaxed);
                    return true;
                }
                // P2(a) — mid-iteration SOFT enforcement, 2×-band only. The
                // normal band (up to one post-hit slice past the soft
                // deadline) is deliberately NOT soft-stopped mid-iteration —
                // finishing the in-flight iteration has real value. But past
                // 2× the slice, cut the iteration: the pre-fix behavior
                // (hard + 500ms grace as the only mid-iteration bound) let a
                // single deep iteration bleed up to ~4s at 10+0.1 — 40% of the
                // base clock on one move.
                // ponderhit_floor carries the post-hit slice length, so
                // `soft + slice` = "elapsed exceeds soft by 2×" in the
                // post-hit frame regardless of how long the ponder ran.
                let ph_soft = self.ponderhit_soft.load(Ordering::Relaxed);
                if ph_soft > 0 && !self.ph_fl_active.load(Ordering::Relaxed) {
                    // FL-EXT v3: while the MAIN thread's root fail-low is
                    // unresolved the soft band is suspended (SF: fail-low
                    // revokes the optimum stop; only maximum time binds).
                    // hard + grace below and the abs forfeit wall above
                    // still bound the re-think — that IS the >1s tail.
                    let slice = self.ponderhit_floor.load(Ordering::Relaxed)
                        .max(MIN_POST_PONDERHIT_MS);
                    if elapsed >= ph_soft.saturating_add(slice) {
                        self.stop.store(true, Ordering::Relaxed);
                        return true;
                    }
                }
                // Hard-deadline grace: enough to finish an iteration close to
                // the wire but not enough to bleed. Was min(remaining/4,
                // 500ms) — at 10+0.1 that grace alone is ~5 base-time
                // increments; shrunk to min(remaining/8, 100ms) (P2).
                let remaining = ph_time.saturating_sub(elapsed);
                let grace = (remaining / 8).min(100);
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
        for a in self.cont_corr.iter_mut() { for b in a.iter_mut() { for c in b.iter_mut() { c.fill(0); } } }
        for row in self.trans_corr.iter_mut() { row.fill(0); }
    }

    pub fn clear_pawn_hist(&mut self) {
        for entry in self.pawn_hist.iter_mut() {
            *entry = [[0i16; 64]; 13];
        }
    }

    pub fn clear_persistent_histories(&mut self) {
        self.history.clear();
        self.clear_pawn_hist();
        self.clear_correction_history();
    }

    #[cfg(test)]
    pub fn dirty_persistent_histories_for_test(&mut self) {
        self.history.main[1][0][2][3] = 123;
        self.history.capture[1][4][2] = -45;
        self.history.cont_hist[1][4][2][5] = 67;
        self.pawn_hist[3][1][7] = 89;
        self.pawn_corr[WHITE as usize][5] = 101;
        self.np_corr[BLACK as usize][WHITE as usize][6] = -202;
        self.cont_corr[1][8][2][9] = 303;
        self.trans_corr[BLACK as usize][10] = -404;
    }

    #[cfg(test)]
    pub fn assert_persistent_histories_clear_for_test(&self) {
        for a in self.history.main.iter() {
            for b in a.iter() {
                for c in b.iter() {
                    for &v in c.iter() {
                        assert_eq!(v, 0, "main history was not cleared");
                    }
                }
            }
        }
        for a in self.history.capture.iter() {
            for b in a.iter() {
                for &v in b.iter() {
                    assert_eq!(v, 0, "capture history was not cleared");
                }
            }
        }
        for a in self.history.cont_hist.iter() {
            for b in a.iter() {
                for c in b.iter() {
                    for &v in c.iter() {
                        assert_eq!(v, 0, "continuation history was not cleared");
                    }
                }
            }
        }
        for a in self.pawn_hist.iter() {
            for b in a.iter() {
                for &v in b.iter() {
                    assert_eq!(v, 0, "pawn history was not cleared");
                }
            }
        }
        for a in self.pawn_corr.iter() {
            for &v in a.iter() {
                assert_eq!(v, 0, "pawn correction history was not cleared");
            }
        }
        for a in self.np_corr.iter() {
            for b in a.iter() {
                for &v in b.iter() {
                    assert_eq!(v, 0, "non-pawn correction history was not cleared");
                }
            }
        }
        for a in self.cont_corr.iter() {
            for b in a.iter() {
                for c in b.iter() {
                    for &v in c.iter() {
                        assert_eq!(v, 0, "continuation correction history was not cleared");
                    }
                }
            }
        }
        for a in self.trans_corr.iter() {
            for &v in a.iter() {
                assert_eq!(v, 0, "transition correction history was not cleared");
            }
        }
    }

    /// Evaluate using NNUE. A net is required to run — shipped/bench builds
    /// always embed one, so the no-net branch below is never hit in practice.
    /// The eval as SEARCH sees it: net output, material scaling, and the
    /// saturation tiebreak. `pub(crate)` so the UCI `eval` command reports
    /// this same number instead of re-deriving a raw net output that drifts
    /// every time a term is added here.
    pub(crate) fn eval(&mut self, board: &Board) -> i32 {
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

                // ---------------------------------------------------------------
                // THREAT-ACCUMULATOR VERIFICATION (diagnostic).
                // The PSQ force_recompute above is blind to threat drift: it
                // re-evaluates with the SAME (possibly desynced) threat_stack.
                // Here we additionally recompute the threat features from
                // scratch (full enumeration, same procedure as
                // ThreatStack::refresh) into LOCAL buffers and compare against
                // the live incremental accumulator for both perspectives.
                // Strictly read-only w.r.t. the stack — nothing the search
                // uses is perturbed. Opt-in via CODA_VERIFY_NNUE (this whole
                // block); CODA_VERIFY_THREATS=panic upgrades mismatch to panic.
                // ---------------------------------------------------------------
                if self.threat_stack.active && net.has_threats {
                    static THREAT_PANIC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                    static THREAT_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    static THREAT_MISMATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let do_panic = *THREAT_PANIC.get_or_init(|| {
                        std::env::var("CODA_VERIFY_THREATS").as_deref() == Ok("panic")
                    });
                    let tn = THREAT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let h = net.hidden_size;
                    let occ = board.colors[0] | board.colors[1];
                    for pov in [WHITE, BLACK] {
                        let ksq = (board.pieces[KING as usize] & board.colors[pov as usize])
                            .trailing_zeros();
                        let mirrored = (ksq % 8) >= 4;
                        // Collect scratch feature indices with the SAME bound as
                        // refresh (MAX_ACTIVE_THREAT_FEATURES) so the verifier can
                        // never be blind to a refresh truncation — a prior shared
                        // 256 cap made both sides truncate identically and agree.
                        let mut indices =
                            [0usize; crate::threat_accum::MAX_ACTIVE_THREAT_FEATURES];
                        let mut ni = 0usize;
                        let mut overflow = false;
                        crate::threats::enumerate_threats(
                            &board.pieces, &board.colors, &board.mailbox,
                            occ, pov, mirrored,
                            |idx| {
                                if idx < net.num_threat_features {
                                    if ni < crate::threat_accum::MAX_ACTIVE_THREAT_FEATURES {
                                        indices[ni] = idx; ni += 1;
                                    } else { overflow = true; }
                                }
                            },
                        );
                        let mut check = [0i16; crate::threat_accum::MAX_FT_SIZE];
                        crate::threats::add_weight_rows(
                            &mut check[..h], &net.threat_weights, h, &indices[..ni]);
                        let live = self.threat_stack.values(pov);
                        if &check[..h] != live {
                            let m = THREAT_MISMATCHES
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if m < 40 || do_panic {
                                let mut nch = 0usize;
                                let mut l1: i64 = 0;
                                let mut firsts = String::new();
                                for j in 0..h {
                                    let d = live[j] as i64 - check[j] as i64;
                                    if d != 0 {
                                        if nch < 8 {
                                            firsts.push_str(&format!(
                                                " ch{}:{}(live)vs{}(scratch)", j, live[j], check[j]));
                                        }
                                        nch += 1;
                                        l1 += d.abs();
                                    }
                                }
                                eprintln!(
                                    "THREAT MISMATCH n={} pov={} ply={} fen=\"{}\" hash={:016x} \
                                     diff_channels={} l1={} scratch_feats={}{} overflow={}{}",
                                    tn,
                                    if pov == WHITE { "W" } else { "B" },
                                    self.threat_stack.index(),
                                    board.to_fen(), board.hash,
                                    nch, l1, ni, firsts, overflow,
                                    if do_panic { " [panic mode]" } else { "" },
                                );
                            }
                            if do_panic {
                                panic!("THREAT MISMATCH (CODA_VERIFY_THREATS=panic): fen={}",
                                    board.to_fen());
                            }
                        }
                    }
                    // Periodic summary so long runs report even without mismatches.
                    if tn == 9_999 || (tn > 0 && tn % 1_000_000 == 0) {
                        let mm = THREAT_MISMATCHES.load(std::sync::atomic::Ordering::Relaxed);
                        eprintln!("THREAT verify: {}/{} mismatches after {} verified evals",
                            mm, tn + 1, tn + 1);
                    }
                }
                s2 // use recomputed value when verifying
            } else {
                s
            }
        } else {
            panic!("no NNUE net loaded — an NNUE net is required to evaluate; \
                    build with `make` to embed a net or pass `--nnue <file>`");
        };
        // Material scaling: dampen eval in low-material endgames (SF/Stormphrax/
        // Halogen pattern — non-pawn material only). Pawn-up endgames
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
        // formula) that bakes in gross errors and wipes out the correction —
        // worth about -8 Elo. The fix is structural: keep TT storage
        // halfmove-independent, and apply the scale freshly on read.
        let mut final_score = score * (tp(&MAT_SCALE_BASE) + material) / 32 / 1024;

        // Signed material balance in SEE units, white-positive. Shared by the
        // saturation tiebreak below and the dispatch instrumentation.
        let signed_material = {
            let w = board.colors[WHITE as usize];
            let b = board.colors[BLACK as usize];
            let mut m = 0i32;
            for pt in 0..5u8 {
                let d = (board.pieces[pt as usize] & w).count_ones() as i32
                    - (board.pieces[pt as usize] & b).count_ones() as i32;
                m += crate::eval::see_value(pt) * d;
            }
            m
        };

        // Saturation material tiebreak (issue #18).
        //
        // Measured on this tree with net-1EB961AF by stripping material from
        // the start position one piece at a time and reading `go depth 14`
        // (ScoreScale 100, i.e. internal units):
        //
        //   deficit    0    -1    -2    -3    -4    -5   -12   -22   -31
        //   eval      +49   -67  -460  -810 -1079 -1079 -1247 -1120 -1017
        //
        // The net tracks material closely out to about 4 pawns (|eval| ~1080)
        // and then stops. Past that knee the slope collapses from ~250 cp per
        // pawn to about -13 -- i.e. slightly INVERTED: shedding material
        // *raises* the eval. The mirror ladder stripping White is the same
        // shape, so this is an output-range ceiling of the net, not a
        // losing-side pathology.
        //
        // Consequence: inside that band the search sees no cost to giving
        // pieces away, and does. Issue #18 reports exactly this -- the eval
        // pinned at the plateau while a queen, two knights, a bishop and two
        // rooks came off the board.
        //
        // So restore a monotone material ordering past the knee. The ramp is
        // zero at the knee and reaches full weight at the measured floor, so
        // the calibrated band below is untouched and there is no step at the
        // boundary for the search to oscillate across.
        //
        // NET-DEPENDENT: the knee is a property of the net's output range.
        // Re-measure these three constants whenever the production net
        // changes; the ladder probe lives in the research repo.
        // The ramp keys off MATERIAL, not off |eval|. Keying it off the eval
        // is the obvious first cut and it is wrong: as material is shed the
        // saturated eval drifts back *toward* zero, which shrinks the ramp and
        // fades the correction out exactly where it is needed. Measured on the
        // first attempt -- the -12 -> -31 band stayed inverted. Material is
        // monotone by construction, so there is no feedback loop.
        if FEAT_SAT_TIEBREAK.load(Ordering::Relaxed) {
            // stm-relative, to match the point of view of `final_score`
            let mat = if board.side_to_move == WHITE {
                signed_material
            } else {
                -signed_material
            };
            let excess = mat.abs() - SAT_MAT_KNEE;
            if excess > 0 {
                let span = SAT_MAT_FULL - SAT_MAT_KNEE;
                let ramp = excess.min(span);
                final_score += mat * SAT_TIEBREAK_W * ramp / (100 * span);
            }
        }

        // Dual-net dispatch instrumentation (both paths still call the big
        // net). Proxy = SIGNED piece-material balance in SEE
        // units — the candidate dispatch signal: position-intrinsic,
        // changes only on captures/promotions. Per |proxy| bucket we count
        // evals, sum |internal eval|, and count near-equal evals
        // (|eval| < 100 internal) — giving, from one bench run, the
        // small-net qualification rate at ANY threshold plus the
        // false-positive rate the re-eval guard would face there.
        {
            let bucket = ((signed_material.abs() / 100) as usize).min(11);
            self.stats.dualnet_evals[bucket] += 1;
            self.stats.dualnet_abseval[bucket] += final_score.unsigned_abs() as u64;
            if final_score.abs() < 100 {
                self.stats.dualnet_neareq[bucket] += 1;
            }
        }

        final_score
    }

    #[inline]
    fn materialize_tt_barrier(&mut self, board: &Board) {
        if let (Some(net), Some(acc)) = (&self.nnue_net, &mut self.nnue_acc) {
            if acc.has_unmaterialized_psq_barrier() {
                acc.materialize(net, board);
            }
        }
    }
}

/// Scale a raw (halfmove-independent) eval toward zero as the halfmove
/// clock approaches the 100-ply draw horizon.
///
/// Formula: `score * (200 - clamp(hm, 0, 100)) / 200`. At `hm=0` returns
/// `score` unchanged; at `hm=100` returns `0.5×`. Callers apply this at the
/// *point of use*, never before storing to TT — see the comment in
/// `SearchInfo::eval`.
///
/// Consensus form, shared by the entire reference set — SF `v - v*rule50/199`,
/// Obsidian/Berserk `(200 - hm)/200`, PlentyChess `(293 - rule50)/293`
/// — which all HALVE (not zero) the eval at the 50-move cliff. The previous
/// `(100 - hm)/100` is a 2x outlier that nulls a won eval to 0.00 at the
/// cliff, and that over-damping has been traced to real won-position draws.
#[inline]
fn apply_halfmove_scale(score: i32, halfmove: u16) -> i32 {
    // Leave sentinel scores untouched so downstream comparisons with
    // `-INFINITY` / `MATE_SCORE - ply` keep their absolute magnitudes.
    if score <= -INFINITY + 1 || score.abs() >= MATE_IN_MAX_PLY {
        return score;
    }
    let hm = (halfmove as i32).min(100);
    score * (200 - hm) / 200
}

/// TT-cutoff child-consistency verification (technique from SF, independently
/// re-implemented). Before trusting a DEEP
/// (depth >= 7) TT cutoff, make the TT move (board-only — no NNUE work),
/// probe the child's TT entry, and unmake. Returns true (decline the cutoff,
/// search the node normally) when the child entry exists and its negated
/// value contradicts the cutoff direction. Deep trees are TT-cutoff-dominated;
/// this rejects stale/one-sided deep cutoffs for the cost of one make/unmake
/// + one probe, on deep cutoffs only.
fn tt_cutoff_child_disagrees(
    info: &SearchInfo,
    board: &mut Board,
    tt_move: Move,
    tt_score: i32,
    beta: i32,
    depth: i32,
    ply: i32,
) -> bool {
    if depth < 7 || tt_move == NO_MOVE || is_decisive(tt_score) {
        return false;
    }
    // Full legality validation before speculatively making the move (stale TT
    // entries / hash collisions can carry garbage moves).
    if !crate::movepicker::is_pseudo_legal(board, tt_move)
        || !board.is_legal(tt_move, board.pinned(), board.checkers())
    {
        return false;
    }
    if !board.make_move(tt_move) {
        return false;
    }
    let child = info.tt.probe(board.hash);
    let child_halfmove = board.halfmove; // 50mr clock of the CHILD position
    board.unmake_move();
    if !child.hit {
        return false; // no child evidence -> trust the cutoff (SF behaviour)
    }
    let child_score = score_from_tt(child.score, ply + 1, child_halfmove);
    if is_decisive(child_score) {
        return false; // skip mate-distance comparisons across plies
    }
    (tt_score >= beta) != (-child_score >= beta)
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

    if moved_pt == KING {
        let mut from_ks = from as usize;
        let mut to_ks = to as usize;
        if us == BLACK { from_ks ^= 56; to_ks ^= 56; }

        let from_bucket = net.king_bucket(from_ks);
        let to_bucket = net.king_bucket(to_ks);
        let from_mirror = net.king_mirror(from_ks);
        let to_mirror = net.king_mirror(to_ks);

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

        if from_bucket != to_bucket || from_mirror != to_mirror {
            return DirtyPiece::refresh_perspective(us, &changes[..n]);
        }
        return DirtyPiece::incremental(&changes[..n]);
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

/// Paired continuation correction. Index by the LAST move
/// (ply-1) and select the subtable by the move at ply-2 AND ply-4, summing both
/// — the SF 2-D continuation form, replacing Coda's flat
/// 1-ply `[piece][to]` (the sole 6/6 flat-1-ply outlier). Uses
/// `moved_piece_stack`/`moved_to_stack` (go_piece 1-12) so the ply-2/ply-4
/// pieces are read correctly — `board.piece_at` on an old destination would be
/// disturbed by later moves. `ply` must be <= MAX_PLY (callers clamp).
#[inline]
fn cont_corr_value(info: &SearchInfo, ply: usize) -> i64 {
    if ply < 2 { return 0; }
    let cur_p = info.moved_piece_stack[ply - 1] as usize;
    let cur_t = info.moved_to_stack[ply - 1] as usize;
    if cur_p == 0 || cur_p >= 13 || cur_t >= 64 { return 0; }
    let mut sum = 0i64;
    for off in [2usize, 4] {
        if ply >= off {
            let pp = info.moved_piece_stack[ply - off] as usize;
            let pt = info.moved_to_stack[ply - off] as usize;
            if pp != 0 && pp < crate::movepicker::CONT_PLANES && pt < 64 {
                sum += info.cont_corr[pp][pt][cur_p][cur_t] as i64;
            }
        }
    }
    sum
}

/// Compute the correction value alone (the centipawn delta corrhist would apply
/// to raw eval). Used by SE-margin formulas to gate extension confidence on
/// |correction| — extend less on uncertain (drifting) evals.
fn correction_value(info: &SearchInfo, board: &Board, ply: usize) -> i32 {
    let stm = board.side_to_move as usize;
    let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
    let pawn_corr = info.pawn_corr[stm][pawn_idx] as i64;
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
    let white_np_corr = info.np_corr[stm][WHITE as usize][white_np_idx] as i64;
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
    let black_np_corr = info.np_corr[stm][BLACK as usize][black_np_idx] as i64;
    let cont_corr = cont_corr_value(info, ply);
    let trans_corr = if !board.undo_stack.is_empty() {
        let last = &board.undo_stack[board.undo_stack.len() - 1];
        if last.mv != NO_MOVE {
            let trans_idx = ((board.hash ^ last.hash) as usize) & (CORR_HIST_SIZE - 1);
            info.trans_corr[stm][trans_idx] as i64
        } else { 0 }
    } else { 0 };
    let total_corr = (pawn_corr * tp(&CORR_W_PAWN) as i64 + white_np_corr * tp(&CORR_W_NP) as i64 + black_np_corr * tp(&CORR_W_NP) as i64
        + cont_corr * tp(&CORR_W_CONT) as i64 + trans_corr * tp(&CORR_W_TRANS) as i64) / tp(&CORR_HIST_DIV) as i64;
    (total_corr as i32) / tp(&CORR_HIST_GRAIN_T)
}

/// Apply correction history to raw static eval.
///
/// Thin wrapper over `correction_value` so the five-source blend exists in ONE
/// place. It previously duplicated that whole computation, which is the shape
/// that hides a bug: add or reweight a source and one copy silently keeps the
/// old blend.
#[inline]
fn corrected_eval(info: &SearchInfo, board: &Board, raw_eval: i32, ply: usize) -> i32 {
    // There is deliberately no material damping here: the residual update
    // baseline makes corrhist converge to the true (~0) correction in
    // low-signal positions, so a piece-count fortress guard is redundant.
    let adjusted = raw_eval + correction_value(info, board, ply);
    // Keep the corrected static eval strictly inside the non-mate band so it
    // can never be read back as a mate by the MATE_IN_MAX_PLY guards. (Real
    // evals live in ±4095, so this clamp is purely defensive.)
    adjusted.clamp(-MATE_IN_MAX_PLY + 1, MATE_IN_MAX_PLY - 1)
}

/// Update correction history entry with gravity.
fn update_corr_entry(entry: &mut i32, scaled_err: i32, cap_div_10x: i32) {
    // Proportional gravity (consensus: every top engine uses this)
    // Self-limiting: values near the limit get pulled back harder
    // cap_div_10x is stored × 10 (fixed-point); cap = LIMIT * 10 / cap_div_10x.
    let cap = CORR_HIST_LIMIT * 10 / cap_div_10x.max(1);
    let bonus = scaled_err.clamp(-cap, cap);
    *entry += bonus - *entry * bonus.abs() / CORR_HIST_LIMIT;
    *entry = (*entry).clamp(-CORR_HIST_LIMIT, CORR_HIST_LIMIT);
}

/// Update all correction history tables.
/// Train the correction tables on the search-vs-eval residual.
///
/// `corrected_baseline` is the CORRECTED, halfmove-scaled static eval — NOT the
/// raw NNUE output. Training against the corrected residual (rather than raw) is
/// deliberate and load-bearing: on raw, the gravity fixed point becomes the rail
/// itself, which manufactures phantom evals in fortress positions. The sole
/// caller passes `static_eval` for this reason; do not "correct" it to pass
/// `raw_eval`.
fn update_correction_history(info: &mut SearchInfo, board: &Board, search_score: i32, corrected_baseline: i32, depth: i32, ply: usize) {
    // Consensus shape: feed the FULL error scaled by depth, clamping only the
    // resulting bonus (at the gravity cap, in update_corr_entry). Pre-clamping
    // the error instead — e.g. to ±3cp — turns corrhist into a sign-only
    // integrator, with a max update of 21 against a cap near 341. No surveyed engine
    // clamps the input error: SF err*depth*12/128, Obsidian err*depth/8, all clamped at the output only.
    let err = search_score - corrected_baseline;
    let weight = (depth + 1).min(tp(&CORR_UPDATE_WEIGHT_MAX));
    let scaled_err = err * weight * 10 / tp(&CORR_ERR_DIV_10X).max(10);
    // Pass raw stored value; consumer treats it as fixed-point (×10).
    let cap_div = CORR_BONUS_CAP_DIV_10X.load(Ordering::Relaxed);
    let stm = board.side_to_move as usize;

    // Pawn correction
    let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.pawn_corr[stm][pawn_idx], scaled_err, cap_div);

    // Non-pawn corrections (per color)
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.np_corr[stm][WHITE as usize][white_np_idx], scaled_err, cap_div);
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
    update_corr_entry(&mut info.np_corr[stm][BLACK as usize][black_np_idx], scaled_err, cap_div);

    // Continuation correction — paired 2-ply/4-ply (H1). Index by the LAST move
    // (ply-1); update the ply-2 and ply-4 subtables. Reads moved_piece_stack
    // into locals before mutating cont_corr (disjoint but same-struct borrow).
    if ply >= 2 {
        let cur_p = info.moved_piece_stack[ply - 1] as usize;
        let cur_t = info.moved_to_stack[ply - 1] as usize;
        if cur_p != 0 && cur_p < crate::movepicker::CONT_PLANES && cur_t < 64 {
            for off in [2usize, 4] {
                if ply >= off {
                    let pp = info.moved_piece_stack[ply - off] as usize;
                    let pt = info.moved_to_stack[ply - off] as usize;
                    if pp != 0 && pp < crate::movepicker::CONT_PLANES && pt < 64 {
                        update_corr_entry(&mut info.cont_corr[pp][pt][cur_p][cur_t], scaled_err, cap_div);
                    }
                }
            }
        }
    }
    // Transition correction (zobrist-delta of last move in context)
    if !board.undo_stack.is_empty() {
        let last = &board.undo_stack[board.undo_stack.len() - 1];
        if last.mv != NO_MOVE {
            let trans_idx = ((board.hash ^ last.hash) as usize) & (CORR_HIST_SIZE - 1);
            update_corr_entry(&mut info.trans_corr[stm][trans_idx], scaled_err, cap_div);
        }
    }
}

/// LMR reduction tables (quiet and capture). Storage is `AtomicI32` so
/// SPSA-driven `setoption LMR_C_QUIET/CAP` (which calls `init_lmr()` from
/// the UCI thread while helper threads are reading) is not Rust UB on
/// concurrent access. Relaxed ordering is sufficient — readers tolerate
/// either-old-or-new per-cell values; there's no data dependency between
/// cells. Do NOT revert this to `static mut [[i32; 64]; 64]`: that is UB under
/// Rust's memory model, and produced ARM-visible inconsistent reads during
/// setoption storms.
static LMR_TABLE: [[AtomicI32; 64]; 64] = {
    const Z: AtomicI32 = AtomicI32::new(0);
    const ROW: [AtomicI32; 64] = [Z; 64];
    [ROW; 64]
};
static LMR_TABLE_CAP: [[AtomicI32; 64]; 64] = {
    const Z: AtomicI32 = AtomicI32::new(0);
    const ROW: [AtomicI32; 64] = [Z; 64];
    [ROW; 64]
};

/// Centi-ply fixed-point scale for the LMR reduction accumulator: reductions
/// are carried in 1/100ths of a ply and FLOOR-rounded to integer plies once,
/// at the end of the LMR block. floor(floor(100x)/100) == floor(x), so at
/// default behaviour the integer plies are bit-identical to an unscaled table.
/// This is what makes fractional (centi-ply) reductions expressible.
pub const LMR_SCALE: i32 = 100;

pub fn init_lmr() {
    for depth in 1..64 {
        for moves in 1..64 {
            // Quiet table: C from tunable (default 130 = 1.30). CENTI-PLY.
            if depth >= 3 && moves >= 3 {
                let c = tp(&LMR_C_QUIET) as f64 / 100.0;
                // Additive base in exact centi (post-scale, so 20 = 0.20 plies
                // uniformly). It must stay OUTSIDE the float-to-int
                // truncation, or the offset rounds away to a no-op.
                let r = tp(&LMR_BASE_CENTI)
                    + (LMR_SCALE as f64 * (depth as f64).ln() * (moves as f64).ln() / c) as i32;
                LMR_TABLE[depth][moves].store(r.min((depth - 2) as i32 * LMR_SCALE), Ordering::Relaxed);
            }
            // Capture table: C from tunable (default 180 = 1.80). CENTI-PLY.
            if depth >= 3 && moves >= 3 {
                let c = tp(&LMR_C_CAP) as f64 / 100.0;
                let r = (LMR_SCALE as f64 * (depth as f64).ln() * (moves as f64).ln() / c) as i32;
                LMR_TABLE_CAP[depth][moves].store(r.min((depth - 2) as i32 * LMR_SCALE), Ordering::Relaxed);
            }
        }
    }
}

fn lmr_cap_reduction(depth: i32, moves: i32) -> i32 {
    let d = (depth as usize).min(63);
    let m = (moves as usize).min(63);
    LMR_TABLE_CAP[d][m].load(Ordering::Relaxed)
}

fn lmr_reduction(depth: i32, moves: i32) -> i32 {
    let d = (depth as usize).min(63);
    let m = (moves as usize).min(63);
    LMR_TABLE[d][m].load(Ordering::Relaxed)
}

#[inline(always)]
fn root_move_index(mv: Move) -> usize {
    let promotion_bucket = if is_promotion(mv) {
        (move_flags(mv) - FLAG_PROMOTE_N) as usize
    } else {
        ROOT_MOVE_BUCKETS - 1
    };
    ((move_from(mv) as usize * 64 + move_to(mv) as usize) * ROOT_MOVE_BUCKETS)
        + promotion_bucket
}

/// Initialize feature flags from environment variables (called once at process startup).
/// NO_XXX=1 disables individual features. DISABLE_ALL=1 disables everything,
/// then ENABLE_XXX=1 re-enables individual features.
///
/// `pub(crate)` and `Once`-guarded: the UCI `eval` command calls it too, so a
/// bare `eval` with no preceding `go` still reflects the ablation env vars
/// rather than silently reporting defaults.
pub(crate) fn init_feature_flags() {
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
            if std::env::var("ENABLE_BAD_NOISY").is_ok() { FEAT_BAD_NOISY.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_EXTENSIONS").is_ok() { FEAT_EXTENSIONS.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_FH_BLEND").is_ok() { FEAT_FH_BLEND.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_IIR").is_ok() { FEAT_IIR.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_HINDSIGHT").is_ok() { FEAT_HINDSIGHT.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_CORRECTION").is_ok() { FEAT_CORRECTION.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_PVS").is_ok() { FEAT_PVS.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_TT_CUTOFF").is_ok() { FEAT_TT_CUTOFF.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_TT_NEARMISS").is_ok() { FEAT_TT_NEARMISS.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_TT_STORE").is_ok() { FEAT_TT_STORE.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_TT_STATIC_EVAL").is_ok() { FEAT_TT_STATIC_EVAL.store(true, Ordering::Relaxed); }
            if std::env::var("ENABLE_QS_CAPTURES").is_ok() { FEAT_QS_CAPTURES.store(true, Ordering::Relaxed); }
        } else {
            if std::env::var("NO_NMP").is_ok() { FEAT_NMP.store(false, Ordering::Relaxed); }
            if std::env::var("NO_RFP").is_ok() { FEAT_RFP.store(false, Ordering::Relaxed); }
            if std::env::var("NO_PROBCUT").is_ok() { FEAT_PROBCUT.store(false, Ordering::Relaxed); }
            if std::env::var("NO_LMR").is_ok() { FEAT_LMR.store(false, Ordering::Relaxed); }
            if std::env::var("NO_LMP").is_ok() { FEAT_LMP.store(false, Ordering::Relaxed); }
            if std::env::var("NO_FUTILITY").is_ok() { FEAT_FUTILITY.store(false, Ordering::Relaxed); }
            if std::env::var("NO_SEE_PRUNE").is_ok() { FEAT_SEE_PRUNE.store(false, Ordering::Relaxed); }
            if std::env::var("NO_BAD_NOISY").is_ok() { FEAT_BAD_NOISY.store(false, Ordering::Relaxed); }
            if std::env::var("NO_EXTENSIONS").is_ok() { FEAT_EXTENSIONS.store(false, Ordering::Relaxed); }
            if std::env::var("NO_FH_BLEND").is_ok() { FEAT_FH_BLEND.store(false, Ordering::Relaxed); }
            if std::env::var("NO_IIR").is_ok() { FEAT_IIR.store(false, Ordering::Relaxed); }
            if std::env::var("NO_HINDSIGHT").is_ok() { FEAT_HINDSIGHT.store(false, Ordering::Relaxed); }
            if std::env::var("NO_CORRECTION").is_ok() { FEAT_CORRECTION.store(false, Ordering::Relaxed); }
            if std::env::var("NO_PVS").is_ok() { FEAT_PVS.store(false, Ordering::Relaxed); }
            if std::env::var("NO_TT_CUTOFF").is_ok() { FEAT_TT_CUTOFF.store(false, Ordering::Relaxed); }
            if std::env::var("NO_TT_NEARMISS").is_ok() { FEAT_TT_NEARMISS.store(false, Ordering::Relaxed); }
            if std::env::var("NO_TT_STORE").is_ok() { FEAT_TT_STORE.store(false, Ordering::Relaxed); }
            if std::env::var("NO_TT_STATIC_EVAL").is_ok() { FEAT_TT_STATIC_EVAL.store(false, Ordering::Relaxed); }
            if std::env::var("NO_SAT_TIEBREAK").is_ok() { FEAT_SAT_TIEBREAK.store(false, Ordering::Relaxed); }
            if std::env::var("NO_QS_CAPTURES").is_ok() { FEAT_QS_CAPTURES.store(false, Ordering::Relaxed); }
            if std::env::var("NO_SINGULAR").is_ok() { FEAT_SINGULAR.store(false, Ordering::Relaxed); }
            if std::env::var("NO_CUCKOO").is_ok() { FEAT_CUCKOO.store(false, Ordering::Relaxed); }
            if std::env::var("NO_4D_HISTORY").is_ok() { FEAT_4D_HISTORY.store(false, Ordering::Relaxed); }
        }
        // Diagnostic audit modes (orthogonal to DISABLE_ALL/NO_XXX).
        if std::env::var("RFP_AUDIT").is_ok() {
            RFP_AUDIT.store(true, Ordering::Relaxed);
            eprintln!("RFP_AUDIT enabled: null-verifying every RFP cutoff (diagnostic, slow)");
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
pub(crate) fn create_helper_info(main: &SearchInfo) -> SearchInfo {
    // Use the shared TT directly (avoids allocating a throwaway 1 MB TT
    // and the misleading "TT 1 MB" info string that prints before swap).
    let mut helper = SearchInfo::new_with_tt(main.tt.clone());
    helper.stop = main.stop.clone();         // share the same stop flag
    // Share ponderhit_time so helpers respect the ponderhit deadline set by
    // the UCI thread. If a helper keeps its own AtomicU64 it stays at 0, the
    // helper ignores the deadline entirely, and it only stops when main sets
    // the shared stop flag — burning CPU for the grace window on every
    // ponderhit.
    helper.ponderhit_time = main.ponderhit_time.clone();
    helper.ponderhit_soft = main.ponderhit_soft.clone();
    helper.ponderhit_floor = main.ponderhit_floor.clone();
    helper.ponderhit_isoft = main.ponderhit_isoft.clone();
    helper.ph_fl_active = main.ph_fl_active.clone();
    // Share the in-flight post-ponderhit forfeit guard too (same rationale as
    // ponderhit_time). root_fail_low is deliberately NOT shared: it is the
    // MAIN thread's root aspiration state — a helper's own fail-lows must not
    // clobber the instant-reply gate (helpers keep their fresh, inert Arc).
    helper.ponderhit_abs = main.ponderhit_abs.clone();
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
    helper.tb_probe_depth = main.tb_probe_depth; // same probe-depth gate as main

    // Seed helpers with the main thread's accumulated, aged history.
    // Cross-engine consensus (SF/Obsidian/Alexandria/
    // PlentyChess): helpers never start from zero — they either share
    // history with main lockless, persist it across `go` calls, or get
    // a copy. Coda spawns helpers per `go`, so we copy from main
    // (which has just been aged ×0.80 at the top of `start_search`).
    //
    // Without this, helpers spend the first few iterations re-learning
    // move ordering from scratch, generating TT entries with worse
    // ordering than main; this both wastes their work AND poisons
    // shared-TT ordering, costing most of the potential SMP Elo.
    // One-time cold seed: copy main's history + corr so the worker's first
    // search isn't ordering-blind. Subsequent searches age the worker's own
    // history (SMP diversity) via refresh_helper_per_go.
    seed_helper_from_main(&mut helper, main);

    helper
}

/// Refresh a helper's per-`go` state from the current main thread: re-share
/// the mutable Arcs (they can be swapped between `go`s — TT resize, a fresh
/// SearchInfo per search, ponderhit deadlines) and seed history/correction
/// tables from main's just-aged copies. Split out of `create_helper_info` so
/// the persistent thread pool can reuse a worker across `go`s: the one-time
/// allocations (NNUE acc, threat stack, table boxes) stay put, only this
/// cheap per-search state is refreshed. Behavior-identical to the old inline
/// copy for the per-`go` spawn path. (Stage 2 will replace the history/corr
/// copy here with in-place aging to keep worker-learned diversity.)
/// State refreshed identically on BOTH the one-time seed and every per-`go`
/// refresh: re-share the mutable Arcs the UCI loop may have swapped (TT on Hash
/// resize; stop/ponderhit/global_nodes on the per-`go` SearchInfo swap), copy
/// the CORRECTION tables from main, and reset the per-search state a fresh
/// helper would have zeroed.
///
/// Corrhist is COPIED from main every go on purpose: it feeds the corrected
/// static eval, so a helper running its OWN corrhist would evaluate differently
/// from main, its shared-TT entries would carry divergent scores, and the
/// search would diverge at T>1 (worth about -8 Elo). Move-ordering history
/// (below) is different — divergence there IS the Lazy-SMP diversity mechanism.
fn refresh_helper_common(helper: &mut SearchInfo, main: &SearchInfo) {
    helper.tt = main.tt.clone();
    helper.stop = main.stop.clone();
    helper.ponderhit_time = main.ponderhit_time.clone();
    helper.ponderhit_soft = main.ponderhit_soft.clone();
    helper.ponderhit_floor = main.ponderhit_floor.clone();
    helper.ponderhit_isoft = main.ponderhit_isoft.clone();
    helper.ph_fl_active = main.ph_fl_active.clone();
    helper.ponderhit_abs = main.ponderhit_abs.clone(); // in-flight forfeit guard
    helper.global_nodes = main.global_nodes.clone();
    helper.thread_bmc = main.thread_bmc.clone(); // shared per-thread bmc array (SF cross-thread TM)
    helper.syzygy = main.syzygy.clone();
    helper.tb_probe_depth = main.tb_probe_depth;

    // Correction tables — copied for eval consistency (see fn-doc). ~3.14 MiB.
    helper.pawn_corr.copy_from_slice(&main.pawn_corr[..]);
    helper.np_corr.copy_from_slice(&main.np_corr[..]);
    helper.cont_corr.copy_from_slice(&main.cont_corr[..]);
    helper.trans_corr.copy_from_slice(&main.trans_corr[..]);

    // pawn_hist is position-specific (indexed by pawn hash); a helper's
    // self-accumulated table carries toxic stale ordering across positions
    // (measured -8 Elo at T=4), so it is cleared every go even in Stage 2.
    helper.clear_pawn_hist();
    // Per-search scalars a fresh helper had zeroed.
    helper.nmp_min_ply = 0;
    helper.rfp_audit_active = false;
    helper.max_nodes = 0;
}

/// One-time seed of a freshly-built pool worker from main: cold workers copy
/// main's accumulated (already-aged) history so their first search isn't
/// starting from zero ordering.
pub(crate) fn seed_helper_from_main(helper: &mut SearchInfo, main: &SearchInfo) {
    refresh_helper_common(helper, main);
    helper.history.copy_from(&main.history);
}

/// Per-`go` refresh of a reused pool worker (Stage 2 — SMP diversity). Unlike
/// the seed, the worker KEEPS its own move-ordering `history` across `go`s and
/// AGES it (same ×4/5 decay main applies at search start) instead of recopying
/// main's. Over successive moves each worker's history diverges from main's and
/// from the other workers', which is the Lazy-SMP search-diversity source Coda
/// previously threw away by rebuilding a fresh helper every move. Eval-side
/// state (corrhist) is still copied from main by `refresh_helper_common` for
/// consistency, and pawn_hist is still cleared.
pub(crate) fn refresh_helper_per_go(helper: &mut SearchInfo, main: &SearchInfo) {
    refresh_helper_common(helper, main);
    helper.history.age(4, 5);
}

/// Per-`go` preparation of a helper `SearchInfo` for a search on `board`:
/// timing zeroed (helpers never manage time), the accumulator rebuilt for the
/// root position, and the few counters `search_helper` does NOT itself reset
/// zeroed (load-bearing when the thread pool reuses a `SearchInfo` across `go`s;
/// a no-op for the fresh per-`go` spawn path). Does NOT set `max_depth` — the
/// caller sets that from main's depth. Threat-stack + the bulk of scratch state
/// are reset inside `search_helper`.
pub(crate) fn prepare_helper_for_search(info: &mut SearchInfo, board: &Board) {
    info.start_time = Instant::now();
    info.time_limit = 0;
    info.soft_limit = 0;
    info.hard_limit = 0;
    info.soft_floor = 0;
    // Reset counters search_helper leaves alone — stale values from a prior
    // pooled search would otherwise leak into the vote if this search completes
    // zero iterations (instant-stop).
    info.completed_depth = 0;
    info.last_score = 0;
    info.sel_depth = 0;
    info.tb_hits = 0;
    // Rebuild the NNUE accumulator for the root position.
    if let Some(acc) = &mut info.nnue_acc {
        acc.reset();
    }
    if let (Some(net), Some(acc)) = (&info.nnue_net, &mut info.nnue_acc) {
        acc.materialize(net, board);
    }
}

/// Run one helper search to completion and return the vote tuple
/// `(nodes, best_move, score, completed_depth, seldepth, tb_hits, ponder)`.
/// Shared verbatim by
/// the per-`go` spawn path and the persistent thread pool so both produce
/// byte-identical helper behavior. `search_helper` ignores its `_limits`
/// (helpers take depth from `info.max_depth` and stop on the shared flag), so
/// a zeroed placeholder is passed.
pub(crate) fn helper_run(
    info: &mut SearchInfo,
    board: &mut Board,
    max_depth: i32,
    thread_id: usize,
) -> (u64, Move, i32, i32, i32, u64, Move) {
    prepare_helper_for_search(info, board);
    info.max_depth = max_depth;
    let placeholder = SearchLimits {
        depth: max_depth, fixed_depth: true, movetime: 0, wtime: 0, btime: 0, winc: 0, binc: 0,
        movestogo: 0, nodes: 0, infinite: false, movetime_floor: 0,
        min_think_ms: 0, abs_clock: 0,
    };
    let mv = search_helper(board, info, &placeholder, thread_id);
    let ponder = if info.pv_len[0] >= 2 { info.pv_table[0][1] } else { NO_MOVE };
    (
        info.nodes,
        mv,
        info.last_score,
        info.completed_depth,
        info.sel_depth,
        info.tb_hits,
        ponder,
    )
}

/// Identity of the NNUE net a `SearchInfo` currently holds (0 if none). The
/// thread pool rebuilds when this changes across `go`s, since each worker's
/// accumulator is sized for a specific net (NNUEFile swap → different net).
pub(crate) fn nnue_net_identity(info: &SearchInfo) -> usize {
    info.nnue_net
        .as_ref()
        .map(|n| std::sync::Arc::as_ptr(n) as *const () as usize)
        .unwrap_or(0)
}

/// Compute time-management budgets from clock state. Returns
/// (soft, hard, max, soft_floor) all in milliseconds.
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
/// MoveOverhead UCI option (default 100ms). `ponder_on` is the `Ponder`
/// UCI option state (callers pass `ponder_enabled()`): when true the
/// optimum gets the +25% ponder pre-funding bump (see PONDER_OPT_BUMP_PCT).
pub fn compute_tm_budgets(
    our_time: u64,
    our_inc: u64,
    movestogo: u32,
    overhead: u64,
    fullmove: u16,
    ponder_on: bool,
) -> (u64, u64, u64, u64) {
    // TM windows: opt/hard/max model with a multiplicative factor product —
    // the structure common to modern engines (SF, Obsidian, Hobbes and
    // PlentyChess all run variants of it):
    //
    //   max_time  = clock × 0.60 − overhead   (absolute single-move ceiling)
    //   hard_time = clock × 0.46              (mid-search abort, clamped to max)
    //   opt_time  = (clock/24 + inc × 0.94 − overhead) × 0.73, clamped to hard
    //
    // The dynamic-block factor multiplier scales opt UP toward hard/max, so no
    // separate cap is needed. A wide hard window with bounded factors is
    // deliberate: the alternative (tight hard window, wide factors, explicit
    // cap) clamps most iterations at the cap, which silently swallows the
    // legitimate 4-11x spikes the factors exist to produce.
    //
    // The window fractions and factor constants are Coda's own tunable
    // parameters, tuned and validated on Coda's own search + net (see the TM
    // tunables block near the top of this file; full provenance in
    // docs/license_analysis_2026-07-13.md). They are ordinary tuning values —
    // functional operating points. Everything wrapping them is Coda-original:
    // the no-inc sudden-death caps, adaptive moves-to-go growth, phase-scaling,
    // the ponder bump, the cross-move score-trend and cross-thread-instability
    // factors, and the inc_cover ceiling.
    //
    // Returns (opt, hard, max, soft_floor). soft_floor is kept at a small
    // value (10ms) so the stockpile-prevention sleep (~line 2520) stays a
    // no-op for movetime-limited searches; the factor multiplier can pull
    // opt × multiplier well below any meaningful floor.
    let time_left = our_time.saturating_sub(overhead).max(1);
    // No-inc TCs require more conservative pacing. With inc, each move
    // costs only `inc` of net time (we regain inc per move). Without
    // inc, every spent second is gone forever. The default 25-moves-left
    // assumption produces ~7s/move on 3+0 (180s base) — but real games
    // run 40-80 moves, so 25-move pacing leaves the engine massively
    // out of time — observed in live play as 8-12s early moves at 3+0 and a
    // forfeit. Use 40 moves at no-inc to pace tighter.
    //
    // At moderate inc the base moves-to-go is deliberately HIGH (i.e. a low
    // base spend, ~2-3% of remaining, matching SF/Obsidian rather than the
    // ~4.8% a 25-move assumption gives). A low baseline is what gives the
    // multiplicative factors headroom to spike 4-5x without forfeiting; spiking
    // off a high baseline just hits the ceiling. The no-inc and explicit-
    // movestogo paths do not use this value.
    // Window fractions (provenance in the header). Read from the (non-core) TM
    // tunables so a TM-cluster SPSA can move them; the defaults reproduce the
    // prior hardcoded values exactly. max_time is the single-move ceiling.
    let max_bank_1000 = tp(&TM_MAX_BANK_1000).max(1) as u64;
    let hard_window_pct = tp(&TM_HARD_WINDOW_PCT).max(1) as u64;
    let opt_window_pct = tp(&TM_OPT_WINDOW_PCT).max(1) as u64;
    let inc_frac_pct = tp(&TM_INC_FRAC_PCT).max(0) as u64;
    let default_moves_to_go = tp(&TM_DEFAULT_MTG).max(2) as u64;

    // No-inc sudden-death TCs need a tighter ceiling. The 60%/46% windows
    // are fine at moderate-inc (each spent ms gets refilled
    // by inc) but catastrophic at 3+0, as seen in live play: at move 6 with a
    // 166s clock, hard_time is 76s; one deep iteration runs to that ceiling
    // and the geometric decay from there reaches flag-fall by move 24.
    // For no-inc sudden death: cap max at 15% of clock, hard at 10%. With
    // movestogo > 0 the explicit count drives allocation, so no extra cap
    // needed there.
    let no_inc_sd = our_inc == 0 && movestogo == 0;
    // Low-inc absolute single-move ceiling (overspend PART2 — see the
    // TM_INC_HARD_MULT tunable comment). Stable across the game: keyed on the
    // constant increment, not on shrinking time_left, so it never flips. Only
    // applied on the inc>0 / non-movestogo path (no_inc_sd has its own tighter
    // 15%/10% caps; movestogo paces from the explicit count).
    let inc_hard_ceiling = if !no_inc_sd && movestogo == 0 {
        our_inc.saturating_mul(tp(&TM_INC_HARD_MULT).max(0) as u64)
            .saturating_add(tp(&TM_INC_HARD_FLOOR_MS).max(0) as u64)
            .max(1)
    } else {
        u64::MAX
    };
    let max_time = if no_inc_sd {
        (time_left * 15 / 100).max(1)
    } else {
        (time_left * max_bank_1000 / 1000).min(inc_hard_ceiling).max(1)
    };
    let hard_time = if no_inc_sd {
        (time_left * 10 / 100).min(max_time).max(1)
    } else {
        (time_left * hard_window_pct / 100).min(max_time).max(1)
    };

    // No-inc sudden death needs a higher moves-left assumption than
    // moderate-inc TCs: each spent ms is gone forever, and real 3+0 games
    // run 40-80 moves. The default-moves-to-go 24 (sudden-death pacing) was
    // calibrated for inc TCs where the inc term dominates pacing — at
    // no-inc the 24 produces an opt that's high enough for the factor
    // multiplier (up to ~6.5×) to consistently blow past hard_time,
    // making hard the binding constraint every move — a flat uniform-spend
    // pattern, seen in live 3+0 play.
    //
    // The assumption also has to ADAPT: a fixed base never tightens as a game
    // outlives it (at move 80 it still assumes "40 moves left"). Symptom, from
    // both live losses and local no-adjudication RR at 30+0: 70-88% of the
    // clock burned by move ~60 in games running 130-220+ plies, then an
    // out-of-time forfeit, while peer engines forfeited none. So once fullmove
    // exceeds NO_INC_MTG_BASE, grow the divisor by NO_INC_MTG_GROWTH_PCT% of
    // the overrun:
    //     effective_mtg = base + growth_pct/100 * max(0, fullmove - base)
    //
    // These two must stay TUNABLE, not hardcoded. Fixed constants (base=40,
    // growth=100%) eliminate forfeits entirely yet still lose ~3.5 Elo: the
    // mechanism is right but that operating point over-tightens mid-game
    // allocation in the majority of games, which were never going to forfeit.
    // Any change here needs both a forfeit count and a non-regression SPRT.
    let no_inc_mtg_base = tp(&NO_INC_MTG_BASE).max(1) as u64;
    let no_inc_growth_pct = tp(&NO_INC_MTG_GROWTH_PCT).max(0) as u64;
    let no_inc_effective_mtg = no_inc_mtg_base
        + (fullmove as u64).saturating_sub(no_inc_mtg_base) * no_inc_growth_pct / 100;
    let mtg_divisor = if no_inc_sd { no_inc_effective_mtg.max(1) } else { default_moves_to_go };

    let opt_time_base = if movestogo > 0 {
        // Movestogo: divisor is clamped to [2, default_mtg]. The increment
        // term matters here too — omitting it (while the sudden-death branch
        // credits its 94%) under-allocates ~0.7*inc per move at movestogo+inc
        // TCs, the CCRL-style ones. Same INC_FRAC weighting as the SD branch.
        let divisor = (movestogo as u64).clamp(2, default_moves_to_go);
        let computed = time_left / divisor + our_inc * inc_frac_pct / 100;
        (computed.min(max_time) * opt_window_pct / 100).max(1)
    } else {
        // Sudden death (or with inc). Add 94% of inc to base computed window.
        let computed = time_left / mtg_divisor + our_inc * inc_frac_pct / 100;
        ((computed.min(max_time) * opt_window_pct / 100).min(hard_time)).max(1)
    };

    // Opening-phase damp (Hobbes pattern). Without it the opening is grossly
    // overspent: stability is 0 at search start, so the factor multiplier is at
    // its 2.5× maximum on the very first moves, and ten of those can eat an
    // entire 30+0.25 budget before the middlegame. Hobbes embeds the damp in
    // its base — `soft_scale = 0.024 + 0.042 × (1 - exp(-0.045 × fm))`, a
    // ~0.36 → 1.0 ramp. Coda applies the same exponential as a multiplier:
    //
    //   phase_mult = 0.22 + 0.78 × (1 - exp(-0.045 × fullmove))
    //     fm=1:  0.25×
    //     fm=5:  0.38×
    //     fm=10: 0.50×
    //     fm=20: 0.68×
    //     fm=40: ~0.87×
    //
    // The 0.22 floor is load-bearing and was arrived at empirically: a softer
    // floor (0.36, giving 0.39× at fm=1) still left most games overspending the
    // opening. Skip the damp when movestogo > 0 — that path paces from the
    // explicit count.
    let opt_time = if movestogo > 0 {
        opt_time_base
    } else {
        let phase_mult = 0.22 + 0.78 * (1.0 - (-0.045 * fullmove as f64).exp());
        ((opt_time_base as f64) * phase_mult.clamp(0.22, 1.0)) as u64
    };
    // +25% optimum when the Ponder UCI option is on — SF's ponder-optimum
    // pre-funding semantics, applied on EVERY move when pondering is enabled.
    // The average move is refunded by the pondered time itself (full-charge
    // model) and by the stopOnPonderhit-style instant replies. Those
    // compensators are not optional: the bump only balances as a set with them.
    //
    // Plain const, DELIBERATELY NOT in `tunables!`: OB/fastchess cannot
    // ponder, so SPSA would only ever see this as a dead knob and detune it
    // on noise. Sweep it in a local ponder gauntlet if needed.
    const PONDER_OPT_BUMP_PCT: u64 = 25;
    let opt_time = if ponder_on {
        opt_time + opt_time * PONDER_OPT_BUMP_PCT / 100
    } else {
        opt_time
    };
    let opt_time = opt_time.max(1).min(hard_time);

    // soft_floor: preserved at a small value (10ms) for stockpile sleep
    // compatibility. The factor multiplier in dynamic TM block can pull
    // soft well below this; there is no other enforced floor.
    let soft_floor: u64 = 10;

    (opt_time, hard_time, max_time, soft_floor)
}

/// A completed root iteration eligible for Lazy-SMP best-thread selection.
#[derive(Clone, Copy)]
struct SmpCandidate {
    mv: Move,
    score: i32,
    depth: i32,
    sel_depth: i32,
    ponder: Move,
    is_main: bool,
}

fn select_smp_candidate(cands: &[SmpCandidate], allow_helper: bool) -> Option<usize> {
    if !allow_helper {
        return cands.iter().position(|c| c.is_main);
    }
    if cands.is_empty() {
        return None;
    }

    // Sum depth-weighted votes per move. The offset keeps the lowest-scoring
    // candidate's vote positive, so depth still breaks equal-score choices.
    let min_score = cands.iter().map(|c| c.score).min().unwrap();
    let mut votes: Vec<(Move, i64)> = Vec::with_capacity(cands.len());
    for c in cands {
        let weight = c.depth as i64 * (c.score as i64 - min_score as i64 + 14);
        if let Some(entry) = votes.iter_mut().find(|(m, _)| *m == c.mv) {
            entry.1 += weight;
        } else {
            votes.push((c.mv, weight));
        }
    }
    let vote_of = |mv: Move| {
        votes
            .iter()
            .find(|(m, _)| *m == mv)
            .map(|(_, weight)| *weight)
            .unwrap_or(0)
    };

    let mut best = 0usize;
    for i in 1..cands.len() {
        let (score, mv, depth) = (cands[i].score, cands[i].mv, cands[i].depth);
        let (best_score, best_move) = (cands[best].score, cands[best].mv);
        if best_score >= MATE_IN_MAX_PLY {
            if score > best_score {
                best = i;
            }
        } else if score >= MATE_IN_MAX_PLY
            || (score > -MATE_IN_MAX_PLY
                && (vote_of(mv) > vote_of(best_move)
                    || (vote_of(mv) == vote_of(best_move) && depth > cands[best].depth)))
        {
            best = i;
        }
    }
    Some(best)
}

/// Run Lazy SMP search: main thread + N-1 helper threads.
pub fn search_smp(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits, threads: usize) -> Move {
    // Advance the TT generation HERE, before spawning helpers — not inside
    // search(). Otherwise helpers write TT entries with the old generation in
    // the microsecond window between spawn and main's new_search() call, and
    // those stale entries then look freshest to replacement. search() does not
    // bump; the single-thread path bumps here too, for consistency.
    info.tt.new_search();
    info.num_threads = threads; // gates the cross-thread instability TM factor

    if threads <= 1 {
        info.global_nodes.store(0, Ordering::Relaxed);
        info.last_flushed_nodes.set(0);
        return search(board, info, limits);
    }

    // Reset the per-thread best-move-change counters for this search.
    for slot in info.thread_bmc.iter().take(threads) {
        slot.store(0, Ordering::Relaxed);
    }

    // Reset shared state.
    // Note: stop flag is cleared by the UCI thread before spawning the search
    // thread, not here. Clearing here races with ponderhit (which sets stop
    // before the search thread starts).
    info.global_nodes.store(0, Ordering::Relaxed); // Reset before helpers start

    // Dispatch the persistent helper pool (Stage 1). Reuses parked worker
    // threads + their SearchInfos across `go` — refreshing per-search state from
    // main via refresh_helper_per_go — instead of spawning `threads-1` fresh
    // threads and allocating ~13 MB of tables per move. Behavior-identical to
    // the old per-go spawn: workers run the same `helper_run`. The pool is
    // dispatched now and collected after main's search + stop below.
    crate::thread_pool::dispatch(threads - 1, info, board, limits.depth);

    // Main thread searches normally
    info.defer_final_info = true;
    let main_move = search(board, info, limits);
    info.defer_final_info = false;
    let main_score = info.last_score;
    let main_depth = info.completed_depth;
    let main_sel_depth = info.sel_depth;

    // Signal all helpers to stop
    info.stop.store(true, Ordering::Relaxed);

    // Collect per-thread candidates.
    // Helpers now also return their 2nd PV move (ponder) so a winning helper
    // can hand uci.rs a consistent bestmove+ponder pair.
    let mut total_nodes = info.nodes;
    let mut cands: Vec<SmpCandidate> = Vec::with_capacity(threads);
    let main_ponder = if info.pv_len[0] >= 2 { info.pv_table[0][1] } else { NO_MOVE };
    if main_move != NO_MOVE && main_depth > 0 {
        cands.push(SmpCandidate {
            mv: main_move,
            score: main_score,
            depth: main_depth,
            sel_depth: main_sel_depth,
            ponder: main_ponder,
            is_main: true,
        });
    }
    for (helper_nodes, mv, score, depth, sel_depth, tb_hits, ponder) in crate::thread_pool::collect() {
        total_nodes += helper_nodes;
        info.tb_hits += tb_hits;
        if mv != NO_MOVE && depth > 0 {
            cands.push(SmpCandidate {
                mv,
                score,
                depth,
                sel_depth,
                ponder,
                is_main: false,
            });
        }
    }
    info.nodes = total_nodes;

    // Nothing completed a real iteration — fall back to main's move (which may
    // itself be NO_MOVE only in pathological instant-stop cases).
    if cands.is_empty() {
        emit_final_info(info, board, total_nodes);
        return main_move;
    }

    // Select the best THREAD, not just the max-vote move (SF get_best_thread):
    // prefer a proven win (shortest mate = highest score); otherwise switch to a
    // thread whose move has more votes (deeper on ties), but never onto a proven
    // loss. Picking a thread (vs a bare move) is what lets us carry a consistent
    // PV/ponder out — the previous `max_by_key(votes)` returned a move with no
    // owning thread, so on any vote-override uci.rs saw pv_table[0][0] != bestmove
    // and dropped the ponder entirely.
    // An explicit fixed-depth request always keeps main's completed result;
    // helpers may be interrupted at a shallower iteration when main finishes.
    let Some(best) = select_smp_candidate(&cands, !limits.fixed_depth) else {
        emit_final_info(info, board, total_nodes);
        return main_move;
    };

    // If a non-main thread won, adopt its move + ponder into info so uci.rs sees
    // pv_table[0][0] == returned bestmove and can emit the ponder. Main's own PV
    // is already in info and richer, so leave it when main wins.
    let winner_mv = cands[best].mv;
    if !cands[best].is_main {
        info.pv_table[0][0] = winner_mv;
        info.last_score = cands[best].score;
        info.completed_depth = cands[best].depth;
        info.sel_depth = cands[best].sel_depth;
        if cands[best].ponder != NO_MOVE {
            info.pv_table[0][1] = cands[best].ponder;
            info.pv_len[0] = 2;
        } else {
            // No ponder from this thread — clear the slot so uci.rs's ponder
            // legality check (which reads pv_table[0][1] without a pv_len gate)
            // fails cleanly instead of emitting a stale ponder.
            info.pv_table[0][1] = NO_MOVE;
            info.pv_len[0] = 1;
        }
    }
    // search() deferred its search-end line. Emit the final thread choice and
    // exact aggregate counters now so the last UCI info agrees with bestmove.
    emit_final_info(info, board, total_nodes);
    winner_mv
}

/// Helper thread search — full aspiration ID loop matching the main
/// thread's `search()`, just silent (no UCI output, no TM).
///
/// Previously this was a stripped-down `negamax(-INF, +INF)` per depth
/// with no aspiration, no score carry, an empty history table, and a
/// `thread_id % 2` depth offset. Cross-engine review (SF/Obsidian/Alexandria/PlentyChess) showed every reference
/// engine runs full aspiration in helpers — aspiration-window variance
/// + slight asynchrony IS the diversity mechanism, not depth offsets.
/// Helpers with full windows search far more nodes per depth than they
/// need to and contribute worse-ordered TT entries.
///
/// History is seeded from main in `create_helper_info` — see comment
/// there. We deliberately do NOT clear it here.
pub(crate) fn search_helper(board: &mut Board, info: &mut SearchInfo, _limits: &SearchLimits, thread_id: usize) -> Move {
    init_feature_flags();

    // History was just seeded from main in create_helper_info — do
    // NOT clear it here. Reset only per-search scratch state.
    info.stats = PruneStats::default();
    info.static_evals = [0; MAX_PLY + 1];
    info.reductions = [0; MAX_PLY + 1];
    info.excluded_move = [NO_MOVE; MAX_PLY + 1];
    info.moved_piece_stack = [0; MAX_PLY + 1];
    info.double_ext_count = [0; MAX_PLY + 1];
    info.cutoff_count = [0; MAX_PLY + 4];
    info.moved_to_stack = [0; MAX_PLY + 1];
    info.pv_table = [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1];
    info.pv_len = [0; MAX_PLY + 1];
    info.nodes = 0;
    info.last_flushed_nodes.set(0);
    info.tm_has_data = false;
    info.tm_best_stable = 0;
    info.tm_best_move_changes = 0;
    info.tm_asp_fail_low = 0;
    info.tm_asp_fail_high = 0;
    info.tm_forced_state = ForcedState::None;

    // Mirror search()'s threat setup — helpers must evaluate consistently
    // with main or shared-TT entries disagree and search diverges at T>1.
    // Decide threat REFRESH mode once per search, from the root piece count.
    // Both sides of the contract read the same flag — the generator on the next
    // line and the consumer in `ensure_computed`. Deciding per node would let
    // them disagree and replay from deltas that were never generated.
    {
        let pmax = THREAT_REFRESH_PIECE_MAX.load(Ordering::Relaxed);
        let pieces = board.occupied().count_ones() as i32;
        crate::threat_accum::REFRESH_MODE.store(pmax > 0 && pieces <= pmax, Ordering::Relaxed);
    }
    board.generate_threat_deltas = info.nnue_net.as_ref().is_some_and(|n| n.has_threats)
        && !crate::threat_accum::refresh_mode();
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
    let effective_max = info.max_depth.min(ROOT_DEPTH_MAX);
    let mut prev_score = 0i32;
    // Cross-thread TM (concept from SF): track this helper's best-move changes between
    // completed iterations and publish into its own slot of the shared array.
    let mut prev_best = NO_MOVE;
    let bmc_slot = thread_id.min(info.thread_bmc.len() - 1);
    for depth in 1..=effective_max {
        if info.stop.load(Ordering::Relaxed) { break; }
        info.root_depth = depth;

        let score;

        // Aspiration windows (skip for mate scores) — mirrors search().
        if depth >= 4 && prev_score > -MATE_IN_MAX_PLY && prev_score < MATE_IN_MAX_PLY {
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
                    info.tm_asp_fail_low = info.tm_asp_fail_low.saturating_add(1);
                    beta = (3 * alpha + 5 * beta) / 8;
                    alpha = (result - delta).max(-INFINITY);
                } else if result >= beta {
                    info.tm_asp_fail_high = info.tm_asp_fail_high.saturating_add(1);
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
        // Publish a best-move change (vs the previous completed iteration) into
        // this helper's cross-thread bmc slot.
        if prev_best != NO_MOVE && best_move != prev_best {
            info.thread_bmc[bmc_slot].fetch_add(1, Ordering::Release);
        }
        prev_best = best_move;
        prev_score = score;
        info.last_score = score;
        info.completed_depth = depth;
    }

    best_move
}

/// Build the UCI PV string from `info.pv_table[0]`, extended via the TT when the
/// stored line is shorter than `target_depth`. Mirrors the per-iteration PV
/// extraction in the ID loop EXACTLY (same legality guard — printing an illegal
/// PV move is a critical bug, see the ID-loop comment). Stops at the first move
/// not legal in the walked position and at threefold repetition. Used for the
/// final search-end info line so the LAST line a GUI sees carries a full,
/// bestmove-consistent PV (some broadcast parsers, e.g. CCRL, show the PV from
/// the latest info line, so a PV-less final line displayed as a short/empty PV).
fn build_pv_string(info: &SearchInfo, board: &Board, target_depth: i32) -> String {
    let mut pv_str = String::new();
    let mut seen_hashes: Vec<u64> = board.undo_stack.iter().map(|u| u.hash).collect();
    seen_hashes.push(board.hash);
    let mut pv_board = board.clone();
    let mut pv_moves = 0usize;

    let pv_len = info.pv_len[0].min(MAX_PLY);
    for i in 0..pv_len {
        let pv_mv = info.pv_table[0][i];
        if pv_mv == NO_MOVE
            || !crate::movepicker::is_pseudo_legal(&pv_board, pv_mv)
            || !pv_board.is_legal(pv_mv, pv_board.pinned(), pv_board.checkers())
        {
            break;
        }
        pv_board.make_move(pv_mv);
        // Emit the move BEFORE testing for the repetition it creates: the
        // move is legal and is the engine's actual choice, so dropping it
        // loses information and, when a line repeats on its second move,
        // collapses the whole PV to a single ply.
        if !pv_str.is_empty() { pv_str.push(' '); }
        pv_str.push_str(&move_to_uci(pv_mv));
        pv_moves += 1;
        if seen_hashes.iter().filter(|&&h| h == pv_board.hash).count() >= 2 { break; }
        seen_hashes.push(pv_board.hash);
    }

    {
        while pv_moves < target_depth as usize + 5 {
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

    pv_str
}

fn emit_final_info(info: &SearchInfo, board: &Board, nodes: u64) {
    if info.silent || info.completed_depth <= 0 {
        return;
    }

    let elapsed = info.start_time.elapsed().as_millis() as u64;
    let nps = if elapsed > 0 { nodes * 1000 / elapsed } else { 0 };
    let score_str = crate::tt::format_uci_score(info.last_score);
    let pv_str = build_pv_string(info, board, info.completed_depth);
    if pv_str.is_empty() {
        println!(
            "info depth {} seldepth {} {} nodes {} nps {} time {} hashfull {} tbhits {}",
            info.completed_depth, info.sel_depth, score_str,
            nodes, nps, elapsed, info.tt.hashfull(), info.tb_hits
        );
    } else {
        println!(
            "info depth {} seldepth {} {} nodes {} nps {} time {} hashfull {} tbhits {} pv {}",
            info.completed_depth, info.sel_depth, score_str,
            nodes, nps, elapsed, info.tt.hashfull(), info.tb_hits, pv_str
        );
    }
}

/// Run iterative deepening search.
pub fn search(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits) -> Move {
    init_feature_flags();

    // Enable threat delta generation if we have a threat net
    // Decide threat REFRESH mode once per search, from the root piece count.
    // Both sides of the contract read the same flag — the generator on the next
    // line and the consumer in `ensure_computed`. Deciding per node would let
    // them disagree and replay from deltas that were never generated.
    {
        let pmax = THREAT_REFRESH_PIECE_MAX.load(Ordering::Relaxed);
        let pieces = board.occupied().count_ones() as i32;
        crate::threat_accum::REFRESH_MODE.store(pmax > 0 && pieces <= pmax, Ordering::Relaxed);
    }
    board.generate_threat_deltas = info.nnue_net.as_ref().is_some_and(|n| n.has_threats)
        && !crate::threat_accum::refresh_mode();

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
    info.last_flushed_nodes.set(0);
    // Note: global_nodes reset is done by callers (search_smp, bench) to avoid
    // clobbering helper thread contributions in SMP mode.
    info.sel_depth = 0;
    info.root_stm = board.side_to_move;

    // SNAP forensics: populate the traced line's per-ply hashes + moves from
    // CODA_TRACE_LINE (space-separated UCI moves, interpreted from THIS root).
    // One-shot per process (harness spawns a fresh engine per position).
    if info.trace_hashes.is_empty() {
        if let Ok(line) = std::env::var("CODA_TRACE_LINE") {
            if !line.trim().is_empty() {
                let mut wb = board.clone();
                for tok in line.split_whitespace() {
                    let legal = generate_legal_moves(&wb);
                    let mut found = NO_MOVE;
                    for i in 0..legal.len {
                        let m = legal.get(i);
                        if move_to_uci(m) == tok { found = m; break; }
                    }
                    if found == NO_MOVE {
                        eprintln!("TRACE error: move {} not legal on line; tracing disabled", tok);
                        info.trace_hashes.clear();
                        info.trace_line_mv.clear();
                        break;
                    }
                    info.trace_hashes.push(wb.hash);
                    info.trace_line_mv.push(found);
                    wb.make_move(found);
                }
                if !info.trace_hashes.is_empty() {
                    eprintln!("TRACE armed: {} plies", info.trace_hashes.len());
                }
            }
        }
    }

    // Age history tables (×0.80) to preserve useful move ordering from prior searches.
    // Killers and counter-moves are cleared (position-specific).
    // Correction history PERSISTS across `go` (cleared on ucinewgame only, in
    // uci.rs) — as it does in every engine surveyed. This only pays off given
    // the full-error corrhist updates: under a tight error pre-clamp the table
    // converges too slowly for persistence to matter, but with them each move
    // starts from a warm eval calibration.
    info.history.age(4, 5);
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
    // Reset the shared instant-reply gate inputs for THIS search. The UCI
    // thread also clears both before spawning (belt-and-braces there); this
    // reset covers non-UCI callers and reuse. Unlike the ponderhit deadline
    // trio (which the UCI thread WRITES on ponderhit and must not be
    // clobbered here), these are only ever written by the search thread —
    // a ponderhit racing this line reads 0/false, which conservatively
    // blocks the instant reply. Stale values from the PREVIOUS search must
    // never satisfy the gate (double-ponderhit guard).
    info.ponder_depth.store(0, std::sync::atomic::Ordering::Relaxed);
    info.ponder_stability.store(0, std::sync::atomic::Ordering::Relaxed);
    info.root_fail_low.store(false, std::sync::atomic::Ordering::Relaxed);
    if !info.silent {
        info.ph_fl_active.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    info.ph_fl_extensions = 0;
    info.reductions = [0; MAX_PLY + 1];
    info.excluded_move = [NO_MOVE; MAX_PLY + 1];
    info.moved_piece_stack = [0; MAX_PLY + 1];
    info.double_ext_count = [0; MAX_PLY + 1];
    info.cutoff_count = [0; MAX_PLY + 4];
    info.moved_to_stack = [0; MAX_PLY + 1];
    info.pv_table = [[NO_MOVE; MAX_PLY + 1]; MAX_PLY + 1];
    info.pv_len = [0; MAX_PLY + 1];
    // Clear TM state
    info.tm_prev_best = NO_MOVE;
    info.tm_prev_score = 0;
    info.tm_best_stable = 0;
    info.tm_best_move_changes = 0;
    info.tm_asp_fail_low = 0;
    info.tm_asp_fail_high = 0;
    info.tm_forced_state = ForcedState::None;
    info.tm_has_data = false;
    info.tm_dbg = TmDbg::default(); // else a factor-block-less move logs stale factors
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

    // SearchInfo persists across `go` commands.
    // Without explicit reset, stale soft_limit/hard_limit from a prior
    // `go wtime/btime` leak into subsequent `go movetime`, `go depth`, or
    // `go nodes` — the dynamic-TM gate then scales them and can break the
    // ID loop early. Zero all four unconditionally up-front; each branch
    // below sets only the ones it needs.
    info.time_limit = 0;
    info.soft_limit = 0;
    info.hard_limit = 0;
    info.soft_floor = 0;
    info.tm_no_inc = false;
    info.tm_baseline = 0;
    info.abs_deadline = 0;
    // tm_max_time belongs in this reset even though it is latent today (every
    // soft_limit setter also sets it) — it is one refactor away from being a
    // stale clamp.
    info.tm_max_time = 0;

    if limits.infinite {
        // Already zero above.
    } else if limits.movetime > 0 {
        info.time_limit = limits.movetime;
        // Respect caller-supplied minimum think time (ponderhit fresh-search uses
        // this to enforce the increment floor; plain `go movetime` callers leave
        // it at 0 so they get exactly the movetime they asked for).
        info.soft_floor = limits.movetime_floor.min(limits.movetime);
        // Absolute forfeit guard when a real clock was supplied (ponderhit
        // fresh-search sets abs_clock). Plain `go movetime` leaves abs_clock=0
        // (no clock concept) and is unaffected.
        if limits.abs_clock > 0 {
            const FORFEIT_MARGIN_MS: u64 = 50;
            let reserve = info.move_overhead + FORFEIT_MARGIN_MS;
            info.abs_deadline = limits.abs_clock.saturating_sub(reserve).max(1);
        }
    } else if our_time > 0 {
        let (soft, hard, max_time, soft_floor) =
            compute_tm_budgets(our_time, our_inc, limits.movestogo, info.move_overhead,
                               board.fullmove, ponder_enabled());
        info.soft_limit = soft;
        info.hard_limit = hard;
        info.tm_max_time = max_time;
        info.tm_our_inc = our_inc;
        info.tm_time_left = our_time.saturating_sub(info.move_overhead).max(1);
        info.soft_floor = soft_floor;
        // Apply ponder-miss min-think floor (set by uci.rs when the prior
        // search was an abandoned ponder). Caps at hard_limit so we never
        // floor above the absolute deadline. uci.rs is responsible for the
        // time-pressure safety cap (don't burn >2% of clock on a floor).
        if limits.min_think_ms > 0 {
            let floor = limits.min_think_ms.min(hard);
            info.soft_limit = info.soft_limit.max(floor);
            info.soft_floor = info.soft_floor.max(floor);
        }
        info.tm_no_inc = our_inc == 0 && limits.movestogo == 0;
        info.time_limit = hard; // search uses hard as absolute limit
        // ABSOLUTE forfeit guard: never spend past (clock - overhead - margin).
        // This is the hard ceiling that makes flagging impossible regardless of
        // the soft/hard budget or any overrun. Plain `go wtime/btime`: start_time
        // is the clock-start, so the guard is our_time minus overhead and a small
        // safety margin (covers the 4096-node check granularity + I/O latency).
        // The ponderhit path overrides this with a baseline-adjusted value.
        {
            const FORFEIT_MARGIN_MS: u64 = 50;
            let reserve = info.move_overhead + FORFEIT_MARGIN_MS;
            info.abs_deadline = our_time.saturating_sub(reserve).max(1);
        }
        info.tm_has_data = false;
        info.tm_best_stable = 0;
        info.tm_best_move_changes = 0;
        info.tm_asp_fail_low = 0;
        info.tm_asp_fail_high = 0;
    } else if !limits.infinite {
        // No clock info (e.g. `go depth N` or `go nodes N`). Already zeroed
        // above; explicit reset kept for clarity.
        info.time_limit = 0;
    }

    info.max_depth = if limits.depth > 0 { limits.depth } else { ROOT_DEPTH_MAX };
    info.max_nodes = limits.nodes;

    // TT generation is advanced by the entry-point caller (search_smp or
    // datagen), not here — see the generation note at the top of search_smp.

    let mut best_move = NO_MOVE;
    let mut prev_score = 0i32;

    // Stable PV snapshot. Updated only at the end of a *completed* iteration.
    // On a mid-iteration interrupt (should_stop fires inside negamax) we
    // restore from this so `best_move` and `pv_table[0]` stay consistent —
    // otherwise the bestmove emit can pair the prior iteration's best_move
    // with the current iteration's *partial* pv_table[0][1], producing a
    // ponder move that doesn't apply to the actual position-after-best
    // (this cost a forfeited game in live play).
    let mut stable_pv_len: usize = 0;
    let mut stable_pv: [Move; MAX_PLY + 1] = [NO_MOVE; MAX_PLY + 1];

    // The best root PV seen during the CURRENT iteration (captured after
    // each aspiration search, so a widening re-search or a mid-iteration abort
    // can't wipe a proven fail-high move). On abort we bank this deepest
    // completed root result instead of reverting to the previous iteration's
    // shallower `stable_pv`. pv_table[0] is internally paired (move + its
    // ponder), so banking it never reintroduces the old-move/new-ponder
    // mismatch described above. iter_pv_len is declared fresh each iteration; the array
    // is reused (only iter_pv[..iter_pv_len] is ever read).
    let mut iter_pv: [Move; MAX_PLY + 1] = [NO_MOVE; MAX_PLY + 1];

    // Get a fallback move and keep the legal list for final validation
    let root_legal = generate_legal_moves(board);
    if root_legal.len > 0 {
        best_move = root_legal.get(0);
        // Prefer the TT move (the previous search's best for this
        // position) over raw movegen order as the
        // emergency fallback — nearly free, and the move actually emitted
        // if abs_deadline expires before depth 1 completes.
        let tt_entry = info.tt.probe(board.hash);
        if tt_entry.hit && tt_entry.best_move != NO_MOVE {
            for i in 0..root_legal.len {
                if root_legal.get(i) == tt_entry.best_move {
                    best_move = tt_entry.best_move;
                    break;
                }
            }
        }
    }

    // Forced move: only one legal move, skip full search (just return it quickly).
    // Still search to depth 1 for a score to display, but cap time at 10ms.
    if root_legal.len == 1 && (info.soft_limit > 0 || info.time_limit > 0) {
        info.soft_limit = 10;
        info.hard_limit = 10;
        info.time_limit = 10;
        info.soft_floor = 0;
    }

    let effective_max = info.max_depth.min(ROOT_DEPTH_MAX);
    for depth in 1..=effective_max {
        if info.should_stop() { break; }
        info.root_depth = depth;
        info.sel_depth = 0; // P2.8: reset per iteration (consensus) — the info line then reports THIS iteration's seldepth, not a whole-search running max
        let mut iter_pv_len: usize = 0; // P1.4: fresh best-PV snapshot per iteration
        // Ponderhit check: stop between iterations (not mid-search) to avoid
        // partial TT entries and PV inconsistency. The engine completes the
        // current iteration fully before stopping, producing clean state.
        // Publish protocol: the ponderhit deadline
        // trio (hard=ponderhit_time, soft=ponderhit_soft, floor=
        // ponderhit_floor) is published by the UCI thread as
        //   floor (Relaxed) → soft (Relaxed) → hard (Release)
        // so hard is the publish flag. We load hard with Acquire FIRST and
        // read soft/floor ONLY after observing hard != 0 — the Acquire pairs
        // with the Release store, guaranteeing both Relaxed-stored fields
        // are visible. Do NOT key the arming off soft alone: on ARM that can
        // observe soft > 0 with a STALE floor == 0 (killing the stockpile
        // floor) or a stale hard == 0.
        let ph = info.ponderhit_time.load(std::sync::atomic::Ordering::Acquire);
        if ph > 0 {
            if info.start_time.elapsed().as_millis() as u64 >= ph {
                break;
            }
            // Post-ponderhit dynamic TM setup. If UCI just stored a soft deadline
            // AND the search has no soft_limit yet (i.e. this started as
            // `go ponder` with no time budget), arm dynamic TM from here onward.
            // Without this, the loop only honours the hard deadline and burns
            // the full ~5s at 60+2 even on stable positions where a 2-3s emit
            // would suffice. The floor (≈ inc-overhead) prevents instant-emit
            // when stability has been confidently held through ponder.
            // (soft stays 0 on the `go ponder movetime` path — no arming there.)
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
                // Init tm_max_time on the ponder path. The dynamic-TM consumer
                // caps via .min(tm_max_time), and nothing else sets it on
                // `go ponder` (both the up-front reset and the infinite branch
                // skip it), so it would carry 0 → min(soft*mult,0).max(1) = 1ms
                // → instant emit. This bites at inc>=500 (the dynamic-TM path);
                // the gate removal above handles the inc<500 path. Both are
                // needed and independent.
                info.tm_max_time = hard_remaining;
            }
        }
        let iter_start = std::time::Instant::now();

        let score;

        // Aspiration windows (skip for mate scores)
        if depth >= 4 && prev_score > -MATE_IN_MAX_PLY && prev_score < MATE_IN_MAX_PLY {
            // Eval-dependent aspiration delta: wider for extreme scores
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

                // Capture this search's root PV before the widening
                // re-search (or an abort) wipes it. A fail-high search populates
                // pv_table[0] with a proven-better move; a fail-low leaves
                // pv_len[0]==0 (no move raised alpha), so nothing to bank there.
                if info.pv_len[0] > 0 {
                    iter_pv_len = info.pv_len[0].min(iter_pv.len());
                    iter_pv[..iter_pv_len].copy_from_slice(&info.pv_table[0][..iter_pv_len]);
                }

                if info.should_stop() {
                    asp_result = result;
                    break;
                }

                if result <= alpha {
                    info.tm_asp_fail_low = info.tm_asp_fail_low.saturating_add(1);
                    info.stats.ts_asp_fail_low += 1;
                    // P1 (ponder instant-reply gate): root is failing low —
                    // publish it so a ponderhit arriving NOW does not
                    // instant-emit the destabilized pondered conclusion
                    // (SF pattern: fail-low revokes
                    // stopOnPonderhit). Cleared when the re-search resolves
                    // below. Relaxed: independent bool gate, no dependent
                    // data (see field doc).
                    info.root_fail_low.store(true, std::sync::atomic::Ordering::Relaxed);
                    // Fail-low extension, during-post-hit half: a root
                    // fail-low at a REAL search frontier (depth floor below)
                    // inflates the intended optimum SF-style and re-publishes
                    // the post-hit soft deadline in the from-go-ponder frame:
                    //   allowed(from go ponder) = intended_soft x (1 + 0.34 n)
                    // Long ponders (elapsed already past the inflated
                    // optimum) correctly get nothing -- SF stops promptly
                    // there too; short-ponder deep fail-lows get the >1s
                    // re-think tail (SF: 3.3% of post-hit moves; we had
                    // 0.0%). Main thread only; at most PH_FL_MAX_EXTENSIONS
                    // effective events (SF's min(2, fl)); every push
                    // saturates at ponderhit_abs -- the forfeit wall never
                    // moves.
                    if !info.silent
                        && info.ph_fl_extensions < PH_FL_MAX_EXTENSIONS
                        && info.root_depth >= PH_FL_MIN_DEPTH
                    {
                        let ph_hard = info.ponderhit_time.load(std::sync::atomic::Ordering::Acquire);
                        if ph_hard > 0 {
                            // v3: suspend the soft band until this fail-low
                            // resolves (cleared in the resolve branch below;
                            // a mid-fail-low abort leaves it true harmlessly
                            // — the search is ending anyway and the next
                            // search resets it).
                            info.ph_fl_active.store(true, std::sync::atomic::Ordering::Relaxed);
                            let isoft = info.ponderhit_isoft.load(std::sync::atomic::Ordering::Relaxed);
                            let abs = info.ponderhit_abs.load(std::sync::atomic::Ordering::Relaxed);
                            let clamp_abs = |v: u64| if abs > 0 { v.min(abs) } else { v };
                            let n = (info.ph_fl_extensions + 1) as u64;
                            let inflated = isoft.saturating_mul(100 + PH_FL_HARD_EXT_PCT * n) / 100;
                            let cur_soft = info.ponderhit_soft.load(std::sync::atomic::Ordering::Relaxed);
                            let new_soft = clamp_abs(inflated.max(cur_soft));
                            if new_soft > cur_soft {
                                info.ph_fl_extensions += 1;
                                let slice = info.ponderhit_floor.load(std::sync::atomic::Ordering::Relaxed)
                                    .max(MIN_POST_PONDERHIT_MS);
                                let new_hard = clamp_abs(ph_hard.max(new_soft.saturating_add(slice)));
                                info.tm_max_time = info.tm_max_time
                                    .max(new_hard.saturating_sub(info.tm_baseline));
                                // A1 publish order: soft Relaxed first, hard
                                // (the publish flag) Release last.
                                info.ponderhit_soft.store(new_soft, std::sync::atomic::Ordering::Relaxed);
                                info.ponderhit_time.store(new_hard, std::sync::atomic::Ordering::Release);
                                if TM_DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
                                    eprintln!(
                                        "PH_FL_EXT n={} depth={} isoft={}ms soft->{}ms hard->{}ms abs={}ms",
                                        info.ph_fl_extensions, info.root_depth, isoft, new_soft, new_hard, abs);
                                }
                            }
                        }
                    }
                    // Fail low: contract beta aggressively toward alpha, widen alpha
                    beta = (3 * alpha + 5 * beta) / 8;
                    alpha = (result - delta).max(-INFINITY);
                } else if result >= beta {
                    info.tm_asp_fail_high = info.tm_asp_fail_high.saturating_add(1);
                    info.stats.ts_asp_fail_high += 1;
                    // Fail high: contract alpha toward beta, widen beta
                    alpha = (5 * alpha + 3 * beta) / 8;
                    beta = (result + delta).min(INFINITY);
                    // Reduce depth for re-search (Alexandria/Midnight/Seer pattern)
                    asp_depth = (asp_depth - 1).max(1);
                } else {
                    // Window resolved — any earlier fail-low this iteration
                    // has been re-searched to a settled score; re-arm the
                    // instant reply (P1). A mid-fail-low abort leaves the
                    // flag true, which safely blocks instant replies until
                    // the next search resets it.
                    info.root_fail_low.store(false, std::sync::atomic::Ordering::Relaxed);
                    if !info.silent {
                        // FL-EXT v3: fail-low resolved — re-enable the
                        // post-hit soft band (the v2-extended deadline now
                        // applies at the next check).
                        info.ph_fl_active.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                    asp_result = result;
                    break;
                }

                delta += delta / 2;
            }

            score = asp_result;
        } else {
            score = negamax(board, info, -INFINITY, INFINITY, depth, 0, false);
            // Snapshot this single search's root PV (depth<4 path).
            if info.pv_len[0] > 0 {
                iter_pv_len = info.pv_len[0].min(iter_pv.len());
                iter_pv[..iter_pv_len].copy_from_slice(&info.pv_table[0][..iter_pv_len]);
            }
        }

        // Unified mid-iteration abort handling. Bank this iteration's
        // deepest completed root result (iter_pv — an internally-paired
        // move+ponder that a root move actually proved this iteration, deeper
        // than stable_pv). Only revert to the previous iteration's stable_pv
        // when this iteration completed no root move (iter_pv_len == 0).
        if info.should_stop() {
            if iter_pv_len > 0 {
                info.pv_len[0] = iter_pv_len;
                info.pv_table[0][..iter_pv_len].copy_from_slice(&iter_pv[..iter_pv_len]);
                // Adopt the banked move only if it validates against the legal list.
                let bm = iter_pv[0];
                let (bf, bt, bfl) = (move_from(bm), move_to(bm), move_flags(bm));
                for i in 0..root_legal.len {
                    let m = root_legal.get(i);
                    if move_from(m) == bf && move_to(m) == bt
                        && (!is_promotion(bm) || move_flags(m) == bfl)
                    {
                        best_move = m;
                        break;
                    }
                }
            } else if stable_pv_len > 0 {
                info.pv_len[0] = stable_pv_len;
                info.pv_table[0][..stable_pv_len].copy_from_slice(&stable_pv[..stable_pv_len]);
            }
            break;
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
        info.ponder_stability.store(info.tm_best_stable.max(0) as u64, std::sync::atomic::Ordering::Relaxed);

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
        // Exact node count: the global counter only updates at 4096-node
        // flushes, so on its own it reports 0 / 4096 / 8192... at shallow
        // depths — garbage for nodes-per-depth measurement. Add this thread's
        // unflushed delta.
        let global = info.global_nodes.load(Ordering::Relaxed)
            + (info.nodes - info.last_flushed_nodes.get());
        let nps = if elapsed > 0 { global * 1000 / elapsed } else { 0 };
        let score_str = crate::tt::format_uci_score(prev_score);

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
                let pv_mv = info.pv_table[0][i];
                // Legality guard: pv_table can carry a stale sibling-line move
                // (e.g. a fail-high / stable_pv restore that overran pv_len, or
                // an unguarded child-PV copy at the propagation site). Printing
                // it verbatim would emit an illegal PV move — a cutechess
                // "Illegal PV move" warning and a latent forfeit risk
                // (the same class of bug guarded at the TT-cutoff stuff
                // site). Stop the PV at the first move not legal in the
                // current pv_board, mirroring the TT-extension tail below.
                if pv_mv == NO_MOVE
                    || !crate::movepicker::is_pseudo_legal(&pv_board, pv_mv)
                    || !pv_board.is_legal(pv_mv, pv_board.pinned(), pv_board.checkers())
                {
                    break;
                }
                pv_board.make_move(pv_mv);
                // Emit the move BEFORE testing for the repetition it creates: the
                // move is legal and is the engine's actual choice, so dropping it
                // loses information and, when a line repeats on its second move,
                // collapses the whole PV to a single ply.
                if !pv_str.is_empty() { pv_str.push(' '); }
                pv_str.push_str(&move_to_uci(pv_mv));
                pv_moves += 1;
                if seen_hashes.iter().filter(|&&h| h == pv_board.hash).count() >= 2 { break; }
                seen_hashes.push(pv_board.hash);
            }

            // Extend with TT toward the same target the gate used to test
            // against. These were inconsistent: the gate required
            // `pv_moves < depth` but the loop then ran to `depth + 5`, so a
            // PV one move shorter than `depth` was extended by six while a PV
            // of exactly `depth` was not extended at all. That produced
            // alternating long/short PVs across iterations — visible in CCRL
            // broadcasts as every other line being truncated.
            {
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

        // MultiPV: primary line carries `multipv 1` only when MultiPV>1, so the
        // default (MultiPV=1) line is byte-identical to before.
        let mpv_tok = if info.multipv > 1 { "multipv 1 " } else { "" };
        if !info.silent {
            println!(
                "info depth {} seldepth {} {}{} nodes {} nps {} time {} hashfull {} tbhits {} pv {}",
                depth, info.sel_depth, mpv_tok, score_str,
                global, nps, elapsed,
                info.tt.hashfull(), info.tb_hits, pv_str
            );
        }

        // MultiPV secondary lines (analysis only). Save/restore the primary
        // line's pv so the rest of this iteration's TM/ponder logic is
        // unaffected, and the final bestmove stays the primary move.
        if info.multipv > 1 && !info.silent && !info.should_stop() {
            let saved_pv = info.pv_table[0];
            let saved_len = info.pv_len[0];
            info.root_ban.clear();
            if saved_len > 0 {
                info.root_ban.push(saved_pv[0]);
            }
            for pv_idx in 1..info.multipv {
                if info.should_stop() {
                    break;
                }
                let sc = negamax(board, info, -INFINITY, INFINITY, depth, 0, false);
                // A stop DURING this search makes negamax return 0 (the
                // convention at every call site). The pre-loop `should_stop`
                // check cannot catch that, and `pv_len` is still non-zero from
                // the aborted search — so the slot was emitted as a bogus
                // `score cp 0` carrying a full, plausible-looking PV. Any GUI
                // or kibitzer reading MultiPV saw a fabricated 0.00 line.
                // Mirror the post-recursion stop-check used elsewhere.
                if info.should_stop() {
                    break;
                }
                if info.pv_len[0] == 0 {
                    break;
                }
                // Build PV string from pv_table[0] via the SHARED guarded
                // helper — do NOT join pv_table[0] verbatim here. The legality
                // walk is dormant at the default MultiPV=1 (this loop is
                // `1..info.multipv`), which makes it easy to skip, but it is a
                // live illegal-PV emitter for anyone running MultiPV>1.
                // Printing an illegal PV move is a critical bug, so this logic
                // is shared, not re-derived.
                let line_pv = build_pv_string(info, board, depth);
                let line_score = crate::tt::format_uci_score(sc);
                let s_elapsed = info.start_time.elapsed().as_millis() as u64;
                let s_global = info.global_nodes.load(Ordering::Relaxed)
                    + (info.nodes - info.last_flushed_nodes.get());
                let s_nps = if s_elapsed > 0 { s_global * 1000 / s_elapsed } else { 0 };
                println!(
                    "info depth {} seldepth {} multipv {} {} nodes {} nps {} time {} hashfull {} tbhits {} pv {}",
                    depth, info.sel_depth, pv_idx + 1, line_score,
                    s_global, s_nps, s_elapsed, info.tt.hashfull(), info.tb_hits, line_pv
                );
                let next = info.pv_table[0][0];
                info.root_ban.push(next);
            }
            info.root_ban.clear();
            info.pv_table[0] = saved_pv;
            info.pv_len[0] = saved_len;
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
                if best_move == info.tm_prev_best {
                    info.tm_best_stable += 1;
                } else {
                    info.tm_best_stable = 0;
                    // Cumulative count of root best-move changes since
                    // search start. Drives an upward multiplier on tactically
                    // unstable positions (Stockfish's best-move-instability
                    // multiplier pattern).
                    info.tm_best_move_changes = info.tm_best_move_changes.saturating_add(1);
                    // Publish main's change into its own slot (thread 0) of the
                    // cross-thread bmc array (concept from SF). Read+reset in the TM block.
                    info.thread_bmc[0].fetch_add(1, Ordering::Release);
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

        // Forced-move detection. Once-per-search verification
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
        // TC gate: skip the detector when the floor already occupies ≥ 1/3 of
        // the soft budget. At high-inc TCs (e.g. 60+5 → floor 2.45s vs soft
        // 6.4s = 38%) the detector's downward multiplier can't push
        // adjusted_soft below the floor by a meaningful amount — verification
        // cost is paid but actual spend barely changes, measured at ~13 Elo
        // worse than not running it at all. At low-inc TCs the floor fraction
        // stays small and the detector pays back (~+2.8 Elo at LTC).
        //
        // Behaviour table:
        // 60+5: floor/soft = 0.38 → skip
        // 60+1: floor/soft = 0.14 → fire
        // LTC:  floor/soft = 0.00 → fire
        //
        // Additional gate: skip at NO-INC TCs. At 1+0, 3+0, 60+0, 180+0 live
        // play showed a time-forfeit regression — detector verification at
        // depth 3-5 costs 50-150ms per move, and with no increment to recover
        // it that overhead accumulates until the clock runs out.
        //
        // Key this on `tm_no_inc` (true only when our increment is exactly
        // zero), NOT on `info.soft_floor == 0`. The latter also fires at STC
        // 10+0.1, where `inc == overhead == 100ms` makes soft_floor 0 by
        // coincidence — that disabled the forced-move detector at STC for
        // about -3 Elo.
        let floor_dominates = info.soft_floor * 3 >= info.soft_limit;
        let no_inc = info.tm_no_inc;
        if info.tm_forced_state == ForcedState::None
            && depth >= 8
            && best_move != NO_MOVE
            && info.soft_limit > 0
            && !floor_dominates
            && !no_inc
            && !info.should_stop()
            && !is_mate_score(prev_score)
        {
            let forced_margin_weak = tp(&TM_FORCED_MARGIN_WEAK);
            let forced_margin_strong = tp(&TM_FORCED_MARGIN_STRONG);
            let margin = if depth >= 12 { forced_margin_weak } else { forced_margin_strong };
            let r_beta = (prev_score - margin).max(-MATE_SCORE + 1);
            // r_depth = (min(12, depth-1) - 1) / 2 — caps verification at depth 5.
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

        // Dynamic TM: a multiplicative factor product applied to the soft
        // budget, clamped to max_time — there is no separate cap. A bounded
        // product (~9.5× max) against a wide hard window (46% of clock, from
        // compute_tm_budgets) lets the factors express real variety without
        // overflowing; max_time (60% of clock) is the only single-move
        // ceiling. Two of the factors are beyond the standard set: the
        // cross-move score trend (#5) and cross-thread instability (#6).
        if info.soft_limit > 0 && depth >= 4 && !info.should_stop() {
            // Mate early-emit: if we've found a forced mate and the best move
            // has held for at least one further iteration, stop deepening —
            // more search cannot improve on a forced mate, and burning the
            // soft budget (and the post-ponderhit floor below) on a position
            // we've already solved just wastes clock and looks terrible (18.8s
            // to play a mate-in-1 at 60+10 under ponder, in one observed
            // game). Require stability >= 1 so a one-iteration mate flicker
            // that later flips doesn't cause a premature emit. Also sets the
            // floor to 0 so the stockpile-prevention sleep is skipped.
            //
            // The `prev_score > 0` gate matters: a sign-agnostic check also
            // fires when we are stably LOSING by force, stopping the search
            // instead of hunting longer defenses or swindle lines with the
            // budget we still have.
            if prev_score > 0 && is_mate_score(prev_score) && info.tm_best_stable >= 1 {
                info.soft_floor = 0;
                break;
            }
            // Factor 1: Stability multiplier, indexed by the consecutive-stable
            // count. Defaults (see the TM_STAB_* tunables):
            //   0 stable:  1.75× (uncertain, search more)
            //   1 stable:  1.22×
            //   2 stable:  0.90× (settling)
            //   3 stable:  0.80×
            //   4+ stable: 0.74× (confident, search less)
            // One table lookup covers what is often split into a stability
            // factor plus a separate best-move-change factor.
            //
            // The [0] entry is the sensitive one and its range is deliberately
            // capped below 2.6: an initial multiplier around 2.5 overshoots the
            // opening badly even with the phase damp applied (opening spend
            // ~37% of clock against a healthy ~21%, worth tens of Elo at
            // moderate-inc TCs). Keep it near the ceiling of what the opening
            // can afford, not at the value the middlegame would prefer.
            let stability_table: [f64; 5] = [
                tp(&TM_STAB_0_100) as f64 / 100.0,
                tp(&TM_STAB_1_100) as f64 / 100.0,
                tp(&TM_STAB_2_100) as f64 / 100.0,
                tp(&TM_STAB_3_100) as f64 / 100.0,
                tp(&TM_STAB_4_100) as f64 / 100.0,
            ];
            let stability_idx = (info.tm_best_stable as usize).min(4);
            let stability_multiplier = stability_table[stability_idx];

            // Factor 2: Aspiration fail-low bonus.
            // Formula: 1.0 + 0.34 × min(2, count), range [1.00, 1.68]
            //   0 fails: 1.00× (baseline)
            //   1 fail:  1.34×
            //   2+ fails: 1.68× (cap)
            // Captures the upward instability signal.
            let failed_low_multiplier =
                1.0 + (tp(&TM_FAIL_LOW_BONUS_1000) as f64 / 1000.0) * (info.tm_asp_fail_low.min(2) as f64);

            // Factor 3: Forced-move multiplier (position-intrinsic).
            //   Strong: 0.386× (alternative -400cp behind)
            //   Weak:   0.627× (alternative -170cp behind)
            //   None:   1.00×
            let forced_move_multiplier = match info.tm_forced_state {
                ForcedState::Strong => tp(&TM_FORCED_STRONG_1000) as f64 / 1000.0,
                ForcedState::Weak   => tp(&TM_FORCED_WEAK_1000) as f64 / 1000.0,
                ForcedState::None   => 1.0,
            };

            // Factor 4: Best-move subtree-size multiplier.
            // Formula: (1.62 - nodes_fraction) × 1.4, range ~[0.87, 2.27]
            //   nodes_fraction = best_move_nodes / total_nodes
            //   high fraction (>0.6): confident → reduce time
            //   low fraction (<0.3):  uncertain → increase time
            let mut subtree_frac = -1.0f64; // diagnostic only; -1 = not computed
            let subtree_size_multiplier = if depth > 9 && best_move != NO_MOVE {
                let best_nodes = info.root_move_nodes[root_move_index(best_move)];
                let total = info.nodes;
                if total > 0 {
                    let frac = best_nodes as f64 / total as f64;
                    subtree_frac = frac;
                    // C1: base tunable-ized and re-centered (see
                    // TM_SUBTREE_BASE_100 in tunables!). Floor 0.55 bounds the
                    // discount on total-consensus moves (frac -> 1).
                    let base = tp(&TM_SUBTREE_BASE_100) as f64 / 100.0;
                    ((base - frac) * (tp(&TM_SUBTREE_MULT_100) as f64 / 100.0)).max(0.55)
                } else {
                    1.0  // default when no node data
                }
            } else {
                1.0  // early depths: neutral
            };

            // Factor 5: Score-trend multiplier (falling-eval). The signal
            // `score_drop` (= tm_prev_score - prev_score, in cp; positive =
            // eval FELL this iteration) was already computed but discarded.
            // 5 of 10 surveyed top engines (SF fallingEval, Obsidian,
            // PlentyChess) feed it into the TM product:
            // give MORE time when the eval is falling (position worsening —
            // don't snap-move into trouble) and LESS when it's stable or
            // improving (calm/winning — move on). Shaped after a
            // clamp(0.8 + 0.05*(prev-cur)) form but in cp units. CENTERED AT 1.0
            // when drop==0, so the flat-eval common case (most moves) leaves
            // the baseline allocation untouched and no retune is required to
            // test direction. Range [0.80, 1.45].
            let score_trend_multiplier = {
                let drop = score_drop as f64;
                // Cross-MOVE deterioration: prev-`go` final score − current
                // running score (both our-side-to-move → sign-comparable).
                // Positive = worsening across the game horizon; add time. Gated
                // on a real previous move (sentinel) and non-decisive scores.
                // is_decisive (not is_mate_score): TB scores (~28600-28800) sit
                // below the mate band and would otherwise pin the factor at a
                // clamp rail. Ceiling
                // raised to 1.55 to give the combined term headroom in
                // worsening positions.
                let cross = if info.tm_cross_prev_score != i32::MIN
                    && !is_decisive(prev_score)
                    && !is_decisive(info.tm_cross_prev_score)
                {
                    (info.tm_cross_prev_score - prev_score) as f64
                } else {
                    0.0
                };
                // The 1.00 floor is deliberate: this factor may add time, never
                // subtract it. A sub-1.0 rail allocates up to 20% LESS time
                // when our own eval is RISING, and that is exactly the shape
                // that precedes blunders. Measured over 317,887 plies: plies
                // whose depth dipped >=3 ply against the local baseline had
                // 2.05x the eval swing into them, spent half the time (0.12s vs
                // 0.22s), and were 1.86x more likely to be followed by a
                // >=400cp reversal. The "more time when worsening" half of the
                // term is validated and stays.
                (1.0 + 0.0025 * drop + (tp(&CROSS_MOVE_TREND) as f64 * 1e-4) * cross)
                    .clamp(1.00, 1.55)
            };

            // Combined multiplier — the standard factors + score-trend + cross-thread.
            // Max product ~ 2.50 × 1.68 × 1.0 × 2.27 × 1.45 = 13.8×
            // Min product ~ 0.75 × 1.0  × 0.386 × 0.87 × 0.80 = 0.20×
            // Factor 6: cross-thread best-move instability (concept from SF, Threads>1
            // only). Sum this iteration's best-move changes across ALL threads,
            // normalize by thread count, and scale time UP when the pool is
            // collectively still churning — main may have momentarily settled
            // while helpers disagree, which its own stability table can't see.
            // Reset the per-thread slots after reading (per-iteration window).
            let cross_thread_instability = if info.num_threads > 1 {
                let n = info.num_threads;
                let mut total: u32 = 0;
                for slot in info.thread_bmc.iter().take(n) {
                    total = total.saturating_add(slot.swap(0, Ordering::AcqRel));
                }
                let base = tp(&TM_BMC_INSTAB_BASE) as f64 / 1000.0;
                let mult = tp(&TM_BMC_INSTAB_MULT) as f64 / 1000.0;
                base + mult * (total as f64) / (n as f64)
            } else {
                1.0
            };

            let mut multiplier = stability_multiplier
                * failed_low_multiplier
                * forced_move_multiplier
                * subtree_size_multiplier
                * score_trend_multiplier
                * cross_thread_instability;
            // No-inc clamp: factor product up to 6.5× at no-inc TCs blows
            // adjusted_soft past hard_time via iteration-overflow even with
            // the smaller no-inc opt baseline — observed at 3+0 as a run of a
            // dozen moves each spending exactly hard = 10% of clock. Cap the
            // multiplier so adjusted_soft stays well below hard at no-inc,
            // letting the soft check actually fire and giving per-move
            // variability instead of uniform hard-cap saturation.
            if info.tm_no_inc {
                multiplier = multiplier.min(2.5);
            } else {
                // Low-increment ceiling. When the increment is small RELATIVE
                // TO THE CLOCK it can't refill a run of deep moves, so cap the
                // factor product. The discriminator is
                // inc_cover = inc / (timeLeft/mtg): ~0.04 at 600+1 (capped),
                // ~0.24 at 10+0.1 and ~0.4 at 600+10 (both ~uncapped). The
                // ceiling is cmin at inc_cover->0, rising to cmax at
                // inc_cover >= TM_INC_COVER_REF/100. mtg here is
                // TM_DEFAULT_MTG (the inc-path sudden-death mtg).
                let base_move = (info.tm_time_left / tp(&TM_DEFAULT_MTG).max(2) as u64).max(1);
                let inc_cover = (info.tm_our_inc as f64) / (base_move as f64);
                let ref_cover = (tp(&TM_INC_COVER_REF) as f64 / 100.0).max(0.001);
                let inc_factor = (inc_cover / ref_cover).clamp(0.0, 1.0);
                let cmin = tp(&TM_MULT_CEIL_MIN_10X) as f64 / 10.0;
                let cmax = tp(&TM_MULT_CEIL_MAX_10X) as f64 / 10.0;
                let inc_ceiling = (cmin + (cmax - cmin) * inc_factor).max(1.0);
                multiplier = multiplier.min(inc_ceiling);
            }

            // adjusted_soft = soft × multiplier, clamped to max_time (the ONLY
            // cap).
            let adjusted_soft_raw = (info.soft_limit as f64 * multiplier) as u64;
            let adjusted_soft = adjusted_soft_raw.min(info.tm_max_time).max(1);
            // Phase-0 instrumentation: snapshot the factor values in force.
            // The last snapshot before the search stops is "the budget the
            // move was played under" — read back by the TMDebug summary line.
            if TM_DEBUG.load(Ordering::Relaxed) {
                info.tm_dbg = TmDbg {
                    stab: stability_multiplier,
                    fail_low: failed_low_multiplier,
                    forced: forced_move_multiplier,
                    subtree: subtree_size_multiplier,
                    trend: score_trend_multiplier,
                    frac: subtree_frac,
                    product: multiplier,
                    adjusted_soft,
                };
            }
            // Compatibility aliases for downstream code that references
            // `scale` / `max_adjusted`. `scale` retained for TMDebug output.
            // Subtract tm_baseline so soft is measured from the TM-start
            // moment, not search start. tm_baseline is 0 for normal `go`
            // (unchanged behaviour); set to elapsed-at-ponderhit when
            // post-ponderhit dynamic TM arms above.
            // Re-read the clock here rather than reusing the iteration-top
            // `elapsed` snapshot: that one predates the info-line print and any
            // forced-move verification (50-150ms), which lets an extra
            // iteration start past the soft budget on exactly the iteration the
            // detector fires.
            let elapsed_now = info.start_time.elapsed().as_millis() as u64;
            let elapsed_since_tm = elapsed_now.saturating_sub(info.tm_baseline);
            if elapsed_since_tm >= adjusted_soft {
                break;
            }

            // Next-iteration estimate: stop if next iteration would exceed time limit.
            // Use 2x last iteration time as estimate (exponential branching).
            // Check both hard_limit (normal) and ponderhit_time (after ponderhit).
            // Without this, ponder searches start arbitrarily deep iterations after
            // ponderhit, get stopped mid-search, and leave incomplete TT entries.
            let effective_hard = {
                // A1: Acquire — hard is the trio's publish flag (see the
                // load at the top of the ID loop).
                let ph = info.ponderhit_time.load(std::sync::atomic::Ordering::Acquire);
                if ph > 0 { ph } else { info.hard_limit }
            };
            if effective_hard > 0 {
                let iter_elapsed = iter_start.elapsed().as_millis() as u64;
                // Use elapsed_now, not the iteration-top `elapsed` snapshot: the
                // A4 fix re-read the clock for the soft check but this hard-window
                // estimate still used the stale value taken before the info print
                // and the 50-150ms forced-move verification, granting an
                // unaffordable next iteration that then dies mid-search. Also
                // stop outright once already past the hard budget (the old
                // `effective_hard > elapsed` guard let a new iteration start when
                // elapsed had already crossed hard).
                if elapsed_now >= effective_hard {
                    break;
                }
                if (effective_hard - elapsed_now) < 2 * iter_elapsed {
                    break;
                }
            }
        }
    }

    // Publish this move's final score for the NEXT `go`'s cross-move trend
    // term. `info.last_score` holds the last completed iteration's score;
    // sign-comparable across our consecutive moves (opponent moves between).
    // Publish ONLY from searches that produced a played game move: a
    // clock-managed search
    // (soft_limit > 0, which includes a `go ponder` converted by ponderhit
    // at the top of the ID loop), any search that saw a ponderhit
    // (ponderhit_time is cleared by the UCI thread before every spawn and
    // set only on a hit — covers a hit whose deadline expired before the
    // next iteration could arm soft_limit), or the ponderhit fresh-search
    // path (`go movetime` with a real clock attached, abs_clock > 0). A
    // pondered-and-MISSED search scores a sibling position we never
    // reached, and analysis `go`s (infinite / depth / nodes / bare
    // movetime) score arbitrary positions — publishing those trends the
    // next real move's budget against the wrong position, pinning the
    // factor at a clamp rail (reproduced: +55% opt for the whole move).
    // Not publishing keeps the PREVIOUS real move's score, which stays
    // position-correct across a miss.
    // Acquire pairs with the UCI thread's Release stores (P2 deadline trio
    // and the P1/low-time instant-stop markers). A hit racing this load can
    // read 0 and skip the publish — benign, see above.
    let ponderhit_seen =
        info.ponderhit_time.load(std::sync::atomic::Ordering::Acquire) > 0;
    if info.completed_depth >= 1
        && (info.soft_limit > 0
            || ponderhit_seen
            || (limits.movetime > 0 && limits.abs_clock > 0))
    {
        info.tm_cross_prev_score = info.last_score;
    }

    // Don't stockpile: if the ID loop finished below the soft_floor (e.g. all
    // iterations were TT hits in a repetitive endgame), wait out the rest of
    // the floor time before emitting. Prevents clock growth from instant emits
    // at 1s-inc bullet. Polls the EXTERNAL stop flag so
    // the UCI thread can still interrupt — `info.stop` cannot serve here
    // because the line below sets it ourselves (to halt helpers), and on the
    // ponderhit fresh-search path the floor can equal the entire remaining
    // clock, so an un-interruptible sleep would block `stop`/`quit` for that
    // long. Skip when there's no time budget (depth/node-limited search) or
    // when already stopped.
    //
    // Set the shared stop flag BEFORE the sleep so helper threads stop
    // searching immediately rather than burning CPU through the entire
    // stockpile-prevention window: otherwise they run until their own
    // hard_limit or until main unblocks, wasting tens to hundreds of ms of CPU
    // per ponderhit grace window at blitz+inc. The main thread already has its
    // best move here and is only waiting to emit.
    // Never sleep out the floor on a forced mate — we've solved the position
    // and should emit immediately (covers mate paths that bypass the dynamic-TM
    // loop above: depth<4 mate, single-legal-move). prev_score holds the last
    // completed-iteration score.
    if info.soft_floor > 0 && !is_mate_score(prev_score) && !info.stop.load(Ordering::Relaxed) {
        info.stop.store(true, Ordering::Relaxed);
        loop {
            // A2: GUI `stop`/`quit` aborts the floor wait. external_stop is
            // set only by the UCI thread (never by search internals), so it
            // unambiguously means "emit bestmove NOW".
            if info.external_stop.load(Ordering::Acquire) { break; }
            let elapsed = info.start_time.elapsed().as_millis() as u64;
            // Floor is a duration measured from tm_baseline (0 for normal
            // `go`; elapsed-at-ponderhit for post-ponderhit dynamic TM).
            let elapsed_since_tm = elapsed.saturating_sub(info.tm_baseline);
            if elapsed_since_tm >= info.soft_floor { break; }
            let remaining = info.soft_floor - elapsed_since_tm;
            std::thread::sleep(std::time::Duration::from_millis(remaining.min(25)));
        }
    }

    // TM diagnostic: one-line per-move summary of the TM signals that
    // fired during this search. Gated by UCI option TMDebug — default
    // off. Format is parseable: key=value space-separated, prefixed by
    // `info string tm-debug` so cutechess captures it.
    if TM_DEBUG.load(Ordering::Relaxed) {
        use std::io::Write;
        let total_elapsed = info.start_time.elapsed().as_millis() as u64;
        let elapsed_since_tm = total_elapsed.saturating_sub(info.tm_baseline);
        // Append to /tmp/coda_tm_debug.log — cutechess strips `info string`
        // from its own log, so writing to a file is the most reliable way
        // to collect per-move data across a gauntlet. File path is fixed
        // for simplicity; concurrent processes append safely (single
        // small line at a time, no fsync needed for diagnostics).
        let path = format!("/tmp/coda_tm_debug_{}.log", std::process::id());
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true).append(true).open(&path) {
            // Overshoot: how far past the in-force soft budget the move
            // actually ran (iteration-boundary quantization diagnostic).
            // Only meaningful when the factor block ever ran (adjusted_soft>0).
            let overshoot = if info.tm_dbg.adjusted_soft > 0 {
                elapsed_since_tm as i64 - info.tm_dbg.adjusted_soft as i64
            } else {
                0
            };
            let _ = writeln!(
                f,
                "tm-debug depth={} bestmove={} score={} \
                 elapsed={} elapsed_since_ph={} soft={} hard={} floor={} \
                 tm_baseline={} stab={} bmc={} asp_fl={} asp_fh={} forced={:?} \
                 cross_prev={} \
                 stabf={:.2} flf={:.2} forcedf={:.3} subf={:.2} subfrac={:.3} \
                 trendf={:.2} mult={:.2} adjsoft={} overshoot={}",
                info.completed_depth,
                move_to_uci(best_move),
                info.last_score,
                total_elapsed,
                elapsed_since_tm,
                info.soft_limit,
                info.hard_limit,
                info.soft_floor,
                info.tm_baseline,
                info.tm_best_stable,
                info.tm_best_move_changes,
                info.tm_asp_fail_low,
                info.tm_asp_fail_high,
                info.tm_forced_state,
                if info.tm_cross_prev_score == i32::MIN { "none".to_string() }
                else { info.tm_cross_prev_score.to_string() },
                info.tm_dbg.stab,
                info.tm_dbg.fail_low,
                info.tm_dbg.forced,
                info.tm_dbg.subtree,
                info.tm_dbg.frac,
                info.tm_dbg.trend,
                info.tm_dbg.product,
                info.tm_dbg.adjusted_soft,
                overshoot,
            );
        }
    }

    // Final info line at search end: the per-iteration print above fires only
    // at completed depths, so a mid-iteration stop (node limit, hard time)
    // under-reported total nodes by up to a full iteration (~29% observed at
    // `go nodes 15000`), corrupting any NPS/nodes-from-logs analysis. Emit the
    // true totals once before returning (SF prints the same final update).
    //
    // The PV must be INCLUDED here. Omitting it (on the assumption that GUIs
    // keep the last full-line PV) breaks broadcast parsers that render the PV
    // from the *latest* info line, which then show a short or empty PV on the
    // played move. Re-emit the current
    // (bestmove-consistent) PV so the last line is always complete. pv_table[0]
    // here holds the adopted final line (deepest completed, or the banked
    // partial on a mid-iteration abort); build_pv_string applies the same
    // legality guard as the per-iteration print.
    let global = info.global_nodes.load(Ordering::Relaxed)
        + (info.nodes - info.last_flushed_nodes.get());
    if !info.defer_final_info {
        emit_final_info(info, board, global);
    }

    best_move
}

/// SNAP-forensics hooks (CODA_TRACE_LINE). `trace_gate!` fires when a node ON
/// the traced line is about to discard the line's NEXT move via a named gate;
/// `trace_node!` fires when an on-line node itself is cut before its move
/// loop (or visited). Zero-cost when tracing is off (empty-vec check).
macro_rules! trace_gate {
    ($info:expr, $hash:expr, $ply:expr, $mv:expr, $gate:literal, $depth:expr, $mc:expr) => {
        if !$info.trace_hashes.is_empty() {
            let p = $ply as usize;
            if p < $info.trace_hashes.len()
                && $info.trace_hashes[p] == $hash
                && $info.trace_line_mv[p] == $mv
            {
                eprintln!("TRACE gate={} ply={} depth={} mc={}", $gate, $ply, $depth, $mc);
            }
        }
    };
}
macro_rules! trace_node {
    ($info:expr, $hash:expr, $ply:expr, $what:literal, $depth:expr) => {
        if !$info.trace_hashes.is_empty() {
            let p = $ply as usize;
            if p < $info.trace_hashes.len() && $info.trace_hashes[p] == $hash {
                eprintln!("TRACE node={} ply={} depth={}", $what, $ply, $depth);
            }
        }
    };
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

    // Reset PV length FIRST — before any early return below — so the parent's
    // PV propagation reads `pv_len[ply_u+1] == 0` for nodes that take a
    // short-circuit path (draw, MAX_PLY, mate-dist). Without this, a child
    // early-return leaves stale pv_len from a prior sibling at this ply, and
    // the parent copies illegal moves out of pv_table[ply_u+1]. Symptom was
    // "Illegal PV move" warnings from cutechess at root.
    if ply_u <= MAX_PLY {
        info.pv_len[ply_u] = 0;
    }

    // Drawn-position detection must run BEFORE the MAX_PLY guard. At ply
    // >= MAX_PLY in drawn-material or repetition positions, the fall-back
    // `apply_halfmove_scale(info.eval(board), halfmove)` returns a possibly-
    // nonzero scaled eval — eval doesn't know about insufficient material
    // or repetition (only halfmove via the scale itself handles 50mr).
    if ply > 0 {
        let draw_score: i32 = 0;
        if board.halfmove >= 100 {
            return draw_score;
        }
        if board.is_insufficient_material() {
            return draw_score;
        }
        if board.is_repetition_draw(ply) {
            return draw_score;
        }
    }

    // Guard against stack overflow — only reached for non-drawn positions
    // at ply >= MAX_PLY.
    if ply_u >= MAX_PLY {
        return apply_halfmove_scale(info.eval(board), board.halfmove);
    }

    // Leaf node — dispatch to quiescence at the TOP of negamax,
    // after the draw checks + MAX_PLY guard but BEFORE the interior preamble
    // (reductions reset, mate-distance pruning, TT prefetch, enemy_attacks /
    // xray computation, TB probe, TT probe). Dispatching later makes every
    // depth<=0 entry (~14% of calls) pay that whole preamble and re-run draw checks,
    // prefetch and a second TT probe inside quiescence, plus a duplicate
    // `info.nodes += 1` (~10% boundary node-count inflation that also leaked
    // into the TM node-fraction signal). SF dives to qsearch as the
    // first thing in search(). Quiescence independently does nodes++, seldepth,
    // stop/time and draw checks, so nothing is lost. Two deliberate semantic
    // deltas (both SF-consistent): boundary nodes no longer get the interior
    // TB probe, nor negamax mate-distance pruning (qsearch's TT cutoff at
    // depth >= -1 is a superset of the depth-0 requirement).
    if depth <= 0 {
        return quiescence(board, info, alpha, beta, ply);
    }

    // TREESTATS: interior-node entry, bucketed by entry depth (captured once
    // so cutoff/width counters below use the same bucket).
    let ts_bucket = depth.min(31) as usize;
    info.stats.nodes_by_depth[ts_bucket] += 1;
    trace_node!(info, board.hash, ply, "visit", depth);

    // Reset the reductions slot at node entry so NMP and any other
    // pre-move-loop child call reads "no prior reduction", not a sibling's
    // stale LMR value from an earlier visit to this ply.
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

    // (PV length already cleared at function entry, before early returns.)

    // Track seldepth
    if ply > info.sel_depth {
        info.sel_depth = ply;
    }

    // Check time periodically
    if info.nodes & 1023 == 0
        && info.should_stop() {
            return 0;
        }

    if info.stop.load(Ordering::Relaxed) {
        return 0;
    }

    info.nodes += 1;


    // (Draw detection lives above the MAX_PLY guard — see the note there.)

    // Syzygy tablebase probe at interior nodes.
    // Probe WDL when piece count is within TB range. Returns a score that
    // causes a cutoff, so the search doesn't waste time in solved endgames.
    // Only at non-root (ply > 0) and non-excluded (not in singular verification).
    //
    // tb_floor: Some(tb_score) when an in-window PV TB hit raised alpha.
    // Search must not return / store below this — TB is ground truth.
    let mut tb_floor: Option<i32> = None;
    // A centipawn RFP result cannot refute a proven TB loss. Track only nodes
    // with concrete tablebase provenance; a blanket loss-window guard was too
    // broad in testing.
    let mut tb_loss_rfp_guard = false;
    if ply > 0 && info.excluded_move[ply_u] == NO_MOVE {
        if let Some(ref tb) = info.syzygy {
            // SF SyzygyProbeDepth gate: at the maximum loaded piece count,
            // only probe when depth >= tb_probe_depth — the depth<gate
            // frontier is the most numerous, least-rewarding layer to probe
            // (a cutoff there saves only a tiny qsearch, but the probe pays
            // full FEN-roundtrip + table-decompression on a cache miss).
            // Below the max piece count we always probe (smaller tables,
            // deeper endgame, a cutoff prunes more).
            let pc = crate::bitboard::popcount(board.occupied()) as usize;
            let max_pc = tb.max_pieces();
            let tb_band_loss = is_loss(beta) && beta > -MATE_IN_MAX_PLY;
            if tb_band_loss && pc == max_pc && depth < info.tb_probe_depth {
                // The probe-depth gate skipped a position which is otherwise
                // covered by the loaded tables.
                tb_loss_rfp_guard = true;
            } else if tb_band_loss && pc == max_pc.saturating_add(1) {
                // A legal capture may take the next ply into the tables. Pseudo
                // attacks are intentionally conservative; a false positive
                // here only declines one shallow RFP cutoff.
                let us_attacks = board.attacks_by_color(board.side_to_move);
                let enemy_pieces = board.colors[flip_color(board.side_to_move) as usize];
                tb_loss_rfp_guard = us_attacks & enemy_pieces != 0;
            }
            if pc <= max_pc && (pc < max_pc || depth >= info.tb_probe_depth) {
                if let Some(wdl) = tb.probe_wdl(board) {
                    info.tb_hits += 1;
                    if wdl < -1 && tb_band_loss {
                        tb_loss_rfp_guard = true;
                    }
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

                    // Exact draw (wdl==0) and ambiguous cursed-win/blessed-loss
                    // (wdl==±1) are non-mate game-theoretic values — return
                    // directly even when in-window. Otherwise the in-window case
                    // below only raises alpha and keeps searching, letting
                    // NNUE/qsearch produce a non-TB score in a solved subtree:
                    // a fortress draw read as +150 steers the root off a real
                    // win, and a cursed win read as winning gets overpressed into
                    // the 50-move draw. Definite Win/Loss (|wdl|>1) fall through
                    // to the bound logic: rule-50 can make a TB win unrealizable,
                    // so they're only trusted as a lower/upper bound (SF pattern).
                    if (-1..=1).contains(&wdl) { return tb_score; }

                    if tb_score >= beta { return tb_score; }
                    if tb_score <= alpha { return tb_score; }
                    // Definite Win/Loss in window: tighten bounds AND remember
                    // the TB ground truth so the final TT store / return doesn't
                    // poison future probes with a sub-TB UPPER bound. (The
                    // ambiguous wdl ∈ {-1, 0, +1} cases now return directly
                    // above via f5d9809; this branch covers definite wdl ±2
                    // landing in a wide window.) Without the floor, if the
                    // local search returns best_score < tb_score the final
                    // flag computation stuffs UPPER at sub-TB best_score —
                    // contradicting TB ground truth on every future probe.
                    alpha = tb_score;
                    tb_floor = Some(tb_score);
                }
            }
        }
    }

    // Cuckoo cycle detection: proactive repetition avoidance (Stockfish/Berserk)
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

    // Prefetch the five correction-history rows corrected_eval will read
    // (~240 lines / a few hundred cycles from here on the common paths).
    // The tables total ~3MB (cont_corr alone 2.8MB) and are evicted by
    // NNUE weight traffic between nodes, so these reads otherwise miss.
    // All indices derive from board state available right now. Wasted on
    // TT-cutoff / in-check exits — measured +0.5% median cycles, 63/100
    // positive pairs, sign-test p=0.006 (same-binary toggle protocol).
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let stm = board.side_to_move as usize;
        unsafe {
            let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
            _mm_prefetch(&info.pawn_corr[stm][pawn_idx] as *const i32 as *const i8, _MM_HINT_T0);
            let wnp_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
            _mm_prefetch(&info.np_corr[stm][WHITE as usize][wnp_idx] as *const i32 as *const i8, _MM_HINT_T0);
            let bnp_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
            _mm_prefetch(&info.np_corr[stm][BLACK as usize][bnp_idx] as *const i32 as *const i8, _MM_HINT_T0);
            if let Some(last) = board.undo_stack.last() {
                if last.mv != NO_MOVE {
                    let trans_idx = ((board.hash ^ last.hash) as usize) & (CORR_HIST_SIZE - 1);
                    _mm_prefetch(&info.trans_corr[stm][trans_idx] as *const i32 as *const i8, _MM_HINT_T0);
                }
            }
            // cont_corr rows: same index derivation as cont_corr_value.
            if ply_u >= 2 {
                let cur_p = info.moved_piece_stack[ply_u - 1] as usize;
                let cur_t = info.moved_to_stack[ply_u - 1] as usize;
                if cur_p != 0 && cur_p < 13 && cur_t < 64 {
                    for off in [2usize, 4] {
                        if ply_u >= off {
                            let pp = info.moved_piece_stack[ply_u - off] as usize;
                            let pt = info.moved_to_stack[ply_u - off] as usize;
                            if pp != 0 && pp < crate::movepicker::CONT_PLANES && pt < 64 {
                                _mm_prefetch(&info.cont_corr[pp][pt][cur_p][cur_t] as *const i32 as *const i8, _MM_HINT_T0);
                            }
                        }
                    }
                }
            }
        }
    }
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
            // 50mr mate/TB downgrade now happens inside score_from_tt
            // (SF value_from_tt placement), so every consumer
            // below sees the sanitized score.
            let tt_score = score_from_tt(tt_entry.score, ply, board.halfmove);

            // Halfmove-gated TT cutoff: TT scores are stored without halfmove
            // context. Near the 50-move cliff a cached mate-in-N may be unreachable,
            // and a stored bound may be over/understated by the time we revisit.
            // Gate ALL return-from-TT paths (direct + bounds-narrow collapse +
            // near-miss + QS) on TT_CUTOFF_HALFMOVE_MAX. Window-narrowing is still
            // applied — it only biases the search, while returning a stale
            // tt_score is unsafe.
            let halfmove_ok = (board.halfmove as i32) < tp(&TT_CUTOFF_HALFMOVE_MAX);
            // Require +1 ply of TT depth for a fail-high (LOWER) cutoff, as
            // SF/Obsidian/PlentyChess do: fail-lows accept at tt_depth>=depth
            // but fail-highs need tt_depth>=depth+1. A symmetric `>= depth` is
            // notably more permissive on fail-highs.
            if tt_depth > depth - (tt_score <= beta) as i32 && FEAT_TT_CUTOFF.load(Ordering::Relaxed) {
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
                // The node-level `is_pv` was captured BEFORE mate-distance
                // pruning, which may since have collapsed a PV window to zero.
                // In that case the TT cutoff SHOULD fire (we're effectively at
                // a zero window) but the stale is_pv blocks it, so recompute
                // from the post-mate-dist window. alpha here is still
                // alpha_orig — TT narrowing happens after this check.
                let tt_cut_is_pv = beta - alpha > 1;
                // Child-consistency verification for DEEP cutoffs (concept
                // from SF, independently re-implemented): at depth>=7,
                // make the TT move, probe the child's entry, unmake; decline
                // the cutoff when the child's (negated) value contradicts the
                // cutoff direction — rejects stale/one-sided deep cutoffs.
                // Cost: one board-only make/unmake + one probe, deep cutoffs
                // only. Shallow cutoffs and the bounds-collapse path below stay
                // unverified (matching SF's single-site scope).
                if !tt_cut_is_pv && cut_node == score_above_beta && bound_matches
                    && halfmove_ok
                    && !tt_cutoff_child_disagrees(info, board, tt_move, tt_score, beta, depth, ply)
                {
                    info.stats.tt_cutoffs += 1;
                    if tt_cross_gen {
                        info.stats.tt_cross_gen_cutoffs += 1;
                    }
                    // Defence-in-depth: validate tt_move is fully legal before
                    // stuffing it into pv_table. This is a path that *could*
                    // plant an illegal move (hash collision, torn write
                    // surviving the XOR check) — empirically it never fires,
                    // but the cost is O(1) and the failure mode is an illegal
                    // PV move and a forfeited game.
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
                    // Read pieces from moved_piece_stack (set pre-move, so it
                    // holds the pre-promotion pawn), NOT board.piece_at(to)
                    // (post-move, reports the promoted piece). The write side
                    // — beta-cutoff bonuses — uses moved_piece_stack, and any
                    // asymmetry here lands the malus for promotion moves in the
                    // queen/rook bin, where reads never look.
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
                            if opp_gp > 0 && opp_gp < crate::movepicker::CONT_PLANES
                                && our_gp > 0 && our_gp < crate::movepicker::CONT_PLANES
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
                    return tt_score;
                }

                // Fall through: use TT bounds to narrow alpha/beta window at non-PV nodes.
                // Gated on halfmove_ok for the same reason the returns below are:
                // near the 50-move cliff the stored tt_score is untrustworthy. At a
                // zero-window node this narrowing can only ever fully collapse the window
                // (tt_score > alpha implies tt_score >= beta), and the collapse-return
                // below IS halfmove-gated — so without this gate, past the halfmove
                // threshold the node falls through with an inverted window
                // (alpha >= beta) and searches + TT-stores a degenerate full-depth
                // bound.
                if halfmove_ok {
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

                        // History bonus for TT cutoff: reinforce move ordering.
                        // Promotions count as NOISY here. Classifying non-capture
                        // promotions as quiet (asymmetric with tt_move_noisy)
                        // makes promo cutoffs write main-history bonuses into
                        // from/to cells that genuine quiets then read.
                        // Promos are ordered in the capture stage (MVV promo bonus
                        // + capt-hist empty slot), so the capture branch below is
                        // the one the read side actually consults.
                        let tt_piece = board.piece_at(move_from(tt_move));
                        let tt_is_cap = board.piece_type_at(move_to(tt_move)) != NO_PIECE_TYPE
                            || move_flags(tt_move) == FLAG_EN_PASSANT
                            || is_promotion(tt_move);
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
                        && !is_decisive(tt_score)
                    {
                        let w10 = tp(&TT_DAMP_TT_WEIGHT_10X);
                        return (w10 * tt_score + 10 * beta) / (w10 + 10);
                    }
                    return tt_score;
                }
            } else if tt_depth >= depth - 1
                && beta - alpha_orig == 1
                && !is_decisive(tt_score)
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

    // (Leaf-node quiescence dispatch happens at the top of negamax.)

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
    // revisited at hm=85 used the stale-scaled value — worth about -8 Elo
    // under aggressive `(100-hm)/100` scaling. TT stores raw_eval; every
    // consumer scales fresh.
    let mut static_eval = -INFINITY;
    let mut raw_eval = -INFINITY;
    let mut scaled_eval = -INFINITY;
    let mut improving = false;
    let mut tt_static_eval_hit = false;
    if !in_check {
        // Consumer threshold matches pack_data's static_eval clamp range
        // (-4095..4095). Stores that pass -INFINITY (in-check positions
        // where eval is undefined) get clamped to -4095; we reject that
        // value here and recompute. The legitimate-eval-at-exactly-(-4095)
        // false positive case (~-40 pawns) is rare enough that re-eval
        // is harmless.
        if FEAT_TT_STATIC_EVAL.load(Ordering::Relaxed) && tt_hit && tt_entry.static_eval > -4095 {
            raw_eval = tt_entry.static_eval;
            info.stats_tt_static_eval_hits += 1;
            tt_static_eval_hit = true;
        } else {
            raw_eval = info.eval(board);
            // Eval-only TT writeback: when we paid for an NNUE eval AND
            // there's no existing TT entry, seed the TT with static_eval
            // so later visits of this position (from different move
            // orders or ID re-searches) skip the NNUE call.
            //
            // Phase-2 NPS lever: reduces
            // evals/node (Coda 0.677).
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
        static_eval = if FEAT_CORRECTION.load(Ordering::Relaxed) { corrected_eval(info, board, scaled_eval, ply_u) } else { scaled_eval };
        if ply_u < MAX_PLY {
            info.static_evals[ply_u] = static_eval;
        }
        // Improving: our eval is better than 2 plies ago.
        //
        // When ply-2 was in-check, static_evals[ply-2] is -INFINITY (see the
        // else branch below), so a naive `static_eval > static_evals[ply-2]`
        // trivialises to true and improving fires on every post-check
        // comeback, inflating RFP/LMP/futility/LMR. Fall back to ply-4 when
        // ply-2 is unavailable (as SF does), and skip entirely when neither
        // ply is usable.
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

    // Clear the grandchild cutoff counter so the `cutoff_count[ply+1]`
    // read in LMR reflects only fail-highs under THIS node's subtree
    // (grandchild cutoff-counter reset, technique from SF).
    info.cutoff_count[ply_u + 2] = 0;

    // Detect if TT move is noisy. Captures, EP, AND promotions
    // (including non-capture promotions — they create a queen, so they are
    // tactically loud). This must stay symmetric with the other
    // quiet/noisy classifications of the TT move in this file.
    let tt_move_noisy = tt_move != NO_MOVE && {
        board.piece_type_at(move_to(tt_move)) != NO_PIECE_TYPE
            || move_flags(tt_move) == FLAG_EN_PASSANT
            || is_promotion(tt_move)
    };

    // Internal Iterative Reduction: reduce depth when no TT move exists.
    // Restricted to PV/cut nodes (Obsidian/Berserk/Stormphrax pattern).
    let is_pv = beta - alpha_orig > 1;

    // Threat square from null-move failure
    let mut threat_sq: i32 = -1;

    // Hindsight reduction: when parent was LMR-reduced and both sides
    // think the position is quiet, reduce depth further.
    // Gate on prior_reduction (Stockfish >= 2, Alexandria >= 1).
    let prior_reduction = if ply_u >= 1 { info.reductions[ply_u - 1] } else { 0 };
    if !in_check && ply >= 1 && depth >= tp10(&HINDSIGHT_MIN_DEPTH_10X) && ply_u >= 1
        && prior_reduction >= 2
        && info.static_evals[ply_u - 1] > -(MATE_IN_MAX_PLY)
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

    // Hindsight extension (a common cross-engine pattern — Stockfish,
    // Alexandria, Halogen, Stormphrax): mirror of the
    // reduction. When parent reduced aggressively (>=3) but the combined
    // eval shows position has worsened (eval_sum <= 0), extend +1 ply to
    // find the threat we missed. Non-PV only (PV already searched fully).
    if !in_check && ply >= 1 && ply_u >= 1
        && !is_pv
        && prior_reduction >= 3
        && info.static_evals[ply_u - 1] > -(MATE_IN_MAX_PLY)
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

    // RFP runs BEFORE NMP (consensus order: SF/Obsidian/Berserk all
    // run the free static prune first, so the null search only sees nodes
    // static pruning couldn't cut). The reorder is Elo-neutral on its own, but
    // it removes the mechanism that kills shallow NMP — NMP-first intercepts
    // free RFP cutoffs — which is what enables the min-depth de-gate below.
    if !in_check {
        // Razoring. 10/10 stronger engines have the
        // qsearch-verified non-PV form: when static eval is hopelessly below
        // alpha at shallow depth, drop to qsearch and trust its fail-low.
        // Runs before RFP (consensus order: razor -> RFP -> NMP).
        if !is_pv
            && ply > 0
            && depth <= tp10(&RAZOR_DEPTH_10X)
            && alpha.abs() < 2000
            && info.excluded_move[ply_u] == NO_MOVE
            && static_eval + tp(&RAZOR_MULT) * depth <= alpha
        {
            let v = quiescence(board, info, alpha, alpha + 1, ply);
            if v <= alpha {
                info.stats.razor_cutoffs += 1;
                return v;
            }
        }

        // Reverse Futility Pruning (Static Null Move Pruning) — pre-NMP site.
        // RFP TT quiet guard: skip RFP when TT has a quiet best move (Tucano/Weiss).
        // If we know a good quiet move exists, don't prune based on static eval alone.
        let tt_move_is_quiet = tt_move != NO_MOVE
            && board.piece_type_at(move_to(tt_move)) == NO_PIECE_TYPE
            && move_flags(tt_move) != FLAG_EN_PASSANT
            && !is_promotion(tt_move);
        // TB/mate guard: every peer skips RFP when eval is near mate/TB range.
        // Without this, RFP could cut a node where NNUE sees forced mate. (RFP audit RFP-3)
        if depth <= tp(&RFP_DEPTH) && ply > 0 && !tt_pv && !tt_move_is_quiet && info.excluded_move[ply_u] == NO_MOVE && FEAT_RFP.load(Ordering::Relaxed)
            && static_eval.abs() < MATE_SCORE - 200 {
            let mut margin = if improving { depth * tp(&RFP_MARGIN_IMP) } else { depth * tp(&RFP_MARGIN_NOIMP) };
            // Root-depth-aware relaxation: + depth*(root_depth-thresh)+ *coef/100.
            // Zero at STC (root_depth <= thresh); grows with both remaining
            // depth and how deep the overall search is, so deep RFP at LTC
            // demands much more confidence. One formula, one tunable set.
            margin += (depth * (info.root_depth - tp(&RFP_ROOT_THRESH)).max(0) * tp(&RFP_ROOT_COEF)) / 100;
            let deep_extra = (depth - tp10(&RFP_DEEP_KNEE_10X)).max(0);
            if deep_extra > 0 {
                margin += deep_extra * tp(&RFP_DEEP_LINEAR);
            }
            // Widen margin when opponent pawns attack our pieces (Minic/Berserk pattern)
            if has_pawn_threats { margin += margin / 3; }
            if static_eval - margin >= beta && !tb_loss_rfp_guard {
                trace_node!(info, board.hash, ply, "rfp_cut", depth);
                info.stats.rfp_cutoffs += 1;
                // RFP_AUDIT (diagnostic): null-verify this static cutoff with
                // the SAME R formula real NMP uses (sans post-capture +1), and
                // count rejections per depth. The cutoff is returned regardless
                // — behavior-preserving, measurement-only. Nested audits are
                // suppressed via rfp_audit_active (each verification subtree
                // contains its own RFP cutoffs). Skipped when a null move is
                // unsound/meaningless (pawn-only, consecutive null, mate beta).
                if RFP_AUDIT.load(Ordering::Relaxed)
                    && !info.rfp_audit_active
                    && stm_non_pawn != 0
                    && !prev_was_null
                    && beta.abs() < MATE_IN_MAX_PLY
                    && !info.stop.load(Ordering::Relaxed)
                {
                    let d_idx = depth.clamp(0, 23) as usize;
                    info.stats.rfp_audit_attempts[d_idx] += 1;
                    // Bucket this audited cutoff by the
                    // SPREAD of the five correction-source cp contributions
                    // (disagreement = low eval confidence). Computed only under
                    // RFP_AUDIT — zero production cost.
                    let var_bucket = {
                        let stm = board.side_to_move as usize;
                        let div = tp(&CORR_HIST_DIV) as i64;
                        let grain = tp(&CORR_HIST_GRAIN_T) as i64;
                        let scale = (div * grain).max(1);
                        let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
                        let c1 = info.pawn_corr[stm][pawn_idx] as i64 * tp(&CORR_W_PAWN) as i64 / scale;
                        let wnp = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
                        let c2 = info.np_corr[stm][WHITE as usize][wnp] as i64 * tp(&CORR_W_NP) as i64 / scale;
                        let bnp = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
                        let c3 = info.np_corr[stm][BLACK as usize][bnp] as i64 * tp(&CORR_W_NP) as i64 / scale;
                        let c4 = cont_corr_value(info, ply_u) * tp(&CORR_W_CONT) as i64 / scale;
                        let c5 = if let Some(last) = board.undo_stack.last() {
                            if last.mv != NO_MOVE {
                                let ti = ((board.hash ^ last.hash) as usize) & (CORR_HIST_SIZE - 1);
                                info.trans_corr[stm][ti] as i64 * tp(&CORR_W_TRANS) as i64 / scale
                            } else { 0 }
                        } else { 0 };
                        let mx = c1.max(c2).max(c3).max(c4).max(c5);
                        let mn = c1.min(c2).min(c3).min(c4).min(c5);
                        let spread = mx - mn;
                        if spread < 8 { 0 } else if spread < 24 { 1 } else { 2 }
                    };
                    info.stats.rfp_audit_var_attempts[var_bucket] += 1;
                    let mut r = tp10(&NMP_BASE_R_10X) + depth / tp10(&NMP_DEPTH_DIV_10X);
                    if static_eval > beta {
                        let eval_r = ((static_eval - beta) / tp(&NMP_EVAL_DIV)).min(tp10(&NMP_EVAL_MAX_10X));
                        r += eval_r;
                    }
                    if depth - r < 1 { r = depth - 1; }
                    info.rfp_audit_active = true;
                    board.make_null_move();
                    if let Some(acc) = &mut info.nnue_acc { acc.push(DirtyPiece::incremental(&[])); }
                    if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }
                    if ply_u <= MAX_PLY {
                        info.moved_piece_stack[ply_u] = 0;
                        info.moved_to_stack[ply_u] = 0;
                    }
                    let null_score = -negamax(board, info, -beta, -beta + 1, depth - r, ply + 1, !cut_node);
                    if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
                    if info.threat_stack.active { info.threat_stack.pop(); }
                    board.unmake_null_move();
                    info.rfp_audit_active = false;
                    if null_score < beta && !info.stop.load(Ordering::Relaxed) {
                        info.stats.rfp_audit_fp[d_idx] += 1;
                        info.stats.rfp_audit_var_fp[var_bucket] += 1;
                    }
                }
                return static_eval - margin;
            }
        }
    }


    let nmp_threat_margin =
        (king_zone_pressure - (tp10(&NMP_KING_ZONE_MAX_10X) - 1)).max(0) * 64
        + (any_threat_count - 2).max(0) * 64;

    if depth >= tp10(&NMP_MIN_DEPTH_10X) && !in_check && ply > 0 && stm_non_pawn != 0
        && beta - alpha == 1 && static_eval >= beta + nmp_threat_margin
        && !prev_was_null  // Prevent consecutive null moves
        && ply >= info.nmp_min_ply  // Ply barrier: verification subtree cannot re-trigger NMP (audit B1)
        && beta.abs() < MATE_IN_MAX_PLY  // Skip NMP for mate/TB scores
        && info.excluded_move[ply_u] == NO_MOVE  // Skip NMP during SE verification
        && cut_node  // cut-node gate: only attempt NMP at expected fail-high nodes (closes 30%->57% NMP cutoff-rate gap)
        && FEAT_NMP.load(Ordering::Relaxed)
    {
        info.stats.nmp_attempts += 1;
        // Adaptive reduction: scales with depth and eval margin above beta
        let mut r = tp10(&NMP_BASE_R_10X) + depth / tp10(&NMP_DEPTH_DIV_10X);
        // Reduce more after captures: opponent just captured, null move more likely to work.
        // NOT a cross-engine consensus: SF's R is flat, and Obsidian keys on
        // the CURRENT node's ttMoveNoisy instead (a shape that has failed four
        // separate SPRTs here). This term is kept purely on Coda's own
        // evidence — worth ~+3.5 Elo with a retune, and removing it costs.
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
        if let Some(acc) = &mut info.nnue_acc { acc.push(DirtyPiece::incremental(&[])); }
        if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }
        // Set the null sentinel on moved_piece_stack /
        // moved_to_stack at ply_u. Without this, the child at ply+1 reads
        // stale (piece, to) from a prior sibling move at this ply, feeding
        // cont-hist and LMR-history adjustment with
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
            let nmp_score = if is_decisive(null_score) { beta } else { null_score };

            // Verification search at high depths to guard against zugzwang
            if depth >= tp10(&NMP_VERIFY_DEPTH_10X) {
                info.stats.nmp_verify += 1;
                // Set ply barrier so NMP cannot fire again inside the verification
                // subtree. All peer engines do this (Alexandria: nmpPlies = ply + (depth-R)*2/3).
                // Without this, NMP can verify itself, defeating zugzwang detection.
                let old_nmp_min_ply = info.nmp_min_ply;
                info.nmp_min_ply = ply + 3 * (depth - r) / 4;
                // Verification re-searches current position (no move made), so ply stays same
                let v_score = negamax(board, info, beta - 1, beta, depth - r, ply, false);
                info.nmp_min_ply = old_nmp_min_ply;
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

    // SF one-liner: a static eval already at/above beta counts
    // as improving for the whole move loop, even if it's worse than 2 plies
    // ago — the node is beating its window. Placed after NMP/RFP (both read
    // the plain 2-ply definition, matching SF's ordering); upgrades LMP's
    // (2 - improving) divisor, ProbCut's margin and LMR's !improving bump
    // for the remainder of the node.
    if !in_check && static_eval >= beta {
        improving = true;
    }

    // IIR: moved after NMP so null search uses full depth, not IIR-reduced depth.
    // All 6 reference engines run NMP at full depth; IIR only applies to the
    // moves loop. Running IIR first silently reduces null depth by 1 at cut nodes.
    if depth >= tp10(&IIR_MIN_DEPTH_10X) && tt_move == NO_MOVE && !in_check && (is_pv || cut_node) && FEAT_IIR.load(Ordering::Relaxed) {
        depth -= 1;
    }

    // (RFP moved above NMP — see pre-NMP site.)

    // ProbCut: at moderate+ depths, if a shallow search of captures with
    // raised beta confirms the position is winning, prune the node.
    //
    // Two subtleties in this gate:
    // - !is_pv is required. SF/Obsidian/Berserk all restrict ProbCut to
    //   non-PV nodes; pruning a PV node on a raised-beta shallow search is too
    //   aggressive for a node whose score we need exactly.
    // - the "TT says no chance" skip must use the PLY-ADJUSTED score and gate
    //   on the bound type. A LOWER bound < probcut_beta means "score is AT
    //   LEAST X" — not "no chance at probcut_beta", since the true score can
    //   be much higher. Only UPPER/EXACT bounds are evidence of a ceiling.
    let probcut_margin = (tp(&PROBCUT_MARGIN)
        - (improving as i32) * tp(&PROBCUT_MARGIN_IMP))
        .max(1);
    let probcut_root_over = (info.root_depth - tp(&PROBCUT_ROOT_THRESH)).max(0);
    let probcut_fade_span = tp(&PROBCUT_ROOT_FADE_10X).max(10);
    let probcut_fade_num = (probcut_fade_span - 10 * probcut_root_over).clamp(0, probcut_fade_span);
    let probcut_beta = beta + probcut_margin
        + (tp(&PROBCUT_ROOT_MARGIN) * probcut_fade_num) / probcut_fade_span;
    let probcut_min_depth_10x = tp(&PROBCUT_MIN_DEPTH_10X)
        + (tp(&PROBCUT_ROOT_MIN_DEPTH_10X) * probcut_fade_num) / probcut_fade_span;
    let probcut_min_depth = (probcut_min_depth_10x + 5) / 10;
    let probcut_tt_noshot = if tt_hit && tt_entry.depth >= depth - tp(&PROBCUT_TT_DEPTH_SLACK) {
        let adj_score = score_from_tt(tt_entry.score, ply, board.halfmove);
        (tt_entry.flag == TT_FLAG_UPPER || tt_entry.flag == TT_FLAG_EXACT)
            && adj_score < probcut_beta
    } else {
        false
    };
    if !in_check && ply > 0 && !is_pv && depth >= probcut_min_depth
        && beta.abs() < MATE_IN_MAX_PLY  // skip for mate/TB scores
        && info.excluded_move[ply_u] == NO_MOVE  // skip during SE verification
        && !probcut_tt_noshot  // TT says no chance
        && king_zone_pressure < tp10(&PROBCUT_KING_ZONE_MAX_10X)  // A3: skip in high-threat positions
        && FEAT_PROBCUT.load(Ordering::Relaxed)
    {
        // SEE threshold: only consider captures that gain enough material
        let see_threshold = (probcut_beta - static_eval).max(0);
        // Improving-conditioned ProbCut depth (SF d6483505) —
        // bundled near-miss; tuned with the LMP/ProbCut margin cluster.
        let pc_depth = depth - 4 - improving as i32;
        let pc_tt_move = if tt_move_noisy
            && is_pseudo_legal(board, tt_move)
            && board.is_legal(tt_move, pinned, checkers)
            && see_ge(board, tt_move, see_threshold)
        {
            tt_move
        } else {
            NO_MOVE
        };
        let mut pc_picker = QMovePicker::new(board, pc_tt_move, false, &info.history, pinned, checkers);
        loop {
            let mv = pc_picker.next(board);
            if mv == NO_MOVE { break; }

            if !see_ge(board, mv, see_threshold) { continue; }

            let pc_moved_pt = board.piece_type_at(move_from(mv));
            // Colored mover, read before make_move — written to the cont-hist
            // context stack after the move so the ply+1 child reads THIS move
            // as its parent (see stack write below).
            let pc_moved_piece = board.piece_at(move_from(mv));
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
                    if info.threat_stack.active {
                        info.threat_stack.absorb_deltas(board);
                    }
            info.tt.prefetch(board.hash);

            // Record this move on the cont-hist context stack before recursing.
            // ProbCut was the only recursion site that skipped this; without it
            // the ply+1 child (and its beta-cutoff cont-hist writes) read a
            // stale sibling's [piece][to] slot, corrupting the shared table.
            // Overwritten by the main move loop's own write, so no restore
            // needed. Mirrors the main loop (piece_at(from) -> go_piece).
            if pc_moved_piece != NO_PIECE && ply_u <= MAX_PLY {
                info.moved_piece_stack[ply_u] = go_piece(pc_moved_piece) as u8;
                info.moved_to_stack[ply_u] = move_to(mv);
            }

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
                // Stockfish. Prior code stored `dampened` and
                // hardcoded tt_pv=false, losing both pruning information on
                // future probes and the PV stickiness used by LMR reduction
                // decisions. Return value is still dampened for normal
                // scores — score was verified at probcut_beta = beta+margin,
                // not beta. Decisive mate/TB scores are exact enough that
                // margin subtraction corrupts their distance/range; SF avoids
                // damped decisive ProbCut returns.
                // Stored depth = verification depth + 1 (the ProbCut move
                // itself). SF/Obsidian/Plenty all keep this invariant; the
                // old `depth - 3` overstated verification by 1 ply whenever
                // `improving` reduced pc_depth, and stored -1/0 for the
                // qsearch-only shallow case (SF stores 1 there).
                info.tt.store(
                    board.hash, pc_depth.max(0) + 1, score_to_tt(score, ply),
                    TT_FLAG_LOWER, mv, raw_eval, tt_pv,
                );
                if is_decisive(score) {
                    return score;
                }
                return score - (probcut_beta - beta);
            }
        }
    }

    if tt_static_eval_hit && depth >= 3 {
        info.materialize_tt_barrier(board);
    }

    // Continuation history lookup from search stack.
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
        MovePicker::new(board, tt_move, safe_ply, checkers, pinned, &info.history, prev_move, pawn_hist_ref, enemy_attacks, our_xray_blockers, &info.moved_piece_stack, &info.moved_to_stack)
    };
    picker.threat_sq = threat_sq;

    let mut best_move = NO_MOVE;
    let mut best_score = -INFINITY;
    let mut move_count = 0i32;
    // Count of how many moves at THIS node have raised alpha so far. Later
    // moves reduce more proportionally (alpha_raises reduction) — once
    // improving moves are found, the rest are progressively less likely to beat
    // them. Only fires at PV nodes (cut nodes break on first fail-high).
    let mut alpha_raise_count = 0i32;
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
    // re-running the gates. Note this is NOT bench-neutral despite reading
    // like a pure short-circuit — it moves the bench ~22%, and the mechanism
    // for that has never been localised.
    let mut skip_quiets = false;

    loop {
        let mv = picker.next(board);
        if mv == NO_MOVE { break; }

        // Skip excluded move (singular extension verification search)
        if mv == info.excluded_move[ply_u] {
            continue;
        }

        // MultiPV: at the root, skip moves already assigned to a higher PV
        // line. root_ban is empty except during MultiPV>1 secondary searches,
        // so single-PV play is byte-identical.
        if ply_u == 0 && !info.root_ban.is_empty() && info.root_ban.iter().any(|&m| m == mv) {
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
        info.stats.moves_searched += 1;

        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);

        // Check if capture BEFORE making the move
        let is_cap = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
        let is_promo = is_promotion(mv);

        if skip_quiets && !is_cap && !is_promo {
            trace_gate!(info, board.hash, ply, mv, "skip_quiets", depth, move_count);
            continue;
        }

        // Late Move Pruning (reordered FIRST, SF Step-14 order): at shallow
        // depths, skip late quiet moves by movecount BEFORE SEE/futility filter
        // them. Running LMP last (the prior Coda order) meant its count check
        // only saw SEE/futility survivors — a pre-filtered residual that made
        // count-pruning riskier and kept LMP_BASE blunt. SF/Berserk/
        // Obsidian all set skipQuiets before SEE/futility.
        // Formula: (LMP_BASE + depth²) / (2 - improving); check carve at depth<4.
        if ply > 0 && !in_check && depth >= 1 && depth <= tp10(&LMP_DEPTH_10X)
            && !is_cap && !is_promo
            && !is_loss(best_score)
            && beta < MATE_IN_MAX_PLY  // forced-win guard: don't count-prune quiets while proving a win
            && FEAT_LMP.load(Ordering::Relaxed)
        {
            let mut lmp_limit = (tp10(&LMP_BASE_10X) + depth * depth) / (2 - improving as i32);
            // Predictive margin dimension: a static eval already far below alpha
            // is the best in-node signal that this will fail low, so spend fewer
            // quiets on it. Guarded on static_eval being real (it is -INFINITY
            // in check, though !in_check above already excludes that).
            if static_eval > -INFINITY && alpha - static_eval >= tp(&LMP_MARGIN_THRESH) {
                lmp_limit = (lmp_limit * tp(&LMP_MARGIN_PCT) / 100).max(1);
            }
            // The gives_direct_check carve sits inside the movecount test — only
            // pay the check-detection call when the count prune would actually
            // fire (node-count identical).
            if move_count > lmp_limit && (depth >= 4 || !board.gives_direct_check(mv)) {
                trace_gate!(info, board.hash, ply, mv, "lmp", depth, move_count);
                info.stats.lmp_prunes += 1;
                skip_quiets = true;
                picker.skip_remaining_quiets();
                continue;
            }
        }

        // SEE capture pruning: at shallow depths, prune captures that lose
        // material. SF-shaped margin: base depth*MULT plus a capture-history
        // relaxation so historically-good captures (cutoff producers) survive a
        // lower base. Prune if SEE < -margin.
        if is_cap && ply > 0 && !in_check && depth <= tp10(&SEE_CAP_DEPTH_10X)
            && mv != tt_move && !is_loss(best_score)
            && FEAT_SEE_PRUNE.load(Ordering::Relaxed)
        {
            let cap_ch = crate::movepicker::capt_hist_score_static(board, &info.history, mv);
            let cap_margin = (depth * tp(&SEE_CAP_MULT) + cap_ch * tp(&SEE_CAP_HIST) / 1024).max(0);
            if !see_ge(board, mv, -cap_margin) {
                trace_gate!(info, board.hash, ply, mv, "see_cap", depth, move_count);
                continue;
            }
        }

        // Estimated LMR depth for pre-MakeMove pruning (SEE quiet, futility).
        // Computed once and shared — no depth ceiling; at high depths lmr_d
        // collapses to 1, so thresholds naturally become permissive.
        let lmr_d = if move_count > 1 && depth >= 2 {
            // Table is centi-ply; gates want integer plies (floor = old value).
            let r = lmr_reduction((depth as usize).min(63) as i32, (move_count as usize).min(63) as i32) / LMR_SCALE;
            if r > 0 { (depth - r).max(1) } else { depth }
        } else {
            depth
        };

        // Futility pruning. Deliberately ABOVE SEE-quiet so the cheap static
        // prune fires first and SEE-quiet only runs see_ge on the survivors;
        // 5/6 reference engines order it this way.
        // Skip quiet moves when static eval + margin is below alpha. Uses shared
        // lmr_d for both gate and margin.
        if ply > 0 && static_eval > -INFINITY && !in_check
            && !is_cap && !is_promo
            && !is_loss(best_score)
            && beta < MATE_IN_MAX_PLY  // forced-win guard: don't futility-prune quiets while proving a win
            && FEAT_FUTILITY.load(Ordering::Relaxed)
            && lmr_d <= tp(&FUT_LMR_DEPTH)
        {
            let main_hist = info.history.main_score(from, to, enemy_attacks);
            let hist_adj = main_hist / 128;
            let threats_adj = any_threat_count * tp(&FUT_THREATS_MARGIN);
            let futility_value = static_eval + tp(&FUT_BASE) + lmr_d * tp(&FUT_PER_DEPTH) + hist_adj + threats_adj;
            // Direct-check carve-out + strong-history exemption (Igel #410).
            if futility_value <= alpha && main_hist < tp(&FUT_HIST_EXEMPT) && !board.gives_direct_check(mv) {
                trace_gate!(info, board.hash, ply, mv, "futility", depth, move_count);
                info.stats.futility_prunes += 1;
                skip_quiets = true;
                picker.skip_remaining_quiets();
                continue;
            }
        }

        // SEE quiet pruning: prune quiet moves landing on attacked squares.
        // Use lmrDepth² scaling (matching Stockfish/Berserk/Obsidian).
        if ply > 0 && !in_check
            && !is_cap && !is_promo
            && mv != tt_move
            && !is_loss(best_score)
            && beta < MATE_IN_MAX_PLY  // forced-win guard: don't SEE-prune quiets while proving a win
            && FEAT_SEE_PRUNE.load(Ordering::Relaxed)
        {
            let see_quiet_threshold = -tp(&SEE_QUIET_MULT) * lmr_d * lmr_d;
            if !see_ge(board, mv, see_quiet_threshold) {
                trace_gate!(info, board.hash, ply, mv, "see_quiet", depth, move_count);
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
            // Deliberately NO !in_check gate. None of SF/Obsidian/Berserk/
            // Stormphrax gate SE on check, and gating it means a deep in-check
            // node's TT move (often the single forced evasion — maximally
            // singular) is never extended and gets no multicut or negative
            // extension either. Mechanically safe:
            // the SE path reads no static_eval (-INFINITY in check); the
            // correction_value margin input is position-keyed.
            && info.excluded_move[ply_u] == NO_MOVE
            && tt_hit
            && tt_entry.flag != TT_FLAG_UPPER
            && tt_entry.depth >= depth - tp(&SE_TT_DEPTH_SLACK)
            && FEAT_SINGULAR.load(Ordering::Relaxed)
        {
            // 50mr downgrade applies here too (SF: singular ttValue
            // is value_from_tt output). A downgraded mate lands in the TB
            // band and is still filtered by the !is_decisive gate below, so
            // the old over-extension concern (downgraded scores sneaking past
            // a mate-only < MATE_IN_MAX_PLY check) no longer applies.
            let tt_score_local = score_from_tt(tt_entry.score, ply, board.halfmove);

            // Skip SE for mate scores (margin comparison meaningless)
            if !is_decisive(tt_score_local) {
                // xray bonus: if TT move uncovers our slider's attack on enemy
                // (from-square ∈ our_xray_blockers), subtract this from
                // singular_beta → WIDER SE margin → STRICTER singularity →
                // FEWER extensions on these discovered-attack TT moves (the
                // margin comment at the tunable def has the full rationale;
                // the effect is the opposite of the old "more extensions" note).
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
                    // Return singular_score (SF pattern) — tighter score
                    // for downstream TT propagation than singular_beta floor.
                    // EXCEPT decisive scores: singular_score is fail-soft from a
                    // reduced (depth-1)/2 search with the TT move EXCLUDED — a
                    // mate/TB score from it is unproven at this node's depth and
                    // would be TT-stored at full depth as LOWER. SF gates with
                    // !is_decisive and falls through; Obsidian/Berserk return
                    // singularBeta. Suppressing multicut entirely in mate
                    // shapes tested WORSE here, so keep FIRING and fix only the
                    // returned value.
                    info.stats.multicut += 1;
                    if is_decisive(singular_score) {
                        return singular_beta;
                    }
                    return singular_score;
                }

                if singular_score < singular_beta {
                    // TT move is singular — no competitive alternatives.
                    //
                    // SF-pattern additive extensions with PV/quiet/
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
                    let corr_abs = correction_value(info, board, ply_u).abs();
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
                    // Consensus: -3 non-PV (SF/Obsidian)
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

        // There is deliberately NO history-based pruning here. It is standard
        // in SF/Obsidian, but three attempts — Coda's own form, SF's form with
        // SF's constants, and SF's form with SPSA-tuned constants — each lost
        // 7-9 Elo. The signal history carries in Coda is already spent in move
        // ordering and in the LMR/futility depth adjustments; pruning on it
        // again double-counts. Re-add only with a contextual (not raw-threshold)
        // formulation.

        // (Futility pruning runs above SEE-quiet; Late Move Pruning runs
        // earlier still, right after the skip_quiets check — SF Step-14 order.)

        // Bad noisy pruning: skip losing captures when eval is far below alpha.
        // Applied before MakeMove. Direct-check carve-out: don't prune moves
        // that give direct check.
        if FEAT_BAD_NOISY.load(Ordering::Relaxed) && is_cap && !in_check && ply > 0 && depth <= tp10(&BAD_NOISY_DEPTH_10X) && mv != tt_move
            && !is_promo && !is_loss(best_score)
            && static_eval > -INFINITY && static_eval + depth * tp(&BAD_NOISY_MARGIN) <= alpha
            && !see_ge(board, mv, 0)
            && !board.gives_direct_check(mv)
        {
            trace_gate!(info, board.hash, ply, mv, "bad_noisy", depth, move_count);
            continue;
        }

        // Pre-make TT prefetch: issue the child-bucket fetch
        // BEFORE build_dirty_piece + NNUE push + make_move + threat absorb, so
        // ~all of that work overlaps the DRAM latency (vs the old post-make
        // prefetch, which had only the short pre-probe window). key_after is an
        // approximate hash (exact for the common cases; see board.rs) —
        // prefetch-only; the real probe below uses the true post-make hash.
        trace_gate!(info, board.hash, ply, mv, "searched", depth, move_count);
        info.tt.prefetch(board.key_after(mv));

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
        if info.threat_stack.active {
            info.threat_stack.absorb_deltas(board);
        }

        // Check if move gives check (opponent is now in check after make_move)
        let gives_check = board.in_check();

        let extension = 0;
        // `extension` is 0: there is no promotion-imminent (7th-rank pawn
        // push) extension here, and none of the 18 stronger engines surveyed
        // has one either. Coda carried one twice and it measured as noise both
        // times — removing it was worth Elo, as was removing the analogous
        // recapture extension.

        // FEAT_EXTENSIONS ablation (NO_EXTENSIONS=1). The flag used to gate the
        // recapture and 7th-rank promotion extensions; both were removed
        // (10a4ed7, 56f07e0) and took the flag's only read sites with them,
        // leaving it silently inert. Re-wired here to the extension mechanism
        // that survives: suppress POSITIVE singular/double extensions only.
        // Negative `singular_extension` values are reductions, and the
        // multi-cut above is a pruning device — both stay on, so the flag
        // isolates exactly what its name claims. The singular_ext/double_ext
        // counters above still count DETECTIONS, not applications.
        if singular_extension > 0 && !FEAT_EXTENSIONS.load(Ordering::Relaxed) {
            singular_extension = 0;
        }

        let mut new_depth = depth - 1 + extension + singular_extension;

        // Propagate double extension counter to child
        if ply_u < MAX_PLY {
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
        // Store piece/to/captured for history updates after search.
        // Non-capture promotions tracked with ct=0 (empty slot) — they train
        // capture history, so they must also be malus-eligible or the empty
        // slot only ever inflates.
        if (is_cap || is_promo) && n_captures_tried < 32
            && moved_piece != NO_PIECE {
                let ct = if flags == FLAG_EN_PASSANT {
                    captured_type(PAWN)
                } else if captured_pt != NO_PIECE_TYPE {
                    captured_type(captured_pt)
                } else {
                    0
                };
                captures_tried[n_captures_tried] = (go_piece(moved_piece) as u8, to, ct as u8);
                n_captures_tried += 1;
            }

        // Late Move Reductions (LMR) + Principal Variation Search (PVS)
        let mut reduction = 0i32;
        // Endgame gate: skip LMR in low-piece-count positions where
        // mate-completing king-restriction moves would be over-reduced.
        let endgame_threshold = tp10(&LMR_ENDGAME_PIECES_10X) as u32;
        let is_endgame_skip = endgame_threshold > 0
            && crate::bitboard::popcount(board.occupied()) <= endgame_threshold;
        // Explicit `move_count > 1 && mv != tt_move` guards (defensive,
        // bench-neutral). Currently safe via LMR_TABLE zero-init at
        // depth<3 / move<3, but if the table is ever populated differently
        // this would start reducing the TT move. Mirrors the capture-LMR
        // gate at line ~3768 for symmetry.
        if !in_check && !is_cap && !is_promo && !is_endgame_skip
            && move_count > 1 && mv != tt_move
            && FEAT_LMR.load(Ordering::Relaxed) {
            let d = (depth as usize).min(63);
            let m = (move_count as usize).min(63);
            reduction = lmr_reduction(d as i32, m as i32); // CENTI-PLY from here to the floor below

            if reduction >= LMR_SCALE {
                // Reduce less at PV nodes where accuracy matters most
                if beta - alpha > 1 {
                    reduction -= LMR_SCALE;
                }

                // Reduce more at expected cut nodes. A flat +1 at every non-PV
                // node — not distinguishing expected cut nodes (fail-high) from
                // all-nodes — leaves Elo on the table; SF reduces ~+4 plies
                // specifically at cutNode, +1 more with no TT move, and less at
                // all-nodes. So: cut nodes get the tunable LMR_CUTNODE_BUMP
                // (+1 if no TT move), all-nodes keep +1.
                if !is_pv {
                    reduction += if cut_node {
                        tp(&LMR_CUTNODE_BUMP_CENTI) + ((tt_move == NO_MOVE) as i32) * LMR_SCALE
                    } else {
                        // Depth-decaying all-node inflation (SF shape): the
                        // flat +1 ply reduced the deep tree as hard as the
                        // shallow one; proportional-and-decaying reduces more
                        // shallow, less deep. `reduction` here is the raw
                        // table value (PV subtraction can't have fired at an
                        // all-node), matching SF applying it to r.
                        reduction * tp(&LMR_ALLNODE_DECAY_NUM) / (256 * depth + 285)
                    };
                }

                // Reduce later moves more once this node has already raised
                // alpha (alpha_raises reduction). Fixed-point ×10. At cut nodes this is
                // 0 (they break on the first fail-high before alpha rises), so it
                // only sharpens late-move reduction at PV nodes where several
                // improving moves have already been found.
                // CONTINUOUS (commit 2): x10 = /10 * LMR_SCALE without the floor.
                reduction += alpha_raise_count * LMR_ALPHA_RAISE_10X.load(Ordering::Relaxed) * 10;

                // Reduce less when the position is improving
                if improving {
                    reduction -= LMR_SCALE;
                }

                // Reduce more when TT move is a capture
                if tt_move_noisy {
                    reduction += LMR_SCALE;
                }

                // Reduce more when opponent has few non-pawn pieces (simpler position)
                // Note: board is post-make_move, so side_to_move IS the opponent
                let opp_non_pawn = board.colors[board.side_to_move as usize]
                    & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
                if popcount(opp_non_pawn) < 3 {
                    reduction += LMR_SCALE;
                }

                // Reduce less when moving a piece away from a pawn-attacked square
                if enemy_attacks & (1u64 << from) != 0 {
                    reduction -= LMR_SCALE;
                }

                // Reduce less when move gives check (Obsidian/Alexandria/Berserk pattern)
                if gives_check {
                    reduction -= LMR_SCALE;
                }

                // Reduce less when position was previously a PV node (Alexandria/Obsidian/Seer pattern).
                // Sticky: once a position is searched as PV, tt_pv stays set even at non-PV nodes.
                if tt_pv {
                    reduction -= LMR_SCALE;
                }

                // LMR correction battery.
                // (a) Winning beta: the window is already in the proven-win band,
                //     move precision matters less — reduce more.
                if is_win(beta) {
                    reduction += tp(&LMR_WINBETA_CENTI);
                }
                if tt_hit && tt_entry.flag != TT_FLAG_NONE {
                    let tt_score_node = score_from_tt(tt_entry.score, ply, board.halfmove);
                    // (b) TT already says this node can't beat alpha.
                    if tt_score_node <= alpha {
                        reduction += tp(&LMR_TTALPHA_CENTI);
                    }
                }
                // (d) Quiet expectation gap: eval far below alpha → this node is
                //     underperforming its window, reduce late quiets more (and
                //     slightly less when eval already exceeds alpha). Continuous,
                //     ~0.32 centi/cp at default. static_eval is valid here (the
                //     quiet-LMR block is gated on !in_check).
                reduction += tp(&LMR_EXPECT_MULT) * (alpha - static_eval).clamp(-65, 91) / 128;

                // cutoff_count: the child ply keeps failing high under
                // this node — refutations come easy down there, so late moves
                // need less depth to refute. SF consensus term.
                if info.cutoff_count[ply_u + 1] > 2 {
                    reduction += tp(&LMR_CUTOFF_CNT_CENTI);
                    if !is_pv && !cut_node {
                        reduction += tp(&LMR_CUTOFF_ALLNODE_CENTI);
                    }
                }

                // Continuous history adjustment: good history reduces less, bad more
                // Uses main history + ply-1 + ply-2 continuation history (consensus).
                // Ply-2 weighted at half to avoid over-scaling the total.
                // SF weights main history 2× vs continuation history.
                let mut hist_score = info.history.main_score(from, to, enemy_attacks) * 2;
                if moved_piece != NO_PIECE {
                    let gp = go_piece(moved_piece);
                    if prev_piece_for_cont != 0 {
                        hist_score += info.history.cont_hist[prev_piece_for_cont][prev_to_for_cont as usize][gp][to as usize] as i32;
                    }
                    if prev2_piece_for_cont != 0 {
                        hist_score += info.history.cont_hist[prev2_piece_for_cont][prev2_to_for_cont as usize][gp][to as usize] as i32 / 2;
                    }
                    // Pawn history: pawn-structure-aware move quality (SF/Alexandria pattern).
                    // Uses the node-level ph_idx (parent pawn hash) — this code runs
                    // post-make_move, and board.pawn_hash here is the CHILD hash, which
                    // for pawn moves reads an unrelated bucket (a write/read
                    // mismatch).
                    hist_score += info.pawn_hist[ph_idx][gp][to as usize] as i32;
                }
                // CONTINUOUS: sub-ply history adjustment (a +8000 history at
                // DIV=12000 now subtracts 0.67 ply instead of 0).
                let hist_adj = hist_score * LMR_SCALE / tp(&LMR_HIST_DIV);
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
                    reduction -= complexity * LMR_SCALE / tp(&LMR_COMPLEXITY_DIV); // CONTINUOUS
                }

                // Threat-density LMR: reduce less when multiple pieces are
                // under pawn attack. Tactical positions need deeper search.
                // Fixed-point divisor: stored × 10. Avoids tp10 swallowing
                // sub-integer SPSA precision on this multiplicative use.
                reduction -= threat_count * 10 * LMR_SCALE / LMR_THREAT_DIV_10X.load(Ordering::Relaxed).max(1); // CONTINUOUS

                // King-pressure LMR modifier: reduce less when enemy has
                // many attackers on our king zone. Parent-node signal reused
                // from NMP/ProbCut gates — tactical king positions need depth.
                reduction -= king_zone_pressure * 10 * LMR_SCALE / LMR_KING_PRESSURE_DIV_10X.load(Ordering::Relaxed).max(1); // CONTINUOUS

                // Clamp: never extend (negative), never reduce past depth 1.
                // Note: `new_depth - 1` can be -1 when negative singular
                // extensions drove `new_depth` to 0; re-clamp to keep the
                // stored value non-negative (downstream reads compare to
                // 0/2/3 thresholds, but a negative `info.reductions[ply_u]`
                // violates the local invariant).
                if reduction < 0 {
                    reduction = 0;
                }
                if reduction > (new_depth - 1) * LMR_SCALE {
                    reduction = (new_depth - 1) * LMR_SCALE;
                }
                if reduction < 0 {
                    reduction = 0;
                }
            }
        }

        // LMR for captures: use separate capture LMR table with capture history adjustments
        if !in_check && is_cap && !is_promo && move_count > 1 && mv != tt_move && !is_endgame_skip && FEAT_LMR.load(Ordering::Relaxed) {
            // Only reduce at non-PV nodes (zero window search)
            if beta - alpha == 1 {
                let d = (depth as usize).min(63);
                let m = (move_count as usize).min(63);
                reduction = lmr_cap_reduction(d as i32, m as i32); // CENTI-PLY

                if reduction >= LMR_SCALE {
                    // Continuous capture history adjustment: `capt_hist /
                    // LMR_HIST_DIV_CAP`, mirroring quiet-LMR's `hist_score /
                    // LMR_HIST_DIV` and Obsidian's
                    // `R -= hist/(isQuiet?Q_DIV:C_DIV)`. Must NOT be a step
                    // function (±1 at some |capt_hist| threshold): the
                    // step-vs-continuous asymmetry against quiet LMR forces
                    // SPSA to compress LMR_C_CAP below LMR_C_QUIET, because a
                    // binary fire gives it no other way to express the tactical
                    // capt_hist signal.
                    if moved_piece != NO_PIECE && captured_pt != NO_PIECE_TYPE {
                        let ct = if flags == FLAG_EN_PASSANT { captured_type(PAWN) } else { captured_type(captured_pt) };
                        let capt_hist_val = info.history.capture[go_piece(moved_piece)][to as usize][ct] as i32;
                        reduction -= capt_hist_val * LMR_SCALE / tp(&LMR_HIST_DIV_CAP); // CONTINUOUS
                    }

                    // Reduce less for captures that give check
                    if gives_check {
                        reduction -= LMR_SCALE;
                    }

                    // Correction battery (a)-(c) — applied to noisy
                    // moves too (computed before the quiet/noisy
                    // split). Same tunables as the quiet block.
                    if is_win(beta) {
                        reduction += tp(&LMR_WINBETA_CENTI);
                    }
                    if tt_hit && tt_entry.flag != TT_FLAG_NONE {
                        let tt_score_node = score_from_tt(tt_entry.score, ply, board.halfmove);
                        if tt_score_node <= alpha {
                            reduction += tp(&LMR_TTALPHA_CENTI);
                        }
                    }

                    // cutoff_count — applied before the
                    // quiet/noisy split, so captures get it too. Same
                    // tunables as the quiet block.
                    if info.cutoff_count[ply_u + 1] > 2 {
                        reduction += tp(&LMR_CUTOFF_CNT_CENTI);
                        if !is_pv && !cut_node {
                            reduction += tp(&LMR_CUTOFF_ALLNODE_CENTI);
                        }
                    }

                    if reduction < 0 {
                        reduction = 0;
                    }
                    // Never reduce past depth 1. Same `new_depth == 0` re-clamp
                    // as the quiet-LMR path above.
                    if reduction > (new_depth - 1) * LMR_SCALE {
                        reduction = (new_depth - 1) * LMR_SCALE;
                    }
                    if reduction < 0 {
                        reduction = 0;
                    }
                }
            }
        }

        // Root-depth-aware LMR relaxation: reduce LESS when the overall
        // search is deep. Grows with how deep the search reaches, so late moves
        // are searched closer to full depth. NOT zero at STC: measured warm-TT
        // at 250ms/move this fires on 61% of moves (7% opening, 100% late
        // middlegame). See the LMR_ROOT_THRESH block for the measurement. Deliberately one formula and one tunable set, rather
        // than separate STC/LTC shapes.
        if reduction >= LMR_SCALE {
            // CONTINUOUS: /100 * LMR_SCALE(=100) cancels exactly.
            reduction -= (info.root_depth - tp(&LMR_ROOT_THRESH)).max(0) * tp(&LMR_ROOT_COEF_10X) / 10;
            if reduction < 0 { reduction = 0; }
        }

        // FLOOR once: centi-ply accumulator -> integer plies. Everything
        // downstream (reductions[] stack for hindsight, doDeeper margin,
        // lmr_depth) keeps integer semantics. floor(floor-composed terms)
        // reproduces the old integer arithmetic bit-for-bit at defaults.
        reduction /= LMR_SCALE;

        // Store reduction for child's hindsight gating
        info.reductions[ply_u] = reduction;

        // Track nodes per root move for node-based time management
        let nodes_before = if ply == 0 { info.nodes } else { 0 };

        if reduction > 0 {
            info.stats.lmr_searches += 1;

            // LMR: reduced depth, zero window
            trace_gate!(info, board.hash, ply, mv, "lmr_reduced", reduction, move_count);
            let lmr_depth = new_depth - reduction;
            let mut lmr_score = -negamax(board, info, -alpha - 1, -alpha, lmr_depth, ply + 1, true);

            // The reduction applies to the reduced search ONLY: zero the slot
            // before any re-search so children of the (near-)full-depth
            // re-searches don't read a stale prior_reduction and mis-fire
            // hindsight reduce/extend (SF and Stormphrax both zero their
            // reduction slot after the reduced search).
            info.reductions[ply_u] = 0;

            if lmr_score > alpha && !info.stop.load(Ordering::Relaxed) {
                // LMR failed high: doDeeper/doShallower before re-search.
                //
                // Both margins below are CENTIPAWNS. Coda once used
                // `new_depth` (an integer depth, 5-20) directly as the
                // do_shallower margin; it happened to land in a workable
                // range, which masked the unit error for a long time.
                num_fail_highs += 1; // LMR fail-high cascade (Starzix pattern).
                let mut do_deeper_adj = 0;
                if lmr_score > best_score + 60 + 10 * reduction {
                    do_deeper_adj = 1;
                } else if lmr_score < best_score + 20 {
                    do_deeper_adj = -1;
                }

                // Mutate new_depth itself so the adjustment persists into the
                // full-window PVS re-search below (SF/Obsidian/Alexandria/
                // Stormphrax all mutate newDepth). Applying it inline instead
                // runs the PV re-search SHALLOWER than the zero-window search
                // that justified it.
                new_depth += do_deeper_adj;
                // Guard: only re-search when new_depth actually changed from lmr_depth.
                // do_shallower with reduction==1 makes new_depth == lmr_depth — the
                // re-search would duplicate the already-completed LMR search. Every
                // reference engine guards with `if new_depth > lmr_depth`.
                if new_depth > lmr_depth {
                    info.stats.ts_lmr_research += 1;
                    lmr_score = -negamax(board, info, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
                }

                // Post-LMR-research cont-hist nudge (Berserk pattern).
                // After the zero-window re-search, nudge cont-hist
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
                        // base = current cont_hist + main_hist / 2 (concept from Stormphrax).
                        let main_score_v = info.history.main_score(from, to, enemy_attacks);
                        let ch_offsets = [1usize, 2, 4, 6];
                        for &off in &ch_offsets {
                            if ply_u >= off {
                                let prior_piece = info.moved_piece_stack[ply_u - off] as usize;
                                let prior_to = info.moved_to_stack[ply_u - off] as usize;
                                if prior_piece > 0 && prior_piece < crate::movepicker::CONT_PLANES && prior_to < 64 {
                                    // Uniform bonus across offsets {1,2,4,6},
                                    // as Berserk/Alexandria/Stormphrax do —
                                    // NOT a [bonus, b/2, b/2, b/2] taper.
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
            // First move: always full window. Child cut_node: at a PV node the
            // first child is itself a PV node (never a cut node) -> false; at a
            // non-PV node the first child's expected type is the negation of
            // this node's (all-node -> child is cut; cut-node -> first child is
            // all after the cutoff move) -> !cut_node. Matches SF (Step 18
            // `!cutNode` for non-PV first move vs `search<PV>(... false)`).
            // Hardcoding `false` here mislabels every non-PV first move as an
            // all-node and disables the NMP/IIR/TT-cutoff node-type guards
            // along that whole spine.
            let child_cut = if is_pv { false } else { !cut_node };
            score = -negamax(board, info, -beta, -alpha, new_depth, ply + 1, child_cut);
        }

        board.unmake_move();
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }

        // Accumulate nodes for this root move
        if ply == 0 {
            let idx = root_move_index(mv);
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
                alpha_raise_count += 1;

                // Update triangular PV table
                if ply_u <= MAX_PLY {
                    info.pv_table[ply_u][0] = mv;
                    let child_len = if ply_u < MAX_PLY { info.pv_len[ply_u + 1] } else { 0 };
                    let copy_len = child_len.min(MAX_PLY - ply_u);
                    for i in 0..copy_len {
                        info.pv_table[ply_u][1 + i] = info.pv_table[ply_u + 1][i];
                    }
                    info.pv_len[ply_u] = 1 + child_len;
                }

                if alpha >= beta {
                    info.stats.beta_cutoffs += 1;
                    info.cutoff_count[ply_u] += 1;
                    if move_count == 1 { info.stats.first_move_cutoffs += 1; }
                    info.stats.cuts_by_depth[ts_bucket] += 1;
                    if move_count == 1 { info.stats.first_cuts_by_depth[ts_bucket] += 1; }
                    // First-move fail-high source split:
                    // class 0 = cutoff move is the TT move, 1 = noisy
                    // (capture/promo), 2 = quiet; separately condition on
                    // whether a TT move existed at this node at all.
                    {
                        let cls = if mv == tt_move { 0 }
                            else if is_cap || is_promo { 1 } else { 2 };
                        info.stats.cut_by_source[cls] += 1;
                        if move_count == 1 { info.stats.first_cut_by_source[cls] += 1; }
                        // True quiet-stage ordering quality: the cutter's rank
                        // AMONG QUIETS (quiets_tried includes the cutter).
                        if cls == 2 && quiets_count > 0 {
                            info.stats.cut_quiet_rank_sum += quiets_count as u64;
                            if quiets_count == 1 { info.stats.cut_quiet_rank1 += 1; }
                        }
                        let had_tt = (tt_move != NO_MOVE) as usize;
                        info.stats.cut_by_ttpresence[had_tt] += 1;
                        if move_count == 1 { info.stats.first_cut_by_ttpresence[had_tt] += 1; }
                    }
                    info.stats.cutoff_movecount_sum += move_count as u64;
                    info.stats.cutoff_movecount_sq_sum += (move_count as u64) * (move_count as u64);

                    // Beta cutoff - update history for quiet moves.
                    // The !is_promo gate matters: letting non-capture promotions
                    // take the quiet branch writes main/cont/pawn-history
                    // bonuses for a move the picker orders in the CAPTURE stage,
                    // polluting cells that genuine quiets read (quiets_tried
                    // already excludes promos). They belong in the capture
                    // branch, training the capt-hist empty slot the read side
                    // uses — SF capture_stage semantics.
                    if !is_cap && !is_promo {
                        // Depth-boost on a big fail-high. Two triggers
                        // (Stormphrax: cutoff beat static eval; improving) can
                        // stack for +2 depth.
                        let bonus_depth = depth
                            + if !in_check && static_eval <= best_score { 1 } else { 0 }
                            + if improving { 1 } else { 0 };
                        // numFailHighs multiplicative scaling (Starzix pattern)
                        // — more cascades = stronger cutoff confidence.
                        let raw_bonus = history_bonus(bonus_depth);
                        let scale_factor = num_fail_highs.min(tp10(&NFH_CAP_10X));
                        // Fixed-point divisor (stored × 10).
                        let mut bonus = raw_bonus + raw_bonus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);
                        // SF searched-count scale: the more moves were refuted
                        // before this one cut, the more informative the cutoff
                        // — scale the bonus up by moves-searched/256 at non-PV
                        // nodes.
                        if !is_pv {
                            bonus += bonus * (move_count - 1).max(0) / 256;
                        }
                        // Also from SF: at non-PV nodes, amplify the cutoff move's
                        // bonus by the number of moves searched before it cut off
                        // (more competition survived = stronger signal). Best-move
                        // bonus only — the malus is left unscaled.
                        if !is_pv {
                            let siblings = (quiets_count + n_captures_tried) as i32;
                            bonus += bonus * siblings / tp(&HIST_SIBLING_DIV);
                        }
                        // Malus magnitude: separate constants, same scaling chain
                        // (identical to bonus at default tunables → bench-identical).
                        let raw_malus = history_malus(bonus_depth);
                        let malus = raw_malus + raw_malus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);

                        // Update main history
                        History::update_history(
                            info.history.main_entry(from, to, enemy_attacks),
                            bonus,
                        );

                        // Update continuation history at plies 1, 2, 4, 6
                        // Ply-1 at full bonus, plies 2/4/6 at half bonus (Obsidian pattern)
                        if moved_piece != NO_PIECE {
                            let gp_mv = go_piece(moved_piece);
                            // base = current cont_hist + main_hist / 2 (concept from Stormphrax).
                            let main_score_v = info.history.main_score(from, to, enemy_attacks);
                            let ch_offsets = [1usize, 2, 4, 6];
                            for &off in ch_offsets.iter() {
                                if ply_u >= off {
                                    let prior_piece = info.moved_piece_stack[ply_u - off] as usize;
                                    let prior_to = info.moved_to_stack[ply_u - off] as usize;
                                    if prior_piece > 0 && prior_piece < crate::movepicker::CONT_PLANES && prior_to < 64 {
                                        // B1: uniform bonus (see LMR nudge site above).
                                        let ch_bonus = bonus;
                                        let cur_cont = info.history.cont_hist[prior_piece][prior_to][gp_mv][to as usize] as i32;
                                        let base = cur_cont + main_score_v / 2;
                                        History::update_cont_history_with_base(
                                            &mut info.history.cont_hist[prior_piece][prior_to][gp_mv][to as usize],
                                            base,
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
                                -malus,
                            );

                            // Penalize continuation history at plies 1, 2, 4, 6.
                            // T6: base uses qf,qt move's main_score (the move being malused).
                            {
                                let q_piece = board.piece_at(qf);
                                if q_piece != NO_PIECE {
                                    let gp_q = go_piece(q_piece);
                                    let q_main_score = info.history.main_score(qf, qt, enemy_attacks);
                                    let ch_offsets = [1usize, 2, 4, 6];
                                    for &off in ch_offsets.iter() {
                                        if ply_u >= off {
                                            let prior_piece = info.moved_piece_stack[ply_u - off] as usize;
                                            let prior_to = info.moved_to_stack[ply_u - off] as usize;
                                            if prior_piece > 0 && prior_piece < crate::movepicker::CONT_PLANES && prior_to < 64 {
                                                // B1: uniform penalty (see bonus site above).
                                                let ch_pen = -malus;
                                                let cur_cont = info.history.cont_hist[prior_piece][prior_to][gp_q][qt as usize] as i32;
                                                let base = cur_cont + q_main_score / 2;
                                                History::update_cont_history_with_base(
                                                    &mut info.history.cont_hist[prior_piece][prior_to][gp_q][qt as usize],
                                                    base,
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
                                    let clamped = (-malus).clamp(-16384, 16384);
                                    let new_v = v + clamped - v * clamped.abs() / 16384;
                                    info.pawn_hist[ph_idx][gp][qt as usize] = new_v.clamp(-32000, 32000) as i16;
                                }
                            }
                        }

                    } else {
                        // Capture caused beta cutoff: bonus the cutoff capture.
                        // No depth boost here, unlike the quiet site above.
                        let cap_bonus_depth = depth;
                        // numFailHighs multiplicative scaling, as at the quiet
                        // site: more cascades = stronger cutoff confidence.
                        let raw_cap_bonus = capture_history_bonus(cap_bonus_depth);
                        let scale_factor = num_fail_highs.min(tp10(&NFH_CAP_10X));
                        // Fixed-point divisor (stored × 10).
                        let mut cap_bonus = raw_cap_bonus + raw_cap_bonus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);
                        // SF searched-count scale (see quiet site above).
                        if !is_pv {
                            cap_bonus += cap_bonus * (move_count - 1).max(0) / 256;
                        }
                        // captured_pt == NO_PIECE_TYPE here means a non-capture
                        // promotion (the only non-capture route into this branch):
                        // train slot 0 ("empty") — the slot capt_hist_score_static
                        // reads when ordering non-capture promos.
                        if moved_piece != NO_PIECE {
                            let cpt = if flags == FLAG_EN_PASSANT {
                                captured_type(PAWN)
                            } else if captured_pt != NO_PIECE_TYPE {
                                captured_type(captured_pt)
                            } else {
                                0
                            };
                            History::update_cont_history(
                                &mut info.history.capture[go_piece(moved_piece)][to as usize][cpt],
                                cap_bonus,
                            );
                        }
                    }


                    // Unconditionally penalize all tried captures that didn't cause cutoff
                    // (matching Stockfish/Obsidian — captures that fail should be
                    // penalized regardless of whether the best move was quiet or tactical)
                    {
                        // numFailHighs multiplicative scaling — mirrors the
                        // capture-BONUS path exactly so the failed captures'
                        // penalty tracks the cutoff capture's bonus. Omitting it
                        // here (the easy mistake: this is the least obvious of
                        // the beta-cutoff history updates) slowly inflates
                        // capture-history magnitude relative to the scaled
                        // bonuses and to quiets.
                        let raw_cap_malus = capture_history_malus(depth);
                        let scale_factor = num_fail_highs.min(tp10(&NFH_CAP_10X));
                        let cap_malus = raw_cap_malus + raw_cap_malus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);
                        // is_promo joins is_cap: a promo cutoff move is the
                        // last captures_tried entry and must not malus itself.
                        let cap_count = if is_cap || is_promo { n_captures_tried.saturating_sub(1) } else { n_captures_tried };
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
    // TREESTATS: node exit — width = moves actually searched at this node
    // (includes cutoff breaks; nodes that pruned before the move loop appear
    // in nodes_by_depth but not here, so 1 − width_cnt/nodes per bucket is
    // the pre-move-loop exit rate, matching the SF-side convention).
    info.stats.width_sum_by_depth[ts_bucket] += move_count as u64;
    info.stats.width_cnt_by_depth[ts_bucket] += 1;

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

    // TB floor: a PV in-window TB hit established `tb_score` as ground
    // truth. If the local search couldn't beat it, return / store the TB
    // value instead of the sub-TB local result. Without this the next
    // block stores UPPER below TB truth and poisons future probes.
    if let Some(floor) = tb_floor {
        if best_score < floor {
            best_score = floor;
        }
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

        // Fail-low node histogram (stats only): depth x margin x quiets.
        if flag == TT_FLAG_UPPER && move_count > 0 && best_score.abs() < MATE_IN_MAX_PLY {
            let d_band = if depth <= 4 { 0 } else if depth <= 8 { 1 } else { 2 };
            let margin = (alpha_orig - best_score).max(0);
            let m_band = if margin < 50 { 0 } else if margin < 150 { 1 }
                else if margin < 300 { 2 } else { 3 };
            let q_band = if quiets_count <= 2 { 0 } else if quiets_count <= 5 { 1 }
                else if quiets_count <= 9 { 2 } else { 3 };
            let idx = m_band * 4 + q_band;
            info.stats.b_probe_nodes[d_band][idx] += 1;
            info.stats.b_probe_quiets[d_band][idx] += quiets_count as u64;
            info.stats.b_probe_late[d_band][idx] += (quiets_count.saturating_sub(2)) as u64;
        }

        // Adjust mate score for storage (relative to this position)
        let store_score = score_to_tt(best_score, ply);

        if FEAT_TT_STORE.load(Ordering::Relaxed) {
            info.tt.store(board.hash, depth, store_score, flag, best_move, raw_eval, tt_pv);
        }
    }

    // Fail-low prior-countermove bonus (SF technique, simplified core).
    // When this node fails low with NO best move, the opponent's previous
    // quiet move "worked" — credit it in the cont-hist context of our move
    // before that, so the PARENT tries better siblings sooner. All-nodes are
    // the majority class in a big tree, and without this the search learns
    // nothing from them (history updates would fire only on beta cutoffs and
    // TT cutoffs). Indexing mirrors the TT-cutoff cont-hist malus site
    // (moved_piece_stack, pre-move pieces).
    // NOTE: SF's gate fires only when no move raised alpha
    // (they set their best-move only on alpha raises). Coda tracks a fail-soft
    // best_move below alpha too, so the equivalent condition here is the
    // fail-low bound itself, NOT best_move == NO_MOVE (which never holds).
    if best_score <= alpha_orig
        && info.excluded_move[ply_u] == NO_MOVE
        && !info.stop.load(Ordering::Relaxed)
        && ply_u >= 2
    {
        let stack_len = board.undo_stack.len();
        if stack_len >= 2 {
            let opp_undo = &board.undo_stack[stack_len - 1];
            let our_undo = &board.undo_stack[stack_len - 2];
            if opp_undo.mv != NO_MOVE && opp_undo.captured == NO_PIECE_TYPE
                && our_undo.mv != NO_MOVE
            {
                let opp_gp = info.moved_piece_stack[ply_u - 1] as usize;
                let our_gp = info.moved_piece_stack[ply_u - 2] as usize;
                let opp_to = info.moved_to_stack[ply_u - 1] as usize;
                let our_to = info.moved_to_stack[ply_u - 2] as usize;
                if opp_gp > 0 && opp_gp < crate::movepicker::CONT_PLANES
                    && our_gp > 0 && our_gp < crate::movepicker::CONT_PLANES
                    && opp_to < 64 && our_to < 64
                {
                    let bonus = history_bonus(depth) * tp(&FAIL_LOW_PREV_BONUS_PCT) / 100;
                    History::update_cont_history(
                        &mut info.history.cont_hist[our_gp][our_to][opp_gp][opp_to],
                        bonus,
                    );
                }
            }
        }
    }

    // Update pawn-hash correction history when we have a reliable score.
    //
    // Skip when best_move is a capture/promotion: the score delta
    // (best_score - raw_eval) is then dominated by material change, not the
    // positional-eval miscalibration correction history is trying to learn.
    // Training on noisy bestmoves pollutes the tables. Matches Stockfish
    // (skip the update when the best move is a capture/promotion).
    let best_move_noisy = best_move != NO_MOVE && {
        board.piece_type_at(move_to(best_move)) != NO_PIECE_TYPE
            || move_flags(best_move) == FLAG_EN_PASSANT
            || is_promotion(best_move)
    };
    // Correction history update: train on BOTH directions of error.
    // Previously gated on `best_score > alpha_orig` (fail-high only), which
    // never trained on fail-low (all-node) positions where static eval was
    // over-optimistic. SF updates on fail-low when the error
    // direction is consistent: bound==Upper && best_score < static_eval means
    // eval predicted higher than any move achieved — train correction downward.
    let corrhist_lower_ok = best_score > alpha_orig   // fail-high: lower bound
        && !(best_score >= beta && best_score <= static_eval); // direction-consistent
    let corrhist_upper_ok = best_score <= alpha_orig  // fail-low: upper bound
        && best_score < static_eval;                   // corrected eval was over-optimistic
    if !in_check
        && !best_move_noisy
        && info.excluded_move[ply_u] == NO_MOVE
        && (corrhist_lower_ok || corrhist_upper_ok)
        // is_decisive covers the mate OR TB range
        && !is_decisive(best_score)
        && scaled_eval > -(MATE_IN_MAX_PLY)
        && !info.stop.load(Ordering::Relaxed)
    {
        // Train the update against the CORRECTED eval (`static_eval`, the
        // residual after correction), like SF/Obsidian/Berserk — NOT the raw
        // `scaled_eval`. Training on raw makes the gravity fixed point the
        // rail itself (magnitude-blind), which manufactures phantom evals in
        // fortress positions; the residual converges to the true correction
        // and self-stabilises. Both are in scaled space, so the err term
        // isolates positional miscalibration rather than halfmove decay.
        update_correction_history(info, board, best_score, static_eval, depth, ply_u);
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
        && !is_decisive(best_score)
        && info.excluded_move[ply_u] == NO_MOVE
        && FEAT_FH_BLEND.load(Ordering::Relaxed)
    {
        // Divisor floor: SPSA can perturb FH_BLEND_OFFSET to 0 AND
        // FH_BLEND_DEPTH_10X low enough that the gate admits depth=0,
        // producing a 0+0 divisor → div-by-zero panic. Clamp.
        return (best_score * depth + beta) / (depth + tp(&FH_BLEND_OFFSET)).max(1);
    }

    best_score
}

/// History bonus: linear depth-based bonus for history updates.
/// Consensus shape across SF/Clarity/Obsidian: a clamped linear-in-depth
/// bonus (min(MAX, MULT*d - OFFSET)), each engine with its own coefficients.
/// Our old depth² formula gave values ~27× too small at low depth to
/// influence move ordering.
fn history_bonus(depth: i32) -> i32 {
    // Offset shape — the consensus linear-in-depth bonus (SF/Clarity/Obsidian
    // all use MULT*d - OFFSET) with Coda's own SPSA-tuned coefficients; same
    // form as our capture-history's `MULT * d - BASE`. Clamped at 0 to avoid
    // negative bonuses at very shallow depth (which would corrupt
    // gravity updates) and at MAX to cap the late-depth plateau.
    (tp(&HIST_BONUS_MULT) * depth - tp(&HIST_BONUS_OFFSET)).clamp(0, tp(&HIST_BONUS_MAX))
}

fn capture_history_bonus(depth: i32) -> i32 {
    (tp(&CAP_HIST_MULT) * depth - tp(&CAP_HIST_BASE)).clamp(0, tp(&CAP_HIST_MAX))
}

/// Malus (penalty) magnitude for quiet-history updates. Same shape as
/// history_bonus but with independent constants — see the tunables!
/// comment at HIST_MALUS_MULT for the cross-engine rationale.
fn history_malus(depth: i32) -> i32 {
    (tp(&HIST_MALUS_MULT) * depth - tp(&HIST_MALUS_OFFSET)).clamp(0, tp(&HIST_MALUS_MAX))
}

/// Malus magnitude for capture-history updates (independent constants).
fn capture_history_malus(depth: i32) -> i32 {
    (tp(&CAP_HIST_MALUS_MULT) * depth - tp(&CAP_HIST_MALUS_BASE)).clamp(0, tp(&CAP_HIST_MALUS_MAX))
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
    info.stats.nodes_by_depth[0] += 1; // TREESTATS: qsearch = bucket 0

    // Draw detection: repetition and 50-move rule. No contempt term.
    let draw_score = 0;
    if board.halfmove >= 100 {
        return draw_score;
    }
    // FIDE Art 5.2: insufficient material to mate (any side). Mirrors
    // negamax's guard, which exists for the drawn KB-vs-K class seen in
    // live play. QS recurses capture chains that can transition into drawn
    // KvK / KBvK / KBvKB-same-color without ever re-entering negamax's
    // check, so the parallel guard is needed here too.
    if board.is_insufficient_material() {
        return draw_score;
    }
    if board.is_repetition_draw(ply) {
        return draw_score;
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
    if info.nodes & 1023 == 0
        && info.should_stop() {
            return 0;
        }

    if info.stop.load(Ordering::Relaxed) {
        return 0;
    }

    // Cuckoo cycle detection in quiescence
    // Gate QS cuckoo on ply > 0, mirroring the main-search check.
    // Cuckoo's root-boundary STM check is undefined at ply 0.
    if ply > 0 && alpha < 0 && FEAT_CUCKOO.load(Ordering::Relaxed) && crate::cuckoo::has_game_cycle(board, ply) {
        alpha = 0;
        if alpha >= beta {
            return alpha;
        }
    }

    // Probe transposition table
    let tt_entry = info.tt.probe(board.hash);
    let tt_move = if tt_entry.hit { tt_entry.best_move } else { NO_MOVE };

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
        // 50mr mate/TB downgrade happens inside score_from_tt, so both the
        // cutoff conditions and the returned value use the sanitized score
        // (SF: ttValue is value_from_tt output everywhere).
        let tt_score = score_from_tt(tt_entry.score, ply, board.halfmove);

        // P2: skip QS TT cutoff near 50mr — stale bound unsafe
        let halfmove_ok = (board.halfmove as i32) < tp(&TT_CUTOFF_HALFMOVE_MAX);
        let qs_is_pv = beta - alpha > 1;
        match tt_entry.flag {
            TT_FLAG_EXACT => {
                if !qs_is_pv && halfmove_ok { return tt_score; }
            }
            TT_FLAG_LOWER => {
                if !qs_is_pv && halfmove_ok && tt_score >= beta { return tt_score; }
            }
            TT_FLAG_UPPER => {
                if !qs_is_pv && halfmove_ok && tt_score <= alpha { return tt_score; }
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
        // Evasion history reads use enemy_attacks, symmetric with the
        // beta-cutoff writes. Computed here because QS doesn't otherwise need
        // the bitboard.
        let qs_enemy_attacks = board.attacks_by_color(
            crate::types::flip_color(board.side_to_move)
        );
        // Clamp ply to the moved_piece_stack / moved_to_stack bounds.
        // Qsearch can recurse past MAX_PLY via tactical extensions and evasion
        // chains; without this clamp MovePicker::new_evasion indexes
        // `moved_piece_stack[ply - off]` with ply > MAX_PLY and panics — which
        // has thrown a won game in live play.
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

            // Skip quiet evasions once we have a non-losing score:
            // SF searches quiet evasions only while still losing
            // (it skips non-captures inside its !is_loss guard);
            // Obsidian breaks after one quiet. Capture evasions always searched. The gate is
            // only satisfiable after at least one legal move scored, so
            // checkmate detection (move_count == 0) is unaffected.
            let ev_is_cap = board.piece_type_at(move_to(mv)) != NO_PIECE_TYPE
                || move_flags(mv) == FLAG_EN_PASSANT;
            if !ev_is_cap && !is_promotion(mv) && !is_loss(best_score) {
                continue;
            }

            let qs_moved_pt = board.piece_type_at(move_from(mv));
            let qs_captured_pt = if move_flags(mv) == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(move_to(mv)) };
            let qs_dirty = if let Some(net) = info.nnue_net.as_deref() {
                build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), qs_moved_pt, qs_captured_pt, net)
            } else { DirtyPiece::recompute() };

            // Record on search stack so deeper QS evasion's MovePicker::new_evasion
            // reads correct ply-1/ply-2 continuation-history context. Without
            // this, QS evasion chains read stale moved_piece_stack from the
            // ply at which main search entered QS.
            if qs_safe_ply <= MAX_PLY {
                let qs_mp = board.piece_at(move_from(mv));
                if qs_mp != NO_PIECE {
                    info.moved_piece_stack[qs_safe_ply] = go_piece(qs_mp) as u8;
                    info.moved_to_stack[qs_safe_ply] = move_to(mv);
                }
            }

            if let Some(acc) = &mut info.nnue_acc { acc.push(qs_dirty); }
        if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }
            if !board.make_move(mv) {
                if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }
                continue;
            }
                    if info.threat_stack.active {
                        info.threat_stack.absorb_deltas(board);
                    }
            info.tt.prefetch(board.hash);
            move_count += 1;

            let score = -quiescence_with_depth(board, info, -beta, -alpha, ply + 1, qs_depth + 1);
            board.unmake_move();
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }

            if info.stop.load(Ordering::Relaxed) {
                return 0;
            }

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

        // Store in TT (skip if stopped — partial QS results corrupt TT).
        // Never EXACT: a QS score is an approximation (children are
        // captures-only subtrees), and EXACT entries satisfy unconditional
        // EXACT cutoffs and the stand-pat refinement at full confidence.
        // SF/Obsidian all store only LOWER/UPPER in QS.
        let store_score = score_to_tt(best_score, ply);
        let flag = if best_score >= beta {
            TT_FLAG_LOWER
        } else {
            TT_FLAG_UPPER
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
    let raw_stand_pat = if FEAT_TT_STATIC_EVAL.load(Ordering::Relaxed)
        && tt_hit
        && tt_entry.static_eval > -4095
    {
        // Threshold matches pack_data's clamp range. -INFINITY sentinels
        // (from in-check TT stores) get clamped to -4095 and would
        // otherwise pass a wider check.
        info.stats_tt_static_eval_hits += 1;
        tt_entry.static_eval
    } else {
        info.eval(board)
    };
    // Apply correction history to the QS stand-pat, mirroring negamax's
    // static-eval path (halfmove-scale THEN corrected_eval).
    // Every reference engine surveyed (SF, Berserk, Obsidian, PlentyChess,
    // Alexandria) corrects the QS stand-pat. It feeds the returned cutoff
    // score, the best_score floor, AND the delta-prune base, so an
    // uncorrected error compounds. TT still stores the RAW value
    // (raw_stand_pat) — correct-on-read discipline is unchanged.
    let scaled_stand_pat = apply_halfmove_scale(raw_stand_pat, board.halfmove);
    let stand_pat = if FEAT_CORRECTION.load(Ordering::Relaxed) {
        corrected_eval(info, board, scaled_stand_pat, (ply as usize).min(MAX_PLY))
    } else {
        scaled_stand_pat
    };
    let mut best_score = stand_pat;

    // TT bound refinement of stand-pat (consensus: every top engine does this)
    // Use TT score as a better estimate when the bound direction agrees.
    // Apply the SAME halfmove guard as the direct cutoff path in negamax:
    // without this, an inflated near-50mr TT lower bound replaces
    // stand_pat and triggers the `best_score >= beta` return below —
    // bypassing the gate that exists for exactly this case.
    if tt_hit && (board.halfmove as i32) < tp(&TT_CUTOFF_HALFMOVE_MAX) {
        // 50mr downgrade applies here too. A downgraded mate becomes a
        // TB-band value and is still filtered by !is_decisive below; a
        // downgraded TB score becomes the highest non-decisive value and
        // may refine stand-pat — same as SF, whose eval
        // refinement consumes value_from_tt output.
        let tt_score = score_from_tt(tt_entry.score, ply, board.halfmove);
        if !is_decisive(tt_score)
            && ((tt_entry.flag == TT_FLAG_LOWER && tt_score > best_score)
                || (tt_entry.flag == TT_FLAG_UPPER && tt_score < best_score)
                || tt_entry.flag == TT_FLAG_EXACT)
            {
                best_score = tt_score;
            }
    }

    if best_score >= beta {
        // Cache eval + LOWER bound on the stand-pat fail-high. This is the
        // most common QS exit; returning here without a TT store leaves the
        // raw eval uncached (revisits re-run NNUE) and no bound for a cheap
        // future cutoff. All 6 reference engines store here
        // (SF/Alexandria/Plenty gate on TT-miss).
        // Store on TT miss only, depth -1, LOWER bound, no move, raw eval —
        // stand-pat is a valid lower bound on the QS node value. score_to_tt
        // stores the halfmove-independent value (raw_stand_pat); the score is
        // best_score (== stand_pat on a miss, since TT refinement needs a hit).
        if !tt_hit
            && FEAT_TT_STORE.load(Ordering::Relaxed)
            && !info.stop.load(Ordering::Relaxed)
        {
            info.tt.store(board.hash, -1, score_to_tt(best_score, ply),
                TT_FLAG_LOWER, NO_MOVE, raw_stand_pat, false);
        }
        // QS beta blending, applied regardless of node type — none of the 6
        // reference engines gates this on non-PV.
        if !is_decisive(best_score) {
            return (best_score + beta) / 2;
        }
        return best_score;
    }

    if best_score > alpha {
        alpha = best_score;
    }

    // FEAT_QS_CAPTURES (NO_QS_CAPTURES=1): skip the capture loop entirely, so
    // qsearch returns the raw stand-pat. This ablates QUIESCENCE ITSELF, not
    // "some captures" — the search then evaluates positions in the middle of
    // capture sequences, scores become noise, cutoffs stop working, and the
    // tree GROWS (+27.7% nodes, +17.9% evals at bench 10). Useful only to
    // confirm quiescence is load-bearing; it is NOT a measure of what
    // searching captures costs, and its nodes/evals delta must not be read as
    // a marginal cost. For that, sweep the QS_MAX_CAPTURES tunable instead —
    // which is graded, and is where the 5->3 win came from.
    if !FEAT_QS_CAPTURES.load(Ordering::Relaxed) {
        return best_score;
    }

    // Use main MovePicker in quiescence mode.
    // This partitions captures into good (SEE>=0) and bad, and uses staged ordering.
    let mut picker = MovePicker::new_quiescence(tt_move, &info.history, qs_checkers, qs_pinned);
    let mut best_move = NO_MOVE;
    let mut qs_move_count = 0i32;
    let qs_max_caps = tp(&QS_MAX_CAPTURES);

    loop {
        let mv = picker.next(board);
        if mv == NO_MOVE { break; }

        // Legality: every other move loop (negamax, QS evasions, probcut)
        // filters illegal moves; this one didn't. A capture by an absolutely
        // pinned piece passes make_move (which only rejects king captures),
        // and the child's refutation — capturing the king — is then blocked
        // by that same guard, so the illegal capture scores as winning.
        // SEE can't catch it (it deliberately ignores pins).
        if !board.is_legal(mv, qs_pinned, qs_checkers) {
            continue;
        }

        // Move-count budget: count only SEARCHED moves. Incrementing before
        // delta/SEE pruning lets pruned moves consume budget, and SPSA then
        // pushes the cap so high the feature is effectively off. Consensus
        // gates: only while best_score isn't a loss (SF's
        // is_loss(futilityBase) / Obsidian's TB_LOSS gate), promotions exempt
        // (SF). `continue`, not `break`, so later promotions still get
        // through — also the SF form.
        if qs_move_count >= qs_max_caps
            && !is_loss(best_score)
            && !is_promotion(mv)
        {
            continue;
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
                let delta_val = stand_pat + see_value(cap_pt) * tp(&SEE_MATERIAL_SCALE) / 100 + tp(&QS_DELTA_MARGIN);
                if delta_val <= alpha {
                    // Fail-soft: delta_val is an upper bound on what this
                    // capture could achieve; raise best_score to it so the returned
                    // UPPER bound reflects it (all 5 value-prune references do this).
                    // delta_val <= alpha, so best_score stays <= alpha — no cutoff.
                    best_score = best_score.max(delta_val);
                    continue;
                }
            }
        }

        // Skip bad captures (SEE below threshold)
        // Negative threshold allows slightly losing captures (e.g. BxN)
        // Obsidian uses -32
        if !see_ge(board, mv, tp(&QS_SEE_THRESHOLD)) {
            continue;
        }

        // Build lazy NNUE update
        let qs_moved_pt = board.piece_type_at(move_from(mv));
        let qs_captured_pt = if move_flags(mv) == FLAG_EN_PASSANT { PAWN } else { board.piece_type_at(move_to(mv)) };
        let qs_dirty = if let Some(net) = info.nnue_net.as_deref() {
            build_dirty_piece(mv, board.side_to_move, flip_color(board.side_to_move), qs_moved_pt, qs_captured_pt, net)
        } else { DirtyPiece::recompute() };

        // Record on search stack — same reason as the evasion path above.
        {
            let qs_idx = (ply as usize).min(MAX_PLY - 1);
            let qs_mp = board.piece_at(move_from(mv));
            if qs_mp != NO_PIECE {
                info.moved_piece_stack[qs_idx] = go_piece(qs_mp) as u8;
                info.moved_to_stack[qs_idx] = move_to(mv);
            }
        }

        if let Some(acc) = &mut info.nnue_acc { acc.push(qs_dirty); }
        if info.threat_stack.active { info.threat_stack.push(crate::types::NO_MOVE, crate::types::NO_PIECE_TYPE); }
        if !board.make_move(mv) {
            if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }
            continue;
        }
        qs_move_count += 1;
        if info.threat_stack.active {
            info.threat_stack.absorb_deltas(board);
        }
        info.tt.prefetch(board.hash);
        let score = -quiescence_with_depth(board, info, -beta, -alpha, ply + 1, qs_depth + 1);
        board.unmake_move();
        if let Some(acc) = &mut info.nnue_acc { acc.pop(); }
        if info.threat_stack.active { info.threat_stack.pop(); }

        if info.stop.load(Ordering::Relaxed) {
            return 0;
        }

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

    // Store in TT (skip if stopped — partial QS results corrupt TT).
    // Never EXACT — see the note at the evasion-path store above.
    let store_score = score_to_tt(best_score, ply);
    let flag = if best_score >= beta {
        TT_FLAG_LOWER
    } else {
        TT_FLAG_UPPER
    };
    if FEAT_TT_STORE.load(Ordering::Relaxed) && !info.stop.load(Ordering::Relaxed) {
        // Store the halfmove-INDEPENDENT value so later probes at a
        // different halfmove get a correct scale — see the doc comment
        // in `SearchInfo::eval`.
        info.tt.store(board.hash, -1, store_score, flag, best_move, raw_stand_pat, false);
    }

    // QS beta blending, regardless of node type (see the stand-pat exit).
    if best_score >= beta && !is_decisive(best_score) {
        return (best_score + beta) / 2;
    }

    best_score
}

/// Standard bench position list — 48 positions, imported from Stockfish's
/// `Defaults` array (chess960 + setoption control lines dropped, two endgame
/// FENs padded to 6 fields, SF Pohl knight-saturation test dropped — many
/// nets show wildly elevated tree size on it, which distorts bench aggregates
/// and the OB scale_nps; it lives in BENCH_PATHOLOGY_POSITIONS instead).
/// Used by `coda bench` and `coda eval-bench` so the prune-stats /
/// move-ordering / NPS aggregates have N=48 sample size, matching the field
/// convention (Halogen 49, Stormphrax 50, Alexandria 51, Stockfish 51).
/// The position set is based on Stockfish's benchmark positions (GPLv3, a
/// licence compatible with Coda's), so bench and NPS compare cleanly against
/// Stockfish.
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
    // pawns. Eval-driven non-convergence: a large fraction of freshly
    // trained nets show elevated tree size here, some exceeding 100M nodes
    // at depth 8. Not in the main bench list; kept here as a tripwire.
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
            // Explicit override failing is fatal — see bench_inner.
            eprintln!("FATAL: failed to load NNUE '{}': {}", path, e);
            std::process::exit(2);
        }
    } else {
        info.auto_discover_nnue();
    }
    let limits = SearchLimits {
        depth,
        fixed_depth: true,
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
        info.last_flushed_nodes.set(0);
        reset_bench_position_state(&mut info);
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
            // An EXPLICIT net override that fails must be fatal: silently
            // falling back to the embedded net produces a wrong bench / wrong
            // ordering stats that look plausible. The override is honoured only
            // when it loads; otherwise abort rather than mask the mistake.
            eprintln!("FATAL: failed to load NNUE '{}': {}", path, e);
            std::process::exit(2);
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
        fixed_depth: true,
        infinite: true,
        ..SearchLimits::new()
    };

    for fen in positions {
        let mut board = Board::from_fen(fen);
        info.nodes = 0;
        info.last_flushed_nodes.set(0);
        info.global_nodes.store(0, Ordering::Relaxed); // P2.8: reset per position — was cumulative, so every info line after #1 printed garbage NPS
        reset_bench_position_state(&mut info);

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
        total_stats.see_prunes += info.stats.see_prunes;
        total_stats.probcut_cutoffs += info.stats.probcut_cutoffs;
        total_stats.lmr_searches += info.stats.lmr_searches;
        total_stats.singular_ext += info.stats.singular_ext;
        total_stats.double_ext += info.stats.double_ext;
        total_stats.negative_ext += info.stats.negative_ext;
        total_stats.multicut += info.stats.multicut;
        total_stats.qnodes += info.stats.qnodes;
        total_stats.beta_cutoffs += info.stats.beta_cutoffs;
        total_stats.first_move_cutoffs += info.stats.first_move_cutoffs;
        for i in 0..3 {
            total_stats.cut_by_source[i] += info.stats.cut_by_source[i];
            total_stats.first_cut_by_source[i] += info.stats.first_cut_by_source[i];
            total_stats.rfp_audit_var_attempts[i] += info.stats.rfp_audit_var_attempts[i];
            total_stats.rfp_audit_var_fp[i] += info.stats.rfp_audit_var_fp[i];
        }
        for i in 0..2 {
            total_stats.cut_by_ttpresence[i] += info.stats.cut_by_ttpresence[i];
            total_stats.first_cut_by_ttpresence[i] += info.stats.first_cut_by_ttpresence[i];
        }
        total_stats.cut_quiet_rank1 += info.stats.cut_quiet_rank1;
        total_stats.cut_quiet_rank_sum += info.stats.cut_quiet_rank_sum;
        for i in 0..12 {
            total_stats.dualnet_evals[i] += info.stats.dualnet_evals[i];
            total_stats.dualnet_abseval[i] += info.stats.dualnet_abseval[i];
            total_stats.dualnet_neareq[i] += info.stats.dualnet_neareq[i];
        }
        for d in 0..3 {
            for i in 0..16 {
                total_stats.b_probe_nodes[d][i] += info.stats.b_probe_nodes[d][i];
                total_stats.b_probe_quiets[d][i] += info.stats.b_probe_quiets[d][i];
                total_stats.b_probe_late[d][i] += info.stats.b_probe_late[d][i];
            }
        }
        total_stats.moves_searched += info.stats.moves_searched;
        total_stats.cutoff_movecount_sum += info.stats.cutoff_movecount_sum;
        total_stats.cutoff_movecount_sq_sum += info.stats.cutoff_movecount_sq_sum;
        for d in 0..24 {
            total_stats.rfp_audit_attempts[d] += info.stats.rfp_audit_attempts[d];
            total_stats.rfp_audit_fp[d] += info.stats.rfp_audit_fp[d];
        }

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
    eprintln!("SEE prunes:     {:>8}", s.see_prunes);
    eprintln!("ProbCut cutoffs:{:>8}", s.probcut_cutoffs);
    eprintln!("LMR searches:   {:>8}  ({:.1}% of nodes)", s.lmr_searches, s.lmr_searches as f64 / total_nodes as f64 * 100.0);
    eprintln!("Singular ext:   {:>8}  (single +1 ply)", s.singular_ext);
    eprintln!("Double ext:     {:>8}  (additional +1 on top of singular)", s.double_ext);
    eprintln!("Negative ext:   {:>8}  (-1/-2/-3 fail-high reduce)", s.negative_ext);
    eprintln!("Multi-cut:      {:>8}  (return singular_beta)", s.multicut);
    eprintln!("QS nodes:       {:>8}  ({:.1}% of total)", s.qnodes, s.qnodes as f64 / total_nodes as f64 * 100.0);
    if s.beta_cutoffs > 0 {
        let avg_pos = s.cutoff_movecount_sum as f64 / s.beta_cutoffs as f64;
        let avg_sq = s.cutoff_movecount_sq_sum as f64 / s.beta_cutoffs as f64;
        let first_pct = s.first_move_cutoffs as f64 / s.beta_cutoffs as f64 * 100.0;
        {
            let names = ["tt-move", "noisy", "quiet"];
            for i in 0..3 {
                let c = s.cut_by_source[i].max(1);
                eprintln!("fh1[{}]: {:.1}% of {} cutoffs ({:.1}% of all cuts)",
                    names[i], 100.0 * s.first_cut_by_source[i] as f64 / c as f64,
                    s.cut_by_source[i],
                    100.0 * s.cut_by_source[i] as f64 / s.beta_cutoffs.max(1) as f64);
            }
            if s.cut_by_source[2] > 0 {
                eprintln!("fh1[quiet-RANK]: rank1 {:.1}%, avg quiet-rank {:.2} (of {} quiet cutoffs)",
                    100.0 * s.cut_quiet_rank1 as f64 / s.cut_by_source[2] as f64,
                    s.cut_quiet_rank_sum as f64 / s.cut_by_source[2] as f64,
                    s.cut_by_source[2]);
            }
            for (i, name) in ["no-tt-move", "has-tt-move"].iter().enumerate() {
                let c = s.cut_by_ttpresence[i].max(1);
                eprintln!("fh1[{}]: {:.1}% of {} cutoffs", name,
                    100.0 * s.first_cut_by_ttpresence[i] as f64 / c as f64,
                    s.cut_by_ttpresence[i]);
            }
            let va: u64 = s.rfp_audit_var_attempts.iter().sum();
            if va > 0 {
                let names = ["spread<8cp", "8-24cp", ">=24cp"];
                for i in 0..3 {
                    let a = s.rfp_audit_var_attempts[i].max(1);
                    eprintln!("RFP-FP[{}]: {:.1}% of {} audited",
                        names[i], 100.0 * s.rfp_audit_var_fp[i] as f64 / a as f64,
                        s.rfp_audit_var_attempts[i]);
                }
            }
        }
        eprintln!("Move ordering:  avg cutoff pos {:.2}, avg pos² {:.1}, first-move {:.1}%",
            avg_pos, avg_sq, first_pct);
        {
            let total: u64 = s.dualnet_evals.iter().sum();
            if total > 0 {
                eprintln!("--- Dual-net dispatch candidate (proxy = |material| in SEE units) ---");
                let mut cum = 0u64;
                for i in (0..12).rev() {
                    cum += s.dualnet_evals[i];
                    let n = s.dualnet_evals[i];
                    if n == 0 { continue; }
                    let lo = i * 100;
                    let label = if i == 11 { "1100+ ".to_string() } else { format!("{:>4}-{:<4}", lo, lo + 99) };
                    eprintln!("proxy {}: {:>8} evals ({:5.2}%)  qualify-if-thresh<=this: {:5.1}%  mean|eval|={:>5}  near-eq {:4.1}%",
                        label, n, 100.0 * n as f64 / total as f64,
                        100.0 * cum as f64 / total as f64,
                        s.dualnet_abseval[i] / n.max(1),
                        100.0 * s.dualnet_neareq[i] as f64 / n.max(1) as f64);
                }
            }
        }
        {
            let bn: u64 = s.b_probe_nodes.iter().flatten().sum();
            if bn > 0 && s.moves_searched > 0 {
                eprintln!("--- Fail-low nodes by depth/margin/quiets (total moves searched {}) ---", s.moves_searched);
                let dnames = ["d<=4", "d5-8", "d>=9"];
                let mnames = ["m<50", "m50-150", "m150-300", "m>=300"];
                for d in 0..3 {
                    for m in 0..4 {
                        let mut nodes = 0u64; let mut quiets = 0u64; let mut late = 0u64;
                        for q in 0..4 {
                            let i = m * 4 + q;
                            nodes += s.b_probe_nodes[d][i];
                            quiets += s.b_probe_quiets[d][i];
                            late += s.b_probe_late[d][i];
                        }
                        if nodes > 0 {
                            eprintln!("B[{} {}]: {} nodes, {} quiets tried ({:.2}/node), late(>2) {} = {:.2}% of all moves",
                                dnames[d], mnames[m], nodes, quiets,
                                quiets as f64 / nodes as f64, late,
                                100.0 * late as f64 / s.moves_searched as f64);
                        }
                    }
                }
            }
        }
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
    eprintln!("SEE prune:      {:>6.1}/Kn", s.see_prunes as f64 / kn);
    eprintln!("LMR searches:   {:>6.1}/Kn", s.lmr_searches as f64 / kn);
    eprintln!("QS nodes:       {:>5.1}%", s.qnodes as f64 / total_nodes as f64 * 100.0);
    eprintln!("First-move cut: {:>5.1}%", if s.beta_cutoffs > 0 { s.first_move_cutoffs as f64 / s.beta_cutoffs as f64 * 100.0 } else { 0.0 });

    eprintln!("Total nodes:    {:>8}", total_nodes);

    // RFP false-positive audit table (only when RFP_AUDIT=1 produced data).
    let audit_total: u64 = s.rfp_audit_attempts.iter().sum();
    if audit_total > 0 {
        let fp_total: u64 = s.rfp_audit_fp.iter().sum();
        eprintln!("--- RFP False-Positive Audit (null-verified, NMP R formula) ---");
        eprintln!("depth | audited  | rejected | FP rate");
        for d in 0..24 {
            let a = s.rfp_audit_attempts[d];
            if a == 0 { continue; }
            let f = s.rfp_audit_fp[d];
            eprintln!("{:>5} | {:>8} | {:>8} | {:>6.2}%", d, a, f, f as f64 * 100.0 / a as f64);
        }
        eprintln!("TOTAL | {:>8} | {:>8} | {:>6.2}%", audit_total, fp_total,
            fp_total as f64 * 100.0 / audit_total as f64);
    }

    // Eval-path decomposition — supports the "evals/node" investigation.
    // Reports how search
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

fn reset_bench_position_state(info: &mut SearchInfo) {
    info.clear_persistent_histories();
    info.tt.new_search();
}

/// Test-only NNUE net locator. Since the PeSTO fallback was removed, any test
/// that drives a real `search()` must load a net. Prefers `CODA_TEST_NET`, then
/// the production net named by `net.txt` (what `make net` downloads). Returns
/// `None` when no net is present locally so callers can skip gracefully.
#[cfg(test)]
pub(crate) fn test_net_path() -> Option<String> {
    std::env::var("CODA_TEST_NET").ok()
        .filter(|p| std::path::Path::new(p).exists())
        .or_else(|| {
            let url = std::fs::read_to_string("net.txt").ok()?;
            let name = url.trim().rsplit('/').next()?.trim().to_string();
            (!name.is_empty() && std::path::Path::new(&name).exists()).then_some(name)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_move_index_distinguishes_promotions() {
        let from = square(0, 6);
        let to = square(0, 7);
        let indices = [
            root_move_index(make_move(from, to, FLAG_PROMOTE_N)),
            root_move_index(make_move(from, to, FLAG_PROMOTE_B)),
            root_move_index(make_move(from, to, FLAG_PROMOTE_R)),
            root_move_index(make_move(from, to, FLAG_PROMOTE_Q)),
            root_move_index(make_move(from, to, FLAG_NONE)),
        ];
        for i in 0..indices.len() {
            for j in i + 1..indices.len() {
                assert_ne!(indices[i], indices[j]);
            }
        }
        assert!(indices.iter().all(|&idx| idx < ROOT_MOVE_TABLE_SIZE));
    }

    #[test]
    fn fixed_depth_smp_selection_keeps_main_candidate() {
        let main_move = make_move(square(4, 1), square(4, 3), FLAG_NONE);
        let helper_move = make_move(square(3, 1), square(3, 3), FLAG_NONE);
        let cands = [
            SmpCandidate {
                mv: main_move,
                score: 0,
                depth: 12,
                sel_depth: 18,
                ponder: NO_MOVE,
                is_main: true,
            },
            SmpCandidate {
                mv: helper_move,
                score: 100,
                depth: 8,
                sel_depth: 14,
                ponder: NO_MOVE,
                is_main: false,
            },
            SmpCandidate {
                mv: helper_move,
                score: 100,
                depth: 8,
                sel_depth: 14,
                ponder: NO_MOVE,
                is_main: false,
            },
        ];

        assert_eq!(select_smp_candidate(&cands, false), Some(0));
        assert_ne!(select_smp_candidate(&cands, true), Some(0));
    }

    #[test]
    fn helper_refresh_updates_syzygy_probe_depth() {
        let mut main = SearchInfo::new(1);
        let mut helper = create_helper_info(&main);
        main.tb_probe_depth = 19;

        refresh_helper_per_go(&mut helper, &main);

        assert_eq!(helper.tb_probe_depth, 19);
    }

    #[test]
    fn persistent_history_reset_clears_all_history_tables() {
        let mut info = SearchInfo::new(1);
        info.dirty_persistent_histories_for_test();

        info.clear_persistent_histories();

        info.assert_persistent_histories_clear_for_test();
    }

    #[test]
    fn bench_position_reset_starts_each_fen_with_clean_histories() {
        let mut info = SearchInfo::new(1);
        info.dirty_persistent_histories_for_test();
        let gen_before = info.tt.current_generation();

        reset_bench_position_state(&mut info);

        info.assert_persistent_histories_clear_for_test();
        assert_eq!(
            info.tt.current_generation(),
            gen_before.wrapping_add(1),
            "bench reset must start a fresh TT generation per FEN"
        );
    }

    /// 50-move eval scaling helper. Locks in both the formula (linear decay
    /// via `(200 - hm)/200`, so the eval is HALVED — not zeroed — at the
    /// 50-move claim cliff, matching engine consensus) and the sentinel
    /// preservation that downstream search relies on when comparing against
    /// `-INFINITY` and mate scores.
    #[test]
    fn test_apply_halfmove_scale() {
        // Linear decay from full at hm=0 to half at the draw horizon.
        assert_eq!(apply_halfmove_scale(100, 0), 100);
        assert_eq!(apply_halfmove_scale(100, 25), 87);
        assert_eq!(apply_halfmove_scale(100, 50), 75);
        assert_eq!(apply_halfmove_scale(100, 75), 62);
        assert_eq!(apply_halfmove_scale(100, 99), 50);
        assert_eq!(apply_halfmove_scale(100, 100), 50);
        // Sign preserved.
        assert_eq!(apply_halfmove_scale(-400, 50), -300);
        // Saturation past 100 (hm > 100 is normally intercepted as a draw,
        // but `hm` is clamped to 100 so it settles at the half-eval floor
        // and never flips sign).
        assert_eq!(apply_halfmove_scale(100, 150), 50);
        // Zero in → zero out at any hm.
        assert_eq!(apply_halfmove_scale(0, 50), 0);
        // Sentinel scores are not scaled — comparisons against -INFINITY /
        // mate-adjusted scores in the search body rely on this.
        assert_eq!(apply_halfmove_scale(-INFINITY, 50), -INFINITY);
        assert_eq!(apply_halfmove_scale(MATE_SCORE - 5, 99), MATE_SCORE - 5);
        assert_eq!(apply_halfmove_scale(-(MATE_SCORE - 5), 99), -(MATE_SCORE - 5));
    }

    /// Regression guard against corrhist fortress drift.
    /// Correction history can self-reinforce into a phantom ±0.45 in
    /// low-material locked/fortress positions (opposite-coloured bishops,
    /// blocked pawns) that Stockfish/Obsidian all read as 0 — the
    /// raw NNUE is fine (~0); it is corrhist railing in the low-signal
    /// regime. Training the update against the CORRECTED eval (the residual)
    /// rather than the raw one is what keeps it convergent. These four
    /// positions must stay near 0; a regression blows them back out to ±0.45.
    ///
    /// Needs an NNUE net (the drift is a corrhist-on-NNUE effect), so it skips
    /// gracefully when no net is present — honours `CODA_TEST_NET`, else
    /// `net.nnue`, else a `net-v*.nnue`.
    #[test]
    fn test_corrhist_fortress_no_drift() {

        use crate::board::Board;
        crate::init();

        // Hermetic net selection: use the PRODUCTION net defined by net.txt
        // (the basename of its URL — what `make net` downloads and what the
        // build embeds), or a CODA_TEST_NET override. No random/first-match
        // `.nnue` fallback: prod nets are hash-named (`net-<HASH>.nnue`, no
        // generation prefix), so any filename heuristic would silently pick a
        // stale net and make this a false guard. Skip if the prod net isn't
        // present locally (run `make net`).
        let net_path = match super::test_net_path() {
            Some(p) => p,
            None => { eprintln!("Skipping fortress-drift test: no NNUE net found"); return; }
        };

        let mut info = SearchInfo::new(16);
        info.silent = true;
        if let Err(e) = info.load_nnue(&net_path) {
            eprintln!("Skipping fortress-drift test: net load failed: {}", e);
            return;
        }

        // (FEN, label) — all are dead draws; truth is ~0.
        let fortresses = [
            ("8/b7/7p/1k6/8/1B3K1P/8/8 w - - 2 54",    "blocked h3/h6 OCB (game aIosNgFS mv54)"),
            ("8/8/7p/2b5/6B1/7P/5k2/2K5 w - - 18 62",  "blocked OCB (game aIosNgFS mv62)"),
            ("8/k7/P2K4/8/5p2/P7/1b2B3/8 b - - 26 67", "7-man OCB down a pawn (game PYiXcgdg mv67)"),
            ("8/k2K4/P7/P7/8/8/8/2b2B2 b - - 0 94",    "K+B+2P vs K+B fortress (game PYiXcgdg mv94)"),
        ];
        let limits = SearchLimits {
            depth: 16,
            fixed_depth: true,
            infinite: true,
            ..SearchLimits::new()
        };
        for (fen, label) in fortresses {
            let mut board = Board::from_fen(fen);
            info.clear_correction_history();   // fresh corrhist per position
            info.history.clear();
            info.tt.new_search();
            info.nodes = 0;
            info.last_flushed_nodes.set(0);
            info.global_nodes.store(0, Ordering::Relaxed);
            let _ = search(&mut board, &mut info, &limits);
            let s = info.last_score;
            assert!(s.abs() <= 25,
                "corrhist fortress-drift regression: {} scored {} cp (want ~0, |s|<=25). \
                 Correction history is re-inflating a dead-drawn/locked position — \
                 see docs/corrhist_fortress_drift_2026-07-06.md",
                label, s);
        }
    }

    /// Singular extensions set `info.excluded_move[ply]` during verification
    /// search and MUST clear it after. A leak would silently corrupt the
    /// next search iteration (subsequent SE would skip, or move loops would
    /// skip a random move).
    ///
    /// There must be no early-return path between the set and the clear.
    #[test]
    fn test_excluded_move_cleared_after_search() {

        use crate::board::Board;

        crate::init();
        let net_path = match super::test_net_path() {
            Some(p) => p,
            None => { eprintln!("Skipping excluded-move test: no NNUE net found"); return; }
        };
        let mut info = SearchInfo::new(16);
        info.silent = true;
        if let Err(e) = info.load_nnue(&net_path) {
            eprintln!("Skipping excluded-move test: net load failed: {}", e);
            return;
        }

        let mut board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

        let limits = SearchLimits {
            depth: 8, // enough to hit SE at SE_DEPTH threshold
            fixed_depth: true,
            movetime: 0,
            wtime: 0, btime: 0, winc: 0, binc: 0,
            movestogo: 0, nodes: 0, infinite: false,
            movetime_floor: 0,
            min_think_ms: 0,
            abs_clock: 0,
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
    /// (a) move the entry in the direction of `scaled_err`,
    /// (b) respect the bound ±CORR_HIST_LIMIT,
    /// (c) apply proportional gravity (saturates at the bound),
    /// (d) be symmetric for positive vs negative errors (equal magnitude
    ///     updates produce equal magnitude changes from 0).
    #[test]
    fn corr_entry_update_basics() {
        // (d) Symmetry from zero.
        let mut pos = 0i32;
        let mut neg = 0i32;
        update_corr_entry(&mut pos, 20, 4);   // scaled_err=+20 (err 4 × w 5)
        update_corr_entry(&mut neg, -20, 4);  // scaled_err=-20
        assert_eq!(pos, -neg, "symmetric updates from zero: pos={}, neg={}", pos, neg);
        assert!(pos > 0, "positive err must raise entry: got {}", pos);

        // (a) Directional.
        let mut e = 0i32;
        update_corr_entry(&mut e, 6, 4);
        assert!(e > 0, "err > 0, weight > 0 → entry must rise, got {}", e);

        // (b) Bounded at ±CORR_HIST_LIMIT.
        let mut e = 0i32;
        for _ in 0..10000 {
            update_corr_entry(&mut e, 1_000_000, 1); // saturate hard
        }
        assert!(e <= CORR_HIST_LIMIT, "entry must stay ≤ LIMIT, got {}", e);
        assert!(e >= -CORR_HIST_LIMIT, "entry must stay ≥ -LIMIT, got {}", e);

        // (c) Proportional gravity: repeated same-sign updates saturate,
        //     don't grow without bound.
        let mut e = CORR_HIST_LIMIT / 2;
        let before = e;
        update_corr_entry(&mut e, 1, 4);
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
        update_corr_entry(&mut e, 0, 4);
        assert_eq!(e, 500, "zero err must not change entry");

        let mut e = -500i32;
        update_corr_entry(&mut e, 0, 4);
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
        assert_eq!(corrected_eval(&info, &board, raw, 0), raw,
            "zero tables must give corrected == raw");

        // === Part 1: direct entry check after one update ===
        // NOTE: passing `raw` as the corrected baseline is legitimate HERE only
        // because the tables start zeroed, so corrected == raw at this point.
        // In the search the caller passes the CORRECTED eval — see the doc
        // comment on update_correction_history. Do not copy this call shape as
        // evidence that a raw eval is what the function expects.
        update_correction_history(&mut info, &board, raw + 400, raw, 20, 0);

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
            // `raw` as baseline is valid here only while the tables are zeroed;
            // see the note at the first call site in this module's tests.
            update_correction_history(&mut info, &board, raw + 400, raw, 20, 0);
        }

        // === Part 2: corrected_eval drift ===
        let corrected_after = corrected_eval(&info, &board, raw, 0);
        assert!(
            corrected_after > raw,
            "after sustained positive-err updates, corrected eval must rise: \
             raw={} corrected={}",
            raw, corrected_after
        );

        // Reference position: pawn_hash / non_pawn_key / minor / major
        // are entirely different from the test fen, so any match would
        // be a 1/16384 random collision — extremely unlikely.
        let other_corrected = corrected_eval(&info, &other, raw, 0);
        let drift = (other_corrected - raw).abs();
        assert!(drift < 100,
            "unrelated position should see near-zero drift, got {} (raw {})",
            other_corrected, raw);
    }

    /// Instant-reply gate — DOUBLE-PONDERHIT CASCADE GUARD. If the opponent
    /// instant-replied out of their own ponderhit, our go-ponder→ponderhit
    /// window can be ~1ms. A ponderhit arriving <10ms after `go ponder`
    /// must NEVER instant-emit, REGARDLESS of what the depth / fail-low
    /// flags claim (they could be stale in a degenerate interleaving).
    #[test]
    fn ponder_instant_reply_double_ponderhit_guard() {
        // The structural elapsed floor: no flag combination may pass below
        // MIN_PONDER_ELAPSED_FOR_INSTANT_MS, even with an absurd tiny soft
        // and a maximal (stale) depth claim.
        for elapsed in 0..MIN_PONDER_ELAPSED_FOR_INSTANT_MS {
            assert!(
                !should_instant_reply(elapsed, 0, 100, false, 4),
                "elapsed={}ms < {}ms must never instant-emit (soft=0, depth=100)",
                elapsed, MIN_PONDER_ELAPSED_FOR_INSTANT_MS
            );
            assert!(!should_instant_reply(elapsed, 1, i32::MAX, false, 4));
        }

        // Big-increment scenario: at 60s+10s the intended soft is multiple
        // seconds — a 1ms ponder must lead to a full normal think via the
        // elapsed >= soft condition alone.
        let (soft, _hard, _max, _floor) =
            compute_tm_budgets(60_000, 10_000, 0, 100, 20, true);
        assert!(soft >= 1000,
            "test premise: big-inc soft should be seconds, got {}ms", soft);
        assert!(!should_instant_reply(1, soft, 64, false, 4),
            "1ms ponder with multi-second soft must not instant-emit");

        // Depth floor backstops degenerate tiny-soft cases even past the
        // structural elapsed floor.
        assert!(!should_instant_reply(50, 20, MIN_PONDER_DEPTH_FOR_INSTANT - 1, false, 4),
            "depth below MIN_PONDER_DEPTH_FOR_INSTANT must block instant reply");
        // Fresh-search reset state (depth=0) always blocks.
        assert!(!should_instant_reply(5000, 20, 0, false, 4));
    }

    /// P1 instant-reply gate — fires when the pondered time covers the soft
    /// budget, the ponder search is deep enough, and the root is settled;
    /// a root fail-low revokes it (SF pattern).
    #[test]
    fn ponder_instant_reply_fires_when_budget_covered() {
        // elapsed >= soft, depth >= floor, not failing low → instant.
        assert!(should_instant_reply(5000, 3000, 15, false, 4));
        assert!(should_instant_reply(3000, 3000, MIN_PONDER_DEPTH_FOR_INSTANT, false, 4));
        // Root failing low revokes the instant reply.
        assert!(!should_instant_reply(5000, 3000, 15, true, 4));
        // Budget not yet covered → keep thinking.
        assert!(!should_instant_reply(2249, 3000, 15, false, 4));
    }

    #[test]
    fn test_instant_reply_stability_gate() {
        // Unstable ponder (stab 0) needs 1.71x soft; settled (4+) needs 0.75x.
        // elapsed exactly = soft qualifies only from stability >= 2 (0.90x).
        assert!(!should_instant_reply(3000, 3000, 15, false, 0));
        assert!(!should_instant_reply(3000, 3000, 15, false, 1));
        assert!(should_instant_reply(3000, 3000, 15, false, 2));
        assert!(should_instant_reply(3000, 3000, 15, false, 4));
        // Unstable qualifies once elapsed covers the inflated threshold.
        assert!(should_instant_reply(5130, 3000, 15, false, 0));
        assert!(!should_instant_reply(5129, 3000, 15, false, 0));
        // Stability index clamps at 4.
        assert!(should_instant_reply(2250, 3000, 15, false, 9));
        // Depth floor.
        assert!(!should_instant_reply(5000, 3000, MIN_PONDER_DEPTH_FOR_INSTANT - 1, false, 4));
    }

    /// P3 — the Ponder-on +25% optimum bump: applied to opt only (hard/max/
    /// floor identical), clamped to hard, and OFF by default (no-ponder
    /// behavior bit-identical).
    #[test]
    fn ponder_opt_bump_only_when_ponder_on() {
        for (t, inc, mtg, fm) in [
            (60_000u64, 600u64, 0u32, 20u16),
            (10_000, 100, 0, 8),
            (300_000, 0, 0, 30),   // no-inc sudden death
            (120_000, 2000, 0, 1),
            (60_000, 0, 40, 15),   // movestogo
        ] {
            let (opt_np, hard_np, max_np, floor_np) =
                compute_tm_budgets(t, inc, mtg, 100, fm, false);
            let (opt_p, hard_p, max_p, floor_p) =
                compute_tm_budgets(t, inc, mtg, 100, fm, true);
            assert_eq!(hard_np, hard_p, "hard must be ponder-independent");
            assert_eq!(max_np, max_p, "max must be ponder-independent");
            assert_eq!(floor_np, floor_p, "floor must be ponder-independent");
            // Bump is +25% pre-clamp; opt_np is the same pre-clamp value
            // un-bumped, so the relation holds whether or not hard clamps.
            let expected = (opt_np + opt_np * 25 / 100).min(hard_np).max(1);
            assert_eq!(opt_p, expected,
                "ponder-on opt at t={} inc={} mtg={} fm={}: got {}, want {}",
                t, inc, mtg, fm, opt_p, expected);
            assert!(opt_p >= opt_np);
        }
    }

    /// P2 — PonderhitCreditPct defaults to the -1 sentinel = INERT = full
    /// 100% charge of pondered time (the Option C 50% default is retired;
    /// explicit sets still work for local A/B).
    #[test]
    fn ponderhit_credit_default_is_full_charge() {
        let saved = PONDERHIT_CREDIT_PCT.load(Ordering::Relaxed);
        PONDERHIT_CREDIT_PCT.store(-1, Ordering::Relaxed);
        assert_eq!(ponderhit_credit_pct(), 100, "sentinel must mean full charge");
        PONDERHIT_CREDIT_PCT.store(50, Ordering::Relaxed);
        assert_eq!(ponderhit_credit_pct(), 50, "explicit set must be honored");
        PONDERHIT_CREDIT_PCT.store(0, Ordering::Relaxed);
        assert_eq!(ponderhit_credit_pct(), 0);
        PONDERHIT_CREDIT_PCT.store(saved, Ordering::Relaxed);
        // Fresh-binary default is the sentinel.
        assert_eq!(saved, -1, "shipping default must be the -1 sentinel");
    }

    /// Regression guard for the PV-print legality check. The pv_table can carry
    /// a STALE sibling-line move — e.g. a king move from a square the king
    /// occupied in a different branch. Printing it verbatim emits an "Illegal
    /// PV move" warning from the match runner, which is a latent forfeit class.
    /// The guard stops the printed PV at the first move that fails
    /// is_pseudo_legal + is_legal
    /// on the running pv_board. This test asserts that exact predicate: a move
    /// legal in a sibling position is rejected against the current one.
    #[test]
    fn pv_print_rejects_stale_sibling_move() {
        use crate::board::Board;
        crate::init();

        // Two positions reachable in the same search tree. In `sib` the white
        // king is on g2 and Kf3 (g2f3) is legal. In `cur` the king has moved
        // to h2 — g2f3 is now illegal (no piece on g2). A stale pv_table entry
        // from the `sib` branch would be `g2f3`.
        let sib = Board::from_fen("1R6/8/8/6pN/6P1/2k4p/3b2K1/5b2 w - - 0 1");
        let cur = Board::from_fen("1R6/8/8/6pN/6P1/2k4p/3b3K/5b2 w - - 0 1");

        // The stale move, encoded raw from/to (g2 -> f3), as it would sit in
        // pv_table after the sibling line wrote it.
        // g2 = file 6, rank 1; f3 = file 5, rank 2 (0-indexed).
        let g2 = crate::types::square(6, 1);
        let f3 = crate::types::square(5, 2);
        let stale = crate::types::make_move(g2, f3, 0);

        // In the sibling position it IS legal (sanity: the scenario is real).
        assert!(
            crate::movepicker::is_pseudo_legal(&sib, stale)
                && sib.is_legal(stale, sib.pinned(), sib.checkers()),
            "Kg2-f3 must be legal in the sibling position (king on g2)"
        );

        // In the current position the guard MUST reject it — this is exactly
        // the predicate the PV-print loop uses before make_move/print.
        let guard_passes = crate::movepicker::is_pseudo_legal(&cur, stale)
            && cur.is_legal(stale, cur.pinned(), cur.checkers());
        assert!(
            !guard_passes,
            "stale g2f3 must be rejected when the king is on h2 (no piece on g2)"
        );
    }
}
