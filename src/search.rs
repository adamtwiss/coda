//! Negamax alpha-beta search with iterative deepening, PVS, aspiration windows, and Lazy SMP.
//! Features: NMP, RFP, LMR, LMP, futility, SEE pruning,
//! singular extensions, cuckoo cycle detection, correction history.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
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
/// ply-indexed arrays (pv_table, moved_piece_stack, etc.).
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
    (NMP_BASE_R_10X, 77, 20, 80, 15.0, true),
    // Ceiling lifted from 60 → 200 (audit 2026-05-20): SPSA at 55, 90%
    // from min, only ~9% headroom. Symmetric to a floor pin — gradient
    // clamped at the top. Lifting lets SPSA find the true optimum.
    (NMP_DEPTH_DIV_10X, 60, 10, 200, 15.0, true),
    (NMP_EVAL_DIV, 101, 50, 400, 17.5, true),
    (NMP_EVAL_MAX_10X, 32, 10, 60, 5.0, true),
    // Lifted 74 → 120 (eff 8 → 12, toward consensus 14-16): at 74 the verify
    // gate sat below the old min-depth gate, so 100% of NMP cutoffs paid a
    // verification re-search — NMP never had a cheap cutoff. #1901 measured
    // verify=120 alone as neutral/slightly positive, supporting this direction.
    // With min-depth de-gated to 3, depths 3-11 now get the classic unverified
    // cutoff; 12+ verify (zugzwang guard).
    (NMP_VERIFY_DEPTH_10X, 110, 40, 200, 20.0, true),
    (RFP_DEPTH, 19, 2, 20, 2.0, true),
    // Floors lifted to 0 (audit 2026-05-20): both pinned within ~10% of floor.
    (RFP_MARGIN_IMP, 21, 0, 150, 6.0, true),
    (RFP_MARGIN_NOIMP, 33, 0, 200, 7.5, true),
    // Root-depth-aware RFP relaxation (single-set, self-adapts STC<->LTC):
    // demand MORE static-eval confidence to RFP-cut as the OVERALL search
    // depth grows past RFP_ROOT_THRESH (diminishing-returns of depth — the
    // marginal ply is cheap at LTC so deep pruning trades blindness for
    // worthless depth). Inactive at STC (root_depth < thresh) by construction
    // -> STC-neutral; relaxes deep RFP at LTC. SPSA tunes both.
    (RFP_ROOT_THRESH, 17, 6, 30, 1.5, true),
    (RFP_ROOT_COEF, 24, 0, 150, 7.5, true),
    // Additional depth-local RFP relaxation: current main already scales RFP
    // by overall root depth; this term raises the margin for high remaining
    // depth regardless of TC. Consensus engines either cap RFP around d9-11
    // or use a quadratic/deepening margin so static eval does not keep
    // cheaply pruning d12+ nodes.
    (RFP_DEEP_KNEE_10X, 45, 40, 170, 20.0, true),
    (RFP_DEEP_LINEAR, 43, 0, 200, 10.0, true),
    (RFP_DEEP_QUAD_10X, 60, 0, 800, 50.0, true),
    // Razoring (re-added 2026-06-11, audit T2.6). Consensus band:
    // Obsidian 352/d<=5, Berserk 214/d<=5, Clover 145/d<=2, Integral
    // 393/d<=4, Stormphrax ~290/d<=4.
    (RAZOR_MULT, 290, 100, 500, 20.0, true),
    (RAZOR_DEPTH_10X, 40, 10, 80, 5.0, true),
    // Futility margin widened after the pruning audit: 80/110 H1'd at STC
    // (#2018 +3.5) and was flat at LTC (#2019), with focused SPSA #2020
    // converging back to 81.5/109.0. Keep depth/threat gates unchanged.
    (FUT_BASE, 69, 0, 200, 9.0, true),
    (FUT_PER_DEPTH, 64, 40, 250, 10.5, true),
    (FUT_LMR_DEPTH, 13, 6, 24, 2.0, true),
    // HIST_PRUNE_DEPTH_10X / HIST_PRUNE_MULT removed 2026-06-02 — see hist-prune
    // removal block in main negamax body for rationale (three H0 SPRTs).
    (SEE_QUIET_MULT, 22, 5, 80, 3.75, true),
    // Low-increment TM multiplier ceiling (2026-06-18). The factor product
    // (stability×fail-low×forced×subtree×score-trend, up to ~13.8×) is only
    // clamped for no_inc; at increments that are SMALL RELATIVE TO THE CLOCK
    // it ran uncapped, so a complex middlegame drew deep on a RUN of moves the
    // increment can't refill -> clock drained by early middlegame -> flag
    // (lichess rapid 10+1 = 600s+1s). The discriminator is increment relative
    // to the per-move budget: inc_cover = inc / (timeLeft/mtg). Cap by an
    // inc_cover-scaled ceiling: cmin at inc_cover->0 (starved), rising to cmax
    // (≈ uncapped) at inc_cover >= TM_INC_COVER_REF/100.
    //   inc_cover ≈ 0.04 at lichess 600+1 (capped);  0.24 at OB STC 10s+0.1s
    //   and 0.4 at 600+10 (both ~uncapped) — so OB STC / LTC / big-inc are
    //   untouched, only true low-inc-vs-clock (rapid) is throttled.
    // NB: an earlier ABSOLUTE-inc form (inc/12000) crushed OB STC (inc 100ms
    // -> 1.6× cap) and lost 24 Elo (#2075). _10X ceilings are /10.
    // core: false — and NOT because TM is "invisible at STC": it is one of the
    // highest-leverage STC levers in BOTH directions. The Phase-13 TM rework was
    // +135 self-play / ~+75 x-engine at 10+0.1 (#1568, biggest single recent
    // gain); a bad TM form lost -24 (#2075). It's kept OUT of the routine --core
    // pruning sweep *because* it's that high-leverage and deployment-critical:
    // TM is tuned deliberately, TC-matched and cross-engine/ponder-validated,
    // not perturbed incidentally by a broad ~33-iter/param STC core retune where
    // a noisy TM movement could regress lichess. Still UCI-loadable for
    // deliberate TM tunes; just not swept by --core.
    (TM_INC_COVER_REF, 20, 5, 60, 4.0, false),
    (TM_MULT_CEIL_MIN_10X, 15, 10, 40, 2.0, false),
    (TM_MULT_CEIL_MAX_10X, 130, 40, 140, 8.0, false),
    // Cross-thread best-move-instability TM factor (SF port, 2026-07-05).
    // factor = BASE/1000 + MULT/1000 * (Σ per-thread bmc)/n_threads, applied
    // to the soft budget only at Threads>1. Defaults are SF's 1.088 / 2.315
    // (search.cpp:519); will want a focused TM-cluster retune-on-branch since
    // Coda's TM is Viridithas-shaped. Fixed-point /1000 for the sub-integer
    // precision these multiplicative constants need. Not --core (TM is tuned
    // deliberately, TC-matched, never swept by the STC core retune).
    // BASE defaults to 1000 (=1.0), NOT SF's 1088: SF's base is balanced
    // against SF's OWN factor product; on Coda's already-calibrated product a
    // >1.0 base would add a blanket ~9% time to EVERY position (settled or not),
    // contaminating the raw test. At 1.0 the factor is neutral when the pool
    // agrees and only scales UP on genuine cross-thread churn — the retune can
    // lift BASE if beneficial. MULT starts at SF's 2.315.
    (TM_BMC_INSTAB_BASE, 1000, 900, 1500, 25.0, false),
    (TM_BMC_INSTAB_MULT, 2315, 500, 4000, 100.0, false),
    // Subtree-factor base (docs/tm_spikiness_experiment_2026-07-10.md).
    // Factor = (BASE/100 - best_move_node_fraction) * 1.4, floor 0.55 (the
    // floor cannot bind at the 1.62 default — frac would need to exceed 1.23
    // — so default behavior is identical to the pre-tunable formula).
    // Phase-0 showed the factor inflating 66% of moves (neutral only at
    // frac=0.905), but the re-center probe (130) was −48 Elo differential vs
    // SF17/SF18 while EVEN in Coda self-play: "our best move dominated our
    // own search" is a bad confidence proxy against stronger opponents, and
    // the up-bias is insurance. DO NOT put this in any OB SPSA — self-play
    // cannot see the cross-engine cost; deliberate cross-engine RR only.
    (TM_SUBTREE_BASE_100, 162, 100, 180, 4.0, false),
    // Low-inc absolute single-move ceiling (2026-06-22, overspend PART2).
    // inc_cover (PART1) caps the factor MULTIPLIER, so adjusted_soft stays
    // ~11% of clock — but a single deep iteration that starts just under
    // adjusted_soft runs uninterrupted (the soft check only fires BETWEEN
    // iterations) until the mid-iteration hard check stops it at hard = 46%
    // of clock. At low-inc-ratio TCs (lichess 600+1 == OB 60+0.1, inc/base
    // ~0.0017) the engine reaches deep enough for one iteration to span
    // soft->hard, so a single move eats 46% of the clock, repeatedly,
    // geometric-draining the clock (lichess J4tHOcvR/OO0ADWTA/ozALf371).
    // Fix: lower the hard/max ceiling directly when the increment is small,
    // keyed on the (constant) increment so it never flips mid-game:
    //   inc_ceiling = inc * TM_INC_HARD_MULT + TM_INC_HARD_FLOOR_MS
    // MULT=30, FLOOR=10s leaves standard OB TCs (10+0.1/40+0.4/60+0.6) and
    // rich TCs (600+10) essentially untouched (the 46%/60% windows still
    // bind), while capping 600+1 at 40s (was 276s) and 60+0.1 at 13s.
    (TM_INC_HARD_MULT, 30, 0, 120, 4.0, false),
    (TM_INC_HARD_FLOOR_MS, 10000, 0, 60000, 1000.0, false),
    // No-inc adaptive mtg divisor (2026-07-02): base assumed moves-to-go
    // and growth rate once a game outlives that assumption — see
    // compute_tm_budgets for the full derivation. Tuned by focused SPSA
    // #2444 (1000 iters, 30+0 zero-inc): base 40->34.4, growth 100->94.3.
    (NO_INC_MTG_BASE, 34, 20, 80, 4.0, false),
    (NO_INC_MTG_GROWTH_PCT, 94, 0, 200, 10.0, false),
    (LMR_HIST_DIV, 16349, 2000, 100000, 4900.0, true),
    // 2026-05-18 audit (outlier #2 deep-dive): capture-LMR was using a
    // step function (±1 at |capt_hist|>2000), while quiet-LMR uses
    // continuous `hist_score / LMR_HIST_DIV`. Obsidian uses continuous
    // `R -= hist/(isQuiet?LmrQuietHistoryDiv:LmrCapHistoryDiv)` with
    // LmrQuietHistoryDiv=9621, LmrCapHistoryDiv=5693 (cap divisor ~60%
    // of quiet — single-source capt_hist needs smaller divisor for
    // equivalent reduction magnitude). Coda's quiet div is 7736; same
    // ratio gives ~4500. Defaulting 5000 as a starting point.
    (LMR_HIST_DIV_CAP, 2682, 1000, 20000, 1500.0, true),
    (LMR_C_QUIET, 149, 40, 300, 13.0, true),
    (LMR_C_CAP, 182, 80, 350, 12.5, true),
    // 3-DOF LMR shape reform (Titan Track B + Zeus's shape-vs-shift point,
    // 2026-07-07). BASE shifts the curve's intercept (Berserk +0.23,
    // Obsidian dBase, SF +1027/1024 — never validly tested on Coda: the
    // atlas/lmr-base-offset branches truncated it away pre-fractional).
    // DECAY_NUM changes the curve's SHAPE: multiplicative all-node
    // inflation r += r*NUM/(256d+285) — proportionally MORE reduction
    // shallow, LESS deep (SF's 272 with r in 1024ths ≈ +11.7% at d8,
    // +3.4% at d30). Replaces the flat +1-ply all-node bump, whose flat
    // profile is the wrong shape (the specific mechanism behind SF's flat
    // deep EBF per docs/ltc_regime_investigation_2026-07-07.md Q3).
    // Seeded at 700 to roughly preserve current shallow all-node
    // reduction at typical r (~250c, d8: +75c vs old flat +100c).
    (LMR_BASE_CENTI, 27, 0, 120, 6.0, true),
    (LMR_ALLNODE_DECAY_NUM, 681, 0, 1600, 80.0, true),
    // Explicit cut-node LMR bump (P1.1 / #2065). Cut nodes reduce by
    // LMR_CUTNODE_BUMP (+1 more with no TT move); all-nodes keep +1. Default 2
    // is a halfway step toward SF's larger cut-node reduction; SPSA can push it.
    (LMR_CUTNODE_BUMP_CENTI, 258, 100, 500, 40.0, true),
    // Reckless LMR correction battery (T1.1, docs/reckless_audit_2026-07-06.md).
    // Sub-ply centi-ply terms — need the fractional LMR accumulator to express.
    // Reseeded at HALF the Reckless-converted values (full: 100/45/32/41)
    // after #2594/#2596 H0'd at full strength — our ln(d)·ln(m) base keeps
    // its move-count term (Reckless deleted theirs) so their constants
    // double-count; ranges run to 0 so SPSA can kill dead terms.
    (LMR_WINBETA_CENTI, 43, 0, 250, 12.0, true),
    (LMR_TTALPHA_CENTI, 21, 0, 150, 8.0, true),
    (LMR_TTDEPTH_CENTI, 14, 0, 150, 8.0, true),
    (LMR_EXPECT_MULT, 24, 0, 120, 6.0, true),
    // cutoff_count LMR terms (T1.2, docs/reckless_audit_2026-07-06.md).
    // Child ply failed high >2 times under this node -> reduce late moves
    // more (+extra at non-PV all-nodes). Defaults = Reckless's tuned
    // values reseeded at half (full: 112/39) — see battery note above.
    // Threshold >2 fixed (SF uses >3) — not a knob.
    (LMR_CUTOFF_CNT_CENTI, 58, 0, 250, 12.0, true),
    (LMR_CUTOFF_ALLNODE_CENTI, 13, 0, 150, 8.0, true),
    // 2026-05-09 cross-engine port (Tier 5.1): SF gates SE at >=6+ttPv,
    // Reckless at >=5+ttPv. Coda's 4 fires SE at shallower depth where
    // singular_depth is too low to judge singularity reliably. Bumping
    // 4 → 6 first; ttPv add deferred to a follow-up if H1.
    (SE_DEPTH_10X, 43, 40, 200, 20.0, true),
    (ASP_DELTA, 11, 5, 30, 1.5, false),
    (ASP_SCORE_DIV, 33378, 8000, 50000, 2100.0, false),
    // 2026-05-09 cross-engine bisect (Tier 5.3a): SF/Obsidian/Reckless all
    // use LMP_BASE=3 with the same `(BASE + d²)/(2 - improving)` formula.
    // Coda's 9 is 3× consensus at d=1: allows 5-10 quiets vs SF's 2-4.
    // Bisecting 9 → 5 first.
    (LMP_BASE_10X, 32, 10, 150, 20.0, true),
    (LMP_DEPTH_10X, 72, 40, 200, 20.0, true),
    // Root-depth-aware LMR relaxation (single-set, self-adapts STC<->LTC):
    // reduce LESS as the OVERALL search depth grows past LMR_ROOT_THRESH
    // (diminishing returns — at LTC the reduced re-search is cheap vs the
    // budget and a wrong reduction costs more). Inactive at STC by
    // construction -> STC-neutral. SPSA tunes both.
    (LMR_ROOT_THRESH, 15, 6, 30, 1.5, true),
    (LMR_ROOT_COEF_10X, 81, 0, 800, 40.0, true),
    (BAD_NOISY_MARGIN, 80, 30, 150, 6.0, true),
    (PROBCUT_MARGIN, 129, 80, 300, 11.0, true),
    // Consensus ProbCut shape (Stockfish/Reckless/Viridithas/Alexandria):
    // improving positions can use a lower verification beta, while
    // non-improving nodes keep the safer base margin. Default 117-27=90cp
    // when improving, matching the promising low-margin STC signal.
    (PROBCUT_MARGIN_IMP, 47, 0, 120, 8.0, true),
    // Root-depth-aware conservative ProbCut:
    // #2021 found PROBCUT_MARGIN=170 / MIN_DEPTH_10X=45 wins at STC,
    // while #2022 rejected it at LTC. Add that conservative offset at
    // shallow root depths, then fade back to current main as root depth grows.
    (PROBCUT_ROOT_THRESH, 16, 8, 28, 1.5, true),
    (PROBCUT_ROOT_FADE_10X, 29, 10, 120, 10.0, true),
    (PROBCUT_ROOT_MARGIN, 67, 0, 120, 8.0, true),
    (HINDSIGHT_THRESH, 185, 50, 400, 17.5, true),
    (UNSTABLE_THRESH, 310, 50, 500, 22.5, false),
    (QS_DELTA_MARGIN, 349, 100, 500, 20.0, true),
    // 24 -> 5 with the T2.10 counting fix: the old counter charged
    // delta/SEE-pruned moves against the budget, so SPSA detuned the cap
    // to near-off. Counting searched-only, consensus is 3 (Obsidian/
    // Reckless) to ~"2 extra" (SF moveCount > 2).
    (QS_MAX_CAPTURES, 5, 2, 32, 2.0, false),
    (CORR_W_PAWN, 286, 100, 600, 25.0, true),
    // Floor lifted from 50 → 0 (audit 2026-05-20): pinned at 63, 4% from floor.
    (CORR_W_NP, 98, 0, 400, 17.5, true),
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
    (CORR_W_CONT, 100, 0, 400, 18.5, true),
    // Transition (zobrist-delta) correction weight (Cinder idea): correction
    // keyed by hash(ply-1) ^ hash(ply) — a hash of the last move IN CONTEXT
    // (from+to+captured+side), richer than cont_corr's [piece][to]. Captures
    // "this structural CHANGE tends to be mis-evaluated."
    (CORR_W_TRANS, 63, 0, 400, 18.5, true),
    (FH_BLEND_DEPTH_10X, 33, 0, 80, 15.0, false),
    // Re-expose 4 hardcoded search constants (audit 2026-05-21).
    // All bench-neutral at current defaults.
    //
    // TT_DAMP_TT_WEIGHT: weight of tt_score in TT-LOWER non-PV cutoff score
    // dampening. Formula: (W*tt_score + beta) / (W+1). Old hardcoded W=3.
    (TT_DAMP_TT_WEIGHT_10X, 30, 10, 100, 5.0, false),
    // PROBCUT_TT_DEPTH_SLACK: TT depth must be >= current depth - SLACK for
    // ProbCut-TT-noshot to consider the entry. Old hardcoded 3.
    (PROBCUT_TT_DEPTH_SLACK, 3, 0, 10, 0.5, false),
    (HIST_BONUS_MULT, 290, 50, 400, 17.5, true),
    (HIST_BONUS_MAX, 1520, 500, 3000, 125.0, true),
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
    (CAP_HIST_MULT, 314, 50, 400, 17.5, true),
    (CAP_HIST_MAX, 1791, 500, 3000, 125.0, true),
    // Malus split (2026-06-11 move-ordering audit): 14/16 stronger engines
    // use SEPARATE malus constants (SF malus slope ~7x its bonus slope;
    // Obsidian goes the other way at 0.74x — the optimum is engine-specific
    // and only discoverable by tuning). Coda's malus was hardwired to
    // -bonus, so SPSA never had this axis (#1922 confirmed symmetric is
    // Coda's optimum at STC). Defaults track the live bonus values
    // (tune-#1915 era) so behavior == the tested -bonus parity.
    (HIST_MALUS_MULT, 330, 50, 900, 40.0, true),
    (HIST_MALUS_OFFSET, 24, 0, 400, 25.0, false),
    (HIST_MALUS_MAX, 1229, 500, 4000, 175.0, true),
    (CAP_HIST_MALUS_MULT, 302, 50, 900, 40.0, true),
    (CAP_HIST_MALUS_BASE, 42, 0, 400, 25.0, false),
    (CAP_HIST_MALUS_MAX, 1884, 500, 4000, 175.0, true),
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
    // Sibling-count history-bonus scaling (SF 645b636d). At non-PV cutoffs,
    // amplify the best move's bonus by (quiets+caps searched)/HIST_SIBLING_DIV:
    // a move that cut off after more competition proved itself more strongly.
    // SF default divisor 256.
    (HIST_SIBLING_DIV, 239, 64, 1024, 40.0, true),
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
    (DEXT_MARGIN_PV, 174, 50, 400, 15.0, true),
    (DEXT_MARGIN_QUIET, 15, 0, 100, 4.0, true),
    (DEXT_MARGIN_CORR, 21, 0, 64, 3.0, true),
    (DEXT_MARGIN_BASE, 24, -50, 150, 6.0, true),
    (DEXT_CAP, 13, 4, 32, 2.0, true),
    (QUIET_CHECK_BONUS, 14805, 2000, 30000, 1400.0, false),
    // SEE gate on the quiet check bonus (SF movepick.cpp: check bonus only
    // applies when see_ge(m, -75)). Without it Coda orders losing check-sacs
    // into the first-searched slot. Margin on Coda's pawn=100 SEE scale:
    // a check that loses more than this by SEE gets no ordering bonus.
    (QUIET_CHECK_SEE_MARGIN, 87, 0, 300, 12.0, true),
    (CORR_HIST_DIV, 450, 256, 4096, 192.0, true),
    // 4 -> 16 with T2.4: the floor-pin at 4 was calibrated for the
    // sign-only (err-clamped) regime; consensus weights ~depth uncapped.
    (CORR_UPDATE_WEIGHT_MAX, 17, 4, 48, 2.2, true),
    // Was 32 (tp10→3). Now FIXED-POINT. Default 30 → eff 3.0 ≡ old behavior.
    (CORR_BONUS_CAP_DIV_10X, 27, 10, 160, 15.0, false),
    (CORR_HIST_GRAIN_T, 13, 1, 32, 1.55, false),
    // Floor lifted from 10 → 0 (audit 2026-05-19): SPSA converged 25, ~2%
    // from the floor. Lifting allows exploration of looser clamps.
    // T2.4: CORR_HIST_ERR_MAX (±3cp input pre-clamp) replaced by output
    // scaling: bonus = err*(depth+1).min(W)/CORR_ERR_DIV, clamped at the
    // gravity cap only. Obsidian err*depth/8; SF err*depth*12/128.
    (CORR_ERR_DIV_10X, 50, 20, 640, 30.0, false),
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
    (NULL_THREAT_ESCAPE_BONUS, 8321, 0, 30000, 1000.0, false),
    (NMP_KING_ZONE_MAX_10X, 49, 20, 90, 15.0, true),
    // T2.1 (Titan's next_ideas 2026-04-21): undefended-piece NMP skip
    // threshold. Count our pieces with ≥1 enemy attacker AND zero of
    // our own defenders ("hanging"). If count >= this threshold, skip
    // NMP — opponent's free tempo is very likely to exploit the hanger.
    // Fits Titan's W2 pattern (binary signal gating a pruning decision).
    // Default 1 = skip NMP whenever any piece is hanging.
    // Min 1 (not 0): the gate is `undefended_count < tp10(this)`. Since
    // undefended_count >= 0, a value of 0 makes the condition impossible
    // and disables NMP entirely — SPSA/ablation hitting this min would
    // accidentally test "NMP off" while labeled "undefended guard off".
    (NMP_UNDEFENDED_MAX_10X, 16, 1, 50, 10.0, true),
    // T2.3 (next_ideas_2026-04-21): mobility-delta quiet-ordering weight.
    // Bonus applied in movepicker quiets = (to_mobility - from_mobility) × this.
    // Default 32 = ±256 typical range, additive to history (~1000s scale).
    (MOBILITY_DELTA_WEIGHT, 34, 0, 256, 8.0, false),
    (PROBCUT_KING_ZONE_MAX_10X, 70, 20, 90, 15.0, true),
    // Was 38 (tp10→4). Now FIXED-POINT. Default 40 → eff 4.0 ≡ old behavior.
    (LMR_THREAT_DIV_10X, 13, 10, 50, 15.0, true),
    // Was 68 (tp10→7). Now FIXED-POINT. Default 70 → eff 7.0 ≡ old behavior.
    (LMR_KING_PRESSURE_DIV_10X, 62, 20, 90, 15.0, true),
    // Reduce later moves more once this node has already raised alpha N times
    // (Viridithas #431 alpha_raises). Fixed-point ×10: reduction += raises *
    // VALUE/10. Only fires at PV nodes (cut nodes break on the first fail-high
    // before alpha is raised). Default 10 = +1.0 reduction per prior alpha-raise.
    (LMR_ALPHA_RAISE_10X, 11, 0, 40, 5.0, true),
    (FUT_THREATS_MARGIN, 22, 0, 200, 10.0, true),
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
    // xray-SE: when the TT move is from an x-ray blocker square (moving it
    // uncovers our slider's attack on an enemy), this flat bonus is
    // SUBTRACTED from singular_beta (`singular_beta = tt_score - depth -
    // xray_bonus`). That LOWERS singular_beta → WIDENS the SE margin →
    // STRICTER singularity test → FEWER extensions on x-ray-blocker TT moves,
    // not more. (Empirically good: #604 H1 +1.1, and SPSA drives the value UP
    // away from the 0 floor — so do NOT "fix" the sign. The earlier comment
    // here described the mechanism backwards.) Ordering signal for these moves
    // is delivered separately in movepicker (#502, +52).
    (SE_XRAY_BLOCKER_MARGIN_10X, 48, 0, 400, 20.0, true),
    // 2026-05-19 audit: floor was pinned at 10 (=1.0 effective), preventing
    // SPSA from exploring below 1× even though SPSA had repeatedly driven
    // the value to the floor across tunes. Widened to allow 0× (full disable)
    // so SPSA can find the genuine optimum. CLAUDE.md previously claimed
    // "3× in move ordering" — stale; corrected to "1× current SPSA basin".
    (CONT_HIST_MULT_10X, 20, 0, 80, 15.0, true),
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
    (LMR_ENDGAME_PIECES_10X, 46, 45, 90, 15.0, true),
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
    (IIR_MIN_DEPTH_10X, 43, 5, 100, 15.0, true),          // was hardcoded 4; tune #743 converged to 2 (strong signal)
    // ProbCut floor lifted from 30 → 10 (audit 2026-05-19): SPSA at 32,
    // ~2% from floor. Lifting to 10 (eff 1) allows exploration of more
    // aggressive ProbCut activation.
    (PROBCUT_MIN_DEPTH_10X, 21, 10, 120, 15.0, true),     // was hardcoded 5 (ProbCut activation gate)
    (PROBCUT_ROOT_MIN_DEPTH_10X, 27, 0, 80, 8.0, true),
    (SEE_CAP_DEPTH_10X, 86, 30, 150, 15.0, true),         // was hardcoded 6 (SEE capture prune depth cap)
    // Capture-SEE prune margin, SF-shaped (search.cpp): margin = depth*MULT +
    // capt_hist*HIST/1024, prune if SEE < -margin. Was sharing the hardcoded
    // SEE_MATERIAL_SCALE=215 (a QS-delta constant) with NO history term, giving
    // a flat 2.15 pawn/depth margin — ~2.5× wider than SF's 0.84 pawn/depth, so
    // Coda pruned far fewer bad captures. The capt-hist term is load-bearing:
    // it protects historically-good captures (which produce cutoffs) so the
    // base can be lowered without over-pruning them (a naive base-only drop to
    // 130 cost +17% bench nodes). Base 110 (1.1 pawn/depth, toward SF's 0.84);
    // HIST 11 ≈ SF's 34/1024 rescaled for Coda's ±16384 capt-hist. Audit #3.
    (SEE_CAP_MULT, 80, 40, 250, 12.0, true),
    (SEE_CAP_HIST, 9, 0, 40, 2.0, true),
    (BAD_NOISY_DEPTH_10X, 89, 40, 150, 15.0, true),       // was hardcoded 4 (BNFP depth cap)
    // Second pass — additional gates exposed for the feature-utility
    // audit tune. Widened ranges allow SPSA to reach disable-endpoint
    // values where appropriate (per feedback_spsa_as_feature_utility_diagnostic).
    // De-gated 75 → 25 (eff 8 → 3) with the RFP-before-NMP reorder: with RFP
    // running first, shallow NMP only sees nodes static pruning couldn't cut,
    // removing the free-cutoff interception that killed #1904. SPSA had pushed
    // this to 8 as compensation for NMP-first ordering + per-cutoff verify cost.
    (NMP_MIN_DEPTH_10X, 55, 20, 200, 15.0, true),              // was hardcoded 3 (NMP activation gate, 2 sites)
    // Floor lifted from 10 → 0 (audit 2026-05-20): pinned at 25, 8% from floor.
    // 1 -> 17 (eff 0 -> 2, consensus floor): tune #1959 on the post-T1.2
    // trunk. The diagnostic was seeded at eff 2 and SPSA HELD (17.1) rather
    // than reverting to the old floor-pin at 0 — the pin was compensation
    // for the stale prior_reduction signal fixed by #1939, not signal.
    (HINDSIGHT_MIN_DEPTH_10X, 39, 0, 200, 15.0, true),
    // Net output scale in percent (eval-scale normalization experiment,
    // 2026-06-12). Final NNUE eval is multiplied by PCT/100. Different
    // nets train to very different natural scales (eval RMS 219-369
    // measured across same-recipe S200 runs) while all cp-denominated
    // search margins are calibrated to prod's scale — this knob lets a
    // probe rescale a candidate net to prod's scale (e.g. 127 = dual-s200
    // RMS 254 -> baseline 323) to de-confound net-vs-net SPRTs. 100 = off.
    (EVAL_SCALE_PCT, 100, 50, 200, 5.0, false),
    // Fail-low prior-countermove cont-hist bonus, % of history_bonus(depth)
    // (SF fail-low history harvesting, simple core — audit 2026-07-05 T1#2).
    (FAIL_LOW_PREV_BONUS_PCT, 60, 0, 150, 15.0, false),
    // Cross-MOVE score-trend TM coefficient (×1e-4). Folds the deterioration
    // across MOVES (prev-`go` final score − current running score) into the
    // score-trend multiplier, giving more time when the position has been
    // worsening over the game horizon — the regime where LTC games are lost.
    // Complements the within-search drop term (fixed 0.0025). Default matches
    // that scale (25 → 0.0025). TM change: validate via local cross-engine RR.
    (CROSS_MOVE_TREND, 25, 0, 150, 8.0, false),
);

// Demoted loose knobs (2026-05-22 cross-tune analysis): SPSA drift dominated
// signal, so removed from SPSA surface to improve SNR for the rest. Values
// frozen at their pre-demotion defaults. Bench-neutral; UCI-invisible.
pub static FH_BLEND_OFFSET: AtomicI32 = AtomicI32::new(1);
pub static SE_TT_DEPTH_SLACK: AtomicI32 = AtomicI32::new(3);
pub static MVV_CAP_MULT: AtomicI32 = AtomicI32::new(28);
// Demote-batch 2 (2026-05-23): 5 more NONCORE_QUIET from cross-tune analysis
// — all moved <20% under #1419 noise. Same rationale as batch 1.
pub static SEE_MATERIAL_SCALE: AtomicI32 = AtomicI32::new(215);
pub static QS_SEE_THRESHOLD: AtomicI32 = AtomicI32::new(-26);
pub static CAP_HIST_BASE: AtomicI32 = AtomicI32::new(42);
pub static LMR_COMPLEXITY_DIV: AtomicI32 = AtomicI32::new(152);
pub static TT_CUTOFF_HALFMOVE_MAX: AtomicI32 = AtomicI32::new(89);

/// Post-ponderhit budget credit: PERCENT of elapsed ponder time deducted from
/// the fresh post-hit think budget. HISTORY: Option C (2026-05-31) defaulted
/// this to 50 ("bank half the ponder time"). The 2026-07-05 ponder diagnosis
/// (docs/ponder_diagnosis_2026-07-05.md) showed 50% credit SATURATES to the
/// 50ms floor at STC (any ponder >= 2×soft zeroes the budget) and the realized
/// spend was then iteration-quantized bleed up to hard+500ms grace — the
/// dominant cause of the ~50 Elo ponder deficit vs SF. The replacement policy
/// is FULL charge for pondered time (budgets fixed at `go ponder`, SF
/// timeman model), made profitable by its two compensators: the
/// stopOnPonderhit-style instant reply (`should_instant_reply`) and the
/// ponder-on +25% optimum bump (`compute_tm_budgets`).
///
/// This knob is kept ONLY for local A/B comparability: default is the -1
/// sentinel = INERT (full 100% charge). Explicitly setting 0..=100 via
/// `setoption name PonderhitCreditPct` re-enables fractional crediting
/// (0 reproduces the pre-P13 full-fresh-budget behavior, 50 reproduces
/// Option C).
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
/// `compute_tm_budgets` (SF timeman.cpp:134 semantics — applied on EVERY
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

/// FL-EXT (2026-07-05, fail-low extension tail): hard-frame extension per
/// root fail-low event in the post-ponderhit frame, in percent of the hard
/// budget — the single step of the aspiration fail-low factor
/// (1 + 0.34·min(2, fl), SF/Viridithas shape) applied to the post-hit
/// deadlines. The 2026-07-05 ponder diagnosis measured SF spending >1s
/// post-hit on 3.3% of moves (its fail-low re-thinks) vs our 0.0% — the P2
/// cap clipped exactly those. Consts, DELIBERATELY NOT tunables: OB cannot
/// ponder, SPSA would detune them on noise; sweep in a local ponder
/// gauntlet only.
pub const PH_FL_HARD_EXT_PCT: u64 = 34;
/// Max fail-low deadline extensions per post-hit search (SF's min(2, fl)).
pub const PH_FL_MAX_EXTENSIONS: u32 = 2;
/// Minimum root depth for a during-post-hit fail-low to trigger a deadline
/// extension. Shallow aspiration-window misses (d4-8) are routine noise and
/// burned the whole extension budget within milliseconds in v1 (mechanism
/// run 2026-07-05: events at now=2ms, tail still 0.0%); only a fail-low at
/// a real search frontier signals genuine destabilization.
pub const PH_FL_MIN_DEPTH: i32 = 10;

/// stopOnPonderhit-class instant-reply decision (SF search.cpp:563-571
/// pattern, evaluated at the ponderhit instead of during pondering — our
/// clock doesn't tick while pondering, so the budgets the move would have
/// are computable at either point and the handler already has all inputs).
/// Instant-emit the pondered bestmove iff:
///   - the pondered time already covers the soft budget the move would have
///     been given (`elapsed >= intended_soft`), AND
///   - the ponder search completed a real search (depth floor), AND
///   - the root is not currently failing low (SF search.cpp:411-418: a root
///     fail-low revokes the instant reply — spend extra time exactly when
///     the pondered conclusion destabilized), AND
///   - the elapsed window is not a double-ponderhit cascade artifact (see
///     MIN_PONDER_ELAPSED_FOR_INSTANT_MS).
/// Stability-scaled soft threshold for the instant reply (percent of the
/// intended soft the pondered elapsed must cover, indexed by the ponder
/// search's best-move stability). SAME SHAPE as the dynamic-TM
/// STABILITY_TABLE [1.71, 1.20, 0.90, 0.80, 0.75] — SF arms stopOnPonderhit
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
/// precision and retain decimal progress across tune cycles. See
/// memory feedback_floor_pin_tunables_cross_recipe.md for rationale.
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
pub static FEAT_PROBCUT: AtomicBool = AtomicBool::new(true); // re-enabled after fixing missing qsearch filter, SEE threshold, and excluded_move guard
pub static FEAT_LMR: AtomicBool = AtomicBool::new(true);
pub static FEAT_LMP: AtomicBool = AtomicBool::new(true);
pub static FEAT_FUTILITY: AtomicBool = AtomicBool::new(true);
pub static FEAT_SEE_PRUNE: AtomicBool = AtomicBool::new(true); // confirmed: -17 Elo without (clean CPU retest)
pub static FEAT_BAD_NOISY: AtomicBool = AtomicBool::new(true); // confirmed: -26 Elo without (retested without CPU contention)
pub static FEAT_EXTENSIONS: AtomicBool = AtomicBool::new(true);
pub static FEAT_FH_BLEND: AtomicBool = AtomicBool::new(true); // gates fail-high score blending (replaces dead FEAT_ALPHA_REDUCE — see below)
// FEAT_ALPHA_REDUCE removed 2026-06-06: it gated the `alpha_raised` LMR
// adjustment that was deleted in 21c8f7f (Apr 7, "LMR simplify", H0'd
// -2..-4). The flag was orphaned — never read (.load) anywhere — so its
// env var / disable_all entry silently did nothing and the "-4 Elo" comment
// was stale (described the removed feature). Repurposed the slot to give
// fail-high score blending (previously unablatable) a real ablation flag.
pub static FEAT_IIR: AtomicBool = AtomicBool::new(true);
pub static FEAT_HINDSIGHT: AtomicBool = AtomicBool::new(true); // confirmed: -18 Elo without (clean CPU retest)
pub static FEAT_CORRECTION: AtomicBool = AtomicBool::new(true);
pub static FEAT_PVS: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_CUTOFF: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_NEARMISS: AtomicBool = AtomicBool::new(true);
pub static FEAT_TT_STORE: AtomicBool = AtomicBool::new(true);
// Static-eval cache: reuse TT-stored static_eval instead of calling NNUE.
// Ablate (NO_TT_STATIC_EVAL=1) to test whether skipping evals hurts more
// than it helps via deeper lazy-replay gaps (fatter threat/finny applies).
pub static FEAT_TT_STATIC_EVAL: AtomicBool = AtomicBool::new(true);
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
    FEAT_TT_STORE.store(false, Ordering::Relaxed); FEAT_QS_CAPTURES.store(false, Ordering::Relaxed);
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
    // TREESTATS parity counters (tree-shape comparison vs the instrumented
    // SF build, 2026-07-11; dumped by the UCI `treestats` command in the
    // same line format as the instr-stockfish patch). Bucket 0 = qsearch;
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

/// Forced-move detection state (Viridithas pattern, set by `detect_forced_move`).
/// Once a position is classified at the root, the result is sticky for the rest of
/// the search — both the verification's TT pollution and the result itself are
/// monotonic. `None` is the default; once `Weak` or `Strong` is observed, the TM
/// multiplier scales down accordingly.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
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
/// Stop-time snapshot of the Phase-13 dynamic-TM factor values, captured on
/// the last iteration that evaluated the factor product. TMDebug-gated
/// diagnostics only — never read by search logic. (TM spikiness Phase 0,
/// docs/tm_spikiness_experiment_2026-07-10.md)
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
    /// Interior Syzygy WDL probe hits this search (main thread). Cosmetic —
    /// surfaced in the UCI `tbhits` field so TB usage is observable. Root
    /// TB-move hits are reported separately at their own info lines.
    pub tb_hits: u64,
    pub global_nodes: std::sync::Arc<AtomicU64>,  // aggregate nodes across SMP threads
    /// Cross-thread best-move-changes, PER THREAD (SF port). Thread `i` writes
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
    /// Last per-thread node count flushed into global_nodes (delta flushing,
    /// TM audit 2026-06-13 A4). Cell: should_stop takes &self.
    last_flushed_nodes: std::cell::Cell<u64>,
    pub silent: bool,  // suppress UCI output (for datagen)
    pub stats: PruneStats,
    // Eval-path decomposition counters (see docs/coda_vs_reckless_nps_*.md).
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
    /// EXTERNAL stop flag (TM audit 2026-06-13 A2): set ONLY by the UCI
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
    /// Reset at search start. Consumed by the Phase 13 fail-low factor
    /// `1.0 + 0.34 * min(2, asp_fail_low)` applied to both opt and hard
    /// windows (Viridithas pattern). The Phase 9 thresholded mechanism
    /// (TM_ASP_THRESHOLD/TM_ASP_MULT_10X) was removed with Phase 13.
    tm_asp_fail_low: u32,
    /// Cumulative count of aspiration fail-highs in the current search.
    /// Currently diagnostic-only; not used by TM yet (asymmetric vs
    /// fail-low because fail-high means we found a BETTER move than
    /// expected — already captured by score_factor's upward sense).
    tm_asp_fail_high: u32,
    /// Cumulative count of root best-move changes between iterations,
    /// reset at search start. Since Phase 13 this is DIAGNOSTIC-ONLY
    /// (TMDebug output) — the upward multiplier it used to drive was
    /// dropped. Candidate for re-use as an SF/Reckless-style
    /// within-iteration instability factor (TM audit 2026-06-13, B3).
    tm_best_move_changes: u32,
    /// Forced-move detection state (Viridithas pattern). Set after an ID iteration
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
    /// True when our increment is exactly zero (lichess 3+0, 60+0, 180+0,
    /// tournament 40/15). Tightened gate (2026-05-23) replacing the earlier
    /// `soft_floor == 0` form, which incidentally captured STC 10+0.1
    /// (inc == overhead == 100ms → soft_floor=0 by coincidence) and disabled
    /// the forced-move detector at our SPRT TC, costing ~3 Elo at STC for
    /// reasons orthogonal to the lichess no-inc fix.
    tm_no_inc: bool,
    /// Phase 13 (2026-05-26, Viridithas-shape rewrite): the absolute max time
    /// we will ever spend on a single move, computed as 60% of our_clock.
    /// Replaces Phase 10h's hard×0.5 cap with Viridithas's max_bank_usable
    /// pattern. Factors multiply soft up against this — no separate cap.
    tm_max_time: u64,
    /// Our increment (ms) for the current search — feeds the low-increment
    /// multiplier ceiling. 0 when there is no increment.
    tm_our_inc: u64,
    /// Our remaining clock (ms, post-overhead) for the current search — feeds
    /// the inc-relative-to-budget ceiling discriminator.
    tm_time_left: u64,
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
    /// the fractional `hard` cap can't prevent at low clock. "Never forfeit with
    /// time on the clock" (lichess MJ4lEpXF no-inc flag, loss-55 inc flag).
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
    /// at ponderhit to REVOKE the instant reply (SF search.cpp:411-418
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
    /// of two constant sets (Adam directive 2026-06-13).
    pub root_depth: i32,
    /// TMDebug-only stop-time snapshot of the dynamic-TM factors (see TmDbg).
    tm_dbg: TmDbg,
    /// Ply barrier for NMP verification: prevents NMP from re-triggering
    /// inside its own verification subtree (all peers: Reckless, Alexandria,
    /// Stormphrax use nmpMinPly / nmp_min_ply). Default 0 = no barrier. (audit B1)
    pub nmp_min_ply: i32,
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
    /// Per-ply beta-cutoff counter (SF cutoffCnt / Reckless cutoff_count,
    /// T1.2 docs/reckless_audit_2026-07-06.md). Incremented at the fail-high
    /// site; each node clears its GRANDCHILD slot on entry so
    /// `cutoff_count[ply+1]` reflects only fail-highs under this node's own
    /// subtree. Read in LMR: a child ply that keeps failing high means
    /// refutations come easy there — reduce late moves more. +4 padding
    /// allows unconditional ply+2 indexing (Reckless pads +16).
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
    /// Continuation correction history: [piece][to_square]
    // Paired continuation correction (H1, 2026-07-10): [prev_piece][prev_to][cur_piece][cur_to],
    // go_piece 1-12 (slot 0 unused). Read/updated at ply-2 and ply-4 offsets.
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
        // Flush local node count to global counter every 4096 nodes.
        // Delta-tracked (TM audit 2026-06-13 A4): the old flat
        // fetch_add(4096) double-counted when should_stop was re-invoked
        // from ID-loop sites at an unchanged boundary-resting count.
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
        // Exact accounting (2026-07-06): include this thread's UNFLUSHED local
        // delta, not just the 4096-granular global counter — the stale-global
        // check made `go nodes N` overshoot by up to 4095 nodes (+23% at
        // N=10000, measured vs SF/Reckless which enforce exactly; it inflated
        // Coda's fixed-node RR results). Helpers still contribute at flush
        // granularity (bounded T*4096 slack, documented above); the main
        // thread — the one that matters at T=1 fixed-node testing — is now
        // node-exact. Zero cost unless max_nodes > 0 (never set on OB).
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
            // (lichess MJ4lEpXF no-inc flag, loss-55 inc flag.)
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
                // single deep iteration bleed up to ~4s at 10+0.1 = 40% of
                // the base clock (ponder diagnosis 2026-07-05, max 4147ms).
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

                // ---------------------------------------------------------------
                // THREAT-ACCUMULATOR VERIFICATION (C1 diagnostic, 2026-07-10).
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
                        // Collect scratch feature indices (same 256 cap as refresh).
                        let mut indices = [0usize; 256];
                        let mut ni = 0usize;
                        let mut overflow = false;
                        crate::threats::enumerate_threats(
                            &board.pieces, &board.colors, &board.mailbox,
                            occ, pov, mirrored,
                            |idx| {
                                if idx < net.num_threat_features {
                                    if ni < 256 { indices[ni] = idx; ni += 1; }
                                    else { overflow = true; }
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
/// Obsidian/Reckless/Berserk `(200 - hm)/200`, PlentyChess `(293 - rule50)/293`
/// — which all HALVE (not zero) the eval at the 50-move cliff. The previous
/// `(100 - hm)/100` was a 2× outlier that nulls a won eval to 0.00 at the
/// cliff; Coda's own conversion study (docs/conversion_failure_study
/// _2026-06-29.md) traced won-position draws to exactly that over-damping.
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

/// TT-cutoff child-consistency verification (SF search.cpp:873-892, ported
/// via the 2026-07-05 SF search audit, Tier 1 #3). Before trusting a DEEP
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

/// Paired continuation correction (H1, 2026-07-10). Index by the LAST move
/// (ply-1) and select the subtable by the move at ply-2 AND ply-4, summing both
/// — the SF/Reckless/Viridithas 2-D continuation form, replacing Coda's flat
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
            if pp != 0 && pp < 13 && pt < 64 {
                sum += info.cont_corr[pp][pt][cur_p][cur_t] as i64;
            }
        }
    }
    sum
}

/// Compute the correction value alone (the centipawn delta corrhist would apply
/// to raw eval). Used by SE-margin formulas to gate extension confidence on
/// |correction| — extend less on uncertain (drifting) evals (Reckless pattern).
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
#[inline]
fn corrected_eval(info: &SearchInfo, board: &Board, raw_eval: i32, ply: usize) -> i32 {
    let stm = board.side_to_move as usize;

    // Pawn correction
    let pawn_idx = (board.pawn_hash as usize) & (CORR_HIST_SIZE - 1);
    let pawn_corr = info.pawn_corr[stm][pawn_idx] as i64;

    // Non-pawn corrections (per color)
    let white_np_idx = (board.non_pawn_key[WHITE as usize] as usize) & (CORR_HIST_SIZE - 1);
    let white_np_corr = info.np_corr[stm][WHITE as usize][white_np_idx] as i64;
    let black_np_idx = (board.non_pawn_key[BLACK as usize] as usize) & (CORR_HIST_SIZE - 1);
    let black_np_corr = info.np_corr[stm][BLACK as usize][black_np_idx] as i64;

    // Continuation correction — paired 2-ply/4-ply (H1)
    let cont_corr = cont_corr_value(info, ply);

    // Transition correction (zobrist-delta of last move in context)
    let trans_corr = if !board.undo_stack.is_empty() {
        let last = &board.undo_stack[board.undo_stack.len() - 1];
        if last.mv != NO_MOVE {
            let trans_idx = ((board.hash ^ last.hash) as usize) & (CORR_HIST_SIZE - 1);
            info.trans_corr[stm][trans_idx] as i64
        } else { 0 }
    } else { 0 };

    // Weighted blend: pawn, whiteNP, blackNP, cont, transition (minor/major dropped 2026-05-19)
    let total_corr = (pawn_corr * tp(&CORR_W_PAWN) as i64 + white_np_corr * tp(&CORR_W_NP) as i64 + black_np_corr * tp(&CORR_W_NP) as i64
        + cont_corr * tp(&CORR_W_CONT) as i64 + trans_corr * tp(&CORR_W_TRANS) as i64) / tp(&CORR_HIST_DIV) as i64;
    // mat_damp (piece-count fortress guard) removed 2026-07-09: the residual
    // update baseline (finding #1) makes corrhist converge to the true (~0)
    // correction in low-signal positions, so the material band-aid is redundant.
    let adjusted = raw_eval + (total_corr as i32) / tp(&CORR_HIST_GRAIN_T);
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
fn update_correction_history(info: &mut SearchInfo, board: &Board, search_score: i32, raw_eval: i32, depth: i32, ply: usize) {
    // T2.4 consensus shape: feed the FULL error scaled by depth, clamping
    // only the resulting bonus (at the gravity cap, in update_corr_entry).
    // The old ±3cp err pre-clamp (CORR_HIST_ERR_MAX) made corrhist a
    // sign-only integrator — max update 21 vs cap ~341. No surveyed engine
    // clamps the input error: SF err*depth*12/128, Obsidian err*depth/8,
    // Reckless 142*depth*err/128, all clamped at the output only.
    let err = search_score - raw_eval;
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
        if cur_p != 0 && cur_p < 13 && cur_t < 64 {
            for off in [2usize, 4] {
                if ply >= off {
                    let pp = info.moved_piece_stack[ply - off] as usize;
                    let pt = info.moved_to_stack[ply - off] as usize;
                    if pp != 0 && pp < 13 && pt < 64 {
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
/// cells. 2026-05-31 audit (H3): prior `static mut [[i32; 64]; 64]` was UB
/// under Rust's memory model and ARM-visible inconsistent reads during
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
/// default behaviour the integer plies are bit-identical to the old tables
/// (fractional-LMR enabler, re-implementation of #2192 on the 2026-07 trunk).
pub const LMR_SCALE: i32 = 100;

pub fn init_lmr() {
    for depth in 1..64 {
        for moves in 1..64 {
            // Quiet table: C from tunable (default 130 = 1.30). CENTI-PLY.
            if depth >= 3 && moves >= 3 {
                let c = tp(&LMR_C_QUIET) as f64 / 100.0;
                // Additive base in exact centi (post-scale, so 20 = 0.20 plies
                // uniformly — NOT inside the float-to-int truncation, which is
                // the bug that made atlas/lmr-base-offset a no-op).
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
    // C8 audit LIKELY #35: share ponderhit_time so helpers respect the
    // ponderhit deadline set by the UCI thread. Previously helpers kept
    // their own AtomicU64 stuck at 0, so they ignored the ponderhit
    // deadline and only stopped when main set the shared stop flag —
    // burning CPU for the grace window on every ponderhit.
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
/// search would diverge at T>1 (the -8 class, OB #2539). Move-ordering history
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

    // Correction tables — copied for eval consistency (see fn-doc). ~260 KB.
    helper.pawn_corr.copy_from_slice(&main.pawn_corr[..]);
    helper.np_corr.copy_from_slice(&main.np_corr[..]);
    helper.cont_corr.copy_from_slice(&main.cont_corr[..]);
    helper.trans_corr.copy_from_slice(&main.trans_corr[..]);

    // pawn_hist is position-specific (indexed by pawn hash); a helper's
    // self-accumulated table carries toxic stale ordering across positions
    // (measured -8 at T=4, OB #2539), so it is cleared every go even in Stage 2.
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
/// `(nodes, best_move, score, completed_depth, ponder)`. Shared verbatim by
/// the per-`go` spawn path and the persistent thread pool so both produce
/// byte-identical helper behavior. `search_helper` ignores its `_limits`
/// (helpers take depth from `info.max_depth` and stop on the shared flag), so
/// a zeroed placeholder is passed.
pub(crate) fn helper_run(
    info: &mut SearchInfo,
    board: &mut Board,
    max_depth: i32,
    thread_id: usize,
) -> (u64, Move, i32, i32, Move) {
    prepare_helper_for_search(info, board);
    info.max_depth = max_depth;
    let placeholder = SearchLimits {
        depth: max_depth, movetime: 0, wtime: 0, btime: 0, winc: 0, binc: 0,
        movestogo: 0, nodes: 0, infinite: false, movetime_floor: 0,
        min_think_ms: 0, abs_clock: 0,
    };
    let mv = search_helper(board, info, &placeholder, thread_id);
    let ponder = if info.pv_len[0] >= 2 { info.pv_table[0][1] } else { NO_MOVE };
    (info.nodes, mv, info.last_score, info.completed_depth, ponder)
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
    // Phase 13 (2026-05-26): Viridithas-shape TM windows.
    //
    // Top-engine audit (2026-05-26) showed Coda's pre-Phase-13 TM was a
    // structural outlier among ~10 modern engines:
    //   - 6 multiplicative factors compounding to ~30× product (Viridithas: 4
    //     factors, ~9.5× max; Hobbes: 3 factors, ~3.3× max; SF/Obsidian/Plenty:
    //     0 factors, fully static)
    //   - Separate hard×0.5 cap (Phase 10h) as band-aid for the factor overflow
    //   - Hard window only ~9% of clock vs Viridithas's 46%
    // The diagnostic (TM_DIAG, 2026-05-26) showed 65% of TM iterations clamped
    // to the Phase 10h cap, blocking factors=4-11× legitimate signals.
    //
    // Phase 13 ports Viridithas's TM window structure:
    //   max_time = clock × 0.60 - overhead  (the only per-move ceiling)
    //   hard_time = clock × 0.46            (mid-search abort, clamped to max)
    //   opt_time = (clock/24 + inc × 0.94 - overhead) × 0.73, clamped to hard
    //
    // The factor multiplier applied in the dynamic TM block scales opt UP
    // (factors can hit ~9.5× max) toward hard/max — no separate cap needed.
    // This is the structural difference: a wide hard window with tight
    // factors, instead of Coda's tight hard window with wide factors + cap.
    //
    // Constants verbatim from Viridithas (per-mille / per-hundred):
    //   MAX_BANK_USABLE: 600 (60% of clock)
    //   HARD_WINDOW_FRAC: 46 (46% of clock)
    //   OPTIMAL_WINDOW_FRAC: 73 (73% of computed_window)
    //   INCREMENT_FRAC: 94 (94% of inc added to computed_window)
    //   DEFAULT_MOVES_TO_GO: 24 (sudden-death pacing assumption)
    //
    // Returns (opt, hard, max, soft_floor). soft_floor preserved at small
    // value (10ms) — Viridithas has no separate floor, but Coda's stockpile-
    // prevention sleep at line ~2520 still needs a non-zero value to be a
    // no-op for movetime-limited searches. The Phase 13 factor multiplier
    // can pull opt × multiplier well below any meaningful floor.
    let time_left = our_time.saturating_sub(overhead).max(1);
    // No-inc TCs require more conservative pacing. With inc, each move
    // costs only `inc` of net time (we regain inc per move). Without
    // inc, every spent second is gone forever. The default 25-moves-left
    // assumption produces ~7s/move on 3+0 (180s base) — but real games
    // run 40-80 moves, so 25-move pacing leaves the engine massively
    // out of time. Use 40 moves at no-inc to pace tighter.
    // Lichess game 1yV9VbAA: Coda at 3+0 spent 8-12s on early moves,
    // forfeited on time.
    // Phase 10a (2026-05-24): lower base optimum at moderate-inc TCs from
    // 25 → 35. Cross-engine review (9 engines) found Coda's base optimum
    // at moderate-inc is ~4.8% of remaining vs SF/Obsidian/Reckless ~2-3%.
    // Lower base gives multiplicative factors more headroom to produce
    // sharp spikes without forfeiting (top engines spike to 4-5× p95/p50
    // partly because they're spiking off a lower baseline). No-inc path
    // unchanged (moves_left=40 from earlier hotfix); movestogo path
    // unchanged (already uses the explicit movestogo count).
    // Viridithas-style windows. max_time is the absolute single-move ceiling.
    const MAX_BANK_USABLE_NUM: u64 = 600;
    const MAX_BANK_USABLE_DEN: u64 = 1000;
    const HARD_WINDOW_NUM: u64 = 46;
    const HARD_WINDOW_DEN: u64 = 100;
    const OPT_WINDOW_NUM: u64 = 73;
    const OPT_WINDOW_DEN: u64 = 100;
    const INC_FRAC_NUM: u64 = 94;
    const INC_FRAC_DEN: u64 = 100;
    const DEFAULT_MOVES_TO_GO: u64 = 24;

    // No-inc sudden-death TCs need a tighter ceiling. Viridithas's verbatim
    // 60%/46% windows are fine at moderate-inc (each spent ms gets refilled
    // by inc) but catastrophic at 3+0: lichess qiHdjT7k (2026-05-27) — at
    // move 6 with 166s clock, hard_time = 76s; a deep single iteration ran
    // to that ceiling, geometric-decayed to flag-fall by move 24.
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
        (time_left * MAX_BANK_USABLE_NUM / MAX_BANK_USABLE_DEN).min(inc_hard_ceiling).max(1)
    };
    let hard_time = if no_inc_sd {
        (time_left * 10 / 100).min(max_time).max(1)
    } else {
        (time_left * HARD_WINDOW_NUM / HARD_WINDOW_DEN).min(max_time).max(1)
    };

    // No-inc sudden death needs a higher moves-left assumption than
    // moderate-inc TCs: each spent ms is gone forever, and real 3+0 games
    // run 40-80 moves. Phase 13's verbatim Viridithas constant (24) was
    // calibrated for inc TCs where the inc term dominates pacing — at
    // no-inc the 24 produces an opt that's high enough for the factor
    // multiplier (up to ~6.5×) to consistently blow past hard_time,
    // making hard the binding constraint every move (uniform-spend
    // pattern, lichess MJ442247 / 3+0).
    //
    // Adaptive tightening (2026-07-02): a FIXED base assumption never
    // tightens as a game outlives it (move 80 still assumes "40 moves
    // left"). Diagnosed from real coda_bot Lichess losses (all "outoftime"
    // zero-inc bullet forfeits) and confirmed via local RR (Coda vs
    // Reckless/Obsidian/Berserk/Alexandria, 30+0, no adjudication): 0/320
    // forfeits for the 4 peer engines vs 7/320 for Coda, all preceded by
    // 70-88% of the clock burned by move ~60 in games running 130-220+
    // plies. Once fullmove exceeds NO_INC_MTG_BASE, grow the divisor by
    // NO_INC_MTG_GROWTH_PCT% of the overrun: effective_mtg = base +
    // growth_pct/100 * max(0, fullmove - base).
    //
    // A first attempt with fixed constants (base=40, growth=100%, OB
    // #2438) fully eliminated forfeits in local RR (0/320) but was a real
    // SPRT regression (-3.5 ±2.5, LLR -2.95 H0 at N=17,670 at 30+0) — the
    // mechanism works but over-tightens mid-game allocation even in games
    // that were never going to forfeit. Exposed as tunables instead and
    // SPSA-tuned (#2444, focused 2-param, 1000 iters, 30+0 zero-inc):
    // base 40->34.4, growth_pct 100->94.3 (both significant movement,
    // held steady across the whole tune). Applied here as new defaults;
    // re-verify forfeit-count + non-regression SPRT before merge.
    let no_inc_mtg_base = tp(&NO_INC_MTG_BASE).max(1) as u64;
    let no_inc_growth_pct = tp(&NO_INC_MTG_GROWTH_PCT).max(0) as u64;
    let no_inc_effective_mtg = no_inc_mtg_base
        + (fullmove as u64).saturating_sub(no_inc_mtg_base) * no_inc_growth_pct / 100;
    let mtg_divisor = if no_inc_sd { no_inc_effective_mtg.max(1) } else { DEFAULT_MOVES_TO_GO };

    let opt_time_base = if movestogo > 0 {
        // Movestogo: divisor is clamped to [2, default_mtg]. TM audit
        // 2026-06-13 (A4): the increment term was missing here — the
        // sudden-death branch credits 94% of inc but this one didn't,
        // systematically under-allocating ~0.7*inc/move at movestogo+inc
        // TCs (CCRL-style). Same INC_FRAC weighting as the SD branch.
        let divisor = (movestogo as u64).clamp(2, DEFAULT_MOVES_TO_GO);
        let computed = time_left / divisor + our_inc * INC_FRAC_NUM / INC_FRAC_DEN;
        (computed.min(max_time) * OPT_WINDOW_NUM / OPT_WINDOW_DEN).max(1)
    } else {
        // Sudden death (or with inc). Add 94% of inc to base computed window.
        let computed = time_left / mtg_divisor + our_inc * INC_FRAC_NUM / INC_FRAC_DEN;
        ((computed.min(max_time) * OPT_WINDOW_NUM / OPT_WINDOW_DEN).min(hard_time)).max(1)
    };

    // Phase 13.1 (2026-05-26): phase scaling (Reckless/Hobbes pattern).
    //
    // First Phase 13 RR at 30+0.25 showed -35 Elo and 2× opening overspend
    // (median 42% vs Coda.main's 24%). Root cause: with stability=0 at search
    // start, the multiplier is 2.5×; at opt_base=1.08s the per-move adj_soft
    // becomes 2.7s — across 10 opening moves that's 27s = entire 30+0.25
    // budget burnt in opening.
    //
    // Top engines (Reckless, Hobbes, Viridithas's other formula) embed phase
    // scaling in their base: `soft_scale = 0.024 + 0.042 × (1 - exp(-0.045 × fm))`
    // grows from ~0.024 at fm=0 to 0.066 at fm=40+. Ratio ~0.36 → 1.0.
    //
    // Apply the same Reckless-style exponential here as a multiplier:
    //   phase_mult = 0.22 + 0.78 × (1 - exp(-0.045 × fullmove))   (Phase 13.3)
    //     fm=1:  0.25×
    //     fm=5:  0.38×
    //     fm=10: 0.50×
    //     fm=20: 0.68×
    //     fm=40: ~0.87×
    // Skip when movestogo > 0 (movestogo path computes from explicit count).
    let opt_time = if movestogo > 0 {
        opt_time_base
    } else {
        // Phase 13.3: tighter floor 0.36 → 0.22, sharper opening discipline.
        // p13_2 still 66% of games over 35% opening-overspend threshold (main: 23%).
        // Lowering floor cuts fm=1 from 0.39× to 0.25×, bringing opening allocation
        // closer to Coda's prior calibrated level.
        let phase_mult = 0.22 + 0.78 * (1.0 - (-0.045 * fullmove as f64).exp());
        ((opt_time_base as f64) * phase_mult.clamp(0.22, 1.0)) as u64
    };
    // P3 (2026-07-05, ponder diagnosis): +25% optimum when the Ponder UCI
    // option is on — SF timeman.cpp:134-135 semantics. Pre-funding applied
    // on EVERY move when pondering is enabled: the average move is refunded
    // by the pondered time itself (full-charge model) and by the
    // stopOnPonderhit-style instant replies. Do NOT ship without those
    // compensators (fix plan P1/P2 — the pieces work as a set).
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
    // soft well below this — Viridithas has no enforced floor.
    let soft_floor: u64 = 10;

    (opt_time, hard_time, max_time, soft_floor)
}

#[allow(dead_code)]


/// Run Lazy SMP search: main thread + N-1 helper threads.
pub fn search_smp(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits, threads: usize) -> Move {
    // C8 audit LIKELY #37: advance TT generation here (before spawning
    // helpers) rather than inside search(). Previously helpers could
    // start writing TT entries with the old generation in the microsecond
    // window between spawn and main's new_search() call, leaving them
    // looking freshest in replacement. Main's search() no longer bumps;
    // single-thread path bumps here too for consistency.
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
    let main_move = search(board, info, limits);
    let main_score = info.last_score;
    let main_depth = info.completed_depth;

    // Signal all helpers to stop
    info.stop.store(true, Ordering::Relaxed);

    // Collect per-thread candidates: (move, score, depth, ponder, is_main).
    // Helpers now also return their 2nd PV move (ponder) so a winning helper
    // can hand uci.rs a consistent bestmove+ponder pair.
    struct Cand { mv: Move, score: i32, depth: i32, ponder: Move, is_main: bool }
    let mut total_nodes = info.nodes;
    let mut cands: Vec<Cand> = Vec::with_capacity(threads);
    let main_ponder = if info.pv_len[0] >= 2 { info.pv_table[0][1] } else { NO_MOVE };
    if main_move != NO_MOVE && main_depth > 0 {
        cands.push(Cand { mv: main_move, score: main_score, depth: main_depth, ponder: main_ponder, is_main: true });
    }
    for (helper_nodes, mv, score, depth, ponder) in crate::thread_pool::collect() {
        total_nodes += helper_nodes;
        if mv != NO_MOVE && depth > 0 {
            cands.push(Cand { mv, score, depth, ponder, is_main: false });
        }
    }
    info.nodes = total_nodes;

    // Nothing completed a real iteration — fall back to main's move (which may
    // itself be NO_MOVE only in pathological instant-stop cases).
    if cands.is_empty() {
        return main_move;
    }

    // Vote-based selection (SF/Obsidian/Plenty). weight = depth * (score -
    // min_score + 14): the +14 keeps the worst-scored thread's vote nonzero so
    // depth still matters on tied scores; ×depth makes shallow helpers count
    // less. Votes are summed per move across threads.
    let min_score = cands.iter().map(|c| c.score).min().unwrap();
    let mut votes: Vec<(Move, i64)> = Vec::with_capacity(cands.len());
    for c in &cands {
        let weight = c.depth as i64 * (c.score as i64 - min_score as i64 + 14);
        if let Some(entry) = votes.iter_mut().find(|(m, _)| *m == c.mv) {
            entry.1 += weight;
        } else {
            votes.push((c.mv, weight));
        }
    }
    let vote_of = |mv: Move| votes.iter().find(|(m, _)| *m == mv).map(|(_, w)| *w).unwrap_or(0);

    // Select the best THREAD, not just the max-vote move (SF get_best_thread):
    // prefer a proven win (shortest mate = highest score); otherwise switch to a
    // thread whose move has more votes (deeper on ties), but never onto a proven
    // loss. Picking a thread (vs a bare move) is what lets us carry a consistent
    // PV/ponder out — the previous `max_by_key(votes)` returned a move with no
    // owning thread, so on any vote-override uci.rs saw pv_table[0][0] != bestmove
    // and dropped the ponder entirely.
    let mut best = 0usize;
    for i in 1..cands.len() {
        let (cs, cmv, cd) = (cands[i].score, cands[i].mv, cands[i].depth);
        let (bs, bmv) = (cands[best].score, cands[best].mv);
        if bs >= MATE_IN_MAX_PLY {
            if cs > bs { best = i; }
        } else if cs >= MATE_IN_MAX_PLY
            || (cs > -MATE_IN_MAX_PLY
                && (vote_of(cmv) > vote_of(bmv)
                    || (vote_of(cmv) == vote_of(bmv) && cd > cands[best].depth)))
        {
            best = i;
        }
    }

    // If a non-main thread won, adopt its move + ponder into info so uci.rs sees
    // pv_table[0][0] == returned bestmove and can emit the ponder. Main's own PV
    // is already in info and richer, so leave it when main wins.
    let winner_mv = cands[best].mv;
    if !cands[best].is_main {
        info.pv_table[0][0] = winner_mv;
        info.last_score = cands[best].score;
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
    winner_mv
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
    board.generate_threat_deltas = info.nnue_net.as_ref().is_some_and(|n| n.has_threats);
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
    // Cross-thread TM (SF port): track this helper's best-move changes between
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

/// Run iterative deepening search.
pub fn search(board: &mut Board, info: &mut SearchInfo, limits: &SearchLimits) -> Move {
    init_feature_flags();

    // Enable threat delta generation if we have a threat net
    board.generate_threat_deltas = info.nnue_net.as_ref().is_some_and(|n| n.has_threats);

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

    // Age history tables (×0.80) to preserve useful move ordering from prior searches.
    // Killers and counter-moves are cleared (position-specific).
    // T2.1: correction history PERSISTS across `go` (cleared on ucinewgame
    // only, uci.rs) — all 5 surveyed engines persist within a game. With
    // the T2.4 full-error updates the table converges fast enough that
    // plain persistence (#1930, flat under the ±3cp clamp) becomes
    // load-bearing: each move starts from a warm eval-calibration.
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
    info.tm_no_inc = false;
    info.tm_baseline = 0;
    info.abs_deadline = 0;
    // TM audit 2026-06-13 (A4): tm_max_time was the one TM field missing
    // from this reset. Latent today (every soft_limit setter also sets
    // it) but one refactor away from a stale clamp.
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

    // P1.4: the best root PV seen during the CURRENT iteration (captured after
    // each aspiration search, so a widening re-search or a mid-iteration abort
    // can't wipe a proven fail-high move). On abort we bank this deepest
    // completed root result instead of reverting to the previous iteration's
    // shallower `stable_pv`. pv_table[0] is internally paired (move + its
    // ponder), so banking it never reintroduces the oeZ7KRUt old-move/new-ponder
    // mismatch. iter_pv_len is declared fresh each iteration (below); the array
    // is reused (only iter_pv[..iter_pv_len] is ever read).
    let mut iter_pv: [Move; MAX_PLY + 1] = [NO_MOVE; MAX_PLY + 1];

    // Get a fallback move and keep the legal list for final validation
    let root_legal = generate_legal_moves(board);
    if root_legal.len > 0 {
        best_move = root_legal.get(0);
        // TM audit 2026-06-13 (A4): prefer the TT move (previous search's
        // best for this position) over raw movegen order as the
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

    let effective_max = info.max_depth.min(MAX_PLY as i32 / 2);
    for depth in 1..=effective_max {
        if info.should_stop() { break; }
        info.root_depth = depth;
        info.sel_depth = 0; // P2.8: reset per iteration (consensus) — the info line then reports THIS iteration's seldepth, not a whole-search running max
        let mut iter_pv_len: usize = 0; // P1.4: fresh best-PV snapshot per iteration
        // Ponderhit check: stop between iterations (not mid-search) to avoid
        // partial TT entries and PV inconsistency. The engine completes the
        // current iteration fully before stopping, producing clean state.
        // A1 publish protocol (TM audit 2026-06-13): the ponderhit deadline
        // trio (hard=ponderhit_time, soft=ponderhit_soft, floor=
        // ponderhit_floor) is published by the UCI thread as
        //   floor (Relaxed) → soft (Relaxed) → hard (Release)
        // so hard is the publish flag. We load hard with Acquire FIRST and
        // read soft/floor ONLY after observing hard != 0 — the Acquire pairs
        // with the Release store, guaranteeing both Relaxed-stored fields
        // are visible. Keying the arming off soft alone (the pre-fix shape)
        // could, on ARM, observe soft > 0 with a STALE floor == 0 (killing
        // the stockpile floor) or stale hard == 0.
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
                // H7 (Atlas review): init tm_max_time on the ponder path. The
                // dynamic-TM consumer caps via .min(tm_max_time); it's never set
                // on `go ponder` (C6 reset + infinite branch both skip it), so it
                // carries 0 → min(soft*mult,0).max(1)=1ms → instant emit. Bites at
                // inc>=500 (the dynamic-TM path); the gate-removal above handles
                // the inc<500 path. Both fixes are needed and independent.
                info.tm_max_time = hard_remaining;
            }
        }
        let iter_start = std::time::Instant::now();

        let score;

        // Aspiration windows (skip for mate scores)
        if depth >= 4 && prev_score > -MATE_IN_MAX_PLY && prev_score < MATE_IN_MAX_PLY {
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

                // P1.4: capture this search's root PV before the widening
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
                    // (SF search.cpp:411-418: fail-low revokes
                    // stopOnPonderhit). Cleared when the re-search resolves
                    // below. Relaxed: independent bool gate, no dependent
                    // data (see field doc).
                    info.root_fail_low.store(true, std::sync::atomic::Ordering::Relaxed);
                    // FL-EXT v2 during-post-hit half (2026-07-05): a root
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
            // P1.4: snapshot this single search's root PV (depth<4 path).
            if info.pv_len[0] > 0 {
                iter_pv_len = info.pv_len[0].min(iter_pv.len());
                iter_pv[..iter_pv_len].copy_from_slice(&info.pv_table[0][..iter_pv_len]);
            }
        }

        // P1.4: unified mid-iteration abort handling. Bank this iteration's
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
        // Exact node count (2026-07-06, sibling of the `go nodes` accounting
        // fix): the global counter only updates at 4096-node flushes, so info
        // lines reported nodes 0 / 4096 / 8192... at shallow depths — garbage
        // for nodes-per-depth measurement. Add this thread's unflushed delta.
        let global = info.global_nodes.load(Ordering::Relaxed)
            + (info.nodes - info.last_flushed_nodes.get());
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
                let pv_mv = info.pv_table[0][i];
                // Legality guard: pv_table can carry a stale sibling-line move
                // (e.g. a fail-high / stable_pv restore that overran pv_len, or
                // an unguarded child-PV copy at the propagation site). Printing
                // it verbatim would emit an illegal PV move — a cutechess
                // "Illegal PV move" warning and a latent lichess forfeit risk
                // (same class as oeZ7KRUt 2026-04-26, guarded at the TT-cutoff
                // stuff site). Stop the PV at the first move not legal in the
                // current pv_board, mirroring the TT-extension tail below.
                if pv_mv == NO_MOVE
                    || !crate::movepicker::is_pseudo_legal(&pv_board, pv_mv)
                    || !pv_board.is_legal(pv_mv, pv_board.pinned(), pv_board.checkers())
                {
                    break;
                }
                pv_board.make_move(pv_mv);
                if seen_hashes.iter().filter(|&&h| h == pv_board.hash).count() >= 2 { break; }
                seen_hashes.push(pv_board.hash);
                if !pv_str.is_empty() { pv_str.push(' '); }
                pv_str.push_str(&move_to_uci(pv_mv));
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
                "info depth {} seldepth {} {} nodes {} nps {} time {} hashfull {} tbhits {} pv {}",
                depth, info.sel_depth, score_str,
                global, nps, elapsed,
                info.tt.hashfull(), info.tb_hits, pv_str
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
                    // Publish main's change into its own slot (thread 0) of the
                    // cross-thread bmc array (SF port). Read+reset in the TM block.
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
        //
        // Additional gate (2026-05-23 hotfix): skip at NO-INC TCs.
        // At no-inc TCs (1+0, 3+0, 60+0, 180+0) lichess showed time
        // forfeit regression — detector verification at depth 3-5 costs
        // 50-150ms per move and accumulates over the game. With no inc
        // to recover spent time, this cumulative overhead drains the
        // clock and causes time forfeits.
        //
        // Earlier (also 2026-05-23) used `info.soft_floor == 0`, which
        // incidentally fired at STC 10+0.1 because `inc == overhead ==
        // 100ms` made `soft_floor=0` by coincidence — SPRT #1475 caught
        // this, the gate was disabling the forced-move detector at STC
        // (~-3 Elo). Now keyed directly on `tm_no_inc`, which is true
        // only when our increment is exactly zero (sudden-death).
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

        // Phase 13 (2026-05-26): Viridithas-shape dynamic TM — 4 factors,
        // no separate Phase-10h cap (max_time clamps directly).
        //
        // Factor product max ~9.5× (vs Coda's prior ~30×). Wider hard window
        // (46% of clock from compute_tm_budgets) means factors can express
        // real variety without overflowing. max_time (60% of clock) is the
        // only single-move ceiling. See top-engine audit 2026-05-26 for
        // rationale; cap-bind diagnostic (TM_DIAG) showed 65% of iterations
        // clamped under prior structure.
        if info.soft_limit > 0 && depth >= 4 && !info.should_stop() {
            // Mate early-emit: if we've found a forced mate and the best move
            // has held for at least one further iteration, stop deepening —
            // more search cannot improve on a forced mate, and burning the
            // soft budget (and the post-ponderhit floor below) on a position
            // we've already solved just wastes clock and looks terrible (e.g.
            // 18.8s to play a mate-in-1 at 60+10 under ponder — lichess
            // KsX0b6KG). Require stability >= 1 so a one-iteration mate flicker
            // that later flips doesn't cause a premature emit. Also sets the
            // floor to 0 so the stockpile-prevention sleep is skipped.
            // TM audit 2026-06-13 (A4): gate on prev_score > 0 — the old
            // sign-agnostic check also fired when stably LOSING by force,
            // stopping the search instead of hunting longer defenses or
            // swindle lines with the remaining budget.
            if prev_score > 0 && is_mate_score(prev_score) && info.tm_best_stable >= 1 {
                info.soft_floor = 0;
                break;
            }
            // Factor 1: Stability multiplier (table-indexed by stable count).
            // Verbatim from Viridithas: [2.50, 1.20, 0.90, 0.80, 0.75]
            //   0 stable:  2.50× (uncertain, search more)
            //   1 stable:  1.20×
            //   2 stable:  0.90× (settling)
            //   3 stable:  0.80×
            //   4+ stable: 0.75× (confident, search less)
            // This single factor absorbs what Coda previously split across
            // stability_factor (0.5-1.71) + bmc_factor (1.0-5.0) — Viridithas
            // doesn't separately track best-move-changes.
            // Phase 13.2 (2026-05-26): lower stability_table[0] from 2.50 to 1.71.
            // Phase 13.1 (Viridithas table 2.5 + phase scaling) still produced
            // -41 Elo at 30+0.25 with opening overspend 37% (main: 21%).
            // The 2.5× initial multiplier was 1.46× more aggressive than
            // Coda's prior 1.71× stability_factor maximum. Even with phase
            // scaling the per-move spend overshot Coda's pre-Phase-10h
            // calibrated opening allocation. 1.71 matches Coda's prior
            // stability_factor ceiling; preserves variety in middlegame
            // (where stab=0 only briefly) while reducing opening spike.
            const STABILITY_TABLE: [f64; 5] = [1.71, 1.20, 0.90, 0.80, 0.75];
            let stability_idx = (info.tm_best_stable as usize).min(4);
            let stability_multiplier = STABILITY_TABLE[stability_idx];

            // Factor 2: Aspiration fail-low bonus (Viridithas event accumulator).
            // Formula: 1.0 + 0.34 × min(2, count), range [1.00, 1.68]
            //   0 fails: 1.00× (baseline)
            //   1 fail:  1.34×
            //   2+ fails: 1.68× (cap)
            // Captures the upward instability signal.
            let failed_low_multiplier = 1.0 + 0.34 * (info.tm_asp_fail_low.min(2) as f64);

            // Factor 3: Forced-move multiplier (Viridithas, position-intrinsic).
            //   Strong: 0.386× (alternative -400cp behind)
            //   Weak:   0.627× (alternative -170cp behind)
            //   None:   1.00×
            let forced_move_multiplier = match info.tm_forced_state {
                ForcedState::Strong => 0.386,
                ForcedState::Weak   => 0.627,
                ForcedState::None   => 1.0,
            };

            // Factor 4: Best-move subtree-size multiplier (Viridithas).
            // Formula: (1.62 - nodes_fraction) × 1.4, range ~[0.87, 2.27]
            //   nodes_fraction = best_move_nodes / total_nodes
            //   high fraction (>0.6): confident → reduce time
            //   low fraction (<0.3):  uncertain → increase time
            let mut subtree_frac = -1.0f64; // diagnostic only; -1 = not computed
            let subtree_size_multiplier = if depth > 9 && best_move != NO_MOVE {
                let bm_from = move_from(best_move) as usize;
                let bm_to = move_to(best_move) as usize;
                let best_nodes = info.root_move_nodes[bm_from * 64 + bm_to];
                let total = info.nodes;
                if total > 0 {
                    let frac = best_nodes as f64 / total as f64;
                    subtree_frac = frac;
                    // C1: base tunable-ized and re-centered (see
                    // TM_SUBTREE_BASE_100 in tunables!). Floor 0.55 bounds the
                    // discount on total-consensus moves (frac -> 1).
                    let base = tp(&TM_SUBTREE_BASE_100) as f64 / 100.0;
                    ((base - frac) * 1.4).max(0.55)
                } else {
                    1.0  // default when no node data
                }
            } else {
                1.0  // early depths: neutral
            };

            // Factor 5: Score-trend multiplier (falling-eval). The signal
            // `score_drop` (= tm_prev_score - prev_score, in cp; positive =
            // eval FELL this iteration) was already computed but discarded.
            // 5 of 10 surveyed top engines (SF fallingEval, Integral, Obsidian,
            // PlentyChess, Reckless score_trend) feed it into the TM product:
            // give MORE time when the eval is falling (position worsening —
            // don't snap-move into trouble) and LESS when it's stable or
            // improving (calm/winning — move on). Shaped after Reckless
            // (clamp(0.8 + 0.05*(prev-cur))) but in cp units. CENTERED AT 1.0
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
                // clamp rail (C2, persistent-state audit 2026-07-10). Ceiling
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
                (1.0 + 0.0025 * drop + (tp(&CROSS_MOVE_TREND) as f64 * 1e-4) * cross)
                    .clamp(0.80, 1.55)
            };

            // Combined multiplier — Viridithas's 4 factors + score-trend.
            // Max product ~ 2.50 × 1.68 × 1.0 × 2.27 × 1.45 = 13.8×
            // Min product ~ 0.75 × 1.0  × 0.386 × 0.87 × 0.80 = 0.20×
            // Factor 6: cross-thread best-move instability (SF port, Threads>1
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
            // the smaller no-inc opt baseline. lichess MJ442247 (3+0):
            // moves 8-19 every spent exactly hard = 10% of clock. Cap the
            // multiplier so adjusted_soft stays well below hard at no-inc,
            // letting the soft check actually fire and giving per-move
            // variability instead of uniform hard-cap saturation.
            if info.tm_no_inc {
                multiplier = multiplier.min(2.5);
            } else {
                // Low-increment ceiling (2026-06-18). When the increment is
                // small RELATIVE TO THE CLOCK it can't refill a run of deep
                // moves, so cap the factor product. Discriminator is
                // inc_cover = inc / (timeLeft/mtg): ~0.04 at lichess 600+1
                // (capped), ~0.24 at OB STC 10s+0.1s and ~0.4 at 600+10 (both
                // ~uncapped). cmin at inc_cover->0, rising to cmax at
                // inc_cover >= TM_INC_COVER_REF/100.
                // 24 = DEFAULT_MOVES_TO_GO (the inc-path sudden-death mtg).
                let base_move = (info.tm_time_left / 24).max(1);
                let inc_cover = (info.tm_our_inc as f64) / (base_move as f64);
                let ref_cover = (tp(&TM_INC_COVER_REF) as f64 / 100.0).max(0.001);
                let inc_factor = (inc_cover / ref_cover).clamp(0.0, 1.0);
                let cmin = tp(&TM_MULT_CEIL_MIN_10X) as f64 / 10.0;
                let cmax = tp(&TM_MULT_CEIL_MAX_10X) as f64 / 10.0;
                let inc_ceiling = (cmin + (cmax - cmin) * inc_factor).max(1.0);
                multiplier = multiplier.min(inc_ceiling);
            }

            // Phase 13: adjusted_soft = soft × multiplier, clamped to max_time
            // (the ONLY cap — no separate hard×0.5). Viridithas pattern.
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
            // `scale` / `max_adjusted`. `scale` retained for TM_DIAG output.
            // Subtract tm_baseline so soft is measured from the TM-start
            // moment, not search start. tm_baseline is 0 for normal `go`
            // (unchanged behaviour); set to elapsed-at-ponderhit when
            // post-ponderhit dynamic TM arms above.
            // TM audit 2026-06-13 (A4): re-read the clock — the iteration-top
            // `elapsed` snapshot predates the info-line print and any
            // forced-move verification (50-150ms), letting one extra
            // iteration start past the soft budget on exactly the
            // iteration the detector fires.
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
    // Publish ONLY from searches that produced a played game move (C2,
    // persistent-state audit 2026-07-10): a clock-managed search
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
    // at 1s-inc bullet on lichess (PZ7pCyrx). Polls the EXTERNAL stop flag so
    // the UCI thread can still interrupt — `info.stop` cannot serve here
    // because the line below sets it ourselves (to halt helpers), and on the
    // ponderhit fresh-search path the floor can equal the entire remaining
    // clock, so an un-interruptible sleep would block `stop`/`quit` for that
    // long (TM audit 2026-06-13 A2). Skip when there's no time budget (depth/
    // node-limited search) or when already stopped.
    //
    // C8 audit LIKELY #29: set the shared stop flag BEFORE the sleep so
    // helper threads stop searching immediately rather than burning CPU
    // through the entire stockpile-prevention window. Previously helpers
    // kept running until hitting their own hard_limit or main unblocked,
    // wasting tens-hundreds of ms of CPU per ponderhit grace window at
    // blitz+inc. Main thread already has its best move, just waiting to
    // emit.
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
    // true totals once before returning (SF prints the same final update). No
    // PV — GUIs keep the last full-line PV; only the counters need correcting.
    if !info.silent && info.completed_depth > 0 {
        let elapsed = info.start_time.elapsed().as_millis() as u64;
        let global = info.global_nodes.load(Ordering::Relaxed)
            + (info.nodes - info.last_flushed_nodes.get());
        let nps = if elapsed > 0 { global * 1000 / elapsed } else { 0 };
        let score_str = if is_mate_score(info.last_score) {
            let mate_in = if info.last_score > 0 {
                (MATE_SCORE - info.last_score + 1) / 2
            } else {
                -(MATE_SCORE + info.last_score + 1) / 2
            };
            format!("score mate {}", mate_in)
        } else {
            format!("score cp {}", info.last_score)
        };
        println!(
            "info depth {} seldepth {} {} nodes {} nps {} time {} hashfull {} tbhits {}",
            info.completed_depth, info.sel_depth, score_str,
            global, nps, elapsed, info.tt.hashfull(), info.tb_hits
        );
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
        if board.is_repetition_draw(ply) {
            return draw_score;
        }
    }

    // Guard against stack overflow — only reached for non-drawn positions
    // at ply >= MAX_PLY.
    if ply_u >= MAX_PLY {
        return apply_halfmove_scale(info.eval(board), board.halfmove);
    }

    // Leaf node — dispatch to quiescence at the TOP of negamax (P2.1),
    // after the draw checks + MAX_PLY guard but BEFORE the interior preamble
    // (reductions reset, mate-distance pruning, TT prefetch, enemy_attacks /
    // xray computation, TB probe, TT probe). Every depth<=0 entry (~14% of
    // calls) previously paid that whole preamble and then re-ran draw checks,
    // prefetch and a second TT probe inside quiescence, plus a duplicate
    // `info.nodes += 1` (~10% boundary node-count inflation that also leaked
    // into the TM node-fraction signal). SF/Reckless dive to qsearch as the
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


    // Draw detection moved above the MAX_PLY guard (2026-05-23 audit fix
    // for ply >= MAX_PLY in drawn positions). Note: kept the broader
    // contempt + jitter context comments at the new location for history.

    // Syzygy tablebase probe at interior nodes.
    // Probe WDL when piece count is within TB range. Returns a score that
    // causes a cutoff, so the search doesn't waste time in solved endgames.
    // Only at non-root (ply > 0) and non-excluded (not in singular verification).
    //
    // tb_floor: Some(tb_score) when an in-window PV TB hit raised alpha.
    // Search must not return / store below this — TB is ground truth.
    let mut tb_floor: Option<i32> = None;
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
            if pc <= max_pc && (pc < max_pc || depth >= info.tb_probe_depth) {
                if let Some(wdl) = tb.probe_wdl(board) {
                    info.tb_hits += 1;
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
            // 50mr mate/TB downgrade now happens inside score_from_tt
            // (SF/Reckless value_from_tt placement), so every consumer
            // below sees the sanitized score.
            let tt_score = score_from_tt(tt_entry.score, ply, board.halfmove);

            // P2 (halfmove-gated TT cutoff): TT scores are stored without halfmove
            // context. Near the 50-move cliff a cached mate-in-N may be unreachable,
            // and a stored bound may be over/understated by the time we revisit.
            // Gate ALL return-from-TT paths (direct + bounds-narrow collapse +
            // near-miss + QS) on halfmove < 90. Window-narrowing is still applied —
            // it only biases the search, while returning stale tt_score is unsafe.
            let halfmove_ok = (board.halfmove as i32) < tp(&TT_CUTOFF_HALFMOVE_MAX);
            // P1.8: require +1 ply of TT depth for a fail-high (LOWER) cutoff —
            // SF/Reckless/Obsidian/PlentyChess all demand `depth > depth - (value
            // <= beta)`, i.e. fail-lows accept at tt_depth>=depth but fail-highs
            // need tt_depth>=depth+1. Coda's symmetric `>= depth` is more
            // permissive on fail-highs than the top 3.
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
                // `is_pv` at line 2559 captured pre-mate-dist (`beta - alpha > 1`).
                // Mate-distance pruning may have collapsed a PV window to zero;
                // in that case the TT cutoff SHOULD fire (we're effectively ZW),
                // but the stale is_pv blocks it. Use the post-mate-dist window
                // directly (alpha at this point is still alpha_orig — TT
                // narrowing happens at line 2776+ after this check).
                // 2026-05-31 audit finding B.
                let tt_cut_is_pv = beta - alpha > 1;
                // Child-consistency verification for DEEP cutoffs (SF
                // search.cpp:873-892; 2026-07-05 audit Tier1 #3): at depth>=7,
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
                    return tt_score;
                }

                // Fall through: use TT bounds to narrow alpha/beta window at non-PV nodes.
                // Gated on halfmove_ok for the same reason the returns below are (P2/#628):
                // near the 50-move cliff the stored tt_score is untrustworthy. At a
                // zero-window node this narrowing can only ever fully collapse the window
                // (tt_score > alpha implies tt_score >= beta), and the collapse-return
                // below IS halfmove-gated — so without this gate, at halfmove >= 89 the
                // node fell through with an inverted window (alpha >= beta) and searched +
                // TT-stored a degenerate full-depth bound. (P0.3, 2026-07-01 review.)
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

    // (Leaf-node quiescence dispatch hoisted to the top of negamax — P2.1.)

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
        static_eval = if FEAT_CORRECTION.load(Ordering::Relaxed) { corrected_eval(info, board, scaled_eval, ply_u) } else { scaled_eval };
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

    // Clear the grandchild cutoff counter so the `cutoff_count[ply+1]`
    // read in LMR reflects only fail-highs under THIS node's subtree
    // (SF `(ss+2)->cutoffCnt = 0` / Reckless search.rs:488 pattern).
    info.cutoff_count[ply_u + 2] = 0;

    // Eval instability: detect sharp eval swings from parent node
    let unstable = !in_check && ply >= 1 && ply_u >= 1
        && info.static_evals[ply_u - 1] > -INFINITY
        && {
            let parent_eval = -info.static_evals[ply_u - 1];
            let diff = (static_eval - parent_eval).abs();
            diff > tp(&UNSTABLE_THRESH)
        };

    // Detect if TT move is noisy. Captures, EP, AND promotions
    // (including non-capture promotions — they create a queen, are
    // tactically loud). Prior version classified non-capture promotion
    // as `!noisy`, asymmetric with tt_move_is_quiet at line 3155 which
    // calls it quiet. 2026-05-31 audit finding D.
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

    // Hindsight extension (Stormphrax search.cpp:749-752): mirror of the
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

    // RFP moved BEFORE NMP (consensus order: SF/Reckless/Obsidian/Berserk all
    // run the free static prune first; the null search only sees nodes static
    // pruning couldn't cut). Reorder alone tested neutral (#1882, -0.06 ±1.9),
    // but it removes the mechanism that killed shallow NMP in #1904 (NMP-first
    // intercepted free RFP cutoffs), enabling the min-depth de-gate below.
    if !in_check {
        // Razoring (re-added 2026-06-11, audit T2.6; removed d996d6f on
        // pre-v9-eval evidence). 10/10 stronger engines have the
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
                margin += deep_extra * tp(&RFP_DEEP_LINEAR)
                    + deep_extra * deep_extra * tp(&RFP_DEEP_QUAD_10X) / 10;
            }
            // Widen margin when opponent pawns attack our pieces (Minic/Berserk pattern)
            if has_pawn_threats { margin += margin / 3; }
            // E2: widen margin when position is unstable (parent-child eval gap
            // > UNSTABLE_THRESH). Static eval can't be trusted for RFP when
            // eval is volatile. Mirrors unstable × ProbCut skip (#542 +6.7).
            if unstable { margin += margin / 3; }
            if static_eval - margin >= beta {
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
                    }
                }
                return static_eval - margin;
            }
        }
    }

    // T2.1: undefended ("hanging") piece count — our non-pawn pieces attacked
    // by the enemy and NOT defended by us. P2.4: moved here (past RFP, right
    // before NMP consumes it) and gated on cut_node/nmp_min_ply — it feeds NMP
    // ONLY, so every RFP-cut node used to pay ~10-15 magic lookups for nothing.
    // Value is identical (no move made between the old site and here), so
    // bench-nodes are unchanged.
    let undefended_count: i32 = {
        let nmp_gate_cheap = depth >= tp10(&NMP_MIN_DEPTH_10X) && !in_check && ply > 0
            && stm_non_pawn != 0 && beta - alpha == 1
            && static_eval >= beta && !prev_was_null
            && beta.abs() < MATE_IN_MAX_PLY
            && info.excluded_move[ply_u] == NO_MOVE
            && cut_node && ply >= info.nmp_min_ply;
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

    let nmp_threat_margin =
        (king_zone_pressure - (tp10(&NMP_KING_ZONE_MAX_10X) - 1)).max(0) * 64
        + (any_threat_count - 2).max(0) * 64
        + (undefended_count - (tp10(&NMP_UNDEFENDED_MAX_10X) - 1)).max(0) * 128;

    if depth >= tp10(&NMP_MIN_DEPTH_10X) && !in_check && ply > 0 && stm_non_pawn != 0
        && beta - alpha == 1 && static_eval >= beta + nmp_threat_margin
        && !prev_was_null  // Prevent consecutive null moves
        && ply >= info.nmp_min_ply  // Ply barrier: verification subtree cannot re-trigger NMP (audit B1)
        && beta.abs() < MATE_IN_MAX_PLY  // Skip NMP for mate/TB scores
        && info.excluded_move[ply_u] == NO_MOVE  // Skip NMP during SE verification
        && cut_node  // Reckless gate: only attempt NMP at expected fail-high nodes (closes 30%->57% NMP cutoff-rate gap)
        && FEAT_NMP.load(Ordering::Relaxed)
    {
        info.stats.nmp_attempts += 1;
        // Adaptive reduction: scales with depth and eval margin above beta
        let mut r = tp10(&NMP_BASE_R_10X) + depth / tp10(&NMP_DEPTH_DIV_10X);
        // Reduce more after captures: opponent just captured, null move more likely to work.
        // NOT a live cross-engine consensus (SF's R is flat; Obsidian keys on the
        // CURRENT node's ttMoveNoisy, and that port is dead ×4 — #732/#754/#768/#2270).
        // This term is kept purely on Coda's own evidence: #195 H1 +3.5 (with retune),
        // removal #1067 H0 −2.6.
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
        // C3 (2026-04-22 audit): set null sentinel on moved_piece_stack /
        // moved_to_stack at ply_u. Without this, child at ply+1 reads
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
                // subtree. All peer engines do this (Reckless: nmp_min_ply = ply +
                // 3*(depth-r)/4; Alexandria: nmpPlies = ply + (depth-R)*2/3).
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

    // SF one-liner (audit T1.5): a static eval already at/above beta counts
    // as improving for the whole move loop, even if it's worse than 2 plies
    // ago — the node is beating its window. Placed after NMP/RFP (both read
    // the plain 2-ply definition, matching SF's ordering); upgrades LMP's
    // (2 - improving) divisor, ProbCut's margin and LMR's !improving bump
    // for the remainder of the node.
    if !in_check && static_eval >= beta {
        improving = true;
    }

    // IIR: moved after NMP so null search uses full depth, not IIR-reduced depth.
    // All 6 reference engines run NMP at full depth; IIR only applies to the moves loop.
    // Coda previously ran IIR before NMP, silently reducing null depth by 1 at cut nodes.
    // (NMP audit N2)
    if depth >= tp10(&IIR_MIN_DEPTH_10X) && tt_move == NO_MOVE && !in_check && (is_pv || cut_node) && FEAT_IIR.load(Ordering::Relaxed) {
        depth -= 1;
    }

    // (RFP moved above NMP — see pre-NMP site.)

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
        && !unstable  // Skip ProbCut in eval-unstable positions (eval can't be trusted)
        && FEAT_PROBCUT.load(Ordering::Relaxed)
    {
        // SEE threshold: only consider captures that gain enough material
        let see_threshold = (probcut_beta - static_eval).max(0);
        // Improving-conditioned ProbCut depth (SF d6483505 / Reckless 08f2cfa4) —
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
                // Stockfish search.cpp:994-995. Prior code stored `dampened` and
                // hardcoded tt_pv=false, losing both pruning information on
                // future probes and the PV stickiness used by LMR reduction
                // decisions. Return value is still dampened for normal
                // scores — score was verified at probcut_beta = beta+margin,
                // not beta. Decisive mate/TB scores are exact enough that
                // margin subtraction corrupts their distance/range; SF avoids
                // damped decisive ProbCut returns and Reckless returns them raw.
                info.tt.store(
                    board.hash, depth - 3, score_to_tt(score, ply),
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
    // moves reduce more proportionally (Viridithas #431 alpha_raises) — once
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

        // Late Move Pruning (reordered FIRST, SF Step-14 order): at shallow
        // depths, skip late quiet moves by movecount BEFORE SEE/futility filter
        // them. Running LMP last (the prior Coda order) meant its count check
        // only saw SEE/futility survivors — a pre-filtered residual that made
        // count-pruning riskier and kept LMP_BASE blunt. SF/Reckless/Berserk/
        // Obsidian all set skipQuiets before SEE/futility.
        // Formula: (LMP_BASE + depth²) / (2 - improving); check carve at depth<4.
        if ply > 0 && !in_check && depth >= 1 && depth <= tp10(&LMP_DEPTH_10X)
            && !is_cap && !is_promo
            && !is_loss(best_score)
            && beta < MATE_IN_MAX_PLY  // forced-win guard (Reckless 4a2efd5a): don't count-prune quiets while proving a win
            && FEAT_LMP.load(Ordering::Relaxed)
        {
            let lmp_limit = (tp10(&LMP_BASE_10X) + depth * depth) / (2 - improving as i32);
            // P2.5: gives_direct_check carve moved inside the movecount test — only
            // pay the check-detection call when the count prune would actually
            // fire (node-count identical).
            if move_count > lmp_limit && (depth >= 4 || !board.gives_direct_check(mv)) {
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

        // Futility pruning (P2.2: moved ABOVE SEE-quiet so the cheap static prune
        // fires first — SEE-quiet then only re-runs see_ge on survivors; 5/6
        // references order it this way, analog LMP reorder #2283 was H1 +0.6).
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
            // Direct-check carve-out + strong-history exemption (Igel/Reckless #410).
            if futility_value <= alpha && main_hist < 12000 && !board.gives_direct_check(mv) {
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
            // No !in_check gate: zero-engine-consensus carve-out removed
            // (audit T2.12). None of SF/Reckless/Obsidian/Berserk/Stormphrax
            // gate SE on check; a deep in-check node's TT move (often the
            // single forced evasion — maximally singular) was never extended,
            // and got no multicut/negative-ext either. Mechanically safe:
            // the SE path reads no static_eval (-INFINITY in check); the
            // correction_value margin input is position-keyed.
            && info.excluded_move[ply_u] == NO_MOVE
            && tt_hit
            && tt_entry.flag != TT_FLAG_UPPER
            && tt_entry.depth >= depth - tp(&SE_TT_DEPTH_SLACK)
            && FEAT_SINGULAR.load(Ordering::Relaxed)
        {
            // 50mr downgrade applies here too (SF/Reckless: singular ttValue
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
                    // Return singular_score (SF pattern, search.cpp:1183) — tighter score
                    // for downstream TT propagation than singular_beta floor.
                    // EXCEPT decisive scores: singular_score is fail-soft from a
                    // reduced (depth-1)/2 search with the TT move EXCLUDED — a
                    // mate/TB score from it is unproven at this node's depth and
                    // would be TT-stored at full depth as LOWER. SF/Reckless
                    // gate with !is_decisive and fall through; Obsidian/Berserk
                    // return singularBeta. Per #761 (mate-clamp H0: suppressing
                    // multicut in mate shapes loses Elo), keep FIRING and fix
                    // only the returned value (audit T1.4).
                    info.stats.multicut += 1;
                    if is_decisive(singular_score) {
                        return singular_beta;
                    }
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

        // History-based pruning REMOVED 2026-06-02.
        //
        // Three SPRT attempts H0'd consecutively despite progressive structural
        // fixes:
        //   #1562 original (main_hist + 4 cont-hist + pawn): H0 −9.4 Elo
        //   #1691 SF pattern (cont[1]+cont[2]+pawn), SF defaults: H0 −7.5
        //   #1697 SF pattern, Coda-tuned SPSA values (#1690): H0 −6.8
        //
        // The cumulative evidence is that hist-prune as a feature genuinely
        // doesn't fit Coda's search/eval shape, irrespective of how it's
        // shaped or tuned. SF/Obsidian benefit; Coda doesn't. Different from
        // "we have a bug"; closer to "this feature isn't a fit." Net win to
        // remove: less code + complexity, one fewer per-node check, no more
        // ambiguity about whether the feature is doing anything useful.
        //
        // Companion removals: HIST_PRUNE_DEPTH_10X, HIST_PRUNE_MULT tunables;
        // FEAT_HIST_PRUNE flag; ENABLE_/NO_HIST_PRUNE env vars; the diagnostic
        // PruneStats fields (hist_prune_*, history_prunes, cont_hist_*).

        // (Futility pruning moved above SEE-quiet — P2.2.)

        // (Late Move Pruning moved earlier — now runs before SEE/futility,
        // immediately after the skip_quiets check. SF Step-14 order.)

        // Bad noisy pruning: skip losing captures when eval is far below alpha.
        // Applied before MakeMove. Direct-check carve-out: don't prune moves
        // that give direct check (Reckless #630 +1.85 STC).
        if FEAT_BAD_NOISY.load(Ordering::Relaxed) && is_cap && !in_check && ply > 0 && depth <= tp10(&BAD_NOISY_DEPTH_10X) && mv != tt_move
            && !is_promo && !is_loss(best_score)
            && static_eval > -INFINITY && static_eval + depth * tp(&BAD_NOISY_MARGIN) <= alpha
            && !see_ge(board, mv, 0)
            && !board.gives_direct_check(mv)
        {
            continue;
        }

        // Pre-make TT prefetch (Reckless #1085): issue the child-bucket fetch
        // BEFORE build_dirty_piece + NNUE push + make_move + threat absorb, so
        // ~all of that work overlaps the DRAM latency (vs the old post-make
        // prefetch, which had only the short pre-probe window). key_after is an
        // approximate hash (exact for the common cases; see board.rs) —
        // prefetch-only; the real probe below uses the true post-make hash.
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
        // N6 promotion-imminent (7th-rank pawn push) extension REMOVED
        // 2026-06-07 (structural-audit experiment). Coda-unique — NONE of the
        // 18 stronger engines in our RR extend 7th-rank pushes — ungated
        // beyond pawn/non-capture. Coda's OWN earlier testing (experiments.md
        // 2026-03-12) called these "pure noise — waste as much depth as they
        // gain." Re-added as N6, silently DEAD ~3 weeks (side_to_move bug),
        // "fixed" 2026-05-09 without an isolated SPRT — positive Elo never
        // established. Same profile as the recapture ext (+5.9 to remove).

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
        // Store piece/to/captured for history updates after search
        if is_cap && n_captures_tried < 32
            && moved_piece != NO_PIECE && captured_pt != NO_PIECE_TYPE {
                let ct = if flags == FLAG_EN_PASSANT { captured_type(PAWN) } else { captured_type(captured_pt) };
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

                // Reduce more at expected cut nodes. Coda historically applied a
                // flat +1 at every non-PV node without distinguishing expected
                // cut nodes (fail-high) from all-nodes; SF reduces ~+4 plies
                // specifically at cutNode, +1 more with no TT move, all-nodes
                // smaller. Split them: cut nodes get the tunable LMR_CUTNODE_BUMP
                // (+1 if no TT move); all-nodes keep +1. (P1.1, rebase of the lost
                // H1 #2065 / e616393.)
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
                // alpha (Viridithas #431). Fixed-point ×10. At cut nodes this is
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

                // Reckless LMR correction battery (T1.1, audit 2026-07-06).
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
                    // (c) TT guidance is shallower than the current search —
                    //     weaker move-ordering confidence, reduce more.
                    if (tt_entry.depth as i32) < depth {
                        reduction += tp(&LMR_TTDEPTH_CENTI);
                    }
                }
                // (d) Quiet expectation gap: eval far below alpha → this node is
                //     underperforming its window, reduce late quiets more (and
                //     slightly less when eval already exceeds alpha). Continuous,
                //     ~0.32 centi/cp at default. static_eval is valid here (the
                //     quiet-LMR block is gated on !in_check).
                reduction += tp(&LMR_EXPECT_MULT) * (alpha - static_eval).clamp(-65, 91) / 128;

                // cutoff_count (T1.2): the child ply keeps failing high under
                // this node — refutations come easy down there, so late moves
                // need less depth to refute. SF/Reckless consensus term.
                if info.cutoff_count[ply_u + 1] > 2 {
                    reduction += tp(&LMR_CUTOFF_CNT_CENTI);
                    if !is_pv && !cut_node {
                        reduction += tp(&LMR_CUTOFF_ALLNODE_CENTI);
                    }
                }

                // Continuous history adjustment: good history reduces less, bad more
                // Uses main history + ply-1 + ply-2 continuation history (consensus).
                // Ply-2 weighted at half to avoid over-scaling the total.
                // SF weights main history 2× vs continuation history. (LMR audit L7)
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
                    // for pawn moves reads an unrelated bucket (write/read mismatch,
                    // fixed 2026-06-11).
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
                        reduction -= capt_hist_val * LMR_SCALE / tp(&LMR_HIST_DIV_CAP); // CONTINUOUS
                    }

                    // Reduce less for captures that give check
                    if gives_check {
                        reduction -= LMR_SCALE;
                    }

                    // Reckless correction battery (a)-(c) — applied to noisy
                    // moves too (Reckless computes them before its quiet/noisy
                    // split). Same tunables as the quiet block.
                    if is_win(beta) {
                        reduction += tp(&LMR_WINBETA_CENTI);
                    }
                    if tt_hit && tt_entry.flag != TT_FLAG_NONE {
                        let tt_score_node = score_from_tt(tt_entry.score, ply, board.halfmove);
                        if tt_score_node <= alpha {
                            reduction += tp(&LMR_TTALPHA_CENTI);
                        }
                        if (tt_entry.depth as i32) < depth {
                            reduction += tp(&LMR_TTDEPTH_CENTI);
                        }
                    }

                    // cutoff_count (T1.2) — Reckless applies it before the
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
        // search is deep. Zero at STC (root_depth <= thresh); grows with how
        // deep the search reaches, so late moves at LTC are searched closer
        // to full depth. One formula, one tunable set (Adam directive).
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
            let lmr_depth = new_depth - reduction;
            let mut lmr_score = -negamax(board, info, -alpha - 1, -alpha, lmr_depth, ply + 1, true);

            // The reduction applies to the reduced search ONLY: zero the slot
            // before any re-search so children of the (near-)full-depth
            // re-searches don't read a stale prior_reduction and mis-fire
            // hindsight reduce/extend (SF `ss->reduction = 0` after the
            // reduced search; Stormphrax identical; audit T1.2).
            info.reductions[ply_u] = 0;

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

                // Mutate new_depth itself so the adjustment persists into the
                // full-window PVS re-search below (SF/Obsidian/Alexandria/
                // Stormphrax/Integral/Reckless all mutate newDepth; the old
                // inline form ran the PV re-search SHALLOWER than the
                // zero-window search that justified it; audit T1.3).
                new_depth += do_deeper_adj;
                // Guard: only re-search when new_depth actually changed from lmr_depth.
                // do_shallower with reduction==1 makes new_depth == lmr_depth — the
                // re-search would duplicate the already-completed LMR search. Every
                // reference engine guards with `if new_depth > lmr_depth`. (audit B2)
                if new_depth > lmr_depth {
                    info.stats.ts_lmr_research += 1;
                    lmr_score = -negamax(board, info, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
                }

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
            // First move: always full window. Child cut_node: at a PV node the
            // first child is itself a PV node (never a cut node) -> false; at a
            // non-PV node the first child's expected type is the negation of
            // this node's (all-node -> child is cut; cut-node -> first child is
            // all after the cutoff move) -> !cut_node. Matches SF (Step 18
            // `!cutNode` for non-PV first move vs `search<PV>(... false)`) and
            // Reckless (FDS `!cut_node` vs PVS `false`). Previously hardcoded
            // `false`, which mislabeled every non-PV first move as an all node
            // and disabled NMP/IIR/TT-cutoff node-type guards on that spine.
            let child_cut = if is_pv { false } else { !cut_node };
            score = -negamax(board, info, -beta, -alpha, new_depth, ply + 1, child_cut);
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
                    info.stats.cutoff_movecount_sum += move_count as u64;
                    info.stats.cutoff_movecount_sq_sum += (move_count as u64) * (move_count as u64);

                    // Beta cutoff - update history for quiet moves.
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
                        let mut bonus = raw_bonus + raw_bonus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);
                        // SF searched-count scale (search.cpp:1911-1914, Jun-2026
                        // patch; audit W2): the more moves were refuted before
                        // this one cut, the more informative the cutoff — scale
                        // the bonus up by moves-searched/256 at non-PV nodes.
                        if !is_pv {
                            bonus += bonus * (move_count - 1).max(0) / 256;
                        }
                        // SF 645b636d: at non-PV nodes, amplify the cutoff move's
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
                            // T6: base = current cont_hist + main_hist / 2 (Stormphrax history.h:120).
                            let main_score_v = info.history.main_score(from, to, enemy_attacks);
                            let ch_offsets = [1usize, 2, 4, 6];
                            for &off in ch_offsets.iter() {
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
                                            if prior_piece > 0 && prior_piece < 13 && prior_to < 64 {
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
                        // BONUS_BOOST_AT depth-boost removed 2026-05-17
                        // (ablation #1277 H0).
                        let cap_bonus_depth = depth;
                        // numFailHighs multiplicative scaling (#1054 ext of #1020):
                        // more cascades = stronger cutoff confidence.
                        let raw_cap_bonus = capture_history_bonus(cap_bonus_depth);
                        let scale_factor = num_fail_highs.min(tp10(&NFH_CAP_10X));
                        // Fixed-point divisor (stored × 10).
                        let mut cap_bonus = raw_cap_bonus + raw_cap_bonus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);
                        // SF searched-count scale (see quiet site above).
                        if !is_pv {
                            cap_bonus += cap_bonus * (move_count - 1).max(0) / 256;
                        }
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
                        // numFailHighs multiplicative scaling — mirror the
                        // capture-BONUS path (#1054) exactly so the failed
                        // captures' penalty tracks the cutoff capture's bonus.
                        // Previously the only beta-cutoff history update missing
                        // NFH scaling, which slowly inflated capture-history
                        // magnitude relative to the (scaled) bonus and quiets.
                        let raw_cap_malus = capture_history_malus(depth);
                        let scale_factor = num_fail_highs.min(tp10(&NFH_CAP_10X));
                        let cap_malus = raw_cap_malus + raw_cap_malus * scale_factor * 10 / NFH_DIV_10X.load(Ordering::Relaxed).max(1);
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

        // Adjust mate score for storage (relative to this position)
        let store_score = score_to_tt(best_score, ply);

        if FEAT_TT_STORE.load(Ordering::Relaxed) {
            info.tt.store(board.hash, depth, store_score, flag, best_move, raw_eval, tt_pv);
        }
    }

    // Fail-low prior-countermove bonus (SF search.cpp:1523-1553, simple core;
    // 2026-07-05 SF audit Tier 1 #2). When this node fails low with NO best
    // move, the opponent's previous quiet move "worked" — credit it in the
    // cont-hist context of our move before that, so the PARENT tries better
    // siblings sooner. This is the majority node class in a big tree
    // (all-nodes); Coda previously learned nothing from it (updates fired only
    // on beta cutoffs + TT cutoffs). Indexing mirrors the TT-cutoff cont-hist
    // malus site (moved_piece_stack, pre-move pieces — C8 audit #6 pattern).
    // NOTE: SF's gate is `!bestMove`, which in SF means "no move raised alpha"
    // (they assign bestMove only on alpha raises). Coda tracks a fail-soft
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
                if opp_gp > 0 && opp_gp < 13
                    && our_gp > 0 && our_gp < 13
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
    // (search.cpp:1495: `!(bestMove && pos.capture(bestMove))`) and Reckless
    // (search.rs:1085: `|| best_move.is_noisy()`).
    let best_move_noisy = best_move != NO_MOVE && {
        board.piece_type_at(move_to(best_move)) != NO_PIECE_TYPE
            || move_flags(best_move) == FLAG_EN_PASSANT
            || is_promotion(best_move)
    };
    // Correction history update: train on BOTH directions of error.
    // Previously gated on `best_score > alpha_orig` (fail-high only), which
    // never trained on fail-low (all-node) positions where static eval was
    // over-optimistic. SF and Reckless both update on fail-low when the error
    // direction is consistent: bound==Upper && best_score < static_eval means
    // eval predicted higher than any move achieved — train correction downward.
    // (audit S1)
    let corrhist_lower_ok = best_score > alpha_orig   // fail-high: lower bound
        && !(best_score >= beta && best_score <= static_eval); // direction-consistent
    let corrhist_upper_ok = best_score <= alpha_orig  // fail-low: upper bound
        && best_score < static_eval;                   // corrected eval was over-optimistic
    if !in_check
        && !best_move_noisy
        && info.excluded_move[ply_u] == NO_MOVE
        && (corrhist_lower_ok || corrhist_upper_ok)
        // T2.3: is_decisive (mate OR TB range)
        && !is_decisive(best_score)
        && scaled_eval > -(MATE_IN_MAX_PLY)
        && !info.stop.load(Ordering::Relaxed)
    {
        // Corrhist audit 2026-07-08 finding #1: train the update against the
        // CORRECTED eval (`static_eval`, the residual after correction), like
        // SF/Obsidian/Reckless/Berserk/Viridithas — NOT the raw `scaled_eval`.
        // Raw-training's gravity fixed point is the rail (magnitude-blind) and
        // manufactured the fortress phantom eval; residual converges to the true
        // correction and self-stabilises. Both in scaled-space so the err term
        // isolates positional miscalibration, not halfmove decay.
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

    // Draw detection: repetition and 50-move rule. Contempt removed (#508).
    let draw_score = 0;
    if board.halfmove >= 100 {
        return draw_score;
    }
    // FIDE Art 5.2: insufficient material to mate (any side). Mirrors
    // negamax's guard (added for Lichess game I4qJhfQw drawn KB-vs-K
    // class). QS recurses capture chains that can transition into drawn
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
        // (SF/Reckless: ttValue is value_from_tt output everywhere).
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

            // Skip quiet evasions once we have a non-losing score (audit
            // T2.10): SF searches quiet evasions only while still losing
            // (search.cpp:1681 `if (!capture) continue` inside !is_loss);
            // Obsidian breaks after one quiet; Reckless gates skip_quiets
            // the same way. Capture evasions always searched. The gate is
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
        // SF/Reckless/Obsidian all store only LOWER/UPPER in QS
        // (2026-06-11 audit T1.5).
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
    let raw_stand_pat = if tt_hit && tt_entry.static_eval > -4095 {
        // Threshold matches pack_data's clamp range. -INFINITY sentinels
        // (from in-check TT stores) get clamped to -4095 and would
        // otherwise pass a wider check.
        info.stats_tt_static_eval_hits += 1;
        tt_entry.static_eval
    } else {
        info.eval(board)
    };
    // Apply correction history to the QS stand-pat, mirroring negamax's
    // static-eval path (halfmove-scale THEN corrected_eval, line ~3361).
    // Consensus QS audit (2026-06-23): all 6 reference engines (SF, Reckless,
    // Berserk, Obsidian, PlentyChess, Alexandria) correct the QS stand-pat;
    // Coda was the sole outlier using the raw eval. The stand-pat feeds the
    // returned cutoff score, the best_score floor, AND the delta-prune base,
    // so the uncorrected error compounds. TT still stores the RAW value
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
    // Apply the SAME halfmove guard as the direct cutoff path at line
    // ~4468: without this, an inflated near-50mr TT lower bound replaces
    // stand_pat and triggers the `best_score >= beta` return below —
    // bypassing the gate that exists for exactly this case.
    if tt_hit && (board.halfmove as i32) < tp(&TT_CUTOFF_HALFMOVE_MAX) {
        // 50mr downgrade applies here too. A downgraded mate becomes a
        // TB-band value and is still filtered by !is_decisive below; a
        // downgraded TB score becomes the highest non-decisive value and
        // may refine stand-pat — same as SF/Reckless, whose eval
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
        // Cache eval + LOWER bound on the stand-pat fail-high (QS audit
        // 2026-06-23). This is the most common QS exit, and Coda was returning
        // here WITHOUT any TT store — so the raw eval wasn't cached (revisits
        // re-run NNUE) and no bound was left for a cheap future cutoff. All 6
        // reference engines store here (SF/Alexandria/Plenty gate on TT-miss).
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
        // QS beta blending (P1.12d: blend regardless of node type — 6/6
        // references don't gate on non-PV; Coda gated `beta - alpha == 1`).
        if !is_decisive(best_score) {
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

        // Move-count budget (audit T2.10): count only SEARCHED moves — the
        // old form incremented before delta/SEE pruning, so pruned moves
        // consumed budget and SPSA pushed the cap to near-off (24; comment
        // cited "Obsidian: 3"). Consensus gates: only while best_score
        // isn't a loss (SF is_loss(futilityBase) / Obsidian TB_LOSS gate)
        // and promotions exempt (SF). `continue` not `break` so later
        // promotions still get through (SF form).
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
                    // Fail-soft (P1.12a): delta_val is an upper bound on what this
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
    // Never EXACT — see the note at the evasion-path store above (audit T1.5).
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

    // QS beta blending (P1.12d: blend regardless of node type).
    if best_score >= beta && !is_decisive(best_score) {
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
            // Explicit override failing is fatal — see bench_inner.
            eprintln!("FATAL: failed to load NNUE '{}': {}", path, e);
            std::process::exit(2);
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
            // falling back (PeSTO or embedded net) produces a wrong bench /
            // wrong ordering stats that look plausible — the ~1M NPS PeSTO
            // numbers have been mistaken for real net stats twice
            // (2026-06-12, 2026-06-14). Fallback is for the no-override
            // auto-discovery path only.
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

fn reset_bench_position_state(info: &mut SearchInfo) {
    info.clear_persistent_histories();
    info.tt.new_search();
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Regression guard for docs/corrhist_fortress_drift_2026-07-06.md.
    /// Correction history used to self-reinforce into a phantom ±0.45 in
    /// low-material locked/fortress positions (opposite-coloured bishops,
    /// blocked pawns) that Stockfish/Reckless/Obsidian all read as 0 — the
    /// raw NNUE was fine (~0), it was corrhist railing in the low-signal
    /// regime. The piece-count damping in `corrected_eval` fixes it. These
    /// four positions (two from the games that surfaced the bug) must stay
    /// near 0; a regression would blow them back out to ±0.45.
    ///
    /// Needs an NNUE net (the drift is a corrhist-on-NNUE effect; the PeSTO
    /// fallback does not reproduce it), so it skips gracefully when no net is
    /// present — honours `CODA_TEST_NET`, else `net.nnue`, else a `net-v*.nnue`.
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
        let net_path: Option<String> = std::env::var("CODA_TEST_NET").ok()
            .filter(|p| std::path::Path::new(p).exists())
            .or_else(|| {
                let url = std::fs::read_to_string("net.txt").ok()?;
                let name = url.trim().rsplit('/').next()?.trim().to_string();
                (!name.is_empty() && std::path::Path::new(&name).exists()).then_some(name)
            });
        let net_path = match net_path {
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
        let limits = SearchLimits { depth: 16, infinite: true, ..SearchLimits::new() };
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

    /// P1 instant-reply gate — DOUBLE-PONDERHIT CASCADE GUARD (Adam's
    /// non-negotiable requirement, 2026-07-05 ponder fix). If the opponent
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
    /// a root fail-low revokes it (SF search.cpp:411-418 pattern).
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

    /// Regression guard for the PV-print legality check (fix/pv-print-legality
    /// -guard, 2026-05-31). The pv_table can carry a STALE sibling-line move —
    /// e.g. a king move from a square the king occupied in a different branch.
    /// The bug: game 132 (Coda vs Velvet, pooled RR) printed `g2f3` 66× while
    /// the king was on h2, emitting cutechess "Illegal PV move" warnings (a
    /// latent lichess forfeit class, cf. oeZ7KRUt 2026-04-26). The fix stops
    /// the printed PV at the first move that fails is_pseudo_legal + is_legal
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
