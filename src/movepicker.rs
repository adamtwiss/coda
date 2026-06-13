//! Staged move picker for search.
//!
//! Order: TT move -> good captures (MVV-LVA + captHist) -> quiets (history-scored,
//!        including main/cont/pawn/etc.) -> bad captures.
//!
//! Killer/counter stages were removed — quiet ordering relies on history alone
//! (SF/Reckless pattern, validated by SPRT commit e28c78a).
//!
//! Evasion order: TT move -> evasions (captures scored above quiets).
//!
//! No legality checks — returns pseudo-legal moves; search caller does legality.

use crate::attacks::*;
use crate::bitboard::*;
use crate::board::Board;
use crate::eval::see_value;
use crate::movegen::{generate_all_moves, generate_captures, generate_quiets, MoveList};
use crate::see::see_ge;
use crate::types::*;

const MAX_HISTORY: i32 = 16384;

/// Bitboard type alias for threat computation.
pub type Threats = u64;

/// History tables shared across the search.
pub struct History {
    /// Main history: [from_threatened][to_threatened][from][to]
    /// Threat-aware 4D indexing — separate history for moves escaping/entering threats.
    pub main: [[[[i32; 64]; 64]; 2]; 2],
    /// Capture history: [piece 1-12][to][captured_type 0-6]
    /// piece uses 1-12 indexing (slot 0 unused).
    /// captured_type uses 0-6 scheme (0=empty, 1=pawn, ..., 6=king).
    /// int16 values (i32 causes different gravity behavior).
    pub capture: [[[i16; 7]; 64]; 13],
    /// Continuation history: [piece 1-12][to][piece 1-12][to]
    /// piece uses 1-12 indexing (slot 0 unused).
    pub cont_hist: [[[[i16; 64]; 13]; 64]; 13],
}

impl History {
    /// Get main history score for a move given enemy threat bitboard.
    #[inline(always)]
    pub fn main_score(&self, from: u8, to: u8, threats: Threats) -> i32 {
        if crate::search::FEAT_4D_HISTORY.load(std::sync::atomic::Ordering::Relaxed) {
            let ft = ((threats >> from) & 1) as usize;
            let tt = ((threats >> to) & 1) as usize;
            self.main[ft][tt][from as usize][to as usize]
        } else {
            self.main[0][0][from as usize][to as usize]
        }
    }

    /// Get mutable reference to main history entry for a move given enemy threats.
    #[inline(always)]
    pub fn main_entry(&mut self, from: u8, to: u8, threats: Threats) -> &mut i32 {
        if crate::search::FEAT_4D_HISTORY.load(std::sync::atomic::Ordering::Relaxed) {
            let ft = ((threats >> from) & 1) as usize;
            let tt = ((threats >> to) & 1) as usize;
            &mut self.main[ft][tt][from as usize][to as usize]
        } else {
            &mut self.main[0][0][from as usize][to as usize]
        }
    }

    /// Allocate a zeroed History directly on the heap. The struct is ~1.4MB;
    /// `new()` builds it via stack array literals, which overflows the default
    /// test-thread stack. All fields are valid when zeroed (NO_MOVE == 0).
    pub fn boxed_zeroed() -> Box<History> {
        unsafe {
            let layout = std::alloc::Layout::new::<History>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut History;
            if ptr.is_null() { std::alloc::handle_alloc_error(layout); }
            Box::from_raw(ptr)
        }
    }

    pub fn clear(&mut self) {
        self.main = [[[[0; 64]; 64]; 2]; 2];
        self.capture = [[[0i16; 7]; 64]; 13];
        self.cont_hist = [[[[0; 64]; 13]; 64]; 13];
    }

    /// Copy all table contents from `src`. Used to seed Lazy SMP
    /// helpers with the main thread's accumulated history at spawn
    /// time, so helpers don't start every search with empty ordering
    /// info. Arrays of POD types — compiles to memcpy on the heap, no
    /// stack alloc.
    pub fn copy_from(&mut self, src: &History) {
        self.main = src.main;
        self.capture = src.capture;
        self.cont_hist = src.cont_hist;
    }

    /// Age all history tables by multiplying by factor/divisor (e.g. 4/5 = 0.80).
    /// Preserves useful information from prior searches while letting new data dominate.
    pub fn age(&mut self, factor: i32, divisor: i32) {
        for t0 in self.main.iter_mut() {
            for t1 in t0.iter_mut() {
                for row in t1.iter_mut() {
                    for v in row.iter_mut() { *v = *v * factor / divisor; }
                }
            }
        }
        for plane in self.capture.iter_mut() {
            for row in plane.iter_mut() {
                for v in row.iter_mut() { *v = (*v as i32 * factor / divisor) as i16; }
            }
        }
        for plane0 in self.cont_hist.iter_mut() {
            for plane1 in plane0.iter_mut() {
                for row in plane1.iter_mut() {
                    for v in row.iter_mut() { *v = (*v as i32 * factor / divisor) as i16; }
                }
            }
        }
    }

    /// Update history with gravity (bonus capped, decayed toward zero).
    pub fn update_history(entry: &mut i32, bonus: i32) {
        let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
        *entry += clamped - *entry * clamped.abs() / MAX_HISTORY;
    }

    /// Update continuation history (i16 entries) with gravity.
    /// Uses same formula as update_history but with i16 values and MAX_HISTORY divisor.
    pub fn update_cont_history(entry: &mut i16, bonus: i32) {
        let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
        let val = *entry as i32;
        let new_val = val + clamped - val * clamped.abs() / MAX_HISTORY;
        *entry = new_val.clamp(-32000, 32000) as i16;
    }

    /// Update cont-hist with gravity factor derived from a combined "base"
    /// score (typically cont_hist + main_hist / 2). Stormphrax T6 pattern
    /// (history.h:120) — gravity uses the move's combined signal strength
    /// instead of just the cell's own value, so cont-hist can converge
    /// even when main_hist already encodes the move's quality.
    pub fn update_cont_history_with_base(entry: &mut i16, base: i32, bonus: i32) {
        let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
        let val = *entry as i32;
        let new_val = val + clamped - base * clamped.abs() / MAX_HISTORY;
        *entry = new_val.clamp(-32000, 32000) as i16;
    }
}

/// Map a Coda piece (0-11, color*6+pt) to history piece index (1-12).
/// White 1-6 (Pawn..King), Black 7-12 (Pawn..King).
/// Coda: White 0-5, Black 6-11.
/// Mapping: coda_piece + 1.
#[inline(always)]
pub fn go_piece(p: Piece) -> usize {
    debug_assert!(p < 12, "go_piece called with NO_PIECE");
    (p + 1) as usize
}

/// Map a piece type (0-5: PAWN..KING) to captured type index (1-6).
/// 0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king.
/// Coda PieceType: 0=PAWN, 1=KNIGHT, 2=BISHOP, 3=ROOK, 4=QUEEN, 5=KING.
/// Mapping: pt + 1.
#[inline(always)]
pub fn captured_type(pt: PieceType) -> usize {
    debug_assert!(pt <= 5, "captured_type called with NO_PIECE_TYPE");
    (pt + 1) as usize
}

/// MovePicker stages.
///
/// Discriminants are explicit so that the four "picking" stages
/// (GoodCaptures, Quiets, BadCaptures, Evasions) form a single contiguous
/// range starting at PICK_BASE. The hot-path check in `next()` becomes
/// `(stage as u8).wrapping_sub(PICK_BASE) < 4` — a single subtract+compare
/// instead of an OR of four equalities.
#[derive(PartialEq, Eq, Clone, Copy)]
#[repr(u8)]
enum Stage {
    // Non-picking transitions
    TTMove = 0,
    GenerateCaptures = 1,
    GenerateQuiets = 2,
    Done = 3,
    EvasionTTMove = 4,
    GenerateEvasions = 5,

    // Picking stages — contiguous range [PICK_BASE..PICK_BASE+4)
    GoodCaptures = 16,
    Quiets = 17,
    BadCaptures = 18,
    Evasions = 19,
}

const PICK_BASE: u8 = Stage::GoodCaptures as u8;

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    // Pointer to the History struct (lives for the duration of search)
    history: *const History,
    // Continuation history sub-table pointers at plies 1, 2, 4, 6 back.
    // cont_hist_subs[0] = ply-1 (3x weight), [1] = ply-2 (3x), [2] = ply-4 (1x), [3] = ply-6 (1x)
    cont_hist_subs: [Option<*const [[i16; 64]; 13]>; 4],
    pawn_hist_ptr: Option<*const [[i16; 64]; 13]>,
    // Main moves list and scores.
    moves: MoveList,
    /// Scores parallel to `moves`. `[MaybeUninit<i32>; 256]` rather than
    /// `[i32; 256]` to skip the 1KB zero-init per picker construction —
    /// same memset-skip pattern as `MoveList` (movegen.rs). Invariant:
    /// every `moves.push(m)` in the generate/restore paths is paired with
    /// a `scores[idx].write(..)` for the same index, so slots
    /// `[0..moves.len)` are always initialized before `pick_best` reads
    /// them, and reads never go beyond `moves.len`.
    scores: [std::mem::MaybeUninit<i32>; 256],
    index: usize,
    /// Bad captures saved from the good/bad partition. Same
    /// writes-before-reads invariant: `generate_and_score_captures`
    /// writes slot `bad_len` of both arrays before incrementing
    /// `bad_len`, and `restore_bad_captures` only reads `[0..bad_len)`.
    bad_moves: [std::mem::MaybeUninit<Move>; 256],
    bad_scores: [std::mem::MaybeUninit<i32>; 256],
    bad_len: usize,
    pub skip_quiet: bool,
    threats: Threats, // enemy attack bitboard for threat-aware history
    // B1: our own pieces blocking a slider's attack on an enemy piece.
    // Moving one of these creates a discovered attack.
    xray_blockers: Bitboard,
    // Evasion support
    checkers: Bitboard,
    pinned: Bitboard,
    // NMP threat square (-1 = none)
    pub threat_sq: i32,
    // Checking squares: from which squares does each piece type give direct check?
    // Indexed by piece type (0=PAWN..5=KING). Computed once per node.
    checking_sqs: [Bitboard; 6],
}

impl MovePicker {
    /// Create a new MovePicker for main search (non-evasion).
    /// Initialize for main search.
    ///
    /// `checkers`/`pinned` are passed in (search already computes both per
    /// node) instead of recomputed here — consistent with `new_evasion`.
    pub fn new(
        _board: &Board,
        tt_move: Move,
        ply: usize,
        checkers: Bitboard,
        pinned: Bitboard,
        history: &History,
        _prev_move: Move,
        pawn_hist: Option<&[[i16; 64]; 13]>,
        threats: Threats,
        xray_blockers: Bitboard,
        moved_piece_stack: &[u8],
        moved_to_stack: &[u8],
    ) -> Self {
        // Get continuation history sub-table pointers at plies 1, 2, 4, 6 back.
        // Uses moved_piece_stack for correct piece lookup (avoids stale board.piece_at).
        // Upper-bound guard: callers (search + qsearch) should clamp ply but
        // we defend here too — indexing out of range panics the search thread.
        let mut cont_hist_subs: [Option<*const [[i16; 64]; 13]>; 4] = [None; 4];
        let offsets = [1usize, 2, 4, 6];
        for (i, &off) in offsets.iter().enumerate() {
            if ply >= off && ply - off < moved_piece_stack.len() && ply - off < moved_to_stack.len() {
                let prior_piece = moved_piece_stack[ply - off] as usize;
                let prior_to = moved_to_stack[ply - off] as usize;
                if prior_piece > 0 && prior_piece < 13 && prior_to < 64 {
                    cont_hist_subs[i] = Some(&history.cont_hist[prior_piece][prior_to] as *const [[i16; 64]; 13]);
                }
            }
        }

        let pawn_hist_ptr = pawn_hist.map(|ph| ph as *const [[i16; 64]; 13]);

        MovePicker {
            stage: Stage::TTMove,
            tt_move,
            history: history as *const History,
            cont_hist_subs,
            pawn_hist_ptr,
            moves: MoveList::new(),
            // SAFETY: `[MaybeUninit<_>; N]::uninit().assume_init()` is sound —
            // each slot is itself a `MaybeUninit`, which has no validity
            // invariants. Reads are gated by the writes-before-reads
            // invariants documented on the field declarations.
            scores: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            index: 0,
            bad_moves: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            bad_scores: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            bad_len: 0,
            skip_quiet: false,
            threats,
            xray_blockers,
            checkers,
            pinned,
            threat_sq: -1,
            // Deferred: computed once at the start of
            // generate_and_score_quiets — only quiet scoring consumes it,
            // so nodes that cut off before the quiet stage never pay the
            // 4 attack computations.
            checking_sqs: [0; 6],
        }
    }

    /// Create a MovePicker for quiescence search (captures only).
    /// Initialize for quiescence search.
    pub fn new_quiescence(
        tt_move: Move,
        history: &History,
        // Passed in (QS already computes both per node) instead of
        // recomputed here — consistent with MovePicker::new / new_evasion
        // (#1923 dedup pattern; audit P1).
        checkers: Bitboard,
        pinned: Bitboard,
    ) -> Self {
        MovePicker {
            stage: Stage::TTMove,
            tt_move,
            history: history as *const History,
            cont_hist_subs: [None; 4],
            pawn_hist_ptr: None,
            moves: MoveList::new(),
            // SAFETY: see MovePicker::new — uninit MaybeUninit arrays are
            // sound; reads gated by writes-before-reads field invariants.
            scores: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            index: 0,
            bad_moves: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            bad_scores: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            bad_len: 0,
            skip_quiet: true,
            threats: 0,
            xray_blockers: 0,
            // Real pin/check masks so the TTMove-stage is_legal check works.
            // Were hardcoded 0 (fixed 2026-06-11) — same bug class as the
            // main-picker fix of 2026-04-26 (#890 +3.4): with pinned=0,
            // is_legal cannot reject pinned-piece TT moves in QS.
            checkers,
            pinned,
            threat_sq: -1,
            checking_sqs: [0; 6], // not used in QS
        }
    }

    /// Create a MovePicker for evasion mode (when in check).
    /// Initialize for evasion generation.
    /// Evasion moves are generated as all moves then filtered for legality during generation.
    pub fn new_evasion(
        tt_move: Move,
        ply: usize,
        checkers: Bitboard,
        pinned: Bitboard,
        history: &History,
        _prev_move: Move,
        pawn_hist: Option<&[[i16; 64]; 13]>,
        threats: Threats,
        moved_piece_stack: &[u8],
        moved_to_stack: &[u8],
    ) -> Self {
        // Build cont-hist pointers for evasion (same as main picker).
        // Also guard the upper bound: qsearch can deepen past MAX_PLY via
        // evasion chains, and the caller's clamp might be missed — indexing
        // moved_piece_stack with ply >= len panics the search thread.
        let mut cont_hist_subs: [Option<*const [[i16; 64]; 13]>; 4] = [None; 4];
        let offsets = [1usize, 2, 4, 6];
        for (i, &off) in offsets.iter().enumerate() {
            if ply >= off && ply - off < moved_piece_stack.len() && ply - off < moved_to_stack.len() {
                let prior_piece = moved_piece_stack[ply - off] as usize;
                let prior_to = moved_to_stack[ply - off] as usize;
                if prior_piece > 0 && prior_piece < 13 && prior_to < 64 {
                    cont_hist_subs[i] = Some(&history.cont_hist[prior_piece][prior_to] as *const [[i16; 64]; 13]);
                }
            }
        }

        let pawn_hist_ptr = pawn_hist.map(|ph| ph as *const [[i16; 64]; 13]);

        MovePicker {
            stage: Stage::EvasionTTMove,
            tt_move,
            history: history as *const History,
            cont_hist_subs,
            pawn_hist_ptr,
            moves: MoveList::new(),
            // SAFETY: see MovePicker::new — uninit MaybeUninit arrays are
            // sound; reads gated by writes-before-reads field invariants.
            scores: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            index: 0,
            bad_moves: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            bad_scores: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            bad_len: 0,
            skip_quiet: false,
            // C8 audit LIKELY #19: evasion history READS must use the same
            // enemy_attacks key as beta-cutoff WRITES. Previously hardcoded
            // to 0, which hashed into a different 4D history slot than the
            // writes — history written from in-check cutoffs was invisible
            // to in-check reads. Reckless/SF keep reads and writes
            // symmetric.
            threats,
            xray_blockers: 0, // evasions don't use discovered-attack bonus
            checkers,
            pinned,
            threat_sq: -1,
            checking_sqs: [0; 6], // not used in evasions
        }
    }

    /// Get the next move to try. Returns NO_MOVE when exhausted.
    /// No legality checks — caller must check legality.
    /// Get next move in staged order.
    ///
    /// Fast path: when we're in one of the four picking stages
    /// (GoodCaptures/Quiets/BadCaptures/Evasions) AND there are moves left,
    /// return pick_best directly. Steady-state move iteration sits here for
    /// most calls; bypassing the 13-arm Stage match saves the bounds-check +
    /// indirect-jump pair on every call.
    #[inline(always)]
    pub fn next(&mut self, board: &Board) -> Move {
        // Picking stages have contiguous discriminants — see Stage definition.
        if self.index < self.moves.len
            && (self.stage as u8).wrapping_sub(PICK_BASE) < 4
        {
            return self.pick_best();
        }
        self.next_slow(board)
    }

    #[inline(never)]
    fn next_slow(&mut self, board: &Board) -> Move {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateCaptures;
                    // Validate is_legal in addition to is_pseudo_legal: pseudo-legal
                    // accepts king-into-attacked-square + pinned-piece-off-line,
                    // and Coda's make_move doesn't verify king safety. Without
                    // is_legal, an illegal TT move can reach pv_table[0][0] and
                    // be emitted as bestmove (lichess 2agDftuq 2026-04-29 forfeit).
                    if self.tt_move != NO_MOVE
                        && is_pseudo_legal(board, self.tt_move)
                        && board.is_legal(self.tt_move, self.pinned, self.checkers)
                    {
                        return self.tt_move;
                    }
                }

                Stage::GenerateCaptures => {
                    self.generate_and_score_captures(board);
                    self.stage = Stage::GoodCaptures;
                    self.index = 0;
                }

                Stage::GoodCaptures => {
                    // TT move already filtered during scoring
                    if self.index < self.moves.len {
                        return self.pick_best();
                    }
                    if self.skip_quiet {
                        self.stage = Stage::BadCaptures;
                        self.restore_bad_captures();
                    } else {
                        self.stage = Stage::GenerateQuiets;
                    }
                }

                Stage::GenerateQuiets => {
                    self.generate_and_score_quiets(board);
                    self.stage = Stage::Quiets;
                }

                Stage::Quiets => {
                    // TT move already filtered during scoring.
                    if self.index < self.moves.len {
                        return self.pick_best();
                    }
                    self.stage = Stage::BadCaptures;
                    self.restore_bad_captures();
                }

                Stage::BadCaptures => {
                    // TT move already filtered during capture scoring
                    if self.index < self.moves.len {
                        return self.pick_best();
                    }
                    self.stage = Stage::Done;
                }

                Stage::Done => {
                    return NO_MOVE;
                }

                // Evasion stages
                Stage::EvasionTTMove => {
                    self.stage = Stage::GenerateEvasions;
                    if self.tt_move != NO_MOVE && is_pseudo_legal(board, self.tt_move)
                        && board.is_legal(self.tt_move, self.pinned, self.checkers) {
                            return self.tt_move;
                        }
                }

                Stage::GenerateEvasions => {
                    self.generate_and_score_evasions(board);
                    self.stage = Stage::Evasions;
                }

                Stage::Evasions => {
                    // TT move already filtered during evasion scoring
                    if self.index < self.moves.len {
                        return self.pick_best();
                    }
                    self.stage = Stage::Done;
                }
            }
        }
    }

    /// Generate all captures, partition into good (SEE >= 0) and bad (SEE < 0).
    /// Generate and score captures. TT move is filtered out.
    fn generate_and_score_captures(&mut self, board: &Board) {
        let caps = generate_captures(board);
        self.moves = MoveList::new();
        self.bad_len = 0;

        let history = unsafe { &*self.history };

        for i in 0..caps.len {
            let m = caps.get(i);
            if m == self.tt_move {
                continue;
            }
            // Dynamic SEE threshold: captures with strong history get a more
            // forgiving threshold. Use captHist only (not MVV) to avoid inflation.
            let capt_hist = capt_hist_score_static(board, history, m);
            let cap_score = mvv_lva(board, m) + capt_hist;
            let see_threshold = -capt_hist / 18;
            if !see_ge(board, m, see_threshold) {
                // Bad capture.
                // C8 audit LIKELY #24: limit raised to 256 (from 64). 64
                // could silently drop moves in pathological tactical
                // positions (multiple queens + rooks with many captures).
                if self.bad_len < 256 {
                    // Write slot bad_len before incrementing — upholds the
                    // [0..bad_len) initialized invariant.
                    self.bad_moves[self.bad_len].write(m);
                    self.bad_scores[self.bad_len].write(cap_score);
                    self.bad_len += 1;
                }
            } else {
                // Good capture
                let idx = self.moves.len;
                self.moves.push(m);
                self.scores[idx].write(cap_score);
            }
        }
        self.index = 0;
    }

    /// Generate and score quiets. TT move is filtered out.
    fn generate_and_score_quiets(&mut self, board: &Board) {
        // Checking squares: from which squares does each piece type give
        // direct check? Computed here (not in MovePicker::new) because only
        // quiet scoring consumes it — nodes that cut off before the quiet
        // stage never pay for the attack computations.
        {
            let opponent = if board.side_to_move == 0 { 1u8 } else { 0u8 };
            let their_king_bb = board.pieces[KING as usize] & board.colors[opponent as usize];
            self.checking_sqs = if their_king_bb != 0 {
                let their_king_sq = their_king_bb.trailing_zeros();
                let occ = board.occupied();
                let bishop = bishop_attacks(their_king_sq, occ);
                let rook = rook_attacks(their_king_sq, occ);
                [
                    pawn_attacks(opponent, their_king_sq), // PAWN
                    knight_attacks(their_king_sq),         // KNIGHT
                    bishop,                                // BISHOP
                    rook,                                  // ROOK
                    bishop | rook,                         // QUEEN
                    0, // KING (can't give direct check)
                ]
            } else {
                [0; 6]
            };
        }

        let quiets = generate_quiets(board);
        self.moves = MoveList::new();

        let history = unsafe { &*self.history };

        // Per-node invariants hoisted out of the scoring loop: atomic
        // tunable loads and board-derived bitboards. Pure loads with no
        // side effects — the per-move scoring arithmetic is unchanged.
        use std::sync::atomic::Ordering;
        let cm = crate::search::tp10(&crate::search::CONT_HIST_MULT_10X);
        let cont_weights = [cm, cm, 1i32, 1]; // ply-1, ply-2, ply-4, ply-6
        let pw = crate::search::tp10(&crate::search::PAWN_HIST_MULT_10X);
        let null_threat_escape_bonus = crate::search::NULL_THREAT_ESCAPE_BONUS.load(Ordering::Relaxed);
        let escape_bonus_q = crate::search::ESCAPE_BONUS_Q.load(Ordering::Relaxed);
        let escape_bonus_r = crate::search::ESCAPE_BONUS_R.load(Ordering::Relaxed);
        let escape_bonus_minor = crate::search::ESCAPE_BONUS_MINOR.load(Ordering::Relaxed);
        let quiet_check_bonus = crate::search::QUIET_CHECK_BONUS.load(Ordering::Relaxed);
        let discovered_attack_bonus = crate::search::DISCOVERED_ATTACK_BONUS.load(Ordering::Relaxed);
        let mobility_delta_weight = crate::search::MOBILITY_DELTA_WEIGHT.load(Ordering::Relaxed);
        let kf_bonus = crate::search::KNIGHT_FORK_BONUS.load(Ordering::Relaxed);
        let us = board.side_to_move;
        let them = 1 - us;
        let occ = board.colors[us as usize] | board.colors[them as usize];
        let enemy_non_pawns = board.colors[them as usize]
            & !(board.pieces[PAWN as usize] | board.pieces[KING as usize]);
        let their_pawns = board.pieces[PAWN as usize] & board.colors[them as usize];
        let enemy_pawn_attacks = if them == WHITE {
            ((their_pawns & !FILE_A) << 7) | ((their_pawns & !FILE_H) << 9)
        } else {
            ((their_pawns & !FILE_A) >> 9) | ((their_pawns & !FILE_H) >> 7)
        };

        for i in 0..quiets.len {
            let m = quiets.get(i);
            if m == self.tt_move {
                continue;
            }

            let from = move_from(m);
            let to = move_to(m);
            let piece = board.piece_at(from);
            // Hoisted: piece_type_at(from) was previously recomputed in up
            // to 4 score-term blocks below.
            let pt = board.piece_type_at(from);

            let mut score = history.main_score(from, to, self.threats);

            // Continuation history: plies 1,2 at CONT_HIST_MULT weight, plies 4,6 at 1x weight.
            // Matches Obsidian/Alexandria/Berserk pattern (default 3).
            if piece != NO_PIECE {
                let gp = go_piece(piece);
                for (i, &w) in cont_weights.iter().enumerate() {
                    if let Some(sub_ptr) = self.cont_hist_subs[i] {
                        let sub = unsafe { &*sub_ptr };
                        score += w * sub[gp][to as usize] as i32;
                    }
                }
            }

            // Pawn history (weight tunable via PAWN_HIST_MULT_10X)
            if let Some(ph_ptr) = self.pawn_hist_ptr {
                if piece != NO_PIECE {
                    let ph = unsafe { &*ph_ptr };
                    score += pw * ph[go_piece(piece)][to as usize] as i32;
                }
            }

            // Null-move threat: bonus for escaping the threatened square
            if self.threat_sq >= 0 && from as i32 == self.threat_sq {
                score += null_threat_escape_bonus;
            }

            // Escape-capture bonus: bonus for moving a piece off a threatened
            // square (Reckless pattern). All four (Q/R/B/N) now tunable.
            if self.threats & (1u64 << from) != 0 && piece != NO_PIECE {
                score += match pt {
                    4 => escape_bonus_q,
                    3 => escape_bonus_r,
                    1 | 2 => escape_bonus_minor,
                    _ => 0,
                };
            }

            // Quiet check bonus: moves that give direct check (SF +16384, Viridithas +10000)
            if piece != NO_PIECE
                && pt < 6 && self.checking_sqs[pt as usize] & (1u64 << to) != 0 {
                score += quiet_check_bonus;
            }

            // B1: Discovered-attack bonus. If `from` is one of our pieces
            // currently blocking our slider's attack on an enemy, moving
            // it uncovers that attack. Flat bonus — victim-value scaling
            // is a follow-up if H1 resolves.
            if self.xray_blockers & (1u64 << from) != 0 {
                score += discovered_attack_bonus;
            }

            // T2.3 (next_ideas_2026-04-21): mobility-delta quiet-ordering bonus.
            // popcount(attacks_from(piece, to)) - popcount(attacks_from(piece, from))
            // Positive = piece attacks more squares after the move (centralization
            // heuristic, implicit without writing an eval term). Small weight —
            // additive to history. King mobility delta excluded (king activity
            // is eval-handled, not ordering-relevant in pre-endgame).
            if piece != NO_PIECE {
                if pt < 5 { // skip king (pt==5)
                    // NOTE: deliberately (piece >> 3) & 1, NOT side_to_move —
                    // preserved exactly from the pre-hoist code (renamed from
                    // `us` to avoid shadowing the hoisted side_to_move local).
                    let mob_color = (piece >> 3) & 1;
                    // For sliders, to_mob must be computed against POST-move
                    // occupancy or the FROM square (which is on a ray from TO
                    // for any natural slider move) acts as a phantom blocker.
                    // Pre-fix `to_mob` systematically undercounts slider attacks;
                    // SPSA drifted MOBILITY_DELTA_WEIGHT toward 0 because the
                    // signal was partially noise. Knights/pawns are blocker-
                    // independent so unaffected by occupancy choice, but the
                    // unified post_occ keeps the branch clean.
                    let post_occ = (occ ^ (1u64 << from)) | (1u64 << to);
                    let from_mob = crate::threats::piece_attacks_occ(pt, mob_color, from as u32, occ).count_ones();
                    let to_mob = crate::threats::piece_attacks_occ(pt, mob_color, to as u32, post_occ).count_ones();
                    score += (to_mob as i32 - from_mob as i32) * mobility_delta_weight;
                }
            }

            // Reckless "offense bonus": quiet move that lands on a square
            // attacking an enemy non-pawn piece. +6000 flat. Not yet present
            // in Coda; Reckless has it at ~+6000. Signal: does our piece on
            // `to` attack an enemy worth threatening?
            // Safety filter: skip if `to` is attacked by any lower-value enemy
            // piece (the capture back would be net negative for us).
            if piece != NO_PIECE {
                if pt < 6 {
                    // We'd be on `to` after the move; compute attacks from `to` by our piece type.
                    let attacks_from_to = match pt {
                        0 => pawn_attacks(us, to as u32),  // pawn
                        1 => knight_attacks(to as u32),
                        2 => bishop_attacks(to as u32, occ & !(1u64 << from)),  // bishop: occ minus our from-square
                        3 => rook_attacks(to as u32, occ & !(1u64 << from)),    // rook
                        4 => queen_attacks(to as u32, occ & !(1u64 << from)),   // queen
                        _ => 0,  // king — no offense bonus, too risky
                    };
                    if attacks_from_to & enemy_non_pawns != 0 {
                        // Safety check: skip if `to` is attacked by enemy pawn
                        // (which could recapture us).
                        // Only skip if WE would be a bigger target than a pawn
                        let unsafe_square = pt != 0 && (enemy_pawn_attacks & (1u64 << to)) != 0;
                        if !unsafe_square {
                            score += 6000;
                            // T3.2: "good quiet" — the offense move's
                            // strongest target is more valuable than us.
                            // Cheap proxy for positive quiet-SEE.
                            // QSEE_BONUS hardcoded 6687 after ablation #1257
                            // H0 at [-3, 3] (-2.1 ±3.0, ~+2 Elo load-bearing).
                            {
                                let our_val = see_value(pt);
                                let mut hits = attacks_from_to & enemy_non_pawns;
                                let mut max_t_val = 0;
                                while hits != 0 {
                                    let t_sq = hits.trailing_zeros() as u8;
                                    hits &= hits - 1;
                                    let t_pt = board.piece_type_at(t_sq);
                                    if t_pt < 6 {
                                        let v = see_value(t_pt);
                                        if v > max_t_val { max_t_val = v; }
                                    }
                                }
                                if max_t_val > our_val {
                                    score += 6687;
                                }
                            }
                        }
                        // Knight-fork bonus: knight move attacking 2+ enemy
                        // non-pawn pieces from `to` is a fork. Tunable
                        // (KNIGHT_FORK_BONUS), stacks on top of offense.
                        if kf_bonus > 0 && pt == 1 && !unsafe_square
                            && popcount(attacks_from_to & enemy_non_pawns) >= 2 {
                            score += kf_bonus;
                        }
                        // T1.4 Battery bonus removed 2026-05-17: ablation
                        // #1278 H0 at [0, 3] (+0.2 ±1.1 at 114K games).
                        // Feature confirmed neutral; code path deleted.
                    }
                }
            }

            let idx = self.moves.len;
            self.moves.push(m);
            self.scores[idx].write(score);
        }
        self.index = 0;
    }

    /// Generate evasion moves and score them.
    /// Captures scored above quiets. TT move filtered out.
    /// Generate and score evasions.
    ///
    /// Since Coda doesn't have a dedicated generate_evasions function,
    /// we use generate_all_moves and filter for legality here.
    fn generate_and_score_evasions(&mut self, board: &Board) {
        let all = generate_all_moves(board);
        self.moves = MoveList::new();

        let history = unsafe { &*self.history };

        // Per-node tunable loads hoisted out of the scoring loop (pure
        // atomic loads — quiet-branch arithmetic unchanged).
        let cm = crate::search::tp10(&crate::search::CONT_HIST_MULT_10X);
        let cont_weights = [cm, cm, 1i32, 1]; // ply-1, ply-2, ply-4, ply-6
        let pw = crate::search::tp10(&crate::search::PAWN_HIST_MULT_10X);

        for i in 0..all.len {
            let m = all.get(i);
            if m == self.tt_move {
                continue;
            }
            // Evasions must be legal
            if !board.is_legal(m, self.pinned, self.checkers) {
                continue;
            }

            let from = move_from(m);
            let to = move_to(m);
            let flags = move_flags(m);

            // C8 audit LIKELY #26: check capture FIRST so capture-promotions
            // (e.g. pawn-takes-and-promotes) get the capture score path, not
            // the flat 9000 promotion score that ranked them BELOW regular
            // captures (10000+MVV+hist).
            let is_cap = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
            let score = if is_cap {
                // Capture (possibly also a promotion): MVV-LVA + capture
                // history. mvv_lva now adds the promotion material delta
                // internally (audit #25), so capture-promotions rank above
                // regular captures.
                10000 + mvv_lva(board, m) + capt_hist_score_static(board, history, m)
            } else if is_promotion(m) {
                if flags == FLAG_PROMOTE_Q {
                    9000
                } else {
                    -1000 // underpromotions
                }
            } else {
                // Quiet: history + continuation history + pawn history
                let piece = board.piece_at(from);

                let mut s = history.main_score(from, to, self.threats);

                if piece != NO_PIECE {
                    let gp = go_piece(piece);
                    for (i, &w) in cont_weights.iter().enumerate() {
                        if let Some(sub_ptr) = self.cont_hist_subs[i] {
                            let sub = unsafe { &*sub_ptr };
                            s += w * sub[gp][to as usize] as i32;
                        }
                    }
                }

                if let Some(ph_ptr) = self.pawn_hist_ptr {
                    if piece != NO_PIECE {
                        let ph = unsafe { &*ph_ptr };
                        s += pw * ph[go_piece(piece)][to as usize] as i32;
                    }
                }

                s
            };

            let idx = self.moves.len;
            self.moves.push(m);
            self.scores[idx].write(score);
        }
        self.index = 0;
    }

    /// Swap in the saved bad captures.
    fn restore_bad_captures(&mut self) {
        self.moves = MoveList::new();
        for i in 0..self.bad_len {
            // SAFETY: generate_and_score_captures wrote bad_moves[i] and
            // bad_scores[i] for all i < bad_len before incrementing bad_len.
            self.moves.push(unsafe { self.bad_moves[i].assume_init() });
            self.scores[i].write(unsafe { self.bad_scores[i].assume_init() });
        }
        self.index = 0;
    }

    /// Abandon the remaining quiets and move straight to bad captures.
    ///
    /// Called by search when LMP fires: every remaining quiet would be
    /// discarded by the caller anyway, so keeping them in the selection
    /// pool costs O(n²) pick_best scans per node. Replicates exactly the
    /// Quiets-exhausted handoff in next_slow (stage = BadCaptures +
    /// restore_bad_captures). The old `picker.skip_quiet = true` at the
    /// LMP site was dead: that flag is only consulted at the
    /// GoodCaptures → GenerateQuiets transition, which has already
    /// happened by the time LMP fires on a quiet move.
    ///
    /// Behavior note: post-LMP quiets no longer reach search at all, so
    /// they no longer inflate move_count for the bad captures that
    /// follow (previously each discarded quiet bumped move_count before
    /// the skip check, deepening LMR on late bad captures).
    ///
    /// No-op outside the Quiets stage (guard for edge orderings).
    pub fn skip_remaining_quiets(&mut self) {
        if self.stage == Stage::Quiets {
            self.stage = Stage::BadCaptures;
            self.restore_bad_captures();
        }
    }

    /// Selection sort: find best from current index, swap to front, return it.
    /// Selection sort: find best scored move and swap to front.
    fn pick_best(&mut self) -> Move {
        if self.index >= self.moves.len {
            return NO_MOVE;
        }

        let mut best_idx = self.index;
        // SAFETY (all assume_init below): scores[0..moves.len) are written
        // by the generate/restore paths before any move is exposed — see
        // the invariant on the `scores` field declaration. index < moves.len
        // is checked above and the loop bound is moves.len.
        let mut best_score = unsafe { self.scores[self.index].assume_init() };

        for i in (self.index + 1)..self.moves.len {
            let s = unsafe { self.scores[i].assume_init() };
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }

        if best_idx != self.index {
            self.moves.swap(self.index, best_idx);
            // Swapping MaybeUninit slots is safe — no reads of the contents.
            self.scores.swap(self.index, best_idx);
        }

        let mv = self.moves.get(self.index);
        self.index += 1;
        mv
    }

}

/// Capture history score for a capture move.
/// Capture history score lookup. Public for use by QMovePicker.
#[inline]
pub fn capt_hist_score_static(board: &Board, history: &History, m: Move) -> i32 {
    let from = move_from(m);
    let to = move_to(m);
    let piece = board.piece_at(from);
    if piece == NO_PIECE {
        return 0;
    }
    let victim_pt = board.piece_type_at(to);
    let ct = if victim_pt == NO_PIECE_TYPE {
        if move_flags(m) == FLAG_EN_PASSANT {
            1 // pawn
        } else {
            0 // empty
        }
    } else {
        captured_type(victim_pt)
    };
    history.capture[go_piece(piece)][to as usize][ct] as i32
}

/// MVV-LVA score for a capture.
fn mvv_lva(board: &Board, m: Move) -> i32 {
    let to = move_to(m);
    let from = move_from(m);

    let mult = crate::search::MVV_CAP_MULT.load(std::sync::atomic::Ordering::Relaxed);
    let target_pt = board.piece_type_at(to);

    // C8 audit LIKELY #25: non-capture promotions scored 0 in MVV and
    // empty-slot in capt_hist, ranking BELOW any regular capture with a
    // small history score. A queen promotion deserves a large base bonus.
    // Add the promotion material delta (promoted piece - pawn) when the
    // move is a promotion.
    let promo_bonus = if is_promotion(m) {
        let promoted = promotion_piece_type(m);
        (see_value(promoted) - see_value(PAWN)) * mult
    } else {
        0
    };

    if target_pt == NO_PIECE_TYPE {
        // En passant
        if move_flags(m) == FLAG_EN_PASSANT {
            return see_value(PAWN) * mult;
        }
        // Non-capture promotion: promo_bonus is the only contribution.
        return promo_bonus;
    }

    let _attacker_pt = board.piece_type_at(from);

    // MVV only (no LVA), multiplier SPSA-tunable (Obsidian/Alexandria/Berserk default 16)
    see_value(target_pt) * mult + promo_bonus
}

/// Check if a move is a capture.
#[inline(always)]
fn is_capture(board: &Board, m: Move) -> bool {
    board.piece_type_at(move_to(m)) != NO_PIECE_TYPE || move_flags(m) == FLAG_EN_PASSANT
}

/// Re-derive move flags from the board state. TT/killer moves may have stale flags.
pub fn fixup_move_flags(board: &Board, mv: Move) -> Move {
    let from = move_from(mv);
    let to = move_to(mv);

    // Keep promotion flags as-is (they're encoded in the move)
    if is_promotion(mv) {
        return mv;
    }

    let pt = board.piece_type_at(from);

    // Re-derive EP: must be pawn moving to ep_square diagonally (file-adjacent)
    if pt == PAWN && to == board.ep_square && board.ep_square != NO_SQUARE {
        let diff = (to as i32 - from as i32).abs();
        let file_diff = ((from & 7) as i32 - (to & 7) as i32).unsigned_abs();
        if (diff == 7 || diff == 9) && file_diff == 1 {
            return make_move(from, to, FLAG_EN_PASSANT);
        }
    }

    // Re-derive castling: king moving 2 squares
    if pt == KING {
        let diff = (to as i32 - from as i32).abs();
        if diff == 2 {
            return make_move(from, to, FLAG_CASTLE);
        }
    }

    // Normal move (double pushes use FLAG_NONE=0, detected by distance in make_move)
    make_move(from, to, FLAG_NONE)
}

/// Thorough pseudo-legality check for TT moves and defensive PV validation.
/// Must validate all special flags to prevent board corruption.
pub fn is_pseudo_legal(board: &Board, mv: Move) -> bool {
    if mv == NO_MOVE { return false; }
    let from = move_from(mv);
    let to = move_to(mv);
    if from > 63 || to > 63 || from == to { return false; }

    let us = board.side_to_move;
    let them = flip_color(us);
    let from_bb = 1u64 << from;
    let to_bb = 1u64 << to;
    let flags = move_flags(mv);

    // Reject invalid flag values (valid: 0,1,2,4,5,6,7)
    if flags == 3 || flags > FLAG_PROMOTE_Q {
        return false;
    }

    // From square must have our piece
    if from_bb & board.colors[us as usize] == 0 {
        return false;
    }
    let pt = board.piece_type_at(from);
    if pt == NO_PIECE_TYPE { return false; }

    // Must not capture a king
    if to_bb & board.pieces[KING as usize] != 0 {
        return false;
    }

    // En passant: validate thoroughly
    //
    // C1 (2026-04-22 audit): EP requires that `from` is on the EP-capture
    // rank (rank 5 for white, rank 4 for black) AND on a file adjacent to
    // `to`. Without these, a TT-collision move with corrupted from + to +
    // flags (e.g. `a2→d6 FLAG_EN_PASSANT` in a position with ep_square=d6)
    // passes all other checks: `cap_sq` contains the enemy pawn and
    // destination is empty. make_move then teleports our pawn and removes
    // the enemy pawn — same 320 Elo hole class as earlier pseudo-legal bugs.
    if flags == FLAG_EN_PASSANT {
        if pt != PAWN { return false; }
        if to != board.ep_square { return false; }
        let from_rank = from >> 3;
        let required_rank = if us == WHITE { 4 } else { 3 }; // 5th rank = index 4
        if from_rank != required_rank { return false; }
        let from_file = from & 7;
        let to_file = to & 7;
        if (from_file as i8 - to_file as i8).abs() != 1 { return false; }
        // Verify capture square has enemy pawn
        let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
        if cap_sq >= 64 || (1u64 << cap_sq) & board.pieces[PAWN as usize] & board.colors[them as usize] == 0 {
            return false;
        }
        return true;
    }

    // Castling: validate rights, path, ROOK-on-corner, and no attacks on
    // king/intermediate/destination. Rook-on-corner check is the
    // 2026-05-31 audit addition (finding G+H) — defends against TT-
    // collision castles on a board where the rook moved away but the
    // synthetic FEN / corrupt state still has the right set.
    if flags == FLAG_CASTLE {
        if pt != KING { return false; }
        let occ = board.occupied();
        let them_bb = board.colors[flip_color(us) as usize];
        let our_rooks = board.pieces[ROOK as usize] & board.colors[us as usize];
        if us == WHITE {
            if from != 4 { return false; } // king must be on e1
            if to == 6 { // kingside
                if board.castling & CASTLE_WK == 0 { return false; }
                if our_rooks & (1u64 << 7) == 0 { return false; } // rook on h1
                if occ & 0x60 != 0 { return false; }
                // King(e1), f1, g1 must not be attacked
                if board.attackers_to(4, occ) & them_bb != 0 { return false; }
                if board.attackers_to(5, occ) & them_bb != 0 { return false; }
                if board.attackers_to(6, occ) & them_bb != 0 { return false; }
            } else if to == 2 { // queenside
                if board.castling & CASTLE_WQ == 0 { return false; }
                if our_rooks & 1u64 == 0 { return false; } // rook on a1
                if occ & 0x0E != 0 { return false; }
                // King(e1), d1, c1 must not be attacked
                if board.attackers_to(4, occ) & them_bb != 0 { return false; }
                if board.attackers_to(3, occ) & them_bb != 0 { return false; }
                if board.attackers_to(2, occ) & them_bb != 0 { return false; }
            } else { return false; }
        } else {
            if from != 60 { return false; } // king must be on e8
            if to == 62 { // kingside
                if board.castling & CASTLE_BK == 0 { return false; }
                if our_rooks & (1u64 << 63) == 0 { return false; } // rook on h8
                if occ & (0x60u64 << 56) != 0 { return false; }
                // King(e8), f8, g8 must not be attacked
                if board.attackers_to(60, occ) & them_bb != 0 { return false; }
                if board.attackers_to(61, occ) & them_bb != 0 { return false; }
                if board.attackers_to(62, occ) & them_bb != 0 { return false; }
            } else if to == 58 { // queenside
                if board.castling & CASTLE_BQ == 0 { return false; }
                if our_rooks & (1u64 << 56) == 0 { return false; } // rook on a8
                if occ & (0x0Eu64 << 56) != 0 { return false; }
                // King(e8), d8, c8 must not be attacked
                if board.attackers_to(60, occ) & them_bb != 0 { return false; }
                if board.attackers_to(59, occ) & them_bb != 0 { return false; }
                if board.attackers_to(58, occ) & them_bb != 0 { return false; }
            } else { return false; }
        }
        return true;
    }

    // Double push check removed: FLAG_DOUBLE_PUSH=0=FLAG_NONE, detected by distance in make_move

    // Promotion: must be a pawn on the 7th rank (2nd rank for Black)
    if is_promotion(mv) {
        if pt != PAWN { return false; }
        if us == WHITE && (from >> 3) != 6 { return false; }
        if us == BLACK && (from >> 3) != 1 { return false; }
    }
    // Non-promotion pawn moves must not reach back rank (Stockfish pattern)
    if !is_promotion(mv) && pt == PAWN {
        let rank = to >> 3;
        if rank == 0 || rank == 7 { return false; }
    }

    // To square must not have our piece
    if to_bb & board.colors[us as usize] != 0 {
        return false;
    }

    // Geometric validity: verify the piece can reach the destination
    let occ = board.occupied();
    match pt {
        PAWN => {
            // Pawn moves: push, double push, or capture
            let signed_diff = to as i32 - from as i32;
            let diff = signed_diff.unsigned_abs() as i32;
            if diff != 7 && diff != 8 && diff != 9 && diff != 16 {
                return false;
            }
            // Direction check: white pawns move up (positive diff), black down (negative)
            if us == WHITE && signed_diff <= 0 { return false; }
            if us == BLACK && signed_diff >= 0 { return false; }
            // Double push: intermediate square must be empty
            if diff == 16 {
                let mid = ((from as u32 + to as u32) / 2) as u8;
                if occ & (1u64 << mid) != 0 {
                    return false;
                }
                // Must also be from starting rank
                if us == WHITE && (from >> 3) != 1 { return false; }
                if us == BLACK && (from >> 3) != 6 { return false; }
                // Destination must be empty (not a capture)
                if board.piece_type_at(to) != NO_PIECE_TYPE { return false; }
            }
            // Single push: destination must be empty
            if diff == 8
                && board.piece_type_at(to) != NO_PIECE_TYPE { return false; }
            // Capture: destination must have enemy piece and be file-adjacent
            // (EP is handled above with FLAG_EN_PASSANT and returns early)
            if diff == 7 || diff == 9 {
                // File adjacency: prevent wrap from h-file to a-file (or vice versa)
                let from_file = from & 7;
                let to_file = to & 7;
                let file_diff = (from_file as i32 - to_file as i32).unsigned_abs();
                if file_diff != 1 { return false; }
                if board.piece_type_at(to) == NO_PIECE_TYPE {
                    return false;
                }
            }
        }
        KNIGHT => {
            if knight_attacks(from as u32) & to_bb == 0 {
                return false;
            }
        }
        BISHOP => {
            if bishop_attacks(from as u32, occ) & to_bb == 0 {
                return false;
            }
        }
        ROOK => {
            if rook_attacks(from as u32, occ) & to_bb == 0 {
                return false;
            }
        }
        QUEEN => {
            if queen_attacks(from as u32, occ) & to_bb == 0 {
                return false;
            }
        }
        KING => {
            if king_attacks(from as u32) & to_bb == 0 {
                return false;
            }
        }
        _ => return false,
    }

    true
}

/// Move picker for quiescence search.
/// QS move picker: TT move first, then captures scored by MVV-LVA + captHist.
/// When in_check, uses evasion mode (all moves, captures scored above quiets).
pub struct QMovePicker {
    tt_move: Move,
    tt_stage: bool, // true = haven't tried TT move yet
    moves: MoveList,
    /// Scores parallel to `moves`. `[MaybeUninit<i32>; 256]` to skip the
    /// 1KB zero-init per picker (same memset-skip pattern as `MoveList`).
    /// Invariant: the constructor writes scores[i] for every i in
    /// [0..moves.len) (every branch of the scoring loop assigns), and
    /// `next` only reads within [idx..moves.len).
    scores: [std::mem::MaybeUninit<i32>; 256],
    idx: usize,
    pinned: Bitboard,
    checkers: Bitboard,
    in_check: bool,
}

impl QMovePicker {
    /// Create QS picker: TT move first, then captures scored by MVV-LVA + captHist.
    /// When in_check, generates all moves (evasions); otherwise captures only.
    pub fn new(
        board: &Board,
        tt_move: Move,
        in_check: bool,
        history: &History,
        // Passed in — the probcut block runs after the node-entry
        // pinned/checkers computation (audit P bundle).
        pinned: Bitboard,
        checkers: Bitboard,
    ) -> Self {

        let moves = if in_check { generate_all_moves(board) } else { generate_captures(board) };
        let mut picker = QMovePicker {
            tt_move: if tt_move != NO_MOVE && is_pseudo_legal(board, tt_move) { tt_move } else { NO_MOVE },
            tt_stage: true,
            moves,
            // SAFETY: uninit MaybeUninit array is sound; the loop below
            // writes scores[i] for every i < moves.len before any read.
            scores: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            idx: 0,
            pinned,
            checkers,
            in_check,
        };

        // Score moves: MVV-LVA + captHist for captures
        for i in 0..picker.moves.len {
            let mv = picker.moves.get(i);
            // Skip TT move in scoring (will be tried first)
            if mv == picker.tt_move {
                picker.scores[i].write(i32::MIN);
                continue;
            }

            let to = move_to(mv);
            let from = move_from(mv);
            let target_pt = board.piece_type_at(to);
            let flags = move_flags(mv);

            if target_pt != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT {
                // Capture: MVV-LVA + captHist/16
                let victim = if flags == FLAG_EN_PASSANT { PAWN } else { target_pt };
                let attacker = board.piece_type_at(from);
                let mvv_lva = see_value(victim) * 10 - see_value(attacker);
                let capt_hist = capt_hist_score_static(board, history, mv);
                if in_check {
                    // Evasion captures scored high (10000 + mvvlva + captHist)
                    picker.scores[i].write(10000 + mvv_lva + capt_hist);
                } else {
                    picker.scores[i].write(mvv_lva + capt_hist);
                }
            } else if is_promotion(mv) {
                if in_check {
                    let pt = promotion_piece_type(mv);
                    picker.scores[i].write(if pt == QUEEN { 9000 } else { -1000 });
                } else {
                    picker.scores[i].write(see_value(promotion_piece_type(mv)));
                }
            } else {
                // Quiet (only in evasion mode)
                picker.scores[i].write(-1_000_000);
            }
        }

        picker
    }

    /// Get next move. Returns NO_MOVE when exhausted.
    pub fn next(&mut self, board: &Board) -> Move {
        // Try TT move first
        if self.tt_stage {
            self.tt_stage = false;
            if self.tt_move != NO_MOVE
                && board.is_legal(self.tt_move, self.pinned, self.checkers) {
                    return self.tt_move;
                }
        }

        while self.idx < self.moves.len {
            // Selection sort: find best remaining
            let mut best_idx = self.idx;
            // SAFETY (both assume_init): constructor wrote scores[i] for
            // every i < moves.len; reads here stay within [idx..moves.len).
            let mut best_score = unsafe { self.scores[self.idx].assume_init() };
            for j in (self.idx + 1)..self.moves.len {
                let s = unsafe { self.scores[j].assume_init() };
                if s > best_score {
                    best_score = s;
                    best_idx = j;
                }
            }
            self.moves.swap(self.idx, best_idx);
            self.scores.swap(self.idx, best_idx);

            let mv = self.moves.get(self.idx);
            self.idx += 1;

            // Skip TT move (already tried)
            if mv == self.tt_move { continue; }

            // Not in check: only return captures/promotions
            if !self.in_check {
                let to = move_to(mv);
                let flags = move_flags(mv);
                let is_cap = board.piece_type_at(to) != NO_PIECE_TYPE
                    || flags == FLAG_EN_PASSANT
                    || is_promotion(mv);
                if !is_cap {
                    continue;
                }
            }

            if board.is_legal(mv, self.pinned, self.checkers) {
                return mv;
            }
        }

        NO_MOVE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::FEAT_4D_HISTORY;
    use std::sync::atomic::Ordering;

    /// Regression test: FEAT_4D_HISTORY toggles which slice of main hist is read.
    ///
    /// When 4D is on, threats at from/to select among 4 tables:
    /// main[from_threatened][to_threatened][from][to].
    /// When 4D is off, main_score/main_entry must always hit [0][0][...],
    /// regardless of the threats bitboard.
    ///
    /// This guards the A/B experiment branch: a future change to the
    /// 4D indexing must not accidentally corrupt the 2D fallback path.
    #[test]
    fn history_4d_flag_routes_correctly() {
        let mut h = History::boxed_zeroed();
        // Give each table slot a distinct value so we can prove which branch ran.
        h.main[0][0][12][28] = 1;
        h.main[0][1][12][28] = 2;
        h.main[1][0][12][28] = 3;
        h.main[1][1][12][28] = 4;

        // Threats bitboard with BOTH from (12) and to (28) set:
        let threats: Threats = (1u64 << 12) | (1u64 << 28);

        let saved = FEAT_4D_HISTORY.load(Ordering::Relaxed);

        // 4D on: lookup must see slot [1][1] = 4.
        FEAT_4D_HISTORY.store(true, Ordering::Relaxed);
        assert_eq!(h.main_score(12, 28, threats), 4,
            "4D on: expected main[1][1][12][28]=4");
        *h.main_entry(12, 28, threats) = 40;
        assert_eq!(h.main[1][1][12][28], 40, "4D on: main_entry wrote to [1][1]");
        h.main[1][1][12][28] = 4; // restore

        // 4D off: lookup must always see slot [0][0] = 1 regardless of threats.
        FEAT_4D_HISTORY.store(false, Ordering::Relaxed);
        assert_eq!(h.main_score(12, 28, threats), 1,
            "4D off: expected main[0][0][12][28]=1");
        assert_eq!(h.main_score(12, 28, 0), 1,
            "4D off: expected main[0][0] with zero threats");
        *h.main_entry(12, 28, threats) = 10;
        assert_eq!(h.main[0][0][12][28], 10, "4D off: main_entry wrote to [0][0]");
        // The other slots must not have been touched by the 2D write path.
        assert_eq!(h.main[1][1][12][28], 4, "4D off: [1][1] unchanged");

        // Restore original flag so other tests are unaffected.
        FEAT_4D_HISTORY.store(saved, Ordering::Relaxed);
    }

    /// Positive fuzzer: every legal move in every position must pass
    /// `is_pseudo_legal`. If this fails, we're rejecting legal moves
    /// that come from TT slots, losing move-ordering
    /// information and potentially missing key moves.
    ///
    /// Also indirectly: tests that `generate_legal_moves` and
    /// `is_pseudo_legal` agree about what flags a move should have.
    #[test]
    fn fuzz_is_pseudo_legal_accepts_all_legal() {
        use crate::board::Board;
        use crate::movegen::generate_legal_moves;

        crate::init();

        const FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // Promotion-rich
            "4k3/PPPPPPPP/8/8/8/8/pppppppp/4K3 w - - 0 1",
            // EP available
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            // Castling rights all sides
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            // In check, evasions only
            "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
            // Double check, only king moves legal
            "rnb1kbnr/pppp1ppp/8/4p3/1P5q/P1N5/2PPPPPP/R1BQKBNR w KQkq - 2 4",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        const PLIES: usize = 40;
        const GAMES: usize = 8;

        for (fen_idx, fen) in FENS.iter().enumerate() {
            for game in 0..GAMES {
                let seed: u32 = 0xBADF00Du32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::from_fen(fen);
                for ply in 0..PLIES {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    // Check every legal move is accepted.
                    for i in 0..legal.len {
                        let mv = legal.get(i);
                        if !is_pseudo_legal(&board, mv) {
                            panic!(
                                "is_pseudo_legal rejected legal move: fen_idx={} game={} ply={} \
                                 move={} (raw {:#x}) from={} to={} flags={} fen={}",
                                fen_idx, game, ply,
                                crate::types::move_to_uci(mv), mv,
                                crate::types::move_from(mv),
                                crate::types::move_to(mv),
                                crate::types::move_flags(mv),
                                board.to_fen(),
                            );
                        }
                    }
                    // Advance the game with a random legal move.
                    let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);
                    board.make_move(mv);
                }
            }
        }
    }

    /// Negative fuzzer: random corrupted moves should rarely pass
    /// `is_pseudo_legal`, and when they do, they must be in the
    /// pseudo-legal generate_all_moves set. This catches cases where
    /// a crafted (e.g. TT-collision) move with wrong flags could slip
    /// through validation and corrupt the board.
    ///
    /// Strategy: take each legal move, flip various fields (flags,
    /// to-square, from-square) to create a "corrupted" move. Any that
    /// happen to be legitimately pseudo-legal must appear in
    /// generate_all_moves; others must be rejected.
    #[test]
    fn fuzz_is_pseudo_legal_rejects_corrupted() {
        use crate::board::Board;
        use crate::movegen::{generate_all_moves, generate_legal_moves};
        use crate::types::*;

        crate::init();

        const FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // EP available (d6) — edge case for FLAG_EN_PASSANT validation
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            // No castling rights — must reject FLAG_CASTLE moves
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        for (fen_idx, fen) in FENS.iter().enumerate() {
            let board = Board::from_fen(fen);
            let mut seed: u32 = 0xC0FFEEu32.wrapping_add((fen_idx as u32).wrapping_mul(7919));
            let mut rng = if seed == 0 { seed = 1; seed } else { seed };

            // Build the full pseudo-legal set so we can distinguish
            // "wrong flag happens to match another legal move" from
            // "genuinely illegal move incorrectly accepted".
            let pseudo = generate_all_moves(&board);
            let mut pseudo_set: Vec<Move> = (0..pseudo.len).map(|i| pseudo.get(i)).collect();
            pseudo_set.sort();
            pseudo_set.dedup();

            let legal = generate_legal_moves(&board);

            for i in 0..legal.len {
                let mv = legal.get(i);
                let from = move_from(mv);
                let to = move_to(mv);

                // Corruption 1: random flag bit.
                for &new_flags in &[1u8, 2, 3, 4, 5, 6, 7] {
                    let orig_flags = move_flags(mv);
                    if new_flags as u16 == orig_flags { continue; }
                    // Use the underlying encoding: preserve from, to, replace flags.
                    let corrupted = (from as u16) | ((to as u16) << 6) | ((new_flags as u16) << 12);
                    if is_pseudo_legal(&board, corrupted) {
                        // Must appear in the pseudo-legal set.
                        if !pseudo_set.contains(&corrupted) {
                            panic!(
                                "is_pseudo_legal accepted corrupted move: fen_idx={} \n\
                                 orig={} (flags={}) corrupted={:#x} (flags={}) \n\
                                 not in generate_all_moves\nfen={}",
                                fen_idx, crate::types::move_to_uci(mv), orig_flags,
                                corrupted, new_flags, board.to_fen(),
                            );
                        }
                    }
                }

                // Corruption 2: random to-square.
                let random_to = (next_u32(&mut rng) % 64) as u8;
                if random_to != to {
                    let corrupted = (from as u16) | ((random_to as u16) << 6); // FLAG_NONE
                    if is_pseudo_legal(&board, corrupted) && !pseudo_set.contains(&corrupted) {
                        panic!(
                            "is_pseudo_legal accepted corrupted move (to-swap): \n\
                             fen_idx={} orig={} new_to={} corrupted={:#x}\nfen={}",
                            fen_idx, crate::types::move_to_uci(mv), random_to,
                            corrupted, board.to_fen(),
                        );
                    }
                }
            }
        }
    }
}
