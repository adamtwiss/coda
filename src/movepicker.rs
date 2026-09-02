//! Staged move picker for search.
//!
//! Order: TT move -> good captures (MVV-LVA + captHist) -> quiets (history-scored,
//!        including main/cont/pawn/etc.) -> bad captures.
//!
//! Killer/counter stages were removed — quiet ordering relies on history alone
//! (SF pattern, validated by SPRT commit e28c78a).
//!
//! Evasion order: TT move -> evasions (captures scored above quiets).
//!
//! No legality checks — returns pseudo-legal moves; search caller does legality.

use crate::attacks::*;
use crate::bitboard::*;
use crate::board::Board;
use crate::eval::see_value;
use crate::movegen::{generate_captures, generate_evasions, generate_quiets, MoveList};
use crate::see::see_ge;
use crate::types::*;

const MAX_HISTORY: i32 = 16384;

/// Bitboard type alias for threat computation.
pub type Threats = u64;

/// History tables shared across the search.
pub struct History {
    /// Main history: [from_threatened][to_threatened][from][to]
    /// Threat-aware 4D indexing — separate history for moves escaping/entering threats.
    /// i16 storage: the gravity update keeps every entry within ±MAX_HISTORY
    /// (16384), so i16 is exact, and it halves the table from 64 KB to 32 KB.
    /// The 4D read in quiet scoring is scattered (threat bits × from × to), and
    /// at 64 KB the table alone exceeded L1D; perf showed that read as the
    /// hottest cache-miss site in `next_slow`.
    pub main: [[[[i16; 64]; 64]; 2]; 2],
    /// Capture history: [piece 1-12][to][captured_type 0-6]
    /// piece uses 1-12 indexing (slot 0 unused).
    /// captured_type uses 0-6 scheme (0=empty, 1=pawn, ..., 6=king).
    /// int16 values (i32 causes different gravity behavior).
    pub capture: [[[i16; 7]; 64]; 13],
    /// Continuation history: [piece 1-12][to][piece 1-12][to]
    /// piece uses 1-12 indexing (slot 0 unused).
    pub cont_hist: [[[[i16; 64]; 13]; 64]; CONT_PLANES],
}

impl History {
    /// Get main history score for a move given enemy threat bitboard.
    #[inline(always)]
    pub fn main_score(&self, from: u8, to: u8, threats: Threats) -> i32 {
        if crate::search::FEAT_4D_HISTORY.load(std::sync::atomic::Ordering::Relaxed) {
            let ft = ((threats >> from) & 1) as usize;
            let tt = ((threats >> to) & 1) as usize;
            self.main[ft][tt][from as usize][to as usize] as i32
        } else {
            self.main[0][0][from as usize][to as usize] as i32
        }
    }

    /// Get mutable reference to main history entry for a move given enemy threats.
    #[inline(always)]
    pub fn main_entry(&mut self, from: u8, to: u8, threats: Threats) -> &mut i16 {
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
        self.cont_hist = [[[[0; 64]; 13]; 64]; CONT_PLANES];
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
                    for v in row.iter_mut() { *v = (*v as i32 * factor / divisor) as i16; }
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

    /// Update main history with gravity (bonus capped, decayed toward zero).
    /// Computed in i32 and stored as i16: with |entry| <= MAX_HISTORY and
    /// |clamped| <= MAX_HISTORY the result is again within ±MAX_HISTORY, so
    /// the narrowing is exact (no clamp needed, unlike `update_cont_history`).
    pub fn update_history(entry: &mut i16, bonus: i32) {
        let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
        let val = *entry as i32;
        let new_val = val + clamped - val * clamped.abs() / MAX_HISTORY;
        debug_assert!(new_val.abs() <= MAX_HISTORY);
        *entry = new_val as i16;
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
    /// score (typically cont_hist + main_hist / 2). A cont-hist blend
    /// technique from Stormphrax — gravity uses the move's combined signal strength
    /// instead of just the cell's own value, so cont-hist can converge
    /// even when main_hist already encodes the move's quality.
    pub fn update_cont_history_with_base(entry: &mut i16, base: i32, bonus: i32) {
        let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
        let val = *entry as i32;
        let new_val = val + clamped - base * clamped.abs() / MAX_HISTORY;
        *entry = new_val.clamp(-32000, 32000) as i16;
    }
}

/// Number of continuation-history *planes* — the size of `cont_hist`'s FIRST
/// dimension, which selects the sub-table by the PRIOR move.
///
/// Every cont-hist read and write derives its sub-table from
/// `moved_piece_stack[ply]`, so this constant is the single source of truth for
/// the bound check at each of those sites. Bound-check against this constant
/// (or `cont_hist.len()`), never a literal `13`: a plane-count change that
/// missed even one hardcoded site would silently drop writes into the new
/// planes rather than failing loudly.
pub const CONT_PLANES: usize = 13;

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
    /// QS mode (SF QCAPTURE shape): skip the SEE partition entirely — every
    /// capture goes to the single picking stage ordered by MVV+captHist, and
    /// the caller's per-move SEE gate is the only exchange evaluation. The
    /// main search keeps the partition (its bad-capture stage is load-bearing
    /// for move_count/LMR interactions); QS's is not, since QS filters
    /// SEE-bad moves anyway.
    no_see_partition: bool,
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
                if prior_piece > 0 && prior_piece < CONT_PLANES && prior_to < 64 {
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
            no_see_partition: false,
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
    pub fn new_quiescence(
        tt_move: Move,
        history: &History,
        // Passed in (QS already computes both per node) instead of
        // recomputed here — consistent with MovePicker::new / new_evasion.
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
            no_see_partition: true,
            threats: 0,
            xray_blockers: 0,
            // Real pin/check masks so the TTMove-stage is_legal check works.
            // These must not be hardcoded to 0: with pinned=0, is_legal cannot
            // reject pinned-piece TT moves in QS.
            checkers,
            pinned,
            threat_sq: -1,
            checking_sqs: [0; 6], // not used in QS
        }
    }

    /// Create a MovePicker for evasion mode (when in check).
    /// Initialize for evasion generation.
    /// Evasion moves are generated by the dedicated legal evasion generator.
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
                if prior_piece > 0 && prior_piece < CONT_PLANES && prior_to < 64 {
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
            no_see_partition: false,
            // Evasion history READS must use the same enemy_attacks key as
            // beta-cutoff WRITES. Hardcoding this to 0 hashes into a different
            // 4D history slot than the writes, making history written from
            // in-check cutoffs invisible to in-check reads. SF keeps reads and
            // writes symmetric.
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
                    // be emitted as bestmove — this has cost a real game by
                    // forfeit.
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
            // In flat QS ordering, history also helps decide which captures
            // survive the searched-move budget. Give it 1.25x weight here as
            // the midpoint between trunk and the neutral 1.5x probe (#3379).
            // Main-search capture ordering remains unchanged.
            let hist_score = if self.no_see_partition {
                capt_hist * 5 / 4
            } else {
                capt_hist
            };
            let cap_score = mvv_lva(board, m) + hist_score;
            if self.no_see_partition {
                // SF QCAPTURE shape: no SEE here at all. Order by score; the
                // caller's gate does the one exchange evaluation per move.
                let idx = self.moves.len;
                self.moves.push(m);
                self.scores[idx].write(cap_score);
                continue;
            }
            let see_threshold = -capt_hist / 18;
            if !see_ge(board, m, see_threshold) {
                // Bad capture. The 256 cap must stay generous: a smaller one
                // silently drops moves in pathological tactical positions
                // (multiple queens and rooks with many captures available).
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
        // Indexed by min(pt, 7): pawn/king/NO_PIECE_TYPE get 0.
        let escape_bonus_by_pt: [i32; 8] = [
            0, escape_bonus_minor, escape_bonus_minor, escape_bonus_r, escape_bonus_q, 0, 0, 0,
        ];
        let quiet_check_bonus = crate::search::QUIET_CHECK_BONUS.load(Ordering::Relaxed);
        let quiet_check_see_margin = crate::search::QUIET_CHECK_SEE_MARGIN.load(Ordering::Relaxed);
        let discovered_attack_bonus = crate::search::DISCOVERED_ATTACK_BONUS.load(Ordering::Relaxed);
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
            // square. All four (Q/R/B/N) now tunable.
            if self.threats & (1u64 << from) != 0 && piece != NO_PIECE {
                // Table instead of a match on piece type (jump table, mispredicts).
                score += escape_bonus_by_pt[pt.min(7) as usize];
            }

            // Quiet check bonus: moves that give direct check (SF +16384).
            // SEE-gated like SF: a check that loses material by more
            // than QUIET_CHECK_SEE_MARGIN is a losing sac — don't order it first.
            if piece != NO_PIECE
                && pt < 6 && self.checking_sqs[pt as usize] & (1u64 << to) != 0
                && see_ge(board, m, -quiet_check_see_margin) {
                score += quiet_check_bonus;
            }

            // B1: Discovered-attack bonus. If `from` is one of our pieces
            // currently blocking our slider's attack on an enemy, moving
            // it uncovers that attack. Flat bonus — victim-value scaling
            // is a follow-up if H1 resolves.
            if self.xray_blockers & (1u64 << from) != 0 {
                score += discovered_attack_bonus;
            }


            // "Offense bonus": quiet move that lands on a square
            // attacking an enemy non-pawn piece. +6000 flat. Not yet present
            // in Coda; consensus places it at ~+6000. Signal: does our piece on
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
                            // "Good quiet": the offense move's strongest target
                            // is worth more than the piece making the move — a
                            // cheap proxy for positive quiet-SEE. The 6687 is a
                            // tuned value and ablation-confirmed load-bearing
                            // (~2 Elo), not an arbitrary constant.
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
    fn generate_and_score_evasions(&mut self, board: &Board) {
        let all = generate_evasions(board, self.checkers, self.pinned);
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
            let from = move_from(m);
            let to = move_to(m);
            let flags = move_flags(m);

            // Test capture FIRST so capture-promotions (e.g.
            // pawn-takes-and-promotes) take the capture path rather than the
            // flat promotion score below, which would rank them BELOW ordinary
            // captures.
            let is_cap = board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;
            let score = if is_cap {
                // Capture (possibly also a promotion): MVV-LVA + capture
                // history. mvv_lva folds in the promotion material delta, so
                // capture-promotions rank above ordinary captures.
                //
                // The (1<<20) base makes the capture band uncrossable: quiet
                // history sums span ±80k, so a smaller offset lets a hot quiet
                // outrank a fresh capture of the checker. SF uses 1<<28,
                // Berserk 1e7. mvv+captHist still order within the band.
                (1 << 20) + mvv_lva(board, m) + capt_hist_score_static(board, history, m)
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

    // Add the promotion material delta (promoted piece - pawn) so promotions
    // carry a large base bonus. Without it a non-capture promotion scores 0 in
    // MVV and hits an empty capt_hist slot, ranking BELOW any ordinary capture
    // that happens to have a small history score.
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

    // MVV only (no LVA), multiplier SPSA-tunable via MVV_CAP_MULT (current default 28)
    see_value(target_pt) * mult + promo_bonus
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
    // EP requires that `from` is on the EP-capture
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
    // king/intermediate/destination. The rook-on-corner check defends against
    // TT-collision castles on a board where the rook has moved away but the
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

/// Capture-only picker used by ProbCut: TT move first, then captures scored by
/// the shared main-search capture scorer. Never runs in check — the old evasion
/// mode was dead code and was removed (QS in-check uses MovePicker::new_evasion).
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
}

impl QMovePicker {
    /// Create QS picker: TT move first, then captures scored by MVV-LVA + captHist.
    /// When in_check, generates all moves (evasions); otherwise captures only.
    pub fn new(
        board: &Board,
        tt_move: Move,
        history: &History,
        // Passed in — the probcut block runs after the node-entry
        // pinned/checkers computation.
        pinned: Bitboard,
        checkers: Bitboard,
    ) -> Self {
        // Captures only. This picker's sole caller is ProbCut, which never
        // runs in check — the old in_check/evasion branch here was dead code
        // (QS in-check uses the full MovePicker::new_evasion, history-scored).
        let moves = generate_captures(board);
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
            let target_pt = board.piece_type_at(to);
            let flags = move_flags(mv);

            if target_pt != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT {
                // Capture: route through the SAME scorer main search uses
                // (MVV-only ×MVV_CAP_MULT + captHist, no LVA term). The old QS
                // fork used `victim*10 - attacker` MVV-LVA, left behind when
                // main search dropped LVA (65dac27); captHist's relative weight
                // was ~1.6× higher here. Unified per Stockfish, which
                // use one capture scorer for both search and QS.
                let mvv = mvv_lva(board, mv);
                let capt_hist = capt_hist_score_static(board, history, mv);
                picker.scores[i].write(mvv + capt_hist);
            } else if is_promotion(mv) {
                // Non-capture promotion: kept on its ORIGINAL see_value scale,
                // deliberately. This ranks it ~25x below the main picker's
                // mvv_lva rank for the same move, which looks like forked-
                // scorer drift -- but "fixing" it to the shared scale failed
                // non-regression at -1.8 Elo (#3366, bundled with this
                // deletion). ProbCut trying promotions ahead of captures
                // evidently changes which cutoffs its verification certifies,
                // for the worse. Intentional-by-evidence; do not unify without
                // its own SPRT.
                picker.scores[i].write(see_value(promotion_piece_type(mv)));
            } else {
                // Unreachable for generate_captures output; defensive score.
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

            // Captures-only picker: everything generated is a capture or
            // promotion, so no per-move class filter is needed.

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

#[cfg(test)]
mod pseudo_legal_fuzz {
    use super::*;
    use crate::board::Board;
    use crate::movegen::{generate_captures, generate_legal_moves, generate_quiets};

    /// Differential fuzz: is_pseudo_legal(m) must agree EXACTLY with
    /// membership in the generated pseudo-legal set (captures + quiets),
    /// for every one of the 64*64*7 encodable moves, across random-playout
    /// positions. The two failure directions are both real bugs:
    ///   is_pl=true, not generated  -> TT-collision hole (corrupt move could
    ///                                 reach make_move — the 320-Elo class)
    ///   generated, is_pl=false     -> engine rejects its own legit TT move
    ///                                 (ordering loss, IIR misfire)
    /// Deterministic (fixed-seed xorshift); ~1.4k positions, ~40M probes.
    #[test]
    fn fuzz_pseudo_legal_differential() {
        crate::init();
        let mut st: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let flags_all: [u16; 7] = [0, FLAG_EN_PASSANT, FLAG_CASTLE,
            FLAG_PROMOTE_N, FLAG_PROMOTE_B, FLAG_PROMOTE_R, FLAG_PROMOTE_Q];
        let (mut positions, mut probes): (u64, u64) = (0, 0);
        for _game in 0..100 {
            let mut board = Board::startpos();
            for ply in 0..120 {
                let legal = generate_legal_moves(&board);
                if legal.len == 0 || board.halfmove >= 100 {
                    break;
                }
                let mv = legal.get((rnd() as usize) % legal.len);
                board.make_move(mv);
                if ply % 3 != 0 {
                    continue;
                }
                positions += 1;
                let mut set = std::collections::HashSet::new();
                let caps = generate_captures(&board);
                for i in 0..caps.len {
                    set.insert(caps.get(i));
                }
                let quiets = generate_quiets(&board);
                for i in 0..quiets.len {
                    set.insert(quiets.get(i));
                }
                for from in 0..64u8 {
                    for to in 0..64u8 {
                        if from == to {
                            continue;
                        }
                        for &fl in flags_all.iter() {
                            let m = crate::types::make_move(from, to, fl);
                            let a = is_pseudo_legal(&board, m);
                            let b = set.contains(&m);
                            probes += 1;
                            if a != b {
                                panic!(
                                    "PSEUDO-LEGAL MISMATCH fen='{}' mv={} flags={} is_pseudo_legal={} generated={}",
                                    board.to_fen(), move_to_uci(m), fl, a, b
                                );
                            }
                        }
                    }
                }
            }
        }
        eprintln!("pseudo-legal differential clean: {} positions, {} probes", positions, probes);
    }
}

#[cfg(test)]
mod pseudo_legal_fuzz_debug {
    use crate::board::Board;
    use crate::movegen::generate_legal_moves;
    use crate::types::*;

    #[test]
    fn debug_playout_invariant() {
        crate::init();
        let mut st: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for game in 0..40 {
            let mut board = Board::startpos();
            let mut hist: Vec<String> = Vec::new();
            for _ply in 0..120 {
                let legal = generate_legal_moves(&board);
                if legal.len == 0 || board.halfmove >= 100 {
                    break;
                }
                let mv = legal.get((rnd() as usize) % legal.len);
                let fen_before = board.to_fen();
                let ok = board.make_move(mv);
                hist.push(move_to_uci(mv));
                assert!(ok, "make_move rejected a LEGAL move {} at '{}' (game {})",
                    move_to_uci(mv), fen_before, game);
                // Invariant: the side that just moved must not have left its
                // own king capturable (enemy king attacked while WE are to move).
                let them = flip_color(board.side_to_move); // the side that just moved
                let their_king = board.king_sq(them);
                let occ = board.occupied();
                if board.attackers_to(their_king as u32, occ)
                    & board.colors[board.side_to_move as usize] != 0
                {
                    panic!(
                        "ILLEGAL STATE game {} after {} (from '{}'):\n  now '{}'\n  moves: {}",
                        game, move_to_uci(mv), fen_before, board.to_fen(), hist.join(" ")
                    );
                }
            }
        }
        eprintln!("playout invariant clean");
    }
}

#[cfg(test)]
mod attackers_probe {
    use crate::board::Board;
    use crate::types::*;

    #[test]
    fn probe_pawn_attack_on_rank8() {
        crate::init();
        // White pawn d7, black king e8: e8 MUST be attacked by white.
        let b = Board::from_fen("rnbqkbnr/3P4/P5p1/1P3pP1/4pP1p/2p1P2P/8/RbBQKBNR w KQkq - 0 20");
        let occ = b.occupied();
        let atk = b.attackers_to(60, occ); // e8 = 60
        eprintln!("attackers_to(e8) = {:#018x}, white mask = {:#018x}", atk, b.colors[WHITE as usize]);
        eprintln!("white attackers of e8: {:#018x}", atk & b.colors[WHITE as usize]);
        assert!(atk & b.colors[WHITE as usize] & (1u64 << 51) != 0,
            "d7 white pawn (sq 51) must attack e8");
    }
}

/// Continuation-history read/write SYMMETRY guardrail.
///
/// Every cont-hist read and write picks its sub-table from
/// `moved_piece_stack[ply - off]`, bound-checked against `CONT_PLANES`. If the
/// plane count is ever changed (e.g. adding an in-check or capture dimension)
/// and a bound check is left at the old literal, writes into the new planes are
/// SILENTLY DROPPED — no panic, no test failure, just missing history. These
/// tests make that loud.
#[cfg(test)]
mod cont_hist_symmetry_tests {
    use super::*;
    use crate::board::Board;

    fn init() { crate::init(); }

    /// The declared plane count and the actual array extent must agree.
    #[test]
    fn cont_planes_matches_array_extent() {
        let h = Box::new(History {
            main: [[[[0; 64]; 64]; 2]; 2],
            capture: [[[0; 7]; 64]; 13],
            cont_hist: [[[[0; 64]; 13]; 64]; CONT_PLANES],
        });
        assert_eq!(h.cont_hist.len(), CONT_PLANES,
            "CONT_PLANES ({}) disagrees with cont_hist's first dimension ({})",
            CONT_PLANES, h.cont_hist.len());
    }

    /// EVERY valid plane must survive the round trip: a value written at
    /// cont_hist[plane][to] must be reachable through the sub-table pointer the
    /// MovePicker derives for that same (plane, to). A bound guard left at a
    /// stale literal shows up here as a None sub-table on the high planes.
    #[test]
    fn every_plane_round_trips_through_movepicker() {
        init();
        let board = Board::from_fen("r1bqkb1r/pp3ppp/2n1pn2/2pp4/3P4/2P1PN2/PP1N1PPP/R1BQK2R w KQkq - 0 6");
        let ply = 7usize;
        for plane in 1..CONT_PLANES {
            for &prior_to in &[0usize, 27, 63] {
                let mut h = Box::new(History {
                    main: [[[[0; 64]; 64]; 2]; 2],
                    capture: [[[0; 7]; 64]; 13],
                    cont_hist: [[[[0; 64]; 13]; 64]; CONT_PLANES],
                });
                // Sentinel unique per (plane, prior_to) so a mis-derived pointer
                // reads the wrong number rather than coincidentally matching.
                let sentinel = (plane as i16) * 100 + prior_to as i16 + 1;
                h.cont_hist[plane][prior_to][3][17] = sentinel;

                let mut mps = [0u8; 64];
                let mut mts = [0u8; 64];
                mps[ply - 1] = plane as u8;
                mts[ply - 1] = prior_to as u8;

                let mp = MovePicker::new(
                    &board, NO_MOVE, ply, 0, 0, &h, NO_MOVE, None,
                    Default::default(), 0, &mps, &mts,
                );

                let sub = mp.cont_hist_subs[0].unwrap_or_else(|| panic!(
                    "plane {plane} (prior_to {prior_to}) produced NO cont-hist sub-table — \
                     a bound guard is still using a stale literal instead of CONT_PLANES"));
                let got = unsafe { (*sub)[3][17] };
                assert_eq!(got, sentinel,
                    "plane {plane} (prior_to {prior_to}) round-tripped to the WRONG slot");
            }
        }
    }

    /// The null-move sentinel (piece 0) must never select a plane — otherwise a
    /// null move would share plane 0 with real moves.
    #[test]
    fn null_sentinel_selects_no_plane() {
        init();
        let board = Board::from_fen("r1bqkb1r/pp3ppp/2n1pn2/2pp4/3P4/2P1PN2/PP1N1PPP/R1BQK2R w KQkq - 0 6");
        let h = Box::new(History {
            main: [[[[0; 64]; 64]; 2]; 2],
            capture: [[[0; 7]; 64]; 13],
            cont_hist: [[[[0; 64]; 13]; 64]; CONT_PLANES],
        });
        let mps = [0u8; 64];
        let mts = [0u8; 64];
        let mp = MovePicker::new(
            &board, NO_MOVE, 7, 0, 0, &h, NO_MOVE, None,
            Default::default(), 0, &mps, &mts,
        );
        assert!(mp.cont_hist_subs[0].is_none(), "null sentinel must not select a plane");
    }
}
