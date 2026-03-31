/// Staged move picker for search.
/// Literal translation of GoChess movepicker.go.
///
/// Order: TT move -> good captures (MVV-LVA + captHist/16) -> killer1 -> killer2 ->
///        counter-move -> quiets (history) -> bad captures.
///
/// Evasion order: TT move -> evasions (captures scored above quiets).
///
/// No legality checks — returns pseudo-legal moves; search caller does legality.

use crate::attacks::*;
use crate::bitboard::*;
use crate::board::Board;
use crate::eval::see_value;
use crate::movegen::{generate_all_moves, generate_captures, generate_quiets, MoveList};
use crate::see::see_ge;
use crate::types::*;

const MAX_HISTORY: i32 = 16384;

/// History tables shared across the search.
pub struct History {
    /// Main history: [from][to] (shared between colors, matches GoChess)
    pub main: [[i32; 64]; 64],
    /// Capture history: [piece 1-12][to][captured_type 0-6]
    /// piece uses GoChess 1-12 indexing (slot 0 unused).
    /// captured_type uses GoChess 0-6 scheme (0=empty, 1=pawn, ..., 6=king).
    /// int16 matches GoChess (was i32 — caused different gravity behavior).
    pub capture: [[[i16; 7]; 64]; 13],
    /// Killer moves: [ply][2]
    pub killers: [[Move; 2]; 64],
    /// Counter-move: [piece 1-12][to]
    /// piece uses GoChess 1-12 indexing (slot 0 unused).
    pub counter: [[Move; 64]; 13],
    /// Continuation history: [piece 1-12][to][piece 1-12][to]
    /// piece uses GoChess 1-12 indexing (slot 0 unused).
    pub cont_hist: [[[[i16; 64]; 13]; 64]; 13],
}

impl History {
    pub fn new() -> Self {
        History {
            main: [[0; 64]; 64],
            capture: [[[0i16; 7]; 64]; 13],
            killers: [[NO_MOVE; 2]; 64],
            counter: [[NO_MOVE; 64]; 13],
            cont_hist: [[[[0; 64]; 13]; 64]; 13],
        }
    }

    pub fn clear(&mut self) {
        self.main = [[0; 64]; 64];
        self.capture = [[[0i16; 7]; 64]; 13];
        self.killers = [[NO_MOVE; 2]; 64];
        self.counter = [[NO_MOVE; 64]; 13];
        self.cont_hist = [[[[0; 64]; 13]; 64]; 13];
    }

    /// Age all history tables by multiplying by factor/divisor (e.g. 4/5 = 0.80).
    /// Preserves useful information from prior searches while letting new data dominate.
    /// Killers and counter-moves are cleared (they're position-specific, not transferable).
    pub fn age(&mut self, factor: i32, divisor: i32) {
        for row in self.main.iter_mut() {
            for v in row.iter_mut() { *v = *v * factor / divisor; }
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
        self.killers = [[NO_MOVE; 2]; 64];
        self.counter = [[NO_MOVE; 64]; 13];
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
}

/// Map a Coda piece (0-11, color*6+pt) to GoChess piece index (1-12).
/// GoChess: White 1-6 (Pawn..King), Black 7-12 (Pawn..King).
/// Coda: White 0-5, Black 6-11.
/// Mapping: coda_piece + 1.
#[inline(always)]
pub fn go_piece(p: Piece) -> usize {
    debug_assert!(p < 12, "go_piece called with NO_PIECE");
    (p + 1) as usize
}

/// Map a Coda piece type (0-5: PAWN..KING) to GoChess captured type (1-6).
/// GoChess capturedType: 0=empty, 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king.
/// Coda PieceType: 0=PAWN, 1=KNIGHT, 2=BISHOP, 3=ROOK, 4=QUEEN, 5=KING.
/// Mapping: pt + 1.
#[inline(always)]
pub fn captured_type(pt: PieceType) -> usize {
    debug_assert!(pt <= 5, "captured_type called with NO_PIECE_TYPE");
    (pt + 1) as usize
}

/// MovePicker stages (matches GoChess exactly).
#[derive(PartialEq, Eq, Clone, Copy)]
enum Stage {
    TTMove,
    GenerateCaptures,
    GoodCaptures,
    Killer1,
    Killer2,
    CounterMove,
    GenerateQuiets,
    Quiets,
    BadCaptures,
    Done,

    // Evasion stages (used when in check)
    EvasionTTMove,
    GenerateEvasions,
    Evasions,
}

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    counter_move: Move,
    // Pointer to the History struct (lives for the duration of search)
    history: *const History,
    // Continuation history sub-table pointers at plies 1, 2, 4, 6 back.
    // cont_hist_subs[0] = ply-1 (3x weight), [1] = ply-2 (3x), [2] = ply-4 (1x), [3] = ply-6 (1x)
    cont_hist_subs: [Option<*const [[i16; 64]; 13]>; 4],
    pawn_hist_ptr: Option<*const [[i16; 64]; 13]>,
    // Main moves list and scores
    moves: MoveList,
    scores: [i32; 256],
    index: usize,
    // Bad captures saved from partition
    bad_moves: [Move; 64],
    bad_scores: [i32; 64],
    bad_len: usize,
    // Ply for killer indexing
    #[allow(dead_code)]
    ply: usize,
    skip_quiet: bool,
    // Evasion support
    checkers: Bitboard,
    pinned: Bitboard,
    // NMP threat square (-1 = none)
    pub threat_sq: i32,
}

impl MovePicker {
    /// Create a new MovePicker for main search (non-evasion).
    /// Matches GoChess Init().
    pub fn new(
        board: &Board,
        tt_move: Move,
        ply: usize,
        history: &History,
        prev_move: Move,
        pawn_hist: Option<&[[i16; 64]; 13]>,
    ) -> Self {
        let killers = if ply < 64 {
            history.killers[ply]
        } else {
            [NO_MOVE; 2]
        };

        let counter_move = if prev_move != NO_MOVE {
            let prev_to = move_to(prev_move);
            let prev_piece = board.piece_at(prev_to);
            if prev_piece != NO_PIECE {
                history.counter[go_piece(prev_piece)][prev_to as usize]
            } else {
                NO_MOVE
            }
        } else {
            NO_MOVE
        };

        // Get continuation history sub-table pointers at plies 1, 2, 4, 6 back.
        // Each uses the piece that moved to the destination square N plies ago.
        let mut cont_hist_subs: [Option<*const [[i16; 64]; 13]>; 4] = [None; 4];
        let stack_len = board.undo_stack.len();
        let offsets = [1usize, 2, 4, 6]; // ply-1, ply-2, ply-4, ply-6
        for (i, &off) in offsets.iter().enumerate() {
            if stack_len >= off {
                let undo = &board.undo_stack[stack_len - off];
                if undo.mv != NO_MOVE {
                    let to = move_to(undo.mv);
                    let piece = board.piece_at(to);
                    if piece != NO_PIECE {
                        cont_hist_subs[i] = Some(&history.cont_hist[go_piece(piece)][to as usize] as *const [[i16; 64]; 13]);
                    }
                }
            }
        }

        let pawn_hist_ptr = pawn_hist.map(|ph| ph as *const [[i16; 64]; 13]);

        MovePicker {
            stage: Stage::TTMove,
            tt_move,
            killers,
            counter_move,
            history: history as *const History,
            cont_hist_subs,
            pawn_hist_ptr,
            moves: MoveList::new(),
            scores: [0; 256],
            index: 0,
            bad_moves: [NO_MOVE; 64],
            bad_scores: [0; 64],
            bad_len: 0,
            ply,
            skip_quiet: false,
            checkers: 0,
            pinned: 0,
            threat_sq: -1,
        }
    }

    /// Create a MovePicker for quiescence search (captures only).
    /// Matches GoChess InitQuiescence().
    pub fn new_quiescence(
        _board: &Board,
        tt_move: Move,
        history: &History,
    ) -> Self {
        MovePicker {
            stage: Stage::TTMove,
            tt_move,
            killers: [NO_MOVE; 2],
            counter_move: NO_MOVE,
            history: history as *const History,
            cont_hist_subs: [None; 4],
            pawn_hist_ptr: None,
            moves: MoveList::new(),
            scores: [0; 256],
            index: 0,
            bad_moves: [NO_MOVE; 64],
            bad_scores: [0; 64],
            bad_len: 0,
            ply: 0,
            skip_quiet: true,
            checkers: 0,
            pinned: 0,
            threat_sq: -1,
        }
    }

    /// Create a MovePicker for evasion mode (when in check).
    /// Matches GoChess InitEvasion().
    /// Evasion moves are generated as all moves then filtered for legality during generation.
    pub fn new_evasion(
        board: &Board,
        tt_move: Move,
        ply: usize,
        checkers: Bitboard,
        pinned: Bitboard,
        history: &History,
        prev_move: Move,
        pawn_hist: Option<&[[i16; 64]; 13]>,
    ) -> Self {
        // Build cont-hist pointers for evasion (same as main picker)
        let mut cont_hist_subs: [Option<*const [[i16; 64]; 13]>; 4] = [None; 4];
        let stack_len = board.undo_stack.len();
        let offsets = [1usize, 2, 4, 6];
        for (i, &off) in offsets.iter().enumerate() {
            if stack_len >= off {
                let undo = &board.undo_stack[stack_len - off];
                if undo.mv != NO_MOVE {
                    let to = move_to(undo.mv);
                    let piece = board.piece_at(to);
                    if piece != NO_PIECE {
                        cont_hist_subs[i] = Some(&history.cont_hist[go_piece(piece)][to as usize] as *const [[i16; 64]; 13]);
                    }
                }
            }
        }

        let pawn_hist_ptr = pawn_hist.map(|ph| ph as *const [[i16; 64]; 13]);

        MovePicker {
            stage: Stage::EvasionTTMove,
            tt_move,
            killers: [NO_MOVE; 2],
            counter_move: NO_MOVE,
            history: history as *const History,
            cont_hist_subs,
            pawn_hist_ptr,
            moves: MoveList::new(),
            scores: [0; 256],
            index: 0,
            bad_moves: [NO_MOVE; 64],
            bad_scores: [0; 64],
            bad_len: 0,
            ply,
            skip_quiet: false,
            checkers,
            pinned,
            threat_sq: -1,
        }
    }

    /// Get the next move to try. Returns NO_MOVE when exhausted.
    /// No legality checks — caller must check legality.
    /// Matches GoChess Next().
    pub fn next(&mut self, board: &Board) -> Move {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateCaptures;
                    if self.tt_move != NO_MOVE && is_pseudo_legal(board, self.tt_move) {
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
                        self.stage = Stage::Killer1;
                    }
                }

                Stage::Killer1 => {
                    self.stage = Stage::Killer2;
                    if self.killers[0] != NO_MOVE && self.killers[0] != self.tt_move {
                        let k = fixup_move_flags(board, self.killers[0]);
                        self.killers[0] = k;
                        if is_pseudo_legal(board, k) && !is_capture(board, k) {
                            return k;
                        }
                    }
                }

                Stage::Killer2 => {
                    self.stage = Stage::CounterMove;
                    if self.killers[1] != NO_MOVE
                        && self.killers[1] != self.tt_move
                        && self.killers[1] != self.killers[0]
                    {
                        let k = fixup_move_flags(board, self.killers[1]);
                        self.killers[1] = k;
                        if is_pseudo_legal(board, k) && !is_capture(board, k) {
                            return k;
                        }
                    }
                }

                Stage::CounterMove => {
                    self.stage = Stage::GenerateQuiets;
                    if self.counter_move != NO_MOVE
                        && self.counter_move != self.tt_move
                        && self.counter_move != self.killers[0]
                        && self.counter_move != self.killers[1]
                    {
                        let cm = fixup_move_flags(board, self.counter_move);
                        self.counter_move = cm;
                        if is_pseudo_legal(board, cm) && !is_capture(board, cm) {
                            return cm;
                        }
                    }
                }

                Stage::GenerateQuiets => {
                    self.generate_and_score_quiets(board);
                    self.stage = Stage::Quiets;
                }

                Stage::Quiets => {
                    // TT/killers/counter already filtered during scoring
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
                    if self.tt_move != NO_MOVE && is_pseudo_legal(board, self.tt_move) {
                        if board.is_legal(self.tt_move, self.pinned, self.checkers) {
                            return self.tt_move;
                        }
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
    /// TT move is filtered out. Matches GoChess generateAndScoreCaptures().
    fn generate_and_score_captures(&mut self, board: &Board) {
        let caps = generate_captures(board);
        self.moves = MoveList::new();
        self.bad_len = 0;

        let history = unsafe { &*self.history };

        for i in 0..caps.len {
            let m = caps.moves[i];
            if m == self.tt_move {
                continue;
            }
            if !see_ge(board, m, 0) {
                // Bad capture
                if self.bad_len < 64 {
                    self.bad_moves[self.bad_len] = m;
                    self.bad_scores[self.bad_len] = capt_hist_score_static(board, history, m);
                    self.bad_len += 1;
                }
            } else {
                // Good capture
                let idx = self.moves.len;
                self.moves.push(m);
                self.scores[idx] = mvv_lva(board, m) + capt_hist_score_static(board, history, m);
            }
        }
        self.index = 0;
    }

    /// Generate quiet moves and score by history.
    /// TT, killers, counter filtered out. Matches GoChess generateAndScoreQuiets().
    fn generate_and_score_quiets(&mut self, board: &Board) {
        let quiets = generate_quiets(board);
        self.moves = MoveList::new();

        let history = unsafe { &*self.history };

        for i in 0..quiets.len {
            let m = quiets.moves[i];
            if m == self.tt_move
                || m == self.killers[0]
                || m == self.killers[1]
                || m == self.counter_move
            {
                continue;
            }

            let from = move_from(m);
            let to = move_to(m);
            let piece = board.piece_at(from);

            let mut score = history.main[from as usize][to as usize];

            // Continuation history: plies 1,2 at 3x weight, plies 4,6 at 1x weight.
            // Matches Obsidian/Alexandria/Berserk pattern.
            if piece != NO_PIECE {
                let gp = go_piece(piece);
                let weights = [3i32, 3, 1, 1]; // ply-1, ply-2, ply-4, ply-6
                for (i, &w) in weights.iter().enumerate() {
                    if let Some(sub_ptr) = self.cont_hist_subs[i] {
                        let sub = unsafe { &*sub_ptr };
                        score += w * sub[gp][to as usize] as i32;
                    }
                }
            }

            // Pawn history
            if let Some(ph_ptr) = self.pawn_hist_ptr {
                if piece != NO_PIECE {
                    let ph = unsafe { &*ph_ptr };
                    score += ph[go_piece(piece)][to as usize] as i32;
                }
            }

            // Null-move threat: bonus for escaping the threatened square
            if self.threat_sq >= 0 && from as i32 == self.threat_sq {
                score += 8000;
            }

            let idx = self.moves.len;
            self.moves.push(m);
            self.scores[idx] = score;
        }
        self.index = 0;
    }

    /// Generate evasion moves and score them.
    /// Captures scored above quiets. TT move filtered out.
    /// Matches GoChess generateAndScoreEvasions().
    ///
    /// Since Coda doesn't have a dedicated generate_evasions function,
    /// we use generate_all_moves and filter for legality here.
    fn generate_and_score_evasions(&mut self, board: &Board) {
        let all = generate_all_moves(board);
        self.moves = MoveList::new();

        let history = unsafe { &*self.history };

        for i in 0..all.len {
            let m = all.moves[i];
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

            let score = if is_promotion(m) {
                if flags == FLAG_PROMOTE_Q {
                    9000
                } else {
                    -1000 // underpromotions
                }
            } else if board.piece_type_at(to) != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT {
                // Capture: MVV-LVA + capture history
                10000 + mvv_lva(board, m) + capt_hist_score_static(board, history, m)
            } else {
                // Quiet: history + continuation history + pawn history
                let piece = board.piece_at(from);

                let mut s = history.main[from as usize][to as usize];

                if piece != NO_PIECE {
                    let gp = go_piece(piece);
                    let weights = [3i32, 3, 1, 1];
                    for (i, &w) in weights.iter().enumerate() {
                        if let Some(sub_ptr) = self.cont_hist_subs[i] {
                            let sub = unsafe { &*sub_ptr };
                            s += w * sub[gp][to as usize] as i32;
                        }
                    }
                }

                if let Some(ph_ptr) = self.pawn_hist_ptr {
                    if piece != NO_PIECE {
                        let ph = unsafe { &*ph_ptr };
                        s += ph[go_piece(piece)][to as usize] as i32;
                    }
                }

                s
            };

            let idx = self.moves.len;
            self.moves.push(m);
            self.scores[idx] = score;
        }
        self.index = 0;
    }

    /// Swap in the saved bad captures. Matches GoChess restoreBadCaptures().
    fn restore_bad_captures(&mut self) {
        self.moves = MoveList::new();
        for i in 0..self.bad_len {
            self.moves.push(self.bad_moves[i]);
            self.scores[i] = self.bad_scores[i];
        }
        self.index = 0;
    }

    /// Selection sort: find best from current index, swap to front, return it.
    /// Matches GoChess pickBest().
    fn pick_best(&mut self) -> Move {
        if self.index >= self.moves.len {
            return NO_MOVE;
        }

        let mut best_idx = self.index;
        let mut best_score = self.scores[self.index];

        for i in (self.index + 1)..self.moves.len {
            if self.scores[i] > best_score {
                best_score = self.scores[i];
                best_idx = i;
            }
        }

        if best_idx != self.index {
            self.moves.moves.swap(self.index, best_idx);
            self.scores.swap(self.index, best_idx);
        }

        let mv = self.moves.moves[self.index];
        self.index += 1;
        mv
    }

}

/// Capture history score for a capture move.
/// Matches GoChess captHistScore(). Public for use by QMovePicker.
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

/// MVV-LVA score for a capture. Matches GoChess mvvLva().
fn mvv_lva(board: &Board, m: Move) -> i32 {
    let to = move_to(m);
    let from = move_from(m);

    let target_pt = board.piece_type_at(to);
    if target_pt == NO_PIECE_TYPE {
        // En passant
        if move_flags(m) == FLAG_EN_PASSANT {
            return see_value(PAWN) * 16;
        }
        return 0;
    }

    let attacker_pt = board.piece_type_at(from);

    // MVV only (no LVA), x16 — matches Obsidian/Alexandria/Berserk
    see_value(target_pt) * 16
}

/// Check if a move is a capture. Matches GoChess isCapture().
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

    // Re-derive EP: must be pawn moving to ep_square diagonally
    if pt == PAWN && to == board.ep_square && board.ep_square != NO_SQUARE {
        let diff = (to as i32 - from as i32).abs();
        if diff == 7 || diff == 9 {
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

/// Thorough pseudo-legality check for TT/killer/counter moves.
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
    if flags == FLAG_EN_PASSANT {
        if pt != PAWN { return false; }
        if to != board.ep_square { return false; }
        // Verify capture square has enemy pawn
        let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
        if cap_sq >= 64 || (1u64 << cap_sq) & board.pieces[PAWN as usize] & board.colors[them as usize] == 0 {
            return false;
        }
        return true;
    }

    // Castling: validate rights, path, and no attacks on king/intermediate/destination
    if flags == FLAG_CASTLE {
        if pt != KING { return false; }
        let occ = board.occupied();
        let them_bb = board.colors[flip_color(us) as usize];
        if us == WHITE {
            if from != 4 { return false; } // king must be on e1
            if to == 6 { // kingside
                if board.castling & CASTLE_WK == 0 { return false; }
                if occ & 0x60 != 0 { return false; }
                // King(e1), f1, g1 must not be attacked
                if board.attackers_to(4, occ) & them_bb != 0 { return false; }
                if board.attackers_to(5, occ) & them_bb != 0 { return false; }
                if board.attackers_to(6, occ) & them_bb != 0 { return false; }
            } else if to == 2 { // queenside
                if board.castling & CASTLE_WQ == 0 { return false; }
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
                if occ & (0x60u64 << 56) != 0 { return false; }
                // King(e8), f8, g8 must not be attacked
                if board.attackers_to(60, occ) & them_bb != 0 { return false; }
                if board.attackers_to(61, occ) & them_bb != 0 { return false; }
                if board.attackers_to(62, occ) & them_bb != 0 { return false; }
            } else if to == 58 { // queenside
                if board.castling & CASTLE_BQ == 0 { return false; }
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

    // Promotion: must be a pawn on the 7th rank
    if is_promotion(mv) {
        if pt != PAWN { return false; }
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
            if diff == 8 {
                if board.piece_type_at(to) != NO_PIECE_TYPE { return false; }
            }
            // Capture: destination must have enemy piece (or be EP square)
            if diff == 7 || diff == 9 {
                if board.piece_type_at(to) == NO_PIECE_TYPE && to != board.ep_square {
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
/// Matches GoChess: tries TT move first, scores captures with MVV-LVA + captHist/16.
/// When in_check, uses evasion mode (all moves, captures scored above quiets).
pub struct QMovePicker {
    tt_move: Move,
    tt_stage: bool, // true = haven't tried TT move yet
    moves: MoveList,
    scores: [i32; 256],
    idx: usize,
    pinned: Bitboard,
    checkers: Bitboard,
    in_check: bool,
}

impl QMovePicker {
    /// Create QS picker matching GoChess InitQuiescence: TT move + captHist scoring.
    /// When in_check, generates all moves (evasions); otherwise captures only.
    pub fn new(board: &Board, tt_move: Move, in_check: bool, history: &History) -> Self {
        let pinned = board.pinned();
        let checkers = board.checkers();

        let moves = if in_check { generate_all_moves(board) } else { generate_captures(board) };
        let mut picker = QMovePicker {
            tt_move: if tt_move != NO_MOVE && is_pseudo_legal(board, tt_move) { tt_move } else { NO_MOVE },
            tt_stage: true,
            moves,
            scores: [0; 256],
            idx: 0,
            pinned,
            checkers,
            in_check,
        };

        // Score moves: MVV-LVA + captHist/16 for captures (matching GoChess)
        for i in 0..picker.moves.len {
            let mv = picker.moves.moves[i];
            // Skip TT move in scoring (will be tried first)
            if mv == picker.tt_move {
                picker.scores[i] = i32::MIN;
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
                    // Evasion captures scored high (matching GoChess: 10000 + mvvlva + captHist/16)
                    picker.scores[i] = 10000 + mvv_lva + capt_hist;
                } else {
                    picker.scores[i] = mvv_lva + capt_hist;
                }
            } else if is_promotion(mv) {
                if in_check {
                    let pt = promotion_piece_type(mv);
                    picker.scores[i] = if pt == QUEEN { 9000 } else { -1000 };
                } else {
                    picker.scores[i] = see_value(promotion_piece_type(mv));
                }
            } else {
                // Quiet (only in evasion mode)
                picker.scores[i] = -1_000_000;
            }
        }

        picker
    }

    /// Get next move. Returns NO_MOVE when exhausted.
    pub fn next(&mut self, board: &Board) -> Move {
        // Try TT move first (matching GoChess stageTTMove in QS)
        if self.tt_stage {
            self.tt_stage = false;
            if self.tt_move != NO_MOVE {
                if board.is_legal(self.tt_move, self.pinned, self.checkers) {
                    return self.tt_move;
                }
            }
        }

        while self.idx < self.moves.len {
            // Selection sort: find best remaining
            let mut best_idx = self.idx;
            let mut best_score = self.scores[self.idx];
            for j in (self.idx + 1)..self.moves.len {
                if self.scores[j] > best_score {
                    best_score = self.scores[j];
                    best_idx = j;
                }
            }
            self.moves.moves.swap(self.idx, best_idx);
            self.scores.swap(self.idx, best_idx);

            let mv = self.moves.moves[self.idx];
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
