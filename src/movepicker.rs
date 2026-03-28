/// Staged move picker for search.
/// Order: TT move -> good captures (MVV-LVA + captHist/16) -> killers -> counter-move -> quiets (history) -> bad captures.
///
/// True staged generation: captures generated first, quiets only when needed.

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
    /// Capture history: [piece][to][captured_pt]
    pub capture: [[[i32; 6]; 64]; 12],
    /// Killer moves: [ply][2]
    pub killers: [[Move; 2]; 128],
    /// Counter-move: [piece][to]
    pub counter: [[Move; 64]; 12],
    /// Continuation history: [piece][to][piece][to]
    pub cont_hist: [[[[i32; 64]; 12]; 64]; 12],
}

impl History {
    pub fn new() -> Self {
        History {
            main: [[0; 64]; 64],
            capture: [[[0; 6]; 64]; 12],
            killers: [[NO_MOVE; 2]; 128],
            counter: [[NO_MOVE; 64]; 12],
            cont_hist: [[[[0; 64]; 12]; 64]; 12],
        }
    }

    pub fn clear(&mut self) {
        self.main = [[0; 64]; 64];
        self.capture = [[[0; 6]; 64]; 12];
        self.killers = [[NO_MOVE; 2]; 128];
        self.counter = [[NO_MOVE; 64]; 12];
        self.cont_hist = [[[[0; 64]; 12]; 64]; 12];
    }

    /// Update history with gravity (bonus capped, decayed toward zero).
    pub fn update_history(entry: &mut i32, bonus: i32) {
        let clamped = bonus.clamp(-MAX_HISTORY, MAX_HISTORY);
        *entry += clamped - *entry * clamped.abs() / MAX_HISTORY;
    }

    /// Get quiet history score for a move (with optional pawn history).
    pub fn quiet_score(&self, board: &Board, mv: Move, prev_move: Move, pawn_hist: Option<&[[i16; 64]; 12]>) -> i32 {
        let from = move_from(mv);
        let to = move_to(mv);
        let color = board.side_to_move;

        let mut score = self.main[from as usize][to as usize];

        // Add continuation history if we have a previous move (3x weight)
        if prev_move != NO_MOVE {
            let prev_to = move_to(prev_move);
            let prev_piece = board.piece_at(prev_to);
            if prev_piece != NO_PIECE && (prev_piece as usize) < 12 {
                let piece = board.piece_at(from);
                if piece != NO_PIECE && (piece as usize) < 12 {
                    score += self.cont_hist[prev_piece as usize][prev_to as usize][piece as usize][to as usize] * 3;
                }
            }
        }

        // Add pawn history
        if let Some(ph) = pawn_hist {
            let piece = board.piece_at(from);
            if piece != NO_PIECE && (piece as usize) < 12 {
                score += ph[piece as usize][to as usize] as i32;
            }
        }

        score
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Stage {
    TTMove,
    GenerateCaptures,
    GoodCaptures,
    Killers,
    CounterMove,
    GenerateQuiets,
    Quiets,
    BadCaptures,
    Done,
}

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    killer1: Move,
    killer2: Move,
    counter_move: Move,
    // Captures (generated first)
    captures: MoveList,
    cap_scores: [i32; 256],
    cap_idx: usize,
    // Quiets (generated later, only if needed)
    quiets: MoveList,
    quiet_scores: [i32; 256],
    quiet_idx: usize,
    // Bad captures deferred to end
    bad_captures: [Move; 32],
    bad_capture_len: usize,
    bad_capture_idx: usize,
    pinned: Bitboard,
    checkers: Bitboard,
}

impl MovePicker {
    pub fn new(
        board: &Board,
        tt_move: Move,
        ply: usize,
        history: &History,
        prev_move: Move,
    ) -> Self {
        let pinned = board.pinned();
        let checkers = board.checkers();

        let killer1 = if ply < 128 { history.killers[ply][0] } else { NO_MOVE };
        let killer2 = if ply < 128 { history.killers[ply][1] } else { NO_MOVE };

        let counter_move = if prev_move != NO_MOVE {
            let prev_to = move_to(prev_move);
            let prev_piece = board.piece_at(prev_to);
            if prev_piece != NO_PIECE && (prev_piece as usize) < 12 {
                history.counter[prev_piece as usize][prev_to as usize]
            } else {
                NO_MOVE
            }
        } else {
            NO_MOVE
        };

        MovePicker {
            stage: Stage::TTMove,
            tt_move,
            killer1,
            killer2,
            counter_move,
            captures: MoveList::new(),
            cap_scores: [0; 256],
            cap_idx: 0,
            quiets: MoveList::new(),
            quiet_scores: [0; 256],
            quiet_idx: 0,
            bad_captures: [NO_MOVE; 32],
            bad_capture_len: 0,
            bad_capture_idx: 0,
            pinned,
            checkers,
        }
    }

    /// Get the next move to try. Returns NO_MOVE when exhausted.
    pub fn next(&mut self, board: &Board, history: &History, prev_move: Move, pawn_hist: Option<&[[i16; 64]; 12]>) -> Move {
        loop {
            match self.stage {
                Stage::TTMove => {
                    self.stage = Stage::GenerateCaptures;
                    if self.tt_move != NO_MOVE {
                        self.tt_move = fixup_move_flags(board, self.tt_move);
                        if is_pseudo_legal(board, self.tt_move)
                            && board.is_legal(self.tt_move, self.pinned, self.checkers)
                        {
                            return self.tt_move;
                        }
                    }
                    self.tt_move = NO_MOVE;
                }
                Stage::GenerateCaptures => {
                    self.captures = generate_captures(board);
                    self.score_captures(board, history);
                    self.cap_idx = 0;
                    self.stage = Stage::GoodCaptures;
                }
                Stage::GoodCaptures => {
                    while self.cap_idx < self.captures.len {
                        let mv = self.pick_best_capture();
                        if mv == self.tt_move { continue; }
                        if !board.is_legal(mv, self.pinned, self.checkers) { continue; }

                        // Promotions always count as good
                        if !is_promotion(mv) && !see_ge(board, mv, 0) {
                            if self.bad_capture_len < 32 {
                                self.bad_captures[self.bad_capture_len] = mv;
                                self.bad_capture_len += 1;
                            }
                            continue;
                        }

                        return mv;
                    }
                    self.stage = Stage::Killers;
                }
                Stage::Killers => {
                    self.stage = Stage::CounterMove;

                    // Try killer 1
                    if self.killer1 != NO_MOVE && self.killer1 != self.tt_move {
                        self.killer1 = fixup_move_flags(board, self.killer1);
                        if is_pseudo_legal(board, self.killer1)
                            && board.is_legal(self.killer1, self.pinned, self.checkers)
                        {
                            let to = move_to(self.killer1);
                            if board.piece_type_at(to) == NO_PIECE_TYPE {
                                return self.killer1;
                            }
                        }
                    }

                    // Try killer 2
                    if self.killer2 != NO_MOVE && self.killer2 != self.tt_move && self.killer2 != self.killer1 {
                        self.killer2 = fixup_move_flags(board, self.killer2);
                        if is_pseudo_legal(board, self.killer2)
                            && board.is_legal(self.killer2, self.pinned, self.checkers)
                        {
                            let to = move_to(self.killer2);
                            if board.piece_type_at(to) == NO_PIECE_TYPE {
                                return self.killer2;
                            }
                        }
                    }
                }
                Stage::CounterMove => {
                    self.stage = Stage::GenerateQuiets;

                    if self.counter_move != NO_MOVE
                        && self.counter_move != self.tt_move
                        && self.counter_move != self.killer1
                        && self.counter_move != self.killer2
                    {
                        self.counter_move = fixup_move_flags(board, self.counter_move);
                        if is_pseudo_legal(board, self.counter_move)
                            && board.is_legal(self.counter_move, self.pinned, self.checkers)
                        {
                            let to = move_to(self.counter_move);
                            if board.piece_type_at(to) == NO_PIECE_TYPE {
                                return self.counter_move;
                            }
                        }
                    }
                }
                Stage::GenerateQuiets => {
                    self.quiets = generate_quiets(board);
                    self.score_quiets(board, history, prev_move, pawn_hist);
                    self.quiet_idx = 0;
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    while self.quiet_idx < self.quiets.len {
                        let mv = self.pick_best_quiet();
                        if mv == self.tt_move { continue; }
                        if mv == self.killer1 || mv == self.killer2 { continue; }
                        if mv == self.counter_move { continue; }

                        if !board.is_legal(mv, self.pinned, self.checkers) { continue; }

                        return mv;
                    }
                    self.stage = Stage::BadCaptures;
                    self.bad_capture_idx = 0;
                }
                Stage::BadCaptures => {
                    if self.bad_capture_idx < self.bad_capture_len {
                        let mv = self.bad_captures[self.bad_capture_idx];
                        self.bad_capture_idx += 1;
                        return mv;
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => {
                    return NO_MOVE;
                }
            }
        }
    }

    /// Score capture/promotion moves.
    fn score_captures(&mut self, board: &Board, history: &History) {
        for i in 0..self.captures.len {
            let mv = self.captures.moves[i];
            let from = move_from(mv);
            let to = move_to(mv);
            let flags = move_flags(mv);

            let target_pt = board.piece_type_at(to);
            let is_cap = target_pt != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT;

            if is_cap {
                let victim = if flags == FLAG_EN_PASSANT { PAWN } else { target_pt };
                let attacker = board.piece_type_at(from);
                let mvv_lva = see_value(victim) * 10 - see_value(attacker);

                let piece = make_piece(board.side_to_move, attacker);
                let cap_hist = if (piece as usize) < 12 && (victim as usize) < 6 {
                    history.capture[piece as usize][to as usize][victim as usize] / 16
                } else {
                    0
                };

                self.cap_scores[i] = 10_000_000 + mvv_lva + cap_hist;
            } else if is_promotion(mv) {
                self.cap_scores[i] = 9_000_000 + see_value(promotion_piece_type(mv));
            }
        }
    }

    /// Score quiet moves.
    fn score_quiets(&mut self, board: &Board, history: &History, prev_move: Move, pawn_hist: Option<&[[i16; 64]; 12]>) {
        for i in 0..self.quiets.len {
            let mv = self.quiets.moves[i];
            self.quiet_scores[i] = history.quiet_score(board, mv, prev_move, pawn_hist);
        }
    }

    /// Selection sort for captures.
    fn pick_best_capture(&mut self) -> Move {
        let mut best_idx = self.cap_idx;
        let mut best_score = self.cap_scores[self.cap_idx];

        for j in (self.cap_idx + 1)..self.captures.len {
            if self.cap_scores[j] > best_score {
                best_score = self.cap_scores[j];
                best_idx = j;
            }
        }

        if best_idx != self.cap_idx {
            self.captures.moves.swap(self.cap_idx, best_idx);
            self.cap_scores.swap(self.cap_idx, best_idx);
        }

        let mv = self.captures.moves[self.cap_idx];
        self.cap_idx += 1;
        mv
    }

    /// Selection sort for quiets.
    fn pick_best_quiet(&mut self) -> Move {
        let mut best_idx = self.quiet_idx;
        let mut best_score = self.quiet_scores[self.quiet_idx];

        for j in (self.quiet_idx + 1)..self.quiets.len {
            if self.quiet_scores[j] > best_score {
                best_score = self.quiet_scores[j];
                best_idx = j;
            }
        }

        if best_idx != self.quiet_idx {
            self.quiets.moves.swap(self.quiet_idx, best_idx);
            self.quiet_scores.swap(self.quiet_idx, best_idx);
        }

        let mv = self.quiets.moves[self.quiet_idx];
        self.quiet_idx += 1;
        mv
    }
}

/// Check if a move (by from/to) exists in a generated move list.
fn move_in_list(list: &MoveList, mv: Move) -> bool {
    let from = move_from(mv);
    let to = move_to(mv);
    for i in 0..list.len {
        let m = list.moves[i];
        if move_from(m) == from && move_to(m) == to {
            return true;
        }
    }
    false
}

/// Re-derive move flags from the board state. TT/killer moves may have stale flags.
fn fixup_move_flags(board: &Board, mv: Move) -> Move {
    let from = move_from(mv);
    let to = move_to(mv);
    let _flags = move_flags(mv);

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

    // Re-derive double push: pawn moving 2 ranks
    if pt == PAWN {
        let diff = (to as i32 - from as i32).abs();
        if diff == 16 {
            return make_move(from, to, FLAG_DOUBLE_PUSH);
        }
    }

    // Normal move
    make_move(from, to, FLAG_NONE)
}

/// Thorough pseudo-legality check for TT/killer/counter moves.
/// Must validate all special flags to prevent board corruption.
fn is_pseudo_legal(board: &Board, mv: Move) -> bool {
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

    // Castling: validate rights and path
    if flags == FLAG_CASTLE {
        if pt != KING { return false; }
        let occ = board.occupied();
        if us == WHITE {
            if to == 6 { // kingside
                if board.castling & CASTLE_WK == 0 { return false; }
                if occ & 0x60 != 0 { return false; }
            } else if to == 2 { // queenside
                if board.castling & CASTLE_WQ == 0 { return false; }
                if occ & 0x0E != 0 { return false; }
            } else { return false; }
        } else {
            if to == 62 { // kingside
                if board.castling & CASTLE_BK == 0 { return false; }
                if occ & (0x60u64 << 56) != 0 { return false; }
            } else if to == 58 { // queenside
                if board.castling & CASTLE_BQ == 0 { return false; }
                if occ & (0x0Eu64 << 56) != 0 { return false; }
            } else { return false; }
        }
        return true;
    }

    // Double push: must be a pawn
    if flags == FLAG_DOUBLE_PUSH {
        if pt != PAWN { return false; }
    }

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
            // Pawn moves: push or capture
            let diff = (to as i32 - from as i32).abs();
            if diff != 7 && diff != 8 && diff != 9 && diff != 16 {
                return false;
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
            // King already handled above (castle check)
            if king_attacks(from as u32) & to_bb == 0 {
                return false;
            }
        }
        _ => return false,
    }

    true
}

/// Simple move picker for quiescence search (captures only + check evasions).
pub struct QMovePicker {
    moves: MoveList,
    scores: [i32; 256],
    idx: usize,
    pinned: Bitboard,
    checkers: Bitboard,
}

impl QMovePicker {
    pub fn new(board: &Board, in_check: bool) -> Self {
        let pinned = board.pinned();
        let checkers = board.checkers();
        // In check: generate all moves (need evasions). Otherwise: captures only.
        let moves = if in_check { generate_all_moves(board) } else { generate_captures(board) };
        let mut picker = QMovePicker {
            moves,
            scores: [0; 256],
            idx: 0,
            pinned,
            checkers,
        };

        // Score by MVV-LVA
        for i in 0..picker.moves.len {
            let mv = picker.moves.moves[i];
            let to = move_to(mv);
            let target_pt = board.piece_type_at(to);
            let flags = move_flags(mv);

            if target_pt != NO_PIECE_TYPE || flags == FLAG_EN_PASSANT {
                let victim = if flags == FLAG_EN_PASSANT { PAWN } else { target_pt };
                let attacker = board.piece_type_at(move_from(mv));
                picker.scores[i] = see_value(victim) * 10 - see_value(attacker);
            } else if is_promotion(mv) {
                picker.scores[i] = see_value(promotion_piece_type(mv));
            } else {
                picker.scores[i] = -1_000_000;
            }
        }

        picker
    }

    /// Get next capture move. Returns NO_MOVE when exhausted.
    pub fn next(&mut self, board: &Board, in_check: bool) -> Move {
        while self.idx < self.moves.len {
            // Selection sort
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

            // In check: try all moves (evasions)
            if !in_check {
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
