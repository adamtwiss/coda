/// Board representation using bitboards + mailbox.
///
/// Bitboards: pieces[6] (by type) + colors[2] (by color).
/// Mailbox: mailbox[64] for O(1) piece-at-square lookup.

use crate::bitboard::*;
use crate::attacks::*;
use crate::types::*;
use crate::zobrist::*;

/// Undo information stored before each move for unmaking.
#[derive(Clone, Copy)]
pub struct UndoInfo {
    pub mv: Move,
    pub captured: u8,       // PieceType of captured piece (NO_PIECE_TYPE if none)
    pub castling: u8,
    pub ep_square: u8,
    pub halfmove: u16,
    pub plies_from_null: u16,
    pub hash: u64,
    pub pawn_hash: u64,
    pub non_pawn_key: [u64; 2],
    pub minor_key: [u64; 2],
    pub major_key: [u64; 2],
    pub checkers: Bitboard,
}

/// The board state.
#[derive(Clone)]
pub struct Board {
    /// Bitboards by piece type (PAWN..KING)
    pub pieces: [Bitboard; 6],
    /// Bitboards by color (WHITE, BLACK)
    pub colors: [Bitboard; 2],
    /// Mailbox: piece type at each square (NO_PIECE_TYPE if empty). O(1) lookup.
    pub mailbox: [u8; 64],

    pub side_to_move: Color,
    pub castling: u8,        // 4-bit castling rights
    pub ep_square: u8,       // NO_SQUARE if none
    pub halfmove: u16,       // halfmove clock for 50-move rule
    pub fullmove: u16,       // fullmove number
    pub plies_from_null: u16, // plies since last null move (for cuckoo cycle detection)

    pub hash: u64,           // Zobrist hash
    pub pawn_hash: u64,      // Zobrist hash of pawns only (for correction history)
    pub non_pawn_key: [u64; 2], // per-color non-pawn, non-king piece Zobrist keys
    pub minor_key: [u64; 2],   // per-color knight+bishop Zobrist keys
    pub major_key: [u64; 2],   // per-color rook+queen Zobrist keys
    pub undo_stack: Vec<UndoInfo>,
    /// Threat deltas accumulated during make_move (cleared on each make_move).
    /// Used by the NNUE threat accumulator for incremental updates.
    pub threat_deltas: Vec<crate::threats::RawThreatDelta>,
    /// Whether to generate threat deltas during make_move (set when threat net is loaded).
    pub generate_threat_deltas: bool,
}

/// Castling rook positions (from, to) indexed by castling flag bit.
pub const CASTLE_ROOK_FROM: [u8; 4] = [7, 0, 63, 56];   // WK, WQ, BK, BQ
pub const CASTLE_ROOK_TO: [u8; 4]   = [5, 3, 61, 59];

/// Castling rights mask per square — AND with this on any move from/to that square.
static mut CASTLE_MASK: [u8; 64] = [0xFF; 64];

pub fn init_castle_masks() {
    unsafe {
        // Reset all to 0xFF (no change)
        CASTLE_MASK = [0xFF; 64];
        // King squares
        CASTLE_MASK[4] = !(CASTLE_WK | CASTLE_WQ);   // e1: white loses both
        CASTLE_MASK[60] = !(CASTLE_BK | CASTLE_BQ);   // e8: black loses both
        // Rook squares
        CASTLE_MASK[7] = !CASTLE_WK;    // h1: white loses kingside
        CASTLE_MASK[0] = !CASTLE_WQ;    // a1: white loses queenside
        CASTLE_MASK[63] = !CASTLE_BK;   // h8: black loses kingside
        CASTLE_MASK[56] = !CASTLE_BQ;   // a8: black loses queenside
    }
}

fn castle_mask(sq: u8) -> u8 {
    unsafe { CASTLE_MASK[sq as usize] }
}

/// Returns true if `ep_sq` is an EP target that `capturing_side` can actually
/// reach with a pawn — i.e., `capturing_side` has a pawn on a file-adjacent
/// square on the same rank as the just-double-pushed pawn.
///
/// The double-pushed pawn's square is implied by `ep_sq`:
///   - if capturing_side == BLACK, white just pushed; pushed pawn = ep_sq + 8
///   - if capturing_side == WHITE, black just pushed; pushed pawn = ep_sq - 8
///
/// The Zobrist hash must XOR `ep_key(file_of(ep_sq))` iff this returns true.
/// Otherwise two physically identical positions — one reached via a double
/// push with no adjacent enemy pawn, one reached any other way — would hash
/// differently, silently breaking threefold-repetition detection.
pub fn ep_capture_available(
    pieces: &[Bitboard; 6],
    colors: &[Bitboard; 2],
    capturing_side: Color,
    ep_sq: u8,
) -> bool {
    if ep_sq >= 64 { return false; }
    // Pushed pawn square (on capturing_side's 4th rank from its own POV).
    let pushed_sq = if capturing_side == BLACK {
        // White pushed: pawn on rank 4 == ep_sq (rank 3) + 8.
        ep_sq.wrapping_add(8)
    } else {
        // Black pushed: pawn on rank 5 == ep_sq (rank 6) - 8.
        ep_sq.wrapping_sub(8)
    };
    if pushed_sq >= 64 { return false; }
    let pushed_bb = 1u64 << pushed_sq;
    // Enemy (capturing_side) pawns that could EP-capture sit on files adjacent
    // to pushed_sq on the same rank.
    let adj = ((pushed_bb << 1) & NOT_FILE_A) | ((pushed_bb >> 1) & NOT_FILE_H);
    let capturing_pawns = pieces[PAWN as usize] & colors[capturing_side as usize];
    adj & capturing_pawns != 0
}

impl Board {
    pub fn new() -> Self {
        Board {
            pieces: [0; 6],
            colors: [0; 2],
            mailbox: [NO_PIECE_TYPE; 64],
            side_to_move: WHITE,
            castling: 0,
            ep_square: NO_SQUARE,
            halfmove: 0,
            fullmove: 1,
            plies_from_null: 0,
            hash: 0,
            pawn_hash: 0,
            non_pawn_key: [0; 2],
            minor_key: [0; 2],
            major_key: [0; 2],
            undo_stack: Vec::with_capacity(512),
            threat_deltas: Vec::with_capacity(128),
            generate_threat_deltas: false,
        }
    }

    /// All occupied squares.
    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    /// Empty squares.
    #[inline(always)]
    pub fn empty(&self) -> Bitboard {
        !self.occupied()
    }

    /// Get piece type at a square, or NO_PIECE_TYPE. O(1) via mailbox.
    #[inline(always)]
    pub fn piece_type_at(&self, sq: u8) -> u8 {
        self.mailbox[sq as usize]
    }

    /// Get color at a square. Undefined if square is empty.
    #[inline(always)]
    pub fn color_at(&self, sq: u8) -> Color {
        if self.colors[BLACK as usize] & (1u64 << sq) != 0 { BLACK } else { WHITE }
    }

    /// Get piece (color*6 + piece_type) at a square, or NO_PIECE.
    #[inline]
    pub fn piece_at(&self, sq: u8) -> Piece {
        let pt = self.piece_type_at(sq);
        if pt == NO_PIECE_TYPE {
            return NO_PIECE;
        }
        make_piece(self.color_at(sq), pt)
    }

    /// Put a piece on the board (no hash update).
    #[inline]
    fn put_piece_no_hash(&mut self, color: Color, pt: u8, sq: u8) {
        let bb = 1u64 << sq;
        self.pieces[pt as usize] |= bb;
        self.colors[color as usize] |= bb;
        self.mailbox[sq as usize] = pt;
    }

    /// Remove a piece from the board (no hash update).
    #[inline]
    fn remove_piece_no_hash(&mut self, color: Color, pt: u8, sq: u8) {
        let bb = 1u64 << sq;
        self.pieces[pt as usize] ^= bb;
        self.colors[color as usize] ^= bb;
        self.mailbox[sq as usize] = NO_PIECE_TYPE;
    }

    /// Put a piece on the board with hash update.
    #[inline]
    fn put_piece(&mut self, color: Color, pt: u8, sq: u8) {
        self.put_piece_no_hash(color, pt, sq);
        let k = piece_key(make_piece(color, pt), sq);
        self.hash ^= k;
        if pt == PAWN { self.pawn_hash ^= k; }
        else if pt != KING {
            self.non_pawn_key[color as usize] ^= k;
            if pt == KNIGHT || pt == BISHOP { self.minor_key[color as usize] ^= k; }
            else { self.major_key[color as usize] ^= k; } // ROOK or QUEEN
        }
    }

    /// Remove a piece from the board with hash update.
    #[inline]
    pub fn remove_piece(&mut self, color: Color, pt: u8, sq: u8) {
        self.remove_piece_no_hash(color, pt, sq);
        let k = piece_key(make_piece(color, pt), sq);
        self.hash ^= k;
        if pt == PAWN { self.pawn_hash ^= k; }
        else if pt != KING {
            self.non_pawn_key[color as usize] ^= k;
            if pt == KNIGHT || pt == BISHOP { self.minor_key[color as usize] ^= k; }
            else { self.major_key[color as usize] ^= k; } // ROOK or QUEEN
        }
    }

    /// Move a piece on the board with hash update.
    #[inline]
    fn move_piece(&mut self, color: Color, pt: u8, from: u8, to: u8) {
        let from_to = (1u64 << from) | (1u64 << to);
        self.pieces[pt as usize] ^= from_to;
        self.colors[color as usize] ^= from_to;
        self.mailbox[from as usize] = NO_PIECE_TYPE;
        self.mailbox[to as usize] = pt;
        let p = make_piece(color, pt);
        let k = piece_key(p, from) ^ piece_key(p, to);
        self.hash ^= k;
        if pt == PAWN { self.pawn_hash ^= k; }
        else if pt != KING {
            self.non_pawn_key[color as usize] ^= k;
            if pt == KNIGHT || pt == BISHOP { self.minor_key[color as usize] ^= k; }
            else { self.major_key[color as usize] ^= k; } // ROOK or QUEEN
        }
    }

    /// Compute the full Zobrist hash from scratch.
    pub fn compute_hash(&self) -> u64 {
        let mut h = 0u64;

        for color in 0..2u8 {
            for pt in 0..6u8 {
                let mut bb = self.pieces[pt as usize] & self.colors[color as usize];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb);
                    h ^= piece_key(make_piece(color, pt), sq as u8);
                }
            }
        }

        // Only hash ep_square if an enemy pawn can actually make the EP capture.
        // Without this guard, the same physical position reached with or without
        // a recent double-push would hash differently, breaking rep detection
        // (Stockfish/Viridithas pattern).
        if self.ep_square != NO_SQUARE
            && ep_capture_available(&self.pieces, &self.colors, self.side_to_move, self.ep_square)
        {
            h ^= ep_key(file_of(self.ep_square));
        }

        h ^= castle_key(self.castling);

        if self.side_to_move == BLACK {
            h ^= side_key();
        }

        h
    }

    /// Parse a FEN string and set the board state.
    pub fn set_fen(&mut self, fen: &str) {
        self.pieces = [0; 6];
        self.colors = [0; 2];
        self.mailbox = [NO_PIECE_TYPE; 64];
        self.pawn_hash = 0;
        self.undo_stack.clear();

        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }

        // Piece placement
        let mut rank = 7i32;
        let mut file = 0i32;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => {
                    file += (ch as i32) - ('0' as i32);
                }
                _ => {
                    let (color, pt) = match ch {
                        'P' => (WHITE, PAWN),
                        'N' => (WHITE, KNIGHT),
                        'B' => (WHITE, BISHOP),
                        'R' => (WHITE, ROOK),
                        'Q' => (WHITE, QUEEN),
                        'K' => (WHITE, KING),
                        'p' => (BLACK, PAWN),
                        'n' => (BLACK, KNIGHT),
                        'b' => (BLACK, BISHOP),
                        'r' => (BLACK, ROOK),
                        'q' => (BLACK, QUEEN),
                        'k' => (BLACK, KING),
                        _ => continue,
                    };
                    let sq = (rank * 8 + file) as u8;
                    self.put_piece_no_hash(color, pt, sq);
                    file += 1;
                }
            }
        }

        // Side to move
        self.side_to_move = if parts.len() > 1 && parts[1] == "b" { BLACK } else { WHITE };

        // Castling
        self.castling = 0;
        if parts.len() > 2 {
            for ch in parts[2].chars() {
                match ch {
                    'K' => self.castling |= CASTLE_WK,
                    'Q' => self.castling |= CASTLE_WQ,
                    'k' => self.castling |= CASTLE_BK,
                    'q' => self.castling |= CASTLE_BQ,
                    _ => {}
                }
            }
        }

        // En passant
        self.ep_square = NO_SQUARE;
        if parts.len() > 3 && parts[3] != "-" {
            let bytes = parts[3].as_bytes();
            if bytes.len() == 2 {
                let f = bytes[0] - b'a';
                let r = bytes[1] - b'1';
                self.ep_square = square(f, r);
            }
        }

        // Halfmove clock
        self.halfmove = if parts.len() > 4 {
            parts[4].parse().unwrap_or(0)
        } else {
            0
        };
        self.plies_from_null = self.halfmove; // assume no null moves in initial position

        // Fullmove number
        self.fullmove = if parts.len() > 5 {
            parts[5].parse().unwrap_or(1)
        } else {
            1
        };

        self.hash = self.compute_hash();
        // Compute pawn hash
        self.pawn_hash = 0;
        for color in 0..2u8 {
            let mut bb = self.pieces[PAWN as usize] & self.colors[color as usize];
            while bb != 0 {
                let sq = pop_lsb(&mut bb) as u8;
                self.pawn_hash ^= piece_key(make_piece(color, PAWN), sq);
            }
        }
        // Compute per-color non-pawn / minor / major keys
        self.non_pawn_key = [0; 2];
        self.minor_key = [0; 2];
        self.major_key = [0; 2];
        for color in 0..2u8 {
            for pt in [KNIGHT, BISHOP, ROOK, QUEEN] {
                let mut bb = self.pieces[pt as usize] & self.colors[color as usize];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb) as u8;
                    let k = piece_key(make_piece(color, pt), sq);
                    self.non_pawn_key[color as usize] ^= k;
                    if pt == KNIGHT || pt == BISHOP { self.minor_key[color as usize] ^= k; }
                    else { self.major_key[color as usize] ^= k; }
                }
            }
        }
    }

    /// Create a board from a FEN string.
    pub fn from_fen(fen: &str) -> Self {
        let mut board = Board::new();
        board.set_fen(fen);
        board
    }

    /// Standard starting position.
    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    }

    /// Get the king square for a color.
    #[inline]
    pub fn king_sq(&self, color: Color) -> u8 {
        let bb = self.pieces[KING as usize] & self.colors[color as usize];
        debug_assert!(bb != 0, "No king found for color {}", color);
        if bb == 0 { return 0; } // safety fallback
        lsb(bb) as u8
    }

    /// Bitboard of every square attacked by any piece of `color`.
    /// Used by search for broader threat-aware history indexing and
    /// LMR escape-from-threat adjustments (upgrade from pawn-only).
    ///
    /// Cost: ~1-2 magic lookups per slider + per-piece table reads.
    /// Typical middlegame: 8-12 magic lookups total. Called once per
    /// non-QS node.
    #[inline]
    pub fn attacks_by_color(&self, color: Color) -> Bitboard {
        let c = color as usize;
        let occ = self.colors[0] | self.colors[1];
        let mut attacks: Bitboard = 0;

        // Pawns (dir depends on color)
        let their_pawns = self.pieces[PAWN as usize] & self.colors[c];
        attacks |= if color == WHITE {
            ((their_pawns & !FILE_A) << 7) | ((their_pawns & !FILE_H) << 9)
        } else {
            ((their_pawns & !FILE_H) >> 7) | ((their_pawns & !FILE_A) >> 9)
        };

        // Knights
        let mut knights = self.pieces[KNIGHT as usize] & self.colors[c];
        while knights != 0 {
            let sq = knights.trailing_zeros();
            knights &= knights - 1;
            attacks |= knight_attacks(sq);
        }

        // Bishops & queens (diagonal)
        let mut diag = (self.pieces[BISHOP as usize] | self.pieces[QUEEN as usize])
            & self.colors[c];
        while diag != 0 {
            let sq = diag.trailing_zeros();
            diag &= diag - 1;
            attacks |= bishop_attacks(sq, occ);
        }

        // Rooks & queens (orthogonal)
        let mut orth = (self.pieces[ROOK as usize] | self.pieces[QUEEN as usize])
            & self.colors[c];
        while orth != 0 {
            let sq = orth.trailing_zeros();
            orth &= orth - 1;
            attacks |= rook_attacks(sq, occ);
        }

        // King
        let king = self.pieces[KING as usize] & self.colors[c];
        if king != 0 {
            attacks |= king_attacks(king.trailing_zeros());
        }

        attacks
    }

    /// Squares of `color`'s pieces that are currently blocking one of
    /// `color`'s own sliders' attack-through to an enemy piece.
    ///
    /// Moving any such piece creates a discovered attack on an enemy
    /// piece along the slider's x-ray ray. Intended for movepicker
    /// bonus (B1 experiment): `bonus = value_of(xray_victim)` when
    /// move.from() ∈ xray_blockers(our_color).
    ///
    /// Cost: 2-6 slider iterations per call, each doing 1-2 magic
    /// lookups + 1 ray_extension table read per direct blocker.
    /// Typical middlegame: 10-20 magic lookups per call.
    ///
    /// Returns a bitboard of `color`'s blocker squares. The bitboard
    /// value is a flag per square — the caller does not learn which
    /// enemy piece lies behind (for that, see xray_blockers_scored).
    ///
    /// Note: this recomputes x-ray state fresh at the call site. The
    /// same information is already being computed inside
    /// `push_threats_for_piece` for NNUE input generation; if B1 lands
    /// Elo, the emission-split refactor (Path 2) amortises this cost
    /// to zero. Keep this function simple so the correctness surface
    /// is minimal and the cost is predictable.
    pub fn xray_blockers(&self, color: Color) -> Bitboard {
        let c = color as usize;
        let occ = self.colors[0] | self.colors[1];
        let our = self.colors[c];
        let their = self.colors[c ^ 1];
        let mut result: Bitboard = 0;

        // Bishops & queens — diagonal rays
        let mut diag = (self.pieces[BISHOP as usize] | self.pieces[QUEEN as usize]) & our;
        while diag != 0 {
            let s_sq = diag.trailing_zeros();
            diag &= diag - 1;
            // Direct blockers: pieces on the slider's ray reach.
            let mut our_blockers = bishop_attacks(s_sq, occ) & our;
            while our_blockers != 0 {
                let b_sq = our_blockers.trailing_zeros();
                our_blockers &= our_blockers - 1;
                // What's past this blocker on the same ray?
                let past = crate::bitboard::ray_extension(s_sq, b_sq) & occ;
                if past == 0 { continue; }
                // First occupant past blocker: low bit if s_sq < b_sq, else high bit.
                let past_sq = if s_sq < b_sq { past.trailing_zeros() } else { 63 - past.leading_zeros() };
                if their & (1u64 << past_sq) != 0 {
                    result |= 1u64 << b_sq;
                }
            }
        }

        // Rooks & queens — orthogonal rays
        let mut orth = (self.pieces[ROOK as usize] | self.pieces[QUEEN as usize]) & our;
        while orth != 0 {
            let s_sq = orth.trailing_zeros();
            orth &= orth - 1;
            let mut our_blockers = rook_attacks(s_sq, occ) & our;
            while our_blockers != 0 {
                let b_sq = our_blockers.trailing_zeros();
                our_blockers &= our_blockers - 1;
                let past = crate::bitboard::ray_extension(s_sq, b_sq) & occ;
                if past == 0 { continue; }
                let past_sq = if s_sq < b_sq { past.trailing_zeros() } else { 63 - past.leading_zeros() };
                if their & (1u64 << past_sq) != 0 {
                    result |= 1u64 << b_sq;
                }
            }
        }

        result
    }

    /// Enemy pieces that defend at least `min_defended` of their own pieces.
    ///
    /// "Overloaded defender" heuristic (T1.3): a piece defending 2+ of its own
    /// pieces can't move without exposing a defendee. MovePick bonus for
    /// moves that attack one of these squares forces the opponent to break
    /// the defence — the classic "remove the defender" tactic.
    ///
    /// Cost: iterate up to 16 enemy pieces, one magic lookup per slider.
    /// Called once per node entry.
    pub fn overloaded_defenders(&self, color: Color, min_defended: u32) -> Bitboard {
        let them = color as usize;
        let their_pieces = self.colors[them];
        let occ = self.colors[0] | self.colors[1];
        let mut result: Bitboard = 0;
        let mut pieces = their_pieces & !self.pieces[KING as usize];
        while pieces != 0 {
            let sq = pieces.trailing_zeros();
            pieces &= pieces - 1;
            let pt = self.piece_type_at(sq as u8);
            let attacks = match pt {
                PAWN => crate::attacks::pawn_attacks(color, sq),
                KNIGHT => crate::attacks::knight_attacks(sq),
                BISHOP => crate::attacks::bishop_attacks(sq, occ),
                ROOK => crate::attacks::rook_attacks(sq, occ),
                QUEEN => crate::attacks::queen_attacks(sq, occ),
                _ => 0,
            };
            if (attacks & their_pieces).count_ones() >= min_defended {
                result |= 1u64 << sq;
            }
        }
        result
    }

    /// Get bitboard of all pieces attacking a square.
    pub fn attackers_to(&self, sq: u32, occ: Bitboard) -> Bitboard {
        let knights = self.pieces[KNIGHT as usize];
        let bishops = self.pieces[BISHOP as usize];
        let rooks = self.pieces[ROOK as usize];
        let queens = self.pieces[QUEEN as usize];
        let kings = self.pieces[KING as usize];

        knight_attacks(sq) & knights
            | king_attacks(sq) & kings
            | pawn_attacks(BLACK, sq) & self.pieces[PAWN as usize] & self.colors[WHITE as usize]
            | pawn_attacks(WHITE, sq) & self.pieces[PAWN as usize] & self.colors[BLACK as usize]
            | bishop_attacks(sq, occ) & (bishops | queens)
            | rook_attacks(sq, occ) & (rooks | queens)
    }

    /// Get checkers bitboard (pieces of opponent attacking our king).
    pub fn checkers(&self) -> Bitboard {
        let ksq = self.king_sq(self.side_to_move);
        self.attackers_to(ksq as u32, self.occupied()) & self.colors[flip_color(self.side_to_move) as usize]
    }

    /// Is the side to move in check?
    #[inline]
    pub fn in_check(&self) -> bool {
        self.checkers() != 0
    }

    /// Get pinned pieces (our pieces pinned to our king by opponent sliders).
    pub fn pinned(&self) -> Bitboard {
        let us = self.side_to_move;
        let them = flip_color(us);
        let ksq = self.king_sq(us) as u32;
        let occ = self.occupied();

        let their_bishops = (self.pieces[BISHOP as usize] | self.pieces[QUEEN as usize]) & self.colors[them as usize];
        let their_rooks = (self.pieces[ROOK as usize] | self.pieces[QUEEN as usize]) & self.colors[them as usize];

        let mut pinned = 0u64;

        // Bishop/queen pinners
        let mut potential = bishop_attacks(ksq, 0) & their_bishops;
        while potential != 0 {
            let pinner_sq = pop_lsb(&mut potential);
            let between_bb = between(ksq, pinner_sq);
            let blockers = between_bb & occ;
            if popcount(blockers) == 1 && blockers & self.colors[us as usize] != 0 {
                pinned |= blockers;
            }
        }

        // Rook/queen pinners
        let mut potential = rook_attacks(ksq, 0) & their_rooks;
        while potential != 0 {
            let pinner_sq = pop_lsb(&mut potential);
            let between_bb = between(ksq, pinner_sq);
            let blockers = between_bb & occ;
            if popcount(blockers) == 1 && blockers & self.colors[us as usize] != 0 {
                pinned |= blockers;
            }
        }

        pinned
    }

    /// Is the given move legal? Assumes move is pseudo-legal.
    /// `pinned`: bitboard of our pinned pieces.
    /// `checkers`: bitboard of enemy pieces giving check.
    pub fn is_legal(&self, mv: Move, pinned: Bitboard, checkers: Bitboard) -> bool {
        let us = self.side_to_move;
        let from = move_from(mv) as u32;
        let to = move_to(mv) as u32;
        let flags = move_flags(mv);
        let ksq = self.king_sq(us) as u32;

        // En passant: special case (discovered check through EP capture)
        if flags == FLAG_EN_PASSANT {
            let captured_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
            if captured_sq >= 64 { return true; } // invalid EP, allow move through (won't be legal)
            // If in check, EP only resolves it if the captured pawn is the checker
            if checkers != 0 {
                // Double check: only king moves resolve (EP is never a king move)
                if checkers & (checkers - 1) != 0 { return false; }
                let checker_sq = crate::bitboard::lsb(checkers) as u32;
                // EP only resolves check if captured pawn is the checker
                if checker_sq != captured_sq { return false; }
            }
            let occ = (self.occupied() ^ (1u64 << from) ^ (1u64 << captured_sq)) | (1u64 << to);
            let their_bishops = (self.pieces[BISHOP as usize] | self.pieces[QUEEN as usize])
                & self.colors[flip_color(us) as usize];
            let their_rooks = (self.pieces[ROOK as usize] | self.pieces[QUEEN as usize])
                & self.colors[flip_color(us) as usize];
            return bishop_attacks(ksq, occ) & their_bishops == 0
                && rook_attacks(ksq, occ) & their_rooks == 0;
        }

        // King moves: check destination not attacked
        if from == ksq {
            let occ = self.occupied() ^ (1u64 << from);
            return self.attackers_to(to, occ) & self.colors[flip_color(us) as usize] == 0;
        }

        // In double check, only king moves are legal
        if more_than_one(checkers) {
            return false;
        }

        // In single check, non-king moves must capture checker or block the check ray
        if checkers != 0 {
            let checker_sq = lsb(checkers);
            let check_mask = (1u64 << checker_sq) | between(checker_sq, ksq);
            if (1u64 << to) & check_mask == 0 {
                return false;
            }
        }

        // Pinned pieces can only move along the pin ray
        if pinned & (1u64 << from) != 0 {
            return line(from, ksq) & (1u64 << to) != 0;
        }

        true
    }

    /// Make a move on the board. Returns false if the move is invalid
    /// (e.g., stale TT entry targeting an empty square).
    pub fn make_move(&mut self, mv: Move) -> bool {
        let us = self.side_to_move;
        let them = flip_color(us);
        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);
        let pt = self.piece_type_at(from);

        // Save undo info
        let captured = if flags == FLAG_EN_PASSANT {
            PAWN
        } else {
            self.piece_type_at(to)
        };

        // Safety: reject invalid moves (stale TT entries with hash collisions)
        if pt == NO_PIECE_TYPE || captured == KING {
            return false;
        }

        // Validate EP: ensure there's an enemy pawn to capture AND destination is empty
        if flags == FLAG_EN_PASSANT {
            let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
            if cap_sq > 63
                || (1u64 << cap_sq) & self.pieces[PAWN as usize] & self.colors[them as usize] == 0
                || self.occupied() & (1u64 << to) != 0  // destination must be empty for EP
            {
                return false;
            }
        }

        // Validate castling: ensure rook is on expected square
        if flags == FLAG_CASTLE {
            let rook_from = if to > from {
                if us == WHITE { 7u8 } else { 63u8 }
            } else {
                if us == WHITE { 0u8 } else { 56u8 }
            };
            if (1u64 << rook_from) & self.pieces[ROOK as usize] & self.colors[us as usize] == 0 {
                return false;
            }
        }

        self.undo_stack.push(UndoInfo {
            mv,
            captured,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove: self.halfmove,
            plies_from_null: self.plies_from_null,
            hash: self.hash,
            pawn_hash: self.pawn_hash,
            non_pawn_key: self.non_pawn_key,
            minor_key: self.minor_key,
            major_key: self.major_key,
            checkers: 0, // populated on demand
        });

        // Remove EP hash — only if the old EP was "legal" (we, side_to_move,
        // have a pawn that could actually make the EP capture). Must mirror
        // the condition used when the key was XOR'd in, or the hash breaks.
        if self.ep_square != NO_SQUARE
            && ep_capture_available(&self.pieces, &self.colors, self.side_to_move, self.ep_square)
        {
            self.hash ^= ep_key(file_of(self.ep_square));
        }

        // Remove old castling hash
        self.hash ^= castle_key(self.castling);

        // Clear threat deltas for this move
        let gen_threats = self.generate_threat_deltas;
        if gen_threats { self.threat_deltas.clear(); }

        // Handle captures
        if flags == FLAG_EN_PASSANT {
            let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
            debug_assert!(cap_sq < 64, "EP cap_sq out of bounds: {}", cap_sq);
            self.remove_piece(them, PAWN, cap_sq);
            if gen_threats { crate::threats::push_threats_on_change(
                &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
                self.colors[0] | self.colors[1], them, PAWN, cap_sq as u32, false); }
        } else if captured != NO_PIECE_TYPE {
            self.remove_piece(them, captured, to);
            if gen_threats { crate::threats::push_threats_on_change(
                &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
                self.colors[0] | self.colors[1], them, captured, to as u32, false); }
        }

        // Move the piece
        self.move_piece(us, pt, from, to);
        if gen_threats { crate::threats::push_threats_on_move(
            &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
            self.colors[0] | self.colors[1], us, pt, from as u32, to as u32); }

        // Handle promotion
        if is_promotion(mv) {
            let promo_pt = promotion_piece_type(mv);
            self.remove_piece(us, pt, to);   // remove pawn
            self.put_piece(us, promo_pt, to); // put promoted piece
            if gen_threats {
                crate::threats::push_threats_on_change(
                    &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
                    self.colors[0] | self.colors[1], us, pt, to as u32, false);
                crate::threats::push_threats_on_change(
                    &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
                    self.colors[0] | self.colors[1], us, promo_pt, to as u32, true);
            }
        }

        // Handle castling
        if flags == FLAG_CASTLE {
            let (rook_from, rook_to) = if to > from {
                if us == WHITE { (7u8, 5u8) } else { (63u8, 61u8) }
            } else {
                if us == WHITE { (0u8, 3u8) } else { (56u8, 59u8) }
            };
            self.move_piece(us, ROOK, rook_from, rook_to);
            if gen_threats { crate::threats::push_threats_on_move(
                &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
                self.colors[0] | self.colors[1], us, ROOK, rook_from as u32, rook_to as u32); }
        }

        // Update castling rights (for any move from/to relevant squares)
        self.castling &= castle_mask(from) & castle_mask(to);

        // Update en passant — detect double push by distance.
        // Only XOR ep_key into the hash if the OTHER side (them = the next
        // mover) actually has a pawn that could EP-capture. Otherwise the
        // ep_square is physically recorded but invisible to the hash, so
        // positions reachable with-or-without this double-push still
        // collide correctly for rep detection.
        self.ep_square = NO_SQUARE;
        if pt == PAWN && ((to as i32) - (from as i32)).unsigned_abs() == 16 {
            let new_ep = if us == WHITE { from.wrapping_add(8) } else { from.wrapping_sub(8) };
            self.ep_square = new_ep;
            if ep_capture_available(&self.pieces, &self.colors, them, new_ep) {
                self.hash ^= ep_key(file_of(new_ep));
            }
        }

        // Update castling hash
        self.hash ^= castle_key(self.castling);

        // Update halfmove clock
        if pt == PAWN || captured != NO_PIECE_TYPE {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }
        self.plies_from_null += 1;

        // Update fullmove
        if us == BLACK {
            self.fullmove += 1;
        }

        // Flip side
        self.side_to_move = them;
        self.hash ^= side_key();

        true
    }

    // Threat delta methods use free functions to avoid borrow issues with &mut self

    /// Unmake the last move.
    pub fn unmake_move(&mut self) {
        let undo = self.undo_stack.pop().expect("unmake_move: empty undo stack");
        let mv = undo.mv;
        let them = self.side_to_move; // after unmake, "them" is who just moved
        let us = flip_color(them);

        self.side_to_move = us;

        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);

        // Undo promotion
        if is_promotion(mv) {
            let promo_pt = promotion_piece_type(mv);
            self.remove_piece_no_hash(us, promo_pt, to);
            self.put_piece_no_hash(us, PAWN, to);
        }

        // Undo piece move
        let pt = self.piece_type_at(to);
        // If the piece is missing at `to`, the board was corrupted by a child search.
        // Restore from the saved hash (which was correct at make time).
        if pt == NO_PIECE_TYPE {
            // Board corruption detected — restore what we can from undo
            self.castling = undo.castling;
            self.ep_square = undo.ep_square;
            self.halfmove = undo.halfmove;
            self.plies_from_null = undo.plies_from_null;
            self.hash = undo.hash;
            self.pawn_hash = undo.pawn_hash;
            self.non_pawn_key = undo.non_pawn_key;
            self.minor_key = undo.minor_key;
            self.major_key = undo.major_key;
            self.side_to_move = us;
            if us == BLACK { self.fullmove -= 1; }
            return;
        }
        let from_to = (1u64 << from) | (1u64 << to);
        self.pieces[pt as usize] ^= from_to;
        self.colors[us as usize] ^= from_to;
        self.mailbox[to as usize] = NO_PIECE_TYPE;
        self.mailbox[from as usize] = pt;

        // Restore captured piece
        if flags == FLAG_EN_PASSANT {
            let cap_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
            if cap_sq < 64 {
                self.put_piece_no_hash(them, PAWN, cap_sq);
            }
        } else if undo.captured != NO_PIECE_TYPE {
            self.put_piece_no_hash(them, undo.captured, to);
        }

        // Undo castling rook move
        if flags == FLAG_CASTLE {
            let (rook_from, rook_to) = if to > from {
                if us == WHITE { (7u8, 5u8) } else { (63u8, 61u8) }
            } else {
                if us == WHITE { (0u8, 3u8) } else { (56u8, 59u8) }
            };
            let rook_from_to = (1u64 << rook_from) | (1u64 << rook_to);
            self.pieces[ROOK as usize] ^= rook_from_to;
            self.colors[us as usize] ^= rook_from_to;
            self.mailbox[rook_to as usize] = NO_PIECE_TYPE;
            self.mailbox[rook_from as usize] = ROOK;
        }

        // Restore state
        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove = undo.halfmove;
        self.plies_from_null = undo.plies_from_null;
        self.hash = undo.hash;
        self.pawn_hash = undo.pawn_hash;
        self.non_pawn_key = undo.non_pawn_key;
        self.minor_key = undo.minor_key;
        self.major_key = undo.major_key;

        if us == BLACK {
            self.fullmove -= 1;
        }
    }

    /// Make a null move (just flip side, update EP).
    pub fn make_null_move(&mut self) {
        self.threat_deltas.clear(); // null move = no piece changes = no threat deltas
        self.undo_stack.push(UndoInfo {
            mv: NO_MOVE,
            captured: NO_PIECE_TYPE,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove: self.halfmove,
            plies_from_null: self.plies_from_null,
            hash: self.hash,
            pawn_hash: self.pawn_hash,
            non_pawn_key: self.non_pawn_key,
            minor_key: self.minor_key,
            major_key: self.major_key,
            checkers: 0,
        });

        if self.ep_square != NO_SQUARE {
            // Mirror the conditional XOR used when the key was added.
            if ep_capture_available(&self.pieces, &self.colors, self.side_to_move, self.ep_square) {
                self.hash ^= ep_key(file_of(self.ep_square));
            }
            self.ep_square = NO_SQUARE;
        }

        self.side_to_move = flip_color(self.side_to_move);
        self.hash ^= side_key();
        self.halfmove += 1;
        self.plies_from_null = 0; // reset on null move for cuckoo
    }

    /// Unmake a null move.
    pub fn unmake_null_move(&mut self) {
        let undo = self.undo_stack.pop().expect("unmake_null_move: empty undo stack");
        self.side_to_move = flip_color(self.side_to_move);
        self.ep_square = undo.ep_square;
        self.halfmove = undo.halfmove;
        self.plies_from_null = undo.plies_from_null;
        self.hash = undo.hash;
        self.pawn_hash = undo.pawn_hash;
        self.non_pawn_key = undo.non_pawn_key;
        self.minor_key = undo.minor_key;
        self.major_key = undo.major_key;
    }

    /// Display the board as ASCII art.
    pub fn display(&self) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            s.push_str(&format!("{}  ", rank + 1));
            for file in 0..8 {
                let sq = square(file, rank);
                let p = self.piece_at(sq);
                let ch = if p == NO_PIECE {
                    '.'
                } else {
                    let pt = piece_type(p);
                    let c = piece_color(p);
                    let base = match pt {
                        PAWN => 'p',
                        KNIGHT => 'n',
                        BISHOP => 'b',
                        ROOK => 'r',
                        QUEEN => 'q',
                        KING => 'k',
                        _ => '?',
                    };
                    if c == WHITE { base.to_ascii_uppercase() } else { base }
                };
                s.push(ch);
                s.push(' ');
            }
            s.push('\n');
        }
        s.push_str("   a b c d e f g h\n");
        s
    }

    /// Convert board to FEN string.
    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = square(file as u8, rank as u8);
                let p = self.piece_at(sq);
                if p == NO_PIECE {
                    empty += 1;
                } else {
                    if empty > 0 {
                        fen.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let pt = piece_type(p);
                    let c = piece_color(p);
                    let ch = match pt {
                        PAWN => 'p',
                        KNIGHT => 'n',
                        BISHOP => 'b',
                        ROOK => 'r',
                        QUEEN => 'q',
                        KING => 'k',
                        _ => '?',
                    };
                    fen.push(if c == WHITE { ch.to_ascii_uppercase() } else { ch });
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(if self.side_to_move == WHITE { 'w' } else { 'b' });

        fen.push(' ');
        if self.castling == 0 {
            fen.push('-');
        } else {
            if self.castling & CASTLE_WK != 0 { fen.push('K'); }
            if self.castling & CASTLE_WQ != 0 { fen.push('Q'); }
            if self.castling & CASTLE_BK != 0 { fen.push('k'); }
            if self.castling & CASTLE_BQ != 0 { fen.push('q'); }
        }

        fen.push(' ');
        if self.ep_square == NO_SQUARE {
            fen.push('-');
        } else {
            fen.push_str(&square_name(self.ep_square));
        }

        fen.push_str(&format!(" {} {}", self.halfmove, self.fullmove));

        fen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() { crate::init(); }

    #[test]
    fn test_startpos() {
        init();
        let b = Board::startpos();
        assert_eq!(popcount(b.occupied()), 32);
        assert_eq!(popcount(b.colors[WHITE as usize]), 16);
        assert_eq!(popcount(b.colors[BLACK as usize]), 16);
        assert_eq!(b.side_to_move, WHITE);
        assert_eq!(b.castling, CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ);
        assert_eq!(b.ep_square, NO_SQUARE);
    }

    #[test]
    fn test_fen_roundtrip() {
        init();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];
        for fen in &fens {
            let b = Board::from_fen(fen);
            assert_eq!(b.to_fen(), *fen, "FEN roundtrip failed for: {}", fen);
        }
    }

    #[test]
    fn test_piece_at() {
        init();
        let b = Board::startpos();
        assert_eq!(b.piece_at(0), make_piece(WHITE, ROOK));   // a1
        assert_eq!(b.piece_at(4), make_piece(WHITE, KING));   // e1
        assert_eq!(b.piece_at(8), make_piece(WHITE, PAWN));   // a2
        assert_eq!(b.piece_at(63), make_piece(BLACK, ROOK));  // h8
        assert_eq!(b.piece_at(28), NO_PIECE);                 // e4
    }

    #[test]
    fn test_make_unmake_simple() {
        init();
        let mut b = Board::startpos();
        let hash_before = b.hash;
        let fen_before = b.to_fen();

        // e2e4
        let mv = make_move(12, 28, FLAG_DOUBLE_PUSH);
        b.make_move(mv);
        assert_eq!(b.piece_type_at(28), PAWN);
        assert_eq!(b.piece_type_at(12), NO_PIECE_TYPE);
        assert_eq!(b.side_to_move, BLACK);
        assert_eq!(b.ep_square, 20); // e3

        b.unmake_move();
        assert_eq!(b.hash, hash_before);
        assert_eq!(b.to_fen(), fen_before);
    }

    #[test]
    fn test_make_unmake_capture() {
        init();
        // Position with a capture available
        let mut b = Board::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
        let hash_before = b.hash;
        let fen_before = b.to_fen();

        // exd5
        let mv = make_move(28, 35, FLAG_NONE);
        b.make_move(mv);
        assert_eq!(b.piece_type_at(35), PAWN);
        assert_eq!(b.color_at(35), WHITE);

        b.unmake_move();
        assert_eq!(b.hash, hash_before);
        assert_eq!(b.to_fen(), fen_before);
    }

    #[test]
    fn test_hash_incremental() {
        init();
        let mut b = Board::startpos();

        // Make a few moves and verify hash stays consistent
        let mv = make_move(12, 28, FLAG_DOUBLE_PUSH); // e2e4
        b.make_move(mv);
        assert_eq!(b.hash, b.compute_hash());

        let mv = make_move(52, 36, FLAG_DOUBLE_PUSH); // e7e5
        b.make_move(mv);
        assert_eq!(b.hash, b.compute_hash());

        let mv = make_move(6, 21, FLAG_NONE); // Nf3
        b.make_move(mv);
        assert_eq!(b.hash, b.compute_hash());
    }

    #[test]
    fn test_castling_move() {
        init();
        let mut b = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        let fen_before = b.to_fen();
        let hash_before = b.hash;

        // White kingside castle: e1g1
        let mv = make_move(4, 6, FLAG_CASTLE);
        b.make_move(mv);
        assert_eq!(b.piece_type_at(6), KING);   // g1
        assert_eq!(b.piece_type_at(5), ROOK);   // f1
        assert_eq!(b.piece_type_at(4), NO_PIECE_TYPE);
        assert_eq!(b.piece_type_at(7), NO_PIECE_TYPE);
        assert_eq!(b.castling & (CASTLE_WK | CASTLE_WQ), 0); // white lost both

        b.unmake_move();
        assert_eq!(b.to_fen(), fen_before);
        assert_eq!(b.hash, hash_before);
    }

    #[test]
    fn test_ep_capture() {
        init();
        let mut b = Board::from_fen("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3");
        let fen_before = b.to_fen();
        let hash_before = b.hash;

        // fxe6 en passant
        let mv = make_move(37, 44, FLAG_EN_PASSANT);
        b.make_move(mv);
        assert_eq!(b.piece_type_at(44), PAWN);     // e6: white pawn
        assert_eq!(b.piece_type_at(37), NO_PIECE_TYPE); // f5: empty
        assert_eq!(b.piece_type_at(36), NO_PIECE_TYPE); // e5: captured pawn gone

        b.unmake_move();
        assert_eq!(b.to_fen(), fen_before);
        assert_eq!(b.hash, hash_before);
    }

    #[test]
    fn test_promotion() {
        init();
        let mut b = Board::from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1");
        let fen_before = b.to_fen();
        let hash_before = b.hash;

        // a7a8=Q
        let mv = make_move(48, 56, FLAG_PROMOTE_Q);
        b.make_move(mv);
        assert_eq!(b.piece_type_at(56), QUEEN);
        assert_eq!(b.color_at(56), WHITE);

        b.unmake_move();
        assert_eq!(b.to_fen(), fen_before);
        assert_eq!(b.hash, hash_before);
    }

    /// Deterministic fuzzer: plays random legal games from several start
    /// positions and verifies that the incremental hash (maintained by
    /// make_move/unmake_move) exactly equals compute_hash() at every ply.
    ///
    /// This catches any drift in the conditional ep_key XOR — if the
    /// condition used at "remove old ep_key" disagrees with the condition
    /// used at "add new ep_key" for any move sequence, the incremental
    /// hash diverges from compute_hash and TT lookup breaks.
    ///
    /// Not a behavioural test — purely a hash-invariant guard for the
    /// 2026-04-17 Zobrist EP fix. Analogous to threat_accum::fuzz_random_games.
    #[test]
    fn fuzz_hash_incremental_matches_compute() {
        use crate::movegen::generate_legal_moves;
        init();

        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // EP-rich positions: lots of pawns near rank 5 / rank 4
            "rnbqkbnr/1ppp1ppp/p7/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3",
            "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
            // Dead-EP position: double push available with no adjacent enemy pawn
            "4k3/p7/8/8/8/8/1P6/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        const PLIES: usize = 80;
        const GAMES: usize = 30;

        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for game in 0..GAMES {
                let seed: u32 = 0x1EE7D00Du32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::from_fen(fen);
                // Baseline invariant on starting position.
                assert_eq!(
                    board.hash, board.compute_hash(),
                    "start hash mismatch at fen_idx={} fen={}", fen_idx, fen
                );

                for ply in 0..PLIES {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mv = legal.moves[(next_u32(&mut rng) as usize) % legal.len];

                    board.make_move(mv);

                    let incr = board.hash;
                    let fresh = board.compute_hash();
                    if incr != fresh {
                        panic!(
                            "hash drift: fen_idx={} game={} ply={} move={} \
                             incremental={:#x} compute_hash={:#x} diff={:#x} seed={:#x}\n\
                             fen={} ep={}",
                            fen_idx, game, ply,
                            crate::types::move_to_uci(mv),
                            incr, fresh, incr ^ fresh, seed,
                            board.to_fen(), board.ep_square,
                        );
                    }
                }
            }
        }
    }

    /// Regression: the Zobrist hash must include ep_key iff an enemy pawn
    /// can actually make the EP capture. Without this guard, the same
    /// physical position reached via a double-push with no adjacent enemy
    /// pawn would hash differently from the same position reached by a
    /// single-push path, silently breaking threefold-repetition detection.
    ///
    /// Caught by an `info depth ... pv` warning from fastchess on a v9
    /// game where a forced-rep endgame was scored as -80cp instead of the
    /// draw contempt value (~-19cp).
    #[test]
    fn zobrist_ep_only_when_capturable() {
        init();

        // Case A: double push where NO enemy pawn can capture.
        // White pushes d2-d4. Black has no pawn on c4 or e4. EP is illegal.
        // The same physical position is reachable via two d-pawn single-pushes
        // (d2-d3, then d3-d4). Both hashes must be equal.
        let mut b_double = Board::from_fen("4k3/8/8/8/8/8/3P4/4K3 w - - 0 1");
        let d2d4 = make_move(11, 27, FLAG_NONE); // d2 -> d4 (distance 16 triggers EP)
        b_double.make_move(d2d4);
        // After d2-d4 with no enemy pawn to capture, ep_square may be set to
        // d3 (internal state) but the hash must NOT include ep_key(d).
        // Give black a no-op and white another no-op to reach a comparable state.
        let ke8f8 = make_move(60, 61, FLAG_NONE); // e8 -> f8 (just to flip sides)
        // Actually simpler: compare the single-push-path hash directly.
        let b_single = {
            let mut b = Board::from_fen("4k3/8/8/8/8/8/3P4/4K3 w - - 0 1");
            let d2d3 = make_move(11, 19, FLAG_NONE);
            b.make_move(d2d3);       // d2-d3 (pawn to d3; black to move)
            let ke8f8 = make_move(60, 61, FLAG_NONE);
            b.make_move(ke8f8);      // black king idle
            let d3d4 = make_move(19, 27, FLAG_NONE);
            b.make_move(d3d4);       // d3-d4 (no EP here either)
            b
        };
        // Reach the same physical position from the double-push path:
        // after d2-d4, play Ke8-f8, then a null move slot... we need white to move
        // for the physical state to match b_single. Instead, just arrange
        // b_double to end with black to move and compare to a b_single with black
        // to move after d2-d4.
        //
        // Easier form: compare *after the double push*, black-to-move, against
        // *after single-push + black no-op + single-push*, black-to-move.
        let _ = ke8f8; // silence unused
        drop(b_double);

        // Simpler head-to-head: two black-to-move positions that should be identical.
        let fen_after_double = {
            let mut b = Board::from_fen("4k3/8/8/8/8/8/3P4/4K3 w - - 0 1");
            let d2d4 = make_move(11, 27, FLAG_NONE);
            b.make_move(d2d4);
            b
        };
        let fen_after_two_single = {
            let mut b = Board::from_fen("4k3/8/8/8/8/3P4/8/4K3 w - - 0 1");
            let d3d4 = make_move(19, 27, FLAG_NONE);
            b.make_move(d3d4);
            b
        };
        // Physical state must match (piece bitboards, side, castling, no EP effect).
        assert_eq!(fen_after_double.pieces, fen_after_two_single.pieces);
        assert_eq!(fen_after_double.colors, fen_after_two_single.colors);
        assert_eq!(fen_after_double.side_to_move, fen_after_two_single.side_to_move);
        assert_eq!(fen_after_double.castling, fen_after_two_single.castling);
        // Hashes must be equal — EP is not legal in either (no black pawn on c4/e4).
        assert_eq!(
            fen_after_double.hash, fen_after_two_single.hash,
            "Zobrist hashes must match when EP is physically unreachable.\n\
             double-push hash = {:#x}, single-push hash = {:#x}",
            fen_after_double.hash, fen_after_two_single.hash
        );

        // Case B: double push where EP IS legal (black pawn on e4).
        // In that case, the hash should include ep_key(d) AND not match the
        // no-EP version.
        let ep_legal_pos = {
            let mut b = Board::from_fen("4k3/8/8/8/4p3/8/3P4/4K3 w - - 0 1");
            let d2d4 = make_move(11, 27, FLAG_NONE); // d2-d4, ep_sq=d3, black pawn on e4 can capture
            b.make_move(d2d4);
            b
        };
        let no_ep_version = {
            let mut b = Board::from_fen("4k3/8/8/8/4p3/3P4/8/4K3 w - - 0 1");
            let d3d4 = make_move(19, 27, FLAG_NONE);
            b.make_move(d3d4);
            b
        };
        // Physical bitboards should match, but the EP-legal version must have
        // ep_key XOR'd in.
        assert_eq!(ep_legal_pos.pieces, no_ep_version.pieces);
        assert_eq!(ep_legal_pos.colors, no_ep_version.colors);
        assert_ne!(
            ep_legal_pos.hash, no_ep_version.hash,
            "Zobrist hashes MUST differ when EP is legal — ep_key(d) should \
             distinguish them. double-push hash = {:#x}, no-EP hash = {:#x}",
            ep_legal_pos.hash, no_ep_version.hash
        );
        // The XOR of the two must equal ep_key(d-file = 3).
        assert_eq!(
            ep_legal_pos.hash ^ no_ep_version.hash,
            crate::zobrist::ep_key(3),
            "hash difference should be exactly ep_key(d)"
        );

        // Case C: incremental hash vs. from-scratch compute_hash must agree
        // in both cases (hash invariant).
        assert_eq!(fen_after_double.hash, fen_after_double.compute_hash());
        assert_eq!(ep_legal_pos.hash, ep_legal_pos.compute_hash());
    }

    /// Recompute the four auxiliary Zobrist keys (pawn_hash, non_pawn_key,
    /// minor_key, major_key) from the current piece/color bitboards. Mirrors
    /// the logic in `set_fen` at the "Compute per-color" block. Used by
    /// fuzzers to verify incremental updates in put_piece / remove_piece /
    /// move_piece stay coherent.
    fn recompute_aux_keys(b: &Board) -> (u64, [u64; 2], [u64; 2], [u64; 2]) {
        let mut pawn = 0u64;
        let mut np = [0u64; 2];
        let mut minor = [0u64; 2];
        let mut major = [0u64; 2];
        for color in 0..2u8 {
            let mut pbb = b.pieces[PAWN as usize] & b.colors[color as usize];
            while pbb != 0 {
                let sq = pop_lsb(&mut pbb) as u8;
                pawn ^= piece_key(make_piece(color, PAWN), sq);
            }
            for pt in [KNIGHT, BISHOP, ROOK, QUEEN] {
                let mut bb = b.pieces[pt as usize] & b.colors[color as usize];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb) as u8;
                    let k = piece_key(make_piece(color, pt), sq);
                    np[color as usize] ^= k;
                    if pt == KNIGHT || pt == BISHOP { minor[color as usize] ^= k; }
                    else { major[color as usize] ^= k; }
                }
            }
        }
        (pawn, np, minor, major)
    }

    /// Property-based fuzzer for the four auxiliary Zobrist keys
    /// (pawn_hash, non_pawn_key, minor_key, major_key) used by correction
    /// history. After every move, incremental values must equal from-scratch
    /// recomputation. These hashes drive correction-history indexing — a
    /// silent divergence would poison per-bucket corrections and show up as
    /// degraded search quality with no other symptom.
    ///
    /// Analogous to `fuzz_hash_incremental_matches_compute` for the main
    /// Zobrist key, but for the four auxiliary keys which have no standalone
    /// from-scratch computation in production code.
    #[test]
    fn fuzz_aux_keys_match_incremental() {
        use crate::movegen::generate_legal_moves;
        init();

        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
            // Rich in promotions / captures
            "4k3/PPPPPPPP/8/8/8/8/pppppppp/4K3 w - - 0 1",
            // Mostly minors to exercise minor_key
            "4k3/pp6/2nb4/8/8/2NB4/PP6/4K3 w - - 0 1",
            // Mostly majors to exercise major_key
            "r2qk2r/8/8/8/8/8/8/R2QK2R w KQkq - 0 1",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        const PLIES: usize = 80;
        const GAMES: usize = 20;

        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for game in 0..GAMES {
                let seed: u32 = 0xA11CEFu32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::from_fen(fen);
                // Baseline invariant on starting position.
                let (p, np, mi, ma) = recompute_aux_keys(&board);
                assert_eq!(board.pawn_hash, p, "start pawn_hash mismatch fen={}", fen);
                assert_eq!(board.non_pawn_key, np, "start np mismatch fen={}", fen);
                assert_eq!(board.minor_key, mi, "start minor mismatch fen={}", fen);
                assert_eq!(board.major_key, ma, "start major mismatch fen={}", fen);

                for ply in 0..PLIES {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mv = legal.moves[(next_u32(&mut rng) as usize) % legal.len];

                    board.make_move(mv);

                    let (p, np, mi, ma) = recompute_aux_keys(&board);
                    if board.pawn_hash != p || board.non_pawn_key != np
                        || board.minor_key != mi || board.major_key != ma
                    {
                        panic!(
                            "aux key drift: fen_idx={} game={} ply={} move={} seed={:#x}\n\
                             pawn  incr={:#x} fresh={:#x}\n\
                             np    incr=[{:#x},{:#x}] fresh=[{:#x},{:#x}]\n\
                             minor incr=[{:#x},{:#x}] fresh=[{:#x},{:#x}]\n\
                             major incr=[{:#x},{:#x}] fresh=[{:#x},{:#x}]\n\
                             fen={}",
                            fen_idx, game, ply,
                            crate::types::move_to_uci(mv), seed,
                            board.pawn_hash, p,
                            board.non_pawn_key[0], board.non_pawn_key[1], np[0], np[1],
                            board.minor_key[0], board.minor_key[1], mi[0], mi[1],
                            board.major_key[0], board.major_key[1], ma[0], ma[1],
                            board.to_fen(),
                        );
                    }
                }
            }
        }
    }

    /// Null-move round-trip fuzzer. Interleaves null moves with regular
    /// moves and verifies full state is restored after unmake.
    /// Null moves never happen while in check (gated in search), which
    /// mirrors real usage.
    #[test]
    fn fuzz_null_move_round_trip() {
        use crate::movegen::generate_legal_moves;
        init();

        const FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // EP-triggering
            "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
            "4k3/p7/8/1P6/8/8/8/4K3 b - - 0 1",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        const PLIES: usize = 40;
        const GAMES: usize = 15;

        for (fen_idx, fen) in FENS.iter().enumerate() {
            for game in 0..GAMES {
                let seed: u32 = 0xDEADBEEFu32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::from_fen(fen);
                let orig_hash = board.hash;
                let orig_fen = board.to_fen();
                let orig_pawn = board.pawn_hash;
                let orig_np = board.non_pawn_key;
                let orig_minor = board.minor_key;
                let orig_major = board.major_key;

                // Track whether each op was null (true) or regular (false)
                // so we unmake with the right function.
                let mut ops: Vec<bool> = Vec::new();
                for _ in 0..PLIES {
                    // 1 in 3 chance of null, but only if not in check.
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let want_null = (next_u32(&mut rng) % 3) == 0;
                    let in_check = board.checkers() != 0;
                    if want_null && !in_check {
                        board.make_null_move();
                        ops.push(true);
                    } else {
                        let mv = legal.moves[(next_u32(&mut rng) as usize) % legal.len];
                        board.make_move(mv);
                        ops.push(false);
                    }

                    // After every op, incremental hash must match compute_hash.
                    assert_eq!(
                        board.hash, board.compute_hash(),
                        "hash drift during null/regular mix: fen_idx={} game={} op_count={}",
                        fen_idx, game, ops.len()
                    );
                }

                for is_null in ops.iter().rev() {
                    if *is_null { board.unmake_null_move(); }
                    else { board.unmake_move(); }
                }

                assert_eq!(board.to_fen(), orig_fen,
                    "fen drift after null/regular roundtrip fen_idx={} game={}", fen_idx, game);
                assert_eq!(board.hash, orig_hash);
                assert_eq!(board.pawn_hash, orig_pawn);
                assert_eq!(board.non_pawn_key, orig_np);
                assert_eq!(board.minor_key, orig_minor);
                assert_eq!(board.major_key, orig_major);
            }
        }
    }

    /// Full make/unmake round-trip fuzzer. Plays a random sequence of
    /// moves, then unmakes them all one at a time and asserts full state
    /// equality with the starting position. Covers every field modified
    /// by make_move/unmake_move: mailbox, bitboards, castling, ep_square,
    /// halfmove, fullmove, all 4+1 hashes. If unmake silently
    /// omits restoring any field, this detects it.
    ///
    /// Complements `fuzz_hash_incremental_matches_compute` (make-only)
    /// and `fuzz_aux_keys_match_incremental` (make-only, aux keys).
    #[test]
    fn fuzz_make_unmake_round_trip() {
        use crate::movegen::generate_legal_moves;
        init();

        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            // Promotion-rich: many different unmake scenarios
            "4k3/PPPPPPPP/8/8/8/8/pppppppp/4K3 w - - 0 1",
            // EP-triggerable
            "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
            "4k3/p7/8/1P6/8/8/8/4K3 b - - 0 1",
            // Endgame with castling rights intact
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        const PLIES: usize = 40;
        const GAMES: usize = 15;

        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for game in 0..GAMES {
                let seed: u32 = 0xB00B1E5u32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::from_fen(fen);
                // Save full baseline state.
                let orig_fen = board.to_fen();
                let orig_hash = board.hash;
                let orig_pawn = board.pawn_hash;
                let orig_np = board.non_pawn_key;
                let orig_minor = board.minor_key;
                let orig_major = board.major_key;
                let orig_mailbox = board.mailbox;
                let orig_pieces = board.pieces;
                let orig_colors = board.colors;
                let orig_castling = board.castling;
                let orig_ep = board.ep_square;
                let orig_halfmove = board.halfmove;
                let orig_fullmove = board.fullmove;
                let orig_stm = board.side_to_move;

                // Random sequence of legal moves.
                let mut move_count = 0;
                for _ in 0..PLIES {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mv = legal.moves[(next_u32(&mut rng) as usize) % legal.len];
                    board.make_move(mv);
                    move_count += 1;
                }

                // Unmake everything.
                for _ in 0..move_count {
                    board.unmake_move();
                }

                // Full state equality check.
                let mismatch =
                    board.hash != orig_hash
                    || board.pawn_hash != orig_pawn
                    || board.non_pawn_key != orig_np
                    || board.minor_key != orig_minor
                    || board.major_key != orig_major
                    || board.mailbox != orig_mailbox
                    || board.pieces != orig_pieces
                    || board.colors != orig_colors
                    || board.castling != orig_castling
                    || board.ep_square != orig_ep
                    || board.halfmove != orig_halfmove
                    || board.fullmove != orig_fullmove
                    || board.side_to_move != orig_stm;
                if mismatch {
                    panic!(
                        "round-trip mismatch: fen_idx={} game={} plies={} seed={:#x}\n\
                         orig fen   = {}\n\
                         post-unmake= {}\n\
                         hash   orig={:#x} now={:#x}\n\
                         pawn   orig={:#x} now={:#x}\n\
                         np     orig={:?} now={:?}\n\
                         minor  orig={:?} now={:?}\n\
                         major  orig={:?} now={:?}\n\
                         castle orig={} now={}  ep orig={} now={}\n\
                         stm    orig={} now={}  hm orig={} now={} fm orig={} now={}",
                        fen_idx, game, move_count, seed,
                        orig_fen, board.to_fen(),
                        orig_hash, board.hash,
                        orig_pawn, board.pawn_hash,
                        orig_np, board.non_pawn_key,
                        orig_minor, board.minor_key,
                        orig_major, board.major_key,
                        orig_castling, board.castling,
                        orig_ep, board.ep_square,
                        orig_stm, board.side_to_move,
                        orig_halfmove, board.halfmove,
                        orig_fullmove, board.fullmove,
                    );
                }
            }
        }
    }

    /// Production-equivalent repetition check (mirrors the inline logic
    /// in search.rs:1567 at the main-search call site). Kept in test
    /// code to pin the exact iteration behaviour; must be updated if
    /// production logic changes.
    fn check_repetition_main(board: &Board) -> bool {
        let stack_len = board.undo_stack.len();
        let limit = (board.halfmove as usize).min(stack_len);
        let mut i = 2usize;
        while i <= limit {
            if board.undo_stack[stack_len - i].hash == board.hash {
                return true;
            }
            i += 2;
        }
        false
    }

    /// QS-variant repetition check (mirrors search.rs:2769). Uses
    /// `halfmove == 0` break instead of a limit = halfmove cap.
    fn check_repetition_qs(board: &Board) -> bool {
        let hash = board.hash;
        for undo in board.undo_stack.iter().rev().skip(1).step_by(2) {
            if undo.hash == hash { return true; }
            if undo.halfmove == 0 { break; }
        }
        false
    }

    /// Basic 4-ply knight dance creates a perfect repetition of the
    /// starting position. Both rep detectors must see it.
    #[test]
    fn repetition_4ply_knight_dance_detected() {
        init();
        let mut b = Board::startpos();
        let hash0 = b.hash;
        // Nf3 (white knight b1→g1? no, g1→f3). White e1-adj is g1=6, f3=21.
        let nf3 = make_move(6, 21, FLAG_NONE);
        let nc6 = make_move(57, 42, FLAG_NONE); // Black b8(57) → c6(42)
        let ng1 = make_move(21, 6, FLAG_NONE);
        let nb8 = make_move(42, 57, FLAG_NONE);
        b.make_move(nf3);
        b.make_move(nc6);
        b.make_move(ng1);
        b.make_move(nb8);
        assert_eq!(b.hash, hash0, "knight dance must return to startpos hash");
        assert!(check_repetition_main(&b), "main-search rep check must detect 4-ply loop");
        assert!(check_repetition_qs(&b), "QS rep check must detect 4-ply loop");
    }

    /// After one knight dance, the current position does NOT match
    /// any previous same-STM position. Rep must NOT trigger.
    #[test]
    fn no_repetition_after_single_knight_move() {
        init();
        let mut b = Board::startpos();
        let nf3 = make_move(6, 21, FLAG_NONE);
        b.make_move(nf3);
        // Now STM is Black, stack_len=1. Limit=min(halfmove=1, 1)=1.
        // Loop starts at i=2; 2 > 1, so loop body never runs → no rep.
        assert!(!check_repetition_main(&b), "single move: no rep possible");
        assert!(!check_repetition_qs(&b), "QS same");
    }

    /// Irreversible-move boundary: an intervening pawn move resets
    /// halfmove to 0, which MUST prevent rep detection from seeing
    /// past it. Scenario: play Nf3 Nc6 then a pawn move, then
    /// attempt to reach the post-Nf3-Nc6 hash would only happen if
    /// we played more knight moves — but halfmove reset means the
    /// earlier position is no longer reachable by definition.
    ///
    /// Test: construct a sequence where the current hash equals a
    /// stack entry OLDER than the most recent pawn move, and verify
    /// the limit capping prevents the false positive.
    #[test]
    fn repetition_honours_halfmove_cap() {
        init();
        let mut b = Board::startpos();

        // 1. Nf3 Nc6 — halfmove 1, 2
        b.make_move(make_move(6, 21, FLAG_NONE)); // Nf3
        b.make_move(make_move(57, 42, FLAG_NONE)); // Nc6

        // Artificially corrupt undo_stack[0].hash to match current
        // hash, simulating a hypothetical distant repetition across
        // an irreversible move. This isn't reachable legally but
        // pins the limit logic.
        let saved = b.undo_stack[0].hash;
        b.undo_stack[0].hash = b.hash;
        // Current halfmove = 2, stack_len = 2, limit = 2.
        // Loop i=2: stack[stack_len - 2] = stack[0] — would match!
        // This is the POSITIVE case: within halfmove window → detected.
        assert!(check_repetition_main(&b), "within-halfmove false-hash should detect");

        // Now play e4, which resets halfmove to 0. Any pre-e4 match
        // must NOT be returned.
        b.undo_stack[0].hash = saved; // undo the corruption
        b.make_move(make_move(12, 28, FLAG_DOUBLE_PUSH)); // e2e4
        assert_eq!(b.halfmove, 0, "pawn move resets halfmove");

        // stack now has 3 entries: startpos, post-Nf3, post-Nc6.
        // Current position is post-Nf3-Nc6-e4 with halfmove=0.
        // Main-search: limit = min(0, 3) = 0, loop never executes → correctly returns false.
        // Corrupt stack[2].hash (most recent = post-Nc6 state) to match current: main-search
        // must NOT see it because limit = 0.
        let saved2 = b.undo_stack[2].hash;
        b.undo_stack[2].hash = b.hash;
        assert!(!check_repetition_main(&b),
            "main-search: halfmove=0 limit must block detection past irreversible move");
        b.undo_stack[2].hash = saved2;
    }

    /// Random-games fuzzer: play random legal games and verify that
    /// both rep detectors agree on their verdict at every ply.
    /// Differing verdicts would indicate an edge case (off-by-one,
    /// halfmove boundary mismatch) between the two inline variants.
    #[test]
    fn fuzz_repetition_detectors_agree() {
        use crate::movegen::generate_legal_moves;
        init();

        const FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "4k3/PPPPPPPP/8/8/8/8/pppppppp/4K3 w - - 0 1",
            // Promotions + irreversible moves
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        const PLIES: usize = 120; // Longer games increase rep chances
        const GAMES: usize = 10;

        for (fen_idx, fen) in FENS.iter().enumerate() {
            for game in 0..GAMES {
                let seed: u32 = 0xDEADF00Du32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::from_fen(fen);
                for ply in 0..PLIES {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mv = legal.moves[(next_u32(&mut rng) as usize) % legal.len];
                    board.make_move(mv);

                    let main = check_repetition_main(&board);
                    let qs = check_repetition_qs(&board);
                    if main != qs {
                        panic!(
                            "rep detector disagreement: fen_idx={} game={} ply={} seed={:#x}\n\
                             main={} qs={}  halfmove={} stack_len={}\nfen={}",
                            fen_idx, game, ply, seed,
                            main, qs, board.halfmove, board.undo_stack.len(),
                            board.to_fen(),
                        );
                    }
                }
            }
        }
    }
}
