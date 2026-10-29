//! Board representation using bitboards + mailbox.
//!
//! Bitboards: pieces[6] (by type) + colors[2] (by color).
//! Mailbox: mailbox[64] for O(1) piece-at-square lookup.

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
    pub halfmove: u32,
    pub plies_from_null: u32,
    pub hash: u64,
    pub pawn_hash: u64,
    pub non_pawn_key: [u64; 2],
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
    pub halfmove: u32,       // halfmove clock for 50-move rule
    pub fullmove: u16,       // fullmove number
    pub plies_from_null: u32, // plies since last null move (for cuckoo cycle detection)

    pub hash: u64,           // Zobrist hash
    pub pawn_hash: u64,      // Zobrist hash of pawns only (for correction history)
    pub non_pawn_key: [u64; 2], // per-color non-pawn, non-king piece Zobrist keys
    pub undo_stack: Vec<UndoInfo>,
    /// Threat deltas accumulated during make_move (cleared on each make_move).
    /// Used by the NNUE threat accumulator for incremental updates.
    pub threat_deltas: Vec<crate::threats::RawThreatDelta>,
    /// Pawn-pair deltas for the last move. Separate from `threat_deltas`
    /// because `RawThreatDelta` is attacker/victim-shaped and a pawn-pair
    /// change does not fit it without a kind-bit in a hot struct.
    pub pawn_pair_deltas: Vec<crate::pawn_pair::PawnPairDelta>,
    /// Whether to generate threat deltas during make_move (set when threat net is loaded).
    pub generate_threat_deltas: bool,
    /// Whether to generate pawn-pair deltas (net has a pawn-pair block).
    pub generate_pawn_pair_deltas: bool,
    // Castling geometry: fixed for a game, read only when castling (rights
    // update, castle generation/validation), so kept clear of the hot fields.
    /// Starting square of the castling rook for each right, indexed by the
    /// right's bit position (0=WK, 1=WQ, 2=BK, 3=BQ). Fixed for a game: set at
    /// FEN parse, never changed by make/unmake. Standard chess: h1, a1, h8, a8.
    /// Chess960 puts the rooks (and king) elsewhere on the back rank.
    pub castle_rooks: [u8; 4],
    /// Castling-rights mask per square: `castling &= mask[from] & mask[to]`
    /// on every move. Derived from the king's and castling rooks' start
    /// squares at FEN parse (see `build_castle_mask`).
    pub castle_mask: [u8; 64],
    /// Per right (same indexing as `castle_rooks`): the squares that must be
    /// empty to castle — everything the king and rook cross or land on,
    /// minus the castling king's and rook's own start squares.
    pub castle_path: [Bitboard; 4],
    /// Per right: the squares that must not be attacked — the king's start,
    /// transit and destination squares.
    pub castle_king_path: [Bitboard; 4],
    /// True when every castling rook stands on its standard a/h-file square
    /// (set at FEN parse, fixed for the game): the fast path in
    /// `castle_hash`, where no right can carry a rook-identity key, so every
    /// standard-chess hash is exactly the four right keys.
    pub std_castling: bool,
}

/// Standard-chess castling rook start squares, indexed by right bit position
/// (WK, WQ, BK, BQ): h1, a1, h8, a8.
pub const STANDARD_CASTLE_ROOKS: [u8; 4] = [7, 0, 63, 56];

/// Index (0..4) of the castling right for `us` castling to `king_to`: the
/// king always lands on the g-file when castling kingside and on the c-file
/// when castling queenside, in standard chess and Chess960 alike.
#[inline(always)]
pub fn castle_index(us: Color, king_to: u8) -> usize {
    (us as usize) * 2 + (file_of(king_to) != 6) as usize
}

/// (rook_from, rook_to) for `us` castling with the king landing on
/// `king_to`. The rook always ends on the f-file (kingside) or d-file
/// (queenside); where it starts comes from the per-game `castle_rooks`.
#[inline(always)]
pub fn castle_rook_squares(castle_rooks: &[u8; 4], us: Color, king_to: u8) -> (u8, u8) {
    let kingside = file_of(king_to) == 6;
    let back_rank = if us == WHITE { 0u8 } else { 56u8 };
    let rook_to = back_rank + if kingside { 5 } else { 3 };
    (castle_rooks[castle_index(us, king_to)], rook_to)
}

/// True when a castling move can be applied as "move king, then move rook"
/// without either step touching an occupied square or moving a piece onto
/// itself. Always true in standard chess; false only for some 960 setups.
#[inline(always)]
pub fn castle_is_simple(king_from: u8, king_to: u8, rook_from: u8, rook_to: u8) -> bool {
    king_from != king_to && king_to != rook_from && rook_from != rook_to
}

/// Squares strictly between two squares of the same rank. Used for the
/// castling paths instead of `bitboard::between` so a fresh `Board::new()`
/// can build them without the line tables having been initialised.
fn rank_between(a: u8, b: u8) -> Bitboard {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if hi <= lo + 1 { return 0; }
    // Bits lo+1 ..= hi-1.
    ((1u64 << hi) - 1) & !((1u64 << (lo + 1)) - 1)
}

/// Per-square castling-rights mask built from the kings' and castling rooks'
/// start squares. A move from or to a king's start square drops both of that
/// colour's rights; from or to a castling rook's start square drops that one
/// right (the rook moved, or was captured at home).
pub fn build_castle_mask(king_start: [u8; 2], castle_rooks: &[u8; 4]) -> [u8; 64] {
    let mut m = [0xFFu8; 64];
    for c in 0..2 {
        m[king_start[c] as usize] &= !(0b11u8 << (2 * c));
    }
    for (i, &sq) in castle_rooks.iter().enumerate() {
        m[sq as usize] &= !(1u8 << i);
    }
    m
}

/// The standard-chess mask: kings on e1/e8, rooks on a/h.
pub fn standard_castle_mask() -> [u8; 64] {
    build_castle_mask([4, 60], &STANDARD_CASTLE_ROOKS)
}

/// Returns true if `capturing_side` has a LEGAL en-passant capture onto
/// `ep_sq`: one of its pawns stands file-adjacent to the just-double-pushed
/// enemy pawn, and making that capture does not leave its own king attacked.
///
/// The double-pushed pawn's square is implied by `ep_sq`:
///   - if capturing_side == BLACK, white just pushed; pushed pawn = ep_sq + 8
///   - if capturing_side == WHITE, black just pushed; pushed pawn = ep_sq - 8
///
/// The Zobrist hash must XOR `ep_key(file_of(ep_sq))` iff this returns true,
/// and every hash path (`compute_hash`, `make_move`, `make_null_move`) calls
/// this one function so they cannot disagree. Testing only adjacency is not
/// enough: a pinned capturer — the ordinary pin, or the rank case where the
/// capturing and captured pawns both leave the rank and expose the king to a
/// rook/queen — has no EP right, so the position is the same as without the
/// EP square and must hash the same, or a genuine threefold is missed.
///
/// Legality is decided by replaying the capture on a scratch occupancy: the
/// capturer leaves its square, the pushed pawn disappears, `ep_sq` fills, and
/// the king must then be attacked by nothing. A kingless (test) position
/// falls back to plain adjacency.
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
    let them = flip_color(capturing_side);
    if pieces[PAWN as usize] & colors[them as usize] & pushed_bb == 0 { return false; }
    // Enemy (capturing_side) pawns that could EP-capture sit on files adjacent
    // to pushed_sq on the same rank.
    let adj = ((pushed_bb << 1) & NOT_FILE_A) | ((pushed_bb >> 1) & NOT_FILE_H);
    let mut capturers = adj & pieces[PAWN as usize] & colors[capturing_side as usize];
    if capturers == 0 { return false; }

    let kings = pieces[KING as usize] & colors[capturing_side as usize];
    if kings == 0 { return true; }
    let ksq = lsb(kings);
    // The pushed pawn is captured, so it never counts as an attacker.
    let theirs = colors[them as usize] & !pushed_bb;
    // A knight/pawn/king attack on the king survives any EP capture (the
    // capture neither removes nor blocks it).
    let leapers = (knight_attacks(ksq) & pieces[KNIGHT as usize])
        | (pawn_attacks(capturing_side, ksq) & pieces[PAWN as usize])
        | (king_attacks(ksq) & pieces[KING as usize]);
    if leapers & theirs != 0 { return false; }
    let diag = (pieces[BISHOP as usize] | pieces[QUEEN as usize]) & theirs;
    let orth = (pieces[ROOK as usize] | pieces[QUEEN as usize]) & theirs;
    let occ = colors[0] | colors[1];
    while capturers != 0 {
        let from = pop_lsb(&mut capturers);
        let occ_after = (occ ^ (1u64 << from) ^ pushed_bb) | (1u64 << ep_sq);
        if bishop_attacks(ksq, occ_after) & diag == 0 && rook_attacks(ksq, occ_after) & orth == 0 {
            return true;
        }
    }
    false
}

impl Board {
    pub fn new() -> Self {
        let mut b = Board {
            pieces: [0; 6],
            colors: [0; 2],
            mailbox: [NO_PIECE_TYPE; 64],
            side_to_move: WHITE,
            castling: 0,
            castle_rooks: STANDARD_CASTLE_ROOKS,
            castle_mask: standard_castle_mask(),
            castle_path: [0; 4],
            castle_king_path: [0; 4],
            std_castling: true,
            ep_square: NO_SQUARE,
            halfmove: 0,
            fullmove: 1,
            plies_from_null: 0,
            hash: 0,
            pawn_hash: 0,
            non_pawn_key: [0; 2],
            undo_stack: Vec::with_capacity(512),
            threat_deltas: Vec::with_capacity(128),
            pawn_pair_deltas: Vec::with_capacity(crate::pawn_pair::MAX_PAWN_PAIR_DELTAS),
            generate_threat_deltas: false,
            generate_pawn_pair_deltas: false,
        };
        b.rebuild_castle_paths([4, 60]);
        b
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
        }
    }

    /// Move a piece on the board with hash update.
    /// Approximate Zobrist key AFTER `mv`, computed WITHOUT mutating the board.
    /// **For TT prefetch only** — a wrong key merely wastes a fetch; the real
    /// probe/store after `make_move` always uses the true `self.hash`, so this
    /// is never a correctness path. Exact for the common cases (quiet move,
    /// capture, en passant, promotion + side flip); castling (moves a rook too)
    /// and ep-square / castling-rights key deltas are skipped as rare — those
    /// only cost an occasional wasted prefetch. Models a `key_after`
    /// (pre-make prefetch).
    #[inline]
    pub fn key_after(&self, mv: Move) -> u64 {
        let from = move_from(mv);
        let to = move_to(mv);
        let flags = move_flags(mv);
        let moved = self.piece_at(from);
        if moved == NO_PIECE || flags == FLAG_CASTLE {
            return self.hash;
        }
        let us = self.side_to_move;
        let mut k = self.hash ^ side_key() ^ piece_key(moved, from);
        if flags == FLAG_EN_PASSANT {
            let cap_sq = if us == WHITE { to - 8 } else { to + 8 };
            k ^= piece_key(make_piece(1 - us, PAWN), cap_sq);
        } else {
            let captured = self.piece_at(to);
            if captured != NO_PIECE {
                k ^= piece_key(captured, to);
            }
        }
        let dest = if is_promotion(mv) {
            make_piece(us, promotion_piece_type(mv))
        } else {
            moved
        };
        k ^ piece_key(dest, to)
    }

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
        }
    }

    /// FIDE Art 5.2: insufficient material to deliver checkmate, by either side.
    /// Returns true for:
    ///   - K vs K
    ///   - K + single minor (N or B) vs K
    ///   - K + minor vs K + minor (any pair of single minors per side — neither
    ///     side can force mate; technically FIDE only mandates draw on agreement
    ///     here, but engines uniformly treat as drawn since no forced win exists)
    ///   - K + B vs K + B with all bishops on the same color (no bishop
    ///     of opposite color exists, so neither side can mate)
    ///
    /// Excluded (deliberately):
    ///   - KNN vs K (technically drawn FIDE-wise but extremely rare; opponent
    ///     can blunder into mate). Excluded so a cooperative mate stays
    ///     reachable. NOTE (measured 2026-09-10 against Syzygy WDL, 4,480
    ///     positions over 140 material classes): eval/search do NOT reach 0
    ///     here — the raw net prices KNNvK at a median |static| of ~3 pawns
    ///     (the single outlier among 13 always-drawn classes, the rest within
    ///     0.11), and the search sits at ~1.15 pawns from 200k to 4M nodes
    ///     without converging. It is a uniform bias inside the class, not a
    ///     decision-changing one: from 60 tablebase-won KNNvKP positions our
    ///     best move preserved the win 60/60 (we over-rate KNNvKP by more than
    ///     KNNvK, so the ordering survives). Kept excluded; do not cite this
    ///     comment as evidence the eval handles it.
    ///   - KBN vs K (winnable for the BN side; not a draw).
    ///
    /// Follows the same standard `draw_by_material` /
    /// `has_insufficient_material` rule. The rule is consensus among engines that
    /// implement IM-in-search (~half of the top engines do).
    #[inline]
    pub fn is_insufficient_material(&self) -> bool {
        // Any pawn / rook / queen → sufficient material to mate.
        if self.pieces[PAWN as usize]
         | self.pieces[ROOK as usize]
         | self.pieces[QUEEN as usize] != 0 {
            return false;
        }

        let piece_count = crate::bitboard::popcount(self.occupied());
        // K vs K, K+minor vs K — drawn.
        if piece_count <= 3 {
            return true;
        }
        // More than 4 pieces (≥3 minors total) — too rich to call draw.
        if piece_count != 4 {
            return false;
        }

        // 4 pieces total = 2 kings + 2 minors. Layouts:
        //   (a) 1 minor each side  → drawn (B vs B, B vs N, N vs N, KBN is excluded)
        //   (b) 2 minors on one side, 0 on other:
        //         - both knights  → NOT drawn (KNN vs K — see exclusion note above)
        //         - B + N         → NOT drawn (KBN vs K — winnable)
        //         - 2 bishops:
        //             - same color → drawn
        //             - opposite color → NOT drawn (mate possible)
        let white_minors = crate::bitboard::popcount(
            self.colors[WHITE as usize] & (self.pieces[BISHOP as usize] | self.pieces[KNIGHT as usize])
        );
        let black_minors = crate::bitboard::popcount(
            self.colors[BLACK as usize] & (self.pieces[BISHOP as usize] | self.pieces[KNIGHT as usize])
        );

        // (a) 1 minor each
        if white_minors == 1 && black_minors == 1 {
            return true;
        }

        // (b) 2 minors on one side. Must be 2 bishops same color to be drawn.
        if self.pieces[KNIGHT as usize] != 0 {
            return false;  // any knight present → not the same-color-bishops case
        }
        // All 2 minors are bishops. Drawn iff all on the same color squares.
        // Light squares: 0x55AA55AA55AA55AA (a1, c1, ... pattern).
        let light_mask: u64 = 0x55AA_55AA_55AA_55AA;
        let bishops = self.pieces[BISHOP as usize];
        let bishops_light = crate::bitboard::popcount(bishops & light_mask);
        // 0 → both dark, 2 → both light, 1 → mixed (not drawn).
        bishops_light != 1
    }

    /// True if the current position is repetition-drawn from a search node at
    /// `ply`.
    ///
    /// A single previous occurrence only counts when it is strictly inside the
    /// current search tree. If the occurrence is at/before the root, require an
    /// older occurrence as well, i.e. a true threefold in the game history.
    #[inline]
    pub fn is_repetition_draw(&self, ply: i32) -> bool {
        let stack_len = self.undo_stack.len();
        let scan_limit = (self.halfmove as usize).min(self.plies_from_null as usize);
        let limit = scan_limit.min(stack_len);
        let mut i = 4usize;
        while i <= limit {
            if self.undo_stack[stack_len - i].hash == self.hash {
                if (i as i32) < ply {
                    return true;
                }
                let mut k = i + 2;
                while k <= limit {
                    if self.undo_stack[stack_len - k].hash == self.hash {
                        return true;
                    }
                    k += 2;
                }
                return false;
            }
            i += 2;
        }
        false
    }

    /// Zobrist contribution of the castling rights `rights`: the per-right
    /// keys, plus a key naming the rook of each held right whose rook is NOT
    /// on its standard a/h-file square (only possible in Chess960). Without
    /// the rook term, 960 positions whose rights name different rooks on the
    /// same side of the king (Shredder `F` vs `H` with rooks on f1 and h1)
    /// hash identically despite having different legal moves.
    ///
    /// Deciding per right rather than per board keeps the hash a function of
    /// the position alone: a 960 game whose remaining rights happen to use
    /// a/h rooks hashes the same as that position parsed from its FEN, and
    /// standard chess never sees a rook key. Every hash path goes through
    /// here, so the incremental and from-scratch hashes cannot disagree.
    #[inline(always)]
    pub fn castle_hash(&self, rights: u8) -> u64 {
        let mut h = castle_key(rights);
        if !self.std_castling {
            for i in 0..4 {
                let rook = self.castle_rooks[i];
                if rights & (1u8 << i) != 0 && rook != STANDARD_CASTLE_ROOKS[i] {
                    h ^= castle_rook_key(i, rook);
                }
            }
        }
        h
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

        // Only hash ep_square if the side to move can legally make the EP capture.
        // Without this guard, the same physical position reached with or without
        // a recent double-push would hash differently, breaking rep detection
        // (Stockfish pattern).
        if self.ep_square != NO_SQUARE
            && ep_capture_available(&self.pieces, &self.colors, self.side_to_move, self.ep_square)
        {
            h ^= ep_key(file_of(self.ep_square));
        }

        h ^= self.castle_hash(self.castling);

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
                    // Malformed FEN (too many files on a rank, or too many
                    // ranks) can drive file/rank out of [0,7]; placing at the
                    // resulting square would index mailbox/bitboards OOB and
                    // panic in release. Skip out-of-board glyphs instead.
                    if (0..8).contains(&rank) && (0..8).contains(&file) {
                        let sq = (rank * 8 + file) as u8;
                        self.put_piece_no_hash(color, pt, sq);
                    }
                    file += 1;
                }
            }
        }

        // Side to move
        self.side_to_move = if parts.len() > 1 && parts[1] == "b" { BLACK } else { WHITE };

        // Castling. Accepts standard/X-FEN letters (K/Q/k/q = the outermost
        // rook on that side of the king) and Shredder-FEN file letters
        // (A-H / a-h = the rook on that file; the side follows from which side
        // of the king it stands on), so Chess960 positions parse too.
        //
        // A right is kept only if the king stands on its back rank and a rook
        // of the same colour actually stands on the named square. A FEN may
        // declare rights for a rook that doesn't exist (corrupt input); like
        // the old e1/a1/h1-only check, this makes "right set => rook present
        // on its castling square" an invariant that movegen and
        // is_pseudo_legal can rely on.
        self.castling = 0;
        self.castle_rooks = STANDARD_CASTLE_ROOKS;
        if parts.len() > 2 {
            for ch in parts[2].chars() {
                if !ch.is_ascii_alphabetic() { continue; }
                let color = if ch.is_ascii_uppercase() { WHITE } else { BLACK };
                let back_rank: u8 = if color == WHITE { 0 } else { 56 };
                let rank_bb = 0xFFu64 << back_rank;
                let kings = self.pieces[KING as usize] & self.colors[color as usize] & rank_bb;
                if kings == 0 || kings & (kings - 1) != 0 { continue; }
                let ksq = lsb(kings) as u8;
                let kfile = file_of(ksq);
                let rooks = self.pieces[ROOK as usize] & self.colors[color as usize] & rank_bb;
                // Rook squares east (higher file) / west of the king on the back rank.
                let below = (1u64 << ksq) - 1;
                let east = rooks & !(below | (1u64 << ksq));
                let west = rooks & below;
                let rook_sq = match ch.to_ascii_lowercase() {
                    'k' => if east != 0 { 63 - east.leading_zeros() as u8 } else { continue },
                    'q' => if west != 0 { lsb(west) as u8 } else { continue },
                    f @ 'a'..='h' => {
                        let sq = back_rank + (f as u8 - b'a');
                        if rooks & (1u64 << sq) == 0 || file_of(sq) == kfile { continue; }
                        sq
                    }
                    _ => continue,
                };
                let kingside = file_of(rook_sq) > kfile;
                let idx = (color as usize) * 2 + (!kingside) as usize;
                self.castling |= 1u8 << idx;
                self.castle_rooks[idx] = rook_sq;
            }
        }
        // Mask and paths from the actual king square of each colour that
        // holds a right; a colour with no right keeps e1/e8 (its entries are
        // never consulted). For standard positions this reproduces the old
        // fixed e1/e8/a1/h1/a8/h8 table exactly.
        let mut king_start = [4u8, 60u8];
        for c in 0..2u8 {
            if self.castling & (0b11u8 << (2 * c)) != 0 {
                king_start[c as usize] = self.king_sq(c);
            }
        }
        self.castle_mask = build_castle_mask(king_start, &self.castle_rooks);
        self.rebuild_castle_paths(king_start);
        self.std_castling = self.castle_rooks == STANDARD_CASTLE_ROOKS;

        // En passant. Bounds-check f and r to avoid wrapping into a
        // huge ep_square that later overflows `1u64 << ep_square`
        // (board.rs / movegen.rs). Then keep the square only if it could
        // be the result of the opponent's last move being a double push:
        // on the 6th rank with White to move (3rd with Black), the target
        // and the pushed pawn's origin both empty, and the pusher's pawn
        // standing just past the target. Anything else is corrupt input
        // and is dropped, so movegen, is_pseudo_legal and make_move never
        // see an EP square they would have to second-guess.
        self.ep_square = NO_SQUARE;
        if parts.len() > 3 && parts[3] != "-" {
            let bytes = parts[3].as_bytes();
            if bytes.len() == 2 {
                let f = bytes[0].wrapping_sub(b'a');
                let r = bytes[1].wrapping_sub(b'1');
                if f < 8 && r < 8 {
                    let ep = square(f, r);
                    let stm = self.side_to_move;
                    let target_rank = if stm == WHITE { 5 } else { 2 };
                    if r == target_rank {
                        let (pushed, origin) = if stm == WHITE { (ep - 8, ep + 8) } else { (ep + 8, ep - 8) };
                        let occ = self.colors[0] | self.colors[1];
                        let pusher_pawns = self.pieces[PAWN as usize] & self.colors[flip_color(stm) as usize];
                        if occ & ((1u64 << ep) | (1u64 << origin)) == 0
                            && pusher_pawns & (1u64 << pushed) != 0
                        {
                            self.ep_square = ep;
                        }
                    }
                }
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
        // Compute per-color non-pawn key
        self.non_pawn_key = [0; 2];
        for color in 0..2u8 {
            for pt in [KNIGHT, BISHOP, ROOK, QUEEN] {
                let mut bb = self.pieces[pt as usize] & self.colors[color as usize];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb) as u8;
                    let k = piece_key(make_piece(color, pt), sq);
                    self.non_pawn_key[color as usize] ^= k;
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
        // Setwise attack aggregation. Replaces the per-square magic-lookup
        // loop that used to live here. See `src/setwise.rs`.
        // Verified bit-equivalent to the per-
        // square version via `setwise::tests::*_matches_per_square` and via
        // perft regression on standard positions.
        let c = color as usize;
        let occ = self.colors[0] | self.colors[1];
        let pieces = |pt: u8| self.pieces[pt as usize] & self.colors[c];

        let pawns = pieces(PAWN);
        let knights = pieces(KNIGHT);
        let diag = pieces(BISHOP) | pieces(QUEEN);
        let orth = pieces(ROOK) | pieces(QUEEN);
        let king = pieces(KING);

        let mut attacks: Bitboard = 0;
        attacks |= crate::setwise::pawn_attacks_setwise(pawns, color);
        attacks |= crate::setwise::knight_attacks_setwise(knights);
        attacks |= crate::setwise::bishop_attacks_setwise(diag, occ);
        attacks |= crate::setwise::rook_attacks_setwise(orth, occ);
        if king != 0 {
            attacks |= king_attacks(king.trailing_zeros());
        }
        attacks
    }

    /// Pre-move check detection for pruning carve-outs.
    /// Covers direct, promotion, discovered, castling-rook, and EP-discovered
    /// checks using the post-move occupancy.
    #[inline]
    pub fn gives_direct_check(&self, mv: Move) -> bool {
        let from = move_from(mv);
        let to = move_to(mv);
        let pt = self.piece_type_at(from);
        if pt == NO_PIECE_TYPE {
            return false;
        }
        let us = self.side_to_move;
        let them = flip_color(us);
        let their_king = self.king_sq(them);
        let their_king_bb = 1u64 << their_king;
        let flags = move_flags(mv);

        let from_bb = 1u64 << from;
        let to_bb = 1u64 << to;
        let mut occ = self.occupied() ^ from_bb;
        let mut our_diag = (self.pieces[BISHOP as usize] | self.pieces[QUEEN as usize])
            & self.colors[us as usize] & !from_bb;
        let mut our_orth = (self.pieces[ROOK as usize] | self.pieces[QUEEN as usize])
            & self.colors[us as usize] & !from_bb;

        if flags == FLAG_CASTLE {
            // Lift the rook before placing king and rook: in Chess960 the
            // king can land on the rook's start square (and either piece may
            // not move at all), so the order matters.
            let (rook_from, rook_to) = self.castle_rook_squares(us, to);
            occ = (occ & !(1u64 << rook_from)) | to_bb | (1u64 << rook_to);
            our_orth = (our_orth & !(1u64 << rook_from)) | (1u64 << rook_to);
        } else {
            if flags == FLAG_EN_PASSANT {
                let captured_sq = if us == WHITE { to.wrapping_sub(8) } else { to.wrapping_add(8) };
                if captured_sq >= 64 {
                    return false;
                }
                occ ^= 1u64 << captured_sq;
            }
            occ |= to_bb;
        }

        let effective_pt = if is_promotion(mv) {
            promotion_piece_type(mv)
        } else {
            pt
        };

        match effective_pt {
            PAWN => {
                if pawn_attacks(us, to as u32) & their_king_bb != 0 {
                    return true;
                }
            }
            KNIGHT => {
                if knight_attacks(to as u32) & their_king_bb != 0 {
                    return true;
                }
            }
            BISHOP => our_diag |= to_bb,
            ROOK => our_orth |= to_bb,
            QUEEN => {
                our_diag |= to_bb;
                our_orth |= to_bb;
            }
            _ => {}
        }

        let needs_slider_check = matches!(effective_pt, BISHOP | ROOK | QUEEN)
            || flags == FLAG_CASTLE
            || flags == FLAG_EN_PASSANT
            || line(from as u32, their_king as u32) != 0;

        needs_slider_check
            && (bishop_attacks(their_king as u32, occ) & our_diag != 0
                || rook_attacks(their_king as u32, occ) & our_orth != 0)
    }

    /// Get bitboard of all pieces attacking a square.
    #[inline]
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
        self.sole_slider_blockers(self.king_sq(us) as u32, flip_color(us), us)
    }

    /// Discovered-check candidates: side-to-move pieces that are the only
    /// piece between one of our own sliders and the enemy king. Moving one
    /// off that line uncovers check. The same sole-blocker scan as
    /// `pinned`, aimed at the other king.
    pub fn discovered_check_candidates(&self) -> Bitboard {
        let us = self.side_to_move;
        self.sole_slider_blockers(self.king_sq(flip_color(us)) as u32, us, us)
    }

    /// Pieces of `blocker_color` that are the sole occupant of the line
    /// between the king on `ksq` and an aligned slider of `slider_color`.
    fn sole_slider_blockers(&self, ksq: u32, slider_color: Color, blocker_color: Color) -> Bitboard {
        let occ = self.occupied();

        let diag_sliders = (self.pieces[BISHOP as usize] | self.pieces[QUEEN as usize]) & self.colors[slider_color as usize];
        let orth_sliders = (self.pieces[ROOK as usize] | self.pieces[QUEEN as usize]) & self.colors[slider_color as usize];

        let mut result = 0u64;

        // Bishop/queen lines, then rook/queen lines.
        for mut potential in [bishop_attacks(ksq, 0) & diag_sliders, rook_attacks(ksq, 0) & orth_sliders] {
            while potential != 0 {
                let slider_sq = pop_lsb(&mut potential);
                let blockers = between(ksq, slider_sq) & occ;
                if popcount(blockers) == 1 && blockers & self.colors[blocker_color as usize] != 0 {
                    result |= blockers;
                }
            }
        }

        result
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
            // Malformed EP (e.g. TT collision delivers EP with `to` on rank
            // 0/1 so captured_sq wraps): contract of is_legal is "true iff
            // legal", so report FALSE rather than allowing it through.
            // make_move still rejects defensively if this ever slips by.
            if captured_sq >= 64 { return false; }
            if checkers != 0 {
                // Double check: only king moves resolve (EP is never a king move)
                if checkers & (checkers - 1) != 0 { return false; }
                let checker_sq = crate::bitboard::lsb(checkers);
                let check_mask = (1u64 << checker_sq) | between(checker_sq, ksq);
                if checker_sq != captured_sq && (check_mask & (1u64 << to)) == 0 {
                    return false;
                }
            }
            let occ = (self.occupied() ^ (1u64 << from) ^ (1u64 << captured_sq)) | (1u64 << to);
            let their_bishops = (self.pieces[BISHOP as usize] | self.pieces[QUEEN as usize])
                & self.colors[flip_color(us) as usize];
            let their_rooks = (self.pieces[ROOK as usize] | self.pieces[QUEEN as usize])
                & self.colors[flip_color(us) as usize];
            return bishop_attacks(ksq, occ) & their_bishops == 0
                && rook_attacks(ksq, occ) & their_rooks == 0;
        }

        // Castling: the full check, including the transit squares and the
        // Chess960 case where the castling rook itself shields the king's
        // path from an enemy slider on the back rank.
        if flags == FLAG_CASTLE {
            return self.castle_is_legal(us, from as u8, to as u8);
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

    /// `is_legal` for a side to move that is NOT in check, reusing the
    /// enemy-attack map the caller already holds (`attacks_by_color(them)`,
    /// computed with the full occupancy). Assumes `mv` is pseudo-legal.
    ///
    /// Why the map is exact for king moves here: `is_legal` tests the king's
    /// destination with the king lifted from the occupancy, so that a king
    /// stepping back along a checking slider's ray is caught. Lifting the king
    /// can only extend a slider ray that currently ENDS on the king, i.e. a
    /// slider giving check. With no checkers, every enemy attack set is the
    /// same with or without our king on the board, so `to` is attacked after
    /// the move iff it is in `enemy_attacks` now. Leaper attacks never depend
    /// on occupancy. The move-loop in `negamax` already holds this map for
    /// history indexing, so the king test costs one AND instead of a full
    /// `attackers_to` (two slider lookups plus four leaper tables).
    ///
    /// Everything else mirrors `is_legal` with `checkers == 0`: en passant
    /// and castling keep their full special-case tests, and a pinned piece may
    /// only move along its pin line.
    #[inline(always)]
    pub fn is_legal_unchecked_side(&self, mv: Move, pinned: Bitboard, enemy_attacks: Bitboard) -> bool {
        let from = move_from(mv) as u32;
        let to = move_to(mv) as u32;
        let flags = move_flags(mv);
        if flags == FLAG_EN_PASSANT || flags == FLAG_CASTLE {
            return self.is_legal(mv, pinned, 0);
        }
        let king_bb = self.pieces[KING as usize] & self.colors[self.side_to_move as usize];
        if king_bb & (1u64 << from) != 0 {
            return enemy_attacks & (1u64 << to) == 0;
        }
        if pinned & (1u64 << from) != 0 {
            return line(from, self.king_sq(self.side_to_move) as u32) & (1u64 << to) != 0;
        }
        true
    }

    /// Precompute `castle_path` / `castle_king_path` from the kings' start
    /// squares and `castle_rooks` (fixed for the game, so this runs once per
    /// FEN). Entries for rights not held are filled but never consulted.
    fn rebuild_castle_paths(&mut self, king_start: [u8; 2]) {
        for idx in 0..4 {
            let us = (idx / 2) as Color;
            let ksq = king_start[us as usize];
            let back_rank: u8 = if us == WHITE { 0 } else { 56 };
            let king_to = back_rank + if idx % 2 == 0 { 6 } else { 2 };
            let (rook_from, rook_to) = castle_rook_squares(&self.castle_rooks, us, king_to);
            let king_path = rank_between(ksq, king_to) | (1u64 << king_to);
            let rook_path = rank_between(rook_from, rook_to) | (1u64 << rook_to);
            self.castle_path[idx] = (king_path | rook_path) & !(1u64 << ksq) & !(1u64 << rook_from);
            self.castle_king_path[idx] = king_path | (1u64 << ksq);
        }
    }

    /// Full legality of castling for `us`, king from `king_from` to `king_to`
    /// (the g- or c-file of `us`'s back rank). Standard chess and Chess960:
    /// - the right is held, our king is on `king_from` and our rook on the
    ///   right's castling-rook square (a held right means neither has moved,
    ///   so `king_from` is the king's start square the paths were built for);
    /// - every square the king or the rook crosses or lands on is empty,
    ///   ignoring the castling king and rook themselves (in 960 they can
    ///   land on each other's start squares, or not move at all);
    /// - no square from the king's start to its destination inclusive is
    ///   attacked. Attacks are computed with BOTH the king and the castling
    ///   rook lifted: in 960 the rook can stand between an enemy rook/queen
    ///   and the king's path, a shield that is gone once it has castled.
    ///   Lifting it is never too strict: an attacker it was shielding sits
    ///   on the back rank beyond it, so the king's path is attacked after
    ///   castling too. For standard positions this gives exactly the old
    ///   e1/f1/g1 (e1/d1/c1) test.
    #[inline]
    pub fn castle_is_legal(&self, us: Color, king_from: u8, king_to: u8) -> bool {
        let back_rank: u8 = if us == WHITE { 0 } else { 56 };
        if king_to != back_rank + 6 && king_to != back_rank + 2 { return false; }
        if king_from >= 64 { return false; }
        let idx = castle_index(us, king_to);
        if self.castling & (1u8 << idx) == 0 { return false; }
        let king_bb = 1u64 << king_from;
        let ours = self.colors[us as usize];
        if king_bb & self.pieces[KING as usize] & ours == 0 { return false; }
        let rook_bb = 1u64 << self.castle_rooks[idx];
        if rook_bb & self.pieces[ROOK as usize] & ours == 0 { return false; }

        let occ = self.occupied() ^ king_bb ^ rook_bb;
        if occ & self.castle_path[idx] != 0 { return false; }

        let them_bb = self.colors[flip_color(us) as usize];
        let mut squares = self.castle_king_path[idx];
        while squares != 0 {
            let sq = pop_lsb(&mut squares);
            if self.attacked_with_occ(sq, them_bb, occ) { return false; }
        }
        true
    }

    /// Is `sq` attacked by any piece in `attackers_bb`, given occupancy `occ`?
    /// Early-exit form of `attackers_to(sq, occ) & attackers_bb != 0`.
    #[inline]
    fn attacked_with_occ(&self, sq: u32, attackers_bb: Bitboard, occ: Bitboard) -> bool {
        let p = &self.pieces;
        // Attacked by a white pawn iff a black pawn on `sq` would attack it.
        let pawn_from = if attackers_bb & self.colors[WHITE as usize] != 0 { BLACK } else { WHITE };
        knight_attacks(sq) & p[KNIGHT as usize] & attackers_bb != 0
            || pawn_attacks(pawn_from, sq) & p[PAWN as usize] & attackers_bb != 0
            || king_attacks(sq) & p[KING as usize] & attackers_bb != 0
            || bishop_attacks(sq, occ) & (p[BISHOP as usize] | p[QUEEN as usize]) & attackers_bb != 0
            || rook_attacks(sq, occ) & (p[ROOK as usize] | p[QUEEN as usize]) & attackers_bb != 0
    }

    /// Rook start/destination squares for a castling move (king to `king_to`).
    #[inline]
    pub fn castle_rook_squares(&self, us: Color, king_to: u8) -> (u8, u8) {
        castle_rook_squares(&self.castle_rooks, us, king_to)
    }

    /// Piece type `mv` captures (NO_PIECE_TYPE if none). En passant captures
    /// a pawn off the destination square; castling never captures, even in
    /// Chess960 where the king's destination can hold its own castling rook.
    #[inline(always)]
    pub fn captured_piece_type(&self, mv: Move) -> u8 {
        let flags = move_flags(mv);
        if flags == FLAG_EN_PASSANT {
            PAWN
        } else if flags == FLAG_CASTLE {
            NO_PIECE_TYPE
        } else {
            self.piece_type_at(move_to(mv))
        }
    }

    /// Does `mv` capture (en passant included; castling never does)?
    #[inline(always)]
    pub fn is_capture(&self, mv: Move) -> bool {
        self.captured_piece_type(mv) != NO_PIECE_TYPE
    }

    /// UCI string for `mv`. With the UCI_Chess960 option on, castling is
    /// written king-from -> own castling rook ("king takes rook"), which is
    /// unambiguous in 960; otherwise the standard king-from -> king-to form.
    pub fn uci_move(&self, mv: Move) -> String {
        self.uci_move_mode(mv, crate::types::chess960())
    }

    /// `uci_move` with the Chess960 output convention given explicitly.
    pub fn uci_move_mode(&self, mv: Move, chess960: bool) -> String {
        if chess960 && mv != NO_MOVE && move_flags(mv) == FLAG_CASTLE {
            let from = move_from(mv);
            let us = if rank_of(from) == 0 { WHITE } else { BLACK };
            let (rook_from, _) = self.castle_rook_squares(us, move_to(mv));
            return format!("{}{}", square_name(from), square_name(rook_from));
        }
        move_to_uci(mv)
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

        // Save undo info. Castling never captures (in Chess960 the king's
        // destination may hold its own castling rook).
        let captured = self.captured_piece_type(mv);

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

        // Validate castling: the king is moving, our rook is on its castling
        // square, and the squares king and rook land on are free (apart from
        // the castling pair themselves, which can swap squares in 960).
        if flags == FLAG_CASTLE {
            let back_rank: u8 = if us == WHITE { 0 } else { 56 };
            if pt != KING || (to != back_rank + 6 && to != back_rank + 2) {
                return false;
            }
            let (rook_from, rook_to) = self.castle_rook_squares(us, to);
            if (1u64 << rook_from) & self.pieces[ROOK as usize] & self.colors[us as usize] == 0 {
                return false;
            }
            let others = self.occupied() & !(1u64 << from) & !(1u64 << rook_from);
            if others & ((1u64 << to) | (1u64 << rook_to)) != 0 {
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
        self.hash ^= self.castle_hash(self.castling);

        // Clear threat deltas for this move
        let gen_threats = self.generate_threat_deltas;
        if gen_threats { self.threat_deltas.clear(); }

        // Pawn-pair deltas are derived from the pawn set BEFORE any mutation,
        // so this must run ahead of the capture/move/promotion handling below.
        // (Threat deltas are interleaved with the mutations instead, because
        // each one needs the intermediate occupancy.)
        if self.generate_pawn_pair_deltas {
            crate::pawn_pair::push_pawn_pair_deltas(
                &mut self.pawn_pair_deltas, self.pieces[PAWN as usize], &self.colors,
                us, mv, captured, pt);
        }

        if flags == FLAG_CASTLE {
            self.make_castle(us, from, to, gen_threats);
        } else {
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
        }

        // Update castling rights (for any move from/to relevant squares)
        self.castling &= self.castle_mask[from as usize] & self.castle_mask[to as usize];

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
        self.hash ^= self.castle_hash(self.castling);

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

    /// Move king and rook for a castling move (king `from` -> `to`), with
    /// hash and threat-delta updates.
    ///
    /// Standard castling, and any 960 castling where king and rook move to
    /// squares the other doesn't occupy, moves the king and then the rook,
    /// exactly as before Chess960 support. Otherwise (the king lands on the
    /// rook's start square, or the king or the rook doesn't move) the rook is
    /// lifted first, then the king moves, then the rook is put down, so no
    /// step ever targets an occupied square or "moves" a piece onto itself.
    /// `threats::emit_post_move_deltas` + `unreplay_move_deltas` mirror this
    /// sequence exactly, backwards.
    fn make_castle(&mut self, us: Color, from: u8, to: u8, gen_threats: bool) {
        let (rook_from, rook_to) = self.castle_rook_squares(us, to);
        if castle_is_simple(from, to, rook_from, rook_to) {
            self.move_piece(us, KING, from, to);
            if gen_threats { crate::threats::push_threats_on_move(
                &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
                self.colors[0] | self.colors[1], us, KING, from as u32, to as u32); }
            self.move_piece(us, ROOK, rook_from, rook_to);
            if gen_threats { crate::threats::push_threats_on_move(
                &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
                self.colors[0] | self.colors[1], us, ROOK, rook_from as u32, rook_to as u32); }
            return;
        }
        self.remove_piece(us, ROOK, rook_from);
        if gen_threats { crate::threats::push_threats_on_change(
            &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
            self.colors[0] | self.colors[1], us, ROOK, rook_from as u32, false); }
        if from != to {
            self.move_piece(us, KING, from, to);
            if gen_threats { crate::threats::push_threats_on_move(
                &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
                self.colors[0] | self.colors[1], us, KING, from as u32, to as u32); }
        }
        self.put_piece(us, ROOK, rook_to);
        if gen_threats { crate::threats::push_threats_on_change(
            &mut self.threat_deltas, &self.pieces, &self.colors, &self.mailbox,
            self.colors[0] | self.colors[1], us, ROOK, rook_to as u32, true); }
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

        // Corruption guard: HOIST above the promotion undo so a corrupt
        // mailbox doesn't get phantom-bit XORed before bail-out. If `to`
        // is empty here, the bitboards/mailbox diverged in an earlier
        // make/unmake pair — never expected in correct code.
        //
        // Order matters: running promotion-undo's remove+put XOR FIRST can
        // set phantom QUEEN+PAWN bits at `to` when the queen wasn't
        // actually there, then "recover" castling/ep/hash from undo while
        // leaving pieces[], colors[] and mailbox[] inconsistent — turning
        // detected corruption into SILENT corruption, where the hash looks
        // clean but the bitboards don't.
        // debug_assert in dev/test surfaces the bug; release bails
        // without making state worse.
        if self.piece_type_at(to) == NO_PIECE_TYPE {
            debug_assert!(
                false,
                "unmake_move: piece missing at to={} for move {:?} — child search corrupted state",
                to, mv,
            );
            // Release-build bail. side_to_move + fullmove already updated
            // above so the parent's turn-tracking is consistent on the
            // corrupt state. Don't XOR phantom bits.
            if us == BLACK { self.fullmove -= 1; }
            return;
        }

        if flags == FLAG_CASTLE {
            // Lift both pieces before putting either back: in Chess960 the
            // king's and rook's squares can coincide or not change.
            let (rook_from, rook_to) = self.castle_rook_squares(us, to);
            self.remove_piece_no_hash(us, KING, to);
            self.remove_piece_no_hash(us, ROOK, rook_to);
            self.put_piece_no_hash(us, KING, from);
            self.put_piece_no_hash(us, ROOK, rook_from);
        } else {
            // Undo promotion
            if is_promotion(mv) {
                let promo_pt = promotion_piece_type(mv);
                self.remove_piece_no_hash(us, promo_pt, to);
                self.put_piece_no_hash(us, PAWN, to);
            }

            // Undo piece move
            let pt = self.piece_type_at(to);
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
        }

        // Restore state
        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove = undo.halfmove;
        self.plies_from_null = undo.plies_from_null;
        self.hash = undo.hash;
        self.pawn_hash = undo.pawn_hash;
        self.non_pawn_key = undo.non_pawn_key;

        if us == BLACK {
            self.fullmove -= 1;
        }
    }

    /// Make a null move (just flip side, update EP).
    pub fn make_null_move(&mut self) {
        self.threat_deltas.clear(); // null move = no piece changes = no threat deltas
        self.pawn_pair_deltas.clear(); // ... and no pawn-structure change either
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
            // K/Q/k/q when the castling rook is on its standard a/h file (so
            // standard positions print exactly as always); otherwise the
            // Shredder-FEN file letter of the castling rook (Chess960).
            let rights = [(CASTLE_WK, 'K', 7u8), (CASTLE_WQ, 'Q', 0), (CASTLE_BK, 'k', 7), (CASTLE_BQ, 'q', 0)];
            for (i, &(bit, letter, std_file)) in rights.iter().enumerate() {
                if self.castling & bit == 0 { continue; }
                let file = file_of(self.castle_rooks[i]);
                if file == std_file {
                    fen.push(letter);
                } else {
                    let c = (b'a' + file) as char;
                    fen.push(if i < 2 { c.to_ascii_uppercase() } else { c });
                }
            }
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
    fn test_insufficient_material() {
        init();
        // Drawn cases
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").is_insufficient_material(),
            "K vs K");
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/4KB2 w - - 0 1").is_insufficient_material(),
            "KB vs K");
        assert!(Board::from_fen("4k3/8/8/8/8/8/8/4KN2 w - - 0 1").is_insufficient_material(),
            "KN vs K");
        assert!(Board::from_fen("4kb2/8/8/8/8/8/8/4KB2 w - - 0 1").is_insufficient_material(),
            "KB vs KB same-color (both light)");
        assert!(Board::from_fen("4kn2/8/8/8/8/8/8/4KN2 w - - 0 1").is_insufficient_material(),
            "KN vs KN");
        assert!(Board::from_fen("4kb2/8/8/8/8/8/8/4KN2 w - - 0 1").is_insufficient_material(),
            "KN vs KB (each side 1 minor)");

        // NOT drawn cases — material sufficient to mate (or near it)
        assert!(!Board::from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1").is_insufficient_material(),
            "K+P vs K (pawn can promote)");
        assert!(!Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").is_insufficient_material(),
            "K+Q vs K");
        assert!(!Board::from_fen("4k3/8/8/8/8/8/8/3RK3 w - - 0 1").is_insufficient_material(),
            "K+R vs K");
        // KBN vs K is winnable → must NOT be detected as draw
        assert!(!Board::from_fen("4k3/8/8/8/8/8/8/2BNK3 w - - 0 1").is_insufficient_material(),
            "K+B+N vs K (winnable)");
        // KBB opposite-color vs K — can mate
        assert!(!Board::from_fen("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1").is_insufficient_material(),
            "KBB opposite-color vs K");
        // KNN vs K — excluded from draw detection
        assert!(!Board::from_fen("4k3/8/8/8/8/8/8/2N1KN2 w - - 0 1").is_insufficient_material(),
            "KNN vs K (excluded from insufficient-material detection)");

        // 5+ pieces — too rich to call IM regardless of types
        assert!(!Board::from_fen("4kb2/8/8/8/8/8/8/3BKB2 w - - 0 1").is_insufficient_material(),
            "3 bishops total");
    }

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
    fn discovered_check_detected_for_pruning_carveouts() {
        init();
        let b = Board::from_fen("k7/8/8/8/8/8/N7/R3K3 w - - 0 1");
        let nb4 = make_move(8, 25, FLAG_NONE); // Na2-b4 uncovers Ra1-a8+
        assert!(b.gives_direct_check(nb4));
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
    fn ep_can_interpose_slider_check() {
        init();
        let b = Board::from_fen("4k3/8/K6r/Pp6/8/8/8/8 w - b6 0 1");
        let checkers = b.checkers();
        let pinned = b.pinned();
        let ep = make_move(32, 41, FLAG_EN_PASSANT); // a5xb6 e.p.

        assert!(b.is_legal(ep, pinned, checkers));
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
    /// Not a behavioural test — purely a hash-invariant guard, with
    /// particular attention to en passant. Analogous to
    /// threat_accum::fuzz_random_games.
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
            // Pinned-EP positions: vertical pin and the shared-rank pin.
            "4r1k1/3p4/8/4P3/8/8/8/4K3 b - - 0 1",
            "8/2p1p3/8/K2P3r/8/8/8/7k b - - 0 1",
            // Chess960: two rooks per side of the king, rights naming the
            // inner ones, so castling, rook moves and rook captures all
            // change rights whose rook keys are hashed.
            "1r1k1rr1/pp1ppp1p/8/2p3p1/2P3P1/8/PP1PPP1P/1R1K1RR1 w FBfb - 0 1",
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
                    let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);

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
        let _b_single = {
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

    /// Regression: an EP target whose only capturer is pinned (the capture
    /// would leave its own king in check) is not a real EP right, so it must
    /// not reach the hash. Otherwise the position right after the double push
    /// and the same position one king shuffle later hash differently and a
    /// genuine threefold is missed.
    #[test]
    fn zobrist_ep_ignores_illegal_capture() {
        use crate::uci::parse_uci_move;
        init();

        // e5 is pinned on the e-file by the e8 rook: e5xd6 is illegal.
        let start = "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
        let mut b = Board::from_fen(start);
        let no_ep = Board::from_fen("4r1k1/8/8/3pP3/8/8/8/4K3 w - - 0 1");
        assert_eq!(b.hash, no_ep.hash, "pinned EP capturer must not hash the EP file");

        let shuffle = ["e1f1", "g8h8", "f1e1", "h8g8"];
        for _ in 0..2 {
            for m in shuffle {
                let mv = parse_uci_move(&b, m).expect("legal shuffle move");
                b.make_move(mv);
                assert_eq!(b.hash, b.compute_hash());
            }
        }
        // Start, after 4 plies, after 8 plies: the same legal position 3 times.
        assert!(b.is_repetition_draw(0), "threefold through a pinned-EP start position missed");

        // Rank case: both pawns leave the 5th rank, exposing the a5 king to h5.
        let rank_pin = Board::from_fen("8/8/8/K2pP2r/8/8/8/7k w - d6 0 1");
        let rank_no_ep = Board::from_fen("8/8/8/K2pP2r/8/8/8/7k w - - 0 1");
        assert_eq!(rank_pin.hash, rank_no_ep.hash, "rank-pinned EP must not hash the EP file");

        // Same rank layout but a second capturer on c5 whose capture leaves
        // the e5 pawn as a shield: EP is legal and must stay hashed.
        let rank_ok = Board::from_fen("8/8/8/K1PpP2r/8/8/8/7k w - d6 0 1");
        let rank_ok_no_ep = Board::from_fen("8/8/8/K1PpP2r/8/8/8/7k w - - 0 1");
        assert_eq!(rank_ok.hash ^ rank_ok_no_ep.hash, crate::zobrist::ep_key(3));

        // Incremental path: the double push itself must agree with the
        // no-EP position and with a from-scratch hash.
        let mut pushed = Board::from_fen("4r1k1/3p4/8/4P3/8/8/8/4K3 b - - 0 1");
        pushed.make_move(parse_uci_move(&pushed, "d7d5").unwrap());
        assert_eq!(pushed.hash, pushed.compute_hash());
        assert_eq!(pushed.hash, no_ep.hash);
        let mut rank_pushed = Board::from_fen("8/3p4/8/K3P2r/8/8/8/7k b - - 0 1");
        rank_pushed.make_move(parse_uci_move(&rank_pushed, "d7d5").unwrap());
        assert_eq!(rank_pushed.hash, rank_pushed.compute_hash());
        assert_eq!(rank_pushed.hash, rank_no_ep.hash);
        // ... and unmaking restores the incremental hash exactly.
        rank_pushed.unmake_move();
        assert_eq!(rank_pushed.hash, rank_pushed.compute_hash());
    }

    /// Regression: in Chess960 two rooks can stand on the same side of the
    /// king, so the right flags alone don't say which rook castles. Positions
    /// whose rights name different rooks have different legal moves and must
    /// hash differently; standard-chess hashes must not change at all.
    #[test]
    fn zobrist_castling_includes_960_rook_identity() {
        use crate::uci::parse_uci_move;
        init();

        let f_rook = Board::from_fen("4k3/8/8/8/8/8/8/4KR1R w F - 0 1");
        let h_rook = Board::from_fen("4k3/8/8/8/8/8/8/4KR1R w H - 0 1");
        assert_eq!(f_rook.castling, h_rook.castling);
        assert_ne!(f_rook.castle_rooks, h_rook.castle_rooks);
        assert_ne!(f_rook.hash, h_rook.hash, "960 rights naming different rooks must hash differently");
        // Losing the right makes the two positions identical again.
        let none = Board::from_fen("4k3/8/8/8/8/8/8/4KR1R w - - 0 1");
        let mut f_moved = f_rook.clone();
        f_moved.make_move(parse_uci_move(&f_moved, "f1g1").unwrap());
        f_moved.make_move(parse_uci_move(&f_moved, "e8d8").unwrap());
        f_moved.make_move(parse_uci_move(&f_moved, "g1f1").unwrap());
        f_moved.make_move(parse_uci_move(&f_moved, "d8e8").unwrap());
        assert_eq!(f_moved.castling, 0);
        assert_eq!(f_moved.hash, f_moved.compute_hash());
        assert_eq!(f_moved.hash, none.hash);

        // Standard geometry: the hash is exactly pieces ^ side ^ the four
        // right keys, unchanged by the 960 rook keys.
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R w HAha - 0 1",
        ] {
            let b = Board::from_fen(fen);
            let bare = Board::from_fen(&fen.replace(" KQkq ", " - ").replace(" HAha ", " - "));
            assert_eq!(b.hash ^ bare.hash, crate::zobrist::castle_key(b.castling), "{}", fen);
        }
    }

    /// FEN parse keeps an EP square only if it could follow a double push:
    /// target on the right rank for the side to move, target and the pushed
    /// pawn's origin empty, and the pusher's pawn standing in front.
    #[test]
    fn fen_ep_square_is_validated() {
        init();
        let ep = |fen: &str| Board::from_fen(fen).ep_square;
        // Well-formed (capturable or not): kept.
        assert_eq!(ep("4k3/8/8/Pp6/8/8/8/4K3 w - b6 0 1"), 41);
        assert_eq!(ep("4k3/8/8/1p6/8/8/8/4K3 w - b6 0 1"), 41);
        assert_eq!(ep("4k3/8/8/8/pP6/8/8/4K3 b - b3 0 1"), 17);
        // Wrong rank for the side to move.
        assert_eq!(ep("4k3/8/1p6/P7/8/8/8/4K3 w - b7 0 1"), NO_SQUARE);
        assert_eq!(ep("4k3/8/8/Pp6/8/8/8/4K3 b - b6 0 1"), NO_SQUARE);
        // Occupied target.
        assert_eq!(ep("4k3/8/1n6/Pp6/8/8/8/4K3 w - b6 0 1"), NO_SQUARE);
        // No pawn of the pusher in front of the target.
        assert_eq!(ep("4k3/8/8/P7/8/8/8/4K3 w - b6 0 1"), NO_SQUARE);
        assert_eq!(ep("4k3/8/8/PP6/8/8/8/4K3 w - b6 0 1"), NO_SQUARE);
        // The pushed pawn's origin square is occupied.
        assert_eq!(ep("4k3/1p6/8/Pp6/8/8/8/4K3 w - b6 0 1"), NO_SQUARE);
    }

    /// Recompute the four auxiliary Zobrist keys (pawn_hash, non_pawn_key,
    /// from the current piece/color bitboards. Mirrors the logic in `set_fen`
    /// at the "Compute per-color" block. Used by fuzzers to verify
    /// incremental updates in put_piece / remove_piece / move_piece stay
    /// coherent.
    fn recompute_aux_keys(b: &Board) -> (u64, [u64; 2]) {
        let mut pawn = 0u64;
        let mut np = [0u64; 2];
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
                }
            }
        }
        (pawn, np)
    }

    /// Property-based fuzzer for the auxiliary Zobrist keys
    /// (pawn_hash, non_pawn_key) used by correction history. After every
    /// move, incremental values must equal from-scratch recomputation. These
    /// hashes drive correction-history indexing — a silent divergence would
    /// poison per-bucket corrections and show up as degraded search quality
    /// with no other symptom.
    ///
    /// Analogous to `fuzz_hash_incremental_matches_compute` for the main
    /// Zobrist key, but for the auxiliary keys which have no standalone
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
            // Mostly minors (knight/bishop only, no rooks/queens)
            "4k3/pp6/2nb4/8/8/2NB4/PP6/4K3 w - - 0 1",
            // Mostly majors (rook/queen only, no minors)
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
                let (p, np) = recompute_aux_keys(&board);
                assert_eq!(board.pawn_hash, p, "start pawn_hash mismatch fen={}", fen);
                assert_eq!(board.non_pawn_key, np, "start np mismatch fen={}", fen);

                for ply in 0..PLIES {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);

                    board.make_move(mv);

                    let (p, np) = recompute_aux_keys(&board);
                    if board.pawn_hash != p || board.non_pawn_key != np
                    {
                        panic!(
                            "aux key drift: fen_idx={} game={} ply={} move={} seed={:#x}\n\
                             pawn  incr={:#x} fresh={:#x}\n\
                             np    incr=[{:#x},{:#x}] fresh=[{:#x},{:#x}]\n\
                             fen={}",
                            fen_idx, game, ply,
                            crate::types::move_to_uci(mv), seed,
                            board.pawn_hash, p,
                            board.non_pawn_key[0], board.non_pawn_key[1], np[0], np[1],
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

                // Track whether each op was null (true) or regular (false)
                // so we unmake with the right function.
                let mut ops: Vec<bool> = Vec::new();
                for _ in 0..PLIES {
                    // 1 in 3 chance of null, but only if not in check.
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let want_null = next_u32(&mut rng).is_multiple_of(3);
                    let in_check = board.checkers() != 0;
                    if want_null && !in_check {
                        board.make_null_move();
                        ops.push(true);
                    } else {
                        let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);
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
                    let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);
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
                         castle orig={} now={}  ep orig={} now={}\n\
                         stm    orig={} now={}  hm orig={} now={} fm orig={} now={}",
                        fen_idx, game, move_count, seed,
                        orig_fen, board.to_fen(),
                        orig_hash, board.hash,
                        orig_pawn, board.pawn_hash,
                        orig_np, board.non_pawn_key,
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

    /// `is_legal_unchecked_side` must agree with `is_legal` on every
    /// pseudo-legal move of every position whose side to move is not in
    /// check, and both must agree with the ground truth (make the move, is
    /// the mover's king attacked?). Walks full trees to depth 3 from
    /// positions rich in pins, king walks next to sliders, en passant
    /// discovered checks and (Chess960) castling, then random playouts for
    /// breadth.
    #[test]
    fn legality_from_enemy_attack_map_matches_is_legal() {
        use crate::movegen::{generate_all_moves, generate_legal_moves};
        init();

        fn check_node(b: &mut Board) -> usize {
            if b.checkers() != 0 { return 0; }
            let pinned = b.pinned();
            let them = flip_color(b.side_to_move);
            let enemy_attacks = b.attacks_by_color(them);
            let moves = generate_all_moves(b);
            let mut n = 0;
            for i in 0..moves.len {
                let mv = moves.get(i);
                let fast = b.is_legal_unchecked_side(mv, pinned, enemy_attacks);
                let slow = b.is_legal(mv, pinned, 0);
                assert_eq!(fast, slow, "fast/slow legality differ: fen={} mv={}", b.to_fen(), b.uci_move(mv));
                let us = b.side_to_move;
                let truth = if b.make_move(mv) {
                    let ksq = b.king_sq(us) as u32;
                    let ok = b.attackers_to(ksq, b.occupied()) & b.colors[flip_color(us) as usize] == 0;
                    b.unmake_move();
                    ok
                } else { false };
                // Castling legality also forbids passing through check, which
                // the post-make king test cannot see; is_legal is the oracle there.
                if move_flags(mv) != FLAG_CASTLE {
                    assert_eq!(fast, truth, "legality wrong vs ground truth: fen={} mv={}", b.to_fen(), b.uci_move(mv));
                }
                n += 1;
            }
            n
        }
        fn walk(b: &mut Board, depth: u32) -> usize {
            let mut n = check_node(b);
            if depth == 0 { return n; }
            let legal = generate_legal_moves(b);
            for i in 0..legal.len {
                b.make_move(legal.get(i));
                n += walk(b, depth - 1);
                b.unmake_move();
            }
            n
        }

        const FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            // Horizontal en-passant pin and kings in mutual opposition.
            "8/8/8/K1pP3r/8/8/8/7k w - c6 0 2",
            "8/8/3k4/8/3K4/8/8/8 w - - 0 1",
            // King beside a long diagonal / file with pinned defenders.
            "4k3/8/8/1b6/8/3N4/4K3/4r3 w - - 0 1",
            "q3k3/8/8/8/4B3/8/6K1/8 w - - 0 1",
            // Chess960 castling with the rook shielding the king path.
            "1r2k1r1/pppppppp/8/8/8/8/PPPPPPPP/1R2K1R1 w GBgb - 0 1",
        ];
        let mut total = 0;
        for fen in FENS {
            let mut b = Board::from_fen(fen);
            total += walk(&mut b, 3);
        }

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }
        let mut rng = 0x1E6A1u32;
        for fen in FENS {
            for _ in 0..20 {
                let mut b = Board::from_fen(fen);
                for _ in 0..60 {
                    total += check_node(&mut b);
                    let legal = generate_legal_moves(&b);
                    if legal.len == 0 { break; }
                    b.make_move(legal.get((next_u32(&mut rng) as usize) % legal.len));
                }
            }
        }
        assert!(total > 100_000, "too few moves checked: {}", total);
    }

    /// A second occurrence at/before the root is not yet a draw claim.
    #[test]
    fn repetition_second_occurrence_at_root_not_draw() {
        init();
        let mut b = Board::startpos();
        let hash0 = b.hash;
        let nf3 = make_move(6, 21, FLAG_NONE);
        let nc6 = make_move(57, 42, FLAG_NONE); // Black b8(57) → c6(42)
        let ng1 = make_move(21, 6, FLAG_NONE);
        let nb8 = make_move(42, 57, FLAG_NONE);
        b.make_move(nf3);
        b.make_move(nc6);
        b.make_move(ng1);
        b.make_move(nb8);
        assert_eq!(b.hash, hash0, "knight dance must return to startpos hash");
        assert!(!b.is_repetition_draw(0), "root second occurrence is not a threefold draw");
        assert!(!b.is_repetition_draw(4), "occurrence exactly at root is not inside-tree draw");
    }

    #[test]
    fn no_repetition_after_single_knight_move() {
        init();
        let mut b = Board::startpos();
        let nf3 = make_move(6, 21, FLAG_NONE);
        b.make_move(nf3);
        assert!(!b.is_repetition_draw(1), "single move: no rep possible");
    }

    /// A second occurrence is drawable when the earlier occurrence is strictly
    /// inside the search tree.
    #[test]
    fn repetition_inside_tree_is_draw() {
        init();
        let mut b = Board::startpos();
        b.make_move(make_move(6, 21, FLAG_NONE));   // Nf3
        b.make_move(make_move(57, 42, FLAG_NONE));  // Nc6
        let hash_after_two = b.hash;
        b.make_move(make_move(21, 6, FLAG_NONE));   // Ng1
        b.make_move(make_move(42, 57, FLAG_NONE));  // Nb8
        b.make_move(make_move(6, 21, FLAG_NONE));   // Nf3
        b.make_move(make_move(57, 42, FLAG_NONE));  // Nc6
        assert_eq!(b.hash, hash_after_two);
        assert!(b.is_repetition_draw(6), "previous occurrence at ply 2 is inside the tree");
    }

    /// A third occurrence is drawable even when both earlier occurrences are
    /// at/before the root.
    #[test]
    fn repetition_third_occurrence_before_root_is_draw() {
        init();
        let mut b = Board::startpos();
        let hash0 = b.hash;
        for _ in 0..2 {
            b.make_move(make_move(6, 21, FLAG_NONE));   // Nf3
            b.make_move(make_move(57, 42, FLAG_NONE));  // Nc6
            b.make_move(make_move(21, 6, FLAG_NONE));   // Ng1
            b.make_move(make_move(42, 57, FLAG_NONE));  // Nb8
        }
        assert_eq!(b.hash, hash0);
        assert!(b.is_repetition_draw(0), "third occurrence in game history is draw");
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

        // Now play e4, which resets halfmove to 0. Any pre-e4 match
        // must NOT be returned.
        b.make_move(make_move(12, 28, FLAG_DOUBLE_PUSH)); // e2e4
        assert_eq!(b.halfmove, 0, "pawn move resets halfmove");

        // stack now has 3 entries: startpos, post-Nf3, post-Nc6.
        // Current position is post-Nf3-Nc6-e4 with halfmove=0.
        // Main-search: limit = min(0, 3) = 0, loop never executes → correctly returns false.
        // Corrupt stack[2].hash (most recent = post-Nc6 state) to match current: main-search
        // must NOT see it because limit = 0.
        let saved2 = b.undo_stack[2].hash;
        b.undo_stack[2].hash = b.hash;
        assert!(!b.is_repetition_draw(3),
            "halfmove=0 limit must block detection past irreversible move");
        b.undo_stack[2].hash = saved2;
    }

    /// Random-games fuzzer: the repetition helper must not panic and must
    /// respect plausible root-boundary choices over legal games.
    #[test]
    fn fuzz_repetition_draw_helper() {
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
                    let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);
                    board.make_move(mv);

                    let stack_len = board.undo_stack.len() as i32;
                    let _ = board.is_repetition_draw(0);
                    let _ = board.is_repetition_draw((ply as i32 + 1).min(stack_len));
                    let _ = board.is_repetition_draw(stack_len + 4);
                }
            }
        }
    }
}
