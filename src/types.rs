/// Core types for the chess engine.

// Colors
pub const WHITE: u8 = 0;
pub const BLACK: u8 = 1;
pub type Color = u8;

#[inline(always)]
pub fn flip_color(c: Color) -> Color {
    c ^ 1
}

// Piece types (0-5)
pub const PAWN: u8 = 0;
pub const KNIGHT: u8 = 1;
pub const BISHOP: u8 = 2;
pub const ROOK: u8 = 3;
pub const QUEEN: u8 = 4;
pub const KING: u8 = 5;
pub const NO_PIECE_TYPE: u8 = 6;
pub type PieceType = u8;

// Piece = color * 6 + piece_type, or NO_PIECE
pub const NO_PIECE: u8 = 12;
pub type Piece = u8;

#[inline(always)]
pub fn make_piece(color: Color, pt: PieceType) -> Piece {
    color * 6 + pt
}

#[inline(always)]
pub fn piece_color(p: Piece) -> Color {
    p / 6
}

#[inline(always)]
pub fn piece_type(p: Piece) -> PieceType {
    p % 6
}

// Squares (0-63, a1=0, h8=63)
pub type Square = u8;
pub const NO_SQUARE: Square = 64;

#[inline(always)]
pub fn square(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

#[inline(always)]
pub fn file_of(sq: Square) -> u8 {
    sq & 7
}

#[inline(always)]
pub fn rank_of(sq: Square) -> u8 {
    sq >> 3
}

#[inline(always)]
pub fn square_bb(sq: Square) -> u64 {
    1u64 << sq
}

// Move encoding: 16 bits
// bits 0-5: from
// bits 6-11: to
// bits 12-15: flags
pub type Move = u16;
pub const NO_MOVE: Move = 0;

pub const FLAG_NONE: u16 = 0;
pub const FLAG_EN_PASSANT: u16 = 1;
pub const FLAG_CASTLE: u16 = 2;
pub const FLAG_DOUBLE_PUSH: u16 = 0; // No special flag — detected by distance in make_move
pub const FLAG_PROMOTE_N: u16 = 4;
pub const FLAG_PROMOTE_B: u16 = 5;
pub const FLAG_PROMOTE_R: u16 = 6;
pub const FLAG_PROMOTE_Q: u16 = 7;

#[inline(always)]
pub fn make_move(from: Square, to: Square, flags: u16) -> Move {
    (from as u16) | ((to as u16) << 6) | (flags << 12)
}

#[inline(always)]
pub fn move_from(m: Move) -> Square {
    (m & 0x3F) as Square
}

#[inline(always)]
pub fn move_to(m: Move) -> Square {
    ((m >> 6) & 0x3F) as Square
}

#[inline(always)]
pub fn move_flags(m: Move) -> u16 {
    m >> 12
}

#[inline(always)]
pub fn is_promotion(m: Move) -> bool {
    move_flags(m) >= FLAG_PROMOTE_N
}

#[inline(always)]
pub fn promotion_piece_type(m: Move) -> PieceType {
    (move_flags(m) - FLAG_PROMOTE_N) as PieceType + KNIGHT
}

// Castling rights (bitfield)
pub const CASTLE_WK: u8 = 1; // White kingside
pub const CASTLE_WQ: u8 = 2; // White queenside
pub const CASTLE_BK: u8 = 4; // Black kingside
pub const CASTLE_BQ: u8 = 8; // Black queenside

/// Format a square as algebraic notation (e.g., "e4")
pub fn square_name(sq: Square) -> String {
    let f = (b'a' + file_of(sq)) as char;
    let r = (b'1' + rank_of(sq)) as char;
    format!("{}{}", f, r)
}

/// Format a move in UCI notation (e.g., "e2e4", "e7e8q")
pub fn move_to_uci(m: Move) -> String {
    if m == NO_MOVE {
        return "0000".to_string();
    }
    let from = square_name(move_from(m));
    let to = square_name(move_to(m));
    if is_promotion(m) {
        let promo = match promotion_piece_type(m) {
            KNIGHT => 'n',
            BISHOP => 'b',
            ROOK => 'r',
            QUEEN => 'q',
            _ => '?',
        };
        format!("{}{}{}", from, to, promo)
    } else {
        format!("{}{}", from, to)
    }
}
