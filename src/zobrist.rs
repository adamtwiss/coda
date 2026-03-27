/// Zobrist hash keys for incremental hashing.
///
/// Generated with a fixed PRNG seed for deterministic keys across builds.
/// Layout matches GoChess: piece[12][64], en_passant[8], castling[16], side_to_move.

use crate::types::*;

/// Piece-square keys: [piece_index][square] where piece_index = color*6 + piece_type
static mut PIECE_KEYS: [[u64; 64]; 12] = [[0; 64]; 12];
/// En passant file keys (8 files)
static mut EP_KEYS: [u64; 8] = [0; 8];
/// Castling keys (16 combinations of 4-bit castling rights)
static mut CASTLE_KEYS: [u64; 16] = [0; 16];
/// Side to move key
static mut SIDE_KEY: u64 = 0;

#[inline(always)]
pub fn piece_key(piece: Piece, sq: Square) -> u64 {
    debug_assert!((piece as usize) < 12);
    debug_assert!((sq as usize) < 64);
    unsafe { PIECE_KEYS[piece as usize][sq as usize] }
}

#[inline(always)]
pub fn ep_key(file: u8) -> u64 {
    debug_assert!((file as usize) < 8);
    unsafe { EP_KEYS[file as usize] }
}

#[inline(always)]
pub fn castle_key(rights: u8) -> u64 {
    unsafe { CASTLE_KEYS[rights as usize & 15] }
}

#[inline(always)]
pub fn side_key() -> u64 {
    unsafe { SIDE_KEY }
}

/// Simple xorshift64 PRNG for deterministic key generation.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Initialize all Zobrist keys. Must be called once at startup.
pub fn init_zobrist() {
    let mut rng = Rng(0x3243F6A8885A308D); // first 8 bytes of pi fractional part

    unsafe {
        for piece in 0..12 {
            for sq in 0..64 {
                PIECE_KEYS[piece][sq] = rng.next();
            }
        }
        for file in 0..8 {
            EP_KEYS[file] = rng.next();
        }
        for i in 0..16 {
            CASTLE_KEYS[i] = rng.next();
        }
        SIDE_KEY = rng.next();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() { crate::init(); }

    #[test]
    fn test_zobrist_deterministic() {
        init();
        let k1 = piece_key(0, 0);
        let k2 = piece_key(0, 0);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_zobrist_unique() {
        init();
        // All piece keys should be different
        let k1 = piece_key(0, 0);
        let k2 = piece_key(0, 1);
        let k3 = piece_key(1, 0);
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k2, k3);
    }
}
