/// Zobrist hash keys for incremental hashing.
/// Generated from Go math/rand with seed 0x1234567890ABCDEF.
/// Piece index 0-11: White P,N,B,R,Q,K then Black P,N,B,R,Q,K.

use crate::types::*;

/// Piece-square keys: [piece_index 0-11][square 0-63]
static mut PIECE_KEYS: [[u64; 64]; 12] = [[0; 64]; 12];
/// En passant file keys (8 files)
static mut EP_KEYS: [u64; 8] = [0; 8];
/// 4 individual castling keys: [WK=0, WQ=1, BK=2, BQ=3]
static mut CASTLE_KEYS_4: [u64; 4] = [0; 4];
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

/// Compute the castling hash for a given set of castling rights.
/// XORs the individual keys for each right that is set.
/// XORs individual keys: [0]=WK, [1]=WQ, [2]=BK, [3]=BQ.
#[inline(always)]
pub fn castle_key(rights: u8) -> u64 {
    let mut h = 0u64;
    unsafe {
        if rights & CASTLE_WK != 0 { h ^= CASTLE_KEYS_4[0]; }
        if rights & CASTLE_WQ != 0 { h ^= CASTLE_KEYS_4[1]; }
        if rights & CASTLE_BK != 0 { h ^= CASTLE_KEYS_4[2]; }
        if rights & CASTLE_BQ != 0 { h ^= CASTLE_KEYS_4[3]; }
    }
    h
}

#[inline(always)]
pub fn side_key() -> u64 {
    unsafe { SIDE_KEY }
}

// Pre-computed Zobrist keys from Go math/rand with seed 0x1234567890ABCDEF.
// Order: pieces[0..11][0..63], side, castling[4], ep[8].
include!("zobrist_keys.rs");

/// Initialize all Zobrist keys. Must be called once at startup.
pub fn init_zobrist() {
    unsafe {
        PIECE_KEYS = PIECE_KEYS_INIT;
        SIDE_KEY = SIDE_KEY_INIT;
        CASTLE_KEYS_4 = CASTLE_KEYS_4_INIT;
        EP_KEYS = EP_KEYS_INIT;
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
        let k1 = piece_key(0, 0);
        let k2 = piece_key(0, 1);
        let k3 = piece_key(1, 0);
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_castle_key_symmetry() {
        init();
        // XOR of all 4 rights should equal castle_key(0b1111)
        let all = castle_key(CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ);
        let individual = castle_key(CASTLE_WK) ^ castle_key(CASTLE_WQ)
            ^ castle_key(CASTLE_BK) ^ castle_key(CASTLE_BQ);
        assert_eq!(all, individual);
    }
}
