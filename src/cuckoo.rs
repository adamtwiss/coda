/// Cuckoo cycle detection for proactive repetition avoidance.
///
/// Detects when a single move could reach a previously seen position,
/// allowing the search to treat such positions as draws before the
/// repetition actually occurs. Eval-agnostic — transfers cleanly.
///
/// Based on Stockfish's implementation. Used by SF, Berserk, Viridithas,
/// Stormphrax, Halogen, PlentyChess.

use crate::attacks::*;
use crate::bitboard::between;
use crate::board::Board;
use crate::types::*;
use crate::zobrist::{piece_key, side_key};

const TABLE_SIZE: usize = 8192;
const TABLE_MASK: usize = 0x1FFF;

static mut CUCKOO_KEYS: [u64; TABLE_SIZE] = [0; TABLE_SIZE];
static mut CUCKOO_MOVES: [u32; TABLE_SIZE] = [0; TABLE_SIZE]; // packed: from | (to << 8)

#[inline(always)]
fn h1(key: u64) -> usize {
    (key as usize) & TABLE_MASK
}

#[inline(always)]
fn h2(key: u64) -> usize {
    ((key >> 16) as usize) & TABLE_MASK
}

/// Pack from/to into a u32 for compact storage.
#[inline(always)]
fn pack_move(from: u8, to: u8) -> u32 {
    (from as u32) | ((to as u32) << 8)
}

#[inline(always)]
fn unpack_from(packed: u32) -> u8 {
    (packed & 0xFF) as u8
}

#[inline(always)]
fn unpack_to(packed: u32) -> u8 {
    ((packed >> 8) & 0xFF) as u8
}

/// Initialize cuckoo hash tables. Must be called after init_zobrist() and init_attacks().
pub fn init_cuckoo() {
    unsafe {
        CUCKOO_KEYS = [0; TABLE_SIZE];
        CUCKOO_MOVES = [0; TABLE_SIZE];
    }

    let mut count = 0u32;

    // For each non-pawn piece type and every pair of squares it can reach
    for color in 0u8..2 {
        for pt in [KNIGHT, BISHOP, ROOK, QUEEN, KING] {
            let piece = color * 6 + pt; // Coda piece index 0-11
            for s1 in 0u8..64 {
                // Get attacks from s1 on empty board
                let attacks = match pt {
                    KNIGHT => knight_attacks(s1 as u32),
                    BISHOP => bishop_attacks(s1 as u32, 0),
                    ROOK => rook_attacks(s1 as u32, 0),
                    QUEEN => queen_attacks(s1 as u32, 0),
                    KING => king_attacks(s1 as u32),
                    _ => 0,
                };

                // Only consider s2 > s1 to avoid duplicates per piece
                let mask_above = if s1 < 63 { !((1u64 << (s1 + 1)) - 1) } else { 0 };
                let mut atk = attacks & mask_above;
                while atk != 0 {
                    let s2 = atk.trailing_zeros() as u8;
                    atk &= atk - 1;

                    // Key = XOR of piece on s1 and s2, plus side toggle
                    let mut key = piece_key(piece, s1) ^ piece_key(piece, s2) ^ side_key();
                    let mut mv = pack_move(s1, s2);

                    // Insert into cuckoo table using cuckoo hashing
                    let mut i = h1(key);
                    let mut iterations = 0u32;
                    loop {
                        unsafe {
                            std::mem::swap(&mut CUCKOO_KEYS[i], &mut key);
                            std::mem::swap(&mut CUCKOO_MOVES[i], &mut mv);
                        }
                        if mv == 0 {
                            break; // Found empty slot
                        }
                        // Alternate between h1 and h2
                        i = if i == h1(key) { h2(key) } else { h1(key) };
                        iterations += 1;
                        if iterations > TABLE_SIZE as u32 {
                            panic!("Cuckoo table insertion failed — infinite cycle");
                        }
                    }
                    count += 1;
                }
            }
        }
    }

    debug_assert_eq!(count, 3668, "Expected 3668 cuckoo entries, got {}", count);
}

/// Check if an upcoming repetition can be reached by a single move.
/// Returns true if such a cycle exists in the game history.
pub fn has_game_cycle(board: &Board, ply: i32) -> bool {
    let stack_len = board.undo_stack.len();
    let end = (board.halfmove as usize).min(board.plies_from_null as usize).min(stack_len);
    if end < 3 {
        return false;
    }

    let original_key = board.hash;
    let occ = board.colors[0] | board.colors[1];

    // key_at(i) returns the Zobrist hash of the position i plies ago.
    let key_at = |i: usize| -> u64 {
        if i == 0 { original_key } else { board.undo_stack[stack_len - i].hash }
    };

    // Track XOR chain: if `other == 0`, positions at distance i differ
    // from current by exactly one piece move (Stockfish pattern).
    // Pre-seed with first pair so the XOR accumulation works correctly.
    let mut other: u64 = original_key ^ key_at(1) ^ side_key();

    for i in (3..=end).step_by(2) {
        other ^= key_at(i - 1) ^ key_at(i) ^ side_key();

        if other != 0 {
            continue;
        }

        // Positions differ by exactly one move — look it up in cuckoo table
        let diff = original_key ^ key_at(i);

        let j;
        unsafe {
            let j1 = h1(diff);
            let j2 = h2(diff);
            if CUCKOO_KEYS[j1] == diff {
                j = j1;
            } else if CUCKOO_KEYS[j2] == diff {
                j = j2;
            } else {
                continue;
            }
        }

        let packed = unsafe { CUCKOO_MOVES[j] };
        let from = unpack_from(packed);
        let to = unpack_to(packed);

        // Verify the path between from and to is unobstructed (strictly between only)
        if (between(from as u32, to as u32) & occ) != 0 {
            continue;
        }

        // Within the search tree: always counts as a cycle
        if ply > i as i32 {
            return true;
        }

        // At the root boundary: verify piece belongs to current side to move
        let piece_sq = if board.mailbox[from as usize] != NO_PIECE_TYPE { from } else { to };
        let piece_color = if board.colors[WHITE as usize] & (1u64 << piece_sq) != 0 { WHITE } else { BLACK };
        if piece_color != board.side_to_move {
            continue;
        }

        // Check if the historical position was itself a repetition
        let hist_key = key_at(i);
        for k in ((i + 2)..=end).step_by(2) {
            if key_at(k) == hist_key {
                return true;
            }
        }
    }

    false
}
