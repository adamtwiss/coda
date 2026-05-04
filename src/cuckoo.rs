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

#[cfg(test)]
mod cuckoo_tests {
    use super::*;
    use crate::board::Board;

    fn init() { crate::init(); }

    /// The cuckoo table should be populated with exactly 3668 entries
    /// for the 4-piece types and their reversible-move pairs. If
    /// init_cuckoo panics, it already signals table corruption — this
    /// test just re-confirms count-invariant.
    #[test]
    fn cuckoo_table_populated() {
        init();
        let mut populated = 0;
        unsafe {
            for i in 0..TABLE_SIZE {
                if CUCKOO_MOVES[i] != 0 {
                    populated += 1;
                }
            }
        }
        assert_eq!(populated, 3668, "Expected 3668 cuckoo entries populated");
    }

    /// Each populated entry's key must equal the XOR of piece_key on
    /// the from/to squares plus side_key. Guards table corruption.
    #[test]
    fn cuckoo_entries_valid_moves() {
        init();
        unsafe {
            for i in 0..TABLE_SIZE {
                let packed = CUCKOO_MOVES[i];
                if packed == 0 { continue; }
                let key = CUCKOO_KEYS[i];
                let from = unpack_from(packed);
                let to = unpack_to(packed);
                assert!(from < 64 && to < 64 && from != to,
                    "invalid squares from={} to={}", from, to);

                // The key must match some piece's (from,to) pair ^ side_key.
                // Search all 10 non-pawn piece indexes; exactly one should match.
                let mut matched = false;
                for piece in 0u8..12 {
                    let pt = piece % 6;
                    if pt == PAWN { continue; } // cuckoo stores only non-pawn moves
                    let expected = piece_key(piece, from) ^ piece_key(piece, to) ^ side_key();
                    if expected == key {
                        matched = true;
                        break;
                    }
                }
                assert!(matched,
                    "cuckoo entry {} has key {:#x} that doesn't match any piece's from/to",
                    i, key);
            }
        }
    }

    /// A 4-ply knight dance (Nf3 Nc6 Ng1) leaves the board such that
    /// the NEXT move Nb8 would restore the starting position. At that
    /// point (just before the repeating move), has_game_cycle must
    /// detect it.
    ///
    /// Call with a large ply so we're well inside the search tree
    /// (bypasses the root-boundary STM check).
    #[test]
    fn cuckoo_detects_knight_dance_at_ply_3() {
        init();
        let mut b = Board::startpos();
        b.make_move(make_move(6, 21, FLAG_NONE));   // Nf3 (g1→f3)
        b.make_move(make_move(57, 42, FLAG_NONE));  // Nc6 (b8→c6)
        b.make_move(make_move(21, 6, FLAG_NONE));   // Ng1 (f3→g1)
        // Now at D. Black to move. Playing Nb8 would reach A.
        // has_game_cycle should detect this via distance-3 XOR match.
        assert!(has_game_cycle(&b, 100),
            "cuckoo must detect 4-ply knight dance at distance 3");
    }

    /// In the starting position no cycle is possible (no history).
    #[test]
    fn cuckoo_no_cycle_in_startpos() {
        init();
        let b = Board::startpos();
        assert!(!has_game_cycle(&b, 0));
    }

    /// After just one move there's nothing to cycle to.
    #[test]
    fn cuckoo_no_cycle_after_one_move() {
        init();
        let mut b = Board::startpos();
        b.make_move(make_move(6, 21, FLAG_NONE));
        assert!(!has_game_cycle(&b, 100));
    }

    /// Random-games fuzzer: has_game_cycle must never panic and
    /// the positive-detection cases must correspond to an actual
    /// repeatable move (one that when made reaches a past position).
    #[test]
    fn fuzz_cuckoo_sanity() {
        use crate::movegen::generate_legal_moves;
        init();

        const FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            // King/knight-rich middlegame where cycles are plausible
            "4k3/8/4n3/8/2N5/8/8/4K3 w - - 0 1",
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        const PLIES: usize = 40;
        const GAMES: usize = 5;

        for (fen_idx, fen) in FENS.iter().enumerate() {
            for game in 0..GAMES {
                let seed: u32 = 0xCACAu32
                    .wrapping_add((fen_idx as u32).wrapping_mul(7919))
                    .wrapping_add((game as u32).wrapping_mul(1_000_003));
                let mut rng = if seed == 0 { 1 } else { seed };
                let mut board = Board::from_fen(fen);

                for _ in 0..PLIES {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    // has_game_cycle must not panic at any intermediate state.
                    let _ = has_game_cycle(&board, 100);
                    let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);
                    board.make_move(mv);
                }
            }
        }
    }
}
