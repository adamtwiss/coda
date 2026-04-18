/// Bitboard operations and precomputed masks.

pub type Bitboard = u64;

pub const EMPTY: Bitboard = 0;
pub const ALL: Bitboard = !0u64;

// File masks
pub const FILE_A: Bitboard = 0x0101010101010101;
pub const FILE_B: Bitboard = FILE_A << 1;
pub const FILE_C: Bitboard = FILE_A << 2;
pub const FILE_D: Bitboard = FILE_A << 3;
pub const FILE_E: Bitboard = FILE_A << 4;
pub const FILE_F: Bitboard = FILE_A << 5;
pub const FILE_G: Bitboard = FILE_A << 6;
pub const FILE_H: Bitboard = FILE_A << 7;

pub const FILES: [Bitboard; 8] = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H];

// Rank masks
pub const RANK_1: Bitboard = 0xFF;
pub const RANK_2: Bitboard = RANK_1 << 8;
pub const RANK_3: Bitboard = RANK_1 << 16;
pub const RANK_4: Bitboard = RANK_1 << 24;
pub const RANK_5: Bitboard = RANK_1 << 32;
pub const RANK_6: Bitboard = RANK_1 << 40;
pub const RANK_7: Bitboard = RANK_1 << 48;
pub const RANK_8: Bitboard = RANK_1 << 56;

pub const RANKS: [Bitboard; 8] = [RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8];

// Not-file masks for shift safety
pub const NOT_FILE_A: Bitboard = !FILE_A;
pub const NOT_FILE_H: Bitboard = !FILE_H;
pub const NOT_FILE_AB: Bitboard = !(FILE_A | FILE_B);
pub const NOT_FILE_GH: Bitboard = !(FILE_G | FILE_H);

#[inline(always)]
pub fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

#[inline(always)]
pub fn lsb(bb: Bitboard) -> u32 {
    debug_assert!(bb != 0);
    bb.trailing_zeros()
}

#[inline(always)]
pub fn msb(bb: Bitboard) -> u32 {
    debug_assert!(bb != 0);
    63 - bb.leading_zeros()
}

/// Pop least significant bit and return its index.
#[inline(always)]
pub fn pop_lsb(bb: &mut Bitboard) -> u32 {
    let sq = lsb(*bb);
    *bb &= *bb - 1;
    sq
}

/// Check if more than one bit is set.
#[inline(always)]
pub fn more_than_one(bb: Bitboard) -> bool {
    bb & bb.wrapping_sub(1) != 0
}

// Directional shifts
#[inline(always)]
pub fn north(bb: Bitboard) -> Bitboard {
    bb << 8
}

#[inline(always)]
pub fn south(bb: Bitboard) -> Bitboard {
    bb >> 8
}

#[inline(always)]
pub fn east(bb: Bitboard) -> Bitboard {
    (bb << 1) & NOT_FILE_A
}

#[inline(always)]
pub fn west(bb: Bitboard) -> Bitboard {
    (bb >> 1) & NOT_FILE_H
}

#[inline(always)]
pub fn north_east(bb: Bitboard) -> Bitboard {
    (bb << 9) & NOT_FILE_A
}

#[inline(always)]
pub fn north_west(bb: Bitboard) -> Bitboard {
    (bb << 7) & NOT_FILE_H
}

#[inline(always)]
pub fn south_east(bb: Bitboard) -> Bitboard {
    (bb >> 7) & NOT_FILE_A
}

#[inline(always)]
pub fn south_west(bb: Bitboard) -> Bitboard {
    (bb >> 9) & NOT_FILE_H
}

/// Iterator over set bits of a bitboard.
pub struct BitIterator(pub Bitboard);

impl Iterator for BitIterator {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 {
            None
        } else {
            Some(pop_lsb(&mut self.0))
        }
    }
}

/// Between masks: squares strictly between two squares on the same ray.
/// Precomputed at init time.
static mut BETWEEN: [[Bitboard; 64]; 64] = [[0; 64]; 64];
/// Line masks: full line through two squares (if on same rank/file/diagonal).
static mut LINE: [[Bitboard; 64]; 64] = [[0; 64]; 64];
/// Ray extension: squares on the ray from `from` through `blocker`,
/// STRICTLY BEYOND `blocker`, in the same direction, until the board
/// edge. Zero when the two squares are not on the same slider ray.
/// Used by threat-generation x-ray scanning to avoid per-blocker
/// magic-table lookups.
static mut RAY_EXTENSION: [[Bitboard; 64]; 64] = [[0; 64]; 64];

pub fn between(sq1: u32, sq2: u32) -> Bitboard {
    unsafe { BETWEEN[sq1 as usize][sq2 as usize] }
}

pub fn line(sq1: u32, sq2: u32) -> Bitboard {
    unsafe { LINE[sq1 as usize][sq2 as usize] }
}

/// Return the bitboard of squares on the ray from `from` through
/// `blocker`, strictly beyond `blocker`. Returns 0 if the two squares
/// are not aligned on a slider ray, or if `blocker` is on a board edge
/// in the relevant direction.
#[inline(always)]
pub fn ray_extension(from: u32, blocker: u32) -> Bitboard {
    unsafe { RAY_EXTENSION[from as usize][blocker as usize] }
}

/// Initialize between and line tables. Must be called once at startup.
pub fn init_bitboards() {
    // We need attack tables first, so this is called after attacks::init()
    // For now, compute ray-based between/line using simple loops
    for sq1 in 0..64u32 {
        for sq2 in 0..64u32 {
            let (bb_between, bb_line, bb_ext) = compute_between_and_line(sq1, sq2);
            unsafe {
                BETWEEN[sq1 as usize][sq2 as usize] = bb_between;
                LINE[sq1 as usize][sq2 as usize] = bb_line;
                RAY_EXTENSION[sq1 as usize][sq2 as usize] = bb_ext;
            }
        }
    }
}

fn compute_between_and_line(sq1: u32, sq2: u32) -> (Bitboard, Bitboard, Bitboard) {
    if sq1 == sq2 {
        return (0, 0, 0);
    }

    let r1 = sq1 / 8;
    let f1 = sq1 % 8;
    let r2 = sq2 / 8;
    let f2 = sq2 % 8;

    let dr = r2 as i32 - r1 as i32;
    let df = f2 as i32 - f1 as i32;

    // Check if on same rank, file, or diagonal
    let (step_r, step_f) = if dr == 0 && df != 0 {
        (0i32, if df > 0 { 1 } else { -1 })
    } else if df == 0 && dr != 0 {
        (if dr > 0 { 1 } else { -1 }, 0i32)
    } else if dr.abs() == df.abs() {
        (if dr > 0 { 1 } else { -1 }, if df > 0 { 1 } else { -1 })
    } else {
        return (0, 0, 0); // Not on same ray
    };

    // Between: squares strictly between sq1 and sq2
    let mut between_bb = 0u64;
    let mut r = r1 as i32 + step_r;
    let mut f = f1 as i32 + step_f;
    while (r != r2 as i32 || f != f2 as i32) && r >= 0 && r < 8 && f >= 0 && f < 8 {
        between_bb |= 1u64 << (r * 8 + f);
        r += step_r;
        f += step_f;
    }

    // Line: full ray through both squares
    let mut line_bb = (1u64 << sq1) | (1u64 << sq2) | between_bb;
    // Extend backward from sq1
    let mut r = r1 as i32 - step_r;
    let mut f = f1 as i32 - step_f;
    while r >= 0 && r < 8 && f >= 0 && f < 8 {
        line_bb |= 1u64 << (r * 8 + f);
        r -= step_r;
        f -= step_f;
    }
    // Extend forward from sq2 — also captures ray-extension bits
    let mut extension_bb = 0u64;
    let mut r = r2 as i32 + step_r;
    let mut f = f2 as i32 + step_f;
    while r >= 0 && r < 8 && f >= 0 && f < 8 {
        let bit = 1u64 << (r * 8 + f);
        line_bb |= bit;
        extension_bb |= bit;
        r += step_r;
        f += step_f;
    }

    (between_bb, line_bb, extension_bb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() { crate::init(); }

    #[test]
    fn test_popcount() {
        assert_eq!(popcount(0), 0);
        assert_eq!(popcount(1), 1);
        assert_eq!(popcount(0xFF), 8);
        assert_eq!(popcount(ALL), 64);
    }

    #[test]
    fn test_lsb_msb() {
        assert_eq!(lsb(1), 0);
        assert_eq!(lsb(8), 3);
        assert_eq!(msb(1), 0);
        assert_eq!(msb(0x80), 7);
    }

    #[test]
    fn test_pop_lsb() {
        let mut bb = 0b1010u64;
        assert_eq!(pop_lsb(&mut bb), 1);
        assert_eq!(bb, 0b1000);
        assert_eq!(pop_lsb(&mut bb), 3);
        assert_eq!(bb, 0);
    }

    #[test]
    fn test_shifts() {
        let e4 = 1u64 << 28; // e4
        assert_eq!(north(e4), 1u64 << 36); // e5
        assert_eq!(south(e4), 1u64 << 20); // e3
        assert_eq!(east(e4), 1u64 << 29); // f4
        assert_eq!(west(e4), 1u64 << 27); // d4
    }

    #[test]
    fn test_file_rank_masks() {
        assert_eq!(popcount(FILE_A), 8);
        assert_eq!(popcount(RANK_1), 8);
        assert_eq!(FILE_A & RANK_1, 1); // a1
    }

    #[test]
    fn test_between() {
        init();
        // Between e1(4) and e8(60) should be e2-e7
        let b = between(4, 60);
        assert_eq!(popcount(b), 6);
        // Between a1(0) and h8(63) should be b2-g7
        let b = between(0, 63);
        assert_eq!(popcount(b), 6);
        // Non-aligned squares: no between
        let b = between(0, 10); // a1 and c2 — not on same ray
        assert_eq!(b, 0);
    }
}
