/// Precomputed attack tables: knights, kings, pawns, and magic bitboards for sliders.

use crate::bitboard::*;
use crate::types::*;

/// Returns true if the CPU has fast (single-cycle) PEXT.
/// AMD Zen 1/2 (family 0x17) has BMI2 but PEXT is microcoded (~18 cycles).
/// AMD Zen 3+ (family 0x19+) and all Intel BMI2 CPUs have fast PEXT.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)]
fn has_fast_pext() -> bool {
    // CPUID leaf 0: get vendor string
    // Safety: CPUID is always available on x86_64
    let cpuid0 = unsafe { std::arch::x86_64::__cpuid(0) };
    let vendor = [cpuid0.ebx, cpuid0.edx, cpuid0.ecx];
    let is_amd = vendor == [0x6874_7541, 0x6974_6E65, 0x444D_4163]; // "AuthenticAMD"

    if !is_amd {
        return true; // Intel (and others) with BMI2 have fast PEXT
    }

    // CPUID leaf 1: EAX bits [11:8] = family, [27:20] = extended family
    let cpuid1 = unsafe { std::arch::x86_64::__cpuid(1) };
    let base_family = (cpuid1.eax >> 8) & 0xF;
    let ext_family = (cpuid1.eax >> 20) & 0xFF;
    let family = if base_family == 0xF {
        base_family + ext_family
    } else {
        base_family
    };

    // AMD family 0x19 = Zen 3, 0x1A = Zen 5. Family 0x17 = Zen 1/2 (slow PEXT).
    family >= 0x19
}

// Precomputed leaper attacks — computed at compile time via const fn.
// Previously `static mut` initialized by `init_attacks()`; now `pub const`
// so the compiler treats them as known constants (better CSE, inlining,
// cache locality — see #557 king-bucket refactor pattern).
pub const KNIGHT_ATTACKS: [Bitboard; 64] = build_knight_attacks();
pub const KING_ATTACKS: [Bitboard; 64] = build_king_attacks();
pub const PAWN_ATTACKS: [[Bitboard; 64]; 2] = build_pawn_attacks();

#[inline(always)]
pub fn knight_attacks(sq: u32) -> Bitboard {
    KNIGHT_ATTACKS[sq as usize]
}

#[inline(always)]
pub fn king_attacks(sq: u32) -> Bitboard {
    KING_ATTACKS[sq as usize]
}

#[inline(always)]
pub fn pawn_attacks(color: Color, sq: u32) -> Bitboard {
    PAWN_ATTACKS[color as usize][sq as usize]
}

/// Build the knight-attacks table at compile time via const fn.
const fn build_knight_attacks() -> [Bitboard; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let bb = 1u64 << sq;
        let mut attacks = 0u64;
        attacks |= (bb << 17) & NOT_FILE_A;
        attacks |= (bb << 15) & NOT_FILE_H;
        attacks |= (bb << 10) & NOT_FILE_AB;
        attacks |= (bb << 6)  & NOT_FILE_GH;
        attacks |= (bb >> 6)  & NOT_FILE_AB;
        attacks |= (bb >> 10) & NOT_FILE_GH;
        attacks |= (bb >> 15) & NOT_FILE_A;
        attacks |= (bb >> 17) & NOT_FILE_H;
        table[sq] = attacks;
        sq += 1;
    }
    table
}

/// Build the king-attacks table at compile time.
const fn build_king_attacks() -> [Bitboard; 64] {
    let mut table = [0u64; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let bb = 1u64 << sq;
        // Same as bitboard::{north, south, east, west, north_east, ...}
        // inlined so the helpers don't need to be const fn.
        let mut attacks = 0u64;
        attacks |= bb << 8;                  // north
        attacks |= bb >> 8;                  // south
        attacks |= (bb << 1) & NOT_FILE_A;   // east
        attacks |= (bb >> 1) & NOT_FILE_H;   // west
        attacks |= (bb << 9) & NOT_FILE_A;   // north_east
        attacks |= (bb << 7) & NOT_FILE_H;   // north_west
        attacks |= (bb >> 7) & NOT_FILE_A;   // south_east
        attacks |= (bb >> 9) & NOT_FILE_H;   // south_west
        table[sq] = attacks;
        sq += 1;
    }
    table
}

/// Build the pawn-attacks table at compile time [color][sq].
const fn build_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut table = [[0u64; 64]; 2];
    let mut sq = 0usize;
    while sq < 64 {
        let bb = 1u64 << sq;
        // White captures NE/NW (forward diagonals)
        table[0][sq] = ((bb << 9) & NOT_FILE_A) | ((bb << 7) & NOT_FILE_H);
        // Black captures SE/SW
        table[1][sq] = ((bb >> 7) & NOT_FILE_A) | ((bb >> 9) & NOT_FILE_H);
        sq += 1;
    }
    table
}

// Magic bitboard tables for sliding pieces
struct MagicEntry {
    mask: Bitboard,
    magic: u64,
    shift: u32,
    offset: u32, // index into the attack table
}

static mut BISHOP_MAGICS: [MagicEntry; 64] = unsafe { std::mem::zeroed() };
static mut ROOK_MAGICS: [MagicEntry; 64] = unsafe { std::mem::zeroed() };
static mut ATTACK_TABLE: Vec<Bitboard> = Vec::new();

// Use PEXT when available (Intel/AMD Zen3+) for perfect hashing
#[cfg(target_arch = "x86_64")]
static mut USE_PEXT: bool = false;

#[inline(always)]
pub fn bishop_attacks(sq: u32, occ: Bitboard) -> Bitboard {
    unsafe {
        let entry = &BISHOP_MAGICS[sq as usize];
        #[cfg(target_arch = "x86_64")]
        if USE_PEXT {
            let idx = std::arch::x86_64::_pext_u64(occ, entry.mask) as usize;
            return ATTACK_TABLE[entry.offset as usize + idx];
        }
        let idx = ((occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
        ATTACK_TABLE[entry.offset as usize + idx]
    }
}

#[inline(always)]
pub fn rook_attacks(sq: u32, occ: Bitboard) -> Bitboard {
    unsafe {
        let entry = &ROOK_MAGICS[sq as usize];
        #[cfg(target_arch = "x86_64")]
        if USE_PEXT {
            let idx = std::arch::x86_64::_pext_u64(occ, entry.mask) as usize;
            return ATTACK_TABLE[entry.offset as usize + idx];
        }
        let idx = ((occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
        ATTACK_TABLE[entry.offset as usize + idx]
    }
}

#[inline(always)]
pub fn queen_attacks(sq: u32, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

// Precomputed magic numbers (from well-known tables)
const ROOK_MAGICS_CONST: [u64; 64] = [
    0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
    0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
    0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
    0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
    0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
    0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
    0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
    0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
    0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
    0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
    0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
    0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
    0x00FFFCDDFCED714A, 0x007FFCDDFCED714A, 0x003FFFCDFFD88096, 0x0004500100982005,
    0x801201100408A002, 0x0001008400220841, 0x0808013042080084, 0x0800008040210412,
];

const BISHOP_MAGICS_CONST: [u64; 64] = [
    0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
    0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
    0x0000040404040400, 0x0000020202020200, 0x0000040102020000, 0x0000040400800000,
    0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
    0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
    0x0000800400A00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
    0x0002080010101000, 0x0001040008080800, 0x0000208004010400, 0x0000404004010200,
    0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
    0x0001041000202000, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
    0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
    0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
    0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
    0x0000410410400000, 0x0000208208200000, 0x0000002084100000, 0x0000000020880000,
    0x0000001002020000, 0x0000040408020000, 0x0004040404040000, 0x0002020202020000,
    0x0000104104104000, 0x0000002082082000, 0x0000000020841000, 0x0000000000208800,
    0x0000000010020200, 0x0000000404080200, 0x0000040404040400, 0x0002020202020200,
];

// Relevant bits for each square (number of bits in the mask)
const ROOK_BITS: [u32; 64] = [
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12,
];

const BISHOP_BITS: [u32; 64] = [
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6,
];

/// Compute rook mask for a square (edges excluded).
fn rook_mask(sq: u32) -> Bitboard {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut mask = 0u64;

    for i in (r + 1)..7 { mask |= 1u64 << (i * 8 + f); }
    for i in 1..r       { mask |= 1u64 << (i * 8 + f); }
    for j in (f + 1)..7 { mask |= 1u64 << (r * 8 + j); }
    for j in 1..f       { mask |= 1u64 << (r * 8 + j); }

    mask
}

/// Compute bishop mask for a square (edges excluded).
fn bishop_mask(sq: u32) -> Bitboard {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut mask = 0u64;

    let dirs: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for &(dr, df) in &dirs {
        let mut ri = r + dr;
        let mut fi = f + df;
        while ri > 0 && ri < 7 && fi > 0 && fi < 7 {
            mask |= 1u64 << (ri * 8 + fi);
            ri += dr;
            fi += df;
        }
    }

    mask
}

/// Compute rook attacks for a given occupancy.
fn rook_attacks_slow(sq: u32, occ: Bitboard) -> Bitboard {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut attacks = 0u64;

    // North
    for i in (r + 1)..8 {
        let bit = 1u64 << (i * 8 + f);
        attacks |= bit;
        if occ & bit != 0 { break; }
    }
    // South
    for i in (0..r).rev() {
        let bit = 1u64 << (i * 8 + f);
        attacks |= bit;
        if occ & bit != 0 { break; }
    }
    // East
    for j in (f + 1)..8 {
        let bit = 1u64 << (r * 8 + j);
        attacks |= bit;
        if occ & bit != 0 { break; }
    }
    // West
    for j in (0..f).rev() {
        let bit = 1u64 << (r * 8 + j);
        attacks |= bit;
        if occ & bit != 0 { break; }
    }

    attacks
}

/// Compute bishop attacks for a given occupancy.
fn bishop_attacks_slow(sq: u32, occ: Bitboard) -> Bitboard {
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    let mut attacks = 0u64;

    let dirs: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for &(dr, df) in &dirs {
        let mut ri = r + dr;
        let mut fi = f + df;
        while ri >= 0 && ri < 8 && fi >= 0 && fi < 8 {
            let bit = 1u64 << (ri * 8 + fi);
            attacks |= bit;
            if occ & bit != 0 { break; }
            ri += dr;
            fi += df;
        }
    }

    attacks
}

/// Enumerate all subsets of a bitboard mask (Carry-Rippler).
fn enumerate_subsets(mask: Bitboard) -> Vec<Bitboard> {
    let mut subsets = Vec::new();
    let mut subset = 0u64;
    loop {
        subsets.push(subset);
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 { break; }
    }
    subsets
}

fn compute_knight_attacks(sq: u32) -> Bitboard {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;

    attacks |= (bb << 17) & NOT_FILE_A;   // NNE
    attacks |= (bb << 15) & NOT_FILE_H;   // NNW
    attacks |= (bb << 10) & NOT_FILE_AB;  // NEE
    attacks |= (bb << 6)  & NOT_FILE_GH;  // NWW
    attacks |= (bb >> 6)  & NOT_FILE_AB;  // SEE
    attacks |= (bb >> 10) & NOT_FILE_GH;  // SWW
    attacks |= (bb >> 15) & NOT_FILE_A;   // SSE
    attacks |= (bb >> 17) & NOT_FILE_H;   // SSW

    attacks
}

fn compute_king_attacks(sq: u32) -> Bitboard {
    let bb = 1u64 << sq;
    let mut attacks = 0u64;

    attacks |= north(bb) | south(bb);
    attacks |= east(bb) | west(bb);
    attacks |= north_east(bb) | north_west(bb);
    attacks |= south_east(bb) | south_west(bb);

    attacks
}

fn compute_pawn_attacks(color: Color, sq: u32) -> Bitboard {
    let bb = 1u64 << sq;
    if color == WHITE {
        north_east(bb) | north_west(bb)
    } else {
        south_east(bb) | south_west(bb)
    }
}

/// Initialize all attack tables. Must be called once at startup.
pub fn init_attacks() {
    // Leaper attacks
    // KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS are now `pub const` —
    // computed at compile time, no runtime init needed (see tables at top
    // of file). Only magic tables still need runtime init.

    // Detect PEXT support — only enable when fast (not microcoded)
    // AMD Zen 1/2 (family 0x17) has BMI2 but PEXT is ~18 cycles (microcoded).
    // AMD Zen 3+ (family 0x19+) and all Intel BMI2 CPUs have fast (1 cycle) PEXT.
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            USE_PEXT = is_x86_feature_detected!("bmi2") && has_fast_pext();
        }
    }

    // Calculate total table size needed
    let mut total_size = 0u32;
    let mut bishop_offsets = [0u32; 64];
    let mut rook_offsets = [0u32; 64];

    for sq in 0..64 {
        bishop_offsets[sq] = total_size;
        #[cfg(target_arch = "x86_64")]
        let use_pext = unsafe { USE_PEXT };
        #[cfg(not(target_arch = "x86_64"))]
        let use_pext = false;

        if use_pext {
            total_size += 1 << popcount(bishop_mask(sq as u32));
        } else {
            total_size += 1 << BISHOP_BITS[sq];
        }
    }
    for sq in 0..64 {
        rook_offsets[sq] = total_size;
        #[cfg(target_arch = "x86_64")]
        let use_pext = unsafe { USE_PEXT };
        #[cfg(not(target_arch = "x86_64"))]
        let use_pext = false;

        if use_pext {
            total_size += 1 << popcount(rook_mask(sq as u32));
        } else {
            total_size += 1 << ROOK_BITS[sq];
        }
    }

    // Allocate attack table
    unsafe {
        ATTACK_TABLE = vec![0; total_size as usize];
    }

    // Fill bishop magic entries and attack table
    for sq in 0..64u32 {
        let mask = bishop_mask(sq);
        let bits = BISHOP_BITS[sq as usize];
        let magic = BISHOP_MAGICS_CONST[sq as usize];
        let offset = bishop_offsets[sq as usize];

        unsafe {
            BISHOP_MAGICS[sq as usize] = MagicEntry {
                mask,
                magic,
                shift: 64 - bits,
                offset,
            };
        }

        for subset in enumerate_subsets(mask) {
            let attacks = bishop_attacks_slow(sq, subset);

            #[cfg(target_arch = "x86_64")]
            let use_pext = unsafe { USE_PEXT };
            #[cfg(not(target_arch = "x86_64"))]
            let use_pext = false;

            let idx = if use_pext {
                #[cfg(target_arch = "x86_64")]
                unsafe { std::arch::x86_64::_pext_u64(subset, mask) as usize }
                #[cfg(not(target_arch = "x86_64"))]
                { 0 }
            } else {
                ((subset.wrapping_mul(magic)) >> (64 - bits)) as usize
            };

            unsafe {
                let entry = &mut ATTACK_TABLE[offset as usize + idx];
                assert!(*entry == 0 || *entry == attacks, "Magic collision for bishop on sq {}", sq);
                *entry = attacks;
            }
        }
    }

    // Fill rook magic entries and attack table
    for sq in 0..64u32 {
        let mask = rook_mask(sq);
        let bits = ROOK_BITS[sq as usize];
        let magic = ROOK_MAGICS_CONST[sq as usize];
        let offset = rook_offsets[sq as usize];

        unsafe {
            ROOK_MAGICS[sq as usize] = MagicEntry {
                mask,
                magic,
                shift: 64 - bits,
                offset,
            };
        }

        for subset in enumerate_subsets(mask) {
            let attacks = rook_attacks_slow(sq, subset);

            #[cfg(target_arch = "x86_64")]
            let use_pext = unsafe { USE_PEXT };
            #[cfg(not(target_arch = "x86_64"))]
            let use_pext = false;

            let idx = if use_pext {
                #[cfg(target_arch = "x86_64")]
                unsafe { std::arch::x86_64::_pext_u64(subset, mask) as usize }
                #[cfg(not(target_arch = "x86_64"))]
                { 0 }
            } else {
                ((subset.wrapping_mul(magic)) >> (64 - bits)) as usize
            };

            unsafe {
                let entry = &mut ATTACK_TABLE[offset as usize + idx];
                assert!(*entry == 0 || *entry == attacks, "Magic collision for rook on sq {}", sq);
                *entry = attacks;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() { crate::init(); }

    #[test]
    fn test_knight_attacks() {
        init();
        // Knight on e4 (sq 28) attacks 8 squares
        assert_eq!(popcount(knight_attacks(28)), 8);
        // Knight on a1 (sq 0) attacks 2 squares
        assert_eq!(popcount(knight_attacks(0)), 2);
        // Knight on h8 (sq 63) attacks 2 squares
        assert_eq!(popcount(knight_attacks(63)), 2);
    }

    #[test]
    fn test_king_attacks() {
        init();
        // King on e4 attacks 8 squares
        assert_eq!(popcount(king_attacks(28)), 8);
        // King on a1 attacks 3 squares
        assert_eq!(popcount(king_attacks(0)), 3);
    }

    #[test]
    fn test_pawn_attacks() {
        init();
        // White pawn on e4 attacks d5 and f5
        let att = pawn_attacks(WHITE, 28);
        assert_eq!(popcount(att), 2);
        assert!(att & (1u64 << 35) != 0); // d5
        assert!(att & (1u64 << 37) != 0); // f5
    }

    #[test]
    fn test_rook_attacks_empty_board() {
        init();
        // Rook on a1, empty board: 14 squares (7 on file a + 7 on rank 1)
        let att = rook_attacks(0, 0);
        assert_eq!(popcount(att), 14);
    }

    #[test]
    fn test_bishop_attacks_empty_board() {
        init();
        // Bishop on e4, empty board: 13 squares
        let att = bishop_attacks(28, 0);
        assert_eq!(popcount(att), 13);
    }

    #[test]
    fn test_rook_attacks_blocked() {
        init();
        // Rook on a1, piece on a4: should see a2,a3,a4 (blocked) + rank 1
        let occ = 1u64 << 24; // a4
        let att = rook_attacks(0, occ);
        assert!(att & (1u64 << 8) != 0);  // a2
        assert!(att & (1u64 << 16) != 0); // a3
        assert!(att & (1u64 << 24) != 0); // a4 (capture)
        assert!(att & (1u64 << 32) == 0); // a5 (blocked)
    }

    #[test]
    fn test_queen_attacks() {
        init();
        // Queen on d4, empty board: rook(14) + bishop(13) = 27
        let att = queen_attacks(27, 0);
        assert_eq!(popcount(att), 27);
    }
}
