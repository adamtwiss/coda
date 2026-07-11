//! Setwise attack generation: compute attack bitboards for ALL pieces of one
//! type in a single batched operation, instead of per-square magic-bitboard
//! lookups.
//!
//! Used by `attacks_by_color`, `attackers_to`, `pinned`, threat enumeration —
//! anywhere the existing code does a `while bb != 0 { sq = bb.trailing_zeros();
//! attacks |= piece_attacks(sq, occ); bb &= bb-1; }` loop.
//!
//! Structure: pawns are pure scalar shifts; knights have a scalar
//! 8-direction shift fallback + an AVX2 path that does 4 directions per
//! 256-bit register; bishops/rooks have a scalar magic-lookup loop fallback
//! + an AVX2 Kogge-Stone fill. The scalar paths are Coda's own. The AVX2
//! Kogge-Stone/knight paths, however, are too closely modelled on Reckless's
//! `src/setwise.rs` vectorised code to stand as genuinely independent, and —
//! per `docs/license_compliance_review_2026-07-11.md` (item 1) — are being
//! removed in favour of the scalar paths and re-derived independently from
//! Coda's own scalar reference. (Reckless is AGPLv3.)
//!
//! Correctness reference is the per-square magic-lookup behaviour — every
//! setwise impl is unit-tested against `attacks_by_color`-style aggregation
//! over the same bitboards. Integration is gated on perft passing.
//!
//! Why setwise wins: replaces N magic-bitboard lookups (each touching the
//! magic table cache lines for that piece type) with a small fixed number of
//! shift+mask+or operations. Per `docs/coda_vs_reckless_nps_2026-04-23.md`
//! Phase 2 perf data, Coda spends ~5% of cycles in `attackers_to` /
//! `attacks_by_color` / `piece_attacks_occ` / `pinned` patterns where the
//! setwise approach applies.

use crate::bitboard::*;
use crate::types::*;

// =============================================================================
// Pawn attacks (scalar, all platforms)
// =============================================================================

/// Bitboard of all squares attacked by the pawns in `bb` for `color`.
/// Pure scalar — bitboard shifts are already efficient on every CPU.
#[inline]
pub fn pawn_attacks_setwise(bb: Bitboard, color: Color) -> Bitboard {
    if color == WHITE {
        ((bb & !FILE_H) << 9) | ((bb & !FILE_A) << 7)
    } else {
        ((bb & !FILE_A) >> 9) | ((bb & !FILE_H) >> 7)
    }
}

// =============================================================================
// Knight attacks (scalar, all platforms)
// =============================================================================

/// Scalar setwise knight attacks: 8 shift+mask+or operations cover the union
/// of all 8 jump directions for every knight in `bb`.
///
/// Direction encoding (offset, mask of squares that *can* jump in that dir):
///   +6  NNW  exclude {A, B, R8}
///   +15 WNW  exclude {A, R7, R8}
///   +17 ENE  exclude {H, R7, R8}
///   +10 NNE  exclude {G, H, R8}
///   -6  SSE  exclude {G, H, R1}
///   -15 ESE  exclude {H, R1, R2}
///   -17 WSW  exclude {A, R1, R2}
///   -10 SSW  exclude {A, B, R1}
#[inline]
pub fn knight_attacks_setwise_scalar(bb: Bitboard) -> Bitboard {
    ((bb & !(FILE_A | FILE_B | RANK_8)) << 6)
        | ((bb & !(FILE_A | RANK_7 | RANK_8)) << 15)
        | ((bb & !(FILE_H | RANK_7 | RANK_8)) << 17)
        | ((bb & !(FILE_G | FILE_H | RANK_8)) << 10)
        | ((bb & !(FILE_G | FILE_H | RANK_1)) >> 6)
        | ((bb & !(FILE_H | RANK_1 | RANK_2)) >> 15)
        | ((bb & !(FILE_A | RANK_1 | RANK_2)) >> 17)
        | ((bb & !(FILE_A | FILE_B | RANK_1)) >> 10)
}

#[inline]
pub fn knight_attacks_setwise(bb: Bitboard) -> Bitboard {
    knight_attacks_setwise_scalar(bb)
}

// =============================================================================
// Bishop attacks (scalar magic-lookup loop, all platforms)
// =============================================================================

/// Scalar setwise bishop attacks: per-square magic lookup loop. Output
/// bitboard includes attacked-by-our-own-blockers; callers mask if needed.
#[inline]
pub fn bishop_attacks_setwise_scalar(bb: Bitboard, occ: Bitboard) -> Bitboard {
    let mut result: Bitboard = 0;
    let mut bb = bb;
    while bb != 0 {
        let sq = bb.trailing_zeros();
        bb &= bb - 1;
        result |= crate::attacks::bishop_attacks(sq, occ);
    }
    result
}

#[inline]
pub fn bishop_attacks_setwise(bb: Bitboard, occ: Bitboard) -> Bitboard {
    bishop_attacks_setwise_scalar(bb, occ)
}

// =============================================================================
// Rook attacks (scalar magic-lookup loop, all platforms)
// =============================================================================

#[inline]
pub fn rook_attacks_setwise_scalar(bb: Bitboard, occ: Bitboard) -> Bitboard {
    let mut result: Bitboard = 0;
    let mut bb = bb;
    while bb != 0 {
        let sq = bb.trailing_zeros();
        bb &= bb - 1;
        result |= crate::attacks::rook_attacks(sq, occ);
    }
    result
}

#[inline]
pub fn rook_attacks_setwise(bb: Bitboard, occ: Bitboard) -> Bitboard {
    rook_attacks_setwise_scalar(bb, occ)
}

// =============================================================================
// Tests — parity with per-square magic-lookup aggregation
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attacks::{bishop_attacks, knight_attacks, pawn_attacks, rook_attacks};

    /// Reference: aggregate per-square attacks for all set bits of `bb`.
    fn aggregate_per_square_knight(bb: Bitboard) -> Bitboard {
        let mut bb = bb;
        let mut result: Bitboard = 0;
        while bb != 0 {
            let sq = bb.trailing_zeros();
            bb &= bb - 1;
            result |= knight_attacks(sq);
        }
        result
    }

    fn aggregate_per_square_bishop(bb: Bitboard, occ: Bitboard) -> Bitboard {
        let mut bb = bb;
        let mut result: Bitboard = 0;
        while bb != 0 {
            let sq = bb.trailing_zeros();
            bb &= bb - 1;
            result |= bishop_attacks(sq, occ);
        }
        result
    }

    fn aggregate_per_square_rook(bb: Bitboard, occ: Bitboard) -> Bitboard {
        let mut bb = bb;
        let mut result: Bitboard = 0;
        while bb != 0 {
            let sq = bb.trailing_zeros();
            bb &= bb - 1;
            result |= rook_attacks(sq, occ);
        }
        result
    }

    fn aggregate_per_square_pawn(bb: Bitboard, color: Color) -> Bitboard {
        let mut bb = bb;
        let mut result: Bitboard = 0;
        while bb != 0 {
            let sq = bb.trailing_zeros();
            bb &= bb - 1;
            result |= pawn_attacks(color, sq);
        }
        result
    }

    /// Seeded xorshift64* for deterministic test inputs.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    #[test]
    fn pawn_setwise_matches_per_square_white() {
        crate::init();
        let mut r = rng(0xc0da_5e7_71_0001);
        for _ in 0..200 {
            let bb = r() & !(RANK_1 | RANK_8); // pawns can't be on 1 or 8
            assert_eq!(
                pawn_attacks_setwise(bb, WHITE),
                aggregate_per_square_pawn(bb, WHITE),
                "white pawn setwise mismatch on bb {:016x}",
                bb
            );
        }
    }

    #[test]
    fn pawn_setwise_matches_per_square_black() {
        crate::init();
        let mut r = rng(0xc0da_5e7_71_0002);
        for _ in 0..200 {
            let bb = r() & !(RANK_1 | RANK_8);
            assert_eq!(
                pawn_attacks_setwise(bb, BLACK),
                aggregate_per_square_pawn(bb, BLACK),
                "black pawn setwise mismatch on bb {:016x}",
                bb
            );
        }
    }

    #[test]
    fn knight_setwise_matches_per_square() {
        crate::init();
        // Single-knight cases for every square — covers all corner / edge masks.
        for sq in 0..64u32 {
            let bb = 1u64 << sq;
            assert_eq!(
                knight_attacks_setwise_scalar(bb),
                aggregate_per_square_knight(bb),
                "scalar knight setwise mismatch on sq {}",
                sq
            );
        }
        // Multi-knight random cases.
        let mut r = rng(0xc0da_5e7_71_0003);
        for _ in 0..200 {
            let bb = r();
            let agg = aggregate_per_square_knight(bb);
            assert_eq!(knight_attacks_setwise_scalar(bb), agg);
        }
    }

    #[test]
    fn bishop_setwise_matches_per_square() {
        crate::init();
        // Single-bishop cases per square × varied occupancy.
        for sq in 0..64u32 {
            let bb = 1u64 << sq;
            for occ_seed in 0..8u64 {
                let occ = if occ_seed == 0 { 0 } else { 0x12345678_9abcdef0u64.wrapping_mul(occ_seed) };
                let agg = aggregate_per_square_bishop(bb, occ | bb);
                assert_eq!(
                    bishop_attacks_setwise_scalar(bb, occ | bb),
                    agg,
                    "scalar bishop sq={} occ={:016x}",
                    sq,
                    occ
                );
            }
        }
        // Multi-bishop random.
        let mut r = rng(0xc0da_5e7_71_0004);
        for _ in 0..200 {
            let bb = r();
            let occ = r() | bb;
            let agg = aggregate_per_square_bishop(bb, occ);
            assert_eq!(bishop_attacks_setwise_scalar(bb, occ), agg);
        }
    }

    #[test]
    fn rook_setwise_matches_per_square() {
        crate::init();
        for sq in 0..64u32 {
            let bb = 1u64 << sq;
            for occ_seed in 0..8u64 {
                let occ = if occ_seed == 0 { 0 } else { 0x12345678_9abcdef0u64.wrapping_mul(occ_seed) };
                let agg = aggregate_per_square_rook(bb, occ | bb);
                assert_eq!(
                    rook_attacks_setwise_scalar(bb, occ | bb),
                    agg,
                    "scalar rook sq={} occ={:016x}",
                    sq,
                    occ
                );
            }
        }
        let mut r = rng(0xc0da_5e7_71_0005);
        for _ in 0..200 {
            let bb = r();
            let occ = r() | bb;
            let agg = aggregate_per_square_rook(bb, occ);
            assert_eq!(rook_attacks_setwise_scalar(bb, occ), agg);
        }
    }
}
