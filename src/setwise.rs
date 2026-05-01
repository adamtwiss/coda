//! Setwise attack generation: compute attack bitboards for ALL pieces of one
//! type in a single batched operation, instead of per-square magic-bitboard
//! lookups.
//!
//! Used by `attacks_by_color`, `attackers_to`, `pinned`, threat enumeration —
//! anywhere the existing code does a `while bb != 0 { sq = bb.trailing_zeros();
//! attacks |= piece_attacks(sq, occ); bb &= bb-1; }` loop.
//!
//! Approach inspired by Reckless's `src/setwise.rs` (PRs #909, #914,
//! +4.65 STC combined for them). Same structural idea: pawns are pure
//! scalar shifts; knights have a scalar 8-direction shift fallback +
//! an AVX2 path that does 4 directions per 256-bit register;
//! bishops/rooks have a scalar magic-lookup loop fallback + an AVX2
//! Kogge-Stone fill. Independent reimplementation against Coda's
//! bitboard / attacks / types modules.
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
// Knight attacks (scalar fallback + AVX2)
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
#[cfg(not(target_feature = "avx2"))]
#[inline]
pub fn knight_attacks_setwise(bb: Bitboard) -> Bitboard {
    knight_attacks_setwise_scalar(bb)
}

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

/// AVX2 setwise knight attacks: pack the 4 positive-shift directions into
/// one ymm register and the 4 negative-shift into another, do all shifts in
/// 2 SIMD ops + 1 OR-fold, and reduce.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn knight_attacks_setwise_avx2(bb: Bitboard) -> Bitboard {
    use std::arch::x86_64::*;

    // mask_up: 4 lanes for positive shifts, in shift-amount order [10, 17, 15, 6]
    //          (set_epi64x is little-endian: lane 0 = last arg).
    // We list as (shift, exclude_mask) so they pair lane-by-lane:
    //   lane 0: shift=10 mask !(G|H|R8)   NNE
    //   lane 1: shift=17 mask !(H|R7|R8)  ENE
    //   lane 2: shift=15 mask !(A|R7|R8)  WNW
    //   lane 3: shift=6  mask !(A|B|R8)   NNW
    let mask_up = _mm256_set_epi64x(
        !(FILE_A | FILE_B | RANK_8) as i64,
        !(FILE_A | RANK_7 | RANK_8) as i64,
        !(FILE_H | RANK_7 | RANK_8) as i64,
        !(FILE_G | FILE_H | RANK_8) as i64,
    );
    // Down: pair the symmetric directions in the same lane order.
    //   lane 0: shift=-10 mask !(A|B|R1)  SSW
    //   lane 1: shift=-17 mask !(A|R1|R2) WSW
    //   lane 2: shift=-15 mask !(H|R1|R2) ESE
    //   lane 3: shift=-6  mask !(G|H|R1)  SSE
    let mask_dn = _mm256_set_epi64x(
        !(FILE_G | FILE_H | RANK_1) as i64,
        !(FILE_H | RANK_1 | RANK_2) as i64,
        !(FILE_A | RANK_1 | RANK_2) as i64,
        !(FILE_A | FILE_B | RANK_1) as i64,
    );

    let bb_v = _mm256_set1_epi64x(bb as i64);
    let up = _mm256_and_si256(bb_v, mask_up);
    let dn = _mm256_and_si256(bb_v, mask_dn);
    // Variable shifts: 4 distinct shift amounts per ymm register.
    let up_shifted = _mm256_sllv_epi64(up, _mm256_set_epi64x(6, 15, 17, 10));
    let dn_shifted = _mm256_srlv_epi64(dn, _mm256_set_epi64x(6, 15, 17, 10));
    fold_to_bitboard(_mm256_or_si256(up_shifted, dn_shifted))
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn knight_attacks_setwise(bb: Bitboard) -> Bitboard {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { knight_attacks_setwise_avx2(bb) }
    } else {
        knight_attacks_setwise_scalar(bb)
    }
}

// =============================================================================
// Bishop attacks (scalar fallback + AVX2 Kogge-Stone)
// =============================================================================

/// Scalar setwise bishop attacks: per-square magic lookup loop. Used as
/// correctness reference and the non-AVX2 fallback. Output bitboard
/// includes attacked-by-our-own-blockers; callers mask if needed.
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

/// AVX2 setwise bishop attacks via Kogge-Stone-style flooding.
///
/// Pack the 4 diagonal directions (-9, -7, +7, +9) into ymm lanes. Iterate
/// 3 times with doubling distance (1 step, 2 steps, 4 steps) — covers all
/// 7 ranks/files of board. Each iteration:
///   - extend `generate` along empty squares
///   - extend `propagate` (the chain of empties) by the same distance
///
/// Final shift produces the attack squares (one beyond the propagation
/// frontier), masked by edge-respecting masks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bishop_attacks_setwise_avx2(bb: Bitboard, occ: Bitboard) -> Bitboard {
    use std::arch::x86_64::*;

    // mask: edge bits we cannot wrap past per direction.
    //   lane 0: -9 SW → !(R1 | A)
    //   lane 1: -7 SE → !(R1 | H)
    //   lane 2: +7 NW → !(R8 | A)
    //   lane 3: +9 NE → !(R8 | H)
    let mask = _mm256_set_epi64x(
        !(RANK_8 | FILE_H) as i64,
        !(RANK_8 | FILE_A) as i64,
        !(RANK_1 | FILE_H) as i64,
        !(RANK_1 | FILE_A) as i64,
    );

    let mut generate = _mm256_set1_epi64x(bb as i64);
    let mut propagate = _mm256_and_si256(_mm256_set1_epi64x(!occ as i64), mask);

    // Step 1 (distance 1).
    generate = _mm256_or_si256(
        generate,
        _mm256_and_si256(propagate, shiftv_avx2::<-9, -7, 7, 9>(generate)),
    );
    propagate = _mm256_and_si256(propagate, shiftv_avx2::<-9, -7, 7, 9>(propagate));
    // Step 2 (distance 2).
    generate = _mm256_or_si256(
        generate,
        _mm256_and_si256(propagate, shiftv_avx2::<-18, -14, 14, 18>(generate)),
    );
    propagate = _mm256_and_si256(propagate, shiftv_avx2::<-18, -14, 14, 18>(propagate));
    // Step 3 (distance 4 → full board).
    generate = _mm256_or_si256(
        generate,
        _mm256_and_si256(propagate, shiftv_avx2::<-36, -28, 28, 36>(generate)),
    );

    // Final attack frontier: one more shift past the filled region, masked.
    let attacks = _mm256_and_si256(shiftv_avx2::<-9, -7, 7, 9>(generate), mask);
    fold_to_bitboard(attacks)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn bishop_attacks_setwise(bb: Bitboard, occ: Bitboard) -> Bitboard {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { bishop_attacks_setwise_avx2(bb, occ) }
    } else {
        bishop_attacks_setwise_scalar(bb, occ)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn bishop_attacks_setwise(bb: Bitboard, occ: Bitboard) -> Bitboard {
    bishop_attacks_setwise_scalar(bb, occ)
}

// =============================================================================
// Rook attacks (scalar fallback + AVX2 Kogge-Stone)
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rook_attacks_setwise_avx2(bb: Bitboard, occ: Bitboard) -> Bitboard {
    use std::arch::x86_64::*;

    // mask: edge bits we cannot wrap past per direction.
    //   lane 0: -8 S  → !R1
    //   lane 1: -1 W  → !A
    //   lane 2: +1 E  → !H
    //   lane 3: +8 N  → !R8
    let mask = _mm256_set_epi64x(
        !RANK_8 as i64,
        !FILE_H as i64,
        !FILE_A as i64,
        !RANK_1 as i64,
    );

    let mut generate = _mm256_set1_epi64x(bb as i64);
    let mut propagate = _mm256_and_si256(_mm256_set1_epi64x(!occ as i64), mask);

    generate = _mm256_or_si256(
        generate,
        _mm256_and_si256(propagate, shiftv_avx2::<-8, -1, 1, 8>(generate)),
    );
    propagate = _mm256_and_si256(propagate, shiftv_avx2::<-8, -1, 1, 8>(propagate));
    generate = _mm256_or_si256(
        generate,
        _mm256_and_si256(propagate, shiftv_avx2::<-16, -2, 2, 16>(generate)),
    );
    propagate = _mm256_and_si256(propagate, shiftv_avx2::<-16, -2, 2, 16>(propagate));
    generate = _mm256_or_si256(
        generate,
        _mm256_and_si256(propagate, shiftv_avx2::<-32, -4, 4, 32>(generate)),
    );

    let attacks = _mm256_and_si256(shiftv_avx2::<-8, -1, 1, 8>(generate), mask);
    fold_to_bitboard(attacks)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn rook_attacks_setwise(bb: Bitboard, occ: Bitboard) -> Bitboard {
    if std::is_x86_feature_detected!("avx2") {
        unsafe { rook_attacks_setwise_avx2(bb, occ) }
    } else {
        rook_attacks_setwise_scalar(bb, occ)
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn rook_attacks_setwise(bb: Bitboard, occ: Bitboard) -> Bitboard {
    rook_attacks_setwise_scalar(bb, occ)
}

// =============================================================================
// AVX2 internal helpers
// =============================================================================

/// 4-lane variable shift: each lane shifts by a different amount. Negative
/// amounts mean right-shift. Implemented as two variable shifts + OR.
///
/// On AVX-512+F+VL hosts a single `_mm256_rolv_epi64` (rotate-vector) does
/// the same in one instruction — see `shiftv_avx512` below. Reckless's
/// micro-opt pattern.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn shiftv_avx2<const A: i64, const B: i64, const C: i64, const D: i64>(
    v: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    _mm256_or_si256(
        _mm256_sllv_epi64(v, _mm256_set_epi64x(A, B, C, D)),
        _mm256_srlv_epi64(v, _mm256_set_epi64x(-A, -B, -C, -D)),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fold_to_bitboard(v: std::arch::x86_64::__m256i) -> Bitboard {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let merged = _mm_or_si128(lo, hi);
    let a = _mm_extract_epi64::<0>(merged) as u64;
    let b = _mm_extract_epi64::<1>(merged) as u64;
    a | b
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
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(
                    unsafe { knight_attacks_setwise_avx2(bb) },
                    aggregate_per_square_knight(bb),
                    "avx2 knight setwise mismatch on sq {}",
                    sq
                );
            }
        }
        // Multi-knight random cases.
        let mut r = rng(0xc0da_5e7_71_0003);
        for _ in 0..200 {
            let bb = r();
            let agg = aggregate_per_square_knight(bb);
            assert_eq!(knight_attacks_setwise_scalar(bb), agg);
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { knight_attacks_setwise_avx2(bb) }, agg);
            }
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
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(
                        unsafe { bishop_attacks_setwise_avx2(bb, occ | bb) },
                        agg,
                        "avx2 bishop sq={} occ={:016x}",
                        sq,
                        occ
                    );
                }
            }
        }
        // Multi-bishop random.
        let mut r = rng(0xc0da_5e7_71_0004);
        for _ in 0..200 {
            let bb = r();
            let occ = r() | bb;
            let agg = aggregate_per_square_bishop(bb, occ);
            assert_eq!(bishop_attacks_setwise_scalar(bb, occ), agg);
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { bishop_attacks_setwise_avx2(bb, occ) }, agg);
            }
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
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") {
                    assert_eq!(
                        unsafe { rook_attacks_setwise_avx2(bb, occ | bb) },
                        agg,
                        "avx2 rook sq={} occ={:016x}",
                        sq,
                        occ
                    );
                }
            }
        }
        let mut r = rng(0xc0da_5e7_71_0005);
        for _ in 0..200 {
            let bb = r();
            let occ = r() | bb;
            let agg = aggregate_per_square_rook(bb, occ);
            assert_eq!(rook_attacks_setwise_scalar(bb, occ), agg);
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { rook_attacks_setwise_avx2(bb, occ) }, agg);
            }
        }
    }
}
