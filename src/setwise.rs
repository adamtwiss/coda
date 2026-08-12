//! Setwise attack generation: compute attack bitboards for ALL pieces of one
//! type in a single batched operation, instead of per-square magic-bitboard
//! lookups.
//!
//! Used by `attacks_by_color`, `attackers_to`, `pinned`, threat enumeration —
//! anywhere the existing code does a `while bb != 0 { sq = bb.trailing_zeros();
//! attacks |= piece_attacks(sq, occ); bb &= bb-1; }` loop.
//!
//! Structure: pawns are pure scalar shifts; knights have a scalar
//! 8-direction shift fallback + an AVX2 path that packs 4 directions per
//! 256-bit register; bishops/rooks have a scalar magic-lookup loop fallback
//! + an AVX2 Kogge-Stone occluded fill.
//!
//! The AVX2 paths here were re-written in Coda's own code (see
//! `docs/license_compliance_review_2026-07-11.md`): the knight kernel is a
//! direct vectorisation of Coda's own scalar shift set below; the slider
//! kernels implement the public-domain Kogge-Stone parallel-prefix fill (Chess
//! Programming Wiki), with the edge masks derived from first principles. Every
//! AVX2 path is asserted byte-identical to Coda's scalar magic-lookup oracle
//! over all 64 squares × varied occupancies (see the tests), so search node
//! counts are unchanged whether or not a host runs the SIMD path.
//!
//! Correctness reference is the per-square magic-lookup behaviour — every
//! setwise impl is unit-tested against `attacks_by_color`-style aggregation
//! over the same bitboards. Integration is gated on perft passing.
//!
//! Why setwise wins: replaces N magic-bitboard lookups (each touching the
//! magic table cache lines for that piece type) with a small fixed number of
//! shift+mask+or operations. Per our Phase-2 perf data, Coda spends ~5% of
//! cycles in `attackers_to` /
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
// Knight attacks (scalar + AVX2)
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

/// AVX2 setwise knight attacks — a direct vectorisation of the scalar function
/// above: the four northbound jumps (+6/+15/+17/+10) share a left-shift, the
/// four southbound (−6/−15/−17/−10) share a right-shift, so each group packs
/// into one 256-bit register and resolves in a single per-lane variable shift
/// (`sllv`/`srlv`). The per-lane masks are exactly the scalar `!(...)` source
/// masks. Result = OR-reduce the eight lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn knight_attacks_setwise_avx2(bb: Bitboard) -> Bitboard {
    use std::arch::x86_64::*;
    // Northbound group: lanes [+6, +15, +17, +10] (set_epi64x is lane3..lane0).
    let up_mask = _mm256_set_epi64x(
        !(FILE_A | FILE_B | RANK_8) as i64, // +6
        !(FILE_A | RANK_7 | RANK_8) as i64, // +15
        !(FILE_H | RANK_7 | RANK_8) as i64, // +17
        !(FILE_G | FILE_H | RANK_8) as i64, // +10
    );
    let up_amt = _mm256_set_epi64x(6, 15, 17, 10);
    // Southbound group: lanes [-6, -15, -17, -10].
    let dn_mask = _mm256_set_epi64x(
        !(FILE_G | FILE_H | RANK_1) as i64, // -6
        !(FILE_H | RANK_1 | RANK_2) as i64, // -15
        !(FILE_A | RANK_1 | RANK_2) as i64, // -17
        !(FILE_A | FILE_B | RANK_1) as i64, // -10
    );
    let dn_amt = _mm256_set_epi64x(6, 15, 17, 10);

    let src = _mm256_set1_epi64x(bb as i64);
    let up = _mm256_sllv_epi64(_mm256_and_si256(src, up_mask), up_amt);
    let dn = _mm256_srlv_epi64(_mm256_and_si256(src, dn_mask), dn_amt);
    fold4_or(_mm256_or_si256(up, dn))
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

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub fn knight_attacks_setwise(bb: Bitboard) -> Bitboard {
    knight_attacks_setwise_scalar(bb)
}

// =============================================================================
// Bishop / rook attacks (scalar magic loop + AVX2 Kogge-Stone occluded fill)
// =============================================================================
//
// The AVX2 sliding-piece paths use the classic Kogge-Stone parallel-prefix
// occluded fill (public technique — see the Chess Programming Wiki
// "Kogge-Stone Algorithm"), packing a piece type's four ray directions into
// the four 64-bit lanes of a 256-bit register. Coda's own scalar magic-lookup
// loop below is the correctness oracle: every AVX2 path is asserted equal to
// it over all 64 squares × varied occupancies (see the tests).
//
// Occluded fill per direction d, with `pro` = empty squares confined to a
// direction that cannot wrap the board edge (`notwrap` mask below):
//   gen |= pro & shift_d(gen)      // reach 1 square
//   pro &= shift_d(pro)            // ... then 2, then 4 (doubling)
//   gen |= pro & shift_2d(gen)
//   pro &= shift_2d(pro)
//   gen |= pro & shift_4d(gen)
//   attacks = shift_d(gen) & notwrap   // one step past the frontier = first blocker/edge
// notwrap is the destination-file mask that kills horizontal wraparound: a
// west-moving ray must not land on the H-file, an east-moving ray not on the
// A-file; vertical rays never wrap.

/// Scalar setwise bishop attacks: per-square magic lookup loop. Output
/// bitboard includes attacked-by-our-own-blockers; callers mask if needed.
/// This is the correctness oracle for the AVX2 path.
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

/// AVX2 setwise bishop attacks: Kogge-Stone fill over the four diagonal
/// directions. Lane order (set_epi64x lane3..lane0): SW(−9), SE(−7), NW(+7),
/// NE(+9). notwrap masks: a westbound ray (SW/NW) can't land on the H-file, an
/// eastbound ray (SE/NE) can't land on the A-file.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bishop_attacks_setwise_avx2(bb: Bitboard, occ: Bitboard) -> Bitboard {
    use std::arch::x86_64::*;
    let notwrap = _mm256_set_epi64x(
        !FILE_H as i64, // SW (−9)
        !FILE_A as i64, // SE (−7)
        !FILE_H as i64, // NW (+7)
        !FILE_A as i64, // NE (+9)
    );
    let gen = _mm256_set1_epi64x(bb as i64);
    let pro0 = _mm256_and_si256(_mm256_set1_epi64x(!occ as i64), notwrap);
    let gen = kogge_stone_fill::<-9, -7, 7, 9, -18, -14, 14, 18, -36, -28, 28, 36>(gen, pro0);
    let attacks = _mm256_and_si256(shift4::<-9, -7, 7, 9>(gen), notwrap);
    fold4_or(attacks)
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

/// Scalar setwise rook attacks: per-square magic lookup loop. Correctness
/// oracle for the AVX2 path.
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

/// AVX2 setwise rook attacks: Kogge-Stone fill over the four orthogonal
/// directions. Lane order: S(−8), W(−1), E(+1), N(+8). Only the horizontal
/// rays need a notwrap mask (W can't land on H-file, E can't land on A-file);
/// vertical rays shift off the board harmlessly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rook_attacks_setwise_avx2(bb: Bitboard, occ: Bitboard) -> Bitboard {
    use std::arch::x86_64::*;
    let notwrap = _mm256_set_epi64x(
        !0i64,          // S (−8)
        !FILE_H as i64, // W (−1)
        !FILE_A as i64, // E (+1)
        !0i64,          // N (+8)
    );
    let gen = _mm256_set1_epi64x(bb as i64);
    let pro0 = _mm256_and_si256(_mm256_set1_epi64x(!occ as i64), notwrap);
    let gen = kogge_stone_fill::<-8, -1, 1, 8, -16, -2, 2, 16, -32, -4, 4, 32>(gen, pro0);
    let attacks = _mm256_and_si256(shift4::<-8, -1, 1, 8>(gen), notwrap);
    fold4_or(attacks)
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
// AVX2 primitives (Coda's own — standard SIMD idioms)
// =============================================================================

/// Per-lane signed shift of a 4×u64 register: lane i shifts left by `+amt` or
/// right by `-amt`. Implemented as `sllv | srlv` — a lane whose amount is
/// positive gets a >= 64 right-count (→ 0) and vice-versa, so exactly one of
/// the two shifts is live per lane. Amounts are const so the count vectors are
/// materialised as immediates.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn shift4<const A: i64, const B: i64, const C: i64, const D: i64>(
    v: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    const fn lc(a: i64) -> i64 { if a > 0 { a } else { 64 } } // left (sll) count
    const fn rc(a: i64) -> i64 { if a < 0 { -a } else { 64 } } // right (srl) count
    let left = _mm256_set_epi64x(lc(A), lc(B), lc(C), lc(D));
    let right = _mm256_set_epi64x(rc(A), rc(B), rc(C), rc(D));
    _mm256_or_si256(_mm256_sllv_epi64(v, left), _mm256_srlv_epi64(v, right))
}

/// Kogge-Stone occluded fill over four lanes, three doubling steps
/// (distances d, 2d, 4d — covers a full 8×8 ray). `pro` must already be the
/// edge-masked empty set. Returns the filled `gen` (piece squares + empty ray
/// squares up to, not including, the first blocker).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn kogge_stone_fill<
    const A1: i64, const B1: i64, const C1: i64, const D1: i64,
    const A2: i64, const B2: i64, const C2: i64, const D2: i64,
    const A4: i64, const B4: i64, const C4: i64, const D4: i64,
>(
    mut gen: std::arch::x86_64::__m256i,
    mut pro: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    gen = _mm256_or_si256(gen, _mm256_and_si256(pro, shift4::<A1, B1, C1, D1>(gen)));
    pro = _mm256_and_si256(pro, shift4::<A1, B1, C1, D1>(pro));
    gen = _mm256_or_si256(gen, _mm256_and_si256(pro, shift4::<A2, B2, C2, D2>(gen)));
    pro = _mm256_and_si256(pro, shift4::<A2, B2, C2, D2>(pro));
    gen = _mm256_or_si256(gen, _mm256_and_si256(pro, shift4::<A4, B4, C4, D4>(gen)));
    gen
}

/// OR-reduce the four 64-bit lanes of a 256-bit register into one u64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fold4_or(v: std::arch::x86_64::__m256i) -> Bitboard {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let m = _mm_or_si128(lo, hi);
    (_mm_extract_epi64::<0>(m) as u64) | (_mm_extract_epi64::<1>(m) as u64)
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
            let agg = aggregate_per_square_knight(bb);
            assert_eq!(knight_attacks_setwise_scalar(bb), agg,
                "scalar knight setwise mismatch on sq {}", sq);
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { knight_attacks_setwise_avx2(bb) }, agg,
                    "avx2 knight setwise mismatch on sq {}", sq);
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
                assert_eq!(unsafe { knight_attacks_setwise_avx2(bb) }, agg,
                    "avx2 knight setwise mismatch on bb {:016x}", bb);
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
                    assert_eq!(unsafe { bishop_attacks_setwise_avx2(bb, occ | bb) }, agg,
                        "avx2 bishop sq={} occ={:016x}", sq, occ);
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
                assert_eq!(unsafe { bishop_attacks_setwise_avx2(bb, occ) }, agg,
                    "avx2 bishop bb={:016x} occ={:016x}", bb, occ);
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
                    assert_eq!(unsafe { rook_attacks_setwise_avx2(bb, occ | bb) }, agg,
                        "avx2 rook sq={} occ={:016x}", sq, occ);
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
                assert_eq!(unsafe { rook_attacks_setwise_avx2(bb, occ) }, agg,
                    "avx2 rook bb={:016x} occ={:016x}", bb, occ);
            }
        }
    }
}
