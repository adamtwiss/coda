//! AVX-512 byteboard-splat threat-delta enumeration with Coda x-ray
//! semantics (Phase A of the threat-pipeline campaign).
//!
//! The SIMD *primitives* (ray permutation, closest-on-rays carry trick,
//! compress-store emission) are modelled on Reckless's
//! `nnue/accumulator/threats/vectorized` machinery, but the ENUMERATION is
//! Coda's own: the feature space encodes direct-OR-depth-1-x-ray at one
//! index (contract §1.1/§1.3, `docs/threat_semantics_contract_2026-07-04.md`),
//! so a direct-only splat double-counts (the May 2026 blocker,
//! `docs/byteboard_splat_scoping_2026-05-03.md`). Instead, everything is
//! derived from ONE focus-square ray frame with TWO hits per ray
//! (`docs/threat_splat_phase_a_2026-07-04.md` §Design): Y_d = first
//! occupant on ray direction d, Z_d = second. All five contract sections
//! read their emissions from the (Y_d, Z_d) pairs — see `splat_focus`.
//!
//! This module only changes how the `RawThreatDelta` list is built; the
//! weight-row apply still goes through `threats::apply_threat_deltas`.
//! Dispatch (CODA_SPLAT=1 + CPU checks, default scalar) lives in
//! `threats::push_threats_on_move` / `push_threats_on_change`.
//!
//! Phase A: AVX-512 F+BW+VBMI+VBMI2 path (Zen 4+/SPR+).
//! Phase B (this file too, second half): AVX2 fallback with the SAME
//! enumeration — 2×__m256i primitives modelled on Reckless's
//! `vectorized/avx2.rs` techniques, scalar mask-drain emission. Most fleet
//! hosts (Hercules Coffee Lake, Atlas Zen 1) only do AVX2; AVX-512-only
//! would shift fleet results away from older hosts which gain the most
//! from cache work. Dispatch: AVX-512 if the CPU qualifies, else AVX2,
//! else scalar (threats.rs `use_splat`); `CODA_SPLAT_FORCE_AVX2=1` pins
//! the AVX2 path on AVX-512 hosts for on-host A/B measurement.
//!
//! Encoding decisions:
//! - **Piece IDs**: Coda's `colored_piece()` uses (color × 6 + piece_type),
//!   giving WP=0 WN=1 WB=2 WR=3 WQ=4 WK=5 BP=6 BN=7 ... BK=11, NO_PIECE=12.
//!   Reckless interleaves: WP=0 BP=1 WN=2 BN=3 ... BK=11, None=12. Tables
//!   below are RE-DERIVED in Coda's encoding so the SIMD pipeline reads
//!   Coda's mailbox bytes directly without translation.
//! - **RawThreatDelta layout**: Coda's `RawThreatDelta(u32)` has the same
//!   bit layout as Reckless's `ThreatDelta(u32)` — byte 0 attacker_cp,
//!   byte 1 from_sq, byte 2 victim_cp, byte 3 = to_sq | (add << 31). So
//!   the splat can write Coda RawThreatDelta values directly into the
//!   `Vec<RawThreatDelta>` buffer.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

/// Per-focus-square byte permutation that arranges the mailbox into
/// "ray order" — 8 rays × 8 positions per ray = 64 bytes per focus.
/// 0x80 marks invalid (off-board) positions.
///
/// Computed via the same const-fn as Reckless's
/// `nnue/accumulator/threats/vectorized.rs:17`. Piece-encoding-INDEPENDENT
/// (purely spatial), so we copy the construction verbatim.
pub(crate) const RAY_PERMUTATIONS: [[u8; 64]; 64] = {
    const OFFSETS: [u8; 64] = [
        0x1F, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, // N
        0x21, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, // NE
        0x12, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // E
        0xF2, 0xF1, 0xE2, 0xD3, 0xC4, 0xB5, 0xA6, 0x97, // SE
        0xE1, 0xF0, 0xE0, 0xD0, 0xC0, 0xB0, 0xA0, 0x90, // S
        0xDF, 0xEF, 0xDE, 0xCD, 0xBC, 0xAB, 0x9A, 0x89, // SW
        0xEE, 0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, // W
        0x0E, 0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, // NW
    ];

    let mut perms = [[0u8; 64]; 64];
    let mut sq = 0;
    while sq < 64 {
        let focus = sq as u8;
        let focus = focus + (focus & 0x38);
        let mut i = 0;
        while i < 64 {
            let wide_result = OFFSETS[i].wrapping_add(focus);
            let valid = wide_result & 0x88 == 0;
            let narrow_result = ((wide_result & 0x70) >> 1) + (wide_result & 0x07);
            perms[sq][i] = if valid { narrow_result } else { 0x80 };
            i += 1;
        }
        sq += 1;
    }
    perms
};

/// Per-ray-position attacker-class bitmask. 64 entries (8 rays × 8 positions).
/// Position 0 = adjacent to focus; position 7 = far end.
/// Piece-encoding-INDEPENDENT — same as Reckless.
///
/// Bits: knight=0b00000100, rook+queen=0b00110000, bishop+queen=0b00101000,
/// king+rook+queen=0b01110000, wpawn-near=0b01101001 (wp+king+bishop+queen),
/// bpawn-near=0b01101010 (bp+king+bishop+queen).
pub(crate) const RAY_ATTACKERS_MASK: [u8; 64] = {
    let horse = 0b00000100;       // knight
    let orth = 0b00110000;        // rook + queen
    let diag = 0b00101000;        // bishop + queen
    let ortho_near = 0b01110000;  // king + rook + queen
    let wpawn_near = 0b01101001;  // wp + king + bishop + queen
    let bpawn_near = 0b01101010;  // bp + king + bishop + queen

    [
        horse, ortho_near, orth, orth, orth, orth, orth, orth, // N
        horse, bpawn_near, diag, diag, diag, diag, diag, diag, // NE
        horse, ortho_near, orth, orth, orth, orth, orth, orth, // E
        horse, wpawn_near, diag, diag, diag, diag, diag, diag, // SE
        horse, ortho_near, orth, orth, orth, orth, orth, orth, // S
        horse, wpawn_near, diag, diag, diag, diag, diag, diag, // SW
        horse, ortho_near, orth, orth, orth, orth, orth, orth, // W
        horse, bpawn_near, diag, diag, diag, diag, diag, diag, // NW
    ]
};

/// Per-ray-position slider-class bitmask (rook=0b00010000, queen=0b00100000,
/// bishop=0b00001000). Position 0 = adjacent (no sliders — only ortho-near
/// king/queen/rook), so 0x80 = invalid.
/// Piece-encoding-INDEPENDENT — same as Reckless.
pub(crate) const RAY_SLIDERS_MASK: [u8; 64] = {
    let orth = 0b00110000;
    let diag = 0b00101000;

    [
        0x80, orth, orth, orth, orth, orth, orth, orth, // N
        0x80, diag, diag, diag, diag, diag, diag, diag, // NE
        0x80, orth, orth, orth, orth, orth, orth, orth, // E
        0x80, diag, diag, diag, diag, diag, diag, diag, // SE
        0x80, orth, orth, orth, orth, orth, orth, orth, // S
        0x80, diag, diag, diag, diag, diag, diag, diag, // SW
        0x80, orth, orth, orth, orth, orth, orth, orth, // W
        0x80, diag, diag, diag, diag, diag, diag, diag, // NW
    ]
};

/// Per-piece attacker-class bit. 16 entries (4-bit indexed). RE-DERIVED for
/// Coda's `colored_piece` encoding (was Reckless's interleaved order).
///
/// Bit layout: see `RAY_ATTACKERS_MASK` comment. WP=0b01, BP=0b10 (need
/// separate bits because pawn attack direction depends on color); knights
/// share 0b100 across colors (attacks are color-symmetric); same for
/// bishops/rooks/queens/kings.
pub(crate) const PIECE_TO_BIT_TABLE: [u8; 16] = [
    0b00000001, // 0  WhitePawn
    0b00000100, // 1  WhiteKnight
    0b00001000, // 2  WhiteBishop
    0b00010000, // 3  WhiteRook
    0b00100000, // 4  WhiteQueen
    0b01000000, // 5  WhiteKing
    0b00000010, // 6  BlackPawn
    0b00000100, // 7  BlackKnight
    0b00001000, // 8  BlackBishop
    0b00010000, // 9  BlackRook
    0b00100000, // 10 BlackQueen
    0b01000000, // 11 BlackKing
    0,          // 12 NO_PIECE / empty
    0,          // 13 unused
    0,          // 14 unused
    0,          // 15 unused
];

/// Per-piece ray-position attack mask. 12 entries indexed by Coda's
/// `colored_piece` (0..11). Each u64 packs 8 attack-position bytes
/// (one per ray direction); a set bit at position N means this piece
/// attacks ray-position N.
///
/// RE-DERIVED for Coda's encoding (Reckless interleaves W/B; Coda groups
/// W then B). Knights/bishops/rooks/queens/kings are color-symmetric so
/// W and B variants are identical; only pawns differ (WP attacks position
/// 1 NE/NW = adjacent diagonals forward; BP attacks position 1 SE/SW).
pub(crate) const RAY_ATTACKS_MASK: [u64; 12] = [
    0x02_00_00_00_00_00_02_00, // 0  WhitePawn  (NE+NW position 1)
    0x01_01_01_01_01_01_01_01, // 1  WhiteKnight (knight pos all rays)
    0xFE_00_FE_00_FE_00_FE_00, // 2  WhiteBishop (diagonals all positions)
    0x00_FE_00_FE_00_FE_00_FE, // 3  WhiteRook   (orthogonals all positions)
    0xFE_FE_FE_FE_FE_FE_FE_FE, // 4  WhiteQueen  (all rays all positions)
    0x02_02_02_02_02_02_02_02, // 5  WhiteKing   (position 1 all rays)
    0x00_00_02_00_02_00_00_00, // 6  BlackPawn   (SE+SW position 1)
    0x01_01_01_01_01_01_01_01, // 7  BlackKnight
    0xFE_00_FE_00_FE_00_FE_00, // 8  BlackBishop
    0x00_FE_00_FE_00_FE_00_FE, // 9  BlackRook
    0xFE_FE_FE_FE_FE_FE_FE_FE, // 10 BlackQueen
    0x02_02_02_02_02_02_02_02, // 11 BlackKing
];

// =============================================================================
// AVX-512 + VBMI2 SIMD primitives (port of vectorized/avx512.rs)
// =============================================================================

/// Load Coda's piece-type mailbox + white-occupancy bitboard and convert to
/// a `colored_piece` SIMD vector (one byte per square, 0..11 for pieces,
/// 12 for empty).
///
/// Reckless's mailbox encoding is colored_piece directly; Coda's mailbox
/// stores `piece_type` (0..5) + `NO_PIECE_TYPE = 6` for empty, with color
/// derived from a separate `colors[]` bitboard. This helper bridges the
/// two — call once at entry to push_threats_on_*, then the rest of the
/// pipeline runs on a Reckless-shape colored mailbox vector.
///
/// Mapping:
///   type=6 (empty)             → 12 (NO_PIECE)
///   type=0..5, square in white → keep as 0..5 (white pieces)
///   type=0..5, square not white → add 6 → 6..11 (black pieces)
///
/// Implementation: add 6 to every byte where the white_bb bit is NOT set
/// (i.e. black piece OR empty); empty (6) becomes 12 ✓; black piece
/// (0..5) becomes 6..11 ✓; white piece keeps original 0..5 ✓.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn mailbox_vector_avx512(mailbox: &[u8; 64], white_bb: u64) -> __m512i {
    let board_types = _mm512_loadu_si512(mailbox.as_ptr().cast());
    let not_white_mask = !white_bb;
    let six = _mm512_set1_epi8(6);
    _mm512_mask_add_epi8(board_types, not_white_mask, board_types, six)
}

/// Compute the byte-permutation + valid-mask for a focus square's rays.
/// Returns `(perm, valid_mask)` where `valid_mask` has a bit set per
/// on-board ray-position (0x80 entries are masked out).
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ray_permutation(focus: u32) -> (__m512i, u64) {
    let perm = _mm512_loadu_si512(RAY_PERMUTATIONS.get_unchecked(focus as usize).as_ptr().cast());
    let mask = _mm512_testn_epi8_mask(perm, _mm512_set1_epi8(0x80u8 as i8));
    (perm, mask)
}

/// Permute the mailbox into ray order and produce per-position attacker-
/// class bytes. Returns `(pboard, rays)` — pboard is the ray-permuted
/// mailbox (piece values per ray-position), rays is the attacker-class
/// bytes (PIECE_TO_BIT_TABLE values per ray-position).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn board_to_rays(perm: __m512i, valid: u64, board: __m512i) -> (__m512i, __m512i) {
    let lut = _mm_loadu_si128(PIECE_TO_BIT_TABLE.as_ptr().cast());
    let pboard = _mm512_permutexvar_epi8(perm, board);
    let rays = _mm512_maskz_shuffle_epi8(valid, _mm512_broadcast_i32x4(lut), pboard);
    (pboard, rays)
}

/// Per-octet mask of true ray positions (distance 1..7). Bit 0 of each
/// octet is the knight position — not on the ray.
pub(crate) const RAY_POSITIONS: u64 = 0xFEFEFEFEFEFEFEFE;

/// Carry trick: lowest set bit per octet of `occupied`, PLUS bit 0 of any
/// octet whose bit 0 is set (the subtract-3 always flips bit 0, so a set
/// knight-position bit survives the `& occupied`). Invariants: per octet,
/// sentinel bits 0+7 (0x81) confine the borrow — no cross-octet carries.
///
/// Callers: `closest_on_rays` passes raw occupancy (knight ring wanted for
/// §1 knight attacks / §3 knight attackers); the second-hit pass in
/// `splat_focus` pre-masks with `RAY_POSITIONS` so only true ray occupants
/// participate.
#[inline(always)]
pub(crate) const fn closest_from_occupied(occupied: u64) -> u64 {
    let o = occupied | 0x8181818181818181;
    let x = o ^ (o - 0x0303030303030303);
    x & occupied
}

/// Mask of ray-positions that are the closest occupied square along their
/// ray (per-ray first hit, Y_d), plus occupied knight-position bits.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn closest_on_rays(rays: __m512i) -> u64 {
    closest_from_occupied(_mm512_test_epi8_mask(rays, rays))
}

/// Fill each octet that has any set bit to a full-octet mask. Used to
/// intersect two per-ray masks down to the octets where BOTH have a bit,
/// which keeps `splat_xray_threats`' compress-pairing aligned (per octet
/// ≤1 bit per operand, k-th set bit of one mask pairs with k-th of the
/// other). Input bits must be at positions 1..7 (bit 0 would not carry
/// into bit 7). The final `|x` restores bit 7 itself: without it a bit at
/// position 7 fills only 0x7F and would drop its own octet's bit — that
/// position is geometrically unreachable in the pairing calls (distance-7
/// hit in direction d puts the focus on the board edge in direction -d,
/// so the opposite octet is empty), but harden rather than rely on it.
#[inline(always)]
pub(crate) const fn ray_fill(x: u64) -> u64 {
    let x = (x + 0x7E7E7E7E7E7E7E7E) & 0x8080808080808080;
    x | (x - (x >> 7))
}

/// Mask of ray-positions that have an attacker (any class) firing toward
/// the focus square — bit set if RAY_ATTACKERS_MASK[pos] AND-mask hits the
/// attacker-class bits at that ray-position.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn attackers_along_rays(rays: __m512i) -> u64 {
    let mask = _mm512_loadu_si512(RAY_ATTACKERS_MASK.as_ptr().cast());
    _mm512_test_epi8_mask(rays, mask)
}

/// Mask of ray-positions where the focus piece (piece) attacks an occupied
/// square. Just `RAY_ATTACKS_MASK[piece] & occupied` — fully scalar.
#[inline(always)]
pub(crate) fn attacking_along_rays(piece_cp: u8, occupied: u64) -> u64 {
    RAY_ATTACKS_MASK[piece_cp as usize] & occupied
}

/// Mask of ray-positions that have a slider (rook/bishop/queen) at the
/// closest-occupied position along their ray. Bottom byte (position 0,
/// adjacent) excluded since 0x80 there marks "no sliders adjacent."
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn sliders_along_rays(rays: __m512i) -> u64 {
    let mask = _mm512_loadu_si512(RAY_SLIDERS_MASK.as_ptr().cast());
    _mm512_test_epi8_mask(rays, mask) & 0xFEFEFEFEFEFEFEFE
}

/// Colored mailbox vector masked to the CALLER's occupancy: squares whose
/// occ bit is clear read as NO_PIECE regardless of mailbox content.
///
/// This enforces contract invariant 6 (occ_transit discipline): during
/// `push_threats_on_move` the mover sits in mailbox/pieces_bb at `to`
/// while occ_transit has `to` clear — every candidate read must respect
/// the passed occ, not the board arrays. Masking the board vector once at
/// entry makes every downstream ray read occ-correct by construction.
/// Precondition (holds at every §3.1 call site): occ ⊆ mailbox-occupied
/// squares, except possibly the focus square (whose mailbox byte is never
/// read — the focus is not in its own ray frame).
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn occ_masked_board(mailbox: &[u8; 64], white_bb: u64, occ: u64) -> __m512i {
    let board = mailbox_vector_avx512(mailbox, white_bb);
    _mm512_mask_blend_epi8(occ, _mm512_set1_epi8(crate::types::NO_PIECE as i8), board)
}

// =============================================================================
// `splat_threats` / `splat_xray_threats` — emit RawThreatDelta records
// =============================================================================

use crate::threats::RawThreatDelta;

/// Emit threat-delta records into `deltas` for all attackers/attackeds along
/// the focus square's rays. Mirrors Reckless's `splat_threats`.
///
/// Output bit layout per RawThreatDelta (matches Reckless's ThreatDelta):
///   byte 0: attacker piece-cp
///   byte 1: attacker square (physical)
///   byte 2: victim piece-cp
///   byte 3 (low 7 bits): victim square; bit 7 = `add` flag
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
pub(crate) unsafe fn splat_threats(
    deltas: &mut Vec<RawThreatDelta>,
    pboard: __m512i,
    perm: __m512i,
    attacked: u64,
    attackers: u64,
    focus_piece: u8,
    focus_sq: u32,
    add: bool,
) {
    let add_bit = (add as u32) << 31;
    let add_v = _mm512_set1_epi32(add_bit as i32);

    // focus_pair packs (piece in byte 0, sq in byte 1) into a u16 broadcast
    // across all 32 i16 lanes.
    let focus_pair = {
        let pair = focus_piece as u16 | ((focus_sq as u16) << 8);
        _mm512_set1_epi16(pair as i16)
    };

    // Compress attacker-class bytes from pboard at masked positions.
    let attacked_pieces = _mm512_castsi512_si256(_mm512_maskz_compress_epi8(attacked, pboard));
    let attacked_squares = _mm512_castsi512_si256(_mm512_maskz_compress_epi8(attacked, perm));
    let attackers_pieces = _mm512_maskz_compress_epi8(attackers, pboard);
    let attackers_squares = _mm512_maskz_compress_epi8(attackers, perm);

    // Index permutation: interleave (piece, sq) bytes into 4-byte
    // RawThreatDelta records.
    let attacked_idx = _mm256_set_epi8(
        39, 7, 39, 7, 38, 6, 38, 6, 37, 5, 37, 5, 36, 4, 36, 4,
        35, 3, 35, 3, 34, 2, 34, 2, 33, 1, 33, 1, 32, 0, 32, 0,
    );
    let attackers_idx = _mm512_set_epi8(
        79, 15, 79, 15, 78, 14, 78, 14, 77, 13, 77, 13, 76, 12, 76, 12,
        75, 11, 75, 11, 74, 10, 74, 10, 73, 9, 73, 9, 72, 8, 72, 8,
        71, 7, 71, 7, 70, 6, 70, 6, 69, 5, 69, 5, 68, 4, 68, 4,
        67, 3, 67, 3, 66, 2, 66, 2, 65, 1, 65, 1, 64, 0, 64, 0,
    );

    let attacked_pairs = _mm256_permutex2var_epi8(attacked_pieces, attacked_idx, attacked_squares);
    let attackers_pairs = _mm512_permutex2var_epi8(attackers_pieces, attackers_idx, attackers_squares);

    // For attacked: focus_piece is the ATTACKER; pair-bytes are VICTIM.
    // For attackers: focus_piece is the VICTIM; pair-bytes are ATTACKER.
    let attacked_vector = _mm256_or_si256(
        _mm256_mask_mov_epi8(_mm512_castsi512_si256(focus_pair), 0xCCCCCCCC, attacked_pairs),
        _mm512_castsi512_si256(add_v),
    );
    let attackers_vector =
        _mm512_or_si512(_mm512_mask_mov_epi8(focus_pair, 0x3333333333333333, attackers_pairs), add_v);

    // Append into deltas Vec via raw pointer writes. The stores below write
    // FULL vector widths (8 records for attacked, 16 for attackers) into
    // reserved-but-unused capacity; only `total` records become visible via
    // set_len. Reserve the maximum store EXTENT (attacked_count + 16), not
    // just `total`, or the trailing lanes scribble past the allocation.
    let attacked_count = attacked.count_ones() as usize;
    let attackers_count = attackers.count_ones() as usize;
    debug_assert!(attacked_count <= 8 && attackers_count <= 16);
    let total = attacked_count + attackers_count;
    deltas.reserve(attacked_count + 16);
    let len_before = deltas.len();
    let dst = deltas.as_mut_ptr().add(len_before) as *mut __m256i;
    _mm256_storeu_si256(dst, attacked_vector);
    let dst = (deltas.as_mut_ptr().add(len_before + attacked_count)) as *mut __m512i;
    _mm512_storeu_si512(dst, attackers_vector);
    deltas.set_len(len_before + total);
}

/// Emit x-ray threat-delta records.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
pub(crate) unsafe fn splat_xray_threats(
    deltas: &mut Vec<RawThreatDelta>,
    pboard: __m512i,
    perm: __m512i,
    sliders: u64,
    victim_mask: u64,
    add: bool,
) {
    let add_bit = (add as u32) << 31;
    let add_v = _mm_set1_epi32(add_bit as i32);

    let flip_rays = |x: __m512i| -> __m512i { _mm512_shuffle_i64x2::<0b01_00_11_10>(x, x) };
    let compress = |m: u64, v: __m512i| -> __m128i {
        _mm512_castsi512_si128(_mm512_maskz_compress_epi8(m, v))
    };

    let p1 = compress(sliders, pboard);
    let sq1 = compress(sliders, perm);
    let p2 = compress(victim_mask, flip_rays(pboard));
    let sq2 = compress(victim_mask, flip_rays(perm));

    let pair1 = _mm_unpacklo_epi8(p1, sq1);
    let pair2 = _mm_unpacklo_epi8(p2, sq2);

    debug_assert_eq!(sliders.count_ones(), victim_mask.count_ones());
    let count = sliders.count_ones() as usize;
    debug_assert!(count <= 8);
    // Both stores below write full 128-bit widths (8 records total extent)
    // regardless of `count` — reserve the extent, not just `count`.
    deltas.reserve(8);
    let len_before = deltas.len();
    let dst = deltas.as_mut_ptr().add(len_before) as *mut __m128i;
    _mm_storeu_si128(dst, _mm_or_si128(_mm_unpacklo_epi16(pair1, pair2), add_v));
    let dst = deltas.as_mut_ptr().add(len_before + 4) as *mut __m128i;
    _mm_storeu_si128(dst, _mm_or_si128(_mm_unpackhi_epi16(pair1, pair2), add_v));
    deltas.set_len(len_before + count);
}

// =============================================================================
// The unified x-ray-aware enumerator
// =============================================================================

/// One focus-square pass: emit all five contract sections
/// (`docs/threat_semantics_contract_2026-07-04.md` §3.3) for the piece
/// `piece_cp` appearing (add) / disappearing (!add) on `square`, given the
/// occ-masked board vector.
///
/// Everything derives from TWO hits per ray direction d:
///   Y_d = first occupant, Z_d = second occupant (first-hit pass rerun
///   with the first hits masked out). Section reads:
/// - §1  direct-from:  Y_d on rays the focus class attacks (+ knight ring).
/// - §1b own-x-ray:    Z_d on the same rays. `RAY_ATTACKS_MASK & second`
///   enforces §1b's slider-only + class-alignment by construction: Z_d
///   sits at distance ≥2, where only slider octet masks have bits (pawn/
///   king masks cover distance 1 only, the knight mask position 0 only).
/// - §2  sliders-see:  S = Y_d of slider class aligned with d. Direct emit
///   (S→focus, add) rides the §3 attackers mask; the Z-delta victim is
///   Z_{-d} (first-past-focus is Y_{-d}, next is Z_{-d}), sign !add.
/// - §2b x-ray-onto-focus: S = Z_d of slider class aligned with d; the
///   single blocker Y_d exists by construction (second implies first) and
///   is deliberately unfiltered (blocker may be ANY piece, §1.3). Focus
///   emit (S→focus, add) + W-delta victim Y_{-d}, sign !add.
/// - §3  non-slider attackers: RAY_ATTACKERS_MASK hits at Y_d / knight
///   ring (shared mask with §2's direct emit).
///
/// Emission is raw physical tuples; pair-map filtering and semi-exclusion
/// stay at index-expansion time (contract invariants 2-4).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
unsafe fn splat_focus(
    deltas: &mut Vec<RawThreatDelta>,
    board: __m512i,
    piece_cp: u8,
    square: u32,
    add: bool,
) {
    let (perm, valid) = ray_permutation(square);
    let (pboard, rays) = board_to_rays(perm, valid, board);

    let closest = closest_on_rays(rays); // Y_d ∪ occupied knight ring
    let first = closest & RAY_POSITIONS; // Y_d
    let ray_occ = _mm512_test_epi8_mask(rays, rays) & RAY_POSITIONS;
    let second = closest_from_occupied(ray_occ & !first); // Z_d

    // §1 direct-from + (§2-direct ∪ §3) incoming direct attackers.
    let attacked_direct = attacking_along_rays(piece_cp, closest);
    let attackers_direct = attackers_along_rays(rays) & closest;
    splat_threats(deltas, pboard, perm, attacked_direct, attackers_direct, piece_cp, square, add);

    // §1b x-rays FROM the focus slider + §2b sliders whose x-ray target is
    // the focus square (their S→focus emissions carry the same sign `add`).
    let attacked_xray = attacking_along_rays(piece_cp, second);
    let slider_hits = sliders_along_rays(rays);
    let xray_attackers = slider_hits & second;
    if attacked_xray | xray_attackers != 0 {
        splat_threats(deltas, pboard, perm, attacked_xray, xray_attackers, piece_cp, square, add);
    }

    // §2 Z-deltas: seeing slider S = Y_d loses/gains its depth-1 x-ray to
    // Z_{-d} when the focus piece appears/disappears (S's feature for
    // Y_{-d} is index-unchanged — direct ↔ x-ray). Sign INVERTED (!add).
    let seers = slider_hits & first;
    let z_opp = second.rotate_right(32); // Z_{-d}, expressed in octet d
    let z_pairs = ray_fill(seers) & ray_fill(z_opp);
    if z_pairs != 0 {
        splat_xray_threats(deltas, pboard, perm, seers & z_pairs, z_opp & z_pairs, !add);
    }

    // §2b W-deltas: x-raying slider S = Z_d shifts its depth-1 victim
    // between the focus square and W = Y_{-d}. Sign INVERTED (!add).
    let y_opp = first.rotate_right(32); // Y_{-d}, expressed in octet d
    let w_pairs = ray_fill(xray_attackers) & ray_fill(y_opp);
    if w_pairs != 0 {
        splat_xray_threats(deltas, pboard, perm, xray_attackers & w_pairs, y_opp & w_pairs, !add);
    }
}

// =============================================================================
// High-level entry points (signatures mirror the scalar §3.2 entry points)
// =============================================================================

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
pub unsafe fn push_threats_on_change_avx512(
    deltas: &mut Vec<RawThreatDelta>,
    mailbox: &[u8; 64],
    white_bb: u64,
    occ: u64,
    piece_cp: u8,
    square: u32,
    add: bool,
) {
    let board = occ_masked_board(mailbox, white_bb, occ);
    splat_focus(deltas, board, piece_cp, square, add);
}

/// `occ` is the post-move_piece occupancy exactly as the scalar entry point
/// receives it (`from` clear, `to` set); occ_transit clears `to` so BOTH
/// legs run with the mover absent from occupancy (contract §3.2).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
pub unsafe fn push_threats_on_move_avx512(
    deltas: &mut Vec<RawThreatDelta>,
    mailbox: &[u8; 64],
    white_bb: u64,
    occ: u64,
    piece_cp: u8,
    src: u32,
    dst: u32,
) {
    let occ_transit = occ ^ (1u64 << dst);
    let board = occ_masked_board(mailbox, white_bb, occ_transit);
    splat_focus(deltas, board, piece_cp, src, false);
    splat_focus(deltas, board, piece_cp, dst, true);
}

// =============================================================================
// AVX2 fallback (Phase B) — same enumeration, 2×__m256i primitives
// =============================================================================
//
// The section logic is IDENTICAL to `splat_focus` (all of it operates on
// u64 ray-position masks and the shared scalar helpers
// `closest_from_occupied` / `ray_fill` / `attacking_along_rays`); only the
// vector primitives change width, and emission drains masks scalar instead
// of vpcompressb. The SIMD techniques (64-byte cross-lane gather via
// permute2x128 + shuffle_epi8 + blendv, movemask-based ray masks, scalar
// mask-drain emission) are modelled on Reckless's AVX2 threat machinery
// (`nnue/accumulator/threats/vectorized/avx2.rs`).
//
// Register budget note (Zen 1 cracks 256-bit ops to 2×128 with tight PRF
// budgets — 2026-07-04 scorecard): the hot gather keeps ~8 live YMMs and
// `splat_focus_avx2` spills pboard/perm to two stack arrays immediately
// after the gather, so the emission phase holds no vector registers at all.

#[target_feature(enable = "avx2")]
unsafe fn loadu2(ptr: *const __m256i) -> [__m256i; 2] {
    [_mm256_loadu_si256(ptr), _mm256_loadu_si256(ptr.add(1))]
}

/// Byte-wise "== 0?" over both halves, inverted: bit i of the result is set
/// iff byte i is NONZERO. The AVX2 stand-in for `_mm512_test_epi8_mask(v, m)`
/// callers (they pre-AND with the mask).
#[target_feature(enable = "avx2")]
unsafe fn nonzero_bytes_mask(v: [__m256i; 2]) -> u64 {
    let z = _mm256_setzero_si256();
    let lo = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v[0], z)) as u32 as u64;
    let hi = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v[1], z)) as u32 as u64;
    !(lo | (hi << 32))
}

/// Expand a 32-square bitmask into a byte-lane mask (0xFF where the bit is
/// set). AVX2 has no mask registers; this is the standard broadcast +
/// per-byte-bit test. Used to synthesize the AVX-512 masked add/blend ops.
#[target_feature(enable = "avx2")]
unsafe fn mask_bytes_avx2(mask: u32) -> __m256i {
    let v = _mm256_set1_epi32(mask as i32);
    // Byte j of each 128-bit lane must read mask byte j/8 (shuffle_epi8 is
    // per-lane, but set1 replicated the mask into every lane).
    #[rustfmt::skip]
    let sel = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    );
    #[rustfmt::skip]
    let bits = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    );
    let picked = _mm256_and_si256(_mm256_shuffle_epi8(v, sel), bits);
    _mm256_cmpeq_epi8(picked, bits)
}

/// AVX2 analogue of `mailbox_vector_avx512`: piece-type mailbox + white
/// bitboard → colored-piece byte vector (add 6 where NOT white; empty 6→12).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn mailbox_vector_avx2(mailbox: &[u8; 64], white_bb: u64) -> [__m256i; 2] {
    let types = loadu2(mailbox.as_ptr().cast());
    let six = _mm256_set1_epi8(6);
    let not_white = !white_bb;
    [
        _mm256_add_epi8(types[0], _mm256_and_si256(six, mask_bytes_avx2(not_white as u32))),
        _mm256_add_epi8(types[1], _mm256_and_si256(six, mask_bytes_avx2((not_white >> 32) as u32))),
    ]
}

/// AVX2 analogue of `occ_masked_board`: squares whose occ bit is clear read
/// as NO_PIECE (contract invariant 6 — occ_transit discipline; see the
/// AVX-512 version's doc for the full rationale).
#[target_feature(enable = "avx2")]
unsafe fn occ_masked_board_avx2(mailbox: &[u8; 64], white_bb: u64, occ: u64) -> [__m256i; 2] {
    let board = mailbox_vector_avx2(mailbox, white_bb);
    let none = _mm256_set1_epi8(crate::types::NO_PIECE as i8);
    [
        _mm256_blendv_epi8(none, board[0], mask_bytes_avx2(occ as u32)),
        _mm256_blendv_epi8(none, board[1], mask_bytes_avx2((occ >> 32) as u32)),
    ]
}

/// AVX2 analogue of `ray_permutation`: returns `(perm, invalid)` where
/// `invalid` is a byte-lane mask (0xFF at off-board 0x80 entries) — AVX2 has
/// no mask registers, so invalidity travels as a vector and is applied with
/// `andnot` in `board_to_rays_avx2` instead of a maskz shuffle.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ray_permutation_avx2(focus: u32) -> ([__m256i; 2], [__m256i; 2]) {
    let perm = loadu2(RAY_PERMUTATIONS.get_unchecked(focus as usize).as_ptr().cast());
    let inv = _mm256_set1_epi8(0x80u8 as i8);
    let invalid = [_mm256_cmpeq_epi8(perm[0], inv), _mm256_cmpeq_epi8(perm[1], inv)];
    (perm, invalid)
}

/// AVX2 analogue of `board_to_rays`: 64-byte gather-by-index across two YMM
/// registers, without VBMI. Technique modelled on Reckless's AVX2
/// half-swizzler: per index byte, bits 0–3 select within a 16-byte lane
/// (shuffle_epi8), bit 4 selects the lane (blendv on idx<<3, after
/// permute2x128 builds lane-duplicated copies), bit 5 selects the source
/// register (blendv on idx<<2). The `slli_epi64` shifts move each byte's
/// bit 4/5 into its bit 7 — blendv only reads bit 7, and cross-byte spill
/// from a 64-bit shift < 8 only pollutes bits below 7, so per-byte selection
/// stays exact. Invalid entries (0x80) have bit 7 set, so all four
/// shuffle_epi8 legs zero those bytes: pboard reads 0 (not garbage like the
/// AVX-512 permutexvar version) at invalid positions — harmless either way,
/// since every consumer mask is a subset of the valid occupancy, and `rays`
/// is additionally forced to 0 there via andnot(invalid).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn board_to_rays_avx2(
    perm: [__m256i; 2],
    invalid: [__m256i; 2],
    board: [__m256i; 2],
) -> ([__m256i; 2], [__m256i; 2]) {
    let gather = |idxs: __m256i| -> __m256i {
        let reg_sel = _mm256_slli_epi64(idxs, 2); // bit 5 → bit 7
        let lane_sel = _mm256_slli_epi64(idxs, 3); // bit 4 → bit 7
        let from_reg = |b: __m256i| -> __m256i {
            let lo = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x00>(b, b), idxs);
            let hi = _mm256_shuffle_epi8(_mm256_permute2x128_si256::<0x11>(b, b), idxs);
            _mm256_blendv_epi8(lo, hi, lane_sel)
        };
        _mm256_blendv_epi8(from_reg(board[0]), from_reg(board[1]), reg_sel)
    };
    let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(PIECE_TO_BIT_TABLE.as_ptr().cast()));
    let pboard = [gather(perm[0]), gather(perm[1])];
    let rays = [
        _mm256_andnot_si256(invalid[0], _mm256_shuffle_epi8(lut, pboard[0])),
        _mm256_andnot_si256(invalid[1], _mm256_shuffle_epi8(lut, pboard[1])),
    ];
    (pboard, rays)
}

/// AVX2 analogue of `attackers_along_rays`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn attackers_along_rays_avx2(rays: [__m256i; 2]) -> u64 {
    let mask = loadu2(RAY_ATTACKERS_MASK.as_ptr().cast());
    nonzero_bytes_mask([
        _mm256_and_si256(rays[0], mask[0]),
        _mm256_and_si256(rays[1], mask[1]),
    ])
}

/// AVX2 analogue of `sliders_along_rays`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn sliders_along_rays_avx2(rays: [__m256i; 2]) -> u64 {
    let mask = loadu2(RAY_SLIDERS_MASK.as_ptr().cast());
    nonzero_bytes_mask([
        _mm256_and_si256(rays[0], mask[0]),
        _mm256_and_si256(rays[1], mask[1]),
    ]) & RAY_POSITIONS
}

// ---- Scalar mask-drain emission ---------------------------------------------
//
// No vpcompressb below AVX-512, and emission counts are tiny (avg ~1.5
// deltas per section call), so draining the masks scalar beats building a
// compress-LUT pipeline. Measured before getting fancy: see the Phase B
// commit message. `pb`/`pm` are the ray-frame pboard/perm spilled to stack.

/// Scalar-drain analogue of `splat_threats`: focus→victim records for each
/// `attacked` bit, attacker→focus records for each `attackers` bit.
#[inline(always)]
fn emit_threats(
    deltas: &mut Vec<RawThreatDelta>,
    pb: &[u8; 64],
    pm: &[u8; 64],
    mut attacked: u64,
    mut attackers: u64,
    focus_piece: u8,
    focus_sq: u8,
    add: bool,
) {
    while attacked != 0 {
        let i = attacked.trailing_zeros() as usize;
        deltas.push(RawThreatDelta::new(focus_piece, focus_sq, pb[i], pm[i], add));
        attacked &= attacked - 1;
    }
    while attackers != 0 {
        let i = attackers.trailing_zeros() as usize;
        deltas.push(RawThreatDelta::new(pb[i], pm[i], focus_piece, focus_sq, add));
        attackers &= attackers - 1;
    }
}

/// Scalar-drain analogue of `splat_xray_threats`. `victims` is expressed in
/// the OPPOSITE ray's octet (the caller rotates the mask by 32 bits — 4
/// octets — instead of flipping the vector halves like the AVX-512 path),
/// so the actual victim ray-position is `(bit + 32) % 64`. Pairing: both
/// masks carry ≤1 bit per octet on MATCHING octets (guaranteed by the
/// caller's ray_fill intersection), so k-th set bits pair up.
#[inline(always)]
fn emit_xray_threats(
    deltas: &mut Vec<RawThreatDelta>,
    pb: &[u8; 64],
    pm: &[u8; 64],
    mut sliders: u64,
    mut victims: u64,
    add: bool,
) {
    debug_assert_eq!(sliders.count_ones(), victims.count_ones());
    while sliders != 0 {
        let s = sliders.trailing_zeros() as usize;
        let v = (victims.trailing_zeros() as usize + 32) % 64;
        deltas.push(RawThreatDelta::new(pb[s], pm[s], pb[v], pm[v], add));
        sliders &= sliders - 1;
        victims &= victims - 1;
    }
}

/// AVX2 focus pass — the exact section logic of `splat_focus` (see its doc
/// for the contract mapping); only the primitives and emission differ.
#[target_feature(enable = "avx2")]
unsafe fn splat_focus_avx2(
    deltas: &mut Vec<RawThreatDelta>,
    board: [__m256i; 2],
    piece_cp: u8,
    square: u32,
    add: bool,
) {
    let (perm, invalid) = ray_permutation_avx2(square);
    let (pboard, rays) = board_to_rays_avx2(perm, invalid, board);

    // Spill the ray frame for the scalar emitters (frees all YMMs).
    let mut pb = [0u8; 64];
    let mut pm = [0u8; 64];
    _mm256_storeu_si256(pb.as_mut_ptr().cast(), pboard[0]);
    _mm256_storeu_si256(pb.as_mut_ptr().add(32).cast(), pboard[1]);
    _mm256_storeu_si256(pm.as_mut_ptr().cast(), perm[0]);
    _mm256_storeu_si256(pm.as_mut_ptr().add(32).cast(), perm[1]);

    let occupied = nonzero_bytes_mask(rays);
    let closest = closest_from_occupied(occupied); // Y_d ∪ occupied knight ring
    let first = closest & RAY_POSITIONS; // Y_d
    let second = closest_from_occupied((occupied & RAY_POSITIONS) & !first); // Z_d

    // §1 direct-from + (§2-direct ∪ §3) incoming direct attackers.
    let attacked_direct = attacking_along_rays(piece_cp, closest);
    let attackers_direct = attackers_along_rays_avx2(rays) & closest;
    emit_threats(deltas, &pb, &pm, attacked_direct, attackers_direct, piece_cp, square as u8, add);

    // §1b x-rays FROM the focus slider + §2b sliders x-raying the focus.
    let attacked_xray = attacking_along_rays(piece_cp, second);
    let slider_hits = sliders_along_rays_avx2(rays);
    let xray_attackers = slider_hits & second;
    emit_threats(deltas, &pb, &pm, attacked_xray, xray_attackers, piece_cp, square as u8, add);

    // §2 Z-deltas (sign inverted — same-index direct↔x-ray property).
    let seers = slider_hits & first;
    let z_opp = second.rotate_right(32); // Z_{-d}, expressed in octet d
    let z_pairs = ray_fill(seers) & ray_fill(z_opp);
    if z_pairs != 0 {
        emit_xray_threats(deltas, &pb, &pm, seers & z_pairs, z_opp & z_pairs, !add);
    }

    // §2b W-deltas (sign inverted).
    let y_opp = first.rotate_right(32); // Y_{-d}, expressed in octet d
    let w_pairs = ray_fill(xray_attackers) & ray_fill(y_opp);
    if w_pairs != 0 {
        emit_xray_threats(deltas, &pb, &pm, xray_attackers & w_pairs, y_opp & w_pairs, !add);
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn push_threats_on_change_avx2(
    deltas: &mut Vec<RawThreatDelta>,
    mailbox: &[u8; 64],
    white_bb: u64,
    occ: u64,
    piece_cp: u8,
    square: u32,
    add: bool,
) {
    let board = occ_masked_board_avx2(mailbox, white_bb, occ);
    splat_focus_avx2(deltas, board, piece_cp, square, add);
}

/// See `push_threats_on_move_avx512` for the occ_transit contract.
#[target_feature(enable = "avx2")]
pub unsafe fn push_threats_on_move_avx2(
    deltas: &mut Vec<RawThreatDelta>,
    mailbox: &[u8; 64],
    white_bb: u64,
    occ: u64,
    piece_cp: u8,
    src: u32,
    dst: u32,
) {
    let occ_transit = occ ^ (1u64 << dst);
    let board = occ_masked_board_avx2(mailbox, white_bb, occ_transit);
    splat_focus_avx2(deltas, board, piece_cp, src, false);
    splat_focus_avx2(deltas, board, piece_cp, dst, true);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_well_formed() {
        for i in 0..12 {
            assert_ne!(PIECE_TO_BIT_TABLE[i], 0, "piece slot {} should be nonzero", i);
        }
        for i in 12..16 {
            assert_eq!(PIECE_TO_BIT_TABLE[i], 0, "unused slot {} should be 0", i);
        }
        assert_eq!(RAY_ATTACKERS_MASK.len(), 64);
        assert_eq!(RAY_SLIDERS_MASK.len(), 64);
        assert_eq!(RAY_PERMUTATIONS.len(), 64);
        assert_eq!(RAY_PERMUTATIONS[0].len(), 64);
        for (i, &v) in RAY_ATTACKS_MASK.iter().enumerate() {
            assert_ne!(v, 0, "RAY_ATTACKS_MASK[{}] should be nonzero", i);
        }
    }

    /// Reconstruct the packed u32 from a RawThreatDelta's public accessors.
    /// (RawThreatDelta exposes no `raw()`; its inner u32 is private to
    /// threats.rs and this diagnostic must not touch production code.)
    fn delta_raw(d: &RawThreatDelta) -> u32 {
        (d.attacker_cp() as u32)
            | (d.from_sq() as u32) << 8
            | (d.victim_cp() as u32) << 16
            | ((d.to_sq() as u32) & 0x7F) << 24
            | if d.add() { 1u32 << 31 } else { 0 }
    }

    fn sq_name(sq: u8) -> String {
        format!("{}{}", (b'a' + (sq & 7)) as char, (b'1' + (sq >> 3)) as char)
    }

    fn cp_name(cp: u8) -> &'static str {
        const NAMES: [&str; 13] =
            ["WP", "WN", "WB", "WR", "WQ", "WK", "BP", "BN", "BB", "BR", "BQ", "BK", "--"];
        NAMES[(cp as usize).min(12)]
    }

    /// Slider in Coda's colored_piece encoding: B=2, R=3, Q=4 (mod 6).
    fn is_slider_cp(cp: u8) -> bool {
        matches!(cp % 6, 2 | 3 | 4)
    }

    fn fmt_delta(d: u32) -> String {
        let attacker = d as u8;
        let from = (d >> 8) as u8;
        let victim = (d >> 16) as u8;
        let to = ((d >> 24) & 0x7F) as u8;
        let add = (d & (1 << 31)) != 0;
        format!(
            "{:08x} ({},{},{},{}) {}",
            d,
            cp_name(attacker),
            sq_name(from),
            cp_name(victim),
            sq_name(to),
            if add { "ADD" } else { "SUB" },
        )
    }

    /// Categorize a single disagreeing delta. `simd_only` = emitted by the
    /// splat but not by Coda's scalar enumerator (vice versa if false).
    /// `change_add` = the board change under test was an ADD (piece appears);
    /// `focus_cp`/`focus_sq` = the changed piece.
    ///
    /// Parity is exact since the two-hit rewrite — any nonzero bucket is a
    /// regression. The categories structure the May 2026 blocker's failure
    /// modes and remain the fastest way to localize a future divergence:
    /// - xray-add-on-blocker-removal / xray-sub-on-blocker-arrival
    ///   (simd-only): Reckless direct-only-space semantics leaking in — a
    ///   blocker arrival/removal toggling a feature that Coda's model keeps
    ///   alive at the same index (contract §1.1).
    /// - scalar-only-slider-focus-xray: §1b gap — x-rays FROM a slider
    ///   focus (second-hit-per-ray enumeration).
    /// - scalar-only-xray-behind-focus: §2-Z / §2b gap — emissions whose
    ///   attacker is a slider elsewhere on the board.
    fn categorize(
        simd_only: bool,
        d: u32,
        change_add: bool,
        focus_cp: u8,
        focus_sq: u32,
    ) -> &'static str {
        let attacker = d as u8;
        let from = (d >> 8) as u8;
        let delta_add = (d & (1 << 31)) != 0;
        let from_focus = from as u32 == focus_sq;
        if simd_only {
            if !from_focus && is_slider_cp(attacker) && delta_add && !change_add {
                "xray-add-on-blocker-removal (simd-only)"
            } else if !from_focus && is_slider_cp(attacker) && !delta_add && change_add {
                "xray-sub-on-blocker-arrival (simd-only)"
            } else if from_focus {
                "simd-only-from-focus"
            } else {
                "simd-only-other"
            }
        } else if from_focus && is_slider_cp(focus_cp) {
            "scalar-only-slider-focus-xray"
        } else if !from_focus && is_slider_cp(attacker) {
            "scalar-only-xray-behind-focus"
        } else {
            "scalar-only-other"
        }
    }

    /// True when the host can run the AVX-512 splat (mirrors threats.rs's
    /// `splat_avx512_ok`, which is private to that module).
    fn splat_host() -> bool {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
            && std::is_x86_feature_detected!("avx512vbmi2")
    }

    /// True when the host can run the AVX2 splat fallback.
    fn avx2_host() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    /// True when `path` is runnable on this host (skip the test otherwise).
    fn path_available(path: crate::threats::SplatPath) -> bool {
        use crate::threats::SplatPath;
        match path {
            SplatPath::Avx512 => splat_host(),
            SplatPath::Avx2 => avx2_host(),
            SplatPath::Scalar => true,
        }
    }

    /// Corpus start positions: deterministic random playouts from varied
    /// starting positions (same seeding scheme as threat_accum's fuzzer).
    /// Includes kiwipete (index 2), promotion-heavy and bare-king endgames.
    const START_FENS: &[&str] = &[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",
        "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
    ];

    /// Tactically dense extensions for the deep corpus (WAC 004/126/188/220):
    /// slider batteries, pins, and multi-x-ray stacking — the shapes the
    /// two-hit enumeration must not get wrong.
    const TACTICAL_FENS: &[&str] = &[
        "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1",
        "r5r1/pQ5p/1qp2R2/2k1p3/4P3/2PP4/P1P3PP/6K1 w - - 0 1",
        "3RNbk1/pp3p2/4rQpp/8/1qr5/7P/P4P2/3R2K1 w - - 0 1",
        "3rr1k1/ppp2ppp/8/5Q2/4n3/1B5R/PPP1qPP1/5RK1 b - - 0 1",
    ];

    fn next_u32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    /// Change-level parity driver: over deterministic random playouts from
    /// `fens`, at every position exercise removal of each piece on the
    /// board and addition of every colored piece at each empty square;
    /// compare scalar `push_threats_on_change` vs the `simd_path` splat
    /// (called DIRECTLY, bypassing dispatch — this dev host would dispatch
    /// AVX-512 otherwise) as exact delta multisets, and assert ZERO
    /// mismatches. On failure, disagreements are categorized (see
    /// `categorize`) to localize the regression.
    fn run_change_parity(fens: &[&str], target_positions: usize, simd_path: crate::threats::SplatPath) {
        if !path_available(simd_path) {
            eprintln!("skipping: host lacks the CPU features for {:?}", simd_path);
            return;
        }
        crate::init();
        use crate::board::Board;
        use crate::movegen::generate_legal_moves;
        use crate::threats::push_threats_on_change_scalar as scalar_push;
        use std::collections::BTreeMap;

        const MAX_PLIES_PER_GAME: usize = 60;

        // (category → (count, up-to-5 example strings))
        let mut cats: BTreeMap<&'static str, (u64, Vec<String>)> = BTreeMap::new();
        let mut positions = 0usize;
        let mut changes_tested = 0u64;
        let mut changes_mismatched = 0u64;

        // Compare one change (cp appearing/disappearing at sq) on `board`.
        let compare = |board: &Board,
                           cp: u8,
                           sq: u32,
                           add: bool,
                           cats: &mut BTreeMap<&'static str, (u64, Vec<String>)>|
         -> bool {
            let pcolor = cp / 6;
            let ptype = cp % 6;
            let occ = board.colors[0] | board.colors[1];
            let mut scalar_d: Vec<RawThreatDelta> = Vec::with_capacity(64);
            scalar_push(
                &mut scalar_d, &board.pieces, &board.colors, &board.mailbox,
                occ, pcolor, ptype, sq, add,
            );
            let mut simd_d: Vec<RawThreatDelta> = Vec::with_capacity(64);
            match simd_path {
                crate::threats::SplatPath::Avx512 => unsafe {
                    push_threats_on_change_avx512(
                        &mut simd_d, &board.mailbox, board.colors[0], occ, cp, sq, add,
                    );
                },
                crate::threats::SplatPath::Avx2 => unsafe {
                    push_threats_on_change_avx2(
                        &mut simd_d, &board.mailbox, board.colors[0], occ, cp, sq, add,
                    );
                },
                crate::threats::SplatPath::Scalar => unreachable!("parity vs scalar itself"),
            }
            let mut av: Vec<u32> = scalar_d.iter().map(delta_raw).collect();
            let mut bv: Vec<u32> = simd_d.iter().map(delta_raw).collect();
            av.sort_unstable();
            bv.sort_unstable();
            if av == bv {
                return false;
            }
            let fen = board.to_fen();
            let change_desc = format!(
                "fen=\"{}\" change={} {} @ {}",
                fen,
                if add { "ADD" } else { "SUB" },
                cp_name(cp),
                sq_name(sq as u8),
            );
            for &d in av.iter().filter(|x| !bv.contains(x)) {
                let cat = categorize(false, d, add, cp, sq);
                let e = cats.entry(cat).or_default();
                e.0 += 1;
                if e.1.len() < 5 {
                    e.1.push(format!("{} scalar-only-delta: {}", change_desc, fmt_delta(d)));
                }
            }
            for &d in bv.iter().filter(|x| !av.contains(x)) {
                let cat = categorize(true, d, add, cp, sq);
                let e = cats.entry(cat).or_default();
                e.0 += 1;
                if e.1.len() < 5 {
                    e.1.push(format!("{} simd-only-delta: {}", change_desc, fmt_delta(d)));
                }
            }
            true
        };

        'corpus: for round in 0.. {
            for (fen_idx, fen) in fens.iter().enumerate() {
                let seed: u32 = 0x5b1a7_a5eu32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((round as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };
                let mut board = Board::from_fen(fen);
                for _ply in 0..MAX_PLIES_PER_GAME {
                    if positions >= target_positions {
                        break 'corpus;
                    }
                    positions += 1;
                    // Exercise every board change at this position:
                    // removal of each occupied square's piece, and addition
                    // of each of the 12 colored pieces at each empty square.
                    for sq in 0..64u32 {
                        let pt = board.mailbox[sq as usize];
                        if pt < 6 {
                            let pcolor =
                                if board.colors[0] & (1u64 << sq) != 0 { 0u8 } else { 1u8 };
                            let cp = pcolor * 6 + pt;
                            changes_tested += 1;
                            changes_mismatched +=
                                compare(&board, cp, sq, false, &mut cats) as u64;
                        } else {
                            for cp in 0..12u8 {
                                changes_tested += 1;
                                changes_mismatched +=
                                    compare(&board, cp, sq, true, &mut cats) as u64;
                            }
                        }
                    }
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 {
                        break;
                    }
                    let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);
                    if !board.make_move(mv) {
                        break;
                    }
                }
            }
        }

        let total_bad_deltas: u64 = cats.values().map(|(n, _)| *n).sum();
        eprintln!("=== splat parity ===");
        eprintln!(
            "positions={} changes_tested={} changes_mismatched={} disagreeing_deltas={}",
            positions, changes_tested, changes_mismatched, total_bad_deltas
        );
        for (cat, (n, examples)) in &cats {
            eprintln!("  [{:>8}] {}", n, cat);
            for ex in examples {
                eprintln!("      e.g. {}", ex);
            }
        }
        assert_eq!(
            changes_mismatched, 0,
            "splat parity mismatches: {} changes ({} deltas) disagree — see category \
             summary above. Any nonzero bucket is silent eval corruption in the \
             CODA_SPLAT=1 path; do NOT ship until zero.",
            changes_mismatched, total_bad_deltas
        );
    }

    /// CI gate: exact change-level parity on a reduced corpus (~1s release).
    #[test]
    fn splat_change_parity_small() {
        run_change_parity(START_FENS, 100, crate::threats::SplatPath::Avx512);
    }

    /// CI gate, AVX2 path.
    #[test]
    fn splat_change_parity_small_avx2() {
        run_change_parity(START_FENS, 100, crate::threats::SplatPath::Avx2);
    }

    /// Deep check: the full 1000-position corpus (553k changes) plus the
    /// tactically dense extension FENs. Run with
    /// `cargo test --release splat_change_parity -- --ignored --nocapture`.
    #[test]
    #[ignore = "deep parity corpus; splat_change_parity_small is the CI gate"]
    fn splat_change_parity_with_scalar() {
        run_change_parity(START_FENS, 1000, crate::threats::SplatPath::Avx512);
        let extended: Vec<&str> =
            START_FENS.iter().chain(TACTICAL_FENS.iter()).copied().collect();
        run_change_parity(&extended, 1200, crate::threats::SplatPath::Avx512);
    }

    /// Deep check, AVX2 path — same corpora as the AVX-512 gate.
    #[test]
    #[ignore = "deep parity corpus; splat_change_parity_small_avx2 is the CI gate"]
    fn splat_change_parity_with_scalar_avx2() {
        run_change_parity(START_FENS, 1000, crate::threats::SplatPath::Avx2);
        let extended: Vec<&str> =
            START_FENS.iter().chain(TACTICAL_FENS.iter()).copied().collect();
        run_change_parity(&extended, 1200, crate::threats::SplatPath::Avx2);
    }

    /// Move-level dispatch parity across ALL move shapes: for every legal
    /// move at each corpus position, generate `board.threat_deltas` twice
    /// through the REAL make_move — once with scalar FORCED (thread-local
    /// SPLAT_TEST_FORCE; the default dispatch would pick a splat path on
    /// qualifying hosts, making the comparison vacuous), once with the
    /// `simd_path` splat forced — and require exact multiset equality.
    /// This exercises the full §3.1 call-site sequence (occ_transit legs,
    /// capture/EP change legs, promotion double-change, castle rook leg)
    /// rather than synthetic single changes, and asserts each move shape
    /// actually occurs.
    fn run_move_parity(simd_path: crate::threats::SplatPath) {
        use crate::threats::{SplatPath, SPLAT_TEST_FORCE};
        if !path_available(simd_path) {
            eprintln!("skipping: host lacks the CPU features for {:?}", simd_path);
            return;
        }
        crate::init();
        use crate::board::Board;
        use crate::movegen::generate_legal_moves;
        use crate::types::{
            is_promotion, move_flags, move_to, move_to_uci, FLAG_CASTLE, FLAG_EN_PASSANT,
        };

        // START_FENS guarantee castles (kiwipete) and promotions / promo-
        // captures (indices 4, 6); the EP FEN makes an immediate EP capture
        // legal (all legal moves are tested at every position, so poised
        // shapes are guaranteed exercised, not left to random playout).
        let mut fens: Vec<&str> = START_FENS.to_vec();
        fens.push("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");

        // [quiet, capture, en-passant, castle, promotion, promo-capture]
        let mut shape_counts = [0u64; 6];
        let mut moves_tested = 0u64;

        for (fen_idx, fen) in fens.iter().enumerate() {
            let mut rng = 0x0ddba11u32.wrapping_add(fen_idx as u32 * 7919);
            let mut board = Board::from_fen(fen);
            board.generate_threat_deltas = true;
            for _ply in 0..40 {
                let legal = generate_legal_moves(&board);
                if legal.len == 0 {
                    break;
                }
                for i in 0..legal.len {
                    let mv = legal.get(i);
                    let flags = move_flags(mv);
                    let is_capture = board.mailbox[move_to(mv) as usize] < 6
                        || flags == FLAG_EN_PASSANT;
                    let shape = if flags == FLAG_EN_PASSANT {
                        2
                    } else if flags == FLAG_CASTLE {
                        3
                    } else if is_promotion(mv) && is_capture {
                        5
                    } else if is_promotion(mv) {
                        4
                    } else if is_capture {
                        1
                    } else {
                        0
                    };

                    let mut b_scalar = board.clone();
                    SPLAT_TEST_FORCE.with(|f| f.set(Some(SplatPath::Scalar)));
                    let ok_scalar = b_scalar.make_move(mv);
                    SPLAT_TEST_FORCE.with(|f| f.set(None));
                    if !ok_scalar {
                        continue;
                    }
                    let mut scalar: Vec<u32> =
                        b_scalar.threat_deltas.iter().map(delta_raw).collect();

                    let mut b_simd = board.clone();
                    SPLAT_TEST_FORCE.with(|f| f.set(Some(simd_path)));
                    let ok = b_simd.make_move(mv);
                    SPLAT_TEST_FORCE.with(|f| f.set(None));
                    assert!(ok);
                    let mut simd: Vec<u32> =
                        b_simd.threat_deltas.iter().map(delta_raw).collect();

                    scalar.sort_unstable();
                    simd.sort_unstable();
                    if scalar != simd {
                        eprintln!("fen=\"{}\" mv={}", board.to_fen(), move_to_uci(mv));
                        for &d in scalar.iter().filter(|x| !simd.contains(x)) {
                            eprintln!("  scalar-only: {}", fmt_delta(d));
                        }
                        for &d in simd.iter().filter(|x| !scalar.contains(x)) {
                            eprintln!("  simd-only:   {}", fmt_delta(d));
                        }
                        panic!("splat move-parity mismatch (see stderr)");
                    }
                    shape_counts[shape] += 1;
                    moves_tested += 1;
                }
                let mv = legal.get((next_u32(&mut rng) as usize) % legal.len);
                if !board.make_move(mv) {
                    break;
                }
            }
        }

        eprintln!(
            "splat move parity: moves={} quiet={} capture={} ep={} castle={} promo={} promo-capture={}",
            moves_tested,
            shape_counts[0],
            shape_counts[1],
            shape_counts[2],
            shape_counts[3],
            shape_counts[4],
            shape_counts[5],
        );
        const SHAPES: [&str; 6] =
            ["quiet", "capture", "en-passant", "castle", "promotion", "promo-capture"];
        for (i, name) in SHAPES.iter().enumerate() {
            assert!(
                shape_counts[i] > 0,
                "move shape '{}' never exercised — corpus regression",
                name
            );
        }
    }

    #[test]
    fn splat_move_parity_all_shapes() {
        run_move_parity(crate::threats::SplatPath::Avx512);
    }

    #[test]
    fn splat_move_parity_all_shapes_avx2() {
        run_move_parity(crate::threats::SplatPath::Avx2);
    }
}
