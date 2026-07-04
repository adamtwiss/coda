//! AVX-512 byteboard-splat threat-delta enumeration (port of Reckless's
//! `nnue/accumulator/threats/vectorized` machinery).
//!
//! Modelled on Reckless's `splat_threats` / `splat_xray_threats` /
//! `push_threats_on_change` / `push_threats_on_move`. The byteboard splat
//! does threat-delta ENUMERATION in batched SIMD over the entire mailbox in
//! one pass, replacing Coda's per-piece scalar magic-bitboard pattern in
//! `threats::push_threats_for_piece`. Cache benefit comes from:
//!   1. Single SIMD sweep over the board (no per-piece scalar loop)
//!   2. Ray-grouped delta emission — consecutive deltas correspond to
//!      spatially adjacent threat features, which means the subsequent
//!      apply path's weight-row reads are HW-prefetcher-friendly
//!   3. No per-piece magic-bitboard lookups
//!
//! The actual weight-row apply still goes through `threats::apply_threat_deltas`;
//! this module only changes how the `RawThreatDelta` list is built.
//!
//! Phase A (this file): AVX-512 + VBMI2 path. Coda already requires VBMI2
//! for the `find_nnz_chunks_avx512` SIMD scan landed in #946 — same hosts
//! (Sapphire Rapids+, Zen 4+).
//!
//! Phase B (followup): AVX-2 fallback (mirror of Reckless's
//! `vectorized/avx2.rs`). Most fleet hosts (Hercules Coffee Lake, Atlas
//! Zen 1) only do AVX-2; AVX-512-only would shift fleet results away from
//! older hosts which gain the most from cache work.
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
#![allow(dead_code)] // gated until production integration in Phase B

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

/// Mask of ray-positions that are the closest occupied square along their
/// ray. (Per-ray: first-from-focus occupied position is "closest".)
///
/// Algorithm: occupied = nonzero positions; o = occupied | sentinel;
/// x = o ^ (o - delta); first-occupied bit per ray = x & occupied.
/// Sentinel + delta values are tuned so the subtract carries from each
/// ray's start (position 0 = LSB of each byte-octet) up to its first
/// occupied position; XOR cancels everything before/after.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn closest_on_rays(rays: __m512i) -> u64 {
    let occupied = _mm512_test_epi8_mask(rays, rays);
    let o = occupied | 0x8181818181818181;
    let x = o ^ (o - 0x0303030303030303);
    x & occupied
}

/// Mask of ray-positions that are valid pawn-victim x-ray endpoints.
/// Used by `splat_xray_threats` to find the slider-then-victim chain.
#[inline(always)]
pub(crate) const fn ray_fill(x: u64) -> u64 {
    let x = (x + 0x7E7E7E7E7E7E7E7E) & 0x8080808080808080;
    x - (x >> 7)
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

/// Exclude a square from the board mailbox by overwriting its byte with
/// NO_PIECE (12). Used by `push_threats_on_move` to compute attackers
/// from the source square as if the moving piece were already removed.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn exclude_square(board: __m512i, sq: u32) -> __m512i {
    let bb = 1u64 << sq;
    _mm512_mask_blend_epi8(bb, board, _mm512_set1_epi8(crate::types::NO_PIECE as i8))
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

    // Append into deltas Vec via raw pointer writes.
    let attacked_count = attacked.count_ones() as usize;
    let attackers_count = attackers.count_ones() as usize;
    let total = attacked_count + attackers_count;
    deltas.reserve(total);
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
    deltas.reserve(count);
    let len_before = deltas.len();
    let dst = deltas.as_mut_ptr().add(len_before) as *mut __m128i;
    _mm_storeu_si128(dst, _mm_or_si128(_mm_unpacklo_epi16(pair1, pair2), add_v));
    let dst = deltas.as_mut_ptr().add(len_before + 4) as *mut __m128i;
    _mm_storeu_si128(dst, _mm_or_si128(_mm_unpackhi_epi16(pair1, pair2), add_v));
    deltas.set_len(len_before + count);
}

// =============================================================================
// High-level entry points
// =============================================================================

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
pub unsafe fn push_threats_on_change_avx512(
    deltas: &mut Vec<RawThreatDelta>,
    mailbox: &[u8; 64],
    white_bb: u64,
    piece_cp: u8,
    square: u32,
    add: bool,
) {
    let board = mailbox_vector_avx512(mailbox, white_bb);
    let (perm, valid) = ray_permutation(square);
    let (pboard, rays) = board_to_rays(perm, valid, board);

    let closest = closest_on_rays(rays);
    let attacked_occupied = attacking_along_rays(piece_cp, closest);
    let attackers = attackers_along_rays(rays) & closest;
    let sliders = sliders_along_rays(rays) & closest;

    if std::env::var_os("CODA_SPLAT_DEBUG").is_some() {
        let mut rays_bytes = [0u8; 64];
        let mut perm_bytes = [0u8; 64];
        let mut pboard_bytes = [0u8; 64];
        let mut board_bytes = [0u8; 64];
        _mm512_storeu_si512(rays_bytes.as_mut_ptr().cast(), rays);
        _mm512_storeu_si512(perm_bytes.as_mut_ptr().cast(), perm);
        _mm512_storeu_si512(pboard_bytes.as_mut_ptr().cast(), pboard);
        _mm512_storeu_si512(board_bytes.as_mut_ptr().cast(), board);
        let occupied = _mm512_test_epi8_mask(rays, rays);
        eprintln!("[splat] focus_sq={} focus_piece={} add={} valid_mask={:016x}", square, piece_cp, add, valid);
        eprintln!("[splat]   occupied = {:016x}", occupied);
        eprintln!("[splat]   closest  = {:016x}", closest);
        eprintln!("[splat]   attacked = {:016x}  count={}", attacked_occupied, attacked_occupied.count_ones());
        eprintln!("[splat]   attackrs = {:016x}  count={}", attackers, attackers.count_ones());
        eprintln!("[splat]   sliders  = {:016x}", sliders);
        eprintln!("[splat]   board[0..8]   = {:?}", &board_bytes[..8]);
        eprintln!("[splat]   board[8..16]  = {:?}", &board_bytes[8..16]);
        eprintln!("[splat]   board[48..56] = {:?}", &board_bytes[48..56]);
        eprintln!("[splat]   board[56..64] = {:?}", &board_bytes[56..64]);
        eprintln!("[splat]   N-ray perm   = {:?}", &perm_bytes[..8]);
        eprintln!("[splat]   N-ray pboard = {:?}", &pboard_bytes[..8]);
        eprintln!("[splat]   N-ray rays   = {:?}", &rays_bytes[..8]);
    }

    splat_threats(deltas, pboard, perm, attacked_occupied, attackers, piece_cp, square, add);

    let victim = (closest & 0xFEFEFEFEFEFEFEFE).rotate_right(32);
    let xray_valid = ray_fill(victim) & ray_fill(sliders);
    splat_xray_threats(
        deltas, pboard, perm,
        sliders & xray_valid,
        victim & xray_valid,
        !add, // X-rays of an `add` are removed (the new piece blocks the x-ray)
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,avx512vbmi2")]
pub unsafe fn push_threats_on_move_avx512(
    deltas: &mut Vec<RawThreatDelta>,
    mailbox: &[u8; 64],
    white_bb: u64,
    piece_cp: u8,
    src: u32,
    dst: u32,
) {
    let board = mailbox_vector_avx512(mailbox, white_bb);

    let board_src = exclude_square(board, dst);
    let (src_perm, src_valid) = ray_permutation(src);
    let (src_pboard, src_rays) = board_to_rays(src_perm, src_valid, board_src);
    let src_closest = closest_on_rays(src_rays);
    let src_attacked = attacking_along_rays(piece_cp, src_closest);
    let src_attackers = attackers_along_rays(src_rays) & src_closest;
    let src_sliders = sliders_along_rays(src_rays) & src_closest;

    let (dst_perm, dst_valid) = ray_permutation(dst);
    let (dst_pboard, dst_rays) = board_to_rays(dst_perm, dst_valid, board);
    let dst_closest = closest_on_rays(dst_rays);
    let dst_attacked = attacking_along_rays(piece_cp, dst_closest);
    let dst_attackers = attackers_along_rays(dst_rays) & dst_closest;
    let dst_sliders = sliders_along_rays(dst_rays) & dst_closest;

    splat_threats(deltas, src_pboard, src_perm, src_attacked, src_attackers, piece_cp, src, false);
    splat_threats(deltas, dst_pboard, dst_perm, dst_attacked, dst_attackers, piece_cp, dst, true);

    let src_victim = (src_closest & 0xFEFEFEFEFEFEFEFE).rotate_right(32);
    let dst_victim = (dst_closest & 0xFEFEFEFEFEFEFEFE).rotate_right(32);
    let src_xray_valid = ray_fill(src_victim) & ray_fill(src_sliders);
    let dst_xray_valid = ray_fill(dst_victim) & ray_fill(dst_sliders);
    splat_xray_threats(
        deltas, src_pboard, src_perm,
        src_sliders & src_xray_valid,
        src_victim & src_xray_valid,
        true,  // src removal → x-ray re-enabled
    );
    splat_xray_threats(
        deltas, dst_pboard, dst_perm,
        dst_sliders & dst_xray_valid,
        dst_victim & dst_xray_valid,
        false, // dst arrival → x-ray blocked
    );
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
    /// Categories (the May 2026 blocker, structured):
    /// - xray-add-on-blocker-removal: removing a blocker makes a slider's
    ///   attack on the piece behind it DIRECT; Reckless's direct-only space
    ///   emits an ADD, but in Coda's space the feature was ALREADY active as
    ///   an x-ray (same feature index) → double-count.
    /// - xray-sub-on-blocker-arrival: the mirror image — a new piece blocks
    ///   a direct slider attack; Reckless emits a SUB, Coda's model keeps the
    ///   feature alive as an x-ray.
    /// - scalar-only-slider-focus-xray: Coda's §1b emits x-rays FROM a slider
    ///   focus piece (through its first blocker); the splat has no
    ///   second-hit-per-ray enumeration (known Phase A scope gap).
    /// - scalar-only-xray-behind-focus: Coda's §2 emits the x-ray target
    ///   BEHIND the focus square for sliders that see it; splat's
    ///   splat_xray_threats emits these with the OPPOSITE add-sense or not
    ///   at all (semantic mismatch sibling of the two categories above).
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

    /// Parity diagnostic: AVX-512 splat vs Coda's scalar
    /// `push_threats_on_change`, over ~1000 positions from deterministic
    /// random playouts. For every position it tests removal of each piece on
    /// the board and addition of every colored piece at each empty square,
    /// compares the two delta sets, and CATEGORIZES each disagreement.
    ///
    /// KNOWN TO FAIL — #[ignore]d by design. This documents the May 2026
    /// Phase A blocker (commit 5096ac0): Coda's threat space encodes x-ray
    /// features (slider → piece beyond first blocker) at the SAME feature
    /// index as direct attacks, while Reckless's byteboard splat assumes a
    /// direct-only space and emits add/sub deltas whenever an attack changes
    /// between direct and x-ray. A direct port therefore double-counts.
    /// Do NOT "fix" this test by tweaking the splat masks — the next phase
    /// is an x-ray-aware enumerator; this diagnostic (run with
    /// `cargo test --release splat_change_parity -- --ignored --nocapture`)
    /// is its design input.
    #[test]
    #[ignore = "documents the known Reckless-vs-Coda x-ray threat-space mismatch (5096ac0); diagnostic, not a regression gate"]
    fn splat_change_parity_with_scalar() {
        if !std::is_x86_feature_detected!("avx512vbmi2")
            || !std::is_x86_feature_detected!("avx512vbmi")
        {
            eprintln!("skipping: host lacks avx512vbmi/vbmi2");
            return;
        }
        crate::init();
        use crate::board::Board;
        use crate::movegen::generate_legal_moves;
        use crate::threats::push_threats_on_change as scalar_push;
        use std::collections::BTreeMap;

        // Corpus: deterministic random playouts from varied starting
        // positions (same seeding scheme as threat_accum's fuzzer).
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
        const TARGET_POSITIONS: usize = 1000;
        const MAX_PLIES_PER_GAME: usize = 60;

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }

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
            unsafe {
                push_threats_on_change_avx512(&mut simd_d, &board.mailbox, board.colors[0], cp, sq, add);
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
            for (fen_idx, fen) in START_FENS.iter().enumerate() {
                let seed: u32 = 0x5b1a7_a5eu32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((round as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };
                let mut board = Board::from_fen(fen);
                for _ply in 0..MAX_PLIES_PER_GAME {
                    if positions >= TARGET_POSITIONS {
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
        eprintln!("=== splat parity diagnostic ===");
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
            "splat parity mismatches: {} changes ({} deltas) disagree — see category summary above. \
             EXPECTED with the current Reckless-semantics splat (x-ray space mismatch, 5096ac0).",
            changes_mismatched, total_bad_deltas
        );
    }
}
