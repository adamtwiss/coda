/// NNUE v5/v6 inference: HalfKA network with CReLU/SCReLU activation.
///
/// Architecture: (12288 → N)×2 → 1×8
/// - 12288 input features = 16 king buckets × 12 piece types × 64 squares
/// - Two perspectives (side-to-move and not-side-to-move)
/// - CReLU: clamp [0, QA=255]
/// - SCReLU: clamp [0, QA=255] then square
/// - 8 output buckets selected by material count

use std::fs::File;
use std::io::{Read as IoRead, BufReader};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::bitboard::*;
use crate::board::Board;
use crate::types::*;

// Network dimensions
pub const NNUE_INPUT_SIZE: usize = 12288; // 16 buckets × 12 piece types × 64 squares
pub const NNUE_OUTPUT_BUCKETS: usize = 8;
pub const NNUE_KING_BUCKETS: usize = 16;
const NNUE_NUM_PIECE_TYPES: usize = 12;

// Quantization
const QA: i32 = 255;  // accumulator scale (CReLU/SCReLU clip max)
const QB: i32 = 64;   // output weight scale
const QAB: i32 = QA * QB; // 16320
const EVAL_SCALE: i32 = 400; // sigmoid → centipawns

// File magic
const NNUE_MAGIC: u32 = 0x4E4E5545; // "NNUE" in LE

// King bucket table: maps square (0-63) to bucket (0-15).
// 4 mirrored files × 4 rank groups. Files e-h mirror to d-a.
static mut KING_BUCKET: [usize; 64] = [0; 64];
static mut KING_MIRROR: [bool; 64] = [false; 64];

pub fn init_nnue() {
    for sq in 0..64 {
        let file = sq % 8;
        let rank = sq / 8;

        let (mirrored_file, mirror) = if file >= 4 {
            (7 - file, true)
        } else {
            (file, false)
        };

        let rank_group = rank / 2;

        unsafe {
            KING_BUCKET[sq] = mirrored_file * 4 + rank_group;
            KING_MIRROR[sq] = mirror;
        }
    }
}

#[inline]
fn king_bucket(sq: usize) -> usize {
    unsafe { KING_BUCKET[sq] }
}

#[inline]
fn king_mirror(sq: usize) -> bool {
    unsafe { KING_MIRROR[sq] }
}

/// Public accessors for same-bucket king move detection in search.
#[inline]
pub fn king_bucket_pub(sq: usize) -> usize { king_bucket(sq) }
#[inline]
pub fn king_mirror_pub(sq: usize) -> bool { king_mirror(sq) }

/// Piece index for HalfKA: maps (color, piece_type) to 0-11.
/// White pieces: 0-5, Black pieces: 6-11.
#[inline]
fn piece_index(color: u8, pt: u8) -> usize {
    (color as usize) * 6 + pt as usize
}

/// Compute HalfKA feature index.
/// perspective: WHITE or BLACK
/// king_sq: the king square for this perspective
/// pc_color: color of the piece
/// pc_type: piece type (PAWN..KING)
/// pc_sq: square of the piece
pub fn halfka_index(perspective: u8, king_sq: u8, pc_color: u8, pc_type: u8, pc_sq: u8) -> usize {
    let mut ks = king_sq as usize;
    let mut ps = pc_sq as usize;
    let mut pi = piece_index(pc_color, pc_type);

    if perspective == BLACK {
        // Mirror vertically
        ks ^= 56;
        ps ^= 56;
        // Swap piece colors: 0-5 ↔ 6-11
        pi = if pi >= 6 { pi - 6 } else { pi + 6 };
    }

    // Check file mirroring (king on files e-h)
    if king_mirror(ks) {
        ps = (ps & !7) | (7 - (ps & 7));
    }

    let bucket = king_bucket(ks);
    bucket * (NNUE_NUM_PIECE_TYPES * 64) + pi * 64 + ps
}

/// Output bucket from piece count.
pub fn output_bucket(piece_count: u32) -> usize {
    let bucket = (piece_count as i32 - 2) / 4;
    bucket.clamp(0, NNUE_OUTPUT_BUCKETS as i32 - 1) as usize
}

// ---- AVX2 SIMD helper functions ----

/// Add a weight row to an accumulator vector (both i16, length h).
/// SAFETY: requires AVX2. Caller must ensure acc and row have length >= h,
/// and h is a multiple of 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_acc_add(acc: &mut [i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
        let sum = _mm256_add_epi16(a, b);
        _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, sum);
        i += 16;
    }
}

/// Fused copy + add: dst[i] = src[i] + row[i]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_acc_copy_add(dst: &mut [i16], src: &[i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, _mm256_add_epi16(a, b));
        i += 16;
    }
}

/// Fused copy + sub: dst[i] = src[i] - row[i]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_acc_copy_sub(dst: &mut [i16], src: &[i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, _mm256_sub_epi16(a, b));
        i += 16;
    }
}

/// Subtract a weight row from an accumulator vector (both i16, length h).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_acc_sub(acc: &mut [i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
        let diff = _mm256_sub_epi16(a, b);
        _mm256_storeu_si256(acc.as_mut_ptr().add(i) as *mut __m256i, diff);
        i += 16;
    }
}

/// CReLU dot product: clamp acc values to [0, QA=255], dot with output weights.
/// Returns i64 sum. acc and weights have length h.
///
/// Uses VPMADDWD for efficient i16×i16→i32 pairwise multiply-accumulate.
/// Drains i32 accumulator to i64 every 128 elements to prevent overflow.
/// (Max per pair: 255*32767 + 255*32767 ≈ 16.7M. After 128 pairs: ≈ 2.1B, near i32 limit.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_crelu_dot(acc: &[i16], weights: &[i16], h: usize) -> i64 {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let mut total = _mm256_setzero_si256(); // 4 × i64
    let mut sum32 = _mm256_setzero_si256(); // 8 × i32
    let mut count = 0u32;

    let mut i = 0;
    while i < h {
        let v = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
        let w = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);
        let prod = _mm256_madd_epi16(clamped, w);
        sum32 = _mm256_add_epi32(sum32, prod);

        count += 16;
        // Drain to i64 every 128 elements (8 VPMADDWD results = 64 pairs ≈ 1B max)
        if count >= 128 {
            // Sign-extend 8×i32 to 2×4×i64 and add
            total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum32)));
            total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum32, 1)));
            sum32 = _mm256_setzero_si256();
            count = 0;
        }

        i += 16;
    }

    // Drain remaining
    if count > 0 {
        total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum32)));
        total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum32, 1)));
    }

    hsum_epi64(total)
}

/// Pairwise dot product: clamp(a[i],0,255) * clamp(b[i],0,255) * weights[i]
/// where a = first half, b = second half of accumulator.
/// Returns i64 sum (caller divides by QA=255).
///
/// Uses byte decomposition: a*b ∈ [0,65025] = byte0 + byte1*256.
/// byte0 = (a*b) & 0xFF, byte1 = (a*b) >> 8.
/// Both accumulated via VPMADDWD, then byte1 shifted left by 8 before combining.
/// Drains i32 → i64 every 128 elements. count must be multiple of 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_pairwise_dot(acc_first: &[i16], acc_second: &[i16], weights: &[i16], count: usize) -> i64 {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16); // 255 = clamp ceiling + byte0 mask
    let mut total = _mm256_setzero_si256(); // 4 × i64
    let mut sum32_b0 = _mm256_setzero_si256(); // byte0 accumulator (8 × i32)
    let mut sum32_b1 = _mm256_setzero_si256(); // byte1 accumulator (8 × i32)
    let mut batch = 0u32;

    let mut i = 0;
    while i < count {
        // Load and clamp a[i..i+16] to [0, 255]
        let a = _mm256_loadu_si256(acc_first.as_ptr().add(i) as *const __m256i);
        let a_clamped = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);

        // Load and clamp b[i..i+16] to [0, 255]
        let b = _mm256_loadu_si256(acc_second.as_ptr().add(i) as *const __m256i);
        let b_clamped = _mm256_min_epi16(_mm256_max_epi16(b, zero), qa);

        // a*b in u16 (max 65025, fits in u16)
        let prod = _mm256_mullo_epi16(a_clamped, b_clamped);

        // Load weights
        let w = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);

        // byte0 = prod & 0xFF
        let byte0 = _mm256_and_si256(prod, qa); // qa = 0x00FF
        // byte1 = prod >> 8
        let byte1 = _mm256_srli_epi16(prod, 8);

        // VPMADDWD: pairwise i16×i16 → i32 (adjacent pairs summed)
        sum32_b0 = _mm256_add_epi32(sum32_b0, _mm256_madd_epi16(byte0, w));
        sum32_b1 = _mm256_add_epi32(sum32_b1, _mm256_madd_epi16(byte1, w));

        batch += 16;
        if batch >= 128 {
            // Drain byte0: sign-extend i32 → i64
            total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum32_b0)));
            total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum32_b0, 1)));
            // Drain byte1: shift left 8 then add
            let b1_lo = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum32_b1)), 8);
            let b1_hi = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum32_b1, 1)), 8);
            total = _mm256_add_epi64(total, b1_lo);
            total = _mm256_add_epi64(total, b1_hi);
            sum32_b0 = _mm256_setzero_si256();
            sum32_b1 = _mm256_setzero_si256();
            batch = 0;
        }

        i += 16;
    }

    // Drain remaining
    if batch > 0 {
        total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum32_b0)));
        total = _mm256_add_epi64(total, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum32_b0, 1)));
        let b1_lo = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum32_b1)), 8);
        let b1_hi = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum32_b1, 1)), 8);
        total = _mm256_add_epi64(total, b1_lo);
        total = _mm256_add_epi64(total, b1_hi);
    }

    hsum_epi64(total)
}

/// SCReLU dot product: clamp acc to [0, QA=255], square, dot with output weights.
/// Returns i64 sum at scale QA² × QB. acc and weights have length h.
///
/// Approach: v²*w computed per-element in i32, accumulated in i64.
/// Process 16 elements per iteration: unpack i16→i32 in two halves,
/// square, multiply by weights, widen to i64 and accumulate.
///
/// Overflow analysis: v² max = 65025, v²*w max = 65025*32767 ≈ 2.13B < i32::MAX.
/// Must go to i64 for accumulation since two i32 products can overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_screlu_dot(acc: &[i16], weights: &[i16], h: usize) -> i64 {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let mut sum0 = _mm256_setzero_si256(); // 4 × i64
    let mut sum1 = _mm256_setzero_si256(); // 4 × i64

    let mut i = 0;
    while i + 32 <= h {
        // === First 16 elements ===
        let v0 = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let c0 = _mm256_min_epi16(_mm256_max_epi16(v0, zero), qa);
        let w0 = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);

        let v0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c0));
        let w0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w0));
        let p0_lo = _mm256_mullo_epi32(_mm256_mullo_epi32(v0_lo, v0_lo), w0_lo);

        let v0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c0, 1));
        let w0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w0, 1));
        let p0_hi = _mm256_mullo_epi32(_mm256_mullo_epi32(v0_hi, v0_hi), w0_hi);

        // === Second 16 elements ===
        let v1 = _mm256_loadu_si256(acc.as_ptr().add(i + 16) as *const __m256i);
        let c1 = _mm256_min_epi16(_mm256_max_epi16(v1, zero), qa);
        let w1 = _mm256_loadu_si256(weights.as_ptr().add(i + 16) as *const __m256i);

        let v1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c1));
        let w1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w1));
        let p1_lo = _mm256_mullo_epi32(_mm256_mullo_epi32(v1_lo, v1_lo), w1_lo);

        let v1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c1, 1));
        let w1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w1, 1));
        let p1_hi = _mm256_mullo_epi32(_mm256_mullo_epi32(v1_hi, v1_hi), w1_hi);

        // Widen i32 → i64 and accumulate (8 widenings for 32 elements)
        sum0 = _mm256_add_epi64(sum0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p0_lo)));
        sum1 = _mm256_add_epi64(sum1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p0_lo, 1)));
        sum0 = _mm256_add_epi64(sum0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p0_hi)));
        sum1 = _mm256_add_epi64(sum1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p0_hi, 1)));
        sum0 = _mm256_add_epi64(sum0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p1_lo)));
        sum1 = _mm256_add_epi64(sum1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p1_lo, 1)));
        sum0 = _mm256_add_epi64(sum0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p1_hi)));
        sum1 = _mm256_add_epi64(sum1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p1_hi, 1)));

        i += 32;
    }

    // Handle remaining 16 elements (if h is not a multiple of 32)
    while i < h {
        let v = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
        let w = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);

        let v_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(clamped));
        let w_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
        let prod_lo = _mm256_mullo_epi32(_mm256_mullo_epi32(v_lo, v_lo), w_lo);

        let v_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(clamped, 1));
        let w_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));
        let prod_hi = _mm256_mullo_epi32(_mm256_mullo_epi32(v_hi, v_hi), w_hi);

        sum0 = _mm256_add_epi64(sum0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod_lo)));
        sum1 = _mm256_add_epi64(sum1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod_lo, 1)));
        sum0 = _mm256_add_epi64(sum0, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod_hi)));
        sum1 = _mm256_add_epi64(sum1, _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod_hi, 1)));

        i += 16;
    }

    hsum_epi64(_mm256_add_epi64(sum0, sum1))
}

/// Fast SCReLU dot product using int8-quantized weights.
/// Since v ∈ [0,255] and w_i8 ∈ [-127,127], v*w fits in i16 (max 255*127 = 32385).
/// Then madd_epi16(v, v*w) gives pairwise i32 sums of v²*w, staying in i16/i32.
/// Returns i32 sum (no i64 needed for h ≤ 2048).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_screlu_dot_i8(acc: &[i16], weights_i8: &[i16], h: usize) -> i32 {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let mut sum0 = _mm256_setzero_si256(); // 8 × i32
    let mut sum1 = _mm256_setzero_si256(); // 8 × i32

    let mut i = 0;
    while i + 32 <= h {
        // First 16 elements
        let v0 = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let c0 = _mm256_min_epi16(_mm256_max_epi16(v0, zero), qa);
        let w0 = _mm256_loadu_si256(weights_i8.as_ptr().add(i) as *const __m256i);
        let vw0 = _mm256_mullo_epi16(c0, w0); // v*w in i16 (fits: 255*127=32385)
        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(c0, vw0)); // pairwise v*(v*w) = v²*w in i32

        // Second 16 elements
        let v1 = _mm256_loadu_si256(acc.as_ptr().add(i + 16) as *const __m256i);
        let c1 = _mm256_min_epi16(_mm256_max_epi16(v1, zero), qa);
        let w1 = _mm256_loadu_si256(weights_i8.as_ptr().add(i + 16) as *const __m256i);
        let vw1 = _mm256_mullo_epi16(c1, w1);
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(c1, vw1));

        i += 32;
    }

    while i < h {
        let v = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
        let w = _mm256_loadu_si256(weights_i8.as_ptr().add(i) as *const __m256i);
        let vw = _mm256_mullo_epi16(clamped, w);
        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(clamped, vw));
        i += 16;
    }

    let total = _mm256_add_epi32(sum0, sum1);
    // Horizontal sum of 8 × i32
    let lo = _mm256_castsi256_si128(total);
    let hi = _mm256_extracti128_si256(total, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b_00_00_00_01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

/// Horizontal sum of 8 × i32 in a __m256i to i64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32_to_i64(v: __m256i) -> i64 {
    // Extract high and low 128-bit lanes, add as i32
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(lo, hi); // 4 × i32
    // Shuffle and add pairs
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64); // 2 × i32 in low 64 bits
    let hi32 = _mm_shuffle_epi32(sum64, 0b_00_00_00_01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32) as i64
}

/// Horizontal sum of 4 × i64 in a __m256i.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi64(v: __m256i) -> i64 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi64(lo, hi); // 2 × i64
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi64(sum128, hi64);
    _mm_cvtsi128_si64(sum64)
}

// ---- AVX-512 SIMD helper functions ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_acc_add(acc: &mut [i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
        let b = _mm512_loadu_si512(row.as_ptr().add(i) as *const __m512i);
        _mm512_storeu_si512(acc.as_mut_ptr().add(i) as *mut __m512i, _mm512_add_epi16(a, b));
        i += 32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_acc_sub(acc: &mut [i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
        let b = _mm512_loadu_si512(row.as_ptr().add(i) as *const __m512i);
        _mm512_storeu_si512(acc.as_mut_ptr().add(i) as *mut __m512i, _mm512_sub_epi16(a, b));
        i += 32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_acc_copy_add(dst: &mut [i16], src: &[i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
        let b = _mm512_loadu_si512(row.as_ptr().add(i) as *const __m512i);
        _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, _mm512_add_epi16(a, b));
        i += 32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_acc_copy_sub(dst: &mut [i16], src: &[i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i);
        let b = _mm512_loadu_si512(row.as_ptr().add(i) as *const __m512i);
        _mm512_storeu_si512(dst.as_mut_ptr().add(i) as *mut __m512i, _mm512_sub_epi16(a, b));
        i += 32;
    }
}

/// AVX-512 CReLU dot product. Processes 32 × i16 per iteration.
/// Uses periodic i64 drain like AVX2 version to prevent i32 overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_crelu_dot(acc: &[i16], weights: &[i16], h: usize) -> i64 {
    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(QA as i16);
    let mut sum32 = _mm512_setzero_si512(); // 16 × i32
    let mut total: i64 = 0;
    let mut count = 0u32;

    let mut i = 0;
    while i < h {
        let v = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
        let clamped = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
        let w = _mm512_loadu_si512(weights.as_ptr().add(i) as *const __m512i);
        sum32 = _mm512_add_epi32(sum32, _mm512_madd_epi16(clamped, w));

        count += 32;
        if count >= 128 {
            total += _mm512_reduce_add_epi32(sum32) as i64;
            sum32 = _mm512_setzero_si512();
            count = 0;
        }
        i += 32;
    }
    if count > 0 {
        total += _mm512_reduce_add_epi32(sum32) as i64;
    }
    total
}

/// AVX-512 SCReLU dot product with int8 weights. Processes 32 × i16 per iteration.
/// Uses periodic i64 drain to prevent i32 overflow in horizontal reduce.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_screlu_dot_i8(acc: &[i16], weights_i8: &[i16], h: usize) -> i64 {
    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(QA as i16);
    let mut sum0 = _mm512_setzero_si512(); // 16 × i32
    let mut sum1 = _mm512_setzero_si512();
    let mut total: i64 = 0;
    let mut count = 0u32;

    let mut i = 0;
    while i + 64 <= h {
        let v0 = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
        let c0 = _mm512_min_epi16(_mm512_max_epi16(v0, zero), qa);
        let w0 = _mm512_loadu_si512(weights_i8.as_ptr().add(i) as *const __m512i);
        sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(c0, _mm512_mullo_epi16(c0, w0)));

        let v1 = _mm512_loadu_si512(acc.as_ptr().add(i + 32) as *const __m512i);
        let c1 = _mm512_min_epi16(_mm512_max_epi16(v1, zero), qa);
        let w1 = _mm512_loadu_si512(weights_i8.as_ptr().add(i + 32) as *const __m512i);
        sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(c1, _mm512_mullo_epi16(c1, w1)));

        count += 64;
        if count >= 512 {
            total += _mm512_reduce_add_epi32(_mm512_add_epi32(sum0, sum1)) as i64;
            sum0 = _mm512_setzero_si512();
            sum1 = _mm512_setzero_si512();
            count = 0;
        }
        i += 64;
    }

    while i < h {
        let v = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
        let clamped = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
        let w = _mm512_loadu_si512(weights_i8.as_ptr().add(i) as *const __m512i);
        sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(clamped, _mm512_mullo_epi16(clamped, w)));
        i += 32;
    }

    total + _mm512_reduce_add_epi32(_mm512_add_epi32(sum0, sum1)) as i64
}

// ---- NEON SIMD helper functions (aarch64) ----

/// Add a weight row to an accumulator (NEON, 8 × i16 per iteration).
#[cfg(target_arch = "aarch64")]
unsafe fn neon_acc_add(acc: &mut [i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = vld1q_s16(acc.as_ptr().add(i));
        let b = vld1q_s16(row.as_ptr().add(i));
        vst1q_s16(acc.as_mut_ptr().add(i), vaddq_s16(a, b));
        i += 8;
    }
}

/// Subtract a weight row from an accumulator (NEON).
#[cfg(target_arch = "aarch64")]
unsafe fn neon_acc_sub(acc: &mut [i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = vld1q_s16(acc.as_ptr().add(i));
        let b = vld1q_s16(row.as_ptr().add(i));
        vst1q_s16(acc.as_mut_ptr().add(i), vsubq_s16(a, b));
        i += 8;
    }
}

/// Fused copy + add: dst = src + row (NEON).
#[cfg(target_arch = "aarch64")]
unsafe fn neon_acc_copy_add(dst: &mut [i16], src: &[i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = vld1q_s16(src.as_ptr().add(i));
        let b = vld1q_s16(row.as_ptr().add(i));
        vst1q_s16(dst.as_mut_ptr().add(i), vaddq_s16(a, b));
        i += 8;
    }
}

/// Fused copy + sub: dst = src - row (NEON).
#[cfg(target_arch = "aarch64")]
unsafe fn neon_acc_copy_sub(dst: &mut [i16], src: &[i16], row: &[i16], h: usize) {
    let mut i = 0;
    while i < h {
        let a = vld1q_s16(src.as_ptr().add(i));
        let b = vld1q_s16(row.as_ptr().add(i));
        vst1q_s16(dst.as_mut_ptr().add(i), vsubq_s16(a, b));
        i += 8;
    }
}

/// CReLU dot product (NEON): clamp [0, QA=255], dot with weights.
/// Processes 8 × i16 per iteration. Uses i64 accumulation for safety.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_crelu_dot(acc: &[i16], weights: &[i16], h: usize) -> i64 {
    let zero = vdupq_n_s16(0);
    let qa = vdupq_n_s16(QA as i16);
    let mut sum0 = vdupq_n_s32(0); // 4 × i32
    let mut sum1 = vdupq_n_s32(0);
    let mut total: i64 = 0;
    let mut count = 0u32;

    let mut i = 0;
    while i < h {
        let v = vld1q_s16(acc.as_ptr().add(i));
        let clamped = vminq_s16(vmaxq_s16(v, zero), qa);
        let w = vld1q_s16(weights.as_ptr().add(i));
        // Multiply-accumulate: lo 4 pairs and hi 4 pairs
        let prod_lo = vmull_s16(vget_low_s16(clamped), vget_low_s16(w));
        let prod_hi = vmull_high_s16(clamped, w);
        sum0 = vaddq_s32(sum0, prod_lo);
        sum1 = vaddq_s32(sum1, prod_hi);

        count += 8;
        if count >= 128 {
            let combined = vaddq_s32(sum0, sum1);
            total += vaddlvq_s32(combined) as i64;
            sum0 = vdupq_n_s32(0);
            sum1 = vdupq_n_s32(0);
            count = 0;
        }
        i += 8;
    }

    if count > 0 {
        let combined = vaddq_s32(sum0, sum1);
        total += vaddlvq_s32(combined) as i64;
    }
    total
}

/// SCReLU dot product with int8 weights (NEON): clamp [0,255], square, dot.
/// v*w_i8 fits in i16 (255*127=32385), then v*(v*w) via madd gives i32.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_screlu_dot_i8(acc: &[i16], weights_i8: &[i16], h: usize) -> i32 {
    let zero = vdupq_n_s16(0);
    let qa = vdupq_n_s16(QA as i16);
    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);

    let mut i = 0;
    while i < h {
        let v = vld1q_s16(acc.as_ptr().add(i));
        let clamped = vminq_s16(vmaxq_s16(v, zero), qa);
        let w = vld1q_s16(weights_i8.as_ptr().add(i));
        // v*w in i16
        let vw = vmulq_s16(clamped, w);
        // v*(v*w) = v²*w: multiply lo/hi halves → i32, accumulate
        sum0 = vmlal_s16(sum0, vget_low_s16(clamped), vget_low_s16(vw));
        sum1 = vmlal_high_s16(sum1, clamped, vw);

        i += 8;
    }

    let combined = vaddq_s32(sum0, sum1);
    vaddvq_s32(combined)
}

/// Detect AVX2 support at runtime.
fn detect_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Detect AVX-512 (F + BW) support at runtime.
fn detect_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Detect NEON support (always true on aarch64).
fn detect_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    { true }
    #[cfg(not(target_arch = "aarch64"))]
    { false }
}

/// NNUE network weights (shared, read-only after loading).
pub struct NNUENet {
    pub hidden_size: usize,
    pub input_weights: Vec<i16>,  // [NNUE_INPUT_SIZE × hidden_size]
    pub input_biases: Vec<i16>,   // [hidden_size]
    pub output_weights: Vec<i16>, // [NNUE_OUTPUT_BUCKETS × 2 × hidden_size]
    /// Quantized output weights for fast SCReLU: i16 but clamped to [-128, 127] range.
    /// Stored alongside scale factor per bucket for precision recovery.
    pub output_weights_i8: Vec<i16>, // [NNUE_OUTPUT_BUCKETS × 2 × hidden_size], values in i8 range
    pub output_scale: [f32; NNUE_OUTPUT_BUCKETS], // per-bucket scale factor
    pub output_bias: [i32; NNUE_OUTPUT_BUCKETS],
    pub use_screlu: bool,
    pub use_pairwise: bool,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
}

impl NNUENet {
    /// Load a v5/v6 .nnue file.
    pub fn load(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let mut reader = BufReader::new(file);

        // Read magic
        let magic = read_u32(&mut reader)?;
        if magic != NNUE_MAGIC {
            return Err(format!("invalid NNUE magic: 0x{:X}", magic));
        }

        // Read version
        let version = read_u32(&mut reader)?;
        let mut use_screlu = false;
        let mut use_pairwise = false;

        let header_size: u64 = match version {
            5 => 8, // magic + version
            6 => {
                let flags = read_u8(&mut reader)?;
                use_screlu = flags & 1 != 0;
                use_pairwise = flags & 2 != 0;
                9
            }
            _ => return Err(format!("unsupported NNUE version: {}", version)),
        };

        // Infer hidden size from file size
        let file_meta = std::fs::metadata(path).map_err(|e| format!("stat {}: {}", path, e))?;
        let file_size = file_meta.len();
        let body_size = file_size - header_size;

        // body = InputWeights(NNUE_INPUT_SIZE * H * 2) + InputBiases(H * 2)
        //      + OutputWeights(8 * out_mul * H * 2) + OutputBias(8 * 4)
        // out_mul = 2 (plain: both perspectives) or 1 (pairwise: halved)
        // body - 32 = 2*H*(NNUE_INPUT_SIZE + 1 + 8*out_mul)
        let out_mul: u64 = if use_pairwise { 8 } else { 16 };
        let h_denom = 2 * (NNUE_INPUT_SIZE as u64 + 1 + out_mul);
        let h_numer = body_size - 32;
        if h_numer % h_denom != 0 {
            return Err(format!("cannot infer hidden size from file size {} (body {})", file_size, body_size));
        }
        let hidden_size = (h_numer / h_denom) as usize;

        // Read input weights
        let mut input_weights = vec![0i16; NNUE_INPUT_SIZE * hidden_size];
        read_i16_slice(&mut reader, &mut input_weights)?;

        // Read input biases
        let mut input_biases = vec![0i16; hidden_size];
        read_i16_slice(&mut reader, &mut input_biases)?;

        // Read output weights: 8 buckets × out_width
        // Plain: out_width = 2*H (stm + ntm), Pairwise: out_width = H (pairwise halves it)
        let out_width = if use_pairwise { hidden_size } else { 2 * hidden_size };
        let mut output_weights = vec![0i16; NNUE_OUTPUT_BUCKETS * out_width];
        read_i16_slice(&mut reader, &mut output_weights)?;

        // Read output bias
        let mut output_bias = [0i32; NNUE_OUTPUT_BUCKETS];
        for i in 0..NNUE_OUTPUT_BUCKETS {
            output_bias[i] = read_i32(&mut reader)?;
        }

        let activation = if use_pairwise { "pairwise" } else if use_screlu { "SCReLU" } else { "CReLU" };
        println!("info string Loaded NNUE v{} {} {} ({})", version, path, activation, hidden_size);

        let has_avx2 = detect_avx2();
        let has_avx512 = detect_avx512();
        let has_neon = detect_neon();
        if has_avx512 {
            println!("info string AVX-512 SIMD detected — using 512-bit NNUE inference");
        } else if has_avx2 {
            println!("info string AVX2 SIMD detected — using vectorised NNUE inference");
        } else if has_neon {
            println!("info string NEON SIMD detected — using vectorised NNUE inference");
        }

        // Quantize output weights to i8 range for fast SCReLU dot product.
        // Per-bucket scale: max_abs(weights) / 127.
        // This lets us use mullo_epi16 + madd_epi16 since v*w_i8 fits in i16.
        let mut output_weights_i8 = vec![0i16; NNUE_OUTPUT_BUCKETS * out_width];
        let mut output_scale = [1.0f32; NNUE_OUTPUT_BUCKETS];

        if use_screlu {
            for bucket in 0..NNUE_OUTPUT_BUCKETS {
                let off = bucket * out_width;
                let slice = &output_weights[off..off + out_width];
                let max_abs = slice.iter().map(|&w| (w as i32).abs()).max().unwrap_or(1).max(1);
                let scale = max_abs as f32 / 127.0;
                output_scale[bucket] = scale;

                for j in 0..out_width {
                    let quantized = ((output_weights[off + j] as f32 / scale).round() as i32).clamp(-127, 127);
                    output_weights_i8[off + j] = quantized as i16;
                }
            }

            // Report quantization stats
            let total = output_weights.len();
            let mut max_err: f32 = 0.0;
            let mut sum_err: f64 = 0.0;
            for bucket in 0..NNUE_OUTPUT_BUCKETS {
                let off = bucket * out_width;
                let scale = output_scale[bucket];
                for j in 0..out_width {
                    let orig = output_weights[off + j] as f32;
                    let reconstructed = output_weights_i8[off + j] as f32 * scale;
                    let err = (orig - reconstructed).abs();
                    max_err = max_err.max(err);
                    sum_err += err as f64;
                }
            }
            println!("info string SCReLU int8 quantization: max_err={:.1} avg_err={:.2}",
                max_err, sum_err / total as f64);
        }

        Ok(NNUENet {
            hidden_size,
            input_weights,
            input_biases,
            output_weights,
            output_weights_i8,
            output_scale,
            output_bias,
            use_screlu,
            use_pairwise,
            has_avx2,
            has_avx512,
            has_neon,
        })
    }

    /// Get input weight row for a feature index.
    #[inline]
    pub fn input_weight_row(&self, idx: usize) -> &[i16] {
        let off = idx * self.hidden_size;
        &self.input_weights[off..off + self.hidden_size]
    }

    /// Output width per bucket: 2*H (plain) or H (pairwise).
    #[inline]
    fn output_width(&self) -> usize {
        if self.use_pairwise { self.hidden_size } else { 2 * self.hidden_size }
    }

    /// Get output weight row for a bucket.
    #[inline]
    fn output_weight_row(&self, bucket: usize) -> &[i16] {
        let w = self.output_width();
        let off = bucket * w;
        &self.output_weights[off..off + w]
    }

    /// Get int8-quantized output weight row for a bucket.
    #[inline]
    fn output_weight_row_i8(&self, bucket: usize) -> &[i16] {
        let w = self.output_width();
        let off = bucket * w;
        &self.output_weights_i8[off..off + w]
    }

    /// Forward pass: CReLU or SCReLU activation → dot product with output weights.
    /// Returns centipawns from side-to-move perspective.
    pub fn forward(&self, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        let bucket = output_bucket(piece_count);
        let h = self.hidden_size;
        let out_w = self.output_weight_row(bucket);

        let (stm_acc, ntm_acc) = if stm == WHITE {
            (acc.white(), acc.black())
        } else {
            (acc.black(), acc.white())
        };

        let mut output = self.output_bias[bucket] as i64;

        // Pairwise path: split acc into halves, clamp, multiply pairs, dot with weights
        if self.use_pairwise {
            let pw = h / 2; // pairwise output size per perspective

            #[cfg(target_arch = "x86_64")]
            if self.has_avx2 && pw % 16 == 0 {
                let bias = output; // save bias before adding dot products
                let mut sum: i64;
                unsafe {
                    sum = simd_pairwise_dot(&stm_acc[..pw], &stm_acc[pw..], &out_w[..pw], pw);
                    sum += simd_pairwise_dot(&ntm_acc[..pw], &ntm_acc[pw..], &out_w[pw..], pw);
                }
                output = sum / QA as i64 + bias;
                let result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
                return result;
            }

            // Scalar fallback
            for i in 0..pw {
                let a = (stm_acc[i] as i32).clamp(0, QA as i32);
                let b = (stm_acc[i + pw] as i32).clamp(0, QA as i32);
                output += ((a * b) / QA as i32) as i64 * out_w[i] as i64;
            }
            for i in 0..pw {
                let a = (ntm_acc[i] as i32).clamp(0, QA as i32);
                let b = (ntm_acc[i + pw] as i32).clamp(0, QA as i32);
                output += ((a * b) / QA as i32) as i64 * out_w[pw + i] as i64;
            }

            let result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
            return result;
        }

        #[cfg(target_arch = "x86_64")]
        if self.has_avx512 && h % 32 == 0 {
            if self.use_screlu {
                let out_w_i8 = self.output_weight_row_i8(bucket);
                let scale = self.output_scale[bucket];
                unsafe {
                    let stm_sum = simd512_screlu_dot_i8(stm_acc, &out_w_i8[..h], h);
                    let ntm_sum = simd512_screlu_dot_i8(ntm_acc, &out_w_i8[h..], h);
                    if scale == 1.0 {
                        output += stm_sum + ntm_sum;
                    } else {
                        output += ((stm_sum + ntm_sum) as f64 * scale as f64) as i64;
                    }
                }
                output /= QA as i64;
            } else {
                unsafe {
                    output += simd512_crelu_dot(stm_acc, &out_w[..h], h);
                    output += simd512_crelu_dot(ntm_acc, &out_w[h..], h);
                }
            }

            let mut result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
            if self.use_screlu {
                result = result * 4 / 5;
            }
            return result;
        }

        #[cfg(target_arch = "x86_64")]
        if self.has_avx2 && h % 16 == 0 {
            if self.use_screlu {
                let out_w_i8 = self.output_weight_row_i8(bucket);
                let scale = self.output_scale[bucket];
                unsafe {
                    let stm_sum = simd_screlu_dot_i8(stm_acc, &out_w_i8[..h], h) as i64;
                    let ntm_sum = simd_screlu_dot_i8(ntm_acc, &out_w_i8[h..], h) as i64;
                    if scale == 1.0 {
                        output += stm_sum + ntm_sum;
                    } else {
                        output += ((stm_sum + ntm_sum) as f64 * scale as f64) as i64;
                    }
                }
                output /= QA as i64;
            } else {
                unsafe {
                    output += simd_crelu_dot(stm_acc, &out_w[..h], h);
                    output += simd_crelu_dot(ntm_acc, &out_w[h..], h);
                }
            }

            let mut result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
            if self.use_screlu {
                result = result * 4 / 5;
            }
            return result;
        }

        #[cfg(target_arch = "aarch64")]
        if self.has_neon && h % 8 == 0 {
            if self.use_screlu {
                let out_w_i8 = self.output_weight_row_i8(bucket);
                let scale = self.output_scale[bucket];
                unsafe {
                    let stm_sum = neon_screlu_dot_i8(stm_acc, &out_w_i8[..h], h) as i64;
                    let ntm_sum = neon_screlu_dot_i8(ntm_acc, &out_w_i8[h..], h) as i64;
                    if scale == 1.0 {
                        output += stm_sum + ntm_sum;
                    } else {
                        output += ((stm_sum + ntm_sum) as f64 * scale as f64) as i64;
                    }
                }
                output /= QA as i64;
            } else {
                unsafe {
                    output += neon_crelu_dot(stm_acc, &out_w[..h], h);
                    output += neon_crelu_dot(ntm_acc, &out_w[h..], h);
                }
            }

            let mut result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
            if self.use_screlu {
                result = result * 4 / 5;
            }
            return result;
        }

        // Scalar fallback
        if self.use_screlu {
            // SCReLU: clamp [0, QA] then square
            for i in 0..h {
                let v = (stm_acc[i] as i32).clamp(0, QA);
                output += (v as i64 * v as i64) * out_w[i] as i64;
            }
            for i in 0..h {
                let v = (ntm_acc[i] as i32).clamp(0, QA);
                output += (v as i64 * v as i64) * out_w[h + i] as i64;
            }
            // SCReLU output is at scale QA² × QB (squared activation)
            // Divide by QA to get back to QA × QB scale
            output /= QA as i64;
        } else {
            // CReLU: clamp [0, QA]
            for i in 0..h {
                let v = (stm_acc[i] as i32).clamp(0, QA);
                output += v as i64 * out_w[i] as i64;
            }
            for i in 0..h {
                let v = (ntm_acc[i] as i32).clamp(0, QA);
                output += v as i64 * out_w[h + i] as i64;
            }
        }

        // Scale: output is at QA × QB = 16320
        // centipawns = output * 400 / 16320 (do division in i64 to avoid overflow)
        let mut result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;

        // SCReLU scale correction: squared activation has wider dynamic range
        if self.use_screlu {
            result = result * 4 / 5;
        }

        result
    }
}

/// Dirty piece info for lazy materialization.
#[derive(Clone, Copy)]
pub struct DirtyPiece {
    /// 0 = needs full recompute (king bucket change), 1+ = incremental
    pub kind: u8,
    pub changes: [(bool, u8, u8, u8); 5], // (add, color, pt, sq)
    pub n_changes: u8,
}

impl DirtyPiece {
    pub fn recompute() -> Self {
        DirtyPiece { kind: 0, changes: [(false, 0, 0, 0); 5], n_changes: 0 }
    }
    pub fn incremental(changes: &[(bool, u8, u8, u8)]) -> Self {
        let mut d = DirtyPiece { kind: 1, changes: [(false, 0, 0, 0); 5], n_changes: changes.len() as u8 };
        for (i, &c) in changes.iter().enumerate().take(5) {
            d.changes[i] = c;
        }
        d
    }
}

/// Single accumulator entry (two perspectives).
#[derive(Clone)]
pub struct AccEntry {
    pub white: Vec<i16>,
    pub black: Vec<i16>,
    computed: bool,
    dirty: DirtyPiece,
}

/// Finny table entry: cached accumulator for a specific king bucket.
struct FinnyEntry {
    acc: Vec<i16>,                      // cached accumulator values
    piece_bbs: ([Bitboard; 6], [Bitboard; 2]), // piece and color bitboards when cached
    valid: bool,
}

const FINNY_SIZE: usize = 2 * NNUE_KING_BUCKETS * 2; // [perspective][bucket][mirror]

/// Accumulator stack with lazy materialization and Finny table.
pub struct NNUEAccumulator {
    stack: Vec<AccEntry>,
    top: usize,
    hidden_size: usize,
    /// Finny table: flat [perspective * 32 + bucket * 2 + mirror]
    finny: Vec<FinnyEntry>, // length = FINNY_SIZE (64)
}

impl NNUEAccumulator {
    pub fn new(hidden_size: usize) -> Self {
        let mut stack = Vec::with_capacity(256);
        for _ in 0..256 {
            stack.push(AccEntry {
                white: vec![0; hidden_size],
                black: vec![0; hidden_size],
                computed: false,
                dirty: DirtyPiece::recompute(),
            });
        }
        // Build finny table (flat array)
        let mut finny = Vec::with_capacity(FINNY_SIZE);
        for _ in 0..FINNY_SIZE {
            finny.push(FinnyEntry {
                acc: vec![0; hidden_size],
                piece_bbs: ([0; 6], [0; 2]),
                valid: false,
            });
        }
        NNUEAccumulator { stack, top: 0, hidden_size, finny }
    }

    pub fn white(&self) -> &[i16] {
        &self.stack[self.top].white[..self.hidden_size]
    }

    pub fn black(&self) -> &[i16] {
        &self.stack[self.top].black[..self.hidden_size]
    }

    /// Get the current accumulator entry for reading.
    pub fn current(&self) -> &AccEntry {
        &self.stack[self.top]
    }

    /// Force a full recompute of both perspectives (for debugging).
    pub fn force_recompute(&mut self, net: &NNUENet, board: &Board) {
        let h = self.hidden_size;
        // White perspective
        {
            let dst = &mut self.stack[self.top].white;
            dst[..h].copy_from_slice(&net.input_biases[..h]);
            for color in 0..2u8 {
                for pt in 0..6u8 {
                    let mut bb = board.pieces[pt as usize] & board.colors[color as usize];
                    while bb != 0 {
                        let sq = pop_lsb(&mut bb) as u8;
                        let idx = halfka_index(WHITE, board.king_sq(WHITE), color, pt, sq);
                        let row = net.input_weight_row(idx);
                        for j in 0..h { dst[j] += row[j]; }
                    }
                }
            }
        }
        // Black perspective
        {
            let dst = &mut self.stack[self.top].black;
            dst[..h].copy_from_slice(&net.input_biases[..h]);
            for color in 0..2u8 {
                for pt in 0..6u8 {
                    let mut bb = board.pieces[pt as usize] & board.colors[color as usize];
                    while bb != 0 {
                        let sq = pop_lsb(&mut bb) as u8;
                        let idx = halfka_index(BLACK, board.king_sq(BLACK), color, pt, sq);
                        let row = net.input_weight_row(idx);
                        for j in 0..h { dst[j] += row[j]; }
                    }
                }
            }
        }
        self.stack[self.top].computed = true;
    }

    /// Push: store dirty info, don't compute yet.
    pub fn push(&mut self, dirty: DirtyPiece) {
        self.top += 1;
        if self.top >= self.stack.len() {
            self.stack.push(AccEntry {
                white: vec![0; self.hidden_size],
                black: vec![0; self.hidden_size],
                computed: false,
                dirty: DirtyPiece::recompute(),
            });
        }
        self.stack[self.top].computed = false;
        self.stack[self.top].dirty = dirty;
    }

    pub fn pop(&mut self) {
        debug_assert!(self.top > 0);
        self.top -= 1;
    }

    /// Materialize: ensure current accumulator is computed.
    pub fn materialize(&mut self, net: &NNUENet, board: &Board) {
        if self.stack[self.top].computed {
            return;
        }

        let dirty = self.stack[self.top].dirty;

        // Full recompute needed?
        if dirty.kind == 0 || self.top == 0 || !self.stack[self.top - 1].computed {
            self.refresh_accumulator(net, board, WHITE);
            self.refresh_accumulator(net, board, BLACK);
            self.stack[self.top].computed = true;
            return;
        }

        // Incremental: fuse first delta with parent copy, then apply remaining in-place.
        // This eliminates the separate copy_from_slice (saves ~100ns for h=1024).
        let h = self.hidden_size;
        let (left, right) = self.stack.split_at_mut(self.top);
        let parent = &left[self.top - 1];
        let current = &mut right[0];

        let w_king_sq = board.king_sq(WHITE);
        let b_king_sq = board.king_sq(BLACK);

        let n = dirty.n_changes as usize;

        // First change: fused copy+delta (reads parent, writes current)
        {
            let (add, color, pt, sq) = dirty.changes[0];
            let w_idx = halfka_index(WHITE, w_king_sq, color, pt, sq);
            let w_row = net.input_weight_row(w_idx);
            let b_idx = halfka_index(BLACK, b_king_sq, color, pt, sq);
            let b_row = net.input_weight_row(b_idx);

            let mut handled = false;
            #[cfg(target_arch = "x86_64")]
            if net.has_avx512 && h % 32 == 0 {
                unsafe {
                    if add {
                        simd512_acc_copy_add(&mut current.white, &parent.white, w_row, h);
                        simd512_acc_copy_add(&mut current.black, &parent.black, b_row, h);
                    } else {
                        simd512_acc_copy_sub(&mut current.white, &parent.white, w_row, h);
                        simd512_acc_copy_sub(&mut current.black, &parent.black, b_row, h);
                    }
                }
                handled = true;
            }
            #[cfg(target_arch = "x86_64")]
            if !handled && net.has_avx2 && h % 16 == 0 {
                unsafe {
                    if add {
                        simd_acc_copy_add(&mut current.white, &parent.white, w_row, h);
                        simd_acc_copy_add(&mut current.black, &parent.black, b_row, h);
                    } else {
                        simd_acc_copy_sub(&mut current.white, &parent.white, w_row, h);
                        simd_acc_copy_sub(&mut current.black, &parent.black, b_row, h);
                    }
                }
                handled = true;
            }
            #[cfg(target_arch = "aarch64")]
            if !handled && net.has_neon && h % 8 == 0 {
                unsafe {
                    if add {
                        neon_acc_copy_add(&mut current.white, &parent.white, w_row, h);
                        neon_acc_copy_add(&mut current.black, &parent.black, b_row, h);
                    } else {
                        neon_acc_copy_sub(&mut current.white, &parent.white, w_row, h);
                        neon_acc_copy_sub(&mut current.black, &parent.black, b_row, h);
                    }
                }
                handled = true;
            }
            if !handled {
                if add {
                    for j in 0..h {
                        current.white[j] = parent.white[j] + w_row[j];
                        current.black[j] = parent.black[j] + b_row[j];
                    }
                } else {
                    for j in 0..h {
                        current.white[j] = parent.white[j] - w_row[j];
                        current.black[j] = parent.black[j] - b_row[j];
                    }
                }
            }
        }

        // Remaining changes: in-place on current
        for i in 1..n {
            let (add, color, pt, sq) = dirty.changes[i];
            let w_idx = halfka_index(WHITE, w_king_sq, color, pt, sq);
            let w_row = net.input_weight_row(w_idx);
            let b_idx = halfka_index(BLACK, b_king_sq, color, pt, sq);
            let b_row = net.input_weight_row(b_idx);

            #[cfg(target_arch = "x86_64")]
            if net.has_avx2 && h % 16 == 0 {
                unsafe {
                    if add {
                        simd_acc_add(&mut current.white, w_row, h);
                        simd_acc_add(&mut current.black, b_row, h);
                    } else {
                        simd_acc_sub(&mut current.white, w_row, h);
                        simd_acc_sub(&mut current.black, b_row, h);
                    }
                }
                continue;
            }

            #[cfg(target_arch = "aarch64")]
            if net.has_neon && h % 8 == 0 {
                unsafe {
                    if add {
                        neon_acc_add(&mut current.white, w_row, h);
                        neon_acc_add(&mut current.black, b_row, h);
                    } else {
                        neon_acc_sub(&mut current.white, w_row, h);
                        neon_acc_sub(&mut current.black, b_row, h);
                    }
                }
                continue;
            }

            if add {
                for j in 0..h { current.white[j] += w_row[j]; current.black[j] += b_row[j]; }
            } else {
                for j in 0..h { current.white[j] -= w_row[j]; current.black[j] -= b_row[j]; }
            }
        }

        current.computed = true;
    }

    /// Refresh one perspective using the Finny table.
    /// Diffs cached vs current piece bitboards, applies only changed features.
    fn refresh_accumulator(&mut self, net: &NNUENet, board: &Board, perspective: u8) {
        let h = self.hidden_size;
        let king_sq = board.king_sq(perspective);
        let mut ks = king_sq as usize;
        if perspective == BLACK { ks ^= 56; }

        let bucket = king_bucket(ks);
        let mirror_idx = if king_mirror(ks) { 1 } else { 0 };

        let entry = &mut self.finny[perspective as usize * 32 + bucket * 2 + mirror_idx];

        let dst = if perspective == WHITE {
            &mut self.stack[self.top].white
        } else {
            &mut self.stack[self.top].black
        };

        if !entry.valid {
            // No cache — full recompute
            dst[..h].copy_from_slice(&net.input_biases[..h]);
            for color in 0..2u8 {
                for pt in 0..6u8 {
                    let mut bb = board.pieces[pt as usize] & board.colors[color as usize];
                    while bb != 0 {
                        let sq = pop_lsb(&mut bb) as u8;
                        let idx = halfka_index(perspective, king_sq, color, pt, sq);
                        let row = net.input_weight_row(idx);
                        acc_add(net, dst, row, h);
                    }
                }
            }
            // Save to cache
            entry.acc[..h].copy_from_slice(&dst[..h]);
            entry.piece_bbs = (board.pieces, board.colors);
            entry.valid = true;
            return;
        }

        // Diff cached vs current — apply only changed features to cached acc
        let cached_acc = &mut entry.acc;

        for color in 0..2u8 {
            for pt in 0..6u8 {
                let prev = entry.piece_bbs.0[pt as usize] & entry.piece_bbs.1[color as usize];
                let curr = board.pieces[pt as usize] & board.colors[color as usize];
                if prev == curr { continue; }

                // Removed: in prev but not curr
                let mut removed = prev & !curr;
                while removed != 0 {
                    let sq = pop_lsb(&mut removed) as u8;
                    let idx = halfka_index(perspective, king_sq, color, pt, sq);
                    let row = net.input_weight_row(idx);
                    acc_sub(net, cached_acc, row, h);
                }

                // Added: in curr but not prev
                let mut added = curr & !prev;
                while added != 0 {
                    let sq = pop_lsb(&mut added) as u8;
                    let idx = halfka_index(perspective, king_sq, color, pt, sq);
                    let row = net.input_weight_row(idx);
                    acc_add(net, cached_acc, row, h);
                }
            }
        }

        // Copy updated cache to accumulator
        dst[..h].copy_from_slice(&cached_acc[..h]);
        entry.piece_bbs = (board.pieces, board.colors);
    }

    /// Reset to bottom of stack and invalidate Finny table.
    pub fn reset(&mut self) {
        self.top = 0;
        self.stack[0].computed = false;
        for entry in self.finny.iter_mut() {
            entry.valid = false;
        }
    }
}

/// Add a weight row to an accumulator (SIMD-aware).
#[inline]
fn acc_add(net: &NNUENet, acc: &mut [i16], row: &[i16], h: usize) {
    #[cfg(target_arch = "x86_64")]
    if net.has_avx512 && h % 32 == 0 {
        unsafe { simd512_acc_add(acc, row, h); }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if net.has_avx2 && h % 16 == 0 {
        unsafe { simd_acc_add(acc, row, h); }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if net.has_neon && h % 8 == 0 {
        unsafe { neon_acc_add(acc, row, h); }
        return;
    }
    for j in 0..h { acc[j] += row[j]; }
}

/// Subtract a weight row from an accumulator (SIMD-aware).
#[inline]
fn acc_sub(net: &NNUENet, acc: &mut [i16], row: &[i16], h: usize) {
    #[cfg(target_arch = "x86_64")]
    if net.has_avx512 && h % 32 == 0 {
        unsafe { simd512_acc_sub(acc, row, h); }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if net.has_avx2 && h % 16 == 0 {
        unsafe { simd_acc_sub(acc, row, h); }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if net.has_neon && h % 8 == 0 {
        unsafe { neon_acc_sub(acc, row, h); }
        return;
    }
    for j in 0..h { acc[j] -= row[j]; }
}

/// Count total pieces on the board.
pub fn piece_count(board: &Board) -> u32 {
    popcount(board.occupied())
}

// Binary read helpers
fn read_u8(r: &mut impl IoRead) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(|e| format!("read u8: {}", e))?;
    Ok(buf[0])
}

fn read_u32(r: &mut impl IoRead) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| format!("read u32: {}", e))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl IoRead) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| format!("read i32: {}", e))?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i16_slice(r: &mut impl IoRead, dst: &mut [i16]) -> Result<(), String> {
    // Read as bytes, then convert
    let byte_len = dst.len() * 2;
    let mut buf = vec![0u8; byte_len];
    r.read_exact(&mut buf).map_err(|e| format!("read i16 slice: {}", e))?;
    for i in 0..dst.len() {
        dst[i] = i16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_king_buckets() {
        init_nnue();
        // a1 (sq 0): file 0, rank 0 → mirrored_file=0, rank_group=0 → bucket 0
        assert_eq!(king_bucket(0), 0);
        // e1 (sq 4): file 4, rank 0 → mirrored_file=3, rank_group=0 → bucket 12, mirror=true
        assert_eq!(king_bucket(4), 12);
        assert!(king_mirror(4));
        // a1 no mirror
        assert!(!king_mirror(0));
    }

    #[test]
    fn test_halfka_index() {
        init_nnue();
        // White perspective, king on e1 (sq 4), white pawn on e2 (sq 12)
        let idx = halfka_index(WHITE, 4, WHITE, PAWN, 12);
        // king sq 4, file=4 >= 4, so mirror: ks=4 stays, ps=12 → file mirror: (12 & !7) | (7 - (12&7)) = 8 | (7-4) = 8|3 = 11
        // bucket = king_bucket(4) = 12
        // pi = 0 (white pawn from white perspective)
        // index = 12 * 768 + 0 * 64 + 11 = 9216 + 11 = 9227
        assert_eq!(idx, 9227);

        // Black perspective, king on e8 (sq 60), white pawn on e2 (sq 12)
        let idx = halfka_index(BLACK, 60, WHITE, PAWN, 12);
        // ks = 60 ^ 56 = 4, ps = 12 ^ 56 = 52
        // pi = 0 (white pawn) → black perspective: 0 + 6 = 6
        // king_mirror(4) = true → ps = (52 & !7) | (7 - (52 & 7)) = 48 | (7-4) = 48|3 = 51
        // bucket = king_bucket(4) = 12
        // index = 12 * 768 + 6 * 64 + 51 = 9216 + 384 + 51 = 9651
        assert_eq!(idx, 9651);
    }

    #[test]
    fn test_output_bucket() {
        assert_eq!(output_bucket(2), 0);
        assert_eq!(output_bucket(5), 0);
        assert_eq!(output_bucket(6), 1);
        assert_eq!(output_bucket(32), 7);
    }
}
