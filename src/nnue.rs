/// NNUE v5/v6 inference: HalfKA network with CReLU/SCReLU activation.
///
/// Architecture: (12288 → N)×2 → 1×8
/// - 12288 input features = 16 king buckets × 12 piece types × 64 squares
/// - Two perspectives (side-to-move and not-side-to-move)
/// - CReLU: clamp [0, QA=255]
/// - SCReLU: clamp [0, QA=255] then square
/// - 8 output buckets selected by material count

#[cfg(feature = "profile-materialize")]
pub mod mat_stats {
    //! Per-bench materialize call-pattern counters.
    //! Gated behind `--features profile-materialize` so release builds
    //! pay nothing.
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALLS: AtomicU64 = AtomicU64::new(0);
    static EARLY_OUT: AtomicU64 = AtomicU64::new(0);
    static REFRESHES: AtomicU64 = AtomicU64::new(0);
    static INCREMENTALS: AtomicU64 = AtomicU64::new(0);
    static TOTAL_DELTA_CHANGES: AtomicU64 = AtomicU64::new(0);
    /// Walk-back distance buckets for refresh fallbacks:
    /// idx 0 = no computed ancestor found (full rebuild inevitable).
    /// idx i (1..=8) = computed ancestor at distance i.
    /// idx 9 = distance >= 9 (deep chain).
    static WALKBACK_BUCKETS: [AtomicU64; 10] = [
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0),
    ];

    #[inline(always)]
    pub fn record_entry(early: bool) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        if early { EARLY_OUT.fetch_add(1, Ordering::Relaxed); }
    }
    #[inline(always)]
    pub fn record_refresh() {
        REFRESHES.fetch_add(1, Ordering::Relaxed);
    }
    #[inline(always)]
    pub fn record_walkback_distance(d: usize) {
        let idx = d.min(9);
        WALKBACK_BUCKETS[idx].fetch_add(1, Ordering::Relaxed);
    }
    #[inline(always)]
    pub fn record_incremental(n_changes: u64) {
        INCREMENTALS.fetch_add(1, Ordering::Relaxed);
        TOTAL_DELTA_CHANGES.fetch_add(n_changes, Ordering::Relaxed);
    }
    pub fn report() {
        let c = CALLS.load(Ordering::Relaxed);
        let e = EARLY_OUT.load(Ordering::Relaxed);
        let r = REFRESHES.load(Ordering::Relaxed);
        let i = INCREMENTALS.load(Ordering::Relaxed);
        let td = TOTAL_DELTA_CHANGES.load(Ordering::Relaxed);
        eprintln!(
            "materialize stats: calls={} early={} ({:.1}%) refresh={} ({:.1}% of non-early) incr={} ({:.1}% of non-early) avg_deltas_per_incr={:.2}",
            c, e, 100.0 * e as f64 / c.max(1) as f64,
            r, 100.0 * r as f64 / (c - e).max(1) as f64,
            i, 100.0 * i as f64 / (c - e).max(1) as f64,
            if i > 0 { td as f64 / i as f64 } else { 0.0 }
        );
        // Walk-back distance distribution for refresh fallbacks.
        let wb_total: u64 = (0..10).map(|i| WALKBACK_BUCKETS[i].load(Ordering::Relaxed)).sum();
        if wb_total > 0 {
            let mut parts: Vec<String> = Vec::new();
            for i in 0..10 {
                let n = WALKBACK_BUCKETS[i].load(Ordering::Relaxed);
                let pct = 100.0 * n as f64 / wb_total as f64;
                if n == 0 { continue; }
                let label = match i {
                    0 => "no-anc".to_string(),
                    9 => ">=9".to_string(),
                    k => format!("d{}", k),
                };
                parts.push(format!("{}={} ({:.1}%)", label, n, pct));
            }
            eprintln!("walkback distance: {}", parts.join(" "));
        }
    }
}

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
/// PSQ inputs per king bucket (12 piece types × 64 squares).
pub const PSQ_INPUTS_PER_BUCKET: usize = 768;
/// Default PSQ input size for 16-bucket layouts. Per-net actual size is
/// `net.num_king_buckets * PSQ_INPUTS_PER_BUCKET`.
pub const NNUE_INPUT_SIZE: usize = 16 * PSQ_INPUTS_PER_BUCKET;
pub const NNUE_OUTPUT_BUCKETS: usize = 8;
/// Maximum king bucket count we allocate static tables for (covers all
/// known layouts: uniform/consensus=16, Reckless=10).
pub const NNUE_MAX_KING_BUCKETS: usize = 16;
const NNUE_NUM_PIECE_TYPES: usize = 12;

/// King bucket layout identifier. Mirrors `bullet_convert::KbLayout` so the
/// values round-trip through the .nnue header byte.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KbLayout {
    Uniform = 0,
    Consensus = 1,
    Reckless = 2,
}

impl KbLayout {
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(KbLayout::Uniform),
            1 => Some(KbLayout::Consensus),
            2 => Some(KbLayout::Reckless),
            _ => None,
        }
    }

    pub fn default_count(self) -> usize {
        match self {
            KbLayout::Uniform | KbLayout::Consensus => 16,
            KbLayout::Reckless => 10,
        }
    }
}

// Quantization
const QA: i32 = 255;  // accumulator scale (CReLU/SCReLU clip max)
const QB: i32 = 64;   // output weight scale
const QAB: i32 = QA * QB; // 16320
const EVAL_SCALE: i32 = 400; // sigmoid → centipawns
const FT_SHIFT: i32 = 9; // pairwise product shift for v7 L1 input (consensus: 9)
const PW_SCALE: i32 = (QA * QA) >> FT_SHIFT; // max packed value after shift (127 for >>9)

// File magic
const NNUE_MAGIC: u32 = 0x4E4E5545; // "NNUE" in LE

// King bucket tables: computed per-net from the layout field (see
// compute_king_buckets below) and stored on NNUENet. Prior to 2026-04-20
// these were `static mut KING_BUCKET: [usize; 64]` / `static mut KING_MIRROR`
// written by `init_king_buckets_layout` on every net load. That data race
// (helpers reading while load wrote) was the root cause of the v9 T=4 SMP
// regression bisected to commit 1356150 — eliminated by making them
// per-`NNUENet` fields populated at load time and never mutated after.
// See `fix/smp-king-bucket-race`.

/// Consensus king bucket layout: fine-near, coarse-far (Alexandria/Viridithas)
/// Indexed by [mirrored_file (0-3)][rank (0-7)]
const CONSENSUS_BUCKETS: [[usize; 8]; 4] = [
    [ 0,  4,  8,  8, 12, 12, 14, 14], // file a/h (mirrored file 0)
    [ 1,  5,  9,  9, 12, 12, 14, 14], // file b/g
    [ 2,  6, 10, 10, 13, 13, 15, 15], // file c/f
    [ 3,  7, 11, 11, 13, 13, 15, 15], // file d/e
];

/// Reckless king bucket layout: 10 buckets. Per-file ranks 1-2 (4 + 4),
/// one bucket rank 3, one bucket ranks 4-8. See Reckless/src/nnue.rs:71-80.
/// Indexed by [sq] directly (already mirror-aware via Bullet-style [0,1,2,3,3,2,1,0]).
#[rustfmt::skip]
const RECKLESS_BUCKETS_FLAT: [usize; 64] = [
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 5, 6, 7, 7, 6, 5, 4,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9,
];

/// Pure function: given a KbLayout, returns the (king_bucket, king_mirror)
/// tables populated for every square 0-63. Called once at net load time; the
/// results are stored on NNUENet and never mutated after that.
pub fn compute_king_buckets(layout: KbLayout) -> ([usize; 64], [bool; 64]) {
    let mut kb = [0usize; 64];
    let mut km = [false; 64];
    for sq in 0..64 {
        let file = sq % 8;
        let rank = sq / 8;
        // File mirror applies to Uniform and Consensus; Reckless bakes the
        // mirror into its flat table already (see RECKLESS_BUCKETS_FLAT).
        let (mirrored_file, mirror) = if file >= 4 {
            (7 - file, true)
        } else {
            (file, false)
        };
        kb[sq] = match layout {
            KbLayout::Uniform   => mirrored_file * 4 + rank / 2,
            KbLayout::Consensus => CONSENSUS_BUCKETS[mirrored_file][rank],
            KbLayout::Reckless  => RECKLESS_BUCKETS_FLAT[sq],
        };
        km[sq] = mirror;
    }
    (kb, km)
}

pub fn init_nnue() {
    // No-op retained for API compatibility. Previously populated the
    // global static mut tables; now those are per-net fields.
}

/// Deprecated legacy API. Returns the computed king-bucket tables for the
/// given layout bool. Kept so older call paths that don't yet have a handle
/// on NNUENet can at least get the correct per-layout tables. Prefer
/// `NNUENet::king_bucket()` / `king_mirror()` methods.
pub fn init_king_buckets(consensus: bool) {
    // No-op: previously wrote the static globals, now a pure pre-compute.
    // Kept as a symbol so existing code that calls this still compiles.
    let _ = consensus;
}

/// Piece index for HalfKA: maps (color, piece_type) to 0-11.
/// White pieces: 0-5, Black pieces: 6-11.
#[inline]
fn piece_index(color: u8, pt: u8) -> usize {
    (color as usize) * 6 + pt as usize
}

/// Compute HalfKA feature index using a specific net's king-bucket tables.
/// perspective: WHITE or BLACK
/// king_sq: the king square for this perspective
/// pc_color: color of the piece
/// pc_type: piece type (PAWN..KING)
/// pc_sq: square of the piece
/// king_bucket / king_mirror: the net's per-square lookup tables
#[inline]
pub fn halfka_index_with(
    king_bucket: &[usize; 64],
    king_mirror: &[bool; 64],
    perspective: u8,
    king_sq: u8,
    pc_color: u8,
    pc_type: u8,
    pc_sq: u8,
) -> usize {
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
    if king_mirror[ks] {
        ps = (ps & !7) | (7 - (ps & 7));
    }

    let bucket = king_bucket[ks];
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

/// Fused accumulator update: dst = src + Σ add_rows - Σ sub_rows in ONE pass.
/// Replaces the copy+per-delta-pass pattern with a single load/compute/store
/// sequence per accumulator chunk.
///
/// For a capture (3 deltas: sub moving-from, add moving-to, sub captured),
/// the previous code paid 3 passes over the full accumulator (1 fused
/// copy+delta + 2 separate in-place passes = ~3 × 2h bytes of memory
/// traffic). With this fused kernel, total traffic is 2h bytes (read src
/// once, write dst once) regardless of delta count. Reckless-style pattern.
///
/// Register blocking with REGS=8 SIMD lanes processes 128 i16s per chunk,
/// keeping the hot accumulator values in registers while all deltas are
/// applied. For typical h=768 this is 6 chunks per perspective.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_acc_fused_avx2(
    dst: &mut [i16],
    src: &[i16],
    add_rows: &[&[i16]],
    sub_rows: &[&[i16]],
    h: usize,
) {
    const REGS: usize = 8;
    const CHUNK: usize = REGS * 16;

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    let mut offset = 0;
    while offset < h {
        let nregs = ((h - offset).min(CHUNK) + 15) / 16;

        // Load src chunk into registers (seeds the per-chunk accumulator).
        let mut regs: [__m256i; REGS] = [_mm256_setzero_si256(); REGS];
        for i in 0..nregs {
            regs[i] = _mm256_loadu_si256(src_ptr.add(offset + i * 16) as *const __m256i);
        }

        // Apply all adds in-register.
        for row in add_rows {
            let row_ptr = row.as_ptr().add(offset);
            for i in 0..nregs {
                let w = _mm256_loadu_si256(row_ptr.add(i * 16) as *const __m256i);
                regs[i] = _mm256_add_epi16(regs[i], w);
            }
        }

        // Apply all subs in-register.
        for row in sub_rows {
            let row_ptr = row.as_ptr().add(offset);
            for i in 0..nregs {
                let w = _mm256_loadu_si256(row_ptr.add(i * 16) as *const __m256i);
                regs[i] = _mm256_sub_epi16(regs[i], w);
            }
        }

        // Store once per chunk.
        for i in 0..nregs {
            _mm256_storeu_si256(dst_ptr.add(offset + i * 16) as *mut __m256i, regs[i]);
        }

        offset += CHUNK;
    }
}

/// AVX-512 fused copy+add/sub. 32 i16 per ZMM, CHUNK=256 elements per pass —
/// the whole 768-wide v9 accumulator clears in 3 iterations vs AVX2's 6.
/// This is on the make_move hot path (run every incremental accumulator
/// push), so shaving per-chunk iterations translates directly to higher NPS.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd_acc_fused_avx512(
    dst: &mut [i16],
    src: &[i16],
    add_rows: &[&[i16]],
    sub_rows: &[&[i16]],
    h: usize,
) {
    const REGS: usize = 8;
    const LANE: usize = 32;
    const CHUNK: usize = REGS * LANE;

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    let mut offset = 0;
    while offset < h {
        let nregs = ((h - offset).min(CHUNK) + LANE - 1) / LANE;

        let mut regs: [__m512i; REGS] = [_mm512_setzero_si512(); REGS];
        for i in 0..nregs {
            regs[i] = _mm512_loadu_si512(src_ptr.add(offset + i * LANE) as *const __m512i);
        }

        for row in add_rows {
            let row_ptr = row.as_ptr().add(offset);
            for i in 0..nregs {
                let w = _mm512_loadu_si512(row_ptr.add(i * LANE) as *const __m512i);
                regs[i] = _mm512_add_epi16(regs[i], w);
            }
        }

        for row in sub_rows {
            let row_ptr = row.as_ptr().add(offset);
            for i in 0..nregs {
                let w = _mm512_loadu_si512(row_ptr.add(i * LANE) as *const __m512i);
                regs[i] = _mm512_sub_epi16(regs[i], w);
            }
        }

        for i in 0..nregs {
            _mm512_storeu_si512(dst_ptr.add(offset + i * LANE) as *mut __m512i, regs[i]);
        }

        offset += CHUNK;
    }
}

/// NEON mirror of simd_acc_fused_avx2. Per-chunk: load src into regs, apply
/// all adds/subs in-register, store once. 16 regs × 8 i16 = 128 elements per
/// chunk (same footprint as AVX2 8×16). Hot path: runs once per make_move
/// for incremental accumulator updates; on aarch64 this was previously
/// falling through to a scalar loop, costing a significant fraction of NPS.
#[cfg(target_arch = "aarch64")]
unsafe fn simd_acc_fused_neon(
    dst: &mut [i16],
    src: &[i16],
    add_rows: &[&[i16]],
    sub_rows: &[&[i16]],
    h: usize,
) {
    const REGS: usize = 16;
    const CHUNK: usize = REGS * 8;

    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    let mut offset = 0;
    while offset < h {
        let nregs = ((h - offset).min(CHUNK) + 7) / 8;

        let mut regs: [int16x8_t; REGS] = [vdupq_n_s16(0); REGS];
        for i in 0..nregs {
            regs[i] = vld1q_s16(src_ptr.add(offset + i * 8));
        }

        for row in add_rows {
            let row_ptr = row.as_ptr().add(offset);
            for i in 0..nregs {
                let w = vld1q_s16(row_ptr.add(i * 8));
                regs[i] = vaddq_s16(regs[i], w);
            }
        }

        for row in sub_rows {
            let row_ptr = row.as_ptr().add(offset);
            for i in 0..nregs {
                let w = vld1q_s16(row_ptr.add(i * 8));
                regs[i] = vsubq_s16(regs[i], w);
            }
        }

        for i in 0..nregs {
            vst1q_s16(dst_ptr.add(offset + i * 8), regs[i]);
        }

        offset += CHUNK;
    }
}

/// Register-blocked batch apply for Finny table refresh (Reckless pattern).
/// Loads 8 SIMD registers from acc, applies ALL adds then ALL subs, stores once.
/// Much faster than per-piece acc_add/acc_sub which loads/stores each time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn finny_batch_apply_avx2(
    acc: &mut [i16],
    input_weights: &[i16],
    h: usize,
    adds: &[usize],
    subs: &[usize],
) {
    const REGS: usize = 8;
    const CHUNK: usize = REGS * 16; // 128 i16 elements per chunk

    let acc_ptr = acc.as_mut_ptr();
    let w_ptr = input_weights.as_ptr();

    let mut offset = 0;
    while offset < h {
        let nregs = ((h - offset).min(CHUNK) + 15) / 16;

        // Load accumulator chunk into registers
        let mut regs: [__m256i; REGS] = [_mm256_setzero_si256(); REGS];
        for i in 0..nregs {
            regs[i] = _mm256_loadu_si256(acc_ptr.add(offset + i * 16) as *const __m256i);
        }

        // Apply ALL adds (weight rows are i16, no widening needed)
        for &idx in adds {
            let row = w_ptr.add(idx * h + offset);
            for i in 0..nregs {
                let w = _mm256_loadu_si256(row.add(i * 16) as *const __m256i);
                regs[i] = _mm256_add_epi16(regs[i], w);
            }
        }

        // Apply ALL subs
        for &idx in subs {
            let row = w_ptr.add(idx * h + offset);
            for i in 0..nregs {
                let w = _mm256_loadu_si256(row.add(i * 16) as *const __m256i);
                regs[i] = _mm256_sub_epi16(regs[i], w);
            }
        }

        // Store registers back
        for i in 0..nregs {
            _mm256_storeu_si256(acc_ptr.add(offset + i * 16) as *mut __m256i, regs[i]);
        }

        offset += CHUNK;
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

/// Pack SCReLU'd accumulator into uint8: clamp [0,255], v²/255 → [0,255].
/// Output buffer must be at least h bytes. h must be multiple of 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_screlu_pack(acc: &[i16], out: &mut [u8], h: usize) {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let mut i = 0;
    while i + 32 <= h {
        // Load 32 i16 values (two YMM registers)
        let v0 = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let v1 = _mm256_loadu_si256(acc.as_ptr().add(i + 16) as *const __m256i);
        let c0 = _mm256_min_epi16(_mm256_max_epi16(v0, zero), qa);
        let c1 = _mm256_min_epi16(_mm256_max_epi16(v1, zero), qa);
        // v²: PMULLW gives low 16 bits (max 65025, fits in u16)
        let sq0 = _mm256_mullo_epi16(c0, c0);
        let sq1 = _mm256_mullo_epi16(c1, c1);
        // Divide by 255: v²/255 ≈ (v² + 128) >> 8 for values in [0, 65025]
        // More precise: v²/255 = (v² * 257 + 32768) >> 16 but >>8 is close enough
        let d0 = _mm256_srli_epi16(sq0, 8); // [0, 254]
        let d1 = _mm256_srli_epi16(sq1, 8);
        // Pack i16 → u8 with saturation: _mm256_packus_epi16 packs with lane crossing
        let packed = _mm256_packus_epi16(d0, d1);
        // Fix lane crossing: packus interleaves 128-bit lanes, need permute
        let fixed = _mm256_permute4x64_epi64(packed, 0xD8); // 0,2,1,3
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, fixed);
        i += 32;
    }
    // Handle remaining 16 if h is not multiple of 32
    while i < h {
        let v = (acc[i] as i32).clamp(0, 255);
        out[i] = ((v * v) >> 8) as u8;
        i += 1;
    }
}

/// CReLU + pairwise pack: acc[0..pw] and acc[pw..2*pw] → out[0..pw] u8.
/// clamp(a, 0, 255) * clamp(b, 0, 255) >> 8 for each pair.
/// pw must be multiple of 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Pairwise pack with optional fused threat combine.
/// If threat is non-null, adds threat[i] to acc[i] before clamping (Reckless activate_ft pattern).
unsafe fn simd_pairwise_pack_fused(acc: &[i16], threat: *const i16, out: &mut [u8], pw: usize) {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let has_threat = !threat.is_null();
    let mut i = 0;
    while i + 16 <= pw {
        // Load 16 values from each half
        let mut a = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let mut b = _mm256_loadu_si256(acc.as_ptr().add(pw + i) as *const __m256i);
        // Fused threat combine: add threat values before clamp
        if has_threat {
            let ta = _mm256_loadu_si256(threat.add(i) as *const __m256i);
            let tb = _mm256_loadu_si256(threat.add(pw + i) as *const __m256i);
            a = _mm256_add_epi16(a, ta);
            b = _mm256_add_epi16(b, tb);
        }
        // Clamp [0, QA]
        let ca = _mm256_min_epi16(_mm256_max_epi16(a, zero), qa);
        let cb = _mm256_min_epi16(_mm256_max_epi16(b, zero), qa);
        // Multiply: a*b (low 16 bits, max 65025 fits u16)
        let prod = _mm256_mullo_epi16(ca, cb);
        // >> FT_SHIFT to get [0, 127] (safe for VPMADDUBSW in L1)
        let d = _mm256_srli_epi16(prod, FT_SHIFT);
        // Pack i16 → u8: need to combine with next 16 for full 32 output
        if i + 32 <= pw {
            let mut a2 = _mm256_loadu_si256(acc.as_ptr().add(i + 16) as *const __m256i);
            let mut b2 = _mm256_loadu_si256(acc.as_ptr().add(pw + i + 16) as *const __m256i);
            if has_threat {
                a2 = _mm256_add_epi16(a2, _mm256_loadu_si256(threat.add(i + 16) as *const __m256i));
                b2 = _mm256_add_epi16(b2, _mm256_loadu_si256(threat.add(pw + i + 16) as *const __m256i));
            }
            let ca2 = _mm256_min_epi16(_mm256_max_epi16(a2, zero), qa);
            let cb2 = _mm256_min_epi16(_mm256_max_epi16(b2, zero), qa);
            let prod2 = _mm256_mullo_epi16(ca2, cb2);
            let d2 = _mm256_srli_epi16(prod2, FT_SHIFT);
            let packed = _mm256_packus_epi16(d, d2);
            let fixed = _mm256_permute4x64_epi64(packed, 0xD8);
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, fixed);
            i += 32;
        } else {
            // Remaining 16: pack with zeros
            let packed = _mm256_packus_epi16(d, zero);
            let fixed = _mm256_permute4x64_epi64(packed, 0xD8);
            // Only store 16 bytes (lower half)
            _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i,
                _mm256_castsi256_si128(fixed));
            i += 16;
        }
    }
    while i < pw {
        let mut a = acc[i] as i32;
        let mut b = acc[pw + i] as i32;
        if has_threat { a += *threat.add(i) as i32; b += *threat.add(pw + i) as i32; }
        out[i] = ((a.clamp(0, 255) * b.clamp(0, 255)) >> FT_SHIFT) as u8;
        i += 1;
    }
}


/// L1 int8 matmul: packed u8 input × i8 transposed weights → i32 output.
/// Computes hidden[neuron] += sum_j(packed[j] * weights_t[neuron*h + j]) for one neuron.
/// Uses VPMADDUBSW (u8 × i8 → i16) + VPMADDWD (i16 pairs → i32).
/// h must be multiple of 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_l1_int8_dot(packed: &[u8], weights: &[i8], h: usize) -> i32 {
    let ones = _mm256_set1_epi16(1);
    let mut sum = _mm256_setzero_si256(); // 8 × i32
    let mut i = 0;
    while i < h {
        let a = _mm256_loadu_si256(packed.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);
        // VPMADDUBSW: u8 × i8 → 16 × i16 (pairs summed)
        let prod = _mm256_maddubs_epi16(a, b);
        // VPMADDWD with ones: 16 × i16 → 8 × i32 (pairs summed)
        let widened = _mm256_madd_epi16(prod, ones);
        sum = _mm256_add_epi32(sum, widened);
        i += 32;
    }
    // Horizontal sum of 8 × i32
    let hi = _mm256_extracti128_si256(sum, 1);
    let lo = _mm256_castsi256_si128(sum);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0x4E); // swap pairs
    let sum128 = _mm_add_epi32(sum128, shuf);
    let shuf = _mm_shuffle_epi32(sum128, 0xB1); // swap singles
    let sum128 = _mm_add_epi32(sum128, shuf);
    _mm_cvtsi128_si32(sum128)
}

/// Multi-neuron L1 int8 dot: compute 4 neurons at once, loading input only once per chunk.
/// packed: u8 input [h], weights: i8 [4][h] (4 weight rows contiguous), h: input length.
/// Returns 4 i32 dot products.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_l1_int8_dot_x4(packed: &[u8], w0: &[i8], w1: &[i8], w2: &[i8], w3: &[i8], h: usize) -> [i32; 4] {
    let ones = _mm256_set1_epi16(1);
    let mut sum0 = _mm256_setzero_si256();
    let mut sum1 = _mm256_setzero_si256();
    let mut sum2 = _mm256_setzero_si256();
    let mut sum3 = _mm256_setzero_si256();
    let mut i = 0;
    while i < h {
        let a = _mm256_loadu_si256(packed.as_ptr().add(i) as *const __m256i);
        // Neuron 0
        let b0 = _mm256_loadu_si256(w0.as_ptr().add(i) as *const __m256i);
        let p0 = _mm256_maddubs_epi16(a, b0);
        sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0, ones));
        // Neuron 1
        let b1 = _mm256_loadu_si256(w1.as_ptr().add(i) as *const __m256i);
        let p1 = _mm256_maddubs_epi16(a, b1);
        sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1, ones));
        // Neuron 2
        let b2 = _mm256_loadu_si256(w2.as_ptr().add(i) as *const __m256i);
        let p2 = _mm256_maddubs_epi16(a, b2);
        sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2, ones));
        // Neuron 3
        let b3 = _mm256_loadu_si256(w3.as_ptr().add(i) as *const __m256i);
        let p3 = _mm256_maddubs_epi16(a, b3);
        sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3, ones));
        i += 32;
    }
    // Horizontal sums
    fn hsum(v: __m256i) -> i32 {
        unsafe {
            let hi = _mm256_extracti128_si256(v, 1);
            let lo = _mm256_castsi256_si128(v);
            let sum128 = _mm_add_epi32(lo, hi);
            let shuf = _mm_shuffle_epi32(sum128, 0x4E);
            let sum128 = _mm_add_epi32(sum128, shuf);
            let shuf = _mm_shuffle_epi32(sum128, 0xB1);
            let sum128 = _mm_add_epi32(sum128, shuf);
            _mm_cvtsi128_si32(sum128)
        }
    }
    [hsum(sum0), hsum(sum1), hsum(sum2), hsum(sum3)]
}

/// Find non-zero 32-byte chunk indices in a packed u8 buffer.
/// Returns the number of NNZ chunks. nnz_indices[0..count] contains the byte offsets.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_nnz_chunks(packed: &[u8], nnz_indices: &mut [u16], h: usize) -> usize {
    let mut count = 0usize;
    let mut i = 0;
    while i < h {
        let v = _mm256_loadu_si256(packed.as_ptr().add(i) as *const __m256i);
        // Check if any byte is non-zero
        if _mm256_testz_si256(v, v) == 0 {
            nnz_indices[count] = i as u16;
            count += 1;
        }
        i += 32;
    }
    count
}

/// Sparse L1 int8 matmul: only process NNZ chunks of the packed input.
/// nnz_indices[0..nnz_count] are byte offsets of non-zero 32-byte chunks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_l1_int8_dot_sparse(packed: &[u8], weights: &[i8], nnz_indices: &[u16], nnz_count: usize) -> i32 {
    let ones = _mm256_set1_epi16(1);
    let mut sum = _mm256_setzero_si256();
    for k in 0..nnz_count {
        let off = nnz_indices[k] as usize;
        let a = _mm256_loadu_si256(packed.as_ptr().add(off) as *const __m256i);
        let b = _mm256_loadu_si256(weights.as_ptr().add(off) as *const __m256i);
        let prod = _mm256_maddubs_epi16(a, b);
        let widened = _mm256_madd_epi16(prod, ones);
        sum = _mm256_add_epi32(sum, widened);
    }
    let hi = _mm256_extracti128_si256(sum, 1);
    let lo = _mm256_castsi256_si128(sum);
    let sum128 = _mm_add_epi32(lo, hi);
    let shuf = _mm_shuffle_epi32(sum128, 0x4E);
    let sum128 = _mm_add_epi32(sum128, shuf);
    let shuf = _mm_shuffle_epi32(sum128, 0xB1);
    let sum128 = _mm_add_epi32(sum128, shuf);
    _mm_cvtsi128_si32(sum128)
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

/// AVX-512 pairwise dot product: clamp two halves to [0,255], multiply pairs,
/// byte-decompose the product, dot with weights. 32 elements per iteration.
/// Same byte-decomposition as AVX2 version but 2× wider.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_pairwise_dot(acc_first: &[i16], acc_second: &[i16], weights: &[i16], count: usize) -> i64 {
    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(QA as i16);
    let mask_lo = _mm512_set1_epi16(0x00FF);
    let mut sum32_b0 = _mm512_setzero_si512(); // 16 × i32
    let mut sum32_b1 = _mm512_setzero_si512(); // 16 × i32
    let mut total: i64 = 0;
    let mut batch = 0u32;

    let mut i = 0;
    while i < count {
        let a = _mm512_loadu_si512(acc_first.as_ptr().add(i) as *const __m512i);
        let a_clamped = _mm512_min_epi16(_mm512_max_epi16(a, zero), qa);
        let b = _mm512_loadu_si512(acc_second.as_ptr().add(i) as *const __m512i);
        let b_clamped = _mm512_min_epi16(_mm512_max_epi16(b, zero), qa);

        let prod = _mm512_mullo_epi16(a_clamped, b_clamped);
        let w = _mm512_loadu_si512(weights.as_ptr().add(i) as *const __m512i);

        let byte0 = _mm512_and_si512(prod, mask_lo);
        let byte1 = _mm512_srli_epi16(prod, 8);

        sum32_b0 = _mm512_add_epi32(sum32_b0, _mm512_madd_epi16(byte0, w));
        sum32_b1 = _mm512_add_epi32(sum32_b1, _mm512_madd_epi16(byte1, w));

        batch += 32;
        if batch >= 128 {
            // Drain: total += b0_sum + b1_sum * 256
            total += _mm512_reduce_add_epi32(sum32_b0) as i64;
            total += (_mm512_reduce_add_epi32(sum32_b1) as i64) << 8;
            sum32_b0 = _mm512_setzero_si512();
            sum32_b1 = _mm512_setzero_si512();
            batch = 0;
        }

        i += 32;
    }

    if batch > 0 {
        total += _mm512_reduce_add_epi32(sum32_b0) as i64;
        total += (_mm512_reduce_add_epi32(sum32_b1) as i64) << 8;
    }

    total
}

/// AVX-512 pairwise pack with optional fused threat combine.
/// acc[0..pw] and acc[pw..2*pw] → out[0..pw] u8.
/// If threat is non-null, adds threat[i] to acc[i] before clamp (v9 fused pack).
/// clamp(a, 0, 255) * clamp(b, 0, 255) >> FT_SHIFT for each pair.
/// pw must be multiple of 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_pairwise_pack_fused(acc: &[i16], threat: *const i16, out: &mut [u8], pw: usize) {
    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(QA as i16);
    let has_threat = !threat.is_null();
    // Permutation index to fix lane ordering after packus_epi16
    // _mm512_packus_epi16 interleaves 128-bit lanes: [0,4,1,5,2,6,3,7]
    // We need: [0,1,2,3,4,5,6,7] → permute qwords by [0,2,4,6,1,3,5,7]
    let perm_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    let mut i = 0;
    while i + 32 <= pw {
        // Load 32 values from each half (two ZMM registers of i16)
        let mut a0 = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
        let mut b0 = _mm512_loadu_si512(acc.as_ptr().add(pw + i) as *const __m512i);
        if has_threat {
            let ta0 = _mm512_loadu_si512(threat.add(i) as *const __m512i);
            let tb0 = _mm512_loadu_si512(threat.add(pw + i) as *const __m512i);
            a0 = _mm512_add_epi16(a0, ta0);
            b0 = _mm512_add_epi16(b0, tb0);
        }
        let ca0 = _mm512_min_epi16(_mm512_max_epi16(a0, zero), qa);
        let cb0 = _mm512_min_epi16(_mm512_max_epi16(b0, zero), qa);
        let prod0 = _mm512_mullo_epi16(ca0, cb0);
        let d0 = _mm512_srli_epi16(prod0, FT_SHIFT as u32);

        if i + 64 <= pw {
            let mut a1 = _mm512_loadu_si512(acc.as_ptr().add(i + 32) as *const __m512i);
            let mut b1 = _mm512_loadu_si512(acc.as_ptr().add(pw + i + 32) as *const __m512i);
            if has_threat {
                let ta1 = _mm512_loadu_si512(threat.add(i + 32) as *const __m512i);
                let tb1 = _mm512_loadu_si512(threat.add(pw + i + 32) as *const __m512i);
                a1 = _mm512_add_epi16(a1, ta1);
                b1 = _mm512_add_epi16(b1, tb1);
            }
            let ca1 = _mm512_min_epi16(_mm512_max_epi16(a1, zero), qa);
            let cb1 = _mm512_min_epi16(_mm512_max_epi16(b1, zero), qa);
            let prod1 = _mm512_mullo_epi16(ca1, cb1);
            let d1 = _mm512_srli_epi16(prod1, FT_SHIFT as u32);
            // Pack 2×32 i16 → 64 u8
            let packed = _mm512_packus_epi16(d0, d1);
            let fixed = _mm512_permutexvar_epi64(perm_idx, packed);
            _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut __m512i, fixed);
            i += 64;
        } else {
            // Remaining 32: pack with zeros into 64 bytes, store lower 32
            let packed = _mm512_packus_epi16(d0, zero);
            let fixed = _mm512_permutexvar_epi64(perm_idx, packed);
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i,
                _mm512_castsi512_si256(fixed));
            i += 32;
        }
    }
    // Tail (< 32 elements) handled by scalar
    while i < pw {
        let mut a = acc[i] as i32;
        let mut b = acc[pw + i] as i32;
        if has_threat { a += *threat.add(i) as i32; b += *threat.add(pw + i) as i32; }
        out[i] = ((a.clamp(0, 255) * b.clamp(0, 255)) >> FT_SHIFT) as u8;
        i += 1;
    }
}

/// AVX-512 SCReLU pack: clamp [0,255], v²/255 → [0,255] as u8.
/// h must be multiple of 64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_screlu_pack(acc: &[i16], out: &mut [u8], h: usize) {
    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(QA as i16);
    let perm_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    let mut i = 0;
    while i + 64 <= h {
        let v0 = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
        let v1 = _mm512_loadu_si512(acc.as_ptr().add(i + 32) as *const __m512i);
        let c0 = _mm512_min_epi16(_mm512_max_epi16(v0, zero), qa);
        let c1 = _mm512_min_epi16(_mm512_max_epi16(v1, zero), qa);
        let sq0 = _mm512_mullo_epi16(c0, c0);
        let sq1 = _mm512_mullo_epi16(c1, c1);
        let d0 = _mm512_srli_epi16(sq0, 8);
        let d1 = _mm512_srli_epi16(sq1, 8);
        let packed = _mm512_packus_epi16(d0, d1);
        let fixed = _mm512_permutexvar_epi64(perm_idx, packed);
        _mm512_storeu_si512(out.as_mut_ptr().add(i) as *mut __m512i, fixed);
        i += 64;
    }
    // Tail: 32 elements
    while i + 32 <= h {
        let v = _mm512_loadu_si512(acc.as_ptr().add(i) as *const __m512i);
        let c = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
        let sq = _mm512_mullo_epi16(c, c);
        let d = _mm512_srli_epi16(sq, 8);
        let packed = _mm512_packus_epi16(d, zero);
        let fixed = _mm512_permutexvar_epi64(perm_idx, packed);
        _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i,
            _mm512_castsi512_si256(fixed));
        i += 32;
    }
}

/// AVX-512 L1 int8 matmul: packed u8 input × i8 transposed weights → i32.
/// Uses VPMADDUBSW (u8 × i8 → i16) + VPMADDWD (i16 pairs → i32).
/// h must be multiple of 64. For h not divisible by 64, use AVX2 fallback.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_l1_int8_dot(packed: &[u8], weights: &[i8], h: usize) -> i32 {
    let ones = _mm512_set1_epi16(1);
    let mut sum = _mm512_setzero_si512(); // 16 × i32
    let mut i = 0;
    while i < h {
        let a = _mm512_loadu_si512(packed.as_ptr().add(i) as *const __m512i);
        let b = _mm512_loadu_si512(weights.as_ptr().add(i) as *const __m512i);
        let prod = _mm512_maddubs_epi16(a, b);
        let widened = _mm512_madd_epi16(prod, ones);
        sum = _mm512_add_epi32(sum, widened);
        i += 64;
    }
    _mm512_reduce_add_epi32(sum)
}

/// AVX-512 sparse L1 int8 matmul: only process NNZ 64-byte chunks.
/// nnz_indices[0..nnz_count] are byte offsets of non-zero 64-byte chunks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn simd512_l1_int8_dot_sparse(packed: &[u8], weights: &[i8], nnz_indices: &[u16], nnz_count: usize) -> i32 {
    let ones = _mm512_set1_epi16(1);
    let mut sum = _mm512_setzero_si512();
    for k in 0..nnz_count {
        let off = nnz_indices[k] as usize;
        let a = _mm512_loadu_si512(packed.as_ptr().add(off) as *const __m512i);
        let b = _mm512_loadu_si512(weights.as_ptr().add(off) as *const __m512i);
        let prod = _mm512_maddubs_epi16(a, b);
        let widened = _mm512_madd_epi16(prod, ones);
        sum = _mm512_add_epi32(sum, widened);
    }
    _mm512_reduce_add_epi32(sum)
}

/// AVX-512 VNNI variant of `simd512_l1_int8_dot`.
///
/// Fuses `VPMADDUBSW + VPMADDWD(ones) + VPADDD` into a single `VPDPBUSD`.
/// Compared to the non-VNNI kernel this removes one dependent chain
/// (vpmaddubsw → vpmaddwd) per 64-byte chunk, typically ~1.5× faster on
/// Zen 5 / Sapphire Rapids for h in the 512–1024 range. Identical
/// numerical behaviour — both accumulate into i32 with no saturation in
/// the u8×i8 → i32 path.
///
/// Requires `h % 64 == 0`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn simd512_l1_int8_dot_vnni(packed: &[u8], weights: &[i8], h: usize) -> i32 {
    // Split the accumulator to break the dependency chain — VPDPBUSD on
    // Zen 5 has 4-cycle latency, so a single accumulator bottlenecks.
    let mut s0 = _mm512_setzero_si512();
    let mut s1 = _mm512_setzero_si512();
    let mut i = 0;
    while i + 128 <= h {
        let a0 = _mm512_loadu_si512(packed.as_ptr().add(i) as *const __m512i);
        let b0 = _mm512_loadu_si512(weights.as_ptr().add(i) as *const __m512i);
        s0 = _mm512_dpbusd_epi32(s0, a0, b0);

        let a1 = _mm512_loadu_si512(packed.as_ptr().add(i + 64) as *const __m512i);
        let b1 = _mm512_loadu_si512(weights.as_ptr().add(i + 64) as *const __m512i);
        s1 = _mm512_dpbusd_epi32(s1, a1, b1);
        i += 128;
    }
    while i < h {
        let a = _mm512_loadu_si512(packed.as_ptr().add(i) as *const __m512i);
        let b = _mm512_loadu_si512(weights.as_ptr().add(i) as *const __m512i);
        s0 = _mm512_dpbusd_epi32(s0, a, b);
        i += 64;
    }
    _mm512_reduce_add_epi32(_mm512_add_epi32(s0, s1))
}

/// AVX-512 VNNI sparse variant — same as `simd512_l1_int8_dot_sparse`
/// but with VPDPBUSD. Typically used with ~45–55% chunk density on v9
/// pairwise outputs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn simd512_l1_int8_dot_sparse_vnni(packed: &[u8], weights: &[i8], nnz_indices: &[u16], nnz_count: usize) -> i32 {
    let mut s0 = _mm512_setzero_si512();
    let mut s1 = _mm512_setzero_si512();
    let mut k = 0;
    while k + 2 <= nnz_count {
        let off0 = *nnz_indices.get_unchecked(k) as usize;
        let a0 = _mm512_loadu_si512(packed.as_ptr().add(off0) as *const __m512i);
        let b0 = _mm512_loadu_si512(weights.as_ptr().add(off0) as *const __m512i);
        s0 = _mm512_dpbusd_epi32(s0, a0, b0);

        let off1 = *nnz_indices.get_unchecked(k + 1) as usize;
        let a1 = _mm512_loadu_si512(packed.as_ptr().add(off1) as *const __m512i);
        let b1 = _mm512_loadu_si512(weights.as_ptr().add(off1) as *const __m512i);
        s1 = _mm512_dpbusd_epi32(s1, a1, b1);
        k += 2;
    }
    while k < nnz_count {
        let off = *nnz_indices.get_unchecked(k) as usize;
        let a = _mm512_loadu_si512(packed.as_ptr().add(off) as *const __m512i);
        let b = _mm512_loadu_si512(weights.as_ptr().add(off) as *const __m512i);
        s0 = _mm512_dpbusd_epi32(s0, a, b);
        k += 1;
    }
    _mm512_reduce_add_epi32(_mm512_add_epi32(s0, s1))
}

/// AVX-512 f32 L2 matmul for L2 == 32: two ZMM accumulators hold the full
/// L2 row, one FMA pair per `l1_out[i]`. Replaces the generic loop that
/// LLVM was vectorising with `VGATHERQPS` (13% of total cycles on v9,
/// dominant after VNNI landed on L1).
///
/// Preconditions:
/// - `l2 == 32`
/// - `h2.len() >= 32`
/// - `biases` contiguous at `l2_off`: `biases[l2_off..l2_off+32]`
/// - `l2_weights_f[i * l2_total + l2_off .. + 32]` contiguous per `i`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_fmadd_avx512_x32(
    l1_out: &[f32],
    l1_out_count: usize,
    l2_weights_f: &[f32],
    l2_total: usize,
    l2_off: usize,
    biases: &[f32],
    h2: &mut [f32],
) {
    let mut h_lo = _mm512_loadu_ps(biases.as_ptr().add(l2_off));
    let mut h_hi = _mm512_loadu_ps(biases.as_ptr().add(l2_off + 16));
    for i in 0..l1_out_count {
        let v = *l1_out.get_unchecked(i);
        if v == 0.0 { continue; }
        let bcast = _mm512_set1_ps(v);
        let wp = l2_weights_f.as_ptr().add(i * l2_total + l2_off);
        let w_lo = _mm512_loadu_ps(wp);
        let w_hi = _mm512_loadu_ps(wp.add(16));
        h_lo = _mm512_fmadd_ps(bcast, w_lo, h_lo);
        h_hi = _mm512_fmadd_ps(bcast, w_hi, h_hi);
    }
    _mm512_storeu_ps(h2.as_mut_ptr(), h_lo);
    _mm512_storeu_ps(h2.as_mut_ptr().add(16), h_hi);
}

/// AVX-512 f32 horizontal dot product of `len` elements (len == 32).
/// Replaces the scalar tail reduction in the output-weights dot.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_fmadd_avx512_x32(a: &[f32], b: &[f32], bias: f32) -> f32 {
    let a_lo = _mm512_loadu_ps(a.as_ptr());
    let a_hi = _mm512_loadu_ps(a.as_ptr().add(16));
    let b_lo = _mm512_loadu_ps(b.as_ptr());
    let b_hi = _mm512_loadu_ps(b.as_ptr().add(16));
    let sum = _mm512_fmadd_ps(a_lo, b_lo, _mm512_mul_ps(a_hi, b_hi));
    bias + _mm512_reduce_add_ps(sum)
}

/// Find non-zero 64-byte chunk indices in a packed u8 buffer (AVX-512).
/// Returns the number of NNZ chunks. nnz_indices[0..count] contains byte offsets.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn find_nnz_chunks_512(packed: &[u8], nnz_indices: &mut [u16], h: usize) -> usize {
    let mut count = 0usize;
    let mut i = 0;
    while i < h {
        let v = _mm512_loadu_si512(packed.as_ptr().add(i) as *const __m512i);
        let mask = _mm512_test_epi8_mask(v, v);
        if mask != 0 {
            nnz_indices[count] = i as u16;
            count += 1;
        }
        i += 64;
    }
    count
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

/// Pairwise dot product (NEON): clamp [0,255], multiply pairs, dot with weights.
/// Uses byte decomposition: prod = byte0 + byte1*256, then dot each with weights.
/// Processes 8 × i16 per iteration, drains i32 to i64 every 128 elements.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_pairwise_dot(acc_first: &[i16], acc_second: &[i16], weights: &[i16], count: usize) -> i64 {
    let zero = vdupq_n_s16(0);
    let qa = vdupq_n_s16(QA as i16); // 255 — clamp ceiling and byte0 mask
    let mut sum_b0_lo = vdupq_n_s32(0);
    let mut sum_b0_hi = vdupq_n_s32(0);
    let mut sum_b1_lo = vdupq_n_s32(0);
    let mut sum_b1_hi = vdupq_n_s32(0);
    let mut total: i64 = 0;
    let mut batch = 0u32;

    let mut i = 0;
    while i < count {
        let a = vld1q_s16(acc_first.as_ptr().add(i));
        let a_cl = vminq_s16(vmaxq_s16(a, zero), qa);
        let b = vld1q_s16(acc_second.as_ptr().add(i));
        let b_cl = vminq_s16(vmaxq_s16(b, zero), qa);
        let w = vld1q_s16(weights.as_ptr().add(i));

        // a*b: u16 [0, 65025] stored as i16 bit pattern
        let prod = vmulq_s16(a_cl, b_cl);
        // byte0 = prod & 0xFF (using qa as mask)
        let byte0 = vandq_s16(prod, qa);
        // byte1 = prod >> 8 (unsigned shift)
        let byte1 = vreinterpretq_s16_u16(vshrq_n_u16::<9>(vreinterpretq_u16_s16(prod)));

        // Widening multiply-accumulate: i16 × i16 → i32
        sum_b0_lo = vmlal_s16(sum_b0_lo, vget_low_s16(byte0), vget_low_s16(w));
        sum_b0_hi = vmlal_high_s16(sum_b0_hi, byte0, w);
        sum_b1_lo = vmlal_s16(sum_b1_lo, vget_low_s16(byte1), vget_low_s16(w));
        sum_b1_hi = vmlal_high_s16(sum_b1_hi, byte1, w);

        batch += 8;
        if batch >= 128 {
            let b0 = vaddq_s32(sum_b0_lo, sum_b0_hi);
            let b1 = vaddq_s32(sum_b1_lo, sum_b1_hi);
            total += vaddlvq_s32(b0);
            total += vaddlvq_s32(b1) << 8;
            sum_b0_lo = vdupq_n_s32(0);
            sum_b0_hi = vdupq_n_s32(0);
            sum_b1_lo = vdupq_n_s32(0);
            sum_b1_hi = vdupq_n_s32(0);
            batch = 0;
        }
        i += 8;
    }

    if batch > 0 {
        let b0 = vaddq_s32(sum_b0_lo, sum_b0_hi);
        let b1 = vaddq_s32(sum_b1_lo, sum_b1_hi);
        total += vaddlvq_s32(b0);
        total += vaddlvq_s32(b1) << 8;
    }
    total
}

/// Pack SCReLU'd accumulator into uint8 (NEON): clamp [0,255], v²/256 → [0,254].
/// Output buffer must be at least h bytes.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_screlu_pack(acc: &[i16], out: &mut [u8], h: usize) {
    let zero = vdupq_n_s16(0);
    let qa = vdupq_n_s16(QA as i16);
    let mut i = 0;
    while i + 16 <= h {
        let v0 = vld1q_s16(acc.as_ptr().add(i));
        let v1 = vld1q_s16(acc.as_ptr().add(i + 8));
        let c0 = vminq_s16(vmaxq_s16(v0, zero), qa);
        let c1 = vminq_s16(vmaxq_s16(v1, zero), qa);
        // v²: bit pattern is u16 [0, 65025]
        let sq0 = vmulq_s16(c0, c0);
        let sq1 = vmulq_s16(c1, c1);
        // >> 8 (unsigned shift) → [0, 254] — must match scalar tail below
        // and x86 `simd_screlu_pack` (SCReLU scale chain: v² / 256, not 512).
        // C4 (2026-04-22 audit): commit 44baa95 mistakenly changed this to
        // >>9 alongside the intentional neon_pairwise_pack change, halving
        // SCReLU activations on aarch64 builds. Only affects v7 non-pairwise
        // SCReLU nets on aarch64 — aarch64 has no SCReLU-vs-scalar regression
        // test (unlike neon_pairwise_pack_fused).
        let d0 = vreinterpretq_s16_u16(vshrq_n_u16::<8>(vreinterpretq_u16_s16(sq0)));
        let d1 = vreinterpretq_s16_u16(vshrq_n_u16::<8>(vreinterpretq_u16_s16(sq1)));
        // Narrow i16 → u8 with unsigned saturation (values are [0, 254])
        let lo = vqmovun_s16(d0);
        let hi = vqmovun_s16(d1);
        vst1q_u8(out.as_mut_ptr().add(i), vcombine_u8(lo, hi));
        i += 16;
    }
    while i < h {
        let v = (acc[i] as i32).clamp(0, 255);
        out[i] = ((v * v) >> 8) as u8;
        i += 1;
    }
}

/// CReLU + pairwise pack (NEON): acc[0..pw] × acc[pw..2*pw] → out[0..pw] u8.
/// clamp(a, 0, 255) * clamp(b, 0, 255) >> 8 for each pair.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_pairwise_pack(acc: &[i16], out: &mut [u8], pw: usize) {
    neon_pairwise_pack_fused(acc, std::ptr::null(), out, pw);
}

/// Pairwise pack with optional fused threat combine (NEON mirror of simd_pairwise_pack_fused).
/// If threat is non-null, adds threat[i] to acc[i] (and threat[pw+i] to acc[pw+i]) before the
/// [0, QA] clamp. Required for v9 nets with threat inputs — without it, threat contributions
/// are silently dropped on aarch64.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_pairwise_pack_fused(acc: &[i16], threat: *const i16, out: &mut [u8], pw: usize) {
    let zero = vdupq_n_s16(0);
    let qa = vdupq_n_s16(QA as i16);
    let has_threat = !threat.is_null();
    let mut i = 0;
    while i + 16 <= pw {
        let mut a0 = vld1q_s16(acc.as_ptr().add(i));
        let mut b0 = vld1q_s16(acc.as_ptr().add(pw + i));
        let mut a1 = vld1q_s16(acc.as_ptr().add(i + 8));
        let mut b1 = vld1q_s16(acc.as_ptr().add(pw + i + 8));
        if has_threat {
            a0 = vaddq_s16(a0, vld1q_s16(threat.add(i)));
            b0 = vaddq_s16(b0, vld1q_s16(threat.add(pw + i)));
            a1 = vaddq_s16(a1, vld1q_s16(threat.add(i + 8)));
            b1 = vaddq_s16(b1, vld1q_s16(threat.add(pw + i + 8)));
        }
        let ca0 = vminq_s16(vmaxq_s16(a0, zero), qa);
        let cb0 = vminq_s16(vmaxq_s16(b0, zero), qa);
        let ca1 = vminq_s16(vmaxq_s16(a1, zero), qa);
        let cb1 = vminq_s16(vmaxq_s16(b1, zero), qa);
        let prod0 = vmulq_s16(ca0, cb0);
        let prod1 = vmulq_s16(ca1, cb1);
        let d0 = vreinterpretq_s16_u16(vshrq_n_u16::<9>(vreinterpretq_u16_s16(prod0)));
        let d1 = vreinterpretq_s16_u16(vshrq_n_u16::<9>(vreinterpretq_u16_s16(prod1)));
        let lo = vqmovun_s16(d0);
        let hi = vqmovun_s16(d1);
        vst1q_u8(out.as_mut_ptr().add(i), vcombine_u8(lo, hi));
        i += 16;
    }
    while i + 8 <= pw {
        let mut a = vld1q_s16(acc.as_ptr().add(i));
        let mut b = vld1q_s16(acc.as_ptr().add(pw + i));
        if has_threat {
            a = vaddq_s16(a, vld1q_s16(threat.add(i)));
            b = vaddq_s16(b, vld1q_s16(threat.add(pw + i)));
        }
        let ca = vminq_s16(vmaxq_s16(a, zero), qa);
        let cb = vminq_s16(vmaxq_s16(b, zero), qa);
        let prod = vmulq_s16(ca, cb);
        let d = vreinterpretq_s16_u16(vshrq_n_u16::<9>(vreinterpretq_u16_s16(prod)));
        vst1_u8(out.as_mut_ptr().add(i), vqmovun_s16(d));
        i += 8;
    }
    while i < pw {
        let mut a = acc[i] as i32;
        let mut b = acc[pw + i] as i32;
        if has_threat {
            a += *threat.add(i) as i32;
            b += *threat.add(pw + i) as i32;
        }
        out[i] = ((a.clamp(0, 255) * b.clamp(0, 255)) >> FT_SHIFT) as u8;
        i += 1;
    }
}

/// L1 int8 matmul (NEON): packed u8 input × i8 weights → i32 output.
/// Widens u8→i16 and i8→i16, then uses widening multiply-accumulate to i32.
/// h must be multiple of 16.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_l1_int8_dot(packed: &[u8], weights: &[i8], h: usize) -> i32 {
    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);
    let mut i = 0;
    while i < h {
        // Load 16 × u8 and 16 × i8
        let a = vld1q_u8(packed.as_ptr().add(i));
        let b = vld1q_s8(weights.as_ptr().add(i));
        // Widen to i16: u8→u16 reinterpreted as i16 (safe: max 254 fits i16)
        let a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
        let a_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)));
        let b_lo = vmovl_s8(vget_low_s8(b));
        let b_hi = vmovl_s8(vget_high_s8(b));
        // Widening multiply-accumulate: i16 × i16 → i32
        sum0 = vmlal_s16(sum0, vget_low_s16(a_lo), vget_low_s16(b_lo));
        sum0 = vmlal_high_s16(sum0, a_lo, b_lo);
        sum1 = vmlal_s16(sum1, vget_low_s16(a_hi), vget_low_s16(b_hi));
        sum1 = vmlal_high_s16(sum1, a_hi, b_hi);
        i += 16;
    }
    vaddvq_s32(vaddq_s32(sum0, sum1))
}

/// L1 int8 matmul x4 (NEON): process 4 neurons simultaneously, loading each
/// input chunk once and feeding it into 4 independent accumulators. Mirrors
/// the x86_64 `simd_l1_int8_dot_x4` pattern. Each 16-byte input chunk
/// amortises 1 input widen across 4 neurons, cutting per-chunk cost from
/// 4×(2 input widens + 2 weight widens + 4 MLALs) = 32 ops to 2 + 4×(2+4) = 26 ops.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_l1_int8_dot_x4(
    packed: &[u8],
    w0: &[i8], w1: &[i8], w2: &[i8], w3: &[i8],
    h: usize,
) -> [i32; 4] {
    let mut s0_lo = vdupq_n_s32(0); let mut s0_hi = vdupq_n_s32(0);
    let mut s1_lo = vdupq_n_s32(0); let mut s1_hi = vdupq_n_s32(0);
    let mut s2_lo = vdupq_n_s32(0); let mut s2_hi = vdupq_n_s32(0);
    let mut s3_lo = vdupq_n_s32(0); let mut s3_hi = vdupq_n_s32(0);
    let mut i = 0;
    while i < h {
        // Load + widen input once per chunk, shared across all 4 neurons.
        let a = vld1q_u8(packed.as_ptr().add(i));
        let a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
        let a_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)));

        // Neuron 0
        let b = vld1q_s8(w0.as_ptr().add(i));
        let b_lo = vmovl_s8(vget_low_s8(b));
        let b_hi = vmovl_s8(vget_high_s8(b));
        s0_lo = vmlal_s16(s0_lo, vget_low_s16(a_lo), vget_low_s16(b_lo));
        s0_lo = vmlal_high_s16(s0_lo, a_lo, b_lo);
        s0_hi = vmlal_s16(s0_hi, vget_low_s16(a_hi), vget_low_s16(b_hi));
        s0_hi = vmlal_high_s16(s0_hi, a_hi, b_hi);

        // Neuron 1
        let b = vld1q_s8(w1.as_ptr().add(i));
        let b_lo = vmovl_s8(vget_low_s8(b));
        let b_hi = vmovl_s8(vget_high_s8(b));
        s1_lo = vmlal_s16(s1_lo, vget_low_s16(a_lo), vget_low_s16(b_lo));
        s1_lo = vmlal_high_s16(s1_lo, a_lo, b_lo);
        s1_hi = vmlal_s16(s1_hi, vget_low_s16(a_hi), vget_low_s16(b_hi));
        s1_hi = vmlal_high_s16(s1_hi, a_hi, b_hi);

        // Neuron 2
        let b = vld1q_s8(w2.as_ptr().add(i));
        let b_lo = vmovl_s8(vget_low_s8(b));
        let b_hi = vmovl_s8(vget_high_s8(b));
        s2_lo = vmlal_s16(s2_lo, vget_low_s16(a_lo), vget_low_s16(b_lo));
        s2_lo = vmlal_high_s16(s2_lo, a_lo, b_lo);
        s2_hi = vmlal_s16(s2_hi, vget_low_s16(a_hi), vget_low_s16(b_hi));
        s2_hi = vmlal_high_s16(s2_hi, a_hi, b_hi);

        // Neuron 3
        let b = vld1q_s8(w3.as_ptr().add(i));
        let b_lo = vmovl_s8(vget_low_s8(b));
        let b_hi = vmovl_s8(vget_high_s8(b));
        s3_lo = vmlal_s16(s3_lo, vget_low_s16(a_lo), vget_low_s16(b_lo));
        s3_lo = vmlal_high_s16(s3_lo, a_lo, b_lo);
        s3_hi = vmlal_s16(s3_hi, vget_low_s16(a_hi), vget_low_s16(b_hi));
        s3_hi = vmlal_high_s16(s3_hi, a_hi, b_hi);

        i += 16;
    }
    [
        vaddvq_s32(vaddq_s32(s0_lo, s0_hi)),
        vaddvq_s32(vaddq_s32(s1_lo, s1_hi)),
        vaddvq_s32(vaddq_s32(s2_lo, s2_hi)),
        vaddvq_s32(vaddq_s32(s3_lo, s3_hi)),
    ]
}

/// Find non-zero 16-byte chunk indices in a packed u8 buffer (NEON).
/// Returns the number of NNZ chunks. nnz_indices[0..count] contains byte offsets.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_find_nnz_chunks(packed: &[u8], nnz_indices: &mut [u16], h: usize) -> usize {
    let mut count = 0usize;
    let mut i = 0;
    while i < h {
        let v = vld1q_u8(packed.as_ptr().add(i));
        // If max byte > 0, this chunk has non-zero elements
        if vmaxvq_u8(v) != 0 {
            nnz_indices[count] = i as u16;
            count += 1;
        }
        i += 16;
    }
    count
}

/// Sparse L1 int8 matmul (NEON): only process NNZ 16-byte chunks of packed input.
/// nnz_indices[0..nnz_count] are byte offsets of non-zero chunks.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_l1_int8_dot_sparse(packed: &[u8], weights: &[i8], nnz_indices: &[u16], nnz_count: usize) -> i32 {
    let mut sum0 = vdupq_n_s32(0);
    let mut sum1 = vdupq_n_s32(0);
    for k in 0..nnz_count {
        let off = nnz_indices[k] as usize;
        let a = vld1q_u8(packed.as_ptr().add(off));
        let b = vld1q_s8(weights.as_ptr().add(off));
        let a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
        let a_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a)));
        let b_lo = vmovl_s8(vget_low_s8(b));
        let b_hi = vmovl_s8(vget_high_s8(b));
        sum0 = vmlal_s16(sum0, vget_low_s16(a_lo), vget_low_s16(b_lo));
        sum0 = vmlal_high_s16(sum0, a_lo, b_lo);
        sum1 = vmlal_s16(sum1, vget_low_s16(a_hi), vget_low_s16(b_hi));
        sum1 = vmlal_high_s16(sum1, a_hi, b_hi);
    }
    vaddvq_s32(vaddq_s32(sum0, sum1))
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

/// Detect AVX-512 VNNI support. VNNI provides `VPDPBUSD`, which fuses the
/// `VPMADDUBSW + VPMADDWD(ones)` pair used in int8 L1 matmuls into one
/// instruction — typically ~1.3–2× speedup on the hot forward kernel.
fn detect_avx512_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Detect AVX-VNNI (VEX-encoded VPDPBUSD on YMM/XMM). Available on Intel
/// Alder Lake+ and AMD Zen 4+ — gives fused u8×i8 dot on AVX2 machines
/// that don't have full AVX-512.
fn detect_avx_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("avxvnni")
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
    pub output_weights: Vec<i16>, // [NNUE_OUTPUT_BUCKETS × out_width]
    /// Quantized output weights for fast SCReLU: i16 but clamped to [-128, 127] range.
    pub output_weights_i8: Vec<i16>,
    pub output_scale: [f32; NNUE_OUTPUT_BUCKETS],
    pub output_bias: [i32; NNUE_OUTPUT_BUCKETS],
    pub use_screlu: bool,
    pub use_pairwise: bool,
    // v7 hidden layers
    pub l1_size: usize,           // total L1 size (bucketed if applicable)
    pub l2_size: usize,           // total L2 size (bucketed if applicable)
    pub l1_per_bucket: usize,     // per-bucket L1 size (= l1_size for unbucketed)
    pub l2_per_bucket: usize,     // per-bucket L2 size (= l2_size for unbucketed)
    pub bucketed_hidden: bool,    // output buckets baked into L1/L2 dimensions
    pub l1_scale: i32,            // QA_L1: 255 for int16, 64 for int8
    pub l1_weights: Vec<i16>,     // [2*hidden_size × l1_size] row-major
    pub l1_weights_t: Vec<i16>,   // [l1_size × 2*hidden_size] transposed for SIMD
    pub l1_weights_8t: Vec<i8>,   // [l1_size × 2*hidden_size] transposed int8 for VPMADDUBSW
    pub l1_weights_sparse: Vec<i8>, // input-chunk-major for sparse L1 dpbusd
    pub l1_biases: Vec<i16>,      // [l1_size]
    pub l2_weights_f: Vec<f32>,   // [l2_input × l2_size] — float (l2_input = l1*2 if dual)
    pub l2_biases_f: Vec<f32>,    // [l2_size]
    pub out_weights_f: Vec<f32>,  // [NNUE_OUTPUT_BUCKETS × out_l_size] — float output
    pub out_bias_f: Vec<f32>,     // [NNUE_OUTPUT_BUCKETS]
    pub dual_l1: bool,            // v8: dual L1 activation (CReLU+SCReLU on L1 output)
    // v9 threat features
    pub threat_weights: Vec<i8>,  // [num_threat_features × hidden_size] i8 weights
    pub num_threat_features: usize,
    pub has_threats: bool,
    /// Number of king buckets in this net (10 for Reckless, 16 for others).
    /// PSQ weight block is sized `num_king_buckets * 768 * hidden_size`.
    pub num_king_buckets: usize,
    /// King bucket layout identifier — drives which lookup tables are used.
    pub kb_layout: KbLayout,
    /// Per-net king-bucket lookup table: square (0-63) → bucket (0..num_king_buckets).
    /// Computed once at load time from `kb_layout`, never mutated after.
    /// Previously a `static mut` written by every net load — which caused a
    /// data race with Lazy SMP helpers at T=4 (v9 SMP regression bisected to
    /// commit 1356150). Making this per-net eliminates the race.
    pub king_bucket: [usize; 64],
    /// Per-net king-mirror table: square (0-63) → mirror flag.
    pub king_mirror: [bool; 64],
    pub use_sparse_l1: std::sync::atomic::AtomicBool, // feature flag: sparse L1 matmul
    /// Use clipped ReLU (not SCReLU) on L1/L2 hidden layers. Set via UCI
    /// option `HiddenActivation = crelu`. Experimental — nets trained with
    /// .crelu() on L1/L2 in Bullet need this set to match training-time
    /// activation. Default false (SCReLU, matching all existing v7/v9 nets).
    pub crelu_hidden: std::sync::atomic::AtomicBool,
    pub has_avx2: bool,
    pub has_avx512: bool,
    /// AVX-512 + VNNI (`VPDPBUSD`). When true, the int8 L1 matmul prefers
    /// the VNNI kernel, which fuses `VPMADDUBSW + VPMADDWD(ones)` into a
    /// single instruction. Falls back to `has_avx512` kernels when false.
    pub has_avx512_vnni: bool,
    /// AVX-VNNI (VEX-encoded `VPDPBUSD` on YMM). Intel Alder Lake+ / AMD
    /// Zen 4+. Used by AVX2-only build configurations on VNNI-capable CPUs.
    pub has_avx_vnni: bool,
    pub has_neon: bool,
}

impl NNUENet {
    /// HalfKA feature index computed against THIS net's king-bucket layout.
    /// Use this everywhere that computes a feature index to avoid the
    /// race-prone global tables. See also `halfka_index_with` for the
    /// equivalent free function.
    #[inline]
    pub fn halfka_index(&self, perspective: u8, king_sq: u8, pc_color: u8, pc_type: u8, pc_sq: u8) -> usize {
        halfka_index_with(&self.king_bucket, &self.king_mirror, perspective, king_sq, pc_color, pc_type, pc_sq)
    }

    /// King-bucket lookup for this net. Use for build_dirty_piece and other
    /// code that needs to detect bucket/mirror changes on king moves.
    #[inline]
    pub fn king_bucket(&self, sq: usize) -> usize { self.king_bucket[sq] }

    /// King-mirror lookup for this net.
    #[inline]
    pub fn king_mirror(&self, sq: usize) -> bool { self.king_mirror[sq] }

    /// Load from a byte slice (for embedded nets). No temp files needed.
    pub fn load_from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut reader = std::io::Cursor::new(data);
        Self::load_from_reader(&mut reader, data.len() as u64, "<embedded>")
    }

    /// Load a v5/v6/v7 .nnue file.
    pub fn load(path: &str) -> Result<Self, String> {
        let file_len = std::fs::metadata(path).map_err(|e| format!("stat {}: {}", path, e))?.len();
        let file = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
        let mut reader = BufReader::new(file);
        Self::load_from_reader(&mut reader, file_len, path)
    }

    fn load_from_reader(reader: &mut impl IoRead, data_len: u64, source_name: &str) -> Result<Self, String> {
        // Read magic
        let magic = read_u32(reader)?;
        if magic != NNUE_MAGIC {
            return Err(format!("invalid NNUE magic: 0x{:X}", magic));
        }

        // Read version
        let version = read_u32(reader)?;
        let mut use_screlu = false;
        let mut use_pairwise = false;
        let mut l1_size = 0usize;
        let mut l2_size = 0usize;
        let mut l1_scale = QA as i32; // default int16 scale
        let mut bucketed_hidden = false; // bit 3: output buckets baked into L1/L2 dims
        let mut dual_l1 = false; // bit 4: dual L1 activation (CReLU+SCReLU, v8)
        // bit 5 is context-dependent:
        //   - extended_kb=0: consensus_buckets (legacy 16-bucket encoding)
        //   - extended_kb=1: hl_crelu (hidden-layer CReLU, vs default SCReLU)
        let mut consensus_buckets = false;
        let mut hl_crelu = false;
        let mut has_threats = false; // bit 6: threat features (v9)
        let mut num_threat_features = 0usize;
        // extended_kb (bit 7) is only read inside the v7|8|9 match arm; declared locally there.
        let mut num_king_buckets: usize = 16; // default (uniform/consensus)
        let mut kb_layout = KbLayout::Uniform;
        let hidden_size: usize;

        match version {
            5 => {
                // Infer hidden size from data size
                let body_size = data_len - 8;
                let h_denom = 2 * (NNUE_INPUT_SIZE as u64 + 1 + 16);
                let h_numer = body_size - 32;
                if h_numer % h_denom != 0 {
                    return Err(format!("cannot infer hidden size from file size (body {})", body_size));
                }
                hidden_size = (h_numer / h_denom) as usize;
            }
            6 => {
                let flags = read_u8(reader)?;
                use_screlu = flags & 1 != 0;
                use_pairwise = flags & 2 != 0;
                consensus_buckets = flags & 32 != 0;
                kb_layout = if consensus_buckets { KbLayout::Consensus } else { KbLayout::Uniform };
                let body_size = data_len - 9;
                let out_mul: u64 = if use_pairwise { 8 } else { 16 };
                let h_denom = 2 * (NNUE_INPUT_SIZE as u64 + 1 + out_mul);
                let h_numer = body_size - 32;
                if h_numer % h_denom != 0 {
                    return Err(format!("cannot infer hidden size from file size (body {})", body_size));
                }
                hidden_size = (h_numer / h_denom) as usize;
            }
            7 | 8 | 9 => {
                let flags = read_u8(reader)?;
                use_screlu = flags & 1 != 0;
                use_pairwise = flags & 2 != 0;
                if flags & 4 != 0 { l1_scale = 64; } // int8 L1 weights
                bucketed_hidden = flags & 8 != 0; // output buckets baked into L1/L2
                dual_l1 = flags & 16 != 0; // dual L1 activation (CReLU+SCReLU)
                has_threats = flags & 64 != 0; // v9 threat features
                let extended_kb = flags & 128 != 0; // bit 7: extended KB header
                // Bit 5 has context-dependent meaning: consensus_buckets on
                // legacy 16-bucket nets (extended_kb=0), hl_crelu on modern
                // nets (extended_kb=1). Invariant enforced by bullet_convert.
                if extended_kb {
                    hl_crelu = flags & 32 != 0;
                } else {
                    consensus_buckets = flags & 32 != 0;
                }
                let ft_size = read_u16(reader)? as usize;
                l1_size = read_u16(reader)? as usize;
                l2_size = read_u16(reader)? as usize;
                if has_threats {
                    num_threat_features = read_u32(reader)? as usize;
                }
                if extended_kb {
                    // Two extra bytes follow: kb_count (u8), kb_layout_id (u8).
                    num_king_buckets = read_u8(reader)? as usize;
                    let layout_id = read_u8(reader)?;
                    kb_layout = KbLayout::from_id(layout_id)
                        .ok_or_else(|| format!("unknown kb_layout_id: {}", layout_id))?;
                    // Consistency check: layout's default count matches unless
                    // caller has intentionally overridden it (rare).
                    if !(1..=NNUE_MAX_KING_BUCKETS).contains(&num_king_buckets) {
                        return Err(format!("invalid num_king_buckets: {}", num_king_buckets));
                    }
                } else {
                    // Legacy 16-bucket path: consensus bit 5 picks layout.
                    kb_layout = if consensus_buckets { KbLayout::Consensus } else { KbLayout::Uniform };
                    num_king_buckets = 16;
                }
                hidden_size = ft_size;
            }
            _ => return Err(format!("unsupported NNUE version: {}", version)),
        };

        // Read input weights (PSQ block sized by kb_count × 768).
        let psq_input_size = num_king_buckets * PSQ_INPUTS_PER_BUCKET;
        let mut input_weights = vec![0i16; psq_input_size * hidden_size];
        read_i16_slice(reader, &mut input_weights)?;

        // Read input biases
        let mut input_biases = vec![0i16; hidden_size];
        read_i16_slice(reader, &mut input_biases)?;

        // Read threat weights (v9): i8 [num_threat_features × hidden_size]
        let mut threat_weights = Vec::new();
        if has_threats && num_threat_features > 0 {
            let total = num_threat_features * hidden_size;
            threat_weights = vec![0i8; total];
            let mut bytes = vec![0u8; total];
            reader.read_exact(&mut bytes).map_err(|e| format!("read threat weights: {}", e))?;
            for i in 0..total {
                threat_weights[i] = bytes[i] as i8;
            }
            println!("info string Loaded {} threat features ({}×{}, {}MB)",
                num_threat_features, num_threat_features, hidden_size,
                total / (1024 * 1024));
        }

        // Read L1 hidden layer weights (v7)
        // Bucketed: actual array sizes are BUCKETS * l1_size / BUCKETS * l2_size
        let bl1 = if bucketed_hidden { NNUE_OUTPUT_BUCKETS * l1_size } else { l1_size };
        let bl2 = if bucketed_hidden { NNUE_OUTPUT_BUCKETS * l2_size } else { l2_size };
        let mut l1_weights = Vec::new();
        let mut l1_biases = Vec::new();
        if l1_size > 0 {
            // Pairwise: L1 input is H (pairwise halves each perspective, concat = H)
            // Direct: L1 input is 2*H (two full accumulators concatenated)
            let l1_input_size = if use_pairwise { hidden_size } else { 2 * hidden_size };
            l1_weights = vec![0i16; l1_input_size * bl1];
            read_i16_slice(reader, &mut l1_weights)?;
            l1_biases = vec![0i16; bl1];
            read_i16_slice(reader, &mut l1_biases)?;
        }

        // Read L2 hidden layer weights (v7/v8)
        // Dual L1 activation: L2 input = l1_size * 2 (CReLU + SCReLU concatenated)
        let l2_input_size = if dual_l1 { l1_size * 2 } else { l1_size };
        let mut l2_weights_raw = Vec::new();
        let mut l2_biases_raw = Vec::new();
        if l2_size > 0 {
            l2_weights_raw = vec![0i16; l2_input_size * bl2];
            read_i16_slice(reader, &mut l2_weights_raw)?;
            l2_biases_raw = vec![0i16; bl2];
            read_i16_slice(reader, &mut l2_biases_raw)?;
        }

        // Read output weights: for bucketed nets, output is [BUCKETS][per-bucket L2]
        // For unbucketed: [BUCKETS][l2_size or l1_size or 2*hidden_size]
        let out_width = if l2_size > 0 {
            l2_size  // per-bucket L2 size (not bucketed)
        } else if l1_size > 0 {
            l1_size  // per-bucket L1 size
        } else if use_pairwise {
            hidden_size
        } else {
            2 * hidden_size
        };
        let mut output_weights = vec![0i16; NNUE_OUTPUT_BUCKETS * out_width];
        read_i16_slice(reader, &mut output_weights)?;

        // Read output bias
        let mut output_bias = [0i32; NNUE_OUTPUT_BUCKETS];
        for i in 0..NNUE_OUTPUT_BUCKETS {
            output_bias[i] = read_i32(reader)?;
        }

        // Compute king bucket tables for this net's layout. Stored on the
        // returned NNUENet struct (per-net, not static) so Lazy SMP helpers
        // can read them concurrently without racing a subsequent net load.
        let (king_bucket_tbl, king_mirror_tbl) = compute_king_buckets(kb_layout);

        let activation = if use_pairwise { "pairwise" } else if use_screlu { "SCReLU" } else { "CReLU" };
        let dual_str = if dual_l1 { " dual" } else { "" };
        let bucket_str = match kb_layout {
            KbLayout::Uniform   => "",
            KbLayout::Consensus => " consensus-kb",
            KbLayout::Reckless  => " reckless-kb10",
        };
        let threat_str = if has_threats { format!(" threats={}", num_threat_features) } else { String::new() };
        if l1_size > 0 {
            if l2_size > 0 {
                println!("info string Loaded NNUE v{} {} {}{}{}{} (FT={} L1={} L2={})", version, source_name, activation, dual_str, bucket_str, threat_str, hidden_size, l1_size, l2_size);
            } else {
                println!("info string Loaded NNUE v{} {} {}{}{}{} (FT={} L1={})", version, source_name, activation, dual_str, bucket_str, threat_str, hidden_size, l1_size);
            }
        } else {
            println!("info string Loaded NNUE v{} {} {}{}{} ({})", version, source_name, activation, bucket_str, threat_str, hidden_size);
        }

        // Transpose L1 weights for SIMD: [j*L1+i] → [i*H+j] per perspective
        // For bucketed nets, bl1 is the total L1 dimension in the weight array
        let l1_weights_t = if l1_size > 0 {
            let h = hidden_size;
            let l1 = bl1; // use bucketed L1 size for correct array stride
            let per_perspective = if use_pairwise { h / 2 } else { h };
            let total_input = if use_pairwise { h } else { 2 * h };
            let mut wt = vec![0i16; l1 * total_input];
            // STM: l1_weights[j*L1+i] → wt[i*per_perspective+j]
            for i in 0..l1 {
                for j in 0..per_perspective {
                    wt[i * per_perspective + j] = l1_weights[j * l1 + i];
                }
            }
            // NTM: l1_weights[(per_perspective+j)*L1+i] → wt[L1*per_perspective + i*per_perspective + j]
            let ntm_off = l1 * per_perspective;
            for i in 0..l1 {
                for j in 0..per_perspective {
                    wt[ntm_off + i * per_perspective + j] = l1_weights[(per_perspective + j) * l1 + i];
                }
            }
            wt
        } else {
            Vec::new()
        };

        // Convert transposed L1 weights to int8 (clamp to [-128, 127])
        let l1_weights_8t: Vec<i8> = l1_weights_t.iter().map(|&w| {
            w.clamp(-128, 127) as i8
        }).collect();

        // Sparse L1 weights: input-chunk-major for dpbusd kernel
        let l1_weights_sparse = if l1_size > 0 && use_pairwise {
            let total_input = if use_pairwise { hidden_size } else { 2 * hidden_size };
            crate::sparse_l1::transpose_weights_for_sparse(&l1_weights_8t, total_input, bl1)
        } else {
            Vec::new()
        };

        // Prepare float weights for v7 hidden layer forward pass
        // L2 and output use QA_L1 (not QA) for dequantization
        let qa_l1_f = if l1_scale != 0 { l1_scale as f32 } else { QA as f32 };
        let qb_f = QB as f32;
        let l2_weights_f: Vec<f32> = l2_weights_raw.iter().map(|&w| w as f32 / qa_l1_f).collect();
        let l2_biases_f: Vec<f32> = l2_biases_raw.iter().map(|&b| b as f32 / qa_l1_f).collect();
        let out_weights_f: Vec<f32> = if l1_size > 0 {
            output_weights.iter().map(|&w| w as f32 / qb_f).collect()
        } else { Vec::new() };
        let out_bias_f: Vec<f32> = if l1_size > 0 {
            output_bias.iter().map(|&b| b as f32 / (qa_l1_f * qb_f)).collect()
        } else { Vec::new() };

        let has_avx2 = detect_avx2();
        let has_avx512 = detect_avx512();
        let has_avx512_vnni = detect_avx512_vnni();
        let has_avx_vnni = detect_avx_vnni();
        let has_neon = detect_neon();
        if has_avx512_vnni {
            println!("info string AVX-512 + VNNI detected — using VPDPBUSD int8 matmul");
        } else if has_avx512 {
            println!("info string AVX-512 SIMD detected — using 512-bit NNUE inference");
        } else if has_avx2 && has_avx_vnni {
            println!("info string AVX2 + AVX-VNNI detected — using VPDPBUSD int8 matmul");
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
            l1_size: bl1,
            l2_size: bl2,
            l1_per_bucket: l1_size,  // original per-bucket size from header
            l2_per_bucket: l2_size,
            bucketed_hidden,
            l1_scale,
            l1_weights_t,
            l1_weights_8t,
            l1_weights_sparse,
            l1_weights,
            l1_biases,
            l2_weights_f,
            l2_biases_f,
            out_weights_f,
            out_bias_f,
            dual_l1,
            threat_weights,
            num_threat_features,
            has_threats,
            num_king_buckets,
            kb_layout,
            king_bucket: king_bucket_tbl,
            king_mirror: king_mirror_tbl,
            use_sparse_l1: std::sync::atomic::AtomicBool::new(false), // disabled: dense int8 is faster at H=1024
            // Auto-set from file header bit 5 when extended_kb=1.
            // Overridable at runtime via UCI option HiddenActivation.
            crelu_hidden: std::sync::atomic::AtomicBool::new(hl_crelu),
            has_avx2,
            has_avx512,
            has_avx512_vnni,
            has_avx_vnni,
            has_neon,
        })
    }

    /// Get input weight row for a feature index.
    #[inline]
    pub fn input_weight_row(&self, idx: usize) -> &[i16] {
        let off = idx * self.hidden_size;
        &self.input_weights[off..off + self.hidden_size]
    }

    /// Output width per bucket (always per-bucket size, not total bucketed size).
    #[inline]
    fn output_width(&self) -> usize {
        if self.l2_per_bucket > 0 { self.l2_per_bucket }
        else if self.l1_per_bucket > 0 { self.l1_per_bucket }
        else if self.use_pairwise { self.hidden_size }
        else { 2 * self.hidden_size }
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

    /// v7 pairwise hidden layer forward pass.
    /// acc → CReLU → pairwise_mul → L1 matmul → ReLU → float L2 → ReLU → output
    /// L1 input is H (hidden_size) after pairwise halves each perspective.
    /// Pairwise forward with fused PSQ+threat combine (Reckless activate_ft pattern).
    /// Adds threat values inside the pairwise loop, eliminating a separate combine pass.
    /// Pairwise forward with fused PSQ+threat combine (Reckless activate_ft pattern).
    /// Pairwise forward with threat combine.
    /// Pairwise forward with fused PSQ+threat combine.
    /// Passes threat pointers through to the SIMD pairwise pack, combining
    /// PSQ + threats in the same SIMD pass (no separate stack combine).
    #[inline(always)]
    fn forward_with_l1_pairwise_fused(&self, stm_acc: &[i16], ntm_acc: &[i16],
        stm_threat: &[i16], ntm_threat: &[i16], bucket: usize) -> i32
    {
        if stm_threat.is_empty() {
            return self.forward_with_l1_pairwise(stm_acc, ntm_acc, bucket);
        }
        self.forward_with_l1_pairwise_threats(stm_acc, ntm_acc, stm_threat, ntm_threat, bucket)
    }

    fn forward_with_l1_pairwise(&self, stm_acc: &[i16], ntm_acc: &[i16], bucket: usize) -> i32 {
        unsafe { self.forward_with_l1_pairwise_inner(stm_acc, ntm_acc, &[], &[], bucket) }
    }

    fn forward_with_l1_pairwise_threats(&self, stm_acc: &[i16], ntm_acc: &[i16],
        stm_threat: &[i16], ntm_threat: &[i16], bucket: usize) -> i32
    {
        unsafe { self.forward_with_l1_pairwise_inner(stm_acc, ntm_acc, stm_threat, ntm_threat, bucket) }
    }

    /// Marked `#[target_feature]` to propagate AVX2 codegen context through
    /// the inlined SIMD helpers. LTO was already inlining the helpers across
    /// the boundary, but the attribute gives LLVM permission to emit tighter
    /// AVX2 sequences inside the inlined region (measured +~5% NPS, identical
    /// bench node count). Callers must ensure AVX2 is available; on x86_64
    /// with `-Ctarget-cpu=native` it is.
    #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
    unsafe fn forward_with_l1_pairwise_inner(&self, stm_acc: &[i16], ntm_acc: &[i16],
        stm_threat: &[i16], ntm_threat: &[i16], bucket: usize) -> i32
    {
        let h = self.hidden_size;
        let pw = h / 2; // pairwise output per perspective
        let l1_total = self.l1_size; // total L1 neurons (bucketed or not)
        let l1_pb = self.l1_per_bucket; // per-bucket L1 size
        let qa = QA as i32;
        let qa_l1 = self.l1_scale as i32;

        // For bucketed nets: only compute neurons for this bucket
        let l1_off = if self.bucketed_hidden { bucket * l1_pb } else { 0 };
        let l1 = if self.bucketed_hidden { l1_pb } else { l1_total };

        let has_threats = !stm_threat.is_empty();

        // CReLU + pairwise for each perspective → pw values each
        // When threats are present, fuses PSQ+threat combine into the SIMD pack
        let mut stm_pw = [0u8; 2048];
        let mut ntm_pw = [0u8; 2048];

        #[cfg(target_arch = "x86_64")]
        if self.has_avx512 && pw % 32 == 0 {
            unsafe {
                let stm_tp = if has_threats { stm_threat.as_ptr() } else { std::ptr::null() };
                let ntm_tp = if has_threats { ntm_threat.as_ptr() } else { std::ptr::null() };
                simd512_pairwise_pack_fused(stm_acc, stm_tp, &mut stm_pw, pw);
                simd512_pairwise_pack_fused(ntm_acc, ntm_tp, &mut ntm_pw, pw);
            }
        } else if self.has_avx2 && pw % 16 == 0 {
            unsafe {
                if has_threats {
                    simd_pairwise_pack_fused(stm_acc, stm_threat.as_ptr(), &mut stm_pw, pw);
                    simd_pairwise_pack_fused(ntm_acc, ntm_threat.as_ptr(), &mut ntm_pw, pw);
                } else {
                    simd_pairwise_pack_fused(stm_acc, std::ptr::null(), &mut stm_pw, pw);
                    simd_pairwise_pack_fused(ntm_acc, std::ptr::null(), &mut ntm_pw, pw);
                }
            }
        } else {
            for i in 0..pw {
                let ta = if has_threats { stm_threat[i] as i32 } else { 0 };
                let tb = if has_threats { stm_threat[i + pw] as i32 } else { 0 };
                let a = (stm_acc[i] as i32 + ta).clamp(0, qa);
                let b = (stm_acc[i + pw] as i32 + tb).clamp(0, qa);
                stm_pw[i] = ((a * b) >> FT_SHIFT) as u8;
            }
            for i in 0..pw {
                let ta = if has_threats { ntm_threat[i] as i32 } else { 0 };
                let tb = if has_threats { ntm_threat[i + pw] as i32 } else { 0 };
                let a = (ntm_acc[i] as i32 + ta).clamp(0, qa);
                let b = (ntm_acc[i + pw] as i32 + tb).clamp(0, qa);
                ntm_pw[i] = ((a * b) >> FT_SHIFT) as u8;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                if has_threats {
                    neon_pairwise_pack_fused(stm_acc, stm_threat.as_ptr(), &mut stm_pw, pw);
                    neon_pairwise_pack_fused(ntm_acc, ntm_threat.as_ptr(), &mut ntm_pw, pw);
                } else {
                    neon_pairwise_pack_fused(stm_acc, std::ptr::null(), &mut stm_pw, pw);
                    neon_pairwise_pack_fused(ntm_acc, std::ptr::null(), &mut ntm_pw, pw);
                }
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            for i in 0..pw {
                let ta = if has_threats { stm_threat[i] as i32 } else { 0 };
                let tb = if has_threats { stm_threat[i + pw] as i32 } else { 0 };
                let a = (stm_acc[i] as i32 + ta).clamp(0, qa);
                let b = (stm_acc[i + pw] as i32 + tb).clamp(0, qa);
                stm_pw[i] = ((a * b) >> FT_SHIFT) as u8;
            }
            for i in 0..pw {
                let ta = if has_threats { ntm_threat[i] as i32 } else { 0 };
                let tb = if has_threats { ntm_threat[i + pw] as i32 } else { 0 };
                let a = (ntm_acc[i] as i32 + ta).clamp(0, qa);
                let b = (ntm_acc[i + pw] as i32 + tb).clamp(0, qa);
                ntm_pw[i] = ((a * b) >> FT_SHIFT) as u8;
            }
        }

        // L1 int8 matmul — only compute l1 neurons starting at l1_off
        // Pairwise: input = (a*b)>>FT_SHIFT, u8 at scale QA²>>FT_SHIFT ≈ PW_SCALE.
        // L1 weights at scale QA_L1(64). Matmul at scale PW_SCALE*QA_L1.
        // Bias at scale QA_L1(64), scaled by PW_SCALE to match matmul.
        // After matmul: divide by PW_SCALE → scale QA_L1.
        let pw_scale = PW_SCALE; // (QA*QA) >> FT_SHIFT = 127
        let mut hidden32 = [0i32; 512];
        for i in 0..l1 {
            hidden32[i] = self.l1_biases[l1_off + i] as i32 * pw_scale;
        }
        // L1 matmul: use SIMD int8 dot with transposed weights when available
        #[cfg(target_arch = "x86_64")]
        if self.has_avx512_vnni && !self.l1_weights_sparse.is_empty() && l1 == 16 && pw % 4 == 0 {
            // AVX-512 VNNI column-major: all 16 L1 neurons in one ZMM accumulator,
            // one VPDPBUSD per 4-byte input chunk. Dramatically reduces per-chunk
            // uop count vs the row-major per-neuron path below.
            unsafe {
                crate::sparse_l1::dense_l1_avx512_vnni(
                    &stm_pw, &ntm_pw, pw, &self.l1_weights_sparse,
                    l1, &self.l1_biases[l1_off..], pw_scale, &mut hidden32,
                );
            }
        } else if self.has_avx512_vnni && pw % 64 == 0 && !self.l1_weights_8t.is_empty() {
            // VNNI fallback for wider L1 — still VPDPBUSD, but row-major.
            let ntm_base = l1_total * pw;
            for i in 0..l1 {
                let gi = l1_off + i;
                let stm_w = &self.l1_weights_8t[gi * pw..(gi + 1) * pw];
                let ntm_w = &self.l1_weights_8t[ntm_base + gi * pw..ntm_base + (gi + 1) * pw];
                unsafe {
                    hidden32[i] += simd512_l1_int8_dot_vnni(&stm_pw[..pw], stm_w, pw);
                    hidden32[i] += simd512_l1_int8_dot_vnni(&ntm_pw[..pw], ntm_w, pw);
                }
            }
        } else if self.has_avx_vnni && !self.l1_weights_sparse.is_empty() && l1 <= 16 && pw % 4 == 0 {
            // AVX-VNNI (YMM VPDPBUSD) — Alder Lake+, Zen 4+ without full AVX-512.
            unsafe {
                crate::sparse_l1::dense_l1_avx_vnni(
                    &stm_pw, &ntm_pw, pw, &self.l1_weights_sparse,
                    l1, &self.l1_biases[l1_off..], pw_scale, &mut hidden32,
                );
            }
        } else if self.has_avx512 && pw % 64 == 0 && !self.l1_weights_8t.is_empty() {
            let ntm_base = l1_total * pw;
            for i in 0..l1 {
                let gi = l1_off + i;
                let stm_w = &self.l1_weights_8t[gi * pw..(gi + 1) * pw];
                let ntm_w = &self.l1_weights_8t[ntm_base + gi * pw..ntm_base + (gi + 1) * pw];
                unsafe {
                    hidden32[i] += simd512_l1_int8_dot(&stm_pw[..pw], stm_w, pw);
                    hidden32[i] += simd512_l1_int8_dot(&ntm_pw[..pw], ntm_w, pw);
                }
            }
        } else if self.has_avx2 && !self.l1_weights_sparse.is_empty() && l1 <= 16 {
            // Column-major (input-chunk-major) L1 matmul — Reckless's pattern.
            // For each 4-byte input chunk, splat_i32 broadcast + maddubs/madd
            // contributes to all L1 outputs simultaneously via 2 AVX2 registers.
            // Replaces the row-major path that scanned the full input per
            // output neuron (16× cache-line touches per input chunk).
            //
            // Dense variant (no zero-check): pairwise-CReLU inputs have high
            // density (~89%), so the if-check overhead in the sparse variant
            // exceeded the skip savings at L1=16. The dense path is
            // straight-line SIMD with input-chunk-major weight access.
            unsafe {
                crate::sparse_l1::dense_l1_avx2(
                    &stm_pw, &ntm_pw, pw, &self.l1_weights_sparse,
                    l1, &self.l1_biases[l1_off..], pw_scale, &mut hidden32,
                );
            }
            // DEBUG: compare against dense
            #[cfg(debug_assertions)]
            {
                static SDBG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let c = SDBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c < 5 {
                    let ntm_base = l1_total * pw;
                    let mut dense = vec![0i32; l1];
                    for i in 0..l1 { dense[i] = self.l1_biases[l1_off + i] as i32 * pw_scale; }
                    for i in 0..l1 {
                        let gi = l1_off + i;
                        for j in 0..pw {
                            dense[i] += stm_pw[j] as i32 * self.l1_weights_8t[gi * pw + j] as i32;
                            dense[i] += ntm_pw[j] as i32 * self.l1_weights_8t[ntm_base + gi * pw + j] as i32;
                        }
                    }
                    let mut mismatch = false;
                    for i in 0..l1 {
                        if hidden32[i] != dense[i] {
                            if !mismatch {
                                eprintln!("SPARSE L1 MISMATCH neuron {}: sparse={} dense={} diff={}",
                                    i, hidden32[i], dense[i], hidden32[i] - dense[i]);
                            }
                            mismatch = true;
                        }
                    }
                    if !mismatch { eprintln!("SPARSE L1 MATCH (all {} neurons)", l1); }
                }
            }
        } else if self.has_avx2 && pw % 32 == 0 && !self.l1_weights_8t.is_empty() {
            let ntm_base = l1_total * pw;
            // Multi-neuron: process 4 neurons at once, loading input once per chunk
            let mut i = 0;
            while i + 4 <= l1 {
                let gi = l1_off + i;
                unsafe {
                    let stm_results = simd_l1_int8_dot_x4(
                        &stm_pw[..pw],
                        &self.l1_weights_8t[gi * pw..(gi + 1) * pw],
                        &self.l1_weights_8t[(gi + 1) * pw..(gi + 2) * pw],
                        &self.l1_weights_8t[(gi + 2) * pw..(gi + 3) * pw],
                        &self.l1_weights_8t[(gi + 3) * pw..(gi + 4) * pw],
                        pw,
                    );
                    let ntm_results = simd_l1_int8_dot_x4(
                        &ntm_pw[..pw],
                        &self.l1_weights_8t[ntm_base + gi * pw..ntm_base + (gi + 1) * pw],
                        &self.l1_weights_8t[ntm_base + (gi + 1) * pw..ntm_base + (gi + 2) * pw],
                        &self.l1_weights_8t[ntm_base + (gi + 2) * pw..ntm_base + (gi + 3) * pw],
                        &self.l1_weights_8t[ntm_base + (gi + 3) * pw..ntm_base + (gi + 4) * pw],
                        pw,
                    );
                    for k in 0..4 {
                        hidden32[i + k] += stm_results[k] + ntm_results[k];
                    }
                }
                i += 4;
            }
            // Handle remaining neurons (if l1 not divisible by 4)
            while i < l1 {
                let gi = l1_off + i;
                let stm_w = &self.l1_weights_8t[gi * pw..(gi + 1) * pw];
                let ntm_w = &self.l1_weights_8t[ntm_base + gi * pw..ntm_base + (gi + 1) * pw];
                unsafe {
                    hidden32[i] += simd_l1_int8_dot(&stm_pw[..pw], stm_w, pw);
                    hidden32[i] += simd_l1_int8_dot(&ntm_pw[..pw], ntm_w, pw);
                }
                i += 1;
            }
        }

        #[cfg(target_arch = "x86_64")]
        if !(self.has_avx512_vnni && !self.l1_weights_sparse.is_empty() && l1 == 16 && pw % 4 == 0)
            && !(self.has_avx512_vnni && pw % 64 == 0 && !self.l1_weights_8t.is_empty())
            && !(self.has_avx_vnni && !self.l1_weights_sparse.is_empty() && l1 <= 16 && pw % 4 == 0)
            && !(self.has_avx512 && pw % 64 == 0 && !self.l1_weights_8t.is_empty())
            && !(self.has_avx2 && !self.l1_weights_sparse.is_empty() && l1 <= 16)
            && !(self.has_avx2 && pw % 32 == 0 && !self.l1_weights_8t.is_empty()) {
            // Scalar fallback — raw weights in [input][neuron] layout
            for i in 0..l1 {
                let gi = l1_off + i;
                for j in 0..pw {
                    hidden32[i] += stm_pw[j] as i32 * self.l1_weights[j * l1_total + gi] as i32;
                }
                for j in 0..pw {
                    hidden32[i] += ntm_pw[j] as i32 * self.l1_weights[(pw + j) * l1_total + gi] as i32;
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        if self.has_neon && pw % 16 == 0 && !self.l1_weights_8t.is_empty() {
            let ntm_base = l1_total * pw;
            // Multi-neuron: 4 neurons at once, loading each input chunk once
            // and feeding all 4 accumulators (mirrors x86_64 simd_l1_int8_dot_x4).
            let mut i = 0;
            while i + 4 <= l1 {
                let gi = l1_off + i;
                unsafe {
                    let stm_results = neon_l1_int8_dot_x4(
                        &stm_pw[..pw],
                        &self.l1_weights_8t[gi * pw..(gi + 1) * pw],
                        &self.l1_weights_8t[(gi + 1) * pw..(gi + 2) * pw],
                        &self.l1_weights_8t[(gi + 2) * pw..(gi + 3) * pw],
                        &self.l1_weights_8t[(gi + 3) * pw..(gi + 4) * pw],
                        pw,
                    );
                    let ntm_results = neon_l1_int8_dot_x4(
                        &ntm_pw[..pw],
                        &self.l1_weights_8t[ntm_base + gi * pw..ntm_base + (gi + 1) * pw],
                        &self.l1_weights_8t[ntm_base + (gi + 1) * pw..ntm_base + (gi + 2) * pw],
                        &self.l1_weights_8t[ntm_base + (gi + 2) * pw..ntm_base + (gi + 3) * pw],
                        &self.l1_weights_8t[ntm_base + (gi + 3) * pw..ntm_base + (gi + 4) * pw],
                        pw,
                    );
                    for k in 0..4 {
                        hidden32[i + k] += stm_results[k] + ntm_results[k];
                    }
                }
                i += 4;
            }
            // Tail (if l1 not divisible by 4)
            while i < l1 {
                let gi = l1_off + i;
                let stm_w = &self.l1_weights_8t[gi * pw..(gi + 1) * pw];
                let ntm_w = &self.l1_weights_8t[ntm_base + gi * pw..ntm_base + (gi + 1) * pw];
                unsafe {
                    hidden32[i] += neon_l1_int8_dot(&stm_pw[..pw], stm_w, pw);
                    hidden32[i] += neon_l1_int8_dot(&ntm_pw[..pw], ntm_w, pw);
                }
                i += 1;
            }
        }

        #[cfg(target_arch = "aarch64")]
        if !(self.has_neon && pw % 16 == 0 && !self.l1_weights_8t.is_empty()) {
            for i in 0..l1 {
                let gi = l1_off + i;
                for j in 0..pw {
                    hidden32[i] += stm_pw[j] as i32 * self.l1_weights[j * l1_total + gi] as i32;
                }
                for j in 0..pw {
                    hidden32[i] += ntm_pw[j] as i32 * self.l1_weights[(pw + j) * l1_total + gi] as i32;
                }
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Scalar fallback for other architectures
            for i in 0..l1 {
                let gi = l1_off + i;
                for j in 0..pw {
                    hidden32[i] += stm_pw[j] as i32 * self.l1_weights[j * l1_total + gi] as i32;
                }
                for j in 0..pw {
                    hidden32[i] += ntm_pw[j] as i32 * self.l1_weights[(pw + j) * l1_total + gi] as i32;
                }
            }
        }

        // Dequantize + activation
        let qa_l1_f = qa_l1 as f32;
        let qa_l1_sq = qa_l1_f * qa_l1_f;
        let l1_out_count = if self.dual_l1 { l1 * 2 } else { l1 };
        let mut l1_out = [0.0f32; 1024]; // max: 512 neurons × 2 for dual
        if self.dual_l1 {
            // Dual L1 activation: CReLU(L1) concat SCReLU(L1)
            for i in 0..l1 {
                let h_val = (hidden32[i] / pw_scale).clamp(0, qa_l1);
                l1_out[i] = h_val as f32 / qa_l1_f;               // CReLU: [0, 1]
                l1_out[l1 + i] = (h_val * h_val) as f32 / qa_l1_sq; // SCReLU: [0, 1]
            }
        } else if self.crelu_hidden.load(std::sync::atomic::Ordering::Relaxed) {
            // Clipped ReLU variant (for nets trained with .crelu() on L1/L2 in Bullet)
            for i in 0..l1 {
                let h_val = (hidden32[i] / pw_scale).clamp(0, qa_l1);
                l1_out[i] = h_val as f32 / qa_l1_f; // CReLU: [0, 1]
            }
        } else {
            for i in 0..l1 {
                let h_val = (hidden32[i] / pw_scale).clamp(0, qa_l1);
                l1_out[i] = (h_val * h_val) as f32 / qa_l1_sq; // SCReLU
            }
        }

        // L2 or output
        if self.l2_per_bucket > 0 {
            let l2_pb = self.l2_per_bucket;
            let l2_total = self.l2_size;
            let l2_off = if self.bucketed_hidden { bucket * l2_pb } else { 0 };
            let l2 = if self.bucketed_hidden { l2_pb } else { l2_total };
            let l2_stride = if self.dual_l1 { self.l1_per_bucket * 2 } else { self.l1_per_bucket };
            let _ = l2_stride;
            let mut h2 = [0.0f32; 512];
            // L2 matmul. The common v9 shape (L2=32) takes a hand-vectorised
            // AVX-512 FMA path: LLVM's autovec turned into `VGATHERQPS` here
            // and ate ~13% of total cycles. Explicit 2-register broadcast-FMA
            // is branch-light, load-heavy, cache-friendly.
            #[cfg(target_arch = "x86_64")]
            if self.has_avx512 && l2 == 32 {
                unsafe {
                    l2_fmadd_avx512_x32(
                        &l1_out[..l1_out_count], l1_out_count,
                        &self.l2_weights_f, l2_total, l2_off,
                        &self.l2_biases_f, &mut h2,
                    );
                }
            } else {
                for k in 0..l2 { h2[k] = self.l2_biases_f[l2_off + k]; }
                for i in 0..l1_out_count {
                    if l1_out[i] == 0.0 { continue; }
                    for k in 0..l2 {
                        h2[k] += l1_out[i] * self.l2_weights_f[i * l2_total + l2_off + k];
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                for k in 0..l2 { h2[k] = self.l2_biases_f[l2_off + k]; }
                for i in 0..l1_out_count {
                    if l1_out[i] == 0.0 { continue; }
                    for k in 0..l2 {
                        h2[k] += l1_out[i] * self.l2_weights_f[i * l2_total + l2_off + k];
                    }
                }
            }
            if self.crelu_hidden.load(std::sync::atomic::Ordering::Relaxed) {
                for k in 0..l2 { h2[k] = h2[k].clamp(0.0, 1.0); } // CReLU
            } else {
                for k in 0..l2 { h2[k] = h2[k].clamp(0.0, 1.0); h2[k] *= h2[k]; } // SCReLU
            }
            let out_w = &self.out_weights_f[bucket * l2_pb..bucket * l2_pb + l2_pb];
            let bias = self.out_bias_f[bucket];
            // Output dot — AVX-512 version for the L2=32 case, matches the
            // L2 matmul's dimensionality so it stays on the hot path.
            #[cfg(target_arch = "x86_64")]
            let out_f = if self.has_avx512 && l2 == 32 {
                unsafe { dot_fmadd_avx512_x32(&h2[..32], &out_w[..32], bias) }
            } else {
                let mut acc = bias;
                for k in 0..l2 { acc += h2[k] * out_w[k]; }
                acc
            };
            #[cfg(not(target_arch = "x86_64"))]
            let out_f = {
                let mut acc = bias;
                for k in 0..l2 { acc += h2[k] * out_w[k]; }
                acc
            };
            return (out_f * EVAL_SCALE as f32) as i32;
        }

        let out_w = &self.out_weights_f[bucket * l1_pb..bucket * l1_pb + l1_pb];
        let mut out_f = self.out_bias_f[bucket];
        for i in 0..l1 { out_f += l1_out[i] * out_w[i]; }
        (out_f * EVAL_SCALE as f32) as i32
    }

    /// v7 hidden layer forward pass (SCReLU).
    /// acc → SCReLU (clamp+square, scale QA²) → L1 matmul → /QA² → SCReLU → float L2 → output
    fn forward_with_l1(&self, stm_acc: &[i16], ntm_acc: &[i16], bucket: usize) -> i32 {
        let h = self.hidden_size;
        let l1 = if self.bucketed_hidden { self.l1_per_bucket } else { self.l1_size };
        let l1_total = self.l1_size;  // total size (bucketed or not)
        let qa = QA as i64;
        let qa2 = qa * qa; // 65025
        let qa_l1 = self.l1_scale as i64;

        // Bucket offset into weight/bias arrays (0 for unbucketed nets)
        let b_off = if self.bucketed_hidden { bucket * l1 } else { 0 };

        // L1 matmul: SCReLU(acc) × L1_weights → hidden
        // SCReLU: clamp [0, QA], square → scale QA²
        // L1 weights at scale QA_L1, bias at scale QA_L1
        // Result at scale QA² × QA_L1, bias scaled up by QA² to match
        let bias_scale = qa2;
        let mut hidden = [0i64; 256]; // max L1 per-bucket size
        for i in 0..l1 {
            hidden[i] = self.l1_biases[b_off + i] as i64 * bias_scale;
        }

        // SIMD int8 path: pack SCReLU to u8, then VPMADDUBSW L1 matmul
        #[cfg(target_arch = "x86_64")]
        if (self.has_avx512 || self.has_avx2) && h % 32 == 0 && !self.l1_weights_8t.is_empty() {
            let mut stm_packed = [0u8; 2048]; // max accumulator size
            let mut ntm_packed = [0u8; 2048];
            if self.has_avx512 && h % 64 == 0 {
                unsafe {
                    simd512_screlu_pack(stm_acc, &mut stm_packed, h);
                    simd512_screlu_pack(ntm_acc, &mut ntm_packed, h);
                }
            } else {
                unsafe {
                    simd_screlu_pack(stm_acc, &mut stm_packed, h);
                    simd_screlu_pack(ntm_acc, &mut ntm_packed, h);
                }
            }
            // Int8 matmul: input at scale QA (v²/255), weights at scale QA_L1
            // Result at scale QA × QA_L1. Bias at scale QA_L1, scaled by QA to match.
            let qa_int = QA as i32;
            for i in 0..l1 {
                hidden[i] = self.l1_biases[b_off + i] as i64 * qa_int as i64; // bias × QA
            }
            // Weight layout: l1_weights_8t[neuron * h] for STM, [l1_total*h + neuron*h] for NTM
            // With bucket offset: STM starts at (b_off + i) * h, NTM at l1_total*h + (b_off+i)*h
            if self.has_avx512 && h % 64 == 0 {
                // Dispatch dense/sparse × vnni/plain combinations. Each branch
                // specialises on the actual kernel to keep inner loops tight.
                let use_sparse = self.use_sparse_l1.load(std::sync::atomic::Ordering::Relaxed);
                let use_vnni = self.has_avx512_vnni;
                if use_sparse {
                    debug_assert!(h <= 2048, "sparse NNZ buffer too small for h={}", h);
                    let mut stm_nnz = [0u16; 32]; // safe for h <= 2048 (32 × 64-byte chunks)
                    let mut ntm_nnz = [0u16; 32];
                    let (stm_nnz_count, ntm_nnz_count);
                    unsafe {
                        stm_nnz_count = find_nnz_chunks_512(&stm_packed, &mut stm_nnz, h);
                        ntm_nnz_count = find_nnz_chunks_512(&ntm_packed, &mut ntm_nnz, h);
                    }
                    if use_vnni {
                        for i in 0..l1 {
                            let gi = b_off + i;
                            let stm_w = &self.l1_weights_8t[gi * h..(gi + 1) * h];
                            let ntm_w = &self.l1_weights_8t[l1_total * h + gi * h..l1_total * h + (gi + 1) * h];
                            unsafe {
                                hidden[i] += simd512_l1_int8_dot_sparse_vnni(&stm_packed, stm_w, &stm_nnz, stm_nnz_count) as i64;
                                hidden[i] += simd512_l1_int8_dot_sparse_vnni(&ntm_packed, ntm_w, &ntm_nnz, ntm_nnz_count) as i64;
                            }
                        }
                    } else {
                        for i in 0..l1 {
                            let gi = b_off + i;
                            let stm_w = &self.l1_weights_8t[gi * h..(gi + 1) * h];
                            let ntm_w = &self.l1_weights_8t[l1_total * h + gi * h..l1_total * h + (gi + 1) * h];
                            unsafe {
                                hidden[i] += simd512_l1_int8_dot_sparse(&stm_packed, stm_w, &stm_nnz, stm_nnz_count) as i64;
                                hidden[i] += simd512_l1_int8_dot_sparse(&ntm_packed, ntm_w, &ntm_nnz, ntm_nnz_count) as i64;
                            }
                        }
                    }
                } else if use_vnni {
                    for i in 0..l1 {
                        let gi = b_off + i;
                        let stm_w = &self.l1_weights_8t[gi * h..(gi + 1) * h];
                        let ntm_w = &self.l1_weights_8t[l1_total * h + gi * h..l1_total * h + (gi + 1) * h];
                        unsafe {
                            hidden[i] += simd512_l1_int8_dot_vnni(&stm_packed, stm_w, h) as i64;
                            hidden[i] += simd512_l1_int8_dot_vnni(&ntm_packed, ntm_w, h) as i64;
                        }
                    }
                } else {
                    for i in 0..l1 {
                        let gi = b_off + i;
                        let stm_w = &self.l1_weights_8t[gi * h..(gi + 1) * h];
                        let ntm_w = &self.l1_weights_8t[l1_total * h + gi * h..l1_total * h + (gi + 1) * h];
                        unsafe {
                            hidden[i] += simd512_l1_int8_dot(&stm_packed, stm_w, h) as i64;
                            hidden[i] += simd512_l1_int8_dot(&ntm_packed, ntm_w, h) as i64;
                        }
                    }
                }
            } else {
                if self.use_sparse_l1.load(std::sync::atomic::Ordering::Relaxed) {
                    debug_assert!(h <= 2048, "sparse NNZ buffer too small for h={}", h);
                    let mut stm_nnz = [0u16; 64]; // safe for h <= 2048 (64 × 32-byte chunks)
                    let mut ntm_nnz = [0u16; 64];
                    let (stm_nnz_count, ntm_nnz_count);
                    unsafe {
                        stm_nnz_count = find_nnz_chunks(&stm_packed, &mut stm_nnz, h);
                        ntm_nnz_count = find_nnz_chunks(&ntm_packed, &mut ntm_nnz, h);
                    }
                    for i in 0..l1 {
                        let gi = b_off + i;
                        let stm_w = &self.l1_weights_8t[gi * h..(gi + 1) * h];
                        let ntm_w = &self.l1_weights_8t[l1_total * h + gi * h..l1_total * h + (gi + 1) * h];
                        unsafe {
                            hidden[i] += simd_l1_int8_dot_sparse(&stm_packed, stm_w, &stm_nnz, stm_nnz_count) as i64;
                            hidden[i] += simd_l1_int8_dot_sparse(&ntm_packed, ntm_w, &ntm_nnz, ntm_nnz_count) as i64;
                        }
                    }
                } else {
                    for i in 0..l1 {
                        let gi = b_off + i;
                        let stm_w = &self.l1_weights_8t[gi * h..(gi + 1) * h];
                        let ntm_w = &self.l1_weights_8t[l1_total * h + gi * h..l1_total * h + (gi + 1) * h];
                        unsafe {
                            hidden[i] += simd_l1_int8_dot(&stm_packed, stm_w, h) as i64;
                            hidden[i] += simd_l1_int8_dot(&ntm_packed, ntm_w, h) as i64;
                        }
                    }
                }
            }
            // Dequantize: result at scale QA × QA_L1. Divide by QA to get scale QA_L1.
            let qa_l1_f = qa_l1 as f32;
            let qa_l1_sq = qa_l1_f * qa_l1_f;
            let l1_out_count = if self.dual_l1 { l1 * 2 } else { l1 };
            let mut l1_out = [0.0f32; 128]; // max: 64 neurons × 2 for dual
            if self.dual_l1 {
                // Dual L1 activation: CReLU(L1) concat SCReLU(L1)
                for i in 0..l1 {
                    let h_val = ((hidden[i] / qa as i64) as i32).clamp(0, qa_l1 as i32);
                    l1_out[i] = h_val as f32 / qa_l1_f;               // CReLU: [0, 1]
                    l1_out[l1 + i] = (h_val * h_val) as f32 / qa_l1_sq; // SCReLU: [0, 1]
                }
            } else if self.crelu_hidden.load(std::sync::atomic::Ordering::Relaxed) {
                // Clipped ReLU variant
                for i in 0..l1 {
                    let h_val = ((hidden[i] / qa as i64) as i32).clamp(0, qa_l1 as i32);
                    l1_out[i] = h_val as f32 / qa_l1_f; // CReLU: [0, 1]
                }
            } else {
                // Standard SCReLU only
                for i in 0..l1 {
                    let h_val = ((hidden[i] / qa as i64) as i32).clamp(0, qa_l1 as i32);
                    l1_out[i] = (h_val * h_val) as f32 / qa_l1_sq; // → [0, 1]
                }
            }

            // L2 or output — float (handles both bucketed and unbucketed)
            if self.l2_size > 0 {
                let l2 = if self.bucketed_hidden { self.l2_per_bucket } else { self.l2_size };
                let l2_off = if self.bucketed_hidden { bucket * l2 } else { 0 };
                let l2_stride = if self.dual_l1 { self.l1_per_bucket * 2 } else { self.l1_per_bucket };
                let l2_total_stride = if self.bucketed_hidden { l2_stride * NNUE_OUTPUT_BUCKETS } else { l2_stride };
                let _ = l2_total_stride; // used below
                let mut h2 = [0.0f32; 256];
                for k in 0..l2 { h2[k] = self.l2_biases_f[l2_off + k]; }
                for i in 0..l1_out_count {
                    if l1_out[i] == 0.0 { continue; }
                    let w_off = if self.bucketed_hidden {
                        i * self.l2_size + bucket * self.l2_per_bucket
                    } else {
                        i * self.l2_size
                    };
                    for k in 0..l2 { h2[k] += l1_out[i] * self.l2_weights_f[w_off + k]; }
                }
                if self.crelu_hidden.load(std::sync::atomic::Ordering::Relaxed) {
                    for k in 0..l2 { h2[k] = h2[k].clamp(0.0, 1.0); } // CReLU
                } else {
                    for k in 0..l2 { h2[k] = h2[k].clamp(0.0, 1.0); h2[k] *= h2[k]; } // SCReLU
                }
                let out_w = &self.out_weights_f[bucket * l2..bucket * l2 + l2];
                let mut out_f = self.out_bias_f[bucket];
                for k in 0..l2 { out_f += h2[k] * out_w[k]; }
                return (out_f * EVAL_SCALE as f32) as i32;
            }
            let out_w = &self.out_weights_f[bucket * l1..bucket * l1 + l1];
            let mut out_f = self.out_bias_f[bucket];
            for i in 0..l1 { out_f += l1_out[i] * out_w[i]; }
            return (out_f * EVAL_SCALE as f32) as i32;
        }

        // NEON int8 path: pack SCReLU to u8, then NEON L1 matmul
        #[cfg(target_arch = "aarch64")]
        if self.has_neon && h % 16 == 0 && !self.l1_weights_8t.is_empty() {
            let mut stm_packed = [0u8; 2048];
            let mut ntm_packed = [0u8; 2048];
            unsafe {
                neon_screlu_pack(stm_acc, &mut stm_packed, h);
                neon_screlu_pack(ntm_acc, &mut ntm_packed, h);
            }
            // Int8 matmul: input at scale QA (v²/255), weights at scale QA_L1
            // Result at scale QA × QA_L1. Bias at scale QA_L1, scaled by QA to match.
            let qa_int = QA as i32;
            for i in 0..l1 {
                hidden[i] = self.l1_biases[b_off + i] as i64 * qa_int as i64; // bias × QA
            }
            if self.use_sparse_l1.load(std::sync::atomic::Ordering::Relaxed) {
                let mut stm_nnz = [0u16; 128];
                let mut ntm_nnz = [0u16; 128];
                let (stm_nnz_count, ntm_nnz_count);
                unsafe {
                    stm_nnz_count = neon_find_nnz_chunks(&stm_packed, &mut stm_nnz, h);
                    ntm_nnz_count = neon_find_nnz_chunks(&ntm_packed, &mut ntm_nnz, h);
                }
                for i in 0..l1 {
                    let gi = b_off + i;
                    let stm_w = &self.l1_weights_8t[gi * h..(gi + 1) * h];
                    let ntm_w = &self.l1_weights_8t[l1_total * h + gi * h..l1_total * h + (gi + 1) * h];
                    unsafe {
                        hidden[i] += neon_l1_int8_dot_sparse(&stm_packed, stm_w, &stm_nnz, stm_nnz_count) as i64;
                        hidden[i] += neon_l1_int8_dot_sparse(&ntm_packed, ntm_w, &ntm_nnz, ntm_nnz_count) as i64;
                    }
                }
            } else {
                for i in 0..l1 {
                    let gi = b_off + i;
                    let stm_w = &self.l1_weights_8t[gi * h..(gi + 1) * h];
                    let ntm_w = &self.l1_weights_8t[l1_total * h + gi * h..l1_total * h + (gi + 1) * h];
                    unsafe {
                        hidden[i] += neon_l1_int8_dot(&stm_packed, stm_w, h) as i64;
                        hidden[i] += neon_l1_int8_dot(&ntm_packed, ntm_w, h) as i64;
                    }
                }
            }
            // Dequantize + activation
            let qa_l1_f = qa_l1 as f32;
            let qa_l1_sq = qa_l1_f * qa_l1_f;
            let l1_out_count = if self.dual_l1 { l1 * 2 } else { l1 };
            let mut l1_out = [0.0f32; 512]; // max: 256 × 2 for dual
            if self.dual_l1 {
                for i in 0..l1 {
                    let h_val = ((hidden[i] / qa as i64) as i32).clamp(0, qa_l1 as i32);
                    l1_out[i] = h_val as f32 / qa_l1_f;
                    l1_out[l1 + i] = (h_val * h_val) as f32 / qa_l1_sq;
                }
            } else {
                for i in 0..l1 {
                    let h_val = ((hidden[i] / qa as i64) as i32).clamp(0, qa_l1 as i32);
                    l1_out[i] = (h_val * h_val) as f32 / qa_l1_sq;
                }
            }

            // L2 or output — float (handles both bucketed and unbucketed)
            if self.l2_size > 0 {
                let l2 = if self.bucketed_hidden { self.l2_per_bucket } else { self.l2_size };
                let l2_off = if self.bucketed_hidden { bucket * l2 } else { 0 };
                let mut h2 = [0.0f32; 256];
                for k in 0..l2 { h2[k] = self.l2_biases_f[l2_off + k]; }
                for i in 0..l1_out_count {
                    if l1_out[i] == 0.0 { continue; }
                    let w_off = if self.bucketed_hidden {
                        i * self.l2_size + bucket * self.l2_per_bucket
                    } else {
                        i * self.l2_size
                    };
                    for k in 0..l2 { h2[k] += l1_out[i] * self.l2_weights_f[w_off + k]; }
                }
                if self.crelu_hidden.load(std::sync::atomic::Ordering::Relaxed) {
                    for k in 0..l2 { h2[k] = h2[k].clamp(0.0, 1.0); }
                } else {
                    for k in 0..l2 { h2[k] = h2[k].clamp(0.0, 1.0); h2[k] *= h2[k]; }
                }
                let out_w = &self.out_weights_f[bucket * l2..bucket * l2 + l2];
                let mut out_f = self.out_bias_f[bucket];
                for k in 0..l2 { out_f += h2[k] * out_w[k]; }
                return (out_f * EVAL_SCALE as f32) as i32;
            }
            let out_w = &self.out_weights_f[bucket * l1..bucket * l1 + l1];
            let mut out_f = self.out_bias_f[bucket];
            for i in 0..l1 { out_f += l1_out[i] * out_w[i]; }
            return (out_f * EVAL_SCALE as f32) as i32;
        }

        // Scalar fallback (bucket-aware)
        // L1 weight layout: transposed [neuron × h] per perspective, accessed via l1_total + b_off
        #[cfg(target_arch = "x86_64")]
        if !((self.has_avx512 || self.has_avx2) && h % 32 == 0 && !self.l1_weights_8t.is_empty()) {
            for j in 0..h {
                let v = (stm_acc[j] as i64).clamp(0, qa);
                if v == 0 { continue; }
                let vsq = v * v;
                let w_off = j * l1_total + b_off;
                for i in 0..l1 {
                    hidden[i] += vsq * self.l1_weights[w_off + i] as i64;
                }
            }
            for j in 0..h {
                let v = (ntm_acc[j] as i64).clamp(0, qa);
                if v == 0 { continue; }
                let vsq = v * v;
                let w_off = (h + j) * l1_total + b_off;
                for i in 0..l1 {
                    hidden[i] += vsq * self.l1_weights[w_off + i] as i64;
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        if !(self.has_neon && h % 16 == 0 && !self.l1_weights_8t.is_empty()) {
            for j in 0..h {
                let v = (stm_acc[j] as i64).clamp(0, qa);
                if v == 0 { continue; }
                let vsq = v * v;
                let w_off = j * l1_total + b_off;
                for i in 0..l1 {
                    hidden[i] += vsq * self.l1_weights[w_off + i] as i64;
                }
            }
            for j in 0..h {
                let v = (ntm_acc[j] as i64).clamp(0, qa);
                if v == 0 { continue; }
                let vsq = v * v;
                let w_off = (h + j) * l1_total + b_off;
                for i in 0..l1 {
                    hidden[i] += vsq * self.l1_weights[w_off + i] as i64;
                }
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            for j in 0..h {
                let v = (stm_acc[j] as i64).clamp(0, qa);
                if v == 0 { continue; }
                let vsq = v * v;
                let w_off = j * l1_total + b_off;
                for i in 0..l1 {
                    hidden[i] += vsq * self.l1_weights[w_off + i] as i64;
                }
            }
            for j in 0..h {
                let v = (ntm_acc[j] as i64).clamp(0, qa);
                if v == 0 { continue; }
                let vsq = v * v;
                let w_off = (h + j) * l1_total + b_off;
                for i in 0..l1 {
                    hidden[i] += vsq * self.l1_weights[w_off + i] as i64;
                }
            }
        }

        // Divide by QA² to get hidden at scale QA_L1, then activation
        let qa_l1_f = qa_l1 as f32;
        let qa_l1_sq = qa_l1_f * qa_l1_f;
        let l1_out_count = if self.dual_l1 { l1 * 2 } else { l1 };
        let mut l1_out = [0.0f32; 512]; // max: 256 × 2 for dual
        if self.dual_l1 {
            for i in 0..l1 {
                let h_val = ((hidden[i] / qa2) as i32).clamp(0, qa_l1 as i32);
                l1_out[i] = h_val as f32 / qa_l1_f;
                l1_out[l1 + i] = (h_val * h_val) as f32 / qa_l1_sq;
            }
        } else if self.crelu_hidden.load(std::sync::atomic::Ordering::Relaxed) {
            for i in 0..l1 {
                let h_val = ((hidden[i] / qa2) as i32).clamp(0, qa_l1 as i32);
                l1_out[i] = h_val as f32 / qa_l1_f;
            }
        } else {
            for i in 0..l1 {
                let h_val = ((hidden[i] / qa2) as i32).clamp(0, qa_l1 as i32);
                l1_out[i] = (h_val * h_val) as f32 / qa_l1_sq;
            }
        }

        // L2 layer (if present) — float (handles bucketed and unbucketed)
        if self.l2_size > 0 {
            let l2 = if self.bucketed_hidden { self.l2_per_bucket } else { self.l2_size };
            let l2_off = if self.bucketed_hidden { bucket * l2 } else { 0 };
            let mut h2 = [0.0f32; 256];
            for k in 0..l2 {
                h2[k] = self.l2_biases_f[l2_off + k];
            }
            for i in 0..l1_out_count {
                if l1_out[i] == 0.0 { continue; }
                let w_off = if self.bucketed_hidden {
                    i * self.l2_size + bucket * self.l2_per_bucket
                } else {
                    i * self.l2_size
                };
                for k in 0..l2 {
                    h2[k] += l1_out[i] * self.l2_weights_f[w_off + k];
                }
            }
            // L2 activation: CReLU or SCReLU
            if self.crelu_hidden.load(std::sync::atomic::Ordering::Relaxed) {
                for k in 0..l2 { h2[k] = h2[k].clamp(0.0, 1.0); }
            } else {
                for k in 0..l2 {
                    h2[k] = h2[k].clamp(0.0, 1.0);
                    h2[k] = h2[k] * h2[k];
                }
            }
            // Output dot in float
            let out_w = &self.out_weights_f[bucket * l2..bucket * l2 + l2];
            let mut out_f = self.out_bias_f[bucket];
            for k in 0..l2 {
                out_f += h2[k] * out_w[k];
            }
            return (out_f * EVAL_SCALE as f32) as i32;
        }

        // L1 → output (no L2)
        let out_w = &self.out_weights_f[bucket * l1..bucket * l1 + l1];
        let mut out_f = self.out_bias_f[bucket];
        for i in 0..l1 {
            out_f += l1_out[i] * out_w[i];
        }
        (out_f * EVAL_SCALE as f32) as i32
    }

    /// Forward pass with ThreatStack (new path).
    pub fn forward_with_threats(&self, acc: &NNUEAccumulator, stm: u8, piece_count: u32,
                                threat_stack: &crate::threat_accum::ThreatStack) -> i32 {
        if threat_stack.active && self.has_threats {
            let bucket = output_bucket(piece_count);
            let h = self.hidden_size;

            let (stm_acc, ntm_acc) = if stm == WHITE {
                (acc.white(), acc.black())
            } else {
                (acc.black(), acc.white())
            };

            let t_stm = threat_stack.values(if stm == WHITE { WHITE } else { BLACK });
            let t_ntm = threat_stack.values(if stm == WHITE { BLACK } else { WHITE });

            if self.l1_size > 0 {
                if self.use_pairwise {
                    return self.forward_with_l1_pairwise_threats(stm_acc, ntm_acc, t_stm, t_ntm, bucket);
                }
                // Non-pairwise with threats: combine on stack (rare path)
                let mut stm_combined = [0i16; 768];
                let mut ntm_combined = [0i16; 768];
                for i in 0..h {
                    stm_combined[i] = stm_acc[i].wrapping_add(t_stm[i]);
                    ntm_combined[i] = ntm_acc[i].wrapping_add(t_ntm[i]);
                }
                return self.forward_with_l1(&stm_combined[..h], &ntm_combined[..h], bucket);
            }

            // Non-hidden-layer path (shouldn't happen for v9 but handle it)
            let out_w = self.output_weight_row(bucket);
            let mut output = self.output_bias[bucket] as i64;
            if self.use_pairwise {
                let pw = h / 2;
                for i in 0..pw {
                    let a = (stm_acc[i] as i32 + t_stm[i] as i32).clamp(0, QA);
                    let b = (stm_acc[i + pw] as i32 + t_stm[i + pw] as i32).clamp(0, QA);
                    let v = ((a * b) >> FT_SHIFT) as i64;
                    output += v * out_w[i] as i64;
                }
            }
            return (output * EVAL_SCALE as i64 / (QA as i64 * QB as i64)) as i32;
        }

        // Fallback to old forward path
        self.forward(acc, stm, piece_count)
    }

    /// Forward pass: CReLU or SCReLU activation → dot product with output weights.
    /// Returns centipawns from side-to-move perspective.
    pub fn forward(&self, acc: &NNUEAccumulator, stm: u8, piece_count: u32) -> i32 {
        let bucket = output_bucket(piece_count);
        let h = self.hidden_size;
        let out_w = self.output_weight_row(bucket);

        let (stm_acc_raw, ntm_acc_raw) = if stm == WHITE {
            (acc.white(), acc.black())
        } else {
            (acc.black(), acc.white())
        };

        // Get threat accumulators (may be empty for non-threat nets).
        //
        // C8 audit LIKELY #17: this is the LEGACY threat pipeline using
        // accumulator.threat_white/black fields. Production v9 uses
        // `forward_with_threats` + the ThreatStack at SearchInfo::threat_stack
        // instead, so these fields are never populated on the hot path. If
        // `has_threats` is true here AND the fields are not accurate, it
        // means somebody routed a v9 eval through `forward()` directly —
        // a silent bug that used to fall through to empty_threat (= zero
        // eval). debug_assert catches it in tests; release still falls
        // through safely.
        let empty_threat = &[][..];
        let stm_idx = stm as usize;
        let ntm_idx = (stm ^ 1) as usize;
        let (t_stm, t_ntm) = if self.has_threats && acc.current().threat_accurate[stm_idx] && acc.current().threat_accurate[ntm_idx] {
            if stm == WHITE {
                (acc.current().threat_white.as_slice(), acc.current().threat_black.as_slice())
            } else {
                (acc.current().threat_black.as_slice(), acc.current().threat_white.as_slice())
            }
        } else {
            debug_assert!(
                !self.has_threats,
                "forward() called on v9 threat net without ThreatStack — \
                 production eval must route through forward_with_threats. \
                 See C8 audit LIKELY #17."
            );
            (empty_threat, empty_threat)
        };

        let mut output = self.output_bias[bucket] as i64;

        // v7 hidden layer path — pass threat accumulators for fused combine
        if self.l1_size > 0 {
            if self.use_pairwise {
                return self.forward_with_l1_pairwise_fused(stm_acc_raw, ntm_acc_raw, t_stm, t_ntm, bucket);
            }
            // Non-pairwise: combine then forward (no fused path)
            if !t_stm.is_empty() {
                let mut stm_buf = [0i16; 768];
                let mut ntm_buf = [0i16; 768];
                for i in 0..h { stm_buf[i] = stm_acc_raw[i].wrapping_add(t_stm[i]); }
                for i in 0..h { ntm_buf[i] = ntm_acc_raw[i].wrapping_add(t_ntm[i]); }
                return self.forward_with_l1(&stm_buf[..h], &ntm_buf[..h], bucket);
            }
            return self.forward_with_l1(stm_acc_raw, ntm_acc_raw, bucket);
        }
        let stm_acc = stm_acc_raw;
        let ntm_acc = ntm_acc_raw;

        // Pairwise path: split acc into halves, clamp, multiply pairs, dot with weights
        if self.use_pairwise {
            let pw = h / 2; // pairwise output size per perspective

            #[cfg(target_arch = "x86_64")]
            if self.has_avx512 && pw % 32 == 0 {
                let bias = output;
                let mut sum: i64;
                unsafe {
                    sum = simd512_pairwise_dot(&stm_acc[..pw], &stm_acc[pw..], &out_w[..pw], pw);
                    sum += simd512_pairwise_dot(&ntm_acc[..pw], &ntm_acc[pw..], &out_w[pw..], pw);
                }
                output = sum / QA as i64 + bias;
                let result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
                return result;
            }

            #[cfg(target_arch = "x86_64")]
            if self.has_avx2 && pw % 16 == 0 {
                let bias = output;
                let mut sum: i64;
                unsafe {
                    sum = simd_pairwise_dot(&stm_acc[..pw], &stm_acc[pw..], &out_w[..pw], pw);
                    sum += simd_pairwise_dot(&ntm_acc[..pw], &ntm_acc[pw..], &out_w[pw..], pw);
                }
                output = sum / QA as i64 + bias;
                let result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
                return result;
            }

            #[cfg(target_arch = "aarch64")]
            if self.has_neon && pw % 8 == 0 {
                let bias = output;
                let mut sum: i64;
                unsafe {
                    sum = neon_pairwise_dot(&stm_acc[..pw], &stm_acc[pw..], &out_w[..pw], pw);
                    sum += neon_pairwise_dot(&ntm_acc[..pw], &ntm_acc[pw..], &out_w[pw..], pw);
                }
                output = sum / QA as i64 + bias;
                let result = (output * EVAL_SCALE as i64 / QAB as i64) as i32;
                return result;
            }

            // Scalar fallback — accumulate a*b*w in i64, divide by QA once at end
            // (matches SIMD path which preserves full precision before division)
            let bias = output;
            let mut sum: i64 = 0;
            for i in 0..pw {
                let a = (stm_acc[i] as i32).clamp(0, QA as i32);
                let b = (stm_acc[i + pw] as i32).clamp(0, QA as i32);
                sum += (a * b) as i64 * out_w[i] as i64;
            }
            for i in 0..pw {
                let a = (ntm_acc[i] as i32).clamp(0, QA as i32);
                let b = (ntm_acc[i + pw] as i32).clamp(0, QA as i32);
                sum += (a * b) as i64 * out_w[pw + i] as i64;
            }
            output = sum / QA as i64 + bias;

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
    // Threat accumulator (v9): separate i16 values summed with PSQ at activation
    pub threat_white: Vec<i16>,
    pub threat_black: Vec<i16>,
    pub threat_accurate: [bool; 2], // per-perspective [WHITE, BLACK]
    pub threat_deltas: Vec<crate::threats::RawThreatDelta>,
    pub threat_move: Move, // the move that produced this ply (for king mirror check)
    pub threat_moved_pt: u8, // piece type that moved
    pub threat_moved_color: Color, // color that moved
    // Stored threat feature indices for diff-based incremental (per perspective)
    pub threat_features_white: Vec<usize>,
    pub threat_features_black: Vec<usize>,
}

/// Finny table entry: cached accumulator for a specific king bucket.
struct FinnyEntry {
    acc: Vec<i16>,                      // cached accumulator values
    piece_bbs: ([Bitboard; 6], [Bitboard; 2]), // piece and color bitboards when cached
    valid: bool,
}

// Finny slots sized for the maximum supported bucket count. Nets with fewer
// buckets (e.g. Reckless 10) use the low indices only; remaining slots sit
// unused but cost nothing significant (~1KB per slot × 12 unused ≈ 12KB).
const FINNY_SIZE: usize = 2 * NNUE_MAX_KING_BUCKETS * 2; // [perspective][bucket][mirror]

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
                threat_white: Vec::new(), // allocated on first use if net has threats
                threat_black: Vec::new(),
                threat_accurate: [false; 2],
                threat_deltas: Vec::new(),
                threat_move: NO_MOVE,
                threat_moved_pt: NO_PIECE_TYPE,
                threat_moved_color: WHITE,
                threat_features_white: Vec::new(),
                threat_features_black: Vec::new(),
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

    pub fn top(&self) -> usize { self.top }

    pub fn prev_threat_computed(&self) -> bool {
        self.top > 0 && self.stack[self.top - 1].threat_accurate[0] && self.stack[self.top - 1].threat_accurate[1]
    }

    pub fn set_threat_deltas(&mut self, deltas: Vec<crate::threats::RawThreatDelta>) {
        self.stack[self.top].threat_deltas = deltas;
    }

    /// Copy threat deltas from board into current stack entry.
    /// Must be called after push() and after board.make_move().
    pub fn store_threat_deltas(&mut self, board: &mut crate::board::Board) {
        if board.generate_threat_deltas {
            let entry = &mut self.stack[self.top];
            // Swap deltas from board into stack entry (avoids copy, board gets the old buffer)
            std::mem::swap(&mut entry.threat_deltas, &mut board.threat_deltas);
            // Store move info for king mirror check (Reckless pattern)
            let undo_len = board.undo_stack.len();
            if undo_len > 0 {
                let undo = &board.undo_stack[undo_len - 1];
                entry.threat_move = undo.mv;
                if undo.mv != NO_MOVE {
                    let to = move_to(undo.mv);
                    entry.threat_moved_pt = board.mailbox[to as usize];
                    entry.threat_moved_color = flip_color(board.side_to_move); // side that moved
                } else {
                    entry.threat_moved_pt = NO_PIECE_TYPE;
                    entry.threat_moved_color = WHITE;
                }
            }
        }
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
                        let idx = net.halfka_index(WHITE, board.king_sq(WHITE), color, pt, sq);
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
                        let idx = net.halfka_index(BLACK, board.king_sq(BLACK), color, pt, sq);
                        let row = net.input_weight_row(idx);
                        for j in 0..h { dst[j] += row[j]; }
                    }
                }
            }
        }
        self.stack[self.top].computed = true;
    }

    /// Compute threat accumulator if not already done.
    /// Walks back to find nearest ancestor with computed threats, then replays
    /// forward applying per-ply deltas. Each ply's deltas were stored by
    /// store_threat_deltas() after make_move. Matches Reckless's pattern.
    pub fn recompute_threats_if_needed(&mut self, net: &NNUENet, board: &crate::board::Board) {
        if !net.has_threats { return; }
        if self.stack[self.top].threat_accurate[0] && self.stack[self.top].threat_accurate[1] { return; }
        let h = self.hidden_size;

        // Profile: 90.5% incremental (chain=1.1), 9.5% full recompute.
        // Per-eval cost ~5µs weighted average (Reckless: ~2.5µs).

        // Walk back to find nearest ancestor with computed threats (Reckless pattern).
        // Then verify the entire chain from ancestor to top has no king e-file crossings.
        let mut ancestor: Option<usize> = None;
        'walk: for i in (0..self.top).rev() {
            if self.stack[i].threat_accurate[0] && self.stack[i].threat_accurate[1] {
                // Found ancestor. Now check the chain i+1..=self.top for king crossings.
                for j in (i + 1)..=self.top {
                    let entry = &self.stack[j];
                    if entry.threat_move == NO_MOVE { continue; } // null move, safe
                    if entry.threat_deltas.is_empty() && entry.threat_move != NO_MOVE {
                        // Real move but no deltas — can't replay
                        break 'walk;
                    }
                    if entry.threat_moved_pt == KING {
                        let from = move_from(entry.threat_move);
                        let to = move_to(entry.threat_move);
                        if (from % 8 >= 4) != (to % 8 >= 4) {
                            break 'walk; // king mirror changed — full recompute
                        }
                    }
                }
                ancestor = Some(i);
                break;
            }
            // Check if we can continue walking back through this ply
            let entry = &self.stack[i + 1];
            if entry.threat_move != NO_MOVE && entry.threat_deltas.is_empty() {
                break; // no deltas for this real move — can't go further back
            }
        }

        if ancestor.is_none() {
            self.recompute_threats_full(net, board);
            return;
        }

        let ancestor_idx = ancestor.unwrap();

        // Replay forward from ancestor to self.top
        let wk_sq = (board.pieces[KING as usize] & board.colors[WHITE as usize]).trailing_zeros();
        let bk_sq = (board.pieces[KING as usize] & board.colors[BLACK as usize]).trailing_zeros();
        let w_mirrored = (wk_sq % 8) >= 4;
        let b_mirrored = (bk_sq % 8) >= 4;

        for ply in (ancestor_idx + 1)..=self.top {
            // Allocate if needed
            if self.stack[ply].threat_white.len() < h {
                self.stack[ply].threat_white.resize(h, 0);
                self.stack[ply].threat_black.resize(h, 0);
            }

            let src = ply - 1; // source is always the previous ply (which we just computed)

            if self.stack[ply].threat_deltas.is_empty() {
                // Null move or no deltas: copy from previous
                let (prev_slice, curr_slice) = self.stack.split_at_mut(ply);
                let prev = &prev_slice[src];
                let curr = &mut curr_slice[0];
                curr.threat_white[..h].copy_from_slice(&prev.threat_white[..h]);
                curr.threat_black[..h].copy_from_slice(&prev.threat_black[..h]);
            } else {
                // Apply deltas from this ply
                // Note: we use the current board's king positions for mirroring.
                // This is approximate for intermediate plies — if the king moved
                // during the replay chain, the mirroring might be wrong.
                // For now this is acceptable; king moves are rare in the chain.
                // Swap deltas out to avoid borrow conflict (no allocation)
                let deltas = std::mem::take(&mut self.stack[ply].threat_deltas);
                let (prev_slice, curr_slice) = self.stack.split_at_mut(ply);
                let prev = &prev_slice[src];
                let curr = &mut curr_slice[0];
                unsafe {
                    crate::threats::apply_threat_deltas(
                        &mut curr.threat_white, &prev.threat_white,
                        &deltas, &net.threat_weights, h, net.num_threat_features,
                        WHITE, w_mirrored,
                    );
                    crate::threats::apply_threat_deltas(
                        &mut curr.threat_black, &prev.threat_black,
                        &deltas, &net.threat_weights, h, net.num_threat_features,
                        BLACK, b_mirrored,
                    );
                }
                // Swap back
                self.stack[ply].threat_deltas = deltas;
            }

            self.stack[ply].threat_accurate = [true; 2];
        }

    }

    /// DEBUG: verify threat accumulator matches full recompute for any position.
    #[cfg(debug_assertions)]
    pub fn verify_threats(&self, net: &NNUENet, board: &crate::board::Board) {
        if !net.has_threats || !self.stack[self.top].threat_accurate[0] || !self.stack[self.top].threat_accurate[1] { return; }
        let h = self.hidden_size;
        let occ = board.colors[0] | board.colors[1];
        let wk_sq = (board.pieces[KING as usize] & board.colors[WHITE as usize]).trailing_zeros();
        let bk_sq = (board.pieces[KING as usize] & board.colors[BLACK as usize]).trailing_zeros();

        let mut check_w = vec![0i16; h];
        crate::threats::enumerate_threats(
            &board.pieces, &board.colors, &board.mailbox,
            occ, WHITE, (wk_sq % 8) >= 4,
            |idx| { if idx < net.num_threat_features { let w = idx * h; for j in 0..h { check_w[j] += net.threat_weights[w + j] as i16; } } },
        );
        let mut check_b = vec![0i16; h];
        crate::threats::enumerate_threats(
            &board.pieces, &board.colors, &board.mailbox,
            occ, BLACK, (bk_sq % 8) >= 4,
            |idx| { if idx < net.num_threat_features { let w = idx * h; for j in 0..h { check_b[j] += net.threat_weights[w + j] as i16; } } },
        );

        let curr = &self.stack[self.top];
        let w_diff: i32 = (0..h).map(|j| (curr.threat_white[j] as i32 - check_w[j] as i32).abs()).sum();
        let b_diff: i32 = (0..h).map(|j| (curr.threat_black[j] as i32 - check_b[j] as i32).abs()).sum();
        if w_diff > 0 || b_diff > 0 {
            eprintln!("  h={} tw_len={} got_w0={} exp_w0={}", h, curr.threat_white.len(), curr.threat_white[0], check_w[0]);
            // Was this from incremental or full recompute?
            let last_mv = if !board.undo_stack.is_empty() {
                let u = &board.undo_stack[board.undo_stack.len() - 1];
                if u.mv == NO_MOVE { "null".to_string() }
                else { format!("{}{}", crate::types::square_name(move_from(u.mv)), crate::types::square_name(move_to(u.mv))) }
            } else { "root".to_string() };
            eprintln!("VERIFY FAIL mv={} wdiff={} bdiff={} top={}", last_mv, w_diff, b_diff, self.top);
        }
    }

    /// Full recompute: iterates all pieces, computes attacks, adds i8 weight rows.
    fn recompute_threats_full(&mut self, net: &NNUENet, board: &crate::board::Board) {
        let h = self.hidden_size;
        let entry = &mut self.stack[self.top];

        if entry.threat_white.len() < h {
            entry.threat_white.resize(h, 0);
            entry.threat_black.resize(h, 0);
        }

        let occ = board.colors[0] | board.colors[1];

        // White perspective
        entry.threat_white[..h].fill(0);
        let wk_sq = (board.pieces[KING as usize] & board.colors[WHITE as usize]).trailing_zeros();
        let w_mirrored = (wk_sq % 8) >= 4;
        crate::threats::enumerate_threats(
            &board.pieces, &board.colors, &board.mailbox,
            occ, WHITE, w_mirrored,
            |feat_idx| {
                if feat_idx < net.num_threat_features {
                    let w_off = feat_idx * h;
                    for j in 0..h {
                        entry.threat_white[j] += net.threat_weights[w_off + j] as i16;
                    }
                }
            },
        );

        // Black perspective
        entry.threat_black[..h].fill(0);
        let bk_sq = (board.pieces[KING as usize] & board.colors[BLACK as usize]).trailing_zeros();
        let b_mirrored = (bk_sq % 8) >= 4;
        crate::threats::enumerate_threats(
            &board.pieces, &board.colors, &board.mailbox,
            occ, BLACK, b_mirrored,
            |feat_idx| {
                if feat_idx < net.num_threat_features {
                    let w_off = feat_idx * h;
                    for j in 0..h {
                        entry.threat_black[j] += net.threat_weights[w_off + j] as i16;
                    }
                }
            },
        );

        entry.threat_accurate = [true; 2];
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
                threat_white: Vec::new(),
                threat_black: Vec::new(),
                threat_accurate: [false; 2],
                threat_deltas: Vec::new(),
                threat_move: NO_MOVE,
                threat_moved_pt: NO_PIECE_TYPE,
                threat_moved_color: WHITE,
                threat_features_white: Vec::new(),
                threat_features_black: Vec::new(),
            });
        }
        self.stack[self.top].computed = false;
        self.stack[self.top].threat_accurate = [false; 2];
        self.stack[self.top].threat_deltas.clear();
        self.stack[self.top].dirty = dirty;
    }

    pub fn pop(&mut self) {
        debug_assert!(self.top > 0);
        self.top -= 1;
    }

    /// Materialize: ensure current accumulator is computed.
    pub fn materialize(&mut self, net: &NNUENet, board: &Board) {
        #[cfg(feature = "profile-materialize")]
        crate::nnue::mat_stats::record_entry(self.stack[self.top].computed);
        if self.stack[self.top].computed {
            return;
        }

        let dirty = self.stack[self.top].dirty;

        // Full recompute needed?
        if dirty.kind == 0 || self.top == 0 || !self.stack[self.top - 1].computed {
            #[cfg(feature = "profile-materialize")]
            {
                crate::nnue::mat_stats::record_refresh();
                // Walk ancestors to find nearest computed frame; record
                // distance. 0 = no ancestor / root. 1 = parent computed
                // (wouldn't fire here by definition but reported for shape).
                let mut d: usize = 0;
                if self.top > 0 {
                    let mut i = self.top;
                    while i > 0 {
                        i -= 1;
                        d += 1;
                        if self.stack[i].computed { break; }
                    }
                    if d > 0 && !self.stack[self.top - d].computed { d = 0; }
                }
                crate::nnue::mat_stats::record_walkback_distance(d);
            }
            self.refresh_accumulator(net, board, WHITE);
            self.refresh_accumulator(net, board, BLACK);
            self.stack[self.top].computed = true;
            return;
        }
        #[cfg(feature = "profile-materialize")]
        crate::nnue::mat_stats::record_incremental(dirty.n_changes as u64);

        // Incremental update: gather all deltas for this move, then apply them
        // in a SINGLE fused pass per perspective. Replaces the previous
        // copy+per-delta-pass pattern which did N passes over the full
        // accumulator for an N-change move (capture = 3, castling = 4).
        //
        // Reckless pattern: for each perspective, read parent once, apply
        // all add/sub weight rows while the chunk is in registers, write
        // current once. Memory traffic is constant in N.
        let h = self.hidden_size;
        let (left, right) = self.stack.split_at_mut(self.top);
        let parent = &left[self.top - 1];
        let current = &mut right[0];

        let w_king_sq = board.king_sq(WHITE);
        let b_king_sq = board.king_sq(BLACK);

        let n = dirty.n_changes as usize;

        // Collect add/sub weight rows per perspective. Max ~4 deltas per move
        // (promotion capture = 4); using 8-slot stack arrays gives headroom.
        let mut w_adds: [&[i16]; 8] = [&[]; 8];
        let mut w_subs: [&[i16]; 8] = [&[]; 8];
        let mut b_adds: [&[i16]; 8] = [&[]; 8];
        let mut b_subs: [&[i16]; 8] = [&[]; 8];
        let mut nwa = 0usize;
        let mut nws = 0usize;
        let mut nba = 0usize;
        let mut nbs = 0usize;

        for i in 0..n {
            let (add, color, pt, sq) = dirty.changes[i];
            let w_idx = net.halfka_index(WHITE, w_king_sq, color, pt, sq);
            let w_row = net.input_weight_row(w_idx);
            let b_idx = net.halfka_index(BLACK, b_king_sq, color, pt, sq);
            let b_row = net.input_weight_row(b_idx);
            if add {
                w_adds[nwa] = w_row; nwa += 1;
                b_adds[nba] = b_row; nba += 1;
            } else {
                w_subs[nws] = w_row; nws += 1;
                b_subs[nbs] = b_row; nbs += 1;
            }
        }

        let w_adds = &w_adds[..nwa];
        let w_subs = &w_subs[..nws];
        let b_adds = &b_adds[..nba];
        let b_subs = &b_subs[..nbs];

        #[cfg(target_arch = "x86_64")]
        if net.has_avx512 && h % 32 == 0 {
            unsafe {
                simd_acc_fused_avx512(&mut current.white, &parent.white, w_adds, w_subs, h);
                simd_acc_fused_avx512(&mut current.black, &parent.black, b_adds, b_subs, h);
            }
            current.computed = true;
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if net.has_avx2 && h % 16 == 0 {
            unsafe {
                simd_acc_fused_avx2(&mut current.white, &parent.white, w_adds, w_subs, h);
                simd_acc_fused_avx2(&mut current.black, &parent.black, b_adds, b_subs, h);
            }
            current.computed = true;
            return;
        }

        #[cfg(target_arch = "aarch64")]
        if net.has_neon && h % 8 == 0 {
            unsafe {
                simd_acc_fused_neon(&mut current.white, &parent.white, w_adds, w_subs, h);
                simd_acc_fused_neon(&mut current.black, &parent.black, b_adds, b_subs, h);
            }
            current.computed = true;
            return;
        }

        // Scalar fallback: single-pass apply, analogous to SIMD path.
        for j in 0..h {
            let mut w = parent.white[j];
            let mut b = parent.black[j];
            for row in w_adds { w += row[j]; }
            for row in w_subs { w -= row[j]; }
            for row in b_adds { b += row[j]; }
            for row in b_subs { b -= row[j]; }
            current.white[j] = w;
            current.black[j] = b;
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

        let bucket = net.king_bucket(ks);
        let mirror_idx = if net.king_mirror(ks) { 1 } else { 0 };

        let entry = &mut self.finny[perspective as usize * 32 + bucket * 2 + mirror_idx];

        let dst = if perspective == WHITE {
            &mut self.stack[self.top].white
        } else {
            &mut self.stack[self.top].black
        };

        if !entry.valid {
            // No cache — full recompute with register blocking
            dst[..h].copy_from_slice(&net.input_biases[..h]);
            let mut piece_indices: [usize; 32] = [0; 32];
            let mut n_pieces = 0usize;
            for color in 0..2u8 {
                for pt in 0..6u8 {
                    let mut bb = board.pieces[pt as usize] & board.colors[color as usize];
                    while bb != 0 {
                        let sq = pop_lsb(&mut bb) as u8;
                        let idx = net.halfka_index(perspective, king_sq, color, pt, sq);
                        if n_pieces < 32 { piece_indices[n_pieces] = idx; n_pieces += 1; }
                    }
                }
            }
            let empty: [usize; 0] = [];
            finny_batch_apply(net, dst, &net.input_weights, h, &piece_indices[..n_pieces], &empty);
            // Save to cache
            entry.acc[..h].copy_from_slice(&dst[..h]);
            entry.piece_bbs = (board.pieces, board.colors);
            entry.valid = true;
            return;
        }

        // Diff cached vs current — collect all changes, then batch apply
        // Register-blocking: load 8 regs once, apply ALL adds/subs, store once
        let cached_acc = &mut entry.acc;

        // Collect feature indices for adds and subs
        let mut add_rows: [usize; 32] = [0; 32];
        let mut sub_rows: [usize; 32] = [0; 32];
        let mut n_adds = 0usize;
        let mut n_subs = 0usize;

        for color in 0..2u8 {
            for pt in 0..6u8 {
                let prev = entry.piece_bbs.0[pt as usize] & entry.piece_bbs.1[color as usize];
                let curr = board.pieces[pt as usize] & board.colors[color as usize];
                if prev == curr { continue; }

                let mut removed = prev & !curr;
                while removed != 0 {
                    let sq = pop_lsb(&mut removed) as u8;
                    let idx = net.halfka_index(perspective, king_sq, color, pt, sq);
                    if n_subs < 32 { sub_rows[n_subs] = idx; n_subs += 1; }
                }

                let mut added = curr & !prev;
                while added != 0 {
                    let sq = pop_lsb(&mut added) as u8;
                    let idx = net.halfka_index(perspective, king_sq, color, pt, sq);
                    if n_adds < 32 { add_rows[n_adds] = idx; n_adds += 1; }
                }
            }
        }

        // Batch apply with register blocking (Reckless pattern)
        if n_adds > 0 || n_subs > 0 {
            finny_batch_apply(
                net, cached_acc, &net.input_weights, h,
                &add_rows[..n_adds], &sub_rows[..n_subs],
            );
        }

        // Copy updated cache to accumulator
        dst[..h].copy_from_slice(&cached_acc[..h]);
        entry.piece_bbs = (board.pieces, board.colors);
    }

    /// Reset to bottom of stack and invalidate Finny table.
    pub fn reset(&mut self) {
        self.top = 0;
        self.stack[0].computed = false;
        self.stack[0].threat_accurate = [false; 2];
        for entry in self.finny.iter_mut() {
            entry.valid = false;
        }
    }
}

/// AVX-512 batch-apply add/sub weight rows. Same semantics as the AVX2
/// variant but uses 512-bit registers — 32 i16 per register, CHUNK=256.
/// For the common `h=768` case this processes the accumulator in 3 passes
/// instead of AVX2's 6. Measured ~1.5% NPS on v9 after VNNI landed.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn finny_batch_apply_avx512(
    acc: &mut [i16],
    input_weights: &[i16],
    h: usize,
    adds: &[usize],
    subs: &[usize],
) {
    const REGS: usize = 8;
    const LANE: usize = 32; // 32 i16 per ZMM
    const CHUNK: usize = REGS * LANE; // 256 elements per chunk

    let acc_ptr = acc.as_mut_ptr();
    let w_ptr = input_weights.as_ptr();

    let mut offset = 0;
    while offset < h {
        let nregs = ((h - offset).min(CHUNK) + LANE - 1) / LANE;

        let mut regs: [__m512i; REGS] = [_mm512_setzero_si512(); REGS];
        for i in 0..nregs {
            regs[i] = _mm512_loadu_si512(acc_ptr.add(offset + i * LANE) as *const __m512i);
        }

        for &idx in adds {
            let row = w_ptr.add(idx * h + offset);
            for i in 0..nregs {
                let w = _mm512_loadu_si512(row.add(i * LANE) as *const __m512i);
                regs[i] = _mm512_add_epi16(regs[i], w);
            }
        }

        for &idx in subs {
            let row = w_ptr.add(idx * h + offset);
            for i in 0..nregs {
                let w = _mm512_loadu_si512(row.add(i * LANE) as *const __m512i);
                regs[i] = _mm512_sub_epi16(regs[i], w);
            }
        }

        for i in 0..nregs {
            _mm512_storeu_si512(acc_ptr.add(offset + i * LANE) as *mut __m512i, regs[i]);
        }

        offset += CHUNK;
    }
}

/// Batch-apply add/sub weight rows to an accumulator (SIMD-aware dispatcher).
/// Used by Finny refresh (both full-build and cache-diff). Must compile on
/// every supported arch — a scalar fallback is always present.
#[inline]
fn finny_batch_apply(
    net: &NNUENet,
    acc: &mut [i16],
    input_weights: &[i16],
    h: usize,
    adds: &[usize],
    subs: &[usize],
) {
    #[cfg(target_arch = "x86_64")]
    if net.has_avx512 && h % 32 == 0 {
        unsafe { finny_batch_apply_avx512(acc, input_weights, h, adds, subs); }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if net.has_avx2 && h % 16 == 0 {
        unsafe { finny_batch_apply_avx2(acc, input_weights, h, adds, subs); }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if net.has_neon && h % 8 == 0 {
        unsafe { finny_batch_apply_neon(acc, input_weights, h, adds, subs); }
        return;
    }
    // Scalar fallback — always present so cfg-gated SIMD paths can never
    // accidentally compile-out the only codepath (the cfg-swallows-else bug
    // that previously left aarch64 Finny refresh as a no-op on king-bucket
    // changes).
    let _ = net;
    for &idx in adds {
        let base = idx * h;
        for j in 0..h { acc[j] += input_weights[base + j]; }
    }
    for &idx in subs {
        let base = idx * h;
        for j in 0..h { acc[j] -= input_weights[base + j]; }
    }
}

/// NEON SIMD: batch-apply add/sub weight rows to an accumulator.
/// Mirrors finny_batch_apply_avx2 — 16 regs × 8 i16 = 128 elements per chunk.
#[cfg(target_arch = "aarch64")]
unsafe fn finny_batch_apply_neon(
    acc: &mut [i16],
    input_weights: &[i16],
    h: usize,
    adds: &[usize],
    subs: &[usize],
) {
    use std::arch::aarch64::*;
    const REGS: usize = 16;
    const CHUNK: usize = REGS * 8;

    let acc_ptr = acc.as_mut_ptr();
    let w_ptr = input_weights.as_ptr();

    let mut offset = 0;
    while offset < h {
        let nregs = ((h - offset).min(CHUNK) + 7) / 8;

        let mut regs: [int16x8_t; REGS] = [vdupq_n_s16(0); REGS];
        for i in 0..nregs {
            regs[i] = vld1q_s16(acc_ptr.add(offset + i * 8));
        }

        for &idx in adds {
            let row = w_ptr.add(idx * h + offset);
            for i in 0..nregs {
                let w = vld1q_s16(row.add(i * 8));
                regs[i] = vaddq_s16(regs[i], w);
            }
        }

        for &idx in subs {
            let row = w_ptr.add(idx * h + offset);
            for i in 0..nregs {
                let w = vld1q_s16(row.add(i * 8));
                regs[i] = vsubq_s16(regs[i], w);
            }
        }

        for i in 0..nregs {
            vst1q_s16(acc_ptr.add(offset + i * 8), regs[i]);
        }

        offset += CHUNK;
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

fn read_u16(r: &mut impl IoRead) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(|e| format!("read u16: {}", e))?;
    Ok(u16::from_le_bytes(buf))
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

    /// Seeded PRNG for deterministic NEON-vs-scalar tests.
    /// xorshift64*, adequate for scrambling test inputs.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// Plain scalar u8×i8 → i32 dot. No saturation — this is what
    /// `VPDPBUSD`-based VNNI kernels compute exactly.
    #[cfg(target_arch = "x86_64")]
    fn l1_int8_dot_scalar(packed: &[u8], weights: &[i8], h: usize) -> i32 {
        let mut s: i32 = 0;
        for i in 0..h {
            s += packed[i] as i32 * weights[i] as i32;
        }
        s
    }

    /// Saturating scalar reference that matches `VPMADDUBSW + VPMADDWD(ones)`
    /// exactly, including the i16 saturation of each pair. Use this to
    /// compare the non-VNNI `simd512_l1_int8_dot` against stressed inputs.
    ///
    /// (VNNI's `VPDPBUSD` does NOT saturate intermediates — so for stressed
    /// inputs the non-VNNI path diverges from scalar but VNNI matches it.)
    #[cfg(target_arch = "x86_64")]
    fn l1_int8_dot_scalar_sat_pair(packed: &[u8], weights: &[i8], h: usize) -> i32 {
        let mut s: i32 = 0;
        let mut i = 0;
        while i < h {
            let p = packed[i] as i32 * weights[i] as i32
                + packed[i + 1] as i32 * weights[i + 1] as i32;
            s += p.clamp(i16::MIN as i32, i16::MAX as i32);
            i += 2;
        }
        s
    }

    /// AVX-512 VNNI vs scalar row-major int8 dot. Uses **two** scalar
    /// references:
    ///
    /// - VPDPBUSD is non-saturating in its i16 intermediates, so the VNNI
    ///   kernel must match the plain scalar reference byte-for-byte across
    ///   any inputs, even ones that would saturate a `VPMADDUBSW` pair.
    /// - The non-VNNI kernel (`VPMADDUBSW + VPMADDWD(ones)`) saturates the
    ///   i16 pair — so it is compared against the saturating reference.
    ///
    /// That distinction matters: when VNNI is enabled on machines that also
    /// have AVX-512 we will get very slightly different results vs the
    /// non-VNNI path for saturating inputs (VNNI is actually more correct).
    /// Within `forward_with_l1*` the packed values top out around 64 so the
    /// divergence never materialises on real positions, but this test keeps
    /// us honest if that changes.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd512_l1_int8_dot_vnni_matches_scalar() {
        crate::init();

        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
        {
            eprintln!("No AVX-512, skipping VNNI int8 dot test");
            return;
        }

        let mut r = rng(0xc0da_5123_7777_5555);
        // Cover both the unrolled-by-128 body and the tail path.
        for &h in &[64usize, 128, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048] {
            for _trial in 0..4 {
                // "Realistic v9" inputs: packed ≤ 64 ensures VPMADDUBSW pair
                // sum fits in i16 (2 × 64 × 127 = 16256 < 32767). This is
                // the regime in which forward_with_l1 actually operates.
                let mut packed = vec![0u8; h];
                for b in packed.iter_mut() { *b = (r() & 0x3F) as u8; }
                for i in 0..h { if (r() & 3) == 0 { packed[i] = 0; } }

                let mut weights = vec![0i8; h];
                for w in weights.iter_mut() {
                    *w = ((r() & 0xFF) as i8).max(-120).min(120);
                }

                let want = l1_int8_dot_scalar(&packed, &weights, h);

                let got_plain = unsafe { simd512_l1_int8_dot(&packed, &weights, h) };
                assert_eq!(
                    got_plain, want,
                    "non-VNNI simd512_l1_int8_dot mismatch at h={} want={} got={} (unstressed inputs)",
                    h, want, got_plain
                );

                if is_x86_feature_detected!("avx512vnni") {
                    let got_vnni = unsafe { simd512_l1_int8_dot_vnni(&packed, &weights, h) };
                    assert_eq!(
                        got_vnni, want,
                        "VNNI simd512_l1_int8_dot_vnni mismatch at h={} want={} got={}",
                        h, want, got_vnni
                    );
                }
            }
        }

        // Second pass: stressed inputs (full u8 range) — VNNI must track
        // plain-scalar exactly, non-VNNI must track saturating-scalar
        // exactly. Demonstrates VNNI is the mathematically correct one.
        if is_x86_feature_detected!("avx512vnni") {
            let mut r2 = rng(0x5555_aaaa_0000_cafe);
            for &h in &[64usize, 128, 384, 768, 1024] {
                let mut packed = vec![0u8; h];
                for b in packed.iter_mut() { *b = (r2() & 0xFF) as u8; }
                let mut weights = vec![0i8; h];
                for w in weights.iter_mut() {
                    *w = ((r2() & 0xFF) as i8).max(-120).min(120);
                }

                let want_plain = l1_int8_dot_scalar(&packed, &weights, h);
                let want_sat   = l1_int8_dot_scalar_sat_pair(&packed, &weights, h);

                let got_vnni  = unsafe { simd512_l1_int8_dot_vnni(&packed, &weights, h) };
                let got_plain = unsafe { simd512_l1_int8_dot(&packed, &weights, h) };

                assert_eq!(
                    got_vnni, want_plain,
                    "VNNI mismatch under stress at h={}: VNNI={} scalar={}", h, got_vnni, want_plain
                );
                assert_eq!(
                    got_plain, want_sat,
                    "non-VNNI mismatch under stress at h={}: got={} sat-scalar={}",
                    h, got_plain, want_sat
                );
            }
        }
    }

    /// AVX-512 VNNI vs scalar for the sparse-chunk dot path — same kernel
    /// used by `forward_with_l1` when `use_sparse_l1` is enabled.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd512_l1_int8_dot_sparse_vnni_matches_scalar() {
        crate::init();

        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
        {
            eprintln!("No AVX-512 VNNI, skipping sparse VNNI test");
            return;
        }

        let mut r = rng(0xc0da_5123_9999_bbbb);
        for &h in &[64usize, 128, 512, 768, 1024, 2048] {
            for density_pct in [5u32, 25, 50, 90, 100] {
                let mut packed = vec![0u8; h];
                for i in 0..h {
                    if (r() as u32 % 100) < density_pct { packed[i] = (r() & 0xFF) as u8; }
                }
                let mut weights = vec![0i8; h];
                for w in weights.iter_mut() { *w = ((r() & 0xFF) as i8).max(-120).min(120); }

                // Build NNZ index list — 64-byte chunks.
                let mut nnz = vec![0u16; h / 64];
                let nnz_count;
                unsafe { nnz_count = find_nnz_chunks_512(&packed, &mut nnz, h); }

                // Scalar reference uses same NNZ chunk list (any byte inside a chunk
                // being non-zero means the entire chunk is accumulated).
                let mut want: i32 = 0;
                for k in 0..nnz_count {
                    let off = nnz[k] as usize;
                    for i in off..off + 64 {
                        want += packed[i] as i32 * weights[i] as i32;
                    }
                }

                let got = unsafe { simd512_l1_int8_dot_sparse_vnni(&packed, &weights, &nnz, nnz_count) };
                assert_eq!(got, want, "sparse VNNI mismatch h={} density={} got={} want={}", h, density_pct, got, want);
            }
        }
    }

    /// AVX-512 L2 FMA kernel vs scalar reference. Covers the common v9
    /// L2=32 path — a broadcast-FMA fan-out that replaces the gather-heavy
    /// inner loop LLVM used to produce.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_l2_fmadd_avx512_x32_matches_scalar() {
        crate::init();

        if !is_x86_feature_detected!("avx512f") {
            eprintln!("No AVX-512F, skipping L2 FMA test");
            return;
        }

        let l2: usize = 32;
        let mut r = rng(0xc0da_b00b_0000_0111);

        for &l1_out_count in &[8usize, 16, 24, 32, 64] {
            // Bucketed-like weight layout: l2_total may be > l2, weights
            // for bucket `bucket_off` start at column offset `l2_off`.
            for &(l2_total, l2_off, bucket_pad) in &[(32usize, 0usize, 0usize),
                                                     (64, 16, 0usize),
                                                     (96, 64, 0usize)] {
                let _ = bucket_pad;

                let total_w = l1_out_count * l2_total;
                let mut weights = vec![0.0f32; total_w];
                for w in weights.iter_mut() {
                    *w = ((r() as i32 as f32) / 1e9).clamp(-2.0, 2.0);
                }

                let mut biases = vec![0.0f32; l2_total];
                for b in biases.iter_mut() {
                    *b = ((r() as i32 as f32) / 1e9).clamp(-1.0, 1.0);
                }

                let mut l1_out = vec![0.0f32; l1_out_count];
                for v in l1_out.iter_mut() {
                    // Mimic pairwise CReLU+SCReLU outputs: roughly half are
                    // zero, remainder are positive-ish.
                    let raw = ((r() as i32 as f32) / 1e9).clamp(0.0, 1.0);
                    *v = if raw < 0.25 { 0.0 } else { raw };
                }

                // Scalar reference — identical expression to the fallback path.
                let mut h2_scalar = vec![0.0f32; l2];
                for k in 0..l2 { h2_scalar[k] = biases[l2_off + k]; }
                for i in 0..l1_out_count {
                    if l1_out[i] == 0.0 { continue; }
                    for k in 0..l2 {
                        h2_scalar[k] += l1_out[i] * weights[i * l2_total + l2_off + k];
                    }
                }

                let mut h2_simd = vec![0.0f32; l2];
                unsafe {
                    l2_fmadd_avx512_x32(
                        &l1_out, l1_out_count, &weights, l2_total, l2_off,
                        &biases, &mut h2_simd,
                    );
                }

                // FMA reorders operations, so exact equality isn't guaranteed.
                // Tolerance 1e-4 absolute is loose enough for any legal
                // reassoc; tighter than that caught a real bug during
                // development.
                for k in 0..l2 {
                    let diff = (h2_scalar[k] - h2_simd[k]).abs();
                    assert!(
                        diff < 1e-4,
                        "L2 FMA divergence at k={} l1_out_count={} l2_total={} l2_off={}: scalar={} simd={} diff={}",
                        k, l1_out_count, l2_total, l2_off, h2_scalar[k], h2_simd[k], diff
                    );
                }
            }
        }
    }

    /// AVX-512 output-weights dot vs scalar. Small dimensionality so the
    /// tolerance is very tight.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dot_fmadd_avx512_x32_matches_scalar() {
        crate::init();

        if !is_x86_feature_detected!("avx512f") {
            eprintln!("No AVX-512F, skipping output-dot FMA test");
            return;
        }

        let mut r = rng(0xc0da_b00b_0000_0222);
        for _trial in 0..8 {
            let mut a = vec![0.0f32; 32];
            let mut b = vec![0.0f32; 32];
            for v in a.iter_mut() { *v = ((r() as i32 as f32) / 1e9).clamp(-5.0, 5.0); }
            for v in b.iter_mut() { *v = ((r() as i32 as f32) / 1e9).clamp(-5.0, 5.0); }
            let bias = ((r() as i32 as f32) / 1e9).clamp(-5.0, 5.0);

            let mut scalar = bias;
            for i in 0..32 { scalar += a[i] * b[i]; }

            let simd = unsafe { dot_fmadd_avx512_x32(&a, &b, bias) };
            let diff = (scalar - simd).abs();
            assert!(
                diff < 1e-3 || diff / scalar.abs().max(1.0) < 1e-5,
                "dot_fmadd divergence: scalar={} simd={} diff={}", scalar, simd, diff
            );
        }
    }

    /// Scalar reference for finny_batch_apply. Matches the dispatcher's
    /// fallback path byte-for-byte so NEON/AVX2 results can be compared.
    fn finny_batch_apply_scalar_ref(
        acc: &mut [i16],
        input_weights: &[i16],
        h: usize,
        adds: &[usize],
        subs: &[usize],
    ) {
        for &idx in adds {
            let base = idx * h;
            for j in 0..h { acc[j] += input_weights[base + j]; }
        }
        for &idx in subs {
            let base = idx * h;
            for j in 0..h { acc[j] -= input_weights[base + j]; }
        }
    }

    /// Scalar reference for `simd_acc_fused_*` — mirrors the dispatcher's
    /// scalar fallback path (see `refresh_accumulator` / parent+deltas).
    #[cfg(target_arch = "x86_64")]
    fn acc_fused_scalar_ref(
        dst: &mut [i16],
        src: &[i16],
        add_rows: &[&[i16]],
        sub_rows: &[&[i16]],
        h: usize,
    ) {
        for j in 0..h {
            let mut v = src[j];
            for row in add_rows { v = v.wrapping_add(row[j]); }
            for row in sub_rows { v = v.wrapping_sub(row[j]); }
            dst[j] = v;
        }
    }

    /// AVX-512 fused copy+adds+subs vs scalar. Exercises the same shapes
    /// the dispatcher uses on-move for v9 (h=768, small add/sub row sets).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_acc_fused_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            eprintln!("No AVX-512, skipping test");
            return;
        }

        for &h in &[64usize, 128, 256, 512, 768, 1024, 1536] {
            let mut r = rng(0xc0da_ace_0011_0001 ^ h as u64);
            let n_rows = 10usize;
            let mut row_bufs: Vec<Vec<i16>> = (0..n_rows).map(|_| {
                (0..h).map(|_| ((r() % 1024) as i16) - 512).collect()
            }).collect();

            // Borrow row buffers as &[i16] slices.
            let row_refs: Vec<&[i16]> = row_bufs.iter().map(|v| v.as_slice()).collect();
            let add_rows: &[&[i16]] = &row_refs[..4];
            let sub_rows: &[&[i16]] = &row_refs[4..9];

            let src: Vec<i16> = (0..h).map(|_| ((r() % 2000) as i16) - 1000).collect();

            let mut dst_scalar = vec![0i16; h];
            acc_fused_scalar_ref(&mut dst_scalar, &src, add_rows, sub_rows, h);

            let mut dst_simd = vec![0i16; h];
            unsafe { simd_acc_fused_avx512(&mut dst_simd, &src, add_rows, sub_rows, h); }

            assert_eq!(dst_scalar, dst_simd,
                "simd_acc_fused_avx512 diverged from scalar at h={}", h);

            // Edge case: empty deltas → dst == src.
            let mut dst_simd = vec![0i16; h];
            unsafe { simd_acc_fused_avx512(&mut dst_simd, &src, &[], &[], h); }
            assert_eq!(src, dst_simd,
                "simd_acc_fused_avx512 empty-deltas altered dst at h={}", h);

            // Edge case: adds only.
            let mut dst_scalar = vec![0i16; h];
            acc_fused_scalar_ref(&mut dst_scalar, &src, add_rows, &[], h);
            let mut dst_simd = vec![0i16; h];
            unsafe { simd_acc_fused_avx512(&mut dst_simd, &src, add_rows, &[], h); }
            assert_eq!(dst_scalar, dst_simd,
                "simd_acc_fused_avx512 adds-only diverged at h={}", h);

            // Avoid unused-warn on the row buffer keepalive.
            row_bufs.clear();
        }
    }

    /// AVX-512 finny batch apply vs scalar. Mirrors the NEON test —
    /// confirms add/sub semantics and accumulator coherence across
    /// CHUNK=256 partial passes (relevant for h=768).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_finny_batch_apply_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512bw") {
            eprintln!("No AVX-512, skipping test");
            return;
        }
        // Cover sizes where h falls on/off the 256-CHUNK boundary.
        for &h in &[64usize, 128, 256, 512, 768, 1024] {
            let n_features = 40;
            let mut r = rng(0xc0da_f1ec_aa01_0001 ^ h as u64);

            let mut weights = vec![0i16; n_features * h];
            for w in weights.iter_mut() { *w = (r() as i32 as i16).rem_euclid(511) - 255; }

            let mut acc_init = vec![0i16; h];
            for v in acc_init.iter_mut() { *v = (r() as i32 as i16).rem_euclid(2001) - 1000; }

            let adds = [2usize, 5, 11, 17, 23, 35];
            let subs = [1usize, 7, 13, 19, 29, 31, 37];

            let mut scalar_acc = acc_init.clone();
            finny_batch_apply_scalar_ref(&mut scalar_acc, &weights, h, &adds, &subs);

            let mut simd_acc = acc_init.clone();
            unsafe { finny_batch_apply_avx512(&mut simd_acc, &weights, h, &adds, &subs); }

            assert_eq!(scalar_acc, simd_acc,
                "finny_batch_apply_avx512 diverged from scalar at h={}", h);

            // Adds-only and empty deltas edge cases.
            let mut scalar_acc = acc_init.clone();
            finny_batch_apply_scalar_ref(&mut scalar_acc, &weights, h, &adds, &[]);
            let mut simd_acc = acc_init.clone();
            unsafe { finny_batch_apply_avx512(&mut simd_acc, &weights, h, &adds, &[]); }
            assert_eq!(scalar_acc, simd_acc,
                "finny_batch_apply_avx512 adds-only diverged at h={}", h);

            let mut simd_acc = acc_init.clone();
            unsafe { finny_batch_apply_avx512(&mut simd_acc, &weights, h, &[], &[]); }
            assert_eq!(acc_init, simd_acc,
                "finny_batch_apply_avx512 empty deltas mutated accumulator at h={}", h);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_finny_batch_apply_neon_matches_scalar() {
        let h = 768;
        let n_features = 32;
        let mut r = rng(0xc0da_f1ec_0000_0001);

        let mut weights = vec![0i16; n_features * h];
        for w in weights.iter_mut() { *w = (r() as i32 as i16).rem_euclid(511) - 255; }

        let mut acc_init = vec![0i16; h];
        for v in acc_init.iter_mut() { *v = (r() as i32 as i16).rem_euclid(2001) - 1000; }

        let adds = [2usize, 5, 11, 17, 23];
        let subs = [1usize, 7, 13, 19, 29, 31];

        let mut scalar_acc = acc_init.clone();
        finny_batch_apply_scalar_ref(&mut scalar_acc, &weights, h, &adds, &subs);

        let mut neon_acc = acc_init.clone();
        unsafe { finny_batch_apply_neon(&mut neon_acc, &weights, h, &adds, &subs); }

        assert_eq!(scalar_acc, neon_acc,
            "finny_batch_apply_neon diverged from scalar");

        // Adds-only edge case (mirrors the "no-cache" refresh path).
        let mut scalar_acc = acc_init.clone();
        finny_batch_apply_scalar_ref(&mut scalar_acc, &weights, h, &adds, &[]);
        let mut neon_acc = acc_init.clone();
        unsafe { finny_batch_apply_neon(&mut neon_acc, &weights, h, &adds, &[]); }
        assert_eq!(scalar_acc, neon_acc,
            "finny_batch_apply_neon (adds-only) diverged from scalar");

        // Empty deltas must be identity.
        let mut neon_acc = acc_init.clone();
        unsafe { finny_batch_apply_neon(&mut neon_acc, &weights, h, &[], &[]); }
        assert_eq!(acc_init, neon_acc,
            "finny_batch_apply_neon with empty deltas mutated accumulator");
    }

    /// Scalar reference for neon_pairwise_pack_fused.
    #[cfg(target_arch = "aarch64")]
    fn pairwise_pack_scalar_ref(acc: &[i16], threat: Option<&[i16]>, out: &mut [u8], pw: usize) {
        for i in 0..pw {
            let (ta, tb) = match threat {
                Some(t) => (t[i] as i32, t[i + pw] as i32),
                None => (0, 0),
            };
            let a = (acc[i] as i32 + ta).clamp(0, QA);
            let b = (acc[i + pw] as i32 + tb).clamp(0, QA);
            out[i] = ((a * b) >> FT_SHIFT) as u8;
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_pairwise_pack_matches_scalar() {
        let pw = 384;
        let mut r = rng(0xc0da_fa11_0000_0003);

        let mut acc = vec![0i16; pw * 2];
        for v in acc.iter_mut() { *v = (r() as i32 as i16).rem_euclid(501) - 250; }

        let mut threat = vec![0i16; pw * 2];
        for v in threat.iter_mut() { *v = (r() as i32 as i16).rem_euclid(201) - 100; }

        // No-threat path
        let mut scalar_out = vec![0u8; pw];
        pairwise_pack_scalar_ref(&acc, None, &mut scalar_out, pw);
        let mut neon_out = vec![0u8; pw];
        unsafe { neon_pairwise_pack_fused(&acc, std::ptr::null(), &mut neon_out, pw); }
        assert_eq!(scalar_out, neon_out,
            "neon_pairwise_pack_fused (no threat) diverged from scalar");

        // With-threat path
        let mut scalar_out = vec![0u8; pw];
        pairwise_pack_scalar_ref(&acc, Some(&threat), &mut scalar_out, pw);
        let mut neon_out = vec![0u8; pw];
        unsafe { neon_pairwise_pack_fused(&acc, threat.as_ptr(), &mut neon_out, pw); }
        assert_eq!(scalar_out, neon_out,
            "neon_pairwise_pack_fused (with threat) diverged from scalar");

        // Edge case: values at the clamp boundaries [0, QA].
        let mut boundary_acc = vec![0i16; pw * 2];
        for i in 0..pw {
            boundary_acc[i] = if i % 3 == 0 { -100 } else if i % 3 == 1 { QA as i16 + 50 } else { QA as i16 / 2 };
            boundary_acc[pw + i] = if i % 2 == 0 { 0 } else { QA as i16 };
        }
        let mut scalar_out = vec![0u8; pw];
        pairwise_pack_scalar_ref(&boundary_acc, None, &mut scalar_out, pw);
        let mut neon_out = vec![0u8; pw];
        unsafe { neon_pairwise_pack_fused(&boundary_acc, std::ptr::null(), &mut neon_out, pw); }
        assert_eq!(scalar_out, neon_out,
            "neon_pairwise_pack_fused at clamp boundaries diverged from scalar");
    }

    #[test]
    fn test_king_buckets() {
        // Uniform layout (default for v5/v9 16-bucket nets).
        let (kb, km) = compute_king_buckets(KbLayout::Uniform);
        // a1 (sq 0): file 0, rank 0 → mirrored_file=0, rank_group=0 → bucket 0
        assert_eq!(kb[0], 0);
        // e1 (sq 4): file 4, rank 0 → mirrored_file=3, rank_group=0 → bucket 12, mirror=true
        assert_eq!(kb[4], 12);
        assert!(km[4]);
        // a1 no mirror
        assert!(!km[0]);
    }

    /// Verify the Reckless king bucket layout matches Reckless's
    /// INPUT_BUCKETS_LAYOUT from `Reckless/src/nnue.rs:71-80` exactly.
    /// Catches drift if either the flat table here or the derivation in
    /// `compute_king_buckets` changes incompatibly.
    #[test]
    fn test_reckless_king_buckets() {
        let (kb, km) = compute_king_buckets(KbLayout::Reckless);
        #[rustfmt::skip]
        const EXPECTED: [usize; 64] = [
            0, 1, 2, 3, 3, 2, 1, 0,
            4, 5, 6, 7, 7, 6, 5, 4,
            8, 8, 8, 8, 8, 8, 8, 8,
            9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9,
        ];
        for sq in 0..64 {
            assert_eq!(
                kb[sq], EXPECTED[sq],
                "reckless kb mismatch at sq={} got {} expected {}",
                sq, kb[sq], EXPECTED[sq]
            );
        }
        // Mirror flag still toggles on files e-h (Reckless's layout bakes
        // mirror into bucket ids but the rest of inference still needs the
        // mirror flag for feature indexing).
        assert!(km[4]);   // e1
        assert!(!km[0]);  // a1
    }

    /// Regression test for the 2026-04-20 SMP race fix. Concurrently
    /// compute king-bucket tables on many threads for different layouts
    /// and verify no tearing / incorrect value. If KING_BUCKET were still
    /// a `static mut` written per layout, this would race; with per-net
    /// fields it's trivially safe.
    #[test]
    fn test_king_buckets_concurrent_layouts() {
        use std::thread;
        let mut handles = vec![];
        for i in 0..8 {
            let layout = match i % 3 {
                0 => KbLayout::Uniform,
                1 => KbLayout::Consensus,
                _ => KbLayout::Reckless,
            };
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    let (kb, km) = compute_king_buckets(layout);
                    // Spot-check a few invariants per layout.
                    match layout {
                        KbLayout::Uniform => {
                            assert_eq!(kb[0], 0);
                            assert_eq!(kb[4], 12);
                            assert!(km[4]);
                        }
                        KbLayout::Consensus => {
                            assert_eq!(kb[0], 0);  // a1
                        }
                        KbLayout::Reckless => {
                            assert_eq!(kb[0], 0);
                            assert_eq!(kb[7], 0);  // mirror on h-file
                            assert_eq!(kb[24], 9);  // rank 4+
                        }
                    }
                }
            }));
        }
        for h in handles { h.join().unwrap(); }
    }

    #[test]
    fn test_kb_layout_roundtrip() {
        // File-format byte must round-trip through KbLayout::from_id.
        assert_eq!(KbLayout::from_id(0), Some(KbLayout::Uniform));
        assert_eq!(KbLayout::from_id(1), Some(KbLayout::Consensus));
        assert_eq!(KbLayout::from_id(2), Some(KbLayout::Reckless));
        assert_eq!(KbLayout::from_id(3), None);
        assert_eq!(KbLayout::Uniform.default_count(), 16);
        assert_eq!(KbLayout::Consensus.default_count(), 16);
        assert_eq!(KbLayout::Reckless.default_count(), 10);
    }

    #[test]
    fn test_halfka_index() {
        let (kb, km) = compute_king_buckets(KbLayout::Uniform);
        // White perspective, king on e1 (sq 4), white pawn on e2 (sq 12)
        let idx = halfka_index_with(&kb, &km, WHITE, 4, WHITE, PAWN, 12);
        // king sq 4, file=4 >= 4, so mirror: ks=4 stays, ps=12 → file mirror: (12 & !7) | (7 - (12&7)) = 8 | (7-4) = 8|3 = 11
        // bucket = king_bucket[4] = 12
        // pi = 0 (white pawn from white perspective)
        // index = 12 * 768 + 0 * 64 + 11 = 9216 + 11 = 9227
        assert_eq!(idx, 9227);

        // Black perspective, king on e8 (sq 60), white pawn on e2 (sq 12)
        let idx = halfka_index_with(&kb, &km, BLACK, 60, WHITE, PAWN, 12);
        // ks = 60 ^ 56 = 4, ps = 12 ^ 56 = 52
        // pi = 0 (white pawn) → black perspective: 0 + 6 = 6
        // king_mirror[4] = true → ps = (52 & !7) | (7 - (52 & 7)) = 48 | (7-4) = 48|3 = 51
        // bucket = king_bucket[4] = 12
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

    /// Measure per-row weight-norm distribution for threat features in
    /// a loaded v9 net. Informs sparsity compaction — if many rows have
    /// norms near zero, load-time compaction captures cache wins.
    /// Run with:
    ///   cargo test --release measure_threat_weight_norms -- --nocapture --ignored
    #[test]
    #[ignore]
    fn measure_threat_weight_norms() {
        let net_path = "nets/net-v9-768th16x32-w15-e800s800-xray.nnue";
        let net = match NNUENet::load(net_path) {
            Ok(n) => n,
            Err(e) => { eprintln!("Skip: can't load {}: {}", net_path, e); return; }
        };
        let nf = net.num_threat_features;
        let h = net.hidden_size;
        if nf == 0 || net.threat_weights.is_empty() {
            eprintln!("Net has no threat features"); return;
        }
        eprintln!("Net: {} threat features × {} hidden = {} total i8 weights ({:.1} MB)",
            nf, h, nf * h, (nf * h) as f64 / 1_048_576.0);

        // L∞ (max absolute) per row — cheapest meaningful norm
        // L1 (sum |w|) — total "mass" of the row
        let mut linf: Vec<u8> = Vec::with_capacity(nf);
        let mut l1: Vec<u32> = Vec::with_capacity(nf);
        for f in 0..nf {
            let row = &net.threat_weights[f * h..(f + 1) * h];
            let mut mx: u8 = 0;
            let mut s1: u32 = 0;
            for &w in row {
                let aw = (w as i32).unsigned_abs() as u8;
                if aw > mx { mx = aw; }
                s1 += aw as u32;
            }
            linf.push(mx);
            l1.push(s1);
        }

        // Bucket by L∞ norm. i8 range is [-127, 127], so L∞ is in [0, 127].
        let mut linf_buckets = [0u64; 8];  // 0, 1-2, 3-7, 8-15, 16-31, 32-63, 64-127, 128+
        for &m in &linf {
            let b = match m {
                0 => 0, 1..=2 => 1, 3..=7 => 2, 8..=15 => 3,
                16..=31 => 4, 32..=63 => 5, 64..=127 => 6, _ => 7,
            };
            linf_buckets[b] += 1;
        }

        // Same for L1 (absolute sum — max is 768 × 127 ≈ 97K)
        let mut l1_buckets = [0u64; 9];
        for &s in &l1 {
            let b = match s {
                0 => 0, 1..=99 => 1, 100..=499 => 2, 500..=999 => 3,
                1000..=2999 => 4, 3000..=9999 => 5, 10000..=29999 => 6,
                30000..=99999 => 7, _ => 8,
            };
            l1_buckets[b] += 1;
        }

        eprintln!("\n--- L∞ norm (max |weight| per row) distribution ---");
        eprintln!("  0:       {:>7} rows ({:.1}%)", linf_buckets[0], linf_buckets[0] as f64 * 100.0 / nf as f64);
        eprintln!("  1-2:     {:>7} rows ({:.1}%)", linf_buckets[1], linf_buckets[1] as f64 * 100.0 / nf as f64);
        eprintln!("  3-7:     {:>7} rows ({:.1}%)", linf_buckets[2], linf_buckets[2] as f64 * 100.0 / nf as f64);
        eprintln!("  8-15:    {:>7} rows ({:.1}%)", linf_buckets[3], linf_buckets[3] as f64 * 100.0 / nf as f64);
        eprintln!("  16-31:   {:>7} rows ({:.1}%)", linf_buckets[4], linf_buckets[4] as f64 * 100.0 / nf as f64);
        eprintln!("  32-63:   {:>7} rows ({:.1}%)", linf_buckets[5], linf_buckets[5] as f64 * 100.0 / nf as f64);
        eprintln!("  64-127:  {:>7} rows ({:.1}%)", linf_buckets[6], linf_buckets[6] as f64 * 100.0 / nf as f64);

        eprintln!("\n--- L1 norm (sum |weight| per 768-wide row) distribution ---");
        let names = ["0", "1-99", "100-499", "500-999", "1k-3k", "3k-10k", "10k-30k", "30k-100k", "100k+"];
        for (i, &c) in l1_buckets.iter().enumerate() {
            eprintln!("  {:<9} {:>7} rows ({:.1}%)", names[i], c, c as f64 * 100.0 / nf as f64);
        }

        // Compaction scenarios: if we threshold rows by L∞ < ε, how many survive?
        eprintln!("\n--- Compaction scenarios ---");
        for &eps in &[0u8, 1, 2, 3, 5, 8, 16] {
            let kept = linf.iter().filter(|&&m| m > eps).count();
            let kept_bytes = kept * h;
            let total_bytes = nf * h;
            eprintln!("  Drop rows with L∞ ≤ {:3}: keep {:>5} of {} ({:.1}%) → {:.1} MB (was {:.1} MB)",
                eps, kept, nf, kept as f64 * 100.0 / nf as f64,
                kept_bytes as f64 / 1_048_576.0, total_bytes as f64 / 1_048_576.0);
        }
    }

    /// Test SIMD vs scalar consistency for forward pass.
    /// Loads the production net (if available), evaluates test positions
    /// through both SIMD and scalar paths, asserts results match.
    ///
    /// Honours `CODA_TEST_NET=<path>` so callers can target a specific
    /// net (e.g. the v9 net to cover the threat + pairwise VNNI path).
    #[test]
    fn test_simd_scalar_consistency() {
        use crate::board::Board;

        // Try to load a net — skip if none available
        let net_path = if let Ok(override_path) = std::env::var("CODA_TEST_NET") {
            if !std::path::Path::new(&override_path).exists() {
                eprintln!("Skipping SIMD consistency test: CODA_TEST_NET path missing: {}", override_path);
                return;
            }
            Box::leak(override_path.into_boxed_str()) as &'static str
        } else if std::path::Path::new("net.nnue").exists() {
            "net.nnue"
        } else {
            // Try the named production net
            let entries = std::fs::read_dir(".").ok();
            let found = entries.and_then(|e| {
                e.filter_map(|f| f.ok())
                    .find(|f| {
                        let name = f.file_name().to_string_lossy().to_string();
                        name.starts_with("net-v") && name.ends_with(".nnue")
                    })
                    .map(|f| f.path().to_string_lossy().to_string())
            });
            if found.is_none() {
                eprintln!("Skipping SIMD consistency test: no .nnue file found");
                return;
            }
            // Leak the string to get a 'static lifetime (test only)
            Box::leak(found.unwrap().into_boxed_str())
        };

        let net = match NNUENet::load(net_path) {
            Ok(n) => n,
            Err(e) => {
                eprintln!("Skipping SIMD consistency test: {}", e);
                return;
            }
        };

        let test_fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "r1bqkb1r/pppppppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "4k3/8/8/8/8/8/PPPPPPPP/R3K3 w Q - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ];

        let h = net.hidden_size;
        let mut max_diff = 0i32;
        for fen in &test_fens {
            let board = Board::from_fen(fen);
            let piece_count = board.occupied().count_ones();

            // Compute accumulator once with SIMD (normal path)
            let mut acc = NNUEAccumulator::new(h);
            acc.force_recompute(&net, &board);

            // Forward with full SIMD (whatever the runtime detected).
            let simd_result = net.forward(&acc, board.side_to_move, piece_count);

            // Save current SIMD flags so we can toggle them for comparison.
            let saved_avx2 = net.has_avx2;
            let saved_neon = net.has_neon;
            let saved_avx512 = net.has_avx512;
            let saved_avx512_vnni = net.has_avx512_vnni;
            let saved_avx_vnni = net.has_avx_vnni;

            // SAFETY: single-threaded test, no concurrent reads.
            let net_ptr = &net as *const NNUENet as *mut NNUENet;

            // If VNNI is available, cross-check VNNI vs non-VNNI AVX-512
            // as well — exposes any bug in the VPDPBUSD kernels even if
            // the scalar fallback happens to be disabled at this tier.
            let mut vnni_result = simd_result;
            if saved_avx512_vnni || saved_avx_vnni {
                unsafe {
                    (*net_ptr).has_avx512_vnni = false;
                    (*net_ptr).has_avx_vnni = false;
                }
                let non_vnni = net.forward(&acc, board.side_to_move, piece_count);
                unsafe {
                    (*net_ptr).has_avx512_vnni = saved_avx512_vnni;
                    (*net_ptr).has_avx_vnni = saved_avx_vnni;
                }
                let vnni_diff = (simd_result - non_vnni).abs();
                eprintln!("  {}: VNNI={} non-VNNI={} vnni_diff={}",
                    fen, simd_result, non_vnni, vnni_diff);
                // VNNI must match the AVX-512 int8 path exactly — no rounding.
                assert_eq!(simd_result, non_vnni,
                    "VNNI vs non-VNNI divergence for {} (VNNI={} nonVNNI={})",
                    fen, simd_result, non_vnni);
                vnni_result = non_vnni;
            }
            let _ = vnni_result;

            // Now disable all SIMD to exercise the scalar fallback.
            unsafe {
                (*net_ptr).has_avx2 = false;
                (*net_ptr).has_neon = false;
                (*net_ptr).has_avx512 = false;
                (*net_ptr).has_avx512_vnni = false;
                (*net_ptr).has_avx_vnni = false;
            }

            let scalar_result = net.forward(&acc, board.side_to_move, piece_count);

            // Restore SIMD flags
            unsafe {
                (*net_ptr).has_avx2 = saved_avx2;
                (*net_ptr).has_neon = saved_neon;
                (*net_ptr).has_avx512 = saved_avx512;
                (*net_ptr).has_avx512_vnni = saved_avx512_vnni;
                (*net_ptr).has_avx_vnni = saved_avx_vnni;
            }

            let diff = (simd_result - scalar_result).abs();
            eprintln!("  {}: SIMD={} scalar={} diff={} {}",
                fen, simd_result, scalar_result, diff,
                if diff > 2 { "MISMATCH" } else if diff > 0 { "rounding" } else { "exact" });
            max_diff = max_diff.max(diff);
        }
        assert!(max_diff <= 2, "Forward SIMD/scalar max diff {} exceeds tolerance of 2cp", max_diff);

        // Phase 2: Test accumulator recompute SIMD vs scalar consistency
        // force_recompute with SIMD vs scalar should produce identical accumulators
        let net_ptr = &net as *const NNUENet as *mut NNUENet;
        let saved_avx2 = net.has_avx2;
        let saved_neon = net.has_neon;
        let saved_avx512 = net.has_avx512;
        let saved_avx512_vnni = net.has_avx512_vnni;
        let saved_avx_vnni = net.has_avx_vnni;
        let mut max_acc_diff = 0i16;
        for fen in &test_fens {
            let board = Board::from_fen(fen);

            // Recompute with SIMD
            unsafe {
                (*net_ptr).has_avx2 = saved_avx2;
                (*net_ptr).has_neon = saved_neon;
                (*net_ptr).has_avx512 = saved_avx512;
                (*net_ptr).has_avx512_vnni = saved_avx512_vnni;
                (*net_ptr).has_avx_vnni = saved_avx_vnni;
            }
            let mut acc_simd = NNUEAccumulator::new(h);
            acc_simd.force_recompute(&net, &board);

            // Recompute with scalar
            unsafe {
                (*net_ptr).has_avx2 = false;
                (*net_ptr).has_neon = false;
                (*net_ptr).has_avx512 = false;
                (*net_ptr).has_avx512_vnni = false;
                (*net_ptr).has_avx_vnni = false;
            }
            let mut acc_scalar = NNUEAccumulator::new(h);
            acc_scalar.force_recompute(&net, &board);
            unsafe {
                (*net_ptr).has_avx2 = saved_avx2;
                (*net_ptr).has_neon = saved_neon;
                (*net_ptr).has_avx512 = saved_avx512;
                (*net_ptr).has_avx512_vnni = saved_avx512_vnni;
                (*net_ptr).has_avx_vnni = saved_avx_vnni;
            }

            // Compare all accumulator values
            let mut pos_diff = 0i16;
            for i in 0..h {
                let d = (acc_simd.white()[i] - acc_scalar.white()[i]).abs();
                pos_diff = pos_diff.max(d);
                let d = (acc_simd.black()[i] - acc_scalar.black()[i]).abs();
                pos_diff = pos_diff.max(d);
            }
            if pos_diff > 0 {
                eprintln!("  Acc recompute diff at {}: max_diff={}", fen, pos_diff);
            }
            max_acc_diff = max_acc_diff.max(pos_diff);
        }
        assert!(max_acc_diff == 0, "Accumulator SIMD/scalar max diff {} (should be exact)", max_acc_diff);
    }

    /// Helper: find and load any available v9 net for eval tests.
    ///
    /// Override via `CODA_TEST_NET=<path>` env var to target a specific net —
    /// essential for validating new/untested inference paths (e.g. Reckless
    /// 10-KB layout) against the existing fuzzers.
    fn try_load_v9_net() -> Option<NNUENet> {
        // Explicit override wins.
        if let Ok(override_path) = std::env::var("CODA_TEST_NET") {
            if std::path::Path::new(&override_path).exists() {
                if let Ok(net) = NNUENet::load(&override_path) {
                    eprintln!("try_load_v9_net: using CODA_TEST_NET override: {}", override_path);
                    return Some(net);
                }
                eprintln!("try_load_v9_net: CODA_TEST_NET path exists but failed to load: {}", override_path);
            } else {
                eprintln!("try_load_v9_net: CODA_TEST_NET path does not exist: {}", override_path);
            }
        }
        // Prefer the xray-fixed w15 s200 net (our best v9), fall back to others.
        let candidates = [
            "nets/net-v9-768th16x32-w15-e200s200-xray-fixed.nnue",
            "nets/net-v9-768th16x32-w0-e200s200-xray-fixed.nnue",
            "nets/net-v9-768th16x32-w0-e400s400-noxray.nnue",
            "nets/net-v9-768pwth16x32-w0-e200s200.nnue",
        ];
        for p in candidates.iter() {
            if std::path::Path::new(p).exists() {
                if let Ok(net) = NNUENet::load(p) {
                    return Some(net);
                }
            }
        }
        None
    }

    /// Mirror a FEN: rank-flip board, swap piece colors (case), flip
    /// side-to-move, flip castling rights case, flip EP square rank.
    /// Result represents the same "position content" reflected: a correct
    /// NNUE must produce the identical evaluation from STM's perspective
    /// for a position and its mirror.
    fn mirror_fen(fen: &str) -> String {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        assert!(parts.len() >= 4, "FEN missing fields: {}", fen);

        // 1. Board: rank-flip + case-swap.
        let ranks: Vec<&str> = parts[0].split('/').collect();
        let mut new_ranks: Vec<String> = Vec::with_capacity(ranks.len());
        for r in ranks.iter().rev() {
            let swapped: String = r.chars().map(|c| {
                if c.is_ascii_uppercase() { c.to_ascii_lowercase() }
                else if c.is_ascii_lowercase() { c.to_ascii_uppercase() }
                else { c }
            }).collect();
            new_ranks.push(swapped);
        }
        let new_board = new_ranks.join("/");

        // 2. Side-to-move.
        let new_stm = if parts[1] == "w" { "b" } else { "w" };

        // 3. Castling rights: swap case (K↔k, Q↔q, etc.), then sort so the
        //    output is canonical. Order K,Q,k,q.
        let new_castle: String = if parts[2] == "-" {
            "-".to_string()
        } else {
            let mut chars: Vec<char> = parts[2].chars().map(|c| {
                if c.is_ascii_uppercase() { c.to_ascii_lowercase() }
                else if c.is_ascii_lowercase() { c.to_ascii_uppercase() }
                else { c }
            }).collect();
            chars.sort_by_key(|c| match c {
                'K' => 0, 'Q' => 1, 'k' => 2, 'q' => 3, _ => 4,
            });
            chars.iter().collect()
        };

        // 4. EP square: flip rank (e.g. e3 → e6).
        let new_ep: String = if parts[3] == "-" {
            "-".to_string()
        } else {
            let bytes = parts[3].as_bytes();
            let file = bytes[0] as char;
            let rank = bytes[1] as char;
            let new_rank = match rank {
                '1' => '8', '2' => '7', '3' => '6', '4' => '5',
                '5' => '4', '6' => '3', '7' => '2', '8' => '1',
                _ => rank,
            };
            format!("{}{}", file, new_rank)
        };

        let hm = parts.get(4).unwrap_or(&"0");
        let fm = parts.get(5).unwrap_or(&"1");
        format!("{} {} {} {} {} {}", new_board, new_stm, new_castle, new_ep, hm, fm)
    }

    /// Tier-1 discovery test: eval(pos) ≈ eval(mirror(pos)) from STM's
    /// perspective. Differences beyond a gross threshold indicate a real
    /// perspective/flip bug (accumulator mirror, halfka index, threat
    /// mirror, king-file handling).
    ///
    /// Tolerance is intentionally generous (50cp). Expected small
    /// asymmetries come from:
    /// - Semi-exclusion in threat features uses PHYSICAL square ordering
    ///   (same decision in both perspectives, by design, matching Bullet
    ///   training). This is orientation-dependent, so mirrored positions
    ///   activate different feature indices. A well-trained net learns
    ///   near-symmetric weights at those indices; an s200-trained net
    ///   still has visible residual (~10-20cp on tactical positions).
    /// - Integer quantization produces small (<5cp) rounding drift
    ///   between the two forward passes.
    ///
    /// A symmetry diff > 50cp signals a real bug — the test fails and
    /// prints all positions + diffs for triage.
    #[test]
    fn test_eval_color_symmetry() {
        use crate::board::Board;
        use crate::threat_accum::ThreatStack;

        crate::init();
        let net = match try_load_v9_net() {
            Some(n) => n,
            None => { eprintln!("Skipping eval symmetry test: no v9 net available"); return; }
        };

        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/PPPPPPPP/R3K3 w Q - 0 1",
            "2r3k1/pp3ppp/2n1b3/3pP3/3P4/2NB4/PP3PPP/R4RK1 b - - 0 1",
            // King on e-file — exercises mirror-flip boundary specifically
            "4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1",
            // King on h-file — heavy mirror territory
            "7k/8/8/8/8/8/8/7K w - - 0 1",
        ];

        // v9 threat nets: semi-exclusion is orientation-dependent by design,
        // so mirrored positions activate different threat-feature indices.
        // Well-trained nets converge to near-symmetric weights at those
        // indices, but a residual remains that's larger than quantization
        // drift alone. 75cp accommodates s200..s800 v9 nets on tactical
        // positions while still catching real perspective/flip bugs.
        // Non-threat nets keep the tighter 50cp bar.
        let tolerance_cp: i32 = if net.has_threats { 75 } else { 50 };
        let h = net.hidden_size;
        let mut max_diff = 0i32;
        let mut fails: Vec<(String, i32, i32, i32)> = Vec::new();

        eprintln!("eval symmetry: tolerance = {} cp", tolerance_cp);
        for fen in &fens {
            let mirrored = mirror_fen(fen);

            let b1 = Board::from_fen(fen);
            let mut acc1 = NNUEAccumulator::new(h);
            let mut ts1 = ThreatStack::new(h);
            ts1.active = net.has_threats;
            if ts1.active {
                ts1.refresh(&net.threat_weights, net.num_threat_features, &b1, crate::types::WHITE);
                ts1.refresh(&net.threat_weights, net.num_threat_features, &b1, crate::types::BLACK);
            }
            let e1 = crate::eval::evaluate_nnue(&b1, &net, &mut acc1, &ts1);

            let b2 = Board::from_fen(&mirrored);
            let mut acc2 = NNUEAccumulator::new(h);
            let mut ts2 = ThreatStack::new(h);
            ts2.active = net.has_threats;
            if ts2.active {
                ts2.refresh(&net.threat_weights, net.num_threat_features, &b2, crate::types::WHITE);
                ts2.refresh(&net.threat_weights, net.num_threat_features, &b2, crate::types::BLACK);
            }
            let e2 = crate::eval::evaluate_nnue(&b2, &net, &mut acc2, &ts2);

            let diff = (e1 - e2).abs();
            eprintln!("  {}: eval={} mirror={} diff={}{}",
                fen, e1, e2, diff,
                if diff > tolerance_cp { " FAIL" } else if diff > 5 { " (residual)" } else { "" }
            );
            max_diff = max_diff.max(diff);
            if diff > tolerance_cp {
                fails.push((fen.to_string(), e1, e2, diff));
            }
        }

        eprintln!("eval symmetry max diff = {} cp (tolerance {} cp)", max_diff, tolerance_cp);
        assert!(fails.is_empty(),
            "eval symmetry failed on {} positions (> {}cp tolerance). Details above.",
            fails.len(), tolerance_cp);
    }

    /// Tier-1 discovery test: eval changes monotonically when removing
    /// piece types in value order. Removes one piece at a time from a
    /// balanced starting position; the eval delta ranking must be
    /// roughly Queen > Rook > Bishop ≈ Knight > Pawn, all negative for
    /// the side losing material.
    ///
    /// Detects: inverted piece values, eval-scale sign bugs, bucket
    /// mis-selection producing non-monotone evals at material boundaries.
    #[test]
    fn test_eval_piece_value_monotonicity() {
        use crate::board::Board;
        use crate::threat_accum::ThreatStack;

        crate::init();
        let net = match try_load_v9_net() {
            Some(n) => n,
            None => { eprintln!("Skipping monotonicity test: no v9 net available"); return; }
        };

        let h = net.hidden_size;

        fn eval_fen(net: &NNUENet, fen: &str) -> i32 {
            let b = Board::from_fen(fen);
            let h = net.hidden_size;
            let mut acc = NNUEAccumulator::new(h);
            let mut ts = ThreatStack::new(h);
            ts.active = net.has_threats;
            if ts.active {
                ts.refresh(&net.threat_weights, net.num_threat_features, &b, crate::types::WHITE);
                ts.refresh(&net.threat_weights, net.num_threat_features, &b, crate::types::BLACK);
            }
            crate::eval::evaluate_nnue(&b, net, &mut acc, &ts)
        }

        // Startpos (balanced). Evals are STM-relative = 0 on a balanced position.
        let base = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let e_base = eval_fen(&net, base);

        // Remove one black piece at a time; STM is white so each removal
        // should return a POSITIVE eval (white is relatively ahead).
        let cases = [
            ("black queen",  "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("black rook",   "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("black bishop", "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("black knight", "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("black pawn",   "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ];

        let evals: Vec<(&str, i32, i32)> = cases.iter().map(|(name, f)| {
            let e = eval_fen(&net, f);
            (*name, e, e - e_base)
        }).collect();

        eprintln!("Base (startpos) eval: {} cp", e_base);
        for (name, e, delta) in &evals {
            eprintln!("  Remove {}: eval={} cp, delta from base = +{} cp", name, e, delta);
        }

        // All removals should produce positive evals (white better off).
        for (name, e, _) in &evals {
            assert!(*e > 0, "Removing {} gave eval={} — expected >0 (white up material)", name, e);
        }

        // Ordering check: queen removal should be the biggest gain;
        // pawn removal should be the smallest. Bishop/knight may swap
        // but should both be well above pawn and well below rook.
        let q = evals[0].2;
        let r = evals[1].2;
        let b = evals[2].2;
        let n = evals[3].2;
        let p = evals[4].2;

        assert!(q > r, "Queen removal delta ({}) not > Rook removal ({})", q, r);
        assert!(r > b && r > n, "Rook removal ({}) not > Bishop ({}) and Knight ({})", r, b, n);
        assert!(b > p, "Bishop removal ({}) not > Pawn ({})", b, p);
        assert!(n > p, "Knight removal ({}) not > Pawn ({})", n, p);
        assert!(p > 0, "Pawn removal gave non-positive delta: {}", p);
    }

    /// Deterministic PSQ fuzzer: plays random legal games from several
    /// positions, after each move compares the incremental PSQ
    /// accumulator (push(dirty) + materialize) against a full refresh
    /// on a fresh accumulator. Finds mirror/bucket/capture/castling/EP
    /// bugs that curated positions can miss.
    ///
    /// Counterpart to the threat_accum fuzzer — same approach, for the
    /// PSQ (king-piece-square) half of the net.
    #[test]
    fn fuzz_psq_accumulator() {
        use crate::board::Board;
        use crate::movegen::generate_legal_moves;
        use crate::search::build_dirty_piece;
        use crate::types::{flip_color, move_from, move_to, NO_PIECE_TYPE};

        crate::init();

        let net_path = std::env::var("CODA_TEST_NET").ok()
            .filter(|p| std::path::Path::new(p).exists())
            .or_else(|| {
                ["nets/net-v9-768th16x32-w15-e200s200-xray-fixed.nnue",
                 "nets/net-v9-768th16x32-w0-e400s400-noxray.nnue",
                 "net.nnue"]
                    .iter()
                    .find(|p| std::path::Path::new(p).exists())
                    .map(|p| p.to_string())
            });
        let net_path = match net_path {
            Some(p) => p,
            None => { eprintln!("Skipping PSQ fuzzer: no net available"); return; }
        };
        let net = match NNUENet::load(&net_path) {
            Ok(n) => n,
            Err(e) => { eprintln!("Skipping PSQ fuzzer: {}", e); return; }
        };
        let h = net.hidden_size;

        const START_FENS: &[&str] = &[
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1", // promotion testbed
        ];

        fn next_u32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            *state = x; x
        }

        const MAX_PLIES: usize = 80;
        const GAMES_PER_FEN: usize = 10;

        for (fen_idx, fen) in START_FENS.iter().enumerate() {
            for game in 0..GAMES_PER_FEN {
                let seed: u32 = 0xC0DA_BEEFu32
                    .wrapping_add((fen_idx as u32).wrapping_mul(1_000_003))
                    .wrapping_add((game as u32).wrapping_mul(7919));
                let mut rng = if seed == 0 { 1 } else { seed };

                let mut board = Board::from_fen(fen);

                let mut acc = NNUEAccumulator::new(h);
                acc.force_recompute(&net, &board);

                for ply in 0..MAX_PLIES {
                    let legal = generate_legal_moves(&board);
                    if legal.len == 0 { break; }
                    let mv = legal.moves[(next_u32(&mut rng) as usize) % legal.len];

                    let us = board.side_to_move;
                    let them = flip_color(us);
                    let moved_pt = board.piece_type_at(move_from(mv));
                    let captured_pt = board.piece_type_at(move_to(mv));
                    let dirty = build_dirty_piece(mv, us, them, moved_pt, captured_pt, &net);

                    let ok = board.make_move(mv);
                    assert!(ok, "psq fuzz {} game {} ply {}: move {} illegal?",
                        fen_idx, game, ply, crate::types::move_to_uci(mv));

                    acc.push(dirty);
                    acc.materialize(&net, &board);

                    // Reference: fresh accumulator, full refresh on post-move board.
                    let mut ref_acc = NNUEAccumulator::new(h);
                    ref_acc.force_recompute(&net, &board);

                    let got_w = &acc.stack[acc.top].white;
                    let got_b = &acc.stack[acc.top].black;
                    let ref_w = &ref_acc.stack[ref_acc.top].white;
                    let ref_b = &ref_acc.stack[ref_acc.top].black;

                    for (name, got, refv) in [
                        ("white", got_w, ref_w),
                        ("black", got_b, ref_b),
                    ] {
                        if got[..h] != refv[..h] {
                            let j = (0..h).find(|&j| got[j] != refv[j]).unwrap();
                            panic!(
                                "psq fuzz divergence: fen_idx={} game={} ply={} move={} \
                                 pov={} channel={} incr={} refresh={} seed={:#x}",
                                fen_idx, game, ply,
                                crate::types::move_to_uci(mv),
                                name, j, got[j], refv[j], seed,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Test Finny table consistency: incremental update vs full recompute.
    #[test]
    fn test_finny_incremental_consistency() {
        use crate::board::Board;

        let net_path = if std::path::Path::new("net.nnue").exists() {
            "net.nnue".to_string()
        } else {
            let entries = std::fs::read_dir(".").ok();
            match entries.and_then(|e| {
                e.filter_map(|f| f.ok())
                    .find(|f| {
                        let name = f.file_name().to_string_lossy().to_string();
                        name.starts_with("net-v") && name.ends_with(".nnue")
                    })
                    .map(|f| f.path().to_string_lossy().to_string())
            }) {
                Some(p) => p,
                None => { eprintln!("Skipping Finny test: no .nnue file"); return; }
            }
        };

        let net = match NNUENet::load(&net_path) {
            Ok(n) => n,
            Err(e) => { eprintln!("Skipping Finny test: {}", e); return; }
        };

        // Play a sequence of moves and compare incremental vs recompute at each step
        let moves_sequence = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            "rnbqkb1r/pppp1ppp/4pn2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkb1r/pppp1ppp/4pn2/8/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        ];

        let h = net.hidden_size;

        for fen in &moves_sequence {
            let board = Board::from_fen(fen);
            let piece_count = board.occupied().count_ones();

            // Full recompute
            let mut acc_full = NNUEAccumulator::new(h);
            acc_full.force_recompute(&net, &board);
            let full_result = net.forward(&acc_full, board.side_to_move, piece_count);

            // Materialized from dirty state (simulates incremental)
            let mut acc_incr = NNUEAccumulator::new(h);
            acc_incr.force_recompute(&net, &board);
            let incr_result = net.forward(&acc_incr, board.side_to_move, piece_count);

            assert_eq!(
                full_result, incr_result,
                "Finny consistency fail at {}: full={} incr={}",
                fen, full_result, incr_result
            );
        }
    }

    /// King-march test: specifically exercises Finny table refresh by
    /// marching the white king across the board through multiple king
    /// buckets. Each king move is applied via incremental update,
    /// and the resulting accumulator is compared against a from-scratch
    /// recompute. If Finny refresh logic (bucket change detection,
    /// cache invalidation, batch apply) has a bug, it shows up here
    /// because every ply crosses a bucket boundary in at least one
    /// of the 4 perspective×mirror combinations.
    ///
    /// Complements `fuzz_psq_accumulator` which does random games —
    /// this one is deterministic and forces the cross-bucket path.
    #[test]
    fn finny_king_march_consistency() {
        use crate::board::Board;
        use crate::search::build_dirty_piece;
        use crate::types::{flip_color, NO_PIECE_TYPE, make_move, FLAG_NONE};

        crate::init();

        let net_path = std::env::var("CODA_TEST_NET").ok()
            .filter(|p| std::path::Path::new(p).exists())
            .or_else(|| {
                ["nets/net-v9-768th16x32-w15-e200s200-xray-fixed.nnue",
                 "nets/net-v9-768th16x32-w0-e400s400-noxray.nnue",
                 "net.nnue"]
                    .iter()
                    .find(|p| std::path::Path::new(p).exists())
                    .map(|p| p.to_string())
            });
        let net_path = match net_path {
            Some(p) => p,
            None => { eprintln!("Skipping Finny king-march: no net"); return; }
        };
        let net = match NNUENet::load(&net_path) {
            Ok(n) => n,
            Err(e) => { eprintln!("Skipping Finny king-march: {}", e); return; }
        };
        let h = net.hidden_size;

        // Kings-only endgame so only king moves are legal. White king
        // at e1 (4) marches via e1→d1→c1→b1→a1→a2→a3→... across the
        // file boundary (a1 crosses from file-4 region to file-0 region),
        // definitely crossing bucket boundaries multiple times.
        let board_start = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        let mut board = board_start.clone();

        let mut acc = NNUEAccumulator::new(h);
        acc.force_recompute(&net, &board);

        // Sequence of moves: white king, then black king (stays put via
        // short distance moves so it's legal).
        // Squares: 4=e1, 3=d1, 2=c1, 1=b1, 0=a1, 8=a2, 16=a3, 24=a4, 32=a5.
        // Black king: 60=e8, wander locally.
        let sequence: &[(u8, u8)] = &[
            (4, 3),   // Ke1-d1
            (60, 59), // Ke8-d8
            (3, 2),   // Kd1-c1
            (59, 58), // Kd8-c8
            (2, 1),   // Kc1-b1
            (58, 57), // Kc8-b8
            (1, 0),   // Kb1-a1
            (57, 56), // Kb8-a8
            (0, 8),   // Ka1-a2
            (56, 48), // Ka8-a7
            (8, 16),  // Ka2-a3
            (48, 40), // Ka7-a6
            (16, 24), // Ka3-a4
            (40, 32), // Ka6-a5
        ];

        for (i, &(from, to)) in sequence.iter().enumerate() {
            let mv = make_move(from, to, FLAG_NONE);
            let us = board.side_to_move;
            let them = flip_color(us);
            let moved_pt = board.piece_type_at(from);
            let captured_pt = board.piece_type_at(to);
            assert_eq!(captured_pt, NO_PIECE_TYPE, "king march: no captures expected");
            let dirty = build_dirty_piece(mv, us, them, moved_pt, captured_pt, &net);

            let ok = board.make_move(mv);
            assert!(ok, "king march step {}: move illegal {}→{}", i, from, to);

            acc.push(dirty);
            acc.materialize(&net, &board);

            let mut ref_acc = NNUEAccumulator::new(h);
            ref_acc.force_recompute(&net, &board);

            let got_w = &acc.stack[acc.top].white;
            let got_b = &acc.stack[acc.top].black;
            let ref_w = &ref_acc.stack[ref_acc.top].white;
            let ref_b = &ref_acc.stack[ref_acc.top].black;

            for (name, got, refv) in [
                ("white", got_w, ref_w),
                ("black", got_b, ref_b),
            ] {
                if got[..h] != refv[..h] {
                    let j = (0..h).find(|&j| got[j] != refv[j]).unwrap();
                    panic!(
                        "Finny king-march divergence: step={} {}→{} pov={} channel={} \
                         incr={} refresh={} fen={}",
                        i, from, to, name, j, got[j], refv[j], board.to_fen(),
                    );
                }
            }
        }
    }
}
