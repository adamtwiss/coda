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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_crelu_dot(acc: &[i16], weights: &[i16], h: usize) -> i64 {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let mut sum0 = _mm256_setzero_si256(); // accumulate i32

    let mut i = 0;
    while i < h {
        // Load 16 accumulator values and clamp to [0, 255]
        let v = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);

        // Load 16 output weights
        let w = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);

        // VPMADDWD: multiply pairs of i16 and add adjacent to i32
        // (clamped[0]*w[0] + clamped[1]*w[1]), (clamped[2]*w[2] + clamped[3]*w[3]), ...
        let prod = _mm256_madd_epi16(clamped, w);
        sum0 = _mm256_add_epi32(sum0, prod);

        i += 16;
    }

    // Horizontal sum of 8 × i32 → i64
    hsum_epi32_to_i64(sum0)
}

/// SCReLU dot product: clamp acc to [0, QA=255], square, dot with output weights.
/// Returns i64 sum at scale QA² × QB. acc and weights have length h.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_screlu_dot(acc: &[i16], weights: &[i16], h: usize) -> i64 {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let mut sum_lo = _mm256_setzero_si256(); // i64 accumulator (low)
    let mut sum_hi = _mm256_setzero_si256(); // i64 accumulator (high)

    let mut i = 0;
    while i < h {
        // Load and clamp to [0, 255]
        let v = _mm256_loadu_si256(acc.as_ptr().add(i) as *const __m256i);
        let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);

        // Square: v² — since values are [0, 255], v² fits in i16 (max 65025) — no, it doesn't!
        // 255² = 65025 which overflows i16 (max 32767). We need i32 for the squares.
        // Strategy: unpack to i32, square, multiply by weights (also unpacked), accumulate as i64.

        let w = _mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i);

        // Unpack low 8 values to i32
        let v_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(clamped));
        let w_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
        let v2_lo = _mm256_mullo_epi32(v_lo, v_lo); // v²
        let prod_lo = _mm256_mullo_epi32(v2_lo, w_lo); // v² × w (i32, fits: 65025 * 32767 ≈ 2.1B < 2³¹)
        // Wait — 65025 * weight where weight can be negative and |weight| up to 32767
        // 65025 * 32767 = 2,130,964,575 which just fits in i32 (max 2,147,483,647). Tight but OK.
        // Actually if weight is negative: 65025 * -32768 = -2,131,722,240 which fits in i32 (min -2,147,483,648). Also OK.

        // Accumulate into i64 to avoid overflow across many elements
        // Sign-extend i32 to i64
        let prod_lo_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod_lo));
        let prod_lo_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod_lo, 1));
        sum_lo = _mm256_add_epi64(sum_lo, prod_lo_lo);
        sum_hi = _mm256_add_epi64(sum_hi, prod_lo_hi);

        // Unpack high 8 values to i32
        let v_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(clamped, 1));
        let w_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));
        let v2_hi = _mm256_mullo_epi32(v_hi, v_hi);
        let prod_hi = _mm256_mullo_epi32(v2_hi, w_hi);

        let prod_hi_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod_hi));
        let prod_hi_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod_hi, 1));
        sum_lo = _mm256_add_epi64(sum_lo, prod_hi_lo);
        sum_hi = _mm256_add_epi64(sum_hi, prod_hi_hi);

        i += 16;
    }

    // Horizontal sum of i64 accumulators
    let total = _mm256_add_epi64(sum_lo, sum_hi);
    hsum_epi64(total)
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

/// NNUE network weights (shared, read-only after loading).
pub struct NNUENet {
    pub hidden_size: usize,
    pub input_weights: Vec<i16>,  // [NNUE_INPUT_SIZE × hidden_size]
    pub input_biases: Vec<i16>,   // [hidden_size]
    pub output_weights: Vec<i16>, // [NNUE_OUTPUT_BUCKETS × 2 × hidden_size]
    pub output_bias: [i32; NNUE_OUTPUT_BUCKETS],
    pub use_screlu: bool,
    pub has_avx2: bool,
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

        let header_size: u64 = match version {
            5 => 8, // magic + version
            6 => {
                let flags = read_u8(&mut reader)?;
                use_screlu = flags & 1 != 0;
                9
            }
            _ => return Err(format!("unsupported NNUE version: {}", version)),
        };

        // Infer hidden size from file size
        let file_meta = std::fs::metadata(path).map_err(|e| format!("stat {}: {}", path, e))?;
        let file_size = file_meta.len();
        let body_size = file_size - header_size;

        // body = InputWeights(NNUE_INPUT_SIZE * H * 2) + InputBiases(H * 2)
        //      + OutputWeights(8 * 2 * H * 2) + OutputBias(8 * 4)
        // body = 2*H*(NNUE_INPUT_SIZE + 1 + 16) + 32
        // body - 32 = 2*H*(12288 + 17) = 2*H*12305
        let h_numer = body_size - 32;
        let h_denom = 2 * 12305;
        if h_numer % h_denom as u64 != 0 {
            return Err(format!("cannot infer hidden size from file size {} (body {})", file_size, body_size));
        }
        let hidden_size = (h_numer / h_denom as u64) as usize;

        // Read input weights
        let mut input_weights = vec![0i16; NNUE_INPUT_SIZE * hidden_size];
        read_i16_slice(&mut reader, &mut input_weights)?;

        // Read input biases
        let mut input_biases = vec![0i16; hidden_size];
        read_i16_slice(&mut reader, &mut input_biases)?;

        // Read output weights: 8 buckets × (2 × hidden_size)
        let out_width = 2 * hidden_size;
        let mut output_weights = vec![0i16; NNUE_OUTPUT_BUCKETS * out_width];
        read_i16_slice(&mut reader, &mut output_weights)?;

        // Read output bias
        let mut output_bias = [0i32; NNUE_OUTPUT_BUCKETS];
        for i in 0..NNUE_OUTPUT_BUCKETS {
            output_bias[i] = read_i32(&mut reader)?;
        }

        println!("info string Loaded NNUE v{} {} {} ({})",
            version, path, if use_screlu { "SCReLU" } else { "CReLU" }, hidden_size);

        let has_avx2 = detect_avx2();
        if has_avx2 {
            println!("info string AVX2 SIMD detected — using vectorised NNUE inference");
        }

        Ok(NNUENet {
            hidden_size,
            input_weights,
            input_biases,
            output_weights,
            output_bias,
            use_screlu,
            has_avx2,
        })
    }

    /// Get input weight row for a feature index.
    #[inline]
    fn input_weight_row(&self, idx: usize) -> &[i16] {
        let off = idx * self.hidden_size;
        &self.input_weights[off..off + self.hidden_size]
    }

    /// Get output weight row for a bucket.
    #[inline]
    fn output_weight_row(&self, bucket: usize) -> &[i16] {
        let w = 2 * self.hidden_size;
        let off = bucket * w;
        &self.output_weights[off..off + w]
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

        #[cfg(target_arch = "x86_64")]
        if self.has_avx2 && h % 16 == 0 {
            if self.use_screlu {
                // SAFETY: has_avx2 is true, h is multiple of 16
                unsafe {
                    output += simd_screlu_dot(stm_acc, &out_w[..h], h);
                    output += simd_screlu_dot(ntm_acc, &out_w[h..], h);
                }
                output /= QA as i64;
            } else {
                unsafe {
                    output += simd_crelu_dot(stm_acc, &out_w[..h], h);
                    output += simd_crelu_dot(ntm_acc, &out_w[h..], h);
                }
            }

            let mut result = (output as i32) * EVAL_SCALE / QAB;
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
        // centipawns = output * 400 / 16320
        let mut result = (output as i32) * EVAL_SCALE / QAB;

        // SCReLU scale correction: squared activation has wider dynamic range
        if self.use_screlu {
            result = result * 4 / 5;
        }

        result
    }
}

/// Single accumulator entry (two perspectives).
#[derive(Clone)]
struct AccEntry {
    white: Vec<i16>,
    black: Vec<i16>,
    computed: bool,
}

/// Accumulator stack for incremental NNUE updates.
/// Push on make_move, pop on unmake_move.
pub struct NNUEAccumulator {
    stack: Vec<AccEntry>,
    top: usize,
    hidden_size: usize,
}

impl NNUEAccumulator {
    pub fn new(hidden_size: usize) -> Self {
        let mut stack = Vec::with_capacity(256);
        for _ in 0..256 {
            stack.push(AccEntry {
                white: vec![0; hidden_size],
                black: vec![0; hidden_size],
                computed: false,
            });
        }
        NNUEAccumulator { stack, top: 0, hidden_size }
    }

    fn current(&self) -> &AccEntry {
        &self.stack[self.top]
    }

    fn current_mut(&mut self) -> &mut AccEntry {
        &mut self.stack[self.top]
    }

    pub fn white(&self) -> &[i16] {
        &self.stack[self.top].white[..self.hidden_size]
    }

    pub fn black(&self) -> &[i16] {
        &self.stack[self.top].black[..self.hidden_size]
    }

    /// Push a new entry (for make_move). Does NOT copy — caller must update or recompute.
    pub fn push(&mut self) {
        self.top += 1;
        if self.top >= self.stack.len() {
            self.stack.push(AccEntry {
                white: vec![0; self.hidden_size],
                black: vec![0; self.hidden_size],
                computed: false,
            });
        }
        self.stack[self.top].computed = false;
    }

    /// Pop (for unmake_move).
    pub fn pop(&mut self) {
        debug_assert!(self.top > 0);
        self.top -= 1;
    }

    /// Recompute both perspectives from scratch at the current stack position.
    pub fn recompute(&mut self, net: &NNUENet, board: &Board) {
        let h = self.hidden_size;
        let entry = &mut self.stack[self.top];

        entry.white[..h].copy_from_slice(&net.input_biases[..h]);
        entry.black[..h].copy_from_slice(&net.input_biases[..h]);

        let w_king_sq = board.king_sq(WHITE);
        let b_king_sq = board.king_sq(BLACK);

        #[cfg(target_arch = "x86_64")]
        let use_simd = net.has_avx2 && h % 16 == 0;

        for color in 0..2u8 {
            for pt in 0..6u8 {
                let mut bb = board.pieces[pt as usize] & board.colors[color as usize];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb) as u8;

                    let w_idx = halfka_index(WHITE, w_king_sq, color, pt, sq);
                    let w_row = net.input_weight_row(w_idx);

                    let b_idx = halfka_index(BLACK, b_king_sq, color, pt, sq);
                    let b_row = net.input_weight_row(b_idx);

                    #[cfg(target_arch = "x86_64")]
                    if use_simd {
                        unsafe {
                            simd_acc_add(&mut entry.white, w_row, h);
                            simd_acc_add(&mut entry.black, b_row, h);
                        }
                        continue;
                    }

                    for j in 0..h {
                        entry.white[j] += w_row[j];
                    }
                    for j in 0..h {
                        entry.black[j] += b_row[j];
                    }
                }
            }
        }

        entry.computed = true;
    }

    /// Incremental update: copy parent, then apply feature changes.
    /// Called after make_move. The board should be in the POST-move state.
    /// `changes` is a list of (add: bool, color, piece_type, square) tuples.
    pub fn update_incremental(
        &mut self,
        net: &NNUENet,
        board: &Board,
        changes: &[(bool, u8, u8, u8)], // (add, color, pt, sq)
    ) {
        let h = self.hidden_size;

        // Copy parent accumulator — parent must be computed
        let parent_idx = self.top - 1;
        if !self.stack[parent_idx].computed {
            // Parent not computed (e.g., after null move) — fall back to full recompute
            self.recompute(net, board);
            return;
        }

        // Split borrow: copy parent to current
        let (left, right) = self.stack.split_at_mut(self.top);
        let parent = &left[parent_idx];
        let current = &mut right[0];

        current.white[..h].copy_from_slice(&parent.white[..h]);
        current.black[..h].copy_from_slice(&parent.black[..h]);

        let w_king_sq = board.king_sq(WHITE);
        let b_king_sq = board.king_sq(BLACK);

        #[cfg(target_arch = "x86_64")]
        let use_simd = net.has_avx2 && h % 16 == 0;

        for &(add, color, pt, sq) in changes {
            // White perspective
            let w_idx = halfka_index(WHITE, w_king_sq, color, pt, sq);
            let w_row = net.input_weight_row(w_idx);

            // Black perspective
            let b_idx = halfka_index(BLACK, b_king_sq, color, pt, sq);
            let b_row = net.input_weight_row(b_idx);

            #[cfg(target_arch = "x86_64")]
            if use_simd {
                // SAFETY: has_avx2 is true, h is multiple of 16
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

            // Scalar fallback
            if add {
                for j in 0..h {
                    current.white[j] += w_row[j];
                    current.black[j] += b_row[j];
                }
            } else {
                for j in 0..h {
                    current.white[j] -= w_row[j];
                    current.black[j] -= b_row[j];
                }
            }
        }

        current.computed = true;
    }

    /// Ensure current entry is computed (recompute if needed).
    pub fn materialize(&mut self, net: &NNUENet, board: &Board) {
        if !self.stack[self.top].computed {
            self.recompute(net, board);
        }
    }

    /// Reset to bottom of stack.
    pub fn reset(&mut self) {
        self.top = 0;
        self.stack[0].computed = false;
    }
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
