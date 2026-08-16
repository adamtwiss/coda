//! L1 matmul kernels, input-chunk-major (dpbusd pattern).
//!
//! **Production path is the DENSE kernels** (`dense_l1_avx2`,
//! `dense_l1_avx512_vnni_l1_32`, etc.) — `select_l1_kernel` in `nnue.rs`
//! only ever dispatches to these. An earlier design rationale here claimed
//! "~89% sparsity in pairwise outputs, processes only ~11% of the work"
//! and shipped zero-skip sparse kernels (`sparse_l1_avx2`,
//! `sparse_l1_avx512_vnni`, `find_nnz_chunks4`) to exploit it — that
//! density figure was stale. Re-measured: the pairwise input is
//! actually ~58% nonzero (L1=16) / ~60% (L1=32), and a proper
//! branch-free find_nnz+list sparse kernel benched **1.8–2.4× SLOWER**
//! than dense at every density tested. L1 is too small (16-32 neurons) for
//! the zero-skip to pay for its own detection cost. See `nnue.rs`'s
//! `DenseAvx2` arm comment for more. The sparse kernels in this file are kept
//! only because their tests (`sparse_l1_scalar` as the reference oracle)
//! still exercise them — they have no production call site.
//!
//! Requires input-chunk-major weight layout:
//!   [input_chunk][neuron * 4] instead of [neuron][input]
//!
//! The dpbusd kernel: for each (zero-skipped, in the sparse variants; all,
//! in dense) 4-byte input chunk, splat it across an AVX2 register and
//! VPMADDUBSW with the weights for all neurons.

/// Transpose L1 weights from neuron-major to input-chunk-major layout.
///
/// Input layout (l1_weights_8t for pairwise):
///   STM block: [neuron * per_perspective + stm_input] for first num_neurons * per_perspective entries
///   NTM block: [num_neurons * per_perspective + neuron * per_perspective + ntm_input]
///   Where per_perspective = pw = hidden_size / 2
///
/// Output layout:
///   [chunk * num_neurons * 4 + neuron * 4 + byte_in_chunk]
///   Chunks 0..pw/4 are STM, chunks pw/4..pw/2 are NTM.
pub fn transpose_weights_for_sparse(
    weights_8t: &[i8],
    total_input: usize,  // pw * 2 (both perspectives)
    num_neurons: usize,
) -> Vec<i8> {
    let per_persp = total_input / 2; // pw
    let num_chunks = total_input / 4; // total chunks for both perspectives
    let mut sparse = vec![0i8; num_chunks * num_neurons * 4];

    let ntm_offset = num_neurons * per_persp; // start of NTM block in weights_8t

    for chunk in 0..num_chunks {
        let is_ntm = chunk >= per_persp / 4;
        let local_chunk = if is_ntm { chunk - per_persp / 4 } else { chunk };

        for neuron in 0..num_neurons {
            for byte in 0..4 {
                let input_idx = local_chunk * 4 + byte;
                let src = if is_ntm {
                    ntm_offset + neuron * per_persp + input_idx
                } else {
                    neuron * per_persp + input_idx
                };
                let dst = chunk * num_neurons * 4 + neuron * 4 + byte;
                if src < weights_8t.len() {
                    sparse[dst] = weights_8t[src];
                }
            }
        }
    }

    sparse
}

/// Find non-zero 4-byte chunks in a u8 array.
/// Returns the number of non-zero chunks found.
/// nnz_indices is filled with the chunk indices.
#[inline]
pub fn find_nnz_chunks4(data: &[u8], len: usize, nnz_indices: &mut [u16]) -> usize {
    let chunks = len / 4;
    // read_unaligned: a &[u8] only guarantees align 1, so materialising a
    // &[u32] over it (the previous code) was UB. Compiles to the same
    // plain `mov` on x86 either way.
    let p = data.as_ptr() as *const u32;
    let mut count = 0;
    for i in 0..chunks {
        if unsafe { p.add(i).read_unaligned() } != 0 {
            nnz_indices[count] = i as u16;
            count += 1;
        }
    }
    count
}

/// Sparse L1 matmul: process only non-zero input chunks.
/// Scalar reference implementation for testing.
pub fn sparse_l1_scalar(
    stm_pw: &[u8],       // STM pairwise output
    ntm_pw: &[u8],       // NTM pairwise output
    pw: usize,           // pairwise size per perspective
    sparse_weights: &[i8], // input-chunk-major weights
    num_neurons: usize,   // L1 neurons
    bias: &[i16],         // L1 biases
    bias_scale: i32,      // PW_SCALE
    output: &mut [i32],   // L1 pre-activations
) {
    let chunk_stride = num_neurons * 4; // bytes per input chunk in weight table

    // Initialize with biases
    for i in 0..num_neurons {
        output[i] = bias[i] as i32 * bias_scale;
    }

    // STM perspective: chunks 0..pw/4. read_unaligned because &[u8] only
    // guarantees align 1 (creating a &[u32] view over it is UB).
    let stm_chunks = stm_pw.as_ptr() as *const u32;
    for chunk_idx in 0..pw / 4 {
        if unsafe { stm_chunks.add(chunk_idx).read_unaligned() } == 0 { continue; }
        let w_base = chunk_idx * chunk_stride;
        for neuron in 0..num_neurons {
            let w_off = w_base + neuron * 4;
            for byte in 0..4 {
                output[neuron] += stm_pw[chunk_idx * 4 + byte] as i32
                    * sparse_weights[w_off + byte] as i32;
            }
        }
    }

    // NTM perspective: chunks pw/4..pw*2/4
    let ntm_chunk_offset = pw / 4;
    let ntm_chunks = ntm_pw.as_ptr() as *const u32;
    for chunk_idx in 0..pw / 4 {
        if unsafe { ntm_chunks.add(chunk_idx).read_unaligned() } == 0 { continue; }
        let w_base = (ntm_chunk_offset + chunk_idx) * chunk_stride;
        for neuron in 0..num_neurons {
            let w_off = w_base + neuron * 4;
            for byte in 0..4 {
                output[neuron] += ntm_pw[chunk_idx * 4 + byte] as i32
                    * sparse_weights[w_off + byte] as i32;
            }
        }
    }
}

/// Sparse L1 matmul with AVX2 dpbusd.
/// For each non-zero input chunk, splat it and VPMADDUBSW with neuron weights.
///
/// # Safety
/// CPU must support AVX2. Slices must be sized for `pw`/`num_neurons`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sparse_l1_avx2(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],
    num_neurons: usize,
    bias: &[i16],
    bias_scale: i32,
    output: &mut [i32],
) {
    use std::arch::x86_64::*;

    let chunk_stride = num_neurons * 4;
    let ones = _mm256_set1_epi16(1);

    // Initialize with biases
    for i in 0..num_neurons { output[i] = bias[i] as i32 * bias_scale; }

    // Accumulate in AVX2 registers (8 neurons per register)
    // For L1=16: 2 registers (neurons 0-7 and 8-15)
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();

    let w_ptr = sparse_weights.as_ptr();

    // STM perspective
    let stm_chunks = stm_pw.as_ptr() as *const u32;
    for chunk_idx in 0..pw / 4 {
        let val = stm_chunks.add(chunk_idx).read_unaligned();
        if val == 0 { continue; }
        let input = _mm256_set1_epi32(val as i32);
        let w_off = chunk_idx * chunk_stride;

        // Neurons 0-7: 32 bytes of weights
        let w0 = _mm256_loadu_si256(w_ptr.add(w_off) as *const __m256i);
        let prod0 = _mm256_maddubs_epi16(input, w0);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod0, ones));

        // Neurons 8-15: next 32 bytes
        if num_neurons > 8 {
            let w1 = _mm256_loadu_si256(w_ptr.add(w_off + 32) as *const __m256i);
            let prod1 = _mm256_maddubs_epi16(input, w1);
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(prod1, ones));
        }
    }

    // NTM perspective
    let ntm_chunk_offset = pw / 4;
    let ntm_chunks = ntm_pw.as_ptr() as *const u32;
    for chunk_idx in 0..pw / 4 {
        let val = ntm_chunks.add(chunk_idx).read_unaligned();
        if val == 0 { continue; }
        let input = _mm256_set1_epi32(val as i32);
        let w_off = (ntm_chunk_offset + chunk_idx) * chunk_stride;

        let w0 = _mm256_loadu_si256(w_ptr.add(w_off) as *const __m256i);
        let prod0 = _mm256_maddubs_epi16(input, w0);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod0, ones));

        if num_neurons > 8 {
            let w1 = _mm256_loadu_si256(w_ptr.add(w_off + 32) as *const __m256i);
            let prod1 = _mm256_maddubs_epi16(input, w1);
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(prod1, ones));
        }
    }

    // Store accumulated results
    let mut results = [0i32; 16];
    _mm256_storeu_si256(results.as_mut_ptr() as *mut __m256i, acc0);
    _mm256_storeu_si256(results.as_mut_ptr().add(8) as *mut __m256i, acc1);
    for i in 0..num_neurons {
        output[i] += results[i];
    }
}

/// Dense column-major L1 matmul, AVX2, specialised to L1=32.
///
/// Mirrors `dense_l1_avx2`'s outer chunk loop but with FOUR YMM
/// accumulators (neurons 0-7, 8-15, 16-23, 24-31) instead of two.
/// Per chunk:
///   - one VPBROADCASTD of the 4-byte input chunk
///   - four VMOVDQU loads of weights (32 bytes each = 8 neurons of weights)
///   - four (VPMADDUBSW + VPMADDWD + VPADDD) sequences
///
/// `chunk_stride = 32 * 4 = 128 bytes` (vs 64 at L1=16). Each chunk's
/// weights occupy 4 contiguous YMM lanes in input-chunk-major layout.
///
/// Specialised because the existing `dense_l1_avx2` hardcodes 2
/// accumulators (`if num_neurons > 8 { ... }` for the second block).
/// Generalising it to a runtime-variable accumulator count would lose
/// the const-folded loads. A dedicated L1=32 variant keeps the inner
/// loop straight-line.
///
/// # Safety
/// CPU must support AVX2. Slices must be sized for `pw`; `output` must point to
/// at least 32 writable i32s.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dense_l1_avx2_l1_32(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],  // input-chunk-major layout (same as sparse_l1_avx2)
    bias: &[i16],
    bias_scale: i32,
    output: *mut i32,
) {
    use std::arch::x86_64::*;

    const NUM_NEURONS: usize = 32;
    const CHUNK_STRIDE: usize = NUM_NEURONS * 4; // = 128 bytes per chunk
    let ones = _mm256_set1_epi16(1);

    let mut acc0 = _mm256_setzero_si256(); // neurons 0-7
    let mut acc1 = _mm256_setzero_si256(); // neurons 8-15
    let mut acc2 = _mm256_setzero_si256(); // neurons 16-23
    let mut acc3 = _mm256_setzero_si256(); // neurons 24-31

    let w_ptr = sparse_weights.as_ptr();
    let total_chunks = pw / 4;

    macro_rules! accumulate_perspective {
        ($chunks:expr, $chunk_offset:expr) => {{
            let chunks: *const u32 = $chunks;
            let chunk_offset: usize = $chunk_offset;
            for c in 0..total_chunks {
                let val = chunks.add(c).read_unaligned();
                let input = _mm256_set1_epi32(val as i32);
                let w_off = (chunk_offset + c) * CHUNK_STRIDE;
                let w0 = _mm256_loadu_si256(w_ptr.add(w_off)       as *const __m256i);
                let w1 = _mm256_loadu_si256(w_ptr.add(w_off + 32)  as *const __m256i);
                let w2 = _mm256_loadu_si256(w_ptr.add(w_off + 64)  as *const __m256i);
                let w3 = _mm256_loadu_si256(w_ptr.add(w_off + 96)  as *const __m256i);
                let p0 = _mm256_maddubs_epi16(input, w0);
                let p1 = _mm256_maddubs_epi16(input, w1);
                let p2 = _mm256_maddubs_epi16(input, w2);
                let p3 = _mm256_maddubs_epi16(input, w3);
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(p2, ones));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(p3, ones));
            }
        }};
    }

    let stm_ptr = stm_pw.as_ptr() as *const u32;
    accumulate_perspective!(stm_ptr, 0);
    let ntm_ptr = ntm_pw.as_ptr() as *const u32;
    accumulate_perspective!(ntm_ptr, pw / 4);

    let scale = _mm256_set1_epi32(bias_scale);
    let b0 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr() as *const __m128i)),
        scale,
    );
    let b1 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(8) as *const __m128i)),
        scale,
    );
    let b2 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(16) as *const __m128i)),
        scale,
    );
    let b3 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(24) as *const __m128i)),
        scale,
    );
    let out_ptr = output;
    _mm256_storeu_si256(out_ptr as *mut __m256i, _mm256_add_epi32(acc0, b0));
    _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, _mm256_add_epi32(acc1, b1));
    _mm256_storeu_si256(out_ptr.add(16) as *mut __m256i, _mm256_add_epi32(acc2, b2));
    _mm256_storeu_si256(out_ptr.add(24) as *mut __m256i, _mm256_add_epi32(acc3, b3));
}

/// Load-time saturation gate for `dense_l1_avx2_l1_32_x2`.
///
/// Checks, in the same input-chunk-major layout the kernel reads, that for
/// every aligned chunk pair, output neuron, and maddubs lane, the fused
/// i16 lane cannot exceed ±32767 at the maximum input value (127, the
/// pairwise activation's upper bound): `127 × Σ|w| ≤ 32767` over the 4
/// co-lane weights, i.e. `Σ|w| ≤ 258`. Runs once at net load; O(weights).
///
/// Production nets have measured worst lane sums around 163 — safe with
/// margin. A net that violates the bound falls back to the unfused kernel
/// automatically via kernel selection.
pub fn x2_fusion_safe(sparse_weights: &[i8], num_neurons: usize) -> bool {
    let chunk_stride = num_neurons * 4;
    if chunk_stride == 0 || !sparse_weights.len().is_multiple_of(chunk_stride) {
        return false;
    }
    let chunks = sparse_weights.len() / chunk_stride;
    if !chunks.is_multiple_of(2) {
        return false;
    }
    for cp in (0..chunks).step_by(2) {
        for j in 0..num_neurons {
            let a = cp * chunk_stride + j * 4;
            let b = a + chunk_stride;
            for lane in 0..2 {
                let s = (sparse_weights[a + 2 * lane] as i32).abs()
                    + (sparse_weights[a + 2 * lane + 1] as i32).abs()
                    + (sparse_weights[b + 2 * lane] as i32).abs()
                    + (sparse_weights[b + 2 * lane + 1] as i32).abs();
                if 127 * s > i16::MAX as i32 {
                    return false;
                }
            }
        }
    }
    true
}

/// `dense_l1_avx2_l1_32` with the maddubs-pair ("double dpbusd") fusion:
/// two input chunks per iteration, summing their VPMADDUBSW products with
/// one VPADDW *before* a single shared VPMADDWD — halving the madd count
/// (4/6 reference engines: Berserk
/// `m256_add_dpbusd_epi32_x2`, PlentyChess, Alexandria).
///
/// # Saturation precondition (why this kernel is gated)
/// VPMADDUBSW saturates its i16 lane at ±32767, and the fusing VPADDW
/// wraps. The fused lane sums FOUR u8×i8 products (two per chunk), so it
/// is exact iff, for every output neuron and every aligned chunk pair,
/// `max_input(127) × Σ|w|` over the 4 co-lane weights stays ≤ 32767 —
/// i.e. `Σ|w| ≤ 258`. The caller MUST verify this at net-load time
/// (`NNUENet::x2_fusion_safe`) and fall back to `dense_l1_avx2_l1_32`
/// otherwise (current prod net E6C62000: worst lane sum 163 — safe).
/// Inputs are ≤127 by construction (pairwise output is (QA·QA)>>FT_SHIFT).
///
/// # Safety
/// CPU must support AVX2. Slices must be sized for `pw`; `pw` must be a
/// multiple of 8 (even chunk count); `output` must point to at least 32
/// writable i32s. Weights must satisfy the saturation precondition above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dense_l1_avx2_l1_32_x2(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],  // input-chunk-major layout (same as dense_l1_avx2_l1_32)
    bias: &[i16],
    bias_scale: i32,
    output: *mut i32,
) {
    use std::arch::x86_64::*;

    const NUM_NEURONS: usize = 32;
    const CHUNK_STRIDE: usize = NUM_NEURONS * 4; // = 128 bytes per chunk
    let ones = _mm256_set1_epi16(1);

    let mut acc0 = _mm256_setzero_si256(); // neurons 0-7
    let mut acc1 = _mm256_setzero_si256(); // neurons 8-15
    let mut acc2 = _mm256_setzero_si256(); // neurons 16-23
    let mut acc3 = _mm256_setzero_si256(); // neurons 24-31

    let w_ptr = sparse_weights.as_ptr();
    let total_chunks = pw / 4;
    debug_assert!(total_chunks.is_multiple_of(2), "x2 kernel needs even chunk count");

    macro_rules! accumulate_perspective {
        ($chunks:expr, $chunk_offset:expr) => {{
            let chunks: *const u32 = $chunks;
            let chunk_offset: usize = $chunk_offset;
            let mut c = 0;
            while c + 2 <= total_chunks {
                let in_a = _mm256_set1_epi32(chunks.add(c).read_unaligned() as i32);
                let in_b = _mm256_set1_epi32(chunks.add(c + 1).read_unaligned() as i32);
                let wa_off = (chunk_offset + c) * CHUNK_STRIDE;
                let wb_off = wa_off + CHUNK_STRIDE;
                let wa0 = _mm256_loadu_si256(w_ptr.add(wa_off)      as *const __m256i);
                let wa1 = _mm256_loadu_si256(w_ptr.add(wa_off + 32) as *const __m256i);
                let wa2 = _mm256_loadu_si256(w_ptr.add(wa_off + 64) as *const __m256i);
                let wa3 = _mm256_loadu_si256(w_ptr.add(wa_off + 96) as *const __m256i);
                let wb0 = _mm256_loadu_si256(w_ptr.add(wb_off)      as *const __m256i);
                let wb1 = _mm256_loadu_si256(w_ptr.add(wb_off + 32) as *const __m256i);
                let wb2 = _mm256_loadu_si256(w_ptr.add(wb_off + 64) as *const __m256i);
                let wb3 = _mm256_loadu_si256(w_ptr.add(wb_off + 96) as *const __m256i);
                let p0 = _mm256_add_epi16(_mm256_maddubs_epi16(in_a, wa0), _mm256_maddubs_epi16(in_b, wb0));
                let p1 = _mm256_add_epi16(_mm256_maddubs_epi16(in_a, wa1), _mm256_maddubs_epi16(in_b, wb1));
                let p2 = _mm256_add_epi16(_mm256_maddubs_epi16(in_a, wa2), _mm256_maddubs_epi16(in_b, wb2));
                let p3 = _mm256_add_epi16(_mm256_maddubs_epi16(in_a, wa3), _mm256_maddubs_epi16(in_b, wb3));
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(p2, ones));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(p3, ones));
                c += 2;
            }
        }};
    }

    let stm_ptr = stm_pw.as_ptr() as *const u32;
    accumulate_perspective!(stm_ptr, 0);
    let ntm_ptr = ntm_pw.as_ptr() as *const u32;
    accumulate_perspective!(ntm_ptr, pw / 4);

    let scale = _mm256_set1_epi32(bias_scale);
    let b0 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr() as *const __m128i)),
        scale,
    );
    let b1 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(8) as *const __m128i)),
        scale,
    );
    let b2 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(16) as *const __m128i)),
        scale,
    );
    let b3 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(24) as *const __m128i)),
        scale,
    );
    let out_ptr = output;
    _mm256_storeu_si256(out_ptr as *mut __m256i, _mm256_add_epi32(acc0, b0));
    _mm256_storeu_si256(out_ptr.add(8) as *mut __m256i, _mm256_add_epi32(acc1, b1));
    _mm256_storeu_si256(out_ptr.add(16) as *mut __m256i, _mm256_add_epi32(acc2, b2));
    _mm256_storeu_si256(out_ptr.add(24) as *mut __m256i, _mm256_add_epi32(acc3, b3));
}

/// `dense_l1_avx2` specialised to L1=16, with the same maddubs-pair
/// ("double dpbusd") fusion as `dense_l1_avx2_l1_32_x2`: two input chunks per
/// iteration, their VPMADDUBSW products summed with one VPADDW before a single
/// shared VPMADDWD — halving the madd and the accumulate count.
///
/// # Why this exists
/// Every VNNI tier already has a dedicated L1<=16 kernel
/// (`dense_l1_avx512_vnni`, `dense_l1_avx_vnni`), but the plain-AVX2 tier only
/// had L1=32 specialisations — both gated on `l1 == 32` exactly. When
/// production moved to L1=16, AVX2-without-VNNI hosts fell through to the
/// generic `dense_l1_avx2` and silently lost the fusion. That tier is not
/// exotic: it is what CCRL's 5950X runs.
///
/// Per chunk PAIR this issues 2 broadcasts, 4 loads, 4 maddubs, 2 addw,
/// 2 madd and 2 add, against the generic kernel's 2 broadcasts, 4 loads,
/// 4 maddubs, 4 madd and 4 add — trading 2 madd + 2 add for 2 addw.
///
/// # Saturation precondition (why this kernel is gated)
/// Identical to the L1=32 variant: VPMADDUBSW saturates at +/-32767 and the
/// fusing VPADDW wraps, so the fused lane's FOUR u8xi8 products are exact iff
/// `127 * sum|w| <= 32767` over the 4 co-lane weights, i.e. `sum|w| <= 258`.
/// `x2_fusion_safe` is width-generic and MUST be checked at net-load time with
/// `num_neurons = 16`; unsafe nets fall back to `dense_l1_avx2`.
///
/// # Safety
/// CPU must support AVX2. `num_neurons` is fixed at 16. `pw` must be a multiple
/// of 8 (even chunk count). `output` must point to at least 16 writable i32s.
/// Weights must satisfy the saturation precondition above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dense_l1_avx2_l1_16_x2(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],  // input-chunk-major layout (same as dense_l1_avx2)
    bias: &[i16],
    bias_scale: i32,
    output: *mut i32,
) {
    use std::arch::x86_64::*;

    const NUM_NEURONS: usize = 16;
    const CHUNK_STRIDE: usize = NUM_NEURONS * 4; // = 64 bytes per chunk
    let ones = _mm256_set1_epi16(1);

    let mut acc0 = _mm256_setzero_si256(); // neurons 0-7
    let mut acc1 = _mm256_setzero_si256(); // neurons 8-15

    let w_ptr = sparse_weights.as_ptr();
    let total_chunks = pw / 4;
    debug_assert!(total_chunks.is_multiple_of(2), "x2 kernel needs even chunk count");

    macro_rules! accumulate_perspective {
        ($chunks:expr, $chunk_offset:expr) => {{
            let chunks: *const u32 = $chunks;
            let chunk_offset: usize = $chunk_offset;
            let mut c = 0;
            while c + 2 <= total_chunks {
                let in_a = _mm256_set1_epi32(chunks.add(c).read_unaligned() as i32);
                let in_b = _mm256_set1_epi32(chunks.add(c + 1).read_unaligned() as i32);
                let wa_off = (chunk_offset + c) * CHUNK_STRIDE;
                let wb_off = wa_off + CHUNK_STRIDE;
                let wa0 = _mm256_loadu_si256(w_ptr.add(wa_off)      as *const __m256i);
                let wa1 = _mm256_loadu_si256(w_ptr.add(wa_off + 32) as *const __m256i);
                let wb0 = _mm256_loadu_si256(w_ptr.add(wb_off)      as *const __m256i);
                let wb1 = _mm256_loadu_si256(w_ptr.add(wb_off + 32) as *const __m256i);
                let p0 = _mm256_add_epi16(_mm256_maddubs_epi16(in_a, wa0), _mm256_maddubs_epi16(in_b, wb0));
                let p1 = _mm256_add_epi16(_mm256_maddubs_epi16(in_a, wa1), _mm256_maddubs_epi16(in_b, wb1));
                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(p0, ones));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(p1, ones));
                c += 2;
            }
        }};
    }

    let stm_ptr = stm_pw.as_ptr() as *const u32;
    accumulate_perspective!(stm_ptr, 0);
    let ntm_ptr = ntm_pw.as_ptr() as *const u32;
    accumulate_perspective!(ntm_ptr, pw / 4);

    let scale = _mm256_set1_epi32(bias_scale);
    let b0 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr() as *const __m128i)),
        scale,
    );
    let b1 = _mm256_mullo_epi32(
        _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(8) as *const __m128i)),
        scale,
    );
    _mm256_storeu_si256(output as *mut __m256i, _mm256_add_epi32(acc0, b0));
    _mm256_storeu_si256(output.add(8) as *mut __m256i, _mm256_add_epi32(acc1, b1));
}

/// Dense column-major L1 matmul: identical layout to sparse_l1_avx2 but
/// without the zero-chunk skip check. For pairwise-CReLU inputs where
/// most chunks are non-zero, the if-check overhead exceeds the skip
/// savings. Dense processing is straight-line SIMD: each 4-byte input
/// chunk contributes to all L1 neurons via splat_i32+dpbusd emulation
/// in one pass over the input.
///
/// Benefit vs row-major: input chunk loaded once per chunk (instead of
/// once per output), weights accessed sequentially in input-chunk-major
/// order (better cache behaviour than strided per-output rows).
///
/// # Safety
/// CPU must support AVX2. Slices must be sized for `pw`; `output` must point to
/// at least `num_neurons` writable i32s.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dense_l1_avx2(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],  // input-chunk-major layout (same as sparse_l1_avx2)
    num_neurons: usize,
    bias: &[i16],
    bias_scale: i32,
    output: *mut i32,
) {
    use std::arch::x86_64::*;

    let chunk_stride = num_neurons * 4;
    let ones = _mm256_set1_epi16(1);

    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();

    let w_ptr = sparse_weights.as_ptr();

    // STM perspective — all chunks, no zero-skip.
    let stm_chunks = stm_pw.as_ptr() as *const u32;
    for chunk_idx in 0..pw / 4 {
        let val = stm_chunks.add(chunk_idx).read_unaligned();
        let input = _mm256_set1_epi32(val as i32);
        let w_off = chunk_idx * chunk_stride;

        let w0 = _mm256_loadu_si256(w_ptr.add(w_off) as *const __m256i);
        let prod0 = _mm256_maddubs_epi16(input, w0);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod0, ones));

        if num_neurons > 8 {
            let w1 = _mm256_loadu_si256(w_ptr.add(w_off + 32) as *const __m256i);
            let prod1 = _mm256_maddubs_epi16(input, w1);
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(prod1, ones));
        }
    }

    // NTM perspective
    let ntm_chunk_offset = pw / 4;
    let ntm_chunks = ntm_pw.as_ptr() as *const u32;
    for chunk_idx in 0..pw / 4 {
        let val = ntm_chunks.add(chunk_idx).read_unaligned();
        let input = _mm256_set1_epi32(val as i32);
        let w_off = (ntm_chunk_offset + chunk_idx) * chunk_stride;

        let w0 = _mm256_loadu_si256(w_ptr.add(w_off) as *const __m256i);
        let prod0 = _mm256_maddubs_epi16(input, w0);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(prod0, ones));

        if num_neurons > 8 {
            let w1 = _mm256_loadu_si256(w_ptr.add(w_off + 32) as *const __m256i);
            let prod1 = _mm256_maddubs_epi16(input, w1);
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(prod1, ones));
        }
    }

    if num_neurons == 16 {
        let scale = _mm256_set1_epi32(bias_scale);
        let b0 = _mm256_mullo_epi32(
            _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr() as *const __m128i)),
            scale,
        );
        let b1 = _mm256_mullo_epi32(
            _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(8) as *const __m128i)),
            scale,
        );
        _mm256_storeu_si256(output as *mut __m256i, _mm256_add_epi32(acc0, b0));
        _mm256_storeu_si256(output.add(8) as *mut __m256i, _mm256_add_epi32(acc1, b1));
    } else {
        for i in 0..num_neurons { output.add(i).write(bias[i] as i32 * bias_scale); }
        let output = std::slice::from_raw_parts_mut(output, num_neurons);
        let mut results = [0i32; 16];
        _mm256_storeu_si256(results.as_mut_ptr() as *mut __m256i, acc0);
        _mm256_storeu_si256(results.as_mut_ptr().add(8) as *mut __m256i, acc1);
        for i in 0..num_neurons {
            output[i] += results[i];
        }
    }
}

/// Dense column-major L1 matmul, AVX-512 VNNI variant.
///
/// Same semantics as `dense_l1_avx2` but processes 16 neurons in a single
/// ZMM register via one `VPDPBUSD` per 4-byte input chunk. For L1=16
/// (v9 pairwise `num_neurons=16`), all neuron outputs fit in one ZMM
/// accumulator, so the loop body is:
///
///   load 64B weights → broadcast 4B input → VPDPBUSD
///
/// That's one load + one broadcast + one fused u8×i8 → i32 per chunk,
/// versus `dense_l1_avx2`'s two loads, two broadcasts, and six uops
/// (load + broadcast + maddubs + madd(ones) + add) per chunk. Net ~3×
/// fewer uops per chunk.
///
/// Only implemented for `num_neurons == 16` (the v9 pairwise case). For
/// other widths, callers should continue to use the non-VNNI paths.
///
/// # Safety
/// CPU must support AVX-512F/BW/VNNI. Slices must be sized for `pw`; `output`
/// must point to at least `num_neurons` writable i32s.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn dense_l1_avx512_vnni(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],  // input-chunk-major layout (same as sparse_l1_avx2)
    num_neurons: usize,
    bias: &[i16],
    bias_scale: i32,
    output: *mut i32,
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(num_neurons, 16, "dense_l1_avx512_vnni currently specialised to 16 neurons");

    let chunk_stride = num_neurons * 4; // = 64 bytes — exactly one ZMM

    // Four interleaved accumulators break the VPDPBUSD dependency chain
    // (4-cycle latency on Zen 5 / Sapphire Rapids). A single accumulator
    // serialises the whole loop; four keeps the dispatcher fed.
    let mut a0 = _mm512_setzero_si512();
    let mut a1 = _mm512_setzero_si512();
    let mut a2 = _mm512_setzero_si512();
    let mut a3 = _mm512_setzero_si512();

    let w_ptr = sparse_weights.as_ptr();
    let total_chunks = pw / 4;

    // Helper: process one perspective's worth of chunks into the four
    // rotating accumulators, with 4-at-a-time unrolling.
    macro_rules! run_perspective {
        ($chunks:expr, $chunk_offset:expr) => {{
            let chunks: *const u32 = $chunks;
            let chunk_offset: usize = $chunk_offset;
            let mut c = 0usize;
            while c + 4 <= total_chunks {
                let v0 = chunks.add(c).read_unaligned();
                let v1 = chunks.add(c + 1).read_unaligned();
                let v2 = chunks.add(c + 2).read_unaligned();
                let v3 = chunks.add(c + 3).read_unaligned();
                let w0 = _mm512_loadu_si512(w_ptr.add((chunk_offset + c) * chunk_stride) as *const __m512i);
                let w1 = _mm512_loadu_si512(w_ptr.add((chunk_offset + c + 1) * chunk_stride) as *const __m512i);
                let w2 = _mm512_loadu_si512(w_ptr.add((chunk_offset + c + 2) * chunk_stride) as *const __m512i);
                let w3 = _mm512_loadu_si512(w_ptr.add((chunk_offset + c + 3) * chunk_stride) as *const __m512i);
                a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(v0 as i32), w0);
                a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32(v1 as i32), w1);
                a2 = _mm512_dpbusd_epi32(a2, _mm512_set1_epi32(v2 as i32), w2);
                a3 = _mm512_dpbusd_epi32(a3, _mm512_set1_epi32(v3 as i32), w3);
                c += 4;
            }
            while c < total_chunks {
                let v = chunks.add(c).read_unaligned();
                let w = _mm512_loadu_si512(w_ptr.add((chunk_offset + c) * chunk_stride) as *const __m512i);
                a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(v as i32), w);
                c += 1;
            }
        }};
    }

    // STM chunks live at offsets [0..pw/4); NTM chunks at [pw/4..pw/2).
    let stm_chunks_ptr = stm_pw.as_ptr() as *const u32;
    run_perspective!(stm_chunks_ptr, 0);
    let ntm_chunks_ptr = ntm_pw.as_ptr() as *const u32;
    run_perspective!(ntm_chunks_ptr, pw / 4);

    let acc = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
    let bias_i32 = _mm512_cvtepi16_epi32(_mm256_loadu_si256(bias.as_ptr() as *const __m256i));
    let scaled_bias = _mm512_mullo_epi32(bias_i32, _mm512_set1_epi32(bias_scale));
    _mm512_storeu_si512(output as *mut __m512i, _mm512_add_epi32(acc, scaled_bias));
}

/// Dense column-major L1 matmul, AVX-512 VNNI variant, specialised to L1=32.
///
/// The column-major counterpart to `dense_l1_avx2_l1_32` (same
/// input-chunk-major weight layout, same outer loop) but emitting one
/// `VPDPBUSD` per neuron-group instead of the AVX2
/// `VPMADDUBSW + VPMADDWD + VPADDD` triple. 32 neurons = 32 i32 = **two
/// ZMM wide** (vs L1=16's single ZMM in `dense_l1_avx512_vnni`): per input
/// chunk the 32 neurons' weights occupy `32 * 4 = 128` contiguous bytes =
/// two ZMM loads — the low ZMM is neurons 0-15, the high ZMM neurons
/// 16-31.
///
/// Why this exists: on VNNI hardware (Zen5 / Sapphire Rapids) an L1=32 net
/// previously fell back to `RowMajorAvx512Vnni`, which re-scans the whole
/// input once *per neuron* (`simd512_l1_int8_dot_vnni` × 32). The
/// column-major form loads each input chunk once and feeds all 32 neurons,
/// the same locality win the L1=16 column-major kernel banks over its
/// row-major sibling. The AVX2-only fleet already had the column-major
/// `dense_l1_avx2_l1_32`; this is the missing VNNI twin.
///
/// Four interleaved accumulator PAIRS (lo0..lo3 for neurons 0-15, hi0..hi3
/// for neurons 16-31) break the VPDPBUSD dependency chain (4-cycle latency
/// on Zen 5). Eight ZMM accumulators + a couple of weight/input temps sit
/// well inside the 32-ZMM file.
///
/// # Safety
/// CPU must support AVX-512F/BW/VNNI. Slices must be sized for `pw`; `output`
/// must point to at least 32 writable i32s.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn dense_l1_avx512_vnni_l1_32(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],  // input-chunk-major layout (same as dense_l1_avx2_l1_32)
    bias: &[i16],
    bias_scale: i32,
    output: *mut i32,
) {
    use std::arch::x86_64::*;

    const NUM_NEURONS: usize = 32;
    const CHUNK_STRIDE: usize = NUM_NEURONS * 4; // = 128 bytes per chunk = 2 ZMM

    // Neurons 0-15 in the lo accumulators, 16-31 in the hi accumulators.
    // Four interleaved pairs hide the VPDPBUSD latency.
    let mut lo0 = _mm512_setzero_si512();
    let mut lo1 = _mm512_setzero_si512();
    let mut lo2 = _mm512_setzero_si512();
    let mut lo3 = _mm512_setzero_si512();
    let mut hi0 = _mm512_setzero_si512();
    let mut hi1 = _mm512_setzero_si512();
    let mut hi2 = _mm512_setzero_si512();
    let mut hi3 = _mm512_setzero_si512();

    let w_ptr = sparse_weights.as_ptr();
    let total_chunks = pw / 4;

    macro_rules! run_perspective {
        ($chunks:expr, $chunk_offset:expr) => {{
            let chunks: *const u32 = $chunks;
            let chunk_offset: usize = $chunk_offset;
            let mut c = 0usize;
            // 4-at-a-time: chunk c..c+3 feed accumulator pairs 0..3.
            while c + 4 <= total_chunks {
                macro_rules! step {
                    ($k:expr, $lo:ident, $hi:ident) => {{
                        let v = chunks.add(c + $k).read_unaligned();
                        let input = _mm512_set1_epi32(v as i32);
                        let base = (chunk_offset + c + $k) * CHUNK_STRIDE;
                        let w_lo = _mm512_loadu_si512(w_ptr.add(base) as *const __m512i);
                        let w_hi = _mm512_loadu_si512(w_ptr.add(base + 64) as *const __m512i);
                        $lo = _mm512_dpbusd_epi32($lo, input, w_lo);
                        $hi = _mm512_dpbusd_epi32($hi, input, w_hi);
                    }};
                }
                step!(0, lo0, hi0);
                step!(1, lo1, hi1);
                step!(2, lo2, hi2);
                step!(3, lo3, hi3);
                c += 4;
            }
            // Tail: remaining chunks fold into pair 0.
            while c < total_chunks {
                let v = chunks.add(c).read_unaligned();
                let input = _mm512_set1_epi32(v as i32);
                let base = (chunk_offset + c) * CHUNK_STRIDE;
                let w_lo = _mm512_loadu_si512(w_ptr.add(base) as *const __m512i);
                let w_hi = _mm512_loadu_si512(w_ptr.add(base + 64) as *const __m512i);
                lo0 = _mm512_dpbusd_epi32(lo0, input, w_lo);
                hi0 = _mm512_dpbusd_epi32(hi0, input, w_hi);
                c += 1;
            }
        }};
    }

    // STM chunks live at offsets [0..pw/4); NTM chunks at [pw/4..pw/2).
    let stm_chunks_ptr = stm_pw.as_ptr() as *const u32;
    run_perspective!(stm_chunks_ptr, 0);
    let ntm_chunks_ptr = ntm_pw.as_ptr() as *const u32;
    run_perspective!(ntm_chunks_ptr, pw / 4);

    let acc_lo = _mm512_add_epi32(_mm512_add_epi32(lo0, lo1), _mm512_add_epi32(lo2, lo3));
    let acc_hi = _mm512_add_epi32(_mm512_add_epi32(hi0, hi1), _mm512_add_epi32(hi2, hi3));

    let scale = _mm512_set1_epi32(bias_scale);
    let bias_lo = _mm512_mullo_epi32(
        _mm512_cvtepi16_epi32(_mm256_loadu_si256(bias.as_ptr() as *const __m256i)),
        scale,
    );
    let bias_hi = _mm512_mullo_epi32(
        _mm512_cvtepi16_epi32(_mm256_loadu_si256(bias.as_ptr().add(16) as *const __m256i)),
        scale,
    );
    _mm512_storeu_si512(output as *mut __m512i, _mm512_add_epi32(acc_lo, bias_lo));
    _mm512_storeu_si512(output.add(16) as *mut __m512i, _mm512_add_epi32(acc_hi, bias_hi));
}

/// Dense column-major L1 matmul, AVX-VNNI (YMM `VPDPBUSD`) variant,
/// specialised to L1=32. For hosts with AVX-VNNI but NOT full AVX-512
/// (Intel Alder Lake / Raptor Lake and similar) — there the AVX-512 L1=32
/// kernel can't run, and L1=32 otherwise falls to `dense_l1_avx2_l1_32`
/// (already column-major, same locality), so this kernel's only delta is
/// the fused inner op: one `VPDPBUSD` per neuron-group replaces AVX2's
/// `VPMADDUBSW + VPMADDWD(ones) + VPADDD` triple. Same column-major weight
/// layout as `dense_l1_avx2_l1_32`.
///
/// 32 neurons = four YMM groups (0-7, 8-15, 16-23, 24-31). AVX2/AVX-VNNI
/// has only 16 YMM registers (vs AVX-512's 32), so unlike the 512-bit
/// kernel's 8 accumulators this uses 8 = four groups × two interleaved
/// chunks (`even`/`odd`), enough to partly hide VPDPBUSD latency while
/// leaving registers for the broadcasts and folded weight loads.
///
/// # Safety
/// CPU must support AVX2 + AVX-VNNI. Slices must be sized for `pw`; `output`
/// must point to at least 32 writable i32s.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn dense_l1_avx_vnni_l1_32(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],  // input-chunk-major layout (same as dense_l1_avx2_l1_32)
    bias: &[i16],
    bias_scale: i32,
    output: *mut i32,
) {
    use std::arch::x86_64::*;

    const NUM_NEURONS: usize = 32;
    const CHUNK_STRIDE: usize = NUM_NEURONS * 4; // = 128 bytes per chunk = 4 YMM

    // Eight accumulators: four neuron groups × two interleaved chunk slots.
    let mut e0 = _mm256_setzero_si256(); // even chunk, neurons 0-7
    let mut e1 = _mm256_setzero_si256(); //             neurons 8-15
    let mut e2 = _mm256_setzero_si256(); //             neurons 16-23
    let mut e3 = _mm256_setzero_si256(); //             neurons 24-31
    let mut o0 = _mm256_setzero_si256(); // odd chunk,  neurons 0-7
    let mut o1 = _mm256_setzero_si256(); //             neurons 8-15
    let mut o2 = _mm256_setzero_si256(); //             neurons 16-23
    let mut o3 = _mm256_setzero_si256(); //             neurons 24-31

    let w_ptr = sparse_weights.as_ptr();
    let total_chunks = pw / 4;

    macro_rules! run_perspective {
        ($chunks:expr, $chunk_offset:expr) => {{
            let chunks: *const u32 = $chunks;
            let chunk_offset: usize = $chunk_offset;
            let mut c = 0usize;
            while c + 2 <= total_chunks {
                let ve = _mm256_set1_epi32(chunks.add(c).read_unaligned() as i32);
                let vo = _mm256_set1_epi32(chunks.add(c + 1).read_unaligned() as i32);
                let be = (chunk_offset + c) * CHUNK_STRIDE;
                let bo = (chunk_offset + c + 1) * CHUNK_STRIDE;
                e0 = _mm256_dpbusd_avx_epi32(e0, ve, _mm256_loadu_si256(w_ptr.add(be) as *const __m256i));
                e1 = _mm256_dpbusd_avx_epi32(e1, ve, _mm256_loadu_si256(w_ptr.add(be + 32) as *const __m256i));
                e2 = _mm256_dpbusd_avx_epi32(e2, ve, _mm256_loadu_si256(w_ptr.add(be + 64) as *const __m256i));
                e3 = _mm256_dpbusd_avx_epi32(e3, ve, _mm256_loadu_si256(w_ptr.add(be + 96) as *const __m256i));
                o0 = _mm256_dpbusd_avx_epi32(o0, vo, _mm256_loadu_si256(w_ptr.add(bo) as *const __m256i));
                o1 = _mm256_dpbusd_avx_epi32(o1, vo, _mm256_loadu_si256(w_ptr.add(bo + 32) as *const __m256i));
                o2 = _mm256_dpbusd_avx_epi32(o2, vo, _mm256_loadu_si256(w_ptr.add(bo + 64) as *const __m256i));
                o3 = _mm256_dpbusd_avx_epi32(o3, vo, _mm256_loadu_si256(w_ptr.add(bo + 96) as *const __m256i));
                c += 2;
            }
            // Tail: a single leftover chunk folds into the even slots.
            while c < total_chunks {
                let v = _mm256_set1_epi32(chunks.add(c).read_unaligned() as i32);
                let b = (chunk_offset + c) * CHUNK_STRIDE;
                e0 = _mm256_dpbusd_avx_epi32(e0, v, _mm256_loadu_si256(w_ptr.add(b) as *const __m256i));
                e1 = _mm256_dpbusd_avx_epi32(e1, v, _mm256_loadu_si256(w_ptr.add(b + 32) as *const __m256i));
                e2 = _mm256_dpbusd_avx_epi32(e2, v, _mm256_loadu_si256(w_ptr.add(b + 64) as *const __m256i));
                e3 = _mm256_dpbusd_avx_epi32(e3, v, _mm256_loadu_si256(w_ptr.add(b + 96) as *const __m256i));
                c += 1;
            }
        }};
    }

    let stm_chunks_ptr = stm_pw.as_ptr() as *const u32;
    run_perspective!(stm_chunks_ptr, 0);
    let ntm_chunks_ptr = ntm_pw.as_ptr() as *const u32;
    run_perspective!(ntm_chunks_ptr, pw / 4);

    let g0 = _mm256_add_epi32(e0, o0);
    let g1 = _mm256_add_epi32(e1, o1);
    let g2 = _mm256_add_epi32(e2, o2);
    let g3 = _mm256_add_epi32(e3, o3);

    let scale = _mm256_set1_epi32(bias_scale);
    macro_rules! bias_grp {
        ($off:expr) => {
            _mm256_mullo_epi32(
                _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add($off) as *const __m128i)),
                scale,
            )
        };
    }
    _mm256_storeu_si256(output as *mut __m256i, _mm256_add_epi32(g0, bias_grp!(0)));
    _mm256_storeu_si256(output.add(8) as *mut __m256i, _mm256_add_epi32(g1, bias_grp!(8)));
    _mm256_storeu_si256(output.add(16) as *mut __m256i, _mm256_add_epi32(g2, bias_grp!(16)));
    _mm256_storeu_si256(output.add(24) as *mut __m256i, _mm256_add_epi32(g3, bias_grp!(24)));
}

/// Sparse column-major L1 matmul, AVX-512 VNNI variant — skips 4-byte
/// zero input chunks. Uses four interleaved accumulators to hide VPDPBUSD
/// latency even when chunks are dense.
///
/// # Safety
/// CPU must support AVX-512F/BW/VNNI. Slices must be sized for `pw`/`num_neurons`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn sparse_l1_avx512_vnni(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],
    num_neurons: usize,
    bias: &[i16],
    bias_scale: i32,
    output: &mut [i32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(num_neurons, 16, "sparse_l1_avx512_vnni currently specialised to 16 neurons");

    let chunk_stride = num_neurons * 4;

    for i in 0..num_neurons { output[i] = bias[i] as i32 * bias_scale; }

    let mut a0 = _mm512_setzero_si512();
    let mut a1 = _mm512_setzero_si512();
    let mut a2 = _mm512_setzero_si512();
    let mut a3 = _mm512_setzero_si512();

    let w_ptr = sparse_weights.as_ptr();
    let mut rot: u32 = 0;

    let stm_chunks = stm_pw.as_ptr() as *const u32;
    for chunk_idx in 0..pw / 4 {
        let val = stm_chunks.add(chunk_idx).read_unaligned();
        if val == 0 { continue; }
        let input = _mm512_set1_epi32(val as i32);
        let w_off = chunk_idx * chunk_stride;
        let w = _mm512_loadu_si512(w_ptr.add(w_off) as *const __m512i);
        match rot & 3 {
            0 => a0 = _mm512_dpbusd_epi32(a0, input, w),
            1 => a1 = _mm512_dpbusd_epi32(a1, input, w),
            2 => a2 = _mm512_dpbusd_epi32(a2, input, w),
            _ => a3 = _mm512_dpbusd_epi32(a3, input, w),
        }
        rot = rot.wrapping_add(1);
    }

    let ntm_chunk_offset = pw / 4;
    let ntm_chunks = ntm_pw.as_ptr() as *const u32;
    for chunk_idx in 0..pw / 4 {
        let val = ntm_chunks.add(chunk_idx).read_unaligned();
        if val == 0 { continue; }
        let input = _mm512_set1_epi32(val as i32);
        let w_off = (ntm_chunk_offset + chunk_idx) * chunk_stride;
        let w = _mm512_loadu_si512(w_ptr.add(w_off) as *const __m512i);
        match rot & 3 {
            0 => a0 = _mm512_dpbusd_epi32(a0, input, w),
            1 => a1 = _mm512_dpbusd_epi32(a1, input, w),
            2 => a2 = _mm512_dpbusd_epi32(a2, input, w),
            _ => a3 = _mm512_dpbusd_epi32(a3, input, w),
        }
        rot = rot.wrapping_add(1);
    }

    let acc = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
    let mut results = [0i32; 16];
    _mm512_storeu_si512(results.as_mut_ptr() as *mut __m512i, acc);
    for i in 0..num_neurons {
        output[i] += results[i];
    }
}

/// Dense column-major L1 matmul, AVX-VNNI variant (AVX2-class machines
/// with `VPDPBUSD` YMM form — Alder Lake+, Zen 4+).
///
/// Same as `dense_l1_avx2` but replaces the
/// `VPMADDUBSW + VPMADDWD(ones) + VPADDD` sequence with a single
/// `VPDPBUSD` per YMM lane.
///
/// Uses 4 interleaved accumulator pairs to hide VPDPBUSD's ~5-cycle
/// latency on Alder Lake / Zen 4. Without this, the loop serialises on
/// the accumulator dependency chain and runs ~2.5× slower than AVX2.
///
/// # Safety
/// CPU must support AVX2 + AVX-VNNI. Slices must be sized for `pw`; `output`
/// must point to at least `num_neurons` writable i32s.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
pub unsafe fn dense_l1_avx_vnni(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8],
    num_neurons: usize,
    bias: &[i16],
    bias_scale: i32,
    output: *mut i32,
) {
    use std::arch::x86_64::*;

    let chunk_stride = num_neurons * 4;

    // Four accumulator pairs (lo=neurons 0-7, hi=neurons 8-15) break
    // the VPDPBUSD dependency chain. Same pattern as dense_l1_avx512_vnni
    // but with YMM registers (8 neurons each instead of 16).
    let mut a0_lo = _mm256_setzero_si256();
    let mut a0_hi = _mm256_setzero_si256();
    let mut a1_lo = _mm256_setzero_si256();
    let mut a1_hi = _mm256_setzero_si256();
    let mut a2_lo = _mm256_setzero_si256();
    let mut a2_hi = _mm256_setzero_si256();
    let mut a3_lo = _mm256_setzero_si256();
    let mut a3_hi = _mm256_setzero_si256();

    let w_ptr = sparse_weights.as_ptr();
    let total_chunks = pw / 4;
    let wide = num_neurons > 8;

    macro_rules! run_perspective {
        ($chunks:expr, $chunk_offset:expr) => {{
            let chunks: *const u32 = $chunks;
            let chunk_offset: usize = $chunk_offset;
            let mut c = 0usize;
            while c + 4 <= total_chunks {
                let v0 = chunks.add(c).read_unaligned();
                let v1 = chunks.add(c + 1).read_unaligned();
                let v2 = chunks.add(c + 2).read_unaligned();
                let v3 = chunks.add(c + 3).read_unaligned();
                let i0 = _mm256_set1_epi32(v0 as i32);
                let i1 = _mm256_set1_epi32(v1 as i32);
                let i2 = _mm256_set1_epi32(v2 as i32);
                let i3 = _mm256_set1_epi32(v3 as i32);
                let base0 = (chunk_offset + c) * chunk_stride;
                let base1 = (chunk_offset + c + 1) * chunk_stride;
                let base2 = (chunk_offset + c + 2) * chunk_stride;
                let base3 = (chunk_offset + c + 3) * chunk_stride;
                let w0 = _mm256_loadu_si256(w_ptr.add(base0) as *const __m256i);
                let w1 = _mm256_loadu_si256(w_ptr.add(base1) as *const __m256i);
                let w2 = _mm256_loadu_si256(w_ptr.add(base2) as *const __m256i);
                let w3 = _mm256_loadu_si256(w_ptr.add(base3) as *const __m256i);
                a0_lo = _mm256_dpbusd_avx_epi32(a0_lo, i0, w0);
                a1_lo = _mm256_dpbusd_avx_epi32(a1_lo, i1, w1);
                a2_lo = _mm256_dpbusd_avx_epi32(a2_lo, i2, w2);
                a3_lo = _mm256_dpbusd_avx_epi32(a3_lo, i3, w3);
                if wide {
                    let w0h = _mm256_loadu_si256(w_ptr.add(base0 + 32) as *const __m256i);
                    let w1h = _mm256_loadu_si256(w_ptr.add(base1 + 32) as *const __m256i);
                    let w2h = _mm256_loadu_si256(w_ptr.add(base2 + 32) as *const __m256i);
                    let w3h = _mm256_loadu_si256(w_ptr.add(base3 + 32) as *const __m256i);
                    a0_hi = _mm256_dpbusd_avx_epi32(a0_hi, i0, w0h);
                    a1_hi = _mm256_dpbusd_avx_epi32(a1_hi, i1, w1h);
                    a2_hi = _mm256_dpbusd_avx_epi32(a2_hi, i2, w2h);
                    a3_hi = _mm256_dpbusd_avx_epi32(a3_hi, i3, w3h);
                }
                c += 4;
            }
            // Tail: remaining chunks into accumulator pair 0.
            while c < total_chunks {
                let v = chunks.add(c).read_unaligned();
                let inp = _mm256_set1_epi32(v as i32);
                let base = (chunk_offset + c) * chunk_stride;
                let w = _mm256_loadu_si256(w_ptr.add(base) as *const __m256i);
                a0_lo = _mm256_dpbusd_avx_epi32(a0_lo, inp, w);
                if wide {
                    let wh = _mm256_loadu_si256(w_ptr.add(base + 32) as *const __m256i);
                    a0_hi = _mm256_dpbusd_avx_epi32(a0_hi, inp, wh);
                }
                c += 1;
            }
        }};
    }

    let stm_chunks_ptr = stm_pw.as_ptr() as *const u32;
    run_perspective!(stm_chunks_ptr, 0);
    let ntm_chunks_ptr = ntm_pw.as_ptr() as *const u32;
    run_perspective!(ntm_chunks_ptr, pw / 4);

    // Merge the four accumulator pairs.
    let lo = _mm256_add_epi32(_mm256_add_epi32(a0_lo, a1_lo), _mm256_add_epi32(a2_lo, a3_lo));
    let hi = _mm256_add_epi32(_mm256_add_epi32(a0_hi, a1_hi), _mm256_add_epi32(a2_hi, a3_hi));
    if num_neurons == 16 {
        let scale = _mm256_set1_epi32(bias_scale);
        let b0 = _mm256_mullo_epi32(
            _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr() as *const __m128i)),
            scale,
        );
        let b1 = _mm256_mullo_epi32(
            _mm256_cvtepi16_epi32(_mm_loadu_si128(bias.as_ptr().add(8) as *const __m128i)),
            scale,
        );
        _mm256_storeu_si256(output as *mut __m256i, _mm256_add_epi32(lo, b0));
        _mm256_storeu_si256(output.add(8) as *mut __m256i, _mm256_add_epi32(hi, b1));
    } else {
        for i in 0..num_neurons { output.add(i).write(bias[i] as i32 * bias_scale); }
        let output = std::slice::from_raw_parts_mut(output, num_neurons);
        let mut results = [0i32; 16];
        _mm256_storeu_si256(results.as_mut_ptr() as *mut __m256i, lo);
        _mm256_storeu_si256(results.as_mut_ptr().add(8) as *mut __m256i, hi);
        for i in 0..num_neurons {
            output[i] += results[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_weights() {
        // 2 neurons, pw=4 per perspective, total_input=8 (4 STM + 4 NTM)
        // Engine layout: STM block [neuron * pw + input], NTM block [num_neurons * pw + neuron * pw + input]
        let dense = vec![
            // STM block: neuron 0 inputs 0-3, neuron 1 inputs 0-3
            10i8, 20, 30, 40,     // neuron 0 STM
            -10, -20, -30, -40,   // neuron 1 STM
            // NTM block: neuron 0 inputs 0-3, neuron 1 inputs 0-3
            50, 60, 70, 80,       // neuron 0 NTM
            -50, -60, -70, -80,   // neuron 1 NTM
        ];
        let sparse = transpose_weights_for_sparse(&dense, 8, 2);

        // STM chunk 0 (inputs 0-3): neuron 0 [10,20,30,40], neuron 1 [-10,-20,-30,-40]
        assert_eq!(sparse[0], 10);   // chunk0, neuron0, byte0
        assert_eq!(sparse[1], 20);   // chunk0, neuron0, byte1
        assert_eq!(sparse[4], -10);  // chunk0, neuron1, byte0

        // NTM chunk 1 (inputs 0-3 of NTM): neuron 0 [50,60,70,80], neuron 1 [-50,-60,-70,-80]
        assert_eq!(sparse[8], 50);   // chunk1(NTM), neuron0, byte0
        assert_eq!(sparse[12], -50); // chunk1(NTM), neuron1, byte0
    }

    #[test]
    fn test_sparse_matches_dense() {
        crate::init();

        // Create test data: 16 neurons, 384 pw per perspective = 768 total input
        let pw = 384;
        let num_neurons = 16;
        let total_input = pw * 2;

        // Random-ish weights in the ACTUAL engine layout:
        // STM block: [neuron * pw + stm_input] for first num_neurons * pw entries
        // NTM block: [num_neurons * pw + neuron * pw + ntm_input]
        let mut dense_weights = vec![0i8; num_neurons * total_input];
        for i in 0..dense_weights.len() {
            dense_weights[i] = ((i * 7 + 13) % 256) as i8;
        }

        let sparse_weights = transpose_weights_for_sparse(&dense_weights, total_input, num_neurons);

        // Create pairwise output with some zeros (simulate 80% sparsity)
        let mut stm_pw = vec![0u8; pw];
        let mut ntm_pw = vec![0u8; pw];
        for i in 0..pw {
            if i % 5 != 0 { continue; } // 80% zeros
            stm_pw[i] = ((i * 3 + 1) % 128) as u8;
            ntm_pw[i] = ((i * 5 + 7) % 128) as u8;
        }

        let bias = vec![100i16; num_neurons];
        let bias_scale = 127; // PW_SCALE

        // Compute sparse
        let mut sparse_output = vec![0i32; num_neurons];
        sparse_l1_scalar(
            &stm_pw, &ntm_pw, pw, &sparse_weights, num_neurons,
            &bias, bias_scale, &mut sparse_output,
        );

        // Compute dense (reference) — using engine's actual weight layout:
        // STM: weights_8t[neuron * pw + j]
        // NTM: weights_8t[num_neurons * pw + neuron * pw + j]
        let ntm_base = num_neurons * pw;
        let mut dense_output = vec![0i32; num_neurons];
        for i in 0..num_neurons { dense_output[i] = bias[i] as i32 * bias_scale; }
        for neuron in 0..num_neurons {
            for j in 0..pw {
                dense_output[neuron] += stm_pw[j] as i32 * dense_weights[neuron * pw + j] as i32;
            }
            for j in 0..pw {
                dense_output[neuron] += ntm_pw[j] as i32 * dense_weights[ntm_base + neuron * pw + j] as i32;
            }
        }

        for i in 0..num_neurons {
            assert_eq!(sparse_output[i], dense_output[i],
                "Neuron {} mismatch: sparse={} dense={}", i, sparse_output[i], dense_output[i]);
        }
        eprintln!("Sparse L1 scalar: all {} neurons match!", num_neurons);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_sparse_avx2_matches_scalar() {
        crate::init();

        let pw = 384;
        let num_neurons = 16;
        let total_input = pw * 2;

        let mut dense_weights = vec![0i8; num_neurons * total_input];
        for i in 0..dense_weights.len() {
            dense_weights[i] = ((i * 7 + 13) % 256) as i8;
        }
        let sparse_weights = transpose_weights_for_sparse(&dense_weights, total_input, num_neurons);

        let mut stm_pw = vec![0u8; pw];
        let mut ntm_pw = vec![0u8; pw];
        for i in 0..pw {
            if i % 5 != 0 { continue; }
            stm_pw[i] = ((i * 3 + 1) % 128) as u8;
            ntm_pw[i] = ((i * 5 + 7) % 128) as u8;
        }

        let bias = vec![100i16; num_neurons];
        let bias_scale = 127;

        // Scalar reference
        let mut scalar_output = vec![0i32; num_neurons];
        sparse_l1_scalar(
            &stm_pw, &ntm_pw, pw, &sparse_weights, num_neurons,
            &bias, bias_scale, &mut scalar_output,
        );

        // AVX2
        let mut avx2_output = vec![0i32; num_neurons];
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                sparse_l1_avx2(
                    &stm_pw, &ntm_pw, pw, &sparse_weights, num_neurons,
                    &bias, bias_scale, &mut avx2_output,
                );
            }
        } else {
            eprintln!("No AVX2, skipping SIMD test");
            return;
        }

        for i in 0..num_neurons {
            assert_eq!(avx2_output[i], scalar_output[i],
                "Neuron {} mismatch: avx2={} scalar={}", i, avx2_output[i], scalar_output[i]);
        }
        eprintln!("Sparse L1 AVX2: all {} neurons match scalar!", num_neurons);
    }

    /// Build a representative L1=16 pairwise test case and return
    /// (sparse_weights, bias, stm_pw, ntm_pw, pw, num_neurons, bias_scale).
    /// Uses a mix of dense and zero chunks so both dense and sparse paths
    /// are exercised meaningfully. `pw` is parameterised so tests cover
    /// both the FT768 (pw=384) and FT1024 (pw=512, current prod) shapes.
    #[cfg(target_arch = "x86_64")]
    fn build_l1_16_test_case(
        seed: u64,
        density_pct: u32,
        pw: usize,
    ) -> (Vec<i8>, Vec<i16>, Vec<u8>, Vec<u8>, usize, usize, i32) {
        let num_neurons = 16;
        let total_input = pw * 2;

        let mut dense_weights = vec![0i8; num_neurons * total_input];
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        for w in dense_weights.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *w = ((s >> 56) as i8).saturating_sub(0).max(-120).min(120);
        }
        let sparse_weights = transpose_weights_for_sparse(&dense_weights, total_input, num_neurons);

        let mut stm_pw = vec![0u8; pw];
        let mut ntm_pw = vec![0u8; pw];
        let mut t = seed.wrapping_add(0xDEAD_BEEF);
        for i in 0..pw {
            t = t.wrapping_mul(6364136223846793005).wrapping_add(1);
            let keep_s = (t as u32 % 100) < density_pct;
            t = t.wrapping_mul(6364136223846793005).wrapping_add(1);
            let keep_n = (t as u32 % 100) < density_pct;
            if keep_s { stm_pw[i] = ((t >> 24) & 0xFF) as u8; }
            if keep_n { ntm_pw[i] = ((t >> 32) & 0xFF) as u8; }
        }
        let bias: Vec<i16> = (0..num_neurons).map(|i| (i as i16) * 3 - 20).collect();
        let bias_scale = 127; // PW_SCALE
        (sparse_weights, bias, stm_pw, ntm_pw, pw, num_neurons, bias_scale)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dense_avx512_vnni_matches_scalar() {
        crate::init();

        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
        {
            eprintln!("No AVX-512 VNNI on this CPU, skipping test");
            return;
        }

        for &pw_dim in &[384usize, 512] {
            for density in [0u32, 25, 50, 75, 100] {
                for seed in 0u64..6 {
                    let (sw, bias, s_pw, n_pw, pw, nn, scale) = build_l1_16_test_case(seed, density, pw_dim);

                    let mut scalar_out = vec![0i32; nn];
                    sparse_l1_scalar(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, &mut scalar_out);

                    let mut vnni_out = vec![0i32; nn];
                    unsafe {
                        dense_l1_avx512_vnni(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, vnni_out.as_mut_ptr());
                    }

                    for i in 0..nn {
                        assert_eq!(
                            vnni_out[i], scalar_out[i],
                            "dense_l1_avx512_vnni mismatch seed={} density={} pw={} neuron={} vnni={} scalar={}",
                            seed, density, pw, i, vnni_out[i], scalar_out[i]
                        );
                    }
                }
            }
        }
        eprintln!("dense_l1_avx512_vnni: all seeds/densities match scalar");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_sparse_avx512_vnni_matches_scalar() {
        crate::init();

        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
        {
            eprintln!("No AVX-512 VNNI on this CPU, skipping test");
            return;
        }

        for &pw_dim in &[384usize, 512] {
            for density in [0u32, 10, 50, 90, 100] {
                for seed in 0u64..6 {
                    let (sw, bias, s_pw, n_pw, pw, nn, scale) = build_l1_16_test_case(seed, density, pw_dim);

                    let mut scalar_out = vec![0i32; nn];
                    sparse_l1_scalar(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, &mut scalar_out);

                    let mut vnni_out = vec![0i32; nn];
                    unsafe {
                        sparse_l1_avx512_vnni(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, &mut vnni_out);
                    }

                    for i in 0..nn {
                        assert_eq!(
                            vnni_out[i], scalar_out[i],
                            "sparse_l1_avx512_vnni mismatch seed={} density={} pw={} neuron={} vnni={} scalar={}",
                            seed, density, pw, i, vnni_out[i], scalar_out[i]
                        );
                    }
                }
            }
        }
        eprintln!("sparse_l1_avx512_vnni: all seeds/densities match scalar");
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dense_avx_vnni_matches_scalar() {
        crate::init();

        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("avxvnni") {
            eprintln!("No AVX-VNNI on this CPU, skipping test");
            return;
        }

        for &pw_dim in &[384usize, 512] {
            for density in [0u32, 50, 100] {
                for seed in 0u64..4 {
                    let (sw, bias, s_pw, n_pw, pw, nn, scale) = build_l1_16_test_case(seed, density, pw_dim);

                    let mut scalar_out = vec![0i32; nn];
                    sparse_l1_scalar(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, &mut scalar_out);

                    let mut vnni_out = vec![0i32; nn];
                    unsafe {
                        dense_l1_avx_vnni(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, vnni_out.as_mut_ptr());
                    }

                    for i in 0..nn {
                        assert_eq!(
                            vnni_out[i], scalar_out[i],
                            "dense_l1_avx_vnni mismatch seed={} density={} pw={} neuron={} vnni={} scalar={}",
                            seed, density, pw, i, vnni_out[i], scalar_out[i]
                        );
                    }
                }
            }
        }
        eprintln!("dense_l1_avx_vnni: all seeds/densities match scalar");
    }

    /// Build a representative L1=N pairwise test case. Same shape as
    /// `build_l1_16_test_case` but parameterised over neuron count and pw.
    ///
    /// Input bytes are bounded to [0, max_input] to control whether
    /// VPMADDUBSW i16 saturation can fire. Production pairwise output
    /// stays in [0, 127] (the engine applies `>> FT_SHIFT` (9) inside
    /// the pairwise pack), so `max_input=100` is a realistic stress.
    /// `max_input=255` exercises full u8 range and WILL trigger
    /// saturation — only use that variant if you want to characterise
    /// the saturation envelope, not for correctness assertions.
    #[cfg(target_arch = "x86_64")]
    fn build_l1_n_test_case(
        seed: u64,
        density_pct: u32,
        num_neurons: usize,
        pw: usize,
        max_input: u32,
    ) -> (Vec<i8>, Vec<i16>, Vec<u8>, Vec<u8>, usize, usize, i32) {
        let total_input = pw * 2;

        let mut dense_weights = vec![0i8; num_neurons * total_input];
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        for w in dense_weights.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *w = ((s >> 56) as i8).max(-120).min(120);
        }
        let sparse_weights = transpose_weights_for_sparse(&dense_weights, total_input, num_neurons);

        let mut stm_pw = vec![0u8; pw];
        let mut ntm_pw = vec![0u8; pw];
        let mut t = seed.wrapping_add(0xDEAD_BEEF);
        for i in 0..pw {
            t = t.wrapping_mul(6364136223846793005).wrapping_add(1);
            let keep_s = (t as u32 % 100) < density_pct;
            t = t.wrapping_mul(6364136223846793005).wrapping_add(1);
            let keep_n = (t as u32 % 100) < density_pct;
            if keep_s { stm_pw[i] = (((t >> 24) as u32) % (max_input + 1)) as u8; }
            if keep_n { ntm_pw[i] = (((t >> 32) as u32) % (max_input + 1)) as u8; }
        }
        let bias: Vec<i16> = (0..num_neurons).map(|i| (i as i16) * 3 - 20).collect();
        let bias_scale = 127;
        (sparse_weights, bias, stm_pw, ntm_pw, pw, num_neurons, bias_scale)
    }

    /// Fuzz: AVX2 L1=32 kernel matches scalar reference across many
    /// random seeds, densities, and pw widths. Catches edge cases the
    /// single-seed test misses (boundary chunks, sparsity patterns,
    /// register-aliasing hazards).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fuzz_dense_avx2_l1_32_matches_scalar() {
        crate::init();
        if !is_x86_feature_detected!("avx2") {
            eprintln!("No AVX2 — skipping L1=32 fuzz test");
            return;
        }

        let mut cases = 0usize;
        // Cap input byte range to 100 to stay inside the VPMADDUBSW
        // i16 saturation envelope. Production pairwise output stays
        // in [0, ~127] (the >>7 shift inside the pairwise kernel
        // bounds it), so this is a realistic stress, not an artificial
        // restriction. A separate `saturation` test below exercises
        // the full u8 range and asserts the kernel and scalar agree
        // *up to saturation* — the kernel is defined by VPMADDUBSW's
        // semantics, not by saturation-free math.
        for &pw in &[64usize, 128, 256, 384, 512] {
            for density in [0u32, 10, 25, 50, 75, 89, 100] {
                for seed in 0u64..30 {
                    let (sw, bias, s_pw, n_pw, pw_, nn, scale) =
                        build_l1_n_test_case(seed, density, 32, pw, 100);

                    let mut scalar_out = vec![0i32; nn];
                    sparse_l1_scalar(&s_pw, &n_pw, pw_, &sw, nn, &bias, scale, &mut scalar_out);

                    let mut avx2_out = vec![0i32; nn];
                    unsafe {
                        dense_l1_avx2_l1_32(
                            &s_pw, &n_pw, pw_, &sw, &bias, scale, avx2_out.as_mut_ptr(),
                        );
                    }

                    for i in 0..nn {
                        assert_eq!(
                            avx2_out[i], scalar_out[i],
                            "dense_l1_avx2_l1_32 mismatch seed={} density={} pw={} neuron={} avx2={} scalar={}",
                            seed, density, pw, i, avx2_out[i], scalar_out[i]
                        );
                    }
                    cases += 1;
                }
            }
        }
        eprintln!("dense_l1_avx2_l1_32 fuzz: {} cases passed", cases);
    }

    /// Fuzz: the maddubs-pair-fused AVX2 **L1=16** kernel matches scalar under
    /// the saturation gate, and matches the generic `dense_l1_avx2` it
    /// replaces. Weights capped at +/-60 (worst lane sum 4x60=240 <= 258) so
    /// every case passes `x2_fusion_safe`; inputs use the full production
    /// range (<=127). The generic-kernel comparison is the one that matters
    /// for the swap: selecting the fused path must be observationally inert.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fuzz_dense_avx2_l1_16_x2_matches_scalar() {
        crate::init();
        if !is_x86_feature_detected!("avx2") {
            eprintln!("No AVX2 — skipping L1=16 x2 fuzz test");
            return;
        }

        let mut cases = 0usize;
        for &pw in &[64usize, 128, 256, 384, 512] {
            for density in [0u32, 25, 50, 89, 100] {
                for seed in 0u64..30 {
                    let (sw, bias, s_pw, n_pw, pw_, nn, scale) =
                        build_l1_n_test_case(seed, density, 16, pw, 127);
                    let sw: Vec<i8> = sw.iter().map(|&w| w.clamp(-60, 60)).collect();
                    assert_eq!(nn, 16);
                    assert!(x2_fusion_safe(&sw, nn), "capped weights must pass the gate");

                    let mut scalar_out = vec![0i32; nn];
                    sparse_l1_scalar(&s_pw, &n_pw, pw_, &sw, nn, &bias, scale, &mut scalar_out);

                    let mut generic_out = vec![0i32; nn];
                    let mut x2_out = vec![0i32; nn];
                    unsafe {
                        dense_l1_avx2(
                            &s_pw, &n_pw, pw_, &sw, nn, &bias, scale, generic_out.as_mut_ptr(),
                        );
                        dense_l1_avx2_l1_16_x2(
                            &s_pw, &n_pw, pw_, &sw, &bias, scale, x2_out.as_mut_ptr(),
                        );
                    }
                    for i in 0..nn {
                        assert_eq!(
                            x2_out[i], scalar_out[i],
                            "l1_16_x2 vs scalar mismatch seed={} density={} pw={} neuron={}",
                            seed, density, pw, i
                        );
                        assert_eq!(
                            x2_out[i], generic_out[i],
                            "l1_16_x2 vs dense_l1_avx2 mismatch seed={} density={} pw={} neuron={}",
                            seed, density, pw, i
                        );
                    }
                    cases += 1;
                }
            }
        }
        eprintln!("dense_l1_avx2_l1_16_x2 fuzz: {} cases passed", cases);

        // Gate rejection at width 16: one co-lane quad summing 259 must fail,
        // so such a net can never select the fused arm.
        let nn = 16;
        let pw = 64usize;
        let mut bad = vec![0i8; (2 * pw / 4) * nn * 4];
        bad[0] = 127;
        bad[1] = 127;
        bad[nn * 4] = 4;
        bad[nn * 4 + 1] = 1;
        assert!(!x2_fusion_safe(&bad, nn), "sum-259 lane must fail the gate at L1=16");
    }

    /// Fuzz: the maddubs-pair-fused AVX2 L1=32 kernel matches scalar under
    /// the saturation gate. Weights are capped at ±60 (worst lane sum
    /// 4×60=240 ≤ 258) so every case passes `x2_fusion_safe`; inputs use
    /// the full production range (≤127). Also asserts the gate itself:
    /// accepts the capped weights, rejects a crafted violating set, and
    /// accepts an exactly-at-the-bound set which the kernel must still
    /// compute exactly (127×258 = 32766 < 32767).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fuzz_dense_avx2_l1_32_x2_matches_scalar() {
        crate::init();
        if !is_x86_feature_detected!("avx2") {
            eprintln!("No AVX2 — skipping L1=32 x2 fuzz test");
            return;
        }

        let mut cases = 0usize;
        for &pw in &[64usize, 128, 256, 384, 512] {
            for density in [0u32, 25, 50, 89, 100] {
                for seed in 0u64..30 {
                    let (sw, bias, s_pw, n_pw, pw_, nn, scale) =
                        build_l1_n_test_case(seed, density, 32, pw, 127);
                    // Cap weights to the always-safe band.
                    let sw: Vec<i8> = sw.iter().map(|&w| w.clamp(-60, 60)).collect();
                    assert!(x2_fusion_safe(&sw, nn), "capped weights must pass the gate");

                    let mut scalar_out = vec![0i32; nn];
                    sparse_l1_scalar(&s_pw, &n_pw, pw_, &sw, nn, &bias, scale, &mut scalar_out);

                    let mut x2_out = vec![0i32; nn];
                    unsafe {
                        dense_l1_avx2_l1_32_x2(
                            &s_pw, &n_pw, pw_, &sw, &bias, scale, x2_out.as_mut_ptr(),
                        );
                    }
                    for i in 0..nn {
                        assert_eq!(
                            x2_out[i], scalar_out[i],
                            "dense_l1_avx2_l1_32_x2 mismatch seed={} density={} pw={} neuron={}",
                            seed, density, pw, i
                        );
                    }
                    cases += 1;
                }
            }
        }
        eprintln!("dense_l1_avx2_l1_32_x2 fuzz: {} cases passed", cases);

        // Gate rejection: one co-lane quad summing 259 (> 258) must fail.
        let nn = 32;
        let pw = 64usize;
        let mut bad = vec![0i8; (2 * pw / 4) * nn * 4];
        // chunk 0, neuron 0, lane 0 weights: 127+127+4+1 = 259
        bad[0] = 127;                 // chunk 0, n0, w0
        bad[1] = 127;                 // chunk 0, n0, w1
        bad[nn * 4] = 4;              // chunk 1, n0, w0
        bad[nn * 4 + 1] = 1;          // chunk 1, n0, w1
        assert!(!x2_fusion_safe(&bad, nn), "sum-259 lane must fail the gate");

        // Exactly at the bound (sum 258): gate accepts, kernel exact at
        // max input (127×258 = 32766 fits i16).
        let mut edge = vec![0i8; (2 * pw / 4) * nn * 4];
        edge[0] = 127;
        edge[1] = 127;
        edge[nn * 4] = 4;             // 127+127+4+0 = 258
        assert!(x2_fusion_safe(&edge, nn), "sum-258 lane must pass the gate");
        let s_pw = vec![127u8; pw];
        let n_pw = vec![127u8; pw];
        let bias: Vec<i16> = vec![0; nn];
        let mut scalar_out = vec![0i32; nn];
        sparse_l1_scalar(&s_pw, &n_pw, pw, &edge, nn, &bias, 127, &mut scalar_out);
        let mut x2_out = vec![0i32; nn];
        unsafe { dense_l1_avx2_l1_32_x2(&s_pw, &n_pw, pw, &edge, &bias, 127, x2_out.as_mut_ptr()); }
        assert_eq!(x2_out, scalar_out, "x2 kernel must be exact at the 258 saturation bound");
    }

    /// Fuzz: AVX-512 VNNI L1=32 column-major kernel matches scalar across
    /// seeds, densities, and pw widths. The VNNI twin of the AVX2 L1=32
    /// fuzz above. Bit-exactness against the scalar reference is the
    /// correctness gate for the column-major-vs-row-major kernel swap.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fuzz_dense_avx512_vnni_l1_32_matches_scalar() {
        crate::init();
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
        {
            eprintln!("No AVX-512 VNNI on this CPU — skipping L1=32 VNNI fuzz test");
            return;
        }

        let mut cases = 0usize;
        // 512 = FT1024 prod width; others exercise boundary chunk counts.
        for &pw in &[64usize, 128, 256, 384, 512] {
            for density in [0u32, 10, 25, 50, 75, 89, 100] {
                for seed in 0u64..30 {
                    let (sw, bias, s_pw, n_pw, pw_, nn, scale) =
                        build_l1_n_test_case(seed, density, 32, pw, 100);

                    let mut scalar_out = vec![0i32; nn];
                    sparse_l1_scalar(&s_pw, &n_pw, pw_, &sw, nn, &bias, scale, &mut scalar_out);

                    let mut vnni_out = vec![0i32; nn];
                    unsafe {
                        dense_l1_avx512_vnni_l1_32(
                            &s_pw, &n_pw, pw_, &sw, &bias, scale, vnni_out.as_mut_ptr(),
                        );
                    }

                    for i in 0..nn {
                        assert_eq!(
                            vnni_out[i], scalar_out[i],
                            "dense_l1_avx512_vnni_l1_32 mismatch seed={} density={} pw={} neuron={} vnni={} scalar={}",
                            seed, density, pw, i, vnni_out[i], scalar_out[i]
                        );
                    }
                    cases += 1;
                }
            }
        }
        eprintln!("dense_l1_avx512_vnni_l1_32 fuzz: {} cases passed", cases);
    }

    /// Fuzz: AVX-VNNI (YMM VPDPBUSD) L1=32 column-major kernel matches
    /// scalar. For Alder/Raptor-Lake-class hosts (AVX-VNNI without AVX-512).
    /// Runs anywhere `avxvnni` is present (incl. Zen5).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fuzz_dense_avx_vnni_l1_32_matches_scalar() {
        crate::init();
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("avxvnni") {
            eprintln!("No AVX-VNNI on this CPU — skipping AVX-VNNI L1=32 fuzz test");
            return;
        }
        let mut cases = 0usize;
        for &pw in &[64usize, 128, 256, 384, 512] {
            for density in [0u32, 10, 25, 50, 75, 89, 100] {
                for seed in 0u64..30 {
                    let (sw, bias, s_pw, n_pw, pw_, nn, scale) =
                        build_l1_n_test_case(seed, density, 32, pw, 100);
                    let mut scalar_out = vec![0i32; nn];
                    sparse_l1_scalar(&s_pw, &n_pw, pw_, &sw, nn, &bias, scale, &mut scalar_out);
                    let mut vnni_out = vec![0i32; nn];
                    unsafe {
                        dense_l1_avx_vnni_l1_32(&s_pw, &n_pw, pw_, &sw, &bias, scale, vnni_out.as_mut_ptr());
                    }
                    for i in 0..nn {
                        assert_eq!(vnni_out[i], scalar_out[i],
                            "dense_l1_avx_vnni_l1_32 mismatch seed={} density={} pw={} neuron={} vnni={} scalar={}",
                            seed, density, pw, i, vnni_out[i], scalar_out[i]);
                    }
                    cases += 1;
                }
            }
        }
        eprintln!("dense_l1_avx_vnni_l1_32 fuzz: {} cases passed", cases);
    }

    /// Microbench: the three L1=32 kernels at pw=512 (FT1024 prod width).
    /// AVX2-maddubs vs AVX-VNNI is the delta an Alder/Raptor-Lake host
    /// sees (both 256-bit); AVX-512 VNNI is the reference for full-512 hosts.
    /// `cargo test --release bench_l1_32_kernels -- --ignored --nocapture`
    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn bench_l1_32_kernels() {
        use std::time::Instant;
        crate::init();
        const PW: usize = 512;
        const ITERS: usize = 400_000;
        const WARMUP: usize = 10_000;
        let (sw, bias, s_pw, n_pw, _pw, _nn, scale) = build_l1_n_test_case(7, 60, 32, PW, 100);
        let mut out = vec![0i32; 32];

        macro_rules! time_kernel {
            ($name:expr, $avail:expr, $call:expr) => {{
                if $avail {
                    for _ in 0..WARMUP { unsafe { $call; } }
                    let t = Instant::now();
                    for _ in 0..ITERS { unsafe { $call; } }
                    let ns = t.elapsed().as_nanos() as f64 / ITERS as f64;
                    eprintln!("  {:<28} {:>7.1} ns/call", $name, ns);
                    ns
                } else { eprintln!("  {:<28} (unavailable)", $name); f64::NAN }
            }};
        }
        eprintln!("L1=32 kernels @ pw={} ({} iters):", PW, ITERS);
        let avx2 = time_kernel!("dense_l1_avx2_l1_32 (AVX2)", is_x86_feature_detected!("avx2"),
            dense_l1_avx2_l1_32(&s_pw, &n_pw, PW, &sw, &bias, scale, out.as_mut_ptr()));
        let avxv = time_kernel!("dense_l1_avx_vnni_l1_32", is_x86_feature_detected!("avxvnni"),
            dense_l1_avx_vnni_l1_32(&s_pw, &n_pw, PW, &sw, &bias, scale, out.as_mut_ptr()));
        let av512 = time_kernel!("dense_l1_avx512_vnni_l1_32",
            is_x86_feature_detected!("avx512vnni"),
            dense_l1_avx512_vnni_l1_32(&s_pw, &n_pw, PW, &sw, &bias, scale, out.as_mut_ptr()));
        if avx2.is_finite() && avxv.is_finite() {
            eprintln!("  AVX-VNNI vs AVX2 (Alder-Lake-relevant): {:+.1}% faster", (avx2 / avxv - 1.0) * 100.0);
        }
        if avx2.is_finite() && av512.is_finite() {
            eprintln!("  AVX-512 VNNI vs AVX2: {:+.1}% faster", (avx2 / av512 - 1.0) * 100.0);
        }
    }

    /// Cross-check: the new VNNI L1=32 kernel agrees with the existing AVX2
    /// L1=32 kernel byte-for-byte. Both are column-major over the same
    /// `l1_weights_sparse` layout, so any divergence is a kernel bug.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dense_l1_avx512_vnni_l1_32_matches_avx2() {
        crate::init();
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vnni")
            || !is_x86_feature_detected!("avx2")
        {
            eprintln!("Need AVX2 + AVX-512 VNNI — skipping L1=32 VNNI-vs-AVX2 cross-check");
            return;
        }
        for &pw in &[128usize, 384, 512] {
            for density in [25u32, 60, 100] {
                for seed in 0u64..8 {
                    let (sw, bias, _s, _n, _pw, _nn, scale) =
                        build_l1_n_test_case(seed, density, 32, pw, 100);
                    let (_sw2, _b2, s_pw, n_pw, pw_, _nn2, _sc2) =
                        build_l1_n_test_case(seed, density, 32, pw, 100);

                    let mut avx2_out = vec![0i32; 32];
                    let mut vnni_out = vec![0i32; 32];
                    unsafe {
                        dense_l1_avx2_l1_32(&s_pw, &n_pw, pw_, &sw, &bias, scale, avx2_out.as_mut_ptr());
                        dense_l1_avx512_vnni_l1_32(&s_pw, &n_pw, pw_, &sw, &bias, scale, vnni_out.as_mut_ptr());
                    }
                    assert_eq!(avx2_out, vnni_out,
                        "VNNI L1=32 != AVX2 L1=32 at pw={} density={} seed={}", pw, density, seed);
                }
            }
        }
    }

    /// Fuzz: `dense_l1_avx2` matches the scalar reference. This is the
    /// production L1 kernel for every AVX2-only host, which is most of them,
    /// so it needs a real correctness test rather than just the debug-build
    /// runtime probe in `forward_with_l1_pairwise_inner`.
    ///
    /// Sweeps pw over both net generations (384 = FT768, 512 = FT1024
    /// prod) plus boundary widths, and num_neurons over 16 (prod) and 8
    /// (the `wide=false` half-register path). max_input=100 keeps inputs
    /// inside the VPMADDUBSW saturation envelope, matching production
    /// pairwise output range [0, 127] (FT_SHIFT=9).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn fuzz_dense_avx2_matches_scalar() {
        crate::init();
        if !is_x86_feature_detected!("avx2") {
            eprintln!("No AVX2 — skipping dense_l1_avx2 fuzz test");
            return;
        }

        let mut cases = 0usize;
        for &nn_dim in &[8usize, 16] {
            for &pw in &[64usize, 128, 256, 384, 512] {
                for density in [0u32, 10, 25, 50, 75, 89, 100] {
                    for seed in 0u64..30 {
                        let (sw, bias, s_pw, n_pw, pw_, nn, scale) =
                            build_l1_n_test_case(seed, density, nn_dim, pw, 100);

                        let mut scalar_out = vec![0i32; nn];
                        sparse_l1_scalar(&s_pw, &n_pw, pw_, &sw, nn, &bias, scale, &mut scalar_out);

                        let mut avx2_out = vec![0i32; nn];
                        unsafe {
                            dense_l1_avx2(
                                &s_pw, &n_pw, pw_, &sw, nn, &bias, scale, avx2_out.as_mut_ptr(),
                            );
                        }

                        for i in 0..nn {
                            assert_eq!(
                                avx2_out[i], scalar_out[i],
                                "dense_l1_avx2 mismatch seed={} density={} pw={} nn={} neuron={} avx2={} scalar={}",
                                seed, density, pw, nn, i, avx2_out[i], scalar_out[i]
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
        eprintln!("dense_l1_avx2 fuzz: {} cases passed", cases);
    }

    /// Micro-benchmark for L1 kernel comparison. Ignored by default —
    /// run explicitly via `cargo test --release bench_l1_kernels -- --ignored --nocapture`.
    ///
    /// Reports ns/call for each available kernel at the canonical
    /// pw=384 (HT16 hidden side) shape, plus the scalar reference,
    /// for both L1=16 and L1=32 widths.
    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn bench_l1_kernels() {
        use std::time::Instant;
        crate::init();

        const PW: usize = 384;
        const ITERS: usize = 200_000;
        const WARMUP: usize = 5_000;
        const DENSITY: u32 = 89; // typical pairwise-CReLU density

        // Sink to prevent the optimizer from eliding the work.
        let mut sink: i64 = 0;

        for &nn in &[16usize, 32] {
            let (sw, bias, s_pw, n_pw, pw, nn, scale) =
                build_l1_n_test_case(42, DENSITY, nn, PW, 100);

            // Always run scalar as the speed baseline.
            let mut out = vec![0i32; nn];
            for _ in 0..WARMUP {
                sparse_l1_scalar(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, &mut out);
            }
            let t0 = Instant::now();
            for _ in 0..ITERS {
                sparse_l1_scalar(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, &mut out);
                sink = sink.wrapping_add(out[0] as i64);
            }
            let ns_scalar = t0.elapsed().as_nanos() as f64 / ITERS as f64;
            eprintln!("L1={:<2}  scalar              {:>7.1} ns/call", nn, ns_scalar);

            if !is_x86_feature_detected!("avx2") {
                eprintln!("  (no AVX2 — skipping SIMD kernels)");
                continue;
            }

            // AVX2 path appropriate for the L1 width.
            let mut out = vec![0i32; nn];
            if nn == 16 {
                for _ in 0..WARMUP {
                    unsafe { dense_l1_avx2(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, out.as_mut_ptr()); }
                }
                let t1 = Instant::now();
                for _ in 0..ITERS {
                    unsafe { dense_l1_avx2(&s_pw, &n_pw, pw, &sw, nn, &bias, scale, out.as_mut_ptr()); }
                    sink = sink.wrapping_add(out[0] as i64);
                }
                let ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;
                eprintln!("L1=16  dense_l1_avx2        {:>7.1} ns/call  ({:.1}x scalar)", ns, ns_scalar / ns);
            } else if nn == 32 {
                for _ in 0..WARMUP {
                    unsafe { dense_l1_avx2_l1_32(&s_pw, &n_pw, pw, &sw, &bias, scale, out.as_mut_ptr()); }
                }
                let t1 = Instant::now();
                for _ in 0..ITERS {
                    unsafe { dense_l1_avx2_l1_32(&s_pw, &n_pw, pw, &sw, &bias, scale, out.as_mut_ptr()); }
                    sink = sink.wrapping_add(out[0] as i64);
                }
                let ns = t1.elapsed().as_nanos() as f64 / ITERS as f64;
                eprintln!("L1=32  dense_l1_avx2_l1_32  {:>7.1} ns/call  ({:.1}x scalar)", ns, ns_scalar / ns);
            }
        }

        // Final sink read so dead-code elimination can't kill the loop.
        eprintln!("(sink = {})", sink);
    }
}
