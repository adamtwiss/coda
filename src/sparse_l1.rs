/// Sparse L1 matmul (Reckless dpbusd pattern).
///
/// Instead of processing ALL input elements for each neuron (dense),
/// this skips zero 4-byte chunks of the pairwise output. With ~89%
/// sparsity in pairwise outputs, this processes only ~11% of the work.
///
/// Requires input-chunk-major weight layout:
///   [input_chunk][neuron * 4] instead of [neuron][input]
///
/// The dpbusd kernel: for each non-zero 4-byte input chunk, splat it
/// across an AVX2 register and VPMADDUBSW with the weights for all neurons.

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
    let data32 = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, chunks) };
    let mut count = 0;
    for i in 0..chunks {
        if data32[i] != 0 {
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

    // STM perspective: chunks 0..pw/4
    let stm_chunks = unsafe { std::slice::from_raw_parts(stm_pw.as_ptr() as *const u32, pw / 4) };
    for chunk_idx in 0..pw / 4 {
        if stm_chunks[chunk_idx] == 0 { continue; }
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
    let ntm_chunks = unsafe { std::slice::from_raw_parts(ntm_pw.as_ptr() as *const u32, pw / 4) };
    for chunk_idx in 0..pw / 4 {
        if ntm_chunks[chunk_idx] == 0 { continue; }
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
    let stm_chunks = std::slice::from_raw_parts(stm_pw.as_ptr() as *const u32, pw / 4);
    for chunk_idx in 0..pw / 4 {
        let val = *stm_chunks.get_unchecked(chunk_idx);
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
    let ntm_chunks = std::slice::from_raw_parts(ntm_pw.as_ptr() as *const u32, pw / 4);
    for chunk_idx in 0..pw / 4 {
        let val = *ntm_chunks.get_unchecked(chunk_idx);
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
}
