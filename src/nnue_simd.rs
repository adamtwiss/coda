//! NNUE SIMD primitive abstractions, gated by `cfg(target_feature)`.
//!
//! Step A of the L1-matmul restructure (see
//! `docs/nps_structural_findings_2026-05-01.md`). This module provides
//! ISA-specialised primitives at the granularity of single SIMD ops
//! (splat, dpbusd, reduce, etc.) so that higher-level kernels can be
//! written once against a uniform API and inlined at compile time.
//!
//! Pattern modelled on Reckless's `nnue/simd/` module: each ISA gets
//! its own `cfg(target_feature)`-gated submodule, and the public API
//! is the union re-exported from the highest-priority active ISA.
//!
//! Coda's build (`make`) uses `RUSTFLAGS=-Ctarget-cpu=native`, so each
//! host compiles a binary with its own native features — the cfg gates
//! resolve at compile time. This is the same model Reckless uses; the
//! existing `has_avx512` / `has_avx2` runtime flags in `nnue.rs` remain
//! as belt-and-suspenders fall-through guards but no longer drive the
//! primitive dispatch.
//!
//! Initial primitive set (Step A): just enough for the int8 L1 dot-
//! product hot path to be re-implemented against the abstraction. The
//! set will grow as Step B / C land — this is intentionally minimal so
//! the bench-neutral verification has tight blast radius.

#![allow(dead_code)]

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
mod avx512 {
    use std::arch::x86_64::*;
    use std::mem::size_of;

    pub type I32x = __m512i;
    pub type I16x = __m512i;
    pub type I8x = __m512i;

    pub const I32_LANES: usize = size_of::<__m512i>() / size_of::<i32>(); // 16
    pub const I16_LANES: usize = size_of::<__m512i>() / size_of::<i16>(); // 32
    pub const I8_LANES: usize = size_of::<__m512i>(); // 64

    #[inline(always)]
    pub unsafe fn zeroed_i32() -> I32x {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    pub unsafe fn splat_i32(a: i32) -> I32x {
        _mm512_set1_epi32(a)
    }

    #[inline(always)]
    pub unsafe fn add_i32(a: I32x, b: I32x) -> I32x {
        _mm512_add_epi32(a, b)
    }

    #[inline(always)]
    pub unsafe fn load_i8(p: *const i8) -> I8x {
        _mm512_loadu_si512(p as *const __m512i)
    }

    #[inline(always)]
    pub unsafe fn load_u8(p: *const u8) -> I8x {
        _mm512_loadu_si512(p as *const __m512i)
    }

    /// Fused multiply-add: `acc + dot(u8s, i8s)`, using VPDPBUSD when
    /// `avx512vnni` is present, falling back to the
    /// `vpmaddubsw + vpmaddwd(ones)` pair otherwise.
    #[cfg(target_feature = "avx512vnni")]
    #[inline(always)]
    pub unsafe fn dpbusd(acc: I32x, u8s: I8x, i8s: I8x) -> I32x {
        _mm512_dpbusd_epi32(acc, u8s, i8s)
    }

    #[cfg(not(target_feature = "avx512vnni"))]
    #[inline(always)]
    pub unsafe fn dpbusd(acc: I32x, u8s: I8x, i8s: I8x) -> I32x {
        let pairwise = _mm512_maddubs_epi16(u8s, i8s);
        let widened = _mm512_madd_epi16(pairwise, _mm512_set1_epi16(1));
        _mm512_add_epi32(acc, widened)
    }

    /// Two `dpbusd`s into one accumulator. On VNNI hardware this is
    /// just two back-to-back VPDPBUSDs; on non-VNNI it merges the
    /// pairwise stage so we only do one VPMADDWD instead of two.
    #[cfg(target_feature = "avx512vnni")]
    #[inline(always)]
    pub unsafe fn double_dpbusd(
        acc: I32x,
        u8s1: I8x,
        i8s1: I8x,
        u8s2: I8x,
        i8s2: I8x,
    ) -> I32x {
        _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(acc, u8s1, i8s1), u8s2, i8s2)
    }

    #[cfg(not(target_feature = "avx512vnni"))]
    #[inline(always)]
    pub unsafe fn double_dpbusd(
        acc: I32x,
        u8s1: I8x,
        i8s1: I8x,
        u8s2: I8x,
        i8s2: I8x,
    ) -> I32x {
        let p1 = _mm512_maddubs_epi16(u8s1, i8s1);
        let p2 = _mm512_maddubs_epi16(u8s2, i8s2);
        let widened = _mm512_madd_epi16(_mm512_add_epi16(p1, p2), _mm512_set1_epi16(1));
        _mm512_add_epi32(acc, widened)
    }

    #[inline(always)]
    pub unsafe fn reduce_add_i32(a: I32x) -> i32 {
        _mm512_reduce_add_epi32(a)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
pub use avx512::*;

// ---------------------------------------------------------------------------
// Step B: Reckless-shape sparse-first L1 matmul.
//
// `propagate_l1_avx512_vnni_v2` is the new kernel — single inlined SIMD path,
// NNZ-iterated, pair-unrolled. Compared to `sparse_l1::dense_l1_avx512_vnni`
// (current production), the loop iterates only the precomputed NNZ chunks,
// not all 192 of them. Both kernels produce byte-identical outputs on the
// same input; the parity test below pins this.
//
// The wrapper retains `#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]`
// so the inner primitive calls inline (Step A established this works).
// ---------------------------------------------------------------------------

/// Sparse-first L1 matmul, AVX-512 VNNI. Takes a single combined buffer
/// with STM at `[0..pw)` and NTM at `[pw..2*pw)`. Caller must concatenate
/// the two perspectives. The single-buffer signature avoids the per-NNZ-
/// entry STM-vs-NTM demux branch that data-dependent-branch-mispredicts
/// in production traffic and cost ~10% NPS in an early Step C-cheap test.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi2")]
pub unsafe fn propagate_l1_avx512_vnni_v2(
    combined_pw: &[u8],
    pw: usize,
    sparse_weights: &[i8], // input-chunk-major layout, same as dense_l1_*
    num_neurons: usize,
    bias: &[i16],
    bias_scale: i32,
    nnz_buf: &mut [u16],   // scratch — must hold up to pw/2 indices
    output: &mut [i32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(num_neurons, 16, "v2 specialised to 16 neurons (v9 pairwise)");
    debug_assert!(nnz_buf.len() >= pw / 2, "nnz_buf too small");
    debug_assert!(combined_pw.len() >= 2 * pw, "combined_pw too small");

    let chunk_stride = num_neurons * 4; // 64 bytes — exactly one ZMM register

    // Bias seed.
    for i in 0..num_neurons {
        output[i] = bias[i] as i32 * bias_scale;
    }

    // ---- NNZ scan: produce a list of non-zero 4-byte chunk indices over
    // the entire combined buffer (STM chunks at [0..pw/4), NTM at [pw/4..pw/2)).
    #[cfg(target_feature = "avx512vbmi2")]
    let nnz_count = find_nnz_chunks_avx512_combined(combined_pw, pw, nnz_buf);

    #[cfg(not(target_feature = "avx512vbmi2"))]
    let nnz_count = {
        let chunks = std::slice::from_raw_parts(combined_pw.as_ptr() as *const u32, pw / 2);
        let mut n = 0;
        for c in 0..pw / 2 {
            if *chunks.get_unchecked(c) != 0 {
                *nnz_buf.get_unchecked_mut(n) = c as u16;
                n += 1;
            }
        }
        n
    };

    // ---- Sparse L1 matmul, pair-unrolled, four interleaved accumulators.
    let mut a0 = _mm512_setzero_si512();
    let mut a1 = _mm512_setzero_si512();
    let mut a2 = _mm512_setzero_si512();
    let mut a3 = _mm512_setzero_si512();

    let w_ptr = sparse_weights.as_ptr();
    let combined_u32 = combined_pw.as_ptr() as *const u32;

    let mut k = 0usize;
    while k + 4 <= nnz_count {
        let i0 = *nnz_buf.get_unchecked(k) as usize;
        let i1 = *nnz_buf.get_unchecked(k + 1) as usize;
        let i2 = *nnz_buf.get_unchecked(k + 2) as usize;
        let i3 = *nnz_buf.get_unchecked(k + 3) as usize;
        let v0 = *combined_u32.add(i0);
        let v1 = *combined_u32.add(i1);
        let v2 = *combined_u32.add(i2);
        let v3 = *combined_u32.add(i3);
        let w0 = _mm512_loadu_si512(w_ptr.add(i0 * chunk_stride) as *const __m512i);
        let w1 = _mm512_loadu_si512(w_ptr.add(i1 * chunk_stride) as *const __m512i);
        let w2 = _mm512_loadu_si512(w_ptr.add(i2 * chunk_stride) as *const __m512i);
        let w3 = _mm512_loadu_si512(w_ptr.add(i3 * chunk_stride) as *const __m512i);
        a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(v0 as i32), w0);
        a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32(v1 as i32), w1);
        a2 = _mm512_dpbusd_epi32(a2, _mm512_set1_epi32(v2 as i32), w2);
        a3 = _mm512_dpbusd_epi32(a3, _mm512_set1_epi32(v3 as i32), w3);
        k += 4;
    }
    while k < nnz_count {
        let i = *nnz_buf.get_unchecked(k) as usize;
        let v = *combined_u32.add(i);
        let w = _mm512_loadu_si512(w_ptr.add(i * chunk_stride) as *const __m512i);
        a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(v as i32), w);
        k += 1;
    }

    let acc = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
    let mut results = [0i32; 16];
    _mm512_storeu_si512(results.as_mut_ptr() as *mut __m512i, acc);
    for i in 0..num_neurons {
        output[i] += results[i];
    }
}

/// SIMD NNZ scan: build a list of non-zero 4-byte chunk indices from a
/// concatenated stm/ntm pairwise output. Mirrors Reckless's
/// `find_nnz` AVX-512 VBMI2 path (`vectorized.rs:189-225`).
///
/// Inputs are the two perspective buffers as separate slices (matches
/// existing call signature). Internally treats them as one logical
/// 2*pw chunk-id space — STM at indices `[0..pw/4)`, NTM at `[pw/4..pw/2)`.
///
/// Returns the NNZ count. `nnz_buf.len()` must be ≥ `pw/2`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi2",
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
pub unsafe fn find_nnz_chunks_avx512(
    stm_pw: &[u8],
    ntm_pw: &[u8],
    pw: usize,
    nnz_buf: &mut [u16],
) -> usize {
    use std::arch::x86_64::*;
    debug_assert!(pw % 64 == 0, "pw must be a multiple of 64 for VBMI2 scan");
    debug_assert!(nnz_buf.len() >= pw / 2, "nnz_buf too small");

    // Each ZMM holds 32 i16 lane-indices. We process 64 i32 chunks
    // (= 256 bytes of pairwise output) per iteration: two ZMMs of i32
    // chunks loaded in pairs, mask-compressed, compacted.
    //
    // Identifier conventions follow Reckless avx512vbmi2 path:
    //   base01: indices 0..32 (low half)
    //   base23: indices 32..64 (high half)
    let mut count = 0usize;

    let do_perspective = |buf: &[u8],
                          base_id: u16,
                          nnz_buf: &mut [u16],
                          count: &mut usize| {
        let bytes = pw; // each perspective buffer is `pw` bytes (= pw/4 i32 chunks)
        let chunks = bytes / 4; // i32 chunks per perspective
        let mut i = 0usize;

        let increment = _mm512_set1_epi16(64);
        let mut base01 = _mm512_set_epi16(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        );
        let mut base23 = _mm512_add_epi16(base01, _mm512_set1_epi16(32));
        // Add base_id offset so STM emits 0..pw/4 and NTM emits pw/4..pw/2.
        base01 = _mm512_add_epi16(base01, _mm512_set1_epi16(base_id as i16));
        base23 = _mm512_add_epi16(base23, _mm512_set1_epi16(base_id as i16));

        // 256 bytes = 64 i32 chunks per iteration. We need to read 4 ZMMs
        // of u8 (= 4×64=256 bytes), interpret each as 16 i32 lanes, and
        // produce a mask of nonzero lanes. Two ZMMs cover 32 i32 lanes.
        while i + 256 <= bytes {
            // Use `cmpneq 0` rather than Reckless's `cmpgt 0`: with
            // FT_SHIFT=9 production values are bounded to [0, 127] and
            // both work, but cmpneq is robust if future arch changes
            // raise the upper bound past 0x80 (where any byte hitting
            // the high lane of an i32 chunk would read as negative).
            let m0 = _mm512_cmpneq_epi32_mask(
                _mm512_loadu_si512(buf.as_ptr().add(i) as *const __m512i),
                _mm512_setzero_si512(),
            );
            let m1 = _mm512_cmpneq_epi32_mask(
                _mm512_loadu_si512(buf.as_ptr().add(i + 64) as *const __m512i),
                _mm512_setzero_si512(),
            );
            let m2 = _mm512_cmpneq_epi32_mask(
                _mm512_loadu_si512(buf.as_ptr().add(i + 128) as *const __m512i),
                _mm512_setzero_si512(),
            );
            let m3 = _mm512_cmpneq_epi32_mask(
                _mm512_loadu_si512(buf.as_ptr().add(i + 192) as *const __m512i),
                _mm512_setzero_si512(),
            );
            let mask01 = _mm512_kunpackw(m1 as u32, m0 as u32);
            let mask23 = _mm512_kunpackw(m3 as u32, m2 as u32);
            let compressed01 = _mm512_maskz_compress_epi16(mask01, base01);
            let compressed23 = _mm512_maskz_compress_epi16(mask23, base23);

            _mm512_storeu_si512(nnz_buf.as_mut_ptr().add(*count) as *mut __m512i, compressed01);
            *count += (mask01 as u32).count_ones() as usize;

            _mm512_storeu_si512(nnz_buf.as_mut_ptr().add(*count) as *mut __m512i, compressed23);
            *count += (mask23 as u32).count_ones() as usize;

            base01 = _mm512_add_epi16(base01, increment);
            base23 = _mm512_add_epi16(base23, increment);
            i += 256;
        }

        // Tail: any remaining chunks (less than 64). Walk scalar.
        let total_so_far_chunk_id = base_id as usize + i / 4;
        let scalar_remaining_chunks = chunks - i / 4;
        let scalar_chunks = std::slice::from_raw_parts(
            buf.as_ptr().add(i) as *const u32,
            scalar_remaining_chunks,
        );
        for c in 0..scalar_remaining_chunks {
            if *scalar_chunks.get_unchecked(c) != 0 {
                *nnz_buf.get_unchecked_mut(*count) = (total_so_far_chunk_id + c) as u16;
                *count += 1;
            }
        }
    };

    do_perspective(stm_pw, 0, nnz_buf, &mut count);
    do_perspective(ntm_pw, (pw / 4) as u16, nnz_buf, &mut count);

    count
}

/// Combined-buffer variant of `find_nnz_chunks_avx512`. Caller passes a
/// single contiguous buffer with both perspectives concatenated; the
/// scan emits chunk-ids in `[0..2*pw/4)` directly, no per-perspective
/// remap. Faster than the two-slice variant because the matmul loop
/// reads via one base pointer with no demux branch.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi2",
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
pub unsafe fn find_nnz_chunks_avx512_combined(
    combined_pw: &[u8],
    pw: usize,
    nnz_buf: &mut [u16],
) -> usize {
    use std::arch::x86_64::*;
    debug_assert!(nnz_buf.len() >= pw / 2);
    debug_assert!(combined_pw.len() >= 2 * pw);

    let total_bytes = 2 * pw; // both perspectives concatenated
    let mut count = 0usize;
    let mut i = 0usize;

    let increment = _mm512_set1_epi16(64);
    let mut base01 = _mm512_set_epi16(
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    );
    let mut base23 = _mm512_add_epi16(base01, _mm512_set1_epi16(32));

    while i + 256 <= total_bytes {
        let m0 = _mm512_cmpneq_epi32_mask(
            _mm512_loadu_si512(combined_pw.as_ptr().add(i) as *const __m512i),
            _mm512_setzero_si512(),
        );
        let m1 = _mm512_cmpneq_epi32_mask(
            _mm512_loadu_si512(combined_pw.as_ptr().add(i + 64) as *const __m512i),
            _mm512_setzero_si512(),
        );
        let m2 = _mm512_cmpneq_epi32_mask(
            _mm512_loadu_si512(combined_pw.as_ptr().add(i + 128) as *const __m512i),
            _mm512_setzero_si512(),
        );
        let m3 = _mm512_cmpneq_epi32_mask(
            _mm512_loadu_si512(combined_pw.as_ptr().add(i + 192) as *const __m512i),
            _mm512_setzero_si512(),
        );
        let mask01 = _mm512_kunpackw(m1 as u32, m0 as u32);
        let mask23 = _mm512_kunpackw(m3 as u32, m2 as u32);
        let compressed01 = _mm512_maskz_compress_epi16(mask01, base01);
        let compressed23 = _mm512_maskz_compress_epi16(mask23, base23);

        _mm512_storeu_si512(nnz_buf.as_mut_ptr().add(count) as *mut __m512i, compressed01);
        count += (mask01 as u32).count_ones() as usize;
        _mm512_storeu_si512(nnz_buf.as_mut_ptr().add(count) as *mut __m512i, compressed23);
        count += (mask23 as u32).count_ones() as usize;

        base01 = _mm512_add_epi16(base01, increment);
        base23 = _mm512_add_epi16(base23, increment);
        i += 256;
    }

    // Scalar tail.
    let chunks_so_far = i / 4;
    let remaining_chunks = total_bytes / 4 - chunks_so_far;
    let scalar_chunks = std::slice::from_raw_parts(
        combined_pw.as_ptr().add(i) as *const u32,
        remaining_chunks,
    );
    for c in 0..remaining_chunks {
        if *scalar_chunks.get_unchecked(c) != 0 {
            *nnz_buf.get_unchecked_mut(count) = (chunks_so_far + c) as u16;
            count += 1;
        }
    }

    count
}

/// Step B diagnostic — pure matmul on a pre-computed NNZ list. Same body
/// as `propagate_l1_avx512_vnni_v2` but with the NNZ scan hoisted out, so
/// the microbench can decompose "sparse iteration win" from "scan
/// overhead". Not for production use — production callers must produce
/// the NNZ list themselves.
/// Step B diagnostic — pure matmul on a pre-computed NNZ list, combined-
/// buffer signature matching `propagate_l1_avx512_vnni_v2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn propagate_l1_matmul_only(
    combined_pw: &[u8],
    _pw: usize,
    sparse_weights: &[i8],
    num_neurons: usize,
    bias: &[i16],
    bias_scale: i32,
    nnz: &[u16],
    output: &mut [i32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(num_neurons, 16);

    let chunk_stride = num_neurons * 4;
    for i in 0..num_neurons {
        output[i] = bias[i] as i32 * bias_scale;
    }

    let mut a0 = _mm512_setzero_si512();
    let mut a1 = _mm512_setzero_si512();
    let mut a2 = _mm512_setzero_si512();
    let mut a3 = _mm512_setzero_si512();
    let w_ptr = sparse_weights.as_ptr();
    let combined_u32 = combined_pw.as_ptr() as *const u32;

    let mut k = 0;
    while k + 4 <= nnz.len() {
        let i0 = *nnz.get_unchecked(k) as usize;
        let i1 = *nnz.get_unchecked(k + 1) as usize;
        let i2 = *nnz.get_unchecked(k + 2) as usize;
        let i3 = *nnz.get_unchecked(k + 3) as usize;
        let v0 = *combined_u32.add(i0);
        let v1 = *combined_u32.add(i1);
        let v2 = *combined_u32.add(i2);
        let v3 = *combined_u32.add(i3);
        let w0 = _mm512_loadu_si512(w_ptr.add(i0 * chunk_stride) as *const __m512i);
        let w1 = _mm512_loadu_si512(w_ptr.add(i1 * chunk_stride) as *const __m512i);
        let w2 = _mm512_loadu_si512(w_ptr.add(i2 * chunk_stride) as *const __m512i);
        let w3 = _mm512_loadu_si512(w_ptr.add(i3 * chunk_stride) as *const __m512i);
        a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(v0 as i32), w0);
        a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32(v1 as i32), w1);
        a2 = _mm512_dpbusd_epi32(a2, _mm512_set1_epi32(v2 as i32), w2);
        a3 = _mm512_dpbusd_epi32(a3, _mm512_set1_epi32(v3 as i32), w3);
        k += 4;
    }
    while k < nnz.len() {
        let i = *nnz.get_unchecked(k) as usize;
        let v = *combined_u32.add(i);
        let w = _mm512_loadu_si512(w_ptr.add(i * chunk_stride) as *const __m512i);
        a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(v as i32), w);
        k += 1;
    }

    let acc = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
    let mut results = [0i32; 16];
    _mm512_storeu_si512(results.as_mut_ptr() as *mut __m512i, acc);
    for i in 0..num_neurons {
        output[i] += results[i];
    }
}

// Future ISA submodules (avx2, neon, scalar) will be added in Step B/C
// as the new kernel needs them.

#[cfg(test)]
mod tests {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    #[test]
    fn dpbusd_matches_scalar_reference() {
        unsafe {
            let u8s: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(3));
            let i8s: [i8; 64] = std::array::from_fn(|i| ((i as i8).wrapping_mul(11)).wrapping_sub(5));

            let acc = super::splat_i32(1234);
            let v_u8 = super::load_u8(u8s.as_ptr());
            let v_i8 = super::load_i8(i8s.as_ptr());
            let result = super::dpbusd(acc, v_u8, v_i8);
            let v_sum = super::reduce_add_i32(result);

            // Scalar reference: dpbusd accumulates 4-byte int8×uint8 dot
            // products into 16 i32 lanes, then reduce-add. Equivalent to:
            // sum(u8s[i] as i32 * i8s[i] as i32) + lanes * 1234
            let lane_count = super::I32_LANES as i32;
            let mut expected: i32 = 1234 * lane_count;
            for i in 0..64 {
                expected += u8s[i] as i32 * i8s[i] as i32;
            }
            assert_eq!(v_sum, expected, "dpbusd ≠ scalar reference");
        }
    }

    /// Parity gate for Step B: the new sparse-first kernel must produce
    /// byte-identical outputs to the existing dense kernel on the same input.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni",
    ))]
    #[test]
    fn propagate_l1_v2_parity_with_dense() {
        unsafe {
            // Realistic v9 op-point: hidden=768, pw=384, 16 neurons.
            const PW: usize = 384;
            const NEURONS: usize = 16;

            // Build pairwise outputs with ~89% sparsity (Coda v9 measured).
            // Use a deterministic LCG so the test is reproducible.
            let mut state: u64 = 0xC0DAC0DA;
            let mut next = || -> u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u32
            };
            let mut stm_pw = vec![0u8; PW];
            let mut ntm_pw = vec![0u8; PW];
            // Emit nonzero only ~11% of 4-byte chunks; within a nonzero chunk,
            // emit nonzero u8 values for ~half the bytes (matches pairwise CReLU).
            for c in 0..PW / 4 {
                if next() % 100 < 11 {
                    for b in 0..4 {
                        stm_pw[c * 4 + b] = ((next() & 0xFF) as u8) | 1;
                    }
                }
                if next() % 100 < 11 {
                    for b in 0..4 {
                        ntm_pw[c * 4 + b] = ((next() & 0xFF) as u8) | 1;
                    }
                }
            }

            // Input-chunk-major weights: chunk_stride = NEURONS * 4 = 64 bytes.
            let total_chunks = PW / 2; // STM + NTM chunk-id range
            let mut weights = vec![0i8; total_chunks * NEURONS * 4];
            for w in weights.iter_mut() {
                *w = (next() & 0x7F) as i8 - 64; // i8 range
            }
            let bias: Vec<i16> = (0..NEURONS).map(|i| (i as i16 * 7) - 56).collect();
            let bias_scale: i32 = 127;

            let mut out_dense = vec![0i32; NEURONS];
            let mut out_v2 = vec![0i32; NEURONS];
            let mut nnz_buf = vec![0u16; total_chunks];

            // Build combined buffer for v2 (STM at [0..PW), NTM at [PW..2*PW)).
            let mut combined_pw = vec![0u8; 2 * PW];
            combined_pw[..PW].copy_from_slice(&stm_pw);
            combined_pw[PW..].copy_from_slice(&ntm_pw);

            crate::sparse_l1::dense_l1_avx512_vnni(
                &stm_pw, &ntm_pw, PW, &weights, NEURONS, &bias, bias_scale, &mut out_dense,
            );
            super::propagate_l1_avx512_vnni_v2(
                &combined_pw, PW, &weights, NEURONS, &bias, bias_scale, &mut nnz_buf, &mut out_v2,
            );

            for i in 0..NEURONS {
                assert_eq!(
                    out_dense[i], out_v2[i],
                    "v2 ≠ dense at neuron {}: dense={} v2={}",
                    i, out_dense[i], out_v2[i],
                );
            }
        }
    }

    /// Step B microbench — falsifiable gate. Times the current production
    /// dense kernel against the new sparse-first kernel on identical
    /// realistic input. Hercules's prediction: 6-8× cycle reduction at the
    /// per-call level. Run with:
    ///   cargo test --release nnue_simd::tests::l1_perf -- --nocapture --ignored
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni",
    ))]
    #[test]
    #[ignore]
    fn l1_perf_dense_vs_v2() {
        use std::time::Instant;

        unsafe {
            const PW: usize = 384;
            const NEURONS: usize = 16;
            const ITERS: usize = 1_000_000;

            let mut state: u64 = 0xC0DAC0DA;
            let mut next = || -> u32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as u32
            };

            // Build a realistic 89%-sparse pairwise input, same shape as the
            // parity test but reproducible.
            let mut stm_pw = vec![0u8; PW];
            let mut ntm_pw = vec![0u8; PW];
            let mut nnz_in = 0;
            for c in 0..PW / 4 {
                if next() % 100 < 11 {
                    for b in 0..4 {
                        stm_pw[c * 4 + b] = ((next() & 0xFF) as u8) | 1;
                    }
                    nnz_in += 1;
                }
                if next() % 100 < 11 {
                    for b in 0..4 {
                        ntm_pw[c * 4 + b] = ((next() & 0xFF) as u8) | 1;
                    }
                    nnz_in += 1;
                }
            }

            let total_chunks = PW / 2;
            let mut weights = vec![0i8; total_chunks * NEURONS * 4];
            for w in weights.iter_mut() {
                *w = (next() & 0x7F) as i8 - 64;
            }
            let bias: Vec<i16> = (0..NEURONS).map(|i| (i as i16 * 7) - 56).collect();
            let bias_scale: i32 = 127;

            let mut out_dense = vec![0i32; NEURONS];
            let mut out_v2 = vec![0i32; NEURONS];
            let mut nnz_buf = vec![0u16; total_chunks];

            // Combined buffer for v2.
            let mut combined_pw = vec![0u8; 2 * PW];
            combined_pw[..PW].copy_from_slice(&stm_pw);
            combined_pw[PW..].copy_from_slice(&ntm_pw);

            // Warm caches with one call each.
            crate::sparse_l1::dense_l1_avx512_vnni(
                &stm_pw, &ntm_pw, PW, &weights, NEURONS, &bias, bias_scale, &mut out_dense,
            );
            super::propagate_l1_avx512_vnni_v2(
                &combined_pw, PW, &weights, NEURONS, &bias, bias_scale, &mut nnz_buf, &mut out_v2,
            );

            // Pre-compute NNZ buf once for the matmul-only timing path.
            let mut nnz_count_pre = 0;
            {
                let combined_u32 = std::slice::from_raw_parts(
                    combined_pw.as_ptr() as *const u32,
                    PW / 2,
                );
                for c in 0..PW / 2 {
                    if *combined_u32.get_unchecked(c) != 0 {
                        nnz_buf[nnz_count_pre] = c as u16;
                        nnz_count_pre += 1;
                    }
                }
            }

            // Time dense.
            let t0 = Instant::now();
            for _ in 0..ITERS {
                crate::sparse_l1::dense_l1_avx512_vnni(
                    &stm_pw, &ntm_pw, PW, &weights, NEURONS, &bias, bias_scale, &mut out_dense,
                );
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
            }
            let dense_ns = t0.elapsed().as_nanos() as u64;

            // Time v2 (NNZ scan included — apples-to-apples with dense from a
            // production caller's perspective: every call must scan).
            let t1 = Instant::now();
            for _ in 0..ITERS {
                super::propagate_l1_avx512_vnni_v2(
                    &combined_pw, PW, &weights, NEURONS, &bias, bias_scale, &mut nnz_buf, &mut out_v2,
                );
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
            }
            let v2_ns = t1.elapsed().as_nanos() as u64;

            // Time matmul-only (pre-computed NNZ). Reveals the ceiling
            // *iteration* of sparse-vs-dense, separating scan overhead.
            let t2 = Instant::now();
            for _ in 0..ITERS {
                super::propagate_l1_matmul_only(
                    &combined_pw, PW, &weights, NEURONS, &bias, bias_scale,
                    &nnz_buf[..nnz_count_pre], &mut out_v2,
                );
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
            }
            let matmul_ns = t2.elapsed().as_nanos() as u64;

            let dense_per = dense_ns as f64 / ITERS as f64;
            let v2_per = v2_ns as f64 / ITERS as f64;
            let matmul_per = matmul_ns as f64 / ITERS as f64;
            let ratio_v2 = dense_per / v2_per;
            let ratio_matmul = dense_per / matmul_per;

            eprintln!();
            eprintln!("=== L1 matmul microbench (PW={}, NEURONS={}, NNZ_chunks={}/{}, iters={}) ===",
                      PW, NEURONS, nnz_in, PW / 2, ITERS);
            eprintln!("  dense_l1_avx512_vnni       : {:>8.2} ns/call", dense_per);
            eprintln!("  propagate_l1_v2 (w/ scan)  : {:>8.2} ns/call → {:.2}× vs dense", v2_per, ratio_v2);
            eprintln!("  matmul_only (no scan)      : {:>8.2} ns/call → {:.2}× vs dense", matmul_per, ratio_matmul);
            eprintln!();
            eprintln!("  scan overhead in v2 path   : {:>8.2} ns/call ({:.0}% of v2 cost)",
                      v2_per - matmul_per,
                      (v2_per - matmul_per) / v2_per * 100.0);
            eprintln!("  gate (6× target)           : {}",
                      if ratio_v2 >= 6.0 {
                          "✓ HIT (apples-to-apples, scan included)"
                      } else if ratio_matmul >= 6.0 {
                          "✓ HIT (matmul-only); scan overhead is the bottleneck — vectorise NNZ scan"
                      } else if ratio_matmul >= 3.0 {
                          "partial — sparse iteration helps but not 6×; model needs adjusting"
                      } else {
                          "MISSED — even matmul-only doesn't hit 3×; dense is closer to optimal than expected"
                      });

            // Final-output sanity check (parity_with_dense covers correctness;
            // here we just guard against the optimizer eliding the work).
            assert_eq!(out_dense[0], out_v2[0]);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    #[test]
    fn double_dpbusd_matches_two_single_dpbusd() {
        unsafe {
            let u8a: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(13));
            let u8b: [u8; 64] = std::array::from_fn(|i| (i as u8).wrapping_mul(17));
            let i8a: [i8; 64] = std::array::from_fn(|i| (i as i8).wrapping_mul(3));
            let i8b: [i8; 64] = std::array::from_fn(|i| (i as i8).wrapping_mul(5));

            let acc = super::zeroed_i32();
            let va = super::load_u8(u8a.as_ptr());
            let vb = super::load_u8(u8b.as_ptr());
            let wa = super::load_i8(i8a.as_ptr());
            let wb = super::load_i8(i8b.as_ptr());

            let r_double = super::double_dpbusd(acc, va, wa, vb, wb);
            let r_singles = super::dpbusd(super::dpbusd(acc, va, wa), vb, wb);

            assert_eq!(
                super::reduce_add_i32(r_double),
                super::reduce_add_i32(r_singles),
                "double_dpbusd ≠ two single dpbusds"
            );
        }
    }
}
