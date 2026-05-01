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

// Future ISA submodules (avx2, neon, scalar) will be added in Step B
// as the new propagate_l1 needs them. Keeping the surface minimal for
// Step A's bench-neutral verification.

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
