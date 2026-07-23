//! x86 AVX/F16C helpers for attention kernels (KV cache f32↔f16 path).
//!
//! Split out from `attention/mod.rs` in mc74 cleanup.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

// --- x86 AVX/F16C helpers for KV cache ---

/// x86 dot product: a(f32) · b(f16 as u16) -> f32
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn avx2_dot_f32_f16(a: *const f32, b_f16: *const u16, len: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= len {
        let a0 = _mm256_loadu_ps(a.add(i));
        let b0 = _mm_loadu_si128(b_f16.add(i) as *const __m128i);
        let b0_f32 = _mm256_cvtph_ps(b0);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0_f32));

        let a1 = _mm256_loadu_ps(a.add(i + 8));
        let b1 = _mm_loadu_si128(b_f16.add(i + 8) as *const __m128i);
        let b1_f32 = _mm256_cvtph_ps(b1);
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1_f32));
        i += 16;
    }
    while i + 8 <= len {
        let av = _mm256_loadu_ps(a.add(i));
        let bv = _mm_loadu_si128(b_f16.add(i) as *const __m128i);
        let bv_f32 = _mm256_cvtph_ps(bv);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av, bv_f32));
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let mut total = _mm_cvtss_f32(_mm_add_ss(sums, shuf));

    while i < len {
        total += *a.add(i) * half::f16::from_bits(*b_f16.add(i)).to_f32();
        i += 1;
    }
    total
}

/// out[d] += scale * v_f16[d], x86 AVX/F16C accelerated.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,f16c")]
pub(super) unsafe fn avx2_scaled_add_f16(out: *mut f32, v_f16: *const u16, scale: f32, len: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let s = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= len {
        let o = _mm256_loadu_ps(out.add(i));
        let v = _mm_loadu_si128(v_f16.add(i) as *const __m128i);
        let v_f32 = _mm256_cvtph_ps(v);
        _mm256_storeu_ps(out.add(i), _mm256_add_ps(o, _mm256_mul_ps(v_f32, s)));
        i += 8;
    }
    while i < len {
        *out.add(i) += scale * half::f16::from_bits(*v_f16.add(i)).to_f32();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn avx_dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= len {
        let a0 = _mm256_loadu_ps(a.add(i));
        let b0 = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(a0, b0));
        let a1 = _mm256_loadu_ps(a.add(i + 8));
        let b1 = _mm256_loadu_ps(b.add(i + 8));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(a1, b1));
        i += 16;
    }
    while i + 8 <= len {
        let av = _mm256_loadu_ps(a.add(i));
        let bv = _mm256_loadu_ps(b.add(i));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av, bv));
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let mut total = _mm_cvtss_f32(_mm_add_ss(sums, shuf));

    while i < len {
        total += *a.add(i) * *b.add(i);
        i += 1;
    }
    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn avx_scale_f32(out: *mut f32, scale: f32, len: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let s = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= len {
        let o = _mm256_loadu_ps(out.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_mul_ps(o, s));
        i += 8;
    }
    while i < len {
        *out.add(i) *= scale;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub(super) unsafe fn avx_scaled_add_f32(out: *mut f32, v: *const f32, scale: f32, len: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let s = _mm256_set1_ps(scale);
    let mut i = 0;
    while i + 8 <= len {
        let o = _mm256_loadu_ps(out.add(i));
        let x = _mm256_loadu_ps(v.add(i));
        _mm256_storeu_ps(out.add(i), _mm256_add_ps(o, _mm256_mul_ps(x, s)));
        i += 8;
    }
    while i < len {
        *out.add(i) += scale * *v.add(i);
        i += 1;
    }
}
