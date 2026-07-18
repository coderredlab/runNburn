//! aarch64 NEON SIMD helpers for attention kernels.
//!
//! mc72 added native fp16 helpers (`neon_vec_scale_f16`,
//! `neon_vec_dot_f16_f16`, `neon_vec_mad_f16`) requiring FEAT_FP16.
//! Split out from `attention/mod.rs` in mc74 cleanup.

#![cfg(target_arch = "aarch64")]

// --- NEON SIMD helpers ---

/// NEON f32 dot product for head_dim-length vectors.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn neon_dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 8 <= len {
        let a0 = vld1q_f32(a.add(i));
        let b0 = vld1q_f32(b.add(i));
        acc0 = vfmaq_f32(acc0, a0, b0);
        let a1 = vld1q_f32(a.add(i + 4));
        let b1 = vld1q_f32(b.add(i + 4));
        acc1 = vfmaq_f32(acc1, a1, b1);
        i += 8;
    }
    while i + 4 <= len {
        let a0 = vld1q_f32(a.add(i));
        let b0 = vld1q_f32(b.add(i));
        acc0 = vfmaq_f32(acc0, a0, b0);
        i += 4;
    }
    acc0 = vaddq_f32(acc0, acc1);
    vaddvq_f32(acc0)
}

/// out[d] += scale * v[d], NEON accelerated
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn neon_scaled_add(out: *mut f32, v: *const f32, scale: f32, len: usize) {
    use std::arch::aarch64::*;
    let s = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= len {
        let o = vld1q_f32(out.add(i));
        let x = vld1q_f32(v.add(i));
        vst1q_f32(out.add(i), vfmaq_f32(o, x, s));
        i += 4;
    }
}

/// out[d] *= scale, NEON accelerated
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn neon_scale(out: *mut f32, scale: f32, len: usize) {
    use std::arch::aarch64::*;
    let s = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= len {
        let o = vld1q_f32(out.add(i));
        vst1q_f32(out.add(i), vmulq_f32(o, s));
        i += 4;
    }
}

// --- NEON F16→F32 helpers for KV cache ---

/// NEON dot product: a(f32) · b(f16 as u16) → f32.
///
/// Reduction tree aligned with GGML's `ggml_vec_dot_f16` fp16-fallback path:
/// 4-way accumulator (sum0..sum3) + 16-element STEP, reducing pairwise as
/// (sum0+sum2, sum1+sum3) -> (s0+s1) -> vaddvq_f32. mc71 finding: our old
/// 2-way accumulator at 8-element STEP produced bit-different results vs
/// llama.cpp on the same NEON hardware, contributing to raw-mode token
/// divergence at small logit margins.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn neon_dot_f32_f16(a: *const f32, b_f16: *const u16, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 16 <= len {
        let a0 = vld1q_f32(a.add(i));
        let b0_f32: float32x4_t;
        core::arch::asm!(
            "ldr d0, [{ptr}]",
            "fcvtl {v}.4s, v0.4h",
            ptr = in(reg) (b_f16 as *const u8).add(i * 2),
            v = lateout(vreg) b0_f32,
            out("v0") _,
        );
        sum0 = vfmaq_f32(sum0, a0, b0_f32);

        let a1 = vld1q_f32(a.add(i + 4));
        let b1_f32: float32x4_t;
        core::arch::asm!(
            "ldr d0, [{ptr}]",
            "fcvtl {v}.4s, v0.4h",
            ptr = in(reg) (b_f16 as *const u8).add((i + 4) * 2),
            v = lateout(vreg) b1_f32,
            out("v0") _,
        );
        sum1 = vfmaq_f32(sum1, a1, b1_f32);

        let a2 = vld1q_f32(a.add(i + 8));
        let b2_f32: float32x4_t;
        core::arch::asm!(
            "ldr d0, [{ptr}]",
            "fcvtl {v}.4s, v0.4h",
            ptr = in(reg) (b_f16 as *const u8).add((i + 8) * 2),
            v = lateout(vreg) b2_f32,
            out("v0") _,
        );
        sum2 = vfmaq_f32(sum2, a2, b2_f32);

        let a3 = vld1q_f32(a.add(i + 12));
        let b3_f32: float32x4_t;
        core::arch::asm!(
            "ldr d0, [{ptr}]",
            "fcvtl {v}.4s, v0.4h",
            ptr = in(reg) (b_f16 as *const u8).add((i + 12) * 2),
            v = lateout(vreg) b3_f32,
            out("v0") _,
        );
        sum3 = vfmaq_f32(sum3, a3, b3_f32);

        i += 16;
    }
    while i + 4 <= len {
        let a0 = vld1q_f32(a.add(i));
        let b0_f32: float32x4_t;
        core::arch::asm!(
            "ldr d0, [{ptr}]",
            "fcvtl {v}.4s, v0.4h",
            ptr = in(reg) (b_f16 as *const u8).add(i * 2),
            v = lateout(vreg) b0_f32,
            out("v0") _,
        );
        sum0 = vfmaq_f32(sum0, a0, b0_f32);
        i += 4;
    }
    // GGML reduction tree: pairs first (0+2, 1+3), then single sum, then
    // horizontal add. This matches `GGML_F32x4_REDUCE` in
    // `ggml-cpu/simd-mappings.h` exactly.
    let s02 = vaddq_f32(sum0, sum2);
    let s13 = vaddq_f32(sum1, sum3);
    let s = vaddq_f32(s02, s13);
    vaddvq_f32(s)
}

/// out[d] += scale * v_f16[d], NEON accelerated with F16→F32 conversion
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) unsafe fn neon_scaled_add_f16(out: *mut f32, v_f16: *const u16, scale: f32, len: usize) {
    use std::arch::aarch64::*;
    let s = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= len {
        let o = vld1q_f32(out.add(i));
        let v_f32: float32x4_t;
        core::arch::asm!(
            "ldr d0, [{ptr}]",
            "fcvtl {v}.4s, v0.4h",
            ptr = in(reg) (v_f16 as *const u8).add(i * 2),
            v = lateout(vreg) v_f32,
            out("v0") _,
        );
        vst1q_f32(out.add(i), vfmaq_f32(o, v_f32, s));
        i += 4;
    }
}

// --- ARMv8.2-A FP16 vector arithmetic helpers (native vmulq_f16 / vfmaq_f16).
//
// Matches GGML's `ggml_vec_scale_f16` / `ggml_vec_mad_f16` exactly: f16
// accumulator stays in f16 throughout, single-rounding multiply / FMA per
// 8-lane vector. Requires FEAT_FP16 (every modern Android cpu — A55+/X1+/A76+).

/// f16 in-place scale: `acc[d] *= alpha`. Operates on raw u16 bits matching
/// `ggml_fp16_t`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub(super) unsafe fn neon_vec_scale_f16(acc: *mut u16, alpha_bits: u16, len: usize) {
    use std::arch::aarch64::*;
    let alpha_v = vdupq_n_f16(f16::from_bits(alpha_bits));
    let mut i = 0;
    while i + 8 <= len {
        let a = vld1q_f16(acc.add(i) as *const f16);
        let r = vmulq_f16(a, alpha_v);
        vst1q_f16(acc.add(i) as *mut f16, r);
        i += 8;
    }
    while i < len {
        let v = f16::from_bits(*acc.add(i));
        let alpha = f16::from_bits(alpha_bits);
        *acc.add(i) = (v * alpha).to_bits();
        i += 1;
    }
}

/// f16 × f16 dot product with native FMA. Matches GGML's `ggml_vec_dot_f16`
/// native fp16 path exactly: GGML_F16_STEP=32, ARR=4, EPR=8 — 4 f16 vector
/// accumulators of 8 lanes each, 32-element step. Reduction tree is pairwise
/// in f16 (sum 0+1, 2+3, then 0+2), then convert low/high halves to f32 and
/// sum. Tail handled in 8-element f16 single-acc, then scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub(super) unsafe fn neon_vec_dot_f16_f16(a: *const u16, b: *const u16, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let zero = 0.0_f16;
    let mut sum0 = vdupq_n_f16(zero);
    let mut sum1 = vdupq_n_f16(zero);
    let mut sum2 = vdupq_n_f16(zero);
    let mut sum3 = vdupq_n_f16(zero);
    let mut i = 0;
    while i + 32 <= len {
        let a0 = vld1q_f16(a.add(i) as *const f16);
        let b0 = vld1q_f16(b.add(i) as *const f16);
        sum0 = vfmaq_f16(sum0, a0, b0);

        let a1 = vld1q_f16(a.add(i + 8) as *const f16);
        let b1 = vld1q_f16(b.add(i + 8) as *const f16);
        sum1 = vfmaq_f16(sum1, a1, b1);

        let a2 = vld1q_f16(a.add(i + 16) as *const f16);
        let b2 = vld1q_f16(b.add(i + 16) as *const f16);
        sum2 = vfmaq_f16(sum2, a2, b2);

        let a3 = vld1q_f16(a.add(i + 24) as *const f16);
        let b3 = vld1q_f16(b.add(i + 24) as *const f16);
        sum3 = vfmaq_f16(sum3, a3, b3);

        i += 32;
    }
    while i + 8 <= len {
        let a0 = vld1q_f16(a.add(i) as *const f16);
        let b0 = vld1q_f16(b.add(i) as *const f16);
        sum0 = vfmaq_f16(sum0, a0, b0);
        i += 8;
    }
    // GGML_F16x8_REDUCE: pairwise add in f16, then to f32 and horizontal sum.
    sum0 = vaddq_f16(sum0, sum1);
    sum2 = vaddq_f16(sum2, sum3);
    sum0 = vaddq_f16(sum0, sum2);
    let t0 = vcvt_f32_f16(vget_low_f16(sum0));
    let t1 = vcvt_f32_f16(vget_high_f16(sum0));
    let mut sumf = vaddvq_f32(vaddq_f32(t0, t1));
    // Scalar tail (matches GGML's `(ggml_float)(GGML_FP16_TO_FP32(x[i]) * GGML_FP16_TO_FP32(y[i]))`).
    while i < len {
        let av = f16::from_bits(*a.add(i));
        let bv = f16::from_bits(*b.add(i));
        sumf += (av as f32) * (bv as f32);
        i += 1;
    }
    sumf
}

/// f16 fused mad: `acc[d] += v[d] * scale` with single-rounding FMA.
/// Matches GGML `ggml_vec_mad_f16`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
pub(super) unsafe fn neon_vec_mad_f16(acc: *mut u16, v: *const u16, scale_bits: u16, len: usize) {
    use std::arch::aarch64::*;
    let scale_v = vdupq_n_f16(f16::from_bits(scale_bits));
    let mut i = 0;
    while i + 8 <= len {
        let a = vld1q_f16(acc.add(i) as *const f16);
        let vv = vld1q_f16(v.add(i) as *const f16);
        let r = vfmaq_f16(a, vv, scale_v);
        vst1q_f16(acc.add(i) as *mut f16, r);
        i += 8;
    }
    while i < len {
        let acc_v = f16::from_bits(*acc.add(i));
        let v_v = f16::from_bits(*v.add(i));
        let scale = f16::from_bits(scale_bits);
        // mul_add = (v_v * scale) + acc_v with single rounding (matches FMA).
        *acc.add(i) = v_v.mul_add(scale, acc_v).to_bits();
        i += 1;
    }
}
