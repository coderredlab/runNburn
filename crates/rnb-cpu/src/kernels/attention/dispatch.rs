//! Cross-arch dispatch wrappers for attention kernels.
//!
//! Each wrapper picks the best path at runtime: NEON on aarch64,
//! AVX/F16C on x86 (when detected), scalar fallback otherwise.
//! Split out from `attention/mod.rs` in mc74 cleanup.

#[cfg(target_arch = "aarch64")]
use super::neon_helpers::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::x86_helpers::*;

// --- Dispatch wrappers (NEON if aarch64 + len >= 4, else scalar) ---

/// dot product: a(f32) · b(f16) → f32
#[inline(always)]
pub(super) fn dot_f32_f16(a: &[f32], b_f16: &[u16], len: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if len >= 4 {
        return unsafe { neon_dot_f32_f16(a.as_ptr(), b_f16.as_ptr(), len) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if len >= 8 && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("f16c") {
        return unsafe { avx2_dot_f32_f16(a.as_ptr(), b_f16.as_ptr(), len) };
    }
    let mut acc = 0.0f32;
    for d in 0..len {
        acc += a[d] * half::f16::from_bits(b_f16[d]).to_f32();
    }
    acc
}

/// out[d] += scale * v_f16[d]
#[inline(always)]
pub(super) fn scaled_add_f16(out: &mut [f32], v_f16: &[u16], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    if out.len() >= 4 {
        unsafe {
            neon_scaled_add_f16(out.as_mut_ptr(), v_f16.as_ptr(), scale, out.len());
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if out.len() >= 8
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("f16c")
    {
        unsafe {
            avx2_scaled_add_f16(out.as_mut_ptr(), v_f16.as_ptr(), scale, out.len());
        }
        return;
    }
    for d in 0..out.len() {
        out[d] += scale * half::f16::from_bits(v_f16[d]).to_f32();
    }
}

#[inline(always)]
pub(super) fn dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if len >= 4 {
        return unsafe { neon_dot_f32(a.as_ptr(), b.as_ptr(), len) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if len >= 8 && std::is_x86_feature_detected!("avx") {
        return unsafe { avx_dot_f32(a.as_ptr(), b.as_ptr(), len) };
    }
    let mut acc = 0.0f32;
    for d in 0..len {
        acc += a[d] * b[d];
    }
    acc
}

#[inline(always)]
pub(super) fn scale_f32(out: &mut [f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    if out.len() >= 4 {
        unsafe {
            neon_scale(out.as_mut_ptr(), scale, out.len());
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if out.len() >= 8 && std::is_x86_feature_detected!("avx") {
        unsafe {
            avx_scale_f32(out.as_mut_ptr(), scale, out.len());
        }
        return;
    }
    for x in out.iter_mut() {
        *x *= scale;
    }
}

#[inline(always)]
pub(super) fn scaled_add_f32(out: &mut [f32], v: &[f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    if out.len() >= 4 {
        unsafe {
            neon_scaled_add(out.as_mut_ptr(), v.as_ptr(), scale, out.len());
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if out.len() >= 8 && std::is_x86_feature_detected!("avx") {
        unsafe {
            avx_scaled_add_f32(out.as_mut_ptr(), v.as_ptr(), scale, out.len());
        }
        return;
    }
    for d in 0..out.len() {
        out[d] += scale * v[d];
    }
}
