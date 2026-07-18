//! F32/F16/BF16 GEMV kernels.

use rayon::prelude::*;

/// BF16 weight x F32 input.
pub fn gemv_bf16(
    weight_u16: &[u16],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    #[inline]
    fn bf16_row_to_f32(row_u16: &[u16], buf: &mut [f32]) {
        for (dst, &bits) in buf.iter_mut().zip(row_u16.iter()) {
            *dst = f32::from_bits((bits as u32) << 16);
        }
    }

    if seq_len == 1 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            ((rows + n_threads - 1) / n_threads).max(1)
        };
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                let mut row_f32 = vec![0.0f32; cols];
                for i in 0..out.len() {
                    let row = start + i;
                    let row_u16 = &weight_u16[row * cols..(row + 1) * cols];
                    bf16_row_to_f32(row_u16, &mut row_f32);
                    out[i] = dot_f32_row(&row_f32, input, cols);
                }
            });
    } else {
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, out)| {
                let mut row_f32 = vec![0.0f32; cols];
                let row_u16 = &weight_u16[row * cols..(row + 1) * cols];
                bf16_row_to_f32(row_u16, &mut row_f32);
                for s in 0..seq_len {
                    out[s] = dot_f32_row(&row_f32, &input[s * cols..s * cols + cols], cols);
                }
            });
        for row in 0..rows {
            for s in 0..seq_len {
                output[s * rows + row] = row_major[row * seq_len + s];
            }
        }
    }
}

/// F16 weight x F32 input.
pub fn gemv_f16(
    weight_u16: &[u16],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    #[inline]
    fn f16_row_to_f32(row_u16: &[u16], buf: &mut [f32]) {
        for (dst, &bits) in buf.iter_mut().zip(row_u16.iter()) {
            *dst = half::f16::from_bits(bits).to_f32();
        }
    }

    if seq_len == 1 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            ((rows + n_threads - 1) / n_threads).max(1)
        };
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                let mut row_f32 = vec![0.0f32; cols];
                for i in 0..out.len() {
                    let row = start + i;
                    let row_u16 = &weight_u16[row * cols..(row + 1) * cols];
                    f16_row_to_f32(row_u16, &mut row_f32);
                    out[i] = dot_f32_row(&row_f32, input, cols);
                }
            });
    } else {
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, out)| {
                let mut row_f32 = vec![0.0f32; cols];
                let row_u16 = &weight_u16[row * cols..(row + 1) * cols];
                f16_row_to_f32(row_u16, &mut row_f32);
                for s in 0..seq_len {
                    out[s] = dot_f32_row(&row_f32, &input[s * cols..s * cols + cols], cols);
                }
            });
        for row in 0..rows {
            for s in 0..seq_len {
                output[s * rows + row] = row_major[row * seq_len + s];
            }
        }
    }
}

/// F32 weight x F32 input.
pub fn gemv_f32(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    if seq_len == 1 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = if rows <= 64 {
            rows
        } else {
            ((rows + n_threads - 1) / n_threads).max(1)
        };
        output[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(ci, out)| {
                let start = ci * chunk;
                for i in 0..out.len() {
                    let row = start + i;
                    let rb = &weight[row * cols..(row + 1) * cols];
                    out[i] = dot_f32_row(rb, input, cols);
                }
            });
    } else {
        let mut row_major = vec![0.0f32; rows * seq_len];
        row_major
            .par_chunks_mut(seq_len)
            .enumerate()
            .for_each(|(row, out)| {
                let rb = &weight[row * cols..(row + 1) * cols];
                for s in 0..seq_len {
                    out[s] = dot_f32_row(rb, &input[s * cols..s * cols + cols], cols);
                }
            });
        for row in 0..rows {
            for s in 0..seq_len {
                output[s * rows + row] = row_major[row * seq_len + s];
            }
        }
    }
}

#[inline]
pub fn dot_f32_row(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        crate::gemm::neon_dot::dot_f32_neon(a.as_ptr(), b.as_ptr(), n)
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                return dot_f32_avx2_fma(a.as_ptr(), b.as_ptr(), n);
            }
        }
        if std::is_x86_feature_detected!("avx2") {
            unsafe {
                return dot_f32_avx2(a.as_ptr(), b.as_ptr(), n);
            }
        }
        dot_f32_scalar(a, b, n)
    }
    #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
    {
        dot_f32_scalar(a, b, n)
    }
}

#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_fma(a: *const f32, b: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let av0 = _mm256_loadu_ps(a.add(i));
        let bv0 = _mm256_loadu_ps(b.add(i));
        let av1 = _mm256_loadu_ps(a.add(i + 8));
        let bv1 = _mm256_loadu_ps(b.add(i + 8));
        acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
        acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
        i += 16;
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let mut out = horizontal_sum_avx2(acc);
    while i < n {
        out += *a.add(i) * *b.add(i);
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let av0 = _mm256_loadu_ps(a.add(i));
        let bv0 = _mm256_loadu_ps(b.add(i));
        let av1 = _mm256_loadu_ps(a.add(i + 8));
        let bv1 = _mm256_loadu_ps(b.add(i + 8));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av0, bv0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(av1, bv1));
        i += 16;
    }
    let mut out = horizontal_sum_avx2(_mm256_add_ps(acc0, acc1));
    while i < n {
        out += *a.add(i) * *b.add(i);
        i += 1;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let mut sum = _mm_add_ps(lo, hi);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x55));
    _mm_cvtss_f32(sum)
}

#[cfg(test)]
mod tests {
    use super::{gemv_bf16, gemv_f16, gemv_f32};

    #[test]
    fn gemv_f32_writes_seq_major_output() {
        let weight = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5];
        let input = [2.0, 1.0, -1.0, 1.0, 0.0, 2.0];
        let mut output = [0.0; 4];

        gemv_f32(&weight, &input, &mut output, 2, 3, 2);

        assert_eq!(output, [1.0, 6.5, 7.0, 5.0]);
    }

    #[test]
    fn gemv_bf16_matches_f32_for_exact_values() {
        let weight_bf16 = [
            (1.0f32.to_bits() >> 16) as u16,
            (2.0f32.to_bits() >> 16) as u16,
            ((-3.0f32).to_bits() >> 16) as u16,
            (4.0f32.to_bits() >> 16) as u16,
        ];
        let input = [0.5, 2.0];
        let mut output = [0.0; 2];

        gemv_bf16(&weight_bf16, &input, &mut output, 2, 2, 1);

        assert_eq!(output, [4.5, 6.5]);
    }

    #[test]
    fn gemv_f16_matches_f32_for_exact_values() {
        let weight_f16 = [
            half::f16::from_f32(1.0).to_bits(),
            half::f16::from_f32(2.0).to_bits(),
            half::f16::from_f32(-3.0).to_bits(),
            half::f16::from_f32(4.0).to_bits(),
        ];
        let input = [0.5, 2.0];
        let mut output = [0.0; 2];

        gemv_f16(&weight_f16, &input, &mut output, 2, 2, 1);

        assert_eq!(output, [4.5, 6.5]);
    }
}
