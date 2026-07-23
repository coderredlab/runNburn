use rnb_core::error::{Result, RnbError};
use rnb_core::tensor::Tensor;

use super::tensor_as_f32_slice;

/// RMSNorm: output[i] = (x[i] / rms(x)) * w[i % hidden_dim]
///
/// input shape: [..., hidden_dim] — 마지막 차원이 weight 크기와 같아야 함.
/// weight shape: [hidden_dim] (1D)
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let w = tensor_as_f32_slice(weight);
    let hidden_dim = w.len();

    if hidden_dim == 0 {
        return Err(RnbError::InvalidGraph("rms_norm: weight is empty".into()));
    }

    let n = x.len();

    let mut out_data = vec![0.0f32; n];
    rms_norm_into(x, w, eps, &mut out_data);

    Ok(Tensor::from_vec(out_data, input.shape()))
}

pub fn rms_norm_unit_offset(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let w = tensor_as_f32_slice(weight);
    let hidden_dim = w.len();

    if hidden_dim == 0 {
        return Err(RnbError::InvalidGraph(
            "rms_norm_unit_offset: weight is empty".into(),
        ));
    }

    let n = x.len();
    let seq_len = if n % hidden_dim == 0 {
        n / hidden_dim
    } else {
        1
    };

    let mut out_data = vec![0.0f32; n];
    for s in 0..seq_len {
        let x_slice = &x[s * hidden_dim..(s + 1) * hidden_dim];
        let mean_sq: f32 = x_slice.iter().map(|v| v * v).sum::<f32>() / hidden_dim as f32;
        let rms = (mean_sq + eps).sqrt();
        for i in 0..hidden_dim {
            out_data[s * hidden_dim + i] = (x_slice[i] / rms) * (1.0 + w[i]);
        }
    }

    Ok(Tensor::from_vec(out_data, input.shape()))
}

/// LayerNorm: output[i] = ((x[i] - mean) / std) * w[i] + b[i]
pub fn layer_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let w = tensor_as_f32_slice(weight);
    let hidden_dim = w.len();
    if hidden_dim == 0 {
        return Err(RnbError::InvalidGraph("layer_norm: weight is empty".into()));
    }

    let n = x.len();
    let seq_len = if n % hidden_dim == 0 {
        n / hidden_dim
    } else {
        1
    };
    let bias_data = bias.map(tensor_as_f32_slice);

    let mut out_data = vec![0.0f32; n];
    for s in 0..seq_len {
        let start = s * hidden_dim;
        let end = start + hidden_dim;
        let x_slice = &x[start..end];
        let mean: f32 = x_slice.iter().sum::<f32>() / hidden_dim as f32;
        let var: f32 = x_slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
        let std = (var + eps).sqrt();

        for i in 0..hidden_dim {
            out_data[start + i] = ((x_slice[i] - mean) / std) * w[i];
            if let Some(bias_slice) = bias_data {
                out_data[start + i] += bias_slice[i];
            }
        }
    }

    Ok(Tensor::from_vec(out_data, input.shape()))
}

pub fn layer_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    for s in 0..seq_len {
        let start = s * hidden_dim;
        let end = start + hidden_dim;
        let x_slice = &input[start..end];
        let mean: f32 = x_slice.iter().sum::<f32>() / hidden_dim as f32;
        let var: f32 = x_slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
        let std = (var + eps).sqrt();
        for i in 0..hidden_dim {
            output[start + i] = ((x_slice[i] - mean) / std) * weight[i];
        }
    }
}

/// L2 normalization: y = x / sqrt(sum(x^2) + eps)
/// Operates on the last dimension (each row independently for 2D+ input).
pub fn l2_norm(input: &Tensor, eps: f32) -> Result<Tensor> {
    let data = tensor_as_f32_slice(input);
    let shape = input.shape();
    let dim = *shape.last().unwrap();
    let n_rows = data.len() / dim;
    let mut out = vec![0.0f32; data.len()];

    for row in 0..n_rows {
        let start = row * dim;
        let row_data = &data[start..start + dim];
        let norm_sq: f32 = row_data.iter().map(|x| x * x).sum();
        let inv_norm = 1.0 / (norm_sq + eps).sqrt();
        for i in 0..dim {
            out[start + i] = row_data[i] * inv_norm;
        }
    }

    Ok(Tensor::from_vec(out, shape))
}

/// RMSNorm slice version: writes to pre-allocated output (zero-alloc).
/// input/output: [seq_len * hidden_dim], weight: [hidden_dim]
pub fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    // Default ON: GGML uses f64 mean-square accumulator. `RNB_RMS_F64=0` opts
    // back into the legacy f32-only path.
    if std::env::var("RNB_RMS_F64")
        .map(|v| v != "0")
        .unwrap_or(true)
    {
        rms_norm_into_f64(input, weight, eps, output);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                rms_norm_into_avx2_fma(input, weight, eps, output);
            }
            return;
        }
    }
    rms_norm_into_scalar(input, weight, eps, output);
}

fn rms_norm_into_scalar(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    for s in 0..seq_len {
        let x_slice = &input[s * hidden_dim..(s + 1) * hidden_dim];
        let mean_sq: f32 = x_slice.iter().map(|v| v * v).sum::<f32>() / hidden_dim as f32;
        let rms = (mean_sq + eps).sqrt();
        let out_slice = &mut output[s * hidden_dim..(s + 1) * hidden_dim];
        for i in 0..hidden_dim {
            out_slice[i] = (x_slice[i] / rms) * weight[i];
        }
    }
}

/// RMSNorm with f64 mean-square accumulator. Opt-in via `RNB_RMS_F64=1`.
/// hidden_dim is typically 1024-4096 so the f32 sum-of-squares accumulates
/// √hidden_dim × ULP error; promoting just the reduce to f64 cuts ULP from
/// ~1e-7 to ~1e-16 without changing the final f32 division/output.
fn rms_norm_into_f64(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    for s in 0..seq_len {
        let x_slice = &input[s * hidden_dim..(s + 1) * hidden_dim];
        let mean_sq_f64: f64 = x_slice
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            / hidden_dim as f64;
        let rms = ((mean_sq_f64 + eps as f64).sqrt()) as f32;
        let out_slice = &mut output[s * hidden_dim..(s + 1) * hidden_dim];
        for i in 0..hidden_dim {
            out_slice[i] = (x_slice[i] / rms) * weight[i];
        }
    }
}

pub fn rms_norm_unit_offset_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                rms_norm_unit_offset_into_avx2_fma(input, weight, eps, output);
            }
            return;
        }
    }
    rms_norm_unit_offset_into_scalar(input, weight, eps, output);
}

fn rms_norm_unit_offset_into_scalar(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    for s in 0..seq_len {
        let x_slice = &input[s * hidden_dim..(s + 1) * hidden_dim];
        let mean_sq: f32 = x_slice.iter().map(|v| v * v).sum::<f32>() / hidden_dim as f32;
        let rms = (mean_sq + eps).sqrt();
        let out_slice = &mut output[s * hidden_dim..(s + 1) * hidden_dim];
        for i in 0..hidden_dim {
            out_slice[i] = (x_slice[i] / rms) * (1.0 + weight[i]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_into_avx2_fma(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    rms_norm_rows_avx2_fma(input, weight, eps, output, false);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_unit_offset_into_avx2_fma(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
) {
    rms_norm_rows_avx2_fma(input, weight, eps, output, true);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_rows_avx2_fma(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    output: &mut [f32],
    unit_offset: bool,
) {
    use std::arch::x86_64::*;

    let hidden_dim = weight.len();
    let seq_len = input.len() / hidden_dim;
    for s in 0..seq_len {
        let row = input.as_ptr().add(s * hidden_dim);
        let out = output.as_mut_ptr().add(s * hidden_dim);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= hidden_dim {
            let x0 = _mm256_loadu_ps(row.add(i));
            let x1 = _mm256_loadu_ps(row.add(i + 8));
            acc0 = _mm256_fmadd_ps(x0, x0, acc0);
            acc1 = _mm256_fmadd_ps(x1, x1, acc1);
            i += 16;
        }
        let mut sum = horizontal_sum_avx2(_mm256_add_ps(acc0, acc1));
        while i < hidden_dim {
            let x = *row.add(i);
            sum += x * x;
            i += 1;
        }

        let inv_rms = ((sum / hidden_dim as f32) + eps).sqrt().recip();
        let inv = _mm256_set1_ps(inv_rms);
        let one = _mm256_set1_ps(1.0);
        i = 0;
        while i + 16 <= hidden_dim {
            let x0 = _mm256_mul_ps(_mm256_loadu_ps(row.add(i)), inv);
            let x1 = _mm256_mul_ps(_mm256_loadu_ps(row.add(i + 8)), inv);
            let mut w0 = _mm256_loadu_ps(weight.as_ptr().add(i));
            let mut w1 = _mm256_loadu_ps(weight.as_ptr().add(i + 8));
            if unit_offset {
                w0 = _mm256_add_ps(w0, one);
                w1 = _mm256_add_ps(w1, one);
            }
            _mm256_storeu_ps(out.add(i), _mm256_mul_ps(x0, w0));
            _mm256_storeu_ps(out.add(i + 8), _mm256_mul_ps(x1, w1));
            i += 16;
        }
        while i < hidden_dim {
            let w = if unit_offset {
                1.0 + *weight.get_unchecked(i)
            } else {
                *weight.get_unchecked(i)
            };
            *out.add(i) = *row.add(i) * inv_rms * w;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

/// L2 normalize slice version: each row (of `dim` elements) normalized independently.
/// input/output: [n_rows * dim]
pub fn l2_norm_into(input: &[f32], eps: f32, output: &mut [f32], dim: usize) {
    let n_rows = input.len() / dim;
    for row in 0..n_rows {
        let start = row * dim;
        let row_data = &input[start..start + dim];
        let norm_sq: f32 = row_data.iter().map(|x| x * x).sum();
        let inv_norm = 1.0 / (norm_sq + eps).sqrt();
        for i in 0..dim {
            output[start + i] = row_data[i] * inv_norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tensor_to_f32_vec;

    #[test]
    fn test_rms_norm() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let weight = Tensor::from_slice(&[1.0f32; 4], &[4]);
        let output = rms_norm(&input, &weight, 1e-5).unwrap();
        let data = tensor_to_f32_vec(&output);
        let rms = (7.5f32 + 1e-5).sqrt();
        assert!((data[0] - 1.0 / rms).abs() < 1e-4);
        assert!((data[1] - 2.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_into_matches_tensor() {
        let input_data = [1.0f32, 2.0, 3.0, 4.0];
        let weight_data = [1.0f32, 0.5, 2.0, 0.25];
        let input_t = Tensor::from_slice(&input_data, &[1, 4]);
        let weight_t = Tensor::from_slice(&weight_data, &[4]);
        let expected = tensor_to_f32_vec(&rms_norm(&input_t, &weight_t, 1e-5).unwrap());
        let mut output = vec![0.0f32; 4];
        rms_norm_into(&input_data, &weight_data, 1e-5, &mut output);
        for (a, b) in output.iter().zip(expected.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "rms_norm_into mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rms_norm_unit_offset_matches_tensor() {
        let input_data = [1.0f32, 2.0, 3.0, 4.0];
        let weight_data = [0.0f32, -0.5, 1.0, -0.75];
        let input_t = Tensor::from_slice(&input_data, &[1, 4]);
        let weight_t = Tensor::from_slice(&weight_data, &[4]);
        let expected = tensor_to_f32_vec(&rms_norm_unit_offset(&input_t, &weight_t, 1e-5).unwrap());
        let mut output = vec![0.0f32; 4];
        rms_norm_unit_offset_into(&input_data, &weight_data, 1e-5, &mut output);
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "rms_norm_unit_offset mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_rms_norm_avx2_fma_matches_scalar_rows() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let hidden_dim = 1536;
        let rows = 3;
        let input: Vec<f32> = (0..hidden_dim * rows)
            .map(|i| ((i * 37 % 251) as f32 - 125.0) / 31.0)
            .collect();
        let weight: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i * 19 % 127) as f32 - 63.0) / 97.0)
            .collect();
        let mut expected = vec![0.0f32; input.len()];
        let mut actual = vec![0.0f32; input.len()];
        rms_norm_into_scalar(&input, &weight, 1e-5, &mut expected);
        unsafe {
            rms_norm_into_avx2_fma(&input, &weight, 1e-5, &mut actual);
        }
        for (idx, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 2e-5,
                "rms norm AVX2 mismatch at {idx}: {a} vs {b}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_rms_norm_unit_offset_avx2_fma_matches_scalar_rows() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        let hidden_dim = 1536;
        let rows = 2;
        let input: Vec<f32> = (0..hidden_dim * rows)
            .map(|i| ((i * 29 % 257) as f32 - 128.0) / 41.0)
            .collect();
        let weight: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i * 17 % 131) as f32 - 65.0) / 89.0)
            .collect();
        let mut expected = vec![0.0f32; input.len()];
        let mut actual = vec![0.0f32; input.len()];
        rms_norm_unit_offset_into_scalar(&input, &weight, 1e-5, &mut expected);
        unsafe {
            rms_norm_unit_offset_into_avx2_fma(&input, &weight, 1e-5, &mut actual);
        }
        for (idx, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 2e-5,
                "unit-offset rms norm AVX2 mismatch at {idx}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_l2_norm_into_matches_tensor() {
        let input_data = [3.0f32, 4.0, 1.0, 0.0];
        let input_t = Tensor::from_slice(&input_data, &[2, 2]);
        let expected = tensor_to_f32_vec(&l2_norm(&input_t, 1e-5).unwrap());
        let mut output = vec![0.0f32; 4];
        l2_norm_into(&input_data, 1e-5, &mut output, 2);
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "l2_norm_into mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_layer_norm() {
        let input = Tensor::from_slice(&[1.0f32, 3.0], &[1, 2]);
        let weight = Tensor::from_slice(&[1.0f32; 2], &[2]);
        let output = layer_norm(&input, &weight, None, 1e-5).unwrap();
        let data = tensor_to_f32_vec(&output);
        // mean=2, var=1, std≈1
        assert!((data[0] - (-1.0)).abs() < 1e-3);
        assert!((data[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_applies_per_row() {
        let input = Tensor::from_slice(&[1.0f32, 3.0, 10.0, 14.0], &[2, 2]);
        let weight = Tensor::from_slice(&[1.0f32; 2], &[2]);
        let output = layer_norm(&input, &weight, None, 1e-5).unwrap();
        let data = tensor_to_f32_vec(&output);

        assert!((data[0] - (-1.0)).abs() < 1e-3);
        assert!((data[1] - 1.0).abs() < 1e-3);
        assert!((data[2] - (-1.0)).abs() < 1e-3);
        assert!((data[3] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_layer_norm_into_matches_tensor_per_row() {
        let input_data = [1.0f32, 3.0, 10.0, 14.0];
        let weight_data = [1.0f32, 0.5];
        let input_t = Tensor::from_slice(&input_data, &[2, 2]);
        let weight_t = Tensor::from_slice(&weight_data, &[2]);
        let expected = tensor_to_f32_vec(&layer_norm(&input_t, &weight_t, None, 1e-5).unwrap());
        let mut output = vec![0.0f32; 4];
        layer_norm_into(&input_data, &weight_data, 1e-5, &mut output);
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "layer_norm_into mismatch: {} vs {}",
                a,
                b
            );
        }
    }
}
