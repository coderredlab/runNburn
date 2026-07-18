use rnb_core::error::Result;
use rnb_core::tensor::Tensor;

use super::tensor_as_f32_slice;

/// RoPE(Rotary Position Embedding)을 입력 텐서에 적용.
///
/// input: flat [seq_len * dim] 텐서 (dim = num_heads * head_dim 또는 num_kv_heads * head_dim)
/// pos_start: 시퀀스 첫 토큰의 position (이후 토큰은 pos_start+1, pos_start+2, ...)
/// head_dim: 각 head의 차원
/// theta: RoPE base frequency (llama 기본값 10000.0)
///
/// 각 토큰의 모든 head에 대해 head_dim 쌍(pair) 단위 회전을 적용한다.
///
/// mc73: GGML-aligned accumulator pattern (`theta_scale = base^(-2/head_dim)`
/// precompute + `angle *= theta_scale` per pair + `cos`/`sin` 분리 호출).
pub fn rope(input: &Tensor, pos_start: usize, head_dim: usize, theta: f32) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let total = x.len();
    let mut out = x.to_vec();

    if total == 0 || head_dim == 0 {
        return Ok(Tensor::from_vec(out, input.shape()));
    }

    // 입력 shape에서 seq_len과 dim(per-token 크기) 파악
    let shape = input.shape();
    let (seq_len, dim) = if shape.len() >= 2 {
        (shape[0], shape[1..].iter().product::<usize>())
    } else {
        // 1D flat: dim = total (seq_len=1)
        (1, total)
    };

    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);

    // 각 토큰별로 다른 position 적용
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_start = t * dim;
        let token_slice = &mut out[token_start..token_start + dim];

        // dim 안의 모든 head에 대해 회전 적용
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut angle = pos as f32;
            let mut i = 0usize;
            while i < head_dim {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[i + 1];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[i + 1] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
                i += 2;
            }
        }
    }

    Ok(Tensor::from_vec(out, input.shape()))
}

/// mc73: GGML-aligned accumulator pattern. `theta_base` accumulates as
/// `pos * theta_scale^pair_idx` so the per-pair `theta = theta_base / factor`
/// matches GGML's `rope_yarn` with freq_factors. `cos`/`sin` separated.
pub fn rope_with_factors(
    input: &Tensor,
    pos_start: usize,
    head_dim: usize,
    theta: f32,
    freq_factors: &[f32],
) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let total = x.len();
    let mut out = x.to_vec();

    if total == 0 || head_dim == 0 {
        return Ok(Tensor::from_vec(out, input.shape()));
    }

    let shape = input.shape();
    let (seq_len, dim) = if shape.len() >= 2 {
        (shape[0], shape[1..].iter().product::<usize>())
    } else {
        (1, total)
    };

    let pair_count = head_dim / 2;
    assert!(
        freq_factors.len() >= pair_count,
        "freq_factors len {} < pair_count {}",
        freq_factors.len(),
        pair_count
    );

    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);

    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_start = t * dim;
        let token_slice = &mut out[token_start..token_start + dim];

        for chunk in token_slice.chunks_mut(head_dim) {
            let mut theta_base = pos as f32;
            for pair_idx in 0..pair_count {
                let i = pair_idx * 2;
                let factor = freq_factors[pair_idx];
                let angle = theta_base / factor;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[i + 1];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[i + 1] = x0 * sin_a + x1 * cos_a;
                theta_base *= theta_scale;
            }
        }
    }

    Ok(Tensor::from_vec(out, input.shape()))
}

/// Partial RoPE: head_dim 중 n_rot 차원만 회전, 나머지는 passthrough.
/// MRoPE(Qwen3.5 등)에서 rope.dimension_count < head_dim인 경우 사용.
///
/// mc72: GGML-aligned accumulator pattern (see `rope_partial_inplace`).
pub fn rope_partial(
    input: &Tensor,
    pos_start: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let total = x.len();
    let mut out = x.to_vec();

    if total == 0 || head_dim == 0 || n_rot == 0 {
        return Ok(Tensor::from_vec(out, input.shape()));
    }

    let shape = input.shape();
    let (seq_len, dim) = if shape.len() >= 2 {
        (shape[0], shape[1..].iter().product::<usize>())
    } else {
        (1, total)
    };

    let n_rot = n_rot.min(head_dim);
    let theta_scale: f32 = theta.powf(-2.0_f32 / n_rot as f32);

    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_start = t * dim;
        let token_slice = &mut out[token_start..token_start + dim];

        for chunk in token_slice.chunks_mut(head_dim) {
            // Only rotate first n_rot dimensions (n_rot/2 pairs)
            let mut angle = pos as f32;
            let mut i = 0usize;
            while i < n_rot {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[i + 1];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[i + 1] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
                i += 2;
            }
            // Remaining dims (n_rot..head_dim) pass through unchanged
        }
    }

    Ok(Tensor::from_vec(out, input.shape()))
}

/// mc73: GGML-aligned accumulator pattern. NeoX-style interleave (chunk[i],
/// chunk[half+i]) with `theta_scale = base^(-2/head_dim)` accumulator.
pub fn rope_partial_proportional(
    input: &Tensor,
    pos_start: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let total = x.len();
    let mut out = x.to_vec();

    if total == 0 || head_dim == 0 || n_rot == 0 {
        return Ok(Tensor::from_vec(out, input.shape()));
    }

    let shape = input.shape();
    let (seq_len, dim) = if shape.len() >= 2 {
        (shape[0], shape[1..].iter().product::<usize>())
    } else {
        (1, total)
    };

    let n_rot = n_rot.min(head_dim);

    let half = head_dim / 2;
    let rope_pairs = (n_rot / 2).min(half);
    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);

    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_start = t * dim;
        let token_slice = &mut out[token_start..token_start + dim];

        for chunk in token_slice.chunks_mut(head_dim) {
            let mut angle = pos as f32;
            for i in 0..rope_pairs {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[half + i];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[half + i] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
            }
        }
    }

    Ok(Tensor::from_vec(out, input.shape()))
}

/// Text M-RoPE as used by llama.cpp's `ggml_rope_multi` for Qwen text tokens.
///
/// Text batches broadcast the same position to every M-RoPE section, so the
/// section boundaries do not change the angle values. The important difference
/// from regular partial RoPE is the NeoX-style pairing inside the rotated
/// prefix: `(0, n_rot/2), (1, n_rot/2 + 1), ...`.
pub fn rope_mrope_text(
    input: &Tensor,
    pos_start: usize,
    head_dim: usize,
    n_rot: usize,
    theta: f32,
) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let mut out = x.to_vec();

    if out.is_empty() || head_dim == 0 || n_rot == 0 {
        return Ok(Tensor::from_vec(out, input.shape()));
    }

    let shape = input.shape();
    let (seq_len, dim) = if shape.len() >= 2 {
        (shape[0], shape[1..].iter().product::<usize>())
    } else {
        (1, out.len())
    };

    rope_mrope_text_inplace(&mut out, pos_start, head_dim, dim, n_rot, theta);
    debug_assert_eq!(seq_len * dim, out.len());
    Ok(Tensor::from_vec(out, input.shape()))
}

/// RoPE in-place: modifies data directly (zero-alloc).
/// data: [seq_len * dim] flat array, dim = num_heads * head_dim
///
/// mc73: GGML-aligned accumulator pattern (`theta_scale` precompute + `angle`
/// 누적 + `cos`/`sin` 분리 호출).
pub fn rope_inplace(data: &mut [f32], pos_start: usize, head_dim: usize, dim: usize, theta: f32) {
    let total = data.len();
    if total == 0 || head_dim == 0 {
        return;
    }
    let seq_len = total / dim;
    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_slice = &mut data[t * dim..(t + 1) * dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut angle = pos as f32;
            let mut i = 0usize;
            while i < head_dim {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[i + 1];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[i + 1] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
                i += 2;
            }
        }
    }
}

/// In-place text M-RoPE. See [`rope_mrope_text`].
pub fn rope_mrope_text_inplace(
    data: &mut [f32],
    pos_start: usize,
    head_dim: usize,
    dim: usize,
    n_rot: usize,
    theta: f32,
) {
    let total = data.len();
    if total == 0 || head_dim == 0 || n_rot == 0 {
        return;
    }

    let seq_len = total / dim;
    let n_rot = n_rot.min(head_dim);
    assert!(
        n_rot % 2 == 0,
        "M-RoPE rotated dimension must be even, got {n_rot}"
    );
    let half = n_rot / 2;
    let theta_scale: f32 = theta.powf(-2.0_f32 / n_rot as f32);

    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_slice = &mut data[t * dim..(t + 1) * dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut angle = pos as f32;
            for i in 0..half {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[half + i];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[half + i] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
            }
        }
    }
}

/// mc73: GGML-aligned accumulator pattern with freq_factors (`theta_base *=
/// theta_scale` per pair, `theta = theta_base / factor`, separated `cos`/`sin`).
pub fn rope_inplace_with_factors(
    data: &mut [f32],
    pos_start: usize,
    head_dim: usize,
    dim: usize,
    theta: f32,
    freq_factors: &[f32],
) {
    let total = data.len();
    if total == 0 || head_dim == 0 {
        return;
    }
    let seq_len = total / dim;
    let pair_count = head_dim / 2;
    assert!(
        freq_factors.len() >= pair_count,
        "freq_factors len {} < pair_count {}",
        freq_factors.len(),
        pair_count
    );
    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_slice = &mut data[t * dim..(t + 1) * dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut theta_base = pos as f32;
            for pair_idx in 0..pair_count {
                let i = pair_idx * 2;
                let factor = freq_factors[pair_idx];
                let angle = theta_base / factor;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[i + 1];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[i + 1] = x0 * sin_a + x1 * cos_a;
                theta_base *= theta_scale;
            }
        }
    }
}

/// Partial RoPE in-place: only rotates first n_rot dimensions.
///
/// mc72: GGML-aligned accumulator pattern. `theta_scale = base^(-2/n_rot)`
/// computed once, then `theta *= theta_scale` per pair. `cosf`/`sinf` called
/// separately (cos first, then sin) to match `rope_yarn` in
/// ggml-cpu/ops.cpp. Mathematically equivalent to per-iter `powf`, but
/// floating-point accumulation order matches GGML so the ULP-level fp drift
/// stays in sync with llama.cpp.
pub fn rope_partial_inplace(
    data: &mut [f32],
    pos_start: usize,
    head_dim: usize,
    dim: usize,
    n_rot: usize,
    theta: f32,
) {
    let total = data.len();
    if total == 0 || head_dim == 0 || n_rot == 0 {
        return;
    }
    let seq_len = total / dim;
    let n_rot = n_rot.min(head_dim);
    let theta_scale: f32 = theta.powf(-2.0_f32 / n_rot as f32);
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_slice = &mut data[t * dim..(t + 1) * dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut angle = pos as f32;
            let mut i = 0usize;
            while i < n_rot {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[i + 1];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[i + 1] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
                i += 2;
            }
        }
    }
}

/// mc73: GGML-aligned accumulator pattern. NeoX-style interleave + theta_scale
/// 누적 + cos/sin 분리 호출.
pub fn rope_partial_proportional_inplace(
    data: &mut [f32],
    pos_start: usize,
    head_dim: usize,
    dim: usize,
    n_rot: usize,
    theta: f32,
) {
    let total = data.len();
    if total == 0 || head_dim == 0 || n_rot == 0 {
        return;
    }
    let seq_len = total / dim;
    let n_rot = n_rot.min(head_dim);
    let half = head_dim / 2;
    let rope_pairs = (n_rot / 2).min(half);
    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_slice = &mut data[t * dim..(t + 1) * dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut angle = pos as f32;
            for i in 0..rope_pairs {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[half + i];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[half + i] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
            }
        }
    }
}

/// NEOX-style RoPE: rotates pairs (x_i, x_{i + head_dim/2}) instead of
/// consecutive pairs. This is the rotation convention used by Gemma,
/// GPT-NeoX, and all llama.cpp GEMMA* architectures.
///
/// mc73: GGML-aligned accumulator pattern (theta_scale precompute + angle 누적
/// + cos/sin 분리 호출).
pub fn rope_neox(input: &Tensor, pos_start: usize, head_dim: usize, theta: f32) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let total = x.len();
    let mut out = x.to_vec();

    if total == 0 || head_dim == 0 {
        return Ok(Tensor::from_vec(out, input.shape()));
    }

    let shape = input.shape();
    let (seq_len, dim) = if shape.len() >= 2 {
        (shape[0], shape[1..].iter().product::<usize>())
    } else {
        (1, total)
    };

    let half = head_dim / 2;
    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_start = t * dim;
        let token_slice = &mut out[token_start..token_start + dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut angle = pos as f32;
            for i in 0..half {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[half + i];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[half + i] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
            }
        }
    }

    Ok(Tensor::from_vec(out, input.shape()))
}

/// mc73: GGML-aligned accumulator pattern with NeoX + freq_factors.
/// `theta_base *= theta_scale` per pair, `theta = theta_base / factor`,
/// separated `cos`/`sin`. Gemma4 path.
pub fn rope_neox_with_factors(
    input: &Tensor,
    pos_start: usize,
    head_dim: usize,
    theta: f32,
    freq_factors: &[f32],
) -> Result<Tensor> {
    let x = tensor_as_f32_slice(input);
    let total = x.len();
    let mut out = x.to_vec();

    if total == 0 || head_dim == 0 {
        return Ok(Tensor::from_vec(out, input.shape()));
    }

    let shape = input.shape();
    let (seq_len, dim) = if shape.len() >= 2 {
        (shape[0], shape[1..].iter().product::<usize>())
    } else {
        (1, total)
    };

    let half = head_dim / 2;
    assert!(
        freq_factors.len() >= half,
        "freq_factors len {} < half {}",
        freq_factors.len(),
        half
    );
    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_start = t * dim;
        let token_slice = &mut out[token_start..token_start + dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut theta_base = pos as f32;
            for i in 0..half {
                let factor = freq_factors[i];
                let angle = theta_base / factor;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[half + i];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[half + i] = x0 * sin_a + x1 * cos_a;
                theta_base *= theta_scale;
            }
        }
    }

    Ok(Tensor::from_vec(out, input.shape()))
}

/// mc73: GGML-aligned accumulator pattern (NeoX in-place).
pub fn rope_neox_inplace(
    data: &mut [f32],
    pos_start: usize,
    head_dim: usize,
    dim: usize,
    theta: f32,
) {
    let total = data.len();
    if total == 0 || head_dim == 0 {
        return;
    }
    let seq_len = total / dim;
    let half = head_dim / 2;
    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_slice = &mut data[t * dim..(t + 1) * dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut angle = pos as f32;
            for i in 0..half {
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[half + i];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[half + i] = x0 * sin_a + x1 * cos_a;
                angle *= theta_scale;
            }
        }
    }
}

/// mc73: GGML-aligned accumulator pattern (NeoX in-place + freq_factors).
/// Gemma4 path. `theta_base *= theta_scale` per pair, `theta = theta_base /
/// factor`, separated `cos`/`sin`.
pub fn rope_neox_inplace_with_factors(
    data: &mut [f32],
    pos_start: usize,
    head_dim: usize,
    dim: usize,
    theta: f32,
    freq_factors: &[f32],
) {
    let total = data.len();
    if total == 0 || head_dim == 0 {
        return;
    }
    let seq_len = total / dim;
    let half = head_dim / 2;
    assert!(
        freq_factors.len() >= half,
        "freq_factors len {} < half {}",
        freq_factors.len(),
        half
    );
    let theta_scale: f32 = theta.powf(-2.0_f32 / head_dim as f32);
    for t in 0..seq_len {
        let pos = pos_start + t;
        let token_slice = &mut data[t * dim..(t + 1) * dim];
        for chunk in token_slice.chunks_mut(head_dim) {
            let mut theta_base = pos as f32;
            for i in 0..half {
                let factor = freq_factors[i];
                let angle = theta_base / factor;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let x0 = chunk[i];
                let x1 = chunk[half + i];
                chunk[i] = x0 * cos_a - x1 * sin_a;
                chunk[half + i] = x0 * sin_a + x1 * cos_a;
                theta_base *= theta_scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tensor_to_f32_vec;

    #[test]
    fn test_rope_position_zero() {
        // position=0이면 모든 angle=0, cos=1, sin=0 → 변화 없어야 함
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let output = rope(&input, 0, 4, 10000.0).unwrap();
        let data = tensor_to_f32_vec(&output);
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_applies_rotation() {
        let input = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0], &[1, 4]);
        let output = rope(&input, 1, 4, 10000.0).unwrap();
        let data = tensor_to_f32_vec(&output);
        // position=1이면 회전 발생 — 값이 바껴야 함
        assert!((data[0] - 1.0).abs() > 1e-3 || (data[1] - 0.0).abs() > 1e-3);
    }

    #[test]
    fn test_rope_inplace_matches_tensor() {
        let input_data = vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let head_dim = 4;
        let dim = 8;
        let input_t = Tensor::from_slice(&input_data, &[1, dim]);
        let expected = tensor_to_f32_vec(&rope(&input_t, 5, head_dim, 10000.0).unwrap());
        let mut data = input_data.clone();
        rope_inplace(&mut data, 5, head_dim, dim, 10000.0);
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "rope_inplace mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_rope_per_token_position() {
        // seq_len=2, head_dim=4: 토큰 0은 pos=0 (변화 없음), 토큰 1은 pos=1 (회전)
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], &[2, 4]);
        let output = rope(&input, 0, 4, 10000.0).unwrap();
        let data = tensor_to_f32_vec(&output);
        // 토큰 0 (pos=0): 변화 없어야 함
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        // 토큰 1 (pos=1): 회전 발생
        assert!((data[4] - 1.0).abs() > 1e-3 || (data[5] - 2.0).abs() > 1e-3);
    }

    #[test]
    fn test_rope_with_freq_factors_can_freeze_upper_pairs() {
        let input = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 4]);
        let factors = [1.0f32, 1.0e30];
        let output = rope_with_factors(&input, 1, 4, 10000.0, &factors).unwrap();
        let data = tensor_to_f32_vec(&output);

        assert!((data[0] - 1.0f32.cos()).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 1.0f32.sin()).abs() < 1e-5, "got {}", data[1]);
        assert!(data[2].abs() < 1e-6, "got {}", data[2]);
        assert!((data[3] - 1.0).abs() < 1e-6, "got {}", data[3]);
    }

    #[test]
    fn test_mrope_text_uses_partial_neox_pairs() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 8]);
        let output = rope_mrope_text(&input, 1, 8, 4, 10000.0).unwrap();
        let data = tensor_to_f32_vec(&output);

        let c0 = 1.0f32.cos();
        let s0 = 1.0f32.sin();
        let a1 = 0.01f32;
        let c1 = a1.cos();
        let s1 = a1.sin();

        assert!((data[0] - (1.0 * c0 - 3.0 * s0)).abs() < 1e-5);
        assert!((data[2] - (1.0 * s0 + 3.0 * c0)).abs() < 1e-5);
        assert!((data[1] - (2.0 * c1 - 4.0 * s1)).abs() < 1e-5);
        assert!((data[3] - (2.0 * s1 + 4.0 * c1)).abs() < 1e-5);
        assert_eq!(&data[4..], &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_mrope_text_inplace_matches_tensor() {
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::from_slice(&input_data, &[1, 8]);
        let expected = tensor_to_f32_vec(&rope_mrope_text(&input, 3, 8, 4, 10000.0).unwrap());
        let mut data = input_data;

        rope_mrope_text_inplace(&mut data, 3, 8, 8, 4, 10000.0);

        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "mrope mismatch: {} vs {}", a, b);
        }
    }
}
