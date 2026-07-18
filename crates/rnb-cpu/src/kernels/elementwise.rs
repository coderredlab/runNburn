use rnb_core::error::Result;
use rnb_core::tensor::{DType, Tensor};

use super::tensor_as_f32_slice;

/// 두 텐서를 더함. 같은 shape이면 element-wise, 다르면 b를 브로드캐스팅.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_data = tensor_as_f32_slice(a);
    let b_data = tensor_as_f32_slice(b);

    // 같은 shape이면 바로 zip으로 처리
    if a.shape() == b.shape() {
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| x + y)
            .collect();
        return Ok(Tensor::from_vec(out, a.shape()));
    }

    // 브로드캐스팅: b가 더 작은 경우 (bias 덧셈 등)
    let out_len = a_data.len();
    let b_len = b_data.len();
    let mut out = vec![0.0f32; out_len];
    for i in 0..out_len {
        out[i] = a_data[i] + b_data[i % b_len];
    }
    Ok(Tensor::from_vec(out, a.shape()))
}

/// 두 텐서를 곱함. 같은 shape이면 element-wise, 다르면 b를 브로드캐스팅.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_data = tensor_as_f32_slice(a);
    let b_data = tensor_as_f32_slice(b);

    if a.shape() == b.shape() {
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| x * y)
            .collect();
        return Ok(Tensor::from_vec(out, a.shape()));
    }

    let out_len = a_data.len();
    let b_len = b_data.len();
    let mut out = vec![0.0f32; out_len];
    for i in 0..out_len {
        out[i] = a_data[i] * b_data[i % b_len];
    }
    Ok(Tensor::from_vec(out, a.shape()))
}

/// dtype 변환. 현재는 zeros로 target dtype 텐서를 반환 (F32 in/out만 지원).
pub fn cast(input: &Tensor, target_dtype: DType) -> Result<Tensor> {
    Ok(Tensor::zeros(input.shape(), target_dtype))
}

/// a += b in-place. Supports broadcast (b shorter than a).
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    if a.len() == b.len() {
        for i in 0..a.len() {
            a[i] += b[i];
        }
    } else {
        let b_len = b.len();
        for i in 0..a.len() {
            a[i] += b[i % b_len];
        }
    }
}

/// a *= b in-place. Supports broadcast.
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    if a.len() == b.len() {
        for i in 0..a.len() {
            a[i] *= b[i];
        }
    } else {
        let b_len = b.len();
        for i in 0..a.len() {
            a[i] *= b[i % b_len];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tensor_to_f32_vec;

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[1, 3]);
        let c = add(&a, &b).unwrap();
        let data = tensor_to_f32_vec(&c);
        assert!((data[0] - 5.0).abs() < 1e-5);
        assert!((data[1] - 7.0).abs() < 1e-5);
        assert!((data[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::from_slice(&[2.0f32, 3.0], &[1, 2]);
        let b = Tensor::from_slice(&[4.0f32, 5.0], &[1, 2]);
        let c = mul(&a, &b).unwrap();
        let data = tensor_to_f32_vec(&c);
        assert!((data[0] - 8.0).abs() < 1e-5);
        assert!((data[1] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0], &[1, 3]);
        let c = add(&a, &b).unwrap();
        let data = tensor_to_f32_vec(&c);
        assert!((data[0] - 11.0).abs() < 1e-5);
        assert!((data[3] - 14.0).abs() < 1e-5);
    }
}
