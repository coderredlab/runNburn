use rnb_core::error::{Result, RnbError};
use rnb_core::tensor::{DType, Tensor};

use crate::quantize;

/// Tensor를 F32 Vec으로 변환한다. dtype에 따라 dequantize 처리.
/// F32 → 직접 변환
/// F16 → f16 → f32
/// U8 (양자화) → 블록 단위 dequantize
/// I8 (Q8_0) → 블록 단위 dequantize
fn tensor_to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
    let bytes = t.as_bytes().ok_or_else(|| {
        RnbError::InvalidGraph("matmul: 텐서에서 bytes 추출 실패 (비연속 또는 device 텐서)".into())
    })?;

    match t.dtype() {
        DType::F32 => {
            let out: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(out)
        }
        DType::F16 => {
            // f16 → f32 변환 (little-endian raw bytes)
            let out: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(out)
        }
        DType::U8 => {
            // 양자화 타입 (Q4_0, Q4_K, etc.)
            // map_tensors에서 실제 바이트 크기로 shape을 [byte_count]로 저장했으므로
            // bytes.len()이 실제 양자화 데이터 크기임.
            // float numel은 블록 구조에서 역산.
            let byte_count = bytes.len();
            dequantize_u8_from_bytes(bytes, byte_count)
        }
        DType::I8 => {
            // Q8_0: f16 scale(2) + 32 × i8 = 34bytes/block
            let byte_count = bytes.len();
            dequantize_i8_from_bytes(bytes, byte_count)
        }
        other => Err(RnbError::InvalidGraph(format!(
            "matmul: 지원하지 않는 weight dtype: {:?}",
            other
        ))),
    }
}

/// U8 양자화 데이터를 byte_count로부터 포맷을 추론해 dequantize.
/// map_tensors에서 [byte_count] shape으로 저장된 텐서를 처리.
///
/// 지원 포맷 (block_size → bytes_per_block → elems_per_block):
/// Q4_0: 18 → 32, Q4_1: 20 → 32
/// Q2_K: 84 → 256, Q3_K: 110 → 256, Q4_K: 144 → 256
/// Q5_K: 176 → 256, Q6_K: 210 → 256
fn dequantize_u8_from_bytes(bytes: &[u8], byte_count: usize) -> Result<Vec<f32>> {
    if byte_count == 0 {
        return Ok(vec![]);
    }

    // 포맷을 block 크기로 추론: 가능한 block 크기 목록 (block_bytes, elems_per_block)
    const FORMATS: &[(usize, usize)] = &[
        (18, 32),   // Q4_0
        (20, 32),   // Q4_1
        (84, 256),  // Q2_K
        (110, 256), // Q3_K
        (144, 256), // Q4_K
        (176, 256), // Q5_K
        (210, 256), // Q6_K
    ];

    for &(block_bytes, elems_per_block) in FORMATS {
        if byte_count % block_bytes == 0 {
            let n_blocks = byte_count / block_bytes;
            let numel = n_blocks * elems_per_block;

            match block_bytes {
                18 => {
                    // Q4_0
                    let mut out = vec![0.0f32; numel];
                    for (block_idx, chunk) in bytes.chunks_exact(18).enumerate() {
                        let d =
                            half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                        let qs = &chunk[2..18];
                        let base = block_idx * 32;
                        for i in 0..16 {
                            out[base + i] = ((qs[i] & 0x0F) as f32 - 8.0) * d;
                            out[base + i + 16] = ((qs[i] >> 4) as f32 - 8.0) * d;
                        }
                    }
                    return Ok(out);
                }
                20 => {
                    // Q4_1
                    let mut out = vec![0.0f32; numel];
                    for (block_idx, chunk) in bytes.chunks_exact(20).enumerate() {
                        let d =
                            half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                        let m =
                            half::f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32();
                        let qs = &chunk[4..20];
                        let base = block_idx * 32;
                        for i in 0..16 {
                            out[base + i] = (qs[i] & 0x0F) as f32 * d + m;
                            out[base + i + 16] = (qs[i] >> 4) as f32 * d + m;
                        }
                    }
                    return Ok(out);
                }
                84 => {
                    // Q2_K
                    let mut out = vec![0.0f32; numel];
                    for (block_idx, chunk) in bytes.chunks_exact(84).enumerate() {
                        let block = unsafe { &*(chunk.as_ptr() as *const quantize::BlockQ2_K) };
                        let mut tmp = [0.0f32; 256];
                        quantize::dequantize_q2_k(block, &mut tmp);
                        out[block_idx * 256..(block_idx + 1) * 256].copy_from_slice(&tmp);
                    }
                    return Ok(out);
                }
                110 => {
                    // Q3_K
                    let mut out = vec![0.0f32; numel];
                    for (block_idx, chunk) in bytes.chunks_exact(110).enumerate() {
                        let block = unsafe { &*(chunk.as_ptr() as *const quantize::BlockQ3_K) };
                        let mut tmp = [0.0f32; 256];
                        quantize::dequantize_q3_k(block, &mut tmp);
                        out[block_idx * 256..(block_idx + 1) * 256].copy_from_slice(&tmp);
                    }
                    return Ok(out);
                }
                144 => {
                    // Q4_K
                    let mut out = vec![0.0f32; numel];
                    for (block_idx, chunk) in bytes.chunks_exact(144).enumerate() {
                        let block = unsafe { &*(chunk.as_ptr() as *const quantize::BlockQ4_K) };
                        let mut tmp = [0.0f32; 256];
                        quantize::dequantize_q4_k(block, &mut tmp);
                        out[block_idx * 256..(block_idx + 1) * 256].copy_from_slice(&tmp);
                    }
                    return Ok(out);
                }
                176 => {
                    // Q5_K
                    let mut out = vec![0.0f32; numel];
                    for (block_idx, chunk) in bytes.chunks_exact(176).enumerate() {
                        let block = unsafe { &*(chunk.as_ptr() as *const quantize::BlockQ5_K) };
                        let mut tmp = [0.0f32; 256];
                        quantize::dequantize_q5_k(block, &mut tmp);
                        out[block_idx * 256..(block_idx + 1) * 256].copy_from_slice(&tmp);
                    }
                    return Ok(out);
                }
                210 => {
                    // Q6_K
                    let mut out = vec![0.0f32; numel];
                    for (block_idx, chunk) in bytes.chunks_exact(210).enumerate() {
                        let block = unsafe { &*(chunk.as_ptr() as *const quantize::BlockQ6_K) };
                        let mut tmp = [0.0f32; 256];
                        quantize::dequantize_q6_k(block, &mut tmp);
                        out[block_idx * 256..(block_idx + 1) * 256].copy_from_slice(&tmp);
                    }
                    return Ok(out);
                }
                _ => {}
            }
        }
    }

    // 알 수 없는 양자화 포맷: byte_count를 numel로 취급해 zeros 반환
    eprintln!(
        "matmul: 알 수 없는 양자화 포맷 (bytes={}), zeros로 fallback",
        byte_count
    );
    Ok(vec![0.0f32; byte_count])
}

/// I8 (Q8_0) 데이터를 byte_count로부터 dequantize.
fn dequantize_i8_from_bytes(bytes: &[u8], byte_count: usize) -> Result<Vec<f32>> {
    if byte_count == 0 {
        return Ok(vec![]);
    }
    // Q8_0: 34 bytes/block → 32 elems
    if byte_count % 34 == 0 {
        let n_blocks = byte_count / 34;
        let numel = n_blocks * 32;
        let mut out = vec![0.0f32; numel];
        for (block_idx, chunk) in bytes.chunks_exact(34).enumerate() {
            let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            let base = block_idx * 32;
            for i in 0..32 {
                out[base + i] = chunk[2 + i] as i8 as f32 * d;
            }
        }
        return Ok(out);
    }
    // Q8_1: 36 bytes/block → 32 elems
    if byte_count % 36 == 0 {
        let n_blocks = byte_count / 36;
        let numel = n_blocks * 32;
        let mut out = vec![0.0f32; numel];
        for (block_idx, chunk) in bytes.chunks_exact(36).enumerate() {
            let d = half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            let base = block_idx * 32;
            for i in 0..32 {
                out[base + i] = chunk[4 + i] as i8 as f32 * d;
            }
        }
        return Ok(out);
    }
    // fallback: raw i8 해석
    Ok(bytes.iter().map(|&b| b as i8 as f32).collect())
}

/// F32 gemv: weight x input → output
///
/// weight shape:
/// - 2D [out_features, in_features]: 직접 사용
/// - 1D [byte_count] (양자화): dequantize 후 in_features 기반으로 reshape
///
/// input shape: [in] 또는 [seq_len, in] 또는 [1, in]
pub fn gemv_f32(weight: &Tensor, input: &Tensor) -> Result<Tensor> {
    let w_shape = weight.shape();
    let x_shape = input.shape();
    let x_numel: usize = x_shape.iter().product();

    if x_numel == 0 {
        return Err(RnbError::InvalidGraph("gemv_f32: empty input".into()));
    }

    // weight를 f32 vec으로 변환 (양자화 포함)
    let w_vec = tensor_to_f32_vec(weight)?;

    // out_features, in_features 결정
    let (out_features, in_features) = if w_shape.len() == 2 {
        // 명시적 2D weight
        (w_shape[0], w_shape[1])
    } else {
        // 1D (양자화 후 flat): in_features는 x_numel의 마지막 dim
        // x가 [seq_len, in] 또는 [in]이면 in_features = x_shape.last()
        let in_f = *x_shape.last().unwrap_or(&x_numel);
        if in_f == 0 || w_vec.len() % in_f != 0 {
            return Err(RnbError::InvalidGraph(format!(
                "gemv_f32: 1D weight(numel={}) is not divisible by in_features({})",
                w_vec.len(),
                in_f
            )));
        }
        (w_vec.len() / in_f, in_f)
    };

    // seq_len 계산
    if x_numel % in_features != 0 {
        return Err(RnbError::ShapeMismatch {
            expected: vec![1, in_features],
            got: x_shape.to_vec(),
        });
    }
    let seq_len = x_numel / in_features;

    let x_vec = tensor_to_f32_vec(input)?;

    let mut out_data = vec![0.0f32; seq_len * out_features];

    for s in 0..seq_len {
        let x_slice = &x_vec[s * in_features..(s + 1) * in_features];
        for row in 0..out_features {
            let w_row = &w_vec[row * in_features..(row + 1) * in_features];
            let mut acc = 0.0f32;
            for col in 0..in_features {
                acc += w_row[col] * x_slice[col];
            }
            out_data[s * out_features + row] = acc;
        }
    }

    Ok(Tensor::from_vec(out_data, &[seq_len, out_features]))
}

/// MatMul 연산 dispatch 함수.
/// 그래프에서 in_port=0 → activation (x), in_port=1 → weight (w)
/// gemv_f32는 weight를 첫 번째 인자로 받으므로 순서를 바꿔서 호출한다.
pub fn execute(inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    if inputs.len() < 2 {
        return Err(RnbError::InvalidGraph(format!(
            "matmul execute: 최소 2개의 입력 필요, 현재 {}개",
            inputs.len()
        )));
    }
    // inputs[0] = activation (x), inputs[1] = weight (w)
    let result = gemv_f32(inputs[1], inputs[0])?;
    Ok(vec![result])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_core::tensor::Tensor;

    fn to_f32_vec(t: &Tensor) -> Vec<f32> {
        tensor_to_f32_vec(t).unwrap()
    }

    #[test]
    fn test_gemv_f32_identity() {
        let w = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let x = Tensor::from_slice(&[3.0f32, 4.0], &[1, 2]);
        let y = gemv_f32(&w, &x).unwrap();
        let data = to_f32_vec(&y);
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_f32_3x2() {
        let w = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let x = Tensor::from_slice(&[1.0f32, 1.0], &[1, 2]);
        let y = gemv_f32(&w, &x).unwrap();
        let data = to_f32_vec(&y);
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 7.0).abs() < 1e-5);
        assert!((data[2] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_execute_input_order() {
        // execute: inputs[0]=activation, inputs[1]=weight
        // w [2,3] x x [1,3] (x가 activation, w가 weight)
        // 아니다 - w [out=2, in=3], x numel=3 이어야 함
        let w = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let x = Tensor::from_slice(&[5.0f32, 6.0, 7.0], &[1, 3]);
        // execute: inputs[0]=x (activation), inputs[1]=w (weight)
        let result = execute(&[&x, &w]).unwrap();
        let data = to_f32_vec(&result[0]);
        // w[0,:] = [1,0,0], x = [5,6,7] → 5
        assert!((data[0] - 5.0).abs() < 1e-5, "got {}", data[0]);
        // w[1,:] = [0,1,0], x = [5,6,7] → 6
        assert!((data[1] - 6.0).abs() < 1e-5, "got {}", data[1]);
    }

    #[test]
    fn test_dequantize_u8_q4_0() {
        // Q4_0 포맷: 32 elements, 18 bytes/block
        // scale=0 이면 모두 0
        let mut block = vec![0u8; 18];
        // d(f16) = 0.0 → bytes [0,0]
        // qs: all 0x88 → nibbles all 8 → (8-8)*0 = 0
        block[2..18].iter_mut().for_each(|b| *b = 0x88);

        let result = dequantize_u8_from_bytes(&block, 18).unwrap();
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&v: &f32| v.abs() < 1e-5));
    }
}
