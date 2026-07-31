pub mod activation;
pub mod attention;
pub mod conv;
pub mod delta_net;
pub mod elementwise;
pub mod matmul;
pub mod norm;
pub mod rope;
pub mod shape;

use rnb_core::tensor::{DType, Tensor};

fn f32_slice_from_bytes(bytes: &[u8]) -> &[f32] {
    assert_eq!(
        bytes.len() % std::mem::size_of::<f32>(),
        0,
        "tensor_as_f32_slice: byte length must be a multiple of 4"
    );
    // f32에는 invalid bit pattern이 없고, prefix/suffix 검사로 참조 생성 전 정렬을 검증한다.
    let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
    assert!(
        prefix.is_empty() && suffix.is_empty(),
        "tensor_as_f32_slice: byte address is not aligned for f32"
    );
    values
}

/// 연속적인 F32 텐서에서 f32 슬라이스를 가져옴.
/// 비연속 텐서나 F32가 아닌 텐서에서 호출하면 panic.
pub fn tensor_as_f32_slice(t: &Tensor) -> &[f32] {
    assert_eq!(
        t.dtype(),
        DType::F32,
        "tensor_as_f32_slice requires an F32 tensor"
    );
    let bytes = t
        .as_bytes()
        .expect("tensor_as_f32_slice requires a contiguous host tensor");
    if bytes.is_empty() {
        return &[];
    }
    f32_slice_from_bytes(bytes)
}

/// 연속적인 F32 텐서를 Vec<f32>로 복사해서 반환.
pub fn tensor_to_f32_vec(t: &Tensor) -> Vec<f32> {
    tensor_as_f32_slice(t).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_as_f32_slice_returns_f32_values() {
        let tensor = Tensor::from_vec(vec![1.25_f32, -2.5], &[2]);

        assert_eq!(tensor_as_f32_slice(&tensor), &[1.25, -2.5]);
    }

    #[test]
    #[should_panic(expected = "requires an F32 tensor")]
    fn tensor_as_f32_slice_rejects_non_f32_tensor() {
        let tensor = Tensor::from_vec(vec![0_u8; 4], &[4]);

        let _ = tensor_as_f32_slice(&tensor);
    }

    #[test]
    #[should_panic(expected = "requires an F32 tensor")]
    fn tensor_as_f32_slice_rejects_empty_non_f32_tensor() {
        let tensor = Tensor::from_vec(Vec::<u8>::new(), &[0]);

        let _ = tensor_as_f32_slice(&tensor);
    }

    #[test]
    #[should_panic(expected = "byte address is not aligned for f32")]
    fn f32_slice_from_bytes_rejects_unaligned_storage() {
        let storage = [0_u8; 5];
        let offset = usize::from(storage.as_ptr().align_offset(std::mem::align_of::<f32>()) == 0);

        let _ = f32_slice_from_bytes(&storage[offset..offset + 4]);
    }
}
