pub mod activation;
pub mod attention;
pub mod conv;
pub mod delta_net;
pub mod elementwise;
pub mod matmul;
pub mod norm;
pub mod rope;
pub mod shape;

use rnb_core::tensor::Tensor;

/// 연속적인 F32 텐서에서 f32 슬라이스를 가져옴.
/// 비연속 텐서나 F32가 아닌 텐서에서 호출하면 panic.
pub fn tensor_as_f32_slice(t: &Tensor) -> &[f32] {
    let bytes = t
        .as_bytes()
        .expect("tensor_as_f32_slice: 연속적인 호스트 F32 텐서가 필요함");
    assert_eq!(
        bytes.len() % 4,
        0,
        "바이트 길이가 4의 배수가 아님 — F32 텐서가 맞는지 확인해"
    );
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
}

/// 연속적인 F32 텐서를 Vec<f32>로 복사해서 반환.
pub fn tensor_to_f32_vec(t: &Tensor) -> Vec<f32> {
    tensor_as_f32_slice(t).to_vec()
}
