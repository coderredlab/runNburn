pub(super) fn gemv_f32(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    #[cfg(feature = "cuda")]
    {
        let weight_bytes = unsafe {
            std::slice::from_raw_parts(weight.as_ptr().cast::<u8>(), std::mem::size_of_val(weight))
        };
        let result = if seq_len == 1 {
            super::cuda_runtime::decode_gemv(
                rnb_loader::GGMLType::F32,
                weight_bytes,
                rows,
                cols,
                input,
            )
        } else {
            super::cuda_runtime::prefill_gemv(
                rnb_loader::GGMLType::F32,
                weight_bytes,
                rows,
                cols,
                input,
                seq_len,
            )
        }
        .unwrap_or_else(|| panic!("CUDA F32 GEMV route is unavailable; CPU fallback is disabled"))
        .unwrap_or_else(|err| panic!("CUDA F32 GEMV failed; CPU fallback is disabled: {err}"));
        assert_eq!(
            result.len(),
            rows * seq_len,
            "CUDA F32 GEMV returned an invalid output length"
        );
        output[..result.len()].copy_from_slice(&result);
    }
    #[cfg(not(feature = "cuda"))]
    super::gemm_runtime::f32_gemv::gemv_f32(weight, input, output, rows, cols, seq_len);
}

#[cfg(not(feature = "cuda"))]
pub(super) fn gemv_bf16(
    weight: &[u16],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    super::gemm_runtime::f32_gemv::gemv_bf16(weight, input, output, rows, cols, seq_len);
}

#[cfg(not(feature = "cuda"))]
pub(super) fn gemv_f16(
    weight: &[u16],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    super::gemm_runtime::f32_gemv::gemv_f16(weight, input, output, rows, cols, seq_len);
}
