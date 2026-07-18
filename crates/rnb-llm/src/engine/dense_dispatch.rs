pub(super) fn gemv_f32(
    weight: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
    seq_len: usize,
) {
    super::gemm_runtime::f32_gemv::gemv_f32(weight, input, output, rows, cols, seq_len);
}

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
