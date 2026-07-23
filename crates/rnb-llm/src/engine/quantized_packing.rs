//! Quantized weight packing and prepack helpers.

#[cfg(target_arch = "aarch64")]
use super::gemm_runtime;
#[cfg(target_arch = "aarch64")]
use super::policy;

#[cfg(target_arch = "aarch64")]
pub(super) fn q80_prepack_load_enabled() -> bool {
    policy::q80_prepack_load_enabled()
}

#[cfg(target_arch = "aarch64")]
pub(super) fn repack_q4k_artifacts(
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> gemm_runtime::repack::Q4KRepackArtifacts {
    gemm_runtime::repack::repack_q4k_artifacts(bytes, rows, cols)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn pack_q80_row_pairs(bytes: &[u8], rows: usize, bytes_per_row: usize) -> Vec<u8> {
    gemm_runtime::pack_q8_0_pair::pack_q8_0_row_pairs(bytes, rows, bytes_per_row)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn build_q80_f32_scales(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    total_bytes: usize,
) -> Vec<f32> {
    gemm_runtime::quantized_packing::build_q80_f32_scales(bytes, rows, cols, total_bytes)
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn quantize_q8_for_test<'a>(
    input: &[f32],
    scratch: &'a mut [gemm_runtime::Q8Block],
) -> &'a [gemm_runtime::Q8Block] {
    gemm_runtime::quantize_input_q8_into(input, scratch);
    scratch
}
