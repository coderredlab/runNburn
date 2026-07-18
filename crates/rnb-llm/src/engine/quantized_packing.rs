//! Quantized weight packing and prepack helpers.

#[cfg(target_arch = "aarch64")]
use super::policy;
use super::{gemm_runtime, packed_runtime};
use rnb_loader::GGMLType;

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) fn gemv_q8k_profile_method(
    packed_quant_type: Option<packed_runtime::QuantType>,
) -> &'static str {
    gemm_runtime::policy::gemv_q8k_profile_method(packed_quant_type)
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) fn runtime_rawmeta_repack_enabled(
    packed_quant_type: Option<packed_runtime::QuantType>,
    ggml_type: GGMLType,
    seq_len: usize,
    rows: usize,
    cols: usize,
) -> bool {
    gemm_runtime::policy::rawmeta_runtime_repack_enabled(
        packed_quant_type,
        ggml_type == GGMLType::Q4_K,
        seq_len,
        rows,
        cols,
    )
}

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

#[cfg(target_arch = "aarch64")]
pub(super) fn pack_q4k_from_raw_meta(data: &[u8], rows: usize, n_blocks: usize) -> Vec<u8> {
    gemm_runtime::pack_q4k::pack_q4k_from_raw_meta(data, rows, n_blocks)
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn pack_q4k_for_test(data: &[u8], rows: usize, blocks_per_row: usize) -> Vec<u8> {
    gemm_runtime::pack_q4k::pack_q4k(data, rows, blocks_per_row)
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn pack_q5k_for_test(data: &[u8], rows: usize, blocks_per_row: usize) -> Vec<u8> {
    gemm_runtime::pack_q5k::pack_q5k(data, rows, blocks_per_row)
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn pack_q6k_for_test(data: &[u8], rows: usize, blocks_per_row: usize) -> Vec<u8> {
    gemm_runtime::pack_q6k::pack_q6k(data, rows, blocks_per_row)
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn quantize_q8_for_test<'a>(
    input: &[f32],
    scratch: &'a mut [gemm_runtime::Q8Block],
) -> &'a [gemm_runtime::Q8Block] {
    gemm_runtime::quantize_input_q8_into(input, scratch);
    scratch
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn quantize_q8k_for_test(input: &[f32]) -> Vec<gemm_runtime::Q8KBlock> {
    gemm_runtime::quantize_input_q8k(input)
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[cfg(target_arch = "aarch64")]
pub(super) fn flatten_q8k_blocks(q8k: &[gemm_runtime::Q8KBlock]) -> (Vec<i8>, Vec<f32>, Vec<i16>) {
    gemm_runtime::quantized_packing::flatten_q8k_blocks(q8k)
}
