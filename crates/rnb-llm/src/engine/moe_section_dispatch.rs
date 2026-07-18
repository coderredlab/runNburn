use super::moe_section_layout::MoeSectionGateUpLayout;

#[cfg(target_arch = "aarch64")]
pub(super) type MoeSectionQ8KBlock = super::gemm_runtime::Q8KBlock;

#[inline]
pub(super) fn gate_up_unit_size(layout: MoeSectionGateUpLayout) -> usize {
    match layout {
        MoeSectionGateUpLayout::Q4KPair | MoeSectionGateUpLayout::ScalePlane => {
            super::gemm_runtime::moe_section::GU_PAIR_Q4K_BYTES
        }
        MoeSectionGateUpLayout::UnpackedScales => {
            super::gemm_runtime::moe_section::GU_PAIR_Q4K_UNPACKED_SCALES_BYTES
        }
    }
}

#[inline]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) fn down_q5k_unit_size() -> usize {
    super::gemm_runtime::moe_section::DOWN_Q5K_INT_SCALE_BYTES
}

#[inline]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) fn shared_gate_up_q80_unit_size() -> usize {
    super::gemm_runtime::moe_section::SHARED_GU_Q8K_UNIT_BYTES
}

#[inline]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) fn shared_down_q80_unit_size() -> usize {
    super::gemm_runtime::moe_section::SHARED_DOWN_Q80_INT_SCALE_BYTES
}

#[cfg(target_arch = "aarch64")]
pub(super) fn quantize_q8k(input: &[f32]) -> Vec<MoeSectionQ8KBlock> {
    super::gemm_runtime::quantize_input_q8k(input)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn quantize_q8k_into<'a>(
    input: &[f32],
    scratch: &'a mut [MoeSectionQ8KBlock],
) -> &'a [MoeSectionQ8KBlock] {
    super::gemm_runtime::quantize_input_q8k_into(input, scratch);
    scratch
}

#[cfg(target_arch = "aarch64")]
pub(super) fn gate_up_gemm_layout(
    layout: MoeSectionGateUpLayout,
) -> super::gemm_runtime::moe_section::GateUpRowLayout {
    match layout {
        MoeSectionGateUpLayout::Q4KPair => {
            super::gemm_runtime::moe_section::GateUpRowLayout::Q4KPair
        }
        MoeSectionGateUpLayout::UnpackedScales => {
            super::gemm_runtime::moe_section::GateUpRowLayout::UnpackedScales
        }
        MoeSectionGateUpLayout::ScalePlane => {
            super::gemm_runtime::moe_section::GateUpRowLayout::ScalePlane
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn dot_gate_up_row_q4k(
    layout: MoeSectionGateUpLayout,
    row_bytes: &[u8],
    scale_bytes: Option<&[u8]>,
    input: &[MoeSectionQ8KBlock],
) -> (f32, f32) {
    super::gemm_runtime::moe_section::dot_gate_up_row_q4k(
        gate_up_gemm_layout(layout),
        row_bytes,
        scale_bytes,
        input,
    )
}

#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn dot_down_row_q5k(row_bytes: &[u8], input: &[MoeSectionQ8KBlock]) -> f32 {
    super::gemm_runtime::moe_section::dot_down_row_q5k(row_bytes, input)
}

#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn dot_shared_gate_up_row_q80(
    row_bytes: &[u8],
    input: &[MoeSectionQ8KBlock],
) -> (f32, f32) {
    super::gemm_runtime::moe_section::dot_shared_gate_up_row_q80(row_bytes, input)
}

#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn dot_shared_down_row_q80(
    row_bytes: &[u8],
    input: &[MoeSectionQ8KBlock],
) -> f32 {
    super::gemm_runtime::moe_section::dot_shared_down_row_q80(row_bytes, input)
}
