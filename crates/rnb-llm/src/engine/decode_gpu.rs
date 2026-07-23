//! Decode GPU helper boundary shared by attention and GDN decode paths.

use super::*;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(super) fn gpu_gemv_into_if_supported(
    weight: &QuantizedWeight,
    input: &[f32],
    output: &mut [f32],
    label: &str,
    rms_used_cuda: bool,
) -> crate::error::Result<bool> {
    backend_runtime::decode_gemv_into_if_supported(weight, input, output, label, rms_used_cuda)
}
