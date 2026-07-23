#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;

#[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
pub(in crate::engine) fn verify_attention_layer(layer_idx: usize) -> bool {
    #[cfg(feature = "vulkan")]
    {
        return gpu::verify_attention_layer(layer_idx);
    }
    #[cfg(not(feature = "vulkan"))]
    false
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code, unused_variables))]
pub(in crate::engine) fn verify_attention_qkv_layer(layer_idx: usize) -> bool {
    #[cfg(feature = "vulkan")]
    {
        return gpu::verify_attention_qkv_layer(layer_idx);
    }
    #[cfg(not(feature = "vulkan"))]
    false
}

#[cfg_attr(not(feature = "vulkan"), allow(unused_variables))]
pub(in crate::engine) fn verify_gdn_layer(layer_idx: usize) -> bool {
    #[cfg(feature = "vulkan")]
    {
        return gpu::verify_gdn_layer(layer_idx);
    }
    #[cfg(not(feature = "vulkan"))]
    false
}
