//! Split sparse-expert MoE with an always-on shared expert.
//!
//! Qwen3.5 MoE and Hy3 share this execution shape. Model behavior remains
//! metadata-driven: routing function, selection bias, normalization, and scale
//! come from the loaded GGUF contract.

mod decode;
mod glm_metal_prefill;
#[cfg(feature = "cuda")]
mod glm_prefill;
pub(in crate::engine) mod jit_request;
mod loading;
#[cfg(target_arch = "aarch64")]
mod moe_section_decode;
pub(in crate::engine) mod moe_types;
mod moe_view;
pub(in crate::engine) mod packed_wiring;
mod page_cache;
mod prefill;
mod prefill_cpu;
#[cfg(target_arch = "aarch64")]
mod prefill_cpu_expert_group;
pub(in crate::engine) mod routing;
mod weights;

pub(in crate::engine) use decode::decode_shared_expert_moe;
pub(in crate::engine) use loading::load_shared_expert_moe_layer;
pub(in crate::engine) use page_cache::wire_sparse_expert_page_cache;
pub(in crate::engine) use prefill::forward_shared_expert_moe;
#[cfg(feature = "cuda")]
pub(in crate::engine) use prefill::{
    qwen35moe_device_input_supported, try_forward_ffn_qwen35moe_device_input,
    try_forward_ffn_qwen35moe_device_input_carrier,
};
pub(in crate::engine) use weights::SharedExpertMoELayerWeights;
