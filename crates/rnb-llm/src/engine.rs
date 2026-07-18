use crate::kv_cache::KVCache;
use crate::tokenizer::Tokenizer;
use cpu_runtime::kernels;
use memory_runtime::memtrace;
use rnb_core::tensor::Tensor;
use rnb_loader::{Architecture as ModelArchitecture, GGMLType};

// =============================================================================
// ARM NEON SIMD
// =============================================================================

mod backend_runtime;
mod debug;
mod decode;
mod decode_attention_compute;
mod decode_attention_output;
mod decode_attention_post_qkv;
mod decode_attention_qkv;
mod decode_attention_residual;
mod decode_attention_rope;
mod decode_attention_verify;
mod decode_attn_prewarm;
mod decode_ffn;
mod decode_ffn_dispatch;
mod decode_gdn;
mod decode_gpu;
mod decode_layer_graph;
mod dense_dispatch;
mod dequant;
mod gemv_profile;
mod inference;
mod init;
mod kv_share;
mod layer_weights;
mod layout;
mod load_profile;
mod logits;
#[cfg(feature = "mediatek")]
mod mediatek_ffn;
mod model_init;
mod models;
pub mod moe;
pub mod moe_jit;
#[cfg(feature = "cuda")]
mod persistent_decode_dispatch;
pub use mtp::{MtpAutoPolicy, MtpAutoResourceHint};
mod moe_profile;
mod moe_routing;
mod moe_section;
mod moe_section_dispatch;
mod moe_section_layout;
mod moe_shadow_dispatch;
pub mod moe_trace;
mod moe_types;
pub(crate) mod mtp;
mod norm;
mod packed_wiring;
mod prefill;
mod prefill_handoff;
pub mod q4_microbench;
mod quantized_dispatch;
mod quantized_packing;
mod quantized_weight;
mod quantized_weight_tensor;
mod quantized_weight_types;
mod residency;
mod runtime;
mod scalar_gemv;
mod state;
mod threading;
mod trace;
mod types;
pub(crate) mod verify_window;
mod weight_loading;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use decode::attn_carrier_eligible;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use decode::qwen_attn_moe_chain_eligible;
use decode::{decode_attention_layer, decode_attention_layer_with_rope_pos};
use decode_attention_compute::decode_attention_compute;
use decode_attention_output::decode_attention_output_projection;
use decode_attention_post_qkv::apply_decode_attention_qkv_postprocess;
use decode_attention_qkv::decode_attention_qkv_projection;
use decode_attention_residual::apply_decode_attention_residual;
use decode_attention_rope::apply_decode_rope;
use decode_attention_verify::{log_decode_attention_gpu_debug, verify_decode_attention_qkv};
use decode_ffn::decode_ffn;
use decode_ffn_dispatch::decode_ffn_layer;
use decode_gdn::decode_gdn_layer;
use decode_gpu::gpu_gemv_into_if_supported;
use dequant::dequantize_bytes_to_f32;
#[cfg(test)]
use dequant::dequantize_row_to_slice_if_supported;
pub use gemv_profile::{gemv_profile_report, reset_gemv_profile};
#[cfg(target_arch = "aarch64")]
pub use quantized_dispatch::{packed_dispatch_report, reset_packed_dispatch};
#[cfg(not(target_arch = "aarch64"))]
pub fn packed_dispatch_report() -> Option<String> {
    None
}
#[cfg(not(target_arch = "aarch64"))]
pub fn reset_packed_dispatch() {}
pub use debug::{PrefillDriftRecord, PrefillDriftTrace};
pub use init::EngineLoadConfig;
use layer_weights::*;
use layout::*;
use logits::*;
pub use moe_jit::moe_jit_report;
pub use moe_profile::{moe_profile_report, reset_moe_profile};
pub(crate) use moe_section::MoeSectionDecodeLayer;
#[cfg(test)]
use moe_section::{convert_moe_section_decode_layer, offset_of_subslice};
#[cfg(test)]
pub(crate) use moe_section::{MoeSectionExpert, MoeSectionRowDown, MoeSectionRowGU};
#[allow(unused_imports)]
use norm::*;
#[cfg(test)]
use platform_runtime::packed_rnb_default_big_affinity;
use prefill_handoff::*;
#[cfg(feature = "vulkan")]
use quantized_dispatch::decode_ffn_up_cpu_best_effort;
use quantized_dispatch::{
    decode_attention_qkv_cpu_into, decode_ffn_gate_up_cpu_into, decode_gdn_qkv_gate_cpu_into,
    prefill_dual_gemv_q8_or_f32, prefill_gate_up_vectors, prefill_gemv_vec,
    prefill_quantized_input_for_weight,
};
use quantized_weight::*;
use quantized_weight_types::*;
#[cfg(feature = "cuda")]
pub(crate) use runtime::cuda_runtime;
#[cfg(feature = "vulkan")]
pub(crate) use runtime::gpu_runtime;
#[cfg(feature = "metal")]
pub(crate) use runtime::metal_runtime;
pub(crate) use runtime::{
    cpu_runtime, gemm_runtime, memory_runtime, packed_runtime, platform_runtime, policy,
};
pub use state::{Engine, EngineSequenceState};
#[allow(unused_imports)]
use trace::*;
pub use types::ModelMetadata;
pub(crate) use types::ScratchBuffers;

use models::gemma::*;

mod decode_inference;
pub(in crate::engine) mod forward;
mod forward_gdn;
use forward::*;
use forward_gdn::*;
use models::shared_expert_moe::{decode_shared_expert_moe, forward_shared_expert_moe};

mod tests;

/// cu59 axis A — chain function sub-phase aggregate 출력 진입점.
/// bench binary 의 main 끝에서 명시적으로 호출.
/// `RNB_CU58_DIAG_CHAIN` 환경변수가 OFF 면 no-op.
pub fn dump_chain_diag_aggregate() {
    #[cfg(feature = "cuda")]
    forward::chain_diag::dump_aggregate();
}

pub fn metal_prefill_atn_full_timing_enabled() -> bool {
    backend_runtime::metal_prefill_atn_full_timing_enabled()
}

pub fn metal_prefill_atn_full_expected_dense_layer() {
    backend_runtime::metal_prefill_atn_full_expected_dense_layer();
}

pub fn metal_prefill_atn_full_record_core_hit() {
    backend_runtime::metal_prefill_atn_full_record_core_hit();
}

pub fn metal_prefill_atn_full_record_full_layer_hit() {
    backend_runtime::metal_prefill_atn_full_record_full_layer_hit();
}

pub fn metal_prefill_atn_full_record_skip() {
    backend_runtime::metal_prefill_atn_full_record_skip();
}

pub fn metal_prefill_atn_full_record_adapter_reject() {
    backend_runtime::metal_prefill_atn_full_record_adapter_reject();
}

pub fn metal_prefill_atn_full_record_backend_err() {
    backend_runtime::metal_prefill_atn_full_record_backend_err();
}

pub fn reset_metal_prefill_atn_full_counters() {
    backend_runtime::metal_prefill_atn_full_counters_reset();
}

pub fn report_metal_prefill_atn_full_counters(label: &str) {
    backend_runtime::metal_prefill_atn_full_counters_report(label);
}

pub fn reset_metal_prefill_atn_o_tail_counters() {
    backend_runtime::metal_prefill_atn_o_tail_counters_reset();
}

pub fn report_metal_prefill_atn_o_tail_counters(label: &str) {
    backend_runtime::metal_prefill_atn_o_tail_counters_report(label);
}

pub fn reset_metal_decode_parity_counters() {
    backend_runtime::metal_decode_parity_counters_reset();
}

pub fn report_metal_decode_parity_counters(label: &str) {
    backend_runtime::metal_decode_parity_counters_report(label);
}
