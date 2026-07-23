pub(crate) use rnb_runtime::memory::ExpertBundleObservationReceipt;
pub(crate) use rnb_runtime::policy::cuda_decode_moe_combined_enabled;
pub use rnb_runtime::{
    default_moe_jit_loader, moe_jit_report, BackendError, BackendErrorKind, BackendKind, BackendOp,
    BackendRegistry, BackendResult, DecodeWeightKind, GpuBackend, KvBucketView, KvarnChainView,
    MoeJitByteRange, MoeJitExpertLoad, MoeJitLoadRequest, MoeJitLoadSink, MoeRouteSlot,
    QuantFormat, QuantizedWeightView,
};

pub(crate) mod cpu {
    pub(crate) use rnb_runtime::cpu::*;
}

pub(crate) mod cpu_phase {
    pub(crate) use rnb_runtime::cpu_phase::*;
}

pub(crate) mod gemm {
    pub(crate) use rnb_runtime::gemm::*;
}
#[cfg(feature = "cuda")]
pub(crate) mod compute {
    pub(crate) use rnb_runtime::compute::*;
}

pub(crate) mod memory {
    pub(crate) use rnb_runtime::memory::*;
}
#[cfg(feature = "mediatek")]
pub(crate) mod mediatek {
    pub(crate) use rnb_runtime::mediatek::*;
}

pub(crate) mod policy {
    pub(crate) use rnb_runtime::policy::*;
}

pub(crate) mod scheduler {
    pub(crate) use rnb_runtime::scheduler::*;
}
#[cfg(feature = "cuda")]
pub(crate) mod tuning {
    pub(crate) use rnb_runtime::compute::tuning::*;
    pub(crate) use rnb_runtime::compute::{
        cu65_device_qkv_enabled, cu68_layer_graph_enabled, cu69_dense_chain_graph_enabled,
        cu71_layer_segment_graph_enabled, cu71_layer_segment_graph_trace_enabled,
    };
}

#[cfg(any(feature = "cuda", test))]
pub(crate) mod workspace {
    pub(crate) use rnb_runtime::{
        plan_nemotron_prefill_workspace, NemotronPrefillWorkspaceDecision,
        NemotronPrefillWorkspacePlan, NemotronPrefillWorkspaceRequest,
    };
}

pub(crate) fn profiling_enabled() -> bool {
    policy::profiling_enabled()
}

pub(crate) fn spec_profile_enabled() -> bool {
    policy::spec_profile_enabled()
}

pub(crate) fn mtp_trace_enabled() -> bool {
    policy::mtp_trace_enabled()
}

pub(crate) fn spec_force_batch_verify_enabled() -> bool {
    policy::spec_force_batch_verify_enabled()
}

pub(crate) fn spec_decode_fast_window_enabled() -> bool {
    policy::spec_decode_fast_window_enabled()
}

pub(crate) fn spec_batch_no_bonus_enabled() -> bool {
    policy::spec_batch_no_bonus_enabled()
}

pub(crate) fn spec_batch_no_bonus_override() -> Option<bool> {
    policy::spec_batch_no_bonus_override()
}

pub(crate) fn spec_batch_prefix_snapshot_enabled() -> bool {
    policy::spec_batch_prefix_snapshot_enabled()
}

pub(crate) fn spec_mtp_sequential_multi_enabled() -> bool {
    policy::spec_mtp_sequential_multi_enabled()
}

pub(crate) fn mtp_batch_verify_enabled() -> bool {
    policy::mtp_batch_verify_enabled()
}

pub(crate) fn mtp_batch_verify_disabled() -> bool {
    policy::mtp_batch_verify_disabled()
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn mtp_batch_decode_chain_enabled() -> bool {
    policy::mtp_batch_decode_chain_enabled()
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn metal_kvarn_chain_attention_enabled() -> bool {
    policy::metal_kvarn_chain_attention_enabled()
}

pub(crate) fn mtp_fast_retain_enabled() -> bool {
    policy::mtp_fast_retain_enabled()
}

pub(crate) fn mtp_draft_only_enabled() -> bool {
    policy::mtp_draft_only_enabled()
}

pub(crate) fn mtp_shadow_precompute_enabled() -> bool {
    policy::mtp_shadow_precompute_enabled()
}

pub(crate) fn mtp_runway_max_extra() -> Option<usize> {
    policy::mtp_runway_max_extra()
}

#[cfg(feature = "cuda")]
pub(crate) fn moe_route_hist_enabled() -> bool {
    cuda::moe_route_hist_enabled()
}

pub(crate) mod platform {
    pub(crate) use rnb_runtime::platform::*;
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    pub(crate) use rnb_runtime::compute::tuning::moe_route_hist_enabled;
    pub(crate) use rnb_runtime::cuda_inference::cuda::*;
}

#[cfg(feature = "metal")]
pub(crate) mod metal {
    pub(crate) use rnb_runtime::metal_inference::*;
}

#[cfg(feature = "vulkan")]
pub(crate) mod gpu {
    pub(crate) use rnb_runtime::vulkan_inference::gpu::*;
}

#[cfg(test)]
mod boundary_tests;
