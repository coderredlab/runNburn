pub(crate) use rnb_runtime::memory::{ExpertBundleCacheStats, ExpertBundleObservationReceipt};
#[cfg(target_arch = "aarch64")]
pub(crate) use rnb_runtime::moe_section_decode_enabled;
pub(crate) use rnb_runtime::policy::{
    cuda_cache_trace_enabled, cuda_decode_moe_combined_enabled,
    cuda_q2k_q3k_mixed_resident_cpu_enabled,
};
pub use rnb_runtime::{
    default_moe_jit_loader, moe_jit_report, BackendError, BackendErrorKind, BackendKind, BackendOp,
    BackendRegistry, BackendResult, DecodeWeightKind, GpuBackend, KvBucketView, MoeJitByteRange,
    MoeJitExpertLoad, MoeJitLoadRequest, MoeJitLoadSink, MoeRouteSlot, QuantFormat,
    QuantizedWeightView,
};

pub(crate) mod cpu {
    pub(crate) use rnb_runtime::cpu::*;
}

pub(crate) mod gemm {
    pub(crate) use rnb_runtime::gemm::*;
}

pub(crate) mod memory {
    pub(crate) use rnb_runtime::memory::*;
}

pub(crate) mod packed_weights {
    pub(crate) use rnb_runtime::packed_weights::*;
}

pub(crate) mod policy {
    pub(crate) use rnb_runtime::policy::*;
}

pub(crate) mod scheduler {
    pub(crate) use rnb_runtime::scheduler::*;
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

pub(crate) fn mtp_fast_retain_enabled() -> bool {
    policy::mtp_fast_retain_enabled()
}

pub(crate) fn mtp_device_verify_enabled() -> bool {
    policy::mtp_device_verify_enabled()
}

pub(crate) fn mtp_draft_only_enabled() -> bool {
    policy::mtp_draft_only_enabled()
}

pub(crate) fn mtp_shadow_precompute_enabled() -> bool {
    policy::mtp_shadow_precompute_enabled()
}

pub(crate) fn mtp_dump_topk() -> Option<usize> {
    policy::mtp_dump_topk()
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
    pub(crate) use rnb_runtime::compute::tuning::{
        decode_attention_hd512_split_enabled, moe_layer_cache_enabled, moe_route_hist_enabled,
    };
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
