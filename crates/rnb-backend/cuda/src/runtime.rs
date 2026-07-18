use rnb_backend_api::MoeJitByteRange;
use rnb_memory::residency::{ResidencyCandidate, ResidencyPlanner, ResidencyTransferStats};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::Instant;

use crate::tuning;

static DEFAULT_CUDA_COMPUTE: OnceLock<Mutex<Option<CudaState>>> = OnceLock::new();
static GLM_EXPERT_PARALLEL_CUDA_COMPUTE: LazyLock<Mutex<Option<CudaState>>> =
    LazyLock::new(|| Mutex::new(None));
static GLM_EXPERT_PARALLEL_LOGGED: OnceLock<()> = OnceLock::new();
mod jit;
use jit::{cache_snapshot, cache_stats};
pub use jit::{
    cache_snapshot as cuda_cache_snapshot, CudaCacheSnapshot, CudaMoeJitLoader, CudaMoeJitStats,
};

mod api;
pub use api::*;

mod attention;
pub mod chain_diag_bridge;
mod dense;
#[cfg(test)]
pub(in crate::runtime) use dense::{
    dense_q4_batch_f16_down_enabled_for_test, dense_q4_batch_f16_gate_up_enabled_for_test,
    dense_q4_down_dispatch_plan_for_test, dense_q4_gate_up_dispatch_plan_for_test,
    dense_q4_projection_dispatch_plan_for_test, dense_q6_down_dispatch_plan_for_test,
    DenseDownDispatchPlanForTest, DenseGateUpDispatchPlanForTest,
    DenseQ4ProjectionDispatchPlanForTest, DenseQ4ProjectionKindForTest,
};
mod gdn;
mod gdn_prefill_chain;
mod gemv;
mod moe;
mod mtp_verify;
mod nemotron_mamba2;
mod persistent_decode;
pub use mtp_verify::{
    qwen35_mtp_verify_buffer_plan, MtpVerifyBufferPlan, Qwen35MtpDeviceVerifyAttentionKvState,
    Qwen35MtpDeviceVerifyAttentionMoeLayer, Qwen35MtpDeviceVerifyGdnMoeLayer,
    Qwen35MtpDeviceVerifyLayerKind, Qwen35MtpDeviceVerifyPrefixState, Qwen35MtpDeviceVerifyRequest,
    Qwen35MtpDeviceVerifyResult, Qwen35MtpDeviceVerifySsmLayerFinalState,
    Qwen35MtpDeviceVerifySsmLayerPrefixState,
};
pub use nemotron_mamba2::NemotronMamba2DeviceOutput;

#[cfg(test)]
mod test_support;
#[cfg(test)]
pub use test_support::*;

mod types;
pub use gdn_prefill_chain::{
    build_gdn_prefill_chain_conv_input_for_test, derive_gdn_prefill_chain_dims,
    gdn_prefill_chain_conv_state_after_prefix_for_test, plan_gdn_prefill_chain,
    plan_gdn_prefill_chain_for_test, validate_gdn_prefill_chain_shape, GdnPrefillChainDims,
    GdnPrefillChainPlan, GdnPrefillChainQ4KOutput, GdnPrefillChainQ4KRequest, GdnPrefillChainShape,
};
use types::*;
pub use types::{
    Cu71LayerSegmentGraphRuntimeContext, CudaWeightResidencyCounters, DeltaStateSnapshot,
    NemotronPrefillWorkspaceConfig, NemotronPrefillWorkspaceSummary,
};

mod decode_device;
mod device_tensor;
mod state;
#[cfg(test)]
pub(in crate::runtime) use state::{
    q4k_packed_q8dot_bytes_per_block_for_test, q4k_raw_bytes_per_block_for_test,
    q6k_packed_batch_kernel_plan_for_test, q6k_packed_q8dot_bytes_per_block_for_test,
    q6k_raw_bytes_per_block_for_test, quant_resident_budget_plan_for_test,
    validate_q4k_packed_payload_bytes_for_test, validate_q6k_packed_payload_bytes_for_test,
    Q6PackedBatchKernelPlanForTest,
};

mod helpers;
use helpers::*;

mod driver;
use driver::*;

#[cfg(test)]
mod tests;
