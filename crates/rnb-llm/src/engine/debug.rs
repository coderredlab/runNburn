use super::cpu_runtime::kernels;
use super::logits::{
    finalize_prefill_logits, gemma_effective_unit_offset_output_norm_decode,
    gemma_effective_unit_offset_output_norm_prefill, gemma_skip_output_norm,
};
use super::models::gemma::{
    apply_embedding_scale, apply_gemma_per_layer_branch, detect_gemma_runtime_flavor,
    prepare_gemma_per_layer_base, shared_kv_source_layer,
};
use super::models::gemma::{gemma_ple_after_final_norm, gemma_ple_pre_emb_scale_base};
use super::norm::apply_model_norm;
use super::prefill::run_prefill_layers_cpu_range;
use super::prefill_handoff::PrefillLayerSnapshot;
use super::state::Engine;
use rnb_core::tensor::Tensor;

mod decode_normed;
mod drift_trace;
mod prefill_logits;
mod prefill_normed;
mod prefill_snapshots;

pub use drift_trace::{PrefillDriftRecord, PrefillDriftTrace};
