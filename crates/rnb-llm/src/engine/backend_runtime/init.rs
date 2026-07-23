#[cfg(feature = "cuda")]
use crate::engine::cuda_runtime;
#[cfg(feature = "vulkan")]
use crate::engine::gpu_runtime as gpu;
#[cfg(feature = "vulkan")]
use crate::engine::layer_weights::{LayerType, ModelWeights};
#[cfg(feature = "vulkan")]
use crate::engine::types::ModelMetadata;

#[cfg(feature = "mediatek")]
fn clear_mediatek_cache_for_engine_init() {
    crate::engine::mediatek_runtime::clear_gated_gelu_ffn_f32_cache();
    crate::engine::mediatek_runtime::clear_gated_gelu_ffn_quantized_cache();
}

#[cfg(not(feature = "mediatek"))]
fn clear_mediatek_cache_for_engine_init() {}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn reset_backend_state_for_engine_init() -> crate::error::Result<()> {
    cuda_runtime::reset_state_for_engine_init().map_err(crate::error::LlmError::Forward)?;
    clear_mediatek_cache_for_engine_init();
    Ok(())
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn release_prefill_residency_after_prefill(
    architecture: rnb_loader::Architecture,
) -> crate::error::Result<()> {
    let env_bool_or = |name: &str, default: bool| {
        crate::engine::policy::env_string(name)
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
            })
            .unwrap_or(default)
    };

    if matches!(
        architecture,
        rnb_loader::Architecture::Qwen35 | rnb_loader::Architecture::Qwen35MoE
    ) {
        let cache_enabled = env_bool_or("RNB_CUDA_MOE_LAYER_CACHE", true);
        let keep_layer_cache = env_bool_or(
            "RNB_CUDA_QWEN35_KEEP_MOE_LAYER_CACHE_AFTER_PREFILL",
            cache_enabled,
        );
        if !keep_layer_cache {
            cuda_runtime::clear_moe_layer_cache().map_err(crate::error::LlmError::Forward)?;
        }
    } else if matches!(architecture, rnb_loader::Architecture::NemotronHMoE) {
        let keep_layer_cache = crate::engine::policy::env_string(
            "RNB_CUDA_NEMOTRON_KEEP_MOE_LAYER_CACHE_AFTER_PREFILL",
        )
        .as_deref()
            == Some("1");
        if keep_layer_cache {
            if crate::engine::policy::env_string("RNB_CUDA_NEMOTRON_CLEAR_Q4K_AFTER_PREFILL")
                .as_deref()
                == Some("1")
            {
                cuda_runtime::clear_q4k_cache().map_err(crate::error::LlmError::Forward)?;
            }
        } else {
            cuda_runtime::clear_moe_layer_cache().map_err(crate::error::LlmError::Forward)?;
        }
    }
    cuda_runtime::release_q4_f32_after_prefill().map_err(crate::error::LlmError::Forward)?;
    cuda_runtime::release_q8_0_prefill_f32_after_prefill()
        .map_err(crate::error::LlmError::Forward)?;
    cuda_runtime::clear_host_registered_ranges().map_err(crate::error::LlmError::Forward)?;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn clear_host_registered_ranges_before_prefill() -> crate::error::Result<()> {
    cuda_runtime::clear_host_registered_ranges().map_err(crate::error::LlmError::Forward)
}

#[cfg(feature = "cuda")]
pub(in crate::engine) fn clear_decode_attention_kv_cache_before_prefill() -> crate::error::Result<()>
{
    cuda_runtime::clear_decode_attention_kv_cache().map_err(crate::error::LlmError::Forward)
}

#[cfg(not(feature = "cuda"))]
pub(in crate::engine) fn release_prefill_residency_after_prefill(
    _architecture: rnb_loader::Architecture,
) -> crate::error::Result<()> {
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(in crate::engine) fn clear_decode_attention_kv_cache_before_prefill() -> crate::error::Result<()>
{
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(in crate::engine) fn clear_host_registered_ranges_before_prefill() -> crate::error::Result<()> {
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(in crate::engine) fn reset_backend_state_for_engine_init() -> crate::error::Result<()> {
    clear_mediatek_cache_for_engine_init();
    Ok(())
}

#[cfg(feature = "vulkan")]
pub(in crate::engine) fn init_prefill_layer_runtime(
    metadata: &ModelMetadata,
    weights: &ModelWeights,
    ffn_inner_dim: usize,
) -> Option<gpu::Runtime> {
    let max_layer_rows = weights
        .layers
        .iter()
        .map(|l| match l {
            LayerType::Attention(w) => w
                .q_weight
                .rows
                .max(w.k_weight.rows)
                .max(w.v_weight.rows)
                .max(w.o_weight.rows)
                .max(w.ffn_gate_weight.rows)
                .max(w.ffn_up_weight.rows)
                .max(w.ffn_down_weight.rows),
            LayerType::GatedDeltaNet(w) => w
                .qkv_weight
                .rows
                .max(w.gate_weight.rows)
                .max(w.ssm_out.rows)
                .max(w.ffn_gate_weight.rows)
                .max(w.ffn_up_weight.rows)
                .max(w.ffn_down_weight.rows),
            LayerType::NemotronMamba2(w) => w.ssm_in.rows.max(w.ssm_out.rows),
            LayerType::NemotronMoE(w) => w
                .router
                .rows
                .max(w.expert_down.rows)
                .max(w.expert_up.rows)
                .max(w.shared_expert_down.rows)
                .max(w.shared_expert_up.rows),
        })
        .max()
        .unwrap_or(0);
    gpu::init_prefill_layer_runtime(
        metadata.hidden_dim,
        ffn_inner_dim,
        max_layer_rows,
        weights.output.rows,
    )
}
