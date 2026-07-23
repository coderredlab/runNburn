pub const NAME: &str = "vulkan";
pub const FEATURE_NAME: &str = "vulkan";
pub const HAS_RUNTIME_ENTRYPOINTS: bool = true;

pub fn feature_enabled() -> bool {
    true
}

pub type LayerGemv = crate::layer_gemv::VulkanLayerGemv;
pub type RuntimeCounters = crate::layer_gemv::RuntimeCounters;
pub type QuantType = crate::weight_cache::QuantType;
pub type WeightId = crate::weight_cache::WeightId;
type WeightKind = crate::weight_cache::WeightKind;
type GpuWeightMode = crate::weight_cache::GpuWeightMode;

const SMALL_MODEL_PREFILL_CHUNK: usize = 32;
const LARGE_MODEL_PREFILL_CHUNK: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerGemvConfig {
    pub max_input: usize,
    pub max_output: usize,
    pub budget_mb: usize,
    pub weight_mode: GpuWeightMode,
}

impl LayerGemvConfig {
    pub fn from_model_shape(
        hidden_dim: usize,
        ffn_inner_dim: usize,
        max_layer_rows: usize,
        output_rows: usize,
    ) -> Self {
        let prefill_chunk = prefill_chunk_size(hidden_dim);
        // mv40: attention dispatch batch 의 sync overhead 가 dominant 한 것을 mv39 측정에서 확인.
        // input_capacity 가 16 token 으로 잘려서 411 prompt 가 26 iter dispatch → 5초 overhead.
        // attention input 식: q_dim × batch ≈ (hidden_dim × 2 GQA Q-only) × 64 batch = hidden_dim × 128.
        // 64 = 보수적 batch 목표 (MAX_BATCH_OUTPUTS / typical num_heads = 512 / 8).
        // 큰 모델 (hidden_dim ≥ 2048, prefill_chunk=128) 은 이미 hidden_dim*128 충족 → 무영향.
        // 작은 모델 (hidden_dim=1024, prefill_chunk=32) 만 32K → 131K 로 4x 증가 (+96KB host buffer).
        let attention_input_floor = hidden_dim.saturating_mul(attention_batch_floor());
        let max_input = hidden_dim
            .max(ffn_inner_dim)
            .max(hidden_dim.saturating_mul(prefill_chunk))
            .max(attention_input_floor);
        let max_output = if output_logits_enabled() {
            output_rows
                .max(max_layer_rows)
                .max(hidden_dim.saturating_mul(2))
        } else {
            max_layer_rows.max(hidden_dim.saturating_mul(2))
        };
        Self {
            max_input,
            max_output,
            budget_mb: budget_mb(),
            weight_mode: weight_mode(),
        }
    }
}

pub fn gpu_disabled_by_env() -> bool {
    std::env::var("RNB_NO_GPU").is_ok()
}

pub fn output_logits_enabled() -> bool {
    std::env::var("RNB_GPU_OUTPUT").is_ok()
}

pub fn decode_layers_enabled() -> bool {
    // decode batch=1은 CPU NEON이 GPU보다 빠름 (Session 8-13 결론)
    // RNB_GPU_DECODE=1 로 명시적으로 opt-in만 허용
    std::env::var("RNB_GPU_DECODE").is_ok()
}

pub fn decode_layers_allowed() -> bool {
    decode_layers_enabled() && std::env::var("RNB_GPU_LAYERS_OFF").is_err()
}

pub fn max_decode_layer() -> usize {
    std::env::var("RNB_GPU_MAX_LAYERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

pub fn verify_enabled() -> bool {
    std::env::var("RNB_GPU_VERIFY").is_ok()
}

pub fn verify_attention_layer(layer_idx: usize) -> bool {
    verify_enabled() && layer_idx < 24
}

pub fn verify_attention_qkv_layer(layer_idx: usize) -> bool {
    verify_enabled() && layer_idx == 3
}

pub fn verify_gdn_layer(layer_idx: usize) -> bool {
    verify_enabled() && layer_idx == 0
}

pub fn initial_fullpath_staging_tokens(hidden_dim: usize) -> usize {
    prefill_chunk_size(hidden_dim)
}

pub fn prefill_chunk_size_for_active_path(active_vulkan: bool, hidden_dim: usize) -> usize {
    explicit_prefill_chunk_size()
        .or_else(|| {
            if active_vulkan && hidden_dim >= 2048 {
                Some(LARGE_MODEL_PREFILL_CHUNK)
            } else {
                None
            }
        })
        .unwrap_or(usize::MAX)
}

pub fn gdn_qkv_prefill_window_chunk(hidden_dim: usize) -> usize {
    prefill_chunk_size(hidden_dim)
}

pub fn q_proj_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::QProj)
}

pub fn q_bias_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::QBias)
}

pub fn q_norm_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::QNorm)
}

pub fn k_norm_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::KNorm)
}

pub fn k_proj_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::KProj)
}

pub fn k_bias_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::KBias)
}

pub fn v_proj_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::VProj)
}

pub fn v_bias_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::VBias)
}

pub fn o_proj_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::OProj)
}

pub fn ffn_gate_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::FfnGate)
}

pub fn ffn_up_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::FfnUp)
}

pub fn ffn_down_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::FfnDown)
}

pub fn gdn_qkv_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnQkv)
}

pub fn gdn_gate_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnGate)
}

pub fn gdn_alpha_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnAlpha)
}

pub fn gdn_beta_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnBeta)
}

pub fn gdn_ssm_out_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnSsmOut)
}

pub fn gdn_attn_norm_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnAttnNorm)
}

pub fn gdn_post_attn_norm_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnPostAttnNorm)
}

pub fn gdn_ssm_a_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnSsmA)
}

pub fn gdn_ssm_conv1d_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnSsmConv1d)
}

pub fn gdn_ssm_dt_bias_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnSsmDtBias)
}

pub fn gdn_ssm_norm_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::GdnSsmNorm)
}

pub fn attn_norm_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::AttnNorm)
}

pub fn ffn_norm_id(layer_idx: usize) -> WeightId {
    weight_id(layer_idx, WeightKind::FfnNorm)
}

pub fn k_proj_shard_id(layer_idx: usize, kvh: u16) -> WeightId {
    weight_id(layer_idx, WeightKind::KProjShard(kvh))
}

pub fn v_proj_shard_id(layer_idx: usize, kvh: u16) -> WeightId {
    weight_id(layer_idx, WeightKind::VProjShard(kvh))
}

pub fn output_logits_id() -> WeightId {
    WeightId {
        layer: u16::MAX,
        kind: WeightKind::OutputLogits,
    }
}

fn weight_id(layer_idx: usize, kind: WeightKind) -> WeightId {
    WeightId {
        layer: layer_idx as u16,
        kind,
    }
}

pub fn init_layer_gemv(config: LayerGemvConfig) -> Option<LayerGemv> {
    if gpu_disabled_by_env() {
        eprintln!("[vulkan] GPU disabled by RNB_NO_GPU");
        return None;
    }

    match LayerGemv::new(
        config.max_input,
        config.max_output,
        config.budget_mb,
        config.weight_mode,
    ) {
        Ok(mut vg) => {
            if init_self_tests_enabled() {
                eprintln!("[vulkan:init] post_new=q4k_self_test");
                match vg.self_test_q4k() {
                    Ok((gpu, expected, diff)) => {
                        eprintln!(
                            "[vulkan] Q4_K self-test: gpu={:.4}, expected={:.4}, diff={:.6}",
                            gpu, expected, diff
                        );
                        if diff > 0.1 {
                            eprintln!("[vulkan] WARNING: Q4_K shader accuracy issue!");
                        }
                    }
                    Err(e) => eprintln!("[vulkan] Q4_K self-test failed: {}", e),
                }
                // mv31 — verify quantize_to_q8k shader on the actual GPU.
                if std::env::var("RNB_VULKAN_Q8K_ACTIVATION").is_ok()
                    || std::env::var("RNB_VULKAN_Q8K_SELF_TEST").is_ok()
                {
                    eprintln!("[vulkan:init] post_new=quantize_to_q8k_self_test");
                    match vg.self_test_quantize_to_q8k() {
                        Ok(0) => eprintln!("[vulkan] quantize_to_q8k self-test PASSED"),
                        Ok(n) => eprintln!(
                            "[vulkan] quantize_to_q8k self-test FAILED: {n}/69 words mismatch"
                        ),
                        Err(e) => {
                            eprintln!("[vulkan] quantize_to_q8k self-test errored: {e}")
                        }
                    }
                    eprintln!("[vulkan:init] post_new=q4k_q8k_self_test");
                    match vg.self_test_q4k_q8k() {
                        Ok(diff) if diff < 0.001 => {
                            eprintln!("[vulkan] q4k_q8k self-test PASSED (diff={diff:.6})")
                        }
                        Ok(diff) => eprintln!("[vulkan] q4k_q8k self-test FAILED: diff={diff:.6}"),
                        Err(e) => eprintln!("[vulkan] q4k_q8k self-test errored: {e}"),
                    }
                }
            }
            if bench_enabled() {
                eprintln!("[vulkan] Running SoA vs row-major benchmark...");
                for &(rows, cols) in &[(9216, 2560), (2560, 9216), (2560, 2560)] {
                    if let Err(e) = vg.bench_soa_vs_rowmajor(rows, cols, 20) {
                        eprintln!("[vulkan] Bench {}x{} failed: {}", rows, cols, e);
                    }
                }
            }
            if init_self_tests_enabled() {
                eprintln!("[vulkan:init] post_new=elementwise_self_test");
                match vg.self_test_elementwise() {
                    Ok(()) => eprintln!("[vulkan] Elementwise shaders self-test passed"),
                    Err(e) => eprintln!("[vulkan] Elementwise self-test failed: {}", e),
                }
            }
            eprintln!("[vulkan:init] post_new=ready");
            Some(vg)
        }
        Err(e) => {
            eprintln!("[vulkan] GPU init failed, CPU fallback: {}", e);
            None
        }
    }
}

pub fn init_layer_gemv_for_test(
    max_input: usize,
    max_output: usize,
    budget_mb: usize,
) -> Result<LayerGemv, String> {
    LayerGemv::new(max_input, max_output, budget_mb, GpuWeightMode::Soa)
}

/// Map GGUF/GGML quant types to Vulkan runtime quant types for layer weights.
pub fn ggml_to_quant(ggml_type: rnb_loader::GGMLType) -> Option<QuantType> {
    if ggml_type == rnb_loader::GGMLType::F32
        || (ggml_type == rnb_loader::GGMLType::Q5_K && !q5k_enabled())
    {
        None
    } else {
        ggml_to_vulkan_quant(ggml_type)
    }
}
/// Map every quant format implemented by the Vulkan fullpath weight cache.
///
/// Unlike [`ggml_to_quant`], this does not apply the legacy partial-offload
/// `RNB_GPU_Q5K` gate. Fullpath keeps intermediates device-resident, so the
/// old per-window submit/download regression that motivated that gate does
/// not apply.
pub fn ggml_to_fullpath_quant(ggml_type: rnb_loader::GGMLType) -> Option<QuantType> {
    ggml_to_vulkan_quant(ggml_type)
}

pub fn ggml_to_output_quant(ggml_type: rnb_loader::GGMLType) -> Option<QuantType> {
    ggml_to_vulkan_quant(ggml_type)
}

fn ggml_to_vulkan_quant(ggml_type: rnb_loader::GGMLType) -> Option<QuantType> {
    match ggml_type {
        rnb_loader::GGMLType::F32 => Some(QuantType::F32),
        rnb_loader::GGMLType::F16 => Some(QuantType::F16),
        rnb_loader::GGMLType::BF16 => Some(QuantType::BF16),
        rnb_loader::GGMLType::Q4_0 => Some(QuantType::Q4_0),
        rnb_loader::GGMLType::Q4_1 => Some(QuantType::Q4_1),
        rnb_loader::GGMLType::Q5_0 => Some(QuantType::Q5_0),
        rnb_loader::GGMLType::Q5_1 => Some(QuantType::Q5_1),
        rnb_loader::GGMLType::Q8_0 => Some(QuantType::Q8_0),
        rnb_loader::GGMLType::Q8_1 => Some(QuantType::Q8_1),
        rnb_loader::GGMLType::Q2_K => Some(QuantType::Q2K),
        rnb_loader::GGMLType::Q3_K => Some(QuantType::Q3K),
        rnb_loader::GGMLType::Q4_K => Some(QuantType::Q4K),
        rnb_loader::GGMLType::Q5_K => Some(QuantType::Q5K),
        rnb_loader::GGMLType::Q6_K => Some(QuantType::Q6K),
        rnb_loader::GGMLType::Q8_K => Some(QuantType::Q8K),
        rnb_loader::GGMLType::IQ2_XXS => Some(QuantType::IQ2_XXS),
        rnb_loader::GGMLType::IQ2_XS => Some(QuantType::IQ2_XS),
        rnb_loader::GGMLType::IQ3_XXS => Some(QuantType::IQ3_XXS),
        rnb_loader::GGMLType::IQ1_S => Some(QuantType::IQ1_S),
        rnb_loader::GGMLType::IQ4_NL => Some(QuantType::IQ4_NL),
        rnb_loader::GGMLType::IQ3_S => Some(QuantType::IQ3_S),
        rnb_loader::GGMLType::IQ2_S => Some(QuantType::IQ2_S),
        rnb_loader::GGMLType::IQ4_XS => Some(QuantType::IQ4_XS),
        rnb_loader::GGMLType::IQ1_M => Some(QuantType::IQ1_M),
        rnb_loader::GGMLType::TQ1_0 => Some(QuantType::TQ1_0),
        rnb_loader::GGMLType::TQ2_0 => Some(QuantType::TQ2_0),
        rnb_loader::GGMLType::MXFP4 => Some(QuantType::MXFP4),
        rnb_loader::GGMLType::NVFP4 => Some(QuantType::NVFP4),
        rnb_loader::GGMLType::Q1_0 => Some(QuantType::Q1_0),
        rnb_loader::GGMLType::Q2_0 => Some(QuantType::Q2_0),
        rnb_loader::GGMLType::I8
        | rnb_loader::GGMLType::I16
        | rnb_loader::GGMLType::I32
        | rnb_loader::GGMLType::I64
        | rnb_loader::GGMLType::F64 => None,
    }
}

fn q5k_enabled() -> bool {
    std::env::var("RNB_GPU_Q5K").is_ok()
}

fn prefill_chunk_size(hidden_dim: usize) -> usize {
    explicit_prefill_chunk_size().unwrap_or(if hidden_dim >= 2048 {
        LARGE_MODEL_PREFILL_CHUNK
    } else {
        SMALL_MODEL_PREFILL_CHUNK
    })
}

fn explicit_prefill_chunk_size() -> Option<usize> {
    std::env::var("RNB_PREFILL_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.max(1))
}

// mv40: input_capacity 의 attention batch 하한을 결정하는 상수.
// max_input >= hidden_dim × attention_batch_floor() 보장.
// 산식: q_dim ≈ hidden_dim × 2 (GQA Q-only 보수), target batch = 64
//   → multiplier = 2 × 64 = 128.
// env override 로 작은 장비에서 buffer 절약 / 큰 장비에서 더 큰 batch 가능.
fn attention_batch_floor() -> usize {
    std::env::var("RNB_VULKAN_ATTN_BATCH_FLOOR")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128)
}

fn budget_mb() -> usize {
    std::env::var("RNB_GPU_BUDGET_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

fn weight_mode_for_platform(
    force_soa: bool,
    force_row_major: bool,
    is_android: bool,
) -> GpuWeightMode {
    if force_soa || (is_android && !force_row_major) {
        GpuWeightMode::Soa
    } else {
        GpuWeightMode::RowMajor
    }
}

fn weight_mode() -> GpuWeightMode {
    weight_mode_for_platform(
        std::env::var_os("RNB_GPU_SOA").is_some(),
        std::env::var_os("RNB_GPU_ROW_MAJOR").is_some(),
        cfg!(target_os = "android"),
    )
}

fn bench_enabled() -> bool {
    std::env::var("RNB_GPU_BENCH").is_ok()
}

fn init_self_tests_enabled() -> bool {
    std::env::var("RNB_VULKAN_INIT_SELF_TESTS").is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_hidden_config_supports_gdn_prefill_window_chunk() {
        std::env::remove_var("RNB_PREFILL_CHUNK_SIZE");

        let config = LayerGemvConfig::from_model_shape(1024, 2816, 4096, 151_936);

        assert!(
            config.max_input >= 1024 * 32,
            "max_input={} must fit the 32-token GDN prefill window",
            config.max_input
        );
    }

    #[test]
    fn fullpath_quant_mapping_includes_q5_without_legacy_gate() {
        assert_eq!(
            ggml_to_fullpath_quant(rnb_loader::GGMLType::Q4_K),
            Some(QuantType::Q4K)
        );
        assert_eq!(
            ggml_to_fullpath_quant(rnb_loader::GGMLType::Q5_K),
            Some(QuantType::Q5K)
        );
        assert_eq!(
            ggml_to_fullpath_quant(rnb_loader::GGMLType::Q6_K),
            Some(QuantType::Q6K)
        );
        assert_eq!(
            ggml_to_fullpath_quant(rnb_loader::GGMLType::Q8_0),
            Some(QuantType::Q8_0)
        );
    }

    #[test]
    fn desktop_defaults_to_adaptive_row_major_weights() {
        assert_eq!(
            weight_mode_for_platform(false, false, false),
            GpuWeightMode::RowMajor
        );
        assert_eq!(
            weight_mode_for_platform(true, true, false),
            GpuWeightMode::Soa
        );
    }

    #[test]
    fn android_keeps_soa_unless_row_major_is_explicit() {
        assert_eq!(
            weight_mode_for_platform(false, false, true),
            GpuWeightMode::Soa
        );
        assert_eq!(
            weight_mode_for_platform(false, true, true),
            GpuWeightMode::RowMajor
        );
    }

    #[test]
    fn fullpath_quant_mapping_includes_importance_quants() {
        for (ggml_type, expected) in [
            (rnb_loader::GGMLType::IQ2_XXS, QuantType::IQ2_XXS),
            (rnb_loader::GGMLType::IQ2_XS, QuantType::IQ2_XS),
            (rnb_loader::GGMLType::IQ3_XXS, QuantType::IQ3_XXS),
            (rnb_loader::GGMLType::IQ1_S, QuantType::IQ1_S),
            (rnb_loader::GGMLType::IQ3_S, QuantType::IQ3_S),
            (rnb_loader::GGMLType::IQ2_S, QuantType::IQ2_S),
            (rnb_loader::GGMLType::IQ4_XS, QuantType::IQ4_XS),
            (rnb_loader::GGMLType::IQ1_M, QuantType::IQ1_M),
        ] {
            assert_eq!(ggml_to_fullpath_quant(ggml_type), Some(expected));
        }
    }
}
