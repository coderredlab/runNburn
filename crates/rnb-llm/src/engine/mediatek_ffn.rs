use super::layer_weights::LayerType;
use super::norm::apply_model_gate_mul_inplace;
use super::quantized_dispatch::prefill_gate_up_vectors;
use super::quantized_weight_types::QuantizedWeight;
use crate::error::LlmError;
use rnb_loader::Architecture as ModelArchitecture;

const DEVICE_ENV: &str = "RNB_MEDIATEK_DEVICE";
const ENABLE_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN";
const LAYER_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_LAYER";
const PARITY_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_PARITY";
const COMPILED_REUSE_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_COMPILED_REUSE";
const QUANTIZED_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_QUANTIZED";
const QUANTIZED_STAGE_PROBE_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_QUANTIZED_STAGE_PROBE";
const PREFILL_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_PREFILL";
const TRACE_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_TRACE";
const LAYERS_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_LAYERS";
const CACHE_MAX_LAYERS_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS";
const DEFAULT_CACHE_MAX_LAYERS: usize = 2;
const HARD_MAX_CACHE_LAYERS: usize = 4;
const DEFAULT_DEVICE: &str = "mtk-neuron";
const PREFILL_BATCH_SIZE: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MediaTekPrefillBatchPlan {
    pub batch_size: usize,
    pub full_chunks: usize,
    pub tail_tokens: usize,
}

pub(super) fn prefill_batch_plan(seq_len: usize) -> Option<MediaTekPrefillBatchPlan> {
    if seq_len < PREFILL_BATCH_SIZE {
        return None;
    }
    Some(MediaTekPrefillBatchPlan {
        batch_size: PREFILL_BATCH_SIZE,
        full_chunks: seq_len / PREFILL_BATCH_SIZE,
        tail_tokens: seq_len % PREFILL_BATCH_SIZE,
    })
}
pub(super) const fn prefill_needs_host_weights(runtime_cache_hit: bool, parity: bool) -> bool {
    !runtime_cache_hit || parity
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MediaTekFfnRejectReason {
    InvalidDevice,
    AllLayersRequireCompiledCache,
    LayerListRequiresCompiledReuse,
    InvalidLayerList,
    LayerListExceedsCacheCap,
    DumpModeActive,
    NonGemmaArchitecture,
    FusedGateUpUnhandled,
    ShapeMismatch,
    MaterializationFailed,
    RuntimeUnavailable,
    ParityFailed,
}

impl MediaTekFfnRejectReason {
    pub(super) const fn trace_reason(self) -> &'static str {
        match self {
            Self::InvalidDevice => "invalid_device",
            Self::AllLayersRequireCompiledCache => "all_layers_require_compiled_cache",
            Self::LayerListRequiresCompiledReuse => "layer_list_requires_compiled_reuse",
            Self::InvalidLayerList => "invalid_layer_list",
            Self::LayerListExceedsCacheCap => "layer_list_exceeds_cache_cap",
            Self::DumpModeActive => "dump_mode_active",
            Self::NonGemmaArchitecture => "non_gemma_architecture",
            Self::FusedGateUpUnhandled => "fused_gate_up_unhandled",
            Self::ShapeMismatch => "shape_mismatch",
            Self::MaterializationFailed => "materialization_failed",
            Self::RuntimeUnavailable => "runtime_unavailable",
            Self::ParityFailed => "parity_failed",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MediaTekFfnParityStats {
    pub max_abs_error: f32,
    pub max_rel_error: f32,
    pub cosine_similarity: f32,
}

impl MediaTekFfnParityStats {
    pub(super) fn passed(self) -> bool {
        (self.max_abs_error <= 1.0e-4 || self.max_rel_error <= 1.0e-3)
            && self.cosine_similarity >= 0.999
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MediaTekFfnLayerDecision {
    pub selected: bool,
    pub max_cache_entries: usize,
}

fn cache_max_layers_from_env_value_with_hard_max(
    value: Option<&str>,
    hard_max: usize,
) -> (usize, bool) {
    match value {
        None => (DEFAULT_CACHE_MAX_LAYERS, false),
        Some(value) => match value.parse::<usize>() {
            Ok(value) if (1..=hard_max).contains(&value) => (value, false),
            _ => (DEFAULT_CACHE_MAX_LAYERS, true),
        },
    }
}

pub(super) fn cache_max_layers_from_env_value(value: Option<&str>) -> (usize, bool) {
    cache_max_layers_from_env_value_with_hard_max(value, HARD_MAX_CACHE_LAYERS)
}

fn trace_invalid_cache_cap_once(raw_value: &str, hard_max: usize) {
    static INVALID_CACHE_CAP_WARNED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if !INVALID_CACHE_CAP_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "[mediatek-ffn] policy_warning=invalid_cache_cap value={raw_value:?} default={DEFAULT_CACHE_MAX_LAYERS} hard_max={hard_max}"
        );
    }
}

pub(super) fn cache_max_layers() -> usize {
    let raw = std::env::var(CACHE_MAX_LAYERS_ENV).ok();
    let (max_entries, invalid_override) = cache_max_layers_from_env_value(raw.as_deref());
    if invalid_override {
        trace_invalid_cache_cap_once(raw.as_deref().unwrap_or(""), HARD_MAX_CACHE_LAYERS);
    }
    max_entries
}

fn parse_layer_list(value: &str) -> Result<Vec<usize>, MediaTekFfnRejectReason> {
    let mut layers = Vec::new();
    for token in value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            return Err(MediaTekFfnRejectReason::InvalidLayerList);
        }
        let layer = token
            .parse::<usize>()
            .map_err(|_| MediaTekFfnRejectReason::InvalidLayerList)?;
        if layers.contains(&layer) {
            return Err(MediaTekFfnRejectReason::InvalidLayerList);
        }
        layers.push(layer);
    }
    if layers.is_empty() {
        return Err(MediaTekFfnRejectReason::InvalidLayerList);
    }
    Ok(layers)
}

fn selected_layer_decision_with_cap(
    layer_idx: usize,
    compiled_reuse: bool,
    max_cache_entries: usize,
) -> Result<MediaTekFfnLayerDecision, MediaTekFfnRejectReason> {
    if let Ok(value) = std::env::var(LAYERS_ENV) {
        if !compiled_reuse {
            return Err(MediaTekFfnRejectReason::LayerListRequiresCompiledReuse);
        }
        let layers = parse_layer_list(&value)?;
        if layers.len() > max_cache_entries {
            return Err(MediaTekFfnRejectReason::LayerListExceedsCacheCap);
        }
        return Ok(MediaTekFfnLayerDecision {
            selected: layers.contains(&layer_idx),
            max_cache_entries,
        });
    }

    let selected = match std::env::var(LAYER_ENV) {
        Ok(value) if value.trim() == "all" => {
            return Err(MediaTekFfnRejectReason::AllLayersRequireCompiledCache);
        }
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(selected) => selected == layer_idx,
            Err(_) => false,
        },
        Err(_) => layer_idx == 0,
    };
    Ok(MediaTekFfnLayerDecision {
        selected,
        max_cache_entries,
    })
}

pub(super) fn selected_layer_decision(
    layer_idx: usize,
    compiled_reuse: bool,
) -> Result<MediaTekFfnLayerDecision, MediaTekFfnRejectReason> {
    selected_layer_decision_with_cap(layer_idx, compiled_reuse, cache_max_layers())
}

pub(super) fn decide_prefill_npu(
    layer_idx: usize,
    seq_len: usize,
    cache_state: rnb_runtime::mediatek::MediaTekPrefillCacheState,
    request_mode: rnb_runtime::mediatek::MediaTekPrefillRequestMode,
    sync_compile_enabled: bool,
) -> rnb_runtime::mediatek::MediaTekPrefillNpuDecision {
    rnb_runtime::mediatek::decide_gemma_prefill_npu(
        rnb_runtime::mediatek::MediaTekPrefillPolicyInput {
            layer_idx,
            seq_len,
            cache_state,
            request_mode,
            sync_compile_enabled,
        },
    )
}

pub(super) fn decide_prefill_npu_for_cache_presence(
    layer_idx: usize,
    seq_len: usize,
    in_process_thread_local_hot: bool,
    disk_aot_present: bool,
    request_mode: rnb_runtime::mediatek::MediaTekPrefillRequestMode,
    sync_compile_enabled: bool,
) -> rnb_runtime::mediatek::MediaTekPrefillNpuDecision {
    let cache_state = rnb_runtime::mediatek::classify_gemma_prefill_cache_state(
        in_process_thread_local_hot,
        disk_aot_present,
    );
    decide_prefill_npu(
        layer_idx,
        seq_len,
        cache_state,
        request_mode,
        sync_compile_enabled,
    )
}

pub(super) fn selected_layer_matches(layer_idx: usize) -> Result<bool, MediaTekFfnRejectReason> {
    selected_layer_decision(layer_idx, compiled_reuse_enabled()).map(|decision| decision.selected)
}

pub(super) fn resolve_device_name() -> Result<String, MediaTekFfnRejectReason> {
    match std::env::var(DEVICE_ENV) {
        Ok(value) if value.trim().is_empty() => Err(MediaTekFfnRejectReason::InvalidDevice),
        Ok(value) => Ok(value.trim().to_string()),
        Err(_) => Ok(DEFAULT_DEVICE.to_string()),
    }
}

pub(super) fn opt_in_enabled() -> bool {
    std::env::var(ENABLE_ENV)
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn parity_required() -> bool {
    std::env::var(PARITY_ENV)
        .map(|value| value != "0")
        .unwrap_or(false)
}

pub(super) fn compiled_reuse_enabled() -> bool {
    std::env::var(COMPILED_REUSE_ENV)
        .map(|value| value != "0")
        .unwrap_or(true)
}

pub(super) fn quantized_enabled() -> bool {
    std::env::var(QUANTIZED_ENV)
        .map(|value| value == "1")
        .unwrap_or(false)
}

pub(super) fn quantized_stage_probe_enabled() -> bool {
    std::env::var(QUANTIZED_STAGE_PROBE_ENV)
        .map(|value| value == "1")
        .unwrap_or(false)
}

pub(super) fn prefill_enabled() -> bool {
    std::env::var(PREFILL_ENV)
        .map(|value| value == "1")
        .unwrap_or(false)
}

pub(super) fn trace_enabled() -> bool {
    std::env::var(TRACE_ENV)
        .map(|value| value == "1")
        .unwrap_or(false)
}

pub(super) const fn mediatek_ffn_measure_timing(trace: bool, parity: bool) -> bool {
    trace || parity
}

fn trace_fallback(layer_idx: usize, reason: MediaTekFfnRejectReason) {
    eprintln!(
        "[mediatek-ffn] used=false layer={layer_idx} reason={}",
        reason.trace_reason()
    );
}

fn trace_prefill_policy_fallback(
    layer_idx: usize,
    reason: rnb_runtime::mediatek::MediaTekPrefillFallbackReason,
) {
    if reason == rnb_runtime::mediatek::MediaTekPrefillFallbackReason::LayerNotSelected
        || !trace_enabled()
    {
        return;
    }
    eprintln!(
        "[mediatek-ffn] used=false layer={layer_idx} reason={}",
        reason.trace_reason()
    );
}

fn is_gemma_dense(architecture: ModelArchitecture) -> bool {
    matches!(
        architecture,
        ModelArchitecture::Gemma | ModelArchitecture::Gemma4
    )
}

pub(super) fn parity_stats(
    reference: &[f32],
    candidate: &[f32],
) -> Result<MediaTekFfnParityStats, MediaTekFfnRejectReason> {
    if reference.len() != candidate.len() {
        return Err(MediaTekFfnRejectReason::ShapeMismatch);
    }
    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut dot = 0.0f64;
    let mut ref_norm = 0.0f64;
    let mut cand_norm = 0.0f64;
    for (&lhs, &rhs) in reference.iter().zip(candidate.iter()) {
        if !lhs.is_finite() || !rhs.is_finite() {
            return Err(MediaTekFfnRejectReason::ParityFailed);
        }
        let abs = (lhs - rhs).abs();
        max_abs_error = max_abs_error.max(abs);
        max_rel_error = max_rel_error.max(abs / lhs.abs().max(1.0e-6));
        dot += (lhs as f64) * (rhs as f64);
        ref_norm += (lhs as f64) * (lhs as f64);
        cand_norm += (rhs as f64) * (rhs as f64);
    }
    let cosine_similarity = if ref_norm == 0.0 && cand_norm == 0.0 {
        1.0
    } else if ref_norm == 0.0 || cand_norm == 0.0 {
        0.0
    } else {
        (dot / (ref_norm.sqrt() * cand_norm.sqrt())) as f32
    };
    Ok(MediaTekFfnParityStats {
        max_abs_error,
        max_rel_error,
        cosine_similarity,
    })
}

pub(super) fn stage_probe_line(
    layer_idx: usize,
    stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage,
    nnapi_vs_cpu_quant: MediaTekFfnParityStats,
    cpu_quant_vs_f32: MediaTekFfnParityStats,
    nnapi_vs_f32: Option<MediaTekFfnParityStats>,
) -> String {
    let mut line = format!(
        "[mediatek-ffn-stage] layer={layer_idx} stage={} nnapi_vs_cpu_quant_max_abs={:.9} nnapi_vs_cpu_quant_max_rel={:.9} nnapi_vs_cpu_quant_cosine={:.9} cpu_quant_vs_f32_max_abs={:.9} cpu_quant_vs_f32_max_rel={:.9} cpu_quant_vs_f32_cosine={:.9}",
        stage.name(),
        nnapi_vs_cpu_quant.max_abs_error,
        nnapi_vs_cpu_quant.max_rel_error,
        nnapi_vs_cpu_quant.cosine_similarity,
        cpu_quant_vs_f32.max_abs_error,
        cpu_quant_vs_f32.max_rel_error,
        cpu_quant_vs_f32.cosine_similarity,
    );
    if let Some(nnapi_vs_f32) = nnapi_vs_f32 {
        line.push_str(&format!(
            " nnapi_vs_f32_max_abs={:.9} nnapi_vs_f32_max_rel={:.9} nnapi_vs_f32_cosine={:.9}",
            nnapi_vs_f32.max_abs_error, nnapi_vs_f32.max_rel_error, nnapi_vs_f32.cosine_similarity,
        ));
    }
    line
}

fn stage_probe_invalidation_line(
    layer_idx: usize,
    stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage,
    error: &str,
) -> String {
    format!(
        "[mediatek-ffn-stage] layer={layer_idx} stage={} invalidated error={error}",
        stage.name()
    )
}
pub(super) fn parity_pass_line(layer_idx: usize, stats: MediaTekFfnParityStats) -> String {
    format!(
        "[mediatek-ffn] parity=pass layer={layer_idx} max_abs_error={:.9} max_rel_error={:.9} cosine_similarity={:.9}",
        stats.max_abs_error, stats.max_rel_error, stats.cosine_similarity
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MediaTekFfnLocalTimings {
    pub materialize_ns: u64,
    pub runtime_total_ns: u64,
    pub parity_ns: Option<u64>,
    pub cache_hit: bool,
    pub total_ns: u64,
}

fn elapsed_ns(start: std::time::Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn optional_ns_label(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string())
}

pub(super) fn timing_line(
    layer_idx: usize,
    local: MediaTekFfnLocalTimings,
    backend: rnb_runtime::mediatek::RunGatedGeluFfnF32Timings,
    duration_hardware_ns: Option<u64>,
    duration_driver_ns: Option<u64>,
) -> String {
    format!(
        "[mediatek-ffn] timing layer={layer_idx} materialize_ns={} runtime_total_ns={} cache_hit={} backend_model_build_ns={} backend_supported_ops_query_ns={} backend_compilation_ns={} backend_execution_setup_ns={} backend_execution_compute_ns={} nnapi_hw_ns={} nnapi_driver_ns={} parity_ns={} total_ns={}",
        local.materialize_ns,
        local.runtime_total_ns,
        local.cache_hit,
        backend.model_build_ns,
        backend.supported_ops_query_ns,
        backend.compilation_ns,
        backend.execution_setup_ns,
        backend.execution_compute_ns,
        optional_ns_label(duration_hardware_ns),
        optional_ns_label(duration_driver_ns),
        optional_ns_label(local.parity_ns),
        local.total_ns
    )
}

pub(super) struct QuantizedGatedGeluAdmission {
    weights: rnb_runtime::mediatek::RunCachedQuantizedGatedGeluFfnWeights,
    reference: Vec<f32>,
    pub(super) stage_references: Option<Vec<QuantizedGatedGeluStageReference>>,
    pub(super) splits: Option<Vec<QuantizedGatedGeluSplit>>,
}

#[derive(Debug, Clone)]
pub(super) struct QuantizedGatedGeluStageReference {
    pub(super) stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage,
    pub(super) f32_reference: Vec<f32>,
    pub(super) cpu_quantized_reference: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct QuantizedGatedGeluSplit {
    pub(super) stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage,
    pub(super) f32_reference: Vec<f32>,
    pub(super) full_w8a8: Vec<f32>,
    pub(super) act_only: Vec<f32>,
    pub(super) weight_only: Vec<f32>,
    pub(super) output_requant_only: Vec<f32>,
}

pub(super) fn build_quantized_gated_gelu_admission(
    _architecture: ModelArchitecture,
    norm_data: &[f32],
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
    hidden_dim: usize,
    ffn_inner: usize,
    stage_probe: bool,
) -> crate::error::Result<QuantizedGatedGeluAdmission> {
    let gate_projection = matvec_rows(
        &gate_weight,
        norm_data,
        ffn_inner,
        hidden_dim,
        "gate_weight",
    )?;
    let up_projection = matvec_rows(&up_weight, norm_data, ffn_inner, hidden_dim, "up_weight")?;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut square = Vec::with_capacity(ffn_inner);
    let mut cube = Vec::with_capacity(ffn_inner);
    let mut poly_scale = Vec::with_capacity(ffn_inner);
    let mut poly = Vec::with_capacity(ffn_inner);
    let mut tanh_arg = Vec::with_capacity(ffn_inner);
    let mut tanh_output = Vec::with_capacity(ffn_inner);
    let mut one_plus = Vec::with_capacity(ffn_inner);
    let mut gelu_factor = Vec::with_capacity(ffn_inner);
    let mut gelu = Vec::with_capacity(ffn_inner);
    let mut gated = Vec::with_capacity(ffn_inner);
    for index in 0..ffn_inner {
        let gate = gate_projection[index];
        let gate_square = gate * gate;
        let gate_cube = gate_square * gate;
        let scaled_cube = 0.044715 * gate_cube;
        let polynomial = gate + scaled_cube;
        let tanh_input = sqrt_2_over_pi * polynomial;
        let tanh_value = tanh_input.tanh();
        let one_plus_tanh = 1.0 + tanh_value;
        let gate_times_one_plus = gate * one_plus_tanh;
        let gelu_value = 0.5 * gate_times_one_plus;
        square.push(gate_square);
        cube.push(gate_cube);
        poly_scale.push(scaled_cube);
        poly.push(polynomial);
        tanh_arg.push(tanh_input);
        tanh_output.push(tanh_value);
        one_plus.push(one_plus_tanh);
        gelu_factor.push(gate_times_one_plus);
        gelu.push(gelu_value);
        gated.push(gelu_value * up_projection[index]);
    }
    let reference = matmul_down(&down_weight, &gated, hidden_dim, ffn_inner)?;
    let input_quant_params = quant_params_for_quantized_gated_gelu(&[norm_data]);
    let gate_weight_quant_params = quant_params_for_quantized_gated_gelu_weight(&gate_weight);
    let up_weight_quant_params = quant_params_for_quantized_gated_gelu_weight(&up_weight);
    let gate_weight_quantized = quantize_f32_to_u8(&gate_weight, gate_weight_quant_params);
    let up_weight_quantized = quantize_f32_to_u8(&up_weight, up_weight_quant_params);
    let quant_params = rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnQuantParams::new(
        input_quant_params,
        gate_weight_quant_params,
        quant_params_for_quantized_gated_gelu(&[&gate_projection]),
        up_weight_quant_params,
        quant_params_for_quantized_gated_gelu(&[&up_projection]),
        quant_params_for_nonnegative_quantized_gated_gelu(&[&square]),
        quant_params_for_quantized_gated_gelu(&[&cube]),
        quant_params_for_nonnegative_scalar(0.044715),
        quant_params_for_quantized_gated_gelu(&[&poly_scale]),
        quant_params_for_quantized_gated_gelu(&[&poly]),
        quant_params_for_nonnegative_scalar(sqrt_2_over_pi),
        quant_params_for_quantized_gated_gelu(&[&tanh_arg]),
        rnb_runtime::mediatek::MediaTekQuantParams::new(1.0 / 128.0, 128),
        rnb_runtime::mediatek::MediaTekQuantParams::new(1.0 / 128.0, 0),
        quant_params_for_nonnegative_quantized_gated_gelu(&[&one_plus]),
        quant_params_for_quantized_gated_gelu(&[&gelu_factor]),
        rnb_runtime::mediatek::MediaTekQuantParams::new(1.0 / 255.0, 0),
        quant_params_for_quantized_gated_gelu(&[&gelu]),
        quant_params_for_quantized_gated_gelu(&[&gated]),
        quant_params_for_symmetric_quantized_gated_gelu(&[&down_weight]),
        quant_params_for_quantized_gated_gelu(&[&reference]),
    );
    let down_weight_quantized = quantize_f32_to_u8(&down_weight, quant_params.down_weight);
    let stage_references = if stage_probe {
        Some(build_quantized_gated_gelu_stage_references(
            norm_data,
            &gate_weight_quantized,
            &up_weight_quantized,
            quant_params,
            &gate_projection,
            &up_projection,
            &gelu,
            &gated,
            &reference,
            &down_weight,
            hidden_dim,
            ffn_inner,
        )?)
    } else {
        None
    };
    let splits = if stage_probe {
        build_quantized_gated_gelu_act_weight_splits(
            norm_data,
            &gate_weight,
            &up_weight,
            &gate_weight_quantized,
            &up_weight_quantized,
            quant_params,
            &gate_projection,
            &up_projection,
            hidden_dim,
            ffn_inner,
        )
    } else {
        None
    };
    let weights = rnb_runtime::mediatek::RunCachedQuantizedGatedGeluFfnWeights {
        quant_params,
        gate_weight: gate_weight_quantized,
        up_weight: up_weight_quantized,
        down_weight: down_weight_quantized,
        down_weight_f32: Some(down_weight),
    };
    Ok(QuantizedGatedGeluAdmission {
        weights,
        reference,
        stage_references,
        splits,
    })
}

pub(super) fn quant_params_for_quantized_gated_gelu(
    tensors: &[&[f32]],
) -> rnb_runtime::mediatek::MediaTekQuantParams {
    let mut min = 0.0f32;
    let mut max = 0.0f32;
    let mut saw_finite = false;
    for tensor in tensors {
        for value in *tensor {
            if value.is_finite() {
                min = min.min(*value);
                max = max.max(*value);
                saw_finite = true;
            }
        }
    }
    if !saw_finite || max <= min {
        return rnb_runtime::mediatek::MediaTekQuantParams::new(1.0e-6, 0);
    }
    let scale = ((max - min) / 255.0).max(1.0e-6);
    let zero_point = (-min / scale).round().clamp(0.0, 255.0) as i32;
    rnb_runtime::mediatek::MediaTekQuantParams::new(scale, zero_point)
}

pub(super) fn quant_params_for_quantized_gated_gelu_weight(
    tensor: &[f32],
) -> rnb_runtime::mediatek::MediaTekQuantParams {
    quant_params_for_quantized_gated_gelu(&[tensor])
}

fn quant_params_for_symmetric_quantized_gated_gelu(
    tensors: &[&[f32]],
) -> rnb_runtime::mediatek::MediaTekQuantParams {
    let mut max_abs = 0.0f32;
    for tensor in tensors {
        for value in *tensor {
            if value.is_finite() {
                max_abs = max_abs.max(value.abs());
            }
        }
    }
    let scale = (max_abs / 127.0).max(1.0e-6);
    rnb_runtime::mediatek::MediaTekQuantParams::new(scale, 128)
}

fn quant_params_for_nonnegative_quantized_gated_gelu(
    tensors: &[&[f32]],
) -> rnb_runtime::mediatek::MediaTekQuantParams {
    let mut max = 1.0f32;
    for tensor in tensors {
        for value in *tensor {
            if value.is_finite() {
                max = max.max(*value);
            }
        }
    }
    let scale = (max / 255.0).max(1.0e-6);
    rnb_runtime::mediatek::MediaTekQuantParams::new(scale, 0)
}

fn quant_params_for_nonnegative_scalar(value: f32) -> rnb_runtime::mediatek::MediaTekQuantParams {
    let scale = (value.abs() / 127.0).max(1.0e-6);
    rnb_runtime::mediatek::MediaTekQuantParams::new(scale, 0)
}
fn quantize_f32_to_u8(
    values: &[f32],
    params: rnb_runtime::mediatek::MediaTekQuantParams,
) -> Vec<u8> {
    values
        .iter()
        .map(|value| {
            let quantized = (*value / params.scale()) + params.zero_point() as f32;
            quantized.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn dequantize_u8(value: u8, params: rnb_runtime::mediatek::MediaTekQuantParams) -> f32 {
    (i32::from(value) - params.zero_point()) as f32 * params.scale()
}

pub(super) fn quantized_fc_output_reference(
    input: &[u8],
    input_params: rnb_runtime::mediatek::MediaTekQuantParams,
    weight: &[u8],
    weight_params: rnb_runtime::mediatek::MediaTekQuantParams,
    output_params: rnb_runtime::mediatek::MediaTekQuantParams,
    rows: usize,
    cols: usize,
    label: &'static str,
) -> crate::error::Result<Vec<f32>> {
    if input.len() != cols {
        return Err(LlmError::Forward(format!(
            "{label} input length mismatch: expected {cols}, got {}",
            input.len()
        )));
    }
    if weight.len() != rows * cols {
        return Err(LlmError::Forward(format!(
            "{label} weight length mismatch: expected {}, got {}",
            rows * cols,
            weight.len()
        )));
    }
    let requant_scale = input_params.scale() * weight_params.scale() / output_params.scale();
    let mut output = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_slice = &weight[row * cols..(row + 1) * cols];
        let mut accumulator = 0i32;
        for col in 0..cols {
            accumulator += (i32::from(input[col]) - input_params.zero_point())
                * (i32::from(row_slice[col]) - weight_params.zero_point());
        }
        let quantized = accumulator as f32 * requant_scale + output_params.zero_point() as f32;
        let quantized = quantized.round().clamp(0.0, 255.0) as u8;
        output.push(dequantize_u8(quantized, output_params));
    }
    Ok(output)
}

fn gelu_and_gated_from_projections(
    gate_projection: &[f32],
    up_projection: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let mut gelu = Vec::with_capacity(gate_projection.len());
    let mut gated = Vec::with_capacity(gate_projection.len());
    for (&gate, &up) in gate_projection.iter().zip(up_projection.iter()) {
        let gate_square = gate * gate;
        let gate_cube = gate_square * gate;
        let tanh_input = sqrt_2_over_pi * (gate + 0.044715 * gate_cube);
        let gelu_value = 0.5 * gate * (1.0 + tanh_input.tanh());
        gelu.push(gelu_value);
        gated.push(gelu_value * up);
    }
    (gelu, gated)
}

fn build_quantized_gated_gelu_stage_references(
    norm_data: &[f32],
    gate_weight_quantized: &[u8],
    up_weight_quantized: &[u8],
    quant_params: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnQuantParams,
    gate_projection: &[f32],
    up_projection: &[f32],
    gelu: &[f32],
    gated: &[f32],
    reference: &[f32],
    down_weight_f32: &[f32],
    hidden_dim: usize,
    ffn_inner: usize,
) -> crate::error::Result<Vec<QuantizedGatedGeluStageReference>> {
    let input_quantized = quantize_f32_to_u8(norm_data, quant_params.input);
    let gate_fc = quantized_fc_output_reference(
        &input_quantized,
        quant_params.input,
        gate_weight_quantized,
        quant_params.gate_weight,
        quant_params.gate,
        ffn_inner,
        hidden_dim,
        "gate_fc",
    )?;
    let up_fc = quantized_fc_output_reference(
        &input_quantized,
        quant_params.input,
        up_weight_quantized,
        quant_params.up_weight,
        quant_params.up,
        ffn_inner,
        hidden_dim,
        "up_fc",
    )?;
    let (gelu_quantized, gated_quantized) = gelu_and_gated_from_projections(&gate_fc, &up_fc);
    let output_quantized = matmul_down(down_weight_f32, &gated_quantized, hidden_dim, ffn_inner)?;
    Ok(vec![
        QuantizedGatedGeluStageReference {
            stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::GateFc,
            f32_reference: gate_projection.to_vec(),
            cpu_quantized_reference: gate_fc.clone(),
        },
        QuantizedGatedGeluStageReference {
            stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::UpFc,
            f32_reference: up_projection.to_vec(),
            cpu_quantized_reference: up_fc.clone(),
        },
        QuantizedGatedGeluStageReference {
            stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::GateDequant,
            f32_reference: gate_projection.to_vec(),
            cpu_quantized_reference: gate_fc,
        },
        QuantizedGatedGeluStageReference {
            stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::UpDequant,
            f32_reference: up_projection.to_vec(),
            cpu_quantized_reference: up_fc,
        },
        QuantizedGatedGeluStageReference {
            stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::Gelu,
            f32_reference: gelu.to_vec(),
            cpu_quantized_reference: gelu_quantized.clone(),
        },
        QuantizedGatedGeluStageReference {
            stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::Gated,
            f32_reference: gated.to_vec(),
            cpu_quantized_reference: gated_quantized.clone(),
        },
        QuantizedGatedGeluStageReference {
            stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::Output,
            f32_reference: reference.to_vec(),
            cpu_quantized_reference: output_quantized,
        },
    ])
}

fn dequantize_u8_vec(
    values: &[u8],
    params: rnb_runtime::mediatek::MediaTekQuantParams,
) -> Vec<f32> {
    values.iter().map(|&v| dequantize_u8(v, params)).collect()
}

fn requantize_f32_through_u8(
    value: f32,
    params: rnb_runtime::mediatek::MediaTekQuantParams,
) -> f32 {
    let q = ((value / params.scale()) + params.zero_point() as f32)
        .round()
        .clamp(0.0, 255.0) as u8;
    dequantize_u8(q, params)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_quantized_gated_gelu_act_weight_splits(
    norm_data: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    gate_weight_quantized: &[u8],
    up_weight_quantized: &[u8],
    quant_params: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnQuantParams,
    gate_projection: &[f32],
    up_projection: &[f32],
    hidden_dim: usize,
    ffn_inner: usize,
) -> Option<Vec<QuantizedGatedGeluSplit>> {
    let build_one = |stage: rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage,
                     weight_f32: &[f32],
                     weight_q8: &[u8],
                     weight_params: rnb_runtime::mediatek::MediaTekQuantParams,
                     out_params: rnb_runtime::mediatek::MediaTekQuantParams,
                     projection: &[f32]|
     -> crate::error::Result<QuantizedGatedGeluSplit> {
        let input_q8 = quantize_f32_to_u8(norm_data, quant_params.input);
        let dequant_input = dequantize_u8_vec(&input_q8, quant_params.input);
        let act_only = matvec_rows(
            weight_f32,
            &dequant_input,
            ffn_inner,
            hidden_dim,
            "split_act",
        )?;
        let dequant_weight = dequantize_u8_vec(weight_q8, weight_params);
        let weight_only = matvec_rows(
            &dequant_weight,
            norm_data,
            ffn_inner,
            hidden_dim,
            "split_weight",
        )?;
        let output_requant_only: Vec<f32> = projection
            .iter()
            .map(|&v| requantize_f32_through_u8(v, out_params))
            .collect();
        let full_w8a8 = quantized_fc_output_reference(
            &input_q8,
            quant_params.input,
            weight_q8,
            weight_params,
            out_params,
            ffn_inner,
            hidden_dim,
            "split_full",
        )?;
        Ok(QuantizedGatedGeluSplit {
            stage,
            f32_reference: projection.to_vec(),
            full_w8a8,
            act_only,
            weight_only,
            output_requant_only,
        })
    };
    let gate = build_one(
        rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::GateFc,
        gate_weight,
        gate_weight_quantized,
        quant_params.gate_weight,
        quant_params.gate,
        gate_projection,
    )
    .ok()?;
    let up = build_one(
        rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::UpFc,
        up_weight,
        up_weight_quantized,
        quant_params.up_weight,
        quant_params.up,
        up_projection,
    )
    .ok()?;
    Some(vec![gate, up])
}

pub(super) fn quantized_split_dominant(
    full: MediaTekFfnParityStats,
    act: MediaTekFfnParityStats,
    weight: MediaTekFfnParityStats,
    output_requant: MediaTekFfnParityStats,
) -> &'static str {
    let max_single = act
        .max_abs_error
        .max(weight.max_abs_error)
        .max(output_requant.max_abs_error);
    if !full.passed() && full.max_abs_error >= 1.5 * max_single {
        return "accumulation_residual";
    }
    let mut dominant: Option<(&'static str, f32)> = None;
    for (name, stats) in [
        ("activation", act),
        ("weight", weight),
        ("output_requant", output_requant),
    ] {
        if !stats.passed() {
            let better = match dominant {
                Some((_, best)) => stats.max_abs_error > best,
                None => true,
            };
            if better {
                dominant = Some((name, stats.max_abs_error));
            }
        }
    }
    if let Some((name, _)) = dominant {
        return name;
    }
    if !full.passed() {
        return "accumulation_residual";
    }
    "none"
}

pub(super) fn quantized_split_line(
    layer_idx: usize,
    stage_name: &str,
    full: MediaTekFfnParityStats,
    act: MediaTekFfnParityStats,
    weight: MediaTekFfnParityStats,
    output_requant: MediaTekFfnParityStats,
    dominant: &str,
) -> String {
    format!(
        "[mediatek-ffn-split] layer={layer_idx} stage={stage_name} full_w8a8_max_abs={:.9} full_w8a8_max_rel={:.9} full_w8a8_cosine={:.9} act_quant_max_abs={:.9} act_quant_max_rel={:.9} act_quant_cosine={:.9} weight_quant_max_abs={:.9} weight_quant_max_rel={:.9} weight_quant_cosine={:.9} output_requant_max_abs={:.9} output_requant_max_rel={:.9} output_requant_cosine={:.9} dominant={dominant}",
        full.max_abs_error,
        full.max_rel_error,
        full.cosine_similarity,
        act.max_abs_error,
        act.max_rel_error,
        act.cosine_similarity,
        weight.max_abs_error,
        weight.max_rel_error,
        weight.cosine_similarity,
        output_requant.max_abs_error,
        output_requant.max_rel_error,
        output_requant.cosine_similarity,
    )
}

fn emit_quantized_split_lines(layer_idx: usize, splits: &[QuantizedGatedGeluSplit]) {
    for split in splits {
        let stage_name = split.stage.name();
        let full = match parity_stats(&split.f32_reference, &split.full_w8a8) {
            Ok(stats) => stats,
            Err(reason) => {
                eprintln!(
                    "[mediatek-ffn-split] layer={layer_idx} stage={stage_name} invalidated error={}",
                    reason.trace_reason()
                );
                continue;
            }
        };
        let act = match parity_stats(&split.f32_reference, &split.act_only) {
            Ok(stats) => stats,
            Err(reason) => {
                eprintln!(
                    "[mediatek-ffn-split] layer={layer_idx} stage={stage_name} invalidated error={}",
                    reason.trace_reason()
                );
                continue;
            }
        };
        let weight = match parity_stats(&split.f32_reference, &split.weight_only) {
            Ok(stats) => stats,
            Err(reason) => {
                eprintln!(
                    "[mediatek-ffn-split] layer={layer_idx} stage={stage_name} invalidated error={}",
                    reason.trace_reason()
                );
                continue;
            }
        };
        let output_requant = match parity_stats(&split.f32_reference, &split.output_requant_only) {
            Ok(stats) => stats,
            Err(reason) => {
                eprintln!(
                    "[mediatek-ffn-split] layer={layer_idx} stage={stage_name} invalidated error={}",
                    reason.trace_reason()
                );
                continue;
            }
        };
        let dominant = quantized_split_dominant(full, act, weight, output_requant);
        eprintln!(
            "{}",
            quantized_split_line(
                layer_idx,
                stage_name,
                full,
                act,
                weight,
                output_requant,
                dominant
            )
        );
    }
}

fn emit_quantized_stage_probe_lines(
    layer_idx: usize,
    device_name: &str,
    hidden_dim: usize,
    ffn_inner: usize,
    norm_data: &[f32],
    weights: &rnb_runtime::mediatek::RunCachedQuantizedGatedGeluFfnWeights,
    stage_references: &[QuantizedGatedGeluStageReference],
) {
    let shape =
        rnb_runtime::mediatek::MediaTekGatedGeluFfnShape::new(hidden_dim, ffn_inner, hidden_dim);
    for stage_reference in stage_references {
        let request = rnb_runtime::mediatek::RunQuantizedGatedGeluFfnStageProbeRequest {
            device_name_substring: device_name.to_string(),
            shape,
            weights: weights.clone(),
            input: norm_data,
            stage: stage_reference.stage,
        };
        let stage_output =
            match rnb_runtime::mediatek::run_quantized_gated_gelu_ffn_stage_probe(request) {
                Ok(stage_output) => stage_output,
                Err(err) => {
                    eprintln!(
                        "{}",
                        stage_probe_invalidation_line(
                            layer_idx,
                            stage_reference.stage,
                            &format!("runtime_stage_probe_failed:{err}"),
                        )
                    );
                    continue;
                }
            };
        let nnapi_vs_cpu_quant = match parity_stats(
            &stage_reference.cpu_quantized_reference,
            &stage_output.output,
        ) {
            Ok(stats) => stats,
            Err(reason) => {
                eprintln!(
                    "{}",
                    stage_probe_invalidation_line(
                        layer_idx,
                        stage_reference.stage,
                        reason.trace_reason(),
                    )
                );
                continue;
            }
        };
        let cpu_quant_vs_f32 = match parity_stats(
            &stage_reference.f32_reference,
            &stage_reference.cpu_quantized_reference,
        ) {
            Ok(stats) => stats,
            Err(reason) => {
                eprintln!(
                    "{}",
                    stage_probe_invalidation_line(
                        layer_idx,
                        stage_reference.stage,
                        reason.trace_reason(),
                    )
                );
                continue;
            }
        };
        let nnapi_vs_f32 = if stage_reference.stage
            == rnb_runtime::mediatek::MediaTekQuantizedGatedGeluFfnStage::Output
        {
            match parity_stats(&stage_reference.f32_reference, &stage_output.output) {
                Ok(stats) => Some(stats),
                Err(reason) => {
                    eprintln!(
                        "{}",
                        stage_probe_invalidation_line(
                            layer_idx,
                            stage_reference.stage,
                            reason.trace_reason(),
                        )
                    );
                    continue;
                }
            }
        } else {
            None
        };
        eprintln!(
            "{}",
            stage_probe_line(
                layer_idx,
                stage_reference.stage,
                nnapi_vs_cpu_quant,
                cpu_quant_vs_f32,
                nnapi_vs_f32,
            )
        );
    }
}

pub(super) fn try_mediatek_gemma_prefill_ffn_down(
    architecture: ModelArchitecture,
    layer_idx: usize,
    hidden_dim: usize,
    seq_len: usize,
    norm_data: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    fused_gate_up_present: bool,
) -> crate::error::Result<Option<Vec<f32>>> {
    try_mediatek_gemma_prefill_ffn_down_with_mode(
        architecture,
        layer_idx,
        hidden_dim,
        seq_len,
        norm_data,
        ffn_gate_weight,
        ffn_up_weight,
        ffn_down_weight,
        fused_gate_up_present,
        rnb_runtime::mediatek::MediaTekPrefillRequestMode::UserPath,
        false,
    )
}

pub(super) fn try_mediatek_gemma_prefill_ffn_down_with_mode(
    architecture: ModelArchitecture,
    layer_idx: usize,
    hidden_dim: usize,
    seq_len: usize,
    norm_data: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    fused_gate_up_present: bool,
    request_mode: rnb_runtime::mediatek::MediaTekPrefillRequestMode,
    sync_compile_enabled: bool,
) -> crate::error::Result<Option<Vec<f32>>> {
    if !opt_in_enabled() {
        return Ok(None);
    }
    let total_start = std::time::Instant::now();
    if super::trace::dump_bin_dir().is_some() {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::DumpModeActive);
        return Ok(None);
    }
    if !is_gemma_dense(architecture) {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::NonGemmaArchitecture);
        return Ok(None);
    }
    if fused_gate_up_present {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::FusedGateUpUnhandled);
        return Ok(None);
    }
    let Some(plan) = prefill_batch_plan(seq_len) else {
        return Ok(None);
    };
    let use_compiled_reuse = compiled_reuse_enabled();
    if !use_compiled_reuse {
        return Ok(None);
    }
    let device_name = match resolve_device_name() {
        Ok(device) => device,
        Err(reason) => {
            trace_fallback(layer_idx, reason);
            return Ok(None);
        }
    };
    if norm_data.len() != seq_len * hidden_dim
        || ffn_gate_weight.cols != hidden_dim
        || ffn_up_weight.cols != hidden_dim
    {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::ShapeMismatch);
        return Ok(None);
    }
    if ffn_gate_weight.rows != ffn_up_weight.rows || ffn_down_weight.cols != ffn_gate_weight.rows {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::ShapeMismatch);
        return Ok(None);
    }
    if ffn_down_weight.rows != hidden_dim {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::ShapeMismatch);
        return Ok(None);
    }

    let ffn_inner = ffn_gate_weight.rows;
    let parity = parity_required();
    let trace = trace_enabled();
    let measure_timing = mediatek_ffn_measure_timing(trace, parity);
    let base_cache_key = match (
        ffn_gate_weight.mediatek_gated_gelu_cache_weight_key(),
        ffn_up_weight.mediatek_gated_gelu_cache_weight_key(),
        ffn_down_weight.mediatek_gated_gelu_cache_weight_key(),
    ) {
        (Some(gate), Some(up), Some(down)) => rnb_runtime::mediatek::GatedGeluFfnF32CacheKey {
            device_name_substring: device_name.clone(),
            layer_idx,
            input_size: hidden_dim,
            ffn_inner_size: ffn_inner,
            output_size: hidden_dim,
            gate,
            up,
            down,
        },
        _ => {
            trace_fallback(layer_idx, MediaTekFfnRejectReason::MaterializationFailed);
            return Ok(None);
        }
    };
    let cache_key = rnb_runtime::mediatek::GatedGeluFfnF32BatchedCacheKey {
        base: base_cache_key,
        batch: plan.batch_size,
    };
    let runtime_cache_hit = rnb_runtime::mediatek::is_gated_gelu_ffn_f32_batched_cached(&cache_key);
    let prefill_decision = decide_prefill_npu_for_cache_presence(
        layer_idx,
        seq_len,
        runtime_cache_hit,
        false,
        request_mode,
        sync_compile_enabled,
    );
    let max_cache_entries = match prefill_decision {
        rnb_runtime::mediatek::MediaTekPrefillNpuDecision::UseWarmNpu { max_cache_entries } => {
            max_cache_entries
        }
        rnb_runtime::mediatek::MediaTekPrefillNpuDecision::AllowPrewarmCompile {
            max_cache_entries,
        } => max_cache_entries,
        rnb_runtime::mediatek::MediaTekPrefillNpuDecision::FallbackCpu { reason } => {
            trace_prefill_policy_fallback(layer_idx, reason);
            return Ok(None);
        }
    };

    let needs_host_weights = prefill_needs_host_weights(runtime_cache_hit, parity);
    let materialize_start = std::time::Instant::now();
    let materialized_weights = if needs_host_weights {
        match (
            ffn_gate_weight.materialize_f32_owned("gate_weight"),
            ffn_up_weight.materialize_f32_owned("up_weight"),
            ffn_down_weight.materialize_f32_owned("down_weight"),
        ) {
            (Ok(gate), Ok(up), Ok(down)) => Some((gate, up, down)),
            _ => {
                trace_fallback(layer_idx, MediaTekFfnRejectReason::MaterializationFailed);
                return Ok(None);
            }
        }
    } else {
        None
    };
    let materialize_ns = if needs_host_weights {
        elapsed_ns(materialize_start)
    } else {
        0
    };
    let mut runtime_weights = if runtime_cache_hit {
        None
    } else {
        let (gate, up, down) = materialized_weights
            .as_ref()
            .expect("cache miss must materialize runtime weights");
        Some(rnb_runtime::mediatek::RunCachedGatedGeluFfnF32Weights {
            gate_weight: gate.clone(),
            up_weight: up.clone(),
            down_weight: down.clone(),
        })
    };
    let mut output = Vec::with_capacity(seq_len * hidden_dim);
    let mut first_chunk_cache_hit = runtime_cache_hit;
    let mut compile_ns = 0u64;
    let mut execution_compute_ns = 0u64;
    let mut execute_hw_ns = 0u64;
    let mut execute_driver_ns = 0u64;

    for chunk_idx in 0..plan.full_chunks {
        let token_start = chunk_idx * plan.batch_size;
        let row_start = token_start * hidden_dim;
        let row_end = row_start + plan.batch_size * hidden_dim;
        let request = rnb_runtime::mediatek::RunCachedGatedGeluFfnF32BatchedRequest {
            cache_key: cache_key.clone(),
            weights: runtime_weights.take(),
            max_cache_entries,
            measure_timing,
            input: &norm_data[row_start..row_end],
        };
        let chunk_output =
            match rnb_runtime::mediatek::run_cached_gated_gelu_ffn_f32_batched(request) {
                Ok(output) => output,
                Err(err) => {
                    eprintln!(
                        "[mediatek-ffn] used=false layer={layer_idx} reason={} error={err}",
                        MediaTekFfnRejectReason::RuntimeUnavailable.trace_reason()
                    );
                    return Ok(None);
                }
            };
        if chunk_idx == 0 {
            first_chunk_cache_hit = chunk_output.cache_hit;
            compile_ns = chunk_output.timings.compilation_ns;
        }
        if chunk_output.output.len() != plan.batch_size * hidden_dim {
            trace_fallback(layer_idx, MediaTekFfnRejectReason::ShapeMismatch);
            return Ok(None);
        }
        if !runtime_output_is_finite(&chunk_output.output) {
            trace_fallback(layer_idx, MediaTekFfnRejectReason::ParityFailed);
            return Ok(None);
        }
        if parity {
            let (gate, up, down) = materialized_weights
                .as_ref()
                .expect("parity requires host weights");
            let mut reference = Vec::with_capacity(plan.batch_size * hidden_dim);
            for token_offset in 0..plan.batch_size {
                let token_row_start = row_start + token_offset * hidden_dim;
                let token_row_end = token_row_start + hidden_dim;
                reference.extend_from_slice(&cpu_reference_down(
                    architecture,
                    &norm_data[token_row_start..token_row_end],
                    gate,
                    up,
                    down,
                    hidden_dim,
                    ffn_inner,
                )?);
            }
            let stats = match parity_stats(&reference, &chunk_output.output) {
                Ok(stats) => stats,
                Err(reason) => {
                    trace_fallback(layer_idx, reason);
                    return Ok(None);
                }
            };
            if !stats.passed() {
                eprintln!(
                    "[mediatek-ffn] used=false layer={layer_idx} chunk={chunk_idx} reason={} max_abs_error={:.9} max_rel_error={:.9} cosine_similarity={:.9}",
                    MediaTekFfnRejectReason::ParityFailed.trace_reason(),
                    stats.max_abs_error,
                    stats.max_rel_error,
                    stats.cosine_similarity
                );
                return Ok(None);
            }
            eprintln!("{}", parity_pass_line(layer_idx, stats));
        }
        execution_compute_ns =
            execution_compute_ns.saturating_add(chunk_output.timings.execution_compute_ns);
        execute_hw_ns =
            execute_hw_ns.saturating_add(chunk_output.duration_hardware_ns.unwrap_or(0));
        execute_driver_ns =
            execute_driver_ns.saturating_add(chunk_output.duration_driver_ns.unwrap_or(0));
        output.extend_from_slice(&chunk_output.output);
    }

    if plan.tail_tokens > 0 {
        let tail_row_start = plan.full_chunks * plan.batch_size * hidden_dim;
        let tail_row_end = tail_row_start + plan.tail_tokens * hidden_dim;
        output.extend_from_slice(&prefill_tail_down_with_quantized_cpu(
            architecture,
            hidden_dim,
            &norm_data[tail_row_start..tail_row_end],
            ffn_gate_weight,
            ffn_up_weight,
            ffn_down_weight,
            plan.tail_tokens,
        )?);
    }

    if trace {
        eprintln!(
            "[mediatek-ffn-prefill] used=true layer={layer_idx} batch={} full_chunks={} tail_tokens={} first_cache_hit={} materialize_ns={} compile_ns={} execution_compute_ns={} execute_hw_ns={} execute_driver_ns={} total_ns={}",
            plan.batch_size,
            plan.full_chunks,
            plan.tail_tokens,
            first_chunk_cache_hit,
            materialize_ns,
            compile_ns,
            execution_compute_ns,
            execute_hw_ns,
            execute_driver_ns,
            elapsed_ns(total_start)
        );
    }
    Ok(Some(output))
}

impl super::Engine {
    pub fn prewarm_mediatek_prefill_ffn(
        &mut self,
        seq_len: usize,
        request_mode: rnb_runtime::mediatek::MediaTekPrefillRequestMode,
    ) -> crate::error::Result<usize> {
        if matches!(
            request_mode,
            rnb_runtime::mediatek::MediaTekPrefillRequestMode::UserPath
        ) {
            return Ok(0);
        }
        if !opt_in_enabled() || !prefill_enabled() {
            return Ok(0);
        }
        let Some(weights) = self.weights.as_ref() else {
            return Ok(0);
        };
        let hidden_dim = self.metadata.hidden_dim;
        let norm_data = vec![0.0f32; seq_len.saturating_mul(hidden_dim)];
        let mut used_layers = 0usize;
        let thread_id = format!("{:?}", std::thread::current().id());
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            let used = match layer {
                LayerType::Attention(w) => try_mediatek_gemma_prefill_ffn_down_with_mode(
                    self.architecture,
                    layer_idx,
                    hidden_dim,
                    seq_len,
                    &norm_data,
                    &w.ffn_gate_weight,
                    &w.ffn_up_weight,
                    &w.ffn_down_weight,
                    w.ffn_gate_up_fused.is_some(),
                    request_mode,
                    true,
                )?
                .is_some(),
                LayerType::GatedDeltaNet(w) => try_mediatek_gemma_prefill_ffn_down_with_mode(
                    self.architecture,
                    layer_idx,
                    hidden_dim,
                    seq_len,
                    &norm_data,
                    &w.ffn_gate_weight,
                    &w.ffn_up_weight,
                    &w.ffn_down_weight,
                    w.ffn_gate_up_fused.is_some(),
                    request_mode,
                    true,
                )?
                .is_some(),
                LayerType::NemotronMamba2(_) | LayerType::NemotronMoE(_) => false,
            };
            if used {
                used_layers += 1;
            }
        }
        eprintln!(
            "[mediatek-ffn-prefill-prewarm] request_mode={request_mode:?} seq_len={seq_len} used_layers={used_layers} thread_id={thread_id}"
        );
        Ok(used_layers)
    }
}

pub(super) fn try_mediatek_gemma_ffn_down(
    architecture: ModelArchitecture,
    layer_idx: usize,
    hidden_dim: usize,
    norm_data: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    fused_gate_up_present: bool,
) -> crate::error::Result<Option<Vec<f32>>> {
    enum ParityReference {
        Ready(Vec<f32>),
        Weights {
            gate: Vec<f32>,
            up: Vec<f32>,
            down: Vec<f32>,
        },
    }

    if !opt_in_enabled() {
        return Ok(None);
    }
    let total_start = std::time::Instant::now();
    if super::trace::dump_bin_dir().is_some() {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::DumpModeActive);
        return Ok(None);
    }
    if !is_gemma_dense(architecture) {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::NonGemmaArchitecture);
        return Ok(None);
    }
    if fused_gate_up_present {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::FusedGateUpUnhandled);
        return Ok(None);
    }
    let use_compiled_reuse = compiled_reuse_enabled();
    let use_quantized = quantized_enabled();
    let quantized_stage_probe = use_quantized && quantized_stage_probe_enabled();
    if use_quantized && !use_compiled_reuse {
        trace_fallback(
            layer_idx,
            MediaTekFfnRejectReason::LayerListRequiresCompiledReuse,
        );
        return Ok(None);
    }
    let layer_decision = match selected_layer_decision(layer_idx, use_compiled_reuse) {
        Ok(decision) => decision,
        Err(reason) => {
            trace_fallback(layer_idx, reason);
            return Ok(None);
        }
    };
    if !layer_decision.selected {
        return Ok(None);
    }
    let max_cache_entries = layer_decision.max_cache_entries;

    let device_name = match resolve_device_name() {
        Ok(device) => device,
        Err(reason) => {
            trace_fallback(layer_idx, reason);
            return Ok(None);
        }
    };
    if norm_data.len() != hidden_dim
        || ffn_gate_weight.cols != hidden_dim
        || ffn_up_weight.cols != hidden_dim
    {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::ShapeMismatch);
        return Ok(None);
    }
    if ffn_gate_weight.rows != ffn_up_weight.rows || ffn_down_weight.cols != ffn_gate_weight.rows {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::ShapeMismatch);
        return Ok(None);
    }
    if ffn_down_weight.rows != hidden_dim {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::ShapeMismatch);
        return Ok(None);
    }

    let ffn_inner = ffn_gate_weight.rows;
    let parity = parity_required();
    let trace = trace_enabled();
    let measure_timing = mediatek_ffn_measure_timing(trace, parity);
    let cache_key = if use_compiled_reuse {
        match (
            ffn_gate_weight.mediatek_gated_gelu_cache_weight_key(),
            ffn_up_weight.mediatek_gated_gelu_cache_weight_key(),
            ffn_down_weight.mediatek_gated_gelu_cache_weight_key(),
        ) {
            (Some(gate), Some(up), Some(down)) => {
                Some(rnb_runtime::mediatek::GatedGeluFfnF32CacheKey {
                    device_name_substring: device_name.clone(),
                    layer_idx,
                    input_size: hidden_dim,
                    ffn_inner_size: ffn_inner,
                    output_size: hidden_dim,
                    gate,
                    up,
                    down,
                })
            }
            _ => {
                trace_fallback(layer_idx, MediaTekFfnRejectReason::MaterializationFailed);
                return Ok(None);
            }
        }
    } else {
        None
    };
    let runtime_cache_hit = cache_key
        .as_ref()
        .map(|key| {
            if use_quantized {
                false
            } else {
                rnb_runtime::mediatek::is_gated_gelu_ffn_f32_cached(key)
            }
        })
        .unwrap_or(false);

    let (output, materialize_ns, runtime_total_ns, parity_reference, parity_precompute_ns) =
        if use_compiled_reuse {
            if use_quantized {
                let materialize_start = std::time::Instant::now();
                let (gate, up, down) = match (
                    ffn_gate_weight.materialize_f32_owned("gate_weight"),
                    ffn_up_weight.materialize_f32_owned("up_weight"),
                    ffn_down_weight.materialize_f32_owned("down_weight"),
                ) {
                    (Ok(gate), Ok(up), Ok(down)) => (gate, up, down),
                    _ => {
                        trace_fallback(layer_idx, MediaTekFfnRejectReason::MaterializationFailed);
                        return Ok(None);
                    }
                };
                let materialize_ns = elapsed_ns(materialize_start);
                let parity_precompute_start = std::time::Instant::now();
                let admission = build_quantized_gated_gelu_admission(
                    architecture,
                    norm_data,
                    gate,
                    up,
                    down,
                    hidden_dim,
                    ffn_inner,
                    quantized_stage_probe,
                )?;
                let parity_precompute_ns = elapsed_ns(parity_precompute_start);
                if let Some(stage_references) = admission.stage_references.as_ref() {
                    emit_quantized_stage_probe_lines(
                        layer_idx,
                        &device_name,
                        hidden_dim,
                        ffn_inner,
                        norm_data,
                        &admission.weights,
                        stage_references,
                    );
                }
                if let Some(splits) = admission.splits.as_ref() {
                    emit_quantized_split_lines(layer_idx, splits);
                }
                let admitted_cache_hit = cache_key
                    .as_ref()
                    .map(|key| {
                        rnb_runtime::mediatek::is_gated_gelu_ffn_quantized_cached_with_params(
                            key,
                            admission.weights.quant_params,
                        )
                    })
                    .unwrap_or(false);
                let parity_reference = Some(ParityReference::Ready(admission.reference));
                let weights = Some(admission.weights);
                let request = rnb_runtime::mediatek::RunCachedQuantizedGatedGeluFfnRequest {
                    cache_key: cache_key
                        .as_ref()
                        .expect("compiled reuse enabled builds a cache key")
                        .clone(),
                    weights,
                    max_cache_entries,
                    measure_timing,
                    input: norm_data,
                };
                let runtime_start = std::time::Instant::now();
                let output =
                    match rnb_runtime::mediatek::run_cached_quantized_gated_gelu_ffn(request) {
                        Ok(output) => output,
                        Err(err) => {
                            eprintln!(
                                "[mediatek-ffn] used=false layer={layer_idx} reason={} error={err}",
                                MediaTekFfnRejectReason::RuntimeUnavailable.trace_reason()
                            );
                            if admitted_cache_hit {
                                return Err(LlmError::Forward(format!(
                                    "MediaTek quantized FFN admitted cache failed at runtime: {err}"
                                )));
                            }
                            if let Some(key) = cache_key.as_ref() {
                                rnb_runtime::mediatek::remove_gated_gelu_ffn_quantized_cache_entry(
                                    key,
                                );
                            }
                            return Ok(None);
                        }
                    };
                (
                    output,
                    materialize_ns,
                    elapsed_ns(runtime_start),
                    parity_reference,
                    parity_precompute_ns,
                )
            } else {
                let skip_materialization = runtime_cache_hit && !parity;
                let mut parity_reference = None;
                let mut parity_precompute_ns = 0;
                let mut materialize_ns = 0;
                let weights = if skip_materialization {
                    None
                } else {
                    let materialize_start = std::time::Instant::now();
                    let (gate, up, down) = match (
                        ffn_gate_weight.materialize_f32_owned("gate_weight"),
                        ffn_up_weight.materialize_f32_owned("up_weight"),
                        ffn_down_weight.materialize_f32_owned("down_weight"),
                    ) {
                        (Ok(gate), Ok(up), Ok(down)) => (gate, up, down),
                        _ => {
                            trace_fallback(
                                layer_idx,
                                MediaTekFfnRejectReason::MaterializationFailed,
                            );
                            return Ok(None);
                        }
                    };
                    materialize_ns = elapsed_ns(materialize_start);
                    if parity {
                        if runtime_cache_hit {
                            parity_reference = Some(ParityReference::Weights { gate, up, down });
                            None
                        } else {
                            let parity_precompute_start = std::time::Instant::now();
                            let reference = cpu_reference_down(
                                architecture,
                                norm_data,
                                &gate,
                                &up,
                                &down,
                                hidden_dim,
                                ffn_inner,
                            )?;
                            parity_precompute_ns = elapsed_ns(parity_precompute_start);
                            parity_reference = Some(ParityReference::Ready(reference));
                            Some(rnb_runtime::mediatek::RunCachedGatedGeluFfnF32Weights {
                                gate_weight: gate,
                                up_weight: up,
                                down_weight: down,
                            })
                        }
                    } else {
                        Some(rnb_runtime::mediatek::RunCachedGatedGeluFfnF32Weights {
                            gate_weight: gate,
                            up_weight: up,
                            down_weight: down,
                        })
                    }
                };
                let request = rnb_runtime::mediatek::RunCachedGatedGeluFfnF32Request {
                    cache_key: cache_key
                        .as_ref()
                        .expect("compiled reuse enabled builds a cache key")
                        .clone(),
                    weights,
                    max_cache_entries,
                    measure_timing,
                    input: norm_data,
                };
                let runtime_start = std::time::Instant::now();
                let output = match rnb_runtime::mediatek::run_cached_gated_gelu_ffn_f32(request) {
                    Ok(output) => output,
                    Err(err) => {
                        eprintln!(
                            "[mediatek-ffn] used=false layer={layer_idx} reason={} error={err}",
                            MediaTekFfnRejectReason::RuntimeUnavailable.trace_reason()
                        );
                        return Ok(None);
                    }
                };
                (
                    output,
                    materialize_ns,
                    elapsed_ns(runtime_start),
                    parity_reference,
                    parity_precompute_ns,
                )
            }
        } else {
            let materialize_start = std::time::Instant::now();
            let (gate, up, down) = match (
                ffn_gate_weight.materialize_f32_owned("gate_weight"),
                ffn_up_weight.materialize_f32_owned("up_weight"),
                ffn_down_weight.materialize_f32_owned("down_weight"),
            ) {
                (Ok(gate), Ok(up), Ok(down)) => (gate, up, down),
                _ => {
                    trace_fallback(layer_idx, MediaTekFfnRejectReason::MaterializationFailed);
                    return Ok(None);
                }
            };
            let materialize_ns = elapsed_ns(materialize_start);
            let request = rnb_runtime::mediatek::RunGatedGeluFfnF32Request {
                device_name_substring: device_name,
                input_size: hidden_dim,
                ffn_inner_size: ffn_inner,
                output_size: hidden_dim,
                gate_weight: &gate,
                up_weight: &up,
                down_weight: &down,
                input: norm_data,
            };
            let runtime_start = std::time::Instant::now();
            let output = match rnb_runtime::mediatek::run_gated_gelu_ffn_f32(request) {
                Ok(output) => output,
                Err(err) => {
                    eprintln!(
                        "[mediatek-ffn] used=false layer={layer_idx} reason={} error={err}",
                        MediaTekFfnRejectReason::RuntimeUnavailable.trace_reason()
                    );
                    return Ok(None);
                }
            };
            (
                output,
                materialize_ns,
                elapsed_ns(runtime_start),
                if parity {
                    Some(ParityReference::Weights { gate, up, down })
                } else {
                    None
                },
                0,
            )
        };
    if output.output.len() != hidden_dim {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::ShapeMismatch);
        if use_quantized {
            if let Some(key) = cache_key.as_ref() {
                rnb_runtime::mediatek::remove_gated_gelu_ffn_quantized_cache_entry(key);
            }
            if output.cache_hit {
                return Err(LlmError::Forward(
                    "MediaTek quantized FFN admitted cache returned wrong output length"
                        .to_string(),
                ));
            }
        }
        return Ok(None);
    }
    if !runtime_output_is_finite(&output.output) {
        trace_fallback(layer_idx, MediaTekFfnRejectReason::ParityFailed);
        if use_quantized {
            if let Some(key) = cache_key.as_ref() {
                rnb_runtime::mediatek::remove_gated_gelu_ffn_quantized_cache_entry(key);
            }
            if output.cache_hit {
                return Err(LlmError::Forward(
                    "MediaTek quantized FFN admitted cache returned non-finite output".to_string(),
                ));
            }
        }
        return Ok(None);
    }

    let mut parity_ns = None;
    let verify_parity = parity || parity_reference.is_some();
    if verify_parity {
        let parity_start = std::time::Instant::now();
        let reference =
            match parity_reference.expect("parity path must keep a CPU reference source") {
                ParityReference::Ready(reference) => reference,
                ParityReference::Weights { gate, up, down } => cpu_reference_down(
                    architecture,
                    norm_data,
                    &gate,
                    &up,
                    &down,
                    hidden_dim,
                    ffn_inner,
                )?,
            };
        let stats = match parity_stats(&reference, &output.output) {
            Ok(stats) => stats,
            Err(reason) => {
                trace_fallback(layer_idx, reason);
                return Ok(None);
            }
        };
        if !stats.passed() {
            eprintln!(
                "[mediatek-ffn] used=false layer={layer_idx} reason={} max_abs_error={:.9} max_rel_error={:.9} cosine_similarity={:.9}",
                MediaTekFfnRejectReason::ParityFailed.trace_reason(),
                stats.max_abs_error,
                stats.max_rel_error,
                stats.cosine_similarity
            );
            if use_quantized {
                if let Some(key) = cache_key.as_ref() {
                    rnb_runtime::mediatek::remove_gated_gelu_ffn_quantized_cache_entry(key);
                }
                if output.cache_hit {
                    return Err(LlmError::Forward(format!(
                        "MediaTek quantized FFN admitted cache failed parity: max_abs_error={:.9} max_rel_error={:.9} cosine_similarity={:.9}",
                        stats.max_abs_error, stats.max_rel_error, stats.cosine_similarity
                    )));
                }
            }
            return Ok(None);
        }
        parity_ns = Some(parity_precompute_ns.saturating_add(elapsed_ns(parity_start)));
        eprintln!("{}", parity_pass_line(layer_idx, stats));
    }

    if trace {
        eprintln!(
            "{}",
            timing_line(
                layer_idx,
                MediaTekFfnLocalTimings {
                    materialize_ns,
                    runtime_total_ns,
                    parity_ns,
                    cache_hit: output.cache_hit,
                    total_ns: elapsed_ns(total_start),
                },
                output.timings,
                output.duration_hardware_ns,
                output.duration_driver_ns,
            )
        );
        eprintln!(
            "[mediatek-ffn] used=true layer={layer_idx} device={} hw_ns={} driver_ns={}",
            output.chosen_device_name,
            output
                .duration_hardware_ns
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            output
                .duration_driver_ns
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string())
        );
    }
    Ok(Some(output.output))
}

pub(super) fn runtime_output_is_finite(output: &[f32]) -> bool {
    output.iter().all(|value| value.is_finite())
}

pub(super) fn prefill_tail_down_with_quantized_cpu(
    architecture: ModelArchitecture,
    hidden_dim: usize,
    norm_data: &[f32],
    ffn_gate_weight: &QuantizedWeight,
    ffn_up_weight: &QuantizedWeight,
    ffn_down_weight: &QuantizedWeight,
    tail_tokens: usize,
) -> crate::error::Result<Vec<f32>> {
    let (mut gate, up) =
        prefill_gate_up_vectors(ffn_gate_weight, ffn_up_weight, None, norm_data, tail_tokens)?;
    apply_model_gate_mul_inplace(&mut gate, &up, architecture);
    let output = ffn_down_weight.gemv_vec(&gate)?;
    let expected = tail_tokens
        .checked_mul(hidden_dim)
        .ok_or_else(|| LlmError::Forward("MediaTek FFN tail output length overflow".to_string()))?;
    if output.len() != expected {
        return Err(LlmError::Forward(format!(
            "MediaTek FFN tail output shape mismatch expected {expected}, got {}",
            output.len()
        )));
    }
    Ok(output)
}

pub(super) fn cpu_reference_down(
    architecture: ModelArchitecture,
    norm_data: &[f32],
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    hidden_dim: usize,
    ffn_inner: usize,
) -> crate::error::Result<Vec<f32>> {
    let mut gate = matvec_rows(gate_weight, norm_data, ffn_inner, hidden_dim, "gate_weight")?;
    let up = matvec_rows(up_weight, norm_data, ffn_inner, hidden_dim, "up_weight")?;
    apply_model_gate_mul_inplace(&mut gate, &up, architecture);
    matmul_down(down_weight, &gate, hidden_dim, ffn_inner)
}

fn matvec_rows(
    weight: &[f32],
    input: &[f32],
    rows: usize,
    cols: usize,
    label: &'static str,
) -> crate::error::Result<Vec<f32>> {
    if weight.len() != rows * cols || input.len() != cols {
        return Err(LlmError::Forward(format!(
            "MediaTek FFN reference {label} matvec shape mismatch"
        )));
    }
    let mut output = vec![0.0f32; rows];
    for row in 0..rows {
        let row_start = row * cols;
        let mut acc = 0.0f32;
        for col in 0..cols {
            acc += weight[row_start + col] * input[col];
        }
        output[row] = acc;
    }
    Ok(output)
}
fn matmul_down(
    down_weight: &[f32],
    gated: &[f32],
    hidden_dim: usize,
    ffn_inner: usize,
) -> crate::error::Result<Vec<f32>> {
    if down_weight.len() != hidden_dim * ffn_inner || gated.len() != ffn_inner {
        return Err(LlmError::Forward(
            "MediaTek FFN reference down projection shape mismatch".to_string(),
        ));
    }
    let mut output = vec![0.0f32; hidden_dim];
    for row in 0..hidden_dim {
        let row_start = row * ffn_inner;
        let mut acc = 0.0f32;
        for col in 0..ffn_inner {
            acc += down_weight[row_start + col] * gated[col];
        }
        output[row] = acc;
    }
    Ok(output)
}
