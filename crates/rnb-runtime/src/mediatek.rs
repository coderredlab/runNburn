pub use rnb_backend_mediatek::{
    BatchedCompileOptions, MediaTekGatedGeluFfnBatchedCompilation, MediaTekGatedGeluFfnShape,
    MediaTekQuantParams, MediaTekQuantizedGatedGeluFfnQuantParams,
    MediaTekQuantizedGatedGeluFfnStage,
};
#[cfg(target_os = "android")]
use std::cell::RefCell;
use std::fmt;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct RunGatedGeluFfnF32Request<'a> {
    pub device_name_substring: String,
    pub input_size: usize,
    pub ffn_inner_size: usize,
    pub output_size: usize,
    pub gate_weight: &'a [f32],
    pub up_weight: &'a [f32],
    pub down_weight: &'a [f32],
    pub input: &'a [f32],
}

impl RunGatedGeluFfnF32Request<'_> {
    pub fn validate(&self) -> Result<(), MediaTekRunError> {
        if self.device_name_substring.trim().is_empty() {
            return Err(MediaTekRunError::InvalidDeviceName);
        }
        if self.input_size == 0 {
            return Err(invalid_shape("input_size must be non-zero"));
        }
        if self.ffn_inner_size == 0 {
            return Err(invalid_shape("ffn_inner_size must be non-zero"));
        }
        if self.output_size == 0 {
            return Err(invalid_shape("output_size must be non-zero"));
        }

        let gate_up_len = self
            .ffn_inner_size
            .checked_mul(self.input_size)
            .ok_or_else(|| invalid_shape("gate/up length overflow"))?;
        let down_len = self
            .output_size
            .checked_mul(self.ffn_inner_size)
            .ok_or_else(|| invalid_shape("down length overflow"))?;
        expect_len("gate_weight", gate_up_len, self.gate_weight.len())?;
        expect_len("up_weight", gate_up_len, self.up_weight.len())?;
        expect_len("down_weight", down_len, self.down_weight.len())?;
        expect_len("input", self.input_size, self.input.len())?;
        expect_finite("gate_weight", self.gate_weight)?;
        expect_finite("up_weight", self.up_weight)?;
        expect_finite("down_weight", self.down_weight)?;
        expect_finite("input", self.input)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ProbeGatedGeluFfnF32Request<'a> {
    pub device_name_substring: String,
    pub input_size: usize,
    pub ffn_inner_size: usize,
    pub output_size: usize,
    pub gate_weight: &'a [f32],
    pub up_weight: &'a [f32],
    pub down_weight: &'a [f32],
    pub input: &'a [f32],
}

impl ProbeGatedGeluFfnF32Request<'_> {
    pub fn validate(&self) -> Result<(), MediaTekProbeError> {
        RunGatedGeluFfnF32Request {
            device_name_substring: self.device_name_substring.clone(),
            input_size: self.input_size,
            ffn_inner_size: self.ffn_inner_size,
            output_size: self.output_size,
            gate_weight: self.gate_weight,
            up_weight: self.up_weight,
            down_weight: self.down_weight,
            input: self.input,
        }
        .validate()
    }
}

impl<'a> From<ProbeGatedGeluFfnF32Request<'a>> for RunGatedGeluFfnF32Request<'a> {
    fn from(request: ProbeGatedGeluFfnF32Request<'a>) -> Self {
        Self {
            device_name_substring: request.device_name_substring,
            input_size: request.input_size,
            ffn_inner_size: request.ffn_inner_size,
            output_size: request.output_size,
            gate_weight: request.gate_weight,
            up_weight: request.up_weight,
            down_weight: request.down_weight,
            input: request.input,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GatedGeluFfnF32WeightKey {
    pub generation_id: u64,
    pub raw_ptr: usize,
    pub raw_len: usize,
    pub rows: usize,
    pub cols: usize,
    pub ggml_type: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GatedGeluFfnF32CacheKey {
    pub device_name_substring: String,
    pub layer_idx: usize,
    pub input_size: usize,
    pub ffn_inner_size: usize,
    pub output_size: usize,
    pub gate: GatedGeluFfnF32WeightKey,
    pub up: GatedGeluFfnF32WeightKey,
    pub down: GatedGeluFfnF32WeightKey,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GatedGeluFfnF32BatchedCacheKey {
    pub base: GatedGeluFfnF32CacheKey,
    pub batch: usize,
}

impl GatedGeluFfnF32BatchedCacheKey {
    #[cfg(test)]
    fn for_test(
        layer_idx: usize,
        input_size: usize,
        ffn_inner_size: usize,
        output_size: usize,
        batch: usize,
    ) -> Self {
        Self {
            base: GatedGeluFfnF32CacheKey::for_test(
                layer_idx,
                input_size,
                ffn_inner_size,
                output_size,
            ),
            batch,
        }
    }
}

impl GatedGeluFfnF32CacheKey {
    #[cfg(test)]
    fn for_test(
        layer_idx: usize,
        input_size: usize,
        ffn_inner_size: usize,
        output_size: usize,
    ) -> Self {
        Self {
            device_name_substring: "mtk-neuron".to_string(),
            layer_idx,
            input_size,
            ffn_inner_size,
            output_size,
            gate: GatedGeluFfnF32WeightKey {
                generation_id: 1,
                raw_ptr: 0x1000,
                raw_len: 144,
                rows: ffn_inner_size,
                cols: input_size,
                ggml_type: 14,
            },
            up: GatedGeluFfnF32WeightKey {
                generation_id: 2,
                raw_ptr: 0x2000,
                raw_len: 144,
                rows: ffn_inner_size,
                cols: input_size,
                ggml_type: 14,
            },
            down: GatedGeluFfnF32WeightKey {
                generation_id: 3,
                raw_ptr: 0x3000,
                raw_len: 210,
                rows: output_size,
                cols: ffn_inner_size,
                ggml_type: 14,
            },
        }
    }
}

const PREFILL_LAYER_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_LAYER";
const PREFILL_LAYERS_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_LAYERS";
const PREFILL_CACHE_MAX_LAYERS_ENV: &str = "RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS";
const DEFAULT_PREFILL_CACHE_MAX_LAYERS: usize = 2;
const HARD_MAX_PREFILL_CACHE_LAYERS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaTekPrefillRequestMode {
    UserPath,
    ExplicitPrewarm,
    BenchWarmup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaTekPrefillCacheState {
    InProcessThreadLocalHot,
    DiskAotPresent,
    ColdMissing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaTekPrefillFallbackReason {
    CacheMissUserPath,
    SyncCompileDisabled,
    LayerNotSelected,
    ResidentCapExceeded,
    PromptTooShort,
    AllLayersRequireCompiledCache,
}

impl MediaTekPrefillFallbackReason {
    pub const fn trace_reason(self) -> &'static str {
        match self {
            Self::CacheMissUserPath => "prefill_cache_miss_user_path",
            Self::SyncCompileDisabled => "prefill_sync_compile_disabled",
            Self::LayerNotSelected => "prefill_layer_not_selected",
            Self::ResidentCapExceeded => "prefill_resident_cap_exceeded",
            Self::PromptTooShort => "prefill_prompt_too_short",
            Self::AllLayersRequireCompiledCache => "all_layers_require_compiled_cache",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaTekPrefillNpuDecision {
    UseWarmNpu {
        max_cache_entries: usize,
    },
    AllowPrewarmCompile {
        max_cache_entries: usize,
    },
    FallbackCpu {
        reason: MediaTekPrefillFallbackReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTekPrefillPolicyInput {
    pub layer_idx: usize,
    pub seq_len: usize,
    pub cache_state: MediaTekPrefillCacheState,
    pub request_mode: MediaTekPrefillRequestMode,
    pub sync_compile_enabled: bool,
}

pub fn prefill_cache_max_layers_from_env_value(value: Option<&str>) -> (usize, bool) {
    match value {
        None => (DEFAULT_PREFILL_CACHE_MAX_LAYERS, false),
        Some(value) => match value.trim().parse::<usize>() {
            Ok(value) if (1..=HARD_MAX_PREFILL_CACHE_LAYERS).contains(&value) => (value, false),
            _ => (DEFAULT_PREFILL_CACHE_MAX_LAYERS, true),
        },
    }
}

fn parse_prefill_layer_list(value: &str) -> Option<Vec<usize>> {
    let mut layers = Vec::new();
    for token in value.split(',') {
        let token = token.trim();
        if token.is_empty() {
            return None;
        }
        let layer = token.parse::<usize>().ok()?;
        if layers.contains(&layer) {
            return None;
        }
        layers.push(layer);
    }
    if layers.is_empty() {
        return None;
    }
    Some(layers)
}

fn prefill_layer_selection(
    layer_idx: usize,
    max_cache_entries: usize,
) -> Result<bool, MediaTekPrefillFallbackReason> {
    if let Ok(value) = std::env::var(PREFILL_LAYERS_ENV) {
        let Some(layers) = parse_prefill_layer_list(&value) else {
            return Err(MediaTekPrefillFallbackReason::LayerNotSelected);
        };
        if layers.len() > max_cache_entries {
            return Err(MediaTekPrefillFallbackReason::ResidentCapExceeded);
        }
        return Ok(layers.contains(&layer_idx));
    }

    let selected = match std::env::var(PREFILL_LAYER_ENV) {
        Ok(value) if value.trim() == "all" => {
            return Err(MediaTekPrefillFallbackReason::AllLayersRequireCompiledCache);
        }
        Ok(value) => value.trim().parse::<usize>().ok() == Some(layer_idx),
        Err(_) => layer_idx == 0,
    };
    Ok(selected)
}

pub const fn classify_gemma_prefill_cache_state(
    in_process_thread_local_hot: bool,
    disk_aot_present: bool,
) -> MediaTekPrefillCacheState {
    if in_process_thread_local_hot {
        MediaTekPrefillCacheState::InProcessThreadLocalHot
    } else if disk_aot_present {
        MediaTekPrefillCacheState::DiskAotPresent
    } else {
        MediaTekPrefillCacheState::ColdMissing
    }
}

pub fn decide_gemma_prefill_npu(input: MediaTekPrefillPolicyInput) -> MediaTekPrefillNpuDecision {
    if input.seq_len < 128 {
        return MediaTekPrefillNpuDecision::FallbackCpu {
            reason: MediaTekPrefillFallbackReason::PromptTooShort,
        };
    }

    let raw_cap = std::env::var(PREFILL_CACHE_MAX_LAYERS_ENV).ok();
    let (max_cache_entries, _) = prefill_cache_max_layers_from_env_value(raw_cap.as_deref());
    let selected = match prefill_layer_selection(input.layer_idx, max_cache_entries) {
        Ok(selected) => selected,
        Err(reason) => return MediaTekPrefillNpuDecision::FallbackCpu { reason },
    };
    if !selected {
        return MediaTekPrefillNpuDecision::FallbackCpu {
            reason: MediaTekPrefillFallbackReason::LayerNotSelected,
        };
    }

    match input.request_mode {
        MediaTekPrefillRequestMode::UserPath => {
            if input.cache_state == MediaTekPrefillCacheState::InProcessThreadLocalHot {
                MediaTekPrefillNpuDecision::UseWarmNpu { max_cache_entries }
            } else {
                MediaTekPrefillNpuDecision::FallbackCpu {
                    reason: MediaTekPrefillFallbackReason::CacheMissUserPath,
                }
            }
        }
        MediaTekPrefillRequestMode::ExplicitPrewarm | MediaTekPrefillRequestMode::BenchWarmup => {
            if input.cache_state == MediaTekPrefillCacheState::InProcessThreadLocalHot {
                return MediaTekPrefillNpuDecision::UseWarmNpu { max_cache_entries };
            }
            if input.sync_compile_enabled {
                MediaTekPrefillNpuDecision::AllowPrewarmCompile { max_cache_entries }
            } else {
                MediaTekPrefillNpuDecision::FallbackCpu {
                    reason: MediaTekPrefillFallbackReason::SyncCompileDisabled,
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunCachedGatedGeluFfnF32Weights {
    pub gate_weight: Vec<f32>,
    pub up_weight: Vec<f32>,
    pub down_weight: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunCachedQuantizedGatedGeluFfnWeights {
    pub quant_params: MediaTekQuantizedGatedGeluFfnQuantParams,
    pub gate_weight: Vec<u8>,
    pub up_weight: Vec<u8>,
    pub down_weight: Vec<u8>,
    pub down_weight_f32: Option<Vec<f32>>,
}

impl RunCachedQuantizedGatedGeluFfnWeights {
    fn into_backend(
        self,
        shape: rnb_backend_mediatek::MediaTekGatedGeluFfnShape,
    ) -> Result<rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnOwnedWeights, MediaTekRunError>
    {
        if let Some(down_weight_f32) = self.down_weight_f32 {
            rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnOwnedWeights::new_with_f32_down(
                shape,
                self.quant_params,
                self.gate_weight,
                self.up_weight,
                self.down_weight,
                down_weight_f32,
            )
        } else {
            rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnOwnedWeights::new(
                shape,
                self.quant_params,
                self.gate_weight,
                self.up_weight,
                self.down_weight,
            )
        }
        .map_err(|err| invalid_shape(err.to_string()))
    }
}

const MAX_GATED_GELU_FFN_CACHE_ENTRIES: usize = 4;
const MAX_GATED_GELU_FFN_BATCHED_CACHE_ENTRIES: usize = 10;

fn validate_cache_entry_cap_with_max(
    max_entries: usize,
    hard_max: usize,
) -> Result<(), MediaTekRunError> {
    if !(1..=hard_max).contains(&max_entries) {
        return Err(invalid_shape(format!(
            "max_cache_entries must be in 1..={hard_max}"
        )));
    }
    Ok(())
}

fn validate_cache_entry_cap(max_entries: usize) -> Result<(), MediaTekRunError> {
    validate_cache_entry_cap_with_max(max_entries, MAX_GATED_GELU_FFN_CACHE_ENTRIES)
}

fn validate_batched_cache_entry_cap(max_entries: usize) -> Result<(), MediaTekRunError> {
    validate_cache_entry_cap_with_max(max_entries, MAX_GATED_GELU_FFN_BATCHED_CACHE_ENTRIES)
}

#[cfg(any(target_os = "android", test))]
fn cache_entry_position<K: PartialEq, T>(entries: &[(K, T)], key: &K) -> Option<usize> {
    entries.iter().position(|(cached_key, _)| cached_key == key)
}

#[cfg(target_os = "android")]
fn quantized_cache_entry_position(
    entries: &[(
        GatedGeluFfnF32CacheKey,
        (
            rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnQuantParams,
            rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnCompilation,
        ),
    )],
    key: &GatedGeluFfnF32CacheKey,
    quant_params: rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnQuantParams,
) -> Option<usize> {
    entries.iter().position(|(cached_key, (cached_params, _))| {
        cached_key == key && *cached_params == quant_params
    })
}

#[cfg(any(target_os = "android", test))]
fn promote_cache_entry<K, T>(entries: &mut Vec<(K, T)>, position: usize) {
    if position + 1 < entries.len() {
        let entry = entries.remove(position);
        entries.push(entry);
    }
}

#[cfg(any(target_os = "android", test))]
fn evict_cache_to_capacity<K: Clone, T>(entries: &mut Vec<(K, T)>, max_entries: usize) -> Vec<K> {
    if entries.len() <= max_entries {
        return Vec::new();
    }
    let evict_count = entries.len() - max_entries;
    entries.drain(0..evict_count).map(|(key, _)| key).collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunCachedGatedGeluFfnF32Request<'a> {
    pub cache_key: GatedGeluFfnF32CacheKey,
    pub weights: Option<RunCachedGatedGeluFfnF32Weights>,
    pub max_cache_entries: usize,
    pub measure_timing: bool,
    pub input: &'a [f32],
}

impl RunCachedGatedGeluFfnF32Request<'_> {
    fn validate(&self) -> Result<(), MediaTekRunError> {
        if self.cache_key.device_name_substring.trim().is_empty() {
            return Err(MediaTekRunError::InvalidDeviceName);
        }
        if self.cache_key.input_size == 0 {
            return Err(invalid_shape("input_size must be non-zero"));
        }
        if self.cache_key.ffn_inner_size == 0 {
            return Err(invalid_shape("ffn_inner_size must be non-zero"));
        }
        if self.cache_key.output_size == 0 {
            return Err(invalid_shape("output_size must be non-zero"));
        }
        if self.cache_key.gate.rows != self.cache_key.ffn_inner_size
            || self.cache_key.gate.cols != self.cache_key.input_size
            || self.cache_key.up.rows != self.cache_key.ffn_inner_size
            || self.cache_key.up.cols != self.cache_key.input_size
            || self.cache_key.down.rows != self.cache_key.output_size
            || self.cache_key.down.cols != self.cache_key.ffn_inner_size
        {
            return Err(invalid_shape("cache key weight dimensions mismatch"));
        }
        validate_cache_entry_cap(self.max_cache_entries)?;
        expect_len("input", self.cache_key.input_size, self.input.len())?;
        expect_finite("input", self.input)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunCachedQuantizedGatedGeluFfnRequest<'a> {
    pub cache_key: GatedGeluFfnF32CacheKey,
    pub weights: Option<RunCachedQuantizedGatedGeluFfnWeights>,
    pub max_cache_entries: usize,
    pub measure_timing: bool,
    pub input: &'a [f32],
}

impl RunCachedQuantizedGatedGeluFfnRequest<'_> {
    fn validate(&self) -> Result<(), MediaTekRunError> {
        if self.cache_key.device_name_substring.trim().is_empty() {
            return Err(MediaTekRunError::InvalidDeviceName);
        }
        if self.cache_key.input_size == 0 {
            return Err(invalid_shape("input_size must be non-zero"));
        }
        if self.cache_key.ffn_inner_size == 0 {
            return Err(invalid_shape("ffn_inner_size must be non-zero"));
        }
        if self.cache_key.output_size == 0 {
            return Err(invalid_shape("output_size must be non-zero"));
        }
        if self.cache_key.gate.rows != self.cache_key.ffn_inner_size
            || self.cache_key.gate.cols != self.cache_key.input_size
            || self.cache_key.up.rows != self.cache_key.ffn_inner_size
            || self.cache_key.up.cols != self.cache_key.input_size
            || self.cache_key.down.rows != self.cache_key.output_size
            || self.cache_key.down.cols != self.cache_key.ffn_inner_size
        {
            return Err(invalid_shape("cache key weight dimensions mismatch"));
        }
        validate_cache_entry_cap(self.max_cache_entries)?;
        expect_len("input", self.cache_key.input_size, self.input.len())?;
        expect_finite("input", self.input)?;
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct RunCachedGatedGeluFfnF32BatchedRequest<'a> {
    pub cache_key: GatedGeluFfnF32BatchedCacheKey,
    pub weights: Option<RunCachedGatedGeluFfnF32Weights>,
    pub max_cache_entries: usize,
    pub measure_timing: bool,
    pub input: &'a [f32],
}

impl RunCachedGatedGeluFfnF32BatchedRequest<'_> {
    fn validate(&self) -> Result<(), MediaTekRunError> {
        let base = &self.cache_key.base;
        if base.device_name_substring.trim().is_empty() {
            return Err(MediaTekRunError::InvalidDeviceName);
        }
        if base.input_size == 0 {
            return Err(invalid_shape("input_size must be non-zero"));
        }
        if base.ffn_inner_size == 0 {
            return Err(invalid_shape("ffn_inner_size must be non-zero"));
        }
        if base.output_size == 0 {
            return Err(invalid_shape("output_size must be non-zero"));
        }
        if self.cache_key.batch == 0 {
            return Err(invalid_shape("batch must be non-zero"));
        }
        if base.gate.rows != base.ffn_inner_size
            || base.gate.cols != base.input_size
            || base.up.rows != base.ffn_inner_size
            || base.up.cols != base.input_size
            || base.down.rows != base.output_size
            || base.down.cols != base.ffn_inner_size
        {
            return Err(invalid_shape("cache key weight dimensions mismatch"));
        }
        validate_batched_cache_entry_cap(self.max_cache_entries)?;
        expect_len(
            "input",
            base.input_size * self.cache_key.batch,
            self.input.len(),
        )?;
        expect_finite("input", self.input)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunQuantizedGatedGeluFfnStageProbeRequest<'a> {
    pub device_name_substring: String,
    pub shape: MediaTekGatedGeluFfnShape,
    pub weights: RunCachedQuantizedGatedGeluFfnWeights,
    pub input: &'a [f32],
    pub stage: MediaTekQuantizedGatedGeluFfnStage,
}

impl RunQuantizedGatedGeluFfnStageProbeRequest<'_> {
    fn validate(&self) -> Result<(), MediaTekRunError> {
        if self.device_name_substring.trim().is_empty() {
            return Err(MediaTekRunError::InvalidDeviceName);
        }
        if self.weights.down_weight_f32.is_none() {
            return Err(invalid_shape(
                "quantized stage probe requires hybrid f32-down weights",
            ));
        }
        expect_len("input", self.shape.input_size(), self.input.len())?;
        expect_finite("input", self.input)?;
        self.weights.clone().into_backend(self.shape).map(|_| ())
    }
}

#[cfg(target_os = "android")]
thread_local! {
    static GATED_GELU_FFN_CACHE: RefCell<Vec<(
        GatedGeluFfnF32CacheKey,
        rnb_backend_mediatek::MediaTekGatedGeluFfnCompilation,
    )>> = const { RefCell::new(Vec::new()) };
}
#[cfg(target_os = "android")]
thread_local! {
    static GATED_GELU_FFN_BATCHED_CACHE: RefCell<Vec<(
        GatedGeluFfnF32BatchedCacheKey,
        rnb_backend_mediatek::MediaTekGatedGeluFfnBatchedCompilation,
    )>> = const { RefCell::new(Vec::new()) };
}

#[cfg(target_os = "android")]
thread_local! {
    static QUANTIZED_GATED_GELU_FFN_CACHE: RefCell<Vec<(
        GatedGeluFfnF32CacheKey,
        (
            rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnQuantParams,
            rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnCompilation,
        ),
    )>> = const { RefCell::new(Vec::new()) };
}

#[cfg(any(target_os = "android", test))]
fn cached_execution_options(
    measure_timing: bool,
) -> rnb_backend_mediatek::MediaTekNnapiExecutionOptions {
    rnb_backend_mediatek::MediaTekNnapiExecutionOptions::new(measure_timing)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RunGatedGeluFfnF32Timings {
    pub model_build_ns: u64,
    pub supported_ops_query_ns: u64,
    pub compilation_ns: u64,
    pub execution_setup_ns: u64,
    pub execution_compute_ns: u64,
    pub token_hash_ns: u64,
}

impl RunGatedGeluFfnF32Timings {
    fn from_backend(timings: rnb_backend_mediatek::MediaTekGatedGeluFfnTimings) -> Self {
        Self {
            model_build_ns: timings.model_build_ns(),
            supported_ops_query_ns: timings.supported_ops_query_ns(),
            compilation_ns: timings.compilation_ns(),
            execution_setup_ns: timings.execution_setup_ns(),
            execution_compute_ns: timings.execution_compute_ns(),
            token_hash_ns: timings.token_hash_ns(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunGatedGeluFfnF32Result {
    pub output: Vec<f32>,
    pub chosen_device_name: String,
    pub chosen_device_type: i32,
    pub chosen_device_feature_level: i64,
    pub chosen_device_version: String,
    pub supported_ops: Vec<(&'static str, bool)>,
    pub duration_hardware_ns: Option<u64>,
    pub duration_driver_ns: Option<u64>,
    pub timings: RunGatedGeluFfnF32Timings,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeGatedGeluFfnF32Result {
    pub output: Vec<f32>,
    pub chosen_device_name: String,
    pub chosen_device_type: i32,
    pub chosen_device_feature_level: i64,
    pub chosen_device_version: String,
    pub supported_ops: Vec<(&'static str, bool)>,
    pub duration_hardware_ns: Option<u64>,
    pub duration_driver_ns: Option<u64>,
    pub timings: RunGatedGeluFfnF32Timings,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunQuantizedGatedGeluFfnStageProbeResult {
    pub stage: MediaTekQuantizedGatedGeluFfnStage,
    pub output: Vec<f32>,
    pub chosen_device_name: String,
    pub chosen_device_type: i32,
    pub chosen_device_feature_level: i64,
    pub chosen_device_version: String,
    pub supported_ops: Vec<(&'static str, bool)>,
    pub duration_hardware_ns: Option<u64>,
    pub duration_driver_ns: Option<u64>,
    pub timings: RunGatedGeluFfnF32Timings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaTekRunError {
    InvalidShape {
        reason: String,
    },
    InvalidDeviceName,
    UnsupportedPlatform,
    NoMatchingAccelerator {
        requested: String,
    },
    CpuDeviceRejected {
        name: String,
    },
    UnsupportedOperation {
        supported_ops: Vec<(&'static str, bool)>,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    CacheMissNeedsWeights,
    NnapiCall {
        call: &'static str,
        code: i32,
    },
}

pub type MediaTekProbeError = MediaTekRunError;

impl fmt::Display for MediaTekRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { reason } => write!(f, "invalid MediaTek FFN shape: {reason}"),
            Self::InvalidDeviceName => write!(f, "MediaTek device name substring is empty"),
            Self::UnsupportedPlatform => write!(f, "MediaTek NNAPI execution requires Android"),
            Self::NoMatchingAccelerator { requested } => {
                write!(f, "no MediaTek NNAPI accelerator matched '{requested}'")
            }
            Self::CpuDeviceRejected { name } => {
                write!(f, "NNAPI device '{name}' is CPU/reference")
            }
            Self::UnsupportedOperation { supported_ops } => {
                write!(f, "MediaTek NNAPI gated GELU FFN op support failed: ")?;
                for (idx, (name, supported)) in supported_ops.iter().enumerate() {
                    if idx > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{name}={supported}")?;
                }
                Ok(())
            }
            Self::InvalidOutputLength { expected, actual } => write!(
                f,
                "MediaTek NNAPI output length mismatch: expected {expected}, got {actual}"
            ),
            Self::CacheMissNeedsWeights => write!(
                f,
                "MediaTek NNAPI gated GELU FFN cache miss requires weights"
            ),
            Self::NnapiCall { call, code } => {
                write!(f, "MediaTek NNAPI call {call} failed with code {code}")
            }
        }
    }
}

impl std::error::Error for MediaTekRunError {}

pub fn run_gated_gelu_ffn_f32(
    request: RunGatedGeluFfnF32Request<'_>,
) -> Result<RunGatedGeluFfnF32Result, MediaTekRunError> {
    request.validate()?;

    let shape = rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(
        request.input_size,
        request.ffn_inner_size,
        request.output_size,
    );
    let tensors = rnb_backend_mediatek::MediaTekGatedGeluFfnTensorView::new(
        shape,
        request.gate_weight,
        request.up_weight,
        request.down_weight,
        request.input,
    )
    .map_err(|err| invalid_shape(err.to_string()))?;

    let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
    let output = backend
        .run_gated_gelu_ffn_f32(
            &tensors,
            rnb_backend_mediatek::MediaTekNnapiOptions::new(request.device_name_substring.trim()),
        )
        .map_err(MediaTekRunError::from_backend)?;
    let chosen_device = output.chosen_device();
    Ok(RunGatedGeluFfnF32Result {
        output: output.output().to_vec(),
        chosen_device_name: chosen_device.name().to_string(),
        chosen_device_type: chosen_device.device_type(),
        chosen_device_feature_level: chosen_device.feature_level(),
        chosen_device_version: chosen_device.version().to_string(),
        supported_ops: output.supported_ops().named().to_vec(),
        duration_hardware_ns: output.duration_hardware_ns(),
        duration_driver_ns: output.duration_driver_ns(),
        timings: RunGatedGeluFfnF32Timings::from_backend(output.timings()),
        cache_hit: false,
    })
}

pub fn probe_gated_gelu_ffn_f32(
    request: ProbeGatedGeluFfnF32Request<'_>,
) -> Result<ProbeGatedGeluFfnF32Result, MediaTekProbeError> {
    let output = run_gated_gelu_ffn_f32(request.into())?;
    Ok(ProbeGatedGeluFfnF32Result {
        output: output.output,
        chosen_device_name: output.chosen_device_name,
        chosen_device_type: output.chosen_device_type,
        chosen_device_feature_level: output.chosen_device_feature_level,
        chosen_device_version: output.chosen_device_version,
        supported_ops: output.supported_ops,
        duration_hardware_ns: output.duration_hardware_ns,
        duration_driver_ns: output.duration_driver_ns,
        timings: output.timings,
        cache_hit: output.cache_hit,
    })
}

#[derive(Debug, Clone)]
pub struct ProbeGatedGeluFfnF32BatchedRequest<'a> {
    pub device_name_substring: String,
    pub input_size: usize,
    pub ffn_inner_size: usize,
    pub output_size: usize,
    pub gate_weight: &'a [f32],
    pub up_weight: &'a [f32],
    pub down_weight: &'a [f32],
    pub input: &'a [f32],
    pub batch: usize,
    pub zero_copy: bool,
    pub cache_dir: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeGatedGeluFfnF32BatchedResult {
    pub output: Vec<f32>,
    pub batch: usize,
    pub supported: bool,
    pub supported_ops: Vec<(String, bool)>,
    pub compile_ns: u64,
    pub token_hash_ns: u64,
    pub execute_hw_ns: Option<u64>,
    pub execution_compute_ns: u64,
    pub chosen_device_name: String,
    pub chosen_device_type: i32,
    pub chosen_device_feature_level: i64,
    pub chosen_device_version: String,
}

pub fn probe_gated_gelu_ffn_f32_batched(
    request: ProbeGatedGeluFfnF32BatchedRequest<'_>,
) -> Result<ProbeGatedGeluFfnF32BatchedResult, MediaTekProbeError> {
    let shape = rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(
        request.input_size,
        request.ffn_inner_size,
        request.output_size,
    );
    let tensors = rnb_backend_mediatek::MediaTekGatedGeluFfnBatchedTensorView::new(
        shape,
        request.gate_weight,
        request.up_weight,
        request.down_weight,
        request.input,
    )
    .map_err(|err| invalid_shape(err.to_string()))?;
    let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
    let device =
        rnb_backend_mediatek::MediaTekNnapiOptions::new(request.device_name_substring.trim());
    if request.zero_copy || request.cache_dir.is_some() {
        let owned_weights = rnb_backend_mediatek::MediaTekGatedGeluFfnOwnedWeights::new(
            shape,
            request.gate_weight.to_vec(),
            request.up_weight.to_vec(),
            request.down_weight.to_vec(),
        )
        .map_err(|err| invalid_shape(err.to_string()))?;
        let cache_dir = match request.cache_dir.as_ref() {
            Some(cache_dir) if cache_dir.trim().is_empty() => {
                return Err(invalid_shape("cache_dir must not be empty"));
            }
            _ => request.cache_dir.clone(),
        };
        let compilation = backend
            .compile_gated_gelu_ffn_f32_batched_with(
                owned_weights,
                request.batch,
                device,
                BatchedCompileOptions {
                    zero_copy: request.zero_copy,
                    cache_dir,
                },
            )
            .map_err(MediaTekRunError::from_backend)?;
        let compile_timings = compilation.timings();
        let output = backend
            .run_compiled_gated_gelu_ffn_f32_batched(
                &compilation,
                request.input,
                request.batch,
                rnb_backend_mediatek::MediaTekNnapiExecutionOptions::new(true),
            )
            .map_err(MediaTekRunError::from_backend)?;
        let mut result = probe_gated_gelu_ffn_f32_batched_result_from_backend(output);
        result.compile_ns = compile_timings.compilation_ns();
        result.token_hash_ns = compile_timings.token_hash_ns();
        return Ok(result);
    }
    let output = backend
        .probe_gated_gelu_ffn_f32_batched(&tensors, request.batch, device)
        .map_err(MediaTekRunError::from_backend)?;
    Ok(probe_gated_gelu_ffn_f32_batched_result_from_backend(output))
}

#[allow(clippy::too_many_arguments)]
pub fn compile_gated_gelu_ffn_f32_batched(
    device: &str,
    input_size: usize,
    ffn_inner_size: usize,
    output_size: usize,
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    batch: usize,
) -> Result<MediaTekGatedGeluFfnBatchedCompilation, MediaTekProbeError> {
    compile_gated_gelu_ffn_f32_batched_with(
        device,
        input_size,
        ffn_inner_size,
        output_size,
        gate_weight,
        up_weight,
        down_weight,
        batch,
        BatchedCompileOptions::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn compile_gated_gelu_ffn_f32_batched_with(
    device: &str,
    input_size: usize,
    ffn_inner_size: usize,
    output_size: usize,
    gate_weight: &[f32],
    up_weight: &[f32],
    down_weight: &[f32],
    batch: usize,
    opts: BatchedCompileOptions,
) -> Result<MediaTekGatedGeluFfnBatchedCompilation, MediaTekProbeError> {
    if device.trim().is_empty() {
        return Err(MediaTekRunError::InvalidDeviceName);
    }
    if matches!(opts.cache_dir.as_deref(), Some(cache_dir) if cache_dir.trim().is_empty()) {
        return Err(invalid_shape("cache_dir must not be empty"));
    }
    let shape = rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(
        input_size,
        ffn_inner_size,
        output_size,
    );
    let owned_weights = rnb_backend_mediatek::MediaTekGatedGeluFfnOwnedWeights::new(
        shape,
        gate_weight.to_vec(),
        up_weight.to_vec(),
        down_weight.to_vec(),
    )
    .map_err(|err| invalid_shape(err.to_string()))?;
    let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
    backend
        .compile_gated_gelu_ffn_f32_batched_with(
            owned_weights,
            batch,
            rnb_backend_mediatek::MediaTekNnapiOptions::new(device.trim()),
            opts,
        )
        .map_err(MediaTekRunError::from_backend)
}

pub fn run_compiled_gated_gelu_ffn_f32_batched(
    compilation: &MediaTekGatedGeluFfnBatchedCompilation,
    input: &[f32],
    batch: usize,
) -> Result<ProbeGatedGeluFfnF32BatchedResult, MediaTekProbeError> {
    let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
    let output = backend
        .run_compiled_gated_gelu_ffn_f32_batched(
            compilation,
            input,
            batch,
            rnb_backend_mediatek::MediaTekNnapiExecutionOptions::new(true),
        )
        .map_err(MediaTekRunError::from_backend)?;
    Ok(probe_gated_gelu_ffn_f32_batched_result_from_backend(output))
}

fn probe_gated_gelu_ffn_f32_batched_result_from_backend(
    output: rnb_backend_mediatek::MediaTekGatedGeluFfnBatchedOutput,
) -> ProbeGatedGeluFfnF32BatchedResult {
    let chosen = output.chosen_device();
    ProbeGatedGeluFfnF32BatchedResult {
        output: output.output().to_vec(),
        batch: output.batch(),
        supported: output.supported(),
        supported_ops: output
            .supported_ops()
            .named()
            .iter()
            .map(|(name, supported)| ((*name).to_string(), *supported))
            .collect(),
        compile_ns: output.compile_ns(),
        token_hash_ns: output.timings().token_hash_ns(),
        execute_hw_ns: output.execute_hw_ns(),
        execution_compute_ns: output.timings().execution_compute_ns(),
        chosen_device_name: chosen.name().to_string(),
        chosen_device_type: chosen.device_type(),
        chosen_device_feature_level: chosen.feature_level(),
        chosen_device_version: chosen.version().to_string(),
    }
}

fn trace_batched_cache_dir_disabled_once(reason: &str) {
    static WARNED: OnceLock<()> = OnceLock::new();
    WARNED.get_or_init(|| {
        eprintln!("[mediatek-ffn-batched] aot_cache_dir=disabled reason={reason}");
    });
}

pub fn default_gated_gelu_ffn_batched_cache_dir() -> Option<String> {
    let cache_root = match crate::platform::cache_dir::resolve_cache_dir() {
        Ok(path) => path,
        Err(err) => {
            trace_batched_cache_dir_disabled_once(&err);
            return None;
        }
    };
    let cache_dir = cache_root.join("mediatek").join("gated_gelu_ffn_batched");
    if let Err(err) = std::fs::create_dir_all(&cache_dir) {
        trace_batched_cache_dir_disabled_once(&format!("create {}: {err}", cache_dir.display()));
        return None;
    }
    match cache_dir.to_str() {
        Some(path) => Some(path.to_string()),
        None => {
            trace_batched_cache_dir_disabled_once("cache dir path is not valid utf-8");
            None
        }
    }
}

pub fn run_quantized_gated_gelu_ffn_stage_probe(
    request: RunQuantizedGatedGeluFfnStageProbeRequest<'_>,
) -> Result<RunQuantizedGatedGeluFfnStageProbeResult, MediaTekRunError> {
    request.validate()?;
    let quantized_input = quantize_f32_to_u8(request.input, request.weights.quant_params.input);
    let owned_weights = request.weights.into_backend(request.shape)?;
    let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
    let output = backend
        .run_quantized_gated_gelu_ffn_stage_probe(
            owned_weights,
            &quantized_input,
            request.stage,
            rnb_backend_mediatek::MediaTekNnapiOptions::new(request.device_name_substring.trim()),
        )
        .map_err(MediaTekRunError::from_backend)?;
    let chosen_device = output.chosen_device();
    Ok(RunQuantizedGatedGeluFfnStageProbeResult {
        stage: output.stage(),
        output: output.output().to_vec(),
        chosen_device_name: chosen_device.name().to_string(),
        chosen_device_type: chosen_device.device_type(),
        chosen_device_feature_level: chosen_device.feature_level(),
        chosen_device_version: chosen_device.version().to_string(),
        supported_ops: output.supported_ops().named().to_vec(),
        duration_hardware_ns: output.duration_hardware_ns(),
        duration_driver_ns: output.duration_driver_ns(),
        timings: RunGatedGeluFfnF32Timings::from_backend(output.timings()),
    })
}

pub fn is_gated_gelu_ffn_f32_batched_cached(key: &GatedGeluFfnF32BatchedCacheKey) -> bool {
    #[cfg(target_os = "android")]
    {
        GATED_GELU_FFN_BATCHED_CACHE
            .with(|slot| cache_entry_position(slot.borrow().as_slice(), key).is_some())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = key;
        false
    }
}

pub fn clear_gated_gelu_ffn_f32_batched_cache() {
    #[cfg(target_os = "android")]
    GATED_GELU_FFN_BATCHED_CACHE.with(|slot| {
        slot.borrow_mut().clear();
    });
}

pub fn run_cached_gated_gelu_ffn_f32_batched(
    request: RunCachedGatedGeluFfnF32BatchedRequest<'_>,
) -> Result<RunGatedGeluFfnF32Result, MediaTekRunError> {
    request.validate()?;
    let cache_hit = is_gated_gelu_ffn_f32_batched_cached(&request.cache_key);
    if !cache_hit && request.weights.is_none() {
        return Err(MediaTekRunError::CacheMissNeedsWeights);
    }

    #[cfg(target_os = "android")]
    {
        run_cached_gated_gelu_ffn_f32_batched_android(request, cache_hit)
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = request;
        Err(MediaTekRunError::UnsupportedPlatform)
    }
}
pub fn is_gated_gelu_ffn_f32_cached(key: &GatedGeluFfnF32CacheKey) -> bool {
    #[cfg(target_os = "android")]
    {
        GATED_GELU_FFN_CACHE
            .with(|slot| cache_entry_position(slot.borrow().as_slice(), key).is_some())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = key;
        false
    }
}

pub fn clear_gated_gelu_ffn_f32_cache() {
    #[cfg(target_os = "android")]
    GATED_GELU_FFN_CACHE.with(|slot| {
        slot.borrow_mut().clear();
    });
}

pub fn is_gated_gelu_ffn_quantized_cached(key: &GatedGeluFfnF32CacheKey) -> bool {
    #[cfg(target_os = "android")]
    {
        QUANTIZED_GATED_GELU_FFN_CACHE
            .with(|slot| cache_entry_position(slot.borrow().as_slice(), key).is_some())
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = key;
        false
    }
}

pub fn is_gated_gelu_ffn_quantized_cached_with_params(
    key: &GatedGeluFfnF32CacheKey,
    quant_params: rnb_backend_mediatek::MediaTekQuantizedGatedGeluFfnQuantParams,
) -> bool {
    #[cfg(target_os = "android")]
    {
        QUANTIZED_GATED_GELU_FFN_CACHE.with(|slot| {
            quantized_cache_entry_position(slot.borrow().as_slice(), key, quant_params).is_some()
        })
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (key, quant_params);
        false
    }
}

pub fn clear_gated_gelu_ffn_quantized_cache() {
    #[cfg(target_os = "android")]
    QUANTIZED_GATED_GELU_FFN_CACHE.with(|slot| {
        slot.borrow_mut().clear();
    });
}

pub fn remove_gated_gelu_ffn_quantized_cache_entry(key: &GatedGeluFfnF32CacheKey) {
    #[cfg(target_os = "android")]
    QUANTIZED_GATED_GELU_FFN_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(position) = cache_entry_position(slot.as_slice(), key) {
            slot.remove(position);
        }
    });
    #[cfg(not(target_os = "android"))]
    {
        let _ = key;
    }
}

pub fn run_cached_quantized_gated_gelu_ffn(
    request: RunCachedQuantizedGatedGeluFfnRequest<'_>,
) -> Result<RunGatedGeluFfnF32Result, MediaTekRunError> {
    request.validate()?;
    if request.weights.is_none() {
        return Err(MediaTekRunError::CacheMissNeedsWeights);
    }

    #[cfg(target_os = "android")]
    {
        run_cached_quantized_gated_gelu_ffn_android(request)
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = request;
        Err(MediaTekRunError::UnsupportedPlatform)
    }
}

pub fn run_cached_gated_gelu_ffn_f32(
    request: RunCachedGatedGeluFfnF32Request<'_>,
) -> Result<RunGatedGeluFfnF32Result, MediaTekRunError> {
    request.validate()?;
    let cache_hit = is_gated_gelu_ffn_f32_cached(&request.cache_key);
    if !cache_hit && request.weights.is_none() {
        return Err(MediaTekRunError::CacheMissNeedsWeights);
    }

    #[cfg(target_os = "android")]
    {
        run_cached_gated_gelu_ffn_f32_android(request, cache_hit)
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = request;
        Err(MediaTekRunError::UnsupportedPlatform)
    }
}

#[cfg(target_os = "android")]
fn run_cached_gated_gelu_ffn_f32_android(
    request: RunCachedGatedGeluFfnF32Request<'_>,
    cache_hit: bool,
) -> Result<RunGatedGeluFfnF32Result, MediaTekRunError> {
    GATED_GELU_FFN_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let compile_timings =
            if let Some(position) = cache_entry_position(slot.as_slice(), &request.cache_key) {
                promote_cache_entry(&mut slot, position);
                let _ = evict_cache_to_capacity(&mut slot, request.max_cache_entries);
                RunGatedGeluFfnF32Timings::default()
            } else {
                let shape = rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(
                    request.cache_key.input_size,
                    request.cache_key.ffn_inner_size,
                    request.cache_key.output_size,
                );
                let weights = request
                    .weights
                    .expect("miss without weights was rejected before backend call")
                    .into_backend(shape)?;
                let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
                let compilation = backend
                    .compile_gated_gelu_ffn_f32(
                        weights,
                        rnb_backend_mediatek::MediaTekNnapiOptions::new(
                            request.cache_key.device_name_substring.trim(),
                        ),
                    )
                    .map_err(MediaTekRunError::from_backend)?;
                let timings = RunGatedGeluFfnF32Timings::from_backend(compilation.timings());
                slot.push((request.cache_key.clone(), compilation));
                let _ = evict_cache_to_capacity(&mut slot, request.max_cache_entries);
                timings
            };

        let (cached_key, compilation) = slot
            .last()
            .expect("compiled cache must exist after hit or miss compile");
        debug_assert_eq!(cached_key, &request.cache_key);
        let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
        let output = backend
            .run_compiled_gated_gelu_ffn_f32(
                compilation,
                request.input,
                cached_execution_options(request.measure_timing),
            )
            .map_err(MediaTekRunError::from_backend)?;
        let chosen_device = output.chosen_device();
        let mut timings = RunGatedGeluFfnF32Timings::from_backend(output.timings());
        if !cache_hit {
            timings.model_build_ns = compile_timings.model_build_ns;
            timings.supported_ops_query_ns = compile_timings.supported_ops_query_ns;
            timings.compilation_ns = compile_timings.compilation_ns;
        }
        Ok(RunGatedGeluFfnF32Result {
            output: output.output().to_vec(),
            chosen_device_name: chosen_device.name().to_string(),
            chosen_device_type: chosen_device.device_type(),
            chosen_device_feature_level: chosen_device.feature_level(),
            chosen_device_version: chosen_device.version().to_string(),
            supported_ops: output.supported_ops().named().to_vec(),
            duration_hardware_ns: output.duration_hardware_ns(),
            duration_driver_ns: output.duration_driver_ns(),
            timings,
            cache_hit,
        })
    })
}
#[cfg(target_os = "android")]
fn run_cached_gated_gelu_ffn_f32_batched_android(
    request: RunCachedGatedGeluFfnF32BatchedRequest<'_>,
    cache_hit: bool,
) -> Result<RunGatedGeluFfnF32Result, MediaTekRunError> {
    GATED_GELU_FFN_BATCHED_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let compile_timings =
            if let Some(position) = cache_entry_position(slot.as_slice(), &request.cache_key) {
                promote_cache_entry(&mut slot, position);
                let _ = evict_cache_to_capacity(&mut slot, request.max_cache_entries);
                None
            } else {
                let shape = rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(
                    request.cache_key.base.input_size,
                    request.cache_key.base.ffn_inner_size,
                    request.cache_key.base.output_size,
                );
                let weights = request
                    .weights
                    .expect("miss without weights was rejected before backend call")
                    .into_backend(shape)?;
                let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
                let compilation = backend
                    .compile_gated_gelu_ffn_f32_batched_with(
                        weights,
                        request.cache_key.batch,
                        rnb_backend_mediatek::MediaTekNnapiOptions::new(
                            request.cache_key.base.device_name_substring.trim(),
                        ),
                        BatchedCompileOptions {
                            zero_copy: true,
                            cache_dir: default_gated_gelu_ffn_batched_cache_dir(),
                        },
                    )
                    .map_err(MediaTekRunError::from_backend)?;
                let timings = compilation.timings();
                slot.push((request.cache_key.clone(), compilation));
                let _ = evict_cache_to_capacity(&mut slot, request.max_cache_entries);
                Some(timings)
            };

        let (cached_key, compilation) = slot
            .last()
            .expect("compiled batched cache must exist after hit or miss compile");
        debug_assert_eq!(cached_key, &request.cache_key);
        let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
        let output = backend
            .run_compiled_gated_gelu_ffn_f32_batched(
                compilation,
                request.input,
                request.cache_key.batch,
                cached_execution_options(request.measure_timing),
            )
            .map_err(MediaTekRunError::from_backend)?;
        Ok(run_gated_gelu_ffn_f32_result_from_batched_backend(
            output,
            cache_hit,
            compile_timings,
        ))
    })
}

#[cfg(target_os = "android")]
fn run_cached_quantized_gated_gelu_ffn_android(
    request: RunCachedQuantizedGatedGeluFfnRequest<'_>,
) -> Result<RunGatedGeluFfnF32Result, MediaTekRunError> {
    let RunCachedQuantizedGatedGeluFfnRequest {
        cache_key,
        weights,
        max_cache_entries,
        measure_timing,
        input,
    } = request;
    let weights = weights.expect("quantized cache facade requires weights before backend call");
    let requested_quant_params = weights.quant_params;
    QUANTIZED_GATED_GELU_FFN_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let mut cache_hit = false;
        let compile_timings = if let Some(position) =
            quantized_cache_entry_position(slot.as_slice(), &cache_key, requested_quant_params)
        {
            promote_cache_entry(&mut slot, position);
            let _ = evict_cache_to_capacity(&mut slot, max_cache_entries);
            cache_hit = true;
            RunGatedGeluFfnF32Timings::default()
        } else {
            if let Some(position) = cache_entry_position(slot.as_slice(), &cache_key) {
                slot.remove(position);
            }
            let shape = rnb_backend_mediatek::MediaTekGatedGeluFfnShape::new(
                cache_key.input_size,
                cache_key.ffn_inner_size,
                cache_key.output_size,
            );
            let quant_params = weights.quant_params;
            let weights = weights.into_backend(shape)?;
            let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
            let compilation = backend
                .compile_quantized_gated_gelu_ffn(
                    weights,
                    rnb_backend_mediatek::MediaTekNnapiOptions::new(
                        cache_key.device_name_substring.trim(),
                    ),
                )
                .map_err(MediaTekRunError::from_backend)?;
            let timings = RunGatedGeluFfnF32Timings::from_backend(compilation.timings());
            slot.push((cache_key.clone(), (quant_params, compilation)));
            let _ = evict_cache_to_capacity(&mut slot, max_cache_entries);
            timings
        };

        let (cached_key, (quant_params, compilation)) = slot
            .last()
            .expect("compiled quantized cache must exist after hit or miss compile");
        debug_assert_eq!(cached_key, &cache_key);
        let input = quantize_f32_to_u8(input, quant_params.input);
        let mut backend = rnb_backend_mediatek::MediaTekBackend::new();
        let output = backend
            .run_compiled_quantized_gated_gelu_ffn(
                compilation,
                &input,
                cached_execution_options(measure_timing),
            )
            .map_err(MediaTekRunError::from_backend)?;
        let chosen_device = output.chosen_device();
        let mut timings = RunGatedGeluFfnF32Timings::from_backend(output.timings());
        if !cache_hit {
            timings.model_build_ns = compile_timings.model_build_ns;
            timings.supported_ops_query_ns = compile_timings.supported_ops_query_ns;
            timings.compilation_ns = compile_timings.compilation_ns;
        }
        Ok(RunGatedGeluFfnF32Result {
            output: output.output().to_vec(),
            chosen_device_name: chosen_device.name().to_string(),
            chosen_device_type: chosen_device.device_type(),
            chosen_device_feature_level: chosen_device.feature_level(),
            chosen_device_version: chosen_device.version().to_string(),
            supported_ops: output.supported_ops().named().to_vec(),
            duration_hardware_ns: output.duration_hardware_ns(),
            duration_driver_ns: output.duration_driver_ns(),
            timings,
            cache_hit,
        })
    })
}

impl MediaTekRunError {
    fn from_backend(err: rnb_backend_mediatek::MediaTekNnapiError) -> Self {
        match err {
            rnb_backend_mediatek::MediaTekNnapiError::UnsupportedPlatform => {
                Self::UnsupportedPlatform
            }
            rnb_backend_mediatek::MediaTekNnapiError::NoMatchingAccelerator { requested } => {
                Self::NoMatchingAccelerator { requested }
            }
            rnb_backend_mediatek::MediaTekNnapiError::CpuDeviceRejected { name } => {
                Self::CpuDeviceRejected { name }
            }
            rnb_backend_mediatek::MediaTekNnapiError::UnsupportedOperation { fc1, fc2 } => {
                Self::UnsupportedOperation {
                    supported_ops: vec![("fc1", fc1), ("fc2", fc2)],
                }
            }
            rnb_backend_mediatek::MediaTekNnapiError::UnsupportedGatedGeluFfnOperation {
                supported_ops,
            } => Self::UnsupportedOperation {
                supported_ops: supported_ops.named().to_vec(),
            },
            rnb_backend_mediatek::MediaTekNnapiError::InvalidOutputLength { expected, actual } => {
                Self::InvalidOutputLength { expected, actual }
            }
            rnb_backend_mediatek::MediaTekNnapiError::InvalidInput { reason } => {
                Self::InvalidShape { reason }
            }
            rnb_backend_mediatek::MediaTekNnapiError::NnapiCall { call, code } => {
                Self::NnapiCall { call, code }
            }
        }
    }
}

fn expect_len(
    tensor: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), MediaTekRunError> {
    if expected != actual {
        return Err(invalid_shape(format!(
            "{tensor} length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn expect_finite(tensor: &'static str, data: &[f32]) -> Result<(), MediaTekRunError> {
    for (index, value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(invalid_shape(format!(
                "{tensor} has non-finite value at index {index}"
            )));
        }
    }
    Ok(())
}

fn quantize_f32_to_u8(data: &[f32], params: rnb_backend_mediatek::MediaTekQuantParams) -> Vec<u8> {
    data.iter()
        .map(|value| {
            let quantized = (*value / params.scale()) + params.zero_point() as f32;
            quantized.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn invalid_shape(reason: impl Into<String>) -> MediaTekRunError {
    MediaTekRunError::InvalidShape {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    fn clear_prefill_policy_env() {
        unsafe {
            std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS");
            std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_LAYER");
            std::env::remove_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS");
        }
    }

    fn prefill_policy_input(
        cache_state: MediaTekPrefillCacheState,
        request_mode: MediaTekPrefillRequestMode,
        sync_compile_enabled: bool,
    ) -> MediaTekPrefillPolicyInput {
        MediaTekPrefillPolicyInput {
            layer_idx: 0,
            seq_len: 384,
            cache_state,
            request_mode,
            sync_compile_enabled,
        }
    }

    #[test]
    fn prefill_policy_user_path_uses_only_thread_local_hot_cache() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        clear_prefill_policy_env();

        let hot = decide_gemma_prefill_npu(prefill_policy_input(
            MediaTekPrefillCacheState::InProcessThreadLocalHot,
            MediaTekPrefillRequestMode::UserPath,
            false,
        ));
        assert_eq!(
            hot,
            MediaTekPrefillNpuDecision::UseWarmNpu {
                max_cache_entries: 2
            }
        );

        for cache_state in [
            MediaTekPrefillCacheState::ColdMissing,
            MediaTekPrefillCacheState::DiskAotPresent,
        ] {
            let decision = decide_gemma_prefill_npu(prefill_policy_input(
                cache_state,
                MediaTekPrefillRequestMode::UserPath,
                true,
            ));
            assert_eq!(
                decision,
                MediaTekPrefillNpuDecision::FallbackCpu {
                    reason: MediaTekPrefillFallbackReason::CacheMissUserPath
                }
            );
        }
    }

    #[test]
    fn prefill_policy_explicit_prewarm_requires_sync_compile_opt_in() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        clear_prefill_policy_env();

        let disabled = decide_gemma_prefill_npu(prefill_policy_input(
            MediaTekPrefillCacheState::ColdMissing,
            MediaTekPrefillRequestMode::ExplicitPrewarm,
            false,
        ));
        assert_eq!(
            disabled,
            MediaTekPrefillNpuDecision::FallbackCpu {
                reason: MediaTekPrefillFallbackReason::SyncCompileDisabled
            }
        );

        let enabled = decide_gemma_prefill_npu(prefill_policy_input(
            MediaTekPrefillCacheState::DiskAotPresent,
            MediaTekPrefillRequestMode::ExplicitPrewarm,
            true,
        ));
        assert_eq!(
            enabled,
            MediaTekPrefillNpuDecision::AllowPrewarmCompile {
                max_cache_entries: 2
            }
        );
    }

    #[test]
    fn prefill_policy_bench_warmup_is_routed_and_invalid_user_mode_is_rejected() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        clear_prefill_policy_env();

        let bench = decide_gemma_prefill_npu(prefill_policy_input(
            MediaTekPrefillCacheState::ColdMissing,
            MediaTekPrefillRequestMode::BenchWarmup,
            true,
        ));
        assert_eq!(
            bench,
            MediaTekPrefillNpuDecision::AllowPrewarmCompile {
                max_cache_entries: 2
            }
        );

        assert_eq!(
            MediaTekPrefillFallbackReason::CacheMissUserPath.trace_reason(),
            "prefill_cache_miss_user_path"
        );
    }

    #[test]
    fn prefill_policy_layer_list_and_cap_are_runtime_owned() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        clear_prefill_policy_env();
        unsafe {
            std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_LAYERS", "1,2");
            std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS", "2");
        }

        let layer0 = decide_gemma_prefill_npu(prefill_policy_input(
            MediaTekPrefillCacheState::InProcessThreadLocalHot,
            MediaTekPrefillRequestMode::UserPath,
            false,
        ));
        assert_eq!(
            layer0,
            MediaTekPrefillNpuDecision::FallbackCpu {
                reason: MediaTekPrefillFallbackReason::LayerNotSelected
            }
        );

        unsafe {
            std::env::set_var("RNB_MEDIATEK_GEMMA_FFN_CACHE_MAX_LAYERS", "1");
        }
        let over_cap = decide_gemma_prefill_npu(MediaTekPrefillPolicyInput {
            layer_idx: 1,
            ..prefill_policy_input(
                MediaTekPrefillCacheState::InProcessThreadLocalHot,
                MediaTekPrefillRequestMode::UserPath,
                false,
            )
        });
        assert_eq!(
            over_cap,
            MediaTekPrefillNpuDecision::FallbackCpu {
                reason: MediaTekPrefillFallbackReason::ResidentCapExceeded
            }
        );

        clear_prefill_policy_env();
    }

    #[test]
    fn prefill_policy_cache_state_derivation_keeps_disk_aot_non_hot() {
        assert_eq!(
            classify_gemma_prefill_cache_state(true, false),
            MediaTekPrefillCacheState::InProcessThreadLocalHot
        );
        assert_eq!(
            classify_gemma_prefill_cache_state(false, true),
            MediaTekPrefillCacheState::DiskAotPresent
        );
        assert_eq!(
            classify_gemma_prefill_cache_state(false, false),
            MediaTekPrefillCacheState::ColdMissing
        );

        let _guard = env_lock().lock().expect("env lock poisoned");
        clear_prefill_policy_env();
        for cache_state in [
            classify_gemma_prefill_cache_state(false, true),
            classify_gemma_prefill_cache_state(false, false),
        ] {
            let decision = decide_gemma_prefill_npu(prefill_policy_input(
                cache_state,
                MediaTekPrefillRequestMode::UserPath,
                false,
            ));
            assert_eq!(
                decision,
                MediaTekPrefillNpuDecision::FallbackCpu {
                    reason: MediaTekPrefillFallbackReason::CacheMissUserPath
                }
            );
        }
    }

    #[test]
    fn facade_rejects_empty_device_name_before_backend_call() {
        let err = ProbeGatedGeluFfnF32Request {
            device_name_substring: "  ".to_string(),
            input_size: 4,
            ffn_inner_size: 3,
            output_size: 2,
            gate_weight: &[0.1; 12],
            up_weight: &[0.2; 12],
            down_weight: &[0.3; 6],
            input: &[0.4; 4],
        }
        .validate()
        .expect_err("blank device name must be rejected");

        assert!(matches!(err, MediaTekProbeError::InvalidDeviceName));
    }

    #[test]
    fn facade_rejects_nonfinite_input_before_backend_call() {
        let input = [0.0, f32::INFINITY, 0.0, 0.0];
        let err = ProbeGatedGeluFfnF32Request {
            device_name_substring: "mtk-neuron".to_string(),
            input_size: 4,
            ffn_inner_size: 3,
            output_size: 2,
            gate_weight: &[0.1; 12],
            up_weight: &[0.2; 12],
            down_weight: &[0.3; 6],
            input: &input,
        }
        .validate()
        .expect_err("non-finite input must be rejected");

        assert!(matches!(
            err,
            MediaTekProbeError::InvalidShape { reason } if reason.contains("input")
        ));
    }

    #[test]
    fn run_facade_rejects_empty_device_name_before_backend_call() {
        let err = RunGatedGeluFfnF32Request {
            device_name_substring: "  ".to_string(),
            input_size: 4,
            ffn_inner_size: 3,
            output_size: 2,
            gate_weight: &[0.1; 12],
            up_weight: &[0.2; 12],
            down_weight: &[0.3; 6],
            input: &[0.4; 4],
        }
        .validate()
        .expect_err("blank device name must be rejected");

        assert!(matches!(err, MediaTekRunError::InvalidDeviceName));
    }

    #[test]
    fn run_facade_keeps_static_supported_op_labels() {
        let err = MediaTekRunError::UnsupportedOperation {
            supported_ops: vec![("gate_fc", true), ("down_fc", false)],
        };

        assert_eq!(
            err.to_string(),
            "MediaTek NNAPI gated GELU FFN op support failed: gate_fc=true down_fc=false"
        );
    }
    #[test]
    fn run_facade_timing_breakdown_preserves_backend_fields() {
        let backend = rnb_backend_mediatek::MediaTekGatedGeluFfnTimings::new_with_token_hash(
            11, 22, 33, 44, 55, 66,
        );
        let timings = RunGatedGeluFfnF32Timings::from_backend(backend);

        assert_eq!(timings.model_build_ns, 11);
        assert_eq!(timings.supported_ops_query_ns, 22);
        assert_eq!(timings.compilation_ns, 33);
        assert_eq!(timings.execution_setup_ns, 44);
        assert_eq!(timings.execution_compute_ns, 55);
        assert_eq!(timings.token_hash_ns, 66);
    }

    #[test]
    fn batched_cached_result_uses_compile_token_hash_timing() {
        let output_timings = rnb_backend_mediatek::MediaTekGatedGeluFfnTimings::new_with_token_hash(
            1, 2, 3, 4, 5, 6,
        );
        let compile_timings =
            rnb_backend_mediatek::MediaTekGatedGeluFfnTimings::new_with_token_hash(
                10, 20, 30, 40, 50, 60,
            );
        let output = rnb_backend_mediatek::MediaTekGatedGeluFfnBatchedOutput::new(
            vec![0.25, -0.5],
            1,
            rnb_backend_mediatek::MediaTekNnapiDeviceInfo::new("mtk-neuron", 1, 1000008, "test"),
            rnb_backend_mediatek::MediaTekGatedGeluFfnSupportedOps::all(true),
            Some(70),
            Some(80),
            output_timings,
        );

        let result = run_gated_gelu_ffn_f32_result_from_batched_backend(
            output,
            false,
            Some(compile_timings),
        );

        assert_eq!(result.timings.model_build_ns, 10);
        assert_eq!(result.timings.supported_ops_query_ns, 20);
        assert_eq!(result.timings.compilation_ns, 30);
        assert_eq!(result.timings.execution_setup_ns, 4);
        assert_eq!(result.timings.execution_compute_ns, 5);
        assert_eq!(result.timings.token_hash_ns, 60);
    }

    #[test]
    fn cache_key_distinguishes_weight_generation_ids() {
        let base = GatedGeluFfnF32WeightKey {
            generation_id: 1,
            raw_ptr: 0x1000,
            raw_len: 144,
            rows: 3,
            cols: 4,
            ggml_type: 14,
        };
        let other_generation = GatedGeluFfnF32WeightKey {
            generation_id: 2,
            ..base
        };

        assert_ne!(base, other_generation);
    }

    #[test]
    fn cache_key_distinguishes_layer_ids() {
        let layer0 = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        let layer1 = GatedGeluFfnF32CacheKey::for_test(1, 4, 3, 2);

        assert_ne!(layer0, layer1);
    }
    #[test]
    fn batched_cache_key_distinguishes_batch_sizes() {
        let batch128 = GatedGeluFfnF32BatchedCacheKey::for_test(0, 4, 3, 2, 128);
        let batch256 = GatedGeluFfnF32BatchedCacheKey::for_test(0, 4, 3, 2, 256);

        assert_ne!(batch128, batch256);
    }

    #[test]
    fn cached_lookup_and_clear_are_idempotent() {
        let key = GatedGeluFfnF32CacheKey {
            device_name_substring: "mtk-neuron".to_string(),
            layer_idx: 0,
            input_size: 4,
            ffn_inner_size: 3,
            output_size: 2,
            gate: GatedGeluFfnF32WeightKey {
                generation_id: 1,
                raw_ptr: 0x1000,
                raw_len: 144,
                rows: 3,
                cols: 4,
                ggml_type: 14,
            },
            up: GatedGeluFfnF32WeightKey {
                generation_id: 2,
                raw_ptr: 0x2000,
                raw_len: 144,
                rows: 3,
                cols: 4,
                ggml_type: 14,
            },
            down: GatedGeluFfnF32WeightKey {
                generation_id: 3,
                raw_ptr: 0x3000,
                raw_len: 210,
                rows: 2,
                cols: 3,
                ggml_type: 14,
            },
        };

        clear_gated_gelu_ffn_f32_cache();
        clear_gated_gelu_ffn_f32_cache();
        assert!(!is_gated_gelu_ffn_f32_cached(&key));
    }
    #[test]
    fn batched_cached_lookup_and_clear_are_idempotent() {
        let key = GatedGeluFfnF32BatchedCacheKey::for_test(0, 4, 3, 2, 128);

        clear_gated_gelu_ffn_f32_batched_cache();
        clear_gated_gelu_ffn_f32_batched_cache();
        assert!(!is_gated_gelu_ffn_f32_batched_cached(&key));
    }

    #[test]
    fn batched_cache_dir_helper_uses_rnb_cache_dir_root() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let root =
            std::env::temp_dir().join(format!("rnb-runtime-mediatek-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("temp cache root must exist");
        unsafe {
            std::env::set_var("RNB_CACHE_DIR", &root);
        }

        let resolved = default_gated_gelu_ffn_batched_cache_dir()
            .expect("helper must resolve subdir when RNB_CACHE_DIR is set");
        let resolved_path = std::path::PathBuf::from(&resolved);
        let expected = root.join("mediatek").join("gated_gelu_ffn_batched");

        assert_eq!(resolved_path, expected);
        assert!(resolved_path.is_dir());

        unsafe {
            std::env::remove_var("RNB_CACHE_DIR");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cached_facade_requires_weights_on_miss_before_backend_call() {
        let key = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        clear_gated_gelu_ffn_f32_cache();

        let err = run_cached_gated_gelu_ffn_f32(RunCachedGatedGeluFfnF32Request {
            cache_key: key,
            weights: None,
            max_cache_entries: 1,
            measure_timing: true,
            input: &[0.4; 4],
        })
        .expect_err("miss without weights must be rejected before backend call");

        assert!(matches!(err, MediaTekRunError::CacheMissNeedsWeights));
    }
    #[test]
    fn batched_cached_facade_requires_weights_on_miss_before_backend_call() {
        let key = GatedGeluFfnF32BatchedCacheKey::for_test(0, 4, 3, 2, 128);
        clear_gated_gelu_ffn_f32_batched_cache();

        let err = run_cached_gated_gelu_ffn_f32_batched(RunCachedGatedGeluFfnF32BatchedRequest {
            cache_key: key,
            weights: None,
            max_cache_entries: 1,
            measure_timing: true,
            input: &[0.4; 512],
        })
        .expect_err("batched miss without weights must be rejected before backend call");

        assert!(matches!(err, MediaTekRunError::CacheMissNeedsWeights));
    }

    #[test]
    fn batched_probe_rejects_empty_cache_dir_before_backend_call() {
        let err = probe_gated_gelu_ffn_f32_batched(ProbeGatedGeluFfnF32BatchedRequest {
            device_name_substring: "mtk-neuron".to_string(),
            input_size: 4,
            ffn_inner_size: 3,
            output_size: 2,
            gate_weight: &[0.1; 12],
            up_weight: &[0.2; 12],
            down_weight: &[0.3; 6],
            input: &[0.4; 8],
            batch: 2,
            zero_copy: true,
            cache_dir: Some("  ".to_string()),
        })
        .expect_err("blank cache dir must be rejected before backend call");

        assert!(matches!(
            err,
            MediaTekRunError::InvalidShape { reason } if reason.contains("cache_dir")
        ));
    }

    #[test]
    fn quantized_cached_facade_requires_weights_on_miss_before_backend_call() {
        let key = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        clear_gated_gelu_ffn_quantized_cache();

        let err = run_cached_quantized_gated_gelu_ffn(RunCachedQuantizedGatedGeluFfnRequest {
            cache_key: key,
            weights: None,
            max_cache_entries: 1,
            measure_timing: true,
            input: &[0.4; 4],
        })
        .expect_err("quantized miss without weights must be rejected before backend call");

        assert!(matches!(err, MediaTekRunError::CacheMissNeedsWeights));
    }

    fn test_quantized_stage_params() -> MediaTekQuantizedGatedGeluFfnQuantParams {
        let q = MediaTekQuantParams::new(0.5, 128);
        let pos = MediaTekQuantParams::new(0.25, 0);
        MediaTekQuantizedGatedGeluFfnQuantParams::new(
            q, q, q, q, q, pos, q, pos, q, q, pos, q, q, pos, pos, q, pos, q, q, q, q,
        )
    }

    fn valid_quantized_stage_probe_weights() -> RunCachedQuantizedGatedGeluFfnWeights {
        RunCachedQuantizedGatedGeluFfnWeights {
            quant_params: test_quantized_stage_params(),
            gate_weight: vec![128; 12],
            up_weight: vec![128; 12],
            down_weight: vec![128; 6],
            down_weight_f32: Some(vec![0.0; 6]),
        }
    }

    #[test]
    fn quantized_stage_probe_request_rejects_empty_device() {
        let err = RunQuantizedGatedGeluFfnStageProbeRequest {
            device_name_substring: "  ".to_string(),
            shape: MediaTekGatedGeluFfnShape::new(4, 3, 2),
            weights: valid_quantized_stage_probe_weights(),
            input: &[0.0; 4],
            stage: MediaTekQuantizedGatedGeluFfnStage::GateFc,
        }
        .validate()
        .expect_err("blank device must be rejected");

        assert!(matches!(err, MediaTekRunError::InvalidDeviceName));
    }

    #[test]
    fn quantized_stage_probe_request_rejects_missing_hybrid_down() {
        let mut weights = valid_quantized_stage_probe_weights();
        weights.down_weight_f32 = None;
        let err = RunQuantizedGatedGeluFfnStageProbeRequest {
            device_name_substring: "mtk-neuron".to_string(),
            shape: MediaTekGatedGeluFfnShape::new(4, 3, 2),
            weights,
            input: &[0.0; 4],
            stage: MediaTekQuantizedGatedGeluFfnStage::GateFc,
        }
        .validate()
        .expect_err("missing hybrid down must be rejected");

        assert!(matches!(
            err,
            MediaTekRunError::InvalidShape { reason } if reason.contains("hybrid f32-down")
        ));
    }

    #[test]
    fn quantized_stage_probe_request_rejects_invalid_shape_before_backend_call() {
        let err = RunQuantizedGatedGeluFfnStageProbeRequest {
            device_name_substring: "mtk-neuron".to_string(),
            shape: MediaTekGatedGeluFfnShape::new(0, 3, 2),
            weights: valid_quantized_stage_probe_weights(),
            input: &[0.0; 4],
            stage: MediaTekQuantizedGatedGeluFfnStage::GateFc,
        }
        .validate()
        .expect_err("invalid shape must be rejected");

        assert!(matches!(err, MediaTekRunError::InvalidShape { .. }));
    }

    #[test]
    fn quantized_stage_probe_request_rejects_invalid_weights_before_backend_call() {
        let mut weights = valid_quantized_stage_probe_weights();
        weights.gate_weight.pop();
        let err = RunQuantizedGatedGeluFfnStageProbeRequest {
            device_name_substring: "mtk-neuron".to_string(),
            shape: MediaTekGatedGeluFfnShape::new(4, 3, 2),
            weights,
            input: &[0.0; 4],
            stage: MediaTekQuantizedGatedGeluFfnStage::GateFc,
        }
        .validate()
        .expect_err("invalid weights must be rejected");

        assert!(matches!(err, MediaTekRunError::InvalidShape { .. }));
    }

    #[test]
    fn quantized_stage_probe_request_quantizes_input() {
        let request = RunQuantizedGatedGeluFfnStageProbeRequest {
            device_name_substring: "mtk-neuron".to_string(),
            shape: MediaTekGatedGeluFfnShape::new(4, 3, 2),
            weights: valid_quantized_stage_probe_weights(),
            input: &[-64.0, -0.5, 0.0, 64.0],
            stage: MediaTekQuantizedGatedGeluFfnStage::GateFc,
        };

        let quantized = quantize_f32_to_u8(request.input, request.weights.quant_params.input);
        assert_eq!(quantized, vec![0, 127, 128, 255]);
    }

    #[test]
    fn quantized_stage_probe_does_not_mutate_quantized_cache() {
        let key = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        clear_gated_gelu_ffn_quantized_cache();
        let result =
            run_quantized_gated_gelu_ffn_stage_probe(RunQuantizedGatedGeluFfnStageProbeRequest {
                device_name_substring: "mtk-neuron".to_string(),
                shape: MediaTekGatedGeluFfnShape::new(4, 3, 2),
                weights: valid_quantized_stage_probe_weights(),
                input: &[0.0; 4],
                stage: MediaTekQuantizedGatedGeluFfnStage::GateFc,
            });

        #[cfg(not(target_os = "android"))]
        assert!(matches!(result, Err(MediaTekRunError::UnsupportedPlatform)));
        assert!(!is_gated_gelu_ffn_quantized_cached(&key));
    }

    #[test]
    fn quantized_input_quantizer_uses_backend_params() {
        let params = rnb_backend_mediatek::MediaTekQuantParams::new(0.5, 128);

        let quantized = quantize_f32_to_u8(&[-64.0, -0.5, 0.0, 0.5, 64.0], params);

        assert_eq!(quantized, vec![0, 127, 128, 129, 255]);
    }

    #[test]
    fn cached_facade_request_carries_measure_timing_policy() {
        let key = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        let request = RunCachedGatedGeluFfnF32Request {
            cache_key: key,
            weights: None,
            max_cache_entries: 1,
            measure_timing: false,
            input: &[0.4; 4],
        };

        assert!(!request.measure_timing);
        request
            .validate()
            .expect("measure flag must not affect validation");
    }
    #[test]
    fn batched_cached_facade_request_carries_measure_timing_policy() {
        let key = GatedGeluFfnF32BatchedCacheKey::for_test(0, 4, 3, 2, 128);
        let request = RunCachedGatedGeluFfnF32BatchedRequest {
            cache_key: key,
            weights: None,
            max_cache_entries: 1,
            measure_timing: false,
            input: &[0.4; 512],
        };

        assert!(!request.measure_timing);
        request
            .validate()
            .expect("measure flag must not affect batched validation");
    }

    #[test]
    fn cached_execution_options_preserves_measure_timing_policy() {
        assert!(cached_execution_options(true).measure_timing());
        assert!(!cached_execution_options(false).measure_timing());
    }

    #[test]
    fn cached_facade_policy_rejects_zero_cache_entries() {
        let key = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        let err = RunCachedGatedGeluFfnF32Request {
            cache_key: key,
            weights: None,
            input: &[0.4; 4],
            max_cache_entries: 0,
            measure_timing: true,
        }
        .validate()
        .expect_err("zero cache cap must be rejected before backend call");

        assert!(matches!(
            err,
            MediaTekRunError::InvalidShape { reason } if reason.contains("max_cache_entries")
        ));
    }
    #[test]
    fn batched_cached_facade_policy_rejects_zero_cache_entries() {
        let key = GatedGeluFfnF32BatchedCacheKey::for_test(0, 4, 3, 2, 128);
        let err = RunCachedGatedGeluFfnF32BatchedRequest {
            cache_key: key,
            weights: None,
            input: &[0.4; 512],
            max_cache_entries: 0,
            measure_timing: true,
        }
        .validate()
        .expect_err("zero batched cache cap must be rejected before backend call");

        assert!(matches!(
            err,
            MediaTekRunError::InvalidShape { reason } if reason.contains("max_cache_entries")
        ));
    }

    #[test]
    fn cached_facade_policy_rejects_above_hard_max_cache_entries() {
        let key = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        let err = RunCachedGatedGeluFfnF32Request {
            cache_key: key,
            weights: None,
            input: &[0.4; 4],
            max_cache_entries: 5,
            measure_timing: true,
        }
        .validate()
        .expect_err("cache cap above hard max must be rejected before backend call");

        assert!(matches!(
            err,
            MediaTekRunError::InvalidShape { reason } if reason.contains("max_cache_entries")
        ));
    }

    #[test]
    fn batched_cached_facade_policy_accepts_ten_cache_entries() {
        let key = GatedGeluFfnF32BatchedCacheKey::for_test(0, 4, 3, 2, 128);
        RunCachedGatedGeluFfnF32BatchedRequest {
            cache_key: key,
            weights: None,
            input: &[0.4; 512],
            max_cache_entries: 10,
            measure_timing: true,
        }
        .validate()
        .expect("ten batched cache entries must be accepted");
    }
    #[test]
    fn batched_cached_facade_policy_rejects_above_hard_max_cache_entries() {
        let key = GatedGeluFfnF32BatchedCacheKey::for_test(0, 4, 3, 2, 128);
        let err = RunCachedGatedGeluFfnF32BatchedRequest {
            cache_key: key,
            weights: None,
            input: &[0.4; 512],
            max_cache_entries: 11,
            measure_timing: true,
        }
        .validate()
        .expect_err("batched cache cap above hard max must be rejected before backend call");

        assert!(matches!(
            err,
            MediaTekRunError::InvalidShape { reason } if reason.contains("max_cache_entries")
        ));
    }

    #[test]
    fn cache_order_promotes_hits_and_evicts_oldest() {
        let key0 = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        let key1 = GatedGeluFfnF32CacheKey::for_test(1, 4, 3, 2);
        let key2 = GatedGeluFfnF32CacheKey::for_test(2, 4, 3, 2);
        let mut entries = vec![(key0.clone(), "layer0"), (key1.clone(), "layer1")];

        assert_eq!(cache_entry_position(&entries, &key0), Some(0));
        promote_cache_entry(&mut entries, 0);
        assert_eq!(entries[0].0, key1);
        assert_eq!(entries[1].0, key0);

        entries.push((key2.clone(), "layer2"));
        let evicted = evict_cache_to_capacity(&mut entries, 2);

        assert_eq!(evicted, vec![key1]);
        assert_eq!(entries[0].0, key0);
        assert_eq!(entries[1].0, key2);
    }

    #[test]
    fn cache_order_preserves_hit_when_trimming_to_smaller_cap() {
        let key0 = GatedGeluFfnF32CacheKey::for_test(0, 4, 3, 2);
        let key1 = GatedGeluFfnF32CacheKey::for_test(1, 4, 3, 2);
        let mut entries = vec![(key0.clone(), "layer0"), (key1.clone(), "layer1")];

        let position = cache_entry_position(&entries, &key0).expect("key0 is cached");
        promote_cache_entry(&mut entries, position);
        let evicted = evict_cache_to_capacity(&mut entries, 1);

        assert_eq!(evicted, vec![key1]);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, key0);
    }

    #[test]
    fn batched_probe_request_carries_zero_copy_flag_and_host_rejects_platform() {
        let request = ProbeGatedGeluFfnF32BatchedRequest {
            device_name_substring: "mtk-neuron".to_string(),
            input_size: 4,
            ffn_inner_size: 3,
            output_size: 2,
            gate_weight: &[0.1; 12],
            up_weight: &[0.2; 12],
            down_weight: &[0.3; 6],
            input: &[0.4; 8],
            batch: 2,
            zero_copy: true,
            cache_dir: None,
        };
        assert!(request.zero_copy);

        let result = probe_gated_gelu_ffn_f32_batched(request);

        #[cfg(not(target_os = "android"))]
        assert!(matches!(result, Err(MediaTekRunError::UnsupportedPlatform)));
    }

    #[test]
    fn batched_compile_with_facade_returns_host_unsupported_platform() {
        let result = compile_gated_gelu_ffn_f32_batched_with(
            "mtk-neuron",
            4,
            3,
            2,
            &[0.1; 12],
            &[0.2; 12],
            &[0.3; 6],
            2,
            BatchedCompileOptions {
                zero_copy: true,
                cache_dir: Some("/tmp/rnb-mk28-cache".to_string()),
            },
        );

        #[cfg(not(target_os = "android"))]
        assert!(matches!(result, Err(MediaTekRunError::UnsupportedPlatform)));
    }

    #[test]
    fn batched_compile_with_facade_rejects_empty_cache_dir_before_backend_call() {
        let result = compile_gated_gelu_ffn_f32_batched_with(
            "mtk-neuron",
            4,
            3,
            2,
            &[0.1; 12],
            &[0.2; 12],
            &[0.3; 6],
            2,
            BatchedCompileOptions {
                zero_copy: false,
                cache_dir: Some(" ".to_string()),
            },
        );

        assert!(matches!(result, Err(MediaTekRunError::InvalidShape { .. })));
    }

    #[test]
    fn run_facade_returns_host_unsupported_platform() {
        let request = RunGatedGeluFfnF32Request {
            device_name_substring: "mtk-neuron".to_string(),
            input_size: 4,
            ffn_inner_size: 3,
            output_size: 2,
            gate_weight: &[0.1; 12],
            up_weight: &[0.2; 12],
            down_weight: &[0.3; 6],
            input: &[0.4; 4],
        };

        let result = run_gated_gelu_ffn_f32(request);

        #[cfg(not(target_os = "android"))]
        assert!(matches!(result, Err(MediaTekRunError::UnsupportedPlatform)));
        #[cfg(target_os = "android")]
        assert!(
            !matches!(result, Err(MediaTekRunError::UnsupportedPlatform)),
            "Android target should attempt backend execution"
        );
    }
}
