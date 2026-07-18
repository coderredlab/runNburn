use rnb_backend_api::{
    Backend, BackendCapabilities, BackendError, BackendKind, BackendOutput, BackendRequest,
    BackendResult,
};
use std::fmt;

pub const MEDIATEK_NNAPI_DEVICE_ACCELERATOR: i32 = 4;

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
const NNAPI_CACHE_TOKEN_LEN: usize = 32;
pub const MK28_CACHE_ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BatchedCompileOptions {
    pub zero_copy: bool,
    pub cache_dir: Option<String>,
}

#[derive(Debug, Default)]
pub struct MediaTekBackend;

impl MediaTekBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn run_quantized_mlp(
        &mut self,
        tensors: &MediaTekQuantizedMlpTensorView<'_>,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekQuantizedMlpOutput, MediaTekNnapiError> {
        #[cfg(target_os = "android")]
        {
            android_nnapi::run_quantized_mlp(tensors, options)
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = tensors;
            let _ = options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn probe_quantized_gated_gelu_ffn_support(
        &mut self,
        probe: &MediaTekQuantizedGatedGeluFfnSupportProbe,
    ) -> Result<MediaTekQuantizedGatedGeluFfnSupportResult, MediaTekNnapiError> {
        validate_quantized_gated_gelu_support_shape(probe.shape())?;
        #[cfg(target_os = "android")]
        {
            android_nnapi::probe_quantized_gated_gelu_ffn_support(probe)
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = probe;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn compile_quantized_gated_gelu_ffn(
        &mut self,
        owned_weights: MediaTekQuantizedGatedGeluFfnOwnedWeights,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekQuantizedGatedGeluFfnCompilation, MediaTekNnapiError> {
        validate_quantized_gated_gelu_owned_weights(&owned_weights)?;
        #[cfg(target_os = "android")]
        {
            android_nnapi::compile_quantized_gated_gelu_ffn(owned_weights, options)
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = owned_weights;
            let _ = options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn run_compiled_quantized_gated_gelu_ffn(
        &mut self,
        compilation: &MediaTekQuantizedGatedGeluFfnCompilation,
        input: &[u8],
        execution_options: MediaTekNnapiExecutionOptions,
    ) -> Result<MediaTekQuantizedGatedGeluFfnOutput, MediaTekNnapiError> {
        validate_quantized_gated_gelu_input(compilation.shape(), input)?;
        #[cfg(target_os = "android")]
        {
            android_nnapi::run_compiled_quantized_gated_gelu_ffn(
                compilation,
                input,
                execution_options,
            )
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = compilation;
            let _ = input;
            let _ = execution_options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }
    pub fn run_quantized_gated_gelu_ffn_stage_probe(
        &mut self,
        owned_weights: MediaTekQuantizedGatedGeluFfnOwnedWeights,
        input: &[u8],
        stage: MediaTekQuantizedGatedGeluFfnStage,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekQuantizedGatedGeluFfnStageOutput, MediaTekNnapiError> {
        validate_quantized_gated_gelu_owned_weights(&owned_weights)?;
        validate_quantized_gated_gelu_hybrid_stage_probe(&owned_weights)?;
        validate_quantized_gated_gelu_input(owned_weights.shape(), input)?;
        #[cfg(target_os = "android")]
        {
            android_nnapi::run_quantized_gated_gelu_ffn_stage_probe(
                owned_weights,
                input,
                stage,
                options,
            )
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = owned_weights;
            let _ = input;
            let _ = stage;
            let _ = options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn probe_gated_gelu_ffn_f32(
        &mut self,
        tensors: &MediaTekGatedGeluFfnTensorView<'_>,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnOutput, MediaTekNnapiError> {
        #[cfg(target_os = "android")]
        {
            android_nnapi::probe_gated_gelu_ffn_f32(tensors, options)
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = tensors;
            let _ = options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn probe_gated_gelu_ffn_f32_batched(
        &mut self,
        tensors: &MediaTekGatedGeluFfnBatchedTensorView<'_>,
        batch: usize,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnBatchedOutput, MediaTekNnapiError> {
        validate_gated_gelu_f32_batched_input(tensors.shape(), batch, tensors.input())?;
        #[cfg(target_os = "android")]
        {
            android_nnapi::probe_gated_gelu_ffn_f32_batched(tensors, batch, options)
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = tensors;
            let _ = batch;
            let _ = options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn run_gated_gelu_ffn_f32(
        &mut self,
        tensors: &MediaTekGatedGeluFfnTensorView<'_>,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnOutput, MediaTekNnapiError> {
        #[cfg(target_os = "android")]
        {
            android_nnapi::run_gated_gelu_ffn_f32(tensors, options)
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = tensors;
            let _ = options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn compile_gated_gelu_ffn_f32(
        &mut self,
        owned_weights: MediaTekGatedGeluFfnOwnedWeights,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnCompilation, MediaTekNnapiError> {
        #[cfg(target_os = "android")]
        {
            android_nnapi::compile_gated_gelu_ffn_f32(owned_weights, options)
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = owned_weights;
            let _ = options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn compile_gated_gelu_ffn_f32_batched(
        &mut self,
        owned_weights: MediaTekGatedGeluFfnOwnedWeights,
        batch: usize,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnBatchedCompilation, MediaTekNnapiError> {
        self.compile_gated_gelu_ffn_f32_batched_with(
            owned_weights,
            batch,
            options,
            BatchedCompileOptions::default(),
        )
    }

    pub fn compile_gated_gelu_ffn_f32_batched_with(
        &mut self,
        owned_weights: MediaTekGatedGeluFfnOwnedWeights,
        batch: usize,
        options: MediaTekNnapiOptions,
        compile_options: BatchedCompileOptions,
    ) -> Result<MediaTekGatedGeluFfnBatchedCompilation, MediaTekNnapiError> {
        validate_gated_gelu_f32_batched_shape(owned_weights.shape(), batch)?;
        #[cfg(target_os = "android")]
        {
            android_nnapi::compile_gated_gelu_ffn_f32_batched_with(
                owned_weights,
                batch,
                options,
                compile_options,
            )
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = owned_weights;
            let _ = options;
            let _ = compile_options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn run_compiled_gated_gelu_ffn_f32(
        &mut self,
        compilation: &MediaTekGatedGeluFfnCompilation,
        input: &[f32],
        execution_options: MediaTekNnapiExecutionOptions,
    ) -> Result<MediaTekGatedGeluFfnOutput, MediaTekNnapiError> {
        #[cfg(target_os = "android")]
        {
            android_nnapi::run_compiled_gated_gelu_ffn_f32(compilation, input, execution_options)
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = compilation;
            let _ = input;
            let _ = execution_options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }

    pub fn run_compiled_gated_gelu_ffn_f32_batched(
        &mut self,
        compilation: &MediaTekGatedGeluFfnBatchedCompilation,
        input: &[f32],
        batch: usize,
        execution_options: MediaTekNnapiExecutionOptions,
    ) -> Result<MediaTekGatedGeluFfnBatchedOutput, MediaTekNnapiError> {
        if batch != compilation.batch() {
            return Err(MediaTekNnapiError::InvalidInput {
                reason: format!(
                    "batch mismatch: compiled for {}, got {}",
                    compilation.batch(),
                    batch
                ),
            });
        }
        validate_gated_gelu_f32_batched_input(compilation.shape(), batch, input)?;
        #[cfg(target_os = "android")]
        {
            android_nnapi::run_compiled_gated_gelu_ffn_f32_batched(
                compilation,
                input,
                batch,
                execution_options,
            )
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = execution_options;
            Err(MediaTekNnapiError::UnsupportedPlatform)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaTekNnapiOptions {
    device_name_substring: String,
}

impl MediaTekNnapiOptions {
    pub fn new(device_name_substring: impl Into<String>) -> Self {
        Self {
            device_name_substring: device_name_substring.into(),
        }
    }

    pub fn device_name_substring(&self) -> &str {
        &self.device_name_substring
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaTekQuantizedGatedGeluFfnSupportProbe {
    shape: MediaTekGatedGeluFfnShape,
    device_name_substring: String,
}

impl MediaTekQuantizedGatedGeluFfnSupportProbe {
    pub fn new(shape: MediaTekGatedGeluFfnShape) -> Self {
        Self {
            shape,
            device_name_substring: "mtk-neuron".to_string(),
        }
    }

    pub fn with_device_name_substring(mut self, device_name_substring: impl Into<String>) -> Self {
        self.device_name_substring = device_name_substring.into();
        self
    }

    pub const fn shape(&self) -> MediaTekGatedGeluFfnShape {
        self.shape
    }

    pub fn device_name_substring(&self) -> &str {
        &self.device_name_substring
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaTekQuantizedGatedGeluFfnSupportResult {
    chosen_device: MediaTekNnapiDeviceInfo,
    supported_ops: MediaTekGatedGeluFfnSupportedOps,
    model_build_ns: u64,
    supported_ops_query_ns: u64,
}

impl MediaTekQuantizedGatedGeluFfnSupportResult {
    pub const fn new(
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
        model_build_ns: u64,
        supported_ops_query_ns: u64,
    ) -> Self {
        Self {
            chosen_device,
            supported_ops,
            model_build_ns,
            supported_ops_query_ns,
        }
    }

    pub fn chosen_device(&self) -> &MediaTekNnapiDeviceInfo {
        &self.chosen_device
    }

    pub const fn supported_ops(&self) -> MediaTekGatedGeluFfnSupportedOps {
        self.supported_ops
    }

    pub fn supported(&self) -> bool {
        self.supported_ops.all_supported()
    }

    pub const fn model_build_ns(&self) -> u64 {
        self.model_build_ns
    }

    pub const fn supported_ops_query_ns(&self) -> u64 {
        self.supported_ops_query_ns
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaTekQuantizedGatedGeluFfnStage {
    GateFc,
    UpFc,
    GateDequant,
    UpDequant,
    Gelu,
    Gated,
    Output,
}

impl MediaTekQuantizedGatedGeluFfnStage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::GateFc => "gate_fc",
            Self::UpFc => "up_fc",
            Self::GateDequant => "gate_dequant",
            Self::UpDequant => "up_dequant",
            Self::Gelu => "gelu",
            Self::Gated => "gated",
            Self::Output => "output",
        }
    }

    pub const fn output_len(self, shape: MediaTekGatedGeluFfnShape) -> usize {
        match self {
            Self::GateFc
            | Self::UpFc
            | Self::GateDequant
            | Self::UpDequant
            | Self::Gelu
            | Self::Gated => shape.ffn_inner_size(),
            Self::Output => shape.output_size(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTekNnapiExecutionOptions {
    measure_timing: bool,
}

impl MediaTekNnapiExecutionOptions {
    pub const fn new(measure_timing: bool) -> Self {
        Self { measure_timing }
    }

    pub const fn measure_timing(self) -> bool {
        self.measure_timing
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaTekNnapiDeviceInfo {
    name: String,
    device_type: i32,
    feature_level: i64,
    version: String,
}

impl MediaTekNnapiDeviceInfo {
    pub fn new(
        name: impl Into<String>,
        device_type: i32,
        feature_level: i64,
        version: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            device_type,
            feature_level,
            version: version.into(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn device_type(&self) -> i32 {
        self.device_type
    }

    pub fn feature_level(&self) -> i64 {
        self.feature_level
    }

    pub fn version(&self) -> &str {
        &self.version
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTekNnapiSupportedOps {
    fc1: bool,
    fc2: bool,
}

impl MediaTekNnapiSupportedOps {
    pub const fn new(fc1: bool, fc2: bool) -> Self {
        Self { fc1, fc2 }
    }

    pub const fn fc1(self) -> bool {
        self.fc1
    }

    pub const fn fc2(self) -> bool {
        self.fc2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTekGatedGeluFfnSupportedOps {
    named: [(&'static str, bool); 16],
}

impl MediaTekGatedGeluFfnSupportedOps {
    pub const NAMES: [&'static str; 16] = [
        "gate_fc",
        "up_fc",
        "gate_dequant",
        "up_dequant",
        "gelu_square_mul",
        "gelu_cube_mul",
        "gelu_poly_scale_mul",
        "gelu_poly_add",
        "gelu_tanh_scale_mul",
        "gelu_tanh",
        "gelu_one_plus_add",
        "gelu_gate_one_plus_mul",
        "gelu_half_mul",
        "gated_mul",
        "gated_quantize",
        "down_fc",
    ];

    pub fn all(value: bool) -> Self {
        Self::from_bools([value; 16])
    }

    pub fn from_bools(values: [bool; 16]) -> Self {
        let mut named = [(Self::NAMES[0], false); 16];
        for (idx, value) in values.into_iter().enumerate() {
            named[idx] = (Self::NAMES[idx], value);
        }
        Self { named }
    }

    pub const fn named(&self) -> &[(&'static str, bool); 16] {
        &self.named
    }

    pub fn all_supported(&self) -> bool {
        self.named.iter().all(|(_, supported)| *supported)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MediaTekGatedGeluFfnTimings {
    model_build_ns: u64,
    supported_ops_query_ns: u64,
    compilation_ns: u64,
    execution_setup_ns: u64,
    execution_compute_ns: u64,
    token_hash_ns: u64,
}

impl MediaTekGatedGeluFfnTimings {
    pub const fn new(
        model_build_ns: u64,
        supported_ops_query_ns: u64,
        compilation_ns: u64,
        execution_setup_ns: u64,
        execution_compute_ns: u64,
    ) -> Self {
        Self::new_with_token_hash(
            model_build_ns,
            supported_ops_query_ns,
            compilation_ns,
            execution_setup_ns,
            execution_compute_ns,
            0,
        )
    }

    pub const fn new_with_token_hash(
        model_build_ns: u64,
        supported_ops_query_ns: u64,
        compilation_ns: u64,
        execution_setup_ns: u64,
        execution_compute_ns: u64,
        token_hash_ns: u64,
    ) -> Self {
        Self {
            model_build_ns,
            supported_ops_query_ns,
            compilation_ns,
            execution_setup_ns,
            execution_compute_ns,
            token_hash_ns,
        }
    }

    pub const fn model_build_ns(self) -> u64 {
        self.model_build_ns
    }

    pub const fn supported_ops_query_ns(self) -> u64 {
        self.supported_ops_query_ns
    }

    pub const fn compilation_ns(self) -> u64 {
        self.compilation_ns
    }

    pub const fn execution_setup_ns(self) -> u64 {
        self.execution_setup_ns
    }

    pub const fn execution_compute_ns(self) -> u64 {
        self.execution_compute_ns
    }

    pub const fn token_hash_ns(self) -> u64 {
        self.token_hash_ns
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaTekQuantizedMlpOutput {
    output: Vec<u8>,
    chosen_device: MediaTekNnapiDeviceInfo,
    supported_ops: MediaTekNnapiSupportedOps,
    duration_hardware_ns: Option<u64>,
    duration_driver_ns: Option<u64>,
}

impl MediaTekQuantizedMlpOutput {
    pub fn new(
        output: Vec<u8>,
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekNnapiSupportedOps,
        duration_hardware_ns: Option<u64>,
        duration_driver_ns: Option<u64>,
    ) -> Self {
        Self {
            output,
            chosen_device,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
        }
    }

    pub fn output(&self) -> &[u8] {
        &self.output
    }

    pub fn chosen_device(&self) -> &MediaTekNnapiDeviceInfo {
        &self.chosen_device
    }

    pub fn supported_ops(&self) -> MediaTekNnapiSupportedOps {
        self.supported_ops
    }

    pub fn duration_hardware_ns(&self) -> Option<u64> {
        self.duration_hardware_ns
    }

    pub fn duration_driver_ns(&self) -> Option<u64> {
        self.duration_driver_ns
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaTekGatedGeluFfnOutput {
    output: Vec<f32>,
    chosen_device: MediaTekNnapiDeviceInfo,
    supported_ops: MediaTekGatedGeluFfnSupportedOps,
    duration_hardware_ns: Option<u64>,
    duration_driver_ns: Option<u64>,
    timings: MediaTekGatedGeluFfnTimings,
}

impl MediaTekGatedGeluFfnOutput {
    pub fn new(
        output: Vec<f32>,
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
        duration_hardware_ns: Option<u64>,
        duration_driver_ns: Option<u64>,
        timings: MediaTekGatedGeluFfnTimings,
    ) -> Self {
        Self {
            output,
            chosen_device,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        }
    }

    pub fn output(&self) -> &[f32] {
        &self.output
    }

    pub fn chosen_device(&self) -> &MediaTekNnapiDeviceInfo {
        &self.chosen_device
    }

    pub fn supported_ops(&self) -> MediaTekGatedGeluFfnSupportedOps {
        self.supported_ops
    }

    pub fn duration_hardware_ns(&self) -> Option<u64> {
        self.duration_hardware_ns
    }

    pub fn duration_driver_ns(&self) -> Option<u64> {
        self.duration_driver_ns
    }

    pub fn timings(&self) -> MediaTekGatedGeluFfnTimings {
        self.timings
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaTekGatedGeluFfnBatchedOutput {
    output: Vec<f32>,
    batch: usize,
    chosen_device: MediaTekNnapiDeviceInfo,
    supported_ops: MediaTekGatedGeluFfnSupportedOps,
    duration_hardware_ns: Option<u64>,
    duration_driver_ns: Option<u64>,
    timings: MediaTekGatedGeluFfnTimings,
}

impl MediaTekGatedGeluFfnBatchedOutput {
    pub fn new(
        output: Vec<f32>,
        batch: usize,
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
        duration_hardware_ns: Option<u64>,
        duration_driver_ns: Option<u64>,
        timings: MediaTekGatedGeluFfnTimings,
    ) -> Self {
        Self {
            output,
            batch,
            chosen_device,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        }
    }

    pub fn output(&self) -> &[f32] {
        &self.output
    }

    pub const fn batch(&self) -> usize {
        self.batch
    }

    pub fn output_len(&self) -> usize {
        self.output.len()
    }

    pub fn chosen_device(&self) -> &MediaTekNnapiDeviceInfo {
        &self.chosen_device
    }

    pub fn supported_ops(&self) -> MediaTekGatedGeluFfnSupportedOps {
        self.supported_ops
    }

    pub fn supported(&self) -> bool {
        self.supported_ops.all_supported()
    }

    pub fn compile_ns(&self) -> u64 {
        self.timings.compilation_ns()
    }

    pub fn materialize_ns(&self) -> u64 {
        self.timings.model_build_ns()
    }

    pub fn execution_setup_ns(&self) -> u64 {
        self.timings.execution_setup_ns()
    }

    pub fn execution_compute_ns(&self) -> u64 {
        self.timings.execution_compute_ns()
    }

    pub fn execute_hw_ns(&self) -> Option<u64> {
        self.duration_hardware_ns
    }

    pub fn duration_hardware_ns(&self) -> Option<u64> {
        self.duration_hardware_ns
    }

    pub fn duration_driver_ns(&self) -> Option<u64> {
        self.duration_driver_ns
    }

    pub fn timings(&self) -> MediaTekGatedGeluFfnTimings {
        self.timings
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaTekQuantizedGatedGeluFfnOutput {
    output: Vec<f32>,
    chosen_device: MediaTekNnapiDeviceInfo,
    supported_ops: MediaTekGatedGeluFfnSupportedOps,
    duration_hardware_ns: Option<u64>,
    duration_driver_ns: Option<u64>,
    timings: MediaTekGatedGeluFfnTimings,
}

impl MediaTekQuantizedGatedGeluFfnOutput {
    pub fn new(
        output: Vec<f32>,
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
        duration_hardware_ns: Option<u64>,
        duration_driver_ns: Option<u64>,
        timings: MediaTekGatedGeluFfnTimings,
    ) -> Self {
        Self {
            output,
            chosen_device,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        }
    }

    pub fn output(&self) -> &[f32] {
        &self.output
    }

    pub fn chosen_device(&self) -> &MediaTekNnapiDeviceInfo {
        &self.chosen_device
    }

    pub fn supported_ops(&self) -> MediaTekGatedGeluFfnSupportedOps {
        self.supported_ops
    }

    pub fn duration_hardware_ns(&self) -> Option<u64> {
        self.duration_hardware_ns
    }

    pub fn duration_driver_ns(&self) -> Option<u64> {
        self.duration_driver_ns
    }

    pub fn timings(&self) -> MediaTekGatedGeluFfnTimings {
        self.timings
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct MediaTekQuantizedGatedGeluFfnStageOutput {
    stage: MediaTekQuantizedGatedGeluFfnStage,
    output: Vec<f32>,
    chosen_device: MediaTekNnapiDeviceInfo,
    supported_ops: MediaTekGatedGeluFfnSupportedOps,
    duration_hardware_ns: Option<u64>,
    duration_driver_ns: Option<u64>,
    timings: MediaTekGatedGeluFfnTimings,
}

impl MediaTekQuantizedGatedGeluFfnStageOutput {
    pub fn new(
        stage: MediaTekQuantizedGatedGeluFfnStage,
        output: Vec<f32>,
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
        duration_hardware_ns: Option<u64>,
        duration_driver_ns: Option<u64>,
        timings: MediaTekGatedGeluFfnTimings,
    ) -> Self {
        Self {
            stage,
            output,
            chosen_device,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        }
    }

    pub const fn stage(&self) -> MediaTekQuantizedGatedGeluFfnStage {
        self.stage
    }

    pub fn output(&self) -> &[f32] {
        &self.output
    }

    pub fn chosen_device(&self) -> &MediaTekNnapiDeviceInfo {
        &self.chosen_device
    }

    pub fn supported_ops(&self) -> MediaTekGatedGeluFfnSupportedOps {
        self.supported_ops
    }

    pub fn duration_hardware_ns(&self) -> Option<u64> {
        self.duration_hardware_ns
    }

    pub fn duration_driver_ns(&self) -> Option<u64> {
        self.duration_driver_ns
    }

    pub fn timings(&self) -> MediaTekGatedGeluFfnTimings {
        self.timings
    }
}

pub struct MediaTekQuantizedGatedGeluFfnCompilation {
    shape: MediaTekGatedGeluFfnShape,
    #[cfg(target_os = "android")]
    inner: android_nnapi::CompiledQuantizedGatedGeluFfn,
}

impl MediaTekQuantizedGatedGeluFfnCompilation {
    pub const fn shape(&self) -> MediaTekGatedGeluFfnShape {
        self.shape
    }

    pub const fn input_len(&self) -> usize {
        self.shape.input_size()
    }

    pub const fn output_len(&self) -> usize {
        self.shape.output_size()
    }

    pub fn timings(&self) -> MediaTekGatedGeluFfnTimings {
        #[cfg(target_os = "android")]
        {
            self.inner.timings
        }
        #[cfg(not(target_os = "android"))]
        {
            MediaTekGatedGeluFfnTimings::default()
        }
    }

    #[cfg(target_os = "android")]
    fn new(
        shape: MediaTekGatedGeluFfnShape,
        inner: android_nnapi::CompiledQuantizedGatedGeluFfn,
    ) -> Self {
        Self { shape, inner }
    }
}

pub struct MediaTekGatedGeluFfnCompilation {
    #[cfg(target_os = "android")]
    inner: android_nnapi::CompiledGatedGeluFfn,
}

impl MediaTekGatedGeluFfnCompilation {
    pub fn timings(&self) -> MediaTekGatedGeluFfnTimings {
        #[cfg(target_os = "android")]
        {
            self.inner.timings
        }
        #[cfg(not(target_os = "android"))]
        {
            MediaTekGatedGeluFfnTimings::default()
        }
    }

    #[cfg(target_os = "android")]
    fn new(inner: android_nnapi::CompiledGatedGeluFfn) -> Self {
        Self { inner }
    }
}

pub struct MediaTekGatedGeluFfnBatchedCompilation {
    shape: MediaTekGatedGeluFfnShape,
    batch: usize,
    #[cfg(target_os = "android")]
    inner: android_nnapi::CompiledGatedGeluFfnBatched,
}

impl MediaTekGatedGeluFfnBatchedCompilation {
    pub const fn shape(&self) -> MediaTekGatedGeluFfnShape {
        self.shape
    }

    pub const fn batch(&self) -> usize {
        self.batch
    }

    pub const fn input_len(&self) -> usize {
        self.shape.input_size() * self.batch
    }

    pub const fn output_len(&self) -> usize {
        self.shape.output_size() * self.batch
    }

    pub fn timings(&self) -> MediaTekGatedGeluFfnTimings {
        #[cfg(target_os = "android")]
        {
            self.inner.timings
        }
        #[cfg(not(target_os = "android"))]
        {
            MediaTekGatedGeluFfnTimings::default()
        }
    }

    #[cfg(target_os = "android")]
    fn new(
        shape: MediaTekGatedGeluFfnShape,
        batch: usize,
        inner: android_nnapi::CompiledGatedGeluFfnBatched,
    ) -> Self {
        Self {
            shape,
            batch,
            inner,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaTekNnapiError {
    UnsupportedPlatform,
    NoMatchingAccelerator {
        requested: String,
    },
    CpuDeviceRejected {
        name: String,
    },
    UnsupportedOperation {
        fc1: bool,
        fc2: bool,
    },
    UnsupportedGatedGeluFfnOperation {
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
    },
    InvalidOutputLength {
        expected: usize,
        actual: usize,
    },
    InvalidInput {
        reason: String,
    },
    NnapiCall {
        call: &'static str,
        code: i32,
    },
}

impl fmt::Display for MediaTekNnapiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(f, "MediaTek NNAPI execution requires Android"),
            Self::NoMatchingAccelerator { requested } => {
                write!(f, "no MediaTek NNAPI accelerator matched '{requested}'")
            }
            Self::CpuDeviceRejected { name } => {
                write!(
                    f,
                    "NNAPI device '{name}' is CPU/reference, not an accelerator"
                )
            }
            Self::UnsupportedOperation { fc1, fc2 } => write!(
                f,
                "MediaTek NNAPI FULLY_CONNECTED support failed: fc1={fc1} fc2={fc2}"
            ),
            Self::UnsupportedGatedGeluFfnOperation { supported_ops } => {
                write!(f, "MediaTek NNAPI gated GELU FFN support failed: ")?;
                for (idx, (name, supported)) in supported_ops.named().iter().enumerate() {
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
            Self::InvalidInput { reason } => write!(f, "MediaTek NNAPI input invalid: {reason}"),
            Self::NnapiCall { call, code } => {
                write!(f, "MediaTek NNAPI call {call} failed with code {code}")
            }
        }
    }
}

impl std::error::Error for MediaTekNnapiError {}

#[cfg(target_os = "android")]
mod android_nnapi {
    use super::MEDIATEK_NNAPI_DEVICE_ACCELERATOR;
    use super::{
        gated_gelu_f32_blob_layout, gated_gelu_f32_cache_token, quantized_gated_gelu_page_size,
        quantized_gated_gelu_probe_quant_params, quantized_gated_gelu_support_blob_layout,
        validate_gated_gelu_f32_batched_input, validate_gated_gelu_f32_batched_shape,
        validate_quantized_gated_gelu_stage_output_len, GatedGeluFfnF32BlobLayout,
        GatedGeluFfnF32BlobRegion, MediaTekGatedGeluFfnBatchedCompilation,
        MediaTekGatedGeluFfnBatchedOutput, MediaTekGatedGeluFfnBatchedTensorView,
        MediaTekGatedGeluFfnCompilation, MediaTekGatedGeluFfnOutput,
        MediaTekGatedGeluFfnOwnedWeights, MediaTekGatedGeluFfnShape,
        MediaTekGatedGeluFfnSupportedOps, MediaTekGatedGeluFfnTensorView,
        MediaTekGatedGeluFfnTimings, MediaTekNnapiDeviceInfo, MediaTekNnapiError,
        MediaTekNnapiExecutionOptions, MediaTekNnapiOptions, MediaTekNnapiSupportedOps,
        MediaTekQuantParams, MediaTekQuantizedGatedGeluFfnCompilation,
        MediaTekQuantizedGatedGeluFfnOutput, MediaTekQuantizedGatedGeluFfnOwnedWeights,
        MediaTekQuantizedGatedGeluFfnQuantParams, MediaTekQuantizedGatedGeluFfnStage,
        MediaTekQuantizedGatedGeluFfnStageOutput, MediaTekQuantizedGatedGeluFfnSupportProbe,
        MediaTekQuantizedGatedGeluFfnSupportResult, MediaTekQuantizedMlpOutput,
        MediaTekQuantizedMlpTensorView, QuantizedGatedGeluBlobRegion,
        QuantizedGatedGeluSupportBlobLayout, MK28_CACHE_ABI_VERSION,
    };
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::ptr;
    use std::time::Instant;

    const ANEURALNETWORKS_INT32: i32 = 1;
    const ANEURALNETWORKS_TENSOR_FLOAT32: i32 = 3;
    const ANEURALNETWORKS_TENSOR_INT32: i32 = 4;
    const ANEURALNETWORKS_TENSOR_QUANT8_ASYMM: i32 = 5;
    const ANEURALNETWORKS_ADD: i32 = 0;
    const ANEURALNETWORKS_DEQUANTIZE: i32 = 6;
    const ANEURALNETWORKS_FULLY_CONNECTED: i32 = 9;
    const ANEURALNETWORKS_MUL: i32 = 18;
    const ANEURALNETWORKS_TANH: i32 = 28;
    const ANEURALNETWORKS_QUANTIZE: i32 = 72;
    const ANEURALNETWORKS_FUSED_NONE: i32 = 0;
    const ANEURALNETWORKS_FUSED_RELU: i32 = 1;
    const ANEURALNETWORKS_DEVICE_CPU: i32 = 2;
    const ANEURALNETWORKS_DURATION_ON_HARDWARE: i32 = 0;
    const ANEURALNETWORKS_DURATION_IN_DRIVER: i32 = 1;

    #[repr(C)]
    struct ANeuralNetworksModel {
        _private: [u8; 0],
    }

    #[repr(C)]
    struct ANeuralNetworksCompilation {
        _private: [u8; 0],
    }

    #[repr(C)]
    struct ANeuralNetworksExecution {
        _private: [u8; 0],
    }

    #[repr(C)]
    struct ANeuralNetworksDevice {
        _private: [u8; 0],
    }

    #[repr(C)]
    struct ANeuralNetworksMemory {
        _private: [u8; 0],
    }

    #[repr(C)]
    struct ANeuralNetworksOperandType {
        type_: i32,
        dimension_count: u32,
        dimensions: *const u32,
        scale: f32,
        zero_point: i32,
    }

    #[link(name = "neuralnetworks")]
    extern "C" {
        fn ANeuralNetworksModel_create(model: *mut *mut ANeuralNetworksModel) -> i32;
        fn ANeuralNetworksModel_free(model: *mut ANeuralNetworksModel);
        fn ANeuralNetworksModel_addOperand(
            model: *mut ANeuralNetworksModel,
            operand_type: *const ANeuralNetworksOperandType,
        ) -> i32;
        fn ANeuralNetworksModel_setOperandValue(
            model: *mut ANeuralNetworksModel,
            index: i32,
            buffer: *const c_void,
            length: usize,
        ) -> i32;
        fn ANeuralNetworksModel_setOperandValueFromMemory(
            model: *mut ANeuralNetworksModel,
            index: i32,
            memory: *const ANeuralNetworksMemory,
            offset: usize,
            length: usize,
        ) -> i32;
        fn ANeuralNetworksModel_addOperation(
            model: *mut ANeuralNetworksModel,
            op_type: i32,
            input_count: u32,
            inputs: *const u32,
            output_count: u32,
            outputs: *const u32,
        ) -> i32;
        fn ANeuralNetworksModel_identifyInputsAndOutputs(
            model: *mut ANeuralNetworksModel,
            input_count: u32,
            inputs: *const u32,
            output_count: u32,
            outputs: *const u32,
        ) -> i32;
        fn ANeuralNetworksModel_finish(model: *mut ANeuralNetworksModel) -> i32;
        fn ANeuralNetworksModel_getSupportedOperationsForDevices(
            model: *const ANeuralNetworksModel,
            devices: *const *const ANeuralNetworksDevice,
            num_devices: u32,
            supported_ops: *mut bool,
        ) -> i32;

        fn ANeuralNetworks_getDeviceCount(num_devices: *mut u32) -> i32;
        fn ANeuralNetworks_getDevice(
            dev_index: u32,
            device: *mut *mut ANeuralNetworksDevice,
        ) -> i32;
        fn ANeuralNetworksDevice_getName(
            device: *const ANeuralNetworksDevice,
            name: *mut *const c_char,
        ) -> i32;
        fn ANeuralNetworksDevice_getType(
            device: *const ANeuralNetworksDevice,
            type_: *mut i32,
        ) -> i32;
        fn ANeuralNetworksDevice_getFeatureLevel(
            device: *const ANeuralNetworksDevice,
            feature_level: *mut i64,
        ) -> i32;
        fn ANeuralNetworksDevice_getVersion(
            device: *const ANeuralNetworksDevice,
            version: *mut *const c_char,
        ) -> i32;

        fn ANeuralNetworksCompilation_createForDevices(
            model: *mut ANeuralNetworksModel,
            devices: *const *const ANeuralNetworksDevice,
            num_devices: u32,
            compilation: *mut *mut ANeuralNetworksCompilation,
        ) -> i32;
        fn ANeuralNetworksCompilation_finish(compilation: *mut ANeuralNetworksCompilation) -> i32;
        fn ANeuralNetworksCompilation_setCaching(
            compilation: *mut ANeuralNetworksCompilation,
            cacheDir: *const c_char,
            token: *const u8,
        ) -> i32;
        fn ANeuralNetworksCompilation_free(compilation: *mut ANeuralNetworksCompilation);

        fn ANeuralNetworksExecution_create(
            compilation: *mut ANeuralNetworksCompilation,
            execution: *mut *mut ANeuralNetworksExecution,
        ) -> i32;
        fn ANeuralNetworksExecution_setInput(
            execution: *mut ANeuralNetworksExecution,
            index: i32,
            operand_type: *const ANeuralNetworksOperandType,
            buffer: *const c_void,
            length: usize,
        ) -> i32;
        fn ANeuralNetworksExecution_setOutput(
            execution: *mut ANeuralNetworksExecution,
            index: i32,
            operand_type: *const ANeuralNetworksOperandType,
            buffer: *mut c_void,
            length: usize,
        ) -> i32;
        fn ANeuralNetworksExecution_setMeasureTiming(
            execution: *mut ANeuralNetworksExecution,
            measure: bool,
        ) -> i32;
        fn ANeuralNetworksExecution_compute(execution: *mut ANeuralNetworksExecution) -> i32;
        fn ANeuralNetworksExecution_getDuration(
            execution: *const ANeuralNetworksExecution,
            duration_code: i32,
            duration: *mut u64,
        ) -> i32;
        fn ANeuralNetworksExecution_free(execution: *mut ANeuralNetworksExecution);
        fn ANeuralNetworksMemory_createFromFd(
            size: usize,
            protect: i32,
            fd: i32,
            offset: usize,
            memory: *mut *mut ANeuralNetworksMemory,
        ) -> i32;
        fn ANeuralNetworksMemory_free(memory: *mut ANeuralNetworksMemory);
    }

    #[link(name = "android")]
    extern "C" {
        fn ASharedMemory_create(name: *const c_char, size: usize) -> i32;
    }

    struct ModelHandle(*mut ANeuralNetworksModel);
    struct CompilationHandle(*mut ANeuralNetworksCompilation);
    struct ExecutionHandle(*mut ANeuralNetworksExecution);

    impl Drop for ModelHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ANeuralNetworksModel_free(self.0) };
            }
        }
    }

    impl Drop for CompilationHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ANeuralNetworksCompilation_free(self.0) };
            }
        }
    }

    impl Drop for ExecutionHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ANeuralNetworksExecution_free(self.0) };
            }
        }
    }

    struct MemoryHandle(*mut ANeuralNetworksMemory);

    impl Drop for MemoryHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { ANeuralNetworksMemory_free(self.0) };
            }
        }
    }

    struct SharedFd(i32);

    impl Drop for SharedFd {
        fn drop(&mut self) {
            if self.0 >= 0 {
                unsafe {
                    libc::close(self.0);
                }
            }
        }
    }

    struct MappedSharedMemory {
        ptr: *mut c_void,
        len: usize,
    }

    impl MappedSharedMemory {
        fn as_mut_bytes(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.len) }
        }
    }

    impl Drop for MappedSharedMemory {
        fn drop(&mut self) {
            if self.ptr != libc::MAP_FAILED {
                unsafe {
                    libc::munmap(self.ptr, self.len);
                }
            }
        }
    }

    struct QuantizedGatedGeluSharedBlob {
        memory: MemoryHandle,
        #[allow(dead_code)]
        mapping: MappedSharedMemory,
        #[allow(dead_code)]
        fd: SharedFd,
        layout: QuantizedGatedGeluSupportBlobLayout,
    }

    impl QuantizedGatedGeluSharedBlob {
        fn new(shape: MediaTekGatedGeluFfnShape) -> Result<Self, MediaTekNnapiError> {
            let params = quantized_gated_gelu_probe_quant_params();
            Self::new_with_filler(shape, |bytes, layout| {
                fill_quantized_gated_gelu_shared_blob(bytes, layout, params)
            })
        }

        fn new_for_weights(
            weights: &MediaTekQuantizedGatedGeluFfnOwnedWeights,
            params: MediaTekQuantizedGatedGeluFfnQuantParams,
        ) -> Result<Self, MediaTekNnapiError> {
            Self::new_with_filler(weights.shape(), |bytes, layout| {
                fill_quantized_gated_gelu_weights_shared_blob(bytes, layout, params, weights)
            })
        }

        fn new_with_filler(
            shape: MediaTekGatedGeluFfnShape,
            fill: impl FnOnce(
                &mut [u8],
                &QuantizedGatedGeluSupportBlobLayout,
            ) -> Result<(), MediaTekNnapiError>,
        ) -> Result<Self, MediaTekNnapiError> {
            let page_size = quantized_gated_gelu_page_size()?;
            let layout = quantized_gated_gelu_support_blob_layout(shape, page_size)?;
            let name = CString::new("rnb-mtk-quant-gated-gelu-ffn").map_err(|_| {
                MediaTekNnapiError::InvalidInput {
                    reason: "shared memory name contains an interior NUL".to_string(),
                }
            })?;
            let fd = create_shared_memory_fd(&name, layout.total_len)?;
            let mut mapping = map_shared_memory(fd.0, layout.total_len)?;
            fill(mapping.as_mut_bytes(), &layout)?;
            let memory = create_nnapi_memory_from_fd(fd.0, layout.total_len)?;
            Ok(Self {
                memory,
                mapping,
                fd,
                layout,
            })
        }

        fn region(
            &self,
            name: &'static str,
        ) -> Result<QuantizedGatedGeluBlobRegion, MediaTekNnapiError> {
            quantized_gated_gelu_layout_region(&self.layout, name)
        }
    }

    struct GatedGeluFfnF32SharedBlob {
        memory: MemoryHandle,
        #[allow(dead_code)]
        mapping: MappedSharedMemory,
        #[allow(dead_code)]
        fd: SharedFd,
        layout: GatedGeluFfnF32BlobLayout,
    }

    impl GatedGeluFfnF32SharedBlob {
        fn new_for_weights(
            weights: &MediaTekGatedGeluFfnOwnedWeights,
        ) -> Result<Self, MediaTekNnapiError> {
            let page_size = quantized_gated_gelu_page_size()?;
            let layout = gated_gelu_f32_blob_layout(weights.shape(), page_size)?;
            let name = CString::new("rnb-mtk-gated-gelu-ffn-f32").map_err(|_| {
                MediaTekNnapiError::InvalidInput {
                    reason: "shared memory name contains an interior NUL".to_string(),
                }
            })?;
            let fd = create_shared_memory_fd(&name, layout.total_len)?;
            let mut mapping = map_shared_memory(fd.0, layout.total_len)?;
            fill_gated_gelu_f32_weights_shared_blob(mapping.as_mut_bytes(), &layout, weights)?;
            let memory = create_nnapi_memory_from_fd(fd.0, layout.total_len)?;
            Ok(Self {
                memory,
                mapping,
                fd,
                layout,
            })
        }

        fn region(
            &self,
            name: &'static str,
        ) -> Result<GatedGeluFfnF32BlobRegion, MediaTekNnapiError> {
            self.layout.region(name)
        }
    }

    struct AotCacheConfig {
        cache_dir: CString,
        token: [u8; super::NNAPI_CACHE_TOKEN_LEN],
    }

    fn fill_gated_gelu_f32_weights_shared_blob(
        bytes: &mut [u8],
        layout: &GatedGeluFfnF32BlobLayout,
        weights: &MediaTekGatedGeluFfnOwnedWeights,
    ) -> Result<(), MediaTekNnapiError> {
        bytes.fill(0);
        copy_gated_gelu_f32_blob_region(bytes, layout, "gate_weight", weights.gate_weight())?;
        copy_gated_gelu_f32_blob_region(bytes, layout, "up_weight", weights.up_weight())?;
        copy_gated_gelu_f32_blob_region(bytes, layout, "down_weight", weights.down_weight())
    }

    fn copy_gated_gelu_f32_blob_region(
        bytes: &mut [u8],
        layout: &GatedGeluFfnF32BlobLayout,
        name: &'static str,
        data: &[f32],
    ) -> Result<(), MediaTekNnapiError> {
        let region = layout.region(name)?;
        let data = super::f32_as_ne_bytes(data);
        if data.len() != region.length {
            return Err(MediaTekNnapiError::InvalidInput {
                reason: format!(
                    "{name} f32 shared blob length mismatch: expected {}, got {}",
                    region.length,
                    data.len()
                ),
            });
        }
        let end = region.offset.checked_add(region.length).ok_or_else(|| {
            MediaTekNnapiError::InvalidInput {
                reason: format!("{} f32 shared blob range overflows usize", region.name),
            }
        })?;
        bytes[region.offset..end].copy_from_slice(data);
        Ok(())
    }

    fn create_shared_memory_fd(
        name: &CString,
        size: usize,
    ) -> Result<SharedFd, MediaTekNnapiError> {
        let fd = unsafe { ASharedMemory_create(name.as_ptr(), size) };
        if fd < 0 {
            return Err(MediaTekNnapiError::NnapiCall {
                call: "ASharedMemory_create",
                code: fd,
            });
        }
        Ok(SharedFd(fd))
    }

    fn map_shared_memory(fd: i32, size: usize) -> Result<MappedSharedMemory, MediaTekNnapiError> {
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(MediaTekNnapiError::NnapiCall {
                call: "mmap",
                code: -1,
            });
        }
        Ok(MappedSharedMemory { ptr, len: size })
    }

    fn create_nnapi_memory_from_fd(
        fd: i32,
        size: usize,
    ) -> Result<MemoryHandle, MediaTekNnapiError> {
        let mut memory = ptr::null_mut();
        check(
            unsafe {
                ANeuralNetworksMemory_createFromFd(
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    fd,
                    0,
                    &mut memory,
                )
            },
            "ANeuralNetworksMemory_createFromFd",
        )?;
        Ok(MemoryHandle(memory))
    }

    fn fill_quantized_gated_gelu_shared_blob(
        bytes: &mut [u8],
        layout: &QuantizedGatedGeluSupportBlobLayout,
        params: MediaTekQuantizedGatedGeluFfnQuantParams,
    ) -> Result<(), MediaTekNnapiError> {
        bytes.fill(params.input.zero_point() as u8);
        for region in layout.regions() {
            if region.name.ends_with("_bias") {
                fill_quantized_gated_gelu_blob_region(bytes, layout, region.name, 0)?;
            }
        }
        fill_quantized_gated_gelu_constants(bytes, layout, params)
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum QuantizedStageOutputKind {
        Quant8(MediaTekQuantParams),
        F32,
    }

    fn quantized_gated_gelu_stage_output_kind(
        stage: MediaTekQuantizedGatedGeluFfnStage,
        params: MediaTekQuantizedGatedGeluFfnQuantParams,
    ) -> QuantizedStageOutputKind {
        match stage {
            MediaTekQuantizedGatedGeluFfnStage::GateFc => {
                QuantizedStageOutputKind::Quant8(params.gate)
            }
            MediaTekQuantizedGatedGeluFfnStage::UpFc => QuantizedStageOutputKind::Quant8(params.up),
            MediaTekQuantizedGatedGeluFfnStage::GateDequant
            | MediaTekQuantizedGatedGeluFfnStage::UpDequant
            | MediaTekQuantizedGatedGeluFfnStage::Gelu
            | MediaTekQuantizedGatedGeluFfnStage::Gated
            | MediaTekQuantizedGatedGeluFfnStage::Output => QuantizedStageOutputKind::F32,
        }
    }

    fn quantized_gated_gelu_stage_output_operand(stage: MediaTekQuantizedGatedGeluFfnStage) -> u32 {
        match stage {
            MediaTekQuantizedGatedGeluFfnStage::GateFc => 4,
            MediaTekQuantizedGatedGeluFfnStage::UpFc => 7,
            MediaTekQuantizedGatedGeluFfnStage::GateDequant => 8,
            MediaTekQuantizedGatedGeluFfnStage::UpDequant => 9,
            MediaTekQuantizedGatedGeluFfnStage::Gelu => 22,
            MediaTekQuantizedGatedGeluFfnStage::Gated => 23,
            MediaTekQuantizedGatedGeluFfnStage::Output => 26,
        }
    }
    fn fill_quantized_gated_gelu_weights_shared_blob(
        bytes: &mut [u8],
        layout: &QuantizedGatedGeluSupportBlobLayout,
        params: MediaTekQuantizedGatedGeluFfnQuantParams,
        weights: &MediaTekQuantizedGatedGeluFfnOwnedWeights,
    ) -> Result<(), MediaTekNnapiError> {
        fill_quantized_gated_gelu_shared_blob(bytes, layout, params)?;
        copy_quantized_gated_gelu_blob_region(bytes, layout, "gate_weight", weights.gate_weight())?;
        copy_quantized_gated_gelu_blob_region(bytes, layout, "up_weight", weights.up_weight())?;
        copy_quantized_gated_gelu_blob_region(bytes, layout, "down_weight", weights.down_weight())
    }

    fn fill_quantized_gated_gelu_constants(
        bytes: &mut [u8],
        layout: &QuantizedGatedGeluSupportBlobLayout,
        params: MediaTekQuantizedGatedGeluFfnQuantParams,
    ) -> Result<(), MediaTekNnapiError> {
        fill_quantized_gated_gelu_blob_region(
            bytes,
            layout,
            "gelu_coeff_044715",
            quantize_quant8_asymm(0.044715, params.coeff_044715),
        )?;
        fill_quantized_gated_gelu_blob_region(
            bytes,
            layout,
            "gelu_coeff_sqrt_2_over_pi",
            quantize_quant8_asymm(
                (2.0f32 / std::f32::consts::PI).sqrt(),
                params.coeff_sqrt_2_over_pi,
            ),
        )?;
        fill_quantized_gated_gelu_blob_region(
            bytes,
            layout,
            "gelu_one",
            quantize_quant8_asymm(1.0, params.one),
        )?;
        fill_quantized_gated_gelu_blob_region(
            bytes,
            layout,
            "gelu_half",
            quantize_quant8_asymm(0.5, params.half),
        )
    }

    fn quantized_gated_gelu_layout_region(
        layout: &QuantizedGatedGeluSupportBlobLayout,
        name: &'static str,
    ) -> Result<QuantizedGatedGeluBlobRegion, MediaTekNnapiError> {
        layout
            .regions()
            .iter()
            .copied()
            .find(|region| region.name == name)
            .ok_or_else(|| MediaTekNnapiError::InvalidInput {
                reason: format!("missing quantized gated GELU shared blob region {name}"),
            })
    }

    fn fill_quantized_gated_gelu_blob_region(
        bytes: &mut [u8],
        layout: &QuantizedGatedGeluSupportBlobLayout,
        name: &'static str,
        value: u8,
    ) -> Result<(), MediaTekNnapiError> {
        let region = quantized_gated_gelu_layout_region(layout, name)?;
        let end = region.offset.checked_add(region.length).ok_or_else(|| {
            MediaTekNnapiError::InvalidInput {
                reason: format!("{} shared blob range overflows usize", region.name),
            }
        })?;
        bytes[region.offset..end].fill(value);
        Ok(())
    }

    fn copy_quantized_gated_gelu_blob_region(
        bytes: &mut [u8],
        layout: &QuantizedGatedGeluSupportBlobLayout,
        name: &'static str,
        data: &[u8],
    ) -> Result<(), MediaTekNnapiError> {
        let region = quantized_gated_gelu_layout_region(layout, name)?;
        if data.len() != region.length {
            return Err(MediaTekNnapiError::InvalidInput {
                reason: format!(
                    "{name} shared blob length mismatch: expected {}, got {}",
                    region.length,
                    data.len()
                ),
            });
        }
        let end = region.offset.checked_add(region.length).ok_or_else(|| {
            MediaTekNnapiError::InvalidInput {
                reason: format!("{} shared blob range overflows usize", region.name),
            }
        })?;
        bytes[region.offset..end].copy_from_slice(data);
        Ok(())
    }

    fn quantize_quant8_asymm(value: f32, params: MediaTekQuantParams) -> u8 {
        (value / params.scale() + params.zero_point() as f32)
            .round()
            .clamp(0.0, 255.0) as u8
    }

    fn dequantize_quant8_asymm(value: u8, params: MediaTekQuantParams) -> f32 {
        (i32::from(value) - params.zero_point()) as f32 * params.scale()
    }

    struct GatedGeluConstantBuffers {
        zero_inner: Vec<f32>,
        zero_output: Vec<f32>,
        coeff_044715: Vec<f32>,
        coeff_sqrt_2_over_pi: Vec<f32>,
        ones: Vec<f32>,
        half: Vec<f32>,
    }

    impl GatedGeluConstantBuffers {
        fn new(shape: MediaTekGatedGeluFfnShape) -> Self {
            Self {
                zero_inner: vec![0.0f32; shape.ffn_inner_size()],
                zero_output: vec![0.0f32; shape.output_size()],
                coeff_044715: vec![0.044715f32; shape.ffn_inner_size()],
                coeff_sqrt_2_over_pi: vec![
                    (2.0f32 / std::f32::consts::PI).sqrt();
                    shape.ffn_inner_size()
                ],
                ones: vec![1.0f32; shape.ffn_inner_size()],
                half: vec![0.5f32; shape.ffn_inner_size()],
            }
        }
    }

    struct GatedGeluGraphValues<'a> {
        shape: MediaTekGatedGeluFfnShape,
        gate_weight: &'a [f32],
        up_weight: &'a [f32],
        down_weight: &'a [f32],
        constants: &'a GatedGeluConstantBuffers,
    }

    impl<'a> GatedGeluGraphValues<'a> {
        fn borrowed<'b: 'a>(
            tensors: &'a MediaTekGatedGeluFfnTensorView<'b>,
            constants: &'a GatedGeluConstantBuffers,
        ) -> Self {
            Self {
                shape: tensors.shape(),
                gate_weight: tensors.gate_weight(),
                up_weight: tensors.up_weight(),
                down_weight: tensors.down_weight(),
                constants,
            }
        }

        fn borrowed_batched<'b: 'a>(
            tensors: &'a MediaTekGatedGeluFfnBatchedTensorView<'b>,
            constants: &'a GatedGeluConstantBuffers,
        ) -> Self {
            Self {
                shape: tensors.shape(),
                gate_weight: tensors.gate_weight(),
                up_weight: tensors.up_weight(),
                down_weight: tensors.down_weight(),
                constants,
            }
        }

        fn shape(&self) -> MediaTekGatedGeluFfnShape {
            self.shape
        }

        fn gate_weight(&self) -> &[f32] {
            self.gate_weight
        }

        fn up_weight(&self) -> &[f32] {
            self.up_weight
        }

        fn down_weight(&self) -> &[f32] {
            self.down_weight
        }

        fn zero_inner(&self) -> &[f32] {
            &self.constants.zero_inner
        }

        fn zero_output(&self) -> &[f32] {
            &self.constants.zero_output
        }

        fn coeff_044715(&self) -> &[f32] {
            &self.constants.coeff_044715
        }

        fn coeff_sqrt_2_over_pi(&self) -> &[f32] {
            &self.constants.coeff_sqrt_2_over_pi
        }

        fn ones(&self) -> &[f32] {
            &self.constants.ones
        }

        fn half(&self) -> &[f32] {
            &self.constants.half
        }
    }

    struct GatedGeluRetainedBuffers {
        weights: MediaTekGatedGeluFfnOwnedWeights,
        constants: GatedGeluConstantBuffers,
    }

    impl GatedGeluRetainedBuffers {
        fn new(weights: MediaTekGatedGeluFfnOwnedWeights) -> Self {
            let constants = GatedGeluConstantBuffers::new(weights.shape());
            Self { weights, constants }
        }

        fn shape(&self) -> MediaTekGatedGeluFfnShape {
            self.weights.shape()
        }

        fn graph_values(&self) -> GatedGeluGraphValues<'_> {
            GatedGeluGraphValues {
                shape: self.weights.shape(),
                gate_weight: self.weights.gate_weight(),
                up_weight: self.weights.up_weight(),
                down_weight: self.weights.down_weight(),
                constants: &self.constants,
            }
        }
    }

    pub(super) struct CompiledGatedGeluFfn {
        shape: MediaTekGatedGeluFfnShape,
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
        pub(super) timings: MediaTekGatedGeluFfnTimings,
        compilation: CompilationHandle,
        #[allow(dead_code)]
        model: ModelHandle,
        #[allow(dead_code)]
        retained: GatedGeluRetainedBuffers,
    }

    #[allow(dead_code)]
    enum BatchedWeightSource {
        Copy(GatedGeluRetainedBuffers),
        ZeroCopy(GatedGeluFfnF32SharedBlob),
    }

    pub(super) struct CompiledGatedGeluFfnBatched {
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
        pub(super) timings: MediaTekGatedGeluFfnTimings,
        compilation: CompilationHandle,
        #[allow(dead_code)]
        model: ModelHandle,
        #[allow(dead_code)]
        weight_source: BatchedWeightSource,
    }

    pub(super) struct CompiledQuantizedGatedGeluFfn {
        shape: MediaTekGatedGeluFfnShape,
        quant_params: MediaTekQuantizedGatedGeluFfnQuantParams,
        chosen_device: MediaTekNnapiDeviceInfo,
        supported_ops: MediaTekGatedGeluFfnSupportedOps,
        pub(super) timings: MediaTekGatedGeluFfnTimings,
        compilation: CompilationHandle,
        #[allow(dead_code)]
        model: ModelHandle,
        #[allow(dead_code)]
        shared_blob: QuantizedGatedGeluSharedBlob,
        #[allow(dead_code)]
        retained_down_weight_f32: Option<Vec<f32>>,
    }

    pub(super) fn run_quantized_mlp(
        tensors: &MediaTekQuantizedMlpTensorView<'_>,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekQuantizedMlpOutput, MediaTekNnapiError> {
        let model = create_model()?;
        build_quantized_mlp_graph(model.0, tensors)?;

        let chosen = choose_accelerator(options.device_name_substring())?;
        let device_list = [chosen.device as *const ANeuralNetworksDevice];
        let supported_ops = supported_ops(model.0, &device_list)?;
        if !supported_ops.fc1() || !supported_ops.fc2() {
            return Err(MediaTekNnapiError::UnsupportedOperation {
                fc1: supported_ops.fc1(),
                fc2: supported_ops.fc2(),
            });
        }

        let mut compilation_raw = ptr::null_mut();
        check(
            unsafe {
                ANeuralNetworksCompilation_createForDevices(
                    model.0,
                    device_list.as_ptr(),
                    device_list.len() as u32,
                    &mut compilation_raw,
                )
            },
            "ANeuralNetworksCompilation_createForDevices",
        )?;
        let compilation = CompilationHandle(compilation_raw);
        check(
            unsafe { ANeuralNetworksCompilation_finish(compilation.0) },
            "ANeuralNetworksCompilation_finish",
        )?;

        let mut execution_raw = ptr::null_mut();
        check(
            unsafe { ANeuralNetworksExecution_create(compilation.0, &mut execution_raw) },
            "ANeuralNetworksExecution_create",
        )?;
        let execution = ExecutionHandle(execution_raw);

        let input_dims = [1u32, tensors.shape().input_size() as u32];
        let output_dims = [1u32, tensors.shape().output_size() as u32];
        let input_ty = operand_type(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &input_dims,
            tensors.input_params().scale(),
            tensors.input_params().zero_point(),
        );
        let output_ty = operand_type(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &output_dims,
            tensors.output_params().scale(),
            tensors.output_params().zero_point(),
        );
        let mut output = vec![0u8; tensors.output_len()];
        check(
            unsafe {
                ANeuralNetworksExecution_setInput(
                    execution.0,
                    0,
                    &input_ty,
                    tensors.input().as_ptr().cast(),
                    tensors.input().len(),
                )
            },
            "ANeuralNetworksExecution_setInput",
        )?;
        check(
            unsafe {
                ANeuralNetworksExecution_setOutput(
                    execution.0,
                    0,
                    &output_ty,
                    output.as_mut_ptr().cast(),
                    output.len(),
                )
            },
            "ANeuralNetworksExecution_setOutput",
        )?;
        check(
            unsafe { ANeuralNetworksExecution_compute(execution.0) },
            "ANeuralNetworksExecution_compute",
        )?;
        if output.len() != tensors.output_len() {
            return Err(MediaTekNnapiError::InvalidOutputLength {
                expected: tensors.output_len(),
                actual: output.len(),
            });
        }

        let duration_hardware_ns =
            query_duration(execution.0, ANEURALNETWORKS_DURATION_ON_HARDWARE);
        let duration_driver_ns = query_duration(execution.0, ANEURALNETWORKS_DURATION_IN_DRIVER);

        Ok(MediaTekQuantizedMlpOutput::new(
            output,
            chosen.info,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
        ))
    }

    pub(super) fn run_quantized_gated_gelu_ffn_stage_probe(
        owned_weights: MediaTekQuantizedGatedGeluFfnOwnedWeights,
        input: &[u8],
        stage: MediaTekQuantizedGatedGeluFfnStage,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekQuantizedGatedGeluFfnStageOutput, MediaTekNnapiError> {
        let model_build_start = Instant::now();
        let shape = owned_weights.shape();
        let quant_params = owned_weights.quant_params();
        let down_weight_f32 =
            owned_weights
                .down_weight_f32()
                .ok_or_else(|| MediaTekNnapiError::InvalidInput {
                    reason: "quantized gated GELU stage probe requires hybrid f32-down weights"
                        .to_string(),
                })?;
        let shared_blob =
            QuantizedGatedGeluSharedBlob::new_for_weights(&owned_weights, quant_params)?;
        let model = create_model()?;
        build_quantized_gated_gelu_ffn_support_graph(
            model.0,
            shape,
            quant_params,
            &shared_blob,
            Some(down_weight_f32),
            Some(stage),
        )?;
        let model_build_ns = elapsed_ns(model_build_start);

        let chosen = choose_accelerator(options.device_name_substring())?;
        let device_list = [chosen.device as *const ANeuralNetworksDevice];
        let supported_ops_start = Instant::now();
        let supported_ops = supported_gated_gelu_ops(model.0, &device_list, true)?;
        let supported_ops_query_ns = elapsed_ns(supported_ops_start);
        if !supported_ops.all_supported() {
            return Err(MediaTekNnapiError::UnsupportedGatedGeluFfnOperation { supported_ops });
        }

        let compilation_start = Instant::now();
        let mut compilation_raw = ptr::null_mut();
        check(
            unsafe {
                ANeuralNetworksCompilation_createForDevices(
                    model.0,
                    device_list.as_ptr(),
                    device_list.len() as u32,
                    &mut compilation_raw,
                )
            },
            "ANeuralNetworksCompilation_createForDevices",
        )?;
        let compilation = CompilationHandle(compilation_raw);
        check(
            unsafe { ANeuralNetworksCompilation_finish(compilation.0) },
            "ANeuralNetworksCompilation_finish",
        )?;
        let compilation_ns = elapsed_ns(compilation_start);

        let execution_setup_start = Instant::now();
        let mut execution_raw = ptr::null_mut();
        check(
            unsafe { ANeuralNetworksExecution_create(compilation.0, &mut execution_raw) },
            "ANeuralNetworksExecution_create",
        )?;
        let execution = ExecutionHandle(execution_raw);
        check(
            unsafe { ANeuralNetworksExecution_setMeasureTiming(execution.0, true) },
            "ANeuralNetworksExecution_setMeasureTiming",
        )?;

        let input_dims = [1u32, shape.input_size() as u32];
        let input_ty = operand_type(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &input_dims,
            quant_params.input.scale(),
            quant_params.input.zero_point(),
        );
        check(
            unsafe {
                ANeuralNetworksExecution_setInput(
                    execution.0,
                    0,
                    &input_ty,
                    input.as_ptr().cast(),
                    std::mem::size_of_val(input),
                )
            },
            "ANeuralNetworksExecution_setInput",
        )?;

        let output_len = stage.output_len(shape);
        let output_kind = quantized_gated_gelu_stage_output_kind(stage, quant_params);
        let mut quantized_output = vec![0u8; output_len];
        let mut f32_output = vec![0.0f32; output_len];
        match output_kind {
            QuantizedStageOutputKind::Quant8(params) => {
                let dims = [1u32, output_len as u32];
                let output_ty = operand_type(
                    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                    &dims,
                    params.scale(),
                    params.zero_point(),
                );
                check(
                    unsafe {
                        ANeuralNetworksExecution_setOutput(
                            execution.0,
                            0,
                            &output_ty,
                            quantized_output.as_mut_ptr().cast(),
                            std::mem::size_of_val(quantized_output.as_slice()),
                        )
                    },
                    "ANeuralNetworksExecution_setOutput",
                )?;
            }
            QuantizedStageOutputKind::F32 => {
                let dims = [1u32, output_len as u32];
                let output_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &dims, 0.0, 0);
                check(
                    unsafe {
                        ANeuralNetworksExecution_setOutput(
                            execution.0,
                            0,
                            &output_ty,
                            f32_output.as_mut_ptr().cast(),
                            std::mem::size_of_val(f32_output.as_slice()),
                        )
                    },
                    "ANeuralNetworksExecution_setOutput",
                )?;
            }
        }
        let execution_setup_ns = elapsed_ns(execution_setup_start);

        let execution_compute_start = Instant::now();
        check(
            unsafe { ANeuralNetworksExecution_compute(execution.0) },
            "ANeuralNetworksExecution_compute",
        )?;
        let execution_compute_ns = elapsed_ns(execution_compute_start);
        let output = match output_kind {
            QuantizedStageOutputKind::Quant8(params) => {
                validate_quantized_gated_gelu_stage_output_len(
                    stage,
                    shape,
                    quantized_output.len(),
                )?;
                quantized_output
                    .into_iter()
                    .map(|value| dequantize_quant8_asymm(value, params))
                    .collect()
            }
            QuantizedStageOutputKind::F32 => {
                validate_quantized_gated_gelu_stage_output_len(stage, shape, f32_output.len())?;
                f32_output
            }
        };

        let duration_hardware_ns =
            query_duration(execution.0, ANEURALNETWORKS_DURATION_ON_HARDWARE);
        let duration_driver_ns = query_duration(execution.0, ANEURALNETWORKS_DURATION_IN_DRIVER);
        let timings = MediaTekGatedGeluFfnTimings::new(
            model_build_ns,
            supported_ops_query_ns,
            compilation_ns,
            execution_setup_ns,
            execution_compute_ns,
        );

        Ok(MediaTekQuantizedGatedGeluFfnStageOutput::new(
            stage,
            output,
            chosen.info,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        ))
    }

    pub(super) fn probe_quantized_gated_gelu_ffn_support(
        probe: &MediaTekQuantizedGatedGeluFfnSupportProbe,
    ) -> Result<MediaTekQuantizedGatedGeluFfnSupportResult, MediaTekNnapiError> {
        let shape = probe.shape();
        let model_build_start = Instant::now();
        let shared_blob = QuantizedGatedGeluSharedBlob::new(shape)?;
        let down_len = shape
            .down_len()
            .map_err(|err| MediaTekNnapiError::InvalidInput {
                reason: err.to_string(),
            })?;
        let zero_down_weight_f32 = vec![0.0f32; down_len];
        let model = create_model()?;
        build_quantized_gated_gelu_ffn_support_graph(
            model.0,
            shape,
            quantized_gated_gelu_probe_quant_params(),
            &shared_blob,
            Some(&zero_down_weight_f32),
            None,
        )?;
        let model_build_ns = elapsed_ns(model_build_start);

        let chosen = choose_accelerator(probe.device_name_substring())?;
        let device_list = [chosen.device as *const ANeuralNetworksDevice];
        let supported_ops_start = Instant::now();
        let supported_ops = supported_gated_gelu_ops(model.0, &device_list, true)?;
        let supported_ops_query_ns = elapsed_ns(supported_ops_start);

        Ok(MediaTekQuantizedGatedGeluFfnSupportResult::new(
            chosen.info,
            supported_ops,
            model_build_ns,
            supported_ops_query_ns,
        ))
    }

    pub(super) fn compile_quantized_gated_gelu_ffn(
        owned_weights: MediaTekQuantizedGatedGeluFfnOwnedWeights,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekQuantizedGatedGeluFfnCompilation, MediaTekNnapiError> {
        let model_build_start = Instant::now();
        let shape = owned_weights.shape();
        let quant_params = owned_weights.quant_params();
        let hybrid_down_f32 = owned_weights.down_weight_f32().is_some();
        let shared_blob =
            QuantizedGatedGeluSharedBlob::new_for_weights(&owned_weights, quant_params)?;
        let model = create_model()?;
        build_quantized_gated_gelu_ffn_support_graph(
            model.0,
            shape,
            quant_params,
            &shared_blob,
            owned_weights.down_weight_f32(),
            None,
        )?;
        let model_build_ns = elapsed_ns(model_build_start);

        let chosen = choose_accelerator(options.device_name_substring())?;
        let device_list = [chosen.device as *const ANeuralNetworksDevice];
        let supported_ops_start = Instant::now();
        let supported_ops = supported_gated_gelu_ops(model.0, &device_list, hybrid_down_f32)?;
        let supported_ops_query_ns = elapsed_ns(supported_ops_start);
        if !supported_ops.all_supported() {
            return Err(MediaTekNnapiError::UnsupportedGatedGeluFfnOperation { supported_ops });
        }

        let compilation_start = Instant::now();
        let mut compilation_raw = ptr::null_mut();
        check(
            unsafe {
                ANeuralNetworksCompilation_createForDevices(
                    model.0,
                    device_list.as_ptr(),
                    device_list.len() as u32,
                    &mut compilation_raw,
                )
            },
            "ANeuralNetworksCompilation_createForDevices",
        )?;
        let compilation = CompilationHandle(compilation_raw);
        check(
            unsafe { ANeuralNetworksCompilation_finish(compilation.0) },
            "ANeuralNetworksCompilation_finish",
        )?;
        let compilation_ns = elapsed_ns(compilation_start);
        let timings = MediaTekGatedGeluFfnTimings::new(
            model_build_ns,
            supported_ops_query_ns,
            compilation_ns,
            0,
            0,
        );
        let retained_down_weight_f32 = owned_weights.down_weight_f32;

        Ok(MediaTekQuantizedGatedGeluFfnCompilation::new(
            shape,
            CompiledQuantizedGatedGeluFfn {
                shape,
                quant_params,
                chosen_device: chosen.info,
                supported_ops,
                timings,
                compilation,
                model,
                shared_blob,
                retained_down_weight_f32,
            },
        ))
    }

    pub(super) fn run_compiled_quantized_gated_gelu_ffn(
        compilation: &MediaTekQuantizedGatedGeluFfnCompilation,
        input: &[u8],
        execution_options: MediaTekNnapiExecutionOptions,
    ) -> Result<MediaTekQuantizedGatedGeluFfnOutput, MediaTekNnapiError> {
        let compiled = &compilation.inner;
        let hybrid_down_f32 = compiled.retained_down_weight_f32.is_some();
        let measure_timing = execution_options.measure_timing();
        let execution_setup_start = measure_timing.then(Instant::now);
        let mut execution_raw = ptr::null_mut();
        check(
            unsafe { ANeuralNetworksExecution_create(compiled.compilation.0, &mut execution_raw) },
            "ANeuralNetworksExecution_create",
        )?;
        let execution = ExecutionHandle(execution_raw);
        if measure_timing {
            check(
                unsafe { ANeuralNetworksExecution_setMeasureTiming(execution.0, true) },
                "ANeuralNetworksExecution_setMeasureTiming",
            )?;
        }

        let input_dims = [1u32, compiled.shape.input_size() as u32];
        let output_dims = [1u32, compiled.shape.output_size() as u32];
        let input_ty = operand_type(
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &input_dims,
            compiled.quant_params.input.scale(),
            compiled.quant_params.input.zero_point(),
        );
        let output_ty = if hybrid_down_f32 {
            operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &output_dims, 0.0, 0)
        } else {
            operand_type(
                ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                &output_dims,
                compiled.quant_params.output.scale(),
                compiled.quant_params.output.zero_point(),
            )
        };
        let mut quantized_output =
            vec![compiled.quant_params.output.zero_point() as u8; compiled.shape.output_size()];
        let mut f32_output = vec![0.0f32; compiled.shape.output_size()];
        check(
            unsafe {
                ANeuralNetworksExecution_setInput(
                    execution.0,
                    0,
                    &input_ty,
                    input.as_ptr().cast(),
                    std::mem::size_of_val(input),
                )
            },
            "ANeuralNetworksExecution_setInput",
        )?;
        if hybrid_down_f32 {
            check(
                unsafe {
                    ANeuralNetworksExecution_setOutput(
                        execution.0,
                        0,
                        &output_ty,
                        f32_output.as_mut_ptr().cast(),
                        std::mem::size_of_val(f32_output.as_slice()),
                    )
                },
                "ANeuralNetworksExecution_setOutput",
            )?;
        } else {
            check(
                unsafe {
                    ANeuralNetworksExecution_setOutput(
                        execution.0,
                        0,
                        &output_ty,
                        quantized_output.as_mut_ptr().cast(),
                        std::mem::size_of_val(quantized_output.as_slice()),
                    )
                },
                "ANeuralNetworksExecution_setOutput",
            )?;
        }
        let execution_setup_ns = execution_setup_start.map(elapsed_ns).unwrap_or(0);

        let execution_compute_start = measure_timing.then(Instant::now);
        check(
            unsafe { ANeuralNetworksExecution_compute(execution.0) },
            "ANeuralNetworksExecution_compute",
        )?;
        let execution_compute_ns = execution_compute_start.map(elapsed_ns).unwrap_or(0);
        let output = if hybrid_down_f32 {
            f32_output
        } else {
            quantized_output
                .into_iter()
                .map(|value| dequantize_quant8_asymm(value, compiled.quant_params.output))
                .collect()
        };
        if output.len() != compiled.shape.output_size() {
            return Err(MediaTekNnapiError::InvalidOutputLength {
                expected: compiled.shape.output_size(),
                actual: output.len(),
            });
        }

        let duration_hardware_ns = if measure_timing {
            query_duration(execution.0, ANEURALNETWORKS_DURATION_ON_HARDWARE)
        } else {
            None
        };
        let duration_driver_ns = if measure_timing {
            query_duration(execution.0, ANEURALNETWORKS_DURATION_IN_DRIVER)
        } else {
            None
        };
        let timings =
            MediaTekGatedGeluFfnTimings::new(0, 0, 0, execution_setup_ns, execution_compute_ns);

        Ok(MediaTekQuantizedGatedGeluFfnOutput::new(
            output,
            compiled.chosen_device.clone(),
            compiled.supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        ))
    }

    fn elapsed_ns(start: Instant) -> u64 {
        start.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }

    pub(super) fn probe_gated_gelu_ffn_f32(
        tensors: &MediaTekGatedGeluFfnTensorView<'_>,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnOutput, MediaTekNnapiError> {
        let model_build_start = Instant::now();
        let constants = GatedGeluConstantBuffers::new(tensors.shape());
        let model = create_model()?;
        build_gated_gelu_ffn_f32_graph(
            model.0,
            GatedGeluGraphValues::borrowed(tensors, &constants),
        )?;
        let model_build_ns = elapsed_ns(model_build_start);

        let chosen = choose_accelerator(options.device_name_substring())?;
        let device_list = [chosen.device as *const ANeuralNetworksDevice];
        let supported_ops_start = Instant::now();
        let supported_ops = supported_gated_gelu_f32_ops(model.0, &device_list)?;
        let supported_ops_query_ns = elapsed_ns(supported_ops_start);
        if !supported_ops.all_supported() {
            return Err(MediaTekNnapiError::UnsupportedGatedGeluFfnOperation { supported_ops });
        }

        let compilation_start = Instant::now();
        let mut compilation_raw = ptr::null_mut();
        check(
            unsafe {
                ANeuralNetworksCompilation_createForDevices(
                    model.0,
                    device_list.as_ptr(),
                    device_list.len() as u32,
                    &mut compilation_raw,
                )
            },
            "ANeuralNetworksCompilation_createForDevices",
        )?;
        let compilation = CompilationHandle(compilation_raw);
        check(
            unsafe { ANeuralNetworksCompilation_finish(compilation.0) },
            "ANeuralNetworksCompilation_finish",
        )?;
        let compilation_ns = elapsed_ns(compilation_start);

        let execution_setup_start = Instant::now();
        let mut execution_raw = ptr::null_mut();
        check(
            unsafe { ANeuralNetworksExecution_create(compilation.0, &mut execution_raw) },
            "ANeuralNetworksExecution_create",
        )?;
        let execution = ExecutionHandle(execution_raw);
        check(
            unsafe { ANeuralNetworksExecution_setMeasureTiming(execution.0, true) },
            "ANeuralNetworksExecution_setMeasureTiming",
        )?;

        let input_dims = [1u32, tensors.shape().input_size() as u32];
        let output_dims = [1u32, tensors.shape().output_size() as u32];
        let input_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &input_dims, 0.0, 0);
        let output_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &output_dims, 0.0, 0);
        let mut output = vec![0.0f32; tensors.output_len()];
        check(
            unsafe {
                ANeuralNetworksExecution_setInput(
                    execution.0,
                    0,
                    &input_ty,
                    tensors.input().as_ptr().cast(),
                    std::mem::size_of_val(tensors.input()),
                )
            },
            "ANeuralNetworksExecution_setInput",
        )?;
        check(
            unsafe {
                ANeuralNetworksExecution_setOutput(
                    execution.0,
                    0,
                    &output_ty,
                    output.as_mut_ptr().cast(),
                    std::mem::size_of_val(output.as_slice()),
                )
            },
            "ANeuralNetworksExecution_setOutput",
        )?;
        let execution_setup_ns = elapsed_ns(execution_setup_start);

        let execution_compute_start = Instant::now();
        check(
            unsafe { ANeuralNetworksExecution_compute(execution.0) },
            "ANeuralNetworksExecution_compute",
        )?;
        let execution_compute_ns = elapsed_ns(execution_compute_start);
        if output.len() != tensors.output_len() {
            return Err(MediaTekNnapiError::InvalidOutputLength {
                expected: tensors.output_len(),
                actual: output.len(),
            });
        }

        let duration_hardware_ns =
            query_duration(execution.0, ANEURALNETWORKS_DURATION_ON_HARDWARE);
        let duration_driver_ns = query_duration(execution.0, ANEURALNETWORKS_DURATION_IN_DRIVER);
        let timings = MediaTekGatedGeluFfnTimings::new(
            model_build_ns,
            supported_ops_query_ns,
            compilation_ns,
            execution_setup_ns,
            execution_compute_ns,
        );

        Ok(MediaTekGatedGeluFfnOutput::new(
            output,
            chosen.info,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        ))
    }

    pub(super) fn probe_gated_gelu_ffn_f32_batched(
        tensors: &MediaTekGatedGeluFfnBatchedTensorView<'_>,
        batch: usize,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnBatchedOutput, MediaTekNnapiError> {
        let (input_len, output_len) =
            validate_gated_gelu_f32_batched_input(tensors.shape(), batch, tensors.input())?;
        let model_build_start = Instant::now();
        let constants = GatedGeluConstantBuffers::new(tensors.shape());
        let model = create_model()?;
        build_gated_gelu_ffn_f32_batched_graph(
            model.0,
            GatedGeluGraphValues::borrowed_batched(tensors, &constants),
            batch,
        )?;
        let model_build_ns = elapsed_ns(model_build_start);

        let chosen = choose_accelerator(options.device_name_substring())?;
        let device_list = [chosen.device as *const ANeuralNetworksDevice];
        let supported_ops_start = Instant::now();
        let supported_ops = supported_gated_gelu_f32_ops(model.0, &device_list)?;
        let supported_ops_query_ns = elapsed_ns(supported_ops_start);
        if !supported_ops.all_supported() {
            return Err(MediaTekNnapiError::UnsupportedGatedGeluFfnOperation { supported_ops });
        }

        let compilation_start = Instant::now();
        let mut compilation_raw = ptr::null_mut();
        check(
            unsafe {
                ANeuralNetworksCompilation_createForDevices(
                    model.0,
                    device_list.as_ptr(),
                    device_list.len() as u32,
                    &mut compilation_raw,
                )
            },
            "ANeuralNetworksCompilation_createForDevices",
        )?;
        let compilation = CompilationHandle(compilation_raw);
        check(
            unsafe { ANeuralNetworksCompilation_finish(compilation.0) },
            "ANeuralNetworksCompilation_finish",
        )?;
        let compilation_ns = elapsed_ns(compilation_start);

        let execution_setup_start = Instant::now();
        let mut execution_raw = ptr::null_mut();
        check(
            unsafe { ANeuralNetworksExecution_create(compilation.0, &mut execution_raw) },
            "ANeuralNetworksExecution_create",
        )?;
        let execution = ExecutionHandle(execution_raw);
        check(
            unsafe { ANeuralNetworksExecution_setMeasureTiming(execution.0, true) },
            "ANeuralNetworksExecution_setMeasureTiming",
        )?;

        let input_dims = [batch as u32, tensors.shape().input_size() as u32];
        let output_dims = [batch as u32, tensors.shape().output_size() as u32];
        let input_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &input_dims, 0.0, 0);
        let output_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &output_dims, 0.0, 0);
        let mut output = vec![0.0f32; output_len];
        check(
            unsafe {
                ANeuralNetworksExecution_setInput(
                    execution.0,
                    0,
                    &input_ty,
                    tensors.input().as_ptr().cast(),
                    input_len * std::mem::size_of::<f32>(),
                )
            },
            "ANeuralNetworksExecution_setInput",
        )?;
        check(
            unsafe {
                ANeuralNetworksExecution_setOutput(
                    execution.0,
                    0,
                    &output_ty,
                    output.as_mut_ptr().cast(),
                    output_len * std::mem::size_of::<f32>(),
                )
            },
            "ANeuralNetworksExecution_setOutput",
        )?;
        let execution_setup_ns = elapsed_ns(execution_setup_start);

        let execution_compute_start = Instant::now();
        check(
            unsafe { ANeuralNetworksExecution_compute(execution.0) },
            "ANeuralNetworksExecution_compute",
        )?;
        let execution_compute_ns = elapsed_ns(execution_compute_start);
        if output.len() != output_len {
            return Err(MediaTekNnapiError::InvalidOutputLength {
                expected: output_len,
                actual: output.len(),
            });
        }

        let duration_hardware_ns =
            query_duration(execution.0, ANEURALNETWORKS_DURATION_ON_HARDWARE);
        let duration_driver_ns = query_duration(execution.0, ANEURALNETWORKS_DURATION_IN_DRIVER);
        let timings = MediaTekGatedGeluFfnTimings::new(
            model_build_ns,
            supported_ops_query_ns,
            compilation_ns,
            execution_setup_ns,
            execution_compute_ns,
        );

        Ok(MediaTekGatedGeluFfnBatchedOutput::new(
            output,
            batch,
            chosen.info,
            supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        ))
    }

    pub(super) fn run_gated_gelu_ffn_f32(
        tensors: &MediaTekGatedGeluFfnTensorView<'_>,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnOutput, MediaTekNnapiError> {
        probe_gated_gelu_ffn_f32(tensors, options)
    }

    pub(super) fn compile_gated_gelu_ffn_f32(
        owned_weights: MediaTekGatedGeluFfnOwnedWeights,
        options: MediaTekNnapiOptions,
    ) -> Result<MediaTekGatedGeluFfnCompilation, MediaTekNnapiError> {
        let model_build_start = Instant::now();
        let retained = GatedGeluRetainedBuffers::new(owned_weights);
        let shape = retained.shape();
        let model = create_model()?;
        build_gated_gelu_ffn_f32_graph(model.0, retained.graph_values())?;
        let model_build_ns = elapsed_ns(model_build_start);

        let chosen = choose_accelerator(options.device_name_substring())?;
        let device_list = [chosen.device as *const ANeuralNetworksDevice];
        let supported_ops_start = Instant::now();
        let supported_ops = supported_gated_gelu_f32_ops(model.0, &device_list)?;
        let supported_ops_query_ns = elapsed_ns(supported_ops_start);
        if !supported_ops.all_supported() {
            return Err(MediaTekNnapiError::UnsupportedGatedGeluFfnOperation { supported_ops });
        }

        let compilation_start = Instant::now();
        let mut compilation_raw = ptr::null_mut();
        check(
            unsafe {
                ANeuralNetworksCompilation_createForDevices(
                    model.0,
                    device_list.as_ptr(),
                    device_list.len() as u32,
                    &mut compilation_raw,
                )
            },
            "ANeuralNetworksCompilation_createForDevices",
        )?;
        let compilation = CompilationHandle(compilation_raw);
        check(
            unsafe { ANeuralNetworksCompilation_finish(compilation.0) },
            "ANeuralNetworksCompilation_finish",
        )?;
        let compilation_ns = elapsed_ns(compilation_start);
        let timings = MediaTekGatedGeluFfnTimings::new(
            model_build_ns,
            supported_ops_query_ns,
            compilation_ns,
            0,
            0,
        );

        Ok(MediaTekGatedGeluFfnCompilation::new(CompiledGatedGeluFfn {
            shape,
            chosen_device: chosen.info,
            supported_ops,
            timings,
            compilation,
            model,
            retained,
        }))
    }

    pub(super) fn run_compiled_gated_gelu_ffn_f32(
        compilation: &MediaTekGatedGeluFfnCompilation,
        input: &[f32],
        execution_options: MediaTekNnapiExecutionOptions,
    ) -> Result<MediaTekGatedGeluFfnOutput, MediaTekNnapiError> {
        let compiled = &compilation.inner;
        if input.len() != compiled.shape.input_size() {
            return Err(MediaTekNnapiError::InvalidInput {
                reason: format!(
                    "input length mismatch: expected {}, got {}",
                    compiled.shape.input_size(),
                    input.len()
                ),
            });
        }
        for (index, value) in input.iter().enumerate() {
            if !value.is_finite() {
                return Err(MediaTekNnapiError::InvalidInput {
                    reason: format!("input has non-finite value at index {index}"),
                });
            }
        }

        let measure_timing = execution_options.measure_timing();
        let execution_setup_start = measure_timing.then(Instant::now);
        let mut execution_raw = ptr::null_mut();
        check(
            unsafe { ANeuralNetworksExecution_create(compiled.compilation.0, &mut execution_raw) },
            "ANeuralNetworksExecution_create",
        )?;
        let execution = ExecutionHandle(execution_raw);
        if measure_timing {
            check(
                unsafe { ANeuralNetworksExecution_setMeasureTiming(execution.0, true) },
                "ANeuralNetworksExecution_setMeasureTiming",
            )?;
        }

        let input_dims = [1u32, compiled.shape.input_size() as u32];
        let output_dims = [1u32, compiled.shape.output_size() as u32];
        let input_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &input_dims, 0.0, 0);
        let output_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &output_dims, 0.0, 0);
        let mut output = vec![0.0f32; compiled.shape.output_size()];
        check(
            unsafe {
                ANeuralNetworksExecution_setInput(
                    execution.0,
                    0,
                    &input_ty,
                    input.as_ptr().cast(),
                    std::mem::size_of_val(input),
                )
            },
            "ANeuralNetworksExecution_setInput",
        )?;
        check(
            unsafe {
                ANeuralNetworksExecution_setOutput(
                    execution.0,
                    0,
                    &output_ty,
                    output.as_mut_ptr().cast(),
                    std::mem::size_of_val(output.as_slice()),
                )
            },
            "ANeuralNetworksExecution_setOutput",
        )?;
        let execution_setup_ns = execution_setup_start.map(elapsed_ns).unwrap_or(0);

        let execution_compute_start = measure_timing.then(Instant::now);
        check(
            unsafe { ANeuralNetworksExecution_compute(execution.0) },
            "ANeuralNetworksExecution_compute",
        )?;
        let execution_compute_ns = execution_compute_start.map(elapsed_ns).unwrap_or(0);
        if output.len() != compiled.shape.output_size() {
            return Err(MediaTekNnapiError::InvalidOutputLength {
                expected: compiled.shape.output_size(),
                actual: output.len(),
            });
        }

        let duration_hardware_ns = if measure_timing {
            query_duration(execution.0, ANEURALNETWORKS_DURATION_ON_HARDWARE)
        } else {
            None
        };
        let duration_driver_ns = if measure_timing {
            query_duration(execution.0, ANEURALNETWORKS_DURATION_IN_DRIVER)
        } else {
            None
        };
        let timings =
            MediaTekGatedGeluFfnTimings::new(0, 0, 0, execution_setup_ns, execution_compute_ns);

        Ok(MediaTekGatedGeluFfnOutput::new(
            output,
            compiled.chosen_device.clone(),
            compiled.supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        ))
    }

    pub(super) fn compile_gated_gelu_ffn_f32_batched_with(
        owned_weights: MediaTekGatedGeluFfnOwnedWeights,
        batch: usize,
        options: MediaTekNnapiOptions,
        compile_options: super::BatchedCompileOptions,
    ) -> Result<MediaTekGatedGeluFfnBatchedCompilation, MediaTekNnapiError> {
        let shape = owned_weights.shape();
        validate_gated_gelu_f32_batched_shape(shape, batch)?;
        let chosen = choose_accelerator(options.device_name_substring())?;
        let variant = if compile_options.zero_copy { 1 } else { 0 };
        let (aot_cache, token_hash_ns) = build_aot_cache_config(
            compile_options.cache_dir.as_deref(),
            variant,
            chosen.info.name(),
            shape,
            batch,
            &owned_weights,
        )?;

        let model_build_start = Instant::now();
        let constants = compile_options
            .zero_copy
            .then(|| GatedGeluConstantBuffers::new(shape));
        // Declare `weight_source` before `model` so that on any early return below,
        // locals drop in reverse declaration order and `model` (which references the
        // shared blob memory via setOperandValueFromMemory) is freed before the blob
        // and GeLU constant buffers it points into. This matches the success-path
        // struct field order (compilation, model, weight_source) and avoids a
        // use-after-free of the NNAPI memory during error cleanup.
        let weight_source;
        let model;
        if compile_options.zero_copy {
            let shared_blob = GatedGeluFfnF32SharedBlob::new_for_weights(&owned_weights)?;
            let built_model = create_model()?;
            build_gated_gelu_ffn_f32_batched_from_memory_graph(
                built_model.0,
                &shared_blob,
                shape,
                batch,
                constants
                    .as_ref()
                    .expect("zero-copy constants must be retained until compilation finish"),
            )?;
            weight_source = BatchedWeightSource::ZeroCopy(shared_blob);
            model = built_model;
        } else {
            let retained = GatedGeluRetainedBuffers::new(owned_weights);
            let built_model = create_model()?;
            build_gated_gelu_ffn_f32_batched_graph(built_model.0, retained.graph_values(), batch)?;
            weight_source = BatchedWeightSource::Copy(retained);
            model = built_model;
        }
        let model_build_ns = elapsed_ns(model_build_start);

        let device_list = [chosen.device as *const ANeuralNetworksDevice];
        let supported_ops_start = Instant::now();
        let supported_ops = supported_gated_gelu_f32_ops(model.0, &device_list)?;
        let supported_ops_query_ns = elapsed_ns(supported_ops_start);
        if !supported_ops.all_supported() {
            return Err(MediaTekNnapiError::UnsupportedGatedGeluFfnOperation { supported_ops });
        }

        let compilation_start = Instant::now();
        let mut compilation_raw = ptr::null_mut();
        check(
            unsafe {
                ANeuralNetworksCompilation_createForDevices(
                    model.0,
                    device_list.as_ptr(),
                    device_list.len() as u32,
                    &mut compilation_raw,
                )
            },
            "ANeuralNetworksCompilation_createForDevices",
        )?;
        let compilation = CompilationHandle(compilation_raw);
        if let Some(cache) = aot_cache.as_ref() {
            check(
                unsafe {
                    ANeuralNetworksCompilation_setCaching(
                        compilation.0,
                        cache.cache_dir.as_ptr(),
                        cache.token.as_ptr(),
                    )
                },
                "ANeuralNetworksCompilation_setCaching",
            )?;
        }
        check(
            unsafe { ANeuralNetworksCompilation_finish(compilation.0) },
            "ANeuralNetworksCompilation_finish",
        )?;
        let compilation_ns = elapsed_ns(compilation_start);
        let timings = MediaTekGatedGeluFfnTimings::new_with_token_hash(
            model_build_ns,
            supported_ops_query_ns,
            compilation_ns,
            0,
            0,
            token_hash_ns,
        );

        drop(constants);
        Ok(MediaTekGatedGeluFfnBatchedCompilation::new(
            shape,
            batch,
            CompiledGatedGeluFfnBatched {
                chosen_device: chosen.info,
                supported_ops,
                timings,
                compilation,
                model,
                weight_source,
            },
        ))
    }

    fn build_aot_cache_config(
        cache_dir: Option<&str>,
        variant: u8,
        device_name: &str,
        shape: MediaTekGatedGeluFfnShape,
        batch: usize,
        weights: &MediaTekGatedGeluFfnOwnedWeights,
    ) -> Result<(Option<AotCacheConfig>, u64), MediaTekNnapiError> {
        let Some(cache_dir) = cache_dir else {
            return Ok((None, 0));
        };
        if cache_dir.trim().is_empty() {
            return Err(MediaTekNnapiError::InvalidInput {
                reason: "NNAPI AOT cache dir must not be empty".to_string(),
            });
        }
        let token_hash_start = Instant::now();
        let _ = MK28_CACHE_ABI_VERSION;
        let token = gated_gelu_f32_cache_token(
            variant,
            device_name,
            shape,
            batch,
            weights.gate_weight(),
            weights.up_weight(),
            weights.down_weight(),
        );
        let token_hash_ns = elapsed_ns(token_hash_start);
        let cache_dir = CString::new(cache_dir).map_err(|_| MediaTekNnapiError::InvalidInput {
            reason: "NNAPI AOT cache dir contains an interior NUL".to_string(),
        })?;
        Ok((Some(AotCacheConfig { cache_dir, token }), token_hash_ns))
    }

    pub(super) fn run_compiled_gated_gelu_ffn_f32_batched(
        compilation: &MediaTekGatedGeluFfnBatchedCompilation,
        input: &[f32],
        batch: usize,
        execution_options: MediaTekNnapiExecutionOptions,
    ) -> Result<MediaTekGatedGeluFfnBatchedOutput, MediaTekNnapiError> {
        let compiled = &compilation.inner;
        if batch != compilation.batch() {
            return Err(MediaTekNnapiError::InvalidInput {
                reason: format!(
                    "batch mismatch: compiled for {}, got {}",
                    compilation.batch(),
                    batch
                ),
            });
        }
        let shape = compilation.shape();
        let (input_len, output_len) = validate_gated_gelu_f32_batched_input(shape, batch, input)?;
        for (index, value) in input.iter().enumerate() {
            if !value.is_finite() {
                return Err(MediaTekNnapiError::InvalidInput {
                    reason: format!("input has non-finite value at index {index}"),
                });
            }
        }

        let measure_timing = execution_options.measure_timing();
        let execution_setup_start = measure_timing.then(Instant::now);
        let mut execution_raw = ptr::null_mut();
        check(
            unsafe { ANeuralNetworksExecution_create(compiled.compilation.0, &mut execution_raw) },
            "ANeuralNetworksExecution_create",
        )?;
        let execution = ExecutionHandle(execution_raw);
        if measure_timing {
            check(
                unsafe { ANeuralNetworksExecution_setMeasureTiming(execution.0, true) },
                "ANeuralNetworksExecution_setMeasureTiming",
            )?;
        }

        let input_dims = [batch as u32, shape.input_size() as u32];
        let output_dims = [batch as u32, shape.output_size() as u32];
        let input_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &input_dims, 0.0, 0);
        let output_ty = operand_type(ANEURALNETWORKS_TENSOR_FLOAT32, &output_dims, 0.0, 0);
        let mut output = vec![0.0f32; output_len];
        check(
            unsafe {
                ANeuralNetworksExecution_setInput(
                    execution.0,
                    0,
                    &input_ty,
                    input.as_ptr().cast(),
                    input_len * std::mem::size_of::<f32>(),
                )
            },
            "ANeuralNetworksExecution_setInput",
        )?;
        check(
            unsafe {
                ANeuralNetworksExecution_setOutput(
                    execution.0,
                    0,
                    &output_ty,
                    output.as_mut_ptr().cast(),
                    output_len * std::mem::size_of::<f32>(),
                )
            },
            "ANeuralNetworksExecution_setOutput",
        )?;
        let execution_setup_ns = execution_setup_start.map(elapsed_ns).unwrap_or(0);

        let execution_compute_start = measure_timing.then(Instant::now);
        check(
            unsafe { ANeuralNetworksExecution_compute(execution.0) },
            "ANeuralNetworksExecution_compute",
        )?;
        let execution_compute_ns = execution_compute_start.map(elapsed_ns).unwrap_or(0);
        if output.len() != output_len {
            return Err(MediaTekNnapiError::InvalidOutputLength {
                expected: output_len,
                actual: output.len(),
            });
        }

        let duration_hardware_ns = if measure_timing {
            query_duration(execution.0, ANEURALNETWORKS_DURATION_ON_HARDWARE)
        } else {
            None
        };
        let duration_driver_ns = if measure_timing {
            query_duration(execution.0, ANEURALNETWORKS_DURATION_IN_DRIVER)
        } else {
            None
        };
        let timings =
            MediaTekGatedGeluFfnTimings::new(0, 0, 0, execution_setup_ns, execution_compute_ns);

        Ok(MediaTekGatedGeluFfnBatchedOutput::new(
            output,
            batch,
            compiled.chosen_device.clone(),
            compiled.supported_ops,
            duration_hardware_ns,
            duration_driver_ns,
            timings,
        ))
    }

    struct ChosenDevice {
        device: *mut ANeuralNetworksDevice,
        info: MediaTekNnapiDeviceInfo,
    }

    fn create_model() -> Result<ModelHandle, MediaTekNnapiError> {
        let mut model = ptr::null_mut();
        check(
            unsafe { ANeuralNetworksModel_create(&mut model) },
            "ANeuralNetworksModel_create",
        )?;
        Ok(ModelHandle(model))
    }

    fn build_quantized_mlp_graph(
        model: *mut ANeuralNetworksModel,
        tensors: &MediaTekQuantizedMlpTensorView<'_>,
    ) -> Result<(), MediaTekNnapiError> {
        let input_dims = [1u32, tensors.shape().input_size() as u32];
        let w1_dims = [
            tensors.shape().hidden_size() as u32,
            tensors.shape().input_size() as u32,
        ];
        let hidden_dim = [tensors.shape().hidden_size() as u32];
        let hidden_dims = [1u32, tensors.shape().hidden_size() as u32];
        let w2_dims = [
            tensors.shape().output_size() as u32,
            tensors.shape().hidden_size() as u32,
        ];
        let output_dim = [tensors.shape().output_size() as u32];
        let output_dims = [1u32, tensors.shape().output_size() as u32];

        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &input_dims,
            tensors.input_params().scale(),
            tensors.input_params().zero_point(),
        )?;
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &w1_dims,
            tensors.w1_params().scale(),
            tensors.w1_params().zero_point(),
        )?;
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_INT32,
            &hidden_dim,
            tensors.input_params().scale() * tensors.w1_params().scale(),
            0,
        )?;
        add_operand(model, ANEURALNETWORKS_INT32, &[], 0.0, 0)?;
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &hidden_dims,
            tensors.hidden_params().scale(),
            tensors.hidden_params().zero_point(),
        )?;
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &w2_dims,
            tensors.w2_params().scale(),
            tensors.w2_params().zero_point(),
        )?;
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_INT32,
            &output_dim,
            tensors.hidden_params().scale() * tensors.w2_params().scale(),
            0,
        )?;
        add_operand(model, ANEURALNETWORKS_INT32, &[], 0.0, 0)?;
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            &output_dims,
            tensors.output_params().scale(),
            tensors.output_params().zero_point(),
        )?;

        set_operand_value_u8(model, 1, tensors.w1(), "set w1")?;
        set_operand_value_i32(model, 2, tensors.b1(), "set b1")?;
        set_operand_value_scalar_i32(model, 3, ANEURALNETWORKS_FUSED_RELU, "set fused relu")?;
        set_operand_value_u8(model, 5, tensors.w2(), "set w2")?;
        set_operand_value_i32(model, 6, tensors.b2(), "set b2")?;
        set_operand_value_scalar_i32(model, 7, ANEURALNETWORKS_FUSED_NONE, "set fused none")?;

        let fc1_inputs = [0u32, 1, 2, 3];
        let fc1_outputs = [4u32];
        let fc2_inputs = [4u32, 5, 6, 7];
        let fc2_outputs = [8u32];
        check(
            unsafe {
                ANeuralNetworksModel_addOperation(
                    model,
                    ANEURALNETWORKS_FULLY_CONNECTED,
                    fc1_inputs.len() as u32,
                    fc1_inputs.as_ptr(),
                    fc1_outputs.len() as u32,
                    fc1_outputs.as_ptr(),
                )
            },
            "ANeuralNetworksModel_addOperation(fc1)",
        )?;
        check(
            unsafe {
                ANeuralNetworksModel_addOperation(
                    model,
                    ANEURALNETWORKS_FULLY_CONNECTED,
                    fc2_inputs.len() as u32,
                    fc2_inputs.as_ptr(),
                    fc2_outputs.len() as u32,
                    fc2_outputs.as_ptr(),
                )
            },
            "ANeuralNetworksModel_addOperation(fc2)",
        )?;
        let inputs = [0u32];
        let outputs = [8u32];
        check(
            unsafe {
                ANeuralNetworksModel_identifyInputsAndOutputs(
                    model,
                    inputs.len() as u32,
                    inputs.as_ptr(),
                    outputs.len() as u32,
                    outputs.as_ptr(),
                )
            },
            "ANeuralNetworksModel_identifyInputsAndOutputs",
        )?;
        check(
            unsafe { ANeuralNetworksModel_finish(model) },
            "ANeuralNetworksModel_finish",
        )
    }

    fn build_quantized_gated_gelu_ffn_support_graph(
        model: *mut ANeuralNetworksModel,
        shape: MediaTekGatedGeluFfnShape,
        params: MediaTekQuantizedGatedGeluFfnQuantParams,
        blob: &QuantizedGatedGeluSharedBlob,
        down_weight_f32: Option<&[f32]>,
        output_stage: Option<MediaTekQuantizedGatedGeluFfnStage>,
    ) -> Result<(), MediaTekNnapiError> {
        let input_dims = [1u32, shape.input_size() as u32];
        let inner_dims = [1u32, shape.ffn_inner_size() as u32];
        let gate_up_weight_dims = [shape.ffn_inner_size() as u32, shape.input_size() as u32];
        let inner_bias_dims = [shape.ffn_inner_size() as u32];
        let down_weight_dims = [shape.output_size() as u32, shape.ffn_inner_size() as u32];
        let output_bias_dims = [shape.output_size() as u32];
        let output_dims = [1u32, shape.output_size() as u32];
        let hybrid_down_f32 = down_weight_f32.is_some();
        if output_stage.is_some() && !hybrid_down_f32 {
            return Err(MediaTekNnapiError::InvalidInput {
                reason: "stage probe requires hybrid f32-down graph".to_string(),
            });
        }

        add_quant8_operand(model, &input_dims, params.input)?; // 0
        add_quant8_operand(model, &gate_up_weight_dims, params.gate_weight)?; // 1
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_INT32,
            &inner_bias_dims,
            params.gate_bias.scale(),
            params.gate_bias.zero_point(),
        )?; // 2
        add_operand(model, ANEURALNETWORKS_INT32, &[], 0.0, 0)?; // 3
        add_quant8_operand(model, &inner_dims, params.gate)?; // 4
        add_quant8_operand(model, &gate_up_weight_dims, params.up_weight)?; // 5
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_INT32,
            &inner_bias_dims,
            params.up_bias.scale(),
            params.up_bias.zero_point(),
        )?; // 6
        add_quant8_operand(model, &inner_dims, params.up)?; // 7
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 8
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 9
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 10
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 11
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 12
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 13
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 14
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 15
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 16
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 17
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 18
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 19
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 20
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 21
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 22
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 23
        if hybrid_down_f32 {
            add_operand(
                model,
                ANEURALNETWORKS_TENSOR_FLOAT32,
                &down_weight_dims,
                0.0,
                0,
            )?; // 24
            add_operand(
                model,
                ANEURALNETWORKS_TENSOR_FLOAT32,
                &output_bias_dims,
                0.0,
                0,
            )?; // 25
            add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &output_dims, 0.0, 0)?;
        // 26
        } else {
            add_quant8_operand(model, &inner_dims, params.gated)?; // 24
            add_quant8_operand(model, &down_weight_dims, params.down_weight)?; // 25
            add_operand(
                model,
                ANEURALNETWORKS_TENSOR_INT32,
                &output_bias_dims,
                params.down_bias.scale(),
                params.down_bias.zero_point(),
            )?; // 26
            add_quant8_operand(model, &output_dims, params.output)?; // 27
        }

        let coeff_044715 = vec![0.044715f32; shape.ffn_inner_size()];
        let coeff_sqrt_2_over_pi =
            vec![(2.0f32 / std::f32::consts::PI).sqrt(); shape.ffn_inner_size()];
        let ones = vec![1.0f32; shape.ffn_inner_size()];
        let half = vec![0.5f32; shape.ffn_inner_size()];
        let zero_output_bias = vec![0.0f32; shape.output_size()];

        set_operand_value_from_memory(model, 1, blob, "gate_weight", "set gate_weight")?;
        set_operand_value_from_memory(model, 2, blob, "gate_bias", "set gate_bias")?;
        set_operand_value_scalar_i32(model, 3, ANEURALNETWORKS_FUSED_NONE, "set fc fused none")?;
        set_operand_value_from_memory(model, 5, blob, "up_weight", "set up_weight")?;
        set_operand_value_from_memory(model, 6, blob, "up_bias", "set up_bias")?;
        set_operand_value_f32(model, 12, &coeff_044715, "set gelu coeff 0.044715")?;
        set_operand_value_f32(
            model,
            15,
            &coeff_sqrt_2_over_pi,
            "set gelu coeff sqrt_2_over_pi",
        )?;
        set_operand_value_f32(model, 18, &ones, "set gelu ones")?;
        set_operand_value_f32(model, 21, &half, "set gelu half")?;
        if let Some(down_weight_f32) = down_weight_f32 {
            set_operand_value_f32(model, 24, down_weight_f32, "set down_weight_f32")?;
            set_operand_value_f32(model, 25, &zero_output_bias, "set down_bias_f32")?;
        } else {
            set_operand_value_from_memory(model, 25, blob, "down_weight", "set down_weight")?;
            set_operand_value_from_memory(model, 26, blob, "down_bias", "set down_bias")?;
        }

        add_operation(
            model,
            ANEURALNETWORKS_FULLY_CONNECTED,
            &[0, 1, 2, 3],
            &[4],
            "gate_fc",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_FULLY_CONNECTED,
            &[0, 5, 6, 3],
            &[7],
            "up_fc",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_DEQUANTIZE,
            &[4],
            &[8],
            "gate_dequant",
        )?;
        add_operation(model, ANEURALNETWORKS_DEQUANTIZE, &[7], &[9], "up_dequant")?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[8, 8, 3],
            &[10],
            "gelu_square_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[10, 8, 3],
            &[11],
            "gelu_cube_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[11, 12, 3],
            &[13],
            "gelu_poly_scale_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_ADD,
            &[8, 13, 3],
            &[14],
            "gelu_poly_add",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[14, 15, 3],
            &[16],
            "gelu_tanh_scale_mul",
        )?;
        add_operation(model, ANEURALNETWORKS_TANH, &[16], &[17], "gelu_tanh")?;
        add_operation(
            model,
            ANEURALNETWORKS_ADD,
            &[18, 17, 3],
            &[19],
            "gelu_one_plus_add",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[8, 19, 3],
            &[20],
            "gelu_gate_one_plus_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[20, 21, 3],
            &[22],
            "gelu_half_mul",
        )?;
        add_operation(model, ANEURALNETWORKS_MUL, &[22, 9, 3], &[23], "gated_mul")?;
        if hybrid_down_f32 {
            add_operation(
                model,
                ANEURALNETWORKS_FULLY_CONNECTED,
                &[23, 24, 25, 3],
                &[26],
                "down_fc",
            )?;
        } else {
            add_operation(
                model,
                ANEURALNETWORKS_QUANTIZE,
                &[23],
                &[24],
                "gated_quantize",
            )?;
            add_operation(
                model,
                ANEURALNETWORKS_FULLY_CONNECTED,
                &[24, 25, 26, 3],
                &[27],
                "down_fc",
            )?;
        }

        let output_operand = output_stage
            .map(quantized_gated_gelu_stage_output_operand)
            .unwrap_or(if hybrid_down_f32 { 26u32 } else { 27u32 });
        let inputs = [0u32];
        let outputs = [output_operand];
        check(
            unsafe {
                ANeuralNetworksModel_identifyInputsAndOutputs(
                    model,
                    inputs.len() as u32,
                    inputs.as_ptr(),
                    outputs.len() as u32,
                    outputs.as_ptr(),
                )
            },
            "ANeuralNetworksModel_identifyInputsAndOutputs",
        )?;
        check(
            unsafe { ANeuralNetworksModel_finish(model) },
            "ANeuralNetworksModel_finish",
        )
    }

    fn build_gated_gelu_ffn_f32_graph(
        model: *mut ANeuralNetworksModel,
        tensors: GatedGeluGraphValues<'_>,
    ) -> Result<(), MediaTekNnapiError> {
        let input_dims = [1u32, tensors.shape().input_size() as u32];
        let inner_dims = [1u32, tensors.shape().ffn_inner_size() as u32];
        let gate_up_weight_dims = [
            tensors.shape().ffn_inner_size() as u32,
            tensors.shape().input_size() as u32,
        ];
        let inner_bias_dims = [tensors.shape().ffn_inner_size() as u32];
        let down_weight_dims = [
            tensors.shape().output_size() as u32,
            tensors.shape().ffn_inner_size() as u32,
        ];
        let output_bias_dims = [tensors.shape().output_size() as u32];
        let output_dims = [1u32, tensors.shape().output_size() as u32];

        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &input_dims, 0.0, 0)?; // 0
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &gate_up_weight_dims,
            0.0,
            0,
        )?; // 1
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &inner_bias_dims,
            0.0,
            0,
        )?; // 2
        add_operand(model, ANEURALNETWORKS_INT32, &[], 0.0, 0)?; // 3
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 4
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &gate_up_weight_dims,
            0.0,
            0,
        )?; // 5
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &inner_bias_dims,
            0.0,
            0,
        )?; // 6
        for _ in 0..15 {
            add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?;
        } // 7..21
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &down_weight_dims,
            0.0,
            0,
        )?; // 22
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &output_bias_dims,
            0.0,
            0,
        )?; // 23
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &output_dims, 0.0, 0)?; // 24
        add_operand(model, ANEURALNETWORKS_INT32, &[], 0.0, 0)?; // 25

        set_operand_value_f32(model, 1, tensors.gate_weight(), "set gate_weight")?;
        set_operand_value_f32(model, 2, tensors.zero_inner(), "set gate_bias")?;
        set_operand_value_scalar_i32(model, 3, ANEURALNETWORKS_FUSED_NONE, "set fc fused none")?;
        set_operand_value_f32(model, 5, tensors.up_weight(), "set up_weight")?;
        set_operand_value_f32(model, 6, tensors.zero_inner(), "set up_bias")?;
        set_operand_value_f32(model, 10, tensors.coeff_044715(), "set gelu coeff 0.044715")?;
        set_operand_value_f32(
            model,
            13,
            tensors.coeff_sqrt_2_over_pi(),
            "set gelu coeff sqrt_2_over_pi",
        )?;
        set_operand_value_f32(model, 16, tensors.ones(), "set gelu ones")?;
        set_operand_value_f32(model, 19, tensors.half(), "set gelu half")?;
        set_operand_value_f32(model, 22, tensors.down_weight(), "set down_weight")?;
        set_operand_value_f32(model, 23, tensors.zero_output(), "set down_bias")?;
        set_operand_value_scalar_i32(model, 25, ANEURALNETWORKS_FUSED_NONE, "set op fused none")?;

        add_operation(
            model,
            ANEURALNETWORKS_FULLY_CONNECTED,
            &[0, 1, 2, 3],
            &[4],
            "gate_fc",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_FULLY_CONNECTED,
            &[0, 5, 6, 3],
            &[7],
            "up_fc",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[4, 4, 25],
            &[8],
            "gelu_square_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[8, 4, 25],
            &[9],
            "gelu_cube_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[9, 10, 25],
            &[11],
            "gelu_poly_scale_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_ADD,
            &[4, 11, 25],
            &[12],
            "gelu_poly_add",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[12, 13, 25],
            &[14],
            "gelu_tanh_scale_mul",
        )?;
        add_operation(model, ANEURALNETWORKS_TANH, &[14], &[15], "gelu_tanh")?;
        add_operation(
            model,
            ANEURALNETWORKS_ADD,
            &[16, 15, 25],
            &[17],
            "gelu_one_plus_add",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[4, 17, 25],
            &[18],
            "gelu_gate_one_plus_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[18, 19, 25],
            &[20],
            "gelu_half_mul",
        )?;
        add_operation(model, ANEURALNETWORKS_MUL, &[20, 7, 25], &[21], "gated_mul")?;
        add_operation(
            model,
            ANEURALNETWORKS_FULLY_CONNECTED,
            &[21, 22, 23, 3],
            &[24],
            "down_fc",
        )?;

        let inputs = [0u32];
        let outputs = [24u32];
        check(
            unsafe {
                ANeuralNetworksModel_identifyInputsAndOutputs(
                    model,
                    inputs.len() as u32,
                    inputs.as_ptr(),
                    outputs.len() as u32,
                    outputs.as_ptr(),
                )
            },
            "ANeuralNetworksModel_identifyInputsAndOutputs",
        )?;
        check(
            unsafe { ANeuralNetworksModel_finish(model) },
            "ANeuralNetworksModel_finish",
        )
    }

    fn build_gated_gelu_ffn_f32_batched_graph(
        model: *mut ANeuralNetworksModel,
        tensors: GatedGeluGraphValues<'_>,
        batch: usize,
    ) -> Result<(), MediaTekNnapiError> {
        validate_gated_gelu_f32_batched_shape(tensors.shape(), batch)?;
        add_gated_gelu_ffn_f32_batched_operands(model, tensors.shape(), batch)?;

        set_operand_value_f32(model, 1, tensors.gate_weight(), "set gate_weight")?;
        set_operand_value_f32(model, 2, tensors.zero_inner(), "set gate_bias")?;
        set_operand_value_scalar_i32(model, 3, ANEURALNETWORKS_FUSED_NONE, "set fc fused none")?;
        set_operand_value_f32(model, 5, tensors.up_weight(), "set up_weight")?;
        set_operand_value_f32(model, 6, tensors.zero_inner(), "set up_bias")?;
        set_operand_value_f32(model, 10, tensors.coeff_044715(), "set gelu coeff 0.044715")?;
        set_operand_value_f32(
            model,
            13,
            tensors.coeff_sqrt_2_over_pi(),
            "set gelu coeff sqrt_2_over_pi",
        )?;
        set_operand_value_f32(model, 16, tensors.ones(), "set gelu ones")?;
        set_operand_value_f32(model, 19, tensors.half(), "set gelu half")?;
        set_operand_value_f32(model, 22, tensors.down_weight(), "set down_weight")?;
        set_operand_value_f32(model, 23, tensors.zero_output(), "set down_bias")?;
        set_operand_value_scalar_i32(model, 25, ANEURALNETWORKS_FUSED_NONE, "set op fused none")?;

        finish_gated_gelu_ffn_f32_batched_graph(model)
    }

    fn build_gated_gelu_ffn_f32_batched_from_memory_graph(
        model: *mut ANeuralNetworksModel,
        blob: &GatedGeluFfnF32SharedBlob,
        shape: MediaTekGatedGeluFfnShape,
        batch: usize,
        constants: &GatedGeluConstantBuffers,
    ) -> Result<(), MediaTekNnapiError> {
        validate_gated_gelu_f32_batched_shape(shape, batch)?;
        add_gated_gelu_ffn_f32_batched_operands(model, shape, batch)?;

        set_operand_value_from_f32_blob(model, 1, blob, "gate_weight", "set gate_weight")?;
        set_operand_value_from_f32_blob(model, 2, blob, "gate_bias", "set gate_bias")?;
        set_operand_value_scalar_i32(model, 3, ANEURALNETWORKS_FUSED_NONE, "set fc fused none")?;
        set_operand_value_from_f32_blob(model, 5, blob, "up_weight", "set up_weight")?;
        set_operand_value_from_f32_blob(model, 6, blob, "up_bias", "set up_bias")?;
        set_operand_value_f32(
            model,
            10,
            &constants.coeff_044715,
            "set gelu coeff 0.044715",
        )?;
        set_operand_value_f32(
            model,
            13,
            &constants.coeff_sqrt_2_over_pi,
            "set gelu coeff sqrt_2_over_pi",
        )?;
        set_operand_value_f32(model, 16, &constants.ones, "set gelu ones")?;
        set_operand_value_f32(model, 19, &constants.half, "set gelu half")?;
        set_operand_value_from_f32_blob(model, 22, blob, "down_weight", "set down_weight")?;
        set_operand_value_from_f32_blob(model, 23, blob, "down_bias", "set down_bias")?;
        set_operand_value_scalar_i32(model, 25, ANEURALNETWORKS_FUSED_NONE, "set op fused none")?;

        finish_gated_gelu_ffn_f32_batched_graph(model)
    }

    fn add_gated_gelu_ffn_f32_batched_operands(
        model: *mut ANeuralNetworksModel,
        shape: MediaTekGatedGeluFfnShape,
        batch: usize,
    ) -> Result<(), MediaTekNnapiError> {
        let input_dims = [batch as u32, shape.input_size() as u32];
        let inner_dims = [batch as u32, shape.ffn_inner_size() as u32];
        let coeff_dims = [1u32, shape.ffn_inner_size() as u32];
        let gate_up_weight_dims = [shape.ffn_inner_size() as u32, shape.input_size() as u32];
        let inner_bias_dims = [shape.ffn_inner_size() as u32];
        let down_weight_dims = [shape.output_size() as u32, shape.ffn_inner_size() as u32];
        let output_bias_dims = [shape.output_size() as u32];
        let output_dims = [batch as u32, shape.output_size() as u32];

        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &input_dims, 0.0, 0)?; // 0
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &gate_up_weight_dims,
            0.0,
            0,
        )?; // 1
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &inner_bias_dims,
            0.0,
            0,
        )?; // 2
        add_operand(model, ANEURALNETWORKS_INT32, &[], 0.0, 0)?; // 3
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 4
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &gate_up_weight_dims,
            0.0,
            0,
        )?; // 5
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &inner_bias_dims,
            0.0,
            0,
        )?; // 6
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 7
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 8
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 9
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &coeff_dims, 0.0, 0)?; // 10
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 11
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 12
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &coeff_dims, 0.0, 0)?; // 13
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 14
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 15
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &coeff_dims, 0.0, 0)?; // 16
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 17
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 18
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &coeff_dims, 0.0, 0)?; // 19
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 20
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &inner_dims, 0.0, 0)?; // 21
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &down_weight_dims,
            0.0,
            0,
        )?; // 22
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_FLOAT32,
            &output_bias_dims,
            0.0,
            0,
        )?; // 23
        add_operand(model, ANEURALNETWORKS_TENSOR_FLOAT32, &output_dims, 0.0, 0)?; // 24
        add_operand(model, ANEURALNETWORKS_INT32, &[], 0.0, 0)?; // 25
        Ok(())
    }

    fn finish_gated_gelu_ffn_f32_batched_graph(
        model: *mut ANeuralNetworksModel,
    ) -> Result<(), MediaTekNnapiError> {
        add_operation(
            model,
            ANEURALNETWORKS_FULLY_CONNECTED,
            &[0, 1, 2, 3],
            &[4],
            "gate_fc",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_FULLY_CONNECTED,
            &[0, 5, 6, 3],
            &[7],
            "up_fc",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[4, 4, 25],
            &[8],
            "gelu_square_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[8, 4, 25],
            &[9],
            "gelu_cube_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[9, 10, 25],
            &[11],
            "gelu_poly_scale_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_ADD,
            &[4, 11, 25],
            &[12],
            "gelu_poly_add",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[12, 13, 25],
            &[14],
            "gelu_tanh_scale_mul",
        )?;
        add_operation(model, ANEURALNETWORKS_TANH, &[14], &[15], "gelu_tanh")?;
        add_operation(
            model,
            ANEURALNETWORKS_ADD,
            &[16, 15, 25],
            &[17],
            "gelu_one_plus_add",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[4, 17, 25],
            &[18],
            "gelu_gate_one_plus_mul",
        )?;
        add_operation(
            model,
            ANEURALNETWORKS_MUL,
            &[18, 19, 25],
            &[20],
            "gelu_half_mul",
        )?;
        add_operation(model, ANEURALNETWORKS_MUL, &[20, 7, 25], &[21], "gated_mul")?;
        add_operation(
            model,
            ANEURALNETWORKS_FULLY_CONNECTED,
            &[21, 22, 23, 3],
            &[24],
            "down_fc",
        )?;

        let inputs = [0u32];
        let outputs = [24u32];
        check(
            unsafe {
                ANeuralNetworksModel_identifyInputsAndOutputs(
                    model,
                    inputs.len() as u32,
                    inputs.as_ptr(),
                    outputs.len() as u32,
                    outputs.as_ptr(),
                )
            },
            "ANeuralNetworksModel_identifyInputsAndOutputs",
        )?;
        check(
            unsafe { ANeuralNetworksModel_finish(model) },
            "ANeuralNetworksModel_finish",
        )
    }

    fn add_operand(
        model: *mut ANeuralNetworksModel,
        type_: i32,
        dims: &[u32],
        scale: f32,
        zero_point: i32,
    ) -> Result<(), MediaTekNnapiError> {
        let ty = operand_type(type_, dims, scale, zero_point);
        check(
            unsafe { ANeuralNetworksModel_addOperand(model, &ty) },
            "ANeuralNetworksModel_addOperand",
        )
    }

    fn add_quant8_operand(
        model: *mut ANeuralNetworksModel,
        dims: &[u32],
        params: MediaTekQuantParams,
    ) -> Result<(), MediaTekNnapiError> {
        add_operand(
            model,
            ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
            dims,
            params.scale(),
            params.zero_point(),
        )
    }

    fn add_operation(
        model: *mut ANeuralNetworksModel,
        op_type: i32,
        inputs: &[u32],
        outputs: &[u32],
        name: &'static str,
    ) -> Result<(), MediaTekNnapiError> {
        check(
            unsafe {
                ANeuralNetworksModel_addOperation(
                    model,
                    op_type,
                    inputs.len() as u32,
                    inputs.as_ptr(),
                    outputs.len() as u32,
                    outputs.as_ptr(),
                )
            },
            name,
        )
    }

    fn operand_type(
        type_: i32,
        dims: &[u32],
        scale: f32,
        zero_point: i32,
    ) -> ANeuralNetworksOperandType {
        ANeuralNetworksOperandType {
            type_,
            dimension_count: dims.len() as u32,
            dimensions: if dims.is_empty() {
                ptr::null()
            } else {
                dims.as_ptr()
            },
            scale,
            zero_point,
        }
    }

    fn set_operand_value_u8(
        model: *mut ANeuralNetworksModel,
        index: i32,
        data: &[u8],
        call: &'static str,
    ) -> Result<(), MediaTekNnapiError> {
        check(
            unsafe {
                ANeuralNetworksModel_setOperandValue(model, index, data.as_ptr().cast(), data.len())
            },
            call,
        )
    }

    fn set_operand_value_i32(
        model: *mut ANeuralNetworksModel,
        index: i32,
        data: &[i32],
        call: &'static str,
    ) -> Result<(), MediaTekNnapiError> {
        check(
            unsafe {
                ANeuralNetworksModel_setOperandValue(
                    model,
                    index,
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data),
                )
            },
            call,
        )
    }

    fn set_operand_value_f32(
        model: *mut ANeuralNetworksModel,
        index: i32,
        data: &[f32],
        call: &'static str,
    ) -> Result<(), MediaTekNnapiError> {
        check(
            unsafe {
                ANeuralNetworksModel_setOperandValue(
                    model,
                    index,
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data),
                )
            },
            call,
        )
    }

    fn set_operand_value_scalar_i32(
        model: *mut ANeuralNetworksModel,
        index: i32,
        value: i32,
        call: &'static str,
    ) -> Result<(), MediaTekNnapiError> {
        check(
            unsafe {
                ANeuralNetworksModel_setOperandValue(
                    model,
                    index,
                    (&value as *const i32).cast(),
                    std::mem::size_of::<i32>(),
                )
            },
            call,
        )
    }

    fn set_operand_value_from_memory(
        model: *mut ANeuralNetworksModel,
        index: i32,
        blob: &QuantizedGatedGeluSharedBlob,
        region_name: &'static str,
        call: &'static str,
    ) -> Result<(), MediaTekNnapiError> {
        let region = blob.region(region_name)?;
        check(
            unsafe {
                ANeuralNetworksModel_setOperandValueFromMemory(
                    model,
                    index,
                    blob.memory.0,
                    region.offset,
                    region.length,
                )
            },
            call,
        )
    }

    fn set_operand_value_from_f32_blob(
        model: *mut ANeuralNetworksModel,
        index: i32,
        blob: &GatedGeluFfnF32SharedBlob,
        region_name: &'static str,
        call: &'static str,
    ) -> Result<(), MediaTekNnapiError> {
        let region = blob.region(region_name)?;
        check(
            unsafe {
                ANeuralNetworksModel_setOperandValueFromMemory(
                    model,
                    index,
                    blob.memory.0,
                    region.offset,
                    region.length,
                )
            },
            call,
        )
    }

    fn choose_accelerator(requested: &str) -> Result<ChosenDevice, MediaTekNnapiError> {
        let requested_lower = requested.to_ascii_lowercase();
        let mut count = 0u32;
        check(
            unsafe { ANeuralNetworks_getDeviceCount(&mut count) },
            "ANeuralNetworks_getDeviceCount",
        )?;
        for index in 0..count {
            let mut device = ptr::null_mut();
            check(
                unsafe { ANeuralNetworks_getDevice(index, &mut device) },
                "ANeuralNetworks_getDevice",
            )?;
            let info = device_info(device)?;
            if !info.name().to_ascii_lowercase().contains(&requested_lower) {
                continue;
            }
            if info.device_type() == ANEURALNETWORKS_DEVICE_CPU {
                return Err(MediaTekNnapiError::CpuDeviceRejected {
                    name: info.name().to_string(),
                });
            }
            if info.device_type() == MEDIATEK_NNAPI_DEVICE_ACCELERATOR {
                return Ok(ChosenDevice { device, info });
            }
        }
        Err(MediaTekNnapiError::NoMatchingAccelerator {
            requested: requested.to_string(),
        })
    }

    fn device_info(
        device: *const ANeuralNetworksDevice,
    ) -> Result<MediaTekNnapiDeviceInfo, MediaTekNnapiError> {
        let mut name_ptr = ptr::null();
        let mut device_type = 0i32;
        let mut feature_level = 0i64;
        let mut version_ptr = ptr::null();
        check(
            unsafe { ANeuralNetworksDevice_getName(device, &mut name_ptr) },
            "ANeuralNetworksDevice_getName",
        )?;
        check(
            unsafe { ANeuralNetworksDevice_getType(device, &mut device_type) },
            "ANeuralNetworksDevice_getType",
        )?;
        check(
            unsafe { ANeuralNetworksDevice_getFeatureLevel(device, &mut feature_level) },
            "ANeuralNetworksDevice_getFeatureLevel",
        )?;
        check(
            unsafe { ANeuralNetworksDevice_getVersion(device, &mut version_ptr) },
            "ANeuralNetworksDevice_getVersion",
        )?;
        Ok(MediaTekNnapiDeviceInfo::new(
            cstr_to_string(name_ptr),
            device_type,
            feature_level,
            cstr_to_string(version_ptr),
        ))
    }

    fn supported_ops(
        model: *const ANeuralNetworksModel,
        devices: &[*const ANeuralNetworksDevice],
    ) -> Result<MediaTekNnapiSupportedOps, MediaTekNnapiError> {
        let mut supported = [false; 2];
        check(
            unsafe {
                ANeuralNetworksModel_getSupportedOperationsForDevices(
                    model,
                    devices.as_ptr(),
                    devices.len() as u32,
                    supported.as_mut_ptr(),
                )
            },
            "ANeuralNetworksModel_getSupportedOperationsForDevices",
        )?;
        Ok(MediaTekNnapiSupportedOps::new(supported[0], supported[1]))
    }

    fn supported_gated_gelu_ops(
        model: *const ANeuralNetworksModel,
        devices: &[*const ANeuralNetworksDevice],
        hybrid_down_f32: bool,
    ) -> Result<MediaTekGatedGeluFfnSupportedOps, MediaTekNnapiError> {
        if hybrid_down_f32 {
            let mut raw_supported = [false; 15];
            check(
                unsafe {
                    ANeuralNetworksModel_getSupportedOperationsForDevices(
                        model,
                        devices.as_ptr(),
                        devices.len() as u32,
                        raw_supported.as_mut_ptr(),
                    )
                },
                "ANeuralNetworksModel_getSupportedOperationsForDevices",
            )?;
            let mut supported = [false; 16];
            supported[..14].copy_from_slice(&raw_supported[..14]);
            supported[14] = true;
            supported[15] = raw_supported[14];
            Ok(MediaTekGatedGeluFfnSupportedOps::from_bools(supported))
        } else {
            let mut supported = [false; 16];
            check(
                unsafe {
                    ANeuralNetworksModel_getSupportedOperationsForDevices(
                        model,
                        devices.as_ptr(),
                        devices.len() as u32,
                        supported.as_mut_ptr(),
                    )
                },
                "ANeuralNetworksModel_getSupportedOperationsForDevices",
            )?;
            Ok(MediaTekGatedGeluFfnSupportedOps::from_bools(supported))
        }
    }

    fn supported_gated_gelu_f32_ops(
        model: *const ANeuralNetworksModel,
        devices: &[*const ANeuralNetworksDevice],
    ) -> Result<MediaTekGatedGeluFfnSupportedOps, MediaTekNnapiError> {
        let mut raw_supported = [false; 13];
        check(
            unsafe {
                ANeuralNetworksModel_getSupportedOperationsForDevices(
                    model,
                    devices.as_ptr(),
                    devices.len() as u32,
                    raw_supported.as_mut_ptr(),
                )
            },
            "ANeuralNetworksModel_getSupportedOperationsForDevices",
        )?;
        let mut supported = [false; 16];
        supported[0] = raw_supported[0];
        supported[1] = raw_supported[1];
        supported[2] = true;
        supported[3] = true;
        supported[4] = raw_supported[2];
        supported[5] = raw_supported[3];
        supported[6] = raw_supported[4];
        supported[7] = raw_supported[5];
        supported[8] = raw_supported[6];
        supported[9] = raw_supported[7];
        supported[10] = raw_supported[8];
        supported[11] = raw_supported[9];
        supported[12] = raw_supported[10];
        supported[13] = raw_supported[11];
        supported[14] = true;
        supported[15] = raw_supported[12];
        Ok(MediaTekGatedGeluFfnSupportedOps::from_bools(supported))
    }

    fn query_duration(execution: *const ANeuralNetworksExecution, code: i32) -> Option<u64> {
        let mut duration = 0u64;
        let result =
            unsafe { ANeuralNetworksExecution_getDuration(execution, code, &mut duration) };
        if result == 0 && duration != u64::MAX {
            Some(duration)
        } else {
            None
        }
    }

    fn check(code: i32, call: &'static str) -> Result<(), MediaTekNnapiError> {
        if code == 0 {
            Ok(())
        } else {
            Err(MediaTekNnapiError::NnapiCall { call, code })
        }
    }

    fn cstr_to_string(ptr: *const c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaTekQuantParams {
    scale: f32,
    zero_point: i32,
}

impl MediaTekQuantParams {
    pub const fn new(scale: f32, zero_point: i32) -> Self {
        Self { scale, zero_point }
    }

    pub const fn scale(self) -> f32 {
        self.scale
    }

    pub const fn zero_point(self) -> i32 {
        self.zero_point
    }

    fn validate(self, tensor: &'static str) -> Result<(), MediaTekQuantizedMlpTensorViewError> {
        if !self.scale.is_finite() || self.scale <= 0.0 || !(0..=255).contains(&self.zero_point) {
            return Err(MediaTekQuantizedMlpTensorViewError::InvalidQuantParam { tensor });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTekGatedGeluFfnShape {
    input_size: usize,
    ffn_inner_size: usize,
    output_size: usize,
}

impl MediaTekGatedGeluFfnShape {
    pub const fn new(input_size: usize, ffn_inner_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            ffn_inner_size,
            output_size,
        }
    }

    pub const fn input_size(self) -> usize {
        self.input_size
    }

    pub const fn ffn_inner_size(self) -> usize {
        self.ffn_inner_size
    }

    pub const fn output_size(self) -> usize {
        self.output_size
    }

    fn validate(self) -> Result<(), MediaTekGatedGeluFfnTensorViewError> {
        if self.input_size == 0 {
            return Err(MediaTekGatedGeluFfnTensorViewError::ZeroDim {
                dimension: "input_size",
            });
        }
        if self.ffn_inner_size == 0 {
            return Err(MediaTekGatedGeluFfnTensorViewError::ZeroDim {
                dimension: "ffn_inner_size",
            });
        }
        if self.output_size == 0 {
            return Err(MediaTekGatedGeluFfnTensorViewError::ZeroDim {
                dimension: "output_size",
            });
        }
        Ok(())
    }

    fn gate_up_len(self) -> Result<usize, MediaTekGatedGeluFfnTensorViewError> {
        self.ffn_inner_size.checked_mul(self.input_size).ok_or(
            MediaTekGatedGeluFfnTensorViewError::LengthOverflow {
                tensor: "gate_weight",
            },
        )
    }

    fn down_len(self) -> Result<usize, MediaTekGatedGeluFfnTensorViewError> {
        self.output_size.checked_mul(self.ffn_inner_size).ok_or(
            MediaTekGatedGeluFfnTensorViewError::LengthOverflow {
                tensor: "down_weight",
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaTekQuantizedGatedGeluFfnOwnedWeights {
    shape: MediaTekGatedGeluFfnShape,
    quant_params: MediaTekQuantizedGatedGeluFfnQuantParams,
    gate_weight: Vec<u8>,
    up_weight: Vec<u8>,
    down_weight: Vec<u8>,
    down_weight_f32: Option<Vec<f32>>,
}

impl MediaTekQuantizedGatedGeluFfnOwnedWeights {
    pub fn new(
        shape: MediaTekGatedGeluFfnShape,
        quant_params: MediaTekQuantizedGatedGeluFfnQuantParams,
        gate_weight: Vec<u8>,
        up_weight: Vec<u8>,
        down_weight: Vec<u8>,
    ) -> Result<Self, MediaTekNnapiError> {
        let weights = Self {
            shape,
            quant_params,
            gate_weight,
            up_weight,
            down_weight,
            down_weight_f32: None,
        };
        validate_quantized_gated_gelu_owned_weights(&weights)?;
        Ok(weights)
    }

    pub fn new_with_f32_down(
        shape: MediaTekGatedGeluFfnShape,
        quant_params: MediaTekQuantizedGatedGeluFfnQuantParams,
        gate_weight: Vec<u8>,
        up_weight: Vec<u8>,
        down_weight: Vec<u8>,
        down_weight_f32: Vec<f32>,
    ) -> Result<Self, MediaTekNnapiError> {
        let weights = Self {
            shape,
            quant_params,
            gate_weight,
            up_weight,
            down_weight,
            down_weight_f32: Some(down_weight_f32),
        };
        validate_quantized_gated_gelu_owned_weights(&weights)?;
        Ok(weights)
    }

    pub const fn shape(&self) -> MediaTekGatedGeluFfnShape {
        self.shape
    }

    pub const fn quant_params(&self) -> MediaTekQuantizedGatedGeluFfnQuantParams {
        self.quant_params
    }

    pub fn gate_weight(&self) -> &[u8] {
        &self.gate_weight
    }

    pub fn up_weight(&self) -> &[u8] {
        &self.up_weight
    }

    pub fn down_weight(&self) -> &[u8] {
        &self.down_weight
    }

    pub fn down_weight_f32(&self) -> Option<&[f32]> {
        self.down_weight_f32.as_deref()
    }

    pub const fn output_len(&self) -> usize {
        self.shape.output_size()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GatedGeluFfnF32BlobRegion {
    name: &'static str,
    offset: usize,
    length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GatedGeluFfnF32BlobLayout {
    regions: [GatedGeluFfnF32BlobRegion; 6],
    total_len: usize,
}

impl GatedGeluFfnF32BlobLayout {
    #[allow(dead_code)]
    fn regions(&self) -> &[GatedGeluFfnF32BlobRegion; 6] {
        &self.regions
    }

    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    fn region(&self, name: &'static str) -> Result<GatedGeluFfnF32BlobRegion, MediaTekNnapiError> {
        self.regions
            .iter()
            .copied()
            .find(|region| region.name == name)
            .ok_or_else(|| MediaTekNnapiError::InvalidInput {
                reason: format!("missing f32 gated GELU shared blob region {name}"),
            })
    }
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn gated_gelu_f32_blob_layout(
    shape: MediaTekGatedGeluFfnShape,
    page_size: usize,
) -> Result<GatedGeluFfnF32BlobLayout, MediaTekNnapiError> {
    let mut regions = [GatedGeluFfnF32BlobRegion {
        name: "",
        offset: 0,
        length: 0,
    }; 6];
    let mut index = 0usize;
    let mut offset = 0usize;

    push_gated_gelu_f32_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "gate_weight",
        checked_quantized_gated_gelu_elements(
            "gate_weight",
            shape.ffn_inner_size(),
            shape.input_size(),
        )?,
    )?;
    push_gated_gelu_f32_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "gate_bias",
        shape.ffn_inner_size(),
    )?;
    push_gated_gelu_f32_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "up_weight",
        checked_quantized_gated_gelu_elements(
            "up_weight",
            shape.ffn_inner_size(),
            shape.input_size(),
        )?,
    )?;
    push_gated_gelu_f32_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "up_bias",
        shape.ffn_inner_size(),
    )?;
    push_gated_gelu_f32_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "down_weight",
        checked_quantized_gated_gelu_elements(
            "down_weight",
            shape.output_size(),
            shape.ffn_inner_size(),
        )?,
    )?;
    push_gated_gelu_f32_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "down_bias",
        shape.output_size(),
    )?;
    debug_assert_eq!(index, regions.len());

    let total_len = round_up_quantized_gated_gelu_blob_size(offset, page_size)?;
    Ok(GatedGeluFfnF32BlobLayout { regions, total_len })
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn push_gated_gelu_f32_blob_region(
    regions: &mut [GatedGeluFfnF32BlobRegion; 6],
    index: &mut usize,
    offset: &mut usize,
    name: &'static str,
    elements: usize,
) -> Result<(), MediaTekNnapiError> {
    let element_size = std::mem::size_of::<f32>();
    let aligned = align_offset(*offset, element_size)?;
    let length =
        elements
            .checked_mul(element_size)
            .ok_or_else(|| MediaTekNnapiError::InvalidInput {
                reason: format!("{name} byte length overflows usize"),
            })?;
    let next = aligned
        .checked_add(length)
        .ok_or_else(|| MediaTekNnapiError::InvalidInput {
            reason: "f32 gated GELU shared blob length overflows usize".to_string(),
        })?;
    regions[*index] = GatedGeluFfnF32BlobRegion {
        name,
        offset: aligned,
        length,
    };
    *index += 1;
    *offset = next;
    Ok(())
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn f32_as_ne_bytes(s: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding bytes, `u8` has alignment 1, and the returned
    // byte slice is immutable and tied to the lifetime of the source slice.
    unsafe { core::slice::from_raw_parts(s.as_ptr().cast::<u8>(), core::mem::size_of_val(s)) }
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn gated_gelu_f32_cache_token(
    variant: u8,
    device_name: &str,
    shape: MediaTekGatedGeluFfnShape,
    batch: usize,
    gate: &[f32],
    up: &[f32],
    down: &[f32],
) -> [u8; NNAPI_CACHE_TOKEN_LEN] {
    use sha2::Digest;

    let mut hasher = sha2::Sha256::new();
    hasher.update(MK28_CACHE_ABI_VERSION.to_le_bytes());
    hasher.update([variant]);
    hasher.update(device_name.as_bytes());
    for value in [
        shape.input_size(),
        shape.ffn_inner_size(),
        shape.output_size(),
        batch,
    ] {
        hasher.update((value as u64).to_le_bytes());
    }
    hasher.update(f32_as_ne_bytes(gate));
    hasher.update(f32_as_ne_bytes(up));
    hasher.update(f32_as_ne_bytes(down));
    let digest = hasher.finalize();
    let mut token = [0u8; NNAPI_CACHE_TOKEN_LEN];
    token.copy_from_slice(&digest);
    token
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizedGatedGeluBlobRegion {
    name: &'static str,
    offset: usize,
    length: usize,
    element_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QuantizedGatedGeluSupportBlobLayout {
    regions: [QuantizedGatedGeluBlobRegion; 10],
    total_len: usize,
}

impl QuantizedGatedGeluSupportBlobLayout {
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    fn regions(&self) -> &[QuantizedGatedGeluBlobRegion; 10] {
        &self.regions
    }
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaTekQuantizedGatedGeluFfnQuantParams {
    pub input: MediaTekQuantParams,
    pub gate_weight: MediaTekQuantParams,
    pub gate_bias: MediaTekQuantParams,
    pub gate: MediaTekQuantParams,
    pub up_weight: MediaTekQuantParams,
    pub up_bias: MediaTekQuantParams,
    pub up: MediaTekQuantParams,
    pub square: MediaTekQuantParams,
    pub cube: MediaTekQuantParams,
    pub coeff_044715: MediaTekQuantParams,
    pub poly_scale: MediaTekQuantParams,
    pub poly: MediaTekQuantParams,
    pub coeff_sqrt_2_over_pi: MediaTekQuantParams,
    pub tanh_arg: MediaTekQuantParams,
    pub tanh_output: MediaTekQuantParams,
    pub one: MediaTekQuantParams,
    pub one_plus: MediaTekQuantParams,
    pub gelu_factor: MediaTekQuantParams,
    pub half: MediaTekQuantParams,
    pub gelu: MediaTekQuantParams,
    pub gated: MediaTekQuantParams,
    pub down_weight: MediaTekQuantParams,
    pub down_bias: MediaTekQuantParams,
    pub output: MediaTekQuantParams,
}

impl MediaTekQuantizedGatedGeluFfnQuantParams {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        input: MediaTekQuantParams,
        gate_weight: MediaTekQuantParams,
        gate: MediaTekQuantParams,
        up_weight: MediaTekQuantParams,
        up: MediaTekQuantParams,
        square: MediaTekQuantParams,
        cube: MediaTekQuantParams,
        coeff_044715: MediaTekQuantParams,
        poly_scale: MediaTekQuantParams,
        poly: MediaTekQuantParams,
        coeff_sqrt_2_over_pi: MediaTekQuantParams,
        tanh_arg: MediaTekQuantParams,
        tanh_output: MediaTekQuantParams,
        one: MediaTekQuantParams,
        one_plus: MediaTekQuantParams,
        gelu_factor: MediaTekQuantParams,
        half: MediaTekQuantParams,
        gelu: MediaTekQuantParams,
        gated: MediaTekQuantParams,
        down_weight: MediaTekQuantParams,
        output: MediaTekQuantParams,
    ) -> Self {
        Self {
            input,
            gate_weight,
            gate_bias: MediaTekQuantParams::new(input.scale() * gate_weight.scale(), 0),
            gate,
            up_weight,
            up_bias: MediaTekQuantParams::new(input.scale() * up_weight.scale(), 0),
            up,
            square,
            cube,
            coeff_044715,
            poly_scale,
            poly,
            coeff_sqrt_2_over_pi,
            tanh_arg,
            tanh_output,
            one,
            one_plus,
            gelu_factor,
            half,
            gelu,
            gated,
            down_weight,
            down_bias: MediaTekQuantParams::new(gated.scale() * down_weight.scale(), 0),
            output,
        }
    }
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn quantized_gated_gelu_probe_quant_params() -> MediaTekQuantizedGatedGeluFfnQuantParams {
    let tensor = MediaTekQuantParams::new(0.01, 128);
    let non_negative = MediaTekQuantParams::new(0.01, 0);
    MediaTekQuantizedGatedGeluFfnQuantParams::new(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        non_negative,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        MediaTekQuantParams::new(1.0 / 128.0, 128),
        MediaTekQuantParams::new(1.0 / 128.0, 0),
        MediaTekQuantParams::new(1.0 / 128.0, 0),
        tensor,
        MediaTekQuantParams::new(1.0 / 255.0, 0),
        tensor,
        tensor,
        tensor,
        tensor,
    )
}

#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn validate_quantized_gated_gelu_quant_params(
    params: MediaTekQuantizedGatedGeluFfnQuantParams,
) -> Result<(), MediaTekNnapiError> {
    for (name, params) in [
        ("input", params.input),
        ("gate_weight", params.gate_weight),
        ("gate_bias", params.gate_bias),
        ("gate", params.gate),
        ("up_weight", params.up_weight),
        ("up_bias", params.up_bias),
        ("up", params.up),
        ("square", params.square),
        ("cube", params.cube),
        ("coeff_044715", params.coeff_044715),
        ("poly_scale", params.poly_scale),
        ("poly", params.poly),
        ("coeff_sqrt_2_over_pi", params.coeff_sqrt_2_over_pi),
        ("tanh_arg", params.tanh_arg),
        ("tanh_output", params.tanh_output),
        ("one", params.one),
        ("one_plus", params.one_plus),
        ("gelu_factor", params.gelu_factor),
        ("half", params.half),
        ("gelu", params.gelu),
        ("gated", params.gated),
        ("down_weight", params.down_weight),
        ("down_bias", params.down_bias),
        ("output", params.output),
    ] {
        validate_quantized_gated_gelu_one_quant_param(params, name)?;
    }
    validate_quantized_gated_gelu_bias_param(
        "gate_bias",
        params.gate_bias,
        params.input.scale() * params.gate_weight.scale(),
    )?;
    validate_quantized_gated_gelu_bias_param(
        "up_bias",
        params.up_bias,
        params.input.scale() * params.up_weight.scale(),
    )?;
    validate_quantized_gated_gelu_bias_param(
        "down_bias",
        params.down_bias,
        params.gated.scale() * params.down_weight.scale(),
    )?;
    Ok(())
}
fn validate_quantized_gated_gelu_one_quant_param(
    params: MediaTekQuantParams,
    name: &'static str,
) -> Result<(), MediaTekNnapiError> {
    if !params.scale().is_finite()
        || params.scale() <= 0.0
        || !(0..=255).contains(&params.zero_point())
    {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: format!("{name} must have finite positive scale and u8 zero_point"),
        });
    }
    Ok(())
}

fn validate_quantized_gated_gelu_bias_param(
    name: &'static str,
    params: MediaTekQuantParams,
    expected_scale: f32,
) -> Result<(), MediaTekNnapiError> {
    let tolerance = (expected_scale.abs() * 1.0e-6).max(1.0e-12);
    if params.zero_point() != 0 || (params.scale() - expected_scale).abs() > tolerance {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: format!("{name} must have zero_point 0 and FC bias scale"),
        });
    }
    Ok(())
}

fn validate_quantized_gated_gelu_support_shape(
    shape: MediaTekGatedGeluFfnShape,
) -> Result<(), MediaTekNnapiError> {
    validate_quantized_gated_gelu_dim("input_size", shape.input_size())?;
    validate_quantized_gated_gelu_dim("ffn_inner_size", shape.ffn_inner_size())?;
    validate_quantized_gated_gelu_dim("output_size", shape.output_size())?;
    let _ = quantized_gated_gelu_support_blob_layout(shape, 1)?;
    Ok(())
}

fn validate_quantized_gated_gelu_owned_weights(
    weights: &MediaTekQuantizedGatedGeluFfnOwnedWeights,
) -> Result<(), MediaTekNnapiError> {
    validate_quantized_gated_gelu_support_shape(weights.shape())?;
    validate_quantized_gated_gelu_quant_params(weights.quant_params())?;
    let gate_up_len =
        weights
            .shape()
            .gate_up_len()
            .map_err(|err| MediaTekNnapiError::InvalidInput {
                reason: err.to_string(),
            })?;
    expect_quantized_gated_gelu_len("gate_weight", gate_up_len, weights.gate_weight().len())?;
    expect_quantized_gated_gelu_len("up_weight", gate_up_len, weights.up_weight().len())?;
    let down_len = weights
        .shape()
        .down_len()
        .map_err(|err| MediaTekNnapiError::InvalidInput {
            reason: err.to_string(),
        })?;
    expect_quantized_gated_gelu_len("down_weight", down_len, weights.down_weight().len())?;
    if let Some(down_weight_f32) = weights.down_weight_f32() {
        expect_quantized_gated_gelu_len("down_weight_f32", down_len, down_weight_f32.len())?;
        for (index, value) in down_weight_f32.iter().enumerate() {
            if !value.is_finite() {
                return Err(MediaTekNnapiError::InvalidInput {
                    reason: format!("down_weight_f32 contains non-finite value at {index}"),
                });
            }
        }
    }
    Ok(())
}

fn validate_quantized_gated_gelu_input(
    shape: MediaTekGatedGeluFfnShape,
    input: &[u8],
) -> Result<(), MediaTekNnapiError> {
    validate_quantized_gated_gelu_support_shape(shape)?;
    expect_quantized_gated_gelu_len("input", shape.input_size(), input.len())
}

fn validate_quantized_gated_gelu_hybrid_stage_probe(
    weights: &MediaTekQuantizedGatedGeluFfnOwnedWeights,
) -> Result<(), MediaTekNnapiError> {
    if weights.down_weight_f32().is_none() {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: "quantized gated GELU stage probe requires hybrid f32-down weights".to_string(),
        });
    }
    Ok(())
}

#[cfg(any(target_arch = "aarch64", test))]
fn validate_quantized_gated_gelu_stage_output_len(
    stage: MediaTekQuantizedGatedGeluFfnStage,
    shape: MediaTekGatedGeluFfnShape,
    actual: usize,
) -> Result<(), MediaTekNnapiError> {
    let expected = stage.output_len(shape);
    if actual != expected {
        return Err(MediaTekNnapiError::InvalidOutputLength { expected, actual });
    }
    Ok(())
}

fn expect_quantized_gated_gelu_len(
    tensor: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), MediaTekNnapiError> {
    if actual != expected {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: format!("{tensor} length mismatch: expected {expected}, got {actual}"),
        });
    }
    Ok(())
}

fn validate_quantized_gated_gelu_dim(
    name: &'static str,
    value: usize,
) -> Result<(), MediaTekNnapiError> {
    if value == 0 {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: format!("{name} must be non-zero"),
        });
    }
    if value > u32::MAX as usize {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: format!("{name} exceeds NNAPI u32 dimension limit"),
        });
    }
    Ok(())
}

fn quantized_gated_gelu_support_blob_layout(
    shape: MediaTekGatedGeluFfnShape,
    page_size: usize,
) -> Result<QuantizedGatedGeluSupportBlobLayout, MediaTekNnapiError> {
    let mut regions = [QuantizedGatedGeluBlobRegion {
        name: "",
        offset: 0,
        length: 0,
        element_size: 1,
    }; 10];
    let mut index = 0usize;
    let mut offset = 0usize;

    push_quantized_gated_gelu_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "gate_weight",
        checked_quantized_gated_gelu_elements(
            "gate_weight",
            shape.ffn_inner_size(),
            shape.input_size(),
        )?,
        1,
    )?;
    push_quantized_gated_gelu_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "gate_bias",
        shape.ffn_inner_size(),
        std::mem::size_of::<i32>(),
    )?;
    push_quantized_gated_gelu_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "up_weight",
        checked_quantized_gated_gelu_elements(
            "up_weight",
            shape.ffn_inner_size(),
            shape.input_size(),
        )?,
        1,
    )?;
    push_quantized_gated_gelu_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "up_bias",
        shape.ffn_inner_size(),
        std::mem::size_of::<i32>(),
    )?;
    for name in [
        "gelu_coeff_044715",
        "gelu_coeff_sqrt_2_over_pi",
        "gelu_one",
        "gelu_half",
    ] {
        push_quantized_gated_gelu_blob_region(
            &mut regions,
            &mut index,
            &mut offset,
            name,
            shape.ffn_inner_size(),
            1,
        )?;
    }
    push_quantized_gated_gelu_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "down_weight",
        checked_quantized_gated_gelu_elements(
            "down_weight",
            shape.output_size(),
            shape.ffn_inner_size(),
        )?,
        1,
    )?;
    push_quantized_gated_gelu_blob_region(
        &mut regions,
        &mut index,
        &mut offset,
        "down_bias",
        shape.output_size(),
        std::mem::size_of::<i32>(),
    )?;
    debug_assert_eq!(index, regions.len());

    let total_len = round_up_quantized_gated_gelu_blob_size(offset, page_size)?;
    Ok(QuantizedGatedGeluSupportBlobLayout { regions, total_len })
}

fn checked_quantized_gated_gelu_elements(
    tensor: &'static str,
    rows: usize,
    cols: usize,
) -> Result<usize, MediaTekNnapiError> {
    rows.checked_mul(cols)
        .ok_or_else(|| MediaTekNnapiError::InvalidInput {
            reason: format!("{tensor} element count overflows usize"),
        })
}

fn push_quantized_gated_gelu_blob_region(
    regions: &mut [QuantizedGatedGeluBlobRegion; 10],
    index: &mut usize,
    offset: &mut usize,
    name: &'static str,
    elements: usize,
    element_size: usize,
) -> Result<(), MediaTekNnapiError> {
    let aligned = align_offset(*offset, element_size)?;
    let length =
        elements
            .checked_mul(element_size)
            .ok_or_else(|| MediaTekNnapiError::InvalidInput {
                reason: format!("{name} byte length overflows usize"),
            })?;
    let next = aligned
        .checked_add(length)
        .ok_or_else(|| MediaTekNnapiError::InvalidInput {
            reason: "quantized gated GELU shared blob length overflows usize".to_string(),
        })?;
    regions[*index] = QuantizedGatedGeluBlobRegion {
        name,
        offset: aligned,
        length,
        element_size,
    };
    *index += 1;
    *offset = next;
    Ok(())
}

fn align_offset(offset: usize, alignment: usize) -> Result<usize, MediaTekNnapiError> {
    if alignment == 0 {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: "alignment must be non-zero".to_string(),
        });
    }
    let remainder = offset % alignment;
    if remainder == 0 {
        Ok(offset)
    } else {
        offset
            .checked_add(alignment - remainder)
            .ok_or_else(|| MediaTekNnapiError::InvalidInput {
                reason: "aligned offset overflows usize".to_string(),
            })
    }
}

fn round_up_quantized_gated_gelu_blob_size(
    size: usize,
    page_size: usize,
) -> Result<usize, MediaTekNnapiError> {
    align_offset(size, page_size)
}

#[cfg(any(unix, target_os = "android"))]
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
fn quantized_gated_gelu_page_size() -> Result<usize, MediaTekNnapiError> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: "sysconf(_SC_PAGESIZE) returned non-positive page size".to_string(),
        });
    }
    Ok(page_size as usize)
}

#[cfg(not(any(unix, target_os = "android")))]
fn quantized_gated_gelu_page_size() -> Result<usize, MediaTekNnapiError> {
    Err(MediaTekNnapiError::InvalidInput {
        reason: "quantized gated GELU support probe requires a Unix page-size query".to_string(),
    })
}

#[derive(Debug, Clone, Copy)]
pub struct MediaTekGatedGeluFfnTensorView<'a> {
    shape: MediaTekGatedGeluFfnShape,
    gate_weight: &'a [f32],
    up_weight: &'a [f32],
    down_weight: &'a [f32],
    input: &'a [f32],
}

impl<'a> MediaTekGatedGeluFfnTensorView<'a> {
    pub fn new(
        shape: MediaTekGatedGeluFfnShape,
        gate_weight: &'a [f32],
        up_weight: &'a [f32],
        down_weight: &'a [f32],
        input: &'a [f32],
    ) -> Result<Self, MediaTekGatedGeluFfnTensorViewError> {
        shape.validate()?;
        expect_gated_gelu_len("gate_weight", shape.gate_up_len()?, gate_weight.len())?;
        expect_gated_gelu_len("up_weight", shape.gate_up_len()?, up_weight.len())?;
        expect_gated_gelu_len("down_weight", shape.down_len()?, down_weight.len())?;
        expect_gated_gelu_len("input", shape.input_size, input.len())?;
        expect_finite_f32("gate_weight", gate_weight)?;
        expect_finite_f32("up_weight", up_weight)?;
        expect_finite_f32("down_weight", down_weight)?;
        expect_finite_f32("input", input)?;

        Ok(Self {
            shape,
            gate_weight,
            up_weight,
            down_weight,
            input,
        })
    }

    pub const fn shape(&self) -> MediaTekGatedGeluFfnShape {
        self.shape
    }

    pub const fn gate_weight(&self) -> &'a [f32] {
        self.gate_weight
    }

    pub const fn up_weight(&self) -> &'a [f32] {
        self.up_weight
    }

    pub const fn down_weight(&self) -> &'a [f32] {
        self.down_weight
    }

    pub const fn input(&self) -> &'a [f32] {
        self.input
    }

    pub const fn output_len(&self) -> usize {
        self.shape.output_size
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MediaTekGatedGeluFfnBatchedTensorView<'a> {
    shape: MediaTekGatedGeluFfnShape,
    gate_weight: &'a [f32],
    up_weight: &'a [f32],
    down_weight: &'a [f32],
    input: &'a [f32],
}

impl<'a> MediaTekGatedGeluFfnBatchedTensorView<'a> {
    pub fn new(
        shape: MediaTekGatedGeluFfnShape,
        gate_weight: &'a [f32],
        up_weight: &'a [f32],
        down_weight: &'a [f32],
        input: &'a [f32],
    ) -> Result<Self, MediaTekGatedGeluFfnTensorViewError> {
        shape.validate()?;
        expect_gated_gelu_len("gate_weight", shape.gate_up_len()?, gate_weight.len())?;
        expect_gated_gelu_len("up_weight", shape.gate_up_len()?, up_weight.len())?;
        expect_gated_gelu_len("down_weight", shape.down_len()?, down_weight.len())?;
        expect_finite_f32("gate_weight", gate_weight)?;
        expect_finite_f32("up_weight", up_weight)?;
        expect_finite_f32("down_weight", down_weight)?;
        expect_finite_f32("input", input)?;

        Ok(Self {
            shape,
            gate_weight,
            up_weight,
            down_weight,
            input,
        })
    }

    pub const fn shape(&self) -> MediaTekGatedGeluFfnShape {
        self.shape
    }

    pub const fn gate_weight(&self) -> &'a [f32] {
        self.gate_weight
    }

    pub const fn up_weight(&self) -> &'a [f32] {
        self.up_weight
    }

    pub const fn down_weight(&self) -> &'a [f32] {
        self.down_weight
    }

    pub const fn input(&self) -> &'a [f32] {
        self.input
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaTekGatedGeluFfnOwnedWeights {
    shape: MediaTekGatedGeluFfnShape,
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
}

impl MediaTekGatedGeluFfnOwnedWeights {
    pub fn new(
        shape: MediaTekGatedGeluFfnShape,
        gate_weight: Vec<f32>,
        up_weight: Vec<f32>,
        down_weight: Vec<f32>,
    ) -> Result<Self, MediaTekGatedGeluFfnTensorViewError> {
        shape.validate()?;
        expect_gated_gelu_len("gate_weight", shape.gate_up_len()?, gate_weight.len())?;
        expect_gated_gelu_len("up_weight", shape.gate_up_len()?, up_weight.len())?;
        expect_gated_gelu_len("down_weight", shape.down_len()?, down_weight.len())?;
        expect_finite_f32("gate_weight", &gate_weight)?;
        expect_finite_f32("up_weight", &up_weight)?;
        expect_finite_f32("down_weight", &down_weight)?;
        Ok(Self {
            shape,
            gate_weight,
            up_weight,
            down_weight,
        })
    }

    pub const fn shape(&self) -> MediaTekGatedGeluFfnShape {
        self.shape
    }

    pub fn gate_weight(&self) -> &[f32] {
        &self.gate_weight
    }

    pub fn up_weight(&self) -> &[f32] {
        &self.up_weight
    }

    pub fn down_weight(&self) -> &[f32] {
        &self.down_weight
    }

    pub const fn output_len(&self) -> usize {
        self.shape.output_size
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaTekGatedGeluFfnTensorViewError {
    ZeroDim {
        dimension: &'static str,
    },
    LengthOverflow {
        tensor: &'static str,
    },
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite {
        tensor: &'static str,
        index: usize,
    },
}

impl fmt::Display for MediaTekGatedGeluFfnTensorViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDim { dimension } => {
                write!(f, "MediaTek gated GELU FFN {dimension} must be non-zero")
            }
            Self::LengthOverflow { tensor } => {
                write!(f, "MediaTek gated GELU FFN {tensor} length overflow")
            }
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "MediaTek gated GELU FFN {tensor} length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFinite { tensor, index } => write!(
                f,
                "MediaTek gated GELU FFN {tensor} has non-finite value at index {index}"
            ),
        }
    }
}

impl std::error::Error for MediaTekGatedGeluFfnTensorViewError {}

fn validate_gated_gelu_f32_batched_input(
    shape: MediaTekGatedGeluFfnShape,
    batch: usize,
    input: &[f32],
) -> Result<(usize, usize), MediaTekNnapiError> {
    let (input_len, output_len) = validate_gated_gelu_f32_batched_shape(shape, batch)?;
    if input.len() != input_len {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: format!(
                "input length mismatch: expected {}, got {}",
                input_len,
                input.len()
            ),
        });
    }
    Ok((input_len, output_len))
}

fn validate_gated_gelu_f32_batched_shape(
    shape: MediaTekGatedGeluFfnShape,
    batch: usize,
) -> Result<(usize, usize), MediaTekNnapiError> {
    validate_gated_gelu_f32_dim("batch", batch)?;
    validate_gated_gelu_f32_dim("input_size", shape.input_size())?;
    validate_gated_gelu_f32_dim("ffn_inner_size", shape.ffn_inner_size())?;
    validate_gated_gelu_f32_dim("output_size", shape.output_size())?;
    let input_len = checked_gated_gelu_f32_batched_len("input", batch, shape.input_size())?;
    let output_len = checked_gated_gelu_f32_batched_len("output", batch, shape.output_size())?;
    Ok((input_len, output_len))
}

fn validate_gated_gelu_f32_dim(name: &'static str, value: usize) -> Result<(), MediaTekNnapiError> {
    if value == 0 {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: format!("{name} must be non-zero"),
        });
    }
    if value > u32::MAX as usize {
        return Err(MediaTekNnapiError::InvalidInput {
            reason: format!("{name} exceeds NNAPI u32 dimension limit"),
        });
    }
    Ok(())
}

fn checked_gated_gelu_f32_batched_len(
    tensor: &'static str,
    batch: usize,
    feature: usize,
) -> Result<usize, MediaTekNnapiError> {
    batch
        .checked_mul(feature)
        .ok_or_else(|| MediaTekNnapiError::InvalidInput {
            reason: format!("{tensor} length overflows usize"),
        })
}

fn expect_gated_gelu_len(
    tensor: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), MediaTekGatedGeluFfnTensorViewError> {
    if expected != actual {
        return Err(MediaTekGatedGeluFfnTensorViewError::LengthMismatch {
            tensor,
            expected,
            actual,
        });
    }
    Ok(())
}

fn expect_finite_f32(
    tensor: &'static str,
    data: &[f32],
) -> Result<(), MediaTekGatedGeluFfnTensorViewError> {
    for (index, value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(MediaTekGatedGeluFfnTensorViewError::NonFinite { tensor, index });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaTekQuantizedMlpShape {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl MediaTekQuantizedMlpShape {
    pub const fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
        }
    }

    pub const fn input_size(self) -> usize {
        self.input_size
    }

    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub const fn output_size(self) -> usize {
        self.output_size
    }

    fn validate(self) -> Result<(), MediaTekQuantizedMlpTensorViewError> {
        if self.input_size == 0 {
            return Err(MediaTekQuantizedMlpTensorViewError::ZeroDim {
                dimension: "input_size",
            });
        }
        if self.hidden_size == 0 {
            return Err(MediaTekQuantizedMlpTensorViewError::ZeroDim {
                dimension: "hidden_size",
            });
        }
        if self.output_size == 0 {
            return Err(MediaTekQuantizedMlpTensorViewError::ZeroDim {
                dimension: "output_size",
            });
        }
        Ok(())
    }

    fn w1_len(self) -> Result<usize, MediaTekQuantizedMlpTensorViewError> {
        self.hidden_size
            .checked_mul(self.input_size)
            .ok_or(MediaTekQuantizedMlpTensorViewError::LengthOverflow { tensor: "w1" })
    }

    fn w2_len(self) -> Result<usize, MediaTekQuantizedMlpTensorViewError> {
        self.output_size
            .checked_mul(self.hidden_size)
            .ok_or(MediaTekQuantizedMlpTensorViewError::LengthOverflow { tensor: "w2" })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MediaTekQuantizedMlpTensorView<'a> {
    shape: MediaTekQuantizedMlpShape,
    input_params: MediaTekQuantParams,
    w1_params: MediaTekQuantParams,
    hidden_params: MediaTekQuantParams,
    w2_params: MediaTekQuantParams,
    output_params: MediaTekQuantParams,
    w1: &'a [u8],
    b1: &'a [i32],
    w2: &'a [u8],
    b2: &'a [i32],
    input: &'a [u8],
}

impl<'a> MediaTekQuantizedMlpTensorView<'a> {
    pub fn new(
        shape: MediaTekQuantizedMlpShape,
        input_params: MediaTekQuantParams,
        w1_params: MediaTekQuantParams,
        hidden_params: MediaTekQuantParams,
        w2_params: MediaTekQuantParams,
        output_params: MediaTekQuantParams,
        w1: &'a [u8],
        b1: &'a [i32],
        w2: &'a [u8],
        b2: &'a [i32],
        input: &'a [u8],
    ) -> Result<Self, MediaTekQuantizedMlpTensorViewError> {
        shape.validate()?;
        input_params.validate("input")?;
        w1_params.validate("w1")?;
        hidden_params.validate("hidden")?;
        w2_params.validate("w2")?;
        output_params.validate("output")?;

        expect_len("w1", shape.w1_len()?, w1.len())?;
        expect_len("b1", shape.hidden_size, b1.len())?;
        expect_len("w2", shape.w2_len()?, w2.len())?;
        expect_len("b2", shape.output_size, b2.len())?;
        expect_len("input", shape.input_size, input.len())?;

        Ok(Self {
            shape,
            input_params,
            w1_params,
            hidden_params,
            w2_params,
            output_params,
            w1,
            b1,
            w2,
            b2,
            input,
        })
    }

    pub const fn shape(&self) -> MediaTekQuantizedMlpShape {
        self.shape
    }

    pub const fn input_params(&self) -> MediaTekQuantParams {
        self.input_params
    }

    pub const fn w1_params(&self) -> MediaTekQuantParams {
        self.w1_params
    }

    pub const fn hidden_params(&self) -> MediaTekQuantParams {
        self.hidden_params
    }

    pub const fn w2_params(&self) -> MediaTekQuantParams {
        self.w2_params
    }

    pub const fn output_params(&self) -> MediaTekQuantParams {
        self.output_params
    }

    pub const fn w1(&self) -> &'a [u8] {
        self.w1
    }

    pub const fn b1(&self) -> &'a [i32] {
        self.b1
    }

    pub const fn w2(&self) -> &'a [u8] {
        self.w2
    }

    pub const fn b2(&self) -> &'a [i32] {
        self.b2
    }

    pub const fn input(&self) -> &'a [u8] {
        self.input
    }

    pub const fn output_len(&self) -> usize {
        self.shape.output_size
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaTekQuantizedMlpTensorViewError {
    ZeroDim {
        dimension: &'static str,
    },
    LengthOverflow {
        tensor: &'static str,
    },
    LengthMismatch {
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidQuantParam {
        tensor: &'static str,
    },
}

impl fmt::Display for MediaTekQuantizedMlpTensorViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDim { dimension } => {
                write!(f, "MediaTek quantized MLP {dimension} must be non-zero")
            }
            Self::LengthOverflow { tensor } => {
                write!(f, "MediaTek quantized MLP {tensor} length overflow")
            }
            Self::LengthMismatch {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "MediaTek quantized MLP {tensor} length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidQuantParam { tensor } => write!(
                f,
                "MediaTek quantized MLP {tensor} quant params must have finite positive scale and u8 zero point"
            ),
        }
    }
}

impl std::error::Error for MediaTekQuantizedMlpTensorViewError {}

fn expect_len(
    tensor: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), MediaTekQuantizedMlpTensorViewError> {
    if expected != actual {
        return Err(MediaTekQuantizedMlpTensorViewError::LengthMismatch {
            tensor,
            expected,
            actual,
        });
    }
    Ok(())
}

impl Backend for MediaTekBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::MediaTekNpu
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::new(BackendKind::MediaTekNpu)
    }

    fn execute(&mut self, request: BackendRequest) -> BackendResult<BackendOutput> {
        Err(BackendError::unsupported(self.kind(), request.op()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_backend_api::{BackendOp, MatMulRequest, QuantFormat, ScalarType, TensorShape};

    #[test]
    fn mediatek_backend_declares_identity_without_fake_matmul_capability() {
        let backend = MediaTekBackend::new();
        let capabilities = backend.capabilities();

        assert_eq!(backend.kind(), BackendKind::MediaTekNpu);
        assert_eq!(capabilities.backend(), BackendKind::MediaTekNpu);
        assert!(!capabilities.supports(BackendOp::MatMul));
    }

    #[test]
    fn mediatek_backend_rejects_matmul_until_real_runtime_implementation_exists() {
        let mut backend = MediaTekBackend::new();
        let request = BackendRequest::matmul(MatMulRequest::new(
            TensorShape::new(16, 32),
            TensorShape::new(1, 32),
            QuantFormat::Q4K,
            ScalarType::F32,
        ));

        assert!(matches!(
            backend.execute(request),
            Err(err)
                if err.kind() == rnb_backend_api::BackendErrorKind::UnsupportedOp
                    && err.backend() == BackendKind::MediaTekNpu
                    && err.op() == Some(BackendOp::MatMul)
        ));
    }

    #[test]
    fn quantized_mlp_tensor_view_accepts_full_gemma_e2b_ffn_in_memory_contract() {
        let shape = MediaTekQuantizedMlpShape::new(1536, 6144, 1536);
        let w1 = vec![128; 1536 * 6144];
        let b1 = vec![0; 6144];
        let w2 = vec![128; 1536 * 6144];
        let b2 = vec![0; 1536];
        let input = vec![128; 1536];
        let tensors = MediaTekQuantizedMlpTensorView::new(
            shape,
            MediaTekQuantParams::new(0.001643489, 128),
            MediaTekQuantParams::new(0.003738719, 128),
            MediaTekQuantParams::new(0.001782878, 0),
            MediaTekQuantParams::new(0.004342868, 128),
            MediaTekQuantParams::new(0.010613354, 128),
            &w1,
            &b1,
            &w2,
            &b2,
            &input,
        )
        .expect("Gemma E2B full FFN quantized MLP in-memory request");

        assert_eq!(tensors.shape(), shape);
        assert_eq!(tensors.w1().len(), 1536 * 6144);
        assert_eq!(tensors.w2().len(), 1536 * 6144);
        assert_eq!(tensors.input().len(), 1536);
        assert_eq!(tensors.output_len(), 1536);
    }

    #[test]
    fn quantized_mlp_tensor_view_rejects_mismatched_in_memory_lengths() {
        let shape = MediaTekQuantizedMlpShape::new(4, 3, 2);
        let err = MediaTekQuantizedMlpTensorView::new(
            shape,
            MediaTekQuantParams::new(0.25, 128),
            MediaTekQuantParams::new(0.5, 128),
            MediaTekQuantParams::new(0.125, 0),
            MediaTekQuantParams::new(0.75, 128),
            MediaTekQuantParams::new(1.0, 128),
            &[128; 11],
            &[0; 3],
            &[128; 6],
            &[0; 2],
            &[128; 4],
        )
        .expect_err("w1 length mismatch must be rejected before runtime invoke");

        assert!(matches!(
            err,
            MediaTekQuantizedMlpTensorViewError::LengthMismatch {
                tensor: "w1",
                expected: 12,
                actual: 11,
            }
        ));
    }

    #[test]
    fn quantized_mlp_tensor_view_rejects_invalid_quant_params() {
        let err = MediaTekQuantParams::new(f32::NAN, 128)
            .validate("input")
            .expect_err("non-finite quant scale must be rejected");

        assert!(matches!(
            err,
            MediaTekQuantizedMlpTensorViewError::InvalidQuantParam { tensor: "input" }
        ));
    }

    #[test]
    fn gated_gelu_ffn_tensor_view_accepts_valid_f32_contract() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let gate = vec![0.1; 12];
        let up = vec![0.2; 12];
        let down = vec![0.3; 6];
        let input = vec![0.4; 4];

        let tensors = MediaTekGatedGeluFfnTensorView::new(shape, &gate, &up, &down, &input)
            .expect("valid in-memory FLOAT32 gated GELU FFN request");

        assert_eq!(tensors.shape(), shape);
        assert_eq!(tensors.gate_weight().len(), 12);
        assert_eq!(tensors.up_weight().len(), 12);
        assert_eq!(tensors.down_weight().len(), 6);
        assert_eq!(tensors.input().len(), 4);
        assert_eq!(tensors.output_len(), 2);
    }

    #[test]
    fn gated_gelu_ffn_tensor_view_rejects_zero_dims_length_mismatch_and_nonfinite() {
        let zero = MediaTekGatedGeluFfnTensorView::new(
            MediaTekGatedGeluFfnShape::new(0, 3, 2),
            &[0.1; 12],
            &[0.2; 12],
            &[0.3; 6],
            &[0.4; 4],
        )
        .expect_err("zero input dim must be rejected");
        assert!(matches!(
            zero,
            MediaTekGatedGeluFfnTensorViewError::ZeroDim {
                dimension: "input_size"
            }
        ));

        let mismatch = MediaTekGatedGeluFfnTensorView::new(
            MediaTekGatedGeluFfnShape::new(4, 3, 2),
            &[0.1; 11],
            &[0.2; 12],
            &[0.3; 6],
            &[0.4; 4],
        )
        .expect_err("gate length mismatch must be rejected");
        assert!(matches!(
            mismatch,
            MediaTekGatedGeluFfnTensorViewError::LengthMismatch {
                tensor: "gate_weight",
                expected: 12,
                actual: 11,
            }
        ));

        let gate = [f32::NAN; 12];
        let nonfinite = MediaTekGatedGeluFfnTensorView::new(
            MediaTekGatedGeluFfnShape::new(4, 3, 2),
            &gate,
            &[0.2; 12],
            &[0.3; 6],
            &[0.4; 4],
        )
        .expect_err("non-finite tensor data must be rejected");
        assert!(matches!(
            nonfinite,
            MediaTekGatedGeluFfnTensorViewError::NonFinite {
                tensor: "gate_weight",
                index: 0,
            }
        ));
    }

    #[test]
    fn gated_gelu_supported_ops_lists_every_probe_operation() {
        let ops = MediaTekGatedGeluFfnSupportedOps::all(true);
        assert!(ops.all_supported());
        assert_eq!(
            ops.named(),
            &[
                ("gate_fc", true),
                ("up_fc", true),
                ("gate_dequant", true),
                ("up_dequant", true),
                ("gelu_square_mul", true),
                ("gelu_cube_mul", true),
                ("gelu_poly_scale_mul", true),
                ("gelu_poly_add", true),
                ("gelu_tanh_scale_mul", true),
                ("gelu_tanh", true),
                ("gelu_one_plus_add", true),
                ("gelu_gate_one_plus_mul", true),
                ("gelu_half_mul", true),
                ("gated_mul", true),
                ("gated_quantize", true),
                ("down_fc", true),
            ]
        );
    }

    #[test]
    fn quantized_gated_gelu_support_probe_exposes_cross_crate_api() {
        let shape = MediaTekGatedGeluFfnShape::new(1536, 6144, 1536);
        let probe = MediaTekQuantizedGatedGeluFfnSupportProbe::new(shape)
            .with_device_name_substring("mtk-neuron");

        assert_eq!(probe.shape(), shape);
        assert_eq!(probe.device_name_substring(), "mtk-neuron");

        let device = MediaTekNnapiDeviceInfo::new(
            "mtk-neuron_shim",
            super::MEDIATEK_NNAPI_DEVICE_ACCELERATOR,
            1000008,
            "7.2.4",
        );
        let result = MediaTekQuantizedGatedGeluFfnSupportResult::new(
            device.clone(),
            MediaTekGatedGeluFfnSupportedOps::all(true),
            11,
            22,
        );
        assert!(result.supported());
        assert_eq!(result.chosen_device(), &device);
        assert_eq!(
            result.supported_ops(),
            MediaTekGatedGeluFfnSupportedOps::all(true)
        );
        assert_eq!(result.model_build_ns(), 11);
        assert_eq!(result.supported_ops_query_ns(), 22);

        let mut backend = MediaTekBackend::new();
        let support = backend.probe_quantized_gated_gelu_ffn_support(&probe);
        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            support,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn quantized_gated_gelu_support_probe_validates_shape_before_platform_gate() {
        let probe = MediaTekQuantizedGatedGeluFfnSupportProbe::new(MediaTekGatedGeluFfnShape::new(
            u32::MAX as usize + 1,
            1,
            1,
        ));
        let mut backend = MediaTekBackend::new();

        assert!(matches!(
            backend.probe_quantized_gated_gelu_ffn_support(&probe),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
    }

    #[test]
    fn gated_gelu_f32_shared_blob_layout_offsets_are_page_rounded() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let layout = gated_gelu_f32_blob_layout(shape, 16).expect("valid f32 blob layout");
        let regions = layout.regions();

        assert_eq!(
            regions,
            &[
                GatedGeluFfnF32BlobRegion {
                    name: "gate_weight",
                    offset: 0,
                    length: 48,
                },
                GatedGeluFfnF32BlobRegion {
                    name: "gate_bias",
                    offset: 48,
                    length: 12,
                },
                GatedGeluFfnF32BlobRegion {
                    name: "up_weight",
                    offset: 60,
                    length: 48,
                },
                GatedGeluFfnF32BlobRegion {
                    name: "up_bias",
                    offset: 108,
                    length: 12,
                },
                GatedGeluFfnF32BlobRegion {
                    name: "down_weight",
                    offset: 120,
                    length: 24,
                },
                GatedGeluFfnF32BlobRegion {
                    name: "down_bias",
                    offset: 144,
                    length: 8,
                },
            ]
        );
        assert_eq!(layout.total_len, 160);
    }

    #[test]
    fn gated_gelu_f32_cache_token_is_variant_weight_and_device_sensitive() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let gate = vec![0.1f32; 12];
        let up = vec![0.2f32; 12];
        let down = vec![0.3f32; 6];

        let base = gated_gelu_f32_cache_token(0, "mtk-neuron", shape, 2, &gate, &up, &down);
        let same = gated_gelu_f32_cache_token(0, "mtk-neuron", shape, 2, &gate, &up, &down);
        assert_eq!(base, same);

        let variant = gated_gelu_f32_cache_token(1, "mtk-neuron", shape, 2, &gate, &up, &down);
        assert_ne!(base, variant);

        let mut changed_gate = gate.clone();
        let first = changed_gate[0].to_bits() ^ 0x1;
        changed_gate[0] = f32::from_bits(first);
        let changed_weight =
            gated_gelu_f32_cache_token(0, "mtk-neuron", shape, 2, &changed_gate, &up, &down);
        assert_ne!(base, changed_weight);

        let changed_device =
            gated_gelu_f32_cache_token(0, "other-neuron", shape, 2, &gate, &up, &down);
        assert_ne!(base, changed_device);

        let changed_batch =
            gated_gelu_f32_cache_token(0, "mtk-neuron", shape, 4, &gate, &up, &down);
        assert_ne!(base, changed_batch);

        let changed_shape = gated_gelu_f32_cache_token(
            0,
            "mtk-neuron",
            MediaTekGatedGeluFfnShape::new(4, 3, 1),
            2,
            &gate,
            &up,
            &[0.3f32; 3],
        );
        assert_ne!(base, changed_shape);
    }

    #[test]
    fn quantized_gated_gelu_shared_blob_offsets_are_element_aligned() {
        let shape = MediaTekGatedGeluFfnShape::new(1536, 6144, 1536);
        let layout = quantized_gated_gelu_support_blob_layout(shape, 4096)
            .expect("valid support blob layout");

        assert_eq!(layout.total_len % 4096, 0);
        for region in layout.regions() {
            assert_eq!(
                region.offset % region.element_size,
                0,
                "{} offset must be element-aligned",
                region.name
            );
            assert_eq!(
                region.length % region.element_size,
                0,
                "{} length must be element-aligned",
                region.name
            );
        }
    }

    #[test]
    fn quantized_gated_gelu_shared_blob_page_rounding_rejects_overflow() {
        assert!(matches!(
            round_up_quantized_gated_gelu_blob_size(usize::MAX - 1, 4096),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
        assert!(matches!(
            round_up_quantized_gated_gelu_blob_size(16, 0),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
    }

    #[test]
    fn quantized_gated_gelu_support_shape_validation_rejects_zero_dims_u32_truncation_and_overflow()
    {
        assert!(matches!(
            validate_quantized_gated_gelu_support_shape(MediaTekGatedGeluFfnShape::new(0, 1, 1)),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
        assert!(matches!(
            validate_quantized_gated_gelu_support_shape(MediaTekGatedGeluFfnShape::new(1, 0, 1)),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
        assert!(matches!(
            validate_quantized_gated_gelu_support_shape(MediaTekGatedGeluFfnShape::new(1, 1, 0)),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
        assert!(matches!(
            validate_quantized_gated_gelu_support_shape(MediaTekGatedGeluFfnShape::new(
                u32::MAX as usize + 1,
                1,
                1
            )),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
        assert!(matches!(
            validate_quantized_gated_gelu_support_shape(MediaTekGatedGeluFfnShape::new(
                u32::MAX as usize,
                u32::MAX as usize,
                u32::MAX as usize
            )),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
    }

    #[test]
    fn quantized_gated_gelu_probe_quant_params_satisfy_nnapi_constraints() {
        let params = quantized_gated_gelu_probe_quant_params();

        assert_eq!(params.input, MediaTekQuantParams::new(0.01, 128));
        assert_eq!(params.gate_weight, MediaTekQuantParams::new(0.01, 128));
        assert_eq!(params.gate_bias, MediaTekQuantParams::new(0.0001, 0));
        assert_eq!(params.up_bias, MediaTekQuantParams::new(0.0001, 0));
        assert_eq!(params.down_bias, MediaTekQuantParams::new(0.0001, 0));
        assert_eq!(
            params.tanh_output,
            MediaTekQuantParams::new(1.0 / 128.0, 128)
        );
        assert!(params.square.scale() > params.input.scale() * params.input.scale());
    }

    #[test]
    fn quantized_gated_gelu_quant_params_reject_invalid_fc_bias_contract() {
        let mut bad_scale = quantized_gated_gelu_probe_quant_params();
        bad_scale.gate_bias = MediaTekQuantParams::new(0.0002, 0);
        assert!(matches!(
            validate_quantized_gated_gelu_quant_params(bad_scale),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));

        let mut bad_zero_point = quantized_gated_gelu_probe_quant_params();
        bad_zero_point.up_bias = MediaTekQuantParams::new(0.0001, 1);
        assert!(matches!(
            validate_quantized_gated_gelu_quant_params(bad_zero_point),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
    }

    #[test]
    fn quantized_gated_gelu_owned_weights_accept_valid_u8_contract() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let gate = vec![128; 12];
        let up = vec![129; 12];
        let down = vec![127; 6];
        let gate_ptr = gate.as_ptr();
        let up_ptr = up.as_ptr();
        let down_ptr = down.as_ptr();

        let owned = MediaTekQuantizedGatedGeluFfnOwnedWeights::new(
            shape,
            quantized_gated_gelu_probe_quant_params(),
            gate,
            up,
            down,
        )
        .expect("valid owned quantized gated GELU weights");

        assert_eq!(owned.shape(), shape);
        assert_eq!(
            owned.quant_params(),
            quantized_gated_gelu_probe_quant_params()
        );
        assert_eq!(owned.gate_weight().as_ptr(), gate_ptr);
        assert_eq!(owned.up_weight().as_ptr(), up_ptr);
        assert_eq!(owned.down_weight().as_ptr(), down_ptr);
        assert_eq!(owned.output_len(), 2);
    }

    #[test]
    fn quantized_gated_gelu_owned_weights_reject_shape_quant_and_length_mismatch() {
        let zero_shape = MediaTekQuantizedGatedGeluFfnOwnedWeights::new(
            MediaTekGatedGeluFfnShape::new(0, 3, 2),
            quantized_gated_gelu_probe_quant_params(),
            vec![128; 12],
            vec![128; 12],
            vec![128; 6],
        )
        .expect_err("zero input dim must be rejected");
        assert!(matches!(
            zero_shape,
            MediaTekNnapiError::InvalidInput { .. }
        ));

        let mut invalid_params = quantized_gated_gelu_probe_quant_params();
        invalid_params.input = MediaTekQuantParams::new(f32::NAN, 128);
        let invalid_quant = MediaTekQuantizedGatedGeluFfnOwnedWeights::new(
            MediaTekGatedGeluFfnShape::new(4, 3, 2),
            invalid_params,
            vec![128; 12],
            vec![128; 12],
            vec![128; 6],
        )
        .expect_err("invalid quant params must be rejected");
        assert!(matches!(
            invalid_quant,
            MediaTekNnapiError::InvalidInput { .. }
        ));

        let mismatch = MediaTekQuantizedGatedGeluFfnOwnedWeights::new(
            MediaTekGatedGeluFfnShape::new(4, 3, 2),
            quantized_gated_gelu_probe_quant_params(),
            vec![128; 11],
            vec![128; 12],
            vec![128; 6],
        )
        .expect_err("gate length mismatch must be rejected");
        assert!(matches!(
            mismatch,
            MediaTekNnapiError::InvalidInput { reason }
                if reason == "gate_weight length mismatch: expected 12, got 11"
        ));
    }

    #[test]
    fn quantized_gated_gelu_stage_probe_stage_names_and_lengths() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        assert_eq!(MediaTekQuantizedGatedGeluFfnStage::GateFc.name(), "gate_fc");
        assert_eq!(MediaTekQuantizedGatedGeluFfnStage::UpFc.name(), "up_fc");
        assert_eq!(
            MediaTekQuantizedGatedGeluFfnStage::GateDequant.name(),
            "gate_dequant"
        );
        assert_eq!(
            MediaTekQuantizedGatedGeluFfnStage::UpDequant.name(),
            "up_dequant"
        );
        assert_eq!(MediaTekQuantizedGatedGeluFfnStage::Gelu.name(), "gelu");
        assert_eq!(MediaTekQuantizedGatedGeluFfnStage::Gated.name(), "gated");
        assert_eq!(MediaTekQuantizedGatedGeluFfnStage::Output.name(), "output");

        for stage in [
            MediaTekQuantizedGatedGeluFfnStage::GateFc,
            MediaTekQuantizedGatedGeluFfnStage::UpFc,
            MediaTekQuantizedGatedGeluFfnStage::GateDequant,
            MediaTekQuantizedGatedGeluFfnStage::UpDequant,
            MediaTekQuantizedGatedGeluFfnStage::Gelu,
            MediaTekQuantizedGatedGeluFfnStage::Gated,
        ] {
            assert_eq!(stage.output_len(shape), 3);
        }
        assert_eq!(
            MediaTekQuantizedGatedGeluFfnStage::Output.output_len(shape),
            2
        );
    }

    #[test]
    fn quantized_gated_gelu_stage_probe_expected_length_rejects_mismatch() {
        assert!(matches!(
            validate_quantized_gated_gelu_stage_output_len(
                MediaTekQuantizedGatedGeluFfnStage::GateFc,
                MediaTekGatedGeluFfnShape::new(4, 3, 2),
                2,
            ),
            Err(MediaTekNnapiError::InvalidOutputLength {
                expected: 3,
                actual: 2,
            })
        ));
    }

    #[test]
    fn quantized_gated_gelu_stage_probe_rejects_missing_hybrid_down() {
        let owned = MediaTekQuantizedGatedGeluFfnOwnedWeights::new(
            MediaTekGatedGeluFfnShape::new(4, 3, 2),
            quantized_gated_gelu_probe_quant_params(),
            vec![128; 12],
            vec![128; 12],
            vec![128; 6],
        )
        .expect("valid non-hybrid owned weights");
        let mut backend = MediaTekBackend::new();

        let result = backend.run_quantized_gated_gelu_ffn_stage_probe(
            owned,
            &[128; 4],
            MediaTekQuantizedGatedGeluFfnStage::GateFc,
            MediaTekNnapiOptions::new("mtk-neuron"),
        );
        assert!(matches!(
            result,
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
    }

    #[test]
    fn mediatek_nnapi_run_quantized_gated_gelu_stage_probe_is_android_only_on_host() {
        let owned = MediaTekQuantizedGatedGeluFfnOwnedWeights::new_with_f32_down(
            MediaTekGatedGeluFfnShape::new(4, 3, 2),
            quantized_gated_gelu_probe_quant_params(),
            vec![128; 12],
            vec![128; 12],
            vec![128; 6],
            vec![0.0; 6],
        )
        .expect("valid hybrid owned weights");
        let mut backend = MediaTekBackend::new();

        let result = backend.run_quantized_gated_gelu_ffn_stage_probe(
            owned,
            &[128; 4],
            MediaTekQuantizedGatedGeluFfnStage::Output,
            MediaTekNnapiOptions::new("mtk-neuron"),
        );

        #[cfg(target_os = "android")]
        assert!(
            !matches!(result, Err(MediaTekNnapiError::UnsupportedPlatform)),
            "Android target should attempt quantized stage probing"
        );
        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            result,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn quantized_gated_gelu_compile_validates_before_host_platform_gate() {
        let invalid = MediaTekQuantizedGatedGeluFfnOwnedWeights {
            shape: MediaTekGatedGeluFfnShape::new(4, 3, 2),
            quant_params: quantized_gated_gelu_probe_quant_params(),
            gate_weight: vec![128; 11],
            up_weight: vec![128; 12],
            down_weight: vec![128; 6],
            down_weight_f32: None,
        };
        let mut backend = MediaTekBackend::new();

        assert!(matches!(
            backend
                .compile_quantized_gated_gelu_ffn(invalid, MediaTekNnapiOptions::new("mtk-neuron")),
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));
    }

    #[test]
    fn mediatek_nnapi_compile_quantized_gated_gelu_ffn_is_android_only_on_host() {
        let owned = MediaTekQuantizedGatedGeluFfnOwnedWeights::new(
            MediaTekGatedGeluFfnShape::new(4, 3, 2),
            quantized_gated_gelu_probe_quant_params(),
            vec![128; 12],
            vec![128; 12],
            vec![128; 6],
        )
        .expect("valid owned quantized weights");
        let mut backend = MediaTekBackend::new();

        let compiled = backend
            .compile_quantized_gated_gelu_ffn(owned, MediaTekNnapiOptions::new("mtk-neuron"));

        #[cfg(target_os = "android")]
        assert!(
            !matches!(compiled, Err(MediaTekNnapiError::UnsupportedPlatform)),
            "Android target should attempt quantized NNAPI compilation"
        );
        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            compiled,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn gated_gelu_output_preserves_timing_breakdown() {
        let timings = MediaTekGatedGeluFfnTimings::new(11, 22, 33, 44, 55);
        let output = MediaTekGatedGeluFfnOutput::new(
            vec![0.5, -0.25],
            MediaTekNnapiDeviceInfo::new(
                "mtk-neuron_shim",
                MEDIATEK_NNAPI_DEVICE_ACCELERATOR,
                1000008,
                "7.2.4",
            ),
            MediaTekGatedGeluFfnSupportedOps::all(true),
            Some(66),
            Some(77),
            timings,
        );

        assert_eq!(output.timings(), timings);
        assert_eq!(output.timings().model_build_ns(), 11);
        assert_eq!(output.timings().supported_ops_query_ns(), 22);
        assert_eq!(output.timings().compilation_ns(), 33);
        assert_eq!(output.timings().execution_setup_ns(), 44);
        assert_eq!(output.timings().execution_compute_ns(), 55);
    }

    #[test]
    fn nnapi_execution_options_carry_measure_timing_policy() {
        assert!(MediaTekNnapiExecutionOptions::new(true).measure_timing());
        assert!(!MediaTekNnapiExecutionOptions::new(false).measure_timing());
    }

    #[test]
    fn gated_gelu_owned_weights_validate_and_retain_moved_buffers() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let gate = vec![0.1; 12];
        let up = vec![0.2; 12];
        let down = vec![0.3; 6];
        let gate_ptr = gate.as_ptr();
        let up_ptr = up.as_ptr();
        let down_ptr = down.as_ptr();

        let owned = MediaTekGatedGeluFfnOwnedWeights::new(shape, gate, up, down)
            .expect("valid owned f32 weights");

        assert_eq!(owned.shape(), shape);
        assert_eq!(owned.gate_weight().as_ptr(), gate_ptr);
        assert_eq!(owned.up_weight().as_ptr(), up_ptr);
        assert_eq!(owned.down_weight().as_ptr(), down_ptr);
        assert_eq!(owned.output_len(), 2);
    }

    #[test]
    fn gated_gelu_owned_weights_reject_mismatch_and_nonfinite() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let mismatch = MediaTekGatedGeluFfnOwnedWeights::new(
            shape,
            vec![0.1; 11],
            vec![0.2; 12],
            vec![0.3; 6],
        )
        .expect_err("gate length mismatch must be rejected");
        assert!(matches!(
            mismatch,
            MediaTekGatedGeluFfnTensorViewError::LengthMismatch {
                tensor: "gate_weight",
                expected: 12,
                actual: 11,
            }
        ));

        let nonfinite = MediaTekGatedGeluFfnOwnedWeights::new(
            shape,
            vec![f32::NAN; 12],
            vec![0.2; 12],
            vec![0.3; 6],
        )
        .expect_err("non-finite gate weight must be rejected");
        assert!(matches!(
            nonfinite,
            MediaTekGatedGeluFfnTensorViewError::NonFinite {
                tensor: "gate_weight",
                index: 0,
            }
        ));
    }

    #[test]
    fn mediatek_nnapi_probe_gated_gelu_ffn_f32_batched_validates_host_contract() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let gate = vec![0.1; 12];
        let up = vec![0.2; 12];
        let down = vec![0.3; 6];
        let input = vec![0.4; 8];
        let tensors = MediaTekGatedGeluFfnBatchedTensorView::new(shape, &gate, &up, &down, &input)
            .expect("valid batched tensors");
        assert_eq!(
            validate_gated_gelu_f32_batched_input(shape, 2, tensors.input())
                .expect("valid batched length"),
            (8, 4)
        );

        let mut backend = MediaTekBackend::new();
        let result = backend.probe_gated_gelu_ffn_f32_batched(
            &tensors,
            2,
            MediaTekNnapiOptions::new("mtk-neuron"),
        );

        #[cfg(target_os = "android")]
        assert!(
            !matches!(result, Err(MediaTekNnapiError::UnsupportedPlatform)),
            "Android target should attempt batched NNAPI probing"
        );
        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            result,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));

        let zero_batch = backend.probe_gated_gelu_ffn_f32_batched(
            &tensors,
            0,
            MediaTekNnapiOptions::new("mtk-neuron"),
        );
        assert!(matches!(
            zero_batch,
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));

        let short_input = vec![0.4; 7];
        let short_tensors =
            MediaTekGatedGeluFfnBatchedTensorView::new(shape, &gate, &up, &down, &short_input)
                .expect("shape-only batched tensors");
        let short = backend.probe_gated_gelu_ffn_f32_batched(
            &short_tensors,
            2,
            MediaTekNnapiOptions::new("mtk-neuron"),
        );
        assert!(matches!(
            short,
            Err(MediaTekNnapiError::InvalidInput { reason })
                if reason == "input length mismatch: expected 8, got 7"
        ));
    }

    #[test]
    fn mediatek_nnapi_compile_gated_gelu_ffn_f32_batched_with_validates_host_contract() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let owned = MediaTekGatedGeluFfnOwnedWeights::new(
            shape,
            vec![0.1; 12],
            vec![0.2; 12],
            vec![0.3; 6],
        )
        .expect("valid owned weights");
        let mut backend = MediaTekBackend::new();

        let compiled = backend.compile_gated_gelu_ffn_f32_batched_with(
            owned,
            2,
            MediaTekNnapiOptions::new("mtk-neuron"),
            BatchedCompileOptions {
                zero_copy: true,
                cache_dir: Some("/tmp/rnb-mk28-cache".to_string()),
            },
        );

        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            compiled,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn mediatek_nnapi_compile_gated_gelu_ffn_f32_batched_with_rejects_invalid_inputs() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let owned = MediaTekGatedGeluFfnOwnedWeights::new(
            shape,
            vec![0.1; 12],
            vec![0.2; 12],
            vec![0.3; 6],
        )
        .expect("valid owned weights");
        let mut backend = MediaTekBackend::new();

        let zero_batch = backend.compile_gated_gelu_ffn_f32_batched_with(
            owned,
            0,
            MediaTekNnapiOptions::new("mtk-neuron"),
            BatchedCompileOptions::default(),
        );
        assert!(matches!(
            zero_batch,
            Err(MediaTekNnapiError::InvalidInput { .. })
        ));

        let mismatched = MediaTekGatedGeluFfnOwnedWeights::new(
            shape,
            vec![0.1; 11],
            vec![0.2; 12],
            vec![0.3; 6],
        );
        assert!(matches!(
            mismatched,
            Err(MediaTekGatedGeluFfnTensorViewError::LengthMismatch {
                tensor: "gate_weight",
                expected: 12,
                actual: 11,
            })
        ));
    }

    #[test]
    fn mediatek_nnapi_compile_and_run_compiled_gated_gelu_ffn_f32_batched_validate_host_contract() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let owned = MediaTekGatedGeluFfnOwnedWeights::new(
            shape,
            vec![0.1; 12],
            vec![0.2; 12],
            vec![0.3; 6],
        )
        .expect("valid owned weights");
        let mut backend = MediaTekBackend::new();

        let compiled = backend.compile_gated_gelu_ffn_f32_batched(
            owned,
            2,
            MediaTekNnapiOptions::new("mtk-neuron"),
        );

        #[cfg(target_os = "android")]
        {
            let compiled = compiled.expect("Android target should compile batched NNAPI graph");
            let input = vec![0.4; 8];
            let result = backend.run_compiled_gated_gelu_ffn_f32_batched(
                &compiled,
                &input,
                2,
                MediaTekNnapiExecutionOptions::new(true),
            );
            assert!(
                !matches!(result, Err(MediaTekNnapiError::UnsupportedPlatform)),
                "Android target should attempt batched compiled NNAPI execution"
            );
        }
        #[cfg(not(target_os = "android"))]
        {
            assert!(matches!(
                compiled,
                Err(MediaTekNnapiError::UnsupportedPlatform)
            ));
            let host_compilation = MediaTekGatedGeluFfnBatchedCompilation { shape, batch: 2 };
            let mismatched_batch = backend.run_compiled_gated_gelu_ffn_f32_batched(
                &host_compilation,
                &[0.4; 12],
                3,
                MediaTekNnapiExecutionOptions::new(true),
            );
            assert!(matches!(
                mismatched_batch,
                Err(MediaTekNnapiError::InvalidInput { reason })
                    if reason == "batch mismatch: compiled for 2, got 3"
            ));

            let result = backend.run_compiled_gated_gelu_ffn_f32_batched(
                &host_compilation,
                &[0.4; 8],
                2,
                MediaTekNnapiExecutionOptions::new(true),
            );
            assert!(matches!(
                result,
                Err(MediaTekNnapiError::UnsupportedPlatform)
            ));
        }
    }

    #[test]
    fn mediatek_nnapi_probe_gated_gelu_ffn_f32_is_android_only_on_host() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let gate = vec![0.1; 12];
        let up = vec![0.2; 12];
        let down = vec![0.3; 6];
        let input = vec![0.4; 4];
        let tensors = MediaTekGatedGeluFfnTensorView::new(shape, &gate, &up, &down, &input)
            .expect("valid in-memory tensors");
        let mut backend = MediaTekBackend::new();

        let result =
            backend.probe_gated_gelu_ffn_f32(&tensors, MediaTekNnapiOptions::new("mtk-neuron"));

        #[cfg(target_os = "android")]
        assert!(
            !matches!(result, Err(MediaTekNnapiError::UnsupportedPlatform)),
            "Android target should attempt NNAPI instead of host platform rejection"
        );
        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            result,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn mediatek_nnapi_run_gated_gelu_ffn_f32_is_android_only_on_host() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let gate = vec![0.1; 12];
        let up = vec![0.2; 12];
        let down = vec![0.3; 6];
        let input = vec![0.4; 4];
        let tensors = MediaTekGatedGeluFfnTensorView::new(shape, &gate, &up, &down, &input)
            .expect("valid in-memory tensors");
        let mut backend = MediaTekBackend::new();

        let result =
            backend.run_gated_gelu_ffn_f32(&tensors, MediaTekNnapiOptions::new("mtk-neuron"));

        #[cfg(target_os = "android")]
        assert!(
            !matches!(result, Err(MediaTekNnapiError::UnsupportedPlatform)),
            "Android target should attempt NNAPI instead of host platform rejection"
        );
        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            result,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn mediatek_nnapi_run_compiled_gated_gelu_ffn_f32_is_android_only_on_host() {
        let shape = MediaTekGatedGeluFfnShape::new(4, 3, 2);
        let owned = MediaTekGatedGeluFfnOwnedWeights::new(
            shape,
            vec![0.1; 12],
            vec![0.2; 12],
            vec![0.3; 6],
        )
        .expect("valid owned weights");
        let mut backend = MediaTekBackend::new();

        let compiled =
            backend.compile_gated_gelu_ffn_f32(owned, MediaTekNnapiOptions::new("mtk-neuron"));

        #[cfg(target_os = "android")]
        {
            let compiled = compiled.expect("Android target should compile NNAPI graph");
            let input = vec![0.4; 4];
            let result = backend.run_compiled_gated_gelu_ffn_f32(
                &compiled,
                &input,
                MediaTekNnapiExecutionOptions::new(true),
            );
            assert!(
                !matches!(result, Err(MediaTekNnapiError::UnsupportedPlatform)),
                "Android target should attempt compiled NNAPI execution"
            );
        }
        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            compiled,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn mediatek_nnapi_run_quantized_mlp_is_android_only_on_host() {
        let shape = MediaTekQuantizedMlpShape::new(4, 3, 2);
        let w1 = vec![128; 12];
        let b1 = vec![0; 3];
        let w2 = vec![128; 6];
        let b2 = vec![0; 2];
        let input = vec![128; 4];
        let tensors = MediaTekQuantizedMlpTensorView::new(
            shape,
            MediaTekQuantParams::new(0.25, 128),
            MediaTekQuantParams::new(0.5, 128),
            MediaTekQuantParams::new(0.125, 0),
            MediaTekQuantParams::new(0.75, 128),
            MediaTekQuantParams::new(1.0, 128),
            &w1,
            &b1,
            &w2,
            &b2,
            &input,
        )
        .expect("valid in-memory tensors");
        let mut backend = MediaTekBackend::new();

        let result = backend.run_quantized_mlp(&tensors, MediaTekNnapiOptions::new("mtk-neuron"));

        #[cfg(target_os = "android")]
        assert!(
            !matches!(result, Err(MediaTekNnapiError::UnsupportedPlatform)),
            "Android target should attempt NNAPI instead of host platform rejection"
        );
        #[cfg(not(target_os = "android"))]
        assert!(matches!(
            result,
            Err(MediaTekNnapiError::UnsupportedPlatform)
        ));
    }

    #[test]
    fn mediatek_nnapi_contract_does_not_advertise_generic_matmul() {
        let backend = MediaTekBackend::new();

        assert!(!backend.capabilities().supports(BackendOp::MatMul));
    }
}
