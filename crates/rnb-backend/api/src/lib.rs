#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Vulkan,
    OpenCl,
    Metal,
    MediaTekNpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendOp {
    MatMul,
    Attention,
    Gdn,
    MoE,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    backend: BackendKind,
    ops: Vec<BackendOp>,
}

impl BackendCapabilities {
    pub fn new(backend: BackendKind) -> Self {
        Self {
            backend,
            ops: Vec::new(),
        }
    }

    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    pub fn with_op(mut self, op: BackendOp) -> Self {
        if !self.ops.contains(&op) {
            self.ops.push(op);
        }
        self
    }

    pub fn supports(&self, op: BackendOp) -> bool {
        self.ops.contains(&op)
    }

    pub fn ops(&self) -> &[BackendOp] {
        &self.ops
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendErrorKind {
    UnsupportedOp,
    InvalidRequest,
    ExecutionFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError {
    kind: BackendErrorKind,
    backend: BackendKind,
    op: Option<BackendOp>,
    message: String,
}

impl BackendError {
    pub fn new(
        kind: BackendErrorKind,
        backend: BackendKind,
        op: Option<BackendOp>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            backend,
            op,
            message: message.into(),
        }
    }

    pub fn unsupported(backend: BackendKind, op: BackendOp) -> Self {
        Self::new(
            BackendErrorKind::UnsupportedOp,
            backend,
            Some(op),
            format!("{backend:?} does not support {op:?}"),
        )
    }

    pub fn kind(&self) -> BackendErrorKind {
        self.kind
    }

    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    pub fn op(&self) -> Option<BackendOp> {
        self.op
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

pub type BackendResult<T> = Result<T, BackendError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F32,
    F16,
    BF16,
    I8,
    U8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceTensorRole {
    Hidden,
    Residual,
    Normalized,
    MoeOutput,
    MambaOutput,
    RouterLogits,
    Scratch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceTensorId {
    backend: BackendKind,
    raw: u64,
}

impl DeviceTensorId {
    pub const fn new(backend: BackendKind, raw: u64) -> Self {
        Self { backend, raw }
    }

    pub const fn backend(self) -> BackendKind {
        self.backend
    }

    pub const fn raw(self) -> u64 {
        self.raw
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceTensorDesc {
    rows: usize,
    cols: usize,
    dtype: ScalarType,
    role: DeviceTensorRole,
}

impl DeviceTensorDesc {
    pub const fn new(rows: usize, cols: usize, dtype: ScalarType, role: DeviceTensorRole) -> Self {
        Self {
            rows,
            cols,
            dtype,
            role,
        }
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn cols(self) -> usize {
        self.cols
    }

    pub const fn dtype(self) -> ScalarType {
        self.dtype
    }

    pub const fn role(self) -> DeviceTensorRole {
        self.role
    }

    pub fn checked_len(self) -> Option<usize> {
        self.rows.checked_mul(self.cols)
    }

    pub fn len(self) -> usize {
        self.checked_len()
            .expect("device tensor element count overflow")
    }

    pub fn byte_len(self) -> Option<usize> {
        let scalar_bytes = match self.dtype {
            ScalarType::F32 => 4,
            ScalarType::F16 | ScalarType::BF16 => 2,
            ScalarType::I8 | ScalarType::U8 => 1,
        };
        self.checked_len()?.checked_mul(scalar_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceOpStatus {
    Completed,
    Unsupported,
    ValidationFailed,
    OutOfMemory,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceTransferCounters {
    h2d_bytes: usize,
    d2h_bytes: usize,
}

impl DeviceTransferCounters {
    pub const fn new() -> Self {
        Self {
            h2d_bytes: 0,
            d2h_bytes: 0,
        }
    }

    pub fn record_h2d(&mut self, bytes: usize) {
        self.h2d_bytes = self.h2d_bytes.saturating_add(bytes);
    }

    pub fn record_d2h(&mut self, bytes: usize) {
        self.d2h_bytes = self.d2h_bytes.saturating_add(bytes);
    }

    pub const fn h2d_bytes(self) -> usize {
        self.h2d_bytes
    }

    pub const fn d2h_bytes(self) -> usize {
        self.d2h_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    F32,
    F16,
    BF16,
    Q40,
    Q41,
    Q50,
    Q51,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q80,
    Q81,
    IQ2XXS,
    IQ2S,
    IQ3XXS,
    IQ4XS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorShape {
    rows: usize,
    cols: usize,
}

impl TensorShape {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn cols(self) -> usize {
        self.cols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatMulRequest {
    weights: TensorShape,
    input: TensorShape,
    quant: QuantFormat,
    output_type: ScalarType,
}

impl MatMulRequest {
    pub const fn new(
        weights: TensorShape,
        input: TensorShape,
        quant: QuantFormat,
        output_type: ScalarType,
    ) -> Self {
        Self {
            weights,
            input,
            quant,
            output_type,
        }
    }

    pub const fn weights(self) -> TensorShape {
        self.weights
    }

    pub const fn input(self) -> TensorShape {
        self.input
    }

    pub const fn quant(self) -> QuantFormat {
        self.quant
    }

    pub const fn output_type(self) -> ScalarType {
        self.output_type
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionRequest {
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: ScalarType,
}

impl AttentionRequest {
    pub const fn new(
        seq_len: usize,
        kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: ScalarType,
    ) -> Self {
        Self {
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            dtype,
        }
    }

    pub const fn seq_len(self) -> usize {
        self.seq_len
    }

    pub const fn kv_len(self) -> usize {
        self.kv_len
    }

    pub const fn num_heads(self) -> usize {
        self.num_heads
    }

    pub const fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn dtype(self) -> ScalarType {
        self.dtype
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionKvMaterializeRequest {
    layer_idx: usize,
    num_kv_heads: usize,
    total_tokens: usize,
    head_dim: usize,
    kv_dim: usize,
}

impl AttentionKvMaterializeRequest {
    pub const fn new(
        layer_idx: usize,
        num_kv_heads: usize,
        total_tokens: usize,
        head_dim: usize,
        kv_dim: usize,
    ) -> Self {
        Self {
            layer_idx,
            num_kv_heads,
            total_tokens,
            head_dim,
            kv_dim,
        }
    }

    pub const fn layer_idx(self) -> usize {
        self.layer_idx
    }

    pub const fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    pub const fn total_tokens(self) -> usize {
        self.total_tokens
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn kv_dim(self) -> usize {
        self.kv_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionKvMaterializeRangeRequest {
    layer_idx: usize,
    num_kv_heads: usize,
    pos_start: usize,
    kv_len: usize,
    head_dim: usize,
}

impl AttentionKvMaterializeRangeRequest {
    pub const fn new(
        layer_idx: usize,
        num_kv_heads: usize,
        pos_start: usize,
        kv_len: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            layer_idx,
            num_kv_heads,
            pos_start,
            kv_len,
            head_dim,
        }
    }

    pub const fn layer_idx(self) -> usize {
        self.layer_idx
    }

    pub const fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    pub const fn pos_start(self) -> usize {
        self.pos_start
    }

    pub const fn kv_len(self) -> usize {
        self.kv_len
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KvarnDecodeRequest<'a> {
    layer_idx: usize,
    query: &'a [f32],
    packed_blocks: &'a [u8],
    sink_key: &'a [u16],
    sink_value: &'a [u16],
    tail_key: &'a [u16],
    tail_value: &'a [u16],
    kv_len: usize,
    tail_start: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    key_bits: u8,
    value_bits: u8,
    group: usize,
    sink_tokens: usize,
    block_bytes: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
}

/// decode chain in-chain KVarn attention 전송용 KV view. `KvarnDecodeRequest` 에서
/// query(=chain 내부에서 device 계산)만 뺀 것. metal backend 가 dummy query 로
/// `KvarnDecodeRequest` 를 재구성해 params/resident.update 에 쓴다(per-op 시그니처 불변).
#[derive(Debug, Clone, Copy)]
pub struct KvarnChainView<'a> {
    pub layer_idx: usize,
    pub packed_blocks: &'a [u8],
    pub sink_key: &'a [u16],
    pub sink_value: &'a [u16],
    pub tail_key: &'a [u16],
    pub tail_value: &'a [u16],
    pub kv_len: usize,
    pub tail_start: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub key_bits: u8,
    pub value_bits: u8,
    pub group: usize,
    pub sink_tokens: usize,
    pub block_bytes: usize,
    pub scale: f32,
    pub sliding_window: Option<usize>,
    pub softcap: Option<f32>,
}

impl<'a> KvarnDecodeRequest<'a> {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        layer_idx: usize,
        query: &'a [f32],
        packed_blocks: &'a [u8],
        sink_key: &'a [u16],
        sink_value: &'a [u16],
        tail_key: &'a [u16],
        tail_value: &'a [u16],
        kv_len: usize,
        tail_start: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        key_bits: u8,
        value_bits: u8,
        group: usize,
        sink_tokens: usize,
        block_bytes: usize,
        scale: f32,
        sliding_window: Option<usize>,
        softcap: Option<f32>,
    ) -> Self {
        Self {
            layer_idx,
            query,
            packed_blocks,
            sink_key,
            sink_value,
            tail_key,
            tail_value,
            kv_len,
            tail_start,
            num_heads,
            num_kv_heads,
            head_dim,
            key_bits,
            value_bits,
            group,
            sink_tokens,
            block_bytes,
            scale,
            sliding_window,
            softcap,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub const fn new_device_query(
        layer_idx: usize,
        packed_blocks: &'a [u8],
        sink_key: &'a [u16],
        sink_value: &'a [u16],
        tail_key: &'a [u16],
        tail_value: &'a [u16],
        kv_len: usize,
        tail_start: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        key_bits: u8,
        value_bits: u8,
        group: usize,
        sink_tokens: usize,
        block_bytes: usize,
        scale: f32,
        sliding_window: Option<usize>,
        softcap: Option<f32>,
    ) -> Self {
        Self::new(
            layer_idx,
            &[],
            packed_blocks,
            sink_key,
            sink_value,
            tail_key,
            tail_value,
            kv_len,
            tail_start,
            num_heads,
            num_kv_heads,
            head_dim,
            key_bits,
            value_bits,
            group,
            sink_tokens,
            block_bytes,
            scale,
            sliding_window,
            softcap,
        )
    }

    pub fn validate_device_query(self, query_rows: usize) -> Result<(), String> {
        self.validate_layout()?;
        if query_rows == 0 {
            return Err("KVarN device query row count must be non-zero".to_string());
        }
        Ok(())
    }

    pub fn validate(self) -> Result<(), String> {
        self.validate_layout()?;
        if self.query.len() != self.num_heads.saturating_mul(self.head_dim) {
            return Err("KVarN decode query length mismatch".to_string());
        }
        Ok(())
    }

    fn validate_layout(self) -> Result<(), String> {
        if self.num_heads == 0 || self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0
        {
            return Err("KVarN decode requires a non-zero integral GQA head ratio".to_string());
        }
        if !matches!(self.head_dim, 128 | 256 | 512) {
            return Err(format!(
                "KVarN device decode unsupported head_dim={}",
                self.head_dim
            ));
        }
        if self.key_bits != 4
            || !matches!(self.value_bits, 2 | 4)
            || !matches!(self.group, 64 | 128)
        {
            return Err(format!(
                "KVarN device decode unsupported K{}V{} G{}",
                self.key_bits, self.value_bits, self.group
            ));
        }
        let row_width = self.num_kv_heads.saturating_mul(self.head_dim);
        if row_width == 0
            || self.sink_key.len() != self.sink_value.len()
            || self.tail_key.len() != self.tail_value.len()
            || self.sink_key.len() % row_width != 0
            || self.tail_key.len() % row_width != 0
        {
            return Err("KVarN decode F16 region shape mismatch".to_string());
        }
        if self.block_bytes == 0 || self.packed_blocks.len() % self.block_bytes != 0 {
            return Err("KVarN decode packed block shape mismatch".to_string());
        }
        let sink_len = self.sink_key.len() / row_width;
        if sink_len > self.sink_tokens {
            return Err("KVarN decode sink exceeds configured capacity".to_string());
        }
        let block_rows = (self.packed_blocks.len() / self.block_bytes).saturating_mul(self.group);
        let tail_len = self.tail_key.len() / row_width;
        if tail_len > self.group {
            return Err("KVarN decode tail exceeds one quantization group".to_string());
        }
        if sink_len.saturating_add(block_rows).saturating_add(tail_len) != self.kv_len
            || (tail_len > 0 && self.tail_start != sink_len.saturating_add(block_rows))
        {
            return Err("KVarN decode logical length mismatch".to_string());
        }
        if self.sliding_window == Some(0) {
            return Err("KVarN decode sliding window must be non-zero".to_string());
        }
        Ok(())
    }

    pub const fn layer_idx(self) -> usize {
        self.layer_idx
    }

    pub const fn query(self) -> &'a [f32] {
        self.query
    }

    pub const fn packed_blocks(self) -> &'a [u8] {
        self.packed_blocks
    }

    pub const fn sink_key(self) -> &'a [u16] {
        self.sink_key
    }

    pub const fn sink_value(self) -> &'a [u16] {
        self.sink_value
    }

    pub const fn tail_key(self) -> &'a [u16] {
        self.tail_key
    }

    pub const fn tail_value(self) -> &'a [u16] {
        self.tail_value
    }

    pub const fn kv_len(self) -> usize {
        self.kv_len
    }

    pub const fn tail_start(self) -> usize {
        self.tail_start
    }

    pub const fn num_heads(self) -> usize {
        self.num_heads
    }

    pub const fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn key_bits(self) -> u8 {
        self.key_bits
    }

    pub const fn value_bits(self) -> u8 {
        self.value_bits
    }

    pub const fn group(self) -> usize {
        self.group
    }

    pub const fn sink_tokens(self) -> usize {
        self.sink_tokens
    }

    pub const fn block_bytes(self) -> usize {
        self.block_bytes
    }

    pub const fn scale(self) -> f32 {
        self.scale
    }

    pub const fn sliding_window(self) -> Option<usize> {
        self.sliding_window
    }

    pub const fn softcap(self) -> Option<f32> {
        self.softcap
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvBucketView {
    layer_idx: usize,
    page_size: usize,
    current_len: usize,
    max_len: usize,
    kv_row_width: usize,
    k_identity: u64,
    v_identity: u64,
    page_count: usize,
    last_page_len: usize,
}

impl KvBucketView {
    pub fn new(
        layer_idx: usize,
        page_size: usize,
        current_len: usize,
        max_len: usize,
        kv_row_width: usize,
        k_identity: u64,
        v_identity: u64,
    ) -> BackendResult<Self> {
        if page_size == 0 {
            return Err(kv_bucket_view_error("KV bucket page size must be non-zero"));
        }
        if max_len == 0 {
            return Err(kv_bucket_view_error(
                "KV bucket max length must be non-zero",
            ));
        }
        if current_len > max_len {
            return Err(kv_bucket_view_error(
                "KV bucket current length cannot exceed max length",
            ));
        }
        if kv_row_width == 0 {
            return Err(kv_bucket_view_error("KV bucket row width must be non-zero"));
        }
        if k_identity == 0 || v_identity == 0 {
            return Err(kv_bucket_view_error(
                "KV bucket K/V identities must be non-zero",
            ));
        }

        let page_count = current_len.div_ceil(page_size).max(1);
        let last_page_len = if current_len == 0 {
            0
        } else {
            let rem = current_len % page_size;
            if rem == 0 {
                page_size
            } else {
                rem
            }
        };

        Ok(Self {
            layer_idx,
            page_size,
            current_len,
            max_len,
            kv_row_width,
            k_identity,
            v_identity,
            page_count,
            last_page_len,
        })
    }

    pub const fn layer_idx(self) -> usize {
        self.layer_idx
    }

    pub const fn page_size(self) -> usize {
        self.page_size
    }

    pub const fn current_len(self) -> usize {
        self.current_len
    }

    pub const fn max_len(self) -> usize {
        self.max_len
    }

    pub const fn kv_row_width(self) -> usize {
        self.kv_row_width
    }

    pub const fn k_identity(self) -> u64 {
        self.k_identity
    }

    pub const fn v_identity(self) -> u64 {
        self.v_identity
    }

    pub const fn page_count(self) -> usize {
        self.page_count
    }

    pub const fn last_page_len(self) -> usize {
        self.last_page_len
    }
}

fn kv_bucket_view_error(message: impl Into<String>) -> BackendError {
    BackendError::new(
        BackendErrorKind::InvalidRequest,
        BackendKind::Cpu,
        Some(BackendOp::Attention),
        message,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoeRequest {
    selected_experts: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    quant: QuantFormat,
}

impl MoeRequest {
    pub const fn new(
        selected_experts: usize,
        hidden_dim: usize,
        ffn_dim: usize,
        quant: QuantFormat,
    ) -> Self {
        Self {
            selected_experts,
            hidden_dim,
            ffn_dim,
            quant,
        }
    }

    pub const fn selected_experts(self) -> usize {
        self.selected_experts
    }

    pub const fn hidden_dim(self) -> usize {
        self.hidden_dim
    }

    pub const fn ffn_dim(self) -> usize {
        self.ffn_dim
    }

    pub const fn quant(self) -> QuantFormat {
        self.quant
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GdnRequest {
    seq_len: usize,
    hidden_dim: usize,
    head_dim: usize,
    num_heads: usize,
    conv_kernel: usize,
    quant: QuantFormat,
}

impl GdnRequest {
    pub const fn new(
        seq_len: usize,
        hidden_dim: usize,
        head_dim: usize,
        num_heads: usize,
        conv_kernel: usize,
        quant: QuantFormat,
    ) -> Self {
        Self {
            seq_len,
            hidden_dim,
            head_dim,
            num_heads,
            conv_kernel,
            quant,
        }
    }

    pub const fn seq_len(self) -> usize {
        self.seq_len
    }

    pub const fn hidden_dim(self) -> usize {
        self.hidden_dim
    }

    pub const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub const fn num_heads(self) -> usize {
        self.num_heads
    }

    pub const fn conv_kernel(self) -> usize {
        self.conv_kernel
    }

    pub const fn quant(self) -> QuantFormat {
        self.quant
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecodeWeightKind {
    QProj,
    KProj,
    VProj,
    OProj,
    FfnGate,
    FfnDown,
    GdnQkv,
    GdnGate,
    GdnAlpha,
    GdnBeta,
    GdnSsmOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantizedWeightView<'a> {
    raw: &'a [u8],
    rows: usize,
    cols: usize,
    quant: QuantFormat,
}

impl<'a> QuantizedWeightView<'a> {
    pub const fn new(raw: &'a [u8], rows: usize, cols: usize, quant: QuantFormat) -> Self {
        Self {
            raw,
            rows,
            cols,
            quant,
        }
    }

    pub const fn raw(self) -> &'a [u8] {
        self.raw
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn cols(self) -> usize {
        self.cols
    }

    pub const fn quant(self) -> QuantFormat {
        self.quant
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformedWeightLayout {
    Q4kCompactMetadata,
    Q6kPackedQ8dot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformedSourceQuant {
    DenseQ4kRowPair,
    DenseQ6k,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransformedWeightView<'a> {
    layout: TransformedWeightLayout,
    source_quant: TransformedSourceQuant,
    rows: usize,
    cols: usize,
    source_fingerprint: u64,
    block_rows: usize,
    block_cols: usize,
    producer_options_hash: u64,
    source_bytes: &'a [u8],
    blocks_per_row: usize,
}

impl<'a> TransformedWeightView<'a> {
    pub fn new(
        layout: TransformedWeightLayout,
        source_quant: TransformedSourceQuant,
        rows: usize,
        cols: usize,
        source_fingerprint: u64,
        block_rows: usize,
        block_cols: usize,
        producer_options_hash: u64,
        source_bytes: &'a [u8],
    ) -> BackendResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(transformed_weight_view_error(
                "transformed weight shape must be non-zero",
            ));
        }
        if block_rows == 0 || block_cols == 0 {
            return Err(transformed_weight_view_error(
                "transformed weight block geometry must be non-zero",
            ));
        }
        if block_rows != 1 || block_cols != 256 {
            return Err(transformed_weight_view_error(format!(
                "unsupported transformed block geometry: {block_rows}x{block_cols}"
            )));
        }
        if cols % block_cols != 0 {
            return Err(transformed_weight_view_error(format!(
                "transformed weight cols={cols} must be a multiple of block_cols={block_cols}"
            )));
        }
        let block_bytes = match (layout, source_quant) {
            (
                TransformedWeightLayout::Q4kCompactMetadata,
                TransformedSourceQuant::DenseQ4kRowPair,
            ) => 144,
            (TransformedWeightLayout::Q6kPackedQ8dot, TransformedSourceQuant::DenseQ6k) => 210,
            _ => {
                return Err(transformed_weight_view_error(format!(
                    "transformed layout {layout:?} is not compatible with source quant {source_quant:?}"
                )));
            }
        };
        let blocks_per_row = cols / block_cols;
        let expected_len = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(block_bytes))
            .ok_or_else(|| {
                transformed_weight_view_error(format!(
                    "transformed weight byte size overflow: rows={rows} cols={cols}"
                ))
            })?;
        if source_bytes.len() != expected_len {
            return Err(transformed_weight_view_error(format!(
                "transformed source byte length mismatch: got {}, expected {expected_len}",
                source_bytes.len()
            )));
        }
        Ok(Self {
            layout,
            source_quant,
            rows,
            cols,
            source_fingerprint,
            block_rows,
            block_cols,
            producer_options_hash,
            source_bytes,
            blocks_per_row,
        })
    }

    pub const fn layout(self) -> TransformedWeightLayout {
        self.layout
    }

    pub const fn source_quant(self) -> TransformedSourceQuant {
        self.source_quant
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn cols(self) -> usize {
        self.cols
    }

    pub const fn source_len(self) -> usize {
        self.source_bytes.len()
    }

    pub const fn source_fingerprint(self) -> u64 {
        self.source_fingerprint
    }

    pub const fn block_rows(self) -> usize {
        self.block_rows
    }

    pub const fn block_cols(self) -> usize {
        self.block_cols
    }

    pub const fn producer_options_hash(self) -> u64 {
        self.producer_options_hash
    }

    pub const fn blocks_per_row(self) -> usize {
        self.blocks_per_row
    }

    pub const fn source_bytes(self) -> &'a [u8] {
        self.source_bytes
    }
}

fn transformed_weight_view_error(message: impl Into<String>) -> BackendError {
    BackendError::new(
        BackendErrorKind::InvalidRequest,
        BackendKind::Cpu,
        Some(BackendOp::MatMul),
        message,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeRouteSlot {
    pub expert: usize,
    pub token: u32,
    pub weight: f32,
}

impl MoeRouteSlot {
    pub const fn new(expert: usize, token: u32, weight: f32) -> Self {
        Self {
            expert,
            token,
            weight,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendWorkload {
    OpOnly(BackendOp),
    MatMul(MatMulRequest),
    Attention(AttentionRequest),
    Gdn(GdnRequest),
    MoE(MoeRequest),
    Sampler,
}

impl BackendWorkload {
    pub const fn op(self) -> BackendOp {
        match self {
            Self::OpOnly(op) => op,
            Self::MatMul(_) => BackendOp::MatMul,
            Self::Attention(_) => BackendOp::Attention,
            Self::Gdn(_) => BackendOp::Gdn,
            Self::MoE(_) => BackendOp::MoE,
            Self::Sampler => BackendOp::Sampler,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendRequest {
    op: BackendOp,
    workload: BackendWorkload,
}

impl BackendRequest {
    pub const fn new(op: BackendOp) -> Self {
        Self {
            op,
            workload: BackendWorkload::OpOnly(op),
        }
    }

    pub const fn from_workload(workload: BackendWorkload) -> Self {
        Self {
            op: workload.op(),
            workload,
        }
    }

    pub const fn matmul(request: MatMulRequest) -> Self {
        Self::from_workload(BackendWorkload::MatMul(request))
    }

    pub const fn attention(request: AttentionRequest) -> Self {
        Self::from_workload(BackendWorkload::Attention(request))
    }

    pub const fn gdn(request: GdnRequest) -> Self {
        Self::from_workload(BackendWorkload::Gdn(request))
    }

    pub const fn moe(request: MoeRequest) -> Self {
        Self::from_workload(BackendWorkload::MoE(request))
    }

    pub const fn op(&self) -> BackendOp {
        self.op
    }

    pub const fn workload(&self) -> BackendWorkload {
        self.workload
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendOutput {
    op: BackendOp,
}

impl BackendOutput {
    pub const fn new(op: BackendOp) -> Self {
        Self { op }
    }

    pub const fn op(&self) -> BackendOp {
        self.op
    }
}

pub trait Backend {
    fn kind(&self) -> BackendKind;
    fn capabilities(&self) -> BackendCapabilities;
    fn execute(&mut self, request: BackendRequest) -> BackendResult<BackendOutput>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeJitLoadRequest {
    pub backend_hint: Option<BackendKind>,
    pub layer_idx: usize,
    pub experts: Vec<usize>,
    pub gate_bytes_per_expert: usize,
    pub up_bytes_per_expert: usize,
    pub down_bytes_per_expert: usize,
    pub expert_loads: Vec<MoeJitExpertLoad>,
}

impl MoeJitLoadRequest {
    pub fn expert_bytes_per_sparse_expert(&self) -> usize {
        self.gate_bytes_per_expert
            .saturating_add(self.up_bytes_per_expert)
            .saturating_add(self.down_bytes_per_expert)
    }

    pub fn requested_bytes(&self) -> usize {
        self.expert_loads
            .iter()
            .map(MoeJitExpertLoad::total_bytes)
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeJitByteRange {
    pub ptr_addr: usize,
    pub tensor_offset: usize,
    pub len: usize,
}

impl MoeJitByteRange {
    pub fn from_tensor_slice(tensor: &[u8], tensor_offset: usize, len: usize) -> Self {
        assert!(
            tensor_offset.saturating_add(len) <= tensor.len(),
            "moe jit byte range out of bounds: offset {} + len {} > tensor {}",
            tensor_offset,
            len,
            tensor.len()
        );
        Self {
            ptr_addr: unsafe { tensor.as_ptr().add(tensor_offset) as usize },
            tensor_offset,
            len,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeJitExpertLoad {
    pub expert: usize,
    pub gate: MoeJitByteRange,
    pub up: MoeJitByteRange,
    pub down: MoeJitByteRange,
}

impl MoeJitExpertLoad {
    pub fn total_bytes(&self) -> usize {
        self.gate
            .len
            .saturating_add(self.up.len)
            .saturating_add(self.down.len)
    }
}

pub trait MoeJitLoadSink: Send + Sync {
    fn request_load(&self, request: &MoeJitLoadRequest);
}

/// Per-layer host inputs for `PersistentDecodeRequest`.
///
/// `*_bytes` pointers point to host-resident Q4_K (or F32 for PLE) weight
/// data. Each backend is free to copy them once into device-resident cache;
/// `kv_*_bytes` similarly are host f16 (u16) bytes of the layer's KV cache.
///
/// All shape fields are u32 so the layout matches what the persistent-decode
/// CUDA kernel reads from device memory.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PersistentDecodeLayerInput {
    pub q_weight_bytes: *const u8,
    pub q_weight_len: usize,
    pub k_weight_bytes: *const u8,
    pub k_weight_len: usize,
    pub v_weight_bytes: *const u8,
    pub v_weight_len: usize,
    pub o_weight_bytes: *const u8,
    pub o_weight_len: usize,
    pub gate_weight_bytes: *const u8,
    pub gate_weight_len: usize,
    pub up_weight_bytes: *const u8,
    pub up_weight_len: usize,
    pub down_weight_bytes: *const u8,
    pub down_weight_len: usize,
    pub attn_norm_bytes: *const u8,
    pub attn_norm_len: usize,
    pub post_attn_norm_bytes: *const u8,
    pub post_attn_norm_len: usize,
    pub ffn_norm_bytes: *const u8,
    pub ffn_norm_len: usize,
    pub post_ffn_norm_bytes: *const u8,
    pub post_ffn_norm_len: usize,
    pub q_norm_bytes: *const u8,
    pub q_norm_len: usize,
    pub k_norm_bytes: *const u8,
    pub k_norm_len: usize,
    pub ple_gate_bytes: *const u8,
    pub ple_gate_len: usize,
    pub ple_proj_bytes: *const u8,
    pub ple_proj_len: usize,
    pub ple_post_norm_bytes: *const u8,
    pub ple_post_norm_len: usize,
    pub ple_input_bytes: *const u8,
    pub ple_input_len: usize,
    pub k_cache_bytes: *const u8,
    pub k_cache_len: usize,
    pub v_cache_bytes: *const u8,
    pub v_cache_len: usize,
    pub head_dim: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub n_ff: u32,
    pub sliding_window: u32,
    pub kv_source_layer: u32,
    pub layer_output_scale: f32,
    pub flags: u32,
}

/// Single-step persistent decode dispatch request.
///
/// `layers` provides per-layer host pointers (Q/K/V/O/FFN/PLE weights + KV
/// cache).  `input_hidden` is the layer-0 input (f32, `hidden_dim`).
/// `output_logits` is a caller-owned `[vocab_size]` f32 buffer that the
/// backend writes to.  `argmax_out` receives the selected token id.
#[repr(C)]
pub struct PersistentDecodeRequest<'a> {
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub vocab_size: u32,
    pub norm_eps: f32,
    pub rope_pos: u32,
    pub kv_len: u32,
    pub max_seq_len: u32,
    pub q_dim_max: u32,
    pub kv_dim_max: u32,
    pub n_ff_max: u32,
    pub ple_dim: u32,
    pub layers: &'a [PersistentDecodeLayerInput],
    pub output_weight_bytes: *const u8,
    pub output_weight_len: usize,
    /// cu100 Milestone 2 — batch prefill seq_len. `1` = decode (current),
    /// `> 1` = persistent prefill batch (token loop in kernel).
    /// `input_hidden.len()` must equal `seq_len * hidden_dim`.
    pub seq_len: u32,
    pub input_hidden: &'a [f32],
    pub output_logits: &'a mut [f32],
    pub argmax_out: &'a mut i32,
    /// cu76 diag: when `Some`, backend D2H-copies the hidden state AFTER the
    /// last active layer (i.e. respecting `RNB_CUDA_PERSISTENT_DECODE_LAYERS`)
    /// into this buffer.  Length must equal `hidden_dim`.  Used to identify
    /// the first divergence layer vs eager scratch.hidden.
    pub hidden_probe: Option<&'a mut [f32]>,
    /// cu76 phase probes (layer-0 only).  D2H snapshots from inside layer 0.
    pub normed_after_attn_norm_probe: Option<&'a mut [f32]>,
    pub hidden_after_attn_probe: Option<&'a mut [f32]>,
    pub hidden_after_ffn_probe: Option<&'a mut [f32]>,
    /// cu77: Gemma4 RoPE freq_factors (length = head_dim/2 = 256, f32 bytes).
    /// `None` when model has no rope_freqs.  FULL attention layers apply this;
    /// SWA layers skip via sliding_window check.
    pub rope_freqs_bytes: *const u8,
    pub rope_freqs_len: usize,
    /// cu78 fine-grained phase probes (layer-0 only, head-0 / head_dim floats).
    pub attn_out_probe: Option<&'a mut [f32]>,
    pub q_proj_probe: Option<&'a mut [f32]>,
    pub k_proj_probe: Option<&'a mut [f32]>,
    pub v_proj_probe: Option<&'a mut [f32]>,
    pub attn_scores_probe: Option<&'a mut [f32]>,
    pub attn_v_probe: Option<&'a mut [f32]>,
    pub attn_acc_probe: Option<&'a mut [f32]>,
    pub attn_row_sum_probe: Option<&'a mut [f32]>,
    pub hidden_after_ffn_only_probe: Option<&'a mut [f32]>,
    pub ffn_gate_probe: Option<&'a mut [f32]>,
    pub ffn_gated_probe: Option<&'a mut [f32]>,
    pub ffn_down_probe: Option<&'a mut [f32]>,
    pub layer_hidden_trace: Option<&'a mut [f32]>,
    /// cu91: Gemma4 output_norm weight (f32, hidden_dim).
    pub output_norm_bytes: *const u8,
    pub output_norm_len: usize,
}

pub const PERSISTENT_DECODE_FLAG_GATED_ATTN: u32 = 1 << 0;
pub const PERSISTENT_DECODE_FLAG_PLE_F32: u32 = 1 << 1;
pub const PERSISTENT_DECODE_FLAG_REUSE_Q: u32 = 1 << 2;
pub const PERSISTENT_DECODE_FLAG_ATTN_ROT: u32 = 1 << 3;
pub const PERSISTENT_DECODE_FLAG_V_Q6K: u32 = 1 << 4;
pub const PERSISTENT_DECODE_FLAG_DOWN_Q6K: u32 = 1 << 5;
pub const PERSISTENT_DECODE_FLAG_O_Q6K: u32 = 1 << 6;
pub const PERSISTENT_DECODE_FLAG_K_Q6K: u32 = 1 << 7;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_record_supported_ops() {
        let caps = BackendCapabilities::new(BackendKind::Cuda)
            .with_op(BackendOp::MatMul)
            .with_op(BackendOp::Attention);

        assert!(caps.supports(BackendOp::MatMul));
        assert!(!caps.supports(BackendOp::Sampler));
    }

    #[test]
    fn unsupported_op_error_names_backend_and_op() {
        let err = BackendError::unsupported(BackendKind::OpenCl, BackendOp::MoE);
        assert_eq!(err.backend(), BackendKind::OpenCl);
        assert_eq!(err.op(), Some(BackendOp::MoE));
    }

    #[test]
    fn moe_jit_load_request_counts_requested_bytes() {
        let tensor = [0u8; 64];
        let load = MoeJitExpertLoad {
            expert: 3,
            gate: MoeJitByteRange::from_tensor_slice(&tensor, 0, 8),
            up: MoeJitByteRange::from_tensor_slice(&tensor, 8, 16),
            down: MoeJitByteRange::from_tensor_slice(&tensor, 24, 32),
        };
        let request = MoeJitLoadRequest {
            backend_hint: Some(BackendKind::Cuda),
            layer_idx: 1,
            experts: vec![3],
            gate_bytes_per_expert: 8,
            up_bytes_per_expert: 16,
            down_bytes_per_expert: 32,
            expert_loads: vec![load],
        };

        assert_eq!(request.expert_bytes_per_sparse_expert(), 56);
        assert_eq!(request.requested_bytes(), 56);
    }

    #[test]
    fn backend_request_can_carry_matmul_shape_contract() {
        let matmul = MatMulRequest::new(
            TensorShape::new(4096, 11008),
            TensorShape::new(1, 4096),
            QuantFormat::Q4K,
            ScalarType::F32,
        );
        let request = BackendRequest::matmul(matmul);

        assert_eq!(request.op(), BackendOp::MatMul);
        assert_eq!(request.workload(), BackendWorkload::MatMul(matmul));
        assert_eq!(matmul.weights().rows(), 4096);
        assert_eq!(matmul.input().cols(), 4096);
    }

    #[test]
    fn backend_request_derives_op_from_attention_and_moe_workloads() {
        let attention =
            BackendRequest::attention(AttentionRequest::new(1, 128, 32, 4, 128, ScalarType::F16));
        let gdn = BackendRequest::gdn(GdnRequest::new(64, 2048, 128, 16, 4, QuantFormat::Q6K));
        let moe = BackendRequest::moe(MoeRequest::new(8, 4096, 11008, QuantFormat::Q6K));

        assert_eq!(attention.op(), BackendOp::Attention);
        assert_eq!(gdn.op(), BackendOp::Gdn);
        assert_eq!(
            gdn.workload(),
            BackendWorkload::Gdn(GdnRequest::new(64, 2048, 128, 16, 4, QuantFormat::Q6K))
        );
        assert_eq!(moe.op(), BackendOp::MoE);
    }

    #[test]
    fn attention_kv_materialize_requests_are_plain_shape_contracts() {
        let full = AttentionKvMaterializeRequest::new(2, 4, 128, 64, 256);
        let range = AttentionKvMaterializeRangeRequest::new(3, 1, 32, 16, 128);

        assert_eq!(full.layer_idx(), 2);
        assert_eq!(full.num_kv_heads(), 4);
        assert_eq!(full.total_tokens(), 128);
        assert_eq!(full.head_dim(), 64);
        assert_eq!(full.kv_dim(), 256);
        assert_eq!(range.layer_idx(), 3);
        assert_eq!(range.num_kv_heads(), 1);
        assert_eq!(range.pos_start(), 32);
        assert_eq!(range.kv_len(), 16);
        assert_eq!(range.head_dim(), 128);
    }

    #[test]
    fn kv_bucket_view_reports_contiguous_single_page_layout() {
        let view = KvBucketView::new(2, 128, 17, 128, 512, 0x1000, 0x2000).expect("kv view");

        assert_eq!(view.layer_idx(), 2);
        assert_eq!(view.page_size(), 128);
        assert_eq!(view.current_len(), 17);
        assert_eq!(view.max_len(), 128);
        assert_eq!(view.kv_row_width(), 512);
        assert_eq!(view.page_count(), 1);
        assert_eq!(view.last_page_len(), 17);
        assert_eq!(view.k_identity(), 0x1000);
        assert_eq!(view.v_identity(), 0x2000);
    }

    #[test]
    fn kv_bucket_view_reports_multi_page_lengths() {
        let view = KvBucketView::new(0, 128, 257, 512, 64, 0x1000, 0x2000).expect("kv view");

        assert_eq!(view.page_count(), 3);
        assert_eq!(view.last_page_len(), 1);
    }

    #[test]
    fn kv_bucket_view_rejects_invalid_metadata() {
        assert!(KvBucketView::new(0, 0, 0, 128, 512, 0x1000, 0x2000).is_err());
        assert!(KvBucketView::new(0, 128, 129, 128, 512, 0x1000, 0x2000).is_err());
        assert!(KvBucketView::new(0, 128, 17, 128, 0, 0x1000, 0x2000).is_err());
        assert!(KvBucketView::new(0, 128, 17, 128, 512, 0, 0x2000).is_err());
        assert!(KvBucketView::new(0, 128, 17, 128, 512, 0x1000, 0).is_err());
    }

    #[test]
    fn transformed_weight_view_accepts_supported_q4_q6_layouts() {
        let q4_source = vec![0xA5u8; 2 * 144];
        let q4 = TransformedWeightView::new(
            TransformedWeightLayout::Q4kCompactMetadata,
            TransformedSourceQuant::DenseQ4kRowPair,
            2,
            256,
            0x1111,
            1,
            256,
            0x1,
            &q4_source,
        )
        .expect("q4 transformed view");
        assert_eq!(q4.layout(), TransformedWeightLayout::Q4kCompactMetadata);
        assert_eq!(q4.blocks_per_row(), 1);
        assert_eq!(q4.source_bytes(), q4_source.as_slice());

        let q6_source = vec![0x5Au8; 3 * 2 * 210];
        let q6 = TransformedWeightView::new(
            TransformedWeightLayout::Q6kPackedQ8dot,
            TransformedSourceQuant::DenseQ6k,
            3,
            512,
            0x2222,
            1,
            256,
            0x1,
            &q6_source,
        )
        .expect("q6 transformed view");
        assert_eq!(q6.layout(), TransformedWeightLayout::Q6kPackedQ8dot);
        assert_eq!(q6.blocks_per_row(), 2);
        assert_eq!(q6.source_len(), q6_source.len());
    }

    #[test]
    fn transformed_weight_view_rejects_mismatched_layout_quant_or_shape() {
        let q4_source = vec![0xA5u8; 144];
        assert!(TransformedWeightView::new(
            TransformedWeightLayout::Q6kPackedQ8dot,
            TransformedSourceQuant::DenseQ4kRowPair,
            1,
            256,
            0x1111,
            1,
            256,
            0x1,
            &q4_source,
        )
        .is_err());
        assert!(TransformedWeightView::new(
            TransformedWeightLayout::Q4kCompactMetadata,
            TransformedSourceQuant::DenseQ4kRowPair,
            1,
            384,
            0x1111,
            1,
            256,
            0x1,
            &q4_source,
        )
        .is_err());
        assert!(TransformedWeightView::new(
            TransformedWeightLayout::Q4kCompactMetadata,
            TransformedSourceQuant::DenseQ4kRowPair,
            1,
            256,
            0x1111,
            1,
            256,
            0x1,
            &q4_source[..143],
        )
        .is_err());
    }

    #[test]
    fn quantized_weight_view_and_moe_route_slot_are_plain_execution_contracts() {
        let raw = [1u8, 2, 3, 4];
        let view = QuantizedWeightView::new(&raw, 2, 2, QuantFormat::Q6K);
        let slot = MoeRouteSlot::new(7, 3, 0.25);

        assert_eq!(view.raw(), &raw);
        assert_eq!(view.rows(), 2);
        assert_eq!(view.cols(), 2);
        assert_eq!(view.quant(), QuantFormat::Q6K);
        assert_eq!(slot.expert, 7);
        assert_eq!(slot.token, 3);
        assert_eq!(slot.weight, 0.25);
    }
}

#[cfg(test)]
mod device_tensor_tests {
    use super::*;

    #[test]
    fn device_tensor_desc_reports_f32_bytes() {
        let desc = DeviceTensorDesc::new(3, 5, ScalarType::F32, DeviceTensorRole::Hidden);

        assert_eq!(desc.rows(), 3);
        assert_eq!(desc.cols(), 5);
        assert_eq!(desc.checked_len(), Some(15));
        assert_eq!(desc.len(), 15);
        assert_eq!(desc.byte_len(), Some(60));
        assert_eq!(desc.role(), DeviceTensorRole::Hidden);
    }

    #[test]
    fn device_tensor_desc_accepts_mamba_output_role() {
        let desc = DeviceTensorDesc::new(2, 7, ScalarType::F32, DeviceTensorRole::MambaOutput);

        assert_eq!(desc.rows(), 2);
        assert_eq!(desc.cols(), 7);
        assert_eq!(desc.role(), DeviceTensorRole::MambaOutput);
        assert_eq!(desc.byte_len(), Some(2 * 7 * 4));
    }

    #[test]
    fn device_tensor_desc_reports_scalar_byte_widths() {
        let cases = [
            (ScalarType::F16, Some(10)),
            (ScalarType::BF16, Some(10)),
            (ScalarType::I8, Some(5)),
            (ScalarType::U8, Some(5)),
        ];

        for (dtype, expected_bytes) in cases {
            let desc = DeviceTensorDesc::new(1, 5, dtype, DeviceTensorRole::Scratch);

            assert_eq!(desc.checked_len(), Some(5));
            assert_eq!(desc.byte_len(), expected_bytes);
        }
    }

    #[test]
    fn device_tensor_desc_reports_overflow_shape_as_none() {
        let desc = DeviceTensorDesc::new(usize::MAX, 2, ScalarType::F32, DeviceTensorRole::Scratch);

        assert_eq!(desc.checked_len(), None);
        assert_eq!(desc.byte_len(), None);
    }

    #[test]
    fn device_tensor_id_keeps_backend_axis() {
        let id = DeviceTensorId::new(BackendKind::Cuda, 42);

        assert_eq!(id.backend(), BackendKind::Cuda);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn device_transfer_counters_accumulate_and_saturate() {
        let mut counters = DeviceTransferCounters::new();

        counters.record_h2d(5);
        counters.record_h2d(7);
        counters.record_d2h(11);
        counters.record_d2h(13);

        assert_eq!(counters.h2d_bytes(), 12);
        assert_eq!(counters.d2h_bytes(), 24);

        counters.record_h2d(usize::MAX);
        counters.record_d2h(usize::MAX);

        assert_eq!(counters.h2d_bytes(), usize::MAX);
        assert_eq!(counters.d2h_bytes(), usize::MAX);
    }
}
