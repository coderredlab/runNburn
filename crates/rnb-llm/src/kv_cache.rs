#[derive(Clone)]
/// 단일 레이어의 K/V 텐서 캐시
pub struct LayerCache {
    pub key: Vec<u16>,   // F16 packed as u16 bits
    pub value: Vec<u16>, // F16 packed as u16 bits
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl LayerCache {
    fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let size = max_seq_len * num_kv_heads * head_dim;
        Self {
            key: vec![0u16; size],
            value: vec![0u16; size],
            num_kv_heads,
            head_dim,
            max_seq_len,
        }
    }

    fn write_at(&mut self, pos: usize, k_slice: &[f32], v_slice: &[f32]) {
        assert!(pos < self.max_seq_len, "KV cache overflow");
        let stride = self.num_kv_heads * self.head_dim;
        let start = pos * stride;
        for i in 0..stride {
            self.key[start + i] = half::f16::from_f32(k_slice[i]).to_bits();
            self.value[start + i] = half::f16::from_f32(v_slice[i]).to_bits();
        }
    }

    fn read_up_to(&self, len: usize) -> (&[u16], &[u16]) {
        let stride = self.num_kv_heads * self.head_dim;
        (&self.key[..len * stride], &self.value[..len * stride])
    }

    fn write_bits_up_to(&mut self, len: usize, k_bits: &[u16], v_bits: &[u16]) {
        let stride = self.num_kv_heads * self.head_dim;
        let count = len * stride;
        assert!(
            k_bits.len() >= count && v_bits.len() >= count,
            "KV bits underflow"
        );
        self.key[..count].copy_from_slice(&k_bits[..count]);
        self.value[..count].copy_from_slice(&v_bits[..count]);
    }

    fn write_q8_up_to(&mut self, len: usize, key: &CompactQ8Tensor, value: &CompactQ8Tensor) {
        let vector_width = self.head_dim;
        let count = len * self.num_kv_heads * vector_width;
        dequantize_compact_q8(&mut self.key[..count], key, vector_width);
        dequantize_compact_q8(&mut self.value[..count], value, vector_width);
    }

    fn write_bits_range(
        &mut self,
        pos_start: usize,
        kv_len: usize,
        k_bits: &[u16],
        v_bits: &[u16],
    ) {
        let stride = self.num_kv_heads * self.head_dim;
        let start = pos_start * stride;
        let count = kv_len * stride;
        let end = start + count;
        assert!(
            end <= self.key.len() && end <= self.value.len(),
            "KV bits range overflow"
        );
        assert!(
            k_bits.len() >= count && v_bits.len() >= count,
            "KV bits range underflow"
        );
        self.key[start..end].copy_from_slice(&k_bits[..count]);
        self.value[start..end].copy_from_slice(&v_bits[..count]);
    }
}

#[derive(Clone)]
/// SSM (Gated Delta Net) 레이어의 상태 캐시
pub struct SsmLayerState {
    /// Conv state: 마지막 (conv_kernel-1)개 입력 보관 [(conv_kernel-1) * conv_channels]
    pub conv_state: Vec<f32>,
    /// Delta net recurrent state: [num_heads * head_v_dim * head_k_dim]
    pub delta_state: Vec<f32>,
    pub conv_kernel: usize,
    pub conv_channels: usize,
}

impl SsmLayerState {
    fn new(
        conv_kernel: usize,
        conv_channels: usize,
        num_heads: usize,
        head_v_dim: usize,
        head_k_dim: usize,
    ) -> Self {
        Self {
            conv_state: vec![0.0f32; (conv_kernel - 1) * conv_channels],
            delta_state: vec![0.0f32; num_heads * head_v_dim * head_k_dim],
            conv_kernel,
            conv_channels,
        }
    }

    pub fn clear(&mut self) {
        self.conv_state.fill(0.0);
        self.delta_state.fill(0.0);
    }
}

#[derive(Clone)]
pub struct KVCache {
    layers: Vec<LayerCache>,
    /// SSM state per GDN layer (indexed by layer_idx, None for attention layers)
    pub ssm_states: Vec<Option<SsmLayerState>>,
    pub max_seq_len: usize,
    pub current_len: usize,
}
#[derive(Clone)]
pub(crate) struct CompactKVCacheSnapshot {
    layers: Vec<CompactLayerCache>,
    ssm_states: Vec<Option<SsmLayerState>>,
    max_seq_len: usize,
    current_len: usize,
}

#[derive(Clone)]
struct CompactQ8Tensor {
    values: Vec<i8>,
    scales: Vec<f32>,
}

impl CompactQ8Tensor {
    fn from_f16(bits: &[u16], vector_width: usize) -> Self {
        assert!(vector_width > 0, "compact Q8 vector width must be positive");
        assert_eq!(
            bits.len() % vector_width,
            0,
            "compact Q8 tensor has a partial vector"
        );
        let mut values = Vec::with_capacity(bits.len());
        let mut scales = Vec::with_capacity(bits.len() / vector_width);
        for vector in bits.chunks_exact(vector_width) {
            let max_abs = vector.iter().fold(0.0f32, |current, &bits| {
                current.max(half::f16::from_bits(bits).to_f32().abs())
            });
            let scale = max_abs / 127.0;
            scales.push(scale);
            if scale == 0.0 {
                values.resize(values.len() + vector_width, 0);
                continue;
            }
            values.extend(vector.iter().map(|&bits| {
                let value = half::f16::from_bits(bits).to_f32();
                (value / scale).round().clamp(-127.0, 127.0) as i8
            }));
        }
        Self { values, scales }
    }

    fn byte_size(&self) -> usize {
        self.values
            .capacity()
            .saturating_mul(std::mem::size_of::<i8>())
            .saturating_add(
                self.scales
                    .capacity()
                    .saturating_mul(std::mem::size_of::<f32>()),
            )
    }

    fn has_layout(&self, vector_count: usize, vector_width: usize) -> bool {
        self.values.len() == vector_count.saturating_mul(vector_width)
            && self.scales.len() == vector_count
    }
}

fn compact_q8_payload_byte_size(vector_count: usize, vector_width: usize) -> usize {
    vector_count
        .saturating_mul(vector_width)
        .saturating_mul(std::mem::size_of::<i8>())
        .saturating_add(vector_count.saturating_mul(std::mem::size_of::<f32>()))
}

fn dequantize_compact_q8(output: &mut [u16], tensor: &CompactQ8Tensor, vector_width: usize) {
    debug_assert!(tensor.has_layout(output.len() / vector_width, vector_width));
    for ((output_vector, quantized_vector), &scale) in output
        .chunks_exact_mut(vector_width)
        .zip(tensor.values.chunks_exact(vector_width))
        .zip(&tensor.scales)
    {
        for (output, &quantized) in output_vector.iter_mut().zip(quantized_vector) {
            *output = half::f16::from_f32(f32::from(quantized) * scale).to_bits();
        }
    }
}

#[derive(Clone)]
struct CompactLayerCache {
    key: CompactQ8Tensor,
    value: CompactQ8Tensor,
    num_kv_heads: usize,
    head_dim: usize,
}

impl CompactKVCacheSnapshot {
    pub(crate) fn byte_size(&self) -> u64 {
        let structural_bytes = std::mem::size_of::<Self>()
            .saturating_add(
                self.layers
                    .capacity()
                    .saturating_mul(std::mem::size_of::<CompactLayerCache>()),
            )
            .saturating_add(
                self.ssm_states
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Option<SsmLayerState>>()),
            );
        let kv_bytes = self
            .layers
            .iter()
            .map(|layer| {
                layer
                    .key
                    .byte_size()
                    .saturating_add(layer.value.byte_size())
            })
            .sum::<usize>();
        let ssm_bytes = self
            .ssm_states
            .iter()
            .flatten()
            .map(|state| {
                state
                    .conv_state
                    .capacity()
                    .saturating_add(state.delta_state.capacity())
                    .saturating_mul(std::mem::size_of::<f32>())
            })
            .sum::<usize>();
        structural_bytes
            .saturating_add(kv_bytes)
            .saturating_add(ssm_bytes) as u64
    }
}

impl KVCache {
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self::new_per_layer(
            max_seq_len,
            &(0..num_layers).map(|_| num_kv_heads).collect::<Vec<_>>(),
            &(0..num_layers).map(|_| head_dim).collect::<Vec<_>>(),
        )
    }

    pub fn new_per_layer(
        max_seq_len: usize,
        layer_num_kv_heads: &[usize],
        layer_head_dims: &[usize],
    ) -> Self {
        assert_eq!(
            layer_num_kv_heads.len(),
            layer_head_dims.len(),
            "KV cache layer config length mismatch"
        );
        let num_layers = layer_head_dims.len();
        let layers = (0..num_layers)
            .map(|i| LayerCache::new(max_seq_len, layer_num_kv_heads[i], layer_head_dims[i]))
            .collect();
        let ssm_states: Vec<Option<SsmLayerState>> = (0..num_layers).map(|_| None).collect();
        Self {
            layers,
            ssm_states,
            max_seq_len,
            current_len: 0,
        }
    }

    /// SSM state 초기화 (GDN 레이어에 대해 호출)
    pub fn init_ssm_state(
        &mut self,
        layer_idx: usize,
        conv_kernel: usize,
        conv_channels: usize,
        num_heads: usize,
        head_v_dim: usize,
        head_k_dim: usize,
    ) {
        if layer_idx < self.ssm_states.len() {
            self.ssm_states[layer_idx] = Some(SsmLayerState::new(
                conv_kernel,
                conv_channels,
                num_heads,
                head_v_dim,
                head_k_dim,
            ));
        }
    }

    pub fn get_ssm_state(&self, layer_idx: usize) -> Option<&SsmLayerState> {
        self.ssm_states.get(layer_idx).and_then(|s| s.as_ref())
    }

    pub fn get_ssm_state_mut(&mut self, layer_idx: usize) -> Option<&mut SsmLayerState> {
        self.ssm_states.get_mut(layer_idx).and_then(|s| s.as_mut())
    }

    pub fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        self.layers[layer].write_at(pos, k, v);
        if layer == self.layers.len() - 1 {
            self.current_len = self.current_len.max(pos + 1);
        }
    }

    /// Batch append for prefill — caller provides pre-converted f16 bits.
    /// Equivalent to per-token `append()` for `seq_len` rows starting at `pos_start`,
    /// but avoids the per-token f32→f16 conversion loop and per-token bounds checks.
    pub fn append_bits_range(
        &mut self,
        layer: usize,
        pos_start: usize,
        seq_len: usize,
        k_bits: &[u16],
        v_bits: &[u16],
    ) {
        self.layers[layer].write_bits_range(pos_start, seq_len, k_bits, v_bits);
        if layer == self.layers.len() - 1 {
            self.current_len = self.current_len.max(pos_start + seq_len);
        }
    }

    pub fn get(&self, layer: usize) -> (&[u16], &[u16]) {
        self.layers[layer].read_up_to(self.current_len)
    }

    /// 지정된 길이까지 K/V를 반환. forward 중 레이어별로 다른 len이 필요할 때 사용.
    pub fn get_up_to(&self, layer: usize, len: usize) -> (&[u16], &[u16]) {
        self.layers[layer].read_up_to(len)
    }

    pub fn replace_layer_f16(&mut self, layer: usize, len: usize, k_bits: &[u16], v_bits: &[u16]) {
        self.layers[layer].write_bits_up_to(len, k_bits, v_bits);
        trace_kv_bits("replace", layer, 0, len, k_bits, v_bits);
    }

    pub fn replace_layer_f16_range(
        &mut self,
        layer: usize,
        pos_start: usize,
        kv_len: usize,
        k_bits: &[u16],
        v_bits: &[u16],
    ) {
        self.layers[layer].write_bits_range(pos_start, kv_len, k_bits, v_bits);
        trace_kv_bits("replace_range", layer, pos_start, kv_len, k_bits, v_bits);
    }

    pub fn get_key(&self, layer: usize) -> &[u16] {
        self.get(layer).0
    }

    pub fn get_value(&self, layer: usize) -> &[u16] {
        self.get(layer).1
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub(crate) fn compact_snapshot_byte_size(&self) -> u64 {
        let structural_bytes = std::mem::size_of::<CompactKVCacheSnapshot>()
            .saturating_add(
                self.layers
                    .len()
                    .saturating_mul(std::mem::size_of::<CompactLayerCache>()),
            )
            .saturating_add(
                self.ssm_states
                    .len()
                    .saturating_mul(std::mem::size_of::<Option<SsmLayerState>>()),
            );
        let kv_bytes = self
            .layers
            .iter()
            .map(|layer| {
                let vector_count = self.current_len.saturating_mul(layer.num_kv_heads);
                compact_q8_payload_byte_size(vector_count, layer.head_dim).saturating_mul(2)
            })
            .sum::<usize>();
        let ssm_bytes = self
            .ssm_states
            .iter()
            .flatten()
            .map(|state| {
                state
                    .conv_state
                    .len()
                    .saturating_add(state.delta_state.len())
                    .saturating_mul(std::mem::size_of::<f32>())
            })
            .sum::<usize>();
        structural_bytes
            .saturating_add(kv_bytes)
            .saturating_add(ssm_bytes) as u64
    }

    /// current_len을 명시적으로 설정. forward 완료 후 호출.
    pub fn set_len(&mut self, len: usize) {
        self.current_len = len;
    }

    pub(crate) fn compact_snapshot(&self) -> CompactKVCacheSnapshot {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let (key, value) = layer.read_up_to(self.current_len);
                CompactLayerCache {
                    key: CompactQ8Tensor::from_f16(key, layer.head_dim),
                    value: CompactQ8Tensor::from_f16(value, layer.head_dim),
                    num_kv_heads: layer.num_kv_heads,
                    head_dim: layer.head_dim,
                }
            })
            .collect();
        CompactKVCacheSnapshot {
            layers,
            ssm_states: self.ssm_states.clone(),
            max_seq_len: self.max_seq_len,
            current_len: self.current_len,
        }
    }

    pub(crate) fn restore_compact(
        &mut self,
        snapshot: &CompactKVCacheSnapshot,
    ) -> Result<(), String> {
        if snapshot.current_len > self.max_seq_len
            || snapshot.max_seq_len != self.max_seq_len
            || snapshot.layers.len() != self.layers.len()
            || snapshot.ssm_states.len() != self.ssm_states.len()
        {
            return Err("sequence state does not match this KV cache".to_string());
        }
        for (current, saved) in self.layers.iter().zip(&snapshot.layers) {
            let vector_count = snapshot.current_len.saturating_mul(current.num_kv_heads);
            if current.num_kv_heads != saved.num_kv_heads
                || current.head_dim != saved.head_dim
                || !saved.key.has_layout(vector_count, current.head_dim)
                || !saved.value.has_layout(vector_count, current.head_dim)
            {
                return Err("sequence state layer layout does not match this KV cache".to_string());
            }
        }

        for (current, saved) in self.layers.iter_mut().zip(&snapshot.layers) {
            current.write_q8_up_to(snapshot.current_len, &saved.key, &saved.value);
        }
        restore_ssm_states(&mut self.ssm_states, &snapshot.ssm_states);
        self.current_len = snapshot.current_len;
        Ok(())
    }

    pub fn clear(&mut self) {
        for ssm in &mut self.ssm_states {
            if let Some(s) = ssm {
                s.clear();
            }
        }
        self.current_len = 0;
    }

    /// Layer 수.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Layer 의 KV-head dim 합 (`head_dim * num_kv_heads`).
    pub fn layer_kv_dim(&self, layer_idx: usize) -> usize {
        let layer = &self.layers[layer_idx];
        layer.num_kv_heads * layer.head_dim
    }

    /// Exposes the current contiguous K/V storage as a single logical bucket.
    ///
    /// `logical_len` is supplied by the caller and must describe rows already
    /// written for this layer; this view validates capacity, not initialization.
    pub fn layer_bucket_view(
        &self,
        layer_idx: usize,
        logical_len: usize,
    ) -> crate::runtime::BackendResult<crate::runtime::KvBucketView> {
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            kv_bucket_view_error(format!(
                "KV bucket layer index {layer_idx} out of range for {} layers",
                self.layers.len()
            ))
        })?;
        let kv_row_width = layer
            .num_kv_heads
            .checked_mul(layer.head_dim)
            .ok_or_else(|| kv_bucket_view_error("KV bucket row width overflow"))?;

        crate::runtime::KvBucketView::new(
            layer_idx,
            self.max_seq_len,
            logical_len,
            self.max_seq_len,
            kv_row_width,
            layer.key.as_ptr() as u64,
            layer.value.as_ptr() as u64,
        )
    }

    /// Layer 의 K cache 를 `[current_len, kv_dim]` f32 row-major 로 dequant.
    /// F16 (u16 bits) 저장이라 호출마다 dequant. Cross-attention drafter 용 read-only 접근.
    pub fn dequant_k_layer(&self, layer_idx: usize) -> Vec<f32> {
        let stride = self.layer_kv_dim(layer_idx);
        let count = self.current_len * stride;
        let bits = &self.layers[layer_idx].key[..count];
        bits.iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect()
    }

    /// Layer 의 V cache 를 `[current_len, kv_dim]` f32 row-major 로 dequant.
    pub fn dequant_v_layer(&self, layer_idx: usize) -> Vec<f32> {
        let stride = self.layer_kv_dim(layer_idx);
        let count = self.current_len * stride;
        let bits = &self.layers[layer_idx].value[..count];
        bits.iter()
            .map(|&b| half::f16::from_bits(b).to_f32())
            .collect()
    }
}

fn restore_ssm_states(current: &mut [Option<SsmLayerState>], saved: &[Option<SsmLayerState>]) {
    for (current, saved) in current.iter_mut().zip(saved) {
        match (current.as_mut(), saved.as_ref()) {
            (Some(current), Some(saved))
                if current.conv_state.len() == saved.conv_state.len()
                    && current.delta_state.len() == saved.delta_state.len() =>
            {
                current.conv_state.copy_from_slice(&saved.conv_state);
                current.delta_state.copy_from_slice(&saved.delta_state);
                current.conv_kernel = saved.conv_kernel;
                current.conv_channels = saved.conv_channels;
            }
            (_, Some(saved)) => *current = Some(saved.clone()),
            (_, None) => *current = None,
        }
    }
}

fn kv_bucket_view_error(message: impl Into<String>) -> crate::runtime::BackendError {
    crate::runtime::BackendError::new(
        crate::runtime::BackendErrorKind::InvalidRequest,
        crate::runtime::BackendKind::Cpu,
        Some(crate::runtime::BackendOp::Attention),
        message,
    )
}

impl crate::KvBorrow for KVCache {
    fn k_layer(&self, target_layer_idx: usize) -> Vec<f32> {
        self.dequant_k_layer(target_layer_idx)
    }

    fn v_layer(&self, target_layer_idx: usize) -> Vec<f32> {
        self.dequant_v_layer(target_layer_idx)
    }

    fn pos(&self) -> usize {
        self.current_len
    }

    fn kv_dim_for_layer(&self, target_layer_idx: usize) -> usize {
        self.layer_kv_dim(target_layer_idx)
    }

    fn n_layers(&self) -> usize {
        self.num_layers()
    }
}

fn trace_kv_bits(
    op: &str,
    layer: usize,
    pos_start: usize,
    kv_len: usize,
    k_bits: &[u16],
    v_bits: &[u16],
) {
    if std::env::var_os("RNB_DEBUG_KV_HASH_TRACE").is_none() {
        return;
    }
    if let Some(filter) = std::env::var("RNB_DEBUG_KV_HASH_LAYER").ok() {
        if !layer_matches_spec(&filter, layer) {
            return;
        }
    }
    eprintln!(
        "[kv-hash] op={} layer={} pos_start={} kv_len={} k_len={} v_len={} k_hash=0x{:016x} v_hash=0x{:016x} k_first={} k_last={} v_first={} v_last={}",
        op,
        layer,
        pos_start,
        kv_len,
        k_bits.len(),
        v_bits.len(),
        hash_u16(k_bits),
        hash_u16(v_bits),
        k_bits.first().copied().unwrap_or(0),
        k_bits.last().copied().unwrap_or(0),
        v_bits.first().copied().unwrap_or(0),
        v_bits.last().copied().unwrap_or(0)
    );
}

fn hash_u16(values: &[u16]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for &value in values {
        hash ^= value as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn layer_matches_spec(raw: &str, layer_idx: usize) -> bool {
    for term in raw
        .split(',')
        .map(str::trim)
        .filter(|term| !term.is_empty())
    {
        if let Some((start, end)) = term.split_once('-') {
            if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                if start <= layer_idx && layer_idx <= end {
                    return true;
                }
            }
        } else if term.parse::<usize>().is_ok_and(|want| want == layer_idx) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    fn assert_f16_approx(actual: &[u16], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            let actual = f16_to_f32(actual);
            assert!(
                (actual - expected).abs() <= tolerance,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(2, 128, 4, 64);
        assert_eq!(cache.current_len(), 0);
        assert_eq!(cache.max_seq_len, 128);
    }

    #[test]
    fn test_append_and_get() {
        let num_layers = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let mut cache = KVCache::new(num_layers, 16, num_kv_heads, head_dim);

        let k0 = vec![1.0f32; num_kv_heads * head_dim];
        let v0 = vec![2.0f32; num_kv_heads * head_dim];

        cache.append(0, 0, &k0, &v0);
        cache.append(1, 0, &k0, &v0);

        assert_eq!(cache.current_len(), 1);

        let (k, v) = cache.get(0);
        assert_eq!(k.len(), num_kv_heads * head_dim);
        // F16 round-trip: 1.0 and 2.0 are exactly representable
        assert!(k.iter().all(|&x| (f16_to_f32(x) - 1.0).abs() < 1e-3));
        assert!(v.iter().all(|&x| (f16_to_f32(x) - 2.0).abs() < 1e-3));
    }

    #[test]
    fn test_append_multiple_positions() {
        let mut cache = KVCache::new(1, 16, 1, 2);
        for pos in 0..4 {
            let k = vec![pos as f32; 2];
            let v = vec![(pos as f32) * 10.0; 2];
            cache.append(0, pos, &k, &v);
        }
        assert_eq!(cache.current_len(), 4);

        let (k, _v) = cache.get(0);
        // F16 round-trip: integers 0-3 and multiples of 10 are exactly representable
        assert!((f16_to_f32(k[0]) - 0.0).abs() < 1e-3);
        assert!((f16_to_f32(k[2]) - 1.0).abs() < 1e-3);
        assert!((f16_to_f32(k[4]) - 2.0).abs() < 1e-3);
        assert!((f16_to_f32(k[6]) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_clear() {
        let mut cache = KVCache::new(1, 16, 1, 2);
        cache.append(0, 0, &[1.0, 1.0], &[1.0, 1.0]);
        assert_eq!(cache.current_len(), 1);

        cache.clear();
        assert_eq!(cache.current_len(), 0);
        let (k, _) = cache.get(0);
        assert_eq!(k.len(), 0);
    }
    #[test]
    fn compact_snapshot_restores_only_logical_rows_and_ssm_state() {
        let mut cache = KVCache::new(1, 16, 1, 2);
        cache.append(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.append(0, 1, &[5.0, 6.0], &[7.0, 8.0]);
        cache.ssm_states[0] = Some(SsmLayerState {
            conv_state: vec![9.0, 10.0],
            delta_state: vec![11.0],
            conv_kernel: 2,
            conv_channels: 1,
        });
        let estimated_bytes = cache.compact_snapshot_byte_size();
        let snapshot = cache.compact_snapshot();
        assert!(snapshot.byte_size() > 28);
        assert_eq!(snapshot.byte_size(), estimated_bytes);

        cache.append(0, 2, &[12.0, 13.0], &[14.0, 15.0]);
        cache.ssm_states[0].as_mut().unwrap().conv_state.fill(0.0);
        cache.restore_compact(&snapshot).unwrap();

        assert_eq!(cache.current_len(), 2);
        let (key, value) = cache.get(0);
        assert_f16_approx(key, &[1.0, 2.0, 5.0, 6.0], 0.05);
        assert_f16_approx(value, &[3.0, 4.0, 7.0, 8.0], 0.05);
        let ssm = cache.ssm_states[0].as_ref().unwrap();
        assert_eq!(ssm.conv_state, vec![9.0, 10.0]);
        assert_eq!(ssm.delta_state, vec![11.0]);
    }

    #[test]
    fn compact_q8_snapshot_halves_kv_payload_and_validates_layout() {
        let num_kv_heads = 2;
        let head_dim = 64;
        let token_count = 4;
        let stride = num_kv_heads * head_dim;
        let mut cache = KVCache::new(1, 16, num_kv_heads, head_dim);
        for pos in 0..token_count {
            let key = (0..stride)
                .map(|index| ((pos * stride + index) % 31) as f32 / 7.0 - 2.0)
                .collect::<Vec<_>>();
            let value = (0..stride)
                .map(|index| ((pos * stride + index) % 29) as f32 / 9.0 - 1.5)
                .collect::<Vec<_>>();
            cache.append(0, pos, &key, &value);
        }
        let (key, value) = cache.get(0);
        let expected_key = key.iter().map(|&bits| f16_to_f32(bits)).collect::<Vec<_>>();
        let expected_value = value
            .iter()
            .map(|&bits| f16_to_f32(bits))
            .collect::<Vec<_>>();

        let estimated_bytes = cache.compact_snapshot_byte_size();
        let snapshot = cache.compact_snapshot();
        let q8_payload = snapshot.layers[0].key.byte_size() + snapshot.layers[0].value.byte_size();
        let f16_payload =
            (expected_key.len() + expected_value.len()).saturating_mul(std::mem::size_of::<u16>());
        assert_eq!(snapshot.byte_size(), estimated_bytes);
        assert!(q8_payload.saturating_mul(100) <= f16_payload.saturating_mul(55));

        let zeros = vec![0.0; stride];
        for pos in 0..token_count {
            cache.append(0, pos, &zeros, &zeros);
        }
        cache.restore_compact(&snapshot).unwrap();
        let (restored_key, restored_value) = cache.get(0);
        assert_f16_approx(restored_key, &expected_key, 0.02);
        assert_f16_approx(restored_value, &expected_value, 0.02);

        let mut invalid = snapshot.clone();
        invalid.layers[0].key.scales.pop();
        assert!(cache.restore_compact(&invalid).is_err());
    }

    #[test]
    fn test_replace_layer_f16_range_preserves_existing_prefix() {
        let mut cache = KVCache::new(1, 16, 1, 2);
        cache.append(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.append(0, 1, &[5.0, 6.0], &[7.0, 8.0]);
        cache.set_len(2);

        let new_k = [
            half::f16::from_f32(9.0).to_bits(),
            half::f16::from_f32(10.0).to_bits(),
        ];
        let new_v = [
            half::f16::from_f32(11.0).to_bits(),
            half::f16::from_f32(12.0).to_bits(),
        ];

        cache.replace_layer_f16_range(0, 1, 1, &new_k, &new_v);

        let (k, v) = cache.get_up_to(0, 2);
        assert!((f16_to_f32(k[0]) - 1.0).abs() < 1e-3);
        assert!((f16_to_f32(k[1]) - 2.0).abs() < 1e-3);
        assert!((f16_to_f32(v[0]) - 3.0).abs() < 1e-3);
        assert!((f16_to_f32(v[1]) - 4.0).abs() < 1e-3);
        assert!((f16_to_f32(k[2]) - 9.0).abs() < 1e-3);
        assert!((f16_to_f32(k[3]) - 10.0).abs() < 1e-3);
        assert!((f16_to_f32(v[2]) - 11.0).abs() < 1e-3);
        assert!((f16_to_f32(v[3]) - 12.0).abs() < 1e-3);
    }

    #[test]
    fn kv_bucket_view_identity_survives_append_within_capacity() {
        let mut cache = KVCache::new(1, 16, 1, 2);
        let before = cache.layer_bucket_view(0, 0).expect("initial bucket");
        assert_eq!(before.page_size(), 16);
        assert_eq!(before.max_len(), 16);

        cache.append(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.append(0, 1, &[5.0, 6.0], &[7.0, 8.0]);

        let after = cache.layer_bucket_view(0, 2).expect("updated bucket");
        assert_eq!(before.k_identity(), after.k_identity());
        assert_eq!(before.v_identity(), after.v_identity());
        assert_eq!(after.current_len(), 2);
        assert_eq!(after.last_page_len(), 2);
        assert_eq!(after.page_count(), 1);
    }

    #[test]
    fn kv_bucket_view_rejects_invalid_layer_or_length() {
        let cache = KVCache::new(1, 16, 1, 2);

        assert!(cache.layer_bucket_view(1, 0).is_err());
        assert!(cache.layer_bucket_view(0, 17).is_err());
    }

    #[test]
    fn kv_bucket_view_reports_layer_specific_rows() {
        let cache = KVCache::new_per_layer(16, &[1, 2], &[4, 8]);

        let layer0 = cache.layer_bucket_view(0, 0).expect("layer0 bucket");
        let layer1 = cache.layer_bucket_view(1, 0).expect("layer1 bucket");

        assert_eq!(layer0.kv_row_width(), 4);
        assert_eq!(layer1.kv_row_width(), 16);
        assert_ne!(layer0.k_identity(), layer1.k_identity());
        assert_ne!(layer0.v_identity(), layer1.v_identity());
    }
}
