mod kvarn;

use crate::engine::cpu_runtime::quantize::kvarn::KvarnKvView;
pub use kvarn::KvCacheFormat;
use kvarn::KvarnLayerCache;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_KV_CACHE_SEQUENCE_EPOCH: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
enum LayerCacheStorage {
    F16 { key: Vec<u16>, value: Vec<u16> },
    Kvarn(KvarnLayerCache),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvCacheMetrics {
    pub current_tokens: usize,
    pub compressed_layers: usize,
    pub quantized_token_rows: usize,
    pub allocated_bytes: u64,
    pub capacity_bytes: u64,
    pub f16_equivalent_capacity_bytes: u64,
    pub key_quant_snr_db: Option<f32>,
    pub value_quant_snr_db: Option<f32>,
}

/// 단일 레이어의 K/V 텐서 캐시
#[derive(Clone)]
pub struct LayerCache {
    storage: LayerCacheStorage,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

pub(crate) enum LayerCacheRead<'a> {
    Borrowed { key: &'a [u16], value: &'a [u16] },
    Materialized { key: Vec<u16>, value: Vec<u16> },
}

impl LayerCacheRead<'_> {
    pub(crate) fn as_slices(&self) -> (&[u16], &[u16]) {
        match self {
            Self::Borrowed { key, value } => (key, value),
            Self::Materialized { key, value } => (key, value),
        }
    }
}

impl LayerCache {
    fn new(
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        format: KvCacheFormat,
    ) -> Result<Self, String> {
        let storage = if let Some(config) = format.kvarn_config() {
            LayerCacheStorage::Kvarn(KvarnLayerCache::new(
                config,
                max_seq_len,
                num_kv_heads,
                head_dim,
            )?)
        } else {
            let size = max_seq_len * num_kv_heads * head_dim;
            LayerCacheStorage::F16 {
                key: vec![0u16; size],
                value: vec![0u16; size],
            }
        };
        Ok(Self {
            storage,
            num_kv_heads,
            head_dim,
            max_seq_len,
        })
    }

    fn write_at(&mut self, pos: usize, k_slice: &[f32], v_slice: &[f32]) {
        assert!(pos < self.max_seq_len, "KV cache overflow");
        let stride = self.num_kv_heads * self.head_dim;
        match &mut self.storage {
            LayerCacheStorage::F16 { key, value } => {
                let start = pos * stride;
                for i in 0..stride {
                    key[start + i] = half::f16::from_f32(k_slice[i]).to_bits();
                    value[start + i] = half::f16::from_f32(v_slice[i]).to_bits();
                }
            }
            LayerCacheStorage::Kvarn(cache) => cache.write_f32(pos, k_slice, v_slice),
        }
    }

    fn read_up_to(&self, len: usize) -> (&[u16], &[u16]) {
        let stride = self.num_kv_heads * self.head_dim;
        match &self.storage {
            LayerCacheStorage::F16 { key, value } => (&key[..len * stride], &value[..len * stride]),
            LayerCacheStorage::Kvarn(_) => {
                panic!("contiguous borrowed K/V is unavailable for a KVarN cache")
            }
        }
    }

    fn read_up_to_materialized(&self, len: usize) -> LayerCacheRead<'_> {
        match &self.storage {
            LayerCacheStorage::F16 { .. } => {
                let (key, value) = self.read_up_to(len);
                LayerCacheRead::Borrowed { key, value }
            }
            LayerCacheStorage::Kvarn(cache) => {
                let (key, value) = cache.materialize(len);
                LayerCacheRead::Materialized { key, value }
            }
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn read_up_to_materialized_if_initialized(
        &self,
        len: usize,
    ) -> Option<LayerCacheRead<'_>> {
        match &self.storage {
            LayerCacheStorage::Kvarn(cache) if cache.stored_len() < len => None,
            _ => Some(self.read_up_to_materialized(len)),
        }
    }

    #[cfg(feature = "cuda")]
    fn bits_up_to_mut(&mut self, len: usize) -> (&mut [u16], &mut [u16]) {
        let count = len * self.num_kv_heads * self.head_dim;
        match &mut self.storage {
            LayerCacheStorage::F16 { key, value } => (&mut key[..count], &mut value[..count]),
            LayerCacheStorage::Kvarn(_) => {
                panic!("mutable contiguous K/V is unavailable for a KVarN cache")
            }
        }
    }

    fn write_bits_up_to(&mut self, len: usize, k_bits: &[u16], v_bits: &[u16]) {
        let stride = self.num_kv_heads * self.head_dim;
        let count = len * stride;
        assert!(
            k_bits.len() >= count && v_bits.len() >= count,
            "KV bits underflow"
        );
        match &mut self.storage {
            LayerCacheStorage::F16 { key, value } => {
                key[..count].copy_from_slice(&k_bits[..count]);
                value[..count].copy_from_slice(&v_bits[..count]);
            }
            LayerCacheStorage::Kvarn(cache) => {
                cache.write_bits_up_to(len, &k_bits[..count], &v_bits[..count]);
            }
        }
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
            k_bits.len() >= count && v_bits.len() >= count,
            "KV bits range underflow"
        );
        match &mut self.storage {
            LayerCacheStorage::F16 { key, value } => {
                assert!(
                    end <= key.len() && end <= value.len(),
                    "KV bits range overflow"
                );
                key[start..end].copy_from_slice(&k_bits[..count]);
                value[start..end].copy_from_slice(&v_bits[..count]);
            }
            LayerCacheStorage::Kvarn(cache) => {
                cache.write_bits_range(pos_start, kv_len, &k_bits[..count], &v_bits[..count]);
            }
        }
    }

    fn compact(&mut self) -> Result<(), String> {
        match &mut self.storage {
            LayerCacheStorage::F16 { .. } => Ok(()),
            LayerCacheStorage::Kvarn(cache) => cache.compact(),
        }
    }

    fn truncate_to(&mut self, len: usize) {
        if let LayerCacheStorage::Kvarn(cache) = &mut self.storage {
            cache.truncate_to(len);
        }
    }

    fn is_kvarn(&self) -> bool {
        matches!(self.storage, LayerCacheStorage::Kvarn(_))
    }

    pub(crate) fn kvarn_view(&self, len: usize) -> Option<KvarnKvView<'_>> {
        match &self.storage {
            LayerCacheStorage::F16 { .. } => None,
            LayerCacheStorage::Kvarn(cache) => Some(cache.view(len)),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn kvarn_view_if_initialized(&self, len: usize) -> Option<KvarnKvView<'_>> {
        match &self.storage {
            LayerCacheStorage::F16 { .. } => None,
            LayerCacheStorage::Kvarn(cache) if cache.is_initialized_up_to(len) => {
                Some(cache.view(len))
            }
            LayerCacheStorage::Kvarn(_) => None,
        }
    }

    fn actual_byte_size(&self) -> usize {
        match &self.storage {
            LayerCacheStorage::F16 { key, value } => key
                .capacity()
                .saturating_add(value.capacity())
                .saturating_mul(std::mem::size_of::<u16>()),
            LayerCacheStorage::Kvarn(cache) => cache.actual_byte_size(),
        }
    }

    fn capacity_byte_size(&self) -> usize {
        match &self.storage {
            LayerCacheStorage::F16 { key, value } => key
                .len()
                .saturating_add(value.len())
                .saturating_mul(std::mem::size_of::<u16>()),
            LayerCacheStorage::Kvarn(cache) => cache.capacity_byte_size(),
        }
    }

    fn f16_capacity_byte_size(&self) -> usize {
        self.max_seq_len
            .saturating_mul(self.num_kv_heads)
            .saturating_mul(self.head_dim)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u16>())
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
    sequence_epoch: u64,
    /// pm119: GLM DSA lightning indexer key 캐시 — 층당 `max_seq_len × key_len`
    /// f16 bits. 빈 Vec = 미사용 (`init_glm_indexer` 로 활성화). 유효 구간은
    /// `current_len` 을 메인 KV 와 공유하므로 checkpoint/restore(len 되감기)
    /// 에 자동 정합.
    glm_indexer_k: Vec<Vec<u16>>,
    glm_indexer_key_len: usize,
    glm_indexer_top_k: usize,
}
#[derive(Clone)]
pub(crate) struct KVCacheSnapshot {
    layers: Vec<LayerCacheSnapshot>,
    ssm_states: Vec<Option<SsmLayerState>>,
    max_seq_len: usize,
    current_len: usize,
    glm_indexer_k: Vec<Vec<u16>>,
    glm_indexer_key_len: usize,
    glm_indexer_top_k: usize,
}

#[derive(Clone)]
enum LayerCacheSnapshotStorage {
    F16 { key: Vec<u16>, value: Vec<u16> },
    Kvarn(KvarnLayerCache),
}

#[derive(Clone)]
struct LayerCacheSnapshot {
    storage: LayerCacheSnapshotStorage,
    num_kv_heads: usize,
    head_dim: usize,
}

impl LayerCacheSnapshot {
    fn byte_size(&self) -> usize {
        match &self.storage {
            LayerCacheSnapshotStorage::F16 { key, value } => key
                .capacity()
                .saturating_add(value.capacity())
                .saturating_mul(std::mem::size_of::<u16>()),
            LayerCacheSnapshotStorage::Kvarn(cache) => cache.actual_byte_size(),
        }
    }
}

impl KVCacheSnapshot {
    pub(crate) fn byte_size(&self) -> u64 {
        let structural_bytes = std::mem::size_of::<Self>()
            .saturating_add(
                self.layers
                    .capacity()
                    .saturating_mul(std::mem::size_of::<LayerCacheSnapshot>()),
            )
            .saturating_add(
                self.ssm_states
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Option<SsmLayerState>>()),
            )
            .saturating_add(
                self.glm_indexer_k
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Vec<u16>>()),
            );
        let kv_bytes = self
            .layers
            .iter()
            .map(LayerCacheSnapshot::byte_size)
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
        let glm_indexer_bytes = self
            .glm_indexer_k
            .iter()
            .map(|keys| keys.capacity().saturating_mul(std::mem::size_of::<u16>()))
            .sum::<usize>();
        structural_bytes
            .saturating_add(kv_bytes)
            .saturating_add(ssm_bytes)
            .saturating_add(glm_indexer_bytes) as u64
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
        let formats = vec![KvCacheFormat::F16; layer_head_dims.len()];
        Self::new_per_layer_with_formats(max_seq_len, layer_num_kv_heads, layer_head_dims, &formats)
            .expect("F16 KV-cache layout is valid")
    }

    pub fn new_per_layer_with_format(
        max_seq_len: usize,
        layer_num_kv_heads: &[usize],
        layer_head_dims: &[usize],
        format: KvCacheFormat,
    ) -> Result<Self, String> {
        let formats = vec![format; layer_head_dims.len()];
        Self::new_per_layer_with_formats(max_seq_len, layer_num_kv_heads, layer_head_dims, &formats)
    }

    pub fn new_per_layer_with_formats(
        max_seq_len: usize,
        layer_num_kv_heads: &[usize],
        layer_head_dims: &[usize],
        layer_formats: &[KvCacheFormat],
    ) -> Result<Self, String> {
        if layer_num_kv_heads.len() != layer_head_dims.len()
            || layer_formats.len() != layer_head_dims.len()
        {
            return Err("KV cache layer config length mismatch".to_string());
        }
        let num_layers = layer_head_dims.len();
        let layers = (0..num_layers)
            .map(|i| {
                LayerCache::new(
                    max_seq_len,
                    layer_num_kv_heads[i],
                    layer_head_dims[i],
                    layer_formats[i],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ssm_states: Vec<Option<SsmLayerState>> = (0..num_layers).map(|_| None).collect();
        Ok(Self {
            layers,
            ssm_states,
            max_seq_len,
            current_len: 0,
            sequence_epoch: NEXT_KV_CACHE_SEQUENCE_EPOCH.fetch_add(1, Ordering::Relaxed),
            glm_indexer_k: Vec::new(),
            glm_indexer_key_len: 0,
            glm_indexer_top_k: 0,
        })
    }

    /// pm119: GLM DSA indexer key 캐시 활성화 (엔진 init 에서 indexer weight
    /// 로드 확인 후 호출). `top_k` 는 attend 길이가 이를 넘을 때 selected-set
    /// attention 이 발동하는 경계 (GLM-5.2 = 2048).
    pub fn init_glm_indexer(&mut self, num_layers: usize, key_len: usize, top_k: usize) {
        self.glm_indexer_key_len = key_len;
        self.glm_indexer_top_k = top_k;
        self.glm_indexer_k = (0..num_layers)
            .map(|_| vec![0u16; self.max_seq_len * key_len])
            .collect();
    }

    pub fn glm_indexer_top_k(&self) -> usize {
        self.glm_indexer_top_k
    }

    pub fn glm_indexer_enabled(&self) -> bool {
        !self.glm_indexer_k.is_empty()
    }

    /// indexer key (LayerNorm+rope 적용 완료분, f32) 를 f16 으로 기록.
    pub fn write_glm_indexer_k(&mut self, layer: usize, pos: usize, k: &[f32]) {
        let key_len = self.glm_indexer_key_len;
        debug_assert_eq!(k.len(), key_len);
        assert!(pos < self.max_seq_len, "GLM indexer cache overflow");
        let row = &mut self.glm_indexer_k[layer][pos * key_len..(pos + 1) * key_len];
        for (dst, &v) in row.iter_mut().zip(k) {
            *dst = half::f16::from_f32(v).to_bits();
        }
    }

    /// `[0, len)` 구간의 indexer key (f16 bits, row-major `len × key_len`).
    pub fn glm_indexer_k_up_to(&self, layer: usize, len: usize) -> &[u16] {
        &self.glm_indexer_k[layer][..len * self.glm_indexer_key_len]
    }

    pub fn glm_indexer_key_len(&self) -> usize {
        self.glm_indexer_key_len
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
    pub(crate) fn read_up_to(&self, layer: usize, len: usize) -> LayerCacheRead<'_> {
        self.layers[layer].read_up_to_materialized(len)
    }

    #[cfg(test)]
    pub(crate) fn read_up_to_if_initialized(
        &self,
        layer: usize,
        len: usize,
    ) -> Option<LayerCacheRead<'_>> {
        self.layers[layer].read_up_to_materialized_if_initialized(len)
    }

    pub(crate) fn compact_layer(&mut self, layer: usize) -> Result<(), String> {
        self.layers[layer].compact()
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn layers(&self) -> &[LayerCache] {
        &self.layers
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn layer_and_ssm_parts_mut(
        &mut self,
    ) -> (&[LayerCache], &mut [Option<SsmLayerState>]) {
        (&self.layers, &mut self.ssm_states)
    }

    pub(crate) fn kvarn_view(&self, layer: usize, len: usize) -> Option<KvarnKvView<'_>> {
        self.layers[layer].kvarn_view(len)
    }

    pub fn layer_uses_kvarn(&self, layer: usize) -> bool {
        self.layers[layer].is_kvarn()
    }

    pub fn allocated_kv_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(LayerCache::actual_byte_size)
            .sum::<usize>() as u64
    }

    pub fn capacity_kv_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(LayerCache::capacity_byte_size)
            .sum::<usize>() as u64
    }

    pub fn metrics(&self) -> KvCacheMetrics {
        let mut compressed_layers = 0;
        let mut quantized_token_rows = 0;
        let mut key_signal = 0.0;
        let mut key_error = 0.0;
        let mut value_signal = 0.0;
        let mut value_error = 0.0;
        for layer in &self.layers {
            if let LayerCacheStorage::Kvarn(cache) = &layer.storage {
                compressed_layers += 1;
                quantized_token_rows += cache.quantized_rows();
                let ((layer_key_signal, layer_key_error), (layer_value_signal, layer_value_error)) =
                    cache.quantization_energy();
                key_signal += layer_key_signal;
                key_error += layer_key_error;
                value_signal += layer_value_signal;
                value_error += layer_value_error;
            }
        }
        KvCacheMetrics {
            current_tokens: self.current_len,
            compressed_layers,
            quantized_token_rows,
            allocated_bytes: self.allocated_kv_bytes(),
            capacity_bytes: self.capacity_kv_bytes(),
            f16_equivalent_capacity_bytes: self
                .layers
                .iter()
                .map(LayerCache::f16_capacity_byte_size)
                .sum::<usize>() as u64,
            key_quant_snr_db: quantization_snr_db(key_signal, key_error),
            value_quant_snr_db: quantization_snr_db(value_signal, value_error),
        }
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn get_up_to_mut(&mut self, layer: usize, len: usize) -> (&mut [u16], &mut [u16]) {
        self.layers[layer].bits_up_to_mut(len)
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

    pub(crate) fn replace_layer_f16_range_compacted(
        &mut self,
        layer: usize,
        pos_start: usize,
        kv_len: usize,
        k_bits: &[u16],
        v_bits: &[u16],
    ) -> Result<(), String> {
        self.replace_layer_f16_range(layer, pos_start, kv_len, k_bits, v_bits);
        if self.layer_uses_kvarn(layer) {
            self.compact_layer(layer)?;
        }
        Ok(())
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

    pub(crate) fn snapshot_byte_size(&self) -> u64 {
        let structural_bytes = std::mem::size_of::<KVCacheSnapshot>()
            .saturating_add(
                self.layers
                    .len()
                    .saturating_mul(std::mem::size_of::<LayerCacheSnapshot>()),
            )
            .saturating_add(
                self.ssm_states
                    .len()
                    .saturating_mul(std::mem::size_of::<Option<SsmLayerState>>()),
            )
            .saturating_add(
                self.glm_indexer_k
                    .len()
                    .saturating_mul(std::mem::size_of::<Vec<u16>>()),
            );
        let kv_bytes = self
            .layers
            .iter()
            .map(|layer| match &layer.storage {
                LayerCacheStorage::F16 { .. } => self
                    .current_len
                    .saturating_mul(layer.num_kv_heads)
                    .saturating_mul(layer.head_dim)
                    .saturating_mul(std::mem::size_of::<u16>())
                    .saturating_mul(2),
                LayerCacheStorage::Kvarn(cache) => cache.actual_byte_size(),
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
        let glm_indexer_bytes = self
            .current_len
            .saturating_mul(self.glm_indexer_key_len)
            .saturating_mul(self.glm_indexer_k.len())
            .saturating_mul(std::mem::size_of::<u16>());
        structural_bytes
            .saturating_add(kv_bytes)
            .saturating_add(ssm_bytes)
            .saturating_add(glm_indexer_bytes) as u64
    }

    /// current_len을 명시적으로 설정. forward 완료 후 호출.
    pub fn set_len(&mut self, len: usize) {
        for layer in &mut self.layers {
            layer.truncate_to(len);
        }
        self.current_len = len;
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn sequence_epoch(&self) -> u64 {
        self.sequence_epoch
    }

    pub(crate) fn snapshot(&self) -> KVCacheSnapshot {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let storage = match &layer.storage {
                    LayerCacheStorage::F16 { .. } => {
                        let (key, value) = layer.read_up_to(self.current_len);
                        LayerCacheSnapshotStorage::F16 {
                            key: key.to_vec(),
                            value: value.to_vec(),
                        }
                    }
                    LayerCacheStorage::Kvarn(cache) => {
                        LayerCacheSnapshotStorage::Kvarn(cache.snapshot(self.current_len))
                    }
                };
                LayerCacheSnapshot {
                    storage,
                    num_kv_heads: layer.num_kv_heads,
                    head_dim: layer.head_dim,
                }
            })
            .collect();
        let glm_indexer_len = self.current_len.saturating_mul(self.glm_indexer_key_len);
        let glm_indexer_k = self
            .glm_indexer_k
            .iter()
            .map(|keys| keys[..glm_indexer_len].to_vec())
            .collect();
        KVCacheSnapshot {
            layers,
            ssm_states: self.ssm_states.clone(),
            max_seq_len: self.max_seq_len,
            current_len: self.current_len,
            glm_indexer_k,
            glm_indexer_key_len: self.glm_indexer_key_len,
            glm_indexer_top_k: self.glm_indexer_top_k,
        }
    }

    pub(crate) fn restore_snapshot(&mut self, snapshot: &KVCacheSnapshot) -> Result<(), String> {
        if snapshot.current_len > self.max_seq_len
            || snapshot.max_seq_len != self.max_seq_len
            || snapshot.layers.len() != self.layers.len()
            || snapshot.ssm_states.len() != self.ssm_states.len()
            || snapshot.glm_indexer_k.len() != self.glm_indexer_k.len()
            || snapshot.glm_indexer_key_len != self.glm_indexer_key_len
            || snapshot.glm_indexer_top_k != self.glm_indexer_top_k
        {
            return Err("sequence state does not match this KV cache".to_string());
        }
        for (current, saved) in self.layers.iter().zip(&snapshot.layers) {
            if current.num_kv_heads != saved.num_kv_heads || current.head_dim != saved.head_dim {
                return Err("sequence state layer layout does not match this KV cache".to_string());
            }
            let element_count = snapshot
                .current_len
                .saturating_mul(current.num_kv_heads)
                .saturating_mul(current.head_dim);
            let storage_matches = match (&current.storage, &saved.storage) {
                (LayerCacheStorage::F16 { .. }, LayerCacheSnapshotStorage::F16 { key, value }) => {
                    key.len() == element_count && value.len() == element_count
                }
                (LayerCacheStorage::Kvarn(current), LayerCacheSnapshotStorage::Kvarn(saved)) => {
                    current.layout_matches(saved) && saved.stored_len() == snapshot.current_len
                }
                _ => false,
            };
            if !storage_matches {
                return Err("sequence state layer storage does not match this KV cache".to_string());
            }
        }
        let glm_indexer_len = snapshot
            .current_len
            .saturating_mul(snapshot.glm_indexer_key_len);
        if snapshot
            .glm_indexer_k
            .iter()
            .any(|keys| keys.len() != glm_indexer_len)
        {
            return Err(
                "sequence state GLM indexer layout does not match this KV cache".to_string(),
            );
        }

        for (current, saved) in self.layers.iter_mut().zip(&snapshot.layers) {
            match (&mut current.storage, &saved.storage) {
                (
                    LayerCacheStorage::F16 { key, value },
                    LayerCacheSnapshotStorage::F16 {
                        key: saved_key,
                        value: saved_value,
                    },
                ) => {
                    key[..saved_key.len()].copy_from_slice(saved_key);
                    value[..saved_value.len()].copy_from_slice(saved_value);
                }
                (LayerCacheStorage::Kvarn(current), LayerCacheSnapshotStorage::Kvarn(saved)) => {
                    *current = saved.clone()
                }
                _ => unreachable!("snapshot layout was validated"),
            }
        }
        for (current, saved) in self.glm_indexer_k.iter_mut().zip(&snapshot.glm_indexer_k) {
            current[..glm_indexer_len].copy_from_slice(saved);
        }
        restore_ssm_states(&mut self.ssm_states, &snapshot.ssm_states);
        self.current_len = snapshot.current_len;
        Ok(())
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.truncate_to(0);
        }
        for ssm in &mut self.ssm_states {
            if let Some(s) = ssm {
                s.clear();
            }
        }
        self.current_len = 0;
        self.sequence_epoch = NEXT_KV_CACHE_SEQUENCE_EPOCH.fetch_add(1, Ordering::Relaxed);
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

        let (key_ptr, value_ptr) = match &layer.storage {
            LayerCacheStorage::F16 { key, value } => (key.as_ptr() as u64, value.as_ptr() as u64),
            LayerCacheStorage::Kvarn(_) => {
                return Err(kv_bucket_view_error(
                    "external contiguous KV sharing is unavailable for a KVarN cache",
                ));
            }
        };
        crate::runtime::KvBucketView::new(
            layer_idx,
            self.max_seq_len,
            logical_len,
            self.max_seq_len,
            kv_row_width,
            key_ptr,
            value_ptr,
        )
    }

    /// Layer 의 K cache 를 `[current_len, kv_dim]` f32 row-major 로 dequant.
    /// F16 (u16 bits) 저장이라 호출마다 dequant. Cross-attention drafter 용 read-only 접근.
    pub fn dequant_k_layer(&self, layer_idx: usize) -> Vec<f32> {
        let read = self.read_up_to(layer_idx, self.current_len);
        let (bits, _) = read.as_slices();
        bits.iter()
            .map(|&entry| half::f16::from_bits(entry).to_f32())
            .collect()
    }

    /// Layer 의 V cache 를 `[current_len, kv_dim]` f32 row-major 로 dequant.
    pub fn dequant_v_layer(&self, layer_idx: usize) -> Vec<f32> {
        let read = self.read_up_to(layer_idx, self.current_len);
        let (_, bits) = read.as_slices();
        bits.iter()
            .map(|&entry| half::f16::from_bits(entry).to_f32())
            .collect()
    }
}

fn quantization_snr_db(signal_energy: f64, error_energy: f64) -> Option<f32> {
    if signal_energy <= 0.0 {
        None
    } else if error_energy <= f64::MIN_POSITIVE {
        Some(f32::INFINITY)
    } else {
        Some((10.0 * (signal_energy / error_energy).log10()) as f32)
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
    if crate::engine::policy::env_os_string("RNB_DEBUG_KV_HASH_TRACE").is_none() {
        return;
    }
    if let Some(filter) = crate::engine::policy::env_string("RNB_DEBUG_KV_HASH_LAYER") {
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
        let sequence_epoch = cache.sequence_epoch();
        cache.append(0, 0, &[1.0, 1.0], &[1.0, 1.0]);
        assert_eq!(cache.current_len(), 1);

        cache.clear();
        assert_ne!(cache.sequence_epoch(), sequence_epoch);
        assert_eq!(cache.current_len(), 0);
        let (k, _) = cache.get(0);
        assert_eq!(k.len(), 0);
    }
    #[test]
    fn snapshot_restores_exact_logical_rows_ssm_and_glm_indexer_state() {
        let mut cache = KVCache::new(1, 16, 1, 2);
        cache.init_glm_indexer(1, 2, 1);
        cache.append(0, 0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.append(0, 1, &[5.0, 6.0], &[7.0, 8.0]);
        cache.write_glm_indexer_k(0, 0, &[0.25, -0.5]);
        cache.write_glm_indexer_k(0, 1, &[1.25, -1.5]);
        cache.ssm_states[0] = Some(SsmLayerState {
            conv_state: vec![9.0, 10.0],
            delta_state: vec![11.0],
            conv_kernel: 2,
            conv_channels: 1,
        });
        let (expected_key, expected_value) = cache.get(0);
        let expected_key = expected_key.to_vec();
        let expected_value = expected_value.to_vec();
        let expected_glm_indexer = cache.glm_indexer_k_up_to(0, 2).to_vec();
        let estimated_bytes = cache.snapshot_byte_size();
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.byte_size(), estimated_bytes);

        cache.append(0, 2, &[12.0, 13.0], &[14.0, 15.0]);
        cache.write_glm_indexer_k(0, 0, &[0.0, 0.0]);
        cache.write_glm_indexer_k(0, 1, &[0.0, 0.0]);
        cache.ssm_states[0].as_mut().unwrap().conv_state.fill(0.0);
        cache.restore_snapshot(&snapshot).unwrap();

        assert_eq!(cache.current_len(), 2);
        let (key, value) = cache.get(0);
        assert_eq!(key, expected_key);
        assert_eq!(value, expected_value);
        assert_eq!(cache.glm_indexer_k_up_to(0, 2), expected_glm_indexer);
        let ssm = cache.ssm_states[0].as_ref().unwrap();
        assert_eq!(ssm.conv_state, vec![9.0, 10.0]);
        assert_eq!(ssm.delta_state, vec![11.0]);
    }

    #[test]
    fn snapshot_preserves_exact_f16_bits_and_validates_layout() {
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
        let expected_key = key.to_vec();
        let expected_value = value.to_vec();

        let estimated_bytes = cache.snapshot_byte_size();
        let snapshot = cache.snapshot();
        let LayerCacheSnapshotStorage::F16 {
            key: snapshot_key,
            value: snapshot_value,
        } = &snapshot.layers[0].storage
        else {
            panic!("expected F16 snapshot");
        };
        let f16_payload = snapshot_key
            .len()
            .saturating_add(snapshot_value.len())
            .saturating_mul(std::mem::size_of::<u16>());
        assert_eq!(snapshot.byte_size(), estimated_bytes);
        assert_eq!(
            f16_payload,
            (expected_key.len() + expected_value.len()) * std::mem::size_of::<u16>()
        );
        assert_eq!(*snapshot_key, expected_key);
        assert_eq!(*snapshot_value, expected_value);

        let zeros = vec![0.0; stride];
        for pos in 0..token_count {
            cache.append(0, pos, &zeros, &zeros);
        }
        cache.restore_snapshot(&snapshot).unwrap();
        let (restored_key, restored_value) = cache.get(0);
        assert_eq!(restored_key, expected_key);
        assert_eq!(restored_value, expected_value);

        let mut invalid = snapshot.clone();
        let LayerCacheSnapshotStorage::F16 { key, .. } = &mut invalid.layers[0].storage else {
            panic!("expected F16 snapshot");
        };
        key.pop();
        assert!(cache.restore_snapshot(&invalid).is_err());
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
    #[test]
    fn kvarn_capacity_is_over_four_times_smaller_than_f16_for_qwen_shape() {
        let f16 = KVCache::new_per_layer(4096, &[2], &[256]);
        let kvarn = KVCache::new_per_layer_with_formats(
            4096,
            &[2],
            &[256],
            &[KvCacheFormat::KvarnK4V2G128],
        )
        .unwrap();
        let kvarn_k4v4 = KVCache::new_per_layer_with_formats(
            4096,
            &[2],
            &[256],
            &[KvCacheFormat::KvarnK4V4G128],
        )
        .unwrap();

        assert!(kvarn.layer_uses_kvarn(0));
        assert!(kvarn.capacity_kv_bytes() * 4 < f16.capacity_kv_bytes());
        assert!(kvarn_k4v4.capacity_kv_bytes() * 3 < f16.capacity_kv_bytes());
        assert_eq!(kvarn.allocated_kv_bytes(), 0);
        assert_eq!(
            kvarn.metrics().f16_equivalent_capacity_bytes,
            f16.capacity_kv_bytes()
        );
        assert_eq!(
            "kvarn-k4v2-g128".parse::<KvCacheFormat>().unwrap(),
            KvCacheFormat::KvarnK4V2G128
        );
    }

    #[test]
    fn zero_width_recurrent_layers_do_not_reserve_kv_storage() {
        let attention_only =
            KVCache::new_per_layer_with_formats(512, &[2], &[256], &[KvCacheFormat::KvarnK4V4G128])
                .unwrap();
        let with_recurrent_layer = KVCache::new_per_layer_with_formats(
            512,
            &[2, 0],
            &[256, 256],
            &[KvCacheFormat::KvarnK4V4G128, KvCacheFormat::F16],
        )
        .unwrap();

        assert_eq!(
            with_recurrent_layer.capacity_kv_bytes(),
            attention_only.capacity_kv_bytes()
        );
    }

    #[test]
    fn kvarn_reports_uninitialized_device_resident_rows_without_materializing() {
        let width = 16;
        let mut cache =
            KVCache::new_per_layer_with_formats(32, &[1], &[width], &[KvCacheFormat::KvarnK4V2G64])
                .unwrap();

        cache.set_len(4);
        assert_eq!(cache.current_len(), 4);
        assert!(cache.read_up_to_if_initialized(0, 4).is_none());

        let key = vec![half::f16::from_f32(0.25).to_bits(); 4 * width];
        let value = vec![half::f16::from_f32(-0.5).to_bits(); 4 * width];
        cache.append_bits_range(0, 0, 4, &key, &value);
        let materialized = cache
            .read_up_to_if_initialized(0, 4)
            .expect("host KVarN rows should now be initialized");
        assert_eq!(materialized.as_slices(), (key.as_slice(), value.as_slice()));
    }

    #[test]
    fn kvarn_small_context_stays_f16_and_full_tiles_compact() {
        let width = 16;
        let mut cache = KVCache::new_per_layer_with_formats(
            384,
            &[1],
            &[width],
            &[KvCacheFormat::KvarnK4V2G64],
        )
        .unwrap();
        let make_bits = |rows: usize, phase: f32| {
            (0..rows * width)
                .map(|index| {
                    let x = index as f32;
                    half::f16::from_f32((x * 0.013 + phase).sin()).to_bits()
                })
                .collect::<Vec<_>>()
        };

        let small_key = make_bits(96, 0.1);
        let small_value = make_bits(96, 0.7);
        cache.append_bits_range(0, 0, 96, &small_key, &small_value);
        cache.compact_layer(0).unwrap();
        let view = cache.kvarn_view(0, 96).unwrap();
        assert!(view.blocks.is_empty());
        assert_eq!(view.sink_key, small_key);
        assert_eq!(view.sink_value, small_value);

        let key = make_bits(256, 0.2);
        let value = make_bits(256, 0.9);
        cache.append_bits_range(0, 0, 256, &key, &value);
        cache.compact_layer(0).unwrap();
        let view = cache.kvarn_view(0, 256).unwrap();
        assert_eq!(view.sink_key.len(), 128 * width);
        assert_eq!(view.blocks.len(), 2);
        assert!(view.tail_key.is_empty());
        assert!(cache.allocated_kv_bytes() < (256 * width * 2 * 2) as u64);
        let metrics = cache.metrics();
        assert_eq!(metrics.quantized_token_rows, 128);
        assert!(metrics.key_quant_snr_db.is_some_and(|value| value > 0.0));
        assert!(metrics.value_quant_snr_db.is_some_and(|value| value > 0.0));
    }

    #[test]
    fn kvarn_compacted_range_replace_builds_device_ready_tail() {
        let width = 16;
        let rows = 1_139;
        let mut cache = KVCache::new_per_layer_with_formats(
            2_048,
            &[1],
            &[width],
            &[KvCacheFormat::KvarnK4V4G128],
        )
        .unwrap();
        let key = vec![half::f16::from_f32(0.25).to_bits(); rows * width];
        let value = vec![half::f16::from_f32(-0.5).to_bits(); rows * width];

        cache
            .replace_layer_f16_range_compacted(0, 0, rows, &key, &value)
            .unwrap();

        let view = cache.kvarn_view(0, rows).unwrap();
        assert!(view.tail_key.len() / width <= view.config.group);
        assert_eq!(
            view.sink_key.len() / width
                + view.device_blocks.len() / view.device_layout.block_bytes * view.config.group
                + view.tail_key.len() / width,
            rows
        );
    }

    #[test]
    fn kvarn_snapshot_restore_keeps_compressed_state_and_supports_partial_rollback() {
        let width = 16;
        let mut cache = KVCache::new_per_layer_with_formats(
            384,
            &[1],
            &[width],
            &[KvCacheFormat::KvarnK4V4G64],
        )
        .unwrap();
        let key = (0..256 * width)
            .map(|index| half::f16::from_f32((index as f32 * 0.019).sin() * 0.8).to_bits())
            .collect::<Vec<_>>();
        let value = (0..256 * width)
            .map(|index| half::f16::from_f32((index as f32 * 0.011 + 0.4).cos() * 0.6).to_bits())
            .collect::<Vec<_>>();
        cache.append_bits_range(0, 0, 256, &key, &value);
        cache.compact_layer(0).unwrap();
        let expected = {
            let read = cache.read_up_to(0, 256);
            let (key, value) = read.as_slices();
            (key.to_vec(), value.to_vec())
        };
        let snapshot = cache.snapshot();
        assert!(matches!(
            snapshot.layers[0].storage,
            LayerCacheSnapshotStorage::Kvarn(_)
        ));

        cache.append(0, 256, &vec![0.25; width], &vec![-0.5; width]);
        cache.restore_snapshot(&snapshot).unwrap();
        assert_eq!(cache.current_len(), 256);
        let restored = cache.read_up_to(0, 256);
        let (restored_key, restored_value) = restored.as_slices();
        assert_eq!(restored_key, expected.0);
        assert_eq!(restored_value, expected.1);

        cache.set_len(200);
        cache.append(0, 200, &vec![0.5; width], &vec![-0.25; width]);
        cache.compact_layer(0).unwrap();
        assert_eq!(cache.current_len(), 201);
        assert_eq!(cache.read_up_to(0, 201).as_slices().0.len(), 201 * width);
    }
}
