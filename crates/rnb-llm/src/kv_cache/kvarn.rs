use crate::engine::cpu_runtime::quantize::kvarn::{
    KvarnBlock, KvarnConfig, KvarnDeviceRecordLayout, KvarnKvView,
};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCacheFormat {
    #[default]
    F16,
    KvarnK4V2G64,
    KvarnK4V4G64,
    KvarnK4V2G128,
    KvarnK4V4G128,
}

impl KvCacheFormat {
    pub fn kvarn_config(self) -> Option<KvarnConfig> {
        match self {
            Self::F16 => None,
            Self::KvarnK4V2G64 => Some(KvarnConfig::K4_V2_G64),
            Self::KvarnK4V4G64 => Some(KvarnConfig::K4_V4_G64),
            Self::KvarnK4V2G128 => Some(KvarnConfig::K4_V2_G128),
            Self::KvarnK4V4G128 => Some(KvarnConfig::K4_V4_G128),
        }
    }

    pub fn label(self) -> &'static str {
        self.kvarn_config().map(KvarnConfig::label).unwrap_or("f16")
    }
}

impl FromStr for KvCacheFormat {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "f16" | "fp16" => Ok(Self::F16),
            "kvarn-k4v2-g64" | "k4v2-g64" => Ok(Self::KvarnK4V2G64),
            "kvarn-k4v4-g64" | "k4v4-g64" => Ok(Self::KvarnK4V4G64),
            "kvarn-k4v2-g128" | "kvarn-k4v2" | "k4v2-g128" | "k4v2" => {
                Ok(Self::KvarnK4V2G128)
            }
            "kvarn-k4v4-g128" | "kvarn-k4v4" | "k4v4-g128" | "k4v4" => {
                Ok(Self::KvarnK4V4G128)
            }
            _ => Err(format!(
                "unknown KV-cache format {value:?}; expected f16, kvarn-k4v2-g64, kvarn-k4v4-g64, kvarn-k4v2-g128, or kvarn-k4v4-g128"
            )),
        }
    }
}

#[derive(Clone)]
pub(super) struct KvarnLayerCache {
    config: KvarnConfig,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    sink_key: Vec<u16>,
    sink_value: Vec<u16>,
    blocks: Vec<KvarnBlock>,
    tail_key: Vec<u16>,
    tail_value: Vec<u16>,
    device_layout: KvarnDeviceRecordLayout,
    device_blocks: Vec<u8>,
    stored_len: usize,
}

impl KvarnLayerCache {
    pub(super) fn new(
        config: KvarnConfig,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, String> {
        config.validate(head_dim)?;
        let device_layout = KvarnDeviceRecordLayout::new(config, num_kv_heads, head_dim)?;
        Ok(Self {
            config,
            num_kv_heads,
            head_dim,
            max_seq_len,
            sink_key: Vec::new(),
            sink_value: Vec::new(),
            blocks: Vec::new(),
            tail_key: Vec::new(),
            tail_value: Vec::new(),
            stored_len: 0,
            device_layout,
            device_blocks: Vec::new(),
        })
    }

    fn row_width(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    #[cfg(feature = "cuda")]
    pub(super) fn is_initialized_up_to(&self, len: usize) -> bool {
        len <= self.stored_len
    }

    pub(super) fn write_f32(&mut self, pos: usize, key: &[f32], value: &[f32]) {
        let width = self.row_width();
        assert_eq!(key.len(), width, "KVarN key row width mismatch");
        assert_eq!(value.len(), width, "KVarN value row width mismatch");
        let key_bits = key
            .iter()
            .map(|&entry| half::f16::from_f32(entry).to_bits())
            .collect::<Vec<_>>();
        let value_bits = value
            .iter()
            .map(|&entry| half::f16::from_f32(entry).to_bits())
            .collect::<Vec<_>>();
        self.write_bits_range(pos, 1, &key_bits, &value_bits);
    }

    pub(super) fn write_bits_up_to(&mut self, len: usize, key: &[u16], value: &[u16]) {
        self.truncate_to(0);
        self.write_bits_range(0, len, key, value);
    }

    pub(super) fn write_bits_range(
        &mut self,
        pos_start: usize,
        len: usize,
        key: &[u16],
        value: &[u16],
    ) {
        assert!(
            pos_start.saturating_add(len) <= self.max_seq_len,
            "KVarN cache overflow"
        );
        let width = self.row_width();
        let count = len * width;
        assert!(
            key.len() >= count && value.len() >= count,
            "KVarN bits underflow"
        );
        if pos_start < self.stored_len {
            self.truncate_to(pos_start);
        }
        while self.stored_len < pos_start {
            let zero = vec![0u16; width];
            self.append_row(&zero, &zero);
        }
        for row in 0..len {
            let offset = row * width;
            self.append_row(&key[offset..offset + width], &value[offset..offset + width]);
        }
    }

    fn append_row(&mut self, key: &[u16], value: &[u16]) {
        assert!(self.stored_len < self.max_seq_len, "KVarN cache overflow");
        if self.stored_len < self.config.sink_tokens {
            self.sink_key.extend_from_slice(key);
            self.sink_value.extend_from_slice(value);
        } else {
            self.tail_key.extend_from_slice(key);
            self.tail_value.extend_from_slice(value);
        }
        self.stored_len += 1;
    }

    pub(super) fn compact(&mut self) -> Result<(), String> {
        if self.stored_len <= self.config.sink_tokens {
            return Ok(());
        }
        let width = self.row_width();
        let block_elements = self.config.group * width;
        let full_blocks = self.tail_key.len() / block_elements;
        if full_blocks == 0 {
            return Ok(());
        }
        let consumed = full_blocks * block_elements;
        for block_index in 0..full_blocks {
            let start = block_index * block_elements;
            let end = start + block_elements;
            let block = KvarnBlock::quantize(
                self.config,
                self.num_kv_heads,
                self.head_dim,
                &self.tail_key[start..end],
                &self.tail_value[start..end],
            )?;
            block.append_device_record(&mut self.device_blocks);
            self.blocks.push(block);
        }
        self.tail_key = self.tail_key.split_off(consumed);
        self.tail_value = self.tail_value.split_off(consumed);
        Ok(())
    }

    pub(super) fn truncate_to(&mut self, len: usize) {
        let len = len.min(self.stored_len);
        let width = self.row_width();
        if len <= self.config.sink_tokens {
            self.sink_key.truncate(len * width);
            self.sink_value.truncate(len * width);
            self.blocks.clear();
            self.device_blocks.clear();
            self.tail_key.clear();
            self.tail_value.clear();
            self.stored_len = len;
            return;
        }

        let after_sink = len - self.config.sink_tokens;
        let quantized_rows = self.blocks.len() * self.config.group;
        if after_sink < quantized_rows {
            let retained_blocks = after_sink / self.config.group;
            let partial_rows = after_sink % self.config.group;
            if partial_rows > 0 {
                let (key, value) = self.blocks[retained_blocks].dequantize_f16();
                self.tail_key = key[..partial_rows * width].to_vec();
                self.tail_value = value[..partial_rows * width].to_vec();
            } else {
                self.tail_key.clear();
                self.tail_value.clear();
            }
            self.blocks.truncate(retained_blocks);
            self.device_blocks
                .truncate(retained_blocks * self.device_layout.block_bytes);
        } else {
            let tail_rows = after_sink - quantized_rows;
            self.tail_key.truncate(tail_rows * width);
            self.tail_value.truncate(tail_rows * width);
        }
        self.stored_len = len;
    }

    pub(super) fn materialize(&self, len: usize) -> (Vec<u16>, Vec<u16>) {
        assert!(
            len <= self.stored_len,
            "KVarN read exceeds initialized rows"
        );
        let width = self.row_width();
        let count = len * width;
        let mut key = Vec::with_capacity(count);
        let mut value = Vec::with_capacity(count);
        let sink_rows = len.min(self.config.sink_tokens);
        key.extend_from_slice(&self.sink_key[..sink_rows * width]);
        value.extend_from_slice(&self.sink_value[..sink_rows * width]);

        for (block_index, block) in self.blocks.iter().enumerate() {
            let block_start = self.config.sink_tokens + block_index * self.config.group;
            if block_start >= len {
                break;
            }
            let rows = (len - block_start).min(self.config.group);
            let (block_key, block_value) = block.dequantize_f16();
            key.extend_from_slice(&block_key[..rows * width]);
            value.extend_from_slice(&block_value[..rows * width]);
        }

        let tail_start = self.config.sink_tokens + self.blocks.len() * self.config.group;
        if len > tail_start {
            let rows = len - tail_start;
            key.extend_from_slice(&self.tail_key[..rows * width]);
            value.extend_from_slice(&self.tail_value[..rows * width]);
        }
        debug_assert_eq!(key.len(), count);
        debug_assert_eq!(value.len(), count);
        (key, value)
    }

    pub(super) fn view(&self, len: usize) -> KvarnKvView<'_> {
        assert!(
            len <= self.stored_len,
            "KVarN view exceeds initialized rows"
        );
        let width = self.row_width();
        let sink_rows = len.min(self.config.sink_tokens);
        let tail_start = self.config.sink_tokens + self.blocks.len() * self.config.group;
        let tail_rows = len.saturating_sub(tail_start);
        KvarnKvView {
            config: self.config,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            sink_key: &self.sink_key[..sink_rows * width],
            sink_value: &self.sink_value[..sink_rows * width],
            blocks: &self.blocks,
            device_layout: self.device_layout,
            device_blocks: &self.device_blocks,
            tail_start,
            tail_key: &self.tail_key[..tail_rows * width],
            tail_value: &self.tail_value[..tail_rows * width],
            len,
        }
    }

    pub(super) fn quantized_rows(&self) -> usize {
        self.blocks.len().saturating_mul(self.config.group)
    }

    pub(super) fn quantization_energy(&self) -> ((f64, f64), (f64, f64)) {
        self.blocks.iter().fold(
            ((0.0, 0.0), (0.0, 0.0)),
            |((key_signal, key_error), (value_signal, value_error)), block| {
                let ((block_key_signal, block_key_error), (block_value_signal, block_value_error)) =
                    block.quantization_energy();
                (
                    (key_signal + block_key_signal, key_error + block_key_error),
                    (
                        value_signal + block_value_signal,
                        value_error + block_value_error,
                    ),
                )
            },
        )
    }

    pub(super) fn actual_byte_size(&self) -> usize {
        self.sink_key
            .capacity()
            .saturating_add(self.sink_value.capacity())
            .saturating_add(self.tail_key.capacity())
            .saturating_add(self.tail_value.capacity())
            .saturating_mul(std::mem::size_of::<u16>())
            .saturating_add(self.blocks.iter().map(KvarnBlock::byte_size).sum::<usize>())
    }

    pub(super) fn capacity_byte_size(&self) -> usize {
        let width = self.row_width();
        let sink_rows = self.max_seq_len.min(self.config.sink_tokens);
        let remaining = self.max_seq_len.saturating_sub(sink_rows);
        let full_blocks = remaining / self.config.group;
        let tail_rows = remaining % self.config.group;
        let payload_per_block = self
            .num_kv_heads
            .saturating_mul(self.head_dim)
            .saturating_mul(self.config.group)
            .saturating_mul(self.config.key_bits as usize + self.config.value_bits as usize)
            / 8;
        let metadata_per_block = self
            .num_kv_heads
            .saturating_mul(3 * self.head_dim + 3 * self.config.group)
            .saturating_mul(std::mem::size_of::<u16>());
        sink_rows
            .saturating_add(tail_rows)
            .saturating_mul(width)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u16>())
            .saturating_add(
                full_blocks.saturating_mul(payload_per_block.saturating_add(metadata_per_block)),
            )
    }

    pub(super) fn snapshot(&self, len: usize) -> Self {
        let mut snapshot = self.clone();
        snapshot.truncate_to(len);
        snapshot
    }

    pub(super) fn layout_matches(&self, other: &Self) -> bool {
        self.config == other.config
            && self.num_kv_heads == other.num_kv_heads
            && self.head_dim == other.head_dim
            && self.max_seq_len == other.max_seq_len
            && other.stored_len <= self.max_seq_len
    }

    pub(super) fn stored_len(&self) -> usize {
        self.stored_len
    }
}
