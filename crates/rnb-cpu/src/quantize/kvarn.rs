//! KVarN KV-cache compression primitives and CPU decode attention.
//!
//! This is a native Rust port of the public KVarN method: normalized Hadamard
//! rotation, log-domain alternating variance normalization with best-state
//! tracking, asymmetric round-to-nearest quantization, and KIVI-oriented
//! per-channel K / per-token V storage. The implementation follows the Apache
//! 2.0 reference backend in `huawei-csl/KVarN` at commit
//! `7586257f1c632e63187bfacbbe21ccb51540f7b3`.

use half::f16;
use rayon::prelude::*;

const STD_MIN: f32 = 1.0e-3;
const STD_MAX: f32 = 1.0e3;
const LOG_SCALE_MIN: f32 = -0.3;
const LOG_SCALE_MAX: f32 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvarnConfig {
    pub key_bits: u8,
    pub value_bits: u8,
    pub group: usize,
    pub sink_tokens: usize,
    pub sinkhorn_iterations: usize,
}

impl KvarnConfig {
    pub const K4_V2_G64: Self = Self {
        key_bits: 4,
        value_bits: 2,
        group: 64,
        sink_tokens: 128,
        sinkhorn_iterations: 8,
    };
    pub const K4_V4_G64: Self = Self {
        key_bits: 4,
        value_bits: 4,
        group: 64,
        sink_tokens: 128,
        sinkhorn_iterations: 8,
    };
    pub const K4_V2_G128: Self = Self {
        key_bits: 4,
        value_bits: 2,
        group: 128,
        sink_tokens: 128,
        sinkhorn_iterations: 8,
    };
    pub const K4_V4_G128: Self = Self {
        key_bits: 4,
        value_bits: 4,
        group: 128,
        sink_tokens: 128,
        sinkhorn_iterations: 8,
    };

    pub fn validate(self, head_dim: usize) -> Result<(), String> {
        if self.key_bits != 4 || !matches!(self.value_bits, 2 | 4) {
            return Err(format!(
                "KVarN supports K4 with V2 or V4, got K{}V{}",
                self.key_bits, self.value_bits
            ));
        }
        if !matches!(self.group, 64 | 128) {
            return Err(format!(
                "KVarN tile size must be 64 or 128, got {}",
                self.group
            ));
        }
        if head_dim < 4 || head_dim % 4 != 0 {
            return Err(format!(
                "KVarN head dimension must be divisible by four, got {head_dim}"
            ));
        }
        if self.sinkhorn_iterations == 0 {
            return Err("KVarN Sinkhorn iteration count must be positive".to_string());
        }
        Ok(())
    }

    pub fn label(self) -> &'static str {
        match (self.key_bits, self.value_bits, self.group) {
            (4, 2, 64) => "kvarn-k4v2-g64",
            (4, 4, 64) => "kvarn-k4v4-g64",
            (4, 2, 128) => "kvarn-k4v2-g128",
            (4, 4, 128) => "kvarn-k4v4-g128",
            _ => "kvarn-custom",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvarnDeviceRecordLayout {
    pub key_packed_offset: usize,
    pub key_scale_offset: usize,
    pub key_zero_offset: usize,
    pub key_token_scale_offset: usize,
    pub value_packed_offset: usize,
    pub value_channel_scale_offset: usize,
    pub value_token_scale_offset: usize,
    pub value_zero_offset: usize,
    pub block_bytes: usize,
}

impl KvarnDeviceRecordLayout {
    pub fn new(config: KvarnConfig, num_kv_heads: usize, head_dim: usize) -> Result<Self, String> {
        config.validate(head_dim)?;
        if num_kv_heads == 0 {
            return Err("KVarN device record requires at least one KV head".to_string());
        }
        let channels = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "KVarN device channel count overflow".to_string())?;
        let token_rows = num_kv_heads
            .checked_mul(config.group)
            .ok_or_else(|| "KVarN device token row count overflow".to_string())?;
        let key_packed_bytes = channels
            .checked_mul(config.group)
            .ok_or_else(|| "KVarN device key size overflow".to_string())?
            / (8 / config.key_bits as usize);
        let value_packed_bytes = token_rows
            .checked_mul(head_dim)
            .ok_or_else(|| "KVarN device value size overflow".to_string())?
            / (8 / config.value_bits as usize);
        let channel_scale_bytes = channels
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "KVarN device channel scale size overflow".to_string())?;
        let token_scale_bytes = token_rows
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| "KVarN device token scale size overflow".to_string())?;

        let mut cursor = 0usize;
        let key_packed_offset = take_device_record_field(&mut cursor, key_packed_bytes)?;
        let key_scale_offset = take_device_record_field(&mut cursor, channel_scale_bytes)?;
        let key_zero_offset = take_device_record_field(&mut cursor, channel_scale_bytes)?;
        let key_token_scale_offset = take_device_record_field(&mut cursor, token_scale_bytes)?;
        let value_packed_offset = take_device_record_field(&mut cursor, value_packed_bytes)?;
        let value_channel_scale_offset =
            take_device_record_field(&mut cursor, channel_scale_bytes)?;
        let value_token_scale_offset = take_device_record_field(&mut cursor, token_scale_bytes)?;
        let value_zero_offset = take_device_record_field(&mut cursor, token_scale_bytes)?;
        Ok(Self {
            key_packed_offset,
            key_scale_offset,
            key_zero_offset,
            key_token_scale_offset,
            value_packed_offset,
            value_channel_scale_offset,
            value_token_scale_offset,
            value_zero_offset,
            block_bytes: cursor,
        })
    }
}

fn take_device_record_field(cursor: &mut usize, bytes: usize) -> Result<usize, String> {
    let offset = *cursor;
    *cursor = cursor
        .checked_add(bytes)
        .ok_or_else(|| "KVarN device record size overflow".to_string())?;
    Ok(offset)
}

fn append_u16_le(output: &mut Vec<u8>, values: &[u16]) {
    output.reserve(values.len().saturating_mul(std::mem::size_of::<u16>()));
    for &value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

#[derive(Clone, Debug)]
pub struct KvarnBlock {
    pub config: KvarnConfig,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub key_packed: Vec<u8>,
    pub key_scale: Vec<u16>,
    pub key_zero: Vec<u16>,
    pub key_token_scale: Vec<u16>,
    pub value_packed: Vec<u8>,
    pub value_channel_scale: Vec<u16>,
    pub value_token_scale: Vec<u16>,
    pub value_zero: Vec<u16>,
    key_signal_energy: f64,
    key_error_energy: f64,
    value_signal_energy: f64,
    value_error_energy: f64,
}

impl KvarnBlock {
    pub fn quantize(
        config: KvarnConfig,
        num_kv_heads: usize,
        head_dim: usize,
        key_f16: &[u16],
        value_f16: &[u16],
    ) -> Result<Self, String> {
        config.validate(head_dim)?;
        let row_width = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "KVarN row width overflow".to_string())?;
        let expected = config
            .group
            .checked_mul(row_width)
            .ok_or_else(|| "KVarN tile size overflow".to_string())?;
        if key_f16.len() != expected || value_f16.len() != expected {
            return Err(format!(
                "KVarN tile length mismatch: K={} V={} expected={expected}",
                key_f16.len(),
                value_f16.len()
            ));
        }

        let key_pack = 8 / config.key_bits as usize;
        let value_pack = 8 / config.value_bits as usize;
        let mut block = Self {
            config,
            num_kv_heads,
            head_dim,
            key_packed: Vec::with_capacity(num_kv_heads * head_dim * config.group / key_pack),
            key_scale: Vec::with_capacity(num_kv_heads * head_dim),
            key_zero: Vec::with_capacity(num_kv_heads * head_dim),
            key_token_scale: Vec::with_capacity(num_kv_heads * config.group),
            value_packed: Vec::with_capacity(num_kv_heads * config.group * head_dim / value_pack),
            value_channel_scale: Vec::with_capacity(num_kv_heads * head_dim),
            value_token_scale: Vec::with_capacity(num_kv_heads * config.group),
            value_zero: Vec::with_capacity(num_kv_heads * config.group),
            key_signal_energy: 0.0,
            key_error_energy: 0.0,
            value_signal_energy: 0.0,
            value_error_energy: 0.0,
        };

        for head in 0..num_kv_heads {
            let mut key_rotated = vec![0.0f32; config.group * head_dim];
            let mut value_rotated = vec![0.0f32; config.group * head_dim];
            let mut row = vec![0.0f32; head_dim];
            for token in 0..config.group {
                let source = token * row_width + head * head_dim;
                copy_f16_to_f32(&key_f16[source..source + head_dim], &mut row);
                normalized_hadamard_in_place(&mut row);
                for dim in 0..head_dim {
                    key_rotated[dim * config.group + token] = row[dim];
                }

                copy_f16_to_f32(&value_f16[source..source + head_dim], &mut row);
                normalized_hadamard_in_place(&mut row);
                value_rotated[token * head_dim..(token + 1) * head_dim].copy_from_slice(&row);
            }

            let key_balanced = variance_normalize(
                &key_rotated,
                head_dim,
                config.group,
                config.sinkhorn_iterations,
            );
            quantize_key_head(&key_balanced, config, &mut block);

            let value_balanced = variance_normalize(
                &value_rotated,
                config.group,
                head_dim,
                config.sinkhorn_iterations,
            );
            quantize_value_head(&value_balanced, config, &mut block);
        }
        Ok(block)
    }

    pub fn byte_size(&self) -> usize {
        self.key_packed
            .capacity()
            .saturating_add(self.value_packed.capacity())
            .saturating_add(
                self.key_scale
                    .capacity()
                    .saturating_add(self.key_zero.capacity())
                    .saturating_add(self.key_token_scale.capacity())
                    .saturating_add(self.value_channel_scale.capacity())
                    .saturating_add(self.value_token_scale.capacity())
                    .saturating_add(self.value_zero.capacity())
                    .saturating_mul(std::mem::size_of::<u16>()),
            )
    }

    pub fn append_device_record(&self, output: &mut Vec<u8>) {
        let layout = KvarnDeviceRecordLayout::new(self.config, self.num_kv_heads, self.head_dim)
            .expect("validated KVarN block must have a valid device layout");
        let start = output.len();
        output.reserve(layout.block_bytes);
        output.extend_from_slice(&self.key_packed);
        append_u16_le(output, &self.key_scale);
        append_u16_le(output, &self.key_zero);
        append_u16_le(output, &self.key_token_scale);
        output.extend_from_slice(&self.value_packed);
        append_u16_le(output, &self.value_channel_scale);
        append_u16_le(output, &self.value_token_scale);
        append_u16_le(output, &self.value_zero);
        debug_assert_eq!(output.len() - start, layout.block_bytes);
    }

    pub fn logical_bytes_per_token(&self) -> usize {
        self.byte_size() / self.config.group
    }

    pub fn quantization_energy(&self) -> ((f64, f64), (f64, f64)) {
        (
            (self.key_signal_energy, self.key_error_energy),
            (self.value_signal_energy, self.value_error_energy),
        )
    }

    pub fn dequantize_f16(&self) -> (Vec<u16>, Vec<u16>) {
        let row_width = self.num_kv_heads * self.head_dim;
        let mut key = vec![0u16; self.config.group * row_width];
        let mut value = vec![0u16; self.config.group * row_width];
        let mut row = vec![0.0f32; self.head_dim];

        for head in 0..self.num_kv_heads {
            for token in 0..self.config.group {
                self.dequantize_key_rotated_row(head, token, &mut row);
                normalized_hadamard_in_place(&mut row);
                let destination = token * row_width + head * self.head_dim;
                for dim in 0..self.head_dim {
                    key[destination + dim] = f16::from_f32(row[dim]).to_bits();
                }

                self.dequantize_value_rotated_row(head, token, &mut row);
                normalized_hadamard_in_place(&mut row);
                for dim in 0..self.head_dim {
                    value[destination + dim] = f16::from_f32(row[dim]).to_bits();
                }
            }
        }
        (key, value)
    }

    fn dequantize_key_rotated_row(&self, head: usize, token: usize, out: &mut [f32]) {
        let group = self.config.group;
        let packed_row = group / 2;
        let packed_head = self.head_dim * packed_row;
        let packed_base = head * packed_head;
        let scale_base = head * self.head_dim;
        let token_scale = f16::from_bits(self.key_token_scale[head * group + token]).to_f32();
        for (dim, destination) in out.iter_mut().enumerate().take(self.head_dim) {
            let packed = self.key_packed[packed_base + dim * packed_row + token / 2];
            let quant = if token & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let scale = f16::from_bits(self.key_scale[scale_base + dim]).to_f32();
            let zero = f16::from_bits(self.key_zero[scale_base + dim]).to_f32();
            *destination = (quant as f32 * scale + zero) * token_scale;
        }
    }

    fn dequantize_value_rotated_row(&self, head: usize, token: usize, out: &mut [f32]) {
        let bits = self.config.value_bits as usize;
        let pack = 8 / bits;
        let packed_row = self.head_dim / pack;
        let packed_head = self.config.group * packed_row;
        let packed_base = head * packed_head + token * packed_row;
        let channel_base = head * self.head_dim;
        let row_scale =
            f16::from_bits(self.value_token_scale[head * self.config.group + token]).to_f32();
        let zero = f16::from_bits(self.value_zero[head * self.config.group + token]).to_f32();
        let mask = (1u8 << bits) - 1;
        for (dim, destination) in out.iter_mut().enumerate().take(self.head_dim) {
            let packed = self.value_packed[packed_base + dim / pack];
            let quant = (packed >> ((dim % pack) * bits)) & mask;
            let channel_scale =
                f16::from_bits(self.value_channel_scale[channel_base + dim]).to_f32();
            *destination = (quant as f32 * row_scale + zero) * channel_scale;
        }
    }
}

#[derive(Clone, Copy)]
pub struct KvarnKvView<'a> {
    pub config: KvarnConfig,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sink_key: &'a [u16],
    pub sink_value: &'a [u16],
    pub blocks: &'a [KvarnBlock],
    pub device_layout: KvarnDeviceRecordLayout,
    pub device_blocks: &'a [u8],
    pub tail_start: usize,
    pub tail_key: &'a [u16],
    pub tail_value: &'a [u16],
    pub len: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode(
    query: &[f32],
    cache: KvarnKvView<'_>,
    output: &mut [f32],
    num_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) {
    let head_dim = cache.head_dim;
    assert_eq!(query.len(), num_heads * head_dim);
    assert_eq!(output.len(), num_heads * head_dim);
    assert!(cache.num_kv_heads > 0 && num_heads % cache.num_kv_heads == 0);
    let heads_per_group = num_heads / cache.num_kv_heads;
    let row_width = cache.num_kv_heads * head_dim;
    let sink_len = cache.sink_key.len() / row_width;
    let tail_len = cache.tail_key.len() / row_width;
    let window_start = sliding_window
        .map(|window| cache.len.saturating_sub(window))
        .unwrap_or(0);

    output
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(head, out)| {
            let kv_head = head / heads_per_group;
            let q = &query[head * head_dim..(head + 1) * head_dim];

            let sink_state = process_f16_region(
                q,
                cache.sink_key,
                cache.sink_value,
                0,
                sink_len,
                row_width,
                kv_head,
                head_dim,
                cache.len,
                window_start,
                scale,
                softcap,
            );

            let mut rotated_q = q.to_vec();
            normalized_hadamard_in_place(&mut rotated_q);
            let mut quantized_state = SoftmaxState::new(head_dim);
            for (block_index, block) in cache.blocks.iter().enumerate() {
                let block_start = cache.config.sink_tokens + block_index * cache.config.group;
                process_quantized_block(
                    &rotated_q,
                    block,
                    block_start,
                    kv_head,
                    cache.len,
                    window_start,
                    scale,
                    softcap,
                    &mut quantized_state,
                );
            }
            if quantized_state.valid() {
                normalized_hadamard_in_place(&mut quantized_state.numerator);
            }

            let tail_state = process_f16_region(
                q,
                cache.tail_key,
                cache.tail_value,
                cache.tail_start,
                tail_len,
                row_width,
                kv_head,
                head_dim,
                cache.len,
                window_start,
                scale,
                softcap,
            );

            let merged = sink_state.merge(quantized_state).merge(tail_state);
            if merged.sum > 0.0 {
                let inverse = 1.0 / merged.sum;
                for dim in 0..head_dim {
                    out[dim] = merged.numerator[dim] * inverse;
                }
            } else {
                out.fill(0.0);
            }
        });
}

#[derive(Debug)]
struct BalancedTile {
    values: Vec<f32>,
    column_scale: Vec<f32>,
    row_scale: Vec<f32>,
    rows: usize,
    columns: usize,
}

fn variance_normalize(
    tile: &[f32],
    rows: usize,
    columns: usize,
    iterations: usize,
) -> BalancedTile {
    debug_assert_eq!(tile.len(), rows * columns);
    let mut log_column_scale = vec![0.0f32; columns];
    let mut log_row_scale = vec![0.0f32; rows];
    let mut column_scale = vec![1.0f32; columns];
    let mut row_scale = vec![1.0f32; rows];
    let mut current = tile.to_vec();
    let mut best_imbalance = imbalance(&current, rows, columns);
    let mut best_column_scale = column_scale.clone();
    let mut best_row_scale = row_scale.clone();

    for _ in 0..iterations {
        let column_std = standard_deviation_columns(&current, rows, columns);
        for column in 0..columns {
            log_column_scale[column] = (log_column_scale[column]
                + column_std[column].clamp(STD_MIN, STD_MAX).ln())
            .clamp(LOG_SCALE_MIN, LOG_SCALE_MAX);
            column_scale[column] = log_column_scale[column].exp();
        }
        rebuild_balanced(tile, &column_scale, &row_scale, rows, columns, &mut current);

        let row_std = standard_deviation_rows(&current, rows, columns);
        for row in 0..rows {
            log_row_scale[row] = (log_row_scale[row] + row_std[row].clamp(STD_MIN, STD_MAX).ln())
                .clamp(LOG_SCALE_MIN, LOG_SCALE_MAX);
            row_scale[row] = log_row_scale[row].exp();
        }
        rebuild_balanced(tile, &column_scale, &row_scale, rows, columns, &mut current);

        let candidate = imbalance(&current, rows, columns);
        if candidate <= best_imbalance {
            best_imbalance = candidate;
            best_column_scale.copy_from_slice(&column_scale);
            best_row_scale.copy_from_slice(&row_scale);
        }
    }

    rebuild_balanced(
        tile,
        &best_column_scale,
        &best_row_scale,
        rows,
        columns,
        &mut current,
    );
    BalancedTile {
        values: current,
        column_scale: best_column_scale,
        row_scale: best_row_scale,
        rows,
        columns,
    }
}

fn rebuild_balanced(
    tile: &[f32],
    column_scale: &[f32],
    row_scale: &[f32],
    rows: usize,
    columns: usize,
    output: &mut [f32],
) {
    for row in 0..rows {
        let row_inverse = 1.0 / row_scale[row];
        for column in 0..columns {
            output[row * columns + column] =
                tile[row * columns + column] * row_inverse / column_scale[column];
        }
    }
}

fn standard_deviation_columns(tile: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut sums = vec![0.0f32; columns];
    let mut squares = vec![0.0f32; columns];
    for row in 0..rows {
        for column in 0..columns {
            let value = tile[row * columns + column];
            sums[column] += value;
            squares[column] += value * value;
        }
    }
    let denominator = rows as f32;
    let sample_correction = if rows > 1 {
        rows as f32 / (rows - 1) as f32
    } else {
        0.0
    };
    (0..columns)
        .map(|column| {
            let mean = sums[column] / denominator;
            ((squares[column] / denominator - mean * mean).max(0.0) * sample_correction).sqrt()
        })
        .collect()
}

fn standard_deviation_rows(tile: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let denominator = columns as f32;
    let sample_correction = if columns > 1 {
        columns as f32 / (columns - 1) as f32
    } else {
        0.0
    };
    (0..rows)
        .map(|row| {
            let values = &tile[row * columns..(row + 1) * columns];
            let sum = values.iter().copied().sum::<f32>();
            let squares = values.iter().map(|value| value * value).sum::<f32>();
            let mean = sum / denominator;
            ((squares / denominator - mean * mean).max(0.0) * sample_correction).sqrt()
        })
        .collect()
}

fn imbalance(tile: &[f32], rows: usize, columns: usize) -> f32 {
    let columns_std = standard_deviation_columns(tile, rows, columns);
    let rows_std = standard_deviation_rows(tile, rows, columns);
    spread(&columns_std) + spread(&rows_std)
}

fn spread(values: &[f32]) -> f32 {
    let minimum = values
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(1.0e-8);
    let maximum = values.iter().copied().fold(0.0f32, f32::max);
    maximum / minimum
}

fn quantize_key_head(tile: &BalancedTile, config: KvarnConfig, block: &mut KvarnBlock) {
    debug_assert_eq!(tile.rows, block.head_dim);
    debug_assert_eq!(tile.columns, config.group);
    let qmax = (1u8 << config.key_bits) - 1;
    for row in 0..tile.rows {
        let values = &tile.values[row * tile.columns..(row + 1) * tile.columns];
        let (minimum, maximum) = min_max(values);
        let scale = ((maximum - minimum) / qmax as f32).max(1.0e-10);
        let absorbed_scale = tile.row_scale[row] * scale;
        let absorbed_zero = tile.row_scale[row] * minimum;
        block
            .key_scale
            .push(f16::from_f32(absorbed_scale).to_bits());
        block.key_zero.push(f16::from_f32(absorbed_zero).to_bits());
        let (signal, error) = pack_row(
            values,
            minimum,
            scale,
            config.key_bits,
            &mut block.key_packed,
        );
        block.key_signal_energy += signal;
        block.key_error_energy += error;
    }
    block.key_token_scale.extend(
        tile.column_scale
            .iter()
            .map(|&value| f16::from_f32(value).to_bits()),
    );
}

fn quantize_value_head(tile: &BalancedTile, config: KvarnConfig, block: &mut KvarnBlock) {
    debug_assert_eq!(tile.rows, config.group);
    debug_assert_eq!(tile.columns, block.head_dim);
    let qmax = (1u8 << config.value_bits) - 1;
    block.value_channel_scale.extend(
        tile.column_scale
            .iter()
            .map(|&value| f16::from_f32(value).to_bits()),
    );
    for row in 0..tile.rows {
        let values = &tile.values[row * tile.columns..(row + 1) * tile.columns];
        let (minimum, maximum) = min_max(values);
        let scale = ((maximum - minimum) / qmax as f32).max(1.0e-10);
        block
            .value_token_scale
            .push(f16::from_f32(tile.row_scale[row] * scale).to_bits());
        block
            .value_zero
            .push(f16::from_f32(tile.row_scale[row] * minimum).to_bits());
        let (signal, error) = pack_row(
            values,
            minimum,
            scale,
            config.value_bits,
            &mut block.value_packed,
        );
        block.value_signal_energy += signal;
        block.value_error_energy += error;
    }
}

fn min_max(values: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON and the helper bounds every load.
        return unsafe { min_max_neon(values) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    values.iter().copied().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
    )
}

fn copy_f16_to_f32(input: &[u16], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("fp16") {
        // SAFETY: runtime feature detection guarantees FP16 conversion support.
        unsafe {
            copy_f16_to_f32_neon(input, output);
        }
        return;
    }
    for (destination, &source) in output.iter_mut().zip(input) {
        *destination = f16::from_bits(source).to_f32();
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn min_max_neon(values: &[f32]) -> (f32, f32) {
    use std::arch::aarch64::*;

    let mut minimum = vdupq_n_f32(f32::INFINITY);
    let mut maximum = vdupq_n_f32(f32::NEG_INFINITY);
    let mut index = 0;
    while index + 4 <= values.len() {
        let current = unsafe { vld1q_f32(values.as_ptr().add(index)) };
        minimum = vminq_f32(minimum, current);
        maximum = vmaxq_f32(maximum, current);
        index += 4;
    }
    let mut minimum_scalar = vminvq_f32(minimum);
    let mut maximum_scalar = vmaxvq_f32(maximum);
    while index < values.len() {
        minimum_scalar = minimum_scalar.min(values[index]);
        maximum_scalar = maximum_scalar.max(values[index]);
        index += 1;
    }
    (minimum_scalar, maximum_scalar)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn copy_f16_to_f32_neon(input: &[u16], output: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut index = 0;
    while index + 4 <= input.len() {
        let bits = unsafe { vld1_u16(input.as_ptr().add(index)) };
        let values = vcvt_f32_f16(vreinterpret_f16_u16(bits));
        unsafe {
            vst1q_f32(output.as_mut_ptr().add(index), values);
        }
        index += 4;
    }
    while index < input.len() {
        output[index] = f16::from_bits(input[index]).to_f32();
        index += 1;
    }
}

fn pack_row(values: &[f32], zero: f32, scale: f32, bits: u8, output: &mut Vec<u8>) -> (f64, f64) {
    let pack = 8 / bits as usize;
    let mask = (1u8 << bits) - 1;
    let mut signal_energy = 0.0f64;
    let mut error_energy = 0.0f64;
    debug_assert_eq!(values.len() % pack, 0);
    for chunk in values.chunks_exact(pack) {
        let mut packed = 0u8;
        for (index, &value) in chunk.iter().enumerate() {
            let quantized = ((value - zero) / scale)
                .round_ties_even()
                .clamp(0.0, mask as f32) as u8;
            packed |= quantized << (index * bits as usize);
            let reconstructed = quantized as f32 * scale + zero;
            let error = reconstructed - value;
            signal_energy += (value as f64) * (value as f64);
            error_energy += (error as f64) * (error as f64);
        }
        output.push(packed);
    }
    (signal_energy, error_energy)
}

pub fn normalized_hadamard_in_place(values: &mut [f32]) {
    let mut offset = 0;
    while offset < values.len() {
        let block_len = largest_power_of_two_at_most(values.len() - offset);
        normalized_power_of_two_hadamard_in_place(&mut values[offset..offset + block_len]);
        offset += block_len;
    }
}

fn largest_power_of_two_at_most(value: usize) -> usize {
    debug_assert!(value > 0);
    1usize << (usize::BITS - 1 - value.leading_zeros())
}

fn normalized_power_of_two_hadamard_in_place(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON, and the helper only reads/writes
        // complete four-lane chunks inside `values`.
        unsafe {
            normalized_hadamard_neon(values);
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        normalized_hadamard_scalar(values);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn normalized_hadamard_scalar(values: &mut [f32]) {
    let mut span = 1;
    while span < values.len() {
        for base in (0..values.len()).step_by(span * 2) {
            for offset in 0..span {
                let left = values[base + offset];
                let right = values[base + span + offset];
                values[base + offset] = left + right;
                values[base + span + offset] = left - right;
            }
        }
        span *= 2;
    }
    let normalization = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value *= normalization;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn normalized_hadamard_neon(values: &mut [f32]) {
    use std::arch::aarch64::*;

    let mut span = 1;
    while span < values.len() {
        for base in (0..values.len()).step_by(span * 2) {
            let mut offset = 0;
            while offset + 4 <= span {
                // SAFETY: base + 2*span never exceeds values.len().
                let left = unsafe { vld1q_f32(values.as_ptr().add(base + offset)) };
                let right = unsafe { vld1q_f32(values.as_ptr().add(base + span + offset)) };
                unsafe {
                    vst1q_f32(
                        values.as_mut_ptr().add(base + offset),
                        vaddq_f32(left, right),
                    );
                    vst1q_f32(
                        values.as_mut_ptr().add(base + span + offset),
                        vsubq_f32(left, right),
                    );
                }
                offset += 4;
            }
            while offset < span {
                let left = values[base + offset];
                let right = values[base + span + offset];
                values[base + offset] = left + right;
                values[base + span + offset] = left - right;
                offset += 1;
            }
        }
        span *= 2;
    }
    let normalization = 1.0 / (values.len() as f32).sqrt();
    let scale = vdupq_n_f32(normalization);
    let mut offset = 0;
    while offset + 4 <= values.len() {
        let current = unsafe { vld1q_f32(values.as_ptr().add(offset)) };
        unsafe {
            vst1q_f32(values.as_mut_ptr().add(offset), vmulq_f32(current, scale));
        }
        offset += 4;
    }
    while offset < values.len() {
        values[offset] *= normalization;
        offset += 1;
    }
}

#[derive(Debug)]
struct SoftmaxState {
    maximum: f32,
    sum: f32,
    numerator: Vec<f32>,
}

impl SoftmaxState {
    fn new(head_dim: usize) -> Self {
        Self {
            maximum: f32::NEG_INFINITY,
            sum: 0.0,
            numerator: vec![0.0; head_dim],
        }
    }

    fn valid(&self) -> bool {
        self.maximum.is_finite()
    }

    fn weight(&mut self, score: f32) -> f32 {
        if score > self.maximum {
            if self.valid() {
                let rescale = (self.maximum - score).exp();
                self.sum *= rescale;
                for value in &mut self.numerator {
                    *value *= rescale;
                }
            }
            self.maximum = score;
            self.sum += 1.0;
            1.0
        } else {
            let weight = (score - self.maximum).exp();
            self.sum += weight;
            weight
        }
    }

    fn merge(mut self, other: Self) -> Self {
        if !self.valid() {
            return other;
        }
        if !other.valid() {
            return self;
        }
        let maximum = self.maximum.max(other.maximum);
        let left_scale = (self.maximum - maximum).exp();
        let right_scale = (other.maximum - maximum).exp();
        self.sum = self.sum * left_scale + other.sum * right_scale;
        for dim in 0..self.numerator.len() {
            self.numerator[dim] =
                self.numerator[dim] * left_scale + other.numerator[dim] * right_scale;
        }
        self.maximum = maximum;
        self
    }
}

#[allow(clippy::too_many_arguments)]
fn process_f16_region(
    query: &[f32],
    key: &[u16],
    value: &[u16],
    global_start: usize,
    row_count: usize,
    row_width: usize,
    kv_head: usize,
    head_dim: usize,
    kv_len: usize,
    window_start: usize,
    scale: f32,
    softcap: Option<f32>,
) -> SoftmaxState {
    let mut state = SoftmaxState::new(head_dim);
    for row in 0..row_count {
        let position = global_start + row;
        if position < window_start || position >= kv_len {
            continue;
        }
        let offset = row * row_width + kv_head * head_dim;
        let mut score = dot_f32_f16(query, &key[offset..offset + head_dim]);
        score *= scale;
        if let Some(cap) = softcap {
            score = cap * (score / cap).tanh();
        }
        let weight = state.weight(score);
        axpy_f16(
            &mut state.numerator,
            &value[offset..offset + head_dim],
            weight,
        );
    }
    state
}

fn dot_f32_f16(left: &[f32], right: &[u16]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("fp16") {
        // SAFETY: runtime feature detection guarantees FP16 conversion support.
        return unsafe { dot_f32_f16_neon(left, right) };
    }
    left.iter()
        .zip(right)
        .map(|(&lhs, &rhs)| lhs * f16::from_bits(rhs).to_f32())
        .sum()
}

fn axpy_f16(output: &mut [f32], input: &[u16], scale: f32) {
    debug_assert_eq!(output.len(), input.len());
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("fp16") {
        // SAFETY: runtime feature detection guarantees FP16 conversion support.
        unsafe {
            axpy_f16_neon(output, input, scale);
        }
        return;
    }
    for (destination, &source) in output.iter_mut().zip(input) {
        *destination += scale * f16::from_bits(source).to_f32();
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn dot_f32_f16_neon(left: &[f32], right: &[u16]) -> f32 {
    use std::arch::aarch64::*;

    let mut accumulator = vdupq_n_f32(0.0);
    let mut index = 0;
    while index + 4 <= left.len() {
        let lhs = unsafe { vld1q_f32(left.as_ptr().add(index)) };
        let rhs_bits = unsafe { vld1_u16(right.as_ptr().add(index)) };
        let rhs = vcvt_f32_f16(vreinterpret_f16_u16(rhs_bits));
        accumulator = vfmaq_f32(accumulator, lhs, rhs);
        index += 4;
    }
    let mut sum = vaddvq_f32(accumulator);
    while index < left.len() {
        sum += left[index] * f16::from_bits(right[index]).to_f32();
        index += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn axpy_f16_neon(output: &mut [f32], input: &[u16], scale: f32) {
    use std::arch::aarch64::*;

    let mut index = 0;
    while index + 4 <= output.len() {
        let current = unsafe { vld1q_f32(output.as_ptr().add(index)) };
        let input_bits = unsafe { vld1_u16(input.as_ptr().add(index)) };
        let values = vcvt_f32_f16(vreinterpret_f16_u16(input_bits));
        unsafe {
            vst1q_f32(
                output.as_mut_ptr().add(index),
                vfmaq_n_f32(current, values, scale),
            );
        }
        index += 4;
    }
    while index < output.len() {
        output[index] += scale * f16::from_bits(input[index]).to_f32();
        index += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn process_quantized_block(
    rotated_query: &[f32],
    block: &KvarnBlock,
    block_start: usize,
    kv_head: usize,
    kv_len: usize,
    window_start: usize,
    scale: f32,
    softcap: Option<f32>,
    state: &mut SoftmaxState,
) {
    let group = block.config.group;
    let head_dim = block.head_dim;
    let local_start = window_start.saturating_sub(block_start).min(group);
    let local_end = kv_len.saturating_sub(block_start).min(group);
    if local_start >= local_end {
        return;
    }

    let packed_key_row = group / 2;
    let packed_key_head = head_dim * packed_key_row;
    let key_base = kv_head * packed_key_head;
    let key_scale_base = kv_head * head_dim;
    let token_scale_base = kv_head * group;
    let mut scores = vec![0.0f32; group];
    let mut zero_dot = 0.0f32;
    for dim in 0..head_dim {
        let key_scale = f16::from_bits(block.key_scale[key_scale_base + dim]).to_f32();
        let key_zero = f16::from_bits(block.key_zero[key_scale_base + dim]).to_f32();
        let query_value = rotated_query[dim];
        zero_dot += query_value * key_zero;
        let scaled_query = query_value * key_scale;
        let packed_row = &block.key_packed
            [key_base + dim * packed_key_row..key_base + (dim + 1) * packed_key_row];
        accumulate_q4_scores(
            &mut scores,
            packed_row,
            local_start,
            local_end,
            scaled_query,
        );
    }

    let value_bits = block.config.value_bits as usize;
    let value_pack = 8 / value_bits;
    let packed_value_row = head_dim / value_pack;
    let packed_value_head = group * packed_value_row;
    let value_base = kv_head * packed_value_head;
    let channel_scale_base = kv_head * head_dim;
    let value_row_base = kv_head * group;

    for token in local_start..local_end {
        let token_scale = f16::from_bits(block.key_token_scale[token_scale_base + token]).to_f32();
        let mut score = (scores[token] + zero_dot) * token_scale * scale;
        if let Some(cap) = softcap {
            score = cap * (score / cap).tanh();
        }
        let weight = state.weight(score);
        let value_scale = f16::from_bits(block.value_token_scale[value_row_base + token]).to_f32();
        let value_zero = f16::from_bits(block.value_zero[value_row_base + token]).to_f32();
        let packed_row = &block.value_packed
            [value_base + token * packed_value_row..value_base + (token + 1) * packed_value_row];
        accumulate_quantized_value(
            &mut state.numerator,
            packed_row,
            value_bits,
            &block.value_channel_scale[channel_scale_base..channel_scale_base + head_dim],
            value_scale,
            value_zero,
            weight,
        );
    }
}

fn accumulate_quantized_value(
    output: &mut [f32],
    packed: &[u8],
    bits: usize,
    channel_scale: &[u16],
    value_scale: f32,
    value_zero: f32,
    weight: f32,
) {
    debug_assert!(matches!(bits, 2 | 4));
    debug_assert_eq!(output.len(), channel_scale.len());
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("fp16") {
        // SAFETY: runtime feature detection guarantees FP16 conversion support.
        unsafe {
            accumulate_quantized_value_neon(
                output,
                packed,
                bits,
                channel_scale,
                value_scale,
                value_zero,
                weight,
            );
        }
        return;
    }
    let pack = 8 / bits;
    let mask = (1u8 << bits) - 1;
    for dim in 0..output.len() {
        let quantized = (packed[dim / pack] >> ((dim % pack) * bits)) & mask;
        let channel = f16::from_bits(channel_scale[dim]).to_f32();
        output[dim] += weight * (quantized as f32 * value_scale + value_zero) * channel;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn accumulate_quantized_value_neon(
    output: &mut [f32],
    packed: &[u8],
    bits: usize,
    channel_scale: &[u16],
    value_scale: f32,
    value_zero: f32,
    weight: f32,
) {
    use std::arch::aarch64::*;

    let mut dim = 0;
    while dim + 16 <= output.len() {
        let (first, second) = if bits == 4 {
            let bytes = unsafe { vld1_u8(packed.as_ptr().add(dim / 2)) };
            let low = vand_u8(bytes, vdup_n_u8(0x0f));
            let high = vshr_n_u8::<4>(bytes);
            (vzip1_u8(low, high), vzip2_u8(low, high))
        } else {
            let mut unpacked = [0u8; 16];
            for index in 0..16 {
                let packed_value = packed[(dim + index) / 4];
                unpacked[index] = (packed_value >> (((dim + index) % 4) * 2)) & 0x03;
            }
            (unsafe { vld1_u8(unpacked.as_ptr()) }, unsafe {
                vld1_u8(unpacked.as_ptr().add(8))
            })
        };
        unsafe {
            accumulate_quantized_u8x8(
                output.as_mut_ptr().add(dim),
                channel_scale.as_ptr().add(dim),
                first,
                value_scale,
                value_zero,
                weight,
            );
            accumulate_quantized_u8x8(
                output.as_mut_ptr().add(dim + 8),
                channel_scale.as_ptr().add(dim + 8),
                second,
                value_scale,
                value_zero,
                weight,
            );
        }
        dim += 16;
    }
    let pack = 8 / bits;
    let mask = (1u8 << bits) - 1;
    while dim < output.len() {
        let quantized = (packed[dim / pack] >> ((dim % pack) * bits)) & mask;
        let channel = f16::from_bits(channel_scale[dim]).to_f32();
        output[dim] += weight * (quantized as f32 * value_scale + value_zero) * channel;
        dim += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn accumulate_quantized_u8x8(
    output: *mut f32,
    channel_scale: *const u16,
    values: std::arch::aarch64::uint8x8_t,
    value_scale: f32,
    value_zero: f32,
    weight: f32,
) {
    use std::arch::aarch64::*;

    let widened = vmovl_u8(values);
    let quantized_first = vcvtq_f32_u32(vmovl_u16(vget_low_u16(widened)));
    let quantized_second = vcvtq_f32_u32(vmovl_u16(vget_high_u16(widened)));
    let channel_first_bits = unsafe { vld1_u16(channel_scale) };
    let channel_second_bits = unsafe { vld1_u16(channel_scale.add(4)) };
    let channel_first = vcvt_f32_f16(vreinterpret_f16_u16(channel_first_bits));
    let channel_second = vcvt_f32_f16(vreinterpret_f16_u16(channel_second_bits));
    let value_first = vmulq_f32(
        vfmaq_n_f32(vdupq_n_f32(value_zero), quantized_first, value_scale),
        channel_first,
    );
    let value_second = vmulq_f32(
        vfmaq_n_f32(vdupq_n_f32(value_zero), quantized_second, value_scale),
        channel_second,
    );
    let output_first = unsafe { vld1q_f32(output) };
    let output_second = unsafe { vld1q_f32(output.add(4)) };
    unsafe {
        vst1q_f32(output, vfmaq_n_f32(output_first, value_first, weight));
        vst1q_f32(
            output.add(4),
            vfmaq_n_f32(output_second, value_second, weight),
        );
    }
}

fn accumulate_q4_scores(scores: &mut [f32], packed: &[u8], start: usize, end: usize, scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON; the helper bounds every 16-token load.
        unsafe {
            accumulate_q4_scores_neon(scores, packed, start, end, scale);
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for token in start..end {
            let byte = packed[token / 2];
            let quantized = if token & 1 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            scores[token] += scale * quantized as f32;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_q4_scores_neon(
    scores: &mut [f32],
    packed: &[u8],
    start: usize,
    end: usize,
    scale: f32,
) {
    use std::arch::aarch64::*;

    let mut token = start;
    while token < end && token % 16 != 0 {
        let byte = packed[token / 2];
        let quantized = if token & 1 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        };
        scores[token] += scale * quantized as f32;
        token += 1;
    }
    while token + 16 <= end {
        let bytes = unsafe { vld1_u8(packed.as_ptr().add(token / 2)) };
        let low = vand_u8(bytes, vdup_n_u8(0x0f));
        let high = vshr_n_u8::<4>(bytes);
        let first = vzip1_u8(low, high);
        let second = vzip2_u8(low, high);
        unsafe {
            accumulate_u8x8_f32(scores.as_mut_ptr().add(token), first, scale);
            accumulate_u8x8_f32(scores.as_mut_ptr().add(token + 8), second, scale);
        }
        token += 16;
    }
    while token < end {
        let byte = packed[token / 2];
        let quantized = if token & 1 == 0 {
            byte & 0x0f
        } else {
            byte >> 4
        };
        scores[token] += scale * quantized as f32;
        token += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_u8x8_f32(output: *mut f32, values: std::arch::aarch64::uint8x8_t, scale: f32) {
    use std::arch::aarch64::*;

    let widened = vmovl_u8(values);
    let first = vcvtq_f32_u32(vmovl_u16(vget_low_u16(widened)));
    let second = vcvtq_f32_u32(vmovl_u16(vget_high_u16(widened)));
    let output_first = unsafe { vld1q_f32(output) };
    let output_second = unsafe { vld1q_f32(output.add(4)) };
    unsafe {
        vst1q_f32(output, vfmaq_n_f32(output_first, first, scale));
        vst1q_f32(output.add(4), vfmaq_n_f32(output_second, second, scale));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(rows: usize, width: usize, phase: f32) -> Vec<u16> {
        (0..rows * width)
            .map(|index| {
                let x = index as f32;
                f16::from_f32((x * 0.017 + phase).sin() * 0.7 + (x * 0.003).cos() * 0.2).to_bits()
            })
            .collect()
    }

    #[test]
    fn normalized_hadamard_is_self_inverse() {
        let source = (0..256)
            .map(|index| (index as f32 * 0.11).sin())
            .collect::<Vec<_>>();
        let mut transformed = source.clone();
        normalized_hadamard_in_place(&mut transformed);
        normalized_hadamard_in_place(&mut transformed);
        let maximum_error = source
            .iter()
            .zip(transformed.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 2.0e-6, "maximum_error={maximum_error}");
    }

    #[test]
    fn all_supported_kvarn_modes_round_trip_with_finite_quality_metrics() {
        for config in [
            KvarnConfig::K4_V2_G64,
            KvarnConfig::K4_V4_G64,
            KvarnConfig::K4_V2_G128,
            KvarnConfig::K4_V4_G128,
        ] {
            let heads = 1;
            let head_dim = 80;
            let key = fixture(config.group, head_dim, 0.3);
            let value = fixture(config.group, head_dim, 1.1);
            let block = KvarnBlock::quantize(config, heads, head_dim, &key, &value).unwrap();
            let (restored_key, restored_value) = block.dequantize_f16();
            let ((key_signal, key_error), (value_signal, value_error)) =
                block.quantization_energy();

            assert_eq!(restored_key.len(), key.len(), "{}", config.label());
            assert_eq!(restored_value.len(), value.len(), "{}", config.label());
            assert!(key_signal > key_error, "{}", config.label());
            assert!(value_signal > value_error, "{}", config.label());
            assert!(cosine_f16(&key, &restored_key) > 0.98, "{}", config.label());
            let value_threshold = if config.value_bits == 2 { 0.88 } else { 0.98 };
            assert!(
                cosine_f16(&value, &restored_value) > value_threshold,
                "{}",
                config.label()
            );
        }
    }

    #[test]
    fn k4v2_block_round_trip_has_expected_layout_and_quality() {
        let config = KvarnConfig::K4_V2_G128;
        let heads = 2;
        let head_dim = 256;
        let width = heads * head_dim;
        let key = fixture(config.group, width, 0.2);
        let value = fixture(config.group, width, 0.7);
        let block = KvarnBlock::quantize(config, heads, head_dim, &key, &value).unwrap();
        assert_eq!(block.key_packed.len(), heads * head_dim * config.group / 2);
        assert_eq!(
            block.value_packed.len(),
            heads * config.group * head_dim / 4
        );
        let (restored_key, restored_value) = block.dequantize_f16();
        assert_eq!(restored_key.len(), key.len());
        assert_eq!(restored_value.len(), value.len());
        let key_cosine = cosine_f16(&key, &restored_key);
        let value_cosine = cosine_f16(&value, &restored_value);
        assert!(key_cosine > 0.99, "key_cosine={key_cosine}");
        assert!(value_cosine > 0.90, "value_cosine={value_cosine}");
        assert!(block.logical_bytes_per_token() < width * 4);
    }

    #[test]
    fn fused_decode_matches_materialized_kvarn_cache() {
        for config in [
            KvarnConfig::K4_V2_G64,
            KvarnConfig::K4_V4_G64,
            KvarnConfig::K4_V2_G128,
            KvarnConfig::K4_V4_G128,
        ] {
            let kv_heads = 2;
            let query_heads = 4;
            let head_dim = 64;
            let width = kv_heads * head_dim;
            let sink = 16;
            let tail = 7;
            let sink_key = fixture(sink, width, 0.1);
            let sink_value = fixture(sink, width, 0.4);
            let block_key = fixture(config.group, width, 0.8);
            let block_value = fixture(config.group, width, 1.2);
            let block =
                KvarnBlock::quantize(config, kv_heads, head_dim, &block_key, &block_value).unwrap();
            let tail_key = fixture(tail, width, 1.8);
            let tail_value = fixture(tail, width, 2.2);
            let len = sink + config.group + tail;
            let query = (0..query_heads * head_dim)
                .map(|index| (index as f32 * 0.031).cos())
                .collect::<Vec<_>>();
            let device_layout = KvarnDeviceRecordLayout::new(config, kv_heads, head_dim).unwrap();
            let mut device_blocks = Vec::with_capacity(device_layout.block_bytes);
            block.append_device_record(&mut device_blocks);
            let view = KvarnKvView {
                config: KvarnConfig {
                    sink_tokens: sink,
                    ..config
                },
                num_kv_heads: kv_heads,
                head_dim,
                sink_key: &sink_key,
                sink_value: &sink_value,
                blocks: std::slice::from_ref(&block),
                device_layout,
                device_blocks: &device_blocks,
                tail_start: sink + config.group,
                tail_key: &tail_key,
                tail_value: &tail_value,
                len,
            };
            let mut actual = vec![0.0f32; query.len()];
            attention_decode(
                &query,
                view,
                &mut actual,
                query_heads,
                1.0 / (head_dim as f32).sqrt(),
                None,
                None,
            );

            let (dequant_key, dequant_value) = block.dequantize_f16();
            let mut full_key = sink_key.clone();
            full_key.extend(dequant_key);
            full_key.extend_from_slice(&tail_key);
            let mut full_value = sink_value.clone();
            full_value.extend(dequant_value);
            full_value.extend_from_slice(&tail_value);
            let mut expected = vec![0.0f32; query.len()];
            crate::kernels::attention::attention_decode_flash(
                &query,
                &full_key,
                &full_value,
                &mut expected,
                query_heads,
                kv_heads,
                head_dim,
                len,
                1.0 / (head_dim as f32).sqrt(),
                None,
                None,
            );
            let maximum_error = actual
                .iter()
                .zip(expected.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            assert!(maximum_error < 3.0e-3, "maximum_error={maximum_error}");
        }
    }

    fn cosine_f16(left: &[u16], right: &[u16]) -> f32 {
        let mut dot = 0.0f64;
        let mut left_norm = 0.0f64;
        let mut right_norm = 0.0f64;
        for (&left, &right) in left.iter().zip(right.iter()) {
            let left = f16::from_bits(left).to_f32() as f64;
            let right = f16::from_bits(right).to_f32() as f64;
            dot += left * right;
            left_norm += left * left;
            right_norm += right * right;
        }
        (dot / (left_norm.sqrt() * right_norm.sqrt())) as f32
    }
}
