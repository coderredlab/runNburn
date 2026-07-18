//! Sidecar `.rnb` v3 encoder.
//!
//! Dense scope: dense Q4_K (row-pair packed via `pack_q4k`) and dense Q6_K.
//! MoE Q4_K hot/cold entries keep raw GGUF expert bytes plus tier metadata.
//! Other quants (Q5_K, Q8_0, F32, F16, BF16) are skipped — engine reads those
//! from the source GGUF directly.
//!
//! Sidecar layout owned by [`rnb_loader::sidecar_v3::spec`].

use memmap2::Mmap;
use rnb_cpu::gemm::pack_q4k::pack_q4k;
use rnb_cpu::gemm::pack_q6k::pack_q6k;
use rnb_loader::arch::extract_metadata;
use rnb_loader::gguf::parser::GGUFFile;
use rnb_loader::gguf::types::GGMLType;
use rnb_loader::sidecar_v3::spec::{
    HeaderV3, LayerPopularityEntry, MetadataV3, TransformedLayoutDescriptorV1,
    TransformedLayoutKind, V3QuantType, FORMAT_VERSION_V3, HEADER_SIZE_BYTES, MAGIC,
};
use sha2::{Digest, Sha256};

use crate::arch_filter::{classify, ConvertDecision};

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Q4_K block byte size on disk in GGUF (matches `rnb_cpu::gemm::repack`).
const Q4K_BLOCK_BYTES: usize = 144;
/// Q6_K block byte size on disk in GGUF (matches `rnb_cpu::gemm::pack_q6k`).
const Q6K_BLOCK_BYTES: usize = 210;
/// Element count per K-quant block.
const K_QUANT_BLOCK_ELEMS: usize = 256;
const TRANSFORMED_LAYOUT_PRODUCER_OPTIONS_HASH: u64 = 0x1;

fn source_fingerprint64(bytes: &[u8]) -> u64 {
    let digest = Sha256::digest(bytes);
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn transformed_layout_descriptor_for_dense_source(
    tensor_name: &str,
    source_quant: V3QuantType,
    rows: usize,
    cols: usize,
    source: &[u8],
) -> Result<Option<TransformedLayoutDescriptorV1>, String> {
    let (layout_kind, block_bytes) = match source_quant {
        V3QuantType::DenseQ4kRowPair => {
            (TransformedLayoutKind::Q4kCompactMetadata, Q4K_BLOCK_BYTES)
        }
        V3QuantType::DenseQ6k => (TransformedLayoutKind::Q6kPackedQ8dot, Q6K_BLOCK_BYTES),
        V3QuantType::MoeQ4kHot | V3QuantType::MoeQ4kCold => return Ok(None),
    };
    if cols % K_QUANT_BLOCK_ELEMS != 0 {
        return Ok(None);
    }
    let blocks_per_row = cols / K_QUANT_BLOCK_ELEMS;
    let expected_len = rows
        .checked_mul(blocks_per_row)
        .and_then(|v| v.checked_mul(block_bytes))
        .ok_or_else(|| format!("transformed descriptor size overflow: rows={rows} cols={cols}"))?;
    if source.len() != expected_len {
        return Err(format!(
            "transformed descriptor byte mismatch for {tensor_name}: got {}, expected {expected_len}",
            source.len()
        ));
    }
    Ok(Some(TransformedLayoutDescriptorV1 {
        tensor_name: tensor_name.to_string(),
        source_quant,
        rows: rows as u64,
        cols: cols as u64,
        source_len: source.len() as u64,
        source_fingerprint: source_fingerprint64(source),
        layout_kind,
        layout_version: layout_kind.current_version(),
        block_rows: 1,
        block_cols: K_QUANT_BLOCK_ELEMS as u32,
        producer_options_hash: TRANSFORMED_LAYOUT_PRODUCER_OPTIONS_HASH,
    }))
}

/// Encoder options for sidecar v3.
///
/// Phase 1 fixes the option bundle to spec §3.3 option B:
/// dense Q4_K row-pair + dense Q6_K + (MoE Q4_K hot/cold — follow-up task).
/// The unit struct keeps the signature forward-compatible so that adding
/// toggles in Phase 2 (Q5_K, output Q8 tile8, ...) does not churn callers.
///
/// Phase 2 entry point: `moe_hot_count`. When `Some(n)`, MoE expert tensors
/// are split into a hot block (first n experts in popularity order) tagged
/// `MoeQ4kHot` and a cold block (remaining experts) tagged `MoeQ4kCold`.
/// When `None`, every expert goes into a single `MoeQ4kHot` entry — phase 1
/// minimal behaviour preserved as default.
///
/// `popularity_order` (when present) gives the expert indices in
/// most-popular-first order; the hot block keeps the first `moe_hot_count`
/// of those. When absent, expert indices are used in their natural order
/// (idx 0 → hot first), which is naive but stable.
#[derive(Debug, Clone, Default)]
pub struct EncoderOptions {
    pub moe_hot_count: Option<usize>,
    pub popularity_order: Option<Vec<u32>>,
    /// in4: per-layer popularity + hot_count overrides. When `Some`, the
    /// encoder picks each MoE layer's hot/cold split using the matching
    /// `LayerPopularityEntry`; layers without an entry fall back to
    /// `moe_hot_count` + `popularity_order`. The cli's `--popularity
    /// <file.json>` flag derives this from `rnb-moe-profile` `hit_counts`.
    pub popularity_per_layer: Option<Vec<LayerPopularityEntry>>,
}

/// Encode a sidecar v3 `.rnb` for the given GGUF.
///
/// Errors:
/// - GGUF open / mmap / parse failure.
/// - Architecture is on the convert blacklist (`ConvertDecision::Skip`).
/// - Output write failure.
pub fn encode_sidecar_v3(
    gguf_path: &Path,
    out_path: &Path,
    opts: EncoderOptions,
) -> Result<(), String> {
    let f = File::open(gguf_path).map_err(|e| format!("open gguf {}: {e}", gguf_path.display()))?;
    // SAFETY: read-only mapping; lifetime tied to this function.
    let mmap =
        unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap gguf {}: {e}", gguf_path.display()))?;

    let gguf = GGUFFile::parse(&mmap[..]).map_err(|e| format!("parse gguf: {e}"))?;
    let metadata =
        extract_metadata(&gguf.metadata).map_err(|e| format!("extract metadata: {e}"))?;
    let arch = metadata.architecture;
    let tensor_names: Vec<&str> = gguf.tensor_infos.iter().map(|t| t.name.as_str()).collect();
    if classify(arch, &tensor_names) == ConvertDecision::Skip {
        return Err(format!(
            "arch {arch:?} skipped by arch_filter — convert is a no-op for this model"
        ));
    }

    // Pass 1: select tensors and pack their payloads in memory.
    struct PackedEntry {
        name: String,
        shape: Vec<u64>,
        quant_type: V3QuantType,
        bytes: Vec<u8>,
    }
    let mut packed: Vec<PackedEntry> = Vec::new();
    let mut transformed_layouts: Vec<TransformedLayoutDescriptorV1> = Vec::new();

    for tensor in &gguf.tensor_infos {
        let role = classify_tensor_role(&tensor.name);
        if role.is_none() {
            continue;
        }
        let role = role.unwrap();
        // Dense path = 2D [rows, in_features]. MoE expert path = 3D
        // [n_experts, rows, in_features]. We pack each expert as one logical
        // payload — the sidecar entry's shape preserves the GGUF 3D shape so
        // the engine can fan out at load time.
        let (rows, in_features, n_experts) = match role {
            TensorRole::Dense => {
                if tensor.shape.len() != 2 {
                    continue;
                }
                let r = tensor.shape[0];
                let c = tensor.shape[1];
                (r, c, 1usize)
            }
            TensorRole::MoeExperts => {
                if tensor.shape.len() != 3 {
                    continue;
                }
                let r = tensor.shape[1];
                let c = tensor.shape[2];
                (r, c, tensor.shape[0])
            }
        };
        if in_features % K_QUANT_BLOCK_ELEMS != 0 {
            continue;
        }
        let cols_in_blocks = in_features / K_QUANT_BLOCK_ELEMS;
        // Phase 1 MoE scope is Q4_K only. Q5_K / Q6_K MoE expert tensors
        // (e.g. Qwen3.6 down_exps) stay raw — engine reads them from GGUF.
        let allowed_for_role = match role {
            TensorRole::Dense => matches!(tensor.ggml_type, GGMLType::Q4_K | GGMLType::Q6_K),
            TensorRole::MoeExperts => matches!(tensor.ggml_type, GGMLType::Q4_K),
        };
        if !allowed_for_role {
            continue;
        }
        let block_bytes = match tensor.ggml_type {
            GGMLType::Q4_K => Q4K_BLOCK_BYTES,
            GGMLType::Q6_K => Q6K_BLOCK_BYTES,
            _ => continue,
        };
        let per_expert_raw_len = rows * cols_in_blocks * block_bytes;
        let raw_byte_len = n_experts * per_expert_raw_len;
        let absolute_offset = gguf.data_start + tensor.offset as usize;
        if absolute_offset + raw_byte_len > mmap.len() {
            return Err(format!(
                "tensor {} bytes [{}, +{}] beyond mmap len {}",
                tensor.name,
                absolute_offset,
                raw_byte_len,
                mmap.len()
            ));
        }
        let (quant_type, bytes) = match (role, tensor.ggml_type) {
            (TensorRole::Dense, GGMLType::Q4_K) => {
                let raw = &mmap[absolute_offset..absolute_offset + per_expert_raw_len];
                let quant_type = V3QuantType::DenseQ4kRowPair;
                if let Some(descriptor) = transformed_layout_descriptor_for_dense_source(
                    &tensor.name,
                    quant_type,
                    rows,
                    in_features,
                    raw,
                )? {
                    transformed_layouts.push(descriptor);
                }
                (quant_type, pack_q4k(raw, rows, cols_in_blocks))
            }
            (TensorRole::Dense, GGMLType::Q6_K) => {
                let raw = &mmap[absolute_offset..absolute_offset + per_expert_raw_len];
                let quant_type = V3QuantType::DenseQ6k;
                if let Some(descriptor) = transformed_layout_descriptor_for_dense_source(
                    &tensor.name,
                    quant_type,
                    rows,
                    in_features,
                    raw,
                )? {
                    transformed_layouts.push(descriptor);
                }
                (quant_type, pack_q6k(raw, rows, cols_in_blocks))
            }
            (TensorRole::MoeExperts, GGMLType::Q4_K) => {
                // Optional popularity-aware hot/cold split. Resolution order:
                //   1. `popularity_per_layer` entry whose `layer_idx` matches
                //      this tensor's `blk.{N}` segment — layer-specific.
                //   2. global `popularity_order` + `moe_hot_count` — same
                //      values across every MoE layer.
                //   3. natural index order + all-hot — phase 1 minimal default.
                //
                // The two `MoeQ4k{Hot,Cold}` entries share the tensor name +
                // 3D shape; payload contains only that tier's experts in
                // `popularity_order` order.
                let layer_idx_opt = parse_blk_idx(&tensor.name);
                let layer_entry = opts.popularity_per_layer.as_ref().and_then(|pl| {
                    let li = layer_idx_opt?;
                    pl.iter().find(|e| e.layer_idx == li)
                });
                let order: Vec<u32> = if let Some(entry) = layer_entry {
                    entry.popularity_order.clone()
                } else {
                    match &opts.popularity_order {
                        Some(o) if !o.is_empty() => o.clone(),
                        _ => (0..n_experts as u32).collect(),
                    }
                };
                if order.len() != n_experts {
                    return Err(format!(
                        "popularity_order has {} indices but tensor {} has {} experts",
                        order.len(),
                        tensor.name,
                        n_experts
                    ));
                }
                let hot_count = if let Some(entry) = layer_entry {
                    let n = entry.hot_count as usize;
                    if n > n_experts {
                        return Err(format!(
                            "popularity_per_layer hot_count={n} exceeds n_experts={n_experts} for tensor {}",
                            tensor.name
                        ));
                    }
                    n
                } else {
                    match opts.moe_hot_count {
                        Some(n) if n <= n_experts => n,
                        Some(n) => {
                            return Err(format!(
                                "moe_hot_count={n} exceeds n_experts={n_experts} for tensor {}",
                                tensor.name
                            ));
                        }
                        None => n_experts, // everything hot — phase 1 minimal
                    }
                };

                // MoE expert bytes stay in the raw GGUF layout — engine
                // forward indexes them through `dot_k_block_row(_, _, _, _,
                // GGMLType::Q4_K)`, which assumes raw GGUF Q4_K block layout.
                // (Dense-FFN Q4_K uses `pack_q4k_compact` to drive the
                // packed-GEMM dispatch, but MoE forward goes through the
                // scalar block-row path and would mis-decode packed bytes.)
                let pack_subset = |range: std::ops::Range<usize>| -> Vec<u8> {
                    let mut out: Vec<u8> = Vec::with_capacity(range.len() * per_expert_raw_len);
                    for &idx in &order[range] {
                        let off = absolute_offset + (idx as usize) * per_expert_raw_len;
                        let raw = &mmap[off..off + per_expert_raw_len];
                        out.extend_from_slice(raw);
                    }
                    out
                };

                let hot_bytes = pack_subset(0..hot_count);
                let cold_bytes = if hot_count < n_experts {
                    Some(pack_subset(hot_count..n_experts))
                } else {
                    None
                };

                packed.push(PackedEntry {
                    name: tensor.name.clone(),
                    shape: tensor.shape.iter().map(|&d| d as u64).collect(),
                    quant_type: V3QuantType::MoeQ4kHot,
                    bytes: hot_bytes,
                });
                if let Some(cold) = cold_bytes {
                    packed.push(PackedEntry {
                        name: tensor.name.clone(),
                        shape: tensor.shape.iter().map(|&d| d as u64).collect(),
                        quant_type: V3QuantType::MoeQ4kCold,
                        bytes: cold,
                    });
                }
                continue;
            }
            _ => continue,
        };
        packed.push(PackedEntry {
            name: tensor.name.clone(),
            shape: tensor.shape.iter().map(|&d| d as u64).collect(),
            quant_type,
            bytes,
        });
    }

    // Build the optional metadata blob carrying per-MoE-layer popularity_order
    // + hot_count. Engine consumers read this through PackedModel and
    // MoeExpertResidencyView to drive hot/cold dispatch. We emit metadata when
    // the caller provided either an explicit hot_count or a popularity_order;
    // every MoE layer in the GGUF gets the same entry (single-shot mode).
    // `EncoderOptions::popularity_per_layer` carries diagnostic profile
    // overrides supplied by the caller.
    let mut metadata = build_metadata(&gguf, &opts)?;
    metadata.transformed_layouts = transformed_layouts;
    let metadata_bytes = metadata.to_bytes();
    let metadata_present = !metadata.is_empty();

    // Pass 2: stream the file. Layout: header | tensor_table | payload | metadata.
    // Metadata sits at the tail so the dense payload region is contiguous and
    // mmap madvise hints applied by `PackedModel::from_v3_sidecar` cover only
    // weight bytes, not metadata.
    //
    // Memory model: a `BufWriter` on the output file means we never hold the
    // full sidecar in memory at once (the legacy implementation did, which
    // matters once 26B+ models start producing >5 GB sidecars). Per-tensor
    // packed bytes already live in `packed`, but each one is dropped from
    // memory as the writer flushes its bytes to disk.
    let table_offset = HEADER_SIZE_BYTES as u64;
    let table_size: usize = packed
        .iter()
        .map(|e| 2 + e.name.len() + 1 + 4 + e.shape.len() * 8 + 8 + 8)
        .sum();
    let payload_offset = table_offset + table_size as u64;
    let total_payload_bytes: u64 = packed.iter().map(|e| e.bytes.len() as u64).sum();
    let metadata_offset: u64 = if metadata_present {
        payload_offset + total_payload_bytes
    } else {
        0
    };
    let metadata_size: u64 = if metadata_present {
        metadata_bytes.len() as u64
    } else {
        0
    };

    let f = File::create(out_path)
        .map_err(|e| format!("create sidecar {}: {e}", out_path.display()))?;
    let mut writer = BufWriter::with_capacity(1 << 20, f);

    let header = HeaderV3 {
        magic: MAGIC,
        version: FORMAT_VERSION_V3,
        tensor_count: packed.len() as u32,
        tensor_table_offset: table_offset,
        payload_offset,
        metadata_offset,
        metadata_size,
    };
    writer
        .write_all(&header.to_bytes())
        .map_err(|e| format!("write header: {e}"))?;

    let mut running_payload_offset = payload_offset;
    for e in &packed {
        writer
            .write_all(&(e.name.len() as u16).to_le_bytes())
            .map_err(|e| format!("write entry: {e}"))?;
        writer
            .write_all(e.name.as_bytes())
            .map_err(|e| format!("write entry: {e}"))?;
        writer
            .write_all(&[e.quant_type as u8])
            .map_err(|e| format!("write entry: {e}"))?;
        writer
            .write_all(&(e.shape.len() as u32).to_le_bytes())
            .map_err(|e| format!("write entry: {e}"))?;
        for d in &e.shape {
            writer
                .write_all(&d.to_le_bytes())
                .map_err(|e| format!("write entry: {e}"))?;
        }
        writer
            .write_all(&running_payload_offset.to_le_bytes())
            .map_err(|e| format!("write entry: {e}"))?;
        writer
            .write_all(&(e.bytes.len() as u64).to_le_bytes())
            .map_err(|e| format!("write entry: {e}"))?;
        running_payload_offset += e.bytes.len() as u64;
    }
    for e in &packed {
        writer
            .write_all(&e.bytes)
            .map_err(|e| format!("write payload: {e}"))?;
    }
    if metadata_present {
        writer
            .write_all(&metadata_bytes)
            .map_err(|e| format!("write metadata: {e}"))?;
    }
    writer.flush().map_err(|e| format!("flush sidecar: {e}"))?;
    Ok(())
}

/// Extract `blk.{N}` from a GGUF tensor name like `blk.5.ffn_gate_exps.weight`.
fn parse_blk_idx(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Walk the GGUF tensor list and pull out one `(layer_idx, n_experts)` per
/// MoE layer (deduped). Layers appear in ascending `layer_idx` order in the
/// returned vec.
fn collect_moe_layers(gguf: &GGUFFile) -> Vec<(u32, usize)> {
    let mut layers: Vec<(u32, usize)> = Vec::new();
    for tensor in &gguf.tensor_infos {
        if !tensor.name.contains(".ffn_") || !tensor.name.contains("_exps.weight") {
            continue;
        }
        if tensor.shape.len() != 3 {
            continue;
        }
        let layer_idx = match parse_blk_idx(&tensor.name) {
            Some(i) => i,
            None => continue,
        };
        let n_experts = tensor.shape[0] as usize;
        if !layers.iter().any(|(l, _)| *l == layer_idx) {
            layers.push((layer_idx, n_experts));
        }
    }
    layers.sort_by_key(|(l, _)| *l);
    layers
}

/// Build the per-layer metadata blob from `EncoderOptions`. Returns an empty
/// `MetadataV3` (which serializes to a 4-byte zero layer_count and is treated
/// as absent by the writer) when no popularity input was supplied.
fn build_metadata(gguf: &GGUFFile, opts: &EncoderOptions) -> Result<MetadataV3, String> {
    build_metadata_from_layers(&collect_moe_layers(gguf), opts)
}

fn build_metadata_from_layers(
    layers: &[(u32, usize)],
    opts: &EncoderOptions,
) -> Result<MetadataV3, String> {
    // popularity_per_layer takes precedence — it's the most explicit input.
    if let Some(per_layer) = &opts.popularity_per_layer {
        // Validate each entry's popularity_order length against the matching
        // GGUF layer (when known). Layers without a matching entry fall back
        // to the global popularity_order / moe_hot_count below.
        let mut entries: Vec<LayerPopularityEntry> = Vec::new();
        for entry in per_layer {
            let n_experts = layers
                .iter()
                .find(|(l, _)| *l == entry.layer_idx)
                .map(|(_, n)| *n);
            if let Some(n) = n_experts {
                if entry.popularity_order.len() != n {
                    return Err(format!(
                        "popularity_per_layer[{}].popularity_order has {} indices but layer has {n} experts",
                        entry.layer_idx,
                        entry.popularity_order.len()
                    ));
                }
                if (entry.hot_count as usize) > n {
                    return Err(format!(
                        "popularity_per_layer[{}].hot_count={} exceeds n_experts={n}",
                        entry.layer_idx, entry.hot_count
                    ));
                }
            }
            entries.push(entry.clone());
        }
        // Layers without a matching entry: fill from globals or natural order
        // so the metadata still covers every MoE layer (engine forward looks
        // up by layer_idx).
        for &(layer_idx, n_experts) in layers {
            if entries.iter().any(|e| e.layer_idx == layer_idx) {
                continue;
            }
            let popularity_order = match &opts.popularity_order {
                Some(o) if o.len() == n_experts => o.clone(),
                _ => (0..n_experts as u32).collect(),
            };
            let resolved_hot = match opts.moe_hot_count {
                Some(n) if n <= n_experts => n as u32,
                _ => n_experts as u32,
            };
            entries.push(LayerPopularityEntry {
                layer_idx,
                hot_count: resolved_hot,
                popularity_order,
            });
        }
        entries.sort_by_key(|e| e.layer_idx);
        return Ok(MetadataV3 {
            layers: entries,
            transformed_layouts: Vec::new(),
        });
    }

    let order: Option<Vec<u32>> = opts
        .popularity_order
        .as_ref()
        .filter(|o| !o.is_empty())
        .cloned();
    let hot_count = opts.moe_hot_count;
    if order.is_none() && hot_count.is_none() {
        return Ok(MetadataV3::default());
    }
    let mut entries = Vec::with_capacity(layers.len());
    for &(layer_idx, n_experts) in layers {
        let popularity_order = match &order {
            Some(o) => {
                if o.len() != n_experts {
                    return Err(format!(
                        "popularity_order has {} indices but layer {layer_idx} has {n_experts} experts",
                        o.len()
                    ));
                }
                o.clone()
            }
            None => (0..n_experts as u32).collect(),
        };
        let resolved_hot = match hot_count {
            Some(n) if n <= n_experts => n as u32,
            Some(n) => {
                return Err(format!(
                    "moe_hot_count={n} exceeds n_experts={n_experts} for layer {layer_idx}"
                ));
            }
            None => n_experts as u32,
        };
        entries.push(LayerPopularityEntry {
            layer_idx,
            hot_count: resolved_hot,
            popularity_order,
        });
    }
    Ok(MetadataV3 {
        layers: entries,
        transformed_layouts: Vec::new(),
    })
}

/// Role classification for the sidecar v3 encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorRole {
    /// Dense 2D weight (attention QKV/O, FFN gate/up/down).
    Dense,
    /// 3D MoE expert tensor (`*.ffn_*_exps.weight`). Phase 1 minimal: all
    /// experts are packed as Q4_K compact and tagged `MoeQ4kHot`. Phase 2
    /// will introduce popularity-based hot/cold split.
    MoeExperts,
}

fn classify_tensor_role(name: &str) -> Option<TensorRole> {
    if is_dense_eligible_name(name) {
        Some(TensorRole::Dense)
    } else if is_moe_expert_name(name) {
        Some(TensorRole::MoeExperts)
    } else {
        None
    }
}

/// Phase 1 dense weight name whitelist. Matches the standard llama.cpp /
/// GGUF tensor naming convention. MoE expert tensors (`*.ffn_*_exps.weight`)
/// are explicitly NOT in this set — see `is_moe_expert_name` for those.
fn is_dense_eligible_name(name: &str) -> bool {
    name.ends_with(".attn_q.weight")
        || name.ends_with(".attn_k.weight")
        || name.ends_with(".attn_v.weight")
        || name.ends_with(".attn_output.weight")
        || name.ends_with(".ffn_gate.weight")
        || name.ends_with(".ffn_up.weight")
        || name.ends_with(".ffn_down.weight")
}

fn is_moe_expert_name(name: &str) -> bool {
    // Qwen35-style separate gate/up tensors and Gemma4-style fused
    // `gate_up_exps`. Both variants are 3D `[n_experts, rows, cols]`.
    name.ends_with(".ffn_gate_exps.weight")
        || name.ends_with(".ffn_up_exps.weight")
        || name.ends_with(".ffn_gate_up_exps.weight")
        || name.ends_with(".ffn_down_exps.weight")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_loader::sidecar_v3::decoder::decode_sidecar_v3;

    #[test]
    fn encoder_options_default_is_phase1_b() {
        let opts = EncoderOptions::default();
        // Phase 1 has no toggles; the unit struct exists so signatures are stable
        // when Phase 2 introduces sidecar-range options.
        let _ = opts; // proves the type is constructible
    }

    #[test]
    fn dense_eligible_name_matches_attn_and_ffn_dense_only() {
        assert!(is_dense_eligible_name("blk.0.attn_q.weight"));
        assert!(is_dense_eligible_name("blk.0.attn_k.weight"));
        assert!(is_dense_eligible_name("blk.0.attn_v.weight"));
        assert!(is_dense_eligible_name("blk.0.attn_output.weight"));
        assert!(is_dense_eligible_name("blk.0.ffn_gate.weight"));
        assert!(is_dense_eligible_name("blk.0.ffn_up.weight"));
        assert!(is_dense_eligible_name("blk.0.ffn_down.weight"));
        // MoE expert tensors must NOT be eligible.
        assert!(!is_dense_eligible_name("blk.0.ffn_gate_exps.weight"));
        assert!(!is_dense_eligible_name("blk.0.ffn_up_exps.weight"));
        assert!(!is_dense_eligible_name("blk.0.ffn_down_exps.weight"));
        assert!(!is_dense_eligible_name("blk.0.ffn_gate_inp.weight"));
        // Norms / embeddings.
        assert!(!is_dense_eligible_name("blk.0.attn_norm.weight"));
        assert!(!is_dense_eligible_name("token_embd.weight"));
        assert!(!is_dense_eligible_name("output.weight"));
    }

    #[test]
    #[ignore = "needs Qwen3.5 0.8B GGUF fixture"]
    fn encode_qwen35_08b_writes_valid_sidecar() {
        let gguf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../models/qwen3.5-0.8B/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf");
        if !gguf.exists() {
            eprintln!("skipping: fixture not present at {:?}", gguf);
            return;
        }
        let out = tempfile::NamedTempFile::new().unwrap();
        encode_sidecar_v3(&gguf, out.path(), EncoderOptions::default()).unwrap();
        let decoded = decode_sidecar_v3(out.path()).unwrap();
        assert!(
            !decoded.tensors.is_empty(),
            "encoder must keep at least one tensor"
        );
        // Sanity: every kept tensor is one of the three v3 quant types Phase 1
        // emits — dense Q4K row-pair, dense Q6K, or MoE Q4K hot.
        for entry in &decoded.tensors {
            assert!(matches!(
                entry.quant_type,
                rnb_loader::sidecar_v3::spec::V3QuantType::DenseQ4kRowPair
                    | rnb_loader::sidecar_v3::spec::V3QuantType::DenseQ6k
                    | rnb_loader::sidecar_v3::spec::V3QuantType::MoeQ4kHot
            ));
        }
    }

    #[test]
    fn classify_tensor_role_recognises_dense_and_moe() {
        assert_eq!(
            classify_tensor_role("blk.0.ffn_gate.weight"),
            Some(TensorRole::Dense)
        );
        assert_eq!(
            classify_tensor_role("blk.0.attn_q.weight"),
            Some(TensorRole::Dense)
        );
        assert_eq!(
            classify_tensor_role("blk.0.ffn_gate_exps.weight"),
            Some(TensorRole::MoeExperts)
        );
        assert_eq!(
            classify_tensor_role("blk.0.ffn_up_exps.weight"),
            Some(TensorRole::MoeExperts)
        );
        assert_eq!(
            classify_tensor_role("blk.0.ffn_gate_up_exps.weight"),
            Some(TensorRole::MoeExperts)
        );
        assert_eq!(
            classify_tensor_role("blk.0.ffn_down_exps.weight"),
            Some(TensorRole::MoeExperts)
        );
        assert_eq!(classify_tensor_role("token_embd.weight"), None);
        assert_eq!(classify_tensor_role("output.weight"), None);
        assert_eq!(classify_tensor_role("blk.0.attn_norm.weight"), None);
    }

    #[test]
    fn parse_blk_idx_extracts_layer_number() {
        assert_eq!(parse_blk_idx("blk.0.ffn_gate_exps.weight"), Some(0));
        assert_eq!(parse_blk_idx("blk.42.attn_q.weight"), Some(42));
        assert_eq!(parse_blk_idx("blk.5.ffn_down_exps.weight"), Some(5));
        // Not a blk.{N} prefix
        assert_eq!(parse_blk_idx("token_embd.weight"), None);
        assert_eq!(parse_blk_idx("output.weight"), None);
        // Malformed
        assert_eq!(parse_blk_idx("blk.x.foo"), None);
    }

    #[test]
    fn build_metadata_returns_empty_when_no_popularity_inputs() {
        let layers = vec![(0u32, 8usize), (1u32, 8usize)];
        let opts = EncoderOptions::default();
        let m = build_metadata_from_layers(&layers, &opts).unwrap();
        assert!(m.layers.is_empty());
    }

    #[test]
    fn build_metadata_applies_hot_count_with_natural_popularity_order() {
        let layers = vec![(0u32, 8usize), (3u32, 8usize)];
        let opts = EncoderOptions {
            moe_hot_count: Some(4),
            popularity_order: None,
            popularity_per_layer: None,
        };
        let m = build_metadata_from_layers(&layers, &opts).unwrap();
        assert_eq!(m.layers.len(), 2);
        assert_eq!(m.layers[0].layer_idx, 0);
        assert_eq!(m.layers[0].hot_count, 4);
        assert_eq!(m.layers[0].popularity_order, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(m.layers[1].layer_idx, 3);
        assert_eq!(m.layers[1].hot_count, 4);
    }

    #[test]
    fn build_metadata_applies_explicit_popularity_order_to_each_layer() {
        let layers = vec![(0u32, 4usize), (7u32, 4usize)];
        let opts = EncoderOptions {
            moe_hot_count: Some(2),
            popularity_order: Some(vec![3, 1, 0, 2]),
            popularity_per_layer: None,
        };
        let m = build_metadata_from_layers(&layers, &opts).unwrap();
        assert_eq!(m.layers.len(), 2);
        for entry in &m.layers {
            assert_eq!(entry.hot_count, 2);
            assert_eq!(entry.popularity_order, vec![3, 1, 0, 2]);
        }
        assert_eq!(m.layers[0].layer_idx, 0);
        assert_eq!(m.layers[1].layer_idx, 7);
    }

    #[test]
    fn build_metadata_rejects_popularity_size_mismatch() {
        let layers = vec![(0u32, 8usize)];
        let opts = EncoderOptions {
            moe_hot_count: Some(4),
            popularity_order: Some(vec![0, 1, 2, 3]), // wrong length
            popularity_per_layer: None,
        };
        let err = build_metadata_from_layers(&layers, &opts).unwrap_err();
        assert!(err.contains("popularity_order"));
    }

    #[test]
    fn build_metadata_rejects_hot_count_above_n_experts() {
        let layers = vec![(0u32, 8usize)];
        let opts = EncoderOptions {
            moe_hot_count: Some(99),
            popularity_order: None,
            popularity_per_layer: None,
        };
        let err = build_metadata_from_layers(&layers, &opts).unwrap_err();
        assert!(err.contains("moe_hot_count"));
    }

    #[test]
    fn build_metadata_roundtrip_via_serialization() {
        let layers = vec![(0u32, 4usize), (5u32, 4usize)];
        let opts = EncoderOptions {
            moe_hot_count: Some(2),
            popularity_order: Some(vec![3, 1, 0, 2]),
            popularity_per_layer: None,
        };
        let m = build_metadata_from_layers(&layers, &opts).unwrap();
        let bytes = m.to_bytes();
        let decoded = MetadataV3::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, m);
    }

    #[test]
    fn transformed_layout_descriptor_is_emitted_for_supported_dense_q4_q6() {
        let q4_raw = vec![0xA5u8; 2 * 1 * Q4K_BLOCK_BYTES];
        let q4 = transformed_layout_descriptor_for_dense_source(
            "blk.0.ffn_gate.weight",
            V3QuantType::DenseQ4kRowPair,
            2,
            K_QUANT_BLOCK_ELEMS,
            &q4_raw,
        )
        .unwrap()
        .expect("dense Q4_K should emit descriptor");
        assert_eq!(q4.layout_kind, TransformedLayoutKind::Q4kCompactMetadata);
        assert_eq!(q4.rows, 2);
        assert_eq!(q4.cols, K_QUANT_BLOCK_ELEMS as u64);
        assert_eq!(q4.source_len, q4_raw.len() as u64);

        let q6_raw = vec![0x5Au8; 3 * 2 * Q6K_BLOCK_BYTES];
        let q6 = transformed_layout_descriptor_for_dense_source(
            "blk.0.ffn_down.weight",
            V3QuantType::DenseQ6k,
            3,
            K_QUANT_BLOCK_ELEMS * 2,
            &q6_raw,
        )
        .unwrap()
        .expect("dense Q6_K should emit descriptor");
        assert_eq!(q6.layout_kind, TransformedLayoutKind::Q6kPackedQ8dot);
        assert_eq!(q6.rows, 3);
        assert_eq!(q6.cols, (K_QUANT_BLOCK_ELEMS * 2) as u64);
        assert_eq!(q6.source_len, q6_raw.len() as u64);
    }

    #[test]
    fn transformed_layout_descriptor_skips_unsupported_source_kind() {
        let moe_raw = vec![0x11u8; Q4K_BLOCK_BYTES];
        let descriptor = transformed_layout_descriptor_for_dense_source(
            "blk.0.ffn_gate_exps.weight",
            V3QuantType::MoeQ4kHot,
            1,
            K_QUANT_BLOCK_ELEMS,
            &moe_raw,
        )
        .unwrap();

        assert!(descriptor.is_none());
    }

    #[test]
    fn transformed_layout_descriptor_rejects_shape_byte_mismatch() {
        let q6_raw = vec![0x22u8; Q6K_BLOCK_BYTES - 1];
        let err = transformed_layout_descriptor_for_dense_source(
            "blk.0.ffn_down.weight",
            V3QuantType::DenseQ6k,
            1,
            K_QUANT_BLOCK_ELEMS,
            &q6_raw,
        )
        .unwrap_err();

        assert!(err.contains("byte mismatch"));
    }
}
