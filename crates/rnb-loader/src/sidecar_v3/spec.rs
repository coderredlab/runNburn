//! Sidecar `.rnb` v3 binary layout.
//!
//! v3 = packed sidecar cache for GGUF inputs. Distinct from v2 (`RNBD`) which
//! was a standalone single-file format being phased out (Tasks 14-18). The v3
//! magic `RNBC` ("RNB Cache") prevents accidental cross-loading: a v2 file's
//! cache filename is also keyed differently (Task 2 cache_filename embeds the
//! format version into the SHA), so v2 caches never match v3 lookups.
//!
//! Header layout (`HEADER_SIZE_BYTES` = 44 bytes since `FORMAT_VERSION_V3 = 4`):
//!
//! ```text
//! offset  size  field
//! 0       4     magic = b"RNBC"
//! 4       4     version (u32 LE)               = 4
//! 8       4     tensor_count (u32 LE)
//! 12      8     tensor_table_offset (u64 LE)
//! 20      8     payload_offset (u64 LE)
//! 28      8     metadata_offset (u64 LE)       0 = absent
//! 36      8     metadata_size   (u64 LE)
//! 44      ..    tensor table (tensor_count entries, see TensorEntryV3 layout)
//! ..      ..    payload (raw packed weight bytes)
//! ..      ..    metadata blob (optional, see MetadataV3)
//! ```
//!
//! Tensor table entry (variable length):
//!
//! ```text
//! u16 LE  name length (bytes)
//! N       name (UTF-8)
//! u8      quant_type (see V3QuantType)
//! u32 LE  ndim
//! [u64 LE; ndim]  shape
//! u64 LE  payload_offset (relative to file start)
//! u64 LE  payload_size (bytes)
//! ```
//!
//! Metadata blob (optional, present when `metadata_offset != 0`):
//!
//! ```text
//! u32 LE  layer_count
//! [layer_entry; layer_count]
//!
//! layer_entry (variable length):
//!   u32 LE  layer_idx
//!   u32 LE  hot_count
//!   u32 LE  n_expert
//!   [u32 LE; n_expert]  popularity_order (rank -> original expert id)
//! ```
//!
//! Version history:
//! - v3 = 3 (in3 phase 1): 28-byte header, no metadata section.
//! - v3 = 4 (in4 phase 2): 44-byte header, optional metadata blob carrying
//!   per-layer popularity_order + hot_count for cold-streaming dispatch.
//!   Old v=3 cache files are auto-invalidated because the cache key embeds
//!   `FORMAT_VERSION_V3` into the SHA (see `cache::key::compute_cache_key`).

pub const MAGIC: [u8; 4] = *b"RNBC";
pub const FORMAT_VERSION_V3: u32 = 4;
pub const HEADER_SIZE_BYTES: usize = 44;

/// Quant type identifiers used in v3 tensor entries. Phase 1 uses option B
/// only: dense Q4_K row-pair, dense Q6_K, MoE Q4_K hot/cold. Other tensors
/// (Q5_K, Q5_1, Q8_0, gather-only embeddings) are not packed into the sidecar
/// and are read directly from the source GGUF at runtime.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V3QuantType {
    DenseQ4kRowPair = 1,
    DenseQ6k = 2,
    MoeQ4kHot = 3,
    MoeQ4kCold = 4,
}

impl V3QuantType {
    pub fn from_u8(v: u8) -> Result<Self, String> {
        match v {
            1 => Ok(Self::DenseQ4kRowPair),
            2 => Ok(Self::DenseQ6k),
            3 => Ok(Self::MoeQ4kHot),
            4 => Ok(Self::MoeQ4kCold),
            _ => Err(format!("unknown sidecar v3 quant_type: {v}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderV3 {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u32,
    pub tensor_table_offset: u64,
    pub payload_offset: u64,
    /// Byte offset of the metadata blob within the sidecar file. `0` = no
    /// metadata blob (popularity / hot_count not persisted; engine treats
    /// every loaded MoE expert as hot-only).
    pub metadata_offset: u64,
    /// Length of the metadata blob in bytes. Must be `0` when
    /// `metadata_offset == 0`.
    pub metadata_size: u64,
}

impl HeaderV3 {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_SIZE_BYTES);
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.tensor_count.to_le_bytes());
        out.extend_from_slice(&self.tensor_table_offset.to_le_bytes());
        out.extend_from_slice(&self.payload_offset.to_le_bytes());
        out.extend_from_slice(&self.metadata_offset.to_le_bytes());
        out.extend_from_slice(&self.metadata_size.to_le_bytes());
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER_SIZE_BYTES {
            return Err(format!(
                "sidecar v3 header truncated: got {} bytes, want {}",
                b.len(),
                HEADER_SIZE_BYTES
            ));
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&b[0..4]);
        if magic != MAGIC {
            return Err(format!(
                "sidecar v3 magic mismatch: got {magic:?}, want {MAGIC:?}"
            ));
        }
        let version = u32::from_le_bytes(b[4..8].try_into().unwrap());
        let tensor_count = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let tensor_table_offset = u64::from_le_bytes(b[12..20].try_into().unwrap());
        let payload_offset = u64::from_le_bytes(b[20..28].try_into().unwrap());
        let metadata_offset = u64::from_le_bytes(b[28..36].try_into().unwrap());
        let metadata_size = u64::from_le_bytes(b[36..44].try_into().unwrap());
        if metadata_offset == 0 && metadata_size != 0 {
            return Err(format!(
                "sidecar v3 metadata_offset=0 but metadata_size={metadata_size} (must be 0)"
            ));
        }
        Ok(Self {
            magic,
            version,
            tensor_count,
            tensor_table_offset,
            payload_offset,
            metadata_offset,
            metadata_size,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntryV3 {
    pub name: String,
    pub quant_type: V3QuantType,
    pub shape: Vec<u64>,
    pub payload_offset: u64,
    pub payload_size: u64,
}

/// Lossless transformed-layout kind carried by RNBC metadata. These values
/// describe how a source GGUF tensor was rearranged, not a product input type.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformedLayoutKind {
    Q4kCompactMetadata = 1,
    Q6kPackedQ8dot = 2,
}

impl TransformedLayoutKind {
    pub fn from_u8(v: u8) -> Result<Self, String> {
        match v {
            1 => Ok(Self::Q4kCompactMetadata),
            2 => Ok(Self::Q6kPackedQ8dot),
            _ => Err(format!("unknown transformed layout kind: {v}")),
        }
    }

    pub fn current_version(self) -> u32 {
        match self {
            Self::Q4kCompactMetadata => 1,
            Self::Q6kPackedQ8dot => 1,
        }
    }
}

/// Source tensor identity used to validate that a transformed cache entry was
/// produced from exactly the tensor the runtime is about to consume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformedLayoutSourceV1 {
    pub tensor_name: String,
    pub source_quant: V3QuantType,
    pub rows: u64,
    pub cols: u64,
    pub source_len: u64,
    pub source_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformedLayoutDescriptorV1 {
    pub tensor_name: String,
    pub source_quant: V3QuantType,
    pub rows: u64,
    pub cols: u64,
    pub source_len: u64,
    pub source_fingerprint: u64,
    pub layout_kind: TransformedLayoutKind,
    pub layout_version: u32,
    pub block_rows: u32,
    pub block_cols: u32,
    pub producer_options_hash: u64,
}

impl TransformedLayoutDescriptorV1 {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let name_bytes = self.tensor_name.as_bytes();
        out.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(name_bytes);
        out.push(self.source_quant as u8);
        out.extend_from_slice(&self.rows.to_le_bytes());
        out.extend_from_slice(&self.cols.to_le_bytes());
        out.extend_from_slice(&self.source_len.to_le_bytes());
        out.extend_from_slice(&self.source_fingerprint.to_le_bytes());
        out.push(self.layout_kind as u8);
        out.extend_from_slice(&self.layout_version.to_le_bytes());
        out.extend_from_slice(&self.block_rows.to_le_bytes());
        out.extend_from_slice(&self.block_cols.to_le_bytes());
        out.extend_from_slice(&self.producer_options_hash.to_le_bytes());
        out
    }

    fn from_bytes(b: &[u8], cursor: &mut usize, index: usize) -> Result<Self, String> {
        let name_len = read_u16_le(b, cursor, "transformed layout name length")? as usize;
        let name_bytes = read_slice(b, cursor, name_len, "transformed layout name")?;
        let tensor_name = std::str::from_utf8(name_bytes)
            .map_err(|e| format!("transformed layout {index} name not utf8: {e}"))?
            .to_string();
        let source_quant = V3QuantType::from_u8(read_u8(b, cursor, "source quant")?)?;
        let rows = read_u64_le(b, cursor, "transformed layout rows")?;
        let cols = read_u64_le(b, cursor, "transformed layout cols")?;
        let source_len = read_u64_le(b, cursor, "transformed layout source_len")?;
        let source_fingerprint = read_u64_le(b, cursor, "transformed layout source_fingerprint")?;
        let layout_kind =
            TransformedLayoutKind::from_u8(read_u8(b, cursor, "transformed layout kind")?)?;
        let layout_version = read_u32_le(b, cursor, "transformed layout version")?;
        let block_rows = read_u32_le(b, cursor, "transformed layout block_rows")?;
        let block_cols = read_u32_le(b, cursor, "transformed layout block_cols")?;
        let producer_options_hash =
            read_u64_le(b, cursor, "transformed layout producer_options_hash")?;
        let descriptor = Self {
            tensor_name,
            source_quant,
            rows,
            cols,
            source_len,
            source_fingerprint,
            layout_kind,
            layout_version,
            block_rows,
            block_cols,
            producer_options_hash,
        };
        descriptor.validate_metadata_fields(index)?;
        Ok(descriptor)
    }

    fn validate_metadata_fields(&self, index: usize) -> Result<(), String> {
        if self.tensor_name.is_empty() {
            return Err(format!("transformed layout {index} tensor_name is empty"));
        }
        if self.rows == 0 || self.cols == 0 {
            return Err(format!(
                "transformed layout {index} shape invalid for {}: {}x{}",
                self.tensor_name, self.rows, self.cols
            ));
        }
        if self.source_len == 0 {
            return Err(format!(
                "transformed layout {index} source_len is zero for {}",
                self.tensor_name
            ));
        }
        if self.layout_version != self.layout_kind.current_version() {
            return Err(format!(
                "transformed layout_version mismatch for {}: got {}, want {} for {:?}",
                self.tensor_name,
                self.layout_version,
                self.layout_kind.current_version(),
                self.layout_kind
            ));
        }
        if self.block_rows == 0 || self.block_cols == 0 {
            return Err(format!(
                "transformed block geometry invalid for {}: {}x{}",
                self.tensor_name, self.block_rows, self.block_cols
            ));
        }
        Ok(())
    }

    pub fn validate_against_source(
        &self,
        source: &TransformedLayoutSourceV1,
    ) -> Result<(), String> {
        self.validate_metadata_fields(0)?;
        if self.tensor_name != source.tensor_name {
            return Err(format!(
                "transformed tensor_name mismatch: descriptor {}, source {}",
                self.tensor_name, source.tensor_name
            ));
        }
        if self.source_quant != source.source_quant {
            return Err(format!(
                "transformed source_quant mismatch for {}: descriptor {:?}, source {:?}",
                self.tensor_name, self.source_quant, source.source_quant
            ));
        }
        if self.rows != source.rows || self.cols != source.cols {
            return Err(format!(
                "transformed shape mismatch for {}: descriptor [{} x {}], source [{} x {}]",
                self.tensor_name, self.rows, self.cols, source.rows, source.cols
            ));
        }
        if self.source_len != source.source_len {
            return Err(format!(
                "transformed source_len mismatch for {}: descriptor {}, source {}",
                self.tensor_name, self.source_len, source.source_len
            ));
        }
        if self.source_fingerprint != source.source_fingerprint {
            return Err(format!(
                "transformed source_fingerprint mismatch for {}: descriptor {:#x}, source {:#x}",
                self.tensor_name, self.source_fingerprint, source.source_fingerprint
            ));
        }
        Ok(())
    }
}

fn read_slice<'a>(
    b: &'a [u8],
    cursor: &mut usize,
    len: usize,
    field: &str,
) -> Result<&'a [u8], String> {
    if cursor
        .checked_add(len)
        .map(|end| end > b.len())
        .unwrap_or(true)
    {
        return Err(format!(
            "sidecar v3 metadata {field} truncated at cursor {}",
            *cursor
        ));
    }
    let start = *cursor;
    *cursor += len;
    Ok(&b[start..start + len])
}

fn read_u8(b: &[u8], cursor: &mut usize, field: &str) -> Result<u8, String> {
    Ok(read_slice(b, cursor, 1, field)?[0])
}

fn read_u16_le(b: &[u8], cursor: &mut usize, field: &str) -> Result<u16, String> {
    let bytes = read_slice(b, cursor, 2, field)?;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u32_le(b: &[u8], cursor: &mut usize, field: &str) -> Result<u32, String> {
    let bytes = read_slice(b, cursor, 4, field)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64_le(b: &[u8], cursor: &mut usize, field: &str) -> Result<u64, String> {
    let bytes = read_slice(b, cursor, 8, field)?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

/// Per-layer popularity + hot_count metadata for MoE cold-streaming dispatch.
/// Engine consumes this through [`crate::packed::PackedModel`] and then
/// [`rnb_memory::moe_residency::MoeExpertResidencyView`] to pick hot vs cold
/// expert byte slices at forward time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPopularityEntry {
    pub layer_idx: u32,
    pub hot_count: u32,
    /// `popularity_order[rank] = original_expert_id` (length = `n_expert`).
    /// Sidecar payload entries are stored in this rank order, so `rank < hot_count`
    /// pages live in the `MoeQ4kHot` tensor and `rank >= hot_count` live in
    /// `MoeQ4kCold`.
    pub popularity_order: Vec<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetadataV3 {
    pub layers: Vec<LayerPopularityEntry>,
    pub transformed_layouts: Vec<TransformedLayoutDescriptorV1>,
}

impl MetadataV3 {
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty() && self.transformed_layouts.is_empty()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            out.extend_from_slice(&layer.layer_idx.to_le_bytes());
            out.extend_from_slice(&layer.hot_count.to_le_bytes());
            out.extend_from_slice(&(layer.popularity_order.len() as u32).to_le_bytes());
            for v in &layer.popularity_order {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.transformed_layouts.len() as u32).to_le_bytes());
        for layout in &self.transformed_layouts {
            out.extend_from_slice(&layout.to_bytes());
        }
        out
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < 4 {
            return Err(format!(
                "sidecar v3 metadata truncated: got {} bytes, want >= 4",
                b.len()
            ));
        }
        let layer_count = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let mut cursor = 4usize;
        let mut layers = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            if cursor + 12 > b.len() {
                return Err(format!(
                    "sidecar v3 metadata layer {i} header truncated at cursor {cursor}"
                ));
            }
            let layer_idx = u32::from_le_bytes(b[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let hot_count = u32::from_le_bytes(b[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            let n_expert = u32::from_le_bytes(b[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            if cursor + n_expert * 4 > b.len() {
                return Err(format!(
                    "sidecar v3 metadata layer {i} popularity_order truncated"
                ));
            }
            let mut popularity_order = Vec::with_capacity(n_expert);
            for _ in 0..n_expert {
                popularity_order.push(u32::from_le_bytes(
                    b[cursor..cursor + 4].try_into().unwrap(),
                ));
                cursor += 4;
            }
            layers.push(LayerPopularityEntry {
                layer_idx,
                hot_count,
                popularity_order,
            });
        }
        let mut transformed_layouts = Vec::new();
        if cursor < b.len() {
            if cursor + 4 > b.len() {
                return Err(format!(
                    "sidecar v3 metadata transformed layout count truncated at cursor {cursor}"
                ));
            }
            let layout_count =
                u32::from_le_bytes(b[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            transformed_layouts.reserve(layout_count);
            for i in 0..layout_count {
                transformed_layouts.push(TransformedLayoutDescriptorV1::from_bytes(
                    b,
                    &mut cursor,
                    i,
                )?);
            }
        }
        if cursor != b.len() {
            return Err(format!(
                "sidecar v3 metadata trailing bytes after transformed layouts: cursor {cursor}, len {}",
                b.len()
            ));
        }
        Ok(Self {
            layers,
            transformed_layouts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_constants() {
        assert_eq!(MAGIC.len(), 4);
        assert_eq!(MAGIC, *b"RNBC");
        assert_eq!(FORMAT_VERSION_V3, 4);
        assert_eq!(HEADER_SIZE_BYTES, 44);
    }

    #[test]
    fn header_roundtrip_without_metadata() {
        let h = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 42,
            tensor_table_offset: 256,
            payload_offset: 4096,
            metadata_offset: 0,
            metadata_size: 0,
        };
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE_BYTES);
        let parsed = HeaderV3::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn header_roundtrip_with_metadata() {
        let h = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 42,
            tensor_table_offset: 256,
            payload_offset: 4096,
            metadata_offset: 1_048_576,
            metadata_size: 2048,
        };
        let bytes = h.to_bytes();
        let parsed = HeaderV3::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn header_rejects_inconsistent_metadata_offset_and_size() {
        let mut bytes = vec![0u8; HEADER_SIZE_BYTES];
        bytes[..4].copy_from_slice(&MAGIC);
        bytes[4..8].copy_from_slice(&FORMAT_VERSION_V3.to_le_bytes());
        // metadata_offset = 0, metadata_size = 16 -> invalid
        bytes[36..44].copy_from_slice(&16u64.to_le_bytes());
        let err = HeaderV3::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("metadata"));
    }

    #[test]
    fn header_rejects_wrong_magic() {
        let mut bytes = vec![0u8; HEADER_SIZE_BYTES];
        bytes[..4].copy_from_slice(b"RNBD"); // v2 magic
        let err = HeaderV3::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("magic"));
    }

    #[test]
    fn header_rejects_truncated_input() {
        let bytes = vec![0u8; 10];
        let err = HeaderV3::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("short") || err.contains("truncated"));
    }

    #[test]
    fn metadata_roundtrip_empty() {
        let m = MetadataV3::default();
        let bytes = m.to_bytes();
        assert_eq!(bytes.len(), 8); // layer_count = 0 + transformed_layout_count = 0
        let parsed = MetadataV3::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn metadata_reads_legacy_empty_blob_without_transformed_count() {
        let bytes = 0u32.to_le_bytes();
        let parsed = MetadataV3::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, MetadataV3::default());
    }

    #[test]
    fn metadata_roundtrip_two_layers() {
        let m = MetadataV3 {
            layers: vec![
                LayerPopularityEntry {
                    layer_idx: 0,
                    hot_count: 39,
                    popularity_order: vec![5, 3, 7, 1, 0, 2, 4, 6],
                },
                LayerPopularityEntry {
                    layer_idx: 7,
                    hot_count: 16,
                    popularity_order: vec![10, 11, 12, 13],
                },
            ],
            transformed_layouts: Vec::new(),
        };
        let bytes = m.to_bytes();
        let parsed = MetadataV3::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn metadata_roundtrip_transformed_layouts() {
        let m = MetadataV3 {
            layers: Vec::new(),
            transformed_layouts: vec![transformed_layout_descriptor_fixture()],
        };
        let bytes = m.to_bytes();
        let parsed = MetadataV3::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, m);
    }

    #[test]
    fn metadata_rejects_transformed_layout_bad_version() {
        let mut m = MetadataV3 {
            layers: Vec::new(),
            transformed_layouts: vec![transformed_layout_descriptor_fixture()],
        };
        m.transformed_layouts[0].layout_version += 1;

        let bytes = m.to_bytes();
        let err = MetadataV3::from_bytes(&bytes).unwrap_err();
        assert!(err.contains("layout_version"));
    }

    #[test]
    fn metadata_rejects_truncated_layer_header() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes()); // layer_count = 1
        buf.extend_from_slice(&[0u8; 8]); // only 8 bytes of layer header (need 12)
        let err = MetadataV3::from_bytes(&buf).unwrap_err();
        assert!(err.contains("truncated"));
    }

    #[test]
    fn metadata_rejects_truncated_popularity_order() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes()); // layer_count = 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // layer_idx
        buf.extend_from_slice(&4u32.to_le_bytes()); // hot_count
        buf.extend_from_slice(&8u32.to_le_bytes()); // n_expert = 8
                                                    // but only 4 entries follow (16 bytes instead of 32)
        for v in 0u32..4 {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let err = MetadataV3::from_bytes(&buf).unwrap_err();
        assert!(err.contains("truncated"));
    }

    fn transformed_layout_source_fixture() -> TransformedLayoutSourceV1 {
        TransformedLayoutSourceV1 {
            tensor_name: "blk.0.ffn_down.weight".to_string(),
            source_quant: V3QuantType::DenseQ6k,
            rows: 2048,
            cols: 4096,
            source_len: 860_160,
            source_fingerprint: 0xA11C_E551_0CA1_2026,
        }
    }

    fn transformed_layout_descriptor_fixture() -> TransformedLayoutDescriptorV1 {
        TransformedLayoutDescriptorV1 {
            tensor_name: "blk.0.ffn_down.weight".to_string(),
            source_quant: V3QuantType::DenseQ6k,
            rows: 2048,
            cols: 4096,
            source_len: 860_160,
            source_fingerprint: 0xA11C_E551_0CA1_2026,
            layout_kind: TransformedLayoutKind::Q6kPackedQ8dot,
            layout_version: TransformedLayoutKind::Q6kPackedQ8dot.current_version(),
            block_rows: 1,
            block_cols: 256,
            producer_options_hash: 0x5151_5151_5151_5151,
        }
    }

    #[test]
    fn transformed_layout_accepts_matching_source_descriptor() {
        let source = transformed_layout_source_fixture();
        let descriptor = transformed_layout_descriptor_fixture();

        descriptor.validate_against_source(&source).unwrap();
    }

    #[test]
    fn transformed_layout_rejects_mismatched_layout_version() {
        let source = transformed_layout_source_fixture();
        let mut descriptor = transformed_layout_descriptor_fixture();
        descriptor.layout_version += 1;

        let err = descriptor.validate_against_source(&source).unwrap_err();
        assert!(err.contains("layout_version"));
    }

    #[test]
    fn transformed_layout_rejects_source_length_mismatch() {
        let mut source = transformed_layout_source_fixture();
        source.source_len += 1;
        let descriptor = transformed_layout_descriptor_fixture();

        let err = descriptor.validate_against_source(&source).unwrap_err();
        assert!(err.contains("source_len"));
    }

    #[test]
    fn transformed_layout_rejects_source_fingerprint_mismatch() {
        let mut source = transformed_layout_source_fixture();
        source.source_fingerprint ^= 0x55AA;
        let descriptor = transformed_layout_descriptor_fixture();

        let err = descriptor.validate_against_source(&source).unwrap_err();
        assert!(err.contains("source_fingerprint"));
    }

    #[test]
    fn transformed_layout_rejects_shape_mismatch() {
        let mut source = transformed_layout_source_fixture();
        source.cols += 256;
        let descriptor = transformed_layout_descriptor_fixture();

        let err = descriptor.validate_against_source(&source).unwrap_err();
        assert!(err.contains("shape"));
    }
}
