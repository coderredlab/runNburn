//! Sidecar `.rnb` v3 decoder.
//!
//! Maps a v3 sidecar file into memory, validates the [`spec::HeaderV3`], and
//! parses the tensor table into an in-memory [`DecodedSidecar`]. The
//! underlying mmap is held by the returned struct so callers can index into
//! `mmap[entry.payload_offset..entry.payload_offset + entry.payload_size]`
//! without copying.
//!
//! v2 (`RNBD`) and unsupported v3 versions are rejected — see
//! [`decode_sidecar_v3`] for the failure modes.

use crate::sidecar_v3::spec::{
    HeaderV3, MetadataV3, TensorEntryV3, V3QuantType, FORMAT_VERSION_V3, HEADER_SIZE_BYTES,
};
use memmap2::Mmap;
use std::path::Path;

#[derive(Debug)]
pub struct DecodedSidecar {
    pub header: HeaderV3,
    pub tensors: Vec<TensorEntryV3>,
    /// Optional per-layer popularity / hot_count metadata. `None` when the
    /// sidecar was produced without popularity input — the engine then treats
    /// every MoE expert as hot-only.
    pub metadata: Option<MetadataV3>,
    pub mmap: Mmap,
}

pub fn decode_sidecar_v3(path: &Path) -> Result<DecodedSidecar, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    // SAFETY: We map a file we just opened read-only. The `Mmap` is owned by
    // the returned struct, which keeps the underlying file descriptor alive
    // for the lifetime of the slice references handed out via `mmap[..]`.
    let mmap = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;
    if mmap.len() < HEADER_SIZE_BYTES {
        return Err(format!(
            "sidecar v3 file too small: {} bytes < header {}",
            mmap.len(),
            HEADER_SIZE_BYTES
        ));
    }
    let header = HeaderV3::from_bytes(&mmap[..HEADER_SIZE_BYTES])?;
    if header.version != FORMAT_VERSION_V3 {
        return Err(format!(
            "unsupported sidecar v3 version: got {}, want {FORMAT_VERSION_V3}",
            header.version
        ));
    }

    let mut tensors = Vec::with_capacity(header.tensor_count as usize);
    let mut cursor = header.tensor_table_offset as usize;

    for i in 0..header.tensor_count {
        if cursor + 2 > mmap.len() {
            return Err(format!(
                "tensor table truncated at entry {i} (cursor {cursor})"
            ));
        }
        let name_len = u16::from_le_bytes(mmap[cursor..cursor + 2].try_into().unwrap()) as usize;
        cursor += 2;
        if cursor + name_len > mmap.len() {
            return Err(format!("tensor {i} name truncated"));
        }
        let name = std::str::from_utf8(&mmap[cursor..cursor + name_len])
            .map_err(|e| format!("tensor {i} name not utf8: {e}"))?
            .to_string();
        cursor += name_len;
        if cursor + 1 + 4 > mmap.len() {
            return Err(format!("tensor {i} header truncated"));
        }
        let quant_type = V3QuantType::from_u8(mmap[cursor])?;
        cursor += 1;
        let ndim = u32::from_le_bytes(mmap[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + ndim * 8 + 16 > mmap.len() {
            return Err(format!("tensor {i} shape/offset truncated"));
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u64::from_le_bytes(
                mmap[cursor..cursor + 8].try_into().unwrap(),
            ));
            cursor += 8;
        }
        let payload_offset = u64::from_le_bytes(mmap[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let payload_size = u64::from_le_bytes(mmap[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        if (payload_offset + payload_size) as usize > mmap.len() {
            return Err(format!(
                "tensor {i} payload [{payload_offset}, +{payload_size}] beyond file"
            ));
        }
        tensors.push(TensorEntryV3 {
            name,
            quant_type,
            shape,
            payload_offset,
            payload_size,
        });
    }

    let metadata = if header.metadata_offset != 0 {
        let off = header.metadata_offset as usize;
        let size = header.metadata_size as usize;
        if off
            .checked_add(size)
            .map(|end| end > mmap.len())
            .unwrap_or(true)
        {
            return Err(format!(
                "sidecar v3 metadata blob [{off}, +{size}] beyond file (len {})",
                mmap.len()
            ));
        }
        Some(MetadataV3::from_bytes(&mmap[off..off + size])?)
    } else {
        None
    };

    Ok(DecodedSidecar {
        header,
        tensors,
        metadata,
        mmap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sidecar_v3::spec::{
        HeaderV3, LayerPopularityEntry, MetadataV3, TransformedLayoutDescriptorV1,
        TransformedLayoutKind, V3QuantType, FORMAT_VERSION_V3, HEADER_SIZE_BYTES, MAGIC,
    };
    use std::io::Write;

    /// Build a minimal valid v3 sidecar with one tensor entry.
    fn write_minimal_v3_sidecar(tensor_name: &str, payload: &[u8]) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf: Vec<u8> = Vec::new();
        // header placeholder (will rewrite after we know offsets)
        buf.extend_from_slice(&vec![0u8; HEADER_SIZE_BYTES]);
        let table_offset = HEADER_SIZE_BYTES as u64;
        let name_bytes = tensor_name.as_bytes();
        // table entry
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.push(V3QuantType::DenseQ4kRowPair as u8);
        buf.extend_from_slice(&1u32.to_le_bytes()); // ndim = 1
        buf.extend_from_slice(&(payload.len() as u64).to_le_bytes()); // shape[0]
        let payload_offset_pos = buf.len();
        buf.extend_from_slice(&[0u8; 8]); // payload_offset placeholder
        buf.extend_from_slice(&(payload.len() as u64).to_le_bytes()); // payload_size
        let payload_offset = buf.len() as u64;
        // patch payload_offset
        buf[payload_offset_pos..payload_offset_pos + 8]
            .copy_from_slice(&payload_offset.to_le_bytes());
        // payload
        buf.extend_from_slice(payload);
        // patch header
        let header = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 1,
            tensor_table_offset: table_offset,
            payload_offset,
            metadata_offset: 0,
            metadata_size: 0,
        };
        let header_bytes = header.to_bytes();
        buf[..HEADER_SIZE_BYTES].copy_from_slice(&header_bytes);
        std::fs::write(f.path(), &buf).unwrap();
        let _ = std::io::stdout().flush();
        f
    }

    #[test]
    fn decode_minimal_sidecar_returns_one_tensor() {
        let f = write_minimal_v3_sidecar("blk.0.attn_q.weight", &[1u8, 2, 3, 4, 5, 6, 7, 8]);
        let decoded = decode_sidecar_v3(f.path()).unwrap();
        assert_eq!(decoded.header.magic, MAGIC);
        assert_eq!(decoded.header.tensor_count, 1);
        assert_eq!(decoded.tensors.len(), 1);
        let entry = &decoded.tensors[0];
        assert_eq!(entry.name, "blk.0.attn_q.weight");
        assert_eq!(entry.quant_type, V3QuantType::DenseQ4kRowPair);
        assert_eq!(entry.shape, vec![8]);
        assert_eq!(entry.payload_size, 8);
    }

    #[test]
    fn decode_rejects_v2_magic() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf = vec![0u8; HEADER_SIZE_BYTES];
        buf[..4].copy_from_slice(b"RNBD"); // v2 magic
        std::fs::write(f.path(), &buf).unwrap();
        let err = decode_sidecar_v3(f.path()).unwrap_err();
        assert!(err.contains("magic"));
    }

    #[test]
    fn decode_rejects_too_small_file() {
        let f = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(f.path(), &[0u8; 10]).unwrap();
        let err = decode_sidecar_v3(f.path()).unwrap_err();
        assert!(err.contains("small") || err.contains("short") || err.contains("truncated"));
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf = vec![0u8; HEADER_SIZE_BYTES];
        buf[..4].copy_from_slice(b"RNBC"); // valid v3 magic
        buf[4..8].copy_from_slice(&99u32.to_le_bytes()); // unsupported version
        std::fs::write(f.path(), &buf).unwrap();
        let err = decode_sidecar_v3(f.path()).unwrap_err();
        assert!(err.contains("version"));
    }

    #[test]
    fn decode_minimal_sidecar_has_no_metadata() {
        let f = write_minimal_v3_sidecar("blk.0.attn_q.weight", &[1u8; 8]);
        let decoded = decode_sidecar_v3(f.path()).unwrap();
        assert!(decoded.metadata.is_none());
    }

    /// Build a v3 sidecar with one tensor entry plus a metadata blob trailing
    /// the payload — mirrors how the encoder writes layer popularity_order.
    fn write_sidecar_with_metadata(metadata: &MetadataV3) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf: Vec<u8> = Vec::new();
        let payload = [0xAAu8; 16];
        let tensor_name = "blk.0.ffn_gate_exps.weight";
        let name_bytes = tensor_name.as_bytes();

        buf.extend_from_slice(&vec![0u8; HEADER_SIZE_BYTES]);
        let table_offset = HEADER_SIZE_BYTES as u64;

        // Tensor table entry
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.push(V3QuantType::MoeQ4kHot as u8);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let payload_offset_pos = buf.len();
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        let payload_offset = buf.len() as u64;
        buf[payload_offset_pos..payload_offset_pos + 8]
            .copy_from_slice(&payload_offset.to_le_bytes());

        // Payload
        buf.extend_from_slice(&payload);

        // Metadata blob at the tail
        let metadata_offset = buf.len() as u64;
        let metadata_bytes = metadata.to_bytes();
        let metadata_size = metadata_bytes.len() as u64;
        buf.extend_from_slice(&metadata_bytes);

        // Patch header
        let header = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 1,
            tensor_table_offset: table_offset,
            payload_offset,
            metadata_offset,
            metadata_size,
        };
        buf[..HEADER_SIZE_BYTES].copy_from_slice(&header.to_bytes());
        std::fs::write(f.path(), &buf).unwrap();
        f
    }

    #[test]
    fn decode_returns_metadata_when_blob_present() {
        let m = MetadataV3 {
            layers: vec![
                LayerPopularityEntry {
                    layer_idx: 0,
                    hot_count: 4,
                    popularity_order: vec![3, 1, 0, 2, 5, 7, 6, 4],
                },
                LayerPopularityEntry {
                    layer_idx: 5,
                    hot_count: 2,
                    popularity_order: vec![0, 1, 2, 3],
                },
            ],
            transformed_layouts: Vec::new(),
        };
        let f = write_sidecar_with_metadata(&m);
        let decoded = decode_sidecar_v3(f.path()).unwrap();
        let parsed = decoded.metadata.expect("metadata blob must be present");
        assert_eq!(parsed, m);
    }

    #[test]
    fn decode_returns_transformed_layout_metadata() {
        let m = MetadataV3 {
            layers: Vec::new(),
            transformed_layouts: vec![TransformedLayoutDescriptorV1 {
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
            }],
        };
        let f = write_sidecar_with_metadata(&m);
        let decoded = decode_sidecar_v3(f.path()).unwrap();
        let parsed = decoded.metadata.expect("metadata blob must be present");
        assert_eq!(parsed.transformed_layouts, m.transformed_layouts);
    }

    #[test]
    fn decode_rejects_invalid_transformed_layout_metadata() {
        let mut m = MetadataV3 {
            layers: Vec::new(),
            transformed_layouts: vec![TransformedLayoutDescriptorV1 {
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
            }],
        };
        m.transformed_layouts[0].layout_version += 1;
        let f = write_sidecar_with_metadata(&m);

        let err = decode_sidecar_v3(f.path()).unwrap_err();
        assert!(err.contains("layout_version"));
    }

    #[test]
    fn decode_rejects_metadata_offset_past_eof() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf = vec![0u8; HEADER_SIZE_BYTES];
        buf[..4].copy_from_slice(&MAGIC);
        buf[4..8].copy_from_slice(&FORMAT_VERSION_V3.to_le_bytes());
        // metadata_offset/size point past EOF
        buf[28..36].copy_from_slice(&1_000_000u64.to_le_bytes());
        buf[36..44].copy_from_slice(&8u64.to_le_bytes());
        std::fs::write(f.path(), &buf).unwrap();
        let err = decode_sidecar_v3(f.path()).unwrap_err();
        assert!(err.contains("metadata"));
    }
}
