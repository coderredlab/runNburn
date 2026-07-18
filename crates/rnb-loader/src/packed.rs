// PackedWeight + PackedModel mmap loader

use crate::rnb_file::{RnbHeader, TensorEntry, MAGIC, VERSION};
use rnb_core::tensor::QuantType;
use rnb_memory::moe_residency::MoeExpertResidencyView;
use std::borrow::Cow;
use std::io;
use std::sync::Arc;

/// MoE expert residency tier. Dense weights and "hot" experts go to
/// `WillNeed` (OS keeps pages resident); "cold" experts go to `Random` so the
/// kernel evicts them under memory pressure and page-faults them in
/// on-demand at forward time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertTier {
    Dense,
    MoeHot,
    MoeCold,
}

/// mmap 된 포인터를 가리키는 패킹 weight
pub struct PackedWeight {
    pub quant_type: QuantType,
    pub rows: usize,
    pub cols: usize,
    pub tier: ExpertTier,
    data_ptr: *const u8,
    data_len: usize,
}

// SAFETY: PackedWeight 는 mmap 된 메모리를 가리키는 포인터만 들고 있음.
// PackedModel이 _mmap 을 소유하며 PackedWeight보다 오래 살기 때문에 Send/Sync 안전.
unsafe impl Send for PackedWeight {}
unsafe impl Sync for PackedWeight {}

impl PackedWeight {
    pub fn data(&self) -> &[u8] {
        // SAFETY: data_ptr은 mmap에서 온 유효한 포인터이고,
        // PackedModel._mmap이 살아있는 동안만 PackedWeight가 존재함
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.data_len) }
    }
}

/// mmap 으로 .rnb 파일을 로드한 모델
pub struct PackedModel {
    _mmap: memmap2::Mmap,
    pub weights: Vec<(String, PackedWeight)>,
    pub metadata: Vec<u8>,
    /// Per-layer popularity / hot_count metadata when the model was loaded from
    /// a sidecar v3 file (`from_v3_sidecar`). `None` for legacy v2 (`RNBD`)
    /// loads where MoE residency was driven by `moe_preheat::parse_*` over the
    /// flat `metadata` byte blob.
    pub metadata_v3: Option<crate::sidecar_v3::spec::MetadataV3>,
}

impl PackedModel {
    pub fn open(path: &std::path::Path) -> io::Result<Self> {
        use std::fs::File;

        // 1. 파일 열고 mmap
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too small for RNB header",
            ));
        }

        // 2. 헤더 읽기 + 검증
        let header = read_header(&mmap)?;

        if header.magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid RNB magic",
            ));
        }
        if header.version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported RNB version: {}", header.version),
            ));
        }

        let num_tensors = header.num_tensors as usize;

        // 3. Tensor table 읽기
        let table_start = header.tensor_table_offset as usize;
        let table_end = table_start + num_tensors * 96;
        if mmap.len() < table_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file truncated: tensor table out of bounds",
            ));
        }

        let data_section_start = header.data_offset as usize;

        let mut weights = Vec::with_capacity(num_tensors);
        for i in 0..num_tensors {
            let entry_offset = table_start + i * 96;
            let entry = read_tensor_entry(&mmap, entry_offset)?;

            // name: null-terminated
            let name_bytes = &entry.name[..];
            let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(64);
            let name = String::from_utf8_lossy(&name_bytes[..name_len]).into_owned();

            let quant_type = QuantType::from_raw_u8(entry.quant_type).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown quant_type: {}", entry.quant_type),
                )
            })?;

            let data_start = data_section_start + entry.data_offset as usize;
            let data_len = entry.data_len as usize;
            let data_end = data_start + data_len;

            if mmap.len() < data_end {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("file truncated: tensor '{name}' data out of bounds"),
                ));
            }

            // SAFETY: mmap은 PackedModel이 소유하며, 포인터는 그 범위 안
            let data_ptr = mmap.as_ptr().wrapping_add(data_start);

            let pw = PackedWeight {
                quant_type,
                rows: entry.rows as usize,
                cols: entry.cols as usize,
                tier: ExpertTier::Dense,
                data_ptr,
                data_len,
            };
            weights.push((name, pw));
        }

        // 4. Metadata 복사
        let meta_start = header.metadata_offset as usize;
        let meta_len = header.metadata_len as usize;
        let meta_end = meta_start + meta_len;

        if mmap.len() < meta_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file truncated: metadata out of bounds",
            ));
        }
        let metadata = mmap[meta_start..meta_end].to_vec();

        Ok(PackedModel {
            _mmap: mmap,
            weights,
            metadata,
            metadata_v3: None,
        })
    }

    pub fn get_weight(&self, name: &str) -> Option<&PackedWeight> {
        self.weights.iter().find(|(n, _)| n == name).map(|(_, w)| w)
    }
    /// Releases clean pages for one immutable packed weight range.
    pub fn reclaim_weight_range(
        &self,
        name: &str,
        relative_byte_offset: usize,
        byte_len: usize,
    ) -> io::Result<()> {
        let weight = self.get_weight(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("packed weight not found: {name}"),
            )
        })?;
        let relative_end = relative_byte_offset.checked_add(byte_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "packed weight range overflow")
        })?;
        if relative_end > weight.data_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "packed weight reclaim range is out of bounds",
            ));
        }

        #[cfg(unix)]
        {
            let mmap_start = self._mmap.as_ptr() as usize;
            let weight_start = weight.data_ptr as usize;
            let mmap_offset = weight_start
                .checked_sub(mmap_start)
                .and_then(|offset| offset.checked_add(relative_byte_offset))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "packed mmap offset overflow")
                })?;
            // SAFETY: `PackedModel` owns an immutable file mapping and every
            // `PackedWeight` is validated to point inside it at load time.
            unsafe {
                self._mmap.unchecked_advise_range(
                    memmap2::UncheckedAdvice::DontNeed,
                    mmap_offset,
                    byte_len,
                )
            }?;
        }
        Ok(())
    }

    /// Look up the hot + cold MoE expert byte slices for a tensor name. Returns
    /// `None` when no matching entries exist. The hot slice is always present
    /// when MoE entries exist; cold may be empty for the all-hot case.
    ///
    /// When `metadata_v3` carries a layer entry matching the tensor name, the
    /// returned set also exposes the `popularity_order` (rank ↔ original_id)
    /// and `hot_count` chosen at convert time. Without metadata, the set
    /// assumes identity ordering and treats every expert as hot.
    pub fn get_moe_expert_set(&self, name: &str) -> Option<MoeExpertSet<'_>> {
        let mut hot: Option<&PackedWeight> = None;
        let mut cold: Option<&PackedWeight> = None;
        for (n, w) in &self.weights {
            if n != name {
                continue;
            }
            match w.tier {
                ExpertTier::MoeHot => hot = Some(w),
                ExpertTier::MoeCold => cold = Some(w),
                ExpertTier::Dense => {}
            }
        }
        let hot = hot?;

        // n_experts: total over hot + cold. PackedWeight.rows holds the
        // 3D shape's first dimension (n_experts) by construction in
        // `from_v3_sidecar` (`rows = entry.shape[0]`).
        let n_experts = hot.rows;

        // Layer popularity entry from sidecar metadata, when available.
        // Tensor name has the form `blk.{N}.ffn_*_exps.weight`; we strip the
        // `blk.` prefix and parse the next dotted segment to extract layer_idx.
        let layer_entry = self.metadata_v3.as_ref().and_then(|m| {
            let idx = parse_blk_idx(name)?;
            m.layers.iter().find(|e| e.layer_idx == idx)
        });

        let hot_count_in_hot_slice = match layer_entry {
            Some(e) => e.hot_count as usize,
            // No metadata = all-hot single-tensor encode (cold == None).
            // hot_count_in_hot_slice == n_experts so per_expert divides cleanly.
            None => n_experts,
        };
        if hot_count_in_hot_slice == 0 {
            return None;
        }
        if hot.data_len % hot_count_in_hot_slice != 0 {
            return None;
        }
        let per_expert_bytes = hot.data_len / hot_count_in_hot_slice;
        let popularity_order = layer_entry.map(|e| e.popularity_order.as_slice());
        let hot_count = match layer_entry {
            Some(e) => e.hot_count as usize,
            None => n_experts,
        };

        Some(MoeExpertSet {
            hot,
            cold,
            n_experts,
            hot_count,
            popularity_order,
            per_expert_bytes,
        })
    }
}

/// Extract `blk.{N}` from a sidecar tensor name like
/// `blk.5.ffn_gate_exps.weight`. Returns `None` for non-layer tensors
/// (`output.weight`, `token_embd.weight`, ...).
fn parse_blk_idx(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Hot + cold expert byte slices for a single MoE tensor name with the
/// popularity / hot_count metadata that drives forward dispatch.
///
/// Both `hot` and `cold` slices share the same per-expert layout
/// (`pack_q4k_compact` output, repeated as many times as that slice's expert
/// count). The hot slice is RAM-resident (`MADV_WILLNEED`); the cold slice is
/// `MADV_RANDOM` so the kernel evicts it under memory pressure and pages back
/// in on demand.
///
/// `expert_bytes(rank)` (via the `MoeExpertResidencyView` impl) translates a
/// 0-based rank into the right `&[u8]`: `rank < hot_count` → hot slice,
/// `rank >= hot_count` → cold slice (offset by `rank - hot_count`).
pub struct MoeExpertSet<'a> {
    pub hot: &'a PackedWeight,
    pub cold: Option<&'a PackedWeight>,
    pub n_experts: usize,
    pub hot_count: usize,
    pub popularity_order: Option<&'a [u32]>,
    pub per_expert_bytes: usize,
}

/// `HotByteSource` impl backed by a `PackedModel`'s `MoeQ4kHot` tensor for
/// the given name. Resolves the `PackedWeight` lazily on each call (Vec scan
/// over a few entries, cheap relative to a GEMV).
pub struct PackedModelHotSource {
    pub pm: Arc<PackedModel>,
    pub name: String,
}

impl rnb_memory::moe_residency::HotByteSource for PackedModelHotSource {
    fn hot(&self, rank: usize, per_expert: usize) -> Cow<'_, [u8]> {
        for (n, w) in &self.pm.weights {
            if n == &self.name && w.tier == ExpertTier::MoeHot {
                return Cow::Borrowed(&w.data()[rank * per_expert..(rank + 1) * per_expert]);
            }
        }
        panic!(
            "PackedModelHotSource: MoeHot tensor `{}` not found",
            self.name
        );
    }
}

/// `ColdByteSource` impl backed by a `PackedModel`'s `MoeQ4kCold` tensor.
pub struct PackedModelColdSource {
    pub pm: Arc<PackedModel>,
    pub name: String,
}

impl rnb_memory::moe_residency::ColdByteSource for PackedModelColdSource {
    fn cold(&self, cold_rank: usize, per_expert: usize) -> Cow<'_, [u8]> {
        for (n, w) in &self.pm.weights {
            if n == &self.name && w.tier == ExpertTier::MoeCold {
                return Cow::Borrowed(
                    &w.data()[cold_rank * per_expert..(cold_rank + 1) * per_expert],
                );
            }
        }
        panic!(
            "PackedModelColdSource: MoeCold tensor `{}` not found",
            self.name
        );
    }
}

/// Owned residency view holding an `Arc<PackedModel>` + the tensor name +
/// resolved hot/cold metadata, suitable to embed in engine layer weights as
/// `Arc<dyn MoeExpertResidencyView>`.
///
/// Borrow-style `MoeExpertSet` keeps lifetimes tied to a `&PackedModel` and
/// is fine for one-shot lookups, but engine layer wiring needs an owned
/// handle that lives as long as the engine itself. This struct wraps the
/// same byte routing logic over an `Arc<PackedModel>` so consumers see the
/// same `MoeExpertResidencyView` surface.
pub struct OwnedMoeResidency {
    pm: Arc<PackedModel>,
    name: String,
    n_experts: usize,
    hot_count: usize,
    popularity_order: Option<Vec<u32>>,
    per_expert_bytes: usize,
}

impl MoeExpertResidencyView for OwnedMoeResidency {
    fn n_experts(&self) -> usize {
        self.n_experts
    }
    fn hot_count(&self) -> usize {
        self.hot_count
    }
    fn popularity_order(&self) -> Option<&[u32]> {
        self.popularity_order.as_deref()
    }
    fn per_expert_bytes(&self) -> usize {
        self.per_expert_bytes
    }
    fn expert_bytes(&self, rank: usize) -> Cow<'_, [u8]> {
        assert!(
            rank < self.n_experts,
            "rank {rank} out of bounds for n_experts {}",
            self.n_experts
        );
        let per = self.per_expert_bytes;
        let mut hot: Option<&PackedWeight> = None;
        let mut cold: Option<&PackedWeight> = None;
        for (n, w) in &self.pm.weights {
            if n != self.name.as_str() {
                continue;
            }
            match w.tier {
                ExpertTier::MoeHot => hot = Some(w),
                ExpertTier::MoeCold => cold = Some(w),
                ExpertTier::Dense => {}
            }
        }
        let hot = hot.expect("MoE hot tier must remain present");
        if rank < self.hot_count {
            Cow::Borrowed(&hot.data()[rank * per..(rank + 1) * per])
        } else {
            let cold = cold.expect("rank >= hot_count requires a cold slice");
            let cold_rank = rank - self.hot_count;
            Cow::Borrowed(&cold.data()[cold_rank * per..(cold_rank + 1) * per])
        }
    }
}

impl PackedModel {
    /// Build an owned `Arc<dyn MoeExpertResidencyView>` for a MoE tensor name,
    /// suitable for embedding in engine layer weights. Returns `None` when
    /// the tensor is not present, the v3 metadata is missing, or per-expert
    /// stride math fails.
    pub fn moe_residency_view(
        self: &Arc<Self>,
        name: &str,
    ) -> Option<Arc<dyn MoeExpertResidencyView>> {
        let set = self.get_moe_expert_set(name)?;
        let owned = OwnedMoeResidency {
            pm: Arc::clone(self),
            name: name.to_string(),
            n_experts: set.n_experts,
            hot_count: set.hot_count,
            popularity_order: set.popularity_order.map(|s| s.to_vec()),
            per_expert_bytes: set.per_expert_bytes,
        };
        Some(Arc::new(owned))
    }
}

impl<'a> MoeExpertResidencyView for MoeExpertSet<'a> {
    fn n_experts(&self) -> usize {
        self.n_experts
    }
    fn hot_count(&self) -> usize {
        self.hot_count
    }
    fn popularity_order(&self) -> Option<&[u32]> {
        self.popularity_order
    }
    fn per_expert_bytes(&self) -> usize {
        self.per_expert_bytes
    }
    fn expert_bytes(&self, rank: usize) -> Cow<'_, [u8]> {
        assert!(
            rank < self.n_experts,
            "rank {rank} out of bounds for n_experts {}",
            self.n_experts
        );
        let per = self.per_expert_bytes;
        if rank < self.hot_count {
            Cow::Borrowed(&self.hot.data()[rank * per..(rank + 1) * per])
        } else {
            let cold = self
                .cold
                .expect("rank >= hot_count requires a cold slice")
                .data();
            let cold_rank = rank - self.hot_count;
            Cow::Borrowed(&cold[cold_rank * per..(cold_rank + 1) * per])
        }
    }
}

impl PackedModel {
    /// Decode a v3 sidecar file (`RNBC` magic) into a `PackedModel`.
    ///
    /// v3 supports dense Q4_K row-pair (`pack_q4k`) and dense Q6_K
    /// (`pack_q6k`) tensors. MoE Q4_K hot/cold entries carry raw GGUF Q4_K
    /// expert bytes plus tier metadata.
    ///
    /// The returned `PackedModel` keeps the v3 sidecar's mmap alive so the
    /// `PackedWeight` raw pointers remain valid for the model's lifetime.
    pub fn from_v3_sidecar(path: &std::path::Path) -> io::Result<Self> {
        let decoded = crate::sidecar_v3::decoder::decode_sidecar_v3(path)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut weights: Vec<(String, PackedWeight)> = Vec::with_capacity(decoded.tensors.len());
        for entry in &decoded.tensors {
            let quant_type = match entry.quant_type {
                // v3 encoder writes dense Q4_K via `pack_q4k` (row-pair interleaved,
                // 2240B per 8-row super-block) so the engine routes through
                // `QuantType::Q4K` → `gemv_vec_q8k_packed` → `gemv_q4k_packed_i8mm`,
                // taking advantage of i8mm `vmmlaq_s32` on ARMv8.6+.
                crate::sidecar_v3::spec::V3QuantType::DenseQ4kRowPair => QuantType::Q4K,
                crate::sidecar_v3::spec::V3QuantType::DenseQ6k => QuantType::Q6K,
                // MoE hot/cold experts stay in raw GGUF Q4_K layout — the
                // encoder copies them as-is. Engine MoE forward indexes
                // per-expert bytes and runs `dot_k_block_row(_, _, _, _,
                // GGMLType::Q4_K)`, which expects raw GGUF blocks. Routing
                // through `Q4KCompact` would force the packed-GEMM dispatch
                // and mis-decode the bytes.
                crate::sidecar_v3::spec::V3QuantType::MoeQ4kHot
                | crate::sidecar_v3::spec::V3QuantType::MoeQ4kCold => QuantType::Q4K,
            };
            let tier = match entry.quant_type {
                crate::sidecar_v3::spec::V3QuantType::DenseQ4kRowPair
                | crate::sidecar_v3::spec::V3QuantType::DenseQ6k => ExpertTier::Dense,
                crate::sidecar_v3::spec::V3QuantType::MoeQ4kHot => ExpertTier::MoeHot,
                crate::sidecar_v3::spec::V3QuantType::MoeQ4kCold => ExpertTier::MoeCold,
            };
            let rows = *entry.shape.first().unwrap_or(&0) as usize;
            let cols = *entry.shape.get(1).unwrap_or(&0) as usize;
            let data_len = entry.payload_size as usize;
            // SAFETY: decoder validated payload_offset + payload_size <= mmap.len();
            // the mmap is owned by `decoded` and moved into the returned PackedModel.
            let data_ptr = unsafe { decoded.mmap.as_ptr().add(entry.payload_offset as usize) };
            // Apply residency hints by tier. Linux/Android `madvise` lets us
            // tell the kernel which pages should be kept resident vs evicted
            // under memory pressure. WillNeed = read-ahead + keep resident
            // (hot/dense path). Random = no read-ahead, evict eagerly (cold).
            #[cfg(unix)]
            unsafe {
                use libc::{c_void, madvise, MADV_RANDOM, MADV_WILLNEED};
                let advice = match tier {
                    ExpertTier::MoeCold => MADV_RANDOM,
                    ExpertTier::Dense | ExpertTier::MoeHot => MADV_WILLNEED,
                };
                let _ = madvise(data_ptr as *mut c_void, data_len, advice);
            }
            weights.push((
                entry.name.clone(),
                PackedWeight {
                    quant_type,
                    rows,
                    cols,
                    tier,
                    data_ptr,
                    data_len,
                },
            ));
        }
        Ok(PackedModel {
            _mmap: decoded.mmap,
            weights,
            metadata: Vec::new(),
            metadata_v3: decoded.metadata,
        })
    }
}

/// mmap 슬라이스에서 헤더를 파싱
fn read_header(mmap: &[u8]) -> io::Result<RnbHeader> {
    // SAFETY: RnbHeader는 repr(C), size=64, mmap 범위 체크 완료
    if mmap.len() < std::mem::size_of::<RnbHeader>() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file too small for header",
        ));
    }
    let header = unsafe { std::ptr::read_unaligned(mmap.as_ptr() as *const RnbHeader) };
    Ok(header)
}

/// mmap 슬라이스의 offset 위치에서 TensorEntry 파싱
fn read_tensor_entry(mmap: &[u8], offset: usize) -> io::Result<TensorEntry> {
    let size = std::mem::size_of::<TensorEntry>();
    if mmap.len() < offset + size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tensor entry out of bounds",
        ));
    }
    // SAFETY: TensorEntry는 repr(C), 범위 체크 완료
    let entry = unsafe {
        std::ptr::read_unaligned(mmap.as_ptr().wrapping_add(offset) as *const TensorEntry)
    };
    Ok(entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rnb_file::write_rnb;
    use rnb_core::tensor::QuantType;

    fn make_temp_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        dir.join(format!("rnb_test_{name}_{}.rnb", std::process::id()))
    }

    // 임시 파일 자동 삭제 헬퍼
    struct TempFile(std::path::PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn roundtrip_empty() {
        let path = make_temp_path("empty");
        let _guard = TempFile(path.clone());

        write_rnb(&path, &[], &[]).unwrap();
        let model = PackedModel::open(&path).unwrap();
        assert_eq!(model.weights.len(), 0);
        assert_eq!(model.metadata.len(), 0);
    }

    #[test]
    fn roundtrip_empty_with_metadata() {
        let path = make_temp_path("empty_meta");
        let _guard = TempFile(path.clone());

        let meta = b"some metadata bytes 12345";
        write_rnb(&path, &[], meta).unwrap();
        let model = PackedModel::open(&path).unwrap();
        assert_eq!(model.weights.len(), 0);
        assert_eq!(model.metadata, meta);
    }

    #[test]
    fn roundtrip_single_tensor() {
        let path = make_temp_path("single");
        let _guard = TempFile(path.clone());

        let packed_data: Vec<u8> = (0u8..=255).collect();
        let tensors = vec![(
            "model.layers.0.weight".to_string(),
            QuantType::Q4K,
            16usize,
            8usize,
            packed_data.clone(),
        )];
        let meta = b"metadata content here";
        write_rnb(&path, &tensors, meta).unwrap();

        let model = PackedModel::open(&path).unwrap();
        assert_eq!(model.weights.len(), 1);
        assert_eq!(model.metadata, meta);

        let (name, pw) = &model.weights[0];
        assert_eq!(name, "model.layers.0.weight");
        assert_eq!(pw.quant_type, QuantType::Q4K);
        assert_eq!(pw.rows, 16);
        assert_eq!(pw.cols, 8);
        assert_eq!(pw.data(), packed_data.as_slice());
    }

    #[test]
    fn roundtrip_multiple_tensors() {
        let path = make_temp_path("multi");
        let _guard = TempFile(path.clone());

        let tensors = vec![
            (
                "layer0.q4k".to_string(),
                QuantType::Q4K,
                8usize,
                4usize,
                vec![0xAAu8; 512],
            ),
            (
                "layer1.q5k".to_string(),
                QuantType::Q5K,
                16usize,
                8usize,
                vec![0xBBu8; 1024],
            ),
            (
                "layer2.q6k".to_string(),
                QuantType::Q6K,
                32usize,
                16usize,
                vec![0xCCu8; 2048],
            ),
        ];
        let meta = b"multi tensor metadata";
        write_rnb(&path, &tensors, meta).unwrap();

        let model = PackedModel::open(&path).unwrap();
        assert_eq!(model.weights.len(), 3);
        assert_eq!(model.metadata, meta);

        // 순서 확인
        assert_eq!(model.weights[0].0, "layer0.q4k");
        assert_eq!(model.weights[1].0, "layer1.q5k");
        assert_eq!(model.weights[2].0, "layer2.q6k");

        // 타입 확인
        assert_eq!(model.weights[0].1.quant_type, QuantType::Q4K);
        assert_eq!(model.weights[1].1.quant_type, QuantType::Q5K);
        assert_eq!(model.weights[2].1.quant_type, QuantType::Q6K);

        // 데이터 확인
        assert!(model.weights[0].1.data().iter().all(|&b| b == 0xAA));
        assert!(model.weights[1].1.data().iter().all(|&b| b == 0xBB));
        assert!(model.weights[2].1.data().iter().all(|&b| b == 0xCC));

        // get_weight 헬퍼
        let w = model.get_weight("layer1.q5k").unwrap();
        assert_eq!(w.rows, 16);
        assert_eq!(w.cols, 8);
        assert_eq!(w.data().len(), 1024);

        assert!(model.get_weight("nonexistent").is_none());
    }

    #[test]
    fn roundtrip_q4k_compact_quant_type() {
        let path = make_temp_path("q4k_compact_type");
        let _guard = TempFile(path.clone());

        let same_len = 8192;
        let compact_data = vec![0x11u8; same_len];
        let tensors = vec![(
            "compact.weight".to_string(),
            QuantType::Q4KCompact,
            8usize,
            1usize,
            compact_data.clone(),
        )];
        write_rnb(&path, &tensors, b"").unwrap();

        let model = PackedModel::open(&path).unwrap();
        let compact = model.get_weight("compact.weight").unwrap();

        assert_eq!(compact.quant_type, QuantType::Q4KCompact);
        assert_eq!(compact.data(), compact_data.as_slice());
    }

    #[test]
    fn roundtrip_data_alignment() {
        // data section이 4096-byte 정렬되어 있는지 확인
        let path = make_temp_path("align");
        let _guard = TempFile(path.clone());

        let tensors = vec![(
            "w".to_string(),
            QuantType::Q6K,
            4usize,
            2usize,
            vec![0x42u8; 128],
        )];
        write_rnb(&path, &tensors, b"").unwrap();

        // 파일의 헤더에서 data_offset 읽어 4096 배수인지 체크
        let raw = std::fs::read(&path).unwrap();
        // data_offset은 헤더 offset 16 (magic4 + version4 + num_tensors4 = 12? 아니 ── 구조체 보면)
        // RnbHeader: magic[4], version[4], num_tensors[4] → 12byte but repr(C) padding?
        // magic[4] + version(u32)[4] + num_tensors(u32)[4] → 12, tensor_table_offset(u64) → 8byte align → pad 4
        // 실제로는 read_header로 파싱
        let header = read_header(&raw).unwrap();
        assert_eq!(header.data_offset % 4096, 0);
        assert_eq!(raw[header.data_offset as usize], 0x42);
    }

    #[test]
    fn roundtrip_large_data() {
        let path = make_temp_path("large");
        let _guard = TempFile(path.clone());

        // 좀 큰 데이터 (64KB)
        let big_data: Vec<u8> = (0..65536).map(|i| (i % 256) as u8).collect();
        let tensors = vec![(
            "big.tensor".to_string(),
            QuantType::Q4K,
            256usize,
            128usize,
            big_data.clone(),
        )];
        write_rnb(&path, &tensors, b"big tensor meta").unwrap();

        let model = PackedModel::open(&path).unwrap();
        assert_eq!(model.weights[0].1.data(), big_data.as_slice());
    }

    #[test]
    fn from_v3_sidecar_round_trips_dense_q4k_q6k() {
        use crate::sidecar_v3::spec::{
            HeaderV3, V3QuantType, FORMAT_VERSION_V3, HEADER_SIZE_BYTES, MAGIC,
        };

        // Build a tiny v3 sidecar with one Q4_K dense tensor.
        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0u8; HEADER_SIZE_BYTES]);
        let table_offset = HEADER_SIZE_BYTES as u64;

        // Tensor table entry
        let name = "blk.0.attn_q.weight";
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.push(V3QuantType::DenseQ4kRowPair as u8);
        buf.extend_from_slice(&2u32.to_le_bytes()); // ndim = 2
        buf.extend_from_slice(&64u64.to_le_bytes()); // shape[0] = 64 rows
        buf.extend_from_slice(&256u64.to_le_bytes()); // shape[1] = 256 cols
        let payload_offset_pos = buf.len();
        buf.extend_from_slice(&[0u8; 8]);
        let payload_size = 144u64; // dummy single-block payload
        buf.extend_from_slice(&payload_size.to_le_bytes());
        let payload_offset = buf.len() as u64;
        buf[payload_offset_pos..payload_offset_pos + 8]
            .copy_from_slice(&payload_offset.to_le_bytes());
        buf.extend_from_slice(&vec![0xABu8; payload_size as usize]);

        let header = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 1,
            tensor_table_offset: table_offset,
            payload_offset,
            metadata_offset: 0,
            metadata_size: 0,
        };
        buf[..HEADER_SIZE_BYTES].copy_from_slice(&header.to_bytes());
        std::fs::write(f.path(), &buf).unwrap();

        let pm = PackedModel::from_v3_sidecar(f.path()).expect("v3 adapter must succeed");
        assert_eq!(pm.weights.len(), 1);
        let (name, w) = &pm.weights[0];
        assert_eq!(name, "blk.0.attn_q.weight");
        assert_eq!(w.quant_type, QuantType::Q4K);
        assert_eq!(w.rows, 64);
        assert_eq!(w.cols, 256);
        assert_eq!(w.data().len(), payload_size as usize);
        assert!(w.data().iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn from_v3_sidecar_maps_dense_q6k() {
        use crate::sidecar_v3::spec::{
            HeaderV3, V3QuantType, FORMAT_VERSION_V3, HEADER_SIZE_BYTES, MAGIC,
        };

        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0u8; HEADER_SIZE_BYTES]);
        let table_offset = HEADER_SIZE_BYTES as u64;

        let name = "output.weight";
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.push(V3QuantType::DenseQ6k as u8);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&32u64.to_le_bytes());
        buf.extend_from_slice(&256u64.to_le_bytes());
        let payload_offset_pos = buf.len();
        buf.extend_from_slice(&[0u8; 8]);
        let payload_size = 210u64;
        buf.extend_from_slice(&payload_size.to_le_bytes());
        let payload_offset = buf.len() as u64;
        buf[payload_offset_pos..payload_offset_pos + 8]
            .copy_from_slice(&payload_offset.to_le_bytes());
        buf.extend_from_slice(&vec![0xCDu8; payload_size as usize]);

        let header = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 1,
            tensor_table_offset: table_offset,
            payload_offset,
            metadata_offset: 0,
            metadata_size: 0,
        };
        buf[..HEADER_SIZE_BYTES].copy_from_slice(&header.to_bytes());
        std::fs::write(f.path(), &buf).unwrap();

        let pm = PackedModel::from_v3_sidecar(f.path()).expect("v3 adapter must succeed");
        assert_eq!(pm.weights.len(), 1);
        let (name, w) = &pm.weights[0];
        assert_eq!(name, "output.weight");
        assert_eq!(w.quant_type, QuantType::Q6K);
        assert_eq!(w.rows, 32);
        assert_eq!(w.cols, 256);
        assert_eq!(w.data().len(), payload_size as usize);
    }

    #[test]
    fn from_v3_sidecar_maps_moe_q4k_hot_to_raw_q4k() {
        use crate::sidecar_v3::spec::{
            HeaderV3, V3QuantType, FORMAT_VERSION_V3, HEADER_SIZE_BYTES, MAGIC,
        };

        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0u8; HEADER_SIZE_BYTES]);
        let table_offset = HEADER_SIZE_BYTES as u64;

        let name = "blk.0.ffn_gate_exps.weight";
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.push(V3QuantType::MoeQ4kHot as u8);
        buf.extend_from_slice(&3u32.to_le_bytes()); // ndim = 3 (n_experts, rows, cols)
        buf.extend_from_slice(&8u64.to_le_bytes()); // n_experts = 8
        buf.extend_from_slice(&64u64.to_le_bytes()); // rows = 64
        buf.extend_from_slice(&256u64.to_le_bytes()); // cols = 256
        let payload_offset_pos = buf.len();
        buf.extend_from_slice(&[0u8; 8]);
        let payload_size = 16u64;
        buf.extend_from_slice(&payload_size.to_le_bytes());
        let payload_offset = buf.len() as u64;
        buf[payload_offset_pos..payload_offset_pos + 8]
            .copy_from_slice(&payload_offset.to_le_bytes());
        buf.extend_from_slice(&vec![0u8; payload_size as usize]);

        let header = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 1,
            tensor_table_offset: table_offset,
            payload_offset,
            metadata_offset: 0,
            metadata_size: 0,
        };
        buf[..HEADER_SIZE_BYTES].copy_from_slice(&header.to_bytes());
        std::fs::write(f.path(), &buf).unwrap();

        let pm = PackedModel::from_v3_sidecar(f.path()).expect("MoeQ4kHot must map to raw Q4K");
        assert_eq!(pm.weights.len(), 1);
        assert_eq!(pm.weights[0].1.quant_type, QuantType::Q4K);
    }

    #[test]
    #[ignore = "MoeQ4kCold is now accepted as raw Q4K; covered by moe_expert_set_routes_hot_and_cold_ranks_via_residency_view"]
    fn from_v3_sidecar_rejects_moe_q4k_cold() {
        use crate::sidecar_v3::spec::{
            HeaderV3, V3QuantType, FORMAT_VERSION_V3, HEADER_SIZE_BYTES, MAGIC,
        };

        let f = tempfile::NamedTempFile::new().unwrap();
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&[0u8; HEADER_SIZE_BYTES]);
        let table_offset = HEADER_SIZE_BYTES as u64;

        let name = "blk.0.ffn_gate_exps.weight";
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.push(V3QuantType::MoeQ4kCold as u8);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&64u64.to_le_bytes());
        let payload_offset_pos = buf.len();
        buf.extend_from_slice(&[0u8; 8]);
        let payload_size = 16u64;
        buf.extend_from_slice(&payload_size.to_le_bytes());
        let payload_offset = buf.len() as u64;
        buf[payload_offset_pos..payload_offset_pos + 8]
            .copy_from_slice(&payload_offset.to_le_bytes());
        buf.extend_from_slice(&vec![0u8; payload_size as usize]);

        let header = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 1,
            tensor_table_offset: table_offset,
            payload_offset,
            metadata_offset: 0,
            metadata_size: 0,
        };
        buf[..HEADER_SIZE_BYTES].copy_from_slice(&header.to_bytes());
        std::fs::write(f.path(), &buf).unwrap();

        let err = match PackedModel::from_v3_sidecar(f.path()) {
            Ok(_) => panic!("MoeQ4kCold should not be supported in Phase 1"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    /// Build a sidecar with one MoE hot tensor (4 experts) + one cold tensor
    /// (4 experts) sharing the same logical name, plus a metadata blob carrying
    /// `hot_count = 4` and a popularity_order. Returns the temp file holding the
    /// bytes.
    fn write_sidecar_with_hot_cold_and_metadata() -> tempfile::NamedTempFile {
        use crate::sidecar_v3::spec::{
            HeaderV3, LayerPopularityEntry, MetadataV3, V3QuantType, FORMAT_VERSION_V3,
            HEADER_SIZE_BYTES, MAGIC,
        };
        let f = tempfile::NamedTempFile::new().unwrap();
        let name = "blk.0.ffn_gate_exps.weight";
        let name_bytes = name.as_bytes();

        // 4 hot experts, each 4 bytes; 4 cold experts, each 4 bytes.
        let per_expert = 4usize;
        let hot_n = 4usize;
        let cold_n = 4usize;
        let mut hot_payload = Vec::with_capacity(hot_n * per_expert);
        for e in 0..hot_n {
            // Each hot expert 'e' is 4 bytes all equal to (10 + e).
            hot_payload.extend_from_slice(&[(10 + e) as u8; 4]);
        }
        let mut cold_payload = Vec::with_capacity(cold_n * per_expert);
        for e in 0..cold_n {
            cold_payload.extend_from_slice(&[(20 + e) as u8; 4]);
        }

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&vec![0u8; HEADER_SIZE_BYTES]);
        let table_offset = HEADER_SIZE_BYTES as u64;

        // Hot entry
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.push(V3QuantType::MoeQ4kHot as u8);
        buf.extend_from_slice(&3u32.to_le_bytes()); // ndim
        buf.extend_from_slice(&((hot_n + cold_n) as u64).to_le_bytes()); // n_experts dim (total)
        buf.extend_from_slice(&1u64.to_le_bytes()); // rows (dummy)
        buf.extend_from_slice(&256u64.to_le_bytes()); // cols (dummy)
        let hot_payload_off_pos = buf.len();
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&(hot_payload.len() as u64).to_le_bytes());

        // Cold entry
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.push(V3QuantType::MoeQ4kCold as u8);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&((hot_n + cold_n) as u64).to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&256u64.to_le_bytes());
        let cold_payload_off_pos = buf.len();
        buf.extend_from_slice(&[0u8; 8]);
        buf.extend_from_slice(&(cold_payload.len() as u64).to_le_bytes());

        let payload_offset = buf.len() as u64;
        // Patch hot payload offset
        buf[hot_payload_off_pos..hot_payload_off_pos + 8]
            .copy_from_slice(&payload_offset.to_le_bytes());
        let cold_payload_offset = payload_offset + hot_payload.len() as u64;
        buf[cold_payload_off_pos..cold_payload_off_pos + 8]
            .copy_from_slice(&cold_payload_offset.to_le_bytes());

        // Payloads
        buf.extend_from_slice(&hot_payload);
        buf.extend_from_slice(&cold_payload);

        // Metadata blob
        let metadata = MetadataV3 {
            layers: vec![LayerPopularityEntry {
                layer_idx: 0,
                hot_count: hot_n as u32,
                // Identity popularity: rank == original_id.
                popularity_order: (0..(hot_n + cold_n) as u32).collect(),
            }],
            transformed_layouts: Vec::new(),
        };
        let metadata_offset = buf.len() as u64;
        let metadata_bytes = metadata.to_bytes();
        let metadata_size = metadata_bytes.len() as u64;
        buf.extend_from_slice(&metadata_bytes);

        // Patch header
        let header = HeaderV3 {
            magic: MAGIC,
            version: FORMAT_VERSION_V3,
            tensor_count: 2,
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
    fn moe_expert_set_routes_hot_and_cold_ranks_via_residency_view() {
        use rnb_memory::moe_residency::MoeExpertResidencyView;

        let f = write_sidecar_with_hot_cold_and_metadata();
        let pm = PackedModel::from_v3_sidecar(f.path()).expect("v3 sidecar with hot+cold");
        let set = pm
            .get_moe_expert_set("blk.0.ffn_gate_exps.weight")
            .expect("MoE expert set must be present");

        // Sanity over the trait surface.
        assert_eq!(set.n_experts(), 8);
        assert_eq!(set.hot_count(), 4);
        assert_eq!(set.per_expert_bytes(), 4);
        let order = set.popularity_order().expect("metadata supplies order");
        assert_eq!(order, &[0u32, 1, 2, 3, 4, 5, 6, 7]);

        // Hot ranks (0..4): bytes (10+rank).repeat(4)
        for rank in 0..4 {
            let bytes = set.expert_bytes(rank);
            assert_eq!(&*bytes, &[(10 + rank) as u8; 4]);
        }
        // Cold ranks (4..8): bytes (20+(rank-4)).repeat(4)
        for rank in 4..8 {
            let cold_rank = rank - 4;
            let bytes = set.expert_bytes(rank);
            assert_eq!(&*bytes, &[(20 + cold_rank) as u8; 4]);
        }
    }
}
