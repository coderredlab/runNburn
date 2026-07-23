use crate::context::{GpuBuffer, VulkanContext};
use crate::ffi::types::*;
use crate::gemv::{
    repack_q4k_transposed, repack_q5k_transposed, repack_q6k_transposed, repack_q8_0_transposed,
};
use std::collections::HashMap;

/// Weight identifier: (layer_index, weight_type)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeightId {
    pub layer: u16,
    pub kind: WeightKind,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WeightKind {
    QProj,
    QBias,
    QNorm,
    KNorm,
    KProj,
    KBias,
    VProj,
    VBias,
    OProj,
    FfnGate,
    FfnUp,
    FfnDown,
    GdnQkv,
    GdnGate,
    GdnAlpha,
    GdnBeta,
    GdnSsmOut,
    /// GDN pre-attn RMS norm weight (Raw32). Distinct from `AttnNorm` so that
    /// hybrid models with both Attention and Recurrent layers can keep cache
    /// keys disjoint (mv28-task10b-5b).
    GdnAttnNorm,
    /// GDN post-attn RMS norm weight (Raw32) — sits in the FFN-norm slot of
    /// the Recurrent path.
    GdnPostAttnNorm,
    /// GDN `A_log` per-head Raw32 vector — `[num_heads]`.
    GdnSsmA,
    /// GDN conv1d kernel Raw32 — `[conv_kernel, conv_channels]`.
    GdnSsmConv1d,
    /// GDN Δt bias Raw32 — `[num_heads]`.
    GdnSsmDtBias,
    /// GDN per-head-dim RMS norm Raw32 — `[head_v_dim]`.
    GdnSsmNorm,
    OutputLogits,
    /// Attention RMS norm weight (pre-attn layernorm)
    AttnNorm,
    /// FFN RMS norm weight (pre-ffn layernorm)
    FfnNorm,
    /// Per-kv-head K projection shard (kvh index)
    KProjShard(u16),
    /// Per-kv-head V projection shard (kvh index)
    VProjShard(u16),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2_XXS,
    IQ2_XS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ3_XXS,
    IQ4_XS,
    IQ1_M,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
    Q1_0,
    Q2_0,
}

impl QuantType {
    pub const fn block_elements(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::NVFP4 | Self::Q2_0 => 64,
            Self::Q1_0 => 128,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ4_XS
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => 256,
        }
    }

    pub const fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 36,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ1_S => 50,
            Self::IQ4_NL => 18,
            Self::IQ3_S => 110,
            Self::IQ2_S => 82,
            Self::IQ3_XXS => 98,
            Self::IQ4_XS => 136,
            Self::IQ1_M => 56,
            Self::TQ1_0 => 54,
            Self::TQ2_0 => 66,
            Self::MXFP4 => 17,
            Self::NVFP4 => 36,
            Self::Q1_0 => 18,
            Self::Q2_0 => 18,
        }
    }

    pub const fn has_soa_kernel(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8_0)
    }
}

/// GPU weight 메모리 레이아웃 모드.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWeightMode {
    /// SoA repack + device-local.
    Soa,
    /// Original GGUF row-major bytes in device-local memory.
    RowMajor,
    /// Pre-converted f32 weights (RMS-norm 등) — no repack, no dequant; the
    /// quant tag is ignored on this path. Host-visible buffer, direct upload.
    Raw32,
}

/// Resolve `(mode, quant, shape)` to the effective upload layout.
///
/// Q4_K row-major wins only when the projection is at least as wide as it is
/// tall. Tall projections retain coalesced SoA packing. Formats without an SoA
/// kernel always keep their native bytes.
pub(crate) fn effective_upload_mode(
    mode: GpuWeightMode,
    quant: QuantType,
    rows: u32,
    cols: u32,
) -> GpuWeightMode {
    match mode {
        GpuWeightMode::Raw32 => GpuWeightMode::Raw32,
        GpuWeightMode::RowMajor if quant == QuantType::Q4K && cols >= rows => {
            GpuWeightMode::RowMajor
        }
        GpuWeightMode::RowMajor if !quant.has_soa_kernel() => GpuWeightMode::RowMajor,
        GpuWeightMode::RowMajor => GpuWeightMode::Soa,
        GpuWeightMode::Soa if quant.has_soa_kernel() => GpuWeightMode::Soa,
        GpuWeightMode::Soa => GpuWeightMode::RowMajor,
    }
}

struct CachedWeight {
    buf: GpuBuffer,
    last_used: u64,
}

pub struct GpuWeightCache {
    entries: HashMap<WeightId, CachedWeight>,
    total_cached: u64,
    budget: u64,
    tick: u64,
    staging_buf: Option<GpuBuffer>,
}

impl GpuWeightCache {
    /// Create a new cache with given memory budget in bytes.
    pub fn new(budget: u64) -> Self {
        Self {
            entries: HashMap::new(),
            total_cached: 0,
            budget,
            tick: 0,
            staging_buf: None,
        }
    }

    /// Get cached buffer or repack+upload from raw_bytes.
    /// Evicts LRU entries if over budget.
    ///
    /// - `GpuWeightMode::Soa`: repack to transposed SoA layout → staging → device-local
    /// - `GpuWeightMode::RowMajor`: select Q4_K layout from matrix shape and upload
    ///   the chosen representation to device-local memory.
    /// - `GpuWeightMode::Raw32`: raw f32 bytes를 host-visible 버퍼에 그대로 업로드.
    ///   norm weight 같은 dequantized scalar 경로용. `quant` 태그는 무시됨.
    ///
    /// # Safety
    /// Must be called with a valid VulkanContext and command_pool.
    pub unsafe fn get_or_upload(
        &mut self,
        ctx: &VulkanContext,
        command_pool: VkCommandPool,
        id: WeightId,
        raw_bytes: &[u8],
        rows: u32,
        cols: u32,
        quant: QuantType,
        mode: GpuWeightMode,
    ) -> Result<&GpuBuffer, String> {
        // Fast path: already cached — update tick and return (mode 무관)
        if self.entries.contains_key(&id) {
            self.tick += 1;
            self.entries.get_mut(&id).unwrap().last_used = self.tick;
            return Ok(&self.entries[&id].buf);
        }

        // RowMajor Q4_K uses the measured shape policy; other SoA kernels keep
        // their existing packing. Raw32 always takes the direct byte-copy path.
        let effective_mode = effective_upload_mode(mode, quant, rows, cols);

        match effective_mode {
            GpuWeightMode::Soa => {
                // Repack weights to transposed SoA layout
                let repacked: Vec<u32> = match quant {
                    QuantType::Q4K => {
                        let blocks_per_row = cols as usize / 256;
                        repack_q4k_transposed(raw_bytes, rows as usize, blocks_per_row)
                    }
                    QuantType::Q5K => {
                        let blocks_per_row = cols as usize / 256;
                        repack_q5k_transposed(raw_bytes, rows as usize, blocks_per_row)
                    }
                    QuantType::Q6K => {
                        let blocks_per_row = cols as usize / 256;
                        repack_q6k_transposed(raw_bytes, rows as usize, blocks_per_row)
                    }
                    QuantType::Q8_0 => {
                        let blocks_per_row = cols as usize / 32;
                        repack_q8_0_transposed(raw_bytes, rows as usize, blocks_per_row)
                    }
                    other => {
                        return Err(format!(
                            "SoA repack is unavailable for native row-major quant {other:?}"
                        ));
                    }
                };

                let buf_size = (repacked.len() * 4) as u64;

                // Evict LRU entries until budget allows new allocation
                while self.total_cached + buf_size > self.budget && !self.entries.is_empty() {
                    let lru_id = self
                        .entries
                        .iter()
                        .min_by_key(|(_, e)| e.last_used)
                        .map(|(id, _)| *id)
                        .unwrap();
                    let evicted = self.entries.remove(&lru_id).unwrap();
                    self.total_cached -= evicted.buf.size;
                    ctx.destroy_buffer(evicted.buf);
                }

                // Ensure staging buffer is large enough
                let staging_ok = self
                    .staging_buf
                    .as_ref()
                    .map_or(false, |s| s.size >= buf_size);
                if !staging_ok {
                    // Drop old staging buffer if it exists
                    if let Some(old) = self.staging_buf.take() {
                        ctx.destroy_buffer(old);
                    }
                    let new_staging = ctx.create_buffer(
                        buf_size,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    )?;
                    self.staging_buf = Some(new_staging);
                }

                // Upload repacked data to staging buffer
                let repacked_bytes =
                    std::slice::from_raw_parts(repacked.as_ptr() as *const u8, repacked.len() * 4);
                let staging = self.staging_buf.as_ref().unwrap();
                ctx.upload_to_buffer(staging, repacked_bytes)?;

                // Allocate device-local buffer, fallback to host-visible
                let device_buf = match ctx.create_buffer(
                    buf_size,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ) {
                    Ok(buf) => {
                        ctx.copy_buffer_and_wait(command_pool, staging, &buf, buf_size)?;
                        buf
                    }
                    Err(_) => {
                        eprintln!(
                            "[weight_cache] device-local alloc failed for {:?}, falling back to host-visible",
                            id
                        );
                        let buf = ctx.create_buffer(
                            buf_size,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        )?;
                        ctx.upload_to_buffer(&buf, repacked_bytes)?;
                        buf
                    }
                };

                self.tick += 1;
                let entry = CachedWeight {
                    buf: device_buf,
                    last_used: self.tick,
                };
                self.total_cached += buf_size;
                self.entries.insert(id, entry);
            }

            GpuWeightMode::RowMajor => {
                self.upload_device_local_direct(ctx, command_pool, id, raw_bytes)?;
            }

            GpuWeightMode::Raw32 => {
                // raw f32 bytes 그대로 업로드 (norm weight 등). quant 태그 무시.
                self.upload_host_visible_direct(ctx, id, raw_bytes)?;
            }
        }

        Ok(&self.entries[&id].buf)
    }

    /// Creates a host-visible buffer and uploads raw bytes without staging.
    /// This path is reserved for mutable/scalar Raw32 data.
    ///
    /// # Safety
    /// `ctx`가 유효해야 하고, 호출자가 LRU eviction / budget 관리 책임.
    /// (현재 함수 내부에서 LRU eviction까지 처리한다.)
    unsafe fn upload_host_visible_direct(
        &mut self,
        ctx: &VulkanContext,
        id: WeightId,
        raw_bytes: &[u8],
    ) -> Result<(), String> {
        // Shaders address raw bytes through u32 storage-buffer words. Pad the
        // allocation so the final byte load never crosses the descriptor.
        let buf_size = raw_bytes.len().max(1).next_multiple_of(4) as u64;

        // Evict LRU entries until budget allows new allocation
        while self.total_cached + buf_size > self.budget && !self.entries.is_empty() {
            let lru_id = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(id, _)| *id)
                .unwrap();
            let evicted = self.entries.remove(&lru_id).unwrap();
            self.total_cached -= evicted.buf.size;
            ctx.destroy_buffer(evicted.buf);
        }

        // host-visible 버퍼 생성 후 raw_bytes 직접 업로드 (staging 불필요)
        let buf = ctx.create_buffer(
            buf_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )?;
        ctx.upload_to_buffer(&buf, raw_bytes)?;

        self.tick += 1;
        let entry = CachedWeight {
            buf,
            last_used: self.tick,
        };
        self.total_cached += buf_size;
        self.entries.insert(id, entry);
        Ok(())
    }

    /// Uploads an unchanged row-major byte stream into device-local memory.
    unsafe fn upload_device_local_direct(
        &mut self,
        ctx: &VulkanContext,
        command_pool: VkCommandPool,
        id: WeightId,
        raw_bytes: &[u8],
    ) -> Result<(), String> {
        let buf_size = raw_bytes.len().max(1).next_multiple_of(4) as u64;
        let upload_bytes = if raw_bytes.len() == buf_size as usize {
            std::borrow::Cow::Borrowed(raw_bytes)
        } else {
            let mut padded = vec![0u8; buf_size as usize];
            padded[..raw_bytes.len()].copy_from_slice(raw_bytes);
            std::borrow::Cow::Owned(padded)
        };

        while self.total_cached + buf_size > self.budget && !self.entries.is_empty() {
            let lru_id = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(entry_id, _)| *entry_id)
                .unwrap();
            let evicted = self.entries.remove(&lru_id).unwrap();
            self.total_cached -= evicted.buf.size;
            ctx.destroy_buffer(evicted.buf);
        }

        let staging_ok = self
            .staging_buf
            .as_ref()
            .is_some_and(|staging| staging.size >= buf_size);
        if !staging_ok {
            if let Some(old) = self.staging_buf.take() {
                ctx.destroy_buffer(old);
            }
            self.staging_buf = Some(ctx.create_buffer(
                buf_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )?);
        }
        let staging = self.staging_buf.as_ref().unwrap();
        ctx.upload_to_buffer(staging, &upload_bytes)?;

        let buf = match ctx.create_buffer(
            buf_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        ) {
            Ok(buf) => {
                ctx.copy_buffer_and_wait(command_pool, staging, &buf, buf_size)?;
                buf
            }
            Err(_) => {
                eprintln!(
                    "[weight_cache] device-local row-major alloc failed for {:?}, falling back to host-visible",
                    id
                );
                let buf = ctx.create_buffer(
                    buf_size,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                )?;
                ctx.upload_to_buffer(&buf, &upload_bytes)?;
                buf
            }
        };

        self.tick += 1;
        self.total_cached += buf_size;
        self.entries.insert(
            id,
            CachedWeight {
                buf,
                last_used: self.tick,
            },
        );
        Ok(())
    }

    /// Read-only borrow accessor — returns the cached `&GpuBuffer` for `id`
    /// without mutating the LRU tick counter.
    ///
    /// This exists so a caller can:
    /// 1. First pass: call `get_or_upload(...)` for every layer's weights,
    ///    which mutates the cache (LRU + uploads), and discard the returned
    ///    references because the borrow checker won't let them coexist with
    ///    further `&mut self` calls in the same loop.
    /// 2. Second pass: call `get(id)` for each weight to collect long-lived
    ///    `&GpuBuffer` references. Since `&self` doesn't conflict with itself,
    ///    we can hold N of these simultaneously and pack them into
    ///    `LayerWeightHandles`.
    ///
    /// Used by `rnb-runtime`'s fullpath prefill/decode wrappers (mv27-task10b-4c-2a).
    /// Returns `None` if the weight has never been uploaded (or has been evicted).
    pub fn get(&self, id: WeightId) -> Option<&GpuBuffer> {
        self.entries.get(&id).map(|e| &e.buf)
    }

    /// Destroy all GPU resources. Must be called before drop if ctx is available.
    pub unsafe fn destroy(&mut self, ctx: &VulkanContext) {
        for (_, entry) in self.entries.drain() {
            ctx.destroy_buffer(entry.buf);
        }
        self.total_cached = 0;
        if let Some(staging) = self.staging_buf.take() {
            ctx.destroy_buffer(staging);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_kind_variants_are_distinct() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WeightKind::AttnNorm);
        set.insert(WeightKind::FfnNorm);
        set.insert(WeightKind::KProjShard(0));
        set.insert(WeightKind::KProjShard(1));
        set.insert(WeightKind::VProjShard(0));
        set.insert(WeightKind::VProjShard(1));
        assert_eq!(set.len(), 6); // 모두 distinct
    }

    #[test]
    fn k_proj_shard_distinct_from_k_proj() {
        // KProjShard(0)은 KProj와 다른 WeightId — partial-offload / fullpath 공존 가능
        let layer = 3u16;
        let id_kproj = WeightId {
            layer,
            kind: WeightKind::KProj,
        };
        let id_shard0 = WeightId {
            layer,
            kind: WeightKind::KProjShard(0),
        };
        let id_shard1 = WeightId {
            layer,
            kind: WeightKind::KProjShard(1),
        };
        assert_ne!(id_kproj, id_shard0);
        assert_ne!(id_shard0, id_shard1);
    }

    #[test]
    fn effective_upload_mode_raw32_ignores_quant() {
        for q in [
            QuantType::Q4K,
            QuantType::Q5K,
            QuantType::Q6K,
            QuantType::Q8_0,
        ] {
            assert_eq!(
                effective_upload_mode(GpuWeightMode::Raw32, q, 9216, 2560),
                GpuWeightMode::Raw32,
                "Raw32 must stay Raw32 regardless of quant tag (got quant={:?})",
                q
            );
        }
    }

    #[test]
    fn effective_upload_mode_rowmajor_q4k_is_shape_adaptive() {
        assert_eq!(
            effective_upload_mode(GpuWeightMode::RowMajor, QuantType::Q4K, 9216, 2560),
            GpuWeightMode::Soa
        );
        assert_eq!(
            effective_upload_mode(GpuWeightMode::RowMajor, QuantType::Q4K, 2560, 9216),
            GpuWeightMode::RowMajor
        );
        assert_eq!(
            effective_upload_mode(GpuWeightMode::RowMajor, QuantType::Q4K, 2560, 2560),
            GpuWeightMode::RowMajor
        );
        for q in [QuantType::Q5K, QuantType::Q6K, QuantType::Q8_0] {
            assert_eq!(
                effective_upload_mode(GpuWeightMode::RowMajor, q, 2560, 9216),
                GpuWeightMode::Soa,
                "RowMajor with non-Q4K quant must fall back to Soa (got quant={:?})",
                q
            );
        }
    }

    #[test]
    fn effective_upload_mode_soa_passthrough() {
        // SoA 커널이 있는 quant는 SoA 요청을 그대로 유지한다.
        for q in [
            QuantType::Q4K,
            QuantType::Q5K,
            QuantType::Q6K,
            QuantType::Q8_0,
        ] {
            assert_eq!(
                effective_upload_mode(GpuWeightMode::Soa, q, 9216, 2560),
                GpuWeightMode::Soa
            );
        }
    }

    #[test]
    fn importance_quants_keep_original_row_major_bytes() {
        for (quant, block_bytes) in [
            (QuantType::IQ2_XXS, 66),
            (QuantType::IQ2_XS, 74),
            (QuantType::IQ3_XXS, 98),
            (QuantType::IQ1_S, 50),
            (QuantType::IQ3_S, 110),
            (QuantType::IQ2_S, 82),
            (QuantType::IQ4_XS, 136),
            (QuantType::IQ1_M, 56),
        ] {
            assert_eq!(quant.block_elements(), 256);
            assert_eq!(quant.block_bytes(), block_bytes);
            assert_eq!(
                effective_upload_mode(GpuWeightMode::RowMajor, quant, 2560, 9216),
                GpuWeightMode::RowMajor
            );
            assert_eq!(
                effective_upload_mode(GpuWeightMode::Soa, quant, 9216, 2560),
                GpuWeightMode::RowMajor
            );
        }
    }

    #[test]
    fn weight_id_hash_consistency() {
        use std::collections::HashMap;
        let mut map: HashMap<WeightId, &str> = HashMap::new();
        map.insert(
            WeightId {
                layer: 0,
                kind: WeightKind::AttnNorm,
            },
            "attn_norm_0",
        );
        map.insert(
            WeightId {
                layer: 0,
                kind: WeightKind::FfnNorm,
            },
            "ffn_norm_0",
        );
        map.insert(
            WeightId {
                layer: 0,
                kind: WeightKind::KProjShard(0),
            },
            "k_shard_0_kvh0",
        );
        map.insert(
            WeightId {
                layer: 0,
                kind: WeightKind::KProjShard(1),
            },
            "k_shard_0_kvh1",
        );
        assert_eq!(map.len(), 4);
        assert_eq!(
            *map.get(&WeightId {
                layer: 0,
                kind: WeightKind::KProjShard(0)
            })
            .unwrap(),
            "k_shard_0_kvh0"
        );
    }
}
