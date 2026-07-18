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
    QNorm,
    KNorm,
    KProj,
    VProj,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantType {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

/// GPU weight 메모리 레이아웃 모드.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWeightMode {
    /// SoA repack + device-local. 성능 최적, 메모리 2x.
    Soa,
    /// Row-major, host-visible. 메모리 절약, 1.5-2.2x 느림.
    RowMajor,
    /// Pre-converted f32 weights (RMS-norm 등) — no repack, no dequant; the
    /// quant tag is ignored on this path. Host-visible buffer, direct upload.
    Raw32,
}

/// Resolve `(mode, quant)` to the effective upload path mode.
///
/// - `Raw32`: 항상 Raw32 (quant 무시).
/// - `RowMajor`: Q4K일 때만 RowMajor 유지, 그 외는 Soa fallback.
/// - `Soa`: 그대로.
///
/// `get_or_upload` 내부 분기 결정에 쓰이고, unit test에서도 같은 함수를
/// 호출해 매핑 일관성을 검증한다.
pub(crate) fn effective_upload_mode(mode: GpuWeightMode, quant: QuantType) -> GpuWeightMode {
    match (mode, quant) {
        (GpuWeightMode::Raw32, _) => GpuWeightMode::Raw32,
        (GpuWeightMode::RowMajor, QuantType::Q4K) => GpuWeightMode::RowMajor,
        (GpuWeightMode::RowMajor, _) => GpuWeightMode::Soa,
        (GpuWeightMode::Soa, _) => GpuWeightMode::Soa,
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
    /// `mode` controls the upload strategy:
    /// - `GpuWeightMode::Soa`: repack to transposed SoA layout → staging → device-local (기존 경로)
    /// - `GpuWeightMode::RowMajor`: repack 없이 raw_bytes를 host-visible 버퍼에 직접 복사
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

        // RowMajor는 Q4K만 지원. Q6K/Q8_0는 SoA로 fallback.
        // Raw32는 quant 무관 — 항상 raw byte-copy 경로.
        let effective_mode = effective_upload_mode(mode, quant);

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
                // repack 없이 raw_bytes를 host-visible 버퍼에 직접 복사 (Q4K row-major 경로)
                self.upload_host_visible_direct(ctx, id, raw_bytes)?;
            }

            GpuWeightMode::Raw32 => {
                // raw f32 bytes 그대로 업로드 (norm weight 등). quant 태그 무시.
                self.upload_host_visible_direct(ctx, id, raw_bytes)?;
            }
        }

        Ok(&self.entries[&id].buf)
    }

    /// host-visible 버퍼 생성 후 raw_bytes 직접 업로드 (staging 불필요).
    /// `RowMajor` / `Raw32` 양쪽이 공용으로 쓰는 직접 업로드 경로.
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
        let buf_size = raw_bytes.len() as u64;

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
        // Raw32는 모든 quant에 대해 Raw32 유지 (norm weight 경로).
        for q in [
            QuantType::Q4K,
            QuantType::Q5K,
            QuantType::Q6K,
            QuantType::Q8_0,
        ] {
            assert_eq!(
                effective_upload_mode(GpuWeightMode::Raw32, q),
                GpuWeightMode::Raw32,
                "Raw32 must stay Raw32 regardless of quant tag (got quant={:?})",
                q
            );
        }
    }

    #[test]
    fn effective_upload_mode_rowmajor_q4k_only() {
        // RowMajor는 Q4K일 때만 유지. Q5K/Q6K/Q8_0는 Soa로 fallback.
        assert_eq!(
            effective_upload_mode(GpuWeightMode::RowMajor, QuantType::Q4K),
            GpuWeightMode::RowMajor
        );
        for q in [QuantType::Q5K, QuantType::Q6K, QuantType::Q8_0] {
            assert_eq!(
                effective_upload_mode(GpuWeightMode::RowMajor, q),
                GpuWeightMode::Soa,
                "RowMajor with non-Q4K quant must fall back to Soa (got quant={:?})",
                q
            );
        }
    }

    #[test]
    fn effective_upload_mode_soa_passthrough() {
        // Soa는 quant 상관없이 그대로.
        for q in [
            QuantType::Q4K,
            QuantType::Q5K,
            QuantType::Q6K,
            QuantType::Q8_0,
        ] {
            assert_eq!(
                effective_upload_mode(GpuWeightMode::Soa, q),
                GpuWeightMode::Soa
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
