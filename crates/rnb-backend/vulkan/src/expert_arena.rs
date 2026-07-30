use crate::context::{GpuBuffer, VulkanContext};
use crate::ffi::types::{
    VkBufferCopy, VkBufferMemoryBarrier, VkCommandBuffer, VK_ACCESS_HOST_WRITE_BIT,
    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
    VK_PIPELINE_BIND_POINT_COMPUTE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_QUEUE_FAMILY_IGNORED,
    VK_SHADER_STAGE_COMPUTE_BIT, VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
};
use crate::gemv::repack_q6k_transposed;
use crate::pipeline::ComputePipeline;
use crate::spirv::emit_q8_arena_repack;
use crate::weight_cache::QuantType;
use rnb_memory::ByteLruPolicy;
use std::collections::HashMap;
use std::ptr;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct ExpertArenaKey {
    pub layer: u16,
    pub expert: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertArenaFormat {
    Q4KGateUp,
    Q8Zero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertArenaLayout {
    pub hidden: u32,
    pub n_ff: u32,
    pub format: ExpertArenaFormat,
    pub gate_offset_words: u32,
    pub up_offset_words: u32,
    pub down_offset_words: u32,
    pub slot_stride_words: u32,
}

impl ExpertArenaLayout {
    pub(crate) fn qwen(hidden: u32, n_ff: u32) -> Result<Self, String> {
        let gate_bytes = quant_matrix_bytes(n_ff, hidden, QuantType::Q4K)?;
        let up_bytes = gate_bytes;
        let down_offset = gate_bytes
            .checked_add(up_bytes)
            .ok_or("expert arena down offset overflow")?;
        let q6_resident_bytes = soa_matrix_bytes(hidden, n_ff, QuantType::Q6K)?;
        let slot_bytes = down_offset
            .checked_add(q6_resident_bytes)
            .ok_or("expert arena slot size overflow")?
            .next_multiple_of(4);
        Ok(Self {
            hidden,
            n_ff,
            format: ExpertArenaFormat::Q4KGateUp,
            gate_offset_words: 0,
            up_offset_words: bytes_to_words(gate_bytes, "gate offset")?,
            down_offset_words: bytes_to_words(down_offset, "down offset")?,
            slot_stride_words: bytes_to_words(slot_bytes, "slot stride")?,
        })
    }

    pub(crate) fn qwen_q8(hidden: u32, n_ff: u32) -> Result<Self, String> {
        let gate_bytes = soa_matrix_bytes(n_ff, hidden, QuantType::Q8_0)?;
        let up_bytes = gate_bytes;
        let down_offset = gate_bytes
            .checked_add(up_bytes)
            .ok_or("Q8 expert arena down offset overflow")?;
        let down_bytes = soa_matrix_bytes(hidden, n_ff, QuantType::Q8_0)?;
        let slot_bytes = down_offset
            .checked_add(down_bytes)
            .ok_or("Q8 expert arena slot size overflow")?
            .next_multiple_of(4);
        Ok(Self {
            hidden,
            n_ff,
            format: ExpertArenaFormat::Q8Zero,
            gate_offset_words: 0,
            up_offset_words: bytes_to_words(gate_bytes, "Q8 gate offset")?,
            down_offset_words: bytes_to_words(down_offset, "Q8 down offset")?,
            slot_stride_words: bytes_to_words(slot_bytes, "Q8 slot stride")?,
        })
    }

    pub(crate) const fn gate_quant(self) -> QuantType {
        match self.format {
            ExpertArenaFormat::Q4KGateUp => QuantType::Q4K,
            ExpertArenaFormat::Q8Zero => QuantType::Q8_0,
        }
    }

    pub(crate) const fn up_quant(self) -> QuantType {
        self.gate_quant()
    }

    pub(crate) const fn supports_down_quant(self, quant: QuantType) -> bool {
        match self.format {
            ExpertArenaFormat::Q4KGateUp => {
                matches!(quant, QuantType::Q5K | QuantType::Q6K)
            }
            ExpertArenaFormat::Q8Zero => matches!(quant, QuantType::Q8_0),
        }
    }

    pub(crate) const fn slot_stride_bytes(self) -> u64 {
        self.slot_stride_words as u64 * 4
    }

    pub(crate) fn required_bytes(self, slots: usize) -> Result<u64, String> {
        let slots = u64::try_from(slots).map_err(|_| "expert arena slot count exceeds u64")?;
        self.slot_stride_bytes()
            .checked_mul(slots)
            .ok_or("expert arena required byte count overflow".into())
    }

    pub(crate) fn max_slot_count(self, available_bytes: u64, descriptor_bytes: u64) -> u64 {
        let addressable_bytes = (u32::MAX as u64) * 4;
        available_bytes.min(addressable_bytes).min(descriptor_bytes) / self.slot_stride_bytes()
    }

    #[cfg(test)]
    fn gate_bytes(self) -> u64 {
        self.up_offset_words as u64 * 4
    }

    #[cfg(test)]
    fn up_bytes(self) -> u64 {
        (self.down_offset_words - self.up_offset_words) as u64 * 4
    }

    fn gate_raw_bytes(self) -> Result<u64, String> {
        quant_matrix_bytes(self.n_ff, self.hidden, self.gate_quant())
    }

    fn up_raw_bytes(self) -> Result<u64, String> {
        quant_matrix_bytes(self.n_ff, self.hidden, self.up_quant())
    }

    fn down_raw_bytes(self, quant: QuantType) -> Result<u64, String> {
        quant_matrix_bytes(self.hidden, self.n_ff, quant)
    }

    fn down_resident_bytes(self, quant: QuantType) -> Result<u64, String> {
        soa_matrix_bytes(self.hidden, self.n_ff, quant)
    }
}

fn bytes_to_words(bytes: u64, label: &str) -> Result<u32, String> {
    if !bytes.is_multiple_of(4) {
        return Err(format!(
            "expert arena {label} byte count {bytes} is not u32 aligned"
        ));
    }
    u32::try_from(bytes / 4).map_err(|_| format!("expert arena {label} exceeds u32 words"))
}

fn quant_matrix_bytes(rows: u32, cols: u32, quant: QuantType) -> Result<u64, String> {
    let block = quant.block_elements() as u32;
    if rows == 0 || cols == 0 || !cols.is_multiple_of(block) {
        return Err(format!(
            "expert arena shape [{rows}, {cols}] is incompatible with {quant:?} block {block}"
        ));
    }
    u64::from(rows)
        .checked_mul(u64::from(cols / block))
        .and_then(|blocks| blocks.checked_mul(quant.block_bytes() as u64))
        .ok_or("expert arena matrix byte count overflow".into())
}

fn soa_matrix_bytes(rows: u32, cols: u32, quant: QuantType) -> Result<u64, String> {
    let (block_elements, words_per_block) = match quant {
        QuantType::Q5K => (256_u32, 44_u64),
        QuantType::Q6K => (256_u32, 53_u64),
        QuantType::Q8_0 => (32_u32, 9_u64),
        other => {
            return Err(format!(
                "expert arena resident projection does not support {other:?}"
            ))
        }
    };
    if rows == 0 || cols == 0 || !cols.is_multiple_of(block_elements) {
        return Err(format!(
            "expert arena SoA shape [{rows}, {cols}] requires cols multiple of {block_elements}"
        ));
    }
    u64::from(rows)
        .checked_mul(u64::from(cols / block_elements))
        .and_then(|blocks| blocks.checked_mul(words_per_block))
        .and_then(|words| words.checked_mul(4))
        .ok_or("expert arena SoA byte count overflow".into())
}

pub(crate) struct ExpertArenaBundle<'a> {
    pub key: ExpertArenaKey,
    pub gate: &'a [u8],
    pub gate_quant: QuantType,
    pub up: &'a [u8],
    pub up_quant: QuantType,
    pub down: &'a [u8],
    pub down_quant: QuantType,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExpertArenaBatchStats {
    pub hits: u64,
    pub misses: u64,
    pub upload_bytes: u64,
    pub repack_bytes: u64,
    pub slot_plan_us: u128,
    pub repack_us: u128,
    pub staging_us: u128,
}

#[derive(Clone)]
struct ExpertSlotMap {
    policy: ByteLruPolicy<ExpertArenaKey>,
    slots: HashMap<ExpertArenaKey, u32>,
    free_slots: Vec<u32>,
    slot_bytes: u64,
}

impl ExpertSlotMap {
    fn new(slot_count: u32, slot_bytes: u64) -> Self {
        Self {
            policy: ByteLruPolicy::new(u64::from(slot_count) * slot_bytes),
            slots: HashMap::with_capacity(slot_count as usize),
            free_slots: (0..slot_count).rev().collect(),
            slot_bytes,
        }
    }

    fn touch_batch(&mut self, keys: &[ExpertArenaKey]) -> Result<(), String> {
        for key in keys {
            if self.slots.contains_key(key) {
                let evicted = self.policy.touch(*key, self.slot_bytes);
                if !evicted.is_empty() {
                    return Err(
                        "expert arena evicted a protected batch key while touching hits".into(),
                    );
                }
            }
        }
        Ok(())
    }

    fn get_or_assign(&mut self, key: ExpertArenaKey) -> Result<(u32, bool), String> {
        if let Some(&slot) = self.slots.get(&key) {
            let evicted = self.policy.touch(key, self.slot_bytes);
            if !evicted.is_empty() {
                return Err("expert arena hit unexpectedly caused eviction".into());
            }
            return Ok((slot, true));
        }

        let evicted = self.policy.touch(key, self.slot_bytes);
        let slot = if let Some(old_key) = evicted.into_iter().next() {
            self.slots
                .remove(&old_key)
                .ok_or("expert arena LRU returned an unknown key")?
        } else {
            self.free_slots
                .pop()
                .ok_or("expert arena has no free or evictable slot")?
        };
        self.slots.insert(key, slot);
        Ok((slot, false))
    }
}
#[derive(Clone, Copy)]
struct PendingQ8Repack {
    rows: u32,
    cols: u32,
    source_offset_bytes: u32,
    destination_offset_words: u32,
}

struct PendingArenaBatch {
    slots: ExpertSlotMap,
    copies: Vec<VkBufferCopy>,
    q8_repacks: Vec<PendingQ8Repack>,
}

pub(crate) struct ExpertArena {
    layout: ExpertArenaLayout,
    slot_count: u32,
    allocation_bytes: u64,
    buffer: GpuBuffer,
    staging: GpuBuffer,
    staging_ptr: *mut u8,
    q8_repack_pipeline: Option<ComputePipeline>,
    slots: ExpertSlotMap,
    pending: Option<PendingArenaBatch>,
}

#[derive(Debug)]
pub(crate) enum ExpertArenaCreateError {
    InsufficientCapacity(String),
    Fatal(String),
}

impl ExpertArena {
    pub(crate) unsafe fn new(
        ctx: &VulkanContext,
        available_bytes: u64,
        min_slots: usize,
        layout: ExpertArenaLayout,
    ) -> Result<Self, ExpertArenaCreateError> {
        let min_slots = u64::try_from(min_slots)
            .map_err(|_| {
                ExpertArenaCreateError::Fatal("expert arena slot count exceeds u64".into())
            })?
            .max(1);
        let slot_bytes = layout.slot_stride_bytes();
        let descriptor_bytes = ctx.max_storage_buffer_range;
        let mut slot_count = layout.max_slot_count(available_bytes, descriptor_bytes);
        if slot_count < min_slots {
            return Err(ExpertArenaCreateError::InsufficientCapacity(format!(
                "budget cannot hold route set: available={} slot_bytes={} slots={} required={min_slots}",
                available_bytes, slot_bytes, slot_count
            )));
        }

        let usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        let (buffer, allocated_slots) = loop {
            let allocation_bytes = slot_count * slot_bytes;
            match ctx.try_create_buffer(
                allocation_bytes,
                usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            ) {
                Ok(buffer) => break (buffer, slot_count),
                Err(error) if error.is_out_of_memory() => {
                    if slot_count == min_slots {
                        return Err(ExpertArenaCreateError::InsufficientCapacity(format!(
                            "device allocation failed at required {min_slots} slots: {error}"
                        )));
                    }
                    slot_count = (slot_count / 2).max(min_slots);
                }
                Err(error) => {
                    return Err(ExpertArenaCreateError::Fatal(error.to_string()));
                }
            }
        };
        let slot_count = u32::try_from(allocated_slots).map_err(|_| {
            ExpertArenaCreateError::Fatal("expert arena slot count exceeds u32".into())
        })?;
        let allocation_bytes = u64::from(slot_count) * slot_bytes;
        let staging_bytes = min_slots.checked_mul(slot_bytes).ok_or_else(|| {
            ExpertArenaCreateError::Fatal("expert arena staging size overflow".into())
        })?;
        let staging = match ctx.try_create_buffer(
            staging_bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        ) {
            Ok(staging) => staging,
            Err(error) if error.is_out_of_memory() => {
                ctx.destroy_buffer(buffer);
                return Err(ExpertArenaCreateError::InsufficientCapacity(format!(
                    "staging allocation failed for required {min_slots} slots: {error}"
                )));
            }
            Err(error) => {
                ctx.destroy_buffer(buffer);
                return Err(ExpertArenaCreateError::Fatal(error.to_string()));
            }
        };
        let staging_ptr = match ctx.map_buffer_persistent(&staging) {
            Ok(ptr) => ptr,
            Err(error) => {
                ctx.destroy_buffer(staging);
                ctx.destroy_buffer(buffer);
                return Err(ExpertArenaCreateError::Fatal(error));
            }
        };
        let q8_repack_pipeline = if layout.format == ExpertArenaFormat::Q8Zero {
            match ComputePipeline::new_2binding(ctx, &emit_q8_arena_repack(64), 1, 16) {
                Ok(pipeline) => Some(pipeline),
                Err(error) => {
                    ctx.unmap_buffer(&staging);
                    ctx.destroy_buffer(staging);
                    ctx.destroy_buffer(buffer);
                    return Err(ExpertArenaCreateError::Fatal(format!(
                        "Q8 expert arena repack pipeline creation failed: {error}"
                    )));
                }
            }
        } else {
            None
        };
        eprintln!(
            "[vulkan:moe-arena] slots={} slot_mib={:.3} arena_gib={:.3} budget_gib={:.3} descriptor_limit_gib={:.3}",
            slot_count,
            slot_bytes as f64 / (1024.0 * 1024.0),
            allocation_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            available_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            descriptor_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        Ok(Self {
            layout,
            slot_count,
            allocation_bytes,
            buffer,
            staging,
            staging_ptr,
            q8_repack_pipeline,
            slots: ExpertSlotMap::new(slot_count, slot_bytes),
            pending: None,
        })
    }

    pub(crate) const fn layout(&self) -> ExpertArenaLayout {
        self.layout
    }

    pub(crate) const fn slot_count(&self) -> u32 {
        self.slot_count
    }

    pub(crate) fn batch_slot_count(&self) -> u32 {
        let staging_slots = self.staging.size / self.layout.slot_stride_bytes();
        self.slot_count
            .min(staging_slots.min(u64::from(u32::MAX)) as u32)
    }

    pub(crate) const fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }

    pub(crate) const fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub(crate) unsafe fn prepare_batch(
        &mut self,
        bundles: &[ExpertArenaBundle<'_>],
    ) -> Result<(Vec<u32>, ExpertArenaBatchStats), String> {
        self.pending = None;
        let profile_enabled = std::env::var_os("RNB_VULKAN_FULLPATH_PROFILE").is_some();
        let slot_timer = profile_enabled.then(Instant::now);
        if bundles.len() > self.slot_count as usize {
            return Err(format!(
                "expert arena batch has {} unique experts but only {} descriptor-safe slots",
                bundles.len(),
                self.slot_count
            ));
        }
        let keys = bundles.iter().map(|bundle| bundle.key).collect::<Vec<_>>();
        let mut sorted_keys = keys.clone();
        sorted_keys.sort_unstable();
        sorted_keys.dedup();
        if sorted_keys.len() != keys.len() {
            return Err("expert arena batch contains duplicate expert keys".into());
        }

        let mut proposed_slots = self.slots.clone();
        proposed_slots.touch_batch(&keys)?;
        let mut assignments = Vec::with_capacity(bundles.len());
        let mut stats = ExpertArenaBatchStats::default();
        for bundle in bundles {
            let (slot, hit) = proposed_slots.get_or_assign(bundle.key)?;
            assignments.push((slot, hit));
            if hit {
                stats.hits += 1;
            } else {
                stats.misses += 1;
            }
        }
        stats.slot_plan_us = slot_timer.map_or(0, |timer| timer.elapsed().as_micros());

        let repack_timer = profile_enabled.then(Instant::now);
        let misses = bundles
            .iter()
            .enumerate()
            .filter(|(bundle_idx, _)| !assignments[*bundle_idx].1)
            .collect::<Vec<_>>();
        for &(_, bundle) in &misses {
            self.validate_bundle(bundle)?;
        }
        let worker_count = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(misses.len().max(1));
        let bundles_per_worker = misses.len().div_ceil(worker_count).max(1);
        let layout = self.layout;
        let (prepared, repack_bytes) = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for chunk in misses.chunks(bundles_per_worker) {
                handles.push(scope.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    let mut local_repack_bytes = 0_u64;
                    for &(bundle_idx, bundle) in chunk {
                        let down_words = match layout.format {
                            ExpertArenaFormat::Q4KGateUp => match bundle.down_quant {
                                QuantType::Q5K => None,
                                QuantType::Q6K => {
                                    local_repack_bytes += bundle.down.len() as u64;
                                    Some(Self::prepare_down(layout, bundle)?)
                                }
                                other => {
                                    return Err(format!(
                                        "expert arena down quant {other:?} is unsupported"
                                    ));
                                }
                            },
                            ExpertArenaFormat::Q8Zero => None,
                        };
                        local.push((bundle_idx, down_words));
                    }
                    Ok::<_, String>((local, local_repack_bytes))
                }));
            }
            let mut prepared = Vec::with_capacity(misses.len());
            let mut repack_bytes = 0_u64;
            for handle in handles {
                let (mut local, local_repack_bytes) = handle
                    .join()
                    .map_err(|_| "expert arena repack worker panicked".to_string())??;
                prepared.append(&mut local);
                repack_bytes += local_repack_bytes;
            }
            Ok::<_, String>((prepared, repack_bytes))
        })?;
        stats.repack_bytes = repack_bytes;
        stats.repack_us = repack_timer.map_or(0, |timer| timer.elapsed().as_micros());
        let staging_timer = profile_enabled.then(Instant::now);
        let staging_need = (prepared.len() as u64)
            .checked_mul(self.layout.slot_stride_bytes())
            .ok_or("expert arena staging size overflow")?;
        if staging_need > self.staging.size {
            return Err(format!(
                "expert arena batch requires {} staging bytes but call was configured for {}",
                staging_need, self.staging.size
            ));
        }

        if !prepared.is_empty() {
            let slot_stride = self.layout.slot_stride_bytes() as usize;
            let staging_bytes =
                std::slice::from_raw_parts_mut(self.staging_ptr, staging_need as usize);
            let worker_count = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .min(prepared.len());
            let bundles_per_worker = prepared.len().div_ceil(worker_count);
            let bytes_per_worker = bundles_per_worker * slot_stride;
            let up_offset = self.layout.up_offset_words as usize * 4;
            let down_offset = self.layout.down_offset_words as usize * 4;
            std::thread::scope(|scope| {
                for (prepared_chunk, staging_chunk) in prepared
                    .chunks(bundles_per_worker)
                    .zip(staging_bytes.chunks_mut(bytes_per_worker))
                {
                    scope.spawn(move || {
                        for (local_idx, (bundle_idx, down_words)) in
                            prepared_chunk.iter().enumerate()
                        {
                            let bundle = &bundles[*bundle_idx];
                            let slot_start = local_idx * slot_stride;
                            let slot_bytes =
                                &mut staging_chunk[slot_start..slot_start + slot_stride];
                            let gate_bytes = bundle.gate;
                            slot_bytes[..gate_bytes.len()].copy_from_slice(gate_bytes);
                            let up_bytes = bundle.up;
                            slot_bytes[up_offset..up_offset + up_bytes.len()]
                                .copy_from_slice(up_bytes);
                            let down_bytes =
                                Self::projection_bytes(down_words.as_ref(), bundle.down);
                            slot_bytes[down_offset..down_offset + down_bytes.len()]
                                .copy_from_slice(down_bytes);
                        }
                    });
                }
            });
        }

        let mut copies = Vec::with_capacity(prepared.len());
        let mut q8_repacks = Vec::with_capacity(prepared.len() * 3);
        for (staging_idx, (bundle_idx, down_words)) in prepared.iter().enumerate() {
            let bundle = &bundles[*bundle_idx];
            let slot = assignments[*bundle_idx].0;
            let staging_offset = staging_idx as u64 * self.layout.slot_stride_bytes();
            match self.layout.format {
                ExpertArenaFormat::Q4KGateUp => {
                    let down_bytes = down_words.as_ref().map_or(bundle.down.len(), |words| {
                        words.len() * std::mem::size_of::<u32>()
                    });
                    let copy_bytes = self.layout.down_offset_words as u64 * 4 + down_bytes as u64;
                    copies.push(VkBufferCopy {
                        src_offset: staging_offset,
                        dst_offset: u64::from(slot) * self.layout.slot_stride_bytes(),
                        size: copy_bytes,
                    });
                    stats.upload_bytes += copy_bytes;
                }
                ExpertArenaFormat::Q8Zero => {
                    let staging_base = u32::try_from(staging_offset)
                        .map_err(|_| "Q8 expert staging byte offset exceeds u32")?;
                    let slot_base = slot
                        .checked_mul(self.layout.slot_stride_words)
                        .ok_or("Q8 expert arena slot word offset overflow")?;
                    let up_source =
                        u32::try_from(staging_offset + u64::from(self.layout.up_offset_words) * 4)
                            .map_err(|_| "Q8 expert up staging byte offset exceeds u32")?;
                    let down_source = u32::try_from(
                        staging_offset + u64::from(self.layout.down_offset_words) * 4,
                    )
                    .map_err(|_| "Q8 expert down staging byte offset exceeds u32")?;
                    let up_destination = slot_base
                        .checked_add(self.layout.up_offset_words)
                        .ok_or("Q8 expert up arena word offset overflow")?;
                    let down_destination = slot_base
                        .checked_add(self.layout.down_offset_words)
                        .ok_or("Q8 expert down arena word offset overflow")?;
                    q8_repacks.extend([
                        PendingQ8Repack {
                            rows: self.layout.n_ff,
                            cols: self.layout.hidden,
                            source_offset_bytes: staging_base,
                            destination_offset_words: slot_base,
                        },
                        PendingQ8Repack {
                            rows: self.layout.n_ff,
                            cols: self.layout.hidden,
                            source_offset_bytes: up_source,
                            destination_offset_words: up_destination,
                        },
                        PendingQ8Repack {
                            rows: self.layout.hidden,
                            cols: self.layout.n_ff,
                            source_offset_bytes: down_source,
                            destination_offset_words: down_destination,
                        },
                    ]);
                    stats.upload_bytes +=
                        (bundle.gate.len() + bundle.up.len() + bundle.down.len()) as u64;
                }
            }
        }
        stats.staging_us = staging_timer.map_or(0, |timer| timer.elapsed().as_micros());
        let result_slots = assignments.into_iter().map(|(slot, _)| slot).collect();
        self.pending = Some(PendingArenaBatch {
            slots: proposed_slots,
            copies,
            q8_repacks,
        });
        Ok((result_slots, stats))
    }

    fn validate_bundle(&self, bundle: &ExpertArenaBundle<'_>) -> Result<(), String> {
        if bundle.gate_quant != self.layout.gate_quant()
            || bundle.up_quant != self.layout.up_quant()
            || !self.layout.supports_down_quant(bundle.down_quant)
        {
            return Err(format!(
                "expert arena bundle {:?} quant mismatch gate={:?}/{:?} up={:?}/{:?} down={:?}",
                bundle.key,
                bundle.gate_quant,
                self.layout.gate_quant(),
                bundle.up_quant,
                self.layout.up_quant(),
                bundle.down_quant,
            ));
        }
        let gate_bytes = self.layout.gate_raw_bytes()?;
        let up_bytes = self.layout.up_raw_bytes()?;
        let down_raw_bytes = self.layout.down_raw_bytes(bundle.down_quant)?;
        if bundle.gate.len() as u64 != gate_bytes
            || bundle.up.len() as u64 != up_bytes
            || bundle.down.len() as u64 != down_raw_bytes
        {
            return Err(format!(
                "expert arena bundle {:?} size mismatch gate={}/{} up={}/{} down={}/{}",
                bundle.key,
                bundle.gate.len(),
                gate_bytes,
                bundle.up.len(),
                up_bytes,
                bundle.down.len(),
                down_raw_bytes
            ));
        }
        Ok(())
    }

    fn prepare_down(
        layout: ExpertArenaLayout,
        bundle: &ExpertArenaBundle<'_>,
    ) -> Result<Vec<u32>, String> {
        let down_words = match bundle.down_quant {
            QuantType::Q6K => repack_q6k_transposed(
                bundle.down,
                layout.hidden as usize,
                layout.n_ff as usize / 256,
            ),
            other => {
                return Err(format!(
                    "expert arena repack for down quant {other:?} is unsupported"
                ))
            }
        };
        let down_resident_bytes = layout.down_resident_bytes(bundle.down_quant)? as usize;
        if down_words.len() * std::mem::size_of::<u32>() != down_resident_bytes {
            return Err(format!(
                "expert arena repack size {} != expected {} for {:?}",
                down_words.len() * std::mem::size_of::<u32>(),
                down_resident_bytes,
                bundle.down_quant
            ));
        }
        Ok(down_words)
    }

    fn projection_bytes<'a>(words: Option<&'a Vec<u32>>, raw: &'a [u8]) -> &'a [u8] {
        words.map_or(raw, |words| unsafe {
            std::slice::from_raw_parts(
                words.as_ptr() as *const u8,
                words.len() * std::mem::size_of::<u32>(),
            )
        })
    }

    pub(crate) unsafe fn record_pending_uploads(
        &self,
        ctx: &VulkanContext,
        cmdbuf: VkCommandBuffer,
    ) -> Result<(), String> {
        let pending = self
            .pending
            .as_ref()
            .ok_or("expert arena has no prepared batch")?;
        if pending.copies.is_empty() && pending.q8_repacks.is_empty() {
            return Ok(());
        }
        if !pending.copies.is_empty() {
            let copy_count = u32::try_from(pending.copies.len())
                .map_err(|_| "expert arena copy count exceeds u32")?;
            (ctx.vk.cmd_copy_buffer)(
                cmdbuf,
                self.staging.buffer,
                self.buffer.buffer,
                copy_count,
                pending.copies.as_ptr(),
            );
            let barrier = VkBufferMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                buffer: self.buffer.buffer,
                offset: 0,
                size: self.allocation_bytes,
            };
            (ctx.vk.cmd_pipeline_barrier)(
                cmdbuf,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                ptr::null(),
                1,
                &barrier,
                0,
                ptr::null(),
            );
        }
        if !pending.q8_repacks.is_empty() {
            let pipeline = self
                .q8_repack_pipeline
                .as_ref()
                .ok_or("Q8 expert arena has no repack pipeline")?;
            let host_barrier = VkBufferMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_HOST_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                buffer: self.staging.buffer,
                offset: 0,
                size: self.staging.size,
            };
            (ctx.vk.cmd_pipeline_barrier)(
                cmdbuf,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                ptr::null(),
                1,
                &host_barrier,
                0,
                ptr::null(),
            );
            pipeline.bind_buffers_2(
                ctx,
                0,
                &self.staging,
                self.staging.size,
                &self.buffer,
                self.allocation_bytes,
            );
            (ctx.vk.cmd_bind_pipeline)(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            (ctx.vk.cmd_bind_descriptor_sets)(
                cmdbuf,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline.pipeline_layout,
                0,
                1,
                &pipeline.descriptor_sets[0],
                0,
                ptr::null(),
            );
            for repack in &pending.q8_repacks {
                let push = [
                    repack.rows,
                    repack.cols,
                    repack.source_offset_bytes,
                    repack.destination_offset_words,
                ];
                (ctx.vk.cmd_push_constants)(
                    cmdbuf,
                    pipeline.pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    (push.len() * std::mem::size_of::<u32>()) as u32,
                    push.as_ptr() as *const std::ffi::c_void,
                );
                (ctx.vk.cmd_dispatch)(cmdbuf, repack.rows.div_ceil(64), repack.cols / 32, 1);
            }
            let shader_barrier = VkBufferMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: VK_ACCESS_SHADER_WRITE_BIT,
                dst_access_mask: VK_ACCESS_SHADER_READ_BIT,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                buffer: self.buffer.buffer,
                offset: 0,
                size: self.allocation_bytes,
            };
            (ctx.vk.cmd_pipeline_barrier)(
                cmdbuf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                ptr::null(),
                1,
                &shader_barrier,
                0,
                ptr::null(),
            );
        }
        Ok(())
    }

    pub(crate) const fn has_pending_batch(&self) -> bool {
        self.pending.is_some()
    }

    pub(crate) fn commit_pending_batch(&mut self) -> Result<(), String> {
        let pending = self
            .pending
            .take()
            .ok_or("expert arena has no submitted batch to commit")?;
        self.slots = pending.slots;
        Ok(())
    }

    pub(crate) unsafe fn destroy(self, ctx: &VulkanContext) {
        if let Some(pipeline) = self.q8_repack_pipeline {
            pipeline.destroy(ctx);
        }
        ctx.unmap_buffer(&self.staging);
        ctx.destroy_buffer(self.staging);
        ctx.destroy_buffer(self.buffer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_layout_keeps_raw_q4_and_sized_q6_soa_slot() {
        let layout = ExpertArenaLayout::qwen(2048, 512).unwrap();
        assert_eq!(layout.gate_offset_words, 0);
        assert_eq!(layout.gate_bytes(), 512 * 8 * 144);
        assert_eq!(layout.up_bytes(), 512 * 8 * 144);
        assert_eq!(
            layout.down_raw_bytes(QuantType::Q5K).unwrap(),
            2048 * 2 * 176
        );
        assert_eq!(
            layout.down_resident_bytes(QuantType::Q6K).unwrap(),
            2048 * 2 * 53 * 4
        );
        assert!(
            layout.slot_stride_bytes() >= layout.down_offset_words as u64 * 4 + 2048 * 2 * 53 * 4
        );
    }

    #[test]
    fn qwen_q8_layout_reserves_three_soa_projections() {
        let layout = ExpertArenaLayout::qwen_q8(2048, 512).unwrap();
        let projection_bytes = 512 * 64 * 9 * 4;
        assert_eq!(layout.format, ExpertArenaFormat::Q8Zero);
        assert_eq!(layout.gate_bytes(), projection_bytes);
        assert_eq!(layout.up_bytes(), projection_bytes);
        assert_eq!(layout.down_offset_words as u64 * 4, projection_bytes * 2);
        assert_eq!(layout.slot_stride_bytes(), projection_bytes * 3);
        assert_eq!(layout.gate_raw_bytes().unwrap(), 512 * 64 * 34);
        assert_eq!(
            layout.down_raw_bytes(QuantType::Q8_0).unwrap(),
            2048 * 16 * 34
        );
    }

    #[test]
    fn arena_capacity_uses_budget_and_descriptor_limits() {
        let layout = ExpertArenaLayout::qwen(2048, 512).unwrap();
        let required = layout.required_bytes(40).unwrap();
        assert_eq!(layout.max_slot_count(required - 1, u64::MAX), 39);
        assert_eq!(layout.max_slot_count(required, u64::MAX), 40);
        assert_eq!(layout.max_slot_count(u64::MAX, required), 40);
    }

    #[test]
    fn slot_map_touches_batch_hits_before_evicting_old_entries() {
        let a = ExpertArenaKey {
            layer: 0,
            expert: 1,
        };
        let b = ExpertArenaKey {
            layer: 0,
            expert: 2,
        };
        let c = ExpertArenaKey {
            layer: 1,
            expert: 3,
        };
        let mut slots = ExpertSlotMap::new(2, 16);
        let (a_slot, _) = slots.get_or_assign(a).unwrap();
        slots.get_or_assign(b).unwrap();
        slots.touch_batch(&[a, c]).unwrap();
        let (c_slot, hit) = slots.get_or_assign(c).unwrap();
        assert!(!hit);
        assert_ne!(a_slot, c_slot);
        assert_eq!(slots.get_or_assign(a).unwrap(), (a_slot, true));
    }
}
