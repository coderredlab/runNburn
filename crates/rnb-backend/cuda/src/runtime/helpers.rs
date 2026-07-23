use super::*;

pub(super) fn q4k_resident_key(weights: &[u8]) -> (usize, usize) {
    // cu114: content(len + 앞뒤 샘플 hash) 기반 키. 이전 ptr 기반은 같은 weight 가
    // 매 forward 다른 slice 주소로 와서 cache miss → prewarm 이 매 prefill·decode token
    // 마다 10MB weight 를 재 H2D upload(decode dispatch 의 주범). 전체 10MB hash 는
    // 매 lookup 비싸서 len + 앞512 + 끝512 bytes FNV 샘플로 식별(같은 weight=같은 키).
    let n = weights.len();
    let sample = 512.min(n);
    let mut h = 0xcbf29ce484222325_u64;
    for &b in &weights[..sample] {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for &b in &weights[n - sample..] {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    (n, h as usize)
}

pub(super) fn unique_q4k_slot_bytes<'a>(slots: impl Iterator<Item = &'a &'a [u8]>) -> usize {
    let mut seen = HashSet::new();
    let mut bytes = 0usize;
    for &weights in slots {
        let key = q4k_resident_key(weights);
        if seen.insert(key) {
            bytes = bytes.saturating_add(weights.len());
        }
    }
    bytes
}

pub(super) fn qwen35_decode_resident_batch_bytes(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
) -> usize {
    unique_q4k_slot_bytes(
        gate_weights
            .iter()
            .chain(up_weights.iter())
            .chain(down_weights.iter()),
    )
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Qwen35SelectedExpertBases<'a> {
    pub(super) gate_all: &'a [u8],
    pub(super) up_all: &'a [u8],
    pub(super) down_all: &'a [u8],
    pub(super) gate_bytes_per_expert: usize,
    pub(super) up_bytes_per_expert: usize,
    pub(super) down_bytes_per_expert: usize,
    pub(super) n_expert: usize,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedExpertSlices<'a> {
    pub(super) gate_weights: Vec<&'a [u8]>,
    pub(super) up_weights: Vec<&'a [u8]>,
    pub(super) down_weights: Vec<&'a [u8]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseSlotOffset {
    pub(super) expert_id: u32,
    pub(super) gate_byte_offset: usize,
    pub(super) up_byte_offset: usize,
    pub(super) down_byte_offset: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseSlotOffsets {
    pub(super) slots: Vec<Qwen35SelectedBaseSlotOffset>,
    pub(super) gate_bytes_per_expert: usize,
    pub(super) up_bytes_per_expert: usize,
    pub(super) down_bytes_per_expert: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Qwen35SelectedBaseWeightRole {
    Gate,
    Up,
    Down,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseTempSlabUpload {
    pub(super) role: Qwen35SelectedBaseWeightRole,
    pub(super) src_byte_offset: usize,
    pub(super) slab_byte_offset: usize,
    pub(super) bytes: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseTempSlabSlotPtrPlan {
    pub(super) gate_ptrs: Vec<u64>,
    pub(super) up_ptrs: Vec<u64>,
    pub(super) down_ptrs: Vec<u64>,
    pub(super) uploads: Vec<Qwen35SelectedBaseTempSlabUpload>,
    pub(super) slab_bytes: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseTempSlabDevicePtrPlan {
    pub(super) expert_slab_indices: Vec<u32>,
    pub(super) uploads: Vec<Qwen35SelectedBaseTempSlabUpload>,
    pub(super) gate_base: u64,
    pub(super) up_base: u64,
    pub(super) down_base: u64,
    pub(super) slab_bytes: usize,
}

fn qwen35_coalesce_selected_base_temp_slab_uploads(
    uploads: Vec<Qwen35SelectedBaseTempSlabUpload>,
) -> Result<Vec<Qwen35SelectedBaseTempSlabUpload>, String> {
    if !qwen35_selected_base_range_upload_enabled() {
        return Ok(uploads);
    }
    let mut coalesced: Vec<Qwen35SelectedBaseTempSlabUpload> = Vec::with_capacity(uploads.len());
    for upload in uploads {
        if let Some(last) = coalesced.last_mut() {
            let last_src_end = last
                .src_byte_offset
                .checked_add(last.bytes)
                .ok_or_else(|| {
                    format!(
                    "Qwen35 selected-base upload coalesce source end overflows: offset={} bytes={}",
                    last.src_byte_offset, last.bytes
                )
                })?;
            let last_dst_end = last
                .slab_byte_offset
                .checked_add(last.bytes)
                .ok_or_else(|| {
                    format!(
                        "Qwen35 selected-base upload coalesce slab end overflows: offset={} bytes={}",
                        last.slab_byte_offset, last.bytes
                    )
                })?;
            if last.role == upload.role
                && last_src_end == upload.src_byte_offset
                && last_dst_end == upload.slab_byte_offset
            {
                last.bytes = last.bytes.checked_add(upload.bytes).ok_or_else(|| {
                    format!(
                        "Qwen35 selected-base upload coalesce bytes overflow: left={} right={}",
                        last.bytes, upload.bytes
                    )
                })?;
                continue;
            }
        }
        coalesced.push(upload);
    }
    Ok(coalesced)
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct Qwen35SelectedBaseResidentRole {
    pub(super) role: Qwen35SelectedBaseWeightRole,
    pub(super) expert_id: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35SelectedBaseMixedWeightSource {
    Resident,
    Temp { slab_byte_offset: usize },
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseMixedExpertSources {
    pub(super) gate: Qwen35SelectedBaseMixedWeightSource,
    pub(super) up: Qwen35SelectedBaseMixedWeightSource,
    pub(super) down: Qwen35SelectedBaseMixedWeightSource,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseMixedResidentTempPlan {
    pub(super) expert_sources: Vec<Option<Qwen35SelectedBaseMixedExpertSources>>,
    pub(super) uploads: Vec<Qwen35SelectedBaseTempSlabUpload>,
    pub(super) slab_bytes: usize,
    pub(super) selected_experts: usize,
    pub(super) resident_upload_bytes_saved: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Qwen35SelectedBaseResidentAdmissionStats {
    pub(super) candidate_pages: usize,
    pub(super) selected_pages: usize,
    pub(super) already_resident_pages: usize,
    pub(super) admitted_pages: usize,
    pub(super) skipped_by_token_window: bool,
    pub(super) skipped_by_cost_gate: bool,
    pub(super) budget_bytes: u64,
    pub(super) selected_bytes: u64,
    pub(super) admitted_bytes: u64,
    pub(super) admission_cost_bytes: u64,
    pub(super) eviction_cost_bytes: u64,
    pub(super) predicted_saved_bytes: u64,
    pub(super) net_saved_bytes: i128,
    pub(super) already_resident_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Qwen35SelectedBaseTempSlabKey {
    role: Qwen35SelectedBaseWeightRole,
    src_byte_offset: usize,
    bytes: usize,
}

pub(super) struct Qwen35SelectedBaseSparseRequest<'a> {
    pub(super) bases: Qwen35SelectedExpertBases<'a>,
    pub(super) expert_ids: &'a [u32],
    pub(super) route_weights: &'a [f32],
    pub(super) token_ids: &'a [u32],
}

pub(super) struct Qwen35SelectedBaseSparseInputs<'a> {
    pub(super) gate_weights: Vec<&'a [u8]>,
    pub(super) up_weights: Vec<&'a [u8]>,
    pub(super) down_weights: Vec<&'a [u8]>,
    pub(super) route_weights: &'a [f32],
    pub(super) token_ids: &'a [u32],
}

pub(in crate::runtime) struct PreparedQwen35MixedExpertPtrs {
    pub(in crate::runtime) gate_ptrs: Vec<u64>,
    pub(in crate::runtime) up_ptrs: Vec<u64>,
    pub(in crate::runtime) down_ptrs: Vec<u64>,
}

pub(in crate::runtime) struct PreparedQwen35DeviceSlotPtrs {
    pub(in crate::runtime) expert_ids: Vec<u32>,
    pub(in crate::runtime) expert_slab_indices: Vec<u32>,
    pub(in crate::runtime) gate_base: u64,
    pub(in crate::runtime) up_base: u64,
    pub(in crate::runtime) down_base: u64,
    pub(in crate::runtime) gate_expert_bytes: usize,
    pub(in crate::runtime) up_expert_bytes: usize,
    pub(in crate::runtime) down_expert_bytes: usize,
    pub(in crate::runtime) selected_upload_calls: usize,
    pub(in crate::runtime) selected_upload_bytes: usize,
    pub(in crate::runtime) mixed_expert_ptrs: Option<PreparedQwen35MixedExpertPtrs>,
    pub(in crate::runtime) group_meta2: Vec<u32>,
    pub(in crate::runtime) group_meta4: Vec<u32>,
    pub(in crate::runtime) group_meta8: Vec<u32>,
    pub(in crate::runtime) group_meta16: Vec<u32>,
    pub(in crate::runtime) group_meta32: Vec<u32>,
    pub(in crate::runtime) group_meta64: Vec<u32>,
}

impl PreparedQwen35DeviceSlotPtrs {
    pub(in crate::runtime) fn group_meta_for_max_group(
        &self,
        max_group: usize,
    ) -> Result<&[u32], String> {
        match max_group {
            2 => Ok(&self.group_meta2),
            4 => Ok(&self.group_meta4),
            8 => Ok(&self.group_meta8),
            16 => Ok(&self.group_meta16),
            32 => Ok(&self.group_meta32),
            64 => Ok(&self.group_meta64),
            other => Err(format!(
                "Qwen35 selected-base direct sparse unsupported group size {other}"
            )),
        }
    }
}

pub(in crate::runtime) struct PreparedQwen35SparseGroupMeta {
    pub(in crate::runtime) group_meta2: Vec<u32>,
    pub(in crate::runtime) group_meta4: Vec<u32>,
    pub(in crate::runtime) group_meta8: Vec<u32>,
    pub(in crate::runtime) group_meta16: Vec<u32>,
    pub(in crate::runtime) group_meta32: Vec<u32>,
    pub(in crate::runtime) group_meta64: Vec<u32>,
}

impl PreparedQwen35SparseGroupMeta {
    pub(in crate::runtime) fn group_meta_for_max_group(
        &self,
        max_group: usize,
    ) -> Result<&[u32], String> {
        match max_group {
            2 => Ok(&self.group_meta2),
            4 => Ok(&self.group_meta4),
            8 => Ok(&self.group_meta8),
            16 => Ok(&self.group_meta16),
            32 => Ok(&self.group_meta32),
            64 => Ok(&self.group_meta64),
            other => Err(format!(
                "Qwen35 prepared sparse unsupported group size {other}"
            )),
        }
    }
}

pub(in crate::runtime) struct PreparedQwen35DeviceSparseRoute {
    pub(in crate::runtime) route_weights_dev: u64,
    pub(in crate::runtime) token_ids_dev: u64,
    pub(in crate::runtime) slots: usize,
}

pub(in crate::runtime) struct PreparedQwen35SparseSlots {
    pub(in crate::runtime) gate_ptrs: Vec<u64>,
    pub(in crate::runtime) up_ptrs: Vec<u64>,
    pub(in crate::runtime) down_ptrs: Vec<u64>,
    pub(in crate::runtime) slot_count: Option<usize>,
    pub(in crate::runtime) temp_slab_ptrs: Vec<u64>,
    pub(in crate::runtime) copy_stream_upload: bool,
    pub(in crate::runtime) device_slot_ptrs: Option<PreparedQwen35DeviceSlotPtrs>,
    pub(in crate::runtime) group_meta: Option<PreparedQwen35SparseGroupMeta>,
    pub(in crate::runtime) device_route: Option<PreparedQwen35DeviceSparseRoute>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35SelectedSparseRouteSource {
    Host,
    Device,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35SelectedSparseSlotPointerSource {
    Host,
    DeviceCompact { selected_experts: usize },
    DeviceMixed { selected_experts: usize },
}

impl Qwen35SelectedSparseSlotPointerSource {
    fn slot_ptr_build_launches(self) -> usize {
        match self {
            Self::Host => 0,
            Self::DeviceCompact { .. } | Self::DeviceMixed { .. } => 1,
        }
    }

    fn descriptor_h2d_calls(self) -> usize {
        match self {
            Self::Host => 3,
            Self::DeviceCompact { .. } => 2,
            Self::DeviceMixed { .. } => 4,
        }
    }

    fn descriptor_bytes(self, slots: usize) -> usize {
        match self {
            Self::Host => slots * 3 * std::mem::size_of::<u64>(),
            Self::DeviceCompact { selected_experts } => {
                (slots + selected_experts) * std::mem::size_of::<u32>()
            }
            Self::DeviceMixed { selected_experts } => {
                slots * std::mem::size_of::<u32>()
                    + selected_experts * 3 * std::mem::size_of::<u64>()
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35SelectedSparseActivationLayout {
    SeparateSilu,
    FusedSilu,
    Q8DotFusedSilu,
    Pack4F32,
    Pack4F32Group8,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35SelectedSparseDownRunner {
    Q4Existing,
    Q5Existing,
    Q6Existing,
    Q6Q8Dot,
    Q6Pack4F32 { vec4_load: bool },
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35SelectedSparseAccumulationOrder {
    ExistingGroupedByExpertToken,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Qwen35SelectedSparseLaunchPlan {
    pub(super) slot_ptr_build: usize,
    pub(super) zero: usize,
    pub(super) gate_up: usize,
    pub(super) silu: usize,
    pub(super) down: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Qwen35SelectedSparseDescriptorH2dPlan {
    pub(super) route_bytes: usize,
    pub(super) token_bytes: usize,
    pub(super) slot_descriptor_bytes: usize,
    pub(super) group_meta_calls: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedSparseDescriptorInput {
    pub(super) slots: usize,
    pub(super) token_count: usize,
    pub(super) route_from_device: bool,
    pub(super) slot_pointer_source: Qwen35SelectedSparseSlotPointerSource,
    pub(super) gate_up_group: Option<usize>,
    pub(super) down_group: Option<usize>,
    pub(super) down_quant: u32,
    pub(super) zero_output: bool,
    pub(super) q4_gate_up_silu_fused: bool,
    pub(super) q4_gate_up_q8dot: bool,
    pub(super) q4_gate_up_silu_pack4_f32: bool,
    pub(super) q4_gate_up_silu_pack4_group8: bool,
    pub(super) q6_down_q8dot: bool,
    pub(super) q6_down_pack4_f32: bool,
    pub(super) q6_down_pack4_f32_vec4: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedSparseRuntimeDescriptorInput {
    pub(super) slots: usize,
    pub(super) token_count: usize,
    pub(super) route_from_device: bool,
    pub(super) slot_pointer_source: Qwen35SelectedSparseSlotPointerSource,
    pub(super) gate_up_group: Option<usize>,
    pub(super) down_group: Option<usize>,
    pub(super) down_quant: u32,
    pub(super) zero_output: bool,
    pub(super) q4_gate_up_silu_fused: bool,
    pub(super) q4_gate_up_q8dot: bool,
    pub(super) q4_gate_up_silu_pack4_f32: bool,
    pub(super) q4_gate_up_silu_pack4_group8: bool,
    pub(super) q6_down_q8dot: bool,
    pub(super) q6_down_pack4_f32: bool,
    pub(super) q6_down_pack4_f32_vec4_enabled: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35SelectedSparseExecutionDescriptor {
    pub(super) route_source: Qwen35SelectedSparseRouteSource,
    pub(super) slot_pointer_source: Qwen35SelectedSparseSlotPointerSource,
    pub(super) activation_layout: Qwen35SelectedSparseActivationLayout,
    pub(super) down_runner: Qwen35SelectedSparseDownRunner,
    pub(super) accumulation_order: Qwen35SelectedSparseAccumulationOrder,
    pub(super) launches: Qwen35SelectedSparseLaunchPlan,
    pub(super) h2d: Qwen35SelectedSparseDescriptorH2dPlan,
    pub(super) exact_reference: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35SelectedSparseRunnerMode {
    LegacyInline,
    ExactReference,
    CompoundExactReference,
}

#[allow(dead_code)]
pub(super) fn qwen35_selected_sparse_runtime_descriptor(
    input: Qwen35SelectedSparseRuntimeDescriptorInput,
) -> Result<Qwen35SelectedSparseExecutionDescriptor, String> {
    qwen35_selected_sparse_execution_descriptor(Qwen35SelectedSparseDescriptorInput {
        slots: input.slots,
        token_count: input.token_count,
        route_from_device: input.route_from_device,
        slot_pointer_source: input.slot_pointer_source,
        gate_up_group: input.gate_up_group,
        down_group: input.down_group,
        down_quant: input.down_quant,
        zero_output: input.zero_output,
        q4_gate_up_silu_fused: input.q4_gate_up_silu_fused,
        q4_gate_up_q8dot: input.q4_gate_up_q8dot,
        q4_gate_up_silu_pack4_f32: input.q4_gate_up_silu_pack4_f32,
        q4_gate_up_silu_pack4_group8: input.q4_gate_up_silu_pack4_group8,
        q6_down_q8dot: input.q6_down_q8dot,
        q6_down_pack4_f32: input.q6_down_pack4_f32,
        q6_down_pack4_f32_vec4: input.q6_down_pack4_f32 && input.q6_down_pack4_f32_vec4_enabled,
    })
}

#[allow(dead_code)]
pub(super) fn qwen35_selected_sparse_runner_mode(
    abi_enabled: bool,
    compound_enabled: bool,
    descriptor: Option<&Qwen35SelectedSparseExecutionDescriptor>,
) -> Result<Qwen35SelectedSparseRunnerMode, String> {
    if !abi_enabled && !compound_enabled {
        return Ok(Qwen35SelectedSparseRunnerMode::LegacyInline);
    }
    let descriptor = descriptor
        .ok_or_else(|| "Qwen35 selected sparse execution ABI requires descriptor".to_string())?;
    if !descriptor.exact_reference {
        return Err(
            "Qwen35 selected sparse execution ABI requires exact-reference descriptor".to_string(),
        );
    }
    if compound_enabled && qwen35_selected_sparse_compound_runner_supported(descriptor) {
        return Ok(Qwen35SelectedSparseRunnerMode::CompoundExactReference);
    }
    Ok(Qwen35SelectedSparseRunnerMode::ExactReference)
}

fn qwen35_selected_sparse_compound_runner_supported(
    descriptor: &Qwen35SelectedSparseExecutionDescriptor,
) -> bool {
    descriptor.exact_reference
        && descriptor.activation_layout == Qwen35SelectedSparseActivationLayout::Pack4F32Group8
        && matches!(
            descriptor.down_runner,
            Qwen35SelectedSparseDownRunner::Q6Pack4F32 { .. }
        )
        && descriptor.accumulation_order
            == Qwen35SelectedSparseAccumulationOrder::ExistingGroupedByExpertToken
}

#[allow(dead_code)]
pub(super) fn qwen35_selected_sparse_execution_descriptor(
    input: Qwen35SelectedSparseDescriptorInput,
) -> Result<Qwen35SelectedSparseExecutionDescriptor, String> {
    if input.slots == 0 {
        return Err("Qwen35 selected sparse descriptor requires at least one slot".to_string());
    }
    if input.token_count == 0 {
        return Err("Qwen35 selected sparse descriptor requires at least one token".to_string());
    }
    if input.q4_gate_up_q8dot && !matches!(input.gate_up_group, Some(8 | 16 | 32)) {
        return Err(
            "Qwen35 selected sparse Q8-dot activation requires gate/up group8, group16, or group32"
                .to_string(),
        );
    }
    if input.q4_gate_up_q8dot
        && (input.q4_gate_up_silu_pack4_f32 || input.q4_gate_up_silu_pack4_group8)
    {
        return Err(
            "Qwen35 selected sparse Q8-dot and pack4-F32 activations are exclusive".to_string(),
        );
    }
    if input.q4_gate_up_silu_pack4_group8 && !input.q4_gate_up_silu_pack4_f32 {
        return Err(
            "Qwen35 selected sparse group8 pack4 activation requires pack4-F32".to_string(),
        );
    }
    if input.q4_gate_up_silu_pack4_group8 && !matches!(input.gate_up_group, Some(8 | 16)) {
        return Err(
            "Qwen35 selected sparse grouped pack4 activation requires gate/up group8 or group16"
                .to_string(),
        );
    }
    if input.q6_down_pack4_f32 && input.down_quant != 14 {
        return Err("Qwen35 selected sparse pack4-F32 down requires Q6_K down".to_string());
    }
    if input.q6_down_pack4_f32 && input.down_group != Some(4) {
        return Err("Qwen35 selected sparse pack4-F32 down requires group4 down".to_string());
    }
    if input.q6_down_pack4_f32_vec4 && !input.q6_down_pack4_f32 {
        return Err("Qwen35 selected sparse vec4 down requires pack4-F32 down".to_string());
    }
    if input.q6_down_q8dot && input.q6_down_pack4_f32 {
        return Err("Qwen35 selected sparse q8dot and pack4-F32 down are exclusive".to_string());
    }

    let activation_layout = if input.q4_gate_up_q8dot {
        Qwen35SelectedSparseActivationLayout::Q8DotFusedSilu
    } else if input.q4_gate_up_silu_pack4_group8 {
        Qwen35SelectedSparseActivationLayout::Pack4F32Group8
    } else if input.q4_gate_up_silu_pack4_f32 {
        Qwen35SelectedSparseActivationLayout::Pack4F32
    } else if input.q4_gate_up_silu_fused {
        Qwen35SelectedSparseActivationLayout::FusedSilu
    } else {
        Qwen35SelectedSparseActivationLayout::SeparateSilu
    };

    let down_runner = match input.down_quant {
        12 => Qwen35SelectedSparseDownRunner::Q4Existing,
        13 => Qwen35SelectedSparseDownRunner::Q5Existing,
        14 if input.q6_down_q8dot => Qwen35SelectedSparseDownRunner::Q6Q8Dot,
        14 if input.q6_down_pack4_f32 => Qwen35SelectedSparseDownRunner::Q6Pack4F32 {
            vec4_load: input.q6_down_pack4_f32_vec4,
        },
        14 => Qwen35SelectedSparseDownRunner::Q6Existing,
        other => {
            return Err(format!(
                "Qwen35 selected sparse descriptor unsupported down quant {other}"
            ))
        }
    };

    let route_source = if input.route_from_device {
        Qwen35SelectedSparseRouteSource::Device
    } else {
        Qwen35SelectedSparseRouteSource::Host
    };
    let silu_launches =
        if input.q4_gate_up_q8dot || input.q4_gate_up_silu_fused || input.q4_gate_up_silu_pack4_f32
        {
            0
        } else {
            1
        };
    let mut group_meta_calls = usize::from(input.gate_up_group.is_some());
    let group8_packed_activation = input.q4_gate_up_q8dot || input.q4_gate_up_silu_pack4_group8;
    if group8_packed_activation {
        group_meta_calls += 1;
    }
    if input.q6_down_pack4_f32 && group8_packed_activation {
        group_meta_calls += 1;
    } else if input.down_group.is_some() && input.gate_up_group != input.down_group {
        group_meta_calls += 1;
    }

    Ok(Qwen35SelectedSparseExecutionDescriptor {
        route_source,
        slot_pointer_source: input.slot_pointer_source,
        activation_layout,
        down_runner,
        accumulation_order: Qwen35SelectedSparseAccumulationOrder::ExistingGroupedByExpertToken,
        launches: Qwen35SelectedSparseLaunchPlan {
            slot_ptr_build: input.slot_pointer_source.slot_ptr_build_launches(),
            zero: usize::from(input.zero_output),
            gate_up: 1 + usize::from(input.q4_gate_up_q8dot),
            silu: silu_launches,
            down: 1,
        },
        h2d: Qwen35SelectedSparseDescriptorH2dPlan {
            route_bytes: if input.route_from_device {
                0
            } else {
                input.slots * std::mem::size_of::<f32>()
            },
            token_bytes: if input.route_from_device {
                0
            } else {
                input.slots * std::mem::size_of::<u32>()
            },
            slot_descriptor_bytes: input.slot_pointer_source.descriptor_bytes(input.slots),
            group_meta_calls,
        },
        exact_reference: true,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Qwen35SelectedSparseBoundaryStats {
    pub(super) slots: usize,
    pub(super) unique_experts: usize,
    pub(super) selected_upload_calls: usize,
    pub(super) selected_upload_bytes: usize,
    pub(super) route_h2d_bytes: usize,
    pub(super) token_h2d_bytes: usize,
    pub(super) device_slot_h2d_bytes: usize,
    pub(super) group_meta_h2d_bytes: usize,
    pub(super) descriptor_h2d_calls: usize,
    pub(super) group_meta_h2d_calls: usize,
    pub(super) slot_ptr_build_launches: usize,
    pub(super) zero_launches: usize,
    pub(super) gate_up_launches: usize,
    pub(super) silu_launches: usize,
    pub(super) down_launches: usize,
}

impl Qwen35SelectedSparseBoundaryStats {
    pub(super) fn apply_execution_descriptor(
        &mut self,
        descriptor: &Qwen35SelectedSparseExecutionDescriptor,
    ) {
        self.route_h2d_bytes = descriptor.h2d.route_bytes;
        self.token_h2d_bytes = descriptor.h2d.token_bytes;
        self.device_slot_h2d_bytes = descriptor.h2d.slot_descriptor_bytes;
        self.descriptor_h2d_calls = descriptor.slot_pointer_source.descriptor_h2d_calls()
            + match descriptor.route_source {
                Qwen35SelectedSparseRouteSource::Host => 2,
                Qwen35SelectedSparseRouteSource::Device => 0,
            };
        self.group_meta_h2d_calls = descriptor.h2d.group_meta_calls;
        self.slot_ptr_build_launches = descriptor.launches.slot_ptr_build;
        self.zero_launches = descriptor.launches.zero;
        self.gate_up_launches = descriptor.launches.gate_up;
        self.silu_launches = descriptor.launches.silu;
        self.down_launches = descriptor.launches.down;
    }

    pub(super) fn total_descriptor_h2d_calls(&self) -> usize {
        self.descriptor_h2d_calls + self.group_meta_h2d_calls
    }

    pub(super) fn total_kernel_launches(&self) -> usize {
        self.slot_ptr_build_launches
            + self.zero_launches
            + self.gate_up_launches
            + self.silu_launches
            + self.down_launches
    }
}

pub(in crate::runtime) struct DeferredQwen35SelectedBaseSparse<'a> {
    pub(in crate::runtime) gate_all: &'a [u8],
    pub(in crate::runtime) up_all: &'a [u8],
    pub(in crate::runtime) down_all: &'a [u8],
    pub(in crate::runtime) expert_ids: &'a [u32],
    pub(in crate::runtime) down_quant: u32,
    pub(in crate::runtime) n_ff: usize,
    pub(in crate::runtime) n_embd: usize,
}

pub(super) struct Qwen35RouteArrays<'a> {
    pub(super) expert_ids: &'a [u32],
    pub(super) route_weights: &'a [f32],
    pub(super) token_ids: &'a [u32],
    pub(super) seq_len: usize,
    pub(super) n_expert: usize,
    pub(super) n_expert_used: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35DownTokenMajorPlan {
    pub(super) token_offsets: Vec<u32>,
    pub(super) slot_indices: Vec<u32>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35ExpertRun {
    pub(super) expert_id: u32,
    pub(super) slot_start: usize,
    pub(super) len: usize,
    pub(super) full_tiles: usize,
    pub(super) tail: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35ExpertRunSpecializedTile {
    pub(super) expert_id: u32,
    pub(super) run_start: usize,
    pub(super) run_len: usize,
    pub(super) tile_start: usize,
    pub(super) tile_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35GroupMetaLenSplit {
    pub(super) matching: Vec<u32>,
    pub(super) other: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35FullLayerSlotPtrPlan {
    HostPointerUpload,
    DevicePointerBuild,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Qwen35ResidentExpertPageRole {
    Gate,
    Up,
    Down,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Qwen35ResidentExpertPageCandidate {
    pub(super) layer_idx: usize,
    pub(super) expert_id: u32,
    pub(super) role: Qwen35ResidentExpertPageRole,
    pub(super) quant: u32,
    pub(super) byte_offset: usize,
    pub(super) bytes: usize,
    pub(super) reuse_count: u32,
    pub(super) route_weight_sum: f32,
    pub(super) window_tokens: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Qwen35ResidentExpertPageKey {
    layer_idx: usize,
    expert_id: u32,
    role: Qwen35ResidentExpertPageRole,
    quant: u32,
    byte_offset: usize,
    bytes: usize,
}

impl From<&Qwen35ResidentExpertPageCandidate> for Qwen35ResidentExpertPageKey {
    fn from(candidate: &Qwen35ResidentExpertPageCandidate) -> Self {
        Self {
            layer_idx: candidate.layer_idx,
            expert_id: candidate.expert_id,
            role: candidate.role,
            quant: candidate.quant,
            byte_offset: candidate.byte_offset,
            bytes: candidate.bytes,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Qwen35ResidentExpertPagePlan {
    pub(super) selected: Vec<Qwen35ResidentExpertPageCandidate>,
    pub(super) spilled: Vec<Qwen35ResidentExpertPageCandidate>,
    pub(super) selected_bytes: u64,
    pub(super) spilled_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Qwen35ResidentExpertPageAdmissionCost {
    pub(super) new_admission_bytes: u64,
    pub(super) eviction_cost_bytes: u64,
    pub(super) predicted_saved_bytes: u64,
    pub(super) net_saved_bytes: i128,
    pub(super) already_resident_bytes: u64,
    pub(super) profitable: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35ResidentExpertPageBudget {
    pub(super) free_after_reserve_bytes: usize,
    pub(super) cache_headroom_bytes: usize,
    pub(super) evicting_budget_bytes: usize,
    pub(super) budget_bytes: usize,
}

pub(super) fn qwen35_full_layer_slot_ptr_plan(
    device_pointer_build_enabled: bool,
    has_range_slab_offsets: bool,
) -> Qwen35FullLayerSlotPtrPlan {
    if device_pointer_build_enabled && !has_range_slab_offsets {
        Qwen35FullLayerSlotPtrPlan::DevicePointerBuild
    } else {
        Qwen35FullLayerSlotPtrPlan::HostPointerUpload
    }
}

pub(super) fn qwen35_selected_base_temp_slab_ptrs_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_PTRS")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

pub(super) fn qwen35_selected_base_device_slot_ptrs_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_DEVICE_SLOT_PTRS")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

pub(super) fn qwen35_selected_base_direct_sparse_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_DIRECT_SPARSE")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

pub(super) fn qwen35_device_sparse_route_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_DEVICE_SPARSE_ROUTE")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_copy_stream_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_COPY_STREAM")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

pub(super) fn qwen35_selected_base_range_upload_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RANGE_UPLOAD")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_temp_slab_cache_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_TEMP_SLAB_CACHE")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_pinned_staging_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_PINNED_STAGING")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_overlap_staging_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_OVERLAP_STAGING")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_mixed_resident_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_RESIDENT")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_mixed_device_slot_ptrs_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_MIXED_DEVICE_SLOT_PTRS")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_resident_admission_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_resident_admission_cost_gate_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_COST_GATE")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

pub(super) fn qwen35_selected_base_resident_admission_history_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_HISTORY")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_base_resident_admission_future_hits() -> Result<u32, String> {
    match std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS") {
        Ok(value) => value.parse::<u32>().map_err(|err| {
            format!(
                "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS must be u32: {err}"
            )
        }),
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(err) => Err(format!(
            "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_FUTURE_HITS read failed: {err}"
        )),
    }
}

pub(super) fn qwen35_selected_base_resident_admission_token_window_allows(
    token_count: usize,
) -> Result<bool, String> {
    match std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_MAX_TOKENS") {
        Ok(value) => {
            let max_tokens = value.parse::<usize>().map_err(|err| {
                format!(
                    "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_MAX_TOKENS must be usize: {err}"
                )
            })?;
            Ok(token_count <= max_tokens)
        }
        Err(std::env::VarError::NotPresent) => Ok(true),
        Err(err) => Err(format!(
            "RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_MAX_TOKENS read failed: {err}"
        )),
    }
}

pub(super) fn qwen35_selected_sparse_fused_boundary_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_SPARSE_FUSED_BOUNDARY")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_sparse_execution_abi_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_SPARSE_EXECUTION_ABI")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn qwen35_selected_sparse_compound_runner_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_SELECTED_SPARSE_COMPOUND_RUNNER")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

pub(super) fn qwen35_sparse_slot_count(
    gate_weight_slots: usize,
    device_slot_slots: Option<usize>,
    prepared_slot_count: Option<usize>,
) -> usize {
    if let Some(slots) = device_slot_slots {
        slots
    } else if gate_weight_slots > 0 {
        gate_weight_slots
    } else {
        prepared_slot_count.unwrap_or(0)
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(super) fn qwen35_resident_expert_page_candidates(
    layer_idx: usize,
    bases: &Qwen35SelectedExpertBases<'_>,
    expert_ids: &[u32],
    route_weights: &[f32],
    window_tokens: usize,
    gate_quant: u32,
    up_quant: u32,
    down_quant: u32,
) -> Result<Vec<Qwen35ResidentExpertPageCandidate>, String> {
    if route_weights.len() != expert_ids.len() {
        return Err(format!(
            "resident expert page route length mismatch: expert={} route={}",
            expert_ids.len(),
            route_weights.len()
        ));
    }
    validate_qwen35_selected_base_len(
        "gate",
        bases.gate_all,
        bases.gate_bytes_per_expert,
        bases.n_expert,
    )?;
    validate_qwen35_selected_base_len(
        "up",
        bases.up_all,
        bases.up_bytes_per_expert,
        bases.n_expert,
    )?;
    validate_qwen35_selected_base_len(
        "down",
        bases.down_all,
        bases.down_bytes_per_expert,
        bases.n_expert,
    )?;

    let mut reuse_counts = vec![0u32; bases.n_expert];
    let mut route_sums = vec![0.0f32; bases.n_expert];
    for (slot, (&expert_id, &route_weight)) in
        expert_ids.iter().zip(route_weights.iter()).enumerate()
    {
        let expert = expert_id as usize;
        if expert >= bases.n_expert {
            return Err(format!(
                "expert id out of range: slot={slot} expert_id={expert_id} n_expert={}",
                bases.n_expert
            ));
        }
        if !route_weight.is_finite() || route_weight < 0.0 {
            return Err(format!(
                "resident expert page route weight must be finite and non-negative: slot={slot} weight={route_weight}"
            ));
        }
        reuse_counts[expert] = reuse_counts[expert].saturating_add(1);
        route_sums[expert] += route_weight;
    }

    let mut candidates = Vec::new();
    for expert in 0..bases.n_expert {
        let reuse_count = reuse_counts[expert];
        if reuse_count == 0 {
            continue;
        }
        let route_weight_sum = route_sums[expert];
        candidates.push(Qwen35ResidentExpertPageCandidate {
            layer_idx,
            expert_id: expert as u32,
            role: Qwen35ResidentExpertPageRole::Gate,
            quant: gate_quant,
            byte_offset: expert * bases.gate_bytes_per_expert,
            bytes: bases.gate_bytes_per_expert,
            reuse_count,
            route_weight_sum,
            window_tokens,
        });
        candidates.push(Qwen35ResidentExpertPageCandidate {
            layer_idx,
            expert_id: expert as u32,
            role: Qwen35ResidentExpertPageRole::Up,
            quant: up_quant,
            byte_offset: expert * bases.up_bytes_per_expert,
            bytes: bases.up_bytes_per_expert,
            reuse_count,
            route_weight_sum,
            window_tokens,
        });
        candidates.push(Qwen35ResidentExpertPageCandidate {
            layer_idx,
            expert_id: expert as u32,
            role: Qwen35ResidentExpertPageRole::Down,
            quant: down_quant,
            byte_offset: expert * bases.down_bytes_per_expert,
            bytes: bases.down_bytes_per_expert,
            reuse_count,
            route_weight_sum,
            window_tokens,
        });
    }
    Ok(candidates)
}

#[allow(dead_code)]
pub(super) fn qwen35_resident_expert_page_future_hit_counts(
    selected: &[Qwen35ResidentExpertPageCandidate],
    future_windows: &[&[Qwen35ResidentExpertPageCandidate]],
) -> Vec<u32> {
    let selected_keys = selected
        .iter()
        .map(Qwen35ResidentExpertPageKey::from)
        .collect::<Vec<_>>();
    let mut future_hits = vec![0u32; selected.len()];

    for window in future_windows {
        let mut window_keys = std::collections::HashSet::new();
        for candidate in *window {
            window_keys.insert(Qwen35ResidentExpertPageKey::from(candidate));
        }
        for (idx, key) in selected_keys.iter().enumerate() {
            if window_keys.contains(key) {
                future_hits[idx] = future_hits[idx].saturating_add(1);
            }
        }
    }

    future_hits
}

pub(super) fn qwen35_resident_expert_page_source_future_hits_and_observe(
    history: &mut std::collections::HashMap<(usize, usize), u32>,
    source_keys: impl IntoIterator<Item = (usize, usize)>,
) -> Vec<u32> {
    let source_keys = source_keys.into_iter().collect::<Vec<_>>();
    let future_hits = source_keys
        .iter()
        .map(|key| history.get(key).copied().unwrap_or(0))
        .collect::<Vec<_>>();

    let mut observed = std::collections::HashSet::new();
    for key in source_keys {
        if observed.insert(key) {
            let entry = history.entry(key).or_insert(0);
            *entry = entry.saturating_add(1);
        }
    }

    future_hits
}

#[allow(dead_code)]
pub(super) fn qwen35_resident_expert_page_plan(
    candidates: &[Qwen35ResidentExpertPageCandidate],
    budget_bytes: u64,
) -> Qwen35ResidentExpertPagePlan {
    let mut ranked = candidates
        .iter()
        .cloned()
        .enumerate()
        .collect::<Vec<(usize, Qwen35ResidentExpertPageCandidate)>>();
    ranked.sort_by(|(left_idx, left), (right_idx, right)| {
        right
            .reuse_count
            .cmp(&left.reuse_count)
            .then_with(|| {
                right
                    .route_weight_sum
                    .total_cmp(&left.route_weight_sum)
                    .then_with(|| left.bytes.cmp(&right.bytes))
            })
            .then_with(|| left_idx.cmp(right_idx))
    });

    let mut remaining = budget_bytes;
    let mut selected_ranked = Vec::new();
    let mut spilled_ranked = Vec::new();
    for (idx, candidate) in ranked {
        let bytes = candidate.bytes as u64;
        if bytes <= remaining {
            remaining -= bytes;
            selected_ranked.push((idx, candidate));
        } else {
            spilled_ranked.push((idx, candidate));
        }
    }
    selected_ranked.sort_by_key(|(idx, _)| *idx);
    spilled_ranked.sort_by_key(|(idx, _)| *idx);
    let selected = selected_ranked
        .into_iter()
        .map(|(_, candidate)| candidate)
        .collect::<Vec<_>>();
    let spilled = spilled_ranked
        .into_iter()
        .map(|(_, candidate)| candidate)
        .collect::<Vec<_>>();
    let selected_bytes = selected
        .iter()
        .map(|candidate| candidate.bytes as u64)
        .sum();
    let spilled_bytes = spilled.iter().map(|candidate| candidate.bytes as u64).sum();
    Qwen35ResidentExpertPagePlan {
        selected,
        spilled,
        selected_bytes,
        spilled_bytes,
    }
}

pub(super) fn qwen35_resident_expert_page_admission_cost(
    selected: &[Qwen35ResidentExpertPageCandidate],
    already_resident: &[bool],
    future_hits: u32,
) -> Result<Qwen35ResidentExpertPageAdmissionCost, String> {
    let page_future_hits = vec![future_hits; selected.len()];
    qwen35_resident_expert_page_admission_cost_with_future_hits(
        selected,
        already_resident,
        &page_future_hits,
    )
}

pub(super) fn qwen35_resident_expert_page_admission_cost_with_future_hits(
    selected: &[Qwen35ResidentExpertPageCandidate],
    already_resident: &[bool],
    future_hits: &[u32],
) -> Result<Qwen35ResidentExpertPageAdmissionCost, String> {
    qwen35_resident_expert_page_admission_cost_with_future_hits_and_eviction_cost(
        selected,
        already_resident,
        future_hits,
        0,
    )
}

pub(super) fn qwen35_resident_expert_page_admission_cost_with_future_hits_and_eviction_cost(
    selected: &[Qwen35ResidentExpertPageCandidate],
    already_resident: &[bool],
    future_hits: &[u32],
    eviction_cost_bytes: u64,
) -> Result<Qwen35ResidentExpertPageAdmissionCost, String> {
    if selected.len() != already_resident.len() {
        return Err(format!(
            "resident expert page admission cost length mismatch: selected={} resident={}",
            selected.len(),
            already_resident.len()
        ));
    }
    if selected.len() != future_hits.len() {
        return Err(format!(
            "resident expert page admission future-hit length mismatch: selected={} future_hits={}",
            selected.len(),
            future_hits.len()
        ));
    }

    let mut new_admission_bytes = 0u64;
    let mut predicted_saved_bytes = 0u64;
    let mut already_resident_bytes = 0u64;
    for ((candidate, already_resident), future_hits) in selected
        .iter()
        .zip(already_resident.iter().copied())
        .zip(future_hits.iter().copied())
    {
        let bytes = candidate.bytes as u64;
        if already_resident {
            already_resident_bytes = already_resident_bytes.saturating_add(bytes);
            continue;
        }
        new_admission_bytes = new_admission_bytes.saturating_add(bytes);
        predicted_saved_bytes =
            predicted_saved_bytes.saturating_add(bytes.saturating_mul(future_hits as u64));
    }
    let net_saved_bytes =
        predicted_saved_bytes as i128 - new_admission_bytes as i128 - eviction_cost_bytes as i128;

    Ok(Qwen35ResidentExpertPageAdmissionCost {
        new_admission_bytes,
        eviction_cost_bytes,
        predicted_saved_bytes,
        net_saved_bytes,
        already_resident_bytes,
        profitable: new_admission_bytes == 0 || net_saved_bytes > 0,
    })
}

#[allow(dead_code)]
pub(super) fn qwen35_resident_expert_page_budget(
    _total_mib: usize,
    free_mib: usize,
    reserve_mib: usize,
    resident_q4k_limit_bytes: usize,
    resident_q4k_bytes: usize,
) -> Qwen35ResidentExpertPageBudget {
    let free_after_reserve_bytes = free_mib
        .saturating_sub(reserve_mib)
        .saturating_mul(1024 * 1024);
    let cache_headroom_bytes = resident_q4k_limit_bytes.saturating_sub(resident_q4k_bytes);
    Qwen35ResidentExpertPageBudget {
        free_after_reserve_bytes,
        cache_headroom_bytes,
        evicting_budget_bytes: free_after_reserve_bytes.min(resident_q4k_limit_bytes),
        budget_bytes: free_after_reserve_bytes.min(cache_headroom_bytes),
    }
}

#[cfg(test)]
pub(super) fn qwen35_selected_base_sparse_inputs_for_test<'a>(
    request: &Qwen35SelectedBaseSparseRequest<'a>,
) -> Result<Qwen35SelectedBaseSparseInputs<'a>, String> {
    qwen35_selected_base_sparse_inputs(request)
}

pub(super) fn qwen35_selected_base_sparse_inputs<'a>(
    request: &Qwen35SelectedBaseSparseRequest<'a>,
) -> Result<Qwen35SelectedBaseSparseInputs<'a>, String> {
    let slots = request.expert_ids.len();
    if request.route_weights.len() != slots || request.token_ids.len() != slots {
        return Err(format!(
            "Qwen35 selected-base sparse length mismatch: expert={} route={} token={}",
            slots,
            request.route_weights.len(),
            request.token_ids.len()
        ));
    }
    let slices = qwen35_selected_base_slices(&request.bases, request.expert_ids)?;
    Ok(Qwen35SelectedBaseSparseInputs {
        gate_weights: slices.gate_weights,
        up_weights: slices.up_weights,
        down_weights: slices.down_weights,
        route_weights: request.route_weights,
        token_ids: request.token_ids,
    })
}

#[cfg(test)]
pub(super) fn qwen35_validate_route_arrays_for_test(
    arrays: &Qwen35RouteArrays<'_>,
) -> Result<(), String> {
    qwen35_validate_route_arrays(arrays)
}

pub(super) fn qwen35_validate_route_arrays(arrays: &Qwen35RouteArrays<'_>) -> Result<(), String> {
    let slots = arrays.expert_ids.len();
    if arrays.route_weights.len() != slots || arrays.token_ids.len() != slots {
        return Err(format!(
            "route array length mismatch: expert={} route={} token={}",
            slots,
            arrays.route_weights.len(),
            arrays.token_ids.len()
        ));
    }
    if arrays.n_expert_used > arrays.n_expert {
        return Err(format!(
            "selected expert count out of range: used={} n_expert={}",
            arrays.n_expert_used, arrays.n_expert
        ));
    }
    let expected_slots = arrays
        .seq_len
        .checked_mul(arrays.n_expert_used)
        .ok_or_else(|| "route array expected slot count overflows usize".to_string())?;
    if slots != expected_slots {
        return Err(format!(
            "route array length mismatch: slots={slots} expected={expected_slots} seq_len={} n_expert_used={}",
            arrays.seq_len, arrays.n_expert_used
        ));
    }
    for (idx, &expert_id) in arrays.expert_ids.iter().enumerate() {
        if expert_id as usize >= arrays.n_expert {
            return Err(format!(
                "expert id out of range: slot={idx} expert_id={expert_id} n_expert={}",
                arrays.n_expert
            ));
        }
    }
    for (idx, &token_id) in arrays.token_ids.iter().enumerate() {
        if token_id as usize >= arrays.seq_len {
            return Err(format!(
                "token id out of range: slot={idx} token_id={token_id} seq_len={}",
                arrays.seq_len
            ));
        }
    }
    Ok(())
}

pub(super) fn qwen35_sort_route_arrays_by_expert_token(
    expert_ids: &mut Vec<u32>,
    route_weights: &mut Vec<f32>,
    token_ids: &mut Vec<u32>,
    seq_len: usize,
    n_expert: usize,
    n_expert_used: usize,
) -> Result<(), String> {
    qwen35_validate_route_arrays(&Qwen35RouteArrays {
        expert_ids,
        route_weights,
        token_ids,
        seq_len,
        n_expert,
        n_expert_used,
    })?;

    let mut order = (0..expert_ids.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|&idx| (expert_ids[idx], token_ids[idx]));
    let sorted_expert_ids = order.iter().map(|&idx| expert_ids[idx]).collect::<Vec<_>>();
    let sorted_route_weights = order
        .iter()
        .map(|&idx| route_weights[idx])
        .collect::<Vec<_>>();
    let sorted_token_ids = order.iter().map(|&idx| token_ids[idx]).collect::<Vec<_>>();
    *expert_ids = sorted_expert_ids;
    *route_weights = sorted_route_weights;
    *token_ids = sorted_token_ids;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qwen35_selected_base_sparse_inputs_from_full_layer<'a>(
    gate_all: &'a [u8],
    up_all: &'a [u8],
    down_all: &'a [u8],
    expert_ids: &'a [u32],
    route_weights: &'a [f32],
    token_ids: &'a [u32],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<Qwen35SelectedBaseSparseInputs<'a>, String> {
    let (gate_bytes_per_expert, up_bytes_per_expert, down_bytes_per_expert, n_expert) =
        qwen35_selected_base_full_layer_shape(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )?;
    let request = Qwen35SelectedBaseSparseRequest {
        bases: Qwen35SelectedExpertBases {
            gate_all,
            up_all,
            down_all,
            gate_bytes_per_expert,
            up_bytes_per_expert,
            down_bytes_per_expert,
            n_expert,
        },
        expert_ids,
        route_weights,
        token_ids,
    };
    qwen35_selected_base_sparse_inputs(&request)
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(super) fn qwen35_selected_base_slot_offsets_from_full_layer(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<Qwen35SelectedBaseSlotOffsets, String> {
    let (gate_bytes_per_expert, up_bytes_per_expert, down_bytes_per_expert, n_expert) =
        qwen35_selected_base_full_layer_shape(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )?;
    qwen35_selected_base_slot_offsets(
        &Qwen35SelectedExpertBases {
            gate_all,
            up_all,
            down_all,
            gate_bytes_per_expert,
            up_bytes_per_expert,
            down_bytes_per_expert,
            n_expert,
        },
        expert_ids,
    )
}

pub(super) fn qwen35_selected_base_full_layer_expert_count(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<usize, String> {
    let (_, _, _, n_expert) = qwen35_selected_base_full_layer_shape(
        gate_all, up_all, down_all, down_quant, n_ff, n_embd,
    )?;
    Ok(n_expert)
}

#[allow(clippy::too_many_arguments)]
fn qwen35_selected_base_full_layer_shape(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<(usize, usize, usize, usize), String> {
    let (gate_bytes_per_expert, up_bytes_per_expert, down_bytes_per_expert) =
        qwen35_selected_base_expert_bytes(down_quant, n_ff, n_embd)?;
    if !gate_all.len().is_multiple_of(gate_bytes_per_expert) {
        return Err(format!(
            "Qwen35 selected-base gate length is not expert-aligned: len={} per_expert={gate_bytes_per_expert}",
            gate_all.len()
        ));
    }
    let n_expert = gate_all.len() / gate_bytes_per_expert;
    validate_qwen35_selected_base_len("up", up_all, up_bytes_per_expert, n_expert)?;
    validate_qwen35_selected_base_len("down", down_all, down_bytes_per_expert, n_expert)?;
    Ok((
        gate_bytes_per_expert,
        up_bytes_per_expert,
        down_bytes_per_expert,
        n_expert,
    ))
}

fn qwen35_selected_base_expert_bytes(
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<(usize, usize, usize), String> {
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 selected-base dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let gate_row_bytes = (n_embd / 256)
        .checked_mul(144)
        .ok_or_else(|| "Qwen35 selected-base gate row bytes overflow usize".to_string())?;
    let gate_bytes_per_expert = n_ff
        .checked_mul(gate_row_bytes)
        .ok_or_else(|| "Qwen35 selected-base gate expert bytes overflow usize".to_string())?;
    let down_row_bytes = match down_quant {
        12 => (n_ff / 256)
            .checked_mul(144)
            .ok_or_else(|| "Qwen35 selected-base Q4 down row bytes overflow usize".to_string())?,
        13 => (n_ff / 256)
            .checked_mul(176)
            .ok_or_else(|| "Qwen35 selected-base Q5 down row bytes overflow usize".to_string())?,
        14 => (n_ff / 256)
            .checked_mul(210)
            .ok_or_else(|| "Qwen35 selected-base Q6 down row bytes overflow usize".to_string())?,
        other => {
            return Err(format!(
                "unsupported Qwen35 selected-base down quant code {other}"
            ))
        }
    };
    let down_bytes_per_expert = n_embd
        .checked_mul(down_row_bytes)
        .ok_or_else(|| "Qwen35 selected-base down expert bytes overflow usize".to_string())?;
    if gate_bytes_per_expert == 0 || down_bytes_per_expert == 0 {
        return Err("Qwen35 selected-base expert byte size must be non-zero".to_string());
    }
    Ok((
        gate_bytes_per_expert,
        gate_bytes_per_expert,
        down_bytes_per_expert,
    ))
}

pub(super) fn qwen35_selected_base_slices<'a>(
    bases: &Qwen35SelectedExpertBases<'a>,
    expert_ids: &[u32],
) -> Result<Qwen35SelectedExpertSlices<'a>, String> {
    let offsets = qwen35_selected_base_slot_offsets(bases, expert_ids)?;

    let mut gate_weights = Vec::with_capacity(expert_ids.len());
    let mut up_weights = Vec::with_capacity(expert_ids.len());
    let mut down_weights = Vec::with_capacity(expert_ids.len());
    for slot in offsets.slots {
        gate_weights.push(qwen35_selected_base_slice_at(
            "gate",
            bases.gate_all,
            slot.gate_byte_offset,
            offsets.gate_bytes_per_expert,
        )?);
        up_weights.push(qwen35_selected_base_slice_at(
            "up",
            bases.up_all,
            slot.up_byte_offset,
            offsets.up_bytes_per_expert,
        )?);
        down_weights.push(qwen35_selected_base_slice_at(
            "down",
            bases.down_all,
            slot.down_byte_offset,
            offsets.down_bytes_per_expert,
        )?);
    }

    Ok(Qwen35SelectedExpertSlices {
        gate_weights,
        up_weights,
        down_weights,
    })
}

fn qwen35_selected_base_slot_offsets(
    bases: &Qwen35SelectedExpertBases<'_>,
    expert_ids: &[u32],
) -> Result<Qwen35SelectedBaseSlotOffsets, String> {
    validate_qwen35_selected_base_len(
        "gate",
        bases.gate_all,
        bases.gate_bytes_per_expert,
        bases.n_expert,
    )?;
    validate_qwen35_selected_base_len(
        "up",
        bases.up_all,
        bases.up_bytes_per_expert,
        bases.n_expert,
    )?;
    validate_qwen35_selected_base_len(
        "down",
        bases.down_all,
        bases.down_bytes_per_expert,
        bases.n_expert,
    )?;

    let mut slots = Vec::with_capacity(expert_ids.len());
    for (slot_idx, &expert_id) in expert_ids.iter().enumerate() {
        let expert = expert_id as usize;
        if expert >= bases.n_expert {
            return Err(format!(
                "expert id out of range: slot={slot_idx} expert_id={expert_id} n_expert={}",
                bases.n_expert
            ));
        }
        slots.push(Qwen35SelectedBaseSlotOffset {
            expert_id,
            gate_byte_offset: qwen35_selected_base_byte_offset(
                "gate",
                bases.gate_bytes_per_expert,
                expert,
            )?,
            up_byte_offset: qwen35_selected_base_byte_offset(
                "up",
                bases.up_bytes_per_expert,
                expert,
            )?,
            down_byte_offset: qwen35_selected_base_byte_offset(
                "down",
                bases.down_bytes_per_expert,
                expert,
            )?,
        });
    }

    Ok(Qwen35SelectedBaseSlotOffsets {
        slots,
        gate_bytes_per_expert: bases.gate_bytes_per_expert,
        up_bytes_per_expert: bases.up_bytes_per_expert,
        down_bytes_per_expert: bases.down_bytes_per_expert,
    })
}

#[allow(dead_code)]
pub(super) fn qwen35_selected_base_temp_slab_slot_ptr_plan(
    offsets: &Qwen35SelectedBaseSlotOffsets,
    slab_base: u64,
) -> Result<Qwen35SelectedBaseTempSlabSlotPtrPlan, String> {
    let mut ptrs_by_key = HashMap::new();
    let mut uploads = Vec::new();
    let mut slab_bytes = 0usize;

    let mut assign_ptr = |role: Qwen35SelectedBaseWeightRole,
                          src_byte_offset: usize,
                          bytes: usize|
     -> Result<u64, String> {
        if bytes == 0 {
            return Err("Qwen35 selected-base temp slab upload bytes must be non-zero".to_string());
        }
        let key = Qwen35SelectedBaseTempSlabKey {
            role,
            src_byte_offset,
            bytes,
        };
        if let Some(&ptr) = ptrs_by_key.get(&key) {
            return Ok(ptr);
        }
        let slab_byte_offset = slab_bytes;
        let ptr = slab_base
            .checked_add(u64::try_from(slab_byte_offset).map_err(|_| {
                format!(
                    "Qwen35 selected-base temp slab offset exceeds u64: {slab_byte_offset}"
                )
            })?)
            .ok_or_else(|| {
                format!(
                    "Qwen35 selected-base temp slab pointer overflows: base={slab_base} offset={slab_byte_offset}"
                )
            })?;
        slab_bytes = slab_bytes.checked_add(bytes).ok_or_else(|| {
            format!(
                "Qwen35 selected-base temp slab byte size overflows: offset={slab_byte_offset} bytes={bytes}"
            )
        })?;
        ptrs_by_key.insert(key, ptr);
        uploads.push(Qwen35SelectedBaseTempSlabUpload {
            role,
            src_byte_offset,
            slab_byte_offset,
            bytes,
        });
        Ok(ptr)
    };

    let mut gate_ptrs = Vec::with_capacity(offsets.slots.len());
    for slot in &offsets.slots {
        gate_ptrs.push(assign_ptr(
            Qwen35SelectedBaseWeightRole::Gate,
            slot.gate_byte_offset,
            offsets.gate_bytes_per_expert,
        )?);
    }

    let mut up_ptrs = Vec::with_capacity(offsets.slots.len());
    for slot in &offsets.slots {
        up_ptrs.push(assign_ptr(
            Qwen35SelectedBaseWeightRole::Up,
            slot.up_byte_offset,
            offsets.up_bytes_per_expert,
        )?);
    }

    let mut down_ptrs = Vec::with_capacity(offsets.slots.len());
    for slot in &offsets.slots {
        down_ptrs.push(assign_ptr(
            Qwen35SelectedBaseWeightRole::Down,
            slot.down_byte_offset,
            offsets.down_bytes_per_expert,
        )?);
    }
    let uploads = qwen35_coalesce_selected_base_temp_slab_uploads(uploads)?;

    Ok(Qwen35SelectedBaseTempSlabSlotPtrPlan {
        gate_ptrs,
        up_ptrs,
        down_ptrs,
        uploads,
        slab_bytes,
    })
}

#[allow(dead_code)]
pub(super) fn qwen35_selected_base_temp_slab_device_ptr_plan(
    offsets: &Qwen35SelectedBaseSlotOffsets,
    n_expert: usize,
    slab_base: u64,
) -> Result<Qwen35SelectedBaseTempSlabDevicePtrPlan, String> {
    let range_upload = qwen35_selected_base_range_upload_enabled();
    let mut seen = vec![false; n_expert];
    let mut expert_slab_indices = vec![u32::MAX; n_expert];
    let mut unique_experts = Vec::new();
    for slot in &offsets.slots {
        let expert = slot.expert_id as usize;
        if expert >= n_expert {
            return Err(format!(
                "Qwen35 selected-base device slot pointer expert id out of range: got {expert}, n_expert={n_expert}"
            ));
        }
        if !seen[expert] {
            seen[expert] = true;
            unique_experts.push(*slot);
        }
    }
    if range_upload {
        unique_experts.sort_by_key(|slot| slot.expert_id);
    }
    for (compact, slot) in unique_experts.iter().enumerate() {
        expert_slab_indices[slot.expert_id as usize] = u32::try_from(compact).map_err(|_| {
            format!(
                "Qwen35 selected-base device slot pointer compact expert count exceeds u32: {compact}"
            )
        })?;
    }

    let gate_upload_bytes = unique_experts
        .len()
        .checked_mul(offsets.gate_bytes_per_expert)
        .ok_or_else(|| {
            format!(
                "Qwen35 selected-base device slot pointer gate slab bytes overflow: experts={} bytes={}",
                unique_experts.len(),
                offsets.gate_bytes_per_expert
            )
        })?;
    let up_upload_bytes = unique_experts
        .len()
        .checked_mul(offsets.up_bytes_per_expert)
        .ok_or_else(|| {
            format!(
                "Qwen35 selected-base device slot pointer up slab bytes overflow: experts={} bytes={}",
                unique_experts.len(),
                offsets.up_bytes_per_expert
            )
        })?;
    let down_upload_bytes = unique_experts
        .len()
        .checked_mul(offsets.down_bytes_per_expert)
        .ok_or_else(|| {
            format!(
                "Qwen35 selected-base device slot pointer down slab bytes overflow: experts={} bytes={}",
                unique_experts.len(),
                offsets.down_bytes_per_expert
            )
        })?;
    let up_base = slab_base
        .checked_add(u64::try_from(gate_upload_bytes).map_err(|_| {
            format!(
                "Qwen35 selected-base device slot pointer gate slab bytes exceed u64: {gate_upload_bytes}"
            )
        })?)
        .ok_or_else(|| {
            format!(
                "Qwen35 selected-base device slot pointer up base overflows: base={slab_base} gate_bytes={gate_upload_bytes}"
            )
        })?;
    let down_base = up_base
        .checked_add(u64::try_from(up_upload_bytes).map_err(|_| {
            format!(
                "Qwen35 selected-base device slot pointer up slab bytes exceed u64: {up_upload_bytes}"
            )
        })?)
        .ok_or_else(|| {
            format!(
                "Qwen35 selected-base device slot pointer down base overflows: up_base={up_base} up_bytes={up_upload_bytes}"
            )
        })?;
    let slab_bytes = gate_upload_bytes
        .checked_add(up_upload_bytes)
        .and_then(|bytes| bytes.checked_add(down_upload_bytes))
        .ok_or_else(|| {
            format!(
                "Qwen35 selected-base device slot pointer slab bytes overflow: gate={gate_upload_bytes} up={up_upload_bytes} down={down_upload_bytes}"
            )
        })?;

    let mut uploads = Vec::with_capacity(unique_experts.len() * 3);
    for (compact, slot) in unique_experts.iter().enumerate() {
        uploads.push(Qwen35SelectedBaseTempSlabUpload {
            role: Qwen35SelectedBaseWeightRole::Gate,
            src_byte_offset: slot.gate_byte_offset,
            slab_byte_offset: compact * offsets.gate_bytes_per_expert,
            bytes: offsets.gate_bytes_per_expert,
        });
    }
    for (compact, slot) in unique_experts.iter().enumerate() {
        uploads.push(Qwen35SelectedBaseTempSlabUpload {
            role: Qwen35SelectedBaseWeightRole::Up,
            src_byte_offset: slot.up_byte_offset,
            slab_byte_offset: gate_upload_bytes + compact * offsets.up_bytes_per_expert,
            bytes: offsets.up_bytes_per_expert,
        });
    }
    for (compact, slot) in unique_experts.iter().enumerate() {
        uploads.push(Qwen35SelectedBaseTempSlabUpload {
            role: Qwen35SelectedBaseWeightRole::Down,
            src_byte_offset: slot.down_byte_offset,
            slab_byte_offset: gate_upload_bytes
                + up_upload_bytes
                + compact * offsets.down_bytes_per_expert,
            bytes: offsets.down_bytes_per_expert,
        });
    }
    let uploads = qwen35_coalesce_selected_base_temp_slab_uploads(uploads)?;

    Ok(Qwen35SelectedBaseTempSlabDevicePtrPlan {
        expert_slab_indices,
        uploads,
        gate_base: slab_base,
        up_base,
        down_base,
        slab_bytes,
    })
}

#[allow(dead_code)]
pub(super) fn qwen35_selected_base_mixed_resident_temp_plan(
    offsets: &Qwen35SelectedBaseSlotOffsets,
    n_expert: usize,
    resident_roles: &HashSet<Qwen35SelectedBaseResidentRole>,
) -> Result<Qwen35SelectedBaseMixedResidentTempPlan, String> {
    let mut seen = vec![false; n_expert];
    let mut unique_experts = Vec::new();
    for slot in &offsets.slots {
        let expert = slot.expert_id as usize;
        if expert >= n_expert {
            return Err(format!(
                "Qwen35 selected-base mixed resident expert id out of range: got {expert}, n_expert={n_expert}"
            ));
        }
        if !seen[expert] {
            seen[expert] = true;
            unique_experts.push(*slot);
        }
    }

    let mut gate_sources = vec![None; n_expert];
    let mut up_sources = vec![None; n_expert];
    let mut down_sources = vec![None; n_expert];
    let mut uploads = Vec::new();
    let mut slab_bytes = 0usize;
    let mut resident_upload_bytes_saved = 0usize;

    let mut assign_role = |role: Qwen35SelectedBaseWeightRole,
                           slot: Qwen35SelectedBaseSlotOffset,
                           bytes: usize,
                           src_byte_offset: usize|
     -> Result<Qwen35SelectedBaseMixedWeightSource, String> {
        let expert_id = slot.expert_id as usize;
        if resident_roles.contains(&Qwen35SelectedBaseResidentRole { role, expert_id }) {
            resident_upload_bytes_saved = resident_upload_bytes_saved
                .checked_add(bytes)
                .ok_or_else(|| {
                    "Qwen35 selected-base mixed resident saved bytes overflow".to_string()
                })?;
            return Ok(Qwen35SelectedBaseMixedWeightSource::Resident);
        }

        let slab_byte_offset = slab_bytes;
        slab_bytes = slab_bytes.checked_add(bytes).ok_or_else(|| {
            format!(
                "Qwen35 selected-base mixed resident temp slab bytes overflow: current={slab_byte_offset} add={bytes}"
            )
        })?;
        uploads.push(Qwen35SelectedBaseTempSlabUpload {
            role,
            src_byte_offset,
            slab_byte_offset,
            bytes,
        });
        Ok(Qwen35SelectedBaseMixedWeightSource::Temp { slab_byte_offset })
    };

    for slot in &unique_experts {
        gate_sources[slot.expert_id as usize] = Some(assign_role(
            Qwen35SelectedBaseWeightRole::Gate,
            *slot,
            offsets.gate_bytes_per_expert,
            slot.gate_byte_offset,
        )?);
    }
    for slot in &unique_experts {
        up_sources[slot.expert_id as usize] = Some(assign_role(
            Qwen35SelectedBaseWeightRole::Up,
            *slot,
            offsets.up_bytes_per_expert,
            slot.up_byte_offset,
        )?);
    }
    for slot in &unique_experts {
        down_sources[slot.expert_id as usize] = Some(assign_role(
            Qwen35SelectedBaseWeightRole::Down,
            *slot,
            offsets.down_bytes_per_expert,
            slot.down_byte_offset,
        )?);
    }

    let mut expert_sources = vec![None; n_expert];
    for expert in 0..n_expert {
        if !seen[expert] {
            continue;
        }
        let gate = gate_sources[expert].ok_or_else(|| {
            format!("Qwen35 selected-base mixed resident missing gate source: expert={expert}")
        })?;
        let up = up_sources[expert].ok_or_else(|| {
            format!("Qwen35 selected-base mixed resident missing up source: expert={expert}")
        })?;
        let down = down_sources[expert].ok_or_else(|| {
            format!("Qwen35 selected-base mixed resident missing down source: expert={expert}")
        })?;
        expert_sources[expert] = Some(Qwen35SelectedBaseMixedExpertSources { gate, up, down });
    }

    Ok(Qwen35SelectedBaseMixedResidentTempPlan {
        expert_sources,
        uploads,
        slab_bytes,
        selected_experts: unique_experts.len(),
        resident_upload_bytes_saved,
    })
}

#[cfg(test)]
pub(super) fn qwen35_selected_sparse_boundary_stats_from_device_ptr_plan(
    slots: usize,
    plan: &Qwen35SelectedBaseTempSlabDevicePtrPlan,
    route_from_device: bool,
) -> Qwen35SelectedSparseBoundaryStats {
    let unique_experts = plan
        .expert_slab_indices
        .iter()
        .filter(|&&index| index != u32::MAX)
        .count();
    Qwen35SelectedSparseBoundaryStats {
        slots,
        unique_experts,
        selected_upload_calls: plan.uploads.len(),
        selected_upload_bytes: plan.uploads.iter().map(|upload| upload.bytes).sum(),
        route_h2d_bytes: if route_from_device {
            0
        } else {
            slots * std::mem::size_of::<f32>()
        },
        token_h2d_bytes: if route_from_device {
            0
        } else {
            slots * std::mem::size_of::<u32>()
        },
        device_slot_h2d_bytes: (slots + plan.expert_slab_indices.len())
            * std::mem::size_of::<u32>(),
        group_meta_h2d_bytes: 0,
        descriptor_h2d_calls: 2 + if route_from_device { 0 } else { 2 },
        group_meta_h2d_calls: 0,
        slot_ptr_build_launches: 0,
        zero_launches: 0,
        gate_up_launches: 0,
        silu_launches: 0,
        down_launches: 0,
    }
}

pub(super) fn qwen35_selected_sparse_boundary_stats_from_device_slots(
    slots: usize,
    device: &PreparedQwen35DeviceSlotPtrs,
    route_from_device: bool,
) -> Qwen35SelectedSparseBoundaryStats {
    let unique_experts = device
        .expert_slab_indices
        .iter()
        .filter(|&&index| index != u32::MAX)
        .count();
    Qwen35SelectedSparseBoundaryStats {
        slots,
        unique_experts,
        selected_upload_calls: device.selected_upload_calls,
        selected_upload_bytes: device.selected_upload_bytes,
        route_h2d_bytes: if route_from_device {
            0
        } else {
            slots * std::mem::size_of::<f32>()
        },
        token_h2d_bytes: if route_from_device {
            0
        } else {
            slots * std::mem::size_of::<u32>()
        },
        device_slot_h2d_bytes: (slots + device.expert_slab_indices.len())
            * std::mem::size_of::<u32>(),
        group_meta_h2d_bytes: 0,
        descriptor_h2d_calls: 2 + if route_from_device { 0 } else { 2 },
        group_meta_h2d_calls: 0,
        slot_ptr_build_launches: 0,
        zero_launches: 0,
        gate_up_launches: 0,
        silu_launches: 0,
        down_launches: 0,
    }
}

fn validate_qwen35_selected_base_len(
    name: &str,
    all_weights: &[u8],
    bytes_per_expert: usize,
    n_expert: usize,
) -> Result<(), String> {
    let expected = bytes_per_expert
        .checked_mul(n_expert)
        .ok_or_else(|| format!("{name} selected base byte size overflows usize"))?;
    if all_weights.len() != expected {
        return Err(format!(
            "{name} selected base length mismatch: len={} expected={expected} bytes_per_expert={bytes_per_expert} n_expert={n_expert}",
            all_weights.len()
        ));
    }
    Ok(())
}

fn qwen35_selected_base_byte_offset(
    name: &str,
    bytes_per_expert: usize,
    expert: usize,
) -> Result<usize, String> {
    expert
        .checked_mul(bytes_per_expert)
        .ok_or_else(|| format!("{name} selected expert byte offset overflows usize"))
}

fn qwen35_selected_base_slice_at<'a>(
    name: &str,
    all_weights: &'a [u8],
    byte_offset: usize,
    bytes_per_expert: usize,
) -> Result<&'a [u8], String> {
    let end = byte_offset
        .checked_add(bytes_per_expert)
        .ok_or_else(|| format!("{name} selected expert byte end overflows usize"))?;
    all_weights.get(byte_offset..end).ok_or_else(|| {
        format!(
            "{name} selected expert slice out of range: start={byte_offset} end={end} len={}",
            all_weights.len()
        )
    })
}

pub(super) fn qwen35_decode_resident_batch_fits(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    resident_q4k_limit: usize,
) -> bool {
    qwen35_decode_resident_batch_bytes(gate_weights, up_weights, down_weights) <= resident_q4k_limit
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35DecodeSelectedSlotPtrPlan {
    ResidentBatch,
    MixedResidentTemp,
    TempSlab,
}

pub(super) fn qwen35_decode_selected_slot_ptr_plan(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    resident_q4k_limit: usize,
    resident_batch_enabled: bool,
    has_resident_slots: bool,
) -> Qwen35DecodeSelectedSlotPtrPlan {
    if resident_batch_enabled
        && qwen35_decode_resident_batch_fits(
            gate_weights,
            up_weights,
            down_weights,
            resident_q4k_limit,
        )
    {
        Qwen35DecodeSelectedSlotPtrPlan::ResidentBatch
    } else if has_resident_slots {
        Qwen35DecodeSelectedSlotPtrPlan::MixedResidentTemp
    } else {
        Qwen35DecodeSelectedSlotPtrPlan::TempSlab
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Qwen35TempUploadStream {
    Main,
    Copy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Qwen35MixedTempSlotUploadPlanEntry {
    pub(super) key: (usize, usize),
    pub(super) offset: usize,
    pub(super) bytes: usize,
    pub(super) stream: Qwen35TempUploadStream,
}

pub(super) fn qwen35_mixed_temp_slot_upload_plan(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    resident_keys: &HashSet<(usize, usize)>,
    overlap_down_copy: bool,
) -> (Vec<Qwen35MixedTempSlotUploadPlanEntry>, usize) {
    let mut gate_up_keys = HashSet::new();
    if overlap_down_copy {
        for &weights in gate_weights.iter().chain(up_weights.iter()) {
            gate_up_keys.insert(q4k_resident_key(weights));
        }
    }

    let mut seen = HashSet::new();
    let mut entries = Vec::new();
    let mut offset = 0usize;
    for &weights in gate_weights
        .iter()
        .chain(up_weights.iter())
        .chain(down_weights.iter())
    {
        let key = q4k_resident_key(weights);
        if resident_keys.contains(&key) || !seen.insert(key) {
            continue;
        }
        let stream = if overlap_down_copy && !gate_up_keys.contains(&key) {
            Qwen35TempUploadStream::Copy
        } else {
            Qwen35TempUploadStream::Main
        };
        entries.push(Qwen35MixedTempSlotUploadPlanEntry {
            key,
            offset,
            bytes: weights.len(),
            stream,
        });
        offset = offset.saturating_add(weights.len());
    }
    (entries, offset)
}

pub(super) fn q4k_resident_hit_count(
    cache: &HashMap<(usize, usize), ResidentQ4k>,
    slots: &[&[u8]],
) -> usize {
    slots
        .iter()
        .filter(|weights| cache.contains_key(&q4k_resident_key(weights)))
        .count()
}

pub(super) fn device_residency_default_reserve_mib(
    total_mib: usize,
    mtp_device_verify: bool,
) -> usize {
    let base_mib =
        rnb_memory::default_device_dynamic_reserve_bytes(total_mib.saturating_mul(1024 * 1024))
            / (1024 * 1024);
    if mtp_device_verify {
        base_mib.saturating_add(q4k_resident_mtp_workspace_reserve_mib(total_mib))
    } else {
        base_mib
    }
}

fn q4k_resident_mtp_workspace_reserve_mib(total_mib: usize) -> usize {
    align_up(total_mib / 4, 256).clamp(512, 4096)
}

pub(super) fn q4k_resident_mtp_slot_cache_cap_mib(total_mib: usize) -> usize {
    align_up(total_mib.saturating_mul(3) / 10, 256).clamp(512, 4096)
}

fn q4k_resident_min_cache_mib(total_mib: usize) -> usize {
    align_up(total_mib / 16, 128).clamp(256, 1024)
}

pub(super) fn q4k_resident_target_decode_cache_cap_mib(total_mib: usize) -> usize {
    align_up(total_mib.saturating_mul(3) / 4, 256).clamp(4096, 12 * 1024)
}

pub(super) fn q4k_resident_nemotron_decode_cache_cap_mib(total_mib: usize) -> usize {
    if total_mib <= 8 * 1024 {
        align_up(total_mib / 2, 256).clamp(1024, 4096)
    } else {
        align_up(total_mib.saturating_mul(3) / 5, 256).clamp(1024, 6144)
    }
}

pub(super) fn q4k_resident_auto_cache_cap_mib(
    total_mib: usize,
    mtp_device_verify: bool,
) -> Option<usize> {
    if mtp_device_verify || total_mib <= 12 * 1024 {
        Some(q4k_resident_mtp_slot_cache_cap_mib(total_mib))
    } else {
        None
    }
}

pub(super) fn device_residency_configured_reserve_mib(
    total_mib: usize,
    mtp_device_verify: bool,
) -> Result<usize, String> {
    let mut reserve_mib = std::env::var("RNB_CUDA_Q4K_CACHE_RESERVE_MB")
        .ok()
        .map(|raw| {
            raw.trim()
                .parse::<usize>()
                .map_err(|e| format!("RNB_CUDA_Q4K_CACHE_RESERVE_MB must be integer MiB: {e}"))
        })
        .transpose()?
        .unwrap_or_else(|| device_residency_default_reserve_mib(total_mib, mtp_device_verify));
    if std::env::var("RNB_CUDA_PREFILL_MOE").is_ok() {
        reserve_mib = reserve_mib.saturating_add(1024);
    }
    Ok(reserve_mib)
}

pub(super) fn qwen35_decode_hot_resident_default_budget_bytes(resident_q4k_limit: usize) -> usize {
    (resident_q4k_limit / 320).clamp(1024 * 1024, 32 * 1024 * 1024)
}

pub(super) fn qwen35_decode_all_resident_touch_hits_enabled() -> bool {
    std::env::var("RNB_CUDA_QWEN35_DECODE_ALL_RESIDENT_SKIP_TOUCH")
        .map(|value| {
            let value = value.to_ascii_lowercase();
            matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

pub(super) fn cuda_mem_alloc_oom(err: &str) -> bool {
    err.contains("cuMemAlloc") && err.contains("CUDA error 2")
}

pub(super) fn cuda_offload_on_oom_enabled() -> bool {
    std::env::var("RNB_CUDA_OFFLOAD_ON_OOM")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
        })
        .unwrap_or(true)
}

pub(super) fn mtp_device_verify_env_enabled() -> bool {
    std::env::var("RNB_MTP_DEVICE_VERIFY")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

pub(super) fn q4k_residency_candidates(
    unique: &HashMap<(usize, usize), &[u8]>,
    slot_counts: &HashMap<(usize, usize), u32>,
) -> Vec<ResidencyCandidate> {
    let mut candidates = unique
        .iter()
        .map(|(&(ptr, len), weights)| {
            let reuse = slot_counts.get(&(ptr, len)).copied().unwrap_or(1);
            ResidencyCandidate::new(format!("{ptr:x}:{len}"), weights.len() as u64, reuse)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.id().cmp(right.id()));
    candidates
}

pub(super) fn prefill_residency_trace_budget_bytes(default_bytes: usize) -> usize {
    let Some(raw) = std::env::var("RNB_CUDA_RESIDENCY_TRACE_BUDGET_MB").ok() else {
        return default_bytes;
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
        return default_bytes;
    }
    match trimmed.parse::<usize>() {
        Ok(mib) => mib.saturating_mul(1024 * 1024),
        Err(_) => default_bytes,
    }
}

pub(super) fn qwen35_moe_layer_key(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Qwen35MoeLayerKey {
    Qwen35MoeLayerKey {
        gate_ptr: gate_all.as_ptr() as usize,
        gate_len: gate_all.len(),
        up_ptr: up_all.as_ptr() as usize,
        up_len: up_all.len(),
        down_ptr: down_all.as_ptr() as usize,
        down_len: down_all.len(),
        down_quant,
        n_ff,
        n_embd,
    }
}

pub(super) fn qwen35_moe_layer_effective_limit(
    configured_limit: usize,
    incoming_bytes: usize,
    qwen_cache_enabled: bool,
) -> usize {
    if qwen_cache_enabled {
        configured_limit
    } else {
        configured_limit.min(incoming_bytes)
    }
}

pub(super) fn nemotron_q5_layer_key(
    up_all: &[u8],
    down_all: &[u8],
    n_ff: usize,
    n_embd: usize,
) -> Qwen35MoeLayerKey {
    nemotron_layer_key(up_all, down_all, n_ff, n_embd, 51)
}

pub(super) fn nemotron_q5_q8_layer_key(
    up_all: &[u8],
    down_all: &[u8],
    n_ff: usize,
    n_embd: usize,
) -> Qwen35MoeLayerKey {
    nemotron_layer_key(up_all, down_all, n_ff, n_embd, 80)
}

pub(super) fn nemotron_layer_key(
    up_all: &[u8],
    down_all: &[u8],
    n_ff: usize,
    n_embd: usize,
    down_quant: u32,
) -> Qwen35MoeLayerKey {
    Qwen35MoeLayerKey {
        gate_ptr: 0,
        gate_len: 0,
        up_ptr: up_all.as_ptr() as usize,
        up_len: up_all.len(),
        down_ptr: down_all.as_ptr() as usize,
        down_len: down_all.len(),
        down_quant,
        n_ff,
        n_embd,
    }
}

pub(super) fn validate_qwen35_moe_layer_weights(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Result<(), String> {
    validate_qwen35_sparse_full_layer_batch(
        gate_all,
        up_all,
        down_all,
        &[],
        &[],
        &[],
        0,
        down_quant,
        n_ff,
        n_embd,
        &[],
    )
}

pub(super) fn validate_nemotron_q5_layer_weights(
    up_all: &[u8],
    down_all: &[u8],
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
) -> Result<(), String> {
    if n_expert == 0 || n_embd % 32 != 0 || n_ff % 32 != 0 {
        return Err(format!(
            "Nemotron full-layer sparse Q5 dims must be non-zero and divisible by 32, got n_expert={n_expert} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let up_expert_bytes = n_ff * (n_embd / 32) * 22;
    let down_expert_bytes = n_embd * (n_ff / 32) * 24;
    if up_all.len() != n_expert * up_expert_bytes {
        return Err(format!(
            "Nemotron full-layer Q5_0 up byte mismatch: got {}, expected {}",
            up_all.len(),
            n_expert * up_expert_bytes
        ));
    }
    if down_all.len() != n_expert * down_expert_bytes {
        return Err(format!(
            "Nemotron full-layer Q5_1 down byte mismatch: got {}, expected {}",
            down_all.len(),
            n_expert * down_expert_bytes
        ));
    }
    Ok(())
}

pub(super) fn validate_nemotron_q5_q8_layer_weights(
    up_all: &[u8],
    down_all: &[u8],
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
) -> Result<(), String> {
    if n_expert == 0 || n_embd % 32 != 0 || n_ff % 32 != 0 {
        return Err(format!(
            "Nemotron full-layer sparse Q5/Q8 dims must be non-zero and divisible by 32, got n_expert={n_expert} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let up_expert_bytes = n_ff * (n_embd / 32) * 22;
    let down_expert_bytes = n_embd * (n_ff / 32) * 34;
    if up_all.len() != n_expert * up_expert_bytes {
        return Err(format!(
            "Nemotron full-layer Q5_0 up byte mismatch: got {}, expected {}",
            up_all.len(),
            n_expert * up_expert_bytes
        ));
    }
    if down_all.len() != n_expert * down_expert_bytes {
        return Err(format!(
            "Nemotron full-layer Q8_0 down byte mismatch: got {}, expected {}",
            down_all.len(),
            n_expert * down_expert_bytes
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_nemotron_q5_sparse_full_layer_batch(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<(), String> {
    let slots = expert_ids.len();
    if route_weights.len() != slots || token_ids.len() != slots {
        return Err("Nemotron full-layer sparse Q5 batch length mismatch".to_string());
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron full-layer sparse Q5 input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    validate_nemotron_q5_layer_weights(up_all, down_all, n_expert, n_ff, n_embd)?;
    if expert_ids.iter().any(|&expert| expert as usize >= n_expert) {
        return Err("Nemotron full-layer sparse Q5 expert id out of range".to_string());
    }
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err("Nemotron full-layer sparse Q5 token id out of range".to_string());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_nemotron_q5_q8_sparse_full_layer_batch(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<(), String> {
    let slots = expert_ids.len();
    if route_weights.len() != slots || token_ids.len() != slots {
        return Err("Nemotron full-layer sparse Q5/Q8 batch length mismatch".to_string());
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron full-layer sparse Q5/Q8 input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    validate_nemotron_q5_q8_layer_weights(up_all, down_all, n_expert, n_ff, n_embd)?;
    if expert_ids.iter().any(|&expert| expert as usize >= n_expert) {
        return Err("Nemotron full-layer sparse Q5/Q8 expert id out of range".to_string());
    }
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err("Nemotron full-layer sparse Q5/Q8 token id out of range".to_string());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_qwen35_sparse_token_batch(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<(), String> {
    let slots = gate_weights.len();
    if up_weights.len() != slots
        || down_weights.len() != slots
        || route_weights.len() != slots
        || token_ids.len() != slots
    {
        return Err("Qwen35 token-batch sparse expert length mismatch".to_string());
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Qwen35 token-batch input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    if token_ids.iter().any(|&t| t as usize >= token_count) {
        return Err("Qwen35 token-batch token id out of range".to_string());
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 token-batch dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let gate_row_bytes = (n_embd / 256) * 144;
    let down_row_bytes = match down_quant {
        12 => (n_ff / 256) * 144,
        13 => (n_ff / 256) * 176,
        14 => (n_ff / 256) * 210,
        other => return Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
    };
    for (i, weights) in gate_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 token-batch gate[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in up_weights.iter().enumerate() {
        if weights.len() != n_ff * gate_row_bytes {
            return Err(format!(
                "Qwen35 token-batch up[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_ff * gate_row_bytes
            ));
        }
    }
    for (i, weights) in down_weights.iter().enumerate() {
        if weights.len() != n_embd * down_row_bytes {
            return Err(format!(
                "Qwen35 token-batch down[{i}] byte mismatch: got {}, expected {}",
                weights.len(),
                n_embd * down_row_bytes
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_qwen35_sparse_full_layer_batch(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<(), String> {
    let slots = expert_ids.len();
    if route_weights.len() != slots || token_ids.len() != slots {
        return Err(format!(
            "Qwen35 full-layer token-batch length mismatch: expert={} route={} token={}",
            slots,
            route_weights.len(),
            token_ids.len()
        ));
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Qwen35 full-layer input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "Qwen35 full-layer dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    if token_ids.iter().any(|&t| t as usize >= token_count) {
        return Err("Qwen35 full-layer token id out of range".to_string());
    }
    let gate_row_bytes = (n_embd / 256) * 144;
    let gate_expert_bytes = n_ff * gate_row_bytes;
    let down_row_bytes = match down_quant {
        12 => (n_ff / 256) * 144,
        13 => (n_ff / 256) * 176,
        14 => (n_ff / 256) * 210,
        other => return Err(format!("unsupported Qwen35 CUDA down quant code {other}")),
    };
    let down_expert_bytes = n_embd * down_row_bytes;
    if gate_expert_bytes == 0 || down_expert_bytes == 0 {
        return Err("Qwen35 full-layer expert byte size must be non-zero".to_string());
    }
    if !gate_all.len().is_multiple_of(gate_expert_bytes)
        || up_all.len() != gate_all.len()
        || !down_all.len().is_multiple_of(down_expert_bytes)
    {
        return Err(format!(
            "Qwen35 full-layer weight shape mismatch: gate={} up={} down={} gate_per={} down_per={}",
            gate_all.len(),
            up_all.len(),
            down_all.len(),
            gate_expert_bytes,
            down_expert_bytes
        ));
    }
    let n_expert = gate_all.len() / gate_expert_bytes;
    if down_all.len() / down_expert_bytes != n_expert {
        return Err(format!(
            "Qwen35 full-layer expert count mismatch: gate/up={} down={}",
            n_expert,
            down_all.len() / down_expert_bytes
        ));
    }
    if expert_ids.iter().any(|&expert| expert as usize >= n_expert) {
        return Err("Qwen35 full-layer expert id out of range".to_string());
    }
    Ok(())
}

pub(super) fn build_group_meta(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    max_group: usize,
) -> Vec<u32> {
    debug_assert!(max_group > 0);
    let mut meta = Vec::new();
    let mut slot = 0usize;
    while slot < gate_weights.len() {
        let gate_key = q4k_resident_key(gate_weights[slot]);
        let up_key = q4k_resident_key(up_weights[slot]);
        let mut len = 1usize;
        while slot + len < gate_weights.len()
            && len < max_group
            && q4k_resident_key(gate_weights[slot + len]) == gate_key
            && q4k_resident_key(up_weights[slot + len]) == up_key
        {
            len += 1;
        }
        meta.push(slot as u32);
        meta.push(len as u32);
        slot += len;
    }
    meta
}

pub(super) fn build_group_meta_from_ids(expert_ids: &[u32], max_group: usize) -> Vec<u32> {
    debug_assert!(max_group > 0);
    let mut meta = Vec::new();
    let mut slot = 0usize;
    while slot < expert_ids.len() {
        let expert = expert_ids[slot];
        let mut len = 1usize;
        while slot + len < expert_ids.len() && len < max_group && expert_ids[slot + len] == expert {
            len += 1;
        }
        meta.push(slot as u32);
        meta.push(len as u32);
        slot += len;
    }
    meta
}

pub(super) fn qwen35_selected_base_group_meta_from_offsets(
    offsets: &Qwen35SelectedBaseSlotOffsets,
    max_group: usize,
) -> Vec<u32> {
    debug_assert!(max_group > 0);
    let mut meta = Vec::new();
    let mut slot = 0usize;
    while slot < offsets.slots.len() {
        let gate_offset = offsets.slots[slot].gate_byte_offset;
        let up_offset = offsets.slots[slot].up_byte_offset;
        let mut len = 1usize;
        while slot + len < offsets.slots.len()
            && len < max_group
            && offsets.slots[slot + len].gate_byte_offset == gate_offset
            && offsets.slots[slot + len].up_byte_offset == up_offset
        {
            len += 1;
        }
        meta.push(slot as u32);
        meta.push(len as u32);
        slot += len;
    }
    meta
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Qwen35GroupShapeSummary {
    pub(super) groups: usize,
    pub(super) slots: usize,
    pub(super) max_len: usize,
    pub(super) len_hist: [usize; 9],
    pub(super) overflow_groups: usize,
}

pub(super) fn qwen35_group_shape_summary(
    group_meta: &[u32],
) -> Result<Qwen35GroupShapeSummary, String> {
    if group_meta.len() % 2 != 0 {
        return Err("group meta must contain start/len pairs".to_string());
    }

    let mut summary = Qwen35GroupShapeSummary {
        groups: 0,
        slots: 0,
        max_len: 0,
        len_hist: [0; 9],
        overflow_groups: 0,
    };

    for pair in group_meta.chunks_exact(2) {
        let len = pair[1] as usize;
        if len == 0 {
            return Err("group meta length must be non-zero".to_string());
        }
        summary.groups += 1;
        summary.slots = summary.slots.saturating_add(len);
        summary.max_len = summary.max_len.max(len);
        if len < summary.len_hist.len() {
            summary.len_hist[len] += 1;
        } else {
            summary.overflow_groups += 1;
        }
    }

    Ok(summary)
}

#[allow(dead_code)]
pub(super) fn qwen35_expert_run_batched_down_plan(
    expert_ids: &[u32],
    max_tile_slots: usize,
) -> Result<Vec<Qwen35ExpertRun>, String> {
    if max_tile_slots == 0 {
        return Err("max tile slots must be non-zero".to_string());
    }

    let mut runs = Vec::new();
    let mut slot = 0usize;
    while slot < expert_ids.len() {
        let expert_id = expert_ids[slot];
        let mut len = 1usize;
        while slot + len < expert_ids.len() && expert_ids[slot + len] == expert_id {
            len += 1;
        }
        runs.push(Qwen35ExpertRun {
            expert_id,
            slot_start: slot,
            len,
            full_tiles: len / max_tile_slots,
            tail: len % max_tile_slots,
        });
        slot += len;
    }

    Ok(runs)
}

#[allow(dead_code)]
pub(super) fn qwen35_expert_run_tile_meta(
    expert_ids: &[u32],
    max_tile_slots: usize,
) -> Result<Vec<u32>, String> {
    let runs = qwen35_expert_run_batched_down_plan(expert_ids, max_tile_slots)?;
    let mut meta = Vec::new();
    for run in runs {
        let mut slot_start = run.slot_start;
        for _ in 0..run.full_tiles {
            meta.push(slot_start as u32);
            meta.push(max_tile_slots as u32);
            slot_start += max_tile_slots;
        }
        if run.tail > 0 {
            meta.push(slot_start as u32);
            meta.push(run.tail as u32);
        }
    }
    Ok(meta)
}

#[allow(dead_code)]
pub(super) fn qwen35_expert_run_specialized_tile_meta(
    expert_ids: &[u32],
    max_tile_slots: usize,
) -> Result<Vec<Qwen35ExpertRunSpecializedTile>, String> {
    let runs = qwen35_expert_run_batched_down_plan(expert_ids, max_tile_slots)?;
    let mut meta = Vec::new();
    for run in runs {
        let mut tile_start = run.slot_start;
        for _ in 0..run.full_tiles {
            meta.push(Qwen35ExpertRunSpecializedTile {
                expert_id: run.expert_id,
                run_start: run.slot_start,
                run_len: run.len,
                tile_start,
                tile_len: max_tile_slots,
            });
            tile_start += max_tile_slots;
        }
        if run.tail > 0 {
            meta.push(Qwen35ExpertRunSpecializedTile {
                expert_id: run.expert_id,
                run_start: run.slot_start,
                run_len: run.len,
                tile_start,
                tile_len: run.tail,
            });
        }
    }
    Ok(meta)
}

#[allow(dead_code)]
pub(super) fn qwen35_expert_run_specialized_tile_words(
    expert_ids: &[u32],
    max_tile_slots: usize,
) -> Result<Vec<u32>, String> {
    fn to_u32(name: &str, value: usize) -> Result<u32, String> {
        u32::try_from(value).map_err(|_| format!("{name} does not fit in u32: {value}"))
    }

    let meta = qwen35_expert_run_specialized_tile_meta(expert_ids, max_tile_slots)?;
    let mut words = Vec::with_capacity(meta.len() * 5);
    for tile in meta {
        words.push(tile.expert_id);
        words.push(to_u32("run start", tile.run_start)?);
        words.push(to_u32("run len", tile.run_len)?);
        words.push(to_u32("tile start", tile.tile_start)?);
        words.push(to_u32("tile len", tile.tile_len)?);
    }
    Ok(words)
}

#[allow(dead_code)]
pub(super) fn qwen35_expert_run_down_weight_identity(
    expert_ids: &[u32],
    down_weights: &[&[u8]],
) -> Result<(), String> {
    if down_weights.len() != expert_ids.len() {
        return Err(format!(
            "run-batched down weight length mismatch: got {}, expected {}",
            down_weights.len(),
            expert_ids.len()
        ));
    }

    let runs = qwen35_expert_run_batched_down_plan(expert_ids, usize::MAX)?;
    for run in runs {
        let first = down_weights[run.slot_start];
        for slot in run.slot_start + 1..run.slot_start + run.len {
            let current = down_weights[slot];
            if first.as_ptr() != current.as_ptr() || first.len() != current.len() {
                return Err(format!(
                    "run-batched down weight mismatch inside expert run: expert_id={} run_start={} slot={}",
                    run.expert_id, run.slot_start, slot
                ));
            }
        }
    }

    Ok(())
}

pub(super) fn qwen35_down_token_major_plan(
    token_ids: &[u32],
    token_count: usize,
) -> Result<Qwen35DownTokenMajorPlan, String> {
    let mut counts = vec![0usize; token_count];
    for &token_id in token_ids {
        let token = token_id as usize;
        if token >= token_count {
            return Err(format!(
                "token id out of range: token_id={} token_count={}",
                token_id, token_count
            ));
        }
        counts[token] += 1;
    }

    let mut token_offsets = Vec::with_capacity(token_count + 1);
    token_offsets.push(0);
    for count in counts {
        token_offsets.push(token_offsets.last().copied().unwrap() + count as u32);
    }

    let mut cursor = token_offsets[..token_count].to_vec();
    let mut slot_indices = vec![0u32; token_ids.len()];
    for (slot, &token_id) in token_ids.iter().enumerate() {
        let token = token_id as usize;
        let dst = cursor[token] as usize;
        slot_indices[dst] = slot as u32;
        cursor[token] += 1;
    }

    Ok(Qwen35DownTokenMajorPlan {
        token_offsets,
        slot_indices,
    })
}

pub(super) fn qwen35_group_meta_split_by_len(
    group_meta: &[u32],
    matching_len: u32,
) -> Result<Qwen35GroupMetaLenSplit, String> {
    if matching_len == 0 {
        return Err("group meta matching length must be non-zero".to_string());
    }
    if group_meta.len() % 2 != 0 {
        return Err("group meta must contain start/len pairs".to_string());
    }

    let mut matching = Vec::new();
    let mut other = Vec::new();
    for pair in group_meta.chunks_exact(2) {
        let dst = if pair[1] == matching_len {
            &mut matching
        } else {
            &mut other
        };
        dst.extend_from_slice(pair);
    }

    Ok(Qwen35GroupMetaLenSplit { matching, other })
}

pub(super) fn q4k_resident_cache_limit(api: &CudaApi) -> Result<usize, String> {
    if let Ok(raw) = std::env::var("RNB_CUDA_Q4K_CACHE_MB") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("auto") {
            let mb = trimmed
                .parse::<usize>()
                .map_err(|e| format!("RNB_CUDA_Q4K_CACHE_MB must be integer MiB or auto: {e}"))?;
            return Ok(mb.saturating_mul(1024 * 1024));
        }
    }

    let (free_bytes, total_bytes) = unsafe { api.mem_get_info() }?;
    let total_mib = total_bytes / (1024 * 1024);
    let free_mib = free_bytes / (1024 * 1024);
    let reserve_mib =
        device_residency_configured_reserve_mib(total_mib, mtp_device_verify_env_enabled())?;
    let cap_mib = total_mib.saturating_sub(reserve_mib);
    let available_mib = free_mib.saturating_sub(reserve_mib);
    let min_cache_mib = q4k_resident_min_cache_mib(total_mib);
    let mut mb = if available_mib < min_cache_mib || cap_mib < min_cache_mib {
        0
    } else {
        available_mib.min(cap_mib)
    };
    if mb > 0 {
        if let Some(cap) =
            q4k_resident_auto_cache_cap_mib(total_mib, mtp_device_verify_env_enabled())
        {
            mb = mb.min(cap);
        }
    }
    if mb < min_cache_mib {
        mb = 0;
    }
    if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
        eprintln!(
            "[cuda] q4k resident cache auto: total={}MiB free={}MiB reserve={}MiB limit={}MiB",
            total_mib, free_mib, reserve_mib, mb
        );
    }
    Ok(mb.saturating_mul(1024 * 1024))
}

pub(super) fn q8_f32_cache_limit() -> Result<usize, String> {
    if let Ok(raw) = std::env::var("RNB_CUDA_Q8_F32_CACHE_MB") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("auto") {
            let mb = trimmed.parse::<usize>().map_err(|e| {
                format!("RNB_CUDA_Q8_F32_CACHE_MB must be integer MiB or auto: {e}")
            })?;
            return Ok(mb.saturating_mul(1024 * 1024));
        }
    }
    let mb: usize = 0;
    if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
        eprintln!("[cuda] Q8 F32 cache limit={}MiB", mb);
    }
    Ok(mb.saturating_mul(1024 * 1024))
}

pub(super) fn q4_packed_cache_limit(api: &CudaApi) -> Result<usize, String> {
    if let Ok(raw) = std::env::var("RNB_CUDA_Q4_PACKED_CACHE_MB") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("auto") {
            let mb = trimmed.parse::<usize>().map_err(|e| {
                format!("RNB_CUDA_Q4_PACKED_CACHE_MB must be integer MiB or auto: {e}")
            })?;
            return Ok(mb.saturating_mul(1024 * 1024));
        }
    }

    let (free_bytes, total_bytes) = unsafe { api.mem_get_info() }?;
    let total_mib = total_bytes / (1024 * 1024);
    let free_mib = free_bytes / (1024 * 1024);
    let reserve_mib = q4_packed_auto_reserve_mib(total_mib);
    let mb = q4_packed_auto_cache_mib(total_mib, free_mib);
    if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
        eprintln!(
            "[cuda] Q4 packed cache auto: total={}MiB free={}MiB reserve={}MiB limit={}MiB",
            total_mib, free_mib, reserve_mib, mb
        );
    }
    Ok(mb.saturating_mul(1024 * 1024))
}

fn q4_packed_auto_reserve_mib(total_mib: usize) -> usize {
    align_up(total_mib / 4, 256).clamp(1024, 5120)
}

fn q4_packed_auto_cache_mib(total_mib: usize, free_mib: usize) -> usize {
    let reserve_mib = q4_packed_auto_reserve_mib(total_mib);
    let target_mib = align_up(total_mib / 10, 128).clamp(512, 2048);
    let available_mib = free_mib.saturating_sub(reserve_mib);
    if available_mib >= target_mib {
        target_mib
    } else if available_mib >= 512 {
        align_down(available_mib, 128)
    } else {
        0
    }
}

pub(super) fn q4_f32_cache_limit(api: &CudaApi) -> Result<usize, String> {
    if !crate::tuning::expanded_weight_cache_allowed() {
        if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
            eprintln!("[cuda] Q4 F32 cache limit=0MiB");
        }
        return Ok(0);
    }
    let auto_requested = match std::env::var("RNB_CUDA_Q4_F32_CACHE_MB") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                true
            } else {
                let mb = trimmed.parse::<usize>().map_err(|e| {
                    format!("RNB_CUDA_Q4_F32_CACHE_MB must be integer MiB or auto: {e}")
                })?;
                if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
                    eprintln!("[cuda] Q4 F32 cache limit={}MiB", mb);
                }
                return Ok(mb.saturating_mul(1024 * 1024));
            }
        }
        Err(_) => false,
    };
    let mb = if auto_requested {
        let (free_bytes, total_bytes) = unsafe { api.mem_get_info() }?;
        let total_mib = total_bytes / (1024 * 1024);
        let free_mib = free_bytes / (1024 * 1024);
        q4_f32_auto_cache_mib(total_mib, free_mib)
    } else {
        0
    };
    if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
        eprintln!("[cuda] Q4 F32 cache limit={}MiB", mb);
    }
    Ok(mb.saturating_mul(1024 * 1024))
}

fn q4_f32_auto_reserve_mib(total_mib: usize) -> usize {
    align_up(total_mib.saturating_mul(3) / 16, 256).clamp(1024, 4096)
}

fn q4_f32_auto_cache_mib(total_mib: usize, free_mib: usize) -> usize {
    let reserve_mib = q4_f32_auto_reserve_mib(total_mib);
    let target_mib = align_up(total_mib / 5, 256).clamp(1024, 3072);
    let available_mib = free_mib.saturating_sub(reserve_mib);
    if available_mib >= target_mib {
        target_mib
    } else if available_mib >= 1024 {
        align_down(available_mib, 256)
    } else {
        0
    }
}

pub(super) fn q6_f32_cache_limit() -> Result<usize, String> {
    if !crate::tuning::expanded_weight_cache_allowed() {
        if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
            eprintln!("[cuda] Q6 F32 cache limit=0MiB");
        }
        return Ok(0);
    }
    let mb = match std::env::var("RNB_CUDA_Q6_F32_CACHE_MB") {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                0
            } else {
                trimmed.parse::<usize>().map_err(|e| {
                    format!("RNB_CUDA_Q6_F32_CACHE_MB must be integer MiB or auto: {e}")
                })?
            }
        }
        Err(_) => 0,
    };
    if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
        eprintln!("[cuda] Q6 F32 cache limit={}MiB", mb);
    }
    Ok(mb.saturating_mul(1024 * 1024))
}

pub(super) fn q6_f16_cache_limit(api: &CudaApi) -> Result<usize, String> {
    if !crate::tuning::expanded_weight_cache_allowed() {
        if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
            eprintln!("[cuda] Q6 F16 cache limit=0MiB");
        }
        return Ok(0);
    }
    if let Ok(raw) = std::env::var("RNB_CUDA_Q6_F16_CACHE_MB") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("auto") {
            let mb = trimmed.parse::<usize>().map_err(|e| {
                format!("RNB_CUDA_Q6_F16_CACHE_MB must be integer MiB or auto: {e}")
            })?;
            return Ok(mb.saturating_mul(1024 * 1024));
        }
    }

    let (free_bytes, total_bytes) = unsafe { api.mem_get_info() }?;
    let total_mib = total_bytes / (1024 * 1024);
    let free_mib = free_bytes / (1024 * 1024);
    let mb = q6_f16_auto_cache_mib(total_mib, free_mib);
    if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
        eprintln!(
            "[cuda] Q6 F16 cache auto: total={}MiB free={}MiB reserve={}MiB limit={}MiB",
            total_mib,
            free_mib,
            q6_f16_auto_reserve_mib(total_mib),
            mb
        );
    }
    Ok(mb.saturating_mul(1024 * 1024))
}

fn q6_f16_auto_reserve_mib(total_mib: usize) -> usize {
    align_up(total_mib / 5, 256).clamp(1024, 4096)
}

fn q6_f16_auto_cache_mib(total_mib: usize, free_mib: usize) -> usize {
    // cu19: Q6 F16 path now has a GPU-dequant transient fallback in
    // q6_f16_cache.rs (cu17 q4 f16 pool is reused), so the resident cache is
    // no longer required for correctness or steady-state perf. Holding a
    // 1–4 GiB resident cache on tight VRAM (10 GiB class) triggers q4k
    // cache offload mid-prefill, which is a net loss. Only enable the
    // resident cache when there is genuine headroom (>= 6 GiB available
    // after reserve), which roughly maps to 16+ GiB total VRAM. Smaller
    // devices stay at 0 and rely on the transient pool. Override with
    // RNB_CUDA_Q6_F16_CACHE_MB for explicit sizing.
    let reserve_mib = q6_f16_auto_reserve_mib(total_mib);
    let available_mib = free_mib.saturating_sub(reserve_mib);
    if available_mib < 6 * 1024 {
        return 0;
    }
    let target_mib = align_up(total_mib / 5, 128).clamp(1024, 4096);
    if available_mib >= target_mib {
        target_mib
    } else {
        available_mib & !127
    }
}

pub(super) fn q6_packed_cache_limit(api: &CudaApi) -> Result<usize, String> {
    if let Ok(raw) = std::env::var("RNB_CUDA_Q6_PACKED_CACHE_MB") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("auto") {
            let mb = trimmed.parse::<usize>().map_err(|e| {
                format!("RNB_CUDA_Q6_PACKED_CACHE_MB must be integer MiB or auto: {e}")
            })?;
            return Ok(mb.saturating_mul(1024 * 1024));
        }
    }

    let (free_bytes, total_bytes) = unsafe { api.mem_get_info() }?;
    let total_mib = total_bytes / (1024 * 1024);
    let free_mib = free_bytes / (1024 * 1024);
    let reserve_mib = q6_packed_auto_reserve_mib(total_mib);
    let mb = q6_packed_auto_cache_mib(total_mib, free_mib);
    if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
        eprintln!(
            "[cuda] Q6 packed cache auto: total={}MiB free={}MiB reserve={}MiB limit={}MiB",
            total_mib, free_mib, reserve_mib, mb
        );
    }
    Ok(mb.saturating_mul(1024 * 1024))
}

fn q6_packed_auto_reserve_mib(total_mib: usize) -> usize {
    align_up(total_mib / 6, 256).clamp(1024, 4096)
}

fn q6_packed_auto_cache_mib(total_mib: usize, free_mib: usize) -> usize {
    let reserve_mib = q6_packed_auto_reserve_mib(total_mib);
    let target_mib = align_up(total_mib / 20, 128).clamp(256, 1536);
    let available_mib = free_mib.saturating_sub(reserve_mib);
    if available_mib >= target_mib {
        target_mib
    } else if available_mib >= 256 {
        available_mib
    } else {
        0
    }
}

pub(super) fn moe_layer_cache_limit(api: &CudaApi) -> Result<usize, String> {
    if !tuning::moe_layer_cache_enabled() && !tuning::nemotron_q5_layer_cache_enabled() {
        return Ok(0);
    }
    if let Ok(raw) = std::env::var("RNB_CUDA_MOE_LAYER_CACHE_MB") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("auto") {
            let mb = trimmed.parse::<usize>().map_err(|e| {
                format!("RNB_CUDA_MOE_LAYER_CACHE_MB must be integer MiB or auto: {e}")
            })?;
            return Ok(mb.saturating_mul(1024 * 1024));
        }
    }

    let (free_bytes, total_bytes) = unsafe { api.mem_get_info() }?;
    let total_mib = total_bytes / (1024 * 1024);
    let free_mib = free_bytes / (1024 * 1024);
    let nemotron_only =
        !tuning::moe_layer_cache_enabled() && tuning::nemotron_q5_layer_cache_enabled();
    let reserve_mib = moe_layer_cache_auto_reserve_mib(total_mib, nemotron_only);
    let mb = moe_layer_cache_auto_limit_mib(total_mib, free_mib, nemotron_only);
    if std::env::var("RNB_CUDA_CACHE_LOG").ok().as_deref() == Some("1") {
        eprintln!(
            "[cuda] MoE layer cache auto: total={}MiB free={}MiB reserve={}MiB limit={}MiB",
            total_mib, free_mib, reserve_mib, mb
        );
    }
    Ok(mb.saturating_mul(1024 * 1024))
}

fn moe_layer_cache_auto_reserve_mib(total_mib: usize, nemotron_only: bool) -> usize {
    if nemotron_only {
        // Nemotron sparse prefill needs a temp_slab roughly the size of all
        // unique expert weights for the layer batch (~half of total VRAM on
        // typical 30B-class models). Reserve scales with total VRAM so the
        // formula degrades to 0-cache on small GPUs (CLAUDE.md proportional
        // policy) while keeping large GPUs unaffected via the upper clamp.
        align_up(total_mib / 2, 256).clamp(2048, 8192)
    } else {
        align_up(total_mib.saturating_mul(3) / 16, 256).clamp(1024, 4096)
    }
}

fn moe_layer_cache_auto_target_mib(total_mib: usize, nemotron_only: bool) -> usize {
    if nemotron_only {
        align_up(total_mib.saturating_mul(2) / 3, 256).clamp(1536, 8192)
    } else {
        // cu35→cu36 revert: target=total/2 도입 후 Qwen3.6 ABAB 8-pair 일관
        // 회귀 (-4.2% prefill but +5.7% decode, total +3.3%). cu34 단일 run
        // -20% 측정 wrong (variance). target=total*3/8 로 원복. opt-in env
        // (RNB_CUDA_MOE_LAYER_CACHE_MB) 그대로.
        align_up(total_mib.saturating_mul(3) / 8, 256).clamp(512, 8192)
    }
}

fn moe_layer_cache_auto_limit_mib(total_mib: usize, free_mib: usize, nemotron_only: bool) -> usize {
    let reserve_mib = moe_layer_cache_auto_reserve_mib(total_mib, nemotron_only);
    let target_mib = moe_layer_cache_auto_target_mib(total_mib, nemotron_only);
    target_mib.min(free_mib.saturating_sub(reserve_mib))
}

pub(super) fn ensure_device_buffer(
    api: &CudaApi,
    ptr: &mut Option<u64>,
    capacity: &mut usize,
    bytes: usize,
) -> Result<u64, String> {
    if let Some(existing) = *ptr {
        if *capacity >= bytes {
            return Ok(existing);
        }
        unsafe { api.mem_free(existing)? };
        *ptr = None;
        *capacity = 0;
    }
    let allocation_bytes = device_buffer_allocation_capacity(api, bytes)?;
    let allocated = match unsafe { api.mem_alloc(allocation_bytes) } {
        Ok(p) => p,
        Err(err) => {
            if cuda_mem_alloc_oom(&err) {
                let (free_bytes, total_bytes) = unsafe { api.mem_get_info() }.unwrap_or((0, 0));
                eprintln!(
                    "[cuda:ensure_device_buffer] OOM: requested={}MiB free={}MiB total={}MiB",
                    allocation_bytes / (1024 * 1024),
                    free_bytes / (1024 * 1024),
                    total_bytes / (1024 * 1024)
                );
            }
            return Err(err);
        }
    };
    *ptr = Some(allocated);
    *capacity = allocation_bytes;
    Ok(allocated)
}

pub(super) fn device_buffer_allocation_capacity(
    api: &CudaApi,
    bytes: usize,
) -> Result<usize, String> {
    let grown = transient_device_buffer_capacity(bytes);
    if grown == bytes {
        return Ok(bytes);
    }
    let (free_bytes, total_bytes) = unsafe { api.mem_get_info() }?;
    let reserve_bytes = transient_device_buffer_growth_reserve_bytes(total_bytes);
    if grown <= free_bytes.saturating_sub(reserve_bytes) {
        Ok(grown)
    } else {
        Ok(bytes)
    }
}

fn transient_device_buffer_capacity(bytes: usize) -> usize {
    let enabled = std::env::var("RNB_CUDA_TRANSIENT_BUFFER_GROWTH")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false);
    if !enabled || bytes < 1024 * 1024 {
        return bytes;
    }

    let extra = if bytes >= 256 * 1024 * 1024 {
        bytes / 16
    } else {
        bytes / 8
    };
    let alignment = if bytes >= 64 * 1024 * 1024 {
        16 * 1024 * 1024
    } else {
        1024 * 1024
    };
    align_up(bytes.saturating_add(extra), alignment).max(bytes)
}

fn transient_device_buffer_growth_reserve_bytes(total_bytes: usize) -> usize {
    align_up(total_bytes / 16, 128 * 1024 * 1024).clamp(256 * 1024 * 1024, 1024 * 1024 * 1024)
}

pub(super) fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn align_down(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    value & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::{
        moe_layer_cache_auto_limit_mib, moe_layer_cache_auto_reserve_mib,
        moe_layer_cache_auto_target_mib, q4_f32_auto_cache_mib, q4_f32_auto_reserve_mib,
        q4_packed_auto_cache_mib, q4_packed_auto_reserve_mib, q6_f16_auto_cache_mib,
        q6_packed_auto_cache_mib, q6_packed_auto_reserve_mib, transient_device_buffer_capacity,
        transient_device_buffer_growth_reserve_bytes,
    };

    #[test]
    fn transient_device_buffer_capacity_grows_proportionally() {
        unsafe {
            std::env::set_var("RNB_CUDA_TRANSIENT_BUFFER_GROWTH", "1");
        }
        assert_eq!(transient_device_buffer_capacity(512 * 1024), 512 * 1024);
        assert_eq!(
            transient_device_buffer_capacity(32 * 1024 * 1024),
            36 * 1024 * 1024
        );
        assert_eq!(
            transient_device_buffer_capacity(499 * 1024 * 1024),
            544 * 1024 * 1024
        );
        unsafe {
            std::env::remove_var("RNB_CUDA_TRANSIENT_BUFFER_GROWTH");
        }
    }

    #[test]
    fn transient_device_buffer_capacity_defaults_to_exact() {
        unsafe {
            std::env::remove_var("RNB_CUDA_TRANSIENT_BUFFER_GROWTH");
        }
        assert_eq!(
            transient_device_buffer_capacity(499 * 1024 * 1024),
            499 * 1024 * 1024
        );
    }

    #[test]
    fn transient_device_buffer_growth_reserve_scales_with_vram() {
        assert_eq!(
            transient_device_buffer_growth_reserve_bytes(4 * 1024 * 1024 * 1024),
            256 * 1024 * 1024
        );
        assert_eq!(
            transient_device_buffer_growth_reserve_bytes(10 * 1024 * 1024 * 1024),
            640 * 1024 * 1024
        );
        assert_eq!(
            transient_device_buffer_growth_reserve_bytes(32 * 1024 * 1024 * 1024),
            1024 * 1024 * 1024
        );
    }

    #[test]
    fn q4_packed_auto_cache_scales_with_vram() {
        assert_eq!(q4_packed_auto_cache_mib(4 * 1024, 4 * 1024), 512);
        assert_eq!(q4_packed_auto_cache_mib(8 * 1024, 8 * 1024), 896);
        assert_eq!(q4_packed_auto_cache_mib(10 * 1024, 10 * 1024), 1024);
        assert_eq!(q4_packed_auto_cache_mib(11_917, 11_385), 1280);
        assert_eq!(q4_packed_auto_cache_mib(16 * 1024, 16 * 1024), 1664);
        assert_eq!(q4_packed_auto_cache_mib(24 * 1024, 24 * 1024), 2048);
    }

    #[test]
    fn q4_packed_auto_cache_respects_free_memory_reserve() {
        assert_eq!(q4_packed_auto_reserve_mib(11_917), 3072);
        assert_eq!(q4_packed_auto_cache_mib(11_917, 3_500), 0);
        assert_eq!(q4_packed_auto_cache_mib(11_917, 3_650), 512);
    }

    #[test]
    fn q4_f32_auto_cache_scales_with_vram() {
        assert_eq!(q4_f32_auto_cache_mib(4 * 1024, 4 * 1024), 1024);
        assert_eq!(q4_f32_auto_cache_mib(8 * 1024, 8 * 1024), 1792);
        assert_eq!(q4_f32_auto_cache_mib(10 * 1024, 10 * 1024), 2048);
        assert_eq!(q4_f32_auto_cache_mib(11_917, 11_385), 2560);
        assert_eq!(q4_f32_auto_cache_mib(16 * 1024, 16 * 1024), 3072);
        assert_eq!(q4_f32_auto_cache_mib(24 * 1024, 24 * 1024), 3072);
    }

    #[test]
    fn q4_f32_auto_cache_respects_free_memory_reserve() {
        assert_eq!(q4_f32_auto_reserve_mib(11_917), 2304);
        assert_eq!(q4_f32_auto_cache_mib(11_917, 3_200), 0);
        assert_eq!(q4_f32_auto_cache_mib(11_917, 3_600), 1280);
    }

    #[test]
    fn q6_packed_auto_reserve_scales_with_vram() {
        assert_eq!(q6_packed_auto_reserve_mib(4 * 1024), 1024);
        assert_eq!(q6_packed_auto_reserve_mib(8 * 1024), 1536);
        assert_eq!(q6_packed_auto_reserve_mib(11_917), 2048);
        assert_eq!(q6_packed_auto_reserve_mib(16 * 1024), 2816);
        assert_eq!(q6_packed_auto_reserve_mib(24 * 1024), 4096);
    }

    #[test]
    fn q6_packed_auto_cache_scales_with_vram() {
        assert_eq!(q6_packed_auto_cache_mib(8 * 1024, 8 * 1024), 512);
        assert_eq!(q6_packed_auto_cache_mib(10 * 1024, 10 * 1024), 512);
        assert_eq!(q6_packed_auto_cache_mib(11_917, 11_385), 640);
        assert_eq!(q6_packed_auto_cache_mib(16 * 1024, 16 * 1024), 896);
        assert_eq!(q6_packed_auto_cache_mib(24 * 1024, 24 * 1024), 1280);
    }

    #[test]
    fn q6_packed_auto_cache_respects_free_memory_reserve() {
        assert_eq!(q6_packed_auto_cache_mib(11_917, 2_200), 0);
        assert_eq!(q6_packed_auto_cache_mib(11_917, 2_400), 352);
    }

    #[test]
    fn moe_layer_cache_auto_budget_scales_with_vram() {
        assert_eq!(moe_layer_cache_auto_reserve_mib(8 * 1024, false), 1536);
        assert_eq!(moe_layer_cache_auto_target_mib(8 * 1024, false), 3072);
        assert_eq!(
            moe_layer_cache_auto_limit_mib(8 * 1024, 8 * 1024, false),
            3072
        );

        assert_eq!(moe_layer_cache_auto_reserve_mib(16 * 1024, false), 3072);
        assert_eq!(moe_layer_cache_auto_target_mib(16 * 1024, false), 6144);
        assert_eq!(
            moe_layer_cache_auto_limit_mib(16 * 1024, 7 * 1024, false),
            4096
        );

        assert_eq!(moe_layer_cache_auto_reserve_mib(8 * 1024, true), 4096);
        assert_eq!(moe_layer_cache_auto_target_mib(8 * 1024, true), 5632);
        assert_eq!(
            moe_layer_cache_auto_limit_mib(8 * 1024, 8 * 1024, true),
            4096
        );
    }

    #[test]
    fn q6_f16_auto_cache_scales_with_vram() {
        assert_eq!(q6_f16_auto_cache_mib(8 * 1024, 8 * 1024), 1664);
        assert_eq!(q6_f16_auto_cache_mib(10 * 1024, 10 * 1024), 2048);
        assert_eq!(q6_f16_auto_cache_mib(11_917, 11_350), 2432);
        assert_eq!(q6_f16_auto_cache_mib(16 * 1024, 16 * 1024), 3328);
        assert_eq!(q6_f16_auto_cache_mib(24 * 1024, 24 * 1024), 4096);
    }

    #[test]
    fn q6_f16_auto_cache_respects_free_memory_reserve() {
        assert_eq!(q6_f16_auto_cache_mib(11_917, 3_000), 0);
        assert_eq!(q6_f16_auto_cache_mib(11_917, 3_600), 0);
        assert_eq!(q6_f16_auto_cache_mib(11_917, 9_000), 2432);
    }
}
