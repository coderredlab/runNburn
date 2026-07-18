use super::super::super::*;

fn env_flag_value(name: &str) -> Option<bool> {
    std::env::var(name)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .ok()
}

fn qwen35_sparse_by_token_temp_slab_enabled(token_count: usize, slots: usize) -> bool {
    if let Some(enabled) = env_flag_value("RNB_CUDA_PREFILL_TEMP_SLAB") {
        return enabled;
    }
    let mtp_device_verify = env_flag_value("RNB_MTP_DEVICE_VERIFY").unwrap_or(false);
    !(mtp_device_verify && token_count <= 8 && slots <= 64)
}

fn qwen35_selected_base_temp_slab_cache_key(
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
) -> Qwen35SelectedBaseTempSlabCacheKey {
    Qwen35SelectedBaseTempSlabCacheKey {
        gate_ptr: gate_all.as_ptr() as usize,
        gate_len: gate_all.len(),
        up_ptr: up_all.as_ptr() as usize,
        up_len: up_all.len(),
        down_ptr: down_all.as_ptr() as usize,
        down_len: down_all.len(),
        expert_ids: expert_ids.to_vec(),
        down_quant,
        n_ff,
        n_embd,
        range_upload: qwen35_selected_base_range_upload_enabled(),
    }
}

fn qwen35_selected_base_temp_slab_cache_device_slots(
    cache: &Qwen35SelectedBaseTempSlabCache,
    selected_upload_calls: usize,
    selected_upload_bytes: usize,
) -> PreparedQwen35DeviceSlotPtrs {
    PreparedQwen35DeviceSlotPtrs {
        expert_ids: cache.key.expert_ids.clone(),
        expert_slab_indices: cache.expert_slab_indices.clone(),
        gate_base: cache.gate_base,
        up_base: cache.up_base,
        down_base: cache.down_base,
        gate_expert_bytes: cache.gate_expert_bytes,
        up_expert_bytes: cache.up_expert_bytes,
        down_expert_bytes: cache.down_expert_bytes,
        selected_upload_calls,
        selected_upload_bytes,
        mixed_expert_ptrs: None,
        group_meta2: cache.group_meta2.clone(),
        group_meta4: cache.group_meta4.clone(),
        group_meta8: cache.group_meta8.clone(),
        group_meta16: cache.group_meta16.clone(),
    }
}

fn qwen35_group_meta_from_ids_enabled() -> bool {
    // The from-ids probe caused CUDA 719/Xid79 during Qwen3.6 35B prefill.
    // Keep the env knob inert until grouped metadata gets a stricter safety
    // harness than slice-equivalence tests.
    let _ = std::env::var("RNB_CUDA_QWEN35_GROUP_META_FROM_IDS");
    false
}

fn qwen35_pack4_group_offsets_from_group_meta(group_meta: &[u32]) -> Result<Vec<u32>, String> {
    if group_meta.len() % 2 != 0 {
        return Err(format!(
            "Qwen35 pack4 group offset meta length must be even, got {}",
            group_meta.len()
        ));
    }
    let mut offsets = Vec::with_capacity(group_meta.len() / 2 + 1);
    let mut next = 0u32;
    offsets.push(next);
    for chunk in group_meta.chunks_exact(2) {
        let group_len = chunk[1];
        if group_len == 0 || group_len > 8 {
            return Err(format!(
                "Qwen35 pack4 group offset requires group length 1..=8, got {group_len}"
            ));
        }
        next = next
            .checked_add((group_len + 3) / 4)
            .ok_or_else(|| "Qwen35 pack4 group offset overflow".to_string())?;
        offsets.push(next);
    }
    Ok(offsets)
}

fn qwen35_pack4_group_offsets_for_down_group_meta(
    gate_up_group_meta: &[u32],
    down_group_meta: &[u32],
) -> Result<Vec<u32>, String> {
    let offsets = qwen35_pack4_group_offsets_from_group_meta(gate_up_group_meta)?;
    if down_group_meta.len() % 2 != 0 {
        return Err(format!(
            "Qwen35 pack4 down group meta length must be even, got {}",
            down_group_meta.len()
        ));
    }
    let down_groups = down_group_meta.len() / 2;
    let pack_groups = offsets.last().copied().unwrap_or(0) as usize;
    if pack_groups != down_groups {
        return Err(format!(
            "Qwen35 Q4 gate/up group8 pack4 offset mismatch: pack_groups={pack_groups} down_groups={down_groups}"
        ));
    }

    for (gate_group, gate_chunk) in gate_up_group_meta.chunks_exact(2).enumerate() {
        let gate_start = gate_chunk[0];
        let gate_len = gate_chunk[1];
        let gate_end = gate_start.checked_add(gate_len).ok_or_else(|| {
            format!(
                "Qwen35 pack4 group8 handoff slot range overflows: start={gate_start} len={gate_len}"
            )
        })?;
        let down_start_group = offsets[gate_group] as usize;
        let down_end_group = offsets[gate_group + 1] as usize;
        let mut next_slot = gate_start;
        for down_group in down_start_group..down_end_group {
            let down_chunk = &down_group_meta[down_group * 2..down_group * 2 + 2];
            let down_start = down_chunk[0];
            let down_len = down_chunk[1];
            if down_len == 0 || down_len > 4 {
                return Err(format!(
                    "Qwen35 pack4 down group length must be 1..=4, got {down_len}"
                ));
            }
            if down_start != next_slot {
                return Err(format!(
                    "Qwen35 pack4 down group {down_group} must start at slot {next_slot}, got {down_start}"
                ));
            }
            next_slot = next_slot.checked_add(down_len).ok_or_else(|| {
                format!(
                    "Qwen35 pack4 down group slot range overflows: start={down_start} len={down_len}"
                )
            })?;
            if next_slot > gate_end {
                return Err(format!(
                    "Qwen35 pack4 down group {down_group} overran gate group {gate_group}: next_slot={next_slot} gate_end={gate_end}"
                ));
            }
        }
        if next_slot != gate_end {
            return Err(format!(
                "Qwen35 pack4 down groups do not cover gate group {gate_group}: next_slot={next_slot} gate_end={gate_end}"
            ));
        }
    }

    Ok(offsets)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35SelectedSparseGateUpActivationRunner {
    Pack4F32Group8,
    Pack4F32Group4,
    FusedSiluGroup8,
    SeparateUngrouped,
    SeparateGrouped,
}

impl Qwen35SelectedSparseGateUpActivationRunner {
    fn needs_separate_silu(self) -> bool {
        matches!(
            self,
            Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped
                | Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped
        )
    }
}

fn qwen35_selected_sparse_gate_up_activation_runner(
    has_pack4_group_offsets: bool,
    q4_gate_up_silu_pack4_f32: bool,
    q4_gate_up_silu_fused: bool,
    gate_up_grouped: bool,
) -> Qwen35SelectedSparseGateUpActivationRunner {
    if has_pack4_group_offsets {
        Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8
    } else if q4_gate_up_silu_pack4_f32 {
        Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group4
    } else if q4_gate_up_silu_fused {
        Qwen35SelectedSparseGateUpActivationRunner::FusedSiluGroup8
    } else if gate_up_grouped {
        Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped
    } else {
        Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped
    }
}

fn qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(
    descriptor: &Qwen35SelectedSparseExecutionDescriptor,
    has_pack4_group_offsets: bool,
    gate_up_grouped: bool,
) -> Result<Qwen35SelectedSparseGateUpActivationRunner, String> {
    match descriptor.activation_layout {
        Qwen35SelectedSparseActivationLayout::Pack4F32Group8 => {
            if !has_pack4_group_offsets {
                return Err(
                    "Qwen35 selected sparse group8 pack4 runner requires pack offsets".to_string(),
                );
            }
            Ok(Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8)
        }
        Qwen35SelectedSparseActivationLayout::Pack4F32 => {
            Ok(Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group4)
        }
        Qwen35SelectedSparseActivationLayout::FusedSilu => {
            if !gate_up_grouped {
                return Err(
                    "Qwen35 selected sparse fused-SiLU runner requires gate/up group metadata"
                        .to_string(),
                );
            }
            Ok(Qwen35SelectedSparseGateUpActivationRunner::FusedSiluGroup8)
        }
        Qwen35SelectedSparseActivationLayout::SeparateSilu => {
            if gate_up_grouped {
                Ok(Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped)
            } else {
                Ok(Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35SelectedSparseGateUpLaunchPayload {
    Pack4F32Group8 {
        packed_dev: u64,
        pack_group_offsets_dev: u64,
        pack_group_count: usize,
    },
    Pack4F32Group4 {
        packed_dev: u64,
        down_group_count: usize,
        reload_down_group_meta: bool,
    },
    FusedSiluGroup8 {
        group_count: usize,
    },
    SeparateUngrouped,
    SeparateGrouped {
        group_count: usize,
    },
}

fn qwen35_selected_sparse_gate_up_launch_payload(
    runner: Qwen35SelectedSparseGateUpActivationRunner,
    down_pack4_f32: Option<u64>,
    pack4_group_offsets: Option<&[u32]>,
    gate_up_group_meta: &[u32],
    down_group_meta: &[u32],
    group_meta_dev: u64,
    gate_up_group_meta_bytes: usize,
) -> Result<Qwen35SelectedSparseGateUpLaunchPayload, String> {
    match runner {
        Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8 => {
            let offsets = pack4_group_offsets.ok_or_else(|| {
                "Qwen35 Q4 gate/up group8 pack4 fused path missing pack offsets".to_string()
            })?;
            let packed_dev = down_pack4_f32.ok_or_else(|| {
                "Qwen35 Q4 gate/up group8 pack4 fused path missing packed activation buffer"
                    .to_string()
            })?;
            Ok(Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group8 {
                packed_dev,
                pack_group_offsets_dev: group_meta_dev + gate_up_group_meta_bytes as u64,
                pack_group_count: offsets.len() - 1,
            })
        }
        Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group4 => {
            let packed_dev = down_pack4_f32.ok_or_else(|| {
                "Qwen35 Q4 gate/up pack4 fused path missing packed activation buffer".to_string()
            })?;
            Ok(Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group4 {
                packed_dev,
                down_group_count: down_group_meta.len() / 2,
                reload_down_group_meta: gate_up_group_meta != down_group_meta,
            })
        }
        Qwen35SelectedSparseGateUpActivationRunner::FusedSiluGroup8 => {
            Ok(Qwen35SelectedSparseGateUpLaunchPayload::FusedSiluGroup8 {
                group_count: gate_up_group_meta.len() / 2,
            })
        }
        Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped => {
            Ok(Qwen35SelectedSparseGateUpLaunchPayload::SeparateUngrouped)
        }
        Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped => {
            Ok(Qwen35SelectedSparseGateUpLaunchPayload::SeparateGrouped {
                group_count: gate_up_group_meta.len() / 2,
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35SelectedSparseSiluLaunchPayload {
    None,
    Q8 {
        qs_dev: u64,
        ds_dev: u64,
    },
    Pack4F32Group4 {
        packed_dev: u64,
        group_count: usize,
        reload_down_group_meta: bool,
    },
    Plain,
}

fn qwen35_selected_sparse_silu_launch_payload(
    runner: Qwen35SelectedSparseGateUpActivationRunner,
    down_q8: Option<(u64, u64)>,
    down_pack4_f32: Option<u64>,
    gate_up_group_meta: &[u32],
    down_group_meta: &[u32],
) -> Qwen35SelectedSparseSiluLaunchPayload {
    if !runner.needs_separate_silu() {
        return Qwen35SelectedSparseSiluLaunchPayload::None;
    }
    if let Some((qs_dev, ds_dev)) = down_q8 {
        return Qwen35SelectedSparseSiluLaunchPayload::Q8 { qs_dev, ds_dev };
    }
    if let Some(packed_dev) = down_pack4_f32 {
        return Qwen35SelectedSparseSiluLaunchPayload::Pack4F32Group4 {
            packed_dev,
            group_count: down_group_meta.len() / 2,
            reload_down_group_meta: gate_up_group_meta != down_group_meta,
        };
    }
    Qwen35SelectedSparseSiluLaunchPayload::Plain
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35SelectedSparseDownMetaStage {
    TokenMajor,
    RunTile,
    UploadDownMeta,
    ReuseExistingDownMeta,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Qwen35SelectedSparseDownMetaStagingPayload<'a> {
    TokenMajor {
        token_offsets: &'a [u32],
        slot_indices: &'a [u32],
    },
    RunTile(&'a [u32]),
    DownMeta(&'a [u32]),
    ReuseExistingDownMeta,
    None,
}

fn qwen35_selected_sparse_down_meta_stage(
    has_token_major_plan: bool,
    has_run_tile_meta: bool,
    q6_down_pack4_f32: bool,
    has_pack4_group_offsets: bool,
    group4_down: bool,
    gate_up_meta_matches_down_meta: bool,
) -> Qwen35SelectedSparseDownMetaStage {
    if has_token_major_plan {
        Qwen35SelectedSparseDownMetaStage::TokenMajor
    } else if has_run_tile_meta {
        Qwen35SelectedSparseDownMetaStage::RunTile
    } else if q6_down_pack4_f32 && has_pack4_group_offsets {
        Qwen35SelectedSparseDownMetaStage::UploadDownMeta
    } else if q6_down_pack4_f32 {
        Qwen35SelectedSparseDownMetaStage::ReuseExistingDownMeta
    } else if group4_down && !gate_up_meta_matches_down_meta {
        Qwen35SelectedSparseDownMetaStage::UploadDownMeta
    } else {
        Qwen35SelectedSparseDownMetaStage::None
    }
}

fn qwen35_selected_sparse_down_meta_staging_payload<'a>(
    stage: Qwen35SelectedSparseDownMetaStage,
    token_major_plan: Option<&'a Qwen35DownTokenMajorPlan>,
    run_tile_meta: Option<&'a [u32]>,
    down_group_meta: &'a [u32],
) -> Result<Qwen35SelectedSparseDownMetaStagingPayload<'a>, String> {
    match stage {
        Qwen35SelectedSparseDownMetaStage::TokenMajor => {
            let plan = token_major_plan.ok_or_else(|| {
                "Qwen35 selected sparse token-major down meta stage missing token-major plan"
                    .to_string()
            })?;
            Ok(Qwen35SelectedSparseDownMetaStagingPayload::TokenMajor {
                token_offsets: &plan.token_offsets,
                slot_indices: &plan.slot_indices,
            })
        }
        Qwen35SelectedSparseDownMetaStage::RunTile => {
            let meta = run_tile_meta.ok_or_else(|| {
                "Qwen35 selected sparse run-tile down meta stage missing run-tile meta".to_string()
            })?;
            Ok(Qwen35SelectedSparseDownMetaStagingPayload::RunTile(meta))
        }
        Qwen35SelectedSparseDownMetaStage::UploadDownMeta => Ok(
            Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(down_group_meta),
        ),
        Qwen35SelectedSparseDownMetaStage::ReuseExistingDownMeta => {
            Ok(Qwen35SelectedSparseDownMetaStagingPayload::ReuseExistingDownMeta)
        }
        Qwen35SelectedSparseDownMetaStage::None => {
            Ok(Qwen35SelectedSparseDownMetaStagingPayload::None)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35SelectedSparseDownLaunchRunner {
    Q6TokenMajor,
    Q6Q8DotGroup4,
    Q6Pack4F32Group4,
    Q6RunTiled4,
    Q6RunBatchedRef,
    Q6RunBatched8,
    Q6Full4Split,
    Q4Group4,
    Q5Group4,
    Q6Group4,
    Q4Warp4,
    Q5Warp4,
    Q6Warp4,
    Q4Scalar,
    Q5Scalar,
    Q6Scalar,
}

#[allow(clippy::too_many_arguments)]
fn qwen35_selected_sparse_down_launch_runner(
    down_quant: u32,
    group4_down: bool,
    warp_down: bool,
    q4_down_group4: bool,
    down_token_major: bool,
    q6_down_q8dot: bool,
    q6_down_pack4_f32: bool,
    q6_down_run_tiled4: bool,
    q6_down_run_batched_ref: bool,
    q6_down_run_batched8: bool,
    q6_down_full4_split: bool,
) -> Result<Qwen35SelectedSparseDownLaunchRunner, String> {
    match down_quant {
        14 if down_token_major => Ok(Qwen35SelectedSparseDownLaunchRunner::Q6TokenMajor),
        14 if group4_down && q6_down_q8dot => {
            Ok(Qwen35SelectedSparseDownLaunchRunner::Q6Q8DotGroup4)
        }
        14 if group4_down && q6_down_pack4_f32 => {
            Ok(Qwen35SelectedSparseDownLaunchRunner::Q6Pack4F32Group4)
        }
        14 if group4_down && q6_down_run_tiled4 => {
            Ok(Qwen35SelectedSparseDownLaunchRunner::Q6RunTiled4)
        }
        14 if group4_down && q6_down_run_batched_ref => {
            Ok(Qwen35SelectedSparseDownLaunchRunner::Q6RunBatchedRef)
        }
        14 if group4_down && q6_down_run_batched8 => {
            Ok(Qwen35SelectedSparseDownLaunchRunner::Q6RunBatched8)
        }
        14 if group4_down && q6_down_full4_split => {
            Ok(Qwen35SelectedSparseDownLaunchRunner::Q6Full4Split)
        }
        12 if group4_down && q4_down_group4 => Ok(Qwen35SelectedSparseDownLaunchRunner::Q4Group4),
        13 if group4_down => Ok(Qwen35SelectedSparseDownLaunchRunner::Q5Group4),
        14 if group4_down => Ok(Qwen35SelectedSparseDownLaunchRunner::Q6Group4),
        12 if warp_down => Ok(Qwen35SelectedSparseDownLaunchRunner::Q4Warp4),
        13 if warp_down => Ok(Qwen35SelectedSparseDownLaunchRunner::Q5Warp4),
        14 if warp_down => Ok(Qwen35SelectedSparseDownLaunchRunner::Q6Warp4),
        12 => Ok(Qwen35SelectedSparseDownLaunchRunner::Q4Scalar),
        13 => Ok(Qwen35SelectedSparseDownLaunchRunner::Q5Scalar),
        14 => Ok(Qwen35SelectedSparseDownLaunchRunner::Q6Scalar),
        other => Err(format!("unsupported Qwen35 token-batch down quant {other}")),
    }
}

#[allow(clippy::too_many_arguments)]
fn qwen35_selected_sparse_down_launch_runner_from_descriptor(
    descriptor: &Qwen35SelectedSparseExecutionDescriptor,
    group4_down: bool,
    warp_down: bool,
    q4_down_group4: bool,
    down_token_major: bool,
    q6_down_run_tiled4: bool,
    q6_down_run_batched_ref: bool,
    q6_down_run_batched8: bool,
    q6_down_full4_split: bool,
) -> Result<Qwen35SelectedSparseDownLaunchRunner, String> {
    let (down_quant, q6_down_q8dot, q6_down_pack4_f32) = match descriptor.down_runner {
        Qwen35SelectedSparseDownRunner::Q4Existing => (12, false, false),
        Qwen35SelectedSparseDownRunner::Q5Existing => (13, false, false),
        Qwen35SelectedSparseDownRunner::Q6Existing => (14, false, false),
        Qwen35SelectedSparseDownRunner::Q6Q8Dot => (14, true, false),
        Qwen35SelectedSparseDownRunner::Q6Pack4F32 { .. } => (14, false, true),
    };
    qwen35_selected_sparse_down_launch_runner(
        down_quant,
        group4_down,
        warp_down,
        q4_down_group4,
        down_token_major,
        q6_down_q8dot,
        q6_down_pack4_f32,
        q6_down_run_tiled4,
        q6_down_run_batched_ref,
        q6_down_run_batched8,
        q6_down_full4_split,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35SelectedSparseDownLaunchPayload<'a> {
    Q6TokenMajor {
        token_offsets_bytes: usize,
    },
    Q6Q8DotGroup4 {
        qs_dev: u64,
        ds_dev: u64,
        group_count: usize,
    },
    Q6Pack4F32Group4 {
        packed_dev: u64,
        group_count: usize,
    },
    Q6RunTiled4 {
        run_count: usize,
    },
    Q6RunBatchedRef {
        run_count: usize,
    },
    Q6RunBatched8 {
        run_count: usize,
    },
    Q6Full4Split {
        matching_meta: &'a [u32],
        other_meta: &'a [u32],
    },
    Group4 {
        kernel: &'static str,
        group_count: usize,
    },
    Warp4 {
        kernel: &'static str,
    },
    Scalar {
        kernel: &'static str,
    },
}

fn qwen35_selected_sparse_down_launch_payload<'a>(
    runner: Qwen35SelectedSparseDownLaunchRunner,
    token_major_plan: Option<&'a Qwen35DownTokenMajorPlan>,
    down_q8: Option<(u64, u64)>,
    down_pack4_f32: Option<u64>,
    down_run_tile_meta: Option<&'a [u32]>,
    down_group_len_split: Option<&'a Qwen35GroupMetaLenSplit>,
    down_group_meta: &'a [u32],
) -> Result<Qwen35SelectedSparseDownLaunchPayload<'a>, String> {
    match runner {
        Qwen35SelectedSparseDownLaunchRunner::Q6TokenMajor => {
            let plan = token_major_plan.ok_or_else(|| {
                "Qwen35 Q6 token-major down launch missing token-major plan".to_string()
            })?;
            Ok(Qwen35SelectedSparseDownLaunchPayload::Q6TokenMajor {
                token_offsets_bytes: std::mem::size_of_val(plan.token_offsets.as_slice()),
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6Q8DotGroup4 => {
            let (qs_dev, ds_dev) = down_q8
                .ok_or_else(|| "Qwen35 Q6 q8dot down launch missing q8 buffers".to_string())?;
            Ok(Qwen35SelectedSparseDownLaunchPayload::Q6Q8DotGroup4 {
                qs_dev,
                ds_dev,
                group_count: down_group_meta.len() / 2,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6Pack4F32Group4 => {
            let packed_dev = down_pack4_f32.ok_or_else(|| {
                "Qwen35 Q6 pack4-F32 down launch missing packed activation buffer".to_string()
            })?;
            Ok(Qwen35SelectedSparseDownLaunchPayload::Q6Pack4F32Group4 {
                packed_dev,
                group_count: down_group_meta.len() / 2,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6RunTiled4 => {
            let meta = down_run_tile_meta.ok_or_else(|| {
                "Qwen35 Q6 run-tile down launch missing run-tile meta".to_string()
            })?;
            Ok(Qwen35SelectedSparseDownLaunchPayload::Q6RunTiled4 {
                run_count: meta.len() / 5,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6RunBatchedRef => {
            let meta = down_run_tile_meta.ok_or_else(|| {
                "Qwen35 Q6 run-batched reference down launch missing run-batched meta".to_string()
            })?;
            Ok(Qwen35SelectedSparseDownLaunchPayload::Q6RunBatchedRef {
                run_count: meta.len() / 2,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6RunBatched8 => {
            let meta = down_run_tile_meta.ok_or_else(|| {
                "Qwen35 Q6 run-batched8 down launch missing run-batched meta".to_string()
            })?;
            Ok(Qwen35SelectedSparseDownLaunchPayload::Q6RunBatched8 {
                run_count: meta.len() / 2,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6Full4Split => {
            let split = down_group_len_split.ok_or_else(|| {
                "Qwen35 Q6 full4 split down launch missing full4 split meta".to_string()
            })?;
            Ok(Qwen35SelectedSparseDownLaunchPayload::Q6Full4Split {
                matching_meta: &split.matching,
                other_meta: &split.other,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q4Group4 => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Group4 {
                kernel: "rnb_q4k_selected_down_accum_by_token_group4",
                group_count: down_group_meta.len() / 2,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q5Group4 => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Group4 {
                kernel: "rnb_q5k_selected_down_accum_by_token_group4",
                group_count: down_group_meta.len() / 2,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6Group4 => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Group4 {
                kernel: "rnb_q6k_selected_down_accum_by_token_group4",
                group_count: down_group_meta.len() / 2,
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q4Warp4 => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Warp4 {
                kernel: "rnb_q4k_selected_down_accum_by_token_warp4",
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q5Warp4 => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Warp4 {
                kernel: "rnb_q5k_selected_down_accum_by_token_warp4",
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6Warp4 => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Warp4 {
                kernel: "rnb_q6k_selected_down_accum_by_token_warp4",
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q4Scalar => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Scalar {
                kernel: "rnb_q4k_selected_down_accum_by_token",
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q5Scalar => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Scalar {
                kernel: "rnb_q5k_selected_down_accum_by_token",
            })
        }
        Qwen35SelectedSparseDownLaunchRunner::Q6Scalar => {
            Ok(Qwen35SelectedSparseDownLaunchPayload::Scalar {
                kernel: "rnb_q6k_selected_down_accum_by_token",
            })
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen35SelectedSparseExactRunnerPlan {
    gate_up: Qwen35SelectedSparseGateUpActivationRunner,
    down_meta: Qwen35SelectedSparseDownMetaStage,
    down_launch: Qwen35SelectedSparseDownLaunchRunner,
}

#[derive(Clone, Copy)]
struct Qwen35SelectedSparseExactRunnerPlanInput<'a> {
    descriptor: Option<&'a Qwen35SelectedSparseExecutionDescriptor>,
    has_pack4_group_offsets: bool,
    q4_gate_up_silu_pack4_f32: bool,
    q4_gate_up_silu_fused: bool,
    has_gate_up_group_meta: bool,
    has_down_token_major_plan: bool,
    has_down_run_tile_meta: bool,
    q6_down_pack4_f32: bool,
    group4_down: bool,
    gate_up_meta_matches_down_meta: bool,
    down_quant: u32,
    warp_down: bool,
    q4_down_group4: bool,
    down_token_major: bool,
    q6_down_q8dot: bool,
    q6_down_run_tiled4: bool,
    q6_down_run_batched_ref: bool,
    q6_down_run_batched8: bool,
    q6_down_full4_split: bool,
}

fn qwen35_selected_sparse_exact_runner_plan(
    input: Qwen35SelectedSparseExactRunnerPlanInput<'_>,
) -> Result<Qwen35SelectedSparseExactRunnerPlan, String> {
    let gate_up = if let Some(descriptor) = input.descriptor {
        qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(
            descriptor,
            input.has_pack4_group_offsets,
            input.has_gate_up_group_meta,
        )?
    } else {
        qwen35_selected_sparse_gate_up_activation_runner(
            input.has_pack4_group_offsets,
            input.q4_gate_up_silu_pack4_f32,
            input.q4_gate_up_silu_fused,
            input.has_gate_up_group_meta,
        )
    };
    let down_meta = qwen35_selected_sparse_down_meta_stage(
        input.has_down_token_major_plan,
        input.has_down_run_tile_meta,
        input.q6_down_pack4_f32,
        input.has_pack4_group_offsets,
        input.group4_down,
        input.gate_up_meta_matches_down_meta,
    );
    let down_launch = if let Some(descriptor) = input.descriptor {
        qwen35_selected_sparse_down_launch_runner_from_descriptor(
            descriptor,
            input.group4_down,
            input.warp_down,
            input.q4_down_group4,
            input.down_token_major,
            input.q6_down_run_tiled4,
            input.q6_down_run_batched_ref,
            input.q6_down_run_batched8,
            input.q6_down_full4_split,
        )?
    } else {
        qwen35_selected_sparse_down_launch_runner(
            input.down_quant,
            input.group4_down,
            input.warp_down,
            input.q4_down_group4,
            input.down_token_major,
            input.q6_down_q8dot,
            input.q6_down_pack4_f32,
            input.q6_down_run_tiled4,
            input.q6_down_run_batched_ref,
            input.q6_down_run_batched8,
            input.q6_down_full4_split,
        )?
    };

    Ok(Qwen35SelectedSparseExactRunnerPlan {
        gate_up,
        down_meta,
        down_launch,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Qwen35SelectedSparseExactRunnerPayload<'a> {
    gate_up: Qwen35SelectedSparseGateUpLaunchPayload,
    silu: Qwen35SelectedSparseSiluLaunchPayload,
    down_meta: Qwen35SelectedSparseDownMetaStagingPayload<'a>,
    down_launch: Qwen35SelectedSparseDownLaunchPayload<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Qwen35SelectedSparseCompoundRunnerPayload<'a> {
    Q4GateUpGroup8Q6DownPack4 {
        packed_dev: u64,
        pack_group_offsets_dev: u64,
        pack_group_count: usize,
        down_meta: Qwen35SelectedSparseDownMetaStagingPayload<'a>,
        down_group_count: usize,
    },
}

#[derive(Clone, Copy, Debug, Default)]
struct Qwen35SelectedSparseCompoundRunnerTimings {
    gate_up_ms: f64,
    silu_ms: f64,
    down_ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn qwen35_selected_sparse_exact_runner_payload<'a>(
    plan: Qwen35SelectedSparseExactRunnerPlan,
    down_pack4_f32: Option<u64>,
    pack4_group_offsets: Option<&'a [u32]>,
    gate_up_group_meta: &'a [u32],
    down_group_meta: &'a [u32],
    group_meta_dev: u64,
    gate_up_group_meta_bytes: usize,
    token_major_plan: Option<&'a Qwen35DownTokenMajorPlan>,
    down_q8: Option<(u64, u64)>,
    down_run_tile_meta: Option<&'a [u32]>,
    down_group_len_split: Option<&'a Qwen35GroupMetaLenSplit>,
) -> Result<Qwen35SelectedSparseExactRunnerPayload<'a>, String> {
    let gate_up = qwen35_selected_sparse_gate_up_launch_payload(
        plan.gate_up,
        down_pack4_f32,
        pack4_group_offsets,
        gate_up_group_meta,
        down_group_meta,
        group_meta_dev,
        gate_up_group_meta_bytes,
    )?;
    let silu = qwen35_selected_sparse_silu_launch_payload(
        plan.gate_up,
        down_q8,
        down_pack4_f32,
        gate_up_group_meta,
        down_group_meta,
    );
    let down_meta = qwen35_selected_sparse_down_meta_staging_payload(
        plan.down_meta,
        token_major_plan,
        down_run_tile_meta,
        down_group_meta,
    )?;
    let down_launch = qwen35_selected_sparse_down_launch_payload(
        plan.down_launch,
        token_major_plan,
        down_q8,
        down_pack4_f32,
        down_run_tile_meta,
        down_group_len_split,
        down_group_meta,
    )?;

    Ok(Qwen35SelectedSparseExactRunnerPayload {
        gate_up,
        silu,
        down_meta,
        down_launch,
    })
}

fn qwen35_selected_sparse_compound_reference_payload<'a>(
    mode: Qwen35SelectedSparseRunnerMode,
    payload: Qwen35SelectedSparseExactRunnerPayload<'a>,
) -> Option<Qwen35SelectedSparseCompoundRunnerPayload<'a>> {
    if mode != Qwen35SelectedSparseRunnerMode::CompoundExactReference {
        return None;
    }
    let Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group8 {
        packed_dev,
        pack_group_offsets_dev,
        pack_group_count,
    } = payload.gate_up
    else {
        return None;
    };
    let no_silu = payload.silu == Qwen35SelectedSparseSiluLaunchPayload::None;
    let Qwen35SelectedSparseDownLaunchPayload::Q6Pack4F32Group4 {
        packed_dev: down_packed_dev,
        group_count: down_group_count,
    } = payload.down_launch
    else {
        return None;
    };
    let down_meta_ready = matches!(
        payload.down_meta,
        Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(_)
            | Qwen35SelectedSparseDownMetaStagingPayload::ReuseExistingDownMeta
    );

    if !no_silu || !down_meta_ready || down_packed_dev != packed_dev {
        return None;
    }

    Some(
        Qwen35SelectedSparseCompoundRunnerPayload::Q4GateUpGroup8Q6DownPack4 {
            packed_dev,
            pack_group_offsets_dev,
            pack_group_count,
            down_meta: payload.down_meta,
            down_group_count,
        },
    )
}

fn qwen35_selected_sparse_compound_graph_meta_workspace_bytes(
    gate_up_group_meta_bytes: usize,
    pack_group_offsets_bytes: usize,
    down_meta_bytes: usize,
    graph_enabled: bool,
) -> usize {
    let gate_meta_bytes = gate_up_group_meta_bytes + pack_group_offsets_bytes;
    if graph_enabled {
        gate_meta_bytes + down_meta_bytes
    } else {
        gate_meta_bytes.max(down_meta_bytes)
    }
}

fn qwen35_selected_sparse_compound_graph_down_meta_dev(
    group_meta_dev: u64,
    gate_up_group_meta_bytes: usize,
    pack_group_offsets_bytes: usize,
    down_meta_bytes: usize,
    graph_enabled: bool,
) -> Result<Option<u64>, String> {
    if !graph_enabled || down_meta_bytes == 0 {
        return Ok(None);
    }
    if group_meta_dev == 0 {
        return Err(
            "Qwen35 selected sparse compound graph requires group-meta workspace".to_string(),
        );
    }
    Ok(Some(
        group_meta_dev + (gate_up_group_meta_bytes + pack_group_offsets_bytes) as u64,
    ))
}

fn qwen35_selected_sparse_compound_graph_captures_zero_output(
    zero_output: bool,
    graph_enabled: bool,
    graph_zero_enabled: bool,
    trace_kernel: bool,
    has_compound_payload: bool,
) -> bool {
    zero_output && graph_enabled && graph_zero_enabled && !trace_kernel && has_compound_payload
}

fn qwen35_selected_sparse_boundary_trace_enabled() -> bool {
    env_flag_value("RNB_CUDA_QWEN35_SELECTED_SPARSE_BOUNDARY_TRACE").unwrap_or(false)
}

fn qwen35_selected_base_upload_source<'a>(
    upload: &Qwen35SelectedBaseTempSlabUpload,
    gate_all: &'a [u8],
    up_all: &'a [u8],
    down_all: &'a [u8],
    label: &str,
) -> Result<&'a [u8], String> {
    let src = match upload.role {
        Qwen35SelectedBaseWeightRole::Gate => gate_all,
        Qwen35SelectedBaseWeightRole::Up => up_all,
        Qwen35SelectedBaseWeightRole::Down => down_all,
    };
    let end = upload
        .src_byte_offset
        .checked_add(upload.bytes)
        .ok_or_else(|| {
            format!(
                "Qwen35 selected-base {label} upload source end overflows: offset={} bytes={}",
                upload.src_byte_offset, upload.bytes
            )
        })?;
    src.get(upload.src_byte_offset..end).ok_or_else(|| {
        format!(
            "Qwen35 selected-base {label} upload out of range: role={:?} offset={} end={} len={}",
            upload.role,
            upload.src_byte_offset,
            end,
            src.len()
        )
    })
}

fn qwen35_selected_base_role_upload(
    role: Qwen35SelectedBaseWeightRole,
    slot: Qwen35SelectedBaseSlotOffset,
    offsets: &Qwen35SelectedBaseSlotOffsets,
    slab_byte_offset: usize,
) -> Qwen35SelectedBaseTempSlabUpload {
    match role {
        Qwen35SelectedBaseWeightRole::Gate => Qwen35SelectedBaseTempSlabUpload {
            role,
            src_byte_offset: slot.gate_byte_offset,
            slab_byte_offset,
            bytes: offsets.gate_bytes_per_expert,
        },
        Qwen35SelectedBaseWeightRole::Up => Qwen35SelectedBaseTempSlabUpload {
            role,
            src_byte_offset: slot.up_byte_offset,
            slab_byte_offset,
            bytes: offsets.up_bytes_per_expert,
        },
        Qwen35SelectedBaseWeightRole::Down => Qwen35SelectedBaseTempSlabUpload {
            role,
            src_byte_offset: slot.down_byte_offset,
            slab_byte_offset,
            bytes: offsets.down_bytes_per_expert,
        },
    }
}

fn qwen35_selected_base_candidate_source<'a>(
    candidate: &Qwen35ResidentExpertPageCandidate,
    gate_all: &'a [u8],
    up_all: &'a [u8],
    down_all: &'a [u8],
) -> Result<&'a [u8], String> {
    let source = match candidate.role {
        Qwen35ResidentExpertPageRole::Gate => gate_all,
        Qwen35ResidentExpertPageRole::Up => up_all,
        Qwen35ResidentExpertPageRole::Down => down_all,
    };
    let end = candidate
        .byte_offset
        .checked_add(candidate.bytes)
        .ok_or_else(|| {
            format!(
                "Qwen35 selected-base resident admission byte range overflows: offset={} bytes={}",
                candidate.byte_offset, candidate.bytes
            )
        })?;
    source.get(candidate.byte_offset..end).ok_or_else(|| {
        format!(
            "Qwen35 selected-base resident admission byte range out of bounds: role={:?} expert={} offset={} bytes={} source={}",
            candidate.role,
            candidate.expert_id,
            candidate.byte_offset,
            candidate.bytes,
            source.len()
        )
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpertRangeUpload {
    expert_start: usize,
    expert_end: usize,
    slab_expert_offset: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpertRangeUploadPlan {
    ranges: Vec<ExpertRangeUpload>,
    expert_offsets: Vec<Option<usize>>,
    selected_experts: usize,
    slab_experts: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PrefillHotResidentPlan {
    slots: Vec<usize>,
    bytes: usize,
}

#[derive(Clone, Debug)]
struct PrefillHotResidentCandidate {
    first_slot: usize,
    count: usize,
    score: f32,
    missing_bytes: usize,
}

fn build_expert_range_upload_plan(
    expert_ids: &[u32],
    n_expert: usize,
    max_gap_experts: usize,
    max_overhead_permille: usize,
) -> Option<ExpertRangeUploadPlan> {
    if expert_ids.is_empty() || n_expert == 0 || max_overhead_permille < 1000 {
        return None;
    }

    let mut seen = vec![false; n_expert];
    let mut selected = Vec::new();
    for &expert in expert_ids {
        let expert = usize::try_from(expert).ok()?;
        if expert >= n_expert {
            return None;
        }
        if !seen[expert] {
            seen[expert] = true;
            selected.push(expert);
        }
    }
    selected.sort_unstable();
    let selected_experts = selected.len();
    if selected_experts == 0 {
        return None;
    }

    let mut raw_ranges = Vec::new();
    let mut range_start = selected[0];
    let mut range_end = range_start + 1;
    for &expert in selected.iter().skip(1) {
        let gap = expert.saturating_sub(range_end);
        if gap <= max_gap_experts {
            range_end = expert + 1;
        } else {
            raw_ranges.push((range_start, range_end));
            range_start = expert;
            range_end = expert + 1;
        }
    }
    raw_ranges.push((range_start, range_end));

    let slab_experts = raw_ranges.iter().fold(0usize, |acc, (start, end)| {
        acc.saturating_add(end.saturating_sub(*start))
    });
    if slab_experts.saturating_mul(1000) > selected_experts.saturating_mul(max_overhead_permille) {
        return None;
    }

    let mut expert_offsets = vec![None; n_expert];
    let mut ranges = Vec::with_capacity(raw_ranges.len());
    let mut slab_expert_offset = 0usize;
    for (expert_start, expert_end) in raw_ranges {
        for expert in expert_start..expert_end {
            if seen[expert] {
                expert_offsets[expert] = Some(slab_expert_offset + expert - expert_start);
            }
        }
        ranges.push(ExpertRangeUpload {
            expert_start,
            expert_end,
            slab_expert_offset,
        });
        slab_expert_offset = slab_expert_offset.saturating_add(expert_end - expert_start);
    }

    Some(ExpertRangeUploadPlan {
        ranges,
        expert_offsets,
        selected_experts,
        slab_experts,
    })
}

fn build_prefill_hot_resident_plan(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    resident_keys: &HashSet<(usize, usize)>,
    budget_bytes: usize,
) -> PrefillHotResidentPlan {
    if gate_weights.len() != up_weights.len()
        || gate_weights.len() != down_weights.len()
        || gate_weights.len() != route_weights.len()
        || budget_bytes == 0
    {
        return PrefillHotResidentPlan {
            slots: Vec::new(),
            bytes: 0,
        };
    }

    let mut candidates_by_key = HashMap::new();
    for slot in 0..gate_weights.len() {
        let key = (
            q4k_resident_key(gate_weights[slot]),
            q4k_resident_key(up_weights[slot]),
            q4k_resident_key(down_weights[slot]),
        );
        let route_weight = route_weights[slot];
        let score = if route_weight.is_finite() {
            route_weight.max(0.0)
        } else {
            0.0
        };
        candidates_by_key
            .entry(key)
            .and_modify(|candidate: &mut PrefillHotResidentCandidate| {
                candidate.count += 1;
                candidate.score += score;
            })
            .or_insert_with(|| {
                let mut seen = HashSet::new();
                let mut missing_bytes = 0usize;
                for weights in [gate_weights[slot], up_weights[slot], down_weights[slot]] {
                    let weight_key = q4k_resident_key(weights);
                    if !resident_keys.contains(&weight_key) && seen.insert(weight_key) {
                        missing_bytes = missing_bytes.saturating_add(weights.len());
                    }
                }
                PrefillHotResidentCandidate {
                    first_slot: slot,
                    count: 1,
                    score,
                    missing_bytes,
                }
            });
    }

    let mut candidates = candidates_by_key
        .into_values()
        .filter(|candidate| candidate.missing_bytes > 0)
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.count.cmp(&left.count))
            .then_with(|| left.first_slot.cmp(&right.first_slot))
    });

    let mut budget_left = budget_bytes;
    let mut slots = Vec::new();
    let mut bytes = 0usize;
    for candidate in candidates {
        if candidate.missing_bytes > budget_left {
            continue;
        }
        slots.push(candidate.first_slot);
        bytes = bytes.saturating_add(candidate.missing_bytes);
        budget_left = budget_left.saturating_sub(candidate.missing_bytes);
        if budget_left == 0 {
            break;
        }
    }

    PrefillHotResidentPlan { slots, bytes }
}

impl CudaState {
    fn promote_qwen35_prefill_hot_resident_slots(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_count: usize,
    ) -> Result<(), String> {
        if !tuning::qwen35_prefill_hot_resident_enabled()
            || token_count < tuning::qwen35_prefill_hot_resident_min_tokens()
        {
            return Ok(());
        }
        let available = self
            .resident_q4k_limit
            .saturating_sub(self.resident_q4k_bytes);
        let budget = tuning::qwen35_prefill_hot_resident_budget_bytes(self.resident_q4k_limit)
            .min(available);
        if budget == 0 {
            return Ok(());
        }
        let resident_keys = self.resident_q4k.keys().copied().collect::<HashSet<_>>();
        let plan = build_prefill_hot_resident_plan(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            &resident_keys,
            budget,
        );
        if plan.slots.is_empty() {
            return Ok(());
        }

        let mut promote_gate = Vec::new();
        let mut promote_up = Vec::new();
        let mut promote_down = Vec::new();
        let mut promoted_keys = resident_keys;
        for slot in &plan.slots {
            for (group, weights) in [
                (0usize, gate_weights[*slot]),
                (1usize, up_weights[*slot]),
                (2usize, down_weights[*slot]),
            ] {
                if !promoted_keys.insert(q4k_resident_key(weights)) {
                    continue;
                }
                match group {
                    0 => promote_gate.push(weights),
                    1 => promote_up.push(weights),
                    2 => promote_down.push(weights),
                    _ => unreachable!("three selected weight groups"),
                }
            }
        }
        if !promote_gate.is_empty() || !promote_up.is_empty() || !promote_down.is_empty() {
            let local_ptrs = HashMap::new();
            self.batch_resident_q4k_slot_misses_many(
                &[&promote_gate, &promote_up, &promote_down],
                &local_ptrs,
            )?;
        }
        if std::env::var("RNB_CUDA_PREFILL_HOT_RESIDENT_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[cuda-prefill-hot-resident] tokens={} selected={} budget_mb={:.2} planned_mb={:.2} resident_mb={:.2}",
                token_count,
                plan.slots.len(),
                budget as f64 / (1024.0 * 1024.0),
                plan.bytes as f64 / (1024.0 * 1024.0),
                self.resident_q4k_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_by_token(
        &mut self,
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
    ) -> Result<Vec<f32>, String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let output_len = token_count * n_embd;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        self.qwen35_sparse_experts_by_token_to_dev(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
            true,
            false,
        )?;
        let mut output = vec![0.0f32; output_len];
        let trace_phase = std::env::var("RNB_CUDA_PHASE_TRACE").ok().as_deref() == Some("1");
        let phase_t0 = trace_phase.then(std::time::Instant::now);
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        if let Some(t0) = phase_t0 {
            eprintln!(
                "[cuda-phase] qwen35_sparse_dtoh dtoh_ms={:.1}",
                t0.elapsed().as_micros() as f64 / 1000.0
            );
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_by_token_to_dev(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_dev: u64,
        output_dev: u64,
        zero_output: bool,
        prefer_group2_down: bool,
    ) -> Result<(), String> {
        self.qwen35_sparse_experts_by_token_to_dev_prepared(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            down_quant,
            n_ff,
            n_embd,
            input_dev,
            output_dev,
            zero_output,
            prefer_group2_down,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prepare_sparse_slots_by_token(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_count: usize,
        upload_on_copy_stream: bool,
    ) -> Result<PreparedQwen35SparseSlots, String> {
        let slots = gate_weights.len();
        let temp_slab = qwen35_sparse_by_token_temp_slab_enabled(token_count, slots);
        let hot_resident = tuning::mtp_expert_hot_resident_enabled();
        let slot_groups = [gate_weights, up_weights, down_weights];
        self.promote_qwen35_prefill_hot_resident_slots(
            gate_weights,
            up_weights,
            down_weights,
            route_weights,
            token_count,
        )?;
        let (gate_ptrs, up_ptrs, down_ptrs, temp_slab_ptrs, copy_stream_upload) = if let Some(
            ptrs,
        ) =
            self.resident_q4k_slot_ptrs_3_if_all_resident(gate_weights, up_weights, down_weights)
        {
            (ptrs.0, ptrs.1, ptrs.2, ptrs.3, false)
        } else if self.q4k_slot_groups_have_resident(&slot_groups) {
            let ptrs =
                self.mixed_resident_temp_q4k_slot_ptrs_3(gate_weights, up_weights, down_weights)?;
            (ptrs.0, ptrs.1, ptrs.2, ptrs.3, false)
        } else if temp_slab || hot_resident {
            let ptrs = if upload_on_copy_stream {
                self.temp_q4k_slot_ptrs_3_copy_stream(gate_weights, up_weights, down_weights)?
            } else {
                self.temp_q4k_slot_ptrs_3(gate_weights, up_weights, down_weights)?
            };
            let copy_stream_upload = upload_on_copy_stream && !ptrs.3.is_empty();
            (ptrs.0, ptrs.1, ptrs.2, ptrs.3, copy_stream_upload)
        } else {
            let mut local_ptrs = HashMap::new();
            let gate_ptrs = self.resident_q4k_slot_ptrs(gate_weights, &mut local_ptrs)?;
            let up_ptrs = self.resident_q4k_slot_ptrs(up_weights, &mut local_ptrs)?;
            let down_ptrs = self.resident_q4k_slot_ptrs(down_weights, &mut local_ptrs)?;
            (gate_ptrs, up_ptrs, down_ptrs, Vec::new(), false)
        };
        Ok(PreparedQwen35SparseSlots {
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            slot_count: None,
            temp_slab_ptrs,
            copy_stream_upload,
            device_slot_ptrs: None,
            group_meta: None,
            device_route: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_upload_selected_base_temp_slab(
        &mut self,
        slab_dev: u64,
        slab_bytes: usize,
        uploads: &[Qwen35SelectedBaseTempSlabUpload],
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        upload_stream: usize,
        label: &str,
    ) -> Result<(), String> {
        if slab_bytes == 0 {
            return Ok(());
        }
        if qwen35_selected_base_pinned_staging_enabled() {
            let mut uploaded_bytes = 0usize;
            for upload in uploads {
                let end = upload
                    .slab_byte_offset
                    .checked_add(upload.bytes)
                    .ok_or_else(|| {
                        format!(
                            "Qwen35 selected-base {label} pinned upload slab end overflows: offset={} bytes={}",
                            upload.slab_byte_offset, upload.bytes
                        )
                    })?;
                if end > slab_bytes {
                    return Err(format!(
                        "Qwen35 selected-base {label} pinned upload exceeds slab: offset={} end={} slab_bytes={}",
                        upload.slab_byte_offset, end, slab_bytes
                    ));
                }
                uploaded_bytes = uploaded_bytes.checked_add(upload.bytes).ok_or_else(|| {
                    format!(
                        "Qwen35 selected-base {label} pinned upload byte count overflows: added={}",
                        upload.bytes
                    )
                })?;
            }
            if uploaded_bytes != slab_bytes {
                return Err(format!(
                    "Qwen35 selected-base {label} pinned upload does not cover compact slab: uploaded={} slab_bytes={}",
                    uploaded_bytes, slab_bytes
                ));
            }

            let host_slab = self.host_temp_slab_ptr(slab_bytes)?;
            for upload in uploads {
                let weights =
                    qwen35_selected_base_upload_source(upload, gate_all, up_all, down_all, label)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        weights.as_ptr(),
                        host_slab.add(upload.slab_byte_offset),
                        weights.len(),
                    );
                }
            }
            unsafe {
                self.api.memcpy_htod_async(
                    slab_dev,
                    host_slab.cast::<libc::c_void>(),
                    slab_bytes,
                    upload_stream,
                )?;
            }
            return Ok(());
        }

        for upload in uploads {
            let weights =
                qwen35_selected_base_upload_source(upload, gate_all, up_all, down_all, label)?;
            unsafe {
                self.api.memcpy_htod_async(
                    slab_dev
                        .checked_add(u64::try_from(upload.slab_byte_offset).map_err(|_| {
                            format!(
                                "Qwen35 selected-base {label} upload offset exceeds u64: {}",
                                upload.slab_byte_offset
                            )
                        })?)
                        .ok_or_else(|| {
                            format!(
                                "Qwen35 selected-base {label} upload pointer overflows: base={slab_dev} offset={}",
                                upload.slab_byte_offset
                            )
                        })?,
                    weights.as_ptr().cast::<libc::c_void>(),
                    weights.len(),
                    upload_stream,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prepare_selected_base_temp_slab_slots_by_token(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<PreparedQwen35SparseSlots, String> {
        let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
            gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
        )?;
        let zero_base_plan = qwen35_selected_base_temp_slab_slot_ptr_plan(&offsets, 0)?;
        if zero_base_plan.slab_bytes == 0 {
            return Ok(PreparedQwen35SparseSlots {
                gate_ptrs: Vec::new(),
                up_ptrs: Vec::new(),
                down_ptrs: Vec::new(),
                slot_count: None,
                temp_slab_ptrs: Vec::new(),
                copy_stream_upload: false,
                device_slot_ptrs: None,
                group_meta: None,
                device_route: None,
            });
        }

        let slab_dev = self.compute_temp_slab_ptr(zero_base_plan.slab_bytes)?;
        let plan = qwen35_selected_base_temp_slab_slot_ptr_plan(&offsets, slab_dev)?;
        let copy_stream_upload = qwen35_selected_base_copy_stream_enabled();
        let upload_stream = if copy_stream_upload {
            self.copy_stream
        } else {
            self.stream
        };
        self.qwen35_upload_selected_base_temp_slab(
            slab_dev,
            plan.slab_bytes,
            &plan.uploads,
            gate_all,
            up_all,
            down_all,
            upload_stream,
            "temp slab",
        )?;
        let temp_slab_ptrs = plan
            .uploads
            .iter()
            .map(|upload| {
                slab_dev
                    .checked_add(u64::try_from(upload.slab_byte_offset).map_err(|_| {
                        format!(
                            "Qwen35 selected-base temp slab pointer offset exceeds u64: {}",
                            upload.slab_byte_offset
                        )
                    })?)
                    .ok_or_else(|| {
                        format!(
                            "Qwen35 selected-base temp slab pointer overflows: base={slab_dev} offset={}",
                            upload.slab_byte_offset
                        )
                    })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(PreparedQwen35SparseSlots {
            gate_ptrs: plan.gate_ptrs,
            up_ptrs: plan.up_ptrs,
            down_ptrs: plan.down_ptrs,
            slot_count: None,
            temp_slab_ptrs,
            copy_stream_upload,
            device_slot_ptrs: None,
            group_meta: None,
            device_route: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prepare_selected_base_temp_slab_device_slot_ptrs_by_token(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<PreparedQwen35SparseSlots, String> {
        let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
            gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
        )?;
        let n_expert = qwen35_selected_base_full_layer_expert_count(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )?;
        let group_meta2 = qwen35_selected_base_group_meta_from_offsets(&offsets, 2);
        let group_meta4 = qwen35_selected_base_group_meta_from_offsets(&offsets, 4);
        let group_meta8 = qwen35_selected_base_group_meta_from_offsets(&offsets, 8);
        let group_meta16 = qwen35_selected_base_group_meta_from_offsets(&offsets, 16);
        let zero_base_plan = qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, n_expert, 0)?;
        if zero_base_plan.slab_bytes == 0 {
            return Ok(PreparedQwen35SparseSlots {
                gate_ptrs: Vec::new(),
                up_ptrs: Vec::new(),
                down_ptrs: Vec::new(),
                slot_count: None,
                temp_slab_ptrs: Vec::new(),
                copy_stream_upload: false,
                device_slot_ptrs: Some(PreparedQwen35DeviceSlotPtrs {
                    expert_ids: expert_ids.to_vec(),
                    expert_slab_indices: zero_base_plan.expert_slab_indices,
                    gate_base: 0,
                    up_base: 0,
                    down_base: 0,
                    gate_expert_bytes: offsets.gate_bytes_per_expert,
                    up_expert_bytes: offsets.up_bytes_per_expert,
                    down_expert_bytes: offsets.down_bytes_per_expert,
                    selected_upload_calls: 0,
                    selected_upload_bytes: 0,
                    mixed_expert_ptrs: None,
                    group_meta2,
                    group_meta4,
                    group_meta8,
                    group_meta16,
                }),
                group_meta: None,
                device_route: None,
            });
        }

        let cache_enabled = qwen35_selected_base_temp_slab_cache_enabled();
        let cache_key = cache_enabled.then(|| {
            qwen35_selected_base_temp_slab_cache_key(
                gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
            )
        });
        if let Some(key) = cache_key.as_ref() {
            let cache_hit_pending = self
                .qwen35_selected_base_temp_slab_cache
                .as_ref()
                .filter(|cache| cache.key == *key && cache.slab_bytes == zero_base_plan.slab_bytes)
                .map(|cache| cache.copy_stream_upload_pending);
            if let Some(pending) = cache_hit_pending {
                if pending {
                    unsafe { self.api.stream_synchronize(self.copy_stream)? };
                    if let Some(cache) = self.qwen35_selected_base_temp_slab_cache.as_mut() {
                        cache.copy_stream_upload_pending = false;
                    }
                }
                let cache = self
                    .qwen35_selected_base_temp_slab_cache
                    .as_ref()
                    .expect("selected-base temp-slab cache hit exists");
                return Ok(PreparedQwen35SparseSlots {
                    gate_ptrs: Vec::new(),
                    up_ptrs: Vec::new(),
                    down_ptrs: Vec::new(),
                    slot_count: None,
                    temp_slab_ptrs: Vec::new(),
                    copy_stream_upload: false,
                    device_slot_ptrs: Some(qwen35_selected_base_temp_slab_cache_device_slots(
                        cache, 0, 0,
                    )),
                    group_meta: None,
                    device_route: None,
                });
            }
        }

        let (slab_dev, slab_capacity) = if cache_enabled {
            let existing = self.qwen35_selected_base_temp_slab_cache.take();
            if let Some(cache) = existing {
                if cache.copy_stream_upload_pending {
                    unsafe { self.api.stream_synchronize(self.copy_stream)? };
                }
                if cache.slab_capacity >= zero_base_plan.slab_bytes {
                    (cache.slab_dev, cache.slab_capacity)
                } else {
                    unsafe { self.api.mem_free(cache.slab_dev)? };
                    self.set_current()?;
                    let slab_dev = unsafe { self.api.mem_alloc(zero_base_plan.slab_bytes)? };
                    (slab_dev, zero_base_plan.slab_bytes)
                }
            } else {
                self.set_current()?;
                let slab_dev = unsafe { self.api.mem_alloc(zero_base_plan.slab_bytes)? };
                (slab_dev, zero_base_plan.slab_bytes)
            }
        } else {
            (
                self.compute_temp_slab_ptr(zero_base_plan.slab_bytes)?,
                zero_base_plan.slab_bytes,
            )
        };
        let plan = qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, n_expert, slab_dev)?;
        let copy_stream_upload = qwen35_selected_base_copy_stream_enabled();
        let upload_stream = if copy_stream_upload {
            self.copy_stream
        } else {
            self.stream
        };
        self.qwen35_upload_selected_base_temp_slab(
            slab_dev,
            plan.slab_bytes,
            &plan.uploads,
            gate_all,
            up_all,
            down_all,
            upload_stream,
            "compact slot",
        )?;
        let selected_upload_calls = plan.uploads.len();
        let selected_upload_bytes = plan.uploads.iter().map(|upload| upload.bytes).sum();

        if let Some(key) = cache_key {
            self.qwen35_selected_base_temp_slab_cache = Some(Qwen35SelectedBaseTempSlabCache {
                key,
                slab_dev,
                slab_bytes: plan.slab_bytes,
                slab_capacity,
                expert_slab_indices: plan.expert_slab_indices,
                gate_base: plan.gate_base,
                up_base: plan.up_base,
                down_base: plan.down_base,
                gate_expert_bytes: offsets.gate_bytes_per_expert,
                up_expert_bytes: offsets.up_bytes_per_expert,
                down_expert_bytes: offsets.down_bytes_per_expert,
                group_meta2,
                group_meta4,
                group_meta8,
                group_meta16,
                copy_stream_upload_pending: copy_stream_upload,
            });
            let cache = self
                .qwen35_selected_base_temp_slab_cache
                .as_ref()
                .expect("selected-base temp-slab cache stored");
            return Ok(PreparedQwen35SparseSlots {
                gate_ptrs: Vec::new(),
                up_ptrs: Vec::new(),
                down_ptrs: Vec::new(),
                slot_count: None,
                temp_slab_ptrs: vec![slab_dev],
                copy_stream_upload,
                device_slot_ptrs: Some(qwen35_selected_base_temp_slab_cache_device_slots(
                    cache,
                    selected_upload_calls,
                    selected_upload_bytes,
                )),
                group_meta: None,
                device_route: None,
            });
        }

        Ok(PreparedQwen35SparseSlots {
            gate_ptrs: Vec::new(),
            up_ptrs: Vec::new(),
            down_ptrs: Vec::new(),
            slot_count: None,
            temp_slab_ptrs: vec![slab_dev],
            copy_stream_upload,
            device_slot_ptrs: Some(PreparedQwen35DeviceSlotPtrs {
                expert_ids: expert_ids.to_vec(),
                expert_slab_indices: plan.expert_slab_indices,
                gate_base: plan.gate_base,
                up_base: plan.up_base,
                down_base: plan.down_base,
                gate_expert_bytes: offsets.gate_bytes_per_expert,
                up_expert_bytes: offsets.up_bytes_per_expert,
                down_expert_bytes: offsets.down_bytes_per_expert,
                selected_upload_calls,
                selected_upload_bytes,
                mixed_expert_ptrs: None,
                group_meta2,
                group_meta4,
                group_meta8,
                group_meta16,
            }),
            group_meta: None,
            device_route: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_selected_base_has_exact_resident_moe_layer(
        &self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> bool {
        let key = qwen35_moe_layer_key(gate_all, up_all, down_all, down_quant, n_ff, n_embd);
        self.resident_moe_layers.contains_key(&key)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prepare_selected_base_residency_aware_device_slot_ptrs_by_token(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<PreparedQwen35SparseSlots, String> {
        let resident_hit = self.qwen35_selected_base_has_exact_resident_moe_layer(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        ) || self.qwen35_selected_base_existing_resident_role_count_by_token(
            gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
        )? > 0;
        if resident_hit {
            self.qwen35_prepare_selected_base_mixed_resident_device_slot_ptrs_by_token(
                gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
            )
        } else {
            self.qwen35_prepare_selected_base_temp_slab_device_slot_ptrs_by_token(
                gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_selected_base_resident_role_ptrs(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        offsets: &Qwen35SelectedBaseSlotOffsets,
        n_expert: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<HashMap<Qwen35SelectedBaseResidentRole, u64>, String> {
        let mut resident_role_ptrs = HashMap::new();
        let layer_key = qwen35_moe_layer_key(gate_all, up_all, down_all, down_quant, n_ff, n_embd);
        if let Some((gate_base, up_base, down_base)) = self
            .resident_moe_layers
            .get(&layer_key)
            .map(|entry| (entry.gate_base, entry.up_base, entry.down_base))
        {
            for slot in &offsets.slots {
                let expert_id = slot.expert_id as usize;
                if expert_id >= n_expert {
                    return Err(format!(
                        "Qwen35 selected-base resident MoE layer expert id out of range: got {expert_id}, n_expert={n_expert}"
                    ));
                }
                for (role, base, bytes_per_expert) in [
                    (
                        Qwen35SelectedBaseWeightRole::Gate,
                        gate_base,
                        offsets.gate_bytes_per_expert,
                    ),
                    (
                        Qwen35SelectedBaseWeightRole::Up,
                        up_base,
                        offsets.up_bytes_per_expert,
                    ),
                    (
                        Qwen35SelectedBaseWeightRole::Down,
                        down_base,
                        offsets.down_bytes_per_expert,
                    ),
                ] {
                    let expert_offset = u64::try_from(expert_id)
                        .ok()
                        .and_then(|expert| {
                            u64::try_from(bytes_per_expert)
                                .ok()
                                .and_then(|stride| expert.checked_mul(stride))
                        })
                        .ok_or_else(|| {
                            format!(
                                "Qwen35 selected-base resident MoE layer offset overflow: role={role:?} expert={expert_id} stride={bytes_per_expert}"
                            )
                        })?;
                    let ptr = base.checked_add(expert_offset).ok_or_else(|| {
                        format!(
                            "Qwen35 selected-base resident MoE layer pointer overflow: role={role:?} expert={expert_id} base={base} offset={expert_offset}"
                        )
                    })?;
                    resident_role_ptrs
                        .insert(Qwen35SelectedBaseResidentRole { role, expert_id }, ptr);
                }
            }
            if !resident_role_ptrs.is_empty() {
                self.touch_resident_moe_layer(layer_key);
            }
        }

        for slot in &offsets.slots {
            let expert_id = slot.expert_id as usize;
            for role in [
                Qwen35SelectedBaseWeightRole::Gate,
                Qwen35SelectedBaseWeightRole::Up,
                Qwen35SelectedBaseWeightRole::Down,
            ] {
                let role_key = Qwen35SelectedBaseResidentRole { role, expert_id };
                if resident_role_ptrs.contains_key(&role_key) {
                    continue;
                }
                let upload = qwen35_selected_base_role_upload(role, *slot, offsets, 0);
                let weights = qwen35_selected_base_upload_source(
                    &upload,
                    gate_all,
                    up_all,
                    down_all,
                    "mixed resident role",
                )?;
                let key = q4k_resident_key(weights);
                if let Some(ptr) = self.resident_q4k.get(&key).map(|entry| entry.ptr) {
                    self.touch_resident_q4k(key);
                    resident_role_ptrs.insert(role_key, ptr);
                }
            }
        }
        Ok(resident_role_ptrs)
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_selected_base_mixed_source_ptr(
        &mut self,
        source: Qwen35SelectedBaseMixedWeightSource,
        role: Qwen35SelectedBaseWeightRole,
        slot: Qwen35SelectedBaseSlotOffset,
        resident_role_ptrs: &HashMap<Qwen35SelectedBaseResidentRole, u64>,
        slab_dev: u64,
    ) -> Result<u64, String> {
        match source {
            Qwen35SelectedBaseMixedWeightSource::Resident => resident_role_ptrs
                .get(&Qwen35SelectedBaseResidentRole {
                    role,
                    expert_id: slot.expert_id as usize,
                })
                .copied()
                .ok_or_else(|| {
                    format!(
                        "Qwen35 selected-base mixed resident missing resolved pointer: role={role:?} expert={}",
                        slot.expert_id
                    )
                }),
            Qwen35SelectedBaseMixedWeightSource::Temp { slab_byte_offset } => slab_dev
                .checked_add(u64::try_from(slab_byte_offset).map_err(|_| {
                    format!(
                        "Qwen35 selected-base mixed resident temp offset exceeds u64: {slab_byte_offset}"
                    )
                })?)
                .ok_or_else(|| {
                    format!(
                        "Qwen35 selected-base mixed resident temp pointer overflows: base={slab_dev} offset={slab_byte_offset}"
                    )
                }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prepare_selected_base_mixed_resident_temp_slots_by_token(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<PreparedQwen35SparseSlots, String> {
        let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
            gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
        )?;
        let n_expert = qwen35_selected_base_full_layer_expert_count(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )?;
        let resident_role_ptrs = self.qwen35_selected_base_resident_role_ptrs(
            gate_all, up_all, down_all, &offsets, n_expert, down_quant, n_ff, n_embd,
        )?;
        let resident_roles = resident_role_ptrs.keys().copied().collect::<HashSet<_>>();

        let plan =
            qwen35_selected_base_mixed_resident_temp_plan(&offsets, n_expert, &resident_roles)?;
        let slab_dev = if plan.slab_bytes == 0 {
            0
        } else {
            self.compute_temp_slab_ptr(plan.slab_bytes)?
        };
        let copy_stream_upload = plan.slab_bytes > 0 && qwen35_selected_base_copy_stream_enabled();
        let upload_stream = if copy_stream_upload {
            self.copy_stream
        } else {
            self.stream
        };
        self.qwen35_upload_selected_base_temp_slab(
            slab_dev,
            plan.slab_bytes,
            &plan.uploads,
            gate_all,
            up_all,
            down_all,
            upload_stream,
            "mixed resident",
        )?;

        let mut gate_ptrs = Vec::with_capacity(offsets.slots.len());
        let mut up_ptrs = Vec::with_capacity(offsets.slots.len());
        let mut down_ptrs = Vec::with_capacity(offsets.slots.len());
        for slot in &offsets.slots {
            let expert = slot.expert_id as usize;
            let sources = plan
                .expert_sources
                .get(expert)
                .and_then(|sources| *sources)
                .ok_or_else(|| {
                    format!("Qwen35 selected-base mixed resident missing source: expert={expert}")
                })?;
            gate_ptrs.push(self.qwen35_selected_base_mixed_source_ptr(
                sources.gate,
                Qwen35SelectedBaseWeightRole::Gate,
                *slot,
                &resident_role_ptrs,
                slab_dev,
            )?);
            up_ptrs.push(self.qwen35_selected_base_mixed_source_ptr(
                sources.up,
                Qwen35SelectedBaseWeightRole::Up,
                *slot,
                &resident_role_ptrs,
                slab_dev,
            )?);
            down_ptrs.push(self.qwen35_selected_base_mixed_source_ptr(
                sources.down,
                Qwen35SelectedBaseWeightRole::Down,
                *slot,
                &resident_role_ptrs,
                slab_dev,
            )?);
        }

        Ok(PreparedQwen35SparseSlots {
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            slot_count: Some(offsets.slots.len()),
            temp_slab_ptrs: if plan.slab_bytes == 0 {
                Vec::new()
            } else {
                vec![slab_dev]
            },
            copy_stream_upload,
            device_slot_ptrs: None,
            group_meta: Some(PreparedQwen35SparseGroupMeta {
                group_meta2: qwen35_selected_base_group_meta_from_offsets(&offsets, 2),
                group_meta4: qwen35_selected_base_group_meta_from_offsets(&offsets, 4),
                group_meta8: qwen35_selected_base_group_meta_from_offsets(&offsets, 8),
                group_meta16: qwen35_selected_base_group_meta_from_offsets(&offsets, 16),
            }),
            device_route: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_prepare_selected_base_mixed_resident_device_slot_ptrs_by_token(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<PreparedQwen35SparseSlots, String> {
        let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
            gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
        )?;
        let n_expert = qwen35_selected_base_full_layer_expert_count(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )?;
        let mut expert_slots = vec![None; n_expert];
        let mut expert_slab_indices = vec![u32::MAX; n_expert];
        let mut selected_experts = 0u32;
        for slot in &offsets.slots {
            let expert = slot.expert_id as usize;
            if expert >= n_expert {
                return Err(format!(
                    "Qwen35 selected-base mixed device-slot expert id out of range: got {expert}, n_expert={n_expert}"
                ));
            }
            if expert_slots[expert].is_none() {
                expert_slots[expert] = Some(*slot);
                expert_slab_indices[expert] = selected_experts;
                selected_experts = selected_experts.checked_add(1).ok_or_else(|| {
                    "Qwen35 selected-base mixed device-slot selected expert count overflow"
                        .to_string()
                })?;
            }
        }

        let resident_role_ptrs = self.qwen35_selected_base_resident_role_ptrs(
            gate_all, up_all, down_all, &offsets, n_expert, down_quant, n_ff, n_embd,
        )?;
        let resident_roles = resident_role_ptrs.keys().copied().collect::<HashSet<_>>();

        let plan =
            qwen35_selected_base_mixed_resident_temp_plan(&offsets, n_expert, &resident_roles)?;
        let slab_dev = if plan.slab_bytes == 0 {
            0
        } else {
            self.compute_temp_slab_ptr(plan.slab_bytes)?
        };
        let copy_stream_upload = plan.slab_bytes > 0 && qwen35_selected_base_copy_stream_enabled();
        let upload_stream = if copy_stream_upload {
            self.copy_stream
        } else {
            self.stream
        };
        self.qwen35_upload_selected_base_temp_slab(
            slab_dev,
            plan.slab_bytes,
            &plan.uploads,
            gate_all,
            up_all,
            down_all,
            upload_stream,
            "mixed resident device slot",
        )?;

        let mut mixed_gate_ptrs = vec![0u64; n_expert];
        let mut mixed_up_ptrs = vec![0u64; n_expert];
        let mut mixed_down_ptrs = vec![0u64; n_expert];
        for expert in 0..n_expert {
            let Some(sources) = plan.expert_sources.get(expert).and_then(|sources| *sources) else {
                continue;
            };
            let slot = expert_slots[expert].ok_or_else(|| {
                format!("Qwen35 selected-base mixed device-slot missing slot: expert={expert}")
            })?;
            mixed_gate_ptrs[expert] = self.qwen35_selected_base_mixed_source_ptr(
                sources.gate,
                Qwen35SelectedBaseWeightRole::Gate,
                slot,
                &resident_role_ptrs,
                slab_dev,
            )?;
            mixed_up_ptrs[expert] = self.qwen35_selected_base_mixed_source_ptr(
                sources.up,
                Qwen35SelectedBaseWeightRole::Up,
                slot,
                &resident_role_ptrs,
                slab_dev,
            )?;
            mixed_down_ptrs[expert] = self.qwen35_selected_base_mixed_source_ptr(
                sources.down,
                Qwen35SelectedBaseWeightRole::Down,
                slot,
                &resident_role_ptrs,
                slab_dev,
            )?;
        }

        Ok(PreparedQwen35SparseSlots {
            gate_ptrs: Vec::new(),
            up_ptrs: Vec::new(),
            down_ptrs: Vec::new(),
            slot_count: None,
            temp_slab_ptrs: if plan.slab_bytes == 0 {
                Vec::new()
            } else {
                vec![slab_dev]
            },
            copy_stream_upload,
            device_slot_ptrs: Some(PreparedQwen35DeviceSlotPtrs {
                expert_ids: expert_ids.to_vec(),
                expert_slab_indices,
                gate_base: 0,
                up_base: 0,
                down_base: 0,
                gate_expert_bytes: offsets.gate_bytes_per_expert,
                up_expert_bytes: offsets.up_bytes_per_expert,
                down_expert_bytes: offsets.down_bytes_per_expert,
                selected_upload_calls: plan.uploads.len(),
                selected_upload_bytes: plan.uploads.iter().map(|upload| upload.bytes).sum(),
                mixed_expert_ptrs: Some(PreparedQwen35MixedExpertPtrs {
                    gate_ptrs: mixed_gate_ptrs,
                    up_ptrs: mixed_up_ptrs,
                    down_ptrs: mixed_down_ptrs,
                }),
                group_meta2: qwen35_selected_base_group_meta_from_offsets(&offsets, 2),
                group_meta4: qwen35_selected_base_group_meta_from_offsets(&offsets, 4),
                group_meta8: qwen35_selected_base_group_meta_from_offsets(&offsets, 8),
                group_meta16: qwen35_selected_base_group_meta_from_offsets(&offsets, 16),
            }),
            group_meta: None,
            device_route: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_selected_base_existing_resident_role_count_by_token(
        &self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
    ) -> Result<usize, String> {
        let layer_key = qwen35_moe_layer_key(gate_all, up_all, down_all, down_quant, n_ff, n_embd);
        let full_layer_hit = self.resident_moe_layers.contains_key(&layer_key);
        if !full_layer_hit && self.resident_q4k.is_empty() {
            return Ok(0);
        }

        let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
            gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
        )?;
        let n_expert = qwen35_selected_base_full_layer_expert_count(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )?;
        let mut seen_roles = HashSet::new();
        let mut hits = 0usize;
        for slot in &offsets.slots {
            let expert_id = slot.expert_id as usize;
            if expert_id >= n_expert {
                return Err(format!(
                    "Qwen35 selected-base existing resident expert id out of range: got {expert_id}, n_expert={n_expert}"
                ));
            }
            for role in [
                Qwen35SelectedBaseWeightRole::Gate,
                Qwen35SelectedBaseWeightRole::Up,
                Qwen35SelectedBaseWeightRole::Down,
            ] {
                let role_key = Qwen35SelectedBaseResidentRole { role, expert_id };
                if !seen_roles.insert(role_key) {
                    continue;
                }
                if full_layer_hit {
                    hits = hits.checked_add(1).ok_or_else(|| {
                        "Qwen35 selected-base existing resident hit count overflow".to_string()
                    })?;
                    continue;
                }
                let upload = qwen35_selected_base_role_upload(role, *slot, &offsets, 0);
                let weights = qwen35_selected_base_upload_source(
                    &upload,
                    gate_all,
                    up_all,
                    down_all,
                    "existing resident role",
                )?;
                if self.resident_q4k.contains_key(&q4k_resident_key(weights)) {
                    hits = hits.checked_add(1).ok_or_else(|| {
                        "Qwen35 selected-base existing resident hit count overflow".to_string()
                    })?;
                }
            }
        }
        Ok(hits)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_admit_selected_base_resident_pages_by_token(
        &mut self,
        gate_all: &[u8],
        up_all: &[u8],
        down_all: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        token_count: usize,
    ) -> Result<Qwen35SelectedBaseResidentAdmissionStats, String> {
        if !qwen35_selected_base_resident_admission_enabled() {
            return Ok(Qwen35SelectedBaseResidentAdmissionStats::default());
        }
        if self.qwen35_selected_base_has_exact_resident_moe_layer(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        ) {
            return Ok(Qwen35SelectedBaseResidentAdmissionStats::default());
        }
        if !qwen35_selected_base_resident_admission_token_window_allows(token_count)? {
            if std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_TRACE")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "[cuda-selected-resident-admit] skipped=token_window tokens={token_count}"
                );
            }
            return Ok(Qwen35SelectedBaseResidentAdmissionStats {
                skipped_by_token_window: true,
                ..Qwen35SelectedBaseResidentAdmissionStats::default()
            });
        }
        let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
            gate_all, up_all, down_all, expert_ids, down_quant, n_ff, n_embd,
        )?;
        let n_expert = qwen35_selected_base_full_layer_expert_count(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )?;
        let bases = Qwen35SelectedExpertBases {
            gate_all,
            up_all,
            down_all,
            gate_bytes_per_expert: offsets.gate_bytes_per_expert,
            up_bytes_per_expert: offsets.up_bytes_per_expert,
            down_bytes_per_expert: offsets.down_bytes_per_expert,
            n_expert,
        };
        let candidates = qwen35_resident_expert_page_candidates(
            0,
            &bases,
            expert_ids,
            route_weights,
            token_count,
            12,
            12,
            down_quant,
        )?;

        self.raise_resident_q4k_limit_for_qwen35_target_decode()?;
        let (free_bytes, total_bytes) = unsafe { self.api.mem_get_info() }?;
        let mib = 1024 * 1024;
        let total_mib = total_bytes / mib;
        let free_mib = free_bytes / mib;
        let reserve_mib = q4k_resident_configured_reserve_mib(total_mib, false)?;
        let budget = qwen35_resident_expert_page_budget(
            total_mib,
            free_mib,
            reserve_mib,
            self.resident_q4k_limit,
            self.resident_q4k_bytes,
        );
        let cost_gate_enabled = qwen35_selected_base_resident_admission_cost_gate_enabled();
        let plan_budget_bytes = if cost_gate_enabled {
            budget.evicting_budget_bytes
        } else {
            budget.budget_bytes
        };
        let plan = qwen35_resident_expert_page_plan(&candidates, plan_budget_bytes as u64);
        let mut stats = Qwen35SelectedBaseResidentAdmissionStats {
            candidate_pages: candidates.len(),
            selected_pages: plan.selected.len(),
            budget_bytes: plan_budget_bytes as u64,
            selected_bytes: plan.selected_bytes,
            ..Qwen35SelectedBaseResidentAdmissionStats::default()
        };

        let selected_sources = plan
            .selected
            .iter()
            .map(|candidate| {
                let weights =
                    qwen35_selected_base_candidate_source(candidate, gate_all, up_all, down_all)?;
                let key = q4k_resident_key(weights);
                Ok((weights, key, self.resident_q4k.contains_key(&key)))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let already_resident = selected_sources
            .iter()
            .map(|(_, _, already)| *already)
            .collect::<Vec<_>>();
        let protected_resident_keys = selected_sources
            .iter()
            .filter_map(|(_, key, already)| already.then_some(*key))
            .collect::<HashSet<_>>();
        let scalar_future_hits = qwen35_selected_base_resident_admission_future_hits()?;
        let (admission_cost, future_hits_for_trace) =
            if qwen35_selected_base_resident_admission_history_enabled() {
                let page_future_hits = qwen35_resident_expert_page_source_future_hits_and_observe(
                    &mut self.qwen35_selected_base_admission_history,
                    selected_sources.iter().map(|(_, key, _)| *key),
                );
                let future_hits_for_trace = page_future_hits.iter().copied().max().unwrap_or(0);
                let initial_cost = qwen35_resident_expert_page_admission_cost_with_future_hits(
                    &plan.selected,
                    &already_resident,
                    &page_future_hits,
                )?;
                let eviction_cost_bytes = if cost_gate_enabled {
                    self.resident_q4k_eviction_cost_bytes_for_incoming(
                        initial_cost.new_admission_bytes as usize,
                        &protected_resident_keys,
                    )
                    .unwrap_or(u64::MAX)
                } else {
                    0
                };
                let admission_cost =
                    qwen35_resident_expert_page_admission_cost_with_future_hits_and_eviction_cost(
                        &plan.selected,
                        &already_resident,
                        &page_future_hits,
                        eviction_cost_bytes,
                    )?;
                (admission_cost, future_hits_for_trace)
            } else {
                let initial_cost = qwen35_resident_expert_page_admission_cost(
                    &plan.selected,
                    &already_resident,
                    scalar_future_hits,
                )?;
                let eviction_cost_bytes = if cost_gate_enabled {
                    self.resident_q4k_eviction_cost_bytes_for_incoming(
                        initial_cost.new_admission_bytes as usize,
                        &protected_resident_keys,
                    )
                    .unwrap_or(u64::MAX)
                } else {
                    0
                };
                let page_future_hits = vec![scalar_future_hits; plan.selected.len()];
                let admission_cost =
                    qwen35_resident_expert_page_admission_cost_with_future_hits_and_eviction_cost(
                        &plan.selected,
                        &already_resident,
                        &page_future_hits,
                        eviction_cost_bytes,
                    )?;
                (admission_cost, scalar_future_hits)
            };
        stats.already_resident_pages = already_resident.iter().filter(|&&already| already).count();
        stats.already_resident_bytes = admission_cost.already_resident_bytes;
        stats.admission_cost_bytes = admission_cost.new_admission_bytes;
        stats.eviction_cost_bytes = admission_cost.eviction_cost_bytes;
        stats.predicted_saved_bytes = admission_cost.predicted_saved_bytes;
        stats.net_saved_bytes = admission_cost.net_saved_bytes;

        if cost_gate_enabled && !admission_cost.profitable {
            stats.skipped_by_cost_gate = true;
            for (_, key, already) in &selected_sources {
                if *already {
                    self.touch_resident_q4k(*key);
                }
            }
            if std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_TRACE")
                .ok()
                .as_deref()
                == Some("1")
            {
                eprintln!(
                    "[cuda-selected-resident-admit] skipped=cost_gate candidates={} selected={} future_hits={} cost_mb={:.2} eviction_cost_mb={:.2} predicted_saved_mb={:.2} net_saved_mb={:.2} already={} already_mb={:.2}",
                    stats.candidate_pages,
                    stats.selected_pages,
                    future_hits_for_trace,
                    stats.admission_cost_bytes as f64 / (1024.0 * 1024.0),
                    stats.eviction_cost_bytes as f64 / (1024.0 * 1024.0),
                    stats.predicted_saved_bytes as f64 / (1024.0 * 1024.0),
                    stats.net_saved_bytes as f64 / (1024.0 * 1024.0),
                    stats.already_resident_pages,
                    stats.already_resident_bytes as f64 / (1024.0 * 1024.0),
                );
            }
            return Ok(stats);
        }

        for (_, key, already) in &selected_sources {
            if *already {
                self.touch_resident_q4k(*key);
            }
        }
        for (weights, _, already) in selected_sources {
            if already {
                continue;
            }
            if self.preload_resident_q4k_weight_slice(weights)? {
                stats.admitted_pages += 1;
                stats.admitted_bytes = stats.admitted_bytes.saturating_add(weights.len() as u64);
            }
        }

        if std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_RESIDENT_ADMISSION_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[cuda-selected-resident-admit] candidates={} budget_mb={:.2} selected={} selected_mb={:.2} admitted={} already={} admitted_mb={:.2} future_hits={} cost_mb={:.2} eviction_cost_mb={:.2} predicted_saved_mb={:.2} net_saved_mb={:.2}",
                stats.candidate_pages,
                stats.budget_bytes as f64 / (1024.0 * 1024.0),
                stats.selected_pages,
                stats.selected_bytes as f64 / (1024.0 * 1024.0),
                stats.admitted_pages,
                stats.already_resident_pages,
                stats.admitted_bytes as f64 / (1024.0 * 1024.0),
                future_hits_for_trace,
                stats.admission_cost_bytes as f64 / (1024.0 * 1024.0),
                stats.eviction_cost_bytes as f64 / (1024.0 * 1024.0),
                stats.predicted_saved_bytes as f64 / (1024.0 * 1024.0),
                stats.net_saved_bytes as f64 / (1024.0 * 1024.0)
            );
        }

        Ok(stats)
    }

    pub(in crate::runtime) fn qwen35_prepare_device_sparse_route_by_token(
        &mut self,
        route_weights: &[f32],
        token_ids: &[u32],
    ) -> Result<PreparedQwen35DeviceSparseRoute, String> {
        let slots = route_weights.len();
        if token_ids.len() != slots {
            return Err(format!(
                "Qwen35 device sparse route/token length mismatch: route={} token={}",
                route_weights.len(),
                token_ids.len()
            ));
        }
        if slots == 0 {
            return Ok(PreparedQwen35DeviceSparseRoute {
                route_weights_dev: 0,
                token_ids_dev: 0,
                slots,
            });
        }
        let route_weights_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
        let token_ids_dev = self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?;
        unsafe {
            self.api.memcpy_htod_async(
                route_weights_dev,
                route_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(route_weights),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                token_ids_dev,
                token_ids.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(token_ids),
                self.stream,
            )?;
        }
        Ok(PreparedQwen35DeviceSparseRoute {
            route_weights_dev,
            token_ids_dev,
            slots,
        })
    }

    fn qwen35_stage_selected_sparse_down_meta(
        &mut self,
        payload: Qwen35SelectedSparseDownMetaStagingPayload<'_>,
        group_meta_dev: u64,
    ) -> Result<(), String> {
        match payload {
            Qwen35SelectedSparseDownMetaStagingPayload::TokenMajor {
                token_offsets,
                slot_indices,
            } => {
                let token_offsets_bytes = std::mem::size_of_val(token_offsets);
                unsafe {
                    self.api.memcpy_htod_async(
                        group_meta_dev,
                        token_offsets.as_ptr().cast::<libc::c_void>(),
                        token_offsets_bytes,
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        group_meta_dev + token_offsets_bytes as u64,
                        slot_indices.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(slot_indices),
                        self.stream,
                    )?;
                }
            }
            Qwen35SelectedSparseDownMetaStagingPayload::RunTile(meta) => unsafe {
                self.api.memcpy_htod_async(
                    group_meta_dev,
                    meta.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(meta),
                    self.stream,
                )?;
            },
            Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(down_group_meta) => unsafe {
                self.api.memcpy_htod_async(
                    group_meta_dev,
                    down_group_meta.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(down_group_meta),
                    self.stream,
                )?;
            },
            Qwen35SelectedSparseDownMetaStagingPayload::ReuseExistingDownMeta => {
                // The pack kernel already needs and leaves down_group_meta in this workspace.
            }
            Qwen35SelectedSparseDownMetaStagingPayload::None => {}
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_launch_selected_sparse_gate_up(
        &mut self,
        payload: Qwen35SelectedSparseGateUpLaunchPayload,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        n_ff: usize,
        n_embd: usize,
        slots: usize,
        input_dev: u64,
        gate_dev: u64,
        up_dev: u64,
        down_group_meta: &[u32],
    ) -> Result<(), String> {
        match payload {
            Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group8 {
                packed_dev,
                pack_group_offsets_dev,
                pack_group_count,
            } => {
                self.launch_selected_q4k_gate_up_silu_pack4_f32_by_token_group8_to_dev(
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    pack_group_offsets_dev,
                    n_ff,
                    pack_group_count,
                    n_embd / 256,
                    n_ff / 256,
                    input_dev,
                    packed_dev,
                )?;
            }
            Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group4 {
                packed_dev,
                down_group_count,
                reload_down_group_meta,
            } => {
                if reload_down_group_meta {
                    unsafe {
                        self.api.memcpy_htod_async(
                            group_meta_dev,
                            down_group_meta.as_ptr().cast::<libc::c_void>(),
                            std::mem::size_of_val(down_group_meta),
                            self.stream,
                        )?;
                    }
                }
                self.launch_selected_q4k_gate_up_silu_pack4_f32_by_token_group4_to_dev(
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_ff,
                    down_group_count,
                    n_embd / 256,
                    n_ff / 256,
                    input_dev,
                    packed_dev,
                )?;
            }
            Qwen35SelectedSparseGateUpLaunchPayload::FusedSiluGroup8 { group_count } => {
                self.launch_selected_q4k_gate_up_silu_by_token_group8_to_dev(
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_ff,
                    group_count,
                    n_embd / 256,
                    input_dev,
                    gate_dev,
                    up_dev,
                )?;
            }
            Qwen35SelectedSparseGateUpLaunchPayload::SeparateUngrouped => {
                self.launch_selected_q4k_gate_up_gemv_by_token_to_dev(
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    token_ids_dev,
                    n_ff,
                    slots,
                    n_embd / 256,
                    input_dev,
                    gate_dev,
                    up_dev,
                )?;
            }
            Qwen35SelectedSparseGateUpLaunchPayload::SeparateGrouped { group_count } => {
                self.launch_selected_q4k_gate_up_gemv_by_token_group4_to_dev(
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_ff,
                    group_count,
                    n_embd / 256,
                    input_dev,
                    gate_dev,
                    up_dev,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_launch_selected_sparse_silu(
        &mut self,
        payload: Qwen35SelectedSparseSiluLaunchPayload,
        gate_dev: u64,
        up_dev: u64,
        group_meta_dev: u64,
        n_ff: usize,
        slots: usize,
        down_group_meta: &[u32],
    ) -> Result<(), String> {
        match payload {
            Qwen35SelectedSparseSiluLaunchPayload::None => {}
            Qwen35SelectedSparseSiluLaunchPayload::Q8 { qs_dev, ds_dev } => {
                self.launch_silu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, slots * n_ff)?;
            }
            Qwen35SelectedSparseSiluLaunchPayload::Pack4F32Group4 {
                packed_dev,
                group_count,
                reload_down_group_meta,
            } => {
                if reload_down_group_meta {
                    unsafe {
                        self.api.memcpy_htod_async(
                            group_meta_dev,
                            down_group_meta.as_ptr().cast::<libc::c_void>(),
                            std::mem::size_of_val(down_group_meta),
                            self.stream,
                        )?;
                    }
                }
                self.launch_silu_mul_group4_pack_f32(
                    gate_dev,
                    up_dev,
                    packed_dev,
                    group_meta_dev,
                    group_count,
                    n_ff / 256,
                )?;
            }
            Qwen35SelectedSparseSiluLaunchPayload::Plain => {
                self.launch_silu_mul(gate_dev, up_dev, slots * n_ff)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_launch_selected_sparse_down(
        &mut self,
        payload: Qwen35SelectedSparseDownLaunchPayload<'_>,
        down_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        n_embd: usize,
        token_count: usize,
        slots: usize,
        n_ff: usize,
        gate_dev: u64,
        route_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        match payload {
            Qwen35SelectedSparseDownLaunchPayload::Q6TokenMajor {
                token_offsets_bytes,
            } => self.launch_selected_q6k_down_accum_token_major_warp4(
                down_ptrs_dev,
                group_meta_dev,
                group_meta_dev + token_offsets_bytes as u64,
                n_embd,
                token_count,
                n_ff / 256,
                gate_dev,
                route_dev,
                output_dev,
            )?,
            Qwen35SelectedSparseDownLaunchPayload::Q6Q8DotGroup4 {
                qs_dev,
                ds_dev,
                group_count,
            } => self.launch_selected_q6k_down_accum_by_token_group4_q8dot(
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                group_count,
                n_ff / 256,
                qs_dev,
                ds_dev,
                route_dev,
                output_dev,
            )?,
            Qwen35SelectedSparseDownLaunchPayload::Q6Pack4F32Group4 {
                packed_dev,
                group_count,
            } => self.launch_selected_q6k_down_accum_by_token_group4_pack4_f32_warp4(
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                group_count,
                n_ff / 256,
                packed_dev,
                route_dev,
                output_dev,
            )?,
            Qwen35SelectedSparseDownLaunchPayload::Q6RunTiled4 { run_count } => self
                .launch_selected_q6k_down_accum_run_tiled4_warp4(
                    down_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    run_count,
                    n_ff / 256,
                    gate_dev,
                    route_dev,
                    output_dev,
                )?,
            Qwen35SelectedSparseDownLaunchPayload::Q6RunBatchedRef { run_count } => self
                .launch_selected_q6k_down_accum_run_batched_ref_warp4(
                    down_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    run_count,
                    n_ff / 256,
                    gate_dev,
                    route_dev,
                    output_dev,
                )?,
            Qwen35SelectedSparseDownLaunchPayload::Q6RunBatched8 { run_count } => self
                .launch_selected_q6k_down_accum_run_batched8_warp4(
                    down_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    run_count,
                    n_ff / 256,
                    gate_dev,
                    route_dev,
                    output_dev,
                )?,
            Qwen35SelectedSparseDownLaunchPayload::Q6Full4Split {
                matching_meta,
                other_meta,
            } => {
                if !matching_meta.is_empty() {
                    unsafe {
                        self.api.memcpy_htod_async(
                            group_meta_dev,
                            matching_meta.as_ptr().cast::<libc::c_void>(),
                            std::mem::size_of_val(matching_meta),
                            self.stream,
                        )?;
                    }
                    self.launch_selected_q6k_down_accum_by_token_group4_full_warp4(
                        down_ptrs_dev,
                        token_ids_dev,
                        group_meta_dev,
                        n_embd,
                        matching_meta.len() / 2,
                        n_ff / 256,
                        gate_dev,
                        route_dev,
                        output_dev,
                    )?;
                }
                if !other_meta.is_empty() {
                    unsafe {
                        self.api.memcpy_htod_async(
                            group_meta_dev,
                            other_meta.as_ptr().cast::<libc::c_void>(),
                            std::mem::size_of_val(other_meta),
                            self.stream,
                        )?;
                    }
                    self.launch_selected_down_accum_by_token_group4(
                        "rnb_q6k_selected_down_accum_by_token_group4",
                        down_ptrs_dev,
                        token_ids_dev,
                        group_meta_dev,
                        n_embd,
                        other_meta.len() / 2,
                        n_ff / 256,
                        gate_dev,
                        route_dev,
                        output_dev,
                    )?;
                }
            }
            Qwen35SelectedSparseDownLaunchPayload::Group4 {
                kernel,
                group_count,
            } => self.launch_selected_down_accum_by_token_group4(
                kernel,
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                group_count,
                n_ff / 256,
                gate_dev,
                route_dev,
                output_dev,
            )?,
            Qwen35SelectedSparseDownLaunchPayload::Warp4 { kernel } => self
                .launch_selected_down_accum_by_token_warp4(
                    kernel,
                    down_ptrs_dev,
                    token_ids_dev,
                    n_embd,
                    slots,
                    n_ff / 256,
                    gate_dev,
                    route_dev,
                    output_dev,
                )?,
            Qwen35SelectedSparseDownLaunchPayload::Scalar { kernel } => self
                .launch_selected_down_accum_by_token(
                    kernel,
                    down_ptrs_dev,
                    token_ids_dev,
                    n_embd,
                    slots,
                    n_ff / 256,
                    gate_dev,
                    route_dev,
                    output_dev,
                )?,
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_launch_selected_sparse_compound_graph(
        &mut self,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        down_ptrs_dev: u64,
        token_ids_dev: u64,
        gate_meta_dev: u64,
        pack_group_offsets_dev: u64,
        down_meta_dev: u64,
        n_ff: usize,
        n_embd: usize,
        graph_zero_output: bool,
        output_len: usize,
        input_dev: u64,
        packed_dev: u64,
        route_dev: u64,
        output_dev: u64,
        pack_group_count: usize,
        down_group_count: usize,
    ) -> Result<(), String> {
        let key = Qwen35CompoundGraphKey {
            n_ff,
            n_embd,
            zero_output: graph_zero_output,
            output_len,
            pack_group_count,
            down_group_count,
            input_dev,
            packed_dev,
            output_dev,
            gate_ptrs_dev,
            up_ptrs_dev,
            down_ptrs_dev,
            token_ids_dev,
            gate_meta_dev,
            pack_group_offsets_dev,
            down_meta_dev,
            route_dev,
        };
        if let Some(graph) = self.qwen35_compound_graphs.get(&key) {
            return unsafe {
                self.api
                    .graph_launch(graph.exec as *mut libc::c_void, self.stream)
            };
        }

        self.ensure_q4k_gemv_module()?;
        unsafe {
            self.api.stream_begin_capture(self.stream)?;
        }
        let capture_result = (|| {
            if graph_zero_output {
                self.launch_zero_f32(output_dev, output_len)?;
            }
            self.launch_selected_q4k_gate_up_silu_pack4_f32_by_token_group8_to_dev(
                gate_ptrs_dev,
                up_ptrs_dev,
                token_ids_dev,
                gate_meta_dev,
                pack_group_offsets_dev,
                n_ff,
                pack_group_count,
                n_embd / 256,
                n_ff / 256,
                input_dev,
                packed_dev,
            )?;
            self.launch_selected_q6k_down_accum_by_token_group4_pack4_f32_warp4(
                down_ptrs_dev,
                token_ids_dev,
                down_meta_dev,
                n_embd,
                down_group_count,
                n_ff / 256,
                packed_dev,
                route_dev,
                output_dev,
            )
        })();
        if let Err(err) = capture_result {
            unsafe {
                let _ = self.api.stream_end_capture(self.stream);
            }
            return Err(err);
        }
        let graph = unsafe { self.api.stream_end_capture(self.stream)? };
        let exec = unsafe { self.api.graph_instantiate(graph)? };
        self.qwen35_compound_graphs.insert(
            key,
            SparseMoeGraph {
                graph: graph as usize,
                exec: exec as usize,
            },
        );
        let graph = self
            .qwen35_compound_graphs
            .get(&key)
            .ok_or_else(|| "missing Qwen35 selected sparse compound CUDA graph".to_string())?;
        unsafe {
            self.api
                .graph_launch(graph.exec as *mut libc::c_void, self.stream)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_launch_selected_sparse_compound_reference(
        &mut self,
        payload: Qwen35SelectedSparseCompoundRunnerPayload<'_>,
        gate_ptrs_dev: u64,
        up_ptrs_dev: u64,
        down_ptrs_dev: u64,
        token_ids_dev: u64,
        group_meta_dev: u64,
        graph_down_meta_dev: Option<u64>,
        n_ff: usize,
        n_embd: usize,
        graph_zero_output: bool,
        output_len: usize,
        input_dev: u64,
        route_dev: u64,
        output_dev: u64,
        sync_down_copy_overlap: bool,
        trace_kernel: bool,
    ) -> Result<Qwen35SelectedSparseCompoundRunnerTimings, String> {
        let mut timings = Qwen35SelectedSparseCompoundRunnerTimings::default();
        match payload {
            Qwen35SelectedSparseCompoundRunnerPayload::Q4GateUpGroup8Q6DownPack4 {
                packed_dev,
                pack_group_offsets_dev,
                pack_group_count,
                down_meta,
                down_group_count,
            } => {
                if tuning::qwen35_selected_sparse_compound_graph_enabled() && !trace_kernel {
                    if sync_down_copy_overlap {
                        unsafe { self.api.stream_synchronize(self.copy_stream)? };
                    }
                    let down_meta_dev = match down_meta {
                        Qwen35SelectedSparseDownMetaStagingPayload::ReuseExistingDownMeta => {
                            group_meta_dev
                        }
                        Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(_) => {
                            graph_down_meta_dev.ok_or_else(|| {
                                "Qwen35 selected sparse compound graph missing disjoint down-meta workspace".to_string()
                            })?
                        }
                        other => {
                            return Err(format!(
                                "Qwen35 selected sparse compound graph unsupported down-meta payload: {other:?}"
                            ));
                        }
                    };
                    self.qwen35_stage_selected_sparse_down_meta(down_meta, down_meta_dev)?;
                    self.qwen35_launch_selected_sparse_compound_graph(
                        gate_ptrs_dev,
                        up_ptrs_dev,
                        down_ptrs_dev,
                        token_ids_dev,
                        group_meta_dev,
                        pack_group_offsets_dev,
                        down_meta_dev,
                        n_ff,
                        n_embd,
                        graph_zero_output,
                        output_len,
                        input_dev,
                        packed_dev,
                        route_dev,
                        output_dev,
                        pack_group_count,
                        down_group_count,
                    )?;
                    return Ok(timings);
                }

                let kernel_t0 = trace_kernel.then(std::time::Instant::now);
                self.launch_selected_q4k_gate_up_silu_pack4_f32_by_token_group8_to_dev(
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    pack_group_offsets_dev,
                    n_ff,
                    pack_group_count,
                    n_embd / 256,
                    n_ff / 256,
                    input_dev,
                    packed_dev,
                )?;
                if let Some(t0) = kernel_t0 {
                    self.stream_synchronize()?;
                    timings.gate_up_ms = t0.elapsed().as_micros() as f64 / 1000.0;
                }

                let kernel_t0 = trace_kernel.then(std::time::Instant::now);
                if let Some(t0) = kernel_t0 {
                    self.stream_synchronize()?;
                    timings.silu_ms = t0.elapsed().as_micros() as f64 / 1000.0;
                }

                if sync_down_copy_overlap {
                    unsafe { self.api.stream_synchronize(self.copy_stream)? };
                }
                self.qwen35_stage_selected_sparse_down_meta(down_meta, group_meta_dev)?;

                let kernel_t0 = trace_kernel.then(std::time::Instant::now);
                self.launch_selected_q6k_down_accum_by_token_group4_pack4_f32_warp4(
                    down_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    down_group_count,
                    n_ff / 256,
                    packed_dev,
                    route_dev,
                    output_dev,
                )?;
                if let Some(t0) = kernel_t0 {
                    self.stream_synchronize()?;
                    timings.down_ms = t0.elapsed().as_micros() as f64 / 1000.0;
                }
            }
        }
        Ok(timings)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_by_token_to_dev_prepared(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        route_weights: &[f32],
        token_ids: &[u32],
        token_count: usize,
        down_quant: u32,
        n_ff: usize,
        n_embd: usize,
        input_dev: u64,
        output_dev: u64,
        zero_output: bool,
        prefer_group2_down: bool,
        expert_ids: Option<&[u32]>,
        prepared_slots: Option<PreparedQwen35SparseSlots>,
    ) -> Result<(), String> {
        let trace_cache = std::env::var("RNB_CUDA_CACHE_TRACE").ok().as_deref() == Some("1");
        let trace_phase = std::env::var("RNB_CUDA_PHASE_TRACE").ok().as_deref() == Some("1");
        let trace_kernel = std::env::var("RNB_CUDA_KERNEL_TRACE").ok().as_deref() == Some("1");
        let trace_t0 = (trace_cache || trace_phase).then(std::time::Instant::now);
        let trace_before = trace_cache.then(cache_snapshot);
        let mut weight_ptr_ms = 0.0f64;
        let mut setup_h2d_ms = 0.0f64;
        let mut kernels_ms = 0.0f64;
        let dtoh_ms = 0.0f64;
        let mut zero_ms = 0.0f64;
        let mut gate_up_ms = 0.0f64;
        let mut silu_ms = 0.0f64;
        let mut down_ms = 0.0f64;
        let phase_t0 = trace_phase.then(std::time::Instant::now);
        let PreparedQwen35SparseSlots {
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            slot_count,
            temp_slab_ptrs,
            copy_stream_upload,
            device_slot_ptrs,
            group_meta,
            device_route,
        } = match prepared_slots {
            Some(prepared) => prepared,
            None => self.qwen35_prepare_sparse_slots_by_token(
                gate_weights,
                up_weights,
                down_weights,
                route_weights,
                token_count,
                false,
            )?,
        };
        let direct_selected_base_sparse = device_slot_ptrs.is_some()
            && gate_weights.is_empty()
            && up_weights.is_empty()
            && down_weights.is_empty();
        let slots = if direct_selected_base_sparse {
            qwen35_sparse_slot_count(
                gate_weights.len(),
                Some(
                    device_slot_ptrs
                        .as_ref()
                        .expect("direct selected-base sparse has device slot ptrs")
                        .expert_ids
                        .len(),
                ),
                slot_count,
            )
        } else {
            qwen35_sparse_slot_count(gate_weights.len(), None, slot_count)
        };
        if route_weights.len() != slots || token_ids.len() != slots {
            return Err(format!(
                "Qwen35 sparse route/token length mismatch: slots={slots} route={} token={}",
                route_weights.len(),
                token_ids.len()
            ));
        }
        self.raise_resident_q4k_limit_for_qwen35_target_decode()?;
        let gate_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let output_len = token_count * n_embd;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let (route_dev, token_ids_dev, route_from_device) = match device_route.as_ref() {
            Some(route) => {
                if route.slots != slots {
                    return Err(format!(
                        "Qwen35 device sparse route slot mismatch: route={} slots={slots}",
                        route.slots
                    ));
                }
                (route.route_weights_dev, route.token_ids_dev, true)
            }
            None => (
                self.compute_route_ptr(std::mem::size_of_val(route_weights))?,
                self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?,
                false,
            ),
        };
        let mut boundary_stats = device_slot_ptrs.as_ref().map(|device| {
            qwen35_selected_sparse_boundary_stats_from_device_slots(
                slots,
                device,
                route_from_device,
            )
        });
        let group_gate_up = std::env::var("RNB_CUDA_GROUP4_GATE_UP").ok().as_deref() != Some("0");
        if let Some(expert_ids) = expert_ids {
            if expert_ids.len() != slots {
                return Err(format!(
                    "Qwen35 sparse expert id length mismatch: got {}, expected {slots}",
                    expert_ids.len()
                ));
            }
        }
        let group8_gate_up = group_gate_up
            && std::env::var("RNB_CUDA_GROUP8_GATE_UP_WARP4")
                .ok()
                .as_deref()
                != Some("0");
        let group16_gate_up = group8_gate_up
            && std::env::var("RNB_CUDA_GROUP16_GATE_UP_WARP4")
                .ok()
                .as_deref()
                == Some("1");
        let gate_up_max_group = if group_gate_up {
            Some(if group16_gate_up {
                16
            } else if group8_gate_up {
                8
            } else {
                4
            })
        } else {
            None
        };
        let gate_up_group_meta = if let Some(max_group) = gate_up_max_group {
            if direct_selected_base_sparse {
                device_slot_ptrs
                    .as_ref()
                    .expect("direct selected-base sparse has device slot ptrs")
                    .group_meta_for_max_group(max_group)?
                    .to_vec()
            } else if let Some(group_meta) = group_meta.as_ref() {
                group_meta.group_meta_for_max_group(max_group)?.to_vec()
            } else {
                match expert_ids.filter(|_| qwen35_group_meta_from_ids_enabled()) {
                    Some(expert_ids) => build_group_meta_from_ids(expert_ids, max_group),
                    None => build_group_meta(gate_weights, up_weights, max_group),
                }
            }
        } else {
            Vec::new()
        };
        let group2_down = prefer_group2_down || tuning::group2_down_warp4_enabled();
        let group8_down = !group2_down
            && std::env::var("RNB_CUDA_GROUP8_DOWN_WARP4").ok().as_deref() == Some("1");
        let down_token_major = down_quant == 14 && tuning::qwen35_down_token_major_enabled();
        let down_token_major_plan = if down_token_major {
            Some(qwen35_down_token_major_plan(token_ids, token_count)?)
        } else {
            None
        };
        let down_max_group = if group_gate_up {
            Some(if group2_down {
                2
            } else if group8_down {
                8
            } else {
                4
            })
        } else {
            None
        };
        let down_group_meta = if let Some(max_group) = down_max_group {
            if direct_selected_base_sparse {
                device_slot_ptrs
                    .as_ref()
                    .expect("direct selected-base sparse has device slot ptrs")
                    .group_meta_for_max_group(max_group)?
                    .to_vec()
            } else if let Some(group_meta) = group_meta.as_ref() {
                group_meta.group_meta_for_max_group(max_group)?.to_vec()
            } else {
                match expert_ids.filter(|_| qwen35_group_meta_from_ids_enabled()) {
                    Some(expert_ids) => build_group_meta_from_ids(expert_ids, max_group),
                    None => build_group_meta(gate_weights, up_weights, max_group),
                }
            }
        } else {
            Vec::new()
        };
        let down_token_major_bytes = down_token_major_plan
            .as_ref()
            .map(|plan| {
                std::mem::size_of_val(plan.token_offsets.as_slice())
                    + std::mem::size_of_val(plan.slot_indices.as_slice())
            })
            .unwrap_or(0);
        let group4_down = std::env::var("RNB_CUDA_GROUP4_DOWN").ok().as_deref() != Some("0")
            && !down_group_meta.is_empty();
        let q4_down_group4 = down_quant == 12
            && group4_down
            && !group2_down
            && !group8_down
            && tuning::qwen35_q4_down_group4_enabled();
        let q6_down_q8dot = down_quant == 14
            && group4_down
            && !down_token_major
            && !group2_down
            && !group8_down
            && tuning::qwen35_q6_down_q8dot_enabled();
        let q6_down_pack4_f32 = down_quant == 14
            && group4_down
            && !down_token_major
            && !group2_down
            && !group8_down
            && !q6_down_q8dot
            && tuning::qwen35_q6_down_pack4_f32_enabled();
        let q4_gate_up_silu_pack4_f32 = q6_down_pack4_f32
            && tuning::qwen35_q4_gate_up_silu_pack4_f32_enabled()
            && !down_group_meta.is_empty()
            && n_ff % 256 == 0;
        let q4_gate_up_silu_pack4_group8 = q4_gate_up_silu_pack4_f32
            && group8_gate_up
            && !group16_gate_up
            && !gate_up_group_meta.is_empty();
        let q4_gate_up_silu_pack4_group_offsets = if q4_gate_up_silu_pack4_group8 {
            let offsets = qwen35_pack4_group_offsets_for_down_group_meta(
                &gate_up_group_meta,
                &down_group_meta,
            )?;
            Some(offsets)
        } else {
            None
        };
        let gate_up_group_meta_bytes = if gate_up_group_meta.is_empty() {
            0
        } else {
            std::mem::size_of_val(gate_up_group_meta.as_slice())
        };
        let q4_gate_up_silu_pack4_offsets_bytes = q4_gate_up_silu_pack4_group_offsets
            .as_ref()
            .map(|offsets| std::mem::size_of_val(offsets.as_slice()))
            .unwrap_or(0);
        let q4_gate_up_silu_pack4_workspace_bytes = if q4_gate_up_silu_pack4_group_offsets.is_some()
        {
            gate_up_group_meta_bytes + q4_gate_up_silu_pack4_offsets_bytes
        } else {
            0
        };
        let q6_down_run_tiled4 = down_quant == 14
            && group4_down
            && !down_token_major
            && !group2_down
            && !group8_down
            && !q6_down_q8dot
            && !q6_down_pack4_f32
            && !direct_selected_base_sparse
            && expert_ids.is_some()
            && tuning::qwen35_q6_down_run_tiled4_enabled();
        let q6_down_run_batched8 = down_quant == 14
            && group4_down
            && !down_token_major
            && !group2_down
            && !group8_down
            && !q6_down_q8dot
            && !q6_down_pack4_f32
            && !q6_down_run_tiled4
            && !direct_selected_base_sparse
            && expert_ids.is_some()
            && tuning::qwen35_q6_down_run_batched8_enabled();
        let q6_down_run_batched_ref = down_quant == 14
            && group4_down
            && !down_token_major
            && !group2_down
            && !group8_down
            && !q6_down_q8dot
            && !q6_down_pack4_f32
            && !q6_down_run_tiled4
            && !q6_down_run_batched8
            && !direct_selected_base_sparse
            && expert_ids.is_some()
            && tuning::qwen35_q6_down_run_batched_ref_enabled();
        let down_run_tile_meta = match (
            q6_down_run_tiled4,
            q6_down_run_batched8 || q6_down_run_batched_ref,
            expert_ids,
        ) {
            (true, _, Some(expert_ids)) => {
                qwen35_expert_run_down_weight_identity(expert_ids, down_weights)?;
                Some(qwen35_expert_run_specialized_tile_words(expert_ids, 4)?)
            }
            (_, true, Some(expert_ids)) => {
                qwen35_expert_run_down_weight_identity(expert_ids, down_weights)?;
                let max_tile_slots = if q6_down_run_batched8 { 8 } else { 4 };
                Some(qwen35_expert_run_tile_meta(expert_ids, max_tile_slots)?)
            }
            _ => None,
        };
        let down_run_tile_meta_bytes = down_run_tile_meta
            .as_ref()
            .map(|meta| std::mem::size_of_val(meta.as_slice()))
            .unwrap_or(0);
        let device_slot_scratch_bytes = device_slot_ptrs
            .as_ref()
            .map(|device| {
                let base_bytes = std::mem::size_of_val(device.expert_ids.as_slice());
                let map_bytes = match device.mixed_expert_ptrs.as_ref() {
                    Some(mixed) => {
                        std::mem::size_of_val(mixed.gate_ptrs.as_slice())
                            + std::mem::size_of_val(mixed.up_ptrs.as_slice())
                            + std::mem::size_of_val(mixed.down_ptrs.as_slice())
                    }
                    None => std::mem::size_of_val(device.expert_slab_indices.as_slice()),
                };
                base_bytes + map_bytes
            })
            .unwrap_or(0);
        let compound_runner_enabled = qwen35_selected_sparse_compound_runner_enabled();
        let compound_graph_enabled = tuning::qwen35_selected_sparse_compound_graph_enabled()
            && compound_runner_enabled
            && q4_gate_up_silu_pack4_group_offsets.is_some()
            && q6_down_pack4_f32;
        let compound_graph_down_meta_bytes = if compound_graph_enabled {
            std::mem::size_of_val(down_group_meta.as_slice())
        } else {
            0
        };
        let compound_graph_meta_workspace_bytes =
            qwen35_selected_sparse_compound_graph_meta_workspace_bytes(
                gate_up_group_meta_bytes,
                q4_gate_up_silu_pack4_offsets_bytes,
                compound_graph_down_meta_bytes,
                compound_graph_enabled,
            );
        let max_group_meta_bytes = std::mem::size_of_val(gate_up_group_meta.as_slice())
            .max(std::mem::size_of_val(down_group_meta.as_slice()))
            .max(down_token_major_bytes)
            .max(down_run_tile_meta_bytes)
            .max(device_slot_scratch_bytes)
            .max(q4_gate_up_silu_pack4_workspace_bytes)
            .max(compound_graph_meta_workspace_bytes);
        let group_meta_dev = if max_group_meta_bytes == 0 {
            0
        } else {
            self.compute_group_meta_ptr(max_group_meta_bytes)?
        };
        let compound_graph_down_meta_dev = qwen35_selected_sparse_compound_graph_down_meta_dev(
            group_meta_dev,
            gate_up_group_meta_bytes,
            q4_gate_up_silu_pack4_offsets_bytes,
            compound_graph_down_meta_bytes,
            compound_graph_enabled,
        )?;
        let q6_down_full4_split = down_quant == 14
            && group4_down
            && !down_token_major
            && !group2_down
            && !group8_down
            && !q6_down_q8dot
            && !q6_down_pack4_f32
            && !q6_down_run_tiled4
            && !q6_down_run_batched8
            && !q6_down_run_batched_ref
            && tuning::qwen35_q6_down_full4_split_enabled();
        let down_group_len_split = if q6_down_full4_split {
            Some(qwen35_group_meta_split_by_len(&down_group_meta, 4)?)
        } else {
            None
        };
        let down_group_meta_h2d_calls = if down_token_major_plan.is_some() {
            2
        } else if down_run_tile_meta.is_some() {
            1
        } else if let Some(split) = down_group_len_split.as_ref() {
            (if split.matching.is_empty() { 0 } else { 1 })
                + (if split.other.is_empty() { 0 } else { 1 })
        } else if (q6_down_pack4_f32 || group4_down) && gate_up_group_meta != down_group_meta {
            1
        } else {
            0
        };
        let down_launches = if let Some(split) = down_group_len_split.as_ref() {
            (if split.matching.is_empty() { 0 } else { 1 })
                + (if split.other.is_empty() { 0 } else { 1 })
        } else {
            1
        };
        let q4_gate_up_silu_fused = tuning::qwen35_q4_gate_up_silu_fused_enabled()
            && !gate_up_group_meta.is_empty()
            && group8_gate_up
            && !group16_gate_up
            && !q6_down_q8dot
            && !q6_down_pack4_f32;
        let zero_launches = if zero_output { 1 } else { 0 };
        let gate_up_launches = 1usize;
        let silu_launches = if q4_gate_up_silu_fused || q4_gate_up_silu_pack4_f32 {
            0
        } else {
            1usize
        };
        let execution_abi_enabled = qwen35_selected_sparse_execution_abi_enabled();
        let execution_descriptor = if execution_abi_enabled
            || compound_runner_enabled
            || boundary_stats.is_some()
        {
            let slot_pointer_source = match device_slot_ptrs.as_ref() {
                Some(device) => {
                    let selected_experts = device
                        .expert_slab_indices
                        .iter()
                        .filter(|&&index| index != u32::MAX)
                        .count();
                    if device.mixed_expert_ptrs.is_some() {
                        Qwen35SelectedSparseSlotPointerSource::DeviceMixed { selected_experts }
                    } else {
                        Qwen35SelectedSparseSlotPointerSource::DeviceCompact { selected_experts }
                    }
                }
                None => Qwen35SelectedSparseSlotPointerSource::Host,
            };
            Some(qwen35_selected_sparse_runtime_descriptor(
                Qwen35SelectedSparseRuntimeDescriptorInput {
                    slots,
                    token_count,
                    route_from_device,
                    slot_pointer_source,
                    gate_up_group: gate_up_max_group,
                    down_group: down_max_group,
                    down_quant,
                    zero_output,
                    q4_gate_up_silu_fused,
                    q4_gate_up_silu_pack4_f32,
                    q4_gate_up_silu_pack4_group8,
                    q6_down_q8dot,
                    q6_down_pack4_f32,
                    q6_down_pack4_f32_vec4_enabled: tuning::qwen35_q6_down_pack4_f32_vec4_enabled(),
                },
            )?)
        } else {
            None
        };
        let execution_runner_mode = qwen35_selected_sparse_runner_mode(
            execution_abi_enabled,
            compound_runner_enabled,
            execution_descriptor.as_ref(),
        )?;
        if let Some(stats) = boundary_stats.as_mut() {
            let gate_up_meta_bytes = gate_up_group_meta_bytes + q4_gate_up_silu_pack4_offsets_bytes;
            let down_meta_bytes = if let Some(plan) = down_token_major_plan.as_ref() {
                std::mem::size_of_val(plan.token_offsets.as_slice())
                    + std::mem::size_of_val(plan.slot_indices.as_slice())
            } else if let Some(meta) = down_run_tile_meta.as_ref() {
                std::mem::size_of_val(meta.as_slice())
            } else if let Some(split) = down_group_len_split.as_ref() {
                std::mem::size_of_val(split.matching.as_slice())
                    + std::mem::size_of_val(split.other.as_slice())
            } else if q6_down_pack4_f32 || (group4_down && gate_up_group_meta != down_group_meta) {
                std::mem::size_of_val(down_group_meta.as_slice())
            } else {
                0
            };
            let descriptor = execution_descriptor
                .as_ref()
                .expect("boundary stats request built selected sparse descriptor");
            stats.apply_execution_descriptor(descriptor);
            stats.group_meta_h2d_bytes = gate_up_meta_bytes + down_meta_bytes;
            stats.group_meta_h2d_calls = (if gate_up_group_meta.is_empty() { 0 } else { 1 })
                + (if q4_gate_up_silu_pack4_group_offsets.is_some() {
                    1
                } else {
                    0
                })
                + down_group_meta_h2d_calls;
        }
        #[cfg(test)]
        {
            self.last_qwen35_selected_sparse_boundary_stats = boundary_stats;
        }
        if qwen35_selected_sparse_boundary_trace_enabled() {
            if let Some(stats) = boundary_stats {
                eprintln!(
                    "[cuda-boundary] qwen35_selected_sparse slots={} unique_experts={} selected_upload_calls={} selected_upload_mb={:.2} route_h2d_bytes={} token_h2d_bytes={} device_slot_h2d_bytes={} group_meta_h2d_bytes={} descriptor_h2d_calls={} total_launches={} slot_ptr_launches={} zero_launches={} gate_up_launches={} silu_launches={} down_launches={}",
                    stats.slots,
                    stats.unique_experts,
                    stats.selected_upload_calls,
                    stats.selected_upload_bytes as f64 / (1024.0 * 1024.0),
                    stats.route_h2d_bytes,
                    stats.token_h2d_bytes,
                    stats.device_slot_h2d_bytes,
                    stats.group_meta_h2d_bytes,
                    stats.total_descriptor_h2d_calls(),
                    stats.total_kernel_launches(),
                    stats.slot_ptr_build_launches,
                    stats.zero_launches,
                    stats.gate_up_launches,
                    stats.silu_launches,
                    stats.down_launches
                );
            }
        }
        let down_q8 = if q6_down_q8dot {
            if n_ff % 32 != 0 {
                return Err(format!(
                    "Qwen35 q8dot down n_ff must be divisible by 32, got {n_ff}"
                ));
            }
            Some((
                self.compute_full_gate_ptr(slots * n_ff)?,
                self.compute_full_up_ptr(slots * (n_ff / 32) * std::mem::size_of::<f32>())?,
            ))
        } else {
            None
        };
        let down_pack4_f32 = if q6_down_pack4_f32 {
            let groups = down_group_meta.len() / 2;
            let bytes = groups
                .checked_mul(4)
                .and_then(|value| value.checked_mul(n_ff))
                .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                .ok_or_else(|| {
                    format!("Qwen35 pack4 F32 activation byte size overflow: groups={groups} n_ff={n_ff}")
                })?;
            Some(self.qwen35_packed_act_ptr(bytes)?)
        } else {
            None
        };
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            weight_ptr_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let phase_t0 = trace_phase.then(std::time::Instant::now);
        unsafe {
            if let Some(device) = device_slot_ptrs.as_ref() {
                let ids_bytes = std::mem::size_of_val(device.expert_ids.as_slice());
                if group_meta_dev == 0 {
                    return Err(
                        "Qwen35 compact slot pointer build missing scratch workspace".to_string(),
                    );
                }
                let expert_ids_dev = group_meta_dev;
                self.api.memcpy_htod_async(
                    expert_ids_dev,
                    device.expert_ids.as_ptr().cast::<libc::c_void>(),
                    ids_bytes,
                    self.stream,
                )?;
                let scratch_tail_dev = group_meta_dev
                    .checked_add(u64::try_from(ids_bytes).map_err(|_| {
                        format!(
                            "Qwen35 compact slot pointer expert id bytes exceed u64: {ids_bytes}"
                        )
                    })?)
                    .ok_or_else(|| {
                        format!(
                            "Qwen35 compact slot pointer workspace overflows: base={group_meta_dev} ids_bytes={ids_bytes}"
                        )
                    })?;
                if let Some(mixed) = device.mixed_expert_ptrs.as_ref() {
                    let gate_bytes = std::mem::size_of_val(mixed.gate_ptrs.as_slice());
                    let up_bytes = std::mem::size_of_val(mixed.up_ptrs.as_slice());
                    let down_bytes = std::mem::size_of_val(mixed.down_ptrs.as_slice());
                    let up_ptrs_table_dev = scratch_tail_dev
                        .checked_add(u64::try_from(gate_bytes).map_err(|_| {
                            format!(
                                "Qwen35 mixed slot pointer gate table bytes exceed u64: {gate_bytes}"
                            )
                        })?)
                        .ok_or_else(|| {
                            format!(
                                "Qwen35 mixed slot pointer up table overflows: base={scratch_tail_dev} gate_bytes={gate_bytes}"
                            )
                        })?;
                    let down_ptrs_table_dev = up_ptrs_table_dev
                        .checked_add(u64::try_from(up_bytes).map_err(|_| {
                            format!(
                                "Qwen35 mixed slot pointer up table bytes exceed u64: {up_bytes}"
                            )
                        })?)
                        .ok_or_else(|| {
                            format!(
                                "Qwen35 mixed slot pointer down table overflows: up_base={up_ptrs_table_dev} up_bytes={up_bytes}"
                            )
                        })?;
                    self.api.memcpy_htod_async(
                        scratch_tail_dev,
                        mixed.gate_ptrs.as_ptr().cast::<libc::c_void>(),
                        gate_bytes,
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        up_ptrs_table_dev,
                        mixed.up_ptrs.as_ptr().cast::<libc::c_void>(),
                        up_bytes,
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        down_ptrs_table_dev,
                        mixed.down_ptrs.as_ptr().cast::<libc::c_void>(),
                        down_bytes,
                        self.stream,
                    )?;
                    self.launch_qwen35_build_q4k_mixed_slot_ptrs(
                        gate_ptrs_dev,
                        up_ptrs_dev,
                        down_ptrs_dev,
                        expert_ids_dev,
                        scratch_tail_dev,
                        up_ptrs_table_dev,
                        down_ptrs_table_dev,
                        slots,
                    )?;
                } else {
                    let index_bytes = std::mem::size_of_val(device.expert_slab_indices.as_slice());
                    self.api.memcpy_htod_async(
                        scratch_tail_dev,
                        device.expert_slab_indices.as_ptr().cast::<libc::c_void>(),
                        index_bytes,
                        self.stream,
                    )?;
                    self.launch_qwen35_build_q4k_compact_slot_ptrs(
                        gate_ptrs_dev,
                        up_ptrs_dev,
                        down_ptrs_dev,
                        expert_ids_dev,
                        scratch_tail_dev,
                        device.gate_base,
                        device.up_base,
                        device.down_base,
                        device.gate_expert_bytes,
                        device.up_expert_bytes,
                        device.down_expert_bytes,
                        slots,
                    )?;
                }
            } else {
                self.api.memcpy_htod_async(
                    gate_ptrs_dev,
                    gate_ptrs.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(gate_ptrs.as_slice()),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    up_ptrs_dev,
                    up_ptrs.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(up_ptrs.as_slice()),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    down_ptrs_dev,
                    down_ptrs.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(down_ptrs.as_slice()),
                    self.stream,
                )?;
            }
            if !route_from_device {
                self.api.memcpy_htod_async(
                    route_dev,
                    route_weights.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(route_weights),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    token_ids_dev,
                    token_ids.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(token_ids),
                    self.stream,
                )?;
            }
            if !gate_up_group_meta.is_empty() {
                self.api.memcpy_htod_async(
                    group_meta_dev,
                    gate_up_group_meta.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(gate_up_group_meta.as_slice()),
                    self.stream,
                )?;
            }
            if let Some(offsets) = q4_gate_up_silu_pack4_group_offsets.as_ref() {
                self.api.memcpy_htod_async(
                    group_meta_dev + gate_up_group_meta_bytes as u64,
                    offsets.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(offsets.as_slice()),
                    self.stream,
                )?;
            }
        }
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            setup_h2d_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let warp_down = std::env::var("RNB_CUDA_WARP_DOWN").ok().as_deref() != Some("0");
        let exact_runner_plan = match execution_runner_mode {
            Qwen35SelectedSparseRunnerMode::LegacyInline
            | Qwen35SelectedSparseRunnerMode::ExactReference
            | Qwen35SelectedSparseRunnerMode::CompoundExactReference => {
                qwen35_selected_sparse_exact_runner_plan(
                    Qwen35SelectedSparseExactRunnerPlanInput {
                        descriptor: execution_descriptor.as_ref(),
                        has_pack4_group_offsets: q4_gate_up_silu_pack4_group_offsets.is_some(),
                        q4_gate_up_silu_pack4_f32,
                        q4_gate_up_silu_fused,
                        has_gate_up_group_meta: !gate_up_group_meta.is_empty(),
                        has_down_token_major_plan: down_token_major_plan.is_some(),
                        has_down_run_tile_meta: down_run_tile_meta.is_some(),
                        q6_down_pack4_f32,
                        group4_down,
                        gate_up_meta_matches_down_meta: gate_up_group_meta == down_group_meta,
                        down_quant,
                        warp_down,
                        q4_down_group4,
                        down_token_major,
                        q6_down_q8dot,
                        q6_down_run_tiled4,
                        q6_down_run_batched_ref,
                        q6_down_run_batched8,
                        q6_down_full4_split,
                    },
                )?
            }
        };
        let exact_runner_payload = qwen35_selected_sparse_exact_runner_payload(
            exact_runner_plan,
            down_pack4_f32,
            q4_gate_up_silu_pack4_group_offsets.as_deref(),
            &gate_up_group_meta,
            &down_group_meta,
            group_meta_dev,
            gate_up_group_meta_bytes,
            down_token_major_plan.as_ref(),
            down_q8,
            down_run_tile_meta.as_deref(),
            down_group_len_split.as_ref(),
        )?;
        let compound_runner_payload = qwen35_selected_sparse_compound_reference_payload(
            execution_runner_mode,
            exact_runner_payload,
        );
        let compound_graph_captures_zero_output =
            qwen35_selected_sparse_compound_graph_captures_zero_output(
                zero_output,
                compound_graph_enabled,
                tuning::qwen35_selected_sparse_compound_graph_zero_enabled(),
                trace_kernel,
                compound_runner_payload.is_some(),
            );

        let phase_t0 = trace_phase.then(std::time::Instant::now);
        let kernel_t0 = trace_kernel.then(std::time::Instant::now);
        if zero_output && !compound_graph_captures_zero_output {
            self.launch_zero_f32(output_dev, output_len)?;
        }
        if let Some(t0) = kernel_t0 {
            self.stream_synchronize()?;
            zero_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        if copy_stream_upload {
            unsafe { self.api.stream_synchronize(self.copy_stream)? };
        }
        if let Some(compound_runner_payload) = compound_runner_payload {
            let timings = self.qwen35_launch_selected_sparse_compound_reference(
                compound_runner_payload,
                gate_ptrs_dev,
                up_ptrs_dev,
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                compound_graph_down_meta_dev,
                n_ff,
                n_embd,
                compound_graph_captures_zero_output,
                output_len,
                input_dev,
                route_dev,
                output_dev,
                !temp_slab_ptrs.is_empty() && tuning::prefill_down_copy_overlap_enabled(),
                trace_kernel,
            )?;
            gate_up_ms = timings.gate_up_ms;
            silu_ms = timings.silu_ms;
            down_ms = timings.down_ms;
        } else {
            let kernel_t0 = trace_kernel.then(std::time::Instant::now);
            self.qwen35_launch_selected_sparse_gate_up(
                exact_runner_payload.gate_up,
                gate_ptrs_dev,
                up_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_ff,
                n_embd,
                slots,
                input_dev,
                gate_dev,
                up_dev,
                &down_group_meta,
            )?;
            if let Some(t0) = kernel_t0 {
                self.stream_synchronize()?;
                gate_up_ms = t0.elapsed().as_micros() as f64 / 1000.0;
            }
            let kernel_t0 = trace_kernel.then(std::time::Instant::now);
            self.qwen35_launch_selected_sparse_silu(
                exact_runner_payload.silu,
                gate_dev,
                up_dev,
                group_meta_dev,
                n_ff,
                slots,
                &down_group_meta,
            )?;
            if let Some(t0) = kernel_t0 {
                self.stream_synchronize()?;
                silu_ms = t0.elapsed().as_micros() as f64 / 1000.0;
            }
            if !temp_slab_ptrs.is_empty() && tuning::prefill_down_copy_overlap_enabled() {
                unsafe { self.api.stream_synchronize(self.copy_stream)? };
            }
            self.qwen35_stage_selected_sparse_down_meta(
                exact_runner_payload.down_meta,
                group_meta_dev,
            )?;
            let kernel_t0 = trace_kernel.then(std::time::Instant::now);
            self.qwen35_launch_selected_sparse_down(
                exact_runner_payload.down_launch,
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                token_count,
                slots,
                n_ff,
                gate_dev,
                route_dev,
                output_dev,
            )?;
            if let Some(t0) = kernel_t0 {
                self.stream_synchronize()?;
                down_ms = t0.elapsed().as_micros() as f64 / 1000.0;
            }
        }
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            kernels_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        if let (Some(t0), Some(before)) = (trace_t0, trace_before) {
            let delta = cache_snapshot().delta(before);
            let hit_rate = if delta.lookups == 0 {
                0.0
            } else {
                (delta.hits as f64 * 100.0) / delta.lookups as f64
            };
            eprintln!(
                "[cuda-cache] qwen35_sparse slots={} tokens={} lookups={} hits={} misses={} hit_rate={:.1}% evictions={} resident_upload_mb={:.2} temp_upload_mb={:.2} resident_mb={:.2} elapsed_ms={:.1}",
                slots,
                token_count,
                delta.lookups,
                delta.hits,
                delta.misses,
                hit_rate,
                delta.evictions,
                delta.resident_upload_bytes as f64 / (1024.0 * 1024.0),
                delta.temp_upload_bytes as f64 / (1024.0 * 1024.0),
                self.resident_q4k_bytes as f64 / (1024.0 * 1024.0),
                t0.elapsed().as_micros() as f64 / 1000.0
            );
        }
        if let Some(t0) = trace_t0.filter(|_| trace_phase) {
            let down_shape = qwen35_group_shape_summary(&down_group_meta)?;
            eprintln!(
                "[cuda-phase] qwen35_sparse slots={} tokens={} down_groups={} down_slots={} down_max_len={} down_g1={} down_g2={} down_g3={} down_g4={} down_g8={} down_g_over8={} weight_ptr_ms={:.1} setup_h2d_ms={:.1} kernels_ms={:.1} dtoh_ms={:.1} total_ms={:.1}",
                slots,
                token_count,
                down_shape.groups,
                down_shape.slots,
                down_shape.max_len,
                down_shape.len_hist[1],
                down_shape.len_hist[2],
                down_shape.len_hist[3],
                down_shape.len_hist[4],
                down_shape.len_hist[8],
                down_shape.overflow_groups,
                weight_ptr_ms,
                setup_h2d_ms,
                kernels_ms,
                dtoh_ms,
                t0.elapsed().as_micros() as f64 / 1000.0
            );
        }
        if trace_kernel {
            let down_shape = qwen35_group_shape_summary(&down_group_meta)?;
            eprintln!(
                "[cuda-kernel] qwen35_sparse slots={} tokens={} down_groups={} down_max_len={} down_g1={} down_g2={} down_g3={} down_g4={} down_g8={} down_g_over8={} zero_launches={} gate_up_launches={} silu_launches={} down_launches={} zero_ms={:.1} gate_up_ms={:.1} silu_ms={:.1} down_ms={:.1}",
                slots,
                token_count,
                down_shape.groups,
                down_shape.max_len,
                down_shape.len_hist[1],
                down_shape.len_hist[2],
                down_shape.len_hist[3],
                down_shape.len_hist[4],
                down_shape.len_hist[8],
                down_shape.overflow_groups,
                zero_launches,
                gate_up_launches,
                silu_launches,
                down_launches,
                zero_ms,
                gate_up_ms,
                silu_ms,
                down_ms
            );
        }
        drop(temp_slab_ptrs);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_full_layer_by_token_to_dev(
        &mut self,
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
        output_dev: u64,
    ) -> Result<(), String> {
        let trace_phase = std::env::var("RNB_CUDA_PHASE_TRACE").ok().as_deref() == Some("1");
        let trace_kernel = std::env::var("RNB_CUDA_KERNEL_TRACE").ok().as_deref() == Some("1");
        let trace_t0 = trace_phase.then(std::time::Instant::now);
        let mut full_h2d_ms = 0.0f64;
        let mut setup_h2d_ms = 0.0f64;
        let mut kernels_ms = 0.0f64;
        let mut zero_ms = 0.0f64;
        let mut gate_up_ms = 0.0f64;
        let mut silu_ms = 0.0f64;
        let mut down_ms = 0.0f64;
        let slots = expert_ids.len();
        let gate_row_bytes = (n_embd / 256) * 144;
        let gate_expert_bytes = n_ff * gate_row_bytes;
        let down_row_bytes = match down_quant {
            12 => (n_ff / 256) * 144,
            13 => (n_ff / 256) * 176,
            14 => (n_ff / 256) * 210,
            other => return Err(format!("unsupported Qwen35 token-batch down quant {other}")),
        };
        let down_expert_bytes = n_embd * down_row_bytes;
        let n_expert = gate_all.len() / gate_expert_bytes;
        let gate_bytes = gate_all.len();
        let up_bytes = up_all.len();
        let down_bytes = down_all.len();
        let key = qwen35_moe_layer_key(gate_all, up_all, down_all, down_quant, n_ff, n_embd);
        let resident = self
            .resident_moe_layers
            .get(&key)
            .map(|entry| (entry.gate_base, entry.up_base, entry.down_base));
        let range_plan = if tuning::prefill_moe_range_slab_enabled() {
            build_expert_range_upload_plan(
                expert_ids,
                n_expert,
                tuning::prefill_moe_range_slab_max_gap_experts(),
                tuning::prefill_moe_range_slab_max_overhead_permille(),
            )
        } else {
            None
        };
        let mut range_expert_offsets = None;
        let (gate_base, up_base, down_base, resident_hit) = if let Some((
            gate_base,
            up_base,
            down_base,
        )) = resident
        {
            self.touch_resident_moe_layer(key);
            (gate_base, up_base, down_base, true)
        } else if let Some(plan) = range_plan {
            let gate_range_bytes = plan
                .slab_experts
                .checked_mul(gate_expert_bytes)
                .ok_or_else(|| "Qwen35 range slab gate byte overflow".to_string())?;
            let up_range_bytes = plan
                .slab_experts
                .checked_mul(gate_expert_bytes)
                .ok_or_else(|| "Qwen35 range slab up byte overflow".to_string())?;
            let down_range_bytes = plan
                .slab_experts
                .checked_mul(down_expert_bytes)
                .ok_or_else(|| "Qwen35 range slab down byte overflow".to_string())?;
            let slab_bytes = gate_range_bytes
                .checked_add(up_range_bytes)
                .and_then(|bytes| bytes.checked_add(down_range_bytes))
                .ok_or_else(|| "Qwen35 range slab byte overflow".to_string())?;
            let slab_dev = self.compute_temp_slab_ptr(slab_bytes)?;
            let gate_base = slab_dev;
            let up_base = gate_base + gate_range_bytes as u64;
            let down_base = up_base + up_range_bytes as u64;
            let phase_t0 = trace_phase.then(std::time::Instant::now);
            for range in &plan.ranges {
                let expert_count = range.expert_end - range.expert_start;
                let src_offset = range.expert_start * gate_expert_bytes;
                let dst_offset = range.slab_expert_offset * gate_expert_bytes;
                let bytes = expert_count * gate_expert_bytes;
                unsafe {
                    self.api.memcpy_htod_async(
                        gate_base + dst_offset as u64,
                        gate_all.as_ptr().add(src_offset).cast::<libc::c_void>(),
                        bytes,
                        self.stream,
                    )?;
                }
            }
            for range in &plan.ranges {
                let expert_count = range.expert_end - range.expert_start;
                let src_offset = range.expert_start * gate_expert_bytes;
                let dst_offset = range.slab_expert_offset * gate_expert_bytes;
                let bytes = expert_count * gate_expert_bytes;
                unsafe {
                    self.api.memcpy_htod_async(
                        up_base + dst_offset as u64,
                        up_all.as_ptr().add(src_offset).cast::<libc::c_void>(),
                        bytes,
                        self.stream,
                    )?;
                }
            }
            for range in &plan.ranges {
                let expert_count = range.expert_end - range.expert_start;
                let src_offset = range.expert_start * down_expert_bytes;
                let dst_offset = range.slab_expert_offset * down_expert_bytes;
                let bytes = expert_count * down_expert_bytes;
                unsafe {
                    self.api.memcpy_htod_async(
                        down_base + dst_offset as u64,
                        down_all.as_ptr().add(src_offset).cast::<libc::c_void>(),
                        bytes,
                        self.stream,
                    )?;
                }
            }
            if let Some(t0) = phase_t0 {
                self.stream_synchronize()?;
                full_h2d_ms = t0.elapsed().as_micros() as f64 / 1000.0;
            }
            if trace_phase {
                eprintln!(
                    "[cuda-phase] qwen35_sparse_range_slab selected_experts={} slab_experts={} ranges={} slab_mb={:.2}",
                    plan.selected_experts,
                    plan.slab_experts,
                    plan.ranges.len(),
                    slab_bytes as f64 / (1024.0 * 1024.0)
                );
            }
            range_expert_offsets = Some(plan.expert_offsets);
            (gate_base, up_base, down_base, false)
        } else if tuning::prefill_moe_full_layer_enabled() {
            let slab_bytes = gate_bytes + up_bytes + down_bytes;
            let slab_dev = self.compute_temp_slab_ptr(slab_bytes)?;
            let gate_base = slab_dev;
            let up_base = gate_base + gate_bytes as u64;
            let down_base = up_base + up_bytes as u64;
            let phase_t0 = trace_phase.then(std::time::Instant::now);
            unsafe {
                self.api.memcpy_htod_async(
                    gate_base,
                    gate_all.as_ptr().cast::<libc::c_void>(),
                    gate_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    up_base,
                    up_all.as_ptr().cast::<libc::c_void>(),
                    up_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    down_base,
                    down_all.as_ptr().cast::<libc::c_void>(),
                    down_bytes,
                    self.stream,
                )?;
            }
            if let Some(t0) = phase_t0 {
                self.stream_synchronize()?;
                full_h2d_ms = t0.elapsed().as_micros() as f64 / 1000.0;
            }
            (gate_base, up_base, down_base, false)
        } else {
            let mut gate_weights = Vec::with_capacity(slots);
            let mut up_weights = Vec::with_capacity(slots);
            let mut down_weights = Vec::with_capacity(slots);
            for &expert in expert_ids {
                let expert = expert as usize;
                if expert >= n_expert {
                    return Err(format!(
                            "Qwen35 full-layer expert id out of range: got {expert}, n_expert={n_expert}"
                        ));
                }
                gate_weights
                    .push(&gate_all[expert * gate_expert_bytes..(expert + 1) * gate_expert_bytes]);
                up_weights
                    .push(&up_all[expert * gate_expert_bytes..(expert + 1) * gate_expert_bytes]);
                down_weights
                    .push(&down_all[expert * down_expert_bytes..(expert + 1) * down_expert_bytes]);
            }
            return self.qwen35_sparse_experts_by_token_to_dev(
                &gate_weights,
                &up_weights,
                &down_weights,
                route_weights,
                token_ids,
                token_count,
                down_quant,
                n_ff,
                n_embd,
                self.compute_input
                    .ok_or_else(|| "missing CUDA compute input".to_string())?,
                output_dev,
                true,
                false,
            );
        };

        let gate_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let up_ptrs_dev = self.compute_up_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let down_ptrs_dev = self.compute_down_ptrs_ptr(slots * std::mem::size_of::<u64>())?;
        let route_dev = self.compute_route_ptr(std::mem::size_of_val(route_weights))?;
        let token_ids_dev = self.compute_token_ids_ptr(std::mem::size_of_val(token_ids))?;
        let slot_ptr_plan = qwen35_full_layer_slot_ptr_plan(
            tuning::qwen35_full_layer_device_slot_ptrs_enabled() && slots > 0,
            range_expert_offsets.is_some(),
        );
        let expert_ids_dev = if slot_ptr_plan == Qwen35FullLayerSlotPtrPlan::DevicePointerBuild {
            Some(self.compute_full_gate_ptr(std::mem::size_of_val(expert_ids))?)
        } else {
            None
        };
        let mut gate_ptrs = Vec::new();
        let mut up_ptrs = Vec::new();
        let mut down_ptrs = Vec::new();
        if slot_ptr_plan == Qwen35FullLayerSlotPtrPlan::HostPointerUpload {
            gate_ptrs.reserve(slots);
            up_ptrs.reserve(slots);
            down_ptrs.reserve(slots);
        }
        for &expert in expert_ids {
            let expert = expert as usize;
            if expert >= n_expert {
                return Err(format!(
                    "Qwen35 full-layer expert id out of range: got {expert}, n_expert={n_expert}"
                ));
            }
            if slot_ptr_plan == Qwen35FullLayerSlotPtrPlan::HostPointerUpload {
                if let Some(offsets) = &range_expert_offsets {
                    let expert_offset =
                        offsets
                            .get(expert)
                            .and_then(|offset| *offset)
                            .ok_or_else(|| {
                                format!("Qwen35 range slab missing expert offset: {expert}")
                            })?;
                    gate_ptrs.push(gate_base + (expert_offset * gate_expert_bytes) as u64);
                    up_ptrs.push(up_base + (expert_offset * gate_expert_bytes) as u64);
                    down_ptrs.push(down_base + (expert_offset * down_expert_bytes) as u64);
                } else {
                    gate_ptrs.push(gate_base + (expert * gate_expert_bytes) as u64);
                    up_ptrs.push(up_base + (expert * gate_expert_bytes) as u64);
                    down_ptrs.push(down_base + (expert * down_expert_bytes) as u64);
                }
            }
        }
        let gate_up_group_meta = {
            let group8_gate_up = std::env::var("RNB_CUDA_GROUP8_GATE_UP_WARP4")
                .ok()
                .as_deref()
                != Some("0");
            let max_group = if group8_gate_up { 8 } else { 4 };
            build_group_meta_from_ids(expert_ids, max_group)
        };
        let group2_down = tuning::group2_down_warp4_enabled();
        let group8_down = !group2_down
            && std::env::var("RNB_CUDA_GROUP8_DOWN_WARP4").ok().as_deref() == Some("1");
        let down_group_meta = build_group_meta_from_ids(
            expert_ids,
            if group2_down {
                2
            } else if group8_down {
                8
            } else {
                4
            },
        );
        let q6_down_q8dot = down_quant == 14
            && !group2_down
            && !group8_down
            && tuning::qwen35_q6_down_q8dot_enabled();
        let q6_down_run_tiled4 = down_quant == 14
            && !group2_down
            && !group8_down
            && !q6_down_q8dot
            && tuning::qwen35_q6_down_run_tiled4_enabled();
        let down_run_tile_meta = if q6_down_run_tiled4 {
            Some(qwen35_expert_run_specialized_tile_words(expert_ids, 4)?)
        } else {
            None
        };
        let down_q8 = if q6_down_q8dot {
            if n_ff % 32 != 0 {
                return Err(format!(
                    "Qwen35 q8dot down n_ff must be divisible by 32, got {n_ff}"
                ));
            }
            Some((
                self.compute_full_up_ptr(slots * n_ff)?,
                self.compute_full_down_ptr(slots * (n_ff / 32) * std::mem::size_of::<f32>())?,
            ))
        } else {
            None
        };
        let max_group_meta_bytes = std::mem::size_of_val(gate_up_group_meta.as_slice())
            .max(std::mem::size_of_val(down_group_meta.as_slice()))
            .max(
                down_run_tile_meta
                    .as_ref()
                    .map(|meta| std::mem::size_of_val(meta.as_slice()))
                    .unwrap_or(0),
            );
        let group_meta_dev = if max_group_meta_bytes == 0 {
            0
        } else {
            self.compute_group_meta_ptr(max_group_meta_bytes)?
        };

        let phase_t0 = trace_phase.then(std::time::Instant::now);
        unsafe {
            match (slot_ptr_plan, expert_ids_dev) {
                (Qwen35FullLayerSlotPtrPlan::HostPointerUpload, _) => {
                    self.api.memcpy_htod_async(
                        gate_ptrs_dev,
                        gate_ptrs.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(gate_ptrs.as_slice()),
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        up_ptrs_dev,
                        up_ptrs.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(up_ptrs.as_slice()),
                        self.stream,
                    )?;
                    self.api.memcpy_htod_async(
                        down_ptrs_dev,
                        down_ptrs.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(down_ptrs.as_slice()),
                        self.stream,
                    )?;
                }
                (Qwen35FullLayerSlotPtrPlan::DevicePointerBuild, Some(expert_ids_dev)) => {
                    self.api.memcpy_htod_async(
                        expert_ids_dev,
                        expert_ids.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(expert_ids),
                        self.stream,
                    )?;
                }
                (Qwen35FullLayerSlotPtrPlan::DevicePointerBuild, None) => {
                    return Err(
                        "Qwen35 full-layer device slot pointer build missing expert id workspace"
                            .to_string(),
                    );
                }
            }
            self.api.memcpy_htod_async(
                route_dev,
                route_weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(route_weights),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                token_ids_dev,
                token_ids.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(token_ids),
                self.stream,
            )?;
            if !gate_up_group_meta.is_empty() {
                self.api.memcpy_htod_async(
                    group_meta_dev,
                    gate_up_group_meta.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(gate_up_group_meta.as_slice()),
                    self.stream,
                )?;
            }
        }
        if let Some(expert_ids_dev) = expert_ids_dev {
            self.launch_qwen35_build_q4k_full_layer_slot_ptrs(
                gate_ptrs_dev,
                up_ptrs_dev,
                down_ptrs_dev,
                expert_ids_dev,
                gate_base,
                up_base,
                down_base,
                gate_expert_bytes,
                down_expert_bytes,
                slots,
            )?;
        }
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            setup_h2d_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let phase_t0 = trace_phase.then(std::time::Instant::now);
        let kernel_t0 = trace_kernel.then(std::time::Instant::now);
        self.launch_zero_f32(output_dev, token_count * n_embd)?;
        if let Some(t0) = kernel_t0 {
            self.stream_synchronize()?;
            zero_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        let kernel_t0 = trace_kernel.then(std::time::Instant::now);
        self.launch_selected_q4k_gate_up_gemv_by_token_group4_to_dev(
            gate_ptrs_dev,
            up_ptrs_dev,
            token_ids_dev,
            group_meta_dev,
            n_ff,
            gate_up_group_meta.len() / 2,
            n_embd / 256,
            self.compute_input
                .ok_or_else(|| "missing CUDA compute input".to_string())?,
            gate_dev,
            up_dev,
        )?;
        if let Some(t0) = kernel_t0 {
            self.stream_synchronize()?;
            gate_up_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        let kernel_t0 = trace_kernel.then(std::time::Instant::now);
        if let Some((qs_dev, ds_dev)) = down_q8 {
            self.launch_silu_mul_q8_1(gate_dev, up_dev, qs_dev, ds_dev, slots * n_ff)?;
        } else {
            self.launch_silu_mul(gate_dev, up_dev, slots * n_ff)?;
        }
        if let Some(t0) = kernel_t0 {
            self.stream_synchronize()?;
            silu_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        if let Some(meta) = down_run_tile_meta.as_ref() {
            unsafe {
                self.api.memcpy_htod_async(
                    group_meta_dev,
                    meta.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(meta.as_slice()),
                    self.stream,
                )?;
            }
        } else if gate_up_group_meta != down_group_meta {
            unsafe {
                self.api.memcpy_htod_async(
                    group_meta_dev,
                    down_group_meta.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(down_group_meta.as_slice()),
                    self.stream,
                )?;
            }
        }
        let kernel_t0 = trace_kernel.then(std::time::Instant::now);
        match down_quant {
            13 => self.launch_selected_down_accum_by_token_group4(
                "rnb_q5k_selected_down_accum_by_token_group4",
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                down_group_meta.len() / 2,
                n_ff / 256,
                gate_dev,
                route_dev,
                output_dev,
            )?,
            14 if q6_down_q8dot => {
                let (qs_dev, ds_dev) = down_q8.expect("Q6 q8dot down buffers exist when enabled");
                self.launch_selected_q6k_down_accum_by_token_group4_q8dot(
                    down_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    down_group_meta.len() / 2,
                    n_ff / 256,
                    qs_dev,
                    ds_dev,
                    route_dev,
                    output_dev,
                )?
            }
            14 if q6_down_run_tiled4 => {
                let meta = down_run_tile_meta
                    .as_ref()
                    .expect("Q6 run-tiled4 meta exists when enabled");
                self.launch_selected_q6k_down_accum_run_tiled4_warp4(
                    down_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    n_embd,
                    meta.len() / 5,
                    n_ff / 256,
                    gate_dev,
                    route_dev,
                    output_dev,
                )?
            }
            14 => self.launch_selected_down_accum_by_token_group4(
                "rnb_q6k_selected_down_accum_by_token_group4",
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                down_group_meta.len() / 2,
                n_ff / 256,
                gate_dev,
                route_dev,
                output_dev,
            )?,
            12 => self.launch_selected_down_accum_by_token_group4(
                "rnb_q4k_selected_down_accum_by_token_group4",
                down_ptrs_dev,
                token_ids_dev,
                group_meta_dev,
                n_embd,
                down_group_meta.len() / 2,
                n_ff / 256,
                gate_dev,
                route_dev,
                output_dev,
            )?,
            other => return Err(format!("unsupported Qwen35 token-batch down quant {other}")),
        }
        if let Some(t0) = kernel_t0 {
            self.stream_synchronize()?;
            down_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        if let Some(t0) = phase_t0 {
            self.stream_synchronize()?;
            kernels_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }
        if let Some(t0) = trace_t0 {
            let slot_ptr_plan_name = match slot_ptr_plan {
                Qwen35FullLayerSlotPtrPlan::HostPointerUpload => "host",
                Qwen35FullLayerSlotPtrPlan::DevicePointerBuild => "device",
            };
            eprintln!(
                "[cuda-phase] qwen35_sparse_full_layer slots={} tokens={} ptr_plan={} full_h2d_ms={:.1} setup_h2d_ms={:.1} kernels_ms={:.1} total_ms={:.1}",
                slots,
                token_count,
                slot_ptr_plan_name,
                full_h2d_ms,
                setup_h2d_ms,
                kernels_ms,
                t0.elapsed().as_micros() as f64 / 1000.0
            );
        }
        if trace_kernel {
            eprintln!(
                "[cuda-kernel] qwen35_sparse_full_layer slots={} tokens={} resident={} zero_ms={:.1} gate_up_ms={:.1} silu_ms={:.1} down_ms={:.1}",
                slots, token_count, resident_hit, zero_ms, gate_up_ms, silu_ms, down_ms
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_expert_range_upload_plan, build_prefill_hot_resident_plan,
        qwen35_group_meta_from_ids_enabled, qwen35_pack4_group_offsets_for_down_group_meta,
        qwen35_selected_sparse_compound_graph_captures_zero_output,
        qwen35_selected_sparse_compound_graph_down_meta_dev,
        qwen35_selected_sparse_compound_graph_meta_workspace_bytes,
        qwen35_selected_sparse_compound_reference_payload,
        qwen35_selected_sparse_down_launch_payload, qwen35_selected_sparse_down_launch_runner,
        qwen35_selected_sparse_down_launch_runner_from_descriptor,
        qwen35_selected_sparse_down_meta_stage, qwen35_selected_sparse_down_meta_staging_payload,
        qwen35_selected_sparse_exact_runner_payload, qwen35_selected_sparse_exact_runner_plan,
        qwen35_selected_sparse_execution_descriptor,
        qwen35_selected_sparse_gate_up_activation_runner,
        qwen35_selected_sparse_gate_up_activation_runner_from_descriptor,
        qwen35_selected_sparse_gate_up_launch_payload, qwen35_selected_sparse_silu_launch_payload,
        qwen35_sparse_by_token_temp_slab_enabled, ExpertRangeUpload,
        Qwen35SelectedSparseCompoundRunnerPayload, Qwen35SelectedSparseDescriptorInput,
        Qwen35SelectedSparseDownLaunchPayload, Qwen35SelectedSparseDownLaunchRunner,
        Qwen35SelectedSparseDownMetaStage, Qwen35SelectedSparseDownMetaStagingPayload,
        Qwen35SelectedSparseExactRunnerPayload, Qwen35SelectedSparseExactRunnerPlan,
        Qwen35SelectedSparseExactRunnerPlanInput, Qwen35SelectedSparseExecutionDescriptor,
        Qwen35SelectedSparseGateUpActivationRunner, Qwen35SelectedSparseGateUpLaunchPayload,
        Qwen35SelectedSparseRunnerMode, Qwen35SelectedSparseSiluLaunchPayload,
        Qwen35SelectedSparseSlotPointerSource,
    };
    use crate::runtime::q4k_resident_key;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn restore_env(name: &str, previous: Option<String>) {
        if let Some(previous) = previous {
            std::env::set_var(name, previous);
        } else {
            std::env::remove_var(name);
        }
    }

    fn with_clean_env(test: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().expect("by-token env lock");
        let prev_temp = std::env::var("RNB_CUDA_PREFILL_TEMP_SLAB").ok();
        let prev_mtp = std::env::var("RNB_MTP_DEVICE_VERIFY").ok();
        std::env::remove_var("RNB_CUDA_PREFILL_TEMP_SLAB");
        std::env::remove_var("RNB_MTP_DEVICE_VERIFY");
        test();
        restore_env("RNB_CUDA_PREFILL_TEMP_SLAB", prev_temp);
        restore_env("RNB_MTP_DEVICE_VERIFY", prev_mtp);
    }

    #[test]
    fn temp_slab_stays_default_for_non_mtp_by_token_prefill() {
        with_clean_env(|| {
            assert!(qwen35_sparse_by_token_temp_slab_enabled(7, 56));
        });
    }

    #[test]
    fn mtp_device_verify_prefers_resident_slots_for_short_windows() {
        with_clean_env(|| {
            std::env::set_var("RNB_MTP_DEVICE_VERIFY", "1");
            assert!(!qwen35_sparse_by_token_temp_slab_enabled(2, 16));
            assert!(!qwen35_sparse_by_token_temp_slab_enabled(7, 56));
            assert!(qwen35_sparse_by_token_temp_slab_enabled(9, 72));
        });
    }

    #[test]
    fn temp_slab_env_override_wins() {
        with_clean_env(|| {
            std::env::set_var("RNB_MTP_DEVICE_VERIFY", "1");
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_SLAB", "1");
            assert!(qwen35_sparse_by_token_temp_slab_enabled(2, 16));
            std::env::set_var("RNB_CUDA_PREFILL_TEMP_SLAB", "0");
            assert!(!qwen35_sparse_by_token_temp_slab_enabled(9, 72));
        });
    }

    #[test]
    fn group_meta_from_ids_stays_quarantined_after_xid79() {
        let _guard = ENV_LOCK.lock().expect("by-token env lock");
        let prev = std::env::var("RNB_CUDA_QWEN35_GROUP_META_FROM_IDS").ok();
        std::env::remove_var("RNB_CUDA_QWEN35_GROUP_META_FROM_IDS");
        assert!(!qwen35_group_meta_from_ids_enabled());

        std::env::set_var("RNB_CUDA_QWEN35_GROUP_META_FROM_IDS", "1");
        assert!(!qwen35_group_meta_from_ids_enabled());
        restore_env("RNB_CUDA_QWEN35_GROUP_META_FROM_IDS", prev);
    }

    #[test]
    fn pack4_group8_handoff_map_matches_down_group4_meta() {
        let gate_up_group_meta = [0, 8, 8, 1, 9, 5];
        let down_group_meta = [0, 4, 4, 4, 8, 1, 9, 4, 13, 1];

        let offsets =
            qwen35_pack4_group_offsets_for_down_group_meta(&gate_up_group_meta, &down_group_meta)
                .expect("group8 pack offsets should cover down group4 meta");

        assert_eq!(offsets, vec![0, 2, 3, 5]);
    }

    #[test]
    fn pack4_group8_handoff_map_rejects_shifted_down_meta() {
        let gate_up_group_meta = [0, 8, 8, 1, 9, 5];
        let shifted_down_group_meta = [0, 4, 5, 4, 8, 1, 9, 4, 13, 1];

        let err = qwen35_pack4_group_offsets_for_down_group_meta(
            &gate_up_group_meta,
            &shifted_down_group_meta,
        )
        .expect_err("shifted down group should not cover the group8 handoff");

        assert!(err.contains("must start at slot 4"));
    }

    #[test]
    fn gate_up_activation_runner_preserves_existing_dispatch_priority() {
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner(true, true, true, true),
            Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner(false, true, true, true),
            Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group4
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner(false, false, true, true),
            Qwen35SelectedSparseGateUpActivationRunner::FusedSiluGroup8
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner(false, false, false, false),
            Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner(false, false, false, true),
            Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped
        );
    }

    #[test]
    fn gate_up_activation_runner_marks_only_separate_paths_for_silu_launch() {
        assert!(!Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8.needs_separate_silu());
        assert!(!Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group4.needs_separate_silu());
        assert!(!Qwen35SelectedSparseGateUpActivationRunner::FusedSiluGroup8.needs_separate_silu());
        assert!(
            Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped.needs_separate_silu()
        );
        assert!(Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped.needs_separate_silu());
    }

    fn sparse_descriptor(
        gate_up_group: Option<usize>,
        down_group: Option<usize>,
        q4_gate_up_silu_fused: bool,
        q4_gate_up_silu_pack4_f32: bool,
        q4_gate_up_silu_pack4_group8: bool,
        q6_down_pack4_f32: bool,
    ) -> Qwen35SelectedSparseExecutionDescriptor {
        qwen35_selected_sparse_execution_descriptor(Qwen35SelectedSparseDescriptorInput {
            slots: 4,
            token_count: 2,
            route_from_device: false,
            slot_pointer_source: Qwen35SelectedSparseSlotPointerSource::Host,
            gate_up_group,
            down_group,
            down_quant: 14,
            zero_output: true,
            q4_gate_up_silu_fused,
            q4_gate_up_silu_pack4_f32,
            q4_gate_up_silu_pack4_group8,
            q6_down_q8dot: false,
            q6_down_pack4_f32,
            q6_down_pack4_f32_vec4: false,
        })
        .expect("selected sparse descriptor")
    }

    #[test]
    fn gate_up_activation_runner_uses_descriptor_layout() {
        let group8_pack4 = sparse_descriptor(Some(8), Some(4), false, true, true, true);
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(
                &group8_pack4,
                true,
                true
            )
            .expect("group8 pack4 descriptor runner"),
            Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8
        );

        let group4_pack4 = sparse_descriptor(Some(4), Some(4), false, true, false, true);
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(
                &group4_pack4,
                false,
                true
            )
            .expect("group4 pack4 descriptor runner"),
            Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group4
        );

        let fused = sparse_descriptor(Some(8), Some(4), true, false, false, false);
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(&fused, false, true)
                .expect("fused descriptor runner"),
            Qwen35SelectedSparseGateUpActivationRunner::FusedSiluGroup8
        );

        let separate = sparse_descriptor(None, Some(4), false, false, false, false);
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(
                &separate, false, false
            )
            .expect("ungrouped descriptor runner"),
            Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(
                &separate, false, true
            )
            .expect("grouped descriptor runner"),
            Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped
        );
    }

    #[test]
    fn gate_up_activation_runner_rejects_missing_descriptor_materialization() {
        let group8_pack4 = sparse_descriptor(Some(8), Some(4), false, true, true, true);
        let err = qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(
            &group8_pack4,
            false,
            true,
        )
        .expect_err("group8 pack4 descriptor requires pack offsets");
        assert!(err.contains("requires pack offsets"));

        let fused = sparse_descriptor(Some(8), Some(4), true, false, false, false);
        let err =
            qwen35_selected_sparse_gate_up_activation_runner_from_descriptor(&fused, false, false)
                .expect_err("fused descriptor requires group metadata");
        assert!(err.contains("requires gate/up group metadata"));
    }

    #[test]
    fn gate_up_launch_payload_materializes_runner_requirements() {
        let gate_up_group_meta = [0, 8, 8, 1];
        let down_group_meta = [0, 4, 4, 4, 8, 1];
        let offsets = [0, 2, 3];

        assert_eq!(
            qwen35_selected_sparse_gate_up_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8,
                Some(77),
                Some(&offsets),
                &gate_up_group_meta,
                &down_group_meta,
                1000,
                16,
            )
            .expect("group8 pack4 payload"),
            Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group8 {
                packed_dev: 77,
                pack_group_offsets_dev: 1016,
                pack_group_count: 2,
            }
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group4,
                Some(88),
                None,
                &gate_up_group_meta,
                &down_group_meta,
                1000,
                16,
            )
            .expect("group4 pack4 payload"),
            Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group4 {
                packed_dev: 88,
                down_group_count: 3,
                reload_down_group_meta: true,
            }
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::FusedSiluGroup8,
                None,
                None,
                &gate_up_group_meta,
                &down_group_meta,
                1000,
                16,
            )
            .expect("fused payload"),
            Qwen35SelectedSparseGateUpLaunchPayload::FusedSiluGroup8 { group_count: 2 }
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped,
                None,
                None,
                &gate_up_group_meta,
                &down_group_meta,
                1000,
                16,
            )
            .expect("ungrouped payload"),
            Qwen35SelectedSparseGateUpLaunchPayload::SeparateUngrouped
        );
        assert_eq!(
            qwen35_selected_sparse_gate_up_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped,
                None,
                None,
                &gate_up_group_meta,
                &down_group_meta,
                1000,
                16,
            )
            .expect("grouped payload"),
            Qwen35SelectedSparseGateUpLaunchPayload::SeparateGrouped { group_count: 2 }
        );

        let err = qwen35_selected_sparse_gate_up_launch_payload(
            Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8,
            Some(77),
            None,
            &gate_up_group_meta,
            &down_group_meta,
            1000,
            16,
        )
        .expect_err("group8 pack4 requires offsets");
        assert!(err.contains("pack offsets"));

        let err = qwen35_selected_sparse_gate_up_launch_payload(
            Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group4,
            None,
            None,
            &gate_up_group_meta,
            &down_group_meta,
            1000,
            16,
        )
        .expect_err("pack4 requires packed activation buffer");
        assert!(err.contains("packed activation buffer"));
    }

    #[test]
    fn silu_launch_payload_materializes_runner_requirements() {
        let gate_up_group_meta = [0, 8, 8, 1];
        let down_group_meta = [0, 4, 4, 4, 8, 1];

        assert_eq!(
            qwen35_selected_sparse_silu_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8,
                Some((11, 12)),
                Some(13),
                &gate_up_group_meta,
                &down_group_meta,
            ),
            Qwen35SelectedSparseSiluLaunchPayload::None
        );
        assert_eq!(
            qwen35_selected_sparse_silu_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped,
                Some((11, 12)),
                Some(13),
                &gate_up_group_meta,
                &down_group_meta,
            ),
            Qwen35SelectedSparseSiluLaunchPayload::Q8 {
                qs_dev: 11,
                ds_dev: 12,
            }
        );
        assert_eq!(
            qwen35_selected_sparse_silu_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped,
                None,
                Some(13),
                &gate_up_group_meta,
                &down_group_meta,
            ),
            Qwen35SelectedSparseSiluLaunchPayload::Pack4F32Group4 {
                packed_dev: 13,
                group_count: 3,
                reload_down_group_meta: true,
            }
        );
        assert_eq!(
            qwen35_selected_sparse_silu_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped,
                None,
                Some(13),
                &down_group_meta,
                &down_group_meta,
            ),
            Qwen35SelectedSparseSiluLaunchPayload::Pack4F32Group4 {
                packed_dev: 13,
                group_count: 3,
                reload_down_group_meta: false,
            }
        );
        assert_eq!(
            qwen35_selected_sparse_silu_launch_payload(
                Qwen35SelectedSparseGateUpActivationRunner::SeparateUngrouped,
                None,
                None,
                &gate_up_group_meta,
                &down_group_meta,
            ),
            Qwen35SelectedSparseSiluLaunchPayload::Plain
        );
    }

    #[test]
    fn down_meta_stage_preserves_existing_priority() {
        assert_eq!(
            qwen35_selected_sparse_down_meta_stage(true, true, true, true, true, false),
            Qwen35SelectedSparseDownMetaStage::TokenMajor
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_stage(false, true, true, true, true, false),
            Qwen35SelectedSparseDownMetaStage::RunTile
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_stage(false, false, true, true, true, true),
            Qwen35SelectedSparseDownMetaStage::UploadDownMeta
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_stage(false, false, true, false, true, false),
            Qwen35SelectedSparseDownMetaStage::ReuseExistingDownMeta
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_stage(false, false, false, false, true, false),
            Qwen35SelectedSparseDownMetaStage::UploadDownMeta
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_stage(false, false, false, false, true, true),
            Qwen35SelectedSparseDownMetaStage::None
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_stage(false, false, false, false, false, false),
            Qwen35SelectedSparseDownMetaStage::None
        );
    }

    #[test]
    fn down_meta_staging_payload_materializes_required_inputs() {
        let token_major = crate::runtime::Qwen35DownTokenMajorPlan {
            token_offsets: vec![0, 2, 3],
            slot_indices: vec![2, 0, 1],
        };
        let run_tile_meta = [3, 0, 2, 0];
        let down_group_meta = [0, 4, 4, 2];

        assert_eq!(
            qwen35_selected_sparse_down_meta_staging_payload(
                Qwen35SelectedSparseDownMetaStage::TokenMajor,
                Some(&token_major),
                Some(&run_tile_meta),
                &down_group_meta,
            )
            .expect("token-major payload"),
            Qwen35SelectedSparseDownMetaStagingPayload::TokenMajor {
                token_offsets: &[0, 2, 3],
                slot_indices: &[2, 0, 1],
            }
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_staging_payload(
                Qwen35SelectedSparseDownMetaStage::RunTile,
                Some(&token_major),
                Some(&run_tile_meta),
                &down_group_meta,
            )
            .expect("run-tile payload"),
            Qwen35SelectedSparseDownMetaStagingPayload::RunTile(&run_tile_meta)
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_staging_payload(
                Qwen35SelectedSparseDownMetaStage::UploadDownMeta,
                None,
                None,
                &down_group_meta,
            )
            .expect("down meta payload"),
            Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(&down_group_meta)
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_staging_payload(
                Qwen35SelectedSparseDownMetaStage::ReuseExistingDownMeta,
                None,
                None,
                &down_group_meta,
            )
            .expect("reuse payload"),
            Qwen35SelectedSparseDownMetaStagingPayload::ReuseExistingDownMeta
        );
        assert_eq!(
            qwen35_selected_sparse_down_meta_staging_payload(
                Qwen35SelectedSparseDownMetaStage::None,
                None,
                None,
                &down_group_meta,
            )
            .expect("none payload"),
            Qwen35SelectedSparseDownMetaStagingPayload::None
        );

        let err = qwen35_selected_sparse_down_meta_staging_payload(
            Qwen35SelectedSparseDownMetaStage::TokenMajor,
            None,
            Some(&run_tile_meta),
            &down_group_meta,
        )
        .expect_err("token-major stage requires token-major plan");
        assert!(err.contains("token-major"));

        let err = qwen35_selected_sparse_down_meta_staging_payload(
            Qwen35SelectedSparseDownMetaStage::RunTile,
            Some(&token_major),
            None,
            &down_group_meta,
        )
        .expect_err("run-tile stage requires run-tile meta");
        assert!(err.contains("run-tile"));
    }

    #[test]
    fn down_launch_runner_preserves_existing_priority() {
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                14, true, true, false, true, true, true, true, true, true, true
            )
            .expect("token-major priority"),
            Qwen35SelectedSparseDownLaunchRunner::Q6TokenMajor
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                14, true, true, false, false, true, true, true, true, true, true
            )
            .expect("q8dot priority"),
            Qwen35SelectedSparseDownLaunchRunner::Q6Q8DotGroup4
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                14, true, true, false, false, false, true, true, true, true, true
            )
            .expect("pack4 priority"),
            Qwen35SelectedSparseDownLaunchRunner::Q6Pack4F32Group4
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                14, true, true, false, false, false, false, true, true, true, true
            )
            .expect("run-tiled priority"),
            Qwen35SelectedSparseDownLaunchRunner::Q6RunTiled4
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                14, true, true, false, false, false, false, false, true, true, true
            )
            .expect("run-batched-ref priority"),
            Qwen35SelectedSparseDownLaunchRunner::Q6RunBatchedRef
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                14, true, true, false, false, false, false, false, false, true, true
            )
            .expect("run-batched8 priority"),
            Qwen35SelectedSparseDownLaunchRunner::Q6RunBatched8
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                14, true, true, false, false, false, false, false, false, false, true
            )
            .expect("full4 split priority"),
            Qwen35SelectedSparseDownLaunchRunner::Q6Full4Split
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                12, true, true, true, false, false, false, false, false, false, false
            )
            .expect("q4 group4 priority"),
            Qwen35SelectedSparseDownLaunchRunner::Q4Group4
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner(
                12, true, true, false, false, false, false, false, false, false, false
            )
            .expect("q4 group4 disabled falls back to warp4"),
            Qwen35SelectedSparseDownLaunchRunner::Q4Warp4
        );
    }

    #[test]
    fn down_launch_runner_maps_descriptor_modes() {
        let q8dot =
            qwen35_selected_sparse_execution_descriptor(Qwen35SelectedSparseDescriptorInput {
                slots: 4,
                token_count: 2,
                route_from_device: false,
                slot_pointer_source: Qwen35SelectedSparseSlotPointerSource::Host,
                gate_up_group: Some(4),
                down_group: Some(4),
                down_quant: 14,
                zero_output: true,
                q4_gate_up_silu_fused: false,
                q4_gate_up_silu_pack4_f32: false,
                q4_gate_up_silu_pack4_group8: false,
                q6_down_q8dot: true,
                q6_down_pack4_f32: false,
                q6_down_pack4_f32_vec4: false,
            })
            .expect("q8dot descriptor");
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner_from_descriptor(
                &q8dot, true, true, false, false, false, false, false, false
            )
            .expect("q8dot descriptor runner"),
            Qwen35SelectedSparseDownLaunchRunner::Q6Q8DotGroup4
        );

        let q6_existing = sparse_descriptor(Some(4), Some(4), false, false, false, false);
        assert_eq!(
            qwen35_selected_sparse_down_launch_runner_from_descriptor(
                &q6_existing,
                true,
                true,
                false,
                false,
                false,
                false,
                false,
                false
            )
            .expect("q6 existing descriptor runner"),
            Qwen35SelectedSparseDownLaunchRunner::Q6Group4
        );
    }

    #[test]
    fn down_launch_payload_materializes_runner_requirements() {
        let token_major = crate::runtime::Qwen35DownTokenMajorPlan {
            token_offsets: vec![0, 2, 3],
            slot_indices: vec![2, 0, 1],
        };
        let run_tile_meta = [3, 0, 2, 0, 2, 1, 1, 1, 1, 0];
        let split = crate::runtime::Qwen35GroupMetaLenSplit {
            matching: vec![0, 4, 4, 4],
            other: vec![8, 2],
        };
        let down_group_meta = [0, 4, 4, 2, 6, 1];

        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q6TokenMajor,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("token-major payload"),
            Qwen35SelectedSparseDownLaunchPayload::Q6TokenMajor {
                token_offsets_bytes: std::mem::size_of_val(token_major.token_offsets.as_slice()),
            }
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q6Q8DotGroup4,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("q8dot payload"),
            Qwen35SelectedSparseDownLaunchPayload::Q6Q8DotGroup4 {
                qs_dev: 11,
                ds_dev: 12,
                group_count: 3,
            }
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q6Pack4F32Group4,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("pack4 payload"),
            Qwen35SelectedSparseDownLaunchPayload::Q6Pack4F32Group4 {
                packed_dev: 13,
                group_count: 3,
            }
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q6RunTiled4,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("run-tiled payload"),
            Qwen35SelectedSparseDownLaunchPayload::Q6RunTiled4 { run_count: 2 }
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q6RunBatched8,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("run-batched payload"),
            Qwen35SelectedSparseDownLaunchPayload::Q6RunBatched8 { run_count: 5 }
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q6Full4Split,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("split payload"),
            Qwen35SelectedSparseDownLaunchPayload::Q6Full4Split {
                matching_meta: &[0, 4, 4, 4],
                other_meta: &[8, 2],
            }
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q4Group4,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("q4 group4 payload"),
            Qwen35SelectedSparseDownLaunchPayload::Group4 {
                kernel: "rnb_q4k_selected_down_accum_by_token_group4",
                group_count: 3,
            }
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q6Warp4,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("q6 warp4 payload"),
            Qwen35SelectedSparseDownLaunchPayload::Warp4 {
                kernel: "rnb_q6k_selected_down_accum_by_token_warp4",
            }
        );
        assert_eq!(
            qwen35_selected_sparse_down_launch_payload(
                Qwen35SelectedSparseDownLaunchRunner::Q5Scalar,
                Some(&token_major),
                Some((11, 12)),
                Some(13),
                Some(&run_tile_meta),
                Some(&split),
                &down_group_meta,
            )
            .expect("q5 scalar payload"),
            Qwen35SelectedSparseDownLaunchPayload::Scalar {
                kernel: "rnb_q5k_selected_down_accum_by_token",
            }
        );

        let err = qwen35_selected_sparse_down_launch_payload(
            Qwen35SelectedSparseDownLaunchRunner::Q6Q8DotGroup4,
            Some(&token_major),
            None,
            Some(13),
            Some(&run_tile_meta),
            Some(&split),
            &down_group_meta,
        )
        .expect_err("q8dot requires q8 buffers");
        assert!(err.contains("q8dot"));

        let err = qwen35_selected_sparse_down_launch_payload(
            Qwen35SelectedSparseDownLaunchRunner::Q6RunTiled4,
            Some(&token_major),
            Some((11, 12)),
            Some(13),
            None,
            Some(&split),
            &down_group_meta,
        )
        .expect_err("run-tiled requires meta");
        assert!(err.contains("run-tile"));
    }

    #[test]
    fn exact_runner_plan_groups_descriptor_backed_launch_decisions() {
        let descriptor = sparse_descriptor(Some(8), Some(4), false, true, true, true);

        let plan =
            qwen35_selected_sparse_exact_runner_plan(Qwen35SelectedSparseExactRunnerPlanInput {
                descriptor: Some(&descriptor),
                has_pack4_group_offsets: true,
                q4_gate_up_silu_pack4_f32: true,
                q4_gate_up_silu_fused: false,
                has_gate_up_group_meta: true,
                has_down_token_major_plan: false,
                has_down_run_tile_meta: false,
                q6_down_pack4_f32: true,
                group4_down: true,
                gate_up_meta_matches_down_meta: false,
                down_quant: 14,
                warp_down: true,
                q4_down_group4: false,
                down_token_major: false,
                q6_down_q8dot: false,
                q6_down_run_tiled4: false,
                q6_down_run_batched_ref: false,
                q6_down_run_batched8: false,
                q6_down_full4_split: false,
            })
            .expect("descriptor-backed exact runner plan");

        assert_eq!(
            plan.gate_up,
            Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8
        );
        assert_eq!(
            plan.down_meta,
            Qwen35SelectedSparseDownMetaStage::UploadDownMeta
        );
        assert_eq!(
            plan.down_launch,
            Qwen35SelectedSparseDownLaunchRunner::Q6Pack4F32Group4
        );
    }

    #[test]
    fn exact_runner_payload_groups_stage_payloads() {
        let gate_up_group_meta = [0, 8, 8, 1];
        let down_group_meta = [0, 4, 4, 4, 8, 1];
        let offsets = [0, 2, 3];
        let token_major = crate::runtime::Qwen35DownTokenMajorPlan {
            token_offsets: vec![0, 2, 3],
            slot_indices: vec![2, 0, 1],
        };
        let run_tile_meta = [3, 0, 2, 0, 2, 1, 1, 1, 1, 0];
        let split = crate::runtime::Qwen35GroupMetaLenSplit {
            matching: vec![0, 4],
            other: vec![4, 2],
        };

        let payload = qwen35_selected_sparse_exact_runner_payload(
            Qwen35SelectedSparseExactRunnerPlan {
                gate_up: Qwen35SelectedSparseGateUpActivationRunner::Pack4F32Group8,
                down_meta: Qwen35SelectedSparseDownMetaStage::UploadDownMeta,
                down_launch: Qwen35SelectedSparseDownLaunchRunner::Q6Pack4F32Group4,
            },
            Some(77),
            Some(&offsets),
            &gate_up_group_meta,
            &down_group_meta,
            1000,
            16,
            Some(&token_major),
            Some((11, 12)),
            Some(&run_tile_meta),
            Some(&split),
        )
        .expect("group8 pack4 exact payload");
        let expected_down_meta =
            Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(&down_group_meta);

        assert_eq!(
            payload,
            Qwen35SelectedSparseExactRunnerPayload {
                gate_up: Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group8 {
                    packed_dev: 77,
                    pack_group_offsets_dev: 1016,
                    pack_group_count: 2,
                },
                silu: Qwen35SelectedSparseSiluLaunchPayload::None,
                down_meta: expected_down_meta,
                down_launch: Qwen35SelectedSparseDownLaunchPayload::Q6Pack4F32Group4 {
                    packed_dev: 77,
                    group_count: 3,
                },
            }
        );

        let payload = qwen35_selected_sparse_exact_runner_payload(
            Qwen35SelectedSparseExactRunnerPlan {
                gate_up: Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped,
                down_meta: Qwen35SelectedSparseDownMetaStage::None,
                down_launch: Qwen35SelectedSparseDownLaunchRunner::Q6Group4,
            },
            Some(88),
            None,
            &gate_up_group_meta,
            &down_group_meta,
            1000,
            16,
            Some(&token_major),
            None,
            Some(&run_tile_meta),
            Some(&split),
        )
        .expect("separate exact payload");

        assert_eq!(
            payload,
            Qwen35SelectedSparseExactRunnerPayload {
                gate_up: Qwen35SelectedSparseGateUpLaunchPayload::SeparateGrouped {
                    group_count: 2,
                },
                silu: Qwen35SelectedSparseSiluLaunchPayload::Pack4F32Group4 {
                    packed_dev: 88,
                    group_count: 3,
                    reload_down_group_meta: true,
                },
                down_meta: Qwen35SelectedSparseDownMetaStagingPayload::None,
                down_launch: Qwen35SelectedSparseDownLaunchPayload::Group4 {
                    kernel: "rnb_q6k_selected_down_accum_by_token_group4",
                    group_count: 3,
                },
            }
        );

        let err = qwen35_selected_sparse_exact_runner_payload(
            Qwen35SelectedSparseExactRunnerPlan {
                gate_up: Qwen35SelectedSparseGateUpActivationRunner::SeparateGrouped,
                down_meta: Qwen35SelectedSparseDownMetaStage::None,
                down_launch: Qwen35SelectedSparseDownLaunchRunner::Q6Q8DotGroup4,
            },
            Some(88),
            None,
            &gate_up_group_meta,
            &down_group_meta,
            1000,
            16,
            Some(&token_major),
            None,
            Some(&run_tile_meta),
            Some(&split),
        )
        .expect_err("q8dot down launch requires q8 buffers");

        assert!(err.contains("q8dot"));
    }

    #[test]
    fn compound_runner_payload_filters_promoted_exact_payload() {
        let promoted = Qwen35SelectedSparseExactRunnerPayload {
            gate_up: Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group8 {
                packed_dev: 77,
                pack_group_offsets_dev: 1016,
                pack_group_count: 2,
            },
            silu: Qwen35SelectedSparseSiluLaunchPayload::None,
            down_meta: Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(&[0, 4, 4, 4, 8, 1]),
            down_launch: Qwen35SelectedSparseDownLaunchPayload::Q6Pack4F32Group4 {
                packed_dev: 77,
                group_count: 3,
            },
        };
        let fallback = Qwen35SelectedSparseExactRunnerPayload {
            gate_up: Qwen35SelectedSparseGateUpLaunchPayload::SeparateGrouped { group_count: 2 },
            silu: Qwen35SelectedSparseSiluLaunchPayload::Plain,
            down_meta: Qwen35SelectedSparseDownMetaStagingPayload::None,
            down_launch: Qwen35SelectedSparseDownLaunchPayload::Group4 {
                kernel: "rnb_q6k_selected_down_accum_by_token_group4",
                group_count: 3,
            },
        };

        assert!(qwen35_selected_sparse_compound_reference_payload(
            Qwen35SelectedSparseRunnerMode::CompoundExactReference,
            promoted,
        )
        .is_some());
        assert_eq!(
            qwen35_selected_sparse_compound_reference_payload(
                Qwen35SelectedSparseRunnerMode::ExactReference,
                promoted,
            ),
            None
        );
        assert_eq!(
            qwen35_selected_sparse_compound_reference_payload(
                Qwen35SelectedSparseRunnerMode::CompoundExactReference,
                fallback,
            ),
            None
        );
    }

    #[test]
    fn compound_reference_payload_materializes_runner_api() {
        let down_meta = [0, 4, 4, 4, 8, 1];
        let promoted = Qwen35SelectedSparseExactRunnerPayload {
            gate_up: Qwen35SelectedSparseGateUpLaunchPayload::Pack4F32Group8 {
                packed_dev: 77,
                pack_group_offsets_dev: 1016,
                pack_group_count: 2,
            },
            silu: Qwen35SelectedSparseSiluLaunchPayload::None,
            down_meta: Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(&down_meta),
            down_launch: Qwen35SelectedSparseDownLaunchPayload::Q6Pack4F32Group4 {
                packed_dev: 77,
                group_count: 3,
            },
        };

        assert_eq!(
            qwen35_selected_sparse_compound_reference_payload(
                Qwen35SelectedSparseRunnerMode::CompoundExactReference,
                promoted,
            ),
            Some(
                Qwen35SelectedSparseCompoundRunnerPayload::Q4GateUpGroup8Q6DownPack4 {
                    packed_dev: 77,
                    pack_group_offsets_dev: 1016,
                    pack_group_count: 2,
                    down_meta: Qwen35SelectedSparseDownMetaStagingPayload::DownMeta(&down_meta),
                    down_group_count: 3,
                }
            )
        );
        assert_eq!(
            qwen35_selected_sparse_compound_reference_payload(
                Qwen35SelectedSparseRunnerMode::ExactReference,
                promoted,
            ),
            None
        );
    }

    #[test]
    fn compound_graph_meta_workspace_keeps_gate_and_down_meta_disjoint() {
        assert_eq!(
            qwen35_selected_sparse_compound_graph_meta_workspace_bytes(16, 12, 24, false),
            28
        );
        assert_eq!(
            qwen35_selected_sparse_compound_graph_meta_workspace_bytes(16, 12, 24, true),
            52
        );
        assert_eq!(
            qwen35_selected_sparse_compound_graph_down_meta_dev(1000, 16, 12, 24, true)
                .expect("graph down meta dev"),
            Some(1028)
        );
        assert_eq!(
            qwen35_selected_sparse_compound_graph_down_meta_dev(1000, 16, 12, 24, false)
                .expect("disabled graph has no alternate down meta"),
            None
        );
        let err = qwen35_selected_sparse_compound_graph_down_meta_dev(0, 16, 12, 24, true)
            .expect_err("enabled graph needs a workspace");
        assert!(err.contains("requires group-meta workspace"));
    }

    #[test]
    fn compound_graph_captures_zero_only_for_untraced_compound_graph_path() {
        assert!(qwen35_selected_sparse_compound_graph_captures_zero_output(
            true, true, true, false, true
        ));
        assert!(!qwen35_selected_sparse_compound_graph_captures_zero_output(
            false, true, true, false, true
        ));
        assert!(!qwen35_selected_sparse_compound_graph_captures_zero_output(
            true, false, true, false, true
        ));
        assert!(!qwen35_selected_sparse_compound_graph_captures_zero_output(
            true, true, false, false, true
        ));
        assert!(!qwen35_selected_sparse_compound_graph_captures_zero_output(
            true, true, true, true, true
        ));
        assert!(!qwen35_selected_sparse_compound_graph_captures_zero_output(
            true, true, true, false, false
        ));
    }

    #[test]
    fn expert_range_upload_plan_merges_small_gaps() {
        let plan = build_expert_range_upload_plan(&[1, 1, 2, 4], 8, 1, 1500)
            .expect("small expert gap should be merged");

        assert_eq!(
            plan.ranges,
            vec![ExpertRangeUpload {
                expert_start: 1,
                expert_end: 5,
                slab_expert_offset: 0,
            }]
        );
        assert_eq!(plan.expert_offsets[1], Some(0));
        assert_eq!(plan.expert_offsets[2], Some(1));
        assert_eq!(plan.expert_offsets[4], Some(3));
        assert_eq!(plan.selected_experts, 3);
        assert_eq!(plan.slab_experts, 4);
    }

    #[test]
    fn expert_range_upload_plan_rejects_excessive_overcopy() {
        let plan = build_expert_range_upload_plan(&[0, 15], 16, 16, 1200);
        assert!(plan.is_none());
    }

    #[test]
    fn prefill_hot_resident_plan_prioritizes_aggregated_route_weight() {
        let gate0 = [0u8; 4];
        let up0 = [1u8; 4];
        let down0 = [2u8; 8];
        let gate1 = [3u8; 4];
        let up1 = [4u8; 4];
        let down1 = [5u8; 8];
        let gate2 = [6u8; 4];
        let up2 = [7u8; 4];
        let down2 = [8u8; 8];
        let gate_weights = vec![&gate0[..], &gate1[..], &gate0[..], &gate2[..]];
        let up_weights = vec![&up0[..], &up1[..], &up0[..], &up2[..]];
        let down_weights = vec![&down0[..], &down1[..], &down0[..], &down2[..]];
        let route_weights = vec![0.35, 0.90, 0.40, 0.20];
        let resident = std::collections::HashSet::new();

        let plan = build_prefill_hot_resident_plan(
            &gate_weights,
            &up_weights,
            &down_weights,
            &route_weights,
            &resident,
            32,
        );

        assert_eq!(plan.slots, vec![1, 0]);
        assert_eq!(plan.bytes, 32);
    }

    #[test]
    fn prefill_hot_resident_plan_counts_only_missing_weight_bytes() {
        let gate0 = [0u8; 4];
        let up0 = [1u8; 4];
        let down0 = [2u8; 8];
        let gate1 = [3u8; 4];
        let up1 = [4u8; 4];
        let down1 = [5u8; 8];
        let gate_weights = vec![&gate0[..], &gate1[..]];
        let up_weights = vec![&up0[..], &up1[..]];
        let down_weights = vec![&down0[..], &down1[..]];
        let route_weights = vec![0.70, 0.60];
        let resident = std::collections::HashSet::from([q4k_resident_key(&gate0)]);

        let plan = build_prefill_hot_resident_plan(
            &gate_weights,
            &up_weights,
            &down_weights,
            &route_weights,
            &resident,
            12,
        );

        assert_eq!(plan.slots, vec![0]);
        assert_eq!(plan.bytes, 12);
    }
}
