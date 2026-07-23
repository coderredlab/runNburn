use super::super::super::*;
use super::by_token::qwen35_selected_base_upload_source;

const STREAM_BUFFER_COUNT: usize = 2;
const CUDA_EVENT_DISABLE_TIMING: u32 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Qwen35StreamBatchRange {
    slot_start: usize,
    slot_end: usize,
}

fn qwen35_stream_batch_ranges(
    expert_ids: &[u32],
    batch_count: usize,
) -> Result<Option<Vec<Qwen35StreamBatchRange>>, String> {
    if expert_ids.is_empty() {
        return Ok(None);
    }
    if batch_count < STREAM_BUFFER_COUNT {
        return Err(format!(
            "Qwen35 stream requires at least {STREAM_BUFFER_COUNT} batches, got {batch_count}"
        ));
    }
    let mut expert_starts = vec![0usize];
    for slot in 1..expert_ids.len() {
        if expert_ids[slot] < expert_ids[slot - 1] {
            return Err(format!(
                "Qwen35 stream requires expert-sorted slots: slot={slot} previous={} current={}",
                expert_ids[slot - 1],
                expert_ids[slot]
            ));
        }
        if expert_ids[slot] != expert_ids[slot - 1] {
            expert_starts.push(slot);
        }
    }
    if expert_starts.len() < batch_count {
        return Ok(None);
    }
    let mut ranges = Vec::with_capacity(batch_count);
    for batch_index in 0..batch_count {
        let start_expert = batch_index * expert_starts.len() / batch_count;
        let end_expert = (batch_index + 1) * expert_starts.len() / batch_count;
        ranges.push(Qwen35StreamBatchRange {
            slot_start: expert_starts[start_expert],
            slot_end: if batch_index + 1 == batch_count {
                expert_ids.len()
            } else {
                expert_starts[end_expert]
            },
        });
    }
    Ok(Some(ranges))
}

struct Qwen35StreamBatchPlan {
    range: Qwen35StreamBatchRange,
    offsets: Qwen35SelectedBaseSlotOffsets,
    slab_bytes: usize,
}

impl CudaState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::runtime) fn qwen35_sparse_experts_selected_base_stream_to_dev(
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
        input_dev: u64,
        output_dev: u64,
    ) -> Result<bool, String> {
        let batch_count = match std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_STREAM_BATCHES") {
            Ok(value) => value.parse::<usize>().map_err(|error| {
                format!("invalid RNB_CUDA_QWEN35_SELECTED_BASE_STREAM_BATCHES={value}: {error}")
            })?,
            Err(_) => expert_ids.len().div_ceil(token_count).clamp(2, 8),
        };
        if !matches!(batch_count, 2 | 4 | 8) {
            return Err(format!(
                "RNB_CUDA_QWEN35_SELECTED_BASE_STREAM_BATCHES must be 2, 4, or 8, got {batch_count}"
            ));
        }
        let Some(ranges) = qwen35_stream_batch_ranges(expert_ids, batch_count)? else {
            return Ok(false);
        };
        if route_weights.len() != expert_ids.len() || token_ids.len() != expert_ids.len() {
            return Err(format!(
                "Qwen35 selected-base stream route length mismatch: experts={} routes={} tokens={}",
                expert_ids.len(),
                route_weights.len(),
                token_ids.len()
            ));
        }
        let n_expert = qwen35_selected_base_full_layer_expert_count(
            gate_all, up_all, down_all, down_quant, n_ff, n_embd,
        )?;
        let mut batches = Vec::with_capacity(batch_count);
        for range in ranges {
            let offsets = qwen35_selected_base_slot_offsets_from_full_layer(
                gate_all,
                up_all,
                down_all,
                &expert_ids[range.slot_start..range.slot_end],
                down_quant,
                n_ff,
                n_embd,
            )?;
            let slab_bytes =
                qwen35_selected_base_temp_slab_device_ptr_plan(&offsets, n_expert, 0)?.slab_bytes;
            if slab_bytes == 0 {
                return Ok(false);
            }
            batches.push(Qwen35StreamBatchPlan {
                range,
                offsets,
                slab_bytes,
            });
        }

        let buffer_stride = batches
            .iter()
            .map(|batch| batch.slab_bytes)
            .max()
            .unwrap_or(0);
        let slab_devs = [
            self.qwen35_selected_base_stream_slab_a_ptr(buffer_stride)?,
            self.qwen35_selected_base_stream_slab_b_ptr(buffer_stride)?,
        ];
        let mut events = Vec::with_capacity(STREAM_BUFFER_COUNT * 2);
        for _ in 0..STREAM_BUFFER_COUNT * 2 {
            match unsafe { self.api.event_create(CUDA_EVENT_DISABLE_TIMING) } {
                Ok(event) => events.push(event),
                Err(error) => {
                    for event in events {
                        let _ = unsafe { self.api.event_destroy(event) };
                    }
                    return Err(error);
                }
            }
        }

        let trace = std::env::var("RNB_CUDA_QWEN35_SELECTED_BASE_STREAM_TRACE")
            .ok()
            .as_deref()
            == Some("1");
        let result = (|| -> Result<(), String> {
            self.launch_zero_f32(output_dev, token_count * n_embd)?;
            for (batch_index, batch) in batches.iter().enumerate() {
                let buffer_index = batch_index % STREAM_BUFFER_COUNT;
                if batch_index >= STREAM_BUFFER_COUNT {
                    unsafe {
                        self.api.stream_wait_event(
                            self.copy_stream,
                            events[STREAM_BUFFER_COUNT + buffer_index],
                        )?;
                    }
                }
                let device_base = slab_devs[buffer_index];
                let plan = qwen35_selected_base_temp_slab_device_ptr_plan(
                    &batch.offsets,
                    n_expert,
                    device_base,
                )?;
                for upload in &plan.uploads {
                    let weights = qwen35_selected_base_upload_source(
                        upload,
                        gate_all,
                        up_all,
                        down_all,
                        "selected-base stream",
                    )?;
                    let dst = device_base
                        .checked_add(u64::try_from(upload.slab_byte_offset).map_err(|_| {
                            format!(
                                "Qwen35 selected-base stream upload offset exceeds u64: {}",
                                upload.slab_byte_offset
                            )
                        })?)
                        .ok_or_else(|| {
                            format!(
                                "Qwen35 selected-base stream upload pointer overflows: base={device_base} offset={}",
                                upload.slab_byte_offset
                            )
                        })?;
                    unsafe {
                        self.api.memcpy_htod_async(
                            dst,
                            weights.as_ptr().cast::<libc::c_void>(),
                            weights.len(),
                            self.copy_stream,
                        )?;
                    }
                }
                unsafe {
                    self.api
                        .event_record(events[buffer_index], self.copy_stream)?;
                    self.api
                        .stream_wait_event(self.stream, events[buffer_index])?;
                }

                let slot_range = batch.range.slot_start..batch.range.slot_end;
                let group_meta2 = qwen35_selected_base_group_meta_from_offsets(&batch.offsets, 2);
                let group_meta4 = qwen35_selected_base_group_meta_from_offsets(&batch.offsets, 4);
                let group_meta8 = qwen35_selected_base_group_meta_from_offsets(&batch.offsets, 8);
                let group_meta16 = qwen35_selected_base_group_meta_from_offsets(&batch.offsets, 16);
                let group_meta32 = qwen35_selected_base_group_meta_from_offsets(&batch.offsets, 32);
                let group_meta64 = qwen35_selected_base_group_meta_from_offsets(&batch.offsets, 64);
                let prepared = PreparedQwen35SparseSlots {
                    gate_ptrs: Vec::new(),
                    up_ptrs: Vec::new(),
                    down_ptrs: Vec::new(),
                    slot_count: None,
                    temp_slab_ptrs: Vec::new(),
                    copy_stream_upload: false,
                    device_slot_ptrs: Some(PreparedQwen35DeviceSlotPtrs {
                        expert_ids: expert_ids[slot_range.clone()].to_vec(),
                        expert_slab_indices: plan.expert_slab_indices,
                        gate_base: plan.gate_base,
                        up_base: plan.up_base,
                        down_base: plan.down_base,
                        gate_expert_bytes: batch.offsets.gate_bytes_per_expert,
                        up_expert_bytes: batch.offsets.up_bytes_per_expert,
                        down_expert_bytes: batch.offsets.down_bytes_per_expert,
                        selected_upload_calls: 1,
                        selected_upload_bytes: plan.slab_bytes,
                        mixed_expert_ptrs: None,
                        group_meta2,
                        group_meta4,
                        group_meta8,
                        group_meta16,
                        group_meta32,
                        group_meta64,
                    }),
                    group_meta: None,
                    device_route: None,
                };
                self.qwen35_sparse_experts_by_token_to_dev_prepared(
                    &[],
                    &[],
                    &[],
                    &route_weights[slot_range.clone()],
                    &token_ids[slot_range.clone()],
                    token_count,
                    down_quant,
                    n_ff,
                    n_embd,
                    input_dev,
                    output_dev,
                    false,
                    false,
                    Some(&expert_ids[slot_range.clone()]),
                    Some(prepared),
                )?;
                unsafe {
                    self.api
                        .event_record(events[STREAM_BUFFER_COUNT + buffer_index], self.stream)?;
                }
                if trace {
                    eprintln!(
                        "[cuda-qwen-selected-base-stream] batch={}/{} slots={} unique={} bytes={}",
                        batch_index + 1,
                        batches.len(),
                        slot_range.len(),
                        plan.uploads.len() / 3,
                        plan.slab_bytes,
                    );
                }
            }
            Ok(())
        })();

        if result.is_err() {
            let _ = unsafe { self.api.stream_synchronize(self.stream) };
            let _ = unsafe { self.api.stream_synchronize(self.copy_stream) };
        }
        let mut cleanup_error = None;
        for event in events {
            if let Err(error) = unsafe { self.api.event_destroy(event) } {
                cleanup_error.get_or_insert(error);
            }
        }
        match (result, cleanup_error) {
            (Ok(()), None) => Ok(true),
            (Ok(()), Some(error)) | (Err(error), _) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_splits_only_at_expert_boundaries() {
        let ranges = qwen35_stream_batch_ranges(&[0, 0, 1, 1, 1, 2, 3, 3], 2)
            .unwrap()
            .unwrap();
        assert_eq!(
            ranges,
            vec![
                Qwen35StreamBatchRange {
                    slot_start: 0,
                    slot_end: 5,
                },
                Qwen35StreamBatchRange {
                    slot_start: 5,
                    slot_end: 8,
                },
            ]
        );
        assert_eq!(
            qwen35_stream_batch_ranges(&[0, 0, 1, 1, 1, 2, 3, 3], 4).unwrap(),
            Some(vec![
                Qwen35StreamBatchRange {
                    slot_start: 0,
                    slot_end: 2,
                },
                Qwen35StreamBatchRange {
                    slot_start: 2,
                    slot_end: 5,
                },
                Qwen35StreamBatchRange {
                    slot_start: 5,
                    slot_end: 6,
                },
                Qwen35StreamBatchRange {
                    slot_start: 6,
                    slot_end: 8,
                },
            ])
        );
    }

    #[test]
    fn stream_rejects_unsorted_experts_and_too_few_experts() {
        assert!(qwen35_stream_batch_ranges(&[1, 0], 2).is_err());
        assert_eq!(qwen35_stream_batch_ranges(&[7, 7], 2).unwrap(), None);
        assert_eq!(qwen35_stream_batch_ranges(&[0, 1, 2], 4).unwrap(), None);
    }
}
