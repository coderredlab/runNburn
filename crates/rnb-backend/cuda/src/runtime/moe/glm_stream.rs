use super::super::*;
use super::glm::glm_slot_identity;
use super::q4k_slots::{plan_direct_file_weights, read_direct_file_plan, DirectFileSlabPlan};
use rnb_core::tensor::FileBackedRegion;

const STREAM_BUFFER_COUNT: usize = 2;
const CUDA_EVENT_DISABLE_TIMING: u32 = 2;

type GlmSlotIdentity = (usize, usize, usize, usize, usize, usize);

pub(super) struct GlmDirectFileStreamRequest<'a> {
    pub(super) gate_weights: &'a [&'a [u8]],
    pub(super) up_weights: &'a [&'a [u8]],
    pub(super) down_weights: &'a [&'a [u8]],
    pub(super) route_weights: &'a [f32],
    pub(super) token_ids: &'a [u32],
    pub(super) group_meta: &'a [u32],
    pub(super) file_regions: &'a [FileBackedRegion; 3],
    pub(super) grouped_gate_kernel: &'static str,
    pub(super) grouped_down_kernel: &'static str,
    pub(super) token_count: usize,
    pub(super) n_ff: usize,
    pub(super) n_embd: usize,
    pub(super) input_dev: u64,
    pub(super) output_dev: u64,
    pub(super) output_bytes: usize,
    pub(super) gate_dev: u64,
    pub(super) up_dev: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GlmStreamBatchRange {
    slot_start: usize,
    slot_end: usize,
    group_start: usize,
    group_end: usize,
}

#[derive(Clone, Copy)]
struct GlmExpertRange {
    identity: GlmSlotIdentity,
    slot_start: usize,
    slot_end: usize,
    group_start: usize,
    group_end: usize,
    logical_bytes: usize,
}

struct GlmDirectFileBatchPlan {
    range: GlmStreamBatchRange,
    group_meta: Vec<u32>,
    weights: DirectFileSlabPlan,
}

fn glm_stream_batch_ranges(
    gate_weights: &[&[u8]],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    group_meta: &[u32],
) -> Result<Vec<GlmStreamBatchRange>, String> {
    if group_meta.is_empty() || group_meta.len() % 2 != 0 {
        return Err("GLM direct-file stream requires non-empty paired group metadata".to_string());
    }
    let mut experts = Vec::<GlmExpertRange>::new();
    for (group_index, group) in group_meta.chunks_exact(2).enumerate() {
        let slot_start = group[0] as usize;
        let slot_end = slot_start
            .checked_add(group[1] as usize)
            .ok_or_else(|| "GLM direct-file stream group range overflow".to_string())?;
        if slot_start >= slot_end || slot_end > gate_weights.len() {
            return Err(format!(
                "GLM direct-file stream group range is invalid: start={slot_start} end={slot_end} slots={}",
                gate_weights.len()
            ));
        }
        let identity = glm_slot_identity(
            gate_weights[slot_start],
            up_weights[slot_start],
            down_weights[slot_start],
        );
        if let Some(expert) = experts
            .last_mut()
            .filter(|expert| expert.identity == identity)
        {
            if expert.slot_end != slot_start || expert.group_end != group_index {
                return Err("GLM direct-file stream expert groups are not contiguous".to_string());
            }
            expert.slot_end = slot_end;
            expert.group_end = group_index + 1;
            continue;
        }
        let logical_bytes = gate_weights[slot_start]
            .len()
            .checked_add(up_weights[slot_start].len())
            .and_then(|bytes| bytes.checked_add(down_weights[slot_start].len()))
            .ok_or_else(|| "GLM direct-file stream expert byte size overflow".to_string())?;
        experts.push(GlmExpertRange {
            identity,
            slot_start,
            slot_end,
            group_start: group_index,
            group_end: group_index + 1,
            logical_bytes,
        });
    }
    if experts.len() < STREAM_BUFFER_COUNT {
        return Err("GLM direct-file stream requires at least two selected experts".to_string());
    }

    let mut remaining_bytes = experts.iter().try_fold(0usize, |total, expert| {
        total
            .checked_add(expert.logical_bytes)
            .ok_or_else(|| "GLM direct-file stream total byte size overflow".to_string())
    })?;
    let mut batches = Vec::with_capacity(STREAM_BUFFER_COUNT);
    let mut expert_start = 0usize;
    while expert_start < experts.len() {
        let remaining_batches = STREAM_BUFFER_COUNT - batches.len();
        let target_bytes = remaining_bytes.div_ceil(remaining_batches);
        let latest_end = experts.len() - (remaining_batches - 1);
        let mut expert_end = expert_start;
        let mut batch_bytes = 0usize;
        while expert_end < latest_end {
            batch_bytes = batch_bytes
                .checked_add(experts[expert_end].logical_bytes)
                .ok_or_else(|| "GLM direct-file stream batch byte size overflow".to_string())?;
            expert_end += 1;
            if batch_bytes >= target_bytes {
                break;
            }
        }
        let first = experts[expert_start];
        let last = experts[expert_end - 1];
        batches.push(GlmStreamBatchRange {
            slot_start: first.slot_start,
            slot_end: last.slot_end,
            group_start: first.group_start,
            group_end: last.group_end,
        });
        remaining_bytes -= batch_bytes;
        expert_start = expert_end;
    }
    Ok(batches)
}

fn build_batch_meta(
    gate_ptrs: &[u64],
    up_ptrs: &[u64],
    down_ptrs: &[u64],
    route_weights: &[f32],
    token_ids: &[u32],
    group_meta: &[u32],
) -> Vec<u8> {
    let ptr_bytes = std::mem::size_of_val(gate_ptrs);
    let route_bytes = std::mem::size_of_val(route_weights);
    let token_bytes = std::mem::size_of_val(token_ids);
    let group_bytes = std::mem::size_of_val(group_meta);
    let mut meta = vec![0u8; ptr_bytes * 3 + route_bytes + token_bytes + group_bytes];
    unsafe {
        std::ptr::copy_nonoverlapping(
            gate_ptrs.as_ptr().cast::<u8>(),
            meta.as_mut_ptr(),
            ptr_bytes,
        );
        std::ptr::copy_nonoverlapping(
            up_ptrs.as_ptr().cast::<u8>(),
            meta.as_mut_ptr().add(ptr_bytes),
            ptr_bytes,
        );
        std::ptr::copy_nonoverlapping(
            down_ptrs.as_ptr().cast::<u8>(),
            meta.as_mut_ptr().add(ptr_bytes * 2),
            ptr_bytes,
        );
        std::ptr::copy_nonoverlapping(
            route_weights.as_ptr().cast::<u8>(),
            meta.as_mut_ptr().add(ptr_bytes * 3),
            route_bytes,
        );
        std::ptr::copy_nonoverlapping(
            token_ids.as_ptr().cast::<u8>(),
            meta.as_mut_ptr().add(ptr_bytes * 3 + route_bytes),
            token_bytes,
        );
        std::ptr::copy_nonoverlapping(
            group_meta.as_ptr().cast::<u8>(),
            meta.as_mut_ptr()
                .add(ptr_bytes * 3 + route_bytes + token_bytes),
            group_bytes,
        );
    }
    meta
}

impl CudaState {
    pub(super) fn glm_sparse_experts_iq_by_token_direct_file_stream(
        &mut self,
        request: GlmDirectFileStreamRequest<'_>,
    ) -> Result<Vec<f32>, String> {
        let ranges = glm_stream_batch_ranges(
            request.gate_weights,
            request.up_weights,
            request.down_weights,
            request.group_meta,
        )?;
        let mut batches = Vec::with_capacity(ranges.len());
        for range in ranges {
            let weights = plan_direct_file_weights(
                request.gate_weights[range.slot_start..range.slot_end]
                    .iter()
                    .copied()
                    .chain(
                        request.up_weights[range.slot_start..range.slot_end]
                            .iter()
                            .copied(),
                    )
                    .chain(
                        request.down_weights[range.slot_start..range.slot_end]
                            .iter()
                            .copied(),
                    ),
                request.file_regions,
            )?;
            let group_meta = request.group_meta[range.group_start * 2..range.group_end * 2]
                .chunks_exact(2)
                .flat_map(|group| [group[0] - range.slot_start as u32, group[1]])
                .collect::<Vec<_>>();
            batches.push(GlmDirectFileBatchPlan {
                range,
                group_meta,
                weights,
            });
        }

        let buffer_stride = batches
            .iter()
            .map(|batch| batch.weights.staging_bytes)
            .max()
            .unwrap_or(0);
        let slab_bytes = buffer_stride
            .checked_mul(STREAM_BUFFER_COUNT)
            .ok_or_else(|| "GLM direct-file stream double-buffer size overflow".to_string())?;
        let max_meta_bytes = batches
            .iter()
            .map(|batch| {
                let slots = batch.range.slot_end - batch.range.slot_start;
                slots * std::mem::size_of::<u64>() * 3
                    + slots * std::mem::size_of::<f32>()
                    + slots * std::mem::size_of::<u32>()
                    + batch.group_meta.len() * std::mem::size_of::<u32>()
            })
            .max()
            .unwrap_or(0);
        let slab_dev = self.compute_temp_slab_ptr(slab_bytes)?;
        let host_slab_ptr = self.host_temp_slab_ptr(slab_bytes)?;
        let meta_dev = self.compute_gate_ptrs_ptr(max_meta_bytes)?;

        let mut events = Vec::with_capacity(STREAM_BUFFER_COUNT * 2);
        for _ in 0..STREAM_BUFFER_COUNT * 2 {
            match unsafe { self.api.event_create(CUDA_EVENT_DISABLE_TIMING) } {
                Ok(event) => events.push(event),
                Err(error) => {
                    for event in events {
                        let _ = unsafe { self.api.event_destroy(event) };
                    }
                    let _ = self.release_compute_temp_slab();
                    return Err(error);
                }
            }
        }

        let result = (|| -> Result<Vec<f32>, String> {
            unsafe {
                self.api.memset_d32_async(
                    request.output_dev,
                    0,
                    request.token_count * request.n_embd,
                    self.stream,
                )?;
            }
            let host_slab = unsafe { std::slice::from_raw_parts_mut(host_slab_ptr, slab_bytes) };
            let mut buffer_in_use = [false; STREAM_BUFFER_COUNT];
            let mut retained_meta = Vec::<Vec<u8>>::with_capacity(batches.len());
            let trace = std::env::var("RNB_CUDA_TEMP_SLAB_TRACE").ok().as_deref() == Some("1");

            for (batch_index, batch) in batches.iter().enumerate() {
                let buffer_index = batch_index % STREAM_BUFFER_COUNT;
                let upload_event = events[buffer_index];
                let compute_event = events[STREAM_BUFFER_COUNT + buffer_index];
                if buffer_in_use[buffer_index] {
                    unsafe { self.api.event_synchronize(compute_event)? };
                }

                let host_base = buffer_index * buffer_stride;
                let device_base = slab_dev + host_base as u64;
                let read_start = std::time::Instant::now();
                read_direct_file_plan(
                    &mut self.direct_file_reader,
                    &batch.weights,
                    request.file_regions,
                    host_slab,
                    host_base,
                )?;
                let read_ms = read_start.elapsed().as_secs_f64() * 1000.0;
                unsafe {
                    self.api.memcpy_htod_async(
                        device_base,
                        host_slab_ptr.add(host_base).cast::<libc::c_void>(),
                        batch.weights.staging_bytes,
                        self.copy_stream,
                    )?;
                    self.api.event_record(upload_event, self.copy_stream)?;
                    self.api.stream_wait_event(self.stream, upload_event)?;
                }

                let ptrs_by_key = batch
                    .weights
                    .offsets_by_key
                    .iter()
                    .map(|(&key, &offset)| (key, device_base + offset as u64))
                    .collect::<HashMap<_, _>>();
                let make_ptrs = |weights: &[&[u8]]| {
                    weights
                        .iter()
                        .map(|weights| ptrs_by_key[&q4k_resident_key(weights)])
                        .collect::<Vec<_>>()
                };
                let slot_range = batch.range.slot_start..batch.range.slot_end;
                let gate_ptrs = make_ptrs(&request.gate_weights[slot_range.clone()]);
                let up_ptrs = make_ptrs(&request.up_weights[slot_range.clone()]);
                let down_ptrs = make_ptrs(&request.down_weights[slot_range.clone()]);
                retained_meta.push(build_batch_meta(
                    &gate_ptrs,
                    &up_ptrs,
                    &down_ptrs,
                    &request.route_weights[slot_range.clone()],
                    &request.token_ids[slot_range.clone()],
                    &batch.group_meta,
                ));
                let meta = retained_meta.last().expect("GLM stream metadata retained");
                unsafe {
                    self.api.memcpy_htod_async(
                        meta_dev,
                        meta.as_ptr().cast::<libc::c_void>(),
                        meta.len(),
                        self.stream,
                    )?;
                }

                let slots = slot_range.len();
                let ptr_bytes = slots * std::mem::size_of::<u64>();
                let route_bytes = slots * std::mem::size_of::<f32>();
                let token_bytes = slots * std::mem::size_of::<u32>();
                let gate_ptrs_dev = meta_dev;
                let up_ptrs_dev = meta_dev + ptr_bytes as u64;
                let down_ptrs_dev = meta_dev + (ptr_bytes * 2) as u64;
                let route_dev = meta_dev + (ptr_bytes * 3) as u64;
                let token_ids_dev = route_dev + route_bytes as u64;
                let group_meta_dev = token_ids_dev + token_bytes as u64;

                self.launch_selected_glm_iq_gate_up_gemv_by_token_group4(
                    request.grouped_gate_kernel,
                    gate_ptrs_dev,
                    up_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    request.n_ff,
                    batch.group_meta.len() / 2,
                    request.n_embd / 256,
                    request.input_dev,
                    request.gate_dev,
                    request.up_dev,
                )?;
                self.launch_silu_mul(request.gate_dev, request.up_dev, slots * request.n_ff)?;
                self.launch_selected_glm_iq_down_accum_by_token_group4(
                    request.grouped_down_kernel,
                    down_ptrs_dev,
                    token_ids_dev,
                    group_meta_dev,
                    request.n_embd,
                    batch.group_meta.len() / 2,
                    request.n_ff / 256,
                    request.gate_dev,
                    route_dev,
                    request.output_dev,
                )?;
                unsafe { self.api.event_record(compute_event, self.stream)? };
                buffer_in_use[buffer_index] = true;

                if trace {
                    eprintln!(
                        "[cuda-direct-file-expert-stream] batch={}/{} slots={} groups={} unique={} logical_bytes={} staging_bytes={} read_runs={} read_ms={read_ms:.3}",
                        batch_index + 1,
                        batches.len(),
                        slots,
                        batch.group_meta.len() / 2,
                        batch.weights.offsets_by_key.len(),
                        batch.weights.logical_bytes,
                        batch.weights.staging_bytes,
                        batch.weights.runs.len(),
                    );
                }
            }

            let mut output = vec![0.0f32; request.token_count * request.n_embd];
            unsafe {
                self.api.memcpy_dtoh_async(
                    output.as_mut_ptr().cast::<libc::c_void>(),
                    request.output_dev,
                    request.output_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            Ok(output)
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
        if let Err(error) = self.release_compute_temp_slab() {
            cleanup_error.get_or_insert(error);
        }
        match (result, cleanup_error) {
            (Ok(output), None) => Ok(output),
            (Ok(_), Some(error)) | (Err(error), _) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_batches_do_not_split_repeated_expert_groups() {
        let gate = [[1u8], [2u8], [3u8], [4u8]];
        let up = [[5u8], [6u8], [7u8], [8u8]];
        let down = [[9u8], [10u8], [11u8], [12u8]];
        let mut gate_slots = Vec::new();
        let mut up_slots = Vec::new();
        let mut down_slots = Vec::new();
        let mut group_meta = Vec::new();
        for expert in 0..4 {
            let start = gate_slots.len();
            for _ in 0..5 {
                gate_slots.push(gate[expert].as_slice());
                up_slots.push(up[expert].as_slice());
                down_slots.push(down[expert].as_slice());
            }
            group_meta.extend_from_slice(&[start as u32, 4, start as u32 + 4, 1]);
        }

        let batches =
            glm_stream_batch_ranges(&gate_slots, &up_slots, &down_slots, &group_meta).unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].slot_start, 0);
        assert_eq!(batches[0].slot_end, 10);
        assert_eq!(batches[1].slot_start, 10);
        assert_eq!(batches[1].slot_end, 20);
        assert_eq!(batches[0].group_end, batches[1].group_start);
    }
}
