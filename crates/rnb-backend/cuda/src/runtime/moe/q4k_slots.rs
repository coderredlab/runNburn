use super::super::*;
use rnb_core::tensor::FileBackedRegion;
use std::sync::atomic::Ordering;

static GLM_IO_URING_FALLBACK_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

const DIRECT_FILE_ALIGNMENT: usize = 4096;

#[derive(Clone, Copy, Debug)]
struct DirectFileEntry {
    key: (usize, usize),
    region_index: usize,
    file_offset: u64,
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DirectFileRun {
    pub(super) region_index: usize,
    pub(super) file_offset: u64,
    pub(super) staging_offset: usize,
    pub(super) staging_len: usize,
    pub(super) required_len: usize,
}

#[derive(Clone)]
pub(super) struct DirectFileSlabPlan {
    pub(super) offsets_by_key: HashMap<(usize, usize), usize>,
    pub(super) runs: Vec<DirectFileRun>,
    pub(super) logical_bytes: usize,
    pub(super) staging_bytes: usize,
}

fn plan_direct_file_slab(mut entries: Vec<DirectFileEntry>) -> Result<DirectFileSlabPlan, String> {
    entries.sort_unstable_by_key(|entry| (entry.region_index, entry.file_offset));
    let logical_bytes = entries.iter().try_fold(0usize, |total, entry| {
        total
            .checked_add(entry.len)
            .ok_or_else(|| "direct file slab logical byte size overflow".to_string())
    })?;
    let mut offsets_by_key = HashMap::with_capacity(entries.len());
    let mut runs = Vec::new();
    let mut staging_bytes = 0usize;
    let mut i = 0usize;
    while i < entries.len() {
        let region_index = entries[i].region_index;
        let logical_start = entries[i].file_offset;
        let mut logical_end = logical_start
            .checked_add(entries[i].len as u64)
            .ok_or_else(|| "direct file slab range overflow".to_string())?;
        let mut j = i + 1;
        while j < entries.len()
            && entries[j].region_index == region_index
            && entries[j].file_offset == logical_end
        {
            logical_end = entries[j]
                .file_offset
                .checked_add(entries[j].len as u64)
                .ok_or_else(|| "direct file slab range overflow".to_string())?;
            j += 1;
        }
        let aligned_file_offset = logical_start & !(DIRECT_FILE_ALIGNMENT as u64 - 1);
        let head_bytes = usize::try_from(logical_start - aligned_file_offset)
            .map_err(|_| "direct file slab head size overflow".to_string())?;
        let logical_run_bytes = usize::try_from(logical_end - logical_start)
            .map_err(|_| "direct file slab run size overflow".to_string())?;
        let required_len = head_bytes
            .checked_add(logical_run_bytes)
            .ok_or_else(|| "direct file slab required size overflow".to_string())?;
        let staging_len = required_len
            .checked_add(DIRECT_FILE_ALIGNMENT - 1)
            .ok_or_else(|| "direct file slab alignment overflow".to_string())?
            / DIRECT_FILE_ALIGNMENT
            * DIRECT_FILE_ALIGNMENT;
        for entry in &entries[i..j] {
            let relative = usize::try_from(entry.file_offset - logical_start)
                .map_err(|_| "direct file slab entry offset overflow".to_string())?;
            let offset = staging_bytes
                .checked_add(head_bytes)
                .and_then(|offset| offset.checked_add(relative))
                .ok_or_else(|| "direct file slab entry placement overflow".to_string())?;
            offsets_by_key.insert(entry.key, offset);
        }
        runs.push(DirectFileRun {
            region_index,
            file_offset: aligned_file_offset,
            staging_offset: staging_bytes,
            staging_len,
            required_len,
        });
        staging_bytes = staging_bytes
            .checked_add(staging_len)
            .ok_or_else(|| "direct file slab staging size overflow".to_string())?;
        i = j;
    }
    Ok(DirectFileSlabPlan {
        offsets_by_key,
        runs,
        logical_bytes,
        staging_bytes,
    })
}

pub(super) fn plan_direct_file_weights<'a>(
    weights: impl IntoIterator<Item = &'a [u8]>,
    file_regions: &[FileBackedRegion; 3],
) -> Result<DirectFileSlabPlan, String> {
    let mut unique = HashMap::new();
    for weights in weights {
        unique.entry(q4k_resident_key(weights)).or_insert(weights);
    }
    let mut entries = Vec::with_capacity(unique.len());
    for (&key, &weights) in &unique {
        let resolved = file_regions
            .iter()
            .enumerate()
            .find_map(|(region_index, region)| {
                region
                    .resolve_subslice(weights)
                    .map(|(file_offset, len)| (region_index, file_offset, len))
            })
            .ok_or_else(|| {
                format!(
                    "direct file GLM weight is outside its mapped tensor regions: ptr={:#x} len={}",
                    weights.as_ptr() as usize,
                    weights.len()
                )
            })?;
        entries.push(DirectFileEntry {
            key,
            region_index: resolved.0,
            file_offset: resolved.1,
            len: resolved.2,
        });
    }
    plan_direct_file_slab(entries)
}

pub(super) fn read_direct_file_plan(
    reader: &mut rnb_memory::moe_cold_io::DirectFileReaderCache,
    plan: &DirectFileSlabPlan,
    file_regions: &[FileBackedRegion; 3],
    host_slab: &mut [u8],
    staging_base: usize,
) -> Result<(), String> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    if tuning::glm_direct_file_io_uring_enabled() {
        let requests = plan
            .runs
            .iter()
            .map(|run| {
                let destination_offset = staging_base
                    .checked_add(run.staging_offset)
                    .ok_or_else(|| "direct file GLM staging offset overflow".to_string())?;
                Ok(rnb_memory::moe_cold_io::DirectFileReadRequest {
                    path: file_regions[run.region_index].path(),
                    file_offset: run.file_offset,
                    destination_offset,
                    read_len: run.staging_len,
                    required_len: run.required_len,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let queue_depth = tuning::glm_direct_file_io_uring_queue_depth(requests.len());
        if let Err(error) = reader.ensure_io_uring(queue_depth) {
            if tuning::glm_direct_file_io_uring_forced() {
                return Err(format!(
                    "initializing io_uring direct file reader failed: {error}"
                ));
            }
            GLM_IO_URING_FALLBACK_LOGGED.get_or_init(|| {
                eprintln!("[INFO] io_uring unavailable; using positioned O_DIRECT reads: {error}");
            });
        } else {
            reader
                .read_aligned_batch(&requests, host_slab, queue_depth)
                .map_err(|error| format!("io_uring direct file GLM batch read failed: {error}"))?;
            return Ok(());
        }
    }
    for run in &plan.runs {
        let region = &file_regions[run.region_index];
        let start = staging_base
            .checked_add(run.staging_offset)
            .ok_or_else(|| "direct file GLM staging offset overflow".to_string())?;
        let end = start
            .checked_add(run.staging_len)
            .ok_or_else(|| "direct file GLM staging length overflow".to_string())?;
        let destination = &mut host_slab[start..end];
        reader
            .read_aligned(
                region.path(),
                run.file_offset,
                destination,
                run.required_len,
            )
            .map_err(|error| {
                format!(
                    "direct file GLM read failed for {} at {}: {error}",
                    region.path().display(),
                    run.file_offset
                )
            })?;
    }
    Ok(())
}

impl CudaState {
    pub(in crate::runtime) fn temp_q4k_slot_ptrs_3(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        self.temp_q4k_slot_ptrs_3_with_upload_stream(
            gate_weights,
            up_weights,
            down_weights,
            None,
            None,
        )
    }

    pub(in crate::runtime) fn temp_q4k_slot_ptrs_3_direct_file(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        file_regions: &[FileBackedRegion; 3],
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        if tuning::glm_direct_file_pipeline_enabled() {
            return self.temp_q4k_slot_ptrs_3_direct_file_pipelined(
                gate_weights,
                up_weights,
                down_weights,
                file_regions,
            );
        }
        let plan = plan_direct_file_weights(
            gate_weights
                .iter()
                .copied()
                .chain(up_weights.iter().copied())
                .chain(down_weights.iter().copied()),
            file_regions,
        )?;
        if plan.offsets_by_key.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        let slab_dev = self.compute_temp_slab_ptr(plan.staging_bytes)?;
        let host_slab_ptr = self.host_temp_slab_ptr(plan.staging_bytes)?;
        let host_slab =
            unsafe { std::slice::from_raw_parts_mut(host_slab_ptr, plan.staging_bytes) };
        let read_start = std::time::Instant::now();
        read_direct_file_plan(
            &mut self.direct_file_reader,
            &plan,
            file_regions,
            host_slab,
            0,
        )?;
        let read_ms = read_start.elapsed().as_secs_f64() * 1000.0;
        unsafe {
            self.api.memcpy_htod_async(
                slab_dev,
                host_slab_ptr.cast::<libc::c_void>(),
                plan.staging_bytes,
                self.stream,
            )?;
        }
        let ptrs_by_key = plan
            .offsets_by_key
            .iter()
            .map(|(&key, &offset)| (key, slab_dev + offset as u64))
            .collect::<HashMap<_, _>>();
        if std::env::var("RNB_CUDA_TEMP_SLAB_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[cuda-direct-file-slab] unique={} logical_bytes={} staging_bytes={} read_runs={} read_ms={read_ms:.3} h2d_runs=1 h2d_bytes={}",
                ptrs_by_key.len(),
                plan.logical_bytes,
                plan.staging_bytes,
                plan.runs.len(),
                plan.staging_bytes
            );
        }
        let make_ptrs = |slot_weights: &[&[u8]]| {
            slot_weights
                .iter()
                .map(|weights| ptrs_by_key[&q4k_resident_key(weights)])
                .collect::<Vec<_>>()
        };
        Ok((
            make_ptrs(gate_weights),
            make_ptrs(up_weights),
            make_ptrs(down_weights),
            ptrs_by_key.into_values().collect(),
        ))
    }

    fn temp_q4k_slot_ptrs_3_direct_file_pipelined(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        file_regions: &[FileBackedRegion; 3],
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        let gate_up_plan = plan_direct_file_weights(
            gate_weights
                .iter()
                .copied()
                .chain(up_weights.iter().copied()),
            file_regions,
        )?;
        let down_plan = plan_direct_file_weights(down_weights.iter().copied(), file_regions)?;
        let staging_bytes = gate_up_plan
            .staging_bytes
            .checked_add(down_plan.staging_bytes)
            .ok_or_else(|| "direct file GLM pipeline staging size overflow".to_string())?;
        if staging_bytes == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }

        let slab_dev = self.compute_temp_slab_ptr(staging_bytes)?;
        let host_slab_ptr = self.host_temp_slab_ptr(staging_bytes)?;
        let host_slab = unsafe { std::slice::from_raw_parts_mut(host_slab_ptr, staging_bytes) };

        let gate_up_read_start = std::time::Instant::now();
        read_direct_file_plan(
            &mut self.direct_file_reader,
            &gate_up_plan,
            file_regions,
            host_slab,
            0,
        )?;
        let gate_up_read_ms = gate_up_read_start.elapsed().as_secs_f64() * 1000.0;
        unsafe {
            self.api.memcpy_htod_async(
                slab_dev,
                host_slab_ptr.cast::<libc::c_void>(),
                gate_up_plan.staging_bytes,
                self.stream,
            )?;
        }

        let down_base = gate_up_plan.staging_bytes;
        let down_read_start = std::time::Instant::now();
        read_direct_file_plan(
            &mut self.direct_file_reader,
            &down_plan,
            file_regions,
            host_slab,
            down_base,
        )?;
        let down_read_ms = down_read_start.elapsed().as_secs_f64() * 1000.0;
        unsafe {
            self.api.memcpy_htod_async(
                slab_dev + down_base as u64,
                host_slab_ptr.add(down_base).cast::<libc::c_void>(),
                down_plan.staging_bytes,
                self.copy_stream,
            )?;
        }

        let gate_up_ptrs = gate_up_plan
            .offsets_by_key
            .iter()
            .map(|(&key, &offset)| (key, slab_dev + offset as u64))
            .collect::<HashMap<_, _>>();
        let down_ptrs = down_plan
            .offsets_by_key
            .iter()
            .map(|(&key, &offset)| (key, slab_dev + down_base as u64 + offset as u64))
            .collect::<HashMap<_, _>>();
        if std::env::var("RNB_CUDA_TEMP_SLAB_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[cuda-direct-file-pipeline] io_uring={} gate_up_unique={} down_unique={} logical_bytes={} staging_bytes={} gate_up_runs={} down_runs={} gate_up_read_ms={gate_up_read_ms:.3} down_read_ms={down_read_ms:.3} h2d_runs=2 h2d_bytes={staging_bytes}",
                tuning::glm_direct_file_io_uring_enabled(),
                gate_up_ptrs.len(),
                down_ptrs.len(),
                gate_up_plan.logical_bytes + down_plan.logical_bytes,
                staging_bytes,
                gate_up_plan.runs.len(),
                down_plan.runs.len()
            );
        }
        let make_ptrs = |slot_weights: &[&[u8]], ptrs: &HashMap<(usize, usize), u64>| {
            slot_weights
                .iter()
                .map(|weights| ptrs[&q4k_resident_key(weights)])
                .collect::<Vec<_>>()
        };
        let tracked_ptrs = gate_up_ptrs
            .values()
            .chain(down_ptrs.values())
            .copied()
            .collect();
        Ok((
            make_ptrs(gate_weights, &gate_up_ptrs),
            make_ptrs(up_weights, &gate_up_ptrs),
            make_ptrs(down_weights, &down_ptrs),
            tracked_ptrs,
        ))
    }

    pub(in crate::runtime) fn temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        tracked_keys: &HashSet<(usize, usize)>,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        self.temp_q4k_slot_ptrs_3_with_upload_stream(
            gate_weights,
            up_weights,
            down_weights,
            None,
            Some(tracked_keys),
        )
    }

    pub(in crate::runtime) fn temp_q4k_slot_ptrs_3_copy_stream(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        self.temp_q4k_slot_ptrs_3_with_upload_stream(
            gate_weights,
            up_weights,
            down_weights,
            Some(self.copy_stream),
            None,
        )
    }

    pub(in crate::runtime) fn clear_pending_nemotron_prefill_sparse(
        &mut self,
    ) -> Result<(), String> {
        if let Some(prefetch) = self.pending_nemotron_prefill_sparse.take() {
            unsafe {
                self.api.stream_synchronize(self.copy_stream)?;
                self.api.mem_free(prefetch.slab)?;
            }
        }
        Ok(())
    }

    pub(in crate::runtime) fn prefetch_nemotron_prefill_sparse_q4k(
        &mut self,
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Result<bool, String> {
        if !tuning::nemotron_prefill_sparse_copy_prefetch_enabled() {
            return Ok(false);
        }
        if up_weights.is_empty() || up_weights.len() != down_weights.len() {
            return Ok(false);
        }
        self.clear_pending_nemotron_prefill_sparse()?;
        self.set_current()?;

        let mut unique = HashMap::new();
        let mut order = Vec::new();
        let mut total_bytes = 0usize;
        for &weights in up_weights.iter().chain(down_weights.iter()) {
            let key = q4k_resident_key(weights);
            if let std::collections::hash_map::Entry::Vacant(entry) = unique.entry(key) {
                let offset = total_bytes;
                total_bytes = total_bytes.saturating_add(weights.len());
                entry.insert((offset, weights));
                order.push(key);
            }
        }
        if total_bytes == 0 {
            return Ok(false);
        }

        let slab = match unsafe { self.api.mem_alloc(total_bytes) } {
            Ok(ptr) => ptr,
            Err(err) if cuda_offload_on_oom_enabled() && cuda_mem_alloc_oom(&err) => {
                let _ = self.offload_non_pinned_resident_q4k()?;
                return Ok(false);
            }
            Err(err) => return Err(err),
        };

        for key in order {
            let (offset, weights) = unique[&key];
            unsafe {
                self.api.memcpy_htod_async(
                    slab + offset as u64,
                    weights.as_ptr().cast::<libc::c_void>(),
                    weights.len(),
                    self.copy_stream,
                )?;
            }
        }

        let make_keys = |slot_weights: &[&[u8]]| {
            slot_weights
                .iter()
                .map(|weights| q4k_resident_key(weights))
                .collect::<Vec<_>>()
        };
        let make_ptrs = |slot_weights: &[&[u8]]| {
            slot_weights
                .iter()
                .map(|weights| {
                    let key = q4k_resident_key(weights);
                    let (offset, _) = unique[&key];
                    slab + offset as u64
                })
                .collect::<Vec<_>>()
        };
        self.pending_nemotron_prefill_sparse = Some(PendingNemotronPrefillSparse {
            slab,
            up_keys: make_keys(up_weights),
            down_keys: make_keys(down_weights),
            up_ptrs: make_ptrs(up_weights),
            down_ptrs: make_ptrs(down_weights),
        });
        if std::env::var("RNB_CUDA_NEMOTRON_PREFILL_COPY_PREFETCH_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[cuda:nemotron-prefill-copy-prefetch] slots={} unique={} bytes={} mb={:.2}",
                up_weights.len(),
                unique.len(),
                total_bytes,
                total_bytes as f64 / (1024.0 * 1024.0)
            );
        }
        Ok(true)
    }

    pub(in crate::runtime) fn pending_nemotron_prefill_sparse_ptrs(
        &mut self,
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Result<Option<(Vec<u64>, Vec<u64>)>, String> {
        let Some(prefetch) = self.pending_nemotron_prefill_sparse.as_ref() else {
            return Ok(None);
        };
        if prefetch.up_keys.len() != up_weights.len()
            || prefetch.down_keys.len() != down_weights.len()
            || prefetch
                .up_keys
                .iter()
                .zip(up_weights.iter())
                .any(|(key, weights)| *key != q4k_resident_key(weights))
            || prefetch
                .down_keys
                .iter()
                .zip(down_weights.iter())
                .any(|(key, weights)| *key != q4k_resident_key(weights))
        {
            self.clear_pending_nemotron_prefill_sparse()?;
            return Ok(None);
        }
        unsafe {
            self.api.stream_synchronize(self.copy_stream)?;
        }
        Ok(Some((prefetch.up_ptrs.clone(), prefetch.down_ptrs.clone())))
    }

    fn temp_q4k_slot_ptrs_3_with_upload_stream(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        upload_stream: Option<usize>,
        tracked_expert_bundle_keys: Option<&HashSet<(usize, usize)>>,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        let mut unique = HashMap::new();
        let mut order = Vec::new();
        let mut slot_counts = HashMap::new();
        let mut total_bytes = 0usize;
        let overlap_down_copy = tuning::prefill_down_copy_overlap_enabled();
        for &weights in gate_weights
            .iter()
            .chain(up_weights.iter())
            .chain(down_weights.iter())
        {
            let key = q4k_resident_key(weights);
            *slot_counts.entry(key).or_insert(0u32) += 1;
            if let std::collections::hash_map::Entry::Vacant(entry) = unique.entry(key) {
                total_bytes = total_bytes.saturating_add(weights.len());
                entry.insert(weights);
                order.push(key);
            }
        }
        if total_bytes == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        let mut unique_entries = order
            .into_iter()
            .map(|key| (key, unique[&key]))
            .collect::<Vec<_>>();
        let mut gate_up_keys = std::collections::HashSet::new();
        if overlap_down_copy {
            for &weights in gate_weights.iter().chain(up_weights.iter()) {
                gate_up_keys.insert(q4k_resident_key(weights));
            }
        }
        let large_selected_moe =
            gate_weights.is_empty() && up_weights.len().max(down_weights.len()) >= 128;
        let coalesce = tuning::prefill_temp_coalesce_enabled()
            || (large_selected_moe
                && std::env::var("RNB_CUDA_PREFILL_TEMP_COALESCE")
                    .ok()
                    .as_deref()
                    != Some("0"));
        if coalesce {
            unique_entries.sort_unstable_by_key(|(_, weights)| weights.as_ptr() as usize);
        }
        let slab_dev = self.compute_temp_slab_ptr(total_bytes)?;
        let mut ptrs_by_key = HashMap::with_capacity(unique.len());
        let mut offset = 0usize;
        let mut entries = Vec::with_capacity(unique_entries.len());
        for (key, weights) in unique_entries {
            let stream = upload_stream.unwrap_or_else(|| {
                if overlap_down_copy && !gate_up_keys.contains(&key) {
                    self.copy_stream
                } else {
                    self.stream
                }
            });
            ptrs_by_key.insert(key, slab_dev + offset as u64);
            entries.push((key, offset, weights, stream));
            offset = offset.saturating_add(weights.len());
        }
        let trace_temp = std::env::var("RNB_CUDA_TEMP_SLAB_TRACE").ok().as_deref() == Some("1");
        let trace_plan = trace_temp
            || std::env::var("RNB_CUDA_RESIDENCY_PLAN_TRACE")
                .ok()
                .as_deref()
                == Some("1");
        let plan = if trace_plan {
            let candidates = q4k_residency_candidates(&unique, &slot_counts);
            Some(
                ResidencyPlanner::new(
                    prefill_residency_trace_budget_bytes(self.resident_q4k_limit) as u64,
                )
                .plan(&candidates),
            )
        } else {
            None
        };
        let mut copy_runs = 0usize;
        let mut copy_bytes = 0usize;
        let explicit_upload_stream = upload_stream.is_some();
        let pinned_staging = total_bytes > 0
            && ((explicit_upload_stream && tuning::prefill_moe_weight_prefetch_pinned_enabled())
                || (tuning::prefill_temp_pinned_staging_enabled() && !overlap_down_copy));
        let slot_count = gate_weights
            .len()
            .max(up_weights.len())
            .max(down_weights.len());
        let host_register = !pinned_staging
            && tuning::prefill_temp_host_register_enabled()
            && total_bytes > 0
            && slot_count >= tuning::prefill_temp_host_register_min_slots();
        let host_register_min_bytes = tuning::prefill_temp_host_register_min_bytes();
        if pinned_staging {
            let host_slab = self.host_temp_slab_ptr(total_bytes)?;
            for (_, offset, weights, _) in entries.iter() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        weights.as_ptr(),
                        host_slab.add(*offset),
                        weights.len(),
                    );
                }
                copy_bytes = copy_bytes.saturating_add(weights.len());
            }
            unsafe {
                self.api.memcpy_htod_async(
                    slab_dev,
                    host_slab.cast::<libc::c_void>(),
                    total_bytes,
                    self.stream,
                )?;
            }
            copy_runs = 1;
        } else if coalesce || tuning::prefill_temp_run_coalesce_enabled() {
            let mut i = 0usize;
            while i < entries.len() {
                let (_, dst_offset, weights, stream) = entries[i];
                let src_start = weights.as_ptr() as usize;
                let mut bytes = weights.len();
                let mut j = i + 1;
                while j < entries.len() {
                    let (_, next_offset, next_weights, next_stream) = entries[j];
                    if next_stream != stream
                        || next_offset != dst_offset + bytes
                        || next_weights.as_ptr() as usize != src_start + bytes
                    {
                        break;
                    }
                    bytes = bytes.saturating_add(next_weights.len());
                    j += 1;
                }
                if host_register && bytes >= host_register_min_bytes {
                    self.ensure_host_registered(src_start as *const u8, bytes)?;
                }
                unsafe {
                    self.api.memcpy_htod_async(
                        slab_dev + dst_offset as u64,
                        (src_start as *const u8).cast::<libc::c_void>(),
                        bytes,
                        stream,
                    )?;
                }
                copy_runs += 1;
                copy_bytes = copy_bytes.saturating_add(bytes);
                i = j;
            }
        } else {
            for (_, offset, weights, stream) in entries {
                if host_register && weights.len() >= host_register_min_bytes {
                    self.ensure_host_registered(weights.as_ptr(), weights.len())?;
                }
                unsafe {
                    self.api.memcpy_htod_async(
                        slab_dev + offset as u64,
                        weights.as_ptr().cast::<libc::c_void>(),
                        weights.len(),
                        stream,
                    )?;
                }
                copy_runs += 1;
                copy_bytes = copy_bytes.saturating_add(weights.len());
            }
        }
        if let Some(tracked_keys) = tracked_expert_bundle_keys {
            let tracked_bytes = unique.iter().fold(0u64, |bytes, (key, weights)| {
                if tracked_keys.contains(key) {
                    bytes.saturating_add(weights.len() as u64)
                } else {
                    bytes
                }
            });
            cache_stats().record_expert_bundle_h2d(tracked_bytes, true);
        }
        let mut transfer_stats = ResidencyTransferStats::default();
        transfer_stats.record_temp_upload(copy_bytes as u64);
        if let Some(plan) = &plan {
            transfer_stats.record_spill(plan.spill_bytes());
        }
        if trace_temp {
            eprintln!(
                "[cuda-temp-slab] unique={} slab_bytes={} copy_runs={} copy_bytes={} h2d_bytes={} spill_bytes={} pinned_staging={} host_register={} coalesce={} run_coalesce={} down_overlap={}",
                ptrs_by_key.len(),
                total_bytes,
                copy_runs,
                copy_bytes,
                transfer_stats.total_h2d_bytes(),
                transfer_stats.spill_bytes,
                pinned_staging,
                host_register,
                coalesce,
                tuning::prefill_temp_run_coalesce_enabled(),
                overlap_down_copy || upload_stream.is_some()
            );
        }
        if let Some(plan) = &plan {
            eprintln!(
                "[cuda-residency-plan] candidates={} budget_mb={} selected={} selected_mb={:.2} spill_mb={:.2} temp_h2d_mb={:.2}",
                unique.len(),
                prefill_residency_trace_budget_bytes(self.resident_q4k_limit) / (1024 * 1024),
                plan.selected().len(),
                plan.selected_bytes() as f64 / (1024.0 * 1024.0),
                plan.spill_bytes() as f64 / (1024.0 * 1024.0),
                transfer_stats.total_h2d_bytes() as f64 / (1024.0 * 1024.0)
            );
        }
        let make_ptrs = |slot_weights: &[&[u8]], ptrs_by_key: &HashMap<(usize, usize), u64>| {
            slot_weights
                .iter()
                .map(|weights| ptrs_by_key[&q4k_resident_key(weights)])
                .collect::<Vec<_>>()
        };
        let gate_ptrs = make_ptrs(gate_weights, &ptrs_by_key);
        let up_ptrs = make_ptrs(up_weights, &ptrs_by_key);
        let down_ptrs = make_ptrs(down_weights, &ptrs_by_key);
        Ok((
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            ptrs_by_key.into_values().collect(),
        ))
    }

    pub(in crate::runtime) fn q4k_slot_groups_have_resident(
        &self,
        slot_groups: &[&[&[u8]]],
    ) -> bool {
        slot_groups.iter().any(|slot_weights| {
            slot_weights
                .iter()
                .any(|weights| self.resident_q4k.contains_key(&q4k_resident_key(weights)))
        })
    }

    pub(in crate::runtime) fn resident_q4k_slot_ptrs_3_if_all_resident(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Option<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>)> {
        let mut touched = HashSet::new();
        let mut collect = |slot_weights: &[&[u8]],
                           resident_q4k: &HashMap<(usize, usize), ResidentQ4k>|
         -> Option<Vec<u64>> {
            let mut ptrs = Vec::with_capacity(slot_weights.len());
            for &weights in slot_weights {
                let key = q4k_resident_key(weights);
                let ptr = resident_q4k.get(&key)?.ptr;
                touched.insert(key);
                ptrs.push(ptr);
            }
            Some(ptrs)
        };

        let gate_ptrs = collect(gate_weights, &self.resident_q4k)?;
        let up_ptrs = collect(up_weights, &self.resident_q4k)?;
        let down_ptrs = collect(down_weights, &self.resident_q4k)?;
        if qwen35_decode_all_resident_touch_hits_enabled() {
            for key in touched {
                self.touch_resident_q4k(key);
            }
        }
        Some((gate_ptrs, up_ptrs, down_ptrs, Vec::new()))
    }

    pub(in crate::runtime) fn mixed_resident_temp_q4k_slot_ptrs_3(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        self.mixed_resident_temp_q4k_slot_ptrs_3_with_expert_bundle_h2d(
            gate_weights,
            up_weights,
            down_weights,
            None,
        )
    }

    pub(in crate::runtime) fn mixed_resident_temp_q4k_slot_ptrs_3_recording_expert_bundle_h2d(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        tracked_keys: &HashSet<(usize, usize)>,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        self.mixed_resident_temp_q4k_slot_ptrs_3_with_expert_bundle_h2d(
            gate_weights,
            up_weights,
            down_weights,
            Some(tracked_keys),
        )
    }

    fn mixed_resident_temp_q4k_slot_ptrs_3_with_expert_bundle_h2d(
        &mut self,
        gate_weights: &[&[u8]],
        up_weights: &[&[u8]],
        down_weights: &[&[u8]],
        tracked_expert_bundle_keys: Option<&HashSet<(usize, usize)>>,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>), String> {
        let mut unique = HashSet::new();
        let mut ptrs_by_key = HashMap::new();
        let mut temp_weights_by_key = HashMap::new();
        let mut resident_key_set = HashSet::new();
        let mut resident_keys = Vec::new();
        let mut resident_bytes = 0usize;
        let mut resident_count = 0usize;
        for &weights in gate_weights
            .iter()
            .chain(up_weights.iter())
            .chain(down_weights.iter())
        {
            let key = q4k_resident_key(weights);
            if !unique.insert(key) {
                continue;
            }
            if let Some(ptr) = self.resident_q4k.get(&key).map(|entry| entry.ptr) {
                ptrs_by_key.insert(key, ptr);
                resident_key_set.insert(key);
                resident_keys.push(key);
                resident_bytes = resident_bytes.saturating_add(weights.len());
                resident_count += 1;
            } else {
                temp_weights_by_key.insert(key, weights);
            }
        }
        for key in resident_keys {
            self.touch_resident_q4k(key);
        }
        let (temp_plan, temp_bytes) = qwen35_mixed_temp_slot_upload_plan(
            gate_weights,
            up_weights,
            down_weights,
            &resident_key_set,
            tuning::prefill_down_copy_overlap_enabled(),
        );
        let temp_count = temp_plan.len();

        let mut temp_slab_ptrs = Vec::new();
        let mut copy_runs = 0usize;
        let mut copy_stream_runs = 0usize;
        if temp_bytes > 0 {
            let slab_dev = self.compute_temp_slab_ptr(temp_bytes)?;
            for entry in temp_plan {
                let Some(weights) = temp_weights_by_key.get(&entry.key).copied() else {
                    return Err("Qwen35 mixed temp slot plan referenced missing weight".to_string());
                };
                let stream = match entry.stream {
                    Qwen35TempUploadStream::Main => self.stream,
                    Qwen35TempUploadStream::Copy => {
                        copy_stream_runs += 1;
                        self.copy_stream
                    }
                };
                ptrs_by_key.insert(entry.key, slab_dev + entry.offset as u64);
                unsafe {
                    self.api.memcpy_htod_async(
                        slab_dev + entry.offset as u64,
                        weights.as_ptr().cast::<libc::c_void>(),
                        weights.len(),
                        stream,
                    )?;
                }
                copy_runs += 1;
            }
            temp_slab_ptrs.push(slab_dev);
        }
        if let Some(tracked_keys) = tracked_expert_bundle_keys {
            let tracked_bytes = temp_weights_by_key
                .iter()
                .fold(0u64, |bytes, (key, weights)| {
                    if tracked_keys.contains(key) {
                        bytes.saturating_add(weights.len() as u64)
                    } else {
                        bytes
                    }
                });
            cache_stats().record_expert_bundle_h2d(tracked_bytes, true);
        }

        if std::env::var("RNB_CUDA_TEMP_SLAB_TRACE").ok().as_deref() == Some("1") {
            eprintln!(
                "[cuda-temp-slab-mixed] unique={} resident={} resident_mb={:.2} temp={} temp_h2d_bytes={} temp_h2d_mb={:.2} copy_runs={} copy_stream_runs={} down_overlap={}",
                ptrs_by_key.len(),
                resident_count,
                resident_bytes as f64 / (1024.0 * 1024.0),
                temp_count,
                temp_bytes,
                temp_bytes as f64 / (1024.0 * 1024.0),
                copy_runs,
                copy_stream_runs,
                tuning::prefill_down_copy_overlap_enabled()
            );
        }

        let make_ptrs = |slot_weights: &[&[u8]], ptrs_by_key: &HashMap<(usize, usize), u64>| {
            slot_weights
                .iter()
                .map(|weights| ptrs_by_key[&q4k_resident_key(weights)])
                .collect::<Vec<_>>()
        };
        let gate_ptrs = make_ptrs(gate_weights, &ptrs_by_key);
        let up_ptrs = make_ptrs(up_weights, &ptrs_by_key);
        let down_ptrs = make_ptrs(down_weights, &ptrs_by_key);
        Ok((gate_ptrs, up_ptrs, down_ptrs, temp_slab_ptrs))
    }

    pub(in crate::runtime) fn resident_q4k_slot_ptrs(
        &mut self,
        slot_weights: &[&[u8]],
        local_ptrs: &mut HashMap<(usize, usize), u64>,
    ) -> Result<Vec<u64>, String> {
        self.resident_q4k_slot_ptrs_with_touch(slot_weights, local_ptrs, false)
    }

    pub(in crate::runtime) fn resident_q4k_slot_ptrs_touch_hits(
        &mut self,
        slot_weights: &[&[u8]],
        local_ptrs: &mut HashMap<(usize, usize), u64>,
    ) -> Result<Vec<u64>, String> {
        self.resident_q4k_slot_ptrs_with_touch(slot_weights, local_ptrs, true)
    }

    fn resident_q4k_slot_ptrs_with_touch(
        &mut self,
        slot_weights: &[&[u8]],
        local_ptrs: &mut HashMap<(usize, usize), u64>,
        touch_hits: bool,
    ) -> Result<Vec<u64>, String> {
        if std::env::var("RNB_CUDA_RESIDENT_Q4K_BATCH_MISS")
            .ok()
            .as_deref()
            != Some("0")
        {
            self.batch_resident_q4k_slot_misses(slot_weights, local_ptrs)?;
        }
        let mut ptrs = Vec::with_capacity(slot_weights.len());
        for &weights in slot_weights {
            let key = q4k_resident_key(weights);
            let ptr = if let Some(&ptr) = local_ptrs.get(&key) {
                ptr
            } else {
                let ptr = self.resident_q4k_weights_ptr(weights)?;
                local_ptrs.insert(key, ptr);
                ptr
            };
            if touch_hits {
                self.touch_resident_q4k(key);
            }
            ptrs.push(ptr);
        }
        Ok(ptrs)
    }

    pub(in crate::runtime) fn batch_resident_q4k_slot_misses(
        &mut self,
        slot_weights: &[&[u8]],
        local_ptrs: &HashMap<(usize, usize), u64>,
    ) -> Result<(), String> {
        self.batch_resident_q4k_slot_misses_many(&[slot_weights], local_ptrs)
    }

    pub(in crate::runtime) fn batch_resident_q4k_slot_misses_many(
        &mut self,
        slot_groups: &[&[&[u8]]],
        local_ptrs: &HashMap<(usize, usize), u64>,
    ) -> Result<(), String> {
        let stream = self.stream;
        self.batch_resident_q4k_slot_misses_many_on_stream(slot_groups, local_ptrs, stream)
            .map(|_| ())
    }

    pub(in crate::runtime) fn batch_resident_q4k_slot_misses_many_recording_expert_bundle_h2d(
        &mut self,
        slot_groups: &[&[&[u8]]],
        local_ptrs: &HashMap<(usize, usize), u64>,
        tracked_keys: &HashSet<(usize, usize)>,
    ) -> Result<(), String> {
        let stream = self.stream;
        self.batch_resident_q4k_slot_misses_many_on_stream_with_expert_bundle_h2d(
            slot_groups,
            local_ptrs,
            stream,
            Some(tracked_keys),
            &HashSet::new(),
            true,
            None,
        )
        .map(|_| ())
    }

    pub(in crate::runtime) fn batch_resident_q4k_slot_misses_many_on_stream(
        &mut self,
        slot_groups: &[&[&[u8]]],
        local_ptrs: &HashMap<(usize, usize), u64>,
        upload_stream: usize,
    ) -> Result<bool, String> {
        self.batch_resident_q4k_slot_misses_many_on_stream_with_expert_bundle_h2d(
            slot_groups,
            local_ptrs,
            upload_stream,
            None,
            &HashSet::new(),
            true,
            None,
        )
        .map(|result| result.uploaded)
    }

    #[cfg(test)]
    pub(in crate::runtime) fn batch_resident_q4k_slot_misses_many_on_stream_recording_expert_bundle_h2d(
        &mut self,
        slot_groups: &[&[&[u8]]],
        local_ptrs: &HashMap<(usize, usize), u64>,
        upload_stream: usize,
        tracked_keys: &HashSet<(usize, usize)>,
    ) -> Result<bool, String> {
        self.batch_resident_q4k_slot_misses_many_on_stream_with_expert_bundle_h2d(
            slot_groups,
            local_ptrs,
            upload_stream,
            Some(tracked_keys),
            &HashSet::new(),
            true,
            None,
        )
        .map(|result| result.uploaded)
    }

    pub(in crate::runtime) fn batch_resident_q4k_slot_misses_many_protecting(
        &mut self,
        slot_groups: &[&[&[u8]]],
        local_ptrs: &HashMap<(usize, usize), u64>,
        protected_keys: &HashSet<(usize, usize)>,
    ) -> Result<bool, String> {
        let stream = self.stream;
        self.batch_resident_q4k_slot_misses_many_on_stream_with_expert_bundle_h2d(
            slot_groups,
            local_ptrs,
            stream,
            None,
            protected_keys,
            false,
            None,
        )
        .map(|result| result.uploaded)
    }

    pub(in crate::runtime) fn batch_resident_q4k_slot_misses_many_recording_expert_bundle_h2d_protecting(
        &mut self,
        slot_groups: &[&[&[u8]]],
        local_ptrs: &HashMap<(usize, usize), u64>,
        protected_keys: &HashSet<(usize, usize)>,
        tracked_keys: &HashSet<(usize, usize)>,
    ) -> Result<bool, String> {
        let stream = self.stream;
        self.batch_resident_q4k_slot_misses_many_on_stream_with_expert_bundle_h2d(
            slot_groups,
            local_ptrs,
            stream,
            Some(tracked_keys),
            protected_keys,
            false,
            None,
        )
        .map(|result| result.uploaded)
    }

    pub(in crate::runtime) fn batch_resident_q4k_slot_misses_many_for_profitable_bundle(
        &mut self,
        slot_groups: &[&[&[u8]]],
        local_ptrs: &HashMap<(usize, usize), u64>,
        protected_keys: &HashSet<(usize, usize)>,
        additional_oom_reload_budget: u64,
    ) -> Result<ResidentQ4kAdmissionResult, String> {
        let stream = self.stream;
        self.batch_resident_q4k_slot_misses_many_on_stream_with_expert_bundle_h2d(
            slot_groups,
            local_ptrs,
            stream,
            None,
            protected_keys,
            false,
            Some(additional_oom_reload_budget),
        )
    }

    fn batch_resident_q4k_slot_misses_many_on_stream_with_expert_bundle_h2d(
        &mut self,
        slot_groups: &[&[&[u8]]],
        local_ptrs: &HashMap<(usize, usize), u64>,
        upload_stream: usize,
        tracked_expert_bundle_keys: Option<&HashSet<(usize, usize)>>,
        protected_keys: &HashSet<(usize, usize)>,
        allow_global_oom_offload: bool,
        additional_oom_reload_budget: Option<u64>,
    ) -> Result<ResidentQ4kAdmissionResult, String> {
        let mut missing = Vec::new();
        let mut seen = HashSet::new();
        let mut slab_bytes = 0usize;
        for slot_weights in slot_groups {
            for &weights in *slot_weights {
                let key = q4k_resident_key(weights);
                if local_ptrs.contains_key(&key)
                    || self.resident_q4k.contains_key(&key)
                    || !seen.insert(key)
                {
                    continue;
                }
                let aligned = align_up(slab_bytes, 256);
                slab_bytes = aligned.saturating_add(weights.len());
                missing.push((key, aligned, weights));
            }
        }
        if missing.len() < 2 || self.resident_q4k_limit < slab_bytes {
            return Ok(ResidentQ4kAdmissionResult::default());
        }

        self.evict_resident_q4k_until_protecting(slab_bytes, protected_keys)?;
        if self.resident_q4k_bytes.saturating_add(slab_bytes) > self.resident_q4k_limit {
            return Ok(ResidentQ4kAdmissionResult::default());
        }
        let (slab, evictions) = if let Some(oom_budget) = additional_oom_reload_budget {
            match self.resident_q4k_mem_alloc_with_profitable_oom_retry(
                slab_bytes,
                protected_keys,
                oom_budget,
            ) {
                Ok(result) => result,
                Err(err) if cuda_mem_alloc_oom(&err) => {
                    return Ok(ResidentQ4kAdmissionResult::default());
                }
                Err(err) => return Err(err),
            }
        } else {
            match self.resident_q4k_mem_alloc(slab_bytes) {
                Ok(ptr) => (ptr, rnb_memory::ExpertBundleCacheStats::default()),
                Err(err) if cuda_mem_alloc_oom(&err) => {
                    match self.resident_q4k_mem_alloc_after_oom_with_bundle_eviction_retry(
                        err,
                        slab_bytes,
                        protected_keys,
                    ) {
                        Ok(result) => result,
                        Err(retry_err)
                            if allow_global_oom_offload
                                && cuda_offload_on_oom_enabled()
                                && cuda_mem_alloc_oom(&retry_err) =>
                        {
                            let _ = self.offload_non_pinned_resident_q4k()?;
                            return Ok(ResidentQ4kAdmissionResult::default());
                        }
                        Err(retry_err) if cuda_mem_alloc_oom(&retry_err) => {
                            return Ok(ResidentQ4kAdmissionResult::default());
                        }
                        Err(retry_err) => return Err(retry_err),
                    }
                }
                Err(err) => return Err(err),
            }
        };
        let pinned_staging =
            tuning::resident_q4k_batch_pinned_staging_enabled(slab_bytes, missing.len());
        if pinned_staging {
            let host_slab = self.host_temp_slab_ptr(slab_bytes)?;
            for (_, offset, weights) in &missing {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        weights.as_ptr(),
                        host_slab.add(*offset),
                        weights.len(),
                    );
                }
            }
            unsafe {
                self.api.memcpy_htod_async(
                    slab,
                    host_slab.cast::<libc::c_void>(),
                    slab_bytes,
                    upload_stream,
                )?;
            }
        } else {
            for (_, offset, weights) in &missing {
                unsafe {
                    self.api.memcpy_htod_async(
                        slab + *offset as u64,
                        weights.as_ptr().cast::<libc::c_void>(),
                        weights.len(),
                        upload_stream,
                    )?;
                }
            }
        }
        if let Some(tracked_keys) = tracked_expert_bundle_keys {
            let tracked_payload_bytes = missing.iter().fold(0u64, |bytes, (key, _, weights)| {
                if tracked_keys.contains(key) {
                    bytes.saturating_add(weights.len() as u64)
                } else {
                    bytes
                }
            });
            cache_stats().record_expert_bundle_h2d(tracked_payload_bytes, false);
        }
        cache_stats()
            .resident_upload_bytes
            .fetch_add(slab_bytes as u64, Ordering::Relaxed);
        self.resident_q4k_slabs.insert(
            slab,
            ResidentQ4kSlab {
                bytes: slab_bytes,
                live_entries: missing.len(),
            },
        );
        let missing_len = missing.len();
        for idx in 0..missing_len {
            let (key, offset, weights) = missing[idx];
            let epoch = self.next_resident_q4k_epoch();
            self.resident_q4k.insert(
                key,
                ResidentQ4k {
                    ptr: slab + offset as u64,
                    bytes: weights.len(),
                    epoch,
                    owned_alloc: false,
                    slab_base: Some(slab),
                    pinned: false,
                },
            );
            self.resident_q4k_lru.push_back((key, epoch));
            self.record_raw_quant_residency("Q4_K", weights.len());
        }
        self.resident_q4k_bytes = self.resident_q4k_bytes.saturating_add(slab_bytes);
        Ok(ResidentQ4kAdmissionResult {
            uploaded: true,
            evictions,
        })
    }
}

#[cfg(test)]
mod direct_file_tests {
    use super::*;

    #[test]
    fn direct_file_slab_plan_coalesces_only_adjacent_ranges() {
        let plan = plan_direct_file_slab(vec![
            DirectFileEntry {
                key: (1, 10),
                region_index: 0,
                file_offset: 4096 + 32,
                len: 4064,
            },
            DirectFileEntry {
                key: (2, 20),
                region_index: 0,
                file_offset: 8192,
                len: 4096,
            },
            DirectFileEntry {
                key: (3, 30),
                region_index: 0,
                file_offset: 16384,
                len: 4096,
            },
        ])
        .unwrap();

        assert_eq!(plan.logical_bytes, 12_256);
        assert_eq!(plan.staging_bytes, 12_288);
        assert_eq!(
            plan.runs,
            vec![
                DirectFileRun {
                    region_index: 0,
                    file_offset: 4096,
                    staging_offset: 0,
                    staging_len: 8192,
                    required_len: 8192,
                },
                DirectFileRun {
                    region_index: 0,
                    file_offset: 16384,
                    staging_offset: 8192,
                    staging_len: 4096,
                    required_len: 4096,
                },
            ]
        );
        assert_eq!(plan.offsets_by_key[&(1, 10)], 32);
        assert_eq!(plan.offsets_by_key[&(2, 20)], 4096);
        assert_eq!(plan.offsets_by_key[&(3, 30)], 8192);
    }
}
