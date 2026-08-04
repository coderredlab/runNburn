//! Deterministic selected-expert execution backed by the resident MoE
//! slice cache.
//!
//! Unlike the grouped prefill path, this path never groups slots across
//! tokens and never uses atomic accumulation: gate/up run per slot, the
//! clamped SwiGLU runs elementwise, and the down projection reduces each
//! token's slots in fixed slot order. A token therefore produces bitwise
//! identical output whether it is executed alone (decode) or inside a
//! batch (speculative verify), and independent of cache hit state.

use super::super::*;

pub(in crate::runtime) struct ResidentSparseExpertsRequest<'a> {
    pub gate_weights: &'a [&'a [u8]],
    pub up_weights: &'a [&'a [u8]],
    pub down_weights: &'a [&'a [u8]],
    /// GGML quant type codes: gate/up and down.
    pub gate_quant: u32,
    pub down_quant: u32,
    pub route_weights: &'a [f32],
    pub token_ids: &'a [u32],
    pub token_count: usize,
    pub n_ff: usize,
    pub n_embd: usize,
    pub input: &'a [f32],
    pub activation_limit: f32,
}

/// Diagnostic split of the resident call into weight resolution (cache
/// lookup plus miss uploads), operand staging, kernels, and the result
/// download that also drains the stream. Enabled by
/// `RNB_CUDA_MOE_RESIDENT_TRACE=1`; the phase boundaries need stream
/// synchronization, so this perturbs timing and is diagnostic only.
mod phase_trace {
    use std::sync::atomic::{AtomicU64, Ordering};

    static CALLS: AtomicU64 = AtomicU64::new(0);
    static RESOLVE_NS: AtomicU64 = AtomicU64::new(0);
    static STAGE_NS: AtomicU64 = AtomicU64::new(0);
    static KERNEL_NS: AtomicU64 = AtomicU64::new(0);
    static DOWNLOAD_NS: AtomicU64 = AtomicU64::new(0);
    static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
    static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);

    pub(super) fn enabled() -> bool {
        std::env::var("RNB_CUDA_MOE_RESIDENT_TRACE")
            .ok()
            .as_deref()
            == Some("1")
    }

    pub(super) fn record(
        resolve_ns: u64,
        stage_ns: u64,
        kernel_ns: u64,
        download_ns: u64,
        upload_bytes: u64,
        upload_ns: u64,
    ) {
        RESOLVE_NS.fetch_add(resolve_ns, Ordering::Relaxed);
        STAGE_NS.fetch_add(stage_ns, Ordering::Relaxed);
        KERNEL_NS.fetch_add(kernel_ns, Ordering::Relaxed);
        DOWNLOAD_NS.fetch_add(download_ns, Ordering::Relaxed);
        UPLOAD_BYTES.fetch_add(upload_bytes, Ordering::Relaxed);
        UPLOAD_NS.fetch_add(upload_ns, Ordering::Relaxed);
        let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        // ~20 decode tokens on a 43-layer model keeps the log readable.
        if calls % 860 != 0 {
            return;
        }
        let ms = |ns: u64| ns as f64 / 1.0e6;
        let resolve = RESOLVE_NS.load(Ordering::Relaxed);
        let upload = UPLOAD_NS.load(Ordering::Relaxed);
        let bytes = UPLOAD_BYTES.load(Ordering::Relaxed);
        // `xfer` isolates the memcpy calls themselves; `resolve - xfer` is the
        // lookup and admission bookkeeping around them.
        let xfer_gbps = if upload > 0 {
            bytes as f64 / upload as f64
        } else {
            0.0
        };
        eprintln!(
            "[cuda-moe-resident] calls={calls} resolve={:.1}ms (xfer={:.1}ms bookkeep={:.1}ms) stage={:.1}ms kernels={:.1}ms download={:.1}ms upload={:.2}GiB xfer_h2d={xfer_gbps:.2}GB/s",
            ms(resolve),
            ms(upload),
            ms(resolve.saturating_sub(upload)),
            ms(STAGE_NS.load(Ordering::Relaxed)),
            ms(KERNEL_NS.load(Ordering::Relaxed)),
            ms(DOWNLOAD_NS.load(Ordering::Relaxed)),
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
}

fn resident_kernels(
    gate_quant: u32,
    down_quant: u32,
) -> Result<(usize, &'static str, usize, &'static str), String> {
    let (gate_block_bytes, gate_kernel) = match gate_quant {
        16 => (66usize, "rnb_iq2_xxs_selected_gate_up_gemv_by_token"),
        22 => (82usize, "rnb_iq2_s_selected_gate_up_gemv_by_token"),
        39 => (136usize, "rnb_mxfp4_selected_gate_up_gemv_by_token"),
        other => {
            return Err(format!(
                "unsupported resident sparse gate/up quant code {other}"
            ));
        }
    };
    let (down_block_bytes, down_kernel) = match down_quant {
        18 => (
            98usize,
            "rnb_iq3_xxs_selected_down_activated_rowreduce_by_token",
        ),
        39 => (
            136usize,
            "rnb_mxfp4_selected_down_activated_rowreduce_by_token",
        ),
        other => {
            return Err(format!(
                "unsupported resident sparse down quant code {other}"
            ));
        }
    };
    Ok((gate_block_bytes, gate_kernel, down_block_bytes, down_kernel))
}

impl CudaState {
    /// Deterministic selected-expert forward through the resident slice
    /// cache. Returns `Ok(None)` when the cache is disabled or cannot hold
    /// the request, so callers can fall back to their existing paths.
    pub(in crate::runtime) fn sparse_experts_by_token_resident(
        &mut self,
        request: ResidentSparseExpertsRequest<'_>,
    ) -> Result<Option<Vec<f32>>, String> {
        let ResidentSparseExpertsRequest {
            gate_weights,
            up_weights,
            down_weights,
            gate_quant,
            down_quant,
            route_weights,
            token_ids,
            token_count,
            n_ff,
            n_embd,
            input,
            activation_limit,
        } = request;
        let slots = gate_weights.len();
        if token_count == 0 || slots == 0 || slots % token_count != 0 {
            return Err(format!(
                "resident sparse slots must be non-zero and divisible by token_count: slots={slots} token_count={token_count}"
            ));
        }
        if up_weights.len() != slots
            || down_weights.len() != slots
            || route_weights.len() != slots
            || token_ids.len() != slots
        {
            return Err(format!(
                "resident sparse slot mismatch: gate={} up={} down={} route={} token_ids={}",
                slots,
                up_weights.len(),
                down_weights.len(),
                route_weights.len(),
                token_ids.len()
            ));
        }
        if input.len() != token_count.saturating_mul(n_embd) {
            return Err(format!(
                "resident sparse input mismatch: got={} expected={}",
                input.len(),
                token_count.saturating_mul(n_embd)
            ));
        }
        if n_embd % 256 != 0 || n_ff % 256 != 0 {
            return Err(format!(
                "resident sparse dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
            ));
        }
        if !activation_limit.is_finite() || activation_limit <= 0.0 {
            return Err(format!(
                "resident sparse SwiGLU clamp must be finite and positive, got {activation_limit}"
            ));
        }
        // Slot order must be token-major so the down row reduce sees each
        // token's slots contiguously in route order.
        let slots_per_token = slots / token_count;
        for (slot, &token) in token_ids.iter().enumerate() {
            if token as usize != slot / slots_per_token {
                return Err(
                    "resident sparse token ids must be token-major and contiguous".to_string(),
                );
            }
        }
        let (gate_block_bytes, gate_kernel, down_block_bytes, down_kernel) =
            resident_kernels(gate_quant, down_quant)?;
        let gate_row_bytes = (n_embd / 256) * gate_block_bytes;
        let down_row_bytes = (n_ff / 256) * down_block_bytes;
        for (slot, weights) in gate_weights.iter().chain(up_weights.iter()).enumerate() {
            if weights.len() != n_ff * gate_row_bytes {
                return Err(format!(
                    "resident sparse gate/up[{slot}] byte mismatch: got {}, expected {}",
                    weights.len(),
                    n_ff * gate_row_bytes
                ));
            }
        }
        for (slot, weights) in down_weights.iter().enumerate() {
            if weights.len() != n_embd * down_row_bytes {
                return Err(format!(
                    "resident sparse down[{slot}] byte mismatch: got {}, expected {}",
                    weights.len(),
                    n_embd * down_row_bytes
                ));
            }
        }

        self.set_current()?;
        let trace = phase_trace::enabled();
        let mut mark = std::time::Instant::now();
        let upload_before = self.moe_slice_cache.resident_upload_bytes
            + self.moe_slice_cache.temp_upload_bytes;
        let upload_ns_before = self.moe_slice_cache.upload_ns;
        let Some((gate_ptrs, up_ptrs, down_ptrs)) =
            self.moe_slice_resident_ptrs_3(gate_weights, up_weights, down_weights)?
        else {
            return Ok(None);
        };
        let resolve_ns = if trace {
            self.stream_synchronize()?;
            let elapsed = mark.elapsed().as_nanos() as u64;
            mark = std::time::Instant::now();
            elapsed
        } else {
            0
        };

        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = token_count * n_embd * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let gate_dev = self.compute_mid_a_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        let up_dev = self.compute_mid_b_ptr(slots * n_ff * std::mem::size_of::<f32>())?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }

        let ptr_bytes = slots * std::mem::size_of::<u64>();
        let route_bytes = std::mem::size_of_val(route_weights);
        let token_bytes = std::mem::size_of_val(token_ids);
        let meta_bytes = ptr_bytes * 3 + route_bytes + token_bytes;
        let gate_ptrs_dev = self.compute_gate_ptrs_ptr(meta_bytes)?;
        let up_ptrs_dev = gate_ptrs_dev + ptr_bytes as u64;
        let down_ptrs_dev = gate_ptrs_dev + (ptr_bytes * 2) as u64;
        let route_dev = gate_ptrs_dev + (ptr_bytes * 3) as u64;
        let token_ids_dev = route_dev + route_bytes as u64;
        let mut meta = vec![0u8; meta_bytes];
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
            self.api.memcpy_htod_async(
                gate_ptrs_dev,
                meta.as_ptr().cast::<libc::c_void>(),
                meta_bytes,
                self.stream,
            )?;
        }
        let stage_ns = if trace {
            self.stream_synchronize()?;
            let elapsed = mark.elapsed().as_nanos() as u64;
            mark = std::time::Instant::now();
            elapsed
        } else {
            0
        };

        self.launch_selected_glm_iq_gate_up_gemv_by_token_to_dev(
            gate_kernel,
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
        self.launch_swiglu_clamped(gate_dev, up_dev, activation_limit, slots * n_ff)?;
        self.launch_selected_down_silu_rowreduce_by_token(
            down_kernel,
            down_ptrs_dev,
            n_embd,
            slots_per_token,
            token_count,
            n_ff / 256,
            gate_dev,
            up_dev,
            route_dev,
            output_dev,
        )?;
        let kernel_ns = if trace {
            self.stream_synchronize()?;
            let elapsed = mark.elapsed().as_nanos() as u64;
            mark = std::time::Instant::now();
            elapsed
        } else {
            0
        };

        let mut output = vec![0.0f32; token_count * n_embd];
        self.dtoh_f32_via_pinned(output_dev, &mut output)?;
        if trace {
            let download_ns = mark.elapsed().as_nanos() as u64;
            let uploaded = (self.moe_slice_cache.resident_upload_bytes
                + self.moe_slice_cache.temp_upload_bytes)
                .saturating_sub(upload_before);
            let upload_ns = self
                .moe_slice_cache
                .upload_ns
                .saturating_sub(upload_ns_before);
            phase_trace::record(
                resolve_ns,
                stage_ns,
                kernel_ns,
                download_ns,
                uploaded,
                upload_ns,
            );
        }
        Ok(Some(output))
    }
}
