// Persistent decode (Gemma4 E2B) — single-launch cooperative kernel.
//
// Layered build:
//   M1 (this file): module/symbol loading, device parameter table layout,
//                   single-launch smoke entry that runs the M1 kernel
//                   (attn_norm + Q projection per layer, no actual decode wiring
//                   into the engine yet).
//   M2: extend the kernel to perform K/V projection, RoPE, KV cache write, and
//       a basic attention path so layer-to-layer hidden flows fully on device.
//   M3: full pipeline (FFN, PLE, residual, output logits) + correctness/perf
//       gates and Rust-side wire into the decode dispatcher.

use super::driver::{PERSISTENT_DECODE_CUBIN, PERSISTENT_DECODE_PTX};
use super::types::{CudaState, PersistentDecodeReusable};
use rnb_backend_api::PersistentDecodeRequest;

// Mirror of `PersistentLayerParams` in cuda/kernels/persistent_decode.cuh.  The
// layout MUST stay byte-identical with the C++ side; any field change requires
// rebuilding both the kernel PTX and this struct in lock-step.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::runtime) struct PersistentLayerParamsHost {
    pub(in crate::runtime) q_weight: u64,
    pub(in crate::runtime) k_weight: u64,
    pub(in crate::runtime) v_weight: u64,
    pub(in crate::runtime) o_weight: u64,
    pub(in crate::runtime) gate_weight: u64,
    pub(in crate::runtime) up_weight: u64,
    pub(in crate::runtime) down_weight: u64,
    pub(in crate::runtime) attn_norm: u64,
    pub(in crate::runtime) post_attn_norm: u64,
    pub(in crate::runtime) ffn_norm: u64,
    pub(in crate::runtime) post_ffn_norm: u64,
    pub(in crate::runtime) q_norm: u64,
    pub(in crate::runtime) k_norm: u64,
    pub(in crate::runtime) ple_gate: u64,
    pub(in crate::runtime) ple_proj: u64,
    pub(in crate::runtime) ple_post_norm: u64,
    pub(in crate::runtime) ple_input: u64,
    pub(in crate::runtime) k_cache: u64,
    pub(in crate::runtime) v_cache: u64,
    pub(in crate::runtime) head_dim: u32,
    pub(in crate::runtime) q_dim: u32,
    pub(in crate::runtime) kv_dim: u32,
    pub(in crate::runtime) n_ff: u32,
    pub(in crate::runtime) sliding_window: u32,
    pub(in crate::runtime) kv_source_layer: u32,
    pub(in crate::runtime) layer_output_scale: f32,
    pub(in crate::runtime) flags: u32,
}

// Mirror of `PersistentDecodeParams` in the .cuh.  This struct is passed by
// value into the cooperative kernel (CUDA driver copies it into kernel param
// memory); device-side it appears at param offset 0.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::runtime) struct PersistentDecodeParamsHost {
    pub(in crate::runtime) layers_dev: u64,
    pub(in crate::runtime) num_layers: u32,
    pub(in crate::runtime) hidden_dim: u32,
    pub(in crate::runtime) vocab_size: u32,
    pub(in crate::runtime) norm_eps: f32,
    pub(in crate::runtime) rope_pos: u32,
    pub(in crate::runtime) kv_len: u32,
    pub(in crate::runtime) hidden_dev: u64,
    pub(in crate::runtime) normed_dev: u64,
    pub(in crate::runtime) attn_out_dev: u64,
    pub(in crate::runtime) q_buf_dev: u64,
    pub(in crate::runtime) k_buf_dev: u64,
    pub(in crate::runtime) v_buf_dev: u64,
    pub(in crate::runtime) gate_buf_dev: u64,
    pub(in crate::runtime) up_buf_dev: u64,
    pub(in crate::runtime) ple_gate_buf_dev: u64,
    pub(in crate::runtime) output_weight: u64,
    pub(in crate::runtime) logits_dev: u64,
    pub(in crate::runtime) argmax_dev: u64,
    pub(in crate::runtime) hidden_probe_dev: u64,
    pub(in crate::runtime) normed_after_attn_norm_probe_dev: u64,
    pub(in crate::runtime) hidden_after_attn_probe_dev: u64,
    pub(in crate::runtime) hidden_after_ffn_probe_dev: u64,
    pub(in crate::runtime) rope_freqs_dev: u64,
    pub(in crate::runtime) attn_out_probe_dev: u64,
    pub(in crate::runtime) q_proj_probe_dev: u64,
    pub(in crate::runtime) k_proj_probe_dev: u64,
    pub(in crate::runtime) v_proj_probe_dev: u64,
    pub(in crate::runtime) attn_scores_probe_dev: u64,
    pub(in crate::runtime) attn_v_probe_dev: u64,
    pub(in crate::runtime) attn_acc_probe_dev: u64,
    pub(in crate::runtime) attn_row_sum_probe_dev: u64,
    pub(in crate::runtime) probe_layer_idx: u32,
    pub(in crate::runtime) hidden_after_ffn_only_probe_dev: u64,
    pub(in crate::runtime) ffn_gate_probe_dev: u64,
    pub(in crate::runtime) ffn_gated_probe_dev: u64,
    pub(in crate::runtime) ffn_down_probe_dev: u64,
    pub(in crate::runtime) layer_hidden_trace_dev: u64,
    pub(in crate::runtime) output_norm_dev: u64,
    pub(in crate::runtime) nan_trace: u32,
    pub(in crate::runtime) gemma_v_norm: u32,
    pub(in crate::runtime) seq_len: u32,
    // cu101 M3 batch attention: token-slot stride for q_buf / attn_out. The
    // kernel advances `q_buf`/`attn_out` by `__t * q_dim_max` between the
    // per-token pre/attention/post phases. Decode (seq_len=1) uses stride 0.
    pub(in crate::runtime) q_dim_max: u32,
    // cu102 M4 batch FFN: batch-slot FFN buffers + n_ff stride. ffn_normed_dev /
    // ffn_down_dev are hidden_dim-wide per-token slots (ffn_norm output / down
    // output); gate_buf/up_buf are walked at n_ff_max stride in the batch FFN
    // phase. Tail-appended per cu77 ABI rule (cuh mirror extends at the tail).
    pub(in crate::runtime) ffn_normed_dev: u64,
    pub(in crate::runtime) ffn_down_dev: u64,
    pub(in crate::runtime) n_ff_max: u32,
    // cu105 QKV batch-tiling — see .cuh for the phase A1/A2/A3 layout note.
    // attn_normed_dev = hidden_dim-wide batch slots (attn_norm output = Q/K/V
    // projection input); k_buf/v_buf are sized to batch slots (kv_dim_max
    // stride) so phase A3 reads each token's K/V after the batch projection.
    // kv_dim_max (u32) before attn_normed_dev (ptr) to avoid a pad slot, matching
    // the .cuh field order. Tail-appended per cu77 ABI rule.
    pub(in crate::runtime) kv_dim_max: u32,
    pub(in crate::runtime) attn_normed_dev: u64,
}

impl CudaState {
    /// Load (or fetch cached) the persistent decode module and return its raw handle.
    pub(in crate::runtime) fn ensure_persistent_decode_module(&mut self) -> Result<usize, String> {
        self.set_current()?;
        if self.persistent_decode_module.is_none() {
            let module = unsafe {
                if crate::tuning::cubin_modules_enabled() {
                    self.api
                        .module_load_cubin_or_ptx(PERSISTENT_DECODE_CUBIN, PERSISTENT_DECODE_PTX)?
                } else {
                    self.api.module_load_data(PERSISTENT_DECODE_PTX)?
                }
            };
            self.persistent_decode_module = Some(module as usize);
        }
        self.persistent_decode_module
            .ok_or_else(|| "missing persistent decode module".to_string())
    }

    /// Resolve the kernel entry point inside the persistent decode module.
    /// `name` is the demangled `extern "C"` symbol (e.g. `rnb_persistent_decode_e2b_m1`).
    pub(in crate::runtime) fn persistent_decode_function(
        &mut self,
        name: &str,
    ) -> Result<*mut libc::c_void, String> {
        let module = self.ensure_persistent_decode_module()?;
        unsafe {
            self.api
                .module_get_function(module as *mut libc::c_void, name)
        }
    }

    /// Cooperative launch of the persistent decode kernel.  Caller supplies the
    /// fully-populated `PersistentDecodeParamsHost`; the CUDA driver copies it
    /// by value into kernel param memory at launch time.
    ///
    /// `grid` should be clamped to `device_multiprocessor_count() *
    /// occupancy_max_active_blocks_per_multiprocessor()` so a single wave fills
    /// every SM (required for `grid.sync()` to deadlock-free progress).
    pub(in crate::runtime) fn launch_persistent_decode_m1(
        &mut self,
        params: &mut PersistentDecodeParamsHost,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
    ) -> Result<(), String> {
        let function = self.persistent_decode_function("rnb_persistent_decode_e2b_m1")?;
        // Single kernel arg: pointer to the params struct in pinned host memory
        // is NOT acceptable here — the kernel reads `PersistentDecodeParams`
        // by value, so we pass a pointer to a local copy and the driver will
        // copy the struct contents into kernel param memory.
        let params_ptr = params as *mut PersistentDecodeParamsHost as *mut libc::c_void;
        let mut args: [*mut libc::c_void; 1] = [params_ptr];
        unsafe {
            self.api.launch_cooperative_kernel(
                function,
                grid,
                block,
                shared_mem_bytes,
                self.stream,
                args.as_mut_ptr(),
            )
        }
    }

    /// Returns `(sm_count, max_blocks_per_sm)` for the M1 kernel under the
    /// given block size and dynamic shared memory request.  Used by the host
    /// to clamp grid size to a single-wave cooperative launch.
    pub(in crate::runtime) fn persistent_decode_m1_occupancy(
        &mut self,
        block_threads: i32,
        shared_mem_bytes: usize,
    ) -> Result<(i32, i32), String> {
        let function = self.persistent_decode_function("rnb_persistent_decode_e2b_m1")?;
        let sm_count = unsafe { self.api.device_multiprocessor_count()? };
        let max_blocks_per_sm = unsafe {
            self.api.occupancy_max_active_blocks_per_multiprocessor(
                function,
                block_threads,
                shared_mem_bytes,
            )?
        };
        Ok((sm_count, max_blocks_per_sm))
    }

    /// Ensure the persistent decode reusable buffer context is allocated and
    /// large enough for the requested shapes.  Per-layer KV slots are allocated
    /// once at `max_seq_len * kv_dim_max * 2` bytes; scratch / output buffers
    /// are allocated once and reused.  Returns a mutable reference to the
    /// reusable buffer set.
    pub(in crate::runtime) fn ensure_persistent_decode_ctx(
        &mut self,
        num_layers: usize,
        max_seq_len: u32,
        batch_seq_cap: u32,
        hidden_dim: u32,
        q_dim_max: u32,
        kv_dim_max: u32,
        n_ff_max: u32,
        ple_dim: u32,
        vocab_size: u32,
        owns_kv: &[bool],
    ) -> Result<(), String> {
        self.set_current()?;
        let needs_realloc = match &self.persistent_decode_ctx {
            None => true,
            Some(ctx) => {
                ctx.num_layers != num_layers
                    || ctx.max_seq_len < max_seq_len
                    || ctx.batch_seq_cap < batch_seq_cap
                    || ctx.kv_dim_max < kv_dim_max
                    || ctx.q_dim_max < q_dim_max
                    || ctx.n_ff_max < n_ff_max
                    || ctx.hidden_dim != hidden_dim
                    || ctx.vocab_size != vocab_size
                    || ctx.ple_dim != ple_dim
            }
        };
        if !needs_realloc {
            return Ok(());
        }
        // Free any previous allocation first (shape grew).
        if let Some(prev) = self.persistent_decode_ctx.take() {
            unsafe {
                for &p in prev.k_cache_devs.iter().chain(prev.v_cache_devs.iter()) {
                    if p != 0 {
                        let _ = self.api.mem_free(p);
                    }
                }
                for p in [
                    prev.layers_dev,
                    prev.hidden_dev,
                    prev.normed_dev,
                    prev.attn_out_dev,
                    prev.q_buf_dev,
                    prev.k_buf_dev,
                    prev.v_buf_dev,
                    prev.gate_buf_dev,
                    prev.up_buf_dev,
                    prev.ple_gate_buf_dev,
                    prev.ffn_normed_dev,
                    prev.ffn_down_dev,
                    prev.attn_normed_dev,
                    prev.logits_dev,
                    prev.argmax_dev,
                ] {
                    if p != 0 {
                        let _ = self.api.mem_free(p);
                    }
                }
            }
        }
        let kv_bytes = (max_seq_len as usize) * (kv_dim_max as usize) * std::mem::size_of::<u16>();
        let bytes_f32 = |n: usize| n * std::mem::size_of::<f32>();
        let layers_bytes = num_layers * std::mem::size_of::<PersistentLayerParamsHost>();

        let mut k_cache_devs = vec![0u64; num_layers];
        let mut v_cache_devs = vec![0u64; num_layers];
        for idx in 0..num_layers {
            if owns_kv[idx] {
                k_cache_devs[idx] = unsafe { self.api.mem_alloc(kv_bytes)? };
                v_cache_devs[idx] = unsafe { self.api.mem_alloc(kv_bytes)? };
            }
        }
        let layers_dev = unsafe { self.api.mem_alloc(layers_bytes)? };
        // cu100/cu101 Milestone 2 — size hidden_dev to fit a reasonable batch
        // prefill seq_len. We cap at min(max_seq_len, MAX_BATCH_PREFILL=4096)
        // so very large KV cache configurations (e.g. 131072 context) don't
        // OOM the staging buffer (1536 × 131072 × 4 = 805 MB). 4096 tokens ×
        // 1536 × 4 = 24 MB which is fine on Ampere.
        // cu104: size batch scratch to the actual batch seq_len, not max_seq_len
        // (= max_ctx). A 47-token prefill no longer over-allocates 4096-slot
        // gate/up/q/attn buffers (which OOM'd large-context models). KV cache
        // below still uses max_seq_len (full context needed).
        const MAX_BATCH_PREFILL: u32 = 4096;
        let hidden_slots = (batch_seq_cap.min(MAX_BATCH_PREFILL).max(1)) as usize;
        let hidden_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (hidden_dim as usize)))?
        };
        let normed_dev = unsafe { self.api.mem_alloc(bytes_f32(hidden_dim as usize))? };
        // cu101 M3 batch attention: q_buf / attn_out must hold all `hidden_slots`
        // tokens token-major (q_buf[__t * q_dim + lane]) so a single batch causal
        // attention dispatch reads every query at once. Decode (seq_len=1) uses
        // only slot 0, so behavior is unchanged. normed / k_buf / v_buf stay
        // single-slot — they are consumed within each token's pre-attention work.
        let attn_out_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (q_dim_max as usize)))?
        };
        let q_buf_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (q_dim_max as usize)))?
        };
        // cu105 QKV batch-tiling: k_buf/v_buf become batch slots (hidden_slots ×
        // kv_dim_max) so phase A2 batch-projects every token's K/V and phase A3
        // reads each token's slot for per-token QK/V norm + RoPE + KV write.
        // Decode (seq_len=1) uses slot 0.
        let k_buf_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (kv_dim_max as usize)))?
        };
        let v_buf_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (kv_dim_max as usize)))?
        };
        // cu102 M4 batch FFN: gate/up batched to hidden_slots × n_ff_max so all
        // tokens' FFN intermediates coexist for batch GEMM. ffn_normed (ffn_norm
        // output = gate/up input) and ffn_down (down output) are hidden_dim-wide
        // batch slots. Decode (seq_len=1) uses slot 0.
        let gate_buf_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (n_ff_max as usize)))?
        };
        let up_buf_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (n_ff_max as usize)))?
        };
        let ple_gate_buf_dev = unsafe { self.api.mem_alloc(bytes_f32(ple_dim as usize))? };
        let ffn_normed_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (hidden_dim as usize)))?
        };
        let ffn_down_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (hidden_dim as usize)))?
        };
        // cu105 QKV batch-tiling: attn_norm output batch slots (= Q/K/V proj
        // input for the batch GEMM). hidden_slots × hidden_dim, like ffn_normed.
        let attn_normed_dev = unsafe {
            self.api
                .mem_alloc(bytes_f32(hidden_slots * (hidden_dim as usize)))?
        };
        let logits_dev = unsafe { self.api.mem_alloc(bytes_f32(vocab_size as usize))? };
        let argmax_dev = unsafe { self.api.mem_alloc(std::mem::size_of::<i32>())? };

        self.persistent_decode_ctx = Some(PersistentDecodeReusable {
            num_layers,
            max_seq_len,
            batch_seq_cap,
            kv_dim_max,
            q_dim_max,
            n_ff_max,
            hidden_dim,
            vocab_size,
            ple_dim,
            k_cache_devs,
            v_cache_devs,
            last_rope_pos: None,
            resident_kv_tokens: 0,
            layers_dev,
            hidden_dev,
            normed_dev,
            attn_out_dev,
            q_buf_dev,
            k_buf_dev,
            v_buf_dev,
            gate_buf_dev,
            up_buf_dev,
            ple_gate_buf_dev,
            ffn_normed_dev,
            ffn_down_dev,
            attn_normed_dev,
            logits_dev,
            argmax_dev,
        });
        Ok(())
    }

    /// One-shot persistent decode dispatch.  Populates per-layer device weight
    /// pointers from the request, uploads KV cache + input hidden, launches
    /// the cooperative kernel, and copies logits + argmax back to the host.
    ///
    /// Reusable buffers (KV cache, scratch, layer table) are allocated once
    /// per CudaState via `ensure_persistent_decode_ctx` and re-used across
    /// all subsequent decode tokens.
    ///
    /// `request.input_hidden` is the start-of-layer hidden state (already
    /// embedded + scaled by the host).  `request.output_logits` / `argmax_out`
    /// receive the result.  KV cache is read-and-written-back per call (host
    /// KVCache stays the single source).
    pub fn dispatch_persistent_decode(
        &mut self,
        request: &mut PersistentDecodeRequest<'_>,
    ) -> Result<(), String> {
        self.set_current()?;

        let num_layers = request.num_layers as usize;
        if request.layers.len() != num_layers {
            return Err(format!(
                "persistent decode layer count mismatch: request.layers.len()={} num_layers={}",
                request.layers.len(),
                num_layers
            ));
        }
        let hidden_dim = request.hidden_dim as usize;
        // cu100: input_hidden length = seq_len * hidden_dim. Decode = 1 * hidden_dim.
        // Batch prefill caller packs N token embeddings into a single slice.
        let seq_len = request.seq_len.max(1) as usize;
        let expected_hidden_len = seq_len * hidden_dim;
        if request.input_hidden.len() != expected_hidden_len {
            return Err(format!(
                "persistent decode input_hidden length mismatch: got {}, expected {expected_hidden_len} (seq_len={seq_len} * hidden_dim={hidden_dim})",
                request.input_hidden.len()
            ));
        }
        let vocab_size = request.vocab_size as usize;
        if request.output_logits.len() != vocab_size {
            return Err(format!(
                "persistent decode output_logits length mismatch: got {}, expected {vocab_size}",
                request.output_logits.len()
            ));
        }
        // 1a) Pre-compute owns_kv so the reusable context allocator knows
        //     which layers need a dedicated KV buffer vs which alias an
        //     anchor.  Owns if kv_source_layer == idx OR out-of-range
        //     (defensive against malformed input).
        let mut owns_kv: Vec<bool> = vec![false; num_layers];
        for (idx, l) in request.layers.iter().enumerate() {
            owns_kv[idx] =
                (l.kv_source_layer as usize) == idx || (l.kv_source_layer as usize) >= num_layers;
        }
        // 1b) Ensure reusable context covers this request's shapes.  After
        //     this call, KV / scratch / layer-table buffers are alive on the
        //     CudaState and reused across all subsequent decode tokens.
        self.ensure_persistent_decode_ctx(
            num_layers,
            request.max_seq_len,
            // cu104: batch scratch sized to this request's seq_len (decode=1,
            // prefill batch=N), not max_ctx — avoids 4096-slot over-alloc OOM.
            request.seq_len.max(1),
            request.hidden_dim,
            request.q_dim_max,
            request.kv_dim_max,
            request.n_ff_max,
            request.ple_dim,
            request.vocab_size,
            &owns_kv,
        )?;
        let ctx = self.persistent_decode_ctx.as_ref().expect("ensured above");
        // Snapshot reusable pointers locally so we can drop the borrow on
        // self.persistent_decode_ctx before the resident-cache lookups below
        // (those take `&mut self`).
        let k_cache_devs: Vec<u64> = ctx.k_cache_devs.clone();
        let v_cache_devs: Vec<u64> = ctx.v_cache_devs.clone();
        // cu76: detect new sequence (prefill restart). If rope_pos didn't
        // advance by exactly 1 from last dispatch, host KV cache content
        // changed (new prompt, prefill rerun, ABAB iteration boundary), and
        // we must re-upload. Otherwise device KV already holds prefill +
        // previous decode token K/V — re-uploading would clobber those
        // (host KV cache never gets the new decode K/V written back).
        let kv_upload_needed = match ctx.last_rope_pos {
            Some(prev) => request.rope_pos != prev + 1,
            None => true,
        };
        let layers_dev_reuse = ctx.layers_dev;
        let hidden_dev_reuse = ctx.hidden_dev;
        let normed_dev_reuse = ctx.normed_dev;
        let attn_out_dev_reuse = ctx.attn_out_dev;
        let q_buf_dev_reuse = ctx.q_buf_dev;
        let k_buf_dev_reuse = ctx.k_buf_dev;
        let v_buf_dev_reuse = ctx.v_buf_dev;
        let gate_buf_dev_reuse = ctx.gate_buf_dev;
        let up_buf_dev_reuse = ctx.up_buf_dev;
        let ple_gate_buf_dev_reuse = ctx.ple_gate_buf_dev;
        let ffn_normed_dev_reuse = ctx.ffn_normed_dev;
        let ffn_down_dev_reuse = ctx.ffn_down_dev;
        let attn_normed_dev_reuse = ctx.attn_normed_dev;
        let logits_dev_reuse = ctx.logits_dev;
        let argmax_dev_reuse = ctx.argmax_dev;

        // 2) Resolve each weight pointer through the resident caches so the
        //    cooperative kernel sees stable device addresses across layers.
        let mut layer_params: Vec<PersistentLayerParamsHost> = Vec::with_capacity(num_layers);
        let kv_owns: Vec<bool> = owns_kv;
        // cu93: dump KV owns/source layer mapping + host K cache NaN scan once
        if std::env::var("RNB_CUDA_PERSISTENT_DUMP_KV_MAP").is_ok() {
            for (idx, l) in request.layers.iter().enumerate() {
                let owns = kv_owns[idx];
                let src = l.kv_source_layer;
                let klen = l.k_cache_len;
                let mut k_nan = 0usize;
                let mut k_inf = 0usize;
                let mut k_max: f32 = 0.0;
                if !l.k_cache_bytes.is_null() && klen > 0 {
                    let bytes = unsafe { std::slice::from_raw_parts(l.k_cache_bytes, klen) };
                    for chunk in bytes.chunks_exact(2) {
                        let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                        // f16 → f32 minimal decode
                        let sign = (h >> 15) & 0x1;
                        let exp = ((h >> 10) & 0x1f) as i32;
                        let mant = (h & 0x3ff) as u32;
                        let f = if exp == 0x1f {
                            if mant != 0 {
                                f32::NAN
                            } else if sign == 1 {
                                f32::NEG_INFINITY
                            } else {
                                f32::INFINITY
                            }
                        } else if exp == 0 {
                            // subnormal
                            let m = mant as f32 * (1.0_f32 / (1u32 << 24) as f32);
                            if sign == 1 {
                                -m
                            } else {
                                m
                            }
                        } else {
                            let e = exp - 15;
                            let m = 1.0_f32 + (mant as f32) / 1024.0_f32;
                            let mut v = m * (2.0_f32).powi(e);
                            if sign == 1 {
                                v = -v;
                            }
                            v
                        };
                        if f.is_nan() {
                            k_nan += 1;
                        } else if f.is_infinite() {
                            k_inf += 1;
                        } else if f.abs() > k_max {
                            k_max = f.abs();
                        }
                    }
                }
                eprintln!(
                    "[cu93-kv-map] layer={idx} owns={owns} src={src} klen={klen} k_nan={k_nan} k_inf={k_inf} k_max={k_max:.4}"
                );
            }
        }

        for (idx, l) in request.layers.iter().enumerate() {
            let host_slice = |ptr: *const u8, len: usize| -> Option<&[u8]> {
                if ptr.is_null() || len == 0 {
                    None
                } else {
                    // SAFETY: Caller guarantees the host bytes outlive this call.
                    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
                }
            };
            let q4k_ptr = |s: &Self, bytes: Option<&[u8]>| -> Result<u64, String> {
                if let Some(b) = bytes {
                    // pinned so concurrent layer warm-ups don't evict each other
                    // mid-dispatch.
                    s.resident_q4k_weights_ptr_pinned_for_persistent(b)
                } else {
                    Ok(0)
                }
            };
            let f32_ptr =
                |s: &mut Self, bytes: Option<&[u8]>, label: &str| -> Result<u64, String> {
                    if let Some(b) = bytes {
                        s.resident_f32_weights_ptr_from_le_bytes(b, label)
                    } else {
                        Ok(0)
                    }
                };

            let q_weight = q4k_ptr(self, host_slice(l.q_weight_bytes, l.q_weight_len))?;
            let k_weight = q4k_ptr(self, host_slice(l.k_weight_bytes, l.k_weight_len))?;
            let v_weight = q4k_ptr(self, host_slice(l.v_weight_bytes, l.v_weight_len))?;
            let o_weight = q4k_ptr(self, host_slice(l.o_weight_bytes, l.o_weight_len))?;
            let gate_weight = q4k_ptr(self, host_slice(l.gate_weight_bytes, l.gate_weight_len))?;
            let up_weight = q4k_ptr(self, host_slice(l.up_weight_bytes, l.up_weight_len))?;
            let down_weight = q4k_ptr(self, host_slice(l.down_weight_bytes, l.down_weight_len))?;

            let attn_norm = f32_ptr(
                self,
                host_slice(l.attn_norm_bytes, l.attn_norm_len),
                "persistent attn_norm",
            )?;
            let post_attn_norm = f32_ptr(
                self,
                host_slice(l.post_attn_norm_bytes, l.post_attn_norm_len),
                "persistent post_attn_norm",
            )?;
            let ffn_norm = f32_ptr(
                self,
                host_slice(l.ffn_norm_bytes, l.ffn_norm_len),
                "persistent ffn_norm",
            )?;
            let post_ffn_norm = f32_ptr(
                self,
                host_slice(l.post_ffn_norm_bytes, l.post_ffn_norm_len),
                "persistent post_ffn_norm",
            )?;
            let q_norm = f32_ptr(
                self,
                host_slice(l.q_norm_bytes, l.q_norm_len),
                "persistent q_norm",
            )?;
            let k_norm = f32_ptr(
                self,
                host_slice(l.k_norm_bytes, l.k_norm_len),
                "persistent k_norm",
            )?;
            // PLE gate / proj can be either F32 (flags & PLE_F32) or Q4K.  The
            // caller sets the appropriate flag; we route the bytes accordingly.
            let ple_is_f32 = l.flags & rnb_backend_api::PERSISTENT_DECODE_FLAG_PLE_F32 != 0;
            let ple_gate = if ple_is_f32 {
                f32_ptr(
                    self,
                    host_slice(l.ple_gate_bytes, l.ple_gate_len),
                    "persistent ple_gate_f32",
                )?
            } else {
                q4k_ptr(self, host_slice(l.ple_gate_bytes, l.ple_gate_len))?
            };
            let ple_proj = if ple_is_f32 {
                f32_ptr(
                    self,
                    host_slice(l.ple_proj_bytes, l.ple_proj_len),
                    "persistent ple_proj_f32",
                )?
            } else {
                q4k_ptr(self, host_slice(l.ple_proj_bytes, l.ple_proj_len))?
            };
            let ple_post_norm = f32_ptr(
                self,
                host_slice(l.ple_post_norm_bytes, l.ple_post_norm_len),
                "persistent ple_post_norm",
            )?;
            let ple_input = f32_ptr(
                self,
                host_slice(l.ple_input_bytes, l.ple_input_len),
                "persistent ple_input",
            )?;

            // KV cache: shared-KV layers reuse the anchor layer's device buf.
            // Reusable buffers (k_cache_devs / v_cache_devs) are allocated by
            // ensure_persistent_decode_ctx; here we only upload the host
            // prefill K/V slice for layers that own their buffer.  Only the
            // _new_ token's K/V is written back to host after the launch (see
            // step 9 below), so subsequent dispatches see the host slice
            // already containing this layer's prefill+previous-decode K/V
            // bytes — we still re-upload because the host KVCache may have
            // been mutated by an out-of-band path (e.g. prefill).
            let (k_cache_dev, v_cache_dev) = if !kv_owns[idx] {
                let anchor = l.kv_source_layer as usize;
                (k_cache_devs[anchor], v_cache_devs[anchor])
            } else {
                let k_dev = k_cache_devs[idx];
                let v_dev = v_cache_devs[idx];
                if kv_upload_needed {
                    if let Some(k_host) = host_slice(l.k_cache_bytes, l.k_cache_len) {
                        unsafe {
                            self.api.memcpy_htod_async(
                                k_dev,
                                k_host.as_ptr().cast::<libc::c_void>(),
                                k_host.len(),
                                self.stream,
                            )?;
                        }
                    }
                    if let Some(v_host) = host_slice(l.v_cache_bytes, l.v_cache_len) {
                        unsafe {
                            self.api.memcpy_htod_async(
                                v_dev,
                                v_host.as_ptr().cast::<libc::c_void>(),
                                v_host.len(),
                                self.stream,
                            )?;
                        }
                    }
                }
                (k_dev, v_dev)
            };
            let _ = k_cache_dev;
            let _ = v_cache_dev;

            layer_params.push(PersistentLayerParamsHost {
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                gate_weight,
                up_weight,
                down_weight,
                attn_norm,
                post_attn_norm,
                ffn_norm,
                post_ffn_norm,
                q_norm,
                k_norm,
                ple_gate,
                ple_proj,
                ple_post_norm,
                ple_input,
                k_cache: if kv_owns[idx] {
                    k_cache_devs[idx]
                } else {
                    k_cache_devs[l.kv_source_layer as usize]
                },
                v_cache: if kv_owns[idx] {
                    v_cache_devs[idx]
                } else {
                    v_cache_devs[l.kv_source_layer as usize]
                },
                head_dim: l.head_dim,
                q_dim: l.q_dim,
                kv_dim: l.kv_dim,
                n_ff: l.n_ff,
                sliding_window: l.sliding_window,
                kv_source_layer: l.kv_source_layer,
                layer_output_scale: l.layer_output_scale,
                flags: l.flags,
            });
        }

        // 2) Upload the layer table into the reusable device buffer.  The
        //    table contents change per token (rope_pos, kv_len) but the
        //    storage is reused, eliminating one mem_alloc + mem_free per
        //    decode token.
        let layers_bytes = layer_params.len() * std::mem::size_of::<PersistentLayerParamsHost>();
        let layers_dev = layers_dev_reuse;
        unsafe {
            self.api.memcpy_htod_async(
                layers_dev,
                layer_params.as_ptr().cast::<libc::c_void>(),
                layers_bytes,
                self.stream,
            )?;
        }

        // 3) Scratch buffers — all reusable (lifetime: CudaState).
        let bytes_f32 = |n: usize| n * std::mem::size_of::<f32>();
        let hidden_dev = hidden_dev_reuse;
        let normed_dev = normed_dev_reuse;
        let attn_out_dev = attn_out_dev_reuse;
        let q_buf_dev = q_buf_dev_reuse;
        let k_buf_dev = k_buf_dev_reuse;
        let v_buf_dev = v_buf_dev_reuse;
        let gate_buf_dev = gate_buf_dev_reuse;
        let up_buf_dev = up_buf_dev_reuse;
        let ple_gate_buf_dev = ple_gate_buf_dev_reuse;
        let ffn_normed_dev = ffn_normed_dev_reuse;
        let ffn_down_dev = ffn_down_dev_reuse;
        let attn_normed_dev = attn_normed_dev_reuse;
        let logits_dev = logits_dev_reuse;
        let argmax_dev = argmax_dev_reuse;

        // 4) Upload input hidden state.
        if std::env::var("RNB_CUDA_PERSISTENT_DECODE_TRACE")
            .ok()
            .as_deref()
            == Some("1")
        {
            eprintln!(
                "[cu74 alloc] hidden={hidden_dev:#x} normed={normed_dev:#x} attn_out={attn_out_dev:#x} q={q_buf_dev:#x} k={k_buf_dev:#x} v={v_buf_dev:#x} gate={gate_buf_dev:#x} up={up_buf_dev:#x}"
            );
        }
        unsafe {
            // cu100: upload seq_len * hidden_dim tokens at once. Decode (seq_len=1)
            // is bitwise identical to the previous one-token H2D.
            self.api.memcpy_htod_async(
                hidden_dev,
                request.input_hidden.as_ptr().cast::<libc::c_void>(),
                bytes_f32(seq_len * hidden_dim),
                self.stream,
            )?;
        }

        // 5) Resolve output weight (Q8_0) via the existing Q8_0 resident
        //    cache so the 427 MB upload happens at most once per process,
        //    not once per decode token.
        let output_weight = if !request.output_weight_bytes.is_null()
            && request.output_weight_len > 0
        {
            let bytes = unsafe {
                std::slice::from_raw_parts(request.output_weight_bytes, request.output_weight_len)
            };
            // Q8_0 layout: 34 bytes per 32-element block. rows = vocab,
            // cols = hidden_dim. Derive blocks_per_row from byte length.
            let cols = request.hidden_dim as usize;
            let blocks_per_row = cols / 32;
            let row_bytes = blocks_per_row * 34;
            if row_bytes == 0 || bytes.len() % row_bytes != 0 {
                return Err(format!(
                    "persistent decode output weight bytes {} not a multiple of Q8_0 row_bytes {}",
                    bytes.len(),
                    row_bytes
                ));
            }
            let rows = bytes.len() / row_bytes;
            self.resident_q8_quant_ptr(bytes, rows, cols)?
        } else {
            0
        };

        // cu76 diag: allocate device-side hidden_probe buffer when caller
        // requested it via `request.hidden_probe`.  After launch we D2H copy
        // into the caller-owned host slice.
        let hidden_probe_dev: u64 = if request.hidden_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(hidden_dim))? }
        } else {
            0
        };
        let normed_after_attn_norm_probe_dev: u64 =
            if request.normed_after_attn_norm_probe.is_some() {
                unsafe { self.api.mem_alloc(bytes_f32(hidden_dim))? }
            } else {
                0
            };
        let hidden_after_attn_probe_dev: u64 = if request.hidden_after_attn_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(hidden_dim))? }
        } else {
            0
        };
        let hidden_after_ffn_probe_dev: u64 = if request.hidden_after_ffn_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(hidden_dim))? }
        } else {
            0
        };

        // cu78 fine probes alloc.
        let head_dim_for_probes: usize = 512;
        let attn_out_probe_dev: u64 = if request.attn_out_probe.is_some() {
            let p = unsafe { self.api.mem_alloc(bytes_f32(head_dim_for_probes))? };
            unsafe {
                self.api
                    .memset_d32_async(p, 0, head_dim_for_probes, self.stream)?;
            }
            p
        } else {
            0
        };
        let q_proj_probe_dev: u64 = if request.q_proj_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(head_dim_for_probes))? }
        } else {
            0
        };
        let k_proj_probe_dev: u64 = if request.k_proj_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(head_dim_for_probes))? }
        } else {
            0
        };
        let v_proj_probe_dev: u64 = if request.v_proj_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(head_dim_for_probes))? }
        } else {
            0
        };
        let attn_scores_probe_dev: u64 = if let Some(s) = request.attn_scores_probe.as_ref() {
            unsafe { self.api.mem_alloc(bytes_f32(s.len()))? }
        } else {
            0
        };
        let attn_v_probe_dev: u64 = if request.attn_v_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(head_dim_for_probes))? }
        } else {
            0
        };
        let attn_acc_probe_dev: u64 = if request.attn_acc_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(head_dim_for_probes))? }
        } else {
            0
        };
        let attn_row_sum_probe_dev: u64 = if request.attn_row_sum_probe.is_some() {
            unsafe { self.api.mem_alloc(bytes_f32(1))? }
        } else {
            0
        };
        let alloc_with_clear = |this: &mut Self, count: usize| -> Result<u64, String> {
            let p = unsafe { this.api.mem_alloc(bytes_f32(count))? };
            unsafe {
                this.api.memset_d32_async(p, 0, count, this.stream)?;
            }
            Ok(p)
        };
        let hidden_after_ffn_only_probe_dev = if request.hidden_after_ffn_only_probe.is_some() {
            alloc_with_clear(self, hidden_dim)?
        } else {
            0
        };
        let ffn_gate_probe_dev = if request.ffn_gate_probe.is_some() {
            alloc_with_clear(self, 1024)?
        } else {
            0
        };
        let ffn_gated_probe_dev = if request.ffn_gated_probe.is_some() {
            alloc_with_clear(self, 1024)?
        } else {
            0
        };
        let ffn_down_probe_dev = if request.ffn_down_probe.is_some() {
            alloc_with_clear(self, hidden_dim)?
        } else {
            0
        };
        let layer_hidden_trace_dev = if request.layer_hidden_trace.is_some() {
            alloc_with_clear(self, request.num_layers as usize)?
        } else {
            0
        };
        // cu91: upload output_norm (f32, hidden_dim) via resident f32 cache.
        let output_norm_dev: u64 =
            if !request.output_norm_bytes.is_null() && request.output_norm_len > 0 {
                let bytes = unsafe {
                    std::slice::from_raw_parts(request.output_norm_bytes, request.output_norm_len)
                };
                self.resident_f32_weights_ptr_from_le_bytes(bytes, "persistent output_norm")?
            } else {
                0
            };

        // cu77: upload rope_freqs (f32, head_dim/2 elements = 1024 bytes for
        // Gemma4 head_dim=512).  Goes through the resident f32 cache so it's
        // uploaded once per process (not per token).
        let rope_freqs_dev: u64 =
            if !request.rope_freqs_bytes.is_null() && request.rope_freqs_len > 0 {
                let bytes = unsafe {
                    std::slice::from_raw_parts(request.rope_freqs_bytes, request.rope_freqs_len)
                };
                self.resident_f32_weights_ptr_from_le_bytes(bytes, "persistent rope_freqs")?
            } else {
                0
            };

        // 6) Build the top-level params struct.
        let mut params = PersistentDecodeParamsHost {
            layers_dev,
            num_layers: request.num_layers,
            hidden_dim: request.hidden_dim,
            vocab_size: request.vocab_size,
            norm_eps: request.norm_eps,
            rope_pos: request.rope_pos,
            kv_len: request.kv_len,
            hidden_dev,
            normed_dev,
            attn_out_dev,
            q_buf_dev,
            k_buf_dev,
            v_buf_dev,
            gate_buf_dev,
            up_buf_dev,
            ple_gate_buf_dev,
            output_weight,
            logits_dev,
            argmax_dev,
            hidden_probe_dev,
            normed_after_attn_norm_probe_dev,
            hidden_after_attn_probe_dev,
            hidden_after_ffn_probe_dev,
            rope_freqs_dev,
            attn_out_probe_dev,
            q_proj_probe_dev,
            k_proj_probe_dev,
            v_proj_probe_dev,
            attn_scores_probe_dev,
            attn_v_probe_dev,
            attn_acc_probe_dev,
            attn_row_sum_probe_dev,
            probe_layer_idx: std::env::var("RNB_CUDA_PERSISTENT_PROBE_LAYER")
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0),
            hidden_after_ffn_only_probe_dev,
            ffn_gate_probe_dev,
            ffn_gated_probe_dev,
            ffn_down_probe_dev,
            layer_hidden_trace_dev,
            output_norm_dev,
            nan_trace: if std::env::var("RNB_CUDA_PERSISTENT_NAN_TRACE").is_ok() {
                1
            } else {
                0
            },
            // cu97: V RMS norm (no-scale) per head. Default ON to match eager's
            // `gemma_v_norm_enabled()` (default-true). Disabled by env override.
            gemma_v_norm: if std::env::var("RNB_DISABLE_GEMMA_V_NORM").is_ok() {
                0
            } else {
                1
            },
            // cu100: pass through caller's seq_len. Decode = 1; batch prefill
            // caller = N (kernel body wraps the layer loop in `for __t in seq_len`).
            seq_len: request.seq_len.max(1),
            // cu101 M3: q_buf / attn_out token-slot stride (host alloc sized
            // hidden_slots * q_dim_max). Kernel walks `__t * q_dim_max`.
            q_dim_max: request.q_dim_max,
            // cu102 M4: batch FFN buffers + n_ff stride.
            ffn_normed_dev,
            ffn_down_dev,
            n_ff_max: request.n_ff_max,
            // cu105 QKV batch-tiling: kv_dim_max = k_buf/v_buf token-slot stride;
            // attn_normed_dev = attn_norm output batch slots (Q/K/V proj input).
            kv_dim_max: request.kv_dim_max,
            attn_normed_dev,
        };

        // 7) Launch (single-wave cooperative grid).
        // block_threads = 512 so head_dim=512 attention/QK-norm fit in a
        // single block (lane==head_dim mask covers head_dim<=512).  Q4K
        // cooperative GEMV uses warps_per_block-derived row_stride so this
        // does not create row-write overlap across blocks.
        let block_threads: u32 = 512;
        // cu103 M4: batch FFN's input-smem tiled GEMM (persistent_q4k_gemm_smem)
        // needs BN_TILE(4) × hidden × f32 of dynamic smem for the gate/up input
        // tile. Decode (seq_len=1) uses the GEMV fallback (no smem staging), so
        // keep its launch at the small attention smem to preserve occupancy.
        let attn_smem: u32 = 512 * std::mem::size_of::<f32>() as u32;
        let shared_mem_bytes: u32 = if params.seq_len > 1 {
            const BN_TILE: u32 = 4;
            (BN_TILE * params.hidden_dim * std::mem::size_of::<f32>() as u32).max(attn_smem)
        } else {
            attn_smem
        };
        let (sm_count, max_blocks_per_sm) =
            self.persistent_decode_m1_occupancy(block_threads as i32, shared_mem_bytes as usize)?;
        if max_blocks_per_sm <= 0 {
            return Err(format!(
                "persistent decode kernel reports 0 active blocks/SM (register pressure?) sm_count={sm_count}"
            ));
        }
        let grid_x = (sm_count as u32) * (max_blocks_per_sm as u32);
        self.launch_persistent_decode_m1(
            &mut params,
            (grid_x, 1, 1),
            (block_threads, 1, 1),
            shared_mem_bytes,
        )?;

        // cu74 NaN-isolation: optionally copy hidden_dev back to inspect
        // the post-layer-loop hidden state before the output projection.
        let probe_hidden = std::env::var("RNB_CUDA_PERSISTENT_DECODE_PROBE_HIDDEN")
            .ok()
            .as_deref()
            == Some("1");
        if probe_hidden {
            let mut hidden_host = vec![0.0f32; hidden_dim];
            unsafe {
                self.api.memcpy_dtoh_async(
                    hidden_host.as_mut_ptr().cast::<libc::c_void>(),
                    hidden_dev,
                    bytes_f32(hidden_dim),
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            let mut nan = 0usize;
            let mut neginf = 0usize;
            let mut posinf = 0usize;
            let mut hi = f32::NEG_INFINITY;
            let mut lo = f32::INFINITY;
            for &v in &hidden_host {
                if v.is_nan() {
                    nan += 1;
                } else if v == f32::NEG_INFINITY {
                    neginf += 1;
                } else if v == f32::INFINITY {
                    posinf += 1;
                } else {
                    if v > hi {
                        hi = v;
                    }
                    if v < lo {
                        lo = v;
                    }
                }
            }
            eprintln!(
                "[cu74 hidden-probe] post-loop hidden: nan={nan} +inf={posinf} -inf={neginf} range=[{lo:.4}, {hi:.4}] first5=[{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
                hidden_host[0], hidden_host[1], hidden_host[2], hidden_host[3], hidden_host[4],
            );
        }

        // 8) Copy logits + argmax back to host.
        unsafe {
            self.api.memcpy_dtoh_async(
                request.output_logits.as_mut_ptr().cast::<libc::c_void>(),
                logits_dev,
                bytes_f32(vocab_size),
                self.stream,
            )?;
            let argmax_ptr = request.argmax_out as *mut i32;
            self.api.memcpy_dtoh_async(
                argmax_ptr.cast::<libc::c_void>(),
                argmax_dev,
                std::mem::size_of::<i32>(),
                self.stream,
            )?;
        }
        // cu76 diag: D2H probes into caller buffers.
        let probe_pairs: [(u64, Option<&mut &mut [f32]>); 17] = [
            (hidden_probe_dev, request.hidden_probe.as_mut()),
            (
                normed_after_attn_norm_probe_dev,
                request.normed_after_attn_norm_probe.as_mut(),
            ),
            (
                hidden_after_attn_probe_dev,
                request.hidden_after_attn_probe.as_mut(),
            ),
            (
                hidden_after_ffn_probe_dev,
                request.hidden_after_ffn_probe.as_mut(),
            ),
            (attn_out_probe_dev, request.attn_out_probe.as_mut()),
            (q_proj_probe_dev, request.q_proj_probe.as_mut()),
            (k_proj_probe_dev, request.k_proj_probe.as_mut()),
            (v_proj_probe_dev, request.v_proj_probe.as_mut()),
            (attn_scores_probe_dev, request.attn_scores_probe.as_mut()),
            (attn_v_probe_dev, request.attn_v_probe.as_mut()),
            (attn_acc_probe_dev, request.attn_acc_probe.as_mut()),
            (attn_row_sum_probe_dev, request.attn_row_sum_probe.as_mut()),
            (
                hidden_after_ffn_only_probe_dev,
                request.hidden_after_ffn_only_probe.as_mut(),
            ),
            (ffn_gate_probe_dev, request.ffn_gate_probe.as_mut()),
            (ffn_gated_probe_dev, request.ffn_gated_probe.as_mut()),
            (ffn_down_probe_dev, request.ffn_down_probe.as_mut()),
            (layer_hidden_trace_dev, request.layer_hidden_trace.as_mut()),
        ];
        for (dev, host_opt) in probe_pairs {
            if dev == 0 {
                continue;
            }
            if let Some(host_slice) = host_opt {
                let copy_len = host_slice.len();
                if copy_len > 0 {
                    unsafe {
                        self.api.memcpy_dtoh_async(
                            host_slice.as_mut_ptr().cast::<libc::c_void>(),
                            dev,
                            bytes_f32(copy_len),
                            self.stream,
                        )?;
                    }
                }
            }
        }
        self.stream_synchronize()?;
        for dev in [
            hidden_probe_dev,
            normed_after_attn_norm_probe_dev,
            hidden_after_attn_probe_dev,
            hidden_after_ffn_probe_dev,
            attn_out_probe_dev,
            q_proj_probe_dev,
            k_proj_probe_dev,
            v_proj_probe_dev,
            attn_scores_probe_dev,
            attn_v_probe_dev,
            attn_acc_probe_dev,
            attn_row_sum_probe_dev,
            hidden_after_ffn_only_probe_dev,
            ffn_gate_probe_dev,
            ffn_gated_probe_dev,
            ffn_down_probe_dev,
            layer_hidden_trace_dev,
        ] {
            if dev != 0 {
                unsafe {
                    self.api.mem_free(dev)?;
                }
            }
        }

        // 9) Write the new token's K/V back to the host KV cache (only the
        //    layers that own their KV buffer; shared layers do not allocate).
        // cu101: env-gated skip — host KV writeback is only needed when a
        // non-persistent caller (eager prefill, chain function decode, sampler
        // dump) reads the host slice. The persistent prefill token loop
        // re-uses the device KV directly (kv_upload_needed=false via
        // last_rope_pos guard), so the D2H sync wait is pure overhead.
        // nsys (cu99) showed D2H API time = 44% of wall (1212 ms / 2750 ms),
        // dominated by pageable host writeback sync.
        let skip_host_writeback =
            std::env::var("RNB_CUDA_PERSISTENT_SKIP_HOST_KV_WRITEBACK").is_ok();
        // cu104: batch prefill writes seq_len tokens, not just one. D2H all
        // of [rope_pos..rope_pos + seq_len) so the host KV cache reflects every
        // token the kernel wrote. Decode (seq_len=1) is bitwise identical to
        // the previous single-token writeback.
        let writeback_count = request.seq_len.max(1) as usize;
        for (idx, l) in request.layers.iter().enumerate() {
            if !kv_owns[idx] || skip_host_writeback {
                continue;
            }
            let token_bytes = (l.kv_dim as usize) * std::mem::size_of::<u16>();
            let offset = (request.rope_pos as usize) * token_bytes;
            let copy_bytes = writeback_count * token_bytes;
            if l.k_cache_bytes.is_null() || l.v_cache_bytes.is_null() {
                continue;
            }
            unsafe {
                let dst_k = (l.k_cache_bytes as *mut u8).add(offset);
                let dst_v = (l.v_cache_bytes as *mut u8).add(offset);
                self.api.memcpy_dtoh_async(
                    dst_k.cast::<libc::c_void>(),
                    k_cache_devs[idx] + offset as u64,
                    copy_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    dst_v.cast::<libc::c_void>(),
                    v_cache_devs[idx] + offset as u64,
                    copy_bytes,
                    self.stream,
                )?;
            }
        }
        self.stream_synchronize()?;

        let resident_kv_tokens =
            (request.rope_pos as usize).saturating_add(request.seq_len.max(1) as usize);
        if let Some(ctx) = self.persistent_decode_ctx.as_mut() {
            ctx.last_rope_pos = Some(request.rope_pos);
            ctx.resident_kv_tokens = resident_kv_tokens;
        }

        // 10) No cleanup needed — all reusable buffers live in
        //     `self.persistent_decode_ctx` and are released only when the
        //     CudaState is dropped (or when a shape mismatch triggers a
        //     realloc in `ensure_persistent_decode_ctx`).  Weights remain in
        //     their resident caches.

        Ok(())
    }

    /// Pinned Q4K residency entry-point exposed for the persistent decode
    /// path.  Delegates to the existing pinned resident cache so weights
    /// shared with the eager dispatch stay in one location.
    fn resident_q4k_weights_ptr_pinned_for_persistent(&self, bytes: &[u8]) -> Result<u64, String> {
        // SAFETY: `&self` view; cast through *mut to reuse the existing
        // pinned cache helper which mutates internal LRU bookkeeping.
        // This mirrors the pattern used elsewhere in the runtime to keep
        // the cache call site signature consistent.
        let me = self as *const Self as *mut Self;
        unsafe { (*me).resident_q4k_weights_ptr_pinned(bytes) }
    }
}

/// Smoke test: allocate dummy device buffers, populate a minimal
/// `PersistentDecodeParamsHost`, and launch the M2 kernel with `num_layers`
/// layers and a single-wave cooperative grid.  Verifies that the cooperative
/// launch + grid.sync() topology runs to completion (no deadlock, no
/// CUDA_ERROR_INVALID_VALUE on grid clamp, no register-pressure blowout).
///
/// `weights_bytes_per_proj` controls how many bytes the dummy Q/K/V/etc.
/// weight buffers point to; the kernel reads at most `q_dim * blocks_per_row
/// * 144` bytes for the projection it touches, so the smoke uses small dims
/// (`hidden_dim = 1536`, `q_dim/kv_dim = 256`) to keep the buffer tiny.
///
/// Returns the elapsed wall-clock duration of the cooperative launch +
/// stream_synchronize.  Callers compare this against the eager-dispatch path
/// to size the dispatch-gap removal opportunity.
#[cfg(test)]
pub(crate) fn persistent_decode_cooperative_smoke(
    num_layers: u32,
) -> Result<std::time::Duration, String> {
    let mut state = CudaState::open()?;
    state.set_current()?;

    // Tiny dummy dims so the smoke runs in milliseconds even on a busy GPU.
    let hidden_dim: u32 = 1536;
    let q_dim: u32 = 256;
    let kv_dim: u32 = 256;
    let blocks_per_row: u32 = hidden_dim / 256;
    let row_bytes: usize = blocks_per_row as usize * 144;
    let q_weight_bytes = q_dim as usize * row_bytes;
    let kv_weight_bytes = kv_dim as usize * row_bytes;

    // Dummy weight buffers — content is irrelevant for the smoke; we only
    // care that the kernel can read without faulting and that grid.sync()
    // walks the full layer loop.  Q4K weight format means each row contains
    // 144 bytes per 256-element block; tiny dims keep this <100 KB.
    let zero_layer_weight = unsafe { state.api.mem_alloc(q_weight_bytes.max(kv_weight_bytes))? };
    let zero_norm = unsafe { state.api.mem_alloc(hidden_dim as usize * 4)? };
    let hidden = unsafe { state.api.mem_alloc(hidden_dim as usize * 4)? };
    let normed = unsafe { state.api.mem_alloc(hidden_dim as usize * 4)? };
    let q_buf = unsafe { state.api.mem_alloc(q_dim as usize * 4)? };
    let k_buf = unsafe { state.api.mem_alloc(kv_dim as usize * 4)? };
    let v_buf = unsafe { state.api.mem_alloc(kv_dim as usize * 4)? };

    let layer = PersistentLayerParamsHost {
        q_weight: zero_layer_weight,
        k_weight: zero_layer_weight,
        v_weight: zero_layer_weight,
        o_weight: 0,
        gate_weight: 0,
        up_weight: 0,
        down_weight: 0,
        attn_norm: zero_norm,
        post_attn_norm: 0,
        ffn_norm: 0,
        post_ffn_norm: 0,
        q_norm: 0,
        k_norm: 0,
        ple_gate: 0,
        ple_proj: 0,
        ple_post_norm: 0,
        ple_input: 0,
        k_cache: 0,
        v_cache: 0,
        head_dim: 256,
        q_dim,
        kv_dim,
        n_ff: 6144,
        sliding_window: 0,
        kv_source_layer: 0,
        layer_output_scale: 1.0,
        flags: 0,
    };

    let layer_count = num_layers as usize;
    let layers_host = vec![layer; layer_count];
    let layers_bytes = layer_count * std::mem::size_of::<PersistentLayerParamsHost>();
    let layers_dev = unsafe { state.api.mem_alloc(layers_bytes)? };
    unsafe {
        state.api.memcpy_htod_async(
            layers_dev,
            layers_host.as_ptr().cast::<libc::c_void>(),
            layers_bytes,
            state.stream,
        )?;
    }

    let mut params = PersistentDecodeParamsHost {
        layers_dev,
        num_layers,
        hidden_dim,
        vocab_size: 0,
        norm_eps: 1.0e-6,
        rope_pos: 0,
        kv_len: 1,
        hidden_dev: hidden,
        normed_dev: normed,
        attn_out_dev: 0,
        q_buf_dev: q_buf,
        k_buf_dev: k_buf,
        v_buf_dev: v_buf,
        gate_buf_dev: 0,
        up_buf_dev: 0,
        ple_gate_buf_dev: 0,
        output_weight: 0,
        logits_dev: 0,
        argmax_dev: 0,
        hidden_probe_dev: 0,
        normed_after_attn_norm_probe_dev: 0,
        hidden_after_attn_probe_dev: 0,
        hidden_after_ffn_probe_dev: 0,
        rope_freqs_dev: 0,
        attn_out_probe_dev: 0,
        q_proj_probe_dev: 0,
        k_proj_probe_dev: 0,
        v_proj_probe_dev: 0,
        attn_scores_probe_dev: 0,
        attn_v_probe_dev: 0,
        attn_acc_probe_dev: 0,
        attn_row_sum_probe_dev: 0,
        probe_layer_idx: 0,
        hidden_after_ffn_only_probe_dev: 0,
        ffn_gate_probe_dev: 0,
        ffn_gated_probe_dev: 0,
        ffn_down_probe_dev: 0,
        layer_hidden_trace_dev: 0,
        output_norm_dev: 0,
        nan_trace: 0,
        gemma_v_norm: 0,
        seq_len: 1,
        q_dim_max: 1,
        ffn_normed_dev: 0,
        ffn_down_dev: 0,
        n_ff_max: 1,
        kv_dim_max: 1,
        attn_normed_dev: 0,
    };

    // Clamp grid to a single SM-wave so grid.sync() can make progress.
    let block_threads: u32 = 256;
    let shared_mem_bytes: u32 = 32 * std::mem::size_of::<f32>() as u32; // ≤ 32 warps.
    let (sm_count, max_blocks_per_sm) =
        state.persistent_decode_m1_occupancy(block_threads as i32, shared_mem_bytes as usize)?;
    if max_blocks_per_sm <= 0 {
        return Err(format!(
            "persistent decode kernel reports 0 active blocks/SM (register pressure?), sm_count={sm_count}"
        ));
    }
    let grid_x = (sm_count as u32) * (max_blocks_per_sm as u32);

    let start = std::time::Instant::now();
    state.launch_persistent_decode_m1(
        &mut params,
        (grid_x, 1, 1),
        (block_threads, 1, 1),
        shared_mem_bytes,
    )?;
    state.stream_synchronize()?;
    let elapsed = start.elapsed();

    // Cleanup.
    unsafe {
        state.api.mem_free(layers_dev)?;
        state.api.mem_free(zero_layer_weight)?;
        state.api.mem_free(zero_norm)?;
        state.api.mem_free(hidden)?;
        state.api.mem_free(normed)?;
        state.api.mem_free(q_buf)?;
        state.api.mem_free(k_buf)?;
        state.api.mem_free(v_buf)?;
    }
    Ok(elapsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn cooperative_smoke_launches_without_deadlock() {
        // Requires an actual GPU; bail with `ignored` if CUDA isn't available
        // (CI doesn't have a GPU but the maintainer's local 3080 does).
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            eprintln!("[skip] CUDA_VISIBLE_DEVICES unset; skipping cooperative smoke");
            return;
        }
        // Print kernel occupancy so the wiki can record the active-blocks/SM
        // value separately from the wall-clock measurement.
        let mut state = CudaState::open().expect("CudaState::open");
        let (sm_count, max_blocks_per_sm) = state
            .persistent_decode_m1_occupancy(256, 32 * 4)
            .expect("occupancy probe");
        drop(state);
        eprintln!(
            "[cu72] occupancy: sm_count={sm_count}, max_active_blocks/SM={max_blocks_per_sm}, total_blocks={}",
            sm_count * max_blocks_per_sm
        );

        // 35 layers to match Gemma4 E2B's depth. Verifies the cooperative
        // kernel survives the full layer count with grid.sync() between each
        // phase without deadlocking or returning CUDA_ERROR_INVALID_VALUE.
        let elapsed = persistent_decode_cooperative_smoke(35)
            .expect("cooperative smoke launch should succeed");
        eprintln!("[cu72] cooperative smoke elapsed: {:?}", elapsed);
        assert!(elapsed.as_millis() < 5_000, "smoke took >5s; likely a hang");
    }

    #[test]
    fn host_layer_params_layout_matches_cuh() {
        // The CUDA-side struct (persistent_decode.cuh::PersistentLayerParams)
        // has 19 u64 pointer fields + 6 u32 dimension fields + 1 f32 scale +
        // 1 u32 flag. On a packed struct that's 19*8 + 7*4 + 4 = 184 bytes.
        // Both ends are #[repr(C)] / default C layout so alignment matches.
        let expected = 19 * size_of::<u64>() + 7 * size_of::<u32>() + size_of::<f32>();
        assert_eq!(
            size_of::<PersistentLayerParamsHost>(),
            expected,
            "PersistentLayerParamsHost size drifted from the .cuh layout"
        );
    }

    #[test]
    fn host_decode_params_layout_is_stable() {
        // Keep the host mirror aligned with persistent_decode.cuh: 35 pointer
        // fields, 12 u32 fields, and one f32 field, rounded to pointer alignment.
        let raw = 35 * size_of::<u64>() + 12 * size_of::<u32>() + size_of::<f32>();
        let expected = raw.next_multiple_of(align_of::<PersistentDecodeParamsHost>());
        assert_eq!(
            size_of::<PersistentDecodeParamsHost>(),
            expected,
            "PersistentDecodeParamsHost size drifted from the .cuh layout"
        );
    }
}
