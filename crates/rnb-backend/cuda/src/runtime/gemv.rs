use super::*;

pub(super) struct Q4kF16QkvHd256PostprocessDevice {
    pub(super) hidden_dev: Option<u64>,
    pub(super) q_post_dev: u64,
    pub(super) k_bits_dev: u64,
    pub(super) v_bits_dev: u64,
    pub(super) attn_out_scratch_dev: u64,
    pub(super) k_f32_scratch_dev: u64,
    pub(super) v_f32_scratch_dev: u64,
    pub(super) q_len: usize,
    pub(super) kv_len: usize,
    pub(super) q_bytes: usize,
    pub(super) kv_f16_bytes: usize,
}

pub(super) struct Q4kF16QkvHd512AttentionDevice {
    pub(super) hidden_dev: Option<u64>,
    pub(super) attn_out_dev: u64,
    pub(super) k_bits_dev: u64,
    pub(super) v_bits_dev: u64,
    pub(super) q_len: usize,
    pub(super) kv_len: usize,
    pub(super) q_bytes: usize,
    pub(super) kv_f16_bytes: usize,
}

#[derive(Debug)]
pub(super) enum Q4kF16DenseChainOutput {
    Host,
    Device(rnb_backend_api::DeviceTensorId),
}

pub(super) enum Q4kF16QkvInput<'a> {
    Normed(&'a [f32]),
    Hidden {
        values: &'a [f32],
        attn_norm: &'a [f32],
        unit_offset: bool,
    },
    DeviceHidden {
        id: rnb_backend_api::DeviceTensorId,
        desc: rnb_backend_api::DeviceTensorDesc,
        attn_norm: &'a [f32],
        unit_offset: bool,
    },
}

impl Q4kF16QkvInput<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Normed(values) | Self::Hidden { values, .. } => values.len(),
            Self::DeviceHidden { desc, .. } => desc.rows().saturating_mul(desc.cols()),
        }
    }

    fn is_hidden(&self) -> bool {
        matches!(self, Self::Hidden { .. } | Self::DeviceHidden { .. })
    }
}

fn q4k_q8dot_decode_enabled(default: bool) -> bool {
    match std::env::var("RNB_CUDA_Q4K_GEMV_Q8DOT") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

fn q4k_q8dot_prefill_enabled(default: bool) -> bool {
    match std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

fn q4k_q8dot_prefill_seq4_enabled(default: bool) -> bool {
    match std::env::var("RNB_CUDA_Q4K_BATCH_Q8DOT_SEQ4") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => default,
    }
}

fn q8_0_prefill_quant_cache_enabled() -> bool {
    match std::env::var("RNB_CUDA_Q8_0_PREFILL_QUANT_CACHE") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => false,
    }
}

fn q8_0_prefill_pinned_dtoh_enabled() -> bool {
    match std::env::var("RNB_CUDA_Q8_0_PREFILL_PINNED_DTOH") {
        Ok(value) => {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        }
        Err(_) => true,
    }
}

fn quantize_q8_1_by_32(input: &[f32], blocks_per_row: usize) -> (Vec<i8>, Vec<f32>) {
    let mut qs = vec![0i8; blocks_per_row * 256];
    let mut ds = vec![0.0f32; blocks_per_row * 8];
    for b in 0..blocks_per_row {
        for j in 0..8 {
            let off = b * 256 + j * 32;
            let chunk = &input[off..off + 32];
            let max_abs = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            if max_abs == 0.0 {
                continue;
            }
            let d = max_abs / 127.0;
            let inv_d = 1.0 / d;
            ds[b * 8 + j] = d;
            for (idx, &value) in chunk.iter().enumerate() {
                qs[off + idx] = (value * inv_d).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (qs, ds)
}

pub(in crate::runtime) fn quantize_q8_1_batch_by_32(
    input: &[f32],
    blocks_per_row: usize,
    seq_len: usize,
) -> (Vec<i8>, Vec<f32>) {
    let cols = blocks_per_row * 256;
    let mut qs = vec![0i8; seq_len * cols];
    let mut ds = vec![0.0f32; seq_len * blocks_per_row * 8];
    for seq in 0..seq_len {
        let input_base = seq * cols;
        let ds_base = seq * blocks_per_row * 8;
        for b in 0..blocks_per_row {
            for j in 0..8 {
                let off = input_base + b * 256 + j * 32;
                let chunk = &input[off..off + 32];
                let max_abs = chunk.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
                if max_abs == 0.0 {
                    continue;
                }
                let d = max_abs / 127.0;
                let inv_d = 1.0 / d;
                ds[ds_base + b * 8 + j] = d;
                for (idx, &value) in chunk.iter().enumerate() {
                    qs[off + idx] = (value * inv_d).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }
    (qs, ds)
}

impl CudaState {
    pub(super) fn q4k_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        if q4k_q8dot_decode_enabled(rows >= 1024 && blocks_per_row >= 4) {
            let (qs, ds) = quantize_q8_1_by_32(input, blocks_per_row);
            let qs_dev = self.compute_input_ptr(qs.len())?;
            let ds_dev = self.compute_aux_output_ptr(std::mem::size_of_val(ds.as_slice()))?;
            unsafe {
                self.api.memcpy_htod_async(
                    qs_dev,
                    qs.as_ptr().cast::<libc::c_void>(),
                    qs.len(),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    ds_dev,
                    ds.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(ds.as_slice()),
                    self.stream,
                )?;
            }
            self.launch_q4k_gemv_q8dot_to_dev(
                weights,
                rows,
                blocks_per_row,
                qs_dev,
                ds_dev,
                output_dev,
            )?;
        } else {
            unsafe {
                self.api.memcpy_htod_async(
                    input_dev,
                    input.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(input),
                    self.stream,
                )?;
            }
            let mut output_arg = output_dev;
            let mut weights_arg = weights_dev;
            let mut input_arg = input_dev;
            let mut rows_arg = rows as u32;
            let mut blocks_per_row_arg = blocks_per_row as u32;
            let (kernel, grid_x) = if tuning::q4k_gemv_warp8_enabled() {
                ("rnb_q4k_gemv_warp8", rows.div_ceil(8) as u32)
            } else {
                ("rnb_q4k_gemv", rows as u32)
            };
            self.launch_cached_gemv(
                kernel,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (grid_x, 1, 1),
                (256, 1, 1),
            )?;
        }
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn q4k_gemv_into(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        self.q4k_gemv_into_with_touch(weights, rows, blocks_per_row, input, output, false)
    }

    // cu42 step 9: 단일 q4k_gemv 의 device input variant. carrier 가 input 으로
    // 직접 사용. RMS norm 의 H2D 와 chain. output 은 host (D2H + sync).
    // q8dot path 안 씀 (host CPU quantize 필요) — `rnb_q4k_gemv_warp8` fallback.
    // cu45 step 23: Q6K device input gemv (Gemma4 V weight). Q4K 와 동일 패턴.
    // host scratch.norm_buf D2H 제거 위한 device input variant.
    pub(super) fn q6k_gemv_with_device_input(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output: &mut [f32],
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let output_bytes = std::mem::size_of_val(output);
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp8_output = tuning::q6k_output_warp8_enabled();
        let (kernel, grid_x) = if warp8_output {
            ("rnb_q6k_gemv_warp8", rows.div_ceil(8) as u32)
        } else {
            ("rnb_q6k_gemv", rows as u32)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
        )?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    // cu63: sync-free device-to-device variant. caller 가 output_dev 를 제공.
    // D2H + stream_synchronize 없음 — 34 sync points per decode token 제거 목표.
    pub(super) fn q6k_gemv_device_to_device(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp8_output = tuning::q6k_output_warp8_enabled();
        let (kernel, grid_x) = if warp8_output {
            ("rnb_q6k_gemv_warp8", rows.div_ceil(8) as u32)
        } else {
            ("rnb_q6k_gemv", rows as u32)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
        )
    }

    pub(super) fn q4k_gemv_with_device_input(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output: &mut [f32],
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let output_bytes = std::mem::size_of_val(output);
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let (kernel, grid_x) = if tuning::q4k_gemv_warp8_enabled() {
            ("rnb_q4k_gemv_warp8", rows.div_ceil(8) as u32)
        } else {
            ("rnb_q4k_gemv", rows as u32)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
        )?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    // cu63: sync-free device-to-device variant. caller 가 output_dev 를 제공.
    // D2H + stream_synchronize 없음 — 34 sync points per decode token 제거 목표.
    pub(super) fn q4k_gemv_device_to_device(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input_dev: u64,
        output_dev: u64,
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let (kernel, grid_x) = if tuning::q4k_gemv_warp8_enabled() {
            ("rnb_q4k_gemv_warp8", rows.div_ceil(8) as u32)
        } else {
            ("rnb_q4k_gemv", rows as u32)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
        )
    }

    pub(super) fn q4k_gemv_into_touch_hit(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        self.q4k_gemv_into_with_touch(weights, rows, blocks_per_row, input, output, true)
    }

    fn q4k_gemv_into_with_touch(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
        output: &mut [f32],
        touch_hit: bool,
    ) -> Result<(), String> {
        let weights_dev = if touch_hit {
            self.resident_q4k_weights_ptr_touch_hit(weights)?
        } else {
            self.resident_q4k_weights_ptr(weights)?
        };
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = std::mem::size_of_val(output);
        let output_dev = self.compute_output_ptr(output_bytes)?;
        if q4k_q8dot_decode_enabled(rows >= 1024 && blocks_per_row >= 4) {
            let (qs, ds) = quantize_q8_1_by_32(input, blocks_per_row);
            let qs_dev = self.compute_input_ptr(qs.len())?;
            let ds_dev = self.compute_aux_output_ptr(std::mem::size_of_val(ds.as_slice()))?;
            unsafe {
                self.api.memcpy_htod_async(
                    qs_dev,
                    qs.as_ptr().cast::<libc::c_void>(),
                    qs.len(),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    ds_dev,
                    ds.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(ds.as_slice()),
                    self.stream,
                )?;
            }
            self.launch_q4k_gemv_q8dot_to_dev(
                weights,
                rows,
                blocks_per_row,
                qs_dev,
                ds_dev,
                output_dev,
            )?;
        } else {
            unsafe {
                self.api.memcpy_htod_async(
                    input_dev,
                    input.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(input),
                    self.stream,
                )?;
            }
            let mut output_arg = output_dev;
            let mut weights_arg = weights_dev;
            let mut input_arg = input_dev;
            let mut rows_arg = rows as u32;
            let mut blocks_per_row_arg = blocks_per_row as u32;
            let (kernel, grid_x) = if tuning::q4k_gemv_warp8_enabled() {
                ("rnb_q4k_gemv_warp8", rows.div_ceil(8) as u32)
            } else {
                ("rnb_q4k_gemv", rows as u32)
            };
            self.launch_cached_gemv(
                kernel,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (grid_x, 1, 1),
                (256, 1, 1),
            )?;
        }
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    pub(super) fn q5_basic_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
        kernel: &'static str,
        prefer_q8_quant_cache: bool,
    ) -> Result<Vec<f32>, String> {
        let cols = blocks_per_row * 32;
        let weights_dev = if prefer_q8_quant_cache {
            match self.resident_q8_quant.get(&q8_f32_key(weights, rows, cols)) {
                Some(entry) => entry.ptr,
                None => self.resident_q4k_weights_ptr(weights)?,
            }
        } else {
            self.resident_q4k_weights_ptr(weights)?
        };
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let (kernel, grid, block) =
            if kernel == "rnb_q8_0_gemv" && tuning::q8_0_gemv_warp8_enabled() {
                (
                    "rnb_q8_0_gemv_warp8",
                    (rows.div_ceil(8) as u32, 1, 1),
                    (256, 1, 1),
                )
            } else if kernel == "rnb_q8_0_gemv" && tuning::q8_0_gemv_warp4_enabled() {
                (
                    "rnb_q8_0_gemv_warp4",
                    (rows.div_ceil(4) as u32, 1, 1),
                    (32, 4, 1),
                )
            } else {
                (kernel, (rows as u32, 1, 1), (256, 1, 1))
            };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )?;
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn iq4_xs_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_iq4_xs_gemv_warp8",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows.div_ceil(8) as u32, 1, 1),
            (256, 1, 1),
        )?;
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn q5_basic_gemv_batch(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
        kernel: &'static str,
    ) -> Result<Vec<f32>, String> {
        if kernel == "rnb_q5_0_gemv_batch"
            && seq_len <= 32
            && std::env::var("RNB_CUDA_Q5_0_BATCH_SEQ32").ok().as_deref() == Some("1")
        {
            return self.gemv_batch_seq32(
                "rnb_q5_0_gemv_batch_seq32",
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input,
            );
        }
        self.gemv_batch(kernel, weights, rows, blocks_per_row, seq_len, input)
    }

    pub(super) fn q6k_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp8_output = tuning::q6k_output_warp8_enabled();
        let (kernel, grid_x) = if warp8_output {
            ("rnb_q6k_gemv_warp8", rows.div_ceil(8) as u32)
        } else {
            ("rnb_q6k_gemv", rows as u32)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
        )?;
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn q6k_gemv_into(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        self.q6k_gemv_into_with_touch(weights, rows, blocks_per_row, input, output, false)
    }

    pub(super) fn q6k_gemv_into_touch_hit(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        self.q6k_gemv_into_with_touch(weights, rows, blocks_per_row, input, output, true)
    }

    fn q6k_gemv_into_with_touch(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
        output: &mut [f32],
        touch_hit: bool,
    ) -> Result<(), String> {
        let weights_dev = if touch_hit {
            self.resident_q4k_weights_ptr_touch_hit(weights)?
        } else {
            self.resident_q4k_weights_ptr(weights)?
        };
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = std::mem::size_of_val(output);
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let warp8_output = tuning::q6k_output_warp8_enabled();
        let (kernel, grid_x) = if warp8_output {
            ("rnb_q6k_gemv_warp8", rows.div_ceil(8) as u32)
        } else {
            ("rnb_q6k_gemv", rows as u32)
        };
        self.launch_cached_gemv(
            kernel,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (grid_x, 1, 1),
            (256, 1, 1),
        )?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    pub(super) fn q6k_gemv_argmax(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<(u32, f32), String> {
        if std::env::var("RNB_CUDA_Q6K_FUSED_ARGMAX").ok().as_deref() != Some("0") {
            return self.q6k_gemv_argmax_fused_warp8(weights, rows, blocks_per_row, input);
        }
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_gemv",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )?;

        let blocks = 256usize.min(rows.max(1).div_ceil(256));
        let values_dev = self.compute_mid_a_ptr(blocks * std::mem::size_of::<f32>())?;
        let indices_dev = self.compute_mid_b_ptr(blocks * std::mem::size_of::<u32>())?;
        let mut values_arg = output_dev;
        let mut block_values_arg = values_dev;
        let mut block_indices_arg = indices_dev;
        let mut len_arg = rows as u32;
        self.launch_cached_gemv(
            "rnb_argmax_f32",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (blocks as u32, 1, 1),
            (256, 1, 1),
        )?;

        let mut block_values = vec![0.0f32; blocks];
        let mut block_indices = vec![0u32; blocks];
        unsafe {
            self.api.memcpy_dtoh_async(
                block_values.as_mut_ptr().cast::<libc::c_void>(),
                values_dev,
                blocks * std::mem::size_of::<f32>(),
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                block_indices.as_mut_ptr().cast::<libc::c_void>(),
                indices_dev,
                blocks * std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;

        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (&value, &idx) in block_values.iter().zip(block_indices.iter()) {
            if value > best_val || (value == best_val && idx < best_idx) {
                best_val = value;
                best_idx = idx;
            }
        }
        Ok((best_idx, best_val))
    }

    pub(super) fn q8_0_gemv_argmax(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<(u32, f32), String> {
        let cols = blocks_per_row * 32;
        let weights_dev = match self.resident_q8_quant.get(&q8_f32_key(weights, rows, cols)) {
            Some(entry) => entry.ptr,
            None => self.resident_q4k_weights_ptr(weights)?,
        };
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }

        let blocks = rows.div_ceil(8);
        let values_dev = self.compute_mid_a_ptr(blocks * std::mem::size_of::<f32>())?;
        let indices_dev = self.compute_mid_b_ptr(blocks * std::mem::size_of::<u32>())?;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut block_values_arg = values_dev;
        let mut block_indices_arg = indices_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q8_0_gemv_argmax_warp8",
            &[
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (blocks as u32, 1, 1),
            (256, 1, 1),
        )?;

        let reduce_blocks = 256usize.min(blocks.max(1).div_ceil(256));
        let reduce_values_dev =
            self.compute_full_up_ptr(reduce_blocks * std::mem::size_of::<f32>())?;
        let reduce_indices_dev =
            self.compute_full_down_ptr(reduce_blocks * std::mem::size_of::<u32>())?;
        let mut values_arg = values_dev;
        let mut indices_arg = indices_dev;
        let mut block_values_arg = reduce_values_dev;
        let mut block_indices_arg = reduce_indices_dev;
        let mut len_arg = blocks as u32;
        self.launch_cached_gemv(
            "rnb_argmax_pairs_f32",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (reduce_blocks as u32, 1, 1),
            (256, 1, 1),
        )?;

        let mut block_values = vec![0.0f32; reduce_blocks];
        let mut block_indices = vec![0u32; reduce_blocks];
        unsafe {
            self.api.memcpy_dtoh_async(
                block_values.as_mut_ptr().cast::<libc::c_void>(),
                reduce_values_dev,
                reduce_blocks * std::mem::size_of::<f32>(),
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                block_indices.as_mut_ptr().cast::<libc::c_void>(),
                reduce_indices_dev,
                reduce_blocks * std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;

        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (&value, &idx) in block_values.iter().zip(block_indices.iter()) {
            if value > best_val || (value == best_val && idx < best_idx) {
                best_val = value;
                best_idx = idx;
            }
        }
        Ok((best_idx, best_val))
    }

    pub(super) fn q8_0_gemv_argmax_q8dot(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<(u32, f32), String> {
        let cols = blocks_per_row * 32;
        let weights_dev = match self.resident_q8_quant.get(&q8_f32_key(weights, rows, cols)) {
            Some(entry) => entry.ptr,
            None => self.resident_q4k_weights_ptr(weights)?,
        };
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }

        let input_qs_dev = self.compute_aux_output_ptr(input.len())?;
        let input_ds_dev =
            self.compute_full_gate_ptr(blocks_per_row * std::mem::size_of::<f32>())?;
        self.launch_quantize_q8_1_by_32(input_dev, input_qs_dev, input_ds_dev, input.len())?;

        let blocks = rows.div_ceil(8);
        let values_dev = self.compute_mid_a_ptr(blocks * std::mem::size_of::<f32>())?;
        let indices_dev = self.compute_mid_b_ptr(blocks * std::mem::size_of::<u32>())?;
        let mut weights_arg = weights_dev;
        let mut input_qs_arg = input_qs_dev;
        let mut input_ds_arg = input_ds_dev;
        let mut block_values_arg = values_dev;
        let mut block_indices_arg = indices_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q8_0_gemv_q8dot_argmax_warp8",
            &[
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (blocks as u32, 1, 1),
            (256, 1, 1),
        )?;

        let reduce_blocks = 256usize.min(blocks.max(1).div_ceil(256));
        let reduce_values_dev =
            self.compute_full_up_ptr(reduce_blocks * std::mem::size_of::<f32>())?;
        let reduce_indices_dev =
            self.compute_full_down_ptr(reduce_blocks * std::mem::size_of::<u32>())?;
        let mut values_arg = values_dev;
        let mut indices_arg = indices_dev;
        let mut block_values_arg = reduce_values_dev;
        let mut block_indices_arg = reduce_indices_dev;
        let mut len_arg = blocks as u32;
        self.launch_cached_gemv(
            "rnb_argmax_pairs_f32",
            &[
                (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (reduce_blocks as u32, 1, 1),
            (256, 1, 1),
        )?;

        let mut block_values = vec![0.0f32; reduce_blocks];
        let mut block_indices = vec![0u32; reduce_blocks];
        unsafe {
            self.api.memcpy_dtoh_async(
                block_values.as_mut_ptr().cast::<libc::c_void>(),
                reduce_values_dev,
                reduce_blocks * std::mem::size_of::<f32>(),
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                block_indices.as_mut_ptr().cast::<libc::c_void>(),
                reduce_indices_dev,
                reduce_blocks * std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;

        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (&value, &idx) in block_values.iter().zip(block_indices.iter()) {
            if value > best_val || (value == best_val && idx < best_idx) {
                best_val = value;
                best_idx = idx;
            }
        }
        Ok((best_idx, best_val))
    }

    pub(super) fn q6k_gemv_argmax_fused_warp8(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<(u32, f32), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }

        let blocks = rows.div_ceil(8);
        let values_dev = self.compute_mid_a_ptr(blocks * std::mem::size_of::<f32>())?;
        let indices_dev = self.compute_mid_b_ptr(blocks * std::mem::size_of::<u32>())?;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut block_values_arg = values_dev;
        let mut block_indices_arg = indices_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q6k_gemv_argmax_warp8",
            &[
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (blocks as u32, 1, 1),
            (256, 1, 1),
        )?;

        let reduce_on_gpu = tuning::q6k_fused_argmax_gpu_reduce_enabled(rows);
        let (read_values_dev, read_indices_dev, read_blocks) = if reduce_on_gpu {
            let reduce_blocks = 256usize.min(blocks.max(1).div_ceil(256));
            let reduce_values_dev =
                self.compute_full_up_ptr(reduce_blocks * std::mem::size_of::<f32>())?;
            let reduce_indices_dev =
                self.compute_full_down_ptr(reduce_blocks * std::mem::size_of::<u32>())?;
            let mut values_arg = values_dev;
            let mut indices_arg = indices_dev;
            let mut block_values_arg = reduce_values_dev;
            let mut block_indices_arg = reduce_indices_dev;
            let mut len_arg = blocks as u32;
            self.launch_cached_gemv(
                "rnb_argmax_pairs_f32",
                &[
                    (&mut values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_values_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut block_indices_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut len_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (reduce_blocks as u32, 1, 1),
                (256, 1, 1),
            )?;
            (reduce_values_dev, reduce_indices_dev, reduce_blocks)
        } else {
            (values_dev, indices_dev, blocks)
        };

        let mut block_values = vec![0.0f32; read_blocks];
        let mut block_indices = vec![0u32; read_blocks];
        unsafe {
            self.api.memcpy_dtoh_async(
                block_values.as_mut_ptr().cast::<libc::c_void>(),
                read_values_dev,
                read_blocks * std::mem::size_of::<f32>(),
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                block_indices.as_mut_ptr().cast::<libc::c_void>(),
                read_indices_dev,
                read_blocks * std::mem::size_of::<u32>(),
                self.stream,
            )?;
        }
        self.stream_synchronize()?;

        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (&value, &idx) in block_values.iter().zip(block_indices.iter()) {
            if value > best_val || (value == best_val && idx < best_idx) {
                best_val = value;
                best_idx = idx;
            }
        }
        Ok((best_idx, best_val))
    }

    pub(super) fn q2k_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.launch_q2k_gemv_to_dev(weights, rows, blocks_per_row, input_dev, output_dev)?;
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn q3k_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        self.launch_q3k_gemv_to_dev(weights, rows, blocks_per_row, input_dev, output_dev)?;
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn q5k_gemv(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = rows * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q5k_gemv",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )?;
        let mut output = vec![0.0f32; rows];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn q5k_gemv_into(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_bytes = std::mem::size_of_val(output);
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        self.launch_cached_gemv(
            "rnb_q5k_gemv",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )?;
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()
    }

    pub(super) fn q4k_gemv_batch(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.gemv_batch(
            "rnb_q4k_gemv_batch",
            weights,
            rows,
            blocks_per_row,
            seq_len,
            input,
        )
    }

    pub(super) fn q6k_gemv_batch(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.gemv_batch(
            "rnb_q6k_gemv_batch",
            weights,
            rows,
            blocks_per_row,
            seq_len,
            input,
        )
    }

    pub(super) fn q5k_gemv_batch(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.gemv_batch(
            "rnb_q5k_gemv_batch",
            weights,
            rows,
            blocks_per_row,
            seq_len,
            input,
        )
    }

    pub(super) fn f32_gemm_batch(
        &mut self,
        weights: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let trace = std::env::var("RNB_CUDA_F32_GEMM_TRACE").ok().as_deref() == Some("1");
        let seq_len = input
            .len()
            .checked_div(cols)
            .ok_or_else(|| "f32 GEMM cols must be non-zero".to_string())?;
        if weights.len() != rows * cols {
            return Err(format!(
                "f32 GEMM weight len mismatch: got {}, expected {}",
                weights.len(),
                rows * cols
            ));
        }
        if input.len() != seq_len * cols {
            return Err(format!(
                "f32 GEMM input len mismatch: got {}, expected multiple of {cols}",
                input.len()
            ));
        }

        let weights_dev = self.compute_weights_ptr(std::mem::size_of_val(weights))?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_len = seq_len * rows;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let h2d_t0 = trace.then(std::time::Instant::now);
        unsafe {
            self.api.memcpy_htod_async(
                weights_dev,
                weights.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(weights),
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut h2d_ms = 0.0;
        if let Some(t0) = h2d_t0 {
            self.stream_synchronize()?;
            h2d_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let gemm_t0 = trace.then(std::time::Instant::now);
        let stream = self.stream;
        let cublas = self.cublas_state_mut()?;
        unsafe {
            cublas.api.set_stream(cublas.handle, stream)?;
            cublas.api.sgemm(
                cublas.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                rows as i32,
                seq_len as i32,
                cols as i32,
                1.0,
                weights_dev,
                cols as i32,
                input_dev,
                cols as i32,
                0.0,
                output_dev,
                rows as i32,
            )?;
        }
        let mut gemm_ms = 0.0;
        if let Some(t0) = gemm_t0 {
            self.stream_synchronize()?;
            gemm_ms = t0.elapsed().as_micros() as f64 / 1000.0;
        }

        let mut output = vec![0.0f32; output_len];
        let dtoh_t0 = trace.then(std::time::Instant::now);
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        if let Some(t0) = dtoh_t0 {
            eprintln!(
                "[cuda-f32-gemm] rows={} cols={} seq={} weights_mb={:.1} input_mb={:.1} output_mb={:.1} h2d_ms={:.1} gemm_ms={:.1} dtoh_ms={:.1}",
                rows,
                cols,
                seq_len,
                std::mem::size_of_val(weights) as f64 / (1024.0 * 1024.0),
                std::mem::size_of_val(input) as f64 / (1024.0 * 1024.0),
                output_bytes as f64 / (1024.0 * 1024.0),
                h2d_ms,
                gemm_ms,
                t0.elapsed().as_micros() as f64 / 1000.0
            );
        }
        Ok(output)
    }

    pub(super) fn q4k_f32_gemm_batch_cached(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        let cols = blocks_per_row
            .checked_mul(256)
            .ok_or_else(|| format!("Q4_K F32 GEMM cols overflow: blocks={blocks_per_row}"))?;
        let expected_input = seq_len.checked_mul(cols).ok_or_else(|| {
            format!("Q4_K F32 GEMM input length overflow: seq_len={seq_len} cols={cols}")
        })?;
        if input.len() != expected_input {
            return Err(format!(
                "Q4_K F32 GEMM input length mismatch: got {}, expected {expected_input}",
                input.len()
            ));
        }
        let expected_weights = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| {
                format!("Q4_K F32 GEMM weight size overflow: rows={rows} blocks={blocks_per_row}")
            })?;
        if weights.len() != expected_weights {
            return Err(format!(
                "Q4_K F32 GEMM weight byte mismatch: got {}, expected {expected_weights}",
                weights.len()
            ));
        }
        // cu19: when the resident f32 cache cannot hold this weight (limit/full),
        // fall back to GPU dequant into a scratch buffer instead of returning
        // None and forcing the caller into host-side dequant + 4-byte H2D.
        // Weight upload still happens (Q4_K bytes only, 0.625 byte/element) but
        // is cached in resident_q4k via resident_q4k_weights_ptr inside the
        // dequant launcher, so subsequent visits to the same layer are H2D-free.
        //
        // IMPORTANT: dequant output MUST NOT share a buffer with the raw Q4
        // weight temp upload. `launch_q4k_dequant_f32_to_dev` →
        // `resident_q4k_weights_ptr` falls back to `upload_temp_q4k_weights_current`
        // when the resident cache misses, and that path uses
        // `compute_weights_ptr` as its staging buffer. If the dequant output
        // also lived in `compute_weights_ptr`, the kernel would overwrite the
        // raw Q4 bytes it still has to read, silently corrupting the GEMM
        // weights. We allocate the dequant output via `compute_full_down_ptr`
        // which is disjoint from the cache fallback buffer.
        let weight_f32_bytes = rows * cols * std::mem::size_of::<f32>();
        let weights_dev =
            if let Some(p) = self.resident_q4k_f32_ptr(weights, rows, blocks_per_row)? {
                p
            } else {
                let scratch = self.compute_full_down_ptr(weight_f32_bytes)?;
                self.launch_q4k_dequant_f32_to_dev(weights, rows, blocks_per_row, scratch)?;
                scratch
            };

        let input_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("Q4_K F32 GEMM input byte overflow: len={}", input.len()))?;
        let output_len = seq_len.checked_mul(rows).ok_or_else(|| {
            format!("Q4_K F32 GEMM output length overflow: seq_len={seq_len} rows={rows}")
        })?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F32 GEMM output byte overflow: seq_len={seq_len} rows={rows}")
            })?;

        let input_dev = self.compute_input_ptr(input_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                self.stream,
            )?;
        }
        self.sgemm_device(weights_dev, rows, cols, input_dev, seq_len, output_dev)?;
        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Some(output))
    }

    pub(super) fn q8_0_f32_gemm_batch_cached(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        let cols = blocks_per_row
            .checked_mul(32)
            .ok_or_else(|| format!("Q8_0 F32 GEMM cols overflow: blocks={blocks_per_row}"))?;
        let expected_input = seq_len.checked_mul(cols).ok_or_else(|| {
            format!("Q8_0 F32 GEMM input length overflow: seq_len={seq_len} cols={cols}")
        })?;
        if input.len() != expected_input {
            return Err(format!(
                "Q8_0 F32 GEMM input length mismatch: got {}, expected {expected_input}",
                input.len()
            ));
        }
        let expected_weights = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(34))
            .ok_or_else(|| {
                format!("Q8_0 F32 GEMM weight size overflow: rows={rows} blocks={blocks_per_row}")
            })?;
        if weights.len() != expected_weights {
            return Err(format!(
                "Q8_0 F32 GEMM weight byte mismatch: got {}, expected {expected_weights}",
                weights.len()
            ));
        }

        let trace = std::env::var("RNB_CUDA_Q8_0_PREFILL_F32_GEMM_TRACE")
            .ok()
            .as_deref()
            == Some("1");
        let quant_cache = q8_0_prefill_quant_cache_enabled();
        let quant_start = trace.then(std::time::Instant::now);
        let quant_dev = if quant_cache {
            self.resident_q8_quant_ptr(weights, rows, cols)?
        } else {
            let ptr = self.compute_weights_ptr(weights.len())?;
            unsafe {
                self.api.memcpy_htod_async(
                    ptr,
                    weights.as_ptr().cast::<libc::c_void>(),
                    weights.len(),
                    self.stream,
                )?;
            }
            ptr
        };
        let weights_dev = self.resident_q8_0_f32_ptr(weights, quant_dev, rows, blocks_per_row)?;
        let mut quant_ms = 0.0;
        if let Some(start) = quant_start {
            self.stream_synchronize()?;
            quant_ms = start.elapsed().as_micros() as f64 / 1000.0;
        }

        let input_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("Q8_0 F32 GEMM input byte overflow: len={}", input.len()))?;
        let output_len = seq_len.checked_mul(rows).ok_or_else(|| {
            format!("Q8_0 F32 GEMM output length overflow: seq_len={seq_len} rows={rows}")
        })?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q8_0 F32 GEMM output byte overflow: seq_len={seq_len} rows={rows}")
            })?;

        let input_dev = self.compute_input_ptr(input_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let h2d_start = trace.then(std::time::Instant::now);
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                self.stream,
            )?;
        }
        let mut h2d_ms = 0.0;
        if let Some(start) = h2d_start {
            self.stream_synchronize()?;
            h2d_ms = start.elapsed().as_micros() as f64 / 1000.0;
        }

        let gemm_start = trace.then(std::time::Instant::now);
        self.sgemm_device(weights_dev, rows, cols, input_dev, seq_len, output_dev)?;
        let mut gemm_ms = 0.0;
        if let Some(start) = gemm_start {
            self.stream_synchronize()?;
            gemm_ms = start.elapsed().as_micros() as f64 / 1000.0;
        }

        let pinned_dtoh = q8_0_prefill_pinned_dtoh_enabled() && output_bytes >= (1 << 20);
        let mut output = vec![0.0f32; output_len];
        let dtoh_start = trace.then(std::time::Instant::now);
        if pinned_dtoh && output_bytes > 0 {
            let host_ptr = self.host_temp_slab_ptr(output_bytes)?;
            unsafe {
                self.api.memcpy_dtoh_async(
                    host_ptr.cast::<libc::c_void>(),
                    output_dev,
                    output_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_ptr.cast::<f32>(),
                    output.as_mut_ptr(),
                    output_len,
                );
            }
        } else {
            unsafe {
                self.api.memcpy_dtoh_async(
                    output.as_mut_ptr().cast::<libc::c_void>(),
                    output_dev,
                    output_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
        }
        if let Some(start) = dtoh_start {
            eprintln!(
                "[cuda-q8-f32-gemm] rows={} cols={} seq={} quant_cache={} pinned_dtoh={} quant_mb={:.1} f32_mb={:.1} input_mb={:.1} output_mb={:.1} quant_ms={:.1} h2d_ms={:.1} gemm_ms={:.1} dtoh_ms={:.1}",
                rows,
                cols,
                seq_len,
                quant_cache,
                pinned_dtoh,
                weights.len() as f64 / (1024.0 * 1024.0),
                (rows * cols * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0),
                input_bytes as f64 / (1024.0 * 1024.0),
                output_bytes as f64 / (1024.0 * 1024.0),
                quant_ms,
                h2d_ms,
                gemm_ms,
                start.elapsed().as_micros() as f64 / 1000.0
            );
        }
        Ok(Some(output))
    }

    pub(super) fn sgemm_device(
        &mut self,
        weights_dev: u64,
        rows: usize,
        cols: usize,
        input_dev: u64,
        seq_len: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        let stream = self.stream;
        let cublas = self.cublas_state_mut()?;
        unsafe {
            cublas.api.set_stream(cublas.handle, stream)?;
            cublas.api.sgemm(
                cublas.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                rows as i32,
                seq_len as i32,
                cols as i32,
                1.0,
                weights_dev,
                cols as i32,
                input_dev,
                cols as i32,
                0.0,
                output_dev,
                rows as i32,
            )
        }
    }

    pub(super) fn hgemm_to_f32_device(
        &mut self,
        weights_dev: u64,
        rows: usize,
        cols: usize,
        input_dev: u64,
        seq_len: usize,
        output_dev: u64,
    ) -> Result<(), String> {
        let stream = self.stream;
        let cublas = self.cublas_state_mut()?;
        unsafe {
            cublas.api.set_stream(cublas.handle, stream)?;
            cublas.api.gemm_ex_half_half_to_f32(
                cublas.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                rows as i32,
                seq_len as i32,
                cols as i32,
                1.0,
                weights_dev,
                cols as i32,
                input_dev,
                cols as i32,
                0.0,
                output_dev,
                rows as i32,
            )
        }
    }

    pub(super) fn q4k_f16_gemm_batch(
        &mut self,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        let cols = blocks_per_row
            .checked_mul(256)
            .ok_or_else(|| format!("Q4_K F16 GEMM cols overflow: blocks={blocks_per_row}"))?;
        let expected_input = seq_len.checked_mul(cols).ok_or_else(|| {
            format!("Q4_K F16 GEMM input length overflow: seq_len={seq_len} cols={cols}")
        })?;
        if input.len() != expected_input {
            return Err(format!(
                "Q4_K F16 GEMM input length mismatch: got {}, expected {expected_input}",
                input.len()
            ));
        }
        let expected_weights = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| {
                format!("Q4_K F16 GEMM weight size overflow: rows={rows} blocks={blocks_per_row}")
            })?;
        if weights.len() != expected_weights {
            return Err(format!(
                "Q4_K F16 GEMM weight byte mismatch: got {}, expected {expected_weights}",
                weights.len()
            ));
        }
        if !tuning::prefill_q4k_f16_gemm_enabled() {
            return Ok(None);
        }
        let Some(weights_dev) = self.resident_q4k_f16_ptr(weights, rows, blocks_per_row)? else {
            return Ok(None);
        };

        let input_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV attention input byte overflow: len={}",
                    input.len()
                )
            })?;
        let input_f16_bytes = expected_input
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!("Q4_K F16 GEMM input f16 byte overflow: seq_len={seq_len} cols={cols}")
            })?;
        let output_len = seq_len.checked_mul(rows).ok_or_else(|| {
            format!("Q4_K F16 GEMM output length overflow: seq_len={seq_len} rows={rows}")
        })?;
        let output_bytes = output_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 GEMM output byte overflow: seq_len={seq_len} rows={rows}")
            })?;

        let input_dev = self.compute_input_ptr(input_bytes)?;
        let input_f16_dev = self.compute_aux_output_ptr(input_f16_bytes)?;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                self.stream,
            )?;
        }
        self.launch_f32_to_f16(input_dev, input_f16_dev, expected_input)?;
        self.hgemm_to_f32_device(weights_dev, rows, cols, input_f16_dev, seq_len, output_dev)?;
        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Some(output))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn q4k_f16_qkv_gemm_batch(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>, String> {
        let cols = blocks_per_row
            .checked_mul(256)
            .ok_or_else(|| format!("Q4_K F16 QKV GEMM cols overflow: blocks={blocks_per_row}"))?;
        let expected_input = seq_len.checked_mul(cols).ok_or_else(|| {
            format!("Q4_K F16 QKV GEMM input length overflow: seq_len={seq_len} cols={cols}")
        })?;
        if input.len() != expected_input {
            return Err(format!(
                "Q4_K F16 QKV GEMM input length mismatch: got {}, expected {expected_input}",
                input.len()
            ));
        }
        let row_bytes = blocks_per_row.checked_mul(144).ok_or_else(|| {
            format!("Q4_K F16 QKV GEMM row byte overflow: blocks={blocks_per_row}")
        })?;
        let q_expected = q_rows.checked_mul(row_bytes).ok_or_else(|| {
            format!("Q4_K F16 QKV q byte overflow: rows={q_rows} row_bytes={row_bytes}")
        })?;
        let kv_expected = kv_rows.checked_mul(row_bytes).ok_or_else(|| {
            format!("Q4_K F16 QKV kv byte overflow: rows={kv_rows} row_bytes={row_bytes}")
        })?;
        if q_weights.len() != q_expected {
            return Err(format!(
                "Q4_K F16 QKV q byte mismatch: got {}, expected {q_expected}",
                q_weights.len()
            ));
        }
        if k_weights.len() != kv_expected || v_weights.len() != kv_expected {
            return Err(format!(
                "Q4_K F16 QKV k/v byte mismatch: k={} v={} expected {kv_expected}",
                k_weights.len(),
                v_weights.len()
            ));
        }

        // Raw Q4 projection experiments do not materialize full F16 weights. The
        // cuBLAS F16 fallback below does, so it stays behind the diagnostic gate.
        let use_fused_naive =
            std::env::var("RNB_CUDA_Q4K_FUSED_NAIVE").ok().as_deref() == Some("1");
        let use_fused_wmma = std::env::var("RNB_CUDA_Q4K_FUSED_WMMA").ok().as_deref() == Some("1");
        let use_mma_4warp = std::env::var("RNB_CUDA_Q4K_MMA_4WARP").ok().as_deref() == Some("1");
        let use_mma = std::env::var("RNB_CUDA_Q4K_MMA").ok().as_deref() == Some("1");
        let use_dp4a_tile = std::env::var("RNB_CUDA_Q4K_DP4A_TILE").ok().as_deref() == Some("1");
        let use_dp4a = std::env::var("RNB_CUDA_Q4K_DP4A").ok().as_deref() == Some("1");
        let use_fused_wmma_4warp = std::env::var("RNB_CUDA_Q4K_FUSED_WMMA_4WARP")
            .ok()
            .as_deref()
            == Some("1");
        let raw_q4_projection_requested = use_fused_naive
            || use_fused_wmma
            || use_mma_4warp
            || use_mma
            || use_dp4a_tile
            || use_dp4a
            || use_fused_wmma_4warp;
        if !raw_q4_projection_requested && !tuning::prefill_q4k_f16_qkv_gemm_enabled() {
            return Ok(None);
        }

        let input_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV attention input byte overflow: len={}",
                    input.len()
                )
            })?;
        let input_f16_bytes = expected_input
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!("Q4_K F16 QKV input f16 byte overflow: seq_len={seq_len} cols={cols}")
            })?;
        let q_len = seq_len.checked_mul(q_rows).ok_or_else(|| {
            format!("Q4_K F16 QKV q output overflow: seq_len={seq_len} rows={q_rows}")
        })?;
        let kv_len = seq_len.checked_mul(kv_rows).ok_or_else(|| {
            format!("Q4_K F16 QKV kv output overflow: seq_len={seq_len} rows={kv_rows}")
        })?;
        let q_bytes = q_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 QKV q byte overflow: seq_len={seq_len} rows={q_rows}")
            })?;
        let kv_bytes = kv_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 QKV kv byte overflow: seq_len={seq_len} rows={kv_rows}")
            })?;

        let input_dev = self.compute_input_ptr(input_bytes)?;
        let input_f16_dev = self.compute_aux_output_ptr(input_f16_bytes)?;
        let q_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_dev = self.compute_mid_b_ptr(kv_bytes)?;
        let v_dev = self.compute_output_ptr(kv_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                input_bytes,
                self.stream,
            )?;
        }
        // cu38 Phase 3 dispatcher: env opt-in.
        // - RNB_CUDA_Q4K_FUSED_NAIVE=1: Phase 1 naive kernel (correctness OK,
        //   prefill +4.4% 회귀 측정됨)
        // - RNB_CUDA_Q4K_FUSED_WMMA=1: Phase 2 wmma tensor core kernel (정확도
        //   검증 중)
        if use_fused_naive {
            self.launch_q4k_sgemm_fused_naive(
                q_weights,
                q_rows,
                blocks_per_row,
                seq_len,
                input_dev,
                q_dev,
            )?;
            self.launch_q4k_sgemm_fused_naive(
                k_weights,
                kv_rows,
                blocks_per_row,
                seq_len,
                input_dev,
                k_dev,
            )?;
            self.launch_q4k_sgemm_fused_naive(
                v_weights,
                kv_rows,
                blocks_per_row,
                seq_len,
                input_dev,
                v_dev,
            )?;
            self.stream_synchronize()?;
            let mut q = vec![0.0f32; q_len];
            let mut k = vec![0.0f32; kv_len];
            let mut v = vec![0.0f32; kv_len];
            unsafe {
                self.api.memcpy_dtoh_async(
                    q.as_mut_ptr().cast::<libc::c_void>(),
                    q_dev,
                    q_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    k.as_mut_ptr().cast::<libc::c_void>(),
                    k_dev,
                    kv_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    v.as_mut_ptr().cast::<libc::c_void>(),
                    v_dev,
                    kv_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            return Ok(Some((q, k, v)));
        }
        // cu39 Phase 1: Q4_K × Q8_1 DP4A matmul (llama.cpp mmq pattern port).
        // llama prefill 2885 tok/s vs 우리 246 tok/s (~11.7×) 격차 정조준. cuBLAS 거치지 않고
        // Q4_K nibble × Q8_1 int8 직접 dp4a multiply-accumulate.
        // - RNB_CUDA_Q4K_DP4A=1: Phase 1 naive (no shared mem tile, cuBLAS와 ε 측정됨)
        // - RNB_CUDA_Q4K_DP4A_TILE=1: Phase 2 shared mem 8×8 tile + weight reuse
        // - RNB_CUDA_Q4K_MMA=1: Phase 3 mma.m16n8k32.s8.s8 1-warp
        // - RNB_CUDA_Q4K_MMA_4WARP=1: Phase 4 mma 4-warp + mmq_y=64
        if use_mma_4warp || use_mma || use_dp4a_tile || use_dp4a {
            // Q8_1 scratch buffers in temp slab (qs + ds + sums). aligned packing.
            let q8_len = expected_input; // seq_len * K elements
            let chunks = q8_len / 32;
            let qs_bytes = q8_len; // 1 byte/elem (i8)
            let ds_bytes = chunks * std::mem::size_of::<f32>();
            let sums_bytes = chunks * std::mem::size_of::<f32>();
            let align = 256usize;
            let qs_off = 0usize;
            let ds_off = (qs_off + qs_bytes).next_multiple_of(align);
            let sums_off = (ds_off + ds_bytes).next_multiple_of(align);
            let total = (sums_off + sums_bytes).next_multiple_of(align);
            let slab = self.compute_temp_slab_ptr(total)?;
            let input_qs_dev = slab + qs_off as u64;
            let input_ds_dev = slab + ds_off as u64;
            let input_sums_dev = slab + sums_off as u64;
            self.launch_quantize_q8_1_with_sum_by_32(
                input_dev,
                input_qs_dev,
                input_ds_dev,
                input_sums_dev,
                q8_len,
            )?;
            if use_mma_4warp {
                self.launch_q4k_q8_1_matmul_mma_4warp(
                    q_weights,
                    q_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    q_dev,
                )?;
                self.launch_q4k_q8_1_matmul_mma_4warp(
                    k_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    k_dev,
                )?;
                self.launch_q4k_q8_1_matmul_mma_4warp(
                    v_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    v_dev,
                )?;
            } else if use_mma {
                self.launch_q4k_q8_1_matmul_mma(
                    q_weights,
                    q_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    q_dev,
                )?;
                self.launch_q4k_q8_1_matmul_mma(
                    k_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    k_dev,
                )?;
                self.launch_q4k_q8_1_matmul_mma(
                    v_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    v_dev,
                )?;
            } else if use_dp4a_tile {
                self.launch_q4k_q8_1_matmul_dp4a_tile(
                    q_weights,
                    q_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    q_dev,
                )?;
                self.launch_q4k_q8_1_matmul_dp4a_tile(
                    k_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    k_dev,
                )?;
                self.launch_q4k_q8_1_matmul_dp4a_tile(
                    v_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    v_dev,
                )?;
            } else {
                self.launch_q4k_q8_1_matmul_dp4a(
                    q_weights,
                    q_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    q_dev,
                )?;
                self.launch_q4k_q8_1_matmul_dp4a(
                    k_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    k_dev,
                )?;
                self.launch_q4k_q8_1_matmul_dp4a(
                    v_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    input_qs_dev,
                    input_ds_dev,
                    input_sums_dev,
                    v_dev,
                )?;
            }
            self.stream_synchronize()?;
            let mut q = vec![0.0f32; q_len];
            let mut k = vec![0.0f32; kv_len];
            let mut v = vec![0.0f32; kv_len];
            unsafe {
                self.api.memcpy_dtoh_async(
                    q.as_mut_ptr().cast::<libc::c_void>(),
                    q_dev,
                    q_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    k.as_mut_ptr().cast::<libc::c_void>(),
                    k_dev,
                    kv_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    v.as_mut_ptr().cast::<libc::c_void>(),
                    v_dev,
                    kv_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            return Ok(Some((q, k, v)));
        }
        if use_fused_wmma_4warp {
            self.launch_f32_to_f16(input_dev, input_f16_dev, expected_input)?;
            self.launch_q4k_sgemm_fused_wmma_4warp(
                q_weights,
                q_rows,
                blocks_per_row,
                seq_len,
                input_f16_dev,
                q_dev,
            )?;
            self.launch_q4k_sgemm_fused_wmma_4warp(
                k_weights,
                kv_rows,
                blocks_per_row,
                seq_len,
                input_f16_dev,
                k_dev,
            )?;
            self.launch_q4k_sgemm_fused_wmma_4warp(
                v_weights,
                kv_rows,
                blocks_per_row,
                seq_len,
                input_f16_dev,
                v_dev,
            )?;
            self.stream_synchronize()?;
            let mut q = vec![0.0f32; q_len];
            let mut k = vec![0.0f32; kv_len];
            let mut v = vec![0.0f32; kv_len];
            unsafe {
                self.api.memcpy_dtoh_async(
                    q.as_mut_ptr().cast::<libc::c_void>(),
                    q_dev,
                    q_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    k.as_mut_ptr().cast::<libc::c_void>(),
                    k_dev,
                    kv_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    v.as_mut_ptr().cast::<libc::c_void>(),
                    v_dev,
                    kv_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            return Ok(Some((q, k, v)));
        }
        if use_fused_wmma {
            // wmma kernel needs F16 input (matrix_b).
            self.launch_f32_to_f16(input_dev, input_f16_dev, expected_input)?;
            self.launch_q4k_sgemm_fused_wmma(
                q_weights,
                q_rows,
                blocks_per_row,
                seq_len,
                input_f16_dev,
                q_dev,
            )?;
            self.launch_q4k_sgemm_fused_wmma(
                k_weights,
                kv_rows,
                blocks_per_row,
                seq_len,
                input_f16_dev,
                k_dev,
            )?;
            self.launch_q4k_sgemm_fused_wmma(
                v_weights,
                kv_rows,
                blocks_per_row,
                seq_len,
                input_f16_dev,
                v_dev,
            )?;
            self.stream_synchronize()?;
            let mut q = vec![0.0f32; q_len];
            let mut k = vec![0.0f32; kv_len];
            let mut v = vec![0.0f32; kv_len];
            unsafe {
                self.api.memcpy_dtoh_async(
                    q.as_mut_ptr().cast::<libc::c_void>(),
                    q_dev,
                    q_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    k.as_mut_ptr().cast::<libc::c_void>(),
                    k_dev,
                    kv_bytes,
                    self.stream,
                )?;
                self.api.memcpy_dtoh_async(
                    v.as_mut_ptr().cast::<libc::c_void>(),
                    v_dev,
                    kv_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            return Ok(Some((q, k, v)));
        }
        self.launch_f32_to_f16(input_dev, input_f16_dev, expected_input)?;
        let q_weights_dev = self.transient_q4k_f16_ptr(q_weights, q_rows, blocks_per_row)?;
        self.hgemm_to_f32_device(q_weights_dev, q_rows, cols, input_f16_dev, seq_len, q_dev)?;
        let k_weights_dev = self.transient_q4k_f16_ptr(k_weights, kv_rows, blocks_per_row)?;
        self.hgemm_to_f32_device(k_weights_dev, kv_rows, cols, input_f16_dev, seq_len, k_dev)?;
        let v_weights_dev = self.transient_q4k_f16_ptr(v_weights, kv_rows, blocks_per_row)?;
        self.hgemm_to_f32_device(v_weights_dev, kv_rows, cols, input_f16_dev, seq_len, v_dev)?;

        let mut q = vec![0.0f32; q_len];
        let mut k = vec![0.0f32; kv_len];
        let mut v = vec![0.0f32; kv_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                q.as_mut_ptr().cast::<libc::c_void>(),
                q_dev,
                q_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                k.as_mut_ptr().cast::<libc::c_void>(),
                k_dev,
                kv_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v.as_mut_ptr().cast::<libc::c_void>(),
                v_dev,
                kv_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Some((q, k, v)))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn q4k_f16_qkv_postprocess_hd256_to_device(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        v_quant: u32,
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: Q4kF16QkvInput<'_>,
        q_norm: &[f32],
        k_norm: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
    ) -> Result<Option<Q4kF16QkvHd256PostprocessDevice>, String> {
        let head_dim = 256usize;
        if q_rows != num_heads.saturating_mul(head_dim)
            || kv_rows != num_kv_heads.saturating_mul(head_dim)
        {
            return Ok(None);
        }
        if q_norm.len() != head_dim || k_norm.len() != head_dim {
            return Ok(None);
        }
        let cols = blocks_per_row.checked_mul(256).ok_or_else(|| {
            format!("Q4_K F16 QKV postprocess cols overflow: blocks={blocks_per_row}")
        })?;
        let expected_input = seq_len.checked_mul(cols).ok_or_else(|| {
            format!("Q4_K F16 QKV postprocess input overflow: seq_len={seq_len} cols={cols}")
        })?;
        if let Q4kF16QkvInput::DeviceHidden { desc, .. } = &input {
            if desc.rows() != seq_len
                || desc.cols() != cols
                || desc.dtype() != rnb_backend_api::ScalarType::F32
            {
                return Ok(None);
            }
        } else if input.len() != expected_input {
            return Err(format!(
                "Q4_K F16 QKV postprocess input length mismatch: got {}, expected {expected_input}",
                input.len()
            ));
        }
        let row_bytes = blocks_per_row.checked_mul(144).ok_or_else(|| {
            format!("Q4_K F16 QKV postprocess row byte overflow: blocks={blocks_per_row}")
        })?;
        let q_expected = q_rows.checked_mul(row_bytes).ok_or_else(|| {
            format!("Q4_K F16 QKV postprocess q byte overflow: rows={q_rows} row_bytes={row_bytes}")
        })?;
        let k_expected = kv_rows.checked_mul(row_bytes).ok_or_else(|| {
            format!(
                "Q4_K F16 QKV postprocess kv byte overflow: rows={kv_rows} row_bytes={row_bytes}"
            )
        })?;
        let v_row_bytes = match v_quant {
            12 => row_bytes,
            14 => blocks_per_row.checked_mul(210).ok_or_else(|| {
                format!("Q6_K F16 QKV postprocess v row byte overflow: blocks={blocks_per_row}")
            })?,
            other => return Err(format!("unsupported QKV postprocess V quant code {other}")),
        };
        let v_expected = kv_rows.checked_mul(v_row_bytes).ok_or_else(|| {
            format!(
                "Q4_K F16 QKV postprocess v byte overflow: rows={kv_rows} row_bytes={v_row_bytes}"
            )
        })?;
        if q_weights.len() != q_expected {
            return Err(format!(
                "Q4_K F16 QKV postprocess q byte mismatch: got {}, expected {q_expected}",
                q_weights.len()
            ));
        }
        if k_weights.len() != k_expected {
            return Err(format!(
                "Q4_K F16 QKV postprocess k byte mismatch: got {}, expected {k_expected}",
                k_weights.len()
            ));
        }
        if v_weights.len() != v_expected {
            return Err(format!(
                "Q4_K F16 QKV postprocess v byte mismatch: got {}, expected {v_expected}",
                v_weights.len()
            ));
        }
        if !tuning::prefill_q4k_f16_qkv_gemm_enabled() {
            return Ok(None);
        }

        let Some(q_weights_dev) = self.resident_q4k_f16_ptr(q_weights, q_rows, blocks_per_row)?
        else {
            return Ok(None);
        };
        let k_weights_dev = self.resident_q4k_f16_ptr(k_weights, kv_rows, blocks_per_row)?;
        let v_weights_dev = match v_quant {
            12 => self.resident_q4k_f16_ptr(v_weights, kv_rows, blocks_per_row)?,
            14 => self.resident_q6k_f16_ptr(v_weights, kv_rows, blocks_per_row)?,
            other => return Err(format!("unsupported QKV postprocess V quant code {other}")),
        };

        let input_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV postprocess input byte overflow: len={}",
                    input.len()
                )
            })?;
        let input_f16_bytes = expected_input
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV postprocess input f16 byte overflow: seq_len={seq_len} cols={cols}"
                )
            })?;
        let q_len = seq_len.checked_mul(q_rows).ok_or_else(|| {
            format!("Q4_K F16 QKV postprocess q output overflow: seq_len={seq_len} rows={q_rows}")
        })?;
        let kv_len = seq_len.checked_mul(kv_rows).ok_or_else(|| {
            format!("Q4_K F16 QKV postprocess kv output overflow: seq_len={seq_len} rows={kv_rows}")
        })?;
        let q_bytes = q_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 QKV postprocess q byte overflow: seq_len={seq_len} rows={q_rows}")
            })?;
        let kv_f32_bytes = kv_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV postprocess kv f32 byte overflow: seq_len={seq_len} rows={kv_rows}"
                )
            })?;
        let kv_f16_bytes = kv_len
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV postprocess kv f16 byte overflow: seq_len={seq_len} rows={kv_rows}"
                )
            })?;
        let rope_table = self.rope_table_ptrs(head_dim, seq_len, pos_start, rope_theta)?;

        let input_dev = match &input {
            Q4kF16QkvInput::DeviceHidden { id, desc, .. } => self.device_tensor_ptr(*id, *desc)?,
            Q4kF16QkvInput::Normed(_) | Q4kF16QkvInput::Hidden { .. } => {
                self.compute_input_ptr(input_bytes)?
            }
        };
        let input_f16_dev = self.compute_aux_output_ptr(input_f16_bytes)?;
        let q_raw_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_raw_dev = self.compute_mid_b_ptr(kv_f32_bytes)?;
        let v_raw_dev = self.compute_weights_ptr(kv_f32_bytes)?;
        let q_post_dev = self.compute_full_gate_ptr(if input.is_hidden() {
            input_bytes.max(q_bytes)
        } else {
            q_bytes
        })?;
        let normed_dev = match &input {
            Q4kF16QkvInput::Normed(_) => input_dev,
            Q4kF16QkvInput::Hidden { .. } | Q4kF16QkvInput::DeviceHidden { .. } => q_post_dev,
        };
        let k_bits_dev = self.compute_full_up_ptr(kv_f16_bytes)?;
        let v_bits_dev = self.compute_full_down_ptr(kv_f16_bytes)?;
        // cu19: cache q_norm/k_norm in the f32 resident cache (keyed by content
        // hash). First prefill incurs the H2D as before; every subsequent layer
        // visit (and all decode steps) skips the 2 H2D launches per layer.
        let q_norm_dev = self.resident_f32_ptr(q_norm)?;
        let k_norm_dev = self.resident_f32_ptr(k_norm)?;
        let rope_sin_dev = rope_table.sin_ptr;
        let rope_cos_dev = rope_table.cos_ptr;
        unsafe {
            match &input {
                Q4kF16QkvInput::Normed(values) | Q4kF16QkvInput::Hidden { values, .. } => {
                    self.api.memcpy_htod_async(
                        input_dev,
                        values.as_ptr().cast::<libc::c_void>(),
                        input_bytes,
                        self.stream,
                    )?;
                }
                Q4kF16QkvInput::DeviceHidden { .. } => {}
            }
            // q_norm / k_norm already uploaded by resident_f32_ptr above; no H2D here.
        }
        if let Some((attn_norm, unit_offset)) = match &input {
            Q4kF16QkvInput::Hidden {
                attn_norm,
                unit_offset,
                ..
            }
            | Q4kF16QkvInput::DeviceHidden {
                attn_norm,
                unit_offset,
                ..
            } => Some((*attn_norm, *unit_offset)),
            Q4kF16QkvInput::Normed(_) => None,
        } {
            if attn_norm.len() != cols {
                return Err(format!(
                    "Q4_K F16 QKV postprocess attn norm length mismatch: got {}, expected {cols}",
                    attn_norm.len()
                ));
            }
            let attn_norm_dev = self.resident_f32_ptr(attn_norm)?;
            self.launch_rms_norm_rows_f32(
                input_dev,
                attn_norm_dev,
                normed_dev,
                norm_eps,
                seq_len,
                cols,
                unit_offset,
            )?;
        }
        self.launch_f32_to_f16(normed_dev, input_f16_dev, expected_input)?;
        self.hgemm_to_f32_device(
            q_weights_dev,
            q_rows,
            cols,
            input_f16_dev,
            seq_len,
            q_raw_dev,
        )?;
        if let Some(k_weights_dev) = k_weights_dev {
            self.hgemm_to_f32_device(
                k_weights_dev,
                kv_rows,
                cols,
                input_f16_dev,
                seq_len,
                k_raw_dev,
            )?;
        } else {
            self.q4k_batch_dev_input_to_dev(
                k_weights,
                kv_rows,
                blocks_per_row,
                seq_len,
                normed_dev,
                k_raw_dev,
            )?;
        }
        if let Some(v_weights_dev) = v_weights_dev {
            self.hgemm_to_f32_device(
                v_weights_dev,
                kv_rows,
                cols,
                input_f16_dev,
                seq_len,
                v_raw_dev,
            )?;
        } else {
            match v_quant {
                12 => self.q4k_batch_dev_input_to_dev(
                    v_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    normed_dev,
                    v_raw_dev,
                )?,
                14 => self.q6k_batch_dev_input_to_dev(
                    v_weights,
                    kv_rows,
                    blocks_per_row,
                    seq_len,
                    normed_dev,
                    v_raw_dev,
                )?,
                other => return Err(format!("unsupported QKV postprocess V quant code {other}")),
            }
        }
        self.launch_qk_norm_rope_neox_hd256_f16kv(
            q_raw_dev,
            k_raw_dev,
            v_raw_dev,
            q_norm_dev,
            k_norm_dev,
            rope_sin_dev,
            rope_cos_dev,
            q_post_dev,
            k_bits_dev,
            v_bits_dev,
            norm_eps,
            seq_len,
            num_heads,
            num_kv_heads,
            pos_start,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )?;

        Ok(Some(Q4kF16QkvHd256PostprocessDevice {
            hidden_dev: matches!(input, Q4kF16QkvInput::DeviceHidden { .. }).then_some(input_dev),
            q_post_dev,
            k_bits_dev,
            v_bits_dev,
            attn_out_scratch_dev: q_raw_dev,
            k_f32_scratch_dev: k_raw_dev,
            v_f32_scratch_dev: v_raw_dev,
            q_len,
            kv_len,
            q_bytes,
            kv_f16_bytes,
        }))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn q4k_f16_qkv_postprocess_hd256(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
        q_norm: &[f32],
        k_norm: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
    ) -> Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>, String> {
        let Some(device) = self.q4k_f16_qkv_postprocess_hd256_to_device(
            q_weights,
            k_weights,
            v_weights,
            12,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            Q4kF16QkvInput::Normed(input),
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )?
        else {
            return Ok(None);
        };
        let mut q = vec![0.0f32; device.q_len];
        let mut k_bits = vec![0u16; device.kv_len];
        let mut v_bits = vec![0u16; device.kv_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                q.as_mut_ptr().cast::<libc::c_void>(),
                device.q_post_dev,
                device.q_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                k_bits.as_mut_ptr().cast::<libc::c_void>(),
                device.k_bits_dev,
                device.kv_f16_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v_bits.as_mut_ptr().cast::<libc::c_void>(),
                device.v_bits_dev,
                device.kv_f16_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Some((q, k_bits, v_bits)))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn q4k_f16_qkv_prefill_attention_hd512_to_device(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: Q4kF16QkvInput<'_>,
        q_norm: &[f32],
        k_norm: &[f32],
        freq_factors: Option<&[f32]>,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
    ) -> Result<Option<Q4kF16QkvHd512AttentionDevice>, String> {
        let head_dim = 512usize;
        if q_rows != num_heads.saturating_mul(head_dim)
            || kv_rows != num_kv_heads.saturating_mul(head_dim)
        {
            return Ok(None);
        }
        if q_norm.len() != head_dim || k_norm.len() != head_dim {
            return Ok(None);
        }
        if let Some(freq_factors) = freq_factors {
            if freq_factors.len() < head_dim / 2 {
                return Ok(None);
            }
        }
        let cols = blocks_per_row.checked_mul(256).ok_or_else(|| {
            format!("Q4_K F16 QKV attention cols overflow: blocks={blocks_per_row}")
        })?;
        let expected_input = seq_len.checked_mul(cols).ok_or_else(|| {
            format!("Q4_K F16 QKV attention input overflow: seq_len={seq_len} cols={cols}")
        })?;
        if let Q4kF16QkvInput::DeviceHidden { desc, .. } = &input {
            if desc.rows() != seq_len
                || desc.cols() != cols
                || desc.dtype() != rnb_backend_api::ScalarType::F32
            {
                return Ok(None);
            }
        } else if input.len() != expected_input {
            return Err(format!(
                "Q4_K F16 QKV attention input length mismatch: got {}, expected {expected_input}",
                input.len()
            ));
        }
        let row_bytes = blocks_per_row.checked_mul(144).ok_or_else(|| {
            format!("Q4_K F16 QKV attention row byte overflow: blocks={blocks_per_row}")
        })?;
        let q_expected = q_rows.checked_mul(row_bytes).ok_or_else(|| {
            format!("Q4_K F16 QKV attention q byte overflow: rows={q_rows} row_bytes={row_bytes}")
        })?;
        let kv_expected = kv_rows.checked_mul(row_bytes).ok_or_else(|| {
            format!("Q4_K F16 QKV attention kv byte overflow: rows={kv_rows} row_bytes={row_bytes}")
        })?;
        if q_weights.len() != q_expected {
            return Err(format!(
                "Q4_K F16 QKV attention q byte mismatch: got {}, expected {q_expected}",
                q_weights.len()
            ));
        }
        if k_weights.len() != kv_expected || v_weights.len() != kv_expected {
            return Err(format!(
                "Q4_K F16 QKV attention k/v byte mismatch: k={} v={} expected {kv_expected}",
                k_weights.len(),
                v_weights.len()
            ));
        }
        if !tuning::prefill_q4k_f16_qkv_gemm_enabled() {
            return Ok(None);
        }

        let Some(q_weights_dev) = self.resident_q4k_f16_ptr(q_weights, q_rows, blocks_per_row)?
        else {
            return Ok(None);
        };
        let Some(k_weights_dev) = self.resident_q4k_f16_ptr(k_weights, kv_rows, blocks_per_row)?
        else {
            return Ok(None);
        };
        let Some(v_weights_dev) = self.resident_q4k_f16_ptr(v_weights, kv_rows, blocks_per_row)?
        else {
            return Ok(None);
        };

        let input_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV attention input byte overflow: len={}",
                    input.len()
                )
            })?;
        let input_f16_bytes = expected_input
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV attention input f16 byte overflow: seq_len={seq_len} cols={cols}"
                )
            })?;
        let q_len = seq_len.checked_mul(q_rows).ok_or_else(|| {
            format!("Q4_K F16 QKV attention q output overflow: seq_len={seq_len} rows={q_rows}")
        })?;
        let kv_len = seq_len.checked_mul(kv_rows).ok_or_else(|| {
            format!("Q4_K F16 QKV attention kv output overflow: seq_len={seq_len} rows={kv_rows}")
        })?;
        let q_bytes = q_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 QKV attention q byte overflow: seq_len={seq_len} rows={q_rows}")
            })?;
        let kv_f32_bytes = kv_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV attention kv f32 byte overflow: seq_len={seq_len} rows={kv_rows}"
                )
            })?;
        let kv_f16_bytes = kv_len
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 QKV attention kv f16 byte overflow: seq_len={seq_len} rows={kv_rows}"
                )
            })?;
        let q_norm_bytes = std::mem::size_of_val(q_norm);
        let k_norm_bytes = std::mem::size_of_val(k_norm);
        let cached_rope_table = if freq_factors.is_none() {
            Some(self.rope_table_ptrs(head_dim, seq_len, pos_start, rope_theta)?)
        } else {
            None
        };
        let mut rope_sin = Vec::new();
        let mut rope_cos = Vec::new();
        if let Some(freq_factors) = freq_factors {
            let inv_freq = (0..head_dim / 2)
                .map(|i| {
                    1.0f32 / (rope_theta.powf((2 * i) as f32 / head_dim as f32) * freq_factors[i])
                })
                .collect::<Vec<_>>();
            let rope_table_len = seq_len.checked_mul(head_dim / 2).ok_or_else(|| {
                format!("Q4_K F16 QKV attention rope table overflow: seq_len={seq_len}")
            })?;
            rope_sin.reserve_exact(rope_table_len);
            rope_cos.reserve_exact(rope_table_len);
            for token in 0..seq_len {
                let pos = (pos_start + token) as f32;
                for &freq in &inv_freq {
                    let (sin_a, cos_a) = (pos * freq).sin_cos();
                    rope_sin.push(sin_a);
                    rope_cos.push(cos_a);
                }
            }
        }
        let rope_sin_bytes = std::mem::size_of_val(rope_sin.as_slice());
        let rope_cos_bytes = std::mem::size_of_val(rope_cos.as_slice());
        let norm_bytes = q_norm_bytes
            .checked_add(k_norm_bytes)
            .and_then(|n| n.checked_add(rope_sin_bytes))
            .and_then(|n| n.checked_add(rope_cos_bytes))
            .ok_or_else(|| "Q4_K F16 QKV attention norm byte overflow".to_string())?;

        let input_dev = match &input {
            Q4kF16QkvInput::DeviceHidden { id, desc, .. } => self.device_tensor_ptr(*id, *desc)?,
            Q4kF16QkvInput::Normed(_) | Q4kF16QkvInput::Hidden { .. } => {
                self.compute_input_ptr(input_bytes)?
            }
        };
        let input_f16_dev = self.compute_aux_output_ptr(input_f16_bytes)?;
        let q_raw_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_raw_dev = self.compute_mid_b_ptr(kv_f32_bytes)?;
        let v_raw_dev = self.compute_weights_ptr(kv_f32_bytes)?;
        let q_post_dev = self.compute_full_gate_ptr(if input.is_hidden() {
            input_bytes.max(q_bytes)
        } else {
            q_bytes
        })?;
        let k_bits_dev = self.compute_full_up_ptr(kv_f16_bytes)?;
        let v_bits_dev = self.compute_full_down_ptr(kv_f16_bytes)?;
        let k_f32_dev = self.compute_gate_ptrs_ptr(kv_f32_bytes)?;
        let v_f32_dev = self.compute_up_ptrs_ptr(kv_f32_bytes)?;
        let output_dev = self.compute_output_ptr(q_bytes)?;
        // cu19: when the rope table is cached we don't need a contiguous
        // norm+rope slab — q_norm/k_norm can live in the f32 resident cache and
        // skip H2D entirely after the first visit to this layer. When the rope
        // table is fresh we still need the legacy contiguous slab so the rope
        // offsets stay valid.
        let rope_cached = cached_rope_table.is_some();
        let (q_norm_dev, k_norm_dev, slab_dev) = if rope_cached {
            (
                self.resident_f32_ptr(q_norm)?,
                self.resident_f32_ptr(k_norm)?,
                None,
            )
        } else {
            let nd = self.compute_temp_slab_ptr(norm_bytes)?;
            (nd, nd + q_norm_bytes as u64, Some(nd))
        };
        let rope_sin_dev = cached_rope_table
            .map(|table| table.sin_ptr)
            .unwrap_or_else(|| slab_dev.unwrap() + (q_norm_bytes + k_norm_bytes) as u64);
        let rope_cos_dev = cached_rope_table
            .map(|table| table.cos_ptr)
            .unwrap_or(rope_sin_dev + rope_sin_bytes as u64);
        let normed_dev = match &input {
            Q4kF16QkvInput::Normed(_) => input_dev,
            Q4kF16QkvInput::Hidden { .. } | Q4kF16QkvInput::DeviceHidden { .. } => q_post_dev,
        };
        unsafe {
            match &input {
                Q4kF16QkvInput::Normed(values) | Q4kF16QkvInput::Hidden { values, .. } => {
                    self.api.memcpy_htod_async(
                        input_dev,
                        values.as_ptr().cast::<libc::c_void>(),
                        input_bytes,
                        self.stream,
                    )?;
                }
                Q4kF16QkvInput::DeviceHidden { .. } => {}
            }
            if !rope_cached {
                // legacy slab path — H2D q_norm/k_norm + rope sin/cos together.
                self.api.memcpy_htod_async(
                    q_norm_dev,
                    q_norm.as_ptr().cast::<libc::c_void>(),
                    q_norm_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    k_norm_dev,
                    k_norm.as_ptr().cast::<libc::c_void>(),
                    k_norm_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    rope_sin_dev,
                    rope_sin.as_ptr().cast::<libc::c_void>(),
                    rope_sin_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    rope_cos_dev,
                    rope_cos.as_ptr().cast::<libc::c_void>(),
                    rope_cos_bytes,
                    self.stream,
                )?;
            }
            // rope cached path: q_norm/k_norm already uploaded by resident_f32_ptr;
            // rope sin/cos already cached. No H2D here.
        }
        if let Some((attn_norm, unit_offset)) = match &input {
            Q4kF16QkvInput::Hidden {
                attn_norm,
                unit_offset,
                ..
            }
            | Q4kF16QkvInput::DeviceHidden {
                attn_norm,
                unit_offset,
                ..
            } => Some((*attn_norm, *unit_offset)),
            Q4kF16QkvInput::Normed(_) => None,
        } {
            if attn_norm.len() != cols {
                return Err(format!(
                    "Q4_K F16 QKV hd512 attention attn norm length mismatch: got {}, expected {cols}",
                    attn_norm.len()
                ));
            }
            let attn_norm_dev = self.resident_f32_ptr(attn_norm)?;
            self.launch_rms_norm_rows_f32(
                input_dev,
                attn_norm_dev,
                normed_dev,
                norm_eps,
                seq_len,
                cols,
                unit_offset,
            )?;
        }
        self.launch_f32_to_f16(normed_dev, input_f16_dev, expected_input)?;
        self.hgemm_to_f32_device(
            q_weights_dev,
            q_rows,
            cols,
            input_f16_dev,
            seq_len,
            q_raw_dev,
        )?;
        self.hgemm_to_f32_device(
            k_weights_dev,
            kv_rows,
            cols,
            input_f16_dev,
            seq_len,
            k_raw_dev,
        )?;
        self.hgemm_to_f32_device(
            v_weights_dev,
            kv_rows,
            cols,
            input_f16_dev,
            seq_len,
            v_raw_dev,
        )?;
        self.launch_qk_norm_rope_neox_hd512_f16kv(
            q_raw_dev,
            k_raw_dev,
            v_raw_dev,
            q_norm_dev,
            k_norm_dev,
            rope_sin_dev,
            rope_cos_dev,
            q_post_dev,
            k_bits_dev,
            v_bits_dev,
            norm_eps,
            seq_len,
            num_heads,
            num_kv_heads,
            pos_start,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )?;
        self.launch_f16_to_f32(k_bits_dev, k_f32_dev, kv_len)?;
        self.launch_f16_to_f32(v_bits_dev, v_f32_dev, kv_len)?;

        let mut output_arg = output_dev;
        let mut q_arg = q_post_dev;
        let mut k_arg = k_f32_dev;
        let mut v_arg = v_f32_dev;
        let mut seq_arg = seq_len as u32;
        let mut kv_len_arg = seq_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        let use_w256 = tuning::prefill_flash_attention_hd512_w256_enabled();
        let kernel_name = if use_w256 {
            "rnb_attention_prefill_flash_hd512_w256"
        } else {
            "rnb_attention_prefill_flash_hd512"
        };
        let block_threads = if use_w256 { 256 } else { 512 };
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
            ],
            (seq_len as u32, num_heads as u32, 1),
            (block_threads, 1, 1),
        )?;

        Ok(Some(Q4kF16QkvHd512AttentionDevice {
            // cu19: host `Hidden` also uploads the raw hidden bytes into
            // `input_dev` above (the same buffer the device tail wants as the
            // residual source). Previously this returned None for host
            // `Hidden`, which made the dense tail re-H2D from a zeroed scratch
            // and broke the attention residual. `Normed` still returns None —
            // its `input_dev` holds post-norm activations, not raw hidden.
            hidden_dev: matches!(
                input,
                Q4kF16QkvInput::DeviceHidden { .. } | Q4kF16QkvInput::Hidden { .. }
            )
            .then_some(input_dev),
            attn_out_dev: output_dev,
            k_bits_dev,
            v_bits_dev,
            q_len,
            kv_len,
            q_bytes,
            kv_f16_bytes,
        }))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn q4k_f16_qkv_prefill_attention_hd512(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
        q_norm: &[f32],
        k_norm: &[f32],
        freq_factors: Option<&[f32]>,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
    ) -> Result<Option<(Vec<f32>, Vec<u16>, Vec<u16>)>, String> {
        let Some(device) = self.q4k_f16_qkv_prefill_attention_hd512_to_device(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            Q4kF16QkvInput::Normed(input),
            q_norm,
            k_norm,
            freq_factors,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )?
        else {
            return Ok(None);
        };

        let mut output = vec![0.0f32; device.q_len];
        let mut k_bits = vec![0u16; device.kv_len];
        let mut v_bits = vec![0u16; device.kv_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                device.attn_out_dev,
                device.q_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                k_bits.as_mut_ptr().cast::<libc::c_void>(),
                device.k_bits_dev,
                device.kv_f16_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v_bits.as_mut_ptr().cast::<libc::c_void>(),
                device.v_bits_dev,
                device.kv_f16_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Some((output, k_bits, v_bits)))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn q4k_f16_qkv_prefill_attention_hd512_dense_chain(
        &mut self,
        q_weights: &[u8],
        k_weights: &[u8],
        v_weights: &[u8],
        q_rows: usize,
        kv_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        hidden_input: &[f32],
        hidden_input_device: Option<(
            rnb_backend_api::DeviceTensorId,
            rnb_backend_api::DeviceTensorDesc,
        )>,
        attn_norm_weight: &[f32],
        q_norm: &[f32],
        k_norm: &[f32],
        freq_factors: Option<&[f32]>,
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        q_unit_offset: bool,
        k_unit_offset: bool,
        v_no_scale_norm: bool,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        ple_gate_weights: Option<&[u8]>,
        ple_proj_weights: Option<&[u8]>,
        ple_post_norm_weight: Option<&[f32]>,
        ple_input: Option<&[f32]>,
        ple_dim: usize,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        layer_out_scale: Option<&[f32]>,
        device_output_desc: Option<rnb_backend_api::DeviceTensorDesc>,
        unit_offset_attn_norm: bool,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<Option<(Vec<u16>, Vec<u16>, Q4kF16DenseChainOutput)>, String> {
        let expected_hidden = seq_len.saturating_mul(n_embd);
        if q_rows != o_cols || hidden.len() != expected_hidden {
            return Ok(None);
        }
        if let Some((_, desc)) = hidden_input_device.as_ref() {
            if desc.rows() != seq_len
                || desc.cols() != n_embd
                || desc.dtype() != rnb_backend_api::ScalarType::F32
            {
                return Ok(None);
            }
        } else if hidden_input.len() != expected_hidden {
            return Ok(None);
        }
        let qkv_input = match hidden_input_device {
            Some((id, desc)) => Q4kF16QkvInput::DeviceHidden {
                id,
                desc,
                attn_norm: attn_norm_weight,
                unit_offset: unit_offset_attn_norm,
            },
            None => Q4kF16QkvInput::Hidden {
                values: hidden_input,
                attn_norm: attn_norm_weight,
                unit_offset: unit_offset_attn_norm,
            },
        };
        let Some(device) = self.q4k_f16_qkv_prefill_attention_hd512_to_device(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            seq_len,
            qkv_input,
            q_norm,
            k_norm,
            freq_factors,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            norm_eps,
            q_unit_offset,
            k_unit_offset,
            v_no_scale_norm,
        )?
        else {
            return Ok(None);
        };

        let mut k_bits = vec![0u16; device.kv_len];
        let mut v_bits = vec![0u16; device.kv_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                k_bits.as_mut_ptr().cast::<libc::c_void>(),
                device.k_bits_dev,
                device.kv_f16_bytes,
                self.stream,
            )?;
            self.api.memcpy_dtoh_async(
                v_bits.as_mut_ptr().cast::<libc::c_void>(),
                device.v_bits_dev,
                device.kv_f16_bytes,
                self.stream,
            )?;
        }
        // The dense tail reuses and may grow the same scratch slabs that hold
        // k_bits_dev/v_bits_dev. Close the async D2H lifetime before those
        // slabs can be reallocated or overwritten.
        self.stream_synchronize()?;

        let output_id = self
            .dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
                o_weights,
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                post_attn_norm_weight,
                ffn_norm_weight,
                post_ffn_norm_weight,
                ple_gate_weights,
                ple_proj_weights,
                ple_post_norm_weight,
                ple_input,
                ple_dim,
                o_cols,
                n_ff,
                n_embd,
                seq_len,
                hidden,
                device.hidden_dev,
                device.attn_out_dev,
                device_output_desc,
                layer_out_scale,
                norm_eps,
                unit_offset_post_attn_norm,
                unit_offset_ffn_norm,
                unit_offset_post_ffn_norm,
            )?;
        Ok(Some((
            k_bits,
            v_bits,
            match output_id {
                Some(id) => Q4kF16DenseChainOutput::Device(id),
                None => Q4kF16DenseChainOutput::Host,
            },
        )))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn q4k_f16_q_prefill_attention_hd512_cached_f16kv_dense_chain(
        &mut self,
        q_weights: &[u8],
        q_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        kv_len: usize,
        hidden_input: &[f32],
        hidden_input_device: Option<(
            rnb_backend_api::DeviceTensorId,
            rnb_backend_api::DeviceTensorDesc,
        )>,
        attn_norm_weight: &[f32],
        q_norm: &[f32],
        freq_factors: Option<&[f32]>,
        cached_k_f16: &[u16],
        cached_v_f16: &[u16],
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        q_unit_offset: bool,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        ple_gate_weights: Option<&[u8]>,
        ple_proj_weights: Option<&[u8]>,
        ple_post_norm_weight: Option<&[f32]>,
        ple_input: Option<&[f32]>,
        ple_dim: usize,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        layer_out_scale: Option<&[f32]>,
        device_output_desc: Option<rnb_backend_api::DeviceTensorDesc>,
        unit_offset_attn_norm: bool,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<Option<Q4kF16DenseChainOutput>, String> {
        let head_dim = 512usize;
        let expected_hidden = seq_len.saturating_mul(n_embd);
        if q_rows != num_heads.saturating_mul(head_dim)
            || q_rows != o_cols
            || q_norm.len() != head_dim
            || hidden.len() != expected_hidden
        {
            return Ok(None);
        }
        if let Some((_, desc)) = hidden_input_device {
            if desc.rows() != seq_len
                || desc.cols() != n_embd
                || desc.dtype() != rnb_backend_api::ScalarType::F32
            {
                return Ok(None);
            }
        } else if hidden_input.len() != expected_hidden {
            return Ok(None);
        }
        if let Some(freq_factors) = freq_factors {
            if freq_factors.len() < head_dim / 2 {
                return Ok(None);
            }
        }
        let cols = blocks_per_row.checked_mul(256).ok_or_else(|| {
            format!("Q4_K F16 Q cached attention cols overflow: blocks={blocks_per_row}")
        })?;
        if cols != n_embd || attn_norm_weight.len() != cols {
            return Ok(None);
        }
        let expected_kv = kv_len
            .checked_mul(num_kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 Q cached attention KV length overflow: kv_len={kv_len} kv_heads={num_kv_heads}"
                )
        })?;
        if cached_k_f16.len() != expected_kv || cached_v_f16.len() != expected_kv {
            return Ok(None);
        }
        let row_bytes = blocks_per_row.checked_mul(144).ok_or_else(|| {
            format!("Q4_K F16 Q cached attention row byte overflow: blocks={blocks_per_row}")
        })?;
        let q_expected = q_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("Q4_K F16 Q cached attention q byte overflow: rows={q_rows}"))?;
        if q_weights.len() != q_expected {
            return Err(format!(
                "Q4_K F16 Q cached attention q byte mismatch: got {}, expected {q_expected}",
                q_weights.len()
            ));
        }
        if !tuning::prefill_q4k_f16_qkv_gemm_enabled() {
            return Ok(None);
        }

        let Some(q_weights_dev) = self.resident_q4k_f16_ptr(q_weights, q_rows, blocks_per_row)?
        else {
            return Ok(None);
        };

        let hidden_bytes = expected_hidden
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 Q cached attention hidden byte overflow: len={expected_hidden}")
            })?;
        let input_f16_bytes = expected_hidden
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 Q cached attention input f16 byte overflow: len={}",
                    expected_hidden
                )
            })?;
        let q_len = seq_len.checked_mul(q_rows).ok_or_else(|| {
            format!("Q4_K F16 Q cached attention q length overflow: seq_len={seq_len}")
        })?;
        let q_bytes = q_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("Q4_K F16 Q cached attention q byte overflow: len={q_len}"))?;
        let kv_f16_bytes = expected_kv
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!("Q4_K F16 Q cached attention KV f16 byte overflow: len={expected_kv}")
            })?;
        let kv_f32_bytes = expected_kv
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 Q cached attention KV f32 byte overflow: len={expected_kv}")
            })?;
        let q_norm_bytes = std::mem::size_of_val(q_norm);
        let cached_rope_table = if freq_factors.is_none() {
            Some(self.rope_table_ptrs(head_dim, seq_len, pos_start, rope_theta)?)
        } else {
            None
        };
        let mut rope_sin = Vec::new();
        let mut rope_cos = Vec::new();
        if let Some(freq_factors) = freq_factors {
            let inv_freq = (0..head_dim / 2)
                .map(|i| {
                    1.0f32 / (rope_theta.powf((2 * i) as f32 / head_dim as f32) * freq_factors[i])
                })
                .collect::<Vec<_>>();
            let rope_table_len = seq_len.checked_mul(head_dim / 2).ok_or_else(|| {
                format!("Q4_K F16 Q cached attention rope table overflow: seq_len={seq_len}")
            })?;
            rope_sin.reserve_exact(rope_table_len);
            rope_cos.reserve_exact(rope_table_len);
            for token in 0..seq_len {
                let pos = (pos_start + token) as f32;
                for &freq in &inv_freq {
                    let (sin_a, cos_a) = (pos * freq).sin_cos();
                    rope_sin.push(sin_a);
                    rope_cos.push(cos_a);
                }
            }
        }
        let rope_sin_bytes = std::mem::size_of_val(rope_sin.as_slice());
        let rope_cos_bytes = std::mem::size_of_val(rope_cos.as_slice());
        let norm_bytes = q_norm_bytes
            .checked_add(rope_sin_bytes)
            .and_then(|n| n.checked_add(rope_cos_bytes))
            .ok_or_else(|| "Q4_K F16 Q cached attention norm byte overflow".to_string())?;

        let input_dev = if let Some((input_id, input_desc)) = hidden_input_device {
            self.device_tensor_ptr(input_id, input_desc)?
        } else {
            self.compute_input_ptr(hidden_bytes)?
        };
        let input_f16_dev = self.compute_aux_output_ptr(input_f16_bytes)?;
        let q_raw_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_bits_dev = self.compute_mid_b_ptr(kv_f16_bytes)?;
        let v_bits_dev = self.compute_weights_ptr(kv_f16_bytes)?;
        let q_post_dev = self.compute_full_gate_ptr(hidden_bytes.max(q_bytes))?;
        let k_f32_dev = self.compute_gate_ptrs_ptr(kv_f32_bytes)?;
        let v_f32_dev = self.compute_up_ptrs_ptr(kv_f32_bytes)?;
        let output_dev = self.compute_output_ptr(q_bytes)?;
        let norm_dev = self.compute_temp_slab_ptr(norm_bytes)?;
        let q_norm_dev = norm_dev;
        let rope_sin_dev = cached_rope_table
            .map(|table| table.sin_ptr)
            .unwrap_or(q_norm_dev + q_norm_bytes as u64);
        let rope_cos_dev = cached_rope_table
            .map(|table| table.cos_ptr)
            .unwrap_or(rope_sin_dev + rope_sin_bytes as u64);
        unsafe {
            if hidden_input_device.is_none() {
                self.api.memcpy_htod_async(
                    input_dev,
                    hidden_input.as_ptr().cast::<libc::c_void>(),
                    hidden_bytes,
                    self.stream,
                )?;
            }
            self.api.memcpy_htod_async(
                q_norm_dev,
                q_norm.as_ptr().cast::<libc::c_void>(),
                q_norm_bytes,
                self.stream,
            )?;
            if cached_rope_table.is_none() {
                self.api.memcpy_htod_async(
                    rope_sin_dev,
                    rope_sin.as_ptr().cast::<libc::c_void>(),
                    rope_sin_bytes,
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    rope_cos_dev,
                    rope_cos.as_ptr().cast::<libc::c_void>(),
                    rope_cos_bytes,
                    self.stream,
                )?;
            }
            self.api.memcpy_htod_async(
                k_bits_dev,
                cached_k_f16.as_ptr().cast::<libc::c_void>(),
                kv_f16_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                v_bits_dev,
                cached_v_f16.as_ptr().cast::<libc::c_void>(),
                kv_f16_bytes,
                self.stream,
            )?;
        }

        let attn_norm_dev = self.resident_f32_ptr(attn_norm_weight)?;
        self.launch_rms_norm_rows_f32(
            input_dev,
            attn_norm_dev,
            q_post_dev,
            norm_eps,
            seq_len,
            cols,
            unit_offset_attn_norm,
        )?;
        self.launch_f32_to_f16(q_post_dev, input_f16_dev, expected_hidden)?;
        self.hgemm_to_f32_device(
            q_weights_dev,
            q_rows,
            cols,
            input_f16_dev,
            seq_len,
            q_raw_dev,
        )?;
        self.launch_q_norm_rope_neox_hd512(
            q_raw_dev,
            q_norm_dev,
            rope_sin_dev,
            rope_cos_dev,
            q_post_dev,
            norm_eps,
            seq_len,
            num_heads,
            pos_start,
            q_unit_offset,
        )?;
        self.launch_f16_to_f32(k_bits_dev, k_f32_dev, expected_kv)?;
        self.launch_f16_to_f32(v_bits_dev, v_f32_dev, expected_kv)?;

        self.launch_attention_prefill_flash_hd512_kernel(
            output_dev,
            q_post_dev,
            k_f32_dev,
            v_f32_dev,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            None,
        )?;

        let output_id = self
            .dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
                o_weights,
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                post_attn_norm_weight,
                ffn_norm_weight,
                post_ffn_norm_weight,
                ple_gate_weights,
                ple_proj_weights,
                ple_post_norm_weight,
                ple_input,
                ple_dim,
                o_cols,
                n_ff,
                n_embd,
                seq_len,
                hidden,
                Some(input_dev),
                output_dev,
                device_output_desc,
                layer_out_scale,
                norm_eps,
                unit_offset_post_attn_norm,
                unit_offset_ffn_norm,
                unit_offset_post_ffn_norm,
            )?;
        Ok(Some(match output_id {
            Some(id) => Q4kF16DenseChainOutput::Device(id),
            None => Q4kF16DenseChainOutput::Host,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn q4k_f16_q_prefill_attention_hd256_cached_f16kv_window_dense_chain(
        &mut self,
        q_weights: &[u8],
        q_rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        kv_len: usize,
        hidden_input: &[f32],
        hidden_input_device: Option<(
            rnb_backend_api::DeviceTensorId,
            rnb_backend_api::DeviceTensorDesc,
        )>,
        attn_norm_weight: &[f32],
        q_norm: &[f32],
        cached_k_f16: &[u16],
        cached_v_f16: &[u16],
        num_heads: usize,
        num_kv_heads: usize,
        scale: f32,
        rope_theta: f32,
        pos_start: usize,
        norm_eps: f32,
        q_unit_offset: bool,
        window: usize,
        o_weights: &[u8],
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: u32,
        post_attn_norm_weight: Option<&[f32]>,
        ffn_norm_weight: &[f32],
        post_ffn_norm_weight: Option<&[f32]>,
        ple_gate_weights: Option<&[u8]>,
        ple_proj_weights: Option<&[u8]>,
        ple_post_norm_weight: Option<&[f32]>,
        ple_input: Option<&[f32]>,
        ple_dim: usize,
        o_cols: usize,
        n_ff: usize,
        n_embd: usize,
        hidden: &mut [f32],
        layer_out_scale: Option<&[f32]>,
        device_output_desc: Option<rnb_backend_api::DeviceTensorDesc>,
        unit_offset_attn_norm: bool,
        unit_offset_post_attn_norm: bool,
        unit_offset_ffn_norm: bool,
        unit_offset_post_ffn_norm: bool,
    ) -> Result<Option<Q4kF16DenseChainOutput>, String> {
        let head_dim = 256usize;
        let expected_hidden = seq_len.saturating_mul(n_embd);
        if window == 0
            || q_rows != num_heads.saturating_mul(head_dim)
            || q_rows != o_cols
            || q_norm.len() != head_dim
            || hidden.len() != expected_hidden
        {
            return Ok(None);
        }
        if let Some((_, desc)) = hidden_input_device {
            if desc.rows() != seq_len
                || desc.cols() != n_embd
                || desc.dtype() != rnb_backend_api::ScalarType::F32
            {
                return Ok(None);
            }
        } else if hidden_input.len() != expected_hidden {
            return Ok(None);
        }
        let cols = blocks_per_row.checked_mul(256).ok_or_else(|| {
            format!("Q4_K F16 Q hd256 cached attention cols overflow: blocks={blocks_per_row}")
        })?;
        if cols != n_embd || attn_norm_weight.len() != cols {
            return Ok(None);
        }
        let expected_kv = kv_len
            .checked_mul(num_kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 Q hd256 cached attention KV length overflow: kv_len={kv_len} kv_heads={num_kv_heads}"
                )
        })?;
        if cached_k_f16.len() != expected_kv || cached_v_f16.len() != expected_kv {
            return Ok(None);
        }
        let row_bytes = blocks_per_row.checked_mul(144).ok_or_else(|| {
            format!("Q4_K F16 Q hd256 cached attention row byte overflow: blocks={blocks_per_row}")
        })?;
        let q_expected = q_rows.checked_mul(row_bytes).ok_or_else(|| {
            format!("Q4_K F16 Q hd256 cached attention q byte overflow: rows={q_rows}")
        })?;
        if q_weights.len() != q_expected {
            return Err(format!(
                "Q4_K F16 Q hd256 cached attention q byte mismatch: got {}, expected {q_expected}",
                q_weights.len()
            ));
        }
        if !tuning::prefill_q4k_f16_qkv_gemm_enabled() {
            return Ok(None);
        }

        let Some(q_weights_dev) = self.resident_q4k_f16_ptr(q_weights, q_rows, blocks_per_row)?
        else {
            return Ok(None);
        };

        let hidden_bytes = expected_hidden
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 Q hd256 cached attention hidden byte overflow: len={expected_hidden}"
                )
            })?;
        let input_f16_bytes = expected_hidden
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!(
                    "Q4_K F16 Q hd256 cached attention input f16 byte overflow: len={}",
                    expected_hidden
                )
            })?;
        let q_len = seq_len.checked_mul(q_rows).ok_or_else(|| {
            format!("Q4_K F16 Q hd256 cached attention q length overflow: seq_len={seq_len}")
        })?;
        let q_bytes = q_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 Q hd256 cached attention q byte overflow: len={q_len}")
            })?;
        let kv_f16_bytes = expected_kv
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                format!("Q4_K F16 Q hd256 cached attention KV f16 byte overflow: len={expected_kv}")
            })?;
        let kv_f32_bytes = expected_kv
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                format!("Q4_K F16 Q hd256 cached attention KV f32 byte overflow: len={expected_kv}")
            })?;
        let q_norm_bytes = std::mem::size_of_val(q_norm);
        let rope_table = self.rope_table_ptrs(head_dim, seq_len, pos_start, rope_theta)?;

        let input_dev = if let Some((input_id, input_desc)) = hidden_input_device {
            self.device_tensor_ptr(input_id, input_desc)?
        } else {
            self.compute_input_ptr(hidden_bytes)?
        };
        let input_f16_dev = self.compute_aux_output_ptr(input_f16_bytes)?;
        let q_raw_dev = self.compute_mid_a_ptr(q_bytes)?;
        let k_bits_dev = self.compute_mid_b_ptr(kv_f16_bytes)?;
        let v_bits_dev = self.compute_weights_ptr(kv_f16_bytes)?;
        let q_post_dev = self.compute_full_gate_ptr(hidden_bytes.max(q_bytes))?;
        let k_f32_dev = self.compute_gate_ptrs_ptr(kv_f32_bytes)?;
        let v_f32_dev = self.compute_up_ptrs_ptr(kv_f32_bytes)?;
        let output_dev = self.compute_output_ptr(q_bytes)?;
        let q_norm_dev = self.compute_temp_slab_ptr(q_norm_bytes)?;
        unsafe {
            if hidden_input_device.is_none() {
                self.api.memcpy_htod_async(
                    input_dev,
                    hidden_input.as_ptr().cast::<libc::c_void>(),
                    hidden_bytes,
                    self.stream,
                )?;
            }
            self.api.memcpy_htod_async(
                q_norm_dev,
                q_norm.as_ptr().cast::<libc::c_void>(),
                q_norm_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                k_bits_dev,
                cached_k_f16.as_ptr().cast::<libc::c_void>(),
                kv_f16_bytes,
                self.stream,
            )?;
            self.api.memcpy_htod_async(
                v_bits_dev,
                cached_v_f16.as_ptr().cast::<libc::c_void>(),
                kv_f16_bytes,
                self.stream,
            )?;
        }

        let attn_norm_dev = self.resident_f32_ptr(attn_norm_weight)?;
        self.launch_rms_norm_rows_f32(
            input_dev,
            attn_norm_dev,
            q_post_dev,
            norm_eps,
            seq_len,
            cols,
            unit_offset_attn_norm,
        )?;
        self.launch_f32_to_f16(q_post_dev, input_f16_dev, expected_hidden)?;
        self.hgemm_to_f32_device(
            q_weights_dev,
            q_rows,
            cols,
            input_f16_dev,
            seq_len,
            q_raw_dev,
        )?;
        self.launch_q_norm_rope_neox_hd256(
            q_raw_dev,
            q_norm_dev,
            rope_table.sin_ptr,
            rope_table.cos_ptr,
            q_post_dev,
            norm_eps,
            seq_len,
            num_heads,
            pos_start,
            q_unit_offset,
        )?;
        self.launch_f16_to_f32(k_bits_dev, k_f32_dev, expected_kv)?;
        self.launch_f16_to_f32(v_bits_dev, v_f32_dev, expected_kv)?;

        let mut output_arg = output_dev;
        let mut q_arg = q_post_dev;
        let mut k_arg = k_f32_dev;
        let mut v_arg = v_f32_dev;
        let mut seq_arg = seq_len as u32;
        let mut kv_len_arg = kv_len as u32;
        let mut heads_arg = num_heads as u32;
        let mut kv_heads_arg = num_kv_heads as u32;
        let mut scale_arg = scale;
        let mut window_arg = window as u32;
        self.launch_cached_gemv(
            "rnb_attention_prefill_flash_hd256_window",
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut q_arg as *mut u64).cast::<libc::c_void>(),
                (&mut k_arg as *mut u64).cast::<libc::c_void>(),
                (&mut v_arg as *mut u64).cast::<libc::c_void>(),
                (&mut seq_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_len_arg as *mut u32).cast::<libc::c_void>(),
                (&mut heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut kv_heads_arg as *mut u32).cast::<libc::c_void>(),
                (&mut scale_arg as *mut f32).cast::<libc::c_void>(),
                (&mut window_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (seq_len as u32, num_heads as u32, 1),
            (256, 1, 1),
        )?;

        let output_id = self
            .dense_q4k_attention_output_gelu_ffn_batch_norm_residual_from_attn_dev(
                o_weights,
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                post_attn_norm_weight,
                ffn_norm_weight,
                post_ffn_norm_weight,
                ple_gate_weights,
                ple_proj_weights,
                ple_post_norm_weight,
                ple_input,
                ple_dim,
                o_cols,
                n_ff,
                n_embd,
                seq_len,
                hidden,
                Some(input_dev),
                output_dev,
                device_output_desc,
                layer_out_scale,
                norm_eps,
                unit_offset_post_attn_norm,
                unit_offset_ffn_norm,
                unit_offset_post_ffn_norm,
            )?;
        Ok(Some(match output_id {
            Some(id) => Q4kF16DenseChainOutput::Device(id),
            None => Q4kF16DenseChainOutput::Host,
        }))
    }

    pub(super) fn gemv_batch(
        &mut self,
        kernel_name: &str,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let output_len = seq_len * rows;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        if kernel_name == "rnb_q4k_gemv_batch"
            && tuning::q4k_prefill_f32_gemm_enabled()
            && seq_len <= 32
            && rows >= 1024
            && blocks_per_row >= 4
        {
            if let Some(weights_dev) = self.resident_q4k_f32_ptr(weights, rows, blocks_per_row)? {
                let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
                unsafe {
                    self.api.memcpy_htod_async(
                        input_dev,
                        input.as_ptr().cast::<libc::c_void>(),
                        std::mem::size_of_val(input),
                        self.stream,
                    )?;
                }
                self.sgemm_device(
                    weights_dev,
                    rows,
                    blocks_per_row * 256,
                    input_dev,
                    seq_len,
                    output_dev,
                )?;
                let mut output = vec![0.0f32; output_len];
                unsafe {
                    self.api.memcpy_dtoh_async(
                        output.as_mut_ptr().cast::<libc::c_void>(),
                        output_dev,
                        output_bytes,
                        self.stream,
                    )?;
                }
                self.stream_synchronize()?;
                return Ok(output);
            }
        }
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;

        // cu39 Phase 5: mma_4warp path on real dominant Q4K prefill dispatcher.
        // gemv_batch 가 prefill 의 진짜 dominant Q4K matmul path (nsys: 39.9% GPU time
        // for `rnb_q4k_gemv_batch_q8dot_seq4_warp8`). cu39 phase 1-4 의 새 dispatcher
        // (`q4k_f16_qkv_gemm_batch`) 는 호출 안 됨 → ABAB ε 측정 noise. 진짜 비교를
        // 위해 이곳에 mma kernel 직접 끼움.
        let use_mma_4warp_batch = std::env::var("RNB_CUDA_Q4K_MMA_4WARP_BATCH")
            .ok()
            .as_deref()
            == Some("1");
        if use_mma_4warp_batch && kernel_name == "rnb_q4k_gemv_batch" && seq_len >= 8 && rows >= 64
        {
            // GPU quantize Q8_1 with sum (host quantize_q8_1_batch_by_32 has no sum).
            let q8_len = seq_len * blocks_per_row * 256;
            let chunks = q8_len / 32;
            let qs_bytes = q8_len;
            let ds_bytes = chunks * std::mem::size_of::<f32>();
            let sums_bytes = chunks * std::mem::size_of::<f32>();
            let align = 256usize;
            let qs_off = 0usize;
            let ds_off = (qs_off + qs_bytes).next_multiple_of(align);
            let sums_off = (ds_off + ds_bytes).next_multiple_of(align);
            let total = (sums_off + sums_bytes).next_multiple_of(align);
            let slab = self.compute_temp_slab_ptr(total)?;
            let input_qs_dev = slab + qs_off as u64;
            let input_ds_dev = slab + ds_off as u64;
            let input_sums_dev = slab + sums_off as u64;
            unsafe {
                self.api.memcpy_htod_async(
                    input_dev,
                    input.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(input),
                    self.stream,
                )?;
            }
            self.launch_quantize_q8_1_with_sum_by_32(
                input_dev,
                input_qs_dev,
                input_ds_dev,
                input_sums_dev,
                q8_len,
            )?;
            self.launch_q4k_q8_1_matmul_mma_4warp(
                weights,
                rows,
                blocks_per_row,
                seq_len,
                input_qs_dev,
                input_ds_dev,
                input_sums_dev,
                output_dev,
            )?;
            let mut output = vec![0.0f32; output_len];
            unsafe {
                self.api.memcpy_dtoh_async(
                    output.as_mut_ptr().cast::<libc::c_void>(),
                    output_dev,
                    output_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            return Ok(output);
        }

        if kernel_name == "rnb_q4k_gemv_batch"
            && !tuning::q4k_batch_raw_seq4_enabled(seq_len, rows, blocks_per_row)
            && q4k_q8dot_prefill_enabled(seq_len <= 32 && rows >= 1024 && blocks_per_row >= 4)
        {
            let (qs, ds) = quantize_q8_1_batch_by_32(input, blocks_per_row, seq_len);
            let qs_dev = self.compute_input_ptr(qs.len())?;
            let ds_dev = self.compute_aux_output_ptr(std::mem::size_of_val(ds.as_slice()))?;
            unsafe {
                self.api.memcpy_htod_async(
                    qs_dev,
                    qs.as_ptr().cast::<libc::c_void>(),
                    qs.len(),
                    self.stream,
                )?;
                self.api.memcpy_htod_async(
                    ds_dev,
                    ds.as_ptr().cast::<libc::c_void>(),
                    std::mem::size_of_val(ds.as_slice()),
                    self.stream,
                )?;
            }
            let mut output_arg = output_dev;
            let mut weights_arg = weights_dev;
            let mut input_qs_arg = qs_dev;
            let mut input_ds_arg = ds_dev;
            let mut rows_arg = rows as u32;
            let mut blocks_per_row_arg = blocks_per_row as u32;
            let mut seq_len_arg = seq_len as u32;
            let use_seq4 = q4k_q8dot_prefill_seq4_enabled(seq_len > 1);
            let kernel = if use_seq4 {
                "rnb_q4k_gemv_batch_q8dot_seq4_warp8"
            } else {
                "rnb_q4k_gemv_batch_q8dot_warp8"
            };
            let grid_y = if use_seq4 {
                seq_len.div_ceil(4) as u32
            } else {
                seq_len as u32
            };
            self.launch_cached_gemv(
                kernel,
                &[
                    (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_qs_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut input_ds_arg as *mut u64).cast::<libc::c_void>(),
                    (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                    (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
                ],
                (rows.div_ceil(8) as u32, grid_y, 1),
                (256, 1, 1),
            )?;
            let mut output = vec![0.0f32; output_len];
            unsafe {
                self.api.memcpy_dtoh_async(
                    output.as_mut_ptr().cast::<libc::c_void>(),
                    output_dev,
                    output_bytes,
                    self.stream,
                )?;
            }
            self.stream_synchronize()?;
            return Ok(output);
        }
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        let (kernel_name, grid, block) = if kernel_name == "rnb_q4k_gemv_batch"
            && tuning::q4k_batch_raw_seq4_enabled(seq_len, rows, blocks_per_row)
        {
            (
                "rnb_q4k_gemv_batch_seq4_warp8",
                (rows.div_ceil(8) as u32, seq_len.div_ceil(4) as u32, 1),
                (256, 1, 1),
            )
        } else if kernel_name == "rnb_q4k_gemv_batch" && tuning::q4k_gemv_batch_warp8_enabled() {
            (
                "rnb_q4k_gemv_batch_warp8",
                (rows.div_ceil(8) as u32, seq_len as u32, 1),
                (256, 1, 1),
            )
        } else if kernel_name == "rnb_iq4_xs_gemv_batch_warp8" {
            (
                kernel_name,
                (rows.div_ceil(8) as u32, seq_len as u32, 1),
                (256, 1, 1),
            )
        } else if kernel_name == "rnb_q6k_gemv_batch" && tuning::q6k_gemv_batch_warp8_enabled() {
            (
                "rnb_q6k_gemv_batch_warp8",
                (rows.div_ceil(8) as u32, seq_len as u32, 1),
                (256, 1, 1),
            )
        } else {
            (kernel_name, (rows as u32, seq_len as u32, 1), (256, 1, 1))
        };
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            grid,
            block,
        )?;
        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }

    pub(super) fn gemv_batch_seq32(
        &mut self,
        kernel_name: &str,
        weights: &[u8],
        rows: usize,
        blocks_per_row: usize,
        seq_len: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let weights_dev = self.resident_q4k_weights_ptr(weights)?;
        let input_dev = self.compute_input_ptr(std::mem::size_of_val(input))?;
        let output_len = seq_len * rows;
        let output_bytes = output_len * std::mem::size_of::<f32>();
        let output_dev = self.compute_output_ptr(output_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                input.as_ptr().cast::<libc::c_void>(),
                std::mem::size_of_val(input),
                self.stream,
            )?;
        }
        let mut output_arg = output_dev;
        let mut weights_arg = weights_dev;
        let mut input_arg = input_dev;
        let mut rows_arg = rows as u32;
        let mut blocks_per_row_arg = blocks_per_row as u32;
        let mut seq_len_arg = seq_len as u32;
        self.launch_cached_gemv(
            kernel_name,
            &[
                (&mut output_arg as *mut u64).cast::<libc::c_void>(),
                (&mut weights_arg as *mut u64).cast::<libc::c_void>(),
                (&mut input_arg as *mut u64).cast::<libc::c_void>(),
                (&mut rows_arg as *mut u32).cast::<libc::c_void>(),
                (&mut blocks_per_row_arg as *mut u32).cast::<libc::c_void>(),
                (&mut seq_len_arg as *mut u32).cast::<libc::c_void>(),
            ],
            (rows as u32, 1, 1),
            (256, 1, 1),
        )?;
        let mut output = vec![0.0f32; output_len];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast::<libc::c_void>(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(output)
    }
}
