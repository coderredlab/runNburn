#[cfg(feature = "cuda")]
use super::backend_runtime::{
    bf16_gemv as cuda_bf16_gemv, cuda_f16_gemv, decode_gemv_cuda as cuda_decode_gemv,
    prefill_gemv_cuda,
};
use super::cpu_runtime::kernels;
use super::dense_dispatch::{gemv_bf16, gemv_f16, gemv_f32};
use super::gemv_profile::GemvProfileGuard;
use super::policy;
use super::quantized_dispatch::force_generic_gemv;
#[cfg(target_arch = "aarch64")]
use super::quantized_dispatch::{
    dispatch_into_fast_gemv, dispatch_q8_gemv, dispatch_q8k_gemv, dispatch_vec_fast_gemv,
    gemv_q8k_profile_method, QuantizedQ8Block as Q8Block, QuantizedQ8KBlock as Q8KBlock,
};
use super::quantized_weight_types::QuantizedWeight;
use super::scalar_gemv::{
    gemv_full_dequant_f32, gemv_generic, gemv_output_f64_logit, gemv_q4_0, gemv_q8_0,
};
use rnb_core::tensor::Tensor;
use rnb_loader::GGMLType;

pub(super) fn force_generic_attn_qkv_layer(layer_idx: usize) -> bool {
    policy::env_layer_matches("RNB_FORCE_GENERIC_ATTN_QKV_LAYER", layer_idx)
}

/// mt94 — every quantized GEMV (Q4_K, Q5_K, Q6_K, Q4_0, Q8_0, …) takes the
/// per-row `dequantize_bytes_to_f32` + f32×f32 dot path. Used to isolate
/// whether Q4_K cumulative weight-quantization drift is the root cause of the
/// L17/L41 softmax_output cos collapse vs forward-path-internal accumulation
/// (norm chain, residual). Dev-only env; production unaffected.
fn all_f32_override_enabled() -> bool {
    std::env::var("RNB_ALL_F32_OVERRIDE")
        .map(|v| v != "0")
        .unwrap_or(false)
}

fn is_quantized_type(t: GGMLType) -> bool {
    !matches!(t, GGMLType::F32 | GGMLType::F16 | GGMLType::BF16)
}

pub(super) fn force_generic_attn_oproj_layer(layer_idx: usize) -> bool {
    policy::env_layer_matches("RNB_FORCE_GENERIC_ATTN_OPROJ_LAYER", layer_idx)
}

pub(super) fn gdn_qkv_weight_name(layer_idx: usize) -> String {
    format!("blk.{layer_idx}.attn_qkv.weight")
}

pub(super) fn gdn_gate_weight_name(layer_idx: usize) -> String {
    format!("blk.{layer_idx}.attn_gate.weight")
}

impl QuantizedWeight {
    /// 양자화 weight × F32 input → F32 output (Vec, zero-copy)
    pub(super) fn gemv_vec(&self, input: &[f32]) -> crate::error::Result<Vec<f32>> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let seq_len = input.len() / self.cols;
        let _profile_guard =
            GemvProfileGuard::new("gemv_vec", self.ggml_type, seq_len, self.rows, self.cols);
        let bytes_per_row = bytes.len() / self.rows;
        let mut output = vec![0.0f32; seq_len * self.rows];

        // mt94 — `RNB_ALL_F32_OVERRIDE=1` forces every quantized GEMV onto the
        // per-row `dequantize_bytes_to_f32` + pure f32×f32 dot path, bypassing
        // all production fast paths (CUDA, NEON Q4×Q8K, quant_gemv). F32/F16/BF16
        // weights are lossless to begin with so the override skips them.
        if all_f32_override_enabled() && is_quantized_type(self.ggml_type) {
            gemv_full_dequant_f32(
                bytes,
                input,
                &mut output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
                self.ggml_type,
            );
            return Ok(output);
        }

        #[cfg(feature = "cuda")]
        if let Some(result) = prefill_gemv_cuda(self, bytes, input, seq_len) {
            return result;
        }

        #[cfg(feature = "cuda")]
        if seq_len == 1 {
            if let Some(result) = cuda_decode_gemv(self, bytes, input) {
                return result;
            }
        }

        if force_generic_gemv(self.rows, self.cols) {
            gemv_generic(
                bytes,
                input,
                &mut output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
                self.ggml_type,
            );
            return Ok(output);
        }

        // Fast path: F32 weight — interpret bytes directly as &[f32], skip row-by-row dequant.
        // Used by Gemma4 PLE `inp_gate` / `proj` which are stored F32 in the GGUF.
        if self.ggml_type == GGMLType::F32 {
            let weight_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            gemv_f32(
                weight_f32,
                input,
                &mut output,
                self.rows,
                self.cols,
                seq_len,
            );
            return Ok(output);
        }

        // Fast path: F16 weight — used by external Qwen3.5 MTP sidecar
        // weights. Avoids generic per-row dequant dispatch overhead.
        if self.ggml_type == GGMLType::F16 {
            #[cfg(feature = "cuda")]
            if seq_len == 1 {
                if let Some(result) = cuda_f16_gemv(self, input)? {
                    return Ok(result);
                }
            }
            let weight_u16: &[u16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2)
            };
            gemv_f16(
                weight_u16,
                input,
                &mut output,
                self.rows,
                self.cols,
                seq_len,
            );
            return Ok(output);
        }

        // Fast path: BF16 weight — per-row bf16→f32 convert on stack then NEON f32 dot.
        // Used by Gemma4 `per_layer_model_proj` (BF16 in Q4_K_M GGUF).
        if self.ggml_type == GGMLType::BF16 {
            #[cfg(feature = "cuda")]
            if seq_len == 1 {
                if let Some(result) = cuda_bf16_gemv(self, input)? {
                    return Ok(result);
                }
            }
            let weight_u16: &[u16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2)
            };
            gemv_bf16(
                weight_u16,
                input,
                &mut output,
                self.rows,
                self.cols,
                seq_len,
            );
            return Ok(output);
        }

        #[cfg(target_arch = "aarch64")]
        if dispatch_vec_fast_gemv(self, bytes, input, &mut output, seq_len, bytes_per_row)? {
            return Ok(output);
        }

        match self.ggml_type {
            GGMLType::Q4_0 => gemv_q4_0(
                bytes,
                input,
                &mut output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
            ),
            GGMLType::Q8_0 => gemv_q8_0(
                bytes,
                input,
                &mut output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
            ),
            _ => gemv_generic(
                bytes,
                input,
                &mut output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
                self.ggml_type,
            ),
        }

        Ok(output)
    }

    /// 양자화 weight × F32 Tensor → F32 Tensor
    /// input: [seq_len, in_features], output: [seq_len, out_features]
    pub(super) fn gemv(&self, input: &Tensor) -> crate::error::Result<Tensor> {
        let x_data = kernels::tensor_as_f32_slice(input);
        let seq_len = x_data.len() / self.cols;
        let output = self.gemv_vec(x_data)?;
        Ok(Tensor::from_vec(output, &[seq_len, self.rows]))
    }

    /// mt94 axis — Tensor-returning wrapper around
    /// [`Self::gemv_into_full_dequant_f32`]. Caller-opt-in path; only invoked
    /// from `engine::forward::projection` when `RNB_QK_F32_OVERRIDE=1`.
    pub(super) fn gemv_full_dequant_f32(&self, input: &Tensor) -> crate::error::Result<Tensor> {
        let x_data = kernels::tensor_as_f32_slice(input);
        let seq_len = x_data.len() / self.cols;
        let mut output = vec![0.0f32; seq_len * self.rows];
        self.gemv_into_full_dequant_f32(x_data, &mut output)?;
        Ok(Tensor::from_vec(output, &[seq_len, self.rows]))
    }

    /// NEON int8 gemv with pre-quantized Q8K input (avoids re-quantization for K-quant types).
    #[cfg(target_arch = "aarch64")]
    pub(super) fn gemv_vec_q8k(&self, q8k: &[Q8KBlock]) -> crate::error::Result<Vec<f32>> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let bytes_per_row = bytes.len() / self.rows;
        let n_blocks = self.cols / 256;
        let seq_len = q8k.len() / n_blocks;
        let _profile_guard = GemvProfileGuard::new(
            gemv_q8k_profile_method(self.packed_gemm_quant_type),
            self.ggml_type,
            seq_len,
            self.rows,
            self.cols,
        );
        let mut output = vec![0.0f32; seq_len * self.rows];

        dispatch_q8k_gemv(self, bytes, q8k, &mut output, seq_len, bytes_per_row)?;
        Ok(output)
    }

    /// NEON int8 gemv with pre-quantized Q8 input (avoids re-quantization).
    /// Falls back to standard gemv_vec if q4_0 data not available.
    #[cfg(target_arch = "aarch64")]
    pub(super) fn gemv_vec_q8(&self, q8: &[Q8Block]) -> crate::error::Result<Vec<f32>> {
        if self.q4_0_data.is_some() || matches!(self.ggml_type, GGMLType::Q4_0 | GGMLType::Q8_0) {
            let seq_len = q8.len() / (self.cols / 32);
            let mut output = vec![0.0f32; seq_len * self.rows];
            let bytes = self.data.as_bytes().ok_or_else(|| {
                crate::error::LlmError::Forward("quantized weight: no bytes".into())
            })?;
            let bytes_per_row = bytes.len() / self.rows;
            dispatch_q8_gemv(self, bytes, q8, &mut output, seq_len, bytes_per_row)?;
            return Ok(output);
        }
        // Fallback: reconstruct f32 input from Q8 blocks and use generic path
        let n_blocks = q8.len();
        let mut input = vec![0.0f32; n_blocks * 32];
        for (bi, blk) in q8.iter().enumerate() {
            for i in 0..32 {
                input[bi * 32 + i] = blk.qs[i] as f32 * blk.d;
            }
        }
        self.gemv_vec(&input)
    }

    /// GEMV into pre-allocated buffer (avoids Vec allocation).
    #[cfg(target_arch = "aarch64")]
    pub(super) fn gemv_into_q8(
        &self,
        q8: &[Q8Block],
        output: &mut [f32],
    ) -> crate::error::Result<()> {
        let seq_len = q8.len() / (self.cols / 32);
        output[..seq_len * self.rows].fill(0.0);

        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let bytes_per_row = bytes.len() / self.rows;
        dispatch_q8_gemv(self, bytes, q8, output, seq_len, bytes_per_row)
    }

    /// GEMV into pre-allocated buffer with pre-quantized Q8K input (avoids Vec allocation).
    #[cfg(target_arch = "aarch64")]
    pub(super) fn gemv_into_q8k(
        &self,
        q8k: &[Q8KBlock],
        output: &mut [f32],
    ) -> crate::error::Result<()> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let bytes_per_row = bytes.len() / self.rows;
        let n_blocks = self.cols / 256;
        let seq_len = q8k.len() / n_blocks;
        let _profile_guard = GemvProfileGuard::new(
            "gemv_into_q8k",
            self.ggml_type,
            seq_len,
            self.rows,
            self.cols,
        );
        let required = seq_len * self.rows;
        assert!(
            output.len() >= required,
            "gemv_into_q8k: output buffer size mismatch"
        );
        let output = &mut output[..required];
        output.fill(0.0);

        dispatch_q8k_gemv(self, bytes, q8k, output, seq_len, bytes_per_row)
    }

    /// GEMV with f64 accumulator, intended for the final output projection
    /// (`hidden → vocab logits`). Opt-in via `RNB_OUTPUT_F64_LOGIT=1`. Bypasses
    /// CUDA / AArch64 fast paths and dequantizes per row, reducing in f64 to
    /// preserve top-1 vs top-2 ranking margins (mc70 finding: f32 reduce
    /// drift can flip 0.1-0.5 logit margins).
    pub(super) fn gemv_into_f64_logit(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> crate::error::Result<()> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let bytes_per_row = bytes.len() / self.rows;
        let required = self.rows;
        assert!(
            output.len() >= required,
            "gemv_into_f64_logit: output buffer size mismatch"
        );
        let output = &mut output[..required];
        output.fill(0.0);
        gemv_output_f64_logit(
            bytes,
            input,
            output,
            self.rows,
            self.cols,
            bytes_per_row,
            self.ggml_type,
        );
        Ok(())
    }

    /// GEMV into pre-allocated buffer with F32 input (avoids Vec allocation).
    pub(super) fn gemv_into(&self, input: &[f32], output: &mut [f32]) -> crate::error::Result<()> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let seq_len = input.len() / self.cols;
        let bytes_per_row = bytes.len() / self.rows;

        let required = seq_len * self.rows;
        assert!(
            output.len() >= required,
            "gemv_into: output buffer size mismatch"
        );
        let output = &mut output[..required];

        // mt94 — see `gemv_vec` for `RNB_ALL_F32_OVERRIDE=1` rationale.
        if all_f32_override_enabled() && is_quantized_type(self.ggml_type) {
            output.fill(0.0);
            gemv_full_dequant_f32(
                bytes,
                input,
                output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
                self.ggml_type,
            );
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        if let Some(result) = prefill_gemv_cuda(self, bytes, input, seq_len) {
            let result = result?;
            output.copy_from_slice(&result[..required]);
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        if seq_len == 1 {
            if let Some(result) = cuda_decode_gemv(self, bytes, input) {
                let result = result?;
                output.copy_from_slice(&result[..required]);
                return Ok(());
            }
            if self.ggml_type == GGMLType::BF16 {
                if let Some(result) = cuda_bf16_gemv(self, input)? {
                    output.copy_from_slice(&result[..required]);
                    return Ok(());
                }
            }
        }

        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        if seq_len == 1 && self.ggml_type == GGMLType::Q4_K {
            use super::metal_runtime;
            if metal_runtime::decode_gemv_into_if_supported(
                rnb_loader::GGMLType::Q4_K,
                bytes,
                self.rows,
                self.cols,
                input,
                output,
                "",
            )
            .map_err(|e| crate::error::LlmError::Forward(e))?
            {
                return Ok(());
            }
        }

        output.fill(0.0);

        // mc71 — opt-in `RNB_LAYER_GEMV_F64=1` routes every decode-time layer
        // GEMV (attention Q/K/V/O, FFN gate/up/down) through the slow scalar
        // dequant + f64 reduction path. mc70 attn-f64 prototype showed the
        // logit ranking margin (0.1-0.5) is sensitive to f32 accumulator
        // drift across 24 layers; this lets us measure whether extending
        // f64 accumulation to FFN/Q/K/V also fixes step-4 onward divergence.
        if seq_len == 1 && std::env::var("RNB_LAYER_GEMV_F64").is_ok() {
            gemv_output_f64_logit(
                bytes,
                input,
                output,
                self.rows,
                self.cols,
                bytes_per_row,
                self.ggml_type,
            );
            return Ok(());
        }

        if force_generic_gemv(self.rows, self.cols) {
            gemv_generic(
                bytes,
                input,
                output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
                self.ggml_type,
            );
            return Ok(());
        }

        if self.ggml_type == GGMLType::F16 {
            let weight_u16: &[u16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2)
            };
            gemv_f16(weight_u16, input, output, self.rows, self.cols, seq_len);
            return Ok(());
        }

        if self.ggml_type == GGMLType::F32 {
            let weight_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
            };
            gemv_f32(weight_f32, input, output, self.rows, self.cols, seq_len);
            return Ok(());
        }

        if self.ggml_type == GGMLType::BF16 {
            let weight_u16: &[u16] = unsafe {
                std::slice::from_raw_parts(bytes.as_ptr() as *const u16, bytes.len() / 2)
            };
            gemv_bf16(weight_u16, input, output, self.rows, self.cols, seq_len);
            return Ok(());
        }

        #[cfg(target_arch = "aarch64")]
        if dispatch_into_fast_gemv(self, bytes, input, output, seq_len, bytes_per_row)? {
            return Ok(());
        }

        match self.ggml_type {
            GGMLType::Q4_0 => gemv_q4_0(
                bytes,
                input,
                output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
            ),
            GGMLType::Q8_0 => gemv_q8_0(
                bytes,
                input,
                output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
            ),
            _ => gemv_generic(
                bytes,
                input,
                output,
                self.rows,
                self.cols,
                seq_len,
                bytes_per_row,
                self.ggml_type,
            ),
        }

        Ok(())
    }

    /// mt91 instrumentation wrapper — production GEMV dispatch. Identical to
    /// [`Self::gemv_into`] but with `pub(crate)` visibility so the
    /// `engine::q4_microbench` module can reach it without leaking `pub(super)`.
    pub(crate) fn gemv_into_for_microbench(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> crate::error::Result<()> {
        self.gemv_into(input, output)
    }

    /// mt91 instrumentation wrapper — generic scalar dequant + f32 reduce path.
    pub(crate) fn gemv_into_generic_for_microbench(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> crate::error::Result<()> {
        self.gemv_into_generic(input, output)
    }

    /// mt91 instrumentation wrapper — full dequant + f64 accumulator path
    /// (`seq_len = 1` only, same constraint as [`Self::gemv_into_f64_logit`]).
    pub(crate) fn gemv_into_f64_logit_for_microbench(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> crate::error::Result<()> {
        self.gemv_into_f64_logit(input, output)
    }

    /// mt94 axis — force per-row `dequantize_bytes_to_f32` + pure f32×f32 dot
    /// for every row of this weight, bypassing both the production NEON Q4×Q8K
    /// integer kernel and the `quant_gemv` scalar quant-aware fallback. Used
    /// only when the caller opts into mixed-precision ablation (e.g.
    /// `RNB_QK_F32_OVERRIDE=1` Q/K projection dequant). prefill (`seq_len > 1`)
    /// is supported. CUDA / aarch64 fast paths are intentionally not consulted.
    pub(super) fn gemv_into_full_dequant_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> crate::error::Result<()> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let seq_len = input.len() / self.cols;
        let bytes_per_row = bytes.len() / self.rows;
        let required = seq_len * self.rows;
        assert!(
            output.len() >= required,
            "gemv_into_full_dequant_f32: output buffer size mismatch"
        );
        let output = &mut output[..required];
        output.fill(0.0);
        gemv_full_dequant_f32(
            bytes,
            input,
            output,
            self.rows,
            self.cols,
            seq_len,
            bytes_per_row,
            self.ggml_type,
        );
        Ok(())
    }

    pub(super) fn gemv_into_generic(
        &self,
        input: &[f32],
        output: &mut [f32],
    ) -> crate::error::Result<()> {
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| crate::error::LlmError::Forward("quantized weight: no bytes".into()))?;
        let seq_len = input.len() / self.cols;
        let bytes_per_row = bytes.len() / self.rows;
        let required = seq_len * self.rows;
        assert!(
            output.len() >= required,
            "gemv_into_generic: output buffer size mismatch"
        );
        let output = &mut output[..required];
        output.fill(0.0);
        gemv_generic(
            bytes,
            input,
            output,
            self.rows,
            self.cols,
            seq_len,
            bytes_per_row,
            self.ggml_type,
        );
        Ok(())
    }
}
