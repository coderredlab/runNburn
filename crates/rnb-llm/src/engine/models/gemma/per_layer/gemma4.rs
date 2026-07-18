use super::*;

/// Gemma4-only PLE per-block fusion implementing runtime contract §4b.
///
/// `pe_in = hidden`
/// `gate  = inp_gate @ hidden`
/// `gate  = gelu(gate) * inp_this_layer`    (elementwise, pre-fused slice from §4a)
/// `delta = proj @ gate`
/// `delta = rmsnorm(delta, post_norm)`
/// `h_out = pe_in + delta`                  (residual — missing in the legacy Gemma path)
///
/// No env-var branches, no layer34 hard-fix, no runtime flavor heuristic. The legacy
/// `apply_gemma_per_layer_branch` path is preserved below for Gemma / Gemma3 / Gemma3n.
pub(in crate::engine) fn apply_gemma4_per_layer_branch(
    hidden: Tensor,
    base: &GemmaPerLayerBase,
    layer_idx: usize,
    weights: &GemmaPerLayerWeights,
    metadata: &ModelMetadata,
    norm_eps: f32,
) -> crate::error::Result<Tensor> {
    apply_gemma4_per_layer_branch_with_output_scale(
        hidden, base, layer_idx, weights, metadata, norm_eps, None,
    )
    .map(|output| output.hidden)
}

pub(in crate::engine) struct Gemma4PerLayerBranchOutput {
    pub(in crate::engine) hidden: Tensor,
    pub(in crate::engine) output_scale_applied: bool,
}

fn gemma4_cuda_ple_out_scale_enabled() -> bool {
    std::env::var("RNB_CUDA_GEMMA_PLE_OUT_SCALE")
        .ok()
        .map(|value| {
            let value = value.to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn gemma4_ple_replay_f32_weight(
    weight: &QuantizedWeight,
    layer_idx: usize,
    label: &str,
) -> crate::error::Result<Vec<f32>> {
    if weight.ggml_type != GGMLType::F32 {
        return Err(crate::error::LlmError::Forward(format!(
            "Gemma4 PLE replay expected F32 {label} at layer {layer_idx}, got {:?}",
            weight.ggml_type
        )));
    }
    let bytes = weight.data.as_bytes().ok_or_else(|| {
        crate::error::LlmError::Forward(format!(
            "Gemma4 PLE replay {label} at layer {layer_idx} has no backing bytes"
        ))
    })?;
    if bytes.len() != weight.rows * weight.cols * std::mem::size_of::<f32>() {
        return Err(crate::error::LlmError::Forward(format!(
            "Gemma4 PLE replay {label} byte mismatch at layer {layer_idx}: got {}, expected {}",
            bytes.len(),
            weight.rows * weight.cols * std::mem::size_of::<f32>()
        )));
    }
    // cu19: `bytes` (mmap-backed GGUF host slice) has no 4-byte alignment
    // guarantee, so reinterpreting it as `&[f32]` via `from_raw_parts` is
    // undefined behavior in Rust. Decode through `f32::from_le_bytes` which
    // is alignment-agnostic. Replay is a dev-only dump path so the Vec
    // allocation overhead is acceptable.
    let out = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(out)
}

pub(in crate::engine) fn apply_gemma4_per_layer_branch_with_output_scale(
    hidden: Tensor,
    base: &GemmaPerLayerBase,
    layer_idx: usize,
    weights: &GemmaPerLayerWeights,
    metadata: &ModelMetadata,
    norm_eps: f32,
    out_scale: Option<&Tensor>,
) -> crate::error::Result<Gemma4PerLayerBranchOutput> {
    if env_flag("RNB_DEBUG_PLE_ENTER") {
        eprintln!("[ple-enter] layer={}", layer_idx);
    }
    let hidden_dim = metadata.hidden_dim;
    let d = metadata.embedding_length_per_layer_input;
    let total_d = metadata.num_layers * d;
    let hidden_shape = hidden.shape().to_vec();
    let hidden_data = kernels::tensor_as_f32_slice(&hidden);
    let seq_len = hidden_data.len() / hidden_dim;
    let layer = &weights.layers[layer_idx];

    if layer.inp_gate.rows != d || layer.inp_gate.cols != hidden_dim {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma4 inp_gate shape mismatch at layer {}: [{}x{}], expected [{}x{}]",
            layer_idx, layer.inp_gate.rows, layer.inp_gate.cols, d, hidden_dim
        )));
    }
    if layer.proj.rows != hidden_dim || layer.proj.cols != d {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma4 proj shape mismatch at layer {}: [{}x{}], expected [{}x{}]",
            layer_idx, layer.proj.rows, layer.proj.cols, hidden_dim, d
        )));
    }
    if layer.post_norm.numel() != hidden_dim {
        return Err(crate::error::LlmError::Forward(format!(
            "gemma4 post_norm mismatch at layer {}: got {}, expected {}",
            layer_idx,
            layer.post_norm.numel(),
            hidden_dim
        )));
    }

    if seq_len == 1 {
        let base_off = layer_idx * d;
        let per_layer_input = &base.mixed[base_off..base_off + d];
        if dump_bin_dir().is_some() {
            dump_bin("decode", layer_idx, "ple_per_layer_input", per_layer_input);
        }

        let mut gate = vec![0.0f32; d];
        let gate_gpu_ok = backend_runtime::decode_gemv_into_if_supported(
            &layer.inp_gate,
            hidden_data,
            &mut gate,
            "gemma4 ple inp_gate",
            false,
        )?;
        if !gate_gpu_ok {
            gate = layer.inp_gate.gemv_vec(hidden_data)?;
        }
        if dump_bin_dir().is_some() {
            dump_bin("decode", layer_idx, "ple_gate", &gate);
        }
        kernels::activation::fused_gelu_mul_inplace(&mut gate, per_layer_input);
        if dump_bin_dir().is_some() {
            dump_bin("decode", layer_idx, "ple_gated", &gate);
        }

        let mut projected = vec![0.0f32; hidden_dim];
        let proj_gpu_ok = backend_runtime::decode_gemv_into_if_supported(
            &layer.proj,
            &gate,
            &mut projected,
            "gemma4 ple proj",
            false,
        )?;
        if !proj_gpu_ok {
            projected = layer.proj.gemv_vec(&gate)?;
        }
        if dump_bin_dir().is_some() {
            dump_bin("decode", layer_idx, "ple_proj", &projected);
        }

        let post_norm = kernels::tensor_as_f32_slice(&layer.post_norm);
        let mut normed = vec![0.0f32; hidden_dim];
        apply_model_norm_into(
            &projected,
            post_norm,
            norm_eps,
            &mut normed,
            ModelArchitecture::Gemma4,
        );
        if dump_bin_dir().is_some() {
            dump_bin("decode", layer_idx, "ple_norm", &normed);
        }

        let mut out = hidden_data.to_vec();
        kernels::elementwise::add_inplace(&mut out, &normed);
        if dump_bin_dir().is_some() {
            dump_bin("decode", layer_idx, "ple_out", &out);
        }
        return Ok(Gemma4PerLayerBranchOutput {
            hidden: Tensor::from_vec(out, &hidden_shape),
            output_scale_applied: false,
        });
    }

    // Slice the pre-fused §4a table (`base.mixed` = (token_embd * sqrt(d) + rmsnorm(model_proj @ h / sqrt(n_embd))) / sqrt(2))
    let mut per_layer_input = vec![0.0f32; seq_len * d];
    for t in 0..seq_len {
        let base_off = t * total_d + layer_idx * d;
        let dst_off = t * d;
        per_layer_input[dst_off..dst_off + d].copy_from_slice(&base.mixed[base_off..base_off + d]);
    }

    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "ple_per_layer_input",
            &per_layer_input,
        );
    }

    let post_norm = kernels::tensor_as_f32_slice(&layer.post_norm);
    let replay_out_scale = active_layer_output_scale(out_scale, layer_idx);
    if gemma4_ple_replay_dump_enabled(layer_idx) {
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_hidden", hidden_data)?;
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_input", &per_layer_input)?;
        let gate_replay = gemma4_ple_replay_f32_weight(&layer.inp_gate, layer_idx, "gate")?;
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_gate_weight", &gate_replay)?;
        let proj_replay = gemma4_ple_replay_f32_weight(&layer.proj, layer_idx, "proj")?;
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_proj_weight", &proj_replay)?;
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_post_norm", post_norm)?;
        if let Some(scale) = replay_out_scale {
            dump_gemma4_ple_replay_f32("host", layer_idx, "ple_out_scale", scale)?;
        }
    }

    if dump_bin_dir().is_none() {
        let mut out = hidden_data.to_vec();
        let active_out_scale = gemma4_cuda_ple_out_scale_enabled()
            .then_some(replay_out_scale)
            .flatten();
        if backend_runtime::gemma4_ple_q4k_batch_norm_residual_if_supported(
            &layer.inp_gate,
            &layer.proj,
            post_norm,
            active_out_scale,
            &per_layer_input,
            d,
            hidden_dim,
            seq_len,
            &mut out,
            norm_eps,
        )? {
            return Ok(Gemma4PerLayerBranchOutput {
                hidden: Tensor::from_vec(out, &hidden_shape),
                output_scale_applied: active_out_scale.is_some(),
            });
        }
    }

    // gate = inp_gate @ hidden  (shape: [seq_len * d])
    let mut gate = layer.inp_gate.gemv_vec(hidden_data)?;
    if gemma4_ple_replay_dump_enabled(layer_idx) {
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_gate", &gate)?;
    }
    if dump_bin_dir().is_some() {
        dump_bin("prefill", layer_idx, "ple_gate", &gate);
    }
    // gate <- gelu(gate) * per_layer_input
    kernels::activation::fused_gelu_mul_inplace(&mut gate, &per_layer_input);
    if gemma4_ple_replay_dump_enabled(layer_idx) {
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_gated", &gate)?;
    }
    if dump_bin_dir().is_some() {
        dump_bin("prefill", layer_idx, "ple_gated", &gate);
    }

    // projected = proj @ gate  (shape: [seq_len * hidden_dim])
    let projected = layer.proj.gemv_vec(&gate)?;
    let projected_tensor = Tensor::from_vec(projected, &[seq_len, hidden_dim]);
    if gemma4_ple_replay_dump_enabled(layer_idx) {
        dump_gemma4_ple_replay_f32(
            "host",
            layer_idx,
            "ple_projected",
            kernels::tensor_as_f32_slice(&projected_tensor),
        )?;
    }
    if dump_bin_dir().is_some() {
        dump_bin(
            "prefill",
            layer_idx,
            "ple_proj",
            kernels::tensor_as_f32_slice(&projected_tensor),
        );
    }

    // plain RMSNorm against blk.i.post_norm (no unit-offset, no skip)
    let normed = kernels::norm::rms_norm(&projected_tensor, &layer.post_norm, norm_eps)
        .map_err(|e| crate::error::LlmError::Forward(e.to_string()))?;
    let normed_data = kernels::tensor_as_f32_slice(&normed);
    if gemma4_ple_replay_dump_enabled(layer_idx) {
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_normed", normed_data)?;
    }
    if dump_bin_dir().is_some() {
        dump_bin("prefill", layer_idx, "ple_norm", normed_data);
    }

    // Residual: h_out = pe_in + delta
    let mut out = hidden_data.to_vec();
    kernels::elementwise::add_inplace(&mut out, normed_data);
    if gemma4_ple_replay_dump_enabled(layer_idx) {
        dump_gemma4_ple_replay_f32("host", layer_idx, "ple_final", &out)?;
        if let Some(scale) = replay_out_scale {
            let scaled = out.iter().map(|value| value * scale[0]).collect::<Vec<_>>();
            dump_gemma4_ple_replay_f32("host", layer_idx, "ple_final_scaled", &scaled)?;
        }
    }
    if dump_bin_dir().is_some() {
        dump_bin("prefill", layer_idx, "ple_out", &out);
    }
    Ok(Gemma4PerLayerBranchOutput {
        hidden: Tensor::from_vec(out, &hidden_shape),
        output_scale_applied: false,
    })
}
