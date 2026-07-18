use crate::engine::cpu_runtime::kernels;
use crate::engine::layer_weights::AttentionLayerWeights;
use crate::engine::norm::{apply_model_gate_mul_inplace, apply_model_norm_into};
use crate::engine::quantized_weight_types::QuantizedWeight;
use crate::engine::scalar_gemv::gemv_generic;
use crate::engine::{decode_shared_expert_moe, ModelArchitecture, ModelMetadata, ScratchBuffers};
use crate::error::{LlmError, Result};
use crate::kv_cache::KVCache;
use rnb_core::tensor::Tensor;
use rnb_loader::{GGMLType, LoadedModel};

const GLM_DSA_ARCH: ModelArchitecture = ModelArchitecture::GlmDsa;

/// One head-batched GGML matrix. GGUF stores these as
/// `[head_count, rows_per_head, cols]`, while quantization blocks run along
/// `cols`, so each head is a contiguous `rows_per_head x cols` matrix.
pub(in crate::engine) struct GlmMlaHeadWeight {
    data: Tensor,
    ggml_type: GGMLType,
    head_count: usize,
    rows_per_head: usize,
    cols: usize,
}

impl GlmMlaHeadWeight {
    /// pm112: 전 head 배치 GEMV 를 Metal 로 시도. input 은 `[head_count * cols]`
    /// packing, output 은 `[head_count * rows_per_head]`. 미지원 시 Ok(false).
    fn gemv_all_heads(&self, packed_input: &[f32], output: &mut [f32]) -> Result<bool> {
        let Some(bytes) = self.data.as_bytes() else {
            return Ok(false);
        };
        let used = crate::engine::backend_runtime::glm_mla_head_gemv_into(
            self.ggml_type,
            bytes,
            self.head_count,
            self.rows_per_head,
            self.cols,
            packed_input,
            output,
        )
        .map_err(LlmError::Forward)?;
        if used && std::env::var("RNB_GLM_MLA_VERIFY").as_deref() == Ok("1") {
            let mut cpu = vec![0.0f32; self.head_count * self.rows_per_head];
            for head in 0..self.head_count {
                let cpu_out = &mut cpu[head * self.rows_per_head..(head + 1) * self.rows_per_head];
                self.gemv_head(
                    head,
                    &packed_input[head * self.cols..(head + 1) * self.cols],
                    cpu_out,
                )?;
            }
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            for (metal_v, cpu_v) in output.iter().zip(&cpu) {
                let abs = (metal_v - cpu_v).abs();
                max_abs = max_abs.max(abs);
                max_rel = max_rel.max(abs / cpu_v.abs().max(1e-5));
            }
            eprintln!(
                "[glm-mla-verify] heads {:?} {}x{}x{} max_abs={max_abs:.6} max_rel={max_rel:.6}",
                self.ggml_type, self.head_count, self.rows_per_head, self.cols
            );
        }
        Ok(used)
    }

    /// pm113: prefill slot-batch GEMV (slot = token*head_count + head) 를 Metal
    /// 단일 dispatch 로 시도. input `[slots * cols]`, output `[slots * rows_per_head]`.
    /// 미지원 시 Ok(false) — caller 가 per-head CPU 로 fallback.
    fn gemv_all_heads_slots(
        &self,
        packed_input: &[f32],
        slots: usize,
        output: &mut [f32],
    ) -> Result<bool> {
        let Some(bytes) = self.data.as_bytes() else {
            return Ok(false);
        };
        crate::engine::backend_runtime::glm_mla_head_slots_gemv_into(
            self.ggml_type,
            bytes,
            slots,
            self.head_count,
            self.rows_per_head,
            self.cols,
            packed_input,
            output,
        )
        .map_err(LlmError::Forward)
    }

    fn gemv_head(&self, head: usize, input: &[f32], output: &mut [f32]) -> Result<()> {
        if head >= self.head_count || input.len() != self.cols || output.len() != self.rows_per_head
        {
            return Err(LlmError::Forward(format!(
                "GLM MLA head GEMV shape mismatch: head={head}/{}, input={} expected {}, output={} expected {}",
                self.head_count,
                input.len(),
                self.cols,
                output.len(),
                self.rows_per_head
            )));
        }
        let bytes = self
            .data
            .as_bytes()
            .ok_or_else(|| LlmError::Forward("GLM MLA head weight has no raw bytes".into()))?;
        let total_rows = self.head_count * self.rows_per_head;
        if total_rows == 0 || bytes.len() % total_rows != 0 {
            return Err(LlmError::Forward(format!(
                "GLM MLA head weight byte geometry mismatch: bytes={} rows={total_rows}",
                bytes.len()
            )));
        }
        let bytes_per_row = bytes.len() / total_rows;
        let bytes_per_head = self.rows_per_head * bytes_per_row;
        let start = head * bytes_per_head;
        gemv_generic(
            &bytes[start..start + bytes_per_head],
            input,
            output,
            self.rows_per_head,
            self.cols,
            1,
            bytes_per_row,
            self.ggml_type,
        );
        Ok(())
    }
}

pub(in crate::engine) struct GlmDsaAttentionLayerWeights {
    q_a: QuantizedWeight,
    q_a_norm: Tensor,
    q_b: QuantizedWeight,
    kv_a: QuantizedWeight,
    kv_a_norm: Tensor,
    k_b: GlmMlaHeadWeight,
    v_b: GlmMlaHeadWeight,
    o: QuantizedWeight,
}

type F32WeightLoader = fn(&LoadedModel, &str) -> Tensor;
type QuantizedWeightLoader = fn(&LoadedModel, &str) -> QuantizedWeight;

pub(in crate::engine) fn load_attention_layers(
    model: &LoadedModel,
    num_layers: usize,
    load_f32_weight: F32WeightLoader,
    load_quantized_weight: QuantizedWeightLoader,
) -> Vec<GlmDsaAttentionLayerWeights> {
    (0..num_layers)
        .map(|layer_idx| {
            load_attention_layer(model, layer_idx, load_f32_weight, load_quantized_weight)
        })
        .collect()
}

pub(in crate::engine) fn load_attention_layer(
    model: &LoadedModel,
    layer_idx: usize,
    load_f32_weight: F32WeightLoader,
    load_quantized_weight: QuantizedWeightLoader,
) -> GlmDsaAttentionLayerWeights {
    let prefix = format!("blk.{layer_idx}");
    GlmDsaAttentionLayerWeights {
        q_a: load_quantized_weight(model, &format!("{prefix}.attn_q_a.weight")),
        q_a_norm: load_f32_weight(model, &format!("{prefix}.attn_q_a_norm.weight")),
        q_b: load_quantized_weight(model, &format!("{prefix}.attn_q_b.weight")),
        kv_a: load_quantized_weight(model, &format!("{prefix}.attn_kv_a_mqa.weight")),
        kv_a_norm: load_f32_weight(model, &format!("{prefix}.attn_kv_a_norm.weight")),
        k_b: load_head_weight(model, &format!("{prefix}.attn_k_b.weight")),
        v_b: load_head_weight(model, &format!("{prefix}.attn_v_b.weight")),
        o: load_quantized_weight(model, &format!("{prefix}.attn_output.weight")),
    }
}

fn load_head_weight(model: &LoadedModel, name: &str) -> GlmMlaHeadWeight {
    let data = model
        .weights
        .get(name)
        .unwrap_or_else(|| panic!("GLM MLA: missing {name}"))
        .clone();
    let shape = model
        .float_shapes
        .get(name)
        .unwrap_or_else(|| panic!("GLM MLA: missing logical shape for {name}"));
    assert_eq!(
        shape.len(),
        3,
        "GLM MLA: {name} must be a 3-D head-batched matrix"
    );
    GlmMlaHeadWeight {
        data,
        ggml_type: model
            .tensor_ggml_types
            .get(name)
            .copied()
            .unwrap_or(GGMLType::F32),
        head_count: shape[0],
        rows_per_head: shape[1],
        cols: shape[2],
    }
}

/// pm112: MLA dense GEMV (q_a/q_b/kv_a/o) 를 Metal 로 라우팅 (Q5_K/Q8_0,
/// backend resident wrap). 미지원/실패 시 CPU `gemv_vec` fallback.
fn mla_dense_gemv(weight: &QuantizedWeight, input: &[f32]) -> Result<Vec<f32>> {
    if let Some(bytes) = weight.data.as_bytes() {
        let mut out = vec![0.0f32; weight.rows];
        if crate::engine::backend_runtime::glm_mla_gemv_into(
            weight.ggml_type,
            bytes,
            weight.rows,
            weight.cols,
            input,
            &mut out,
        )
        .unwrap_or(false)
        {
            if std::env::var("RNB_GLM_MLA_VERIFY").as_deref() == Ok("1") {
                let cpu = weight.gemv_vec(input)?;
                let mut max_abs = 0.0f32;
                let mut max_rel = 0.0f32;
                for (metal_v, cpu_v) in out.iter().zip(&cpu) {
                    let abs = (metal_v - cpu_v).abs();
                    max_abs = max_abs.max(abs);
                    max_rel = max_rel.max(abs / cpu_v.abs().max(1e-5));
                }
                eprintln!(
                    "[glm-mla-verify] dense {:?} {}x{} max_abs={max_abs:.6} max_rel={max_rel:.6}",
                    weight.ggml_type, weight.rows, weight.cols
                );
            }
            return Ok(out);
        }
    }
    weight.gemv_vec(input)
}

/// pm112 진단: `RNB_GLM_MLA_PROFILE=1` 시 decode_layer 단계별 누적 ms.
/// [qa, qb, kva, kb_absorb, core, vb, o, ffn] — 78콜(1 token pass)마다 1줄.
fn mla_stage_profile(stage: usize, elapsed_ms: f64) {
    use std::cell::Cell;
    thread_local! {
        static ACC: Cell<[f64; 9]> = const { Cell::new([0.0; 9]) };
    }
    ACC.with(|acc| {
        let mut values = acc.get();
        values[stage] += elapsed_ms;
        values[8] += if stage == 7 { 1.0 } else { 0.0 };
        acc.set(values);
        if stage == 7 && (values[8] as usize) % 78 == 0 {
            let n = values[8];
            eprintln!(
                "[glm-mla-profile] pass={} avg ms: qa={:.2} qb={:.2} kva={:.2} kb={:.2} core={:.2} vb={:.2} o={:.2} ffn={:.2}",
                (n as usize) / 78,
                values[0] / n, values[1] / n, values[2] / n, values[3] / n,
                values[4] / n, values[5] / n, values[6] / n, values[7] / n,
            );
        }
    });
}

fn mla_profile_enabled() -> bool {
    std::env::var("RNB_GLM_MLA_PROFILE").as_deref() == Ok("1")
}

/// pm112: front chain (q_a→rms→q_b, kv_a, k_b) 를 단일 command buffer 로 시도.
/// 성공 시 (q, kv_raw, q_absorbed) 반환, 미지원이면 None.
fn mla_front_chain(
    mla: &GlmDsaAttentionLayerWeights,
    metadata: &ModelMetadata,
    input: &[f32],
) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
    let (Some(qa_bytes), Some(norm_bytes), Some(qb_bytes), Some(kva_bytes), Some(kb_bytes)) = (
        mla.q_a.data.as_bytes(),
        mla.q_a_norm.as_bytes(),
        mla.q_b.data.as_bytes(),
        mla.kv_a.data.as_bytes(),
        mla.k_b.data.as_bytes(),
    ) else {
        return Ok(None);
    };
    let mut q = vec![0.0f32; mla.q_b.rows];
    let mut kv_raw = vec![0.0f32; mla.kv_a.rows];
    let mut q_absorbed = vec![0.0f32; mla.k_b.head_count * mla.k_b.rows_per_head];
    let used = crate::engine::backend_runtime::glm_mla_front_into(
        mla.q_a.ggml_type,
        mla.q_b.ggml_type,
        mla.kv_a.ggml_type,
        mla.k_b.ggml_type,
        qa_bytes,
        norm_bytes,
        qb_bytes,
        kva_bytes,
        kb_bytes,
        mla.q_a.cols,
        mla.q_a.rows,
        mla.q_b.rows,
        mla.kv_a.rows,
        mla.k_b.head_count,
        mla.k_b.rows_per_head,
        mla.k_b.cols,
        mla.v_b.rows_per_head,
        metadata.norm_eps,
        input,
        &mut q,
        &mut kv_raw,
        &mut q_absorbed,
    )
    .map_err(LlmError::Forward)?;
    Ok(used.then_some((q, kv_raw, q_absorbed)))
}

/// pm112: back chain (v_b 64-head → o) 를 단일 command buffer 로 시도.
fn mla_back_chain(
    mla: &GlmDsaAttentionLayerWeights,
    metadata: &ModelMetadata,
    latent_all: &[f32],
) -> Result<Option<Vec<f32>>> {
    let (Some(vb_bytes), Some(o_bytes)) = (mla.v_b.data.as_bytes(), mla.o.data.as_bytes()) else {
        return Ok(None);
    };
    let mut out = vec![0.0f32; mla.o.rows];
    let used = crate::engine::backend_runtime::glm_mla_back_into(
        mla.v_b.ggml_type,
        mla.o.ggml_type,
        vb_bytes,
        o_bytes,
        mla.o.rows,
        mla.q_a.rows,
        mla.q_b.rows,
        mla.kv_a.rows,
        mla.v_b.head_count,
        mla.v_b.cols,
        mla.k_b.cols,
        mla.v_b.rows_per_head,
        metadata.norm_eps,
        latent_all,
        &mut out,
    )
    .map_err(LlmError::Forward)?;
    Ok(used.then_some(out))
}

pub(in crate::engine) fn decode_layer(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    scratch: &mut ScratchBuffers,
    base: &AttentionLayerWeights,
    mla: &GlmDsaAttentionLayerWeights,
    layer_idx: usize,
    pos: usize,
) -> Result<()> {
    decode_layer_with_positions(
        kv_cache, metadata, scratch, base, mla, layer_idx, layer_idx, pos, pos,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn decode_layer_with_positions(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    scratch: &mut ScratchBuffers,
    base: &AttentionLayerWeights,
    mla: &GlmDsaAttentionLayerWeights,
    kv_layer_idx: usize,
    model_layer_idx: usize,
    cache_pos: usize,
    rope_pos: usize,
) -> Result<()> {
    let hidden_dim = metadata.hidden_dim;
    let norm_weight = kernels::tensor_as_f32_slice(&base.attn_norm);
    apply_model_norm_into(
        &scratch.hidden[..hidden_dim],
        norm_weight,
        metadata.norm_eps,
        &mut scratch.norm_buf[..hidden_dim],
        GLM_DSA_ARCH,
    );

    let profiling = mla_profile_enabled();
    let stage_start = std::time::Instant::now();
    let front_chain = mla_front_chain(mla, metadata, &scratch.norm_buf[..hidden_dim])?;
    let (q, kv_raw, chain_q_absorbed) = if let Some((q, kv_raw, q_absorbed)) = front_chain {
        (q, kv_raw, Some(q_absorbed))
    } else {
        let q_a = mla_dense_gemv(&mla.q_a, &scratch.norm_buf[..hidden_dim])?;
        let q_rank = mla.q_a.rows;
        if q_a.len() != q_rank {
            return Err(LlmError::Forward(
                "GLM MLA q_a output length mismatch".into(),
            ));
        }
        let mut q_rank_norm = vec![0.0f32; q_rank];
        kernels::norm::rms_norm_into(
            &q_a,
            kernels::tensor_as_f32_slice(&mla.q_a_norm),
            metadata.norm_eps,
            &mut q_rank_norm,
        );
        let q = mla_dense_gemv(&mla.q_b, &q_rank_norm)?;
        let kv_raw = mla_dense_gemv(&mla.kv_a, &scratch.norm_buf[..hidden_dim])?;
        (q, kv_raw, None)
    };
    if profiling {
        mla_stage_profile(0, stage_start.elapsed().as_secs_f64() * 1000.0);
    }
    let kv_rank = kernels::tensor_as_f32_slice(&mla.kv_a_norm).len();
    let rope_dim = metadata.rope_dim;
    if rope_dim == 0 || kv_raw.len() != kv_rank + rope_dim {
        return Err(LlmError::Forward(format!(
            "GLM MLA compressed KV shape mismatch: got {}, expected {}+{}",
            kv_raw.len(),
            kv_rank,
            rope_dim
        )));
    }
    let mut kv_latent = vec![0.0f32; kv_rank];
    kernels::norm::rms_norm_into(
        &kv_raw[..kv_rank],
        kernels::tensor_as_f32_slice(&mla.kv_a_norm),
        metadata.norm_eps,
        &mut kv_latent,
    );
    let mut k_pe = kv_raw[kv_rank..].to_vec();
    kernels::rope::rope_inplace(&mut k_pe, rope_pos, rope_dim, rope_dim, metadata.rope_theta);

    let head_count = metadata.num_heads;
    if q.len() % head_count != 0 {
        return Err(LlmError::Forward(format!(
            "GLM MLA Q shape {} is not divisible by {head_count} heads",
            q.len()
        )));
    }
    let qk_dim = q.len() / head_count;
    let q_nope_dim = qk_dim
        .checked_sub(rope_dim)
        .ok_or_else(|| LlmError::Forward("GLM MLA qk_dim is smaller than rope_dim".into()))?;
    if mla.k_b.head_count != head_count
        || mla.k_b.rows_per_head != kv_rank
        || mla.k_b.cols != q_nope_dim
        || mla.v_b.head_count != head_count
        || mla.v_b.cols != kv_rank
    {
        return Err(LlmError::Forward(
            "GLM MLA K/V-B geometry does not match metadata".into(),
        ));
    }

    let mut q_pe = vec![0.0f32; head_count * rope_dim];
    let stage_start = std::time::Instant::now();
    let q_absorbed = if let Some(q_absorbed) = chain_q_absorbed {
        for head in 0..head_count {
            let q_head = &q[head * qk_dim..(head + 1) * qk_dim];
            let pe = &mut q_pe[head * rope_dim..(head + 1) * rope_dim];
            pe.copy_from_slice(&q_head[q_nope_dim..]);
            kernels::rope::rope_inplace(pe, rope_pos, rope_dim, rope_dim, metadata.rope_theta);
        }
        q_absorbed
    } else {
        let mut q_absorbed = vec![0.0f32; head_count * kv_rank];
        let mut q_nope_packed = vec![0.0f32; head_count * q_nope_dim];
        for head in 0..head_count {
            let q_head = &q[head * qk_dim..(head + 1) * qk_dim];
            q_nope_packed[head * q_nope_dim..(head + 1) * q_nope_dim]
                .copy_from_slice(&q_head[..q_nope_dim]);
            let pe = &mut q_pe[head * rope_dim..(head + 1) * rope_dim];
            pe.copy_from_slice(&q_head[q_nope_dim..]);
            kernels::rope::rope_inplace(pe, rope_pos, rope_dim, rope_dim, metadata.rope_theta);
        }
        if !mla.k_b.gemv_all_heads(&q_nope_packed, &mut q_absorbed)? {
            for head in 0..head_count {
                mla.k_b.gemv_head(
                    head,
                    &q_nope_packed[head * q_nope_dim..(head + 1) * q_nope_dim],
                    &mut q_absorbed[head * kv_rank..(head + 1) * kv_rank],
                )?;
            }
        }
        q_absorbed
    };
    if profiling {
        mla_stage_profile(3, stage_start.elapsed().as_secs_f64() * 1000.0);
    }

    let cache_stride = kv_rank + rope_dim;
    if metadata.num_kv_heads * metadata.head_dim != cache_stride {
        return Err(LlmError::Forward(format!(
            "GLM MLA KV cache stride {} does not match compressed stride {cache_stride}",
            metadata.num_kv_heads * metadata.head_dim
        )));
    }
    let mut compressed_k = Vec::with_capacity(cache_stride);
    compressed_k.extend_from_slice(&kv_latent);
    compressed_k.extend_from_slice(&k_pe);
    scratch.v_buf[..cache_stride].fill(0.0);
    kv_cache.append(
        kv_layer_idx,
        cache_pos,
        &compressed_k,
        &scratch.v_buf[..cache_stride],
    );
    let (cache, _) = kv_cache.get_up_to(kv_layer_idx, cache_pos + 1);

    let value_dim = mla.v_b.rows_per_head;
    let mut attention_concat = vec![0.0f32; head_count * value_dim];
    let mut scores = vec![0.0f32; cache_pos + 1];
    let mut latent_all = vec![0.0f32; head_count * kv_rank];
    let scale = 1.0f32 / (qk_dim as f32).sqrt();
    let stage_start = std::time::Instant::now();
    for head in 0..head_count {
        let q_latent = &q_absorbed[head * kv_rank..(head + 1) * kv_rank];
        let q_rope = &q_pe[head * rope_dim..(head + 1) * rope_dim];
        for token_pos in 0..=cache_pos {
            let cached = &cache[token_pos * cache_stride..(token_pos + 1) * cache_stride];
            let latent_dot = q_latent
                .iter()
                .zip(&cached[..kv_rank])
                .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                .sum::<f32>();
            let rope_dot = q_rope
                .iter()
                .zip(&cached[kv_rank..])
                .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                .sum::<f32>();
            scores[token_pos] = (latent_dot + rope_dot) * scale;
        }
        softmax_inplace(&mut scores);
        let latent_sum = &mut latent_all[head * kv_rank..(head + 1) * kv_rank];
        for (token_pos, &probability) in scores.iter().enumerate() {
            let cached = &cache[token_pos * cache_stride..token_pos * cache_stride + kv_rank];
            for (sum, &bits) in latent_sum.iter_mut().zip(cached) {
                *sum += probability * half::f16::from_bits(bits).to_f32();
            }
        }
    }
    if profiling {
        mla_stage_profile(4, stage_start.elapsed().as_secs_f64() * 1000.0);
    }
    let stage_start = std::time::Instant::now();
    let projected = if let Some(projected) = mla_back_chain(mla, metadata, &latent_all)? {
        projected
    } else {
        if !mla.v_b.gemv_all_heads(&latent_all, &mut attention_concat)? {
            for head in 0..head_count {
                mla.v_b.gemv_head(
                    head,
                    &latent_all[head * kv_rank..(head + 1) * kv_rank],
                    &mut attention_concat[head * value_dim..(head + 1) * value_dim],
                )?;
            }
        }
        mla_dense_gemv(&mla.o, &attention_concat)?
    };
    if profiling {
        mla_stage_profile(6, stage_start.elapsed().as_secs_f64() * 1000.0);
    }
    if projected.len() != hidden_dim {
        return Err(LlmError::Forward(
            "GLM MLA output projection length mismatch".into(),
        ));
    }
    kernels::elementwise::add_inplace(&mut scratch.hidden[..hidden_dim], &projected);

    let stage_start = std::time::Instant::now();
    let ffn_result = decode_ffn(metadata, scratch, base, model_layer_idx);
    if profiling {
        mla_stage_profile(7, stage_start.elapsed().as_secs_f64() * 1000.0);
    }
    ffn_result
}

/// pm114: prefill dense GEMV 를 Metal slot-batch(heads=1, slot=token) 커널로
/// 시도. Q8_0 만 지원 — 미지원 quant/빌드면 None 으로 CPU gemv_vec fallback.
fn mla_dense_prefill_gemv(
    weight: &QuantizedWeight,
    input: &[f32],
    seq_len: usize,
) -> Result<Option<Vec<f32>>> {
    if seq_len <= 1 {
        return Ok(None);
    }
    if std::env::var("RNB_METAL_GLM_MLA_PREFILL_DENSE").as_deref() == Ok("0") {
        return Ok(None);
    }
    let Some(bytes) = weight.data.as_bytes() else {
        return Ok(None);
    };
    let mut out = vec![0.0f32; seq_len * weight.rows];
    let used = crate::engine::backend_runtime::glm_mla_head_slots_gemv_into(
        weight.ggml_type,
        bytes,
        seq_len,
        1,
        weight.rows,
        weight.cols,
        input,
        &mut out,
    )
    .map_err(LlmError::Forward)?;
    Ok(used.then_some(out))
}

pub(in crate::engine) fn prefill_layer(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    hidden: Tensor,
    base: &AttentionLayerWeights,
    mla: &GlmDsaAttentionLayerWeights,
    layer_idx: usize,
    seq_len: usize,
    pos_start: usize,
) -> Result<Tensor> {
    let hidden_dim = metadata.hidden_dim;
    let hidden_data = kernels::tensor_as_f32_slice(&hidden);
    if seq_len == 0 || hidden_data.len() != seq_len.saturating_mul(hidden_dim) {
        return Err(LlmError::Forward(format!(
            "GLM MLA prefill hidden mismatch: tokens={seq_len} hidden={} n_embd={hidden_dim}",
            hidden_data.len()
        )));
    }

    let norm_weight = kernels::tensor_as_f32_slice(&base.attn_norm);
    let mut normed = vec![0.0f32; hidden_data.len()];
    for token in 0..seq_len {
        let start = token * hidden_dim;
        apply_model_norm_into(
            &hidden_data[start..start + hidden_dim],
            norm_weight,
            metadata.norm_eps,
            &mut normed[start..start + hidden_dim],
            GLM_DSA_ARCH,
        );
    }

    let q_rank = mla.q_a.rows;
    let q_a = mla.q_a.gemv_vec(&normed)?;
    if q_a.len() != seq_len * q_rank {
        return Err(LlmError::Forward(
            "GLM MLA prefill q_a output length mismatch".into(),
        ));
    }
    let q_a_norm_weight = kernels::tensor_as_f32_slice(&mla.q_a_norm);
    let mut q_rank_norm = vec![0.0f32; q_a.len()];
    for token in 0..seq_len {
        kernels::norm::rms_norm_into(
            &q_a[token * q_rank..(token + 1) * q_rank],
            q_a_norm_weight,
            metadata.norm_eps,
            &mut q_rank_norm[token * q_rank..(token + 1) * q_rank],
        );
    }
    let q = match mla_dense_prefill_gemv(&mla.q_b, &q_rank_norm, seq_len)? {
        Some(out) => out,
        None => mla.q_b.gemv_vec(&q_rank_norm)?,
    };

    let kv_rank = kernels::tensor_as_f32_slice(&mla.kv_a_norm).len();
    let rope_dim = metadata.rope_dim;
    let kv_width = kv_rank + rope_dim;
    let kv_raw = match mla_dense_prefill_gemv(&mla.kv_a, &normed, seq_len)? {
        Some(out) => out,
        None => mla.kv_a.gemv_vec(&normed)?,
    };
    if rope_dim == 0 || kv_raw.len() != seq_len * kv_width {
        return Err(LlmError::Forward(format!(
            "GLM MLA prefill compressed KV shape mismatch: got {}, expected {}x{}",
            kv_raw.len(),
            seq_len,
            kv_width
        )));
    }

    let head_count = metadata.num_heads;
    if q.len() % (seq_len * head_count) != 0 {
        return Err(LlmError::Forward(format!(
            "GLM MLA prefill Q shape {} is not divisible by {seq_len}x{head_count} heads",
            q.len()
        )));
    }
    let qk_dim = q.len() / (seq_len * head_count);
    let q_nope_dim = qk_dim
        .checked_sub(rope_dim)
        .ok_or_else(|| LlmError::Forward("GLM MLA qk_dim is smaller than rope_dim".into()))?;
    if mla.k_b.head_count != head_count
        || mla.k_b.rows_per_head != kv_rank
        || mla.k_b.cols != q_nope_dim
        || mla.v_b.head_count != head_count
        || mla.v_b.cols != kv_rank
        || metadata.num_kv_heads * metadata.head_dim != kv_width
    {
        return Err(LlmError::Forward(
            "GLM MLA prefill K/V-B geometry does not match metadata".into(),
        ));
    }

    let slot_count = seq_len * head_count;
    let mut q_absorbed = vec![0.0f32; slot_count * kv_rank];
    let mut q_pe = vec![0.0f32; slot_count * rope_dim];
    let kv_norm_weight = kernels::tensor_as_f32_slice(&mla.kv_a_norm);
    let mut compressed_k = vec![0.0f32; seq_len * kv_width];
    let zero_v = vec![0.0f32; kv_width];

    // pm113: k_b 를 (token, head) slot 전체 단일 Metal dispatch 로 배치.
    // 미지원이면 아래 per-head CPU 루프가 그대로 처리.
    let kb_batched = {
        let mut q_nope_packed = vec![0.0f32; slot_count * q_nope_dim];
        for slot in 0..slot_count {
            let q_offset = slot * qk_dim;
            q_nope_packed[slot * q_nope_dim..(slot + 1) * q_nope_dim]
                .copy_from_slice(&q[q_offset..q_offset + q_nope_dim]);
        }
        mla.k_b
            .gemv_all_heads_slots(&q_nope_packed, slot_count, &mut q_absorbed)?
    };

    for token in 0..seq_len {
        let rope_pos = pos_start + token;
        let raw = &kv_raw[token * kv_width..(token + 1) * kv_width];
        let compressed = &mut compressed_k[token * kv_width..(token + 1) * kv_width];
        kernels::norm::rms_norm_into(
            &raw[..kv_rank],
            kv_norm_weight,
            metadata.norm_eps,
            &mut compressed[..kv_rank],
        );
        compressed[kv_rank..].copy_from_slice(&raw[kv_rank..]);
        kernels::rope::rope_inplace(
            &mut compressed[kv_rank..],
            rope_pos,
            rope_dim,
            rope_dim,
            metadata.rope_theta,
        );
        for head in 0..head_count {
            let q_offset = (token * head_count + head) * qk_dim;
            let q_head = &q[q_offset..q_offset + qk_dim];
            if !kb_batched {
                let absorbed_offset = (token * head_count + head) * kv_rank;
                mla.k_b.gemv_head(
                    head,
                    &q_head[..q_nope_dim],
                    &mut q_absorbed[absorbed_offset..absorbed_offset + kv_rank],
                )?;
            }
            let pe_offset = (token * head_count + head) * rope_dim;
            let pe = &mut q_pe[pe_offset..pe_offset + rope_dim];
            pe.copy_from_slice(&q_head[q_nope_dim..]);
            kernels::rope::rope_inplace(pe, rope_pos, rope_dim, rope_dim, metadata.rope_theta);
        }
        kv_cache.append(layer_idx, rope_pos, compressed, &zero_v);
    }

    let (cache, _) = kv_cache.get_up_to(layer_idx, pos_start + seq_len);
    let value_dim = mla.v_b.rows_per_head;
    let mut attention_concat = vec![0.0f32; slot_count * value_dim];
    let mut scores = vec![0.0f32; pos_start + seq_len];
    // pm113: v_b 배치를 위해 latent 를 (token, head) slot 전체로 모아 계산 후
    // 단일 Metal dispatch. 미지원이면 slot 별 per-head CPU fallback.
    let mut latent_all = vec![0.0f32; slot_count * kv_rank];
    let scale = 1.0f32 / (qk_dim as f32).sqrt();
    for token in 0..seq_len {
        let attend_len = pos_start + token + 1;
        for head in 0..head_count {
            let absorbed_offset = (token * head_count + head) * kv_rank;
            let q_latent = &q_absorbed[absorbed_offset..absorbed_offset + kv_rank];
            let pe_offset = (token * head_count + head) * rope_dim;
            let q_rope = &q_pe[pe_offset..pe_offset + rope_dim];
            for token_pos in 0..attend_len {
                let cached = &cache[token_pos * kv_width..(token_pos + 1) * kv_width];
                let latent_dot = q_latent
                    .iter()
                    .zip(&cached[..kv_rank])
                    .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                    .sum::<f32>();
                let rope_dot = q_rope
                    .iter()
                    .zip(&cached[kv_rank..])
                    .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                    .sum::<f32>();
                scores[token_pos] = (latent_dot + rope_dot) * scale;
            }
            softmax_inplace(&mut scores[..attend_len]);
            let latent_sum = &mut latent_all[(token * head_count + head) * kv_rank..][..kv_rank];
            for (token_pos, &probability) in scores[..attend_len].iter().enumerate() {
                let cached = &cache[token_pos * kv_width..token_pos * kv_width + kv_rank];
                for (sum, &bits) in latent_sum.iter_mut().zip(cached) {
                    *sum += probability * half::f16::from_bits(bits).to_f32();
                }
            }
        }
    }
    let vb_batched =
        mla.v_b
            .gemv_all_heads_slots(&latent_all, slot_count, &mut attention_concat)?;
    if !vb_batched {
        for slot in 0..slot_count {
            let head = slot % head_count;
            mla.v_b.gemv_head(
                head,
                &latent_all[slot * kv_rank..(slot + 1) * kv_rank],
                &mut attention_concat[slot * value_dim..(slot + 1) * value_dim],
            )?;
        }
    }

    let projected = match mla_dense_prefill_gemv(&mla.o, &attention_concat, seq_len)? {
        Some(out) => out,
        None => mla.o.gemv_vec(&attention_concat)?,
    };
    if projected.len() != hidden_data.len() {
        return Err(LlmError::Forward(
            "GLM MLA prefill output projection length mismatch".into(),
        ));
    }
    let mut attention_output = hidden_data.to_vec();
    kernels::elementwise::add_inplace(&mut attention_output, &projected);
    crate::engine::forward::ffn::forward_prefill_ffn(
        metadata,
        GLM_DSA_ARCH,
        Tensor::from_vec(attention_output, &[seq_len, hidden_dim]),
        base,
        layer_idx,
        seq_len,
        metadata.norm_eps,
    )
}

fn decode_ffn(
    metadata: &ModelMetadata,
    scratch: &mut ScratchBuffers,
    base: &AttentionLayerWeights,
    layer_idx: usize,
) -> Result<()> {
    if let Some(moe) = base.shared_expert_moe.as_ref() {
        return decode_shared_expert_moe(
            scratch,
            GLM_DSA_ARCH,
            &base.ffn_norm,
            moe,
            metadata.hidden_dim,
            metadata.norm_eps,
            layer_idx,
        );
    }

    let hidden_dim = metadata.hidden_dim;
    apply_model_norm_into(
        &scratch.hidden[..hidden_dim],
        kernels::tensor_as_f32_slice(&base.ffn_norm),
        metadata.norm_eps,
        &mut scratch.norm_buf[..hidden_dim],
        GLM_DSA_ARCH,
    );
    let mut gate = base
        .ffn_gate_weight
        .gemv_vec(&scratch.norm_buf[..hidden_dim])?;
    let up = base
        .ffn_up_weight
        .gemv_vec(&scratch.norm_buf[..hidden_dim])?;
    if gate.len() != up.len() {
        return Err(LlmError::Forward(
            "GLM dense FFN gate/up shape mismatch".into(),
        ));
    }
    apply_model_gate_mul_inplace(&mut gate, &up, GLM_DSA_ARCH);
    let down = base.ffn_down_weight.gemv_vec(&gate)?;
    if down.len() != hidden_dim {
        return Err(LlmError::Forward(
            "GLM dense FFN down shape mismatch".into(),
        ));
    }
    kernels::elementwise::add_inplace(&mut scratch.hidden[..hidden_dim], &down);
    Ok(())
}

fn softmax_inplace(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if sum != 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::softmax_inplace;

    #[test]
    fn softmax_is_normalized_and_stable() {
        let mut values = [1000.0, 1001.0, 999.0];
        softmax_inplace(&mut values);
        assert!(values.iter().all(|value| value.is_finite()));
        assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(values[1] > values[0] && values[0] > values[2]);
    }
}
