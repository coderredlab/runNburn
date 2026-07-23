use crate::engine::cpu_runtime::kernels;
use crate::engine::layer_weights::AttentionLayerWeights;
use crate::engine::norm::{
    add_f32_inplace, apply_model_gate_mul_inplace, apply_model_norm_into, apply_plain_rms_norm_into,
};
use crate::engine::quantized_weight_types::QuantizedWeight;
#[cfg(not(feature = "cuda"))]
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
        if used && crate::engine::policy::env_string("RNB_GLM_MLA_VERIFY").as_deref() == Some("1") {
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

    fn gemv_head_batch(
        &self,
        token_count: usize,
        packed_input: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        let expected_input = token_count
            .checked_mul(self.head_count)
            .and_then(|value| value.checked_mul(self.cols))
            .ok_or_else(|| LlmError::Forward("GLM MLA head batch input overflow".into()))?;
        if packed_input.len() != expected_input {
            return Err(LlmError::Forward(format!(
                "GLM MLA head batch input mismatch: got={} expected={expected_input}",
                packed_input.len()
            )));
        }
        let Some(bytes) = self.data.as_bytes() else {
            return Ok(None);
        };
        crate::engine::backend_runtime::glm_mla_head_gemv_batch(
            self.ggml_type,
            bytes,
            self.head_count,
            self.rows_per_head,
            self.cols,
            token_count,
            packed_input,
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
        #[cfg(feature = "cuda")]
        {
            let values = crate::engine::cuda_runtime::decode_gemv(
                self.ggml_type,
                &bytes[start..start + bytes_per_head],
                self.rows_per_head,
                self.cols,
                input,
            )
            .ok_or_else(|| {
                LlmError::Forward(format!(
                    "CUDA GLM MLA head {:?} GEMV is unavailable; CPU fallback is disabled",
                    self.ggml_type
                ))
            })?
            .map_err(LlmError::Forward)?;
            output.copy_from_slice(&values);
        }
        #[cfg(not(feature = "cuda"))]
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

/// pm119: DSA lightning indexer weight (deepseek32 경로가 수식 reference —
/// llama.cpp glm-dsa 는 dense fallback 이라 참조 불가, 저널 pm119 후속 2~3).
/// head 내부 layout 은 [앞 rope(64) | 뒤 nope(64)], rope 는 NEOX. Hadamard
/// 회전은 f32/f16 경로에서 산술 불변이라 생략.
pub(in crate::engine) struct GlmDsaIndexerWeights {
    attn_q_b: QuantizedWeight, // [head_count*key_length × q_rank] Q8_0
    attn_k: QuantizedWeight,   // [key_length × hidden] Q8_0
    k_norm: Tensor,            // [key_length] f32 (LayerNorm weight)
    k_norm_bias: Tensor,       // [key_length] f32 (LayerNorm bias)
    proj: Tensor,              // [head_count × hidden] f32 — head 별 score weight
    head_count: usize,
    key_length: usize,
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
    /// pm119: 5개 indexer 텐서가 모두 있고 메타와 shape 이 일치할 때만 Some.
    indexer: Option<GlmDsaIndexerWeights>,
}

impl GlmDsaAttentionLayerWeights {
    /// pm119: indexer 로드 여부 (엔진 init 의 indexer 캐시 활성 판단용).
    #[cfg(not(feature = "cuda"))]
    pub(in crate::engine) fn indexer_key_len(&self) -> Option<usize> {
        self.indexer.as_ref().map(|idx| idx.key_length)
    }
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
        indexer: load_indexer_weights(model, &prefix, load_f32_weight, load_quantized_weight),
    }
}

/// pm119: DSA indexer 텐서 5개가 모두 존재하고 loader 메타(`glm_indexer`)와
/// shape 이 일치할 때만 로드. 불일치는 경고 후 None (dense 경로 유지).
fn load_indexer_weights(
    model: &LoadedModel,
    prefix: &str,
    load_f32_weight: F32WeightLoader,
    load_quantized_weight: QuantizedWeightLoader,
) -> Option<GlmDsaIndexerWeights> {
    let names = [
        format!("{prefix}.indexer.attn_q_b.weight"),
        format!("{prefix}.indexer.attn_k.weight"),
        format!("{prefix}.indexer.k_norm.weight"),
        format!("{prefix}.indexer.k_norm.bias"),
        format!("{prefix}.indexer.proj.weight"),
    ];
    if !names.iter().all(|n| model.weights.contains_key(n)) {
        return None;
    }
    let meta = model.metadata.glm_indexer?;
    let attn_q_b = load_quantized_weight(model, &names[0]);
    let attn_k = load_quantized_weight(model, &names[1]);
    let k_norm = load_f32_weight(model, &names[2]);
    let k_norm_bias = load_f32_weight(model, &names[3]);
    let proj = load_f32_weight(model, &names[4]);
    let key_length = kernels::tensor_as_f32_slice(&k_norm).len();
    let head_count = if key_length > 0 {
        attn_q_b.rows / key_length
    } else {
        0
    };
    if key_length != meta.key_length
        || head_count != meta.head_count
        || attn_q_b.rows != head_count * key_length
        || attn_k.rows != key_length
        || kernels::tensor_as_f32_slice(&k_norm_bias).len() != key_length
        || kernels::tensor_as_f32_slice(&proj).len() != head_count * attn_k.cols
    {
        eprintln!(
            "[WARN] GLM DSA indexer shape/meta mismatch at {prefix} — indexer disabled for this layer"
        );
        return None;
    }
    Some(GlmDsaIndexerWeights {
        attn_q_b,
        attn_k,
        k_norm,
        k_norm_bias,
        proj,
        head_count,
        key_length,
    })
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
            if crate::engine::policy::env_string("RNB_GLM_MLA_VERIFY").as_deref() == Some("1") {
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
    crate::engine::policy::env_string("RNB_GLM_MLA_PROFILE").as_deref() == Some("1")
}

/// pm116 진단: `RNB_GLM_PREFILL_PROFILE=1` 시 prefill_layer 스테이지별 wall ms 를
/// 레이어당 1줄로 출력. 합산은 로그 후처리(awk)로 한다.
pub(in crate::engine) fn prefill_profile_enabled() -> bool {
    crate::engine::policy::env_string("RNB_GLM_PREFILL_PROFILE").as_deref() == Some("1")
}

/// pm119: DSA indexer probe (`RNB_GLM_DSA_INDEXER_PROBE=1`) — prefill 배치의
/// 마지막 토큰이 배치 내 전 토큰에 갖는 indexer score 를 CPU 로 계산해
/// 분포를 로그한다 (수식 스모크, selected-set 통합 전 발판).
fn indexer_probe_enabled() -> bool {
    crate::engine::policy::env_string("RNB_GLM_DSA_INDEXER_PROBE").as_deref() == Some("1")
}

/// NEOX rope: 쌍 (i, i+n_rot/2), angle_i = pos·theta^(-2i/n_rot).
/// (deepseek32 indexer 는 메인 attention 의 인접페어 rope 와 달리 NEOX.)
fn indexer_neox_rope(vals: &mut [f32], pos: usize, n_rot: usize, theta: f32) {
    #[cfg(feature = "cuda")]
    {
        crate::engine::backend_runtime::cuda_rope_f32_inplace(
            vals,
            n_rot,
            n_rot,
            n_rot,
            pos,
            theta,
            crate::engine::backend_runtime::CudaForwardRopeMode::Neox,
            None,
        )
        .unwrap_or_else(|err| panic!("CUDA GLM indexer NEOX RoPE failed: {err}"));
    }
    #[cfg(not(feature = "cuda"))]
    {
        let half = n_rot / 2;
        for i in 0..half {
            let angle = (pos as f32) * theta.powf(-2.0 * i as f32 / n_rot as f32);
            let (c, s) = (angle.cos(), angle.sin());
            let (x0, x1) = (vals[i], vals[i + half]);
            vals[i] = x0 * c - x1 * s;
            vals[i + half] = x0 * s + x1 * c;
        }
    }
}

fn glm_adjacent_rope(vals: &mut [f32], pos: usize, head_dim: usize, n_rot: usize, theta: f32) {
    #[cfg(feature = "cuda")]
    {
        crate::engine::backend_runtime::cuda_rope_f32_inplace(
            vals,
            vals.len(),
            head_dim,
            n_rot,
            pos,
            theta,
            crate::engine::backend_runtime::CudaForwardRopeMode::Adjacent,
            None,
        )
        .unwrap_or_else(|err| panic!("CUDA GLM adjacent RoPE failed: {err}"));
    }
    #[cfg(not(feature = "cuda"))]
    kernels::rope::rope_inplace(vals, pos, head_dim, n_rot, theta);
}

/// pm119 2단계: token 하나의 indexer key row — attn_k gemv → LayerNorm(+bias)
/// → NEOX rope. probe 와 캐시 기록이 공유하는 참조 구현.
fn indexer_k_row(
    idx: &GlmDsaIndexerWeights,
    normed_token: &[f32],
    pos: usize,
    rope_dim: usize,
    theta: f32,
    norm_eps: f32,
) -> Result<Vec<f32>> {
    let key_len = idx.key_length;
    let n_rot = rope_dim.min(key_len);
    let raw = idx.attn_k.gemv_vec(normed_token)?;
    let k_norm_w = kernels::tensor_as_f32_slice(&idx.k_norm);
    let k_norm_b = kernels::tensor_as_f32_slice(&idx.k_norm_bias);
    let mean = raw.iter().sum::<f32>() / key_len as f32;
    let var = raw.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / key_len as f32;
    let inv = 1.0f32 / (var + norm_eps).sqrt();
    let mut k = vec![0.0f32; key_len];
    for i in 0..key_len {
        k[i] = (raw[i] - mean) * inv * k_norm_w[i] + k_norm_b[i];
    }
    indexer_neox_rope(&mut k[..n_rot], pos, n_rot, theta);
    Ok(k)
}

/// pm119: indexer score (deepseek32 수식 reference, 저널 pm119 후속 2~3).
/// `score_j = Σ_h ReLU(q_h·k_j) · w_h · 1/√(key_len·heads)`. head layout 은
/// [앞 rope(n_rot) | 뒤 nope], k 는 일반 LayerNorm(+bias). Hadamard 생략
/// (f32 경로 산술 불변). 반환: 마지막 토큰의 배치 내 score `[seq_len]`.
#[allow(clippy::too_many_arguments)]
fn indexer_probe_scores(
    idx: &GlmDsaIndexerWeights,
    normed: &[f32],
    q_rank_norm_last: &[f32],
    seq_len: usize,
    hidden: usize,
    pos_start: usize,
    rope_dim: usize,
    theta: f32,
    norm_eps: f32,
) -> Result<Vec<f32>> {
    let heads = idx.head_count;
    let key_len = idx.key_length;
    let n_rot = rope_dim.min(key_len);
    let last = seq_len - 1;
    // q (마지막 토큰): q_rank_norm @ attn_q_b → heads×key_len, head 별 rope.
    let mut q = idx.attn_q_b.gemv_vec(q_rank_norm_last)?;
    for head in 0..heads {
        indexer_neox_rope(
            &mut q[head * key_len..head * key_len + n_rot],
            pos_start + last,
            n_rot,
            theta,
        );
    }
    // w (마지막 토큰): proj[h]·normed_last, pre-scale.
    let proj = kernels::tensor_as_f32_slice(&idx.proj);
    let normed_last = &normed[last * hidden..(last + 1) * hidden];
    let scale = 1.0f32 / ((key_len * heads) as f32).sqrt();
    let weights: Vec<f32> = (0..heads)
        .map(|h| {
            proj[h * hidden..(h + 1) * hidden]
                .iter()
                .zip(normed_last)
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                * scale
        })
        .collect();
    // k_j (전 토큰): 공유 참조 구현 (`indexer_k_row`).
    let mut scores = vec![0.0f32; seq_len];
    for j in 0..seq_len {
        let k = indexer_k_row(
            idx,
            &normed[j * hidden..(j + 1) * hidden],
            pos_start + j,
            n_rot,
            theta,
            norm_eps,
        )?;
        let mut score = 0.0f32;
        for (h, &w) in weights.iter().enumerate() {
            let dot = q[h * key_len..(h + 1) * key_len]
                .iter()
                .zip(&k)
                .map(|(&a, &b)| a * b)
                .sum::<f32>();
            score += dot.max(0.0) * w;
        }
        scores[j] = score;
    }
    Ok(scores)
}

/// pm119 2b: indexer top-k selected-set attention (CPU 참조 구현).
/// 토큰별로 indexer score → top-k 선택 → 선택된 위치만 main MLA attention.
/// 산술은 기존 CPU 스칼라 fallback 과 동일하되 j 가 selected 로 제한된다.
#[allow(clippy::too_many_arguments)]
fn selected_set_attention(
    idx: &GlmDsaIndexerWeights,
    indexer_cache: &[u16],
    top_k_limit: usize,
    q_absorbed: &[f32],
    q_pe: &[f32],
    q_rank_norm: &[f32],
    normed: &[f32],
    cache: &[u16],
    pos_start: usize,
    head_count: usize,
    kv_rank: usize,
    rope_dim: usize,
    hidden: usize,
    theta: f32,
    scale: f32,
    latent_all: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;
    let key_len = idx.key_length;
    let idx_heads = idx.head_count;
    let n_rot = rope_dim.min(key_len);
    let kv_width = kv_rank + rope_dim;
    let q_rank = idx.attn_q_b.cols;
    let proj = kernels::tensor_as_f32_slice(&idx.proj);
    let pre_scale = 1.0f32 / ((key_len * idx_heads) as f32).sqrt();

    let results: Vec<Result<()>> = latent_all
        .par_chunks_mut(head_count * kv_rank)
        .enumerate()
        .map(|(token, latent_token)| -> Result<()> {
            let attend_len = pos_start + token + 1;
            // indexer q (이 토큰): q_rank_norm @ attn_q_b + head 별 rope.
            let mut iq = idx
                .attn_q_b
                .gemv_vec(&q_rank_norm[token * q_rank..(token + 1) * q_rank])?;
            for head in 0..idx_heads {
                indexer_neox_rope(
                    &mut iq[head * key_len..head * key_len + n_rot],
                    pos_start + token,
                    n_rot,
                    theta,
                );
            }
            let normed_token = &normed[token * hidden..(token + 1) * hidden];
            let weights: Vec<f32> = (0..idx_heads)
                .map(|h| {
                    proj[h * hidden..(h + 1) * hidden]
                        .iter()
                        .zip(normed_token)
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                        * pre_scale
                })
                .collect();
            // indexer score over [0, attend_len).
            let mut score_idx: Vec<(f32, usize)> = (0..attend_len)
                .map(|j| {
                    let k_row = &indexer_cache[j * key_len..(j + 1) * key_len];
                    let mut score = 0.0f32;
                    for (h, &w) in weights.iter().enumerate() {
                        let dot = iq[h * key_len..(h + 1) * key_len]
                            .iter()
                            .zip(k_row)
                            .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                            .sum::<f32>();
                        score += dot.max(0.0) * w;
                    }
                    (score, j)
                })
                .collect();
            // top-k = min(attend_len, top_k_limit) 선택.
            let keep = attend_len.min(top_k_limit);
            if keep < attend_len {
                score_idx.select_nth_unstable_by(keep - 1, |a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                score_idx.truncate(keep);
            }
            // main MLA attention (selected 만) — 기존 스칼라 fallback 과 동일 산술.
            let mut scores = vec![0.0f32; keep];
            for head in 0..head_count {
                let slot = token * head_count + head;
                let q_latent = &q_absorbed[slot * kv_rank..(slot + 1) * kv_rank];
                let q_rope = &q_pe[slot * rope_dim..(slot + 1) * rope_dim];
                for (s, &(_, j)) in scores.iter_mut().zip(&score_idx) {
                    let cached = &cache[j * kv_width..(j + 1) * kv_width];
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
                    *s = (latent_dot + rope_dot) * scale;
                }
                softmax_inplace(&mut scores);
                let latent_sum = &mut latent_token[head * kv_rank..(head + 1) * kv_rank];
                latent_sum.fill(0.0);
                for (&p, &(_, j)) in scores.iter().zip(&score_idx) {
                    let cached = &cache[j * kv_width..j * kv_width + kv_rank];
                    for (sum, &bits) in latent_sum.iter_mut().zip(cached) {
                        *sum += p * half::f16::from_bits(bits).to_f32();
                    }
                }
            }
            Ok(())
        })
        .collect();
    for r in results {
        r?;
    }
    Ok(())
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

    // pm119 2단계: decode 토큰의 indexer key 캐시 기록 (opt-in).
    if kv_cache.glm_indexer_enabled() {
        if let Some(idx) = &mla.indexer {
            let k = indexer_k_row(
                idx,
                &scratch.norm_buf[..hidden_dim],
                rope_pos,
                metadata.rope_dim,
                metadata.rope_theta,
                metadata.norm_eps,
            )?;
            kv_cache.write_glm_indexer_k(kv_layer_idx, rope_pos, &k);
        }
    }
    // pm119 2b: attend 가 top_k 를 넘으면 selected-set — indexer q/weights 를
    // norm_buf 가 유효한 지금 미리 계산해 둔다 (1토큰이라 재계산 비용 무시).
    let indexer_top_k = kv_cache.glm_indexer_top_k();
    let decode_selected_setup: Option<(Vec<f32>, Vec<f32>)> =
        if kv_cache.glm_indexer_enabled() && indexer_top_k > 0 && cache_pos + 1 > indexer_top_k {
            mla.indexer
                .as_ref()
                .map(|idx| -> Result<(Vec<f32>, Vec<f32>)> {
                    let normed_token = &scratch.norm_buf[..hidden_dim];
                    let q_a = mla.q_a.gemv_vec(normed_token)?;
                    let mut q_rank_norm = vec![0.0f32; q_a.len()];
                    apply_plain_rms_norm_into(
                        &q_a,
                        kernels::tensor_as_f32_slice(&mla.q_a_norm),
                        metadata.norm_eps,
                        &mut q_rank_norm,
                    );
                    let key_len = idx.key_length;
                    let n_rot = metadata.rope_dim.min(key_len);
                    let mut iq = idx.attn_q_b.gemv_vec(&q_rank_norm)?;
                    for head in 0..idx.head_count {
                        indexer_neox_rope(
                            &mut iq[head * key_len..head * key_len + n_rot],
                            rope_pos,
                            n_rot,
                            metadata.rope_theta,
                        );
                    }
                    let proj = kernels::tensor_as_f32_slice(&idx.proj);
                    let pre_scale = 1.0f32 / ((key_len * idx.head_count) as f32).sqrt();
                    let weights: Vec<f32> = (0..idx.head_count)
                        .map(|h| {
                            proj[h * hidden_dim..(h + 1) * hidden_dim]
                                .iter()
                                .zip(normed_token)
                                .map(|(&a, &b)| a * b)
                                .sum::<f32>()
                                * pre_scale
                        })
                        .collect();
                    Ok((iq, weights))
                })
                .transpose()?
        } else {
            None
        };

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
        apply_plain_rms_norm_into(
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
    apply_plain_rms_norm_into(
        &kv_raw[..kv_rank],
        kernels::tensor_as_f32_slice(&mla.kv_a_norm),
        metadata.norm_eps,
        &mut kv_latent,
    );
    let mut k_pe = kv_raw[kv_rank..].to_vec();
    glm_adjacent_rope(&mut k_pe, rope_pos, rope_dim, rope_dim, metadata.rope_theta);

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
            glm_adjacent_rope(pe, rope_pos, rope_dim, rope_dim, metadata.rope_theta);
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
            glm_adjacent_rope(pe, rope_pos, rope_dim, rope_dim, metadata.rope_theta);
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
    let scale = 1.0f32 / (qk_dim as f32).sqrt();
    let stage_start = std::time::Instant::now();
    #[cfg(feature = "cuda")]
    let latent_all = {
        if decode_selected_setup.is_some() {
            return Err(LlmError::Forward(
                "GLM DSA indexer selected attention has no CUDA implementation; CPU fallback is disabled"
                    .into(),
            ));
        }
        crate::engine::backend_runtime::glm_mla_prefill_attention(
            &q_absorbed,
            &q_pe,
            cache,
            cache_pos,
            1,
            head_count,
            cache_pos + 1,
            kv_rank,
            rope_dim,
            scale,
        )
        .map_err(LlmError::Forward)?
        .ok_or_else(|| {
            LlmError::Forward(
                "CUDA GLM MLA decode attention is unavailable; CPU fallback is disabled".into(),
            )
        })?
    };
    #[cfg(not(feature = "cuda"))]
    let latent_all = {
        let attend_positions: Vec<usize> = match &decode_selected_setup {
            Some((iq, weights)) => {
                let idx = mla
                    .indexer
                    .as_ref()
                    .expect("decode_selected_setup implies indexer weights");
                let key_len = idx.key_length;
                let indexer_cache = kv_cache.glm_indexer_k_up_to(kv_layer_idx, cache_pos + 1);
                let mut score_idx: Vec<(f32, usize)> = (0..=cache_pos)
                    .map(|j| {
                        let k_row = &indexer_cache[j * key_len..(j + 1) * key_len];
                        let mut score = 0.0f32;
                        for (h, &w) in weights.iter().enumerate() {
                            let dot = iq[h * key_len..(h + 1) * key_len]
                                .iter()
                                .zip(k_row)
                                .map(|(&a, &b)| a * half::f16::from_bits(b).to_f32())
                                .sum::<f32>();
                            score += dot.max(0.0) * w;
                        }
                        (score, j)
                    })
                    .collect();
                let keep = (cache_pos + 1).min(indexer_top_k);
                if keep < cache_pos + 1 {
                    score_idx.select_nth_unstable_by(keep - 1, |a, b| {
                        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    score_idx.truncate(keep);
                }
                score_idx.into_iter().map(|(_, j)| j).collect()
            }
            None => (0..=cache_pos).collect(),
        };
        let mut latent_all = vec![0.0f32; head_count * kv_rank];
        let mut scores = vec![0.0f32; attend_positions.len()];
        for head in 0..head_count {
            let q_latent = &q_absorbed[head * kv_rank..(head + 1) * kv_rank];
            let q_rope = &q_pe[head * rope_dim..(head + 1) * rope_dim];
            for (s, &token_pos) in scores.iter_mut().zip(&attend_positions) {
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
                *s = (latent_dot + rope_dot) * scale;
            }
            softmax_inplace(&mut scores);
            let latent_sum = &mut latent_all[head * kv_rank..(head + 1) * kv_rank];
            latent_sum.fill(0.0);
            for (&probability, &token_pos) in scores.iter().zip(&attend_positions) {
                let cached = &cache[token_pos * cache_stride..token_pos * cache_stride + kv_rank];
                for (sum, &bits) in latent_sum.iter_mut().zip(cached) {
                    *sum += probability * half::f16::from_bits(bits).to_f32();
                }
            }
        }
        latent_all
    };
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
    add_f32_inplace(&mut scratch.hidden[..hidden_dim], &projected);

    let stage_start = std::time::Instant::now();
    let ffn_result = decode_ffn(metadata, scratch, base, model_layer_idx);
    if profiling {
        mla_stage_profile(7, stage_start.elapsed().as_secs_f64() * 1000.0);
    }
    ffn_result
}

/// pm114: prefill dense GEMV 를 Metal slot-batch(heads=1, slot=token) 커널로
/// 시도. Q8_0(pm114)/Q5_K(pm117 — o/q_a) 지원, 미지원 quant/빌드면 None 으로
/// CPU gemv_vec fallback.
fn mla_dense_prefill_gemv(
    weight: &QuantizedWeight,
    input: &[f32],
    seq_len: usize,
) -> Result<Option<Vec<f32>>> {
    if seq_len <= 1 {
        return Ok(None);
    }
    if crate::engine::policy::env_string("RNB_METAL_GLM_MLA_PREFILL_DENSE").as_deref() == Some("0")
    {
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

    let profiling = prefill_profile_enabled();
    let mut stage_ms = [0.0f64; 7];
    let mut mark = std::time::Instant::now();

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
    let kv_rank = kernels::tensor_as_f32_slice(&mla.kv_a_norm).len();
    let rope_dim = metadata.rope_dim;
    let kv_width = kv_rank + rope_dim;

    // pm119 2단계: indexer key 캐시 기록 (opt-in `RNB_GLM_DSA_INDEXER=1` 로
    // 엔진 init 에서 활성). fused/비-fused 공통 지점 — CPU 참조 구현.
    if kv_cache.glm_indexer_enabled() {
        if let Some(idx) = &mla.indexer {
            for token in 0..seq_len {
                let k = indexer_k_row(
                    idx,
                    &normed[token * hidden_dim..(token + 1) * hidden_dim],
                    pos_start + token,
                    rope_dim,
                    metadata.rope_theta,
                    metadata.norm_eps,
                )?;
                kv_cache.write_glm_indexer_k(layer_idx, pos_start + token, &k);
            }
        }
    }

    // pm119: DSA indexer probe — 진단 전용 (성능 경로 밖, CPU 재계산 자족).
    if indexer_probe_enabled() && seq_len > 1 {
        if let Some(idx) = &mla.indexer {
            let last = seq_len - 1;
            let q_a_last = mla
                .q_a
                .gemv_vec(&normed[last * hidden_dim..(last + 1) * hidden_dim])?;
            let mut q_rank_norm_last = vec![0.0f32; q_rank];
            apply_plain_rms_norm_into(
                &q_a_last,
                kernels::tensor_as_f32_slice(&mla.q_a_norm),
                metadata.norm_eps,
                &mut q_rank_norm_last,
            );
            let scores = indexer_probe_scores(
                idx,
                &normed,
                &q_rank_norm_last,
                seq_len,
                hidden_dim,
                pos_start,
                rope_dim,
                metadata.rope_theta,
                metadata.norm_eps,
            )?;
            let (mut min, mut max, mut sum, mut pos_cnt) =
                (f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0usize);
            for &s in &scores {
                min = min.min(s);
                max = max.max(s);
                sum += s as f64;
                if s > 0.0 {
                    pos_cnt += 1;
                }
            }
            eprintln!(
                "[glm-dsa-indexer-probe] layer={layer_idx} seq={seq_len} pos_start={pos_start} heads={} key_len={} score min={min:.4} max={max:.4} mean={:.4} pos_frac={:.3}",
                idx.head_count,
                idx.key_length,
                sum / seq_len as f64,
                pos_cnt as f32 / seq_len as f32,
            );
        } else if layer_idx == 0 {
            eprintln!("[glm-dsa-indexer-probe] indexer weights not loaded — probe skipped");
        }
    }

    // pm119 2b: attend 총 길이가 indexer top_k 를 넘으면 selected-set
    // attention 이 필요 — dense 전제인 fused/GPU attn 경로를 회피하고
    // 비-fused + CPU selected 경로로 간다.
    let attend_total = pos_start + seq_len;
    let indexer_top_k = kv_cache.glm_indexer_top_k();
    let use_selected = kv_cache.glm_indexer_enabled()
        && indexer_top_k > 0
        && attend_total > indexer_top_k
        && mla.indexer.is_some();

    // pm119: MLA 층 전체 (front→kv rms/rope→q_pe rope→attn→v_b→o) 를 단일
    // command buffer 로 시도. 성공 시 cache append + residual + ffn 만 CPU —
    // 층당 GPU 왕복 1. 전체 시간은 dense(stage 0) 에 기록.
    if !use_selected && seq_len > 1 && mla.kv_a.rows == kv_width && mla.k_b.rows_per_head == kv_rank
    {
        if let (
            Some(qa_bytes),
            Some(qb_bytes),
            Some(kva_bytes),
            Some(kb_bytes),
            Some(vb_bytes),
            Some(o_bytes),
        ) = (
            mla.q_a.data.as_bytes(),
            mla.q_b.data.as_bytes(),
            mla.kv_a.data.as_bytes(),
            mla.k_b.data.as_bytes(),
            mla.v_b.data.as_bytes(),
            mla.o.data.as_bytes(),
        ) {
            let q_dim = mla.q_b.rows;
            let fused_heads = mla.k_b.head_count;
            let fused_qk_dim = if fused_heads > 0 {
                q_dim / fused_heads
            } else {
                0
            };
            let fused_result = {
                let (cache_base, _) = kv_cache.get_up_to(layer_idx, pos_start);
                crate::engine::backend_runtime::glm_mla_layer_fused(
                    [
                        mla.q_a.ggml_type,
                        mla.q_b.ggml_type,
                        mla.kv_a.ggml_type,
                        mla.k_b.ggml_type,
                        mla.v_b.ggml_type,
                        mla.o.ggml_type,
                    ],
                    qa_bytes,
                    kernels::tensor_as_f32_slice(&mla.q_a_norm),
                    qb_bytes,
                    kva_bytes,
                    kb_bytes,
                    kernels::tensor_as_f32_slice(&mla.kv_a_norm),
                    cache_base,
                    vb_bytes,
                    o_bytes,
                    &normed,
                    seq_len,
                    hidden_dim,
                    q_rank,
                    q_dim,
                    fused_heads,
                    mla.k_b.cols,
                    kv_rank,
                    rope_dim,
                    pos_start,
                    mla.v_b.rows_per_head,
                    mla.o.rows,
                    mla.o.cols,
                    metadata.norm_eps,
                    metadata.rope_theta,
                    1.0f32 / (fused_qk_dim as f32).max(1.0).sqrt(),
                )
                .map_err(LlmError::Forward)?
            };
            if let Some(out) = fused_result {
                if out.projected.len() != hidden_data.len()
                    || out.cache_tail.len() != seq_len * kv_width
                {
                    return Err(LlmError::Forward(
                        "GLM MLA layer fused output shape mismatch".into(),
                    ));
                }
                let zero_v = vec![0.0f32; kv_width];
                let mut compressed = vec![0.0f32; kv_width];
                for token in 0..seq_len {
                    let row = &out.cache_tail[token * kv_width..(token + 1) * kv_width];
                    for (dst, &bits) in compressed.iter_mut().zip(row) {
                        *dst = half::f16::from_bits(bits).to_f32();
                    }
                    kv_cache.append(layer_idx, pos_start + token, &compressed, &zero_v);
                }
                if profiling {
                    stage_ms[0] = mark.elapsed().as_secs_f64() * 1000.0;
                    mark = std::time::Instant::now();
                }
                let mut attention_output = hidden_data.to_vec();
                add_f32_inplace(&mut attention_output, &out.projected);
                if profiling {
                    stage_ms[5] = mark.elapsed().as_secs_f64() * 1000.0;
                    mark = std::time::Instant::now();
                }
                let result = crate::engine::forward::ffn::forward_prefill_ffn(
                    metadata,
                    GLM_DSA_ARCH,
                    Tensor::from_vec(attention_output, &[seq_len, hidden_dim]),
                    base,
                    layer_idx,
                    seq_len,
                    metadata.norm_eps,
                );
                if profiling {
                    stage_ms[6] = mark.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "[glm-prefill-profile] layer={layer_idx} seq={seq_len} dense={:.1} kb={:.1} rope={:.1} attn={:.1} vb={:.1} o={:.1} ffn={:.1}",
                        stage_ms[0], stage_ms[1], stage_ms[2], stage_ms[3], stage_ms[4], stage_ms[5], stage_ms[6],
                    );
                }
                return result;
            }
        }
    }

    // pm118 연장: front 4 dispatch (q_a→rms→q_b/kv_a→pack→k_b) 를 단일
    // command buffer 로. 성공 시 (q, kv_raw, q_absorbed) 를 한 번에 받고
    // dense(stage 0)에 전체 시간이 기록되며 kb(stage 1)는 0 이 된다.
    let mut front_fused: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> = None;
    if !use_selected && seq_len > 1 && mla.kv_a.rows == kv_width && mla.k_b.rows_per_head == kv_rank
    {
        if let (Some(qa_bytes), Some(qb_bytes), Some(kva_bytes), Some(kb_bytes)) = (
            mla.q_a.data.as_bytes(),
            mla.q_b.data.as_bytes(),
            mla.kv_a.data.as_bytes(),
            mla.k_b.data.as_bytes(),
        ) {
            let q_dim = mla.q_b.rows;
            let fused_heads = mla.k_b.head_count;
            let mut q_buf = vec![0.0f32; seq_len * q_dim];
            let mut kv_buf = vec![0.0f32; seq_len * kv_width];
            let mut qabs_buf = vec![0.0f32; seq_len * fused_heads * kv_rank];
            if crate::engine::backend_runtime::glm_mla_front_slots_fused_into(
                mla.q_a.ggml_type,
                mla.q_b.ggml_type,
                mla.kv_a.ggml_type,
                mla.k_b.ggml_type,
                qa_bytes,
                kernels::tensor_as_f32_slice(&mla.q_a_norm),
                qb_bytes,
                kva_bytes,
                kb_bytes,
                &normed,
                seq_len,
                hidden_dim,
                q_rank,
                q_dim,
                kv_width,
                fused_heads,
                mla.k_b.cols,
                kv_rank,
                metadata.norm_eps,
                &mut q_buf,
                &mut kv_buf,
                &mut qabs_buf,
            )
            .map_err(LlmError::Forward)?
            {
                front_fused = Some((q_buf, kv_buf, qabs_buf));
            }
        }
    }

    let (q, kv_raw, fused_qabs, q_rank_norm_kept) = match front_fused {
        Some((q, kv_raw, qabs)) => (q, kv_raw, Some(qabs), None),
        None => {
            let q_a = match mla_dense_prefill_gemv(&mla.q_a, &normed, seq_len)? {
                Some(out) => out,
                None => mla.q_a.gemv_vec(&normed)?,
            };
            if q_a.len() != seq_len * q_rank {
                return Err(LlmError::Forward(
                    "GLM MLA prefill q_a output length mismatch".into(),
                ));
            }
            let q_a_norm_weight = kernels::tensor_as_f32_slice(&mla.q_a_norm);
            let mut q_rank_norm = vec![0.0f32; q_a.len()];
            for token in 0..seq_len {
                apply_plain_rms_norm_into(
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
            let kv_raw = match mla_dense_prefill_gemv(&mla.kv_a, &normed, seq_len)? {
                Some(out) => out,
                None => mla.kv_a.gemv_vec(&normed)?,
            };
            // pm119 2b: selected-set 경로가 indexer q 를 만들 때 재사용.
            let kept = use_selected.then_some(q_rank_norm);
            (q, kv_raw, None, kept)
        }
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

    if profiling {
        stage_ms[0] = mark.elapsed().as_secs_f64() * 1000.0;
        mark = std::time::Instant::now();
    }

    let slot_count = seq_len * head_count;
    let q_absorbed = match fused_qabs {
        Some(qabs) => qabs,
        None => {
            let mut q_nope_packed = vec![0.0f32; slot_count * q_nope_dim];
            for slot in 0..slot_count {
                let q_offset = slot * qk_dim;
                q_nope_packed[slot * q_nope_dim..(slot + 1) * q_nope_dim]
                    .copy_from_slice(&q[q_offset..q_offset + q_nope_dim]);
            }
            let mut q_absorbed = vec![0.0f32; slot_count * kv_rank];
            let mut kb_batched =
                mla.k_b
                    .gemv_all_heads_slots(&q_nope_packed, slot_count, &mut q_absorbed)?;
            if !kb_batched {
                if let Some(output) = mla.k_b.gemv_head_batch(seq_len, &q_nope_packed)? {
                    q_absorbed = output;
                    kb_batched = true;
                }
            }
            if !kb_batched {
                for slot in 0..slot_count {
                    let head = slot % head_count;
                    mla.k_b.gemv_head(
                        head,
                        &q_nope_packed[slot * q_nope_dim..(slot + 1) * q_nope_dim],
                        &mut q_absorbed[slot * kv_rank..(slot + 1) * kv_rank],
                    )?;
                }
            }
            q_absorbed
        }
    };

    if profiling {
        stage_ms[1] = mark.elapsed().as_secs_f64() * 1000.0;
        mark = std::time::Instant::now();
    }

    let mut q_pe = vec![0.0f32; slot_count * rope_dim];
    let kv_norm_weight = kernels::tensor_as_f32_slice(&mla.kv_a_norm);
    let mut compressed_k = vec![0.0f32; seq_len * kv_width];
    let zero_v = vec![0.0f32; kv_width];
    for token in 0..seq_len {
        let rope_pos = pos_start + token;
        let raw = &kv_raw[token * kv_width..(token + 1) * kv_width];
        let compressed = &mut compressed_k[token * kv_width..(token + 1) * kv_width];
        apply_plain_rms_norm_into(
            &raw[..kv_rank],
            kv_norm_weight,
            metadata.norm_eps,
            &mut compressed[..kv_rank],
        );
        compressed[kv_rank..].copy_from_slice(&raw[kv_rank..]);
        glm_adjacent_rope(
            &mut compressed[kv_rank..],
            rope_pos,
            rope_dim,
            rope_dim,
            metadata.rope_theta,
        );
        for head in 0..head_count {
            let q_offset = (token * head_count + head) * qk_dim;
            let q_head = &q[q_offset..q_offset + qk_dim];
            let pe_offset = (token * head_count + head) * rope_dim;
            let pe = &mut q_pe[pe_offset..pe_offset + rope_dim];
            pe.copy_from_slice(&q_head[q_nope_dim..]);
            glm_adjacent_rope(pe, rope_pos, rope_dim, rope_dim, metadata.rope_theta);
        }
        kv_cache.append(layer_idx, rope_pos, compressed, &zero_v);
    }

    if profiling {
        stage_ms[2] = mark.elapsed().as_secs_f64() * 1000.0;
        mark = std::time::Instant::now();
    }

    let (cache, _) = kv_cache.get_up_to(layer_idx, pos_start + seq_len);
    let value_dim = mla.v_b.rows_per_head;
    let scale = 1.0f32 / (qk_dim as f32).sqrt();
    let mut latent_all = vec![0.0f32; slot_count * kv_rank];
    // pm119 2b: attend 가 top_k 를 넘으면 indexer top-k selected-set CPU
    // attention (dense 전제인 GPU attn 커널은 스킵). CPU 참조 구현 —
    // correctness 확정 후 Metal 화.
    #[cfg(feature = "cuda")]
    if use_selected {
        return Err(LlmError::Forward(
            "GLM DSA indexer selected prefill attention has no CUDA implementation; CPU fallback is disabled"
                .into(),
        ));
    }
    let mut attn_batched = if use_selected {
        let idx = mla
            .indexer
            .as_ref()
            .expect("use_selected implies indexer weights");
        let q_rank_norm = q_rank_norm_kept.as_deref().ok_or_else(|| {
            LlmError::Forward("GLM DSA selected-set requires the non-fused front path".into())
        })?;
        let indexer_cache = kv_cache.glm_indexer_k_up_to(layer_idx, pos_start + seq_len);
        selected_set_attention(
            idx,
            indexer_cache,
            indexer_top_k,
            &q_absorbed,
            &q_pe,
            q_rank_norm,
            &normed,
            cache,
            pos_start,
            head_count,
            kv_rank,
            rope_dim,
            hidden_dim,
            metadata.rope_theta,
            scale,
            &mut latent_all,
        )?;
        true
    } else {
        false
    };
    attn_batched = attn_batched
        || crate::engine::backend_runtime::glm_mla_prefill_attn_into(
            &q_absorbed,
            &q_pe,
            cache,
            slot_count,
            head_count,
            kv_rank,
            rope_dim,
            pos_start,
            scale,
            &mut latent_all,
        )
        .map_err(LlmError::Forward)?;
    if !attn_batched {
        if let Some(output) = crate::engine::backend_runtime::glm_mla_prefill_attention(
            &q_absorbed,
            &q_pe,
            cache,
            pos_start,
            seq_len,
            head_count,
            pos_start + seq_len,
            kv_rank,
            rope_dim,
            scale,
        )
        .map_err(LlmError::Forward)?
        {
            latent_all = output;
            attn_batched = true;
        }
    }
    if !attn_batched {
        let mut scores = vec![0.0f32; pos_start + seq_len];
        for token in 0..seq_len {
            let attend_len = pos_start + token + 1;
            for head in 0..head_count {
                let slot = token * head_count + head;
                let q_latent = &q_absorbed[slot * kv_rank..(slot + 1) * kv_rank];
                let q_rope = &q_pe[slot * rope_dim..(slot + 1) * rope_dim];
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
                let latent_sum = &mut latent_all[slot * kv_rank..(slot + 1) * kv_rank];
                latent_sum.fill(0.0);
                for (token_pos, &probability) in scores[..attend_len].iter().enumerate() {
                    let cached = &cache[token_pos * kv_width..token_pos * kv_width + kv_rank];
                    for (sum, &bits) in latent_sum.iter_mut().zip(cached) {
                        *sum += probability * half::f16::from_bits(bits).to_f32();
                    }
                }
            }
        }
    }

    if profiling {
        stage_ms[3] = mark.elapsed().as_secs_f64() * 1000.0;
        mark = std::time::Instant::now();
    }

    // pm118: v_b slots → o slots 를 단일 command buffer 로 (slots 커널 유지,
    // attention_concat 은 device-resident — commit+wait 왕복 1회 제거).
    // fused 성공 시 vb 스테이지(stage 4)에 전체 시간이 기록되고 o(stage 5)는 0.
    let mut projected_fused: Option<Vec<f32>> = None;
    if seq_len > 1 {
        if let (Some(vb_bytes), Some(o_bytes)) = (mla.v_b.data.as_bytes(), mla.o.data.as_bytes()) {
            let mut out = vec![0.0f32; seq_len * mla.o.rows];
            if crate::engine::backend_runtime::glm_mla_vb_o_fused_into(
                mla.v_b.ggml_type,
                mla.o.ggml_type,
                vb_bytes,
                o_bytes,
                slot_count,
                head_count,
                value_dim,
                kv_rank,
                mla.o.rows,
                mla.o.cols,
                &latent_all,
                &mut out,
            )
            .map_err(LlmError::Forward)?
            {
                projected_fused = Some(out);
            }
        }
    }
    let projected = match projected_fused {
        Some(out) => {
            if profiling {
                stage_ms[4] = mark.elapsed().as_secs_f64() * 1000.0;
                mark = std::time::Instant::now();
            }
            out
        }
        None => {
            let mut attention_concat = vec![0.0f32; slot_count * value_dim];
            let mut vb_batched =
                mla.v_b
                    .gemv_all_heads_slots(&latent_all, slot_count, &mut attention_concat)?;
            if !vb_batched {
                if let Some(output) = mla.v_b.gemv_head_batch(seq_len, &latent_all)? {
                    attention_concat = output;
                    vb_batched = true;
                }
            }
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

            if profiling {
                stage_ms[4] = mark.elapsed().as_secs_f64() * 1000.0;
                mark = std::time::Instant::now();
            }

            match mla_dense_prefill_gemv(&mla.o, &attention_concat, seq_len)? {
                Some(out) => out,
                None => mla.o.gemv_vec(&attention_concat)?,
            }
        }
    };
    if projected.len() != hidden_data.len() {
        return Err(LlmError::Forward(
            "GLM MLA prefill output projection length mismatch".into(),
        ));
    }
    let mut attention_output = hidden_data.to_vec();
    add_f32_inplace(&mut attention_output, &projected);

    if profiling {
        stage_ms[5] = mark.elapsed().as_secs_f64() * 1000.0;
        mark = std::time::Instant::now();
    }

    let result = crate::engine::forward::ffn::forward_prefill_ffn(
        metadata,
        GLM_DSA_ARCH,
        Tensor::from_vec(attention_output, &[seq_len, hidden_dim]),
        base,
        layer_idx,
        seq_len,
        metadata.norm_eps,
    );

    if profiling {
        stage_ms[6] = mark.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[glm-prefill-profile] layer={layer_idx} seq={seq_len} dense={:.1} kb={:.1} rope={:.1} attn={:.1} vb={:.1} o={:.1} ffn={:.1}",
            stage_ms[0], stage_ms[1], stage_ms[2], stage_ms[3], stage_ms[4], stage_ms[5], stage_ms[6],
        );
    }

    result
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
    add_f32_inplace(&mut scratch.hidden[..hidden_dim], &down);
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
