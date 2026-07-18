use super::super::*;

pub fn bf16_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .bf16_gemv(weights, rows, cols, input)
}

pub fn f16_gemv(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .f16_gemv(weights, rows, cols, input)
}

pub fn f32_gemm_batch(
    weights: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if cols == 0 {
        return Err("f32 GEMM cols must be non-zero".to_string());
    }
    if weights.len() != rows * cols {
        return Err(format!(
            "f32 GEMM weight len mismatch: got {}, expected {}",
            weights.len(),
            rows * cols
        ));
    }
    if input.len() % cols != 0 {
        return Err(format!(
            "f32 GEMM input len must be multiple of cols: input={}, cols={cols}",
            input.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .f32_gemm_batch(weights, rows, cols, input)
}

pub fn f32_shared_expert(
    gate_weights: &[f32],
    up_weights: &[f32],
    down_weights: &[f32],
    route: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .f32_shared_expert(
            gate_weights,
            up_weights,
            down_weights,
            route,
            n_ff,
            n_embd,
            input,
        )
}

pub fn dense_q4k_gelu_ffn(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if input.len() != n_embd {
        return Err(format!(
            "dense GELU FFN input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "dense GELU FFN dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let gate_row_bytes = (n_embd / 256) * 144;
    let down_row_bytes = match down_quant {
        12 => (n_ff / 256) * 144,
        13 => (n_ff / 256) * 176,
        14 => (n_ff / 256) * 210,
        other => {
            return Err(format!(
                "unsupported dense GELU FFN down quant code {other}"
            ))
        }
    };
    if gate_weights.len() != n_ff * gate_row_bytes {
        return Err(format!(
            "dense GELU FFN gate byte mismatch: got {}, expected {}",
            gate_weights.len(),
            n_ff * gate_row_bytes
        ));
    }
    if up_weights.len() != n_ff * gate_row_bytes {
        return Err(format!(
            "dense GELU FFN up byte mismatch: got {}, expected {}",
            up_weights.len(),
            n_ff * gate_row_bytes
        ));
    }
    if down_weights.len() != n_embd * down_row_bytes {
        return Err(format!(
            "dense GELU FFN down byte mismatch: got {}, expected {}",
            down_weights.len(),
            n_embd * down_row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .qwen35_expert(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            input,
            true,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_gelu_ffn_batch(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    seq_len: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if input.len() != seq_len * n_embd {
        return Err(format!(
            "dense GELU FFN batch input length mismatch: got {}, expected {}",
            input.len(),
            seq_len * n_embd
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_gelu_ffn_batch(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            seq_len,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_silu_ffn_batch(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    n_ff: usize,
    n_embd: usize,
    seq_len: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if input.len() != seq_len * n_embd {
        return Err(format!(
            "dense SiLU FFN batch input length mismatch: got {}, expected {}",
            input.len(),
            seq_len * n_embd
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_silu_ffn_batch(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            n_ff,
            n_embd,
            seq_len,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_gelu_ffn_norm_residual(
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    norm_weight: &[f32],
    post_norm_weight: Option<&[f32]>,
    n_ff: usize,
    n_embd: usize,
    hidden: &[f32],
    norm_eps: f32,
    unit_offset_norm: bool,
) -> Result<Vec<f32>, String> {
    if hidden.len() != n_embd {
        return Err(format!(
            "dense GELU FFN residual hidden length mismatch: got {}, expected {n_embd}",
            hidden.len()
        ));
    }
    if norm_weight.len() != n_embd {
        return Err(format!(
            "dense GELU FFN residual norm length mismatch: got {}, expected {n_embd}",
            norm_weight.len()
        ));
    }
    if let Some(post_norm_weight) = post_norm_weight {
        if post_norm_weight.len() != n_embd {
            return Err(format!(
                "dense GELU FFN residual post norm length mismatch: got {}, expected {n_embd}",
                post_norm_weight.len()
            ));
        }
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 {
        return Err(format!(
            "dense GELU FFN residual dims must be divisible by 256, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let gate_row_bytes = (n_embd / 256) * 144;
    let down_row_bytes = match down_quant {
        12 => (n_ff / 256) * 144,
        13 => (n_ff / 256) * 176,
        14 => (n_ff / 256) * 210,
        other => {
            return Err(format!(
                "unsupported dense GELU FFN residual down quant code {other}"
            ))
        }
    };
    if gate_weights.len() != n_ff * gate_row_bytes {
        return Err(format!(
            "dense GELU FFN residual gate byte mismatch: got {}, expected {}",
            gate_weights.len(),
            n_ff * gate_row_bytes
        ));
    }
    if up_weights.len() != n_ff * gate_row_bytes {
        return Err(format!(
            "dense GELU FFN residual up byte mismatch: got {}, expected {}",
            up_weights.len(),
            n_ff * gate_row_bytes
        ));
    }
    if down_weights.len() != n_embd * down_row_bytes {
        return Err(format!(
            "dense GELU FFN residual down byte mismatch: got {}, expected {}",
            down_weights.len(),
            n_embd * down_row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_gelu_ffn_norm_residual(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            norm_weight,
            post_norm_weight,
            n_ff,
            n_embd,
            hidden,
            norm_eps,
            unit_offset_norm,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_output_gelu_ffn_norm_residual(
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
    ple_input_device_offset: Option<usize>,
    ple_dim: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_ple_norm: bool,
    hidden_carrier_dev: Option<u64>,
    skip_h2d_hidden: bool,
    skip_d2h_hidden: bool,
    layer_output_scale: Option<f32>,
    attn_out_dev_carrier: Option<u64>,
    ffn_uses_gelu: bool,
    dense_chain_graph_allowed: bool,
    layer_segment_graph_context: Option<Cu71LayerSegmentGraphRuntimeContext>,
) -> Result<(), String> {
    if hidden.len() != n_embd {
        return Err(format!(
            "dense attention+FFN hidden length mismatch: got {}, expected {n_embd}",
            hidden.len()
        ));
    }
    if attn_out.len() != o_cols {
        return Err(format!(
            "dense attention+FFN attn_out length mismatch: got {}, expected {o_cols}",
            attn_out.len()
        ));
    }
    if ffn_norm_weight.len() != n_embd {
        return Err(format!(
            "dense attention+FFN norm length mismatch: got {}, expected {n_embd}",
            ffn_norm_weight.len()
        ));
    }
    if let Some(weight) = post_attn_norm_weight {
        if weight.len() != n_embd {
            return Err(format!(
                "dense attention+FFN post attention norm length mismatch: got {}, expected {n_embd}",
                weight.len()
            ));
        }
    }
    if let Some(weight) = post_ffn_norm_weight {
        if weight.len() != n_embd {
            return Err(format!(
                "dense attention+FFN post FFN norm length mismatch: got {}, expected {n_embd}",
                weight.len()
            ));
        }
    }
    if ple_gate_weights.is_some()
        || ple_proj_weights.is_some()
        || ple_post_norm_weight.is_some()
        || ple_input.is_some()
        || ple_input_device_offset.is_some()
    {
        let (
            Some(ple_gate_weights),
            Some(ple_proj_weights),
            Some(ple_post_norm_weight),
            Some(ple_input),
        ) = (
            ple_gate_weights,
            ple_proj_weights,
            ple_post_norm_weight,
            ple_input,
        )
        else {
            return Err(
                "dense attention+FFN PLE parameters must be all present or all absent".to_string(),
            );
        };
        if ple_dim == 0 || ple_dim % 256 != 0 {
            return Err(format!(
                "dense attention+FFN PLE dim must be non-zero and divisible by 256, got {ple_dim}"
            ));
        }
        if ple_input.len() != ple_dim {
            return Err(format!(
                "dense attention+FFN PLE input length mismatch: got {}, expected {ple_dim}",
                ple_input.len()
            ));
        }
        if let Some(offset) = ple_input_device_offset {
            let end = offset.checked_add(ple_dim).ok_or_else(|| {
                format!(
                    "dense attention+FFN PLE device offset overflow: offset={offset} dim={ple_dim}"
                )
            })?;
            if end == 0 {
                return Err("dense attention+FFN PLE device slice must be non-empty".to_string());
            }
        }
        if ple_post_norm_weight.len() != n_embd {
            return Err(format!(
                "dense attention+FFN PLE post norm length mismatch: got {}, expected {n_embd}",
                ple_post_norm_weight.len()
            ));
        }
        let ple_gate_row_bytes = (n_embd / 256) * 144;
        let ple_proj_row_bytes = (ple_dim / 256) * 144;
        let q4k_gate_bytes = ple_dim * ple_gate_row_bytes;
        let q4k_proj_bytes = n_embd * ple_proj_row_bytes;
        let f32_gate_bytes = ple_dim
            .checked_mul(n_embd)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "dense attention+FFN PLE F32 gate byte overflow".to_string())?;
        let f32_proj_bytes = n_embd
            .checked_mul(ple_dim)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "dense attention+FFN PLE F32 proj byte overflow".to_string())?;
        let q4k_ple =
            ple_gate_weights.len() == q4k_gate_bytes && ple_proj_weights.len() == q4k_proj_bytes;
        let f32_ple =
            ple_gate_weights.len() == f32_gate_bytes && ple_proj_weights.len() == f32_proj_bytes;
        if !q4k_ple && !f32_ple {
            return Err(format!(
                "dense attention+FFN PLE byte mismatch: gate got {}, expected q4k {} or f32 {}; proj got {}, expected q4k {} or f32 {}",
                ple_gate_weights.len(),
                q4k_gate_bytes,
                f32_gate_bytes,
                ple_proj_weights.len(),
                q4k_proj_bytes,
                f32_proj_bytes
            ));
        }
    }
    if n_embd % 256 != 0 || n_ff % 256 != 0 || o_cols % 256 != 0 {
        return Err(format!(
            "dense attention+FFN dims must be divisible by 256, got o_cols={o_cols} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let o_row_bytes = (o_cols / 256) * 144;
    let gate_row_bytes = (n_embd / 256) * 144;
    let down_row_bytes = match down_quant {
        12 => (n_ff / 256) * 144,
        13 => (n_ff / 256) * 176,
        14 => (n_ff / 256) * 210,
        other => {
            return Err(format!(
                "unsupported dense attention+FFN down quant code {other}"
            ))
        }
    };
    if o_weights.len() != n_embd * o_row_bytes {
        return Err(format!(
            "dense attention+FFN o-proj byte mismatch: got {}, expected {}",
            o_weights.len(),
            n_embd * o_row_bytes
        ));
    }
    if gate_weights.len() != n_ff * gate_row_bytes {
        return Err(format!(
            "dense attention+FFN gate byte mismatch: got {}, expected {}",
            gate_weights.len(),
            n_ff * gate_row_bytes
        ));
    }
    if up_weights.len() != n_ff * gate_row_bytes {
        return Err(format!(
            "dense attention+FFN up byte mismatch: got {}, expected {}",
            up_weights.len(),
            n_ff * gate_row_bytes
        ));
    }
    if down_weights.len() != n_embd * down_row_bytes {
        return Err(format!(
            "dense attention+FFN down byte mismatch: got {}, expected {}",
            down_weights.len(),
            n_embd * down_row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_attention_output_gelu_ffn_norm_residual(
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
            ple_input_device_offset,
            ple_dim,
            o_cols,
            n_ff,
            n_embd,
            hidden,
            attn_out,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_ple_norm,
            hidden_carrier_dev,
            skip_h2d_hidden,
            skip_d2h_hidden,
            layer_output_scale,
            attn_out_dev_carrier,
            ffn_uses_gelu,
            dense_chain_graph_allowed,
            layer_segment_graph_context,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
    o_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    post_attn_norm_weight: Option<&[f32]>,
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: Option<&[f32]>,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    seq_len: usize,
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_attention_output_gelu_ffn_batch_norm_residual(
            o_weights,
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            post_attn_norm_weight,
            ffn_norm_weight,
            post_ffn_norm_weight,
            o_cols,
            n_ff,
            n_embd,
            seq_len,
            hidden,
            attn_out,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
}

pub fn upload_gemma_ple_base(data: &[f32]) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .upload_gemma_ple_base(data)
}

#[allow(clippy::too_many_arguments)]
pub fn gemma4_ple_q4k_batch_norm_residual(
    gate_weights: &[u8],
    proj_weights: &[u8],
    post_norm_weight: &[f32],
    out_scale: Option<&[f32]>,
    ple_input: &[f32],
    ple_dim: usize,
    n_embd: usize,
    seq_len: usize,
    hidden: &mut [f32],
    norm_eps: f32,
) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .gemma4_ple_q4k_batch_norm_residual(
            gate_weights,
            proj_weights,
            post_norm_weight,
            out_scale,
            ple_input,
            ple_dim,
            n_embd,
            seq_len,
            hidden,
            norm_eps,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_qkv(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    n_embd: usize,
    input: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) -> Result<(), String> {
    if input.len() != n_embd {
        return Err(format!(
            "dense QKV input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 256 != 0 {
        return Err(format!(
            "dense QKV cols must be divisible by 256, got {n_embd}"
        ));
    }
    if q.len() < q_rows || k.len() < kv_rows || v.len() < kv_rows {
        return Err(format!(
            "dense QKV output length mismatch: q={} k={} v={} expected q>={q_rows} k/v>={kv_rows}",
            q.len(),
            k.len(),
            v.len()
        ));
    }
    let row_bytes = (n_embd / 256) * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "dense QKV q byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    if k_weights.len() != kv_rows * row_bytes {
        return Err(format!(
            "dense QKV k byte mismatch: got {}, expected {}",
            k_weights.len(),
            kv_rows * row_bytes
        ));
    }
    if v_weights.len() != kv_rows * row_bytes {
        return Err(format!(
            "dense QKV v byte mismatch: got {}, expected {}",
            v_weights.len(),
            kv_rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_attention_qkv(
            q_weights, k_weights, v_weights, q_rows, kv_rows, n_embd, input, q, k, v,
        )
}

// cu29 Phase 2: Llama / Mistral hd=128 path. Q4K QKV + GPU RoPE + f16 K/V pack.
// Q는 RoPE 적용된 f32 host slice 로 반환, K/V 는 KvCache append_bits_range 에
// 그대로 쓸 수 있는 f16 bits.
#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_qkv_rope_hd128_decode(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    n_embd: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    input: &[f32],
    q_rope: &mut [f32],
    k_bits: &mut [u16],
    v_bits: &mut [u16],
) -> Result<(), String> {
    if input.len() != n_embd {
        return Err(format!(
            "dense QKV+RoPE hd128 input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 256 != 0 {
        return Err(format!(
            "dense QKV+RoPE hd128 cols must be divisible by 256, got {n_embd}"
        ));
    }
    if q_rows != num_heads * 128 || kv_rows != num_kv_heads * 128 {
        return Err(format!(
            "dense QKV+RoPE hd128 shape mismatch: q_rows={q_rows} kv_rows={kv_rows} \
             num_heads={num_heads} num_kv_heads={num_kv_heads}"
        ));
    }
    if q_rope.len() < q_rows || k_bits.len() < kv_rows || v_bits.len() < kv_rows {
        return Err(format!(
            "dense QKV+RoPE hd128 output length mismatch: q={} k={} v={} \
             expected q>={q_rows} k/v>={kv_rows}",
            q_rope.len(),
            k_bits.len(),
            v_bits.len(),
        ));
    }
    let row_bytes = (n_embd / 256) * 144;
    if q_weights.len() != q_rows * row_bytes
        || k_weights.len() != kv_rows * row_bytes
        || v_weights.len() != kv_rows * row_bytes
    {
        return Err(format!(
            "dense QKV+RoPE hd128 weight byte mismatch: q={} k={} v={} \
             expected q={} k/v={}",
            q_weights.len(),
            k_weights.len(),
            v_weights.len(),
            q_rows * row_bytes,
            kv_rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_attention_qkv_rope_hd128_decode(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            n_embd,
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            input,
            q_rope,
            k_bits,
            v_bits,
        )
}

// cu30 Phase 2c: multi-token (prefill) 변형.
#[allow(clippy::too_many_arguments)]
pub fn dense_q4k_attention_qkv_rope_hd128_prefill(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    n_embd: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    pos_start: usize,
    seq_len: usize,
    input: &[f32],
    q_rope: &mut [f32],
    k_bits: &mut [u16],
    v_bits: &mut [u16],
) -> Result<(), String> {
    if seq_len == 0 {
        return Ok(());
    }
    if input.len() != seq_len * n_embd {
        return Err(format!(
            "dense QKV+RoPE hd128 prefill input length mismatch: got {}, expected {}",
            input.len(),
            seq_len * n_embd
        ));
    }
    if n_embd % 256 != 0 {
        return Err(format!(
            "dense QKV+RoPE hd128 prefill cols must be divisible by 256, got {n_embd}"
        ));
    }
    if q_rows != num_heads * 128 || kv_rows != num_kv_heads * 128 {
        return Err(format!(
            "dense QKV+RoPE hd128 prefill shape mismatch: q_rows={q_rows} kv_rows={kv_rows} \
             num_heads={num_heads} num_kv_heads={num_kv_heads}"
        ));
    }
    if q_rope.len() < seq_len * q_rows
        || k_bits.len() < seq_len * kv_rows
        || v_bits.len() < seq_len * kv_rows
    {
        return Err(format!(
            "dense QKV+RoPE hd128 prefill output length mismatch: q={} k={} v={} \
             expected q>={} k/v>={}",
            q_rope.len(),
            k_bits.len(),
            v_bits.len(),
            seq_len * q_rows,
            seq_len * kv_rows
        ));
    }
    let row_bytes = (n_embd / 256) * 144;
    if q_weights.len() != q_rows * row_bytes
        || k_weights.len() != kv_rows * row_bytes
        || v_weights.len() != kv_rows * row_bytes
    {
        return Err(format!(
            "dense QKV+RoPE hd128 prefill weight byte mismatch: q={} k={} v={} \
             expected q={} k/v={}",
            q_weights.len(),
            k_weights.len(),
            v_weights.len(),
            q_rows * row_bytes,
            kv_rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_attention_qkv_rope_hd128_prefill(
            q_weights,
            k_weights,
            v_weights,
            q_rows,
            kv_rows,
            n_embd,
            num_heads,
            num_kv_heads,
            rope_theta,
            pos_start,
            seq_len,
            input,
            q_rope,
            k_bits,
            v_bits,
        )
}

// cu41 Phase 1: decode loop 의 device-resident hidden state carrier API.
// caller (rnb-llm engine 의 decode loop) 가 token decode 시작 시 acquire +
// upload, 마지막 layer 후 download + sync. chain function 의 host↔device
// round-trip 35×100=3500 sync point → 100 sync (per-token only).

pub fn acquire_decode_hidden_carrier(bytes: usize) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_hidden_carrier_ptr(bytes)
}

pub fn acquire_decode_norm_buf_carrier(bytes: usize) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_norm_buf_carrier_ptr(bytes)
}

// cu49 step 38: K/V projection device output buffer acquire.
pub fn acquire_decode_k_carrier(bytes: usize) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_k_carrier_ptr(bytes)
}

pub fn acquire_decode_v_carrier(bytes: usize) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_v_carrier_ptr(bytes)
}

// cu52 step 47: K/V f16 carrier acquire.
pub fn acquire_decode_k_f16_carrier(bytes: usize) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_k_f16_carrier_ptr(bytes)
}

pub fn acquire_decode_v_f16_carrier(bytes: usize) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_v_f16_carrier_ptr(bytes)
}

// cu52 step 48: f32 → f16 pack (K/V projection result → KV cache).
pub fn f32_to_f16_pack_device(src_dev: u64, dst_dev: u64, len: usize) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .launch_f32_to_f16_pack(src_dev, dst_dev, len)
}

// cu47 step 32: attention forward 의 device output buffer (attn_out) acquire.
pub fn acquire_decode_attn_out_carrier(bytes: usize) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_attn_out_carrier_ptr(bytes)
}

pub fn upload_to_decode_hidden_carrier(host: &[f32], dev: u64) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let bytes = std::mem::size_of_val(host);
    unsafe {
        state.api.memcpy_htod_async(
            dev,
            host.as_ptr().cast::<libc::c_void>(),
            bytes,
            state.stream,
        )
    }
}

pub fn download_from_decode_hidden_carrier(dev: u64, host: &mut [f32]) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let bytes = std::mem::size_of_val(host);
    unsafe {
        state.api.memcpy_dtoh_async(
            host.as_mut_ptr().cast::<libc::c_void>(),
            dev,
            bytes,
            state.stream,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_full_layer_device_resident(
    layer_idx: usize,
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    o_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    attn_norm: &[f32],
    ffn_norm: &[f32],
    n_embd: usize,
    n_ff: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    q_rows: usize,
    q_norm_weight: Option<&[f32]>,
    k_norm_weight: Option<&[f32]>,
    out_scale: f32,
    rope_theta: f32,
    rope_pos: usize,
    kv_len: usize,
    norm_eps: f32,
    hidden_dev: u64,
) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_full_layer_device_resident(
            layer_idx,
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            gate_weights,
            up_weights,
            down_weights,
            attn_norm,
            ffn_norm,
            n_embd,
            n_ff,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            q_rows,
            q_norm_weight,
            k_norm_weight,
            out_scale,
            rope_theta,
            rope_pos,
            kv_len,
            norm_eps,
            hidden_dev,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_device_qkv_rope_kv(
    layer_idx: usize,
    norm_carrier_dev: u64,
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_norm_weight: Option<&[f32]>,
    k_norm_weight: Option<&[f32]>,
    q_rows: usize,
    kv_dim: usize,
    n_embd: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    rope_pos: usize,
    kv_len: usize,
    norm_eps: f32,
    q_host_out: &mut [f32],
    k_host_out: &mut [f32],
    v_host_out: &mut [f32],
) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .decode_device_qkv_rope_kv(
            layer_idx,
            norm_carrier_dev,
            q_weights,
            k_weights,
            v_weights,
            q_norm_weight,
            k_norm_weight,
            q_rows,
            kv_dim,
            n_embd,
            num_heads,
            num_kv_heads,
            rope_theta,
            rope_pos,
            kv_len,
            norm_eps,
            q_host_out,
            k_host_out,
            v_host_out,
        )
}

pub fn decode_device_qkv_rope_kv_graph(
    layer_idx: usize,
    norm_carrier_dev: u64,
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_norm_weight: Option<&[f32]>,
    k_norm_weight: Option<&[f32]>,
    q_rows: usize,
    kv_dim: usize,
    n_embd: usize,
    num_heads: usize,
    num_kv_heads: usize,
    rope_theta: f32,
    rope_pos: usize,
    kv_len: usize,
    norm_eps: f32,
    q_host_out: &mut [f32],
    k_host_out: &mut [f32],
    v_host_out: &mut [f32],
) -> Result<u64, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    state.decode_device_qkv_rope_kv_graph(
        layer_idx,
        norm_carrier_dev,
        q_weights,
        k_weights,
        v_weights,
        q_norm_weight,
        k_norm_weight,
        q_rows,
        kv_dim,
        n_embd,
        num_heads,
        num_kv_heads,
        rope_theta,
        rope_pos,
        kv_len,
        norm_eps,
        q_host_out,
        k_host_out,
        v_host_out,
    )
}

pub fn launch_attention_decode_device(
    layer_idx: usize,
    q_dev: u64,
    output_dev: u64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .launch_attention_decode_device(
            layer_idx,
            q_dev,
            output_dev,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        )
}

pub fn launch_attention_decode_device_len_device(
    layer_idx: usize,
    q_dev: u64,
    output_dev: u64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .launch_attention_decode_device_len_device(
            layer_idx,
            q_dev,
            output_dev,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_len,
        )
}

pub fn populate_device_kv_cache_f16(
    layer_idx: usize,
    k_bits: &[u16],
    v_bits: &[u16],
    kv_dim: usize,
    num_tokens: usize,
) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .populate_device_kv_cache_f16(layer_idx, k_bits, v_bits, kv_dim, num_tokens)
}

pub fn cu65_device_qkv_enabled() -> bool {
    crate::tuning::cu65_device_qkv_enabled()
}

pub fn cu68_layer_graph_enabled() -> bool {
    crate::tuning::cu68_layer_graph_enabled()
}

pub fn cu69_dense_chain_graph_enabled() -> bool {
    crate::tuning::cu69_dense_chain_graph_enabled()
}

pub fn cu71_layer_segment_graph_enabled() -> bool {
    crate::tuning::cu71_layer_segment_graph_enabled()
}

pub fn persistent_decode_enabled() -> bool {
    crate::tuning::persistent_decode_enabled()
}

pub fn cu71_layer_segment_graph_trace_enabled() -> bool {
    crate::tuning::cu71_layer_segment_graph_trace_enabled()
}

pub fn cu63_device_decode_enabled() -> bool {
    crate::tuning::cu63_device_decode_enabled()
}

pub fn sync_decode_stream() -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .stream_synchronize()
}

// cu41 Phase 1 step 3: RMS norm cuda — host input + resident weight → device
// output (carrier). caller 가 D2H 또는 chain 의 다음 op 의 device input 사용.
// 즉시 sync 안 함 (caller 결정).
// cu42 step 11: RMS norm 의 device input variant. carrier 의 hidden 을 input 으로
// 사용. chain function 의 carrier output 과 chain 가능 (host scratch.hidden 의 H2D
// 제거).
pub fn rms_norm_f32_dev_input_to_carrier(
    input_dev: u64,
    weight: &[f32],
    output_carrier: u64,
    len: usize,
    eps: f32,
    unit_offset: bool,
) -> Result<(), String> {
    if weight.len() != len {
        return Err(format!(
            "rms_norm_f32_dev_input_to_carrier weight len mismatch: weight={} len={len}",
            weight.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let weight_dev = state.resident_f32_ptr(weight)?;
    state.launch_rms_norm_f32(input_dev, weight_dev, output_carrier, eps, len, unit_offset)
}

pub fn rms_norm_f32_to_carrier(
    input: &[f32],
    weight: &[f32],
    output_carrier: u64,
    eps: f32,
    unit_offset: bool,
) -> Result<(), String> {
    if input.len() != weight.len() {
        return Err(format!(
            "rms_norm_f32_to_carrier len mismatch: input={} weight={}",
            input.len(),
            weight.len()
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let input_bytes = std::mem::size_of_val(input);
    // cu41 step 8: dedicated decode_rms_input buffer — compute_input cache 와 분리.
    // 다음 cuda op (QKV device input 등) 의 alloc 으로 overwrite 되는 race 회피.
    let input_dev = state.decode_rms_input_ptr(input_bytes)?;
    unsafe {
        state.api.memcpy_htod_async(
            input_dev,
            input.as_ptr().cast::<libc::c_void>(),
            input_bytes,
            state.stream,
        )?;
    }
    let weight_dev = state.resident_f32_ptr(weight)?;
    state.launch_rms_norm_f32(
        input_dev,
        weight_dev,
        output_carrier,
        eps,
        input.len(),
        unit_offset,
    )
}

// cu41 Phase 1 step 4: QKV gemv 의 device input variant (host output).
pub fn dense_q4k_attention_qkv_with_device_input(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    n_embd: usize,
    input_dev: u64,
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
) -> Result<(), String> {
    if n_embd % 256 != 0 {
        return Err(format!(
            "dense_q4k_attention_qkv_with_device_input: n_embd must be divisible by 256, got {n_embd}"
        ));
    }
    if q.len() != q_rows {
        return Err(format!(
            "dense_q4k_attention_qkv_with_device_input q len {} != q_rows {q_rows}",
            q.len()
        ));
    }
    if k.len() != kv_rows {
        return Err(format!(
            "dense_q4k_attention_qkv_with_device_input k len {} != kv_rows {kv_rows}",
            k.len()
        ));
    }
    if v.len() != kv_rows {
        return Err(format!(
            "dense_q4k_attention_qkv_with_device_input v len {} != kv_rows {kv_rows}",
            v.len()
        ));
    }
    let row_bytes = (n_embd / 256) * 144;
    if q_weights.len() != q_rows * row_bytes {
        return Err(format!(
            "dense_q4k_attention_qkv_with_device_input q weight byte mismatch: got {}, expected {}",
            q_weights.len(),
            q_rows * row_bytes
        ));
    }
    if k_weights.len() != kv_rows * row_bytes {
        return Err(format!(
            "dense_q4k_attention_qkv_with_device_input k weight byte mismatch: got {}, expected {}",
            k_weights.len(),
            kv_rows * row_bytes
        ));
    }
    if v_weights.len() != kv_rows * row_bytes {
        return Err(format!(
            "dense_q4k_attention_qkv_with_device_input v weight byte mismatch: got {}, expected {}",
            v_weights.len(),
            kv_rows * row_bytes
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dense_q4k_attention_qkv_with_device_input(
            q_weights, k_weights, v_weights, q_rows, kv_rows, n_embd, input_dev, q, k, v,
        )
}

pub fn dispatch_persistent_decode(
    request: &mut rnb_backend_api::PersistentDecodeRequest<'_>,
) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    guard
        .as_mut()
        .expect("cuda compute state initialized")
        .dispatch_persistent_decode(request)
}
