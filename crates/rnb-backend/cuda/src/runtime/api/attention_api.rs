use super::super::*;

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd512(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != seq_len * num_heads * 512 {
        return Err(format!(
            "CUDA attention q len mismatch: got {}, expected {}",
            q.len(),
            seq_len * num_heads * 512
        ));
    }
    if k.len() != kv_len * num_kv_heads * 512 || v.len() != kv_len * num_kv_heads * 512 {
        return Err(format!(
            "CUDA attention k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 512
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
        .attention_prefill_flash_hd512(q, k, v, seq_len, kv_len, num_heads, num_kv_heads, scale)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd512_f16kv(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != seq_len * num_heads * 512 {
        return Err(format!(
            "CUDA attention q len mismatch: got {}, expected {}",
            q.len(),
            seq_len * num_heads * 512
        ));
    }
    if k.len() != kv_len * num_kv_heads * 512 || v.len() != kv_len * num_kv_heads * 512 {
        return Err(format!(
            "CUDA attention f16 k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 512
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
        .attention_prefill_flash_hd512_f16kv(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd512_f16kv_window(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    window: usize,
) -> Result<Vec<f32>, String> {
    if window == 0 {
        return Err("CUDA attention window must be non-zero".to_string());
    }
    if q.len() != seq_len * num_heads * 512 {
        return Err(format!(
            "CUDA attention window q len mismatch: got {}, expected {}",
            q.len(),
            seq_len * num_heads * 512
        ));
    }
    if k.len() != kv_len * num_kv_heads * 512 || v.len() != kv_len * num_kv_heads * 512 {
        return Err(format!(
            "CUDA attention window f16 k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 512
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
        .attention_prefill_flash_hd512_f16kv_window(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd256_f16kv_window(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    window: usize,
) -> Result<Vec<f32>, String> {
    if window == 0 {
        return Err("CUDA hd256 attention window must be non-zero".to_string());
    }
    if q.len() != seq_len * num_heads * 256 {
        return Err(format!(
            "CUDA hd256 attention window q len mismatch: got {}, expected {}",
            q.len(),
            seq_len * num_heads * 256
        ));
    }
    if k.len() != kv_len * num_kv_heads * 256 || v.len() != kv_len * num_kv_heads * 256 {
        return Err(format!(
            "CUDA hd256 attention window f16 k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 256
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
        .attention_prefill_flash_hd256_f16kv_window(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd512_f16kv_dense_chain(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
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
    hidden: &mut [f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<(), String> {
    if q.len() != seq_len * num_heads * 512 {
        return Err(format!(
            "CUDA attention chain q len mismatch: got {}, expected {}",
            q.len(),
            seq_len * num_heads * 512
        ));
    }
    if k.len() != kv_len * num_kv_heads * 512 || v.len() != kv_len * num_kv_heads * 512 {
        return Err(format!(
            "CUDA attention chain f16 k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 512
        ));
    }
    if hidden.len() != seq_len * n_embd {
        return Err(format!(
            "CUDA attention chain hidden len mismatch: got {}, expected {}",
            hidden.len(),
            seq_len * n_embd
        ));
    }
    if o_cols != num_heads * 512 {
        return Err(format!(
            "CUDA attention chain o_cols mismatch: got {o_cols}, expected {}",
            num_heads * 512
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
        .attention_prefill_flash_hd512_f16kv_dense_chain(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
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
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd512_f16kv_window_dense_chain(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    window: usize,
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
    hidden: &mut [f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<(), String> {
    if window == 0 {
        return Err("CUDA attention chain window must be non-zero".to_string());
    }
    if q.len() != seq_len * num_heads * 512 {
        return Err(format!(
            "CUDA attention chain window q len mismatch: got {}, expected {}",
            q.len(),
            seq_len * num_heads * 512
        ));
    }
    if k.len() != kv_len * num_kv_heads * 512 || v.len() != kv_len * num_kv_heads * 512 {
        return Err(format!(
            "CUDA attention chain window f16 k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 512
        ));
    }
    if hidden.len() != seq_len * n_embd {
        return Err(format!(
            "CUDA attention chain window hidden len mismatch: got {}, expected {}",
            hidden.len(),
            seq_len * n_embd
        ));
    }
    if o_cols != num_heads * 512 {
        return Err(format!(
            "CUDA attention chain window o_cols mismatch: got {o_cols}, expected {}",
            num_heads * 512
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
        .attention_prefill_flash_hd512_f16kv_window_dense_chain(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
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
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd256_f16kv_window_dense_chain(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    window: usize,
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
    hidden: &mut [f32],
    norm_eps: f32,
    unit_offset_post_attn_norm: bool,
    unit_offset_ffn_norm: bool,
    unit_offset_post_ffn_norm: bool,
) -> Result<(), String> {
    if window == 0 {
        return Err("CUDA hd256 attention chain window must be non-zero".to_string());
    }
    if q.len() != seq_len * num_heads * 256 {
        return Err(format!(
            "CUDA hd256 attention chain window q len mismatch: got {}, expected {}",
            q.len(),
            seq_len * num_heads * 256
        ));
    }
    if k.len() != kv_len * num_kv_heads * 256 || v.len() != kv_len * num_kv_heads * 256 {
        return Err(format!(
            "CUDA hd256 attention chain window f16 k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 256
        ));
    }
    if hidden.len() != seq_len * n_embd {
        return Err(format!(
            "CUDA hd256 attention chain window hidden len mismatch: got {}, expected {}",
            hidden.len(),
            seq_len * n_embd
        ));
    }
    if o_cols != num_heads * 256 {
        return Err(format!(
            "CUDA hd256 attention chain window o_cols mismatch: got {o_cols}, expected {}",
            num_heads * 256
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
        .attention_prefill_flash_hd256_f16kv_window_dense_chain(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            window,
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
            hidden,
            norm_eps,
            unit_offset_post_attn_norm,
            unit_offset_ffn_norm,
            unit_offset_post_ffn_norm,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd256(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    attention_prefill_flash_f32(
        q,
        k,
        v,
        seq_len,
        kv_len,
        num_heads,
        num_kv_heads,
        256,
        scale,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    softcap: Option<f32>,
) -> Result<Vec<f32>, String> {
    if seq_len == 0 || kv_len < seq_len || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
        return Err(format!(
            "CUDA attention invalid shape: seq_len={seq_len} kv_len={kv_len} heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim}"
        ));
    }
    if num_heads % num_kv_heads != 0 {
        return Err(format!(
            "CUDA attention GQA mismatch: heads={num_heads} is not divisible by kv_heads={num_kv_heads}"
        ));
    }
    if [seq_len, kv_len, num_heads, num_kv_heads, head_dim]
        .into_iter()
        .any(|value| u32::try_from(value).is_err())
    {
        return Err("CUDA attention shape exceeds u32 kernel limits".to_string());
    }
    if sliding_window == Some(0) {
        return Err("CUDA attention sliding window must be positive".to_string());
    }
    if sliding_window.is_some_and(|window| u32::try_from(window).is_err()) {
        return Err("CUDA attention sliding window exceeds u32 kernel limits".to_string());
    }
    if softcap.is_some_and(|cap| !cap.is_finite() || cap <= 0.0) {
        return Err("CUDA attention softcap must be finite and positive".to_string());
    }
    let q_expected = seq_len
        .checked_mul(num_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "CUDA attention q element count overflow".to_string())?;
    let kv_expected = kv_len
        .checked_mul(num_kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| "CUDA attention k/v element count overflow".to_string())?;
    if q.len() != q_expected {
        return Err(format!(
            "CUDA attention q len mismatch: got {}, expected {q_expected}",
            q.len()
        ));
    }
    if k.len() != kv_expected || v.len() != kv_expected {
        return Err(format!(
            "CUDA attention k/v len mismatch: k={} v={} expected {kv_expected}",
            k.len(),
            v.len()
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
        .attention_prefill_flash_hd256(
            q,
            k,
            v,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            sliding_window,
            softcap,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd128(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != seq_len * num_heads * 128 {
        return Err(format!(
            "CUDA attention q len mismatch: got {}, expected {}",
            q.len(),
            seq_len * num_heads * 128
        ));
    }
    if k.len() != kv_len * num_kv_heads * 128 || v.len() != kv_len * num_kv_heads * 128 {
        return Err(format!(
            "CUDA attention k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 128
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
        .attention_prefill_flash_hd128(q, k, v, seq_len, kv_len, num_heads, num_kv_heads, scale)
}

// cu47 step 32: attention_decode_cached 의 device output variant.
// caller (decode_attention_compute) 가 attn_out carrier ptr 제공.
// internal attention compute 의 D2H + sync 안 함 → chain function 의 attn_out
// H2D round-trip 제거. host return 없음 (Result<()>).
pub fn attention_decode_cached_to_device(
    layer_index: usize,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    output_dev_target: u64,
    // cu51 step 41: K/V device source (KV cache device-resident). Some 시 host
    // k/v slice 무시 + device → device copy. 마지막 1 token row 만.
    last_token_k_dev: Option<u64>,
    last_token_v_dev: Option<u64>,
    q_dev_override: Option<u64>,
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
        .attention_decode_cached_to_device(
            layer_index,
            q,
            k,
            v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            output_dev_target,
            last_token_k_dev,
            last_token_v_dev,
            q_dev_override,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_cached_to_device_len_device(
    layer_index: usize,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    output_dev_target: u64,
    last_token_k_dev: Option<u64>,
    last_token_v_dev: Option<u64>,
    q_dev_override: Option<u64>,
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
        .attention_decode_cached_to_device_len_device(
            layer_index,
            q,
            k,
            v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            output_dev_target,
            last_token_k_dev,
            last_token_v_dev,
            q_dev_override,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_cached_to_device_len_device_graph(
    layer_index: usize,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    output_dev_target: u64,
    last_token_k_dev: Option<u64>,
    last_token_v_dev: Option<u64>,
    q_dev_override: Option<u64>,
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
        .attention_decode_cached_to_device_len_device_graph(
            layer_index,
            q,
            k,
            v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            output_dev_target,
            last_token_k_dev,
            last_token_v_dev,
            q_dev_override,
        )
}

pub fn attention_decode_hd256(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != num_heads * 256 {
        return Err(format!(
            "CUDA decode attention q len mismatch: got {}, expected {}",
            q.len(),
            num_heads * 256
        ));
    }
    if k.len() != kv_len * num_kv_heads * 256 || v.len() != kv_len * num_kv_heads * 256 {
        return Err(format!(
            "CUDA decode attention k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 256
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
        .attention_decode_hd256(q, k, v, kv_len, num_heads, num_kv_heads, scale)
}

pub fn attention_decode_hd128(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != num_heads * 128 {
        return Err(format!(
            "CUDA decode attention q len mismatch: got {}, expected {}",
            q.len(),
            num_heads * 128
        ));
    }
    if k.len() != kv_len * num_kv_heads * 128 || v.len() != kv_len * num_kv_heads * 128 {
        return Err(format!(
            "CUDA decode attention k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 128
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
        .attention_decode_hd128(q, k, v, kv_len, num_heads, num_kv_heads, scale)
}

pub fn attention_decode_hd512(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != num_heads * 512 {
        return Err(format!(
            "CUDA decode attention q len mismatch: got {}, expected {}",
            q.len(),
            num_heads * 512
        ));
    }
    if k.len() != kv_len * num_kv_heads * 512 || v.len() != kv_len * num_kv_heads * 512 {
        return Err(format!(
            "CUDA decode attention k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 512
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
        .attention_decode_hd512(q, k, v, kv_len, num_heads, num_kv_heads, scale)
}

pub fn attention_decode_hd512_len_device(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != num_heads * 512 {
        return Err(format!(
            "CUDA decode attention q len mismatch: got {}, expected {}",
            q.len(),
            num_heads * 512
        ));
    }
    if k.len() != kv_len * num_kv_heads * 512 || v.len() != kv_len * num_kv_heads * 512 {
        return Err(format!(
            "CUDA decode attention k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * 512
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
        .attention_decode_hd512_len_device(q, k, v, kv_len, num_heads, num_kv_heads, scale)
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_cached(
    layer_index: usize,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != num_heads * head_dim {
        return Err(format!(
            "CUDA cached decode attention q len mismatch: got {}, expected {}",
            q.len(),
            num_heads * head_dim
        ));
    }
    if k.len() != kv_len * num_kv_heads * head_dim || v.len() != kv_len * num_kv_heads * head_dim {
        return Err(format!(
            "CUDA cached decode attention k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * head_dim
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
        .attention_decode_cached(
            layer_index,
            q,
            k,
            v,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_cached_window(
    layer_index: usize,
    q: &[f32],
    k: &[u16],
    v: &[u16],
    kv_len: usize,
    window_start: usize,
    window_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if q.len() != num_heads * head_dim {
        return Err(format!(
            "CUDA cached window decode attention q len mismatch: got {}, expected {}",
            q.len(),
            num_heads * head_dim
        ));
    }
    if k.len() != kv_len * num_kv_heads * head_dim || v.len() != kv_len * num_kv_heads * head_dim {
        return Err(format!(
            "CUDA cached window decode attention k/v len mismatch: k={} v={} expected {}",
            k.len(),
            v.len(),
            kv_len * num_kv_heads * head_dim
        ));
    }
    let window_end = window_start
        .checked_add(window_len)
        .ok_or_else(|| "CUDA cached window decode attention window overflow".to_string())?;
    if window_len == 0 || window_start > kv_len || window_end > kv_len {
        return Err(format!(
            "CUDA cached window decode attention invalid window: kv_len={kv_len} start={window_start} len={window_len}"
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
        .attention_decode_cached_window(
            layer_index,
            q,
            k,
            v,
            kv_len,
            window_start,
            window_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn glm_mla_prefill_attention_f16(
    q_absorbed: &[f32],
    q_pe: &[f32],
    cache: &[u16],
    pos_start: usize,
    seq_len: usize,
    num_heads: usize,
    kv_len: usize,
    kv_rank: usize,
    rope_dim: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if seq_len == 0 || num_heads == 0 || kv_len == 0 || kv_rank == 0 || rope_dim == 0 {
        return Err(format!(
            "CUDA GLM MLA prefill dimensions must be non-zero: seq={seq_len} heads={num_heads} kv_len={kv_len} kv_rank={kv_rank} rope_dim={rope_dim}"
        ));
    }
    if pos_start
        .checked_add(seq_len)
        .ok_or_else(|| "CUDA GLM MLA position range overflow".to_string())?
        != kv_len
    {
        return Err(format!(
            "CUDA GLM MLA KV length mismatch: pos_start={pos_start} seq_len={seq_len} kv_len={kv_len}"
        ));
    }
    for (label, value) in [
        ("pos_start", pos_start),
        ("seq_len", seq_len),
        ("num_heads", num_heads),
        ("kv_len", kv_len),
        ("kv_rank", kv_rank),
        ("rope_dim", rope_dim),
    ] {
        if u32::try_from(value).is_err() {
            return Err(format!("CUDA GLM MLA {label} exceeds u32: {value}"));
        }
    }
    let query_count = seq_len
        .checked_mul(num_heads)
        .ok_or_else(|| "CUDA GLM MLA query count overflow".to_string())?;
    let expected_q_absorbed = query_count
        .checked_mul(kv_rank)
        .ok_or_else(|| "CUDA GLM MLA absorbed query length overflow".to_string())?;
    let expected_q_pe = query_count
        .checked_mul(rope_dim)
        .ok_or_else(|| "CUDA GLM MLA RoPE query length overflow".to_string())?;
    let kv_width = kv_rank
        .checked_add(rope_dim)
        .ok_or_else(|| "CUDA GLM MLA KV width overflow".to_string())?;
    let expected_cache = kv_len
        .checked_mul(kv_width)
        .ok_or_else(|| "CUDA GLM MLA cache length overflow".to_string())?;
    if q_absorbed.len() != expected_q_absorbed
        || q_pe.len() != expected_q_pe
        || cache.len() != expected_cache
    {
        return Err(format!(
            "CUDA GLM MLA prefill input mismatch: q_absorbed={} expected_q_absorbed={} q_pe={} expected_q_pe={} cache={} expected_cache={}",
            q_absorbed.len(),
            expected_q_absorbed,
            q_pe.len(),
            expected_q_pe,
            cache.len(),
            expected_cache,
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
        .glm_mla_prefill_attention_f16(
            q_absorbed, q_pe, cache, pos_start, seq_len, num_heads, kv_len, kv_rank, rope_dim,
            scale,
        )
}

pub fn attention_decode_kvarn(
    request: rnb_backend_api::KvarnDecodeRequest<'_>,
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
        .attention_decode_kvarn(request)
}

pub fn attention_decode_kvarn_to_device(
    request: rnb_backend_api::KvarnDecodeRequest<'_>,
    output_dev_target: u64,
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
        .attention_decode_kvarn_to_device(request, output_dev_target)
}
