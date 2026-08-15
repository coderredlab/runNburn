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
    attention_prefill_flash_f32_with_mask(
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
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_f32_non_causal(
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
) -> Result<Vec<f32>, String> {
    attention_prefill_flash_f32_with_mask(
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
        None,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn attention_prefill_flash_f32_with_mask(
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
    causal: bool,
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
    if !causal && softcap.is_some() {
        return Err("CUDA non-causal attention does not support softcap".to_string());
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
            causal,
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

const MUSE_HD128_MAX_GRID_Y: usize = 65_535;

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd128_muse_dense_chain(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attention_gate: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    o_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    post_attn_norm_weight: &[f32],
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: &[f32],
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    norm_eps: f32,
    post_norm_eps: f32,
) -> Result<(), String> {
    if seq_len == 0
        || kv_len < seq_len
        || num_heads == 0
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
        || n_ff == 0
        || n_embd == 0
    {
        return Err(format!(
            "CUDA Muse attention chain invalid attention geometry: seq_len={seq_len} kv_len={kv_len} num_heads={num_heads} num_kv_heads={num_kv_heads}"
        ));
    }
    if seq_len > u32::MAX as usize
        || kv_len > u32::MAX as usize
        || num_heads > MUSE_HD128_MAX_GRID_Y
        || num_kv_heads > u32::MAX as usize
        || sliding_window.is_some_and(|window| window == 0 || window > u32::MAX as usize)
    {
        return Err(format!(
            "CUDA Muse attention chain kernel argument out of range: seq_len={seq_len} kv_len={kv_len} num_heads={num_heads} num_kv_heads={num_kv_heads} sliding_window={sliding_window:?}"
        ));
    }
    let q_rows = num_heads
        .checked_mul(128)
        .ok_or_else(|| "CUDA Muse attention chain Q row overflow".to_string())?;
    let kv_rows = num_kv_heads
        .checked_mul(128)
        .ok_or_else(|| "CUDA Muse attention chain KV row overflow".to_string())?;
    if !muse_hd128_attention_extents_fit_u32(seq_len, kv_len, q_rows, kv_rows, o_cols, n_ff, n_embd)
    {
        return Err(format!(
            "CUDA Muse attention chain kernel extent out of u32 range: seq_len={seq_len} kv_len={kv_len} q_rows={q_rows} kv_rows={kv_rows} o_cols={o_cols} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let expected_q = seq_len
        .checked_mul(q_rows)
        .ok_or_else(|| "CUDA Muse attention chain Q length overflow".to_string())?;
    let expected_kv = kv_len
        .checked_mul(kv_rows)
        .ok_or_else(|| "CUDA Muse attention chain KV length overflow".to_string())?;
    let o_blocks = o_cols
        .checked_div(256)
        .filter(|_| o_cols.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse attention chain o_cols must be divisible by 256: {o_cols}")
        })?;
    let hidden_blocks = n_embd
        .checked_div(256)
        .filter(|_| n_embd.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse attention chain n_embd must be divisible by 256: {n_embd}")
        })?;
    let down_blocks = n_ff
        .checked_div(256)
        .filter(|_| n_ff.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse attention chain n_ff must be divisible by 256: {n_ff}")
        })?;
    let down_row_bytes = match down_quant {
        12 => down_blocks.checked_mul(144),
        13 => down_blocks.checked_mul(176),
        14 => down_blocks.checked_mul(210),
        other => {
            return Err(format!(
                "CUDA Muse attention chain unsupported down quant {other}"
            ))
        }
    }
    .ok_or_else(|| "CUDA Muse attention chain down row byte overflow".to_string())?;
    let expected_o_bytes = n_embd
        .checked_mul(o_blocks)
        .and_then(|len| len.checked_mul(144))
        .ok_or_else(|| "CUDA Muse attention chain O weight byte overflow".to_string())?;
    let expected_gate_up_bytes = n_ff
        .checked_mul(hidden_blocks)
        .and_then(|len| len.checked_mul(144))
        .ok_or_else(|| "CUDA Muse attention chain gate/up weight byte overflow".to_string())?;
    let expected_down_bytes = n_embd
        .checked_mul(down_row_bytes)
        .ok_or_else(|| "CUDA Muse attention chain down weight byte overflow".to_string())?;
    if !muse_hd128_weight_extents_fit_u32(&[
        expected_o_bytes,
        expected_gate_up_bytes,
        expected_down_bytes,
    ]) {
        return Err(format!(
            "CUDA Muse attention chain weight extent out of u32 range: o={expected_o_bytes} gate_up={expected_gate_up_bytes} down={expected_down_bytes}"
        ));
    }
    let expected_hidden = seq_len
        .checked_mul(n_embd)
        .ok_or_else(|| "CUDA Muse attention chain hidden length overflow".to_string())?;
    if q.len() != expected_q
        || k.len() != expected_kv
        || v.len() != expected_kv
        || attention_gate.len() != expected_q
        || hidden.len() != expected_hidden
        || o_cols != q_rows
        || post_attn_norm_weight.len() != n_embd
        || ffn_norm_weight.len() != n_embd
        || post_ffn_norm_weight.len() != n_embd
        || o_weights.len() != expected_o_bytes
        || gate_weights.len() != expected_gate_up_bytes
        || up_weights.len() != expected_gate_up_bytes
        || down_weights.len() != expected_down_bytes
    {
        return Err(format!(
            "CUDA Muse attention chain shape mismatch: q={} expected_q={expected_q} k={} v={} expected_kv={expected_kv} gate={} hidden={} expected_hidden={} o_cols={o_cols} expected_o_cols={} post_attn_norm={} ffn_norm={} post_ffn_norm={} n_embd={n_embd}",
            q.len(),
            k.len(),
            v.len(),
            attention_gate.len(),
            hidden.len(),
            expected_hidden,
            q_rows,
            post_attn_norm_weight.len(),
            ffn_norm_weight.len(),
            post_ffn_norm_weight.len(),
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
        .attention_prefill_flash_hd128_muse_dense_chain(
            q,
            k,
            v,
            attention_gate,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            sliding_window,
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
            post_norm_eps,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_flash_hd128_f16kv_muse_dense_chain(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    attention_gate: &[f32],
    seq_len: usize,
    kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    sliding_window: Option<usize>,
    o_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    post_attn_norm_weight: &[f32],
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: &[f32],
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    norm_eps: f32,
    post_norm_eps: f32,
) -> Result<(), String> {
    if seq_len == 0
        || kv_len < seq_len
        || num_heads == 0
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
        || n_ff == 0
        || n_embd == 0
    {
        return Err(format!(
            "CUDA Muse F16 KV attention chain invalid attention geometry: seq_len={seq_len} kv_len={kv_len} num_heads={num_heads} num_kv_heads={num_kv_heads}"
        ));
    }
    if seq_len > u32::MAX as usize
        || kv_len > u32::MAX as usize
        || num_heads > MUSE_HD128_MAX_GRID_Y
        || num_kv_heads > u32::MAX as usize
        || sliding_window.is_some_and(|window| window == 0 || window > u32::MAX as usize)
    {
        return Err(format!(
            "CUDA Muse F16 KV attention chain kernel argument out of range: seq_len={seq_len} kv_len={kv_len} num_heads={num_heads} num_kv_heads={num_kv_heads} sliding_window={sliding_window:?}"
        ));
    }
    let q_rows = num_heads
        .checked_mul(128)
        .ok_or_else(|| "CUDA Muse F16 KV attention chain Q row overflow".to_string())?;
    let kv_rows = num_kv_heads
        .checked_mul(128)
        .ok_or_else(|| "CUDA Muse F16 KV attention chain KV row overflow".to_string())?;
    if !muse_hd128_attention_extents_fit_u32(seq_len, kv_len, q_rows, kv_rows, o_cols, n_ff, n_embd)
    {
        return Err(format!(
            "CUDA Muse F16 KV attention chain kernel extent out of u32 range: seq_len={seq_len} kv_len={kv_len} q_rows={q_rows} kv_rows={kv_rows} o_cols={o_cols} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let expected_q = seq_len
        .checked_mul(q_rows)
        .ok_or_else(|| "CUDA Muse F16 KV attention chain Q length overflow".to_string())?;
    let expected_kv = kv_len
        .checked_mul(kv_rows)
        .ok_or_else(|| "CUDA Muse F16 KV attention chain KV length overflow".to_string())?;
    let o_blocks = o_cols
        .checked_div(256)
        .filter(|_| o_cols.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse F16 KV attention chain o_cols must be divisible by 256: {o_cols}")
        })?;
    let hidden_blocks = n_embd
        .checked_div(256)
        .filter(|_| n_embd.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse F16 KV attention chain n_embd must be divisible by 256: {n_embd}")
        })?;
    let down_blocks = n_ff
        .checked_div(256)
        .filter(|_| n_ff.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse F16 KV attention chain n_ff must be divisible by 256: {n_ff}")
        })?;
    let down_row_bytes = match down_quant {
        12 => down_blocks.checked_mul(144),
        13 => down_blocks.checked_mul(176),
        14 => down_blocks.checked_mul(210),
        other => {
            return Err(format!(
                "CUDA Muse F16 KV attention chain unsupported down quant {other}"
            ))
        }
    }
    .ok_or_else(|| "CUDA Muse F16 KV attention chain down row byte overflow".to_string())?;
    let expected_o_bytes = n_embd
        .checked_mul(o_blocks)
        .and_then(|len| len.checked_mul(144))
        .ok_or_else(|| "CUDA Muse F16 KV attention chain O weight byte overflow".to_string())?;
    let expected_gate_up_bytes = n_ff
        .checked_mul(hidden_blocks)
        .and_then(|len| len.checked_mul(144))
        .ok_or_else(|| {
            "CUDA Muse F16 KV attention chain gate/up weight byte overflow".to_string()
        })?;
    let expected_down_bytes = n_embd
        .checked_mul(down_row_bytes)
        .ok_or_else(|| "CUDA Muse F16 KV attention chain down weight byte overflow".to_string())?;
    if !muse_hd128_weight_extents_fit_u32(&[
        expected_o_bytes,
        expected_gate_up_bytes,
        expected_down_bytes,
    ]) {
        return Err(format!(
            "CUDA Muse F16 KV attention chain weight extent out of u32 range: o={expected_o_bytes} gate_up={expected_gate_up_bytes} down={expected_down_bytes}"
        ));
    }
    let expected_hidden = seq_len
        .checked_mul(n_embd)
        .ok_or_else(|| "CUDA Muse F16 KV attention chain hidden length overflow".to_string())?;
    if q.len() != expected_q
        || k.len() != expected_kv
        || v.len() != expected_kv
        || attention_gate.len() != expected_q
        || hidden.len() != expected_hidden
        || o_cols != q_rows
        || post_attn_norm_weight.len() != n_embd
        || ffn_norm_weight.len() != n_embd
        || post_ffn_norm_weight.len() != n_embd
        || o_weights.len() != expected_o_bytes
        || gate_weights.len() != expected_gate_up_bytes
        || up_weights.len() != expected_gate_up_bytes
        || down_weights.len() != expected_down_bytes
    {
        return Err(format!(
            "CUDA Muse F16 KV attention chain shape mismatch: q={} expected_q={expected_q} k={} v={} expected_kv={expected_kv} gate={} hidden={} expected_hidden={} o_cols={o_cols} expected_o_cols={} post_attn_norm={} ffn_norm={} post_ffn_norm={} n_embd={n_embd}",
            q.len(),
            k.len(),
            v.len(),
            attention_gate.len(),
            hidden.len(),
            expected_hidden,
            q_rows,
            post_attn_norm_weight.len(),
            ffn_norm_weight.len(),
            post_ffn_norm_weight.len(),
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
        .attention_prefill_flash_hd128_f16kv_muse_dense_chain(
            q,
            k,
            v,
            attention_gate,
            seq_len,
            kv_len,
            num_heads,
            num_kv_heads,
            scale,
            sliding_window,
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
            post_norm_eps,
        )
}

fn muse_hd128_attention_extents_fit_u32(
    seq_len: usize,
    kv_len: usize,
    q_rows: usize,
    kv_rows: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
) -> bool {
    let max = u32::MAX as usize;
    seq_len <= MUSE_HD128_MAX_GRID_Y
        && [seq_len, kv_len, q_rows, kv_rows, o_cols, n_ff, n_embd]
            .into_iter()
            .all(|value| value <= max)
        && [
            (seq_len, q_rows),
            (kv_len, kv_rows),
            (seq_len, o_cols),
            (seq_len, n_ff),
            (seq_len, n_embd),
        ]
        .into_iter()
        .all(|(rows, width)| {
            rows.checked_mul(width)
                .is_some_and(|elements| elements <= max)
        })
}

fn muse_hd128_weight_extents_fit_u32(extents: &[usize]) -> bool {
    extents.iter().all(|&bytes| bytes <= u32::MAX as usize)
}

fn muse_hd128_rope_positions_fit_u32(apply_rope: bool, pos_start: usize, seq_len: usize) -> bool {
    !apply_rope
        || (seq_len > 0
            && pos_start
                .checked_add(seq_len - 1)
                .is_some_and(|last_pos| last_pos <= u32::MAX as usize))
}

fn muse_hd128_kernel_extents_fit_u32(
    seq_len: usize,
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
) -> bool {
    muse_hd128_attention_extents_fit_u32(seq_len, seq_len, q_rows, kv_rows, o_cols, n_ff, n_embd)
        && cols <= u32::MAX as usize
        && seq_len
            .checked_mul(cols)
            .is_some_and(|elements| elements <= u32::MAX as usize)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn q4k_muse_prefill_hd128_dense_chain(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    v_quant: u32,
    attention_gate_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    hidden_input: &[f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    apply_rope: bool,
    sliding_window: Option<usize>,
    o_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    post_attn_norm_weight: &[f32],
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: &[f32],
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    hidden: &mut [f32],
    norm_eps: f32,
    post_norm_eps: f32,
) -> Result<Option<(Vec<u16>, Vec<u16>)>, String> {
    if sliding_window.is_some_and(|window| window == 0 || window > u32::MAX as usize) {
        return Err(format!(
            "CUDA Muse QKV dense chain invalid sliding window: {sliding_window:?}"
        ));
    }
    let expected_q_rows = num_heads.checked_mul(128);
    let expected_kv_rows = num_kv_heads.checked_mul(128);
    if num_heads == 0
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
        || num_heads > MUSE_HD128_MAX_GRID_Y
        || num_kv_heads > u32::MAX as usize
        || expected_q_rows != Some(q_rows)
        || expected_kv_rows != Some(kv_rows)
    {
        return Err(format!(
            "CUDA Muse QKV dense chain invalid head geometry: q_rows={q_rows} kv_rows={kv_rows} num_heads={num_heads} num_kv_heads={num_kv_heads}"
        ));
    }
    if cols == 0 || !cols.is_multiple_of(256) {
        return Err(format!(
            "CUDA Muse QKV dense chain cols must be non-zero and divisible by 256: {cols}"
        ));
    }
    if !hidden_input.len().is_multiple_of(cols) {
        return Err(format!(
            "CUDA Muse QKV dense chain hidden input length {} is not divisible by cols {cols}",
            hidden_input.len()
        ));
    }
    let seq_len = hidden_input.len() / cols;
    if seq_len == 0 || n_ff == 0 || n_embd == 0 {
        return Err(format!(
            "CUDA Muse QKV dense chain dimensions must be non-zero: seq_len={seq_len} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    if !muse_hd128_rope_positions_fit_u32(apply_rope, pos_start, seq_len) {
        return Err(format!(
            "CUDA Muse QKV dense chain RoPE position out of u32 range: pos_start={pos_start} seq_len={seq_len}"
        ));
    }
    if !muse_hd128_kernel_extents_fit_u32(seq_len, q_rows, kv_rows, cols, o_cols, n_ff, n_embd) {
        return Err(format!(
            "CUDA Muse QKV dense chain kernel extent out of u32 range: seq_len={seq_len} q_rows={q_rows} kv_rows={kv_rows} cols={cols} o_cols={o_cols} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let blocks_per_row = cols / 256;
    let row_bytes = blocks_per_row
        .checked_mul(144)
        .ok_or_else(|| "CUDA Muse QKV row byte overflow".to_string())?;
    let expected_q = q_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| "CUDA Muse Q weight byte overflow".to_string())?;
    let expected_k = kv_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| "CUDA Muse K weight byte overflow".to_string())?;
    let v_row_bytes = match v_quant {
        12 => blocks_per_row.checked_mul(144),
        14 => blocks_per_row.checked_mul(210),
        other => return Err(format!("CUDA Muse QKV unsupported V quant {other}")),
    }
    .ok_or_else(|| "CUDA Muse V row byte overflow".to_string())?;
    let expected_v = kv_rows
        .checked_mul(v_row_bytes)
        .ok_or_else(|| "CUDA Muse V weight byte overflow".to_string())?;
    let expected_gate = num_heads
        .checked_mul(128)
        .and_then(|rows| rows.checked_mul(row_bytes))
        .ok_or_else(|| "CUDA Muse attention gate weight byte overflow".to_string())?;
    let o_blocks = o_cols
        .checked_div(256)
        .filter(|_| o_cols.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse QKV dense chain o_cols must be divisible by 256: {o_cols}")
        })?;
    let hidden_blocks = n_embd
        .checked_div(256)
        .filter(|_| n_embd.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse QKV dense chain n_embd must be divisible by 256: {n_embd}")
        })?;
    let down_blocks = n_ff
        .checked_div(256)
        .filter(|_| n_ff.is_multiple_of(256))
        .ok_or_else(|| {
            format!("CUDA Muse QKV dense chain n_ff must be divisible by 256: {n_ff}")
        })?;
    let down_row_bytes = match down_quant {
        12 => down_blocks.checked_mul(144),
        13 => down_blocks.checked_mul(176),
        14 => down_blocks.checked_mul(210),
        other => {
            return Err(format!(
                "CUDA Muse QKV dense chain unsupported down quant {other}"
            ))
        }
    }
    .ok_or_else(|| "CUDA Muse QKV dense chain down row byte overflow".to_string())?;
    let expected_o_bytes = n_embd
        .checked_mul(o_blocks)
        .and_then(|len| len.checked_mul(144))
        .ok_or_else(|| "CUDA Muse QKV dense chain O weight byte overflow".to_string())?;
    let expected_gate_up_bytes = n_ff
        .checked_mul(hidden_blocks)
        .and_then(|len| len.checked_mul(144))
        .ok_or_else(|| "CUDA Muse QKV dense chain gate/up weight byte overflow".to_string())?;
    let expected_down_bytes = n_embd
        .checked_mul(down_row_bytes)
        .ok_or_else(|| "CUDA Muse QKV dense chain down weight byte overflow".to_string())?;
    if !muse_hd128_weight_extents_fit_u32(&[
        expected_q,
        expected_k,
        expected_v,
        expected_gate,
        expected_o_bytes,
        expected_gate_up_bytes,
        expected_down_bytes,
    ]) {
        return Err(format!(
            "CUDA Muse QKV dense chain weight extent out of u32 range: q={expected_q} k={expected_k} v={expected_v} attention_gate={expected_gate} o={expected_o_bytes} gate_up={expected_gate_up_bytes} down={expected_down_bytes}"
        ));
    }
    let expected_hidden = seq_len
        .checked_mul(n_embd)
        .ok_or_else(|| "CUDA Muse QKV dense chain hidden length overflow".to_string())?;
    if q_weights.len() != expected_q
        || k_weights.len() != expected_k
        || v_weights.len() != expected_v
        || attention_gate_weights.len() != expected_gate
        || hidden_input.len() != expected_hidden
        || hidden.len() != expected_hidden
        || attn_norm_weight.len() != n_embd
        || q_norm.len() != 128
        || k_norm.len() != 128
        || o_cols != num_heads.saturating_mul(128)
        || post_attn_norm_weight.len() != n_embd
        || ffn_norm_weight.len() != n_embd
        || post_ffn_norm_weight.len() != n_embd
        || o_weights.len() != expected_o_bytes
        || gate_weights.len() != expected_gate_up_bytes
        || up_weights.len() != expected_gate_up_bytes
        || down_weights.len() != expected_down_bytes
    {
        return Err(format!(
            "CUDA Muse QKV dense chain shape mismatch: q={} expected_q={expected_q} k={} expected_k={expected_k} v={} expected_v={expected_v} attn_gate={} expected_attn_gate={expected_gate} hidden_input={} hidden={} expected_hidden={expected_hidden} attn_norm={} q_norm={} k_norm={} o_cols={o_cols} expected_o_cols={} post_attn_norm={} ffn_norm={} post_ffn_norm={} n_embd={n_embd} o_bytes={} expected_o_bytes={expected_o_bytes} gate_bytes={} up_bytes={} expected_gate_up_bytes={expected_gate_up_bytes} down_bytes={} expected_down_bytes={expected_down_bytes}",
            q_weights.len(),
            k_weights.len(),
            v_weights.len(),
            attention_gate_weights.len(),
            hidden_input.len(),
            hidden.len(),
            attn_norm_weight.len(),
            q_norm.len(),
            k_norm.len(),
            num_heads.saturating_mul(128),
            post_attn_norm_weight.len(),
            ffn_norm_weight.len(),
            post_ffn_norm_weight.len(),
            o_weights.len(),
            gate_weights.len(),
            up_weights.len(),
            down_weights.len(),
        ));
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let output = guard
        .as_mut()
        .expect("cuda compute state initialized")
        .q4k_muse_prefill_hd128_dense_chain(
            q_weights,
            k_weights,
            v_weights,
            v_quant,
            attention_gate_weights,
            q_rows,
            kv_rows,
            blocks_per_row,
            hidden_input.len() / cols,
            Some(hidden_input),
            attn_norm_weight,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            apply_rope,
            sliding_window,
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
            None,
            None,
            norm_eps,
            post_norm_eps,
            None,
        )?;
    Ok(output.map(|(k_bits, v_bits, _)| (k_bits, v_bits)))
}

#[derive(Debug)]
pub struct MuseQ4kPrefillDeviceOutput {
    pub k_bits: Vec<u16>,
    pub v_bits: Vec<u16>,
    pub output_id: rnb_backend_api::DeviceTensorId,
    pub output_desc: rnb_backend_api::DeviceTensorDesc,
}

#[allow(clippy::too_many_arguments)]
pub fn q4k_muse_prefill_hd128_dense_chain_device_input(
    input_id: rnb_backend_api::DeviceTensorId,
    input_desc: rnb_backend_api::DeviceTensorDesc,
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    v_quant: u32,
    attention_gate_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    apply_rope: bool,
    sliding_window: Option<usize>,
    o_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    down_quant: u32,
    post_attn_norm_weight: &[f32],
    ffn_norm_weight: &[f32],
    post_ffn_norm_weight: &[f32],
    o_cols: usize,
    n_ff: usize,
    n_embd: usize,
    norm_eps: f32,
    post_norm_eps: f32,
) -> Result<Option<MuseQ4kPrefillDeviceOutput>, String> {
    let seq_len = input_desc.rows();
    if input_desc.cols() != cols
        || input_desc.dtype() != rnb_backend_api::ScalarType::F32
        || !matches!(
            input_desc.role(),
            rnb_backend_api::DeviceTensorRole::Hidden
                | rnb_backend_api::DeviceTensorRole::MoeOutput
        )
    {
        return Err(format!(
            "CUDA Muse QKV device input desc mismatch: got {input_desc:?}, expected rows={seq_len} cols={cols} dtype=F32 role=Hidden|MoeOutput"
        ));
    }
    if sliding_window.is_some_and(|window| window == 0 || window > u32::MAX as usize)
        || num_heads == 0
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
        || num_heads > MUSE_HD128_MAX_GRID_Y
        || num_kv_heads > u32::MAX as usize
        || num_heads.checked_mul(128) != Some(q_rows)
        || num_kv_heads.checked_mul(128) != Some(kv_rows)
        || cols == 0
        || !cols.is_multiple_of(256)
        || seq_len == 0
        || n_ff == 0
        || n_embd == 0
        || !muse_hd128_rope_positions_fit_u32(apply_rope, pos_start, seq_len)
        || !muse_hd128_kernel_extents_fit_u32(seq_len, q_rows, kv_rows, cols, o_cols, n_ff, n_embd)
    {
        return Err("CUDA Muse QKV device input dimensions are invalid".to_string());
    }
    let blocks_per_row = cols / 256;
    let q4_row_bytes = blocks_per_row
        .checked_mul(144)
        .ok_or_else(|| "CUDA Muse QKV row byte overflow".to_string())?;
    let v_row_bytes = match v_quant {
        12 => blocks_per_row.checked_mul(144),
        14 => blocks_per_row.checked_mul(210),
        other => return Err(format!("CUDA Muse QKV unsupported V quant {other}")),
    }
    .ok_or_else(|| "CUDA Muse V row byte overflow".to_string())?;
    let o_blocks = o_cols
        .checked_div(256)
        .filter(|_| o_cols.is_multiple_of(256))
        .ok_or_else(|| "CUDA Muse O cols must be divisible by 256".to_string())?;
    let hidden_blocks = n_embd
        .checked_div(256)
        .filter(|_| n_embd.is_multiple_of(256))
        .ok_or_else(|| "CUDA Muse hidden dim must be divisible by 256".to_string())?;
    let down_blocks = n_ff
        .checked_div(256)
        .filter(|_| n_ff.is_multiple_of(256))
        .ok_or_else(|| "CUDA Muse FFN dim must be divisible by 256".to_string())?;
    let down_row_bytes = match down_quant {
        12 => down_blocks.checked_mul(144),
        13 => down_blocks.checked_mul(176),
        14 => down_blocks.checked_mul(210),
        other => return Err(format!("CUDA Muse unsupported down quant {other}")),
    }
    .ok_or_else(|| "CUDA Muse down row byte overflow".to_string())?;
    let expected_q = q_rows.saturating_mul(q4_row_bytes);
    let expected_k = kv_rows.saturating_mul(q4_row_bytes);
    let expected_v = kv_rows.saturating_mul(v_row_bytes);
    let expected_gate = q_rows.saturating_mul(q4_row_bytes);
    let expected_o = n_embd.saturating_mul(o_blocks).saturating_mul(144);
    let expected_gate_up = n_ff.saturating_mul(hidden_blocks).saturating_mul(144);
    let expected_down = n_embd.saturating_mul(down_row_bytes);
    if q_weights.len() != expected_q
        || k_weights.len() != expected_k
        || v_weights.len() != expected_v
        || attention_gate_weights.len() != expected_gate
        || o_weights.len() != expected_o
        || gate_weights.len() != expected_gate_up
        || up_weights.len() != expected_gate_up
        || down_weights.len() != expected_down
        || attn_norm_weight.len() != n_embd
        || q_norm.len() != 128
        || k_norm.len() != 128
        || post_attn_norm_weight.len() != n_embd
        || ffn_norm_weight.len() != n_embd
        || post_ffn_norm_weight.len() != n_embd
        || o_cols != q_rows
    {
        return Err("CUDA Muse QKV device input weight shape mismatch".to_string());
    }
    let output_desc = rnb_backend_api::DeviceTensorDesc::new(
        seq_len,
        n_embd,
        rnb_backend_api::ScalarType::F32,
        rnb_backend_api::DeviceTensorRole::Hidden,
    );
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    if guard.is_none() {
        *guard = Some(CudaState::open()?);
    }
    let state = guard.as_mut().expect("cuda compute state initialized");
    let input_dev = state.device_tensor_ptr(input_id, input_desc)?;
    let output = state.q4k_muse_prefill_hd128_dense_chain(
        q_weights,
        k_weights,
        v_weights,
        v_quant,
        attention_gate_weights,
        q_rows,
        kv_rows,
        blocks_per_row,
        seq_len,
        None,
        attn_norm_weight,
        q_norm,
        k_norm,
        num_heads,
        num_kv_heads,
        scale,
        rope_theta,
        pos_start,
        apply_rope,
        sliding_window,
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
        &mut [],
        Some(input_dev),
        Some(output_desc),
        norm_eps,
        post_norm_eps,
        Some(input_id),
    )?;
    let Some((k_bits, v_bits, Some(output_id))) = output else {
        return Ok(None);
    };
    Ok(Some(MuseQ4kPrefillDeviceOutput {
        k_bits,
        v_bits,
        output_id,
        output_desc,
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn dflash_q4k_layer_chain(
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    o_weights: &[u8],
    gate_weights: &[u8],
    up_weights: &[u8],
    down_weights: &[u8],
    q_rows: usize,
    kv_rows: usize,
    cols: usize,
    prior_k: &[u16],
    prior_v: &[u16],
    hidden: &mut [f32],
    attn_norm_weight: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    ffn_norm_weight: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    scale: f32,
    rope_theta: f32,
    pos_start: usize,
    window: usize,
    n_ff: usize,
    n_embd: usize,
    norm_eps: f32,
) -> Result<(), String> {
    if cols == 0
        || !cols.is_multiple_of(256)
        || n_ff == 0
        || !n_ff.is_multiple_of(256)
        || n_embd != cols
        || hidden.is_empty()
        || !hidden.len().is_multiple_of(n_embd)
        || num_heads == 0
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
        || q_rows != num_heads.saturating_mul(128)
        || kv_rows != num_kv_heads.saturating_mul(128)
        || window == 0
        || window > u32::MAX as usize
    {
        return Err(format!(
            "CUDA DFlash layer chain invalid geometry: q_rows={q_rows} kv_rows={kv_rows} cols={cols} hidden={} heads={num_heads}/{num_kv_heads} window={window} n_ff={n_ff} n_embd={n_embd}",
            hidden.len()
        ));
    }
    let seq_len = hidden.len() / n_embd;
    if !muse_hd128_rope_positions_fit_u32(true, pos_start, seq_len)
        || !muse_hd128_attention_extents_fit_u32(
            seq_len,
            prior_k.len() / kv_rows + seq_len,
            q_rows,
            kv_rows,
            q_rows,
            n_ff,
            n_embd,
        )
    {
        return Err(format!(
            "CUDA DFlash layer chain kernel extent out of range: seq_len={seq_len} prior_values={} pos_start={pos_start}",
            prior_k.len()
        ));
    }
    if prior_k.len() != prior_v.len()
        || !prior_k.len().is_multiple_of(kv_rows)
        || prior_k.len() / kv_rows >= window
        || attn_norm_weight.len() != n_embd
        || ffn_norm_weight.len() != n_embd
        || q_norm.len() != 128
        || k_norm.len() != 128
    {
        return Err(format!(
            "CUDA DFlash layer chain state shape mismatch: prior_k={} prior_v={} kv_rows={kv_rows} attn_norm={} ffn_norm={} q_norm={} k_norm={}",
            prior_k.len(),
            prior_v.len(),
            attn_norm_weight.len(),
            ffn_norm_weight.len(),
            q_norm.len(),
            k_norm.len(),
        ));
    }
    let blocks = cols / 256;
    let q4_row_bytes = blocks
        .checked_mul(144)
        .ok_or_else(|| "CUDA DFlash Q4 row byte overflow".to_string())?;
    let q6_row_bytes = blocks
        .checked_mul(210)
        .ok_or_else(|| "CUDA DFlash Q6 row byte overflow".to_string())?;
    let o_blocks = q_rows / 256;
    let down_blocks = n_ff / 256;
    let expected = [
        (
            "q",
            q_weights.len(),
            q_rows
                .checked_mul(q4_row_bytes)
                .ok_or_else(|| "CUDA DFlash Q weight byte overflow".to_string())?,
        ),
        (
            "k",
            k_weights.len(),
            kv_rows
                .checked_mul(q4_row_bytes)
                .ok_or_else(|| "CUDA DFlash K weight byte overflow".to_string())?,
        ),
        (
            "v",
            v_weights.len(),
            kv_rows
                .checked_mul(q6_row_bytes)
                .ok_or_else(|| "CUDA DFlash V weight byte overflow".to_string())?,
        ),
        (
            "o",
            o_weights.len(),
            n_embd
                .checked_mul(o_blocks)
                .and_then(|rows| rows.checked_mul(144))
                .ok_or_else(|| "CUDA DFlash O weight byte overflow".to_string())?,
        ),
        (
            "gate",
            gate_weights.len(),
            n_ff.checked_mul(q4_row_bytes)
                .ok_or_else(|| "CUDA DFlash gate weight byte overflow".to_string())?,
        ),
        (
            "up",
            up_weights.len(),
            n_ff.checked_mul(q4_row_bytes)
                .ok_or_else(|| "CUDA DFlash up weight byte overflow".to_string())?,
        ),
        (
            "down",
            down_weights.len(),
            n_embd
                .checked_mul(down_blocks)
                .and_then(|rows| rows.checked_mul(210))
                .ok_or_else(|| "CUDA DFlash down weight byte overflow".to_string())?,
        ),
    ];
    if let Some((name, actual, expected)) = expected
        .into_iter()
        .find(|(_, actual, expected)| actual != expected)
    {
        return Err(format!(
            "CUDA DFlash {name} weight byte mismatch: got {actual}, expected {expected}"
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
        .dflash_q4k_layer_chain(
            q_weights,
            k_weights,
            v_weights,
            o_weights,
            gate_weights,
            up_weights,
            down_weights,
            q_rows,
            kv_rows,
            blocks,
            seq_len,
            prior_k,
            prior_v,
            hidden,
            attn_norm_weight,
            q_norm,
            k_norm,
            ffn_norm_weight,
            num_heads,
            num_kv_heads,
            scale,
            rope_theta,
            pos_start,
            window,
            n_ff,
            n_embd,
            norm_eps,
        )
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

/// DeepSeek4 attention output projection: `output_groups` block-diagonal Q8_0
/// GEMVs feeding one Q8_0 GEMV.
///
/// Callers used to run each group through the host-slice GEMV API, so a single
/// layer paid `groups + 1` input uploads and result downloads to move a few
/// kilobytes each. The intermediate low-rank vector is pure concatenation with
/// no host work in between, so the whole projection can stay on the device and
/// only the final hidden row needs to come back.
///
/// The scratch choice matters: `compute_input`/`compute_output` are the natural
/// endpoints, and the low-rank intermediate deliberately lives in
/// `compute_mid_a` because the Q8_0 GEMV itself stages through
/// `compute_temp_slab` and resolves weights through `compute_weights`. Reusing
/// either of those here would recreate the cu107 aliasing failure.
pub fn deepseek4_q8_output_projection(
    group_weights: &[&[u8]],
    output_b_weights: &[u8],
    rows_per_group: usize,
    cols_per_group: usize,
    hidden_dim: usize,
    attention_output: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !crate::tuning::deepseek4_output_projection_fused_enabled() {
        return Ok(None);
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
        .deepseek4_q8_output_projection(
            group_weights,
            output_b_weights,
            rows_per_group,
            cols_per_group,
            hidden_dim,
            attention_output,
        )
}

impl CudaState {
    fn deepseek4_q8_output_projection(
        &mut self,
        group_weights: &[&[u8]],
        output_b_weights: &[u8],
        rows_per_group: usize,
        cols_per_group: usize,
        hidden_dim: usize,
        attention_output: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        let groups = group_weights.len();
        if groups == 0 {
            return Err("DeepSeek4 output projection needs at least one group".to_string());
        }
        if cols_per_group % 32 != 0 {
            return Err(format!(
                "DeepSeek4 output projection cols_per_group must be a multiple of 32, got {cols_per_group}"
            ));
        }
        let low_rank_len = groups
            .checked_mul(rows_per_group)
            .ok_or_else(|| "DeepSeek4 output projection low-rank length overflow".to_string())?;
        if low_rank_len % 32 != 0 {
            return Err(format!(
                "DeepSeek4 output projection low-rank length must be a multiple of 32, got {low_rank_len}"
            ));
        }
        let expected_input = groups
            .checked_mul(cols_per_group)
            .ok_or_else(|| "DeepSeek4 output projection input length overflow".to_string())?;
        if attention_output.len() != expected_input {
            return Err(format!(
                "DeepSeek4 output projection input len mismatch: got {}, expected {expected_input}",
                attention_output.len()
            ));
        }

        // Q8_0 stores 32 values per 34-byte block. The kernels index
        // `rows * blocks_per_row` blocks unconditionally, so an undersized
        // slice would read past the mapping.
        const Q8_0_BLOCK_BYTES: usize = 34;
        let group_row_bytes = cols_per_group / 32 * Q8_0_BLOCK_BYTES;
        let expected_group_bytes = rows_per_group
            .checked_mul(group_row_bytes)
            .ok_or_else(|| "DeepSeek4 output projection group byte overflow".to_string())?;
        for (index, weights) in group_weights.iter().enumerate() {
            if weights.len() != expected_group_bytes {
                return Err(format!(
                    "DeepSeek4 output projection group {index} byte mismatch: got {}, expected {expected_group_bytes}",
                    weights.len()
                ));
            }
        }
        let expected_output_b_bytes = hidden_dim
            .checked_mul(low_rank_len / 32 * Q8_0_BLOCK_BYTES)
            .ok_or_else(|| "DeepSeek4 output projection output_b byte overflow".to_string())?;
        if output_b_weights.len() != expected_output_b_bytes {
            return Err(format!(
                "DeepSeek4 output projection output_b byte mismatch: got {}, expected {expected_output_b_bytes}",
                output_b_weights.len()
            ));
        }

        let f32_bytes = std::mem::size_of::<f32>();
        let input_bytes = expected_input * f32_bytes;
        let input_dev = self.compute_input_ptr(input_bytes)?;
        unsafe {
            self.api.memcpy_htod_async(
                input_dev,
                attention_output.as_ptr().cast(),
                input_bytes,
                self.stream,
            )?;
        }

        let low_rank_dev = self.compute_mid_a_ptr(low_rank_len * f32_bytes)?;
        let group_input_bytes = (cols_per_group * f32_bytes) as u64;
        let group_output_bytes = (rows_per_group * f32_bytes) as u64;
        for (index, weights) in group_weights.iter().enumerate() {
            let weights_dev =
                self.resident_q8_gemv_weights_ptr(weights, rows_per_group, cols_per_group)?;
            let index = index as u64;
            self.launch_basic_quant_gemv_dev(
                "rnb_q8_0_gemv",
                weights_dev,
                rows_per_group,
                cols_per_group / 32,
                input_dev + index * group_input_bytes,
                low_rank_dev + index * group_output_bytes,
            )?;
        }

        let output_bytes = hidden_dim * f32_bytes;
        let output_dev = self.compute_output_ptr(output_bytes)?;
        let output_b_dev =
            self.resident_q8_gemv_weights_ptr(output_b_weights, hidden_dim, low_rank_len)?;
        self.launch_basic_quant_gemv_dev(
            "rnb_q8_0_gemv",
            output_b_dev,
            hidden_dim,
            low_rank_len / 32,
            low_rank_dev,
            output_dev,
        )?;

        let mut output = vec![0.0f32; hidden_dim];
        unsafe {
            self.api.memcpy_dtoh_async(
                output.as_mut_ptr().cast(),
                output_dev,
                output_bytes,
                self.stream,
            )?;
        }
        self.stream_synchronize()?;
        Ok(Some(output))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        attention_prefill_flash_hd128_muse_dense_chain, muse_hd128_attention_extents_fit_u32,
        muse_hd128_kernel_extents_fit_u32, muse_hd128_rope_positions_fit_u32,
        muse_hd128_weight_extents_fit_u32, q4k_muse_prefill_hd128_dense_chain,
        MUSE_HD128_MAX_GRID_Y,
    };

    #[test]
    fn muse_hd128_kernel_extents_reject_derived_u32_overflow() {
        assert!(muse_hd128_kernel_extents_fit_u32(
            64, 256, 128, 256, 256, 1024, 256
        ));
        if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
            assert!(!muse_hd128_kernel_extents_fit_u32(
                1, too_large, 128, 256, 256, 1024, 256,
            ));
        }
        assert!(!muse_hd128_kernel_extents_fit_u32(
            2,
            1usize << 31,
            128,
            256,
            256,
            1024,
            256,
        ));
        assert!(!muse_hd128_kernel_extents_fit_u32(
            1usize << 24,
            256,
            128,
            256,
            256,
            1024,
            256,
        ));
    }

    #[test]
    fn muse_hd128_attention_extents_reject_kv_and_output_overflow() {
        assert!(muse_hd128_attention_extents_fit_u32(
            64, 128, 256, 128, 256, 1024, 256,
        ));
        assert!(!muse_hd128_attention_extents_fit_u32(
            1,
            2,
            256,
            1usize << 31,
            256,
            1024,
            256,
        ));
        assert!(!muse_hd128_attention_extents_fit_u32(
            1usize << 24,
            1usize << 24,
            256,
            128,
            256,
            1024,
            256,
        ));
        assert!(muse_hd128_attention_extents_fit_u32(
            MUSE_HD128_MAX_GRID_Y,
            MUSE_HD128_MAX_GRID_Y,
            256,
            128,
            256,
            256,
            256,
        ));
        assert!(!muse_hd128_attention_extents_fit_u32(
            MUSE_HD128_MAX_GRID_Y + 1,
            MUSE_HD128_MAX_GRID_Y + 1,
            256,
            128,
            256,
            256,
            256,
        ));
    }

    #[test]
    fn muse_hd128_weight_extents_reject_u32_overflow() {
        assert!(muse_hd128_weight_extents_fit_u32(
            &[144, u32::MAX as usize,]
        ));
        if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
            assert!(!muse_hd128_weight_extents_fit_u32(&[144, too_large,]));
        }
    }

    #[test]
    fn muse_hd128_rope_positions_reject_last_position_overflow() {
        assert!(muse_hd128_rope_positions_fit_u32(
            true,
            u32::MAX as usize,
            1,
        ));
        assert!(!muse_hd128_rope_positions_fit_u32(
            true,
            u32::MAX as usize,
            2,
        ));
        assert!(muse_hd128_rope_positions_fit_u32(false, usize::MAX, 2,));
        assert!(!muse_hd128_rope_positions_fit_u32(true, 0, 0));
    }

    #[test]
    fn muse_hd128_chain_rejects_flat_extent_before_cuda_work() {
        let error = attention_prefill_flash_hd128_muse_dense_chain(
            &[],
            &[],
            &[],
            &[],
            2,
            2,
            1,
            1,
            1.0,
            None,
            &[],
            &[],
            &[],
            &[],
            12,
            &[],
            &[],
            &[],
            128,
            1usize << 31,
            256,
            &mut [],
            1e-5,
            1e-5,
        )
        .expect_err("overflowing flat extent must be rejected");
        assert!(error.contains("kernel extent out of u32 range"), "{error}");
    }

    #[test]
    fn muse_hd128_qkv_chain_rejects_rope_overflow_before_cuda_work() {
        let hidden_input = vec![0.0; 512];
        let error = q4k_muse_prefill_hd128_dense_chain(
            &[],
            &[],
            &[],
            12,
            &[],
            128,
            128,
            256,
            &hidden_input,
            &[],
            &[0.0; 128],
            &[0.0; 128],
            1,
            1,
            1.0,
            10_000.0,
            u32::MAX as usize,
            true,
            None,
            &[],
            &[],
            &[],
            &[],
            12,
            &[],
            &[],
            &[],
            128,
            256,
            256,
            &mut [],
            1e-5,
            1e-5,
        )
        .expect_err("overflowing RoPE position must be rejected");
        assert!(error.contains("RoPE position out of u32 range"), "{error}");
    }

    #[test]
    fn muse_hd128_qkv_chain_rejects_weight_extent_before_cuda_work() {
        let hidden_input = vec![0.0; 1024];
        let num_heads = 65_534;
        let q_rows = num_heads * 128;
        let error = q4k_muse_prefill_hd128_dense_chain(
            &[],
            &[],
            &[],
            12,
            &[],
            q_rows,
            128,
            1024,
            &hidden_input,
            &[],
            &[0.0; 128],
            &[0.0; 128],
            num_heads,
            1,
            1.0,
            10_000.0,
            0,
            false,
            None,
            &[],
            &[],
            &[],
            &[],
            12,
            &[],
            &[],
            &[],
            q_rows,
            256,
            1024,
            &mut [],
            1e-5,
            1e-5,
        )
        .expect_err("overflowing quantized weight extent must be rejected");
        assert!(error.contains("weight extent out of u32 range"), "{error}");
    }

    #[test]
    fn muse_hd128_chain_rejects_zero_dense_dimensions_before_cuda_work() {
        let q = vec![0.0; 128];
        for (n_ff, n_embd) in [(0, 256), (256, 0)] {
            let error = attention_prefill_flash_hd128_muse_dense_chain(
                &q,
                &q,
                &q,
                &q,
                1,
                1,
                1,
                1,
                1.0,
                None,
                &[],
                &[],
                &[],
                &[],
                12,
                &[],
                &[],
                &[],
                128,
                n_ff,
                n_embd,
                &mut [],
                1e-5,
                1e-5,
            )
            .expect_err("zero dense dimension must be rejected");
            assert!(error.contains("invalid attention geometry"), "{error}");
        }
    }

    #[test]
    fn muse_hd128_chain_rejects_sequence_grid_y_overflow_before_cuda_work() {
        let error = attention_prefill_flash_hd128_muse_dense_chain(
            &[],
            &[],
            &[],
            &[],
            MUSE_HD128_MAX_GRID_Y + 1,
            MUSE_HD128_MAX_GRID_Y + 1,
            2,
            1,
            1.0,
            None,
            &[],
            &[],
            &[],
            &[],
            12,
            &[],
            &[],
            &[],
            256,
            256,
            256,
            &mut [],
            1e-5,
            1e-5,
        )
        .expect_err("sequence grid Y overflow must be rejected");
        assert!(error.contains("kernel extent out of u32 range"), "{error}");
    }

    #[test]
    fn muse_hd128_qkv_chain_rejects_zero_sequence_before_cuda_work() {
        let error = q4k_muse_prefill_hd128_dense_chain(
            &[],
            &[],
            &[],
            12,
            &[],
            256,
            128,
            256,
            &[],
            &[],
            &[0.0; 128],
            &[0.0; 128],
            2,
            1,
            1.0,
            10_000.0,
            0,
            false,
            None,
            &[],
            &[],
            &[],
            &[],
            12,
            &[],
            &[],
            &[],
            256,
            256,
            256,
            &mut [],
            1e-5,
            1e-5,
        )
        .expect_err("zero sequence must be rejected");
        assert!(error.contains("dimensions must be non-zero"), "{error}");
    }

    #[test]
    fn muse_hd128_qkv_chain_rejects_zero_dense_dimensions_before_cuda_work() {
        let hidden_input = vec![0.0; 256];
        for (n_ff, n_embd) in [(0, 256), (256, 0)] {
            let error = q4k_muse_prefill_hd128_dense_chain(
                &[],
                &[],
                &[],
                12,
                &[],
                256,
                128,
                256,
                &hidden_input,
                &[],
                &[0.0; 128],
                &[0.0; 128],
                2,
                1,
                1.0,
                10_000.0,
                0,
                false,
                None,
                &[],
                &[],
                &[],
                &[],
                12,
                &[],
                &[],
                &[],
                256,
                n_ff,
                n_embd,
                &mut [],
                1e-5,
                1e-5,
            )
            .expect_err("zero fused dense dimension must be rejected");
            assert!(error.contains("dimensions must be non-zero"), "{error}");
        }
    }

    #[test]
    fn muse_hd128_chains_reject_attention_grid_y_overflow_before_cuda_work() {
        let num_heads = 65_536;
        let q_rows = num_heads * 128;
        let error = attention_prefill_flash_hd128_muse_dense_chain(
            &[],
            &[],
            &[],
            &[],
            1,
            1,
            num_heads,
            1,
            1.0,
            None,
            &[],
            &[],
            &[],
            &[],
            12,
            &[],
            &[],
            &[],
            q_rows,
            256,
            256,
            &mut [],
            1e-5,
            1e-5,
        )
        .expect_err("attention grid Y overflow must be rejected");
        assert!(error.contains("kernel argument out of range"), "{error}");

        let hidden_input = vec![0.0; 256];
        let error = q4k_muse_prefill_hd128_dense_chain(
            &[],
            &[],
            &[],
            12,
            &[],
            q_rows,
            128,
            256,
            &hidden_input,
            &[],
            &[0.0; 128],
            &[0.0; 128],
            num_heads,
            1,
            1.0,
            10_000.0,
            0,
            false,
            None,
            &[],
            &[],
            &[],
            &[],
            12,
            &[],
            &[],
            &[],
            q_rows,
            256,
            256,
            &mut [],
            1e-5,
            1e-5,
        )
        .expect_err("fused attention grid Y overflow must be rejected");
        assert!(error.contains("invalid head geometry"), "{error}");
    }
}
