use super::super::*;
use crate::runtime::moe::NemotronDeviceRoutePackOutput as RawNemotronDeviceRoutePackOutput;

#[derive(Debug, Clone, Copy)]
pub struct NemotronDeviceRouterLogitsOutput {
    pub normalized_id: rnb_backend_api::DeviceTensorId,
    pub normalized_desc: rnb_backend_api::DeviceTensorDesc,
    pub router_logits_id: rnb_backend_api::DeviceTensorId,
    pub router_logits_desc: rnb_backend_api::DeviceTensorDesc,
}

#[derive(Debug)]
pub struct NemotronDeviceRoutePack {
    raw: RawNemotronDeviceRoutePackOutput,
}

impl NemotronDeviceRoutePack {
    pub fn slots(&self) -> usize {
        self.raw.slots
    }

    pub fn seq_len(&self) -> usize {
        self.raw.seq_len
    }

    pub fn expert_used(&self) -> usize {
        self.raw.expert_used
    }
}

pub fn nemotron_prefill_sparse_copy_prefetch(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    n_ff: usize,
    n_embd: usize,
    down_q8: bool,
) -> Result<bool, String> {
    if !tuning::nemotron_prefill_sparse_copy_prefetch_enabled() {
        return Ok(false);
    }
    let slots = up_weights.len();
    if slots == 0 {
        return Ok(false);
    }
    if down_weights.len() != slots {
        return Err("Nemotron sparse prefill prefetch length mismatch".to_string());
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 {
        return Err(format!(
            "Nemotron sparse prefill prefetch dims must be divisible by 32, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = if down_q8 {
        n_embd * (n_ff / 32) * 34
    } else {
        n_embd * (n_ff / 32) * 24
    };
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron sparse prefill prefetch Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron sparse prefill prefetch down weight byte mismatch".to_string());
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
        .prefetch_nemotron_prefill_sparse_q4k(up_weights, down_weights)
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_sparse_relu_sqr_by_token(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let slots = up_weights.len();
    if slots == 0 {
        return Ok(vec![0.0; token_count * n_embd]);
    }
    if down_weights.len() != slots || route_weights.len() != slots || token_ids.len() != slots {
        return Err("Nemotron sparse Q5 batch length mismatch".to_string());
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron sparse Q5 input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 {
        return Err(format!(
            "Nemotron sparse Q5 dims must be divisible by 32, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = n_embd * (n_ff / 32) * 24;
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron sparse Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron sparse Q5_1 down weight byte mismatch".to_string());
    }
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err("Nemotron sparse Q5 token id out of range".to_string());
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
        .nemotron_q5_sparse_relu_sqr_by_token(
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_q8_sparse_relu_sqr_by_token(
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let slots = up_weights.len();
    if slots == 0 {
        return Ok(vec![0.0; token_count * n_embd]);
    }
    if down_weights.len() != slots || route_weights.len() != slots || token_ids.len() != slots {
        return Err("Nemotron sparse Q5/Q8 batch length mismatch".to_string());
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron sparse Q5/Q8 input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 {
        return Err(format!(
            "Nemotron sparse Q5/Q8 dims must be divisible by 32, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = n_embd * (n_ff / 32) * 34;
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron sparse Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron sparse Q8_0 down weight byte mismatch".to_string());
    }
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err("Nemotron sparse Q5/Q8 token id out of range".to_string());
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
        .nemotron_q5_q8_sparse_relu_sqr_by_token(
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            token_count,
            n_ff,
            n_embd,
            input,
        )
}

pub fn nemotron_q8_shared_prefill(
    shared_up: &[u8],
    shared_down: &[u8],
    shared_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !tuning::nemotron_prefill_q8_shared_fused_enabled() {
        return Ok(None);
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron Q8 shared prefill input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    if n_embd % 32 != 0 || shared_ff % 32 != 0 {
        return Err(format!(
            "Nemotron Q8 shared prefill dims must be divisible by 32, got shared_ff={shared_ff} n_embd={n_embd}"
        ));
    }
    let shared_up_bytes = shared_ff * (n_embd / 32) * 34;
    let shared_down_bytes = n_embd * (shared_ff / 32) * 34;
    if shared_up.len() != shared_up_bytes {
        return Err(format!(
            "Nemotron Q8 shared prefill up byte mismatch: got {}, expected {shared_up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != shared_down_bytes {
        return Err(format!(
            "Nemotron Q8 shared prefill down byte mismatch: got {}, expected {shared_down_bytes}",
            shared_down.len()
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
        .nemotron_q8_shared_prefill(
            shared_up,
            shared_down,
            shared_ff,
            n_embd,
            token_count,
            input,
        )
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !tuning::nemotron_prefill_q8_shared_sparse_fused_enabled() {
        return Ok(None);
    }
    let slots = up_weights.len();
    if down_weights.len() != slots || route_weights.len() != slots || token_ids.len() != slots {
        return Err("Nemotron prefill Q8 shared + Q5 sparse batch length mismatch".to_string());
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron prefill Q8 shared + Q5 sparse input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    if residual.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron prefill Q8 shared + Q5 sparse residual length mismatch: got {}, expected {}",
            residual.len(),
            token_count * n_embd
        ));
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 || shared_ff % 32 != 0 {
        return Err(format!(
            "Nemotron prefill Q8 shared + Q5 sparse dims must be divisible by 32, got shared_ff={shared_ff} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let shared_up_bytes = shared_ff * (n_embd / 32) * 34;
    let shared_down_bytes = n_embd * (shared_ff / 32) * 34;
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = n_embd * (n_ff / 32) * 24;
    if shared_up.len() != shared_up_bytes {
        return Err(format!(
            "Nemotron prefill shared Q8_0 up byte mismatch: got {}, expected {shared_up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != shared_down_bytes {
        return Err(format!(
            "Nemotron prefill shared Q8_0 down byte mismatch: got {}, expected {shared_down_bytes}",
            shared_down.len()
        ));
    }
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron prefill sparse Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron prefill sparse Q5_1 down weight byte mismatch".to_string());
    }
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err("Nemotron prefill Q8 shared + Q5 sparse token id out of range".to_string());
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
        .nemotron_q8_shared_q5_sparse_prefill_moe(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input,
            residual,
        )
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_router_logits_from_device_f32(
    input_id: rnb_backend_api::DeviceTensorId,
    input_desc: rnb_backend_api::DeviceTensorDesc,
    norm_weight: &[f32],
    router_weight_f32: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    n_expert: usize,
    norm_eps: f32,
) -> Result<NemotronDeviceRouterLogitsOutput, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    let output = state.nemotron_router_logits_from_device_f32(
        input_id,
        input_desc,
        norm_weight,
        router_weight_f32,
        seq_len,
        hidden_dim,
        n_expert,
        norm_eps,
    )?;
    Ok(NemotronDeviceRouterLogitsOutput {
        normalized_id: output.normalized_id,
        normalized_desc: output.normalized_desc,
        router_logits_id: output.router_logits_id,
        router_logits_desc: output.router_logits_desc,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_device_route_pack_from_logits(
    router_logits_id: rnb_backend_api::DeviceTensorId,
    router_logits_desc: rnb_backend_api::DeviceTensorDesc,
    bias: Option<&[f32]>,
    seq_len: usize,
    n_expert: usize,
    expert_used: usize,
    expert_weight_scale: f32,
) -> Result<NemotronDeviceRoutePack, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    let raw = state.nemotron_device_route_pack_from_logits(
        router_logits_id,
        router_logits_desc,
        bias,
        seq_len,
        n_expert,
        expert_used,
        expert_weight_scale,
    )?;
    Ok(NemotronDeviceRoutePack { raw })
}

pub fn nemotron_device_route_pack_expert_ids(
    route: &NemotronDeviceRoutePack,
) -> Result<Vec<u32>, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    let mut expert_ids = vec![0_u32; route.raw.slots];
    unsafe {
        state.api.memcpy_dtoh_async(
            expert_ids.as_mut_ptr().cast::<libc::c_void>(),
            route.raw.expert_ids_dev,
            std::mem::size_of_val(expert_ids.as_slice()),
            state.stream,
        )?;
    }
    state.stream_synchronize()?;
    Ok(expert_ids)
}

pub fn nemotron_reorder_device_route_pack(
    route: &NemotronDeviceRoutePack,
    order_indices: &[u32],
) -> Result<NemotronDeviceRoutePack, String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    let raw = state.nemotron_reorder_device_route_pack(route.raw, order_indices)?;
    Ok(NemotronDeviceRoutePack { raw })
}

pub fn release_nemotron_device_route_pack(route: NemotronDeviceRoutePack) -> Result<(), String> {
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    state.release_nemotron_device_route_pack(route.raw)
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_device(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    residual_id: rnb_backend_api::DeviceTensorId,
) -> Result<Option<rnb_backend_api::DeviceTensorId>, String> {
    if !tuning::nemotron_prefill_q8_shared_sparse_fused_enabled() {
        return Ok(None);
    }
    let slots = up_weights.len();
    if down_weights.len() != slots || route_weights.len() != slots || token_ids.len() != slots {
        return Err("Nemotron prefill Q8 shared + Q5 sparse batch length mismatch".to_string());
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 || shared_ff % 32 != 0 {
        return Err(format!(
            "Nemotron prefill Q8 shared + Q5 sparse dims must be divisible by 32, got shared_ff={shared_ff} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let shared_up_bytes = shared_ff * (n_embd / 32) * 34;
    let shared_down_bytes = n_embd * (shared_ff / 32) * 34;
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = n_embd * (n_ff / 32) * 24;
    if shared_up.len() != shared_up_bytes {
        return Err(format!(
            "Nemotron prefill shared Q8_0 up byte mismatch: got {}, expected {shared_up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != shared_down_bytes {
        return Err(format!(
            "Nemotron prefill shared Q8_0 down byte mismatch: got {}, expected {shared_down_bytes}",
            shared_down.len()
        ));
    }
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron prefill sparse Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron prefill sparse Q5_1 down weight byte mismatch".to_string());
    }
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err("Nemotron prefill Q8 shared + Q5 sparse token id out of range".to_string());
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
        .nemotron_q8_shared_q5_sparse_prefill_moe_device(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_id,
            residual_id,
        )
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    residual_id: rnb_backend_api::DeviceTensorId,
    residual_desc: rnb_backend_api::DeviceTensorDesc,
) -> Result<Option<rnb_backend_api::DeviceTensorId>, String> {
    if !tuning::nemotron_prefill_q8_shared_sparse_fused_enabled() {
        return Ok(None);
    }
    let slots = up_weights.len();
    if down_weights.len() != slots || route_weights.len() != slots || token_ids.len() != slots {
        return Err("Nemotron prefill Q8 shared + Q5 sparse batch length mismatch".to_string());
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 || shared_ff % 32 != 0 {
        return Err(format!(
            "Nemotron prefill Q8 shared + Q5 sparse dims must be divisible by 32, got shared_ff={shared_ff} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let shared_up_bytes = shared_ff * (n_embd / 32) * 34;
    let shared_down_bytes = n_embd * (shared_ff / 32) * 34;
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = n_embd * (n_ff / 32) * 24;
    if shared_up.len() != shared_up_bytes {
        return Err(format!(
            "Nemotron prefill shared Q8_0 up byte mismatch: got {}, expected {shared_up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != shared_down_bytes {
        return Err(format!(
            "Nemotron prefill shared Q8_0 down byte mismatch: got {}, expected {shared_down_bytes}",
            shared_down.len()
        ));
    }
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron prefill sparse Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron prefill sparse Q5_1 down weight byte mismatch".to_string());
    }
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err("Nemotron prefill Q8 shared + Q5 sparse token id out of range".to_string());
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
        .nemotron_q8_shared_q5_sparse_prefill_moe_device_with_residual_desc(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_id,
            residual_id,
            residual_desc,
        )
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_pack: &NemotronDeviceRoutePack,
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input_id: rnb_backend_api::DeviceTensorId,
    residual_id: rnb_backend_api::DeviceTensorId,
    residual_desc: rnb_backend_api::DeviceTensorDesc,
) -> Result<Option<rnb_backend_api::DeviceTensorId>, String> {
    if !tuning::nemotron_prefill_q8_shared_sparse_fused_enabled() {
        return Ok(None);
    }
    let slots = route_pack.slots();
    if route_pack.seq_len() != token_count {
        return Err(format!(
            "Nemotron device route pack token count mismatch: got {}, expected {token_count}",
            route_pack.seq_len()
        ));
    }
    if down_weights.len() != slots || up_weights.len() != slots {
        return Err("Nemotron prefill Q8 shared + Q5 sparse batch length mismatch".to_string());
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 || shared_ff % 32 != 0 {
        return Err(format!(
            "Nemotron prefill Q8 shared + Q5 sparse dims must be divisible by 32, got shared_ff={shared_ff} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let shared_up_bytes = shared_ff * (n_embd / 32) * 34;
    let shared_down_bytes = n_embd * (shared_ff / 32) * 34;
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = n_embd * (n_ff / 32) * 24;
    if shared_up.len() != shared_up_bytes {
        return Err(format!(
            "Nemotron prefill shared Q8_0 up byte mismatch: got {}, expected {shared_up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != shared_down_bytes {
        return Err(format!(
            "Nemotron prefill shared Q8_0 down byte mismatch: got {}, expected {shared_down_bytes}",
            shared_down.len()
        ));
    }
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron prefill sparse Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron prefill sparse Q5_1 down weight byte mismatch".to_string());
    }
    let compute = DEFAULT_CUDA_COMPUTE.get_or_init(|| Mutex::new(None));
    let mut guard = compute
        .lock()
        .map_err(|_| "cuda compute state lock poisoned".to_string())?;
    let state = guard
        .as_mut()
        .ok_or_else(|| "cuda compute state is not initialized".to_string())?;
    state
        .nemotron_q8_shared_q5_sparse_prefill_moe_device_route_pack(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_pack.raw,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input_id,
            residual_id,
            residual_desc,
        )
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
    shared_up: &[u8],
    shared_down: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    shared_ff: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    token_count: usize,
    input: &[f32],
    residual: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    if !tuning::nemotron_prefill_q8_shared_sparse_fused_enabled() {
        return Ok(None);
    }
    validate_nemotron_q5_sparse_full_layer_batch(
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        token_count,
        n_expert,
        n_ff,
        n_embd,
        input,
    )?;
    if residual.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron cached prefill Q8 shared + Q5 sparse residual length mismatch: got {}, expected {}",
            residual.len(),
            token_count * n_embd
        ));
    }
    if shared_ff % 32 != 0 {
        return Err(format!(
            "Nemotron cached prefill Q8 shared + Q5 sparse shared_ff must be divisible by 32, got {shared_ff}"
        ));
    }
    let shared_up_bytes = shared_ff * (n_embd / 32) * 34;
    let shared_down_bytes = n_embd * (shared_ff / 32) * 34;
    if shared_up.len() != shared_up_bytes {
        return Err(format!(
            "Nemotron cached prefill shared Q8_0 up byte mismatch: got {}, expected {shared_up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != shared_down_bytes {
        return Err(format!(
            "Nemotron cached prefill shared Q8_0 down byte mismatch: got {}, expected {shared_down_bytes}",
            shared_down.len()
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
        .nemotron_q8_shared_q5_sparse_prefill_moe_cached_layer(
            shared_up,
            shared_down,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            shared_ff,
            n_ff,
            n_embd,
            token_count,
            input,
            residual,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_decode_moe_shared_sparse(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let sparse_slots = up_weights.len();
    if down_weights.len() != sparse_slots || route_weights.len() != sparse_slots {
        return Err("Nemotron decode Q5 shared+sparse batch length mismatch".to_string());
    }
    if input.len() != n_embd {
        return Err(format!(
            "Nemotron decode Q5 shared+sparse input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 {
        return Err(format!(
            "Nemotron decode Q5 shared+sparse dims must be divisible by 32, got n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = n_embd * (n_ff / 32) * 24;
    if shared_up.len() != up_bytes {
        return Err(format!(
            "Nemotron decode shared Q5_0 up byte mismatch: got {}, expected {up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != down_bytes {
        return Err(format!(
            "Nemotron decode shared Q5_1 down byte mismatch: got {}, expected {down_bytes}",
            shared_down.len()
        ));
    }
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron decode sparse Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron decode sparse Q5_1 down weight byte mismatch".to_string());
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
        .nemotron_q5_decode_moe_shared_sparse(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_decode_moe(
    shared_up: &[u8],
    shared_down: &[u8],
    up_weights: &[&[u8]],
    down_weights: &[&[u8]],
    route_weights: &[f32],
    shared_ff: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let sparse_slots = up_weights.len();
    if down_weights.len() != sparse_slots || route_weights.len() != sparse_slots {
        return Err("Nemotron decode Q8 shared + Q5 sparse batch length mismatch".to_string());
    }
    if input.len() != n_embd {
        return Err(format!(
            "Nemotron decode Q8 shared + Q5 sparse input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 || shared_ff % 32 != 0 {
        return Err(format!(
            "Nemotron decode Q8 shared + Q5 sparse dims must be divisible by 32, got shared_ff={shared_ff} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let shared_up_bytes = shared_ff * (n_embd / 32) * 34;
    let shared_down_bytes = n_embd * (shared_ff / 32) * 34;
    let up_bytes = n_ff * (n_embd / 32) * 22;
    let down_bytes = n_embd * (n_ff / 32) * 24;
    if shared_up.len() != shared_up_bytes {
        return Err(format!(
            "Nemotron decode shared Q8_0 up byte mismatch: got {}, expected {shared_up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != shared_down_bytes {
        return Err(format!(
            "Nemotron decode shared Q8_0 down byte mismatch: got {}, expected {shared_down_bytes}",
            shared_down.len()
        ));
    }
    if up_weights.iter().any(|w| w.len() != up_bytes) {
        return Err("Nemotron decode sparse Q5_0 up weight byte mismatch".to_string());
    }
    if down_weights.iter().any(|w| w.len() != down_bytes) {
        return Err("Nemotron decode sparse Q5_1 down weight byte mismatch".to_string());
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
        .nemotron_q8_shared_q5_sparse_decode_moe(
            shared_up,
            shared_down,
            up_weights,
            down_weights,
            route_weights,
            shared_ff,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q8_shared_q5_sparse_decode_moe_cached_layer(
    shared_up: &[u8],
    shared_down: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    shared_ff: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    let sparse_slots = expert_ids.len();
    if route_weights.len() != sparse_slots {
        return Err("Nemotron decode cached Q8 shared + Q5 sparse length mismatch".to_string());
    }
    if input.len() != n_embd {
        return Err(format!(
            "Nemotron decode cached Q8 shared + Q5 sparse input length mismatch: got {}, expected {n_embd}",
            input.len()
        ));
    }
    if n_embd % 32 != 0 || n_ff % 32 != 0 || shared_ff % 32 != 0 {
        return Err(format!(
            "Nemotron decode cached Q8 shared + Q5 sparse dims must be divisible by 32, got shared_ff={shared_ff} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let shared_up_bytes = shared_ff * (n_embd / 32) * 34;
    let shared_down_bytes = n_embd * (shared_ff / 32) * 34;
    if shared_up.len() != shared_up_bytes {
        return Err(format!(
            "Nemotron decode shared Q8_0 up byte mismatch: got {}, expected {shared_up_bytes}",
            shared_up.len()
        ));
    }
    if shared_down.len() != shared_down_bytes {
        return Err(format!(
            "Nemotron decode shared Q8_0 down byte mismatch: got {}, expected {shared_down_bytes}",
            shared_down.len()
        ));
    }
    validate_nemotron_q5_layer_weights(up_all, down_all, n_expert, n_ff, n_embd)?;
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
        .nemotron_q8_shared_q5_sparse_decode_moe_cached_layer(
            shared_up,
            shared_down,
            up_all,
            down_all,
            expert_ids,
            route_weights,
            shared_ff,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_sparse_relu_sqr_full_layer_by_token(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let slots = expert_ids.len();
    if slots == 0 {
        return Ok(vec![0.0; token_count * n_embd]);
    }
    if route_weights.len() != slots || token_ids.len() != slots {
        return Err("Nemotron full-layer sparse Q5 batch length mismatch".to_string());
    }
    if input.len() != token_count * n_embd {
        return Err(format!(
            "Nemotron full-layer sparse Q5 input length mismatch: got {}, expected {}",
            input.len(),
            token_count * n_embd
        ));
    }
    if n_expert == 0 || n_embd % 32 != 0 || n_ff % 32 != 0 {
        return Err(format!(
            "Nemotron full-layer sparse Q5 dims must be non-zero and divisible by 32, got n_expert={n_expert} n_ff={n_ff} n_embd={n_embd}"
        ));
    }
    let up_expert_bytes = n_ff * (n_embd / 32) * 22;
    let down_expert_bytes = n_embd * (n_ff / 32) * 24;
    if up_all.len() != n_expert * up_expert_bytes {
        return Err(format!(
            "Nemotron full-layer Q5_0 up byte mismatch: got {}, expected {}",
            up_all.len(),
            n_expert * up_expert_bytes
        ));
    }
    if down_all.len() != n_expert * down_expert_bytes {
        return Err(format!(
            "Nemotron full-layer Q5_1 down byte mismatch: got {}, expected {}",
            down_all.len(),
            n_expert * down_expert_bytes
        ));
    }
    if expert_ids.iter().any(|&expert| expert as usize >= n_expert) {
        return Err("Nemotron full-layer sparse Q5 expert id out of range".to_string());
    }
    if token_ids.iter().any(|&token| token as usize >= token_count) {
        return Err("Nemotron full-layer sparse Q5 token id out of range".to_string());
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
        .nemotron_q5_sparse_relu_sqr_full_layer_by_token(
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            n_expert,
            n_ff,
            n_embd,
            input,
        )
}

pub fn nemotron_q5_register_layer(
    up_all: &[u8],
    down_all: &[u8],
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
) -> Result<bool, String> {
    if !tuning::nemotron_q5_layer_cache_enabled() {
        return Ok(false);
    }
    validate_nemotron_q5_layer_weights(up_all, down_all, n_expert, n_ff, n_embd)?;
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
        .register_nemotron_q5_layer(up_all, down_all, n_expert, n_ff, n_embd)
}

pub fn nemotron_q5_q8_register_layer(
    up_all: &[u8],
    down_all: &[u8],
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
) -> Result<bool, String> {
    if !tuning::nemotron_q5_layer_cache_enabled() {
        return Ok(false);
    }
    validate_nemotron_q5_q8_layer_weights(up_all, down_all, n_expert, n_ff, n_embd)?;
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
        .register_nemotron_q5_q8_layer(up_all, down_all, n_expert, n_ff, n_embd)
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_sparse_relu_sqr_cached_layer_by_token(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    validate_nemotron_q5_sparse_full_layer_batch(
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        token_count,
        n_expert,
        n_ff,
        n_embd,
        input,
    )?;
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
        .nemotron_q5_sparse_relu_sqr_cached_layer_by_token(
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            n_expert,
            n_ff,
            n_embd,
            input,
        )
}

#[allow(clippy::too_many_arguments)]
pub fn nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token(
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[u32],
    route_weights: &[f32],
    token_ids: &[u32],
    token_count: usize,
    n_expert: usize,
    n_ff: usize,
    n_embd: usize,
    input: &[f32],
) -> Result<Option<Vec<f32>>, String> {
    validate_nemotron_q5_q8_sparse_full_layer_batch(
        up_all,
        down_all,
        expert_ids,
        route_weights,
        token_ids,
        token_count,
        n_expert,
        n_ff,
        n_embd,
        input,
    )?;
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
        .nemotron_q5_q8_sparse_relu_sqr_cached_layer_by_token(
            up_all,
            down_all,
            expert_ids,
            route_weights,
            token_ids,
            token_count,
            n_expert,
            n_ff,
            n_embd,
            input,
        )
}
