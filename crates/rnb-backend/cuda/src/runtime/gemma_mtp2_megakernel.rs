use super::driver::{GEMMA_MTP2_MEGAKERNEL_CUBIN, GEMMA_MTP2_MEGAKERNEL_PTX};
use super::types::CudaState;

const Q5_1_QUANT: u32 = 7;
const Q8_0_QUANT: u32 = 8;
const BLOCK_THREADS: i32 = 256;

#[derive(Debug, Clone, Copy)]
pub(in crate::runtime) struct GemmaMtp2FinalizeRequest<'a> {
    pub residual_dev: u64,
    pub shared_raw_dev: u64,
    pub post_norm_1: &'a [f32],
    pub post_norm_2: &'a [f32],
    pub common_post_norm: &'a [f32],
    pub norm_eps: f32,
    pub unit_offset: bool,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::runtime) struct GemmaMtp2SelectedSparseRequest<'a> {
    pub normalized_dev: u64,
    pub expert_ids_dev: u64,
    pub route_weights_dev: u64,
    pub gate_up_weights: &'a [u8],
    pub down_weights: &'a [u8],
    pub down_scale: &'a [f32],
    pub tokens: usize,
    pub hidden_dim: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub top_k: usize,
    pub down_quant: u32,
    pub finalize: Option<GemmaMtp2FinalizeRequest<'a>>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct GemmaMtp2SelectedSparseParamsHost {
    normalized: u64,
    normalized_qs: u64,
    normalized_ds: u64,
    expert_ids: u64,
    route_weights: u64,
    gate_up_weights: u64,
    down_weights: u64,
    down_scale: u64,
    gate_up_scratch: u64,
    rank_output: u64,
    output: u64,
    residual: u64,
    shared_raw: u64,
    post_norm_1: u64,
    post_norm_2: u64,
    common_post_norm: u64,
    tokens: u32,
    hidden_dim: u32,
    n_ff: u32,
    n_expert: u32,
    top_k: u32,
    down_quant: u32,
    q8dot_wide: u32,
    norm_eps: f32,
    finalize_enabled: u32,
    unit_offset: u32,
}

impl CudaState {
    fn ensure_gemma_mtp2_module(&mut self) -> Result<usize, String> {
        self.set_current()?;
        if self.gemma_mtp2_module.is_none() {
            let module = unsafe {
                if crate::tuning::cubin_modules_enabled() {
                    self.api.module_load_cubin_or_ptx(
                        GEMMA_MTP2_MEGAKERNEL_CUBIN,
                        GEMMA_MTP2_MEGAKERNEL_PTX,
                    )?
                } else {
                    self.api.module_load_data(GEMMA_MTP2_MEGAKERNEL_PTX)?
                }
            };
            self.gemma_mtp2_module = Some(module as usize);
        }
        self.gemma_mtp2_module
            .ok_or_else(|| "missing Gemma MTP2 megakernel module".to_string())
    }

    fn gemma_mtp2_function(&mut self) -> Result<*mut libc::c_void, String> {
        let module = self.ensure_gemma_mtp2_module()?;
        unsafe {
            self.api.module_get_function(
                module as *mut libc::c_void,
                "rnb_gemma_mtp2_selected_sparse_sm86",
            )
        }
    }

    pub(in crate::runtime) fn stage_gemma_mtp2_selected_sparse(
        &mut self,
        request: GemmaMtp2SelectedSparseRequest<'_>,
    ) -> Result<u64, String> {
        validate_request(&request)?;
        self.set_current()?;

        let slots = request
            .tokens
            .checked_mul(request.top_k)
            .ok_or_else(|| "Gemma MTP2 route slot count overflow".to_string())?;
        let gate_up_values = slots
            .checked_mul(request.n_ff)
            .and_then(|values| values.checked_mul(2))
            .ok_or_else(|| "Gemma MTP2 gate/up scratch size overflow".to_string())?;
        let rank_values = slots
            .checked_mul(request.hidden_dim)
            .ok_or_else(|| "Gemma MTP2 rank output size overflow".to_string())?;
        let output_values = request
            .tokens
            .checked_mul(request.hidden_dim)
            .ok_or_else(|| "Gemma MTP2 output size overflow".to_string())?;
        let gate_up_bytes = gate_up_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma MTP2 gate/up byte size overflow".to_string())?;
        let rank_bytes = rank_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma MTP2 rank output byte size overflow".to_string())?;
        let output_bytes = output_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Gemma MTP2 output byte size overflow".to_string())?;
        let normalized_qs_bytes = request
            .tokens
            .checked_mul(request.hidden_dim)
            .ok_or_else(|| "Gemma MTP2 normalized q8 byte size overflow".to_string())?;
        let normalized_ds_bytes = normalized_qs_bytes
            .checked_div(32)
            .and_then(|chunks| chunks.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Gemma MTP2 normalized q8 scale size overflow".to_string())?;

        let normalized_qs = super::ensure_device_buffer(
            &self.api,
            &mut self.gemma_mtp2_ctx.normalized_qs,
            &mut self.gemma_mtp2_ctx.normalized_qs_capacity,
            normalized_qs_bytes,
        )?;
        let normalized_ds = super::ensure_device_buffer(
            &self.api,
            &mut self.gemma_mtp2_ctx.normalized_ds,
            &mut self.gemma_mtp2_ctx.normalized_ds_capacity,
            normalized_ds_bytes,
        )?;
        let gate_up_scratch = super::ensure_device_buffer(
            &self.api,
            &mut self.gemma_mtp2_ctx.gate_up,
            &mut self.gemma_mtp2_ctx.gate_up_capacity,
            gate_up_bytes,
        )?;
        let rank_output = super::ensure_device_buffer(
            &self.api,
            &mut self.gemma_mtp2_ctx.rank_output,
            &mut self.gemma_mtp2_ctx.rank_output_capacity,
            rank_bytes,
        )?;
        let output = super::ensure_device_buffer(
            &self.api,
            &mut self.gemma_mtp2_ctx.output,
            &mut self.gemma_mtp2_ctx.output_capacity,
            output_bytes,
        )?;

        let down_scale = self.resident_f32_ptr_stable_source(request.down_scale)?;
        let (gate_up_weights, gate_up_pin) =
            self.resident_q4k_weights_ptr_pinned_with_lease(request.gate_up_weights)?;
        if let Some(key) = gate_up_pin {
            self.gemma_mtp2_ctx.pending_weight_pins.push(key);
        }
        let (down_weights, down_pin) =
            match self.resident_q4k_weights_ptr_pinned_with_lease(request.down_weights) {
                Ok(resident) => resident,
                Err(err) => {
                    return match self.flush_gemma_mtp2_weight_pins() {
                        Ok(()) => Err(err),
                        Err(cleanup_err) => Err(format!(
                            "{err}; failed to release Gemma MTP2 weight pins: {cleanup_err}"
                        )),
                    };
                }
            };
        if let Some(key) = down_pin {
            self.gemma_mtp2_ctx.pending_weight_pins.push(key);
        }

        let result = (|| {
            let finalize_ptrs = request
                .finalize
                .map(|finalize| {
                    Ok::<_, String>((
                        self.resident_f32_ptr_stable_source(finalize.post_norm_1)?,
                        self.resident_f32_ptr_stable_source(finalize.post_norm_2)?,
                        self.resident_f32_ptr_stable_source(finalize.common_post_norm)?,
                    ))
                })
                .transpose()?;
            let function = self.gemma_mtp2_function()?;
            let sm_count = unsafe { self.api.device_multiprocessor_count()? };
            let blocks_per_sm = unsafe {
                self.api.occupancy_max_active_blocks_per_multiprocessor(
                    function,
                    BLOCK_THREADS,
                    0,
                )?
            };
            if sm_count <= 0 || blocks_per_sm <= 0 {
                return Err(format!(
                    "Gemma MTP2 cooperative occupancy invalid: sm_count={sm_count}, blocks_per_sm={blocks_per_sm}"
                ));
            }
            let grid_x = u32::try_from(sm_count.saturating_mul(blocks_per_sm))
                .map_err(|_| "Gemma MTP2 cooperative grid overflow".to_string())?;
            if request.finalize.is_some() && request.tokens > grid_x as usize {
                return Err(format!(
                    "Gemma MTP2 finalize requires one cooperative block per token: tokens={}, grid_x={grid_x}",
                    request.tokens
                ));
            }
            let mut params = GemmaMtp2SelectedSparseParamsHost {
                normalized: request.normalized_dev,
                normalized_qs,
                normalized_ds,
                expert_ids: request.expert_ids_dev,
                route_weights: request.route_weights_dev,
                gate_up_weights,
                down_weights,
                down_scale,
                gate_up_scratch,
                rank_output,
                output,
                residual: request.finalize.map_or(0, |finalize| finalize.residual_dev),
                shared_raw: request
                    .finalize
                    .map_or(0, |finalize| finalize.shared_raw_dev),
                post_norm_1: finalize_ptrs.map_or(0, |ptrs| ptrs.0),
                post_norm_2: finalize_ptrs.map_or(0, |ptrs| ptrs.1),
                common_post_norm: finalize_ptrs.map_or(0, |ptrs| ptrs.2),
                tokens: u32::try_from(request.tokens)
                    .map_err(|_| "Gemma MTP2 tokens exceed u32".to_string())?,
                hidden_dim: u32::try_from(request.hidden_dim)
                    .map_err(|_| "Gemma MTP2 hidden_dim exceeds u32".to_string())?,
                n_ff: u32::try_from(request.n_ff)
                    .map_err(|_| "Gemma MTP2 n_ff exceeds u32".to_string())?,
                n_expert: u32::try_from(request.n_expert)
                    .map_err(|_| "Gemma MTP2 n_expert exceeds u32".to_string())?,
                top_k: u32::try_from(request.top_k)
                    .map_err(|_| "Gemma MTP2 top_k exceeds u32".to_string())?,
                down_quant: request.down_quant,
                q8dot_wide: u32::from(crate::tuning::q4k_q8dot_wide_enabled()),
                norm_eps: request.finalize.map_or(0.0, |finalize| finalize.norm_eps),
                finalize_enabled: u32::from(request.finalize.is_some()),
                unit_offset: u32::from(
                    request
                        .finalize
                        .is_some_and(|finalize| finalize.unit_offset),
                ),
            };
            let params_ptr = (&mut params as *mut GemmaMtp2SelectedSparseParamsHost).cast();
            let mut args = [params_ptr];
            unsafe {
                self.api.launch_cooperative_kernel(
                    function,
                    (grid_x, 1, 1),
                    (BLOCK_THREADS as u32, 1, 1),
                    0,
                    self.stream,
                    args.as_mut_ptr(),
                )?;
            }
            Ok(output)
        })();

        match result {
            Ok(output) => Ok(output),
            Err(err) => match self.flush_gemma_mtp2_weight_pins() {
                Ok(()) => Err(err),
                Err(cleanup_err) => Err(format!(
                    "{err}; failed to release Gemma MTP2 weight pins: {cleanup_err}"
                )),
            },
        }
    }

    pub(in crate::runtime) fn release_gemma_mtp2_weight_pins_after_sync(&mut self) {
        let keys = std::mem::take(&mut self.gemma_mtp2_ctx.pending_weight_pins);
        for key in keys {
            self.unpin_resident_q4k_key(key);
        }
    }

    pub(in crate::runtime) fn flush_gemma_mtp2_weight_pins(&mut self) -> Result<(), String> {
        if self.gemma_mtp2_ctx.pending_weight_pins.is_empty() {
            return Ok(());
        }
        let sync_result = self.stream_synchronize();
        self.release_gemma_mtp2_weight_pins_after_sync();
        sync_result
    }
}

fn validate_request(request: &GemmaMtp2SelectedSparseRequest<'_>) -> Result<(), String> {
    if request.tokens == 0 || request.top_k == 0 {
        return Err("Gemma MTP2 selected sparse requires non-zero tokens and top_k".to_string());
    }
    if request.hidden_dim == 0 || request.hidden_dim % 256 != 0 {
        return Err(format!(
            "Gemma MTP2 hidden_dim must be a non-zero multiple of 256: {}",
            request.hidden_dim
        ));
    }
    if request.n_ff == 0 || request.n_ff % 32 != 0 {
        return Err(format!(
            "Gemma MTP2 n_ff must be a non-zero multiple of 32: {}",
            request.n_ff
        ));
    }
    if request.n_expert == 0 || request.top_k > request.n_expert {
        return Err(format!(
            "Gemma MTP2 expert shape invalid: experts={}, top_k={}",
            request.n_expert, request.top_k
        ));
    }
    if !matches!(request.down_quant, Q5_1_QUANT | Q8_0_QUANT) {
        return Err(format!(
            "Gemma MTP2 down quant must be Q5_1({Q5_1_QUANT}) or Q8_0({Q8_0_QUANT}), got {}",
            request.down_quant
        ));
    }
    if request.down_scale.len() != request.n_expert {
        return Err(format!(
            "Gemma MTP2 down_scale length mismatch: got {}, expected {}",
            request.down_scale.len(),
            request.n_expert
        ));
    }
    if let Some(finalize) = request.finalize {
        for (name, values) in [
            ("post_norm_1", finalize.post_norm_1),
            ("post_norm_2", finalize.post_norm_2),
            ("common_post_norm", finalize.common_post_norm),
        ] {
            if values.len() != request.hidden_dim {
                return Err(format!(
                    "Gemma MTP2 finalize {name} length mismatch: got {}, expected {}",
                    values.len(),
                    request.hidden_dim
                ));
            }
        }
        if finalize.residual_dev == 0 || finalize.shared_raw_dev == 0 {
            return Err(
                "Gemma MTP2 finalize requires non-zero residual and shared pointers".into(),
            );
        }
        if !finalize.norm_eps.is_finite() || finalize.norm_eps <= 0.0 {
            return Err(format!(
                "Gemma MTP2 finalize norm_eps must be finite and positive, got {}",
                finalize.norm_eps
            ));
        }
    }

    let q4_blocks = request.hidden_dim / 256;
    let gate_up_row_bytes = q4_blocks
        .checked_mul(144)
        .ok_or_else(|| "Gemma MTP2 gate/up row size overflow".to_string())?;
    let expected_gate_up = request
        .n_expert
        .checked_mul(request.n_ff)
        .and_then(|rows| rows.checked_mul(2))
        .and_then(|rows| rows.checked_mul(gate_up_row_bytes))
        .ok_or_else(|| "Gemma MTP2 gate/up weight size overflow".to_string())?;
    if request.gate_up_weights.len() != expected_gate_up {
        return Err(format!(
            "Gemma MTP2 gate/up length mismatch: got {}, expected {expected_gate_up}",
            request.gate_up_weights.len()
        ));
    }

    let down_block_bytes = if request.down_quant == Q5_1_QUANT {
        24
    } else {
        34
    };
    let down_row_bytes = (request.n_ff / 32)
        .checked_mul(down_block_bytes)
        .ok_or_else(|| "Gemma MTP2 down row size overflow".to_string())?;
    let expected_down = request
        .n_expert
        .checked_mul(request.hidden_dim)
        .and_then(|rows| rows.checked_mul(down_row_bytes))
        .ok_or_else(|| "Gemma MTP2 down weight size overflow".to_string())?;
    if request.down_weights.len() != expected_down {
        return Err(format!(
            "Gemma MTP2 down length mismatch: got {}, expected {expected_down}",
            request.down_weights.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_inexact_weight_layout() {
        let request = GemmaMtp2SelectedSparseRequest {
            normalized_dev: 1,
            expert_ids_dev: 2,
            route_weights_dev: 3,
            gate_up_weights: &[0; 143],
            down_weights: &[0; 23],
            down_scale: &[1.0],
            tokens: 2,
            hidden_dim: 256,
            n_ff: 32,
            n_expert: 1,
            top_k: 1,
            down_quant: Q5_1_QUANT,
            finalize: None,
        };
        assert!(validate_request(&request)
            .unwrap_err()
            .contains("gate/up length mismatch"));
    }
}
