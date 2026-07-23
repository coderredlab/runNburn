//! Qwen MoE shared expert decode path.

use super::*;

pub(super) fn compute_shared_expert(
    view: &Qwen35MoEView<'_>,
    h: &[f32],
    gate_scalar: f32,
    shared_in_sparse_gpu: bool,
    profile_enabled: bool,
) -> Vec<f32> {
    let shared_start = Instant::now();
    let n_ff = view.n_ff;
    let n_embd = view.n_embd;

    let sh_gate_bpr = qwen35_bytes_per_row(n_embd, view.shared_gate_quant, "shared_gate");
    let sh_up_bpr = qwen35_bytes_per_row(n_embd, view.shared_up_quant, "shared_up");
    let sh_down_bpr = qwen35_bytes_per_row(n_ff, view.shared_down_quant, "shared_down");
    let inner_gemv = qwen_moe_decode_inner_gemv_enabled();
    let mut shared_out = Vec::new();
    if shared_in_sparse_gpu {
        shared_out.resize(n_embd, 0.0);
    }
    if shared_out.is_empty()
        && view.shared_gate_quant == GGMLType::Q4_K
        && view.shared_up_quant == GGMLType::Q4_K
    {
        if let Some(out) = qwen_moe_backend::qwen_moe_decode_expert(
            view.shared_gate_bytes,
            view.shared_up_bytes,
            view.shared_down_bytes,
            view.shared_down_quant,
            n_ff,
            n_embd,
            h,
        ) {
            shared_out = out;
        }
    }
    if shared_out.is_empty() {
        let mut sh_gate = vec![0f32; n_ff];
        let mut sh_up = vec![0f32; n_ff];
        #[cfg(target_arch = "aarch64")]
        let pair_done = inner_gemv
            && qwen_moe_decode_gate_up_pair_gemv_enabled()
            && view.shared_gate_quant == GGMLType::Q4_K
            && view.shared_up_quant == GGMLType::Q4_K
            && crate::engine::quantized_dispatch::dispatch_q4k_pair_q8k(
                view.shared_gate_bytes,
                view.shared_up_bytes,
                h,
                &mut sh_gate,
                &mut sh_up,
                n_ff,
                n_embd,
                sh_gate_bpr,
                sh_up_bpr,
            );
        #[cfg(not(target_arch = "aarch64"))]
        let pair_done = false;

        if pair_done {
            // paired path filled sh_gate and sh_up
        } else if inner_gemv {
            gemv_generic(
                view.shared_gate_bytes,
                h,
                &mut sh_gate,
                n_ff,
                n_embd,
                1,
                sh_gate_bpr,
                view.shared_gate_quant,
            );
            gemv_generic(
                view.shared_up_bytes,
                h,
                &mut sh_up,
                n_ff,
                n_embd,
                1,
                sh_up_bpr,
                view.shared_up_quant,
            );
        } else {
            for r in 0..n_ff {
                let rb = &view.shared_gate_bytes[r * sh_gate_bpr..(r + 1) * sh_gate_bpr];
                sh_gate[r] = dot_k_block_row(rb, h, n_embd, sh_gate_bpr, view.shared_gate_quant);
            }
            for r in 0..n_ff {
                let rb = &view.shared_up_bytes[r * sh_up_bpr..(r + 1) * sh_up_bpr];
                sh_up[r] = dot_k_block_row(rb, h, n_embd, sh_up_bpr, view.shared_up_quant);
            }
        }
        apply_model_gate_mul_inplace(&mut sh_gate, &sh_up, ModelArchitecture::Qwen35MoE);
        shared_out = vec![0f32; n_embd];
        if inner_gemv {
            gemv_generic(
                view.shared_down_bytes,
                &sh_gate,
                &mut shared_out,
                n_embd,
                n_ff,
                1,
                sh_down_bpr,
                view.shared_down_quant,
            );
        } else {
            for r in 0..n_embd {
                let rb = &view.shared_down_bytes[r * sh_down_bpr..(r + 1) * sh_down_bpr];
                shared_out[r] =
                    dot_k_block_row(rb, &sh_gate, n_ff, sh_down_bpr, view.shared_down_quant);
            }
        }
    }
    for v in shared_out.iter_mut() {
        *v *= gate_scalar;
    }
    if profile_enabled {
        record_moe_profile("qwen35moe:decode:shared_expert", shared_start.elapsed());
    }
    shared_out
}
