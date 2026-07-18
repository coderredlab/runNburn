//! Qwen MoE shared expert decode path.

use super::*;

pub(in crate::engine) fn compute_shared_expert(
    view: &SharedExpertMoEView<'_>,
    h: &[f32],
    gate_scalar: f32,
    shared_in_sparse_gpu: bool,
    profile_enabled: bool,
    prefer_sparse_moe_cuda: bool,
) -> Vec<f32> {
    let shared_start = Instant::now();
    let n_ff = view.n_ff;
    let n_embd = view.n_embd;

    let sh_gate_bpr = expert_bytes_per_row(n_embd, view.shared_gate_quant, "shared_gate");
    let sh_up_bpr = expert_bytes_per_row(n_embd, view.shared_up_quant, "shared_up");
    let sh_down_bpr = expert_bytes_per_row(n_ff, view.shared_down_quant, "shared_down");
    let q2q3_matrix = view.shared_gate_quant == GGMLType::Q2_K
        && view.shared_up_quant == GGMLType::Q2_K
        && view.shared_down_quant == GGMLType::Q3_K;
    let glm_cuda_supported = prefer_sparse_moe_cuda
        && view.shared_gate_quant == GGMLType::Q5_K
        && view.shared_up_quant == GGMLType::Q5_K
        && view.shared_down_quant == GGMLType::Q6_K;
    let mut shared_out = if glm_cuda_supported {
        match qwen_moe_backend::glm_moe_decode_shared_expert_q5k_q6k(
            view.shared_gate_bytes,
            view.shared_up_bytes,
            view.shared_down_bytes,
            n_ff,
            n_embd,
            h,
        ) {
            Ok(out) => out,
            Err(err) => {
                eprintln!("[WARN] CUDA GLM shared expert failed, using CPU path: {err}");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
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
        if q2q3_matrix {
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
        if q2q3_matrix {
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
