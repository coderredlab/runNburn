//! Diagnostic verification for attention decode GPU projections.

use super::*;

pub(super) fn log_decode_attention_gpu_debug(
    layer_idx: usize,
    gpu_runtime_present: bool,
    w: &AttentionLayerWeights,
) {
    eprintln!(
        "[GPU DEBUG] ATN L{}: gpu_runtime={}, q={:?} k={:?} v={:?} o={:?} ffn_gate={:?}",
        layer_idx,
        gpu_runtime_present,
        w.q_weight.ggml_type,
        w.k_weight.ggml_type,
        w.v_weight.ggml_type,
        w.o_weight.ggml_type,
        w.ffn_gate_weight.ggml_type
    );
}

pub(super) fn verify_decode_attention_qkv(
    layer_idx: usize,
    scratch: &ScratchBuffers,
    w: &AttentionLayerWeights,
    hidden_dim: usize,
    q_out_dim: usize,
    kv_dim: usize,
) {
    let mut cpu_q = vec![0.0f32; q_out_dim];
    w.q_weight
        .gemv_into(&scratch.norm_buf[..hidden_dim], &mut cpu_q)
        .ok();
    let q_max = scratch.q_buf[..q_out_dim]
        .iter()
        .zip(&cpu_q)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mut cpu_k = vec![0.0f32; kv_dim];
    w.k_weight
        .gemv_into(&scratch.norm_buf[..hidden_dim], &mut cpu_k)
        .ok();
    let k_max = scratch.k_buf[..kv_dim]
        .iter()
        .zip(&cpu_k)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mut cpu_v = vec![0.0f32; kv_dim];
    w.v_weight
        .gemv_into(&scratch.norm_buf[..hidden_dim], &mut cpu_v)
        .ok();
    let v_max = scratch.v_buf[..kv_dim]
        .iter()
        .zip(&cpu_v)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let v_err_big = scratch.v_buf[..kv_dim]
        .iter()
        .zip(cpu_v.iter())
        .filter(|&(a, b)| (a - b).abs() > 0.1)
        .count();

    eprintln!(
        "[GPU VERIFY] L{} Q({:?})max={:.4} K({:?})max={:.4} V({:?})max={:.4} V>0.1:{}",
        layer_idx,
        w.q_weight.ggml_type,
        q_max,
        w.k_weight.ggml_type,
        k_max,
        w.v_weight.ggml_type,
        v_max,
        v_err_big
    );

    if v_max > 0.5 {
        let v_bytes = w.v_weight.data.as_bytes().unwrap_or(&[]);
        let v_bpr = v_bytes.len() / kv_dim;
        if v_bpr >= 210 {
            let d_raw = v_bytes[208] as u16 | ((v_bytes[209] as u16) << 8);
            let d_val = half::f16::from_bits(d_raw).to_f32();
            let sc: Vec<i8> = (0..16).map(|i| v_bytes[192 + i] as i8).collect();
            eprintln!(
                "[GPU VERIFY] V row0 blk0: d=0x{:04x}({:.8}), bpr={}, scales={:?}",
                d_raw, d_val, v_bpr, sc
            );
        }
        for i in 0..4.min(kv_dim) {
            eprintln!(
                "[GPU VERIFY] V[{}]: gpu={:.6}, cpu={:.6}",
                i, scratch.v_buf[i], cpu_v[i]
            );
        }
    }
}
