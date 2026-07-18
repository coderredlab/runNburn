//! 백엔드 런타임 이름 없이 유지하는 GDN prefill 헬퍼.

use crate::engine::backend_runtime;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use crate::engine::layer_weights::GdnLayerWeights;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use crate::engine::metal_runtime;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use crate::kv_cache::SsmLayerState;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use rnb_loader::GGMLType;

pub(in crate::engine) fn make_gdn_prefill_chain_shape(
    seq_len: usize,
    hidden_dim: usize,
    d_inner: usize,
    d_state: usize,
    n_group: usize,
    dt_rank: usize,
    conv_kernel: usize,
    conv_state_len: usize,
    delta_state_len: usize,
) -> backend_runtime::GdnPrefillChainShape {
    backend_runtime::GdnPrefillChainShape {
        seq_len,
        hidden_dim,
        d_inner,
        d_state,
        n_group,
        dt_rank,
        conv_kernel,
        conv_state_len,
        delta_state_len,
    }
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn metal_qwen_prefill_gdn_weight(
    weight: &crate::engine::quantized_weight_types::QuantizedWeight,
) -> Option<metal_runtime::MetalGdnWeightRef<'_>> {
    if weight.ggml_type == GGMLType::F32 {
        if weight.data.dtype() != rnb_core::tensor::DType::F32 {
            return None;
        }
        return Some(metal_runtime::MetalGdnWeightRef::F32 {
            values: crate::engine::kernels::tensor_as_f32_slice(&weight.data),
            rows: weight.rows,
            cols: weight.cols,
        });
    }
    Some(metal_runtime::MetalGdnWeightRef::Quant(
        metal_runtime::MetalQuantWeightRef {
            ggml_type: weight.ggml_type,
            raw: weight.data.as_bytes()?,
            rows: weight.rows,
            cols: weight.cols,
        },
    ))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[allow(clippy::too_many_arguments)]
pub(in crate::engine) fn metal_qwen_prefill_gdn_spec<'a>(
    w: &'a GdnLayerWeights,
    state: &'a SsmLayerState,
    seq_len: usize,
    hidden_dim: usize,
    d_inner: usize,
    d_state: usize,
    n_group: usize,
    dt_rank: usize,
    conv_kernel_size: usize,
    norm_eps: f32,
) -> Option<metal_runtime::MetalQwenPrefillGdnSpec<'a>> {
    Some(metal_runtime::MetalQwenPrefillGdnSpec {
        seq_len,
        hidden_dim,
        d_inner,
        d_state,
        n_group,
        dt_rank,
        conv_kernel_size,
        attn_norm_w: crate::engine::kernels::tensor_as_f32_slice(&w.attn_norm),
        qkv_weight: metal_qwen_prefill_gdn_weight(&w.qkv_weight)?,
        gate_weight: metal_qwen_prefill_gdn_weight(&w.gate_weight)?,
        alpha_weight: metal_qwen_prefill_gdn_weight(&w.ssm_alpha)?,
        beta_weight: metal_qwen_prefill_gdn_weight(&w.ssm_beta)?,
        conv_state: &state.conv_state,
        conv_kernel: crate::engine::kernels::tensor_as_f32_slice(&w.ssm_conv1d),
        dt_bias: crate::engine::kernels::tensor_as_f32_slice(&w.ssm_dt_bias),
        ssm_a: crate::engine::kernels::tensor_as_f32_slice(&w.ssm_a),
        delta_state: &state.delta_state,
        ssm_norm: crate::engine::kernels::tensor_as_f32_slice(&w.ssm_norm),
        ssm_out_weight: metal_qwen_prefill_gdn_weight(&w.ssm_out)?,
        post_attn_norm_w: crate::engine::kernels::tensor_as_f32_slice(&w.post_attn_norm),
        norm_eps,
    })
}

pub(in crate::engine) fn apply_dt_gate_inplace(
    gate_data: &mut [f32],
    dt_bias: &[f32],
    ssm_a: &[f32],
    seq_len: usize,
    num_v_heads: usize,
) {
    for t in 0..seq_len {
        for h in 0..num_v_heads {
            let idx = t * num_v_heads + h;
            let a_biased = gate_data[idx] + dt_bias[h];
            let sp = (1.0 + a_biased.exp()).ln();
            gate_data[idx] = sp * ssm_a[h];
        }
    }
}

pub(in crate::engine) fn build_conv_input_and_advance_state(
    conv_state: &mut [f32],
    qkv_data: &[f32],
    seq_len: usize,
    conv_channels: usize,
    conv_kernel: usize,
) -> Vec<f32> {
    let conv_state_len = (conv_kernel - 1) * conv_channels;
    let total_conv_len = (conv_kernel - 1) + seq_len;
    let mut conv_input = Vec::with_capacity(total_conv_len * conv_channels);
    conv_input.extend_from_slice(conv_state);
    conv_input.extend_from_slice(qkv_data);

    let new_state_start = seq_len * conv_channels;
    conv_state.copy_from_slice(&conv_input[new_state_start..new_state_start + conv_state_len]);
    conv_input
}

pub(in crate::engine) fn conv_state_after_prefix_tokens(
    conv_input: &[f32],
    prefix_tokens: usize,
    conv_channels: usize,
    conv_kernel: usize,
) -> Vec<f32> {
    let conv_state_len = (conv_kernel - 1) * conv_channels;
    let start = prefix_tokens * conv_channels;
    conv_input[start..start + conv_state_len].to_vec()
}

pub(in crate::engine) fn split_conv_qkv(
    conv_data: &[f32],
    seq_len: usize,
    conv_channels: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q_data = vec![0.0f32; seq_len * q_dim];
    let mut k_data = vec![0.0f32; seq_len * k_dim];
    let mut v_data = vec![0.0f32; seq_len * v_dim];

    for t in 0..seq_len {
        let off = t * conv_channels;
        q_data[t * q_dim..(t + 1) * q_dim].copy_from_slice(&conv_data[off..off + q_dim]);
        k_data[t * k_dim..(t + 1) * k_dim]
            .copy_from_slice(&conv_data[off + q_dim..off + q_dim + k_dim]);
        v_data[t * v_dim..(t + 1) * v_dim]
            .copy_from_slice(&conv_data[off + q_dim + k_dim..off + q_dim + k_dim + v_dim]);
    }

    (q_data, k_data, v_data)
}

pub(in crate::engine) fn repeat_qk_for_value_heads(
    q_raw: &[f32],
    k_raw: &[f32],
    seq_len: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    if num_v_heads == num_k_heads {
        return (q_raw.to_vec(), k_raw.to_vec());
    }

    assert!(
        num_v_heads % num_k_heads == 0,
        "gdn GQA: num_v_heads ({}) must be multiple of num_k_heads ({})",
        num_v_heads,
        num_k_heads
    );

    // ggml_repeat_4d 의미: 교차 복제가 아니라 순환 브로드캐스트다.
    let mut q_rep = Vec::with_capacity(seq_len * num_v_heads * head_k_dim);
    let mut k_rep = Vec::with_capacity(seq_len * num_v_heads * head_k_dim);
    for t in 0..seq_len {
        for vh in 0..num_v_heads {
            let kh = vh % num_k_heads;
            let src = (t * num_k_heads + kh) * head_k_dim;
            q_rep.extend_from_slice(&q_raw[src..src + head_k_dim]);
            k_rep.extend_from_slice(&k_raw[src..src + head_k_dim]);
        }
    }
    (q_rep, k_rep)
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct GdnPrefillReferenceSnapshot {
    qkv_len: usize,
    gate_len: usize,
    delta_out_len: usize,
    final_len: usize,
    qkv_sum: f32,
    gate_sum: f32,
    delta_out_sum: f32,
    final_sum: f32,
}

#[cfg(test)]
fn stable_sum(values: &[f32]) -> f32 {
    values
        .iter()
        .copied()
        .fold(0.0f32, |acc, value| acc + value)
}

#[cfg(test)]
fn make_gdn_prefill_reference_snapshot(
    qkv: &[f32],
    gate: &[f32],
    delta_out: &[f32],
    final_out: &[f32],
) -> GdnPrefillReferenceSnapshot {
    GdnPrefillReferenceSnapshot {
        qkv_len: qkv.len(),
        gate_len: gate.len(),
        delta_out_len: delta_out.len(),
        final_len: final_out.len(),
        qkv_sum: stable_sum(qkv),
        gate_sum: stable_sum(gate),
        delta_out_sum: stable_sum(delta_out),
        final_sum: stable_sum(final_out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_gdn_prefill_chain_shape_uses_runtime_state_lengths() {
        let shape = make_gdn_prefill_chain_shape(
            32,
            2048,
            4096,
            128,
            16,
            32,
            4,
            3 * (4096 + 2 * 16 * 128),
            4096 * 128,
        );

        assert_eq!(shape.seq_len, 32);
        assert_eq!(shape.hidden_dim, 2048);
        assert_eq!(shape.d_inner, 4096);
        assert_eq!(shape.d_state, 128);
        assert_eq!(shape.n_group, 16);
        assert_eq!(shape.dt_rank, 32);
        assert_eq!(shape.conv_kernel, 4);
        assert_eq!(shape.conv_state_len, 3 * (4096 + 2 * 16 * 128));
        assert_eq!(shape.delta_state_len, 4096 * 128);
    }

    #[test]
    fn gdn_prefill_reference_snapshot_records_lengths_and_sums() {
        let snapshot = make_gdn_prefill_reference_snapshot(
            &[1.0, 2.0, -0.5],
            &[0.25, 0.75],
            &[3.0, -1.0],
            &[4.0, 5.0],
        );

        assert_eq!(snapshot.qkv_len, 3);
        assert_eq!(snapshot.gate_len, 2);
        assert_eq!(snapshot.delta_out_len, 2);
        assert_eq!(snapshot.final_len, 2);
        assert!((snapshot.qkv_sum - 2.5).abs() < 0.0001);
        assert!((snapshot.gate_sum - 1.0).abs() < 0.0001);
        assert!((snapshot.delta_out_sum - 2.0).abs() < 0.0001);
        assert!((snapshot.final_sum - 9.0).abs() < 0.0001);
    }

    #[test]
    fn repeat_qk_for_value_heads_uses_cyclic_head_broadcast() {
        let q = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0];
        let k = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let (q_rep, k_rep) = repeat_qk_for_value_heads(&q, &k, 1, 4, 8, 2);

        assert_eq!(
            q_rep,
            vec![
                10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0,
                40.0, 41.0
            ]
        );
        assert_eq!(
            k_rep,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn build_conv_input_and_advance_state_appends_qkv_and_keeps_suffix() {
        let mut state = vec![1.0, 2.0, 3.0, 4.0];
        let qkv = vec![10.0, 11.0, 20.0, 21.0];

        let input = build_conv_input_and_advance_state(&mut state, &qkv, 2, 2, 3);

        assert_eq!(input, vec![1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 20.0, 21.0]);
        assert_eq!(state, vec![10.0, 11.0, 20.0, 21.0]);
    }

    #[test]
    fn conv_state_after_prefix_tokens_uses_prefix_suffix() {
        let mut state = vec![1.0, 2.0, 3.0, 4.0];
        let qkv = vec![10.0, 11.0, 20.0, 21.0];
        let input = build_conv_input_and_advance_state(&mut state, &qkv, 2, 2, 3);

        let prefix_state = conv_state_after_prefix_tokens(&input, 1, 2, 3);

        assert_eq!(prefix_state, vec![3.0, 4.0, 10.0, 11.0]);
    }

    #[test]
    fn apply_dt_gate_inplace_matches_softplus_times_ssm_a() {
        let mut gate = vec![0.0, 1.0, -1.0, 2.0];
        let dt_bias = vec![0.5, -0.5];
        let ssm_a = vec![-2.0, -3.0];

        apply_dt_gate_inplace(&mut gate, &dt_bias, &ssm_a, 2, 2);

        let expected = [
            -2.0 * (1.0_f32 + 0.5_f32.exp()).ln(),
            -3.0 * (1.0_f32 + 0.5_f32.exp()).ln(),
            -2.0 * (1.0_f32 + (-0.5_f32).exp()).ln(),
            -3.0 * (1.0_f32 + 1.5_f32.exp()).ln(),
        ];
        for (got, expected) in gate.iter().zip(expected) {
            assert!((got - expected).abs() < 0.0001);
        }
    }

    #[test]
    fn split_conv_qkv_preserves_token_order() {
        let conv = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        ];

        let (q, k, v) = split_conv_qkv(&conv, 2, 6, 2, 1, 3);

        assert_eq!(q, vec![1.0, 2.0, 10.0, 20.0]);
        assert_eq!(k, vec![3.0, 30.0]);
        assert_eq!(v, vec![4.0, 5.0, 6.0, 40.0, 50.0, 60.0]);
    }
}
