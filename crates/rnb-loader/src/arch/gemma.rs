use super::ModelMetadata;
use rnb_core::ir::graph::Graph;
use rnb_core::ir::op::{Attr, OpType};
use rnb_core::tensor::dtype::DType;
use std::collections::HashMap;

/// Gemma 아키텍처 그래프 빌더.
/// LLaMA와 유사하지만 LayerNorm + GeLU(GeGLU) 사용.
pub fn build_gemma_graph(meta: &ModelMetadata) -> Graph {
    let mut g = Graph::new();
    let dtype = DType::F16;

    fn attrs(pairs: &[(&str, Attr)]) -> HashMap<String, Attr> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn weight_placeholder(g: &mut Graph, weight_name: &str) -> usize {
        g.add_node(
            OpType::Placeholder,
            [("weight".to_string(), Attr::String(weight_name.to_string()))]
                .into_iter()
                .collect(),
        )
    }

    let token_ids = g.add_node(
        OpType::Placeholder,
        attrs(&[("input_type", Attr::String("token_ids".to_string()))]),
    );
    let emb_weight = weight_placeholder(&mut g, "token_embd.weight");
    let emb = g.add_node(OpType::Gather, HashMap::new());
    g.add_edge(token_ids, 0, emb, 0, dtype);
    g.add_edge(emb_weight, 0, emb, 1, dtype);

    let mut hidden = emb;

    for layer in 0..meta.num_layers {
        let prefix = format!("blk.{layer}");

        // --- Attention (LayerNorm) ---
        let attn_norm_w = weight_placeholder(&mut g, &format!("{prefix}.attn_norm.weight"));
        let attn_norm = g.add_node(
            OpType::LayerNorm,
            attrs(&[("eps", Attr::Float(meta.norm_eps as f64))]),
        );
        g.add_edge(hidden, 0, attn_norm, 0, dtype);
        g.add_edge(attn_norm_w, 0, attn_norm, 1, dtype);

        let q_w = weight_placeholder(&mut g, &format!("{prefix}.attn_q.weight"));
        let q_proj = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(attn_norm, 0, q_proj, 0, dtype);
        g.add_edge(q_w, 0, q_proj, 1, dtype);

        let k_w = weight_placeholder(&mut g, &format!("{prefix}.attn_k.weight"));
        let k_proj = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(attn_norm, 0, k_proj, 0, dtype);
        g.add_edge(k_w, 0, k_proj, 1, dtype);

        let v_w = weight_placeholder(&mut g, &format!("{prefix}.attn_v.weight"));
        let v_proj = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(attn_norm, 0, v_proj, 0, dtype);
        g.add_edge(v_w, 0, v_proj, 1, dtype);

        let q_rope = g.add_node(
            OpType::RoPE,
            attrs(&[("theta", Attr::Float(meta.rope_theta as f64))]),
        );
        g.add_edge(q_proj, 0, q_rope, 0, dtype);

        let k_rope = g.add_node(
            OpType::RoPE,
            attrs(&[("theta", Attr::Float(meta.rope_theta as f64))]),
        );
        g.add_edge(k_proj, 0, k_rope, 0, dtype);

        let attn_score = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(q_rope, 0, attn_score, 0, dtype);
        g.add_edge(k_rope, 0, attn_score, 1, dtype);

        let softmax = g.add_node(
            OpType::Softmax,
            attrs(&[("scale", Attr::Float(1.0 / (meta.head_dim as f64).sqrt()))]),
        );
        g.add_edge(attn_score, 0, softmax, 0, dtype);

        let attn_out = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(softmax, 0, attn_out, 0, dtype);
        g.add_edge(v_proj, 0, attn_out, 1, dtype);

        let o_w = weight_placeholder(&mut g, &format!("{prefix}.attn_output.weight"));
        let o_proj = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(attn_out, 0, o_proj, 0, dtype);
        g.add_edge(o_w, 0, o_proj, 1, dtype);

        let residual1 = g.add_node(OpType::Add, HashMap::new());
        g.add_edge(hidden, 0, residual1, 0, dtype);
        g.add_edge(o_proj, 0, residual1, 1, dtype);

        // --- FFN (GeGLU: GeLU gating, LayerNorm) ---
        let ffn_norm_w = weight_placeholder(&mut g, &format!("{prefix}.ffn_norm.weight"));
        let ffn_norm = g.add_node(
            OpType::LayerNorm,
            attrs(&[("eps", Attr::Float(meta.norm_eps as f64))]),
        );
        g.add_edge(residual1, 0, ffn_norm, 0, dtype);
        g.add_edge(ffn_norm_w, 0, ffn_norm, 1, dtype);

        let gate_w = weight_placeholder(&mut g, &format!("{prefix}.ffn_gate.weight"));
        let gate_proj = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(ffn_norm, 0, gate_proj, 0, dtype);
        g.add_edge(gate_w, 0, gate_proj, 1, dtype);

        let up_w = weight_placeholder(&mut g, &format!("{prefix}.ffn_up.weight"));
        let up_proj = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(ffn_norm, 0, up_proj, 0, dtype);
        g.add_edge(up_w, 0, up_proj, 1, dtype);

        let gelu = g.add_node(OpType::GeLU, HashMap::new());
        g.add_edge(gate_proj, 0, gelu, 0, dtype);

        let gate_up = g.add_node(OpType::Mul, HashMap::new());
        g.add_edge(gelu, 0, gate_up, 0, dtype);
        g.add_edge(up_proj, 0, gate_up, 1, dtype);

        let down_w = weight_placeholder(&mut g, &format!("{prefix}.ffn_down.weight"));
        let down_proj = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(gate_up, 0, down_proj, 0, dtype);
        g.add_edge(down_w, 0, down_proj, 1, dtype);

        let residual2 = g.add_node(OpType::Add, HashMap::new());
        g.add_edge(residual1, 0, residual2, 0, dtype);
        g.add_edge(down_proj, 0, residual2, 1, dtype);

        hidden = residual2;
    }

    let final_norm_w = weight_placeholder(&mut g, "output_norm.weight");
    let final_norm = g.add_node(
        OpType::LayerNorm,
        attrs(&[("eps", Attr::Float(meta.norm_eps as f64))]),
    );
    g.add_edge(hidden, 0, final_norm, 0, dtype);
    g.add_edge(final_norm_w, 0, final_norm, 1, dtype);

    let lm_head_w = weight_placeholder(&mut g, "output.weight");
    let lm_head = g.add_node(OpType::MatMul, HashMap::new());
    g.add_edge(final_norm, 0, lm_head, 0, dtype);
    g.add_edge(lm_head_w, 0, lm_head, 1, dtype);

    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{Architecture, ModelLayerKind, ModelMetadata};
    use crate::TokenizerData;

    fn mini_gemma_meta() -> ModelMetadata {
        ModelMetadata {
            architecture: Architecture::Gemma,
            vocab_size: 256000,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 16,
            intermediate_size: 128,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_theta_swa: 10000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-6,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 256.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            sliding_window_pattern: vec![],
            key_length_full: 0,
            key_length_swa: 0,
            value_length_swa: 0,
            embedding_length_per_layer_input: 0,
            expert_count: 0,
            expert_used_count: 0,
            expert_shared_count: 0,
            leading_dense_block_count: 0,
            expert_gating_func: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            expert_feed_forward_length: 0,
            head_count_kv_per_layer: None,
            tokenizer: TokenizerData::placeholder(256000),
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: 0,
            layer_kinds: vec![ModelLayerKind::Attention; 2],
            mtp: None,
            assistant: None,
        }
    }

    #[test]
    fn test_gemma_graph_is_valid() {
        let g = build_gemma_graph(&mini_gemma_meta());
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_gemma_graph_acyclic() {
        let g = build_gemma_graph(&mini_gemma_meta());
        assert!(g.topological_order().is_ok());
    }

    #[test]
    fn test_gemma_graph_single_output() {
        let g = build_gemma_graph(&mini_gemma_meta());
        assert_eq!(g.output_nodes().len(), 1);
    }

    #[test]
    fn test_gemma_uses_layernorm() {
        let g = build_gemma_graph(&mini_gemma_meta());
        // per layer: attn_norm + ffn_norm = 2, final_norm = 1 → 2*2+1 = 5
        let layernorm_count = g
            .nodes()
            .iter()
            .filter(|n| n.op == OpType::LayerNorm)
            .count();
        assert_eq!(layernorm_count, 5);
    }

    #[test]
    fn test_gemma_uses_gelu_not_silu() {
        let g = build_gemma_graph(&mini_gemma_meta());
        assert!(g.nodes().iter().any(|n| n.op == OpType::GeLU));
        assert!(!g.nodes().iter().any(|n| n.op == OpType::SiLU));
    }

    #[test]
    fn test_gemma_no_rmsnorm() {
        let g = build_gemma_graph(&mini_gemma_meta());
        assert!(!g.nodes().iter().any(|n| n.op == OpType::RMSNorm));
    }
}
