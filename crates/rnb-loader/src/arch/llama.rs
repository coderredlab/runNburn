use super::ModelMetadata;
use rnb_core::ir::graph::Graph;
use rnb_core::ir::op::{Attr, OpType};
use rnb_core::tensor::dtype::DType;
use std::collections::HashMap;

pub fn build_llama_graph(meta: &ModelMetadata) -> Graph {
    let mut g = Graph::new();
    let dtype = DType::F16;

    fn attrs(pairs: &[(&str, Attr)]) -> HashMap<String, Attr> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// weight placeholder 노드를 추가하고 NodeId를 반환하는 헬퍼.
    fn weight_placeholder(g: &mut Graph, weight_name: &str) -> usize {
        g.add_node(
            OpType::Placeholder,
            [("weight".to_string(), Attr::String(weight_name.to_string()))]
                .into_iter()
                .collect(),
        )
    }

    // 입력: token IDs placeholder → embedding Gather
    let token_ids = g.add_node(
        OpType::Placeholder,
        attrs(&[("input_type", Attr::String("token_ids".to_string()))]),
    );
    let emb_weight = weight_placeholder(&mut g, "token_embd.weight");
    let emb = g.add_node(OpType::Gather, HashMap::new());
    // Gather: in_port=0 → token_ids, in_port=1 → embedding table
    g.add_edge(token_ids, 0, emb, 0, dtype);
    g.add_edge(emb_weight, 0, emb, 1, dtype);

    let mut hidden = emb;

    for layer in 0..meta.num_layers {
        let prefix = format!("blk.{layer}");

        // --- Attention ---
        let attn_norm_w = weight_placeholder(&mut g, &format!("{prefix}.attn_norm.weight"));
        let attn_norm = g.add_node(
            OpType::RMSNorm,
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
            attrs(&[
                ("theta", Attr::Float(meta.rope_theta as f64)),
                ("head_dim", Attr::Int(meta.head_dim as i64)),
            ]),
        );
        g.add_edge(q_proj, 0, q_rope, 0, dtype);

        let k_rope = g.add_node(
            OpType::RoPE,
            attrs(&[
                ("theta", Attr::Float(meta.rope_theta as f64)),
                ("head_dim", Attr::Int(meta.head_dim as i64)),
            ]),
        );
        g.add_edge(k_proj, 0, k_rope, 0, dtype);

        // fused scaled dot-product attention (q_rope, k_rope, v_proj → attn_out)
        let attn_out = g.add_node(
            OpType::Attention,
            attrs(&[
                ("num_heads", Attr::Int(meta.num_heads as i64)),
                ("num_kv_heads", Attr::Int(meta.num_kv_heads as i64)),
                ("head_dim", Attr::Int(meta.head_dim as i64)),
            ]),
        );
        g.add_edge(q_rope, 0, attn_out, 0, dtype);
        g.add_edge(k_rope, 0, attn_out, 1, dtype);
        g.add_edge(v_proj, 0, attn_out, 2, dtype);

        let o_w = weight_placeholder(&mut g, &format!("{prefix}.attn_output.weight"));
        let o_proj = g.add_node(OpType::MatMul, HashMap::new());
        g.add_edge(attn_out, 0, o_proj, 0, dtype);
        g.add_edge(o_w, 0, o_proj, 1, dtype);

        let residual1 = g.add_node(OpType::Add, HashMap::new());
        g.add_edge(hidden, 0, residual1, 0, dtype);
        g.add_edge(o_proj, 0, residual1, 1, dtype);

        // --- FFN (SwiGLU) ---
        let ffn_norm_w = weight_placeholder(&mut g, &format!("{prefix}.ffn_norm.weight"));
        let ffn_norm = g.add_node(
            OpType::RMSNorm,
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

        let silu = g.add_node(OpType::SiLU, HashMap::new());
        g.add_edge(gate_proj, 0, silu, 0, dtype);

        let gate_up = g.add_node(OpType::Mul, HashMap::new());
        g.add_edge(silu, 0, gate_up, 0, dtype);
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

    // Final norm + lm_head
    let final_norm_w = weight_placeholder(&mut g, "output_norm.weight");
    let final_norm = g.add_node(
        OpType::RMSNorm,
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

    fn mini_llama_meta(num_layers: usize) -> ModelMetadata {
        ModelMetadata {
            architecture: Architecture::LLaMA,
            vocab_size: 32000,
            hidden_size: 64,
            num_layers,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            intermediate_size: 128,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_theta_swa: 10000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-5,
            final_logit_softcapping: 0.0,
            query_pre_attn_scalar: 16.0,
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
            tokenizer: TokenizerData::placeholder(32000),
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: 0,
            layer_kinds: vec![ModelLayerKind::Attention; num_layers],
            mtp: None,
            assistant: None,
        }
    }

    #[test]
    fn test_llama_graph_node_count_1_layer() {
        let meta = mini_llama_meta(1);
        let g = build_llama_graph(&meta);
        // token_ids(1) + emb_weight(1) + emb/Gather(1) = 3 입력 노드
        // per layer compute: attn_norm(1) + q_proj(1) + k_proj(1) + v_proj(1)
        //   + q_rope(1) + k_rope(1) + attn_out/Attention(1)
        //   + o_proj(1) + residual1(1) + ffn_norm(1) + gate_proj(1) + up_proj(1)
        //   + silu(1) + gate_up(1) + down_proj(1) + residual2(1) = 16
        // per layer weight placeholders: attn_norm_w + q_w + k_w + v_w + o_w
        //   + ffn_norm_w + gate_w + up_w + down_w = 9
        // final: final_norm_w(1) + final_norm(1) + lm_head_w(1) + lm_head(1) = 4
        // total 1-layer: 3 + 16 + 9 + 4 = 32
        assert_eq!(g.nodes().len(), 32);
    }

    #[test]
    fn test_llama_graph_node_count_2_layers() {
        let meta = mini_llama_meta(2);
        let g = build_llama_graph(&meta);
        // 3 + (16+9)*2 + 4 = 3 + 50 + 4 = 57
        assert_eq!(g.nodes().len(), 57);
    }

    #[test]
    fn test_llama_graph_is_valid() {
        let meta = mini_llama_meta(2);
        let g = build_llama_graph(&meta);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_llama_graph_is_acyclic() {
        let meta = mini_llama_meta(2);
        let g = build_llama_graph(&meta);
        assert!(g.topological_order().is_ok());
    }

    #[test]
    fn test_llama_graph_single_output() {
        let meta = mini_llama_meta(1);
        let g = build_llama_graph(&meta);
        let outputs = g.output_nodes();
        assert_eq!(outputs.len(), 1);
    }

    #[test]
    fn test_llama_uses_rmsnorm_not_layernorm() {
        let meta = mini_llama_meta(1);
        let g = build_llama_graph(&meta);
        assert!(g.nodes().iter().any(|n| n.op == OpType::RMSNorm));
        assert!(!g.nodes().iter().any(|n| n.op == OpType::LayerNorm));
    }

    #[test]
    fn test_llama_uses_silu() {
        let meta = mini_llama_meta(1);
        let g = build_llama_graph(&meta);
        assert!(g.nodes().iter().any(|n| n.op == OpType::SiLU));
        assert!(!g.nodes().iter().any(|n| n.op == OpType::GeLU));
    }

    #[test]
    fn test_llama_uses_attention_op() {
        let meta = mini_llama_meta(1);
        let g = build_llama_graph(&meta);
        // fused Attention op을 사용하고 개별 Softmax가 없어야 함
        assert!(g.nodes().iter().any(|n| n.op == OpType::Attention));
        assert!(!g.nodes().iter().any(|n| n.op == OpType::Softmax));
    }

    #[test]
    fn test_llama_weight_placeholders_have_weight_attr() {
        let meta = mini_llama_meta(1);
        let g = build_llama_graph(&meta);
        // 모든 Placeholder 노드는 "weight" 또는 "input_type" attr를 가져야 함
        let placeholders: Vec<_> = g
            .nodes()
            .iter()
            .filter(|n| n.op == OpType::Placeholder)
            .collect();
        assert!(!placeholders.is_empty());
        for ph in &placeholders {
            assert!(
                ph.attrs.contains_key("weight") || ph.attrs.contains_key("input_type"),
                "Placeholder 노드 {} 에 weight 또는 input_type attr 없음",
                ph.id
            );
        }
    }
}
