use super::ModelMetadata;
use rnb_core::ir::graph::Graph;
use rnb_core::ir::op::{Attr, OpType};
use rnb_core::tensor::dtype::DType;
use std::collections::HashMap;

fn attrs(pairs: &[(&str, Attr)]) -> HashMap<String, Attr> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), value.clone()))
        .collect()
}

fn weight(graph: &mut Graph, name: String) -> usize {
    graph.add_node(
        OpType::Placeholder,
        attrs(&[("weight", Attr::String(name))]),
    )
}

fn unary_weighted(
    graph: &mut Graph,
    input: usize,
    weight_name: String,
    op: OpType,
    op_attrs: HashMap<String, Attr>,
    dtype: DType,
) -> usize {
    let weight = weight(graph, weight_name);
    let output = graph.add_node(op, op_attrs);
    graph.add_edge(input, 0, output, 0, dtype);
    graph.add_edge(weight, 0, output, 1, dtype);
    output
}

fn matmul(graph: &mut Graph, input: usize, weight_name: String, dtype: DType) -> usize {
    unary_weighted(
        graph,
        input,
        weight_name,
        OpType::MatMul,
        HashMap::new(),
        dtype,
    )
}

fn rms_norm(graph: &mut Graph, input: usize, weight_name: String, eps: f32, dtype: DType) -> usize {
    unary_weighted(
        graph,
        input,
        weight_name,
        OpType::RMSNorm,
        attrs(&[("eps", Attr::Float(eps as f64))]),
        dtype,
    )
}

fn add(graph: &mut Graph, left: usize, right: usize, dtype: DType) -> usize {
    let output = graph.add_node(OpType::Add, HashMap::new());
    graph.add_edge(left, 0, output, 0, dtype);
    graph.add_edge(right, 0, output, 1, dtype);
    output
}

fn mul(graph: &mut Graph, left: usize, right: usize, dtype: DType) -> usize {
    let output = graph.add_node(OpType::Mul, HashMap::new());
    graph.add_edge(left, 0, output, 0, dtype);
    graph.add_edge(right, 0, output, 1, dtype);
    output
}

pub fn build_muse_glimmer_graph(meta: &ModelMetadata) -> Graph {
    let mut graph = Graph::new();
    let dtype = DType::F16;
    let token_ids = graph.add_node(
        OpType::Placeholder,
        attrs(&[("input_type", Attr::String("token_ids".to_string()))]),
    );
    let token_weight = weight(&mut graph, "token_embd.weight".to_string());
    let embedding = graph.add_node(OpType::Gather, HashMap::new());
    graph.add_edge(token_ids, 0, embedding, 0, dtype);
    graph.add_edge(token_weight, 0, embedding, 1, dtype);

    let mut hidden = graph.add_node(
        OpType::RMSNorm,
        attrs(&[
            ("eps", Attr::Float(meta.norm_eps as f64)),
            ("unscaled", Attr::Int(1)),
        ]),
    );
    graph.add_edge(embedding, 0, hidden, 0, dtype);

    for layer in 0..meta.num_layers {
        let prefix = format!("blk.{layer}");
        let attn_input = rms_norm(
            &mut graph,
            hidden,
            format!("{prefix}.attn_norm.weight"),
            meta.norm_eps,
            dtype,
        );
        let mut q = matmul(
            &mut graph,
            attn_input,
            format!("{prefix}.attn_q.weight"),
            dtype,
        );
        let mut k = matmul(
            &mut graph,
            attn_input,
            format!("{prefix}.attn_k.weight"),
            dtype,
        );
        let v = matmul(
            &mut graph,
            attn_input,
            format!("{prefix}.attn_v.weight"),
            dtype,
        );
        q = rms_norm(
            &mut graph,
            q,
            format!("{prefix}.attn_q_norm.weight"),
            meta.norm_eps,
            dtype,
        );
        k = rms_norm(
            &mut graph,
            k,
            format!("{prefix}.attn_k_norm.weight"),
            meta.norm_eps,
            dtype,
        );

        let sliding = meta
            .sliding_window_pattern
            .get(layer)
            .copied()
            .unwrap_or(false);
        if sliding {
            let rope_attrs = attrs(&[
                ("theta", Attr::Float(meta.rope_theta as f64)),
                ("head_dim", Attr::Int(meta.head_dim as i64)),
            ]);
            let q_rope = graph.add_node(OpType::RoPE, rope_attrs.clone());
            graph.add_edge(q, 0, q_rope, 0, dtype);
            q = q_rope;
            let k_rope = graph.add_node(OpType::RoPE, rope_attrs);
            graph.add_edge(k, 0, k_rope, 0, dtype);
            k = k_rope;
        }

        let attention = graph.add_node(
            OpType::Attention,
            attrs(&[
                ("num_heads", Attr::Int(meta.num_heads as i64)),
                ("num_kv_heads", Attr::Int(meta.num_kv_heads as i64)),
                ("head_dim", Attr::Int(meta.head_dim as i64)),
                (
                    "sliding_window",
                    Attr::Int(if sliding {
                        meta.sliding_window as i64
                    } else {
                        0
                    }),
                ),
            ]),
        );
        graph.add_edge(q, 0, attention, 0, dtype);
        graph.add_edge(k, 0, attention, 1, dtype);
        graph.add_edge(v, 0, attention, 2, dtype);

        let gate = matmul(
            &mut graph,
            attn_input,
            format!("{prefix}.attn_gate.weight"),
            dtype,
        );
        let sigmoid = graph.add_node(OpType::Custom("Sigmoid".to_string()), HashMap::new());
        graph.add_edge(gate, 0, sigmoid, 0, dtype);
        let gated_attention = mul(&mut graph, attention, sigmoid, dtype);
        let attention_output = matmul(
            &mut graph,
            gated_attention,
            format!("{prefix}.attn_output.weight"),
            dtype,
        );
        let attention_output = rms_norm(
            &mut graph,
            attention_output,
            format!("{prefix}.post_attention_norm.weight"),
            meta.post_norm_eps,
            dtype,
        );
        let ffn_residual = add(&mut graph, hidden, attention_output, dtype);

        let ffn_input = rms_norm(
            &mut graph,
            ffn_residual,
            format!("{prefix}.ffn_norm.weight"),
            meta.norm_eps,
            dtype,
        );
        let gate = matmul(
            &mut graph,
            ffn_input,
            format!("{prefix}.ffn_gate.weight"),
            dtype,
        );
        let up = matmul(
            &mut graph,
            ffn_input,
            format!("{prefix}.ffn_up.weight"),
            dtype,
        );
        let silu = graph.add_node(OpType::SiLU, HashMap::new());
        graph.add_edge(gate, 0, silu, 0, dtype);
        let gated = mul(&mut graph, silu, up, dtype);
        let down = matmul(
            &mut graph,
            gated,
            format!("{prefix}.ffn_down.weight"),
            dtype,
        );
        let down = rms_norm(
            &mut graph,
            down,
            format!("{prefix}.post_ffw_norm.weight"),
            meta.post_norm_eps,
            dtype,
        );
        hidden = add(&mut graph, ffn_residual, down, dtype);
    }

    let normalized = rms_norm(
        &mut graph,
        hidden,
        "output_norm.weight".to_string(),
        meta.norm_eps,
        dtype,
    );
    let logits = matmul(&mut graph, normalized, "output.weight".to_string(), dtype);
    let scaled = graph.add_node(
        OpType::Custom("Scale".to_string()),
        attrs(&[("value", Attr::Float(meta.logit_scale as f64))]),
    );
    graph.add_edge(logits, 0, scaled, 0, dtype);
    let softcap = graph.add_node(
        OpType::Custom("TanhSoftcap".to_string()),
        attrs(&[("value", Attr::Float(meta.final_logit_softcapping as f64))]),
    );
    graph.add_edge(scaled, 0, softcap, 0, dtype);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{Architecture, ModelLayerKind};
    use crate::TokenizerData;

    fn mini_metadata() -> ModelMetadata {
        ModelMetadata {
            architecture: Architecture::MuseGlimmer,
            vocab_size: 64,
            hidden_size: 16,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            max_seq_len: 128,
            rope_theta: 500000.0,
            rope_theta_swa: 500000.0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_sections: [0; 4],
            norm_eps: 1e-5,
            final_logit_softcapping: 20.0,
            query_pre_attn_scalar: 8.0,
            post_norm_eps: 1e-8,
            logit_scale: 0.19611613,
            sliding_window: 32,
            shared_kv_layers: 0,
            sliding_window_pattern: vec![true, true, true, false],
            key_length_full: 8,
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
            tokenizer: TokenizerData::placeholder(64),
            ssm_d_inner: 0,
            ssm_d_state: 0,
            ssm_n_group: 0,
            ssm_dt_rank: 0,
            ssm_conv_kernel: 0,
            full_attention_interval: 0,
            layer_kinds: vec![ModelLayerKind::Attention; 4],
            mtp: None,
            assistant: None,
            glm_indexer: None,
            deepseek4: None,
        }
    }

    #[test]
    fn graph_encodes_local_rope_and_separate_attention_gate() {
        let graph = build_muse_glimmer_graph(&mini_metadata());

        assert!(graph.validate().is_ok());
        assert!(graph.topological_order().is_ok());
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| node.op == OpType::RoPE)
                .count(),
            6
        );
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| node.op == OpType::Custom("Sigmoid".to_string()))
                .count(),
            4
        );
        for layer in 0..4 {
            let gate_name = format!("blk.{layer}.attn_gate.weight");
            assert!(graph.nodes().iter().any(|node| {
                node.attrs.get("weight") == Some(&Attr::String(gate_name.clone()))
            }));
        }
    }

    #[test]
    fn graph_applies_logit_scale_before_softcap() {
        let graph = build_muse_glimmer_graph(&mini_metadata());
        let scale = graph
            .nodes()
            .iter()
            .find(|node| node.op == OpType::Custom("Scale".to_string()))
            .unwrap();
        let softcap = graph
            .nodes()
            .iter()
            .find(|node| node.op == OpType::Custom("TanhSoftcap".to_string()))
            .unwrap();

        assert_eq!(
            scale.attrs.get("value"),
            Some(&Attr::Float(0.19611613f32 as f64))
        );
        assert_eq!(softcap.attrs.get("value"), Some(&Attr::Float(20.0)));
        assert!(graph
            .outputs_of(scale.id)
            .iter()
            .any(|edge| edge.to.0 == softcap.id));
        assert_eq!(graph.output_nodes(), vec![softcap.id]);
    }
}
