use super::ModelMetadata;
use rnb_core::ir::graph::Graph;
use rnb_core::ir::op::{Attr, OpType};
use rnb_core::tensor::dtype::DType;
use std::collections::HashMap;

pub fn build_deepseek4_graph(meta: &ModelMetadata) -> Graph {
    build_graph(meta.num_layers)
}

fn build_graph(num_layers: usize) -> Graph {
    let mut graph = Graph::new();
    let token_ids = graph.add_node(
        OpType::Placeholder,
        [(
            "input_type".to_string(),
            Attr::String("token_ids".to_string()),
        )]
        .into_iter()
        .collect(),
    );
    let forward = graph.add_node(
        OpType::Custom("DeepSeek4Forward".to_string()),
        [
            ("num_layers".to_string(), Attr::Int(num_layers as i64)),
            (
                "state_model".to_string(),
                Attr::String("compressed_attention".to_string()),
            ),
        ]
        .into_iter()
        .collect::<HashMap<_, _>>(),
    );
    graph.add_edge(token_ids, 0, forward, 0, DType::F16);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek4_graph_is_an_explicit_opaque_runtime_node() {
        let graph = build_graph(43);

        assert!(graph.validate().is_ok());
        assert_eq!(graph.nodes().len(), 2);
        assert!(matches!(
            &graph.nodes()[1].op,
            OpType::Custom(name) if name == "DeepSeek4Forward"
        ));
        assert_eq!(
            graph.nodes()[1].attrs.get("num_layers"),
            Some(&Attr::Int(43))
        );
    }
}
