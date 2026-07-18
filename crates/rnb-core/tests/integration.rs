//! 통합 테스트: 그래프 빌드 → 스케줄러 실행 전체 파이프라인.

use std::collections::{HashMap, HashSet};

use rnb_core::backend::{Backend, Scheduler};
use rnb_core::error::Result;
use rnb_core::ir::{Graph, Node, OpType, Pass};
use rnb_core::tensor::{DType, Tensor};

struct DummyCpuBackend {
    ops: HashSet<OpType>,
}

impl DummyCpuBackend {
    fn new() -> Self {
        let mut ops = HashSet::new();
        ops.insert(OpType::MatMul);
        ops.insert(OpType::Add);
        ops.insert(OpType::Reshape);
        ops.insert(OpType::RMSNorm);
        ops.insert(OpType::SiLU);
        Self { ops }
    }
}

impl Backend for DummyCpuBackend {
    fn name(&self) -> &str {
        "dummy-cpu"
    }

    fn supported_ops(&self) -> &HashSet<OpType> {
        &self.ops
    }

    fn supports_dtype(&self, _dtype: DType) -> bool {
        true
    }

    fn execute(&self, _node: &Node, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        if let Some(first) = inputs.first() {
            Ok(vec![Tensor::zeros(first.shape(), first.dtype())])
        } else {
            Ok(vec![Tensor::zeros(&[1], DType::F32)])
        }
    }
}

#[test]
fn test_linear_layer_pipeline() {
    let mut graph = Graph::new();
    let input = graph.add_node(OpType::Reshape, Default::default());
    let weights = graph.add_node(OpType::Reshape, Default::default());
    let bias = graph.add_node(OpType::Reshape, Default::default());
    let matmul = graph.add_node(OpType::MatMul, Default::default());
    let add = graph.add_node(OpType::Add, Default::default());

    graph.add_edge(input, 0, matmul, 0, DType::F32);
    graph.add_edge(weights, 0, matmul, 1, DType::F32);
    graph.add_edge(matmul, 0, add, 0, DType::F32);
    graph.add_edge(bias, 0, add, 1, DType::F32);

    assert!(graph.validate().is_ok());

    let mut inputs = HashMap::new();
    inputs.insert(input, vec![Tensor::zeros(&[1, 768], DType::F32)]);
    inputs.insert(weights, vec![Tensor::zeros(&[768, 768], DType::F32)]);
    inputs.insert(bias, vec![Tensor::zeros(&[1, 768], DType::F32)]);

    let scheduler = Scheduler::new(vec![Box::new(DummyCpuBackend::new())]);
    let result = scheduler.run(&graph, &inputs).unwrap();

    // add 노드가 출력
    assert!(result.contains_key(&4));
}

#[test]
fn test_graph_optimize_and_run() {
    let mut graph = Graph::new();
    let a = graph.add_node(OpType::Reshape, Default::default());
    let b = graph.add_node(OpType::MatMul, Default::default());
    let _orphan = graph.add_node(OpType::SiLU, Default::default());
    graph.add_edge(a, 0, b, 0, DType::F32);

    graph.optimize(&[Pass::DeadNodeElimination]).unwrap();
    assert_eq!(graph.nodes().len(), 2);

    let mut inputs = HashMap::new();
    inputs.insert(0, vec![Tensor::zeros(&[1, 64], DType::F32)]);

    let scheduler = Scheduler::new(vec![Box::new(DummyCpuBackend::new())]);
    let result = scheduler.run(&graph, &inputs).unwrap();
    assert!(!result.is_empty());
}
