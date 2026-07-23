use rnb_core::backend::Scheduler;
use rnb_core::ir::graph::Graph;
use rnb_core::ir::op::OpType;
use rnb_core::tensor::{DType, Tensor};
use rnb_cpu::CpuBackend;
use std::collections::HashMap;

#[test]
fn test_matmul_with_scheduler() {
    let mut graph = Graph::new();
    // input, weights 노드는 external_inputs로 제공할 source 노드
    let input = graph.add_node(OpType::Reshape, Default::default());
    let weights = graph.add_node(OpType::Reshape, Default::default());
    let matmul = graph.add_node(OpType::MatMul, Default::default());
    graph.add_edge(weights, 0, matmul, 0, DType::F32);
    graph.add_edge(input, 0, matmul, 1, DType::F32);

    let scheduler = Scheduler::new(vec![Box::new(CpuBackend::new())]);
    let mut inputs = HashMap::new();
    // weight: [2, 2] identity matrix
    inputs.insert(
        weights,
        vec![Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2])],
    );
    // input: [1, 2]
    inputs.insert(input, vec![Tensor::from_slice(&[1.0f32, 2.0], &[1, 2])]);

    let result = scheduler.run(&graph, &inputs).unwrap();
    // matmul은 output node (나가는 엣지 없음)
    assert!(result.contains_key(&matmul));
    let tensors = &result[&matmul];
    assert_eq!(tensors.len(), 1);
}

#[test]
fn test_add_with_scheduler() {
    let mut graph = Graph::new();
    let a_node = graph.add_node(OpType::Reshape, Default::default());
    let b_node = graph.add_node(OpType::Reshape, Default::default());
    let add_node = graph.add_node(OpType::Add, Default::default());
    graph.add_edge(a_node, 0, add_node, 0, DType::F32);
    graph.add_edge(b_node, 0, add_node, 1, DType::F32);

    let scheduler = Scheduler::new(vec![Box::new(CpuBackend::new())]);
    let mut inputs = HashMap::new();
    inputs.insert(a_node, vec![Tensor::from_slice(&[1.0f32, 2.0], &[1, 2])]);
    inputs.insert(b_node, vec![Tensor::from_slice(&[3.0f32, 4.0], &[1, 2])]);

    let result = scheduler.run(&graph, &inputs).unwrap();
    assert!(result.contains_key(&add_node));
}

#[test]
fn test_softmax_with_scheduler() {
    let mut graph = Graph::new();
    let src = graph.add_node(OpType::Reshape, Default::default());
    let softmax_node = graph.add_node(OpType::Softmax, Default::default());
    graph.add_edge(src, 0, softmax_node, 0, DType::F32);

    let scheduler = Scheduler::new(vec![Box::new(CpuBackend::new())]);
    let mut inputs = HashMap::new();
    inputs.insert(src, vec![Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3])]);

    let result = scheduler.run(&graph, &inputs).unwrap();
    assert!(result.contains_key(&softmax_node));
}
