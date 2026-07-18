use std::collections::HashMap;

use crate::error::{Result, RnbError};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::tensor::Tensor;

use super::traits::Backend;

pub struct Scheduler {
    backends: Vec<Box<dyn Backend>>,
}

impl Scheduler {
    /// 예전 core IR 그래프 실행기를 만든다.
    ///
    /// 새 `rnb-scheduler`의 레이어 배치 계획기와 다르게, 이 타입은
    /// `crate::ir::Graph`의 노드를 topological order로 직접 실행한다.
    pub fn new(backends: Vec<Box<dyn Backend>>) -> Self {
        Self { backends }
    }

    /// 그래프를 topological order로 실행.
    /// `external_inputs`에 있는 노드는 실행 건너뛰고 제공된 텐서를 그대로 사용.
    /// output nodes (나가는 엣지 없는 노드)의 결과만 반환.
    pub fn run(
        &self,
        graph: &Graph,
        external_inputs: &HashMap<NodeId, Vec<Tensor>>,
    ) -> Result<HashMap<NodeId, Vec<Tensor>>> {
        let order = graph.topological_order()?;

        // 각 노드의 출력 텐서를 저장
        let mut node_outputs: HashMap<NodeId, Vec<Tensor>> = HashMap::new();

        for node_id in &order {
            // external input이 제공된 경우 그대로 사용
            if let Some(tensors) = external_inputs.get(node_id) {
                node_outputs.insert(*node_id, tensors.clone());
                continue;
            }

            let node = graph.node(*node_id).expect("node must exist");

            // 이 노드의 입력 엣지들을 in_port 순으로 수집
            let mut input_edges = graph.inputs_of(*node_id);
            input_edges.sort_by_key(|e| e.to.1); // in_port 기준 정렬

            let mut inputs: Vec<&Tensor> = Vec::new();
            let mut input_tensor_holders: Vec<Tensor> = Vec::new();

            for edge in &input_edges {
                let src_node_id = edge.from.0;
                let out_port = edge.from.1;

                if let Some(outputs) = node_outputs.get(&src_node_id) {
                    if let Some(tensor) = outputs.get(out_port) {
                        input_tensor_holders.push(tensor.clone());
                    }
                }
            }

            for t in &input_tensor_holders {
                inputs.push(t);
            }

            // 첫 번째로 지원하는 백엔드 찾기 (fallback)
            let backend = self
                .backends
                .iter()
                .find(|b| {
                    b.supported_ops().contains(&node.op)
                        // 모든 입력 dtype을 지원하는지 확인
                        && input_edges.iter().all(|e| b.supports_dtype(e.dtype))
                })
                .ok_or_else(|| RnbError::UnsupportedOp {
                    op: format!("{}", node.op),
                    backend: "all".to_string(),
                })?;

            let outputs = backend.execute(node, &inputs)?;
            node_outputs.insert(*node_id, outputs);
        }

        // output nodes의 결과만 반환
        let output_node_ids = graph.output_nodes();
        let mut results = HashMap::new();
        for id in output_node_ids {
            if let Some(tensors) = node_outputs.remove(&id) {
                results.insert(id, tensors);
            }
        }

        Ok(results)
    }

    /// 그래프를 실행하되 모든 노드의 출력을 반환 (디버그용)
    pub fn run_all(
        &self,
        graph: &Graph,
        external_inputs: &HashMap<NodeId, Vec<Tensor>>,
    ) -> Result<HashMap<NodeId, Vec<Tensor>>> {
        let order = graph.topological_order()?;
        let mut node_outputs: HashMap<NodeId, Vec<Tensor>> = HashMap::new();

        for node_id in &order {
            if let Some(tensors) = external_inputs.get(node_id) {
                node_outputs.insert(*node_id, tensors.clone());
                continue;
            }

            let node = graph.node(*node_id).expect("node must exist");
            let mut input_edges = graph.inputs_of(*node_id);
            input_edges.sort_by_key(|e| e.to.1);

            let mut input_tensor_holders: Vec<Tensor> = Vec::new();
            for edge in &input_edges {
                if let Some(outputs) = node_outputs.get(&edge.from.0) {
                    if let Some(tensor) = outputs.get(edge.from.1) {
                        input_tensor_holders.push(tensor.clone());
                    }
                }
            }

            let inputs: Vec<&Tensor> = input_tensor_holders.iter().collect();
            let backend = self
                .backends
                .iter()
                .find(|b| {
                    b.supported_ops().contains(&node.op)
                        && input_edges.iter().all(|e| b.supports_dtype(e.dtype))
                })
                .ok_or_else(|| RnbError::UnsupportedOp {
                    op: format!("{}", node.op),
                    backend: "all".to_string(),
                })?;

            let outputs = backend.execute(node, &inputs)?;
            node_outputs.insert(*node_id, outputs);
        }

        Ok(node_outputs)
    }

    /// 특정 op + dtype을 지원하는 첫 번째 백엔드 찾기
    pub fn find_backend(
        &self,
        op: &crate::ir::op::OpType,
        dtype: crate::tensor::DType,
    ) -> Option<&dyn Backend> {
        self.backends
            .iter()
            .find(|b| b.supported_ops().contains(op) && b.supports_dtype(dtype))
            .map(|b| b.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::Graph;
    use crate::ir::op::OpType;
    use crate::tensor::{DType, Tensor};
    use std::collections::{HashMap, HashSet};

    struct MockBackend {
        name: String,
        ops: HashSet<OpType>,
    }

    impl MockBackend {
        fn new(name: &str, ops: Vec<OpType>) -> Self {
            Self {
                name: name.to_string(),
                ops: ops.into_iter().collect(),
            }
        }
    }

    impl Backend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }
        fn supported_ops(&self) -> &HashSet<OpType> {
            &self.ops
        }
        fn supports_dtype(&self, _dtype: DType) -> bool {
            true
        }
        fn execute(
            &self,
            _node: &crate::ir::node::Node,
            _inputs: &[&Tensor],
        ) -> crate::error::Result<Vec<Tensor>> {
            Ok(vec![Tensor::zeros(&[1], DType::F32)])
        }
    }

    #[test]
    fn test_scheduler_simple_graph() {
        let mut g = Graph::new();
        let a = g.add_node(OpType::Reshape, Default::default());
        let b = g.add_node(OpType::MatMul, Default::default());
        g.add_edge(a, 0, b, 0, DType::F32);
        let cpu = MockBackend::new("cpu", vec![OpType::Reshape, OpType::MatMul]);
        let scheduler = Scheduler::new(vec![Box::new(cpu)]);
        let inputs: HashMap<NodeId, Vec<Tensor>> = HashMap::new();
        let result = scheduler.run(&g, &inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scheduler_fallback() {
        let mut g = Graph::new();
        let a = g.add_node(OpType::Reshape, Default::default());
        let b = g.add_node(OpType::MatMul, Default::default());
        g.add_edge(a, 0, b, 0, DType::F32);
        let npu = MockBackend::new("npu", vec![OpType::MatMul]);
        let cpu = MockBackend::new("cpu", vec![OpType::Reshape, OpType::MatMul]);
        let scheduler = Scheduler::new(vec![Box::new(npu), Box::new(cpu)]);
        let inputs: HashMap<NodeId, Vec<Tensor>> = HashMap::new();
        let result = scheduler.run(&g, &inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scheduler_no_backend() {
        let mut g = Graph::new();
        let _a = g.add_node(OpType::RoPE, Default::default());
        let cpu = MockBackend::new("cpu", vec![OpType::MatMul]);
        let scheduler = Scheduler::new(vec![Box::new(cpu)]);
        let inputs: HashMap<NodeId, Vec<Tensor>> = HashMap::new();
        let result = scheduler.run(&g, &inputs);
        assert!(result.is_err());
    }
}
