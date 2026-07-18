use crate::error::{Result, RnbError};
use crate::ir::edge::Edge;
use crate::ir::node::{Node, NodeId};
use crate::ir::op::{Attr, OpType};
use crate::tensor::DType;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Default)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
    pub(crate) edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, op: OpType, attrs: HashMap<String, Attr>) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op,
            attrs,
            shape: None,
        });
        id
    }

    pub fn add_edge(
        &mut self,
        from: NodeId,
        out_port: usize,
        to: NodeId,
        in_port: usize,
        dtype: DType,
    ) {
        self.edges.push(Edge {
            from: (from, out_port),
            to: (to, in_port),
            dtype,
        });
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut [Node] {
        &mut self.nodes
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id)
    }

    pub fn validate(&self) -> Result<()> {
        let node_count = self.nodes.len();
        for edge in &self.edges {
            if edge.from.0 >= node_count {
                return Err(RnbError::InvalidGraph(format!(
                    "edge references non-existent source node {}",
                    edge.from.0
                )));
            }
            if edge.to.0 >= node_count {
                return Err(RnbError::InvalidGraph(format!(
                    "edge references non-existent destination node {}",
                    edge.to.0
                )));
            }
        }
        Ok(())
    }

    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<NodeId>> = vec![vec![]; n];

        for edge in &self.edges {
            let from = edge.from.0;
            let to = edge.to.0;
            if from < n && to < n {
                adj[from].push(to);
                in_degree[to] += 1;
            }
        }

        let mut queue: VecDeque<NodeId> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != n {
            return Err(RnbError::InvalidGraph(
                "cycle detected in graph".to_string(),
            ));
        }

        Ok(order)
    }

    pub fn output_nodes(&self) -> Vec<NodeId> {
        let n = self.nodes.len();
        // output nodes = nodes that have no outgoing edges
        let mut has_outgoing = vec![false; n];
        for edge in &self.edges {
            if edge.from.0 < n {
                has_outgoing[edge.from.0] = true;
            }
        }
        (0..n).filter(|&i| !has_outgoing[i]).collect()
    }

    pub fn inputs_of(&self, id: NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.to.0 == id).collect()
    }

    pub fn outputs_of(&self, id: NodeId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from.0 == id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::op::OpType;
    use crate::tensor::DType;

    fn build_linear_graph() -> Graph {
        let mut g = Graph::new();
        let input = g.add_node(OpType::Reshape, Default::default());
        let weights = g.add_node(OpType::Reshape, Default::default());
        let bias = g.add_node(OpType::Reshape, Default::default());
        let matmul = g.add_node(OpType::MatMul, Default::default());
        let add = g.add_node(OpType::Add, Default::default());
        g.add_edge(input, 0, matmul, 0, DType::F32);
        g.add_edge(weights, 0, matmul, 1, DType::F32);
        g.add_edge(matmul, 0, add, 0, DType::F32);
        g.add_edge(bias, 0, add, 1, DType::F32);
        g
    }

    #[test]
    fn test_graph_add_nodes() {
        let g = build_linear_graph();
        assert_eq!(g.nodes().len(), 5);
        assert_eq!(g.edges().len(), 4);
    }

    #[test]
    fn test_graph_validate_ok() {
        let g = build_linear_graph();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_graph_validate_invalid_edge() {
        let mut g = Graph::new();
        let a = g.add_node(OpType::Add, Default::default());
        g.add_edge(a, 0, 999, 0, DType::F32);
        assert!(g.validate().is_err());
    }

    #[test]
    fn test_topological_order() {
        let g = build_linear_graph();
        let order = g.topological_order().unwrap();
        assert_eq!(order.len(), 5);
        let pos = |id: NodeId| order.iter().position(|&x| x == id).unwrap();
        assert!(pos(0) < pos(3));
        assert!(pos(1) < pos(3));
        assert!(pos(3) < pos(4));
        assert!(pos(2) < pos(4));
    }

    #[test]
    fn test_graph_output_nodes() {
        let g = build_linear_graph();
        let outputs = g.output_nodes();
        assert_eq!(outputs, vec![4]);
    }
}
