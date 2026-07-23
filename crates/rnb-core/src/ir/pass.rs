use crate::error::Result;
use crate::ir::graph::Graph;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub enum Pass {
    ConstantFolding,
    OpFusion,
    DeadNodeElimination,
    LayoutOptimize,
}

pub fn dead_node_elimination(graph: &mut Graph) {
    let n = graph.nodes().len();

    // 어떤 edge에도 등장하지 않는 노드 = 완전히 고립된 dead 노드
    let mut in_any_edge: HashSet<usize> = HashSet::new();
    for edge in graph.edges() {
        in_any_edge.insert(edge.from.0);
        in_any_edge.insert(edge.to.0);
    }

    // 역방향 인접 리스트 구성 (to -> [from])
    let mut reverse_adj: Vec<Vec<usize>> = vec![vec![]; n];
    for edge in graph.edges() {
        if edge.from.0 < n && edge.to.0 < n {
            reverse_adj[edge.to.0].push(edge.from.0);
        }
    }

    // edge에 등장하는 노드 중 outgoing edge 없는 것 = 진짜 output 노드
    let mut has_outgoing = vec![false; n];
    for edge in graph.edges() {
        if edge.from.0 < n {
            has_outgoing[edge.from.0] = true;
        }
    }
    let real_outputs: Vec<usize> = (0..n)
        .filter(|&i| in_any_edge.contains(&i) && !has_outgoing[i])
        .collect();

    // 진짜 output들에서 역방향 BFS
    let mut alive: HashSet<usize> = HashSet::new();
    let mut queue: std::collections::VecDeque<usize> = real_outputs.into_iter().collect();

    while let Some(node) = queue.pop_front() {
        if alive.contains(&node) {
            continue;
        }
        alive.insert(node);
        for &pred in &reverse_adj[node] {
            if !alive.contains(&pred) {
                queue.push_back(pred);
            }
        }
    }

    let dead: HashSet<usize> = (0..n).filter(|id| !alive.contains(id)).collect();
    if dead.is_empty() {
        return;
    }
    graph.remove_nodes(&dead);
}

impl Graph {
    pub fn optimize(&mut self, passes: &[Pass]) -> Result<()> {
        for pass in passes {
            match pass {
                Pass::DeadNodeElimination => dead_node_elimination(self),
                Pass::ConstantFolding => {}
                Pass::OpFusion => {}
                Pass::LayoutOptimize => {}
            }
        }
        Ok(())
    }

    pub fn remove_nodes(&mut self, to_remove: &HashSet<usize>) {
        let old_count = self.nodes.len();

        // old ID -> new ID 매핑 생성
        let mut id_map: Vec<Option<usize>> = Vec::with_capacity(old_count);
        let mut new_id = 0usize;
        for i in 0..old_count {
            if to_remove.contains(&i) {
                id_map.push(None);
            } else {
                id_map.push(Some(new_id));
                new_id += 1;
            }
        }

        // 살아남는 노드만 필터링하고 ID 업데이트
        let old_nodes = std::mem::take(&mut self.nodes);
        self.nodes = old_nodes
            .into_iter()
            .filter(|n| !to_remove.contains(&n.id))
            .map(|mut n| {
                n.id = id_map[n.id].unwrap();
                n
            })
            .collect();

        // 제거된 노드와 연결된 엣지 필터링 후 ID 업데이트
        let old_edges = std::mem::take(&mut self.edges);
        self.edges = old_edges
            .into_iter()
            .filter(|e| !to_remove.contains(&e.from.0) && !to_remove.contains(&e.to.0))
            .map(|mut e| {
                e.from.0 = id_map[e.from.0].unwrap();
                e.to.0 = id_map[e.to.0].unwrap();
                e
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::graph::Graph;
    use crate::ir::op::OpType;
    use crate::tensor::DType;

    #[test]
    fn test_dead_node_elimination() {
        let mut g = Graph::new();
        let input = g.add_node(OpType::Reshape, Default::default());
        let weights = g.add_node(OpType::Reshape, Default::default());
        let matmul = g.add_node(OpType::MatMul, Default::default());
        let _orphan = g.add_node(OpType::Add, Default::default());
        g.add_edge(input, 0, matmul, 0, DType::F32);
        g.add_edge(weights, 0, matmul, 1, DType::F32);
        let original_count = g.nodes().len();
        dead_node_elimination(&mut g);
        assert!(g.nodes().len() < original_count);
    }

    #[test]
    fn test_optimize_pipeline() {
        let mut g = Graph::new();
        let a = g.add_node(OpType::Reshape, Default::default());
        let b = g.add_node(OpType::MatMul, Default::default());
        g.add_edge(a, 0, b, 0, DType::F32);
        let result = g.optimize(&[Pass::DeadNodeElimination]);
        assert!(result.is_ok());
    }
}
