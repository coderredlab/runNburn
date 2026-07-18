use crate::ir::node::NodeId;
use crate::tensor::DType;

pub type InputPort = usize;
pub type OutputPort = usize;

#[derive(Debug, Clone)]
pub struct Edge {
    pub from: (NodeId, OutputPort),
    pub to: (NodeId, InputPort),
    pub dtype: DType,
}
