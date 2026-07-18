use crate::ir::op::{Attr, OpType};
use std::collections::HashMap;

pub type NodeId = usize;

#[derive(Debug, Clone)]
pub struct ShapeInfo {
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: OpType,
    pub attrs: HashMap<String, Attr>,
    pub shape: Option<ShapeInfo>,
}
