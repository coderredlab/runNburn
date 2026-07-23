use std::collections::HashSet;

use crate::error::Result;
use crate::ir::node::Node;
use crate::ir::op::OpType;
use crate::tensor::{DType, Tensor};

/// 예전 `crate::ir::Graph` 노드 실행용 backend trait.
///
/// `rnb_backend_api::Backend`와는 별개의 옛 테스트 실행 계약이다.
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn supported_ops(&self) -> &HashSet<OpType>;
    fn supports_dtype(&self, dtype: DType) -> bool;
    fn execute(&self, node: &Node, inputs: &[&Tensor]) -> Result<Vec<Tensor>>;
}
