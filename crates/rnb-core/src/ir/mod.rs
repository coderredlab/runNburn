//! 예전 op 단위 core IR.
//!
//! `rnb-model-ir`의 `ModelGraph`가 모델 레이어 단위의 스케줄 입력을 담당하는
//! 반면, 이 모듈은 개별 op/node/edge 그래프와 간단한 그래프 pass를 담는다.
//! `rnb-loader`의 아키텍처별 그래프 빌더와 `rnb-core`/`rnb-cpu` 테스트에서
//! 아직 사용 중이라 공개 API를 유지한다.

pub mod edge;
pub mod graph;
pub mod node;
pub mod op;
pub mod pass;

pub use edge::{Edge, InputPort, OutputPort};
pub use graph::Graph;
pub use node::{Node, NodeId, ShapeInfo};
pub use op::{Attr, OpType};
pub use pass::Pass;
