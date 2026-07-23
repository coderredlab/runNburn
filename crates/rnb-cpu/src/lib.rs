#![cfg_attr(
    target_arch = "aarch64",
    feature(stdarch_neon_dotprod, stdarch_neon_i8mm, stdarch_neon_f16, f16)
)]

pub mod gemm;
pub mod kernels;
pub mod quantize;

use rnb_core::backend::Backend;
use rnb_core::error::{Result, RnbError};
use rnb_core::ir::node::Node;
use rnb_core::ir::op::OpType;
use rnb_core::tensor::{DType, Tensor};
use std::collections::HashSet;

pub struct CpuBackend {
    supported: HashSet<OpType>,
}

impl CpuBackend {
    pub fn new() -> Self {
        let mut supported = HashSet::new();
        for op in [
            OpType::MatMul,
            OpType::Add,
            OpType::Mul,
            OpType::RMSNorm,
            OpType::LayerNorm,
            OpType::SiLU,
            OpType::GeLU,
            OpType::Softmax,
            OpType::RoPE,
            OpType::Attention,
            OpType::Cast,
            OpType::Reshape,
            OpType::Transpose,
            OpType::Gather,
            OpType::Concat,
            OpType::Split,
            OpType::Placeholder,
        ] {
            supported.insert(op);
        }
        Self { supported }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn supported_ops(&self) -> &HashSet<OpType> {
        &self.supported
    }

    fn supports_dtype(&self, _dtype: DType) -> bool {
        true
    }

    fn execute(&self, node: &Node, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        match &node.op {
            OpType::MatMul => kernels::matmul::execute(inputs),
            OpType::Add => Ok(vec![kernels::elementwise::add(inputs[0], inputs[1])?]),
            OpType::Mul => Ok(vec![kernels::elementwise::mul(inputs[0], inputs[1])?]),
            OpType::RMSNorm => {
                let eps = get_attr_f32(&node.attrs, "eps").unwrap_or(1e-5);
                Ok(vec![kernels::norm::rms_norm(inputs[0], inputs[1], eps)?])
            }
            OpType::LayerNorm => {
                let eps = get_attr_f32(&node.attrs, "eps").unwrap_or(1e-5);
                let bias = inputs.get(2).copied();
                Ok(vec![kernels::norm::layer_norm(
                    inputs[0], inputs[1], bias, eps,
                )?])
            }
            OpType::SiLU => Ok(vec![kernels::activation::silu(inputs[0])?]),
            OpType::GeLU => Ok(vec![kernels::activation::gelu(inputs[0])?]),
            OpType::Softmax => Ok(vec![kernels::activation::softmax(inputs[0])?]),
            OpType::RoPE => {
                let position = get_attr_int(&node.attrs, "position").unwrap_or(0) as usize;
                let head_dim = get_attr_int(&node.attrs, "head_dim").unwrap_or(64) as usize;
                let theta = get_attr_f32(&node.attrs, "theta").unwrap_or(10000.0);
                Ok(vec![kernels::rope::rope(
                    inputs[0], position, head_dim, theta,
                )?])
            }
            OpType::Cast => {
                // attrs에 dtype 정보 없으면 F32로 캐스팅
                Ok(vec![kernels::elementwise::cast(inputs[0], DType::F32)?])
            }
            OpType::Reshape => {
                // 간단히: 두 번째 입력이 없으면 원본 반환
                Ok(vec![inputs[0].clone()])
            }
            OpType::Transpose => Ok(vec![inputs[0].transpose(0, 1)]),
            OpType::Attention => {
                if inputs.len() < 3 {
                    return Err(RnbError::InvalidGraph(format!(
                        "Attention: q/k/v 3개 입력 필요, 현재 {}개",
                        inputs.len()
                    )));
                }
                let num_heads = get_attr_int(&node.attrs, "num_heads").unwrap_or(1) as usize;
                let num_kv_heads =
                    get_attr_int(&node.attrs, "num_kv_heads").unwrap_or(num_heads as i64) as usize;
                let head_dim = get_attr_int(&node.attrs, "head_dim").unwrap_or(64) as usize;
                Ok(vec![kernels::attention::attention(
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )?])
            }
            OpType::Gather => {
                // in_port=0 → token_ids (indices), in_port=1 → embedding table
                // llama 그래프에서 Gather: token_ids → in_port=0, emb_weight → in_port=1
                if inputs.len() < 2 {
                    return Err(RnbError::InvalidGraph(format!(
                        "Gather: 최소 2개의 입력 필요 (token_ids, embedding_table), 현재 {}개",
                        inputs.len()
                    )));
                }
                let token_ids_tensor = inputs[0];
                let emb_table = inputs[1];

                // token_ids 텐서에서 u32 인덱스 추출
                // token_ids는 f32로 저장된 정수값
                let token_ids: Vec<u32> = if let Some(bytes) = token_ids_tensor.as_bytes() {
                    bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                        .collect()
                } else {
                    return Err(RnbError::InvalidGraph(
                        "Gather: token_ids 텐서에서 bytes 추출 실패".into(),
                    ));
                };

                // embedding table이 F32가 아니면 zeros fallback
                // (양자화 embedding의 경우 일단 zeros로 처리)
                if emb_table.dtype() != DType::F32 {
                    let embed_dim = emb_table.shape().last().copied().unwrap_or(1);
                    let out_data = vec![0.0f32; token_ids.len() * embed_dim];
                    return Ok(vec![Tensor::from_slice(
                        &out_data,
                        &[token_ids.len(), embed_dim],
                    )]);
                }

                Ok(vec![kernels::shape::gather(emb_table, &token_ids)?])
            }
            OpType::Concat => {
                let refs: Vec<&Tensor> = inputs.to_vec();
                let axis = get_attr_int(&node.attrs, "axis").unwrap_or(0) as usize;
                Ok(vec![kernels::shape::concat(&refs, axis)?])
            }
            OpType::Split => {
                // 간단히: 원본 반환
                Ok(vec![inputs[0].clone()])
            }
            OpType::Placeholder => {
                // Placeholder 노드는 external_inputs 로 텐서가 주입되어야 한다.
                // 혹시라도 실행이 오면 빈 F32 텐서를 반환한다.
                Ok(vec![Tensor::zeros(&[1], DType::F32)])
            }
            other => Err(RnbError::UnsupportedOp {
                op: format!("{other}"),
                backend: "cpu".into(),
            }),
        }
    }
}

fn get_attr_f32(
    attrs: &std::collections::HashMap<String, rnb_core::ir::op::Attr>,
    key: &str,
) -> Option<f32> {
    match attrs.get(key)? {
        rnb_core::ir::op::Attr::Float(v) => Some(*v as f32),
        _ => None,
    }
}

fn get_attr_int(
    attrs: &std::collections::HashMap<String, rnb_core::ir::op::Attr>,
    key: &str,
) -> Option<i64> {
    match attrs.get(key)? {
        rnb_core::ir::op::Attr::Int(v) => Some(*v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnb_core::ir::node::Node;

    #[test]
    fn test_cpu_backend_supports_all_ops() {
        let backend = CpuBackend::new();
        assert!(backend.supported_ops().contains(&OpType::MatMul));
        assert!(backend.supported_ops().contains(&OpType::RMSNorm));
        assert!(backend.supported_ops().contains(&OpType::SiLU));
        assert!(backend.supported_ops().contains(&OpType::Add));
    }

    #[test]
    fn test_cpu_backend_execute_add() {
        let backend = CpuBackend::new();
        let a = Tensor::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let b = Tensor::from_slice(&[3.0f32, 4.0], &[1, 2]);
        let node = Node {
            id: 0,
            op: OpType::Add,
            attrs: Default::default(),
            shape: None,
        };
        let result = backend.execute(&node, &[&a, &b]).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cpu_backend_execute_matmul() {
        let backend = CpuBackend::new();
        let w = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]);
        let x = Tensor::from_slice(&[3.0f32, 4.0], &[1, 2]);
        let node = Node {
            id: 0,
            op: OpType::MatMul,
            attrs: Default::default(),
            shape: None,
        };
        let result = backend.execute(&node, &[&w, &x]).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cpu_backend_unsupported_op() {
        let backend = CpuBackend::new();
        let t = Tensor::zeros(&[1], DType::F32);
        let node = Node {
            id: 0,
            op: OpType::Custom("UnknownOp".into()),
            attrs: Default::default(),
            shape: None,
        };
        let result = backend.execute(&node, &[&t]);
        assert!(result.is_err());
    }
}
