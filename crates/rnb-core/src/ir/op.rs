use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OpType {
    MatMul,
    Add,
    Mul,
    RMSNorm,
    LayerNorm,
    RoPE,
    SiLU,
    GeLU,
    Softmax,
    Reshape,
    Transpose,
    Gather,
    Concat,
    Split,
    Cast,
    /// Scaled dot-product attention (fused).
    /// inputs: [q, k, v]
    /// attrs:  head_dim (Int), num_heads (Int), num_kv_heads (Int)
    /// output: [seq_len, num_heads * head_dim]
    Attention,
    /// 외부에서 텐서를 주입받는 placeholder 노드.
    /// Engine::forward 에서 external_inputs 로 실제 텐서를 제공한다.
    Placeholder,
    Custom(String),
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpType::Custom(name) => write!(f, "Custom({name})"),
            other => write!(f, "{other:?}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_op_display() {
        assert_eq!(format!("{}", OpType::MatMul), "MatMul");
        assert_eq!(format!("{}", OpType::Custom("MyOp".into())), "Custom(MyOp)");
    }
    #[test]
    fn test_attr_variants() {
        let i = Attr::Int(42);
        let f = Attr::Float(2.5);
        let s = Attr::String("hello".into());
        assert!(matches!(i, Attr::Int(42)));
        assert!(matches!(f, Attr::Float(_)));
        assert!(matches!(s, Attr::String(_)));
    }
}
