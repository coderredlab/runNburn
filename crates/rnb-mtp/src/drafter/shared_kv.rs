/// Drafter cross-attention 용 shared K/V state — single layer.
///
/// `k`, `v` 는 row-major `[n_kv_heads, seq_len, head_dim]` flatten f32 (target prefill
/// 시 F16 으로 저장된 KV cache 를 F32 로 dequant + transpose 한 결과). mt84 Stage β.
///
/// Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md`
/// §6 "Target 측 — shared_kv_states dict 생성".
pub struct SharedKvLayer {
    /// row-major `[n_kv_heads, seq_len, head_dim]` flatten.
    pub k: Vec<f32>,
    /// 동일 layout.
    pub v: Vec<f32>,
    pub n_kv_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
}

/// Drafter forward 가 매 step 마다 받는 shared K/V dict.
///
/// `Gemma4TextModel.forward(... shared_kv_states=...)` argument 와 대응. drafter
/// 의 모든 layer 가 `is_kv_shared_layer = True` 라 K/V projection 없이 layer_type
/// 별로 이 dict 에서 K/V 를 가져온다.
///
/// Spec §Engine 변경 사항.
pub struct SharedKvStates {
    pub sliding_attention: SharedKvLayer,
    pub full_attention: SharedKvLayer,
}
