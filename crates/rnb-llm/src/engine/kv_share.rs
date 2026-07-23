//! mt83 Stage B: Engine accessors for speculative drafter cross-attention.
//!
//! 이 모듈은 `rnb-mtp` 의 drafter 가 target backbone 에 접근해야 하는 세 가지
//! 채널을 노출한다:
//!
//! 1. `kv_view()` — target 의 KV cache 를 [`crate::KvBorrow`] trait object 로
//!    빌려옴. drafter cross-attention 의 K/V 입력.
//! 2. `last_layer_hidden()` — 마지막 forward step 의 normed last-layer hidden
//!    (lm_head 입력 형태). drafter Q 의 시드.
//! 3. `token_embd_row(token_id)` — target 의 token embedding 한 행을 f32 로
//!    dequant. drafter 가 last accepted token 의 embedding 을 사용할 때 쓴다.
//!
//! Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md`
//! §6 "Target — shared_kv_states dict 생성" + §"Engine 변경 사항".
//! (mt83 spec `...-cross-attention-design.md` 는 mt84 spec 으로 superseded.)

#[cfg(not(feature = "cuda"))]
use super::dequant::dequantize_bytes_to_f32;
use super::state::Engine;
use crate::{KvBorrow, SharedKvLayer, SharedKvStates};

impl Engine {
    /// Read-only borrow of target's KV cache for drafter cross-attention.
    ///
    /// 반환된 trait object 는 layer 별 K/V 를 매 호출마다 F16 → f32 dequant 해서
    /// 돌려준다. Drafter 의 cross-attention 호출은 layer 당 1-2 회라 dequant 비용
    /// 은 sub-ms 수준 (Gemma 4 E4B 기준 pos * kv_dim = ~수 KB).
    pub fn kv_view(&self) -> &dyn KvBorrow {
        &self.kv_cache
    }

    /// 마지막 decode/prefill step 의 last decoder layer 의 normed hidden state.
    /// `output_norm` 적용 직후 (lm_head 입력과 같은 형태).
    /// 길이 = `metadata.hidden_dim`.
    ///
    /// `forward` 한 번도 안 돌린 상태에서 호출하면 빈 slice 가 돌아온다. 호출처
    /// 는 `is_empty()` 또는 `len() == hidden_dim` 으로 사전 확인할 것.
    pub fn last_layer_hidden(&self) -> &[f32] {
        &self.last_layer_hidden_cached
    }

    /// Returns the post-output_norm last layer hidden activation captured at
    /// the most recent decode step. Length = `hidden_dim`. Empty (zero-length
    /// slice) if no decode has been performed yet.
    ///
    /// Consumed by external drafter MTP (mc78 Task 12).
    ///
    /// Note: `last_layer_hidden_cached` is also updated by prefill (so the
    /// last forward step wins). In the generate loop prefill always precedes
    /// the first decode call, so Task 12 must only read this after at least
    /// one decode step has executed.
    pub(crate) fn last_hidden_for_decode(&self) -> &[f32] {
        &self.last_layer_hidden_cached
    }

    /// layer `idx` 가 sliding-window attention 인지 판정. `sliding_window_pattern`
    /// 의 cyclic lookup (pattern length 가 num_layers 면 그대로, 짧으면 modulo).
    /// pattern 이 비어있는 모델 (non-Gemma4) 은 `false` 반환.
    pub fn is_sliding_at(&self, layer_idx: usize) -> bool {
        let pattern = &self.metadata.sliding_window_pattern;
        if pattern.is_empty() {
            return false;
        }
        pattern[layer_idx % pattern.len()]
    }

    /// layer `idx` 의 attention head dimension. Gemma4 E4B 의 SWA vs full 분기에 사용:
    /// sliding layer = `key_length_swa` (256), full layer = `key_length_full` (512).
    /// pattern 이 없거나 둘 다 0 이면 metadata 의 단일 `head_dim` 반환.
    pub fn head_dim_for_layer(&self, layer_idx: usize) -> usize {
        if self.is_sliding_at(layer_idx) && self.metadata.key_length_swa > 0 {
            self.metadata.key_length_swa
        } else if self.metadata.key_length_full > 0 {
            self.metadata.key_length_full
        } else {
            self.metadata.head_dim
        }
    }

    /// mt84 Stage β: drafter 의 cross-attention 에 넘길 shared K/V dict.
    ///
    /// Gemma 4 target 의 `store_full_length_kv = True` layer 두 개 (sliding 의
    /// 마지막 non-shared layer, full 의 마지막 non-shared layer) 에서 K/V 를
    /// 뽑아 dict 로 묶는다. 각 layer 의 K/V 는 F16 → F32 dequant 후
    /// `[seq_len, n_kv_heads * head_dim]` (KvBorrow 의 raw layout) 에서
    /// `[n_kv_heads, seq_len, head_dim]` row-major 로 transpose.
    ///
    /// Layer 자동 결정 (spec §6):
    /// - `first_kv_shared = num_layers - shared_kv_layers`
    /// - `prev_layers = layer_types[..first_kv_shared]`
    /// - sliding 의 마지막 idx = `prev_layers` 의 마지막 sliding layer
    /// - full 의 마지막 idx = `prev_layers` 의 마지막 full layer
    ///
    /// Spec: `docs/superpowers/specs/2026-05-14-gemma4-assistant-backbone-reuse-design.md` §6.
    pub fn shared_kv_states_for_drafter(&self) -> SharedKvStates {
        assert!(
            !self.metadata.sliding_window_pattern.is_empty(),
            "shared_kv_states_for_drafter requires sliding_window_pattern \
             (Gemma4 only); current arch has empty pattern"
        );
        let n_layers = self.metadata.num_layers;
        let shared_kv = self.metadata.shared_kv_layers;
        let first_kv_shared = n_layers.checked_sub(shared_kv).unwrap_or_else(|| {
            panic!("shared_kv_layers ({shared_kv}) > num_layers ({n_layers}) — corrupt metadata")
        });
        assert!(
            first_kv_shared > 0,
            "first_kv_shared == 0: shared_kv_layers must be < num_layers \
             (current: shared_kv={shared_kv}, num_layers={n_layers})"
        );

        // Find the last sliding / full layer in prev_layers (= 0..first_kv_shared).
        // is_sliding_at(idx) 는 sliding_window_pattern 의 cyclic lookup 이므로
        // pattern 이 비어있으면 모든 layer 가 false (full) 로 잡힘.
        let mut sliding_idx: Option<usize> = None;
        let mut full_idx: Option<usize> = None;
        for i in (0..first_kv_shared).rev() {
            if self.is_sliding_at(i) {
                if sliding_idx.is_none() {
                    sliding_idx = Some(i);
                }
            } else if full_idx.is_none() {
                full_idx = Some(i);
            }
            if sliding_idx.is_some() && full_idx.is_some() {
                break;
            }
        }

        SharedKvStates {
            sliding_attention: self.extract_shared_kv_layer(
                sliding_idx.expect("no sliding layer found in prev_layers for drafter dict"),
            ),
            full_attention: self.extract_shared_kv_layer(
                full_idx.expect("no full attention layer found in prev_layers for drafter dict"),
            ),
        }
    }

    /// Target layer `layer_idx` 의 K/V 를 dequant + transpose 해서 SharedKvLayer 로 반환.
    fn extract_shared_kv_layer(&self, layer_idx: usize) -> SharedKvLayer {
        let head_dim = self.head_dim_for_layer(layer_idx);
        let kv = self.kv_view();
        let seq_len = kv.pos();
        // Layer 별 KV head 수가 다른 모델 (per-layer GQA) 대비: kv_dim_for_layer 가
        // single source. metadata.num_kv_heads 는 global 이라 layer 별 가변이면 어긋남.
        let kv_dim = kv.kv_dim_for_layer(layer_idx);
        let n_kv_heads = kv_dim / head_dim;
        assert_eq!(
            n_kv_heads * head_dim,
            kv_dim,
            "kv_dim {kv_dim} not divisible by head_dim {head_dim} at layer {layer_idx}"
        );

        let k_raw = kv.k_layer(layer_idx);
        let v_raw = kv.v_layer(layer_idx);

        let k = transpose_to_head_major(&k_raw, seq_len, n_kv_heads, head_dim);
        let v = transpose_to_head_major(&v_raw, seq_len, n_kv_heads, head_dim);

        SharedKvLayer {
            k,
            v,
            n_kv_heads,
            seq_len,
            head_dim,
        }
    }

    /// mc78 Task 9/14: `generate_with_external_drafter` (Task 12) 에서 호출하는
    /// thin wrapper. 구현 본체는 `shared_kv_states_for_drafter()` 가 소유한다.
    ///
    /// **mc78 Task 14 검증 — Scenario A (host-resident KV):**
    ///
    /// `KVCache` (host-resident `Vec<u16>` F16 슬롯) 는 CUDA backend 에서도 항상
    /// 최신 상태다. 두 경로 모두 확인됨:
    ///
    /// 1. **Decode 경로** (`decode_attention_compute`): `kv_cache.append()` 가 매
    ///    decode step 마다 호출돼 host KV 를 F16 으로 갱신한다. CUDA `decode_attention_kv`
    ///    device-side 버퍼는 GPU 에서 attention compute 를 위한 *추가 device copy* 이며
    ///    host copy 를 무효화하지 않는다.
    ///
    /// 2. **Prefill 경로** (`inference.rs`): CUDA prefill 이후
    ///    `self.kv_cache.replace_layer_f16_range()` 로 device 결과를 host 에 반영한다.
    ///
    /// 따라서 D2H readback 없이 `shared_kv_states_for_drafter()` 의 기존 CPU 경로
    /// (F16 → f32 dequant + head-major transpose) 가 CUDA backend 에서도 올바른 KV 를
    /// 반환한다. 별도 device 분기 불필요.
    ///
    /// **mc78 Task 15: Vulkan 정책 결정**
    ///
    /// Vulkan backend 는 자체 GPU 내부 attention cache buffer (`attention_cache_layers`)
    /// 를 소유하며, host `KVCache` 를 update 하지 않는다. 하지만 **production 기본값**
    /// (CLAUDE.md / mv40 정책) 에서 mobile Vulkan 은 **opt-in only** (`RNB_FORCE_MOBILE_VULKAN=1`).
    /// Default 는 CPU NEON 이다. 따라서 현재 `shared_kv_view()` 호출처들은 모두 CPU path
    /// 에서만 동작하므로 readback 불필요. Opt-in Vulkan 사용자가 `shared_kv_view()` 를
    /// 호출하려면 별도 구현이 필요하지만, 현재 external_drafter 는 default build 에서만 지원.
    pub(crate) fn shared_kv_view(&self) -> SharedKvStates {
        self.shared_kv_states_for_drafter()
    }

    /// N+1 token parallel forward for MTP verify (mc78 Task 10).
    ///
    /// `tokens` 슬라이스 (보통 `[anchor_token, draft_t1, ..., draft_tN]`, 길이 N+1) 를
    /// target 모델에 batch forward 로 돌리고, 각 위치의 top-1 argmax 를 반환한다.
    /// `result[i]` 는 `tokens[i]` 를 소비한 뒤 모델이 예측한 다음 토큰.
    ///
    /// KV cache 는 `tokens.len()` 포지션 전체에 대해 커밋된다. Caller 는 verify 가
    /// 일부 draft 토큰을 거부하면 `commit_kv_through` (Task 11) 로 롤백해야 한다.
    ///
    /// `position_offset` 은 `kv_cache.current_len()` 이 이미 해당 값과 일치한다는
    /// 전제 하에 document 용도로만 받는다 (실제 pos_start 는 `forward_prefill_all_logits`
    /// 내부에서 `kv_cache.current_len()` 으로 자동 결정).
    ///
    /// **구현 선택 — Option B (true batch parallel)**: `forward_prefill_all_logits` 를
    /// 재사용. 이 함수는 이미 N-token prefill + 모든 위치의 logits 반환을 지원하므로
    /// sequential fallback(Option A) 보다 빠르고 KV 쓰기도 한 번에 완료됨.
    pub(crate) fn forward_batch_verify(
        &mut self,
        tokens: &[u32],
        _position_offset: u32,
    ) -> crate::error::Result<Vec<u32>> {
        // mc78 verify wall fix: 기존 forward_prefill_all_logits 는 host CPU lm_head
        // (모든 N+1 position × 262144 vocab × 1536 hidden Q6_K dequant + matmul)
        // 로 verify wall 의 큰 부분. forward_prefill_argmax_tokens_collect_mtp 가
        // GPU argmax-only path (cu39 의 prefill_output_argmax_token_cuda 활용) 로
        // logits 전체 안 만들고 token id 만 반환 — 훨씬 빠름.
        let result = self.forward_prefill_argmax_tokens_collect_mtp(tokens)?;
        Ok(result.target_tokens)
    }

    /// Target Engine 의 KV cache 를 `new_position` 으로 truncate 한다.
    ///
    /// MTP verify 가 드래프트 일부를 거부하면 `forward_batch_verify` 가 N+1 포지션
    /// 전체에 대해 KV 를 커밋한 상태다. Caller 는 실제 수락된 토큰 수 M 에 맞춰
    /// `commit_kv_through(position_before + M + 1)` 로 롤백해야 한다.
    ///
    /// `new_position` 이상 인덱스의 KV 슬롯은 stale 이 되지만 명시적 clear 불필요 —
    /// attention mask 가 `kv_cache.current_len()` 까지만 attend 하고, 다음 decode
    /// 스텝이 해당 슬롯을 자동으로 덮어쓴다. RoPE 도 position 기반이라 재계산 자동.
    ///
    /// Engine 의 decode position 은 별도 scalar counter 없이 `kv_cache.current_len()`
    /// 이 단독 source 이므로 KV truncate 한 번으로 충분하다.
    pub(crate) fn commit_kv_through(&mut self, new_position: u32) {
        let new_len = (new_position as usize).min(self.kv_cache.max_seq_len);
        self.kv_cache.set_len(new_len);
    }

    /// Target `token_embd.weight` 의 row `[token_id]` 를 f32 로 dequant 한 결과.
    /// 길이 = `metadata.hidden_dim`.
    ///
    /// Drafter 가 last accepted token 의 embedding 을 가져갈 때 사용. Q6_K
    /// (Gemma 4 E4B 의 token_embd dtype) 기준 row 당 ~2KB, dequant 0.1ms 미만.
    pub fn token_embd_row(&self, token_id: u32) -> Vec<f32> {
        let Some(weights) = self.weights.as_ref() else {
            // Mock engine path: 모델 weight 가 없으면 빈 Vec 반환. caller 가
            // `is_empty()` 로 확인. (production code 는 항상 weights 가 있음.)
            return Vec::new();
        };
        let embd = &weights.token_embd;
        #[cfg(feature = "cuda")]
        {
            return embd
                .embedding_gather_cuda(&[token_id])
                .unwrap_or_else(|err| {
                    panic!(
                    "CUDA token embedding gather failed; CPU quantized fallback is disabled: {err}"
                )
                });
        }
        #[cfg(not(feature = "cuda"))]
        {
            let cols = embd.cols;
            let bytes = match embd.data.as_bytes() {
                Some(bytes) => bytes,
                None => return Vec::new(),
            };
            let bytes_per_row = bytes.len() / embd.rows;
            let row_start = token_id as usize * bytes_per_row;
            let row_bytes = &bytes[row_start..row_start + bytes_per_row];
            let mut row = dequantize_bytes_to_f32(row_bytes, embd.ggml_type);
            row.truncate(cols);
            row
        }
    }
}

/// `[seq_len, n_kv_heads * head_dim]` (row-major) → `[n_kv_heads, seq_len, head_dim]`
/// (row-major) 로 axis swap.
///
/// KvBorrow 의 `k_layer`/`v_layer` 는 KV cache 의 native layout `[seq_len, n_kv_heads,
/// head_dim]` flatten 을 돌려준다. drafter 의 cross-attention 은 head-major access
/// 가 필요해서 head 축을 outermost 로 옮긴다.
///
/// Spec §6 의 `key_states.transpose(1, 2)` (transformers 의 `[B, n_kv_heads, seq_len,
/// head_dim]`) 와 동치.
fn transpose_to_head_major(
    raw: &[f32],
    seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(
        raw.len(),
        seq_len * n_kv_heads * head_dim,
        "transpose_to_head_major input length mismatch (release-safe assert)"
    );
    let mut out = vec![0f32; raw.len()];
    for s in 0..seq_len {
        for h in 0..n_kv_heads {
            for d in 0..head_dim {
                let src = s * (n_kv_heads * head_dim) + h * head_dim + d;
                let dst = h * (seq_len * head_dim) + s * head_dim + d;
                out[dst] = raw[src];
            }
        }
    }
    out
}

#[cfg(test)]
mod transpose_tests {
    use super::*;

    /// 작은 fixture 로 transpose 가 axis 0↔1 swap 을 올바르게 수행하는지 검증.
    /// seq_len=2, n_kv_heads=3, head_dim=2.
    /// raw layout: token0 의 (h0, h1, h2) followed by token1 의 (h0, h1, h2).
    #[test]
    fn transpose_axis_swap_correctness() {
        let seq_len = 2;
        let n_kv_heads = 3;
        let head_dim = 2;
        // token-major: [t0h0d0,t0h0d1, t0h1d0,t0h1d1, t0h2d0,t0h2d1,
        //               t1h0d0,t1h0d1, t1h1d0,t1h1d1, t1h2d0,t1h2d1]
        let raw: Vec<f32> = (0..(seq_len * n_kv_heads * head_dim))
            .map(|i| i as f32)
            .collect();

        let out = transpose_to_head_major(&raw, seq_len, n_kv_heads, head_dim);

        // head-major: [h0(t0d0,t0d1,t1d0,t1d1), h1(...), h2(...)]
        // expected[h*seq*head + t*head + d] = raw[t*(n_kv*head) + h*head + d]
        for s in 0..seq_len {
            for h in 0..n_kv_heads {
                for d in 0..head_dim {
                    let src = s * (n_kv_heads * head_dim) + h * head_dim + d;
                    let dst = h * (seq_len * head_dim) + s * head_dim + d;
                    assert_eq!(out[dst], raw[src], "mismatch at s={s} h={h} d={d}");
                }
            }
        }
    }
}
