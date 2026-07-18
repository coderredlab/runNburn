//! Decode-time attention cache read/write and attention compute.

use super::*;

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_attention_compute(
    kv_cache: &mut KVCache,
    metadata: &ModelMetadata,
    architecture: ModelArchitecture,
    layer_idx: usize,
    kv_cache_layer: usize,
    owns_kv: bool,
    pos: usize,
    layout: AttentionLayout,
    q_slice: &[f32],
    k_slice: &[f32],
    v_slice: &[f32],
    attn_out: &mut [f32],
    // cu29 Phase 2: hd=128 fused path. caller 가 이미 kv_cache.append_bits_range
    // 한 상태로 들어오니까 내부 append 는 skip.
    skip_kv_append: bool,
    // cu47 step 32: caller-provided device output carrier. Some 시 cuda cached
    // attention path 가 결과를 carrier 에 직접 write (D2H + Vec<f32> alloc 제거).
    // host attn_out slice 는 caller 책임 (host post-processing 시).
    output_dev_target: Option<u64>,
    // cu51 step 43: KV cache device source. Some 시 attention 의 K/V H2D 제거.
    last_token_k_dev: Option<u64>,
    last_token_v_dev: Option<u64>,
    q_dev_override: Option<u64>,
    use_device_len: bool,
    #[cfg(feature = "vulkan")] mut gpu_runtime: Option<&mut backend_runtime::GpuRuntime>,
) -> crate::error::Result<bool> {
    #[cfg(not(feature = "cuda"))]
    let _ = (
        last_token_k_dev,
        last_token_v_dev,
        q_dev_override,
        use_device_len,
    );
    let num_heads = layout.num_heads;
    let num_kv_heads = layout.num_kv_heads;
    let head_dim = layout.head_dim;
    let q_dim = layout.q_dim;
    let kv_dim = layout.kv_dim;
    let kv_len = pos + 1;

    let _ = output_dev_target; // legacy host-return path; carrier only used in cuda cached branch below.
    let gpu_attn_ok = if owns_kv && num_heads == 1 && num_kv_heads == 1 {
        let (cached_k_f16, cached_v_f16) = kv_cache.get_up_to(kv_cache_layer, kv_len);
        match backend_runtime::try_decode_attention_single_head_if_supported(
            #[cfg(feature = "vulkan")]
            gpu_runtime.as_deref_mut(),
            layer_idx,
            kv_cache_layer,
            pos,
            &q_slice[..head_dim],
            k_slice,
            v_slice,
            &cached_k_f16[..kv_len * head_dim],
            &cached_v_f16[..kv_len * head_dim],
            head_dim,
            kv_len,
            &mut attn_out[..head_dim],
        ) {
            Ok(true) => true,
            Ok(false) => false,
            Err(e) => {
                if let Some(Ok((k_bits, v_bits))) =
                    backend_runtime::materialize_attention_kv_for_layer_if_supported(
                        #[cfg(feature = "vulkan")]
                        gpu_runtime.as_deref_mut(),
                        kv_cache_layer,
                        num_kv_heads,
                        kv_len,
                        head_dim,
                        kv_dim,
                    )
                {
                    kv_cache.replace_layer_f16(kv_cache_layer, kv_len, &k_bits, &v_bits);
                } else {
                    kv_cache.append(layer_idx, pos, k_slice, v_slice);
                }
                eprintln!("[gpu] attention decode failed, CPU fallback: {}", e);
                false
            }
        }
    } else {
        false
    };

    if !gpu_attn_ok {
        // cu32: skip_kv_append 적용. cu29 에서 인자 추가됐지만 내부에서 미적용
        // 이라 fused path 에서 KvCache 두 번 append (bits + host f32 RoPE 안 적용
        // k_buf) → 두 번째가 RoPE 안 된 K_raw 로 덮어써서 정확도 깨짐.
        if owns_kv && !skip_kv_append {
            kv_cache.append(kv_cache_layer, pos, k_slice, v_slice);
        }
        let (cached_k_owned, cached_v_owned);
        let (cached_k_f16, cached_v_f16) = if super::policy::cache_read_disabled() {
            cached_k_owned = k_slice
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect::<Vec<_>>();
            cached_v_owned = v_slice
                .iter()
                .map(|&x| half::f16::from_f32(x).to_bits())
                .collect::<Vec<_>>();
            (&cached_k_owned[..], &cached_v_owned[..])
        } else {
            kv_cache.get_up_to(kv_cache_layer, kv_len)
        };
        if kv_trace_enabled() {
            eprintln!(
                "[kv-trace][decode-read] layer={} cache_layer={} owns_kv={} kv_len={} cached_k={} cached_v={} expected={}",
                layer_idx,
                kv_cache_layer,
                owns_kv,
                kv_len,
                cached_k_f16.len(),
                cached_v_f16.len(),
                kv_len * kv_dim
            );
        }
        if layer_idx == 0 && attn_trace_enabled() {
            let cached_k_f32: Vec<f32> = cached_k_f16
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            let cached_v_f32: Vec<f32> = cached_v_f16
                .iter()
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect();
            emit_vec_trace("decode", layer_idx, "cached_k", &cached_k_f32);
            emit_vec_trace("decode", layer_idx, "cached_v", &cached_v_f32);
        }
        let attn_scale = resolve_attention_scale(metadata, architecture);
        let sliding_window = active_sliding_window(metadata, architecture, layer_idx);
        let softcap = resolve_attention_softcap(architecture);
        // cu47 step 32: caller carrier 제공 + cuda cached path 지원 시 device output.
        // 그 외 host return path fallback.
        #[cfg(feature = "cuda")]
        if let Some(dev_target) = output_dev_target {
            let dev_result = backend_runtime::decode_attention_cached_to_device_if_supported(
                Some(kv_cache_layer),
                q_slice,
                cached_k_f16,
                cached_v_f16,
                kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                attn_scale,
                sliding_window,
                softcap.is_some(),
                dev_target,
                last_token_k_dev,
                last_token_v_dev,
                q_dev_override,
                use_device_len,
            );
            if let Some(res) = dev_result {
                res?;
                if layer_idx == 0 && attn_trace_enabled() {
                    emit_vec_trace("decode", layer_idx, "attn_out", &attn_out[..q_dim]);
                }
                return Ok(true);
            }
        }
        #[cfg(feature = "cuda")]
        let mut attention_done = false;
        #[cfg(not(feature = "cuda"))]
        let attention_done;
        #[cfg(feature = "cuda")]
        if let Some(out) = backend_runtime::decode_attention_hd256_if_supported(
            Some(kv_cache_layer),
            q_slice,
            cached_k_f16,
            cached_v_f16,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            attn_scale,
            sliding_window,
            softcap.is_some(),
        ) {
            let out = out?;
            attn_out[..q_dim].copy_from_slice(&out[..q_dim]);
            attention_done = true;
        }
        #[cfg(not(feature = "cuda"))]
        {
            // M2 KV residency(RNB_METAL_ATTN_RESIDENT=1): host KV 에서 device 에 없는
            // token 만 incremental 복사 후 attention(매 토큰 전체 KV 업로드 제거).
            // 미설정 시 M1b host-KV-upload metal(RNB_METAL_ATTN_COMPUTE=1) → cpu 순.
            let capacity = kv_cache.max_seq_len;
            attention_done = backend_runtime::metal_attn_decode_kv_resident_into_if_supported(
                kv_cache_layer,
                q_slice,
                cached_k_f16,
                cached_v_f16,
                &mut attn_out[..q_dim],
                num_heads,
                num_kv_heads,
                head_dim,
                kv_len,
                attn_scale,
                capacity,
                sliding_window,
                softcap.is_some(),
            )? || backend_runtime::metal_attn_decode_into_if_supported(
                q_slice,
                cached_k_f16,
                cached_v_f16,
                &mut attn_out[..q_dim],
                num_heads,
                num_kv_heads,
                head_dim,
                kv_len,
                attn_scale,
                sliding_window,
                softcap.is_some(),
            )?;
        }
        if !attention_done {
            if super::policy::flash_decode_enabled() {
                kernels::attention::attention_decode_flash(
                    q_slice,
                    cached_k_f16,
                    cached_v_f16,
                    &mut attn_out[..q_dim],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_len,
                    attn_scale,
                    sliding_window,
                    softcap,
                );
            } else {
                kernels::attention::attention_decode_into_with_scale_window_and_softcap(
                    q_slice,
                    cached_k_f16,
                    cached_v_f16,
                    &mut attn_out[..q_dim],
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    kv_len,
                    attn_scale,
                    sliding_window,
                    softcap,
                );
            }
        }
        if layer_idx == 0 && attn_trace_enabled() {
            emit_vec_trace("decode", layer_idx, "attn_out", &attn_out[..q_dim]);
        }
    }

    Ok(false)
}
