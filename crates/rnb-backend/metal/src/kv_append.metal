#include <metal_stdlib>
using namespace metal;

// KV append: rope/norm 끝난 k/v(f32, [kv_dim])를 KV cache device buffer 의
// pos 슬롯에 f16 bits 로 write. host roundtrip 없이 device 에서 직접 append.
// k_cache/v_cache: [capacity*kv_dim] ushort(f16 bits) — attn_decode 커널이
// 그대로 읽는다. f32→f16 변환은 CPU `half::f16::from_f32`(round-to-nearest-even)
// 와 일치 (metal (half) cast 도 RNE).
kernel void kv_append(
    device const float* k_f32 [[buffer(0)]],
    device const float* v_f32 [[buffer(1)]],
    device ushort*      k_cache [[buffer(2)]],
    device ushort*      v_cache [[buffer(3)]],
    constant uint&      kv_dim [[buffer(4)]],
    constant uint&      pos    [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= kv_dim) return;
    uint off = pos * kv_dim + i;
    k_cache[off] = as_type<ushort>((half)k_f32[i]);
    v_cache[off] = as_type<ushort>((half)v_f32[i]);
}
