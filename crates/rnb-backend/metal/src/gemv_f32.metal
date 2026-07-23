#include <metal_stdlib>
using namespace metal;

// out[row] = sum_c weight[row*K + c] * input[c]
kernel void gemv_f32(
    device const float* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      K       [[buffer(3)]],
    uint                row     [[thread_position_in_grid]])
{
    float acc = 0.0;
    for (uint c = 0; c < K; c++) {
        acc += weight[row * K + c] * input[c];
    }
    out[row] = acc;
}

// pm26: chain 용 F32 GEMV. weight 를 byte slice + byte offset 으로 받아(NoCopy carrier
// weight 와 동일 인터페이스, gemv_q8_0 패턴), N row 1-thread/row dispatch. 27B GDN 의
// F32 ssm_alpha/beta 를 무손실 device GEMV 로 처리(gdn_quant_code 가 F32 를 None 처리해
// 48 GDN 이 host per-op 로 추락했던 것의 fix). out[row] = sum_c w[row*K+c]*input[c].
kernel void gemv_f32_chain(
    device const uchar* weight_bytes       [[buffer(0)]],  // N*K*4 bytes (f32 LE)
    device const float* input              [[buffer(1)]],  // K f32
    device float*       out                [[buffer(2)]],  // N f32
    constant uint&      N                  [[buffer(3)]],
    constant uint&      K                  [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint                row                [[thread_position_in_grid]])
{
    if (row >= N) return;
    device const float* w =
        (device const float*)(weight_bytes + weight_byte_offset);
    float acc = 0.0f;
    for (uint c = 0; c < K; c++) {
        acc += w[row * K + c] * input[c];
    }
    out[row] = acc;
}

// Router 전용 F32 GEMV. threadgroup 하나가 output row 하나를 맡고 SIMD-group의
// 32 lane이 연속 K 원소를 읽어 합산한다.
kernel void gemv_f32_chain_simd(
    device const uchar* weight_bytes       [[buffer(0)]],
    device const float* input              [[buffer(1)]],
    device float*       out                [[buffer(2)]],
    constant uint&      N                  [[buffer(3)]],
    constant uint&      K                  [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint3               threadgroup_pos    [[threadgroup_position_in_grid]],
    uint                lane               [[thread_index_in_simdgroup]])
{
    uint row = threadgroup_pos.x;
    if (row >= N) return;

    device const float* w =
        (device const float*)(weight_bytes + weight_byte_offset) + row * K;
    float partial = 0.0f;
    for (uint c = lane; c < K; c += 32) {
        partial += w[c] * input[c];
    }
    float sum = simd_sum(partial);
    if (lane == 0) {
        out[row] = sum;
    }
}

kernel void prefill_f32_proj(
    device const uchar* weight_bytes       [[buffer(0)]],
    device const float* input              [[buffer(1)]],
    device float*       out                [[buffer(2)]],
    constant uint&      N                  [[buffer(3)]],
    constant uint&      K                  [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      M                  [[buffer(6)]],
    uint2               gid                [[thread_position_in_grid]])
{
    uint token = gid.x;
    uint row = gid.y;
    if (token >= M || row >= N) return;

    device const float* w = (device const float*)(weight_bytes + weight_byte_offset);
    device const float* x = input + token * K;

    float acc = 0.0f;
    for (uint c = 0; c < K; c++) {
        acc += w[row * K + c] * x[c];
    }
    out[token * N + row] = acc;
}
