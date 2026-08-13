#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#include <metal_stdlib>
using namespace metal;

// One SIMD-group computes one gate/up row pair. This preserves the two-output
// accumulator pressure of the production Q4_K kernel while sharing input loads
// and writing the SwiGLU activation directly.
kernel void gemv_q4k_swiglu_pair_nsg2(
    device const uchar* gate_weight [[buffer(0)]],
    device const uchar* up_weight   [[buffer(1)]],
    device const float* input       [[buffer(2)]],
    device float*       out         [[buffer(3)]],
    constant uint&      N           [[buffer(4)]],
    constant uint&      K           [[buffer(5)]],
    constant uint&      gate_offset [[buffer(6)]],
    constant uint&      up_offset   [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]])
{
    const uint row = group * 2u + (uint)sg;
    if (row >= N) return;

    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* gate_row = gate_weight + gate_offset + row * (nb * 144u);
    device const uchar* up_row = up_weight + up_offset + row * (nb * 144u);
    device const float* y4 = input + ix * 256u + 64u * iq + 8u * ir;
    float yl[16];
    float yh[16];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        float4 sumy = 0.0f;
        FOR_UNROLL (ushort i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i + 0];   sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i + 32];  sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        float block_sum[2];
        device const uchar* blocks[2] = {
            gate_row + ib * 144u,
            up_row + ib * 144u,
        };
        FOR_UNROLL (ushort matrix = 0u; matrix < 2u; ++matrix) {
            device const uchar* blk = blocks[matrix];
            device const ushort* sc = (device const ushort*)(blk + 4u) + iq;
            device const ushort* q1 = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
            device const ushort* q2 = q1 + 32u;
            device const half* dh = (device const half*)blk;
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            float4 acc1 = 0.0f;
            float4 acc2 = 0.0f;
            FOR_UNROLL (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            block_sum[matrix] =
                (float)dh[0] * ((acc1[0] + acc1[1]/256.0f)*sc8[0]
                              + (acc1[2] + acc1[3]/256.0f)*sc8[1]/16.0f
                              + (acc2[0] + acc2[1]/256.0f)*sc8[4]
                              + (acc2[2] + acc2[3]/256.0f)*sc8[5]/16.0f)
              - (float)dh[1] * dot(sumy, float4(sc8[2], sc8[3], sc8[6], sc8[7]));
        }
        gate_sum += block_sum[0];
        up_sum += block_sum[1];
        y4 += 4u * 256u;
    }

    const float gate = simd_sum(gate_sum);
    const float up = simd_sum(up_sum);
    if (lane == 0u) {
        out[row] = (gate / (1.0f + exp(-gate))) * up;
    }
}
