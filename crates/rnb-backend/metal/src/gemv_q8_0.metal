#include <metal_stdlib>
using namespace metal;

// Q8_0 block layout (34 bytes, BlockQ8_0 repr(C)):
//   offset 0-1 : d (f16 little-endian)
//   offset 2-33: qs[32] (i8 signed quants)
// Dequant (rnb-cpu dequantize_q8_0 1:1): y[i] = qs[i] * d.
// Kernel: thread 1개 = 출력 row 1개. weight_byte_offset = NoCopy page 내 시작.
kernel void gemv_q8_0(
    device const uchar* weight_bytes      [[buffer(0)]],  // N * (K/32 * 34) bytes
    device const float* input             [[buffer(1)]],  // K f32
    device float*       out               [[buffer(2)]],  // N f32
    constant uint&      N                 [[buffer(3)]],
    constant uint&      K                 [[buffer(4)]],   // K = num_blocks * 32
    constant uint&      weight_byte_offset [[buffer(5)]],
    uint                row               [[thread_position_in_grid]])
{
    if (row >= N) return;
    uint num_blocks = K / 32u;
    float acc = 0.0f;
    for (uint b = 0; b < num_blocks; b++) {
        device const uchar* blk =
            weight_bytes + weight_byte_offset + (row * num_blocks + b) * 34u;
        ushort d_bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        float d = (float)as_type<half>(d_bits);
        device const char* qs = (device const char*)(blk + 2);
        uint x_base = b * 32u;
        for (uint i = 0; i < 32u; i++) {
            acc += d * (float)qs[i] * input[x_base + i];
        }
    }
    out[row] = acc;
}
