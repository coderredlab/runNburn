#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint BM = 64u;
constant constexpr uint BN = 32u;
constant constexpr uint BK = 64u;
constant constexpr uint THREADS = 128u;

kernel void vision_linear_bf16(
    device const bfloat *weight [[buffer(0)]],
    device const float  *input  [[buffer(1)]],
    device const float  *bias   [[buffer(2)]],
    device float        *output [[buffer(3)]],
    constant uint       &rows   [[buffer(4)]],
    constant uint       &cols   [[buffer(5)]],
    constant uint       &tokens [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint row0 = group.x * BN;
    uint token0 = group.y * BM;
    uint chunks = (cols + BK - 1u) / BK;

    threadgroup float input_stage[BM * BK];
    threadgroup bfloat weight_stage[BN * BK];
    threadgroup float output_stage[BM * BN];

    auto input_tensor = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        input_stage, dextents<int32_t, 2>((int)BK, (int)BM));
    auto weight_tensor = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>(
        weight_stage, dextents<int32_t, 2>((int)BK, (int)BN));
    constexpr auto descriptor = matmul2d_descriptor(
        BM, BN, BK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto result = operation.template get_destination_cooperative_tensor<
        decltype(input_tensor), decltype(weight_tensor), float>();

    for (uint chunk = 0u; chunk < chunks; chunk++) {
        uint col0 = chunk * BK;
        for (uint index = tid; index < BM * BK; index += THREADS) {
            uint local_token = index / BK;
            uint local_col = index % BK;
            uint token = token0 + local_token;
            uint col = col0 + local_col;
            input_stage[index] = token < tokens && col < cols
                ? input[(ulong)token * cols + col]
                : 0.0f;
        }
        for (uint index = tid; index < BN * BK; index += THREADS) {
            uint local_row = index / BK;
            uint local_col = index % BK;
            uint row = row0 + local_row;
            uint col = col0 + local_col;
            weight_stage[index] = row < rows && col < cols
                ? weight[(ulong)row * cols + col]
                : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        operation.run(input_tensor, weight_tensor, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto output_tensor = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        output_stage, dextents<int32_t, 2>((int)BN, (int)BM));
    result.store(output_tensor);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < BM * BN; index += THREADS) {
        uint local_token = index / BN;
        uint local_row = index % BN;
        uint token = token0 + local_token;
        uint row = row0 + local_row;
        if (token < tokens && row < rows) {
            output[(ulong)token * rows + row] = output_stage[index] + bias[row];
        }
    }
}
