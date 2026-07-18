// pm46 Phase 1: GDN delta scan STEP5 outer product microbench (R2 게이트).
// STEP5(delta_net_scan_chunk.metal:130-144) 의 state hand-off 누적항:
//   S[vi,ki] += Σ_{j<C} Us[j,vi] * Kk[j,ki]     (Us = rescale_j·u_corr, Kk = k)
//   = C[HV×HK] = Us^T(HV×C) · Kk(C×HK)          (M=HV=128, N=HK=128, K=C=38)
// matmul2d_descriptor 가 컴파일타임 상수 M/N/K 만 받으므로 STEP5 실측 shape 로 고정 특수화한다.
// K=C(=38, runtime c_real)는 KPAD(=48, 16배수)로 staging 0-패딩(matmul2d K%16==0 요구, 결과 불변).
// head = grid(threadgroup) batch. f16 staging + fp32 accumulate(dest=device float).
// scalar 참조(step5_outer_scalar_ref)는 현 커널의 thread-vi-per-row 매핑 그대로 = 격리 등가.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint HV = 128u;    // head_v_dim (27B 실측)
constant constexpr uint HK = 128u;    // head_k_dim (27B 실측)
constant constexpr uint KPAD = 48u;   // ceil(C=38 / 16) * 16 (matmul2d: K must be multiple of 16)
constant constexpr uint S4_CPAD = 48u; // pm47 STEP4 M=c padding (c_real<=48, 16배수)

// matmul2d f16 GEMM. C[h][HV×HK] = Us^T(HV×C) · Kk(C×HK). head = grid.
kernel void gemm_step5_outer_f16(
    device const float* Us     [[buffer(0)]],   // [nh][C_real × HV] row-major
    device const float* Kk     [[buffer(1)]],   // [nh][C_real × HK] row-major
    device float*       Cout   [[buffer(2)]],   // [nh][HV × HK] row-major
    constant uint&      c_real [[buffer(3)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tsz [[threads_per_threadgroup]])
{
    threadgroup half A_stage[HV * KPAD];   // [HV × KPAD] = Us^T (transpose-on-load, pad)
    threadgroup half B_stage[KPAD * HK];   // [KPAD × HK] = Kk (pad)
    uint ub = h * c_real * HV;
    uint kb = h * c_real * HK;

    // A_stage[vi, j] = Us[h][j, vi]  (transpose); j >= c_real → 0
    for (uint i = tid; i < HV * KPAD; i += tsz) {
        uint vi = i / KPAD;
        uint j  = i % KPAD;
        A_stage[i] = (j < c_real) ? (half)Us[ub + j * HV + vi] : (half)0;
    }
    // B_stage[j, ki] = Kk[h][j, ki]; j >= c_real → 0
    for (uint i = tid; i < KPAD * HK; i += tsz) {
        uint j  = i / HK;
        uint ki = i % HK;
        B_stage[i] = (j < c_real) ? (half)Kk[kb + j * HK + ki] : (half)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // row-major extents = (inner=stride1축, outer): A[HV×KPAD]→(KPAD,HV),
    //   B[KPAD×HK]→(HK,KPAD), C[HV×HK]→(HK,HV). (gemm_tensorops_poc.metal:29-38 규칙)
    auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        A_stage, dextents<int32_t, 2>(KPAD, HV));
    auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        B_stage, dextents<int32_t, 2>(HK, KPAD));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        Cout + h * HV * HK, dextents<int32_t, 2>(HK, HV));

    constexpr auto desc = matmul2d_descriptor(
        HV, HK, KPAD, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    op.run(A, B, C);
}

// scalar 참조: 현 delta_net_scan_chunk.metal STEP5(thread vi = state row) 1:1 격리.
kernel void step5_outer_scalar_ref(
    device const float* Us     [[buffer(0)]],   // [nh][C_real × HV]
    device const float* Kk     [[buffer(1)]],   // [nh][C_real × HK]
    device float*       Cout   [[buffer(2)]],   // [nh][HV × HK]
    constant uint&      c_real [[buffer(3)]],
    uint h  [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]])
{
    if (vi >= HV) return;
    uint ub = h * c_real * HV;
    uint kb = h * c_real * HK;
    uint cb = h * HV * HK;
    for (uint ki = 0; ki < HK; ki++) {
        float acc = 0.0f;
        for (uint j = 0; j < c_real; j++) {
            acc += Us[ub + j * HV + vi] * Kk[kb + j * HK + ki];
        }
        Cout[cb + vi * HK + ki] = acc;
    }
}

// pm47 STEP4 inter(S·q dense) microbench (성능 게이트). reviewer 권고 (b): inter 단독 격리.
//   inter[C×HV] = q(C×HK) · state^T(HK×HV)     (M=CPAD=48, N=HV=128, K=HK=128)
// STEP5(outer, scalar 최악)와 달리 STEP4 inter scalar 는 mat-vec(연속 reduction)라 이미 빠를
// 수 있음 → GEMM 이득이 STEP5 21x 만큼 안 날 가능성을 격리 측정해 go/no-go 판정.
// state[h][vi][ki]=32KB(128×128)는 threadgroup 한계 초과 → device workspace 로 transpose staging
// (실제 step45gemm 커널과 동일 비용, staging 포함 end-to-end). reviewer 권고 (c).
kernel void gemm_step4_inter_f16(
    device const float* q       [[buffer(0)]],   // [nh][C_real × HK] row-major
    device const float* state   [[buffer(1)]],   // [nh][HV × HK] row-major (state[vi][ki])
    device float*       Cout    [[buffer(2)]],   // [nh][CPAD × HV] row-major (inter[r][vi])
    constant uint&      c_real  [[buffer(3)]],
    device half*        a_dev   [[buffer(4)]],   // [nh][CPAD × HK] workspace (q, pad)
    device half*        b_dev   [[buffer(5)]],   // [nh][HK × HV] workspace (state^T)
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tsz [[threads_per_threadgroup]])
{
    uint qb = h * c_real * HK;
    uint sb = h * HV * HK;
    uint ab = h * S4_CPAD * HK;
    uint bb = h * HK * HV;
    // a_dev[r, ki] = q[h][r, ki]; r >= c_real → 0
    for (uint i = tid; i < S4_CPAD * HK; i += tsz) {
        uint r = i / HK, ki = i % HK;
        a_dev[ab + i] = (r < c_real) ? (half)q[qb + r * HK + ki] : (half)0;
    }
    // b_dev[ki, vi] = state[h][vi, ki]  (transpose-on-load)
    for (uint i = tid; i < HK * HV; i += tsz) {
        uint ki = i / HV, vi = i % HV;
        b_dev[bb + i] = (half)state[sb + vi * HK + ki];
    }
    threadgroup_barrier(mem_flags::mem_device);
    // extents = (inner=stride1, outer): A[CPAD×HK]→(HK,CPAD), B[HK×HV]→(HV,HK), C[CPAD×HV]→(HV,CPAD).
    auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        a_dev + ab, dextents<int32_t, 2>(HK, S4_CPAD));
    auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
        b_dev + bb, dextents<int32_t, 2>(HV, HK));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        Cout + h * S4_CPAD * HV, dextents<int32_t, 2>(HV, S4_CPAD));
    constexpr auto desc = matmul2d_descriptor(
        S4_CPAD, HV, HK, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<4>> op;
    op.run(A, B, C);
}

// scalar 참조: 현 delta_net_scan_chunk STEP4 inter(thread vi = state row, mat-vec) 1:1 격리.
//   inter[r][vi] = Σ_ki state[vi,ki]·q[r,ki]   (γ_r·intra 제외, 순수 inter)
kernel void step4_inter_scalar_ref(
    device const float* q       [[buffer(0)]],   // [nh][C_real × HK]
    device const float* state   [[buffer(1)]],   // [nh][HV × HK]
    device float*       Cout    [[buffer(2)]],   // [nh][CPAD × HV]
    constant uint&      c_real  [[buffer(3)]],
    uint h  [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]])
{
    if (vi >= HV) return;
    uint qb = h * c_real * HK;
    uint sb = h * HV * HK;
    uint cb = h * S4_CPAD * HV;
    for (uint r = 0; r < c_real; r++) {
        float acc = 0.0f;
        for (uint ki = 0; ki < HK; ki++) {
            acc += state[sb + vi * HK + ki] * q[qb + r * HK + ki];
        }
        Cout[cb + r * HV + vi] = acc;
    }
}
