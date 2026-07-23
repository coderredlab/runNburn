#include <metal_stdlib>
using namespace metal;

// B-column (weight-amortized) GEMV kernels — milestone 2 (MTP verify amortization).
//
// 목적: weight tile 을 device memory 에서 **1회** 읽어(super-block/block 당 register 로
// 로드) B 개의 입력 컬럼에 재사용한다. decode GEMV 는 memory-bandwidth-bound(weight
// 바이트가 지배)라 weight 를 B 번(per-lane loop) 대신 1번 읽으면 batch-B verify forward 가
// ~B decode 가 아니라 ~1 decode 비용에 근접한다(MTP speculative decode 승리의 핵심).
//
// 접근: llama.cpp nr0=2 coalesced 커널(gemv_q*_coalesced.metal)의 **단일 row** 형태에
// B-컬럼 재사용을 얹었다. nr0=2 는 input 을 2 output row 에 공유(activation reuse)했는데,
// 여기서는 반대로 **weight 를 B input 컬럼에 공유**(weight reuse) — coalesced lane 배치는
// 원본 그대로라 bandwidth 효율(≈peak)을 유지하면서 weight 읽기를 batch-B 로 amortize.
//   - 1 threadgroup(=1 SIMD-group, 32 lane) = 출력 row 1개, grid = N row.
//   - 각 super-block: weight quant/scale/d 를 register 배열에 1회 load → B 컬럼 dot 재사용.
//   - sumf[c] 누적은 super-block 단위(coarse)라 register array 여도 hot-loop spill 없음.
//   - 마지막에 컬럼별 simd_sum → out[c*N + row].
//
// 레이아웃(column-major-by-lane): input=[B*K](컬럼 c=input[c*K..]), out=[B*N].
// 수치는 gemv_q*_coalesced 를 컬럼마다 1회 호출한 것과 동일(rel<1e-3, reduction 순서만).
// B <= BCOL_MAX(8). MTP verify 배치(B~2..4)를 안전하게 덮는다.

constant constexpr uint BCOL_MAX = 8u;

// ---------------------------------------------------------------------------
// Q4_K (144 bytes/super-block, 256 elem)
// ---------------------------------------------------------------------------
kernel void gemv_q4k_bcol(
    device const uchar* weight_bytes      [[buffer(0)]],
    device const float* input             [[buffer(1)]],
    device float*       out               [[buffer(2)]],
    constant uint&      N                 [[buffer(3)]],
    constant uint&      K                 [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      B                 [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;

    const ushort ix = lane / 8u;
    const ushort it = lane % 8u;
    const ushort iq = it / 4u;
    const ushort ir = it % 4u;
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = weight_bytes + weight_byte_offset + row * (nb * 144u);

    float sumf[BCOL_MAX];
    for (uint c = 0; c < BCOL_MAX; c++) sumf[c] = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        // ── weight 를 register 로 1회 load ──
        device const uchar*  blk = x0 + ib * 144u;
        device const ushort* sc  = (device const ushort*)(blk + 4u) + iq;
        device const ushort* q1  = (device const ushort*)(blk + 16u) + 16u * iq + 4u * ir;
        device const ushort* q2  = q1 + 32u;
        device const half*   dh  = (device const half*)blk;
        sc16[0] = sc[0] & kmask1;
        sc16[1] = sc[2] & kmask1;
        sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
        sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
        ushort qv1[4];
        ushort qv2[4];
        for (ushort i = 0; i < 4; ++i) { qv1[i] = q1[i]; qv2[i] = q2[i]; }
        float d0 = (float)dh[0];
        float dm = (float)dh[1];

        // ── B input 컬럼에 재사용 ──
        for (uint c = 0; c < B; c++) {
            device const float* y4 = input + c * K + ib * 256u + 64u * iq + 8u * ir;
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (ushort i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
            }
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (ushort i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (qv1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (qv1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (qv1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (qv1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (qv2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (qv2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (qv2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (qv2[i] & 0xF000);
            }
            sumf[c] += d0 * ((acc1[0] + 1.f/256.f*acc1[1]) * sc8[0] +
                             (acc1[2] + 1.f/256.f*acc1[3]) * sc8[1] * 1.f/16.f +
                             (acc2[0] + 1.f/256.f*acc2[1]) * sc8[4] +
                             (acc2[2] + 1.f/256.f*acc2[3]) * sc8[5] * 1.f/16.f) -
                      dm * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }
    }

    for (uint c = 0; c < B; c++) {
        float total = simd_sum(sumf[c]);
        if (lane == 0) out[c * N + row] = total;
    }
}

// ---------------------------------------------------------------------------
// Q5_K (176 bytes/super-block, 256 elem)
// ---------------------------------------------------------------------------
kernel void gemv_q5k_bcol(
    device const uchar* weight_bytes      [[buffer(0)]],
    device const float* input             [[buffer(1)]],
    device float*       out               [[buffer(2)]],
    constant uint&      N                 [[buffer(3)]],
    constant uint&      K                 [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      B                 [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;

    const ushort ix  = lane % 4u;
    const ushort tid = lane / 4u;
    const ushort iq  = tid / 4u;
    const ushort ir  = tid % 4u;
    const ushort l0       = 8u * ir;
    const ushort q_offset = 32u * iq + l0;
    const ushort y_offset = 64u * iq + l0;
    const uchar hm1 = (uchar)(1u << (2u * iq));
    const uchar hm2 = (uchar)(hm1 << 1);
    const uchar hm3 = (uchar)(hm1 << 4);
    const uchar hm4 = (uchar)(hm2 << 4);
    const uint nb = K / 256u;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* x0 = weight_bytes + weight_byte_offset + row * (nb * 176u);

    float sumf[BCOL_MAX];
    for (uint c = 0; c < BCOL_MAX; c++) sumf[c] = 0.0f;
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (uint ib = ix; ib < nb; ib += 4u) {
        // ── weight 1회 load ──
        device const uchar*  blk = x0 + ib * 176u;
        device const half*   dh  = (device const half*)blk;
        device const ushort* a   = (device const ushort*)(blk + 4u) + iq;
        device const uchar*  q1  = (blk + 48u) + q_offset;
        device const uchar*  qh  = (blk + 16u) + l0;
        device const uchar*  q2  = q1 + 64u;
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
        uchar qv1[8];
        uchar qv2[8];
        uchar qvh[8];
        for (ushort l = 0; l < 8; ++l) { qv1[l] = q1[l]; qv2[l] = q2[l]; qvh[l] = qh[l]; }
        float d0 = (float)dh[0];
        float dm = (float)dh[1];

        for (uint c = 0; c < B; c++) {
            device const float* y1 = input + c * K + ib * 256u + y_offset;
            device const float* y2 = y1 + 128u;
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 8; ++l) {
                yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
                yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
                yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
                yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
            }
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 8; ++l) {
                uchar h = qvh[l];
                acc1[0] += yl[l+0] * (qv1[l] & 0x0F);
                acc1[1] += yl[l+8] * (qv1[l] & 0xF0);
                acc1[2] += yh[l+0] * (qv2[l] & 0x0F);
                acc1[3] += yh[l+8] * (qv2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
            }
            sumf[c] += d0 * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                             sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                             sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                             sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                      dm * (sumy[0]*sc8[2] + sumy[1]*sc8[3] + sumy[2]*sc8[6] + sumy[3]*sc8[7]);
        }
    }

    for (uint c = 0; c < B; c++) {
        float total = simd_sum(sumf[c]);
        if (lane == 0) out[c * N + row] = total;
    }
}

// ---------------------------------------------------------------------------
// Q6_K (210 bytes/super-block, 256 elem)
// ---------------------------------------------------------------------------
kernel void gemv_q6k_bcol(
    device const uchar* weight_bytes      [[buffer(0)]],
    device const float* input             [[buffer(1)]],
    device float*       out               [[buffer(2)]],
    constant uint&      N                 [[buffer(3)]],
    constant uint&      K                 [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      B                 [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const ushort tid = lane / 2u;   // 0..15
    const ushort ix  = lane % 2u;   // 0,1
    const ushort ip  = tid / 8u;    // 0 or 1
    const ushort il  = tid % 8u;    // 0..7
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;
    const uint nb = K / 256u;

    device const uchar* x0 = weight_bytes + weight_byte_offset + row * (nb * 210u);

    float sumf[BCOL_MAX];
    for (uint c = 0; c < BCOL_MAX; c++) sumf[c] = 0.0f;

    for (uint ib = ix; ib < nb; ib += 2u) {
        // ── weight 1회 load ──
        device const uchar* blk = x0 + ib * 210u;
        device const uchar* q1  = (blk + 0u)   + q_offset_l;
        device const uchar* q2  = q1 + 32u;
        device const uchar* qh  = (blk + 128u) + q_offset_h;
        device const char*  sc  = (device const char*)(blk + 192u) + is;
        device const half*  dh  = (device const half*)(blk + 208u);
        uchar qv1[4];
        uchar qv2[4];
        uchar qvh[4];
        for (ushort l = 0; l < 4; ++l) { qv1[l] = q1[l]; qv2[l] = q2[l]; qvh[l] = qh[l]; }
        float sc0 = (float)sc[0];
        float sc2 = (float)sc[2];
        float sc4 = (float)sc[4];
        float sc6 = (float)sc[6];
        float d0 = (float)dh[0];

        for (uint c = 0; c < B; c++) {
            device const float* y = input + c * K + ib * 256u + y_offset;
            float yl[16];
            for (ushort l = 0; l < 4; ++l) {
                yl[4*l + 0] = y[l +  0];
                yl[4*l + 1] = y[l + 32];
                yl[4*l + 2] = y[l + 64];
                yl[4*l + 3] = y[l + 96];
            }
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (ushort l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int)((qv1[l] & 0xF) | ((qvh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int)((qv2[l] & 0xF) | ((qvh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int)((qv1[l]  >> 4) | ((qvh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int)((qv2[l]  >> 4) | ((qvh[l] & kmask4) >> 2)) - 32);
            }
            sumf[c] += d0 * (sums[0]*sc0 + sums[1]*sc2 + sums[2]*sc4 + sums[3]*sc6);
        }
    }

    for (uint c = 0; c < B; c++) {
        float total = simd_sum(sumf[c]);
        if (lane == 0) out[c * N + row] = total;
    }
}

// ---------------------------------------------------------------------------
// Q8_0 (34 bytes/block, 32 elem) — block_elements=32, block_bytes=34
// ---------------------------------------------------------------------------
kernel void gemv_q8_0_bcol(
    device const uchar* weight_bytes      [[buffer(0)]],
    device const float* input             [[buffer(1)]],
    device float*       out               [[buffer(2)]],
    constant uint&      N                 [[buffer(3)]],
    constant uint&      K                 [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      B                 [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;

    constexpr ushort NQ = 8u;
    const uint nb = K / 32u;
    const ushort ix = lane / 4u;   // 0..7
    const ushort il = lane % 4u;   // 0..3

    device const uchar* x0 = weight_bytes + weight_byte_offset + row * (nb * 34u);

    float sumf[BCOL_MAX];
    for (uint c = 0; c < BCOL_MAX; c++) sumf[c] = 0.0f;

    for (uint ib = ix; ib < nb; ib += 8u) {
        // ── weight 1회 load ──
        device const uchar* blk = x0 + ib * 34u;
        ushort d_bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        float d = (float)as_type<half>(d_bits);
        device const char* qs = (device const char*)(blk + 2u) + (uint)il * NQ;
        float qv[NQ];
        for (ushort i = 0; i < NQ; ++i) qv[i] = (float)qs[i];

        for (uint c = 0; c < B; c++) {
            device const float* yb = input + c * K + ib * 32u + (uint)il * NQ;
            float sumq = 0.f;
            for (ushort i = 0; i < NQ; ++i) sumq += qv[i] * yb[i];
            sumf[c] += sumq * d;
        }
    }

    for (uint c = 0; c < B; c++) {
        float total = simd_sum(sumf[c]);
        if (lane == 0) out[c * N + row] = total;
    }
}

// ---------------------------------------------------------------------------
// F32 (row-major dense, arbitrary K) — GDN 의 F32 ssm_alpha/beta/ssm_out(K=z_dim)
// 무손실 B-column amortize. weight_bytes = N*K*4 (row-major, w[row*K+k]).
// K 는 임의(블록 제약 없음) — K=1(rank-1 ssm_out) 도 정상 동작.
// ---------------------------------------------------------------------------
kernel void gemv_f32_bcol(
    device const uchar* weight_bytes      [[buffer(0)]],
    device const float* input             [[buffer(1)]],
    device float*       out               [[buffer(2)]],
    constant uint&      N                 [[buffer(3)]],
    constant uint&      K                 [[buffer(4)]],
    constant uint&      weight_byte_offset [[buffer(5)]],
    constant uint&      B                 [[buffer(6)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    if (row >= N) return;

    device const float* w = (device const float*)(weight_bytes + weight_byte_offset) + (uint)row * K;

    float sumf[BCOL_MAX];
    for (uint c = 0; c < BCOL_MAX; c++) sumf[c] = 0.0f;

    // 각 lane 이 K 를 stride 32 로 분할, weight 원소 1회 load → B 컬럼 재사용.
    for (uint k = lane; k < K; k += 32u) {
        float wv = w[k];
        for (uint c = 0; c < B; c++) {
            sumf[c] += wv * input[c * K + k];
        }
    }

    for (uint c = 0; c < B; c++) {
        float total = simd_sum(sumf[c]);
        if (lane == 0) out[c * N + row] = total;
    }
}
