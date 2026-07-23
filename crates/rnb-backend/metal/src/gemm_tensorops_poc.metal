// pm34 milestone 2: Metal 4 mpp::tensor_ops::matmul2d go/no-go PoC.
// weight-only 경로의 핵심 검증 — A/B 를 threadgroup half 로 staging(실커널에선 A=Q4_K
// dequant, B=activation f32->half) → barrier → tensor_inline matmul2d, dest=device float(f32 accum).
// 비대칭 shape(A[16x32]·B[32x16]=C[16x16])로 transpose/layout 오류를 노출(SF1).
// host 는 f32 buffer 를 그대로 올리고(half 변환은 staging 에서) C f32 를 readback 한다.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// A_dev[16x32] f32 (row-major M x K), B_dev[32x16] f32 (K x N) -> C_dev[16x16] f32 (M x N)
kernel void gemm_tensorops_poc(
    device const float *A_dev [[buffer(0)]],
    device const float *B_dev [[buffer(1)]],
    device float *C_dev [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half A_stage[16 * 32];
    threadgroup half B_stage[32 * 16];
    for (uint i = tid; i < 16 * 32; i += 32) {
        A_stage[i] = (half)A_dev[i];
    }
    for (uint i = tid; i < 32 * 16; i += 32) {
        B_stage[i] = (half)B_dev[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // extents 는 (inner, outer) = (stride-1 축, 바깥 축). row-major 라 inner=열.
    // A[M=16][K=32] row-major → inner=K=32, outer=M=16 → dextents(32, 16).
    // B[K=32][N=16] row-major → inner=N=16, outer=K=32 → dextents(16, 32).
    // C[M=16][N=16] row-major → dextents(16, 16).
    auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        A_stage, dextents<int32_t, 2>(32, 16));
    auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        B_stage, dextents<int32_t, 2>(16, 32));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_dev, dextents<int32_t, 2>(16, 16));

    constexpr auto desc = matmul2d_descriptor(
        16, 16, 32, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<1>> op;
    op.run(A, B, C);
}

// pm34 milestone 3: synthetic Q4_K dequant 좌표 검증.
// weight 16 rows(각 Q4_K 1 superblock K=256) 를 커널 안에서 half dequant → B_stage 에
// transposed(K x N=row) 적재. input[M=tok x K] → A_stage. matmul2d → C[M=tok x N=row].
//   C[tok][row] = sum_k input[tok][k] * dequant(weight[row])[k]  (= cpu_q4k_gemm_reference)
// dequant 비트 인덱싱은 gemm_q4k.metal 1:1. K=256 single descriptor 시도(Task 3 K=32 성공 근거).
kernel void gemm_tensorops_poc_q4k(
    device const uchar *weight_bytes [[buffer(0)]],  // 16 rows * 144 bytes (K=256, 1 superblock)
    device const float *input        [[buffer(1)]],  // [16 x 256] f32 (M tok x K)
    device float       *C_dev        [[buffer(2)]],  // [16 x 16] f32 (M tok x N=weight row)
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half A_stage[16 * 256];  // input (M x K)
    threadgroup half B_stage[256 * 16];  // dequant weight^T (K x N=row)

    for (uint i = tid; i < 16u * 256u; i += 32u) {
        A_stage[i] = (half)input[i];
    }
    // weight dequant: thread t<16 → weight row t (256 elems) → B_stage[k*16 + row] (transposed).
    if (tid < 16u) {
        uint row = tid;
        device const uchar *blk = weight_bytes + row * 144u;
        ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
        ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
        float d    = (float)as_type<half>(d_bits);
        float dmin = (float)as_type<half>(dmin_bits);
        device const uchar *sc = blk + 4;
        float scales_f[8];
        float mins_f[8];
        for (uint j = 0; j < 8u; j++) {
            uchar s, m;
            if (j < 4u) {
                s = sc[j] & 63u;
                m = sc[j + 4u] & 63u;
            } else {
                s = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                m = (sc[j + 4u] >> 4u)   | ((sc[j]       >> 6u) << 4u);
            }
            scales_f[j] = (float)s;
            mins_f[j]   = (float)m;
        }
        device const uchar *qs = blk + 16;
        for (uint g = 0; g < 4u; g++) {
            uint is = g * 2u;
            float d1 = d * scales_f[is];      float m1 = dmin * mins_f[is];
            float d2 = d * scales_f[is + 1u]; float m2 = dmin * mins_f[is + 1u];
            uint q_off = g * 32u;
            uint y_off = g * 64u;
            for (uint l = 0; l < 32u; l++) {
                float q = (float)(qs[q_off + l] & 0x0Fu);
                B_stage[(y_off + l) * 16u + row] = (half)(d1 * q - m1);
            }
            for (uint l = 0; l < 32u; l++) {
                float q = (float)(qs[q_off + l] >> 4u);
                B_stage[(y_off + 32u + l) * 16u + row] = (half)(d2 * q - m2);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A=input[M=16][K=256] → dextents(256,16). B=deqW^T[K=256][N=16] → dextents(16,256).
    auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        A_stage, dextents<int32_t, 2>(256, 16));
    auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        B_stage, dextents<int32_t, 2>(16, 256));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        C_dev, dextents<int32_t, 2>(16, 16));

    constexpr auto desc = matmul2d_descriptor(
        16, 16, 256, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<1>> op;
    op.run(A, B, C);
}

// pm34 milestone 4: 임의 K(256 배수) K-loop multiply_accumulate. M=16 tok / N=16 row tile 고정.
// chunk(256=1 superblock)마다 input+weight 를 staging 재사용 → threadgroup 16KB 고정(K 무관).
// C_dev 는 host 가 0 초기화, 각 chunk 가 multiply_accumulate 로 누적(S2). 실모델 K(4096+) 처리 기반.
kernel void gemm_q4k_tensorops_mn16(
    device const uchar *weight_bytes [[buffer(0)]],  // 16 rows * (K/256)*144 bytes
    device const float *input        [[buffer(1)]],  // 16 tok * K f32
    device float       *C_dev        [[buffer(2)]],  // 16*16 f32 (host zero-init)
    constant uint      &K            [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]])
{
    uint num_blocks = K / 256u;
    threadgroup half A_stage[16 * 256];
    threadgroup half B_stage[256 * 16];

    for (uint b = 0; b < num_blocks; b++) {
        // input chunk: 16 tok × 256 (block b)
        for (uint i = tid; i < 16u * 256u; i += 32u) {
            uint t = i / 256u;
            uint kk = i % 256u;
            A_stage[i] = (half)input[t * K + b * 256u + kk];
        }
        // weight chunk dequant: 16 rows block b → B_stage[k*16 + row] (transposed)
        if (tid < 16u) {
            uint row = tid;
            device const uchar *blk = weight_bytes + (row * num_blocks + b) * 144u;
            ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
            ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
            float d    = (float)as_type<half>(d_bits);
            float dmin = (float)as_type<half>(dmin_bits);
            device const uchar *sc = blk + 4;
            float scales_f[8];
            float mins_f[8];
            for (uint j = 0; j < 8u; j++) {
                uchar s, m;
                if (j < 4u) {
                    s = sc[j] & 63u;
                    m = sc[j + 4u] & 63u;
                } else {
                    s = (sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u);
                    m = (sc[j + 4u] >> 4u)   | ((sc[j]       >> 6u) << 4u);
                }
                scales_f[j] = (float)s;
                mins_f[j]   = (float)m;
            }
            device const uchar *qs = blk + 16;
            for (uint g = 0; g < 4u; g++) {
                uint is = g * 2u;
                float d1 = d * scales_f[is];      float m1 = dmin * mins_f[is];
                float d2 = d * scales_f[is + 1u]; float m2 = dmin * mins_f[is + 1u];
                uint q_off = g * 32u;
                uint y_off = g * 64u;
                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] & 0x0Fu);
                    B_stage[(y_off + l) * 16u + row] = (half)(d1 * q - m1);
                }
                for (uint l = 0; l < 32u; l++) {
                    float q = (float)(qs[q_off + l] >> 4u);
                    B_stage[(y_off + 32u + l) * 16u + row] = (half)(d2 * q - m2);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(256, 16));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(16, 256));
        auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
            C_dev, dextents<int32_t, 2>(16, 16));
        constexpr auto desc = matmul2d_descriptor(
            16, 16, 256, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<1>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// pm34 M(opt): neural accelerator 최적 타일. 출력 BM=64(tok)×BN=32(row), K_chunk=64(=Q4_K
// group 1개), execution_simdgroups<4>(128 thread). threadgroup A 8KB+B 4KB+C 8KB=20KB<32KB.
// 16×16 single-simdgroup(naive 대비 1.9x 느림) 대비 큰 타일+4 simdgroup 으로 NA 잠재력 활용.
//   out[tok*N + row] = sum_k input[tok][k] * dequant(weight[row])[k]
// grid = (ceil(N/32), ceil(M/64)), tg = 128. K_chunk=64 → Q4_K group 단위 dequant.
kernel void gemm_q4k_tensorops(
    device const uchar *weight_bytes [[buffer(0)]],  // N rows * (K/256)*144
    device const float *input        [[buffer(1)]],  // M * K f32
    device float       *out          [[buffer(2)]],  // M * N f32 (row-major)
    constant uint      &N            [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M            [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])  // 0..128
{
    const uint BM = 64u, BN = 32u, KC = 64u;  // KC = 1 Q4_K group (4 chunk / 256-superblock)
    uint row0 = tgid.x * BN;
    uint tok0 = tgid.y * BM;
    uint nb_super = K / 256u;
    uint nchunk = K / KC;

    threadgroup half  A_stage[64 * 64];   // [BM][KC]
    threadgroup half  B_stage[64 * 32];   // [KC][BN] transposed
    threadgroup float C_stage[64 * 32];   // [BM][BN]

    for (uint i = tid; i < BM * BN; i += 128u) {
        C_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < nchunk; c++) {
        // input chunk: BM tok × KC, bound check
        for (uint i = tid; i < BM * KC; i += 128u) {
            uint t = i / KC;
            uint kk = i % KC;
            uint tok = tok0 + t;
            A_stage[i] = (tok < M) ? (half)input[tok * K + c * KC + kk] : (half)0;
        }
        // weight chunk dequant: BN rows × KC(=Q4_K group g), thread t<BN → row t
        if (tid < BN) {
            uint r = tid;
            uint row = row0 + r;
            if (row < N) {
                uint sb = c / 4u;  // 256-superblock
                uint g = c % 4u;   // group(0..4) in superblock
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                // get_scale_min: sub-block is, is+1 (is=g*2)
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;  // group g 의 32 byte
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    B_stage[l * BN + r] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    B_stage[(32u + l) * BN + r] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < KC; k++) {
                    B_stage[k * BN + r] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A[BM,KC], B[KC,BN], C[BM,BN]. extents=(inner,outer).
        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(64, 64));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(32, 64));
        auto C = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            C_stage, dextents<int32_t, 2>(32, 64));
        constexpr auto desc = matmul2d_descriptor(
            64, 32, 64, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < BM * BN; i += 128u) {
        uint t = i / BN;
        uint r = i % BN;
        uint tok = tok0 + t;
        uint row = row0 + r;
        if (tok < M && row < N) {
            out[tok * N + row] = C_stage[t * BN + r];
        }
    }
}

// pm40 M1: llama kernel_mul_mm(tensor) 패턴 채택 — weight(A) threadgroup dequant + activation(B)
//   device-direct tensor + C cooperative tensor(register). 현 gemm_q4k_tensorops(A+B+C threadgroup
//   20KB, C 매 chunk barrier)와 달리 threadgroup 8KB만(weight) → NA occupancy↑, barrier chunk당 1개.
// 매핑(llama 동일): M=N_out(weight rows), N=M_tok(tokens), K=hidden. out[tok*N_out+row](llama dst
//   stride{1,M}과 동일). A=weight[N_out,K] dequant→threadgroup, B=activation[K,M_tok] device f16,
//   C(M=N_out,N=M_tok) cooperative. activation 은 wrapper 가 f32→f16 device 변환해 넘긴다(matmul2d f16 강제).
//   grid=(ceil(M_tok/NRB), ceil(N_out/NRA)), tg=128. dequant 산식은 gemm_q4k_tensorops 1:1.
template<uint NRA, uint NRB, uint NSG>
static void gemm_q4k_v2_tmpl(
    device const uchar *weight_bytes,
    device const half  *input_f16,
    device float       *out,
    constant uint      &N_out,
    constant uint      &K,
    constant uint      &M_tok,
    threadgroup char   *shmem,
    uint2 tgid,
    ushort tid)
{
    const uint NK  = 64u;  // K chunk = 1 Q4_K group
    const uint NUM_THREADS = NSG * 32u;
    uint ra = tgid.y * NRA;  // weight row base
    uint rb = tgid.x * NRB;  // token base
    uint nb_super = K / 256u;
    uint nchunk = K / NK;

    threadgroup half *sa = (threadgroup half *)shmem;  // NRA*NK = 64*64 half = 8KB
    auto tA = tensor(sa, dextents<int32_t, 2>((int)NK, (int)NRA));
    auto tB = tensor((device half *)input_f16, dextents<int32_t, 2>((int)K, (int)M_tok),
                     array<int, 2>({1, (int)K}));

    constexpr auto desc = matmul2d_descriptor(
        NRB, NRA, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (uint c = 0; c < nchunk; c++) {
        uint sb = c / 4u;
        uint g  = c % 4u;
        // weight dequant: NRA rows, 각 row 의 group g(64 elem) → sa[w*NK + ...]. gemm_q4k_tensorops 1:1.
        for (uint w = tid; w < NRA; w += NUM_THREADS) {
            uint row = ra + w;
            if (row < N_out) {
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    sa[w * NK + l] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    sa[w * NK + 32u + l] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < NK; k++) { sa[w * NK + k] = (half)0; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice((int)(c * NK), (int)rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tD = tensor(out, dextents<int32_t, 2>((int)N_out, (int)M_tok), array<int, 2>({1, (int)N_out}));
    cT.store(tD.slice((int)ra, (int)rb));
}

// pm40 M2: 타일/sg variant entry (template instantiation). v2(64×32×4sg, 현)/64×128×4sg(llama)/
//   32×32×1sg(tzakharko 권장). 측정으로 winner 확정. threadgroup = NRA*NK*2(weight only).
#define GEMM_Q4K_V2_ENTRY(NAME, NRA, NRB, NSG)                                         \
    kernel void NAME(                                                                  \
        device const uchar *weight_bytes [[buffer(0)]],                                \
        device const half  *input_f16    [[buffer(1)]],                                \
        device float       *out          [[buffer(2)]],                                \
        constant uint      &N_out        [[buffer(3)]],                                \
        constant uint      &K            [[buffer(4)]],                                \
        constant uint      &M_tok        [[buffer(5)]],                                \
        threadgroup char   *shmem        [[threadgroup(0)]],                           \
        uint2 tgid [[threadgroup_position_in_grid]],                                   \
        ushort tid [[thread_index_in_threadgroup]])                                    \
    {                                                                                  \
        gemm_q4k_v2_tmpl<NRA, NRB, NSG>(                                               \
            weight_bytes, input_f16, out, N_out, K, M_tok, shmem, tgid, tid);          \
    }

// correct winner. NSG=4 고정(matmul2d execution_simdgroups<N>은 타일과 구조적 연동 —
//   NSG≠4 또는 타일이 16×BLOCK×SG 비연동이면 rel=1.0 틀림. 측정 기록은 perf-journal pm41).
GEMM_Q4K_V2_ENTRY(gemm_q4k_tensorops_v2, 64u, 32u, 4u)        // M1 llama 구조(2.96x)
GEMM_Q4K_V2_ENTRY(gemm_q4k_tensorops_v2_64x128, 64u, 128u, 4u) // M2 winner(4.10x), M3 production 타일

// Qwen MoE prefill gate/up pair: 같은 activation tile을 읽는 Q4_K gate/up GEMM 두 dispatch를
// 한 dispatch 안에서 순차 실행한다. weight는 raw Q4_K 그대로 받고, 각 호출 내부 threadgroup
// half tile에서만 transient dequant한다. full gate/up scratch 크기는 기존 경로와 동일하다.
kernel void gemm_q4k_tensorops_v2_pair_64x128(
    device const uchar *gate_weight_bytes [[buffer(0)]],
    device const uchar *up_weight_bytes   [[buffer(1)]],
    device const half  *input_f16         [[buffer(2)]],
    device float       *gate_out          [[buffer(3)]],
    device float       *up_out            [[buffer(4)]],
    constant uint      &N_out             [[buffer(5)]],
    constant uint      &K                 [[buffer(6)]],
    constant uint      &M_tok             [[buffer(7)]],
    threadgroup char   *shmem             [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q4k_v2_tmpl<64u, 128u, 4u>(
        gate_weight_bytes, input_f16, gate_out, N_out, K, M_tok, shmem, tgid, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gemm_q4k_v2_tmpl<64u, 128u, 4u>(
        up_weight_bytes, input_f16, up_out, N_out, K, M_tok, shmem, tgid, tid);
}

// Qwen MoE prefill down path: keep Q4_K raw, but fold the final route-weighted
// scatter into the GEMM store. NRB=32 leaves room for a threadgroup C tile.
template<uint NRA, uint NRB, uint NSG>
static void gemm_q4k_v2_scatter_accum_tmpl(
    device const uchar *weight_bytes,
    device const half  *input_f16,
    device float       *accum_out,
    device const uint  *token_ids,
    device const float *route_weights,
    constant uint      &N_out,
    constant uint      &K,
    constant uint      &M_tok,
    constant uint      &group_start,
    threadgroup char   *shmem,
    uint2 tgid,
    ushort tid)
{
    const uint NK  = 64u;
    const uint NUM_THREADS = NSG * 32u;
    uint ra = tgid.y * NRA;
    uint rb = tgid.x * NRB;
    uint nb_super = K / 256u;
    uint nchunk = K / NK;

    threadgroup half  *sa = (threadgroup half *)shmem;
    threadgroup float *sc = (threadgroup float *)(sa + NRA * NK);
    auto tA = tensor(sa, dextents<int32_t, 2>((int)NK, (int)NRA));
    auto tB = tensor((device half *)input_f16, dextents<int32_t, 2>((int)K, (int)M_tok),
                     array<int, 2>({1, (int)K}));

    constexpr auto desc = matmul2d_descriptor(
        NRB, NRA, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (uint c = 0; c < nchunk; c++) {
        uint sb = c / 4u;
        uint g  = c % 4u;
        for (uint w = tid; w < NRA; w += NUM_THREADS) {
            uint row = ra + w;
            if (row < N_out) {
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 144u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *scales = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = scales[is] & 63u; m0 = scales[is + 4u] & 63u; }
                else { s0 = (scales[is + 4u] & 0x0Fu) | ((scales[is - 4u] >> 6u) << 4u);
                       m0 = (scales[is + 4u] >> 4u) | ((scales[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = scales[i1] & 63u; m1 = scales[i1 + 4u] & 63u; }
                else { s1 = (scales[i1 + 4u] & 0x0Fu) | ((scales[i1 - 4u] >> 6u) << 4u);
                       m1 = (scales[i1 + 4u] >> 4u) | ((scales[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qs = blk + 16 + g * 32u;
                for (uint l = 0; l < 32u; l++) {
                    float ql = (float)(qs[l] & 0x0Fu);
                    sa[w * NK + l] = (half)(d1 * ql - mm1);
                    float qh = (float)(qs[l] >> 4u);
                    sa[w * NK + 32u + l] = (half)(d2 * qh - mm2);
                }
            } else {
                for (uint k = 0; k < NK; k++) { sa[w * NK + k] = (half)0; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice((int)(c * NK), (int)rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor(sc, dextents<int32_t, 2>((int)NRA, (int)NRB));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < NRA * NRB; i += NUM_THREADS) {
        uint local = i / NRA;
        uint row_in_tile = i - local * NRA;
        uint token_local = rb + local;
        uint row = ra + row_in_tile;
        if (token_local < M_tok && row < N_out) {
            uint slot = group_start + token_local;
            uint token = token_ids[slot];
            accum_out[token * N_out + row] += route_weights[slot] * sc[local * NRA + row_in_tile];
        }
    }
}

kernel void gemm_q4k_tensorops_v2_scatter_accum_64x32(
    device const uchar *weight_bytes  [[buffer(0)]],
    device const half  *input_f16     [[buffer(1)]],
    device float       *accum_out     [[buffer(2)]],
    device const uint  *token_ids     [[buffer(3)]],
    device const float *route_weights [[buffer(4)]],
    constant uint      &N_out         [[buffer(5)]],
    constant uint      &K             [[buffer(6)]],
    constant uint      &M_tok         [[buffer(7)]],
    constant uint      &group_start   [[buffer(8)]],
    threadgroup char   *shmem         [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q4k_v2_scatter_accum_tmpl<64u, 32u, 4u>(
        weight_bytes, input_f16, accum_out, token_ids, route_weights,
        N_out, K, M_tok, group_start, shmem, tgid, tid);
}

kernel void gemm_q4k_tensorops_v2_scatter_accum_64x64(
    device const uchar *weight_bytes  [[buffer(0)]],
    device const half  *input_f16     [[buffer(1)]],
    device float       *accum_out     [[buffer(2)]],
    device const uint  *token_ids     [[buffer(3)]],
    device const float *route_weights [[buffer(4)]],
    constant uint      &N_out         [[buffer(5)]],
    constant uint      &K             [[buffer(6)]],
    constant uint      &M_tok         [[buffer(7)]],
    constant uint      &group_start   [[buffer(8)]],
    threadgroup char   *shmem         [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q4k_v2_scatter_accum_tmpl<64u, 64u, 4u>(
        weight_bytes, input_f16, accum_out, token_ids, route_weights,
        N_out, K, M_tok, group_start, shmem, tgid, tid);
}

// pm42 M3 step1: Q5_K v2 — gemm_q4k_v2_tmpl 구조(cooperative C, device-direct B) + Q5_K dequant.
//   NK=64(Q4_K 와 동일 group), 176B/superblock. dequant 은 gemm_q5k_tensorops 1:1(qh high-bit +16).
//   chunk c → superblock c/4, group g=c%4. GDN ssm_out(Q5_K) + 9B GDN projection.
template<uint NRA, uint NRB, uint NSG>
static void gemm_q5k_v2_tmpl(
    device const uchar *weight_bytes,
    device const half  *input_f16,
    device float       *out,
    constant uint      &N_out,
    constant uint      &K,
    constant uint      &M_tok,
    threadgroup char   *shmem,
    uint2 tgid,
    ushort tid)
{
    const uint NK  = 64u;  // K chunk = 1 Q5_K group (= Q4_K)
    const uint NUM_THREADS = NSG * 32u;
    uint ra = tgid.y * NRA;
    uint rb = tgid.x * NRB;
    uint nb_super = K / 256u;
    uint nchunk = K / NK;

    threadgroup half *sa = (threadgroup half *)shmem;  // NRA*NK half = 8KB(NRA=64)
    auto tA = tensor(sa, dextents<int32_t, 2>((int)NK, (int)NRA));
    auto tB = tensor((device half *)input_f16, dextents<int32_t, 2>((int)K, (int)M_tok),
                     array<int, 2>({1, (int)K}));

    constexpr auto desc = matmul2d_descriptor(
        NRB, NRA, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (uint c = 0; c < nchunk; c++) {
        uint sb = c / 4u;
        uint g  = c % 4u;
        for (uint w = tid; w < NRA; w += NUM_THREADS) {
            uint row = ra + w;
            if (row < N_out) {
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 176u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qh = blk + 16;            // 32 byte superblock 전체
                device const uchar *ql = blk + 48 + g * 32u;  // group g 의 32 byte
                uchar u1 = (uchar)(1u << (2u * g));
                uchar u2 = (uchar)(2u << (2u * g));
                for (uint l = 0; l < 32u; l++) {
                    float high1 = (qh[l] & u1) ? 16.0f : 0.0f;
                    float qlow  = (float)(ql[l] & 0x0Fu) + high1;
                    sa[w * NK + l] = (half)(d1 * qlow - mm1);
                    float high2 = (qh[l] & u2) ? 16.0f : 0.0f;
                    float qhigh = (float)(ql[l] >> 4u) + high2;
                    sa[w * NK + 32u + l] = (half)(d2 * qhigh - mm2);
                }
            } else {
                for (uint k = 0; k < NK; k++) { sa[w * NK + k] = (half)0; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice((int)(c * NK), (int)rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tD = tensor(out, dextents<int32_t, 2>((int)N_out, (int)M_tok), array<int, 2>({1, (int)N_out}));
    cT.store(tD.slice((int)ra, (int)rb));
}

// pm42 M3 step1: Q6_K v2 — gemm_q4k_v2_tmpl 구조 + Q6_K dequant. NK=128(Q6_K superblock 절반),
//   210B/superblock. dequant 은 gemm_q6k_tensorops 1:1. chunk c → superblock c/2, half n=c%2(128-half).
//   sa = NRA*128 half = 16KB(NRA=64). FFN down(Q6_K) + GDN in_proj(Q6_K).
template<uint NRA, uint NRB, uint NSG>
static void gemm_q6k_v2_tmpl(
    device const uchar *weight_bytes,
    device const half  *input_f16,
    device float       *out,
    constant uint      &N_out,
    constant uint      &K,
    constant uint      &M_tok,
    threadgroup char   *shmem,
    uint2 tgid,
    ushort tid)
{
    const uint NK  = 128u;  // K chunk = Q6_K superblock 절반(128)
    const uint NUM_THREADS = NSG * 32u;
    uint ra = tgid.y * NRA;
    uint rb = tgid.x * NRB;
    uint nb_super = K / 256u;
    uint nchunk = K / NK;  // = 2 * nb_super

    threadgroup half *sa = (threadgroup half *)shmem;  // NRA*128 half = 16KB(NRA=64)
    auto tA = tensor(sa, dextents<int32_t, 2>((int)NK, (int)NRA));
    auto tB = tensor((device half *)input_f16, dextents<int32_t, 2>((int)K, (int)M_tok),
                     array<int, 2>({1, (int)K}));

    constexpr auto desc = matmul2d_descriptor(
        NRB, NRA, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (uint c = 0; c < nchunk; c++) {
        uint sb = c / 2u;   // 256-superblock
        uint n  = c % 2u;   // half(0..2) in superblock
        for (uint w = tid; w < NRA; w += NUM_THREADS) {
            uint row = ra + w;
            if (row < N_out) {
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 210u;
                device const uchar *ql = blk;                              // 0..127
                device const uchar *qh = blk + 128;                        // 128..191
                device const char  *sc = (device const char *)(blk + 192); // 192..207 (i8)
                ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
                float d = (float)as_type<half>(d_bits);
                uint ql_base = n * 64u;
                uint qh_base = n * 32u;
                uint sc_base = n * 8u;
                for (uint l = 0; l < 32u; l++) {
                    uint is = l / 16u;
                    int q1 = (int)((ql[ql_base + l]       & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u));
                    int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u));
                    int q3 = (int)((ql[ql_base + l]       >> 4u)   | (((qh[qh_base + l] >> 4u) & 3u) << 4u));
                    int q4 = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qh[qh_base + l] >> 6u) & 3u) << 4u));
                    float w1 = d * (float)sc[sc_base + is]      * (float)(q1 - 32);
                    float w2 = d * (float)sc[sc_base + is + 2u] * (float)(q2 - 32);
                    float w3 = d * (float)sc[sc_base + is + 4u] * (float)(q3 - 32);
                    float w4 = d * (float)sc[sc_base + is + 6u] * (float)(q4 - 32);
                    sa[w * NK + l]        = (half)w1;
                    sa[w * NK + l + 32u]  = (half)w2;
                    sa[w * NK + l + 64u]  = (half)w3;
                    sa[w * NK + l + 96u]  = (half)w4;
                }
            } else {
                for (uint k = 0; k < NK; k++) { sa[w * NK + k] = (half)0; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice((int)(c * NK), (int)rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tD = tensor(out, dextents<int32_t, 2>((int)N_out, (int)M_tok), array<int, 2>({1, (int)N_out}));
    cT.store(tD.slice((int)ra, (int)rb));
}

// pm42 M3 step1: Q5_K/Q6_K v2 entry(64×128 winner 타일, 4sg). 매크로(GEMM_Q4K_V2_ENTRY)는
//   gemm_q4k_v2_tmpl hardcode 라 quant 별 직접 instantiate. signature 는 매크로와 동일.
kernel void gemm_q5k_tensorops_v2_64x128(
    device const uchar *weight_bytes [[buffer(0)]],
    device const half  *input_f16    [[buffer(1)]],
    device float       *out          [[buffer(2)]],
    constant uint      &N_out        [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M_tok        [[buffer(5)]],
    threadgroup char   *shmem        [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q5k_v2_tmpl<64u, 128u, 4u>(weight_bytes, input_f16, out, N_out, K, M_tok, shmem, tgid, tid);
}

kernel void gemm_q6k_tensorops_v2_64x128(
    device const uchar *weight_bytes [[buffer(0)]],
    device const half  *input_f16    [[buffer(1)]],
    device float       *out          [[buffer(2)]],
    constant uint      &N_out        [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M_tok        [[buffer(5)]],
    threadgroup char   *shmem        [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q6k_v2_tmpl<64u, 128u, 4u>(weight_bytes, input_f16, out, N_out, K, M_tok, shmem, tgid, tid);
}

// pm123: Q3_K v2 — gemm_q6k_v2_tmpl 구조 + Q3_K dequant. NK=128(256-superblock 절반),
//   110B/superblock. dequant 은 rnb-cpu dequantize_q3_k 1:1(6-bit scale unpack + hmask high bit).
//   chunk c → superblock c/2, half nn=c%2(128-half). sa = NRA*128 half = 16KB(NRA=64).
template<uint NRA, uint NRB, uint NSG>
static void gemm_q3k_v2_tmpl(
    device const uchar *weight_bytes,
    device const half  *input_f16,
    device float       *out,
    constant uint      &N_out,
    constant uint      &K,
    constant uint      &M_tok,
    threadgroup char   *shmem,
    uint2 tgid,
    ushort tid)
{
    const uint NK  = 128u;  // K chunk = Q3_K superblock 절반(128)
    const uint NUM_THREADS = NSG * 32u;
    uint ra = tgid.y * NRA;
    uint rb = tgid.x * NRB;
    uint nb_super = K / 256u;
    uint nchunk = K / NK;  // = 2 * nb_super

    threadgroup half *sa = (threadgroup half *)shmem;  // NRA*128 half = 16KB(NRA=64)
    auto tA = tensor(sa, dextents<int32_t, 2>((int)NK, (int)NRA));
    auto tB = tensor((device half *)input_f16, dextents<int32_t, 2>((int)K, (int)M_tok),
                     array<int, 2>({1, (int)K}));

    constexpr auto desc = matmul2d_descriptor(
        NRB, NRA, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;

    for (uint c = 0; c < nchunk; c++) {
        uint sb = c / 2u;   // 256-superblock
        uint nn = c % 2u;   // half(0..2) in superblock
        for (uint w = tid; w < NRA; w += NUM_THREADS) {
            uint row = ra + w;
            if (row < N_out) {
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 110u;
                device const uchar *hm  = blk;         // 0..31   hmask
                device const uchar *qs  = blk + 32u;   // 32..95  low 2 bits
                device const uchar *scb = blk + 96u;   // 96..107 6-bit scales
                ushort d_bits = (ushort)blk[108] | ((ushort)blk[109] << 8);
                float d = (float)as_type<half>(d_bits);
                // 6-bit scale unpack (rnb-cpu dequantize_q3_k 1:1)
                uint a0 = (uint)scb[0] | ((uint)scb[1] << 8) | ((uint)scb[2] << 16) | ((uint)scb[3] << 24);
                uint a1 = (uint)scb[4] | ((uint)scb[5] << 8) | ((uint)scb[6] << 16) | ((uint)scb[7] << 24);
                uint a2 = (uint)scb[8] | ((uint)scb[9] << 8) | ((uint)scb[10] << 16) | ((uint)scb[11] << 24);
                uint tmp = a2;
                uint scw[4];
                scw[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                scw[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                scw[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
                scw[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
                uint q_off = nn * 32u;
                for (uint j = 0; j < 4u; j++) {
                    uint shift = j * 2u;
                    uchar mbit = (uchar)(1u << (nn * 4u + j));
                    uint is0 = nn * 8u + j * 2u;
                    uint is1 = is0 + 1u;
                    int s0 = (int)(char)((scw[is0 >> 2u] >> ((is0 & 3u) * 8u)) & 0xffu);
                    int s1 = (int)(char)((scw[is1 >> 2u] >> ((is1 & 3u) * 8u)) & 0xffu);
                    float dl0 = d * (float)(s0 - 32);
                    float dl1 = d * (float)(s1 - 32);
                    for (uint l = 0; l < 16u; l++) {
                        int qv0 = (int)((qs[q_off + l] >> shift) & 3u);
                        int hv0 = (hm[l] & mbit) ? 0 : 4;
                        sa[w * NK + j * 32u + l] = (half)(dl0 * (float)(qv0 - hv0));
                        int qv1 = (int)((qs[q_off + l + 16u] >> shift) & 3u);
                        int hv1 = (hm[l + 16u] & mbit) ? 0 : 4;
                        sa[w * NK + j * 32u + 16u + l] = (half)(dl1 * (float)(qv1 - hv1));
                    }
                }
            } else {
                for (uint k = 0; k < NK; k++) { sa[w * NK + k] = (half)0; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice((int)(c * NK), (int)rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tD = tensor(out, dextents<int32_t, 2>((int)N_out, (int)M_tok), array<int, 2>({1, (int)N_out}));
    cT.store(tD.slice((int)ra, (int)rb));
}

kernel void gemm_q3k_tensorops_v2_64x128(
    device const uchar *weight_bytes [[buffer(0)]],
    device const half  *input_f16    [[buffer(1)]],
    device float       *out          [[buffer(2)]],
    constant uint      &N_out        [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M_tok        [[buffer(5)]],
    threadgroup char   *shmem        [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q3k_v2_tmpl<64u, 128u, 4u>(weight_bytes, input_f16, out, N_out, K, M_tok, shmem, tgid, tid);
}

// pm123: Q2_K v2 — gemm_q6k_v2_tmpl 구조 + Q2_K dequant. NK=128(256-superblock 절반),
//   84B/superblock. dequant 은 rnb-cpu dequantize_q2_k 1:1(scale=low4·d, min=high4·dmin, 2-bit qs).
//   chunk c → superblock c/2, half nn=c%2. j=nn*8+jj, q_base=nn*32+(jj&1)*16, shift=(jj>>1)*2.
template<uint NRA, uint NRB, uint NSG>
static void gemm_q2k_v2_tmpl(
    device const uchar *weight_bytes,
    device const half  *input_f16,
    device float       *out,
    constant uint      &N_out,
    constant uint      &K,
    constant uint      &M_tok,
    threadgroup char   *shmem,
    uint2 tgid,
    ushort tid)
{
    const uint NK  = 128u;
    const uint NUM_THREADS = NSG * 32u;
    uint ra = tgid.y * NRA;
    uint rb = tgid.x * NRB;
    uint nb_super = K / 256u;
    uint nchunk = K / NK;

    threadgroup half *sa = (threadgroup half *)shmem;  // NRA*128 half = 16KB(NRA=64)
    auto tA = tensor(sa, dextents<int32_t, 2>((int)NK, (int)NRA));
    auto tB = tensor((device half *)input_f16, dextents<int32_t, 2>((int)K, (int)M_tok),
                     array<int, 2>({1, (int)K}));

    constexpr auto desc = matmul2d_descriptor(
        NRB, NRA, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (uint c = 0; c < nchunk; c++) {
        uint sb = c / 2u;   // 256-superblock
        uint nn = c % 2u;   // half(0..2) in superblock
        for (uint w = tid; w < NRA; w += NUM_THREADS) {
            uint row = ra + w;
            if (row < N_out) {
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 84u;
                device const uchar *scb = blk;         // 0..15  scales(low4=scale,high4=min)
                device const uchar *qs  = blk + 16u;   // 16..79 2-bit quants
                ushort d_bits  = (ushort)blk[80] | ((ushort)blk[81] << 8);
                ushort dm_bits = (ushort)blk[82] | ((ushort)blk[83] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dm_bits);
                for (uint jj = 0; jj < 8u; jj++) {
                    uint sc = (uint)scb[nn * 8u + jj];
                    float scale = d * (float)(sc & 0x0Fu);
                    float mn = dmin * (float)(sc >> 4u);
                    uint qbase = nn * 32u + (jj & 1u) * 16u;
                    uint shift = (jj >> 1u) * 2u;
                    for (uint l = 0; l < 16u; l++) {
                        int q = (int)((qs[qbase + l] >> shift) & 3u);
                        sa[w * NK + jj * 16u + l] = (half)((float)q * scale - mn);
                    }
                }
            } else {
                for (uint k = 0; k < NK; k++) { sa[w * NK + k] = (half)0; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice((int)(c * NK), (int)rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tD = tensor(out, dextents<int32_t, 2>((int)N_out, (int)M_tok), array<int, 2>({1, (int)N_out}));
    cT.store(tD.slice((int)ra, (int)rb));
}

kernel void gemm_q2k_tensorops_v2_64x128(
    device const uchar *weight_bytes [[buffer(0)]],
    device const half  *input_f16    [[buffer(1)]],
    device float       *out          [[buffer(2)]],
    constant uint      &N_out        [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M_tok        [[buffer(5)]],
    threadgroup char   *shmem        [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q2k_v2_tmpl<64u, 128u, 4u>(weight_bytes, input_f16, out, N_out, K, M_tok, shmem, tgid, tid);
}

template<uint NRA, uint NRB, uint NSG>
static void gemm_q6k_v2_scatter_accum_tmpl(
    device const uchar *weight_bytes,
    device const half  *input_f16,
    device float       *accum_out,
    device const uint  *token_ids,
    device const float *route_weights,
    constant uint      &N_out,
    constant uint      &K,
    constant uint      &M_tok,
    constant uint      &group_start,
    threadgroup char   *shmem,
    uint2 tgid,
    ushort tid)
{
    const uint NK  = 128u;
    const uint NUM_THREADS = NSG * 32u;
    uint ra = tgid.y * NRA;
    uint rb = tgid.x * NRB;
    uint nb_super = K / 256u;
    uint nchunk = K / NK;

    threadgroup half  *sa = (threadgroup half *)shmem;
    threadgroup float *sc = (threadgroup float *)(sa + NRA * NK);
    auto tA = tensor(sa, dextents<int32_t, 2>((int)NK, (int)NRA));
    auto tB = tensor((device half *)input_f16, dextents<int32_t, 2>((int)K, (int)M_tok),
                     array<int, 2>({1, (int)K}));

    constexpr auto desc = matmul2d_descriptor(
        NRB, NRA, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (uint c = 0; c < nchunk; c++) {
        uint sb = c / 2u;
        uint n  = c % 2u;
        for (uint w = tid; w < NRA; w += NUM_THREADS) {
            uint row = ra + w;
            if (row < N_out) {
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 210u;
                device const uchar *ql = blk;
                device const uchar *qh = blk + 128;
                device const char  *scale = (device const char *)(blk + 192);
                ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
                float d = (float)as_type<half>(d_bits);
                uint ql_base = n * 64u;
                uint qh_base = n * 32u;
                uint sc_base = n * 8u;
                for (uint l = 0; l < 32u; l++) {
                    uint is = l / 16u;
                    int q1 = (int)((ql[ql_base + l]       & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u));
                    int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u));
                    int q3 = (int)((ql[ql_base + l]       >> 4u)   | (((qh[qh_base + l] >> 4u) & 3u) << 4u));
                    int q4 = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qh[qh_base + l] >> 6u) & 3u) << 4u));
                    float w1 = d * (float)scale[sc_base + is]      * (float)(q1 - 32);
                    float w2 = d * (float)scale[sc_base + is + 2u] * (float)(q2 - 32);
                    float w3 = d * (float)scale[sc_base + is + 4u] * (float)(q3 - 32);
                    float w4 = d * (float)scale[sc_base + is + 6u] * (float)(q4 - 32);
                    sa[w * NK + l]        = (half)w1;
                    sa[w * NK + l + 32u]  = (half)w2;
                    sa[w * NK + l + 64u]  = (half)w3;
                    sa[w * NK + l + 96u]  = (half)w4;
                }
            } else {
                for (uint k = 0; k < NK; k++) { sa[w * NK + k] = (half)0; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice((int)(c * NK), (int)rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC = tensor(sc, dextents<int32_t, 2>((int)NRA, (int)NRB));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < NRA * NRB; i += NUM_THREADS) {
        uint local = i / NRA;
        uint row_in_tile = i - local * NRA;
        uint token_local = rb + local;
        uint row = ra + row_in_tile;
        if (token_local < M_tok && row < N_out) {
            uint slot = group_start + token_local;
            uint token = token_ids[slot];
            accum_out[token * N_out + row] += route_weights[slot] * sc[local * NRA + row_in_tile];
        }
    }
}

kernel void gemm_q6k_tensorops_v2_scatter_accum_64x32(
    device const uchar *weight_bytes  [[buffer(0)]],
    device const half  *input_f16     [[buffer(1)]],
    device float       *accum_out     [[buffer(2)]],
    device const uint  *token_ids     [[buffer(3)]],
    device const float *route_weights [[buffer(4)]],
    constant uint      &N_out         [[buffer(5)]],
    constant uint      &K             [[buffer(6)]],
    constant uint      &M_tok         [[buffer(7)]],
    constant uint      &group_start   [[buffer(8)]],
    threadgroup char   *shmem         [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q6k_v2_scatter_accum_tmpl<64u, 32u, 4u>(
        weight_bytes, input_f16, accum_out, token_ids, route_weights,
        N_out, K, M_tok, group_start, shmem, tgid, tid);
}

kernel void gemm_q6k_tensorops_v2_scatter_accum_64x64(
    device const uchar *weight_bytes  [[buffer(0)]],
    device const half  *input_f16     [[buffer(1)]],
    device float       *accum_out     [[buffer(2)]],
    device const uint  *token_ids     [[buffer(3)]],
    device const float *route_weights [[buffer(4)]],
    constant uint      &N_out         [[buffer(5)]],
    constant uint      &K             [[buffer(6)]],
    constant uint      &M_tok         [[buffer(7)]],
    constant uint      &group_start   [[buffer(8)]],
    threadgroup char   *shmem         [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q6k_v2_scatter_accum_tmpl<64u, 64u, 4u>(
        weight_bytes, input_f16, accum_out, token_ids, route_weights,
        N_out, K, M_tok, group_start, shmem, tgid, tid);
}

// pm42 M3 step2: f32 device buffer → f16 device buffer elementwise cast. v2 GEMM(matmul2d
//   device tensor)은 activation f16 강제 → chain 에서 normed(gate/up 공유) + silu 결과(down)를
//   각 1회 cast. grid=(ceil(n/256),1), tg=256.
kernel void cast_f32_to_f16(
    device const float *src [[buffer(0)]],
    device half        *dst [[buffer(1)]],
    constant uint      &n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) { dst[gid] = (half)src[gid]; }
}

// pm35: Q6_K tensorops 64×32 최적 타일(gemm_q4k_tensorops 구조 차용, dequant 만 Q6_K).
// 출력 BM=64(tok)×BN=32(row), KC=128(Q6_K superblock 절반), execution_simdgroups<4>(128 thread).
// threadgroup A 16KB+B 8KB+C 8KB=32KB(M5 한계 딱). 16×16 single-simdgroup 대비 큰 타일+4
// simdgroup 으로 NA 활용(Q4_K 와 동일 패턴). dequant 은 gemm_q6k.metal 1:1(210B/superblock).
//   out[tok*N + row] = sum_k input[tok][k] * dequant(weight[row])[k]
// grid = (ceil(N/32), ceil(M/64)), tg = 128. chunk c → superblock c/2, half c%2(128-half).
kernel void gemm_q6k_tensorops(
    device const uchar *weight_bytes [[buffer(0)]],  // N rows * (K/256)*210
    device const float *input        [[buffer(1)]],  // M * K f32
    device float       *out          [[buffer(2)]],  // M * N f32 (row-major)
    constant uint      &N            [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M            [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])  // 0..128
{
    const uint BM = 64u, BN = 32u, KC = 128u;  // KC = Q6_K superblock 절반(128)
    uint row0 = tgid.x * BN;
    uint tok0 = tgid.y * BM;
    uint nb_super = K / 256u;
    uint nchunk = K / KC;  // = 2 * nb_super

    threadgroup half  A_stage[64 * 128];  // [BM][KC]
    threadgroup half  B_stage[128 * 32];  // [KC][BN] transposed
    threadgroup float C_stage[64 * 32];   // [BM][BN]

    for (uint i = tid; i < BM * BN; i += 128u) {
        C_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < nchunk; c++) {
        // input chunk: BM tok × KC, bound check
        for (uint i = tid; i < BM * KC; i += 128u) {
            uint t = i / KC;
            uint kk = i % KC;
            uint tok = tok0 + t;
            A_stage[i] = (tok < M) ? (half)input[tok * K + c * KC + kk] : (half)0;
        }
        // weight chunk dequant: BN rows × KC(=Q6_K 128-half), thread t<BN → row t
        if (tid < BN) {
            uint r = tid;
            uint row = row0 + r;
            if (row < N) {
                uint sb = c / 2u;   // 256-superblock
                uint n  = c % 2u;   // half(0..2) in superblock
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 210u;
                device const uchar *ql = blk;                              // 0..127
                device const uchar *qh = blk + 128;                        // 128..191
                device const char  *sc = (device const char *)(blk + 192); // 192..207 (i8)
                ushort d_bits = (ushort)blk[208] | ((ushort)blk[209] << 8);
                float d = (float)as_type<half>(d_bits);
                uint ql_base = n * 64u;
                uint qh_base = n * 32u;
                uint sc_base = n * 8u;
                // chunk-local k = l, l+32, l+64, l+96 (superblock 위치 n*128 + 그 값과 정렬)
                for (uint l = 0; l < 32u; l++) {
                    uint is = l / 16u;
                    int q1 = (int)((ql[ql_base + l]       & 0x0Fu) | (((qh[qh_base + l] >> 0u) & 3u) << 4u));
                    int q2 = (int)((ql[ql_base + l + 32u] & 0x0Fu) | (((qh[qh_base + l] >> 2u) & 3u) << 4u));
                    int q3 = (int)((ql[ql_base + l]       >> 4u)   | (((qh[qh_base + l] >> 4u) & 3u) << 4u));
                    int q4 = (int)((ql[ql_base + l + 32u] >> 4u)   | (((qh[qh_base + l] >> 6u) & 3u) << 4u));
                    float w1 = d * (float)sc[sc_base + is]      * (float)(q1 - 32);
                    float w2 = d * (float)sc[sc_base + is + 2u] * (float)(q2 - 32);
                    float w3 = d * (float)sc[sc_base + is + 4u] * (float)(q3 - 32);
                    float w4 = d * (float)sc[sc_base + is + 6u] * (float)(q4 - 32);
                    B_stage[(l)       * BN + r] = (half)w1;
                    B_stage[(l + 32u) * BN + r] = (half)w2;
                    B_stage[(l + 64u) * BN + r] = (half)w3;
                    B_stage[(l + 96u) * BN + r] = (half)w4;
                }
            } else {
                for (uint k = 0; k < KC; k++) {
                    B_stage[k * BN + r] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A[BM,KC], B[KC,BN], C[BM,BN]. extents=(inner,outer).
        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(128, 64));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(32, 128));
        auto C = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            C_stage, dextents<int32_t, 2>(32, 64));
        constexpr auto desc = matmul2d_descriptor(
            64, 32, 128, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < BM * BN; i += 128u) {
        uint t = i / BN;
        uint r = i % BN;
        uint tok = tok0 + t;
        uint row = row0 + r;
        if (tok < M && row < N) {
            out[tok * N + row] = C_stage[t * BN + r];
        }
    }
}

// pm36: Q5_K tensorops 64×32 최적 타일(gemm_q4k_tensorops 구조 차용, dequant 에 qh high-bit 추가).
// 출력 BM=64(tok)×BN=32(row), KC=64(Q5_K group 1개 = Q4_K 와 동일), execution_simdgroups<4>(128 thread).
// threadgroup A 8KB+B 4KB+C 8KB=20KB(Q4_K 와 동일, 여유). dequant 은 gemv_q5k.metal 1:1(176B/superblock):
//   Q4_K low/high nibble + qh high-bit(+16). group g: u1=1<<2g, u2=2<<2g. qh[l](32 byte, superblock 전체).
//   out[tok*N + row] = sum_k input[tok][k] * dequant(weight[row])[k]
// grid = (ceil(N/32), ceil(M/64)), tg = 128. chunk c → superblock c/4, group g=c%4.
kernel void gemm_q5k_tensorops(
    device const uchar *weight_bytes [[buffer(0)]],  // N rows * (K/256)*176
    device const float *input        [[buffer(1)]],  // M * K f32
    device float       *out          [[buffer(2)]],  // M * N f32 (row-major)
    constant uint      &N            [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M            [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])  // 0..128
{
    const uint BM = 64u, BN = 32u, KC = 64u;  // KC = 1 Q5_K group (= Q4_K 와 동일)
    uint row0 = tgid.x * BN;
    uint tok0 = tgid.y * BM;
    uint nb_super = K / 256u;
    uint nchunk = K / KC;

    threadgroup half  A_stage[64 * 64];   // [BM][KC]
    threadgroup half  B_stage[64 * 32];   // [KC][BN] transposed
    threadgroup float C_stage[64 * 32];   // [BM][BN]

    for (uint i = tid; i < BM * BN; i += 128u) {
        C_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < nchunk; c++) {
        // input chunk: BM tok × KC, bound check
        for (uint i = tid; i < BM * KC; i += 128u) {
            uint t = i / KC;
            uint kk = i % KC;
            uint tok = tok0 + t;
            A_stage[i] = (tok < M) ? (half)input[tok * K + c * KC + kk] : (half)0;
        }
        // weight chunk dequant: BN rows × KC(=Q5_K group g), thread t<BN → row t
        if (tid < BN) {
            uint r = tid;
            uint row = row0 + r;
            if (row < N) {
                uint sb = c / 4u;  // 256-superblock
                uint g = c % 4u;   // group(0..4) in superblock
                device const uchar *blk = weight_bytes + (row * nb_super + sb) * 176u;
                ushort d_bits    = (ushort)blk[0] | ((ushort)blk[1] << 8);
                ushort dmin_bits = (ushort)blk[2] | ((ushort)blk[3] << 8);
                float d    = (float)as_type<half>(d_bits);
                float dmin = (float)as_type<half>(dmin_bits);
                device const uchar *sc = blk + 4;
                // get_scale_min: sub-block is, is+1 (is=g*2)
                uint is = g * 2u;
                uint i1 = is + 1u;
                uchar s0, m0, s1, m1;
                if (is < 4u) { s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; }
                else { s0 = (sc[is + 4u] & 0x0Fu) | ((sc[is - 4u] >> 6u) << 4u);
                       m0 = (sc[is + 4u] >> 4u) | ((sc[is] >> 6u) << 4u); }
                if (i1 < 4u) { s1 = sc[i1] & 63u; m1 = sc[i1 + 4u] & 63u; }
                else { s1 = (sc[i1 + 4u] & 0x0Fu) | ((sc[i1 - 4u] >> 6u) << 4u);
                       m1 = (sc[i1 + 4u] >> 4u) | ((sc[i1] >> 6u) << 4u); }
                float d1 = d * (float)s0;  float mm1 = dmin * (float)m0;
                float d2 = d * (float)s1;  float mm2 = dmin * (float)m1;
                device const uchar *qh = blk + 16;            // 32 byte(superblock 전체)
                device const uchar *ql = blk + 48 + g * 32u;  // group g 의 32 byte
                uchar u1 = (uchar)(1u << (2u * g));
                uchar u2 = (uchar)(2u << (2u * g));
                for (uint l = 0; l < 32u; l++) {
                    float high1 = (qh[l] & u1) ? 16.0f : 0.0f;
                    float qlow  = (float)(ql[l] & 0x0Fu) + high1;
                    B_stage[l * BN + r] = (half)(d1 * qlow - mm1);
                    float high2 = (qh[l] & u2) ? 16.0f : 0.0f;
                    float qhigh = (float)(ql[l] >> 4u) + high2;
                    B_stage[(32u + l) * BN + r] = (half)(d2 * qhigh - mm2);
                }
            } else {
                for (uint k = 0; k < KC; k++) {
                    B_stage[k * BN + r] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A[BM,KC], B[KC,BN], C[BM,BN]. extents=(inner,outer). KC=64 → Q4_K 와 동일.
        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(64, 64));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(32, 64));
        auto C = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            C_stage, dextents<int32_t, 2>(32, 64));
        constexpr auto desc = matmul2d_descriptor(
            64, 32, 64, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < BM * BN; i += 128u) {
        uint t = i / BN;
        uint r = i % BN;
        uint tok = tok0 + t;
        uint row = row0 + r;
        if (tok < M && row < N) {
            out[tok * N + row] = C_stage[t * BN + r];
        }
    }
}

// ===== Q8_0 tensorops GEMM (unsloth UD attn/GDN/shared projection 대응) =====
// Q8_0 block(34B, BlockQ8_0 repr(C)): offset 0-1 d(f16), offset 2-33 qs[32](i8).
// dequant (rnb-cpu dequantize_q8_0 1:1): y[i] = qs[i] * d. superblock 없음 — 32-elem block.
// NK=32(block 1개) 고정: K % 32 == 0 은 Q8_0 quant 가 항상 보장(K%64 가정 회피).
//
// v2 (pm40 llama 패턴, gemm_q4k_v2_tmpl 구조 1:1 — dequant 만 Q8_0): weight threadgroup
// dequant + activation f16 device-direct tensor + C cooperative. threadgroup = NRA*NK half.
template<uint NRA, uint NRB, uint NSG>
static void gemm_q8_0_v2_tmpl(
    device const uchar *weight_bytes,
    device const half  *input_f16,
    device float       *out,
    constant uint      &N_out,
    constant uint      &K,
    constant uint      &M_tok,
    threadgroup char   *shmem,
    uint2 tgid,
    ushort tid)
{
    const uint NK  = 32u;  // Q8_0 block = 32 elems
    const uint NUM_THREADS = NSG * 32u;
    uint ra = tgid.y * NRA;  // weight row base
    uint rb = tgid.x * NRB;  // token base
    uint num_blk = K / 32u;
    uint nchunk = num_blk;   // K / NK

    threadgroup half *sa = (threadgroup half *)shmem;  // NRA*NK = 64*32 half = 4KB
    auto tA = tensor(sa, dextents<int32_t, 2>((int)NK, (int)NRA));
    auto tB = tensor((device half *)input_f16, dextents<int32_t, 2>((int)K, (int)M_tok),
                     array<int, 2>({1, (int)K}));

    constexpr auto desc = matmul2d_descriptor(
        NRB, NRA, NK, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    for (uint c = 0; c < nchunk; c++) {
        uint b = c;  // NK=32 → chunk index == block index
        // weight dequant: NRA rows, block b(32 elem) → sa[w*NK + l]. gemv_q8_0.metal 1:1.
        for (uint w = tid; w < NRA; w += NUM_THREADS) {
            uint row = ra + w;
            if (row < N_out) {
                device const uchar *blk = weight_bytes + (row * num_blk + b) * 34u;
                ushort d_bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
                float d = (float)as_type<half>(d_bits);
                device const char *qs = (device const char *)(blk + 2);
                for (uint l = 0; l < 32u; l++) {
                    sa[w * NK + l] = (half)(d * (float)qs[l]);
                }
            } else {
                for (uint k = 0; k < NK; k++) { sa[w * NK + k] = (half)0; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0);
        auto mB = tB.slice((int)(c * NK), (int)rb);
        mm.run(mB, mA, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tD = tensor(out, dextents<int32_t, 2>((int)N_out, (int)M_tok), array<int, 2>({1, (int)N_out}));
    cT.store(tD.slice((int)ra, (int)rb));
}

kernel void gemm_q8_0_tensorops_v2_64x128(
    device const uchar *weight_bytes [[buffer(0)]],
    device const half  *input_f16    [[buffer(1)]],
    device float       *out          [[buffer(2)]],
    constant uint      &N_out        [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M_tok        [[buffer(5)]],
    threadgroup char   *shmem        [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    gemm_q8_0_v2_tmpl<64u, 128u, 4u>(weight_bytes, input_f16, out, N_out, K, M_tok, shmem, tgid, tid);
}

// v1 (A+B+C threadgroup staging, input f32). gemm_q6k_tensorops 구조 1:1, dequant 만 Q8_0.
// KC=32(Q8_0 block). grid=(ceil(N/32), ceil(M/64)), tg=128. 비-tensorops-capable/OFF fallback.
kernel void gemm_q8_0_tensorops(
    device const uchar *weight_bytes [[buffer(0)]],  // N rows * (K/32)*34
    device const float *input        [[buffer(1)]],  // M * K f32
    device float       *out          [[buffer(2)]],  // M * N f32 (row-major)
    constant uint      &N            [[buffer(3)]],
    constant uint      &K            [[buffer(4)]],
    constant uint      &M            [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])  // 0..128
{
    const uint BM = 64u, BN = 32u, KC = 32u;  // KC = Q8_0 block
    uint row0 = tgid.x * BN;
    uint tok0 = tgid.y * BM;
    uint num_blk = K / 32u;
    uint nchunk = num_blk;  // K / KC

    threadgroup half  A_stage[64 * 32];  // [BM][KC]
    threadgroup half  B_stage[32 * 32];  // [KC][BN] transposed
    threadgroup float C_stage[64 * 32];  // [BM][BN]

    for (uint i = tid; i < BM * BN; i += 128u) {
        C_stage[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < nchunk; c++) {
        for (uint i = tid; i < BM * KC; i += 128u) {
            uint t = i / KC;
            uint kk = i % KC;
            uint tok = tok0 + t;
            A_stage[i] = (tok < M) ? (half)input[tok * K + c * KC + kk] : (half)0;
        }
        if (tid < BN) {
            uint r = tid;
            uint row = row0 + r;
            if (row < N) {
                device const uchar *blk = weight_bytes + (row * num_blk + c) * 34u;
                ushort d_bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
                float d = (float)as_type<half>(d_bits);
                device const char *qs = (device const char *)(blk + 2);
                for (uint l = 0; l < 32u; l++) {
                    B_stage[l * BN + r] = (half)(d * (float)qs[l]);
                }
            } else {
                for (uint k = 0; k < KC; k++) {
                    B_stage[k * BN + r] = (half)0;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto A = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            A_stage, dextents<int32_t, 2>(32, 64));
        auto B = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            B_stage, dextents<int32_t, 2>(32, 32));
        auto C = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
            C_stage, dextents<int32_t, 2>(32, 64));
        constexpr auto desc = matmul2d_descriptor(
            64, 32, 32, false, false, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroups<4>> op;
        op.run(A, B, C);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < BM * BN; i += 128u) {
        uint t = i / BN;
        uint r = i % BN;
        uint tok = tok0 + t;
        uint row = row0 + r;
        if (tok < M && row < N) {
            out[tok * N + row] = C_stage[t * BN + r];
        }
    }
}
