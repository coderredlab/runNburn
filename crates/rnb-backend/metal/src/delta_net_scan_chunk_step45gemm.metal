// pm149: pm47 STEP4+STEP5 GEMM에 STEP2 initial-state prediction을 추가한다.
// 각 row가 반복하던 γ_r·(S_init·k_r)를 chunk당 state×K TensorOps GEMM으로 계산하고,
// PRECOMPUTE와 triangular correction은 기존 f32 순서를 유지한다.
//
// STEP4: o[c×hv] = [q_scaled | W](c×KPAD) · [state^T ; u_corr](KPAD×hv)   (M=CPAD48, N=hv128, K=KPAD176)
//   A_cat[r, 0:hk]   = exp(g_cum[r])·q[r,:]                      (inter, q_scaled)
//   A_cat[r, hk:hk+c]= qk_sh[r·c+j]·exp(g_cum[r]-g_cum[j]) (j≤r) (intra, W)
//   B_cat[0:hk, vi]  = state[vi,:]   (OLD state, transpose-on-load)
//   B_cat[hk:hk+c,vi]= u_corr[j,vi]
//   → step4_temp(device f32) → out copy(r<c). M=c 가변이라 CPAD=48 패딩(r≥c 행 0, out copy 제외).
// STEP5: state = γ_C·state + scaled_u^T·k   (step5gemm 와 동일).
//
// threadgroup 32KiB 한계(u_corr+kk_sh+qk_sh) + state 32KB → STEP2/4/5 staging 전부
// 기존 device workspace를 재사용한다. matmul2d descriptor가 compile-time 상수라
// hv=hk=128, cs≤48 고정이며 f32 oracle과 chunk drift를 계속 대조한다.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr uint S5_HV   = 128u;  // head_v_dim (27B/9B 고정)
constant constexpr uint S5_HK   = 128u;  // head_k_dim (27B/9B 고정)
constant constexpr uint S5_KPAD = 48u;   // STEP5 K = ceil(cs/16)*16, cs<=48
constant constexpr uint S4_CPAD = 48u;   // STEP4 M = c padding (cs<=48, 16배수)
constant constexpr uint S4_KPAD = 176u;  // STEP4 K = hk(128) + CPAD(48), 16배수(11*16)

kernel void delta_net_scan_chunk_step45gemm(
    device const float* q          [[buffer(0)]],
    device const float* k          [[buffer(1)]],
    device const float* v          [[buffer(2)]],
    device const float* gate       [[buffer(3)]],
    device const float* beta       [[buffer(4)]],
    device float*       state      [[buffer(5)]],
    device float*       out        [[buffer(6)]],
    constant uint&      seq_len    [[buffer(7)]],
    constant uint&      head_k_dim [[buffer(8)]],
    constant uint&      head_v_dim [[buffer(9)]],
    constant uint&      chunk_size [[buffer(10)]],
    constant uint&      num_heads  [[buffer(11)]],
    device half*        su_half    [[buffer(12)]],  // [nh * S5_HV * S5_KPAD]  (STEP5 A)
    device half*        kh_half    [[buffer(13)]],  // [nh * S5_KPAD * S5_HK]  (STEP5 B)
    device float*       temp_dev   [[buffer(14)]],  // [nh * S5_HV * S5_HK]    (STEP5 C)
    device half*        a_cat_half [[buffer(15)]],  // [nh * S4_CPAD * S4_KPAD] (STEP4 A=[q_scaled|W])
    device half*        b_cat_half [[buffer(16)]],  // [nh * S4_KPAD * S5_HV]   (STEP4 B=[state^T;u_corr])
    device float*       step4_temp [[buffer(17)]],  // [nh * S4_CPAD * S5_HV]   (STEP4 C)
    threadgroup float*  u_corr     [[threadgroup(0)]],
    threadgroup float*  kk_sh      [[threadgroup(1)]],
    threadgroup float*  qk_sh      [[threadgroup(2)]],
    uint h       [[threadgroup_position_in_grid]],
    uint vi      [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float g_cum[256];
    uint hk = head_k_dim;
    uint hv = head_v_dim;
    uint s_base = h * hv * hk;

    uint t0 = 0;
    while (t0 < seq_len) {
        uint c = min(chunk_size, seq_len - t0);
        uint base = (t0 * num_heads + h);

        // STEP 1 (동일)
        if (vi == 0) {
            float acc = 0.0f;
            for (uint r = 0; r < c; r++) { acc += gate[(t0 + r) * num_heads + h]; g_cum[r] = acc; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PRECOMPUTE (동일)
        for (uint p = vi; p < c * c; p += tg_size) {
            uint a = p / c, b = p % c;
            uint ka_base = ((t0 + a) * num_heads + h) * hk;
            uint kb_base = ((t0 + b) * num_heads + h) * hk;
            float kk = 0.0f, qk = 0.0f;
            for (uint d = 0; d < hk; d++) {
                float kb = k[kb_base + d];
                kk += k[ka_base + d] * kb;
                qk += q[ka_base + d] * kb;
            }
            kk_sh[p] = kk;
            qk_sh[p] = qk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint pred_a_base = h * S4_KPAD * S5_HV;
        uint pred_b_base = h * S4_CPAD * S4_KPAD;
        uint pred_c_base = h * S4_CPAD * S5_HV;
        // STEP2의 γ_r·(S_init·k_r)를 128개 row별 scalar dot 대신 한 번의
        // 128×128 · 128×48 TensorOps GEMM으로 계산한다. FLA의 current
        // chunk path와 같은 state×K chunk projection이며, 뒤의 triangular
        // correction은 기존 f32 순서를 유지한다. STEP4가 나중에 같은
        // workspaces를 덮어쓰므로 추가 device allocation은 없다.
        for (uint p = vi; p < S5_HV * S5_HK; p += tg_size) {
            b_cat_half[pred_a_base + p] = (half)state[s_base + p];
        }
        for (uint p = vi; p < S5_HK * S4_CPAD; p += tg_size) {
            uint ki = p / S4_CPAD;
            uint r = p % S4_CPAD;
            half value = (half)0;
            if (r < c) {
                uint kr_base = (base + r * num_heads) * hk;
                value = (half)k[kr_base + ki];
            }
            a_cat_half[pred_b_base + p] = value;
        }
        threadgroup_barrier(mem_flags::mem_device);
        {
            auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
                b_cat_half + pred_a_base, dextents<int32_t, 2>(S5_HK, S5_HV));
            auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
                a_cat_half + pred_b_base, dextents<int32_t, 2>(S4_CPAD, S5_HK));
            auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
                step4_temp + pred_c_base, dextents<int32_t, 2>(S4_CPAD, S5_HV));
            constexpr auto desc = matmul2d_descriptor(
                S5_HV, S4_CPAD, S5_HK, false, false, false,
                matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;
            op.run(A, B, C);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // STEP 2 (동일)
        for (uint r = 0; r < c; r++) {
            if (vi < hv) {
                float gr = exp(g_cum[r]);
                float br = beta[(t0 + r) * num_heads + h];
                float pred = step4_temp[pred_c_base + vi * S4_CPAD + r];
                float a = v[(base + r * num_heads) * hv + vi] - gr * pred;
                for (uint i = 0; i < r; i++) {
                    float s_kk = kk_sh[i * c + r] * exp(g_cum[r] - g_cum[i]);
                    a -= s_kk * u_corr[i * hv + vi];
                }
                u_corr[r * hv + vi] = br * a;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // STEP 4 (GEMM, OLD state 읽음): o[c×hv] = [q_scaled|W] · [state^T;u_corr].
        uint a_base = h * S4_CPAD * S4_KPAD;
        uint b_base = h * S4_KPAD * S5_HV;
        // (a) A_cat[r, k2] staging (vi-thread 분담). q_scaled(k2<hk) + W(hk<=k2<hk+c, j≤r) + 패딩 0.
        for (uint p = vi; p < S4_CPAD * S4_KPAD; p += tg_size) {
            uint r = p / S4_KPAD;
            uint k2 = p % S4_KPAD;
            half val = (half)0;
            if (r < c) {
                if (k2 < hk) {
                    float gr = exp(g_cum[r]);
                    uint qr = (base + r * num_heads) * hk + k2;
                    val = (half)(gr * q[qr]);
                } else if (k2 < hk + c) {
                    uint j = k2 - hk;
                    if (j <= r) val = (half)(qk_sh[r * c + j] * exp(g_cum[r] - g_cum[j]));
                }
            }
            a_cat_half[a_base + p] = val;
        }
        // (b) B_cat[k2, vcol] staging (vi-thread 분담). state^T(k2<hk, transpose) + u_corr(hk<=k2<hk+c) + 패딩 0.
        for (uint p = vi; p < S4_KPAD * S5_HV; p += tg_size) {
            uint k2 = p / S5_HV;
            uint vcol = p % S5_HV;
            half val = (half)0;
            if (k2 < hk) {
                val = (half)state[s_base + vcol * hk + k2];   // state[vcol][k2] → b_cat[k2][vcol]
            } else if (k2 < hk + c) {
                uint j = k2 - hk;
                val = (half)u_corr[j * hv + vcol];
            }
            b_cat_half[b_base + p] = val;
        }
        threadgroup_barrier(mem_flags::mem_device); // device staging write → matmul read 가시

        // (c) matmul: step4_temp[CPAD×hv] = A_cat(CPAD×KPAD) · B_cat(KPAD×hv).
        //   extents=(inner,outer): A→(KPAD,CPAD), B→(hv,KPAD), C→(hv,CPAD).
        {
            auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
                a_cat_half + a_base, dextents<int32_t, 2>(S4_KPAD, S4_CPAD));
            auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
                b_cat_half + b_base, dextents<int32_t, 2>(S5_HV, S4_KPAD));
            auto Cmat = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
                step4_temp + h * S4_CPAD * S5_HV, dextents<int32_t, 2>(S5_HV, S4_CPAD));
            constexpr auto desc = matmul2d_descriptor(
                S4_CPAD, S5_HV, S4_KPAD, false, false, false, matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;
            op.run(A, B, Cmat);
        }
        threadgroup_barrier(mem_flags::mem_device); // matmul write → out copy read 가시

        // (d) out copy: out[r,vi] = step4_temp[r,vi] (r<c).
        if (vi < hv) {
            for (uint r = 0; r < c; r++) {
                out[(base + r * num_heads) * hv + vi] = step4_temp[h * S4_CPAD * S5_HV + r * S5_HV + vi];
            }
        }
        threadgroup_barrier(mem_flags::mem_device); // STEP4 OLD-state read 완료 후 STEP5 state write

        // STEP 5 (GEMM): γ_C·state + scaled_u^T·k. (step5gemm 동일)
        float g_last = g_cum[c - 1];
        float gc = exp(g_last);
        uint su_base = h * S5_HV * S5_KPAD;
        uint kh_base = h * S5_KPAD * S5_HK;
        if (vi < hv) {
            for (uint j = 0; j < c; j++) {
                float rescale = exp(g_last - g_cum[j]);
                su_half[su_base + vi * S5_KPAD + j] = (half)(rescale * u_corr[j * hv + vi]);
            }
            for (uint j = c; j < S5_KPAD; j++) su_half[su_base + vi * S5_KPAD + j] = (half)0;
        }
        for (uint ki = vi; ki < hk; ki += tg_size) {
            for (uint j = 0; j < c; j++) {
                uint kj = (base + j * num_heads) * hk;
                kh_half[kh_base + j * S5_HK + ki] = (half)k[kj + ki];
            }
            for (uint j = c; j < S5_KPAD; j++) kh_half[kh_base + j * S5_HK + ki] = (half)0;
        }
        threadgroup_barrier(mem_flags::mem_device);

        {
            auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
                su_half + su_base, dextents<int32_t, 2>(S5_KPAD, S5_HV));
            auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
                kh_half + kh_base, dextents<int32_t, 2>(S5_HK, S5_KPAD));
            auto Ct = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
                temp_dev + s_base, dextents<int32_t, 2>(S5_HK, S5_HV));
            constexpr auto desc = matmul2d_descriptor(
                S5_HV, S5_HK, S5_KPAD, false, false, false, matmul2d_descriptor::mode::multiply);
            matmul2d<desc, execution_simdgroups<4>> op;
            op.run(A, B, Ct);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // state = γ_C·state + temp
        if (vi < hv) {
            for (uint ki = 0; ki < hk; ki++) {
                state[s_base + vi * hk + ki] = gc * state[s_base + vi * hk + ki] + temp_dev[s_base + vi * hk + ki];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        t0 += c;
    }
}

