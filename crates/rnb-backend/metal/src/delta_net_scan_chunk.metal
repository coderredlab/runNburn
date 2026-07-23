#include <metal_stdlib>
using namespace metal;

// GDN delta_net chunkwise parallel scan, prefill (seq_len > 1).
// rnb-cpu kernels/delta_net.rs delta_net_scan_chunkwise(M1 oracle) 와 token-identical
// (f32, 같은 reduction 순서 j 오름차순). decode 1-step delta_net_step.metal 의 chunk 확장.
//
// 매핑:
//   threadgroup 1개 = v-head 1개 (grid = num_heads).
//   thread vi 1개   = state row vi 1개 (tg = head_v_dim). state[h,vi,*] 는 thread vi 전용 → race 없음.
//   head 독립, vi(head_v_dim) 완전 병렬. chunk 시간축(r)만 sequential (forward substitution).
//   GQA: caller(prefill repeat_qk_for_value_heads)가 이미 q/k 를 num_heads 로 repeat → 커널 내 분기 없음.
//
// 알고리즘 (delta_net_scan_chunkwise 1:1):
//   STEP1: g_cum[r] = Σ_{m≤r} gate_m  (chunk-local log-cumsum, thread 0).
//   PRECOMPUTE: KKᵀ(kk_sh[i*c+r]=k_i·k_r) / QKᵀ(qk_sh[r*c+j]=q_r·k_j). vi 무관 → thread 협력 1회.
//     (1st PoC 는 vi 마다 redundant dot 이라 production CPU 보다 2.1x 느렸음 → precompute 로 제거.)
//   STEP2: forward substitution (lower-tri solve). r sequential, vi 병렬.
//     d_r = β_r·(v_r − γ_r·(S·k_r) − Σ_{i<r} s_kk[i]·d_i),  s_kk[i]=kk_sh[i*c+r]·exp(G_r−G_i), γ_r=exp(G_r).
//     교차항에 보정된 d_i(=u_corr[i]) 사용(S_init 병합). u_corr[r] = d_r.
//   STEP4: o_r = γ_r·(S·q_r) + Σ_{j≤r} qk_sh[r*c+j]·exp(G_r−G_j)·u_corr[j]   (OLD state, STEP5 전).
//   STEP5: S ← γ_C·S + Σ_j exp(G_last−G_j)·u_corr[j]·k_jᵀ   (output 계산 후, in-place).
//
// decay 는 항상 상대형 exp(G_r−G_j), j≤r (≤1, underflow 안전). 절대 exp(G) 누적 금지.
//   pred(S·k_r) / inter(S·q_r) / STEP5 outer 는 state 가 vi 별이라 precompute 불가 → dot 유지(중복 아님).
//
// threadgroup dynamic memory:
//   index0 u_corr [chunk_size*head_v_dim], index1 kk_sh [chunk_size*chunk_size],
//   index2 qk_sh [chunk_size*chunk_size]. g_cum 은 정적 [256] (chunk_size ≤ 256, wrapper assert).
kernel void delta_net_scan_chunk(
    device const float* q          [[buffer(0)]],   // [seq*num_heads*head_k_dim]
    device const float* k          [[buffer(1)]],   // [seq*num_heads*head_k_dim]
    device const float* v          [[buffer(2)]],   // [seq*num_heads*head_v_dim]
    device const float* gate       [[buffer(3)]],   // [seq*num_heads]
    device const float* beta       [[buffer(4)]],   // [seq*num_heads]
    device float*       state      [[buffer(5)]],   // [num_heads*head_v_dim*head_k_dim] in-place
    device float*       out        [[buffer(6)]],   // [seq*num_heads*head_v_dim]
    constant uint&      seq_len    [[buffer(7)]],
    constant uint&      head_k_dim [[buffer(8)]],
    constant uint&      head_v_dim [[buffer(9)]],
    constant uint&      chunk_size [[buffer(10)]],
    constant uint&      num_heads  [[buffer(11)]],
    threadgroup float*  u_corr     [[threadgroup(0)]], // [chunk_size*head_v_dim]
    threadgroup float*  kk_sh      [[threadgroup(1)]], // [chunk_size*chunk_size] k_i·k_r
    threadgroup float*  qk_sh      [[threadgroup(2)]], // [chunk_size*chunk_size] q_r·k_j
    uint h       [[threadgroup_position_in_grid]],     // v-head
    uint vi      [[thread_position_in_threadgroup]],   // state row vi (= thread index)
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float g_cum[256];

    uint hk = head_k_dim;
    uint hv = head_v_dim;
    uint s_base = h * hv * hk;

    uint t0 = 0;
    while (t0 < seq_len) {
        uint c = min(chunk_size, seq_len - t0);
        uint base = (t0 * num_heads + h); // token t0 의 (heads,*) base 단위(아래서 *hk/*hv).

        // STEP 1: chunk-local 누적 log-decay G[r] = Σ_{m≤r} gate_m (thread 0).
        if (vi == 0) {
            float acc = 0.0f;
            for (uint r = 0; r < c; r++) {
                acc += gate[(t0 + r) * num_heads + h];
                g_cum[r] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // PRECOMPUTE: kk_sh[a*c+b]=k_a·k_b, qk_sh[a*c+b]=q_a·k_b. thread 가 (a,b) 페어 분담.
        //   dot 은 d 오름차순(oracle 과 동일 reduction). vi 무관이라 중복 제거.
        for (uint p = vi; p < c * c; p += tg_size) {
            uint a = p / c;
            uint b = p % c;
            uint ka_base = ((t0 + a) * num_heads + h) * hk;
            uint kb_base = ((t0 + b) * num_heads + h) * hk;
            float kk = 0.0f;
            float qk = 0.0f;
            for (uint d = 0; d < hk; d++) {
                float kb = k[kb_base + d];
                kk += k[ka_base + d] * kb;
                qk += q[ka_base + d] * kb;
            }
            kk_sh[p] = kk;
            qk_sh[p] = qk;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // STEP 2: WY forward substitution + S_init 보정(한 recurrence). r sequential.
        for (uint r = 0; r < c; r++) {
            if (vi < hv) {
                float gr = exp(g_cum[r]);
                float br = beta[(t0 + r) * num_heads + h];
                uint kr_base = (base + r * num_heads) * hk;
                // pred = γ_r·(S_init[vi,:] · k_r) — state 가 vi 별이라 dot 유지.
                float pred = 0.0f;
                for (uint ki = 0; ki < hk; ki++) {
                    pred += state[s_base + vi * hk + ki] * k[kr_base + ki];
                }
                float a = v[(base + r * num_heads) * hv + vi] - gr * pred;
                // − Σ_{i<r} s_kk[i]·u_corr[i,vi], s_kk[i]=kk_sh[i*c+r]·exp(G_r−G_i)
                for (uint i = 0; i < r; i++) {
                    float s_kk = kk_sh[i * c + r] * exp(g_cum[r] - g_cum[i]);
                    a -= s_kk * u_corr[i * hv + vi];
                }
                u_corr[r * hv + vi] = br * a;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); // u_corr[r] → r+1 가시
        }

        // STEP 4: output o_r = γ_r·(S·q_r) + Σ_{j≤r} qk_sh[r*c+j]·exp(G_r−G_j)·u_corr[j].
        // OLD state(STEP5 전)를 thread vi 전용으로 읽는다.
        if (vi < hv) {
            for (uint r = 0; r < c; r++) {
                float gr = exp(g_cum[r]);
                uint qr_base = (base + r * num_heads) * hk;
                float inter = 0.0f;
                for (uint ki = 0; ki < hk; ki++) {
                    inter += state[s_base + vi * hk + ki] * q[qr_base + ki];
                }
                float o = inter * gr;
                for (uint j = 0; j <= r; j++) {
                    o += qk_sh[r * c + j] * exp(g_cum[r] - g_cum[j]) * u_corr[j * hv + vi];
                }
                out[(base + r * num_heads) * hv + vi] = o;
            }
        }

        // STEP 5: state hand-off S ← γ_C·S + Σ_j exp(G_last−G_j)·u_corr[j]·k_jᵀ (output 후).
        // state[vi,:] 는 thread vi 전용이라 STEP4(read OLD)→STEP5(write)는 thread 내 순차로 보장.
        if (vi < hv) {
            float g_last = g_cum[c - 1];
            float gc = exp(g_last);
            for (uint ki = 0; ki < hk; ki++) {
                float a = gc * state[s_base + vi * hk + ki];
                for (uint j = 0; j < c; j++) {
                    uint kj_base = (base + j * num_heads) * hk;
                    float rescale = exp(g_last - g_cum[j]);
                    a += rescale * u_corr[j * hv + vi] * k[kj_base + ki];
                }
                state[s_base + vi * hk + ki] = a;
            }
        }
        // chunk 경계: 다음 chunk STEP1(g_cum)/precompute(kk_sh,qk_sh)/STEP2(u_corr) write 전에
        // 이 chunk 의 read 완료 보장.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        t0 += c;
    }
}
