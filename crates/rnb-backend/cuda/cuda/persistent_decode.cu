// Persistent decode kernel — single cooperative launch processes the entire
// decode forward pass for Gemma4 E2B.
//
// M3 (this revision): all decode phases implemented as device functions and
// wired into a single cooperative kernel.  Phase layout per layer:
//
//   p0  attn_norm                                    (single block, n_embd)
//   p1a/b/c   Q / K / V projection                   (cooperative Q4K GEMV)
//   p2  QK RoPE NEOX                                 (1 block per head)
//   p3  KV cache write (f32 → f16, append at kv_len) (1 block)
//   p4  attention decode (hd256 SWA / hd512 FULL)    (1 block per head)
//   p5  O projection + post_attn_norm + residual + ffn_norm
//   p6  gate + up + GELU·mul + down + post_ffn_norm + residual
//   p7  PLE gate (F32 cooperative GEMV) + GELU·mul × ple_input + proj +
//       post_norm + residual + out_scale
//
// Final phase: p_out output logits (Q8_0 cooperative GEMV) + argmax.
//
// All phases share scratch buffers in `PersistentDecodeParams`; correctness
// depends on the host pre-sizing those buffers to `max(q_dim, n_ff)` etc.
//
// Note on register pressure: every phase is `__noinline__ __device__` so the
// PTX compiler reuses register windows between phases.  Without this the
// kernel exceeds 255 registers and cooperative launch occupancy hits 0.

#include <cuda_fp16.h>
#include <cooperative_groups.h>

#include "kernels/persistent_decode.cuh"

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// p0: RMS norm (single block).
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_rms_norm(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    float* __restrict__ out,
    unsigned int n,
    float eps,
    int unit_offset) {
    extern __shared__ float smem[];
    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = in[i];
        sumsq += v * v;
    }
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        sumsq += __shfl_down_sync(0xffffffffu, sumsq, offset);
    }
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    if (lane == 0u) {
        smem[warp_id] = sumsq;
    }
    __syncthreads();
    if (warp_id == 0u) {
        const unsigned int num_warps = (blockDim.x + 31u) >> 5;
        float total = (lane < num_warps) ? smem[lane] : 0.0f;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0u) {
            smem[0] = total;
        }
    }
    __syncthreads();
    const float inv_rms = rsqrtf(smem[0] / (float)n + eps);

    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        float w = weight[i];
        if (unit_offset) {
            w += 1.0f;
        }
        out[i] = in[i] * inv_rms * w;
    }
}

// Add-then-RMS-norm: `out_normed[i] = ((residual[i] + delta[i]) / rms) * weight[i]`,
// and `residual_out[i] = residual[i] + delta[i]`.  Single block, used for the
// post-attn / post-FFN paths.
static __device__ __noinline__ void persistent_add_rms_norm(
    const float* __restrict__ residual_in,
    const float* __restrict__ delta,
    float* __restrict__ residual_out,
    const float* __restrict__ post_attn_weight,
    float* __restrict__ post_attn_out,
    const float* __restrict__ next_norm_weight,
    float* __restrict__ next_normed_out,
    unsigned int n,
    float eps,
    int post_unit_offset,
    int next_unit_offset) {
    extern __shared__ float smem[];

    // Step A: write residual + delta, accumulate sum-of-squares for the
    // post-attn RMS.
    float sumsq = 0.0f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        const float r = residual_in[i] + delta[i];
        residual_out[i] = r;
        sumsq += r * r;
    }
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        sumsq += __shfl_down_sync(0xffffffffu, sumsq, offset);
    }
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    if (lane == 0u) {
        smem[warp_id] = sumsq;
    }
    __syncthreads();
    if (warp_id == 0u) {
        const unsigned int num_warps = (blockDim.x + 31u) >> 5;
        float total = (lane < num_warps) ? smem[lane] : 0.0f;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0u) {
            smem[0] = total;
        }
    }
    __syncthreads();
    const float inv_rms = rsqrtf(smem[0] / (float)n + eps);

    // Step B: write post-attn norm output (if requested) and accumulate
    // sum-of-squares of the *re-normed* values for the next-norm step.
    float sumsq2 = 0.0f;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        const float r = residual_out[i];
        float w_pa = post_attn_weight ? post_attn_weight[i] : 1.0f;
        if (post_unit_offset) {
            w_pa += 1.0f;
        }
        const float pa = r * inv_rms * w_pa;
        if (post_attn_out) {
            post_attn_out[i] = pa;
        }
        sumsq2 += pa * pa;
    }
    if (!next_norm_weight || !next_normed_out) {
        return;
    }
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        sumsq2 += __shfl_down_sync(0xffffffffu, sumsq2, offset);
    }
    if (lane == 0u) {
        smem[warp_id] = sumsq2;
    }
    __syncthreads();
    if (warp_id == 0u) {
        const unsigned int num_warps = (blockDim.x + 31u) >> 5;
        float total = (lane < num_warps) ? smem[lane] : 0.0f;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0u) {
            smem[1] = total;
        }
    }
    __syncthreads();
    const float inv_rms2 = rsqrtf(smem[1] / (float)n + eps);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        // Re-read post-attn norm output via the same formula (cheaper than
        // staging via shared memory for hidden ≤ 1536).
        const float r = residual_out[i];
        float w_pa = post_attn_weight ? post_attn_weight[i] : 1.0f;
        if (post_unit_offset) {
            w_pa += 1.0f;
        }
        const float pa = r * inv_rms * w_pa;
        float w_nx = next_norm_weight[i];
        if (next_unit_offset) {
            w_nx += 1.0f;
        }
        next_normed_out[i] = pa * inv_rms2 * w_nx;
    }
}

// ---------------------------------------------------------------------------
// p1: cooperative Q4K GEMV (reuses cu72 M2 implementation).
// ---------------------------------------------------------------------------
// cu99 Milestone 2: batch GEMM wrapper around the existing GEMV. Same weights
// applied to `seq_len` token rows of input → `seq_len` rows of output. First
// step uses the simplest possible loop (outer token, inner GEMV); a real
// batched GEMM with shared-memory weight tiling can replace this once the
// callers are wired up. `seq_len == 1` is identical to a direct GEMV call,
// which lets the decode path adopt the wrapper without correctness changes.
static __device__ __noinline__ void persistent_q4k_gemm_coop(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row,
    unsigned int seq_len,
    unsigned int out_slot_stride = 0u);  // cu105: 0 → rows

static __device__ __noinline__ void persistent_q4k_gemv_coop(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row) {
    const unsigned int warps_per_block = (blockDim.x + 31u) >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_stride = gridDim.x * warps_per_block;

    for (unsigned int row_base = blockIdx.x * warps_per_block;
         row_base < rows;
         row_base += row_stride) {
        const unsigned int row = row_base + warp;
        const bool valid = row < rows;

        float acc = 0.0f;
        const unsigned int row_bytes = blocks_per_row * 144u;
        const unsigned char* row_ptr = weights + (unsigned long long)row * row_bytes;

        if (valid) {
            for (unsigned int b = 0; b < blocks_per_row; ++b) {
                const unsigned char* block = row_ptr + b * 144u;
                const unsigned int raw_d = (unsigned int)block[0] | ((unsigned int)block[1] << 8);
                const unsigned int raw_dmin = (unsigned int)block[2] | ((unsigned int)block[3] << 8);
                const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
                const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

                for (unsigned int tid = lane; tid < 256u; tid += 32u) {
                    const unsigned int j = tid >> 5;
                    unsigned int sc;
                    unsigned int mn;
                    if (j < 4u) {
                        sc = block[4u + j] & 63u;
                        mn = block[4u + j + 4u] & 63u;
                    } else {
                        sc = (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
                        mn = (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
                    }

                    const unsigned int local = tid & 63u;
                    const unsigned int q_index = (tid >> 6) * 32u + (tid & 31u);
                    unsigned int q = block[16u + q_index];
                    q = local < 32u ? (q & 0x0fu) : (q >> 4);

                    const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
                    acc += y * input[b * 256u + tid];
                }
            }
        }

        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }

        if (valid && lane == 0u) {
            out[row] = acc;
        }
    }
}

// Q4K batch GEMM body — outer token loop, inner GEMV (cu99 baseline). cu103
// tried a register-tile (acc[CHUNK]) weight-reuse variant but it measured
// NEGATIVE on RTX 3080 (47t prefill 782→848ms): the register tile cut SM
// occupancy more than it saved weight traffic, and strided per-token input
// reads thrashed L2. Reverted to N-GEMV; the real ROI path is a shared-memory
// tiled GEMM (input tile in smem, register blocking, occupancy preserved) —
// see perf-journal cu103. seq_len==1 == a single GEMV (decode path unchanged).
static __device__ __noinline__ void persistent_q4k_gemm_coop(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row,
    unsigned int seq_len,
    unsigned int out_slot_stride) {  // cu105: 0 → rows (default in fwd decl)
    const unsigned long long input_stride =
        (unsigned long long)blocks_per_row * 256ull;
    const unsigned int oss = (out_slot_stride == 0u) ? rows : out_slot_stride;
    for (unsigned int t = 0; t < seq_len; ++t) {
        persistent_q4k_gemv_coop(
            out + (unsigned long long)t * oss,
            weights,
            input + (unsigned long long)t * input_stride,
            rows,
            blocks_per_row);
    }
}

// cu103 M4: Q4K input-smem tiled batch GEMM. Stages the input tile [tcount × K]
// into shared memory once per token-tile so every weight row reuses it (no
// strided per-token global re-reads → no L2 thrash). Weight is read once per
// token-tile (BN_TILE× fewer reads than the N-GEMV body). The acc[BN_TILE]
// register tile is IDENTICAL to the cu102 naive variant — only the input path
// differs (global-strided → smem), isolating the L2-thrash hypothesis from the
// occupancy hypothesis. Caller must launch with >= BN_TILE*K*4 dynamic smem.
// Used for gate/up (K=hidden=1536 → 24KB at BN_TILE=4). down (K=n_ff=6144) is
// too large for 48KB smem and stays on the N-GEMV path. seq_len==1 falls back
// to plain GEMV (decode path, no smem staging overhead).
// cu105: out_slot_stride decouples the output's per-token stride from `rows`.
// On Gemma4 q_dim / kv_dim vary by layer (sliding head_dim 256 vs full 512) so
// QKV batch slots are sized to q_dim_max / kv_dim_max, NOT the per-layer dim —
// pass that slot stride here. Default 0 → use `rows` (gate/up/down where the
// dim is layer-invariant). seq_len<=1 → GEMV writes a single slot, stride moot.
static __device__ __noinline__ void persistent_q4k_gemm_smem(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row,
    unsigned int seq_len,
    unsigned int out_slot_stride = 0u) {
    if (seq_len <= 1u) {
        persistent_q4k_gemv_coop(out, weights, input, rows, blocks_per_row);
        return;
    }
    const unsigned int oss = (out_slot_stride == 0u) ? rows : out_slot_stride;
    extern __shared__ float smem_input[];  // [BN_TILE * K]
    const unsigned int BN_TILE = 4u;
    const unsigned int K = blocks_per_row * 256u;
    const unsigned int warps_per_block = (blockDim.x + 31u) >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_stride = gridDim.x * warps_per_block;
    const unsigned long long input_stride2 = (unsigned long long)K;

    for (unsigned int t0 = 0u; t0 < seq_len; t0 += BN_TILE) {
        const unsigned int tcount = (seq_len - t0) < BN_TILE ? (seq_len - t0) : BN_TILE;
        // Stage input tile [tcount][K] into shared memory (all threads cooperate).
        __syncthreads();
        for (unsigned int idx = threadIdx.x; idx < tcount * K; idx += blockDim.x) {
            const unsigned int c = idx / K;
            const unsigned int k = idx - c * K;
            smem_input[idx] = input[(unsigned long long)(t0 + c) * input_stride2 + k];
        }
        __syncthreads();

        for (unsigned int row_base = blockIdx.x * warps_per_block;
             row_base < rows;
             row_base += row_stride) {
            const unsigned int row = row_base + warp;
            const bool valid = row < rows;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (valid) {
                const unsigned char* row_ptr =
                    weights + (unsigned long long)row * blocks_per_row * 144u;
                for (unsigned int b = 0; b < blocks_per_row; ++b) {
                    const unsigned char* block = row_ptr + b * 144u;
                    const unsigned int raw_d = (unsigned int)block[0] | ((unsigned int)block[1] << 8);
                    const unsigned int raw_dmin = (unsigned int)block[2] | ((unsigned int)block[3] << 8);
                    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
                    const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
                    for (unsigned int tid = lane; tid < 256u; tid += 32u) {
                        const unsigned int j = tid >> 5;
                        unsigned int sc;
                        unsigned int mn;
                        if (j < 4u) {
                            sc = block[4u + j] & 63u;
                            mn = block[4u + j + 4u] & 63u;
                        } else {
                            sc = (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
                            mn = (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
                        }
                        const unsigned int local = tid & 63u;
                        const unsigned int q_index = (tid >> 6) * 32u + (tid & 31u);
                        unsigned int q = block[16u + q_index];
                        q = local < 32u ? (q & 0x0fu) : (q >> 4);
                        const float wval = (d * (float)sc) * (float)q - dmin * (float)mn;
                        const unsigned int k = b * 256u + tid;
                        #pragma unroll
                        for (unsigned int c = 0u; c < 4u; ++c) {
                            if (c < tcount) {
                                acc[c] += wval * smem_input[c * K + k];
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (unsigned int c = 0u; c < 4u; ++c) {
                if (c < tcount) {
                    float a = acc[c];
                    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
                        a += __shfl_down_sync(0xffffffffu, a, offset);
                    }
                    if (valid && lane == 0u) {
                        out[(unsigned long long)(t0 + c) * oss + row] = a;
                    }
                }
            }
        }
    }
}

// Cooperative Q6K GEMV.  Q6K block layout (210 bytes per 256 elements):
//   [0..127]   ql   — 128 bytes, low 4 bits of each q (256 q's)
//   [128..191] qh   —  64 bytes, high 2 bits of each q
//   [192..207] sc   —  16 bytes, int8 scales (16 sub-blocks of 16 elements)
//   [208..209] d    —  2 bytes, f16 super-block scale
// q_final = (int)q - 32, value = d * sc * (q - 32).
static __device__ __noinline__ void persistent_q6k_gemv_coop(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row,
    unsigned int debug_tag) {  // cu96: nonzero = print row 0 first iter
    const unsigned int warps_per_block = (blockDim.x + 31u) >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_stride = gridDim.x * warps_per_block;
    const unsigned int row_bytes = blocks_per_row * 210u;

    for (unsigned int row_base = blockIdx.x * warps_per_block;
         row_base < rows;
         row_base += row_stride) {
        const unsigned int row = row_base + warp;
        const bool valid = row < rows;

        float acc = 0.0f;
        const unsigned char* row_ptr = weights + (unsigned long long)row * row_bytes;

        if (valid) {
            for (unsigned int b = 0; b < blocks_per_row; ++b) {
                const unsigned char* block = row_ptr + b * 210u;
                const unsigned int raw_d =
                    (unsigned int)block[208] | ((unsigned int)block[209] << 8);
                const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
                for (unsigned int tid = lane; tid < 256u; tid += 32u) {
                    const unsigned int n = tid >> 7;       // 0 or 1
                    const unsigned int rem = tid & 127u;   // 0..127
                    const unsigned int l = rem & 31u;      // 0..31
                    const unsigned int is = l >> 4;        // 0 or 1 (sub-block idx)
                    const unsigned int ql_base = n * 64u;
                    const unsigned int qh_base = 128u + n * 32u;
                    const unsigned int sc_base = 192u + n * 8u;

                    unsigned int q;
                    int sc;
                    const unsigned int qh = block[qh_base + l];
                    if (rem < 32u) {
                        q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                        sc = (int)((signed char)block[sc_base + is]);
                    } else if (rem < 64u) {
                        q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                        sc = (int)((signed char)block[sc_base + is + 2u]);
                    } else if (rem < 96u) {
                        q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
                        sc = (int)((signed char)block[sc_base + is + 4u]);
                    } else {
                        q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                        sc = (int)((signed char)block[sc_base + is + 6u]);
                    }
                    const float y = d * (float)sc * (float)((int)q - 32);
                    acc += y * input[b * 256u + tid];
                }
            }
        }
        (void)debug_tag;  // cu96 probe retired after cu97 fix.
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (valid && lane == 0u) {
            out[row] = acc;
        }
    }
}

// cu99: Q6K batch GEMM — same wrapper pattern as the Q4K version. Sequenced
// GEMV calls share the weight matrix across `seq_len` rows of input.
static __device__ __noinline__ void persistent_q6k_gemm_coop(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row,
    unsigned int seq_len,
    unsigned int debug_tag,
    unsigned int out_slot_stride = 0u) {  // cu105: 0 → rows (layer-invariant dim)
    const unsigned long long input_stride =
        (unsigned long long)blocks_per_row * 256ull;
    const unsigned int oss = (out_slot_stride == 0u) ? rows : out_slot_stride;
    for (unsigned int t = 0; t < seq_len; ++t) {
        persistent_q6k_gemv_coop(
            out + (unsigned long long)t * oss,
            weights,
            input + (unsigned long long)t * input_stride,
            rows,
            blocks_per_row,
            debug_tag);
    }
}

// cu106 M4: Q6K BK-tiled input-smem batch GEMM. The down projection has
// K=n_ff=6144, too large to stage entirely in smem (BN_TILE*K*4 = 96KB), so the
// cu103 gemm_smem (whole-K-in-smem) path skipped it and down stayed on the
// N-GEMV body — which re-reads+re-decodes the WHOLE weight matrix once per token
// (seq_len times). cu105 phase timing measured that as 49% of prefill cycles.
//
// This variant tiles the K dimension: BK = rows (== hidden_dim for down) wide
// chunks, staged into the same BN_TILE*hidden_dim*4 = 24KB smem the host already
// budgets for gate/up. acc[BN_TILE] accumulates across the K-tile loop. Weight
// is read+decoded once per (token-tile × row-pass) — for seq_len=1115 that is
// ~P(grid row-passes, 2-3) reads instead of 1115. The smem footprint is
// IDENTICAL to gate/up so occupancy is unchanged (cu103: occupancy is the ROI
// gate, not register/L2 traffic). seq_len<=1 → plain GEMV (decode path).
// out_slot_stride 0 → rows (down output dim is layer-invariant = hidden_dim).
// Assumes rows is a 256-multiple (Gemma4 E2B hidden_dim=1536=6*256).
static __device__ __noinline__ void persistent_q6k_gemm_smem_bk(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row,
    unsigned int seq_len,
    unsigned int out_slot_stride = 0u) {
    if (seq_len <= 1u) {
        persistent_q6k_gemv_coop(out, weights, input, rows, blocks_per_row, 0u);
        return;
    }
    const unsigned int oss = (out_slot_stride == 0u) ? rows : out_slot_stride;
    extern __shared__ float smem_input[];  // [BN_TILE * BK]
    const unsigned int BN_TILE = 4u;
    // BK = rows == hidden_dim → smem_input is exactly BN_TILE*hidden_dim floats,
    // matching the host's gate/up smem budget. BK_BLOCKS = rows/256.
    const unsigned int BK = rows;
    const unsigned int BK_BLOCKS = rows >> 8;  // rows / 256
    const unsigned int K = blocks_per_row * 256u;
    const unsigned int warps_per_block = (blockDim.x + 31u) >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_stride = gridDim.x * warps_per_block;
    const unsigned int row_bytes = blocks_per_row * 210u;
    const unsigned long long input_stride2 = (unsigned long long)K;

    for (unsigned int t0 = 0u; t0 < seq_len; t0 += BN_TILE) {
        const unsigned int tcount = (seq_len - t0) < BN_TILE ? (seq_len - t0) : BN_TILE;
        for (unsigned int row_base = blockIdx.x * warps_per_block;
             row_base < rows;
             row_base += row_stride) {
            const unsigned int row = row_base + warp;
            const bool valid = row < rows;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const unsigned char* row_ptr = weights + (unsigned long long)row * row_bytes;

            for (unsigned int kb0 = 0u; kb0 < blocks_per_row; kb0 += BK_BLOCKS) {
                const unsigned int bkc =
                    (blocks_per_row - kb0) < BK_BLOCKS ? (blocks_per_row - kb0) : BK_BLOCKS;
                const unsigned int tile_k = bkc * 256u;
                // Stage input[t0..t0+tcount][kb0*256 .. kb0*256+tile_k] into smem.
                // Re-staged per row-pass (P passes) — cheap vs the weight-reuse win.
                __syncthreads();
                for (unsigned int idx = threadIdx.x; idx < tcount * tile_k; idx += blockDim.x) {
                    const unsigned int c = idx / tile_k;
                    const unsigned int kk = idx - c * tile_k;
                    smem_input[c * BK + kk] =
                        input[(unsigned long long)(t0 + c) * input_stride2 + kb0 * 256u + kk];
                }
                __syncthreads();

                if (valid) {
                    for (unsigned int b = kb0; b < kb0 + bkc; ++b) {
                        const unsigned char* block = row_ptr + b * 210u;
                        const unsigned int raw_d =
                            (unsigned int)block[208] | ((unsigned int)block[209] << 8);
                        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
                        const unsigned int smem_b = (b - kb0) * 256u;
                        for (unsigned int tid = lane; tid < 256u; tid += 32u) {
                            const unsigned int n = tid >> 7;       // 0 or 1
                            const unsigned int rem = tid & 127u;   // 0..127
                            const unsigned int l = rem & 31u;      // 0..31
                            const unsigned int is = l >> 4;        // 0 or 1
                            const unsigned int ql_base = n * 64u;
                            const unsigned int qh_base = 128u + n * 32u;
                            const unsigned int sc_base = 192u + n * 8u;

                            unsigned int q;
                            int sc;
                            const unsigned int qh = block[qh_base + l];
                            if (rem < 32u) {
                                q = (block[ql_base + l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                                sc = (int)((signed char)block[sc_base + is]);
                            } else if (rem < 64u) {
                                q = (block[ql_base + l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                                sc = (int)((signed char)block[sc_base + is + 2u]);
                            } else if (rem < 96u) {
                                q = (block[ql_base + l] >> 4) | (((qh >> 4) & 3u) << 4);
                                sc = (int)((signed char)block[sc_base + is + 4u]);
                            } else {
                                q = (block[ql_base + l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                                sc = (int)((signed char)block[sc_base + is + 6u]);
                            }
                            const float y = d * (float)sc * (float)((int)q - 32);
                            const unsigned int sk = smem_b + tid;
                            #pragma unroll
                            for (unsigned int c = 0u; c < 4u; ++c) {
                                if (c < tcount) {
                                    acc[c] += y * smem_input[c * BK + sk];
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (unsigned int c = 0u; c < 4u; ++c) {
                if (c < tcount) {
                    float a = acc[c];
                    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
                        a += __shfl_down_sync(0xffffffffu, a, offset);
                    }
                    if (valid && lane == 0u) {
                        out[(unsigned long long)(t0 + c) * oss + row] = a;
                    }
                }
            }
        }
    }
}

// Cooperative GEMV with accumulation: `out[row] += dot(weights[row], input)`.
// Used by O-projection where output is added to the residual carrier.
static __device__ __noinline__ void persistent_q4k_gemv_coop_accum(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row) {
    const unsigned int warps_per_block = (blockDim.x + 31u) >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_stride = gridDim.x * warps_per_block;

    for (unsigned int row_base = blockIdx.x * warps_per_block;
         row_base < rows;
         row_base += row_stride) {
        const unsigned int row = row_base + warp;
        const bool valid = row < rows;

        float acc = 0.0f;
        const unsigned int row_bytes = blocks_per_row * 144u;
        const unsigned char* row_ptr = weights + (unsigned long long)row * row_bytes;

        if (valid) {
            for (unsigned int b = 0; b < blocks_per_row; ++b) {
                const unsigned char* block = row_ptr + b * 144u;
                const unsigned int raw_d = (unsigned int)block[0] | ((unsigned int)block[1] << 8);
                const unsigned int raw_dmin = (unsigned int)block[2] | ((unsigned int)block[3] << 8);
                const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
                const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

                for (unsigned int tid = lane; tid < 256u; tid += 32u) {
                    const unsigned int j = tid >> 5;
                    unsigned int sc;
                    unsigned int mn;
                    if (j < 4u) {
                        sc = block[4u + j] & 63u;
                        mn = block[4u + j + 4u] & 63u;
                    } else {
                        sc = (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
                        mn = (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
                    }

                    const unsigned int local = tid & 63u;
                    const unsigned int q_index = (tid >> 6) * 32u + (tid & 31u);
                    unsigned int q = block[16u + q_index];
                    q = local < 32u ? (q & 0x0fu) : (q >> 4);

                    const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
                    acc += y * input[b * 256u + tid];
                }
            }
        }

        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }

        if (valid && lane == 0u) {
            out[row] += acc;
        }
    }
}

// F32 cooperative GEMV (used for PLE F32 gate/proj projections).  `weights`
// is row-major `[rows x cols]`, `input` is `[cols]`, `out` is `[rows]`.
static __device__ __noinline__ void persistent_f32_gemv_coop(
    float* __restrict__ out,
    const float* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int cols) {
    const unsigned int warps_per_block = (blockDim.x + 31u) >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_stride = gridDim.x * warps_per_block;

    for (unsigned int row_base = blockIdx.x * warps_per_block;
         row_base < rows;
         row_base += row_stride) {
        const unsigned int row = row_base + warp;
        const bool valid = row < rows;

        float acc = 0.0f;
        if (valid) {
            const float* row_ptr = weights + (unsigned long long)row * cols;
            for (unsigned int c = lane; c < cols; c += 32u) {
                acc += row_ptr[c] * input[c];
            }
        }
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (valid && lane == 0u) {
            out[row] = acc;
        }
    }
}

// Q8_0 cooperative GEMV with row-wise argmax reduction. Used for the output
// logits projection — but the full argmax across vocab is too coarse for a
// cooperative loop and is implemented as a post-process below in the kernel.
static __device__ __noinline__ void persistent_q8_0_gemv_coop(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    unsigned int rows,
    unsigned int blocks_per_row) {
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_stride = gridDim.x * 8u;
    // Q8_0 block layout: 32 elements / block, 34 bytes / block (f16 scale + 32 i8).
    const unsigned int row_bytes = blocks_per_row * 34u;

    for (unsigned int row_base = blockIdx.x * 8u; row_base < rows; row_base += row_stride) {
        const unsigned int row = row_base + warp;
        const bool valid = row < rows;

        float acc = 0.0f;
        if (valid) {
            const unsigned char* row_ptr = weights + (unsigned long long)row * row_bytes;
            for (unsigned int b = 0; b < blocks_per_row; ++b) {
                const unsigned char* block = row_ptr + b * 34u;
                const unsigned int raw_d = (unsigned int)block[0] | ((unsigned int)block[1] << 8);
                const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
                for (unsigned int tid = lane; tid < 32u; tid += 32u) {
                    const signed char q = (signed char)block[2u + tid];
                    acc += d * (float)q * input[b * 32u + tid];
                }
            }
        }
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (valid && lane == 0u) {
            out[row] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// p2-pre: per-head RMS norm on Q and K (Qwen-style q_norm / k_norm).
// One block per Q head; blocks that fall past `k_heads` skip the K branch.
// Reads `head_dim` floats from `head_data + head_idx * head_dim`, computes
// the head's RMS, then scales by `(weight + 1)` element-wise.
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_per_head_rms_norm(
    float* __restrict__ head_data,
    const float* __restrict__ weight,
    unsigned int head_dim,
    float eps) {
    extern __shared__ float qk_smem[];
    const unsigned int lane = threadIdx.x;
    const bool active = lane < head_dim;
    float* head_ptr = head_data + (unsigned long long)blockIdx.x * head_dim;
    const float v = active ? head_ptr[lane] : 0.0f;
    float sumsq = v * v;
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        sumsq += __shfl_down_sync(0xffffffffu, sumsq, offset);
    }
    const unsigned int warp_id = lane >> 5;
    const unsigned int warp_lane = lane & 31u;
    const unsigned int active_warps = (head_dim + 31u) >> 5;
    if (warp_lane == 0u && warp_id < active_warps) {
        qk_smem[warp_id] = sumsq;
    }
    __syncthreads();
    if (warp_id == 0u) {
        float total = (warp_lane < active_warps) ? qk_smem[warp_lane] : 0.0f;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (warp_lane == 0u) {
            qk_smem[0] = total;
        }
    }
    __syncthreads();
    const float inv_rms = rsqrtf(qk_smem[0] / (float)head_dim + eps);
    if (active) {
        // cu76: GGUF Q/K norm weights already include unit_offset (eager
        // matches). Hardcoded "+1.0f" caused double-offset blow-up.
        // cu97: weight == nullptr → no-scale RMS norm (Gemma4 V projection
        // post-processing — eager applies `apply_rms_norm_no_scale_into` per
        // head before V cache write; persistent decode was missing this and
        // wrote raw V projection output (~50x larger magnitude).
        const float w = (weight != nullptr) ? weight[lane] : 1.0f;
        head_ptr[lane] = v * inv_rms * w;
    }
}

// ---------------------------------------------------------------------------
// p2: QK NEOX RoPE (single block per head).
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_rope_neox_qk(
    float* __restrict__ q,
    float* __restrict__ k,
    unsigned int q_dim,
    unsigned int kv_dim,
    unsigned int head_dim,
    unsigned int pos,
    float theta_base,
    const float* __restrict__ freq_factors) {
    const unsigned int half = head_dim / 2u;
    const unsigned int lane = threadIdx.x;
    const unsigned int h = blockIdx.x;
    const unsigned int q_heads = q_dim / head_dim;
    const unsigned int k_heads = kv_dim / head_dim;
    // cu77 fix: do NOT early-return on lane >= half — leaves threads outside
    // any subsequent __syncthreads() and risks UB.  All threads in the block
    // progress through the function; only `lane < half` does work.
    if (h >= q_heads) {
        return;  // block-level early exit is OK (entire block).
    }
    const bool active = lane < half;
    const unsigned int idx = active ? (h * head_dim + lane) : 0u;
    const float freq = active ? powf(theta_base, -((float)lane) / (float)half) : 0.0f;
    float angle = active ? ((float)pos * freq) : 0.0f;
    if (active && freq_factors != nullptr) {
        angle /= freq_factors[lane];
    }
    if (!active) {
        return;
    }
    const float c = cosf(angle);
    const float s = sinf(angle);
    const float x0q = q[idx];
    const float x1q = q[idx + half];
    q[idx] = x0q * c - x1q * s;
    q[idx + half] = x0q * s + x1q * c;
    if (h < k_heads) {
        const float x0k = k[idx];
        const float x1k = k[idx + half];
        k[idx] = x0k * c - x1k * s;
        k[idx + half] = x0k * s + x1k * c;
    }
}

// ---------------------------------------------------------------------------
// p3: KV cache write (block 0 only — kv_dim is small, ≤ 512).
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_kv_cache_write(
    unsigned short* __restrict__ k_cache,
    unsigned short* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    unsigned int kv_dim,
    unsigned int kv_pos) {
    const unsigned int base = kv_pos * kv_dim;
    for (unsigned int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        k_cache[base + i] = __half_as_ushort(__float2half(k[i]));
        v_cache[base + i] = __half_as_ushort(__float2half(v[i]));
    }
}

// ---------------------------------------------------------------------------
// p4: attention decode (1 block per query head; KV is shared with grouping
//     factor `num_heads / num_kv_heads`).  Online softmax over kv_len keys.
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_attention_decode(
    float* __restrict__ out,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    unsigned int kv_len,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float scale,
    unsigned int sliding_window,
    float* __restrict__ scores_probe,      // cu80: [kv_len] head-0 score per token
    float* __restrict__ v_probe,           // cu80: [head_dim] head-0 V cache at j=0
    float* __restrict__ acc_probe,         // cu82: [head_dim] head-0 final acc
    float* __restrict__ row_sum_probe,     // cu82: [1] head-0 lane-0 row_sum at end
    unsigned int nan_trace_layer) {        // cu93: layer index (UINT32_MAX = trace disabled)
    extern __shared__ float partial_smem[];  // size = head_dim (≤512 floats).
    const unsigned int lane = threadIdx.x;
    const unsigned int h = blockIdx.x;
    if (h >= num_heads) {
        return;  // block-level skip: entire block exits together (safe).
    }
    // Critical: do NOT early-return on lane >= head_dim because that would
    // leave some threads outside __syncthreads() and produce UB.  Instead,
    // mark them inactive and let them participate in all syncs.
    const bool active = lane < head_dim;
    const unsigned int heads_per_group = num_heads / num_kv_heads;
    const unsigned int kv_h = h / heads_per_group;
    const float qv = active ? q[h * head_dim + lane] : 0.0f;

    // Sliding-window: skip keys older than `kv_len - sliding_window`.
    unsigned int j_start = 0u;
    if (sliding_window > 0u && kv_len > sliding_window) {
        j_start = kv_len - sliding_window;
    }

    // cu79: explicit shared-mem init so partial_smem inactive slots (lane >=
    // head_dim) don't carry stale data from prior phases.  All threads
    // participate in the init + sync; reduction uses lane < stride and
    // lane+stride < head_dim so inactive slots are never read but defensive
    // init is cheap insurance.
    if (lane < head_dim) {
        partial_smem[lane] = 0.0f;
    }
    __syncthreads();

    float row_max = -3.4028234663852886e38f;
    float row_sum = 0.0f;
    float acc = 0.0f;

    // cu93: K cache inf scan — find first j where K has inf/nan in head-0 block.
    if (nan_trace_layer != 0xFFFFFFFFu && blockIdx.x == 0) {
        __shared__ int s_first_bad_j;
        __shared__ int s_qv_bad;
        if (threadIdx.x == 0) { s_first_bad_j = -1; s_qv_bad = 0; }
        __syncthreads();
        if (active && (isnan(qv) || isinf(qv))) atomicOr(&s_qv_bad, 1);
        for (unsigned int j = j_start; j < kv_len; ++j) {
            const unsigned int kv_base = j * num_kv_heads * head_dim + kv_h * head_dim;
            if (active) {
                const float kv_check = __half2float(__ushort_as_half(k_cache[kv_base + lane]));
                if (isnan(kv_check) || isinf(kv_check)) {
                    atomicCAS(reinterpret_cast<unsigned int*>(&s_first_bad_j),
                              0xFFFFFFFFu, j);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && (s_first_bad_j >= 0 || s_qv_bad != 0)) {
            printf("[cu93-Kcache-NaN] layer=%u qv_nan=%d first_bad_j=%d kv_len=%u sw=%u\n",
                   nan_trace_layer, s_qv_bad, s_first_bad_j, kv_len, sliding_window);
        }
    }
    for (unsigned int j = j_start; j < kv_len; ++j) {
        const unsigned int kv_base = j * num_kv_heads * head_dim + kv_h * head_dim;
        const float kv =
            active ? __half2float(__ushort_as_half(k_cache[kv_base + lane])) : 0.0f;
        if (active) {
            partial_smem[lane] = qv * kv;
        }
        __syncthreads();
        for (unsigned int stride = head_dim / 2u; stride > 0u; stride >>= 1u) {
            if (active && lane < stride) {
                partial_smem[lane] += partial_smem[lane + stride];
            }
            __syncthreads();
        }
        const float score = partial_smem[0] * scale;
        // cu80: dump score for head-0 only (block 0, lane 0).
        if (scores_probe != nullptr && blockIdx.x == 0 && lane == 0u) {
            scores_probe[j] = score;
        }
        const float new_max = fmaxf(row_max, score);
        const float old_scale = (row_max == -3.4028234663852886e38f)
            ? 0.0f
            : expf(row_max - new_max);
        const float p = expf(score - new_max);
        const float vv =
            active ? __half2float(__ushort_as_half(v_cache[kv_base + lane])) : 0.0f;
        // cu80: dump V cache value at j=0, head-0.
        if (v_probe != nullptr && blockIdx.x == 0 && active && j == j_start) {
            v_probe[lane] = vv;
        }
        acc = acc * old_scale + p * vv;
        row_sum = row_sum * old_scale + p;
        row_max = new_max;
        __syncthreads();
    }

    // cu82: dump head-0 acc and row_sum BEFORE final divide.
    if (acc_probe != nullptr && blockIdx.x == 0 && active) {
        acc_probe[lane] = acc;
    }
    if (row_sum_probe != nullptr && blockIdx.x == 0 && lane == 0u) {
        row_sum_probe[0] = row_sum;
    }
    if (active) {
        const float written = row_sum > 0.0f ? acc / row_sum : 0.0f;
        out[h * head_dim + lane] = written;
    }
}

// ---------------------------------------------------------------------------
// cu100 Milestone 2 — causal attention for batch prefill.
//
// Per-query block (one block per (token, head) pair). Each query at position
// `t` attends to keys `[j_start .. rope_pos_base + t + 1)` (causal mask).
// SWA same as decode attention but with `max_j` bounded per query.
//
// Inputs:
//   q              [seq_len * num_heads * head_dim] (token-major)
//   k_cache        [(rope_pos_base + seq_len) * num_kv_heads * head_dim] fp16
//   v_cache        same shape, fp16
//   seq_len        N queries in this dispatch
//   rope_pos_base  position of the first query (=past tokens count). Query t
//                  attends to keys [0..rope_pos_base + t].
//   num_heads / num_kv_heads / head_dim / scale / sliding_window — same as
//   the decode kernel.
//
// Grid contract: `gridDim.x >= num_heads * seq_len`. Block layout = head_dim
// lanes (mirrors `persistent_attention_decode`).
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_attention_prefill(
    float* __restrict__ out,
    const float* __restrict__ q,
    const unsigned short* __restrict__ k_cache,
    const unsigned short* __restrict__ v_cache,
    unsigned int seq_len,
    unsigned int rope_pos_base,
    unsigned int num_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    float scale,
    unsigned int sliding_window,
    // cu101 M3: token-slot stride of `q` / `out` (= q_dim_max on the host).
    // May exceed num_heads*head_dim when SWA layers (smaller q_dim) share the
    // FULL-layer-sized buffer; the per-token slot is still q_dim_max wide.
    unsigned int q_token_stride) {
    extern __shared__ float partial_smem[];
    const unsigned int lane = threadIdx.x;
    const unsigned int total_qb = num_heads * seq_len;
    const bool active = lane < head_dim;
    const unsigned int heads_per_group = num_heads / num_kv_heads;
    // cu105 grid stride: gridDim.x may be smaller than total_qb (cooperative
    // launch max blocks is bounded by SM occupancy). Wrap block_idx so every
    // (token, head) pair gets processed.
    for (unsigned int block_idx = blockIdx.x;
         block_idx < total_qb;
         block_idx += gridDim.x) {
        const unsigned int t = block_idx / num_heads;        // query token index
        const unsigned int h = block_idx % num_heads;        // head index
        const unsigned int kv_h = h / heads_per_group;

        const unsigned int q_offset =
            t * q_token_stride + h * head_dim;
        const float qv = active ? q[q_offset + lane] : 0.0f;

        const unsigned int max_j = rope_pos_base + t + 1u;   // causal mask
        unsigned int j_start = 0u;
        if (sliding_window > 0u && max_j > sliding_window) {
            j_start = max_j - sliding_window;
        }

        if (lane < head_dim) {
            partial_smem[lane] = 0.0f;
        }
        __syncthreads();

        float row_max = -3.4028234663852886e38f;
        float row_sum = 0.0f;
        float acc = 0.0f;

        for (unsigned int j = j_start; j < max_j; ++j) {
            const unsigned int kv_base =
                j * num_kv_heads * head_dim + kv_h * head_dim;
            const float kv = active
                ? __half2float(__ushort_as_half(k_cache[kv_base + lane]))
                : 0.0f;
            if (active) {
                partial_smem[lane] = qv * kv;
            }
            __syncthreads();
            for (unsigned int stride = head_dim / 2u; stride > 0u; stride >>= 1u) {
                if (active && lane < stride) {
                    partial_smem[lane] += partial_smem[lane + stride];
                }
                __syncthreads();
            }
            const float score = partial_smem[0] * scale;
            const float new_max = fmaxf(row_max, score);
            const float old_scale = (row_max == -3.4028234663852886e38f)
                ? 0.0f
                : expf(row_max - new_max);
            const float p = expf(score - new_max);
            const float vv = active
                ? __half2float(__ushort_as_half(v_cache[kv_base + lane]))
                : 0.0f;
            acc = acc * old_scale + p * vv;
            row_sum = row_sum * old_scale + p;
            row_max = new_max;
            __syncthreads();
        }
        if (active) {
            const float written = row_sum > 0.0f ? acc / row_sum : 0.0f;
            out[q_offset + lane] = written;
        }
    }
}

// ---------------------------------------------------------------------------
// p5/6 helpers: GELU·mul (in-place gate *= GELU(gate) * up).
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_gelu_mul(
    float* __restrict__ gate,
    const float* __restrict__ up_or_ple,
    unsigned int n) {
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < n; i += stride) {
        const float x = gate[i];
        const float c = 0.044715f * x * x * x;
        const float t = 0.7978845608f * (x + c);
        const float gelu = 0.5f * x * (1.0f + tanhf(t));
        gate[i] = gelu * up_or_ple[i];
    }
}

// ---------------------------------------------------------------------------
// Single-block argmax over `n` floats. Designed for block 0.
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_argmax_single_block(
    const float* __restrict__ in,
    unsigned int n,
    int* __restrict__ out_idx) {
    __shared__ float best_val[32];
    __shared__ int best_idx[32];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    float local_v = -3.4028234663852886e38f;
    int local_i = -1;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = in[i];
        if (v > local_v) {
            local_v = v;
            local_i = (int)i;
        }
    }
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        const float ov = __shfl_down_sync(0xffffffffu, local_v, offset);
        const int oi = __shfl_down_sync(0xffffffffu, local_i, offset);
        if (ov > local_v) {
            local_v = ov;
            local_i = oi;
        }
    }
    if (lane == 0u) {
        best_val[warp] = local_v;
        best_idx[warp] = local_i;
    }
    __syncthreads();
    if (warp == 0u) {
        const unsigned int num_warps = (blockDim.x + 31u) >> 5;
        float v = (lane < num_warps) ? best_val[lane] : -3.4028234663852886e38f;
        int i = (lane < num_warps) ? best_idx[lane] : -1;
        for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
            const float ov = __shfl_down_sync(0xffffffffu, v, offset);
            const int oi = __shfl_down_sync(0xffffffffu, i, offset);
            if (ov > v) {
                v = ov;
                i = oi;
            }
        }
        if (lane == 0u) {
            *out_idx = i;
        }
    }
}

// ---------------------------------------------------------------------------
// Layer-output scale (in-place): `hidden[i] *= scale`.
// ---------------------------------------------------------------------------
static __device__ __noinline__ void persistent_scale_inplace(
    float* __restrict__ hidden,
    unsigned int n,
    float scale) {
    if (scale == 1.0f) {
        return;
    }
    const unsigned int stride = gridDim.x * blockDim.x;
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < n; i += stride) {
        hidden[i] *= scale;
    }
}

// ---------------------------------------------------------------------------
// M3 entry point — single cooperative launch processes all layers + output.
// ---------------------------------------------------------------------------
extern "C" __global__ void rnb_persistent_decode_e2b_m1(PersistentDecodeParams params) {
    cg::grid_group grid = cg::this_grid();

    // cu102 sentinel — confirm kernel actually launches and reaches outer loop.
    if (params.nan_trace == 2u && blockIdx.x == 0 && threadIdx.x == 0) {
        printf("[cu102-kernel] enter seq_len=%u rope_pos=%u kv_len=%u\n",
               params.seq_len, params.rope_pos, params.kv_len);
    }

    const float theta_base = 1.0e6f;  // Gemma4 RoPE theta.
    const unsigned int hidden_dim = params.hidden_dim;
    const unsigned int blocks_per_row_hidden = hidden_dim / 256u;

    // cu74 phase-by-phase debug printf.  Enabled by sentinel
    // params.flags == 1 in PersistentDecodeParams (we hijack the unused
    // first byte of `norm_eps` field's debug carrier; here we instead
    // rely on a runtime check: only block 0 thread 0 prints).
    #define PHASE_LOG(tag, ptr, n) do {} while(0)

    // cu100 Milestone 2 — outer token loop wraps the entire layer-by-layer
    // body so a single cooperative launch can process `seq_len` tokens
    // sequentially. seq_len=1 (decode) keeps the existing behavior; seq_len > 1
    // (prefill batch caller, env-gated) runs the body once per token, with
    // host-supplied per-token hidden embeddings packed into `params.hidden`
    // (sized seq_len * hidden_dim, indexed via `params.hidden_token_stride`).
    // Output projection only fires for the last token to skip per-token
    // vocab-sized logits writes.
    const unsigned int __seq_len = params.seq_len;
    const unsigned int __rope_pos_base = params.rope_pos;
    const unsigned long long __hidden_stride =
        (__seq_len > 1u)
            ? (unsigned long long)hidden_dim
            : 0ull;
    float* const __hidden_base = params.hidden;
    // cu101 M3: split each layer's body into per-token pre-attention /
    // attention / post-attention phases so attention can become a single batch
    // causal call (cu103). `hidden` / `q_buf` / `attn_out` are advanced by
    // `__t * stride` at the head of each phase loop. Decode (seq_len=1) → stride
    // 0 → slot 0, behavior unchanged. q_buf/attn_out are sized hidden_slots *
    // q_dim_max on the host; normed / k_buf / v_buf / gate_buf stay single-slot
    // (consumed within each token's own work).
    float* const __q_buf_base = params.q_buf;
    float* const __attn_out_base = params.attn_out;
    // cu105 fix: capture attn_normed / k_buf / v_buf bases ONCE before the layer
    // loop. A3 advances params.k_buf/v_buf by __t*__kv_stride per token; if these
    // bases were re-read inside the layer loop they would compound the advance
    // every layer (k_buf_base += (seq_len-1)*kv_dim_max each iter) → OOB write at
    // layer 5+ corrupting adjacent weight memory.
    float* const __attn_normed_base = params.attn_normed;
    float* const __k_buf_base = params.k_buf;
    float* const __v_buf_base = params.v_buf;
    const unsigned long long __q_stride =
        (__seq_len > 1u) ? (unsigned long long)params.q_dim_max : 0ull;
    const unsigned long long __kv_stride =
        (__seq_len > 1u) ? (unsigned long long)params.kv_dim_max : 0ull;

    for (unsigned int layer = 0; layer < params.num_layers; ++layer) {
        const PersistentLayerParams& lp = params.layers[layer];
        // cu76: Gemma4 attention uses NO scale (eager's resolve_attention_scale
        // returns 1.0 default). The standard 1/sqrt(head_dim) was wrong here
        // and made attention scores far too flat → softmax close to uniform.
        const float head_scale = 1.0f;
        const unsigned int q_heads = lp.q_dim / lp.head_dim;
        const unsigned int kv_heads = lp.kv_dim / lp.head_dim;

        // ===== PHASE A1 (pre-attn norm): per-token attn_norm → attn_normed[t]
        // batch slots (cu105). attn_normed / k_buf / v_buf bases + __kv_stride
        // are captured ONCE before the layer loop (above) — see the cu105 fix
        // note there. A1 writes attn_normed[t], A2 batch-projects Q/K/V, A3 walks
        // each token's slot. Decode (seq_len=1) → __kv_stride 0 → slot 0.
        for (unsigned int __t = 0; __t < __seq_len; ++__t) {
        params.hidden = __hidden_base + (unsigned long long)__t * __hidden_stride;
        float* const __normed_t =
            __attn_normed_base + (unsigned long long)__t * (unsigned long long)hidden_dim;
        // cu93: NaN sentinel at every layer entry. Scan hidden across block 0.
        if (params.nan_trace != 0 && blockIdx.x == 0) {
            __shared__ int s_nan_flag;
            if (threadIdx.x == 0) s_nan_flag = 0;
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
                if (isnan(params.hidden[i]) || isinf(params.hidden[i])) {
                    atomicOr(&s_nan_flag, 1);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && s_nan_flag != 0) {
                printf("[cu93-NaN] layer=%u hidden has NaN/Inf at entry\n", layer);
            }
        }

        // p0: attn_norm.
        if (layer == params.probe_layer_idx) {
            PHASE_LOG("hidden_in", params.hidden, hidden_dim);
            PHASE_LOG("hidden_in_BEFORE_anything", params.hidden, hidden_dim);
        }
        if (blockIdx.x == 0) {
            persistent_rms_norm(
                params.hidden,
                reinterpret_cast<const float*>(lp.attn_norm),
                __normed_t,
                hidden_dim,
                params.norm_eps,
                // cu76: GGUF Gemma4 weights already have unit_offset baked in
                // (eager's apply_model_norm_into uses unit_offset=false by
                // default). Persistent kernel had hardcoded unit_offset=1
                // which caused double-offset → 1.1x systematic blow-up at
                // every RMSNorm site (attn_norm / post_attn / ffn_norm / etc).
                /*unit_offset=*/0);
        }
        grid.sync();
        if (layer == params.probe_layer_idx && __t == 0u) {
            PHASE_LOG("after_attn_norm_normed", __normed_t, hidden_dim);
            PHASE_LOG("after_attn_norm_hidden", params.hidden, hidden_dim);
            if (params.normed_after_attn_norm_probe != 0) {
                const unsigned int total = gridDim.x * blockDim.x;
                const unsigned int my_idx = blockIdx.x * blockDim.x + threadIdx.x;
                for (unsigned int i = my_idx; i < hidden_dim; i += total) {
                    params.normed_after_attn_norm_probe[i] = __normed_t[i];
                }
                grid.sync();
            }
        }
        }  // end PHASE A1 token loop

        // ===== PHASE A2 (batch QKV projection): Q/K/V over ALL tokens at once
        // via the shared-memory tiled GEMM (input = attn_normed[*], K=hidden).
        // cu105 replaces the cu101 per-token GEMV which re-read every weight row
        // seq_len times (weight-read-bound on long prompts). q4k → smem-tiled;
        // q6k K/V → batch N-GEMV (no q6k smem tiler yet). q_dim / kv_dim are
        // layer-invariant on Gemma4, so the GEMM's row-stride output (== q_dim /
        // kv_dim) matches the host q_dim_max / kv_dim_max slot stride.
        persistent_q4k_gemm_smem(
            __q_buf_base,
            reinterpret_cast<const unsigned char*>(lp.q_weight),
            __attn_normed_base,
            lp.q_dim,
            blocks_per_row_hidden,
            __seq_len,
            params.q_dim_max);  // cu105: q_buf slot stride (q_dim varies by layer)
        grid.sync();
        // cu78: dump Q after projection (head-0, head_dim floats) — slot 0.
        if (layer == params.probe_layer_idx && params.q_proj_probe != 0) {
            const unsigned int total = gridDim.x * blockDim.x;
            const unsigned int my_idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (unsigned int i = my_idx; i < lp.head_dim; i += total) {
                params.q_proj_probe[i] = __q_buf_base[i];
            }
            grid.sync();
        }
        // p1b/c: K/V projection — skipped on shared-KV layers (Gemma4 E2B
        // layers 15-34 reuse anchor 11/14's K and V; their k_cache/v_cache
        // pointers already alias the anchor's device buffers).
        const bool reuse_q = (lp.flags & PERSISTENT_FLAG_REUSE_Q) != 0;
        if (!reuse_q) {
            // cu105: K/V batch projection (attn_normed[*] → k_buf/v_buf batch
            // slots). q4k → smem-tiled GEMM; q6k → batch N-GEMV. grid.sync
            // between K and V so the two smem-tiled GEMMs don't race on the
            // shared input tile (same pattern as the gate/up FFN GEMMs).
            if (lp.flags & PERSISTENT_FLAG_K_Q6K) {
                persistent_q6k_gemm_coop(
                    __k_buf_base,
                    reinterpret_cast<const unsigned char*>(lp.k_weight),
                    __attn_normed_base,
                    lp.kv_dim,
                    blocks_per_row_hidden,
                    __seq_len,
                    0u,
                    params.kv_dim_max);  // cu105: k_buf slot stride
            } else {
                persistent_q4k_gemm_smem(
                    __k_buf_base,
                    reinterpret_cast<const unsigned char*>(lp.k_weight),
                    __attn_normed_base,
                    lp.kv_dim,
                    blocks_per_row_hidden,
                    __seq_len,
                    params.kv_dim_max);  // cu105: k_buf slot stride
            }
            grid.sync();
            if (lp.flags & PERSISTENT_FLAG_V_Q6K) {
                persistent_q6k_gemm_coop(
                    __v_buf_base,
                    reinterpret_cast<const unsigned char*>(lp.v_weight),
                    __attn_normed_base,
                    lp.kv_dim,
                    blocks_per_row_hidden,
                    __seq_len,
                    // cu96 probe: tag = 1 for V projection of layer 0 only.
                    (layer == 0u) ? 1u : 0u,
                    params.kv_dim_max);  // cu105: v_buf slot stride
            } else {
                persistent_q4k_gemm_smem(
                    __v_buf_base,
                    reinterpret_cast<const unsigned char*>(lp.v_weight),
                    __attn_normed_base,
                    lp.kv_dim,
                    blocks_per_row_hidden,
                    __seq_len,
                    params.kv_dim_max);  // cu105: v_buf slot stride
            }
        }
        grid.sync();
        if (layer == params.probe_layer_idx) {
            if (params.k_proj_probe != 0) {
                const unsigned int total = gridDim.x * blockDim.x;
                const unsigned int my_idx = blockIdx.x * blockDim.x + threadIdx.x;
                for (unsigned int i = my_idx; i < lp.head_dim; i += total) {
                    params.k_proj_probe[i] = __k_buf_base[i];
                }
                grid.sync();
            }
            if (params.v_proj_probe != 0) {
                const unsigned int total = gridDim.x * blockDim.x;
                const unsigned int my_idx = blockIdx.x * blockDim.x + threadIdx.x;
                for (unsigned int i = my_idx; i < lp.head_dim; i += total) {
                    params.v_proj_probe[i] = __v_buf_base[i];
                }
                grid.sync();
            }
            PHASE_LOG("after_QKV_hidden", params.hidden, hidden_dim);
            PHASE_LOG("after_Q", __q_buf_base, lp.q_dim);
            PHASE_LOG("after_K", __k_buf_base, lp.kv_dim);
            PHASE_LOG("after_V", __v_buf_base, lp.kv_dim);
        }

        // ===== PHASE A3 (per-token): QK/V norm + RoPE + KV cache write. Each
        // token's Q/K/V slot was produced by the batch GEMM in A2; the RoPE
        // position and KV cache slot differ per token so this stays a token
        // loop. q_buf / k_buf / v_buf are advanced to the token's batch slot.
        for (unsigned int __t = 0; __t < __seq_len; ++__t) {
        params.rope_pos = __rope_pos_base + __t;
        params.q_buf = __q_buf_base + (unsigned long long)__t * __q_stride;
        params.k_buf = __k_buf_base + (unsigned long long)__t * __kv_stride;
        params.v_buf = __v_buf_base + (unsigned long long)__t * __kv_stride;

        // p2-pre: Q/K RMS norm (Qwen-style, present on some Gemma4 layers).
        if (lp.q_norm != 0 && blockIdx.x < q_heads) {
            persistent_per_head_rms_norm(
                params.q_buf,
                reinterpret_cast<const float*>(lp.q_norm),
                lp.head_dim,
                params.norm_eps);
        }
        if (lp.k_norm != 0 && blockIdx.x < kv_heads && !reuse_q) {
            persistent_per_head_rms_norm(
                params.k_buf,
                reinterpret_cast<const float*>(lp.k_norm),
                lp.head_dim,
                params.norm_eps);
        }
        // cu97: V RMS norm (no-scale) for Gemma4. Eager applies
        // `apply_rms_norm_no_scale_into` per-head after V projection (see
        // forward.rs:706-722 with `gemma_v_norm_enabled()` default-true).
        // Persistent decode was missing this — V cache was ~50x larger than
        // eager → attention output exploded → garbage tokens.
        if (params.gemma_v_norm != 0u && blockIdx.x < kv_heads && !reuse_q) {
            persistent_per_head_rms_norm(
                params.v_buf,
                /*weight=*/nullptr,
                lp.head_dim,
                params.norm_eps);
        }
        grid.sync();
        if (layer == params.probe_layer_idx) {
            PHASE_LOG("after_QKnorm_Q", params.q_buf, lp.q_dim);
            PHASE_LOG("after_QKnorm_K", params.k_buf, lp.kv_dim);
        }

        // cu93: NaN check BEFORE RoPE — is Q already broken from projection?
        if (params.nan_trace != 0 && blockIdx.x == 0) {
            __shared__ int s_nan_qpre, s_nan_kpre;
            if (threadIdx.x == 0) { s_nan_qpre = 0; s_nan_kpre = 0; }
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < lp.q_dim; i += blockDim.x) {
                if (isnan(params.q_buf[i]) || isinf(params.q_buf[i])) atomicOr(&s_nan_qpre, 1);
            }
            for (unsigned int i = threadIdx.x; i < lp.kv_dim; i += blockDim.x) {
                if (isnan(params.k_buf[i]) || isinf(params.k_buf[i])) atomicOr(&s_nan_kpre, 1);
            }
            __syncthreads();
            if (threadIdx.x == 0 && (s_nan_qpre != 0 || s_nan_kpre != 0)) {
                printf("[cu93-preRoPE-NaN] layer=%u q=%d k=%d pos=%u\n",
                       layer, s_nan_qpre, s_nan_kpre, params.rope_pos);
            }
        }
        // p2: RoPE on Q and K (one block per head, blocks beyond q_heads exit).
        // cu77: pass freq_factors for FULL-attention layers (SWA layers
        // explicitly skip per huggingface gemma_rope_freq_factors).
        if (blockIdx.x < q_heads) {
            const float* freq_factors_ptr = (lp.sliding_window == 0u)
                ? reinterpret_cast<const float*>(params.rope_freqs)
                : nullptr;
            persistent_rope_neox_qk(
                params.q_buf,
                params.k_buf,
                lp.q_dim,
                lp.kv_dim,
                lp.head_dim,
                params.rope_pos,
                theta_base,
                freq_factors_ptr);
        }
        grid.sync();
        // cu93: NaN check AFTER RoPE — does RoPE introduce NaN at long position?
        if (params.nan_trace != 0 && blockIdx.x == 0) {
            __shared__ int s_nan_qpost, s_nan_kpost;
            if (threadIdx.x == 0) { s_nan_qpost = 0; s_nan_kpost = 0; }
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < lp.q_dim; i += blockDim.x) {
                if (isnan(params.q_buf[i]) || isinf(params.q_buf[i])) atomicOr(&s_nan_qpost, 1);
            }
            for (unsigned int i = threadIdx.x; i < lp.kv_dim; i += blockDim.x) {
                if (isnan(params.k_buf[i]) || isinf(params.k_buf[i])) atomicOr(&s_nan_kpost, 1);
            }
            __syncthreads();
            if (threadIdx.x == 0 && (s_nan_qpost != 0 || s_nan_kpost != 0)) {
                printf("[cu93-postRoPE-NaN] layer=%u q=%d k=%d pos=%u sw=%u\n",
                       layer, s_nan_qpost, s_nan_kpost, params.rope_pos, lp.sliding_window);
            }
        }
        if (layer == params.probe_layer_idx) {
            PHASE_LOG("after_RoPE_hidden", params.hidden, hidden_dim);
            PHASE_LOG("after_RoPE_Q", params.q_buf, lp.q_dim);
            PHASE_LOG("after_RoPE_K", params.k_buf, lp.kv_dim);
        }

        // p3: KV cache write (block 0).  Skipped on shared-KV layers — the
        // anchor layer already wrote the new token's K/V to the shared buffer.
        if (blockIdx.x == 0 && lp.k_cache != 0 && lp.v_cache != 0 && !reuse_q) {
            persistent_kv_cache_write(
                reinterpret_cast<unsigned short*>(lp.k_cache),
                reinterpret_cast<unsigned short*>(lp.v_cache),
                params.k_buf,
                params.v_buf,
                lp.kv_dim,
                params.rope_pos);
        }
        grid.sync();
        }  // end PHASE A3 token loop

        // ===== PHASE B (attention): single batch causal attention call.
        // persistent_attention_prefill processes every (token, head) query at
        // once (grid-stride over num_heads*seq_len blocks); each query t attends
        // causally to keys [j_start .. rope_pos_base + t + 1). This replaces the
        // cu101 per-token attention_decode loop → one grid.sync for the whole
        // batch instead of seq_len of them. q_buf / attn_out are token-major
        // with slot stride q_dim_max; the function writes out[q_offset+lane] for
        // every active lane so no zero-init is needed. Decode (seq_len=1, t=0)
        // is identical to a single-query attention.
        if (lp.k_cache != 0 && lp.v_cache != 0) {
            persistent_attention_prefill(
                __attn_out_base,
                __q_buf_base,
                reinterpret_cast<const unsigned short*>(lp.k_cache),
                reinterpret_cast<const unsigned short*>(lp.v_cache),
                __seq_len,
                __rope_pos_base,
                q_heads,
                kv_heads,
                lp.head_dim,
                head_scale,
                lp.sliding_window,
                params.q_dim_max);
        }
        grid.sync();
        // cu93: attn_out NaN sentinel — scan the last token's slot.
        if (params.nan_trace != 0 && blockIdx.x == 0) {
            const float* attn_src =
                __attn_out_base + (unsigned long long)(__seq_len - 1u) * __q_stride;
            __shared__ int s_nan_attn;
            if (threadIdx.x == 0) s_nan_attn = 0;
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < lp.q_dim; i += blockDim.x) {
                if (isnan(attn_src[i]) || isinf(attn_src[i])) {
                    atomicOr(&s_nan_attn, 1);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && s_nan_attn != 0) {
                printf("[cu93-attn-NaN] layer=%u sliding_window=%u kv_len=%u\n",
                       layer, lp.sliding_window, __rope_pos_base + __seq_len);
            }
        }

        // ===== PHASE C (post-attention): per-token O proj + residual + FFN + PLE.
        for (unsigned int __t = 0; __t < __seq_len; ++__t) {
        params.hidden = __hidden_base + (unsigned long long)__t * __hidden_stride;
        params.attn_out = __attn_out_base + (unsigned long long)__t * __q_stride;

        // p5: O projection — overwrite normed buffer with `O @ attn_out`, then
        // add to hidden (residual).  Skipped (with grid.sync still issued) when
        // `o_weight == 0` so the kernel topology stays valid for tests that
        // populate only the QKV path.
        if (lp.o_weight != 0) {
            const float* o_input = params.attn_out != 0
                ? reinterpret_cast<const float*>(params.attn_out)
                : params.q_buf;
            if (lp.flags & PERSISTENT_FLAG_O_Q6K) {
                persistent_q6k_gemm_coop(
                    params.normed,
                    reinterpret_cast<const unsigned char*>(lp.o_weight),
                    o_input,
                    hidden_dim,
                    lp.q_dim / 256u,
                    /*seq_len=*/1u,
                    0u);
            } else {
                persistent_q4k_gemm_coop(
                    params.normed,
                    reinterpret_cast<const unsigned char*>(lp.o_weight),
                    o_input,
                    hidden_dim,
                    lp.q_dim / 256u,
                    /*seq_len=*/1u);
            }
        }
        grid.sync();
        if (layer == params.probe_layer_idx) PHASE_LOG("after_O", params.normed, hidden_dim);

        if (layer == params.probe_layer_idx) {
            PHASE_LOG("before_post_attn_norm_hidden", params.hidden, hidden_dim);
            PHASE_LOG("before_post_attn_norm_O", params.normed, hidden_dim);
        }
        // cu92: eager order = norm(O) → residual → ffn_norm(hidden).
        // huggingface Gemma2+ pattern (post_attn_norm applied to O proj
        // OUTPUT, not to residual sum). 우리 이전 식 = add-then-norm 잘못.
        // p5b-1: normed = post_attn_norm(O proj = params.normed) in-place.
        if (blockIdx.x == 0 && lp.o_weight != 0 && lp.post_attn_norm != 0) {
            persistent_rms_norm(
                params.normed,
                reinterpret_cast<const float*>(lp.post_attn_norm),
                params.normed,
                hidden_dim,
                params.norm_eps,
                /*unit_offset=*/0);
        }
        grid.sync();
        // p5b-2: hidden += normed (residual, parallel over all blocks).
        if (lp.o_weight != 0) {
            const unsigned int total2 = gridDim.x * blockDim.x;
            const unsigned int my_idx2 = blockIdx.x * blockDim.x + threadIdx.x;
            for (unsigned int i = my_idx2; i < hidden_dim; i += total2) {
                params.hidden[i] += params.normed[i];
            }
        }
        grid.sync();
        // p5b-3: ffn_pre_norm = ffn_norm(hidden) into ffn_normed[__t] (batch
        // slot). The batch FFN phase reads every token's ffn_normed at once.
        if (blockIdx.x == 0 && lp.ffn_norm != 0) {
            persistent_rms_norm(
                params.hidden,
                reinterpret_cast<const float*>(lp.ffn_norm),
                params.ffn_normed + (unsigned long long)__t * hidden_dim,
                hidden_dim,
                params.norm_eps,
                /*unit_offset=*/0);
        }
        grid.sync();
        if (layer == params.probe_layer_idx) {
            PHASE_LOG("after_post_attn_norm_hidden", params.hidden, hidden_dim);
            PHASE_LOG("after_post_attn_norm_normed", params.normed, hidden_dim);
            if (params.hidden_after_attn_probe != 0) {
                const unsigned int total = gridDim.x * blockDim.x;
                const unsigned int my_idx = blockIdx.x * blockDim.x + threadIdx.x;
                for (unsigned int i = my_idx; i < hidden_dim; i += total) {
                    params.hidden_after_attn_probe[i] = params.hidden[i];
                }
                grid.sync();
            }
        }

        }  // end PHASE C1 token loop (pre-FFN per-token work)

        // ===== PHASE C2 (batch FFN): gate / up / GELU·mul / down over ALL
        // tokens at once. ffn_normed[*] → gate_buf[*] / up_buf[*] → gelu·mul →
        // down → ffn_down[*]. cu102 uses the seq_len GEMM wrapper (still inner
        // GEMV per token); cu103 swaps in weight-tile-reuse batch tiling for ROI.
        // gate_buf/up_buf slot stride = n_ff (== n_ff_max for Gemma4); ffn_normed
        // slot stride = hidden_dim; ffn_down slot stride = hidden_dim — all match
        // the GEMM wrapper's out+t*rows / input+t*(blocks*256) indexing.
        // cu103: gate/up via input-smem tiled GEMM (K=hidden fits 24KB smem at
        // BN_TILE=4). down stays on N-GEMV (K=n_ff too large for smem).
        if (lp.gate_weight != 0 && params.gate_buf != 0) {
            persistent_q4k_gemm_smem(
                reinterpret_cast<float*>(params.gate_buf),
                reinterpret_cast<const unsigned char*>(lp.gate_weight),
                params.ffn_normed,
                lp.n_ff,
                blocks_per_row_hidden,
                __seq_len);
        }
        grid.sync();
        if (lp.up_weight != 0 && params.up_buf != 0) {
            persistent_q4k_gemm_smem(
                reinterpret_cast<float*>(params.up_buf),
                reinterpret_cast<const unsigned char*>(lp.up_weight),
                params.ffn_normed,
                lp.n_ff,
                blocks_per_row_hidden,
                __seq_len);
        }
        grid.sync();
        // GELU·mul over all tokens (elementwise): gate_buf[0..seq_len*n_ff].
        if (params.gate_buf != 0 && params.up_buf != 0
            && lp.gate_weight != 0 && lp.up_weight != 0) {
            persistent_gelu_mul(
                reinterpret_cast<float*>(params.gate_buf),
                reinterpret_cast<const float*>(params.up_buf),
                lp.n_ff * __seq_len);
        }
        grid.sync();
        // down: gate_buf[*] → ffn_down[*].
        if (lp.down_weight != 0 && params.gate_buf != 0) {
            const float* down_input =
                reinterpret_cast<const float*>(params.gate_buf);
            if (lp.flags & PERSISTENT_FLAG_DOWN_Q6K) {
                // cu106: BK-tiled input-smem batch GEMM (K=n_ff too large for
                // whole-K smem → tile K into hidden_dim-wide chunks). Replaces
                // the N-GEMV body that re-decoded the weight per token (cu105
                // 49% bottleneck). same-build ABAB: 13960→10065ms (−28%).
                // seq_len<=1 (decode) falls back to GEMV inside.
                persistent_q6k_gemm_smem_bk(
                    params.ffn_down,
                    reinterpret_cast<const unsigned char*>(lp.down_weight),
                    down_input,
                    hidden_dim,
                    lp.n_ff / 256u,
                    __seq_len);
            } else {
                persistent_q4k_gemm_coop(
                    params.ffn_down,
                    reinterpret_cast<const unsigned char*>(lp.down_weight),
                    down_input,
                    hidden_dim,
                    lp.n_ff / 256u,
                    __seq_len);
            }
        }
        grid.sync();

        // ===== PHASE C3 (post-FFN tail): per-token post_ffn_norm + residual +
        // PLE + output scale.
        for (unsigned int __t = 0; __t < __seq_len; ++__t) {
        params.hidden = __hidden_base + (unsigned long long)__t * __hidden_stride;
        float* ffn_down_t = params.ffn_down + (unsigned long long)__t * hidden_dim;

        // cu92/cu102: eager order = post_ffn_norm(down) → residual. down output
        // is ffn_down[__t] (batch slot from PHASE C2); norm into params.normed
        // then add to hidden[__t].
        // p6e-1: normed = post_ffn_norm(ffn_down[__t]) (single block).
        if (blockIdx.x == 0 && lp.down_weight != 0 && lp.post_ffn_norm != 0) {
            persistent_rms_norm(
                ffn_down_t,
                reinterpret_cast<const float*>(lp.post_ffn_norm),
                params.normed,
                hidden_dim,
                params.norm_eps,
                /*unit_offset=*/0);
        }
        grid.sync();
        // p6e-2: hidden[__t] += (post_ffn_norm output, or raw ffn_down if none).
        if (lp.down_weight != 0) {
            const float* resid_src = (lp.post_ffn_norm != 0)
                ? reinterpret_cast<const float*>(params.normed)
                : ffn_down_t;
            const unsigned int total3 = gridDim.x * blockDim.x;
            const unsigned int my_idx3 = blockIdx.x * blockDim.x + threadIdx.x;
            for (unsigned int i = my_idx3; i < hidden_dim; i += total3) {
                params.hidden[i] += resid_src[i];
            }
        }
        grid.sync();

        if (layer == params.probe_layer_idx && params.hidden_after_ffn_probe != 0) {
            const unsigned int total = gridDim.x * blockDim.x;
            const unsigned int my_idx = blockIdx.x * blockDim.x + threadIdx.x;
            for (unsigned int i = my_idx; i < hidden_dim; i += total) {
                params.hidden_after_ffn_probe[i] = params.hidden[i];
            }
            grid.sync();
        }

        // p7: PLE (Per-Layer Embedding).  Only when PLE inputs are present.
        if (lp.ple_gate != 0 && lp.ple_proj != 0 && lp.ple_input != 0
            && params.ple_gate_buf != 0) {
            // cu76: PLE branch input must be `params.hidden` (matches eager's
            // gemma4.rs `apply_gemma4_per_layer_branch`: `gate = inp_gate @
            // hidden`). The previous code used `params.normed` (= last ffn
            // norm output) which is completely wrong source.
            // p7a: ple_gate projection (F32 GEMV) into ple_gate_buf.
            const unsigned int ple_dim = 256u;  // Gemma4 E2B fixed.
            const bool is_f32 = (lp.flags & PERSISTENT_FLAG_PLE_F32) != 0;
            if (is_f32) {
                persistent_f32_gemv_coop(
                    reinterpret_cast<float*>(params.ple_gate_buf),
                    reinterpret_cast<const float*>(lp.ple_gate),
                    params.hidden,
                    ple_dim,
                    hidden_dim);
            } else {
                persistent_q4k_gemm_coop(
                    reinterpret_cast<float*>(params.ple_gate_buf),
                    reinterpret_cast<const unsigned char*>(lp.ple_gate),
                    params.hidden,
                    ple_dim,
                    blocks_per_row_hidden,
                    /*seq_len=*/1u);
            }
            grid.sync();

            // p7b: GELU(ple_gate) * ple_input.
            // cu101 Milestone 2 — batch caller stores per-token PLE bases in
            // a packed buffer (seq_len * ple_dim per layer). Decode (seq_len=1)
            // points lp.ple_input at the single ple_dim slot, so __t * ple_dim
            // == 0 keeps the existing behavior. Batch prefill caller pre-uploads
            // all tokens' PLE bases and the outer token loop walks the offset.
            const unsigned long long ple_token_off =
                (__seq_len > 1u)
                    ? (unsigned long long)__t * (unsigned long long)ple_dim
                    : 0ull;
            persistent_gelu_mul(
                reinterpret_cast<float*>(params.ple_gate_buf),
                reinterpret_cast<const float*>(lp.ple_input) + ple_token_off,
                ple_dim);
            grid.sync();

            // p7c: ple_proj projection — F32 (rows=hidden, cols=ple_dim).
            if (is_f32) {
                persistent_f32_gemv_coop(
                    params.normed,
                    reinterpret_cast<const float*>(lp.ple_proj),
                    reinterpret_cast<const float*>(params.ple_gate_buf),
                    hidden_dim,
                    ple_dim);
            } else {
                persistent_q4k_gemm_coop(
                    params.normed,
                    reinterpret_cast<const unsigned char*>(lp.ple_proj),
                    reinterpret_cast<const float*>(params.ple_gate_buf),
                    hidden_dim,
                    ple_dim / 256u,
                    /*seq_len=*/1u);
            }
            grid.sync();

            // cu76: PLE residual must match eager exactly:
            //   normed = rmsnorm(projected, post_norm)   <- norm `projected`
            //   hidden = hidden + normed                  <- residual AFTER norm
            // The previous code computed `rmsnorm(hidden + projected)` (a
            // different op entirely) and discarded the result while keeping
            // `hidden + projected` as residual.
            // p7d-1: rmsnorm projected (= params.normed) in-place using one block.
            if (blockIdx.x == 0) {
                persistent_rms_norm(
                    params.normed,
                    lp.ple_post_norm != 0
                        ? reinterpret_cast<const float*>(lp.ple_post_norm)
                        : nullptr,
                    params.normed,
                    hidden_dim,
                    params.norm_eps,
                    /*unit_offset=*/0);
            }
            grid.sync();
            // p7d-2: hidden += normed (parallel across all blocks).
            {
                const unsigned int total = gridDim.x * blockDim.x;
                const unsigned int my_idx = blockIdx.x * blockDim.x + threadIdx.x;
                for (unsigned int i = my_idx; i < hidden_dim; i += total) {
                    params.hidden[i] += params.normed[i];
                }
            }
            grid.sync();
        }

        // p7e: layer output scale (Gemma4 E2B-specific).
        persistent_scale_inplace(params.hidden, hidden_dim, lp.layer_output_scale);
        grid.sync();
        // cu90: trace layer hidden max_abs after each layer.
        if (params.layer_hidden_trace != 0 && blockIdx.x == 0 && threadIdx.x == 0) {
            float local_max = 0.0f;
            for (unsigned int i = 0; i < hidden_dim; ++i) {
                float v = params.hidden[i];
                if (v < 0.0f) v = -v;
                if (v > local_max) local_max = v;
            }
            params.layer_hidden_trace[layer] = local_max;
        }
        grid.sync();
        }  // end PHASE C token loop
    }  // end layer loop

    // cu76 diag: dump hidden state after last active layer's full pipeline.
    // Host compares with eager scratch.hidden at same layer cap.
    if (params.hidden_probe != 0) {
        const unsigned int tid = threadIdx.x;
        const unsigned int bid = blockIdx.x;
        const unsigned int block_count = gridDim.x;
        const unsigned int block_threads = blockDim.x;
        const unsigned int total = block_count * block_threads;
        const unsigned int my_idx = bid * block_threads + tid;
        for (unsigned int i = my_idx; i < hidden_dim; i += total) {
            params.hidden_probe[i] = params.hidden[i];
        }
        grid.sync();
    }

    // cu101 M3: output projection for the last token only. The per-token phase
    // loops above already walked hidden per token; point hidden at the last
    // token's slot for the final norm + vocab GEMV. For prefill batch
    // (seq_len > 1) the host consumes only the last token's logits.
    params.hidden = __hidden_base + (unsigned long long)(__seq_len - 1u) * __hidden_stride;
    {
    // cu91: apply output_norm BEFORE output projection. eager 의
    // finalize_decode_logits 가 apply_model_norm_into(hidden, output_norm)
    // → norm_buf → output.gemv(norm_buf). 우리는 이걸 skip 해서 hidden 가
    // unnormalized 상태로 projection → logits 폭주 + wrong argmax.
    if (params.output_norm != 0) {
        if (blockIdx.x == 0) {
            persistent_rms_norm(
                params.hidden,
                reinterpret_cast<const float*>(params.output_norm),
                params.normed,
                hidden_dim,
                params.norm_eps,
                // cu91: Gemma4 E2B output_norm uses unit_offset (eager
                // gemma_effective_unit_offset_output_norm_decode = true
                // for Gemma4 E2B). DIFFERENT from layer norm (unit_offset=0).
                /*unit_offset=*/1);
        }
        grid.sync();
    }

    // p_out: output logits + argmax.
    if (params.nan_trace == 2u && blockIdx.x == 0 && threadIdx.x == 0) {
        printf("[cu102-kernel] output proj entry last_t=%u seq_len=%u out_w=%llu logits=%llu argmax=%llu\n",
               __seq_len - 1u, __seq_len,
               (unsigned long long)params.output_weight,
               (unsigned long long)params.logits,
               (unsigned long long)params.argmax_out);
    }
    if (params.output_weight != 0 && params.logits != 0 && params.argmax_out != 0) {
        persistent_q8_0_gemv_coop(
            reinterpret_cast<float*>(params.logits),
            reinterpret_cast<const unsigned char*>(params.output_weight),
            params.output_norm != 0 ? params.normed : params.hidden,
            params.vocab_size,
            hidden_dim / 32u);
        grid.sync();
        if (blockIdx.x == 0) {
            persistent_argmax_single_block(
                reinterpret_cast<const float*>(params.logits),
                params.vocab_size,
                reinterpret_cast<int*>(params.argmax_out));
            if (params.nan_trace == 2u && threadIdx.x == 0) {
                printf("[cu102-kernel] argmax written = %d\n",
                       *reinterpret_cast<int*>(params.argmax_out));
            }
        }
    }
    }  // end output projection block
    grid.sync();
}
