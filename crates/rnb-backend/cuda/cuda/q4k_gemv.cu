#include <cuda_fp16.h>
#include <mma.h>

#include "kernels/quant_gemv.cuh"
#include "kernels/quant_batch_and_ops.cuh"
#include "kernels/gdn_attention.cuh"
#include "kernels/kvarn_attention.cuh"
#include "kernels/mma_flash.cuh"
#include "kernels/prefill_post.cuh"
#include "kernels/qwen_selected_gemv.cuh"
#include "kernels/selected_down.cuh"
#include "kernels/glm_selected_gemv.cuh"
#include "kernels/grouped_down.cuh"
#include "kernels/sequence.cuh"

template <typename Decoder>
__device__ __forceinline__ void rnb_quant_embedding_gather_body(
    float* __restrict__ out,
    const unsigned char* __restrict__ weights,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row,
    unsigned token_count) {
    const unsigned token_idx = blockIdx.x;
    const unsigned block_idx = blockIdx.y;
    const unsigned index = threadIdx.x;
    const unsigned block_elems = Decoder::block_elems();
    if (token_idx >= token_count || block_idx >= blocks_per_row || index >= block_elems) {
        return;
    }
    const unsigned row = token_ids[token_idx];
    if (row >= rows) {
        return;
    }
    const unsigned block_bytes = Decoder::block_bytes();
    const unsigned char* block =
        weights
        + (unsigned long long)row * blocks_per_row * block_bytes
        + block_idx * block_bytes;
    out[(unsigned long long)token_idx * blocks_per_row * block_elems
        + block_idx * block_elems + index] = Decoder::value(block, index);
}

struct RnbF32Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 128u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned offset = index * 4u;
        const unsigned raw =
            (unsigned)block[offset]
            | ((unsigned)block[offset + 1u] << 8)
            | ((unsigned)block[offset + 2u] << 16)
            | ((unsigned)block[offset + 3u] << 24);
        return __uint_as_float(raw);
    }
};

struct RnbQ50Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 22u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned qh =
            (unsigned)block[2]
            | ((unsigned)block[3] << 8)
            | ((unsigned)block[4] << 16)
            | ((unsigned)block[5] << 24);
        const unsigned packed = block[6u + (index & 15u)];
        const unsigned low = index < 16u ? (packed & 0x0fu) : (packed >> 4);
        const unsigned q = low | (((qh >> index) & 1u) << 4);
        return ((float)q - 16.0f) * d;
    }
};

struct RnbQ51Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 24u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_m = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float m = __half2float(__ushort_as_half((unsigned short)raw_m));
        const unsigned qh =
            (unsigned)block[4]
            | ((unsigned)block[5] << 8)
            | ((unsigned)block[6] << 16)
            | ((unsigned)block[7] << 24);
        const unsigned packed = block[8u + (index & 15u)];
        const unsigned low = index < 16u ? (packed & 0x0fu) : (packed >> 4);
        const unsigned q = low | (((qh >> index) & 1u) << 4);
        return (float)q * d + m;
    }
};

struct RnbQ80Decoder {
    __device__ static __forceinline__ unsigned block_elems() { return 32u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 34u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        return (float)(signed char)block[2u + index] * d;
    }
};

struct RnbQ4KEmbeddingDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 144u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_q4k_value_at(block, index);
    }
};

struct RnbQ5KEmbeddingDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 176u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_q5k_value_at(block, index);
    }
};

struct RnbQ6KEmbeddingDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 210u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_q6k_value_at(block, index);
    }
};

struct RnbIq2XxsEmbeddingDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 66u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_iq2_xxs_value(block, index);
    }
};

struct RnbIq2SEmbeddingDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 82u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_iq2_s_value(block, index);
    }
};

struct RnbIq3XxsEmbeddingDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 98u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_iq3_xxs_value(block, index);
    }
};

struct RnbIq4XsEmbeddingDecoder {
    __device__ static __forceinline__ unsigned block_elems() { return 256u; }
    __device__ static __forceinline__ unsigned block_bytes() { return 136u; }
    __device__ static __forceinline__ float value(
        const unsigned char* block,
        unsigned index) {
        return rnb_glm_iq4_xs_value(block, index);
    }
};

#define RNB_DEFINE_QUANT_EMBEDDING_KERNEL(NAME, DECODER) \
extern "C" __global__ void NAME( \
    float* out, const unsigned char* weights, const unsigned* token_ids, \
    unsigned rows, unsigned blocks_per_row, unsigned token_count) { \
    rnb_quant_embedding_gather_body<DECODER>( \
        out, weights, token_ids, rows, blocks_per_row, token_count); \
}

RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_f32_embedding_gather, RnbF32Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_f16_embedding_gather, RnbF16Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_bf16_embedding_gather, RnbBf16Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q4_0_embedding_gather, RnbQ40Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q4_1_embedding_gather, RnbQ41Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q5_0_embedding_gather, RnbQ50Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q5_1_embedding_gather, RnbQ51Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q8_0_embedding_gather, RnbQ80Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q8_1_embedding_gather, RnbQ81Decoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q2k_embedding_gather, RnbQ2KDecoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q3k_embedding_gather, RnbQ3KDecoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q4k_embedding_gather, RnbQ4KEmbeddingDecoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q5k_embedding_gather, RnbQ5KEmbeddingDecoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_q6k_embedding_gather, RnbQ6KEmbeddingDecoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(
    rnb_quant_iq2_xxs_embedding_gather, RnbIq2XxsEmbeddingDecoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_iq2_s_embedding_gather, RnbIq2SEmbeddingDecoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(
    rnb_quant_iq3_xxs_embedding_gather, RnbIq3XxsEmbeddingDecoder)
RNB_DEFINE_QUANT_EMBEDDING_KERNEL(rnb_quant_iq4_xs_embedding_gather, RnbIq4XsEmbeddingDecoder)

#undef RNB_DEFINE_QUANT_EMBEDDING_KERNEL
