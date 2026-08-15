#pragma once

// Adapted from llama.cpp/ggml-cuda's current Ampere MMQ activation contract
// (mmq.cuh block_q8_1_mmq and quantize.cu quantize_mmq_q8_1).  This local
// implementation deliberately supports only the non-fallback, no-MUL_MAT_ID
// dense Q4_K/Q6_K J=128 path and has no llama.cpp build dependency.
//
// A block owns 128 values: four 32-value Q8 groups plus sixteen bytes of
// metadata/padding.  The launcher stores blocks as [k128][sequence], so the
// J=128 MMQ CTA can copy its activation tile contiguously into shared memory.
struct RnbBlockQ8_1Mmq {
    union {
        float d4[4];
        __half2 ds4[4];
    };
    signed char qs[128];
};

static_assert(sizeof(RnbBlockQ8_1Mmq) == 144, "RnbBlockQ8_1Mmq must be 144 bytes");

__device__ __forceinline__ void rnb_quantize_q8_1_mmq_body(
    const float* __restrict__ input,
    RnbBlockQ8_1Mmq* __restrict__ output,
    unsigned cols,
    unsigned seq_len,
    bool with_sums) {
    const unsigned lane = threadIdx.x;
    const unsigned k128 = blockIdx.x;
    const unsigned seq = blockIdx.y;
    if (lane >= 128u || seq >= seq_len) {
        return;
    }

    const unsigned col = k128 * 128u + lane;
    const float value = col < cols ? input[(unsigned long long)seq * cols + col] : 0.0f;
    const unsigned lane32 = lane & 31u;
    float amax = fabsf(value);
#pragma unroll
    for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    float sum = value;
    if (with_sums) {
#pragma unroll
        for (unsigned offset = 16u; offset > 0u; offset >>= 1u) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
    }

    const float d = amax == 0.0f ? 0.0f : amax / 127.0f;
    const signed char q = d == 0.0f ? 0 : static_cast<signed char>(roundf(value / d));
    RnbBlockQ8_1Mmq* const destination = output + (unsigned long long)k128 * seq_len + seq;
    destination->qs[lane] = q;
    if (lane32 == 0u) {
        const unsigned group = lane >> 5;
        if (with_sums) {
            destination->ds4[group] = __floats2half2_rn(d, sum);
        } else {
            destination->d4[group] = d;
        }
    }
}

extern "C" __global__ void rnb_quantize_q8_1_mmq_q4(
    const float* __restrict__ input,
    RnbBlockQ8_1Mmq* __restrict__ output,
    unsigned cols,
    unsigned seq_len) {
    rnb_quantize_q8_1_mmq_body(input, output, cols, seq_len, true);
}

extern "C" __global__ void rnb_quantize_q8_1_mmq_q6(
    const float* __restrict__ input,
    RnbBlockQ8_1Mmq* __restrict__ output,
    unsigned cols,
    unsigned seq_len) {
    rnb_quantize_q8_1_mmq_body(input, output, cols, seq_len, false);
}
