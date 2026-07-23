__device__ __forceinline__ float rnb_q5k_value_at(
    const unsigned char* __restrict__ block,
    unsigned tid) {
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
    const unsigned j = tid >> 5;
    unsigned sc;
    unsigned mn;
    if (j < 4u) {
        sc = block[4 + j] & 63u;
        mn = block[4 + j + 4] & 63u;
    } else {
        sc = (block[4 + j + 4] & 0x0fu) | ((block[4 + j - 4] >> 6) << 4);
        mn = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
    }
    const unsigned local = tid & 63u;
    const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
    unsigned q = block[48 + q_index];
    q = local < 32u ? (q & 0x0fu) : (q >> 4);
    const unsigned high = (block[16 + (tid & 31u)] >> (tid >> 5)) & 1u;
    q |= high << 4;
    return (d * (float)sc) * (float)q - dmin * (float)mn;
}

__device__ __forceinline__ float rnb_q6k_value_at(
    const unsigned char* __restrict__ block,
    unsigned tid) {
    const unsigned n = tid >> 7;
    const unsigned rem = tid & 127u;
    const unsigned l = rem & 31u;
    const unsigned is = l >> 4;
    const unsigned ql_base = n * 64u;
    const unsigned qh_base = 128u + n * 32u;
    const unsigned sc_base = 192u + n * 8u;
    const unsigned qh = block[qh_base + l];
    unsigned q;
    int sc;
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
    const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    return d * (float)sc * (float)((int)q - 32);
}

__device__ __forceinline__ float rnb_q4k_value_at(
    const unsigned char* __restrict__ block,
    unsigned tid) {
    const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
    const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
    const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
    const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
    const unsigned j = tid >> 5;
    unsigned sc;
    unsigned mn;
    if (j < 4u) {
        sc = block[4 + j] & 63u;
        mn = block[4 + j + 4] & 63u;
    } else {
        sc = (block[4 + j + 4] & 0x0fu) | ((block[4 + j - 4] >> 6) << 4);
        mn = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
    }
    const unsigned local = tid & 63u;
    const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
    unsigned q = block[16 + q_index];
    q = local < 32u ? (q & 0x0fu) : (q >> 4);
    return (d * (float)sc) * (float)q - dmin * (float)mn;
}

template <unsigned ROW_BYTES, float (*ValueAt)(const unsigned char*, unsigned)>
__device__ __forceinline__ void rnb_selected_down_group2_warp4_impl(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 2u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned slot_stride = blocks_per_row * 256u;
    const unsigned base0 = slot_start * slot_stride;
    const unsigned base1 = base0 + slot_stride;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * ROW_BYTES;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned x_base = b * 256u + lane;
        const unsigned char* block = row_ptr + b * ROW_BYTES;
        const float y0 = ValueAt(block, lane + 0u);
        const float y1 = ValueAt(block, lane + 32u);
        const float y2 = ValueAt(block, lane + 64u);
        const float y3 = ValueAt(block, lane + 96u);
        const float y4 = ValueAt(block, lane + 128u);
        const float y5 = ValueAt(block, lane + 160u);
        const float y6 = ValueAt(block, lane + 192u);
        const float y7 = ValueAt(block, lane + 224u);

        acc0 += y0 * input[base0 + x_base + 0u]
              + y1 * input[base0 + x_base + 32u]
              + y2 * input[base0 + x_base + 64u]
              + y3 * input[base0 + x_base + 96u]
              + y4 * input[base0 + x_base + 128u]
              + y5 * input[base0 + x_base + 160u]
              + y6 * input[base0 + x_base + 192u]
              + y7 * input[base0 + x_base + 224u];
        if (group_len > 1u) {
            acc1 += y0 * input[base1 + x_base + 0u]
                  + y1 * input[base1 + x_base + 32u]
                  + y2 * input[base1 + x_base + 64u]
                  + y3 * input[base1 + x_base + 96u]
                  + y4 * input[base1 + x_base + 128u]
                  + y5 * input[base1 + x_base + 160u]
                  + y6 * input[base1 + x_base + 192u]
                  + y7 * input[base1 + x_base + 224u];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
    }
}

__device__ __forceinline__ float rnb_accum8_default_order(
    float acc,
    const float y0,
    const float y1,
    const float y2,
    const float y3,
    const float y4,
    const float y5,
    const float y6,
    const float y7,
    const float* __restrict__ input,
    const unsigned x_base) {
    acc += y0 * input[x_base + 0u];
    acc += y1 * input[x_base + 32u];
    acc += y2 * input[x_base + 64u];
    acc += y3 * input[x_base + 96u];
    acc += y4 * input[x_base + 128u];
    acc += y5 * input[x_base + 160u];
    acc += y6 * input[x_base + 192u];
    acc += y7 * input[x_base + 224u];
    return acc;
}

template <bool EXACT_ORDER, unsigned ROW_BYTES, float (*ValueAt)(const unsigned char*, unsigned)>
__device__ __forceinline__ void rnb_selected_down_group4_warp4_impl(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * ROW_BYTES;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const float y0 = ValueAt(row_ptr + b * ROW_BYTES, lane + 0u);
        const float y1 = ValueAt(row_ptr + b * ROW_BYTES, lane + 32u);
        const float y2 = ValueAt(row_ptr + b * ROW_BYTES, lane + 64u);
        const float y3 = ValueAt(row_ptr + b * ROW_BYTES, lane + 96u);
        const float y4 = ValueAt(row_ptr + b * ROW_BYTES, lane + 128u);
        const float y5 = ValueAt(row_ptr + b * ROW_BYTES, lane + 160u);
        const float y6 = ValueAt(row_ptr + b * ROW_BYTES, lane + 192u);
        const float y7 = ValueAt(row_ptr + b * ROW_BYTES, lane + 224u);
        const unsigned x_base = b * 256u + lane;

        const float* input0 = input + slot_start * blocks_per_row * 256u;
        if (EXACT_ORDER) {
            acc0 = rnb_accum8_default_order(
                acc0, y0, y1, y2, y3, y4, y5, y6, y7, input0, x_base);
        } else {
            acc0 += y0 * input0[x_base + 0u]
                  + y1 * input0[x_base + 32u]
                  + y2 * input0[x_base + 64u]
                  + y3 * input0[x_base + 96u]
                  + y4 * input0[x_base + 128u]
                  + y5 * input0[x_base + 160u]
                  + y6 * input0[x_base + 192u]
                  + y7 * input0[x_base + 224u];
        }
        if (group_len > 1u) {
            const float* input1 = input + (slot_start + 1u) * blocks_per_row * 256u;
            if (EXACT_ORDER) {
                acc1 = rnb_accum8_default_order(
                    acc1, y0, y1, y2, y3, y4, y5, y6, y7, input1, x_base);
            } else {
                acc1 += y0 * input1[x_base + 0u]
                      + y1 * input1[x_base + 32u]
                      + y2 * input1[x_base + 64u]
                      + y3 * input1[x_base + 96u]
                      + y4 * input1[x_base + 128u]
                      + y5 * input1[x_base + 160u]
                      + y6 * input1[x_base + 192u]
                      + y7 * input1[x_base + 224u];
            }
        }
        if (group_len > 2u) {
            const float* input2 = input + (slot_start + 2u) * blocks_per_row * 256u;
            if (EXACT_ORDER) {
                acc2 = rnb_accum8_default_order(
                    acc2, y0, y1, y2, y3, y4, y5, y6, y7, input2, x_base);
            } else {
                acc2 += y0 * input2[x_base + 0u]
                      + y1 * input2[x_base + 32u]
                      + y2 * input2[x_base + 64u]
                      + y3 * input2[x_base + 96u]
                      + y4 * input2[x_base + 128u]
                      + y5 * input2[x_base + 160u]
                      + y6 * input2[x_base + 192u]
                      + y7 * input2[x_base + 224u];
            }
        }
        if (group_len > 3u) {
            const float* input3 = input + (slot_start + 3u) * blocks_per_row * 256u;
            if (EXACT_ORDER) {
                acc3 = rnb_accum8_default_order(
                    acc3, y0, y1, y2, y3, y4, y5, y6, y7, input3, x_base);
            } else {
                acc3 += y0 * input3[x_base + 0u]
                      + y1 * input3[x_base + 32u]
                      + y2 * input3[x_base + 64u]
                      + y3 * input3[x_base + 96u]
                      + y4 * input3[x_base + 128u]
                      + y5 * input3[x_base + 160u]
                      + y6 * input3[x_base + 192u]
                      + y7 * input3[x_base + 224u];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

template <unsigned ROW_BYTES, float (*ValueAt)(const unsigned char*, unsigned)>
__device__ __forceinline__ void rnb_selected_down_group8_warp4_impl(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 8u) {
        return;
    }

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * ROW_BYTES;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const float y0 = ValueAt(row_ptr + b * ROW_BYTES, lane + 0u);
        const float y1 = ValueAt(row_ptr + b * ROW_BYTES, lane + 32u);
        const float y2 = ValueAt(row_ptr + b * ROW_BYTES, lane + 64u);
        const float y3 = ValueAt(row_ptr + b * ROW_BYTES, lane + 96u);
        const float y4 = ValueAt(row_ptr + b * ROW_BYTES, lane + 128u);
        const float y5 = ValueAt(row_ptr + b * ROW_BYTES, lane + 160u);
        const float y6 = ValueAt(row_ptr + b * ROW_BYTES, lane + 192u);
        const float y7 = ValueAt(row_ptr + b * ROW_BYTES, lane + 224u);
        const unsigned x_base = b * 256u + lane;

        const float* input0 = input + slot_start * blocks_per_row * 256u;
        acc0 += y0 * input0[x_base + 0u]
              + y1 * input0[x_base + 32u]
              + y2 * input0[x_base + 64u]
              + y3 * input0[x_base + 96u]
              + y4 * input0[x_base + 128u]
              + y5 * input0[x_base + 160u]
              + y6 * input0[x_base + 192u]
              + y7 * input0[x_base + 224u];
        if (group_len > 1u) {
            const float* input1 = input + (slot_start + 1u) * blocks_per_row * 256u;
            acc1 += y0 * input1[x_base + 0u]
                  + y1 * input1[x_base + 32u]
                  + y2 * input1[x_base + 64u]
                  + y3 * input1[x_base + 96u]
                  + y4 * input1[x_base + 128u]
                  + y5 * input1[x_base + 160u]
                  + y6 * input1[x_base + 192u]
                  + y7 * input1[x_base + 224u];
        }
        if (group_len > 2u) {
            const float* input2 = input + (slot_start + 2u) * blocks_per_row * 256u;
            acc2 += y0 * input2[x_base + 0u]
                  + y1 * input2[x_base + 32u]
                  + y2 * input2[x_base + 64u]
                  + y3 * input2[x_base + 96u]
                  + y4 * input2[x_base + 128u]
                  + y5 * input2[x_base + 160u]
                  + y6 * input2[x_base + 192u]
                  + y7 * input2[x_base + 224u];
        }
        if (group_len > 3u) {
            const float* input3 = input + (slot_start + 3u) * blocks_per_row * 256u;
            acc3 += y0 * input3[x_base + 0u]
                  + y1 * input3[x_base + 32u]
                  + y2 * input3[x_base + 64u]
                  + y3 * input3[x_base + 96u]
                  + y4 * input3[x_base + 128u]
                  + y5 * input3[x_base + 160u]
                  + y6 * input3[x_base + 192u]
                  + y7 * input3[x_base + 224u];
        }
        if (group_len > 4u) {
            const float* input4 = input + (slot_start + 4u) * blocks_per_row * 256u;
            acc4 += y0 * input4[x_base + 0u]
                  + y1 * input4[x_base + 32u]
                  + y2 * input4[x_base + 64u]
                  + y3 * input4[x_base + 96u]
                  + y4 * input4[x_base + 128u]
                  + y5 * input4[x_base + 160u]
                  + y6 * input4[x_base + 192u]
                  + y7 * input4[x_base + 224u];
        }
        if (group_len > 5u) {
            const float* input5 = input + (slot_start + 5u) * blocks_per_row * 256u;
            acc5 += y0 * input5[x_base + 0u]
                  + y1 * input5[x_base + 32u]
                  + y2 * input5[x_base + 64u]
                  + y3 * input5[x_base + 96u]
                  + y4 * input5[x_base + 128u]
                  + y5 * input5[x_base + 160u]
                  + y6 * input5[x_base + 192u]
                  + y7 * input5[x_base + 224u];
        }
        if (group_len > 6u) {
            const float* input6 = input + (slot_start + 6u) * blocks_per_row * 256u;
            acc6 += y0 * input6[x_base + 0u]
                  + y1 * input6[x_base + 32u]
                  + y2 * input6[x_base + 64u]
                  + y3 * input6[x_base + 96u]
                  + y4 * input6[x_base + 128u]
                  + y5 * input6[x_base + 160u]
                  + y6 * input6[x_base + 192u]
                  + y7 * input6[x_base + 224u];
        }
        if (group_len > 7u) {
            const float* input7 = input + (slot_start + 7u) * blocks_per_row * 256u;
            acc7 += y0 * input7[x_base + 0u]
                  + y1 * input7[x_base + 32u]
                  + y2 * input7[x_base + 64u]
                  + y3 * input7[x_base + 96u]
                  + y4 * input7[x_base + 128u]
                  + y5 * input7[x_base + 160u]
                  + y6 * input7[x_base + 192u]
                  + y7 * input7[x_base + 224u];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
        acc4 += __shfl_down_sync(0xffffffffu, acc4, offset);
        acc5 += __shfl_down_sync(0xffffffffu, acc5, offset);
        acc6 += __shfl_down_sync(0xffffffffu, acc6, offset);
        acc7 += __shfl_down_sync(0xffffffffu, acc7, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
        if (group_len > 4u) {
            atomicAdd(out + token_ids[slot_start + 4u] * rows + row, acc4 * route[slot_start + 4u]);
        }
        if (group_len > 5u) {
            atomicAdd(out + token_ids[slot_start + 5u] * rows + row, acc5 * route[slot_start + 5u]);
        }
        if (group_len > 6u) {
            atomicAdd(out + token_ids[slot_start + 6u] * rows + row, acc6 * route[slot_start + 6u]);
        }
        if (group_len > 7u) {
            atomicAdd(out + token_ids[slot_start + 7u] * rows + row, acc7 * route[slot_start + 7u]);
        }
    }
}

extern "C" __global__ void rnb_q4k_selected_down_accum_by_token_group8_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    rnb_selected_down_group8_warp4_impl<144u, rnb_q4k_value_at>(
        out, weights, input, route, token_ids, group_meta, rows, blocks_per_row);
}

extern "C" __global__ void rnb_q5k_selected_down_accum_by_token_group8_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    rnb_selected_down_group8_warp4_impl<176u, rnb_q5k_value_at>(
        out, weights, input, route, token_ids, group_meta, rows, blocks_per_row);
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group8_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    rnb_selected_down_group8_warp4_impl<210u, rnb_q6k_value_at>(
        out, weights, input, route, token_ids, group_meta, rows, blocks_per_row);
}

extern "C" __global__ void rnb_q4k_selected_down_accum_by_token_group2_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    rnb_selected_down_group2_warp4_impl<144u, rnb_q4k_value_at>(
        out, weights, input, route, token_ids, group_meta, rows, blocks_per_row);
}

extern "C" __global__ void rnb_q5k_selected_down_accum_by_token_group2_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    rnb_selected_down_group2_warp4_impl<176u, rnb_q5k_value_at>(
        out, weights, input, route, token_ids, group_meta, rows, blocks_per_row);
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group2_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    rnb_selected_down_group2_warp4_impl<210u, rnb_q6k_value_at>(
        out, weights, input, route, token_ids, group_meta, rows, blocks_per_row);
}

extern "C" __global__ void rnb_q4k_selected_down_accum_by_token_group4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    rnb_selected_down_group4_warp4_impl<true, 144u, rnb_q4k_value_at>(
        out, weights, input, route, token_ids, group_meta, rows, blocks_per_row);
}

__device__ __forceinline__ void rnb_q4k_selected_down_q8dot_accum(
    int weight_pack,
    float weight_scale,
    float weight_min,
    const signed char* __restrict__ input_qs,
    float input_d,
    float& acc) {
    const int input_pack = rnb_load_i32_aligned4(input_qs);
    const int input_sum = __dp4a(0x01010101, input_pack, 0);
    acc += input_d * (weight_scale * (float)__dp4a(weight_pack, input_pack, 0)
        - weight_min * (float)input_sum);
}

extern "C" __global__ void rnb_q4k_selected_down_accum_by_token_group4_q8dot_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned slot_stride_qs = blocks_per_row * 256u;
    const unsigned slot_stride_ds = blocks_per_row * 8u;
    const unsigned base0_qs = slot_start * slot_stride_qs;
    const unsigned base1_qs = base0_qs + slot_stride_qs;
    const unsigned base2_qs = base1_qs + slot_stride_qs;
    const unsigned base3_qs = base2_qs + slot_stride_qs;
    const unsigned base0_ds = slot_start * slot_stride_ds;
    const unsigned base1_ds = base0_ds + slot_stride_ds;
    const unsigned base2_ds = base1_ds + slot_stride_ds;
    const unsigned base3_ds = base2_ds + slot_stride_ds;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 144u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 144u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));

        for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
            const unsigned j = chunk >> 3;
            unsigned sc;
            unsigned mn;
            if (j < 4u) {
                sc = block[4u + j] & 63u;
                mn = block[4u + j + 4u] & 63u;
            } else {
                sc = (block[4u + j + 4u] & 0x0fu)
                    | ((block[4u + j - 4u] >> 6) << 4);
                mn = (block[4u + j + 4u] >> 4)
                    | ((block[4u + j] >> 6) << 4);
            }

            const unsigned elem = (chunk & 7u) * 4u;
            const unsigned q_index = (j >> 1) * 32u + elem;
            const int weight_pack = rnb_q4_pack4(block + 16u + q_index, j);
            const float weight_scale = d * (float)sc;
            const float weight_min = dmin * (float)mn;
            const unsigned qs_off = b * 256u + j * 32u + elem;
            const unsigned ds_off = b * 8u + j;

            rnb_q4k_selected_down_q8dot_accum(
                weight_pack, weight_scale, weight_min,
                input_qs + base0_qs + qs_off, input_ds[base0_ds + ds_off], acc0);
            if (group_len > 1u) {
                rnb_q4k_selected_down_q8dot_accum(
                    weight_pack, weight_scale, weight_min,
                    input_qs + base1_qs + qs_off, input_ds[base1_ds + ds_off], acc1);
            }
            if (group_len > 2u) {
                rnb_q4k_selected_down_q8dot_accum(
                    weight_pack, weight_scale, weight_min,
                    input_qs + base2_qs + qs_off, input_ds[base2_ds + ds_off], acc2);
            }
            if (group_len > 3u) {
                rnb_q4k_selected_down_q8dot_accum(
                    weight_pack, weight_scale, weight_min,
                    input_qs + base3_qs + qs_off, input_ds[base3_ds + ds_off], acc3);
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q5k_selected_down_accum_q8dot_mmq_group16(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
#if __CUDA_ARCH__ < 750
    (void)out;
    (void)weights;
    (void)input_qs;
    (void)input_ds;
    (void)route;
    (void)token_ids;
    (void)group_meta;
    (void)rows;
    (void)blocks_per_row;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned row_base = blockIdx.x * 32u;
    const unsigned group = blockIdx.y;
    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (warp >= 2u || group_len == 0u || group_len > 16u) {
        return;
    }

    __shared__ signed char weight_tile[32 * 32];
    __shared__ signed char input_tile[16 * 32];
    __shared__ float weight_d[32];
    __shared__ float weight_dmin[32];
    __shared__ unsigned char weight_sc[32];
    __shared__ unsigned char weight_mn[32];
    __shared__ float activation_d[16];

    const unsigned warp_row_off = warp * 16u;
    const unsigned frag_row_a = lane >> 2;
    const unsigned frag_row_b = frag_row_a + 8u;
    const unsigned local_slot_a = (lane & 3u) << 1;
    const unsigned local_slot_b = local_slot_a + 1u;
    const unsigned local_slot_c = local_slot_a + 8u;
    const unsigned local_slot_d = local_slot_b + 8u;
    const unsigned row_a = row_base + warp_row_off + frag_row_a;
    const unsigned row_b = row_base + warp_row_off + frag_row_b;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;
    const unsigned row_bytes = blocks_per_row * 176u;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        for (unsigned sub = 0; sub < 8u; ++sub) {
            if (tid < 32u) {
                const unsigned global_row = row_base + tid;
                if (global_row < rows) {
                    const unsigned char* block =
                        weights[slot_start] + global_row * row_bytes + block_idx * 176u;
                    const unsigned raw_d =
                        (unsigned)block[0] | ((unsigned)block[1] << 8);
                    const unsigned raw_dmin =
                        (unsigned)block[2] | ((unsigned)block[3] << 8);
                    weight_d[tid] =
                        __half2float(__ushort_as_half((unsigned short)raw_d));
                    weight_dmin[tid] =
                        __half2float(__ushort_as_half((unsigned short)raw_dmin));
                    if (sub < 4u) {
                        weight_sc[tid] = block[4u + sub] & 63u;
                        weight_mn[tid] = block[8u + sub] & 63u;
                    } else {
                        weight_sc[tid] = (block[8u + sub] & 0x0fu)
                            | ((block[sub] >> 6) << 4);
                        weight_mn[tid] = (block[8u + sub] >> 4)
                            | ((block[4u + sub] >> 6) << 4);
                    }
                } else {
                    weight_d[tid] = 0.0f;
                    weight_dmin[tid] = 0.0f;
                    weight_sc[tid] = 0u;
                    weight_mn[tid] = 0u;
                }
            }

            for (unsigned load = tid; load < 1024u; load += 64u) {
                const unsigned load_row = load >> 5;
                const unsigned elem = load & 31u;
                const unsigned global_row = row_base + load_row;
                signed char value = 0;
                if (global_row < rows) {
                    const unsigned char* block =
                        weights[slot_start] + global_row * row_bytes + block_idx * 176u;
                    const unsigned q_index = (sub >> 1) * 32u + elem;
                    const unsigned packed = block[48u + q_index];
                    const unsigned low = (sub & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
                    const unsigned high = (block[16u + elem] >> sub) & 1u;
                    value = (signed char)(low | (high << 4));
                }
                weight_tile[load] = value;
            }

            for (unsigned load = tid; load < 512u; load += 64u) {
                const unsigned local_slot = load >> 5;
                const unsigned elem = load & 31u;
                if (local_slot < group_len) {
                    const unsigned chunk = block_idx * 8u + sub;
                    input_tile[load] =
                        input_qs[(slot_start + local_slot) * blocks_per_row * 256u
                            + chunk * 32u + elem];
                    if (elem == 0u) {
                        activation_d[local_slot] =
                            input_ds[(slot_start + local_slot) * blocks_per_row * 8u + chunk];
                    }
                } else {
                    input_tile[load] = 0;
                    if (elem == 0u) {
                        activation_d[local_slot] = 0.0f;
                    }
                }
            }
            __syncthreads();

            const unsigned a_col_lo = (lane & 3u) * 4u;
            const unsigned a_col_hi = a_col_lo + 16u;
            const int a0 = *reinterpret_cast<const int*>(
                &weight_tile[(warp_row_off + frag_row_a) * 32u + a_col_lo]);
            const int a1 = *reinterpret_cast<const int*>(
                &weight_tile[(warp_row_off + frag_row_b) * 32u + a_col_lo]);
            const int a2 = *reinterpret_cast<const int*>(
                &weight_tile[(warp_row_off + frag_row_a) * 32u + a_col_hi]);
            const int a3 = *reinterpret_cast<const int*>(
                &weight_tile[(warp_row_off + frag_row_b) * 32u + a_col_hi]);
            const unsigned b_slot = lane >> 2;
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
            const int b0 =
                *reinterpret_cast<const int*>(&input_tile[b_slot * 32u + b_col_lo]);
            const int b1 =
                *reinterpret_cast<const int*>(&input_tile[b_slot * 32u + b_col_hi]);
            const int b2 =
                *reinterpret_cast<const int*>(&input_tile[(b_slot + 8u) * 32u + b_col_lo]);
            const int b3 =
                *reinterpret_cast<const int*>(&input_tile[(b_slot + 8u) * 32u + b_col_hi]);

            int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
            int dot4 = 0, dot5 = 0, dot6 = 0, dot7 = 0;
            rnb_mma_m16n8k32_s8(
                dot0, dot1, dot2, dot3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);
            rnb_mma_m16n8k32_s8(
                dot4, dot5, dot6, dot7, a0, a1, a2, a3, b2, b3, 0, 0, 0, 0);

            int sum_a = 0;
            int sum_b = 0;
            int sum_c = 0;
            int sum_d = 0;
#pragma unroll
            for (int k = 0; k < 32; k += 4) {
                if (local_slot_a < group_len) {
                    sum_a = __dp4a(
                        0x01010101,
                        *reinterpret_cast<const int*>(&input_tile[local_slot_a * 32u + k]),
                        sum_a);
                }
                if (local_slot_b < group_len) {
                    sum_b = __dp4a(
                        0x01010101,
                        *reinterpret_cast<const int*>(&input_tile[local_slot_b * 32u + k]),
                        sum_b);
                }
                if (local_slot_c < group_len) {
                    sum_c = __dp4a(
                        0x01010101,
                        *reinterpret_cast<const int*>(&input_tile[local_slot_c * 32u + k]),
                        sum_c);
                }
                if (local_slot_d < group_len) {
                    sum_d = __dp4a(
                        0x01010101,
                        *reinterpret_cast<const int*>(&input_tile[local_slot_d * 32u + k]),
                        sum_d);
                }
            }

            const unsigned local_row_a = warp_row_off + frag_row_a;
            const unsigned local_row_b = warp_row_off + frag_row_b;
#define RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(ACC, DOT, SLOT, SUM, ROW) \
            if (SLOT < group_len) { \
                ACC += activation_d[SLOT] \
                    * (weight_d[ROW] * (float)weight_sc[ROW] * (float)DOT \
                        - weight_dmin[ROW] * (float)weight_mn[ROW] * (float)SUM); \
            }
            RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(acc[0], dot0, local_slot_a, sum_a, local_row_a);
            RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(acc[1], dot1, local_slot_b, sum_b, local_row_a);
            RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(acc[2], dot2, local_slot_a, sum_a, local_row_b);
            RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(acc[3], dot3, local_slot_b, sum_b, local_row_b);
            RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(acc[4], dot4, local_slot_c, sum_c, local_row_a);
            RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(acc[5], dot5, local_slot_d, sum_d, local_row_a);
            RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(acc[6], dot6, local_slot_c, sum_c, local_row_b);
            RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM(acc[7], dot7, local_slot_d, sum_d, local_row_b);
#undef RNB_Q5K_SELECTED_DOWN_MMQ_ACCUM
            __syncthreads();
        }
    }

#define RNB_Q5K_SELECTED_DOWN_MMQ_STORE(ACC, SLOT, ROW, VALID) \
    if (VALID && SLOT < group_len) { \
        const unsigned slot = slot_start + SLOT; \
        out[slot * rows + ROW] = ACC * route[slot]; \
    }
    RNB_Q5K_SELECTED_DOWN_MMQ_STORE(acc[0], local_slot_a, row_a, row_a_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ_STORE(acc[1], local_slot_b, row_a, row_a_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ_STORE(acc[2], local_slot_a, row_b, row_b_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ_STORE(acc[3], local_slot_b, row_b, row_b_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ_STORE(acc[4], local_slot_c, row_a, row_a_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ_STORE(acc[5], local_slot_d, row_a, row_a_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ_STORE(acc[6], local_slot_c, row_b, row_b_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ_STORE(acc[7], local_slot_d, row_b, row_b_valid);
#undef RNB_Q5K_SELECTED_DOWN_MMQ_STORE
#endif
}

extern "C" __global__ void rnb_q5k_selected_down_accum_q8dot_mmq_group32(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
#if __CUDA_ARCH__ < 750
    (void)out;
    (void)weights;
    (void)input_qs;
    (void)input_ds;
    (void)route;
    (void)token_ids;
    (void)group_meta;
    (void)rows;
    (void)blocks_per_row;
    return;
#else
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5;
    const unsigned lane = tid & 31u;
    const unsigned row_base = blockIdx.x * 32u;
    const unsigned group = blockIdx.y;
    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 64u) {
        return;
    }

    __shared__ signed char weight_tile[32 * 32];
    __shared__ signed char input_tile[64 * 32];
    __shared__ float weight_d[32];
    __shared__ float weight_dmin[32];
    __shared__ unsigned char weight_sc[32];
    __shared__ unsigned char weight_mn[32];
    __shared__ float activation_d[64];

    const unsigned warp_row_off = (warp & 1u) * 16u;
    const unsigned warp_slot_off = (warp >> 1) * 8u;
    const unsigned frag_row_a = lane >> 2;
    const unsigned frag_row_b = frag_row_a + 8u;
    const unsigned local_slot_a = warp_slot_off + ((lane & 3u) << 1);
    const unsigned local_slot_b = local_slot_a + 1u;
    const unsigned row_a = row_base + warp_row_off + frag_row_a;
    const unsigned row_b = row_base + warp_row_off + frag_row_b;
    const bool row_a_valid = row_a < rows;
    const bool row_b_valid = row_b < rows;
    const unsigned row_bytes = blocks_per_row * 176u;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (unsigned block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        for (unsigned sub = 0; sub < 8u; ++sub) {
            if (tid < 32u) {
                const unsigned global_row = row_base + tid;
                if (global_row < rows) {
                    const unsigned char* block =
                        weights[slot_start] + global_row * row_bytes + block_idx * 176u;
                    const unsigned raw_d =
                        (unsigned)block[0] | ((unsigned)block[1] << 8);
                    const unsigned raw_dmin =
                        (unsigned)block[2] | ((unsigned)block[3] << 8);
                    weight_d[tid] =
                        __half2float(__ushort_as_half((unsigned short)raw_d));
                    weight_dmin[tid] =
                        __half2float(__ushort_as_half((unsigned short)raw_dmin));
                    if (sub < 4u) {
                        weight_sc[tid] = block[4u + sub] & 63u;
                        weight_mn[tid] = block[8u + sub] & 63u;
                    } else {
                        weight_sc[tid] = (block[8u + sub] & 0x0fu)
                            | ((block[sub] >> 6) << 4);
                        weight_mn[tid] = (block[8u + sub] >> 4)
                            | ((block[4u + sub] >> 6) << 4);
                    }
                } else {
                    weight_d[tid] = 0.0f;
                    weight_dmin[tid] = 0.0f;
                    weight_sc[tid] = 0u;
                    weight_mn[tid] = 0u;
                }
            }

            for (unsigned load = tid; load < 1024u; load += blockDim.x) {
                const unsigned load_row = load >> 5;
                const unsigned elem = load & 31u;
                const unsigned global_row = row_base + load_row;
                signed char value = 0;
                if (global_row < rows) {
                    const unsigned char* block =
                        weights[slot_start] + global_row * row_bytes + block_idx * 176u;
                    const unsigned q_index = (sub >> 1) * 32u + elem;
                    const unsigned packed = block[48u + q_index];
                    const unsigned low =
                        (sub & 1u) == 0u ? (packed & 0x0fu) : (packed >> 4);
                    const unsigned high = (block[16u + elem] >> sub) & 1u;
                    value = (signed char)(low | (high << 4));
                }
                weight_tile[load] = value;
            }

            for (unsigned load = tid; load < blockDim.x * 4u; load += blockDim.x) {
                const unsigned local_slot = load >> 5;
                const unsigned elem = load & 31u;
                if (local_slot < group_len) {
                    const unsigned chunk = block_idx * 8u + sub;
                    input_tile[load] =
                        input_qs[(slot_start + local_slot) * blocks_per_row * 256u
                            + chunk * 32u + elem];
                    if (elem == 0u) {
                        activation_d[local_slot] =
                            input_ds[(slot_start + local_slot) * blocks_per_row * 8u + chunk];
                    }
                } else {
                    input_tile[load] = 0;
                    if (elem == 0u) {
                        activation_d[local_slot] = 0.0f;
                    }
                }
            }
            __syncthreads();

            const unsigned a_col_lo = (lane & 3u) * 4u;
            const unsigned a_col_hi = a_col_lo + 16u;
            const int a0 = *reinterpret_cast<const int*>(
                &weight_tile[(warp_row_off + frag_row_a) * 32u + a_col_lo]);
            const int a1 = *reinterpret_cast<const int*>(
                &weight_tile[(warp_row_off + frag_row_b) * 32u + a_col_lo]);
            const int a2 = *reinterpret_cast<const int*>(
                &weight_tile[(warp_row_off + frag_row_a) * 32u + a_col_hi]);
            const int a3 = *reinterpret_cast<const int*>(
                &weight_tile[(warp_row_off + frag_row_b) * 32u + a_col_hi]);
            const unsigned b_slot = warp_slot_off + (lane >> 2);
            const unsigned b_col_lo = (lane & 3u) * 4u;
            const unsigned b_col_hi = b_col_lo + 16u;
            const int b0 =
                *reinterpret_cast<const int*>(&input_tile[b_slot * 32u + b_col_lo]);
            const int b1 =
                *reinterpret_cast<const int*>(&input_tile[b_slot * 32u + b_col_hi]);

            int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
            rnb_mma_m16n8k32_s8(
                dot0, dot1, dot2, dot3, a0, a1, a2, a3, b0, b1, 0, 0, 0, 0);

            int sum_a = 0;
            int sum_b = 0;
#pragma unroll
            for (int k = 0; k < 32; k += 4) {
                if (local_slot_a < group_len) {
                    sum_a = __dp4a(
                        0x01010101,
                        *reinterpret_cast<const int*>(&input_tile[local_slot_a * 32u + k]),
                        sum_a);
                }
                if (local_slot_b < group_len) {
                    sum_b = __dp4a(
                        0x01010101,
                        *reinterpret_cast<const int*>(&input_tile[local_slot_b * 32u + k]),
                        sum_b);
                }
            }

            const unsigned local_row_a = warp_row_off + frag_row_a;
            const unsigned local_row_b = warp_row_off + frag_row_b;
#define RNB_Q5K_SELECTED_DOWN_MMQ32_ACCUM(ACC, DOT, SLOT, SUM, ROW) \
            if (SLOT < group_len) { \
                ACC += activation_d[SLOT] \
                    * (weight_d[ROW] * (float)weight_sc[ROW] * (float)DOT \
                        - weight_dmin[ROW] * (float)weight_mn[ROW] * (float)SUM); \
            }
            RNB_Q5K_SELECTED_DOWN_MMQ32_ACCUM(
                acc[0], dot0, local_slot_a, sum_a, local_row_a);
            RNB_Q5K_SELECTED_DOWN_MMQ32_ACCUM(
                acc[1], dot1, local_slot_b, sum_b, local_row_a);
            RNB_Q5K_SELECTED_DOWN_MMQ32_ACCUM(
                acc[2], dot2, local_slot_a, sum_a, local_row_b);
            RNB_Q5K_SELECTED_DOWN_MMQ32_ACCUM(
                acc[3], dot3, local_slot_b, sum_b, local_row_b);
#undef RNB_Q5K_SELECTED_DOWN_MMQ32_ACCUM
            __syncthreads();
        }
    }

#define RNB_Q5K_SELECTED_DOWN_MMQ32_STORE(ACC, SLOT, ROW, VALID) \
    if (VALID && SLOT < group_len) { \
        const unsigned slot = slot_start + SLOT; \
        out[slot * rows + ROW] = ACC * route[slot]; \
    }
    RNB_Q5K_SELECTED_DOWN_MMQ32_STORE(acc[0], local_slot_a, row_a, row_a_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ32_STORE(acc[1], local_slot_b, row_a, row_a_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ32_STORE(acc[2], local_slot_a, row_b, row_b_valid);
    RNB_Q5K_SELECTED_DOWN_MMQ32_STORE(acc[3], local_slot_b, row_b, row_b_valid);
#undef RNB_Q5K_SELECTED_DOWN_MMQ32_STORE
#endif
}

extern "C" __global__ void rnb_q5k_selected_down_reduce_slots_deterministic(
    float* __restrict__ out,
    const float* __restrict__ slot_outputs,
    const unsigned* __restrict__ token_offsets,
    const unsigned* __restrict__ slot_indices,
    unsigned rows) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned token = blockIdx.y;
    if (row >= rows) {
        return;
    }
    float acc = 0.0f;
    for (unsigned idx = token_offsets[token]; idx < token_offsets[token + 1u]; ++idx) {
        acc += slot_outputs[slot_indices[idx] * rows + row];
    }
    out[token * rows + row] += acc;
}

extern "C" __global__ void rnb_q5k_selected_down_accum_by_token_group4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    rnb_selected_down_group4_warp4_impl<false, 176u, rnb_q5k_value_at>(
        out, weights, input, route, token_ids, group_meta, rows, blocks_per_row);
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 210u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

        float y[8];
        #pragma unroll
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned tid = lane + i * 32u;
            const unsigned n = tid >> 7;
            const unsigned rem = tid & 127u;
            const unsigned l = rem & 31u;
            const unsigned is = l >> 4;
            const unsigned ql_base = n * 64u;
            const unsigned qh_base = 128u + n * 32u;
            const unsigned sc_base = 192u + n * 8u;
            const unsigned qh = block[qh_base + l];
            unsigned q;
            int sc;
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
            y[i] = d * (float)sc * (float)((int)q - 32);
        }

        const unsigned x_base = b * 256u + lane;
        const float* input0 = input + slot_start * blocks_per_row * 256u;
        acc0 += y[0] * input0[x_base + 0u]
              + y[1] * input0[x_base + 32u]
              + y[2] * input0[x_base + 64u]
              + y[3] * input0[x_base + 96u]
              + y[4] * input0[x_base + 128u]
              + y[5] * input0[x_base + 160u]
              + y[6] * input0[x_base + 192u]
              + y[7] * input0[x_base + 224u];
        if (group_len > 1u) {
            const float* input1 = input + (slot_start + 1u) * blocks_per_row * 256u;
            acc1 += y[0] * input1[x_base + 0u]
                  + y[1] * input1[x_base + 32u]
                  + y[2] * input1[x_base + 64u]
                  + y[3] * input1[x_base + 96u]
                  + y[4] * input1[x_base + 128u]
                  + y[5] * input1[x_base + 160u]
                  + y[6] * input1[x_base + 192u]
                  + y[7] * input1[x_base + 224u];
        }
        if (group_len > 2u) {
            const float* input2 = input + (slot_start + 2u) * blocks_per_row * 256u;
            acc2 += y[0] * input2[x_base + 0u]
                  + y[1] * input2[x_base + 32u]
                  + y[2] * input2[x_base + 64u]
                  + y[3] * input2[x_base + 96u]
                  + y[4] * input2[x_base + 128u]
                  + y[5] * input2[x_base + 160u]
                  + y[6] * input2[x_base + 192u]
                  + y[7] * input2[x_base + 224u];
        }
        if (group_len > 3u) {
            const float* input3 = input + (slot_start + 3u) * blocks_per_row * 256u;
            acc3 += y[0] * input3[x_base + 0u]
                  + y[1] * input3[x_base + 32u]
                  + y[2] * input3[x_base + 64u]
                  + y[3] * input3[x_base + 96u]
                  + y[4] * input3[x_base + 128u]
                  + y[5] * input3[x_base + 160u]
                  + y[6] * input3[x_base + 192u]
                  + y[7] * input3[x_base + 224u];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group4_pack4_f32_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 210u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

        float y[8];
        #pragma unroll
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned tid = lane + i * 32u;
            const unsigned n = tid >> 7;
            const unsigned rem = tid & 127u;
            const unsigned l = rem & 31u;
            const unsigned is = l >> 4;
            const unsigned ql_base = n * 64u;
            const unsigned qh_base = 128u + n * 32u;
            const unsigned sc_base = 192u + n * 8u;
            const unsigned qh = block[qh_base + l];
            unsigned q;
            int sc;
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
            y[i] = d * (float)sc * (float)((int)q - 32);
        }

        const unsigned pack_base = ((group * blocks_per_row + b) * 256u + lane) * 4u;
        acc0 += y[0] * input[pack_base + 0u]
              + y[1] * input[pack_base + 32u * 4u + 0u]
              + y[2] * input[pack_base + 64u * 4u + 0u]
              + y[3] * input[pack_base + 96u * 4u + 0u]
              + y[4] * input[pack_base + 128u * 4u + 0u]
              + y[5] * input[pack_base + 160u * 4u + 0u]
              + y[6] * input[pack_base + 192u * 4u + 0u]
              + y[7] * input[pack_base + 224u * 4u + 0u];
        if (group_len > 1u) {
            acc1 += y[0] * input[pack_base + 1u]
                  + y[1] * input[pack_base + 32u * 4u + 1u]
                  + y[2] * input[pack_base + 64u * 4u + 1u]
                  + y[3] * input[pack_base + 96u * 4u + 1u]
                  + y[4] * input[pack_base + 128u * 4u + 1u]
                  + y[5] * input[pack_base + 160u * 4u + 1u]
                  + y[6] * input[pack_base + 192u * 4u + 1u]
                  + y[7] * input[pack_base + 224u * 4u + 1u];
        }
        if (group_len > 2u) {
            acc2 += y[0] * input[pack_base + 2u]
                  + y[1] * input[pack_base + 32u * 4u + 2u]
                  + y[2] * input[pack_base + 64u * 4u + 2u]
                  + y[3] * input[pack_base + 96u * 4u + 2u]
                  + y[4] * input[pack_base + 128u * 4u + 2u]
                  + y[5] * input[pack_base + 160u * 4u + 2u]
                  + y[6] * input[pack_base + 192u * 4u + 2u]
                  + y[7] * input[pack_base + 224u * 4u + 2u];
        }
        if (group_len > 3u) {
            acc3 += y[0] * input[pack_base + 3u]
                  + y[1] * input[pack_base + 32u * 4u + 3u]
                  + y[2] * input[pack_base + 64u * 4u + 3u]
                  + y[3] * input[pack_base + 96u * 4u + 3u]
                  + y[4] * input[pack_base + 128u * 4u + 3u]
                  + y[5] * input[pack_base + 160u * 4u + 3u]
                  + y[6] * input[pack_base + 192u * 4u + 3u]
                  + y[7] * input[pack_base + 224u * 4u + 3u];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group4_pack4_f32_vec4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 210u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

        float y[8];
        #pragma unroll
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned tid = lane + i * 32u;
            const unsigned n = tid >> 7;
            const unsigned rem = tid & 127u;
            const unsigned l = rem & 31u;
            const unsigned is = l >> 4;
            const unsigned ql_base = n * 64u;
            const unsigned qh_base = 128u + n * 32u;
            const unsigned sc_base = 192u + n * 8u;
            const unsigned qh = block[qh_base + l];
            unsigned q;
            int sc;
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
            y[i] = d * (float)sc * (float)((int)q - 32);
        }

        const unsigned pack_base = ((group * blocks_per_row + b) * 256u + lane) * 4u;
        const float4* act = reinterpret_cast<const float4*>(input + pack_base);
        const float4 act0 = act[0u];
        const float4 act1 = act[32u];
        const float4 act2 = act[64u];
        const float4 act3 = act[96u];
        const float4 act4 = act[128u];
        const float4 act5 = act[160u];
        const float4 act6 = act[192u];
        const float4 act7 = act[224u];
        acc0 += y[0] * act0.x
              + y[1] * act1.x
              + y[2] * act2.x
              + y[3] * act3.x
              + y[4] * act4.x
              + y[5] * act5.x
              + y[6] * act6.x
              + y[7] * act7.x;
        if (group_len > 1u) {
            acc1 += y[0] * act0.y
                  + y[1] * act1.y
                  + y[2] * act2.y
                  + y[3] * act3.y
                  + y[4] * act4.y
                  + y[5] * act5.y
                  + y[6] * act6.y
                  + y[7] * act7.y;
        }
        if (group_len > 2u) {
            acc2 += y[0] * act0.z
                  + y[1] * act1.z
                  + y[2] * act2.z
                  + y[3] * act3.z
                  + y[4] * act4.z
                  + y[5] * act5.z
                  + y[6] * act6.z
                  + y[7] * act7.z;
        }
        if (group_len > 3u) {
            acc3 += y[0] * act0.w
                  + y[1] * act1.w
                  + y[2] * act2.w
                  + y[3] * act3.w
                  + y[4] * act4.w
                  + y[5] * act5.w
                  + y[6] * act6.w
                  + y[7] * act7.w;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_run_batched_ref_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ run_tile_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned tile = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = run_tile_meta[tile * 2u + 0u];
    const unsigned group_len = run_tile_meta[tile * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 210u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

        float y[8];
        #pragma unroll
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned tid = lane + i * 32u;
            const unsigned n = tid >> 7;
            const unsigned rem = tid & 127u;
            const unsigned l = rem & 31u;
            const unsigned is = l >> 4;
            const unsigned ql_base = n * 64u;
            const unsigned qh_base = 128u + n * 32u;
            const unsigned sc_base = 192u + n * 8u;
            const unsigned qh = block[qh_base + l];
            unsigned q;
            int sc;
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
            y[i] = d * (float)sc * (float)((int)q - 32);
        }

        const unsigned x_base = b * 256u + lane;
        const float* input0 = input + slot_start * blocks_per_row * 256u;
        acc0 += y[0] * input0[x_base + 0u]
              + y[1] * input0[x_base + 32u]
              + y[2] * input0[x_base + 64u]
              + y[3] * input0[x_base + 96u]
              + y[4] * input0[x_base + 128u]
              + y[5] * input0[x_base + 160u]
              + y[6] * input0[x_base + 192u]
              + y[7] * input0[x_base + 224u];
        if (group_len > 1u) {
            const float* input1 = input + (slot_start + 1u) * blocks_per_row * 256u;
            acc1 += y[0] * input1[x_base + 0u]
                  + y[1] * input1[x_base + 32u]
                  + y[2] * input1[x_base + 64u]
                  + y[3] * input1[x_base + 96u]
                  + y[4] * input1[x_base + 128u]
                  + y[5] * input1[x_base + 160u]
                  + y[6] * input1[x_base + 192u]
                  + y[7] * input1[x_base + 224u];
        }
        if (group_len > 2u) {
            const float* input2 = input + (slot_start + 2u) * blocks_per_row * 256u;
            acc2 += y[0] * input2[x_base + 0u]
                  + y[1] * input2[x_base + 32u]
                  + y[2] * input2[x_base + 64u]
                  + y[3] * input2[x_base + 96u]
                  + y[4] * input2[x_base + 128u]
                  + y[5] * input2[x_base + 160u]
                  + y[6] * input2[x_base + 192u]
                  + y[7] * input2[x_base + 224u];
        }
        if (group_len > 3u) {
            const float* input3 = input + (slot_start + 3u) * blocks_per_row * 256u;
            acc3 += y[0] * input3[x_base + 0u]
                  + y[1] * input3[x_base + 32u]
                  + y[2] * input3[x_base + 64u]
                  + y[3] * input3[x_base + 96u]
                  + y[4] * input3[x_base + 128u]
                  + y[5] * input3[x_base + 160u]
                  + y[6] * input3[x_base + 192u]
                  + y[7] * input3[x_base + 224u];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_run_tiled4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ run_tile_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned tile = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned meta_base = tile * 5u;
    const unsigned run_start = run_tile_meta[meta_base + 1u];
    const unsigned run_len = run_tile_meta[meta_base + 2u];
    const unsigned tile_start = run_tile_meta[meta_base + 3u];
    const unsigned tile_len = run_tile_meta[meta_base + 4u];
    if (tile_len == 0u || tile_len > 4u || tile_start < run_start ||
        tile_start + tile_len > run_start + run_len) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned slot_stride = blocks_per_row * 256u;
    const unsigned base0 = tile_start * slot_stride;
    const unsigned base1 = base0 + slot_stride;
    const unsigned base2 = base1 + slot_stride;
    const unsigned base3 = base2 + slot_stride;
    const unsigned char* row_ptr = weights[run_start] + row * blocks_per_row * 210u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

        float y[8];
        #pragma unroll
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned tid = lane + i * 32u;
            const unsigned n = tid >> 7;
            const unsigned rem = tid & 127u;
            const unsigned l = rem & 31u;
            const unsigned is = l >> 4;
            const unsigned ql_base = n * 64u;
            const unsigned qh_base = 128u + n * 32u;
            const unsigned sc_base = 192u + n * 8u;
            const unsigned qh = block[qh_base + l];
            unsigned q;
            int sc;
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
            y[i] = d * (float)sc * (float)((int)q - 32);
        }

        const unsigned x_base = b * 256u + lane;
        acc0 += y[0] * input[base0 + x_base + 0u]
              + y[1] * input[base0 + x_base + 32u]
              + y[2] * input[base0 + x_base + 64u]
              + y[3] * input[base0 + x_base + 96u]
              + y[4] * input[base0 + x_base + 128u]
              + y[5] * input[base0 + x_base + 160u]
              + y[6] * input[base0 + x_base + 192u]
              + y[7] * input[base0 + x_base + 224u];
        if (tile_len > 1u) {
            acc1 += y[0] * input[base1 + x_base + 0u]
                  + y[1] * input[base1 + x_base + 32u]
                  + y[2] * input[base1 + x_base + 64u]
                  + y[3] * input[base1 + x_base + 96u]
                  + y[4] * input[base1 + x_base + 128u]
                  + y[5] * input[base1 + x_base + 160u]
                  + y[6] * input[base1 + x_base + 192u]
                  + y[7] * input[base1 + x_base + 224u];
        }
        if (tile_len > 2u) {
            acc2 += y[0] * input[base2 + x_base + 0u]
                  + y[1] * input[base2 + x_base + 32u]
                  + y[2] * input[base2 + x_base + 64u]
                  + y[3] * input[base2 + x_base + 96u]
                  + y[4] * input[base2 + x_base + 128u]
                  + y[5] * input[base2 + x_base + 160u]
                  + y[6] * input[base2 + x_base + 192u]
                  + y[7] * input[base2 + x_base + 224u];
        }
        if (tile_len > 3u) {
            acc3 += y[0] * input[base3 + x_base + 0u]
                  + y[1] * input[base3 + x_base + 32u]
                  + y[2] * input[base3 + x_base + 64u]
                  + y[3] * input[base3 + x_base + 96u]
                  + y[4] * input[base3 + x_base + 128u]
                  + y[5] * input[base3 + x_base + 160u]
                  + y[6] * input[base3 + x_base + 192u]
                  + y[7] * input[base3 + x_base + 224u];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[tile_start] * rows + row, acc0 * route[tile_start]);
        if (tile_len > 1u) {
            atomicAdd(out + token_ids[tile_start + 1u] * rows + row, acc1 * route[tile_start + 1u]);
        }
        if (tile_len > 2u) {
            atomicAdd(out + token_ids[tile_start + 2u] * rows + row, acc2 * route[tile_start + 2u]);
        }
        if (tile_len > 3u) {
            atomicAdd(out + token_ids[tile_start + 3u] * rows + row, acc3 * route[tile_start + 3u]);
        }
    }
}

__device__ __forceinline__ float rnb_q6k_value_at_with_d(
    const unsigned char* __restrict__ block,
    unsigned tid,
    float d) {
    const unsigned n = tid >> 7;
    const unsigned rem = tid & 127u;
    const unsigned l = rem & 31u;
    const unsigned is = l >> 4;
    const unsigned ql_base = n * 64u;
    const unsigned qh_base = 128u + n * 32u;
    const unsigned sc_base = 192u + n * 8u;
    const unsigned qh = block[qh_base + l];
    unsigned q;
    int sc;
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
    return d * (float)sc * (float)((int)q - 32);
}

extern "C" __global__ void rnb_q6k_selected_down_accum_token_major_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_offsets,
    const unsigned* __restrict__ slot_indices,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned token = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    const unsigned slot_begin = token_offsets[token];
    const unsigned slot_end = token_offsets[token + 1u];
    const unsigned slot_stride = blocks_per_row * 256u;
    float token_acc = 0.0f;

    for (unsigned pos = slot_begin; pos < slot_end; ++pos) {
        const unsigned slot = slot_indices[pos];
        const unsigned char* row_ptr = weights[slot] + row * blocks_per_row * 210u;
        const float* slot_input = input + slot * slot_stride;
        float acc = 0.0f;

        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const unsigned x_base = b * 256u + lane;

            #pragma unroll
            for (unsigned i = 0; i < 8u; ++i) {
                const unsigned x = x_base + i * 32u;
                const float y = rnb_q6k_value_at_with_d(block, lane + i * 32u, d);
                acc += y * slot_input[x];
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, offset);
        }
        if (lane == 0u) {
            token_acc += acc * route[slot];
        }
    }

    if (lane == 0u) {
        out[token * rows + row] += token_acc;
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group4_full_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len != 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned slot_stride = blocks_per_row * 256u;
    const unsigned base0 = slot_start * slot_stride;
    const unsigned base1 = base0 + slot_stride;
    const unsigned base2 = base1 + slot_stride;
    const unsigned base3 = base2 + slot_stride;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 210u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned x_base = b * 256u + lane;

        #pragma unroll
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned x = x_base + i * 32u;
            const float y = rnb_q6k_value_at_with_d(block, lane + i * 32u, d);
            acc0 += y * input[base0 + x];
            acc1 += y * input[base1 + x];
            acc2 += y * input[base2 + x];
            acc3 += y * input[base3 + x];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group4_fast4_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned slot_stride = blocks_per_row * 256u;
    const unsigned base0 = slot_start * slot_stride;
    const unsigned base1 = base0 + slot_stride;
    const unsigned base2 = base1 + slot_stride;
    const unsigned base3 = base2 + slot_stride;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 210u;

    if (group_len == 4u) {
        for (unsigned b = 0; b < blocks_per_row; ++b) {
            const unsigned char* block = row_ptr + b * 210u;
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

            float y[8];
            #pragma unroll
            for (unsigned i = 0; i < 8u; ++i) {
                const unsigned tid = lane + i * 32u;
                const unsigned n = tid >> 7;
                const unsigned rem = tid & 127u;
                const unsigned l = rem & 31u;
                const unsigned is = l >> 4;
                const unsigned ql_base = n * 64u;
                const unsigned qh_base = 128u + n * 32u;
                const unsigned sc_base = 192u + n * 8u;
                const unsigned qh = block[qh_base + l];
                unsigned q;
                int sc;
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
                y[i] = d * (float)sc * (float)((int)q - 32);
            }

            const unsigned x_base = b * 256u + lane;
            acc0 += y[0] * input[base0 + x_base + 0u]
                  + y[1] * input[base0 + x_base + 32u]
                  + y[2] * input[base0 + x_base + 64u]
                  + y[3] * input[base0 + x_base + 96u]
                  + y[4] * input[base0 + x_base + 128u]
                  + y[5] * input[base0 + x_base + 160u]
                  + y[6] * input[base0 + x_base + 192u]
                  + y[7] * input[base0 + x_base + 224u];
            acc1 += y[0] * input[base1 + x_base + 0u]
                  + y[1] * input[base1 + x_base + 32u]
                  + y[2] * input[base1 + x_base + 64u]
                  + y[3] * input[base1 + x_base + 96u]
                  + y[4] * input[base1 + x_base + 128u]
                  + y[5] * input[base1 + x_base + 160u]
                  + y[6] * input[base1 + x_base + 192u]
                  + y[7] * input[base1 + x_base + 224u];
            acc2 += y[0] * input[base2 + x_base + 0u]
                  + y[1] * input[base2 + x_base + 32u]
                  + y[2] * input[base2 + x_base + 64u]
                  + y[3] * input[base2 + x_base + 96u]
                  + y[4] * input[base2 + x_base + 128u]
                  + y[5] * input[base2 + x_base + 160u]
                  + y[6] * input[base2 + x_base + 192u]
                  + y[7] * input[base2 + x_base + 224u];
            acc3 += y[0] * input[base3 + x_base + 0u]
                  + y[1] * input[base3 + x_base + 32u]
                  + y[2] * input[base3 + x_base + 64u]
                  + y[3] * input[base3 + x_base + 96u]
                  + y[4] * input[base3 + x_base + 128u]
                  + y[5] * input[base3 + x_base + 160u]
                  + y[6] * input[base3 + x_base + 192u]
                  + y[7] * input[base3 + x_base + 224u];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
            acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
            acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
        }

        if (lane == 0u) {
            atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
        return;
    }

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

        float y[8];
        #pragma unroll
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned tid = lane + i * 32u;
            const unsigned n = tid >> 7;
            const unsigned rem = tid & 127u;
            const unsigned l = rem & 31u;
            const unsigned is = l >> 4;
            const unsigned ql_base = n * 64u;
            const unsigned qh_base = 128u + n * 32u;
            const unsigned sc_base = 192u + n * 8u;
            const unsigned qh = block[qh_base + l];
            unsigned q;
            int sc;
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
            y[i] = d * (float)sc * (float)((int)q - 32);
        }

        const unsigned x_base = b * 256u + lane;
        acc0 += y[0] * input[base0 + x_base + 0u]
              + y[1] * input[base0 + x_base + 32u]
              + y[2] * input[base0 + x_base + 64u]
              + y[3] * input[base0 + x_base + 96u]
              + y[4] * input[base0 + x_base + 128u]
              + y[5] * input[base0 + x_base + 160u]
              + y[6] * input[base0 + x_base + 192u]
              + y[7] * input[base0 + x_base + 224u];
        if (group_len > 1u) {
            acc1 += y[0] * input[base1 + x_base + 0u]
                  + y[1] * input[base1 + x_base + 32u]
                  + y[2] * input[base1 + x_base + 64u]
                  + y[3] * input[base1 + x_base + 96u]
                  + y[4] * input[base1 + x_base + 128u]
                  + y[5] * input[base1 + x_base + 160u]
                  + y[6] * input[base1 + x_base + 192u]
                  + y[7] * input[base1 + x_base + 224u];
        }
        if (group_len > 2u) {
            acc2 += y[0] * input[base2 + x_base + 0u]
                  + y[1] * input[base2 + x_base + 32u]
                  + y[2] * input[base2 + x_base + 64u]
                  + y[3] * input[base2 + x_base + 96u]
                  + y[4] * input[base2 + x_base + 128u]
                  + y[5] * input[base2 + x_base + 160u]
                  + y[6] * input[base2 + x_base + 192u]
                  + y[7] * input[base2 + x_base + 224u];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group4_q8dot_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const signed char* __restrict__ input_qs,
    const float* __restrict__ input_ds,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned slot_stride_qs = blocks_per_row * 256u;
    const unsigned slot_stride_ds = blocks_per_row * 8u;
    const unsigned base0_qs = slot_start * slot_stride_qs;
    const unsigned base1_qs = base0_qs + slot_stride_qs;
    const unsigned base2_qs = base1_qs + slot_stride_qs;
    const unsigned base3_qs = base2_qs + slot_stride_qs;
    const unsigned base0_ds = slot_start * slot_stride_ds;
    const unsigned base1_ds = base0_ds + slot_stride_ds;
    const unsigned base2_ds = base1_ds + slot_stride_ds;
    const unsigned base3_ds = base2_ds + slot_stride_ds;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 210u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));

        for (unsigned chunk = lane; chunk < 64u; chunk += 32u) {
            const unsigned elem = chunk * 4u;
            const unsigned n = elem >> 7;
            const unsigned rem = elem & 127u;
            const unsigned l = rem & 31u;
            const unsigned is = l >> 4;
            const unsigned ql_base = n * 64u;
            const unsigned qh_base = 128u + n * 32u;
            const unsigned sc_base = 192u + n * 8u;
            int sc;
            if (rem < 32u) {
                sc = (int)((signed char)block[sc_base + is]);
            } else if (rem < 64u) {
                sc = (int)((signed char)block[sc_base + is + 2u]);
            } else if (rem < 96u) {
                sc = (int)((signed char)block[sc_base + is + 4u]);
            } else {
                sc = (int)((signed char)block[sc_base + is + 6u]);
            }

            int q_pack = 0;
            #pragma unroll
            for (unsigned k = 0; k < 4u; ++k) {
                const unsigned e = elem + k;
                const unsigned e_rem = e & 127u;
                const unsigned e_l = e_rem & 31u;
                const unsigned qh = block[qh_base + e_l];
                unsigned q;
                if (e_rem < 32u) {
                    q = (block[ql_base + e_l] & 0x0fu) | (((qh >> 0) & 3u) << 4);
                } else if (e_rem < 64u) {
                    q = (block[ql_base + e_l + 32u] & 0x0fu) | (((qh >> 2) & 3u) << 4);
                } else if (e_rem < 96u) {
                    q = (block[ql_base + e_l] >> 4) | (((qh >> 4) & 3u) << 4);
                } else {
                    q = (block[ql_base + e_l + 32u] >> 4) | (((qh >> 6) & 3u) << 4);
                }
                q_pack |= (int)(q & 0xffu) << (8u * k);
            }

            const unsigned qs_off = b * 256u + elem;
            const unsigned ds_off = b * 8u + (elem >> 5);
            const float yd = d * (float)sc;
            const int x_pack0 = rnb_load_i32_aligned4(input_qs + base0_qs + qs_off);
            const int dot0 = __dp4a(q_pack, x_pack0, 0);
            const int x_sum0 = __dp4a(0x01010101, x_pack0, 0);
            acc0 += input_ds[base0_ds + ds_off] * yd * (float)(dot0 - 32 * x_sum0);
            if (group_len > 1u) {
                const int x_pack1 = rnb_load_i32_aligned4(input_qs + base1_qs + qs_off);
                const int dot1 = __dp4a(q_pack, x_pack1, 0);
                const int x_sum1 = __dp4a(0x01010101, x_pack1, 0);
                acc1 += input_ds[base1_ds + ds_off] * yd * (float)(dot1 - 32 * x_sum1);
            }
            if (group_len > 2u) {
                const int x_pack2 = rnb_load_i32_aligned4(input_qs + base2_qs + qs_off);
                const int dot2 = __dp4a(q_pack, x_pack2, 0);
                const int x_sum2 = __dp4a(0x01010101, x_pack2, 0);
                acc2 += input_ds[base2_ds + ds_off] * yd * (float)(dot2 - 32 * x_sum2);
            }
            if (group_len > 3u) {
                const int x_pack3 = rnb_load_i32_aligned4(input_qs + base3_qs + qs_off);
                const int dot3 = __dp4a(q_pack, x_pack3, 0);
                const int x_sum3 = __dp4a(0x01010101, x_pack3, 0);
                acc3 += input_ds[base3_ds + ds_off] * yd * (float)(dot3 - 32 * x_sum3);
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group4_lowreg_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned group = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const unsigned slot_stride = blocks_per_row * 256u;
    const unsigned base0 = slot_start * slot_stride;
    const unsigned base1 = base0 + slot_stride;
    const unsigned base2 = base1 + slot_stride;
    const unsigned base3 = base2 + slot_stride;
    const unsigned char* row_ptr = weights[slot_start] + row * blocks_per_row * 210u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const unsigned x_base = b * 256u + lane;

        #pragma unroll
        for (unsigned i = 0; i < 8u; ++i) {
            const unsigned x = x_base + i * 32u;
            const float y = rnb_q6k_value_at_with_d(block, lane + i * 32u, d);
            acc0 += y * input[base0 + x];
            if (group_len > 1u) {
                acc1 += y * input[base1 + x];
            }
            if (group_len > 2u) {
                acc2 += y * input[base2 + x];
            }
            if (group_len > 3u) {
                acc3 += y * input[base3 + x];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
        acc2 += __shfl_down_sync(0xffffffffu, acc2, offset);
        acc3 += __shfl_down_sync(0xffffffffu, acc3, offset);
    }

    if (lane == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, acc0 * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, acc1 * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, acc2 * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, acc3 * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q5k_selected_down_accum_by_token_group4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned group = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    __shared__ float partial0[256];
    __shared__ float partial1[256];
    __shared__ float partial2[256];
    __shared__ float partial3[256];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    const unsigned row_bytes = blocks_per_row * 176u;
    const unsigned char* row_ptr = weights[slot_start] + row * row_bytes;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const float y = rnb_q5k_value_at(row_ptr + b * 176u, tid);
        const unsigned x_off = b * 256u + tid;
        const float* input0 = input + slot_start * blocks_per_row * 256u;
        acc0 += y * input0[x_off];
        if (group_len > 1u) {
            const float* input1 = input + (slot_start + 1u) * blocks_per_row * 256u;
            acc1 += y * input1[x_off];
        }
        if (group_len > 2u) {
            const float* input2 = input + (slot_start + 2u) * blocks_per_row * 256u;
            acc2 += y * input2[x_off];
        }
        if (group_len > 3u) {
            const float* input3 = input + (slot_start + 3u) * blocks_per_row * 256u;
            acc3 += y * input3[x_off];
        }
    }

    partial0[tid] = acc0;
    partial1[tid] = acc1;
    partial2[tid] = acc2;
    partial3[tid] = acc3;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial0[tid] += partial0[tid + stride];
            partial1[tid] += partial1[tid + stride];
            partial2[tid] += partial2[tid + stride];
            partial3[tid] += partial3[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, partial0[0] * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, partial1[0] * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, partial2[0] * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, partial3[0] * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_group4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ group_meta,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x;
    const unsigned group = blockIdx.y;
    const unsigned tid = threadIdx.x;
    if (row >= rows || tid >= 256u) {
        return;
    }

    const unsigned slot_start = group_meta[group * 2u + 0u];
    const unsigned group_len = group_meta[group * 2u + 1u];
    if (group_len == 0u || group_len > 4u) {
        return;
    }

    __shared__ float partial0[256];
    __shared__ float partial1[256];
    __shared__ float partial2[256];
    __shared__ float partial3[256];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights[slot_start] + row * row_bytes;
    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const float y = rnb_q6k_value_at(row_ptr + b * 210u, tid);
        const unsigned x_off = b * 256u + tid;
        const float* input0 = input + slot_start * blocks_per_row * 256u;
        acc0 += y * input0[x_off];
        if (group_len > 1u) {
            const float* input1 = input + (slot_start + 1u) * blocks_per_row * 256u;
            acc1 += y * input1[x_off];
        }
        if (group_len > 2u) {
            const float* input2 = input + (slot_start + 2u) * blocks_per_row * 256u;
            acc2 += y * input2[x_off];
        }
        if (group_len > 3u) {
            const float* input3 = input + (slot_start + 3u) * blocks_per_row * 256u;
            acc3 += y * input3[x_off];
        }
    }

    partial0[tid] = acc0;
    partial1[tid] = acc1;
    partial2[tid] = acc2;
    partial3[tid] = acc3;
    __syncthreads();
    for (unsigned stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial0[tid] += partial0[tid + stride];
            partial1[tid] += partial1[tid + stride];
            partial2[tid] += partial2[tid + stride];
            partial3[tid] += partial3[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        atomicAdd(out + token_ids[slot_start] * rows + row, partial0[0] * route[slot_start]);
        if (group_len > 1u) {
            atomicAdd(out + token_ids[slot_start + 1u] * rows + row, partial1[0] * route[slot_start + 1u]);
        }
        if (group_len > 2u) {
            atomicAdd(out + token_ids[slot_start + 2u] * rows + row, partial2[0] * route[slot_start + 2u]);
        }
        if (group_len > 3u) {
            atomicAdd(out + token_ids[slot_start + 3u] * rows + row, partial3[0] * route[slot_start + 3u]);
        }
    }
}

extern "C" __global__ void rnb_q4k_selected_down_accum_by_token_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * 4u + threadIdx.y;
    const unsigned slot = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 144u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const float* slot_input = input + slot * blocks_per_row * 256u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 144u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            unsigned sc;
            unsigned mn;
            if (j < 4u) {
                sc = block[4 + j] & 63u;
                mn = block[4 + j + 4] & 63u;
            } else {
                sc = (block[4 + j + 4] & 0x0fu) | ((block[4 + j - 4] >> 6) << 4);
                mn = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned q = block[16 + q_index];
            q = local < 32u ? (q & 0x0fu) : (q >> 4);
            const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
            acc += y * slot_input[b * 256u + tid];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        atomicAdd(out + token_ids[slot] * rows + row, acc * route[slot]);
    }
}

extern "C" __global__ void rnb_q5k_selected_down_accum_by_token_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * 4u + threadIdx.y;
    const unsigned slot = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 176u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const float* slot_input = input + slot * blocks_per_row * 256u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 176u;
        const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
        const unsigned raw_dmin = (unsigned)block[2] | ((unsigned)block[3] << 8);
        const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
        const float dmin = __half2float(__ushort_as_half((unsigned short)raw_dmin));
        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            unsigned sc;
            unsigned mn;
            if (j < 4u) {
                sc = block[4 + j] & 63u;
                mn = block[4 + j + 4] & 63u;
            } else {
                sc = (block[4 + j + 4] & 0x0fu) | ((block[4 + j - 4] >> 6) << 4);
                mn = (block[4 + j + 4] >> 4) | ((block[4 + j] >> 6) << 4);
            }
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned q = block[48 + q_index];
            q = local < 32u ? (q & 0x0fu) : (q >> 4);
            const unsigned high = (block[16 + (tid & 31u)] >> (tid >> 5)) & 1u;
            q |= high << 4;
            const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
            acc += y * slot_input[b * 256u + tid];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        atomicAdd(out + token_ids[slot] * rows + row, acc * route[slot]);
    }
}

extern "C" __global__ void rnb_q5k_selected_down_accum_by_token_pair2_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    const unsigned* __restrict__ expert_ids,
    const unsigned* __restrict__ pair_slots,
    unsigned rows,
    unsigned slots_per_token,
    unsigned blocks_per_row) {
    constexpr unsigned INVALID_SLOT = 0xffffffffu;
    const unsigned candidate_slot = blockIdx.y;
    const unsigned total_slots = slots_per_token * 2u;
    const unsigned row = blockIdx.x * 4u + threadIdx.y;
    const unsigned lane = threadIdx.x;
    __shared__ unsigned shared_primary_slot;
    __shared__ unsigned shared_partner_slot;

    unsigned primary_slot;
    unsigned partner_slot;
    if (pair_slots != nullptr) {
        primary_slot = candidate_slot < total_slots ? candidate_slot : INVALID_SLOT;
        partner_slot =
            primary_slot != INVALID_SLOT ? pair_slots[candidate_slot] : INVALID_SLOT;
        if (partner_slot == 0xfffffffeu) {
            primary_slot = INVALID_SLOT;
        }
    } else {
        if (threadIdx.x == 0u && threadIdx.y == 0u) {
            shared_primary_slot =
                candidate_slot < total_slots ? candidate_slot : INVALID_SLOT;
            shared_partner_slot = INVALID_SLOT;
            if (shared_primary_slot != INVALID_SLOT) {
                if (candidate_slot < slots_per_token) {
                    const unsigned expert = expert_ids[candidate_slot];
                    for (unsigned second = slots_per_token; second < total_slots; ++second) {
                        if (expert_ids[second] == expert) {
                            shared_partner_slot = second;
                            break;
                        }
                    }
                } else {
                    const unsigned expert = expert_ids[candidate_slot];
                    for (unsigned first = 0; first < slots_per_token; ++first) {
                        if (expert_ids[first] == expert) {
                            shared_primary_slot = INVALID_SLOT;
                            break;
                        }
                    }
                }
            }
        }
        __syncthreads();
        primary_slot = shared_primary_slot;
        partner_slot = shared_partner_slot;
    }
    if (primary_slot == INVALID_SLOT || row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    const bool paired = partner_slot != INVALID_SLOT;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned row_bytes = blocks_per_row * 176u;
    const unsigned char* row_ptr = weights[primary_slot] + row * row_bytes;
    const float* input0 = input + primary_slot * blocks_per_row * 256u;
    const float* input1 =
        paired ? input + partner_slot * blocks_per_row * 256u : input0;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 176u;
        float d_lane = 0.0f;
        float dmin_lane = 0.0f;
        if (lane == 0u) {
            const unsigned raw_d = (unsigned)block[0] | ((unsigned)block[1] << 8);
            const unsigned raw_dmin =
                (unsigned)block[2] | ((unsigned)block[3] << 8);
            d_lane = __half2float(__ushort_as_half((unsigned short)raw_d));
            dmin_lane = __half2float(__ushort_as_half((unsigned short)raw_dmin));
        }
        const float d = __shfl_sync(0xffffffffu, d_lane, 0);
        const float dmin = __shfl_sync(0xffffffffu, dmin_lane, 0);

        unsigned sc_lane = 0u;
        unsigned mn_lane = 0u;
        if (lane < 8u) {
            const unsigned j = lane;
            if (j < 4u) {
                sc_lane = block[4u + j] & 63u;
                mn_lane = block[4u + j + 4u] & 63u;
            } else {
                sc_lane =
                    (block[4u + j + 4u] & 0x0fu) | ((block[4u + j - 4u] >> 6) << 4);
                mn_lane =
                    (block[4u + j + 4u] >> 4) | ((block[4u + j] >> 6) << 4);
            }
        }
        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned j = tid >> 5;
            const unsigned sc = __shfl_sync(0xffffffffu, sc_lane, j);
            const unsigned mn = __shfl_sync(0xffffffffu, mn_lane, j);
            const unsigned local = tid & 63u;
            const unsigned q_index = (tid >> 6) * 32u + (tid & 31u);
            unsigned q = block[48 + q_index];
            q = local < 32u ? (q & 0x0fu) : (q >> 4);
            const unsigned high = (block[16 + (tid & 31u)] >> (tid >> 5)) & 1u;
            q |= high << 4;
            const float y = (d * (float)sc) * (float)q - dmin * (float)mn;
            const unsigned x_off = b * 256u + tid;
            acc0 += y * input0[x_off];
            if (paired) {
                acc1 += y * input1[x_off];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
    }
    if (lane == 0u) {
        atomicAdd(
            out + token_ids[primary_slot] * rows + row,
            acc0 * route[primary_slot]);
        if (paired) {
            atomicAdd(
                out + token_ids[partner_slot] * rows + row,
                acc1 * route[partner_slot]);
        }
    }
}


extern "C" __global__ void rnb_q6k_selected_down_accum_by_token_warp4(
    float* __restrict__ out,
    const unsigned char* const* __restrict__ weights,
    const float* __restrict__ input,
    const float* __restrict__ route,
    const unsigned* __restrict__ token_ids,
    unsigned rows,
    unsigned blocks_per_row) {
    const unsigned row = blockIdx.x * 4u + threadIdx.y;
    const unsigned slot = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || lane >= 32u || threadIdx.y >= 4u) {
        return;
    }

    float acc = 0.0f;
    const unsigned row_bytes = blocks_per_row * 210u;
    const unsigned char* row_ptr = weights[slot] + row * row_bytes;
    const float* slot_input = input + slot * blocks_per_row * 256u;

    for (unsigned b = 0; b < blocks_per_row; ++b) {
        const unsigned char* block = row_ptr + b * 210u;
        for (unsigned tid = lane; tid < 256u; tid += 32u) {
            const unsigned n = tid >> 7;
            const unsigned rem = tid & 127u;
            const unsigned l = rem & 31u;
            const unsigned is = l >> 4;
            const unsigned ql_base = n * 64u;
            const unsigned qh_base = 128u + n * 32u;
            const unsigned sc_base = 192u + n * 8u;
            const unsigned qh = block[qh_base + l];
            unsigned q;
            int sc;
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
            const unsigned raw_d = (unsigned)block[208] | ((unsigned)block[209] << 8);
            const float d = __half2float(__ushort_as_half((unsigned short)raw_d));
            const float y = d * (float)sc * (float)((int)q - 32);
            acc += y * slot_input[b * 256u + tid];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0u) {
        atomicAdd(out + token_ids[slot] * rows + row, acc * route[slot]);
    }
}
