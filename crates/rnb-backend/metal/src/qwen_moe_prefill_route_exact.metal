#include <metal_stdlib>
using namespace metal;

struct qwen_moe_float2 {
    float hi;
    float lo;
};


inline qwen_moe_float2 qwen_moe_float2_add(
    qwen_moe_float2 lhs,
    qwen_moe_float2 rhs)
{
    float sum = lhs.hi + rhs.hi;
    float rhs_virtual = sum - lhs.hi;
    float error = (lhs.hi - (sum - rhs_virtual)) + (rhs.hi - rhs_virtual);
    float tail = lhs.lo + rhs.lo;
    float combined = error + tail;
    float hi = sum + combined;
    return { hi, combined - (hi - sum) };
}

inline qwen_moe_float2 qwen_moe_float2_square(float value)
{
    float hi = value * value;
    return { hi, fma(value, value, -hi) };
}



inline qwen_moe_float2 qwen_moe_float2_multiply(
    qwen_moe_float2 lhs,
    qwen_moe_float2 rhs)
{
    float product = lhs.hi * rhs.hi;
    float error = fma(lhs.hi, rhs.hi, -product);
    error = fma(lhs.hi, rhs.lo, error);
    error = fma(lhs.lo, rhs.hi, error);
    error = fma(lhs.lo, rhs.lo, error);
    float hi = product + error;
    return { hi, error - (hi - product) };
}

constant uint qwen_moe_exp2_hi[128] = {
    0x3f800000u, 0x3f80b1edu, 0x3f8164d2u, 0x3f8218afu, 0x3f82cd87u, 0x3f838359u, 0x3f843a29u, 0x3f84f1f6u, 0x3f85aac3u, 0x3f866491u, 0x3f871f62u, 0x3f87db35u, 0x3f88980fu, 0x3f8955eeu, 0x3f8a14d5u, 0x3f8ad4c6u,
    0x3f8b95c2u, 0x3f8c57cau, 0x3f8d1adfu, 0x3f8ddf04u, 0x3f8ea43au, 0x3f8f6a81u, 0x3f9031dcu, 0x3f90fa4du, 0x3f91c3d3u, 0x3f928e72u, 0x3f935a2bu, 0x3f9426ffu, 0x3f94f4f0u, 0x3f95c3ffu, 0x3f96942du, 0x3f97657du,
    0x3f9837f0u, 0x3f990b88u, 0x3f99e046u, 0x3f9ab62bu, 0x3f9b8d3au, 0x3f9c6573u, 0x3f9d3edau, 0x3f9e196eu, 0x3f9ef532u, 0x3f9fd228u, 0x3fa0b051u, 0x3fa18fafu, 0x3fa27043u, 0x3fa3520fu, 0x3fa43516u, 0x3fa51958u,
    0x3fa5fed7u, 0x3fa6e595u, 0x3fa7cd94u, 0x3fa8b6d5u, 0x3fa9a15bu, 0x3faa8d26u, 0x3fab7a3au, 0x3fac6897u, 0x3fad583fu, 0x3fae4934u, 0x3faf3b79u, 0x3fb02f0eu, 0x3fb123f6u, 0x3fb21a32u, 0x3fb311c4u, 0x3fb40aafu,
    0x3fb504f3u, 0x3fb60094u, 0x3fb6fd92u, 0x3fb7fbf0u, 0x3fb8fbafu, 0x3fb9fcd2u, 0x3fbaff5bu, 0x3fbc034au, 0x3fbd08a4u, 0x3fbe0f68u, 0x3fbf179au, 0x3fc0213bu, 0x3fc12c4du, 0x3fc238d2u, 0x3fc346cdu, 0x3fc4563fu,
    0x3fc5672au, 0x3fc67991u, 0x3fc78d75u, 0x3fc8a2d8u, 0x3fc9b9beu, 0x3fcad226u, 0x3fcbec15u, 0x3fcd078cu, 0x3fce248cu, 0x3fcf4319u, 0x3fd06334u, 0x3fd184dfu, 0x3fd2a81eu, 0x3fd3ccf1u, 0x3fd4f35bu, 0x3fd61b5eu,
    0x3fd744fdu, 0x3fd87039u, 0x3fd99d16u, 0x3fdacb94u, 0x3fdbfbb8u, 0x3fdd2d82u, 0x3fde60f5u, 0x3fdf9613u, 0x3fe0ccdfu, 0x3fe2055bu, 0x3fe33f89u, 0x3fe47b6du, 0x3fe5b907u, 0x3fe6f85bu, 0x3fe8396au, 0x3fe97c38u,
    0x3feac0c7u, 0x3fec0719u, 0x3fed4f30u, 0x3fee9910u, 0x3fefe4bau, 0x3ff13231u, 0x3ff28177u, 0x3ff3d290u, 0x3ff5257du, 0x3ff67a41u, 0x3ff7d0dfu, 0x3ff9295au, 0x3ffa83b3u, 0x3ffbdfedu, 0x3ffd3e0cu, 0x3ffe9e11u,
};

constant uint qwen_moe_exp2_lo[128] = {
    0x00000000u, 0x331fb333u, 0xb1c43fd0u, 0x3306e7f8u, 0xb34ea7a9u, 0x331ddf6eu, 0xb2f14c87u, 0x332c6f38u, 0x334f9891u, 0x3337247fu, 0xb352c2e6u, 0x337fed32u, 0xb37eda4bu, 0x30d86398u, 0x336a92deu, 0x330a58e5u,
    0xb260aba1u, 0xb2ee6e43u, 0x3336fcb7u, 0x32808b9au, 0xb3697465u, 0x323f3647u, 0x330628cdu, 0xb3682237u, 0x33675624u, 0x337b2a64u, 0x32bc4f9cu, 0x31fab1c0u, 0xb32e0212u, 0xb3725267u, 0x32dc8061u, 0x3313e2f5u,
    0x33231b71u, 0xb26cc9f4u, 0xb359be90u, 0xb0dac01eu, 0xb30c5563u, 0x33505d86u, 0xb331a601u, 0x3244ea39u, 0x33412342u, 0x32959004u, 0x31fb9715u, 0xb2d5eaedu, 0x30c3125au, 0x3351d005u, 0xb323ec33u, 0xb37282c2u,
    0xb32c9d5eu, 0xb2c0445eu, 0xb3162d36u, 0x3233d990u, 0xb3162b08u, 0x3325d921u, 0xb314ad82u, 0xb3368380u, 0xb22deaf6u, 0x3325946bu, 0xb3252decu, 0xb2d1247fu, 0xb37c5aa8u, 0xb33333ceu, 0x32154889u, 0xb33b3569u,
    0x32cfe77au, 0xb32f4254u, 0xb266b974u, 0xb2d5cd70u, 0x330ec5f7u, 0x330a5817u, 0xb31bd983u, 0x337de5d4u, 0xb3414fe8u, 0x31986099u, 0xb3130b1au, 0xb33c1e5fu, 0xb2d6663eu, 0x32c478f6u, 0xb2976da2u, 0xb2ceb32du,
    0x320aa837u, 0xb314abb7u, 0xb2dd5119u, 0x33391ffcu, 0xb37323a2u, 0x333c8521u, 0xb006c6c0u, 0xb3735f84u, 0x3228fc24u, 0xb2c39b9cu, 0xb2944353u, 0x3344a2d3u, 0xb35c1daau, 0xb34cf4cau, 0xb3286024u, 0xb0303218u,
    0xb2d4a58au, 0x3318db66u, 0xb2f61d41u, 0x335e5594u, 0xb3504a1cu, 0xb375ef9bu, 0xb37b43e3u, 0xb2851c3fu, 0xb21eab59u, 0xac3e1800u, 0x33657d15u, 0xb33f9185u, 0xb2441be6u, 0xb32a23c0u, 0x33207898u, 0x3300d89fu,
    0xb24116deu, 0xb31367c6u, 0x3276cca1u, 0xb34fe4bau, 0xb348464au, 0xb330a5edu, 0x32f167ffu, 0xb2871670u, 0x32292436u, 0x3358e67fu, 0x336615a2u, 0xb3094457u, 0xb2923758u, 0x3359cbe1u, 0x31cf486cu, 0x3338f71fu,
};

inline float qwen_moe_exp_f32(float value)
{
    uint word = as_type<uint>(value);
    uint sign = word >> 31;
    uint magnitude = word & 0x7fffffffu;
    const float huge_value = as_type<float>(0x7149f2cau);
    const float two_to_minus_100 = as_type<float>(0x0d800000u);

    if (magnitude >= 0x42b17218u) {
        if (magnitude > 0x7f800000u) return value + value;
        if (magnitude == 0x7f800000u) return sign == 0u ? value : 0.0f;
        if (value > as_type<float>(0x42b17180u)) return huge_value * huge_value;
        if (value < as_type<float>(0xc2cff1b5u)) {
            return two_to_minus_100 * two_to_minus_100;
        }
    }

    const qwen_moe_float2 inv_ln2_times_128 = {
        as_type<float>(0x4338aa3bu),
        as_type<float>(0x36257060u)
    };
    qwen_moe_float2 scaled = qwen_moe_float2_multiply(
        inv_ln2_times_128,
        { value, 0.0f });
    int lower = (int)floor(scaled.hi);
    qwen_moe_float2 fraction = qwen_moe_float2_add(
        { scaled.hi - (float)lower, scaled.lo },
        { 0.0f, 0.0f });
    bool round_up = fraction.hi > 0.5f
        || (fraction.hi == 0.5f
            && (fraction.lo > 0.0f || (fraction.lo == 0.0f && (lower & 1) != 0)));
    int exponent_128 = lower + (round_up ? 1 : 0);

    qwen_moe_float2 reduced = qwen_moe_float2_add(
        scaled,
        { -(float)exponent_128, 0.0f });
    const qwen_moe_float2 quadratic = {
        as_type<float>(0x3775fdf0u),
        as_type<float>(0xa8cf29aau)
    };
    const qwen_moe_float2 linear = {
        as_type<float>(0x3bb17223u),
        as_type<float>(0xaf41ef25u)
    };
    qwen_moe_float2 polynomial = qwen_moe_float2_add(
        qwen_moe_float2_multiply(quadratic, reduced),
        linear);
    polynomial = qwen_moe_float2_multiply(polynomial, reduced);

    int table_index = exponent_128 % 128;
    int scale_exponent = exponent_128 / 128;
    if (table_index < 0) {
        table_index += 128;
        scale_exponent -= 1;
    }
    qwen_moe_float2 table_value = {
        ldexp(as_type<float>(qwen_moe_exp2_hi[table_index]), scale_exponent),
        ldexp(as_type<float>(qwen_moe_exp2_lo[table_index]), scale_exponent)
    };
    qwen_moe_float2 result = qwen_moe_float2_add(
        qwen_moe_float2_multiply(polynomial, table_value),
        table_value);
    return result.hi + result.lo;
}


kernel void qwen_moe_prefill_router_f32_exact(
    device const uchar* weight_bytes [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& n_expert [[buffer(3)]],
    constant uint& hidden_dim [[buffer(4)]],
    constant uint& weight_byte_offset [[buffer(5)]],
    constant uint& n_tokens [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint token = gid.x;
    uint expert = gid.y;
    if (token >= n_tokens || expert >= n_expert) return;

    device const float* weights =
        (device const float*)(weight_bytes + weight_byte_offset)
        + expert * hidden_dim;
    device const float* values = input + token * hidden_dim;
    float4 acc0 = 0.0f;
    float4 acc1 = 0.0f;
    uint col = 0;
    for (; col + 8 <= hidden_dim; col += 8) {
        float4 weight0 = {
            weights[col], weights[col + 1], weights[col + 2], weights[col + 3]
        };
        float4 value0 = {
            values[col], values[col + 1], values[col + 2], values[col + 3]
        };
        float4 weight1 = {
            weights[col + 4], weights[col + 5], weights[col + 6], weights[col + 7]
        };
        float4 value1 = {
            values[col + 4], values[col + 5], values[col + 6], values[col + 7]
        };
        acc0 = fma(weight0, value0, acc0);
        acc1 = fma(weight1, value1, acc1);
    }
    if (col + 4 <= hidden_dim) {
        float4 weight = {
            weights[col], weights[col + 1], weights[col + 2], weights[col + 3]
        };
        float4 value = {
            values[col], values[col + 1], values[col + 2], values[col + 3]
        };
        acc0 = fma(weight, value, acc0);
        col += 4;
    }
    float4 combined = acc0 + acc1;
    float dot = (combined.x + combined.y) + (combined.z + combined.w);
    for (; col < hidden_dim; col++) {
        dot += weights[col] * values[col];
    }
    output[token * n_expert + expert] = dot;
}

kernel void qwen_moe_prefill_rms_norm_exact(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      cols   [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint token [[thread_position_in_grid]])
{
    uint base = token * cols;
    qwen_moe_float2 sum = { 0.0f, 0.0f };
    for (uint col = 0; col < cols; col++) {
        sum = qwen_moe_float2_add(sum, qwen_moe_float2_square(input[base + col]));
    }

    float reciprocal_cols = 1.0f / (float)cols;
    qwen_moe_float2 mean = {
        sum.hi * reciprocal_cols,
        sum.lo * reciprocal_cols
    };
    mean = qwen_moe_float2_add(mean, { eps, 0.0f });

    float rms = precise::sqrt(mean.hi);
    float residual = fma(-rms, rms, mean.hi) + mean.lo;
    rms += residual / (2.0f * rms);
    for (uint col = 0; col < cols; col++) {
        float normalized = input[base + col] / rms;
        output[base + col] = normalized * weight[col];
    }
}

kernel void qwen_gdn_prefill_rms_norm_f32_exact(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      cols   [[buffer(3)]],
    constant float&     eps    [[buffer(4)]],
    uint token [[thread_position_in_grid]])
{
    uint base = token * cols;
    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        float value = input[base + col];
        volatile float square = value * value;
        volatile float next_sum = sum + square;
        sum = next_sum;
    }
    float rms = precise::sqrt(sum / (float)cols + eps);
    for (uint col = 0; col < cols; col++) {
        output[base + col] = (input[base + col] / rms) * weight[col];
    }
}

kernel void qwen_prefill_l2_norm_exact(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant uint& cols       [[buffer(2)]],
    constant float& eps       [[buffer(3)]],
    constant float& scale     [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * cols;
    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        float value = input[base + col];
        float square = value * value;
        sum += square;
    }
    float norm = precise::sqrt(sum + eps);
    for (uint col = 0; col < cols; col++) {
        float normalized = input[base + col] / norm;
        output[base + col] = normalized * scale;
    }
}

kernel void qwen_prefill_gated_rmsnorm_silu_chain_exact(
    device const float* input  [[buffer(0)]],
    device const float* z      [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output       [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint row [[thread_position_in_grid]])
{
    uint base = row * cols;
    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        float value = input[base + col];
        float square = value * value;
        sum += square;
    }
    float mean = sum / (float)cols;
    float rms = precise::sqrt(mean + eps);

    for (uint col = 0; col < cols; col++) {
        float normalized = input[base + col] / rms;
        float weighted = normalized * weight[col];
        float gate = z[base + col];
        float silu = gate / (1.0f + qwen_moe_exp_f32(-gate));
        output[base + col] = weighted * silu;
    }
}

kernel void qwen_moe_prefill_shared_gate_chain_exact(
    device const float* input  [[buffer(0)]],
    device const float* scale  [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      cols   [[buffer(3)]],
    constant uint&      tokens [[buffer(4)]],
    uint token [[thread_position_in_grid]])
{
    if (token >= tokens) return;
    uint base = token * cols;
    float dot = 0.0f;
    for (uint col = 0; col < cols; col++) {
        float product = input[base + col] * scale[col];
        dot += product;
    }
    output[token] = 1.0f / (1.0f + qwen_moe_exp_f32(-dot));
}

kernel void qwen_moe_prefill_topk_from_logits_chain_exact(
    device const float* logits        [[buffer(0)]],
    device uint*        expert_ids    [[buffer(1)]],
    device float*       route_weights [[buffer(2)]],
    device uint*        route_aux     [[buffer(3)]],
    constant uint&      n_expert      [[buffer(4)]],
    constant uint&      n_used        [[buffer(5)]],
    constant uint&      seq_and_mode  [[buffer(6)]],
    uint token [[thread_position_in_grid]])
{
    const bool shared_sigmoid_mode = (seq_and_mode & 0x80000000u) != 0u;
    const uint seq_len = seq_and_mode & 0x7fffffffu;
    if (token >= seq_len || n_used == 0 || n_used > 32) return;

    float best_vals[32];
    float best_weights[32];
    uint best_ids[32];
    for (uint i = 0; i < n_used; i++) {
        best_vals[i] = -INFINITY;
        best_weights[i] = 0.0f;
        best_ids[i] = 0xffffffffu;
    }

    device const float* row = logits + token * n_expert;
    for (uint expert = 0; expert < n_expert; expert++) {
        float value = row[expert];
        for (uint rank = 0; rank < n_used; rank++) {
            if (value > best_vals[rank] ||
                (value == best_vals[rank] && expert < best_ids[rank])) {
                for (uint shift = n_used - 1; shift > rank; shift--) {
                    best_vals[shift] = best_vals[shift - 1];
                    best_ids[shift] = best_ids[shift - 1];
                }
                best_vals[rank] = value;
                best_ids[rank] = expert;
                break;
            }
        }
    }

    float selected_max = best_vals[0];
    float selected_sum = 0.0f;
    for (uint rank = 0; rank < n_used; rank++) {
        best_weights[rank] = qwen_moe_exp_f32(best_vals[rank] - selected_max);
        selected_sum += best_weights[rank];
    }

    uint base = token * n_used;
    for (uint rank = 0; rank < n_used; rank++) {
        expert_ids[base + rank] = best_ids[rank];
        route_weights[base + rank] = selected_sum != 0.0f
            ? best_weights[rank] / selected_sum
            : best_weights[rank];
        if (!shared_sigmoid_mode) {
            route_aux[base + rank] = token;
        }
    }
}
