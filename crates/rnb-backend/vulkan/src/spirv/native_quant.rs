use super::builder::{
    builtin, decoration, memory_semantics, op, scope, storage_class, Id, SpirvModule,
};
use crate::weight_cache::QuantType;
use rnb_core::quant_codebooks::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
};

struct ShaderTypes {
    bool_: Id,
    u32_: Id,
    i32_: Id,
    f32_: Id,
    ptr_sb_u32: Id,
    ptr_private_u32: Id,
}

struct Constants {
    zero: Id,
    one: Id,
    two: Id,
    three: Id,
    four: Id,
    eight: Id,
    ff: Id,
}

struct Codebooks {
    grid: Option<Id>,
    signs: Option<Id>,
    grid_words: u32,
}

fn private_u32_table(m: &mut SpirvModule, u32_type: Id, values: &[u32]) -> Id {
    let length = m.constant_u32(u32_type, values.len() as u32);
    let array_type = m.type_array(u32_type, length);
    let pointer_type = m.type_pointer(storage_class::PRIVATE, array_type);
    let constituents = values
        .iter()
        .map(|&value| m.constant_u32(u32_type, value))
        .collect::<Vec<_>>();
    let initializer = m.constant_composite(array_type, &constituents);
    m.variable_with_initializer(pointer_type, storage_class::PRIVATE, initializer)
}

fn pack_u64_table<const N: usize>(values: &[u64; N]) -> Vec<u32> {
    let mut words = Vec::with_capacity(N * 2);
    for &value in values {
        words.push(value as u32);
        words.push((value >> 32) as u32);
    }
    words
}

fn pack_u8_table<const N: usize>(values: &[u8; N]) -> Vec<u32> {
    let mut words = Vec::with_capacity(N.div_ceil(4));
    for chunk in values.chunks(4) {
        let mut word = 0u32;
        for (index, &value) in chunk.iter().enumerate() {
            word |= (value as u32) << (8 * index);
        }
        words.push(word);
    }
    words
}

fn emit_codebooks(m: &mut SpirvModule, u32_type: Id, quant: QuantType) -> Codebooks {
    let grid_values = match quant {
        QuantType::IQ2_XXS => Some((pack_u64_table(&IQ2XXS_GRID), 2)),
        QuantType::IQ2_XS => Some((pack_u64_table(&IQ2XS_GRID), 2)),
        QuantType::IQ2_S => Some((pack_u64_table(&IQ2S_GRID), 2)),
        QuantType::IQ3_XXS => Some((IQ3XXS_GRID.to_vec(), 1)),
        QuantType::IQ3_S => Some((IQ3S_GRID.to_vec(), 1)),
        QuantType::IQ1_S | QuantType::IQ1_M => Some((pack_u64_table(&IQ1S_GRID), 2)),
        _ => None,
    };
    let signs = matches!(
        quant,
        QuantType::IQ2_XXS | QuantType::IQ2_XS | QuantType::IQ3_XXS
    )
    .then(|| private_u32_table(m, u32_type, &pack_u8_table(&KSIGNS_IQ2XS)));
    let grid_words = grid_values
        .as_ref()
        .map(|(_, words_per_entry)| *words_per_entry)
        .unwrap_or(0);
    let grid = grid_values
        .as_ref()
        .map(|(values, _)| private_u32_table(m, u32_type, values));
    Codebooks {
        grid,
        signs,
        grid_words,
    }
}

fn load_byte(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    buffer: Id,
    byte_offset: Id,
) -> Id {
    let word_index = m.udiv(types.u32_, byte_offset, constants.four);
    let byte_in_word = m.umod(types.u32_, byte_offset, constants.four);
    let shift = m.imul(types.u32_, byte_in_word, constants.eight);
    let ptr = m.access_chain(types.ptr_sb_u32, buffer, &[constants.zero, word_index]);
    let word = m.load(types.u32_, ptr);
    let shifted = m.shift_right_logical(types.u32_, word, shift);
    m.bitwise_and(types.u32_, shifted, constants.ff)
}

fn load_private_byte(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    table: Id,
    byte_index: Id,
) -> Id {
    let word_index = m.udiv(types.u32_, byte_index, constants.four);
    let byte_in_word = m.umod(types.u32_, byte_index, constants.four);
    let shift = m.imul(types.u32_, byte_in_word, constants.eight);
    let ptr = m.access_chain(types.ptr_private_u32, table, &[word_index]);
    let word = m.load(types.u32_, ptr);
    let shifted = m.shift_right_logical(types.u32_, word, shift);
    m.bitwise_and(types.u32_, shifted, constants.ff)
}

fn load_grid_byte(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    codebooks: &Codebooks,
    grid_index: Id,
    lane: Id,
) -> Id {
    let grid = codebooks
        .grid
        .expect("importance quant shader requires a grid codebook");
    let entry_bytes = m.constant_u32(types.u32_, codebooks.grid_words * 4);
    let entry_offset = m.imul(types.u32_, grid_index, entry_bytes);
    let byte_index = m.iadd(types.u32_, entry_offset, lane);
    load_private_byte(m, types, constants, grid, byte_index)
}

fn load_sign_byte(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    codebooks: &Codebooks,
    index: Id,
) -> Id {
    let signs = codebooks
        .signs
        .expect("importance quant shader requires a sign codebook");
    load_private_byte(m, types, constants, signs, index)
}

fn load_u16_le(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    buffer: Id,
    byte_offset: Id,
) -> Id {
    let lo = load_byte(m, types, constants, buffer, byte_offset);
    let hi_offset = m.iadd(types.u32_, byte_offset, constants.one);
    let hi = load_byte(m, types, constants, buffer, hi_offset);
    let hi_shifted = m.shift_left_logical(types.u32_, hi, constants.eight);
    m.bitwise_or(types.u32_, lo, hi_shifted)
}

fn load_u32_le(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    buffer: Id,
    byte_offset: Id,
) -> Id {
    let b0 = load_byte(m, types, constants, buffer, byte_offset);
    let o1 = m.iadd(types.u32_, byte_offset, constants.one);
    let b1 = load_byte(m, types, constants, buffer, o1);
    let o2 = m.iadd(types.u32_, byte_offset, constants.two);
    let b2 = load_byte(m, types, constants, buffer, o2);
    let o3 = m.iadd(types.u32_, byte_offset, constants.three);
    let b3 = load_byte(m, types, constants, buffer, o3);
    let c16 = m.constant_u32(types.u32_, 16);
    let c24 = m.constant_u32(types.u32_, 24);
    let p1 = m.shift_left_logical(types.u32_, b1, constants.eight);
    let p2 = m.shift_left_logical(types.u32_, b2, c16);
    let p3 = m.shift_left_logical(types.u32_, b3, c24);
    let lo = m.bitwise_or(types.u32_, b0, p1);
    let hi = m.bitwise_or(types.u32_, p2, p3);
    m.bitwise_or(types.u32_, lo, hi)
}

fn f16_to_f32(m: &mut SpirvModule, types: &ShaderTypes, bits: Id) -> Id {
    let c_u32_0 = m.constant_u32(types.u32_, 0);
    let c_u32_1 = m.constant_u32(types.u32_, 1);
    let c_u32_10 = m.constant_u32(types.u32_, 10);
    let c_u32_13 = m.constant_u32(types.u32_, 13);
    let c_u32_15 = m.constant_u32(types.u32_, 15);
    let c_u32_23 = m.constant_u32(types.u32_, 23);
    let c_u32_31 = m.constant_u32(types.u32_, 31);
    let c_u32_1f = m.constant_u32(types.u32_, 0x1f);
    let c_u32_3ff = m.constant_u32(types.u32_, 0x3ff);
    let c_u32_112 = m.constant_u32(types.u32_, 112);
    let c_f32_2pow_neg24 = m.constant_f32(types.f32_, 5.9604644775390625e-8);

    let sign = m.shift_right_logical(types.u32_, bits, c_u32_15);
    let sign_bit = m.bitwise_and(types.u32_, sign, c_u32_1);
    let exp_raw = m.shift_right_logical(types.u32_, bits, c_u32_10);
    let exp = m.bitwise_and(types.u32_, exp_raw, c_u32_1f);
    let mant = m.bitwise_and(types.u32_, bits, c_u32_3ff);
    let sign_part = m.shift_left_logical(types.u32_, sign_bit, c_u32_31);
    let exp_adj = m.iadd(types.u32_, exp, c_u32_112);
    let exp_part = m.shift_left_logical(types.u32_, exp_adj, c_u32_23);
    let mant_part = m.shift_left_logical(types.u32_, mant, c_u32_13);
    let bits_mid = m.bitwise_or(types.u32_, sign_part, exp_part);
    let f32_bits = m.bitwise_or(types.u32_, bits_mid, mant_part);
    let normal = m.bitcast(types.f32_, f32_bits);
    let mant_f = m.convert_u_to_f(types.f32_, mant);
    let denorm_abs = m.fmul(types.f32_, mant_f, c_f32_2pow_neg24);
    let denorm_neg = m.fnegate(types.f32_, denorm_abs);
    let sign_set = m.i_not_equal(types.bool_, sign_bit, c_u32_0);
    let denormal = m.select(types.f32_, sign_set, denorm_neg, denorm_abs);
    let exp_nonzero = m.i_not_equal(types.bool_, exp, c_u32_0);
    m.select(types.f32_, exp_nonzero, normal, denormal)
}

fn load_f16(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    buffer: Id,
    byte_offset: Id,
) -> Id {
    let bits = load_u16_le(m, types, constants, buffer, byte_offset);
    f16_to_f32(m, types, bits)
}

fn add_offset(m: &mut SpirvModule, types: &ShaderTypes, base: Id, offset: u32) -> Id {
    if offset == 0 {
        base
    } else {
        let constant = m.constant_u32(types.u32_, offset);
        m.iadd(types.u32_, base, constant)
    }
}

fn signed_byte_to_f32(m: &mut SpirvModule, types: &ShaderTypes, byte: Id) -> Id {
    let c_i32_24 = m.constant_u32(types.i32_, 24);
    let as_i32 = m.bitcast(types.i32_, byte);
    let shifted = m.shift_left_logical(types.i32_, as_i32, c_i32_24);
    let signed = m.shift_right_arithmetic(types.i32_, shifted, c_i32_24);
    m.convert_s_to_f(types.f32_, signed)
}

fn apply_sign_bit(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    magnitude: Id,
    signs: Id,
    lane: Id,
) -> Id {
    let shifted = m.shift_right_logical(types.u32_, signs, lane);
    let sign_bit = m.bitwise_and(types.u32_, shifted, constants.one);
    let negative = m.i_not_equal(types.bool_, sign_bit, constants.zero);
    let magnitude_f = m.convert_u_to_f(types.f32_, magnitude);
    let negated = m.fnegate(types.f32_, magnitude_f);
    m.select(types.f32_, negative, negated, magnitude_f)
}

fn lookup_f32(m: &mut SpirvModule, types: &ShaderTypes, index: Id, values: &[f32]) -> Id {
    let mut value = m.constant_f32(types.f32_, values[0]);
    for (index_value, &candidate_value) in values.iter().enumerate().skip(1) {
        let candidate_index = m.constant_u32(types.u32_, index_value as u32);
        let not_equal = m.i_not_equal(types.bool_, index, candidate_index);
        let candidate = m.constant_f32(types.f32_, candidate_value);
        value = m.select(types.f32_, not_equal, value, candidate);
    }
    value
}

fn lookup_u32(m: &mut SpirvModule, types: &ShaderTypes, index: Id, values: &[u32]) -> Id {
    let mut value = m.constant_u32(types.u32_, values[0]);
    for (index_value, &candidate_value) in values.iter().enumerate().skip(1) {
        let candidate_index = m.constant_u32(types.u32_, index_value as u32);
        let not_equal = m.i_not_equal(types.bool_, index, candidate_index);
        let candidate = m.constant_u32(types.u32_, candidate_value);
        value = m.select(types.u32_, not_equal, value, candidate);
    }
    value
}

fn decode_packed_nibble(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    _constants: &Constants,
    packed: Id,
    lower_half: Id,
) -> Id {
    let mask = m.constant_u32(types.u32_, 0x0f);
    let shift = m.constant_u32(types.u32_, 4);
    let lower = m.bitwise_and(types.u32_, packed, mask);
    let upper = m.shift_right_logical(types.u32_, packed, shift);
    m.select(types.u32_, lower_half, lower, upper)
}

fn e8m0_half_to_f32(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    exponent: Id,
) -> Id {
    let two = m.constant_u32(types.u32_, 2);
    let twenty_three = m.constant_u32(types.u32_, 23);
    let denorm_base = m.constant_u32(types.u32_, 0x0020_0000);
    let is_denorm = m.u_less_than(types.bool_, exponent, two);
    let denorm_bits = m.shift_left_logical(types.u32_, denorm_base, exponent);
    let adjusted = m.isub(types.u32_, exponent, constants.one);
    let normal_bits = m.shift_left_logical(types.u32_, adjusted, twenty_three);
    let bits = m.select(types.u32_, is_denorm, denorm_bits, normal_bits);
    m.bitcast(types.f32_, bits)
}

fn ue4m3_to_f32(
    m: &mut SpirvModule,
    types: &ShaderTypes,
    constants: &Constants,
    encoded: Id,
) -> Id {
    let seven = m.constant_u32(types.u32_, 7);
    let twenty = m.constant_u32(types.u32_, 20);
    let twenty_three = m.constant_u32(types.u32_, 23);
    let exponent_bias = m.constant_u32(types.u32_, 119);
    let special = m.constant_u32(types.u32_, 0x7f);
    let denorm_scale = m.constant_f32(types.f32_, 0.0009765625);
    let zero_f = m.constant_f32(types.f32_, 0.0);
    let exponent_mask = m.constant_u32(types.u32_, 0x0f);

    let exponent_shifted = m.shift_right_logical(types.u32_, encoded, constants.three);
    let exponent = m.bitwise_and(types.u32_, exponent_shifted, exponent_mask);
    let mantissa = m.bitwise_and(types.u32_, encoded, seven);
    let mantissa_f = m.convert_u_to_f(types.f32_, mantissa);
    let denormal = m.fmul(types.f32_, mantissa_f, denorm_scale);
    let biased_exponent = m.iadd(types.u32_, exponent, exponent_bias);
    let exponent_bits = m.shift_left_logical(types.u32_, biased_exponent, twenty_three);
    let mantissa_bits = m.shift_left_logical(types.u32_, mantissa, twenty);
    let normal_bits = m.bitwise_or(types.u32_, exponent_bits, mantissa_bits);
    let normal = m.bitcast(types.f32_, normal_bits);
    let exponent_nonzero = m.i_not_equal(types.bool_, exponent, constants.zero);
    let decoded = m.select(types.f32_, exponent_nonzero, normal, denormal);
    let encoded_nonzero = m.i_not_equal(types.bool_, encoded, constants.zero);
    let encoded_not_special = m.i_not_equal(types.bool_, encoded, special);
    let valid = m.logical_and(types.bool_, encoded_nonzero, encoded_not_special);
    m.select(types.f32_, valid, decoded, zero_f)
}

fn decode_value(
    m: &mut SpirvModule,
    quant: QuantType,
    types: &ShaderTypes,
    constants: &Constants,
    codebooks: &Codebooks,
    weight: Id,
    block_base: Id,
    intra: Id,
) -> Result<Id, String> {
    let f8 = m.constant_f32(types.f32_, 8.0);
    let f16 = m.constant_f32(types.f32_, 16.0);
    let f32 = m.constant_f32(types.f32_, 32.0);
    let c_u32_0f = m.constant_u32(types.u32_, 0x0f);
    let c_u32_3 = m.constant_u32(types.u32_, 3);
    let c_u32_4 = m.constant_u32(types.u32_, 4);
    let c_u32_8 = m.constant_u32(types.u32_, 8);
    let c_u32_16 = m.constant_u32(types.u32_, 16);
    let c_u32_32 = m.constant_u32(types.u32_, 32);
    let c_u32_6 = m.constant_u32(types.u32_, 6);
    let c_u32_7 = m.constant_u32(types.u32_, 7);
    let c_u32_9 = m.constant_u32(types.u32_, 9);
    let c_u32_12 = m.constant_u32(types.u32_, 12);
    let c_u32_28 = m.constant_u32(types.u32_, 28);
    let c_u32_127 = m.constant_u32(types.u32_, 127);
    let c_u32_80 = m.constant_u32(types.u32_, 0x80);
    let c_u32_f0 = m.constant_u32(types.u32_, 0x00f0);
    let c_u32_700 = m.constant_u32(types.u32_, 0x0700);
    let c_u32_f00 = m.constant_u32(types.u32_, 0x0f00);
    let c_u32_f000 = m.constant_u32(types.u32_, 0xf000);
    let c_u32_8000 = m.constant_u32(types.u32_, 0x8000);

    let value = match quant {
        QuantType::F32 => {
            let byte_offset = m.imul(types.u32_, intra, constants.four);
            let addr = m.iadd(types.u32_, block_base, byte_offset);
            let bits = load_u32_le(m, types, constants, weight, addr);
            m.bitcast(types.f32_, bits)
        }
        QuantType::F16 => {
            let byte_offset = m.imul(types.u32_, intra, constants.two);
            let addr = m.iadd(types.u32_, block_base, byte_offset);
            load_f16(m, types, constants, weight, addr)
        }
        QuantType::BF16 => {
            let byte_offset = m.imul(types.u32_, intra, constants.two);
            let addr = m.iadd(types.u32_, block_base, byte_offset);
            let bits16 = load_u16_le(m, types, constants, weight, addr);
            let shifted = m.shift_left_logical(types.u32_, bits16, c_u32_16);
            m.bitcast(types.f32_, shifted)
        }
        QuantType::Q4_0 | QuantType::Q4_1 => {
            let d = load_f16(m, types, constants, weight, block_base);
            let has_min = quant == QuantType::Q4_1;
            let qs_offset = if has_min { 4 } else { 2 };
            let index = m.umod(types.u32_, intra, c_u32_16);
            let qs_base = add_offset(m, types, block_base, qs_offset);
            let q_addr = m.iadd(types.u32_, qs_base, index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let lower_half = m.u_less_than(types.bool_, intra, c_u32_16);
            let upper = m.shift_right_logical(types.u32_, packed, c_u32_4);
            let lower = m.bitwise_and(types.u32_, packed, c_u32_0f);
            let q = m.select(types.u32_, lower_half, lower, upper);
            let qf = m.convert_u_to_f(types.f32_, q);
            if has_min {
                let m_addr = add_offset(m, types, block_base, 2);
                let min = load_f16(m, types, constants, weight, m_addr);
                let scaled = m.fmul(types.f32_, qf, d);
                m.fadd(types.f32_, scaled, min)
            } else {
                let centered = m.fsub(types.f32_, qf, f8);
                m.fmul(types.f32_, centered, d)
            }
        }
        QuantType::Q5_0 | QuantType::Q5_1 => {
            let d = load_f16(m, types, constants, weight, block_base);
            let has_min = quant == QuantType::Q5_1;
            let qh_offset = if has_min { 4 } else { 2 };
            let qs_offset = if has_min { 8 } else { 6 };
            let qh_addr = add_offset(m, types, block_base, qh_offset);
            let qh = load_u32_le(m, types, constants, weight, qh_addr);
            let high_shifted = m.shift_right_logical(types.u32_, qh, intra);
            let high_bit = m.bitwise_and(types.u32_, high_shifted, constants.one);
            let high_value = m.shift_left_logical(types.u32_, high_bit, c_u32_4);
            let index = m.umod(types.u32_, intra, c_u32_16);
            let qs_base = add_offset(m, types, block_base, qs_offset);
            let q_addr = m.iadd(types.u32_, qs_base, index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let lower_half = m.u_less_than(types.bool_, intra, c_u32_16);
            let upper = m.shift_right_logical(types.u32_, packed, c_u32_4);
            let lower = m.bitwise_and(types.u32_, packed, c_u32_0f);
            let low = m.select(types.u32_, lower_half, lower, upper);
            let q = m.bitwise_or(types.u32_, low, high_value);
            let qf = m.convert_u_to_f(types.f32_, q);
            if has_min {
                let m_addr = add_offset(m, types, block_base, 2);
                let min = load_f16(m, types, constants, weight, m_addr);
                let scaled = m.fmul(types.f32_, qf, d);
                m.fadd(types.f32_, scaled, min)
            } else {
                let centered = m.fsub(types.f32_, qf, f16);
                m.fmul(types.f32_, centered, d)
            }
        }
        QuantType::Q8_1 => {
            let d = load_f16(m, types, constants, weight, block_base);
            let qs_base = add_offset(m, types, block_base, 4);
            let q_addr = m.iadd(types.u32_, qs_base, intra);
            let q = load_byte(m, types, constants, weight, q_addr);
            let qf = signed_byte_to_f32(m, types, q);
            m.fmul(types.f32_, qf, d)
        }
        QuantType::Q2K => {
            let d_addr = add_offset(m, types, block_base, 80);
            let dmin_addr = add_offset(m, types, block_base, 82);
            let d = load_f16(m, types, constants, weight, d_addr);
            let dmin = load_f16(m, types, constants, weight, dmin_addr);
            let group = m.udiv(types.u32_, intra, c_u32_16);
            let sc_addr = m.iadd(types.u32_, block_base, group);
            let sc = load_byte(m, types, constants, weight, sc_addr);
            let scale_code = m.bitwise_and(types.u32_, sc, c_u32_0f);
            let min_code = m.shift_right_logical(types.u32_, sc, c_u32_4);
            let half = m.udiv(types.u32_, group, c_u32_8);
            let group_in_half = m.umod(types.u32_, group, c_u32_8);
            let parity = m.umod(types.u32_, group_in_half, constants.two);
            let half_q_offset = m.imul(types.u32_, half, c_u32_32);
            let parity_q_offset = m.imul(types.u32_, parity, c_u32_16);
            let q_base0 = m.iadd(types.u32_, half_q_offset, parity_q_offset);
            let lane = m.umod(types.u32_, intra, c_u32_16);
            let q_base1 = m.iadd(types.u32_, q_base0, lane);
            let q_base = add_offset(m, types, block_base, 16);
            let q_addr = m.iadd(types.u32_, q_base, q_base1);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let plane = m.udiv(types.u32_, group_in_half, constants.two);
            let shift = m.imul(types.u32_, plane, constants.two);
            let shifted = m.shift_right_logical(types.u32_, packed, shift);
            let q = m.bitwise_and(types.u32_, shifted, c_u32_3);
            let qf = m.convert_u_to_f(types.f32_, q);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let min_f = m.convert_u_to_f(types.f32_, min_code);
            let scale = m.fmul(types.f32_, d, scale_f);
            let min = m.fmul(types.f32_, dmin, min_f);
            let scaled = m.fmul(types.f32_, qf, scale);
            m.fsub(types.f32_, scaled, min)
        }
        QuantType::Q3K => {
            let d_addr = add_offset(m, types, block_base, 108);
            let d = load_f16(m, types, constants, weight, d_addr);
            let group = m.udiv(types.u32_, intra, c_u32_16);
            let scale_quartet = m.udiv(types.u32_, group, constants.four);
            let scale_lane = m.umod(types.u32_, group, constants.four);
            let low_source_group = m.umod(types.u32_, scale_quartet, constants.two);
            let low_source_base = m.imul(types.u32_, low_source_group, constants.four);
            let low_source_index = m.iadd(types.u32_, low_source_base, scale_lane);
            let scales_base = add_offset(m, types, block_base, 96);
            let low_addr = m.iadd(types.u32_, scales_base, low_source_index);
            let low_byte = load_byte(m, types, constants, weight, low_addr);
            let lower_quartet = m.u_less_than(types.bool_, scale_quartet, constants.two);
            let low_upper = m.shift_right_logical(types.u32_, low_byte, constants.four);
            let low_lower = m.bitwise_and(types.u32_, low_byte, c_u32_0f);
            let low = m.select(types.u32_, lower_quartet, low_lower, low_upper);
            let high_addr = m.iadd(types.u32_, scales_base, c_u32_8);
            let high_addr = m.iadd(types.u32_, high_addr, scale_lane);
            let high_byte = load_byte(m, types, constants, weight, high_addr);
            let high_shift = m.imul(types.u32_, scale_quartet, constants.two);
            let high_shifted = m.shift_right_logical(types.u32_, high_byte, high_shift);
            let high = m.bitwise_and(types.u32_, high_shifted, c_u32_3);
            let high_part = m.shift_left_logical(types.u32_, high, constants.four);
            let scale_code = m.bitwise_or(types.u32_, low, high_part);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let centered_scale = m.fsub(types.f32_, scale_f, f32);

            let half = m.udiv(types.u32_, group, c_u32_8);
            let group_in_half = m.umod(types.u32_, group, c_u32_8);
            let parity = m.umod(types.u32_, group_in_half, constants.two);
            let q_half_offset = m.imul(types.u32_, half, c_u32_32);
            let q_parity_offset = m.imul(types.u32_, parity, c_u32_16);
            let q_lane = m.umod(types.u32_, intra, c_u32_16);
            let q_index0 = m.iadd(types.u32_, q_half_offset, q_parity_offset);
            let q_index = m.iadd(types.u32_, q_index0, q_lane);
            let qs_base = add_offset(m, types, block_base, 32);
            let q_addr = m.iadd(types.u32_, qs_base, q_index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let plane = m.udiv(types.u32_, group_in_half, constants.two);
            let q_shift = m.imul(types.u32_, plane, constants.two);
            let q_shifted = m.shift_right_logical(types.u32_, packed, q_shift);
            let q_low = m.bitwise_and(types.u32_, q_shifted, c_u32_3);

            let hmask_index0 = m.imul(types.u32_, parity, c_u32_16);
            let hmask_index = m.iadd(types.u32_, hmask_index0, q_lane);
            let hm_addr = m.iadd(types.u32_, block_base, hmask_index);
            let hm = load_byte(m, types, constants, weight, hm_addr);
            let bit_index = m.udiv(types.u32_, group, constants.two);
            let high_mask = m.shift_left_logical(types.u32_, constants.one, bit_index);
            let high_set_bits = m.bitwise_and(types.u32_, hm, high_mask);
            let high_set = m.i_not_equal(types.bool_, high_set_bits, constants.zero);
            let q_unsigned = m.convert_u_to_f(types.f32_, q_low);
            let c_f32_4 = m.constant_f32(types.f32_, 4.0);
            let q_negative = m.fsub(types.f32_, q_unsigned, c_f32_4);
            let qf = m.select(types.f32_, high_set, q_unsigned, q_negative);
            let scaled_d = m.fmul(types.f32_, d, centered_scale);
            m.fmul(types.f32_, scaled_d, qf)
        }
        QuantType::Q8K => {
            let d_bits = load_u32_le(m, types, constants, weight, block_base);
            let d = m.bitcast(types.f32_, d_bits);
            let qs_base = add_offset(m, types, block_base, 4);
            let q_addr = m.iadd(types.u32_, qs_base, intra);
            let q = load_byte(m, types, constants, weight, q_addr);
            let qf = signed_byte_to_f32(m, types, q);
            m.fmul(types.f32_, qf, d)
        }
        QuantType::IQ4_NL => {
            let d = load_f16(m, types, constants, weight, block_base);
            let lane = m.umod(types.u32_, intra, c_u32_16);
            let qs_base = add_offset(m, types, block_base, 2);
            let q_addr = m.iadd(types.u32_, qs_base, lane);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let lower_half = m.u_less_than(types.bool_, intra, c_u32_16);
            let q = decode_packed_nibble(m, types, constants, packed, lower_half);
            let qf = lookup_f32(
                m,
                types,
                q,
                &[
                    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0,
                    38.0, 53.0, 69.0, 89.0, 113.0,
                ],
            );
            m.fmul(types.f32_, qf, d)
        }
        QuantType::IQ4_XS => {
            let d = load_f16(m, types, constants, weight, block_base);
            let group = m.udiv(types.u32_, intra, c_u32_32);
            let group_pair = m.udiv(types.u32_, group, constants.two);
            let scales_l_base = add_offset(m, types, block_base, 4);
            let scales_l_addr = m.iadd(types.u32_, scales_l_base, group_pair);
            let scales_l = load_byte(m, types, constants, weight, scales_l_addr);
            let group_parity = m.umod(types.u32_, group, constants.two);
            let low_shift = m.imul(types.u32_, group_parity, constants.four);
            let low_shifted = m.shift_right_logical(types.u32_, scales_l, low_shift);
            let low = m.bitwise_and(types.u32_, low_shifted, c_u32_0f);
            let scales_h_addr = add_offset(m, types, block_base, 2);
            let scales_h = load_u16_le(m, types, constants, weight, scales_h_addr);
            let high_shift = m.imul(types.u32_, group, constants.two);
            let high_shifted = m.shift_right_logical(types.u32_, scales_h, high_shift);
            let high = m.bitwise_and(types.u32_, high_shifted, c_u32_3);
            let high_part = m.shift_left_logical(types.u32_, high, constants.four);
            let scale_code = m.bitwise_or(types.u32_, low, high_part);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let centered_scale = m.fsub(types.f32_, scale_f, f32);
            let scale = m.fmul(types.f32_, d, centered_scale);

            let within = m.umod(types.u32_, intra, c_u32_32);
            let lane = m.umod(types.u32_, within, c_u32_16);
            let group_q_offset = m.imul(types.u32_, group, c_u32_16);
            let q_index = m.iadd(types.u32_, group_q_offset, lane);
            let qs_base = add_offset(m, types, block_base, 8);
            let q_addr = m.iadd(types.u32_, qs_base, q_index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let lower_half = m.u_less_than(types.bool_, within, c_u32_16);
            let q = decode_packed_nibble(m, types, constants, packed, lower_half);
            let qf = lookup_f32(
                m,
                types,
                q,
                &[
                    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0,
                    38.0, 53.0, 69.0, 89.0, 113.0,
                ],
            );
            m.fmul(types.f32_, scale, qf)
        }
        QuantType::TQ1_0 => {
            let first_limit = m.constant_u32(types.u32_, 160);
            let second_base = m.constant_u32(types.u32_, 160);
            let second_limit = m.constant_u32(types.u32_, 240);
            let third_base = m.constant_u32(types.u32_, 240);
            let sixteen = m.constant_u32(types.u32_, 16);
            let thirty_two = m.constant_u32(types.u32_, 32);
            let forty_eight = m.constant_u32(types.u32_, 48);
            let first = m.u_less_than(types.bool_, intra, first_limit);
            let first_index = m.umod(types.u32_, intra, thirty_two);
            let first_power = m.udiv(types.u32_, intra, thirty_two);
            let second_rel = m.isub(types.u32_, intra, second_base);
            let second_lane = m.umod(types.u32_, second_rel, sixteen);
            let second_index = m.iadd(types.u32_, thirty_two, second_lane);
            let second_power = m.udiv(types.u32_, second_rel, sixteen);
            let third_rel = m.isub(types.u32_, intra, third_base);
            let third_lane = m.umod(types.u32_, third_rel, constants.four);
            let third_index = m.iadd(types.u32_, forty_eight, third_lane);
            let third_power = m.udiv(types.u32_, third_rel, constants.four);
            let second = m.u_less_than(types.bool_, intra, second_limit);
            let later_index = m.select(types.u32_, second, second_index, third_index);
            let byte_index = m.select(types.u32_, first, first_index, later_index);
            let later_power = m.select(types.u32_, second, second_power, third_power);
            let power_index = m.select(types.u32_, first, first_power, later_power);
            let q_addr = m.iadd(types.u32_, block_base, byte_index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let power = lookup_u32(m, types, power_index, &[1, 3, 9, 27, 81]);
            let multiplied = m.imul(types.u32_, packed, power);
            let wrapped = m.bitwise_and(types.u32_, multiplied, constants.ff);
            let tripled = m.imul(types.u32_, wrapped, constants.three);
            let decoded = m.shift_right_logical(types.u32_, tripled, constants.eight);
            let decoded_f = m.convert_u_to_f(types.f32_, decoded);
            let one_f = m.constant_f32(types.f32_, 1.0);
            let centered = m.fsub(types.f32_, decoded_f, one_f);
            let d_addr = add_offset(m, types, block_base, 52);
            let d = load_f16(m, types, constants, weight, d_addr);
            m.fmul(types.f32_, centered, d)
        }
        QuantType::TQ2_0 => {
            let group_span = m.constant_u32(types.u32_, 128);
            let lane_span = m.constant_u32(types.u32_, 32);
            let group = m.udiv(types.u32_, intra, group_span);
            let within_group = m.umod(types.u32_, intra, group_span);
            let group_offset = m.imul(types.u32_, group, lane_span);
            let lane = m.umod(types.u32_, within_group, lane_span);
            let byte_index = m.iadd(types.u32_, group_offset, lane);
            let q_addr = m.iadd(types.u32_, block_base, byte_index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let plane = m.udiv(types.u32_, within_group, lane_span);
            let shift = m.imul(types.u32_, plane, constants.two);
            let shifted = m.shift_right_logical(types.u32_, packed, shift);
            let q = m.bitwise_and(types.u32_, shifted, c_u32_3);
            let qf = m.convert_u_to_f(types.f32_, q);
            let one_f = m.constant_f32(types.f32_, 1.0);
            let centered = m.fsub(types.f32_, qf, one_f);
            let d_addr = add_offset(m, types, block_base, 64);
            let d = load_f16(m, types, constants, weight, d_addr);
            m.fmul(types.f32_, centered, d)
        }
        QuantType::MXFP4 => {
            let exponent = load_byte(m, types, constants, weight, block_base);
            let d = e8m0_half_to_f32(m, types, constants, exponent);
            let lane = m.umod(types.u32_, intra, c_u32_16);
            let qs_base = add_offset(m, types, block_base, 1);
            let q_addr = m.iadd(types.u32_, qs_base, lane);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let lower_half = m.u_less_than(types.bool_, intra, c_u32_16);
            let q = decode_packed_nibble(m, types, constants, packed, lower_half);
            let qf = lookup_f32(
                m,
                types,
                q,
                &[
                    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0,
                    -8.0, -12.0,
                ],
            );
            m.fmul(types.f32_, qf, d)
        }
        QuantType::NVFP4 => {
            let subblock = m.udiv(types.u32_, intra, c_u32_16);
            let within = m.umod(types.u32_, intra, c_u32_16);
            let scale_addr = m.iadd(types.u32_, block_base, subblock);
            let scale_encoded = load_byte(m, types, constants, weight, scale_addr);
            let d = ue4m3_to_f32(m, types, constants, scale_encoded);
            let subblock_q_offset = m.imul(types.u32_, subblock, constants.eight);
            let lane = m.umod(types.u32_, within, constants.eight);
            let q_index = m.iadd(types.u32_, subblock_q_offset, lane);
            let qs_base = add_offset(m, types, block_base, 4);
            let q_addr = m.iadd(types.u32_, qs_base, q_index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let lower_half = m.u_less_than(types.bool_, within, constants.eight);
            let q = decode_packed_nibble(m, types, constants, packed, lower_half);
            let qf = lookup_f32(
                m,
                types,
                q,
                &[
                    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0,
                    -8.0, -12.0,
                ],
            );
            m.fmul(types.f32_, qf, d)
        }
        QuantType::Q1_0 => {
            let d = load_f16(m, types, constants, weight, block_base);
            let byte_index = m.udiv(types.u32_, intra, constants.eight);
            let qs_base = add_offset(m, types, block_base, 2);
            let q_addr = m.iadd(types.u32_, qs_base, byte_index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let bit_index = m.umod(types.u32_, intra, constants.eight);
            let shifted = m.shift_right_logical(types.u32_, packed, bit_index);
            let bit = m.bitwise_and(types.u32_, shifted, constants.one);
            let positive = m.i_not_equal(types.bool_, bit, constants.zero);
            let neg_d = m.fnegate(types.f32_, d);
            m.select(types.f32_, positive, d, neg_d)
        }
        QuantType::Q2_0 => {
            let d = load_f16(m, types, constants, weight, block_base);
            let byte_index = m.udiv(types.u32_, intra, constants.four);
            let qs_base = add_offset(m, types, block_base, 2);
            let q_addr = m.iadd(types.u32_, qs_base, byte_index);
            let packed = load_byte(m, types, constants, weight, q_addr);
            let lane = m.umod(types.u32_, intra, constants.four);
            let shift = m.imul(types.u32_, lane, constants.two);
            let shifted = m.shift_right_logical(types.u32_, packed, shift);
            let q = m.bitwise_and(types.u32_, shifted, c_u32_3);
            let qf = m.convert_u_to_f(types.f32_, q);
            let one_f = m.constant_f32(types.f32_, 1.0);
            let centered = m.fsub(types.f32_, qf, one_f);
            m.fmul(types.f32_, centered, d)
        }
        QuantType::IQ2_XXS => {
            let d = load_f16(m, types, constants, weight, block_base);
            let ib32 = m.udiv(types.u32_, intra, c_u32_32);
            let within = m.umod(types.u32_, intra, c_u32_32);
            let subgroup = m.udiv(types.u32_, within, constants.eight);
            let lane = m.umod(types.u32_, within, constants.eight);
            let group_offset = m.imul(types.u32_, ib32, constants.eight);
            let packed_base = add_offset(m, types, block_base, 2);
            let group_base = m.iadd(types.u32_, packed_base, group_offset);
            let index_addr = m.iadd(types.u32_, group_base, subgroup);
            let grid_index = load_byte(m, types, constants, weight, index_addr);
            let aux_addr = add_offset(m, types, group_base, 4);
            let aux = load_u32_le(m, types, constants, weight, aux_addr);
            let scale_code = m.shift_right_logical(types.u32_, aux, c_u32_28);
            let sign_shift = m.imul(types.u32_, subgroup, c_u32_7);
            let sign_index_raw = m.shift_right_logical(types.u32_, aux, sign_shift);
            let sign_index = m.bitwise_and(types.u32_, sign_index_raw, c_u32_127);
            let signs = load_sign_byte(m, types, constants, codebooks, sign_index);
            let magnitude = load_grid_byte(m, types, constants, codebooks, grid_index, lane);
            let signed = apply_sign_bit(m, types, constants, magnitude, signs, lane);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let half = m.constant_f32(types.f32_, 0.5);
            let quarter = m.constant_f32(types.f32_, 0.25);
            let scale_plus_half = m.fadd(types.f32_, scale_f, half);
            let block_scale = m.fmul(types.f32_, d, scale_plus_half);
            let db = m.fmul(types.f32_, block_scale, quarter);
            m.fmul(types.f32_, db, signed)
        }
        QuantType::IQ2_XS => {
            let d = load_f16(m, types, constants, weight, block_base);
            let ib32 = m.udiv(types.u32_, intra, c_u32_32);
            let within = m.umod(types.u32_, intra, c_u32_32);
            let subgroup = m.udiv(types.u32_, within, constants.eight);
            let lane = m.umod(types.u32_, within, constants.eight);
            let group_offset = m.imul(types.u32_, ib32, constants.eight);
            let subgroup_offset = m.imul(types.u32_, subgroup, constants.two);
            let packed_offset = m.iadd(types.u32_, group_offset, subgroup_offset);
            let qs_base = add_offset(m, types, block_base, 2);
            let packed_addr = m.iadd(types.u32_, qs_base, packed_offset);
            let packed = load_u16_le(m, types, constants, weight, packed_addr);
            let index_mask = m.constant_u32(types.u32_, 0x01ff);
            let grid_index = m.bitwise_and(types.u32_, packed, index_mask);
            let sign_index = m.shift_right_logical(types.u32_, packed, c_u32_9);
            let signs = load_sign_byte(m, types, constants, codebooks, sign_index);
            let magnitude = load_grid_byte(m, types, constants, codebooks, grid_index, lane);
            let signed = apply_sign_bit(m, types, constants, magnitude, signs, lane);
            let scales_base = add_offset(m, types, block_base, 66);
            let scale_addr = m.iadd(types.u32_, scales_base, ib32);
            let scale_byte = load_byte(m, types, constants, weight, scale_addr);
            let lower_scale = m.bitwise_and(types.u32_, scale_byte, c_u32_0f);
            let upper_scale = m.shift_right_logical(types.u32_, scale_byte, constants.four);
            let first_half = m.u_less_than(types.bool_, subgroup, constants.two);
            let scale_code = m.select(types.u32_, first_half, lower_scale, upper_scale);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let half = m.constant_f32(types.f32_, 0.5);
            let quarter = m.constant_f32(types.f32_, 0.25);
            let scale_plus_half = m.fadd(types.f32_, scale_f, half);
            let block_scale = m.fmul(types.f32_, d, scale_plus_half);
            let db = m.fmul(types.f32_, block_scale, quarter);
            m.fmul(types.f32_, db, signed)
        }
        QuantType::IQ2_S => {
            let d = load_f16(m, types, constants, weight, block_base);
            let ib32 = m.udiv(types.u32_, intra, c_u32_32);
            let within = m.umod(types.u32_, intra, c_u32_32);
            let subgroup = m.udiv(types.u32_, within, constants.eight);
            let lane = m.umod(types.u32_, within, constants.eight);
            let group_offset = m.imul(types.u32_, ib32, constants.four);
            let low_index = m.iadd(types.u32_, group_offset, subgroup);
            let qs_base = add_offset(m, types, block_base, 2);
            let low_addr = m.iadd(types.u32_, qs_base, low_index);
            let low = load_byte(m, types, constants, weight, low_addr);
            let qh_base = add_offset(m, types, block_base, 66);
            let qh_addr = m.iadd(types.u32_, qh_base, ib32);
            let qh = load_byte(m, types, constants, weight, qh_addr);
            let high_shift_base = m.constant_u32(types.u32_, 8);
            let high_shift_delta = m.imul(types.u32_, subgroup, constants.two);
            let high_shift = m.isub(types.u32_, high_shift_base, high_shift_delta);
            let high_shifted = m.shift_left_logical(types.u32_, qh, high_shift);
            let high_mask = m.constant_u32(types.u32_, 0x0300);
            let high = m.bitwise_and(types.u32_, high_shifted, high_mask);
            let grid_index = m.bitwise_or(types.u32_, low, high);
            let sign_index = add_offset(m, types, low_index, 32);
            let sign_addr = m.iadd(types.u32_, qs_base, sign_index);
            let signs = load_byte(m, types, constants, weight, sign_addr);
            let magnitude = load_grid_byte(m, types, constants, codebooks, grid_index, lane);
            let signed = apply_sign_bit(m, types, constants, magnitude, signs, lane);
            let scales_base = add_offset(m, types, block_base, 74);
            let scale_addr = m.iadd(types.u32_, scales_base, ib32);
            let scale_byte = load_byte(m, types, constants, weight, scale_addr);
            let lower_scale = m.bitwise_and(types.u32_, scale_byte, c_u32_0f);
            let upper_scale = m.shift_right_logical(types.u32_, scale_byte, constants.four);
            let first_half = m.u_less_than(types.bool_, subgroup, constants.two);
            let scale_code = m.select(types.u32_, first_half, lower_scale, upper_scale);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let half = m.constant_f32(types.f32_, 0.5);
            let quarter = m.constant_f32(types.f32_, 0.25);
            let scale_plus_half = m.fadd(types.f32_, scale_f, half);
            let block_scale = m.fmul(types.f32_, d, scale_plus_half);
            let db = m.fmul(types.f32_, block_scale, quarter);
            m.fmul(types.f32_, db, signed)
        }
        QuantType::IQ3_XXS => {
            let d = load_f16(m, types, constants, weight, block_base);
            let ib32 = m.udiv(types.u32_, intra, c_u32_32);
            let within = m.umod(types.u32_, intra, c_u32_32);
            let subgroup = m.udiv(types.u32_, within, constants.eight);
            let lane8 = m.umod(types.u32_, within, constants.eight);
            let second_grid = m.u_less_than(types.bool_, lane8, constants.four);
            let grid_pair_base = m.imul(types.u32_, ib32, constants.eight);
            let subgroup_pair = m.imul(types.u32_, subgroup, constants.two);
            let first_index = m.iadd(types.u32_, grid_pair_base, subgroup_pair);
            let second_index = m.iadd(types.u32_, first_index, constants.one);
            let selected_index = m.select(types.u32_, second_grid, first_index, second_index);
            let qs_base = add_offset(m, types, block_base, 2);
            let index_addr = m.iadd(types.u32_, qs_base, selected_index);
            let grid_index = load_byte(m, types, constants, weight, index_addr);
            let grid_lane = m.umod(types.u32_, lane8, constants.four);
            let magnitude = load_grid_byte(m, types, constants, codebooks, grid_index, grid_lane);
            let aux_offset = m.imul(types.u32_, ib32, constants.four);
            let aux_base = add_offset(m, types, block_base, 66);
            let aux_addr = m.iadd(types.u32_, aux_base, aux_offset);
            let aux = load_u32_le(m, types, constants, weight, aux_addr);
            let sign_shift = m.imul(types.u32_, subgroup, c_u32_7);
            let sign_index_raw = m.shift_right_logical(types.u32_, aux, sign_shift);
            let sign_index = m.bitwise_and(types.u32_, sign_index_raw, c_u32_127);
            let signs = load_sign_byte(m, types, constants, codebooks, sign_index);
            let signed = apply_sign_bit(m, types, constants, magnitude, signs, lane8);
            let scale_code = m.shift_right_logical(types.u32_, aux, c_u32_28);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let half = m.constant_f32(types.f32_, 0.5);
            let scale_plus_half = m.fadd(types.f32_, scale_f, half);
            let block_scale = m.fmul(types.f32_, d, scale_plus_half);
            let db = m.fmul(types.f32_, block_scale, half);
            m.fmul(types.f32_, db, signed)
        }
        QuantType::IQ3_S => {
            let d = load_f16(m, types, constants, weight, block_base);
            let ib32 = m.udiv(types.u32_, intra, c_u32_32);
            let within = m.umod(types.u32_, intra, c_u32_32);
            let subgroup = m.udiv(types.u32_, within, constants.eight);
            let lane8 = m.umod(types.u32_, within, constants.eight);
            let first_grid = m.u_less_than(types.bool_, lane8, constants.four);
            let grid_pair_base = m.imul(types.u32_, ib32, constants.eight);
            let subgroup_pair = m.imul(types.u32_, subgroup, constants.two);
            let index0 = m.iadd(types.u32_, grid_pair_base, subgroup_pair);
            let index1 = m.iadd(types.u32_, index0, constants.one);
            let qs_base = add_offset(m, types, block_base, 2);
            let index0_addr = m.iadd(types.u32_, qs_base, index0);
            let index1_addr = m.iadd(types.u32_, qs_base, index1);
            let low0 = load_byte(m, types, constants, weight, index0_addr);
            let low1 = load_byte(m, types, constants, weight, index1_addr);
            let qh_base = add_offset(m, types, block_base, 66);
            let qh_addr = m.iadd(types.u32_, qh_base, ib32);
            let qh = load_byte(m, types, constants, weight, qh_addr);
            let shift0_delta = m.imul(types.u32_, subgroup, constants.two);
            let shift0 = m.isub(types.u32_, constants.eight, shift0_delta);
            let shift1 = m.isub(types.u32_, shift0, constants.one);
            let high0_shifted = m.shift_left_logical(types.u32_, qh, shift0);
            let high1_shifted = m.shift_left_logical(types.u32_, qh, shift1);
            let high_mask = m.constant_u32(types.u32_, 0x0100);
            let high0 = m.bitwise_and(types.u32_, high0_shifted, high_mask);
            let high1 = m.bitwise_and(types.u32_, high1_shifted, high_mask);
            let grid_index0 = m.bitwise_or(types.u32_, low0, high0);
            let grid_index1 = m.bitwise_or(types.u32_, low1, high1);
            let grid_index = m.select(types.u32_, first_grid, grid_index0, grid_index1);
            let grid_lane = m.umod(types.u32_, lane8, constants.four);
            let magnitude = load_grid_byte(m, types, constants, codebooks, grid_index, grid_lane);
            let sign_group_base = m.imul(types.u32_, ib32, constants.four);
            let sign_index = m.iadd(types.u32_, sign_group_base, subgroup);
            let signs_base = add_offset(m, types, block_base, 74);
            let sign_addr = m.iadd(types.u32_, signs_base, sign_index);
            let signs = load_byte(m, types, constants, weight, sign_addr);
            let signed = apply_sign_bit(m, types, constants, magnitude, signs, lane8);
            let scale_pair = m.udiv(types.u32_, ib32, constants.two);
            let scales_base = add_offset(m, types, block_base, 106);
            let scale_addr = m.iadd(types.u32_, scales_base, scale_pair);
            let scale_byte = load_byte(m, types, constants, weight, scale_addr);
            let parity = m.umod(types.u32_, ib32, constants.two);
            let shift = m.imul(types.u32_, parity, constants.four);
            let scale_shifted = m.shift_right_logical(types.u32_, scale_byte, shift);
            let scale_code = m.bitwise_and(types.u32_, scale_shifted, c_u32_0f);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let two_f = m.constant_f32(types.f32_, 2.0);
            let one_f = m.constant_f32(types.f32_, 1.0);
            let doubled = m.fmul(types.f32_, scale_f, two_f);
            let odd_scale = m.fadd(types.f32_, doubled, one_f);
            let db = m.fmul(types.f32_, d, odd_scale);
            m.fmul(types.f32_, db, signed)
        }
        QuantType::IQ1_S => {
            let d = load_f16(m, types, constants, weight, block_base);
            let ib32 = m.udiv(types.u32_, intra, c_u32_32);
            let within = m.umod(types.u32_, intra, c_u32_32);
            let subgroup = m.udiv(types.u32_, within, constants.eight);
            let lane = m.umod(types.u32_, within, constants.eight);
            let qh_offset = m.imul(types.u32_, ib32, constants.two);
            let qh_base = add_offset(m, types, block_base, 34);
            let qh_addr = m.iadd(types.u32_, qh_base, qh_offset);
            let qh = load_u16_le(m, types, constants, weight, qh_addr);
            let scale_raw = m.shift_right_logical(types.u32_, qh, c_u32_12);
            let scale_code = m.bitwise_and(types.u32_, scale_raw, c_u32_7);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let two_f = m.constant_f32(types.f32_, 2.0);
            let one_f = m.constant_f32(types.f32_, 1.0);
            let doubled = m.fmul(types.f32_, scale_f, two_f);
            let odd_scale = m.fadd(types.f32_, doubled, one_f);
            let dl = m.fmul(types.f32_, d, odd_scale);
            let low_group_base = m.imul(types.u32_, ib32, constants.four);
            let low_offset = m.iadd(types.u32_, low_group_base, subgroup);
            let qs_base = add_offset(m, types, block_base, 2);
            let low_addr = m.iadd(types.u32_, qs_base, low_offset);
            let low = load_byte(m, types, constants, weight, low_addr);
            let high_shift = m.imul(types.u32_, subgroup, constants.three);
            let high_raw = m.shift_right_logical(types.u32_, qh, high_shift);
            let high_three = m.bitwise_and(types.u32_, high_raw, c_u32_7);
            let high = m.shift_left_logical(types.u32_, high_three, constants.eight);
            let grid_index = m.bitwise_or(types.u32_, low, high);
            let grid_byte = load_grid_byte(m, types, constants, codebooks, grid_index, lane);
            let grid_value = signed_byte_to_f32(m, types, grid_byte);
            let delta_sign_raw = m.bitwise_and(types.u32_, qh, c_u32_8000);
            let delta_negative = m.i_not_equal(types.bool_, delta_sign_raw, constants.zero);
            let delta_pos = m.constant_f32(types.f32_, 0.125);
            let delta_neg = m.constant_f32(types.f32_, -0.125);
            let delta = m.select(types.f32_, delta_negative, delta_neg, delta_pos);
            let adjusted = m.fadd(types.f32_, grid_value, delta);
            m.fmul(types.f32_, dl, adjusted)
        }
        QuantType::IQ1_M => {
            let scale0_addr = add_offset(m, types, block_base, 48);
            let scale1_addr = add_offset(m, types, block_base, 50);
            let scale2_addr = add_offset(m, types, block_base, 52);
            let scale3_addr = add_offset(m, types, block_base, 54);
            let scale0 = load_u16_le(m, types, constants, weight, scale0_addr);
            let scale1 = load_u16_le(m, types, constants, weight, scale1_addr);
            let scale2 = load_u16_le(m, types, constants, weight, scale2_addr);
            let scale3 = load_u16_le(m, types, constants, weight, scale3_addr);
            let part0 = m.shift_right_logical(types.u32_, scale0, c_u32_12);
            let part1_raw = m.shift_right_logical(types.u32_, scale1, constants.eight);
            let part1 = m.bitwise_and(types.u32_, part1_raw, c_u32_f0);
            let part2_raw = m.shift_right_logical(types.u32_, scale2, constants.four);
            let part2 = m.bitwise_and(types.u32_, part2_raw, c_u32_f00);
            let part3 = m.bitwise_and(types.u32_, scale3, c_u32_f000);
            let d01 = m.bitwise_or(types.u32_, part0, part1);
            let d23 = m.bitwise_or(types.u32_, part2, part3);
            let d_bits = m.bitwise_or(types.u32_, d01, d23);
            let d = f16_to_f32(m, types, d_bits);

            let ib32 = m.udiv(types.u32_, intra, c_u32_32);
            let within = m.umod(types.u32_, intra, c_u32_32);
            let subgroup = m.udiv(types.u32_, within, constants.eight);
            let lane = m.umod(types.u32_, within, constants.eight);
            let pair = m.udiv(types.u32_, subgroup, constants.two);
            let qh_offset0 = m.imul(types.u32_, ib32, constants.two);
            let qh_offset = m.iadd(types.u32_, qh_offset0, pair);
            let qh_base = add_offset(m, types, block_base, 32);
            let qh_addr = m.iadd(types.u32_, qh_base, qh_offset);
            let qh = load_byte(m, types, constants, weight, qh_addr);
            let subgroup_parity = m.umod(types.u32_, subgroup, constants.two);
            let first_in_pair = m.u_less_than(types.bool_, subgroup_parity, constants.one);
            let high_shift = m.select(types.u32_, first_in_pair, constants.eight, constants.four);
            let high_shifted = m.shift_left_logical(types.u32_, qh, high_shift);
            let high = m.bitwise_and(types.u32_, high_shifted, c_u32_700);
            let low_group_base = m.imul(types.u32_, ib32, constants.four);
            let low_offset = m.iadd(types.u32_, low_group_base, subgroup);
            let low_addr = m.iadd(types.u32_, block_base, low_offset);
            let low = load_byte(m, types, constants, weight, low_addr);
            let grid_index = m.bitwise_or(types.u32_, low, high);
            let grid_byte = load_grid_byte(m, types, constants, codebooks, grid_index, lane);
            let grid_value = signed_byte_to_f32(m, types, grid_byte);
            let delta_mask = m.select(types.u32_, first_in_pair, constants.eight, c_u32_80);
            let delta_raw = m.bitwise_and(types.u32_, qh, delta_mask);
            let delta_negative = m.i_not_equal(types.bool_, delta_raw, constants.zero);
            let delta_pos = m.constant_f32(types.f32_, 0.125);
            let delta_neg = m.constant_f32(types.f32_, -0.125);
            let delta = m.select(types.f32_, delta_negative, delta_neg, delta_pos);
            let adjusted = m.fadd(types.f32_, grid_value, delta);

            let scale_pair = m.udiv(types.u32_, ib32, constants.two);
            let scale_addr_offset = m.imul(types.u32_, scale_pair, constants.two);
            let scales_base = add_offset(m, types, block_base, 48);
            let scale_addr = m.iadd(types.u32_, scales_base, scale_addr_offset);
            let scale_word = load_u16_le(m, types, constants, weight, scale_addr);
            let ib_parity = m.umod(types.u32_, ib32, constants.two);
            let base_shift = m.imul(types.u32_, ib_parity, c_u32_6);
            let second_scale = m.u_less_than(types.bool_, subgroup, constants.two);
            let scale_shift_delta =
                m.select(types.u32_, second_scale, constants.zero, constants.three);
            let scale_shift = m.iadd(types.u32_, base_shift, scale_shift_delta);
            let scale_raw = m.shift_right_logical(types.u32_, scale_word, scale_shift);
            let scale_code = m.bitwise_and(types.u32_, scale_raw, c_u32_7);
            let scale_f = m.convert_u_to_f(types.f32_, scale_code);
            let two_f = m.constant_f32(types.f32_, 2.0);
            let one_f = m.constant_f32(types.f32_, 1.0);
            let doubled = m.fmul(types.f32_, scale_f, two_f);
            let odd_scale = m.fadd(types.f32_, doubled, one_f);
            let dl = m.fmul(types.f32_, d, odd_scale);
            m.fmul(types.f32_, dl, adjusted)
        }
        other => {
            return Err(format!(
                "native quant scalar decoder is not implemented for {other:?}"
            ));
        }
    };

    Ok(value)
}

pub fn emit_native_quant_gemv(quant: QuantType, local_size_x: u32) -> Result<Vec<u32>, String> {
    if quant.has_soa_kernel() {
        return Err(format!(
            "{quant:?} uses a dedicated SoA shader, not native row-major"
        ));
    }

    let mut m = SpirvModule::new();
    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);
    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_private_u32 = m.type_pointer(storage_class::PRIVATE, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let types = ShaderTypes {
        bool_: t_bool,
        u32_: t_u32,
        i32_: t_i32,
        f32_: t_f32,
        ptr_sb_u32: t_ptr_sb_u32,
        ptr_private_u32: t_ptr_private_u32,
    };
    let constants = Constants {
        zero: m.constant_u32(t_u32, 0),
        one: m.constant_u32(t_u32, 1),
        two: m.constant_u32(t_u32, 2),
        three: m.constant_u32(t_u32, 3),
        four: m.constant_u32(t_u32, 4),
        eight: m.constant_u32(t_u32, 8),
        ff: m.constant_u32(t_u32, 0xff),
    };
    let codebooks = emit_codebooks(&mut m, t_u32, quant);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_block_elements = m.constant_u32(t_u32, quant.block_elements() as u32);
    let c_block_bytes = m.constant_u32(t_u32, quant.block_bytes() as u32);

    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let global_id = m.variable(t_ptr_input_u32, storage_class::INPUT);
    m.decorate(weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(weight, decoration::BINDING, &[0]);
    m.decorate(input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(input, decoration::BINDING, &[1]);
    m.decorate(output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(output, decoration::BINDING, &[2]);
    m.decorate(
        global_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let function = m.alloc_id();
    m.entry_point(5, function, "main", &[global_id]);
    m.execution_mode_local_size(function, local_size_x, 1, 1);
    let entry = m.alloc_id();
    let bounds_true = m.alloc_id();
    let bounds_merge = m.alloc_id();
    let loop_header = m.alloc_id();
    let loop_condition = m.alloc_id();
    let loop_body = m.alloc_id();
    let loop_continue = m.alloc_id();
    let loop_merge = m.alloc_id();

    m.function(t_void, function, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[entry.0]));
    let sum_var = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let col_var = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let global = m.load(t_v3u32, global_id);
    let row = m.composite_extract(t_u32, global, 0);
    let rows_ptr = m.access_chain(t_ptr_pc_u32, pc, &[constants.zero]);
    let rows = m.load(t_u32, rows_ptr);
    let cols_ptr = m.access_chain(t_ptr_pc_u32, pc, &[constants.one]);
    let cols = m.load(t_u32, cols_ptr);
    let in_bounds = m.u_less_than(t_bool, row, rows);
    m.selection_merge(bounds_merge, 0);
    m.branch_conditional(in_bounds, bounds_true, bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[bounds_true.0]));
    m.store(sum_var, c_f32_0);
    m.store(col_var, constants.zero);
    let blocks_per_row = m.udiv(t_u32, cols, c_block_elements);
    let row_bytes = m.imul(t_u32, blocks_per_row, c_block_bytes);
    let row_base = m.imul(t_u32, row, row_bytes);
    m.branch(loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[loop_header.0]));
    m.loop_merge(loop_merge, loop_continue, 0);
    m.branch(loop_condition);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[loop_condition.0]));
    let col = m.load(t_u32, col_var);
    let keep_going = m.u_less_than(t_bool, col, cols);
    m.branch_conditional(keep_going, loop_body, loop_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[loop_body.0]));
    let block_index = m.udiv(t_u32, col, c_block_elements);
    let intra = m.umod(t_u32, col, c_block_elements);
    let block_offset = m.imul(t_u32, block_index, c_block_bytes);
    let block_base = m.iadd(t_u32, row_base, block_offset);
    let weight_value = decode_value(
        &mut m, quant, &types, &constants, &codebooks, weight, block_base, intra,
    )?;
    let input_ptr = m.access_chain(t_ptr_sb_f32, input, &[constants.zero, col]);
    let input_value = m.load(t_f32, input_ptr);
    let product = m.fmul(t_f32, weight_value, input_value);
    let sum = m.load(t_f32, sum_var);
    let next_sum = m.fadd(t_f32, sum, product);
    m.store(sum_var, next_sum);
    m.branch(loop_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[loop_continue.0]));
    let next_col = m.iadd(t_u32, col, constants.one);
    m.store(col_var, next_col);
    m.branch(loop_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[loop_merge.0]));
    let final_sum = m.load(t_f32, sum_var);
    let output_ptr = m.access_chain(t_ptr_sb_f32, output, &[constants.zero, row]);
    m.store(output_ptr, final_sum);
    m.branch(bounds_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[bounds_merge.0]));
    m.ret();
    m.function_end();
    Ok(m.encode())
}

pub fn emit_native_quant_logit_argmax(
    quant: QuantType,
    local_size_x: u32,
) -> Result<Vec<u32>, String> {
    if quant.has_soa_kernel() {
        return Err(format!(
            "{quant:?} uses a dedicated output shader, not native row-major"
        ));
    }
    if !local_size_x.is_power_of_two() {
        return Err("native quant argmax local size must be a power of two".into());
    }

    let mut m = SpirvModule::new();
    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_struct_u32 = m.type_struct(&[t_arr_u32]);
    let t_struct_f32 = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32]);
    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_f32 = m.type_array(t_f32, c_local_size);
    let t_shared_u32 = m.type_array(t_u32, c_local_size);
    let t_ptr_sb_struct_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_u32);
    let t_ptr_sb_struct_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_f32);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_private_u32 = m.type_pointer(storage_class::PRIVATE, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_ptr_wg_arr_f32 = m.type_pointer(storage_class::WORKGROUP, t_shared_f32);
    let t_ptr_wg_arr_u32 = m.type_pointer(storage_class::WORKGROUP, t_shared_u32);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_ptr_wg_u32 = m.type_pointer(storage_class::WORKGROUP, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let types = ShaderTypes {
        bool_: t_bool,
        u32_: t_u32,
        i32_: t_i32,
        f32_: t_f32,
        ptr_sb_u32: t_ptr_sb_u32,
        ptr_private_u32: t_ptr_private_u32,
    };
    let constants = Constants {
        zero: m.constant_u32(t_u32, 0),
        one: m.constant_u32(t_u32, 1),
        two: m.constant_u32(t_u32, 2),
        three: m.constant_u32(t_u32, 3),
        four: m.constant_u32(t_u32, 4),
        eight: m.constant_u32(t_u32, 8),
        ff: m.constant_u32(t_u32, 0xff),
    };
    let codebooks = emit_codebooks(&mut m, t_u32, quant);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_neg_inf = m.constant_f32(t_f32, f32::NEG_INFINITY);
    let c_block_elements = m.constant_u32(t_u32, quant.block_elements() as u32);
    let c_block_bytes = m.constant_u32(t_u32, quant.block_bytes() as u32);

    m.decorate(t_struct_u32, decoration::BLOCK, &[]);
    m.decorate(t_struct_f32, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_u32, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_f32, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);

    let input = m.variable(t_ptr_sb_struct_f32, storage_class::STORAGE_BUFFER);
    let weight = m.variable(t_ptr_sb_struct_u32, storage_class::STORAGE_BUFFER);
    let output = m.variable(t_ptr_sb_struct_u32, storage_class::STORAGE_BUFFER);
    let pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let local_id_var = m.variable(t_ptr_input_u32, storage_class::INPUT);
    let shared_values = m.variable(t_ptr_wg_arr_f32, storage_class::WORKGROUP);
    let shared_indices = m.variable(t_ptr_wg_arr_u32, storage_class::WORKGROUP);
    m.decorate(input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(input, decoration::BINDING, &[0]);
    m.decorate(weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(weight, decoration::BINDING, &[1]);
    m.decorate(output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(output, decoration::BINDING, &[2]);
    m.decorate(
        local_id_var,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );

    let function = m.alloc_id();
    m.entry_point(5, function, "main", &[local_id_var]);
    m.execution_mode_local_size(function, local_size_x, 1, 1);
    m.function(t_void, function, 0, t_fn_void);
    let entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[entry.0]));

    let row_var = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let col_var = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let sum_var = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let best_value_var = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let best_index_var = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let step_var = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let local_vector = m.load(t_v3u32, local_id_var);
    let local_id = m.composite_extract(t_u32, local_vector, 0);
    let vocab_ptr = m.access_chain(t_ptr_pc_u32, pc, &[constants.zero]);
    let vocab = m.load(t_u32, vocab_ptr);
    let hidden_ptr = m.access_chain(t_ptr_pc_u32, pc, &[constants.one]);
    let hidden = m.load(t_u32, hidden_ptr);
    let blocks_per_row = m.udiv(t_u32, hidden, c_block_elements);
    let row_bytes = m.imul(t_u32, blocks_per_row, c_block_bytes);
    m.store(row_var, local_id);
    m.store(best_value_var, c_f32_neg_inf);
    m.store(best_index_var, constants.zero);

    let row_header = m.alloc_id();
    let row_condition = m.alloc_id();
    let row_body = m.alloc_id();
    let row_continue = m.alloc_id();
    let row_merge = m.alloc_id();
    m.branch(row_header);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[row_header.0]));
    m.loop_merge(row_merge, row_continue, 0);
    m.branch(row_condition);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[row_condition.0]));
    let row = m.load(t_u32, row_var);
    let row_in_bounds = m.u_less_than(t_bool, row, vocab);
    m.branch_conditional(row_in_bounds, row_body, row_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[row_body.0]));
    m.store(sum_var, c_f32_0);
    m.store(col_var, constants.zero);

    let col_header = m.alloc_id();
    let col_condition = m.alloc_id();
    let col_body = m.alloc_id();
    let col_continue = m.alloc_id();
    let col_merge = m.alloc_id();
    m.branch(col_header);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[col_header.0]));
    m.loop_merge(col_merge, col_continue, 0);
    m.branch(col_condition);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[col_condition.0]));
    let col = m.load(t_u32, col_var);
    let col_in_bounds = m.u_less_than(t_bool, col, hidden);
    m.branch_conditional(col_in_bounds, col_body, col_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[col_body.0]));
    let row_base = m.imul(t_u32, row, row_bytes);
    let block_index = m.udiv(t_u32, col, c_block_elements);
    let intra = m.umod(t_u32, col, c_block_elements);
    let block_offset = m.imul(t_u32, block_index, c_block_bytes);
    let block_base = m.iadd(t_u32, row_base, block_offset);
    let weight_value = decode_value(
        &mut m, quant, &types, &constants, &codebooks, weight, block_base, intra,
    )?;
    let input_ptr = m.access_chain(t_ptr_sb_f32, input, &[constants.zero, col]);
    let input_value = m.load(t_f32, input_ptr);
    let product = m.fmul(t_f32, weight_value, input_value);
    let sum = m.load(t_f32, sum_var);
    let next_sum = m.fadd(t_f32, sum, product);
    m.store(sum_var, next_sum);
    m.branch(col_continue);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[col_continue.0]));
    let next_col = m.iadd(t_u32, col, constants.one);
    m.store(col_var, next_col);
    m.branch(col_header);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[col_merge.0]));

    let final_sum = m.load(t_f32, sum_var);
    let best_value = m.load(t_f32, best_value_var);
    let beats = m.f_ord_greater_than(t_bool, final_sum, best_value);
    let updated_value = m.select(t_f32, beats, final_sum, best_value);
    let best_index = m.load(t_u32, best_index_var);
    let updated_index = m.select(t_u32, beats, row, best_index);
    m.store(best_value_var, updated_value);
    m.store(best_index_var, updated_index);
    m.branch(row_continue);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[row_continue.0]));
    let next_row = m.iadd(t_u32, row, c_local_size);
    m.store(row_var, next_row);
    m.branch(row_header);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[row_merge.0]));

    let final_best_value = m.load(t_f32, best_value_var);
    let final_best_index = m.load(t_u32, best_index_var);
    let shared_value_ptr = m.access_chain(t_ptr_wg_f32, shared_values, &[local_id]);
    let shared_index_ptr = m.access_chain(t_ptr_wg_u32, shared_indices, &[local_id]);
    m.store(shared_value_ptr, final_best_value);
    m.store(shared_index_ptr, final_best_index);
    let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
    let c_semantics = m.constant_u32(
        t_u32,
        memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
    );
    m.control_barrier(c_scope_wg, c_scope_wg, c_semantics);

    let c_half = m.constant_u32(t_u32, local_size_x / 2);
    m.store(step_var, c_half);
    let reduce_header = m.alloc_id();
    let reduce_condition = m.alloc_id();
    let reduce_body = m.alloc_id();
    let reduce_continue = m.alloc_id();
    let reduce_merge = m.alloc_id();
    m.branch(reduce_header);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[reduce_header.0]));
    m.loop_merge(reduce_merge, reduce_continue, 0);
    m.branch(reduce_condition);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[reduce_condition.0]));
    let step = m.load(t_u32, step_var);
    let step_nonzero = m.u_less_than(t_bool, constants.zero, step);
    m.branch_conditional(step_nonzero, reduce_body, reduce_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[reduce_body.0]));
    let active = m.u_less_than(t_bool, local_id, step);
    let active_body = m.alloc_id();
    let active_merge = m.alloc_id();
    m.selection_merge(active_merge, 0);
    m.branch_conditional(active, active_body, active_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[active_body.0]));
    let other_id = m.iadd(t_u32, local_id, step);
    let own_value_ptr = m.access_chain(t_ptr_wg_f32, shared_values, &[local_id]);
    let own_index_ptr = m.access_chain(t_ptr_wg_u32, shared_indices, &[local_id]);
    let other_value_ptr = m.access_chain(t_ptr_wg_f32, shared_values, &[other_id]);
    let other_index_ptr = m.access_chain(t_ptr_wg_u32, shared_indices, &[other_id]);
    let own_value = m.load(t_f32, own_value_ptr);
    let own_index = m.load(t_u32, own_index_ptr);
    let other_value = m.load(t_f32, other_value_ptr);
    let other_index = m.load(t_u32, other_index_ptr);
    let other_beats = m.f_ord_greater_than(t_bool, other_value, own_value);
    let reduced_value = m.select(t_f32, other_beats, other_value, own_value);
    let reduced_index = m.select(t_u32, other_beats, other_index, own_index);
    m.store(own_value_ptr, reduced_value);
    m.store(own_index_ptr, reduced_index);
    m.branch(active_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[active_merge.0]));
    m.control_barrier(c_scope_wg, c_scope_wg, c_semantics);
    m.branch(reduce_continue);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[reduce_continue.0]));
    let next_step = m.shift_right_logical(t_u32, step, constants.one);
    m.store(step_var, next_step);
    m.branch(reduce_header);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[reduce_merge.0]));

    let is_first_lane = m.u_less_than(t_bool, local_id, constants.one);
    let write_body = m.alloc_id();
    let write_merge = m.alloc_id();
    m.selection_merge(write_merge, 0);
    m.branch_conditional(is_first_lane, write_body, write_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[write_body.0]));
    let winner_ptr = m.access_chain(t_ptr_wg_u32, shared_indices, &[constants.zero]);
    let winner = m.load(t_u32, winner_ptr);
    let output_ptr = m.access_chain(t_ptr_sb_u32, output, &[constants.zero, constants.zero]);
    m.store(output_ptr, winner);
    m.branch(write_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[write_merge.0]));
    m.ret();
    m.function_end();
    Ok(m.encode())
}

pub fn emit_native_quant_embed_lookup(
    quant: QuantType,
    local_size_x: u32,
) -> Result<Vec<u32>, String> {
    if quant.has_soa_kernel() {
        return Err(format!(
            "{quant:?} uses a dedicated embedding shader, not native row-major"
        ));
    }

    let mut m = SpirvModule::new();
    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_struct_u32 = m.type_struct(&[t_arr_u32]);
    let t_struct_f32 = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);
    let t_ptr_sb_struct_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_u32);
    let t_ptr_sb_struct_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_f32);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_private_u32 = m.type_pointer(storage_class::PRIVATE, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let types = ShaderTypes {
        bool_: t_bool,
        u32_: t_u32,
        i32_: t_i32,
        f32_: t_f32,
        ptr_sb_u32: t_ptr_sb_u32,
        ptr_private_u32: t_ptr_private_u32,
    };
    let constants = Constants {
        zero: m.constant_u32(t_u32, 0),
        one: m.constant_u32(t_u32, 1),
        two: m.constant_u32(t_u32, 2),
        three: m.constant_u32(t_u32, 3),
        four: m.constant_u32(t_u32, 4),
        eight: m.constant_u32(t_u32, 8),
        ff: m.constant_u32(t_u32, 0xff),
    };
    let codebooks = emit_codebooks(&mut m, t_u32, quant);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_block_elements = m.constant_u32(t_u32, quant.block_elements() as u32);
    let c_block_bytes = m.constant_u32(t_u32, quant.block_bytes() as u32);

    m.decorate(t_struct_u32, decoration::BLOCK, &[]);
    m.decorate(t_struct_f32, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_u32, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_f32, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let token_ids = m.variable(t_ptr_sb_struct_u32, storage_class::STORAGE_BUFFER);
    let weight = m.variable(t_ptr_sb_struct_u32, storage_class::STORAGE_BUFFER);
    let output = m.variable(t_ptr_sb_struct_f32, storage_class::STORAGE_BUFFER);
    let pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let global_id_var = m.variable(t_ptr_input_u32, storage_class::INPUT);
    m.decorate(token_ids, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(token_ids, decoration::BINDING, &[0]);
    m.decorate(weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(weight, decoration::BINDING, &[1]);
    m.decorate(output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(output, decoration::BINDING, &[2]);
    m.decorate(
        global_id_var,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let function = m.alloc_id();
    m.entry_point(5, function, "main", &[global_id_var]);
    m.execution_mode_local_size(function, local_size_x, 1, 1);
    m.function(t_void, function, 0, t_fn_void);
    let entry = m.alloc_id();
    let in_bounds_body = m.alloc_id();
    let function_merge = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[entry.0]));
    let global_vector = m.load(t_v3u32, global_id_var);
    let global_id = m.composite_extract(t_u32, global_vector, 0);
    let num_tokens_ptr = m.access_chain(t_ptr_pc_u32, pc, &[constants.zero]);
    let num_tokens = m.load(t_u32, num_tokens_ptr);
    let hidden_ptr = m.access_chain(t_ptr_pc_u32, pc, &[constants.one]);
    let hidden = m.load(t_u32, hidden_ptr);
    let vocab_ptr = m.access_chain(t_ptr_pc_u32, pc, &[constants.two]);
    let vocab = m.load(t_u32, vocab_ptr);
    let total = m.imul(t_u32, num_tokens, hidden);
    let in_bounds = m.u_less_than(t_bool, global_id, total);
    m.selection_merge(function_merge, 0);
    m.branch_conditional(in_bounds, in_bounds_body, function_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[in_bounds_body.0]));

    let token_index = m.udiv(t_u32, global_id, hidden);
    let col = m.umod(t_u32, global_id, hidden);
    let token_ptr = m.access_chain(t_ptr_sb_u32, token_ids, &[constants.zero, token_index]);
    let token_id = m.load(t_u32, token_ptr);
    let valid_token = m.u_less_than(t_bool, token_id, vocab);
    let valid_body = m.alloc_id();
    let invalid_body = m.alloc_id();
    let token_merge = m.alloc_id();
    m.selection_merge(token_merge, 0);
    m.branch_conditional(valid_token, valid_body, invalid_body);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[valid_body.0]));
    let blocks_per_row = m.udiv(t_u32, hidden, c_block_elements);
    let row_bytes = m.imul(t_u32, blocks_per_row, c_block_bytes);
    let row_base = m.imul(t_u32, token_id, row_bytes);
    let block_index = m.udiv(t_u32, col, c_block_elements);
    let intra = m.umod(t_u32, col, c_block_elements);
    let block_offset = m.imul(t_u32, block_index, c_block_bytes);
    let block_base = m.iadd(t_u32, row_base, block_offset);
    let value = decode_value(
        &mut m, quant, &types, &constants, &codebooks, weight, block_base, intra,
    )?;
    let output_ptr = m.access_chain(t_ptr_sb_f32, output, &[constants.zero, global_id]);
    m.store(output_ptr, value);
    m.branch(token_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[invalid_body.0]));
    let invalid_output_ptr = m.access_chain(t_ptr_sb_f32, output, &[constants.zero, global_id]);
    m.store(invalid_output_ptr, c_f32_0);
    m.branch(token_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[token_merge.0]));
    m.branch(function_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[function_merge.0]));
    m.ret();
    m.function_end();
    Ok(m.encode())
}
