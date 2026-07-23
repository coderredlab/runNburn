use super::builder::{
    builtin, decoration, memory_semantics, op, scope, storage_class, Id, SpirvModule,
};

#[derive(Clone, Copy)]
struct Q4kBlockMetadata {
    d: Id,
    dmin: Id,
    scales: [Id; 8],
    mins: [Id; 8],
}

fn emit_f16_to_f32(m: &mut SpirvModule, t_bool: Id, t_u32: Id, t_f32: Id, raw: Id) -> Id {
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_1f = m.constant_u32(t_u32, 0x1f);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3ff);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    let exp_raw = m.shift_right_logical(t_u32, raw, c_u32_10);
    let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
    let mant = m.bitwise_and(t_u32, raw, c_u32_3ff);
    let sign = m.shift_right_logical(t_u32, raw, c_u32_15);
    let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
    let sign_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
    let exponent = m.iadd(t_u32, exp, c_u32_112);
    let exponent_part = m.shift_left_logical(t_u32, exponent, c_u32_23);
    let mantissa_part = m.shift_left_logical(t_u32, mant, c_u32_13);
    let normal_bits = m.bitwise_or(t_u32, sign_part, exponent_part);
    let normal_bits = m.bitwise_or(t_u32, normal_bits, mantissa_part);
    let normal = m.bitcast(t_f32, normal_bits);

    let mant_f32 = m.convert_u_to_f(t_f32, mant);
    let denormal_abs = m.fmul(t_f32, mant_f32, c_f32_2pow_neg24);
    let denormal_neg = m.fnegate(t_f32, denormal_abs);
    let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
    let denormal = m.select(t_f32, sign_set, denormal_neg, denormal_abs);
    let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
    m.select(t_f32, exp_nonzero, normal, denormal)
}

#[allow(clippy::too_many_arguments)]
fn emit_q4k_block_metadata(
    m: &mut SpirvModule,
    t_bool: Id,
    t_u32: Id,
    t_f32: Id,
    t_ptr_sb_u32: Id,
    weight: Id,
    plane_base: Id,
    row: Id,
    rows: Id,
) -> Q4kBlockMetadata {
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_ff = m.constant_u32(t_u32, 0xff);
    let c_u32_0f = m.constant_u32(t_u32, 0x0f);
    let c_u32_3f = m.constant_u32(t_u32, 0x3f);
    let c_u32_ffff = m.constant_u32(t_u32, 0xffff);

    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, weight, &[c_u32_0, packed_addr]);
    let packed = m.load(t_u32, packed_ptr);
    let d_raw = m.bitwise_and(t_u32, packed, c_u32_ffff);
    let dmin_raw = m.shift_right_logical(t_u32, packed, c_u32_16);
    let d = emit_f16_to_f32(m, t_bool, t_u32, t_f32, d_raw);
    let dmin = emit_f16_to_f32(m, t_bool, t_u32, t_f32, dmin_raw);

    let s0_addr = m.iadd(t_u32, plane_base, rows);
    let s0_addr = m.iadd(t_u32, s0_addr, row);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, weight, &[c_u32_0, s0_addr]);
    let s0 = m.load(t_u32, s0_ptr);

    let two_rows = m.imul(t_u32, c_u32_2, rows);
    let s1_addr = m.iadd(t_u32, plane_base, two_rows);
    let s1_addr = m.iadd(t_u32, s1_addr, row);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, weight, &[c_u32_0, s1_addr]);
    let s1 = m.load(t_u32, s1_ptr);

    let three_rows = m.imul(t_u32, c_u32_3, rows);
    let s2_addr = m.iadd(t_u32, plane_base, three_rows);
    let s2_addr = m.iadd(t_u32, s2_addr, row);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, weight, &[c_u32_0, s2_addr]);
    let s2 = m.load(t_u32, s2_ptr);

    let mut scale_bytes = [c_u32_0; 12];
    for (word_index, word) in [s0, s1, s2].into_iter().enumerate() {
        for byte_index in 0..4u32 {
            let byte = if byte_index == 0 {
                m.bitwise_and(t_u32, word, c_u32_ff)
            } else {
                let shift = m.constant_u32(t_u32, byte_index * 8);
                let shifted = m.shift_right_logical(t_u32, word, shift);
                m.bitwise_and(t_u32, shifted, c_u32_ff)
            };
            scale_bytes[word_index * 4 + byte_index as usize] = byte;
        }
    }

    let mut scales = [c_u32_0; 8];
    let mut mins = [c_u32_0; 8];
    for j in 0..4usize {
        scales[j] = m.bitwise_and(t_u32, scale_bytes[j], c_u32_3f);
        mins[j] = m.bitwise_and(t_u32, scale_bytes[j + 4], c_u32_3f);
    }
    for j in 4..8usize {
        let scale_lo = m.bitwise_and(t_u32, scale_bytes[j + 4], c_u32_0f);
        let scale_hi = m.shift_right_logical(t_u32, scale_bytes[j - 4], c_u32_6);
        let scale_hi = m.shift_left_logical(t_u32, scale_hi, c_u32_4);
        scales[j] = m.bitwise_or(t_u32, scale_lo, scale_hi);

        let min_lo = m.shift_right_logical(t_u32, scale_bytes[j + 4], c_u32_4);
        let min_hi = m.shift_right_logical(t_u32, scale_bytes[j], c_u32_6);
        let min_hi = m.shift_left_logical(t_u32, min_hi, c_u32_4);
        mins[j] = m.bitwise_or(t_u32, min_lo, min_hi);
    }

    Q4kBlockMetadata {
        d,
        dmin,
        scales,
        mins,
    }
}

/// Four-token Q4_K gate/up projection. A row invocation decodes both weights
/// while reusing the same four staged activation columns.
pub fn emit_q4k_gate_up_batch4(local_size_x: u32) -> Vec<u32> {
    debug_assert_eq!(local_size_x, 64);
    const LANES: usize = 4;

    let mut m = SpirvModule::new();
    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_v4f32 = m.type_vector(t_f32, 4);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let c_shared_len = m.constant_u32(t_u32, (LANES * 256) as u32);
    let t_shared_input = m.type_array(t_f32, c_shared_len);
    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_weight = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weight);
    let t_ptr_sb_struct_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_struct_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_ptr_wg_shared = m.type_pointer(storage_class::WORKGROUP, t_shared_input);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_36 = m.constant_u32(t_u32, 36);
    let c_u32_0f = m.constant_u32(t_u32, 0x0f);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_v4f32_0 = m.constant_composite(t_v4f32, &[c_f32_0; 4]);

    m.decorate(t_struct_weight, decoration::BLOCK, &[]);
    m.decorate(t_struct_input, decoration::BLOCK, &[]);
    m.decorate(t_struct_output, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_weight, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_input, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_output, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    for field in 0..5 {
        m.member_decorate(t_struct_pc, field, decoration::OFFSET, &[field * 4]);
    }

    let gvar_gate_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_gate_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_up_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_up_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_global_id = m.variable(t_ptr_input_u32, storage_class::INPUT);
    let gvar_local_id = m.variable(t_ptr_input_u32, storage_class::INPUT);
    let gvar_shared_input = m.variable(t_ptr_wg_shared, storage_class::WORKGROUP);

    for (binding, variable) in [
        gvar_gate_weight,
        gvar_input,
        gvar_gate_output,
        gvar_up_weight,
        gvar_up_output,
    ]
    .into_iter()
    .enumerate()
    {
        m.decorate(variable, decoration::DESCRIPTOR_SET, &[0]);
        m.decorate(variable, decoration::BINDING, &[binding as u32]);
    }
    m.decorate(
        gvar_global_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );
    m.decorate(
        gvar_local_id,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );

    let function = m.alloc_id();
    m.entry_point(5, function, "main", &[gvar_global_id, gvar_local_id]);
    m.execution_mode_local_size(function, local_size_x, 1, 1);

    let label_entry = m.alloc_id();
    let label_bounds_true = m.alloc_id();
    let label_bounds_merge = m.alloc_id();
    let label_outer_header = m.alloc_id();
    let label_outer_body = m.alloc_id();
    let label_outer_continue = m.alloc_id();
    let label_outer_merge = m.alloc_id();

    m.function(t_void, function, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_entry.0]));

    let gate_sums: Vec<_> = (0..LANES)
        .map(|_| m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION))
        .collect();
    let up_sums: Vec<_> = (0..LANES)
        .map(|_| m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION))
        .collect();
    let block = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let global_id = m.load(t_v3u32, gvar_global_id);
    let row = m.composite_extract(t_u32, global_id, 0);
    let token_group = m.composite_extract(t_u32, global_id, 1);
    let local_id = m.load(t_v3u32, gvar_local_id);
    let local_x = m.composite_extract(t_u32, local_id, 0);

    let pc_rows_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let rows = m.load(t_u32, pc_rows_ptr);
    let pc_cols_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let cols = m.load(t_u32, pc_cols_ptr);
    let pc_seq_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let seq_len = m.load(t_u32, pc_seq_ptr);
    let pc_input_stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_3]);
    let input_stride = m.load(t_u32, pc_input_stride_ptr);
    let pc_output_stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_4]);
    let output_stride = m.load(t_u32, pc_output_stride_ptr);

    let token_base = m.imul(t_u32, token_group, c_u32_4);
    let last_token = m.isub(t_u32, seq_len, c_u32_1);
    let safe_tokens: Vec<_> = (0..LANES)
        .map(|lane| {
            let lane_id = m.constant_u32(t_u32, lane as u32);
            let token = m.iadd(t_u32, token_base, lane_id);
            let valid = m.u_less_than(t_bool, token, seq_len);
            m.select(t_u32, valid, token, last_token)
        })
        .collect();

    let in_bounds = m.u_less_than(t_bool, row, rows);
    m.selection_merge(label_bounds_merge, 0);
    m.branch_conditional(in_bounds, label_bounds_true, label_bounds_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_bounds_true.0]));

    for sum in gate_sums.iter().chain(up_sums.iter()) {
        m.store(*sum, c_f32_0);
    }
    let num_blocks = m.udiv(t_u32, cols, c_u32_256);
    m.store(block, c_u32_0);
    m.branch(label_outer_header);

    let label_outer_condition = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_outer_header.0]));
    m.loop_merge(label_outer_merge, label_outer_continue, 0);
    m.branch(label_outer_condition);
    m.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_outer_condition.0],
    ));
    let block_index = m.load(t_u32, block);
    let has_block = m.u_less_than(t_bool, block_index, num_blocks);
    m.branch_conditional(has_block, label_outer_body, label_outer_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_outer_body.0]));
    let block_planes = m.imul(t_u32, block_index, c_u32_36);
    let plane_base = m.imul(t_u32, block_planes, rows);
    let gate_metadata = emit_q4k_block_metadata(
        &mut m,
        t_bool,
        t_u32,
        t_f32,
        t_ptr_sb_u32,
        gvar_gate_weight,
        plane_base,
        row,
        rows,
    );
    let up_metadata = emit_q4k_block_metadata(
        &mut m,
        t_bool,
        t_u32,
        t_f32,
        t_ptr_sb_u32,
        gvar_up_weight,
        plane_base,
        row,
        rows,
    );

    let block_input_base = m.imul(t_u32, block_index, c_u32_256);
    for (lane, safe_token) in safe_tokens.iter().copied().enumerate() {
        let lane_base = m.constant_u32(t_u32, (lane * 256) as u32);
        let token_offset = m.imul(t_u32, safe_token, input_stride);
        for chunk in 0..4u32 {
            let element = if chunk == 0 {
                local_x
            } else {
                let chunk_offset = m.constant_u32(t_u32, chunk * local_size_x);
                m.iadd(t_u32, local_x, chunk_offset)
            };
            let input_element = m.iadd(t_u32, block_input_base, element);
            let input_index = m.iadd(t_u32, token_offset, input_element);
            let input_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, input_index]);
            let input = m.load(t_f32, input_ptr);
            let shared_index = m.iadd(t_u32, lane_base, element);
            let shared_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_input, &[shared_index]);
            m.store(shared_ptr, input);
        }
    }
    let c_scope_workgroup = m.constant_u32(t_u32, scope::WORKGROUP);
    let c_workgroup_barrier = m.constant_u32(
        t_u32,
        memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
    );
    m.control_barrier(c_scope_workgroup, c_scope_workgroup, c_workgroup_barrier);

    let mut gate_block_sums = [c_f32_0; LANES];
    let mut up_block_sums = [c_f32_0; LANES];
    for subblock in 0..8u32 {
        let gate_scale = m.convert_u_to_f(t_f32, gate_metadata.scales[subblock as usize]);
        let gate_min = m.convert_u_to_f(t_f32, gate_metadata.mins[subblock as usize]);
        let up_scale = m.convert_u_to_f(t_f32, up_metadata.scales[subblock as usize]);
        let up_min = m.convert_u_to_f(t_f32, up_metadata.mins[subblock as usize]);
        let quant_group = subblock / 2;
        let high_nibble = (subblock & 1) != 0;
        let quant_plane = m.constant_u32(t_u32, 4 + quant_group * 8);
        let quant_plane_offset = m.imul(t_u32, quant_plane, rows);
        let quant_plane_start = m.iadd(t_u32, plane_base, quant_plane_offset);
        let subblock_offset = m.constant_u32(t_u32, subblock * 32);

        let mut gate_nibble_sum = c_v4f32_0;
        let mut up_nibble_sum = c_v4f32_0;
        let mut input_sum = c_v4f32_0;
        for word_index in 0..8u32 {
            let word_offset = if word_index == 0 {
                quant_plane_start
            } else {
                let offset = m.constant_u32(t_u32, word_index);
                let offset = m.imul(t_u32, offset, rows);
                m.iadd(t_u32, quant_plane_start, offset)
            };
            let quant_address = m.iadd(t_u32, word_offset, row);
            let gate_quant_ptr =
                m.access_chain(t_ptr_sb_u32, gvar_gate_weight, &[c_u32_0, quant_address]);
            let gate_quant_word = m.load(t_u32, gate_quant_ptr);
            let up_quant_ptr =
                m.access_chain(t_ptr_sb_u32, gvar_up_weight, &[c_u32_0, quant_address]);
            let up_quant_word = m.load(t_u32, up_quant_ptr);

            for byte_index in 0..4u32 {
                let shift = byte_index * 8 + if high_nibble { 4 } else { 0 };
                let gate_nibble = if shift == 0 {
                    m.bitwise_and(t_u32, gate_quant_word, c_u32_0f)
                } else {
                    let shift = m.constant_u32(t_u32, shift);
                    let shifted = m.shift_right_logical(t_u32, gate_quant_word, shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                let up_nibble = if shift == 0 {
                    m.bitwise_and(t_u32, up_quant_word, c_u32_0f)
                } else {
                    let shift = m.constant_u32(t_u32, shift);
                    let shifted = m.shift_right_logical(t_u32, up_quant_word, shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                let gate_nibble = m.convert_u_to_f(t_f32, gate_nibble);
                let gate_nibbles = m.composite_construct(t_v4f32, &[gate_nibble; LANES]);
                let up_nibble = m.convert_u_to_f(t_f32, up_nibble);
                let up_nibbles = m.composite_construct(t_v4f32, &[up_nibble; LANES]);

                let inner = m.constant_u32(t_u32, word_index * 4 + byte_index);
                let shared_element = m.iadd(t_u32, subblock_offset, inner);
                let mut input_lanes = [c_f32_0; LANES];
                for (lane, input_lane) in input_lanes.iter_mut().enumerate() {
                    let lane_base = m.constant_u32(t_u32, (lane * 256) as u32);
                    let shared_index = m.iadd(t_u32, lane_base, shared_element);
                    let shared_ptr =
                        m.access_chain(t_ptr_wg_f32, gvar_shared_input, &[shared_index]);
                    *input_lane = m.load(t_f32, shared_ptr);
                }
                let inputs = m.composite_construct(t_v4f32, &input_lanes);
                let gate_product = m.fmul(t_v4f32, gate_nibbles, inputs);
                gate_nibble_sum = m.fadd(t_v4f32, gate_nibble_sum, gate_product);
                let up_product = m.fmul(t_v4f32, up_nibbles, inputs);
                up_nibble_sum = m.fadd(t_v4f32, up_nibble_sum, up_product);
                input_sum = m.fadd(t_v4f32, input_sum, inputs);
            }
        }

        let gate_scaled_d = m.fmul(t_f32, gate_metadata.d, gate_scale);
        let gate_scaled_min = m.fmul(t_f32, gate_metadata.dmin, gate_min);
        let up_scaled_d = m.fmul(t_f32, up_metadata.d, up_scale);
        let up_scaled_min = m.fmul(t_f32, up_metadata.dmin, up_min);
        for lane in 0..LANES {
            let gate_nibble = m.composite_extract(t_f32, gate_nibble_sum, lane as u32);
            let up_nibble = m.composite_extract(t_f32, up_nibble_sum, lane as u32);
            let input = m.composite_extract(t_f32, input_sum, lane as u32);

            let gate_term = m.fmul(t_f32, gate_scaled_d, gate_nibble);
            let gate_min_term = m.fmul(t_f32, gate_scaled_min, input);
            let gate_result = m.fsub(t_f32, gate_term, gate_min_term);
            gate_block_sums[lane] = m.fadd(t_f32, gate_block_sums[lane], gate_result);

            let up_term = m.fmul(t_f32, up_scaled_d, up_nibble);
            let up_min_term = m.fmul(t_f32, up_scaled_min, input);
            let up_result = m.fsub(t_f32, up_term, up_min_term);
            up_block_sums[lane] = m.fadd(t_f32, up_block_sums[lane], up_result);
        }
    }

    for lane in 0..LANES {
        let gate_previous = m.load(t_f32, gate_sums[lane]);
        let gate_total = m.fadd(t_f32, gate_previous, gate_block_sums[lane]);
        m.store(gate_sums[lane], gate_total);
        let up_previous = m.load(t_f32, up_sums[lane]);
        let up_total = m.fadd(t_f32, up_previous, up_block_sums[lane]);
        m.store(up_sums[lane], up_total);
    }
    m.control_barrier(c_scope_workgroup, c_scope_workgroup, c_workgroup_barrier);
    m.branch(label_outer_continue);

    m.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_outer_continue.0],
    ));
    let next_block = m.iadd(t_u32, block_index, c_u32_1);
    m.store(block, next_block);
    m.branch(label_outer_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_outer_merge.0]));
    for lane in 0..LANES {
        let token_offset = m.imul(t_u32, safe_tokens[lane], output_stride);
        let output_index = m.iadd(t_u32, token_offset, row);
        let gate_output_ptr =
            m.access_chain(t_ptr_sb_f32, gvar_gate_output, &[c_u32_0, output_index]);
        let gate_result = m.load(t_f32, gate_sums[lane]);
        m.store(gate_output_ptr, gate_result);
        let up_output_ptr = m.access_chain(t_ptr_sb_f32, gvar_up_output, &[c_u32_0, output_index]);
        let up_result = m.load(t_f32, up_sums[lane]);
        m.store(up_output_ptr, up_result);
    }
    m.branch(label_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_bounds_merge.0]));
    m.ret();
    m.function_end();
    m.encode()
}
