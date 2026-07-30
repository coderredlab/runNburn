use super::builder::{builtin, decoration, op, storage_class, Id, SpirvModule};

#[derive(Clone, Copy, PartialEq, Eq)]
enum GroupedQ8Mode {
    Gate,
    UpSilu,
    Down,
}

/// Emit grouped Q8_0 gate projection over token-major hidden rows.
pub fn emit_moe_grouped_q8_gate(local_size_x: u32) -> Vec<u32> {
    emit_moe_grouped_q8(local_size_x, GroupedQ8Mode::Gate)
}

/// Emit grouped Q8_0 up projection and fuse `SiLU(gate) * up`.
pub fn emit_moe_grouped_q8_up_silu(local_size_x: u32) -> Vec<u32> {
    emit_moe_grouped_q8(local_size_x, GroupedQ8Mode::UpSilu)
}

/// Emit grouped Q8_0 down projection over route-major activations.
pub fn emit_moe_grouped_q8_down(local_size_x: u32) -> Vec<u32> {
    emit_moe_grouped_q8(local_size_x, GroupedQ8Mode::Down)
}

fn f16_bits_to_f32(
    m: &mut SpirvModule,
    t_bool: Id,
    t_u32: Id,
    t_f32: Id,
    raw: Id,
    c_u32_0: Id,
    c_u32_1: Id,
    c_u32_10: Id,
    c_u32_13: Id,
    c_u32_15: Id,
    c_u32_23: Id,
    c_u32_31: Id,
    c_u32_1f: Id,
    c_u32_3ff: Id,
    c_u32_112: Id,
    c_f32_2pow_neg24: Id,
) -> Id {
    let exp_raw = m.shift_right_logical(t_u32, raw, c_u32_10);
    let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
    let mant = m.bitwise_and(t_u32, raw, c_u32_3ff);
    let sign = m.shift_right_logical(t_u32, raw, c_u32_15);
    let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
    let sign_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
    let exp_adj = m.iadd(t_u32, exp, c_u32_112);
    let exp_part = m.shift_left_logical(t_u32, exp_adj, c_u32_23);
    let mant_part = m.shift_left_logical(t_u32, mant, c_u32_13);
    let bits_mid = m.bitwise_or(t_u32, sign_part, exp_part);
    let bits = m.bitwise_or(t_u32, bits_mid, mant_part);
    let normal = m.bitcast(t_f32, bits);
    let mant_f = m.convert_u_to_f(t_f32, mant);
    let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
    let denorm_neg = m.fnegate(t_f32, denorm_abs);
    let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
    let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
    let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
    m.select(t_f32, exp_nonzero, normal, denormal)
}

fn emit_moe_grouped_q8(local_size_x: u32, mode: GroupedQ8Mode) -> Vec<u32> {
    let mut m = SpirvModule::new();
    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = (mode == GroupedQ8Mode::UpSilu).then(|| m.ext_inst_import("GLSL.std.450"));
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
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32]);
    let t_ptr_sb_u32_struct = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_u32);
    let t_ptr_sb_f32_struct = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_f32);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_5 = m.constant_u32(t_u32, 5);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_9 = m.constant_u32(t_u32, 9);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_i32_24 = m.constant_u32(t_i32, 24);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_1 = m.constant_f32(t_f32, 1.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    m.decorate(t_struct_u32, decoration::BLOCK, &[]);
    m.decorate(t_struct_f32, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_u32, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_f32, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    for field in 0..7 {
        m.member_decorate(t_struct_pc, field, decoration::OFFSET, &[field * 4]);
    }

    let gvar_arena = m.variable(t_ptr_sb_u32_struct, storage_class::STORAGE_BUFFER);
    let gvar_slot_ids = m.variable(t_ptr_sb_u32_struct, storage_class::STORAGE_BUFFER);
    let gvar_token_ids = m.variable(t_ptr_sb_u32_struct, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_f32_struct, storage_class::STORAGE_BUFFER);
    let gvar_gate = m.variable(t_ptr_sb_f32_struct, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_f32_struct, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_global_id = m.variable(t_ptr_input_v3, storage_class::INPUT);
    for (binding, variable) in [
        gvar_arena,
        gvar_slot_ids,
        gvar_token_ids,
        gvar_input,
        gvar_gate,
        gvar_output,
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

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_global_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);
    let lbl_entry = m.alloc_id();
    let lbl_active = m.alloc_id();
    let lbl_merge = m.alloc_id();
    let lbl_block_header = m.alloc_id();
    let lbl_block_cond = m.alloc_id();
    let lbl_block_body = m.alloc_id();
    let lbl_block_continue = m.alloc_id();
    let lbl_block_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_block = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let global_id = m.load(t_v3u32, gvar_global_id);
    let row = m.composite_extract(t_u32, global_id, 0);
    let route = m.composite_extract(t_u32, global_id, 1);
    let rows_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let rows = m.load(t_u32, rows_ptr);
    let cols_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let cols = m.load(t_u32, cols_ptr);
    let route_count_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let route_count = m.load(t_u32, route_count_ptr);
    let row_active = m.u_less_than(t_bool, row, rows);
    let route_active = m.u_less_than(t_bool, route, route_count);
    let active = m.logical_and(t_bool, row_active, route_active);
    m.selection_merge(lbl_merge, 0);
    m.branch_conditional(active, lbl_active, lbl_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_active.0]));

    let stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_3]);
    let slot_stride = m.load(t_u32, stride_ptr);
    let projection_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_4]);
    let projection_offset = m.load(t_u32, projection_ptr);
    let input_stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_5]);
    let input_stride = m.load(t_u32, input_stride_ptr);
    let output_stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_6]);
    let output_stride = m.load(t_u32, output_stride_ptr);
    let slot_ptr = m.access_chain(t_ptr_sb_u32, gvar_slot_ids, &[c_u32_0, route]);
    let slot = m.load(t_u32, slot_ptr);
    let slot_base = m.imul(t_u32, slot, slot_stride);
    let projection_base = m.iadd(t_u32, slot_base, projection_offset);
    let input_row = if mode == GroupedQ8Mode::Down {
        route
    } else {
        let token_ptr = m.access_chain(t_ptr_sb_u32, gvar_token_ids, &[c_u32_0, route]);
        m.load(t_u32, token_ptr)
    };
    let input_base = m.imul(t_u32, input_row, input_stride);
    let output_base = m.imul(t_u32, route, output_stride);
    let output_index = m.iadd(t_u32, output_base, row);
    let num_blocks = m.udiv(t_u32, cols, c_u32_32);
    m.store(var_sum, c_f32_0);
    m.store(var_block, c_u32_0);
    m.branch(lbl_block_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_block_header.0]));
    m.loop_merge(lbl_block_merge, lbl_block_continue, 0);
    m.branch(lbl_block_cond);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_block_cond.0]));
    let block = m.load(t_u32, var_block);
    let block_active = m.u_less_than(t_bool, block, num_blocks);
    m.branch_conditional(block_active, lbl_block_body, lbl_block_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_block_body.0]));

    let block_x_9 = m.imul(t_u32, block, c_u32_9);
    let plane_words = m.imul(t_u32, block_x_9, rows);
    let plane_base = m.iadd(t_u32, projection_base, plane_words);
    let scale_addr = m.iadd(t_u32, plane_base, row);
    let scale_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, scale_addr]);
    let scale_word = m.load(t_u32, scale_ptr);
    let scale_raw = m.bitwise_and(t_u32, scale_word, c_u32_ffff);
    let scale = f16_bits_to_f32(
        &mut m,
        t_bool,
        t_u32,
        t_f32,
        scale_raw,
        c_u32_0,
        c_u32_1,
        c_u32_10,
        c_u32_13,
        c_u32_15,
        c_u32_23,
        c_u32_31,
        c_u32_1f,
        c_u32_3ff,
        c_u32_112,
        c_f32_2pow_neg24,
    );
    let quant_plane_base = m.iadd(t_u32, plane_base, rows);
    let block_input_offset = m.imul(t_u32, block, c_u32_32);
    let block_input_base = m.iadd(t_u32, input_base, block_input_offset);
    let mut block_sum = c_f32_0;
    for word in 0..8u32 {
        let word_plane = if word == 0 {
            quant_plane_base
        } else {
            let word_constant = m.constant_u32(t_u32, word);
            let word_rows = m.imul(t_u32, word_constant, rows);
            m.iadd(t_u32, quant_plane_base, word_rows)
        };
        let word_addr = m.iadd(t_u32, word_plane, row);
        let weight_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, word_addr]);
        let packed = m.load(t_u32, weight_ptr);
        let word_input = if word == 0 {
            block_input_base
        } else {
            let offset = m.constant_u32(t_u32, word * 4);
            m.iadd(t_u32, block_input_base, offset)
        };
        for byte in 0..4u32 {
            let raw = if byte == 0 {
                m.bitwise_and(t_u32, packed, c_u32_ff)
            } else {
                let shift = match byte {
                    1 => c_u32_8,
                    2 => c_u32_16,
                    3 => c_u32_24,
                    _ => unreachable!(),
                };
                let shifted = m.shift_right_logical(t_u32, packed, shift);
                m.bitwise_and(t_u32, shifted, c_u32_ff)
            };
            let as_i32 = m.bitcast(t_i32, raw);
            let shifted_left = m.shift_left_logical(t_i32, as_i32, c_i32_24);
            let signed = m.shift_right_arithmetic(t_i32, shifted_left, c_i32_24);
            let weight = m.convert_s_to_f(t_f32, signed);
            let input_index = if byte == 0 {
                word_input
            } else {
                let byte_constant = match byte {
                    1 => c_u32_1,
                    2 => c_u32_2,
                    3 => c_u32_3,
                    _ => unreachable!(),
                };
                m.iadd(t_u32, word_input, byte_constant)
            };
            let input_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, input_index]);
            let input = m.load(t_f32, input_ptr);
            let product = m.fmul(t_f32, weight, input);
            block_sum = m.fadd(t_f32, block_sum, product);
        }
    }
    let scaled = m.fmul(t_f32, block_sum, scale);
    let previous = m.load(t_f32, var_sum);
    let total = m.fadd(t_f32, previous, scaled);
    m.store(var_sum, total);
    m.branch(lbl_block_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_block_continue.0]));
    let next_block = m.iadd(t_u32, block, c_u32_1);
    m.store(var_block, next_block);
    m.branch(lbl_block_header);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_block_merge.0]));

    let sum = m.load(t_f32, var_sum);
    let value = if let Some(glsl) = glsl {
        let gate_ptr = m.access_chain(t_ptr_sb_f32, gvar_gate, &[c_u32_0, output_index]);
        let gate = m.load(t_f32, gate_ptr);
        let neg_gate = m.fnegate(t_f32, gate);
        let exp_neg_gate = m.ext_inst(t_f32, glsl, 27, &[neg_gate]);
        let denominator = m.fadd(t_f32, c_f32_1, exp_neg_gate);
        let silu_gate = m.fdiv(t_f32, gate, denominator);
        m.fmul(t_f32, silu_gate, sum)
    } else {
        sum
    };
    let output_var = if mode == GroupedQ8Mode::Gate {
        gvar_gate
    } else {
        gvar_output
    };
    let output_ptr = m.access_chain(t_ptr_sb_f32, output_var, &[c_u32_0, output_index]);
    m.store(output_ptr, value);
    m.branch(lbl_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_merge.0]));
    m.ret();
    m.function_end();
    m.encode()
}
