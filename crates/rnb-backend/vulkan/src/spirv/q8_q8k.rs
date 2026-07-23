use super::builder::{builtin, decoration, op, storage_class, SpirvModule};

/// Emit a Q8_0 weight × Q8K activation integer-dot GEMV shader.
///
/// Weight rows use the backend's transposed SoA layout: each Q8_0 block has
/// one scale plane followed by eight packed-i8 planes. Activations use the
/// 69-u32 Q8K block layout produced by `emit_quantize_to_q8k`.
pub fn emit_q8_q8k_gemv(local_size_x: u32) -> Vec<u32> {
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
    let t_struct_input = m.type_struct(&[t_arr_u32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

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
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_9 = m.constant_u32(t_u32, 9);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_69 = m.constant_u32(t_u32, 69);
    let c_u32_ff = m.constant_u32(t_u32, 0xff);
    let c_u32_ffff = m.constant_u32(t_u32, 0xffff);
    let c_u32_1f = m.constant_u32(t_u32, 0x1f);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3ff);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_i32_24 = m.constant_u32(t_i32, 24);
    let c_i32_0 = m.constant_u32(t_i32, 0);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

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

    let gvar_weight = m.variable(t_ptr_sb_struct_weight, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_struct_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_struct_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);
    m.decorate(gvar_weight, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_weight, decoration::BINDING, &[0]);
    m.decorate(gvar_input, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_input, decoration::BINDING, &[1]);
    m.decorate(gvar_output, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_output, decoration::BINDING, &[2]);
    m.decorate(
        gvar_glob_id,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);
    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));
    m.store(var_sum, c_f32_0);

    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_32);
    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    let lbl_outer_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_header.0]));
    m.loop_merge(lbl_outer_merge, lbl_outer_continue, 0);
    m.branch(lbl_outer_cond);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_cond.0]));
    let blk_cur = m.load(t_u32, var_blk);
    let outer_cond = m.u_less_than(t_bool, blk_cur, num_blocks);
    m.branch_conditional(outer_cond, lbl_outer_body, lbl_outer_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    // Weight block: 9 SoA planes (scale + 8 packed i8 words).
    let block_plane = m.imul(t_u32, blk_cur, c_u32_9);
    let plane_base = m.imul(t_u32, block_plane, pc_rows);
    let scale_addr = m.iadd(t_u32, plane_base, row);
    let scale_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, scale_addr]);
    let scale_word = m.load(t_u32, scale_ptr);
    let f16_bits_raw = m.bitwise_and(t_u32, scale_word, c_u32_ffff);
    let f16_sign = m.shift_right_logical(t_u32, f16_bits_raw, c_u32_15);
    let f16_sign_bit = m.bitwise_and(t_u32, f16_sign, c_u32_1);
    let f16_exp_raw = m.shift_right_logical(t_u32, f16_bits_raw, c_u32_10);
    let f16_exp = m.bitwise_and(t_u32, f16_exp_raw, c_u32_1f);
    let f16_mant = m.bitwise_and(t_u32, f16_bits_raw, c_u32_3ff);
    let f32_sign_part = m.shift_left_logical(t_u32, f16_sign_bit, c_u32_31);
    let f32_exp_adj = m.iadd(t_u32, f16_exp, c_u32_112);
    let f32_exp_part = m.shift_left_logical(t_u32, f32_exp_adj, c_u32_23);
    let f32_mant_part = m.shift_left_logical(t_u32, f16_mant, c_u32_13);
    let f32_bits_mid = m.bitwise_or(t_u32, f32_sign_part, f32_exp_part);
    let f32_bits = m.bitwise_or(t_u32, f32_bits_mid, f32_mant_part);
    let normal_scale = m.bitcast(t_f32, f32_bits);
    let mant_f = m.convert_u_to_f(t_f32, f16_mant);
    let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
    let denorm_neg = m.fnegate(t_f32, denorm_abs);
    let sign_set = m.i_not_equal(t_bool, f16_sign_bit, c_u32_0);
    let denorm_scale = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
    let exp_nonzero = m.i_not_equal(t_bool, f16_exp, c_u32_0);
    let weight_scale = m.select(t_f32, exp_nonzero, normal_scale, denorm_scale);

    // One Q8K activation block covers eight Q8_0 weight blocks.
    let act_block = m.udiv(t_u32, blk_cur, c_u32_8);
    let act_subblock = m.umod(t_u32, blk_cur, c_u32_8);
    let act_base = m.imul(t_u32, act_block, c_u32_69);
    let act_d_addr = m.iadd(t_u32, act_base, c_u32_64);
    let act_d_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, act_d_addr]);
    let act_d_bits = m.load(t_u32, act_d_ptr);
    let act_scale = m.bitcast(t_f32, act_d_bits);
    let act_word_offset = m.imul(t_u32, act_subblock, c_u32_8);
    let act_qs_base = m.iadd(t_u32, act_base, act_word_offset);
    let weight_qs_base = m.iadd(t_u32, plane_base, pc_rows);

    let mut dot = c_i32_0;
    for word_index in 0..8u32 {
        let weight_word_base = if word_index == 0 {
            weight_qs_base
        } else {
            let c_word = m.constant_u32(t_u32, word_index);
            let word_plane = m.imul(t_u32, c_word, pc_rows);
            m.iadd(t_u32, weight_qs_base, word_plane)
        };
        let weight_addr = m.iadd(t_u32, weight_word_base, row);
        let weight_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, weight_addr]);
        let weight_word = m.load(t_u32, weight_ptr);

        let act_word_addr = if word_index == 0 {
            act_qs_base
        } else {
            let c_word = m.constant_u32(t_u32, word_index);
            m.iadd(t_u32, act_qs_base, c_word)
        };
        let act_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, act_word_addr]);
        let act_word = m.load(t_u32, act_ptr);

        for byte_index in 0..4u32 {
            let shift = byte_index * 8;
            let weight_shifted = if shift == 0 {
                weight_word
            } else {
                let c_shift = m.constant_u32(t_u32, shift);
                m.shift_right_logical(t_u32, weight_word, c_shift)
            };
            let act_shifted = if shift == 0 {
                act_word
            } else {
                let c_shift = m.constant_u32(t_u32, shift);
                m.shift_right_logical(t_u32, act_word, c_shift)
            };
            let weight_raw = m.bitwise_and(t_u32, weight_shifted, c_u32_ff);
            let act_raw = m.bitwise_and(t_u32, act_shifted, c_u32_ff);
            let weight_i32 = m.bitcast(t_i32, weight_raw);
            let act_i32 = m.bitcast(t_i32, act_raw);
            let weight_left = m.shift_left_logical(t_i32, weight_i32, c_i32_24);
            let act_left = m.shift_left_logical(t_i32, act_i32, c_i32_24);
            let weight_signed = m.shift_right_arithmetic(t_i32, weight_left, c_i32_24);
            let act_signed = m.shift_right_arithmetic(t_i32, act_left, c_i32_24);
            let product = m.imul(t_i32, weight_signed, act_signed);
            dot = m.iadd(t_i32, dot, product);
        }
    }

    let dot_f32 = m.convert_s_to_f(t_f32, dot);
    let combined_scale = m.fmul(t_f32, weight_scale, act_scale);
    let block_value = m.fmul(t_f32, combined_scale, dot_f32);
    let previous_sum = m.load(t_f32, var_sum);
    let next_sum = m.fadd(t_f32, previous_sum, block_value);
    m.store(var_sum, next_sum);
    m.branch(lbl_outer_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));
    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();
    m.encode()
}
