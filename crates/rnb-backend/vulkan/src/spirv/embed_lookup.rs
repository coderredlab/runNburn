use super::builder::{builtin, decoration, op, storage_class, Id, SpirvModule};

/// Gather and dequantize row-major Q4_K embedding rows.
pub fn emit_embed_lookup_q4k(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let _t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32_ids = m.type_runtime_array(t_u32);
    let t_arr_f32_out = m.type_runtime_array(t_f32);

    let t_struct_ids = m.type_struct(&[t_arr_u32_ids]);
    let t_struct_table = m.type_struct(&[t_arr_u32_ids]);
    let t_struct_out = m.type_struct(&[t_arr_f32_out]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_ids = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_ids);
    let t_ptr_sb_struct_table = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_table);
    let t_ptr_sb_struct_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_63 = m.constant_u32(t_u32, 63);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);
    let c_u32_144 = m.constant_u32(t_u32, 144);
    let c_u32_256 = m.constant_u32(t_u32, 256);

    m.decorate(t_struct_ids, decoration::BLOCK, &[]);
    m.decorate(t_struct_table, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_ids, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_table, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32_ids, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32_out, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_ids = m.variable(t_ptr_sb_struct_ids, storage_class::STORAGE_BUFFER);
    let gvar_table = m.variable(t_ptr_sb_struct_table, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_struct_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_ids, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_ids, decoration::BINDING, &[0]);
    m.decorate(gvar_table, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_table, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_ptr_nt = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let num_tokens = m.load(t_u32, pc_ptr_nt);
    let pc_ptr_h = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let hidden = m.load(t_u32, pc_ptr_h);

    let total = m.imul(t_u32, num_tokens, hidden);
    let in_bounds = m.u_less_than(t_bool, gid, total);
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    let token_idx = m.udiv(t_u32, gid, hidden);
    let in_row_idx = m.umod(t_u32, gid, hidden);

    let tok_ptr = m.access_chain(t_ptr_sb_u32, gvar_ids, &[c_u32_0, token_idx]);
    let token_id = m.load(t_u32, tok_ptr);

    let pc_ptr_vocab = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let vocab = m.load(t_u32, pc_ptr_vocab);
    let in_vocab = m.u_less_than(t_bool, token_id, vocab);

    let lbl_valid = m.alloc_id();
    let lbl_oob = m.alloc_id();
    let lbl_body_end = m.alloc_id();
    m.selection_merge(lbl_body_end, 0);
    m.branch_conditional(in_vocab, lbl_valid, lbl_oob);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_valid.0]));

    let blocks_per_row = m.udiv(t_u32, hidden, c_u32_256);
    let block_idx_in_row = m.udiv(t_u32, in_row_idx, c_u32_256);
    let elem_in_block = m.umod(t_u32, in_row_idx, c_u32_256);
    let row_block_count_x_id = m.imul(t_u32, token_id, blocks_per_row);
    let global_block_idx = m.iadd(t_u32, row_block_count_x_id, block_idx_in_row);
    let block_byte_off = m.imul(t_u32, global_block_idx, c_u32_144);
    let group = m.udiv(t_u32, elem_in_block, c_u32_64);
    let elem_in_group = m.umod(t_u32, elem_in_block, c_u32_64);
    let high_nibble = m.udiv(t_u32, elem_in_group, c_u32_32);
    let lane = m.umod(t_u32, elem_in_group, c_u32_32);
    let scale_idx_base = m.imul(t_u32, group, c_u32_2);
    let scale_idx = m.iadd(t_u32, scale_idx_base, high_nibble);
    let read_byte = |m: &mut SpirvModule, byte_off: Id| -> Id {
        let word_idx = m.shift_right_logical(t_u32, byte_off, c_u32_2);
        let byte_in_word = m.bitwise_and(t_u32, byte_off, c_u32_3);
        let shift_bits = m.shift_left_logical(t_u32, byte_in_word, c_u32_3);
        let word_ptr = m.access_chain(t_ptr_sb_u32, gvar_table, &[c_u32_0, word_idx]);
        let word = m.load(t_u32, word_ptr);
        let shifted = m.shift_right_logical(t_u32, word, shift_bits);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };
    let low_scale_off_in_block = m.iadd(t_u32, c_u32_4, scale_idx);
    let low_min_off_in_block = m.iadd(t_u32, c_u32_8, scale_idx);
    let packed_off_in_block = m.iadd(t_u32, c_u32_8, scale_idx);
    let scale_upper_off_in_block = scale_idx;
    let min_upper_off_in_block = m.iadd(t_u32, c_u32_4, scale_idx);
    let low_scale_off = m.iadd(t_u32, block_byte_off, low_scale_off_in_block);
    let low_min_off = m.iadd(t_u32, block_byte_off, low_min_off_in_block);
    let packed_off = m.iadd(t_u32, block_byte_off, packed_off_in_block);
    let scale_upper_off = m.iadd(t_u32, block_byte_off, scale_upper_off_in_block);
    let min_upper_off = m.iadd(t_u32, block_byte_off, min_upper_off_in_block);
    let low_scale_byte = read_byte(&mut m, low_scale_off);
    let low_min_byte = read_byte(&mut m, low_min_off);
    let packed_byte = read_byte(&mut m, packed_off);
    let scale_upper_byte = read_byte(&mut m, scale_upper_off);
    let min_upper_byte = read_byte(&mut m, min_upper_off);
    let scale_low = m.bitwise_and(t_u32, low_scale_byte, c_u32_63);
    let min_low = m.bitwise_and(t_u32, low_min_byte, c_u32_63);
    let scale_packed_low = m.bitwise_and(t_u32, packed_byte, c_u32_0f);
    let scale_upper_bits = m.shift_right_logical(t_u32, scale_upper_byte, c_u32_6);
    let scale_upper = m.shift_left_logical(t_u32, scale_upper_bits, c_u32_4);
    let scale_high = m.bitwise_or(t_u32, scale_packed_low, scale_upper);
    let min_packed_low = m.shift_right_logical(t_u32, packed_byte, c_u32_4);
    let min_upper_bits = m.shift_right_logical(t_u32, min_upper_byte, c_u32_6);
    let min_upper = m.shift_left_logical(t_u32, min_upper_bits, c_u32_4);
    let min_high = m.bitwise_or(t_u32, min_packed_low, min_upper);
    let low_scale_layout = m.u_less_than(t_bool, scale_idx, c_u32_4);
    let scale_u32 = m.select(t_u32, low_scale_layout, scale_low, scale_high);
    let min_u32 = m.select(t_u32, low_scale_layout, min_low, min_high);
    let scale_f = m.convert_u_to_f(t_f32, scale_u32);
    let min_f = m.convert_u_to_f(t_f32, min_u32);
    let qs_group_off = m.imul(t_u32, group, c_u32_32);
    let qs_idx = m.iadd(t_u32, qs_group_off, lane);
    let qs_off_in_block = m.iadd(t_u32, c_u32_16, qs_idx);
    let qs_off = m.iadd(t_u32, block_byte_off, qs_off_in_block);
    let qs_byte = read_byte(&mut m, qs_off);
    let nibble_shift = m.shift_left_logical(t_u32, high_nibble, c_u32_2);
    let qs_shifted = m.shift_right_logical(t_u32, qs_byte, nibble_shift);
    let quant = m.bitwise_and(t_u32, qs_shifted, c_u32_0f);
    let quant_f = m.convert_u_to_f(t_f32, quant);
    let d_lo = read_byte(&mut m, block_byte_off);
    let d_hi_off = m.iadd(t_u32, block_byte_off, c_u32_1);
    let d_hi = read_byte(&mut m, d_hi_off);
    let dmin_lo_off = m.iadd(t_u32, block_byte_off, c_u32_2);
    let dmin_hi_off = m.iadd(t_u32, block_byte_off, c_u32_3);
    let dmin_lo = read_byte(&mut m, dmin_lo_off);
    let dmin_hi = read_byte(&mut m, dmin_hi_off);
    let d_hi_sh8 = m.shift_left_logical(t_u32, d_hi, c_u32_8);
    let d_raw = m.bitwise_or(t_u32, d_lo, d_hi_sh8);
    let dmin_hi_sh8 = m.shift_left_logical(t_u32, dmin_hi, c_u32_8);
    let dmin_raw = m.bitwise_or(t_u32, dmin_lo, dmin_hi_sh8);
    let f16_to_f32 = |m: &mut SpirvModule, raw: Id| -> Id {
        let exp_raw = m.shift_right_logical(t_u32, raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };
    let d = f16_to_f32(&mut m, d_raw);
    let dmin = f16_to_f32(&mut m, dmin_raw);
    let d_scale = m.fmul(t_f32, d, scale_f);
    let scaled_quant = m.fmul(t_f32, d_scale, quant_f);
    let dmin_min = m.fmul(t_f32, dmin, min_f);
    let val = m.fsub(t_f32, scaled_quant, dmin_min);

    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr, val);

    m.branch(lbl_body_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_oob.0]));
    let out_ptr_oob = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr_oob, c_f32_0);
    m.branch(lbl_body_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body_end.0]));
    m.branch(lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Gather and dequantize row-major Q5_K embedding rows.
pub fn emit_embed_lookup_q5k(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let _t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32_ids = m.type_runtime_array(t_u32);
    let t_arr_f32_out = m.type_runtime_array(t_f32);

    let t_struct_ids = m.type_struct(&[t_arr_u32_ids]);
    let t_struct_table = m.type_struct(&[t_arr_u32_ids]);
    let t_struct_out = m.type_struct(&[t_arr_f32_out]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_ids = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_ids);
    let t_ptr_sb_struct_table = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_table);
    let t_ptr_sb_struct_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_63 = m.constant_u32(t_u32, 63);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);
    let c_u32_48 = m.constant_u32(t_u32, 48);
    let c_u32_176 = m.constant_u32(t_u32, 176);
    let c_u32_256 = m.constant_u32(t_u32, 256);

    m.decorate(t_struct_ids, decoration::BLOCK, &[]);
    m.decorate(t_struct_table, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_ids, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_table, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32_ids, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32_out, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_ids = m.variable(t_ptr_sb_struct_ids, storage_class::STORAGE_BUFFER);
    let gvar_table = m.variable(t_ptr_sb_struct_table, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_struct_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_ids, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_ids, decoration::BINDING, &[0]);
    m.decorate(gvar_table, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_table, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_ptr_nt = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let num_tokens = m.load(t_u32, pc_ptr_nt);
    let pc_ptr_h = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let hidden = m.load(t_u32, pc_ptr_h);

    let total = m.imul(t_u32, num_tokens, hidden);
    let in_bounds = m.u_less_than(t_bool, gid, total);
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    let token_idx = m.udiv(t_u32, gid, hidden);
    let in_row_idx = m.umod(t_u32, gid, hidden);

    let tok_ptr = m.access_chain(t_ptr_sb_u32, gvar_ids, &[c_u32_0, token_idx]);
    let token_id = m.load(t_u32, tok_ptr);

    let pc_ptr_vocab = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let vocab = m.load(t_u32, pc_ptr_vocab);
    let in_vocab = m.u_less_than(t_bool, token_id, vocab);

    let lbl_valid = m.alloc_id();
    let lbl_oob = m.alloc_id();
    let lbl_body_end = m.alloc_id();
    m.selection_merge(lbl_body_end, 0);
    m.branch_conditional(in_vocab, lbl_valid, lbl_oob);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_valid.0]));

    let blocks_per_row = m.udiv(t_u32, hidden, c_u32_256);
    let block_idx_in_row = m.udiv(t_u32, in_row_idx, c_u32_256);
    let elem_in_block = m.umod(t_u32, in_row_idx, c_u32_256);
    let row_block_count_x_id = m.imul(t_u32, token_id, blocks_per_row);
    let global_block_idx = m.iadd(t_u32, row_block_count_x_id, block_idx_in_row);
    let block_byte_off = m.imul(t_u32, global_block_idx, c_u32_176);
    let group = m.udiv(t_u32, elem_in_block, c_u32_64);
    let elem_in_group = m.umod(t_u32, elem_in_block, c_u32_64);
    let high_nibble = m.udiv(t_u32, elem_in_group, c_u32_32);
    let lane = m.umod(t_u32, elem_in_group, c_u32_32);
    let scale_idx_base = m.imul(t_u32, group, c_u32_2);
    let scale_idx = m.iadd(t_u32, scale_idx_base, high_nibble);
    let read_byte = |m: &mut SpirvModule, byte_off: Id| -> Id {
        let word_idx = m.shift_right_logical(t_u32, byte_off, c_u32_2);
        let byte_in_word = m.bitwise_and(t_u32, byte_off, c_u32_3);
        let shift_bits = m.shift_left_logical(t_u32, byte_in_word, c_u32_3);
        let word_ptr = m.access_chain(t_ptr_sb_u32, gvar_table, &[c_u32_0, word_idx]);
        let word = m.load(t_u32, word_ptr);
        let shifted = m.shift_right_logical(t_u32, word, shift_bits);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };
    let low_scale_off_in_block = m.iadd(t_u32, c_u32_4, scale_idx);
    let low_min_off_in_block = m.iadd(t_u32, c_u32_8, scale_idx);
    let packed_off_in_block = m.iadd(t_u32, c_u32_8, scale_idx);
    let scale_upper_off_in_block = scale_idx;
    let min_upper_off_in_block = m.iadd(t_u32, c_u32_4, scale_idx);
    let low_scale_off = m.iadd(t_u32, block_byte_off, low_scale_off_in_block);
    let low_min_off = m.iadd(t_u32, block_byte_off, low_min_off_in_block);
    let packed_off = m.iadd(t_u32, block_byte_off, packed_off_in_block);
    let scale_upper_off = m.iadd(t_u32, block_byte_off, scale_upper_off_in_block);
    let min_upper_off = m.iadd(t_u32, block_byte_off, min_upper_off_in_block);
    let low_scale_byte = read_byte(&mut m, low_scale_off);
    let low_min_byte = read_byte(&mut m, low_min_off);
    let packed_byte = read_byte(&mut m, packed_off);
    let scale_upper_byte = read_byte(&mut m, scale_upper_off);
    let min_upper_byte = read_byte(&mut m, min_upper_off);
    let scale_low = m.bitwise_and(t_u32, low_scale_byte, c_u32_63);
    let min_low = m.bitwise_and(t_u32, low_min_byte, c_u32_63);
    let scale_packed_low = m.bitwise_and(t_u32, packed_byte, c_u32_0f);
    let scale_upper_bits = m.shift_right_logical(t_u32, scale_upper_byte, c_u32_6);
    let scale_upper = m.shift_left_logical(t_u32, scale_upper_bits, c_u32_4);
    let scale_high = m.bitwise_or(t_u32, scale_packed_low, scale_upper);
    let min_packed_low = m.shift_right_logical(t_u32, packed_byte, c_u32_4);
    let min_upper_bits = m.shift_right_logical(t_u32, min_upper_byte, c_u32_6);
    let min_upper = m.shift_left_logical(t_u32, min_upper_bits, c_u32_4);
    let min_high = m.bitwise_or(t_u32, min_packed_low, min_upper);
    let low_scale_layout = m.u_less_than(t_bool, scale_idx, c_u32_4);
    let scale_u32 = m.select(t_u32, low_scale_layout, scale_low, scale_high);
    let min_u32 = m.select(t_u32, low_scale_layout, min_low, min_high);
    let scale_f = m.convert_u_to_f(t_f32, scale_u32);
    let min_f = m.convert_u_to_f(t_f32, min_u32);
    let qs_group_off = m.imul(t_u32, group, c_u32_32);
    let qs_idx = m.iadd(t_u32, qs_group_off, lane);
    let qs_off_in_block = m.iadd(t_u32, c_u32_48, qs_idx);
    let qs_off = m.iadd(t_u32, block_byte_off, qs_off_in_block);
    let qs_byte = read_byte(&mut m, qs_off);
    let nibble_shift = m.shift_left_logical(t_u32, high_nibble, c_u32_2);
    let qs_shifted = m.shift_right_logical(t_u32, qs_byte, nibble_shift);
    let low_quant = m.bitwise_and(t_u32, qs_shifted, c_u32_0f);
    let qh_off_in_block = m.iadd(t_u32, c_u32_16, lane);
    let qh_off = m.iadd(t_u32, block_byte_off, qh_off_in_block);
    let qh_byte = read_byte(&mut m, qh_off);
    let group_bit_base = m.imul(t_u32, group, c_u32_2);
    let qh_bit = m.iadd(t_u32, group_bit_base, high_nibble);
    let qh_shifted = m.shift_right_logical(t_u32, qh_byte, qh_bit);
    let high_bit = m.bitwise_and(t_u32, qh_shifted, c_u32_1);
    let high_quant = m.shift_left_logical(t_u32, high_bit, c_u32_4);
    let quant = m.bitwise_or(t_u32, low_quant, high_quant);
    let quant_f = m.convert_u_to_f(t_f32, quant);
    let d_lo = read_byte(&mut m, block_byte_off);
    let d_hi_off = m.iadd(t_u32, block_byte_off, c_u32_1);
    let d_hi = read_byte(&mut m, d_hi_off);
    let dmin_lo_off = m.iadd(t_u32, block_byte_off, c_u32_2);
    let dmin_hi_off = m.iadd(t_u32, block_byte_off, c_u32_3);
    let dmin_lo = read_byte(&mut m, dmin_lo_off);
    let dmin_hi = read_byte(&mut m, dmin_hi_off);
    let d_hi_sh8 = m.shift_left_logical(t_u32, d_hi, c_u32_8);
    let d_raw = m.bitwise_or(t_u32, d_lo, d_hi_sh8);
    let dmin_hi_sh8 = m.shift_left_logical(t_u32, dmin_hi, c_u32_8);
    let dmin_raw = m.bitwise_or(t_u32, dmin_lo, dmin_hi_sh8);
    let f16_to_f32 = |m: &mut SpirvModule, raw: Id| -> Id {
        let exp_raw = m.shift_right_logical(t_u32, raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };
    let d = f16_to_f32(&mut m, d_raw);
    let dmin = f16_to_f32(&mut m, dmin_raw);
    let d_scale = m.fmul(t_f32, d, scale_f);
    let scaled_quant = m.fmul(t_f32, d_scale, quant_f);
    let dmin_min = m.fmul(t_f32, dmin, min_f);
    let val = m.fsub(t_f32, scaled_quant, dmin_min);

    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr, val);

    m.branch(lbl_body_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_oob.0]));
    let out_ptr_oob = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr_oob, c_f32_0);
    m.branch(lbl_body_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body_end.0]));
    m.branch(lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}

/// Gather and dequantize row-major Q8_0 embedding rows.
pub fn emit_embed_lookup_q8_0(local_size_x: u32) -> Vec<u32> {
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

    let t_arr_u32_ids = m.type_runtime_array(t_u32);
    let t_arr_f32_out = m.type_runtime_array(t_f32);

    let t_struct_ids = m.type_struct(&[t_arr_u32_ids]);
    let t_struct_table = m.type_struct(&[t_arr_u32_ids]);
    let t_struct_out = m.type_struct(&[t_arr_f32_out]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_struct_ids = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_ids);
    let t_ptr_sb_struct_table = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_table);
    let t_ptr_sb_struct_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);
    let c_u32_34 = m.constant_u32(t_u32, 34);

    m.decorate(t_struct_ids, decoration::BLOCK, &[]);
    m.decorate(t_struct_table, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_ids, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_table, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32_ids, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32_out, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_ids = m.variable(t_ptr_sb_struct_ids, storage_class::STORAGE_BUFFER);
    let gvar_table = m.variable(t_ptr_sb_struct_table, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_struct_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_gid = m.variable(t_ptr_input_v3, storage_class::INPUT);

    m.decorate(gvar_ids, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_ids, decoration::BINDING, &[0]);
    m.decorate(gvar_table, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_table, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_gid,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_gid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    let lbl_body = m.alloc_id();
    let lbl_end = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let gid_vec = m.load(t_v3u32, gvar_gid);
    let gid = m.composite_extract(t_u32, gid_vec, 0);

    let pc_ptr_nt = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let num_tokens = m.load(t_u32, pc_ptr_nt);
    let pc_ptr_h = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let hidden = m.load(t_u32, pc_ptr_h);

    let total = m.imul(t_u32, num_tokens, hidden);
    let in_bounds = m.u_less_than(t_bool, gid, total);
    m.selection_merge(lbl_end, 0);
    m.branch_conditional(in_bounds, lbl_body, lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body.0]));

    let token_idx = m.udiv(t_u32, gid, hidden);
    let in_row_idx = m.umod(t_u32, gid, hidden);

    let tok_ptr = m.access_chain(t_ptr_sb_u32, gvar_ids, &[c_u32_0, token_idx]);
    let token_id = m.load(t_u32, tok_ptr);

    let pc_ptr_vocab = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let vocab = m.load(t_u32, pc_ptr_vocab);
    let in_vocab = m.u_less_than(t_bool, token_id, vocab);

    let lbl_valid = m.alloc_id();
    let lbl_oob = m.alloc_id();
    let lbl_body_end = m.alloc_id();
    m.selection_merge(lbl_body_end, 0);
    m.branch_conditional(in_vocab, lbl_valid, lbl_oob);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_valid.0]));

    let blocks_per_row = m.udiv(t_u32, hidden, c_u32_32);
    let block_idx_in_row = m.udiv(t_u32, in_row_idx, c_u32_32);
    let elem_in_block = m.umod(t_u32, in_row_idx, c_u32_32);
    let row_block_count_x_id = m.imul(t_u32, token_id, blocks_per_row);
    let global_block_idx = m.iadd(t_u32, row_block_count_x_id, block_idx_in_row);
    let block_byte_off = m.imul(t_u32, global_block_idx, c_u32_34);
    let read_byte = |m: &mut SpirvModule, byte_off: Id| -> Id {
        let word_idx = m.shift_right_logical(t_u32, byte_off, c_u32_2);
        let byte_in_word = m.bitwise_and(t_u32, byte_off, c_u32_3);
        let shift_bits = m.shift_left_logical(t_u32, byte_in_word, c_u32_3);
        let word_ptr = m.access_chain(t_ptr_sb_u32, gvar_table, &[c_u32_0, word_idx]);
        let word = m.load(t_u32, word_ptr);
        let shifted = m.shift_right_logical(t_u32, word, shift_bits);
        m.bitwise_and(t_u32, shifted, c_u32_ff)
    };
    let d_lo = read_byte(&mut m, block_byte_off);
    let d_hi_off = m.iadd(t_u32, block_byte_off, c_u32_1);
    let d_hi = read_byte(&mut m, d_hi_off);
    let quant_off_base = m.iadd(t_u32, block_byte_off, c_u32_2);
    let quant_off = m.iadd(t_u32, quant_off_base, elem_in_block);
    let quant_byte = read_byte(&mut m, quant_off);
    let quant_shl24 = m.shift_left_logical(t_u32, quant_byte, c_u32_24);
    let quant_shl24_i32 = m.bitcast(t_i32, quant_shl24);
    let quant_i32 = m.shift_right_arithmetic(t_i32, quant_shl24_i32, c_u32_24);
    let quant_f = m.convert_s_to_f(t_f32, quant_i32);
    let d_hi_sh8 = m.shift_left_logical(t_u32, d_hi, c_u32_8);
    let d_raw = m.bitwise_or(t_u32, d_lo, d_hi_sh8);
    let f16_to_f32 = |m: &mut SpirvModule, raw: Id| -> Id {
        let exp_raw = m.shift_right_logical(t_u32, raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, raw, c_u32_15);
        let sign_bit = m.bitwise_and(t_u32, sign, c_u32_1);
        let s_part = m.shift_left_logical(t_u32, sign_bit, c_u32_31);
        let e_adj = m.iadd(t_u32, exp, c_u32_112);
        let e_part = m.shift_left_logical(t_u32, e_adj, c_u32_23);
        let m_part = m.shift_left_logical(t_u32, mant, c_u32_13);
        let bits_mid = m.bitwise_or(t_u32, s_part, e_part);
        let bits = m.bitwise_or(t_u32, bits_mid, m_part);
        let normal = m.bitcast(t_f32, bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denorm_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denorm_neg = m.fnegate(t_f32, denorm_abs);
        let sign_set = m.i_not_equal(t_bool, sign_bit, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denorm_neg, denorm_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };
    let d = f16_to_f32(&mut m, d_raw);
    let val = m.fmul(t_f32, d, quant_f);

    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr, val);

    m.branch(lbl_body_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_oob.0]));
    let out_ptr_oob = m.access_chain(t_ptr_sb_f32, gvar_out, &[c_u32_0, gid]);
    m.store(out_ptr_oob, c_f32_0);
    m.branch(lbl_body_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_body_end.0]));
    m.branch(lbl_end);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_end.0]));
    m.ret();
    m.function_end();

    m.encode()
}
