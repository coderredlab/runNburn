use super::builder::{
    builtin, decoration, memory_semantics, op, scope, storage_class, Id, SpirvModule,
};

/// Fused row-major Q4_K output projection and argmax reduction.
///
/// Bindings and push constants match the Q6_K/Q8_0 argmax kernels.
pub fn emit_logit_argmax_q4k(local_size_x: u32) -> Vec<u32> {
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

    let t_arr_f32_in = m.type_runtime_array(t_f32);
    let t_arr_u32_table = m.type_runtime_array(t_u32);
    let t_arr_u32_out = m.type_runtime_array(t_u32);

    let t_struct_in = m.type_struct(&[t_arr_f32_in]);
    let t_struct_table = m.type_struct(&[t_arr_u32_table]);
    let t_struct_out = m.type_struct(&[t_arr_u32_out]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32]);

    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_arr_f32 = m.type_array(t_f32, c_local_size);
    let t_shared_arr_u32 = m.type_array(t_u32, c_local_size);
    let t_ptr_wg_arr_f32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_f32);
    let t_ptr_wg_arr_u32 = m.type_pointer(storage_class::WORKGROUP, t_shared_arr_u32);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_ptr_wg_u32 = m.type_pointer(storage_class::WORKGROUP, t_u32);

    let t_ptr_sb_in = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_in);
    let t_ptr_sb_table = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_table);
    let t_ptr_sb_out = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_out);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
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
    let c_u32_144 = m.constant_u32(t_u32, 144);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_neg_inf = m.constant_f32(t_f32, f32::NEG_INFINITY);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    m.decorate(t_struct_in, decoration::BLOCK, &[]);
    m.decorate(t_struct_table, decoration::BLOCK, &[]);
    m.decorate(t_struct_out, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_in, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_table, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_out, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_f32_in, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_u32_table, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_u32_out, decoration::ARRAY_STRIDE, &[4]);

    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);

    let gvar_in = m.variable(t_ptr_sb_in, storage_class::STORAGE_BUFFER);
    let gvar_table = m.variable(t_ptr_sb_table, storage_class::STORAGE_BUFFER);
    let gvar_out = m.variable(t_ptr_sb_out, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_lid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_shared_vals = m.variable(t_ptr_wg_arr_f32, storage_class::WORKGROUP);
    let gvar_shared_idxs = m.variable(t_ptr_wg_arr_u32, storage_class::WORKGROUP);

    m.decorate(gvar_in, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_in, decoration::BINDING, &[0]);
    m.decorate(gvar_table, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_table, decoration::BINDING, &[1]);
    m.decorate(gvar_out, decoration::DESCRIPTOR_SET, &[0]);
    m.decorate(gvar_out, decoration::BINDING, &[2]);
    m.decorate(
        gvar_lid,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_lid]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    m.function(t_void, func_id, 0, t_fn_void);
    let lbl_entry = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    // Function variables (must be declared at top of first block)
    let var_v = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_h = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_val = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_best_idx = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_step = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    let lid_vec = m.load(t_v3u32, gvar_lid);
    let lid = m.composite_extract(t_u32, lid_vec, 0);

    let pc_vocab_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let vocab = m.load(t_u32, pc_vocab_ptr);
    let pc_hidden_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let hidden = m.load(t_u32, pc_hidden_ptr);
    let blocks_per_row = m.udiv(t_u32, hidden, c_u32_256);

    m.store(var_best_val, c_f32_neg_inf);
    m.store(var_best_idx, c_u32_0);
    m.store(var_v, lid);

    // ---- Outer loop: stride over vocab rows ----
    let lbl_v_h = m.alloc_id();
    let lbl_v_c = m.alloc_id();
    let lbl_v_b = m.alloc_id();
    let lbl_v_cont = m.alloc_id();
    let lbl_v_m = m.alloc_id();

    m.branch(lbl_v_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_h.0]));
    m.loop_merge(lbl_v_m, lbl_v_cont, 0);
    m.branch(lbl_v_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_c.0]));
    let cur_v = m.load(t_u32, var_v);
    let v_in_bounds = m.u_less_than(t_bool, cur_v, vocab);
    m.branch_conditional(v_in_bounds, lbl_v_b, lbl_v_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_b.0]));
    m.store(var_sum, c_f32_0);
    m.store(var_h, c_u32_0);

    // ---- Inner loop: hidden dot product with Q6_K dequant per element ----
    let lbl_h_h = m.alloc_id();
    let lbl_h_c = m.alloc_id();
    let lbl_h_b = m.alloc_id();
    let lbl_h_cont = m.alloc_id();
    let lbl_h_m = m.alloc_id();

    m.branch(lbl_h_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_h.0]));
    m.loop_merge(lbl_h_m, lbl_h_cont, 0);
    m.branch(lbl_h_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_c.0]));
    let cur_h = m.load(t_u32, var_h);
    let h_in_bounds = m.u_less_than(t_bool, cur_h, hidden);
    m.branch_conditional(h_in_bounds, lbl_h_b, lbl_h_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_b.0]));

    // Q4_K dequant for (v=cur_v, elem_in_row=cur_h).
    let block_idx_in_row = m.udiv(t_u32, cur_h, c_u32_256);
    let elem_in_block = m.umod(t_u32, cur_h, c_u32_256);
    let row_block_count_x_v = m.imul(t_u32, cur_v, blocks_per_row);
    let global_block_idx = m.iadd(t_u32, row_block_count_x_v, block_idx_in_row);
    let block_byte_off = m.imul(t_u32, global_block_idx, c_u32_144);

    let group = m.udiv(t_u32, elem_in_block, c_u32_64);
    let elem_in_group = m.umod(t_u32, elem_in_block, c_u32_64);
    let high_nibble = m.udiv(t_u32, elem_in_group, c_u32_32);
    let lane = m.umod(t_u32, elem_in_group, c_u32_32);
    let scale_idx_base = m.imul(t_u32, group, c_u32_2);
    let scale_idx = m.iadd(t_u32, scale_idx_base, high_nibble);
    let qs_group_off = m.imul(t_u32, group, c_u32_32);
    let qs_idx = m.iadd(t_u32, qs_group_off, lane);

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
    let dmin_lo_off = m.iadd(t_u32, block_byte_off, c_u32_2);
    let dmin_hi_off = m.iadd(t_u32, block_byte_off, c_u32_3);
    let dmin_lo = read_byte(&mut m, dmin_lo_off);
    let dmin_hi = read_byte(&mut m, dmin_hi_off);

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

    let qs_off_in_block = m.iadd(t_u32, c_u32_16, qs_idx);
    let qs_off = m.iadd(t_u32, block_byte_off, qs_off_in_block);
    let qs_byte = read_byte(&mut m, qs_off);
    let nibble_shift = m.shift_left_logical(t_u32, high_nibble, c_u32_2);
    let qs_shifted = m.shift_right_logical(t_u32, qs_byte, nibble_shift);
    let quant = m.bitwise_and(t_u32, qs_shifted, c_u32_0f);
    let quant_f = m.convert_u_to_f(t_f32, quant);

    let d_hi_sh8 = m.shift_left_logical(t_u32, d_hi, c_u32_8);
    let d_f16_raw = m.bitwise_or(t_u32, d_lo, d_hi_sh8);
    let dmin_hi_sh8 = m.shift_left_logical(t_u32, dmin_hi, c_u32_8);
    let dmin_f16_raw = m.bitwise_or(t_u32, dmin_lo, dmin_hi_sh8);
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
    let d_f32 = f16_to_f32(&mut m, d_f16_raw);
    let dmin_f32 = f16_to_f32(&mut m, dmin_f16_raw);
    let d_scale = m.fmul(t_f32, d_f32, scale_f);
    let scaled_quant = m.fmul(t_f32, d_scale, quant_f);
    let dmin_min = m.fmul(t_f32, dmin_f32, min_f);
    let weight_val = m.fsub(t_f32, scaled_quant, dmin_min);

    let hv_ptr = m.access_chain(t_ptr_sb_f32, gvar_in, &[c_u32_0, cur_h]);
    let hv = m.load(t_f32, hv_ptr);
    let prod = m.fmul(t_f32, weight_val, hv);
    let cur_sum = m.load(t_f32, var_sum);
    let new_sum = m.fadd(t_f32, cur_sum, prod);
    m.store(var_sum, new_sum);

    m.branch(lbl_h_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_cont.0]));
    let next_h = m.iadd(t_u32, cur_h, c_u32_1);
    m.store(var_h, next_h);
    m.branch(lbl_h_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_h_m.0]));

    // Update local best (val, idx) if sum > best_val
    let final_sum = m.load(t_f32, var_sum);
    let cur_best = m.load(t_f32, var_best_val);
    let beats = m.f_ord_greater_than(t_bool, final_sum, cur_best);
    let lbl_upd = m.alloc_id();
    let lbl_upd_m = m.alloc_id();
    m.selection_merge(lbl_upd_m, 0);
    m.branch_conditional(beats, lbl_upd, lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd.0]));
    m.store(var_best_val, final_sum);
    m.store(var_best_idx, cur_v);
    m.branch(lbl_upd_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_upd_m.0]));

    m.branch(lbl_v_cont);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_cont.0]));
    let next_v = m.iadd(t_u32, cur_v, c_local_size);
    m.store(var_v, next_v);
    m.branch(lbl_v_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_v_m.0]));

    // ---- Stash this thread's local best into shared memory ----
    let final_best_val = m.load(t_f32, var_best_val);
    let final_best_idx = m.load(t_u32, var_best_idx);
    let sv_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let si_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    m.store(sv_ptr, final_best_val);
    m.store(si_ptr, final_best_idx);
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }

    // ---- Pairwise reduction (paired val + idx) ----
    let c_half = m.constant_u32(t_u32, local_size_x / 2);
    m.store(var_step, c_half);

    let lbl_r_h = m.alloc_id();
    let lbl_r_c = m.alloc_id();
    let lbl_r_b = m.alloc_id();
    let lbl_r_cont = m.alloc_id();
    let lbl_r_m = m.alloc_id();

    m.branch(lbl_r_h);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_h.0]));
    m.loop_merge(lbl_r_m, lbl_r_cont, 0);
    m.branch(lbl_r_c);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_c.0]));
    let step_v = m.load(t_u32, var_step);
    let step_pos = m.u_less_than(t_bool, c_u32_0, step_v);
    m.branch_conditional(step_pos, lbl_r_b, lbl_r_m);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_b.0]));
    let lid_lt_step = m.u_less_than(t_bool, lid, step_v);
    let lbl_r_a = m.alloc_id();
    let lbl_r_am = m.alloc_id();
    m.selection_merge(lbl_r_am, 0);
    m.branch_conditional(lid_lt_step, lbl_r_a, lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_a.0]));
    let other_lid = m.iadd(t_u32, lid, step_v);
    let self_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[lid]);
    let self_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[lid]);
    let other_v_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared_vals, &[other_lid]);
    let other_i_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[other_lid]);
    let self_v = m.load(t_f32, self_v_ptr);
    let self_i = m.load(t_u32, self_i_ptr);
    let other_v = m.load(t_f32, other_v_ptr);
    let other_i = m.load(t_u32, other_i_ptr);
    let other_beats = m.f_ord_greater_than(t_bool, other_v, self_v);
    let new_v = m.select(t_f32, other_beats, other_v, self_v);
    let new_i = m.select(t_u32, other_beats, other_i, self_i);
    m.store(self_v_ptr, new_v);
    m.store(self_i_ptr, new_i);
    m.branch(lbl_r_am);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_am.0]));
    {
        let c_scope_wg = m.constant_u32(t_u32, scope::WORKGROUP);
        let c_sem = m.constant_u32(
            t_u32,
            memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
        );
        m.control_barrier(c_scope_wg, c_scope_wg, c_sem);
    }
    m.branch(lbl_r_cont);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_cont.0]));
    let step_half = m.shift_right_logical(t_u32, step_v, c_u32_1);
    m.store(var_step, step_half);
    m.branch(lbl_r_h);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_r_m.0]));

    // ---- Thread 0 writes argmax_out[0] ----
    let is_lid_zero = m.u_less_than(t_bool, lid, c_u32_1);
    let lbl_w = m.alloc_id();
    let lbl_w_m = m.alloc_id();
    m.selection_merge(lbl_w_m, 0);
    m.branch_conditional(is_lid_zero, lbl_w, lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w.0]));
    let win_idx_ptr = m.access_chain(t_ptr_wg_u32, gvar_shared_idxs, &[c_u32_0]);
    let win_idx = m.load(t_u32, win_idx_ptr);
    let out_ptr = m.access_chain(t_ptr_sb_u32, gvar_out, &[c_u32_0, c_u32_0]);
    m.store(out_ptr, win_idx);
    m.branch(lbl_w_m);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_w_m.0]));

    m.ret();
    m.function_end();

    m.encode()
}
