use super::builder::{builtin, decoration, op, storage_class, Id, SpirvModule};

#[derive(Clone, Copy)]
enum DownFormat {
    Q5K,
    Q6K,
}

/// Emit grouped Q5_K down-projection GEMV over raw row-major blocks.
///
/// Bindings: arena u32, slot ids u32, activation f32, down output f32.
/// Push constants: rows, cols, route_count, slot_stride_words, down_offset_words.
pub fn emit_moe_grouped_q5k_down(local_size_x: u32) -> Vec<u32> {
    emit_moe_grouped_down(local_size_x, DownFormat::Q5K)
}

/// Emit grouped Q6_K down-projection GEMV over route-major activations.
///
/// Bindings: arena u32, slot ids u32, activation f32, down output f32.
/// Push constants: rows, cols, route_count, slot_stride_words, down_offset_words.
pub fn emit_moe_grouped_q6k_down(local_size_x: u32) -> Vec<u32> {
    emit_moe_grouped_down(local_size_x, DownFormat::Q6K)
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
}

fn emit_moe_grouped_down(local_size_x: u32, format: DownFormat) -> Vec<u32> {
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
    let t_struct_arena = m.type_struct(&[t_arr_u32]);
    let t_struct_slots = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_f32]);
    let t_struct_output = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32, t_u32, t_u32]);

    let t_ptr_sb_arena = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_arena);
    let t_ptr_sb_slots = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_slots);
    let t_ptr_sb_input = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_input);
    let t_ptr_sb_output = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_output);
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
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_44 = m.constant_u32(t_u32, 44);
    let c_u32_53 = m.constant_u32(t_u32, 53);
    let c_u32_ff = m.constant_u32(t_u32, 0xff);
    let c_u32_0f = m.constant_u32(t_u32, 0x0f);
    let c_u32_3f = m.constant_u32(t_u32, 0x3f);
    let c_u32_ffff = m.constant_u32(t_u32, 0xffff);
    let c_u32_1f = m.constant_u32(t_u32, 0x1f);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3ff);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);
    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    for structure in [
        t_struct_arena,
        t_struct_slots,
        t_struct_input,
        t_struct_output,
    ] {
        m.decorate(structure, decoration::BLOCK, &[]);
        m.member_decorate(structure, 0, decoration::OFFSET, &[0]);
    }
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    for member in 0..5u32 {
        m.member_decorate(t_struct_pc, member, decoration::OFFSET, &[member * 4]);
    }

    let gvar_arena = m.variable(t_ptr_sb_arena, storage_class::STORAGE_BUFFER);
    let gvar_slots = m.variable(t_ptr_sb_slots, storage_class::STORAGE_BUFFER);
    let gvar_input = m.variable(t_ptr_sb_input, storage_class::STORAGE_BUFFER);
    let gvar_output = m.variable(t_ptr_sb_output, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    for (var, binding) in [
        (gvar_arena, 0),
        (gvar_slots, 1),
        (gvar_input, 2),
        (gvar_output, 3),
    ] {
        m.decorate(var, decoration::DESCRIPTOR_SET, &[0]);
        m.decorate(var, decoration::BINDING, &[binding]);
    }
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

    let glob_id = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id, 0);
    let route = m.composite_extract(t_u32, glob_id, 1);

    let pc_rows_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_rows_ptr);
    let pc_cols_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_cols_ptr);
    let pc_routes_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let pc_routes = m.load(t_u32, pc_routes_ptr);
    let pc_stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_3]);
    let pc_slot_stride = m.load(t_u32, pc_stride_ptr);
    let pc_down_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_4]);
    let pc_down_offset = m.load(t_u32, pc_down_ptr);

    let row_in_bounds = m.u_less_than(t_bool, row, pc_rows);
    let route_in_bounds = m.u_less_than(t_bool, route, pc_routes);
    let in_bounds = m.logical_and(t_bool, row_in_bounds, route_in_bounds);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));
    m.store(var_sum, c_f32_0);

    let slot_ptr = m.access_chain(t_ptr_sb_u32, gvar_slots, &[c_u32_0, route]);
    let slot = m.load(t_u32, slot_ptr);
    let slot_base = m.imul(t_u32, slot, pc_slot_stride);
    let expert_base = m.iadd(t_u32, slot_base, pc_down_offset);
    let input_route_base = m.imul(t_u32, route, pc_cols);
    let output_route_base = m.imul(t_u32, route, pc_rows);
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

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

    match format {
        DownFormat::Q5K => emit_q5k_block(
            &mut m,
            t_bool,
            t_u32,
            t_f32,
            t_ptr_sb_u32,
            t_ptr_sb_f32,
            gvar_arena,
            gvar_input,
            var_sum,
            row,
            pc_cols,
            blk_cur,
            expert_base,
            input_route_base,
            c_u32_0,
            c_u32_1,
            c_u32_2,
            c_u32_3,
            c_u32_4,
            c_u32_6,
            c_u32_10,
            c_u32_13,
            c_u32_15,
            c_u32_16,
            c_u32_23,
            c_u32_31,
            c_u32_32,
            c_u32_44,
            c_u32_ff,
            c_u32_0f,
            c_u32_3f,
            c_u32_ffff,
            c_u32_1f,
            c_u32_3ff,
            c_u32_112,
            c_u32_256,
            c_f32_0,
            c_f32_2pow_neg24,
        ),
        DownFormat::Q6K => emit_q6k_block(
            &mut m,
            t_bool,
            t_u32,
            t_i32,
            t_f32,
            t_ptr_sb_u32,
            t_ptr_sb_f32,
            gvar_arena,
            gvar_input,
            var_sum,
            row,
            pc_rows,
            blk_cur,
            expert_base,
            input_route_base,
            c_u32_0,
            c_u32_1,
            c_u32_3,
            c_u32_4,
            c_u32_10,
            c_u32_13,
            c_u32_15,
            c_u32_23,
            c_u32_24,
            c_u32_31,
            c_u32_32,
            c_u32_53,
            c_u32_ff,
            c_u32_0f,
            c_u32_ffff,
            c_u32_1f,
            c_u32_3ff,
            c_u32_112,
            c_u32_256,
            c_f32_0,
            c_f32_2pow_neg24,
        ),
    }

    m.branch(lbl_outer_continue);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));
    let final_sum = m.load(t_f32, var_sum);
    let out_idx = m.iadd(t_u32, output_route_base, row);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, out_idx]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();
    m.encode()
}

#[allow(clippy::too_many_arguments)]
fn emit_q5k_block(
    m: &mut SpirvModule,
    t_bool: Id,
    t_u32: Id,
    t_f32: Id,
    t_ptr_sb_u32: Id,
    t_ptr_sb_f32: Id,
    gvar_arena: Id,
    gvar_input: Id,
    var_sum: Id,
    row: Id,
    pc_cols: Id,
    blk_cur: Id,
    expert_base: Id,
    input_route_base: Id,
    c_u32_0: Id,
    c_u32_1: Id,
    c_u32_2: Id,
    c_u32_3: Id,
    c_u32_4: Id,
    c_u32_6: Id,
    c_u32_10: Id,
    c_u32_13: Id,
    c_u32_15: Id,
    c_u32_16: Id,
    c_u32_23: Id,
    c_u32_31: Id,
    c_u32_32: Id,
    c_u32_44: Id,
    c_u32_ff: Id,
    c_u32_0f: Id,
    c_u32_3f: Id,
    c_u32_ffff: Id,
    c_u32_1f: Id,
    c_u32_3ff: Id,
    c_u32_112: Id,
    c_u32_256: Id,
    c_f32_0: Id,
    c_f32_2pow_neg24: Id,
) {
    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);
    let row_block_base = m.imul(t_u32, row, num_blocks);
    let block_index = m.iadd(t_u32, row_block_base, blk_cur);
    let block_words = m.imul(t_u32, block_index, c_u32_44);
    let plane_base = m.iadd(t_u32, expert_base, block_words);

    let packed_addr = plane_base;
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);
    let d_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = f16_bits_to_f32(
        m,
        t_bool,
        t_u32,
        t_f32,
        d_raw,
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
    let dmin_raw = m.shift_right_logical(t_u32, packed_word, c_u32_16);
    let dmin_f32 = f16_bits_to_f32(
        m,
        t_bool,
        t_u32,
        t_f32,
        dmin_raw,
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

    let s0_addr = m.iadd(t_u32, plane_base, c_u32_1);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, s0_addr]);
    let s0_word = m.load(t_u32, s0_ptr);
    let s1_addr = m.iadd(t_u32, plane_base, c_u32_2);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, s1_addr]);
    let s1_word = m.load(t_u32, s1_ptr);
    let s2_addr = m.iadd(t_u32, plane_base, c_u32_3);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, s2_addr]);
    let s2_word = m.load(t_u32, s2_ptr);

    let mut sb = [c_u32_0; 12];
    for i in 0..4u32 {
        if i == 0 {
            sb[i as usize] = m.bitwise_and(t_u32, s0_word, c_u32_ff);
            sb[4 + i as usize] = m.bitwise_and(t_u32, s1_word, c_u32_ff);
            sb[8 + i as usize] = m.bitwise_and(t_u32, s2_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let s0_shifted = m.shift_right_logical(t_u32, s0_word, shift);
            let s1_shifted = m.shift_right_logical(t_u32, s1_word, shift);
            let s2_shifted = m.shift_right_logical(t_u32, s2_word, shift);
            sb[i as usize] = m.bitwise_and(t_u32, s0_shifted, c_u32_ff);
            sb[4 + i as usize] = m.bitwise_and(t_u32, s1_shifted, c_u32_ff);
            sb[8 + i as usize] = m.bitwise_and(t_u32, s2_shifted, c_u32_ff);
        }
    }

    let mut scales = [c_u32_0; 8];
    let mut mins = [c_u32_0; 8];
    for j in 0..4usize {
        scales[j] = m.bitwise_and(t_u32, sb[j], c_u32_3f);
        mins[j] = m.bitwise_and(t_u32, sb[j + 4], c_u32_3f);
    }
    for j in 4..8usize {
        let lo = m.bitwise_and(t_u32, sb[j + 4], c_u32_0f);
        let hi_raw = m.shift_right_logical(t_u32, sb[j - 4], c_u32_6);
        let hi = m.shift_left_logical(t_u32, hi_raw, c_u32_4);
        scales[j] = m.bitwise_or(t_u32, lo, hi);
        let lo2 = m.shift_right_logical(t_u32, sb[j + 4], c_u32_4);
        let hi2_raw = m.shift_right_logical(t_u32, sb[j], c_u32_6);
        let hi2 = m.shift_left_logical(t_u32, hi2_raw, c_u32_4);
        mins[j] = m.bitwise_or(t_u32, lo2, hi2);
    }

    let blk_x_256 = m.imul(t_u32, blk_cur, c_u32_256);
    let mut total_sum = c_f32_0;
    let mut qh_words = [c_u32_0; 8];
    for w in 0..8u32 {
        let qh_word = m.constant_u32(t_u32, 4 + w);
        let qh_addr = m.iadd(t_u32, plane_base, qh_word);
        let qh_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, qh_addr]);
        qh_words[w as usize] = m.load(t_u32, qh_ptr);
    }

    for qs_group in 0..4u32 {
        let sb_lo = (qs_group * 2) as usize;
        let sb_hi = sb_lo + 1;
        let sc_lo_f = m.convert_u_to_f(t_f32, scales[sb_lo]);
        let mn_lo_f = m.convert_u_to_f(t_f32, mins[sb_lo]);
        let sc_hi_f = m.convert_u_to_f(t_f32, scales[sb_hi]);
        let mn_hi_f = m.convert_u_to_f(t_f32, mins[sb_hi]);
        let qs_base_word = m.constant_u32(t_u32, 12 + qs_group * 8);
        let qs_start = m.iadd(t_u32, plane_base, qs_base_word);
        let sb_offset = m.constant_u32(t_u32, qs_group * 64);
        let inp_sb_base = m.iadd(t_u32, blk_x_256, sb_offset);
        let mut q_input_sum_lo = c_f32_0;
        let mut q_input_sum_hi = c_f32_0;
        let mut input_sum_lo = c_f32_0;
        let mut input_sum_hi = c_f32_0;

        for w in 0..8u32 {
            let qs_addr = if w == 0 {
                qs_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                m.iadd(t_u32, qs_start, c_w)
            };
            let qs_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, qs_addr]);
            let qs_word = m.load(t_u32, qs_ptr);
            let qh_word = qh_words[w as usize];
            let inp_w_base = if w == 0 {
                inp_sb_base
            } else {
                let c_w4 = m.constant_u32(t_u32, w * 4);
                m.iadd(t_u32, inp_sb_base, c_w4)
            };

            for byte_idx in 0..4u32 {
                let lo_shift = byte_idx * 8;
                let nibble_lo = if lo_shift == 0 {
                    m.bitwise_and(t_u32, qs_word, c_u32_0f)
                } else {
                    let shift = m.constant_u32(t_u32, lo_shift);
                    let shifted = m.shift_right_logical(t_u32, qs_word, shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                let hi_shift = m.constant_u32(t_u32, byte_idx * 8 + 4);
                let hi_shifted = m.shift_right_logical(t_u32, qs_word, hi_shift);
                let nibble_hi = m.bitwise_and(t_u32, hi_shifted, c_u32_0f);
                let qh_shift_lo = byte_idx * 8 + qs_group * 2;
                let high_bit_lo = if qh_shift_lo == 0 {
                    m.bitwise_and(t_u32, qh_word, c_u32_1)
                } else {
                    let shift = m.constant_u32(t_u32, qh_shift_lo);
                    let shifted = m.shift_right_logical(t_u32, qh_word, shift);
                    m.bitwise_and(t_u32, shifted, c_u32_1)
                };
                let qh_shift_hi = m.constant_u32(t_u32, qh_shift_lo + 1);
                let high_hi_shifted = m.shift_right_logical(t_u32, qh_word, qh_shift_hi);
                let high_bit_hi = m.bitwise_and(t_u32, high_hi_shifted, c_u32_1);
                let high_lo = m.shift_left_logical(t_u32, high_bit_lo, c_u32_4);
                let high_hi = m.shift_left_logical(t_u32, high_bit_hi, c_u32_4);
                let q5_lo = m.bitwise_or(t_u32, nibble_lo, high_lo);
                let q5_hi = m.bitwise_or(t_u32, nibble_hi, high_hi);
                let q5_lo_f = m.convert_u_to_f(t_f32, q5_lo);
                let q5_hi_f = m.convert_u_to_f(t_f32, q5_hi);
                let inp_lo_element = if byte_idx == 0 {
                    inp_w_base
                } else {
                    let c_byte = m.constant_u32(t_u32, byte_idx);
                    m.iadd(t_u32, inp_w_base, c_byte)
                };
                let inp_hi_element = m.iadd(t_u32, inp_lo_element, c_u32_32);
                let inp_lo_idx = m.iadd(t_u32, input_route_base, inp_lo_element);
                let inp_hi_idx = m.iadd(t_u32, input_route_base, inp_hi_element);
                let inp_lo_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_lo_idx]);
                let inp_hi_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_hi_idx]);
                let inp_lo = m.load(t_f32, inp_lo_ptr);
                let inp_hi = m.load(t_f32, inp_hi_ptr);
                let qi_prod_lo = m.fmul(t_f32, q5_lo_f, inp_lo);
                let qi_prod_hi = m.fmul(t_f32, q5_hi_f, inp_hi);
                q_input_sum_lo = m.fadd(t_f32, q_input_sum_lo, qi_prod_lo);
                q_input_sum_hi = m.fadd(t_f32, q_input_sum_hi, qi_prod_hi);
                input_sum_lo = m.fadd(t_f32, input_sum_lo, inp_lo);
                input_sum_hi = m.fadd(t_f32, input_sum_hi, inp_hi);
            }
        }

        let d_sc_lo = m.fmul(t_f32, d_f32, sc_lo_f);
        let dmin_mn_lo = m.fmul(t_f32, dmin_f32, mn_lo_f);
        let d_sc_hi = m.fmul(t_f32, d_f32, sc_hi_f);
        let dmin_mn_hi = m.fmul(t_f32, dmin_f32, mn_hi_f);
        let term1_lo = m.fmul(t_f32, d_sc_lo, q_input_sum_lo);
        let term2_lo = m.fmul(t_f32, dmin_mn_lo, input_sum_lo);
        let result_lo = m.fsub(t_f32, term1_lo, term2_lo);
        total_sum = m.fadd(t_f32, total_sum, result_lo);
        let term1_hi = m.fmul(t_f32, d_sc_hi, q_input_sum_hi);
        let term2_hi = m.fmul(t_f32, dmin_mn_hi, input_sum_hi);
        let result_hi = m.fsub(t_f32, term1_hi, term2_hi);
        total_sum = m.fadd(t_f32, total_sum, result_hi);
    }

    let previous = m.load(t_f32, var_sum);
    let accumulated = m.fadd(t_f32, previous, total_sum);
    m.store(var_sum, accumulated);
}

struct Q6SubBlock {
    ql_off: u32,
    high_nibble: bool,
    qh_off: u32,
    qh_shift: u32,
    elem_off: u32,
    scale_idx: u32,
}

const Q6_SUB_BLOCKS: [Q6SubBlock; 16] = [
    Q6SubBlock {
        ql_off: 5,
        high_nibble: false,
        qh_off: 37,
        qh_shift: 0,
        elem_off: 0,
        scale_idx: 0,
    },
    Q6SubBlock {
        ql_off: 9,
        high_nibble: false,
        qh_off: 41,
        qh_shift: 0,
        elem_off: 16,
        scale_idx: 1,
    },
    Q6SubBlock {
        ql_off: 13,
        high_nibble: false,
        qh_off: 37,
        qh_shift: 2,
        elem_off: 32,
        scale_idx: 2,
    },
    Q6SubBlock {
        ql_off: 17,
        high_nibble: false,
        qh_off: 41,
        qh_shift: 2,
        elem_off: 48,
        scale_idx: 3,
    },
    Q6SubBlock {
        ql_off: 5,
        high_nibble: true,
        qh_off: 37,
        qh_shift: 4,
        elem_off: 64,
        scale_idx: 4,
    },
    Q6SubBlock {
        ql_off: 9,
        high_nibble: true,
        qh_off: 41,
        qh_shift: 4,
        elem_off: 80,
        scale_idx: 5,
    },
    Q6SubBlock {
        ql_off: 13,
        high_nibble: true,
        qh_off: 37,
        qh_shift: 6,
        elem_off: 96,
        scale_idx: 6,
    },
    Q6SubBlock {
        ql_off: 17,
        high_nibble: true,
        qh_off: 41,
        qh_shift: 6,
        elem_off: 112,
        scale_idx: 7,
    },
    Q6SubBlock {
        ql_off: 21,
        high_nibble: false,
        qh_off: 45,
        qh_shift: 0,
        elem_off: 128,
        scale_idx: 8,
    },
    Q6SubBlock {
        ql_off: 25,
        high_nibble: false,
        qh_off: 49,
        qh_shift: 0,
        elem_off: 144,
        scale_idx: 9,
    },
    Q6SubBlock {
        ql_off: 29,
        high_nibble: false,
        qh_off: 45,
        qh_shift: 2,
        elem_off: 160,
        scale_idx: 10,
    },
    Q6SubBlock {
        ql_off: 33,
        high_nibble: false,
        qh_off: 49,
        qh_shift: 2,
        elem_off: 176,
        scale_idx: 11,
    },
    Q6SubBlock {
        ql_off: 21,
        high_nibble: true,
        qh_off: 45,
        qh_shift: 4,
        elem_off: 192,
        scale_idx: 12,
    },
    Q6SubBlock {
        ql_off: 25,
        high_nibble: true,
        qh_off: 49,
        qh_shift: 4,
        elem_off: 208,
        scale_idx: 13,
    },
    Q6SubBlock {
        ql_off: 29,
        high_nibble: true,
        qh_off: 45,
        qh_shift: 6,
        elem_off: 224,
        scale_idx: 14,
    },
    Q6SubBlock {
        ql_off: 33,
        high_nibble: true,
        qh_off: 49,
        qh_shift: 6,
        elem_off: 240,
        scale_idx: 15,
    },
];

#[allow(clippy::too_many_arguments)]
fn emit_q6k_block(
    m: &mut SpirvModule,
    t_bool: Id,
    t_u32: Id,
    t_i32: Id,
    t_f32: Id,
    t_ptr_sb_u32: Id,
    t_ptr_sb_f32: Id,
    gvar_arena: Id,
    gvar_input: Id,
    var_sum: Id,
    row: Id,
    pc_rows: Id,
    blk_cur: Id,
    expert_base: Id,
    input_route_base: Id,
    c_u32_0: Id,
    c_u32_1: Id,
    c_u32_3: Id,
    c_u32_4: Id,
    c_u32_10: Id,
    c_u32_13: Id,
    c_u32_15: Id,
    c_u32_23: Id,
    c_u32_24: Id,
    c_u32_31: Id,
    c_u32_32: Id,
    c_u32_53: Id,
    c_u32_ff: Id,
    c_u32_0f: Id,
    c_u32_ffff: Id,
    c_u32_1f: Id,
    c_u32_3ff: Id,
    c_u32_112: Id,
    c_u32_256: Id,
    c_f32_0: Id,
    c_f32_2pow_neg24: Id,
) {
    let blk_x_53 = m.imul(t_u32, blk_cur, c_u32_53);
    let local_plane_base = m.imul(t_u32, blk_x_53, pc_rows);
    let plane_base = m.iadd(t_u32, expert_base, local_plane_base);

    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);
    let d_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = f16_bits_to_f32(
        m,
        t_bool,
        t_u32,
        t_f32,
        d_raw,
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

    let mut scale_words = [c_u32_0; 4];
    for i in 0..4u32 {
        let plane = m.constant_u32(t_u32, i + 1);
        let plane_offset = m.imul(t_u32, plane, pc_rows);
        let addr = m.iadd(t_u32, plane_base, plane_offset);
        let addr = m.iadd(t_u32, addr, row);
        let ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, addr]);
        scale_words[i as usize] = m.load(t_u32, ptr);
    }

    let blk_x_256 = m.imul(t_u32, blk_cur, c_u32_256);
    let mut total_sum = c_f32_0;
    for subblock in &Q6_SUB_BLOCKS {
        let scale_word = scale_words[(subblock.scale_idx / 4) as usize];
        let byte_shift = (subblock.scale_idx % 4) * 8;
        let scale_byte = if byte_shift == 0 {
            m.bitwise_and(t_u32, scale_word, c_u32_ff)
        } else {
            let shift = m.constant_u32(t_u32, byte_shift);
            let shifted = m.shift_right_logical(t_u32, scale_word, shift);
            m.bitwise_and(t_u32, shifted, c_u32_ff)
        };
        let shl24 = m.shift_left_logical(t_u32, scale_byte, c_u32_24);
        let shl24_i32 = m.bitcast(t_i32, shl24);
        let sign_ext = m.shift_right_arithmetic(t_i32, shl24_i32, c_u32_24);
        let scale_f = m.convert_s_to_f(t_f32, sign_ext);

        let ql_plane = m.constant_u32(t_u32, subblock.ql_off);
        let ql_offset = m.imul(t_u32, ql_plane, pc_rows);
        let ql_start = m.iadd(t_u32, plane_base, ql_offset);
        let qh_plane = m.constant_u32(t_u32, subblock.qh_off);
        let qh_offset = m.imul(t_u32, qh_plane, pc_rows);
        let qh_start = m.iadd(t_u32, plane_base, qh_offset);
        let elem_offset = m.constant_u32(t_u32, subblock.elem_off);
        let inp_sb_base = if subblock.elem_off == 0 {
            blk_x_256
        } else {
            m.iadd(t_u32, blk_x_256, elem_offset)
        };
        let d_scale = m.fmul(t_f32, d_f32, scale_f);

        for w in 0..4u32 {
            let ql_word_base = if w == 0 {
                ql_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, ql_start, w_rows)
            };
            let ql_addr = m.iadd(t_u32, ql_word_base, row);
            let ql_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, ql_addr]);
            let ql_word = m.load(t_u32, ql_ptr);
            let qh_word_base = if w == 0 {
                qh_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, qh_start, w_rows)
            };
            let qh_addr = m.iadd(t_u32, qh_word_base, row);
            let qh_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, qh_addr]);
            let qh_word = m.load(t_u32, qh_ptr);
            let inp_w_base = if w == 0 {
                inp_sb_base
            } else {
                let c_w4 = m.constant_u32(t_u32, w * 4);
                m.iadd(t_u32, inp_sb_base, c_w4)
            };

            for byte_idx in 0..4u32 {
                let ql_shift_amount = if subblock.high_nibble {
                    byte_idx * 8 + 4
                } else {
                    byte_idx * 8
                };
                let ql_nibble = if ql_shift_amount == 0 {
                    m.bitwise_and(t_u32, ql_word, c_u32_0f)
                } else {
                    let shift = m.constant_u32(t_u32, ql_shift_amount);
                    let shifted = m.shift_right_logical(t_u32, ql_word, shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                let qh_shift_amount = byte_idx * 8 + subblock.qh_shift;
                let qh_bits = if qh_shift_amount == 0 {
                    m.bitwise_and(t_u32, qh_word, c_u32_3)
                } else {
                    let shift = m.constant_u32(t_u32, qh_shift_amount);
                    let shifted = m.shift_right_logical(t_u32, qh_word, shift);
                    m.bitwise_and(t_u32, shifted, c_u32_3)
                };
                let qh_shifted = m.shift_left_logical(t_u32, qh_bits, c_u32_4);
                let q6 = m.bitwise_or(t_u32, ql_nibble, qh_shifted);
                let q6_i32 = m.bitcast(t_i32, q6);
                let c_32_i32 = m.bitcast(t_i32, c_u32_32);
                let q6_centered = m.isub(t_i32, q6_i32, c_32_i32);
                let inp_element = if byte_idx == 0 {
                    inp_w_base
                } else {
                    let c_byte = m.constant_u32(t_u32, byte_idx);
                    m.iadd(t_u32, inp_w_base, c_byte)
                };
                let q6_f = m.convert_s_to_f(t_f32, q6_centered);
                let dequant = m.fmul(t_f32, d_scale, q6_f);
                let inp_idx = m.iadd(t_u32, input_route_base, inp_element);
                let inp_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, inp_idx]);
                let inp_val = m.load(t_f32, inp_ptr);
                let product = m.fmul(t_f32, dequant, inp_val);
                total_sum = m.fadd(t_f32, total_sum, product);
            }
        }
    }

    let previous = m.load(t_f32, var_sum);
    let accumulated = m.fadd(t_f32, previous, total_sum);
    m.store(var_sum, accumulated);
}
