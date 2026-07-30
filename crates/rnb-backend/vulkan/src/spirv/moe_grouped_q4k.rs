use super::builder::{
    builtin, decoration, memory_semantics, op, scope, storage_class, Id, SpirvModule,
};

fn emit_label(m: &mut SpirvModule, label: Id) {
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label.0]));
}

/// Emit the grouped selected-expert Q4_K gate projection.
///
/// Workgroup X retains the row-major two-row mapping used by
/// `emit_q4k_gemv_rowmajor`; workgroup Y selects one route. The selected
/// arena slot remains in its original packed Q4_K representation.
pub fn emit_moe_grouped_q4k_gate(local_size_x: u32) -> Vec<u32> {
    emit_moe_grouped_q4k_impl(local_size_x, false)
}

/// Emit the grouped selected-expert Q4_K up projection and fuse
/// `SiLU(gate) * up` into the route-major activation output.
pub fn emit_moe_grouped_q4k_up_silu(local_size_x: u32) -> Vec<u32> {
    emit_moe_grouped_q4k_impl(local_size_x, true)
}

fn emit_moe_grouped_q4k_impl(local_size_x: u32, up_silu: bool) -> Vec<u32> {
    assert_eq!(
        local_size_x, 64,
        "grouped Q4_K row-major two-row subgroup kernel requires 64 lanes"
    );

    let mut m = SpirvModule::new();
    m.capability(1); // Shader
    m.capability(61); // GroupNonUniform
    m.capability(63); // GroupNonUniformArithmetic
    m.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = up_silu.then(|| m.ext_inst_import("GLSL.std.450"));
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_struct_u32 = m.type_struct(&[t_arr_u32]);
    let t_struct_f32 = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32, t_u32, t_u32, t_u32, t_u32]);
    let c_local_size = m.constant_u32(t_u32, local_size_x);
    let t_shared_arr = m.type_array(t_f32, c_local_size);

    let t_ptr_sb_u32_struct = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_u32);
    let t_ptr_sb_f32_struct = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_f32);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_ptr_wg_arr = m.type_pointer(storage_class::WORKGROUP, t_shared_arr);
    let t_ptr_wg_f32 = m.type_pointer(storage_class::WORKGROUP, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_5 = m.constant_u32(t_u32, 5);
    let c_u32_6 = m.constant_u32(t_u32, 6);
    let c_u32_8 = m.constant_u32(t_u32, 8);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_32 = m.constant_u32(t_u32, 32);
    let c_u32_36 = m.constant_u32(t_u32, 36);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_128 = m.constant_u32(t_u32, 128);
    let c_u32_160 = m.constant_u32(t_u32, 160);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_3f = m.constant_u32(t_u32, 0x3F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_block_groups = m.constant_u32(t_u32, local_size_x / 32);
    let c_scope_subgroup = m.constant_u32(t_u32, scope::SUBGROUP);
    let c_scope_workgroup = m.constant_u32(t_u32, scope::WORKGROUP);
    let c_workgroup_semantics = m.constant_u32(
        t_u32,
        memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
    );
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
    let gvar_activation = m.variable(t_ptr_sb_f32_struct, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let gvar_lid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_wgid = m.variable(t_ptr_input_v3, storage_class::INPUT);
    let gvar_subgroup_local_id = m.variable(t_ptr_input_u32, storage_class::INPUT);
    let gvar_shared = m.variable(t_ptr_wg_arr, storage_class::WORKGROUP);

    for (binding, variable) in [
        gvar_arena,
        gvar_slot_ids,
        gvar_token_ids,
        gvar_input,
        gvar_gate,
        gvar_activation,
    ]
    .into_iter()
    .enumerate()
    {
        m.decorate(variable, decoration::DESCRIPTOR_SET, &[0]);
        m.decorate(variable, decoration::BINDING, &[binding as u32]);
    }
    m.decorate(
        gvar_lid,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );
    m.decorate(gvar_wgid, decoration::BUILTIN, &[builtin::WORKGROUP_ID]);
    m.decorate(
        gvar_subgroup_local_id,
        decoration::BUILTIN,
        &[builtin::SUBGROUP_LOCAL_INVOCATION_ID],
    );

    let func_id = m.alloc_id();
    m.entry_point(
        5,
        func_id,
        "main",
        &[gvar_lid, gvar_wgid, gvar_subgroup_local_id],
    );
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    let lbl_entry = m.alloc_id();
    let lbl_route_active = m.alloc_id();
    let lbl_route_merge = m.alloc_id();
    let lbl_block_header = m.alloc_id();
    let lbl_block_cond = m.alloc_id();
    let lbl_block_body = m.alloc_id();
    let lbl_block_continue = m.alloc_id();
    let lbl_block_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    emit_label(&mut m, lbl_entry);

    let var_block = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);

    let lid_vec = m.load(t_v3u32, gvar_lid);
    let lid = m.composite_extract(t_u32, lid_vec, 0);
    let wgid_vec = m.load(t_v3u32, gvar_wgid);
    let row_group = m.composite_extract(t_u32, wgid_vec, 0);
    let route = m.composite_extract(t_u32, wgid_vec, 1);
    let pc_route_count_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let route_count = m.load(t_u32, pc_route_count_ptr);
    let route_active = m.u_less_than(t_bool, route, route_count);
    m.selection_merge(lbl_route_merge, 0);
    m.branch_conditional(route_active, lbl_route_active, lbl_route_merge);
    emit_label(&mut m, lbl_route_active);

    let first_row = m.imul(t_u32, row_group, c_u32_2);
    let row_in_workgroup = m.udiv(t_u32, lid, c_u32_32);
    let row_candidate = m.iadd(t_u32, first_row, row_in_workgroup);
    let pc_rows_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_rows_ptr);
    let last_row = m.isub(t_u32, pc_rows, c_u32_1);
    let row_in_bounds = m.u_less_than(t_bool, row_candidate, pc_rows);
    let row = m.select(t_u32, row_in_bounds, row_candidate, last_row);
    let subgroup_local_id = m.load(t_u32, gvar_subgroup_local_id);

    let lane_within_row = m.umod(t_u32, lid, c_u32_32);
    let block_group = m.udiv(t_u32, lane_within_row, c_u32_16);
    let block_lane = m.umod(t_u32, lane_within_row, c_u32_16);
    let lane_group = m.udiv(t_u32, block_lane, c_u32_4);
    let lane_in_group = m.umod(t_u32, block_lane, c_u32_4);
    let vector_group = m.udiv(t_u32, lane_group, c_u32_2);
    let vector_half = m.umod(t_u32, lane_group, c_u32_2);
    let twice_lane = m.imul(t_u32, lane_in_group, c_u32_2);
    let vector_index = m.iadd(t_u32, twice_lane, vector_half);
    let lane_offset = m.imul(t_u32, vector_index, c_u32_4);
    let quant_group_offset = m.imul(t_u32, vector_group, c_u32_8);
    let quant_word_index = m.iadd(t_u32, quant_group_offset, vector_index);
    let input_group_offset = m.imul(t_u32, vector_group, c_u32_64);
    let input_lane_offset = m.iadd(t_u32, input_group_offset, lane_offset);

    let pc_cols_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_cols_ptr);
    let pc_slot_stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_3]);
    let slot_stride_words = m.load(t_u32, pc_slot_stride_ptr);
    let pc_projection_offset_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_4]);
    let projection_offset_words = m.load(t_u32, pc_projection_offset_ptr);
    let pc_input_stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_5]);
    let input_stride = m.load(t_u32, pc_input_stride_ptr);
    let pc_output_stride_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_6]);
    let output_stride = m.load(t_u32, pc_output_stride_ptr);

    let slot_ptr = m.access_chain(t_ptr_sb_u32, gvar_slot_ids, &[c_u32_0, route]);
    let slot = m.load(t_u32, slot_ptr);
    let slot_base = m.imul(t_u32, slot, slot_stride_words);
    let projection_base = m.iadd(t_u32, slot_base, projection_offset_words);
    let token_ptr = m.access_chain(t_ptr_sb_u32, gvar_token_ids, &[c_u32_0, route]);
    let token = m.load(t_u32, token_ptr);
    let input_token_offset = m.imul(t_u32, token, input_stride);
    let output_route_offset = m.imul(t_u32, route, output_stride);

    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);
    let row_stride_words = m.imul(t_u32, num_blocks, c_u32_36);
    let row_words = m.imul(t_u32, row, row_stride_words);
    let row_weight_base = m.iadd(t_u32, projection_base, row_words);

    m.store(var_sum, c_f32_0);
    m.store(var_block, block_group);
    m.branch(lbl_block_header);

    emit_label(&mut m, lbl_block_header);
    m.loop_merge(lbl_block_merge, lbl_block_continue, 0);
    m.branch(lbl_block_cond);
    emit_label(&mut m, lbl_block_cond);
    let block = m.load(t_u32, var_block);
    let block_active = m.u_less_than(t_bool, block, num_blocks);
    m.branch_conditional(block_active, lbl_block_body, lbl_block_merge);

    emit_label(&mut m, lbl_block_body);
    let block_words = m.imul(t_u32, block, c_u32_36);
    let block_base = m.iadd(t_u32, row_weight_base, block_words);
    let block_input_base = m.imul(t_u32, block, c_u32_256);

    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, block_base]);
    let packed = m.load(t_u32, packed_ptr);
    let d_f16 = m.bitwise_and(t_u32, packed, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16, c_u32_3ff);
        let sign_raw = m.shift_right_logical(t_u32, d_f16, c_u32_15);
        let sign = m.bitwise_and(t_u32, sign_raw, c_u32_1);
        let sign_bits = m.shift_left_logical(t_u32, sign, c_u32_31);
        let adjusted_exp = m.iadd(t_u32, exp, c_u32_112);
        let exp_bits = m.shift_left_logical(t_u32, adjusted_exp, c_u32_23);
        let mant_bits = m.shift_left_logical(t_u32, mant, c_u32_13);
        let sign_exp = m.bitwise_or(t_u32, sign_bits, exp_bits);
        let normal_bits = m.bitwise_or(t_u32, sign_exp, mant_bits);
        let normal = m.bitcast(t_f32, normal_bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denormal_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denormal_neg = m.fnegate(t_f32, denormal_abs);
        let sign_set = m.i_not_equal(t_bool, sign, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denormal_neg, denormal_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };
    let dmin_f16 = m.shift_right_logical(t_u32, packed, c_u32_16);
    let dmin_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, dmin_f16, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, dmin_f16, c_u32_3ff);
        let sign_raw = m.shift_right_logical(t_u32, dmin_f16, c_u32_15);
        let sign = m.bitwise_and(t_u32, sign_raw, c_u32_1);
        let sign_bits = m.shift_left_logical(t_u32, sign, c_u32_31);
        let adjusted_exp = m.iadd(t_u32, exp, c_u32_112);
        let exp_bits = m.shift_left_logical(t_u32, adjusted_exp, c_u32_23);
        let mant_bits = m.shift_left_logical(t_u32, mant, c_u32_13);
        let sign_exp = m.bitwise_or(t_u32, sign_bits, exp_bits);
        let normal_bits = m.bitwise_or(t_u32, sign_exp, mant_bits);
        let normal = m.bitcast(t_f32, normal_bits);
        let mant_f = m.convert_u_to_f(t_f32, mant);
        let denormal_abs = m.fmul(t_f32, mant_f, c_f32_2pow_neg24);
        let denormal_neg = m.fnegate(t_f32, denormal_abs);
        let sign_set = m.i_not_equal(t_bool, sign, c_u32_0);
        let denormal = m.select(t_f32, sign_set, denormal_neg, denormal_abs);
        let exp_nonzero = m.i_not_equal(t_bool, exp, c_u32_0);
        m.select(t_f32, exp_nonzero, normal, denormal)
    };

    let scale0_addr = m.iadd(t_u32, block_base, c_u32_1);
    let scale0_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, scale0_addr]);
    let scale0_word = m.load(t_u32, scale0_ptr);
    let scale1_addr = m.iadd(t_u32, block_base, c_u32_2);
    let scale1_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, scale1_addr]);
    let scale1_word = m.load(t_u32, scale1_ptr);
    let scale2_addr = m.iadd(t_u32, block_base, c_u32_3);
    let scale2_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, scale2_addr]);
    let scale2_word = m.load(t_u32, scale2_ptr);

    let mut scales = [c_u32_0; 8];
    let mut mins = [c_u32_0; 8];
    for index in 0..4u32 {
        let shift = m.constant_u32(t_u32, index * 8);
        let scale_shifted = if index == 0 {
            scale0_word
        } else {
            m.shift_right_logical(t_u32, scale0_word, shift)
        };
        let min_shifted = if index == 0 {
            scale1_word
        } else {
            m.shift_right_logical(t_u32, scale1_word, shift)
        };
        scales[index as usize] = m.bitwise_and(t_u32, scale_shifted, c_u32_3f);
        mins[index as usize] = m.bitwise_and(t_u32, min_shifted, c_u32_3f);

        let high_source = if index == 0 {
            scale2_word
        } else {
            m.shift_right_logical(t_u32, scale2_word, shift)
        };
        let high_byte = m.bitwise_and(t_u32, high_source, c_u32_ff);
        let scale_low = m.bitwise_and(t_u32, high_byte, c_u32_0f);
        let scale_high_raw = m.shift_right_logical(t_u32, scale_shifted, c_u32_6);
        let scale_high_bits = m.bitwise_and(t_u32, scale_high_raw, c_u32_3);
        let scale_high = m.shift_left_logical(t_u32, scale_high_bits, c_u32_4);
        scales[index as usize + 4] = m.bitwise_or(t_u32, scale_low, scale_high);

        let min_low = m.shift_right_logical(t_u32, high_byte, c_u32_4);
        let min_high_raw = m.shift_right_logical(t_u32, min_shifted, c_u32_6);
        let min_high_bits = m.bitwise_and(t_u32, min_high_raw, c_u32_3);
        let min_high = m.shift_left_logical(t_u32, min_high_bits, c_u32_4);
        mins[index as usize + 4] = m.bitwise_or(t_u32, min_low, min_high);
    }

    let use_second_scale_group = m.i_not_equal(t_bool, vector_group, c_u32_0);
    let scale_ids = [
        m.select(t_u32, use_second_scale_group, scales[2], scales[0]),
        m.select(t_u32, use_second_scale_group, scales[3], scales[1]),
        m.select(t_u32, use_second_scale_group, scales[6], scales[4]),
        m.select(t_u32, use_second_scale_group, scales[7], scales[5]),
    ];
    let min_ids = [
        m.select(t_u32, use_second_scale_group, mins[2], mins[0]),
        m.select(t_u32, use_second_scale_group, mins[3], mins[1]),
        m.select(t_u32, use_second_scale_group, mins[6], mins[4]),
        m.select(t_u32, use_second_scale_group, mins[7], mins[5]),
    ];

    let quant_base = m.iadd(t_u32, block_base, c_u32_4);
    let quant0_addr = m.iadd(t_u32, quant_base, quant_word_index);
    let quant0_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, quant0_addr]);
    let quant0 = m.load(t_u32, quant0_ptr);
    let quant64_index = m.iadd(t_u32, quant_word_index, c_u32_16);
    let quant64_addr = m.iadd(t_u32, quant_base, quant64_index);
    let quant64_ptr = m.access_chain(t_ptr_sb_u32, gvar_arena, &[c_u32_0, quant64_addr]);
    let quant64 = m.load(t_u32, quant64_ptr);

    let mut quant_sums = [c_f32_0; 4];
    let mut input_sums = [c_f32_0; 4];
    for element in 0..4u32 {
        let shift = m.constant_u32(t_u32, element * 8);
        let quant0_shifted = if element == 0 {
            quant0
        } else {
            m.shift_right_logical(t_u32, quant0, shift)
        };
        let quant64_shifted = if element == 0 {
            quant64
        } else {
            m.shift_right_logical(t_u32, quant64, shift)
        };
        let quant_values = [
            m.bitwise_and(t_u32, quant0_shifted, c_u32_0f),
            {
                let high = m.shift_right_logical(t_u32, quant0_shifted, c_u32_4);
                m.bitwise_and(t_u32, high, c_u32_0f)
            },
            m.bitwise_and(t_u32, quant64_shifted, c_u32_0f),
            {
                let high = m.shift_right_logical(t_u32, quant64_shifted, c_u32_4);
                m.bitwise_and(t_u32, high, c_u32_0f)
            },
        ];

        let element_id = m.constant_u32(t_u32, element);
        let input0_base = m.iadd(t_u32, block_input_base, input_lane_offset);
        let input0_index = m.iadd(t_u32, input0_base, element_id);
        let input1_index = m.iadd(t_u32, input0_index, c_u32_32);
        let input2_index = m.iadd(t_u32, input0_index, c_u32_128);
        let input3_index = m.iadd(t_u32, input0_index, c_u32_160);
        let element_indices = [input0_index, input1_index, input2_index, input3_index];

        for group in 0..4 {
            let quant_f = m.convert_u_to_f(t_f32, quant_values[group]);
            let input_index = m.iadd(t_u32, input_token_offset, element_indices[group]);
            let input_ptr = m.access_chain(t_ptr_sb_f32, gvar_input, &[c_u32_0, input_index]);
            let input_value = m.load(t_f32, input_ptr);
            let quant_product = m.fmul(t_f32, input_value, quant_f);
            quant_sums[group] = m.fadd(t_f32, quant_sums[group], quant_product);
            input_sums[group] = m.fadd(t_f32, input_sums[group], input_value);
        }
    }

    let mut block_sum = c_f32_0;
    for group in 0..4 {
        let scale_f = m.convert_u_to_f(t_f32, scale_ids[group]);
        let min_f = m.convert_u_to_f(t_f32, min_ids[group]);
        let d_scale = m.fmul(t_f32, d_f32, scale_f);
        let dmin_scale = m.fmul(t_f32, dmin_f32, min_f);
        let quant_term = m.fmul(t_f32, d_scale, quant_sums[group]);
        let min_term = m.fmul(t_f32, dmin_scale, input_sums[group]);
        let group_sum = m.fsub(t_f32, quant_term, min_term);
        block_sum = m.fadd(t_f32, block_sum, group_sum);
    }
    let previous_sum = m.load(t_f32, var_sum);
    let new_sum = m.fadd(t_f32, previous_sum, block_sum);
    m.store(var_sum, new_sum);
    m.branch(lbl_block_continue);

    emit_label(&mut m, lbl_block_continue);
    let next_block = m.iadd(t_u32, block, c_block_groups);
    m.store(var_block, next_block);
    m.branch(lbl_block_header);

    emit_label(&mut m, lbl_block_merge);
    let lane_sum = m.load(t_f32, var_sum);
    let row_sum = m.group_non_uniform_fadd_clustered(t_f32, c_scope_subgroup, lane_sum, c_u32_32);
    let cluster_lane = m.umod(t_u32, subgroup_local_id, c_u32_32);
    let is_cluster_leader = m.u_less_than(t_bool, cluster_lane, c_u32_1);
    let lbl_store_partial = m.alloc_id();
    let lbl_store_partial_merge = m.alloc_id();
    m.selection_merge(lbl_store_partial_merge, 0);
    m.branch_conditional(
        is_cluster_leader,
        lbl_store_partial,
        lbl_store_partial_merge,
    );
    emit_label(&mut m, lbl_store_partial);
    let partial_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared, &[row_in_workgroup]);
    m.store(partial_ptr, row_sum);
    m.branch(lbl_store_partial_merge);
    emit_label(&mut m, lbl_store_partial_merge);
    m.control_barrier(c_scope_workgroup, c_scope_workgroup, c_workgroup_semantics);

    let is_workgroup_leader = m.u_less_than(t_bool, lid, c_u32_1);
    let lbl_write_rows = m.alloc_id();
    let lbl_write_rows_merge = m.alloc_id();
    m.selection_merge(lbl_write_rows_merge, 0);
    m.branch_conditional(is_workgroup_leader, lbl_write_rows, lbl_write_rows_merge);
    emit_label(&mut m, lbl_write_rows);

    let first_sum_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared, &[c_u32_0]);
    let first_sum = m.load(t_f32, first_sum_ptr);
    let first_output_index = m.iadd(t_u32, output_route_offset, first_row);
    let first_value = if let Some(glsl) = glsl {
        let gate_ptr = m.access_chain(t_ptr_sb_f32, gvar_gate, &[c_u32_0, first_output_index]);
        let gate = m.load(t_f32, gate_ptr);
        let neg_gate = m.fnegate(t_f32, gate);
        let exp_neg_gate = m.ext_inst(t_f32, glsl, 27, &[neg_gate]);
        let denominator = m.fadd(t_f32, c_f32_1, exp_neg_gate);
        let silu_gate = m.fdiv(t_f32, gate, denominator);
        m.fmul(t_f32, silu_gate, first_sum)
    } else {
        first_sum
    };
    let output = if up_silu { gvar_activation } else { gvar_gate };
    let first_output_ptr = m.access_chain(t_ptr_sb_f32, output, &[c_u32_0, first_output_index]);
    m.store(first_output_ptr, first_value);

    let second_row = m.iadd(t_u32, first_row, c_u32_1);
    let has_second_row = m.u_less_than(t_bool, second_row, pc_rows);
    let lbl_write_second = m.alloc_id();
    let lbl_write_second_merge = m.alloc_id();
    m.selection_merge(lbl_write_second_merge, 0);
    m.branch_conditional(has_second_row, lbl_write_second, lbl_write_second_merge);
    emit_label(&mut m, lbl_write_second);
    let second_sum_ptr = m.access_chain(t_ptr_wg_f32, gvar_shared, &[c_u32_1]);
    let second_sum = m.load(t_f32, second_sum_ptr);
    let second_output_index = m.iadd(t_u32, output_route_offset, second_row);
    let second_value = if let Some(glsl) = glsl {
        let gate_ptr = m.access_chain(t_ptr_sb_f32, gvar_gate, &[c_u32_0, second_output_index]);
        let gate = m.load(t_f32, gate_ptr);
        let neg_gate = m.fnegate(t_f32, gate);
        let exp_neg_gate = m.ext_inst(t_f32, glsl, 27, &[neg_gate]);
        let denominator = m.fadd(t_f32, c_f32_1, exp_neg_gate);
        let silu_gate = m.fdiv(t_f32, gate, denominator);
        m.fmul(t_f32, silu_gate, second_sum)
    } else {
        second_sum
    };
    let second_output_ptr = m.access_chain(t_ptr_sb_f32, output, &[c_u32_0, second_output_index]);
    m.store(second_output_ptr, second_value);
    m.branch(lbl_write_second_merge);
    emit_label(&mut m, lbl_write_second_merge);
    m.branch(lbl_write_rows_merge);

    emit_label(&mut m, lbl_write_rows_merge);
    m.branch(lbl_route_merge);
    emit_label(&mut m, lbl_route_merge);
    m.ret();
    m.function_end();
    m.encode()
}
