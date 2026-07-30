use super::builder::{builtin, decoration, op, storage_class, SpirvModule};

/// Emit token-owned route reduction and residual add.
///
/// Bindings: hidden f32, route-major down output f32, route weights f32.
/// Push constants: seq_len, top_k, hidden.
pub fn emit_moe_grouped_reduce(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_f32 = m.type_float(32);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_f32 = m.type_runtime_array(t_f32);
    let t_struct_hidden = m.type_struct(&[t_arr_f32]);
    let t_struct_down = m.type_struct(&[t_arr_f32]);
    let t_struct_weights = m.type_struct(&[t_arr_f32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32]);

    let t_ptr_sb_hidden = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_hidden);
    let t_ptr_sb_down = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_down);
    let t_ptr_sb_weights = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_weights);
    let t_ptr_pc_struct = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_input_u32 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_ptr_sb_f32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_f32);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_fn_u32 = m.type_pointer(storage_class::FUNCTION, t_u32);
    let t_ptr_fn_f32 = m.type_pointer(storage_class::FUNCTION, t_f32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    for structure in [t_struct_hidden, t_struct_down, t_struct_weights] {
        m.decorate(structure, decoration::BLOCK, &[]);
        m.member_decorate(structure, 0, decoration::OFFSET, &[0]);
    }
    m.decorate(t_arr_f32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_pc, 0, decoration::OFFSET, &[0]);
    m.member_decorate(t_struct_pc, 1, decoration::OFFSET, &[4]);
    m.member_decorate(t_struct_pc, 2, decoration::OFFSET, &[8]);

    let gvar_hidden = m.variable(t_ptr_sb_hidden, storage_class::STORAGE_BUFFER);
    let gvar_down = m.variable(t_ptr_sb_down, storage_class::STORAGE_BUFFER);
    let gvar_weights = m.variable(t_ptr_sb_weights, storage_class::STORAGE_BUFFER);
    let gvar_pc = m.variable(t_ptr_pc_struct, storage_class::PUSH_CONSTANT);
    let gvar_glob_id = m.variable(t_ptr_input_u32, storage_class::INPUT);

    for (var, binding) in [(gvar_hidden, 0), (gvar_down, 1), (gvar_weights, 2)] {
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
    let lbl_bounds_true = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_rank_header = m.alloc_id();
    let lbl_rank_body = m.alloc_id();
    let lbl_rank_continue = m.alloc_id();
    let lbl_rank_merge = m.alloc_id();

    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));
    let var_rank = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);
    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);

    let glob_id = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id, 0);
    let token = m.composite_extract(t_u32, glob_id, 1);
    let seq_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let seq_len = m.load(t_u32, seq_ptr);
    let top_k_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let top_k = m.load(t_u32, top_k_ptr);
    let hidden_ptr = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_2]);
    let hidden = m.load(t_u32, hidden_ptr);

    let row_in_bounds = m.u_less_than(t_bool, row, hidden);
    let token_in_bounds = m.u_less_than(t_bool, token, seq_len);
    let in_bounds = m.logical_and(t_bool, row_in_bounds, token_in_bounds);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));
    m.store(var_rank, c_u32_0);
    m.store(var_sum, c_f32_0);
    m.branch(lbl_rank_header);

    let lbl_rank_cond = m.alloc_id();
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_rank_header.0]));
    m.loop_merge(lbl_rank_merge, lbl_rank_continue, 0);
    m.branch(lbl_rank_cond);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_rank_cond.0]));
    let rank = m.load(t_u32, var_rank);
    let has_rank = m.u_less_than(t_bool, rank, top_k);
    m.branch_conditional(has_rank, lbl_rank_body, lbl_rank_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_rank_body.0]));
    let token_route_base = m.imul(t_u32, token, top_k);
    let route = m.iadd(t_u32, token_route_base, rank);
    let route_output_base = m.imul(t_u32, route, hidden);
    let down_idx = m.iadd(t_u32, route_output_base, row);
    let down_ptr = m.access_chain(t_ptr_sb_f32, gvar_down, &[c_u32_0, down_idx]);
    let down_value = m.load(t_f32, down_ptr);
    let weight_ptr = m.access_chain(t_ptr_sb_f32, gvar_weights, &[c_u32_0, route]);
    let weight = m.load(t_f32, weight_ptr);
    let weighted = m.fmul(t_f32, weight, down_value);
    let sum = m.load(t_f32, var_sum);
    let next_sum = m.fadd(t_f32, sum, weighted);
    m.store(var_sum, next_sum);
    m.branch(lbl_rank_continue);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_rank_continue.0]));
    let next_rank = m.iadd(t_u32, rank, c_u32_1);
    m.store(var_rank, next_rank);
    m.branch(lbl_rank_header);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_rank_merge.0]));
    let hidden_token_base = m.imul(t_u32, token, hidden);
    let hidden_idx = m.iadd(t_u32, hidden_token_base, row);
    let hidden_element_ptr = m.access_chain(t_ptr_sb_f32, gvar_hidden, &[c_u32_0, hidden_idx]);
    let hidden_value = m.load(t_f32, hidden_element_ptr);
    let reduced = m.load(t_f32, var_sum);
    let output = m.fadd(t_f32, hidden_value, reduced);
    m.store(hidden_element_ptr, output);
    m.branch(lbl_bounds_merge);

    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();
    m.encode()
}
