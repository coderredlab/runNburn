use super::builder::{
    builtin, decoration, memory_semantics, op, scope, storage_class, SpirvModule,
};

/// First stage of a hierarchical argmax over `(value, token_id)` pairs.
///
/// Each workgroup scans one contiguous value range, reduces it in shared memory,
/// and writes one winning pair. Push constants are `[count, values_per_group]`.
pub fn emit_argmax_pairs_f32_stage1(local_size_x: u32) -> Vec<u32> {
    let mut module = SpirvModule::new();

    module.capability(1);
    module.extension("SPV_KHR_storage_buffer_storage_class");
    module.memory_model(0, 1);

    let type_void = module.type_void();
    let type_bool = module.type_bool();
    let type_u32 = module.type_int(32, 0);
    let type_f32 = module.type_float(32);
    let type_vec3_u32 = module.type_vector(type_u32, 3);

    let type_array_f32 = module.type_runtime_array(type_f32);
    let type_array_u32 = module.type_runtime_array(type_u32);
    let type_struct_values = module.type_struct(&[type_array_f32]);
    let type_struct_indices = module.type_struct(&[type_array_u32]);
    let type_struct_partial_values = module.type_struct(&[type_array_f32]);
    let type_struct_partial_indices = module.type_struct(&[type_array_u32]);
    let type_struct_push = module.type_struct(&[type_u32, type_u32]);

    let constant_local_size = module.constant_u32(type_u32, local_size_x);
    let type_shared_array_f32 = module.type_array(type_f32, constant_local_size);
    let type_shared_array_u32 = module.type_array(type_u32, constant_local_size);
    let pointer_workgroup_array_f32 =
        module.type_pointer(storage_class::WORKGROUP, type_shared_array_f32);
    let pointer_workgroup_array_u32 =
        module.type_pointer(storage_class::WORKGROUP, type_shared_array_u32);
    let pointer_workgroup_f32 = module.type_pointer(storage_class::WORKGROUP, type_f32);
    let pointer_workgroup_u32 = module.type_pointer(storage_class::WORKGROUP, type_u32);

    let pointer_storage_values =
        module.type_pointer(storage_class::STORAGE_BUFFER, type_struct_values);
    let pointer_storage_indices =
        module.type_pointer(storage_class::STORAGE_BUFFER, type_struct_indices);
    let pointer_storage_partial_values =
        module.type_pointer(storage_class::STORAGE_BUFFER, type_struct_partial_values);
    let pointer_storage_partial_indices =
        module.type_pointer(storage_class::STORAGE_BUFFER, type_struct_partial_indices);
    let pointer_push = module.type_pointer(storage_class::PUSH_CONSTANT, type_struct_push);
    let pointer_input_vec3_u32 = module.type_pointer(storage_class::INPUT, type_vec3_u32);
    let pointer_storage_f32 = module.type_pointer(storage_class::STORAGE_BUFFER, type_f32);
    let pointer_storage_u32 = module.type_pointer(storage_class::STORAGE_BUFFER, type_u32);
    let pointer_push_u32 = module.type_pointer(storage_class::PUSH_CONSTANT, type_u32);
    let pointer_function_u32 = module.type_pointer(storage_class::FUNCTION, type_u32);
    let pointer_function_f32 = module.type_pointer(storage_class::FUNCTION, type_f32);
    let type_function_void = module.type_function(type_void, &[]);

    let constant_u32_0 = module.constant_u32(type_u32, 0);
    let constant_u32_1 = module.constant_u32(type_u32, 1);
    let constant_negative_infinity = module.constant_f32(type_f32, f32::NEG_INFINITY);

    for structure in [
        type_struct_values,
        type_struct_indices,
        type_struct_partial_values,
        type_struct_partial_indices,
    ] {
        module.decorate(structure, decoration::BLOCK, &[]);
        module.member_decorate(structure, 0, decoration::OFFSET, &[0]);
    }
    module.decorate(type_array_f32, decoration::ARRAY_STRIDE, &[4]);
    module.decorate(type_array_u32, decoration::ARRAY_STRIDE, &[4]);
    module.decorate(type_struct_push, decoration::BLOCK, &[]);
    module.member_decorate(type_struct_push, 0, decoration::OFFSET, &[0]);
    module.member_decorate(type_struct_push, 1, decoration::OFFSET, &[4]);

    let global_values = module.variable(pointer_storage_values, storage_class::STORAGE_BUFFER);
    let global_indices = module.variable(pointer_storage_indices, storage_class::STORAGE_BUFFER);
    let global_partial_values = module.variable(
        pointer_storage_partial_values,
        storage_class::STORAGE_BUFFER,
    );
    let global_partial_indices = module.variable(
        pointer_storage_partial_indices,
        storage_class::STORAGE_BUFFER,
    );
    let global_push = module.variable(pointer_push, storage_class::PUSH_CONSTANT);
    let global_local_id = module.variable(pointer_input_vec3_u32, storage_class::INPUT);
    let global_workgroup_id = module.variable(pointer_input_vec3_u32, storage_class::INPUT);
    let global_shared_values =
        module.variable(pointer_workgroup_array_f32, storage_class::WORKGROUP);
    let global_shared_indices =
        module.variable(pointer_workgroup_array_u32, storage_class::WORKGROUP);

    for (variable, binding) in [
        (global_values, 0),
        (global_indices, 1),
        (global_partial_values, 2),
        (global_partial_indices, 3),
    ] {
        module.decorate(variable, decoration::DESCRIPTOR_SET, &[0]);
        module.decorate(variable, decoration::BINDING, &[binding]);
    }
    module.decorate(
        global_local_id,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );
    module.decorate(
        global_workgroup_id,
        decoration::BUILTIN,
        &[builtin::WORKGROUP_ID],
    );

    let function_id = module.alloc_id();
    module.entry_point(
        5,
        function_id,
        "main",
        &[global_local_id, global_workgroup_id],
    );
    module.execution_mode_local_size(function_id, local_size_x, 1, 1);

    module.function(type_void, function_id, 0, type_function_void);
    let label_entry = module.alloc_id();
    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_entry.0]));

    let variable_index = module.function_variable(pointer_function_u32, storage_class::FUNCTION);
    let variable_best_value =
        module.function_variable(pointer_function_f32, storage_class::FUNCTION);
    let variable_best_index =
        module.function_variable(pointer_function_u32, storage_class::FUNCTION);
    let variable_step = module.function_variable(pointer_function_u32, storage_class::FUNCTION);

    let local_id_vector = module.load(type_vec3_u32, global_local_id);
    let local_id = module.composite_extract(type_u32, local_id_vector, 0);
    let workgroup_id_vector = module.load(type_vec3_u32, global_workgroup_id);
    let workgroup_id = module.composite_extract(type_u32, workgroup_id_vector, 0);
    let count_pointer = module.access_chain(pointer_push_u32, global_push, &[constant_u32_0]);
    let count = module.load(type_u32, count_pointer);
    let values_per_group_pointer =
        module.access_chain(pointer_push_u32, global_push, &[constant_u32_1]);
    let values_per_group = module.load(type_u32, values_per_group_pointer);
    let group_start = module.imul(type_u32, workgroup_id, values_per_group);
    let group_end_unclamped = module.iadd(type_u32, group_start, values_per_group);
    let group_end_in_bounds = module.u_less_than(type_bool, group_end_unclamped, count);
    let group_end = module.select(type_u32, group_end_in_bounds, group_end_unclamped, count);
    let first_index = module.iadd(type_u32, group_start, local_id);

    module.store(variable_best_value, constant_negative_infinity);
    module.store(variable_best_index, constant_u32_0);
    module.store(variable_index, first_index);

    let label_scan_header = module.alloc_id();
    let label_scan_condition = module.alloc_id();
    let label_scan_body = module.alloc_id();
    let label_scan_continue = module.alloc_id();
    let label_scan_merge = module.alloc_id();

    module.branch(label_scan_header);
    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_scan_header.0]));
    module.loop_merge(label_scan_merge, label_scan_continue, 0);
    module.branch(label_scan_condition);

    module.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_scan_condition.0],
    ));
    let current_index = module.load(type_u32, variable_index);
    let index_in_group = module.u_less_than(type_bool, current_index, group_end);
    module.branch_conditional(index_in_group, label_scan_body, label_scan_merge);

    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_scan_body.0]));
    let value_pointer = module.access_chain(
        pointer_storage_f32,
        global_values,
        &[constant_u32_0, current_index],
    );
    let value = module.load(type_f32, value_pointer);
    let index_pointer = module.access_chain(
        pointer_storage_u32,
        global_indices,
        &[constant_u32_0, current_index],
    );
    let token_index = module.load(type_u32, index_pointer);
    let current_best = module.load(type_f32, variable_best_value);
    let value_beats_best = module.f_ord_greater_than(type_bool, value, current_best);
    let label_update = module.alloc_id();
    let label_update_merge = module.alloc_id();
    module.selection_merge(label_update_merge, 0);
    module.branch_conditional(value_beats_best, label_update, label_update_merge);
    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_update.0]));
    module.store(variable_best_value, value);
    module.store(variable_best_index, token_index);
    module.branch(label_update_merge);
    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_update_merge.0]));
    module.branch(label_scan_continue);

    module.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_scan_continue.0],
    ));
    let next_index = module.iadd(type_u32, current_index, constant_local_size);
    module.store(variable_index, next_index);
    module.branch(label_scan_header);

    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_scan_merge.0]));
    let final_best_value = module.load(type_f32, variable_best_value);
    let final_best_index = module.load(type_u32, variable_best_index);
    let shared_value_pointer =
        module.access_chain(pointer_workgroup_f32, global_shared_values, &[local_id]);
    let shared_index_pointer =
        module.access_chain(pointer_workgroup_u32, global_shared_indices, &[local_id]);
    module.store(shared_value_pointer, final_best_value);
    module.store(shared_index_pointer, final_best_index);
    let workgroup_scope = module.constant_u32(type_u32, scope::WORKGROUP);
    let workgroup_semantics = module.constant_u32(
        type_u32,
        memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
    );
    module.control_barrier(workgroup_scope, workgroup_scope, workgroup_semantics);

    let half_local_size = module.constant_u32(type_u32, local_size_x / 2);
    module.store(variable_step, half_local_size);

    let label_reduce_header = module.alloc_id();
    let label_reduce_condition = module.alloc_id();
    let label_reduce_body = module.alloc_id();
    let label_reduce_continue = module.alloc_id();
    let label_reduce_merge = module.alloc_id();

    module.branch(label_reduce_header);
    module.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_reduce_header.0],
    ));
    module.loop_merge(label_reduce_merge, label_reduce_continue, 0);
    module.branch(label_reduce_condition);

    module.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_reduce_condition.0],
    ));
    let step = module.load(type_u32, variable_step);
    let step_is_positive = module.u_less_than(type_bool, constant_u32_0, step);
    module.branch_conditional(step_is_positive, label_reduce_body, label_reduce_merge);

    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_reduce_body.0]));
    let lane_is_active = module.u_less_than(type_bool, local_id, step);
    let label_reduce_active = module.alloc_id();
    let label_reduce_active_merge = module.alloc_id();
    module.selection_merge(label_reduce_active_merge, 0);
    module.branch_conditional(
        lane_is_active,
        label_reduce_active,
        label_reduce_active_merge,
    );

    module.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_reduce_active.0],
    ));
    let other_lane = module.iadd(type_u32, local_id, step);
    let own_value_pointer =
        module.access_chain(pointer_workgroup_f32, global_shared_values, &[local_id]);
    let own_index_pointer =
        module.access_chain(pointer_workgroup_u32, global_shared_indices, &[local_id]);
    let other_value_pointer =
        module.access_chain(pointer_workgroup_f32, global_shared_values, &[other_lane]);
    let other_index_pointer =
        module.access_chain(pointer_workgroup_u32, global_shared_indices, &[other_lane]);
    let own_value = module.load(type_f32, own_value_pointer);
    let own_index = module.load(type_u32, own_index_pointer);
    let other_value = module.load(type_f32, other_value_pointer);
    let other_index = module.load(type_u32, other_index_pointer);
    let other_beats_own = module.f_ord_greater_than(type_bool, other_value, own_value);
    let selected_value = module.select(type_f32, other_beats_own, other_value, own_value);
    let selected_index = module.select(type_u32, other_beats_own, other_index, own_index);
    module.store(own_value_pointer, selected_value);
    module.store(own_index_pointer, selected_index);
    module.branch(label_reduce_active_merge);

    module.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_reduce_active_merge.0],
    ));
    module.control_barrier(workgroup_scope, workgroup_scope, workgroup_semantics);
    module.branch(label_reduce_continue);

    module.functions.push(SpirvModule::encode_inst(
        op::LABEL,
        &[label_reduce_continue.0],
    ));
    let next_step = module.shift_right_logical(type_u32, step, constant_u32_1);
    module.store(variable_step, next_step);
    module.branch(label_reduce_header);

    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_reduce_merge.0]));
    let lane_is_zero = module.u_less_than(type_bool, local_id, constant_u32_1);
    let label_write = module.alloc_id();
    let label_write_merge = module.alloc_id();
    module.selection_merge(label_write_merge, 0);
    module.branch_conditional(lane_is_zero, label_write, label_write_merge);
    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_write.0]));
    let winning_value_pointer = module.access_chain(
        pointer_workgroup_f32,
        global_shared_values,
        &[constant_u32_0],
    );
    let winning_index_pointer = module.access_chain(
        pointer_workgroup_u32,
        global_shared_indices,
        &[constant_u32_0],
    );
    let winning_value = module.load(type_f32, winning_value_pointer);
    let winning_index = module.load(type_u32, winning_index_pointer);
    let partial_value_pointer = module.access_chain(
        pointer_storage_f32,
        global_partial_values,
        &[constant_u32_0, workgroup_id],
    );
    let partial_index_pointer = module.access_chain(
        pointer_storage_u32,
        global_partial_indices,
        &[constant_u32_0, workgroup_id],
    );
    module.store(partial_value_pointer, winning_value);
    module.store(partial_index_pointer, winning_index);
    module.branch(label_write_merge);
    module
        .functions
        .push(SpirvModule::encode_inst(op::LABEL, &[label_write_merge.0]));

    module.ret();
    module.function_end();
    module.encode()
}
