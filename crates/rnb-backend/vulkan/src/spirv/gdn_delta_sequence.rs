use super::builder::{
    builtin, decoration, memory_semantics, scope, storage_class, Id, SpirvModule,
};

const HEAD_DIM: u32 = 128;
const LANES_PER_ROW: u32 = 8;
const ROWS_PER_WORKGROUP: u32 = 4;
const WORKGROUP_SIZE: u32 = LANES_PER_ROW * ROWS_PER_WORKGROUP;
const VALUES_PER_LANE: u32 = HEAD_DIM / LANES_PER_ROW;

struct ReduceIds {
    u32_ty: Id,
    f32_ty: Id,
    zero: Id,
    eight: Id,
    scope_workgroup: Id,
    semantics: Id,
    shared_element_ptr: Id,
    shared: Id,
}

fn reduce_row(
    module: &mut SpirvModule,
    ids: &ReduceIds,
    row_lane: Id,
    row_base: Id,
    partial: Id,
) -> Id {
    let own_index = module.iadd(ids.u32_ty, row_base, row_lane);
    let own_ptr = module.access_chain(ids.shared_element_ptr, ids.shared, &[own_index]);
    module.store(own_ptr, partial);
    module.control_barrier(ids.scope_workgroup, ids.scope_workgroup, ids.semantics);

    for shift in [4, 2, 1] {
        let shift = module.constant_u32(ids.u32_ty, shift);
        let partner_lane_sum = module.iadd(ids.u32_ty, row_lane, shift);
        let partner_lane = module.umod(ids.u32_ty, partner_lane_sum, ids.eight);
        let partner_index = module.iadd(ids.u32_ty, row_base, partner_lane);
        let partner_ptr = module.access_chain(ids.shared_element_ptr, ids.shared, &[partner_index]);
        let own = module.load(ids.f32_ty, own_ptr);
        let partner = module.load(ids.f32_ty, partner_ptr);
        module.control_barrier(ids.scope_workgroup, ids.scope_workgroup, ids.semantics);
        let sum = module.fadd(ids.f32_ty, own, partner);
        module.store(own_ptr, sum);
        module.control_barrier(ids.scope_workgroup, ids.scope_workgroup, ids.semantics);
    }

    let result = module.load(ids.f32_ty, own_ptr);
    module.control_barrier(ids.scope_workgroup, ids.scope_workgroup, ids.semantics);
    let _ = ids.zero;
    result
}

/// Gated DeltaNet sequence kernel specialized for 128x128 state heads.
///
/// One 32-lane workgroup advances four state rows. Eight lanes cooperate on
/// each row, keep sixteen state values per lane resident across the complete
/// token sequence, and use shared-memory clustered reductions for the two dot
/// products. Dispatch is `[num_v_heads, 32, 1]`.
pub fn emit_gdn_delta_sequence_d128() -> Vec<u32> {
    let mut module = SpirvModule::new();
    module.capability(1); // Shader
    module.extension("SPV_KHR_storage_buffer_storage_class");
    module.memory_model(0, 1); // Logical, GLSL450

    let void_ty = module.type_void();
    let bool_ty = module.type_bool();
    let u32_ty = module.type_int(32, 0);
    let f32_ty = module.type_float(32);
    let v3u32_ty = module.type_vector(u32_ty, 3);
    let f32_array_ty = module.type_runtime_array(f32_ty);
    let buffer_ty = module.type_struct(&[f32_array_ty]);
    let pc_ty = module.type_struct(&[u32_ty; 11]);
    let shared_len = module.constant_u32(u32_ty, WORKGROUP_SIZE);
    let shared_array_ty = module.type_array(f32_ty, shared_len);

    let buffer_ptr_ty = module.type_pointer(storage_class::STORAGE_BUFFER, buffer_ty);
    let pc_ptr_ty = module.type_pointer(storage_class::PUSH_CONSTANT, pc_ty);
    let input_v3_ptr_ty = module.type_pointer(storage_class::INPUT, v3u32_ty);
    let buffer_f32_ptr_ty = module.type_pointer(storage_class::STORAGE_BUFFER, f32_ty);
    let pc_u32_ptr_ty = module.type_pointer(storage_class::PUSH_CONSTANT, u32_ty);
    let function_u32_ptr_ty = module.type_pointer(storage_class::FUNCTION, u32_ty);
    let function_f32_ptr_ty = module.type_pointer(storage_class::FUNCTION, f32_ty);
    let shared_array_ptr_ty = module.type_pointer(storage_class::WORKGROUP, shared_array_ty);
    let shared_f32_ptr_ty = module.type_pointer(storage_class::WORKGROUP, f32_ty);
    let function_ty = module.type_function(void_ty, &[]);

    let zero_u32 = module.constant_u32(u32_ty, 0);
    let one_u32 = module.constant_u32(u32_ty, 1);
    let two_u32 = module.constant_u32(u32_ty, 2);
    let three_u32 = module.constant_u32(u32_ty, 3);
    let four_u32 = module.constant_u32(u32_ty, 4);
    let five_u32 = module.constant_u32(u32_ty, 5);
    let seven_u32 = module.constant_u32(u32_ty, 7);
    let eight_u32 = module.constant_u32(u32_ty, 8);
    let nine_u32 = module.constant_u32(u32_ty, 9);
    let ten_u32 = module.constant_u32(u32_ty, 10);
    let head_dim_u32 = module.constant_u32(u32_ty, HEAD_DIM);
    let state_head_size_u32 = module.constant_u32(u32_ty, HEAD_DIM * HEAD_DIM);
    let zero_f32 = module.constant_f32(f32_ty, 0.0);
    let scope_workgroup = module.constant_u32(u32_ty, scope::WORKGROUP);
    let semantics = module.constant_u32(
        u32_ty,
        memory_semantics::WORKGROUP_MEMORY | memory_semantics::ACQUIRE_RELEASE,
    );

    module.decorate(buffer_ty, decoration::BLOCK, &[]);
    module.member_decorate(buffer_ty, 0, decoration::OFFSET, &[0]);
    module.decorate(f32_array_ty, decoration::ARRAY_STRIDE, &[4]);
    module.decorate(pc_ty, decoration::BLOCK, &[]);
    for member in 0..11 {
        module.member_decorate(pc_ty, member, decoration::OFFSET, &[member * 4]);
    }

    let buffers: Vec<_> = (0..7)
        .map(|_| module.variable(buffer_ptr_ty, storage_class::STORAGE_BUFFER))
        .collect();
    let pc = module.variable(pc_ptr_ty, storage_class::PUSH_CONSTANT);
    let workgroup_id = module.variable(input_v3_ptr_ty, storage_class::INPUT);
    let local_id = module.variable(input_v3_ptr_ty, storage_class::INPUT);
    let shared = module.variable(shared_array_ptr_ty, storage_class::WORKGROUP);

    for (binding, &buffer) in buffers.iter().enumerate() {
        module.decorate(buffer, decoration::DESCRIPTOR_SET, &[0]);
        module.decorate(buffer, decoration::BINDING, &[binding as u32]);
    }
    module.decorate(workgroup_id, decoration::BUILTIN, &[builtin::WORKGROUP_ID]);
    module.decorate(
        local_id,
        decoration::BUILTIN,
        &[builtin::LOCAL_INVOCATION_ID],
    );

    let function = module.alloc_id();
    module.entry_point(5, function, "main", &[workgroup_id, local_id]);
    module.execution_mode_local_size(function, LANES_PER_ROW, ROWS_PER_WORKGROUP, 1);

    let entry = module.alloc_id();
    let token_header = module.alloc_id();
    let token_cond = module.alloc_id();
    let token_body = module.alloc_id();
    let token_continue = module.alloc_id();
    let token_merge = module.alloc_id();
    let store_output = module.alloc_id();
    let store_output_merge = module.alloc_id();

    module.function(void_ty, function, 0, function_ty);
    module.functions.push(SpirvModule::encode_inst(
        super::builder::op::LABEL,
        &[entry.0],
    ));
    let token_var = module.function_variable(function_u32_ptr_ty, storage_class::FUNCTION);
    let state_vars: Vec<_> = (0..VALUES_PER_LANE)
        .map(|_| module.function_variable(function_f32_ptr_ty, storage_class::FUNCTION))
        .collect();

    let load_pc = |module: &mut SpirvModule, index: Id| {
        let ptr = module.access_chain(pc_u32_ptr_ty, pc, &[index]);
        module.load(u32_ty, ptr)
    };
    let conv_channels = load_pc(&mut module, zero_u32);
    let d_inner = load_pc(&mut module, one_u32);
    let num_k_heads = load_pc(&mut module, two_u32);
    let num_v_heads = load_pc(&mut module, three_u32);
    let _head_k_dim = load_pc(&mut module, four_u32);
    let _head_v_dim = load_pc(&mut module, five_u32);
    let seq_len = load_pc(&mut module, seven_u32);
    let conv_stride = load_pc(&mut module, eight_u32);
    let _head_stride = load_pc(&mut module, nine_u32);
    let output_stride = load_pc(&mut module, ten_u32);

    let workgroup = module.load(v3u32_ty, workgroup_id);
    let head = module.composite_extract(u32_ty, workgroup, 0);
    let row_group = module.composite_extract(u32_ty, workgroup, 1);
    let local = module.load(v3u32_ty, local_id);
    let local_x = module.composite_extract(u32_ty, local, 0);
    let row_lane = module.umod(u32_ty, local_x, eight_u32);
    let row_in_group = module.composite_extract(u32_ty, local, 1);
    let row_group_base = module.imul(u32_ty, row_group, four_u32);
    let row = module.iadd(u32_ty, row_group_base, row_in_group);
    let shared_row_base = module.imul(u32_ty, row_in_group, eight_u32);
    let key_head = module.umod(u32_ty, head, num_k_heads);
    let q_dim = module.imul(u32_ty, num_k_heads, head_dim_u32);
    let q_head_base = module.imul(u32_ty, key_head, head_dim_u32);
    let k_head_base = module.iadd(u32_ty, q_dim, q_head_base);
    let v_start = module.isub(u32_ty, conv_channels, d_inner);
    let v_head_base = module.imul(u32_ty, head, head_dim_u32);
    let state_head_base = module.imul(u32_ty, head, state_head_size_u32);
    let state_row_base = module.imul(u32_ty, row, head_dim_u32);
    let state_base = module.iadd(u32_ty, state_head_base, state_row_base);

    for (r, &state_var) in state_vars.iter().enumerate() {
        let row_offset = module.constant_u32(u32_ty, r as u32 * LANES_PER_ROW);
        let lane_offset = module.iadd(u32_ty, row_offset, row_lane);
        let state_index = module.iadd(u32_ty, state_base, lane_offset);
        let state_ptr =
            module.access_chain(buffer_f32_ptr_ty, buffers[5], &[zero_u32, state_index]);
        let state = module.load(f32_ty, state_ptr);
        module.store(state_var, state);
    }

    module.store(token_var, zero_u32);
    module.branch(token_header);
    module.functions.push(SpirvModule::encode_inst(
        super::builder::op::LABEL,
        &[token_header.0],
    ));
    module.loop_merge(token_merge, token_continue, 0);
    module.branch(token_cond);
    module.functions.push(SpirvModule::encode_inst(
        super::builder::op::LABEL,
        &[token_cond.0],
    ));
    let token = module.load(u32_ty, token_var);
    let has_token = module.u_less_than(bool_ty, token, seq_len);
    module.branch_conditional(has_token, token_body, token_merge);
    module.functions.push(SpirvModule::encode_inst(
        super::builder::op::LABEL,
        &[token_body.0],
    ));

    let conv_token_base = module.imul(u32_ty, token, conv_stride);
    let two_k_heads = module.imul(u32_ty, two_u32, num_k_heads);
    let two_v_heads = module.imul(u32_ty, two_u32, num_v_heads);
    let params_stride = module.iadd(u32_ty, two_k_heads, two_v_heads);
    let params_token_base = module.imul(u32_ty, token, params_stride);
    let q_inv_index = module.iadd(u32_ty, params_token_base, key_head);
    let k_inv_base = module.iadd(u32_ty, params_token_base, num_k_heads);
    let k_inv_index = module.iadd(u32_ty, k_inv_base, key_head);
    let beta_base = module.iadd(u32_ty, params_token_base, two_k_heads);
    let beta_index = module.iadd(u32_ty, beta_base, head);
    let decay_base = module.iadd(u32_ty, beta_base, num_v_heads);
    let decay_index = module.iadd(u32_ty, decay_base, head);
    let q_inv_ptr = module.access_chain(buffer_f32_ptr_ty, buffers[1], &[zero_u32, q_inv_index]);
    let k_inv_ptr = module.access_chain(buffer_f32_ptr_ty, buffers[1], &[zero_u32, k_inv_index]);
    let beta_ptr = module.access_chain(buffer_f32_ptr_ty, buffers[1], &[zero_u32, beta_index]);
    let decay_ptr = module.access_chain(buffer_f32_ptr_ty, buffers[1], &[zero_u32, decay_index]);
    let q_inv = module.load(f32_ty, q_inv_ptr);
    let k_inv = module.load(f32_ty, k_inv_ptr);
    let beta = module.load(f32_ty, beta_ptr);
    let decay = module.load(f32_ty, decay_ptr);

    let mut decayed_states = Vec::with_capacity(VALUES_PER_LANE as usize);
    let mut normalized_keys = Vec::with_capacity(VALUES_PER_LANE as usize);
    let mut scaled_queries = Vec::with_capacity(VALUES_PER_LANE as usize);
    let mut state_key_partial = zero_f32;
    for (r, &state_var) in state_vars.iter().enumerate() {
        let row_offset = module.constant_u32(u32_ty, r as u32 * LANES_PER_ROW);
        let element = module.iadd(u32_ty, row_offset, row_lane);
        let q_element = module.iadd(u32_ty, q_head_base, element);
        let k_element = module.iadd(u32_ty, k_head_base, element);
        let q_index = module.iadd(u32_ty, conv_token_base, q_element);
        let k_index = module.iadd(u32_ty, conv_token_base, k_element);
        let q_ptr = module.access_chain(buffer_f32_ptr_ty, buffers[0], &[zero_u32, q_index]);
        let k_ptr = module.access_chain(buffer_f32_ptr_ty, buffers[0], &[zero_u32, k_index]);
        let q = module.load(f32_ty, q_ptr);
        let k = module.load(f32_ty, k_ptr);
        let q_scaled = module.fmul(f32_ty, q, q_inv);
        let k_normalized = module.fmul(f32_ty, k, k_inv);
        let state = module.load(f32_ty, state_var);
        let state_decayed = module.fmul(f32_ty, state, decay);
        let state_key = module.fmul(f32_ty, state_decayed, k_normalized);
        state_key_partial = module.fadd(f32_ty, state_key_partial, state_key);
        decayed_states.push(state_decayed);
        normalized_keys.push(k_normalized);
        scaled_queries.push(q_scaled);
    }

    let reduce_ids = ReduceIds {
        u32_ty,
        f32_ty,
        zero: zero_f32,
        eight: eight_u32,
        scope_workgroup,
        semantics,
        shared_element_ptr: shared_f32_ptr_ty,
        shared,
    };
    let state_key = reduce_row(
        &mut module,
        &reduce_ids,
        row_lane,
        shared_row_base,
        state_key_partial,
    );
    let v_element = module.iadd(u32_ty, v_head_base, row);
    let v_element = module.iadd(u32_ty, v_start, v_element);
    let v_index = module.iadd(u32_ty, conv_token_base, v_element);
    let v_ptr = module.access_chain(buffer_f32_ptr_ty, buffers[0], &[zero_u32, v_index]);
    let v = module.load(f32_ty, v_ptr);
    let residual = module.fsub(f32_ty, v, state_key);
    let delta = module.fmul(f32_ty, residual, beta);

    let mut output_partial = zero_f32;
    for r in 0..VALUES_PER_LANE as usize {
        let key_delta = module.fmul(f32_ty, normalized_keys[r], delta);
        let state_new = module.fadd(f32_ty, decayed_states[r], key_delta);
        module.store(state_vars[r], state_new);
        let output_term = module.fmul(f32_ty, state_new, scaled_queries[r]);
        output_partial = module.fadd(f32_ty, output_partial, output_term);
    }
    let output = reduce_row(
        &mut module,
        &reduce_ids,
        row_lane,
        shared_row_base,
        output_partial,
    );

    let is_row_writer = module.u_less_than(bool_ty, row_lane, one_u32);
    module.selection_merge(store_output_merge, 0);
    module.branch_conditional(is_row_writer, store_output, store_output_merge);
    module.functions.push(SpirvModule::encode_inst(
        super::builder::op::LABEL,
        &[store_output.0],
    ));
    let output_token_base = module.imul(u32_ty, token, output_stride);
    let output_head_base = module.imul(u32_ty, head, head_dim_u32);
    let output_row = module.iadd(u32_ty, output_head_base, row);
    let output_index = module.iadd(u32_ty, output_token_base, output_row);
    let output_ptr = module.access_chain(buffer_f32_ptr_ty, buffers[6], &[zero_u32, output_index]);
    module.store(output_ptr, output);
    module.branch(store_output_merge);
    module.functions.push(SpirvModule::encode_inst(
        super::builder::op::LABEL,
        &[store_output_merge.0],
    ));
    module.branch(token_continue);
    module.functions.push(SpirvModule::encode_inst(
        super::builder::op::LABEL,
        &[token_continue.0],
    ));
    let token_next = module.iadd(u32_ty, token, one_u32);
    module.store(token_var, token_next);
    module.branch(token_header);

    module.functions.push(SpirvModule::encode_inst(
        super::builder::op::LABEL,
        &[token_merge.0],
    ));
    for (r, &state_var) in state_vars.iter().enumerate() {
        let row_offset = module.constant_u32(u32_ty, r as u32 * LANES_PER_ROW);
        let lane_offset = module.iadd(u32_ty, row_offset, row_lane);
        let state_index = module.iadd(u32_ty, state_base, lane_offset);
        let state_ptr =
            module.access_chain(buffer_f32_ptr_ty, buffers[5], &[zero_u32, state_index]);
        let state = module.load(f32_ty, state_var);
        module.store(state_ptr, state);
    }
    module.ret();
    module.function_end();
    module.encode()
}
