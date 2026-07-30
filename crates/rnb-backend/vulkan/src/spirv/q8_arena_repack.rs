use super::builder::{builtin, decoration, op, storage_class, Id, SpirvModule};

/// Repack raw row-major GGML Q8_0 blocks into the arena's transposed SoA layout.
///
/// Bindings: raw staging bytes as u32 words, destination arena as u32 words.
/// Push constants: rows, cols, source byte offset, destination word offset.
pub fn emit_q8_arena_repack(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();
    m.capability(1);
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1);

    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_v3u32 = m.type_vector(t_u32, 3);
    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_struct_u32 = m.type_struct(&[t_arr_u32]);
    let t_struct_pc = m.type_struct(&[t_u32, t_u32, t_u32, t_u32]);
    let t_ptr_sb_struct = m.type_pointer(storage_class::STORAGE_BUFFER, t_struct_u32);
    let t_ptr_sb_u32 = m.type_pointer(storage_class::STORAGE_BUFFER, t_u32);
    let t_ptr_pc = m.type_pointer(storage_class::PUSH_CONSTANT, t_struct_pc);
    let t_ptr_pc_u32 = m.type_pointer(storage_class::PUSH_CONSTANT, t_u32);
    let t_ptr_input_v3 = m.type_pointer(storage_class::INPUT, t_v3u32);
    let t_fn_void = m.type_function(t_void, &[]);

    let c0 = m.constant_u32(t_u32, 0);
    let c1 = m.constant_u32(t_u32, 1);
    let c2 = m.constant_u32(t_u32, 2);
    let c3 = m.constant_u32(t_u32, 3);
    let c8 = m.constant_u32(t_u32, 8);
    let c9 = m.constant_u32(t_u32, 9);
    let c16 = m.constant_u32(t_u32, 16);
    let c24 = m.constant_u32(t_u32, 24);
    let c32 = m.constant_u32(t_u32, 32);
    let c34 = m.constant_u32(t_u32, 34);
    let cff = m.constant_u32(t_u32, 0xFF);

    m.decorate(t_struct_u32, decoration::BLOCK, &[]);
    m.member_decorate(t_struct_u32, 0, decoration::OFFSET, &[0]);
    m.decorate(t_arr_u32, decoration::ARRAY_STRIDE, &[4]);
    m.decorate(t_struct_pc, decoration::BLOCK, &[]);
    for field in 0..4 {
        m.member_decorate(t_struct_pc, field, decoration::OFFSET, &[field * 4]);
    }

    let source = m.variable(t_ptr_sb_struct, storage_class::STORAGE_BUFFER);
    let destination = m.variable(t_ptr_sb_struct, storage_class::STORAGE_BUFFER);
    let pc = m.variable(t_ptr_pc, storage_class::PUSH_CONSTANT);
    let global_id_var = m.variable(t_ptr_input_v3, storage_class::INPUT);
    for (binding, variable) in [source, destination].into_iter().enumerate() {
        m.decorate(variable, decoration::DESCRIPTOR_SET, &[0]);
        m.decorate(variable, decoration::BINDING, &[binding as u32]);
    }
    m.decorate(
        global_id_var,
        decoration::BUILTIN,
        &[builtin::GLOBAL_INVOCATION_ID],
    );

    let function = m.alloc_id();
    m.entry_point(5, function, "main", &[global_id_var]);
    m.execution_mode_local_size(function, local_size_x, 1, 1);
    let entry = m.alloc_id();
    let active_label = m.alloc_id();
    let merge_label = m.alloc_id();

    m.function(t_void, function, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[entry.0]));

    let global_id = m.load(t_v3u32, global_id_var);
    let row = m.composite_extract(t_u32, global_id, 0);
    let block = m.composite_extract(t_u32, global_id, 1);
    let rows_ptr = m.access_chain(t_ptr_pc_u32, pc, &[c0]);
    let rows = m.load(t_u32, rows_ptr);
    let cols_ptr = m.access_chain(t_ptr_pc_u32, pc, &[c1]);
    let cols = m.load(t_u32, cols_ptr);
    let blocks_per_row = m.udiv(t_u32, cols, c32);
    let row_active = m.u_less_than(t_bool, row, rows);
    let block_active = m.u_less_than(t_bool, block, blocks_per_row);
    let active = m.logical_and(t_bool, row_active, block_active);
    m.selection_merge(merge_label, 0);
    m.branch_conditional(active, active_label, merge_label);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[active_label.0]));

    let source_base_ptr = m.access_chain(t_ptr_pc_u32, pc, &[c2]);
    let source_base = m.load(t_u32, source_base_ptr);
    let destination_base_ptr = m.access_chain(t_ptr_pc_u32, pc, &[c3]);
    let destination_base = m.load(t_u32, destination_base_ptr);
    let row_block_base = m.imul(t_u32, row, blocks_per_row);
    let raw_block = m.iadd(t_u32, row_block_base, block);
    let raw_block_offset = m.imul(t_u32, raw_block, c34);
    let raw_block_base = m.iadd(t_u32, source_base, raw_block_offset);

    let read_byte = |m: &mut SpirvModule, byte_offset: Id| -> Id {
        let word_index = m.shift_right_logical(t_u32, byte_offset, c2);
        let byte_in_word = m.bitwise_and(t_u32, byte_offset, c3);
        let source_ptr = m.access_chain(t_ptr_sb_u32, source, &[c0, word_index]);
        let word = m.load(t_u32, source_ptr);
        let shift = m.imul(t_u32, byte_in_word, c8);
        let shifted = m.shift_right_logical(t_u32, word, shift);
        m.bitwise_and(t_u32, shifted, cff)
    };

    let scale_lo = read_byte(&mut m, raw_block_base);
    let scale_hi_offset = m.iadd(t_u32, raw_block_base, c1);
    let scale_hi = read_byte(&mut m, scale_hi_offset);
    let scale_hi_shifted = m.shift_left_logical(t_u32, scale_hi, c8);
    let scale_word = m.bitwise_or(t_u32, scale_lo, scale_hi_shifted);
    let block_plane = m.imul(t_u32, block, c9);
    let scale_plane_base = m.imul(t_u32, block_plane, rows);
    let scale_destination = m.iadd(t_u32, destination_base, scale_plane_base);
    let scale_destination = m.iadd(t_u32, scale_destination, row);
    let scale_ptr = m.access_chain(t_ptr_sb_u32, destination, &[c0, scale_destination]);
    m.store(scale_ptr, scale_word);

    for word in 0..8_u32 {
        let word_byte_offset = m.constant_u32(t_u32, 2 + word * 4);
        let word_base = m.iadd(t_u32, raw_block_base, word_byte_offset);
        let b0 = read_byte(&mut m, word_base);
        let b1_offset = m.iadd(t_u32, word_base, c1);
        let b1 = read_byte(&mut m, b1_offset);
        let b2_offset = m.iadd(t_u32, word_base, c2);
        let b2 = read_byte(&mut m, b2_offset);
        let b3_offset = m.iadd(t_u32, word_base, c3);
        let b3 = read_byte(&mut m, b3_offset);
        let b1 = m.shift_left_logical(t_u32, b1, c8);
        let b2 = m.shift_left_logical(t_u32, b2, c16);
        let b3 = m.shift_left_logical(t_u32, b3, c24);
        let packed01 = m.bitwise_or(t_u32, b0, b1);
        let packed23 = m.bitwise_or(t_u32, b2, b3);
        let packed = m.bitwise_or(t_u32, packed01, packed23);

        let plane = m.constant_u32(t_u32, word + 1);
        let plane = m.iadd(t_u32, block_plane, plane);
        let plane_base = m.imul(t_u32, plane, rows);
        let destination_index = m.iadd(t_u32, destination_base, plane_base);
        let destination_index = m.iadd(t_u32, destination_index, row);
        let destination_ptr = m.access_chain(t_ptr_sb_u32, destination, &[c0, destination_index]);
        m.store(destination_ptr, packed);
    }

    m.branch(merge_label);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[merge_label.0]));
    m.ret();
    m.function_end();
    m.encode()
}
