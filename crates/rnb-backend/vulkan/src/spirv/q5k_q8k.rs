use super::builder::{builtin, decoration, op, storage_class, SpirvModule};

pub fn emit_q5k_q8k_gemv(local_size_x: u32) -> Vec<u32> {
    let mut m = SpirvModule::new();

    m.capability(1); // Shader
    m.extension("SPV_KHR_storage_buffer_storage_class");
    m.memory_model(0, 1); // Logical, GLSL450

    // --- Types ---
    let t_void = m.type_void();
    let t_bool = m.type_bool();
    let t_u32 = m.type_int(32, 0);
    let t_i32 = m.type_int(32, 1);
    let t_f32 = m.type_float(32);

    let t_v3u32 = m.type_vector(t_u32, 3);

    let t_arr_u32 = m.type_runtime_array(t_u32);
    let t_arr_f32 = m.type_runtime_array(t_f32);

    let t_struct_weight = m.type_struct(&[t_arr_u32]);
    let t_struct_input = m.type_struct(&[t_arr_u32]); // Q8K packed
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

    // --- Constants ---
    let c_u32_0 = m.constant_u32(t_u32, 0);
    let c_u32_1 = m.constant_u32(t_u32, 1);
    let c_u32_2 = m.constant_u32(t_u32, 2);
    let c_u32_3 = m.constant_u32(t_u32, 3);
    let c_u32_4 = m.constant_u32(t_u32, 4);
    let c_u32_10 = m.constant_u32(t_u32, 10);
    let c_u32_13 = m.constant_u32(t_u32, 13);
    let c_u32_15 = m.constant_u32(t_u32, 15);
    let c_u32_16 = m.constant_u32(t_u32, 16);
    let c_u32_23 = m.constant_u32(t_u32, 23);
    let c_u32_24 = m.constant_u32(t_u32, 24);
    let c_u32_31 = m.constant_u32(t_u32, 31);
    let c_u32_44 = m.constant_u32(t_u32, 44);
    let c_u32_64 = m.constant_u32(t_u32, 64);
    let c_u32_65 = m.constant_u32(t_u32, 65);
    let c_u32_69 = m.constant_u32(t_u32, 69);
    let c_u32_256 = m.constant_u32(t_u32, 256);
    let c_u32_ff = m.constant_u32(t_u32, 0xFF);
    let c_u32_0f = m.constant_u32(t_u32, 0x0F);
    let c_u32_3f = m.constant_u32(t_u32, 0x3F);
    let c_u32_ffff = m.constant_u32(t_u32, 0xFFFF);
    let c_u32_1f = m.constant_u32(t_u32, 0x1F);
    let c_u32_3ff = m.constant_u32(t_u32, 0x3FF);
    let c_u32_112 = m.constant_u32(t_u32, 112);
    let c_u32_6 = m.constant_u32(t_u32, 6);

    let c_i32_0 = m.constant_u32(t_i32, 0);

    let c_f32_0 = m.constant_f32(t_f32, 0.0);

    // --- Decorations ---
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

    // --- Global variables ---
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

    // --- Entry point ---
    let func_id = m.alloc_id();
    m.entry_point(5, func_id, "main", &[gvar_glob_id]);
    m.execution_mode_local_size(func_id, local_size_x, 1, 1);

    // Pre-allocate labels
    let lbl_entry = m.alloc_id();
    let lbl_bounds_merge = m.alloc_id();
    let lbl_bounds_true = m.alloc_id();
    let lbl_outer_header = m.alloc_id();
    let lbl_outer_body = m.alloc_id();
    let lbl_outer_continue = m.alloc_id();
    let lbl_outer_merge = m.alloc_id();

    // --- Function ---
    m.function(t_void, func_id, 0, t_fn_void);
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_entry.0]));

    let var_sum = m.function_variable(t_ptr_fn_f32, storage_class::FUNCTION);
    let var_blk = m.function_variable(t_ptr_fn_u32, storage_class::FUNCTION);

    // Load GlobalInvocationID.x → row
    let glob_id_vec = m.load(t_v3u32, gvar_glob_id);
    let row = m.composite_extract(t_u32, glob_id_vec, 0);

    let pc_ptr_rows = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_0]);
    let pc_rows = m.load(t_u32, pc_ptr_rows);
    let pc_ptr_cols = m.access_chain(t_ptr_pc_u32, gvar_pc, &[c_u32_1]);
    let pc_cols = m.load(t_u32, pc_ptr_cols);

    let in_bounds = m.u_less_than(t_bool, row, pc_rows);
    m.selection_merge(lbl_bounds_merge, 0);
    m.branch_conditional(in_bounds, lbl_bounds_true, lbl_bounds_merge);

    // --- In-bounds block ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_true.0]));

    m.store(var_sum, c_f32_0);

    let num_blocks = m.udiv(t_u32, pc_cols, c_u32_256);

    m.store(var_blk, c_u32_0);
    m.branch(lbl_outer_header);

    // --- Outer loop header ---
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

    // --- Outer loop body ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_body.0]));

    // Weight: transposed SoA, plane_base = block * 36 * rows
    let blk_x_44 = m.imul(t_u32, blk_cur, c_u32_44);
    let plane_base = m.imul(t_u32, blk_x_44, pc_rows);

    // Activation: per-block stride 69 u32, qs words 0..64, d word 64, bsums words 65..69
    let act_base = m.imul(t_u32, blk_cur, c_u32_69);

    // ---- Load weight d, dmin from plane 0 ----
    let packed_addr = m.iadd(t_u32, plane_base, row);
    let packed_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, packed_addr]);
    let packed_word = m.load(t_u32, packed_ptr);

    let c_f32_2pow_neg24 = m.constant_f32(t_f32, 5.9604644775390625e-8);

    let d_f16_raw = m.bitwise_and(t_u32, packed_word, c_u32_ffff);
    let d_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, d_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, d_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, d_f16_raw, c_u32_15);
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

    let dmin_f16_raw = m.shift_right_logical(t_u32, packed_word, c_u32_16);
    let dmin_f32 = {
        let exp_raw = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_10);
        let exp = m.bitwise_and(t_u32, exp_raw, c_u32_1f);
        let mant = m.bitwise_and(t_u32, dmin_f16_raw, c_u32_3ff);
        let sign = m.shift_right_logical(t_u32, dmin_f16_raw, c_u32_15);
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

    // ---- Load 12 bytes scales/mins from planes 1..3 ----
    let s0_addr = m.iadd(t_u32, plane_base, pc_rows);
    let s0_addr = m.iadd(t_u32, s0_addr, row);
    let s0_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s0_addr]);
    let s0_word = m.load(t_u32, s0_ptr);

    let s1_offset = {
        let two_rows = m.imul(t_u32, c_u32_2, pc_rows);
        m.iadd(t_u32, plane_base, two_rows)
    };
    let s1_addr = m.iadd(t_u32, s1_offset, row);
    let s1_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s1_addr]);
    let s1_word = m.load(t_u32, s1_ptr);

    let s2_offset = {
        let three_rows = m.imul(t_u32, c_u32_3, pc_rows);
        m.iadd(t_u32, plane_base, three_rows)
    };
    let s2_addr = m.iadd(t_u32, s2_offset, row);
    let s2_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, s2_addr]);
    let s2_word = m.load(t_u32, s2_ptr);

    let mut sb = [c_u32_0; 12];
    for i in 0..4u32 {
        if i == 0 {
            sb[0] = m.bitwise_and(t_u32, s0_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s0_word, shift);
            sb[i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[4] = m.bitwise_and(t_u32, s1_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s1_word, shift);
            sb[4 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }
    for i in 0..4u32 {
        if i == 0 {
            sb[8] = m.bitwise_and(t_u32, s2_word, c_u32_ff);
        } else {
            let shift = m.constant_u32(t_u32, i * 8);
            let shifted = m.shift_right_logical(t_u32, s2_word, shift);
            sb[8 + i as usize] = m.bitwise_and(t_u32, shifted, c_u32_ff);
        }
    }

    // 8 scales + 8 mins (6-bit packed) — same extraction as f32 variant
    let mut scales_u = [c_u32_0; 8];
    let mut mins_u = [c_u32_0; 8];
    for j in 0..4usize {
        scales_u[j] = m.bitwise_and(t_u32, sb[j], c_u32_3f);
        mins_u[j] = m.bitwise_and(t_u32, sb[j + 4], c_u32_3f);
    }
    for j in 4..8usize {
        let lo = m.bitwise_and(t_u32, sb[j + 4], c_u32_0f);
        let hi_raw = m.shift_right_logical(t_u32, sb[j - 4], c_u32_6);
        let hi = m.shift_left_logical(t_u32, hi_raw, c_u32_4);
        scales_u[j] = m.bitwise_or(t_u32, lo, hi);

        let lo2 = m.shift_right_logical(t_u32, sb[j + 4], c_u32_4);
        let hi2_raw = m.shift_right_logical(t_u32, sb[j], c_u32_6);
        let hi2 = m.shift_left_logical(t_u32, hi2_raw, c_u32_4);
        mins_u[j] = m.bitwise_or(t_u32, lo2, hi2);
    }
    // Reinterpret as i32 for integer multiply
    let mut scales_i = [c_i32_0; 8];
    let mut mins_i = [c_i32_0; 8];
    for j in 0..8usize {
        scales_i[j] = m.bitcast(t_i32, scales_u[j]);
        mins_i[j] = m.bitcast(t_i32, mins_u[j]);
    }

    // ---- Load activation block scale d (word 64) ----
    let q8k_d_addr = m.iadd(t_u32, act_base, c_u32_64);
    let q8k_d_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, q8k_d_addr]);
    let q8k_d_word = m.load(t_u32, q8k_d_ptr);
    let q8k_d = m.bitcast(t_f32, q8k_d_word);

    // ---- Integer dot accumulators across 4 groups ----
    // Per CPU semantics: sumi & summ are per-block i32 accumulators,
    // promoted to f32 once at end via q8b.d * (d * sumi - dmin * summ).
    let mut sumi = c_i32_0;
    let mut summ = c_i32_0;

    // Precompute act_qs base (word offset 0) and act_bsums base (word offset 65).
    let act_bsums_base = m.iadd(t_u32, act_base, c_u32_65);

    // Q5_K high-bit planes 36..43: one u32 for each group word.
    let mut qh_words = [c_u32_0; 8];
    for w in 0..8u32 {
        let c_qh_plane = m.constant_u32(t_u32, 36 + w);
        let qh_plane_offset = m.imul(t_u32, c_qh_plane, pc_rows);
        let qh_plane_start = m.iadd(t_u32, plane_base, qh_plane_offset);
        let qh_addr = m.iadd(t_u32, qh_plane_start, row);
        let qh_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qh_addr]);
        qh_words[w as usize] = m.load(t_u32, qh_ptr);
    }

    for group in 0..4u32 {
        let is = (group * 2) as usize;
        // Weight nibble planes for this group: planes (4 + group*8) .. (4 + group*8 + 8)
        let qs_base_plane = 4 + group * 8;
        let c_qs_bp = m.constant_u32(t_u32, qs_base_plane);
        let qs_plane_offset = m.imul(t_u32, c_qs_bp, pc_rows);
        let qs_plane_start = m.iadd(t_u32, plane_base, qs_plane_offset);

        // Activation qs words for this group:
        //   sub-block 2g (lo nibbles): act_base + group*16 + (0..4)   — 4 u32 = 16 bytes? No: 4 u32 * 4 = 16 bytes only.
        // Wait: each sub-block has 32 i8 → 8 u32 words. 2 sub-blocks per group (lo+hi paired) → 16 u32.
        // group*16 stride is correct for activation qs.
        let c_g16 = m.constant_u32(t_u32, group * 16);
        let act_lo_base = m.iadd(t_u32, act_base, c_g16);
        let c_g16_plus_8 = m.constant_u32(t_u32, group * 16 + 8);
        let act_hi_base = m.iadd(t_u32, act_base, c_g16_plus_8);

        let mut isum0 = c_i32_0;
        let mut isum1 = c_i32_0;

        // 8 weight u32 words per sub-block-pair → covers 32 bytes nibble
        // each byte: lo nibble × q8b.qs[lo_block + l], hi nibble × q8b.qs[hi_block + l]
        for w in 0..8u32 {
            // weight word at plane (qs_base_plane + w)
            let w_offset = if w == 0 {
                qs_plane_start
            } else {
                let c_w = m.constant_u32(t_u32, w);
                let w_x_rows = m.imul(t_u32, c_w, pc_rows);
                m.iadd(t_u32, qs_plane_start, w_x_rows)
            };
            let qs_addr = m.iadd(t_u32, w_offset, row);
            let qs_ptr = m.access_chain(t_ptr_sb_u32, gvar_weight, &[c_u32_0, qs_addr]);
            let qs_word = m.load(t_u32, qs_ptr);
            let qh_word = qh_words[w as usize];

            // activation u32 words: lo @ act_lo_base + w, hi @ act_hi_base + w
            let act_lo_addr = if w == 0 {
                act_lo_base
            } else {
                let c_w = m.constant_u32(t_u32, w);
                m.iadd(t_u32, act_lo_base, c_w)
            };
            let act_lo_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, act_lo_addr]);
            let act_lo_word = m.load(t_u32, act_lo_ptr);

            let act_hi_addr = if w == 0 {
                act_hi_base
            } else {
                let c_w = m.constant_u32(t_u32, w);
                m.iadd(t_u32, act_hi_base, c_w)
            };
            let act_hi_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, act_hi_addr]);
            let act_hi_word = m.load(t_u32, act_hi_ptr);

            for byte_idx in 0..4u32 {
                // lo nibble (unsigned 0..15) at byte_idx
                let lo_nib = if byte_idx == 0 {
                    m.bitwise_and(t_u32, qs_word, c_u32_0f)
                } else {
                    let c_shift = m.constant_u32(t_u32, byte_idx * 8);
                    let shifted = m.shift_right_logical(t_u32, qs_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_0f)
                };
                // hi nibble (unsigned 0..15) at byte_idx + 4 bits
                let c_shift_hi = m.constant_u32(t_u32, byte_idx * 8 + 4);
                let hi_shifted = m.shift_right_logical(t_u32, qs_word, c_shift_hi);
                let hi_nib = m.bitwise_and(t_u32, hi_shifted, c_u32_0f);

                let qh_shift_lo = byte_idx * 8 + group * 2;
                let high_bit_lo = if qh_shift_lo == 0 {
                    m.bitwise_and(t_u32, qh_word, c_u32_1)
                } else {
                    let c_shift = m.constant_u32(t_u32, qh_shift_lo);
                    let shifted = m.shift_right_logical(t_u32, qh_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_1)
                };
                let qh_shift_hi = qh_shift_lo + 1;
                let high_bit_hi = {
                    let c_shift = m.constant_u32(t_u32, qh_shift_hi);
                    let shifted = m.shift_right_logical(t_u32, qh_word, c_shift);
                    m.bitwise_and(t_u32, shifted, c_u32_1)
                };
                let high_lo = m.shift_left_logical(t_u32, high_bit_lo, c_u32_4);
                let high_hi = m.shift_left_logical(t_u32, high_bit_hi, c_u32_4);
                let q5_lo = m.bitwise_or(t_u32, lo_nib, high_lo);
                let q5_hi = m.bitwise_or(t_u32, hi_nib, high_hi);
                let lo_nib_i = m.bitcast(t_i32, q5_lo);
                let hi_nib_i = m.bitcast(t_i32, q5_hi);

                // Q8K activation byte (signed i8) — sign-extend via 24-bit shift trick.
                // For lo sub-block:
                let act_lo_byte = if byte_idx == 0 {
                    act_lo_word
                } else {
                    let c_shift = m.constant_u32(t_u32, byte_idx * 8);
                    m.shift_right_logical(t_u32, act_lo_word, c_shift)
                };
                let act_lo_byte_top = m.shift_left_logical(t_u32, act_lo_byte, c_u32_24);
                let act_lo_byte_top_i = m.bitcast(t_i32, act_lo_byte_top);
                let x_lo = m.shift_right_arithmetic(t_i32, act_lo_byte_top_i, c_u32_24);

                let act_hi_byte = if byte_idx == 0 {
                    act_hi_word
                } else {
                    let c_shift = m.constant_u32(t_u32, byte_idx * 8);
                    m.shift_right_logical(t_u32, act_hi_word, c_shift)
                };
                let act_hi_byte_top = m.shift_left_logical(t_u32, act_hi_byte, c_u32_24);
                let act_hi_byte_top_i = m.bitcast(t_i32, act_hi_byte_top);
                let x_hi = m.shift_right_arithmetic(t_i32, act_hi_byte_top_i, c_u32_24);

                let prod_lo = m.imul(t_i32, lo_nib_i, x_lo);
                let prod_hi = m.imul(t_i32, hi_nib_i, x_hi);

                isum0 = m.iadd(t_i32, isum0, prod_lo);
                isum1 = m.iadd(t_i32, isum1, prod_hi);
            }
        }

        // sumi += sc[is]*isum0 + sc[is+1]*isum1
        let term_a = m.imul(t_i32, scales_i[is], isum0);
        let term_b = m.imul(t_i32, scales_i[is + 1], isum1);
        let term_ab = m.iadd(t_i32, term_a, term_b);
        sumi = m.iadd(t_i32, sumi, term_ab);

        // Load bsums word for this group (word 65 + group), low half = bsums[2g], high half = bsums[2g+1].
        let bsum_word_addr = if group == 0 {
            act_bsums_base
        } else {
            let c_g = m.constant_u32(t_u32, group);
            m.iadd(t_u32, act_bsums_base, c_g)
        };
        let bsum_ptr = m.access_chain(t_ptr_sb_u32, gvar_input, &[c_u32_0, bsum_word_addr]);
        let bsum_word = m.load(t_u32, bsum_ptr);
        // sign-extend low 16 bits → i32
        let bsum_lo_top = m.shift_left_logical(t_u32, bsum_word, c_u32_16);
        let bsum_lo_top_i = m.bitcast(t_i32, bsum_lo_top);
        let bsum_lo = m.shift_right_arithmetic(t_i32, bsum_lo_top_i, c_u32_16);
        // sign-extend high 16 bits → i32
        let bsum_word_i = m.bitcast(t_i32, bsum_word);
        let bsum_hi = m.shift_right_arithmetic(t_i32, bsum_word_i, c_u32_16);

        // summ += mn[is]*bsum_lo + mn[is+1]*bsum_hi
        let term_m_a = m.imul(t_i32, mins_i[is], bsum_lo);
        let term_m_b = m.imul(t_i32, mins_i[is + 1], bsum_hi);
        let term_m_ab = m.iadd(t_i32, term_m_a, term_m_b);
        summ = m.iadd(t_i32, summ, term_m_ab);
    }

    // Block-level f32 finalization: acc += q8b.d * (d * sumi - dmin * summ)
    let sumi_f = m.convert_s_to_f(t_f32, sumi);
    let summ_f = m.convert_s_to_f(t_f32, summ);
    let d_sumi = m.fmul(t_f32, d_f32, sumi_f);
    let dmin_summ = m.fmul(t_f32, dmin_f32, summ_f);
    let inner = m.fsub(t_f32, d_sumi, dmin_summ);
    let block_term = m.fmul(t_f32, q8k_d, inner);

    let prev_total = m.load(t_f32, var_sum);
    let new_total = m.fadd(t_f32, prev_total, block_term);
    m.store(var_sum, new_total);
    m.branch(lbl_outer_continue);

    // --- Outer loop continue ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_continue.0]));
    let blk_next = m.iadd(t_u32, blk_cur, c_u32_1);
    m.store(var_blk, blk_next);
    m.branch(lbl_outer_header);

    // --- Outer loop merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_outer_merge.0]));

    let final_sum = m.load(t_f32, var_sum);
    let out_ptr = m.access_chain(t_ptr_sb_f32, gvar_output, &[c_u32_0, row]);
    m.store(out_ptr, final_sum);
    m.branch(lbl_bounds_merge);

    // --- Bounds merge ---
    m.functions
        .push(SpirvModule::encode_inst(op::LABEL, &[lbl_bounds_merge.0]));
    m.ret();
    m.function_end();

    m.encode()
}
